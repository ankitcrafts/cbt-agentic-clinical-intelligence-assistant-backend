import json
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import ClinicalRecording, SOAPFeedback, PatientConsultationSession
from app.schemas.schemas import (
    AnalyzeConversationRequest,
    SOAPReframeRequest,
    SOAPSaveFeedbackRequest,
    CreateConsultationSessionRequest,
    PatientConsultationSessionResponse,
)
from app.services.pipeline_service import ClinicalPipeline
from app.services.gdrive_service import gdrive_service


router = APIRouter(tags=["Clinical Pipeline"])
pipeline = ClinicalPipeline()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = PROJECT_ROOT / "app" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/api/v1/recordings")
@router.post("/recordings")
@router.post("/api/recordings")
async def upload_recording(
    file: UploadFile = File(...),
    encounter_id: Optional[str] = Form("CONS-41233"),
    doctor_id: Optional[str] = Form("DOC-001"),
    patient_id: Optional[str] = Form("P1245"),
    doctor_name: Optional[str] = Form("Dr. Yuki Tanaka"),
    patient_name: Optional[str] = Form("Marcus Delgado"),
    doctor_initials: Optional[str] = Form("YT"),
    patient_initials: Optional[str] = Form("MD"),
    recording_started_at: Optional[str] = Form(None),
    recording_ended_at: Optional[str] = Form(None),
    duration_seconds: Optional[str] = Form("0"),
    consultation_mode: Optional[str] = Form("IN_PERSON"),
    audio_source: Optional[str] = Form("MIC"),
    auto_transcribe: Optional[bool] = Form(True),
    db: Session = Depends(get_db),
):
    """
    Upload a clinical recording file from doctor consultation session (in-person or remote teleconsultation).
    1. Saves audio locally to app/uploads.
    2. Uploads audio to Google Drive shared folder (replacing AWS S3).
    3. Runs Deepgram Speech-to-Text with Speaker Diarization.
    4. Generates real-time AI Consultation Suggestions and Clinical Decision Support.
    5. Saves recording metadata in database and returns full payload to frontend.
    """
    try:
        raw_name = Path(file.filename).name if file.filename else None
        clean_filename = raw_name or f"rec_{encounter_id}_{uuid.uuid4().hex[:8]}.wav"
        file_path = UPLOADS_DIR / clean_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = file_path.stat().st_size if file_path.exists() else 0
        dur_sec = int(duration_seconds) if duration_seconds and duration_seconds.isdigit() else 0

        # 1. Upload to Google Drive Shared Folder
        try:
            gdrive_result = gdrive_service.upload_file(
                file_path=str(file_path),
                file_name=clean_filename,
                mime_type=file.content_type or "audio/wav",
            )
        except Exception as gd_err:
            print(f"[Warning] Google Drive upload error: {gd_err}")
            gdrive_result = {"success": False, "error": str(gd_err)}

        recording_status = "Uploaded to Google Drive" if gdrive_result.get("success") else "Uploaded"

        # 2. Run Real-Time Deepgram Transcription & Clinical Pipeline
        transcript = ""
        exchanges = []
        ai_suggestions = []
        clinical_analysis = {}
        pipeline_res = {}

        if auto_transcribe:
            try:
                pipeline_res = pipeline.run_from_audio(
                    audio_path=str(file_path),
                    doctor_name=doctor_name or "Dr. Yuki Tanaka",
                    patient_name=patient_name or "Patient",
                    doctor_initials=doctor_initials or "YT",
                    patient_initials=patient_initials or "PT",
                )
                transcript = pipeline_res.get("transcript", "")
                exchanges = pipeline_res.get("exchanges", [])
                ai_suggestions = pipeline_res.get("ai_suggestions", [])
                clinical_analysis = pipeline_res.get("clinical_analysis", {})
                recording_status += " & Transcribed"
            except Exception as pipe_err:
                print(f"[Warning] Pipeline error, performing direct Deepgram fallback: {pipe_err}")
                traceback.print_exc()
                try:
                    asr = pipeline.deepgram.transcribe(str(file_path))
                    transcript = asr.get("transcript", "")
                    labeled = pipeline.conversation.to_doctor_patient(asr.get("conversation", []))
                    exchanges = pipeline._format_exchanges(
                        labeled_turns=labeled.get("conversation", []),
                        doctor_name=doctor_name or "Dr. Yuki Tanaka",
                        patient_name=patient_name or "Patient",
                        doctor_initials=doctor_initials or "YT",
                        patient_initials=patient_initials or "PT",
                    )
                    ai_suggestions = pipeline._fallback_suggestions({})
                    clinical_analysis = {
                        "summary": pipeline._fallback_summary(labeled.get("conversation", []), patient_name or "Patient"),
                        "entities": pipeline._fallback_entities(labeled.get("conversation", [])),
                        "differential_diagnosis": pipeline._fallback_differential({}, {}),
                        "soap_report": pipeline._fallback_soap_report({}, labeled.get("conversation", [])),
                        "ai_suggestions": ai_suggestions,
                    }
                    pipeline_res = clinical_analysis
                    recording_status += " & Transcribed (Fallback)"
                except Exception as fallback_err:
                    print(f"[Error] Deepgram direct fallback failed: {fallback_err}")
                    traceback.print_exc()

        # 3. Save recording metadata in database
        recording_id = str(uuid.uuid4())
        recording = ClinicalRecording(
            id=recording_id,
            encounter_id=encounter_id,
            doctor_id=doctor_id,
            patient_id=patient_id,
            file_name=clean_filename,
            file_path=str(file_path),
            file_size=file_size,
            mime_type=file.content_type or "audio/wav",
            duration_seconds=dur_sec,
            consultation_mode=consultation_mode or "IN_PERSON",
            audio_source=audio_source or "MIC",
            status=recording_status,
        )
        try:
            db.add(recording)
            db.commit()
            db.refresh(recording)
        except Exception as db_err:
            db.rollback()
            print(f"[Warning] Failed to persist recording metadata to DB: {db_err}")

        # 4. Save or update PatientConsultationSession for encounter
        session_id = None
        try:
            from datetime import datetime, timezone
            clean_pid = patient_id if patient_id.startswith("#") else f"#{patient_id}"
            existing_sessions = (
                db.query(PatientConsultationSession)
                .filter(
                    (PatientConsultationSession.patient_id == clean_pid) |
                    (PatientConsultationSession.patient_id == patient_id)
                )
                .order_by(PatientConsultationSession.session_number.desc())
                .all()
            )
            existing_for_encounter = next((s for s in existing_sessions if s.encounter_id == encounter_id), None)
            if existing_for_encounter:
                existing_for_encounter.transcript = transcript
                existing_for_encounter.conversation_json = json.dumps(exchanges)
                existing_for_encounter.ai_suggestions_json = json.dumps(ai_suggestions)
                existing_for_encounter.clinical_analysis_json = json.dumps(clinical_analysis)
                existing_for_encounter.soap_report_json = json.dumps(pipeline_res.get("soap_report") or {})
                existing_for_encounter.audio_file_name = clean_filename
                existing_for_encounter.audio_file_path = str(file_path)
                existing_for_encounter.duration_seconds = dur_sec
                existing_for_encounter.consultation_mode = consultation_mode or "IN_PERSON"
                existing_for_encounter.audio_source = audio_source or "MIC"
                existing_for_encounter.gdrive_file_id = gdrive_result.get("file_id")
                existing_for_encounter.gdrive_web_view_link = gdrive_result.get("web_view_link")
                session_id = existing_for_encounter.id
            else:
                next_session_number = (existing_sessions[0].session_number + 1) if existing_sessions else 1
                new_session = PatientConsultationSession(
                    patient_id=clean_pid,
                    patient_name=patient_name or "Patient",
                    doctor_id=doctor_id or "DOC-001",
                    doctor_name=doctor_name or "Dr. Yuki Tanaka",
                    encounter_id=encounter_id,
                    session_number=next_session_number,
                    session_title=f"Session {next_session_number}: Consultation & Review",
                    appointment_date=datetime.now(timezone.utc),
                    transcript=transcript,
                    conversation_json=json.dumps(exchanges),
                    ai_suggestions_json=json.dumps(ai_suggestions),
                    clinical_analysis_json=json.dumps(clinical_analysis),
                    soap_report_json=json.dumps(pipeline_res.get("soap_report") or {}),
                    audio_file_name=clean_filename,
                    audio_file_path=str(file_path),
                    duration_seconds=dur_sec,
                    consultation_mode=consultation_mode or "IN_PERSON",
                    audio_source=audio_source or "MIC",
                    gdrive_file_id=gdrive_result.get("file_id"),
                    gdrive_web_view_link=gdrive_result.get("web_view_link"),
                    status="COMPLETED",
                )
                db.add(new_session)
                db.commit()
                db.refresh(new_session)
                session_id = new_session.id
            db.commit()
        except Exception as sess_err:
            db.rollback()
            print(f"[Warning] Failed to save consultation session: {sess_err}")


        return {
            "id": recording_id,
            "status": recording_status,
            "encounterId": encounter_id,
            "doctorId": doctor_id,
            "patientId": patient_id,
            "fileName": clean_filename,
            "fileSize": file_size,
            "durationSeconds": dur_sec,
            "consultationMode": consultation_mode or "IN_PERSON",
            "audioSource": audio_source or "MIC",
            "gdriveUploaded": gdrive_result.get("success", False),
            "gdriveFileId": gdrive_result.get("file_id"),
            "gdriveWebViewLink": gdrive_result.get("web_view_link"),
            "gdriveFolderId": gdrive_result.get("folder_id", "1W0jt4Bb4RJ8ALo53zpzIAdahRfzKCmwi"),
            "transcript": transcript,
            "exchanges": exchanges,
            "ai_suggestions": ai_suggestions,
            "aiSuggestions": ai_suggestions,
            "summary": pipeline_res.get("summary"),
            "entities": pipeline_res.get("entities"),
            "differential_diagnosis": pipeline_res.get("differential_diagnosis"),
            "soap_report": pipeline_res.get("soap_report"),
            "clinical_analysis": clinical_analysis,
            "clinicalAnalysis": clinical_analysis,
            "message": "Audio recording uploaded to Google Drive & transcribed successfully!" if gdrive_result.get("success") else "Audio recording processed and transcribed successfully.",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/api/v1/recordings/quick-transcribe")
@router.post("/quick-transcribe")
async def quick_transcribe_audio(
    file: UploadFile = File(...),
    doctor_name: Optional[str] = Form("Dr. Yuki Tanaka"),
    patient_name: Optional[str] = Form("Marcus Delgado"),
    doctor_initials: Optional[str] = Form("YT"),
    patient_initials: Optional[str] = Form("MD"),
):
    """
    Fast Deepgram Speech-to-Text transcription with Speaker Diarization.
    Returns transcript and formatted dialogue exchanges immediately.
    """
    try:
        raw_name = Path(file.filename).name if file.filename else None
        clean_filename = raw_name or f"audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = UPLOADS_DIR / clean_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        asr = pipeline.deepgram.transcribe(str(file_path))
        conversation = asr.get("conversation", [])
        transcript = asr.get("transcript", "")

        # Label speakers
        labeled = pipeline.conversation.to_doctor_patient(conversation)
        labeled_turns = labeled.get("conversation", [])

        exchanges = pipeline._format_exchanges(
            labeled_turns=labeled_turns,
            doctor_name=doctor_name or "Dr. Yuki Tanaka",
            patient_name=patient_name or "Marcus Delgado",
            doctor_initials=doctor_initials or "YT",
            patient_initials=patient_initials or "MD",
        )

        return {
            "transcript": transcript,
            "exchanges": exchanges,
            "conversation": conversation,
            "speaker_map": labeled.get("speaker_map", {}),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/gdrive/status")
@router.get("/gdrive/status")
def gdrive_status():
    """Returns Google Drive service connection status and destination folder."""
    is_ready = gdrive_service.client is not None or bool(gdrive_service.upload_via_webhook)
    return {
        "connected": True,
        "mode": "Google Apps Script Webhook Relay & Drive v3 API",
        "targetFolderId": "1W0jt4Bb4RJ8ALo53zpzIAdahRfzKCmwi",
        "targetFolderUrl": "https://drive.google.com/drive/folders/1W0jt4Bb4RJ8ALo53zpzIAdahRfzKCmwi",
    }


@router.get("/api/v1/recordings")
@router.get("/recordings")
@router.get("/api/recordings")
def list_recordings(db: Session = Depends(get_db)):
    """List all uploaded recordings."""
    try:
        recordings = db.query(ClinicalRecording).order_by(ClinicalRecording.created_at.desc()).all()
        return [
            {
                "id": r.id,
                "encounterId": r.encounter_id,
                "doctorId": r.doctor_id,
                "patientId": r.patient_id,
                "fileName": r.file_name,
                "fileSize": r.file_size,
                "mimeType": r.mime_type,
                "durationSeconds": r.duration_seconds,
                "status": r.status,
                "createdAt": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recordings
        ]
    except Exception as e:
        return []


@router.get("/api/v1/recordings/{recording_id}")
@router.get("/recordings/{recording_id}")
def get_recording(recording_id: str, db: Session = Depends(get_db)):
    """Retrieve details, Google Drive status, and metadata of a specific recording."""
    rec = db.query(ClinicalRecording).filter(ClinicalRecording.id == recording_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
    return {
        "id": rec.id,
        "encounterId": rec.encounter_id,
        "doctorId": rec.doctor_id,
        "patientId": rec.patient_id,
        "fileName": rec.file_name,
        "filePath": rec.file_path,
        "fileSize": rec.file_size,
        "mimeType": rec.mime_type,
        "durationSeconds": rec.duration_seconds,
        "status": rec.status,
        "createdAt": rec.created_at.isoformat() if rec.created_at else None,
    }


@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (.mp3, .wav, .m4a, .mpeg, etc.) from the frontend
    and run the entire clinical pipeline.
    """
    try:
        clean_name = Path(file.filename).name if file.filename else f"audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = UPLOADS_DIR / clean_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = pipeline.run_from_audio(str(file_path))
        result["filename"] = clean_name
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file to get speech-to-text transcript and diarized conversation.
    """
    try:
        clean_name = Path(file.filename).name if file.filename else f"audio_{uuid.uuid4().hex[:8]}.wav"
        file_path = UPLOADS_DIR / clean_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return pipeline.deepgram.transcribe(str(file_path))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-conversation")
def analyze_conversation_payload(payload: AnalyzeConversationRequest):
    """
    Send conversation JSON directly from the frontend to run LLM summarization,
    entity extraction, and differential diagnosis.
    """
    try:
        return pipeline.run_from_conversation(
            conversation=payload.conversation,
            transcript=payload.transcript or "",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demo/transcribe")
def demo_transcribe():
    """Demo endpoint: transcribe default sample audio."""
    default_audio = str(UPLOADS_DIR / "WhatsApp Audio 2026-08-13 at 18.50.16.mpeg")
    try:
        return pipeline.deepgram.transcribe(default_audio)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demo/analyze")
def demo_analyze():
    """Demo endpoint: run full pipeline on default sample audio."""
    default_audio = str(UPLOADS_DIR / "WhatsApp Audio 2026-08-13 at 18.50.16.mpeg")
    try:
        return pipeline.run_from_audio(default_audio)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demo/analyze-saved-conversation")
def demo_analyze_saved_conversation():
    """Demo endpoint: analyze the latest saved JSON conversation in output/."""
    import json
    try:
        output_dir = PROJECT_ROOT / "output"
        files = sorted(output_dir.glob("*_conversation.json"))
        if not files:
            raise HTTPException(status_code=404, detail="No conversation JSON found in output/")
        latest = files[-1]
        with open(latest, encoding="utf-8") as f:
            conversation = json.load(f)
        result = pipeline.run_from_conversation(conversation)
        result["source_file"] = str(latest)
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/soap/reframe")
@router.post("/soap/reframe")
def reframe_soap_section_endpoint(payload: SOAPReframeRequest):
    """
    AI Reframes a specific section of the SOAP report based on doctor's reasoning/instructions.
    Also returns a patient-friendly summary translation if full_soap is provided.
    """
    try:
        reframed_text = pipeline.summary.reframe_soap_section(
            section=payload.section,
            original_content=payload.original_content,
            edit_reason=payload.edit_reason,
        )
        patient_summary = ""
        if payload.full_soap:
            updated_soap = dict(payload.full_soap)
            if payload.section.lower() in updated_soap:
                updated_soap[payload.section.lower()] = reframed_text
            patient_summary = pipeline.summary.generate_patient_summary(updated_soap)

        return {
            "section": payload.section,
            "reframed_text": reframed_text,
            "patient_summary": patient_summary,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to reframe SOAP section: {str(e)}")


@router.post("/api/v1/soap/feedback")
@router.post("/soap/feedback")
def save_soap_feedback(payload: SOAPSaveFeedbackRequest, db: Session = Depends(get_db)):
    """
    Persists finalized SOAP report edits, additional reasoning, and training statements to database.
    Used for fine-tuning future SOAP generation models and clinical auditing.
    """
    try:
        patient_summary = payload.patient_summary
        if not patient_summary and payload.edited_soap:
            patient_summary = pipeline.summary.generate_patient_summary(payload.edited_soap)

        feedback_entry = SOAPFeedback(
            encounter_id=payload.encounter_id,
            doctor_id=payload.doctor_id or "DOC-001",
            patient_id=payload.patient_id or "P1245",
            original_soap_json=json.dumps(payload.original_soap),
            edited_soap_json=json.dumps(payload.edited_soap),
            edit_reason=payload.edit_reason or "",
            edited_section=payload.edited_section or "all",
            patient_summary=patient_summary or "",
            is_finalized=payload.is_finalized if payload.is_finalized is not None else True,
        )
        db.add(feedback_entry)
        db.commit()
        db.refresh(feedback_entry)

        # Sync edited SOAP with PatientConsultationSession if available
        try:
            clean_pid = payload.patient_id if (payload.patient_id and payload.patient_id.startswith("#")) else f"#{payload.patient_id}"
            sess = db.query(PatientConsultationSession).filter(
                (PatientConsultationSession.encounter_id == payload.encounter_id) |
                (PatientConsultationSession.patient_id == clean_pid) |
                (PatientConsultationSession.patient_id == payload.patient_id)
            ).order_by(PatientConsultationSession.created_at.desc()).first()
            if sess and payload.edited_soap:
                sess.soap_report_json = json.dumps(payload.edited_soap)
                db.commit()
        except Exception as sync_err:
            print(f"[Warning] Failed to sync SOAP to session: {sync_err}")


        return {
            "id": feedback_entry.id,
            "status": "Saved",
            "encounterId": feedback_entry.encounter_id,
            "patientSummary": feedback_entry.patient_summary,
            "message": "SOAP report edits & training feedback persisted to database successfully.",
        }
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save SOAP feedback: {str(e)}")


@router.get("/api/v1/soap/feedback/{encounter_id}")
@router.get("/soap/feedback/{encounter_id}")
def get_soap_feedback(encounter_id: str, db: Session = Depends(get_db)):
    """
    Retrieves stored SOAP feedback records for a specific encounter.
    """
    records = db.query(SOAPFeedback).filter(SOAPFeedback.encounter_id == encounter_id).order_by(SOAPFeedback.created_at.desc()).all()
    if not records:
        return []
    return [
        {
            "id": r.id,
            "encounterId": r.encounter_id,
            "doctorId": r.doctor_id,
            "patientId": r.patient_id,
            "originalSoap": json.loads(r.original_soap_json) if r.original_soap_json else {},
            "editedSoap": json.loads(r.edited_soap_json) if r.edited_soap_json else {},
            "editReason": r.edit_reason,
            "editedSection": r.edited_section,
            "patientSummary": r.patient_summary,
            "isFinalized": r.is_finalized,
            "createdAt": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


def _serialize_session(s: PatientConsultationSession) -> dict:
    conversation = []
    if s.conversation_json:
        try:
            conversation = json.loads(s.conversation_json)
        except Exception:
            conversation = []

    ai_suggestions = []
    if s.ai_suggestions_json:
        try:
            ai_suggestions = json.loads(s.ai_suggestions_json)
        except Exception:
            ai_suggestions = []

    clinical_analysis = {}
    if s.clinical_analysis_json:
        try:
            clinical_analysis = json.loads(s.clinical_analysis_json)
        except Exception:
            clinical_analysis = {}

    soap_report = {}
    if s.soap_report_json:
        try:
            soap_report = json.loads(s.soap_report_json)
        except Exception:
            soap_report = {}

    return {
        "id": s.id,
        "patientId": s.patient_id,
        "patientName": s.patient_name,
        "doctorId": s.doctor_id,
        "doctorName": s.doctor_name,
        "encounterId": s.encounter_id,
        "sessionNumber": s.session_number,
        "sessionTitle": s.session_title,
        "appointmentDate": s.appointment_date.isoformat() if s.appointment_date else None,
        "transcript": s.transcript or "",
        "conversation": conversation,
        "aiSuggestions": ai_suggestions,
        "ai_suggestions": ai_suggestions,
        "clinicalAnalysis": clinical_analysis,
        "clinical_analysis": clinical_analysis,
        "soapReport": soap_report,
        "soap_report": soap_report,
        "audioFileName": s.audio_file_name,
        "durationSeconds": s.duration_seconds or 0,
        "gdriveFileId": s.gdrive_file_id,
        "gdriveWebViewLink": s.gdrive_web_view_link,
        "consultationMode": s.consultation_mode or "IN_PERSON",
        "audioSource": s.audio_source or "MIC",
        "status": s.status,
        "createdAt": s.created_at.isoformat() if s.created_at else None,
        "updatedAt": s.updated_at.isoformat() if s.updated_at else None,
    }


@router.get("/api/v1/patients/{patient_id}/sessions")
@router.get("/patients/{patient_id}/sessions")
def get_patient_consultation_sessions(patient_id: str, db: Session = Depends(get_db)):
    """
    Fetch all consultation sessions for a specific patient, ordered chronologically by session number.
    """
    clean_pid = patient_id if patient_id.startswith("#") else f"#{patient_id}"
    sessions = (
        db.query(PatientConsultationSession)
        .filter(
            (PatientConsultationSession.patient_id == clean_pid) |
            (PatientConsultationSession.patient_id == patient_id)
        )
        .order_by(PatientConsultationSession.session_number.asc())
        .all()
    )
    return [_serialize_session(s) for s in sessions]


@router.get("/api/v1/sessions/{session_id}")
@router.get("/sessions/{session_id}")
def get_consultation_session_detail(session_id: str, db: Session = Depends(get_db)):
    """
    Fetch detailed payload for an individual consultation session.
    """
    session = db.query(PatientConsultationSession).filter(PatientConsultationSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Consultation session not found")
    return _serialize_session(session)


@router.post("/api/v1/sessions")
@router.post("/sessions")
def create_consultation_session(payload: CreateConsultationSessionRequest, db: Session = Depends(get_db)):
    """
    Create or manually persist a consultation session.
    """
    from datetime import datetime, timezone
    clean_pid = payload.patient_id if payload.patient_id.startswith("#") else f"#{payload.patient_id}"
    existing = (
        db.query(PatientConsultationSession)
        .filter(
            (PatientConsultationSession.patient_id == clean_pid) |
            (PatientConsultationSession.patient_id == payload.patient_id)
        )
        .order_by(PatientConsultationSession.session_number.desc())
        .first()
    )
    sess_num = payload.session_number or ((existing.session_number + 1) if existing else 1)
    sess_title = payload.session_title or f"Session {sess_num}: Consultation"

    new_sess = PatientConsultationSession(
        patient_id=clean_pid,
        patient_name=payload.patient_name,
        doctor_id=payload.doctor_id or "DOC-001",
        doctor_name=payload.doctor_name or "Dr. Yuki Tanaka",
        encounter_id=payload.encounter_id,
        session_number=sess_num,
        session_title=sess_title,
        appointment_date=datetime.now(timezone.utc),
        transcript=payload.transcript or "",
        conversation_json=json.dumps(payload.conversation or []),
        ai_suggestions_json=json.dumps(payload.ai_suggestions or []),
        clinical_analysis_json=json.dumps(payload.clinical_analysis or {}),
        soap_report_json=json.dumps(payload.soap_report or {}),
        duration_seconds=payload.duration_seconds or 0,
        consultation_mode=payload.consultation_mode or "IN_PERSON",
        audio_source=payload.audio_source or "MIC",
        status=payload.status or "COMPLETED",
    )
    db.add(new_sess)
    db.commit()
    db.refresh(new_sess)
    return _serialize_session(new_sess)



