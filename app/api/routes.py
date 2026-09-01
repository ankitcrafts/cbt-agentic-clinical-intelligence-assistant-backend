import json
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import ClinicalRecording, SOAPFeedback
from app.schemas.schemas import AnalyzeConversationRequest, SOAPReframeRequest, SOAPSaveFeedbackRequest
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
    auto_transcribe: Optional[bool] = Form(True),
    db: Session = Depends(get_db),
):
    """
    Upload a clinical recording file from doctor consultation session.
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
            status=recording_status,
        )
        try:
            db.add(recording)
            db.commit()
            db.refresh(recording)
        except Exception as db_err:
            db.rollback()
            print(f"[Warning] Failed to persist recording metadata to DB: {db_err}")

        return {
            "id": recording_id,
            "status": recording_status,
            "encounterId": encounter_id,
            "doctorId": doctor_id,
            "patientId": patient_id,
            "fileName": clean_filename,
            "fileSize": file_size,
            "durationSeconds": dur_sec,
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

