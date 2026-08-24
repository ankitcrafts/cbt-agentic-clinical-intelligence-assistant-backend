import shutil
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import ClinicalRecording
from app.schemas.schemas import AnalyzeConversationRequest
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
    recording_started_at: Optional[str] = Form(None),
    recording_ended_at: Optional[str] = Form(None),
    duration_seconds: Optional[str] = Form("0"),
    db: Session = Depends(get_db),
):
    """
    Upload a clinical recording file from doctor consultation session.
    Saves audio to app/uploads, syncs to Google Drive shared folder, and records metadata in database.
    """
    try:
        clean_filename = file.filename or f"rec_{encounter_id}_{uuid.uuid4().hex[:8]}.wav"
        file_path = UPLOADS_DIR / clean_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = file_path.stat().st_size if file_path.exists() else 0
        dur_sec = int(duration_seconds) if duration_seconds and duration_seconds.isdigit() else 0

        # Upload to Google Drive Shared Folder
        gdrive_result = gdrive_service.upload_file(
            file_path=str(file_path),
            file_name=clean_filename,
            mime_type=file.content_type or "audio/wav",
        )

        recording_status = "Uploaded to Google Drive" if gdrive_result.get("success") else "Uploaded"

        # Save to database
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
            "fileName": clean_filename,
            "fileSize": file_size,
            "gdriveUploaded": gdrive_result.get("success", False),
            "gdriveFileId": gdrive_result.get("file_id"),
            "gdriveWebViewLink": gdrive_result.get("web_view_link"),
            "gdriveFolderId": gdrive_result.get("folder_id", "1W0jt4Bb4RJ8ALo53zpzIAdahRfzKCmwi"),
            "message": "Audio recording uploaded to Google Drive successfully!" if gdrive_result.get("success") else "Audio recording uploaded to ClinIQ server.",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


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


@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (.mp3, .wav, .m4a, .mpeg, etc.) from the frontend
    and run the entire clinical pipeline.
    """
    try:
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = pipeline.run_from_audio(str(file_path))
        result["filename"] = file.filename
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
        file_path = UPLOADS_DIR / file.filename
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
            transcript=payload.transcript,
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
