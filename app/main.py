import json
import sys
import traceback
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.tracing import setup_langsmith_tracing, tracing_status

TRACING_INFO = setup_langsmith_tracing()

from app.services.pipeline_service import ClinicalPipeline

app = FastAPI(title="Clinical Intelligence Agent")
pipeline = ClinicalPipeline()

AUDIO_PATH = "/Users/durgeshwarpratapsingh/Desktop/ISB_project/app/uploads/WhatsApp Audio 2026-08-13 at 18.50.16.mpeg"
OUTPUT_DIR = Path("output")


@app.on_event("startup")
def ensure_guideline_index():
    try:
        pipeline.graph.retrieval.ensure_index()
    except Exception:
        traceback.print_exc()


@app.get("/")
def home():
    return {
        "message": "Clinical agent running (LangGraph + RAG + MemorySaver + LangSmith)",
        "observability": tracing_status(),
        "steps": [
            "1. conversation transcription",
            "2. doctor-patient conversion",
            "3. conversation summary",
            "4. medical entity extraction",
            "5. guideline retrieval (RAG)",
            "6. evidence-based differential diagnosis",
            "7. save analysis report PDF + SOAP note PDF to output/",
        ],
    }


@app.get("/observability")
def observability():
    status = tracing_status()
    status["startup"] = TRACING_INFO
    return status


@app.get("/transcribe")
def transcribe():
    try:
        return pipeline.deepgram.transcribe(AUDIO_PATH)
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/analyze")
def analyze(
    visit_id: Optional[str] = Query(
        default=None,
        description="Optional visit id. Leave empty to auto-generate one (e.g. visit-a1b2c3d4).",
    )
):
    try:
        return pipeline.run_from_audio(AUDIO_PATH, visit_id=visit_id)
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/analyze-conversation")
def analyze_saved_conversation(
    visit_id: Optional[str] = Query(
        default=None,
        description="Optional visit id. Leave empty to auto-generate one (e.g. visit-a1b2c3d4).",
    )
):
    try:
        files = sorted(OUTPUT_DIR.glob("*_conversation.json"))
        if not files:
            return {"status": "error", "message": "No conversation JSON found in output/"}
        latest = files[-1]
        with open(latest, encoding="utf-8") as f:
            conversation = json.load(f)
        result = pipeline.run_from_conversation(
            conversation,
            visit_id=visit_id,
            source_file=str(latest),
        )
        if tracing_status()["tracing_enabled"]:
            result["langsmith"] = {
                "project": tracing_status()["project"],
                "visit_id": result.get("visit_id"),
                "hint": "Open LangSmith dashboard and filter by visit_id metadata",
            }
        return result
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/reports")
def list_reports():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = sorted(OUTPUT_DIR.glob("*_analysis_report.pdf"), reverse=True)
    soap = sorted(OUTPUT_DIR.glob("*_soap_note.pdf"), reverse=True)
    return {
        "analysis_count": len(analysis),
        "soap_count": len(soap),
        "analysis_reports": [
            {
                "filename": path.name,
                "path": str(path),
                "download_url": f"/reports/{path.name}",
            }
            for path in analysis
        ],
        "soap_notes": [
            {
                "filename": path.name,
                "path": str(path),
                "download_url": f"/reports/{path.name}",
            }
            for path in soap
        ],
    }


@app.get("/reports/{filename}")
def download_report(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename,
    )


@app.get("/rag/ingest")
def ingest_guidelines():
    try:
        return pipeline.ingest_guidelines()
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/visit/{visit_id}")
def get_visit_state(visit_id: str):
    try:
        state = pipeline.get_visit_state(visit_id)
        if not state:
            return {"status": "error", "message": f"No memory found for visit_id={visit_id}"}
        return state
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
