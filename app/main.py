import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.tracing import setup_langsmith_tracing, tracing_status

TRACING_INFO = setup_langsmith_tracing()

from app.services.pipeline_service import ClinicalPipeline

app = FastAPI(title="Clinical Intelligence Agent")
pipeline = ClinicalPipeline()

AUDIO_PATH = str(
    PROJECT_ROOT
    / "app"
    / "uploads"
    / "WhatsApp Audio 2026-08-13 at 18.50.16.mpeg"
)
OUTPUT_DIR = PROJECT_ROOT / "output"


class FollowUpRequest(BaseModel):
    answers: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="List of {question_id, answer}. Preferred for testing without a second audio.",
    )
    conversation: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional raw or role-labeled follow-up turns instead of answers.",
    )
    evaluate_ragas: bool = Field(
        default=False,
        description="If true, run RAGAS validation and include scores in the JSON response.",
    )


@app.on_event("startup")
def ensure_guideline_index():
    try:
        pipeline.graph.retrieval.ensure_index()
    except Exception:
        traceback.print_exc()
    try:
        pipeline.ensure_persistence()
    except Exception:
        traceback.print_exc()


@app.get("/")
def home():
    return {
        "message": "Clinical agent running (LangGraph + RAG + MemorySaver + LangSmith)",
        "observability": tracing_status(),
        "persistence": pipeline.persistence_status(),
        "steps": [
            "1. conversation transcription",
            "2. doctor-patient conversion",
            "3. conversation summary",
            "4. medical entity extraction",
            "5. guideline retrieval (RAG hybrid BM25+FAISS)",
            "6. evidence-based differential diagnosis",
            "7. follow-up questions if confidence < 0.70 (one round)",
            "8. save analysis report PDF + SOAP note PDF to output/",
            "9. persist visit in Postgres + PDFs in S3 (when configured)",
            "10. optional RAGAS validation scores (ragas=true)",
        ],
    }


@app.get("/persistence")
def persistence():
    try:
        return pipeline.ensure_persistence()
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/visit/{visit_id}/conversation")
def visit_conversation(visit_id: str):
    """Return ordered conversation turns from Postgres (old + follow-up)."""
    try:
        state = pipeline.get_visit_state(visit_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"No visit found for visit_id={visit_id}")
        turns = pipeline.visit_store.list_conversation(visit_id)
        if not turns:
            turns = state.get("doctor_patient_conversation") or state.get("conversation") or []
        return {
            "visit_id": visit_id,
            "status": state.get("status"),
            "follow_up_iteration": state.get("follow_up_iteration"),
            "turns": turns,
            "postgres_enabled": pipeline.visit_store.enabled,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}



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
    ),
    force_follow_up: bool = Query(
        default=False,
        description="For testing: always generate follow-up questions even if confidence >= 0.70.",
    ),
    ragas: bool = Query(
        default=False,
        description="If true, run RAGAS metrics (faithfulness, answer_relevancy, context_precision).",
    ),
):
    try:
        return pipeline.run_from_audio(
            AUDIO_PATH,
            visit_id=visit_id,
            force_follow_up=force_follow_up,
            evaluate_ragas=ragas,
        )
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/analyze-conversation")
def analyze_saved_conversation(
    visit_id: Optional[str] = Query(
        default=None,
        description="Optional visit id. Leave empty to auto-generate one (e.g. visit-a1b2c3d4).",
    ),
    force_follow_up: bool = Query(
        default=False,
        description="For testing: always generate follow-up questions even if confidence >= 0.70.",
    ),
    ragas: bool = Query(
        default=False,
        description="If true, run RAGAS metrics (faithfulness, answer_relevancy, context_precision).",
    ),
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
            force_follow_up=force_follow_up,
            evaluate_ragas=ragas,
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


@app.post("/visit/{visit_id}/follow-up")
def apply_follow_up(visit_id: str, body: FollowUpRequest):
    """
    Apply one follow-up round: merge answers into prior summary, re-run entities/RAG/diagnosis/SOAP.
    No second follow-up is generated even if confidence stays below 0.70.
    """
    try:
        return pipeline.apply_follow_up(
            visit_id=visit_id,
            answers=body.answers,
            conversation=body.conversation,
            evaluate_ragas=body.evaluate_ragas,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/visit/{visit_id}/ragas")
def evaluate_ragas(visit_id: str):
    """Run RAGAS validation on an existing visit and return JSON scores."""
    try:
        return pipeline.evaluate_visit_ragas(visit_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
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
