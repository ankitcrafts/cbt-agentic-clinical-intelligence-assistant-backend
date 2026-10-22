import json
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.auth import router as auth_router
from app.api.routes import router as api_router
from app.api.scheduling import router as scheduling_router
from app.core.database import init_db
from app.core.tracing import setup_langsmith_tracing, tracing_status
from app.services.pipeline_service import ClinicalPipeline

TRACING_INFO = setup_langsmith_tracing()
OUTPUT_DIR = PROJECT_ROOT / "output"
AUDIO_PATH = str(
    PROJECT_ROOT
    / "app"
    / "uploads"
    / "WhatsApp Audio 2026-08-13 at 18.50.16.mpeg"
)
pipeline = ClinicalPipeline()


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize PostgreSQL / SQLite tables and seed demo accounts on startup
    try:
        init_db()
    except Exception as e:
        print(f"[Lifespan Error] Database initialization error: {e}")

    # Ensure RAG guideline index is available
    try:
        pipeline.graph.retrieval.ensure_index()
    except Exception as e:
        print(f"[Lifespan Notice] RAG guideline index check: {e}")

    # Ensure persistence schema is initialized
    try:
        pipeline.ensure_persistence()
    except Exception as e:
        print(f"[Lifespan Notice] Persistence check: {e}")

    yield


app = FastAPI(
    title="ClinIQ Clinical Intelligence Pipeline & Auth API",
    description="Agentic Clinical Assistant Backend with LangGraph, RAG Guidelines, PostgreSQL Authentication, Gmail OTP Delivery, Speech-to-Text, and Clinical Decision Support.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(auth_router)
app.include_router(api_router)
app.include_router(scheduling_router)


@app.get("/health")
def health_check():
    db_status = "healthy"
    dialect = "unknown"
    try:
        from sqlalchemy import text
        from app.core.database import engine
        dialect = engine.dialect.name
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    is_healthy = "unhealthy" not in db_status
    return {
        "status": "healthy" if is_healthy else "degraded",
        "service": "cliniq-backend",
        "version": "1.0.0",
        "database": {
            "status": db_status,
            "dialect": dialect,
        },
        "tracing": tracing_status(),
        "persistence": pipeline.persistence_status(),
    }


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "ClinIQ Clinical Intelligence & Authentication API",
        "database": "PostgreSQL (cliniq_db / cliniIQ)",
        "observability": tracing_status(),
        "persistence": pipeline.persistence_status(),
        "authentication": {
            "send_otp": "POST /api/auth/send-otp",
            "verify_otp": "POST /api/auth/verify-otp",
            "current_user": "GET /api/auth/me",
            "list_users": "GET /api/auth/users",
        },
        "roles_supported": ["ADMIN", "DOCTOR", "PATIENT"],
        "pipeline_features": [
            "Deepgram STT & Diarization",
            "LangGraph Agentic State & Memory",
            "NICE Clinical Guideline RAG Retrieval (Hybrid BM25 + FAISS)",
            "Differential Diagnosis Ranking",
            "Confidence-based Follow-up Question Loop (< 0.70 threshold)",
            "Live AI Suggestions & SOAP Reframing",
            "PDF Clinical Analysis & SOAP Reports (Local + S3)",
            "PostgreSQL Visit & Conversation Persistence",
            "Optional RAGAS Validation Metrics (Faithfulness, Answer Relevancy, Context Precision)",
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
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
