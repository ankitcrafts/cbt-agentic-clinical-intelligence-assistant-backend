import json
import sys
import traceback
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.database import init_db
from app.core.tracing import setup_langsmith_tracing, tracing_status
from app.services.pipeline_service import ClinicalPipeline
from app.api.routes import router as api_router
from app.api.auth import router as auth_router

TRACING_INFO = setup_langsmith_tracing()
OUTPUT_DIR = PROJECT_ROOT / "output"
pipeline = ClinicalPipeline()


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


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "ClinIQ Clinical Intelligence & Authentication API",
        "database": "PostgreSQL (cliniq_db)",
        "observability": tracing_status(),
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
            "NICE Clinical Guideline RAG Retrieval",
            "Differential Diagnosis Ranking",
            "Live AI Suggestions & SOAP Reframing",
            "PDF Clinical Analysis & SOAP Reports",
        ],
    }


@app.get("/observability")
def observability():
    status = tracing_status()
    status["startup"] = TRACING_INFO
    return status


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
