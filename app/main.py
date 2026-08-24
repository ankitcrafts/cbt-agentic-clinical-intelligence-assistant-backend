import sys
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import init_db
from app.api.routes import router as api_router
from app.api.auth import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize PostgreSQL tables and seed demo accounts on startup
    init_db()
    yield


app = FastAPI(
    title="ClinIQ Clinical Intelligence Pipeline & Auth API",
    description="Agentic Clinical Assistant Backend with PostgreSQL Authentication, Gmail OTP Delivery, Speech-to-Text, and Clinical Decision Support.",
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
        "authentication": {
            "send_otp": "POST /api/auth/send-otp",
            "verify_otp": "POST /api/auth/verify-otp",
            "current_user": "GET /api/auth/me",
            "list_users": "GET /api/auth/users",
        },
        "roles_supported": ["ADMIN", "DOCTOR", "PATIENT"],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
