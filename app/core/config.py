import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUIDELINES_DIR = PROJECT_ROOT / "Rag" / "data" / "guidelines"
VECTORSTORE_DIR = PROJECT_ROOT / "Rag" / "vectorstore"

# Default one-click Postgres (docker-compose / local)
DEFAULT_DB_USER = "clinIQ_user"
DEFAULT_DB_PASSWORD = "password"
DEFAULT_DB_NAME = "cliniIQ"
DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_PORT = "5432"


def _clean_env(name: str) -> str:
    value = os.getenv(name) or ""
    return value.strip().strip('"').strip("'")


def _build_database_url() -> str:
    explicit = _clean_env("DATABASE_URL")
    if explicit:
        return explicit
    user = _clean_env("DB_USER") or DEFAULT_DB_USER
    password = _clean_env("DB_PASSWORD") or DEFAULT_DB_PASSWORD
    host = _clean_env("DB_HOST") or DEFAULT_DB_HOST
    port = _clean_env("DB_PORT") or DEFAULT_DB_PORT
    name = _clean_env("DB_NAME") or DEFAULT_DB_NAME
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def load_settings():
    load_dotenv(ENV_PATH, override=True)
    return {
        "DEEPGRAM_API_KEY": _clean_env("DEEPGRAM_API_KEY"),
        "GOOGLE_API_KEY": _clean_env("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": _clean_env("OPENROUTER_API_KEY"),
        "GROQ_API_KEY": _clean_env("GROQ_API_KEY"),
        "LLM_PROVIDER": _clean_env("LLM_PROVIDER") or "openrouter",
        "OPENROUTER_MODEL": _clean_env("OPENROUTER_MODEL") or "deepseek/deepseek-chat",
        "LANGSMITH_API_KEY": _clean_env("LANGSMITH_API_KEY"),
        "LANGSMITH_PROJECT": _clean_env("LANGSMITH_PROJECT") or "clinical-intelligence-agent",
        "LANGSMITH_TRACING": _clean_env("LANGSMITH_TRACING") or "false",
        "DB_USER": _clean_env("DB_USER") or DEFAULT_DB_USER,
        "DB_PASSWORD": _clean_env("DB_PASSWORD") or DEFAULT_DB_PASSWORD,
        "DB_HOST": _clean_env("DB_HOST") or DEFAULT_DB_HOST,
        "DB_PORT": _clean_env("DB_PORT") or DEFAULT_DB_PORT,
        "DB_NAME": _clean_env("DB_NAME") or DEFAULT_DB_NAME,
        "DATABASE_URL": _build_database_url(),
        "AWS_ACCESS_KEY_ID": _clean_env("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": _clean_env("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": _clean_env("AWS_REGION") or "ap-south-1",
        "S3_BUCKET": _clean_env("S3_BUCKET"),
        "S3_PREFIX": _clean_env("S3_PREFIX") or "clinical-visits",
    }


settings = load_settings()
DEEPGRAM_API_KEY = settings["DEEPGRAM_API_KEY"]
GOOGLE_API_KEY = settings["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = settings["OPENROUTER_API_KEY"]
GROQ_API_KEY = settings["GROQ_API_KEY"]
LLM_PROVIDER = settings["LLM_PROVIDER"]
OPENROUTER_MODEL = settings["OPENROUTER_MODEL"]
DATABASE_URL = settings["DATABASE_URL"]
AWS_ACCESS_KEY_ID = settings["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = settings["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = settings["AWS_REGION"]
S3_BUCKET = settings["S3_BUCKET"]
S3_PREFIX = settings["S3_PREFIX"]
