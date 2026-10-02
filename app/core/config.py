import os
from pathlib import Path
from dotenv import load_dotenv

# Project roots & directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_ENV = PROJECT_ROOT / ".env"
APP_ENV = Path(__file__).resolve().parent.parent / ".env"

GUIDELINES_DIR = PROJECT_ROOT / "Rag" / "data" / "guidelines"
VECTORSTORE_DIR = PROJECT_ROOT / "Rag" / "vectorstore"

# Default one-click Postgres (docker-compose / local)
DEFAULT_DB_USER = "clinIQ_user"
DEFAULT_DB_PASSWORD = "password"
DEFAULT_DB_NAME = "cliniIQ"
DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_PORT = "5432"


def _clean_env(name: str, default: str = "") -> str:
    """Read and clean an environment variable."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().strip('"').strip("'")


def _build_database_url() -> str:
    explicit = _clean_env("DATABASE_URL")
    if explicit:
        return explicit
    user = _clean_env("POSTGRES_USER") or _clean_env("DB_USER") or DEFAULT_DB_USER
    password = _clean_env("POSTGRES_PASSWORD") or _clean_env("DB_PASSWORD") or DEFAULT_DB_PASSWORD
    host = _clean_env("POSTGRES_HOST") or _clean_env("DB_HOST") or DEFAULT_DB_HOST
    port = _clean_env("POSTGRES_PORT") or _clean_env("DB_PORT") or DEFAULT_DB_PORT
    name = _clean_env("POSTGRES_DB") or _clean_env("DB_NAME") or DEFAULT_DB_NAME
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def load_settings():
    """Load application configuration from environment variables."""

    # Load .env files if they exist.
    load_dotenv(ROOT_ENV, override=True)
    load_dotenv(APP_ENV, override=True)

    db_url = _build_database_url()
    pg_user = _clean_env("POSTGRES_USER") or _clean_env("DB_USER", DEFAULT_DB_USER)
    pg_password = _clean_env("POSTGRES_PASSWORD") or _clean_env("DB_PASSWORD", DEFAULT_DB_PASSWORD)
    pg_host = _clean_env("POSTGRES_HOST") or _clean_env("DB_HOST", DEFAULT_DB_HOST)
    pg_port = _clean_env("POSTGRES_PORT") or _clean_env("DB_PORT", DEFAULT_DB_PORT)
    pg_db = _clean_env("POSTGRES_DB") or _clean_env("DB_NAME", DEFAULT_DB_NAME)

    return {
        # AI / API Keys
        "DEEPGRAM_API_KEY": _clean_env("DEEPGRAM_API_KEY"),
        "GOOGLE_API_KEY": _clean_env("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": _clean_env("OPENROUTER_API_KEY"),
        "GROQ_API_KEY": _clean_env("GROQ_API_KEY"),
        "LLM_PROVIDER": _clean_env("LLM_PROVIDER", "openrouter"),
        "OPENROUTER_MODEL": _clean_env("OPENROUTER_MODEL", "deepseek/deepseek-chat"),

        # LangSmith Tracing
        "LANGSMITH_API_KEY": _clean_env("LANGSMITH_API_KEY"),
        "LANGSMITH_PROJECT": _clean_env("LANGSMITH_PROJECT", "clinical-intelligence-agent"),
        "LANGSMITH_TRACING": _clean_env("LANGSMITH_TRACING", "false"),

        # Database Settings
        "DATABASE_URL": db_url,
        "POSTGRES_USER": pg_user,
        "POSTGRES_PASSWORD": pg_password,
        "POSTGRES_HOST": pg_host,
        "POSTGRES_PORT": pg_port,
        "POSTGRES_DB": pg_db,
        "DB_USER": pg_user,
        "DB_PASSWORD": pg_password,
        "DB_HOST": pg_host,
        "DB_PORT": pg_port,
        "DB_NAME": pg_db,

        # JWT
        "JWT_SECRET_KEY": _clean_env("JWT_SECRET_KEY", "clinIQ_super_secret_jwt_key_2026"),
        "JWT_ALGORITHM": _clean_env("JWT_ALGORITHM", "HS256"),
        "ACCESS_TOKEN_EXPIRE_MINUTES": int(_clean_env("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")),  # 24 hours
        "OTP_EXPIRE_MINUTES": int(_clean_env("OTP_EXPIRE_MINUTES", "10")),  # 10 minutes

        # Gmail SMTP Settings
        "SMTP_HOST": _clean_env("SMTP_HOST", "smtp.gmail.com"),
        "SMTP_PORT": int(_clean_env("SMTP_PORT", "587")),
        "SMTP_EMAIL": _clean_env("SMTP_EMAIL"),
        "SMTP_PASSWORD": _clean_env("SMTP_PASSWORD").replace(" ", ""),
        "SMTP_FROM_NAME": _clean_env("SMTP_FROM_NAME", "ClinIQ Health Assistant"),

        # AWS / S3
        "AWS_ACCESS_KEY_ID": _clean_env("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": _clean_env("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": _clean_env("AWS_REGION", "ap-south-1"),
        "S3_BUCKET": _clean_env("S3_BUCKET"),
        "S3_PREFIX": _clean_env("S3_PREFIX", "clinical-visits"),

        # Google Drive Integration Settings
        "GDRIVE_FOLDER_ID": _clean_env("GDRIVE_FOLDER_ID"),
        "GDRIVE_WEBHOOK_URL": _clean_env("GDRIVE_WEBHOOK_URL"),
        "GDRIVE_CREDENTIALS_FILE": _clean_env("GDRIVE_CREDENTIALS_FILE", "service_account.json"),
        "GDRIVE_SERVICE_ACCOUNT_JSON": _clean_env("GDRIVE_SERVICE_ACCOUNT_JSON"),
    }


# ----------------------------------------------------------------------
# Load settings
# ----------------------------------------------------------------------
settings = load_settings()

# AI / API Keys
DEEPGRAM_API_KEY: str = str(settings["DEEPGRAM_API_KEY"])
GOOGLE_API_KEY: str = str(settings["GOOGLE_API_KEY"])
OPENROUTER_API_KEY: str = str(settings["OPENROUTER_API_KEY"])
GROQ_API_KEY: str = str(settings["GROQ_API_KEY"])
LLM_PROVIDER: str = str(settings["LLM_PROVIDER"])
OPENROUTER_MODEL: str = str(settings["OPENROUTER_MODEL"])

# LangSmith Tracing
LANGSMITH_API_KEY: str = str(settings["LANGSMITH_API_KEY"])
LANGSMITH_PROJECT: str = str(settings["LANGSMITH_PROJECT"])
LANGSMITH_TRACING: str = str(settings["LANGSMITH_TRACING"])

# Database
DATABASE_URL: str = str(settings["DATABASE_URL"])
POSTGRES_USER: str = str(settings["POSTGRES_USER"])
POSTGRES_PASSWORD: str = str(settings["POSTGRES_PASSWORD"])
POSTGRES_HOST: str = str(settings["POSTGRES_HOST"])
POSTGRES_PORT: str = str(settings["POSTGRES_PORT"])
POSTGRES_DB: str = str(settings["POSTGRES_DB"])
DB_USER: str = str(settings["DB_USER"])
DB_PASSWORD: str = str(settings["DB_PASSWORD"])
DB_HOST: str = str(settings["DB_HOST"])
DB_PORT: str = str(settings["DB_PORT"])
DB_NAME: str = str(settings["DB_NAME"])

# JWT
JWT_SECRET_KEY: str = str(settings["JWT_SECRET_KEY"])
JWT_ALGORITHM: str = str(settings["JWT_ALGORITHM"])
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(settings["ACCESS_TOKEN_EXPIRE_MINUTES"])
OTP_EXPIRE_MINUTES: int = int(settings["OTP_EXPIRE_MINUTES"])

# SMTP
SMTP_HOST: str = str(settings["SMTP_HOST"])
SMTP_PORT: int = int(settings["SMTP_PORT"])
SMTP_EMAIL: str = str(settings["SMTP_EMAIL"])
SMTP_PASSWORD: str = str(settings["SMTP_PASSWORD"])
SMTP_FROM_NAME: str = str(settings["SMTP_FROM_NAME"])

# AWS / S3
AWS_ACCESS_KEY_ID: str = str(settings["AWS_ACCESS_KEY_ID"])
AWS_SECRET_ACCESS_KEY: str = str(settings["AWS_SECRET_ACCESS_KEY"])
AWS_REGION: str = str(settings["AWS_REGION"])
S3_BUCKET: str = str(settings["S3_BUCKET"])
S3_PREFIX: str = str(settings["S3_PREFIX"])

# Google Drive
GDRIVE_FOLDER_ID: str = str(settings["GDRIVE_FOLDER_ID"])
GDRIVE_WEBHOOK_URL: str = str(settings["GDRIVE_WEBHOOK_URL"])
GDRIVE_CREDENTIALS_FILE: str = str(settings["GDRIVE_CREDENTIALS_FILE"])
GDRIVE_SERVICE_ACCOUNT_JSON: str = str(settings["GDRIVE_SERVICE_ACCOUNT_JSON"])
