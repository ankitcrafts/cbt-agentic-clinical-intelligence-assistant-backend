import os
from pathlib import Path
from dotenv import load_dotenv

# Project root:
# app/core/config.py
#   -> core
#   -> app
#   -> project root
ROOT_ENV = Path(__file__).resolve().parent.parent.parent / ".env"

# app/.env
APP_ENV = Path(__file__).resolve().parent.parent / ".env"


def _clean_env(name: str, default: str = "") -> str:
    """Read and clean an environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().strip('"').strip("'")


def load_settings():
    """Load application configuration from environment variables."""

    # Load .env files if they exist.
    load_dotenv(ROOT_ENV, override=True)
    load_dotenv(APP_ENV, override=True)

    # ------------------------------------------------------------------
    # DB settings with auto-fallback to localhost postgres
    # ------------------------------------------------------------------
    pg_user = _clean_env("POSTGRES_USER", "postgres")
    pg_password = _clean_env("POSTGRES_PASSWORD", "postgres")
    pg_host = _clean_env("POSTGRES_HOST", "localhost")
    pg_port = _clean_env("POSTGRES_PORT", "5432")
    pg_db = _clean_env("POSTGRES_DB", "cliniq_db")
    
    default_db_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    db_url = _clean_env("DATABASE_URL", default_db_url)
    
    # ------------------------------------------------------------------
    # Application settings
    # ------------------------------------------------------------------

    return {
        # AI / API Keys
        "DEEPGRAM_API_KEY": _clean_env("DEEPGRAM_API_KEY"),
        "GOOGLE_API_KEY": _clean_env("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": _clean_env("OPENROUTER_API_KEY"),
        "GROQ_API_KEY": _clean_env("GROQ_API_KEY"),
        
        # Database Settings
        "DATABASE_URL": db_url,
        "POSTGRES_USER": pg_user,
        "POSTGRES_PASSWORD": pg_password,
        "POSTGRES_HOST": pg_host,
        "POSTGRES_PORT": pg_port,
        "POSTGRES_DB": pg_db,
        
        # JWT
        # IMPORTANT:
        # JWT Security Settings - Do not use a hardcoded production secret.
        "JWT_SECRET_KEY": _clean_env("JWT_SECRET_KEY"),
        "JWT_ALGORITHM": _clean_env("JWT_ALGORITHM", "HS256"),
        "ACCESS_TOKEN_EXPIRE_MINUTES": int(_clean_env("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")), # 24 hours
        "OTP_EXPIRE_MINUTES": int(_clean_env("OTP_EXPIRE_MINUTES", "10")), # 10 minutes
        
        # Gmail SMTP Settings
        "SMTP_HOST": _clean_env("SMTP_HOST", "smtp.gmail.com"),
        "SMTP_PORT": int(_clean_env("SMTP_PORT", "587")),
        "SMTP_EMAIL": _clean_env("SMTP_EMAIL"),
        "SMTP_PASSWORD": _clean_env("SMTP_PASSWORD").replace(" ", ""),
        "SMTP_FROM_NAME": _clean_env("SMTP_FROM_NAME", "ClinIQ Health Assistant"),
        
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

# ----------------------------------------------------------------------
# Export configuration values
# ----------------------------------------------------------------------

DEEPGRAM_API_KEY = settings["DEEPGRAM_API_KEY"]
GOOGLE_API_KEY = settings["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = settings["OPENROUTER_API_KEY"]
GROQ_API_KEY = settings["GROQ_API_KEY"]

# Database
DATABASE_URL = settings["DATABASE_URL"]
POSTGRES_USER = settings["POSTGRES_USER"]
POSTGRES_PASSWORD = settings["POSTGRES_PASSWORD"]
POSTGRES_HOST = settings["POSTGRES_HOST"]
POSTGRES_PORT = settings["POSTGRES_PORT"]
POSTGRES_DB = settings["POSTGRES_DB"]

# JWT
JWT_SECRET_KEY = settings["JWT_SECRET_KEY"]
JWT_ALGORITHM = settings["JWT_ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = settings["ACCESS_TOKEN_EXPIRE_MINUTES"]
OTP_EXPIRE_MINUTES = settings["OTP_EXPIRE_MINUTES"]

# SMTP
SMTP_HOST = settings["SMTP_HOST"]
SMTP_PORT = settings["SMTP_PORT"]
SMTP_EMAIL = settings["SMTP_EMAIL"]
SMTP_PASSWORD = settings["SMTP_PASSWORD"]
SMTP_FROM_NAME = settings["SMTP_FROM_NAME"]

# Google Drive
GDRIVE_FOLDER_ID = settings["GDRIVE_FOLDER_ID"]
GDRIVE_WEBHOOK_URL = settings["GDRIVE_WEBHOOK_URL"]
GDRIVE_CREDENTIALS_FILE = settings["GDRIVE_CREDENTIALS_FILE"]
GDRIVE_SERVICE_ACCOUNT_JSON = settings["GDRIVE_SERVICE_ACCOUNT_JSON"]

