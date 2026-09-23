import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GUIDELINES_DIR = PROJECT_ROOT / "Rag" / "data" / "guidelines"
VECTORSTORE_DIR = PROJECT_ROOT / "Rag" / "vectorstore"


def _clean_env(name: str) -> str:
    value = os.getenv(name) or ""
    return value.strip().strip('"').strip("'")


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
    }


settings = load_settings()
DEEPGRAM_API_KEY = settings["DEEPGRAM_API_KEY"]
GOOGLE_API_KEY = settings["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = settings["OPENROUTER_API_KEY"]
GROQ_API_KEY = settings["GROQ_API_KEY"]
LLM_PROVIDER = settings["LLM_PROVIDER"]
OPENROUTER_MODEL = settings["OPENROUTER_MODEL"]
