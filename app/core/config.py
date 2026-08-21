import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


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
    }


settings = load_settings()
DEEPGRAM_API_KEY = settings["DEEPGRAM_API_KEY"]
GOOGLE_API_KEY = settings["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = settings["OPENROUTER_API_KEY"]
GROQ_API_KEY = settings["GROQ_API_KEY"]
