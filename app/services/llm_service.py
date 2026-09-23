import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import RateLimitError

from app.core.config import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    load_settings,
)


class LLMService:

    def __init__(self, provider=None):
        settings = load_settings()
        self.provider = provider or settings["LLM_PROVIDER"]
        self.llm = None

        if self.provider == "gemini":
            google_api_key = settings["GOOGLE_API_KEY"]
            if not google_api_key:
                return
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=google_api_key,
                temperature=0,
            )

        elif self.provider == "openrouter":
            self.llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings["OPENROUTER_API_KEY"] or None,
                model=settings["OPENROUTER_MODEL"],
                temperature=0,
            )

        elif self.provider == "groq":
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=settings["GROQ_API_KEY"] or None,
            )

    def invoke(self, prompt, retries: int = 4, base_delay: float = 8.0):
        if self.llm is None:
            raise RuntimeError(
                f"LLM provider '{self.provider}' is not configured. "
                "Check OPENROUTER_API_KEY or GOOGLE_API_KEY in app/.env."
            )
        last_error = None
        for attempt in range(retries):
            try:
                return self.llm.invoke(prompt).content
            except RateLimitError as exc:
                last_error = exc
                if attempt >= retries - 1:
                    break
                time.sleep(base_delay * (2 ** attempt))
        raise last_error
