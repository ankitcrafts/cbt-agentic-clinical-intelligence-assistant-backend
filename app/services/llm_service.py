import os
import time
from typing import Any, Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    import openai
    from openai import RateLimitError
except Exception:
    openai = None
    RateLimitError = Exception


class _DirectOpenAIWrapper:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 45, max_tokens: int = 2048):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout) if openai else None

    def invoke(self, prompt):
        if not self.client:
            raise RuntimeError("Direct OpenAI client is not available.")
        text_prompt = str(prompt)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text_prompt}],
            temperature=0.2,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


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
        self.provider = str(provider or settings.get("LLM_PROVIDER") or "openrouter")
        self.llm = None
        self.fallback_llm = None

        google_api_key = str(settings.get("GOOGLE_API_KEY") or "") or None
        openrouter_key = str(settings.get("OPENROUTER_API_KEY") or "") or None
        groq_key = str(settings.get("GROQ_API_KEY") or "") or None
        openrouter_model = str(settings.get("OPENROUTER_MODEL") or "deepseek/deepseek-chat")

        # 1. Gemini
        gemini_client = None
        if google_api_key and ChatGoogleGenerativeAI:
            try:
                gemini_client = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.2,
                )
            except Exception as ge:
                print(f"[LLMService] Gemini init warning: {ge}")

        # 2. OpenRouter
        openrouter_client = None
        if openrouter_key:
            if ChatOpenAI:
                try:
                    openrouter_client = ChatOpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_key,
                        model=openrouter_model,
                        temperature=0.2,
                        max_tokens=2048,
                    )
                except Exception as oe:
                    print(f"[LLMService] OpenRouter ChatOpenAI warning: {oe}")
            if not openrouter_client and openai:
                openrouter_client = _DirectOpenAIWrapper(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    model=openrouter_model,
                    timeout=45,
                    max_tokens=2048,
                )

        # 3. Groq
        groq_client = None
        groq_model = str(settings.get("GROQ_MODEL") or "qwen/qwen3.8-27b")
        if groq_key:
            if ChatGroq:
                try:
                    groq_client = ChatGroq(
                        model=groq_model,
                        api_key=groq_key,
                        max_tokens=2048,
                    )
                except Exception as gqe:
                    print(f"[LLMService] ChatGroq warning: {gqe}")
            if not groq_client and openai:
                groq_client = _DirectOpenAIWrapper(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_key,
                    model=groq_model,
                    timeout=30,
                    max_tokens=2048,
                )

        if self.provider == "gemini":
            self.llm = gemini_client or openrouter_client or groq_client
            self.fallback_llm = openrouter_client or groq_client
        elif self.provider == "groq":
            self.llm = groq_client or openrouter_client or gemini_client
            self.fallback_llm = openrouter_client or gemini_client
        else:  # openrouter default
            self.llm = openrouter_client or groq_client or gemini_client
            self.fallback_llm = groq_client or gemini_client

    def _extract_text(self, response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            return "\n".join(text_parts)
        elif isinstance(content, str):
            return content
        return str(content)

    def invoke(self, prompt: Any, retries: int = 3, base_delay: float = 2.0) -> str:
        if self.llm is None and self.fallback_llm is None:
            raise RuntimeError(
                f"LLM provider '{self.provider}' is not configured. "
                "Check OPENROUTER_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in environment variables."
            )

        last_error = None
        # Try primary LLM with retries
        if self.llm is not None:
            for attempt in range(retries):
                try:
                    response = self.llm.invoke(prompt)
                    return self._extract_text(response)
                except RateLimitError as exc:
                    last_error = exc
                    if attempt < retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                except Exception as exc:
                    last_error = exc
                    print(f"[LLMService Warning] Primary LLM invoke failed: {exc}")
                    break

        # Try fallback LLM
        if self.fallback_llm is not None:
            print("[LLMService] Attempting fallback LLM invocation...")
            for attempt in range(retries):
                try:
                    fb_response = self.fallback_llm.invoke(prompt)
                    return self._extract_text(fb_response)
                except Exception as fb_exc:
                    last_error = fb_exc
                    if attempt < retries - 1:
                        time.sleep(base_delay * (2 ** attempt))

        raise last_error or RuntimeError("LLM invocation failed across all configured providers.")
