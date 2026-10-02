import os
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
except Exception:
    openai = None


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
        self.provider = str(provider or settings.get("LLM_PROVIDER") or "groq")
        self.llm = None
        self.fallback_llm = None
        self.tertiary_llm = None

        google_api_key = str(settings.get("GOOGLE_API_KEY") or "") or None
        openrouter_key = str(settings.get("OPENROUTER_API_KEY") or "") or None
        groq_key = str(settings.get("GROQ_API_KEY") or "") or None
        openrouter_model = str(settings.get("OPENROUTER_MODEL") or "qwen/qwen3.8-27b:free")

        # Initialize Groq client (High speed & low latency)
        groq_client = None
        if groq_key:
            try:
                if openai is not None:
                    groq_client = _DirectOpenAIWrapper(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=groq_key,
                        model="qwen/qwen3.8-27b",
                        timeout=30,
                        max_tokens=2048,
                    )
            except Exception as ge:
                print(f"[LLMService] Groq wrapper init warning: {ge}")

        # Initialize OpenRouter client
        openrouter_client = None
        if openrouter_key:
            try:
                if openai is not None:
                    openrouter_client = _DirectOpenAIWrapper(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_key,
                        model=openrouter_model,
                        timeout=45,
                        max_tokens=1500,
                    )
            except Exception as oe:
                print(f"[LLMService] OpenRouter wrapper init warning: {oe}")

        # Primary selection
        if self.provider == "groq" and groq_client:
            self.llm = groq_client
            self.fallback_llm = openrouter_client
        elif self.provider == "openrouter" and openrouter_client:
            self.llm = openrouter_client
            self.fallback_llm = groq_client
        else:
            self.llm = groq_client or openrouter_client
            self.fallback_llm = openrouter_client if self.llm != openrouter_client else groq_client

    def _extract_text(self, response) -> str:
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

    def invoke(self, prompt) -> str:
        # Try primary LLM
        if self.llm is not None:
            try:
                response = self.llm.invoke(prompt)
                return self._extract_text(response)
            except Exception as e:
                print(f"[LLMService Warning] Primary LLM ({self.provider}) invocation failed: {e}")
                if self.fallback_llm is not None:
                    print("[LLMService] Switching to fallback LLM...")
                    try:
                        fb_response = self.fallback_llm.invoke(prompt)
                        return self._extract_text(fb_response)
                    except Exception as fb_e:
                        print(f"[LLMService Error] Fallback invocation also failed: {fb_e}")
                        raise fb_e
                raise e

        # If primary wasn't initialized but fallback is ready
        if self.fallback_llm is not None:
            fb_response = self.fallback_llm.invoke(prompt)
            return self._extract_text(fb_response)

        raise RuntimeError(
            f"LLM provider '{self.provider}' is not configured. "
            "Check GROQ_API_KEY or OPENROUTER_API_KEY in environment variables."
        )
