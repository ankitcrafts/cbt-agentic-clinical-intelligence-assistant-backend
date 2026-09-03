from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.core.config import load_settings


class LLMService:

    def __init__(self, provider="gemini"):
        self.provider = provider
        self.llm = None
        self.fallback_llm = None
        settings = load_settings()

        google_api_key = settings.get("GOOGLE_API_KEY")
        openrouter_key = settings.get("OPENROUTER_API_KEY")

        # Initialize primary LLM
        if provider == "gemini":
            if google_api_key:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-3.6-flash",
                        google_api_key=google_api_key,
                        temperature=0.1,
                        max_retries=1,
                        timeout=60,
                    )
                except Exception as e:
                    print(f"[LLMService] Gemini init warning: {e}")
                    self.llm = None

            # Set OpenRouter as resilient fallback
            if openrouter_key:
                try:
                    self.fallback_llm = ChatOpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_key,
                        model="deepseek/deepseek-chat",
                        timeout=45,
                    )
                except Exception as fb_err:
                    print(f"[LLMService] Fallback OpenRouter init warning: {fb_err}")

        elif provider == "openrouter":
            if openrouter_key:
                self.llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    model="deepseek/deepseek-chat",
                    timeout=45,
                )

        elif provider == "groq":
            groq_key = settings.get("GROQ_API_KEY")
            if groq_key:
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=groq_key,
                    timeout=30,
                )

    def _extract_text(self, response) -> str:
        content = response.content
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
                    print("[LLMService] Switching to OpenRouter fallback LLM...")
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

        raise RuntimeError("No LLM provider initialized successfully")

