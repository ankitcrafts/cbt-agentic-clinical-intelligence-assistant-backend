from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.core.config import load_settings


class LLMService:

    def __init__(self, provider="gemini"):
        self.provider = provider
        self.llm = None
        settings = load_settings()

        if provider == "gemini":
            google_api_key = settings["GOOGLE_API_KEY"]
            if not google_api_key:
                return

            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-3.6-flash",
                    google_api_key=google_api_key,
                    temperature=0,
                    max_retries=0,
                    timeout=10,
                )
            except Exception:
                self.llm = None

        elif provider == "openrouter":
            self.llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings["OPENROUTER_API_KEY"] or None,
                model="deepseek/deepseek-chat",
                timeout=10,
            )

        elif provider == "groq":
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=settings["GROQ_API_KEY"] or None,
                timeout=10,
            )

    def invoke(self, prompt) -> str:
        if self.llm is None:
            raise RuntimeError("LLM not initialized")

        response = self.llm.invoke(prompt)
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

