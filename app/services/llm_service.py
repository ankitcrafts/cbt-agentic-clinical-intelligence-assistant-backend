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
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=google_api_key,
                temperature=0,
            )

        elif provider == "openrouter":
            self.llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings["OPENROUTER_API_KEY"] or None,
                model="deepseek/deepseek-chat",
            )

        elif provider == "groq":
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=settings["GROQ_API_KEY"] or None,
            )

    def invoke(self, prompt) -> str:
        if self.llm is None:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing in app/.env. "
                "Add it to run doctor-patient conversion, summary, and entity extraction."
            )
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
