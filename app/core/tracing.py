import os
from typing import Any, Optional

from app.core.config import load_settings

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


def setup_langsmith_tracing() -> dict[str, Any]:
    """Enable LangSmith tracing for LangChain, LangGraph, and @traceable spans."""
    settings = load_settings()
    api_key = settings.get("LANGSMITH_API_KEY", "")
    project = settings.get("LANGSMITH_PROJECT", "clinical-intelligence-agent")
    enabled = settings.get("LANGSMITH_TRACING", "false").lower() == "true" and bool(api_key)

    if not enabled:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return {
            "enabled": False,
            "project": project,
            "message": "LangSmith tracing disabled. Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in app/.env",
        }

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGSMITH_TRACING"] = "true"

    return {
        "enabled": True,
        "project": project,
        "dashboard_url": f"https://smith.langchain.com/o/default/projects/p/{project}",
        "message": "LangSmith tracing enabled",
    }


def build_run_config(
    visit_id: str,
    *,
    source: str = "api",
    run_name: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = {
        "visit_id": visit_id,
        "source": source,
        "service": "clinical-intelligence-agent",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "configurable": {"thread_id": visit_id},
        "metadata": metadata,
        "tags": ["clinical-agent", "langgraph", source],
        "run_name": run_name or f"clinical_pipeline:{visit_id}",
    }


def tracing_status() -> dict[str, Any]:
    settings = load_settings()
    enabled = os.getenv("LANGCHAIN_TRACING_V2") == "true" and bool(settings.get("LANGSMITH_API_KEY"))
    project = settings.get("LANGSMITH_PROJECT", "clinical-intelligence-agent")
    return {
        "tracing_enabled": enabled,
        "project": project,
        "dashboard_url": f"https://smith.langchain.com/" if enabled else None,
        "tracked_components": [
            "LangGraph nodes (transcribe → rag_diagnose)",
            "LLM calls (OpenRouter via LangChain)",
            "Deepgram transcription",
            "RAG retrieval and diagnosis ranking",
        ],
    }
