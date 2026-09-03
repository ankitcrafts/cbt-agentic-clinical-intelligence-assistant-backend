from typing import Any, Optional

from app.graph.clinical_graph import ClinicalGraph
from app.rag.ingestion_service import IngestionService


class ClinicalPipeline:
    """
    Agentic clinical workflow backed by LangGraph + MemorySaver + RAG.

    Steps:
    1. Deepgram transcription (optional if conversation already exists)
    2. Doctor / Patient role labeling
    3. Encounter summary
    4. Medical entity extraction
    5. Guideline retrieval (RAG)
    6. Evidence-based differential diagnosis ranking
    """

    def __init__(self):
        self.graph = ClinicalGraph()
        self.deepgram = self.graph.deepgram
        self.ingestion = IngestionService()

    def ingest_guidelines(self) -> dict[str, Any]:
        return self.ingestion.ingest()

    def run_from_audio(self, audio_path: str, visit_id: Optional[str] = None) -> dict[str, Any]:
        return self.graph.run_from_audio(audio_path, visit_id=visit_id)

    def run_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        transcript: str = "",
        visit_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> dict[str, Any]:
        return self.graph.run_from_conversation(
            conversation=conversation,
            transcript=transcript,
            visit_id=visit_id,
            source_file=source_file,
        )

    def get_visit_state(self, visit_id: str) -> Optional[dict[str, Any]]:
        return self.graph.get_visit_state(visit_id)
