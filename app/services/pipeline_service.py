from typing import Any, Optional

from app.graph.clinical_graph import ClinicalGraph
from app.rag.ingestion_service import IngestionService
from app.services.insights_service import InsightsService
from app.services.report_service import ReportService


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
    7. Build report insights + save analysis PDF and SOAP PDF to output/
    """

    def __init__(self):
        self.graph = ClinicalGraph()
        self.deepgram = self.graph.deepgram
        self.ingestion = IngestionService()
        self.insights = InsightsService()
        self.reports = ReportService()

    def ingest_guidelines(self) -> dict[str, Any]:
        return self.ingestion.ingest()

    def run_from_audio(self, audio_path: str, visit_id: Optional[str] = None) -> dict[str, Any]:
        result = self.graph.run_from_audio(audio_path, visit_id=visit_id)
        return self._finalize_with_reports(result)

    def run_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        transcript: str = "",
        visit_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> dict[str, Any]:
        result = self.graph.run_from_conversation(
            conversation=conversation,
            transcript=transcript,
            visit_id=visit_id,
            source_file=source_file,
        )
        return self._finalize_with_reports(result)

    def get_visit_state(self, visit_id: str) -> Optional[dict[str, Any]]:
        return self.graph.get_visit_state(visit_id)

    def _finalize_with_reports(self, result: dict[str, Any]) -> dict[str, Any]:
        conversation = (
            result.get("doctor_patient_conversation")
            or result.get("conversation")
            or []
        )
        insights = self.insights.build_insights(
            conversation=conversation,
            summary=result.get("summary") or {},
            entities=result.get("entities") or {},
            diagnoses=result.get("differential_diagnosis") or [],
        )
        result["report_insights"] = insights
        result["soap"] = insights.get("soap") or {}
        report_info = self.reports.save_reports(result)
        result.update(report_info)
        return result
