from typing import Any, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.tracing import build_run_config
from app.graph.state import ClinicalState
from app.services.json_utils import sanitize_for_state
from app.rag.rag_diagnosis_service import RagDiagnosisService
from app.rag.retrieval_service import RetrievalService
from app.services.conversation_service import ConversationService
from app.services.deepgram_service import DeepgramService
from app.services.entity_service import EntityService
from app.services.summary_service import SummaryService


class ClinicalGraph:
    def __init__(self):
        self.deepgram = DeepgramService()
        self.conversation = ConversationService()
        self.summary = SummaryService()
        self.entity = EntityService()
        self.retrieval = RetrievalService()
        self.rag_diagnosis = RagDiagnosisService()
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ClinicalState)

        graph.add_node("transcribe", self._transcribe_node)
        graph.add_node("role_label", self._role_label_node)
        graph.add_node("summarize", self._summarize_node)
        graph.add_node("extract_entities", self._extract_entities_node)
        graph.add_node("retrieve_guidelines", self._retrieve_guidelines_node)
        graph.add_node("rag_diagnose", self._rag_diagnose_node)

        graph.set_conditional_entry_point(
            self._route_start,
            {
                "transcribe": "transcribe",
                "role_label": "role_label",
            },
        )
        graph.add_edge("transcribe", "role_label")
        graph.add_edge("role_label", "summarize")
        graph.add_edge("summarize", "extract_entities")
        graph.add_edge("extract_entities", "retrieve_guidelines")
        graph.add_edge("retrieve_guidelines", "rag_diagnose")
        graph.add_edge("rag_diagnose", END)

        return graph.compile(checkpointer=self.memory)

    def _route_start(self, state: ClinicalState) -> str:
        if state.get("conversation"):
            return "role_label"
        return "transcribe"

    def _transcribe_node(self, state: ClinicalState) -> dict[str, Any]:
        asr = self.deepgram.transcribe(state["audio_path"])
        return {
            "transcript": asr.get("transcript", ""),
            "conversation": asr["conversation"],
        }

    def _role_label_node(self, state: ClinicalState) -> dict[str, Any]:
        labeled = self.conversation.to_doctor_patient(state["conversation"])
        return {
            "speaker_map": labeled["speaker_map"],
            "doctor_patient_conversation": labeled["conversation"],
        }

    def _summarize_node(self, state: ClinicalState) -> dict[str, Any]:
        return {
            "summary": self.summary.summarize(state["doctor_patient_conversation"]),
        }

    def _extract_entities_node(self, state: ClinicalState) -> dict[str, Any]:
        return {
            "entities": self.entity.extract(state["doctor_patient_conversation"]),
        }

    def _retrieve_guidelines_node(self, state: ClinicalState) -> dict[str, Any]:
        chunks = self.retrieval.retrieve(state["summary"], state["entities"])
        return sanitize_for_state(
            {
                "retrieved_chunks": chunks,
                "rag_context": self.retrieval.format_context(chunks),
            }
        )

    def _rag_diagnose_node(self, state: ClinicalState) -> dict[str, Any]:
        result = self.rag_diagnosis.rank_diagnoses(
            state["summary"],
            state["entities"],
            state["retrieved_chunks"],
        )
        return sanitize_for_state(
            {
                "primary_guideline_used": result.get("primary_guideline_used", ""),
                "differential_diagnosis": result.get("differential_diagnosis", []),
            }
        )

    def _resolve_visit_id(self, visit_id: Optional[str] = None) -> str:
        cleaned = (visit_id or "").strip()
        if cleaned:
            return cleaned
        return f"visit-{uuid4().hex[:8]}"

    def _run_config(
        self,
        visit_id: str,
        source: str = "api",
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return build_run_config(
            visit_id,
            source=source,
            extra_metadata=extra_metadata,
        )

    def run_from_audio(
        self,
        audio_path: str,
        visit_id: Optional[str] = None,
        source: str = "analyze_audio",
    ) -> dict[str, Any]:
        resolved_visit_id = self._resolve_visit_id(visit_id)
        result = self.graph.invoke(
            {"audio_path": audio_path, "visit_id": resolved_visit_id},
            config=self._run_config(
                resolved_visit_id,
                source=source,
                extra_metadata={"audio_path": audio_path},
            ),
        )
        return self._format_result(result, resolved_visit_id)

    def run_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        transcript: str = "",
        visit_id: Optional[str] = None,
        source_file: Optional[str] = None,
        source: str = "analyze_conversation",
    ) -> dict[str, Any]:
        resolved_visit_id = self._resolve_visit_id(visit_id)
        initial_state: ClinicalState = {
            "conversation": conversation,
            "transcript": transcript,
            "visit_id": resolved_visit_id,
        }
        if source_file:
            initial_state["source_file"] = source_file

        result = self.graph.invoke(
            initial_state,
            config=self._run_config(
                resolved_visit_id,
                source=source,
                extra_metadata={"source_file": source_file or ""},
            ),
        )
        return self._format_result(result, resolved_visit_id)

    def get_visit_state(self, visit_id: str) -> Optional[dict[str, Any]]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": visit_id}})
        if not snapshot or not snapshot.values:
            return None
        return self._format_result(snapshot.values, visit_id)

    def _format_result(self, state: ClinicalState, visit_id: str) -> dict[str, Any]:
        return {
            "visit_id": state.get("visit_id") or visit_id,
            "thread_id": visit_id,
            "transcript": state.get("transcript", ""),
            "conversation": state.get("conversation", []),
            "doctor_patient_conversation": state.get("doctor_patient_conversation", []),
            "speaker_map": state.get("speaker_map", {}),
            "summary": state.get("summary", {}),
            "entities": state.get("entities", {}),
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "primary_guideline_used": state.get("primary_guideline_used", ""),
            "differential_diagnosis": state.get("differential_diagnosis", []),
            "source_file": state.get("source_file"),
        }
