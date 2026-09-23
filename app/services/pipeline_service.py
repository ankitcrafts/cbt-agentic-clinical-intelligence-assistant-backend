from typing import Any, Optional

from app.graph.clinical_graph import ClinicalGraph
from app.rag.ingestion_service import IngestionService
from app.services.followup_service import (
    CONFIDENCE_FOLLOW_UP_THRESHOLD,
    MAX_FOLLOW_UP_ROUNDS,
    FollowUpService,
)
from app.services.insights_service import InsightsService
from app.services.report_service import ReportService
from app.services.json_utils import sanitize_for_state


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
    7. If confidence < 0.70, attach follow-up questions (one round only)
    8. Build report insights + save analysis PDF and SOAP PDF to output/
    """

    def __init__(self):
        self.graph = ClinicalGraph()
        self.deepgram = self.graph.deepgram
        self.ingestion = IngestionService()
        self.insights = InsightsService()
        self.reports = ReportService()
        self.followup = FollowUpService()
        self._visits: dict[str, dict[str, Any]] = {}

    def ingest_guidelines(self) -> dict[str, Any]:
        return self.ingestion.ingest()

    def run_from_audio(
        self,
        audio_path: str,
        visit_id: Optional[str] = None,
        force_follow_up: bool = False,
    ) -> dict[str, Any]:
        result = self.graph.run_from_audio(audio_path, visit_id=visit_id)
        return self._finalize_with_reports(result, force_follow_up=force_follow_up)

    def run_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        transcript: str = "",
        visit_id: Optional[str] = None,
        source_file: Optional[str] = None,
        force_follow_up: bool = False,
    ) -> dict[str, Any]:
        result = self.graph.run_from_conversation(
            conversation=conversation,
            transcript=transcript,
            visit_id=visit_id,
            source_file=source_file,
        )
        return self._finalize_with_reports(result, force_follow_up=force_follow_up)

    def get_visit_state(self, visit_id: str) -> Optional[dict[str, Any]]:
        if visit_id in self._visits:
            return self._visits[visit_id]
        return self.graph.get_visit_state(visit_id)

    def apply_follow_up(
        self,
        visit_id: str,
        answers: Optional[list[dict[str, Any]]] = None,
        conversation: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        prior = self.get_visit_state(visit_id)
        if not prior:
            raise ValueError(f"No visit found for visit_id={visit_id}")

        iteration = int(prior.get("follow_up_iteration") or 0)
        if iteration >= MAX_FOLLOW_UP_ROUNDS:
            raise ValueError(
                "Follow-up already used for this visit. No further follow-up rounds are allowed."
            )
        pending_questions = prior.get("follow_up_questions") or []
        if prior.get("status") != "needs_follow_up" and not pending_questions:
            raise ValueError(
                "Visit has no pending follow-up. Re-run analyze with confidence below "
                f"{CONFIDENCE_FOLLOW_UP_THRESHOLD:.2f} (or force_follow_up=true for testing)."
            )

        previous_confidence = self.followup.top_confidence(
            prior.get("differential_diagnosis") or []
        )
        previous_summary = prior.get("summary") or {}

        new_turns: list[dict[str, Any]]
        if conversation:
            # Accept either raw Deepgram turns or role-labeled turns
            if conversation and ("role" in conversation[0] or "speaker" in conversation[0]):
                if "role" in conversation[0]:
                    new_turns = conversation
                else:
                    labeled = self.graph.conversation.to_doctor_patient(conversation)
                    new_turns = labeled.get("conversation") or []
            else:
                new_turns = []
        elif answers:
            new_turns = self.followup.answers_to_conversation(
                prior.get("follow_up_questions") or [],
                answers,
            )
            # ensure role fields for downstream services
            for turn in new_turns:
                if "role" not in turn:
                    turn["role"] = "Doctor" if turn.get("speaker") == "Speaker 1" else "Patient"
        else:
            raise ValueError("Provide either answers or conversation for follow-up.")

        if not new_turns:
            raise ValueError("Follow-up produced no usable conversation turns.")

        old_conversation = list(prior.get("doctor_patient_conversation") or [])
        merged_conversation = old_conversation + new_turns

        merged_summary = self.followup.merge_summary(previous_summary, new_turns)
        entities = self.graph.entity.extract(merged_conversation)
        retrieval = self.graph.retrieval.retrieve(merged_summary, entities)
        chunks = retrieval.get("retrieved_chunks") or []
        diagnosis = self.graph.rag_diagnosis.rank_diagnoses(
            merged_summary,
            entities,
            chunks,
        )

        result = {
            "visit_id": visit_id,
            "thread_id": visit_id,
            "transcript": prior.get("transcript", ""),
            "conversation": prior.get("conversation") or [],
            "doctor_patient_conversation": merged_conversation,
            "speaker_map": prior.get("speaker_map") or {},
            "summary": merged_summary,
            "entities": entities,
            "retrieved_chunks": chunks,
            "retrieval_mode": retrieval.get("retrieval_mode", ""),
            "routed_documents": retrieval.get("routed_documents", []),
            "retrieval_query": retrieval.get("query", ""),
            "dense_hit_count": retrieval.get("dense_hit_count", 0),
            "bm25_hit_count": retrieval.get("bm25_hit_count", 0),
            "primary_guideline_used": diagnosis.get("primary_guideline_used", ""),
            "differential_diagnosis": diagnosis.get("differential_diagnosis") or [],
            "source_file": prior.get("source_file"),
            "previous_summary": previous_summary,
            "previous_confidence": previous_confidence,
            "follow_up_iteration": 1,
            "follow_up_questions": [],
            "status": "complete",
        }
        new_confidence = self.followup.top_confidence(
            result["differential_diagnosis"]
        )
        result["confidence"] = new_confidence
        result["confidence_improved"] = new_confidence > previous_confidence
        result["follow_up_threshold"] = CONFIDENCE_FOLLOW_UP_THRESHOLD
        result["follow_up_note"] = (
            "Follow-up round completed. No further follow-up will be generated "
            "even if confidence remains below threshold."
        )

        finalized = self._attach_reports(result)
        self._visits[visit_id] = finalized
        return finalized

    def _finalize_with_reports(
        self,
        result: dict[str, Any],
        force_follow_up: bool = False,
    ) -> dict[str, Any]:
        diagnoses = result.get("differential_diagnosis") or []
        top_confidence = self.followup.top_confidence(diagnoses)
        result["confidence"] = top_confidence
        result["follow_up_threshold"] = CONFIDENCE_FOLLOW_UP_THRESHOLD
        result["follow_up_iteration"] = int(result.get("follow_up_iteration") or 0)

        if self.followup.needs_follow_up(
            diagnoses,
            follow_up_iteration=result["follow_up_iteration"],
            force=force_follow_up,
        ):
            questions = self.followup.generate_questions(
                summary=result.get("summary") or {},
                entities=result.get("entities") or {},
                diagnoses=diagnoses,
            )
            result["status"] = "needs_follow_up"
            result["follow_up_questions"] = questions
            result["follow_up_note"] = (
                f"Top confidence {top_confidence:.3f} is below "
                f"{CONFIDENCE_FOLLOW_UP_THRESHOLD:.2f}. Ask follow-up questions, "
                "then POST /visit/{visit_id}/follow-up with answers."
            )
        else:
            result["status"] = "complete"
            result["follow_up_questions"] = []
            result["follow_up_note"] = (
                "Confidence at or above threshold, or follow-up already used."
            )

        finalized = self._attach_reports(result)
        visit_id = finalized.get("visit_id")
        if visit_id:
            self._visits[visit_id] = finalized
        return finalized

    def _attach_reports(self, result: dict[str, Any]) -> dict[str, Any]:
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
        result["report_insights"] = sanitize_for_state(insights)
        result["soap"] = insights.get("soap") or {}
        report_info = self.reports.save_reports(result)
        result.update(report_info)
        return result
