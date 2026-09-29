from typing import Any, Optional

from app.db.visit_store import VisitStore
from app.graph.clinical_graph import ClinicalGraph
from app.rag.ingestion_service import IngestionService
from app.services.followup_service import (
    CONFIDENCE_FOLLOW_UP_THRESHOLD,
    MAX_FOLLOW_UP_ROUNDS,
    FollowUpService,
)
from app.services.insights_service import InsightsService
from app.services.report_service import ReportService
from app.services.ragas_service import RagasEvaluationService
from app.services.json_utils import sanitize_for_state
from app.storage.s3_service import S3Service


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
    9. Persist visit (Postgres) + artifacts (S3) when configured
    10. Optional RAGAS validation scores in JSON
    """

    def __init__(self):
        self.graph = ClinicalGraph()
        self.deepgram = self.graph.deepgram
        self.ingestion = IngestionService()
        self.insights = InsightsService()
        self.reports = ReportService()
        self.followup = FollowUpService()
        self.visit_store = VisitStore()
        self.s3 = S3Service()
        self.ragas = RagasEvaluationService()
        self._visits: dict[str, dict[str, Any]] = {}

    def ingest_guidelines(self) -> dict[str, Any]:
        return self.ingestion.ingest()

    def persistence_status(self) -> dict[str, Any]:
        return {
            "postgres_enabled": self.visit_store.enabled,
            "s3_enabled": self.s3.enabled,
            "s3_bucket": self.s3.bucket or None,
            "memory_visits_cached": len(self._visits),
        }

    def ensure_persistence(self) -> dict[str, Any]:
        status = self.persistence_status()
        if self.visit_store.enabled:
            status["schema"] = self.visit_store.ensure_schema()
        return status

    def run_from_audio(
        self,
        audio_path: str,
        visit_id: Optional[str] = None,
        force_follow_up: bool = False,
        evaluate_ragas: bool = False,
    ) -> dict[str, Any]:
        result = self.graph.run_from_audio(audio_path, visit_id=visit_id)
        return self._finalize_with_reports(
            result,
            force_follow_up=force_follow_up,
            evaluate_ragas=evaluate_ragas,
        )

    def run_from_conversation(
        self,
        conversation: list[dict[str, Any]],
        transcript: str = "",
        visit_id: Optional[str] = None,
        source_file: Optional[str] = None,
        force_follow_up: bool = False,
        evaluate_ragas: bool = False,
    ) -> dict[str, Any]:
        result = self.graph.run_from_conversation(
            conversation=conversation,
            transcript=transcript,
            visit_id=visit_id,
            source_file=source_file,
        )
        return self._finalize_with_reports(
            result,
            force_follow_up=force_follow_up,
            evaluate_ragas=evaluate_ragas,
        )

    def get_visit_state(self, visit_id: str) -> Optional[dict[str, Any]]:
        if visit_id in self._visits:
            return self._visits[visit_id]
        stored = self.visit_store.get_visit(visit_id)
        if stored:
            self._visits[visit_id] = stored
            return stored
        return self.graph.get_visit_state(visit_id)

    def apply_follow_up(
        self,
        visit_id: str,
        answers: Optional[list[dict[str, Any]]] = None,
        conversation: Optional[list[dict[str, Any]]] = None,
        evaluate_ragas: bool = False,
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
            for turn in new_turns:
                if "role" not in turn:
                    turn["role"] = "Doctor" if turn.get("speaker") == "Speaker 1" else "Patient"
        else:
            raise ValueError("Provide either answers or conversation for follow-up.")

        if not new_turns:
            raise ValueError("Follow-up produced no usable conversation turns.")

        old_conversation = list(prior.get("doctor_patient_conversation") or [])
        for turn in old_conversation:
            turn.setdefault("round", 0)
        for turn in new_turns:
            turn["round"] = 1
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
        if evaluate_ragas:
            finalized = self._attach_ragas(finalized)
        self._persist_visit(finalized)
        return finalized

    def _finalize_with_reports(
        self,
        result: dict[str, Any],
        force_follow_up: bool = False,
        evaluate_ragas: bool = False,
    ) -> dict[str, Any]:
        diagnoses = result.get("differential_diagnosis") or []
        top_confidence = self.followup.top_confidence(diagnoses)
        result["confidence"] = top_confidence
        result["follow_up_threshold"] = CONFIDENCE_FOLLOW_UP_THRESHOLD
        result["follow_up_iteration"] = int(result.get("follow_up_iteration") or 0)

        for turn in result.get("doctor_patient_conversation") or []:
            if isinstance(turn, dict):
                turn.setdefault("round", 0)

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
        if evaluate_ragas:
            finalized = self._attach_ragas(finalized)
        self._persist_visit(finalized)
        return finalized

    def evaluate_visit_ragas(self, visit_id: str) -> dict[str, Any]:
        state = self.get_visit_state(visit_id)
        if not state:
            raise ValueError(f"No visit found for visit_id={visit_id}")
        updated = self._attach_ragas(dict(state))
        self._persist_visit(updated)
        return {
            "visit_id": visit_id,
            "ragas": updated.get("ragas"),
            "status": updated.get("status"),
        }

    def _attach_ragas(self, result: dict[str, Any]) -> dict[str, Any]:
        try:
            result["ragas"] = sanitize_for_state(self.ragas.evaluate_result(result))
        except Exception as exc:
            result["ragas"] = {
                "evaluated": False,
                "framework": "ragas",
                "error": str(exc),
            }
        return result

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

    def _persist_visit(self, result: dict[str, Any]) -> None:
        visit_id = result.get("visit_id")
        if not visit_id:
            return

        self._visits[visit_id] = result

        try:
            self.visit_store.save_visit(result)
        except Exception as exc:
            result["persistence_error"] = f"postgres: {exc}"

        s3_meta = {}
        for artifact_type, path_key in (
            ("analysis_pdf", "report_pdf"),
            ("soap_pdf", "soap_pdf"),
        ):
            local_path = result.get(path_key)
            if not local_path:
                continue
            try:
                uploaded = self.s3.upload_file(
                    local_path,
                    visit_id=visit_id,
                    artifact_type=artifact_type,
                )
                if uploaded.get("uploaded"):
                    s3_meta[f"{artifact_type}_s3_key"] = uploaded.get("s3_key")
                    s3_meta[f"{artifact_type}_s3_url"] = uploaded.get("s3_url")
                    self.visit_store.save_artifact(
                        visit_id,
                        artifact_type,
                        local_path=local_path,
                        s3_key=uploaded.get("s3_key"),
                        s3_url=uploaded.get("s3_url"),
                    )
                else:
                    self.visit_store.save_artifact(
                        visit_id,
                        artifact_type,
                        local_path=local_path,
                    )
            except Exception as exc:
                result.setdefault("persistence_error", "")
                result["persistence_error"] = (
                    (result.get("persistence_error") or "") + f" s3:{artifact_type}:{exc}"
                ).strip()

        if s3_meta:
            result.update(s3_meta)
            self._visits[visit_id] = result
            try:
                self.visit_store.save_visit(result)
            except Exception:
                pass
