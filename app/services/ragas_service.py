"""Optional RAGAS evaluation for clinical RAG outputs.

Returns JSON-friendly metric scores. Requires LLM (OpenRouter/etc.) and
runs only when explicitly requested to avoid extra latency/cost.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from app.services.llm_service import LLMService


class RagasEvaluationService:
    """Score faithfulness / relevancy / context precision for one visit result."""

    def __init__(self):
        self.llm_service = LLMService()
        self._evaluator_llm = None
        self._evaluator_embeddings = None

    def _get_llm_wrapper(self):
        if self._evaluator_llm is not None:
            return self._evaluator_llm
        if self.llm_service.llm is None:
            raise RuntimeError("LLM is not configured for RAGAS evaluation.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.llms import LangchainLLMWrapper

            self._evaluator_llm = LangchainLLMWrapper(self.llm_service.llm)
        return self._evaluator_llm

    def _get_embeddings_wrapper(self):
        if self._evaluator_embeddings is not None:
            return self._evaluator_embeddings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self._evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
        return self._evaluator_embeddings

    @staticmethod
    def build_user_input(result: dict[str, Any]) -> str:
        query = (result.get("retrieval_query") or "").strip()
        if query:
            return query
        summary = result.get("summary") or {}
        entities = result.get("entities") or {}
        parts = [
            f"Chief complaint: {summary.get('chief_complaint', '')}",
            f"Assessment: {summary.get('assessment', '')}",
            f"Symptoms: {', '.join(entities.get('symptoms') or [])}",
            f"Possible conditions: {', '.join(entities.get('possible_conditions') or [])}",
        ]
        return "\n".join(p for p in parts if p.split(":", 1)[-1].strip())

    @staticmethod
    def build_response(result: dict[str, Any]) -> str:
        diagnoses = result.get("differential_diagnosis") or []
        if not diagnoses:
            soap = result.get("soap") or {}
            return str(soap.get("assessment") or soap.get("plan") or "No diagnosis available.")
        lines = []
        for index, dx in enumerate(diagnoses[:3], start=1):
            name = dx.get("name") or f"Diagnosis {index}"
            conf = dx.get("confidence")
            reason = dx.get("reason") or ""
            plan = dx.get("recommended_plan") or ""
            conf_txt = f" (confidence={conf})" if conf is not None else ""
            lines.append(f"{index}. {name}{conf_txt}. {reason} Plan: {plan}".strip())
        guideline = result.get("primary_guideline_used") or ""
        if guideline:
            lines.append(f"Primary guideline used: {guideline}")
        return "\n".join(lines)

    @staticmethod
    def build_contexts(result: dict[str, Any], max_chunks: int = 6) -> list[str]:
        contexts = []
        for chunk in (result.get("retrieved_chunks") or [])[:max_chunks]:
            if not isinstance(chunk, dict):
                continue
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            source = chunk.get("source_file") or chunk.get("guideline") or ""
            page = chunk.get("page")
            header = f"[{source} p.{page}] " if source else ""
            contexts.append((header + content)[:1800])
        return contexts

    def evaluate_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Run RAGAS metrics and return a JSON-serializable dict.

        Metrics (no ground-truth labels required):
        - faithfulness
        - answer_relevancy (ResponseRelevancy)
        - context_precision (LLMContextPrecisionWithoutReference)
        """
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except Exception:
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas import SingleTurnSample
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                ResponseRelevancy,
            )

        user_input = self.build_user_input(result)
        response = self.build_response(result)
        contexts = self.build_contexts(result)

        if not user_input or not response or not contexts:
            return {
                "evaluated": False,
                "error": "Missing user_input, response, or retrieved_contexts for RAGAS.",
                "user_input": user_input,
                "response_preview": response[:240],
                "context_count": len(contexts),
            }

        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=contexts,
        )
        llm = self._get_llm_wrapper()
        embeddings = self._get_embeddings_wrapper()

        scores: dict[str, Any] = {}
        errors: dict[str, str] = {}

        metric_runners = [
            ("faithfulness", Faithfulness(llm=llm)),
            (
                "answer_relevancy",
                ResponseRelevancy(llm=llm, embeddings=embeddings),
            ),
            (
                "context_precision",
                LLMContextPrecisionWithoutReference(llm=llm),
            ),
        ]

        for name, metric in metric_runners:
            try:
                score = metric.single_turn_score(sample)
                scores[name] = round(float(score), 4) if score is not None else None
            except Exception as exc:
                scores[name] = None
                errors[name] = str(exc)

        numeric = [v for v in scores.values() if isinstance(v, (int, float))]
        overall = round(sum(numeric) / len(numeric), 4) if numeric else None

        payload = {
            "evaluated": True,
            "framework": "ragas",
            "metrics": scores,
            "overall_score": overall,
            "inputs": {
                "user_input_preview": user_input[:300],
                "response_preview": response[:300],
                "context_count": len(contexts),
            },
        }
        if errors:
            payload["metric_errors"] = errors
        return payload
