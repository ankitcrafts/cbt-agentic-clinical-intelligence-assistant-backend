from app.core.tracing import traceable
from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class RagDiagnosisService:
    def __init__(self):
        self.llm = LLMService()

    @traceable(name="rag_rank_diagnoses", run_type="chain")
    def rank_diagnoses(self, summary, entities, retrieved_chunks):
        context = self._format_chunks_for_prompt(retrieved_chunks)
        prompt = f"""
You are an experienced physician using ONLY the retrieved clinical guideline excerpts below.

Based on the patient summary and entities, generate the top 5 differential diagnoses ranked by evidence match.
Use ONLY information from the patient data and retrieved guideline excerpts.
Do not invent findings.

For each diagnosis return:
- name
- llm_confidence (0.0 to 1.0, your estimate from guideline match)
- criteria_matched (list of patient findings that match the guideline)
- reason
- missing_information
- guideline_citations (list with guideline, source_file, page, excerpt)

Return ONLY JSON:
{{
  "primary_guideline_used": "",
  "differential_diagnosis": [
    {{
      "name": "",
      "llm_confidence": 0.0,
      "criteria_matched": [],
      "reason": "",
      "missing_information": [],
      "guideline_citations": [
        {{
          "guideline": "",
          "source_file": "",
          "page": "",
          "excerpt": ""
        }}
      ],
      "recommended_plan": ""
    }}
  ]
}}

Patient summary:
{summary}

Patient entities:
{entities}

Retrieved guideline excerpts:
{context}
"""
        parsed = parse_llm_json(self.llm.invoke(prompt))
        diagnoses = parsed.get("differential_diagnosis", [])
        ranked = self._apply_hybrid_confidence(diagnoses, retrieved_chunks)
        ranked.sort(key=lambda item: item.get("confidence", 0), reverse=True)

        return {
            "primary_guideline_used": parsed.get("primary_guideline_used", ""),
            "differential_diagnosis": ranked,
        }

    def _format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        sections = []
        for index, chunk in enumerate(chunks, start=1):
            sections.append(
                "\n".join(
                    [
                        f"[{index}] {chunk.get('guideline', '')} | {chunk.get('source_file', '')} | page {chunk.get('page', '')}",
                        chunk.get("content", ""),
                    ]
                )
            )
        return "\n\n".join(sections)

    def _apply_hybrid_confidence(self, diagnoses: list[dict], chunks: list[dict]) -> list[dict]:
        top_retrieval = max((chunk.get("retrieval_score", 0) for chunk in chunks), default=0.0)

        for diagnosis in diagnoses:
            criteria = diagnosis.get("criteria_matched") or []
            criteria_score = min(len(criteria) / 5.0, 1.0)
            llm_confidence = float(diagnosis.get("llm_confidence", 0) or 0)
            confidence = (0.5 * float(top_retrieval)) + (0.3 * criteria_score) + (0.2 * llm_confidence)
            diagnosis["confidence"] = float(round(confidence, 3))
            diagnosis["retrieval_score"] = float(top_retrieval)

        return diagnoses
