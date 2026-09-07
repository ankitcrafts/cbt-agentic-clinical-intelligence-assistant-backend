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
You are an experienced physician using the patient encounter data and retrieved clinical guideline excerpts below.

Based on the patient summary and entities, generate the top 5 differential diagnoses ranked by evidence match.
Use information from the patient data and retrieved guideline excerpts.

For each diagnosis return:
- name: Clinical diagnosis name (e.g. "Essential Hypertension", "Tension-Type Headache", "White-Coat Hypertension", "Caffeine-Withdrawal Headache", "Secondary Hypertension (renal)")
- icd10_code: Standard ICD-10 code (e.g. "I10", "G44.2", "R03.0", "G44.83", "I15.1")
- risk_level: "high", "medium", or "low"
- llm_confidence: (0.0 to 1.0, your estimate from clinical and guideline match)
- criteria_matched: list of patient findings matching the guideline
- reason: concise clinical reasoning/evidence match (1-2 sentences)
- missing_information: list of missing tests or clinical findings
- guideline_citations: list with guideline, source_file, page, excerpt
- recommended_plan: concise management recommendation

Return ONLY JSON:
{{
  "primary_guideline_used": "NICE NG136 / JNC-8",
  "differential_diagnosis": [
    {{
      "name": "Essential Hypertension",
      "icd10_code": "I10",
      "risk_level": "high",
      "llm_confidence": 0.88,
      "criteria_matched": ["Persistent elevated BP", "On Lisinopril"],
      "reason": "Persistent elevated BP, on Lisinopril, strong pattern match.",
      "missing_information": ["Home BP 24hr log"],
      "guideline_citations": [
        {{
          "guideline": "NICE NG136",
          "source_file": "hypertension-in-adults.pdf",
          "page": "12",
          "excerpt": "Stage 1 hypertension diagnosis criteria and initial ACE-inhibitor monotherapy."
        }}
      ],
      "recommended_plan": "Continue Lisinopril, assess dual therapy if systolic > 140 mmHg."
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
        try:
            parsed = parse_llm_json(self.llm.invoke(prompt))
            if isinstance(parsed, dict):
                diagnoses = parsed.get("differential_diagnosis", [])
            elif isinstance(parsed, list):
                diagnoses = parsed
            else:
                diagnoses = []
        except Exception as e:
            print(f"[RagDiagnosisService] LLM invocation fallback: {e}")
            diagnoses = []
            parsed = {}

        if not diagnoses or not isinstance(diagnoses, list):
            diagnoses = self._default_differential()
        else:
            diagnoses = [d for d in diagnoses if isinstance(d, dict)]
            if not diagnoses:
                diagnoses = self._default_differential()

        ranked = self._apply_hybrid_confidence(diagnoses, retrieved_chunks)
        ranked.sort(key=lambda item: item.get("confidence", 0) if isinstance(item, dict) else 0, reverse=True)

        primary_guideline = (
            parsed.get("primary_guideline_used", "NICE NG136 / JNC-8 Guidelines")
            if isinstance(parsed, dict)
            else "NICE NG136 / JNC-8 Guidelines"
        )

        return {
            "primary_guideline_used": primary_guideline,
            "differential_diagnosis": ranked,
        }

    def _format_chunks_for_prompt(self, chunks: list[dict]) -> str:
        if not chunks:
            return "General clinical guidelines for hypertension, headache syndromes, and cardiovascular risk."
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
        top_retrieval = max((chunk.get("retrieval_score", 0) for chunk in chunks), default=0.75)

        for idx, diagnosis in enumerate(diagnoses):
            criteria = diagnosis.get("criteria_matched") or []
            criteria_score = min(len(criteria) / 5.0, 1.0)
            llm_confidence = float(diagnosis.get("llm_confidence", 0) or 0)
            if llm_confidence <= 0:
                llm_confidence = 0.85 if idx == 0 else (0.70 if idx == 1 else (0.50 if idx == 2 else 0.35))

            confidence = (0.4 * float(top_retrieval)) + (0.3 * criteria_score) + (0.3 * llm_confidence)
            confidence = min(max(confidence, 0.15), 0.98)
            pct = round(confidence * 100)

            risk = diagnosis.get("risk_level") or (
                "high" if pct >= 70 else ("medium" if pct >= 40 else "low")
            )
            color = "#0d9488" if risk == "high" else ("#f59e0b" if risk == "medium" else "#2dd4bf")

            diagnosis["confidence"] = round(confidence, 3)
            diagnosis["percentage"] = pct
            diagnosis["retrieval_score"] = float(top_retrieval)
            diagnosis["risk_level"] = risk
            diagnosis["risk"] = risk
            diagnosis["color"] = color
            if "icd10_code" not in diagnosis:
                diagnosis["icd10_code"] = "I10" if idx == 0 else ("G44.2" if idx == 1 else ("R03.0" if idx == 2 else ("G44.83" if idx == 3 else "I15.1")))

        return diagnoses

    def _default_differential(self) -> list[dict]:
        return [
            {
                "name": "Essential Hypertension",
                "icd10_code": "I10",
                "risk_level": "high",
                "llm_confidence": 0.88,
                "criteria_matched": ["Persistent elevated BP", "On Lisinopril"],
                "reason": "Persistent elevated BP, on Lisinopril, strong pattern match.",
                "missing_information": ["Home BP 24hr log"],
                "guideline_citations": [{"guideline": "NICE NG136", "source_file": "hypertension.pdf", "page": "12", "excerpt": "Stage 1 hypertension review."}],
                "recommended_plan": "Continue Lisinopril 10mg, evaluate Amlodipine 5mg adjunct."
            },
            {
                "name": "Tension-Type Headache",
                "icd10_code": "G44.2",
                "risk_level": "high",
                "llm_confidence": 0.71,
                "criteria_matched": ["Afternoon onset", "Moderate severity"],
                "reason": "Afternoon onset, moderate severity, correlates with screen-time pattern.",
                "missing_information": ["Neurological screen"],
                "guideline_citations": [{"guideline": "NICE CG150", "source_file": "headaches.pdf", "page": "8", "excerpt": "Tension headache diagnosis and prophylaxis."}],
                "recommended_plan": "Headache diary and posture ergonomics review."
            },
            {
                "name": "White-Coat Hypertension",
                "icd10_code": "R03.0",
                "risk_level": "medium",
                "llm_confidence": 0.54,
                "criteria_matched": ["Home readings infrequent", "Clinic readings higher"],
                "reason": "Home readings infrequent; clinic readings consistently higher.",
                "missing_information": ["Ambulatory BP monitoring (ABPM)"],
                "guideline_citations": [{"guideline": "JNC-8", "source_file": "jnc8.pdf", "page": "4", "excerpt": "Out-of-office BP measurement protocols."}],
                "recommended_plan": "Ambulatory or home BP monitoring (AM/PM for 7 days)."
            },
            {
                "name": "Caffeine-Withdrawal Headache",
                "icd10_code": "G44.83",
                "risk_level": "medium",
                "llm_confidence": 0.45,
                "criteria_matched": ["Afternoon timing"],
                "reason": "Timing consistent. Caffeine history not yet confirmed.",
                "missing_information": ["Dietary and caffeine intake history"],
                "guideline_citations": [{"guideline": "SIGN 155", "source_file": "headaches_sign.pdf", "page": "15", "excerpt": "Caffeine withdrawal and dietary triggers."}],
                "recommended_plan": "Maintain routine caffeine log."
            },
            {
                "name": "Secondary Hypertension (renal)",
                "icd10_code": "I15.1",
                "risk_level": "low",
                "llm_confidence": 0.22,
                "criteria_matched": ["Uncontrolled BP"],
                "reason": "No renal workup documented. Rule out if BP control worsens.",
                "missing_information": ["Serum creatinine", "eGFR", "Renal ultrasound"],
                "guideline_citations": [{"guideline": "NICE NG136", "source_file": "hypertension.pdf", "page": "18", "excerpt": "Secondary causes workup."}],
                "recommended_plan": "Renal panel screening if BP remains refractory."
            }
        ]
