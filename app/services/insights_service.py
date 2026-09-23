from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class InsightsService:
    """Generate sample-style analysis sections for the UI PDF report."""

    def __init__(self):
        self.llm = LLMService()

    def build_insights(self, conversation, summary, entities, diagnoses) -> dict:
        prompt = f"""
You are a clinical documentation assistant.

From the doctor-patient conversation and analysis below, create a detailed report insights object.
Use ONLY information supported by the conversation. Do not invent findings.
Keep lists concise but clinically useful (typically 4-10 items each).
This must work for ANY disease (UTI, ear infection, diabetes, hypertension, etc.).

Return ONLY JSON:
{{
  "key_points": [],
  "action_items": [],
  "patient_concerns": [],
  "medical_terms": [],
  "follow_up_questions": [],
  "symptoms": [],
  "medications": [],
  "diagnosis_considerations": [],
  "risk_factors": [],
  "differential_diagnosis_list": [],
  "soap": {{
    "subjective": "",
    "objective": "Not documented",
    "assessment": "",
    "plan": ""
  }}
}}

Rules:
- key_points: important clinical facts from the encounter
- action_items: next steps for clinician/patient
- patient_concerns: what the patient is worried about or asking
- medical_terms: relevant clinical terms (include brief expansion in parentheses when useful)
- follow_up_questions: questions a clinician may still ask
- symptoms / medications: from conversation
- diagnosis_considerations: possible conditions to consider
- risk_factors: relevant risk factors mentioned or implied
- differential_diagnosis_list: short names of differentials (can align with ranked diagnoses)
- soap.subjective: patient-reported history in 1-3 sentences
- soap.objective: exam/vitals if present, otherwise "Not documented"
- soap.assessment: clinical impression in 1-2 sentences
- soap.plan: plan if present, otherwise "Not documented"

Conversation:
{conversation}

Summary:
{summary}

Entities:
{entities}

Ranked diagnoses:
{diagnoses}
"""
        parsed = parse_llm_json(self.llm.invoke(prompt))
        return self._normalize(parsed, summary, entities, diagnoses)

    def _normalize(self, parsed: dict, summary, entities, diagnoses) -> dict:
        data = parsed if isinstance(parsed, dict) else {}
        soap = data.get("soap") if isinstance(data.get("soap"), dict) else {}

        summary = summary if isinstance(summary, dict) else {}
        entities = entities if isinstance(entities, dict) else {}
        diagnoses = diagnoses if isinstance(diagnoses, list) else []

        if not soap.get("subjective"):
            soap["subjective"] = (
                summary.get("history_of_present_illness")
                or summary.get("chief_complaint")
                or summary.get("summary")
                or "Not documented"
            )
        if not soap.get("objective"):
            soap["objective"] = "Not documented"
        if not soap.get("assessment"):
            soap["assessment"] = summary.get("assessment") or "Not documented"
        if not soap.get("plan"):
            plan = summary.get("plan")
            if not plan and diagnoses:
                plan = diagnoses[0].get("recommended_plan")
            soap["plan"] = plan or "Not documented"

        symptoms = data.get("symptoms") or entities.get("symptoms") or []
        medications = data.get("medications") or entities.get("medications") or []
        dx_list = data.get("differential_diagnosis_list") or [
            d.get("name") for d in diagnoses if d.get("name")
        ]

        return {
            "key_points": data.get("key_points") or [],
            "action_items": data.get("action_items") or [],
            "patient_concerns": data.get("patient_concerns") or [],
            "medical_terms": data.get("medical_terms") or [],
            "follow_up_questions": data.get("follow_up_questions") or [],
            "symptoms": symptoms,
            "medications": medications,
            "diagnosis_considerations": data.get("diagnosis_considerations") or dx_list,
            "risk_factors": data.get("risk_factors") or [],
            "differential_diagnosis_list": dx_list,
            "soap": {
                "subjective": soap.get("subjective") or "Not documented",
                "objective": soap.get("objective") or "Not documented",
                "assessment": soap.get("assessment") or "Not documented",
                "plan": soap.get("plan") or "Not documented",
            },
        }
