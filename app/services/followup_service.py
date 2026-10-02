from typing import Any, Optional

from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService

CONFIDENCE_FOLLOW_UP_THRESHOLD = 0.70
MAX_FOLLOW_UP_ROUNDS = 1


class FollowUpService:
    """Generate one-round follow-up questions and merge prior summary with new answers."""

    def __init__(self):
        self.llm = LLMService()

    @staticmethod
    def top_confidence(diagnoses: list[dict]) -> float:
        if not diagnoses:
            return 0.0
        values = []
        for item in diagnoses:
            try:
                values.append(float(item.get("confidence") or 0))
            except (TypeError, ValueError):
                values.append(0.0)
        return max(values) if values else 0.0

    def needs_follow_up(
        self,
        diagnoses: list[dict],
        follow_up_iteration: int = 0,
        force: bool = False,
    ) -> bool:
        if follow_up_iteration >= MAX_FOLLOW_UP_ROUNDS:
            return False
        if force:
            return True
        return self.top_confidence(diagnoses) < CONFIDENCE_FOLLOW_UP_THRESHOLD

    def generate_questions(
        self,
        summary: dict,
        entities: dict,
        diagnoses: list[dict],
        max_questions: int = 3,
    ) -> list[dict[str, Any]]:
        top = diagnoses[:3] if diagnoses else []
        missing = []
        for dx in top:
            for item in dx.get("missing_information") or []:
                if item and item not in missing:
                    missing.append(item)

        prompt = f"""
You are a clinical assistant preparing follow-up questions for a doctor-patient encounter.

Confidence is below threshold, so ask concise questions that would most improve diagnosis certainty.
Use ONLY gaps supported by the summary, entities, missing_information, and differentials.
Do not invent unrelated topics.
Ask at most {max_questions} questions.
Prefer short, clinically useful questions a patient can answer verbally.

Return ONLY JSON:
{{
  "follow_up_questions": [
    {{
      "id": "q1",
      "question": "",
      "reason": "",
      "related_diagnosis": ""
    }}
  ]
}}

Patient summary:
{summary}

Entities:
{entities}

Top differentials:
{top}

Known missing information:
{missing}
"""
        parsed = parse_llm_json(self.llm.invoke(prompt))
        questions = parsed.get("follow_up_questions") or []
        cleaned = []
        for index, item in enumerate(questions[:max_questions], start=1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            cleaned.append(
                {
                    "id": str(item.get("id") or f"q{index}"),
                    "question": question,
                    "reason": str(item.get("reason") or "").strip(),
                    "related_diagnosis": str(item.get("related_diagnosis") or "").strip(),
                }
            )
        return cleaned

    def answers_to_conversation(
        self,
        questions: list[dict[str, Any]],
        answers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Turn [{question_id, answer}] into doctor/patient style turns."""
        by_id = {str(q.get("id")): q for q in questions if q.get("id")}
        turns = []
        for index, item in enumerate(answers, start=1):
            if not isinstance(item, dict):
                continue
            qid = str(item.get("question_id") or item.get("id") or "")
            answer = str(item.get("answer") or item.get("text") or "").strip()
            if not answer:
                continue
            question = by_id.get(qid, {}).get("question") or f"Follow-up question {index}"
            turns.append(
                {
                    "speaker": "Speaker 1",
                    "role": "Doctor",
                    "text": question,
                }
            )
            turns.append(
                {
                    "speaker": "Speaker 0",
                    "role": "Patient",
                    "text": answer,
                }
            )
        return turns

    def merge_summary(self, previous_summary: dict, new_conversation: list[dict]) -> dict:
        prompt = f"""
You are updating a clinical encounter summary after a follow-up conversation.

Keep all important facts from the PREVIOUS summary.
Add or correct facts from the NEW follow-up dialogue.
Do not invent findings.
If follow-up answers rule something out, reflect that clearly.

Return ONLY JSON:
{{
  "chief_complaint": "",
  "history_of_present_illness": "",
  "associated_symptoms": [],
  "relevant_negatives": [],
  "assessment": "",
  "plan": "",
  "summary": ""
}}

PREVIOUS summary:
{previous_summary}

NEW follow-up conversation:
{new_conversation}
"""
        return parse_llm_json(self.llm.invoke(prompt))
