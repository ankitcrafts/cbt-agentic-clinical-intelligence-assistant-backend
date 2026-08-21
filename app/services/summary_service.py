from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class SummaryService:
    def __init__(self):
        self.llm = LLMService("gemini")

    def summarize(self, conversation):
        prompt = f"""
You are a clinical documentation assistant.

Write a structured summary of this doctor-patient conversation.
Do not invent facts that are not in the conversation.

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

Conversation:
{conversation}
"""
        return parse_llm_json(self.llm.invoke(prompt))
