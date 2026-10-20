from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class EntityService:
    def __init__(self):
        self.llm = LLMService()

    def extract(self, conversation):
        prompt = f"""
Extract medical entities from this doctor-patient conversation.
Use only information stated in the conversation.

Return ONLY JSON:
{{
  "symptoms": [],
  "duration": "",
  "medical_history": [],
  "medications": [],
  "tests": [],
  "allergies": [],
  "vital_signs": {{}},
  "warning_signs": [],
  "possible_conditions": []
}}

Conversation:
{conversation}
"""
        return parse_llm_json(self.llm.invoke(prompt))
