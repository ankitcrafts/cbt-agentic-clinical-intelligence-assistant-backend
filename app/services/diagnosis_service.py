from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class DiagnosisService:

    def __init__(self):

        self.llm = LLMService("gemini")

    def differential_diagnosis(
        self,
        summary,
        entities
    ):

        prompt = f"""

You are an experienced physician.

Based ONLY on the structured patient information below,
generate the top 5 differential diagnoses.

Do not invent findings that are not in the summary or entities.
confidence must be a number between 0.0 and 1.0.

Return ONLY JSON:

{{
"differential_diagnosis":[
{{
"name":"",
"confidence":0.0,
"reason":"",
"missing_information":[]
}}
]
}}

Summary

{summary}

Entities

{entities}

"""

        return parse_llm_json(self.llm.invoke(prompt))