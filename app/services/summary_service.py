from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class SummaryService:
    def __init__(self):
        self.llm = LLMService()

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

    def reframe_soap_section(self, section: str, original_content: str, edit_reason: str) -> str:
        """
        Reframes a specific SOAP section based on the doctor's additional reasoning or instructions.
        """
        prompt = f"""
You are an expert clinical medical documentation editor.

A doctor wants to refine/reframe the following section of a SOAP report.
Target Section: {section.upper()}

Original Content:
{original_content}

Doctor's Additional Reason / Clinical Instruction:
{edit_reason}

Instructions:
1. Re-write and refine the {section.upper()} section to incorporate the doctor's instruction precisely.
2. Keep professional clinical terminology (or ICD-10 formatting if relevant).
3. Do not add conversational intro/outro text. Return ONLY the reframed section text.
"""
        try:
            response = self.llm.invoke(prompt)
            return response.strip().strip('"')
        except Exception as e:
            print(f"[Warning] Failed to reframe SOAP section: {e}")
            return f"{original_content}\n\n[Doctor Note]: {edit_reason}"

    def generate_patient_summary(self, soap_note: dict) -> str:
        """
        Translates a technical SOAP report into clear, patient-friendly guidance that the doctor
        can verbally explain or print for the patient.
        """
        prompt = f"""
You are a helpful clinical communicator translating a technical medical SOAP note into clear, empathetic patient-facing advice.

SOAP Note:
- Subjective: {soap_note.get('subjective', '')}
- Objective: {soap_note.get('objective', '')}
- Assessment: {soap_note.get('assessment', '')}
- Plan: {soap_note.get('plan', '')}

Write a 3-4 bullet point summary in plain, easy-to-understand language explaining:
1. What was discussed today (Key takeaways)
2. What the assessment/diagnosis means for the patient
3. Clear action steps for the patient (Medications, lifestyle, follow-up)

Do not use overly complex jargon. Use bullet points and warm tone.
"""
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"[Warning] Failed to generate patient summary: {e}")
            return (
                "• Today we reviewed your current symptoms and health history.\n"
                "• Your current treatment plan is active and should be continued as discussed.\n"
                "• Please contact the clinic if symptoms change or worsen."
            )

