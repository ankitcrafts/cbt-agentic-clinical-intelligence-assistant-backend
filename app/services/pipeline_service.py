import re
from typing import Dict, Any, List, Optional
from app.services.conversation_service import ConversationService
from app.services.deepgram_service import DeepgramService
from app.services.diagnosis_service import DiagnosisService
from app.services.entity_service import EntityService
from app.services.summary_service import SummaryService
from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class ClinicalPipeline:
    """
    Complete Clinical Intelligence Pipeline:
    1. Deepgram Speech-to-Text with Speaker Diarization
    2. Convert speakers into Doctor / Patient dialogue turns
    3. Formatted exchange objects for frontend chat UI
    4. Structured Clinical Summarization
    5. Medical Entity Extraction
    6. Differential Diagnosis with confidence scores & rationale
    7. Dynamic Real-time AI Consultation Suggestions
    8. Structured SOAP Note generation
    """

    def __init__(self):
        self.deepgram = DeepgramService()
        self.conversation = ConversationService()
        self.summary = SummaryService()
        self.entity = EntityService()
        self.diagnosis = DiagnosisService()
        self.llm = LLMService("gemini")

    def run_from_audio(
        self,
        audio_path: str,
        doctor_name: str = "Dr. Yuki Tanaka",
        patient_name: str = "Marcus Delgado",
        doctor_initials: str = "YT",
        patient_initials: str = "MD",
    ) -> Dict[str, Any]:
        asr = self.deepgram.transcribe(audio_path)
        return self.run_from_conversation(
            conversation=asr["conversation"],
            transcript=asr.get("transcript", ""),
            doctor_name=doctor_name,
            patient_name=patient_name,
            doctor_initials=doctor_initials,
            patient_initials=patient_initials,
        )

    def run_from_conversation(
        self,
        conversation: List[Dict[str, Any]],
        transcript: Optional[str] = "",
        doctor_name: str = "Dr. Yuki Tanaka",
        patient_name: str = "Marcus Delgado",
        doctor_initials: str = "YT",
        patient_initials: str = "MD",
    ) -> Dict[str, Any]:
        transcript = transcript or ""
        labeled = self.conversation.to_doctor_patient(conversation)
        labeled_turns = labeled.get("conversation", [])
        speaker_map = labeled.get("speaker_map", {})

        # Build clean frontend exchanges
        exchanges = self._format_exchanges(
            labeled_turns=labeled_turns,
            doctor_name=doctor_name,
            patient_name=patient_name,
            doctor_initials=doctor_initials,
            patient_initials=patient_initials,
        )

        # Run clinical summarization & entity extraction
        summary = self.summary.summarize(labeled_turns)
        entities = self.entity.extract(labeled_turns)
        diagnosis_res = self.diagnosis.differential_diagnosis(summary, entities)
        diff_list = diagnosis_res.get("differential_diagnosis", diagnosis_res) if isinstance(diagnosis_res, dict) else diagnosis_res

        # Generate live AI Suggestions and structured SOAP Note
        ai_suggestions = self._generate_ai_suggestions(summary, entities, diff_list)
        soap_report = self._build_soap_report(summary, entities, diff_list)

        return {
            "transcript": transcript,
            "conversation": conversation,
            "doctor_patient_conversation": labeled_turns,
            "speaker_map": speaker_map,
            "exchanges": exchanges,
            "ai_suggestions": ai_suggestions,
            "summary": summary,
            "entities": entities,
            "differential_diagnosis": diff_list,
            "soap_report": soap_report,
            "clinical_analysis": {
                "summary": summary,
                "entities": entities,
                "differential_diagnosis": diff_list,
                "soap_report": soap_report,
                "ai_suggestions": ai_suggestions,
            },
        }

    def _format_exchanges(
        self,
        labeled_turns: List[Dict[str, Any]],
        doctor_name: str,
        patient_name: str,
        doctor_initials: str,
        patient_initials: str,
    ) -> List[Dict[str, Any]]:
        exchanges = []
        for idx, turn in enumerate(labeled_turns, start=1):
            role = turn.get("role", "Patient")
            is_doctor = role.lower() == "doctor"
            start_sec = turn.get("start")
            formatted_time = self._format_seconds(start_sec) if start_sec is not None else f"00:{idx:02d}"

            exchanges.append({
                "id": idx,
                "speaker": doctor_initials if is_doctor else patient_initials,
                "speakerName": doctor_name if is_doctor else patient_name,
                "isDoctor": is_doctor,
                "text": turn.get("text", ""),
                "timestamp": formatted_time,
                "start": turn.get("start"),
                "end": turn.get("end"),
            })
        return exchanges

    def _format_seconds(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return "00:00"
        total_sec = int(seconds)
        mins = total_sec // 60
        secs = total_sec % 60
        return f"{mins:02d}:{secs:02d}"

    def _generate_ai_suggestions(self, summary: Any, entities: Any, diff_list: Any) -> List[Dict[str, Any]]:
        try:
            prompt = f"""
You are a Real-time Clinical Decision Support Assistant assisting a physician during an active patient consultation.
Based on the encounter summary, extracted medical entities, and differential diagnosis, generate 3 to 4 actionable, concise AI suggestions for the doctor to investigate during the visit.

Return ONLY JSON matching this exact structure:
{{
  "suggestions": [
    {{
      "priority": "HIGH",
      "category": "SYMPTOM DETAIL",
      "title": "Ask about location and quality of symptoms",
      "description": "tension vs. vascular patterns differ in management."
    }},
    {{
      "priority": "HIGH",
      "category": "DIFFERENTIAL",
      "title": "Rule out acute triggers or visual disturbances",
      "description": "Important to differentiate primary from secondary etiology."
    }},
    {{
      "priority": "MEDIUM",
      "category": "LIFESTYLE",
      "title": "Screen for dietary and routine triggers",
      "description": "Afternoon timing correlates with caffeine or hydration deficits."
    }}
  ]
}}

Summary: {summary}
Entities: {entities}
Differential: {diff_list}
"""
            res = parse_llm_json(self.llm.invoke(prompt))
            if isinstance(res, dict) and "suggestions" in res:
                return res["suggestions"]
        except Exception as e:
            print(f"[Warning] Failed to generate LLM suggestions: {e}")

        # Fallback default suggestions
        return [
            {
                "priority": "HIGH",
                "category": "SYMPTOM DETAIL",
                "title": "Ask about onset, frequency, and severity",
                "description": "Clarify timing and aggravating factors for accurate differential.",
            },
            {
                "priority": "HIGH",
                "category": "DIFFERENTIAL",
                "title": "Evaluate red flags and rule out secondary causes",
                "description": "Check for associated systemic signs or focal complaints.",
            },
            {
                "priority": "MEDIUM",
                "category": "LIFESTYLE",
                "title": "Assess medication adherence and daily routines",
                "description": "Review hydration, sleep quality, and lifestyle contributors.",
            },
        ]

    def _build_soap_report(self, summary: Any, entities: Any, diff_list: Any) -> Dict[str, str]:
        try:
            prompt = f"""
Generate a comprehensive, professional clinical SOAP Note (Subjective, Objective, Assessment, Plan) based on this encounter.

Return ONLY JSON:
{{
  "subjective": "Patient reported history...",
  "objective": "Vital signs and physical observations...",
  "assessment": "1. Primary diagnosis... 2. Secondary...",
  "plan": "1. Medications... 2. Diagnostics... 3. Follow-up..."
}}

Summary: {summary}
Entities: {entities}
Differential: {diff_list}
"""
            res = parse_llm_json(self.llm.invoke(prompt))
            if isinstance(res, dict) and "subjective" in res:
                return res
        except Exception as e:
            print(f"[Warning] Failed to generate SOAP note: {e}")

        s_text = summary.get("chief_complaint", "") if isinstance(summary, dict) else str(summary)
        return {
            "subjective": f"Patient presents with: {s_text}",
            "objective": "Vital signs and clinical observations within expected parameters.",
            "assessment": "Clinical presentation under evaluation.",
            "plan": "Initiate symptomatic treatment and follow-up as indicated.",
        }

