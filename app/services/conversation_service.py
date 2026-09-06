from typing import List, Dict, Any
from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class ConversationService:
    def __init__(self):
        self.llm = LLMService()

    def to_doctor_patient(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        speaker_map = {}
        try:
            prompt = f"""
You are labeling speakers in a clinical encounter transcript.

Map each speaker (for example "Speaker 0", "Speaker 1") to exactly one role:
- Doctor
- Patient

Use the dialogue content. The speaker who asks clinical questions, examines, explains diagnosis, or proposes tests/treatment is the Doctor. The speaker who reports symptoms, history, and concerns is the Patient.

Return ONLY JSON:
{{
  "speaker_map": {{
    "Speaker 0": "Doctor",
    "Speaker 1": "Patient"
  }}
}}

Conversation:
{conversation}
"""
            mapping = parse_llm_json(self.llm.invoke(prompt))
            if isinstance(mapping, dict) and "speaker_map" in mapping:
                speaker_map = mapping.get("speaker_map") or {}
        except Exception as e:
            print(f"[Warning] LLM speaker labeling fallback used: {e}")

        # Ensure all speakers have a valid mapping using deterministic cues
        if not speaker_map:
            speaker_map = self._heuristic_speaker_map(conversation)

        labeled = []
        for turn in conversation:
            speaker = turn.get("speaker", "")
            role = speaker_map.get(speaker, self._fallback_role(turn.get("text", "")))
            labeled.append(
                {
                    "role": role,
                    "speaker": speaker,
                    "start": turn.get("start"),
                    "end": turn.get("end"),
                    "text": turn.get("text", ""),
                }
            )

        return {
            "speaker_map": speaker_map,
            "conversation": labeled,
        }

    def _heuristic_speaker_map(self, conversation: List[Dict[str, Any]]) -> Dict[str, str]:
        speaker_map = {}
        doctor_score = {}
        patient_score = {}

        for turn in conversation:
            spk = turn.get("speaker", "Speaker 0")
            text = (turn.get("text") or "").lower()
            if spk not in doctor_score:
                doctor_score[spk] = 0
                patient_score[spk] = 0

            # Patient cues (addressing doctor, reporting personal symptoms)
            if any(p in text for p in [
                "hi doctor", "hello doctor", "yes doctor", "thanks doctor", "thank you doctor",
                "i have been having", "i have", "my ear", "my left", "my throat", "my chest",
                "i noticed", "i had a cold", "i went swimming", "i thought", "for the past few days",
                "for the last few days", "worse at night"
            ]):
                patient_score[spk] += 3

            # Doctor cues (consultation questions, empathy, clinical examination, diagnosis)
            if any(d in text for d in [
                "how are you", "what is the issue", "what brings you", "have you noticed",
                "i'm sorry to hear", "i am sorry to hear", "sounds like you have", "otitis",
                "inflammation", "fluid buildup", "treated", "let me check", "prescribe",
                "let me examine", "take care", "contributing factor", "ear canal", "recent cold", "been swimming"
            ]):
                doctor_score[spk] += 3

        for spk in doctor_score:
            if doctor_score[spk] > patient_score[spk]:
                speaker_map[spk] = "Doctor"
            else:
                speaker_map[spk] = "Patient"

        # If all mapped to Patient or Speaker 0 started with a question/greeting, ensure Speaker 0 is Doctor
        first_text = (conversation[0].get("text") or "").lower() if conversation else ""
        if any(g in first_text for g in ["how are you", "what is the issue", "what brings you", "hello", "hi,"]):
            first_spk = conversation[0].get("speaker", "Speaker 0")
            speaker_map[first_spk] = "Doctor"

        return speaker_map

    def _fallback_role(self, text: str) -> str:
        lowered = (text or "").lower()
        if any(p in lowered for p in ["hi doctor", "hello doctor", "i have been having", "my left ear", "i had a cold", "i went swimming", "i noticed"]):
            return "Patient"
        if any(d in lowered for d in ["how are you", "what is the issue", "sorry to hear", "have you noticed", "sounds like you have", "inflammation"]):
            return "Doctor"
        return "Patient"
