from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class ConversationService:
    def __init__(self):
        self.llm = LLMService()

    def to_doctor_patient(self, conversation):
        prompt = f"""
You are labeling speakers in a clinical encounter transcript.

Map each speaker (for example "Speaker 0", "Speaker 1") to exactly one role:
- Doctor
- Patient

Use the dialogue content. The speaker who asks clinical questions, examines, explains diagnosis, or proposes tests/treatment is the Doctor. The speaker who reports symptoms, history, and concerns is the Patient.

Return ONLY JSON:
{{
  "speaker_map": {{
    "Speaker 0": "Patient",
    "Speaker 1": "Doctor"
  }}
}}

Conversation:
{conversation}
"""
        mapping = parse_llm_json(self.llm.invoke(prompt))
        speaker_map = mapping.get("speaker_map") or {}

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

    def _fallback_role(self, text: str) -> str:
        lowered = (text or "").lower()
        doctor_cues = (
            "what's been troubling",
            "have you noticed",
            "i'd like to",
            "urine test",
            "consistent with",
            "please tell me",
        )
        if any(cue in lowered for cue in doctor_cues):
            return "Doctor"
        return "Patient"
