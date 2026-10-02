import re
from typing import Any, Dict, List, Optional
from app.services.json_utils import parse_llm_json
from app.services.llm_service import LLMService


class ConversationService:
    def __init__(self):
        self.llm = LLMService()

    def to_doctor_patient(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not conversation:
            return {"speaker_map": {}, "conversation": []}

        # Step 1: Pre-process and split merged turns (e.g. physician question + patient immediate response)
        preprocessed = self._split_and_clean_turns(conversation)

        # Step 2: Try LLM-based speaker role classification
        speaker_map = {}
        try:
            prompt = f"""
You are an expert clinical speech-to-text analyzer classifying speaker roles in a patient consultation transcript.

Map each speaker (e.g., "Speaker 0", "Speaker 1") to exactly one role:
- "Doctor" (The clinician who conducts the consultation, asks clinical history, reviews vitals, examines, diagnoses, or explains treatment)
- "Patient" (The individual describing their symptoms, medical history, home readings, feelings, and concerns)

In a 2-person clinical consultation, exactly ONE speaker MUST be "Doctor" and the other MUST be "Patient".

Return ONLY JSON:
{{
  "speaker_map": {{
    "Speaker 0": "Doctor",
    "Speaker 1": "Patient"
  }}
}}

Conversation transcript:
{preprocessed}
"""
            mapping = parse_llm_json(self.llm.invoke(prompt))
            if isinstance(mapping, dict) and "speaker_map" in mapping:
                candidate_map = mapping.get("speaker_map") or {}
                # Validate that not all speakers are mapped to the same role in multi-speaker dialogue
                roles = set(candidate_map.values())
                if len(candidate_map) >= 2 and len(roles) >= 2:
                    speaker_map = candidate_map
        except Exception as e:
            print(f"[ConversationService] LLM speaker labeling notice: {e}")

        # Step 3: Heuristic speaker mapping if LLM failed or mapped all to same role
        if not speaker_map or len(set(speaker_map.values())) <= 1:
            speaker_map = self._heuristic_speaker_map(preprocessed)

        # Step 4: Turn-level clinical refinement
        labeled = self._apply_turn_level_refinement(preprocessed, speaker_map)

        # Step 5: Merge consecutive dialogue turns from the same speaker
        merged_conversation = self._merge_consecutive_turns(labeled)

        return {
            "speaker_map": speaker_map,
            "conversation": merged_conversation,
        }

    def _split_and_clean_turns(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        refined = []
        for turn in conversation:
            text = (turn.get("text") or "").strip()
            speaker = turn.get("speaker", "Speaker 0")
            start = turn.get("start")
            end = turn.get("end")

            if not text:
                continue

            # Detect doctor question immediately followed by patient answer in a single Deepgram utterance
            parts = re.split(r'(\?)\s*(?=[A-Z][^?]*\b(?:i|my|we|it|no|yes)\b)', text, flags=re.IGNORECASE)
            if len(parts) >= 3 and parts[1] == "?":
                q_part = (parts[0] + parts[1]).strip()
                a_part = "".join(parts[2:]).strip()

                mid = ((start + end) / 2) if (start is not None and end is not None) else start

                refined.append({
                    "speaker": speaker,
                    "role": "Doctor",
                    "start": start,
                    "end": mid,
                    "text": q_part,
                })
                refined.append({
                    "speaker": "Speaker 1" if speaker == "Speaker 0" else "Speaker 0",
                    "role": "Patient",
                    "start": mid,
                    "end": end,
                    "text": a_part,
                })
            else:
                refined.append(turn)
        return refined

    def _heuristic_speaker_map(self, conversation: List[Dict[str, Any]]) -> Dict[str, str]:
        doctor_cues = [
            r"\bwhat would you like to discuss\b",
            r"\bwhat brings you\b",
            r"\bhow can i help\b",
            r"\bhow are you\b",
            r"\bwhat reading.*did you get\b",
            r"\bwhat reading\b",
            r"\bhave you had.*before\b",
            r"\bhave you been experiencing\b",
            r"\bhow often do you\b",
            r"\bhow long have you\b",
            r"\blet me check\b",
            r"\blet me examine\b",
            r"\bwhat is the issue\b",
            r"\bany other symptoms\b",
            r"\bprescrib",
            r"\bhypertension\b",
            r"\btake these\b",
            r"\bstopping it\b",
            r"\bexercise\b",
            r"\bmedication\b",
        ]
        patient_cues = [
            r"\b(good morning|hello|hi|yes|no|thanks|thank you)\s*,?\s*doctor\b",
            r"\bi came because\b",
            r"\bmy blood pressure was\b",
            r"\bmy blood pressure has been\b",
            r"\bit was around\b",
            r"\bon two different occasions\b",
            r"\bi have never been diagnosed\b",
            r"\bi sometimes get\b",
            r"\bi don't have\b",
            r"\bi have been feeling\b",
            r"\bmy (head|chest|ear|throat|back|stomach|arm|leg|blood)\b",
            r"\bi feel\b",
            r"\bat home\b",
        ]

        spk_scores: Dict[str, Dict[str, int]] = {}
        for t in conversation:
            spk = t.get("speaker", "Speaker 0")
            txt = (t.get("text") or "").lower()
            if spk not in spk_scores:
                spk_scores[spk] = {"doc": 0, "pat": 0}

            for dc in doctor_cues:
                if re.search(dc, txt):
                    spk_scores[spk]["doc"] += 3
            for pc in patient_cues:
                if re.search(pc, txt):
                    spk_scores[spk]["pat"] += 3

        speakers = list(spk_scores.keys())
        speaker_map: Dict[str, str] = {}

        if len(speakers) == 2:
            s0, s1 = speakers[0], speakers[1]
            s0_net = spk_scores[s0]["doc"] - spk_scores[s0]["pat"]
            s1_net = spk_scores[s1]["doc"] - spk_scores[s1]["pat"]

            if s0_net > s1_net:
                speaker_map[s0] = "Doctor"
                speaker_map[s1] = "Patient"
            elif s1_net > s0_net:
                speaker_map[s0] = "Patient"
                speaker_map[s1] = "Doctor"
            else:
                # Default: Speaker 0 is Doctor (initiates clinical visit), Speaker 1 is Patient
                speaker_map[s0] = "Doctor"
                speaker_map[s1] = "Patient"
        elif len(speakers) == 1:
            speaker_map[speakers[0]] = "Doctor"
        else:
            for s in speakers:
                speaker_map[s] = "Doctor" if spk_scores[s]["doc"] >= spk_scores[s]["pat"] else "Patient"

        return speaker_map

    def _apply_turn_level_refinement(
        self, conversation: List[Dict[str, Any]], speaker_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        doctor_cues = [
            r"\bwhat would you like to discuss\b",
            r"\bwhat brings you\b",
            r"\bhow can i help\b",
            r"\bwhat reading.*did you get\b",
            r"\bwhat reading\b",
            r"\bhave you had.*before\b",
            r"\bhave you been experiencing\b",
            r"\bhow often do you\b",
            r"\bhow long have you\b",
            r"\blet me check\b",
            r"\blet me examine\b",
            r"\bwhat is the issue\b",
            r"\bany other symptoms\b",
            r"\bprescrib",
            r"\btake these tablets\b",
            r"\bstopping it\b",
            r"\bexercise\b",
        ]
        patient_cues = [
            r"\b(good morning|hello|hi|yes|no|thanks|thank you)\s*,?\s*doctor\b",
            r"\bi came because\b",
            r"\bmy blood pressure was\b",
            r"\bmy blood pressure has been\b",
            r"\bit was around\b",
            r"\bon two different occasions\b",
            r"\bi have never been diagnosed\b",
            r"\bi sometimes get\b",
            r"\bi don't have\b",
            r"\bi have been feeling\b",
            r"\bmy (head|chest|ear|throat|back|stomach|arm|leg|blood)\b",
        ]

        labeled = []
        for t in conversation:
            if "role" in t and t["role"] in ["Doctor", "Patient"]:
                labeled.append(t)
                continue

            txt = t.get("text", "")
            low = txt.lower()
            spk = t.get("speaker", "Speaker 0")

            # Check direct cues first
            role = None
            for pc in patient_cues:
                if re.search(pc, low):
                    role = "Patient"
                    break
            if not role:
                for dc in doctor_cues:
                    if re.search(dc, low):
                        role = "Doctor"
                        break
            if not role:
                role = speaker_map.get(spk, "Patient")

            labeled.append({
                "role": role,
                "speaker": spk,
                "start": t.get("start"),
                "end": t.get("end"),
                "text": txt,
            })
        return labeled

    def _merge_consecutive_turns(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not turns:
            return []

        merged = []
        current = dict(turns[0])

        for nxt in turns[1:]:
            if nxt.get("role") == current.get("role"):
                # Append text
                current["text"] = f"{current.get('text', '').strip()} {nxt.get('text', '').strip()}".strip()
                if nxt.get("end") is not None:
                    current["end"] = nxt.get("end")
            else:
                merged.append(current)
                current = dict(nxt)

        if current:
            merged.append(current)

        return merged
