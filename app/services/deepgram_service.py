import json
import os
from datetime import datetime
import requests

from app.core.config import DEEPGRAM_API_KEY

os.makedirs("output", exist_ok=True)


class DeepgramService:
    def __init__(self):
        self.api_key = DEEPGRAM_API_KEY

    def transcribe(self, file_path: str):
        with open(file_path, "rb") as f:
            audio = f.read()

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/*",
        }
        params = {
            "model": "nova-3",
            "language": "en",
            "smart_format": "true",
            "punctuate": "true",
            "diarize": "true",
            "utterances": "true",
            "paragraphs": "true",
        }

        try:
            resp = requests.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                params=params,
                data=audio,
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Deepgram API error ({resp.status_code}): {resp.text}")
            result = resp.json()
        except Exception as api_err:
            print(f"[Deepgram Error] {api_err}")
            raise api_err
        conversation = self._build_conversation(result)
        transcript = self._extract_transcript(result, conversation)

        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_conversation.json"
        conversation_file = os.path.join("output", filename)
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=4, ensure_ascii=False)

        return {
            "transcript": transcript,
            "conversation": conversation,
            "conversation_file": conversation_file,
        }

    def _extract_transcript(self, result, conversation) -> str:
        try:
            return result["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            return " ".join(turn.get("text", "") for turn in conversation)

    def _build_conversation(self, result):
        utterances = (result.get("results") or {}).get("utterances") or []
        if utterances:
            return [
                {
                    "speaker": f"Speaker {utt.get('speaker', 0)}",
                    "start": utt.get("start"),
                    "end": utt.get("end"),
                    "text": utt.get("transcript", ""),
                }
                for utt in utterances
            ]

        words = (
            ((result.get("results") or {}).get("channels") or [{}])[0]
            .get("alternatives", [{}])[0]
            .get("words")
            or []
        )
        return self._group_words_by_speaker(words)

    def _group_words_by_speaker(self, words):
        conversation = []
        current = None

        for word in words:
            speaker = f"Speaker {word.get('speaker', 0)}"
            token = word.get("punctuated_word") or word.get("word") or ""
            if current is None or current["speaker"] != speaker:
                if current:
                    conversation.append(current)
                current = {
                    "speaker": speaker,
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "text": token,
                }
            else:
                current["text"] = f"{current['text']} {token}".strip()
                current["end"] = word.get("end")

        if current:
            conversation.append(current)
        return conversation
