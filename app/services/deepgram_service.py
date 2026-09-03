import json
import os
from datetime import datetime

from deepgram import DeepgramClient

from app.core.config import DEEPGRAM_API_KEY
from app.core.tracing import traceable

os.makedirs("output", exist_ok=True)


class DeepgramService:
    def __init__(self):
        self.client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    @traceable(name="deepgram_transcribe", run_type="tool")
    def transcribe(self, file_path: str):
        with open(file_path, "rb") as f:
            audio = f.read()

        response = self.client.listen.v1.media.transcribe_file(
            request=audio,
            model="nova-3",
            language="en",
            smart_format=True,
            punctuate=True,
            diarize=True,
            utterances=True,
            paragraphs=True,
        )

        result = response.model_dump(mode="json")
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
