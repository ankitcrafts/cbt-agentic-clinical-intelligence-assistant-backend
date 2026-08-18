import json
from deepgram import DeepgramClient
from app.core.config import DEEPGRAM_API_KEY
from datetime import datetime
import os

os.makedirs("output", exist_ok=True)

filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_conversation.json"
conversation_file = os.path.join("output", filename)




class DeepgramService:

    def __init__(self):
        self.client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

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

        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]

        conversation = []

        for utt in result["results"]["utterances"]:

            conversation.append(
                {
                    "speaker": f"Speaker {utt['speaker']}",
                    "start": utt["start"],
                    "end": utt["end"],
                    "text": utt["transcript"],
                }
            )

        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=4, ensure_ascii=False)
        return {
            "transcript": transcript,
            "conversation": conversation,
            "raw_response": result,
        }