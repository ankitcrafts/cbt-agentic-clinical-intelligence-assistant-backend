import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI

from app.services.pipeline_service import ClinicalPipeline

app = FastAPI(title="Clinical Intelligence Pipeline")
pipeline = ClinicalPipeline()

AUDIO_PATH = "/Users/durgeshwarpratapsingh/Desktop/ISB_project/app/uploads/WhatsApp Audio 2026-08-13 at 18.50.16.mpeg"


@app.get("/")
def home():
    return {
        "message": "Clinical pipeline running",
        "steps": [
            "1. conversation transcription",
            "2. doctor-patient conversion",
            "3. conversation summary",
            "4. medical entity extraction",
            "5. differential diagnosis",
            "6. rag (next)",
        ],
    }


@app.get("/transcribe")
def transcribe():
    try:
        return pipeline.deepgram.transcribe(AUDIO_PATH)
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/analyze")
def analyze():
    try:
        return pipeline.run_from_audio(AUDIO_PATH)
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.get("/analyze-conversation")
def analyze_saved_conversation():
    try:
        output_dir = Path("output")
        files = sorted(output_dir.glob("*_conversation.json"))
        if not files:
            return {"status": "error", "message": "No conversation JSON found in output/"}
        latest = files[-1]
        with open(latest, encoding="utf-8") as f:
            conversation = json.load(f)
        result = pipeline.run_from_conversation(conversation)
        result["source_file"] = str(latest)
        return result
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
