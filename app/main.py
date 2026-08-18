from fastapi import FastAPI
from app.services.deepgram_service import DeepgramService
import traceback
import uvicorn


app = FastAPI()
deepgram = DeepgramService()

@app.get("/")
def home():
    return {"message": "Deepgram Service Running"}


@app.get("/transcribe")
def transcribe():
    try:
        audio_path = "/Users/durgeshwarpratapsingh/Desktop/ISB_project/app/uploads/WhatsApp Audio 2026-08-13 at 18.50.16.mpeg"
        result = deepgram.transcribe(audio_path)
        
        return result

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",          # Use "app.main:app" if main.py is inside the app folder
        host="0.0.0.0",
        port=8000,
        reload=True
    )