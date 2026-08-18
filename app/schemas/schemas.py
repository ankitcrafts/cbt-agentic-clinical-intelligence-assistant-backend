from pydantic import BaseModel


class Conversation(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    transcript: str
    conversation: list[Conversation]