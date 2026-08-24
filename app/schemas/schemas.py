from typing import Any, Optional

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    speaker: str
    start: Optional[float] = None
    end: Optional[float] = None
    text: str


class DoctorPatientTurn(BaseModel):
    role: str
    speaker: str
    start: Optional[float] = None
    end: Optional[float] = None
    text: str


class TranscriptionResponse(BaseModel):
    transcript: str
    conversation: list[ConversationTurn]


class PipelineResponse(BaseModel):
    transcript: str
    conversation: list[ConversationTurn]
    doctor_patient_conversation: list[DoctorPatientTurn]
    speaker_map: dict[str, str]
    summary: dict[str, Any]
    entities: dict[str, Any]
    differential_diagnosis: list[dict[str, Any]]


class AnalyzeConversationRequest(BaseModel):
    conversation: list[dict[str, Any]]
    transcript: Optional[str] = ""

