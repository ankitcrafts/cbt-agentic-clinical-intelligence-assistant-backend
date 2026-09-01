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


class SOAPReportSchema(BaseModel):
    subjective: Optional[str] = ""
    objective: Optional[str] = ""
    assessment: Optional[str] = ""
    plan: Optional[str] = ""


class SOAPReframeRequest(BaseModel):
    section: str  # subjective, objective, assessment, plan, all
    original_content: str
    edit_reason: str
    full_soap: Optional[dict[str, Any]] = None


class SOAPSaveFeedbackRequest(BaseModel):
    encounter_id: str
    doctor_id: Optional[str] = "DOC-001"
    patient_id: Optional[str] = "P1245"
    original_soap: dict[str, Any]
    edited_soap: dict[str, Any]
    edit_reason: Optional[str] = ""
    edited_section: Optional[str] = "all"
    patient_summary: Optional[str] = ""
    is_finalized: Optional[bool] = True


