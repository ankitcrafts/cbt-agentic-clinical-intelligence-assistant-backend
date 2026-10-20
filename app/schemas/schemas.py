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
    doctor_name: Optional[str] = None
    patient_name: Optional[str] = None
    doctor_initials: Optional[str] = None
    patient_initials: Optional[str] = None
    visit_id: Optional[str] = None


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


class SOAPExportPDFRequest(BaseModel):
    visit_id: Optional[str] = "visit"
    encounter_id: Optional[str] = None
    patient_name: Optional[str] = "Patient"
    patient_id: Optional[str] = ""
    doctor_name: Optional[str] = "Dr. Yuki Tanaka"
    date: Optional[str] = None
    soap: Optional[dict[str, Any]] = None
    soap_report: Optional[dict[str, Any]] = None
    differential_diagnosis: Optional[list[dict[str, Any]]] = None



class PatientConsultationSessionResponse(BaseModel):
    id: str
    patient_id: str
    patient_name: Optional[str] = None
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    encounter_id: Optional[str] = None
    session_number: int
    session_title: str
    appointment_date: Optional[str] = None
    transcript: Optional[str] = ""
    conversation: Optional[list[dict[str, Any]]] = []
    ai_suggestions: Optional[list[dict[str, Any]]] = []
    clinical_analysis: Optional[dict[str, Any]] = {}
    soap_report: Optional[dict[str, Any]] = {}
    audio_file_name: Optional[str] = None
    duration_seconds: int = 0
    gdrive_file_id: Optional[str] = None
    gdrive_web_view_link: Optional[str] = None
    consultation_mode: Optional[str] = "IN_PERSON"
    audio_source: Optional[str] = "MIC"
    status: str = "COMPLETED"
    created_at: Optional[str] = None


class CreateConsultationSessionRequest(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    doctor_id: Optional[str] = "DOC-001"
    doctor_name: Optional[str] = "Dr. Yuki Tanaka"
    encounter_id: Optional[str] = None
    session_number: Optional[int] = None
    session_title: Optional[str] = None
    appointment_date: Optional[str] = None
    transcript: Optional[str] = ""
    conversation: Optional[list[dict[str, Any]]] = []
    ai_suggestions: Optional[list[dict[str, Any]]] = []
    clinical_analysis: Optional[dict[str, Any]] = {}
    soap_report: Optional[dict[str, Any]] = {}
    duration_seconds: Optional[int] = 0
    consultation_mode: Optional[str] = "IN_PERSON"
    audio_source: Optional[str] = "MIC"
    status: Optional[str] = "COMPLETED"




