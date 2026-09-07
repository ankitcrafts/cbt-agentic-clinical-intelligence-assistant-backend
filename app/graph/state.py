from typing import Any
from typing_extensions import TypedDict



class ClinicalState(TypedDict, total=False):
    audio_path: str
    transcript: str
    conversation: list[dict[str, Any]]
    speaker_map: dict[str, str]
    doctor_patient_conversation: list[dict[str, Any]]
    summary: dict[str, Any]
    entities: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    rag_context: str
    primary_guideline_used: str
    differential_diagnosis: list[dict[str, Any]]
    visit_id: str
    source_file: str
