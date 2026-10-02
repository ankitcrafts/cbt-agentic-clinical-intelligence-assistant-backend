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
    retrieval_mode: str
    routed_documents: list[dict[str, Any]]
    retrieval_query: str
    dense_hit_count: int
    bm25_hit_count: int
    primary_guideline_used: str
    differential_diagnosis: list[dict[str, Any]]
    visit_id: str
    source_file: str
    status: str
    follow_up_iteration: int
    follow_up_questions: list[dict[str, Any]]
    previous_summary: dict[str, Any]
    previous_confidence: float
    confidence: float
    confidence_improved: bool
    follow_up_threshold: float
    follow_up_note: str
