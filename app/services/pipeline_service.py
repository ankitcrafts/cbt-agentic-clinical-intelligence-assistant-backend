from app.services.conversation_service import ConversationService
from app.services.deepgram_service import DeepgramService
from app.services.diagnosis_service import DiagnosisService
from app.services.entity_service import EntityService
from app.services.summary_service import SummaryService


class ClinicalPipeline:
    """
    1. Deepgram transcription (raw conversation)
    2. Convert speakers into Doctor / Patient turns
    3. Summarize the encounter
    4. Extract medical entities
    5. Differential diagnosis (RAG can refine this later)
    """

    def __init__(self):
        self.deepgram = DeepgramService()
        self.conversation = ConversationService()
        self.summary = SummaryService()
        self.entity = EntityService()
        self.diagnosis = DiagnosisService()

    def run_from_audio(self, audio_path: str):
        asr = self.deepgram.transcribe(audio_path)
        return self.run_from_conversation(
            conversation=asr["conversation"],
            transcript=asr.get("transcript", ""),
        )

    def run_from_conversation(self, conversation, transcript: str = ""):
        labeled = self.conversation.to_doctor_patient(conversation)
        summary = self.summary.summarize(labeled["conversation"])
        entities = self.entity.extract(labeled["conversation"])
        diagnosis = self.diagnosis.differential_diagnosis(summary, entities)

        return {
            "transcript": transcript,
            "conversation": conversation,
            "doctor_patient_conversation": labeled["conversation"],
            "speaker_map": labeled["speaker_map"],
            "summary": summary,
            "entities": entities,
            "differential_diagnosis": diagnosis.get("differential_diagnosis", diagnosis),
        }
