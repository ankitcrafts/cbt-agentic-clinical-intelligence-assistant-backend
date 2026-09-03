import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(50), nullable=False, default="DOCTOR")  # ADMIN, DOCTOR, PATIENT
    department = Column(String(255), nullable=True)
    specialization = Column(String(255), nullable=True)
    designation = Column(String(255), nullable=True)
    phone_number = Column(String(50), nullable=True)
    avatar_initials = Column(String(10), nullable=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def get_initials(self) -> str:
        initials = self.avatar_initials
        if isinstance(initials, str) and initials.strip():
            return initials.strip()
        name = self.full_name
        if isinstance(name, str) and name.strip():
            parts = name.replace("Dr.", "").strip().split()
            if len(parts) >= 2:
                return (parts[0][0] + parts[1][0]).upper()
            elif len(parts) == 1 and parts[0]:
                return parts[0][:2].upper()
        return "CL"


class OTPVerification(Base):
    __tablename__ = "otp_verifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), index=True, nullable=False)
    otp_code = Column(String(10), nullable=False)
    role = Column(String(50), nullable=False)  # Target role (ADMIN, DOCTOR, PATIENT)
    purpose = Column(String(50), default="SIGNIN")  # SIGNIN, SIGNUP
    is_used = Column(Boolean, default=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def is_expired(self) -> bool:
        now = datetime.now(timezone.utc)
        # Handle naive or timezone-aware expires_at
        exp = self.expires_at
        if not isinstance(exp, datetime):
            return True
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return now > exp


class ClinicalRecording(Base):
    __tablename__ = "clinical_recordings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encounter_id = Column(String(100), nullable=True, index=True)
    doctor_id = Column(String(100), nullable=True)
    patient_id = Column(String(100), nullable=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, default=0)
    mime_type = Column(String(100), default="audio/wav")
    duration_seconds = Column(Integer, default=0)
    recording_started_at = Column(DateTime(timezone=True), nullable=True)
    recording_ended_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), default="Uploaded")
    consultation_mode = Column(String(50), default="IN_PERSON")  # IN_PERSON, TELECONSULTATION, REMOTE_TEAMS, REMOTE_MEET, REMOTE_WHATSAPP, FILE_UPLOAD
    audio_source = Column(String(50), default="MIC")  # MIC, SYSTEM_MIXED, FILE_UPLOAD
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SOAPFeedback(Base):
    __tablename__ = "soap_feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encounter_id = Column(String(100), nullable=False, index=True)
    doctor_id = Column(String(100), nullable=True)
    patient_id = Column(String(100), nullable=True)
    original_soap_json = Column(Text, nullable=False)
    edited_soap_json = Column(Text, nullable=False)
    edit_reason = Column(Text, nullable=True)
    edited_section = Column(String(50), default="all")  # subjective, objective, assessment, plan, all
    patient_summary = Column(Text, nullable=True)
    is_finalized = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PatientConsultationSession(Base):
    __tablename__ = "patient_consultation_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(100), nullable=False, index=True)
    patient_name = Column(String(255), nullable=True)
    doctor_id = Column(String(100), nullable=True)
    doctor_name = Column(String(255), nullable=True)
    encounter_id = Column(String(100), nullable=True, index=True)
    session_number = Column(Integer, nullable=False, default=1)
    session_title = Column(String(255), nullable=False, default="Session 1")
    appointment_date = Column(DateTime(timezone=True), nullable=True)
    transcript = Column(Text, nullable=True)
    conversation_json = Column(Text, nullable=True)  # JSON array of exchanges
    ai_suggestions_json = Column(Text, nullable=True)  # JSON array of suggestions
    clinical_analysis_json = Column(Text, nullable=True)  # JSON summary & findings
    soap_report_json = Column(Text, nullable=True)  # JSON soap note
    audio_file_name = Column(String(255), nullable=True)
    audio_file_path = Column(String(500), nullable=True)
    duration_seconds = Column(Integer, default=0)
    gdrive_file_id = Column(String(255), nullable=True)
    gdrive_web_view_link = Column(String(500), nullable=True)
    consultation_mode = Column(String(50), default="IN_PERSON")  # IN_PERSON, TELECONSULTATION, REMOTE_TEAMS, REMOTE_MEET, REMOTE_WHATSAPP, FILE_UPLOAD
    audio_source = Column(String(50), default="MIC")  # MIC, SYSTEM_MIXED, FILE_UPLOAD
    status = Column(String(50), default="COMPLETED")  # COMPLETED, IN_PROGRESS, ACTIVE
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())



