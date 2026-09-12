import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="DOCTOR")  # ADMIN, DOCTOR, PATIENT
    department: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    specialization: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    designation: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    avatar_initials: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

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
                parts[0][:2].upper()
        return "CL"


class OTPVerification(Base):
    __tablename__ = "otp_verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    otp_code: Mapped[str] = mapped_column(String(10), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # Target role (ADMIN, DOCTOR, PATIENT)
    purpose: Mapped[str] = mapped_column(String(50), default="SIGNIN")  # SIGNIN, SIGNUP
    is_used: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())

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

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encounter_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    doctor_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    patient_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[str] = mapped_column(String(100), default="audio/wav")
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    recording_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    recording_ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="Uploaded")
    consultation_mode: Mapped[str] = mapped_column(String(50), default="IN_PERSON")  # IN_PERSON, TELECONSULTATION, REMOTE_TEAMS, REMOTE_MEET, REMOTE_WHATSAPP, FILE_UPLOAD
    audio_source: Mapped[str] = mapped_column(String(50), default="MIC")  # MIC, SYSTEM_MIXED, FILE_UPLOAD
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SOAPFeedback(Base):
    __tablename__ = "soap_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encounter_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    doctor_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    patient_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    original_soap_json: Mapped[str] = mapped_column(Text, nullable=False)
    edited_soap_json: Mapped[str] = mapped_column(Text, nullable=False)
    edit_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    edited_section: Mapped[str] = mapped_column(String(50), default="all")  # subjective, objective, assessment, plan, all
    patient_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_finalized: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PatientConsultationSession(Base):
    __tablename__ = "patient_consultation_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    patient_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    doctor_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    doctor_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    encounter_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    session_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    session_title: Mapped[str] = mapped_column(String(255), nullable=False, default="Session 1")
    appointment_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    conversation_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of exchanges
    ai_suggestions_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of suggestions
    clinical_analysis_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON summary & findings
    soap_report_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON soap note
    audio_file_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    audio_file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    gdrive_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    gdrive_web_view_link: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    consultation_mode: Mapped[str] = mapped_column(String(50), default="IN_PERSON")  # IN_PERSON, TELECONSULTATION, REMOTE_TEAMS, REMOTE_MEET, REMOTE_WHATSAPP, FILE_UPLOAD
    audio_source: Mapped[str] = mapped_column(String(50), default="MIC")  # MIC, SYSTEM_MIXED, FILE_UPLOAD
    status: Mapped[str] = mapped_column(String(50), default="COMPLETED")  # COMPLETED, IN_PROGRESS, ACTIVE
    is_shared_with_patient: Mapped[bool] = mapped_column(Boolean, default=False)
    shared_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PatientModel(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # e.g. "#00847261"
    pid: Mapped[str] = mapped_column(String(50), index=True, nullable=False)  # e.g. "P-00847261"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    initials: Mapped[str] = mapped_column(String(10), nullable=False, default="PT")
    age: Mapped[int] = mapped_column(Integer, nullable=False, default=35)
    gender: Mapped[str] = mapped_column(String(20), nullable=False, default="M")
    phone: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    demographics: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    arrival_time: Mapped[str] = mapped_column(String(50), nullable=False, default="09:00 AM")
    arrival_status: Mapped[str] = mapped_column(String(50), nullable=False, default="ARRIVED")  # ARRIVED, SCHEDULED, IN_CONSULTATION, COMPLETED
    reason: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    condition: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    last_visit: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    vitals_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Doctor allocation & Preferences
    assigned_doctor_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    assigned_doctor_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    assigned_time_slot: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    assigned_duration: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="30 min")
    is_preferred_doctor_choice: Mapped[bool] = mapped_column(Boolean, default=False)
    preferred_doctor_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    visit_type: Mapped[str] = mapped_column(String(100), default="Consultation")
    booking_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DoctorSlotModel(Base):
    __tablename__ = "doctor_slots"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # e.g. "DOC-001-s1"
    doctor_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    doctor_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    time_slot: Mapped[str] = mapped_column(String(50), nullable=False)
    duration: Mapped[str] = mapped_column(String(50), nullable=False, default="30 min")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="AVAILABLE")  # AVAILABLE, BOOKED
    booked_patient_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    booked_patient_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())





