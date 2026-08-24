import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, Integer
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
        if self.avatar_initials:
            return self.avatar_initials
        if self.full_name:
            parts = self.full_name.replace("Dr.", "").strip().split()
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())

