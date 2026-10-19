import json
import logging
import re
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc, func

from app.core.database import get_db
from app.models.models import PatientModel, DoctorSlotModel, PatientConsultationSession, User, SOAPFeedback, OTPVerification, DoctorModel
from app.services.email_service import send_otp_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scheduling", tags=["Patient Scheduling & Doctor Allocation"])


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class ShareSoapReportSchema(BaseModel):
    encounter_id: Optional[str] = None
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    soap_report: Dict[str, Any]
    prescription: Optional[Dict[str, Any]] = None
    patient_name: Optional[str] = None
    patient_summary: Optional[str] = None


class CompleteConsultationSchema(BaseModel):
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    encounter_id: Optional[str] = None
    notes: Optional[str] = None


class VitalsSchema(BaseModel):
    bp: Optional[str] = "120/80 mmHg"
    hr: Optional[str] = "72 bpm"
    pain: Optional[str] = "0/10"
    spo2: Optional[str] = "98%"
    temp: Optional[str] = "36.6 °C"


class PatientCreateSchema(BaseModel):
    name: str = Field(..., examples=["Marcus Delgado"])
    age: int = Field(35, examples=[42])
    gender: str = Field("M", examples=["M"])
    phone: str = Field("+1 (555) 000-0000", examples=["+1 (555) 234-5678"])
    email: Optional[str] = Field(None, examples=["patient@gmail.com"])
    reason: str = Field(..., examples=["Hypertension review + morning headache"])
    condition: str = Field("Triage review", examples=["Hypertension Stage 1"])
    visit_type: str = Field("New Patient", examples=["Follow-up"])
    vitals: Optional[VitalsSchema] = None
    preferred_doctor_name: Optional[str] = None
    is_preferred_doctor_choice: Optional[bool] = False
    assigned_doctor_id: Optional[str] = None
    assigned_doctor_name: Optional[str] = None
    assigned_time_slot: Optional[str] = None
    booking_notes: Optional[str] = None


class PatientUpdateSchema(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    arrival_status: Optional[str] = None
    reason: Optional[str] = None
    condition: Optional[str] = None
    last_visit: Optional[str] = None
    visit_type: Optional[str] = None
    vitals: Optional[VitalsSchema] = None
    assigned_doctor_id: Optional[str] = None
    assigned_doctor_name: Optional[str] = None
    assigned_time_slot: Optional[str] = None
    assigned_duration: Optional[str] = None
    is_preferred_doctor_choice: Optional[bool] = None
    preferred_doctor_name: Optional[str] = None
    booking_notes: Optional[str] = None


class AssignDoctorSlotSchema(BaseModel):
    patient_id: str = Field(..., examples=["#00847261"])
    doctor_id: str = Field(..., examples=["DOC-001"])
    doctor_name: Optional[str] = None
    time_slot: str = Field(..., examples=["09:00 AM"])
    duration: Optional[str] = "30 min"
    is_preferred_choice: bool = False
    preferred_doctor_name: Optional[str] = None
    visit_type: Optional[str] = "Consultation"
    booking_notes: Optional[str] = None


class DoctorCreateSchema(BaseModel):
    name: str = Field(..., examples=["Dr. Maya Lin"])
    email: str = Field(..., examples=["maya.lin@stmercy.org"])
    specialty: str = Field("General Physician", examples=["Cardiologist"])
    department: str = Field("Internal Medicine", examples=["Cardiology"])
    phone: Optional[str] = "+1 (555) 000-0000"
    room: Optional[str] = "Suite 101"
    experience: Optional[str] = "Senior Consultant"
    qualification: Optional[str] = "MD, MBBS"
    is_new_upcoming: Optional[bool] = False
    duty_status: Optional[str] = "On duty"
    shift_timing: Optional[str] = "09:00 AM - 05:00 PM"
    slot_duration: Optional[str] = "30 min"


class DoctorUpdateSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    specialty: Optional[str] = None
    department: Optional[str] = None
    phone: Optional[str] = None
    room: Optional[str] = None
    experience: Optional[str] = None
    qualification: Optional[str] = None
    is_new_upcoming: Optional[bool] = None
    duty_status: Optional[str] = None
    available: Optional[bool] = None
    shift_timing: Optional[str] = None


class DoctorSlotCreateSchema(BaseModel):
    time_slot: str = Field(..., examples=["09:00 AM"])
    duration: Optional[str] = "30 min"
    status: Optional[str] = "AVAILABLE"


class CreateConsultationSessionSchema(BaseModel):
    session_title: str = Field("Consultation Session", examples=["Session 2: Follow-up Consultation"])
    encounter_id: Optional[str] = None
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    transcript: Optional[str] = ""
    conversation: Optional[List[Dict[str, Any]]] = None
    ai_suggestions: Optional[List[Dict[str, Any]]] = None
    clinical_analysis: Optional[Dict[str, Any]] = None
    soap_report: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[int] = 0
    consultation_mode: Optional[str] = "IN_PERSON"
    audio_source: Optional[str] = "MIC"
    status: Optional[str] = "COMPLETED"


# ==============================================================================
# HARDCODED DOCTOR PROFILES WITH SLOTS SYNC
# ==============================================================================

DEFAULT_DOCTORS: List[Dict[str, Any]] = [
    {
        "id": "DOC-001",
        "name": "Dr. Yuki Tanaka",
        "initials": "YT",
        "specialty": "Senior Cardiologist",
        "department": "Cardiology",
        "email": "doctor@cliniq.health",
        "phone": "+1 (555) 201-4401",
        "room": "Suite 204",
        "experience": "Senior Consultant",
        "qualification": "MD, FACC, FSCAI",
        "isNewUpcoming": False,
        "shiftTiming": "09:00 AM - 05:00 PM",
        "available": True,
        "dutyStatus": "On duty",
        "default_slots": [
            ("09:00 AM", "30 min"),
            ("09:45 AM", "30 min"),
            ("10:30 AM", "30 min"),
            ("11:15 AM", "45 min"),
            ("02:00 PM", "30 min"),
            ("02:45 PM", "30 min"),
            ("03:30 PM", "30 min"),
            ("04:15 PM", "30 min"),
        ]
    },
    {
        "id": "DOC-002",
        "name": "Dr. Rahul Sharma",
        "initials": "RS",
        "specialty": "Interventional Cardiologist",
        "department": "Cardiology",
        "email": "rahul.sharma@stmercy.org",
        "phone": "+1 (555) 201-4402",
        "room": "Suite 206",
        "experience": "Attending Specialist",
        "qualification": "MD, DM Cardiology",
        "isNewUpcoming": False,
        "shiftTiming": "09:30 AM - 04:30 PM",
        "available": True,
        "dutyStatus": "On duty",
        "default_slots": [
            ("09:30 AM", "30 min"),
            ("10:15 AM", "30 min"),
            ("11:00 AM", "30 min"),
            ("02:00 PM", "30 min"),
            ("02:45 PM", "30 min"),
            ("03:30 PM", "30 min"),
        ]
    },
    {
        "id": "DOC-003",
        "name": "Dr. Neha Verma",
        "initials": "NV",
        "specialty": "Neurologist",
        "department": "Neurology",
        "email": "neha.verma@stmercy.org",
        "phone": "+1 (555) 201-4403",
        "room": "Suite 310",
        "experience": "Senior Consultant",
        "qualification": "MD, DM Neurology",
        "isNewUpcoming": False,
        "shiftTiming": "09:00 AM - 04:00 PM",
        "available": True,
        "dutyStatus": "On duty",
        "default_slots": [
            ("09:00 AM", "45 min"),
            ("10:00 AM", "30 min"),
            ("10:45 AM", "30 min"),
            ("11:30 AM", "30 min"),
            ("02:15 PM", "30 min"),
            ("03:00 PM", "30 min"),
        ]
    },
    {
        "id": "DOC-004",
        "name": "Dr. Amit Patel",
        "initials": "AP",
        "specialty": "Orthopedic Surgeon",
        "department": "Orthopedics",
        "email": "amit.patel@stmercy.org",
        "phone": "+1 (555) 201-4404",
        "room": "Suite 108",
        "experience": "Attending Surgeon",
        "qualification": "MS Orthopedics, FRCS",
        "isNewUpcoming": False,
        "shiftTiming": "09:30 AM - 03:00 PM",
        "available": True,
        "dutyStatus": "On duty",
        "default_slots": [
            ("09:30 AM", "30 min"),
            ("10:15 AM", "30 min"),
            ("11:00 AM", "30 min"),
            ("02:00 PM", "30 min"),
        ]
    },
    {
        "id": "DOC-005",
        "name": "Dr. Sneha Iyer",
        "initials": "SI",
        "specialty": "Pediatrician",
        "department": "Pediatrics",
        "email": "sneha.iyer@stmercy.org",
        "phone": "+1 (555) 201-4405",
        "room": "Exam Room 2B",
        "experience": "Upcoming Fellow",
        "qualification": "MD Pediatrics",
        "isNewUpcoming": True,
        "shiftTiming": "09:00 AM - 01:00 PM",
        "available": False,
        "dutyStatus": "Off duty",
        "default_slots": [
            ("09:00 AM", "30 min"),
            ("09:45 AM", "30 min"),
            ("10:30 AM", "30 min"),
        ]
    },
    {
        "id": "DOC-006",
        "name": "Dr. Vikram Singh",
        "initials": "VS",
        "specialty": "General Physician",
        "department": "Internal Medicine",
        "email": "vikram.singh@stmercy.org",
        "phone": "+1 (555) 201-4406",
        "room": "Suite 102",
        "experience": "Attending Physician",
        "qualification": "MBBS, MD Internal Medicine",
        "isNewUpcoming": False,
        "shiftTiming": "09:00 AM - 03:00 PM",
        "available": True,
        "dutyStatus": "On duty",
        "default_slots": [
            ("09:00 AM", "30 min"),
            ("09:45 AM", "30 min"),
            ("10:30 AM", "30 min"),
            ("11:15 AM", "30 min"),
            ("02:00 PM", "30 min"),
        ]
    },
    {
        "id": "DOC-007",
        "name": "Dr. Maya Lin",
        "initials": "ML",
        "specialty": "Clinical Psychologist",
        "department": "Psychiatry",
        "email": "maya.lin@stmercy.org",
        "phone": "+1 (555) 201-4407",
        "room": "Suite 405",
        "experience": "New Upcoming Specialist",
        "qualification": "PsyD, CBT Certified",
        "isNewUpcoming": True,
        "shiftTiming": "10:00 AM - 04:00 PM",
        "available": True,
        "dutyStatus": "On duty",
        "default_slots": [
            ("10:00 AM", "45 min"),
            ("11:00 AM", "45 min"),
            ("02:00 PM", "45 min"),
            ("03:00 PM", "45 min"),
        ]
    },
    {
        "id": "DOC-008",
        "name": "Dr. Ankit Kumar",
        "initials": "AK",
        "specialty": "Associate Cardiologist",
        "department": "Cardiology",
        "email": "ankitkumar021303@gmail.com",
        "phone": "+1 (555) 201-4408",
        "room": "Suite 201",
        "experience": "Senior Resident",
        "qualification": "MD Cardiology",
        "isNewUpcoming": False,
        "shiftTiming": "09:00 AM - 05:00 PM",
        "available": True,
        "dutyStatus": "On duty",
        "default_slots": [
            ("09:00 AM", "30 min"),
            ("09:45 AM", "30 min"),
            ("10:30 AM", "30 min"),
            ("11:15 AM", "30 min"),
            ("02:00 PM", "30 min"),
            ("02:45 PM", "30 min"),
            ("03:30 PM", "30 min"),
        ]
    }
]


def format_patient_dict(p: PatientModel, session_count: int = 0) -> Dict[str, Any]:
    vitals = {}
    if p.vitals_json:
        try:
            vitals = json.loads(str(p.vitals_json))
        except Exception:
            vitals = {}

    return {
        "id": p.id,
        "pid": p.pid,
        "name": p.name,
        "initials": p.initials,
        "age": p.age,
        "gender": p.gender,
        "phone": p.phone,
        "email": p.email,
        "demographics": p.demographics or f"{p.age}{p.gender} · {p.id}",
        "arrivalTime": p.arrival_time,
        "arrivalStatus": p.arrival_status,
        "reason": p.reason,
        "condition": p.condition,
        "lastVisit": p.last_visit or "—",
        "vitals": vitals,
        "assignedDoctorId": p.assigned_doctor_id,
        "assignedDoctorName": p.assigned_doctor_name,
        "assignedTimeSlot": p.assigned_time_slot,
        "assignedDuration": p.assigned_duration or "30 min",
        "isPreferredDoctorChoice": bool(p.is_preferred_doctor_choice),
        "preferredDoctorName": p.preferred_doctor_name,
        "visitType": p.visit_type,
        "bookingNotes": p.booking_notes,
        "totalPastSessions": session_count,
        "createdAt": p.created_at.isoformat() if p.created_at else None,
        "updatedAt": p.updated_at.isoformat() if p.updated_at else None,
    }


def ensure_initial_seed_data(db: Session):
    """Seed default patients, doctors in User table, and doctor slots if database is freshly created."""
    # Seed default doctors in User table if not present
    for doc in DEFAULT_DOCTORS:
        existing_u = db.query(User).filter(
            or_(
                User.id == doc["id"],
                User.email == doc["email"],
                func.lower(User.full_name) == doc["name"].lower()
            )
        ).first()
        if not existing_u:
            new_u = User(
                id=doc["id"],
                email=doc["email"],
                full_name=doc["name"],
                role="DOCTOR",
                department=doc["department"],
                specialization=doc["specialty"],
                designation=f"{doc['specialty']} ({doc['department']})",
                avatar_initials=doc["initials"],
                is_active=doc["available"]
            )
            db.add(new_u)
    db.commit()

    patient_count = db.query(PatientModel).count()
    if patient_count == 0:
        logger.info("Seeding initial hospital patients into database...")
        initial_patients = [
            PatientModel(
                id="#00847261",
                pid="P-00847261",
                name="Marcus Delgado",
                initials="MD",
                age=42,
                gender="M",
                phone="+1 (555) 234-5678",
                demographics="42M · #00847261",
                arrival_time="08:45 AM",
                arrival_status="IN_CONSULTATION",
                reason="Hypertension review + new headache complaint",
                condition="Hypertension Stage 1",
                last_visit="July 14, 2026",
                vitals_json=json.dumps({"bp": "138/88 mmHg", "hr": "72 bpm", "pain": "5-6/10", "spo2": "98%", "temp": "36.6 °C"}),
                assigned_doctor_id="DOC-001",
                assigned_doctor_name="Dr. Yuki Tanaka",
                assigned_time_slot="09:00 AM",
                assigned_duration="30 min",
                is_preferred_doctor_choice=True,
                preferred_doctor_name="Dr. Yuki Tanaka",
                visit_type="Follow-up",
                booking_notes="Patient requested Dr. Yuki Tanaka specifically for cardiology follow-up.",
            ),
            PatientModel(
                id="#00391847",
                pid="P-00391847",
                name="Priya Sharma",
                initials="PS",
                age=34,
                gender="F",
                phone="+1 (555) 345-6789",
                demographics="34F · #00391847",
                arrival_time="09:15 AM",
                arrival_status="SCHEDULED",
                reason="Persistent morning dizziness & fatigue, thyroid review",
                condition="Hypothyroidism",
                last_visit="May 02, 2026",
                vitals_json=json.dumps({"bp": "118/76 mmHg", "hr": "68 bpm", "pain": "0/10", "spo2": "99%", "temp": "36.8 °C"}),
                assigned_doctor_id="DOC-001",
                assigned_doctor_name="Dr. Yuki Tanaka",
                assigned_time_slot="09:45 AM",
                assigned_duration="30 min",
                is_preferred_doctor_choice=False,
                visit_type="Consultation",
                booking_notes="Assigned based on cardiology queue availability.",
            ),
            PatientModel(
                id="#00512938",
                pid="P-00512938",
                name="Elias Voss",
                initials="EV",
                age=58,
                gender="M",
                phone="+1 (555) 456-7890",
                demographics="58M · #00512938",
                arrival_time="08:50 AM",
                arrival_status="SCHEDULED",
                reason="Routine annual check, peripheral tingling review",
                condition="Type 2 Diabetes",
                last_visit="Feb 11, 2026",
                vitals_json=json.dumps({"bp": "132/84 mmHg", "hr": "74 bpm", "pain": "0/10"}),
                assigned_doctor_id="DOC-003",
                assigned_doctor_name="Dr. Neha Verma",
                assigned_time_slot="09:00 AM",
                assigned_duration="45 min",
                is_preferred_doctor_choice=True,
                preferred_doctor_name="Dr. Neha Verma",
                visit_type="Annual Check",
                booking_notes="Patient preferred Dr. Neha Verma for neurology evaluation.",
            ),
            PatientModel(
                id="#00673920",
                pid="P-00673920",
                name="Anya Kowalski",
                initials="AK",
                age=29,
                gender="F",
                phone="+1 (555) 567-8901",
                demographics="29F · #00673920",
                arrival_time="09:30 AM",
                arrival_status="SCHEDULED",
                reason="Chest tightness, shortness of breath on exertion",
                condition="New presentation",
                last_visit="—",
                vitals_json=json.dumps({"bp": "122/80 mmHg", "hr": "82 bpm", "pain": "3/10"}),
                assigned_doctor_id="DOC-001",
                assigned_doctor_name="Dr. Yuki Tanaka",
                assigned_time_slot="11:15 AM",
                assigned_duration="45 min",
                is_preferred_doctor_choice=True,
                preferred_doctor_name="Dr. Yuki Tanaka",
                visit_type="New Patient",
                booking_notes="New referral requested senior cardiologist Dr. Yuki Tanaka.",
            ),
            PatientModel(
                id="#00182736",
                pid="P-00182736",
                name="Samuel Obi",
                initials="SO",
                age=67,
                gender="M",
                phone="+1 (555) 678-9012",
                demographics="67M · #00182736",
                arrival_time="09:10 AM",
                arrival_status="SCHEDULED",
                reason="Post-catheterisation review, stable angina check",
                condition="Stable Angina",
                last_visit="June 20, 2026",
                vitals_json=json.dumps({"bp": "128/82 mmHg", "hr": "64 bpm", "pain": "1/10"}),
                assigned_doctor_id="DOC-002",
                assigned_doctor_name="Dr. Rahul Sharma",
                assigned_time_slot="09:30 AM",
                assigned_duration="30 min",
                is_preferred_doctor_choice=False,
                visit_type="Follow-up",
                booking_notes="Assigned to Dr. Rahul Sharma for catheterisation follow-up.",
            ),
            PatientModel(
                id="#00847193",
                pid="P-00847193",
                name="Lin Wei",
                initials="LW",
                age=45,
                gender="F",
                phone="+1 (555) 789-0123",
                demographics="45F · #00847193",
                arrival_time="09:40 AM",
                arrival_status="SCHEDULED",
                reason="AF management, medication titration check",
                condition="Atrial Fibrillation",
                last_visit="March 18, 2026",
                vitals_json=json.dumps({"bp": "126/80 mmHg", "hr": "76 bpm", "pain": "0/10"}),
                assigned_doctor_id="DOC-001",
                assigned_doctor_name="Dr. Yuki Tanaka",
                assigned_time_slot="02:00 PM",
                assigned_duration="30 min",
                is_preferred_doctor_choice=True,
                preferred_doctor_name="Dr. Yuki Tanaka",
                visit_type="Follow-up",
                booking_notes="Patient requested Dr. Yuki Tanaka for AF check.",
            ),
            PatientModel(
                id="#00928174",
                pid="P-00928174",
                name="Arjun Nair",
                initials="AN",
                age=51,
                gender="M",
                phone="+1 (555) 890-1234",
                demographics="51M · #00928174",
                arrival_time="09:50 AM",
                arrival_status="ARRIVED",
                reason="Elevated home BP readings, palpitation episodes",
                condition="Arrhythmia screening",
                last_visit="—",
                vitals_json=json.dumps({"bp": "144/92 mmHg", "hr": "88 bpm", "pain": "0/10"}),
                assigned_doctor_id=None,
                assigned_doctor_name=None,
                assigned_time_slot=None,
                is_preferred_doctor_choice=False,
                visit_type="Consultation",
                booking_notes="Walk-in patient awaiting doctor allocation.",
            ),
            PatientModel(
                id="#00472910",
                pid="P-00472910",
                name="Fatima Al-Zahra",
                initials="FA",
                age=39,
                gender="F",
                phone="+1 (555) 901-2345",
                demographics="39F · #00472910",
                arrival_time="10:05 AM",
                arrival_status="ARRIVED",
                reason="Acute joint inflammation, knee swelling post-exercise",
                condition="Acute Arthralgia",
                last_visit="—",
                vitals_json=json.dumps({"bp": "120/78 mmHg", "hr": "70 bpm", "pain": "6/10"}),
                assigned_doctor_id=None,
                assigned_doctor_name=None,
                assigned_time_slot=None,
                is_preferred_doctor_choice=False,
                visit_type="New Patient",
                booking_notes="Requested Orthopedic specialist assessment.",
            ),
        ]
        db.add_all(initial_patients)
        db.commit()

    # Seed Doctor Slots if empty
    slot_count = db.query(DoctorSlotModel).count()
    if slot_count == 0:
        logger.info("Seeding initial Doctor timing slots into database...")
        slot_entries = []
        for doc in DEFAULT_DOCTORS:
            default_slots: List[Tuple[str, str]] = doc.get("default_slots", [])
            for idx, (slot_time, duration) in enumerate(default_slots):
                slot_id = f"{doc['id']}-s{idx+1}"
                matched_pat = db.query(PatientModel).filter(
                    PatientModel.assigned_doctor_id == doc["id"],
                    PatientModel.assigned_time_slot == slot_time
                ).first()
                
                slot_entries.append(
                    DoctorSlotModel(
                        id=slot_id,
                        doctor_id=doc["id"],
                        doctor_name=doc["name"],
                        time_slot=slot_time,
                        duration=duration,
                        status="BOOKED" if matched_pat else "AVAILABLE",
                        booked_patient_id=matched_pat.id if matched_pat else None,
                        booked_patient_name=matched_pat.name if matched_pat else None,
                    )
                )
        db.add_all(slot_entries)
        db.commit()


def get_all_doctors_from_db(db: Session) -> List[Dict[str, Any]]:
    """
    Fetch all active and registered doctors dynamically from the User database (role='DOCTOR').
    Ensures newly signed up doctors automatically have timing slots and appear in all scheduling desks.
    """
    ensure_initial_seed_data(db)

    # Fetch all doctors from User table (case-insensitive role check)
    doctor_users = db.query(User).filter(func.upper(User.role) == "DOCTOR").all()

    doctors_list = []
    seen_ids = set()
    seen_emails = set()
    seen_names = set()

    for u in doctor_users:
        seen_ids.add(str(u.id).lower())
        if u.email:
            seen_emails.add(str(u.email).lower())

        raw_name = (u.full_name or "").strip()
        if not raw_name:
            raw_name = u.email.split("@")[0].replace(".", " ").title()
        
        if not (raw_name.lower().startswith("dr.") or raw_name.lower().startswith("dr ")):
            formatted_name = f"Dr. {raw_name}"
        else:
            formatted_name = raw_name

        seen_names.add(formatted_name.lower())

        specialty = u.specialization or u.department or "Senior Cardiologist"
        department = u.department or "Cardiology"
        initials = u.get_initials()

        # Ensure doctor slots exist in DoctorSlotModel for this doctor
        existing_slots = db.query(DoctorSlotModel).filter(DoctorSlotModel.doctor_id == u.id).all()
        if not existing_slots:
            standard_slots = [
                ("09:00 AM", "30 min"),
                ("09:45 AM", "30 min"),
                ("10:30 AM", "30 min"),
                ("11:15 AM", "30 min"),
                ("02:00 PM", "30 min"),
                ("02:45 PM", "30 min"),
                ("03:30 PM", "30 min"),
                ("04:15 PM", "30 min"),
            ]
            new_slots = []
            for idx, (slot_time, duration) in enumerate(standard_slots):
                slot_id = f"{u.id}-s{idx+1}"
                matched_pat = db.query(PatientModel).filter(
                    PatientModel.assigned_doctor_id == u.id,
                    PatientModel.assigned_time_slot == slot_time
                ).first()
                new_slots.append(
                    DoctorSlotModel(
                        id=slot_id,
                        doctor_id=u.id,
                        doctor_name=formatted_name,
                        time_slot=slot_time,
                        duration=duration,
                        status="BOOKED" if matched_pat else "AVAILABLE",
                        booked_patient_id=matched_pat.id if matched_pat else None,
                        booked_patient_name=matched_pat.name if matched_pat else None,
                    )
                )
            db.add_all(new_slots)
            db.commit()
            existing_slots = db.query(DoctorSlotModel).filter(DoctorSlotModel.doctor_id == u.id).all()

        slot_items = [
            {
                "id": s.id,
                "time": s.time_slot,
                "duration": s.duration,
                "status": s.status,
                "bookedPatientId": s.booked_patient_id,
                "bookedPatientName": s.booked_patient_name,
            }
            for s in existing_slots
        ]

        # Dynamic queue count of patients assigned to this doctor
        clean_search = formatted_name.replace("Dr.", "").strip()
        queue_count = db.query(PatientModel).filter(
            or_(
                PatientModel.assigned_doctor_id == u.id,
                PatientModel.assigned_doctor_name.ilike(f"%{clean_search}%"),
                PatientModel.assigned_doctor_name.ilike(f"%{formatted_name}%")
            ),
            PatientModel.arrival_status.in_(["SCHEDULED", "IN_CONSULTATION", "ARRIVED"])
        ).count()

        doctors_list.append({
            "id": u.id,
            "name": formatted_name,
            "initials": initials,
            "specialty": specialty,
            "department": department,
            "email": u.email,
            "phone": u.phone_number or "+1 (555) 000-0000",
            "room": "Suite 101",
            "experience": "Senior Specialist",
            "qualification": "MD, MBBS",
            "isNewUpcoming": False,
            "shiftTiming": "09:00 AM - 05:00 PM",
            "available": True,
            "dutyStatus": "On duty",
            "queueCount": queue_count,
            "slots": slot_items
        })

    # Include any DEFAULT_DOCTORS that are not already in User table
    for doc in DEFAULT_DOCTORS:
        doc_id_low = str(doc["id"]).lower()
        doc_email_low = str(doc["email"]).lower()
        doc_name_low = str(doc["name"]).lower()

        if doc_id_low not in seen_ids and doc_email_low not in seen_emails and doc_name_low not in seen_names:
            seen_ids.add(doc_id_low)
            slots_in_db = db.query(DoctorSlotModel).filter(
                DoctorSlotModel.doctor_id == doc["id"]
            ).all()

            slot_items = [
                {
                    "id": s.id,
                    "time": s.time_slot,
                    "duration": s.duration,
                    "status": s.status,
                    "bookedPatientId": s.booked_patient_id,
                    "bookedPatientName": s.booked_patient_name,
                }
                for s in slots_in_db
            ]

            queue_count = db.query(PatientModel).filter(
                PatientModel.assigned_doctor_id == doc["id"],
                PatientModel.arrival_status.in_(["SCHEDULED", "IN_CONSULTATION", "ARRIVED"])
            ).count()

            doctors_list.append({
                "id": doc["id"],
                "name": doc["name"],
                "initials": doc["initials"],
                "specialty": doc["specialty"],
                "department": doc["department"],
                "email": doc["email"],
                "phone": doc.get("phone", "+1 (555) 201-4400"),
                "room": doc.get("room", "Suite 201"),
                "experience": doc.get("experience", "Senior Consultant"),
                "qualification": doc.get("qualification", "MD"),
                "isNewUpcoming": doc.get("isNewUpcoming", False),
                "shiftTiming": doc.get("shiftTiming", "09:00 AM - 05:00 PM"),
                "available": doc.get("available", True),
                "dutyStatus": doc.get("dutyStatus", "On duty"),
                "queueCount": queue_count,
                "slots": slot_items
            })

    return doctors_list


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@router.get("/patients")
def list_patients(
    search: Optional[str] = Query(None, description="Search by name, phone, MRN, reason or doctor"),
    status: Optional[str] = Query("ALL", description="Filter by status: ALL, UNASSIGNED, SCHEDULED, IN_CONSULTATION, COMPLETED"),
    db: Session = Depends(get_db)
):
    """
    List all hospital patients with optional search and status filtering.
    Administrator can see all new & returning patients.
    """
    ensure_initial_seed_data(db)

    query = db.query(PatientModel)

    if search:
        search_term = f"%{search.strip().lower()}%"
        query = query.filter(
            or_(
                PatientModel.name.ilike(search_term),
                PatientModel.id.ilike(search_term),
                PatientModel.pid.ilike(search_term),
                PatientModel.phone.ilike(search_term),
                PatientModel.reason.ilike(search_term),
                PatientModel.assigned_doctor_name.ilike(search_term),
            )
        )

    if status and status != "ALL":
        if status == "UNASSIGNED":
            query = query.filter(
                or_(
                    PatientModel.assigned_doctor_id.is_(None),
                    PatientModel.arrival_status == "ARRIVED"
                )
            )
        elif status == "SCHEDULED":
            query = query.filter(
                and_(
                    PatientModel.assigned_doctor_id.isnot(None),
                    PatientModel.arrival_status == "SCHEDULED"
                )
            )
        elif status == "IN_CONSULTATION":
            query = query.filter(PatientModel.arrival_status == "IN_CONSULTATION")
        elif status == "COMPLETED":
            query = query.filter(PatientModel.arrival_status == "COMPLETED")

    patients = query.order_by(desc(PatientModel.created_at)).all()

    # Calculate past session counts for each patient
    results = []
    for p in patients:
        session_count = db.query(PatientConsultationSession).filter(
            PatientConsultationSession.patient_id == p.id
        ).count()
        results.append(format_patient_dict(p, session_count=session_count))

    return {
        "success": True,
        "count": len(results),
        "data": results
    }


@router.get("/patients/{patient_id}")
def get_patient_detail(patient_id: str, db: Session = Depends(get_db)):
    """
    Retrieve single patient details, including all past consultation sessions & clinical records.
    """
    ensure_initial_seed_data(db)

    patient = db.query(PatientModel).filter(PatientModel.id == patient_id).first()
    if not patient:
        patient = db.query(PatientModel).filter(PatientModel.pid == patient_id).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient with ID '{patient_id}' not found.")

    sessions = db.query(PatientConsultationSession).filter(
        PatientConsultationSession.patient_id == patient.id
    ).order_by(PatientConsultationSession.session_number.asc()).all()

    session_list = []
    for s in sessions:
        conversation = []
        ai_suggestions = []
        clinical_analysis = {}
        soap_report = {}
        try:
            if s.conversation_json:
                conversation = json.loads(str(s.conversation_json))
            if s.ai_suggestions_json:
                ai_suggestions = json.loads(str(s.ai_suggestions_json))
            if s.clinical_analysis_json:
                clinical_analysis = json.loads(str(s.clinical_analysis_json))
            if s.soap_report_json:
                soap_report = json.loads(str(s.soap_report_json))
        except Exception as e:
            logger.warning(f"Error parsing session JSON for {s.id}: {e}")

        session_list.append({
            "id": s.id,
            "encounter_id": s.encounter_id,
            "session_number": s.session_number,
            "session_title": s.session_title,
            "doctor_id": s.doctor_id,
            "doctor_name": s.doctor_name,
            "appointment_date": s.appointment_date.isoformat() if s.appointment_date else None,
            "transcript": s.transcript,
            "conversation": conversation,
            "ai_suggestions": ai_suggestions,
            "clinical_analysis": clinical_analysis,
            "soap_report": soap_report,
            "duration_seconds": s.duration_seconds,
            "status": s.status,
            "consultation_mode": s.consultation_mode,
            "audio_source": s.audio_source,
            "gdrive_web_view_link": s.gdrive_web_view_link,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        })

    session_count = len(session_list)
    patient_dict = format_patient_dict(patient, session_count=session_count)
    patient_dict["sessions"] = session_list

    return {
        "success": True,
        "data": patient_dict
    }


@router.post("/patients")
def register_walkin_patient(payload: PatientCreateSchema, db: Session = Depends(get_db)):
    """
    Register a new walk-in patient at clinic front desk.
    Optionally assign a doctor & slot immediately, or leave unassigned in arriving triage queue.
    """
    ensure_initial_seed_data(db)

    id_num = f"{uuid.uuid4().int % 9000000 + 1000000}"
    new_id = f"#{id_num}"
    new_pid = f"P-{id_num}"

    initials = "".join([part[0] for part in payload.name.split() if part])[:2].upper() or "PT"
    vitals_dict = payload.vitals.model_dump() if payload.vitals else {
        "bp": "120/80 mmHg", "hr": "72 bpm", "pain": "0/10", "spo2": "98%", "temp": "36.6 °C"
    }

    new_patient = PatientModel(
        id=new_id,
        pid=new_pid,
        name=payload.name.strip(),
        initials=initials,
        age=payload.age,
        gender=payload.gender,
        phone=payload.phone.strip(),
        email=payload.email.strip().lower() if payload.email else None,
        demographics=f"{payload.age}{payload.gender} · {new_id}",
        arrival_time=datetime.now().strftime("%I:%M %p"),
        arrival_status="SCHEDULED" if payload.assigned_doctor_id else "ARRIVED",
        reason=payload.reason.strip(),
        condition=payload.condition.strip(),
        last_visit="—",
        vitals_json=json.dumps(vitals_dict),
        assigned_doctor_id=payload.assigned_doctor_id,
        assigned_doctor_name=payload.assigned_doctor_name,
        assigned_time_slot=payload.assigned_time_slot,
        assigned_duration="30 min",
        is_preferred_doctor_choice=payload.is_preferred_doctor_choice or False,
        preferred_doctor_name=payload.preferred_doctor_name,
        visit_type=payload.visit_type or "New Patient",
        booking_notes=payload.booking_notes or ("Front desk arrival registration"),
    )

    db.add(new_patient)

    email_clean = (payload.email or "").strip().lower()
    otp_generated = None

    if email_clean:
        existing_user = db.query(User).filter(User.email.ilike(email_clean)).first()
        if not existing_user:
            new_user = User(
                id=new_id,
                email=email_clean,
                full_name=payload.name.strip(),
                role="PATIENT",
                phone_number=payload.phone.strip(),
                avatar_initials=initials,
                is_active=True
            )
            db.add(new_user)
        else:
            existing_user.full_name = payload.name.strip()
            existing_user.phone_number = payload.phone.strip()

        otp_generated = f"{secrets.randbelow(900000) + 100000}"
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        otp_record = OTPVerification(
            email=email_clean,
            otp_code=otp_generated,
            role="PATIENT",
            purpose="REGISTRATION",
            expires_at=expires_at,
            is_used=False
        )
        db.add(otp_record)
        db.commit()

        try:
            send_otp_email(to_email=email_clean, otp_code=otp_generated, role="PATIENT", purpose="REGISTRATION")
        except Exception as e:
            logger.warning(f"Failed sending registration OTP email: {e}")

    if payload.assigned_doctor_id and payload.assigned_time_slot:
        slot = db.query(DoctorSlotModel).filter(
            DoctorSlotModel.doctor_id == payload.assigned_doctor_id,
            DoctorSlotModel.time_slot == payload.assigned_time_slot
        ).first()
        if slot:
            slot.status = "BOOKED"
            slot.booked_patient_id = new_id
            slot.booked_patient_name = payload.name.strip()

    db.commit()
    db.refresh(new_patient)

    session_count = db.query(PatientConsultationSession).filter(
        PatientConsultationSession.patient_id == new_patient.id
    ).count()

    return {
        "success": True,
        "message": f"Patient '{new_patient.name}' registered successfully.",
        "otpSent": bool(email_clean),
        "data": format_patient_dict(new_patient, session_count=session_count)
    }


@router.put("/patients/{patient_id}")
def update_patient(patient_id: str, payload: PatientUpdateSchema, db: Session = Depends(get_db)):
    """
    Update patient triage information, vitals, visit status, or personal profile.
    Syncs with User table when name or phone number changes.
    """
    clean_id = patient_id.strip()
    patient = db.query(PatientModel).filter(
        or_(
            PatientModel.id == clean_id,
            PatientModel.pid == clean_id,
            PatientModel.id == f"#{clean_id}",
            PatientModel.email.ilike(clean_id),
            PatientModel.name.ilike(f"%{clean_id}%")
        )
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    if payload.name is not None and payload.name.strip():
        patient.name = payload.name.strip()
        patient.initials = "".join([part[0] for part in patient.name.split() if part])[:2].upper() or "PT"
        if patient.email:
            u = db.query(User).filter(User.email.ilike(patient.email)).first()
            if u:
                u.full_name = patient.name
    if payload.age is not None:
        patient.age = payload.age
    if payload.gender is not None and payload.gender.strip():
        patient.gender = payload.gender.strip()
    if payload.phone is not None:
        patient.phone = payload.phone.strip()
        if patient.email:
            u = db.query(User).filter(User.email.ilike(patient.email)).first()
            if u:
                u.phone_number = patient.phone
    if payload.email is not None and payload.email.strip():
        patient.email = payload.email.strip().lower()
    if payload.arrival_status is not None:
        patient.arrival_status = payload.arrival_status
    if payload.reason is not None:
        patient.reason = payload.reason.strip()
    if payload.condition is not None:
        patient.condition = payload.condition.strip()
    if payload.last_visit is not None:
        patient.last_visit = payload.last_visit
    if payload.visit_type is not None:
        patient.visit_type = payload.visit_type
    if payload.booking_notes is not None:
        patient.booking_notes = payload.booking_notes
    if payload.is_preferred_doctor_choice is not None:
        patient.is_preferred_doctor_choice = payload.is_preferred_doctor_choice
    if payload.preferred_doctor_name is not None:
        patient.preferred_doctor_name = payload.preferred_doctor_name

    if payload.vitals is not None:
        patient.vitals_json = json.dumps(payload.vitals.model_dump())

    patient.demographics = f"{patient.age}{patient.gender} · {patient.id}"

    db.commit()
    db.refresh(patient)

    session_count = db.query(PatientConsultationSession).filter(
        PatientConsultationSession.patient_id == patient.id
    ).count()

    return {
        "success": True,
        "message": f"Patient '{patient.id}' updated successfully.",
        "data": format_patient_dict(patient, session_count=session_count)
    }


@router.delete("/patients/{patient_id}")
def delete_patient(patient_id: str, db: Session = Depends(get_db)):
    """
    Remove or discharge a patient and free their booked slot.
    """
    patient = db.query(PatientModel).filter(PatientModel.id == patient_id).first()
    if not patient:
        patient = db.query(PatientModel).filter(PatientModel.pid == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    if patient.assigned_doctor_id and patient.assigned_time_slot:
        slot = db.query(DoctorSlotModel).filter(
            DoctorSlotModel.doctor_id == patient.assigned_doctor_id,
            DoctorSlotModel.time_slot == patient.assigned_time_slot
        ).first()
        if slot and slot.booked_patient_id == patient.id:
            slot.status = "AVAILABLE"
            slot.booked_patient_id = None
            slot.booked_patient_name = None

    db.delete(patient)
    db.commit()

    return {
        "success": True,
        "message": f"Patient '{patient_id}' deleted successfully."
    }


# ==============================================================================
# DOCTORS & SLOTS MANAGEMENT
# ==============================================================================

@router.get("/doctors")
def list_doctors_with_slots(db: Session = Depends(get_db)):
    """
    Get full list of hospital doctors (including newly registered doctors from database)
    with their real-time timing slots and availability.
    """
    doctors = get_all_doctors_from_db(db)
    return {
        "success": True,
        "count": len(doctors),
        "data": doctors
    }


@router.post("/doctors")
def register_new_doctor(payload: DoctorCreateSchema, db: Session = Depends(get_db)):
    """
    Register a new upcoming or attending doctor into the hospital roster with auto-generated timing slots.
    """
    ensure_initial_seed_data(db)

    clean_name = payload.name.strip()
    formatted_name = clean_name if (clean_name.lower().startswith("dr.") or clean_name.lower().startswith("dr ")) else f"Dr. {clean_name}"
    initials = "".join([part[0] for part in formatted_name.replace("Dr.", "").strip().split() if part])[:2].upper() or "DR"
    
    # Generate doctor ID
    doc_id = f"DOC-{uuid.uuid4().int % 900 + 100}"
    email_clean = payload.email.strip().lower()

    # Add to DEFAULT_DOCTORS in-memory list
    new_doc_dict: Dict[str, Any] = {
        "id": doc_id,
        "name": formatted_name,
        "initials": initials,
        "specialty": payload.specialty.strip(),
        "department": payload.department.strip(),
        "email": email_clean,
        "phone": payload.phone or "+1 (555) 000-0000",
        "room": payload.room or "Suite 101",
        "experience": payload.experience or ("New Upcoming Specialist" if payload.is_new_upcoming else "Attending Specialist"),
        "qualification": payload.qualification or "MD, MBBS",
        "isNewUpcoming": bool(payload.is_new_upcoming),
        "shiftTiming": payload.shift_timing or "09:00 AM - 05:00 PM",
        "available": payload.duty_status == "On duty",
        "dutyStatus": payload.duty_status or "On duty",
    }
    DEFAULT_DOCTORS.insert(0, new_doc_dict)

    # Check or create User table record for login
    existing_user = db.query(User).filter(User.email.ilike(email_clean)).first()
    if not existing_user:
        new_user = User(
            id=doc_id,
            email=email_clean,
            full_name=formatted_name,
            role="DOCTOR",
            department=payload.department.strip(),
            specialization=payload.specialty.strip(),
            phone_number=payload.phone or "+1 (555) 000-0000",
            avatar_initials=initials,
            is_active=True
        )
        db.add(new_user)
        db.commit()

    # Generate standard slots for the new doctor
    slot_duration = payload.slot_duration or "30 min"
    standard_times = ["09:00 AM", "09:45 AM", "10:30 AM", "11:15 AM", "02:00 PM", "02:45 PM", "03:30 PM", "04:15 PM"]
    new_slots = []
    for idx, time_str in enumerate(standard_times):
        slot_obj = DoctorSlotModel(
            id=f"{doc_id}-s{idx+1}",
            doctor_id=doc_id,
            doctor_name=formatted_name,
            time_slot=time_str,
            duration=slot_duration,
            status="AVAILABLE"
        )
        new_slots.append(slot_obj)
    db.add_all(new_slots)
    db.commit()

    slot_items = [
        {
            "id": s.id,
            "time": s.time_slot,
            "duration": s.duration,
            "status": s.status,
            "bookedPatientId": s.booked_patient_id,
            "bookedPatientName": s.booked_patient_name,
        }
        for s in new_slots
    ]

    new_doc_dict["slots"] = slot_items
    new_doc_dict["queueCount"] = 0

    return {
        "success": True,
        "message": f"Doctor {formatted_name} registered successfully with {len(new_slots)} consultation slots.",
        "data": new_doc_dict
    }


@router.put("/doctors/{doctor_id}")
def update_doctor(doctor_id: str, payload: DoctorUpdateSchema, db: Session = Depends(get_db)):
    """
    Update doctor details, room, shift timings, duty status, and specialization.
    """
    clean_id = doctor_id.strip().lower()
    
    # Update in DEFAULT_DOCTORS
    matched_doc = None
    for doc in DEFAULT_DOCTORS:
        if doc["id"].lower() == clean_id:
            matched_doc = doc
            break

    if matched_doc:
        if payload.name is not None and payload.name.strip():
            matched_doc["name"] = payload.name.strip()
            matched_doc["initials"] = "".join([p[0] for p in matched_doc["name"].replace("Dr.", "").strip().split() if p])[:2].upper() or "DR"
        if payload.specialty is not None:
            matched_doc["specialty"] = payload.specialty.strip()
        if payload.department is not None:
            matched_doc["department"] = payload.department.strip()
        if payload.phone is not None:
            matched_doc["phone"] = payload.phone.strip()
        if payload.room is not None:
            matched_doc["room"] = payload.room.strip()
        if payload.experience is not None:
            matched_doc["experience"] = payload.experience.strip()
        if payload.qualification is not None:
            matched_doc["qualification"] = payload.qualification.strip()
        if payload.is_new_upcoming is not None:
            matched_doc["isNewUpcoming"] = payload.is_new_upcoming
        if payload.duty_status is not None:
            matched_doc["dutyStatus"] = payload.duty_status
            matched_doc["available"] = (payload.duty_status == "On duty")
        if payload.available is not None:
            matched_doc["available"] = payload.available
            matched_doc["dutyStatus"] = "On duty" if payload.available else "Off duty"
        if payload.shift_timing is not None:
            matched_doc["shiftTiming"] = payload.shift_timing.strip()

    # Also update in User table if present
    u = db.query(User).filter(User.id.ilike(clean_id)).first()
    if u:
        if payload.name is not None and payload.name.strip():
            u.full_name = payload.name.strip()
        if payload.department is not None:
            u.department = payload.department.strip()
        if payload.specialty is not None:
            u.specialization = payload.specialty.strip()
        if payload.phone is not None:
            u.phone_number = payload.phone.strip()
        db.commit()

    # Re-fetch doctor profile
    all_docs = get_all_doctors_from_db(db)
    updated_doc = next((d for d in all_docs if d["id"].lower() == clean_id), matched_doc)

    return {
        "success": True,
        "message": f"Doctor {doctor_id} updated successfully.",
        "data": updated_doc
    }


@router.post("/doctors/{doctor_id}/toggle-duty")
def toggle_doctor_duty(doctor_id: str, db: Session = Depends(get_db)):
    """
    Toggle doctor availability and duty status between 'On duty' and 'Off duty'.
    """
    clean_id = doctor_id.strip().lower()
    new_status = "On duty"
    new_avail = True

    for doc in DEFAULT_DOCTORS:
        if doc["id"].lower() == clean_id:
            if doc.get("dutyStatus") == "On duty":
                doc["dutyStatus"] = "Off duty"
                doc["available"] = False
            else:
                doc["dutyStatus"] = "On duty"
                doc["available"] = True
            new_status = doc["dutyStatus"]
            new_avail = doc["available"]
            break

    return {
        "success": True,
        "doctorId": doctor_id,
        "dutyStatus": new_status,
        "available": new_avail,
        "message": f"Duty status updated to '{new_status}'."
    }


@router.post("/doctors/{doctor_id}/slots")
def add_doctor_slot(doctor_id: str, payload: DoctorSlotCreateSchema, db: Session = Depends(get_db)):
    """
    Add a new custom timing slot for a clinician.
    """
    ensure_initial_seed_data(db)
    all_docs = get_all_doctors_from_db(db)
    doc = next((d for d in all_docs if d["id"].lower() == doctor_id.lower()), None)
    doc_name = doc["name"] if doc else f"Doctor {doctor_id}"

    slot_id = f"{doctor_id}-{payload.time_slot.replace(' ', '').replace(':', '')}"
    
    # Check if already exists
    existing = db.query(DoctorSlotModel).filter(DoctorSlotModel.id == slot_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Slot at {payload.time_slot} already exists for this doctor.")

    new_slot = DoctorSlotModel(
        id=slot_id,
        doctor_id=doctor_id,
        doctor_name=doc_name,
        time_slot=payload.time_slot,
        duration=payload.duration or "30 min",
        status=payload.status or "AVAILABLE"
    )
    db.add(new_slot)
    db.commit()
    db.refresh(new_slot)

    return {
        "success": True,
        "message": f"Timing slot {payload.time_slot} added successfully.",
        "data": {
            "id": new_slot.id,
            "time": new_slot.time_slot,
            "duration": new_slot.duration,
            "status": new_slot.status
        }
    }


@router.delete("/doctors/{doctor_id}/slots/{slot_id}")
def delete_doctor_slot(doctor_id: str, slot_id: str, db: Session = Depends(get_db)):
    """
    Remove a timing slot for a clinician.
    """
    slot = db.query(DoctorSlotModel).filter(
        and_(
            DoctorSlotModel.doctor_id == doctor_id,
            DoctorSlotModel.id == slot_id
        )
    ).first()

    if not slot:
        raise HTTPException(status_code=404, detail=f"Slot '{slot_id}' not found for doctor '{doctor_id}'.")

    if slot.status == "BOOKED":
        raise HTTPException(status_code=400, detail=f"Cannot delete slot '{slot.time_slot}' because it is booked by {slot.booked_patient_name}. Reassign patient first.")

    db.delete(slot)
    db.commit()

    return {
        "success": True,
        "message": f"Slot '{slot_id}' deleted successfully."
    }


@router.delete("/doctors/{doctor_id}")
def delete_doctor(doctor_id: str, db: Session = Depends(get_db)):
    """
    Delete or archive a doctor profile from roster.
    """
    clean_id = doctor_id.strip().lower()
    global DEFAULT_DOCTORS
    DEFAULT_DOCTORS = [d for d in DEFAULT_DOCTORS if d["id"].lower() != clean_id]

    # Delete slots
    db.query(DoctorSlotModel).filter(DoctorSlotModel.doctor_id == doctor_id).delete()
    
    # Delete user if exists
    u = db.query(User).filter(User.id == doctor_id).first()
    if u:
        db.delete(u)

    db.commit()

    return {
        "success": True,
        "message": f"Doctor '{doctor_id}' removed from hospital roster."
    }


@router.get("/doctors/{doctor_id}/slots")
def get_doctor_slots(doctor_id: str, db: Session = Depends(get_db)):
    """
    Get timing slots for a specific doctor.
    """
    ensure_initial_seed_data(db)

    slots = db.query(DoctorSlotModel).filter(
        DoctorSlotModel.doctor_id == doctor_id
    ).all()

    if not slots:
        # Trigger get_all_doctors_from_db to seed slots if newly registered doctor
        get_all_doctors_from_db(db)
        slots = db.query(DoctorSlotModel).filter(
            DoctorSlotModel.doctor_id == doctor_id
        ).all()

    if not slots:
        raise HTTPException(status_code=404, detail=f"No slots found for doctor ID '{doctor_id}'.")

    return {
        "success": True,
        "doctorId": doctor_id,
        "data": [
            {
                "id": s.id,
                "time": s.time_slot,
                "duration": s.duration,
                "status": s.status,
                "bookedPatientId": s.booked_patient_id,
                "bookedPatientName": s.booked_patient_name,
            }
            for s in slots
        ]
    }


@router.post("/assign")
def assign_doctor_and_slot(payload: AssignDoctorSlotSchema, db: Session = Depends(get_db)):
    """
    Administrator assigns or distributes a doctor and timing slot to a patient.
    Handles preferred doctor choice of patient or administrative slot allocation across all registered doctors.
    """
    ensure_initial_seed_data(db)

    patient = db.query(PatientModel).filter(PatientModel.id == payload.patient_id).first()
    if not patient:
        patient = db.query(PatientModel).filter(PatientModel.pid == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{payload.patient_id}' not found.")

    all_docs = get_all_doctors_from_db(db)
    doc = next((d for d in all_docs if d["id"] == payload.doctor_id or d["name"].lower() == payload.doctor_id.lower()), None)
    if not doc and payload.doctor_name:
        doc = next((d for d in all_docs if d["name"].lower() == payload.doctor_name.lower() or payload.doctor_name.lower() in d["name"].lower()), None)

    if not doc:
        u = db.query(User).filter(and_(User.id == payload.doctor_id, User.role == "DOCTOR")).first()
        if u:
            doc_name = u.full_name if (u.full_name and u.full_name.startswith("Dr.")) else f"Dr. {u.full_name or 'Doctor'}"
            doc = {
                "id": u.id,
                "name": doc_name,
                "specialty": u.specialization or u.department or "General Physician"
            }

    if not doc:
        raise HTTPException(status_code=404, detail=f"Doctor '{payload.doctor_id}' not found.")

    slot = db.query(DoctorSlotModel).filter(
        DoctorSlotModel.doctor_id == doc["id"],
        DoctorSlotModel.time_slot == payload.time_slot
    ).first()

    if not slot:
        slot = DoctorSlotModel(
            id=f"{doc['id']}-{payload.time_slot.replace(' ', '').replace(':', '')}",
            doctor_id=doc["id"],
            doctor_name=doc["name"],
            time_slot=payload.time_slot,
            duration=payload.duration or "30 min",
            status="AVAILABLE"
        )
        db.add(slot)
        db.commit()
        db.refresh(slot)

    if slot.status == "BOOKED" and slot.booked_patient_id and slot.booked_patient_id != patient.id:
        raise HTTPException(
            status_code=400,
            detail=f"Slot '{payload.time_slot}' is already booked for another patient ({slot.booked_patient_name}). Please select another available slot."
        )

    # Release previously booked slot
    if patient.assigned_doctor_id and patient.assigned_time_slot:
        prev_slot = db.query(DoctorSlotModel).filter(
            DoctorSlotModel.doctor_id == patient.assigned_doctor_id,
            DoctorSlotModel.time_slot == patient.assigned_time_slot
        ).first()
        if prev_slot and prev_slot.id != slot.id and prev_slot.booked_patient_id == patient.id:
            prev_slot.status = "AVAILABLE"
            prev_slot.booked_patient_id = None
            prev_slot.booked_patient_name = None

    # Book the new slot
    slot.status = "BOOKED"
    slot.booked_patient_id = patient.id
    slot.booked_patient_name = patient.name
    slot.doctor_name = doc["name"]

    # Update patient record
    patient.assigned_doctor_id = doc["id"]
    patient.assigned_doctor_name = doc["name"]
    patient.assigned_time_slot = payload.time_slot
    patient.assigned_duration = payload.duration or slot.duration
    patient.arrival_status = "SCHEDULED"
    patient.is_preferred_doctor_choice = payload.is_preferred_choice
    patient.preferred_doctor_name = (payload.preferred_doctor_name or doc["name"]) if payload.is_preferred_choice else None
    
    if payload.visit_type:
        patient.visit_type = payload.visit_type
    
    if payload.booking_notes:
        patient.booking_notes = payload.booking_notes
    else:
        patient.booking_notes = f"{'Preferred doctor choice: ' if payload.is_preferred_choice else 'Allocated to '}{doc['name']} at {payload.time_slot} ({slot.duration})."

    db.commit()
    db.refresh(patient)

    session_count = db.query(PatientConsultationSession).filter(
        PatientConsultationSession.patient_id == patient.id
    ).count()

    return {
        "success": True,
        "message": f"Successfully assigned {patient.name} ({patient.id}) to {doc['name']} for {payload.time_slot}{' [Patient Preferred Choice]' if payload.is_preferred_choice else ''}.",
        "data": format_patient_dict(patient, session_count=session_count)
    }


@router.get("/queue/{doctor_id_or_name}")
def get_doctor_queue(doctor_id_or_name: str, db: Session = Depends(get_db)):
    """
    Dynamic queue for Doctor's screen.
    Returns patient list assigned to this doctor via admin distribution or patient preferred booking.
    Works for default doctors as well as newly registered doctors.
    """
    ensure_initial_seed_data(db)
    all_docs = get_all_doctors_from_db(db)

    clean_term = doctor_id_or_name.strip().lower()
    matched_doc = None

    for doc in all_docs:
        if doc["id"].lower() == clean_term:
            matched_doc = doc
            break

    if not matched_doc:
        for doc in all_docs:
            if (
                clean_term in doc["name"].lower()
                or doc["name"].lower() in clean_term
                or (doc.get("email") and clean_term in doc["email"].lower())
            ):
                matched_doc = doc
                break

    if not matched_doc:
        u = db.query(User).filter(
            or_(
                User.id.ilike(f"%{clean_term}%"),
                User.full_name.ilike(f"%{clean_term}%"),
                User.email.ilike(f"%{clean_term}%")
            ),
            User.role == "DOCTOR"
        ).first()
        if u:
            doc_name = u.full_name if (u.full_name and u.full_name.startswith("Dr.")) else f"Dr. {u.full_name or 'Doctor'}"
            matched_doc = {
                "id": u.id,
                "name": doc_name,
                "email": u.email
            }

    doc_id = matched_doc["id"] if matched_doc else doctor_id_or_name
    doc_name = matched_doc["name"] if matched_doc else doctor_id_or_name
    clean_name = doc_name.replace("Dr.", "").strip()

    patients = db.query(PatientModel).filter(
        or_(
            PatientModel.assigned_doctor_id == doc_id,
            PatientModel.assigned_doctor_name.ilike(f"%{clean_name}%"),
            PatientModel.assigned_doctor_name.ilike(f"%{doc_name}%"),
            and_(
                PatientModel.preferred_doctor_name.ilike(f"%{clean_name}%"),
                PatientModel.is_preferred_doctor_choice == True
            )
        )
    ).order_by(PatientModel.assigned_time_slot.asc()).all()

    queue_items = []
    for p in patients:
        session_count = db.query(PatientConsultationSession).filter(
            PatientConsultationSession.patient_id == p.id
        ).count()
        queue_items.append(format_patient_dict(p, session_count=session_count))

    return {
        "success": True,
        "doctorId": doc_id,
        "doctorName": doc_name,
        "queueCount": len(queue_items),
        "data": queue_items
    }


# ==============================================================================
# PATIENT SESSIONS & ENCOUNTER HISTORY
# ==============================================================================

@router.get("/patients/{patient_id}/sessions")
def get_patient_sessions(patient_id: str, db: Session = Depends(get_db)):
    """
    Get all past consultation sessions & encounters for a specific patient.
    """
    ensure_initial_seed_data(db)

    clean_id = patient_id if patient_id.startswith("#") else f"#{patient_id}"
    raw_id = patient_id.replace("#", "")
    sessions = db.query(PatientConsultationSession).filter(
        (PatientConsultationSession.patient_id == patient_id) |
        (PatientConsultationSession.patient_id == clean_id) |
        (PatientConsultationSession.patient_id == raw_id)
    ).order_by(PatientConsultationSession.session_number.asc()).all()

    formatted_sessions = []
    for s in sessions:
        conversation = []
        ai_suggestions = []
        clinical_analysis = {}
        soap_report = {}
        try:
            if s.conversation_json:
                conversation = json.loads(str(s.conversation_json))
            if s.ai_suggestions_json:
                ai_suggestions = json.loads(str(s.ai_suggestions_json))
            if s.clinical_analysis_json:
                clinical_analysis = json.loads(str(s.clinical_analysis_json))
            if s.soap_report_json:
                soap_report = json.loads(str(s.soap_report_json))
        except Exception:
            pass

        formatted_sessions.append({
            "id": s.id,
            "patientId": s.patient_id,
            "patientName": s.patient_name,
            "doctorId": s.doctor_id,
            "doctorName": s.doctor_name,
            "encounterId": s.encounter_id,
            "sessionNumber": s.session_number,
            "sessionTitle": s.session_title,
            "appointmentDate": s.appointment_date.isoformat() if s.appointment_date else None,
            "transcript": s.transcript,
            "conversation": conversation,
            "aiSuggestions": ai_suggestions,
            "clinicalAnalysis": clinical_analysis,
            "soapReport": soap_report,
            "durationSeconds": s.duration_seconds,
            "status": s.status,
            "consultationMode": s.consultation_mode,
            "audioSource": s.audio_source,
            "gdriveWebViewLink": s.gdrive_web_view_link,
            "createdAt": s.created_at.isoformat() if s.created_at else None,
        })

    return {
        "success": True,
        "patientId": patient_id,
        "count": len(formatted_sessions),
        "data": formatted_sessions
    }


@router.post("/patients/{patient_id}/sessions")
def create_patient_session(
    patient_id: str,
    payload: CreateConsultationSessionSchema,
    db: Session = Depends(get_db)
):
    """
    Create / finalize a new consultation session for a patient.
    """
    ensure_initial_seed_data(db)

    patient = db.query(PatientModel).filter(PatientModel.id == patient_id).first()
    if not patient:
        patient = db.query(PatientModel).filter(PatientModel.pid == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    latest_session = db.query(PatientConsultationSession).filter(
        PatientConsultationSession.patient_id == patient.id
    ).order_by(PatientConsultationSession.session_number.desc()).first()

    next_num = (latest_session.session_number + 1) if latest_session else 1
    encounter_id = payload.encounter_id or f"CONS-{patient.id.replace('#', '')}-{next_num}"

    conv_json = json.dumps(payload.conversation) if payload.conversation else None
    sug_json = json.dumps(payload.ai_suggestions) if payload.ai_suggestions else None
    analysis_json = json.dumps(payload.clinical_analysis) if payload.clinical_analysis else None
    soap_json = json.dumps(payload.soap_report) if payload.soap_report else None

    new_session = PatientConsultationSession(
        patient_id=patient.id,
        patient_name=patient.name,
        doctor_id=payload.doctor_id or patient.assigned_doctor_id or "DOC-001",
        doctor_name=payload.doctor_name or patient.assigned_doctor_name or "Dr. Yuki Tanaka",
        encounter_id=encounter_id,
        session_number=next_num,
        session_title=payload.session_title or f"Session {next_num}: Clinical Consultation",
        appointment_date=datetime.now(timezone.utc),
        transcript=payload.transcript,
        conversation_json=conv_json,
        ai_suggestions_json=sug_json,
        clinical_analysis_json=analysis_json,
        soap_report_json=soap_json,
        duration_seconds=payload.duration_seconds or 0,
        consultation_mode=payload.consultation_mode or "IN_PERSON",
        audio_source=payload.audio_source or "MIC",
        status=payload.status or "COMPLETED"
    )

    db.add(new_session)

    patient.last_visit = datetime.now().strftime("%B %d, %Y")
    if payload.status == "COMPLETED":
        patient.arrival_status = "COMPLETED"
    else:
        patient.arrival_status = "IN_CONSULTATION"

    db.commit()
    db.refresh(new_session)

    return {
        "success": True,
        "message": f"Consultation session #{new_session.session_number} saved for patient {patient.name}.",
        "data": {
            "id": new_session.id,
            "sessionNumber": new_session.session_number,
            "sessionTitle": new_session.session_title,
            "encounterId": new_session.encounter_id,
            "status": new_session.status
        }
    }


@router.post("/patients/{patient_id}/share-soap")
def share_soap_report_with_patient(
    patient_id: str,
    payload: ShareSoapReportSchema,
    db: Session = Depends(get_db)
):
    """
    Doctor explicitly finalizes and shares SOAP report with the patient.
    This dynamically updates the patient portal with Follow-up Alerts, Medical Records, and Prescriptions (with SOAP note).
    """
    ensure_initial_seed_data(db)

    # Resolve patient
    clean_id = patient_id.strip()
    patient = db.query(PatientModel).filter(
        or_(
            PatientModel.id == clean_id,
            PatientModel.pid == clean_id,
            PatientModel.id == f"#{clean_id}",
            PatientModel.name.ilike(f"%{clean_id}%")
        )
    ).first()

    if not patient and payload.patient_name:
        patient = db.query(PatientModel).filter(
            PatientModel.name.ilike(f"%{payload.patient_name.strip()}%")
        ).first()

    if not patient and ("patient" in clean_id.lower() or "@" in clean_id):
        patient = db.query(PatientModel).filter(PatientModel.id == "#00847261").first()

    if not patient:
        patient = db.query(PatientModel).first()

    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    now_utc = datetime.now(timezone.utc)
    
    # Embed prescription and patient demographics into soap_report dict if present
    if payload.prescription:
        payload.soap_report["prescription"] = payload.prescription
    if payload.patient_name or patient.name:
        payload.soap_report["patientName"] = payload.patient_name or patient.name
    payload.soap_report["patientId"] = patient.id
    if payload.doctor_name:
        payload.soap_report["doctorName"] = payload.doctor_name
    if payload.encounter_id:
        payload.soap_report["encounterId"] = payload.encounter_id

    soap_json_str = json.dumps(payload.soap_report)

    # Check if a session already exists for this patient or encounter
    existing_session = None
    if payload.encounter_id:
        existing_session = db.query(PatientConsultationSession).filter(
            PatientConsultationSession.encounter_id == payload.encounter_id
        ).first()

    if not existing_session:
        existing_session = db.query(PatientConsultationSession).filter(
            or_(
                PatientConsultationSession.patient_id == patient.id,
                PatientConsultationSession.patient_id == patient.pid,
                PatientConsultationSession.patient_name.ilike(patient.name)
            )
        ).order_by(PatientConsultationSession.session_number.desc()).first()

    if existing_session:
        existing_session.soap_report_json = soap_json_str
        existing_session.is_shared_with_patient = True
        existing_session.shared_at = now_utc
        if payload.doctor_id:
            existing_session.doctor_id = payload.doctor_id
        if payload.doctor_name:
            existing_session.doctor_name = payload.doctor_name
        if payload.encounter_id:
            existing_session.encounter_id = payload.encounter_id
        session_obj = existing_session
    else:
        session_obj = PatientConsultationSession(
            patient_id=patient.id,
            patient_name=patient.name,
            doctor_id=payload.doctor_id or patient.assigned_doctor_id or "DOC-001",
            doctor_name=payload.doctor_name or patient.assigned_doctor_name or "Dr. Yuki Tanaka",
            encounter_id=payload.encounter_id or f"CONS-{patient.id.replace('#', '')}-1",
            session_number=1,
            session_title=f"Session 1: Clinical Consultation & SOAP Report",
            appointment_date=now_utc,
            transcript="",
            conversation_json=None,
            ai_suggestions_json=None,
            clinical_analysis_json=None,
            soap_report_json=soap_json_str,
            duration_seconds=900,
            consultation_mode="IN_PERSON",
            audio_source="MIC",
            status="COMPLETED",
            is_shared_with_patient=True,
            shared_at=now_utc
        )
        db.add(session_obj)

    patient.arrival_status = "COMPLETED"
    patient.last_visit = datetime.now().strftime("%B %d, %Y")

    db.commit()
    db.refresh(session_obj)

    logger.info(f"Doctor shared SOAP report for patient {patient.name} (MRN: {patient.id})")

    return {
        "success": True,
        "message": f"SOAP report successfully shared with patient {patient.name}.",
        "sharedAt": session_obj.shared_at.isoformat() if session_obj.shared_at else now_utc.isoformat(),
        "sessionId": session_obj.id,
        "encounterId": session_obj.encounter_id,
        "patientId": patient.id,
        "patientName": patient.name
    }


@router.post("/patients/{patient_id}/complete-consultation")
@router.post("/patients/{patient_id}/close-consultation")
def complete_patient_consultation(
    patient_id: str,
    payload: Optional[CompleteConsultationSchema] = None,
    db: Session = Depends(get_db)
):
    """
    Close and complete the consultation session for a patient.
    - Sets arrival_status to COMPLETED
    - Updates last_visit date to current date (e.g. 'September 12, 2026')
    - Updates consultation session status to COMPLETED and records active doctor name & ID
    """
    ensure_initial_seed_data(db)

    clean_id = patient_id.strip()
    patient = db.query(PatientModel).filter(
        or_(
            PatientModel.id == clean_id,
            PatientModel.pid == clean_id,
            PatientModel.id == f"#{clean_id}",
            PatientModel.name.ilike(f"%{clean_id}%")
        )
    ).first()

    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")

    today_str = datetime.now().strftime("%B %d, %Y")
    patient.arrival_status = "COMPLETED"
    patient.last_visit = today_str

    doc_id = payload.doctor_id if (payload and payload.doctor_id) else (patient.assigned_doctor_id or "DOC-001")
    doc_name = payload.doctor_name if (payload and payload.doctor_name) else (patient.assigned_doctor_name or "Dr. Yuki Tanaka")

    latest_sess = db.query(PatientConsultationSession).filter(
        (PatientConsultationSession.patient_id == patient.id) |
        (PatientConsultationSession.patient_id == clean_id)
    ).order_by(PatientConsultationSession.session_number.desc()).first()

    if latest_sess:
        latest_sess.status = "COMPLETED"
        if doc_id:
            latest_sess.doctor_id = doc_id
        if doc_name:
            latest_sess.doctor_name = doc_name

    db.commit()
    db.refresh(patient)

    session_count = db.query(PatientConsultationSession).filter(
        PatientConsultationSession.patient_id == patient.id
    ).count()

    logger.info(f"Consultation completed for patient {patient.name} ({patient.id}) with {doc_name}. Last visit set to {today_str}.")

    return {
        "success": True,
        "message": f"Consultation session completed for {patient.name}. Last visit updated to {today_str}.",
        "lastVisit": today_str,
        "data": format_patient_dict(patient, session_count=session_count)
    }


def _extract_prescriptions_from_soap(soap_dict: Dict[str, Any], doctor_name: str, date_str: str) -> List[Dict[str, Any]]:
    """Helper to extract structured prescriptions from SOAP plan & explicit doctor prescription directives."""
    plan_text = soap_dict.get("plan", "") or ""
    prescriptions = []
    rx_id_counter = 1
    existing_med_names = set()

    # 1. Prioritize explicit doctor prescription directive if present
    custom_rx = soap_dict.get("prescription")
    if custom_rx and isinstance(custom_rx, dict) and custom_rx.get("medication"):
        med = custom_rx.get("medication", "").strip()
        strength = custom_rx.get("strength", "").strip()
        route = custom_rx.get("route", "").strip()
        freq = custom_rx.get("frequency", "").strip()
        dur = custom_rx.get("duration", "").strip()
        qty = custom_rx.get("quantity", "").strip()
        refills = custom_rx.get("refills", "0")
        inst = custom_rx.get("instructions", "").strip() or f"Take {strength} {route} {freq} for {dur}."

        prescriptions.append({
            "id": f"rx-prescribed-{rx_id_counter}",
            "title": f"{med} — {strength}",
            "name": med,
            "dosage": f"{strength} · {route}",
            "badge": "active",
            "statusClass": "active",
            "doctorDate": f"{doctor_name} · Issued {date_str}",
            "instructions": inst,
            "refillInfo": f"{qty} · {refills} refill(s)",
            "soapReport": soap_dict,
            "prescription": custom_rx
        })
        rx_id_counter += 1
        existing_med_names.add(med.lower())

    # Common medication patterns
    med_pattern = re.compile(r'([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+\s*(?:mg|mcg|g|ml|IU))\s*([^\.\n\d;]*)', re.IGNORECASE)
    matches = med_pattern.findall(plan_text)

    if matches:
        for match in matches:
            med_name = match[0].strip()
            # Filter out non-medication phrases and duplicates
            if med_name.lower() in ["schedule", "recommend", "encourage", "systolic", "diastolic", "follow up", "follow-up"] or med_name.lower() in existing_med_names:
                continue
            dosage = match[1].strip()
            instructions = match[2].strip() or "As directed by physician"
            # Format clean instruction
            clean_inst = instructions.strip(" -:.,")
            if not clean_inst:
                clean_inst = "Take orally as prescribed"
            else:
                clean_inst = clean_inst[0].upper() + clean_inst[1:]

            prescriptions.append({
                "id": f"rx-{rx_id_counter}",
                "title": f"{med_name} — {dosage}",
                "name": med_name,
                "dosage": dosage,
                "badge": "active",
                "statusClass": "active",
                "doctorDate": f"{doctor_name} · Issued {date_str}",
                "instructions": clean_inst,
                "refillInfo": "3 refills left",
                "soapReport": soap_dict
            })
            rx_id_counter += 1

    # If regex did not catch individual meds, parse numbered lines from plan
    if not prescriptions and plan_text:
        lines = [line.strip() for line in re.split(r'\d+\.\s*', plan_text) if line.strip()]
        for line in lines:
            if any(k in line.lower() for k in ["mg", "tablet", "capsule", "daily", "lisinopril", "amlodipine", "atorvastatin", "metformin"]):
                prescriptions.append({
                    "id": f"rx-{rx_id_counter}",
                    "title": line.split('.')[0][:40],
                    "name": line.split('.')[0][:30],
                    "dosage": "As prescribed",
                    "badge": "active",
                    "statusClass": "active",
                    "doctorDate": f"{doctor_name} · Issued {date_str}",
                    "instructions": line,
                    "refillInfo": "3 refills left",
                    "soapReport": soap_dict
                })
                rx_id_counter += 1

    return prescriptions


@router.get("/patients/{patient_id_or_email}/portal-summary")
def get_patient_portal_summary(
    patient_id_or_email: str,
    db: Session = Depends(get_db)
):
    """
    Fetch comprehensive patient health summary including Follow-up Alerts, Medical Records,
    and Prescriptions (with full doctor-shared SOAP notes).
    """
    ensure_initial_seed_data(db)

    clean_target = patient_id_or_email.strip()

    # 1. Match User first by ID, email, or full name
    u = db.query(User).filter(
        or_(
            User.id == clean_target,
            User.email.ilike(clean_target),
            User.full_name.ilike(f"%{clean_target}%")
        )
    ).first()

    patient = None
    if u:
        match_conditions = [
            PatientModel.id == u.id,
            PatientModel.pid == u.id,
            PatientModel.id == f"#{u.id}"
        ]
        if u.email:
            match_conditions.append(PatientModel.email.ilike(u.email))
        if u.phone_number:
            match_conditions.append(PatientModel.phone == u.phone_number)
        if u.full_name:
            match_conditions.append(PatientModel.name.ilike(u.full_name))

        patient = db.query(PatientModel).filter(or_(*match_conditions)).first()

    if not patient:
        patient = db.query(PatientModel).filter(
            or_(
                PatientModel.id == clean_target,
                PatientModel.pid == clean_target,
                PatientModel.id == f"#{clean_target}",
                PatientModel.phone == clean_target,
                PatientModel.email.ilike(clean_target),
                PatientModel.name.ilike(f"%{clean_target}%")
            )
        ).first()

    # Auto-provision PatientModel if User exists but PatientModel not created yet
    if not patient and u and u.role == "PATIENT":
        id_num = f"{uuid.uuid4().int % 9000000 + 1000000}"
        pat_id = f"#{id_num}"
        pat_name = u.full_name or clean_target.split('@')[0].replace('.', ' ').title()
        initials = "".join([part[0] for part in pat_name.split() if part])[:2].upper() or "PT"
        patient = PatientModel(
            id=pat_id,
            pid=f"P-{id_num}",
            name=pat_name,
            initials=initials,
            age=35,
            gender="M",
            phone=u.phone_number or "+1 (555) 000-0000",
            email=u.email,
            demographics=f"35M · {pat_id}",
            arrival_time=datetime.now().strftime("%I:%M %p"),
            arrival_status="ARRIVED",
            reason="General Consultation",
            condition="Triage Assessment",
            last_visit="—",
            vitals_json=json.dumps({
                "bp": "120/80 mmHg",
                "hr": "72 bpm",
                "pain": "0/10",
                "spo2": "98%",
                "temp": "36.6 °C"
            }),
            visit_type="New Patient",
            booking_notes="Self-registered via Patient Portal."
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # Fallback to Marcus Delgado only if no patient found and requested 'demo' or 'marcus'
    if not patient and ("marcus" in clean_target.lower() or clean_target in ["1", "demo"]):
        patient = db.query(PatientModel).filter(PatientModel.id == "#00847261").first()

    if not patient:
        patient = db.query(PatientModel).first()

    if not patient:
        raise HTTPException(status_code=404, detail="Patient records not found.")

    # Fetch all consultation sessions for this patient (by ID, PID, clean ID, patient name, or user ID)
    search_ids = [patient.id, patient.pid, patient.id.replace('#', '')]
    if u:
        search_ids.extend([u.id, f"#{u.id}", u.id.replace('#', '')])
    if clean_target not in search_ids:
        search_ids.extend([clean_target, f"#{clean_target}", clean_target.replace('#', '')])
    
    session_conditions = [
        PatientConsultationSession.patient_id.in_(search_ids)
    ]
    if patient.name:
        session_conditions.append(PatientConsultationSession.patient_name.ilike(patient.name))
    if u and u.full_name:
        session_conditions.append(PatientConsultationSession.patient_name.ilike(u.full_name))

    all_sessions = db.query(PatientConsultationSession).filter(
        or_(*session_conditions)
    ).order_by(PatientConsultationSession.session_number.desc(), PatientConsultationSession.created_at.desc()).all()

    # If no patient-specific sessions found, check if there's any recent session shared
    shared_sessions = [s for s in all_sessions if s.is_shared_with_patient]

    # Effective attending doctor
    effective_doc_name = str(patient.assigned_doctor_name or "Dr. Ankit Kumar")
    effective_doc_id = str(patient.assigned_doctor_id or "DOC-001")

    # Base profile
    profile = {
        "name": patient.name,
        "dob": "1991-05-18",
        "mrn": patient.id,
        "age": patient.age,
        "gender": patient.gender,
        "phone": patient.phone or "+1 (555) 000-0000",
        "email": patient.email or (u.email if u else ""),
        "reason": patient.reason or "General Health Review",
        "condition": patient.condition or "Triage Assessment",
        "lastVisit": patient.last_visit or datetime.now().strftime("%B %d, %Y")
    }

    # Dynamically build records, prescriptions, alerts, and consultations from DB sessions
    dynamic_records = []
    dynamic_prescriptions = []
    dynamic_alerts = []
    dynamic_consultations = []

    parsed_vitals = {
        "bp": "120/80 mmHg",
        "hr": "72 bpm",
        "weight": "75 kg",
        "temp": "36.6 °C"
    }
    if patient.vitals_json:
        try:
            v_dict = json.loads(str(patient.vitals_json))
            parsed_vitals["bp"] = v_dict.get("bp", parsed_vitals["bp"])
            parsed_vitals["hr"] = v_dict.get("hr", parsed_vitals["hr"])
            parsed_vitals["pain"] = v_dict.get("pain", "0/10")
            parsed_vitals["spo2"] = v_dict.get("spo2", "98%")
            if "weight" in v_dict:
                parsed_vitals["weight"] = v_dict["weight"]
            if "temp" in v_dict:
                parsed_vitals["temp"] = v_dict["temp"]
        except Exception:
            pass

    for session in all_sessions:
        s_soap = {}
        if session.soap_report_json:
            try:
                s_soap = json.loads(str(session.soap_report_json))
            except Exception:
                s_soap = {}

        conv_list = []
        if session.conversation_json:
            try:
                conv_list = json.loads(str(session.conversation_json))
            except Exception:
                conv_list = []

        sess_date_str = session.appointment_date.strftime("%Y-%m-%d") if session.appointment_date else (
            session.shared_at.strftime("%Y-%m-%d") if session.shared_at else (
                session.created_at.strftime("%Y-%m-%d") if session.created_at else datetime.now().strftime("%Y-%m-%d")
            )
        )
        doctor_name = str(session.doctor_name or effective_doc_name)
        doctor_id = str(session.doctor_id or effective_doc_id)

        what_happened = (
            s_soap.get("assessment") or 
            s_soap.get("subjective") or 
            patient.reason or 
            f"Clinical consultation conducted with {doctor_name}."
        )

        consultation_item = {
            "id": session.id,
            "sessionNumber": session.session_number,
            "sessionTitle": session.session_title or f"Consultation Session {session.session_number}",
            "doctorName": doctor_name,
            "doctorId": doctor_id,
            "doctorSpecialty": "Attending Physician / Specialist",
            "date": sess_date_str,
            "consultationMode": (session.consultation_mode or "IN_PERSON").replace("_", " ").title(),
            "status": session.status or "COMPLETED",
            "isShared": session.is_shared_with_patient,
            "summary": what_happened,
            "assessment": s_soap.get("assessment", f"Consultation for {patient.condition or patient.reason or 'health review'}"),
            "plan": s_soap.get("plan", "Follow prescribed clinical instructions and recommendations."),
            "subjective": s_soap.get("subjective", f"Patient {patient.name} presented for clinical consultation with {doctor_name}."),
            "objective": s_soap.get("objective", f"Vital signs: BP {parsed_vitals.get('bp', '120/80 mmHg')}, HR {parsed_vitals.get('hr', '72 bpm')}. Examination completed."),
            "vitals": parsed_vitals,
            "conversation": conv_list,
            "transcript": session.transcript or "",
            "soapReport": {
                **s_soap,
                "doctorName": doctor_name,
                "date": sess_date_str,
                "sessionTitle": session.session_title,
                "encounterId": session.encounter_id or f"ENC-{session.session_number}",
                "patientName": patient.name,
                "mrn": patient.id
            } if (s_soap or session.is_shared_with_patient) else None
        }
        dynamic_consultations.append(consultation_item)

        # 1. Add Medical Record for this shared consultation
        if session.is_shared_with_patient or s_soap:
            dynamic_records.append({
                "id": session.id,
                "title": f"Clinical Review — {session.session_title or 'SOAP Consultation'}",
                "badge": "Verified SOAP" if session.is_shared_with_patient else "Recorded",
                "doctorDate": f"{doctor_name} · {sess_date_str}",
                "notes": (s_soap.get("assessment") or s_soap.get("subjective") or patient.reason or "Clinical consultation completed.")[:220],
                "vitals": parsed_vitals,
                "hasSoap": bool(s_soap or session.is_shared_with_patient),
                "soapReport": {
                    **s_soap,
                    "doctorName": doctor_name,
                    "date": sess_date_str,
                    "sessionTitle": session.session_title,
                    "encounterId": session.encounter_id,
                    "patientName": patient.name,
                    "mrn": patient.id
                }
            })

            # 2. Extract Prescriptions from Plan
            extracted_rx = _extract_prescriptions_from_soap(s_soap, doctor_name, sess_date_str)
            if extracted_rx:
                for rx in extracted_rx:
                    rx["soapReport"] = {
                        **s_soap,
                        "doctorName": doctor_name,
                        "date": sess_date_str,
                        "sessionTitle": session.session_title,
                        "encounterId": session.encounter_id,
                        "patientName": patient.name,
                        "mrn": patient.id
                    }
                    dynamic_prescriptions.append(rx)
            elif s_soap:
                dynamic_prescriptions.append({
                    "id": f"rx-soap-{session.id}",
                    "title": "Clinical Care Plan & Medication Directives",
                    "name": "Consultation Directives",
                    "dosage": "Active Plan",
                    "badge": "active",
                    "statusClass": "active",
                    "doctorDate": f"{doctor_name} · Issued {sess_date_str}",
                    "instructions": (s_soap.get("plan") or "Follow verified doctor treatment plan")[:120],
                    "refillInfo": "Refills as per plan",
                    "soapReport": {
                        **s_soap,
                        "doctorName": doctor_name,
                        "date": sess_date_str,
                        "sessionTitle": session.session_title,
                        "encounterId": session.encounter_id,
                        "patientName": patient.name,
                        "mrn": patient.id
                    }
                })

            # 3. Extract Follow-up Alerts from Plan
            plan_str = s_soap.get("plan", "")
            dynamic_alerts.append({
                "id": f"alert-{session.id}",
                "title": f"Follow-up Clinical Review & Medication Check",
                "doctor": f"With {doctor_name}",
                "date": (datetime.now() + timedelta(days=28)).strftime("%Y-%m-%d"),
                "priority": "High priority",
                "priorityClass": "high",
                "dotColor": "#ef4444",
                "daysAway": "28 days away",
                "soapReport": s_soap
            })

    # Isolate Marcus Delgado seed data strictly to Marcus Delgado demo account
    is_marcus_demo = (patient.id == "#00847261" or patient.name.lower() == "marcus delgado")

    if is_marcus_demo:
        baseline_alerts = [
            {
                "id": 1,
                "title": "Blood Pressure Check",
                "doctor": f"With {effective_doc_name}",
                "date": "2026-08-25",
                "priority": "High priority",
                "priorityClass": "high",
                "dotColor": "#ef4444",
                "daysAway": "12 days away"
            },
            {
                "id": 2,
                "title": "Cholesterol Panel",
                "doctor": f"With {effective_doc_name}",
                "date": "2026-09-10",
                "priority": "Medium priority",
                "priorityClass": "medium",
                "dotColor": "#f59e0b",
                "daysAway": "28 days away"
            }
        ]
        baseline_records = [
            {
                "id": 101,
                "title": "Hypertension — Stage 1",
                "badge": "Follow-up",
                "doctorDate": f"{effective_doc_name} · 2026-07-14",
                "notes": "BP trending down. Continue current regimen. Sodium reduction showing results. Next check in 6 weeks.",
                "vitals": {"bp": "138/88 mmHg", "hr": "72 bpm", "weight": "81 kg", "temp": "36.6 °C"},
                "hasSoap": False
            }
        ]
        baseline_prescriptions = [
            {
                "id": "rx-base-1",
                "title": "Lisinopril — 10 mg",
                "badge": "active",
                "statusClass": "active",
                "doctorDate": f"{effective_doc_name} · Issued 2026-04-03",
                "instructions": "Once daily — morning",
                "refillInfo": "2 refills left",
                "soapReport": None
            }
        ]
        final_alerts = dynamic_alerts if dynamic_alerts else baseline_alerts
        final_records = dynamic_records if dynamic_records else baseline_records
        final_prescriptions = dynamic_prescriptions if dynamic_prescriptions else baseline_prescriptions
        final_consultations = dynamic_consultations
    else:
        # For non-Marcus patients (e.g. Abdul Jabbar), strictly return their actual consulted doctor
        final_alerts = dynamic_alerts
        final_records = dynamic_records
        final_prescriptions = dynamic_prescriptions
        final_consultations = dynamic_consultations

        # If this patient has no recorded sessions yet, create an initial consultation, prescription, record, and alert with their doctor
        if not final_consultations:
            initial_soap = {
                "subjective": f"Patient {patient.name} presented for clinical evaluation and consultation. Chief complaint: {patient.reason or 'Health review'}.",
                "objective": f"BP {parsed_vitals.get('bp', '120/80 mmHg')}, HR {parsed_vitals.get('hr', '72 bpm')}. Physical exam completed.",
                "assessment": f"1. {patient.condition or patient.reason or 'Clinical Assessment & Triage'}.",
                "plan": f"1. Attending consultation and medication review with {effective_doc_name}.\n2. Follow prescribed clinical care directives.\n3. Return for follow-up review in 4 weeks.",
                "doctorName": effective_doc_name,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "sessionTitle": "Initial Consultation & Clinical Intake",
                "encounterId": f"ENC-{patient.id.replace('#', '')}-01",
                "patientName": patient.name,
                "mrn": patient.id
            }

            final_consultations = [
                {
                    "id": f"sess-init-{patient.id.replace('#', '')}",
                    "sessionNumber": 1,
                    "sessionTitle": "Initial Consultation & Clinical Intake",
                    "doctorName": effective_doc_name,
                    "doctorId": effective_doc_id,
                    "doctorSpecialty": "Attending Physician",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "consultationMode": "In-Person Consultation",
                    "status": "COMPLETED",
                    "isShared": True,
                    "summary": f"Initial clinical assessment and record verified for {patient.name} with {effective_doc_name}.",
                    "assessment": f"{patient.condition or patient.reason or 'General Health Review'}",
                    "plan": f"1. Attending consultation with {effective_doc_name}.\n2. Follow prescribed clinical directives.",
                    "subjective": f"Patient {patient.name} registered for clinical review. Reason: {patient.reason or 'Consultation'}.",
                    "objective": f"BP {parsed_vitals.get('bp', '120/80 mmHg')}, HR {parsed_vitals.get('hr', '72 bpm')}. Vitals recorded.",
                    "vitals": parsed_vitals,
                    "conversation": [],
                    "transcript": "",
                    "soapReport": initial_soap
                }
            ]

            final_records = [
                {
                    "id": 101,
                    "title": f"Clinical Consultation — {patient.condition or 'Health Review'}",
                    "badge": "Verified SOAP",
                    "doctorDate": f"{effective_doc_name} · {datetime.now().strftime('%Y-%m-%d')}",
                    "notes": f"Initial consultation completed. Assessment: {patient.condition or 'Clinical review'}. Follow-up scheduled.",
                    "vitals": parsed_vitals,
                    "hasSoap": True,
                    "soapReport": initial_soap
                }
            ]

            final_prescriptions = [
                {
                    "id": f"rx-{patient.id.replace('#', '')}-1",
                    "title": "Prescribed Clinical Regimen & Care Plan",
                    "name": "Clinical Directives",
                    "dosage": "Active Plan",
                    "badge": "active",
                    "statusClass": "active",
                    "doctorDate": f"{effective_doc_name} · Issued {datetime.now().strftime('%Y-%m-%d')}",
                    "instructions": "Follow verified doctor treatment plan and clinical directives",
                    "refillInfo": "Active Care Plan",
                    "soapReport": initial_soap
                }
            ]

            final_alerts = [
                {
                    "id": 1,
                    "title": "Clinical Follow-up & Progress Review",
                    "doctor": f"With {effective_doc_name}",
                    "date": (datetime.now() + timedelta(days=28)).strftime("%Y-%m-%d"),
                    "priority": "High priority",
                    "priorityClass": "high",
                    "dotColor": "#ef4444",
                    "daysAway": "28 days away",
                    "soapReport": initial_soap
                }
            ]

    last_consultation = final_consultations[0] if final_consultations else None

    return {
        "success": True,
        "patientProfile": profile,
        "hasSharedSoap": True,
        "sharedSessionCount": max(len(shared_sessions), len(final_consultations)),
        "followUpAlerts": final_alerts,
        "medicalRecords": final_records,
        "prescriptions": final_prescriptions,
        "consultations": final_consultations,
        "lastConsultation": last_consultation
    }

