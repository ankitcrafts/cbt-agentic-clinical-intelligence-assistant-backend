import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import DATABASE_URL, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB

logger = logging.getLogger(__name__)

Base = declarative_base()


def ensure_postgres_database():
    """Ensure the target PostgreSQL database exists before connecting."""
    if not isinstance(DATABASE_URL, str) or not DATABASE_URL.startswith("postgresql"):
        return

    admin_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
    try:
        temp_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with temp_engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{POSTGRES_DB}'")
            )
            exists = result.scalar()
            if not exists:
                conn.execute(text(f"CREATE DATABASE {POSTGRES_DB}"))
                logger.info(f"Created PostgreSQL database: {POSTGRES_DB}")
        temp_engine.dispose()
    except Exception as e:
        logger.warning(f"Could not auto-create database '{POSTGRES_DB}': {e}")


def create_db_engine():
    """Create database engine with fallback for seamless developer experience."""
    ensure_postgres_database()
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database: {POSTGRES_DB}")
        return engine
    except Exception as e:
        logger.error(f"PostgreSQL connection failed ({e}). Falling back to local SQLite database.")
        sqlite_url = "sqlite:///./cliniq_local.db"
        return create_engine(sqlite_url, connect_args={"check_same_thread": False})


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for yielding database sessions per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and seed default accounts and demonstrative sessions."""
    import json
    from datetime import datetime, timezone
    import app.models.models  # Ensure models are imported for metadata registration
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

    # Lightweight migration for existing SQLite / Postgres tables
    with engine.connect() as conn:
        for col_name, col_type in [
            ("gdrive_file_id", "VARCHAR(255)"),
            ("gdrive_web_view_link", "VARCHAR(500)"),
            ("consultation_mode", "VARCHAR(50) DEFAULT 'IN_PERSON'"),
            ("audio_source", "VARCHAR(50) DEFAULT 'MIC'"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE patient_consultation_sessions ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass
    
    # Seed default demonstration users if table is empty
    from app.models.models import User, PatientConsultationSession
    db = SessionLocal()
    try:
        count = db.query(User).count()
        if count == 0:
            demo_users = [
                User(
                    email="admin@cliniq.health",
                    full_name="Hospital Administrator",
                    role="ADMIN",
                    department="Hospital Operations",
                    designation="System Administrator",
                ),
                User(
                    email="doctor@cliniq.health",
                    full_name="Dr. Yuki Tanaka",
                    role="DOCTOR",
                    department="Cardiology",
                    specialization="Senior Cardiologist",
                    designation="Attending Physician",
                ),
                User(
                    email="patient@cliniq.health",
                    full_name="Marcus Delgado",
                    role="PATIENT",
                    phone_number="+1-555-0199",
                ),
            ]
            db.add_all(demo_users)
            db.commit()
            logger.info("Seeded initial demo accounts (admin, doctor, patient).")

        # Seed demonstrative prior sessions for patients (Session 1)
        session_count = db.query(PatientConsultationSession).count()
        if session_count == 0:
            marcus_exchanges = [
                {"speaker": "Doctor", "isDoctor": True, "text": "Good morning Marcus, nice to see you again. How have you been feeling with the blood pressure medication?", "timestamp": "09:02"},
                {"speaker": "Patient", "isDoctor": False, "text": "Good morning Dr. Tanaka. Generally okay, but over the last couple of weeks I have had these throbbing headaches in the morning.", "timestamp": "09:03"},
                {"speaker": "Doctor", "isDoctor": True, "text": "I see. Have you noticed any dizziness, blurred vision, or chest tightness when those headaches occur?", "timestamp": "09:05"},
                {"speaker": "Patient", "isDoctor": False, "text": "A bit of dizziness if I stand up too fast, but no chest pain. My home blood pressure cuff showed around 140 over 90 a few times.", "timestamp": "09:07"},
                {"speaker": "Doctor", "isDoctor": True, "text": "Your vitals today show 138/88 mmHg. Let us review your medication adherence and consider adding an ACE inhibitor or titration.", "timestamp": "09:10"},
                {"speaker": "Patient", "isDoctor": False, "text": "Understood Doctor, I will keep taking them daily with breakfast as instructed.", "timestamp": "09:12"}
            ]

            marcus_soap = {
                "subjective": "Marcus Delgado (42M) presents for hypertension review. Reports 2-week history of throbbing morning headaches and occasional postural dizziness. Home BP readings ~140/90 mmHg. Denies chest pain or shortness of breath.",
                "objective": "BP: 138/88 mmHg, HR: 72 bpm, SpO2: 98% room air. Cardiac auscultation: regular rate and rhythm, S1/S2 present, no murmurs or gallops. Lungs clear to auscultation bilaterally. No peripheral edema.",
                "assessment": "1. Essential Hypertension Stage 1 (ICD-10 I10), suboptimally controlled. 2. Tension-type headaches secondary to elevated arterial pressure (ICD-10 G44.2).",
                "plan": "1. Titrate Amlodipine to 5mg PO daily. 2. Continue daily home BP logging morning and night. 3. Comprehensive metabolic panel and renal function panel scheduled. 4. Follow-up consultation in 4 weeks."
            }

            marcus_suggestions = [
                {"id": 1, "priority": "HIGH", "category": "TITRATION", "title": "Medication Titration Recommended", "description": "Patient BP 138/88 indicates sub-optimal control. Consider adding ACE inhibitor (Lisinopril 10mg) or titrating calcium channel blocker."},
                {"id": 2, "priority": "MEDIUM", "category": "DIAGNOSTIC", "title": "Renal Function & Electrolyte Monitoring", "description": "Order serum creatinine and electrolytes before modifying antihypertensive regimen."},
                {"id": 3, "priority": "LOW", "category": "COUNSELING", "title": "Dietary Sodium Counseling", "description": "Reinforce DASH diet and daily sodium intake under 2,000 mg."}
            ]

            marcus_analysis = {
                "summary": {
                    "chief_complaint": "Hypertension follow-up with morning headaches",
                    "history_of_present_illness": "42-year-old male with known Stage 1 Essential Hypertension presenting for scheduled follow-up. Reports new onset throbbing morning headaches for 2 weeks with home BP 140/90.",
                    "identified_issues": [
                        "Suboptimal blood pressure control (138/88 mmHg)",
                        "Morning tension headaches associated with elevated BP",
                        "Mild orthostatic dizziness upon standing"
                    ]
                },
                "entities": {
                    "symptoms": ["Morning headaches", "Orthostatic dizziness", "Elevated home BP"],
                    "vital_signs": {"bp": "138/88 mmHg", "hr": "72 bpm", "spo2": "98%"},
                    "medications": [
                        {"name": "Amlodipine", "dosage": "5mg once daily", "drug_class": "Calcium Channel Blocker", "grade": "Grade A", "source": "ACC/AHA Guidelines"},
                        {"name": "Lisinopril", "dosage": "10mg once daily", "drug_class": "ACE Inhibitor", "grade": "Grade A", "source": "JNC 8 Guidelines"}
                    ]
                },
                "differential_diagnosis": [
                    {"diagnosis": "Essential Hypertension Stage 1", "icd10_code": "I10", "confidence_level": "High", "confidence_score": 0.94, "rationale": "Sustained clinic and ambulatory BP elevations consistent with primary hypertension."},
                    {"diagnosis": "Tension-type Headache", "icd10_code": "G44.2", "confidence_level": "Moderate", "confidence_score": 0.78, "rationale": "Bilateral throbbing headache without focal neurological deficits."}
                ],
                "soap_report": marcus_soap,
                "ai_suggestions": marcus_suggestions
            }

            session_marcus = PatientConsultationSession(
                patient_id="#00847261",
                patient_name="Marcus Delgado",
                doctor_id="DOC-001",
                doctor_name="Dr. Yuki Tanaka",
                encounter_id="CONS-00847261-1",
                session_number=1,
                session_title="Session 1: Initial Presentation & BP Review",
                appointment_date=datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc),
                transcript=" ".join([e["text"] for e in marcus_exchanges]),
                conversation_json=json.dumps(marcus_exchanges),
                ai_suggestions_json=json.dumps(marcus_suggestions),
                clinical_analysis_json=json.dumps(marcus_analysis),
                soap_report_json=json.dumps(marcus_soap),
                duration_seconds=780,
                status="COMPLETED",
            )

            # Seed Priya Sharma Session 1
            priya_exchanges = [
                {"speaker": "Doctor", "isDoctor": True, "text": "Hello Priya, what brings you in today for our consultation?", "timestamp": "09:46"},
                {"speaker": "Patient", "isDoctor": False, "text": "Hi Doctor. I have been feeling unusually fatigued over the last six weeks, despite sleeping 8 hours a night. Also noticed some cold sensitivity.", "timestamp": "09:47"},
                {"speaker": "Doctor", "isDoctor": True, "text": "Let us check your thyroid hormone levels and thyroid stimulating hormone (TSH). Are you on any thyroid supplement currently?", "timestamp": "09:49"},
                {"speaker": "Patient", "isDoctor": False, "text": "I was prescribed Levothyroxine 50mcg last year, but stopped it three months ago thinking my levels were fine.", "timestamp": "09:51"}
            ]

            priya_soap = {
                "subjective": "Priya Sharma (34F) reports 6 weeks of severe fatigue, lethargy, and cold intolerance. Discontinued Levothyroxine 50mcg 3 months ago without physician guidance.",
                "objective": "Vitals: BP 118/76 mmHg, HR 68 bpm, Temp 36.4C. Mild periorbital puffiness. Thyroid gland non-tender, no nodules palpated.",
                "assessment": "Recurrent Primary Hypothyroidism secondary to medication discontinuation (ICD-10 E03.9).",
                "plan": "1. Resume Levothyroxine 50mcg PO daily on empty stomach. 2. Serum TSH and Free T4 panel in 6 weeks. 3. Patient education regarding lifelong thyroid replacement adherence."
            }

            priya_suggestions = [
                {"id": 1, "priority": "HIGH", "category": "ENDOCRINOLOGY", "title": "Thyroid Hormone Titration", "description": "Restart Levothyroxine 50mcg PO daily on empty stomach 30-60 minutes before breakfast."},
                {"id": 2, "priority": "MEDIUM", "category": "LABORATORY", "title": "Serum TSH & Free T4 Panel", "description": "Re-evaluate TSH in 6 weeks to confirm biochemical euthyroidism."}
            ]

            priya_analysis = {
                "summary": {
                    "chief_complaint": "Persistent fatigue and cold intolerance",
                    "history_of_present_illness": "34-year-old female presenting with 6-week progressive fatigue and cold intolerance after self-discontinuing Levothyroxine.",
                    "identified_issues": ["Thyroid medication non-compliance", "Fatigue impacting daily activities"]
                },
                "entities": {
                    "symptoms": ["Persistent fatigue", "Cold sensitivity", "Periorbital puffiness"],
                    "vital_signs": {"bp": "118/76 mmHg", "hr": "68 bpm"}
                },
                "differential_diagnosis": [
                    {"diagnosis": "Primary Hypothyroidism", "icd10_code": "E03.9", "confidence_level": "High", "confidence_score": 0.96, "rationale": "Symptom complex and history of discontinued thyroid replacement."}
                ],
                "soap_report": priya_soap,
                "ai_suggestions": priya_suggestions
            }

            session_priya = PatientConsultationSession(
                patient_id="#00391847",
                patient_name="Priya Sharma",
                doctor_id="DOC-001",
                doctor_name="Dr. Yuki Tanaka",
                encounter_id="CONS-00391847-1",
                session_number=1,
                session_title="Session 1: Initial Thyroid Consultation",
                appointment_date=datetime(2026, 5, 2, 9, 45, tzinfo=timezone.utc),
                transcript=" ".join([e["text"] for e in priya_exchanges]),
                conversation_json=json.dumps(priya_exchanges),
                ai_suggestions_json=json.dumps(priya_suggestions),
                clinical_analysis_json=json.dumps(priya_analysis),
                soap_report_json=json.dumps(priya_soap),
                duration_seconds=640,
                status="COMPLETED",
            )

            db.add_all([session_marcus, session_priya])
            db.commit()
            logger.info("Seeded demonstrative Session 1 consultation records for patients.")
    except Exception as e:
        logger.error(f"Error seeding demo accounts / sessions: {e}")
        db.rollback()
    finally:
        db.close()

