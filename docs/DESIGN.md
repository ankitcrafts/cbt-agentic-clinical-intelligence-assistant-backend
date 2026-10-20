# Backend API Design & Specifications
## ClinIQ: REST API Contracts, Schemas & Endpoints

---

## 1. REST API Endpoints Overview

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root health check, system status & feature flags | No |
| `GET` | `/observability` | LangSmith tracing telemetry status | No |
| `POST` | `/api/auth/send-otp` | Generate & dispatch 6-digit OTP to user email | No |
| `POST` | `/api/auth/verify-otp` | Verify OTP code & return JWT access token | No |
| `GET` | `/api/auth/me` | Retrieve current authenticated user profile | Yes (Bearer) |
| `GET` | `/api/auth/users` | List all registered users (Admin only) | Yes (Admin) |
| `POST` | `/api/process-audio` | Upload audio for Deepgram transcription & diarization | No |
| `POST` | `/api/rag-diagnose` | Run RAG pipeline & return differential diagnoses | No |
| `POST` | `/api/generate-soap` | Synthesize complete structured SOAP note | No |
| `POST` | `/api/reframe-soap` | Live reframe / edit specific SOAP section | No |
| `POST` | `/api/generate-reports` | Render ReportLab Analysis & SOAP PDFs | No |
| `POST` | `/api/upload-gdrive` | Upload generated PDF report to Google Drive | No |
| `GET` | `/reports` | List all generated clinical analysis & SOAP PDFs | No |
| `GET` | `/reports/{filename}` | Download generated PDF report by filename | No |
| `GET` | `/rag/ingest` | Trigger ingestion of NICE guideline documents | No |
| `GET` | `/visit/{visit_id}` | Retrieve LangGraph clinical state memory for visit | No |
| `GET` | `/api/scheduling/doctors` | List doctors with availability and specialties | No |
| `GET` | `/api/scheduling/slots` | Query open appointment slots for a given doctor & date | No |
| `POST` | `/api/scheduling/book` | Book an appointment slot | No |
| `GET` | `/api/scheduling/appointments` | List appointments filtered by patient, doctor, or status | Yes |

---

## 2. API Request & Response Schemas

### 2.1 OTP Authentication Schemas
```json
// POST /api/auth/send-otp Request:
{
  "email": "doctor@hospital.com",
  "role": "DOCTOR"
}

// POST /api/auth/send-otp Response:
{
  "status": "success",
  "message": "OTP sent successfully to doctor@hospital.com",
  "expires_in_seconds": 300
}

// POST /api/auth/verify-otp Request:
{
  "email": "doctor@hospital.com",
  "otp": "481920"
}

// POST /api/auth/verify-otp Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6...",
  "token_type": "bearer",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "doctor@hospital.com",
    "full_name": "Dr. Sarah Mitchell",
    "role": "DOCTOR",
    "specialty": "Cardiology"
  }
}
```

### 2.2 Audio Processing & Diarization Schemas
```json
// POST /api/process-audio (multipart/form-data: file=<audio_file>)
// Response:
{
  "status": "success",
  "transcript": "Doctor: Good morning Jane... Patient: I've had chest tightness for 3 days...",
  "duration_seconds": 184.5,
  "utterances": [
    {
      "speaker": "Doctor",
      "text": "Good morning Jane, how can I help you today?",
      "start": 0.5,
      "end": 3.2,
      "confidence": 0.98
    },
    {
      "speaker": "Patient",
      "text": "I've had chest tightness for 3 days when climbing stairs.",
      "start": 3.8,
      "end": 7.1,
      "confidence": 0.96
    }
  ]
}
```

### 2.3 RAG Diagnosis & Differential Synthesis Schemas
```json
// POST /api/rag-diagnose Request:
{
  "visit_id": "V-2026-1049",
  "transcript": "..."
}

// Response:
{
  "visit_id": "V-2026-1049",
  "red_flags": [
    "Exertional retrosternal chest pain radiating to left jaw"
  ],
  "differential_diagnoses": [
    {
      "condition": "Stable Angina Pectoris",
      "confidence": 0.88,
      "icd_10": "I20.9",
      "rationale": "Consistent with exertional onset and rapid relief with rest.",
      "guideline_reference": "NICE Clinical Guideline CG95: Chest pain of recent onset"
    },
    {
      "condition": "Gastroesophageal Reflux Disease (GERD)",
      "confidence": 0.62,
      "icd_10": "K21.9",
      "rationale": "Post-prandial burning sensation, but exertional component makes cardiac evaluation primary.",
      "guideline_reference": "NICE Guideline NG1: Gastro-oesophageal reflux disease"
    }
  ],
  "suggested_questions": [
    "Does the pain ever occur at rest or wake you from sleep?",
    "Have you experienced any shortness of breath, nausea, or sweating with the discomfort?"
  ]
}
```

### 2.4 Doctor Share SOAP & Prescription Schema
```json
// POST /api/scheduling/patients/{patient_id}/share-soap
{
  "encounter_id": "ENC-6554743-1284",
  "doctor_id": "DOC-8492",
  "doctor_name": "Dr. Yuki Tanaka",
  "patient_name": "Asmit Bhardwaj",
  "soap_report": {
    "subjective": "Patient presents with increased urinary frequency...",
    "objective": "BP: 120/80 mmHg, HR: 72 bpm. Abdomen soft...",
    "assessment": "Urinary Tract Infection (UTI) under evaluation...",
    "plan": "1. Prescribed Amoxicillin 500mg. 2. Increase hydration. 3. Follow-up in 7 days."
  },
  "prescription": {
    "medication": "Amoxicillin",
    "strength": "500 mg",
    "route": "Oral",
    "frequency": "Three times daily",
    "duration": "7 days",
    "quantity": "21 capsules",
    "refills": 0,
    "instructions": "Take one capsule three times daily with a full glass of water after meals until completed."
  }
}
```

### 2.5 Consultation Session Retrieval Schema
```json
// GET /api/scheduling/patients/{patient_id}/sessions
{
  "success": true,
  "patientId": "#6554743",
  "count": 1,
  "data": [
    {
      "id": "sess-6554743-1-3829",
      "encounterId": "CONS-6554743-1",
      "sessionNumber": 1,
      "sessionTitle": "Session 1: Going toilet every time for urination",
      "doctorId": "DOC-2849",
      "doctorName": "Dr. Ankit Kumar",
      "appointmentDate": "2026-10-11T09:15:00Z",
      "transcript": "Doctor: Hello Asmit...",
      "conversation": [ ... ],
      "aiSuggestions": [
        {
          "id": 1,
          "type": "high",
          "tag": "HIGH · CLINICAL EVALUATION",
          "text": "Clarify symptom onset timeline — Establish whether symptoms are acute, subacute, or progressive."
        }
      ],
      "ai_suggestions": [ ... ],
      "clinicalAnalysis": { ... },
      "soapReport": { ... },
      "durationSeconds": 480,
      "status": "COMPLETED"
    }
  ]
}
```

### 2.6 Patient Vitals Schema
```json
// Embedded in PatientModel / format_patient_dict / VitalsSchema
{
  "bp": "120/80 mmHg",
  "hr": "72 bpm",
  "pain": "0/10",
  "spo2": "98%",
  "temp": "36.6 °C",
  "weight": "75 kg",
  "glucose": "95 mg/dL"
}
```
* Note: `format_patient_dict` guarantees standard baseline keys are always populated, preventing `{}` from ever reaching clients.

### 2.6 Dynamic 5-Tab CDS Clinical Analysis Contract
```json
// Root & nested clinical_analysis response contract:
{
  "differential_diagnosis": [
    {
      "rank": 1,
      "title": "Type 2 Diabetes Mellitus",
      "icdCode": "E11.9",
      "percentage": 88,
      "risk": "high",
      "color": "#0d9488",
      "description": "Evaluated based on clinical symptoms and encounter context."
    }
  ],
  "confidence_metrics": {
    "overallScore": 78,
    "overallLevel": "High — Validated",
    "subtext": "Strong diagnostic concordance achieved.",
    "domains": [
      { "name": "Symptom completeness", "percentage": 85, "color": "#0d9488" },
      { "name": "Medical history coverage", "percentage": 90, "color": "#0f766e" },
      { "name": "Lifestyle & social history", "percentage": 75, "color": "#0d9488" },
      { "name": "DDx alignment", "percentage": 88, "color": "#0d9488" },
      { "name": "Guideline adherence", "percentage": 86, "color": "#0d9488" }
    ]
  },
  "guideline_medicines": [
    {
      "name": "Metformin Hydrochloride",
      "drugClass": "Biguanide",
      "grade": "Grade A",
      "source": "NICE NG28 / BNF",
      "dosage": "500 mg PO once daily with meals",
      "recommendation": "First-line therapy for glycemic control.",
      "warning": "Discontinue if eGFR < 30 mL/min/1.73m²."
    }
  ],
  "red_flags": [
    {
      "severity": "CRITICAL",
      "title": "Diabetic Ketoacidosis / HHS Screen",
      "recommendation": "Immediate capillary blood glucose and ketone check."
    }
  ],
  "follow_up_questions": [
    {
      "id": "fq-1",
      "question": "Have you noticed increased frequency of urination at night?",
      "priority": "HIGH",
      "asked": false,
      "answered": false,
      "doctor_snippet": "Have you noticed waking up to urinate?",
      "patient_snippet": "Yes, two or three times every night."
    }
  ],
  "soap_report": {
    "subjective": "Patient reports...",
    "objective": "BP 120/80 mmHg...",
    "assessment": "1. Type 2 Diabetes Mellitus (E11.9)...",
    "plan": "1. Metformin 500mg..."
  }
}
```

---


## 3. Database Schema Models (SQLAlchemy)

```python
class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum("ADMIN", "DOCTOR", "PATIENT", name="user_roles"), nullable=False)
    phone = Column(String(50), nullable=True)
    specialty = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    slot_id = Column(UUID(as_uuid=True), ForeignKey("appointment_slots.id"), nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String(50), default="SCHEDULED")
    chief_complaint = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```
