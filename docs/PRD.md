# Product Requirements Document (PRD)
## ClinIQ: Backend Clinical Intelligence & Agentic Engine API

---

## 1. Executive Summary
The **ClinIQ Backend** is a high-performance, agentic microservice built on FastAPI, LangGraph, ChromaDB, and Deepgram. It serves as the clinical intelligence core for ambient medical consultations, powering real-time medical speech diarization, clinical entity extraction, RAG guideline retrieval over official NICE medical guidelines, multi-hypothesis differential diagnosis ranking, automated SOAP note synthesis, ReportLab PDF clinical report generation, Google Drive archival, Gmail OTP authentication, and doctor-patient appointment scheduling.

---

## 2. Core Capabilities & Backend Objectives
1. **Medical Audio Processing:** Ingest raw audio files or live streaming chunks, perform speaker diarization (`Doctor` vs `Patient`), and produce timestamped medical transcripts.
2. **LangGraph Agentic Workflow:** Orchestrate a multi-node clinical state graph that manages visit memory, tracks dialogue turns, extracts clinical entities, queries guideline vectors, and synthesizes differential diagnoses.
3. **Retrieval-Augmented Decision Support:** Ground clinical diagnoses in vector-indexed National Institute for Health and Care Excellence (NICE) guidelines with semantic retrieval and similarity scores.
4. **Automated Clinical Documentation:** Synthesize comprehensive SOAP notes (Subjective, Objective, Assessment, Plan) with rich ICD-10 suggestions, red flag alerts, and patient aftercare instructions.
5. **PDF Report Engine & Cloud Archival:** Generate high-resolution, branded clinical analysis reports and patient SOAP notes via ReportLab, and automatically synchronize them to Google Drive.
6. **Authentication & Multi-Role Scheduling:** Manage user identities (`ADMIN`, `DOCTOR`, `PATIENT`), 6-digit OTP email delivery via Gmail SMTP, doctor availability calendars, and booking appointment lifecycles.

---

## 3. System Personas & API Consumers

| API Consumer | Functional Needs | Backend Service Endpoints |
| :--- | :--- | :--- |
| **Doctor Console Client** | Audio processing, real-time clinical suggestions, live SOAP drafting, final PDF report generation, Google Drive upload. | `/api/process-audio`, `/api/rag-diagnose`, `/api/generate-soap`, `/api/upload-gdrive` |
| **Patient Portal Client** | Slot discovery, booking creation, OTP authentication, report downloads. | `/api/scheduling/slots`, `/api/scheduling/book`, `/api/auth/send-otp`, `/reports/{file}` |
| **Admin Dashboard Client** | Doctor management, slot generation, user directory, system observability. | `/api/scheduling/doctors`, `/api/auth/users`, `/observability` |

---

## 4. Functional Requirements

### 4.1 Speech-to-Text & Diarization Engine
* **FR-BE-1.1:** Support multi-format audio uploads (`audio/wav`, `audio/mp3`, `audio/m4a`, `audio/webm`, `audio/ogg`).
* **FR-BE-1.2:** Integrate Deepgram Nova-2 Medical STT model with speaker diarization enabled.
* **FR-BE-1.3:** Map Deepgram speaker channels (`speaker: 0`, `speaker: 1`) to clinical roles (`Doctor`, `Patient`).

### 4.2 LangGraph Agentic Clinical Pipeline
* **FR-BE-2.1:** Maintain stateful consultation memory across dialogue turns using `ClinicalState` schemas.
* **FR-BE-2.2:** Clinical Entity Extractor node extracting Chief Complaint, Symptoms, Duration, Severity, Medical History, Medications, and Allergies.
* **FR-BE-2.3:** Clinical Red Flag Detector alerting for emergency symptoms requiring immediate intervention.
* **FR-BE-2.4:** Guideline Retrieval Node querying ChromaDB for relevant NICE guidance chunks.
* **FR-BE-2.5:** Differential Diagnosis Synthesizer producing ranked differential diagnoses with confidence percentages and evidence citations.

### 4.3 Documentation, PDF & Google Drive Services
* **FR-BE-3.1:** Synthesize complete SOAP notes with explicit Subjective, Objective, Assessment, and Plan fields.
* **FR-BE-3.2:** Provide `/api/reframe-soap` endpoint allowing clinicians to adjust tone, detail level, or focus of individual SOAP sections.
* **FR-BE-3.3:** Generate two distinct ReportLab PDF artifacts:
  * `*_analysis_report.pdf`: Physician diagnostic analysis and NICE guideline evidence.
  * `*_soap_note.pdf`: Official SOAP note with patient discharge instructions.
* **FR-BE-3.4:** Upload generated reports to Google Drive via Service Account OAuth2 and return preview/download URLs.

### 4.4 Authentication, Scheduling & Email Services
* **FR-BE-4.1:** Generate secure 6-digit cryptographic OTPs with 5-minute time-to-live (TTL).
* **FR-BE-4.2:** Dispatch OTPs and booking confirmations via Gmail SMTP.
* **FR-BE-4.3:** Issue signed JSON Web Tokens (JWT) containing user ID, email, and role.
* **FR-4.4:** Doctor schedule configuration with automated 30-minute slot generation, buffer intervals, and double-booking prevention.
* **FR-BE-4.5:** Consultation Session Persistence & Symmetrical Suggestion Serialization: Persist finalized encounter session records (`PatientConsultationSession`) with complete transcript exchanges, clinical analysis, SOAP reports, and AI suggestions. Endpoints must symmetrically emit both `ai_suggestions` and `aiSuggestions` arrays to guarantee compatibility across diverse client schemas.
* **FR-BE-4.6:** Triage Vitals Baseline & Smart Dialogue Fallback Extraction: Patient retrieval schemas must emit baseline adult triage vitals (`bp`, `hr`, `spo2`, `temp`) rather than empty objects `{}`. Fallback entity extraction must analyze conversation text to extract reported symptoms, blood pressure, heart rate, and elevated blood glucose.
* **FR-BE-4.7:** Full Dynamic 5-Tab CDS Output Serialization: All audio upload endpoints (MS Teams, Google Meet, WhatsApp, In-Person) and dialogue analysis endpoints must return complete, dynamically synthesized clinical models for all 5 CDS tabs (`differential_diagnosis`, `guideline_medicines`, `red_flags`, `confidence_metrics`, `follow_up_questions`, `soap_report`). Fallback pipelines must generate condition-adaptive hypotheses with ICD-10 codes, dosages, 5-domain confidence breakdowns, and dialogue-derived SOAP reports.


---

## 5. Non-Functional Requirements

* **Throughput & Concurrency:** Asynchronous FastAPI endpoint handling using `async`/`await` and non-blocking I/O.
* **Database Agility:** SQLAlchemy ORM supporting both PostgreSQL (production) and SQLite (local development/testing).
* **Observability:** Native LangSmith tracing integration for tracking LLM token usage, latency, and LangGraph node transitions.
* **Resilience:** Fallback mechanisms when external APIs (e.g. Deepgram or OpenRouter) encounter transient rate limits.
* **Type Safety & Configuration Integrity:** Strict static type safety across environment settings loading and database connection strings, ensuring heterogeneous configuration types (integers, strings, booleans) are defensively validated and cast without runtime exceptions or static language server attribute errors.

---

## 6. Production Deployment & Cloudflare Zero Trust Ingress
* **FR-BE-6.1 Zero Trust Host Infrastructure:** Containerized deployment executing on Ubuntu 24.04 server `ubtsrv01` at internal IP `172.16.17.181`, secured through Cloudflare Zero Trust organization `cbtlab` (`cbtlab.cloudflareaccess.com`) via WARP client virtual interface (`10.64.20.180`).
* **FR-BE-6.2 Dockerized Microservice Architecture:** Orchestrated via Docker Compose v2 with 4 services (`postgres:16-alpine`, `redis:7-alpine`, `isb_project-backend`, `isb_project-frontend`).
* **FR-BE-6.3 Single-Origin Unified Ingress:** Ingress handled by Nginx 1.25 on port 80 routing to backend `:8000`, eliminating CORS preflight overhead.
* **FR-BE-6.4 Automated Telemetry Verification:** Real-time `/health` telemetry endpoint monitoring database health, tracing, and storage persistence. 100% test pass rate across 6 production verification routes.

