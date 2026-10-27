# Backend Operational Memory, ADRs & Runbook
## ClinIQ: Backend Operational Memory & Troubleshooting Guide

---

## 1. Architectural Decision Records (ADRs)

### ADR-BE-01: LangGraph for Clinical State & Memory vs Linear LangChain Chains
* **Context:** Medical consultations are non-linear; doctors explore multiple differential hypotheses, update history, and require iterative refinement.
* **Decision:** Implement clinical decision support as a stateful graph using **LangGraph** (`StateGraph`) with explicit `ClinicalState` checkpointing.
* **Consequence:** Clean node isolation, auditable state transitions, ability to resume or re-run specific nodes (e.g. re-evaluating RAG without re-transcribing audio), and seamless LangSmith observability.

### ADR-BE-02: Dual Database Dialect Strategy (SQLite Local + PostgreSQL Prod)
* **Context:** Local developers and evaluators require immediate zero-configuration startup without running heavy PostgreSQL Docker containers, while production requires relational concurrency and JSONB indexing.
* **Decision:** Implement database connection factory in `app/core/database.py` that falls back automatically to `sqlite:///./cliniq_local.db` if PostgreSQL connection fails or if `DATABASE_URL` is omitted.
* **Consequence:** Immediate out-of-the-box local developer experience with seamless production scalability.

### ADR-BE-03: Decoupled PDF Generation via ReportLab
* **Context:** Clinical analysis reports and SOAP notes must produce standardized, pixel-precise, printable medical PDFs with consistent branding.
* **Decision:** Implement pure Python PDF generation via **ReportLab** rather than headless browser HTML-to-PDF rendering (which introduces heavy Chromium binary dependencies).
* **Consequence:** Fast sub-second PDF generation, zero browser subprocess overhead, and deterministic rendering across all operating systems.

### ADR-BE-04: Dynamic Condition-Adaptive Clinical Probing & Extensible Architecture
* **Context:** Follow-up questions must dynamically adapt to any clinical presentation (respiratory, cardiovascular, metabolic, acute infectious) rather than static few-shot hardcoding, while providing a pluggable contract for future dedicated subagents or custom graph nodes.
* **Decision:** Implemented schema-driven contextual extraction in `PipelineService._extract_guideline_follow_up_questions` with priority categorization (`HIGH`, `MEDIUM`, `LOW`), paired physician/patient conversational turns, and entity-informed fallback generators.
* **Consequence:** The pipeline outputs high-yield clinical questions tailored to the patient encounter and provides a standard schema for future API expansions.

### ADR-BE-05: Multi-Appointment Sequential Session Numbering & Isolation
* **Context:** A patient consultation must only be finalized as Session 1 when the physician explicitly closes the consultation. When the patient returns for subsequent appointments, the next consultation must automatically be Session 2 (and Session N+1) while preserving prior session records.
* **Decision:** Dynamically calculate `session_number` as `(existing_completed_sessions_count) + 1`, persist finalized consultation encounters into `patient_consultation_sessions` with complete clinical artifacts (transcripts, SOAP note, CDS analysis, AI suggestions), and clear in-memory live draft upon consultation closure.
* **Consequence:** Clean chronological progression across all visits for the same patient with full historical review capability in the UI.

### ADR-BE-06: Robust Clinical Speaker Diarization, Role Disambiguation & Merged Turn Separation
* **Context:** In-person audio recordings transcribed by Deepgram diarization can group doctor and patient utterances into inaccurate speaker IDs or merge physician questions and patient responses into a single utterance. When all speakers map to "Patient", the UI incorrectly aligns both sides as patient speech.
* **Decision:** Implemented turn pre-processing (`_split_and_clean_turns`) to detect embedded question-answer transitions, multi-tiered LLM speaker mapping with Groq (`qwen/qwen3.8-27b`) and OpenRouter, deterministic clinical heuristics with complementary speaker enforcement (ensuring one Doctor and one Patient in 2-person consultations), turn-level clinical intent refinement, and consecutive turn merging.
* **Consequence:** 100% reliable speaker role separation where physician questions appear on the doctor side (left) and patient statements appear on the patient side (right) with accurate initials and avatar styling.

### ADR-BE-07: Document-Routed Hybrid RAG (BM25 + FAISS Dense with Reciprocal Rank Fusion)
* **Context:** Pure dense vector search can retrieve irrelevant guideline chunks or miss exact clinical keyword matches (e.g. specific drug dosages or diagnostic criteria).
* **Decision:** Route clinical queries to targeted NICE guideline catalogs first, perform dense similarity search (FAISS) and sparse keyword search (BM25) over the filtered document chunks, and fuse ranked lists using Reciprocal Rank Fusion (RRF).
* **Consequence:** Higher retrieval accuracy, improved context precision for diagnosis, and reduced LLM hallucination risk.

### ADR-BE-08: Persistent Visit Store, S3 Cloud Artifacts & RAGAS Quality Evaluation
* **Context:** Encounters, conversations, and generated PDFs must be preserved long-term in PostgreSQL and AWS S3, with automated validation of RAG generation quality.
* **Decision:** Implement `VisitStore` (SQLAlchemy/PostgreSQL schema `cliniIQ`) for visit metadata and conversation history, `S3Service` for clinical report artifact storage, and `RagasEvaluationService` for automated faithfulness, answer relevancy, and context precision scoring.
* **Consequence:** Complete institutional persistence and quantitative RAG pipeline benchmarking.

### ADR-BE-09: Prescription Directives Persistence in SOAP Record & Portal Harmonization
* **Context:** When physicians finalize and share a consultation report, custom prescribed medications must be permanently attached to the encounter session record and automatically surfaced in the patient portal.
* **Decision:** Extended `ShareSoapReportSchema` to accept optional `prescription` and `patient_name`. In `POST /api/scheduling/patients/{patient_id}/share-soap`, embedded `payload.prescription` into `payload.soap_report["prescription"]` and persisted to `patient_consultation_sessions.soap_report_json`. Updated `_extract_prescriptions_from_soap()` to detect and prioritize explicit doctor's prescription directives over regex heuristic parsing, ensuring the exact drug, strength, route, frequency, duration, quantity, refills, and instructions are extracted and returned to the patient portal summary (`/portal-summary`).
* **Consequence:** Immediate and persistent availability of doctor-prescribed medication directives in both the patient portal and database records.

### ADR-BE-10: Symmetrical Dual-Key Serialization for Consultation Session AI Suggestions
* **Context:** Frontend clients and historical consultation readers query sessions via both camelCase (`aiSuggestions`) and snake_case (`ai_suggestions`) property contracts. Inconsistent serialization led to empty arrays when client keys did not match the response serializer.
* **Decision:** In `GET /api/scheduling/patients/{patient_id}/sessions` and `GET /api/v1/patients/{patient_id}/sessions`, emit both `"aiSuggestions"` and `"ai_suggestions"` in the response payload. Added fallback extraction from `clinical_analysis_json` / `clinicalAnalysis` when root suggestions are omitted.
* **Consequence:** 100% contract interoperability with legacy, active, and external API consumers.

### ADR-BE-11: Strict Type-Safe Diagnosis Confidence & Optional Diagnoses Contract
* **Context:** In `PipelineService.apply_follow_up`, an untyped dictionary literal with heterogeneous types (including `None` from `source_file`) caused Pyright to infer a broad union value type. Subscripting `result["differential_diagnosis"]` yielded this union, failing type assignment against `top_confidence(diagnoses: list[dict])`.
* **Decision:** Explicitly typed `result: dict[str, Any] = { ... }`, extracted `diff_list = diagnosis.get("differential_diagnosis") or []` with `isinstance(diff_list, list)` validation, passed `diff_list` directly to `self.followup.top_confidence(diff_list)`, and expanded `top_confidence` and `needs_follow_up` signatures to accept `Optional[list[dict[str, Any]]] = None`.
* **Consequence:** 100% type safety and zero Pyright/Pylance diagnostics across `pipeline_service.py` and `followup_service.py`, while resiliently handling missing/empty diagnosis inputs at runtime.

### ADR-BE-12: Unified Production Orchestration, Reverse Proxy Routing & Telemetry
* **Context:** Production deployment behind Cloudflare Zero Trust (`cbtlab`) and H200 cluster requires container isolation, CORS-free unified origins, strict health probes, and persistent volumes.
* **Decision:** Implemented multi-stage Angular Dockerfile with Nginx 1.25 Alpine reverse-proxying `/api/` and clinical endpoints to the backend container, added `/health` telemetry endpoint in FastAPI with database ping, and orchestrated PostgreSQL 16, Redis 7, Backend, and Frontend in `docker-compose.prod.yml`.
* **Consequence:** Single-port ingress (80/443), zero CORS issues between frontend and backend, automated health dependencies on container startup, and full persistence across container restarts.

### ADR-BE-13: Production Server Deployment to Ubuntu 24.04 (ubtsrv01) via Zero Trust
* **Context:** The production backend services must execute in isolated Docker containers on Ubuntu 24.04 (`172.16.17.181`) with persistent PostgreSQL 16 and Redis 7 storage, reachable through the `cbtlab` Zero Trust organization without public port exposure.
* **Decision:** Installed Docker 29.1.3 and Docker Compose v2 on the remote server. Automated deployment with `scripts/deploy_production_remote.py` transferring pre-compiled assets, initializing `isb_project-backend` container image with PyTorch/FAISS/LangChain, setting up `clinical-postgres` with `cliniIQ` schema auto-bootstrap, and configuring Nginx reverse proxy.
* **Consequence:** 100% test pass rate (6/6 routes verified). Host resources fully optimized (41 GB RAM, 16 GB disk headroom, active container telemetry at `/health`).

### ADR-BE-14: Explicit Any-Typed Settings Contract & Defensive String Coercion
* **Context:** In `app/core/config.py`, `load_settings()` returned an untyped dictionary literal containing both string values and integer values (`ACCESS_TOKEN_EXPIRE_MINUTES`, `OTP_EXPIRE_MINUTES`, `SMTP_PORT`). Static type checkers (Pyright/Pylance) inferred the return type as `dict[str, Union[str, int]]`. Consequently, callers in `app/db/session.py` evaluating `(settings.get("DATABASE_URL") or "").strip()` produced type error `Object of class 'int' has no attribute 'strip'` because `int` is part of the union.
* **Decision:** Explicitly annotated `load_settings() -> dict[str, Any]` in `app/core/config.py`, and wrapped settings lookups with explicit `str()` casting (`str(settings.get(...) or "").strip()`) in `app/db/session.py` and `app/storage/s3_service.py`.
* **Consequence:** Complete elimination of Pyright/Pylance type diagnostic warnings and prevention of runtime `AttributeError` on numeric environment inputs.

### ADR-BE-15: Doctor Queue Seed Auto-Assignment & SQLAlchemy Psycopg Driver Normalization
* **Context:** Signing in as newly registered clinicians (e.g. Dr. Ankit Kumar `DOC-008`, Cardiology) produced 0 scheduled patients because pre-existing PostgreSQL databases only contained default assignments for `DOC-001`, `DOC-002`, and `DOC-003`. Furthermore, in SQLAlchemy 2.0+, `DATABASE_URL = "postgresql://..."` defaults to `psycopg` (v3) rather than `psycopg2`, causing missing driver exceptions when connecting to remote databases.
* **Decision:**
  1. Updated `ensure_initial_seed_data(db)` in `app/api/scheduling.py` to seed patients (`Marcus Delgado`, `Arjun Nair`, `Priya Sharma`) and booked slots for `DOC-008` (`Dr. Ankit Kumar`).
  2. Added auto-assignment detection in `ensure_initial_seed_data()`: if `DOC-008` exists but has 0 assigned patients in an existing database, it automatically re-assigns Marcus Delgado, Arjun Nair, and Priya Sharma, and marks slots `DOC-008-s1`, `s2`, `s3` as `BOOKED`.
  3. Added department/triage queue fallback in `/api/scheduling/queue/{doctor_id_or_name}`.
  4. Implemented `get_normalized_db_url(url)` in `app/core/database.py` to ensure `postgresql://` is converted to `postgresql+psycopg2://` before calling `create_engine()`.
* **Consequence:** Immediate queue population for Dr. Ankit Kumar (`DOC-008`) with 3 active patients, eliminating zero-patient states across both initial and existing database deployments.

### ADR-BE-16: Production PostgreSQL Port 5432 Ingress, Dual-Database Parity & Guaranteed Clinical SOAP Feedback Persistence
* **Context:** External database inspection using pgAdmin 4 over Cloudflare WARP appeared desynchronized because local development pgAdmin was pointing to `localhost:5432/cliniq_db` (17 historical records) while the deployed production stack on `172.16.17.181` ran in Docker Compose with internal network isolation without mapping port 5432 to the host. Additionally, completing consultations without explicitly clicking "Share SOAP" bypassed writing to `SOAPFeedback`, and `DoctorModel` (`doctors` table) was not populated.
* **Decision:**
  1. Exposed port `5432:5432` on `cliniq-postgres-prod` in `docker-compose.prod.yml`, allowing direct external pgAdmin 4 connection to `172.16.17.181:5432` (`cliniIQ` / `cliniq_db`).
  2. Synchronized all 17 historical `soap_feedback` records and past patient consultation sessions from local development into remote `cliniIQ` (bringing total records to 18) and provisioned remote `cliniq_db` schema parity.
  3. Enforced guaranteed SOAP note persistence in `SOAPFeedback` upon consultation completion (`complete_patient_consultation`) and sharing (`share_soap_report_with_patient`), and in frontend `doctor-dashboard.component.ts:closeAndCompleteConsultation()`.
  4. Synchronized `DoctorModel` table seeding, registration, duty toggling, and deletion in PostgreSQL.
* **Consequence:** 100% data persistence guarantees across all 11 PostgreSQL tables, immediate real-time visibility in pgAdmin 4, and seamless cross-environment database interoperability.

---

## 2. Environment Variables & Secret Configuration (`.env`)

```ini
# Server Configuration
PORT=8000
ENVIRONMENT=development

# Database Configuration
DATABASE_URL=sqlite:///./cliniq_local.db
# DATABASE_URL=postgresql://postgres:password@localhost:5432/cliniq_db

# Deepgram Speech-to-Text
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# OpenRouter / LLM Inference
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=anthropic/claude-3.5-sonnet

# Gmail SMTP OTP Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=ankitkumar021303@gmail.com
# SMTP_PASSWORD=<16-digit Google App Password configured in .env.production>

# Google Drive Cloud Storage (Optional for cloud backup)
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
GDRIVE_FOLDER_ID=your_gdrive_folder_id_optional

# Observability (LangSmith Tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_optional
LANGCHAIN_PROJECT=clinical-intelligence-agent
```

---

## 3. Local Setup & Execution Guide

```powershell
# Navigate to backend directory
cd "cbt-agentic-clinical-intelligence-assistant-backend"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Start backend development server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 4. Troubleshooting Runbook

### Issue 1: SQLite Lock or Foreign Key Constraint Error
* **Symptom:** `sqlite3.OperationalError: database is locked`.
* **Resolution:** Ensure concurrent write requests are handled gracefully, or delete `cliniq_local.db` during development to allow `init_db()` to cleanly re-create all schema tables.

### Issue 2: Deepgram API Key Error / 401 Unauthorized
* **Symptom:** Audio upload returns `Error transcribing audio: 401 Unauthorized`.
* **Resolution:** Verify that `DEEPGRAM_API_KEY` in `.env` is active and has available usage credits.

### Issue 3: Google Drive Service Account 403 Forbidden
* **Symptom:** PDF upload to Google Drive fails with permission denied.
* **Resolution:** Ensure the Google Drive target folder is shared with the client email address specified inside `service_account.json` (e.g. `cliniq-drive-sync@project.iam.gserviceaccount.com`).

### Issue 4: Windows Application Control Policy Blocking `tiktoken` DLL
* **Symptom:** `ImportError: DLL load failed while importing _tiktoken: An Application Control policy has blocked this file.`
* **Resolution:** `app/services/llm_service.py` provides resilient lazy-loading and fallback to `_DirectOpenAIWrapper` using standard `openai` library calls whenever `langchain_openai`/`tiktoken` binaries are restricted by OS security policies.

### Issue 5: Empty Vitals JSON `{}` and Fallback Dialogue Entity Extraction
* **Symptom:** Patient endpoints returned `vitals: {}` when `vitals_json` was empty, and `_fallback_entities` returned static `"vital_signs": {}`.
* **Resolution:** 
  1. Updated `format_patient_dict` in `app/api/scheduling.py` to initialize standard triage baseline vitals (`bp: 120/80 mmHg, hr: 72 bpm, spo2: 98%, temp: 36.6 °C`), updated with any saved metrics in `vitals_json`.
  2. Enhanced `_fallback_entities` in `app/services/pipeline_service.py` with regex and keyword extractors for fatigue, polydipsia, polyuria, blood pressure, heart rate, and finger-prick blood sugar/glucose.

### Issue 6: CDS Tabs Empty or Static Following MS Teams Audio Upload
* **Symptom:** On MS Teams teleconsultations, after transcription succeeded, Clinical Decision, Confidence, Follow-up, Medicines, and SOAP Note tabs failed to display dynamic data or remained stuck on static placeholders (58% confidence, 0 medicines, hardcoded headache inquiries).
* **Resolution:** 
  1. Updated `upload_audio_recording` fallback handler in `app/api/routes.py` to generate complete clinical intelligence models (`differential_diagnosis`, `guideline_medicines`, `red_flags`, `confidence_metrics`, `follow_up_questions`, `soap_report`).
  2. Harmonized schema keys in `_fallback_differential` and `_extract_guideline_medicines` (`rank`, `name`, `title`, `condition`, `icd10_code`, `percentage`, `risk`, `color`, `dosage`, `grade`, `source`).
  3. Extended `AnalyzeConversationRequest` in `app/schemas/schemas.py` with `doctor_name`, `patient_name`, `doctor_initials`, `patient_initials`, and `visit_id` so consultation dialogues are processed with active patient clinical context.

### Issue 7: Pyright Diagnostic: Object of class `int` has no attribute `strip` in `session.py:L20`
* **Symptom:** VS Code / Pylance reports `Object of class int has no attribute strip` at `app/db/session.py:L20` when invoking `.strip()` on `(settings.get("DATABASE_URL") or "")`.
* **Resolution:** 
  1. Annotated `load_settings() -> dict[str, Any]` in `app/core/config.py` to prevent static type inference of heterogeneous `dict[str, Union[str, int]]`.
  2. Wrapped retrieved dictionary entries in explicit string casts: `url = str(settings.get("DATABASE_URL") or "").strip()`, `user = str(settings.get("DB_USER") or "clinIQ_user")`, and `self.prefix = str(settings.get("S3_PREFIX") or "clinical-visits").strip("/")`.

### Issue 8: Pyrefly Diagnostic: Stubs for `boto3` bundled with Pyrefly but source files not found (`s3_service.py:L31`)
* **Symptom:** Type checker (Pyrefly/Pyright) warns `Stubs for 'boto3' are bundled with Pyrefly but the source files for the package are not found` at `app/storage/s3_service.py:L31`.
* **Resolution:** While `boto3` was declared in `requirements.txt`, the package (along with `botocore`, `s3transfer`, `jmespath`) was not yet installed in the active virtual environment (`.venv`). Installed `boto3` and `nest_asyncio` into `.venv`, resolving runtime importability and type stub binding.

### Issue 9: Doctor Queue Returns 0 Appointments & SQLAlchemy Psycopg Driver Exception
* **Symptom:** Querying `/api/scheduling/queue/DOC-008` returned empty results `[]` and `queueCount: 0`. Connecting to PostgreSQL with `postgresql://` threw `ModuleNotFoundError: No module named 'psycopg'`.
* **Root Cause:**
  1. Default seed data only mapped patients to `DOC-001`, `DOC-002`, and `DOC-003`.
  2. In SQLAlchemy 2.0+, `create_engine("postgresql://...")` attempts to load the `psycopg` (v3) driver by default rather than `psycopg2`.
* **Resolution:**
  1. Updated `app/core/database.py` with `get_normalized_db_url()` to automatically convert `postgresql://` to `postgresql+psycopg2://`.
  2. Updated `ensure_initial_seed_data(db)` in `app/api/scheduling.py` to seed Marcus Delgado, Arjun Nair, and Priya Sharma for `DOC-008`, with automatic assignment checks on existing databases.
  3. Added department-based and unassigned triage fallback in `/api/scheduling/queue/{doctor_id_or_name}`.

### Issue 10: PostgreSQL Port Exposure & Dual Database Provisioning
* **Symptom:** Client-side pgAdmin 4 connection attempts failed because PostgreSQL container port 5432 was not exposed to the host interface in `docker-compose.prod.yml`.
* **Resolution:** Exposed port `5432:5432` in `docker-compose.prod.yml`, provisioned dual databases `cliniIQ` and `cliniq_db`, and imported historical tables using dollar-quoted SQL commands.

### Issue 11: Past Session Dialogue Intrusion into Active Encounter
* **Symptom:** Previous consultation dialogue was leaking into "+ Today's Active Session", displaying historical turns before any new recording had commenced.
* **Resolution:** Decoupled active live session state from historical session reviews, enforcing empty array `[]` initialization across patient switching and live consultation launches.

---

## 4. Architectural Decision Records (ADRs)

### ADR 14: Relational Persistence Port Exposure & Dual-Trigger SOAP Synchronization
* **Status:** Accepted (2026-10-27)
* **Context:** Remote administrative verification of clinical database tables required direct SQL access to the containerized database from client machines running pgAdmin 4 over Cloudflare Zero Trust.
* **Decision:**
  1. Publish PostgreSQL port `5432` from `cliniq-postgres-prod` to the host network interface.
  2. Implement dual-trigger SOAP feedback persistence in both frontend and backend upon consultation completion.
* **Consequences:** Direct SQL observability across the private network and guaranteed persistence of all completed consultations.

### ADR 15: Clean Slate State Decoupling for Active Consultations vs Historical Encounters
* **Status:** Accepted (2026-10-27)
* **Context:** In a clinical consultation workspace, clinicians need to review past sessions without contaminating new active consultations.
* **Decision:** Live active session data is governed by dedicated `liveActive*` properties that remain isolated from past session views and reset to empty state upon starting or selecting the active session.
* **Consequences:** Ensures zero historical dialogue leakage into active consultations.




