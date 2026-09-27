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
GMAIL_USER=your_clinic_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_16_digit_app_password

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
