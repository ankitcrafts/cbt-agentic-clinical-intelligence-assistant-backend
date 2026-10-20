# Backend Technical & Domain Knowledge Base
## ClinIQ: Backend Clinical Knowledge & AI Engine Mechanics

---

## 1. Clinical Data Structuring & Domain Logic

### 1.1 Clinical Entity Taxonomy
The entity extractor parses unstructured transcripts into structured dictionary objects:
* `chief_complaint`: Primary symptom or reason for medical encounter.
* `symptoms`: List of identified symptoms with onset, duration, character, and severity (1-10).
* `history_of_present_illness (HPI)`: Chronological progression of symptoms.
* `past_medical_history (PMHx)`: Pre-existing chronic illnesses, surgeries, or hospitalizations.
* `current_medications`: Name, dosage, frequency, and adherence.
* `allergies`: Known drug and environmental hypersensitivities.
* `review_of_systems (ROS)`: Multi-system screening (Cardiovascular, Respiratory, Gastrointestinal, Neurological).

### 1.2 Differential Diagnosis Calculation
The differential diagnosis synthesizer executes multi-hypothesis evaluation:
$$\text{Score}(D) = w_1 \cdot \text{SymptomMatch}(D) + w_2 \cdot \text{GuidelineSimilarity}(D) - w_3 \cdot \text{Contraindications}(D)$$
Where each candidate $D$ is assigned a confidence percentage (e.g. 85%), rationale explanation, supporting evidence from NICE guidelines, and ICD-10 clinical coding.

---

## 2. RAG Guideline Subsystem & Vector Indexing

```mermaid
flowchart LR
    GuidelineDocs[NICE Guidelines .md / .pdf] --> TextSplitter[RecursiveCharacterTextSplitter<br>ChunkSize: 800, Overlap: 100]
    TextSplitter --> Embeddings[HuggingFace / OpenAI Embeddings]
    Embeddings --> ChromaVectorStore[(ChromaDB Vector Store<br>Collection: nice_guidelines)]
    
    Query[Extracted Patient Symptoms] --> QueryEmbedding[Generate Query Embedding]
    QueryEmbedding --> VectorSearch[Cosine Similarity Search (k=4)]
    ChromaVectorStore --> VectorSearch
    VectorSearch --> GuidelineContext[Top Matching Guideline Excerpts]
```

* **Index Management:** The `/rag/ingest` endpoint or automated lifespan check (`ensure_index()`) verifies vector collection existence. If missing, it parses documents in the `Rag/` or `knowledge-base/` directory and builds embeddings locally.
* **Vector Store Implementation:** ChromaDB runs in embedded persistent client mode, storing vector indexes in `.chroma/` or local memory.

---

## 3. Deepgram Medical Speech-to-Text & Diarization

* **Model Configuration:** `model: "nova-2-medical"`, `smart_format: true`, `diarize: true`, `punctuate: true`.
* **Speaker Mapping Algorithm:**
  ```python
  # Speaker channel analysis assigns speaker labels based on dialogue turn patterns
  for utterance in response["results"]["utterances"]:
      speaker_id = utterance["speaker"]
      role = "Doctor" if speaker_id == 0 else "Patient"
      parsed_dialogue.append({
          "speaker": role,
          "text": utterance["transcript"],
          "start": utterance["start"],
          "end": utterance["end"],
          "confidence": utterance["confidence"]
      })
  ```

---

## 4. Google Drive Cloud Archival Service

* **Service Account Authentication:** Utilizes Google Cloud IAM Service Account JSON key with scopes:
  * `https://www.googleapis.com/auth/drive.file`
  * `https://www.googleapis.com/auth/drive`
* **Folder Hierarchy:** Scans for an existing folder named `ClinIQ_Reports`. If absent, creates the folder with `mimeType: application/vnd.google-apps.folder`.
* **File Upload & Permissions:** Uploads generated PDF files, grants public/shared reader permissions, and returns both `id` and `webViewLink`.

---

## 5. Clinical AI Suggestions Generation & Schema

* **LLM Prompt Strategy:** Generated during active consultation from encounter summary, extracted medical entities, and differential diagnosis candidate list.
* **Schema Contract:**
  ```json
  [
    {
      "priority": "HIGH",
      "category": "CLINICAL EVALUATION",
      "title": "Clarify symptom onset timeline",
      "description": "Establish whether symptoms are acute, subacute, or progressive."
    }
  ]
  ```
* **Fallback Generator:** `PipelineService._fallback_suggestions()` ensures 3 evidence-grounded suggestions covering Clinical Evaluation, Medication Reconciliation, and Follow-Up if the LLM invocation times out or rate limits.

---

## 6. Triage Vitals Clinical Norms & Dialogue Extraction

* **Vital Signs Hierarchy:**
  * Primary: Blood Pressure (`BP`, mmHg), Heart Rate (`HR`, bpm), Oxygen Saturation (`SpO2`, %), Body Temperature (`Temp`, °C).
  * Metabolic / Diagnostic: Blood Glucose (`glucose`, mg/dL), Weight (`weight`, kg).
* **Extraction Strategy:**
  * When dialogues discuss fatigue, excessive thirst (polydipsia), and urinary frequency without formal BP readings, finger-prick glucose discussions ("numbers are indeed quite high") are mapped into the vital signs block as elevated blood sugar findings.
  * Ensures physical examination and triage metrics are always available for clinical decision support.

---

## 7. Guideline Medication & Condition Taxonomy for CDS Tabs

* **Type 2 Diabetes Mellitus (E11.9):**
  * Medications: Metformin Hydrochloride (500mg - 1000mg BID), Empagliflozin (10mg OD), Oral Rehydration Solution.
  * Guidelines: NICE NG28, ADA / EASD 2024 Standards of Care.
* **Essential Hypertension (I10):**
  * Medications: Lisinopril (10mg - 20mg OD), Amlodipine Besylate (5mg OD), Atorvastatin (20mg OD).
  * Guidelines: NICE NG136, JNC8, AHA/ACC Hypertension Guidelines.
* **Tension-Type Headache & Cephalea (G44.2):**
  * Medications: Paracetamol (500mg - 1000mg PRN), Ibuprofen (400mg PRN), Amitriptyline (10mg - 25mg bedtime prophylaxis).
  * Guidelines: SIGN 107 (Diagnosis and Management of Headache in Adults), BNF.
* **Acute Respiratory & Viral Infections (J06.9):**
  * Medications: First-line symptomatic antipyretics, saline hydration, supportive throat lozenges.
  * Guidelines: NICE CKS Upper Respiratory Tract Infections.


