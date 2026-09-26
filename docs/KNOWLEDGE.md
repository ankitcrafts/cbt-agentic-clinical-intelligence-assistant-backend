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
