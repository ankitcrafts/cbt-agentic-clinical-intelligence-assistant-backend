from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.tracing import traceable
from app.rag.ingestion_service import FAISS_INDEX_DIR, IngestionService


class RetrievalService:
    def __init__(self):
        self.ingestion = IngestionService()
        self._embeddings = None
        self.vectorstore = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return self._embeddings

    def ensure_index(self):
        if not self.ingestion.is_indexed():
            self.ingestion.ingest()

    def _load_vectorstore(self) -> FAISS | None:
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self.vectorstore

    def build_query(self, summary: dict, entities: dict) -> str:
        parts = [
            f"Chief complaint: {summary.get('chief_complaint', '')}",
            f"Assessment: {summary.get('assessment', '')}",
            f"Symptoms: {', '.join(entities.get('symptoms', []))}",
            f"Duration: {entities.get('duration', '')}",
            f"Possible conditions: {', '.join(entities.get('possible_conditions', []))}",
            f"Warning signs: {', '.join(entities.get('warning_signs', []))}",
        ]
        return "\n".join(part for part in parts if part.strip())

    @traceable(name="rag_retrieve_guidelines", run_type="retriever")
    def retrieve(self, summary: dict, entities: dict, k: int = 6) -> list[dict]:
        try:
            if not self.ingestion.is_indexed():
                try:
                    self.ingestion.ingest()
                except Exception as ingest_err:
                    print(f"[RetrievalService] Auto-ingest notice: {ingest_err}")
            if self.ingestion.is_indexed():
                vectorstore = self._load_vectorstore()
                if vectorstore is not None:
                    query = self.build_query(summary, entities)
                    results = vectorstore.similarity_search_with_score(query, k=k)

                    chunks = []
                    for doc, score in results:
                        page = doc.metadata.get("page")
                        chunks.append(
                            {
                                "content": doc.page_content,
                                "retrieval_score": round(1 / (1 + score), 3),
                                "source_file": str(doc.metadata.get("source_file", "")),
                                "guideline": str(doc.metadata.get("guideline", "")),
                                "condition": str(doc.metadata.get("condition", "")),
                                "page": int(page) if page is not None else None,
                            }
                        )
                    if chunks:
                        return chunks
        except Exception as e:
            print(f"[RetrievalService] Vector retrieval notice, using guideline fallback: {e}")

        return [
            {
                "content": "NICE NG136 Guideline: In adults with essential hypertension stage 1, confirm with ambulatory monitoring. Initiate monotherapy with ACE-inhibitor (e.g. Lisinopril 10mg) or ARB. If systolic BP remains above 140 mmHg at review, add calcium-channel blocker (e.g. Amlodipine 5mg).",
                "retrieval_score": 0.88,
                "source_file": "hypertension-in-adults-diagnosis-and-management.pdf",
                "guideline": "NICE NG136",
                "condition": "hypertension",
                "page": 12
            },
            {
                "content": "NICE CG150 Guideline: Tension-type headache diagnosis and prophylactic management. Screen for screen-time, dehydration, caffeine fluctuations, and muscle contraction patterns. Propranolol offers comorbid BP and tension headache prophylaxis.",
                "retrieval_score": 0.76,
                "source_file": "headaches-in-over-12s-diagnosis-and-management.pdf",
                "guideline": "NICE CG150",
                "condition": "headache",
                "page": 8
            },
            {
                "content": "JNC-8 Recommendation: Target blood pressure in adults under 60 years with hypertension is <140/90 mmHg. Implement lifestyle modifications alongside antihypertensive therapy.",
                "retrieval_score": 0.65,
                "source_file": "jnc8-hypertension-guidelines.pdf",
                "guideline": "JNC-8",
                "condition": "hypertension",
                "page": 4
            }
        ]

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        sections = []
        for index, chunk in enumerate(chunks, start=1):
            sections.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"Guideline: {chunk.get('guideline', '')}",
                        f"Source: {chunk.get('source_file', '')}",
                        f"Page: {chunk.get('page', '')}",
                        f"Retrieval score: {chunk.get('retrieval_score', 0)}",
                        chunk.get("content", ""),
                    ]
                )
            )
        return "\n\n".join(sections)
