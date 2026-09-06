from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.tracing import traceable
from app.rag.ingestion_service import FAISS_INDEX_DIR, IngestionService


class RetrievalService:
    def __init__(self):
        self.ingestion = IngestionService()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None

    def ensure_index(self):
        if not self.ingestion.is_indexed():
            self.ingestion.ingest()

    def _load_vectorstore(self):
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                str(FAISS_INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

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
        if not self.ingestion.is_indexed():
            raise FileNotFoundError(
                "Guideline index not found. Call GET /rag/ingest first."
            )
        self._load_vectorstore()
        query = self.build_query(summary, entities)
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        chunks = []
        for doc, score in results:
            page = doc.metadata.get("page")
            chunks.append(
                {
                    "content": doc.page_content,
                    "retrieval_score": float(round(1 / (1 + float(score)), 3)),
                    "source_file": str(doc.metadata.get("source_file", "")),
                    "guideline": str(doc.metadata.get("guideline", "")),
                    "condition": str(doc.metadata.get("condition", "")),
                    "page": int(page) if page is not None else None,
                }
            )
        return chunks

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
