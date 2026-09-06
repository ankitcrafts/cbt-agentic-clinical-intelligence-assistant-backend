from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import GUIDELINES_DIR, VECTORSTORE_DIR
from app.core.tracing import traceable

FAISS_INDEX_DIR = VECTORSTORE_DIR / "faiss_index"


def _guideline_metadata(filename: str) -> dict:
    lowered = filename.lower()
    if "ng109" in lowered or "uti" in lowered or "urinary" in lowered:
        return {"guideline": "NICE NG109", "condition": "lower urinary tract infection"}
    if "hypertension" in lowered or "blood pressure" in lowered:
        return {"guideline": "NICE NG136", "condition": "hypertension"}
    if "diabetes" in lowered:
        return {"guideline": "NICE NG28", "condition": "type 2 diabetes"}
    return {"guideline": "clinical guideline", "condition": "general"}


class IngestionService:
    def __init__(self):
        self._embeddings = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
        )

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return self._embeddings

    @traceable(name="rag_ingest_guidelines", run_type="tool")
    def ingest(self) -> dict:
        pdf_files = sorted(GUIDELINES_DIR.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found under {GUIDELINES_DIR}")

        documents = []
        for pdf_path in pdf_files:
            meta = _guideline_metadata(pdf_path.name)
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata.update(
                    {
                        "source_file": pdf_path.name,
                        "guideline": meta["guideline"],
                        "condition": meta["condition"],
                    }
                )
            documents.extend(pages)

        chunks = self.splitter.split_documents(documents)
        if FAISS_INDEX_DIR.exists():
            shutil.rmtree(FAISS_INDEX_DIR)
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(str(FAISS_INDEX_DIR))

        return {
            "pdf_count": len(pdf_files),
            "chunk_count": len(chunks),
            "vectorstore": str(FAISS_INDEX_DIR),
            "files": [path.name for path in pdf_files],
        }

    def is_indexed(self) -> bool:
        return (FAISS_INDEX_DIR / "index.faiss").exists() and (FAISS_INDEX_DIR / "index.pkl").exists()
