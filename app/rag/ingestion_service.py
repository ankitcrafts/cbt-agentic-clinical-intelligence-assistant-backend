from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import GUIDELINES_DIR, VECTORSTORE_DIR
from app.core.tracing import traceable
from app.rag.guideline_catalog import metadata_for_filename

FAISS_INDEX_DIR = VECTORSTORE_DIR / "faiss_index"


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
            meta = metadata_for_filename(pdf_path.name)
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata.update(
                    {
                        "source_file": pdf_path.name,
                        "doc_id": meta["doc_id"],
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
            "doc_ids": sorted({metadata_for_filename(p.name)["doc_id"] for p in pdf_files}),
        }

    def is_indexed(self) -> bool:
        return (FAISS_INDEX_DIR / "index.faiss").exists() and (FAISS_INDEX_DIR / "index.pkl").exists()
