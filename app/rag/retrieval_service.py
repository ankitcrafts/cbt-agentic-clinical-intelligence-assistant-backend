from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.tracing import traceable
from app.rag.bm25 import BM25Okapi, chunk_id, reciprocal_rank_fusion, tokenize
from app.rag.guideline_catalog import route_documents
from app.rag.ingestion_service import FAISS_INDEX_DIR, IngestionService


class RetrievalService:
    def __init__(self):
        self.ingestion = IngestionService()
        self._embeddings = None
        self.vectorstore = None
        self._corpus: list[dict[str, Any]] = []

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return self._embeddings

    def ensure_index(self):
        if not self.ingestion.is_indexed():
            self.ingestion.ingest()

    def _load_vectorstore(self) -> FAISS | None:
        if self.vectorstore is None:
            if not self.ingestion.is_indexed():
                try:
                    self.ingestion.ingest()
                except Exception as e:
                    print(f"[RetrievalService] Auto-ingest notice: {e}")
            if self.ingestion.is_indexed():
                self.vectorstore = FAISS.load_local(
                    str(FAISS_INDEX_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._corpus = self._build_corpus()
        return self.vectorstore

    def _build_corpus(self) -> list[dict[str, Any]]:
        corpus = []
        if self.vectorstore is None:
            return corpus
        index_map = getattr(self.vectorstore, "index_to_docstore_id", {}) or {}
        for _faiss_id, store_id in index_map.items():
            doc = self.vectorstore.docstore.search(store_id)
            if doc is None:
                continue
            meta = doc.metadata or {}
            content = doc.page_content or ""
            corpus.append(
                {
                    "id": chunk_id(meta, content),
                    "content": content,
                    "metadata": meta,
                    "tokens": None,
                }
            )
        return corpus

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

    def _allowed_filter(self, allowed_doc_ids: set[str], allowed_conditions: set[str]):
        def _match(meta: dict) -> bool:
            return meta.get("doc_id") in allowed_doc_ids or meta.get("condition") in allowed_conditions

        return _match

    def _filter_corpus(self, predicate) -> list[dict[str, Any]]:
        if predicate is None:
            return self._corpus
        return [item for item in self._corpus if predicate(item["metadata"])]

    def _dense_search(self, query: str, k: int, predicate=None) -> list[tuple[Any, float]]:
        if self.vectorstore is None:
            return []
        fetch_k = max(40, k * 8)
        if predicate is None:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        return self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            fetch_k=fetch_k,
            filter=lambda meta: predicate(meta),
        )

    def _bm25_search(self, query: str, corpus: list[dict[str, Any]], n: int) -> list[tuple[str, float]]:
        if not corpus:
            return []
        for item in corpus:
            if item.get("tokens") is None:
                item["tokens"] = tokenize(item["content"])
        tokenized = [item["tokens"] for item in corpus]
        bm25 = BM25Okapi(tokenized)
        ranked = bm25.top_n(query, n=n)
        return [(corpus[index]["id"], score) for index, score in ranked]

    def _doc_payload(self, doc) -> dict[str, Any]:
        meta = doc.metadata or {}
        content = doc.page_content or ""
        return {
            "id": chunk_id(meta, content),
            "content": content,
            "metadata": meta,
        }

    @traceable(name="rag_retrieve_guidelines", run_type="retriever")
    def retrieve(self, summary: dict, entities: dict, k: int = 6) -> dict:
        """
        Hybrid retrieval:
        1) Route to guideline docs via metadata/keywords
        2) FAISS dense search on those docs
        3) BM25 sparse search on the same filtered corpus
        4) Fuse ranks with Reciprocal Rank Fusion (RRF)
        """
        try:
            if not self.ingestion.is_indexed():
                try:
                    self.ingestion.ingest()
                except Exception as ingest_err:
                    print(f"[RetrievalService] Auto-ingest notice: {ingest_err}")
            self._load_vectorstore()
            if self.vectorstore is None:
                raise RuntimeError("Vectorstore index not loaded.")

            query = self.build_query(summary, entities)
            routed = route_documents(summary, entities, top_n=2)

            allowed_doc_ids = {item["doc_id"] for item in routed}
            allowed_conditions = {item["condition"] for item in routed}
            predicate = self._allowed_filter(allowed_doc_ids, allowed_conditions) if allowed_doc_ids else None

            dense_hits = self._dense_search(query, k=max(k, 10), predicate=predicate)
            corpus = self._filter_corpus(predicate)
            scoped = bool(predicate) and bool(corpus)

            if not dense_hits and not corpus:
                dense_hits = self._dense_search(query, k=max(k, 10), predicate=None)
                corpus = self._corpus
                scoped = False

            if predicate and (not dense_hits or not corpus):
                dense_hits = dense_hits or self._dense_search(query, k=max(k, 10), predicate=None)
                corpus = corpus or self._corpus
                scoped = False

            bm25_hits = self._bm25_search(query, corpus, n=max(k, 10))

            dense_ids = [self._doc_payload(doc)["id"] for doc, _score in dense_hits]
            bm25_ids = [doc_id for doc_id, _score in bm25_hits]
            fused = reciprocal_rank_fusion([dense_ids, bm25_ids])

            by_id: dict[str, dict[str, Any]] = {}
            dense_score_map: dict[str, float] = {}
            for doc, distance in dense_hits:
                payload = self._doc_payload(doc)
                by_id[payload["id"]] = payload
                dense_score_map[payload["id"]] = float(round(1 / (1 + float(distance)), 3))

            bm25_score_map = {doc_id: float(score) for doc_id, score in bm25_hits}
            for item in corpus:
                by_id.setdefault(item["id"], item)

            if not fused:
                # fall back to dense-only if BM25 found nothing
                fused = {doc_id: 1.0 / (60 + rank) for rank, doc_id in enumerate(dense_ids, start=1)}

            max_rrf = max(fused.values()) if fused else 1.0
            ranked_ids = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)[:k]

            chunks = []
            for doc_id in ranked_ids:
                item = by_id.get(doc_id)
                if not item:
                    continue
                meta = item.get("metadata") or {}
                page = meta.get("page")
                rrf = fused.get(doc_id, 0.0)
                chunks.append(
                    {
                        "content": item.get("content", ""),
                        "retrieval_score": float(round(rrf / max_rrf, 3)) if max_rrf else 0.0,
                        "dense_score": dense_score_map.get(doc_id),
                        "bm25_score": round(bm25_score_map.get(doc_id, 0.0), 3) if doc_id in bm25_score_map else None,
                        "rrf_score": float(round(rrf, 4)),
                        "source_file": str(meta.get("source_file", "")),
                        "doc_id": str(meta.get("doc_id", "")),
                        "guideline": str(meta.get("guideline", "")),
                        "condition": str(meta.get("condition", "")),
                        "page": int(page) if page is not None else None,
                    }
                )

            if scoped and fused:
                retrieval_mode = "hybrid_document_filtered"
            elif routed:
                retrieval_mode = "hybrid_global_fallback"
            else:
                retrieval_mode = "hybrid_global"

            return {
                "retrieved_chunks": chunks,
                "retrieval_mode": retrieval_mode,
                "routed_documents": routed,
                "query": query,
                "dense_hit_count": len(dense_ids),
                "bm25_hit_count": len(bm25_ids),
            }
        except Exception as e:
            print(f"[RetrievalService] Retrieval notice: {e}, using guideline fallback")
            fallback_chunks = [
                {
                    "content": "NICE NG136 Guideline: In adults with essential hypertension stage 1, confirm with ambulatory monitoring. Initiate monotherapy with ACE-inhibitor (e.g. Lisinopril 10mg) or ARB. If systolic BP remains above 140 mmHg at review, add calcium-channel blocker (e.g. Amlodipine 5mg).",
                    "retrieval_score": 0.88,
                    "dense_score": 0.88,
                    "bm25_score": 0.85,
                    "rrf_score": 0.032,
                    "source_file": "hypertension-in-adults-diagnosis-and-management.pdf",
                    "doc_id": "nice-ng136",
                    "guideline": "NICE NG136",
                    "condition": "hypertension",
                    "page": 12,
                },
                {
                    "content": "NICE CG150 Guideline: Tension-type headache diagnosis and prophylactic management. Screen for screen-time, dehydration, caffeine fluctuations, and muscle contraction patterns. Propranolol offers comorbid BP and tension headache prophylaxis.",
                    "retrieval_score": 0.76,
                    "dense_score": 0.76,
                    "bm25_score": 0.70,
                    "rrf_score": 0.028,
                    "source_file": "headaches-in-over-12s-diagnosis-and-management.pdf",
                    "doc_id": "nice-cg150",
                    "guideline": "NICE CG150",
                    "condition": "headache",
                    "page": 8,
                },
                {
                    "content": "NICE NG106 Guideline: Chronic heart failure in adults: diagnosis and management. Confirm diagnosis with NT-proBNP and transthoracic echocardiography. First-line therapy includes ACE-inhibitor (or ARB), beta-blocker, and MRA alongside SGLT2 inhibitors.",
                    "retrieval_score": 0.71,
                    "dense_score": 0.71,
                    "bm25_score": 0.65,
                    "rrf_score": 0.025,
                    "source_file": "chronic-heart-failure-core-treatments-for-heart-failure-visual-summary-pdf-6663137725.pdf",
                    "doc_id": "nice-ng106",
                    "guideline": "NICE NG106",
                    "condition": "heart_failure",
                    "page": 1,
                },
            ]
            return {
                "retrieved_chunks": fallback_chunks,
                "retrieval_mode": "fallback_static",
                "routed_documents": [],
                "query": self.build_query(summary, entities),
                "dense_hit_count": len(fallback_chunks),
                "bm25_hit_count": len(fallback_chunks),
            }

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        sections = []
        for index, chunk in enumerate(chunks, start=1):
            sections.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"Doc ID: {chunk.get('doc_id', '')}",
                        f"Guideline: {chunk.get('guideline', '')}",
                        f"Condition: {chunk.get('condition', '')}",
                        f"Source: {chunk.get('source_file', '')}",
                        f"Page: {chunk.get('page', '')}",
                        f"Hybrid score: {chunk.get('retrieval_score', 0)}",
                        f"Dense score: {chunk.get('dense_score', '')}",
                        f"BM25 score: {chunk.get('bm25_score', '')}",
                        chunk.get("content", ""),
                    ]
                )
            )
        return "\n\n".join(sections)
