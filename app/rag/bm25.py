import math
import re
from collections import Counter
from typing import Any, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Okapi:
    """Okapi BM25 over an in-memory corpus (no extra package)."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        df: Counter = Counter()
        for tokens in corpus_tokens:
            df.update(set(tokens))
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        if not query_tokens or not self.n_docs:
            return scores

        for index, tokens in enumerate(self.corpus_tokens):
            if not tokens:
                continue
            tf = Counter(tokens)
            denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len[index] / (self.avgdl or 1.0))
            score = 0.0
            for term in query_tokens:
                term_tf = tf.get(term, 0)
                if not term_tf:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * (term_tf * (self.k1 + 1)) / (term_tf + denom_norm)
            scores[index] = score
        return scores

    def top_n(self, query: str, n: int = 10) -> list[tuple[int, float]]:
        scores = self.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:n] if score > 0]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    RRF: score(d) = sum 1 / (k + rank_d) across ranked id lists.
    rank is 1-based.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def chunk_id(meta: dict[str, Any], content: str) -> str:
    source = str(meta.get("source_file", ""))
    page = str(meta.get("page", ""))
    doc_id = str(meta.get("doc_id", ""))
    snippet = (content or "").strip()[:80]
    return f"{doc_id}|{source}|{page}|{snippet}"
