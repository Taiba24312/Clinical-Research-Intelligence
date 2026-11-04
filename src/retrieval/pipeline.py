"""
End-to-end trial search: parse -> metadata filter -> BM25 + dense -> RRF.

Same steps and parameters as the experiment runs (Exp 15/16/21):
BM25 scores the whole corpus (corpus-wide IDF) and ranks only trials that
pass the filters; dense retrieval uses normalized bge-small embeddings and
exact inner product; the top-100 of each are fused with RRF (k = 60).
The optional cross-encoder is off by default: on the dev set it did not
improve over RRF once unjudged documents were accounted for.

Used by the evaluation scripts, the RAG step and the Streamlit demo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.preprocessing.trial_representation import add_trial_documents
from src.query_understanding.parser import parse_query
from src.query_understanding.parser_v1_1 import parse_query_v1_1
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import DenseEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata_filter import MetadataFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/trials.parquet"
CORPUS_V2 = PROJECT_ROOT / "data/processed/trials_v2.parquet"
CORPORA = {"v1": DEFAULT_CORPUS, "v2": CORPUS_V2}
INDEX_DIR = PROJECT_ROOT / "data/indexes"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RETRIEVAL_TOP_K = 100
FUSION_TOP_K = 50
FUSION_K = 60

PARSERS = {"v1": parse_query, "v1.1": parse_query_v1_1}


def pick_device() -> str:
    """The CUDA device with the most free memory, else CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{int(np.argmax(free))}"


def trial_url(nct_id: str) -> str:
    return f"https://clinicaltrials.gov/study/{nct_id}"


@dataclass
class SearchResult:
    query: str
    parser_version: str
    semantic_query: str
    filters: dict[str, Any]
    eligible_trials: int
    trials: list[dict[str, Any]] = field(default_factory=list)


class TrialSearchEngine:

    def __init__(self, corpus_path: Path | None = None, device: str | None = None,
                 parser_version: str = "v1.1", doc_version: str = "v1",
                 tokenizer: str = "v1") -> None:
        if parser_version not in PARSERS:
            raise ValueError(f"parser_version must be one of {sorted(PARSERS)}")
        self.parser_version = parser_version
        self.doc_version = doc_version
        self.tokenizer = tokenizer
        self.device = device or pick_device()

        corpus_path = Path(corpus_path or CORPORA[doc_version])
        self.corpus_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        self.trials = add_trial_documents(pd.read_parquet(corpus_path),
                                          version=doc_version).reset_index(drop=True)
        documents = self.trials["document"].tolist()
        self.documents_sha256 = hashlib.sha256("\x00".join(documents).encode()).hexdigest()

        self.bm25 = BM25Retriever(documents, tokenizer=tokenizer)
        self.embedder = DenseEmbedder(model_name=DENSE_MODEL, device=self.device)
        self.embeddings = self._load_or_build_embeddings(documents)
        self.metadata_filter = MetadataFilter(self.trials)
        self.hybrid = HybridRetriever(fusion_k=FUSION_K)
        self._reranker = None

    def describe(self) -> dict[str, Any]:
        """Configuration of this engine, for run configs and the demo."""
        return {
            "parser_version": self.parser_version, "doc_version": self.doc_version,
            "bm25_tokenizer": self.tokenizer, "dense_model": DENSE_MODEL,
            "fusion": f"RRF k={FUSION_K}, top-{RETRIEVAL_TOP_K} per retriever",
            "corpus_sha256": self.corpus_sha256, "documents_sha256": self.documents_sha256,
            "device": self.device,
        }

    def _load_or_build_embeddings(self, documents: list[str]) -> np.ndarray:
        cache = INDEX_DIR / f"bge_small_docs_{self.documents_sha256[:12]}.npy"
        if cache.exists():
            embeddings = np.load(cache)
            if embeddings.shape == (len(documents), self.embedder.dimension):
                return embeddings
        embeddings = self.embedder.encode_documents(documents, batch_size=64)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, embeddings)
        return embeddings

    @staticmethod
    def _top_k(scores: np.ndarray, eligible: np.ndarray, k: int) -> list[tuple[int, float]]:
        local = scores[eligible]
        order = np.argsort(-local, kind="stable")[: min(k, eligible.size)]
        return [(int(eligible[i]), float(local[i])) for i in order]

    def parse(self, query: str) -> dict[str, Any]:
        return PARSERS[self.parser_version](query)

    def search(self, query: str, top_k: int = 10, rerank: bool = False) -> SearchResult:
        parsed = self.parse(query)
        semantic_query = parsed["semantic_query"]
        eligible = np.asarray(self.metadata_filter.apply(parsed["filters"]), dtype=np.int64)

        result = SearchResult(query=query, parser_version=self.parser_version,
                              semantic_query=semantic_query, filters=parsed["filters"],
                              eligible_trials=int(eligible.size))
        if eligible.size == 0:
            return result

        lexical = self._top_k(self.bm25.score(semantic_query), eligible, RETRIEVAL_TOP_K)
        dense = self._top_k(self.embeddings @ self.embedder.encode_query(semantic_query),
                            eligible, RETRIEVAL_TOP_K)
        fused = self.hybrid.fuse(lexical, dense, top_k=FUSION_TOP_K)
        if rerank:
            fused = self._rerank(query, fused)

        for rank, (position, score) in enumerate(fused[:top_k], start=1):
            row = self.trials.iloc[position]
            result.trials.append({
                "rank": rank,
                "nct_id": row["nct_id"],
                "score": float(score),
                "title": row["title"],
                "conditions": row["conditions"],
                "interventions": row["interventions"] or "(none listed)",
                "status": row["status"],
                "phase": row["phase"] or "NA",
                "study_type": row["study_type"],
                "sex": row["sex"],
                "min_age": row["min_age"],
                "max_age": row["max_age"],
                "summary": row["summary"],
                "url": trial_url(row["nct_id"]),
            })
        return result

    def _rerank(self, query: str, fused: list[tuple[int, float]]) -> list[tuple[int, float]]:
        from sentence_transformers import CrossEncoder
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL, device=self.device, max_length=512)
        documents = self.trials["document"]
        scores = self._reranker.predict([(query, documents.iloc[p]) for p, _ in fused],
                                        batch_size=16, show_progress_bar=False)
        order = sorted(range(len(fused)), key=lambda i: (-float(scores[i]), i))
        return [(fused[i][0], float(scores[i])) for i in order]
