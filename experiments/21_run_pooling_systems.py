"""
M1 Step 4a - run all pooling systems on query set v1 and save run files.

Pipeline (same as Exp 15/16/17, existing modules unchanged):
    parse_query -> filter -> BM25 (corpus-wide IDF, rank eligible) and
    dense bge-small (eligible) -> RRF top-50 -> cross-encoder rerank top-50

Systems (each saved as one run file, Step 2 run format):
    bm25_ws       BM25Retriever (whitespace tokenizer)        top-100
    bm25_punct    BM25Okapi, punctuation-splitting tokenizer  top-100  [pooling-only]
    dense         DenseEmbedder bge-small-en-v1.5             top-100
    rrf           HybridRetriever(fusion_k=60), bm25_ws+dense  top-50
    rrf_ce_raw    bge-reranker-base on the raw query text      top-50
    rrf_ce_semq   bge-reranker-base on parser semantic_query   top-50  [pooling-only]

Filter modes:
    parser_filters  the real current system (QueryParser + MetadataFilter)
    gold_filters    the query spec's intended structured constraints, applied
                    with the evaluation constraint checker. POOLING-ONLY: it
                    stands in for a future fixed parser so that its results
                    are judged. Never report it as a baseline.

Real baseline runs = {bm25_ws, dense, rrf, rrf_ce_raw} x parser_filters.

Run from the project root:
    PYTHONPATH=$PWD python experiments/21_run_pooling_systems.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.evaluation.judgments import load_query_specs
from src.evaluation.retrieval_metrics import constraint_violations
from src.preprocessing.trial_representation import add_trial_documents
from src.query_understanding.parser import parse_query
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import DenseEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata_filter import MetadataFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "data/processed/trials.parquet"
SPECS_PATH = PROJECT_ROOT / "data/evaluation/query_specs_v1.json"
RUN_DIR = PROJECT_ROOT / "data/runs/v1"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RETRIEVAL_TOP_K = 100      # BM25 / dense candidates (fusion input), as in Exp 15/16
FUSION_TOP_K = 50          # RRF output, reranked by the cross-encoder, as in Exp 16/17
FUSION_K = 60
BM25_K1, BM25_B = 1.5, 0.75
RERANK_BATCH, RERANK_MAX_LENGTH = 16, 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SYSTEM_ROLES = {
    "bm25_ws": "baseline",
    "bm25_punct": "pooling_only",
    "dense": "baseline",
    "rrf": "baseline",
    "rrf_ce_raw": "baseline",
    "rrf_ce_semq": "pooling_only",
}


def punct_tokenize(text: str) -> list[str]:
    """Pooling-only BM25 variant: lowercase, split on every non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hf_revision(model_name: str) -> str | None:
    ref = (Path.home() / ".cache/huggingface/hub"
           / f"models--{model_name.replace('/', '--')}" / "refs/main")
    return ref.read_text().strip() if ref.exists() else None


def top_k_positions(scores: np.ndarray, eligible: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Stable descending order of eligible positions (ties broken by position)."""
    local = scores[eligible]
    order = np.argsort(-local, kind="stable")[: min(k, eligible.size)]
    return [(int(eligible[i]), float(local[i])) for i in order]


def rerank(model, query: str, fused: list[tuple[int, float]], documents: list[str]):
    """Cross-encoder rerank; ties broken by the original RRF rank (Exp 17)."""
    if not fused:
        return []
    scores = model.predict(
        [(query, documents[pos]) for pos, _ in fused],
        batch_size=RERANK_BATCH, show_progress_bar=False,
    )
    order = sorted(range(len(fused)), key=lambda i: (-float(scores[i]), i))
    return [(fused[i][0], float(scores[i])) for i in order]


def main() -> None:
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite runs in {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS_PATH)
    trials = add_trial_documents(pd.read_parquet(PARQUET_PATH)).reset_index(drop=True)
    documents = trials["document"].tolist()
    nct_ids = trials["nct_id"].tolist()
    records = trials.drop(columns=["document"]).to_dict("index")

    print(f"Corpus: {len(trials)} trials | queries: {len(specs)} | device: {DEVICE}")

    bm25_ws = BM25Retriever(documents, k1=BM25_K1, b=BM25_B)
    bm25_punct = BM25Okapi([punct_tokenize(d) for d in documents], k1=BM25_K1, b=BM25_B)
    embedder = DenseEmbedder(model_name=DENSE_MODEL, device=DEVICE)
    doc_embeddings = embedder.encode_documents(documents, batch_size=64)
    reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=RERANK_MAX_LENGTH)
    metadata_filter = MetadataFilter(trials)
    hybrid = HybridRetriever(fusion_k=FUSION_K)

    rows = {}
    query_log = {}

    for query_id, spec in specs.items():
        parsed = parse_query(spec.text)
        semantic_query = parsed["semantic_query"]

        eligible_by_mode = {
            "parser_filters": np.asarray(metadata_filter.apply(parsed["filters"]), dtype=np.int64),
            "gold_filters": np.asarray(
                [i for i in range(len(trials))
                 if not constraint_violations(records[i], spec.structured_constraints)],
                dtype=np.int64,
            ),
        }
        query_log[query_id] = {
            "text": spec.text,
            "semantic_query": semantic_query,
            "parser_filters": parsed["filters"],
            "gold_constraints": spec.structured_constraints,
            "eligible": {m: int(e.size) for m, e in eligible_by_mode.items()},
        }

        bm25_ws_scores = bm25_ws.score(semantic_query)
        bm25_punct_scores = bm25_punct.get_scores(punct_tokenize(semantic_query))
        dense_scores = doc_embeddings @ embedder.encode_query(semantic_query)

        for mode, eligible in eligible_by_mode.items():
            lists = {}
            lists["bm25_ws"] = top_k_positions(bm25_ws_scores, eligible, RETRIEVAL_TOP_K)
            lists["bm25_punct"] = top_k_positions(bm25_punct_scores, eligible, RETRIEVAL_TOP_K)
            lists["dense"] = top_k_positions(dense_scores, eligible, RETRIEVAL_TOP_K)
            lists["rrf"] = (
                hybrid.fuse(lists["bm25_ws"], lists["dense"], top_k=FUSION_TOP_K)
                if eligible.size else []
            )
            lists["rrf_ce_raw"] = rerank(reranker, spec.text, lists["rrf"], documents)
            lists["rrf_ce_semq"] = rerank(reranker, semantic_query, lists["rrf"], documents)

            for system, ranked in lists.items():
                key = f"{system}.{mode}"
                rows.setdefault(key, []).extend(
                    {"query_id": query_id, "system": key, "nct_id": nct_ids[pos],
                     "rank": rank, "score": score}
                    for rank, (pos, score) in enumerate(ranked, start=1)
                )

        print(f"{query_id}: eligible parser={query_log[query_id]['eligible']['parser_filters']:5d} "
              f"gold={query_log[query_id]['eligible']['gold_filters']:5d} | semq='{semantic_query}'")

    for key, items in sorted(rows.items()):
        pd.DataFrame(items, columns=["query_id", "system", "nct_id", "rank", "score"]).to_csv(
            RUN_DIR / f"{key}.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True).stdout.strip()
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiments/21_run_pooling_systems.py",
        "git_commit_before_run": commit,
        "corpus_parquet_sha256": sha256_of(PARQUET_PATH),
        "query_specs_sha256": sha256_of(SPECS_PATH),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "library_versions": {p: version(p) for p in
                             ["torch", "transformers", "sentence-transformers", "rank-bm25",
                              "numpy", "pandas"]},
        "document_representation": "src/preprocessing/trial_representation.py "
                                   "(TITLE, CONDITIONS, INTERVENTIONS, SUMMARY, ELIGIBILITY)",
        "systems": {
            "bm25_ws": {"module": "src/retrieval/bm25.py BM25Retriever", "tokenizer": "lower().split()",
                        "k1": BM25_K1, "b": BM25_B, "query": "semantic_query", "top_k": RETRIEVAL_TOP_K,
                        "idf": "corpus-wide; only eligible trials ranked"},
            "bm25_punct": {"module": "rank_bm25.BM25Okapi (script-local)",
                           "tokenizer": "re.findall('[a-z0-9]+', lower())", "k1": BM25_K1, "b": BM25_B,
                           "query": "semantic_query", "top_k": RETRIEVAL_TOP_K,
                           "idf": "corpus-wide; only eligible trials ranked"},
            "dense": {"module": "src/retrieval/embeddings.py DenseEmbedder", "model": DENSE_MODEL,
                      "model_revision": hf_revision(DENSE_MODEL), "normalized": True,
                      "similarity": "inner product (exact, numpy; FAISS not used)",
                      "query": "semantic_query", "top_k": RETRIEVAL_TOP_K},
            "rrf": {"module": "src/retrieval/hybrid.py HybridRetriever", "fusion_k": FUSION_K,
                    "inputs": ["bm25_ws", "dense"], "top_k": FUSION_TOP_K},
            "rrf_ce_raw": {"model": RERANKER_MODEL, "model_revision": hf_revision(RERANKER_MODEL),
                           "max_length": RERANK_MAX_LENGTH, "query": "raw query text",
                           "candidates": "rrf top-50", "tie_break": "rrf rank"},
            "rrf_ce_semq": {"model": RERANKER_MODEL, "model_revision": hf_revision(RERANKER_MODEL),
                            "max_length": RERANK_MAX_LENGTH, "query": "semantic_query",
                            "candidates": "rrf top-50", "tie_break": "rrf rank"},
        },
        "system_roles": SYSTEM_ROLES,
        "filter_modes": {
            "parser_filters": "QueryParser filters applied with MetadataFilter (real current system)",
            "gold_filters": "query spec structured_constraints applied with "
                            "src/evaluation/retrieval_metrics.constraint_violations (POOLING-ONLY)",
        },
        "real_baseline_runs": [f"{s}.parser_filters" for s, r in SYSTEM_ROLES.items() if r == "baseline"],
        "queries": query_log,
        "run_files": sorted(f"{key}.csv" for key in rows),
    }
    (RUN_DIR / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(rows)} run files and run_config.json to {RUN_DIR}")


if __name__ == "__main__":
    main()
