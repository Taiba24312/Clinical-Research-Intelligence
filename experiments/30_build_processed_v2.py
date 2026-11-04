"""
Build processed corpus v2 (same 4,695 trials of corpus v1).

trials_v2.parquet = the 13 v1 columns, unchanged, plus official_title,
intervention_other_names, condition_mesh, intervention_mesh, keywords.
Checks that the v1 columns are identical to data/processed/trials.parquet.

Outputs (never overwritten):
    data/processed/trials_v2.parquet
    data/processed/trials_v2_manifest.json

Run from the project root:
    PYTHONPATH=$PWD python experiments/30_build_processed_v2.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocessing.processor import V2_EXTRA_COLUMNS, build_processed_dataset_v2
from src.preprocessing.trial_representation import add_trial_documents


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/clinical_trials_raw.json"
V1 = ROOT / "data/processed/trials.parquet"
V2 = ROOT / "data/processed/trials_v2.parquet"
MANIFEST = ROOT / "data/processed/trials_v2_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite {MANIFEST}")
    v2 = build_processed_dataset_v2(RAW, V2)
    v1 = pd.read_parquet(V1)

    pd.testing.assert_frame_equal(v2[v1.columns.tolist()].astype(v1.dtypes.to_dict()), v1)

    filled = {c: int((v2[c].fillna("").str.strip() != "").sum()) for c in V2_EXTRA_COLUMNS}
    docs = add_trial_documents(v2, version="v2")["document"]
    old_docs = add_trial_documents(v1)["document"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": "corpus v1 (data/corpus_v1_manifest.json), same trials and order",
        "raw_json_sha256": sha256(RAW),
        "file": str(V2.relative_to(ROOT)),
        "sha256": sha256(V2),
        "rows": len(v2),
        "columns": v2.columns.tolist(),
        "v1_columns_identical_to": {"file": str(V1.relative_to(ROOT)), "sha256": sha256(V1)},
        "new_columns_nonempty_rows": filled,
        "document_v2_mean_chars": round(float(docs.str.len().mean()), 1),
        "document_v1_mean_chars": round(float(old_docs.str.len().mean()), 1),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "new_columns_nonempty_rows",
                                              "document_v1_mean_chars", "document_v2_mean_chars")}, indent=1))
    print(f"Saved {V2} and {MANIFEST}")


if __name__ == "__main__":
    main()
