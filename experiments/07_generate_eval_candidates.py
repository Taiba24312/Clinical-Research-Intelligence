"""Generate manual review pools for Q02-Q10, preserving existing judgments."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from src.preprocessing.trial_representation import add_trial_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data/processed/trials.parquet"
QUERY_FILE = PROJECT_ROOT / "data/evaluation/queries.json"
OUTPUT_FILE = PROJECT_ROOT / "data/evaluation/Q01_candidates.csv"

# AND between groups; OR between alternatives within each group.
CANCER = ("cancer", "carcinoma", "tumor", "tumour", "neoplasm")
BREAST = ("breast cancer", "breast carcinoma", "breast neoplasm")
KEYWORD_GROUPS = {
    "Q02": [BREAST],
    "Q03": [("alzheimer",)],
    "Q04": [("diabetes", "diabetic")],
    "Q05": [BREAST],
    "Q06": [CANCER, ("immunotherapy", "immunotherapeutic", "immune checkpoint",
                      "pembrolizumab", "nivolumab", "atezolizumab", "ipilimumab")],
    "Q07": [("cardiovascular", "coronary", "heart disease", "heart failure")],
    "Q08": [("metastatic", "metastasis", "metastases"), CANCER],
    "Q09": [("targeted therapy", "targeted therapies", "tyrosine kinase inhibitor"),
            ("solid tumor", "solid tumour", "solid neoplasm")],
    "Q10": [("infectious", "infection", "hiv", "tuberculosis", "hepatitis",
             "covid", "influenza", "malaria")],
}


def age_in_years(value):
    """Treat missing bounds as open; reject unrecognized age formats."""
    if pd.isna(value) or str(value).strip().upper() in {"", "NA", "N/A"}:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s+(Years?|Months?|Weeks?|Days?|Hours?|Minutes?)\s*", str(value), re.I)
    if not match:
        raise ValueError(f"Unsupported age: {value!r}")
    divisors = {"year": 1, "month": 12, "week": 52.1775,
                "day": 365.25, "hour": 8766, "minute": 525960}
    return float(match[1]) / divisors[match[2].lower().rstrip("s")]


def candidate_mask(df, query):
    mask = pd.Series(True, index=df.index)
    for field, values in query["filters"].items():
        if field == "min_age":
            # Adults: eligibility interval overlaps [18, infinity).
            minimum = df["min_age"].map(age_in_years)
            maximum = df["max_age"].map(age_in_years)
            mask &= maximum.isna() | maximum.ge(values)
            mask &= minimum.isna() | maximum.isna() | minimum.le(maximum)
        elif field == "phase":
            mask &= df[field].fillna("").map(
                lambda value: bool({item.strip() for item in value.split(";")} & set(values))
            )
        elif field == "sex":
            mask &= df[field].isin([*values, "ALL"])
        else:
            mask &= df[field].isin(values)
    for alternatives in KEYWORD_GROUPS[query["query_id"]]:
        group = pd.Series(False, index=df.index)
        for keyword in alternatives:
            group |= df["document"].str.contains(keyword, case=False, regex=False, na=False)
        mask &= group
    return mask


def main():
    with QUERY_FILE.open(encoding="utf-8") as handle:
        queries = json.load(handle)
    q01_hash = hashlib.sha256(OUTPUT_FILE.read_bytes()).digest()
    columns = pd.read_csv(OUTPUT_FILE, nrows=0).columns.tolist()
    df = add_trial_documents(pd.read_parquet(DATA_FILE))
    for query in queries:
        query_id = query["query_id"]
        if query_id == "Q01":
            continue
        candidates = df.loc[candidate_mask(df, query)].copy()
        candidates.insert(0, "query_id", query_id)
        candidates.insert(1, "query", query["query"])
        candidates["relevance"] = ""
        candidates["notes"] = ""
        candidates = candidates[columns]
        path = OUTPUT_FILE.with_name(f"{query_id}_candidates.csv")
        existing = pd.DataFrame(columns=columns)
        if path.exists():
            existing = pd.read_csv(path, keep_default_na=False, dtype=str)
            if existing.columns.tolist() != columns:
                raise ValueError(f"Unexpected columns: {path}")
            if not existing["query_id"].eq(query_id).all() or not existing["nct_id"].is_unique:
                raise ValueError(f"Invalid candidate rows: {path}")
        additions = candidates.loc[~candidates["nct_id"].isin(existing["nct_id"])]
        if not path.exists() or not additions.empty:
            combined = pd.concat([existing, additions], ignore_index=True)
            combined.to_csv(path, index=False, encoding="utf-8-sig")
        verified = pd.read_csv(path, keep_default_na=False, dtype=str)
        assert verified.columns.tolist() == columns
        assert verified["nct_id"].is_unique
        if not existing.empty:
            pd.testing.assert_frame_equal(verified.iloc[:len(existing)].reset_index(drop=True), existing)
        new_rows = verified.loc[verified["nct_id"].isin(additions["nct_id"])]
        assert new_rows[["relevance", "notes"]].eq("").all().all()
        print(f"{query_id}: {len(verified)} candidates | {', '.join(verified['nct_id']) or '(none)'}")
    assert hashlib.sha256(OUTPUT_FILE.read_bytes()).digest() == q01_hash
    print("Verified Q02-Q10 files, matching columns, blank new labels, and unchanged Q01.")


if __name__ == "__main__":
    main()
