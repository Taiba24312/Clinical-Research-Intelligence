import json
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


RAW_FILE = Path("data/raw/clinical_trials_raw.json")
OUTPUT_FILE = Path("data/processed/trials.parquet")


def load_raw_trials(file_path: Path) -> list:
    """Load raw ClinicalTrials.gov records."""

    with open(file_path, "r", encoding="utf-8") as f:
        trials = json.load(f)

    logger.info("Loaded %s raw trials.", len(trials))

    return trials


def extract_interventions(intervention_list: list) -> str:
    """Extract intervention names into a searchable string."""

    if not intervention_list:
        return ""

    names = []

    for intervention in intervention_list:
        name = intervention.get("name")

        if name:
            names.append(name.strip())

    return "; ".join(names)


def extract_trial(trial: dict) -> dict:
    """Extract relevant fields from one raw trial."""

    protocol = trial.get("study", {}).get("protocolSection", {})

    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    description = protocol.get("descriptionModule", {})
    conditions = protocol.get("conditionsModule", {})
    design = protocol.get("designModule", {})
    interventions_module = protocol.get(
        "armsInterventionsModule", {}
    )
    eligibility = protocol.get("eligibilityModule", {})

    last_update = status.get(
        "lastUpdatePostDateStruct", {}
    ).get("date")

    return {
        "nct_id": trial.get("nct_id"),

        "title": identification.get("briefTitle", ""),

        "conditions": "; ".join(
            conditions.get("conditions", [])
        ),

        "interventions": extract_interventions(
            interventions_module.get("interventions", [])
        ),

        "summary": description.get(
            "briefSummary", ""
        ),

        "eligibility": eligibility.get(
            "eligibilityCriteria", ""
        ),

        "min_age": eligibility.get(
            "minimumAge"
        ),

        "max_age": eligibility.get(
            "maximumAge"
        ),

        "sex": eligibility.get(
            "sex"
        ),

        "phase": "; ".join(
            design.get("phases", [])
        ),

        "status": status.get(
            "overallStatus"
        ),

        "study_type": design.get(
            "studyType"
        ),

        "last_update": last_update,
    }


def build_processed_dataset(
    raw_file: Path = RAW_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:

    trials = load_raw_trials(raw_file)

    processed_trials = []

    for trial in trials:
        processed_trials.append(
            extract_trial(trial)
        )

    df = pd.DataFrame(processed_trials)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_file,
        index=False,
    )

    logger.info(
        "Saved processed dataset to %s",
        output_file,
    )

    return df


# =============================================================
# Processed v2: the v1 columns unchanged + fields that carry brand names,
# synonyms and normalized terms (official title, intervention otherNames,
# condition/intervention MeSH terms, keywords).
# =============================================================

V2_EXTRA_COLUMNS = [
    "official_title",
    "intervention_other_names",
    "condition_mesh",
    "intervention_mesh",
    "keywords",
]


def extract_trial_v2(trial: dict) -> dict:
    """extract_trial() plus the v2 fields."""

    record = extract_trial(trial)
    study = trial.get("study", {})
    protocol = study.get("protocolSection", {})
    derived = study.get("derivedSection", {})
    interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])

    def mesh(module: str) -> str:
        return "; ".join(m["term"] for m in derived.get(module, {}).get("meshes", []))

    record.update({
        "official_title": protocol.get("identificationModule", {}).get("officialTitle", ""),
        "intervention_other_names": "; ".join(
            dict.fromkeys(name.strip() for item in interventions
                          for name in item.get("otherNames", []) if name.strip())),
        "condition_mesh": mesh("conditionBrowseModule"),
        "intervention_mesh": mesh("interventionBrowseModule"),
        "keywords": "; ".join(protocol.get("conditionsModule", {}).get("keywords", [])),
    })
    return record


def build_processed_dataset_v2(raw_file: Path, output_file: Path) -> pd.DataFrame:
    """Write processed v2; refuses to overwrite an existing file."""

    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite {output_file}")
    df = pd.DataFrame([extract_trial_v2(trial) for trial in load_raw_trials(raw_file)])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    return df


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    df = build_processed_dataset()

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())