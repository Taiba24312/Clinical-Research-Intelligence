import json
import logging
from pathlib import Path

from src.ingestion.clinicaltrials_api import ClinicalTrialsAPI


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DOMAIN_QUERIES = {
    "oncology": "cancer",
    "neurology": "neurological",
    "cardiovascular": "cardiovascular",
    "infectious_disease": "infectious disease",
    "metabolic_endocrine": "diabetes",
}

TARGET_PER_DOMAIN = 1000

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "clinical_trials_raw.json"


# ---------------------------------------------------------
# Helper: extract NCT ID
# ---------------------------------------------------------

def get_nct_id(study: dict) -> str:
    """Extract NCT ID from a raw ClinicalTrials.gov record."""

    return (
        study["protocolSection"]
        ["identificationModule"]
        ["nctId"]
    )


# ---------------------------------------------------------
# Main corpus-building function
# ---------------------------------------------------------

def build_corpus():

    api = ClinicalTrialsAPI(
        timeout=30,
        max_retries=3,
    )

    # Global dictionary:
    # NCT ID -> study + source domains
    unique_trials = {}

    domain_statistics = {}

    # -----------------------------------------------------
    # Fetch each domain
    # -----------------------------------------------------

    for domain, condition in DOMAIN_QUERIES.items():

        logger.info("=" * 60)
        logger.info(
            "Processing domain: %s | Query: %s",
            domain,
            condition,
        )

        studies = api.fetch_studies(
            condition=condition,
            page_size=100,
            max_studies=TARGET_PER_DOMAIN,
        )

        domain_statistics[domain] = {
            "requested": TARGET_PER_DOMAIN,
            "fetched": len(studies),
            "new_unique": 0,
            "duplicates": 0,
        }

        # -------------------------------------------------
        # Deduplicate by NCT ID
        # -------------------------------------------------

        for study in studies:

            nct_id = get_nct_id(study)

            if nct_id not in unique_trials:

                unique_trials[nct_id] = {
                    "study": study,
                    "source_domains": [domain],
                }

                domain_statistics[domain]["new_unique"] += 1

            else:

                domain_statistics[domain]["duplicates"] += 1

                if (
                    domain
                    not in unique_trials[nct_id]["source_domains"]
                ):
                    unique_trials[nct_id]["source_domains"].append(
                        domain
                    )

        logger.info(
            "%s: fetched=%s, new_unique=%s, duplicates=%s",
            domain,
            domain_statistics[domain]["fetched"],
            domain_statistics[domain]["new_unique"],
            domain_statistics[domain]["duplicates"],
        )

    # -----------------------------------------------------
    # Prepare output
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_records = []

    for nct_id, record in unique_trials.items():

        output_records.append(
            {
                "nct_id": nct_id,
                "source_domains": record["source_domains"],
                "study": record["study"],
            }
        )

    # -----------------------------------------------------
    # Save raw corpus
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total_requested = (
        TARGET_PER_DOMAIN
        * len(DOMAIN_QUERIES)
    )

    total_unique = len(output_records)

    duplicates_removed = (
        total_requested - total_unique
    )

    logger.info("=" * 60)
    logger.info("CORPUS BUILD COMPLETE")
    logger.info("=" * 60)

    logger.info(
        "Total records requested: %s",
        total_requested,
    )

    logger.info(
        "Unique trials: %s",
        total_unique,
    )

    logger.info(
        "Duplicate records removed: %s",
        duplicates_removed,
    )

    logger.info(
        "Output file: %s",
        OUTPUT_FILE,
    )

    print("\nDomain statistics:")

    for domain, stats in domain_statistics.items():

        print(
            f"{domain}: "
            f"fetched={stats['fetched']}, "
            f"new_unique={stats['new_unique']}, "
            f"duplicates={stats['duplicates']}"
        )

    print("\nFinal unique trials:", total_unique)
    print("Saved to:", OUTPUT_FILE)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    build_corpus()