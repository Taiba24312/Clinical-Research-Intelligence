import logging

from src.ingestion.clinicaltrials_api import ClinicalTrialsAPI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


api = ClinicalTrialsAPI()


domains = {
    "oncology": "cancer",
    "cardiovascular": "cardiovascular",
}


unique_trials = {}

domain_counts = {}


for domain, condition in domains.items():

    print(f"\nFetching {domain}...")

    studies = api.fetch_studies(
        condition=condition,
        page_size=50,
        max_studies=100,
    )

    domain_counts[domain] = len(studies)

    for study in studies:

        nct_id = (
            study["protocolSection"]
            ["identificationModule"]
            ["nctId"]
        )

        if nct_id not in unique_trials:

            unique_trials[nct_id] = {
                "study": study,
                "domains": [domain],
            }

        else:

            if domain not in unique_trials[nct_id]["domains"]:
                unique_trials[nct_id]["domains"].append(domain)


print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

print("\nTrials fetched per domain:")

for domain, count in domain_counts.items():
    print(f"{domain}: {count}")

print("\nTotal records fetched:")
print(sum(domain_counts.values()))

print("\nUnique trials after NCT-ID deduplication:")
print(len(unique_trials))


duplicate_count = (
    sum(domain_counts.values()) - len(unique_trials)
)

print("\nDuplicate records removed:")
print(duplicate_count)


print("\nExample unique trials:")

for nct_id, record in list(unique_trials.items())[:5]:

    print(
        nct_id,
        "→",
        record["domains"]
    )