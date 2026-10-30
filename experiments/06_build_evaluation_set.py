import pandas as pd


DATA_FILE = "data/processed/trials.parquet"


def show_trial(df, index):
    """Display important information about one trial."""

    row = df.iloc[index]

    print("\n" + "=" * 70)
    print(f"TRIAL INDEX: {index}")
    print("=" * 70)

    print("NCT ID:", row["nct_id"])
    print("Title:", row["title"])
    print("Conditions:", row["conditions"])
    print("Interventions:", row["interventions"])
    print("Phase:", row["phase"])
    print("Status:", row["status"])
    print("Study Type:", row["study_type"])
    print("Sex:", row["sex"])
    print("Min Age:", row["min_age"])
    print("Max Age:", row["max_age"])


if __name__ == "__main__":

    df = pd.read_parquet(DATA_FILE)

    print("Total trials:", len(df))

    # Show a few real trials
    for index in range(5):
        show_trial(df, index)