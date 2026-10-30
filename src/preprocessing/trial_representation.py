import pandas as pd


SEARCHABLE_FIELDS = [
    "title",
    "conditions",
    "interventions",
    "summary",
    "eligibility",
]


def build_trial_document(row: pd.Series) -> str:
    """Build one searchable document from a trial row."""

    sections = []

    sections.append(
        f"TITLE: {row['title']}"
    )

    sections.append(
        f"CONDITIONS: {row['conditions']}"
    )

    sections.append(
        f"INTERVENTIONS: {row['interventions']}"
    )

    sections.append(
        f"SUMMARY: {row['summary']}"
    )

    sections.append(
        f"ELIGIBILITY: {row['eligibility']}"
    )

    return "\n\n".join(sections)


def add_trial_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Add searchable document representation to dataframe."""

    df = df.copy()

    df["document"] = df.apply(
        build_trial_document,
        axis=1,
    )

    return df


if __name__ == "__main__":

    df = pd.read_parquet(
        "data/processed/trials.parquet"
    )

    df = add_trial_documents(df)

    print("Dataset shape:", df.shape)

    print("\nExample trial document:")
    print("=" * 70)
    print(df.loc[0, "document"])