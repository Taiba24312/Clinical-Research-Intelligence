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


def build_trial_document_v2(row: pd.Series) -> str:
    """
    v1 document plus official title, keywords, MeSH terms and intervention
    other names (brand names, code names). The short new fields come before
    the long summary/eligibility text so the dense model (512-token limit)
    still sees them.
    """

    def field(label: str, value) -> str | None:
        value = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
        return f"{label}: {value}" if value.strip() else None

    sections = [
        f"TITLE: {row['title']}",
        field("OFFICIAL TITLE", row["official_title"]),
        f"CONDITIONS: {row['conditions']}",
        field("CONDITION MESH TERMS", row["condition_mesh"]),
        field("KEYWORDS", row["keywords"]),
        f"INTERVENTIONS: {row['interventions']}",
        field("INTERVENTION OTHER NAMES", row["intervention_other_names"]),
        field("INTERVENTION MESH TERMS", row["intervention_mesh"]),
        f"SUMMARY: {row['summary']}",
        f"ELIGIBILITY: {row['eligibility']}",
    ]
    return "\n\n".join(s for s in sections if s)


DOCUMENT_BUILDERS = {"v1": build_trial_document, "v2": build_trial_document_v2}


def add_trial_documents(df: pd.DataFrame, version: str = "v1") -> pd.DataFrame:
    """Add searchable document representation to dataframe (v1 by default)."""

    df = df.copy()

    df["document"] = df.apply(
        DOCUMENT_BUILDERS[version],
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