import pandas as pd

DATA_FILE = "data/processed/trials.parquet"

# --------------------------------------------------
# 1. Load processed dataset
# --------------------------------------------------

df = pd.read_parquet(DATA_FILE)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Shape:", df.shape)
print("Total trials:", len(df))


# --------------------------------------------------
# 2. Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_count = df.isna().sum()

missing_percentage = (
    df.isna().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage,
})

print(missing_report)


# --------------------------------------------------
# 3. Empty strings
# --------------------------------------------------

print("\n" + "=" * 60)
print("EMPTY STRING COUNTS")
print("=" * 60)

text_columns = [
    "title",
    "conditions",
    "interventions",
    "summary",
    "eligibility",
]

for column in text_columns:
    empty_count = (
        df[column]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"{column}: {empty_count}")


# --------------------------------------------------
# 4. Duplicate NCT IDs
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE NCT IDs")
print("=" * 60)

duplicate_count = df["nct_id"].duplicated().sum()

print("Duplicate NCT IDs:", duplicate_count)


# --------------------------------------------------
# 5. Status distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("STATUS DISTRIBUTION")
print("=" * 60)

print(
    df["status"]
    .value_counts(dropna=False)
)


# --------------------------------------------------
# 6. Phase distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("PHASE DISTRIBUTION")
print("=" * 60)

print(
    df["phase"]
    .value_counts(dropna=False)
)


# --------------------------------------------------
# 7. Study type distribution
# --------------------------------------------------

print("\n" + "=" * 60)
print("STUDY TYPE DISTRIBUTION")
print("=" * 60)

print(
    df["study_type"]
    .value_counts(dropna=False)
)



# 8. Sex distribution

print("\n" + "=" * 60)
print("SEX DISTRIBUTION")
print("=" * 60)

print(
    df["sex"]
    .value_counts(dropna=False)
)
print(
    df.groupby("phase")["study_type"]
      .value_counts()
)
print(
    df[df["phase"] == "NA"]["study_type"].value_counts()
)

print(
    df[df["phase"] == ""]["study_type"].value_counts()
)