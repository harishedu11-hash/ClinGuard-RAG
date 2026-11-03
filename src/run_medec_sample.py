import pandas as pd

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

sample_df = df.head(20)

print(sample_df["Error Flag"].value_counts())

for i, row in sample_df.iterrows():

    print("\n" + "=" * 80)

    print(f"CASE {i}")

    print("=" * 80)

    print(row["Text"][:500])

    print("\nGROUND TRUTH:")

    print(
        "ERROR"
        if row["Error Flag"] == 1
        else "NO_ERROR"
    )
