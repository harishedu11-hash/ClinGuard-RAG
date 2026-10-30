import pandas as pd

DATASET = "data/raw/medec/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"

df = pd.read_csv(DATASET)

print("\nMEDEC Dataset Summary")
print("-" * 40)

print(f"Total Notes: {len(df)}")
print(f"Notes With Errors: {(df['Error Flag'] == 1).sum()}")
print(f"Notes Without Errors: {(df['Error Flag'] == 0).sum()}")

print("\nError Types:")
print(df["Error Type"].value_counts(dropna=False))
