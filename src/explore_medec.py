import pandas as pd

DATASET = "data/raw/medec/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"

df = pd.read_csv(DATASET)

row = df[df["Error Flag"] == 1].iloc[0]

print("=" * 80)
print("CLINICAL NOTE")
print("=" * 80)
print(row["Text"])

print("\n")
print("=" * 80)
print("ERROR TYPE")
print("=" * 80)
print(row["Error Type"])

print("\n")
print("=" * 80)
print("ERROR SENTENCE")
print("=" * 80)
print(row["Error Sentence"])

print("\n")
print("=" * 80)
print("CORRECTED SENTENCE")
print("=" * 80)
print(row["Corrected Sentence"])
