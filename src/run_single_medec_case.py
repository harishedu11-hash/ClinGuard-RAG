import pandas as pd

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

case_number = 4

row = df.iloc[case_number]

print("\n" + "=" * 80)
print(f"CASE {case_number}")
print("=" * 80)

print(row["Text"])

print("\nGROUND TRUTH:")

print(
    "ERROR"
    if row["Error Flag"] == 1
    else "NO_ERROR"
)
