import pandas as pd

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

for i in range(3):
    row = df.iloc[i]

    print("\n" + "=" * 80)
    print(f"EXAMPLE {i+1}")
    print("=" * 80)

    print("\nCLINICAL NOTE:\n")
    print(row["Text"])

    print("\nGROUND TRUTH:")
    print(row["Error Flag"])
