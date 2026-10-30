import pandas as pd

df = pd.read_csv(
    "data/raw/medec/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
)

error_df = df[df["Error Flag"] == 1]

for i in range(5):
    row = error_df.iloc[i]

    print("\n" + "=" * 80)
    print(f"ERROR EXAMPLE {i+1}")
    print("=" * 80)

    print("\nERROR TYPE:")
    print(row["Error Type"])

    print("\nERROR SENTENCE:")
    print(row["Error Sentence"])

    print("\nCORRECTED SENTENCE:")
    print(row["Corrected Sentence"])
