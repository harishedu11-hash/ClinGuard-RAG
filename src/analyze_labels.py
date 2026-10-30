import pandas as pd

df = pd.read_csv(
    "data/raw/medec/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
)

print(df["Error Flag"].value_counts())
