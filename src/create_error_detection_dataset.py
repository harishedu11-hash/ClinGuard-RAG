import pandas as pd

INPUT_FILE = "data/raw/medec/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv"
OUTPUT_FILE = "data/processed/error_detection_dataset.csv"

df = pd.read_csv(INPUT_FILE)

agent1_df = df[["Text", "Error Flag"]]

agent1_df.to_csv(OUTPUT_FILE, index=False)

print("Dataset created.")
print(agent1_df.head())
print(f"\nRows: {len(agent1_df)}")
