import pandas as pd

val_df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

majority_class = 1

predictions = [majority_class] * len(val_df)

correct = (
    predictions == val_df["Error Flag"]
).sum()

accuracy = correct / len(val_df)

print(f"Accuracy: {accuracy:.4f}")
