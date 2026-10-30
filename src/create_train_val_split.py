import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/processed/error_detection_dataset.csv"

df = pd.read_csv(INPUT_FILE)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Error Flag"]
)

train_df.to_csv(
    "data/processed/error_detection_train.csv",
    index=False
)

val_df.to_csv(
    "data/processed/error_detection_val.csv",
    index=False
)

print("Train:", len(train_df))
print("Validation:", len(val_df))

print("\nTrain Labels:")
print(train_df["Error Flag"].value_counts())

print("\nValidation Labels:")
print(val_df["Error Flag"].value_counts())
