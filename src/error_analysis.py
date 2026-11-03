import pandas as pd

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

sample_df = df.head(20)

error_count = 0
no_error_count = 0

for _, row in sample_df.iterrows():

    if row["Error Flag"] == 1:
        error_count += 1
    else:
        no_error_count += 1

print("\nValidation Sample")

print(f"\nERROR notes: {error_count}")
print(f"NO_ERROR notes: {no_error_count}")
