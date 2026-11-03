import pandas as pd
import ollama

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

sample_df = df.head(20)

correct = 0

for idx, row in sample_df.iterrows():

    note = row["Text"]

    prompt = f"""
You are a clinical auditor.

Respond with EXACTLY one word.

ERROR

or

NO_ERROR

Clinical Note:

{note}
"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    prediction = (
        response["message"]["content"]
        .strip()
        .upper()
    )

    if "NO_ERROR" in prediction:
        prediction = "NO_ERROR"
    elif "ERROR" in prediction:
        prediction = "ERROR"

    ground_truth = (
        "ERROR"
        if row["Error Flag"] == 1
        else "NO_ERROR"
    )

    print(
        f"{idx:02d} | Pred={prediction:10s} | True={ground_truth}"
    )

    if prediction == ground_truth:
        correct += 1

accuracy = correct / len(sample_df)

print("\n" + "=" * 50)
print(f"Accuracy: {accuracy:.2%}")
print("=" * 50)
