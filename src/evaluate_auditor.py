import pandas as pd
import ollama

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

sample_df = df.head(10)

correct = 0

for idx, row in sample_df.iterrows():

    prompt = f"""
You are reviewing a clinical note.

Your job is NOT to suggest improvements.

Your job is ONLY to determine whether the note contains a clear medical error.

If the note is medically reasonable, respond:

NO_ERROR

If the note contains a clear diagnosis, treatment, management, pharmacotherapy, or causal organism error, respond:

ERROR

Return EXACTLY one word.

ERROR

or

NO_ERROR

Clinical Note:

{row['Text']}
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

    prediction = response["message"]["content"].strip()

    ground_truth = (
        "ERROR"
        if row["Error Flag"] == 1
        else "NO_ERROR"
    )

    print(
        f"Prediction: {prediction} | Ground Truth: {ground_truth}"
    )

    if prediction == ground_truth:
        correct += 1

accuracy = correct / len(sample_df)

print(f"\nAccuracy: {accuracy:.2f}")
