import pandas as pd
import ollama

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

sample_df = df.head(10)

correct = 0

for idx, row in sample_df.iterrows():

    prompt = f"""
You are a clinical note auditor.

Example 1

Clinical Note:
A 5-year-old girl is brought to the physician because she avoids eye contact, repeats behaviors, and has difficulty interacting with peers. Patient was diagnosed with autism spectrum disorder.

Answer:
NO_ERROR


Example 2

Clinical Note:
A patient with clear symptoms and laboratory findings consistent with bacterial pneumonia is documented as having a viral infection.

Answer:
ERROR


Task

Determine whether the following clinical note contains a clear medical error.

Only return one word:

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
