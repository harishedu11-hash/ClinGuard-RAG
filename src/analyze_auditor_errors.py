import pandas as pd
import ollama

df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

sample_df = df.head(10)

for idx, row in sample_df.iterrows():

    prompt = f"""
You are a clinical note auditor.

Example 1

Clinical Note:
A child avoids eye contact and shows repetitive behaviors. Patient was diagnosed with autism spectrum disorder.

Answer:
NO_ERROR

Example 2

Clinical Note:
A patient with bacterial pneumonia is documented as having a viral infection.

Answer:
ERROR


Determine whether the following clinical note contains a clear medical error.

Return ONLY:

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

    if prediction != ground_truth:

        print("\n" + "=" * 80)
        print("MODEL MISTAKE")
        print("=" * 80)

        print("\nPrediction:")
        print(prediction)

        print("\nExpected:")
        print(ground_truth)

        print("\nClinical Note:")
        print(row["Text"])
