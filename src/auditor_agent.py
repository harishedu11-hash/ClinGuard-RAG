import pandas as pd
import ollama

# Load validation dataset
df = pd.read_csv(
    "data/processed/error_detection_val.csv"
)

# Pick one example
row = df.iloc[1]

clinical_note = row["Text"]

prompt = f"""
You are a clinical note auditor.

Your task is to determine whether the clinical note contains a medical error.

Respond with ONLY one word:

ERROR

or

NO_ERROR

Clinical Note:

{clinical_note}
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

print("\nMODEL RESPONSE:")
print(response["message"]["content"])

print("\nGROUND TRUTH:")
print(row["Error Flag"])
