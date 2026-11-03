import ollama

clinical_note = input(
    "\nEnter Clinical Note:\n\n"
)

prompt = f"""
You are a clinical concept extractor.

Extract the most important clinical concepts.

Include:

- Symptoms
- Diagnoses
- Organisms
- Medications
- Procedures
- Clinical findings

Return ONLY a comma separated list.

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

print("\nCONCEPTS:\n")

print(
    response["message"]["content"]
)
