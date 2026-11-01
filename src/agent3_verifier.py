import ollama

clinical_note = """
The patient has productive cough, fever and consolidation on chest imaging.
The causal organism was documented as Haemophilus influenzae.
"""

retrieved_evidence = """
COMMUNITY ACQUIRED PNEUMONIA

Symptoms:
- Fever
- Chills
- Productive cough
- Lung consolidation

Common Bacterial Cause:
- Streptococcus pneumoniae
"""

prompt = f"""
You are a clinical evidence verifier.

Clinical Note:

{clinical_note}

Retrieved Evidence:

{retrieved_evidence}

Tasks:

1. Decide whether the note is SUPPORTED or CONTRADICTED by the evidence.

2. If contradicted, suggest a corrected version of the incorrect statement.

3. Provide a brief explanation.

Format:

Verdict:
SUPPORTED or CONTRADICTED

Correction:

Explanation:
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

print(response["message"]["content"])
