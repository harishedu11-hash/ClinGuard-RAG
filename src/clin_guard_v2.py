import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# =====================================================
# LOAD KNOWLEDGE BASE
# =====================================================

KNOWLEDGE_FILE = "knowledge_base/msd/raw/clinical_core_v2.txt"

with open(KNOWLEDGE_FILE, "r") as f:
    text = f.read()

chunks = []

for chunk in text.split("--------------------------------------------------------"):

    chunk = chunk.strip()

    if len(chunk) > 20:
        chunks.append(chunk)

print(f"\nLoaded {len(chunks)} knowledge chunks.")

# =====================================================
# BUILD RETRIEVER
# =====================================================

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embed_model.encode(
    chunks,
    convert_to_numpy=True
)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(
    np.array(
        embeddings,
        dtype=np.float32
    )
)

# =====================================================
# INPUT NOTE
# =====================================================

clinical_note = input(
    "\nEnter Clinical Note:\n\n"
)

# =====================================================
# AGENT 1
# =====================================================

audit_prompt = f"""
You are an expert clinical safety reviewer.

Consider whether the note contains:

- Diagnosis errors
- Treatment errors
- Pharmacotherapy errors
- Drug interaction errors
- Management errors

Be conservative.

If a potentially harmful clinical inconsistency,
drug interaction,
contraindication,
or incorrect management decision exists,
return:

ERROR

Otherwise return:

NO_ERROR

Return EXACTLY one word.

Clinical Note:

{clinical_note}
"""

audit_result = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": audit_prompt
        }
    ]
)

agent1_output = (
    audit_result["message"]["content"]
    .strip()
)

if "NO_ERROR" in agent1_output:
    agent1_output = "NO_ERROR"
elif "ERROR" in agent1_output:
    agent1_output = "ERROR"

print("\n" + "=" * 80)
print("AGENT 1")
print("=" * 80)
print(agent1_output)

# =====================================================
# AGENT 1.5
# =====================================================

concept_prompt = f"""
You are a clinical concept extractor.

Extract concepts in the following format.

Symptoms:
...

Medications:
...

Diagnoses:
...

Organisms:
...

Procedures:
...

Return only extracted concepts.

Clinical Note:

{clinical_note}
"""

concept_result = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": concept_prompt
        }
    ]
)

concept_text = (
    concept_result["message"]["content"]
    .strip()
)

retrieval_query = concept_text

# prioritize medication retrieval

if "Medications:" in concept_text:

    medication_section = (
        concept_text.split("Medications:")[1]
    )

    stop_markers = [
        "Diagnoses:",
        "Organisms:",
        "Procedures:"
    ]

    for marker in stop_markers:

        if marker in medication_section:
            medication_section = medication_section.split(marker)[0]

    medication_section = medication_section.strip()

    if medication_section:
        retrieval_query = medication_section

print("\n" + "=" * 80)
print("AGENT 1.5 CONCEPTS")
print("=" * 80)
print(retrieval_query)

# =====================================================
# AGENT 2
# =====================================================

query_embedding = embed_model.encode(
    [retrieval_query],
    convert_to_numpy=True
)

distances, indices = index.search(
    np.array(
        query_embedding,
        dtype=np.float32
    ),
    k=5
)

evidence_chunks = []

for idx in indices[0]:
    evidence_chunks.append(
        chunks[idx]
    )

evidence = "\n\n".join(
    evidence_chunks
)

print("\n" + "=" * 80)
print("AGENT 2 EVIDENCE")
print("=" * 80)

for rank, idx in enumerate(indices[0]):

    print(f"\nRANK {rank+1}")
    print("-" * 40)

    print(chunks[idx])

    print(
        f"\nDISTANCE: {distances[0][rank]:.4f}\n"
    )

# =====================================================
# AGENT 3
# =====================================================

verify_prompt = f"""
You are a senior clinical reviewer.

Clinical Note:

{clinical_note}

Retrieved Evidence:

{evidence}

Instructions:

1. Determine the single most relevant evidence.

2. Ignore unrelated evidence.

3. Decide whether the note is:

SUPPORTED

or

CONTRADICTED

4. If contradicted, provide a corrected statement.

Format:

Verdict:
SUPPORTED or CONTRADICTED

Most Relevant Evidence:

Correction:
(only if contradicted)

Explanation:
"""

verify_result = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": verify_prompt
        }
    ]
)
