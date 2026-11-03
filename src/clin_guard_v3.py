import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# =====================================================
# LOAD KNOWLEDGE BASE
# =====================================================

KNOWLEDGE_FILE = "knowledge_base/msd/raw/clinical_core_v2.txt"

from pathlib import Path

CHUNK_DIR = "knowledge_base/msd/all_chunks"

chunks = []

for file in sorted(
    Path(CHUNK_DIR).glob("*.txt")
):
    with open(file, "r") as f:
        chunks.append(f.read())

print(
    f"\nLoaded {len(chunks)} chunks"
)

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

Consider:

- Diagnosis errors
- Treatment errors
- Drug interactions
- Medication errors
- Management errors

If any potentially harmful issue exists respond:

ERROR

Otherwise:

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
Extract the most clinically relevant concepts.

Include:

Diagnoses:
Symptoms:
Medications:

Return only the concepts.

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

retrieval_query = (
    concept_result["message"]["content"]
    .strip()
)

print("\n" + "=" * 80)
print("AGENT 1.5 MEDICATIONS")
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

Determine:

SUPPORTED

or

CONTRADICTED

Use only the most relevant evidence.

If contradicted, propose a correction.

Format:

Verdict:

Most Relevant Evidence:

Correction:

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

print("\n" + "=" * 80)
print("AGENT 3")
print("=" * 80)

print(
    verify_result["message"]["content"]
)
