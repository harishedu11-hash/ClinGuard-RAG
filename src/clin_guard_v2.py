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

for chunk in text.split(
    "--------------------------------------------------------"
):
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
# AGENT 1 - AUDITOR
# =====================================================

audit_prompt = f"""
You are a clinical auditor.

Determine whether the clinical note
contains a medical error.

Respond with EXACTLY one word.

ERROR

or

NO_ERROR

Do not explain.
Do not provide reasoning.
Do not provide any other text.

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
# AGENT 2 - RETRIEVAL
# =====================================================

query_embedding = embed_model.encode(
    [clinical_note],
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
# AGENT 3 - VERIFIER
# =====================================================

verify_prompt = f"""
You are a senior clinical reviewer.

Clinical Note:

{clinical_note}

Retrieved Evidence:

{evidence}

Instructions:

1. Review all retrieved evidence.

2. Determine which evidence is clinically
most relevant.

3. Ignore unrelated evidence.

4. Decide if the note is SUPPORTED or CONTRADICTED.

5. If contradicted, provide a corrected statement.

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

print("\n" + "=" * 80)
print("AGENT 3")
print("=" * 80)

print(
    verify_result["message"]["content"]
)
