import ollama
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# ------------------
# Load Knowledge Base
# ------------------

with open(
    "knowledge_base/msd/raw/clinical_knowledge.txt",
    "r"
) as f:

    knowledge = f.read()

sections = [
    section.strip()
    for section in knowledge.split("\n\n")
    if section.strip()
]

# ------------------
# Build Embeddings
# ------------------

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embed_model.encode(
    sections,
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

# ------------------
# Input Note
# ------------------

clinical_note = input(
    "\nEnter Clinical Note:\n\n"
)

# ------------------
# Agent 1
# ------------------

prompt = f"""
You are a clinical note auditor.

Determine whether the note
contains a medical error.

Respond ONLY:

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

prediction = (
    response["message"]["content"]
    .strip()
)

print("\n" + "=" * 80)
print("AGENT 1")
print("=" * 80)
print(prediction)

# ------------------
# Agent 2
# ------------------

query_embedding = embed_model.encode(
    [clinical_note],
    convert_to_numpy=True
)

distances, indices = index.search(
    np.array(
        query_embedding,
        dtype=np.float32
    ),
    k=3
)

print("\n" + "=" * 80)
print("AGENT 2 EVIDENCE")
print("=" * 80)

for idx in indices[0]:

    print("\n")
    print(sections[idx])
