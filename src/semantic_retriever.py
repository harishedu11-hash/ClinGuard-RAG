from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

KNOWLEDGE_FILE = "knowledge_base/msd/raw/clinical_core_v2.txt"

# -----------------------------
# Load file
# -----------------------------
with open(KNOWLEDGE_FILE, "r") as f:
    text = f.read()

# -----------------------------
# Split into chunks
# -----------------------------
sections = []

raw_chunks = text.split("--------------------------------------------------------")

for chunk in raw_chunks:

    chunk = chunk.strip()

    if len(chunk) > 20:
        sections.append(chunk)

# -----------------------------
# Show chunks
# -----------------------------
print(f"\nLoaded {len(sections)} chunks\n")

# -----------------------------
# Embeddings
# -----------------------------
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
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

# -----------------------------
# Query
# -----------------------------
query = input("Enter query: ")

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

distances, indices = index.search(
    np.array(
        query_embedding,
        dtype=np.float32
    ),
    k=3
)

# -----------------------------
# Results
# -----------------------------
print("\n" + "=" * 80)
print("TOP MATCHES")
print("=" * 80)

for rank, idx in enumerate(indices[0]):

    print("\n")
    print(f"RANK {rank+1}")
    print("=" * 80)

    print(sections[idx])

    print("\nDISTANCE:")
    print(distances[0][rank])
