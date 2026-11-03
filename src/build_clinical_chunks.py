from pathlib import Path
import re

SOURCE_FILE = "knowledge_base/msd/raw/clinical_expansion_v1.txt"
OUTPUT_DIR = "knowledge_base/msd/chunks"

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

with open(SOURCE_FILE, "r") as f:
    text = f.read()

# Split documents
sections = text.split("### DOCUMENT")

documents = []

for section in sections:

    section = section.strip()

    if not section:
        continue

    documents.append(section)

print(f"\nFound {len(documents)} documents\n")

for idx, document in enumerate(
    documents,
    start=1
):

    lines = [
        line.strip()
        for line in document.splitlines()
        if line.strip() and line.strip() != "<br>"
    ]

    if not lines:
        continue

    print("\nDOCUMENT:")
    print(lines[:5])

    title = lines[0]

    safe_name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        title
    ).strip("_")

    filename = (
        f"{idx:03d}_{safe_name}.txt"
    )

    filepath = (
        Path(OUTPUT_DIR)
        / filename
    )

    with open(filepath, "w") as out:
        out.write(document)

    print(f"Created: {filename}")

print("\nChunk build complete.")
