KNOWLEDGE_FILE = "knowledge_base/msd/raw/clinical_knowledge.txt"

with open(KNOWLEDGE_FILE, "r") as f:
    knowledge = f.read()

query = input("Enter query: ")

query_words = query.lower().split()

print("\n" + "=" * 80)
print("RETRIEVED KNOWLEDGE")
print("=" * 80)

for line in knowledge.split("\n"):

    lower_line = line.lower()

    for word in query_words:

        if word in lower_line:
            print(line)
            break
