from rag.loader import load_repository
from rag.chunker import create_chunks


REPO_PATH = "data/urban-heat-repo"

documents = load_repository(REPO_PATH)

chunks = create_chunks(documents)

print("\n")
print("=" * 60)
print("CHUNKING TEST")
print("=" * 60)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):

    print("\n" + "-" * 60)

    print("Chunk:", i)
    print("File:", chunk["file_name"])
    print("Section:", chunk["section"])

    print("\nContent preview:")
    print(chunk["content"][:200])