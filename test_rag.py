from rag.loader import load_repository
from rag.chunker import create_chunks
from rag.embeddings import EmbeddingModel
from rag.hybrid_retriever import HybridRetriever
from rag.generator import QwenGenerator


REPO_PATH = "data/urban-heat-repo"


# --------------------------------
# 1. Load repository
# --------------------------------

print("Loading repository...")

documents = load_repository(REPO_PATH)

print(f"Documents: {len(documents)}")


# --------------------------------
# 2. Create chunks
# --------------------------------

print("Creating chunks...")

chunks = create_chunks(documents)

print(f"Chunks: {len(chunks)}")


# --------------------------------
# 3. Create embeddings
# --------------------------------

print("Loading embedding model...")

embedding_model = EmbeddingModel()

texts = [chunk["content"] for chunk in chunks]

print("Creating embeddings...")

embeddings = embedding_model.encode(texts)

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------
# 4. Create hybrid retriever
# --------------------------------

retriever = HybridRetriever(
    embeddings,
    chunks
)


# --------------------------------
# 5. Load Qwen
# --------------------------------

print("Connecting to Qwen3...")

generator = QwenGenerator(
    model="qwen3:8b"
)


# --------------------------------
# 6. Ask question
# --------------------------------

query = input(
    "\nAsk a question about the repository: "
)


# --------------------------------
# 7. Embed query
# --------------------------------

query_embedding = embedding_model.encode(
    [query]
)[0]


# --------------------------------
# 8. Retrieve
# --------------------------------

print("\nRetrieving relevant information...")

results = retriever.search(
    query,
    query_embedding,
    top_k=5
)


# --------------------------------
# 9. Generate
# --------------------------------

print("Generating answer with Qwen3...")

answer = generator.generate(
    query,
    results
)


# --------------------------------
# 10. Display answer
# --------------------------------

print("\n")
print("=" * 70)
print("GENERATED ANSWER")
print("=" * 70)

print(answer)


# --------------------------------
# 11. Display sources
# --------------------------------

print("\n")
print("=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)

for i, result in enumerate(results, start=1):

    print(
        f"{i}. "
        f"{result['file_name']} "
        f"→ {result.get('section')}"
    )