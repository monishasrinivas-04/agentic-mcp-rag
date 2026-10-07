from rag.loader import load_repository
from rag.chunker import create_chunks
from rag.embeddings import EmbeddingModel
from rag.hybrid_retriever import HybridRetriever


REPO_PATH = "data/urban-heat-repo"


# -----------------------------
# LOAD REPOSITORY
# -----------------------------

documents = load_repository(REPO_PATH)

print(f"Number of documents: {len(documents)}")


# -----------------------------
# CREATE CHUNKS
# -----------------------------

chunks = create_chunks(documents)

print(f"Number of chunks: {len(chunks)}")


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

print("\nLoading embedding model...")

embedding_model = EmbeddingModel()

print("Embedding model loaded.")


# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------

texts = [
    chunk["content"]
    for chunk in chunks
]

print("\nCreating embeddings...")

embeddings = embedding_model.encode(texts)

print("Embeddings created.")

print(
    "Embedding shape:",
    embeddings.shape
)


# -----------------------------
# CREATE HYBRID RETRIEVER
# -----------------------------

retriever = HybridRetriever(
    embeddings,
    chunks
)


# -----------------------------
# QUERY
# -----------------------------

query = input(
    "\nAsk a question about the repository: "
)

query_embedding = embedding_model.encode(
    [query]
)[0]


# -----------------------------
# RETRIEVE
# -----------------------------

results = retriever.search(
    query,
    query_embedding,
    top_k=5
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\n")
print("=" * 60)
print("HYBRID RETRIEVAL RESULTS")
print("=" * 60)


for i, result in enumerate(
    results,
    start=1
):

    print(f"\nResult {i}")

    print(
        "File:",
        result["file_name"]
    )

    print(
        "Chunk ID:",
        result["chunk_id"]
    )
    print("Section:", result.get("section"))

    print(
        "Hybrid Score:",
        round(result["score"], 4)
    )

    print(
        "Dense Score:",
        round(result["dense_score"], 4)
    )

    print(
        "BM25 Score:",
        round(result["lexical_score"], 4)
    )

    print("\nContent:")

    print(
        result["content"][:1000]
    )

    print("-" * 60)