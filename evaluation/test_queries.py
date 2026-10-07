from rag.loader import load_repository
from rag.chunker import create_chunks
from rag.embeddings import EmbeddingModel
from rag.hybrid_retriever import HybridRetriever

REPO_PATH = "data/urban-heat-repo"

documents = load_repository(REPO_PATH)
chunks = create_chunks(documents)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

print("\nLoading embedding model...")
embedding_model = EmbeddingModel()
print("Embedding model loaded.")

texts = [chunk["content"] for chunk in chunks]

print("\nCreating embeddings...")
embeddings = embedding_model.encode(texts)
print("Embeddings created.")

retriever = HybridRetriever(
    embeddings,
    chunks
)


test_queries = [
    {
        "query": "How is the Land Surface Temperature data loaded?",
        "expected": "LOAD LST TIFF FILE"
    },
    {
        "query": "How is rooftop temperature calculated?",
        "expected": "TEMPERATURE ANALYSIS"
    },
    {
        "query": "How is the temperature heatmap generated?",
        "expected": "LST HEATMAP"
    },
    {
        "query": "What model is used for rooftop detection?",
        "expected": "LOAD YOLO MODEL"
    },
    {
        "query": "How are rooftops detected?",
        "expected": "YOLO PREDICTION"
    },
    {
        "query": "How are temperature statistics displayed?",
        "expected": "DASHBOARD METRICS"
    },
    {
        "query": "How is the roof temperature distribution visualized?",
        "expected": "TEMPERATURE HISTOGRAM"
    },
    {
        "query": "How are hot, medium and cool roofs categorized?",
        "expected": "PIE CHART"
    },
    {
        "query": "How are individual roof temperatures displayed?",
        "expected": "DATA TABLE"
    },
    {
        "query": "How is the satellite image uploaded?",
        "expected": "IMAGE UPLOAD"
    }
]


for test in test_queries:

    query = test["query"]
    expected = test["expected"]

    query_embedding = embedding_model.encode([query])[0]

    dense_scores = embeddings @ query_embedding
    dense_ranking = dense_scores.argsort()[::-1]

    lexical_scores = retriever.bm25.get_scores(
        query.lower().split()
    )
    lexical_ranking = lexical_scores.argsort()[::-1]

    hybrid_results = retriever.search(
        query,
        query_embedding,
        top_k=5
    )

    print("\n")
    print("=" * 70)
    print("QUERY:")
    print(query)

    print("EXPECTED:")
    print(expected)

    # --------------------------------------------------
    # DENSE RETRIEVAL
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("DENSE RETRIEVAL - TOP 5")
    print("-" * 70)

    for rank, index in enumerate(dense_ranking[:5], start=1):

        chunk = chunks[index]

        print(
            f"{rank}. "
            f"{chunk['file_name']} | "
            f"{chunk.get('section')} | "
            f"Score: {dense_scores[index]:.4f}"
        )

    # --------------------------------------------------
    # BM25 RETRIEVAL
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("BM25 RETRIEVAL - TOP 5")
    print("-" * 70)

    for rank, index in enumerate(lexical_ranking[:5], start=1):

        chunk = chunks[index]

        print(
            f"{rank}. "
            f"{chunk['file_name']} | "
            f"{chunk.get('section')} | "
            f"Score: {lexical_scores[index]:.4f}"
        )

    # --------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("HYBRID RETRIEVAL - TOP 5")
    print("-" * 70)

    for rank, result in enumerate(hybrid_results, start=1):

        print(
            f"{rank}. "
            f"{result['file_name']} | "
            f"{result.get('section')} | "
            f"Hybrid: {result['score']:.4f} | "
            f"Dense: {result['dense_score']:.4f} | "
            f"BM25: {result['lexical_score']:.4f}"
        )