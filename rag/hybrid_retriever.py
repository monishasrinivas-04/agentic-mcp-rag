import numpy as np
from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(
        self,
        embeddings,
        chunks,
        rrf_k=60,
        dense_weight=0.4,
        lexical_weight=0.6
    ):
        self.embeddings = embeddings
        self.chunks = chunks
        self.rrf_k = rrf_k

        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

        tokenized_chunks = []

        for chunk in chunks:

            section = chunk.get("section")

            searchable_text = ""

            if section:
                searchable_text += section + " "

            searchable_text += chunk["content"]

            tokenized_chunks.append(
                searchable_text.lower().split()
            )

        self.bm25 = BM25Okapi(tokenized_chunks)


    def search(self, query, query_embedding, top_k=5):

        # -----------------------------
        # Dense retrieval
        # -----------------------------

        dense_scores = np.dot(
            self.embeddings,
            query_embedding
        )

        dense_ranking = np.argsort(
            dense_scores
        )[::-1]


        # -----------------------------
        # BM25 retrieval
        # -----------------------------

        lexical_scores = self.bm25.get_scores(
            query.lower().split()
        )

        lexical_ranking = np.argsort(
            lexical_scores
        )[::-1]


        # -----------------------------
        # Weighted RRF
        # -----------------------------

        rrf_scores = np.zeros(
            len(self.chunks)
        )

        for rank, index in enumerate(dense_ranking):

            rrf_scores[index] += (
                self.dense_weight
                / (self.rrf_k + rank + 1)
            )


        for rank, index in enumerate(lexical_ranking):

            rrf_scores[index] += (
                self.lexical_weight
                / (self.rrf_k + rank + 1)
            )


        # -----------------------------
        # Select top results
        # -----------------------------

        top_indices = np.argsort(
            rrf_scores
        )[::-1][:top_k]


        results = []

        for index in top_indices:

            results.append({

                "score": float(
                    rrf_scores[index]
                ),

                "dense_score": float(
                    dense_scores[index]
                ),

                "lexical_score": float(
                    lexical_scores[index]
                ),

                "file_path":
                    self.chunks[index]["file_path"],

                "file_name":
                    self.chunks[index]["file_name"],

                "chunk_id":
                    self.chunks[index]["chunk_id"],

                "section":
                    self.chunks[index].get("section"),

                "content":
                    self.chunks[index]["content"]
            })

        return results