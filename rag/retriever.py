import numpy as np


class Retriever:

    def __init__(self, embeddings, chunks):
        self.embeddings = embeddings
        self.chunks = chunks

    def search(self, query_embedding, top_k=5):

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "score": float(scores[index]),
                "file_path": self.chunks[index]["file_path"],
                "file_name": self.chunks[index]["file_name"],
                "chunk_id": self.chunks[index]["chunk_id"],
                "content": self.chunks[index]["content"]
            })

        return results