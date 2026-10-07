import ollama


class QwenGenerator:

    def __init__(self, model="qwen3:4b"):
        self.model = model

    def generate(self, query, retrieved_chunks):

        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):

            context_parts.append(
                f"""
SOURCE {i}
File: {chunk["file_name"]}
Section: {chunk.get("section")}

Content:
{chunk["content"]}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a repository question-answering assistant.

Answer the user's question using ONLY the retrieved
repository context provided below.

Rules:
1. Do not invent information.
2. Do not assume information that is not present.
3. Do not create filenames, functions, or implementation
   details that are not supported by the context.
4. If the context is insufficient, say:
   "I could not find enough information in the repository."
5. Give a clear technical explanation.
6. Mention the relevant file and section when possible.

User question:
{query}

Retrieved repository context:
{context}

Answer:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]