import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Explain Retrieval-Augmented Generation in three simple sentences."
        }
    ]
)

print(response["message"]["content"])