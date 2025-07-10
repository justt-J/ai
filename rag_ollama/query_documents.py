# query_documents.py

import json
import requests
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction

# ========== CONFIGURATION ==========
OLLAMA_URL = "http://140.31.104.139:11434/"
EMBED_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3.1:latest"
CHROMA_PATH = "storage"

# ========== EMBEDDING WRAPPER ==========
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model=EMBED_MODEL, base_url=OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def __call__(self, texts: Documents):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                headers={"Content-Type": "application/json"},
                json={"model": self.model, "prompt": text}
            )
            if response.status_code == 200:
                data = response.json()
                if "embedding" in data:
                    embeddings.append(data["embedding"])
                else:
                    raise RuntimeError("Missing 'embedding' field in Ollama response.")
            else:
                raise RuntimeError(f"Ollama embedding error: {response.status_code} - {response.text}")
        return embeddings

# ========== QUERY GENERATION ==========
def query_ollama(context, question):
    prompt = f"""
You are a highly knowledgeable assistant. Use only the provided context to answer the question below. 
If the answer is not explicitly present or cannot be inferred from the context, reply with: "I don't know based on the context."

Instructions:
- Carefully analyze the context and synthesize a clear, accurate, and complete answer.
- If the question is complex, break down your answer into logical sections.
- Do not mention the context or data source in your answer.
- Avoid speculation; only answer what can be supported by the context.

Context:
{context}

Question:
{question}

Answer:
"""
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2
    }
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        raise RuntimeError(f"Ollama generation error: {response.status_code} - {response.text}")

# ========== RAG QUERY ==========
def rag_query(question, n_results=5):
    embed_fn = OllamaEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="rag_docs", embedding_function=embed_fn)

    results = collection.query(query_texts=[question], n_results=n_results)
    context_chunks = results.get("documents", [[]])[0]

    if not context_chunks:
        print("⚠️ No relevant context found.")
        return

    context = "\n".join(context_chunks)
    answer = query_ollama(context, question)
    print("\n🔍 Answer:\n" + answer)

# ========== MAIN ==========
if __name__ == "__main__":
    print("💬 RAG Query Interface with Ollama + ChromaDB")
    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        try:
            rag_query(question)
        except Exception as e:
            print(f"❌ Error: {e}")
