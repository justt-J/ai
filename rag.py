# rag_pipeline.py

import requests
import json
import re
from neo4j import GraphDatabase

# ========== CONFIGURATION ==========
# Ollama API settings
OLLAMA_URL = "http://140.31.105.132:11434//api/generate"
OLLAMA_MODEL = "llama3.1:latest"

# Neo4j connection
NEO4J_URI = "neo4j+s://ac78a829.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "jPG_oE4ySe25ZO2X2STL7vgKG8CYNQoW1UmKQ365Vag"

# ========== NEO4J DRIVER ==========
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ========== ENTITY KEYWORD EXTRACTION ==========
def extract_keywords(text):
    words = re.findall(r'\w+', text)
    return [w for w in words if len(w) > 3]

# ========== CONTEXT RETRIEVAL FROM NEO4J ==========
def retrieve_context_from_graph(question):
    keywords = extract_keywords(question)
    results = []

    with driver.session() as session:
        for word in keywords:
            query = """
              MATCH (n)-[r]->(m)
            RETURN n.name, type(r), m.name
            LIMIT 20
            """
            result = session.run(query, word=word)
            for record in result:
                results.append(f"{record['source']} -[{record['relation']}]-> {record['target']}")
                
    return results


# ========== BUILD AUGMENTED PROMPT ==========
def build_augmented_prompt(question, graph_knowledge):
    context = "\n".join(graph_knowledge) if graph_knowledge else "No relevant context found."
    return f"""
You are a knowledgeable assistant with access to a knowledge graph.

Context from the knowledge graph:
{context}

User Question:
{question}

Based on the context above, generate a concise and accurate answer.
"""

# ========== QUERY OLLAMA ==========
def query_ollama_with_prompt(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json().get("response")
    else:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")

# ========== FULL RAG PIPELINE ==========
def rag_pipeline(question):
    context = retrieve_context_from_graph(question)
    print(f"CONTEXT:  {context}\n")

    prompt = build_augmented_prompt(question, context)
    response = query_ollama_with_prompt(prompt)
    
    print("✅ Final Answer:\n")
    print(response)

# ========== RUN INTERACTIVE ==========
if __name__ == "__main__":
    print("🔍 Retrieval-Augmented Generation (RAG) via Ollama + Neo4j")
    while True:
        try:
            question = input("\nAsk a question (or 'exit'): ").strip()
            if question.lower() in ["exit", "quit"]:
                break
            rag_pipeline(question)
        except Exception as e:
            print(f"❌ Error: {e}")
