from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn
from rag_app import index_documents_in_folder, rag_query

# Lifespan handler to replace deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📁 Indexing documents on startup...")
    index_documents_in_folder()
    yield
    print("🛑 Server is shutting down...")

# App with lifespan
app = FastAPI(lifespan=lifespan)

# Request body model
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

# Endpoint to handle RAG queries
@app.post("/rag/query")
def query_docs(req: QueryRequest):
    try:
        print(f"📥 Received query: '{req.question}' (top_k={req.top_k})")
        result = rag_query(req.question, req.top_k)
        return {"response": result}
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return {"error": str(e)}

# Entry point
if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="0.0.0.0", port=8001)
