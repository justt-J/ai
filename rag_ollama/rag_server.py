from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import uvicorn
from rag_app import index_documents_in_folder, rag_query

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.on_event("startup")
def startup():
    index_documents_in_folder()

@app.post("/rag/query")
def query_docs(req: QueryRequest):
    try:
        print(f"Received query: {req.question} with top_k={req.top_k}")
        return {"response": rag_query(req.question, req.top_k)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="0.0.0.0", port=8001)
