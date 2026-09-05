import os
import shutil
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import run_agent_query
from agent.llm import LocalOllamaClient
from ingestion.chunker import DocumentChunker
from ingestion.vector_store import ChromaVectorStore

app = FastAPI(
    title="ResearchPilot API",
    description="100% Free, Local, Open-Source Agentic RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vector_store = ChromaVectorStore()
chunker = DocumentChunker(chunk_size=600, chunk_overlap=100)
llm_client = LocalOllamaClient()

class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    query: str
    route: str
    route_reasoning: str
    final_answer: str
    sources_used: List[str]
    citations: List[dict]
    decision_trace: List[str]

@app.get("/api/health")
def health():
    ollama_ok = llm_client.is_available()
    models = llm_client.list_installed_models() if ollama_ok else []
    return {
        "status": "healthy",
        "ollama_available": ollama_ok,
        "ollama_models": models,
        "indexed_chunks": vector_store.count(),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (local)",
        "vector_db": "ChromaDB (local persistent)"
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = run_agent_query(req.query, thread_id=req.thread_id)
    return ChatResponse(
        query=req.query,
        route=result.get("route", "hybrid"),
        route_reasoning=result.get("route_reasoning", ""),
        final_answer=result.get("final_answer", ""),
        sources_used=result.get("sources_used", []),
        citations=result.get("citations", []),
        decision_trace=result.get("decision_trace", [])
    )

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    target_path = UPLOAD_DIR / file.filename
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Chunk and ingest
        chunks = chunker.chunk_document(str(target_path))
        num_added = vector_store.add_chunks(chunks)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "chunks_indexed": num_added,
            "total_collection_chunks": vector_store.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/api/documents")
def list_documents():
    return {
        "sources": vector_store.list_sources(),
        "total_chunks": vector_store.count()
    }

@app.delete("/api/documents")
def clear_documents():
    vector_store.clear()
    return {"status": "cleared", "total_chunks": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
