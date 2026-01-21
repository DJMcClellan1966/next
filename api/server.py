"""
Production API Server
FastAPI-based REST and WebSocket API for AI platform
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, List as ListType
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quantum AI Platform API",
    description="Production-ready AI platform with RAG, vector search, and generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use dependency injection)
_kernel = None
_llm = None
_rag = None
_vector_db = None


# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7
    top_k: int = 5


class ChatRequest(BaseModel):
    message: str
    context: Optional[List[str]] = None
    top_k: int = 5


class AddKnowledgeRequest(BaseModel):
    documents: List[str]
    metadata: Optional[List[Dict]] = None


class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    count: int


class GenerateResponse(BaseModel):
    answer: str
    sources: List[Dict]
    citations: List[str]
    confidence: float
    retrieved_count: int


class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict]] = None
    citations: Optional[List[str]] = None
    confidence: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    stats: Dict


# Dependency Injection Functions
def get_kernel():
    """Get kernel instance"""
    global _kernel
    if _kernel is None:
        from quantum_kernel import get_kernel
        _kernel = get_kernel()
    return _kernel


def get_llm():
    """Get LLM instance"""
    global _llm
    if _llm is None:
        try:
            from llm.quantum_llm_standalone import StandaloneQuantumLLM
            kernel = get_kernel()
            _llm = StandaloneQuantumLLM(kernel=kernel)
        except ImportError:
            logger.warning("LLM not available")
            return None
    return _llm


def get_vector_db():
    """Get vector database instance"""
    global _vector_db
    if _vector_db is None:
        try:
            from vector_db import FAISSVectorDB
            kernel = get_kernel()
            dimension = kernel.config.embedding_dim
            # Adjust dimension if using sentence transformers
            if kernel.use_sentence_transformers:
                dimension = 384  # Sentence transformer default
            
            _vector_db = FAISSVectorDB(
                dimension=dimension,
                index_type='IP',  # Inner product for cosine similarity
                use_gpu=False  # Set to True if GPU available
            )
        except ImportError:
            logger.warning("Vector DB not available")
            return None
    return _vector_db


def get_rag():
    """Get RAG system instance"""
    global _rag
    if _rag is None:
        try:
            from rag import RAGSystem
            kernel = get_kernel()
            llm = get_llm()
            vector_db = get_vector_db()
            
            if llm is None or vector_db is None:
                logger.warning("RAG requires both LLM and Vector DB")
                return None
            
            _rag = RAGSystem(kernel=kernel, llm=llm, vector_db=vector_db)
        except ImportError:
            logger.warning("RAG system not available")
            return None
    return _rag


# Routes

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum AI Platform API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/api/search",
            "generate": "/api/generate",
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    kernel = get_kernel()
    stats = kernel.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        stats=stats
    )


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search endpoint"""
    kernel = get_kernel()
    vector_db = get_vector_db()
    
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not available")
    
    # Embed query
    query_embedding = kernel.embed(request.query)
    
    # Search
    results_raw = vector_db.search(query_embedding, k=request.top_k)
    
    # Format results
    results = []
    for doc_id, similarity, metadata in results_raw:
        results.append({
            "id": doc_id,
            "similarity": float(similarity),
            "metadata": metadata or {}
        })
    
    return SearchResponse(
        query=request.query,
        results=results,
        count=len(results)
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """RAG-based text generation endpoint"""
    rag = get_rag()
    
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    # Generate with RAG
    result = rag.generate(
        query=request.prompt,
        top_k=request.top_k,
        max_length=request.max_length,
        temperature=request.temperature
    )
    
    return GenerateResponse(
        answer=result['answer'],
        sources=result['sources'],
        citations=result['citations'],
        confidence=result['confidence'],
        retrieved_count=result['retrieved_count']
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG"""
    rag = get_rag()
    
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    # Generate response
    result = rag.generate(
        query=request.message,
        top_k=request.top_k,
        max_length=200,
        temperature=0.7
    )
    
    # Build response with sources
    response_text = result['answer']
    if result['citations']:
        response_text += "\n\nSources:\n" + "\n".join(result['citations'])
    
    return ChatResponse(
        response=response_text,
        sources=result['sources'],
        citations=result['citations'],
        confidence=result['confidence']
    )


@app.post("/api/knowledge/add")
async def add_knowledge(request: AddKnowledgeRequest):
    """Add documents to knowledge base"""
    rag = get_rag()
    
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    # Add documents
    rag.add_knowledge(
        documents=request.documents,
        metadata=request.metadata
    )
    
    return {
        "status": "success",
        "documents_added": len(request.documents),
        "total_documents": len(rag.documents)
    }


@app.get("/api/stats")
async def stats():
    """Get system statistics"""
    kernel = get_kernel()
    rag = get_rag()
    
    stats_dict = {
        "kernel": kernel.get_stats()
    }
    
    if rag:
        stats_dict["rag"] = rag.get_stats()
    
    return stats_dict


# WebSocket Streaming Endpoint
@app.websocket("/api/stream")
async def stream_generate(websocket: WebSocket):
    """Streaming text generation endpoint"""
    await websocket.accept()
    
    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            query = data.get("query", "")
            top_k = data.get("top_k", 5)
            max_length = data.get("max_length", 200)
            
            if not query:
                await websocket.send_json({"error": "No query provided"})
                continue
            
            rag = get_rag()
            if rag is None:
                await websocket.send_json({"error": "RAG system not available"})
                continue
            
            # Generate with streaming
            result = rag.generate(
                query=query,
                top_k=top_k,
                max_length=max_length
            )
            
            # Stream answer token by token
            answer = result['answer']
            tokens = answer.split()
            
            full_text = ""
            for token in tokens:
                full_text += token + " "
                
                await websocket.send_json({
                    "token": token + " ",
                    "full_text": full_text.strip(),
                    "done": False
                })
                
                # Small delay for streaming effect
                await asyncio.sleep(0.05)
            
            # Send final message with sources
            await websocket.send_json({
                "token": "",
                "full_text": full_text.strip(),
                "done": True,
                "sources": result.get('citations', []),
                "confidence": result.get('confidence', 0.0)
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    import os
    # Allow port to be configured via environment variable
    port = int(os.getenv("PORT", 8001))  # Default to 8001 if 8000 is busy
    uvicorn.run(app, host="0.0.0.0", port=port)
