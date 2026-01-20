"""
Understanding Bible App - API Backend
Deep understanding and scholar-level insights powered by AI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
from vector_db.faiss_db import FAISSVectorDB
from llm.quantum_llm_standalone import StandaloneQuantumLLM
from rag import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
understanding_rag = None
kernel = None
vector_db = None
llm = None


def initialize_understanding_service():
    """Initialize Understanding Bible service"""
    global understanding_rag, kernel, vector_db, llm
    
    logger.info("Initializing Understanding Bible service...")
    
    # Initialize kernel with quantum methods
    config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum',
        cache_type='lru',
        cache_size=10000
    )
    kernel = get_kernel(config)
    
    # Initialize vector database
    vector_db = FAISSVectorDB(dimension=384)
    
    # Initialize LLM with longer generation capability
    llm = StandaloneQuantumLLM(
        kernel=kernel,
        config={
            'use_quantum_sampling': True,
            'use_quantum_coherence': True,
            'confidence_threshold': 0.6
        }
    )
    
    # Initialize RAG system
    understanding_rag = RAGSystem(kernel, vector_db, llm)
    
    logger.info("Understanding Bible service initialized successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    initialize_understanding_service()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Understanding Bible App API",
    description="Deep understanding and scholar-level insights for Bible study",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class UnderstandingRequest(BaseModel):
    verse_reference: str
    verse_text: str
    depth_level: str = "deep"  # "basic", "deep", "scholar"
    scholar_style: Optional[str] = None  # "academic", "devotional", "pastor", "theologian"


class UnderstandingResponse(BaseModel):
    verse_reference: str
    verse_text: str
    understanding: Dict  # Contains: explanation, context, application, reflection
    related_verses: List[Dict]
    confidence: float


class ScholarVoiceRequest(BaseModel):
    verse_reference: str
    verse_text: str
    scholar_style: str  # "academic", "devotional", "pastor", "theologian"
    length: str = "medium"  # "short", "medium", "long", "book"


class ScholarVoiceResponse(BaseModel):
    verse_reference: str
    scholar_style: str
    explanation: str
    length: int
    confidence: float


class DailyUnderstandingRequest(BaseModel):
    date: Optional[str] = None  # ISO format, defaults to today


class DailyUnderstandingResponse(BaseModel):
    date: str
    verse_reference: str
    verse_text: str
    understanding: Dict
    related_verses: List[Dict]
    reflection_questions: List[str]


class ConnectionDiscoveryRequest(BaseModel):
    verse_reference: str
    verse_text: str
    top_k: int = 10


class ConnectionDiscoveryResponse(BaseModel):
    verse_reference: str
    connections: List[Dict]  # Related verses with explanations
    themes_discovered: List[str]


class JournalEntry(BaseModel):
    user_id: Optional[str] = None
    verse_reference: str
    insights: str
    questions: Optional[str] = None
    date: Optional[str] = None


# Helper function to generate deep understanding
def generate_deep_understanding(verse_reference: str, verse_text: str, 
                                depth_level: str = "deep",
                                scholar_style: Optional[str] = None) -> Dict:
    """Generate deep understanding of a verse"""
    
    # Build prompt based on depth and style
    if scholar_style:
        style_prompts = {
            "academic": "Write an academic, scholarly explanation with historical, linguistic, and cultural context. Be thorough and precise.",
            "devotional": "Write a devotional explanation focusing on practical application and spiritual growth. Be warm and encouraging.",
            "pastor": "Write in a teaching/pastor style with relatable examples and clear application. Be engaging and accessible.",
            "theologian": "Write a theological explanation exploring deep doctrinal insights. Be thoughtful and profound."
        }
        style_prompt = style_prompts.get(scholar_style.lower(), style_prompts["devotional"])
    else:
        style_prompt = "Write in a scholarly yet accessible style."
    
    depth_prompts = {
        "basic": "Provide a brief explanation (200-300 words).",
        "deep": "Provide a thorough, in-depth explanation (500-1000 words) covering context, meaning, and application.",
        "scholar": "Provide a comprehensive, scholar-level explanation (1000-2000 words) with extensive context, analysis, and insights."
    }
    depth_prompt = depth_prompts.get(depth_level, depth_prompts["deep"])
    
    # Build full prompt
    query = f"""
Verse: {verse_reference}
Text: "{verse_text}"

{style_prompt}
{depth_prompt}

Provide:
1. Deep Explanation - What this verse means in context
2. Historical Context - When and why it was written
3. Cultural Background - What the original audience would understand
4. Application - How this applies to life today
5. Reflection Questions - Questions for personal growth

Write as if you're a Bible scholar who deeply understands Scripture and wants to help readers grow in their relationship with God.
"""
    
    # Generate with RAG (longer generation for deep understanding)
    max_length = {
        "basic": 300,
        "deep": 1000,
        "scholar": 2000
    }.get(depth_level, 1000)
    
    result = understanding_rag.generate(
        query=query,
        top_k=10,  # More context for better understanding
        max_length=max_length,
        temperature=0.7,
        include_sources=True
    )
    
    # Parse the generated explanation
    explanation_text = result.get('answer', result.get('generated', ''))
    
    # Extract sections (basic parsing)
    understanding = {
        "explanation": explanation_text,
        "context": "",  # Would be extracted or generated separately
        "application": "",
        "reflection": ""
    }
    
    # Try to extract sections if the LLM structured it
    sections = explanation_text.split('\n\n')
    for section in sections:
        if 'explanation' in section.lower() or 'meaning' in section.lower():
            understanding["explanation"] = section
        elif 'context' in section.lower() or 'historical' in section.lower():
            understanding["context"] = section
        elif 'application' in section.lower() or 'applies' in section.lower():
            understanding["application"] = section
        elif 'reflection' in section.lower() or 'question' in section.lower():
            understanding["reflection"] = section
    
    # If sections not found, use full explanation
    if not understanding["explanation"] or len(understanding["explanation"]) < 100:
        understanding["explanation"] = explanation_text
    
    return understanding


@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Understanding Bible App API",
        "initialized": understanding_rag is not None
    }


@app.post("/api/understanding/generate", response_model=UnderstandingResponse)
async def generate_understanding(request: UnderstandingRequest):
    """
    Generate deep understanding of a Bible verse
    Returns scholar-level explanation, context, application, and reflection
    """
    if understanding_rag is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Generate understanding
        understanding = generate_deep_understanding(
            request.verse_reference,
            request.verse_text,
            request.depth_level,
            request.scholar_style
        )
        
        # Find related verses
        query_embedding = kernel.embed(request.verse_text)
        query_embedding = query_embedding.reshape(1, -1)
        related_results = vector_db.search(query_embedding, k=10)
        
        related_verses = []
        for result in related_results[:5]:  # Top 5
            if isinstance(result, tuple) and len(result) >= 3:
                doc_id, similarity, metadata = result[0], result[1], result[2]
                doc_text = understanding_rag.document_map.get(doc_id, '')
                if doc_text != request.verse_text:  # Don't include self
                    related_verses.append({
                        'text': doc_text,
                        'reference': metadata.get('reference', 'Unknown'),
                        'book': metadata.get('book', 'Unknown'),
                        'similarity': float(similarity),
                        'connection': "Semantically related - similar meaning or theme"
                    })
        
        return UnderstandingResponse(
            verse_reference=request.verse_reference,
            verse_text=request.verse_text,
            understanding=understanding,
            related_verses=related_verses,
            confidence=0.85  # Would be calculated from RAG confidence
        )
    
    except Exception as e:
        logger.error(f"Error generating understanding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/understanding/scholar-voice", response_model=ScholarVoiceResponse)
async def generate_scholar_voice(request: ScholarVoiceRequest):
    """
    Generate explanation in a specific scholar's voice/style
    Can generate book-length explanations, not just summaries
    """
    if understanding_rag is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Determine length
        length_map = {
            "short": 500,
            "medium": 1500,
            "long": 3000,
            "book": 8000  # Book-length!
        }
        max_length = length_map.get(request.length, 1500)
        
        # Build scholar-specific prompt
        scholar_prompts = {
            "academic": """
You are an academic Bible scholar. Write a comprehensive, scholarly explanation of this verse.
Include:
- Historical-critical analysis
- Linguistic insights (original languages)
- Cultural and social context
- Scholarly interpretations
- Textual analysis
Write in an academic, scholarly style. Be thorough and precise.
""",
            "devotional": """
You are a devotional writer who helps people grow spiritually. Write a warm, encouraging explanation.
Include:
- Personal spiritual insights
- Practical application for daily life
- Connection to God's character
- Encouragement for growth
Write in a devotional, encouraging style. Be warm and accessible.
""",
            "pastor": """
You are a pastor teaching this verse to your congregation. Write a clear, engaging explanation.
Include:
- Clear explanation of the meaning
- Relatable examples and illustrations
- Practical application
- Teaching points
Write in a teaching/pastor style. Be engaging and accessible.
""",
            "theologian": """
You are a theologian exploring deep theological insights. Write a profound, thoughtful explanation.
Include:
- Doctrinal implications
- Theological connections
- Deep spiritual insights
- Systematic theology links
Write in a theological, profound style. Be thoughtful and deep.
"""
        }
        
        scholar_prompt = scholar_prompts.get(
            request.scholar_style.lower(),
            scholar_prompts["devotional"]
        )
        
        query = f"""
Verse: {request.verse_reference}
Text: "{request.verse_text}"

{scholar_prompt}

Generate a {request.length} explanation ({max_length} words or more). This is not a summary - write the actual full content as if you were writing a book chapter on this verse.
"""
        
        # Generate (long generation for book-length content)
        result = understanding_rag.generate(
            query=query,
            top_k=15,  # More context for scholar-level content
            max_length=max_length,
            temperature=0.7,
            include_sources=True
        )
        
        explanation = result.get('answer', result.get('generated', ''))
        
        return ScholarVoiceResponse(
            verse_reference=request.verse_reference,
            scholar_style=request.scholar_style,
            explanation=explanation,
            length=len(explanation),
            confidence=result.get('confidence', 0.8)
        )
    
    except Exception as e:
        logger.error(f"Error generating scholar voice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/understanding/daily", response_model=DailyUnderstandingResponse)
async def get_daily_understanding(request: DailyUnderstandingRequest):
    """
    Get daily verse with understanding
    No reading plan pressure - just understanding for today
    """
    if understanding_rag is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # For now, use a sample verse (in production, would select based on date, user progress, etc.)
        # TODO: Implement daily verse selection algorithm
        
        sample_verse = {
            "reference": "Psalm 23:1",
            "text": "The Lord is my shepherd, I lack nothing."
        }
        
        # Generate understanding
        understanding = generate_deep_understanding(
            sample_verse["reference"],
            sample_verse["text"],
            depth_level="deep"
        )
        
        # Find related verses
        query_embedding = kernel.embed(sample_verse["text"])
        query_embedding = query_embedding.reshape(1, -1)
        related_results = vector_db.search(query_embedding, k=5)
        
        related_verses = []
        for result in related_results:
            if isinstance(result, tuple) and len(result) >= 3:
                doc_id, similarity, metadata = result[0], result[1], result[2]
                doc_text = understanding_rag.document_map.get(doc_id, '')
                related_verses.append({
                    'text': doc_text,
                    'reference': metadata.get('reference', 'Unknown'),
                    'book': metadata.get('book', 'Unknown'),
                    'similarity': float(similarity)
                })
        
        # Generate reflection questions
        reflection_query = f"Generate 3-5 thoughtful reflection questions for: {sample_verse['text']}"
        reflection_result = understanding_rag.generate(
            query=reflection_query,
            top_k=3,
            max_length=200,
            temperature=0.8
        )
        
        # Extract questions (simple parsing)
        questions_text = reflection_result.get('answer', '')
        questions = [
            q.strip() 
            for q in questions_text.split('\n') 
            if q.strip() and ('?' in q or q[0].isdigit())
        ][:5]
        
        if not questions:
            questions = [
                "What does this verse mean to you personally?",
                "How does this verse relate to your relationship with God?",
                "What action can you take based on this verse?"
            ]
        
        date = request.date or datetime.now().isoformat().split('T')[0]
        
        return DailyUnderstandingResponse(
            date=date,
            verse_reference=sample_verse["reference"],
            verse_text=sample_verse["text"],
            understanding=understanding,
            related_verses=related_verses,
            reflection_questions=questions
        )
    
    except Exception as e:
        logger.error(f"Error getting daily understanding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/understanding/search", response_model=ConnectionDiscoveryResponse)
async def search_verses(request: ConnectionDiscoveryRequest):
    """
    Search verses semantically
    Finds verses by meaning, not just keywords
    """
    if understanding_rag is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Semantic search for verses
        query_embedding = kernel.embed(request.verse_text or request.verse_reference)
        query_embedding = query_embedding.reshape(1, -1)
        results = vector_db.search(query_embedding, k=request.top_k)
        
        verses = []
        for result in results:
            if isinstance(result, tuple) and len(result) >= 3:
                doc_id, similarity, metadata = result[0], result[1], result[2]
                doc_text = understanding_rag.document_map.get(doc_id, '')
                verses.append({
                    'reference': metadata.get('reference', 'Unknown'),
                    'verse_reference': metadata.get('reference', 'Unknown'),
                    'text': doc_text,
                    'verse_text': doc_text,
                    'book': metadata.get('book', 'Unknown'),
                    'similarity': float(similarity)
                })
        
        return ConnectionDiscoveryResponse(
            verse_reference=request.verse_reference or 'Search',
            connections=verses,
            themes_discovered=[]
        )
    
    except Exception as e:
        logger.error(f"Error searching verses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/understanding/connections", response_model=ConnectionDiscoveryResponse)
async def discover_connections(request: ConnectionDiscoveryRequest):
    """
    Discover connections between verses
    Semantic search finds meaning-based relationships
    """
    if understanding_rag is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Semantic search for related verses
        query_embedding = kernel.embed(request.verse_text)
        query_embedding = query_embedding.reshape(1, -1)
        results = vector_db.search(query_embedding, k=request.top_k)
        
        connections = []
        themes = set()
        
        for result in results:
            if isinstance(result, tuple) and len(result) >= 3:
                doc_id, similarity, metadata = result[0], result[1], result[2]
                doc_text = understanding_rag.document_map.get(doc_id, '')
                
                if doc_text != request.verse_text:  # Don't include self
                    connection_info = {
                        'verse_reference': metadata.get('reference', 'Unknown'),
                        'verse_text': doc_text,
                        'book': metadata.get('book', 'Unknown'),
                        'similarity': float(similarity),
                        'connection_explanation': f"Similar meaning or theme (similarity: {float(similarity):.2f})"
                    }
                    connections.append(connection_info)
                    
                    # Extract themes
                    theme = metadata.get('theme', '')
                    if theme:
                        themes.add(theme)
        
        # Discover themes using kernel
        all_texts = [conn['verse_text'] for conn in connections[:5]]
        if all_texts:
            discovered_themes = kernel.discover_themes(all_texts, min_cluster_size=2)
            for theme_obj in discovered_themes:
                themes.add(theme_obj.get('theme', 'Unknown Theme'))
        
        return ConnectionDiscoveryResponse(
            verse_reference=request.verse_reference,
            connections=connections,
            themes_discovered=list(themes)[:10]
        )
    
    except Exception as e:
        logger.error(f"Error discovering connections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/bible/add-content")
async def add_bible_content(verses: List[Dict]):
    """Add Bible verses to the knowledge base"""
    if understanding_rag is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        documents = []
        metadata_list = []
        
        for verse in verses:
            documents.append(verse.get('text', ''))
            metadata_list.append({
                'reference': verse.get('reference', 'Unknown'),
                'book': verse.get('book', 'Unknown'),
                'chapter': verse.get('chapter', ''),
                'verse': verse.get('verse', ''),
                'theme': verse.get('theme', '')
            })
        
        understanding_rag.add_knowledge(documents, metadata=metadata_list)
        
        return {
            "status": "success",
            "verses_added": len(documents),
            "total_verses": len(understanding_rag.documents) if hasattr(understanding_rag, 'documents') else 0
        }
    
    except Exception as e:
        logger.error(f"Error adding Bible content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/journal/save")
async def save_journal_entry(entry: JournalEntry):
    """Save journal entry (insights, questions)"""
    # TODO: Implement journal storage (SQLite, PostgreSQL, etc.)
    # For now, return success
    return {
        "status": "success",
        "entry_id": f"entry_{datetime.now().timestamp()}",
        "saved_at": datetime.now().isoformat()
    }


@app.get("/api/journal/entries")
async def get_journal_entries(user_id: Optional[str] = None):
    """Get journal entries"""
    # TODO: Implement journal retrieval
    return {
        "entries": [],
        "count": 0
    }


if __name__ == "__main__":
    import uvicorn
    import os
    import socket
    
    # Find free port
    def find_free_port(start_port=8003, end_port=8010):
        for port in range(start_port, end_port + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                sock.close()
                continue
        return None
    
    port = int(os.getenv("UNDERSTANDING_APP_PORT", 0))
    if port == 0:
        port = find_free_port()
        if port is None:
            logger.error("Could not find free port")
            exit(1)
        logger.info(f"Using port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
