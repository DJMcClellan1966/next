"""
Retrieval-Augmented Generation (RAG) System
Combines vector search with LLM generation
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation System
    
    Combines vector database retrieval with LLM generation
    to provide grounded, source-cited responses
    """
    
    def __init__(self, kernel, llm, vector_db):
        """
        Initialize RAG system
        
        Args:
            kernel: QuantumKernel instance for embeddings
            llm: StandaloneQuantumLLM instance for generation
            vector_db: VectorDatabase instance for retrieval
        """
        self.kernel = kernel
        self.llm = llm
        self.vector_db = vector_db
        self.documents = []  # Store original documents
        self.document_map = {}  # id -> document
        
    def add_knowledge(self, documents: List[str], metadata: List[Dict] = None):
        """
        Add documents to knowledge base
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not documents:
            return
        
        # Embed all documents
        logger.info(f"Embedding {len(documents)} documents...")
        embeddings = self.kernel.embed_batch(documents, batch_size=32)
        
        # Generate IDs
        start_id = len(self.documents)
        ids = list(range(start_id, start_id + len(documents)))
        
        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in documents]
        
        # Add metadata with document text
        enhanced_metadata = []
        for doc, meta in zip(documents, metadata):
            enhanced_meta = {
                **meta,
                'text': doc[:200]  # Store first 200 chars for reference
            }
            enhanced_metadata.append(enhanced_meta)
        
        # Add to vector database
        self.vector_db.add_vectors(embeddings, ids, enhanced_metadata)
        
        # Store documents
        for doc_id, doc in zip(ids, documents):
            self.documents.append(doc)
            self.document_map[doc_id] = doc
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def generate(self, query: str, top_k: int = 5, max_length: int = 200,
                 temperature: float = 0.7, include_sources: bool = True) -> Dict:
        """
        Generate response with retrieval-augmented context
        
        Args:
            query: User query
            top_k: Number of relevant documents to retrieve
            max_length: Maximum generation length
            temperature: Generation temperature
            include_sources: Whether to include source citations
        
        Returns:
            Dictionary with answer, sources, confidence, etc.
        """
        # 1. Retrieve relevant documents
        logger.info(f"Retrieving top {top_k} documents for query: {query[:50]}...")
        query_embedding = self.kernel.embed(query)
        retrieved = self.vector_db.search(query_embedding, k=top_k)
        
        if not retrieved:
            return {
                'answer': "I couldn't find any relevant information.",
                'sources': [],
                'confidence': 0.0,
                'context_used': '',
                'warning': 'No relevant documents found'
            }
        
        # 2. Build context from retrieved documents
        context = self._build_context(retrieved)
        
        # 3. Build prompt with context
        prompt = self._build_prompt(query, context)
        
        # 4. Generate with LLM
        logger.info("Generating response with RAG...")
        result = self.llm.generate_grounded(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            require_validation=False  # More flexible for RAG
        )
        
        # 5. Extract answer
        generated_text = result.get('generated', '')
        answer = self._extract_answer(generated_text, query)
        
        # 6. Build sources list
        sources = []
        if include_sources:
            for doc_id, similarity, meta in retrieved:
                source_info = {
                    'id': doc_id,
                    'relevance': float(similarity),
                    'document': self.document_map.get(doc_id, ''),
                    'metadata': meta
                }
                sources.append(source_info)
        
        # 7. Build citations
        citations = self._build_citations(sources)
        
        return {
            'answer': answer,
            'sources': sources,
            'citations': citations,
            'confidence': result.get('confidence', 0.0),
            'context_used': context,
            'retrieved_count': len(retrieved),
            'generation_stats': {
                'model_confidence': result.get('confidence', 0.0),
                'is_safe': result.get('is_safe', True)
            }
        }
    
    def _build_context(self, retrieved: List[Tuple]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, (doc_id, similarity, meta) in enumerate(retrieved):
            doc = self.document_map.get(doc_id, '')
            if doc:
                context_parts.append(f"[Document {i+1} (Relevance: {similarity:.2f})]\n{doc[:500]}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt with context for LLM"""
        prompt = f"""Context Information:
{context}

Question: {query}

Answer the question based on the context information provided above. 
If the answer cannot be found in the context, say so.
If you use information from the context, note which document it came from.
"""
        return prompt
    
    def _extract_answer(self, generated_text: str, query: str) -> str:
        """Extract answer from generated text"""
        # Try to find answer after question
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[1].strip()
        elif "Question:" in generated_text:
            # Extract text after question
            parts = generated_text.split("Question:")
            if len(parts) > 1:
                answer = parts[1].split("\n")[0].strip()
            else:
                answer = generated_text
        else:
            # Use generated text as-is
            answer = generated_text.strip()
        
        # Clean up
        answer = answer.split("\n\n")[0]  # Take first paragraph
        return answer
    
    def _build_citations(self, sources: List[Dict]) -> List[str]:
        """Build citation strings"""
        citations = []
        for i, source in enumerate(sources):
            citation = f"[{i+1}] "
            if 'metadata' in source and source['metadata']:
                meta = source['metadata']
                if 'title' in meta:
                    citation += f"{meta['title']}"
                elif 'source' in meta:
                    citation += f"{meta['source']}"
                else:
                    citation += f"Document {source['id']}"
            else:
                citation += f"Document {source['id']}"
            
            if 'metadata' in source and 'page' in source['metadata']:
                citation += f", page {source['metadata']['page']}"
            
            citations.append(citation)
        
        return citations
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Simple semantic search without generation"""
        query_embedding = self.kernel.embed(query)
        retrieved = self.vector_db.search(query_embedding, k=top_k)
        
        results = []
        for doc_id, similarity, meta in retrieved:
            results.append({
                'id': doc_id,
                'document': self.document_map.get(doc_id, ''),
                'similarity': float(similarity),
                'metadata': meta
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        return {
            'total_documents': len(self.documents),
            'vector_db_stats': self.vector_db.get_stats(),
            'kernel_stats': self.kernel.get_stats()
        }
