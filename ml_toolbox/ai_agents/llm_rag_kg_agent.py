"""
LLM + RAG + Knowledge Graph Agent

Combines:
- LLMs for reasoning and generation
- RAG for knowledge retrieval
- Knowledge Graphs for structured knowledge
"""
from typing import Dict, List, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Import components
try:
    from ..llm_engineering import (
        PromptEngineer, RAGSystem, ChainOfThoughtReasoner,
        FewShotLearner, SafetyGuardrails
    )
    LLM_ENGINEERING_AVAILABLE = True
except ImportError:
    LLM_ENGINEERING_AVAILABLE = False

try:
    from .knowledge_graph_agent import KnowledgeGraphAgent
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    KnowledgeGraphAgent = None


class LLMRAGKGAgent:
    """
    Comprehensive AI Agent combining LLMs, RAG, and Knowledge Graphs
    
    Architecture:
    1. Knowledge Graph - Structured knowledge
    2. RAG System - Retrieval from documents
    3. LLM - Reasoning and generation
    4. Agent Orchestration - Coordinate all components
    """
    
    def __init__(self, toolbox=None):
        """
        Initialize LLM+RAG+KG Agent
        
        Parameters
        ----------
        toolbox : MLToolbox, optional
            ML Toolbox instance
        """
        self.toolbox = toolbox
        
        # Initialize components
        self.llm_components = {}
        self.rag_system = None
        self.kg_agent = None
        
        self._initialize_components()
        
        logger.info("[LLMRAGKGAgent] Initialized")
    
    def _initialize_components(self):
        """Initialize all components"""
        
        # LLM Engineering components
        if LLM_ENGINEERING_AVAILABLE:
            try:
                self.llm_components['prompt_engineer'] = PromptEngineer()
                self.llm_components['rag'] = RAGSystem()
                self.llm_components['cot'] = ChainOfThoughtReasoner()
                self.llm_components['few_shot'] = FewShotLearner()
                self.llm_components['safety'] = SafetyGuardrails()
                
                # Initialize few-shot examples
                self.llm_components['few_shot'].initialize_ml_examples()
                
                self.rag_system = self.llm_components['rag']
                logger.info("[LLMRAGKGAgent] LLM Engineering components enabled")
            except Exception as e:
                logger.warning(f"Could not initialize LLM Engineering: {e}")
        
        # Knowledge Graph Agent
        if KG_AVAILABLE and KnowledgeGraphAgent:
            try:
                self.kg_agent = KnowledgeGraphAgent()
                logger.info("[LLMRAGKGAgent] Knowledge Graph Agent enabled")
            except Exception as e:
                logger.warning(f"Could not initialize Knowledge Graph Agent: {e}")
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process query using LLM + RAG + Knowledge Graph
        
        Parameters
        ----------
        query : str
            User query
        context : dict, optional
            Additional context
            
        Returns
        -------
        result : dict
            Processed result with reasoning
        """
        # Step 1: Safety check
        if self.llm_components.get('safety'):
            safety_check = self.llm_components['safety'].check_prompt(query)
            if not safety_check['is_safe']:
                return {
                    'error': 'Safety check failed',
                    'safety_check': safety_check
                }
        
        # Step 2: Query Knowledge Graph
        kg_results = None
        if self.kg_agent:
            try:
                kg_results = self.kg_agent.query_graph(query)
                logger.info(f"[LLMRAGKGAgent] KG query returned {len(kg_results.get('results', []))} results")
            except Exception as e:
                logger.warning(f"KG query failed: {e}")
        
        # Step 3: Retrieve from RAG
        rag_context = None
        if self.rag_system:
            try:
                # Augment query with KG context
                enhanced_query = query
                if kg_results and kg_results.get('results'):
                    kg_info = " ".join([str(r) for r in kg_results['results'][:3]])
                    enhanced_query = f"{query} Context: {kg_info}"
                
                # Retrieve relevant documents
                retrieved_docs = self.rag_system.retriever.retrieve(enhanced_query, top_k=3)
                if retrieved_docs:
                    rag_context = "\n".join([doc['content'] for doc in retrieved_docs])
                    logger.info(f"[LLMRAGKGAgent] RAG retrieved {len(retrieved_docs)} documents")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Step 4: Build prompt with Chain-of-Thought
        prompt = None
        if self.llm_components.get('prompt_engineer'):
            try:
                # Determine task type
                task_type = self._detect_task_type(query)
                
                # Create prompt with context
                prompt_data = {
                    'task_description': query,
                    'kg_context': str(kg_results) if kg_results else None,
                    'rag_context': rag_context
                }
                
                # Use Chain-of-Thought for complex queries
                if self.llm_components.get('cot') and self._is_complex_query(query):
                    reasoning_steps = self.llm_components['cot'].break_down_task(query)
                    prompt = self.llm_components['cot'].create_reasoning_prompt(query, 'problem_solving')
                else:
                    prompt = self.llm_components['prompt_engineer'].create_prompt(task_type, **prompt_data)
                
                # Augment with RAG context
                if rag_context and self.rag_system:
                    prompt = self.rag_system.augment_prompt(prompt, query, top_k=3)
                
                logger.info(f"[LLMRAGKGAgent] Generated prompt (length: {len(prompt)})")
            except Exception as e:
                logger.warning(f"Prompt generation failed: {e}")
        
        # Step 5: Generate response (simulated - in production, call actual LLM)
        response = self._generate_response(query, prompt, kg_results, rag_context)
        
        # Step 6: Update Knowledge Graph with new information
        if self.kg_agent and response.get('new_information'):
            try:
                self.kg_agent.build_from_text(response['new_information'], doc_id='agent_response')
            except Exception as e:
                logger.warning(f"KG update failed: {e}")
        
        return {
            'query': query,
            'response': response,
            'kg_results': kg_results,
            'rag_context': rag_context[:200] if rag_context else None,  # Truncate for display
            'reasoning': response.get('reasoning', []),
            'sources': {
                'knowledge_graph': kg_results is not None,
                'rag': rag_context is not None,
                'llm': prompt is not None
            }
        }
    
    def _detect_task_type(self, query: str) -> str:
        """Detect task type from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['classify', 'classification', 'predict category']):
            return 'classification'
        elif any(word in query_lower for word in ['predict', 'forecast', 'regression']):
            return 'regression'
        elif any(word in query_lower for word in ['analyze', 'explore', 'understand']):
            return 'analysis'
        elif any(word in query_lower for word in ['feature', 'engineer', 'transform']):
            return 'feature_engineering'
        else:
            return 'general'
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if query is complex"""
        complex_indicators = ['how', 'why', 'explain', 'analyze', 'compare', 'multiple']
        return any(indicator in query.lower() for indicator in complex_indicators)
    
    def _generate_response(self, query: str, prompt: Optional[str], 
                          kg_results: Optional[Dict], rag_context: Optional[str]) -> Dict:
        """
        Generate response (simulated - in production, call actual LLM)
        
        Parameters
        ----------
        query : str
            Original query
        prompt : str, optional
            Generated prompt
        kg_results : dict, optional
            Knowledge graph results
        rag_context : str, optional
            RAG context
            
        Returns
        -------
        response : dict
            Generated response
        """
        # Simulated response generation
        # In production, this would call an actual LLM API
        
        reasoning = []
        
        # Use KG results if available
        if kg_results and kg_results.get('results'):
            reasoning.append(f"Found {len(kg_results['results'])} relevant entities in knowledge graph")
            reasoning.append("Using structured knowledge to inform response")
        
        # Use RAG context if available
        if rag_context:
            reasoning.append("Retrieved relevant documents from knowledge base")
            reasoning.append("Using retrieved context to enhance response")
        
        # Generate response based on query type
        task_type = self._detect_task_type(query)
        
        if task_type == 'classification':
            response_text = "I'll help you with classification. Based on the knowledge graph and retrieved documents, I recommend using a Random Forest classifier for this task."
        elif task_type == 'regression':
            response_text = "I'll help you with regression. Based on the structured knowledge, I recommend using Gradient Boosting for better performance."
        elif task_type == 'analysis':
            response_text = "I'll analyze the data. Using knowledge graph relationships and retrieved context, I can provide comprehensive insights."
        else:
            response_text = f"I'll help you with: {query}. Using combined knowledge from graph, documents, and reasoning."
        
        return {
            'text': response_text,
            'reasoning': reasoning,
            'task_type': task_type,
            'confidence': 0.85
        }
    
    def add_knowledge(self, text: str, doc_id: str, add_to_kg: bool = True, 
                     add_to_rag: bool = True):
        """
        Add knowledge to the system
        
        Parameters
        ----------
        text : str
            Knowledge text
        doc_id : str
            Document identifier
        add_to_kg : bool
            Add to knowledge graph
        add_to_rag : bool
            Add to RAG system
        """
        # Add to Knowledge Graph
        if add_to_kg and self.kg_agent:
            try:
                self.kg_agent.build_from_text(text, doc_id)
                logger.info(f"[LLMRAGKGAgent] Added to KG: {doc_id}")
            except Exception as e:
                logger.warning(f"Failed to add to KG: {e}")
        
        # Add to RAG
        if add_to_rag and self.rag_system:
            try:
                self.rag_system.add_knowledge(doc_id, text)
                logger.info(f"[LLMRAGKGAgent] Added to RAG: {doc_id}")
            except Exception as e:
                logger.warning(f"Failed to add to RAG: {e}")
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        stats = {
            'components': {
                'llm_engineering': LLM_ENGINEERING_AVAILABLE,
                'rag': self.rag_system is not None,
                'knowledge_graph': self.kg_agent is not None
            }
        }
        
        if self.rag_system:
            stats['rag'] = self.rag_system.get_retrieval_stats()
        
        if self.kg_agent:
            stats['knowledge_graph'] = self.kg_agent.get_graph().get_statistics()
        
        return stats
