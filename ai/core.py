"""
Complete AI System Core
Main system that integrates all components
"""
from quantum_kernel import QuantumKernel, KernelConfig, get_kernel
from .components import (
    SemanticUnderstandingEngine,
    KnowledgeGraphBuilder,
    IntelligentSearch,
    ReasoningEngine,
    LearningSystem,
    ConversationalAI
)
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteAISystem:
    """
    Complete AI system built around quantum kernel
    Integrates all AI capabilities into one system
    """
    
    def __init__(self, config: KernelConfig = None, use_llm: bool = False, llm=None):
        """
        Initialize the complete AI system
        
        Args:
            config: Optional kernel configuration. If None, uses defaults.
            use_llm: Whether to enable LLM integration for conversation and learning
            llm: Optional StandaloneQuantumLLM instance to use
        """
        # Initialize kernel
        if config is None:
            config = KernelConfig(
                embedding_dim=256,
                cache_size=50000,
                enable_caching=True
            )
        
        self.kernel = get_kernel(config)
        
        # Initialize LLM if requested
        self.llm = llm
        if use_llm and llm is None:
            try:
                from llm.quantum_llm_standalone import StandaloneQuantumLLM
                self.llm = StandaloneQuantumLLM(kernel=self.kernel)
                logger.info("LLM initialized for AI System")
            except ImportError:
                logger.warning("LLM not available. Continuing without LLM integration.")
                self.llm = None
        
        # Build all components
        self.understanding = SemanticUnderstandingEngine(self.kernel)
        self.knowledge_graph = KnowledgeGraphBuilder(self.kernel)
        self.search = IntelligentSearch(self.kernel)
        self.reasoning = ReasoningEngine(self.kernel)
        self.learning = LearningSystem(self.kernel, use_llm=(self.llm is not None))
        self.conversation = ConversationalAI(self.kernel, llm=self.llm)
        
        logger.info("Complete AI System initialized" + (" with LLM" if self.llm else ""))
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input through complete AI system
        
        Args:
            input_data: Dictionary with keys:
                - query: Search query
                - message: Conversation message
                - premises: List of premises for reasoning
                - question: Question for reasoning
                - documents: List of documents for knowledge graph
                - examples: List of (input, output) tuples for learning
                - context: List of context strings
        
        Returns:
            Dictionary with results from relevant components
        """
        query = input_data.get("query", "")
        message = input_data.get("message", "")
        premises = input_data.get("premises", [])
        question = input_data.get("question", "")
        documents = input_data.get("documents", [])
        
        results = {}
        
        # Understanding
        if query or message:
            text = query or message
            results["understanding"] = self.understanding.understand_intent(
                text, input_data.get("context", [])
            )
        
        # Search
        if query and documents:
            results["search"] = self.search.search_and_discover(query, documents)
        
        # Reasoning
        if premises and question:
            results["reasoning"] = self.reasoning.reason(premises, question)
        
        # Conversation
        if message:
            results["conversation"] = self.conversation.respond(message)
        
        # Knowledge graph
        if documents:
            results["knowledge_graph"] = self.knowledge_graph.build_graph(documents)
        
        # Learning
        if input_data.get("examples"):
            results["learning"] = self.learning.learn_from_examples(
                input_data["examples"]
            )
        
        results["kernel_stats"] = self.kernel.get_stats()
        
        return results
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "kernel": self.kernel.get_stats(),
            "conversation_history_length": len(self.conversation.conversation_history),
            "learned_patterns": len(self.learning.learned_patterns),
            "knowledge_graph_nodes": len(self.knowledge_graph.graph.get("nodes", []))
        }
    
    def reset(self):
        """Reset the system (clear caches, history, etc.)"""
        self.kernel.clear_cache()
        self.conversation.clear_history()
        self.learning.learned_patterns = {}
        self.knowledge_graph.graph = {}
        logger.info("System reset complete")
