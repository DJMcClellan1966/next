"""
AI System Components
Individual components that can be used independently or together
"""
from quantum_kernel import QuantumKernel
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticUnderstandingEngine:
    """Understand meaning, intent, and context with quantum-inspired methods"""
    
    def __init__(self, kernel: QuantumKernel, use_quantum_measurement: bool = True):
        self.kernel = kernel
        self.use_quantum_measurement = use_quantum_measurement
        self.known_intents = [
            "search for information",
            "ask a question",
            "request recommendation",
            "create content",
            "analyze data",
            "find relationships"
        ]
    
    def understand_intent(self, query: str, context: List[str] = None) -> Dict:
        """Understand user intent with quantum-inspired measurement"""
        if self.use_quantum_measurement:
            return self._quantum_measure_intent(query, context)
        else:
            return self._standard_understand_intent(query, context)
    
    def _quantum_measure_intent(self, query: str, context: List[str] = None) -> Dict:
        """Quantum-inspired probabilistic intent measurement"""
        import numpy as np
        
        # Create superposition of query
        query_embed = self.kernel.embed(query)
        
        # Measure against each intent (quantum measurement)
        measurements = []
        for intent in self.known_intents:
            intent_embed = self.kernel.embed(intent)
            
            # Use quantum interference similarity if available
            if self.kernel.config.similarity_metric == 'quantum' or \
               self.kernel.config.use_quantum_methods:
                from quantum_kernel.kernel import _quantum_interference_similarity
                amplitude = _quantum_interference_similarity(query_embed, intent_embed)
            else:
                amplitude = float(np.abs(np.dot(query_embed, intent_embed)))
            
            # Quantum probability (Born rule: probability = |amplitude|^2)
            probability = amplitude ** 2
            
            measurements.append((intent, probability))
        
        # Normalize probabilities (quantum normalization)
        total = sum(prob for _, prob in measurements)
        if total > 0:
            measurements = [(intent, prob/total) for intent, prob in measurements]
        
        # Sort by probability
        measurements.sort(key=lambda x: x[1], reverse=True)
        
        # Context relevance with quantum interference
        context_relevance = 0.0
        if context:
            context_text = " ".join(context)
            context_embed = self.kernel.embed(context_text)
            if self.kernel.config.use_quantum_methods:
                from quantum_kernel.kernel import _quantum_interference_similarity
                context_relevance = _quantum_interference_similarity(query_embed, context_embed)
            else:
                context_relevance = self.kernel.similarity(context_text, query)
        
        # Collapse to most likely intent (quantum collapse)
        most_likely = measurements[0][0] if measurements else "unknown"
        confidence = measurements[0][1] if measurements else 0.0
        
        return {
            "intent": most_likely,
            "confidence": confidence,
            "context_relevance": context_relevance,
            "probabilities": [
                {"intent": intent, "probability": prob}
                for intent, prob in measurements
            ],
            "alternative_intents": [
                {"intent": intent, "confidence": prob}
                for intent, prob in measurements[1:]
            ],
            "quantum_measured": True
        }
    
    def _standard_understand_intent(self, query: str, context: List[str] = None) -> Dict:
        """Standard intent understanding (fallback)"""
        # Find similar intents
        similar_intents = self.kernel.find_similar(
            query, self.known_intents, top_k=3
        )
        
        # Context relevance
        context_relevance = 0.0
        if context:
            context_text = " ".join(context)
            context_relevance = self.kernel.similarity(context_text, query)
        
        return {
            "intent": similar_intents[0][0] if similar_intents else "unknown",
            "confidence": similar_intents[0][1] if similar_intents else 0.0,
            "context_relevance": context_relevance,
            "alternative_intents": [
                {"intent": intent, "confidence": conf}
                for intent, conf in similar_intents[1:]
            ],
            "quantum_measured": False
        }
    
    def add_intent(self, intent: str):
        """Add a new intent to the system"""
        if intent not in self.known_intents:
            self.known_intents.append(intent)
            logger.info(f"Added new intent: {intent}")


class KnowledgeGraphBuilder:
    """Build and maintain knowledge graphs"""
    
    def __init__(self, kernel: QuantumKernel):
        self.kernel = kernel
        self.graph = {}
    
    def build_graph(self, documents: List[str]) -> Dict:
        """Build knowledge graph from documents"""
        # Discover relationships
        relationship_graph = self.kernel.build_relationship_graph(documents)
        
        # Discover themes
        themes = self.kernel.discover_themes(documents, min_cluster_size=2)
        
        # Build graph structure
        graph = {
            "nodes": [
                {"id": i, "text": doc, "embedding": self.kernel.embed(doc).tolist()}
                for i, doc in enumerate(documents)
            ],
            "edges": [
                {
                    "source": i,
                    "target": j,
                    "weight": sim,
                    "type": "semantic_similarity"
                }
                for i, (text, related) in enumerate(relationship_graph.items())
                for rel_text, sim in related
                for j, doc in enumerate(documents)
                if doc == rel_text
            ],
            "themes": [
                {
                    "theme": theme["theme"],
                    "nodes": [documents.index(t) for t in theme["texts"]],
                    "confidence": theme["confidence"]
                }
                for theme in themes
            ]
        }
        
        self.graph = graph
        return graph
    
    def get_graph(self) -> Dict:
        """Get the current knowledge graph"""
        return self.graph
    
    def add_document(self, document: str) -> Dict:
        """Efficiently add document to existing graph (incremental update)"""
        # Get existing documents
        existing_docs = [node['text'] for node in self.graph.get('nodes', [])]
        
        if not existing_docs:
            # First document - build initial graph
            return self.build_graph([document])
        
        # Compute similarities only with new document (O(n) instead of O(n²))
        new_edges = []
        new_node_id = len(existing_docs)
        
        # Get embedding for new document
        new_embedding = self.kernel.embed(document).tolist()
        
        for i, existing_doc in enumerate(existing_docs):
            sim = self.kernel.similarity(document, existing_doc)
            threshold = self.kernel.config.similarity_threshold
            if sim >= threshold:
                # Add edge in both directions
                new_edges.append({
                    "source": new_node_id,
                    "target": i,
                    "weight": sim,
                    "type": "semantic_similarity"
                })
                new_edges.append({
                    "source": i,
                    "target": new_node_id,
                    "weight": sim,
                    "type": "semantic_similarity"
                })
        
        # Add new node
        self.graph['nodes'].append({
            "id": new_node_id,
            "text": document,
            "embedding": new_embedding
        })
        
        # Add new edges
        self.graph['edges'].extend(new_edges)
        
        # Update themes incrementally (optional - can be expensive)
        # For now, skip theme update on incremental addition
        
        return self.graph


class IntelligentSearch:
    """Advanced search with discovery"""
    
    def __init__(self, kernel: QuantumKernel):
        self.kernel = kernel
    
    def search_and_discover(self, query: str, corpus: List[str]) -> Dict:
        """Search and discover related concepts"""
        # Semantic search
        results = self.kernel.find_similar(query, corpus, top_k=20)
        
        # Discover related concepts from top results
        top_results_texts = [text for text, _ in results[:5]]
        if top_results_texts:
            related_concepts = self.kernel.build_relationship_graph(top_results_texts)
        else:
            related_concepts = {}
        
        # Find patterns
        if len(results) >= 3:
            themes = self.kernel.discover_themes(
                [text for text, _ in results[:10]], min_cluster_size=2
            )
        else:
            themes = []
        
        return {
            "query": query,
            "results": [
                {"text": text, "similarity": sim}
                for text, sim in results
            ],
            "related_concepts": related_concepts,
            "themes": themes,
            "count": len(results)
        }
    
    def search(self, query: str, corpus: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Simple semantic search"""
        return self.kernel.find_similar(query, corpus, top_k=top_k)


class ReasoningEngine:
    """Logical and causal reasoning"""
    
    def __init__(self, kernel: QuantumKernel):
        self.kernel = kernel
        self.reasoning_patterns = [
            "if A then B",
            "A causes B",
            "A is similar to B",
            "A implies B"
        ]
    
    def reason(self, premises: List[str], question: str) -> Dict:
        """Perform reasoning"""
        # Build relationship graph of premises
        premise_graph = self.kernel.build_relationship_graph(premises)
        
        # Find logical connections
        connections = []
        for premise, related in premise_graph.items():
            for rel_premise, similarity in related:
                if similarity > 0.7:  # Strong connection
                    connections.append({
                        "from": premise,
                        "to": rel_premise,
                        "strength": similarity,
                        "type": "semantic_connection"
                    })
        
        # Compute overall coherence
        coherence = self._compute_coherence(premises, premise_graph)
        
        # Answer question based on premises
        answer_relevance = self.kernel.similarity(
            " ".join(premises), question
        )
        
        return {
            "premises": premises,
            "question": question,
            "connections": connections,
            "coherence": coherence,
            "answer_relevance": answer_relevance,
            "confidence": min(coherence, answer_relevance)
        }
    
    def _compute_coherence(self, premises: List[str], graph: Dict) -> float:
        """Compute how coherent the premises are"""
        if len(premises) < 2:
            return 1.0
        
        similarities = []
        for i, p1 in enumerate(premises):
            for p2 in premises[i+1:]:
                sim = self.kernel.similarity(p1, p2)
                similarities.append(sim)
        
        return float(sum(similarities) / len(similarities)) if similarities else 0.0


class LearningSystem:
    """Continuous learning and adaptation with LLM integration"""
    
    def __init__(self, kernel: QuantumKernel, use_llm: bool = False):
        self.kernel = kernel
        self.learned_patterns = {}
        self.llm = None
        self.example_database = []
        
        if use_llm:
            try:
                from llm.quantum_llm_standalone import StandaloneQuantumLLM
                self.llm = StandaloneQuantumLLM(kernel=kernel)
            except ImportError:
                logger.warning("LLM not available for LearningSystem. Using kernel-only learning.")
    
    def learn_from_examples(self, examples: List[Tuple[str, str]]) -> Dict:
        """Learn patterns from examples with optional LLM integration"""
        # Store examples
        self.example_database.extend(examples)
        input_texts = [ex[0] for ex in examples]
        output_texts = [ex[1] for ex in examples]
        
        # Learn patterns with kernel
        kernel_patterns = self._kernel_learning(examples, input_texts, output_texts)
        
        # If LLM available, also learn generation patterns
        llm_patterns = {}
        if self.llm:
            try:
                # Format examples for LLM
                source_texts = [f"{inp} | {out}" for inp, out in examples]
                self.llm.add_source_texts(source_texts)
                llm_patterns = {
                    'vocab_size': len(self.llm.verified_phrases),
                    'quality': self.llm._estimate_quality(),
                    'phrases_learned': len(self.llm.verified_phrases)
                }
            except Exception as e:
                logger.warning(f"LLM learning failed: {e}")
                llm_patterns = {'error': str(e)}
        
        return {
            'kernel_patterns': kernel_patterns,
            'llm_patterns': llm_patterns,
            'total_examples': len(self.example_database),
            'patterns_learned': kernel_patterns.get('patterns_learned', 0)
        }
    
    def _kernel_learning(self, examples: List[Tuple[str, str]], 
                        input_texts: List[str], output_texts: List[str]) -> Dict:
        """Kernel-based pattern learning"""
        # Discover themes in inputs
        input_themes = self.kernel.discover_themes(input_texts, min_cluster_size=2)
        
        # Discover themes in outputs
        output_themes = self.kernel.discover_themes(output_texts, min_cluster_size=2)
        
        # Map input themes to output themes
        patterns = {}
        for in_theme in input_themes:
            # Find which output themes correspond
            best_match = None
            best_score = 0.0
            
            for out_theme in output_themes:
                # Compute similarity between theme texts
                in_text = " ".join(in_theme["texts"])
                out_text = " ".join(out_theme["texts"])
                score = self.kernel.similarity(in_text, out_text)
                
                if score > best_score:
                    best_score = score
                    best_match = out_theme
            
            if best_match and best_score > 0.6:
                patterns[in_theme["theme"]] = {
                    "output_theme": best_match["theme"],
                    "confidence": best_score
                }
        
        self.learned_patterns.update(patterns)
        
        return {
            "patterns_learned": len(patterns),
            "input_themes": len(input_themes),
            "output_themes": len(output_themes),
            "patterns": patterns
        }
    
    def get_patterns(self) -> Dict:
        """Get learned patterns"""
        return self.learned_patterns
    
    def apply_pattern(self, input_text: str) -> Optional[str]:
        """Apply learned pattern to input"""
        # Find matching pattern
        for pattern_input, pattern_data in self.learned_patterns.items():
            similarity = self.kernel.similarity(input_text, pattern_input)
            if similarity > 0.7:
                return pattern_data.get("output_theme")
        return None


class ConversationalAI:
    """Natural conversation with context and LLM integration"""
    
    def __init__(self, kernel: QuantumKernel, llm=None):
        self.kernel = kernel
        self.llm = llm  # Optional StandaloneQuantumLLM instance
        self.conversation_history = []
        self.responses = {
            "search for information": "I found some relevant information for you.",
            "ask a question": "Let me help answer that question.",
            "request recommendation": "Based on your preferences, I recommend:",
            "create content": "I can help create content on that topic.",
            "analyze data": "Let me analyze that data for you.",
            "find relationships": "I discovered some interesting relationships."
        }
    
    def respond(self, user_message: str, context_documents: List[str] = None) -> str:
        """Generate contextual response using LLM if available"""
        # Build context from conversation history
        context = self._build_context()
        
        # If LLM available, use it for grounded generation
        if self.llm:
            try:
                # Build prompt with context
                prompt = self._build_prompt(user_message, context, context_documents)
                
                # Generate grounded response
                result = self.llm.generate_grounded(
                    prompt=prompt,
                    max_length=100,
                    temperature=0.7,
                    require_validation=False  # More flexible for conversations
                )
                
                response = result.get('generated', self._template_response(user_message, context))
                
                # Update history
                self.conversation_history.append({
                    "user": user_message,
                    "assistant": response,
                    "intent": "conversation",
                    "llm_generated": True,
                    "confidence": result.get('confidence', 0.0)
                })
                
                return response
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}. Falling back to template-based response.")
                # Fallback to template-based
                return self._template_response(user_message, context)
        else:
            # Use template-based response
            return self._template_response(user_message, context)
    
    def _build_context(self) -> List[str]:
        """Build context from conversation history"""
        if not self.conversation_history:
            return []
        
        context = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            user_msg = msg.get("user", "") or msg.get("text", "")
            assistant_msg = msg.get("assistant", "")
            if user_msg:
                context.append(f"User: {user_msg}")
            if assistant_msg:
                context.append(f"Assistant: {assistant_msg}")
        return context
    
    def _build_prompt(self, user_message: str, context: List[str], 
                     context_documents: List[str] = None) -> str:
        """Build prompt for LLM generation"""
        prompt_parts = []
        
        # Add context documents if provided
        if context_documents:
            prompt_parts.append("Context: " + " ".join(context_documents[:3]))
        
        # Add conversation history
        if context:
            prompt_parts.append("Previous conversation: " + " | ".join(context[-4:]))
        
        # Add current message
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Assistant:")
        
        return " ".join(prompt_parts)
    
    def _template_response(self, user_message: str, context: List[str]) -> str:
        """Generate template-based response (fallback)"""
        # Find relevant conversation history
        if self.conversation_history:
            recent_messages = [
                msg.get("user", "") or msg.get("text", "")
                for msg in self.conversation_history[-5:]
            ]
            recent_messages = [m for m in recent_messages if m]
            if recent_messages:
                relevant = self.kernel.find_similar(
                    user_message, recent_messages, top_k=2
                )
            else:
                relevant = []
        else:
            relevant = []
        
        # Understand intent
        understanding = SemanticUnderstandingEngine(self.kernel)
        context_messages = [
            msg.get("user", "") or msg.get("text", "")
            for msg in self.conversation_history[-3:]
        ]
        context_messages = [m for m in context_messages if m]
        intent_result = understanding.understand_intent(
            user_message,
            context_messages
        )
        
        # Generate response
        intent = intent_result["intent"]
        base_response = self.responses.get(intent, "I understand. Let me help with that.")
        
        # Add context if relevant
        if relevant:
            context_note = f" (Related to our previous discussion about '{relevant[0][0][:30]}...')"
            response = base_response + context_note
        else:
            response = base_response
        
        # Update history
        self.conversation_history.append({
            "user": user_message,
            "assistant": response,
            "intent": intent,
            "llm_generated": False
        })
        
        return response
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def add_response_template(self, intent: str, template: str):
        """Add a custom response template"""
        self.responses[intent] = template
        logger.info(f"Added response template for intent: {intent}")
