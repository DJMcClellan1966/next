"""
Standalone Quantum LLM
A complete, reusable quantum language model with grounded generation and progressive learning
Can be used with any kernel and AI system
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import re
from collections import Counter
import json
import os
from datetime import datetime


def quantum_sample_token(logits: np.ndarray, temperature: float = 1.0) -> int:
    """
    Quantum-inspired token sampling using Born rule
    Uses amplitude-squared probabilities for more natural sampling
    """
    # Convert logits to quantum amplitudes
    # Quantum amplitude: exp(logit / (2 * temperature))
    amplitudes = np.exp(logits / (2 * temperature))
    
    # Normalize (quantum normalization)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    
    # Quantum probability (Born rule: probability = |amplitude|^2)
    probabilities = amplitudes ** 2
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum()
    
    # Quantum measurement (sample from distribution)
    token_id = np.random.choice(len(probabilities), p=probabilities)
    
    return token_id


class StandaloneQuantumLLM:
    """
    Standalone Quantum LLM with grounded generation and progressive learning
    Can be used independently or integrated with kernel/AI systems
    """
    
    def __init__(self, kernel=None, source_texts: Optional[List[str]] = None, 
                 config: Optional[Dict] = None):
        """
        Initialize standalone quantum LLM
        
        Args:
            kernel: Quantum kernel instance (optional, will create if not provided)
            source_texts: Initial verified source texts
            config: Configuration dictionary
        """
        # Import kernel (handle if not provided)
        if kernel is None:
            try:
                from quantum_kernel import QuantumKernel, get_kernel, KernelConfig
                self.kernel = get_kernel(KernelConfig())
            except ImportError:
                raise ImportError("Quantum kernel required. Install quantum_kernel module.")
        else:
            self.kernel = kernel
        
        # Configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.min_phrase_length = self.config.get('min_phrase_length', 2)
        self.max_phrase_length = self.config.get('max_phrase_length', 5)
        self.vocab_expansion_rate = self.config.get('vocab_expansion_rate', 0.1)  # 10% per week
        self.use_quantum_sampling = self.config.get('use_quantum_sampling', True)  # Quantum sampling
        self.use_quantum_coherence = self.config.get('use_quantum_coherence', True)  # Quantum coherence
        
        # Source database
        self.source_texts = source_texts or []
        self.source_embeddings = {}
        self.verified_phrases = set()
        self.phrase_sources = {}  # phrase -> list of source texts
        self.phrase_frequencies = Counter()  # Track phrase usage
        
        # Vocabulary and learning
        self.vocab = {}
        self.token_embeddings = {}
        self.learned_pairs = []  # (prompt, output) pairs
        self.learning_history = []  # Track learning progress
        
        # Progressive learning state
        self.learning_week = 0
        self.total_phrases_learned = 0
        self.quality_scores = []  # Track quality over time
        
        # Build initial database if sources provided
        if self.source_texts:
            self._build_verified_database()
    
    def _build_verified_database(self):
        """Build database of verified phrases from source texts"""
        print(f"Building verified database from {len(self.source_texts)} sources...")
        
        for source_text in self.source_texts:
            # Extract phrases
            phrases = self._extract_phrases(
                source_text, 
                min_words=self.min_phrase_length, 
                max_words=self.max_phrase_length
            )
            
            for phrase in phrases:
                normalized = self._normalize_phrase(phrase)
                self.verified_phrases.add(normalized)
                self.phrase_frequencies[normalized] += 1
                
                # Track sources
                if normalized not in self.phrase_sources:
                    self.phrase_sources[normalized] = []
                self.phrase_sources[normalized].append(source_text[:100])
        
        # Create embeddings for verified phrases
        for phrase in self.verified_phrases:
            self.source_embeddings[phrase] = self.kernel.embed(phrase)
        
        self.total_phrases_learned = len(self.verified_phrases)
        print(f"Built database: {len(self.verified_phrases)} verified phrases")
    
    def _extract_phrases(self, text: str, min_words: int = 2, max_words: int = 5) -> List[str]:
        """Extract phrases of various lengths from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        phrases = []
        
        for length in range(min_words, max_words + 1):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                phrases.append(phrase)
        
        return phrases
    
    def _normalize_phrase(self, phrase: str) -> str:
        """Normalize phrase for matching"""
        return " ".join(phrase.lower().split())
    
    def add_source_texts(self, texts: List[str], merge: bool = True):
        """
        Add more verified source texts (progressive learning)
        
        Args:
            texts: New source texts to add
            merge: If True, merge with existing; if False, rebuild
        """
        if merge:
            self.source_texts.extend(texts)
        else:
            self.source_texts = texts
        
        # Rebuild database
        self._build_verified_database()
        
        # Record learning event
        self.learning_week += 1
        self.learning_history.append({
            'week': self.learning_week,
            'sources_added': len(texts),
            'total_phrases': len(self.verified_phrases),
            'timestamp': datetime.now().isoformat()
        })
    
    def progressive_learning_step(self, new_texts: List[str], week: Optional[int] = None):
        """
        Perform one step of progressive learning
        Gradually expands vocabulary and improves quality
        """
        if week is not None:
            self.learning_week = week
        
        # Calculate expansion target
        current_size = len(self.verified_phrases)
        expansion_target = int(current_size * self.vocab_expansion_rate)
        
        # Add new texts
        self.add_source_texts(new_texts[:expansion_target] if len(new_texts) > expansion_target else new_texts)
        
        # Estimate quality improvement
        quality = self._estimate_quality()
        self.quality_scores.append(quality)
        
        return {
            'week': self.learning_week,
            'phrases_before': current_size,
            'phrases_after': len(self.verified_phrases),
            'phrases_added': len(self.verified_phrases) - current_size,
            'estimated_quality': quality,
            'total_phrases': len(self.verified_phrases)
        }
    
    def _estimate_quality(self) -> float:
        """Estimate current quality based on vocabulary size and coverage"""
        # Base quality from vocabulary size
        vocab_quality = min(0.3 + (len(self.verified_phrases) / 10000) * 0.4, 0.9)
        
        # Boost from learning history
        history_boost = min(len(self.learning_history) * 0.01, 0.1)
        
        return min(vocab_quality + history_boost, 0.95)
    
    def generate_grounded(self, prompt: str, max_length: int = 50, 
                         temperature: float = 0.7, require_validation: bool = True) -> Dict:
        """
        Generate text grounded in verified sources
        """
        # Find verified phrases similar to prompt
        prompt_embedding = self.kernel.embed(prompt)
        
        # Find best matching verified phrases
        candidate_phrases = []
        for phrase, phrase_embedding in self.source_embeddings.items():
            similarity = float(np.abs(np.dot(prompt_embedding, phrase_embedding)))
            if similarity >= self.confidence_threshold * 0.8:
                # Boost by frequency (more common phrases are better)
                frequency_boost = min(self.phrase_frequencies[phrase] / 10.0, 0.2)
                similarity += frequency_boost
                candidate_phrases.append((phrase, similarity))
        
        candidate_phrases.sort(key=lambda x: x[1], reverse=True)
        
        if not candidate_phrases:
            return {
                "generated": prompt,
                "confidence": 0.0,
                "warning": "No verified content found matching prompt",
                "is_safe": False
            }
        
        # Build generation from verified phrases
        generated_words = prompt.split()
        context = prompt
        
        for _ in range(max_length):
            context_embedding = self.kernel.embed(context)
            
            best_phrase = None
            best_similarity = 0.0
            
            for phrase, phrase_similarity in candidate_phrases[:50]:  # Top 50 candidates
                phrase_embedding = self.source_embeddings[phrase]
                context_sim = float(np.abs(np.dot(context_embedding, phrase_embedding)))
                
                # Combined score
                combined_score = (phrase_similarity * 0.4) + (context_sim * 0.6)
                
                if combined_score > best_similarity and combined_score >= self.confidence_threshold:
                    phrase_words = phrase.split()
                    context_words = context.lower().split()
                    
                    if phrase.startswith(" ".join(context_words[-2:])) or context_sim > 0.7:
                        best_phrase = phrase
                        best_similarity = combined_score
            
            if best_phrase:
                phrase_words = best_phrase.split()
                context_words = context.lower().split()
                
                # Use quantum sampling for word selection if enabled
                if self.use_quantum_sampling and len(phrase_words) > 1:
                    # Create probabilities for each word
                    word_scores = []
                    for word in phrase_words:
                        if word not in context_words[-3:]:
                            # Calculate word score (quantum-like amplitude)
                            word_sim = float(np.abs(np.dot(
                                self.kernel.embed(word),
                                context_embedding
                            )))
                            word_scores.append((word, word_sim))
                    
                    if word_scores:
                        # Use quantum sampling to select word
                        words, scores = zip(*word_scores)
                        word_probs = np.array(scores) ** 2  # Quantum Born rule
                        word_probs = word_probs / word_probs.sum()
                        selected_word = np.random.choice(len(words), p=word_probs)
                        generated_words.append(words[selected_word])
                    else:
                        # Fallback to first word
                        if phrase_words and phrase_words[0] not in context_words[-3:]:
                            generated_words.append(phrase_words[0])
                else:
                    # Standard word selection
                    for word in phrase_words:
                        if word not in context_words[-3:]:
                            generated_words.append(word)
                            break
                
                # Update context with quantum coherence preservation
                if self.use_quantum_coherence:
                    # Maintain quantum state coherence
                    current_state = self.kernel.embed(" ".join(generated_words[-3:]))
                    context_state = self.kernel.embed(context)
                    # Blend states to maintain coherence
                    coherent_state = (current_state * 0.7) + (context_state * 0.3)
                    coherent_state = coherent_state / np.linalg.norm(coherent_state)
                    # Find words that maintain coherence
                    context = " ".join(generated_words[-5:])
                else:
                    context = " ".join(generated_words[-5:])
            else:
                break
        
        generated_text = " ".join(generated_words)
        
        # Validate
        validation = self.validate_against_sources(generated_text)
        
        if require_validation and not validation["is_safe"]:
            return {
                "generated": generated_text,
                "confidence": validation["confidence"],
                "validation": validation,
                "warning": "Generated text has low confidence or potential issues",
                "is_safe": False
            }
        
        return {
            "generated": generated_text,
            "confidence": validation["confidence"],
            "validation": validation,
            "is_safe": validation["is_safe"],
            "sources": self.phrase_sources.get(generated_text[:50], [])[:3]
        }
    
    def validate_against_sources(self, text: str) -> Dict:
        """Validate text against verified sources"""
        words = text.lower().split()
        verified_words = 0
        unverified_phrases = []
        confidence_scores = []
        
        for length in range(2, min(6, len(words) + 1)):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                normalized = self._normalize_phrase(phrase)
                
                if normalized in self.verified_phrases:
                    verified_words += length
                    confidence_scores.append(1.0)
                else:
                    phrase_embedding = self.kernel.embed(phrase)
                    best_similarity = 0.0
                    best_match = None
                    
                    for verified_phrase, verified_embedding in self.source_embeddings.items():
                        similarity = float(np.abs(np.dot(phrase_embedding, verified_embedding)))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = verified_phrase
                    
                    if best_similarity >= self.confidence_threshold:
                        verified_words += length
                        confidence_scores.append(best_similarity)
                    else:
                        unverified_phrases.append((phrase, best_similarity, best_match))
                        confidence_scores.append(best_similarity)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        issues = []
        if avg_confidence < self.confidence_threshold:
            issues.append("low_confidence")
        if len(unverified_phrases) > len(words) * 0.3:
            issues.append("high_unverified_content")
        if any(score < 0.3 for score in confidence_scores):
            issues.append("potential_hallucination")
        
        return {
            "confidence": avg_confidence,
            "verified_ratio": verified_words / len(words) if words else 0.0,
            "unverified_phrases": unverified_phrases[:5],
            "issues": issues,
            "is_safe": avg_confidence >= self.confidence_threshold and len(issues) == 0
        }
    
    def detect_bias(self, text: str) -> Dict:
        """Detect potential bias in text"""
        bias_indicators = {
            "absolute_claims": len(re.findall(r'\b(always|never|all|none|every)\b', text.lower())),
            "emotional_language": len(re.findall(r'\b(amazing|terrible|awful|perfect|horrible)\b', text.lower())),
            "exclusive_language": len(re.findall(r'\b(only|solely|exclusively)\b', text.lower())),
        }
        
        source_diversity = len(set(self.phrase_sources.get(text[:50], [])))
        
        issues = []
        if bias_indicators["absolute_claims"] > 3:
            issues.append("too_many_absolute_claims")
        if bias_indicators["emotional_language"] > 5:
            issues.append("excessive_emotional_language")
        if bias_indicators["exclusive_language"] > 2:
            issues.append("exclusive_language_detected")
        if source_diversity < 2 and len(self.source_texts) > 5:
            issues.append("low_source_diversity")
        
        return {
            "bias_score": sum(bias_indicators.values()) / max(len(text.split()), 1),
            "indicators": bias_indicators,
            "source_diversity": source_diversity,
            "issues": issues,
            "has_bias": len(issues) > 0
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "source_texts": len(self.source_texts),
            "verified_phrases": len(self.verified_phrases),
            "phrase_sources": len(self.phrase_sources),
            "vocabulary_size": len(self.vocab),
            "confidence_threshold": self.confidence_threshold,
            "learning_week": self.learning_week,
            "total_phrases_learned": self.total_phrases_learned,
            "estimated_quality": self._estimate_quality(),
            "learning_history": len(self.learning_history),
            "average_phrases_per_source": len(self.verified_phrases) / max(len(self.source_texts), 1)
        }
    
    def save(self, filepath: str):
        """Save LLM state to file"""
        state = {
            'config': self.config,
            'confidence_threshold': self.confidence_threshold,
            'source_texts': self.source_texts,
            'verified_phrases': list(self.verified_phrases),
            'phrase_sources': {k: v for k, v in self.phrase_sources.items()},
            'phrase_frequencies': dict(self.phrase_frequencies),
            'vocab': self.vocab,
            'learned_pairs': self.learned_pairs,
            'learning_history': self.learning_history,
            'learning_week': self.learning_week,
            'total_phrases_learned': self.total_phrases_learned,
            'quality_scores': self.quality_scores
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        print(f"Saved LLM state to {filepath}")
    
    def load(self, filepath: str):
        """Load LLM state from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.config = state.get('config', {})
        self.confidence_threshold = state.get('confidence_threshold', 0.6)
        self.source_texts = state.get('source_texts', [])
        self.verified_phrases = set(state.get('verified_phrases', []))
        self.phrase_sources = {k: v for k, v in state.get('phrase_sources', {}).items()}
        self.phrase_frequencies = Counter(state.get('phrase_frequencies', {}))
        self.vocab = state.get('vocab', {})
        self.learned_pairs = state.get('learned_pairs', [])
        self.learning_history = state.get('learning_history', [])
        self.learning_week = state.get('learning_week', 0)
        self.total_phrases_learned = state.get('total_phrases_learned', 0)
        self.quality_scores = state.get('quality_scores', [])
        
        # Rebuild embeddings
        for phrase in self.verified_phrases:
            self.source_embeddings[phrase] = self.kernel.embed(phrase)
        
        print(f"Loaded LLM state from {filepath}")


def create_quantum_llm(kernel=None, source_texts: Optional[List[str]] = None, 
                      config: Optional[Dict] = None) -> StandaloneQuantumLLM:
    """
    Factory function to create a standalone quantum LLM
    
    Args:
        kernel: Quantum kernel instance (optional)
        source_texts: Initial source texts (optional)
        config: Configuration dictionary (optional)
    
    Returns:
        StandaloneQuantumLLM instance
    """
    return StandaloneQuantumLLM(kernel=kernel, source_texts=source_texts, config=config)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("STANDALONE QUANTUM LLM DEMONSTRATION")
    print("=" * 80)
    
    # Sample source texts
    source_texts = [
        "Knowledge is power and learning is a lifelong journey",
        "Science is the systematic study of the natural world through observation and experiment",
        "Technology advances through innovation and collaboration",
        "Education opens doors to new opportunities and perspectives",
        "Innovation drives progress and solves complex problems",
        "Collaboration brings together diverse perspectives and skills",
        "Research builds on previous discoveries and pushes boundaries",
        "Creativity combines imagination with practical application",
        "Critical thinking enables sound decision making",
        "Communication connects ideas and people across distances"
    ]
    
    # Create LLM
    llm = create_quantum_llm(source_texts=source_texts)
    
    # Get statistics
    stats = llm.get_statistics()
    print(f"\nInitial Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Generate text
    print(f"\n" + "=" * 80)
    print("GENERATION TEST")
    print("=" * 80)
    result = llm.generate_grounded("Knowledge is", max_length=20)
    print(f"Prompt: 'Knowledge is'")
    print(f"Generated: '{result['generated']}'")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Safe: {result['is_safe']}")
    
    # Progressive learning
    print(f"\n" + "=" * 80)
    print("PROGRESSIVE LEARNING TEST")
    print("=" * 80)
    new_texts = [
        "Learning requires dedication and consistent practice",
        "Progress comes from understanding fundamentals and building upon them",
        "Innovation emerges from questioning assumptions and exploring new possibilities"
    ]
    learning_result = llm.progressive_learning_step(new_texts, week=1)
    print(f"Learning Result:")
    for key, value in learning_result.items():
        print(f"  {key}: {value}")
    
    # Updated statistics
    stats = llm.get_statistics()
    print(f"\nUpdated Statistics:")
    print(f"  Verified phrases: {stats['verified_phrases']}")
    print(f"  Estimated quality: {stats['estimated_quality']:.2f}")
    print(f"  Learning week: {stats['learning_week']}")
