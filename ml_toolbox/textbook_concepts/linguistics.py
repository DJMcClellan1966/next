"""
Linguistics & Syntactic Structure - Inspired by Noam Chomsky

Implements:
- Syntactic Parsing (simplified)
- Grammar-Based Feature Engineering
- Hierarchical Text Processing
- Recursive Structures
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SimpleSyntacticParser:
    """
    Simplified syntactic parser for feature engineering
    
    Based on context-free grammar concepts
    """
    
    def __init__(self):
        """Initialize parser"""
        # Simple grammar rules (POS tags -> phrases)
        self.grammar_rules = {
            'NP': ['DT', 'NN', 'NNP'],  # Noun phrase
            'VP': ['VB', 'VBD', 'VBG'],  # Verb phrase
            'PP': ['IN', 'NP'],  # Prepositional phrase
            'ADJP': ['JJ', 'JJR', 'JJS'],  # Adjective phrase
        }
        
        # POS tag patterns (simplified)
        self.pos_patterns = {
            'DT': r'\b(the|a|an|this|that|these|those)\b',
            'NN': r'\b\w+ing\b|\b\w+tion\b|\b\w+ness\b',  # Nouns
            'VB': r'\b(is|are|was|were|be|been|being)\b',
            'VBD': r'\b\w+ed\b',
            'JJ': r'\b\w+ful\b|\b\w+less\b',  # Adjectives
            'IN': r'\b(in|on|at|by|for|with|from|to)\b',
        }
    
    def extract_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract POS tags from text (simplified)
        
        Args:
            text: Input text
        
        Returns:
            List of (word, pos_tag) tuples
        """
        words = text.lower().split()
        pos_tags = []
        
        for word in words:
            tag = 'UNK'  # Unknown
            for pos, pattern in self.pos_patterns.items():
                if re.search(pattern, word):
                    tag = pos
                    break
            pos_tags.append((word, tag))
        
        return pos_tags
    
    def extract_phrases(self, text: str) -> Dict[str, List[str]]:
        """
        Extract syntactic phrases from text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary of phrase types and phrases
        """
        pos_tags = self.extract_pos_tags(text)
        phrases = defaultdict(list)
        
        # Extract noun phrases
        current_np = []
        for word, tag in pos_tags:
            if tag in ['DT', 'NN', 'NNP']:
                current_np.append(word)
            else:
                if len(current_np) > 0:
                    phrases['NP'].append(' '.join(current_np))
                    current_np = []
        
        if len(current_np) > 0:
            phrases['NP'].append(' '.join(current_np))
        
        # Extract verb phrases
        current_vp = []
        for word, tag in pos_tags:
            if tag in ['VB', 'VBD', 'VBG']:
                current_vp.append(word)
            else:
                if len(current_vp) > 0:
                    phrases['VP'].append(' '.join(current_vp))
                    current_vp = []
        
        if len(current_vp) > 0:
            phrases['VP'].append(' '.join(current_vp))
        
        return dict(phrases)
    
    def calculate_syntactic_features(self, text: str) -> Dict[str, float]:
        """
        Calculate syntactic features for ML
        
        Args:
            text: Input text
        
        Returns:
            Dictionary of syntactic features
        """
        pos_tags = self.extract_pos_tags(text)
        phrases = self.extract_phrases(text)
        
        features = {
            'num_noun_phrases': len(phrases.get('NP', [])),
            'num_verb_phrases': len(phrases.get('VP', [])),
            'avg_phrase_length': np.mean([len(p.split()) for p in phrases.get('NP', []) + phrases.get('VP', [])]) if phrases else 0,
            'noun_verb_ratio': len(phrases.get('NP', [])) / (len(phrases.get('VP', [])) + 1),
            'pos_diversity': len(set(tag for _, tag in pos_tags)) / (len(pos_tags) + 1),
            'avg_sentence_length': len(text.split()) / (text.count('.') + text.count('!') + text.count('?') + 1),
        }
        
        return features


class GrammarBasedFeatureExtractor:
    """
    Extract features based on grammatical structure
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        self.parser = SimpleSyntacticParser()
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract grammar-based features from texts
        
        Args:
            texts: List of input texts
        
        Returns:
            Feature matrix
        """
        features = []
        
        for text in texts:
            syntactic_features = self.parser.calculate_syntactic_features(text)
            feature_vector = list(syntactic_features.values())
            features.append(feature_vector)
        
        return np.array(features)


class HierarchicalTextProcessor:
    """
    Process text hierarchically (word -> phrase -> sentence -> document)
    """
    
    def __init__(self, embedding_dim: int = 100):
        """
        Initialize hierarchical processor
        
        Args:
            embedding_dim: Dimension for embeddings
        """
        self.embedding_dim = embedding_dim
    
    def word_embeddings(self, words: List[str]) -> np.ndarray:
        """Simple word embeddings (character-based)"""
        embeddings = []
        for word in words:
            # Simple character-based embedding
            embedding = np.zeros(self.embedding_dim)
            for i, char in enumerate(word[:self.embedding_dim]):
                embedding[i] = ord(char) / 128.0
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def phrase_embedding(self, phrase_words: List[str]) -> np.ndarray:
        """Aggregate word embeddings to phrase embedding"""
        word_embeds = self.word_embeddings(phrase_words)
        return np.mean(word_embeds, axis=0)
    
    def sentence_embedding(self, phrases: List[List[str]]) -> np.ndarray:
        """Aggregate phrase embeddings to sentence embedding"""
        phrase_embeds = [self.phrase_embedding(phrase) for phrase in phrases]
        return np.mean(phrase_embeds, axis=0)
    
    def document_embedding(self, sentences: List[List[List[str]]]) -> np.ndarray:
        """Aggregate sentence embeddings to document embedding"""
        sentence_embeds = [self.sentence_embedding(sent) for sent in sentences]
        return np.mean(sentence_embeds, axis=0)
    
    def process_hierarchically(self, text: str) -> Dict[str, np.ndarray]:
        """
        Process text at all hierarchical levels
        
        Args:
            text: Input text
        
        Returns:
            Embeddings at each level
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Split sentences into phrases (simplified: by commas)
        sentence_phrases = [s.split(',') for s in sentences]
        
        # Split phrases into words
        phrase_words = [[p.split() for p in sent] for sent in sentence_phrases]
        
        # Get embeddings at each level
        all_words = [word for sent in phrase_words for phrase in sent for word in phrase]
        word_embeds = self.word_embeddings(all_words)
        
        phrase_embeds = [self.phrase_embedding(phrase) for sent in phrase_words for phrase in sent]
        
        sentence_embeds = [self.sentence_embedding(sent) for sent in phrase_words]
        
        doc_embed = self.document_embedding(phrase_words)
        
        return {
            'word_embeddings': word_embeds,
            'phrase_embeddings': np.array(phrase_embeds),
            'sentence_embeddings': np.array(sentence_embeds),
            'document_embedding': doc_embed
        }
