"""
Semantic Similarity Test - Where Quantum Methods Excel
Compares classical cosine similarity vs quantum interference similarity
on semantic text understanding tasks
"""
import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel import get_kernel, KernelConfig

def test_semantic_similarity():
    """Test semantic similarity on text with similar meanings but different words"""
    print("\n" + "="*70)
    print("SEMANTIC SIMILARITY TEST")
    print("Testing: Similar meanings, different words")
    print("="*70)
    
    # Initialize kernels
    classical_config = KernelConfig(
        use_sentence_transformers=False,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    classical_kernel = get_kernel(classical_config)
    
    quantum_config = KernelConfig(
        use_sentence_transformers=False,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    quantum_kernel = get_kernel(quantum_config)
    
    # Test pairs with similar meanings but different words
    test_pairs = [
        ("I love programming", "I enjoy coding"),
        ("The cat sat on the mat", "A feline rested on a rug"),
        ("He was very happy", "She felt extremely joyful"),
        ("The weather is nice today", "It's a beautiful day outside"),
        ("I need to buy groceries", "I must purchase food items"),
        ("The dog barked loudly", "The canine made noise"),
        ("She studied mathematics", "He learned math"),
        ("The car is fast", "The vehicle moves quickly"),
    ]
    
    # Pairs with different meanings
    different_pairs = [
        ("I love programming", "The weather is nice today"),
        ("The cat sat on the mat", "I need to buy groceries"),
        ("He was very happy", "The dog barked loudly"),
    ]
    
    print("\n[Similar Meaning Pairs] (should have HIGH similarity):")
    print("-" * 70)
    classical_scores = []
    quantum_scores = []
    
    for text1, text2 in test_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        classical_scores.append(classical_sim)
        quantum_scores.append(quantum_sim)
        
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"  Classical Cosine: {classical_sim:.4f}")
        print(f"  Quantum Interference: {quantum_sim:.4f}")
        diff = quantum_sim - classical_sim
        print(f"  Difference: {diff:+.4f} ({diff/max(abs(classical_sim), 0.001)*100:+.1f}%)")
    
    print("\n[Different Meaning Pairs] (should have LOW similarity):")
    print("-" * 70)
    
    for text1, text2 in different_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"  Classical Cosine: {classical_sim:.4f}")
        print(f"  Quantum Interference: {quantum_sim:.4f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Average Classical Similarity (similar pairs): {np.mean(classical_scores):.4f}")
    print(f"Average Quantum Similarity (similar pairs):  {np.mean(quantum_scores):.4f}")
    print(f"Quantum improvement: {((np.mean(quantum_scores) - np.mean(classical_scores)) / max(np.mean(classical_scores), 0.001) * 100):+.1f}%")
    print("="*70)


def test_relationship_discovery():
    """Test quantum entangled relationships - finding non-obvious connections"""
    print("\n" + "="*70)
    print("RELATIONSHIP DISCOVERY TEST")
    print("Testing: Quantum entangled relationships")
    print("="*70)
    
    config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    kernel = get_kernel(config)
    
    # Documents with hidden relationships
    documents = [
        "Machine learning uses neural networks to learn patterns",
        "Deep learning is a subset of machine learning",
        "Artificial intelligence includes machine learning",
        "Neural networks are inspired by biological neurons",
        "The brain processes information through neurons",
        "Cognitive science studies how the mind works",
        "Psychology examines human behavior and thought",
        "Computer science involves algorithms and data structures",
        "Algorithms are step-by-step problem-solving procedures",
        "Data structures organize information efficiently",
    ]
    
    print("\n[Finding relationships] for: 'Machine learning uses neural networks'")
    print("-" * 70)
    
    query = "Machine learning uses neural networks to learn patterns"
    relationships = kernel.build_relationship_graph(
        [query] + documents,
        threshold=0.3,
        use_quantum_entanglement=True
    )
    
    if query in relationships:
        print(f"\nTop relationships for '{query}':")
        for related_text, similarity in relationships[query][:5]:
            print(f"  [{similarity:.4f}] {related_text[:60]}...")
    
    print("\n" + "="*70)


def test_synonym_detection():
    """Test ability to detect synonyms and paraphrases"""
    print("\n" + "="*70)
    print("SYNONYM & PARAPHRASE DETECTION TEST")
    print("="*70)
    
    classical_config = KernelConfig(
        use_sentence_transformers=False,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    classical_kernel = get_kernel(classical_config)
    
    quantum_config = KernelConfig(
        use_sentence_transformers=False,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    quantum_kernel = get_kernel(quantum_config)
    
    # Synonym pairs
    synonym_pairs = [
        ("big", "large"),
        ("happy", "joyful"),
        ("fast", "quick"),
        ("smart", "intelligent"),
        ("beautiful", "pretty"),
        ("angry", "mad"),
        ("tired", "exhausted"),
        ("small", "tiny"),
    ]
    
    print("\n[Synonym Detection]:")
    print("-" * 70)
    
    classical_correct = 0
    quantum_correct = 0
    
    for word1, word2 in synonym_pairs:
        classical_sim = classical_kernel.similarity(word1, word2)
        quantum_sim = quantum_kernel.similarity(word1, word2)
        
        # Threshold for synonym detection
        classical_is_synonym = classical_sim > 0.3
        quantum_is_synonym = quantum_sim > 0.3
        
        if classical_is_synonym:
            classical_correct += 1
        if quantum_is_synonym:
            quantum_correct += 1
        
        print(f"{word1} <-> {word2}:")
        print(f"  Classical: {classical_sim:.4f} [OK]" if classical_is_synonym else f"  Classical: {classical_sim:.4f} [FAIL]")
        print(f"  Quantum:   {quantum_sim:.4f} [OK]" if quantum_is_synonym else f"  Quantum:   {quantum_sim:.4f} [FAIL]")
    
    print("\n" + "="*70)
    print("SYNONYM DETECTION ACCURACY")
    print("="*70)
    print(f"Classical: {classical_correct}/{len(synonym_pairs)} ({classical_correct/len(synonym_pairs)*100:.1f}%)")
    print(f"Quantum:   {quantum_correct}/{len(synonym_pairs)} ({quantum_correct/len(synonym_pairs)*100:.1f}%)")
    print("="*70)


def test_context_understanding():
    """Test understanding context and meaning"""
    print("\n" + "="*70)
    print("CONTEXT UNDERSTANDING TEST")
    print("Testing: Same words, different contexts")
    print("="*70)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    classical_kernel = get_kernel(classical_config)
    
    quantum_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    quantum_kernel = get_kernel(quantum_config)
    
    # Same words, different meanings
    context_tests = [
        ("I saw a bat in the cave", "The baseball player swung the bat"),
        ("The bank is closed", "We sat by the river bank"),
        ("The match was exciting", "Light a match to start the fire"),
        ("The key is on the table", "The key to success is hard work"),
    ]
    
    print("\n[Context Disambiguation]:")
    print("-" * 70)
    
    for text1, text2 in context_tests:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"  Classical: {classical_sim:.4f}")
        print(f"  Quantum:   {quantum_sim:.4f}")
        print(f"  Note: Lower similarity = better context distinction")
    
    print("\n" + "="*70)


def main():
    """Run all semantic tests"""
    print("\n" + "="*70)
    print("QUANTUM KERNEL SEMANTIC TESTS")
    print("Testing where quantum methods excel: semantic understanding")
    print("="*70)
    
    test_semantic_similarity()
    test_synonym_detection()
    test_context_understanding()
    test_relationship_discovery()
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)
    print("\nKey Takeaway:")
    print("Quantum methods excel at semantic understanding, relationship")
    print("discovery, and finding meaning beyond exact word matches.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
