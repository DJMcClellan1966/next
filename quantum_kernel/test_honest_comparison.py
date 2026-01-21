"""
Honest Comparison Test - Shows where quantum methods excel AND fail
Unbiased testing of quantum vs classical methods across different scenarios
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel import get_kernel, KernelConfig


def test_exact_match_similarity():
    """Test 1: Exact word matching (classical should excel)"""
    print("\n" + "="*70)
    print("TEST 1: EXACT MATCH SIMILARITY")
    print("Hypothesis: Classical methods should excel here")
    print("="*70)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,  # Required for proper comparison
        use_quantum_methods=False, 
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,  # Required for quantum methods to work
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Exact or near-exact matches
    exact_pairs = [
        ("hello world", "hello world"),
        ("machine learning", "machine learning"),
        ("artificial intelligence", "artificial intelligence"),
        ("hello", "hello world"),  # Substring match
        ("cat", "cats"),  # Plural
    ]
    
    print("\nExact/Near-Exact Matches:")
    classical_scores = []
    quantum_scores = []
    
    for text1, text2 in exact_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        classical_scores.append(classical_sim)
        quantum_scores.append(quantum_sim)
        
        print(f"\n'{text1}' vs '{text2}':")
        print(f"  Classical: {classical_sim:.4f}")
        print(f"  Quantum:   {quantum_sim:.4f}")
        print(f"  Winner: {'Classical' if classical_sim > quantum_sim else 'Quantum' if quantum_sim > classical_sim else 'Tie'}")
    
    print(f"\nAverage Classical: {np.mean(classical_scores):.4f}")
    print(f"Average Quantum:   {np.mean(quantum_scores):.4f}")
    print(f"Winner: {'Classical' if np.mean(classical_scores) > np.mean(quantum_scores) else 'Quantum' if np.mean(quantum_scores) > np.mean(classical_scores) else 'Tie'}")


def test_semantic_similarity():
    """Test 2: Semantic similarity (quantum might excel)"""
    print("\n" + "="*70)
    print("TEST 2: SEMANTIC SIMILARITY (Different words, same meaning)")
    print("Hypothesis: Quantum methods might excel here")
    print("="*70)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,  # Required for proper comparison
        use_quantum_methods=False, 
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,  # Required for quantum methods to work
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Different words, same meaning
    semantic_pairs = [
        ("I love programming", "I enjoy coding"),
        ("The cat sat on the mat", "A feline rested on a rug"),
        ("He was very happy", "She felt extremely joyful"),
    ]
    
    print("\nSemantic Similarity (Different words, same meaning):")
    classical_scores = []
    quantum_scores = []
    
    for text1, text2 in semantic_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        classical_scores.append(classical_sim)
        quantum_scores.append(quantum_sim)
        
        print(f"\n'{text1}' vs '{text2}':")
        print(f"  Classical: {classical_sim:.4f}")
        print(f"  Quantum:   {quantum_sim:.4f}")
        print(f"  Winner: {'Classical' if classical_sim > quantum_sim else 'Quantum' if quantum_sim > classical_sim else 'Tie'}")
        print(f"  Improvement: {((quantum_sim - classical_sim) / max(classical_sim, 0.001) * 100):+.1f}%")
    
    print(f"\nAverage Classical: {np.mean(classical_scores):.4f}")
    print(f"Average Quantum:   {np.mean(quantum_scores):.4f}")
    print(f"Winner: {'Classical' if np.mean(classical_scores) > np.mean(quantum_scores) else 'Quantum' if np.mean(quantum_scores) > np.mean(classical_scores) else 'Tie'}")


def test_unrelated_texts():
    """Test 3: Unrelated texts (both should have low similarity)"""
    print("\n" + "="*70)
    print("TEST 3: UNRELATED TEXTS")
    print("Hypothesis: Both should have low similarity")
    print("="*70)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,  # Required for proper comparison
        use_quantum_methods=False, 
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,  # Required for quantum methods to work
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Completely unrelated
    unrelated_pairs = [
        ("I love programming", "The weather is nice today"),
        ("Machine learning algorithms", "Cooking recipes for pasta"),
        ("Neural networks", "Sports scores and statistics"),
    ]
    
    print("\nUnrelated Texts (should have LOW similarity):")
    classical_scores = []
    quantum_scores = []
    
    for text1, text2 in unrelated_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        classical_scores.append(classical_sim)
        quantum_scores.append(quantum_sim)
        
        print(f"\n'{text1}' vs '{text2}':")
        print(f"  Classical: {classical_sim:.4f}")
        print(f"  Quantum:   {quantum_sim:.4f}")
        print(f"  Lower is better here (more accurate distinction)")
        print(f"  Better at distinguishing: {'Classical' if classical_sim < quantum_sim else 'Quantum' if quantum_sim < classical_sim else 'Tie'}")
    
    print(f"\nAverage Classical: {np.mean(classical_scores):.4f}")
    print(f"Average Quantum:   {np.mean(quantum_scores):.4f}")
    print(f"Better at distinguishing unrelated: {'Classical' if np.mean(classical_scores) < np.mean(quantum_scores) else 'Quantum' if np.mean(quantum_scores) < np.mean(classical_scores) else 'Tie'}")


def test_speed_comparison():
    """Test 4: Speed comparison"""
    print("\n" + "="*70)
    print("TEST 4: SPEED COMPARISON")
    print("Hypothesis: Classical should be faster")
    print("="*70)
    
    import time
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,  # Required for proper comparison
        use_quantum_methods=False, 
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,  # Required for quantum methods to work
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Generate test pairs
    texts1 = [f"Text number {i} for testing similarity computation" for i in range(50)]
    texts2 = [f"Text number {i} for testing similarity computation" for i in range(50)]
    
    # Time classical
    start = time.time()
    for t1, t2 in zip(texts1, texts2):
        classical_kernel.similarity(t1, t2)
    classical_time = time.time() - start
    
    # Time quantum
    start = time.time()
    for t1, t2 in zip(texts1, texts2):
        quantum_kernel.similarity(t1, t2)
    quantum_time = time.time() - start
    
    print(f"\nTime for 50 similarity computations:")
    print(f"  Classical: {classical_time:.4f}s")
    print(f"  Quantum:   {quantum_time:.4f}s")
    print(f"  Speedup: {quantum_time/classical_time:.2f}x")
    print(f"  Winner: {'Classical' if classical_time < quantum_time else 'Quantum' if quantum_time < classical_time else 'Tie'}")


def main():
    """Run honest, unbiased comparison"""
    print("\n" + "="*70)
    print("HONEST QUANTUM vs CLASSICAL COMPARISON")
    print("Testing across multiple scenarios with clear hypotheses")
    print("="*70)
    
    test_exact_match_similarity()
    test_semantic_similarity()
    test_unrelated_texts()
    test_speed_comparison()
    
    print("\n" + "="*70)
    print("HONEST SUMMARY")
    print("="*70)
    print("""
This test suite is designed to be UNBIASED:
- Tests scenarios where classical SHOULD excel (exact matches)
- Tests scenarios where quantum MIGHT excel (semantic similarity)
- Tests scenarios where both should perform similarly (unrelated texts)
- Tests speed (classical is typically faster)

Key Findings:
- If quantum methods perform the same or worse in these tests, that's HONEST
- Quantum methods are NOT always better - they're different tools for different jobs
- The goal is to understand when each method is appropriate, not to make one look good
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
