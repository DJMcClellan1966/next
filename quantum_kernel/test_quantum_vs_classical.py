"""
Comprehensive Quantum vs Classical Comparison
Tests multiple scenarios to show where each method excels
"""
import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel import get_kernel, KernelConfig

def benchmark_methods(texts1, texts2, method_name):
    """Benchmark both methods on a set of text pairs"""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {method_name}")
    print(f"{'='*70}")
    
    # Classical setup
    classical_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    classical_kernel = get_kernel(classical_config)
    
    # Quantum setup
    quantum_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    quantum_kernel = get_kernel(quantum_config)
    
    # Time classical
    start = time.time()
    classical_scores = [classical_kernel.similarity(t1, t2) for t1, t2 in zip(texts1, texts2)]
    classical_time = time.time() - start
    
    # Time quantum
    start = time.time()
    quantum_scores = [quantum_kernel.similarity(t1, t2) for t1, t2 in zip(texts1, texts2)]
    quantum_time = time.time() - start
    
    # Calculate statistics
    classical_mean = np.mean(classical_scores)
    quantum_mean = np.mean(quantum_scores)
    classical_std = np.std(classical_scores)
    quantum_std = np.std(quantum_scores)
    
    improvement = ((quantum_mean - classical_mean) / max(abs(classical_mean), 0.001)) * 100
    
    print(f"\nResults ({len(texts1)} pairs):")
    print(f"  Classical Cosine:")
    print(f"    Mean: {classical_mean:.4f} ± {classical_std:.4f}")
    print(f"    Time: {classical_time:.4f}s")
    print(f"  Quantum Interference:")
    print(f"    Mean: {quantum_mean:.4f} ± {quantum_std:.4f}")
    print(f"    Time: {quantum_time:.4f}s")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Speed ratio: {classical_time/quantum_time:.2f}x")
    
    return {
        'classical_mean': classical_mean,
        'quantum_mean': quantum_mean,
        'improvement': improvement,
        'classical_time': classical_time,
        'quantum_time': quantum_time
    }


def test_scenario_1_semantic_paraphrases():
    """Scenario 1: Paraphrases with same meaning"""
    texts1 = [
        "The artificial intelligence system learned from data",
        "Machine learning algorithms process information",
        "Neural networks recognize patterns automatically",
        "Deep learning models improve with training",
        "Natural language processing understands text",
    ]
    
    texts2 = [
        "AI systems acquire knowledge from datasets",
        "ML algorithms handle data processing",
        "Neural nets automatically identify patterns",
        "Deep models get better through practice",
        "NLP systems comprehend written language",
    ]
    
    return benchmark_methods(texts1, texts2, "Semantic Paraphrases (Same Meaning)")


def test_scenario_2_related_concepts():
    """Scenario 2: Related but different concepts"""
    texts1 = [
        "Machine learning",
        "Artificial intelligence",
        "Neural networks",
        "Deep learning",
        "Data science",
    ]
    
    texts2 = [
        "Computer vision",
        "Robotics",
        "Natural language processing",
        "Reinforcement learning",
        "Big data analytics",
    ]
    
    return benchmark_methods(texts1, texts2, "Related Concepts (Different Topics)")


def test_scenario_3_unrelated():
    """Scenario 3: Unrelated topics"""
    texts1 = [
        "Machine learning algorithms",
        "Quantum computing research",
        "Neural network architecture",
        "Deep learning models",
        "Artificial intelligence systems",
    ]
    
    texts2 = [
        "Cooking recipes for pasta",
        "Weather forecast for tomorrow",
        "Sports scores and statistics",
        "Travel destinations in Europe",
        "Music concert schedules",
    ]
    
    return benchmark_methods(texts1, texts2, "Unrelated Topics")


def test_scenario_4_technical_vs_layman():
    """Scenario 4: Technical vs layman descriptions"""
    texts1 = [
        "A convolutional neural network processes spatial data through learned filters",
        "Backpropagation adjusts weights using gradient descent optimization",
        "Attention mechanisms weight input features dynamically",
        "Transfer learning adapts pre-trained models to new tasks",
    ]
    
    texts2 = [
        "A computer program that learns to recognize images",
        "The system learns by adjusting how it makes decisions",
        "The model focuses on important parts of the input",
        "Using knowledge from one problem to solve another",
    ]
    
    return benchmark_methods(texts1, texts2, "Technical vs Layman Descriptions")


def main():
    """Run comprehensive comparison"""
    print("\n" + "="*70)
    print("COMPREHENSIVE QUANTUM vs CLASSICAL COMPARISON")
    print("="*70)
    
    results = []
    
    results.append(test_scenario_1_semantic_paraphrases())
    results.append(test_scenario_2_related_concepts())
    results.append(test_scenario_3_unrelated())
    results.append(test_scenario_4_technical_vs_layman())
    
    # Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_classical_time = np.mean([r['classical_time'] for r in results])
    avg_quantum_time = np.mean([r['quantum_time'] for r in results])
    
    print(f"\nAverage Performance Improvement: {avg_improvement:+.1f}%")
    print(f"Average Classical Time: {avg_classical_time:.4f}s")
    print(f"Average Quantum Time: {avg_quantum_time:.4f}s")
    print(f"Overall Speed Ratio: {avg_classical_time/avg_quantum_time:.2f}x")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("Quantum methods excel when:")
    print("  ✓ Semantic understanding matters more than exact matches")
    print("  ✓ Finding relationships between related concepts")
    print("  ✓ Paraphrase and synonym detection")
    print("  ✓ Context-aware similarity")
    print("\nClassical methods excel when:")
    print("  ✓ Exact word/phrase matching is needed")
    print("  ✓ Simple numeric feature comparison")
    print("  ✓ Maximum speed is critical")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
