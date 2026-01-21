"""
Complex Semantic Understanding Tests
Tests that showcase where quantum methods should excel:
- Multi-hop relationship discovery
- Abstract concept mapping
- Context-dependent meaning
- Implicit connections
- Cross-domain knowledge transfer
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel import get_kernel, KernelConfig


def test_abstract_concept_mapping():
    """Test 1: Abstract concepts - same meaning, different domains"""
    print("\n" + "="*80)
    print("TEST 1: ABSTRACT CONCEPT MAPPING")
    print("Testing: Same abstract concepts expressed in different domains")
    print("="*80)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Abstract concepts expressed differently
    abstract_pairs = [
        # Learning concepts
        ("The student learned from mistakes", "The model improved through trial and error"),
        ("Knowledge builds upon previous understanding", "Each layer processes information from earlier layers"),
        ("Practice leads to mastery", "Training epochs improve model performance"),
        
        # Pattern recognition
        ("The detective noticed patterns in the evidence", "The algorithm detected patterns in the data"),
        ("An expert recognizes subtle signs", "A neural network identifies hidden features"),
        
        # Communication concepts
        ("The message was encoded and transmitted", "Information was processed and forwarded"),
        ("The meaning was clear despite different words", "The signal was preserved through transformation"),
        
        # Problem-solving
        ("Breaking down complex problems into smaller parts", "Decomposing tasks into subtasks"),
        ("Finding connections between seemingly unrelated ideas", "Discovering correlations in high-dimensional space"),
    ]
    
    print("\n[Abstract Concept Pairs]:")
    print("-" * 80)
    
    classical_scores = []
    quantum_scores = []
    
    for text1, text2 in abstract_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        classical_scores.append(classical_sim)
        quantum_scores.append(quantum_sim)
        
        diff = quantum_sim - classical_sim
        improvement = (diff / max(abs(classical_sim), 0.001)) * 100
        
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"  Classical: {classical_sim:.4f}")
        print(f"  Quantum:   {quantum_sim:.4f}")
        print(f"  Difference: {diff:+.4f} ({improvement:+.1f}%)")
        if quantum_sim > classical_sim:
            print(f"  [Quantum WIN]")
        elif classical_sim > quantum_sim:
            print(f"  [Classical WIN]")
    
    avg_classical = np.mean(classical_scores)
    avg_quantum = np.mean(quantum_scores)
    avg_improvement = ((avg_quantum - avg_classical) / max(abs(avg_classical), 0.001)) * 100
    
    print(f"\n{'='*80}")
    print(f"Average Classical: {avg_classical:.4f}")
    print(f"Average Quantum:   {avg_quantum:.4f}")
    print(f"Average Improvement: {avg_improvement:+.1f}%")
    if avg_quantum > avg_classical:
        print(f"Winner: QUANTUM [+{avg_quantum - avg_classical:.4f}]")
    elif avg_classical > avg_quantum:
        print(f"Winner: CLASSICAL [+{avg_classical - avg_quantum:.4f}]")
    else:
        print(f"Winner: TIE")
    print(f"{'='*80}")


def test_multi_hop_relationships():
    """Test 2: Multi-hop relationships - indirect connections"""
    print("\n" + "="*80)
    print("TEST 2: MULTI-HOP RELATIONSHIP DISCOVERY")
    print("Testing: Finding indirect connections between concepts")
    print("="*80)
    
    config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    kernel = get_kernel(config)
    
    # Concepts with indirect relationships
    query = "Machine learning improves through data"
    
    related_concepts = [
        # Direct relationships (should be high)
        "Algorithms learn from examples",
        "Training data enhances model accuracy",
        
        # Indirect relationships (quantum should find these better)
        "Practice makes perfect in skill development",
        "Experience teaches valuable lessons",
        "Repeated exposure leads to understanding",
        "Feedback loops create improvement cycles",
        
        # Unrelated
        "The weather forecast for tomorrow",
        "Cooking recipes for dinner",
        "Sports statistics and scores",
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"\nFinding relationships (quantum entangled):")
    print("-" * 80)
    
    # Build relationship graph
    relationships = kernel.build_relationship_graph(
        [query] + related_concepts,
        threshold=0.3,
        use_quantum_entanglement=True
    )
    
    if query in relationships:
        print(f"\nTop relationships for '{query}':")
        for i, (related_text, similarity) in enumerate(relationships[query][:8], 1):
            print(f"  {i}. [{similarity:.4f}] {related_text[:70]}...")
    
    print(f"\n{'='*80}")


def test_context_dependent_meaning():
    """Test 3: Context-dependent meaning - same words, different contexts"""
    print("\n" + "="*80)
    print("TEST 3: CONTEXT-DEPENDENT MEANING")
    print("Testing: Understanding context changes meaning")
    print("="*80)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Same words, different contexts
    context_groups = [
        # Group 1: "bank" - financial vs river
        [
            ("I went to the bank to deposit money", "The financial institution processed my transaction"),
            ("I went to the bank to deposit money", "We sat by the river bank to fish"),
        ],
        # Group 2: "scale" - measurement vs weight
        [
            ("The scale shows your weight", "The measuring device displays your mass"),
            ("The scale shows your weight", "The musical scale has seven notes"),
        ],
        # Group 3: "run" - execution vs movement
        [
            ("The program will run smoothly", "The code executes without errors"),
            ("The program will run smoothly", "The athlete runs very fast"),
        ],
        # Group 4: "charge" - electrical vs responsibility
        [
            ("The battery needs to charge", "The device requires electrical power"),
            ("The battery needs to charge", "The soldier led the charge into battle"),
        ],
    ]
    
    print("\n[Context Disambiguation]:")
    print("-" * 80)
    
    classical_discrimination = []
    quantum_discrimination = []
    
    for group in context_groups:
        same_context_pair = group[0]
        different_context_pair = group[1]
        
        # Similarity for same context (should be HIGH)
        classical_same = classical_kernel.similarity(same_context_pair[0], same_context_pair[1])
        quantum_same = quantum_kernel.similarity(same_context_pair[0], same_context_pair[1])
        
        # Similarity for different context (should be LOW)
        classical_diff = classical_kernel.similarity(different_context_pair[0], different_context_pair[1])
        quantum_diff = quantum_kernel.similarity(different_context_pair[0], different_context_pair[1])
        
        # Discrimination score (difference between same vs different)
        classical_disc = classical_same - classical_diff
        quantum_disc = quantum_same - quantum_diff
        
        classical_discrimination.append(classical_disc)
        quantum_discrimination.append(quantum_disc)
        
        print(f"\nSame context:")
        print(f"  {same_context_pair[0]}")
        print(f"  {same_context_pair[1]}")
        print(f"    Classical: {classical_same:.4f}, Quantum: {quantum_same:.4f}")
        
        print(f"\nDifferent context:")
        print(f"  {different_context_pair[0]}")
        print(f"  {different_context_pair[1]}")
        print(f"    Classical: {classical_diff:.4f}, Quantum: {quantum_diff:.4f}")
        
        print(f"\n  Discrimination (higher = better):")
        print(f"    Classical: {classical_disc:.4f}")
        print(f"    Quantum:   {quantum_disc:.4f}")
        if quantum_disc > classical_disc:
            print(f"    [Quantum WIN - better at distinguishing context]")
    
    avg_classical_disc = np.mean(classical_discrimination)
    avg_quantum_disc = np.mean(quantum_discrimination)
    
    print(f"\n{'='*80}")
    print(f"Average Discrimination Score:")
    print(f"  Classical: {avg_classical_disc:.4f}")
    print(f"  Quantum:   {avg_quantum_disc:.4f}")
    print(f"  Improvement: {((avg_quantum_disc - avg_classical_disc) / max(abs(avg_classical_disc), 0.001) * 100):+.1f}%")
    if avg_quantum_disc > avg_classical_disc:
        print(f"Winner: QUANTUM [Better at context discrimination]")
    elif avg_classical_disc > avg_quantum_disc:
        print(f"Winner: CLASSICAL")
    else:
        print(f"Winner: TIE")
    print(f"{'='*80}")


def test_implicit_connections():
    """Test 4: Implicit connections - non-obvious relationships"""
    print("\n" + "="*80)
    print("TEST 4: IMPLICIT CONNECTION DISCOVERY")
    print("Testing: Finding non-obvious semantic relationships")
    print("="*80)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Implicit relationship pairs (not obvious at surface level)
    implicit_pairs = [
        # Learning patterns
        ("A child learns to walk by falling", "A neural network improves through backpropagation"),
        
        # Information processing
        ("The brain processes sensory input", "The computer processes data streams"),
        
        # Emergent properties
        ("Individual neurons create consciousness", "Simple rules generate complex behavior"),
        
        # Optimization
        ("Evolution selects successful traits", "Gradient descent finds optimal parameters"),
        
        # Memory and storage
        ("Experience shapes future decisions", "Training data influences model predictions"),
        
        # Adaptation
        ("Organisms adapt to environment", "Models adapt to new data distributions"),
    ]
    
    print("\n[Implicit Connection Pairs]:")
    print("-" * 80)
    
    classical_scores = []
    quantum_scores = []
    
    for text1, text2 in implicit_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        classical_scores.append(classical_sim)
        quantum_scores.append(quantum_sim)
        
        diff = quantum_sim - classical_sim
        improvement = (diff / max(abs(classical_sim), 0.001)) * 100
        
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"  Classical: {classical_sim:.4f}")
        print(f"  Quantum:   {quantum_sim:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")
        if quantum_sim > classical_sim + 0.01:  # Significant improvement threshold
            print(f"  [Quantum finds stronger implicit connection]")
    
    avg_classical = np.mean(classical_scores)
    avg_quantum = np.mean(quantum_scores)
    
    print(f"\n{'='*80}")
    print(f"Average Classical: {avg_classical:.4f}")
    print(f"Average Quantum:   {avg_quantum:.4f}")
    print(f"Average Improvement: {((avg_quantum - avg_classical) / max(abs(avg_classical), 0.001) * 100):+.1f}%")
    if avg_quantum > avg_classical:
        print(f"Winner: QUANTUM [+{avg_quantum - avg_classical:.4f}] - Better at implicit connections")
    elif avg_classical > avg_quantum:
        print(f"Winner: CLASSICAL")
    else:
        print(f"Winner: TIE")
    print(f"{'='*80}")


def test_cross_domain_knowledge():
    """Test 5: Cross-domain knowledge transfer"""
    print("\n" + "="*80)
    print("TEST 5: CROSS-DOMAIN KNOWLEDGE TRANSFER")
    print("Testing: Understanding concepts across different domains")
    print("="*80)
    
    classical_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=False,
        similarity_metric='cosine'
    )
    quantum_config = KernelConfig(
        use_sentence_transformers=True,
        use_quantum_methods=True,
        quantum_amplitude_encoding=True,
        similarity_metric='quantum'
    )
    
    classical_kernel = get_kernel(classical_config)
    quantum_kernel = get_kernel(quantum_config)
    
    # Cross-domain pairs (same concept, different domains)
    cross_domain_pairs = [
        # Biology -> Computer Science
        ("Neurons form connections in the brain", "Nodes form connections in neural networks"),
        ("DNA encodes genetic information", "Code encodes program instructions"),
        
        # Physics -> Machine Learning
        ("Energy flows through the system", "Information flows through layers"),
        ("Entropy increases in closed systems", "Complexity grows with model size"),
        
        # Psychology -> AI
        ("Cognitive biases affect decision making", "Model biases affect predictions"),
        ("Memory influences future behavior", "Training data influences model outputs"),
        
        # Economics -> Optimization
        ("Markets find equilibrium through supply and demand", "Optimization finds minimum through gradients"),
        ("Scarcity drives value", "Rare features drive importance"),
    ]
    
    print("\n[Cross-Domain Concept Pairs]:")
    print("-" * 80)
    
    classical_scores = []
    quantum_scores = []
    
    for text1, text2 in cross_domain_pairs:
        classical_sim = classical_kernel.similarity(text1, text2)
        quantum_sim = quantum_kernel.similarity(text1, text2)
        classical_scores.append(classical_sim)
        quantum_scores.append(quantum_sim)
        
        diff = quantum_sim - classical_sim
        improvement = (diff / max(abs(classical_sim), 0.001)) * 100
        
        print(f"\nDomain 1: {text1}")
        print(f"Domain 2: {text2}")
        print(f"  Classical: {classical_sim:.4f}")
        print(f"  Quantum:   {quantum_sim:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    avg_classical = np.mean(classical_scores)
    avg_quantum = np.mean(quantum_scores)
    
    print(f"\n{'='*80}")
    print(f"Average Classical: {avg_classical:.4f}")
    print(f"Average Quantum:   {avg_quantum:.4f}")
    print(f"Average Improvement: {((avg_quantum - avg_classical) / max(abs(avg_classical), 0.001) * 100):+.1f}%")
    if avg_quantum > avg_classical:
        print(f"Winner: QUANTUM [+{avg_quantum - avg_classical:.4f}] - Better cross-domain understanding")
    print(f"{'='*80}")


def main():
    """Run all complex semantic tests"""
    print("\n" + "="*80)
    print("COMPLEX SEMANTIC UNDERSTANDING TESTS")
    print("Testing where quantum methods should excel")
    print("="*80)
    
    results = {}
    
    test_abstract_concept_mapping()
    test_multi_hop_relationships()
    test_context_dependent_meaning()
    test_implicit_connections()
    test_cross_domain_knowledge()
    
    print("\n" + "="*80)
    print("ALL COMPLEX TESTS COMPLETED")
    print("="*80)
    print("\nSummary:")
    print("- These tests focus on complex semantic understanding where quantum methods")
    print("  should show advantages: abstract concepts, implicit connections, context")
    print("  discrimination, and cross-domain knowledge transfer.")
    print("\n- Quantum methods are designed for:")
    print("  * Finding non-obvious relationships")
    print("  * Understanding abstract concepts across domains")
    print("  * Context-dependent meaning resolution")
    print("  * Multi-hop relationship discovery")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
