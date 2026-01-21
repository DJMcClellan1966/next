"""
Demonstration: Advanced ML Toolbox
Shows how to use the advanced toolbox with big data, Quantum AI, and LLM
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ml_toolbox.advanced import AdvancedMLToolbox
except ImportError:
    print("Error: Advanced ML Toolbox not available")
    sys.exit(1)


def demonstrate_advanced_toolbox():
    """Demonstrate the advanced ML toolbox"""
    
    print("="*80)
    print("ADVANCED ML TOOLBOX DEMONSTRATION")
    print("="*80)
    
    # Initialize advanced toolbox
    print("\n[Initializing Advanced ML Toolbox]")
    advanced_toolbox = AdvancedMLToolbox()
    print(f"  Advanced Toolbox: {advanced_toolbox}")
    
    # Show compartments
    print("\n[Advanced Toolbox Structure]")
    print(f"  Advanced Compartment 1 (Big Data): {advanced_toolbox.big_data.get_info()['component_count']} components")
    print(f"  Advanced Compartment 2 (Infrastructure): {advanced_toolbox.infrastructure.get_info()['component_count']} components")
    print(f"  Advanced Compartment 3 (Algorithms): {advanced_toolbox.algorithms.get_info()['component_count']} components")
    
    # Sample data
    print("\n[Sample Data]")
    texts = [
        "Python programming is great for data science",
        "Machine learning uses neural networks",
        "Revenue increased by twenty percent",
        "Customer satisfaction drives business growth",
        "I need help with technical issues",
        "Support team provides assistance",
        "Learn Python through online courses",
        "Educational content helps students learn"
    ] * 1500  # Make it big data (12,000 items)
    
    print(f"  Texts: {len(texts)} items")
    print(f"  Big data threshold: {advanced_toolbox.big_data.big_data_threshold}")
    print(f"  Is big data: {advanced_toolbox.big_data.is_big_data(texts)}")
    
    # Advanced Compartment 1: Big Data
    print("\n" + "="*80)
    print("ADVANCED COMPARTMENT 1: BIG DATA")
    print("="*80)
    
    print("\n[Using AdvancedDataPreprocessor]")
    print("  Location: Advanced Compartment 1: Big Data")
    print("  Purpose: Large-scale data preprocessing")
    
    # Process with big data detection
    print("\n[Processing with Big Data Detection]")
    results = advanced_toolbox.big_data.preprocess(
        texts[:100],  # Use subset for demo
        advanced=True,
        detect_big_data=True,
        verbose=False
    )
    
    print(f"  Original samples: {len(texts[:100])}")
    print(f"  After deduplication: {len(results['deduplicated'])}")
    print(f"  Big data detected: {results['big_data_info']['is_big_data']}")
    print(f"  Optimized for big data: {results['big_data_info']['optimized_for_big_data']}")
    
    if 'compressed_embeddings' in results and results['compressed_embeddings'] is not None:
        X = results['compressed_embeddings']
        print(f"  Compressed embeddings shape: {X.shape}")
    
    print(f"\n  Why AdvancedDataPreprocessor is in Advanced Compartment 1:")
    print(f"    - Handles large-scale data preprocessing")
    print(f"    - Automatic big data detection and optimization")
    print(f"    - Batch processing support")
    print(f"    - Memory-efficient operations")
    
    # Advanced Compartment 2: Infrastructure
    print("\n" + "="*80)
    print("ADVANCED COMPARTMENT 2: INFRASTRUCTURE")
    print("="*80)
    
    print("\n[Using Quantum AI]")
    print("  Location: Advanced Compartment 2: Infrastructure")
    print("  Purpose: AI infrastructure services")
    
    try:
        ai = advanced_toolbox.infrastructure.get_ai_system(use_llm=False)
        print(f"  Quantum AI system initialized")
        
        # Test understanding
        understanding = ai.understanding.understand_intent("What is Python?")
        print(f"  Intent understanding: {understanding.get('intent', 'N/A')}")
        
        # Test search
        if results['deduplicated']:
            search_results = ai.search.search(
                "Python programming",
                results['deduplicated'][:5],
                top_k=3
            )
            print(f"  Search results: {len(search_results)} found")
    except Exception as e:
        print(f"  Error using Quantum AI: {e}")
    
    print("\n[Using LLM]")
    print("  Location: Advanced Compartment 2: Infrastructure")
    print("  Purpose: Text generation infrastructure")
    
    try:
        llm = advanced_toolbox.infrastructure.get_llm()
        print(f"  LLM initialized")
        
        # Test generation (short for demo)
        generated = llm.generate_grounded(
            "Explain Python",
            max_length=50
        )
        print(f"  Generated text: {generated.get('generated', 'N/A')[:50]}...")
    except Exception as e:
        print(f"  Error using LLM: {e}")
    
    print(f"\n  Why Quantum AI and LLM are in Advanced Compartment 2:")
    print(f"    - Provide AI infrastructure services")
    print(f"    - Use Quantum Kernel as infrastructure")
    print(f"    - Support advanced AI operations")
    
    # Advanced Compartment 3: Algorithms
    print("\n" + "="*80)
    print("ADVANCED COMPARTMENT 3: ALGORITHMS")
    print("="*80)
    
    print("\n[Available Algorithm Components]")
    print("  - ML Evaluation (MLEvaluator)")
    print("  - Hyperparameter Tuning (HyperparameterTuner)")
    print("  - Ensemble Learning (EnsembleLearner)")
    
    try:
        evaluator = advanced_toolbox.algorithms.get_evaluator()
        print(f"\n  ML Evaluator available")
        
        tuner = advanced_toolbox.algorithms.get_tuner()
        print(f"  Hyperparameter Tuner available")
        
        ensemble = advanced_toolbox.algorithms.get_ensemble()
        print(f"  Ensemble Learner available")
    except Exception as e:
        print(f"  Error accessing algorithms: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n[Advanced Compartment Organization]")
    print("\n  Advanced Compartment 1: BIG DATA")
    print("    - AdvancedDataPreprocessor (large-scale preprocessing)")
    print("    - Big data detection and optimization")
    print("    - Batch processing")
    print("    - Purpose: Handle large-scale data")
    
    print("\n  Advanced Compartment 2: INFRASTRUCTURE")
    print("    - Quantum AI (CompleteAISystem, components)")
    print("    - LLM (StandaloneQuantumLLM)")
    print("    - Quantum Kernel (semantic operations)")
    print("    - Purpose: Provide AI infrastructure")
    
    print("\n  Advanced Compartment 3: ALGORITHMS")
    print("    - ML Evaluation (model evaluation)")
    print("    - Hyperparameter Tuning (optimization)")
    print("    - Ensemble Learning (combining models)")
    print("    - Purpose: Advanced ML algorithms")
    
    print("\n[Component Placement]")
    print("  AdvancedDataPreprocessor -> Advanced Compartment 1: Big Data")
    print("  Quantum AI -> Advanced Compartment 2: Infrastructure")
    print("  LLM -> Advanced Compartment 2: Infrastructure")
    print("  Algorithms -> Advanced Compartment 3: Algorithms")
    
    print("="*80 + "\n")
    
    return advanced_toolbox


if __name__ == "__main__":
    try:
        toolbox = demonstrate_advanced_toolbox()
        print("[+] Demonstration complete!")
        print("\nYou can now use the Advanced ML Toolbox with:")
        print("  - advanced_toolbox.big_data.preprocess() for big data preprocessing")
        print("  - advanced_toolbox.infrastructure.get_ai_system() for Quantum AI")
        print("  - advanced_toolbox.infrastructure.get_llm() for LLM")
        print("  - advanced_toolbox.algorithms.get_evaluator() for model evaluation")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
