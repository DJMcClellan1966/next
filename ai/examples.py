"""
Example Usage of Complete AI System
Demonstrates various use cases
"""
from .core import CompleteAISystem
from quantum_kernel import KernelConfig


def example_basic_usage():
    """Basic usage example"""
    print("=" * 80)
    print("EXAMPLE 1: BASIC USAGE")
    print("=" * 80)
    
    # Create system
    ai = CompleteAISystem()
    
    # Process query with documents
    result = ai.process({
        "query": "divine love",
        "documents": [
            "God is love and love is patient",
            "Faith is the assurance of things hoped for",
            "By grace you have been saved through faith"
        ]
    })
    
    print("\nSearch Results:")
    for item in result["search"]["results"][:3]:
        print(f"  - {item['text'][:50]}... (similarity: {item['similarity']:.3f})")
    
    print(f"\nThemes Discovered: {len(result['search']['themes'])}")
    print(f"Knowledge Graph Nodes: {len(result['knowledge_graph']['nodes'])}")


def example_understanding():
    """Understanding example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: SEMANTIC UNDERSTANDING")
    print("=" * 80)
    
    ai = CompleteAISystem()
    
    result = ai.understanding.understand_intent(
        "I want to search for information about machine learning"
    )
    
    print(f"Query: 'I want to search for information about machine learning'")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.3f}")


def example_reasoning():
    """Reasoning example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: REASONING")
    print("=" * 80)
    
    ai = CompleteAISystem()
    
    result = ai.reasoning.reason(
        premises=["God is love", "Love is patient"],
        question="What is God like?"
    )
    
    print(f"Premises: {result['premises']}")
    print(f"Question: {result['question']}")
    print(f"Connections: {len(result['connections'])}")
    print(f"Coherence: {result['coherence']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")


def example_conversation():
    """Conversation example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: CONVERSATION")
    print("=" * 80)
    
    ai = CompleteAISystem()
    
    response1 = ai.conversation.respond("I want to search for information")
    print(f"User: 'I want to search for information'")
    print(f"AI: {response1}")
    
    response2 = ai.conversation.respond("Tell me about love")
    print(f"\nUser: 'Tell me about love'")
    print(f"AI: {response2}")


def example_learning():
    """Learning example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: LEARNING")
    print("=" * 80)
    
    ai = CompleteAISystem()
    
    result = ai.learning.learn_from_examples([
        ("What is love?", "Love is patient and kind"),
        ("What is faith?", "Faith is the assurance of things hoped for"),
        ("What is grace?", "Grace is unmerited favor")
    ])
    
    print(f"Examples: 3")
    print(f"Patterns Learned: {result['patterns_learned']}")
    print(f"Input Themes: {result['input_themes']}")
    print(f"Output Themes: {result['output_themes']}")


def example_knowledge_graph():
    """Knowledge graph example"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: KNOWLEDGE GRAPH")
    print("=" * 80)
    
    ai = CompleteAISystem()
    
    documents = [
        "God is love",
        "Love is patient and kind",
        "Faith, hope, and love",
        "By grace through faith"
    ]
    
    graph = ai.knowledge_graph.build_graph(documents)
    
    print(f"Documents: {len(documents)}")
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")
    print(f"Themes: {len(graph['themes'])}")
    
    for theme in graph['themes'][:2]:
        print(f"  - {theme['theme']}: {len(theme['nodes'])} nodes")


def run_all_examples():
    """Run all examples"""
    example_basic_usage()
    example_understanding()
    example_reasoning()
    example_conversation()
    example_learning()
    example_knowledge_graph()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_all_examples()
