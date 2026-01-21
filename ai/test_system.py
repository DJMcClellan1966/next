"""
Test suite for Complete AI System
"""
from .core import CompleteAISystem
from .components import (
    SemanticUnderstandingEngine,
    KnowledgeGraphBuilder,
    IntelligentSearch,
    ReasoningEngine,
    LearningSystem,
    ConversationalAI
)
from quantum_kernel import get_kernel, KernelConfig


def test_complete_system():
    """Test the complete AI system"""
    print("Testing Complete AI System...")
    
    # Create system
    ai = CompleteAISystem()
    
    # Test basic processing
    result = ai.process({
        "query": "divine love",
        "documents": [
            "God is love",
            "Love is patient",
            "Faith, hope, and love"
        ]
    })
    
    assert "search" in result
    assert "knowledge_graph" in result
    assert "understanding" in result
    assert "kernel_stats" in result
    
    print("[OK] Complete system test passed")


def test_individual_components():
    """Test individual components"""
    print("Testing individual components...")
    
    kernel = get_kernel(KernelConfig())
    
    # Test understanding
    understanding = SemanticUnderstandingEngine(kernel)
    intent = understanding.understand_intent("I want to search")
    assert "intent" in intent
    assert "confidence" in intent
    print("[OK] Understanding component test passed")
    
    # Test search
    search = IntelligentSearch(kernel)
    results = search.search("love", ["God is love", "Love is patient"])
    assert len(results) > 0
    print("[OK] Search component test passed")
    
    # Test knowledge graph
    graph_builder = KnowledgeGraphBuilder(kernel)
    graph = graph_builder.build_graph(["Doc 1", "Doc 2"])
    assert "nodes" in graph
    assert "edges" in graph
    print("[OK] Knowledge graph component test passed")
    
    # Test reasoning
    reasoning = ReasoningEngine(kernel)
    result = reasoning.reason(["A is B", "B is C"], "What is A?")
    assert "connections" in result
    assert "confidence" in result
    print("[OK] Reasoning component test passed")
    
    # Test learning
    learning = LearningSystem(kernel)
    result = learning.learn_from_examples([("input", "output")])
    assert "patterns_learned" in result
    print("[OK] Learning component test passed")
    
    # Test conversation
    conversation = ConversationalAI(kernel)
    response = conversation.respond("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
    print("[OK] Conversation component test passed")


def test_system_stats():
    """Test system statistics"""
    print("Testing system statistics...")
    
    ai = CompleteAISystem()
    
    # Use system
    ai.process({
        "query": "test",
        "documents": ["test document"]
    })
    
    # Get stats
    stats = ai.get_stats()
    assert "kernel" in stats
    assert "conversation_history_length" in stats
    assert "learned_patterns" in stats
    
    print("[OK] System statistics test passed")


def test_system_reset():
    """Test system reset"""
    print("Testing system reset...")
    
    ai = CompleteAISystem()
    
    # Use system
    ai.process({
        "query": "test",
        "documents": ["test document"]
    })
    
    # Reset
    ai.reset()
    
    # Verify reset
    assert len(ai.conversation.conversation_history) == 0
    assert len(ai.learning.learned_patterns) == 0
    
    print("[OK] System reset test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("COMPLETE AI SYSTEM TEST SUITE")
    print("=" * 80)
    
    try:
        test_complete_system()
        test_individual_components()
        test_system_stats()
        test_system_reset()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
