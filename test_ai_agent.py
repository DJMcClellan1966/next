"""
Test AI Agent - Code Generation System
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_ai_agent():
    """Test AI Agent code generation"""
    print("="*80)
    print("TESTING AI AGENT - CODE GENERATION")
    print("="*80)
    print()
    
    try:
        from ml_toolbox.ai_agent import MLCodeAgent
        
        # Initialize agent
        print("[1/3] Initializing AI Agent...")
        agent = MLCodeAgent(use_llm=False)  # Use templates for testing
        print("[OK] Agent initialized")
        
        # Test 1: Simple classification
        print("\n[2/3] Testing code generation...")
        print("\nTest 1: Simple Classification")
        result1 = agent.build("Classify data into 2 classes")
        
        if result1['success']:
            print("[OK] Code generated successfully")
            print(f"[OK] Iterations: {result1['iterations']}")
            print(f"\nGenerated Code:\n{result1['code'][:500]}...")
        else:
            print(f"[WARNING] Generation had issues: {result1.get('error', 'Unknown')}")
        
        # Test 2: Knowledge base
        print("\n[3/3] Testing knowledge base...")
        from ml_toolbox.ai_agent import ToolboxKnowledgeBase
        
        kb = ToolboxKnowledgeBase()
        solutions = kb.find_solution("classification")
        print(f"[OK] Found {len(solutions)} solutions for 'classification'")
        
        capabilities = kb.get_capabilities()
        print(f"[OK] Knowledge base has {len(capabilities)} capability categories")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        print("\n[SUCCESS] AI Agent foundation is working!")
        print("\nNext Steps:")
        print("  1. Add more patterns to knowledge base")
        print("  2. Test with LLM (use_llm=True)")
        print("  3. Add task planning")
        print("  4. Improve error handling")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_ai_agent()
