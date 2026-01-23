"""
Comprehensive Test Suite for AI Agent
Tests agent with real ML tasks to validate functionality
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

def test_ai_agent_comprehensive():
    """Comprehensive test suite for AI Agent"""
    print("="*80)
    print("COMPREHENSIVE AI AGENT TEST SUITE")
    print("="*80)
    print()
    
    try:
        from ml_toolbox.ai_agent import MLCodeAgent
        
        # Initialize agent
        print("[1/6] Initializing AI Agent...")
        agent = MLCodeAgent(use_llm=False, use_pattern_composition=True)
        print("[OK] Agent initialized with pattern composition")
        
        # Test cases
        test_cases = [
            {
                'name': 'Simple Classification',
                'task': 'Classify data into 2 classes',
                'expected': 'classification'
            },
            {
                'name': 'Simple Regression',
                'task': 'Predict continuous values',
                'expected': 'regression'
            },
            {
                'name': 'With Preprocessing',
                'task': 'Preprocess data and train classifier',
                'expected': 'preprocessing + classification'
            },
            {
                'name': 'Model Evaluation',
                'task': 'Train model and evaluate accuracy',
                'expected': 'training + evaluation'
            },
            {
                'name': 'Complete Pipeline',
                'task': 'Load data, preprocess, train, and evaluate',
                'expected': 'full pipeline'
            }
        ]
        
        results = []
        
        print("\n[2/6] Running test cases...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}/{len(test_cases)}: {test_case['name']}")
            print(f"  Task: {test_case['task']}")
            
            start_time = time.time()
            result = agent.build(test_case['task'])
            elapsed = time.time() - start_time
            
            success = result.get('success', False)
            status = "[PASS]" if success else "[FAIL]"
            
            print(f"  Status: {status}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Iterations: {result.get('iterations', 0)}")
            
            if success:
                code_length = len(result.get('code', ''))
                print(f"  Code length: {code_length} chars")
                print(f"  Output: {result.get('output', '')[:100]}...")
            else:
                error = result.get('error', 'Unknown error')
                print(f"  Error: {error[:100]}...")
            
            results.append({
                'test': test_case['name'],
                'success': success,
                'time': elapsed,
                'iterations': result.get('iterations', 0)
            })
        
        # Statistics
        print("\n[3/6] Test Statistics...")
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['success'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        avg_time = sum(r['time'] for r in results) / total_tests if total_tests > 0 else 0
        avg_iterations = sum(r['iterations'] for r in results) / total_tests if total_tests > 0 else 0
        
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average iterations: {avg_iterations:.1f}")
        
        # Pattern graph statistics
        print("\n[4/6] Pattern Graph Statistics...")
        stats = agent.graph.get_statistics()
        print(f"  Total patterns: {stats['total_patterns']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print(f"  Successful compositions: {stats['total_compositions']}")
        print(f"  Failed compositions: {stats['total_failures']}")
        if stats['total_compositions'] + stats['total_failures'] > 0:
            print(f"  Graph success rate: {stats['success_rate']:.1%}")
        
        # Agent history
        print("\n[5/6] Agent History...")
        history = agent.get_history()
        print(f"  Total tasks attempted: {len(history)}")
        successful = sum(1 for h in history if h.get('success', False))
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(history) - successful}")
        
        # Recommendations
        print("\n[6/6] Recommendations...")
        if success_rate < 50:
            print("  ⚠️  Low success rate - need to:")
            print("     - Add more patterns")
            print("     - Improve error handling")
            print("     - Fix common failures")
        elif success_rate < 70:
            print("  ✅ Good progress - can improve by:")
            print("     - Adding more patterns")
            print("     - Better error handling")
        else:
            print("  ✅ Excellent! Agent is working well")
            print("     - Consider adding advanced features")
            print("     - Add task planning")
            print("     - Add meta-learning")
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        
        return {
            'success_rate': success_rate,
            'results': results,
            'stats': stats
        }
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    test_ai_agent_comprehensive()
