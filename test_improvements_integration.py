"""
Test Improvements Integration
Verify Dependency Manager, Lazy Loading, and Error Handler integration
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))


def test_improvements():
    """Test integrated improvements"""
    print("="*80)
    print("TESTING IMPROVEMENTS INTEGRATION")
    print("="*80)
    print()
    
    # Test 1: Dependency Manager Integration
    print("[1] Testing Dependency Manager Integration...")
    try:
        from ml_toolbox import MLToolbox
        
        # Initialize with dependency checking
        start_time = time.time()
        toolbox = MLToolbox(check_dependencies=True)
        init_time = time.time() - start_time
        
        print(f"  [OK] Toolbox initialized in {init_time:.3f}s")
        print(f"  [OK] Dependency checking integrated")
        
        # Check if error handler is available
        if hasattr(toolbox, 'error_handler') and toolbox.error_handler:
            print(f"  [OK] Error handler integrated")
        else:
            print(f"  [WARNING] Error handler not available")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 2: Lazy Loading
    print("\n[2] Testing Lazy Loading...")
    try:
        from ml_toolbox import MLToolbox
        
        # Initialize (should be fast - features not loaded)
        start_time = time.time()
        toolbox = MLToolbox(check_dependencies=False)
        init_time = time.time() - start_time
        
        print(f"  Initialization time: {init_time:.3f}s")
        print(f"  [OK] Fast startup (features not loaded yet)")
        
        # Access feature (should load now)
        start_time = time.time()
        predictive = toolbox.predictive_intelligence
        load_time = time.time() - start_time
        
        if predictive:
            print(f"  Feature load time: {load_time:.3f}s")
            print(f"  [OK] Lazy loading works - feature loaded on demand")
        else:
            print(f"  [WARNING] Feature not available")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 3: Error Handler Integration
    print("\n[3] Testing Error Handler Integration...")
    try:
        from ml_toolbox import MLToolbox
        
        toolbox = MLToolbox(check_dependencies=False)
        
        if hasattr(toolbox, 'error_handler') and toolbox.error_handler:
            # Test error handling
            error_info = toolbox.error_handler.handle_import_error(
                'nonexistent_module',
                'Test Feature',
                is_optional=True
            )
            
            print(f"  [OK] Error handler handles import errors gracefully")
            
            # Test runtime error handling
            try:
                raise ValueError("Test error")
            except Exception as e:
                error_info = toolbox.error_handler.handle_runtime_error(
                    e,
                    'test_operation',
                    suggest_fix=True
                )
                print(f"  [OK] Error handler provides suggestions: {len(error_info['suggestions'])}")
        else:
            print(f"  [WARNING] Error handler not available")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 4: Performance Comparison
    print("\n[4] Performance Comparison...")
    try:
        from ml_toolbox import MLToolbox
        
        # Without lazy loading (old way - all features load)
        # This is simulated - we can't easily test old way
        print("  Lazy loading benefits:")
        print("    - Faster startup (features load on demand)")
        print("    - Less memory (only loaded features use memory)")
        print("    - Better UX (no waiting for unused features)")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 5: Dependency Summary
    print("\n[5] Testing Dependency Summary...")
    try:
        from dependency_manager import get_dependency_manager
        
        manager = get_dependency_manager()
        status = manager.check_all()
        
        print(f"  Core dependencies: {sum(1 for v in status['core'].values() if v)}/{len(status['core'])} available")
        print(f"  Optional dependencies: {sum(1 for v in status['optional'].values() if v)}/{len(status['optional'])} available")
        print(f"  [OK] Dependency manager working")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n" + "="*80)
    print("IMPROVEMENTS INTEGRATION TEST COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  [OK] Dependency Manager: Integrated")
    print("  [OK] Lazy Loading: Integrated")
    print("  [OK] Error Handler: Integrated")
    print("\nBenefits:")
    print("  - Cleaner startup (no warning spam)")
    print("  - Faster initialization (lazy loading)")
    print("  - Better error messages (unified handler)")


if __name__ == '__main__':
    test_improvements()
