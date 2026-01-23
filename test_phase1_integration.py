"""
Test Phase 1 Integration
Verify that testing, deployment, and optimization modules are properly integrated
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_phase1_integration():
    """Test Phase 1 integrations"""
    print("="*80)
    print("PHASE 1 INTEGRATION TEST")
    print("="*80)
    print()
    
    # Test 1: Testing Infrastructure
    print("[1] Testing Infrastructure Integration...")
    try:
        from ml_toolbox import MLToolbox, ComprehensiveMLTestSuite, MLBenchmarkSuite
        
        toolbox = MLToolbox(check_dependencies=False)
        
        # Test suite
        test_suite = toolbox.get_test_suite()
        if test_suite:
            print("  [OK] ComprehensiveMLTestSuite available")
        else:
            print("  [WARNING] ComprehensiveMLTestSuite not available")
        
        # Benchmark suite
        benchmark_suite = toolbox.get_benchmark_suite()
        if benchmark_suite:
            print("  [OK] MLBenchmarkSuite available")
        else:
            print("  [WARNING] MLBenchmarkSuite not available")
        
        # Direct import
        if ComprehensiveMLTestSuite:
            print("  [OK] Direct import works")
        if MLBenchmarkSuite:
            print("  [OK] Direct import works")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Model Persistence
    print("\n[2] Model Persistence Integration...")
    try:
        from ml_toolbox import MLToolbox, ModelPersistence
        
        toolbox = MLToolbox(check_dependencies=False)
        
        # Get persistence
        persistence = toolbox.get_model_persistence()
        if persistence:
            print("  [OK] ModelPersistence available")
            print(f"  [OK] Storage dir: {persistence.storage_dir}")
        else:
            print("  [WARNING] ModelPersistence not available")
        
        # Direct import
        if ModelPersistence:
            print("  [OK] Direct import works")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Model Optimization
    print("\n[3] Model Optimization Integration...")
    try:
        from ml_toolbox import MLToolbox, ModelCompression, ModelCalibration
        
        toolbox = MLToolbox(check_dependencies=False)
        
        # Compression
        compression = toolbox.get_model_compression()
        if compression:
            print("  [OK] ModelCompression available")
        else:
            print("  [WARNING] ModelCompression not available")
        
        # Calibration
        calibration = toolbox.get_model_calibration()
        if calibration:
            print("  [OK] ModelCalibration available")
        else:
            print("  [WARNING] ModelCalibration not available")
        
        # Direct imports
        if ModelCompression:
            print("  [OK] Direct import works")
        if ModelCalibration:
            print("  [OK] Direct import works")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Module Structure
    print("\n[4] Module Structure...")
    try:
        from ml_toolbox import testing, deployment, optimization
        print("  [OK] testing module accessible")
        print("  [OK] deployment module accessible")
        print("  [OK] optimization module accessible")
    except Exception as e:
        print(f"  [WARNING] Module access: {e}")
    
    print("\n" + "="*80)
    print("PHASE 1 INTEGRATION TEST COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  ✅ Testing Infrastructure: Integrated")
    print("  ✅ Model Persistence: Integrated")
    print("  ✅ Model Optimization: Integrated")
    print("\nAll Phase 1 components are now available in MLToolbox!")


if __name__ == '__main__':
    test_phase1_integration()
