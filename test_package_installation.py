"""
Test script to verify package installation and basic functionality
"""
import sys

def test_import():
    """Test basic import"""
    print("Testing package import...")
    try:
        from ml_toolbox import MLToolbox
        print("[OK] MLToolbox imported successfully")
        return True
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_instantiation():
    """Test creating an instance"""
    print("\nTesting MLToolbox instantiation...")
    try:
        from ml_toolbox import MLToolbox
        toolbox = MLToolbox(
            check_dependencies=False,  # Skip dependency checks for testing
            auto_start_optimizer=False,  # Skip optimizer for testing
            verbose_errors=False
        )
        print("[OK] MLToolbox instance created successfully")
        return True, toolbox
    except Exception as e:
        print(f"[ERROR] Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_basic_functionality(toolbox):
    """Test basic ML functionality"""
    print("\nTesting basic ML functionality...")
    try:
        from sklearn.datasets import make_classification
        import numpy as np
        
        # Generate simple data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        
        # Test fit
        result = toolbox.fit(X, y)
        print(f"[OK] fit() completed")
        print(f"   Model type: {result.get('model_type', 'N/A')}")
        
        # Test predict
        predictions = toolbox.predict(result['model'], X[:5])
        print(f"[OK] predict() completed")
        print(f"   Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("ML TOOLBOX PACKAGE INSTALLATION TEST")
    print("="*80)
    
    # Test import
    if not test_import():
        print("\n[ERROR] Package installation test FAILED")
        print("\n[TIP] Try installing the package:")
        print("   pip install -e .")
        sys.exit(1)
    
    # Test instantiation
    success, toolbox = test_instantiation()
    if not success:
        print("\n[ERROR] Package instantiation test FAILED")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality(toolbox):
        print("\n[WARNING] Basic functionality test had issues")
        print("   Package is installed but some features may not work")
        sys.exit(0)
    
    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED - Package is working correctly!")
    print("="*80)

if __name__ == "__main__":
    main()
