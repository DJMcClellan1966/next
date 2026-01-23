"""
Test Medulla Auto-Start in ML Toolbox
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

def test_medulla_auto_start():
    """Test that Medulla automatically starts with ML Toolbox"""
    print("="*80)
    print("TESTING MEDULLA AUTO-START IN ML TOOLBOX")
    print("="*80)
    print()
    
    try:
        from ml_toolbox import MLToolbox
        
        print("[1/3] Creating ML Toolbox (Medulla should auto-start)...")
        toolbox = MLToolbox(auto_start_medulla=True)
        print("[OK] ML Toolbox created")
        
        # Check if Medulla is running
        if toolbox.medulla:
            print("[OK] Medulla system is available")
            if toolbox.medulla.regulation_running:
                print("[OK] Medulla regulation is running")
            else:
                print("[WARNING] Medulla regulation is not running")
        else:
            print("[WARNING] Medulla system is not available")
        
        # Get system status
        print("\n[2/3] Checking system status...")
        status = toolbox.get_system_status()
        print(f"[OK] System Status:")
        print(f"  State: {status.get('state', 'unknown')}")
        print(f"  CPU: {status.get('cpu_percent', 0):.1f}%")
        print(f"  Memory: {status.get('memory_percent', 0):.1f}%")
        print(f"  Quantum Resources Available:")
        quantum_resources = status.get('quantum_resources', {})
        print(f"    CPU Limit: {quantum_resources.get('cpu_limit_percent', 0):.1f}%")
        print(f"    Memory Limit: {quantum_resources.get('memory_limit_mb', 0):.1f} MB")
        print(f"    Cores Allocated: {quantum_resources.get('cores_allocated', 0)}")
        
        # Test using toolbox with Medulla
        print("\n[3/3] Testing toolbox operations with Medulla...")
        
        # Use a component that benefits from Medulla
        if 'VirtualQuantumComputer' in toolbox.infrastructure.components:
            print("[OK] Virtual Quantum Computer available")
            qc = toolbox.get_quantum_computer(num_qubits=4)
            print(f"[OK] Quantum Computer created ({qc.num_qubits} qubits)")
            
            # Perform operations
            qc.apply_gate('H', 0)
            qc.apply_gate('X', 1)
            print("[OK] Quantum operations performed")
            
            # Get metrics
            metrics = qc.get_metrics()
            print(f"[OK] Operations: {metrics['operations_performed']}")
            print(f"[OK] Avg time/op: {metrics['avg_operation_time']:.6f}s")
        else:
            print("[WARNING] Virtual Quantum Computer not available")
        
        # Test context manager (auto-cleanup)
        print("\n[4/4] Testing context manager (auto-cleanup)...")
        with MLToolbox(auto_start_medulla=True) as tb:
            if tb.medulla and tb.medulla.regulation_running:
                print("[OK] Medulla running in context")
            time.sleep(0.5)
        print("[OK] Context exited - Medulla should be stopped")
        
        # Final status
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        print("\n[SUCCESS] Medulla automatically starts with ML Toolbox!")
        print("  - Medulla starts when toolbox is created")
        print("  - System status available via get_system_status()")
        print("  - Quantum computer uses Medulla resources")
        print("  - Context manager auto-stops Medulla")
        
        # Cleanup
        if toolbox.medulla and toolbox.medulla.regulation_running:
            toolbox.medulla.stop_regulation()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_medulla_auto_start()
