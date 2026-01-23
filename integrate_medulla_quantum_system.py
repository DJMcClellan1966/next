"""
Integrate Medulla Oblongata System with Virtual Quantum Computer
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

def integrate_systems():
    """Integrate Medulla and Virtual Quantum Computer"""
    print("="*80)
    print("INTEGRATING MEDULLA OBLONGATA + VIRTUAL QUANTUM COMPUTER")
    print("="*80)
    print()
    
    try:
        from medulla_oblongata_system import MedullaOblongataSystem
        from virtual_quantum_computer import VirtualQuantumComputer
        
        print("[1/3] Creating Medulla Oblongata System...")
        medulla = MedullaOblongataSystem(
            max_cpu_percent=80.0,
            max_memory_percent=75.0,
            min_cpu_reserve=20.0,
            min_memory_reserve_mb=1024.0
        )
        print("[OK] Medulla system created")
        
        print("\n[2/3] Creating Virtual Quantum Computer...")
        qc = VirtualQuantumComputer(
            num_qubits=8,
            medulla=medulla,
            use_architecture_optimizations=True
        )
        print(f"[OK] Quantum computer created ({qc.num_qubits} qubits)")
        
        print("\n[3/3] Starting integrated system...")
        with medulla:
            # Wait for regulation to stabilize
            time.sleep(1)
            
            # Get system status
            status = medulla.get_system_status()
            print(f"\n[OK] System Status:")
            print(f"  State: {status['state']}")
            print(f"  CPU: {status['cpu_percent']:.1f}%")
            print(f"  Memory: {status['memory_percent']:.1f}%")
            print(f"  Quantum Resources:")
            print(f"    CPU Limit: {status['quantum_resources']['cpu_limit_percent']:.1f}%")
            print(f"    Memory Limit: {status['quantum_resources']['memory_limit_mb']:.1f} MB")
            print(f"    Cores Allocated: {status['quantum_resources']['cores_allocated']}")
            
            # Test quantum operations
            print("\n[TEST] Performing quantum operations...")
            start = time.time()
            
            # Create superposition
            qc.apply_gate('H', 0)
            qc.apply_gate('H', 1)
            qc.apply_gate('H', 2)
            
            # Apply gates
            qc.apply_gate('X', 3)
            qc.apply_gate('Z', 4)
            
            # Parallel operations
            operations = [('H', i) for i in range(5, 8)]
            parallel_time = qc.parallel_quantum_operation(operations)
            
            elapsed = time.time() - start
            
            print(f"[OK] Operations completed in {elapsed:.4f}s")
            print(f"  Parallel operations: {parallel_time:.4f}s")
            
            # Get metrics
            qc_metrics = qc.get_metrics()
            print(f"\n[OK] Quantum Computer Metrics:")
            print(f"  Operations: {qc_metrics['operations_performed']}")
            print(f"  Total time: {qc_metrics['total_compute_time']:.4f}s")
            print(f"  Avg time/op: {qc_metrics['avg_operation_time']:.6f}s")
            print(f"  Parallel ops: {qc_metrics['parallel_operations']}")
            
            # Final status
            final_status = medulla.get_system_status()
            print(f"\n[OK] Final System Status:")
            print(f"  State: {final_status['state']}")
            print(f"  CPU: {final_status['cpu_percent']:.1f}%")
            print(f"  Memory: {final_status['memory_percent']:.1f}%")
            print(f"  System Disruptions: {final_status['performance_metrics']['system_disruptions']}")
        
        print("\n" + "="*80)
        print("INTEGRATION COMPLETE")
        print("="*80)
        print("\n[SUCCESS] Medulla + Virtual Quantum Computer integrated!")
        print("  - Medulla regulates system resources")
        print("  - Quantum computer uses allocated resources")
        print("  - Architecture optimizations enabled")
        print("  - Minimal system disruption")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    integrate_systems()
