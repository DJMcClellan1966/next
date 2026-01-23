"""
Medulla Oblongata System
Autonomic resource regulation for virtual quantum computing

The Medulla regulates system resources like the brain's medulla oblongata
regulates autonomic functions (breathing, heart rate, etc.)
"""
import sys
from pathlib import Path
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install with: pip install psutil")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    from architecture_optimizer import get_architecture_optimizer
    ARCHITECTURE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ARCHITECTURE_OPTIMIZER_AVAILABLE = False
    warnings.warn("Architecture optimizer not available")


class SystemState(Enum):
    """System state for resource regulation"""
    IDLE = "idle"
    NORMAL = "normal"
    STRESSED = "stressed"
    CRITICAL = "critical"
    RECOVERING = "recovering"


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    cpu_cores_available: int
    memory_available_mb: float
    load_average: float
    timestamp: float


@dataclass
class QuantumTask:
    """Quantum computing task"""
    task_id: str
    priority: int  # 1-10, higher = more important
    cores_requested: int
    memory_requested_mb: float
    estimated_duration: float
    callback: Optional[Callable] = None
    status: str = "pending"


class MedullaOblongataSystem:
    """
    Medulla Oblongata System
    
    Regulates system resources like the brain's medulla oblongata:
    - Monitors CPU, memory, and system load
    - Allocates resources to virtual quantum computer
    - Prevents system disruption
    - Balances performance vs stability
    """
    
    def __init__(
        self,
        max_cpu_percent: float = 80.0,
        max_memory_percent: float = 75.0,
        min_cpu_reserve: float = 20.0,
        min_memory_reserve_mb: float = 1024.0,
        regulation_interval: float = 0.5,
        adaptive_threshold: bool = True
    ):
        """
        Initialize Medulla System
        
        Args:
            max_cpu_percent: Maximum CPU usage before throttling
            max_memory_percent: Maximum memory usage before throttling
            min_cpu_reserve: Minimum CPU to reserve for system
            min_memory_reserve_mb: Minimum memory to reserve for system
            regulation_interval: How often to check system state (seconds)
            adaptive_threshold: Adjust thresholds based on system load
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.min_cpu_reserve = min_cpu_reserve
        self.min_memory_reserve_mb = min_memory_reserve_mb
        self.regulation_interval = regulation_interval
        self.adaptive_threshold = adaptive_threshold
        
        # System state
        self.state = SystemState.IDLE
        self.metrics_history = deque(maxlen=100)
        self.quantum_tasks = queue.PriorityQueue()
        self.active_tasks: Dict[str, QuantumTask] = {}
        
        # Resource allocation
        self.quantum_cpu_limit = 0.0
        self.quantum_memory_limit_mb = 0.0
        self.quantum_cores_allocated = 0
        
        # Regulation thread
        self.regulation_thread: Optional[threading.Thread] = None
        self.regulation_running = False
        self.regulation_lock = threading.Lock()
        
        # Architecture optimizer
        if ARCHITECTURE_OPTIMIZER_AVAILABLE:
            self.arch_optimizer = get_architecture_optimizer()
            self.optimal_threads = self.arch_optimizer.get_optimal_thread_count()
        else:
            self.arch_optimizer = None
            self.optimal_threads = 4
        
        # Performance tracking
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_throttled': 0,
            'system_disruptions': 0,
            'avg_task_time': 0.0,
            'total_compute_time': 0.0
        }
    
    def start_regulation(self):
        """Start the regulation thread"""
        if self.regulation_running:
            return
        
        self.regulation_running = True
        self.regulation_thread = threading.Thread(target=self._regulation_loop, daemon=True)
        self.regulation_thread.start()
        print("[Medulla] Regulation started")
    
    def stop_regulation(self):
        """Stop the regulation thread"""
        self.regulation_running = False
        if self.regulation_thread:
            self.regulation_thread.join(timeout=2.0)
        print("[Medulla] Regulation stopped")
    
    def _regulation_loop(self):
        """Main regulation loop - monitors and adjusts resources"""
        while self.regulation_running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Determine system state
                self._update_system_state(metrics)
                
                # Allocate resources
                self._allocate_resources(metrics)
                
                # Regulate active tasks
                self._regulate_tasks(metrics)
                
                time.sleep(self.regulation_interval)
            except Exception as e:
                warnings.warn(f"Regulation loop error: {e}")
                time.sleep(self.regulation_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        if not PSUTIL_AVAILABLE:
            # Fallback metrics
            return ResourceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                cpu_cores_available=self.optimal_threads,
                memory_available_mb=4096.0,
                load_average=0.0,
                timestamp=time.time()
            )
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            cpu_cores_available=cpu_count,
            memory_available_mb=memory.available / (1024 * 1024),
            load_average=cpu_percent / 100.0,
            timestamp=time.time()
        )
    
    def _update_system_state(self, metrics: ResourceMetrics):
        """Update system state based on metrics"""
        if metrics.cpu_percent < 30 and metrics.memory_percent < 50:
            self.state = SystemState.IDLE
        elif metrics.cpu_percent < self.max_cpu_percent and metrics.memory_percent < self.max_memory_percent:
            self.state = SystemState.NORMAL
        elif metrics.cpu_percent < 90 and metrics.memory_percent < 90:
            self.state = SystemState.STRESSED
        else:
            self.state = SystemState.CRITICAL
    
    def _allocate_resources(self, metrics: ResourceMetrics):
        """Allocate resources to virtual quantum computer"""
        with self.regulation_lock:
            # Calculate available resources
            available_cpu = max(0, metrics.cpu_percent - self.min_cpu_reserve)
            available_memory_mb = max(0, metrics.memory_available_mb - self.min_memory_reserve_mb)
            
            # State-based allocation
            if self.state == SystemState.IDLE:
                # Use more resources when idle
                self.quantum_cpu_limit = min(available_cpu, self.max_cpu_percent * 0.9)
                self.quantum_memory_limit_mb = available_memory_mb * 0.9
                self.quantum_cores_allocated = int(self.optimal_threads * 0.9)
            elif self.state == SystemState.NORMAL:
                # Normal allocation
                self.quantum_cpu_limit = min(available_cpu, self.max_cpu_percent * 0.7)
                self.quantum_memory_limit_mb = available_memory_mb * 0.7
                self.quantum_cores_allocated = int(self.optimal_threads * 0.7)
            elif self.state == SystemState.STRESSED:
                # Reduced allocation
                self.quantum_cpu_limit = min(available_cpu, self.max_cpu_percent * 0.5)
                self.quantum_memory_limit_mb = available_memory_mb * 0.5
                self.quantum_cores_allocated = int(self.optimal_threads * 0.5)
            else:  # CRITICAL or RECOVERING
                # Minimal allocation
                self.quantum_cpu_limit = min(available_cpu, self.max_cpu_percent * 0.2)
                self.quantum_memory_limit_mb = available_memory_mb * 0.2
                self.quantum_cores_allocated = max(1, int(self.optimal_threads * 0.2))
    
    def _regulate_tasks(self, metrics: ResourceMetrics):
        """Regulate active quantum tasks based on system state"""
        if self.state == SystemState.CRITICAL:
            # Throttle all tasks
            for task_id, task in list(self.active_tasks.items()):
                if task.status == "running":
                    task.status = "throttled"
                    self.performance_metrics['tasks_throttled'] += 1
        elif self.state == SystemState.STRESSED:
            # Throttle low-priority tasks
            for task_id, task in list(self.active_tasks.items()):
                if task.status == "running" and task.priority < 5:
                    task.status = "throttled"
                    self.performance_metrics['tasks_throttled'] += 1
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get currently available resources for quantum computing"""
        with self.regulation_lock:
            return {
                'cpu_limit_percent': self.quantum_cpu_limit,
                'memory_limit_mb': self.quantum_memory_limit_mb,
                'cores_allocated': self.quantum_cores_allocated,
                'system_state': self.state.value,
                'optimal_threads': self.optimal_threads
            }
    
    def can_accept_task(self, cores: int, memory_mb: float) -> bool:
        """Check if a task can be accepted"""
        resources = self.get_available_resources()
        return (
            cores <= resources['cores_allocated'] and
            memory_mb <= resources['memory_limit_mb'] and
            self.state != SystemState.CRITICAL
        )
    
    def submit_quantum_task(
        self,
        task_id: str,
        cores: int,
        memory_mb: float,
        priority: int = 5,
        callback: Optional[Callable] = None
    ) -> bool:
        """Submit a quantum computing task"""
        if not self.can_accept_task(cores, memory_mb):
            return False
        
        task = QuantumTask(
            task_id=task_id,
            priority=priority,
            cores_requested=cores,
            memory_requested_mb=memory_mb,
            estimated_duration=0.0,
            callback=callback,
            status="pending"
        )
        
        # Use negative priority for max-heap behavior
        self.quantum_tasks.put((-priority, time.time(), task))
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        
        return {
            'state': self.state.value,
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'available_cores': latest.cpu_cores_available,
            'available_memory_mb': latest.memory_available_mb,
            'quantum_resources': self.get_available_resources(),
            'active_tasks': len(self.active_tasks),
            'performance_metrics': self.performance_metrics.copy()
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_regulation()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_regulation()


# Example usage
if __name__ == '__main__':
    print("Medulla Oblongata System")
    print("="*80)
    
    # Create Medulla system
    medulla = MedullaOblongataSystem(
        max_cpu_percent=80.0,
        max_memory_percent=75.0,
        min_cpu_reserve=20.0,
        min_memory_reserve_mb=1024.0
    )
    
    # Start regulation
    with medulla:
        print("\n[Medulla] System started")
        
        # Monitor for a few seconds
        for i in range(10):
            status = medulla.get_system_status()
            print(f"\n[{i+1}] System Status:")
            print(f"  State: {status['state']}")
            print(f"  CPU: {status['cpu_percent']:.1f}%")
            print(f"  Memory: {status['memory_percent']:.1f}%")
            print(f"  Quantum Resources:")
            print(f"    CPU Limit: {status['quantum_resources']['cpu_limit_percent']:.1f}%")
            print(f"    Memory Limit: {status['quantum_resources']['memory_limit_mb']:.1f} MB")
            print(f"    Cores Allocated: {status['quantum_resources']['cores_allocated']}")
            
            time.sleep(1)
        
        print("\n[Medulla] System stopped")
