# Medulla Auto-Start Guide

## ✅ **Yes - Medulla Automatically Runs with ML Toolbox!**

The Medulla Oblongata System now **automatically starts** when you create an ML Toolbox instance.

---

## 🚀 **Automatic Startup**

### **Default Behavior:**

```python
from ml_toolbox import MLToolbox

# Medulla automatically starts!
toolbox = MLToolbox()

# Medulla is running and regulating resources
status = toolbox.get_system_status()
print(f"System State: {status['state']}")
print(f"CPU: {status['cpu_percent']:.1f}%")
print(f"Quantum Resources: {status['quantum_resources']}")
```

### **Disable Auto-Start (if needed):**

```python
# Disable Medulla auto-start
toolbox = MLToolbox(auto_start_medulla=False)
```

---

## 📊 **What Happens Automatically**

1. **Medulla Starts** - Resource regulation begins immediately
2. **System Monitoring** - CPU, memory, and load are monitored
3. **Resource Allocation** - Resources allocated to quantum computing
4. **System Protection** - Reserves resources for OS (20% CPU, 1GB RAM)

---

## 🎯 **Usage Examples**

### **Basic Usage:**

```python
from ml_toolbox import MLToolbox

# Create toolbox (Medulla auto-starts)
toolbox = MLToolbox()

# Check system status
status = toolbox.get_system_status()
print(f"State: {status['state']}")
print(f"CPU: {status['cpu_percent']:.1f}%")
print(f"Memory: {status['memory_percent']:.1f}%")
```

### **With Quantum Computer:**

```python
# Get quantum computer (uses Medulla resources)
qc = toolbox.get_quantum_computer(num_qubits=8)

# Perform quantum operations
qc.apply_gate('H', 0)
qc.apply_gate('X', 1)

# Operations use Medulla-regulated resources
```

### **Context Manager (Auto-Cleanup):**

```python
# Medulla auto-stops when exiting context
with MLToolbox() as toolbox:
    # Use toolbox
    qc = toolbox.get_quantum_computer()
    qc.apply_gate('H', 0)
    # Medulla stops automatically on exit
```

---

## 🔍 **Accessing Medulla**

### **Direct Access:**

```python
toolbox = MLToolbox()

# Access Medulla instance
if toolbox.medulla:
    print(f"Medulla running: {toolbox.medulla.regulation_running}")
    print(f"System state: {toolbox.medulla.state.value}")
```

### **Via Infrastructure Compartment:**

```python
# Medulla is also in Infrastructure compartment
if 'medulla' in toolbox.infrastructure.components:
    medulla = toolbox.infrastructure.components['medulla']
```

---

## 📈 **System Status**

### **Get Status:**

```python
status = toolbox.get_system_status()

# Returns:
# {
#   'state': 'idle' | 'normal' | 'stressed' | 'critical',
#   'cpu_percent': 45.2,
#   'memory_percent': 62.1,
#   'quantum_resources': {
#     'cpu_limit_percent': 56.0,
#     'memory_limit_mb': 2048.0,
#     'cores_allocated': 6
#   },
#   'active_tasks': 0,
#   'performance_metrics': {...}
# }
```

---

## ✅ **Benefits**

1. **No Manual Setup** - Medulla starts automatically
2. **Always Available** - System status always accessible
3. **Resource Regulation** - Automatic resource management
4. **System Protection** - Prevents system overload
5. **Clean Cleanup** - Context manager auto-stops Medulla

---

## 🎯 **Integration Points**

The Medulla system integrates with:

1. **ML Toolbox** - Auto-starts on initialization
2. **Infrastructure Compartment** - Medulla available as component
3. **Virtual Quantum Computer** - Uses Medulla resources
4. **Architecture Optimizer** - Optimal thread counts
5. **Optimized ML Operations** - Resource-aware operations

---

## 📊 **System States**

| State | CPU Allocation | Memory Allocation | Cores | Trigger |
|-------|---------------|-------------------|-------|---------|
| **IDLE** | 90% | 90% | 90% | CPU < 30%, Memory < 50% |
| **NORMAL** | 70% | 70% | 70% | CPU < 80%, Memory < 75% |
| **STRESSED** | 50% | 50% | 50% | CPU < 90%, Memory < 90% |
| **CRITICAL** | 20% | 20% | 20% | CPU >= 90% or Memory >= 90% |

---

## 🚀 **Next Steps**

1. **Use ML Toolbox** - Medulla starts automatically
2. **Check Status** - Use `get_system_status()`
3. **Use Quantum Computer** - Get via `get_quantum_computer()`
4. **Monitor Resources** - Check system state regularly

---

## ✅ **Summary**

**YES - Medulla automatically runs when using the ML Toolbox!**

- ✅ Starts automatically on `MLToolbox()` creation
- ✅ Regulates system resources continuously
- ✅ Available via `toolbox.medulla` and `toolbox.get_system_status()`
- ✅ Integrates with Virtual Quantum Computer
- ✅ Auto-stops with context manager

**No manual setup required - just use the toolbox!** 🚀

---

**Files:**
- `ml_toolbox/__init__.py` - Auto-start integration
- `ml_toolbox/compartment2_infrastructure.py` - Medulla component
- `test_medulla_auto_start.py` - Test script
- `MEDULLA_AUTO_START_GUIDE.md` - This guide
