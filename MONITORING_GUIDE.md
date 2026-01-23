# ML Toolbox Resource Monitoring Guide

## 🎯 **Overview**

The ML Resource Monitoring System tracks CPU and memory usage to identify resource bottlenecks in your ML pipelines. It provides real-time monitoring and comprehensive analysis to optimize resource utilization.

---

## 🚀 **Quick Start**

### **Basic Usage**

```python
from ml_monitor import ResourceMonitor

# Create monitor
monitor = ResourceMonitor(sample_interval=0.1)

# Start monitoring
monitor.start_monitoring()

# Monitor a function
@monitor.monitor_function
def my_ml_function(data):
    # Your ML code here
    return result

# Run function
result = my_ml_function(data)

# Stop monitoring
monitor.stop_monitoring()

# Generate report
report = monitor.generate_report('monitoring_report.txt')
print(report)

# Get bottlenecks
bottlenecks = monitor.identify_resource_bottlenecks()
for bottleneck in bottlenecks:
    print(f"{bottleneck['type']}: {bottleneck['value']:.2f}")
```

### **Get Current Usage**

```python
monitor = ResourceMonitor()
usage = monitor.get_current_usage()

print(f"CPU: {usage['cpu_percent']:.2f}%")
print(f"Memory: {usage['memory_mb']:.2f} MB")
```

---

## 📊 **Features**

### **1. Real-Time Monitoring**

Continuous background monitoring of CPU and memory:

```python
monitor = ResourceMonitor(sample_interval=0.1)
monitor.start_monitoring()

# Your code runs here
# Monitor collects samples in background

monitor.stop_monitoring()
stats = monitor.get_statistics()
```

**Tracks:**
- CPU usage (percent)
- Memory usage (MB)
- Time-series data
- Statistical metrics

### **2. Function-Level Monitoring**

Monitor individual functions:

```python
@monitor.monitor_function
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
```

**Tracks:**
- CPU usage per function call
- Memory usage per function call
- Memory deltas (memory increase)
- Call counts

### **3. Resource Statistics**

Get detailed statistics:

```python
stats = monitor.get_statistics()

print(f"CPU Mean: {stats['cpu']['mean']:.2f}%")
print(f"CPU Max: {stats['cpu']['max']:.2f}%")
print(f"CPU P95: {stats['cpu']['p95']:.2f}%")
print(f"Memory Mean: {stats['memory']['mean_mb']:.2f} MB")
print(f"Memory Max: {stats['memory']['max_mb']:.2f} MB")
```

### **4. Bottleneck Identification**

Automatically identify resource bottlenecks:

```python
bottlenecks = monitor.identify_resource_bottlenecks(
    cpu_threshold=80.0,
    memory_threshold_mb=1000.0
)

for bottleneck in bottlenecks:
    print(f"Type: {bottleneck['type']}")
    print(f"Severity: {bottleneck['severity']}")
    print(f"Value: {bottleneck['value']:.2f}")
    print(f"Recommendation: {bottleneck['recommendation']}")
```

**Identifies:**
- CPU bottlenecks (high CPU usage)
- Memory bottlenecks (high memory usage)
- Function-level bottlenecks
- System-level bottlenecks
- Severity classification

### **5. Comprehensive Reports**

Generate detailed monitoring reports:

```python
report = monitor.generate_report('monitoring_report.txt')
```

**Report Includes:**
- System resource usage statistics
- Function-level resource usage
- Bottleneck analysis
- Optimization recommendations

---

## 🔍 **Monitoring ML Toolbox Components**

### **Monitor All Components**

Run comprehensive monitoring:

```bash
python monitor_ml_toolbox.py
```

This monitors:
- Data preprocessing
- Model training
- Full ML pipeline

### **Monitor Specific Component**

```python
from monitor_ml_toolbox import monitor_data_preprocessing

monitor = monitor_data_preprocessing()
report = monitor.generate_report()
print(report)
```

---

## 💡 **Best Practices**

### **1. Monitor Before Optimizing**

Always monitor first to identify actual bottlenecks:

```python
monitor = ResourceMonitor()
monitor.start_monitoring()

# Run your ML pipeline
result = run_ml_pipeline()

monitor.stop_monitoring()

# Check bottlenecks
bottlenecks = monitor.identify_resource_bottlenecks()
```

### **2. Use Appropriate Sampling Interval**

Balance accuracy vs. overhead:

```python
# High-frequency monitoring (more accurate, more overhead)
monitor = ResourceMonitor(sample_interval=0.05)

# Standard monitoring (balanced)
monitor = ResourceMonitor(sample_interval=0.1)

# Low-frequency monitoring (less overhead, less accurate)
monitor = ResourceMonitor(sample_interval=0.5)
```

### **3. Monitor Real Workloads**

Use production-like data and scenarios:

```python
# Use realistic data
real_data = load_production_data()
monitor = ResourceMonitor()
monitor.start_monitoring()

@monitor.monitor_function
def process_data(data):
    return preprocess(data)

result = process_data(real_data)
monitor.stop_monitoring()
```

### **4. Set Appropriate Thresholds**

Customize thresholds for your environment:

```python
# For CPU-intensive workloads
bottlenecks = monitor.identify_resource_bottlenecks(
    cpu_threshold=70.0,  # Lower threshold
    memory_threshold_mb=2000.0
)

# For memory-intensive workloads
bottlenecks = monitor.identify_resource_bottlenecks(
    cpu_threshold=80.0,
    memory_threshold_mb=500.0  # Lower threshold
)
```

### **5. Combine with Profiling**

Use monitoring with profiling for complete analysis:

```python
from ml_profiler import MLProfiler
from ml_monitor import ResourceMonitor

profiler = MLProfiler()
monitor = ResourceMonitor()

# Profile and monitor simultaneously
@profiler.profile_function
@monitor.monitor_function
def my_function():
    # Your code
    pass
```

---

## 🎯 **Resource Bottleneck Types**

### **CPU Bottlenecks**

**Symptoms:**
- High CPU usage (>80%)
- Slow execution
- CPU-bound operations

**Recommendations:**
- Parallelize operations
- Optimize algorithms
- Use batch processing
- Consider GPU acceleration

### **Memory Bottlenecks**

**Symptoms:**
- High memory usage (>1GB)
- Memory errors
- Slow performance due to swapping

**Recommendations:**
- Use memory-efficient data structures
- Implement data streaming
- Reduce batch sizes
- Clear unused variables
- Use generators instead of lists

---

## 📊 **Example Output**

### **Monitoring Report**

```
================================================================================
ML TOOLBOX RESOURCE MONITORING REPORT
================================================================================

Generated: 2024-01-20 15:30:00

SYSTEM RESOURCE USAGE
--------------------------------------------------------------------------------

CPU Usage:
  Mean: 45.23%
  Max: 95.67%
  P95: 87.34%
  Samples: 1,234

Memory Usage:
  Mean: 512.45 MB
  Max: 1,234.56 MB
  P95: 987.65 MB
  Samples: 1,234

Monitoring Duration: 123.45 seconds

FUNCTION-LEVEL RESOURCE USAGE
--------------------------------------------------------------------------------

data_preprocessor.AdvancedDataPreprocessor.clean_data:
  CPU: Mean 65.23%, Max 89.45% (100 calls)
  Memory: Mean 234.56 MB, Max 456.78 MB
  Memory Delta: 12.34 MB per call

ml_toolbox.algorithms.train_classifier:
  CPU: Mean 78.90%, Max 95.67% (50 calls)
  Memory: Mean 567.89 MB, Max 1,234.56 MB
  Memory Delta: 45.67 MB per call

RESOURCE BOTTLENECKS IDENTIFIED
--------------------------------------------------------------------------------

Found 3 resource bottlenecks:

1. CPU Bottleneck [HIGH]
   Level: function
   Function: ml_toolbox.algorithms.train_classifier
   Metric: cpu_percent
   Value: 78.90
   Threshold: 80.00
   Recommendation: Optimize ml_toolbox.algorithms.train_classifier or consider parallelization

2. MEMORY Bottleneck [HIGH]
   Level: function
   Function: ml_toolbox.algorithms.train_classifier
   Metric: memory_mb
   Value: 567.89
   Threshold: 1000.00
   Recommendation: Optimize ml_toolbox.algorithms.train_classifier memory usage or use memory-efficient algorithms

OPTIMIZATION RECOMMENDATIONS
--------------------------------------------------------------------------------

High CPU Usage:
  • Consider parallelization for CPU-intensive operations
  • Optimize algorithms to reduce computational complexity
  • Use batch processing to reduce overhead
  • Consider using GPU acceleration if available

High Memory Usage:
  • Use memory-efficient data structures
  • Implement data streaming for large datasets
  • Reduce batch sizes
  • Clear unused variables and cache
  • Consider using generators instead of lists
```

---

## 🔧 **Integration with ML Toolbox**

### **Automatic Monitoring**

Use `MonitoredMLToolbox` for automatic monitoring:

```python
from ml_monitor import MonitoredMLToolbox
from ml_toolbox import MLToolbox

toolbox = MLToolbox()
monitored = MonitoredMLToolbox(toolbox)

# All operations are automatically monitored
result = monitored.toolbox.algorithms.get_simple_ml_tasks().train_classifier(X, y)

# Get report
report = monitored.get_monitoring_report()
bottlenecks = monitored.get_resource_bottlenecks()

# Get current usage
usage = monitored.get_current_usage()
print(f"CPU: {usage['cpu_percent']:.2f}%")
```

### **Via ML Toolbox API**

```python
from ml_toolbox import MLToolbox

toolbox = MLToolbox()

# Get monitor
monitor = toolbox.algorithms.get_resource_monitor()
monitor.start_monitoring()

# Monitor operations
@monitor.monitor_function
def my_operation():
    return toolbox.algorithms.get_simple_ml_tasks().train_classifier(X, y)
```

---

## 📚 **Advanced Usage**

### **Custom Monitoring**

```python
monitor = ResourceMonitor(sample_interval=0.05, max_samples=50000)
monitor.start_monitoring()

# Your code
with monitor.profile_pipeline('custom_pipeline'):
    step1()
    step2()
    step3()

monitor.stop_monitoring()
```

### **Export Monitoring Data**

```python
monitor.export_data('monitoring_data.json')
```

### **Reset Monitor**

```python
monitor.reset()  # Clear all monitoring data
```

### **Get Function Statistics**

```python
# All functions
stats = monitor.get_function_statistics()

# Specific function
stats = monitor.get_function_statistics('my_function')
```

---

## 🎓 **Tips for Effective Monitoring**

1. **Monitor Continuously**
   - Start monitoring before operations
   - Stop after operations complete
   - Let monitor collect samples

2. **Use Appropriate Intervals**
   - High-frequency (0.05s): Detailed analysis
   - Standard (0.1s): Balanced
   - Low-frequency (0.5s): Less overhead

3. **Set Realistic Thresholds**
   - CPU: 70-80% for CPU-intensive
   - Memory: Based on available RAM
   - Adjust for your environment

4. **Combine with Profiling**
   - Profiling: Time-based bottlenecks
   - Monitoring: Resource-based bottlenecks
   - Together: Complete picture

5. **Monitor Production Workloads**
   - Use realistic data sizes
   - Use actual workloads
   - Monitor under normal conditions

---

## 📞 **Troubleshooting**

### **"psutil not available"**
- Install: `pip install psutil`
- Monitoring will be disabled without psutil

### **"Monitoring not collecting samples"**
- Ensure `start_monitoring()` was called
- Check `sample_interval` is appropriate
- Verify psutil is installed

### **"Reports are empty"**
- Ensure functions are being monitored
- Check monitoring data exists
- Verify monitoring was started

### **"High overhead from monitoring"**
- Increase `sample_interval`
- Reduce `max_samples`
- Monitor only critical sections

---

## 🔗 **Integration with Profiling**

Combine monitoring with profiling for complete analysis:

```python
from ml_profiler import MLProfiler
from ml_monitor import ResourceMonitor

profiler = MLProfiler()
monitor = ResourceMonitor()

# Start both
monitor.start_monitoring()

# Profile and monitor
@profiler.profile_function
@monitor.monitor_function
def my_function():
    # Your code
    pass

# Get both reports
profiling_report = profiler.generate_report()
monitoring_report = monitor.generate_report()

# Analyze together
time_bottlenecks = profiler.identify_bottlenecks()
resource_bottlenecks = monitor.identify_resource_bottlenecks()
```

---

**Happy Monitoring! 🚀**
