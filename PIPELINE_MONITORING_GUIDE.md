# Pipeline Bottleneck Monitoring Guide

## 🎯 **Overview**

Comprehensive monitoring system to track CPU/memory usage and identify bottlenecks in the ML pipeline.

---

## 📊 **Features**

### **1. CPU and Memory Tracking**
- Real-time CPU usage monitoring
- Memory usage tracking (RSS, peak, average)
- Per-function resource tracking
- Per-pipeline-stage resource tracking

### **2. Bottleneck Identification**
- Automatic bottleneck detection (>threshold% of time)
- Slow pipeline stage identification
- Function-level performance analysis
- Resource usage trends

### **3. Comprehensive Reporting**
- Resource usage summaries
- Top bottlenecks by time
- Slow pipeline stages
- Function execution statistics
- CPU/memory usage per function/stage

---

## 🚀 **Usage**

### **Basic Function Monitoring**

```python
from pipeline_bottleneck_monitor import PipelineBottleneckMonitor, monitor_function

# Create monitor
monitor = PipelineBottleneckMonitor(sample_interval=0.1)

# Monitor a function
@monitor_function(monitor, 'my_function')
def my_function():
    # Your code here
    return result

# Run function
result = my_function()

# Generate report
report = monitor.generate_report()
print(report)
```

### **Pipeline Stage Monitoring**

```python
from pipeline_bottleneck_monitor import PipelineBottleneckMonitor

monitor = PipelineBottleneckMonitor()
monitor.start_monitoring()

# Monitor pipeline stages
with monitor.monitor_pipeline_stage('data_preprocessing'):
    # Preprocessing code
    preprocessed_data = preprocess(data)

with monitor.monitor_pipeline_stage('model_training'):
    # Training code
    model = train_model(preprocessed_data)

monitor.stop_monitoring()

# Generate report
report = monitor.generate_report()
print(report)
```

### **ML Toolbox Integration**

```python
from monitor_ml_pipeline import monitor_comprehensive_pipeline

# Monitor complete ML pipeline
monitor = monitor_comprehensive_pipeline()

# Get bottlenecks
bottlenecks = monitor.identify_bottlenecks(threshold_percent=5.0)
for bottleneck in bottlenecks:
    print(f"{bottleneck['function']}: {bottleneck['percent_time']:.1f}% of time")
```

---

## 📈 **Monitoring Capabilities**

### **1. Function-Level Monitoring**

Tracks for each function:
- Execution time (total, average, min, max)
- Number of calls
- CPU usage (per call, average, peak)
- Memory usage (per call, average, peak)

### **2. Pipeline Stage Monitoring**

Tracks for each pipeline stage:
- Duration
- CPU usage (average, peak)
- Memory usage (average, peak)
- Resource usage over time

### **3. Bottleneck Identification**

Identifies:
- Functions taking >threshold% of total time
- Slow pipeline stages
- High CPU/memory usage areas
- Optimization opportunities

---

## 🔍 **Bottleneck Detection**

### **Automatic Detection**

```python
# Identify bottlenecks (>10% of total time)
bottlenecks = monitor.identify_bottlenecks(threshold_percent=10.0)

for bottleneck in bottlenecks:
    print(f"Function: {bottleneck['function']}")
    print(f"  Time: {bottleneck['total_time']:.3f}s ({bottleneck['percent_time']:.1f}%)")
    print(f"  Calls: {bottleneck['calls']}")
    print(f"  Peak Memory: {bottleneck['peak_memory_mb']:.1f} MB")
```

### **Slow Pipeline Stages**

```python
# Identify slow stages
slow_stages = monitor.identify_slow_pipeline_stages()

for stage in slow_stages:
    print(f"Stage: {stage['stage']}")
    print(f"  Duration: {stage['duration']:.3f}s")
    print(f"  Peak CPU: {stage['peak_cpu']:.1f}%")
    print(f"  Peak Memory: {stage['peak_memory_mb']:.1f} MB")
```

---

## 📊 **Report Generation**

### **Comprehensive Report**

```python
report = monitor.generate_report()
print(report)

# Save to file
with open('monitoring_report.txt', 'w') as f:
    f.write(report)
```

### **Report Contents**

1. **Resource Usage Summary**
   - Current CPU/memory usage
   - Total function time
   - Total pipeline time
   - Number of functions/stages monitored

2. **Bottlenecks**
   - Top functions by time
   - Percentage of total time
   - Peak memory usage
   - Number of calls

3. **Slow Pipeline Stages**
   - Duration
   - Peak CPU/memory
   - Average CPU/memory

4. **Top Functions by Time**
   - Total time
   - Average time
   - Number of calls

---

## 🎯 **Example: Monitoring Data Preprocessing**

```python
from pipeline_bottleneck_monitor import PipelineBottleneckMonitor
from ml_toolbox import MLToolbox

monitor = PipelineBottleneckMonitor(sample_interval=0.05)
monitor.start_monitoring()

toolbox = MLToolbox()
data = toolbox.data

test_data = ["text1", "text2", ...] * 100

# Monitor preprocessing
with monitor.monitor_pipeline_stage('preprocessing'):
    preprocessor = data.get_preprocessor(advanced=True)
    results = preprocessor.preprocess(test_data)

monitor.stop_monitoring()

# Get bottlenecks
bottlenecks = monitor.identify_bottlenecks(threshold_percent=5.0)
print("Bottlenecks:")
for b in bottlenecks:
    print(f"  {b['function']}: {b['percent_time']:.1f}%")

# Generate report
report = monitor.generate_report()
print(report)
```

---

## 🔧 **Advanced Usage**

### **Profiling Functions**

```python
# Profile a function with cProfile
result, profile_output = monitor.profile_function(my_function, arg1, arg2)

# Profile output contains detailed timing information
print(profile_output)
```

### **Resource Usage Summary**

```python
summary = monitor.get_resource_usage_summary()

print(f"Current CPU: {summary['current_cpu_percent']:.1f}%")
print(f"Current Memory: {summary['current_memory_mb']:.1f} MB")
print(f"Total Function Time: {summary['total_function_time']:.3f}s")
print(f"Bottlenecks Found: {summary['bottlenecks_found']}")
```

---

## 📈 **Interpreting Results**

### **Bottleneck Thresholds**

- **>50% of time:** Critical bottleneck - optimize immediately
- **>25% of time:** Major bottleneck - high priority optimization
- **>10% of time:** Moderate bottleneck - consider optimization
- **>5% of time:** Minor bottleneck - optimize if time permits

### **CPU Usage**

- **>80%:** High CPU usage - may benefit from parallelization
- **>50%:** Moderate CPU usage
- **<50%:** Low CPU usage - may be I/O bound

### **Memory Usage**

- **Peak > 1GB:** High memory usage - consider memory optimization
- **Peak > 500MB:** Moderate memory usage
- **Peak < 500MB:** Low memory usage

---

## ✅ **Best Practices**

1. **Monitor Before Optimizing**
   - Always profile first to identify actual bottlenecks
   - Don't optimize what's not slow

2. **Use Appropriate Thresholds**
   - Start with 10% threshold for bottlenecks
   - Adjust based on your needs

3. **Monitor Complete Pipelines**
   - Monitor end-to-end workflows
   - Identify stage-level bottlenecks

4. **Track Trends**
   - Monitor over multiple runs
   - Identify performance regressions

5. **Save Reports**
   - Save monitoring reports for comparison
   - Track improvements over time

---

## 🎯 **Integration with ML Toolbox**

The monitoring system integrates seamlessly with ML Toolbox:

```python
from monitor_ml_pipeline import (
    monitor_data_preprocessing,
    monitor_ml_training,
    monitor_comprehensive_pipeline
)

# Monitor specific operations
monitor = monitor_data_preprocessing()
monitor = monitor_ml_training()
monitor = monitor_comprehensive_pipeline()
```

---

## 📊 **Example Output**

```
================================================================================
PIPELINE BOTTLENECK MONITORING REPORT
================================================================================

RESOURCE USAGE SUMMARY
--------------------------------------------------------------------------------
Current CPU: 45.2%
Current Memory: 512.3 MB
Total Function Time: 2.345s
Total Function Calls: 150
Total Pipeline Time: 2.500s

BOTTLENECKS (Top Functions)
--------------------------------------------------------------------------------
Function                                 % Time     Total Time   Calls    Peak Memory
--------------------------------------------------------------------------------
quantum_kernel.embed                     35.2       0.825s       100      128.5 MB
data_preprocessor._deduplicate_semantic  28.5       0.668s       1        256.3 MB
similarity_computation                    15.3       0.359s       50       64.2 MB

SLOW PIPELINE STAGES
--------------------------------------------------------------------------------
Stage                                    Duration    Peak CPU     Peak Memory
--------------------------------------------------------------------------------
data_preprocessing                        1.250s      65.3%        512.3 MB
model_training                            0.850s      45.2%        256.1 MB
```

---

## 🚀 **Next Steps**

1. **Run Monitoring**
   ```bash
   python monitor_ml_pipeline.py
   ```

2. **Review Reports**
   - Check `pipeline_monitoring_report.txt`
   - Identify bottlenecks
   - Review resource usage

3. **Optimize Bottlenecks**
   - Focus on top bottlenecks first
   - Apply vectorization, parallelization
   - Re-monitor to verify improvements

---

**Files:**
- `pipeline_bottleneck_monitor.py` - Core monitoring system
- `monitor_ml_pipeline.py` - ML Toolbox integration
- `PIPELINE_MONITORING_GUIDE.md` - This guide

**Status:** Ready to use for bottleneck identification and optimization!
