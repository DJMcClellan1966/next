"""
Generate Medulla Performance Report
Creates a markdown report from performance comparison results
"""
import json
from pathlib import Path
from typing import Dict, Any

def generate_report():
    """Generate performance report"""
    results_file = Path('medulla_performance_comparison.json')
    
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        print("Run test_medulla_performance_impact.py first")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    report_file = Path('MEDULLA_PERFORMANCE_REPORT.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Medulla Performance Impact Report\n\n")
        f.write("## 🎯 **Executive Summary**\n\n")
        
        comparison = data.get('comparison', {})
        cpu_diff = comparison.get('cpu_diff_percent', 0)
        mem_diff = comparison.get('memory_diff_percent', 0)
        time_diff = comparison.get('time_diff_percent', 0)
        
        f.write("Comparison of ML Toolbox performance with and without Medulla Oblongata System.\n\n")
        
        f.write("### **Key Findings:**\n\n")
        f.write(f"- **CPU Impact:** {cpu_diff:+.1f}% change\n")
        f.write(f"- **Memory Impact:** {mem_diff:+.1f}% change\n")
        f.write(f"- **Time Impact:** {time_diff:+.1f}% change\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 **System Performance Metrics**\n\n")
        
        without = data.get('without_medulla', {}).get('system_performance', {})
        with_med = data.get('with_medulla', {}).get('system_performance', {})
        
        f.write("| Metric | Without Medulla | With Medulla | Difference |\n")
        f.write("|--------|-----------------|--------------|------------|\n")
        
        cpu_diff_val = with_med.get('avg_cpu', 0) - without.get('avg_cpu', 0)
        f.write(f"| Avg CPU % | {without.get('avg_cpu', 0):.1f}% | {with_med.get('avg_cpu', 0):.1f}% | {cpu_diff_val:+.1f}% |\n")
        
        max_cpu_diff = with_med.get('max_cpu', 0) - without.get('max_cpu', 0)
        f.write(f"| Max CPU % | {without.get('max_cpu', 0):.1f}% | {with_med.get('max_cpu', 0):.1f}% | {max_cpu_diff:+.1f}% |\n")
        
        mem_diff_val = with_med.get('avg_memory', 0) - without.get('avg_memory', 0)
        f.write(f"| Avg Memory % | {without.get('avg_memory', 0):.1f}% | {with_med.get('avg_memory', 0):.1f}% | {mem_diff_val:+.1f}% |\n")
        
        max_mem_diff = with_med.get('max_memory', 0) - without.get('max_memory', 0)
        f.write(f"| Max Memory % | {without.get('max_memory', 0):.1f}% | {with_med.get('max_memory', 0):.1f}% | {max_mem_diff:+.1f}% |\n")
        
        time_diff_val = data.get('with_medulla', {}).get('test_results', {}).get('total_time', 0) - \
                       data.get('without_medulla', {}).get('test_results', {}).get('total_time', 0)
        time_without = data.get('without_medulla', {}).get('test_results', {}).get('total_time', 0)
        time_with = data.get('with_medulla', {}).get('test_results', {}).get('total_time', 0)
        f.write(f"| Total Time (s) | {time_without:.2f}s | {time_with:.2f}s | {time_diff_val:+.2f}s |\n")
        
        f.write("\n---\n\n")
        f.write("## 🧪 **Test Results Comparison**\n\n")
        
        without_tests = data.get('without_medulla', {}).get('test_results', {}).get('tests', {})
        with_tests = data.get('with_medulla', {}).get('test_results', {}).get('tests', {})
        
        common_tests = set(without_tests.keys()) & set(with_tests.keys())
        
        for test_name in sorted(common_tests):
            test_without = without_tests[test_name]
            test_with = with_tests[test_name]
            
            if test_without.get('status') == 'success' and test_with.get('status') == 'success':
                f.write(f"### **{test_name.replace('_', ' ').title()}**\n\n")
                
                time_without = test_without.get('time', 0)
                time_with = test_with.get('time', 0)
                time_diff = time_with - time_without
                time_diff_pct = (time_diff / time_without * 100) if time_without > 0 else 0
                
                f.write(f"- **Time:** {time_without:.4f}s → {time_with:.4f}s ({time_diff:+.4f}s, {time_diff_pct:+.1f}%)\n")
                
                if 'accuracy' in test_without and 'accuracy' in test_with:
                    acc_without = test_without['accuracy']
                    acc_with = test_with['accuracy']
                    acc_diff = acc_with - acc_without
                    f.write(f"- **Accuracy:** {acc_without:.4f} → {acc_with:.4f} ({acc_diff:+.4f})\n")
                elif 'r2_score' in test_without and 'r2_score' in test_with:
                    r2_without = test_without['r2_score']
                    r2_with = test_with['r2_score']
                    r2_diff = r2_with - r2_without
                    f.write(f"- **R2 Score:** {r2_without:.4f} → {r2_with:.4f} ({r2_diff:+.4f})\n")
                
                f.write("\n")
        
        f.write("---\n\n")
        f.write("## ✅ **Conclusion**\n\n")
        
        if abs(cpu_diff) < 5 and abs(mem_diff) < 5 and abs(time_diff) < 5:
            f.write("Medulla has **minimal impact** on system performance.\n\n")
            f.write("- ✅ CPU usage: Minimal change\n")
            f.write("- ✅ Memory usage: Minimal change\n")
            f.write("- ✅ Execution time: Minimal change\n")
            f.write("- ✅ System stability: Maintained\n")
        elif cpu_diff < 0 and mem_diff < 0 and time_diff < 0:
            f.write("Medulla **improves** system performance.\n\n")
            f.write("- ✅ Lower CPU usage\n")
            f.write("- ✅ Lower memory usage\n")
            f.write("- ✅ Faster execution\n")
        else:
            f.write("Medulla has **moderate impact** on system performance.\n\n")
            if cpu_diff > 0:
                f.write(f"- ⚠️ CPU usage: {cpu_diff:+.1f}% higher (regulation overhead)\n")
            else:
                f.write(f"- ✅ CPU usage: {cpu_diff:+.1f}% lower\n")
            
            if mem_diff > 0:
                f.write(f"- ⚠️ Memory usage: {mem_diff:+.1f}% higher (regulation overhead)\n")
            else:
                f.write(f"- ✅ Memory usage: {mem_diff:+.1f}% lower\n")
            
            if time_diff > 0:
                f.write(f"- ⚠️ Execution time: {time_diff:+.1f}% slower (regulation overhead)\n")
            else:
                f.write(f"- ✅ Execution time: {time_diff:+.1f}% faster\n")
        
        f.write("\n**Recommendation:** ")
        if abs(cpu_diff) < 5 and abs(mem_diff) < 5:
            f.write("Medulla can be used with minimal performance impact. The resource regulation benefits outweigh the small overhead.\n")
        else:
            f.write("Consider Medulla for resource-intensive workloads where system stability is more important than raw performance.\n")
        
        f.write("\n---\n\n")
        f.write("**Generated from:** `medulla_performance_comparison.json`\n")
        f.write("**Test Script:** `test_medulla_performance_impact.py`\n")
    
    print(f"[OK] Report generated: {report_file}")


if __name__ == '__main__':
    generate_report()
