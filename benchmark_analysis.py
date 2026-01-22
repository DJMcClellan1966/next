"""
Benchmark Analysis and Improvement Recommendations
Analyzes benchmark results and provides detailed improvement suggestions
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def analyze_benchmark_results(results_file: str = "benchmark_results.json") -> Dict[str, Any]:
    """Analyze benchmark results and generate improvement recommendations"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    analysis = {
        'overall_performance': {},
        'detailed_analysis': [],
        'improvements': [],
        'comparisons': []
    }
    
    # Overall performance
    all_times = []
    all_accuracies = []
    all_r2_scores = []
    successful_tests = 0
    failed_tests = 0
    
    for benchmark in results.get('benchmarks', []):
        for model in benchmark.get('models', []):
            if model.get('status') == 'success':
                successful_tests += 1
                
                if 'training_time' in model:
                    all_times.append(model['training_time'])
                
                if 'accuracy' in model:
                    all_accuracies.append(model['accuracy'])
                
                if 'r2_score' in model:
                    all_r2_scores.append(model['r2_score'])
            elif model.get('status') == 'failed':
                failed_tests += 1
    
    analysis['overall_performance'] = {
        'success_rate': successful_tests / (successful_tests + failed_tests) * 100 if (successful_tests + failed_tests) > 0 else 0,
        'total_tests': successful_tests + failed_tests,
        'successful': successful_tests,
        'failed': failed_tests
    }
    
    if all_times:
        analysis['overall_performance']['speed'] = {
            'avg_time': np.mean(all_times),
            'min_time': np.min(all_times),
            'max_time': np.max(all_times),
            'median_time': np.median(all_times)
        }
    
    if all_accuracies:
        analysis['overall_performance']['accuracy'] = {
            'avg_accuracy': np.mean(all_accuracies),
            'min_accuracy': np.min(all_accuracies),
            'max_accuracy': np.max(all_accuracies),
            'median_accuracy': np.median(all_accuracies)
        }
    
    if all_r2_scores:
        analysis['overall_performance']['regression'] = {
            'avg_r2': np.mean(all_r2_scores),
            'min_r2': np.min(all_r2_scores),
            'max_r2': np.max(all_r2_scores)
        }
    
    # Detailed analysis per benchmark
    for benchmark in results.get('benchmarks', []):
        benchmark_analysis = {
            'name': benchmark.get('name'),
            'difficulty': benchmark.get('difficulty'),
            'performance': {},
            'issues': [],
            'recommendations': []
        }
        
        toolbox_model = None
        baseline_model = None
        
        for model in benchmark.get('models', []):
            if 'ML Toolbox' in model.get('name', ''):
                toolbox_model = model
            elif 'baseline' in model.get('name', '').lower() or 'scikit' in model.get('name', '').lower():
                baseline_model = model
        
        if toolbox_model and toolbox_model.get('status') == 'success':
            benchmark_analysis['performance']['toolbox'] = {
                'status': 'success',
                'metrics': {k: v for k, v in toolbox_model.items() 
                           if k not in ['name', 'status', 'error']}
            }
            
            # Compare with baseline
            if baseline_model and baseline_model.get('status') == 'success':
                comparison = {}
                
                if 'accuracy' in toolbox_model and 'accuracy' in baseline_model:
                    diff = toolbox_model['accuracy'] - baseline_model['accuracy']
                    comparison['accuracy_diff'] = diff
                    if diff < -0.05:
                        benchmark_analysis['issues'].append('Accuracy significantly lower than baseline')
                        benchmark_analysis['recommendations'].append('Improve model selection or hyperparameters')
                
                if 'training_time' in toolbox_model and 'training_time' in baseline_model:
                    speed_ratio = toolbox_model['training_time'] / baseline_model['training_time']
                    comparison['speed_ratio'] = speed_ratio
                    if speed_ratio > 1.5:
                        benchmark_analysis['issues'].append('Training time significantly slower than baseline')
                        benchmark_analysis['recommendations'].append('Optimize training pipeline and add caching')
                    elif speed_ratio < 0.8:
                        benchmark_analysis['performance']['toolbox']['faster_than_baseline'] = True
                
                benchmark_analysis['comparison'] = comparison
        
        elif toolbox_model and toolbox_model.get('status') == 'failed':
            benchmark_analysis['performance']['toolbox'] = {
                'status': 'failed',
                'error': toolbox_model.get('error', 'Unknown error')
            }
            benchmark_analysis['issues'].append(f"Test failed: {toolbox_model.get('error', 'Unknown error')}")
            benchmark_analysis['recommendations'].append('Fix error handling and improve robustness')
        
        analysis['detailed_analysis'].append(benchmark_analysis)
    
    # Overall improvements
    improvements = []
    
    # Speed improvements
    if all_times and np.mean(all_times) > 5:
        improvements.append({
            'category': 'Performance - Speed',
            'priority': 'High',
            'issue': f'Average training time is {np.mean(all_times):.2f}s (high)',
            'recommendations': [
                'Add model caching for repeated training',
                'Optimize data preprocessing pipeline',
                'Add parallel processing where possible',
                'Use more efficient data structures'
            ]
        })
    
    # Accuracy improvements
    if all_accuracies and np.mean(all_accuracies) < 0.85:
        improvements.append({
            'category': 'Performance - Accuracy',
            'priority': 'High',
            'issue': f'Average accuracy is {np.mean(all_accuracies):.4f} (could be improved)',
            'recommendations': [
                'Add hyperparameter tuning',
                'Improve feature engineering',
                'Add ensemble methods',
                'Better model selection algorithms'
            ]
        })
    
    # Reliability improvements
    if failed_tests > 0:
        improvements.append({
            'category': 'Reliability',
            'priority': 'Critical',
            'issue': f'{failed_tests} tests failed',
            'recommendations': [
                'Improve error handling',
                'Add input validation',
                'Better dependency management',
                'Add fallback mechanisms'
            ]
        })
    
    # Feature improvements
    improvements.append({
        'category': 'Features',
        'priority': 'Medium',
        'issue': 'Some advanced features not tested',
        'recommendations': [
            'Add deep learning benchmarks',
            'Test advanced preprocessing features',
            'Benchmark AutoML capabilities',
            'Test model registry and versioning',
            'Test pre-trained model hub'
        ]
    })
    
    analysis['improvements'] = improvements
    
    return analysis


def generate_improvement_report(analysis: Dict[str, Any]) -> str:
    """Generate detailed improvement report"""
    
    report = []
    report.append("=" * 80)
    report.append("ML Toolbox - Benchmark Analysis & Improvement Report")
    report.append("=" * 80)
    report.append("")
    
    # Overall Performance
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 80)
    perf = analysis['overall_performance']
    report.append(f"Success Rate: {perf.get('success_rate', 0):.1f}%")
    report.append(f"Total Tests: {perf.get('total_tests', 0)}")
    report.append(f"Successful: {perf.get('successful', 0)}")
    report.append(f"Failed: {perf.get('failed', 0)}")
    report.append("")
    
    if 'speed' in perf:
        speed = perf['speed']
        report.append("SPEED METRICS")
        report.append(f"  Average Training Time: {speed['avg_time']:.4f}s")
        report.append(f"  Min Training Time: {speed['min_time']:.4f}s")
        report.append(f"  Max Training Time: {speed['max_time']:.4f}s")
        report.append(f"  Median Training Time: {speed['median_time']:.4f}s")
        report.append("")
    
    if 'accuracy' in perf:
        acc = perf['accuracy']
        report.append("ACCURACY METRICS")
        report.append(f"  Average Accuracy: {acc['avg_accuracy']:.4f}")
        report.append(f"  Min Accuracy: {acc['min_accuracy']:.4f}")
        report.append(f"  Max Accuracy: {acc['max_accuracy']:.4f}")
        report.append(f"  Median Accuracy: {acc['median_accuracy']:.4f}")
        report.append("")
    
    # Detailed Analysis
    report.append("DETAILED BENCHMARK ANALYSIS")
    report.append("-" * 80)
    for bench_analysis in analysis['detailed_analysis']:
        report.append(f"\n{bench_analysis['name']} ({bench_analysis['difficulty']})")
        
        if 'toolbox' in bench_analysis['performance']:
            toolbox = bench_analysis['performance']['toolbox']
            if toolbox.get('status') == 'success':
                report.append("  Status: [SUCCESS]")
                if 'metrics' in toolbox:
                    for key, value in toolbox['metrics'].items():
                        if isinstance(value, float):
                            report.append(f"    {key}: {value:.4f}")
                        else:
                            report.append(f"    {key}: {value}")
            else:
                report.append("  Status: [FAILED]")
                report.append(f"    Error: {toolbox.get('error', 'Unknown')}")
        
        if bench_analysis.get('issues'):
            report.append("  Issues:")
            for issue in bench_analysis['issues']:
                report.append(f"    [WARNING] {issue}")
        
        if bench_analysis.get('recommendations'):
            report.append("  Recommendations:")
            for rec in bench_analysis['recommendations']:
                report.append(f"    [TIP] {rec}")
        
        if 'comparison' in bench_analysis:
            comp = bench_analysis['comparison']
            if 'accuracy_diff' in comp:
                report.append(f"  vs Baseline Accuracy: {comp['accuracy_diff']:+.4f}")
            if 'speed_ratio' in comp:
                report.append(f"  vs Baseline Speed: {comp['speed_ratio']:.2f}x")
    
    # Improvements
    report.append("")
    report.append("=" * 80)
    report.append("IMPROVEMENT RECOMMENDATIONS")
    report.append("=" * 80)
    
    for improvement in analysis['improvements']:
        report.append(f"\n[{improvement['priority']}] {improvement['category']}")
        report.append(f"Issue: {improvement['issue']}")
        report.append("Recommendations:")
        for rec in improvement['recommendations']:
            report.append(f"  - {rec}")
    
    return "\n".join(report)


if __name__ == '__main__':
    analysis = analyze_benchmark_results()
    report = generate_improvement_report(analysis)
    
    print(report)
    
    # Save analysis
    with open('benchmark_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    with open('improvement_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 80)
    print("Analysis saved to:")
    print("  - benchmark_analysis.json")
    print("  - improvement_report.txt")
