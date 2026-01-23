"""
Compare Performance: Before vs After Architecture Optimizations
"""
import json
from pathlib import Path
from typing import Dict, List, Any

def compare_with_architecture_optimizations():
    """Compare test results before and after architecture optimizations"""
    
    # Previous results (before architecture optimizations)
    previous_results = {
        'simple_tests': {
            'binary_classification': {'toolbox': 0.2213, 'sklearn': 0.0190},
            'multiclass_classification': {'toolbox': 0.2009, 'sklearn': 0.0332},
            'simple_regression': {'toolbox': 0.1598, 'sklearn': 0.0247},
            'basic_clustering': {'toolbox': 2.9877, 'sklearn': 0.0546}
        },
        'medium_tests': {
            'high_dim_classification': {'toolbox': 0.4148, 'sklearn': 0.0568},
            'imbalanced_classification': {'toolbox': 0.2241, 'sklearn': 0.0286},
            'time_series_regression': {'toolbox': 0.1367, 'sklearn': 0.0072},
            'multi_output_regression': {'toolbox': 0.1931, 'sklearn': 0.0209},
            'feature_selection': {'toolbox': 0.0203, 'sklearn': 0.0006}
        },
        'hard_tests': {
            'very_high_dim': {'toolbox': 0.9264, 'sklearn': 0.1442},
            'nonlinear_patterns': {'toolbox': 0.2418, 'sklearn': 0.0296},
            'sparse_data': {'toolbox': 0.1739, 'sklearn': 0.0109},
            'noisy_data': {'toolbox': 0.1475, 'sklearn': 0.0283},
            'ensemble': {'toolbox': 0.3368, 'sklearn': 0.0338}
        }
    }
    
    # Load current results if available
    current_results_file = Path('comprehensive_test_results.json')
    if current_results_file.exists():
        with open(current_results_file, 'r') as f:
            current_data = json.load(f)
        
        # Extract current results
        current_results = {}
        for category in ['simple_tests', 'medium_tests', 'hard_tests']:
            if category in current_data:
                current_results[category] = {}
                for test_name, test_data in current_data[category].items():
                    if 'toolbox' in test_data and 'sklearn' in test_data:
                        tb = test_data['toolbox']
                        sk = test_data['sklearn']
                        if 'time' in tb and 'time' in sk:
                            current_results[category][test_name] = {
                                'toolbox': tb['time'],
                                'sklearn': sk['time']
                            }
    else:
        print("Current test results not found. Run comprehensive_ml_test_suite.py first.")
        return
    
    print("="*80)
    print("ARCHITECTURE OPTIMIZATION IMPACT ANALYSIS")
    print("="*80)
    print()
    
    improvements = []
    regressions = []
    no_change = []
    
    for category in ['simple_tests', 'medium_tests', 'hard_tests']:
        if category not in current_results:
            continue
        
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-"*80)
        print(f"{'Test':<30} {'Before':<12} {'After':<12} {'Change':<12} {'vs sklearn':<15}")
        print("-"*80)
        
        for test_name in previous_results[category].keys():
            if test_name not in current_results[category]:
                continue
            
            prev_time = previous_results[category][test_name]['toolbox']
            curr_time = current_results[category][test_name]['toolbox']
            sklearn_time = current_results[category][test_name]['sklearn']
            
            if prev_time > 0:
                change = ((prev_time - curr_time) / prev_time) * 100
                vs_sklearn_before = prev_time / sklearn_time if sklearn_time > 0 else 0
                vs_sklearn_after = curr_time / sklearn_time if sklearn_time > 0 else 0
                
                if abs(change) < 1.0:  # Less than 1% change
                    no_change.append({
                        'category': category,
                        'test': test_name,
                        'time': curr_time,
                        'vs_sklearn': vs_sklearn_after
                    })
                    status = "~"
                elif change > 0:
                    improvements.append({
                        'category': category,
                        'test': test_name,
                        'before': prev_time,
                        'after': curr_time,
                        'improvement': change,
                        'vs_sklearn_before': vs_sklearn_before,
                        'vs_sklearn_after': vs_sklearn_after
                    })
                    status = "+"
                else:
                    regressions.append({
                        'category': category,
                        'test': test_name,
                        'before': prev_time,
                        'after': curr_time,
                        'regression': abs(change),
                        'vs_sklearn_before': vs_sklearn_before,
                        'vs_sklearn_after': vs_sklearn_after
                    })
                    status = "-"
                
                print(f"{test_name:<30} {prev_time:<12.4f} {curr_time:<12.4f} {change:<11.1f}% {status} {vs_sklearn_after:<14.2f}x")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nImprovements: {len(improvements)} tests")
    if improvements:
        avg_improvement = sum(i['improvement'] for i in improvements) / len(improvements)
        avg_vs_sklearn_improvement = sum(i['vs_sklearn_before'] - i['vs_sklearn_after'] for i in improvements) / len(improvements)
        print(f"  Average improvement: {avg_improvement:.1f}% faster")
        print(f"  Average vs sklearn improvement: {avg_vs_sklearn_improvement:.2f}x better")
        
        print("\nTop Improvements:")
        for imp in sorted(improvements, key=lambda x: x['improvement'], reverse=True)[:5]:
            print(f"  - {imp['test']}: {imp['improvement']:.1f}% faster")
            print(f"    vs sklearn: {imp['vs_sklearn_before']:.2f}x -> {imp['vs_sklearn_after']:.2f}x")
    
    print(f"\nRegressions: {len(regressions)} tests")
    if regressions:
        avg_regression = sum(r['regression'] for r in regressions) / len(regressions)
        print(f"  Average regression: {avg_regression:.1f}% slower")
    
    print(f"\nNo Significant Change: {len(no_change)} tests")
    
    # Overall comparison
    print("\n" + "="*80)
    print("OVERALL IMPACT")
    print("="*80)
    
    # Calculate average vs sklearn
    prev_avg_vs_sklearn = []
    curr_avg_vs_sklearn = []
    
    for category in ['simple_tests', 'medium_tests', 'hard_tests']:
        if category not in current_results:
            continue
        
        for test_name in previous_results[category].keys():
            if test_name not in current_results[category]:
                continue
            
            prev_time = previous_results[category][test_name]['toolbox']
            curr_time = current_results[category][test_name]['toolbox']
            sklearn_time = current_results[category][test_name]['sklearn']
            
            if sklearn_time > 0:
                prev_avg_vs_sklearn.append(prev_time / sklearn_time)
                curr_avg_vs_sklearn.append(curr_time / sklearn_time)
    
    if prev_avg_vs_sklearn and curr_avg_vs_sklearn:
        prev_avg = sum(prev_avg_vs_sklearn) / len(prev_avg_vs_sklearn)
        curr_avg = sum(curr_avg_vs_sklearn) / len(curr_avg_vs_sklearn)
        improvement = ((prev_avg - curr_avg) / prev_avg) * 100
        
        print(f"\nAverage vs sklearn (before): {prev_avg:.2f}x slower")
        print(f"Average vs sklearn (after): {curr_avg:.2f}x slower")
        print(f"Improvement: {improvement:.1f}% closer to sklearn")
        
        if improvement > 0:
            print(f"\n[SUCCESS] Architecture optimizations improved performance!")
            print(f"  Toolbox is now {improvement:.1f}% closer to sklearn performance")
        elif improvement < -5:
            print(f"\n[NOTE] Some variance detected (may be test variance)")
        else:
            print(f"\n[NOTE] Minimal change detected")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if len(improvements) > len(regressions):
        print("[SUCCESS] Architecture optimizations show positive impact!")
        print(f"   {len(improvements)} tests improved vs {len(regressions)} regressions")
    elif len(improvements) == len(regressions):
        print("[WARNING] Mixed results - some improvements, some regressions")
        print("   (May be due to test variance)")
    else:
        print("[WARNING] More regressions than improvements")
        print("   (May need more optimization or test variance)")


if __name__ == '__main__':
    compare_with_architecture_optimizations()
