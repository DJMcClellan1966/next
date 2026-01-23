"""
Generate Comprehensive Statistics Report
From comprehensive test results
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def generate_statistics_report(results_file: str = 'comprehensive_test_results.json'):
    """Generate comprehensive statistics report"""
    
    # Load results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Results file {results_file} not found. Run comprehensive_ml_test_suite.py first.")
        return
    
    print("="*80)
    print("COMPREHENSIVE ML TEST STATISTICS REPORT")
    print("="*80)
    
    summary = results.get('summary', {})
    
    # Overall Statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"\nTotal Tests Run: {summary.get('total_tests', 0)}")
    print(f"ML Toolbox Wins: {summary.get('toolbox_wins', 0)} ({summary.get('toolbox_win_rate', 0):.1f}%)")
    print(f"scikit-learn Wins: {summary.get('sklearn_wins', 0)} ({summary.get('sklearn_win_rate', 0):.1f}%)")
    print(f"Ties: {summary.get('ties', 0)} ({summary.get('tie_rate', 0):.1f}%)")
    print(f"\nToolbox Errors: {summary.get('toolbox_errors', 0)}")
    print(f"sklearn Errors: {summary.get('sklearn_errors', 0)}")
    
    # Category Breakdown
    print("\n" + "="*80)
    print("CATEGORY BREAKDOWN")
    print("="*80)
    
    categories = summary.get('test_categories', {})
    for category, stats in categories.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Total Tests: {stats.get('total', 0)}")
        print(f"  Toolbox Wins: {stats.get('toolbox_wins', 0)}")
        print(f"  sklearn Wins: {stats.get('sklearn_wins', 0)}")
        print(f"  Ties: {stats.get('ties', 0)}")
    
    # Detailed Performance Comparison
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE COMPARISON")
    print("="*80)
    
    # Collect all test results
    all_tests = []
    for category in ['simple_tests', 'medium_tests', 'hard_tests', 'np_complete_tests']:
        if category in results:
            for test_name, test_results in results[category].items():
                if isinstance(test_results, dict):
                    toolbox = test_results.get('toolbox', {})
                    sklearn = test_results.get('sklearn', test_results.get('baseline', {}))
                    
                    if toolbox.get('success') and (sklearn.get('success') or 'distance' in sklearn or 'num_colors' in sklearn):
                        test_info = {
                            'category': category.replace('_tests', ''),
                            'test_name': test_name,
                            'toolbox_metric': toolbox.get('accuracy') or toolbox.get('r2_score') or toolbox.get('silhouette_score') or toolbox.get('distance') or toolbox.get('num_colors'),
                            'sklearn_metric': sklearn.get('accuracy') or sklearn.get('r2_score') or sklearn.get('silhouette_score') or sklearn.get('distance') or sklearn.get('num_colors'),
                            'toolbox_time': toolbox.get('time', 0),
                            'sklearn_time': sklearn.get('time', 0),
                            'winner': 'toolbox' if (toolbox.get('accuracy') or toolbox.get('r2_score') or 0) > (sklearn.get('accuracy') or sklearn.get('r2_score') or 0) else 'sklearn' if (sklearn.get('accuracy') or sklearn.get('r2_score') or 0) > (toolbox.get('accuracy') or toolbox.get('r2_score') or 0) else 'tie'
                        }
                        all_tests.append(test_info)
    
    # Create DataFrame for analysis
    if all_tests:
        df = pd.DataFrame(all_tests)
        
        print("\nPerformance Metrics:")
        print(df[['test_name', 'toolbox_metric', 'sklearn_metric', 'winner']].to_string(index=False))
        
        print("\nSpeed Comparison:")
        speed_df = df[['test_name', 'toolbox_time', 'sklearn_time']].copy()
        speed_df['speed_ratio'] = speed_df['sklearn_time'] / speed_df['toolbox_time']
        speed_df['faster'] = speed_df['speed_ratio'].apply(lambda x: 'toolbox' if x > 1 else 'sklearn' if x < 1 else 'tie')
        print(speed_df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        toolbox_wins = len(df[df['winner'] == 'toolbox'])
        sklearn_wins = len(df[df['winner'] == 'sklearn'])
        ties = len(df[df['winner'] == 'tie'])
        
        print(f"\nWins:")
        print(f"  Toolbox: {toolbox_wins} ({toolbox_wins/len(df)*100:.1f}%)")
        print(f"  sklearn: {sklearn_wins} ({sklearn_wins/len(df)*100:.1f}%)")
        print(f"  Ties: {ties} ({ties/len(df)*100:.1f}%)")
        
        # Average metrics
        if 'toolbox_metric' in df.columns and 'sklearn_metric' in df.columns:
            print(f"\nAverage Metrics:")
            print(f"  Toolbox: {df['toolbox_metric'].mean():.4f}")
            print(f"  sklearn: {df['sklearn_metric'].mean():.4f}")
        
        # Average speed
        print(f"\nAverage Speed:")
        print(f"  Toolbox: {df['toolbox_time'].mean():.4f}s")
        print(f"  sklearn: {df['sklearn_time'].mean():.4f}s")
        print(f"  Speed Ratio (sklearn/toolbox): {df['sklearn_time'].mean() / df['toolbox_time'].mean():.2f}x")
    
    # Save detailed report
    report_file = 'comprehensive_test_statistics.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE ML TEST STATISTICS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Tests: {summary.get('total_tests', 0)}\n")
        f.write(f"Toolbox Wins: {summary.get('toolbox_wins', 0)}\n")
        f.write(f"sklearn Wins: {summary.get('sklearn_wins', 0)}\n")
        f.write(f"Ties: {summary.get('ties', 0)}\n")
    
    print(f"\n\nDetailed report saved to: {report_file}")
    print("JSON results saved to: comprehensive_test_results.json")


if __name__ == '__main__':
    generate_statistics_report()
