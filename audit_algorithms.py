"""
Algorithm Audit Script
Compares ML Toolbox algorithms with scikit-learn
Run this to see what's missing and what needs improvement
"""
import sys
from pathlib import Path
import inspect

sys.path.insert(0, str(Path(__file__).parent))

# Try to import scikit-learn
try:
    from sklearn import (
        linear_model, svm, tree, ensemble, naive_bayes, neighbors,
        cluster, decomposition, feature_selection, preprocessing,
        model_selection, metrics, discriminant_analysis
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

# Try to import ML Toolbox
try:
    from ml_toolbox import MLToolbox
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    print("Error: ML Toolbox not available")

# Scikit-learn algorithms to check
SKLEARN_ALGORITHMS = {
    'classification': {
        'linear_model': ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier', 'Perceptron', 'PassiveAggressiveClassifier'],
        'svm': ['SVC', 'LinearSVC', 'NuSVC'],
        'tree': ['DecisionTreeClassifier'],
        'ensemble': ['RandomForestClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier', 'ExtraTreesClassifier', 'VotingClassifier', 'BaggingClassifier'],
        'naive_bayes': ['GaussianNB', 'MultinomialNB', 'BernoulliNB', 'ComplementNB', 'CategoricalNB'],
        'neighbors': ['KNeighborsClassifier', 'RadiusNeighborsClassifier'],
        'neural_network': ['MLPClassifier'],
        'discriminant_analysis': ['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis'],
    },
    'regression': {
        'linear_model': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge', 'ARDRegression', 'TheilSenRegressor', 'HuberRegressor', 'RANSACRegressor', 'SGDRegressor', 'PassiveAggressiveRegressor'],
        'svm': ['SVR', 'LinearSVR', 'NuSVR'],
        'tree': ['DecisionTreeRegressor'],
        'ensemble': ['RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'VotingRegressor', 'BaggingRegressor'],
        'neighbors': ['KNeighborsRegressor', 'RadiusNeighborsRegressor'],
        'neural_network': ['MLPRegressor'],
    },
    'clustering': {
        'cluster': ['KMeans', 'DBSCAN', 'AgglomerativeClustering', 'SpectralClustering', 'MeanShift', 'AffinityPropagation', 'OPTICS', 'Birch', 'MiniBatchKMeans'],
    },
    'dimensionality_reduction': {
        'decomposition': ['PCA', 'TruncatedSVD', 'FactorAnalysis', 'FastICA', 'KernelPCA', 'IncrementalPCA', 'SparsePCA', 'MiniBatchSparsePCA', 'DictionaryLearning', 'MiniBatchDictionaryLearning'],
        'manifold': ['LocallyLinearEmbedding', 'Isomap', 'MDS', 'TSNE', 'SpectralEmbedding'],
    },
    'preprocessing': {
        'preprocessing': ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'Binarizer', 'PolynomialFeatures', 'OneHotEncoder', 'LabelEncoder', 'OrdinalEncoder', 'TargetEncoder', 'SimpleImputer', 'KNNImputer', 'IterativeImputer'],
    },
    'feature_selection': {
        'feature_selection': ['VarianceThreshold', 'SelectKBest', 'SelectPercentile', 'SelectFpr', 'SelectFdr', 'SelectFwe', 'GenericUnivariateSelect', 'RFE', 'RFECV', 'SelectFromModel'],
    },
    'model_selection': {
        'model_selection': ['KFold', 'StratifiedKFold', 'TimeSeriesSplit', 'GroupKFold', 'ShuffleSplit', 'StratifiedShuffleSplit', 'LeaveOneOut', 'LeavePOut', 'GridSearchCV', 'RandomizedSearchCV', 'HalvingGridSearchCV', 'HalvingRandomSearchCV'],
    }
}

def get_sklearn_algorithms():
    """Get all available scikit-learn algorithms"""
    algorithms = {}
    
    if not SKLEARN_AVAILABLE:
        return algorithms
    
    for category, modules in SKLEARN_ALGORITHMS.items():
        algorithms[category] = {}
        for module_name, algo_list in modules.items():
            module = getattr(sys.modules.get('sklearn', None), module_name, None)
            if module:
                available = []
                for algo_name in algo_list:
                    if hasattr(module, algo_name):
                        available.append(algo_name)
                algorithms[category][module_name] = available
    
    return algorithms

def check_toolbox_algorithm(category, algo_name):
    """Check if algorithm exists in toolbox"""
    if not TOOLBOX_AVAILABLE:
        return False
    
    toolbox = MLToolbox()
    
    # Check various possible names
    possible_names = [
        algo_name.lower(),
        algo_name,
        algo_name.replace('Classifier', '').replace('Regressor', ''),
        algo_name.replace('Classifier', '_classifier').replace('Regressor', '_regressor'),
    ]
    
    for name in possible_names:
        if hasattr(toolbox, name):
            return True
    
    # Check compartments
    if hasattr(toolbox, 'data'):
        if hasattr(toolbox.data, algo_name.lower()):
            return True
    
    if hasattr(toolbox, 'algorithms'):
        if hasattr(toolbox.algorithms, algo_name.lower()):
            return True
    
    return False

def audit_algorithms():
    """Main audit function"""
    print("="*80)
    print("ML TOOLBOX ALGORITHM AUDIT")
    print("="*80)
    print()
    
    sklearn_algorithms = get_sklearn_algorithms()
    
    results = {
        'present': [],
        'missing': [],
        'by_category': {}
    }
    
    total_sklearn = 0
    total_present = 0
    total_missing = 0
    
    for category, modules in sklearn_algorithms.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        print("-" * 80)
        
        category_present = []
        category_missing = []
        
        for module_name, algo_list in modules.items():
            for algo_name in algo_list:
                total_sklearn += 1
                present = check_toolbox_algorithm(category, algo_name)
                
                status = "✅" if present else "❌"
                print(f"  {status} {algo_name}")
                
                if present:
                    total_present += 1
                    category_present.append(algo_name)
                    results['present'].append((category, algo_name))
                else:
                    total_missing += 1
                    category_missing.append(algo_name)
                    results['missing'].append((category, algo_name))
        
        results['by_category'][category] = {
            'present': len(category_present),
            'missing': len(category_missing),
            'total': len(category_present) + len(category_missing),
            'coverage': len(category_present) / (len(category_present) + len(category_missing)) * 100 if (len(category_present) + len(category_missing)) > 0 else 0
        }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total scikit-learn algorithms: {total_sklearn}")
    print(f"Present in ML Toolbox: {total_present} ({total_present/total_sklearn*100:.1f}%)")
    print(f"Missing: {total_missing} ({total_missing/total_sklearn*100:.1f}%)")
    print()
    
    print("Coverage by Category:")
    for category, stats in results['by_category'].items():
        print(f"  {category}: {stats['present']}/{stats['total']} ({stats['coverage']:.1f}%)")
    
    # Priority missing algorithms
    print("\n" + "="*80)
    print("PRIORITY MISSING ALGORITHMS (Top 20)")
    print("="*80)
    
    # Prioritize by category importance
    priority_order = [
        ('classification', 'LogisticRegression'),
        ('classification', 'SVC'),
        ('classification', 'RandomForestClassifier'),
        ('classification', 'GradientBoostingClassifier'),
        ('regression', 'LinearRegression'),
        ('regression', 'Ridge'),
        ('regression', 'RandomForestRegressor'),
        ('clustering', 'KMeans'),
        ('clustering', 'DBSCAN'),
        ('preprocessing', 'StandardScaler'),
        ('preprocessing', 'MinMaxScaler'),
        ('preprocessing', 'SimpleImputer'),
        ('feature_selection', 'SelectKBest'),
        ('feature_selection', 'RFE'),
        ('model_selection', 'GridSearchCV'),
        ('model_selection', 'KFold'),
        ('dimensionality_reduction', 'PCA'),
        ('dimensionality_reduction', 'TruncatedSVD'),
    ]
    
    priority_missing = []
    for category, algo_name in priority_order:
        if (category, algo_name) in results['missing']:
            priority_missing.append((category, algo_name))
            if len(priority_missing) >= 20:
                break
    
    for i, (category, algo_name) in enumerate(priority_missing, 1):
        print(f"{i:2d}. {category}: {algo_name}")
    
    return results

if __name__ == '__main__':
    results = audit_algorithms()
    
    # Save results
    import json
    with open('algorithm_audit_results.json', 'w') as f:
        json.dump({
            'total_sklearn': sum(len(algos) for modules in SKLEARN_ALGORITHMS.values() for algos in modules.values()),
            'present': len(results['present']),
            'missing': len(results['missing']),
            'coverage': len(results['present']) / (len(results['present']) + len(results['missing'])) * 100 if (len(results['present']) + len(results['missing'])) > 0 else 0,
            'by_category': results['by_category'],
            'missing_list': results['missing']
        }, f, indent=2)
    
    print("\nResults saved to: algorithm_audit_results.json")
