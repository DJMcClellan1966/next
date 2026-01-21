"""
Test AdvancedDataPreprocessor for Regression Analysis
Predicting continuous target variables
"""
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor

# Try to import sklearn for regression
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class RegressionAnalysisTest:
    """Test regression analysis with AdvancedDataPreprocessor"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = []
    
    def generate_regression_data(self, n_samples: int = 200) -> tuple:
        """Generate text data with continuous target variables"""
        # Create text data with associated continuous values
        # Example: Product reviews with ratings (1-5 scale)
        data = [
            ("This product is absolutely amazing and exceeded all my expectations", 5.0),
            ("Excellent quality product, highly recommend to everyone", 4.8),
            ("Great product, works perfectly as described", 4.5),
            ("Good product, meets expectations", 4.0),
            ("Decent product, nothing special but works fine", 3.5),
            ("Average product, could be better", 3.0),
            ("Below average, has some issues", 2.5),
            ("Poor quality, not worth the money", 2.0),
            ("Terrible product, complete waste of money", 1.5),
            ("Worst product I've ever bought, avoid at all costs", 1.0),
            # Technical documentation quality scores
            ("Clear and comprehensive documentation with excellent examples", 4.8),
            ("Well-written documentation that's easy to follow", 4.5),
            ("Good documentation with helpful code samples", 4.2),
            ("Decent documentation, covers the basics", 3.8),
            ("Documentation is okay but could use more examples", 3.5),
            ("Average documentation, some parts unclear", 3.2),
            ("Poor documentation, hard to understand", 2.8),
            ("Very confusing documentation, needs improvement", 2.5),
            ("Terrible documentation, almost useless", 2.0),
            ("No useful documentation at all", 1.5),
            # Customer satisfaction scores
            ("Outstanding customer service, very helpful and responsive", 4.9),
            ("Excellent support team, quick response times", 4.7),
            ("Good customer service, resolved my issue quickly", 4.4),
            ("Decent support, took a while but got it done", 4.0),
            ("Average customer service, nothing special", 3.6),
            ("Slow response times, but eventually helpful", 3.3),
            ("Poor customer service, unhelpful staff", 2.9),
            ("Very bad experience with customer support", 2.6),
            ("Terrible customer service, avoid this company", 2.2),
            ("Worst customer service ever, completely useless", 1.8),
            # Code quality scores
            ("Clean and well-structured code with excellent documentation", 4.8),
            ("Well-written code that's easy to read and maintain", 4.6),
            ("Good code quality with proper error handling", 4.3),
            ("Decent code, follows best practices", 4.0),
            ("Average code quality, could use some improvements", 3.7),
            ("Code works but needs refactoring", 3.4),
            ("Poor code quality, hard to maintain", 3.0),
            ("Bad code structure, many issues", 2.7),
            ("Terrible code quality, needs complete rewrite", 2.3),
            ("Worst code I've ever seen, completely unmaintainable", 1.9),
        ]
        
        texts = []
        targets = []
        
        # Add base data
        for text, target in data:
            texts.append(text)
            targets.append(target)
        
        # Add semantic variations with similar scores
        variations = [
            ("This product is fantastic and went beyond what I expected", 5.0),
            ("Amazing quality product, I strongly recommend it", 4.8),
            ("Wonderful product, functions exactly as promised", 4.5),
            ("Satisfactory product, meets my needs", 4.0),
            ("Okay product, nothing remarkable but acceptable", 3.5),
            ("Mediocre product, room for improvement", 3.0),
            ("Subpar quality, has problems", 2.5),
            ("Low quality, not a good purchase", 2.0),
            ("Awful product, total disappointment", 1.5),
            ("Horrible product, do not buy this", 1.0),
        ]
        
        for text, target in variations:
            texts.append(text)
            targets.append(target)
        
        # Add more variations to reach desired size
        import random
        random.seed(self.random_state)
        while len(texts) < n_samples:
            base_idx = random.randint(0, len(data) - 1)
            base_text, base_target = data[base_idx]
            
            # Add slight variations with similar targets
            var_texts = [
                base_text + ".",
                base_text.lower(),
                base_text.replace("product", "item"),
                base_text.replace("excellent", "great")
            ]
            
            for var_text in var_texts[:min(4, n_samples - len(texts))]:
                texts.append(var_text)
                # Add small random noise to target
                targets.append(base_target + random.uniform(-0.2, 0.2))
        
        return texts[:n_samples], np.array(targets[:n_samples])
    
    def test_regression_with_preprocessor(
        self,
        texts: list,
        targets: np.ndarray,
        preprocessor_name: str,
        preprocessor_results: dict,
        use_embeddings: bool = True,
        verbose: bool = True
    ) -> dict:
        """Test regression models with preprocessed data"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        start_time = time.time()
        
        # Use preprocessed data
        processed_texts = preprocessor_results['deduplicated']
        # Match targets to processed texts (simplified - in practice would track mapping)
        processed_targets = targets[:len(processed_texts)]
        
        # Create features
        if use_embeddings and 'compressed_embeddings' in preprocessor_results and preprocessor_results['compressed_embeddings'] is not None:
            # Use compressed embeddings from preprocessor
            X = preprocessor_results['compressed_embeddings']
            feature_type = 'compressed_embeddings'
        elif use_embeddings:
            # Use original embeddings
            preprocessor = AdvancedDataPreprocessor() if preprocessor_name == 'AdvancedDataPreprocessor' else None
            if preprocessor and preprocessor.quantum_kernel:
                X = np.array([preprocessor.quantum_kernel.embed(text) for text in processed_texts])
                feature_type = 'quantum_embeddings'
            else:
                # Fallback to bag of words
                vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
                X = vectorizer.fit_transform(processed_texts).toarray()
                feature_type = 'tfidf'
        else:
            # Use bag of words
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X = vectorizer.fit_transform(processed_texts).toarray()
            feature_type = 'tfidf'
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, processed_targets, test_size=0.2, random_state=self.random_state
        )
        
        # Test multiple regression models
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)
        }
        
        results = {
            'preprocessor': preprocessor_name,
            'feature_type': feature_type,
            'n_samples': len(processed_texts),
            'n_features': X.shape[1],
            'models': {},
            'processing_time': time.time() - start_time
        }
        
        # Evaluate each model
        for model_name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            cv_r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            
            results['models'][model_name] = {
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'test_mae': float(test_mae),
                'test_r2': float(test_r2),
                'cv_mse_mean': float(-np.mean(cv_scores)),
                'cv_mse_std': float(np.std(cv_scores)),
                'cv_r2_mean': float(np.mean(cv_r2_scores)),
                'cv_r2_std': float(np.std(cv_r2_scores)),
                'overfitting_gap': float(train_mse - test_mse)
            }
        
        if verbose:
            self._print_results(results, preprocessor_name)
        
        return results
    
    def _print_results(self, results: dict, preprocessor_name: str):
        """Print regression test results"""
        print("\n" + "="*80)
        print(f"REGRESSION RESULTS: {preprocessor_name.upper()}")
        print("="*80)
        
        print(f"\nFeature Type: {results['feature_type']}")
        print(f"Preprocessed Samples: {results['n_samples']}")
        print(f"Features: {results['n_features']}")
        print(f"Processing Time: {results['processing_time']:.3f}s")
        
        print("\n[Model Performance]")
        print("-" * 80)
        for model_name, metrics in results['models'].items():
            print(f"\n{model_name}:")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
            print(f"  Test MSE: {metrics['test_mse']:.4f}")
            print(f"  Test MAE: {metrics['test_mae']:.4f}")
            print(f"  CV R² Mean: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
            print(f"  CV MSE Mean: {metrics['cv_mse_mean']:.4f} ± {metrics['cv_mse_std']:.4f}")
            print(f"  Overfitting Gap: {metrics['overfitting_gap']:.4f}")
        
        print("="*80)
    
    def compare_preprocessors(self, n_samples: int = 200, use_embeddings: bool = True, verbose: bool = True) -> dict:
        """Compare AdvancedDataPreprocessor vs ConventionalPreprocessor for regression"""
        print("\n" + "="*80)
        print("REGRESSION ANALYSIS COMPARISON TEST")
        print("="*80)
        
        # Generate test data
        print(f"\n[Generating Regression Data]")
        texts, targets = self.generate_regression_data(n_samples)
        print(f"  Generated: {len(texts)} samples")
        print(f"  Target range: {targets.min():.2f} - {targets.max():.2f}")
        print(f"  Target mean: {targets.mean():.2f}")
        
        # Preprocess with AdvancedDataPreprocessor
        print(f"\n[AdvancedDataPreprocessor]")
        print("-" * 80)
        advanced_preprocessor = AdvancedDataPreprocessor(
            dedup_threshold=0.9,
            enable_compression=True,
            compression_ratio=0.5
        )
        advanced_results = advanced_preprocessor.preprocess(texts.copy(), verbose=verbose)
        
        # Test regression with advanced preprocessor
        advanced_regression_results = self.test_regression_with_preprocessor(
            texts, targets, "AdvancedDataPreprocessor",
            advanced_results, use_embeddings=use_embeddings, verbose=verbose
        )
        
        # Preprocess with ConventionalPreprocessor
        print(f"\n[ConventionalPreprocessor]")
        print("-" * 80)
        conventional_preprocessor = ConventionalPreprocessor()
        conventional_results = conventional_preprocessor.preprocess(texts.copy(), verbose=verbose)
        
        # Test regression with conventional preprocessor
        conventional_regression_results = self.test_regression_with_preprocessor(
            texts, targets, "ConventionalPreprocessor",
            conventional_results, use_embeddings=False, verbose=verbose  # Use TF-IDF for conventional
        )
        
        # Compare results
        comparison = self._compare_results(advanced_regression_results, conventional_regression_results, verbose)
        
        return {
            'advanced': advanced_regression_results,
            'conventional': conventional_regression_results,
            'comparison': comparison
        }
    
    def _compare_results(self, advanced: dict, conventional: dict, verbose: bool = True) -> dict:
        """Compare regression results between preprocessors"""
        comparison = {
            'samples': {
                'advanced': advanced['n_samples'],
                'conventional': conventional['n_samples'],
                'difference': advanced['n_samples'] - conventional['n_samples']
            },
            'features': {
                'advanced': advanced['n_features'],
                'conventional': conventional['n_features'],
                'difference': advanced['n_features'] - conventional['n_features']
            },
            'models': {}
        }
        
        # Compare each model
        for model_name in advanced['models'].keys():
            if model_name not in conventional['models']:
                continue
                
            adv_metrics = advanced['models'][model_name]
            conv_metrics = conventional['models'][model_name]
            
            comparison['models'][model_name] = {
                'test_r2': {
                    'advanced': adv_metrics['test_r2'],
                    'conventional': conv_metrics['test_r2'],
                    'improvement': adv_metrics['test_r2'] - conv_metrics['test_r2']
                },
                'test_mse': {
                    'advanced': adv_metrics['test_mse'],
                    'conventional': conv_metrics['test_mse'],
                    'improvement': conv_metrics['test_mse'] - adv_metrics['test_mse']  # Lower is better
                },
                'test_mae': {
                    'advanced': adv_metrics['test_mae'],
                    'conventional': conv_metrics['test_mae'],
                    'improvement': conv_metrics['test_mae'] - adv_metrics['test_mae']  # Lower is better
                },
                'cv_r2_mean': {
                    'advanced': adv_metrics['cv_r2_mean'],
                    'conventional': conv_metrics['cv_r2_mean'],
                    'improvement': adv_metrics['cv_r2_mean'] - conv_metrics['cv_r2_mean']
                },
                'overfitting_gap': {
                    'advanced': adv_metrics['overfitting_gap'],
                    'conventional': conv_metrics['overfitting_gap'],
                    'difference': adv_metrics['overfitting_gap'] - conv_metrics['overfitting_gap']
                }
            }
        
        if verbose:
            self._print_comparison(comparison)
        
        return comparison
    
    def _print_comparison(self, comparison: dict):
        """Print comparison results"""
        print("\n" + "="*80)
        print("REGRESSION COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n[Samples]")
        print(f"  Advanced: {comparison['samples']['advanced']}")
        print(f"  Conventional: {comparison['samples']['conventional']}")
        print(f"  Difference: {comparison['samples']['difference']:+d}")
        
        print(f"\n[Features]")
        print(f"  Advanced: {comparison['features']['advanced']}")
        print(f"  Conventional: {comparison['features']['conventional']}")
        print(f"  Difference: {comparison['features']['difference']:+d}")
        
        print(f"\n[Model Performance]")
        print("-" * 80)
        for model_name, metrics in comparison['models'].items():
            print(f"\n{model_name}:")
            print(f"  Test R²:")
            print(f"    Advanced: {metrics['test_r2']['advanced']:.4f}")
            print(f"    Conventional: {metrics['test_r2']['conventional']:.4f}")
            print(f"    Improvement: {metrics['test_r2']['improvement']:+.4f}")
            
            print(f"  Test MSE (lower is better):")
            print(f"    Advanced: {metrics['test_mse']['advanced']:.4f}")
            print(f"    Conventional: {metrics['test_mse']['conventional']:.4f}")
            print(f"    Improvement: {metrics['test_mse']['improvement']:+.4f}")
            
            print(f"  Test MAE (lower is better):")
            print(f"    Advanced: {metrics['test_mae']['advanced']:.4f}")
            print(f"    Conventional: {metrics['test_mae']['conventional']:.4f}")
            print(f"    Improvement: {metrics['test_mae']['improvement']:+.4f}")
            
            print(f"  CV R² Mean:")
            print(f"    Advanced: {metrics['cv_r2_mean']['advanced']:.4f}")
            print(f"    Conventional: {metrics['cv_r2_mean']['conventional']:.4f}")
            print(f"    Improvement: {metrics['cv_r2_mean']['improvement']:+.4f}")
        
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        findings = []
        
        # Check for improvements
        for model_name, metrics in comparison['models'].items():
            if metrics['test_r2']['improvement'] > 0.05:
                findings.append(f"{model_name}: Advanced preprocessor improves R² by {metrics['test_r2']['improvement']:.4f}")
            elif metrics['test_r2']['improvement'] < -0.05:
                findings.append(f"{model_name}: Conventional preprocessor improves R² by {abs(metrics['test_r2']['improvement']):.4f}")
            
            if metrics['test_mse']['improvement'] > 0.1:
                findings.append(f"{model_name}: Advanced preprocessor reduces MSE by {metrics['test_mse']['improvement']:.4f}")
            elif metrics['test_mse']['improvement'] < -0.1:
                findings.append(f"{model_name}: Conventional preprocessor reduces MSE by {abs(metrics['test_mse']['improvement']):.4f}")
        
        if comparison['samples']['difference'] < 0:
            findings.append(f"Advanced preprocessor removes {abs(comparison['samples']['difference'])} more duplicates")
        
        for finding in findings:
            print(f"  • {finding}")
        
        if not findings:
            print("  • Results are similar between both methods")
        
        print("="*80 + "\n")


def main():
    """Run regression comparison test"""
    if not SKLEARN_AVAILABLE:
        print("Error: sklearn not available. Install with: pip install scikit-learn")
        return
    
    try:
        test = RegressionAnalysisTest(random_state=42)
        results = test.compare_preprocessors(n_samples=200, use_embeddings=True, verbose=True)
        
        print("\n" + "="*80)
        print("REGRESSION TEST COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
