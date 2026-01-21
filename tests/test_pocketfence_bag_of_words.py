"""
Test PocketFence Kernel vs Regular Preprocessing on Bag of Words Model
Comparing safety filtering impact on model performance
"""
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor

# Try to import sklearn for bag of words
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    import requests
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class PocketFenceBagOfWordsTest:
    """Compare PocketFence Kernel vs Regular preprocessing with Bag of Words"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = []
    
    def generate_test_data(self, n_samples: int = 200, include_unsafe: bool = True) -> tuple:
        """Generate test data with safe and unsafe content"""
        # Safe content
        safe_data = {
            'technical': [
                "Python programming language is great for data science",
                "Machine learning algorithms use neural networks",
                "Software development requires coding skills",
                "JavaScript is used for web development",
                "Database systems store structured data",
                "API endpoints provide data access",
                "Code optimization improves performance",
                "Version control systems track changes",
                "Debugging tools help find errors",
                "Cloud computing enables scalability"
            ],
            'business': [
                "Revenue increased by twenty percent this quarter",
                "Customer satisfaction drives business growth",
                "Market analysis shows positive trends",
                "Sales team achieved record profits",
                "Business strategy focuses on expansion",
                "Profit margins improved significantly",
                "Customer retention is important",
                "Market share increased this year",
                "Business development creates opportunities",
                "Revenue growth exceeded expectations"
            ],
            'support': [
                "I need help with technical issues",
                "Customer support is available twenty four seven",
                "How do I fix errors in my code",
                "Troubleshooting guide helps resolve problems",
                "Support team provides assistance",
                "Error messages indicate problems",
                "Help documentation explains solutions",
                "Technical support resolves issues",
                "Customer service helps users",
                "Problem solving requires patience"
            ],
            'education': [
                "Learn Python programming through online courses",
                "Educational tutorials teach coding skills",
                "Training programs cover programming fundamentals",
                "Study materials help students learn",
                "Teaching programming requires patience",
                "Learning resources are available online",
                "Educational content improves understanding",
                "Course materials explain concepts",
                "Training sessions teach new skills",
                "Study guides help with learning"
            ]
        }
        
        # Potentially unsafe content (for testing PocketFence)
        unsafe_data = [
            "This is spam content with malicious links",
            "Scam website trying to steal information",
            "Hack into this system with these tools",
            "Virus infected file download here",
            "Phishing attempt to get your password",
            "Malware distribution site",
            "Fraudulent scheme to make money",
            "Illegal content and activities",
            "Harmful software and tools",
            "Dangerous website with threats"
        ]
        
        texts = []
        labels = []
        
        # Add safe content
        for category, items in safe_data.items():
            for item in items:
                texts.append(item)
                labels.append(category)
        
        # Add semantic duplicates (safe)
        semantic_duplicates = [
            ("Python is excellent for data science", "technical"),
            ("ML algorithms employ neural networks", "technical"),
            ("Revenue grew twenty percent this period", "business"),
            ("I require assistance with technical problems", "support"),
            ("Study Python coding via internet tutorials", "education")
        ]
        
        for text, label in semantic_duplicates:
            texts.append(text)
            labels.append(label)
        
        # Add potentially unsafe content if requested
        if include_unsafe:
            for unsafe_text in unsafe_data:
                texts.append(unsafe_text)
                labels.append('unsafe')  # Mark as unsafe category
        
        # Add exact duplicates
        for i in range(min(20, len(texts))):
            texts.append(texts[i])
            labels.append(labels[i])
        
        # Add variations to reach desired size
        import random
        random.seed(self.random_state)
        while len(texts) < n_samples:
            base_idx = random.randint(0, len(safe_data['technical']) - 1)
            base_text = safe_data['technical'][base_idx]
            base_label = 'technical'
            
            variations = [
                base_text + ".",
                base_text.lower(),
                base_text.replace("Python", "Python programming"),
                base_text.replace("code", "source code")
            ]
            
            for var in variations[:min(4, n_samples - len(texts))]:
                texts.append(var)
                labels.append(base_label)
        
        return texts[:n_samples], labels[:n_samples]
    
    def check_pocketfence_available(self, url: str = "http://localhost:5000") -> bool:
        """Check if PocketFence service is available"""
        try:
            response = requests.get(f"{url}/api/kernel/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def test_bag_of_words_with_pocketfence(
        self,
        texts: list,
        labels: list,
        use_pocketfence: bool,
        pocketfence_url: str = "http://localhost:5000",
        verbose: bool = True
    ) -> dict:
        """Test bag of words model with/without PocketFence filtering"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        start_time = time.time()
        
        # Preprocess with PocketFence or without
        if use_pocketfence:
            preprocessor = AdvancedDataPreprocessor(
                pocketfence_url=pocketfence_url,
                dedup_threshold=0.9,
                enable_compression=False  # Don't compress for BoW comparison
            )
            preprocessor_name = "With PocketFence Kernel"
        else:
            # Use ConventionalPreprocessor for fair comparison (no semantic dedup)
            preprocessor = ConventionalPreprocessor()
            preprocessor_name = "Without PocketFence (Regular)"
        
        # Preprocess
        preprocessor_results = preprocessor.preprocess(texts.copy(), verbose=verbose)
        
        # Get processed data
        processed_texts = preprocessor_results['deduplicated']
        safe_count = len(preprocessor_results['safe_data'])
        unsafe_count = len(preprocessor_results['unsafe_data'])
        
        # Match labels to processed texts (simplified)
        processed_labels = labels[:len(processed_texts)]
        
        # Filter out 'unsafe' category if present (for classification)
        if 'unsafe' in processed_labels:
            # Keep unsafe items for analysis but mark them
            pass
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(processed_labels)
        
        # Create bag of words
        vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(processed_texts)
        
        # Train/test split (check if we have enough classes)
        if len(np.unique(y)) < 2:
            # Not enough classes, skip this test
            if verbose:
                print(f"  [Warning] Only {len(np.unique(y))} class(es) after preprocessing, skipping model training")
            return {
                'preprocessor': preprocessor_name,
                'use_pocketfence': use_pocketfence,
                'n_samples': len(processed_texts),
                'safe_count': safe_count,
                'unsafe_count': unsafe_count,
                'n_features': X.shape[1],
                'models': {},
                'processing_time': time.time() - start_time,
                'error': 'Insufficient classes for classification'
            }
        
        # Use smaller test size if we have few samples
        test_size = 0.2 if len(processed_texts) > 20 else 0.3
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        except:
            # If stratification fails, use regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        # Test multiple models
        models = {
            'NaiveBayes': MultinomialNB(),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        }
        
        results = {
            'preprocessor': preprocessor_name,
            'use_pocketfence': use_pocketfence,
            'n_samples': len(processed_texts),
            'safe_count': safe_count,
            'unsafe_count': unsafe_count,
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
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            # Cross-validation (adjust folds based on sample size)
            n_train_samples = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
            n_folds = min(5, n_train_samples // 2) if n_train_samples > 10 else 3
            if n_folds < 2:
                cv_scores = np.array([test_acc])
            else:
                try:
                    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                except:
                    # If CV fails, use simple split
                    cv_scores = np.array([test_acc])
            
            results['models'][model_name] = {
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1': float(test_f1),
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'overfitting_gap': float(train_acc - test_acc)
            }
        
        if verbose:
            self._print_results(results, preprocessor_name)
        
        return results
    
    def _print_results(self, results: dict, preprocessor_name: str):
        """Print test results"""
        print("\n" + "="*80)
        print(f"BAG OF WORDS RESULTS: {preprocessor_name.upper()}")
        print("="*80)
        
        print(f"\nPocketFence Used: {results['use_pocketfence']}")
        print(f"Preprocessed Samples: {results['n_samples']}")
        print(f"Safe Content: {results['safe_count']}")
        print(f"Unsafe Content Filtered: {results['unsafe_count']}")
        print(f"Features: {results['n_features']}")
        print(f"Processing Time: {results['processing_time']:.3f}s")
        
        print("\n[Model Performance]")
        print("-" * 80)
        for model_name, metrics in results['models'].items():
            print(f"\n{model_name}:")
            print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  CV Mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            print(f"  Overfitting Gap: {metrics['overfitting_gap']:.4f}")
        
        print("="*80)
    
    def compare_pocketfence_vs_regular(
        self,
        n_samples: int = 200,
        include_unsafe: bool = True,
        pocketfence_url: str = "http://localhost:5000",
        verbose: bool = True
    ) -> dict:
        """Compare PocketFence Kernel vs Regular preprocessing"""
        print("\n" + "="*80)
        print("POCKETFENCE KERNEL vs REGULAR PREPROCESSING - BAG OF WORDS TEST")
        print("="*80)
        
        # Check PocketFence availability
        pocketfence_available = self.check_pocketfence_available(pocketfence_url)
        if not pocketfence_available:
            print(f"\n[Warning] PocketFence service not available at {pocketfence_url}")
            print("          Will simulate PocketFence filtering (may not filter unsafe content)")
        
        # Generate test data
        print(f"\n[Generating Test Data]")
        texts, labels = self.generate_test_data(n_samples, include_unsafe=include_unsafe)
        print(f"  Generated: {len(texts)} samples")
        print(f"  Categories: {len(set(labels))}")
        if include_unsafe:
            unsafe_count = labels.count('unsafe')
            print(f"  Unsafe samples: {unsafe_count}")
        
        # Test with PocketFence
        print(f"\n[With PocketFence Kernel]")
        print("-" * 80)
        with_pocketfence_results = self.test_bag_of_words_with_pocketfence(
            texts.copy(), labels.copy(),
            use_pocketfence=True,
            pocketfence_url=pocketfence_url,
            verbose=verbose
        )
        
        # Test without PocketFence (regular) - use ConventionalPreprocessor
        print(f"\n[Without PocketFence (Regular - ConventionalPreprocessor)]")
        print("-" * 80)
        without_pocketfence_results = self.test_bag_of_words_with_pocketfence(
            texts.copy(), labels.copy(),
            use_pocketfence=False,
            pocketfence_url=pocketfence_url,
            verbose=verbose
        )
        
        # Compare results
        comparison = self._compare_results(
            with_pocketfence_results,
            without_pocketfence_results,
            verbose
        )
        
        return {
            'with_pocketfence': with_pocketfence_results,
            'without_pocketfence': without_pocketfence_results,
            'comparison': comparison,
            'pocketfence_available': pocketfence_available
        }
    
    def _compare_results(self, with_pf: dict, without_pf: dict, verbose: bool = True) -> dict:
        """Compare results between PocketFence and regular preprocessing"""
        comparison = {
            'samples': {
                'with_pocketfence': with_pf['n_samples'],
                'without_pocketfence': without_pf['n_samples'],
                'difference': with_pf['n_samples'] - without_pf['n_samples']
            },
            'safety_filtering': {
                'with_pocketfence': {
                    'safe': with_pf['safe_count'],
                    'unsafe_filtered': with_pf['unsafe_count']
                },
                'without_pocketfence': {
                    'safe': without_pf['safe_count'],
                    'unsafe_filtered': without_pf['unsafe_count']
                }
            },
            'features': {
                'with_pocketfence': with_pf['n_features'],
                'without_pocketfence': without_pf['n_features'],
                'difference': with_pf['n_features'] - without_pf['n_features']
            },
            'models': {}
        }
        
        # Compare each model
        for model_name in with_pf['models'].keys():
            if model_name not in without_pf['models']:
                continue
            
            pf_metrics = with_pf['models'][model_name]
            reg_metrics = without_pf['models'][model_name]
            
            comparison['models'][model_name] = {
                'test_accuracy': {
                    'with_pocketfence': pf_metrics['test_accuracy'],
                    'without_pocketfence': reg_metrics['test_accuracy'],
                    'difference': pf_metrics['test_accuracy'] - reg_metrics['test_accuracy']
                },
                'f1_score': {
                    'with_pocketfence': pf_metrics['f1'],
                    'without_pocketfence': reg_metrics['f1'],
                    'difference': pf_metrics['f1'] - reg_metrics['f1']
                },
                'cv_mean': {
                    'with_pocketfence': pf_metrics['cv_mean'],
                    'without_pocketfence': reg_metrics['cv_mean'],
                    'difference': pf_metrics['cv_mean'] - reg_metrics['cv_mean']
                },
                'overfitting_gap': {
                    'with_pocketfence': pf_metrics['overfitting_gap'],
                    'without_pocketfence': reg_metrics['overfitting_gap'],
                    'difference': pf_metrics['overfitting_gap'] - reg_metrics['overfitting_gap']
                }
            }
        
        if verbose:
            self._print_comparison(comparison)
        
        return comparison
    
    def _print_comparison(self, comparison: dict):
        """Print comparison results"""
        print("\n" + "="*80)
        print("POCKETFENCE vs REGULAR COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n[Samples]")
        print(f"  With PocketFence: {comparison['samples']['with_pocketfence']}")
        print(f"  Without PocketFence: {comparison['samples']['without_pocketfence']}")
        print(f"  Difference: {comparison['samples']['difference']:+d}")
        
        print(f"\n[Safety Filtering]")
        print(f"  With PocketFence:")
        print(f"    Safe: {comparison['safety_filtering']['with_pocketfence']['safe']}")
        print(f"    Unsafe Filtered: {comparison['safety_filtering']['with_pocketfence']['unsafe_filtered']}")
        print(f"  Without PocketFence:")
        print(f"    Safe: {comparison['safety_filtering']['without_pocketfence']['safe']}")
        print(f"    Unsafe Filtered: {comparison['safety_filtering']['without_pocketfence']['unsafe_filtered']}")
        
        print(f"\n[Features]")
        print(f"  With PocketFence: {comparison['features']['with_pocketfence']}")
        print(f"  Without PocketFence: {comparison['features']['without_pocketfence']}")
        print(f"  Difference: {comparison['features']['difference']:+d}")
        
        print(f"\n[Model Performance]")
        print("-" * 80)
        for model_name, metrics in comparison['models'].items():
            print(f"\n{model_name}:")
            print(f"  Test Accuracy:")
            print(f"    With PocketFence: {metrics['test_accuracy']['with_pocketfence']:.4f}")
            print(f"    Without PocketFence: {metrics['test_accuracy']['without_pocketfence']:.4f}")
            print(f"    Difference: {metrics['test_accuracy']['difference']:+.4f}")
            
            print(f"  F1 Score:")
            print(f"    With PocketFence: {metrics['f1_score']['with_pocketfence']:.4f}")
            print(f"    Without PocketFence: {metrics['f1_score']['without_pocketfence']:.4f}")
            print(f"    Difference: {metrics['f1_score']['difference']:+.4f}")
            
            print(f"  CV Mean:")
            print(f"    With PocketFence: {metrics['cv_mean']['with_pocketfence']:.4f}")
            print(f"    Without PocketFence: {metrics['cv_mean']['without_pocketfence']:.4f}")
            print(f"    Difference: {metrics['cv_mean']['difference']:+.4f}")
        
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        findings = []
        
        # Safety filtering impact
        unsafe_filtered_pf = comparison['safety_filtering']['with_pocketfence']['unsafe_filtered']
        unsafe_filtered_reg = comparison['safety_filtering']['without_pocketfence']['unsafe_filtered']
        if unsafe_filtered_pf > unsafe_filtered_reg:
            findings.append(f"PocketFence filtered {unsafe_filtered_pf - unsafe_filtered_reg} more unsafe items")
        
        # Performance impact
        for model_name, metrics in comparison['models'].items():
            if abs(metrics['test_accuracy']['difference']) > 0.05:
                if metrics['test_accuracy']['difference'] > 0:
                    findings.append(f"{model_name}: PocketFence improves accuracy by {metrics['test_accuracy']['difference']:.4f}")
                else:
                    findings.append(f"{model_name}: Regular preprocessing improves accuracy by {abs(metrics['test_accuracy']['difference']):.4f}")
        
        if comparison['samples']['difference'] != 0:
            findings.append(f"Sample count difference: {comparison['samples']['difference']:+d}")
        
        for finding in findings:
            print(f"  • {finding}")
        
        if not findings:
            print("  • Results are similar between both methods")
        
        print("="*80 + "\n")


def main():
    """Run comparison test"""
    if not SKLEARN_AVAILABLE:
        print("Error: sklearn not available. Install with: pip install scikit-learn")
        return
    
    try:
        test = PocketFenceBagOfWordsTest(random_state=42)
        results = test.compare_pocketfence_vs_regular(
            n_samples=200,
            include_unsafe=True,
            verbose=True
        )
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        
        if not results['pocketfence_available']:
            print("\n[Note] PocketFence service was not available.")
            print("       Results show simulated filtering behavior.")
            print("       For real filtering, start PocketFence service at http://localhost:5000")
        
        print("\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
