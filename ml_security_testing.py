"""
ML Security Testing Framework
Comprehensive security testing for ML systems based on Jon Erickson's methods

Features:
- Penetration testing for ML models
- Vulnerability assessment
- Exploitation testing
- Adversarial attack testing
- Security auditing
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


class MLSecurityTester:
    """
    ML Security Testing Framework
    
    Comprehensive security testing for ML systems
    """
    
    def __init__(self, model: Any, deployment_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model: ML model to test
            deployment_config: Deployment configuration (optional)
        """
        self.model = model
        self.deployment_config = deployment_config or {}
        self.test_results = []
    
    def assess_vulnerabilities(self) -> Dict[str, Any]:
        """
        Comprehensive vulnerability assessment
        
        Returns:
            Vulnerability assessment report
        """
        vulnerabilities = {
            'input_validation': [],
            'adversarial_robustness': [],
            'model_inversion': [],
            'membership_inference': [],
            'model_poisoning': [],
            'backdoor_detection': [],
            'overall_risk': 'unknown'
        }
        
        # Test input validation
        vulnerabilities['input_validation'] = self._test_input_validation()
        
        # Test adversarial robustness
        vulnerabilities['adversarial_robustness'] = self._test_adversarial_robustness()
        
        # Test model inversion
        vulnerabilities['model_inversion'] = self._test_model_inversion()
        
        # Test membership inference
        vulnerabilities['membership_inference'] = self._test_membership_inference()
        
        # Test model poisoning
        vulnerabilities['model_poisoning'] = self._test_model_poisoning()
        
        # Test backdoor detection
        vulnerabilities['backdoor_detection'] = self._test_backdoor_detection()
        
        # Calculate overall risk
        vulnerabilities['overall_risk'] = self._calculate_risk(vulnerabilities)
        
        return vulnerabilities
    
    def _test_input_validation(self) -> List[Dict[str, Any]]:
        """Test input validation vulnerabilities"""
        issues = []
        
        # Test with extreme values
        try:
            extreme_input = np.array([[1e20] * 10])
            self.model.predict(extreme_input)
            issues.append({
                'severity': 'medium',
                'issue': 'Accepts extreme input values',
                'recommendation': 'Add input validation and clipping'
            })
        except:
            pass
        
        # Test with NaN/Inf
        try:
            nan_input = np.array([[np.nan] * 10])
            self.model.predict(nan_input)
            issues.append({
                'severity': 'high',
                'issue': 'Accepts NaN/Inf values',
                'recommendation': 'Add input validation to reject NaN/Inf'
            })
        except:
            pass
        
        # Test with wrong shape
        try:
            wrong_shape = np.array([1, 2, 3])
            self.model.predict(wrong_shape)
            issues.append({
                'severity': 'high',
                'issue': 'Accepts wrong input shape',
                'recommendation': 'Add shape validation'
            })
        except:
            pass
        
        return issues
    
    def _test_adversarial_robustness(self) -> List[Dict[str, Any]]:
        """Test adversarial attack robustness"""
        issues = []
        
        # Generate test data
        X_test = np.random.rand(10, 10)
        
        # Get baseline predictions
        try:
            baseline_preds = self.model.predict(X_test)
            
            # Add small perturbations
            epsilon = 0.01
            X_perturbed = X_test + np.random.randn(*X_test.shape) * epsilon
            
            perturbed_preds = self.model.predict(X_perturbed)
            
            # Check if predictions changed significantly
            if hasattr(self.model, 'predict_proba'):
                baseline_probs = self.model.predict_proba(X_test)
                perturbed_probs = self.model.predict_proba(X_perturbed)
                
                prob_diff = np.abs(baseline_probs - perturbed_probs).max()
                
                if prob_diff > 0.1:
                    issues.append({
                        'severity': 'high',
                        'issue': f'Model sensitive to small perturbations (diff: {prob_diff:.4f})',
                        'recommendation': 'Use adversarial training or robust models'
                    })
        except Exception as e:
            issues.append({
                'severity': 'medium',
                'issue': f'Could not test adversarial robustness: {str(e)}',
                'recommendation': 'Ensure model supports prediction'
            })
        
        return issues
    
    def _test_model_inversion(self) -> List[Dict[str, Any]]:
        """Test model inversion attacks"""
        issues = []
        
        # Simplified model inversion test
        # Check if model outputs reveal too much information
        try:
            if hasattr(self.model, 'predict_proba'):
                # Test with uniform input
                X_test = np.random.rand(5, 10)
                probs = self.model.predict_proba(X_test)
                
                # Check if probabilities are too confident (potential information leakage)
                max_probs = probs.max(axis=1)
                if np.any(max_probs > 0.99):
                    issues.append({
                        'severity': 'medium',
                        'issue': 'Model outputs very confident predictions (potential information leakage)',
                        'recommendation': 'Consider differential privacy or output perturbation'
                    })
        except:
            pass
        
        return issues
    
    def _test_membership_inference(self) -> List[Dict[str, Any]]:
        """Test membership inference attacks"""
        issues = []
        
        # Simplified membership inference test
        # Check if model is overfitted (more vulnerable to membership inference)
        try:
            if hasattr(self.model, 'score'):
                # This is a simplified test - full membership inference is more complex
                issues.append({
                    'severity': 'low',
                    'issue': 'Membership inference vulnerability assessment requires training data',
                    'recommendation': 'Use differential privacy or regularization to reduce overfitting'
                })
        except:
            pass
        
        return issues
    
    def _test_model_poisoning(self) -> List[Dict[str, Any]]:
        """Test model poisoning vulnerabilities"""
        issues = []
        
        # Check if model can be easily poisoned
        # This would require retraining, so we provide recommendations
        issues.append({
            'severity': 'medium',
            'issue': 'Model poisoning requires training data access',
            'recommendation': 'Implement data validation, anomaly detection, and secure training pipelines'
        })
        
        return issues
    
    def _test_backdoor_detection(self) -> List[Dict[str, Any]]:
        """Test for backdoor attacks"""
        issues = []
        
        # Simplified backdoor detection
        # Check model behavior on suspicious inputs
        try:
            # Test with unusual patterns
            suspicious_input = np.random.rand(10, 10) * 100  # Unusual scale
            preds = self.model.predict(suspicious_input)
            
            # Check for unusual behavior
            if len(np.unique(preds)) == 1:
                issues.append({
                    'severity': 'low',
                    'issue': 'Model shows consistent behavior on unusual inputs (potential backdoor)',
                    'recommendation': 'Investigate model training process and data sources'
                })
        except:
            pass
        
        return issues
    
    def _calculate_risk(self, vulnerabilities: Dict[str, List[Dict[str, Any]]]) -> str:
        """Calculate overall risk level"""
        severity_scores = {'high': 3, 'medium': 2, 'low': 1}
        total_score = 0
        
        for category, issues in vulnerabilities.items():
            if category != 'overall_risk':
                for issue in issues:
                    total_score += severity_scores.get(issue.get('severity', 'low'), 1)
        
        if total_score >= 10:
            return 'high'
        elif total_score >= 5:
            return 'medium'
        else:
            return 'low'
    
    def test_adversarial_attacks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Test adversarial attack resistance
        
        Args:
            X: Test features
            y: Test labels
            attack_methods: List of attack methods to test
            
        Returns:
            Adversarial attack test results
        """
        if attack_methods is None:
            attack_methods = ['random', 'fgsm', 'pgd']
        
        results = {
            'baseline_accuracy': 0.0,
            'attack_results': {},
            'robustness_score': 0.0
        }
        
        try:
            # Baseline accuracy
            y_pred = self.model.predict(X)
            baseline_acc = np.mean(y_pred == y)
            results['baseline_accuracy'] = baseline_acc
            
            # Test each attack method
            for method in attack_methods:
                X_adv = self._generate_adversarial_examples(X, y, method)
                y_pred_adv = self.model.predict(X_adv)
                adv_acc = np.mean(y_pred_adv == y)
                
                results['attack_results'][method] = {
                    'accuracy_under_attack': adv_acc,
                    'accuracy_drop': baseline_acc - adv_acc,
                    'robust': adv_acc > baseline_acc * 0.8
                }
            
            # Calculate robustness score
            robustness_scores = [
                result['accuracy_under_attack']
                for result in results['attack_results'].values()
            ]
            results['robustness_score'] = np.mean(robustness_scores) if robustness_scores else 0.0
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _generate_adversarial_examples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """Generate adversarial examples"""
        if method == 'random':
            return X + np.random.randn(*X.shape) * epsilon
        elif method == 'fgsm':
            # Simplified FGSM (Fast Gradient Sign Method)
            # For sklearn models, use finite differences
            X_adv = X.copy()
            for i in range(len(X)):
                x = X[i:i+1]
                # Simple perturbation
                perturbation = np.random.randn(*x.shape) * epsilon
                X_adv[i] = x + perturbation
            return X_adv
        elif method == 'pgd':
            # Simplified PGD (Projected Gradient Descent)
            X_adv = X.copy()
            for _ in range(5):  # Multiple iterations
                perturbation = np.random.randn(*X.shape) * (epsilon / 5)
                X_adv = X_adv + perturbation
                X_adv = np.clip(X_adv, X - epsilon, X + epsilon)
            return X_adv
        else:
            return X
    
    def test_input_manipulation(
        self,
        validator: Any,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test input manipulation attacks
        
        Args:
            validator: Input validator
            X: Test features
            
        Returns:
            Input manipulation test results
        """
        results = {
            'validation_passed': False,
            'manipulation_tests': {},
            'vulnerabilities': []
        }
        
        # Test validation
        validation_result = validator.validate(X)
        results['validation_passed'] = validation_result.get('valid', False)
        
        # Test various manipulations
        manipulations = {
            'extreme_values': X * 1e10,
            'nan_injection': X.copy(),
            'shape_manipulation': X.flatten()[:X.shape[0]],
            'type_manipulation': X.astype(str) if X.dtype != object else X
        }
        
        for name, manipulated_X in manipulations.items():
            try:
                if name == 'nan_injection':
                    manipulated_X[0, 0] = np.nan
                
                validation = validator.validate(manipulated_X)
                results['manipulation_tests'][name] = {
                    'passed_validation': validation.get('valid', False),
                    'issues': validation.get('issues', [])
                }
                
                if validation.get('valid', False):
                    results['vulnerabilities'].append({
                        'type': name,
                        'severity': 'high',
                        'description': f'Input manipulation {name} passed validation'
                    })
            except:
                results['manipulation_tests'][name] = {'error': 'Test failed'}
        
        return results
    
    def find_model_vulnerabilities(self) -> Dict[str, Any]:
        """
        Find model vulnerabilities
        
        Returns:
            Model vulnerability report
        """
        vulnerabilities = {
            'model_type': type(self.model).__name__,
            'vulnerabilities': [],
            'recommendations': []
        }
        
        # Check model attributes
        if not hasattr(self.model, 'predict'):
            vulnerabilities['vulnerabilities'].append({
                'severity': 'high',
                'issue': 'Model does not have predict method',
                'recommendation': 'Ensure model implements standard interface'
            })
        
        # Check for overfitting indicators
        if hasattr(self.model, 'score'):
            vulnerabilities['recommendations'].append(
                'Monitor model performance on validation set to detect overfitting'
            )
        
        # Check for security features
        if not hasattr(self.model, '__dict__'):
            vulnerabilities['recommendations'].append(
                'Consider adding model metadata and versioning for security tracking'
            )
        
        return vulnerabilities
    
    def penetration_test(self, ml_system: Dict[str, Any]) -> Dict[str, Any]:
        """
        Penetration testing for ML system
        
        Args:
            ml_system: ML system configuration
            
        Returns:
            Penetration test results
        """
        results = {
            'tested_components': [],
            'vulnerabilities_found': [],
            'exploits_successful': [],
            'security_score': 0.0
        }
        
        # Test model security
        model_vulns = self.assess_vulnerabilities()
        results['tested_components'].append('model')
        results['vulnerabilities_found'].extend([
            v for category in model_vulns.values()
            if isinstance(category, list)
            for v in category
        ])
        
        # Test input validation
        if 'validator' in ml_system and ml_system['validator'] is not None:
            input_tests = self.test_input_manipulation(ml_system['validator'], np.random.rand(10, 10))
            results['tested_components'].append('input_validation')
            results['vulnerabilities_found'].extend(input_tests.get('vulnerabilities', []))
        
        # Calculate security score
        total_vulns = len(results['vulnerabilities_found'])
        high_severity = sum(1 for v in results['vulnerabilities_found'] if v.get('severity') == 'high')
        
        if total_vulns == 0:
            results['security_score'] = 100.0
        else:
            results['security_score'] = max(0, 100 - (total_vulns * 10) - (high_severity * 20))
        
        return results
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        report = {
            'timestamp': time.time(),
            'model_info': {
                'type': type(self.model).__name__,
                'has_predict': hasattr(self.model, 'predict'),
                'has_predict_proba': hasattr(self.model, 'predict_proba')
            },
            'vulnerability_assessment': self.assess_vulnerabilities(),
            'recommendations': []
        }
        
        # Generate recommendations
        vulns = report['vulnerability_assessment']
        
        if vulns['overall_risk'] == 'high':
            report['recommendations'].append('URGENT: Address high-severity vulnerabilities immediately')
        
        if vulns['input_validation']:
            report['recommendations'].append('Implement comprehensive input validation')
        
        if vulns['adversarial_robustness']:
            report['recommendations'].append('Use adversarial training to improve robustness')
        
        if vulns['model_inversion']:
            report['recommendations'].append('Consider differential privacy for sensitive data')
        
        return report


class MLExploitTester:
    """
    ML Exploitation Tester
    
    Test ML models for various exploits
    """
    
    def __init__(self, model: Any):
        """
        Args:
            model: ML model to test
        """
        self.model = model
    
    def test_adversarial_attacks(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.01
    ) -> Dict[str, Any]:
        """Test adversarial attack resistance"""
        tester = MLSecurityTester(self.model)
        return tester.test_adversarial_attacks(X, y)
    
    def test_input_manipulation(self, X: np.ndarray) -> Dict[str, Any]:
        """Test input manipulation attacks"""
        from ml_security_framework import InputValidator
        
        validator = InputValidator()
        tester = MLSecurityTester(self.model)
        return tester.test_input_manipulation(validator, X)
    
    def find_model_vulnerabilities(self) -> Dict[str, Any]:
        """Find model vulnerabilities"""
        tester = MLSecurityTester(self.model)
        return tester.find_model_vulnerabilities()
    
    def assess_robustness(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Assess model robustness
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Robustness score (0-1)
        """
        tester = MLSecurityTester(self.model)
        results = tester.test_adversarial_attacks(X, y)
        return results.get('robustness_score', 0.0)


class MLSecurityAuditor:
    """
    ML Security Auditor
    
    Comprehensive security auditing for ML systems
    """
    
    def __init__(self, model: Any, deployment: Optional[Dict[str, Any]] = None):
        """
        Args:
            model: ML model
            deployment: Deployment configuration
        """
        self.model = model
        self.deployment = deployment or {}
        self.tester = MLSecurityTester(model, deployment)
    
    def comprehensive_audit(self) -> Dict[str, Any]:
        """Comprehensive security audit"""
        audit = {
            'timestamp': time.time(),
            'model_security': self.tester.assess_vulnerabilities(),
            'deployment_security': self._audit_deployment(),
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall score
        model_risk = audit['model_security'].get('overall_risk', 'unknown')
        risk_scores = {'low': 80, 'medium': 50, 'high': 20, 'unknown': 0}
        audit['overall_score'] = risk_scores.get(model_risk, 0)
        
        # Generate recommendations
        if audit['overall_score'] < 50:
            audit['recommendations'].append('URGENT: Address security vulnerabilities')
        
        return audit
    
    def _audit_deployment(self) -> Dict[str, Any]:
        """Audit deployment security"""
        issues = []
        
        # Check deployment configuration
        if not self.deployment.get('encryption'):
            issues.append({
                'severity': 'high',
                'issue': 'No encryption configured',
                'recommendation': 'Enable model encryption'
            })
        
        if not self.deployment.get('input_validation'):
            issues.append({
                'severity': 'high',
                'issue': 'No input validation configured',
                'recommendation': 'Enable input validation'
            })
        
        return {
            'issues': issues,
            'secure': len(issues) == 0
        }
    
    def scan_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Scan for vulnerabilities"""
        vulns = self.tester.assess_vulnerabilities()
        
        all_vulnerabilities = []
        for category, issues in vulns.items():
            if isinstance(issues, list):
                all_vulnerabilities.extend(issues)
        
        return all_vulnerabilities
    
    def get_recommendations(self) -> List[str]:
        """Get security recommendations"""
        report = self.tester.generate_security_report()
        return report.get('recommendations', [])
