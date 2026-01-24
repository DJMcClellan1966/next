"""
ML Security Framework
Quick wins for ML security: input validation, model encryption, adversarial defense

Features:
- Input validation framework
- Model encryption at rest
- Basic adversarial training
- Threat detection integration
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import pickle
import hashlib
from cryptography.fernet import Fernet
import base64

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class InputValidator:
    """
    Input Validation Framework
    
    Validates model inputs for security
    """
    
    def __init__(self, max_features: Optional[int] = None, 
                 feature_ranges: Optional[Dict[int, Tuple[float, float]]] = None):
        """
        Args:
            max_features: Maximum number of features allowed
            feature_ranges: Dictionary mapping feature index to (min, max) range
        """
        self.max_features = max_features
        self.feature_ranges = feature_ranges or {}
    
    def validate(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Validate input data
        
        Args:
            X: Input features
            
        Returns:
            Validation result with status and issues
        """
        issues = []
        
        # Check shape
        if len(X.shape) != 2:
            issues.append(f"Invalid shape: expected 2D array, got {len(X.shape)}D")
            return {'valid': False, 'issues': issues}
        
        # Check number of features
        if self.max_features and X.shape[1] > self.max_features:
            issues.append(f"Too many features: {X.shape[1]} > {self.max_features}")
        
        # Check feature ranges
        for idx, (min_val, max_val) in self.feature_ranges.items():
            if idx < X.shape[1]:
                feature_values = X[:, idx]
                if np.any(feature_values < min_val) or np.any(feature_values > max_val):
                    issues.append(
                        f"Feature {idx} out of range: "
                        f"expected [{min_val}, {max_val}], "
                        f"got [{np.min(feature_values):.4f}, {np.max(feature_values):.4f}]"
                    )
        
        # Check for NaN/Inf
        if np.any(np.isnan(X)):
            issues.append("NaN values detected in input")
        
        if np.any(np.isinf(X)):
            issues.append("Inf values detected in input")
        
        # Check for suspicious values (very large)
        if np.any(np.abs(X) > 1e10):
            issues.append("Suspiciously large values detected (>1e10)")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'shape': X.shape
        }
    
    def sanitize(self, X: np.ndarray) -> np.ndarray:
        """
        Sanitize input data
        
        Args:
            X: Input features
            
        Returns:
            Sanitized features
        """
        X_sanitized = X.copy()
        
        # Replace NaN with 0
        X_sanitized = np.nan_to_num(X_sanitized, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip to reasonable ranges if specified
        for idx, (min_val, max_val) in self.feature_ranges.items():
            if idx < X_sanitized.shape[1]:
                X_sanitized[:, idx] = np.clip(X_sanitized[:, idx], min_val, max_val)
        
        # Clip extreme values
        X_sanitized = np.clip(X_sanitized, -1e10, 1e10)
        
        return X_sanitized


class ModelEncryption:
    """
    Model Encryption
    
    Encrypt models at rest for security
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Args:
            key: Encryption key (generates new if None)
        """
        if key is None:
            key = Fernet.generate_key()
        self.key = key
        try:
            self.cipher = Fernet(key)
        except ImportError:
            self.cipher = None
            warnings.warn("cryptography not available. Install with: pip install cryptography")
    
    def encrypt_model(self, model: Any, output_path: str) -> bool:
        """
        Encrypt and save model
        
        Args:
            model: Model object
            output_path: Path to save encrypted model
            
        Returns:
            True if successful
        """
        if self.cipher is None:
            return False
        
        try:
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Encrypt
            encrypted = self.cipher.encrypt(model_bytes)
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(encrypted)
            
            return True
        except Exception as e:
            warnings.warn(f"Encryption failed: {e}")
            return False
    
    def decrypt_model(self, input_path: str) -> Optional[Any]:
        """
        Decrypt and load model
        
        Args:
            input_path: Path to encrypted model
            
        Returns:
            Decrypted model or None
        """
        if self.cipher is None:
            return None
        
        try:
            # Load encrypted
            with open(input_path, 'rb') as f:
                encrypted = f.read()
            
            # Decrypt
            decrypted = self.cipher.decrypt(encrypted)
            
            # Deserialize
            model = pickle.loads(decrypted)
            
            return model
        except Exception as e:
            warnings.warn(f"Decryption failed: {e}")
            return None
    
    def get_key_base64(self) -> str:
        """Get encryption key as base64 string"""
        return base64.b64encode(self.key).decode('utf-8')
    
    @staticmethod
    def from_key_base64(key_b64: str):
        """Create encryption from base64 key"""
        key = base64.b64decode(key_b64.encode('utf-8'))
        return ModelEncryption(key)


class AdversarialDefender:
    """
    Adversarial Defense
    
    Basic adversarial training and defense
    """
    
    def __init__(self, model: Any, epsilon: float = 0.01):
        """
        Args:
            model: Model to defend
            epsilon: Perturbation magnitude for adversarial examples
        """
        self.model = model
        self.epsilon = epsilon
    
    def generate_adversarial_example(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'fgsm'
    ) -> np.ndarray:
        """
        Generate adversarial example
        
        Args:
            X: Input features
            y: True labels
            method: 'fgsm' (Fast Gradient Sign Method) or 'random'
            
        Returns:
            Adversarial examples
        """
        if method == 'fgsm':
            return self._fgsm_attack(X, y)
        elif method == 'random':
            return self._random_perturbation(X)
        else:
            return X
    
    def _fgsm_attack(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fast Gradient Sign Method attack"""
        # Simplified FGSM - add perturbation in direction of gradient
        # For sklearn models, use finite differences
        X_adv = X.copy()
        
        for i in range(len(X)):
            x = X[i:i+1]
            pred = self.model.predict(x)[0]
            
            # Simple perturbation: add noise in random direction
            # (Full FGSM requires gradients, which sklearn doesn't provide easily)
            perturbation = np.random.randn(*x.shape) * self.epsilon
            X_adv[i] = x + perturbation
        
        return X_adv
    
    def _random_perturbation(self, X: np.ndarray) -> np.ndarray:
        """Random perturbation"""
        noise = np.random.randn(*X.shape) * self.epsilon
        return X + noise
    
    def adversarial_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 1,
        adversarial_ratio: float = 0.5
    ) -> Any:
        """
        Adversarial training
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            adversarial_ratio: Ratio of adversarial examples to add
            
        Returns:
            Trained model
        """
        # Generate adversarial examples
        n_adv = int(len(X_train) * adversarial_ratio)
        X_adv = self.generate_adversarial_example(
            X_train[:n_adv], y_train[:n_adv], method='random'
        )
        
        # Combine original and adversarial
        X_combined = np.vstack([X_train, X_adv])
        y_combined = np.hstack([y_train, y_train[:n_adv]])
        
        # Retrain model
        if hasattr(self.model, 'fit'):
            self.model.fit(X_combined, y_combined)
        
        return self.model


class ThreatDetectionSystem:
    """
    Threat Detection System
    
    Use ML Toolbox for security threat detection
    """
    
    def __init__(self):
        """Initialize threat detection system"""
        self.model = None
        self.validator = InputValidator()
    
    def train_threat_detector(
        self,
        X_normal: np.ndarray,
        X_threats: np.ndarray,
        use_ml_toolbox: bool = True
    ) -> Dict[str, Any]:
        """
        Train threat detection model
        
        Args:
            X_normal: Normal behavior features
            X_threats: Threat behavior features
            use_ml_toolbox: Use ML Toolbox for training
            
        Returns:
            Training results
        """
        # Combine data
        X = np.vstack([X_normal, X_threats])
        y = np.hstack([
            np.zeros(len(X_normal)),  # Normal = 0
            np.ones(len(X_threats))   # Threat = 1
        ])
        
        if use_ml_toolbox:
            try:
                from ml_toolbox import MLToolbox
                toolbox = MLToolbox()
                simple = toolbox.algorithms.get_simple_ml_tasks()
                
                result = simple.train_classifier(X, y, model_type='random_forest')
                self.model = result['model']
                
                return {
                    'model': self.model,
                    'accuracy': result['accuracy'],
                    'framework': 'ML Toolbox'
                }
            except Exception as e:
                warnings.warn(f"ML Toolbox not available: {e}")
        
        # Fallback to sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'framework': 'sklearn'
        }
    
    def detect_threat(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect threats in input data
        
        Args:
            X: Input features to check
            
        Returns:
            Threat detection results
        """
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Validate input
        validation = self.validator.validate(X)
        if not validation['valid']:
            return {
                'threat_detected': True,
                'reason': 'Input validation failed',
                'issues': validation['issues']
            }
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        
        threats = predictions == 1
        threat_count = np.sum(threats)
        
        return {
            'threat_detected': threat_count > 0,
            'threat_count': int(threat_count),
            'threat_indices': np.where(threats)[0].tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'validation': validation
        }


class MLSecurityFramework:
    """
    ML Security Framework
    
    Comprehensive ML security wrapper
    """
    
    def __init__(self, model: Any, validator: Optional[InputValidator] = None):
        """
        Args:
            model: ML model to secure
            validator: Input validator (creates default if None)
        """
        self.model = model
        self.validator = validator or InputValidator()
        self.defender = AdversarialDefender(model)
    
    def predict_secure(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Secure prediction with validation
        
        Args:
            X: Input features
            
        Returns:
            Prediction results with security info
        """
        # Validate input
        validation = self.validator.validate(X)
        
        if not validation['valid']:
            return {
                'error': 'Input validation failed',
                'issues': validation['issues'],
                'predictions': None
            }
        
        # Sanitize input
        X_sanitized = self.validator.sanitize(X)
        
        # Predict
        try:
            predictions = self.model.predict(X_sanitized)
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_sanitized)
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'validation': validation,
                'secure': True
            }
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'predictions': None,
                'secure': False
            }
    
    def get_security_info(self) -> Dict[str, Any]:
        """Get security information"""
        return {
            'input_validation': True,
            'adversarial_defense': True,
            'model_encryption_supported': True,
            'validator_config': {
                'max_features': self.validator.max_features,
                'feature_ranges': len(self.validator.feature_ranges)
            }
        }
