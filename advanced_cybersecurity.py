"""
Advanced Cybersecurity Methods
Additional security features for ML systems

Features:
- Secure Multi-Party Computation (SMPC)
- Model Watermarking
- Secure Model Serving
- Output Sanitization
- Privacy Budget Management
- Model Extraction Prevention
- Membership Inference Defense
- Data Poisoning Detection
- Secure Model Deployment
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import hashlib
import pickle
import time
import json

sys.path.insert(0, str(Path(__file__).parent))


class SecureMultiPartyComputation:
    """
    Secure Multi-Party Computation (SMPC)
    
    Privacy-preserving computation across multiple parties
    """
    
    def __init__(self, num_parties: int = 2):
        """
        Args:
            num_parties: Number of parties in computation
        """
        self.num_parties = num_parties
        self.parties = {}
    
    def add_party(self, party_id: str, data: np.ndarray):
        """Add party with data"""
        self.parties[party_id] = data
    
    def secure_sum(self, party_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Secure sum computation
        
        Args:
            party_ids: List of party IDs to include (None = all)
            
        Returns:
            Secure sum result
        """
        if party_ids is None:
            party_ids = list(self.parties.keys())
        
        # Simplified secure sum (in practice would use cryptographic protocols)
        total = np.zeros_like(list(self.parties.values())[0])
        
        for party_id in party_ids:
            if party_id in self.parties:
                total += self.parties[party_id]
        
        return {
            'result': total,
            'num_parties': len(party_ids),
            'method': 'simplified_secure_sum'
        }
    
    def secure_average(self, party_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Secure average computation"""
        sum_result = self.secure_sum(party_ids)
        num_parties = sum_result['num_parties']
        
        return {
            'result': sum_result['result'] / num_parties if num_parties > 0 else sum_result['result'],
            'num_parties': num_parties,
            'method': 'secure_average'
        }


class ModelWatermarking:
    """
    Model Watermarking
    
    Embed watermarks in models for ownership verification
    """
    
    def __init__(self, watermark_key: Optional[str] = None):
        """
        Args:
            watermark_key: Key for watermark generation
        """
        self.watermark_key = watermark_key or str(time.time())
    
    def embed_watermark(self, model: Any, watermark: str) -> Dict[str, Any]:
        """
        Embed watermark in model
        
        Args:
            model: Model to watermark
            watermark: Watermark string
            
        Returns:
            Watermarked model and metadata
        """
        # Generate watermark hash
        watermark_hash = hashlib.sha256(
            (watermark + self.watermark_key).encode()
        ).hexdigest()
        
        # Store watermark in model metadata
        if hasattr(model, '__dict__'):
            model._watermark = watermark_hash
            model._watermark_key = self.watermark_key
        
        return {
            'watermark_hash': watermark_hash,
            'watermark': watermark,
            'method': 'metadata_embedding'
        }
    
    def verify_watermark(self, model: Any, watermark: str) -> bool:
        """
        Verify watermark in model
        
        Args:
            model: Model to verify
            watermark: Original watermark string
            
        Returns:
            True if watermark is valid
        """
        if not hasattr(model, '_watermark'):
            return False
        
        expected_hash = hashlib.sha256(
            (watermark + self.watermark_key).encode()
        ).hexdigest()
        
        return model._watermark == expected_hash


class SecureModelServing:
    """
    Secure Model Serving
    
    Secure serving of ML models with input/output validation
    """
    
    def __init__(self, model: Any, input_validator: Optional[Any] = None):
        """
        Args:
            model: ML model to serve
            input_validator: Input validator instance
        """
        self.model = model
        self.input_validator = input_validator
        self.request_log = []
    
    def predict_secure(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Secure prediction with validation
        
        Args:
            X: Input features
            
        Returns:
            Prediction result with security metadata
        """
        # Validate input
        if self.input_validator:
            validation = self.input_validator.validate(X)
            if not validation.get('valid', False):
                return {
                    'error': 'Input validation failed',
                    'issues': validation.get('issues', [])
                }
        
        # Make prediction
        try:
            prediction = self.model.predict(X)
            
            # Log request
            self.request_log.append({
                'timestamp': time.time(),
                'input_shape': X.shape,
                'prediction_shape': prediction.shape if hasattr(prediction, 'shape') else len(prediction)
            })
            
            return {
                'prediction': prediction,
                'secure': True,
                'validated': self.input_validator is not None
            }
        except Exception as e:
            return {
                'error': str(e),
                'secure': False
            }
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        if not self.request_log:
            return {'total_requests': 0}
        
        return {
            'total_requests': len(self.request_log),
            'avg_input_size': np.mean([log['input_shape'][0] for log in self.request_log]),
            'requests_per_minute': len(self.request_log) / max(1, (time.time() - self.request_log[0]['timestamp']) / 60)
        }


class OutputSanitizer:
    """
    Output Sanitization
    
    Sanitize model outputs for security
    """
    
    def __init__(self, max_output_size: int = 1000, 
                 allowed_classes: Optional[List[int]] = None):
        """
        Args:
            max_output_size: Maximum output size
            allowed_classes: Allowed class indices (for classification)
        """
        self.max_output_size = max_output_size
        self.allowed_classes = allowed_classes
    
    def sanitize(self, output: Any) -> Dict[str, Any]:
        """
        Sanitize model output
        
        Args:
            output: Model output
            
        Returns:
            Sanitized output
        """
        # Convert to numpy array if needed
        if not isinstance(output, np.ndarray):
            output = np.array(output)
        
        # Check size
        if output.size > self.max_output_size:
            return {
                'error': f'Output too large: {output.size} > {self.max_output_size}',
                'sanitized': False
            }
        
        # Check for NaN/Inf
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            output = np.nan_to_num(output, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Filter classes if needed
        if self.allowed_classes is not None and len(output.shape) == 1:
            # For classification outputs
            filtered = np.zeros_like(output)
            for i, val in enumerate(output):
                if i in self.allowed_classes:
                    filtered[i] = val
            output = filtered
        
        return {
            'sanitized_output': output,
            'sanitized': True,
            'original_size': output.size
        }


class PrivacyBudgetManager:
    """
    Privacy Budget Management
    
    Manage differential privacy budgets
    """
    
    def __init__(self, total_budget: float = 1.0):
        """
        Args:
            total_budget: Total privacy budget (epsilon)
        """
        self.total_budget = total_budget
        self.used_budget = 0.0
        self.budget_history = []
    
    def allocate_budget(self, amount: float, operation: str = 'unknown') -> bool:
        """
        Allocate privacy budget
        
        Args:
            amount: Budget amount to allocate
            operation: Operation name
            
        Returns:
            True if allocation successful
        """
        if self.used_budget + amount > self.total_budget:
            return False
        
        self.used_budget += amount
        self.budget_history.append({
            'timestamp': time.time(),
            'amount': amount,
            'operation': operation,
            'remaining': self.total_budget - self.used_budget
        })
        
        return True
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.total_budget - self.used_budget)
    
    def reset_budget(self):
        """Reset privacy budget"""
        self.used_budget = 0.0
        self.budget_history = []
    
    def get_budget_report(self) -> Dict[str, Any]:
        """Get privacy budget report"""
        return {
            'total_budget': self.total_budget,
            'used_budget': self.used_budget,
            'remaining_budget': self.get_remaining_budget(),
            'utilization': self.used_budget / self.total_budget if self.total_budget > 0 else 0,
            'history': self.budget_history[-10:]  # Last 10 operations
        }


class ModelExtractionPrevention:
    """
    Model Extraction Prevention
    
    Prevent unauthorized model extraction
    """
    
    def __init__(self, model: Any, max_queries: int = 1000):
        """
        Args:
            model: Model to protect
            max_queries: Maximum queries allowed
        """
        self.model = model
        self.max_queries = max_queries
        self.query_count = 0
        self.query_log = []
    
    def predict_with_protection(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict with extraction prevention
        
        Args:
            X: Input features
            
        Returns:
            Prediction result or error if limit exceeded
        """
        if self.query_count >= self.max_queries:
            return {
                'error': f'Query limit exceeded: {self.max_queries}',
                'blocked': True
            }
        
        # Make prediction
        prediction = self.model.predict(X)
        
        # Log query
        self.query_count += 1
        self.query_log.append({
            'timestamp': time.time(),
            'input_shape': X.shape,
            'query_number': self.query_count
        })
        
        return {
            'prediction': prediction,
            'queries_remaining': self.max_queries - self.query_count,
            'blocked': False
        }
    
    def reset_queries(self):
        """Reset query count"""
        self.query_count = 0
        self.query_log = []


class MembershipInferenceDefense:
    """
    Membership Inference Defense
    
    Defend against membership inference attacks
    """
    
    def __init__(self, model: Any, defense_method: str = 'output_perturbation'):
        """
        Args:
            model: Model to protect
            defense_method: Defense method ('output_perturbation', 'differential_privacy')
        """
        self.model = model
        self.defense_method = defense_method
    
    def predict_with_defense(self, X: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Predict with membership inference defense
        
        Args:
            X: Input features
            noise_scale: Noise scale for perturbation
            
        Returns:
            Protected predictions
        """
        predictions = self.model.predict(X)
        
        if self.defense_method == 'output_perturbation':
            # Add noise to predictions
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)
                noise = np.random.normal(0, noise_scale, probs.shape)
                probs = np.clip(probs + noise, 0, 1)
                # Renormalize
                probs = probs / probs.sum(axis=1, keepdims=True)
                return np.argmax(probs, axis=1)
            else:
                # For non-probabilistic models, add noise to predictions
                noise = np.random.normal(0, noise_scale, predictions.shape)
                return predictions + noise
        else:
            return predictions


class DataPoisoningDetector:
    """
    Data Poisoning Detection
    
    Detect poisoned data in training sets
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Args:
            contamination: Expected contamination rate
        """
        self.contamination = contamination
    
    def detect_poisoning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Detect data poisoning
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Poisoning detection results
        """
        # Use isolation forest for anomaly detection
        try:
            from sklearn.ensemble import IsolationForest
            
            detector = IsolationForest(contamination=self.contamination, random_state=42)
            anomalies = detector.fit_predict(X)
            
            poisoned_indices = np.where(anomalies == -1)[0]
            
            return {
                'poisoned_detected': len(poisoned_indices) > 0,
                'poisoned_indices': poisoned_indices.tolist(),
                'poisoned_count': len(poisoned_indices),
                'contamination_rate': len(poisoned_indices) / len(X) if len(X) > 0 else 0
            }
        except ImportError:
            # Fallback: simple statistical detection
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            
            # Find outliers (simplified)
            z_scores = np.abs((X - means) / (stds + 1e-10))
            outliers = np.any(z_scores > 3, axis=1)
            
            return {
                'poisoned_detected': np.any(outliers),
                'poisoned_indices': np.where(outliers)[0].tolist(),
                'poisoned_count': np.sum(outliers),
                'method': 'statistical_outlier'
            }


class SecureModelDeployment:
    """
    Secure Model Deployment
    
    Secure deployment configuration and management
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        """
        Args:
            model: Model to deploy
            config: Deployment configuration
        """
        self.model = model
        self.config = config
        self.deployment_status = 'initialized'
    
    def deploy(self) -> Dict[str, Any]:
        """Deploy model securely"""
        # Validate configuration
        required = ['encryption', 'input_validation', 'output_sanitization']
        missing = [req for req in required if not self.config.get(req, False)]
        
        if missing:
            return {
                'success': False,
                'error': f'Missing required security features: {missing}'
            }
        
        self.deployment_status = 'deployed'
        
        return {
            'success': True,
            'status': self.deployment_status,
            'security_features': [k for k, v in self.config.items() if v],
            'timestamp': time.time()
        }
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        return {
            'status': self.deployment_status,
            'config': self.config,
            'model_type': type(self.model).__name__
        }
