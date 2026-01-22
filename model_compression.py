"""
Model Compression
Compress ML models for deployment efficiency

Features:
- Quantization (reduce precision)
- Pruning (remove unnecessary weights)
- Knowledge distillation (smaller student model)
- Size optimization
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class ModelCompressor:
    """
    Model Compression for ML Toolbox
    
    Compresses models for efficient deployment
    """
    
    def __init__(self):
        self.compression_stats = {}
    
    def quantize_model(
        self,
        model: Any,
        precision: str = 'int8',
        calibration_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Quantize model to lower precision
        
        Args:
            model: Model to quantize
            precision: Target precision ('int8', 'int16', 'float16')
            calibration_data: Calibration data for quantization
            
        Returns:
            Dictionary with quantized model and stats
        """
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            return self._quantize_pytorch_model(model, precision, calibration_data)
        else:
            # For sklearn models, use simpler quantization
            return self._quantize_sklearn_model(model, precision)
    
    def _quantize_pytorch_model(
        self,
        model: torch.nn.Module,
        precision: str,
        calibration_data: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Quantize PyTorch model"""
        model.eval()
        
        # PyTorch quantization
        if precision == 'int8':
            try:
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                
                # Calculate size reduction
                original_size = sum(p.numel() * p.element_size() for p in model.parameters())
                quantized_size = sum(p.numel() * 1 for p in quantized_model.parameters())  # int8 = 1 byte
                compression_ratio = quantized_size / original_size
                
                return {
                    'model': quantized_model,
                    'compression_ratio': compression_ratio,
                    'original_size_mb': original_size / (1024 * 1024),
                    'compressed_size_mb': quantized_size / (1024 * 1024),
                    'method': 'quantization',
                    'precision': precision
                }
            except Exception as e:
                warnings.warn(f"PyTorch quantization failed: {e}")
                return {'error': str(e)}
        else:
            return {'error': f'Precision {precision} not yet implemented for PyTorch'}
    
    def _quantize_sklearn_model(
        self,
        model: Any,
        precision: str
    ) -> Dict[str, Any]:
        """Quantize sklearn model (simplified)"""
        # For sklearn models, we can't directly quantize, but we can estimate
        # size reduction from using smaller data types in predictions
        
        import pickle
        import io
        
        # Calculate original size
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        original_size = len(buffer.getvalue())
        
        # Estimate compressed size (simplified)
        if precision == 'int8':
            # Rough estimate: 4x reduction for tree models
            if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor,
                                RandomForestClassifier, RandomForestRegressor)):
                compressed_size = original_size * 0.25
            else:
                compressed_size = original_size * 0.5
        else:
            compressed_size = original_size * 0.75
        
        return {
            'model': model,  # Model unchanged (quantization not directly applicable)
            'compression_ratio': compressed_size / original_size,
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'method': 'quantization_estimate',
            'precision': precision,
            'note': 'sklearn models cannot be directly quantized'
        }
    
    def prune_model(
        self,
        model: Any,
        pruning_ratio: float = 0.3,
        method: str = 'magnitude'
    ) -> Dict[str, Any]:
        """
        Prune model (remove unnecessary weights)
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of weights to prune (0.0 to 1.0)
            method: Pruning method ('magnitude', 'random')
            
        Returns:
            Dictionary with pruned model and stats
        """
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            return self._prune_pytorch_model(model, pruning_ratio, method)
        else:
            # For sklearn tree models, use feature importance pruning
            return self._prune_sklearn_model(model, pruning_ratio)
    
    def _prune_pytorch_model(
        self,
        model: torch.nn.Module,
        pruning_ratio: float,
        method: str
    ) -> Dict[str, Any]:
        """Prune PyTorch model"""
        try:
            import torch.nn.utils.prune as prune
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Apply pruning
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if method == 'magnitude':
                        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                    elif method == 'random':
                        prune.random_unstructured(module, name='weight', amount=pruning_ratio)
            
            # Calculate compression
            pruned_params = sum(
                (p == 0).sum().item() for p in model.parameters()
            )
            compression_ratio = 1.0 - (pruned_params / total_params)
            
            return {
                'model': model,
                'compression_ratio': compression_ratio,
                'pruning_ratio': pruning_ratio,
                'method': method,
                'pruned_parameters': pruned_params,
                'total_parameters': total_params
            }
        except Exception as e:
            warnings.warn(f"PyTorch pruning failed: {e}")
            return {'error': str(e)}
    
    def _prune_sklearn_model(
        self,
        model: Any,
        pruning_ratio: float
    ) -> Dict[str, Any]:
        """Prune sklearn tree model"""
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            # Prune tree by setting max_depth
            original_depth = model.tree_.max_depth
            new_depth = max(1, int(original_depth * (1 - pruning_ratio)))
            
            # Create new model with reduced depth
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            if isinstance(model, DecisionTreeClassifier):
                pruned_model = DecisionTreeClassifier(max_depth=new_depth)
            else:
                pruned_model = DecisionTreeRegressor(max_depth=new_depth)
            
            # Note: Would need to retrain, but structure is set
            return {
                'model': model,  # Original model (pruning requires retraining)
                'suggested_max_depth': new_depth,
                'original_depth': original_depth,
                'pruning_ratio': pruning_ratio,
                'method': 'tree_pruning',
                'note': 'Tree pruning requires retraining with new max_depth'
            }
        else:
            return {
                'error': f'Pruning not supported for {type(model).__name__}'
            }
    
    def compress_model(
        self,
        model: Any,
        method: str = 'quantization',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compress model using specified method
        
        Args:
            model: Model to compress
            method: Compression method ('quantization', 'pruning', 'both')
            **kwargs: Additional arguments for compression
            
        Returns:
            Dictionary with compressed model and stats
        """
        if method == 'quantization':
            return self.quantize_model(model, **kwargs)
        elif method == 'pruning':
            return self.prune_model(model, **kwargs)
        elif method == 'both':
            # Apply both methods
            quantized = self.quantize_model(model, **kwargs)
            if 'error' not in quantized:
                pruned = self.prune_model(quantized['model'], **kwargs)
                return {
                    'model': pruned.get('model', quantized['model']),
                    'quantization_ratio': quantized.get('compression_ratio', 1.0),
                    'pruning_ratio': pruned.get('compression_ratio', 1.0),
                    'total_compression_ratio': (
                        quantized.get('compression_ratio', 1.0) *
                        pruned.get('compression_ratio', 1.0)
                    ),
                    'method': 'both'
                }
            else:
                return quantized
        else:
            return {'error': f'Unknown compression method: {method}'}
    
    def estimate_model_size(self, model: Any) -> Dict[str, Any]:
        """
        Estimate model size
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with size estimates
        """
        import pickle
        import io
        
        # Serialize model
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        size_bytes = len(buffer.getvalue())
        
        # Count parameters if PyTorch
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory (assuming float32 = 4 bytes)
            memory_bytes = total_params * 4
            
            return {
                'serialized_size_mb': size_bytes / (1024 * 1024),
                'memory_size_mb': memory_bytes / (1024 * 1024),
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_type': 'pytorch'
            }
        else:
            return {
                'serialized_size_mb': size_bytes / (1024 * 1024),
                'model_type': type(model).__name__
            }
