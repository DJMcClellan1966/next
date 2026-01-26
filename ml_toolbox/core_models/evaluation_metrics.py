"""
Evaluation Metrics

Implements:
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC
- Cross-Entropy Loss (Information-Theoretic)
- KL Divergence for Model Comparison (Information-Theoretic)
"""
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, 
                   average: str = 'binary') -> float:
    """
    Calculate precision score
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    average : str
        Averaging method ('binary', 'macro', 'micro', 'weighted')
        
    Returns
    -------
    precision : float
        Precision score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        # Binary classification
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    else:
        # Multi-class
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            # Micro-averaged: total TP / (total TP + total FP)
            total_tp = sum(np.sum((y_true == cls) & (y_pred == cls)) for cls in classes)
            total_fp = sum(np.sum((y_true != cls) & (y_pred == cls)) for cls in classes)
            return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        elif average == 'weighted':
            # Weight by support
            supports = [np.sum(y_true == cls) for cls in classes]
            return np.average(precisions, weights=supports)
        else:
            return np.mean(precisions)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
               average: str = 'binary') -> float:
    """
    Calculate recall score
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    average : str
        Averaging method
        
    Returns
    -------
    recall : float
        Recall score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        # Binary classification
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        if tp + fn == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    else:
        # Multi-class
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            total_tp = sum(np.sum((y_true == cls) & (y_pred == cls)) for cls in classes)
            total_fn = sum(np.sum((y_true == cls) & (y_pred != cls)) for cls in classes)
            return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        elif average == 'weighted':
            supports = [np.sum(y_true == cls) for cls in classes]
            return np.average(recalls, weights=supports)
        else:
            return np.mean(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
            average: str = 'binary') -> float:
    """
    Calculate F1 score
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    average : str
        Averaging method
        
    Returns
    -------
    f1 : float
        F1 score
    """
    prec = precision_score(y_true, y_pred, average)
    rec = recall_score(y_true, y_pred, average)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve
    
    Parameters
    ----------
    y_true : array
        True binary labels
    y_scores : array
        Target scores (probability of positive class)
        
    Returns
    -------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
    thresholds : array
        Thresholds
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, thresholds[-1] - 1)  # Add one more point
    
    fpr = []
    tpr = []
    
    # Calculate TPR and FPR for each threshold
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    return np.array(fpr), np.array(tpr), thresholds


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve (AUC)
    
    Parameters
    ----------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
        
    Returns
    -------
    auc : float
        AUC score
    """
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i-1]
        height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2
        auc += width * height
    
    return auc


def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate ROC AUC score
    
    Parameters
    ----------
    y_true : array
        True binary labels
    y_scores : array
        Target scores
        
    Returns
    -------
    auc : float
        ROC AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc_score(fpr, tpr)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                         target_names: Optional[List[str]] = None) -> Dict:
    """
    Generate classification report
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    target_names : list of str, optional
        Class names
        
    Returns
    -------
    report : dict
        Classification report
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = [f'Class {i}' for i in classes]
    
    report = {
        'precision': {},
        'recall': {},
        'f1-score': {},
        'support': {}
    }
    
    for i, cls in enumerate(classes):
        name = target_names[i] if i < len(target_names) else f'Class {cls}'
        
        # Calculate metrics for this class
        prec = precision_score(y_true, y_pred, average='binary' if len(classes) == 2 else None)
        rec = recall_score(y_true, y_pred, average='binary' if len(classes) == 2 else None)
        f1 = f1_score(y_true, y_pred, average='binary' if len(classes) == 2 else None)
        support = np.sum(y_true == cls)
        
        # For multi-class, calculate per-class
        if len(classes) > 2:
            # Binary classification for this class vs all others
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            prec = precision_score(y_true_binary, y_pred_binary, average='binary')
            rec = recall_score(y_true_binary, y_pred_binary, average='binary')
            f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
        
        report['precision'][name] = prec
        report['recall'][name] = rec
        report['f1-score'][name] = f1
        report['support'][name] = int(support)
    
    # Add averages
    report['precision']['macro avg'] = precision_score(y_true, y_pred, average='macro')
    report['recall']['macro avg'] = recall_score(y_true, y_pred, average='macro')
    report['f1-score']['macro avg'] = f1_score(y_true, y_pred, average='macro')
    report['support']['macro avg'] = len(y_true)
    
    report['precision']['weighted avg'] = precision_score(y_true, y_pred, average='weighted')
    report['recall']['weighted avg'] = recall_score(y_true, y_pred, average='weighted')
    report['f1-score']['weighted avg'] = f1_score(y_true, y_pred, average='weighted')
    
    return report


def cross_entropy_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       epsilon: float = 1e-15) -> float:
    """
    Calculate Cross-Entropy Loss (Information-Theoretic)
    
    H(P, Q) = -Σ P(x) * log(Q(x))
    
    Parameters
    ----------
    y_true : array
        True labels (one-hot encoded or class indices)
    y_pred_proba : array
        Predicted probabilities (shape: [n_samples, n_classes])
    epsilon : float
        Small value to avoid log(0)
        
    Returns
    -------
    loss : float
        Cross-entropy loss
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    # Handle one-hot encoding
    if y_true.ndim == 1:
        # Convert class indices to one-hot
        n_classes = y_pred_proba.shape[1] if y_pred_proba.ndim > 1 else 2
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = y_true_onehot
    elif y_true.ndim == 2 and y_true.shape[1] == 1:
        # Reshape if needed
        n_classes = y_pred_proba.shape[1] if y_pred_proba.ndim > 1 else 2
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true.ravel().astype(int)] = 1
        y_true = y_true_onehot
    
    # Ensure shapes match
    if y_pred_proba.ndim == 1:
        # Binary classification: convert to 2D
        y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
    
    # Calculate cross-entropy
    loss = -np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1))
    
    return loss


def kl_divergence_score(y_true_proba: np.ndarray, y_pred_proba: np.ndarray,
                        epsilon: float = 1e-15) -> float:
    """
    Calculate KL Divergence between True and Predicted Distributions
    
    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    
    Useful for comparing model predictions to true probability distributions
    
    Parameters
    ----------
    y_true_proba : array
        True probability distribution (shape: [n_samples, n_classes])
    y_pred_proba : array
        Predicted probability distribution (shape: [n_samples, n_classes])
    epsilon : float
        Small value to avoid log(0)
        
    Returns
    -------
    kl_div : float
        Average KL divergence across samples
    """
    y_true_proba = np.asarray(y_true_proba)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Normalize to ensure they're probability distributions
    y_true_proba = y_true_proba / (y_true_proba.sum(axis=1, keepdims=True) + epsilon)
    y_pred_proba = y_pred_proba / (y_pred_proba.sum(axis=1, keepdims=True) + epsilon)
    
    # Clip to avoid log(0)
    y_true_proba = np.clip(y_true_proba, epsilon, 1.0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1.0)
    
    # Calculate KL divergence for each sample
    kl_per_sample = np.sum(y_true_proba * np.log(y_true_proba / y_pred_proba), axis=1)
    
    # Return average
    return np.mean(kl_per_sample)


def model_comparison_kl(model1_proba: np.ndarray, model2_proba: np.ndarray,
                        reference_proba: Optional[np.ndarray] = None,
                        epsilon: float = 1e-15) -> Dict[str, float]:
    """
    Compare two models using KL Divergence
    
    Compares how well each model approximates the reference distribution
    (or compares them to each other)
    
    Parameters
    ----------
    model1_proba : array
        Model 1 predicted probabilities
    model2_proba : array
        Model 2 predicted probabilities
    reference_proba : array, optional
        Reference/true probability distribution
        If None, compares models to each other
    epsilon : float
        Small value to avoid log(0)
        
    Returns
    -------
    comparison : dict
        Dictionary with KL divergence scores
    """
    model1_proba = np.asarray(model1_proba)
    model2_proba = np.asarray(model2_proba)
    
    comparison = {}
    
    if reference_proba is not None:
        # Compare both models to reference
        reference_proba = np.asarray(reference_proba)
        comparison['model1_to_reference'] = kl_divergence_score(reference_proba, model1_proba, epsilon)
        comparison['model2_to_reference'] = kl_divergence_score(reference_proba, model2_proba, epsilon)
        comparison['better_model'] = 'model1' if comparison['model1_to_reference'] < comparison['model2_to_reference'] else 'model2'
    else:
        # Compare models to each other (symmetric)
        comparison['model1_to_model2'] = kl_divergence_score(model1_proba, model2_proba, epsilon)
        comparison['model2_to_model1'] = kl_divergence_score(model2_proba, model1_proba, epsilon)
        comparison['symmetric_kl'] = (comparison['model1_to_model2'] + comparison['model2_to_model1']) / 2
    
    return comparison
