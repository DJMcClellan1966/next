"""
Data Quality Assessment (Information-Theoretic)

Implements:
- Feature Informativeness (Entropy-based)
- Feature Redundancy Detection (Mutual Information)
- Data Quality Score
- Missing Value Impact Assessment
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def feature_informativeness(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Calculate informativeness of each feature using entropy
    
    Higher entropy = more informative (more uniform distribution)
    Lower entropy = less informative (more concentrated)
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Feature matrix
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    informativeness : array, shape (n_features,)
        Entropy-based informativeness scores for each feature
    """
    try:
        from ml_toolbox.textbook_concepts.information_theory import entropy
    except ImportError:
        logger.error("[DataQuality] Could not import entropy from information_theory")
        return np.zeros(X.shape[1])
    
    X = np.asarray(X)
    informativeness_scores = []
    
    for feature_idx in range(X.shape[1]):
        feature = X[:, feature_idx]
        
        # Skip if constant
        if np.std(feature) < 1e-10:
            informativeness_scores.append(0.0)
            continue
        
        # Discretize feature
        feature_binned = np.digitize(feature, np.linspace(feature.min(), feature.max(), n_bins))
        
        # Calculate probability distribution
        counts = np.bincount(feature_binned, minlength=n_bins)
        probs = counts / counts.sum()
        
        # Calculate entropy (informativeness)
        info_score = entropy(probs, base=2.0)
        informativeness_scores.append(info_score)
    
    return np.array(informativeness_scores)


def feature_redundancy(X: np.ndarray, threshold: float = 0.7, n_bins: int = 10) -> Dict:
    """
    Detect redundant features using mutual information
    
    Features with high mutual information are redundant
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Feature matrix
    threshold : float
        MI threshold above which features are considered redundant
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    redundancy_info : dict
        Dictionary containing:
        - 'redundant_pairs': List of (i, j, mi_score) tuples
        - 'redundancy_matrix': Matrix of MI scores between all feature pairs
        - 'highly_redundant_features': List of feature indices to consider removing
    """
    try:
        from ml_toolbox.textbook_concepts.information_theory import mutual_information
    except ImportError:
        logger.error("[DataQuality] Could not import mutual_information from information_theory")
        return {
            'redundant_pairs': [],
            'redundancy_matrix': np.zeros((X.shape[1], X.shape[1])),
            'highly_redundant_features': []
        }
    
    X = np.asarray(X)
    n_features = X.shape[1]
    
    redundancy_matrix = np.zeros((n_features, n_features))
    redundant_pairs = []
    
    # Calculate MI between all feature pairs
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi = mutual_information(X[:, i], X[:, j], n_bins=n_bins)
            redundancy_matrix[i, j] = mi
            redundancy_matrix[j, i] = mi  # Symmetric
            
            if mi >= threshold:
                redundant_pairs.append((i, j, mi))
    
    # Find features that are redundant with many others
    feature_redundancy_counts = np.sum(redundancy_matrix >= threshold, axis=1)
    highly_redundant_features = np.where(feature_redundancy_counts > 1)[0].tolist()
    
    return {
        'redundant_pairs': redundant_pairs,
        'redundancy_matrix': redundancy_matrix,
        'highly_redundant_features': highly_redundant_features
    }


def data_quality_score(X: np.ndarray, y: Optional[np.ndarray] = None,
                      n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate overall data quality score using information-theoretic measures
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Feature matrix
    y : array, optional, shape (n_samples,)
        Target variable (if available, calculates feature-target MI)
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    quality_scores : dict
        Dictionary containing:
        - 'avg_feature_informativeness': Average entropy across features
        - 'feature_redundancy_ratio': Ratio of redundant feature pairs
        - 'feature_target_mi' (if y provided): Average MI between features and target
        - 'overall_quality': Composite quality score (0-1, higher is better)
    """
    X = np.asarray(X)
    
    # Feature informativeness
    informativeness = feature_informativeness(X, n_bins=n_bins)
    avg_informativeness = np.mean(informativeness)
    
    # Feature redundancy
    redundancy_info = feature_redundancy(X, threshold=0.7, n_bins=n_bins)
    n_pairs = X.shape[1] * (X.shape[1] - 1) / 2
    redundancy_ratio = len(redundancy_info['redundant_pairs']) / max(n_pairs, 1)
    
    quality_scores = {
        'avg_feature_informativeness': float(avg_informativeness),
        'feature_redundancy_ratio': float(redundancy_ratio),
    }
    
    # Feature-target mutual information (if target provided)
    if y is not None:
        try:
            from ml_toolbox.textbook_concepts.information_theory import mutual_information
            y = np.asarray(y).ravel()
            
            mi_scores = []
            for feature_idx in range(X.shape[1]):
                mi = mutual_information(X[:, feature_idx], y, n_bins=n_bins)
                mi_scores.append(mi)
            
            quality_scores['avg_feature_target_mi'] = float(np.mean(mi_scores))
            quality_scores['max_feature_target_mi'] = float(np.max(mi_scores))
        except Exception as e:
            logger.warning(f"[DataQuality] Could not calculate feature-target MI: {e}")
    
    # Overall quality score (normalized, higher is better)
    # Higher informativeness + lower redundancy + higher MI (if available) = better quality
    informativeness_norm = min(avg_informativeness / np.log2(n_bins), 1.0)  # Normalize by max entropy
    redundancy_penalty = 1.0 - min(redundancy_ratio, 1.0)
    
    if y is not None and 'avg_feature_target_mi' in quality_scores:
        mi_norm = min(quality_scores['avg_feature_target_mi'] / 1.0, 1.0)  # Normalize (MI typically < 1)
        overall_quality = (informativeness_norm * 0.3 + redundancy_penalty * 0.3 + mi_norm * 0.4)
    else:
        overall_quality = (informativeness_norm * 0.5 + redundancy_penalty * 0.5)
    
    quality_scores['overall_quality'] = float(overall_quality)
    
    return quality_scores


def missing_value_impact(X: np.ndarray, missing_mask: np.ndarray,
                        y: Optional[np.ndarray] = None,
                        n_bins: int = 10) -> Dict:
    """
    Assess the information-theoretic impact of missing values
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Feature matrix
    missing_mask : array, shape (n_samples, n_features)
        Boolean mask indicating missing values
    y : array, optional, shape (n_samples,)
        Target variable
    n_bins : int
        Number of bins for discretization
        
    Returns
    -------
    impact_info : dict
        Dictionary containing:
        - 'feature_missing_ratios': Ratio of missing values per feature
        - 'information_loss_per_feature': Estimated information loss due to missingness
        - 'recommended_imputation_priority': Features ranked by information loss
    """
    X = np.asarray(X)
    missing_mask = np.asarray(missing_mask, dtype=bool)
    
    n_features = X.shape[1]
    feature_missing_ratios = np.mean(missing_mask, axis=0)
    
    information_loss = []
    
    try:
        from ml_toolbox.textbook_concepts.information_theory import entropy, mutual_information
    except ImportError:
        logger.error("[DataQuality] Could not import information theory functions")
        return {
            'feature_missing_ratios': feature_missing_ratios.tolist(),
            'information_loss_per_feature': [0.0] * n_features,
            'recommended_imputation_priority': list(range(n_features))
        }
    
    for feature_idx in range(n_features):
        feature = X[:, feature_idx]
        missing = missing_mask[:, feature_idx]
        
        if np.sum(~missing) == 0:
            # All values missing
            information_loss.append(1.0)
            continue
        
        # Calculate entropy of non-missing values
        feature_complete = feature[~missing]
        if len(feature_complete) > 0 and np.std(feature_complete) > 1e-10:
            feature_binned = np.digitize(feature_complete, 
                                        np.linspace(feature_complete.min(), 
                                                   feature_complete.max(), n_bins))
            counts = np.bincount(feature_binned, minlength=n_bins)
            probs = counts / counts.sum()
            feature_entropy = entropy(probs, base=2.0)
        else:
            feature_entropy = 0.0
        
        # If target available, calculate MI loss
        if y is not None:
            y_complete = y[~missing]
            if len(y_complete) > 0:
                mi_complete = mutual_information(feature_complete, y_complete, n_bins=n_bins)
            else:
                mi_complete = 0.0
            
            # Estimate information loss as combination of entropy and MI
            max_entropy = np.log2(n_bins)
            entropy_loss = (max_entropy - feature_entropy) / max_entropy if max_entropy > 0 else 0
            mi_loss = 1.0 - min(mi_complete, 1.0)  # Normalize
            
            info_loss = (entropy_loss * 0.5 + mi_loss * 0.5) * feature_missing_ratios[feature_idx]
        else:
            # Only entropy-based
            max_entropy = np.log2(n_bins)
            entropy_loss = (max_entropy - feature_entropy) / max_entropy if max_entropy > 0 else 0
            info_loss = entropy_loss * feature_missing_ratios[feature_idx]
        
        information_loss.append(info_loss)
    
    information_loss = np.array(information_loss)
    
    # Rank features by information loss (highest loss = highest priority for imputation)
    imputation_priority = np.argsort(information_loss)[::-1]
    
    return {
        'feature_missing_ratios': feature_missing_ratios.tolist(),
        'information_loss_per_feature': information_loss.tolist(),
        'recommended_imputation_priority': imputation_priority.tolist()
    }


class DataQualityAssessor:
    """Data Quality Assessment Tool"""
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize data quality assessor
        
        Parameters
        ----------
        n_bins : int
            Number of bins for discretization
        """
        self.n_bins = n_bins
    
    def assess(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive data quality assessment
        
        Parameters
        ----------
        X : array
            Feature matrix
        y : array, optional
            Target variable
            
        Returns
        -------
        assessment : dict
            Comprehensive quality assessment report
        """
        X = np.asarray(X)
        
        # Feature informativeness
        informativeness = feature_informativeness(X, n_bins=self.n_bins)
        
        # Feature redundancy
        redundancy_info = feature_redundancy(X, threshold=0.7, n_bins=self.n_bins)
        
        # Overall quality score
        quality_scores = data_quality_score(X, y=y, n_bins=self.n_bins)
        
        # Missing value analysis (if any)
        missing_mask = np.isnan(X) if np.any(np.isnan(X)) else None
        missing_impact = None
        if missing_mask is not None and np.any(missing_mask):
            missing_impact = missing_value_impact(X, missing_mask, y=y, n_bins=self.n_bins)
        
        return {
            'feature_informativeness': informativeness.tolist(),
            'redundancy_analysis': {
                'redundant_pairs': redundancy_info['redundant_pairs'],
                'highly_redundant_features': redundancy_info['highly_redundant_features']
            },
            'quality_scores': quality_scores,
            'missing_value_impact': missing_impact,
            'recommendations': self._generate_recommendations(
                informativeness, redundancy_info, quality_scores, missing_impact
            )
        }
    
    def _generate_recommendations(self, informativeness: np.ndarray,
                                  redundancy_info: Dict,
                                  quality_scores: Dict,
                                  missing_impact: Optional[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Low informativeness features
        low_info_threshold = np.percentile(informativeness, 25)
        low_info_features = np.where(informativeness < low_info_threshold)[0]
        if len(low_info_features) > 0:
            recommendations.append(
                f"Consider removing or transforming {len(low_info_features)} low-informativeness features: {low_info_features.tolist()}"
            )
        
        # Redundant features
        if len(redundancy_info['highly_redundant_features']) > 0:
            recommendations.append(
                f"Remove redundant features: {redundancy_info['highly_redundant_features']}"
            )
        
        # Overall quality
        if quality_scores['overall_quality'] < 0.5:
            recommendations.append(
                "Data quality is below optimal. Consider feature engineering or data collection improvements."
            )
        
        # Missing values
        if missing_impact is not None:
            priority_features = missing_impact['recommended_imputation_priority'][:3]
            recommendations.append(
                f"Prioritize imputation for features: {priority_features} (highest information loss)"
            )
        
        if len(recommendations) == 0:
            recommendations.append("Data quality is good. No immediate recommendations.")
        
        return recommendations
