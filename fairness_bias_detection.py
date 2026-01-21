"""
Fairness Metrics and Bias Detection
Ethical AI: Detect and measure bias in ML models

Features:
- Demographic parity
- Equalized odds
- Equal opportunity
- Calibration by group
- Disparate impact analysis
- Statistical parity testing
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.metrics import confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class FairnessAnalyzer:
    """
    Analyze fairness and bias in ML models
    
    Implements various fairness metrics and bias detection methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def demographic_parity(
        self,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        positive_label: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate demographic parity (statistical parity)
        
        P(Y=1|A=0) = P(Y=1|A=1)
        
        Measures if positive predictions are distributed equally across groups
        
        Args:
            y_pred: Predictions
            protected_attribute: Protected group membership (0 or 1)
            positive_label: Positive class label
            
        Returns:
            Dictionary with demographic parity metrics
        """
        y_pred = np.asarray(y_pred)
        protected_attribute = np.asarray(protected_attribute)
        
        # Get positive prediction rates by group
        group_0_mask = protected_attribute == 0
        group_1_mask = protected_attribute == 1
        
        rate_group_0 = np.mean(y_pred[group_0_mask] == positive_label) if np.sum(group_0_mask) > 0 else 0.0
        rate_group_1 = np.mean(y_pred[group_1_mask] == positive_label) if np.sum(group_1_mask) > 0 else 0.0
        
        # Demographic parity difference
        dp_difference = abs(rate_group_0 - rate_group_1)
        
        # Demographic parity ratio
        dp_ratio = rate_group_1 / (rate_group_0 + 1e-10) if rate_group_0 > 0 else 0.0
        
        # Fair if difference < 0.1 (10% threshold)
        is_fair = dp_difference < 0.1
        
        return {
            'rate_group_0': float(rate_group_0),
            'rate_group_1': float(rate_group_1),
            'demographic_parity_difference': float(dp_difference),
            'demographic_parity_ratio': float(dp_ratio),
            'is_fair': is_fair,
            'threshold': 0.1
        }
    
    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        positive_label: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate equalized odds
        
        P(Y=1|Y_true=1, A=0) = P(Y=1|Y_true=1, A=1)  (True Positive Rate)
        P(Y=1|Y_true=0, A=0) = P(Y=1|Y_true=0, A=1)  (False Positive Rate)
        
        Measures if TPR and FPR are equal across groups
        
        Args:
            y_true: True labels
            y_pred: Predictions
            protected_attribute: Protected group membership
            positive_label: Positive class label
            
        Returns:
            Dictionary with equalized odds metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        protected_attribute = np.asarray(protected_attribute)
        
        # Calculate confusion matrices for each group
        group_0_mask = protected_attribute == 0
        group_1_mask = protected_attribute == 1
        
        # Group 0
        y_true_0 = y_true[group_0_mask]
        y_pred_0 = y_pred[group_0_mask]
        if len(y_true_0) > 0:
            tn_0, fp_0, fn_0, tp_0 = confusion_matrix(y_true_0, y_pred_0, labels=[0, positive_label]).ravel()
            tpr_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
            fpr_0 = fp_0 / (fp_0 + tn_0) if (fp_0 + tn_0) > 0 else 0.0
        else:
            tpr_0, fpr_0 = 0.0, 0.0
        
        # Group 1
        y_true_1 = y_true[group_1_mask]
        y_pred_1 = y_pred[group_1_mask]
        if len(y_true_1) > 0:
            tn_1, fp_1, fn_1, tp_1 = confusion_matrix(y_true_1, y_pred_1, labels=[0, positive_label]).ravel()
            tpr_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0.0
            fpr_1 = fp_1 / (fp_1 + tn_1) if (fp_1 + tn_1) > 0 else 0.0
        else:
            tpr_1, fpr_1 = 0.0, 0.0
        
        # Equalized odds differences
        tpr_difference = abs(tpr_0 - tpr_1)
        fpr_difference = abs(fpr_0 - fpr_1)
        
        # Equalized odds: max of TPR and FPR differences
        eo_difference = max(tpr_difference, fpr_difference)
        
        # Fair if difference < 0.1
        is_fair = eo_difference < 0.1
        
        return {
            'tpr_group_0': float(tpr_0),
            'tpr_group_1': float(tpr_1),
            'fpr_group_0': float(fpr_0),
            'fpr_group_1': float(fpr_1),
            'tpr_difference': float(tpr_difference),
            'fpr_difference': float(fpr_difference),
            'equalized_odds_difference': float(eo_difference),
            'is_fair': is_fair,
            'threshold': 0.1
        }
    
    def equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        positive_label: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate equal opportunity (subset of equalized odds)
        
        P(Y=1|Y_true=1, A=0) = P(Y=1|Y_true=1, A=1)  (True Positive Rate)
        
        Only requires TPR equality (not FPR)
        
        Args:
            y_true: True labels
            y_pred: Predictions
            protected_attribute: Protected group membership
            positive_label: Positive class label
            
        Returns:
            Dictionary with equal opportunity metrics
        """
        eo_result = self.equalized_odds(y_true, y_pred, protected_attribute, positive_label)
        
        return {
            'tpr_group_0': eo_result['tpr_group_0'],
            'tpr_group_1': eo_result['tpr_group_1'],
            'tpr_difference': eo_result['tpr_difference'],
            'is_fair': eo_result['tpr_difference'] < 0.1,
            'threshold': 0.1
        }
    
    def calibration_by_group(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        protected_attribute: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate calibration by group
        
        Measures if predicted probabilities are calibrated equally across groups
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            protected_attribute: Protected group membership
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with calibration metrics
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        protected_attribute = np.asarray(protected_attribute)
        
        # Bin probabilities
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate calibration by group
        group_0_mask = protected_attribute == 0
        group_1_mask = protected_attribute == 1
        
        calibration_group_0 = []
        calibration_group_1 = []
        
        for i in range(n_bins):
            # Group 0
            mask_0 = (bin_indices == i) & group_0_mask
            if np.sum(mask_0) > 0:
                mean_proba_0 = np.mean(y_proba[mask_0])
                mean_actual_0 = np.mean(y_true[mask_0])
                calibration_group_0.append(abs(mean_proba_0 - mean_actual_0))
            
            # Group 1
            mask_1 = (bin_indices == i) & group_1_mask
            if np.sum(mask_1) > 0:
                mean_proba_1 = np.mean(y_proba[mask_1])
                mean_actual_1 = np.mean(y_true[mask_1])
                calibration_group_1.append(abs(mean_proba_1 - mean_actual_1))
        
        # Average calibration error
        ece_group_0 = np.mean(calibration_group_0) if len(calibration_group_0) > 0 else 0.0
        ece_group_1 = np.mean(calibration_group_1) if len(calibration_group_1) > 0 else 0.0
        
        # Calibration difference
        calibration_difference = abs(ece_group_0 - ece_group_1)
        
        return {
            'ece_group_0': float(ece_group_0),
            'ece_group_1': float(ece_group_1),
            'calibration_difference': float(calibration_difference),
            'is_fair': calibration_difference < 0.1,
            'threshold': 0.1
        }
    
    def disparate_impact(
        self,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        positive_label: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate disparate impact ratio
        
        DI = P(Y=1|A=1) / P(Y=1|A=0)
        
        Legal threshold: DI >= 0.8 (80% rule)
        
        Args:
            y_pred: Predictions
            protected_attribute: Protected group membership
            positive_label: Positive class label
            
        Returns:
            Dictionary with disparate impact metrics
        """
        dp_result = self.demographic_parity(y_pred, protected_attribute, positive_label)
        
        # Disparate impact ratio (same as demographic parity ratio)
        di_ratio = dp_result['demographic_parity_ratio']
        
        # Legal threshold: 0.8 (80% rule)
        is_fair = di_ratio >= 0.8
        
        return {
            'disparate_impact_ratio': float(di_ratio),
            'rate_group_0': dp_result['rate_group_0'],
            'rate_group_1': dp_result['rate_group_1'],
            'is_fair': is_fair,
            'legal_threshold': 0.8,
            'meets_legal_standard': is_fair
        }
    
    def statistical_parity_test(
        self,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        positive_label: int = 1,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Statistical test for demographic parity
        
        Chi-square test or two-proportion z-test
        
        Args:
            y_pred: Predictions
            protected_attribute: Protected group membership
            positive_label: Positive class label
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        y_pred = np.asarray(y_pred)
        protected_attribute = np.asarray(protected_attribute)
        
        # Create contingency table
        group_0_mask = protected_attribute == 0
        group_1_mask = protected_attribute == 1
        
        # Count positive predictions by group
        positive_0 = np.sum((y_pred == positive_label) & group_0_mask)
        total_0 = np.sum(group_0_mask)
        positive_1 = np.sum((y_pred == positive_label) & group_1_mask)
        total_1 = np.sum(group_1_mask)
        
        # Two-proportion z-test
        p1 = positive_0 / total_0 if total_0 > 0 else 0.0
        p2 = positive_1 / total_1 if total_1 > 0 else 0.0
        
        # Pooled proportion
        p_pooled = (positive_0 + positive_1) / (total_0 + total_1) if (total_0 + total_1) > 0 else 0.0
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total_0 + 1/total_1)) if (total_0 > 0 and total_1 > 0) else 0.0
        
        # Z-statistic
        z_stat = (p1 - p2) / (se + 1e-10) if se > 0 else 0.0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if hasattr(stats, 'norm') else 0.0
        
        # Significant difference
        is_significant = p_value < alpha
        
        return {
            'p1': float(p1),
            'p2': float(p2),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'alpha': alpha,
            'interpretation': 'Significant difference' if is_significant else 'No significant difference'
        }
    
    def comprehensive_fairness_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        protected_attribute: np.ndarray,
        positive_label: int = 1
    ) -> Dict[str, Any]:
        """
        Comprehensive fairness analysis
        
        Computes all fairness metrics
        
        Args:
            y_true: True labels
            y_pred: Predictions
            y_proba: Predicted probabilities (optional)
            protected_attribute: Protected group membership
            positive_label: Positive class label
            
        Returns:
            Dictionary with all fairness metrics
        """
        results = {
            'demographic_parity': self.demographic_parity(y_pred, protected_attribute, positive_label),
            'equalized_odds': self.equalized_odds(y_true, y_pred, protected_attribute, positive_label),
            'equal_opportunity': self.equal_opportunity(y_true, y_pred, protected_attribute, positive_label),
            'disparate_impact': self.disparate_impact(y_pred, protected_attribute, positive_label),
            'statistical_parity_test': self.statistical_parity_test(y_pred, protected_attribute, positive_label)
        }
        
        if y_proba is not None:
            results['calibration_by_group'] = self.calibration_by_group(
                y_true, y_proba, protected_attribute
            )
        
        # Overall fairness assessment
        is_fair_overall = all([
            results['demographic_parity']['is_fair'],
            results['equalized_odds']['is_fair'],
            results['equal_opportunity']['is_fair'],
            results['disparate_impact']['is_fair']
        ])
        
        results['overall_fairness'] = {
            'is_fair': is_fair_overall,
            'fairness_score': sum([
                results['demographic_parity']['is_fair'],
                results['equalized_odds']['is_fair'],
                results['equal_opportunity']['is_fair'],
                results['disparate_impact']['is_fair']
            ]) / 4.0
        }
        
        return results
