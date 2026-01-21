"""
Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE)
Visual interpretability for feature effects

Features:
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE) plots
- Feature interaction plots
- 2D PDP plots
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.inspection import partial_dependence, plot_partial_dependence
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PartialDependenceAnalyzer:
    """
    Partial Dependence Plots and ICE plots
    
    Visualizes feature effects on model predictions
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pdp_results_ = {}
        self.is_fitted = False
    
    def compute_partial_dependence(
        self,
        model: Any,
        X: np.ndarray,
        features: Union[int, List[int], Tuple[int, int]],
        grid_resolution: int = 50,
        percentiles: Tuple[float, float] = (0.05, 0.95),
        kind: str = 'average'
    ) -> Dict[str, Any]:
        """
        Compute partial dependence
        
        Args:
            model: Fitted model
            X: Training data
            features: Feature index(es) to plot
            grid_resolution: Number of grid points
            percentiles: Percentile range for grid
            kind: 'average' (PDP) or 'individual' (ICE)
            
        Returns:
            Dictionary with PDP/ICE values, grid points
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        X = np.asarray(X)
        
        try:
            # Compute partial dependence
            pdp_result = partial_dependence(
                model,
                X,
                features=features,
                grid_resolution=grid_resolution,
                percentiles=percentiles,
                kind=kind
            )
            
            # Extract results
            grid_values = pdp_result['grid_values']
            averaged_predictions = pdp_result['average']
            
            result = {
                'grid_values': [g.tolist() if isinstance(g, np.ndarray) else g for g in grid_values],
                'average_predictions': averaged_predictions.tolist() if isinstance(averaged_predictions, np.ndarray) else averaged_predictions,
                'features': features if isinstance(features, (list, tuple)) else [features],
                'kind': kind,
                'grid_resolution': grid_resolution
            }
            
            # For ICE plots, also get individual predictions
            if kind == 'individual' and 'individual' in pdp_result:
                result['individual_predictions'] = pdp_result['individual'].tolist()
            
            self.pdp_results_[str(features)] = result
            self.is_fitted = True
            
            return result
        except Exception as e:
            return {'error': f'Partial dependence computation failed: {str(e)}'}
    
    def plot_partial_dependence(
        self,
        model: Any,
        X: np.ndarray,
        features: Union[int, List[int], Tuple[int, int]],
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        kind: str = 'average',
        **kwargs
    ):
        """
        Plot partial dependence
        
        Args:
            model: Fitted model
            X: Training data
            features: Feature index(es) to plot
            feature_names: Feature names
            save_path: Path to save figure
            show: Whether to display plot
            kind: 'average' (PDP) or 'individual' (ICE)
            **kwargs: Additional plotting parameters
        """
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("Warning: sklearn or matplotlib not available for plotting")
            return
        
        try:
            # Use sklearn's plotting function
            fig, axes = plot_partial_dependence(
                model,
                X,
                features=features,
                feature_names=feature_names,
                kind=kind,
                **kwargs
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Warning: Failed to plot partial dependence: {e}")
    
    def plot_ice(
        self,
        model: Any,
        X: np.ndarray,
        feature: int,
        feature_names: Optional[List[str]] = None,
        n_ice_lines: int = 50,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot Individual Conditional Expectation (ICE) curves
        
        Args:
            model: Fitted model
            X: Training data
            feature: Feature index to plot
            feature_names: Feature names
            n_ice_lines: Number of ICE lines to plot
            save_path: Path to save figure
            show: Whether to display plot
        """
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("Warning: sklearn or matplotlib not available for plotting")
            return
        
        # Compute ICE
        result = self.compute_partial_dependence(
            model, X, features=feature, kind='individual'
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            grid_values = result['grid_values'][0]
            individual_predictions = result.get('individual_predictions', [])
            
            if len(individual_predictions) > 0:
                # Plot individual lines
                for i in range(min(n_ice_lines, len(individual_predictions))):
                    ax.plot(grid_values, individual_predictions[i], alpha=0.3, linewidth=0.5)
                
                # Plot average (PDP)
                avg_predictions = result['average_predictions']
                ax.plot(grid_values, avg_predictions, 'r-', linewidth=2, label='Average (PDP)')
            
            feature_name = feature_names[feature] if feature_names and feature < len(feature_names) else f'Feature {feature}'
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'ICE Plot: {feature_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Warning: Failed to plot ICE: {e}")
    
    def plot_interaction(
        self,
        model: Any,
        X: np.ndarray,
        features: Tuple[int, int],
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot 2D partial dependence (feature interaction)
        
        Args:
            model: Fitted model
            X: Training data
            features: Tuple of two feature indices
            feature_names: Feature names
            save_path: Path to save figure
            show: Whether to display plot
        """
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("Warning: sklearn or matplotlib not available for plotting")
            return
        
        # Compute 2D PDP
        result = self.compute_partial_dependence(
            model, X, features=features, grid_resolution=20
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        try:
            # Plot 2D heatmap
            grid_values_0 = result['grid_values'][0]
            grid_values_1 = result['grid_values'][1]
            averaged_predictions = np.array(result['average_predictions'])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.contourf(grid_values_0, grid_values_1, averaged_predictions, levels=20, cmap='viridis')
            plt.colorbar(im, ax=ax)
            
            feature_name_0 = feature_names[features[0]] if feature_names and features[0] < len(feature_names) else f'Feature {features[0]}'
            feature_name_1 = feature_names[features[1]] if feature_names and features[1] < len(feature_names) else f'Feature {features[1]}'
            
            ax.set_xlabel(feature_name_0)
            ax.set_ylabel(feature_name_1)
            ax.set_title(f'2D Partial Dependence: {feature_name_0} vs {feature_name_1}')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            print(f"Warning: Failed to plot interaction: {e}")
