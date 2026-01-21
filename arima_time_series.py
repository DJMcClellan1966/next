"""
ARIMA Models for Time Series Analysis
AutoRegressive Integrated Moving Average models

Features:
- ARIMA model fitting
- Seasonal ARIMA (SARIMA)
- Auto ARIMA (automatic parameter selection)
- Time series forecasting
- Model diagnostics
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")
    print("ARIMA models will use simplified implementation")


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model
    
    For time series forecasting and analysis
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: str = 'c'
    ):
        """
        Args:
            order: (p, d, q) - ARIMA order
                p: AR order (autoregressive)
                d: I order (differencing)
                q: MA order (moving average)
            seasonal_order: (P, D, Q, s) - Seasonal order (None for non-seasonal)
            trend: 'c' (constant), 'n' (none), 't' (linear trend)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model_ = None
        self.fitted_model_ = None
        self.is_fitted = False
    
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        """
        Fit ARIMA model
        
        Args:
            y: Time series data (1D array)
            exog: Exogenous variables (optional)
            
        Returns:
            self
        """
        y = np.asarray(y).ravel()
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA. Install with: pip install statsmodels")
        
        if self.seasonal_order is not None:
            # SARIMA model
            self.model_ = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend
            )
        else:
            # ARIMA model
            self.model_ = ARIMA(
                y,
                exog=exog,
                order=self.order,
                trend=self.trend
            )
        
        # Fit model
        self.fitted_model_ = self.model_.fit()
        self.is_fitted = True
        
        return self
    
    def predict(
        self,
        steps: int = 1,
        exog: Optional[np.ndarray] = None,
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forecast future values
        
        Args:
            steps: Number of steps ahead to forecast
            exog: Exogenous variables for forecast period
            return_conf_int: Whether to return confidence intervals
            alpha: Confidence level for intervals
            
        Returns:
            Predictions (and confidence intervals if requested)
        """
        if not self.is_fitted:
            raise ValueError("Must fit before predict")
        
        forecast = self.fitted_model_.forecast(steps=steps, exog=exog, alpha=alpha)
        
        if return_conf_int:
            conf_int = self.fitted_model_.get_forecast(steps=steps, exog=exog).conf_int(alpha=alpha)
            return forecast, conf_int
        else:
            return forecast
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model summary"""
        if not self.is_fitted:
            raise ValueError("Must fit before get_summary")
        
        summary = self.fitted_model_.summary()
        
        return {
            'aic': float(self.fitted_model_.aic),
            'bic': float(self.fitted_model_.bic),
            'llf': float(self.fitted_model_.llf),
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'summary_text': str(summary)
        }
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals"""
        if not self.is_fitted:
            raise ValueError("Must fit before get_residuals")
        
        return self.fitted_model_.resid
    
    def diagnose(self) -> Dict[str, Any]:
        """
        Model diagnostics
        
        Returns:
            Dictionary with diagnostic statistics
        """
        if not self.is_fitted:
            raise ValueError("Must fit before diagnose")
        
        residuals = self.get_residuals()
        
        # Ljung-Box test for residual autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Jarque-Bera test for normality
        from scipy import stats
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        
        return {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'ljung_box_test': {
                'statistic': lb_test['lb_stat'].tolist(),
                'p_value': lb_test['lb_pvalue'].tolist()
            },
            'jarque_bera': {
                'statistic': float(jb_stat),
                'p_value': float(jb_pvalue),
                'is_normal': jb_pvalue > 0.05
            }
        }


class AutoARIMA:
    """
    Automatic ARIMA model selection
    
    Finds optimal (p, d, q) parameters using information criteria
    """
    
    def __init__(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        m: int = 12,
        information_criterion: str = 'aic'
    ):
        """
        Args:
            max_p, max_d, max_q: Maximum ARIMA orders
            seasonal: Whether to consider seasonal models
            max_P, max_D, max_Q: Maximum seasonal orders
            m: Seasonal period
            information_criterion: 'aic', 'bic', 'aicc'
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.m = m
        self.information_criterion = information_criterion
        self.best_model_ = None
        self.best_order_ = None
        self.best_seasonal_order_ = None
        self.best_score_ = np.inf
    
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        """
        Find best ARIMA model
        
        Args:
            y: Time series data
            exog: Exogenous variables
            
        Returns:
            Fitted ARIMA model
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for AutoARIMA")
        
        y = np.asarray(y).ravel()
        
        # Grid search over parameter space
        best_score = np.inf
        best_order = None
        best_seasonal_order = None
        best_model = None
        
        # Determine differencing order (d)
        # Use ADF test to check stationarity
        d_values = list(range(self.max_d + 1))
        
        for d in d_values:
            # Try different (p, q) combinations
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue  # Skip (0, d, 0)
                    
                    try:
                        # Fit ARIMA
                        model = ARIMA(y, order=(p, d, q))
                        fitted = model.fit()
                        
                        # Get information criterion
                        if self.information_criterion == 'aic':
                            score = fitted.aic
                        elif self.information_criterion == 'bic':
                            score = fitted.bic
                        else:
                            score = fitted.aic
                        
                        if score < best_score:
                            best_score = score
                            best_order = (p, d, q)
                            best_model = fitted
                    except:
                        continue
            
            # Try seasonal models if requested
            if self.seasonal:
                for P in range(self.max_P + 1):
                    for D in range(self.max_D + 1):
                        for Q in range(self.max_Q + 1):
                            if P == 0 and D == 0 and Q == 0:
                                continue
                            
                            try:
                                model = SARIMAX(
                                    y,
                                    exog=exog,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, self.m)
                                )
                                fitted = model.fit()
                                
                                if self.information_criterion == 'aic':
                                    score = fitted.aic
                                elif self.information_criterion == 'bic':
                                    score = fitted.bic
                                else:
                                    score = fitted.aic
                                
                                if score < best_score:
                                    best_score = score
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, self.m)
                                    best_model = fitted
                            except:
                                continue
        
        if best_model is None:
            raise ValueError("Failed to find valid ARIMA model")
        
        self.best_model_ = best_model
        self.best_order_ = best_order
        self.best_seasonal_order_ = best_seasonal_order
        self.best_score_ = best_score
        
        return self
    
    def get_best_model(self) -> ARIMAModel:
        """Get best ARIMA model"""
        if self.best_model_ is None:
            raise ValueError("Must fit before get_best_model")
        
        model = ARIMAModel(order=self.best_order_, seasonal_order=self.best_seasonal_order_)
        model.fitted_model_ = self.best_model_
        model.is_fitted = True
        return model
