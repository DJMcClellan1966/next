"""
Time Series Feature Engineering
Create features for time series ML models

Features:
- Lag features
- Rolling statistics (mean, std, min, max)
- Seasonal decomposition
- Time-based features (hour, day, month, etc.)
- Difference features
- Autocorrelation features
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict
import warnings
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Try to import pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Install with: pip install pandas")
    print("Time series features will use numpy implementation")

# Try to import statsmodels
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available for seasonal decomposition")


class TimeSeriesFeatureEngineer:
    """
    Engineer features for time series ML models
    """
    
    def __init__(self):
        self.feature_names_ = []
        self.is_fitted = False
    
    def create_lag_features(
        self,
        y: np.ndarray,
        lags: List[int] = [1, 2, 3, 7, 14, 30]
    ) -> np.ndarray:
        """
        Create lag features
        
        Args:
            y: Time series data
            lags: List of lag values
            
        Returns:
            Array of lag features (n_samples, n_lags)
        """
        y = np.asarray(y).ravel()
        n_samples = len(y)
        max_lag = max(lags)
        
        lag_features = []
        feature_names = []
        
        for lag in lags:
            lag_feature = np.full(n_samples, np.nan)
            lag_feature[lag:] = y[:-lag]
            lag_features.append(lag_feature)
            feature_names.append(f'lag_{lag}')
        
        self.feature_names_.extend(feature_names)
        return np.column_stack(lag_features)
    
    def create_rolling_features(
        self,
        y: np.ndarray,
        windows: List[int] = [3, 7, 14, 30],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> np.ndarray:
        """
        Create rolling window features
        
        Args:
            y: Time series data
            windows: List of window sizes
            functions: List of functions ('mean', 'std', 'min', 'max', 'median')
            
        Returns:
            Array of rolling features
        """
        y = np.asarray(y).ravel()
        n_samples = len(y)
        
        rolling_features = []
        feature_names = []
        
        for window in windows:
            for func in functions:
                rolling_feature = np.full(n_samples, np.nan)
                
                for i in range(window - 1, n_samples):
                    window_data = y[i - window + 1:i + 1]
                    
                    if func == 'mean':
                        rolling_feature[i] = np.mean(window_data)
                    elif func == 'std':
                        rolling_feature[i] = np.std(window_data)
                    elif func == 'min':
                        rolling_feature[i] = np.min(window_data)
                    elif func == 'max':
                        rolling_feature[i] = np.max(window_data)
                    elif func == 'median':
                        rolling_feature[i] = np.median(window_data)
                
                rolling_features.append(rolling_feature)
                feature_names.append(f'rolling_{func}_{window}')
        
        self.feature_names_.extend(feature_names)
        return np.column_stack(rolling_features)
    
    def create_difference_features(
        self,
        y: np.ndarray,
        differences: List[int] = [1, 7, 30]
    ) -> np.ndarray:
        """
        Create difference features
        
        Args:
            y: Time series data
            differences: List of difference periods
            
        Returns:
            Array of difference features
        """
        y = np.asarray(y).ravel()
        n_samples = len(y)
        
        diff_features = []
        feature_names = []
        
        for diff in differences:
            diff_feature = np.full(n_samples, np.nan)
            diff_feature[diff:] = y[diff:] - y[:-diff]
            diff_features.append(diff_feature)
            feature_names.append(f'diff_{diff}')
        
        self.feature_names_.extend(feature_names)
        return np.column_stack(diff_features)
    
    def create_seasonal_features(
        self,
        dates: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Create seasonal/time-based features
        
        Args:
            dates: Array of dates (datetime objects or strings)
            n_samples: Number of samples (if dates not provided)
            
        Returns:
            Array of seasonal features
        """
        if dates is None:
            if n_samples is None:
                raise ValueError("Either dates or n_samples must be provided")
            # Generate dummy dates
            dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D') if PANDAS_AVAILABLE else None
        
        if PANDAS_AVAILABLE and dates is not None:
            if isinstance(dates, np.ndarray):
                dates = pd.to_datetime(dates)
            
            seasonal_features = []
            feature_names = []
            
            # Hour (if datetime has time component)
            if hasattr(dates, 'hour'):
                seasonal_features.append(dates.hour.values)
                feature_names.append('hour')
            
            # Day of week
            seasonal_features.append(dates.dayofweek.values)
            feature_names.append('day_of_week')
            
            # Day of month
            seasonal_features.append(dates.day.values)
            feature_names.append('day_of_month')
            
            # Month
            seasonal_features.append(dates.month.values)
            feature_names.append('month')
            
            # Quarter
            seasonal_features.append(dates.quarter.values)
            feature_names.append('quarter')
            
            # Year
            seasonal_features.append(dates.year.values)
            feature_names.append('year')
            
            # Cyclical encoding (sin/cos)
            seasonal_features.append(np.sin(2 * np.pi * dates.dayofweek / 7))
            feature_names.append('day_of_week_sin')
            seasonal_features.append(np.cos(2 * np.pi * dates.dayofweek / 7))
            feature_names.append('day_of_week_cos')
            
            seasonal_features.append(np.sin(2 * np.pi * dates.month / 12))
            feature_names.append('month_sin')
            seasonal_features.append(np.cos(2 * np.pi * dates.month / 12))
            feature_names.append('month_cos')
            
            self.feature_names_.extend(feature_names)
            return np.column_stack(seasonal_features)
        else:
            # Fallback: return dummy features
            if n_samples is None:
                n_samples = len(dates) if dates is not None else 100
            
            return np.zeros((n_samples, 8))  # Dummy features
    
    def create_autocorrelation_features(
        self,
        y: np.ndarray,
        lags: List[int] = [1, 7, 14, 30]
    ) -> np.ndarray:
        """
        Create autocorrelation features
        
        Args:
            y: Time series data
            lags: List of lag values for autocorrelation
            
        Returns:
            Array of autocorrelation features
        """
        y = np.asarray(y).ravel()
        n_samples = len(y)
        
        acf_features = []
        feature_names = []
        
        # Calculate autocorrelation for each lag
        for lag in lags:
            if lag >= n_samples:
                acf_feature = np.full(n_samples, 0.0)
            else:
                acf_feature = np.full(n_samples, np.nan)
                
                # Calculate autocorrelation
                for i in range(lag, n_samples):
                    y1 = y[i - lag:i]
                    y2 = y[i - lag + 1:i + 1]
                    
                    if len(y1) > 0 and len(y2) > 0:
                        corr = np.corrcoef(y1, y2)[0, 1]
                        acf_feature[i] = corr if not np.isnan(corr) else 0.0
            
            acf_features.append(acf_feature)
            feature_names.append(f'autocorr_{lag}')
        
        self.feature_names_.extend(feature_names)
        return np.column_stack(acf_features)
    
    def create_all_features(
        self,
        y: np.ndarray,
        dates: Optional[np.ndarray] = None,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_differences: bool = True,
        include_seasonal: bool = True,
        include_autocorr: bool = False
    ) -> np.ndarray:
        """
        Create all time series features
        
        Args:
            y: Time series data
            dates: Optional dates array
            include_lags: Include lag features
            include_rolling: Include rolling features
            include_differences: Include difference features
            include_seasonal: Include seasonal features
            include_autocorr: Include autocorrelation features
            
        Returns:
            Array of all features
        """
        all_features = []
        
        if include_lags:
            lag_features = self.create_lag_features(y)
            all_features.append(lag_features)
        
        if include_rolling:
            rolling_features = self.create_rolling_features(y)
            all_features.append(rolling_features)
        
        if include_differences:
            diff_features = self.create_difference_features(y)
            all_features.append(diff_features)
        
        if include_seasonal:
            seasonal_features = self.create_seasonal_features(dates, n_samples=len(y))
            all_features.append(seasonal_features)
        
        if include_autocorr:
            acf_features = self.create_autocorrelation_features(y)
            all_features.append(acf_features)
        
        if len(all_features) > 0:
            combined = np.hstack(all_features)
            self.is_fitted = True
            return combined
        else:
            return np.array([]).reshape(len(y), 0)
