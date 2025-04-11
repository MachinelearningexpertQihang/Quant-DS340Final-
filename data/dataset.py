import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Dict, Tuple, Optional, Union
import logging

class FinancialDataset(Dataset):
    """
    Dataset for financial time series data
    """
    def __init__(self, 
                data: pd.DataFrame, 
                seq_length: int = 60,
                target_columns: List[str] = ['Close'],
                feature_columns: Optional[List[str]] = None,
                target_horizon: int = 1,
                transform: bool = True,
                add_time_features: bool = False,
                add_technical_indicators: bool = False,
                augment: bool = False,
                augment_prob: float = 0.0,
                scaler_type: str = 'standard'):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with financial data
            seq_length: Length of input sequences
            target_columns: List of target column names
            feature_columns: List of feature column names (if None, use all columns)
            target_horizon: Prediction horizon in time steps
            transform: Whether to standardize features
            add_time_features: Whether to add time-based features
            add_technical_indicators: Whether to add technical indicators
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying augmentation
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.data = data.copy()
        self.seq_length = seq_length
        self.target_horizon = target_horizon
        self.transform = transform
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Add time features if requested
        if add_time_features:
            self._add_time_features()
        
        # Add technical indicators if requested
        if add_technical_indicators:
            self._add_technical_indicators()
        
        # Set feature columns
        if feature_columns is None:
            self.feature_columns = list(self.data.columns)
        else:
            self.feature_columns = feature_columns
            
        # Ensure all feature columns exist in the data
        missing_cols = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Feature columns {missing_cols} not found in data")
        
        # Set target columns
        self.target_columns = target_columns
        
        # Ensure all target columns exist in the data
        missing_cols = [col for col in self.target_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Target columns {missing_cols} not found in data")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Extract features and targets
        self.features = self.data[self.feature_columns].values
        self.targets = self.data[self.target_columns].values
        
        # Calculate time intervals between data points if timestamps are available
        if 'timestamp' in self.data.columns:
            self.time_intervals = np.diff(self.data['timestamp'].values, prepend=self.data['timestamp'].values[0])
            self.time_intervals = self.time_intervals.reshape(-1, 1)
        else:
            self.time_intervals = None
        
        # Standardize features if requested
        if self.transform:
            self._standardize_features(scaler_type)
        
        # Create sequences
        self._create_sequences()
        
        # Log dataset info
        logging.info(f"Created dataset with {len(self.X)} sequences")
        logging.info(f"Feature columns: {self.feature_columns}")
        logging.info(f"Target columns: {self.target_columns}")
        
    def _add_time_features(self):
        """Add time-based features"""
        # Check if index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except:
                logging.warning("Could not convert index to datetime, skipping time features")
                return
        
        # Extract time features
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_month'] = self.data.index.day
        self.data['week_of_year'] = self.data.index.isocalendar().week
        self.data['month'] = self.data.index.month
        self.data['quarter'] = self.data.index.quarter
        self.data['year'] = self.data.index.year
        
        # Add cyclical encoding for periodic features
        self.data['sin_day_of_week'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['cos_day_of_week'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['sin_month'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['cos_month'] = np.cos(2 * np.pi * self.data['month'] / 12)
        
        # Add timestamp column for time interval calculation
        self.data['timestamp'] = self.data.index.astype(np.int64) // 10**9  # Convert to Unix timestamp
    
    def _add_technical_indicators(self):
        """Add technical indicators"""
        # Check if required price columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.data.columns for col in required_cols):
            logging.warning("Required price columns missing, skipping technical indicators")
            return
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            self.data[f'ma_{window}'] = self.data['Close'].rolling(window=window).mean()
            self.data[f'ma_ratio_{window}'] = self.data['Close'] / self.data[f'ma_{window}']
        
        # Exponential moving averages
        for window in [5, 10, 20, 50]:
            self.data[f'ema_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        
        # Bollinger Bands
        for window in [20]:
            self.data[f'bb_middle_{window}'] = self.data['Close'].rolling(window=window).mean()
            self.data[f'bb_std_{window}'] = self.data['Close'].rolling(window=window).std()
            self.data[f'bb_upper_{window}'] = self.data[f'bb_middle_{window}'] + 2 * self.data[f'bb_std_{window}']
            self.data[f'bb_lower_{window}'] = self.data[f'bb_middle_{window}'] - 2 * self.data[f'bb_std_{window}']
            self.data[f'bb_width_{window}'] = (self.data[f'bb_upper_{window}'] - self.data[f'bb_lower_{window}']) / self.data[f'bb_middle_{window}']
            self.data[f'bb_z_{window}'] = (self.data['Close'] - self.data[f'bb_middle_{window}']) / self.data[f'bb_std_{window}']
        
        # RSI (Relative Strength Index)
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        for window in [14]:
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            self.data[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['macd'] = ema12 - ema26
        self.data['macd_signal'] = self.data['macd'].ewm(span=9, adjust=False).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']
        
        # Volume indicators
        self.data['volume_ma_10'] = self.data['Volume'].rolling(window=10).mean()
        self.data['volume_ratio'] = self.data['Volume'] / self.data['volume_ma_10']
        
        # Price momentum
        for window in [1, 5, 10, 20]:
            self.data[f'momentum_{window}'] = self.data['Close'].pct_change(window)
        
        # Volatility
        for window in [5, 10, 20]:
            self.data[f'volatility_{window}'] = self.data['Close'].pct_change().rolling(window=window).std()
    
    def _handle_missing_values(self):
        """Handle missing values in the data"""
        # Check for missing values
        missing_count = self.data.isna().sum()
        if missing_count.sum() > 0:
            logging.warning(f"Found {missing_count.sum()} missing values")
            
            # Forward fill first (for time series data)
            self.data = self.data.ffill()
            
            # Then backward fill any remaining NaNs
            self.data = self.data.bfill()
            
            # Check if there are still missing values
            missing_count = self.data.isna().sum()
            if missing_count.sum() > 0:
                logging.warning(f"Still have {missing_count.sum()} missing values after filling")
                
                # Drop rows with missing values as a last resort
                self.data = self.data.dropna()
                logging.warning(f"Dropped rows with missing values, new shape: {self.data.shape}")
    
    def _standardize_features(self, scaler_type='standard'):
        """Standardize features"""
        # Choose scaler based on type
        if scaler_type == 'robust':
            self.scaler = RobustScaler()  # More robust to outliers
        else:
            self.scaler = StandardScaler()
            
        # Fit and transform features
        self.features = self.scaler.fit_transform(self.features)
    
    def _create_sequences(self):
        """Create input sequences and target values"""
        X, y, times = [], [], []
        
        # Create sequences
        for i in range(len(self.features) - self.seq_length - self.target_horizon + 1):
            # Input sequence
            X.append(self.features[i:i+self.seq_length])
            
            # Target value (future price)
            target_idx = i + self.seq_length + self.target_horizon - 1
            y.append(self.targets[target_idx])
            
            # Time intervals if available
            if self.time_intervals is not None:
                times.append(self.time_intervals[i:i+self.seq_length])
        
        # Convert to numpy arrays
        self.X = np.array(X)
        self.y = np.array(y)
        
        if self.time_intervals is not None:
            self.times = np.array(times)
        else:
            self.times = None
    
    def __len__(self):
        """Return dataset length"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Get item by index"""
        # Get sequence and target
        X = self.X[idx]
        y = self.y[idx]
        
        # Apply data augmentation if enabled
        if self.augment and np.random.random() < self.augment_prob:
            X = self._augment_sequence(X)
        
        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'price': torch.tensor(y[0], dtype=torch.float32).reshape(1),
            'volatility': torch.tensor(y[1] if len(y) > 1 else 0.0, dtype=torch.float32).reshape(1),
            'signal': torch.tensor(y[2] if len(y) > 2 else 0, dtype=torch.long)
        }
        
        # Add time intervals if available
        if self.times is not None:
            times = self.times[idx]
            times = torch.tensor(times, dtype=torch.float32)
            return X, target, times
        
        return X, target
    
    def _augment_sequence(self, sequence):
        """Apply data augmentation to a sequence"""
        # Choose a random augmentation method
        aug_method = np.random.choice([
            'jitter', 'scaling', 'magnitude_warp', 'time_warp', 'window_slice'
        ])
        
        if aug_method == 'jitter':
            # Add random noise
            noise = np.random.normal(0, 0.01, sequence.shape)
            return sequence + noise
        
        elif aug_method == 'scaling':
            # Apply random scaling
            scaling_factor = np.random.normal(1.0, 0.1, (1, sequence.shape[1]))
            return sequence * scaling_factor
        
        elif aug_method == 'magnitude_warp':
            # Warp the magnitude of the sequence
            window_size = sequence.shape[0] // 10
            knot_points = np.random.choice(sequence.shape[0], 3)
            knot_values = np.random.normal(1.0, 0.2, (3, sequence.shape[1]))
            
            warps = np.ones_like(sequence)
            for i, point in enumerate(knot_points):
                if i < len(knot_points) - 1:
                    window = slice(point, knot_points[i+1])
                    warps[window] = knot_values[i]
            
            return sequence * warps
        
        elif aug_method == 'time_warp':
            # Simple time warping by skipping or repeating some points
            if np.random.random() < 0.5:
                # Skip some points
                indices = np.random.choice(sequence.shape[0], size=int(sequence.shape[0] * 0.9), replace=False)
                indices = np.sort(indices)
                warped = sequence[indices]
                
                # Resize back to original length
                return np.resize(warped, sequence.shape)
            else:
                # Repeat some points
                indices = np.random.choice(sequence.shape[0], size=int(sequence.shape[0] * 1.1), replace=True)
                indices = np.sort(indices)
                warped = sequence[indices[:sequence.shape[0]]]
                return warped
        
        elif aug_method == 'window_slice':
            # Take a random window slice and resize
            window_size = int(sequence.shape[0] * np.random.uniform(0.7, 0.9))
            start = np.random.randint(0, sequence.shape[0] - window_size)
            window = sequence[start:start+window_size]
            
            # Resize to original length
            resized = np.zeros_like(sequence)
            for i in range(sequence.shape[1]):
                resized[:, i] = np.interp(
                    np.linspace(0, 1, sequence.shape[0]),
                    np.linspace(0, 1, window.shape[0]),
                    window[:, i]
                )
            return resized
        
        return sequence

def create_dataset_from_config(data, config):
    """
    Create dataset from configuration
    
    Args:
        data: DataFrame with financial data
        config: Configuration dictionary
        
    Returns:
        FinancialDataset instance
    """
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    
    # Determine whether to use enhanced features
    use_enhanced = data_config.get('use_enhanced_dataset', False)
    
    # Create dataset with appropriate parameters
    dataset = FinancialDataset(
        data=data,
        seq_length=model_config.get('seq_length', 60),
        target_columns=data_config.get('target_columns', ['Close']),
        feature_columns=data_config.get('feature_columns', None),
        target_horizon=data_config.get('target_horizon', 1),
        transform=True,
        add_time_features=use_enhanced and data_config.get('add_time_features', False),
        add_technical_indicators=use_enhanced and data_config.get('add_technical_indicators', False),
        augment=use_enhanced,
        augment_prob=0.5 if use_enhanced else 0.0,
        scaler_type='robust' if use_enhanced else 'standard'
    )
    
    return dataset
