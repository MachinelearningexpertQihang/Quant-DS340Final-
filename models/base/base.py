import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
from models.common.layers import AttentionLayer, TemporalConvLayer

class FinancialForecastingModel(nn.Module):
    """
    Base financial forecasting model
    
    This model predicts:
    1. Price movements (regression)
    2. Volatility (regression)
    3. Trading signals (classification)
    """
    def __init__(self, 
                input_dim: int = 5, 
                hidden_dim: int = 64, 
                output_dim: int = 1,
                num_layers: int = 2,
                dropout: float = 0.2,
                use_attention: bool = True):
        """
        Initialize model
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention
        """
        super().__init__()
        
        # Save parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Feature extraction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Price prediction (with uncertainty)
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * 3)  # 3 outputs for lower, median, upper
        )
        
        # Volatility prediction
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Trading signal prediction (buy, hold, sell)
        self.signal_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 classes: buy, hold, sell
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Dictionary with predictions:
                - price: Dictionary with lower, median, upper bounds
                - volatility: Predicted volatility
                - signal: Trading signal probabilities
        """
        # Feature extraction
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]
        
        # Price prediction with uncertainty
        price_pred = self.price_predictor(context)
        
        # Split into lower, median, upper bounds
        batch_size = x.size(0)
        price_pred = price_pred.view(batch_size, self.output_dim, 3)
        price_lower = price_pred[:, :, 0]
        price_median = price_pred[:, :, 1]
        price_upper = price_pred[:, :, 2]
        
        # Ensure lower <= median <= upper
        price_lower = torch.min(price_lower, price_median)
        price_upper = torch.max(price_upper, price_median)
        
        # Volatility prediction
        volatility = self.volatility_predictor(context)
        
        # Trading signal prediction
        signal_logits = self.signal_predictor(context)
        signal_probs = F.softmax(signal_logits, dim=1)
        
        return {
            'price': {
                'lower': price_lower,
                'median': price_median,
                'upper': price_upper
            },
            'volatility': volatility,
            'signal': signal_probs
        }

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for financial forecasting
    
    Combines:
    1. Quantile loss for price prediction
    2. MSE loss for volatility prediction
    3. Cross-entropy loss for trading signal classification
    """
    def __init__(self, 
                price_weight: float = 1.0,
                volatility_weight: float = 0.5,
                signal_weight: float = 1.0):
        """
        Initialize loss function
        
        Args:
            price_weight: Weight for price prediction loss
            volatility_weight: Weight for volatility prediction loss
            signal_weight: Weight for signal prediction loss
        """
        super().__init__()
        self.price_weight = price_weight
        self.volatility_weight = volatility_weight
        self.signal_weight = signal_weight
        
        # Loss functions
        self.volatility_loss_fn = nn.MSELoss()
        self.signal_loss_fn = nn.CrossEntropyLoss()
    
    def quantile_loss(self, 
                     preds: Dict[str, torch.Tensor], 
                     target: torch.Tensor,
                     q_lower: float = 0.1,
                     q_median: float = 0.5,
                     q_upper: float = 0.9) -> torch.Tensor:
        """
        Quantile loss for price prediction with uncertainty
        
        Args:
            preds: Dictionary with lower, median, upper predictions
            target: Target values
            q_lower: Lower quantile
            q_median: Median quantile
            q_upper: Upper quantile
            
        Returns:
            Quantile loss
        """
        # Extract predictions
        y_lower = preds['lower']
        y_median = preds['median']
        y_upper = preds['upper']
        
        # Calculate quantile losses
        lower_loss = self._quantile_loss(y_lower, target, q_lower)
        median_loss = self._quantile_loss(y_median, target, q_median)
        upper_loss = self._quantile_loss(y_upper, target, q_upper)
        
        # Combine losses
        return lower_loss + median_loss + upper_loss
    
    def _quantile_loss(self, 
                      preds: torch.Tensor, 
                      target: torch.Tensor, 
                      q: float) -> torch.Tensor:
        """
        Quantile loss for a single quantile
        
        Args:
            preds: Predictions
            target: Target values
            q: Quantile
            
        Returns:
            Quantile loss
        """
        errors = target - preds
        return torch.mean(torch.max(q * errors, (q - 1) * errors))
    
    def forward(self, 
               outputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], 
               targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass
        
        Args:
            outputs: Dictionary with model outputs
            targets: Dictionary with target values
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Price prediction loss
        price_loss = self.quantile_loss(outputs['price'], targets['price'])
        
        # Volatility prediction loss
        volatility_loss = self.volatility_loss_fn(outputs['volatility'], targets['volatility'])
        
        # Trading signal prediction loss
        signal_loss = self.signal_loss_fn(outputs['signal'], targets['signal'])
        
        # Combine losses
        total_loss = (
            self.price_weight * price_loss +
            self.volatility_weight * volatility_loss +
            self.signal_weight * signal_loss
        )
        
        # Return total loss and individual losses
        loss_dict = {
            'price_loss': price_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'signal_loss': signal_loss.item()
        }
        
        return total_loss, loss_dict
