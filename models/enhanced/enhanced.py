import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
from models.common.layers import AttentionLayer, TemporalConvLayer, GatedResidualNetwork
from models.base import FinancialForecastingModel, MultiTaskLoss

class EnhancedFinancialModel(FinancialForecastingModel):
    """
    Enhanced financial forecasting model with advanced features:
    
    1. Temporal convolutional layers
    2. Multi-head attention
    3. Gated residual connections
    4. Uncertainty calibration
    5. Adversarial training support
    """
    def __init__(self, 
                input_dim: int = 5, 
                hidden_dim: int = 128, 
                output_dim: int = 1,
                num_layers: int = 3,
                dropout: float = 0.3,
                use_attention: bool = True,
                use_tcn: bool = True,
                num_heads: int = 4,
                use_gated_residual: bool = True):
        """
        Initialize enhanced model
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention
            use_tcn: Whether to use temporal convolutional networks
            num_heads: Number of attention heads
            use_gated_residual: Whether to use gated residual connections
        """
        # Initialize base model
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Save additional parameters
        self.use_tcn = use_tcn
        self.num_heads = num_heads
        self.use_gated_residual = use_gated_residual
        
        # Replace single-head attention with multi-head if enabled
        if use_attention:
            del self.attention  # Remove base attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Add temporal convolutional network if enabled
        if use_tcn:
            self.tcn = TemporalConvLayer(
                input_dim=input_dim,
                output_dim=hidden_dim,
                kernel_size=3,
                dilation=2
            )
            
            # Fusion layer to combine LSTM and TCN features
            self.fusion = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim * 2)
        
        # Replace predictors with gated residual networks if enabled
        if use_gated_residual:
            # Price predictor
            self.price_predictor = nn.Sequential(
                GatedResidualNetwork(hidden_dim * 2, hidden_dim, dropout),
                nn.Linear(hidden_dim, output_dim * 3)
            )
            
            # Volatility predictor
            self.volatility_predictor = nn.Sequential(
                GatedResidualNetwork(hidden_dim * 2, hidden_dim, dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus()
            )
            
            # Signal predictor
            self.signal_predictor = nn.Sequential(
                GatedResidualNetwork(hidden_dim * 2, hidden_dim, dropout),
                nn.Linear(hidden_dim, 3)
            )
        
        # Uncertainty calibration layer
        self.uncertainty_calibration = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Scale and shift parameters
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
                - attention_weights: Attention weights if attention is enabled
        """
        # Feature extraction with LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply TCN if enabled
        if self.use_tcn:
            tcn_out = self.tcn(x)
            
            # Ensure tcn_out has the same sequence length as lstm_out
            if tcn_out.size(1) != lstm_out.size(1):
                # Pad or truncate tcn_out to match lstm_out
                if tcn_out.size(1) < lstm_out.size(1):
                    padding = torch.zeros(
                        tcn_out.size(0),
                        lstm_out.size(1) - tcn_out.size(1),
                        tcn_out.size(2),
                        device=tcn_out.device
                    )
                    tcn_out = torch.cat([tcn_out, padding], dim=1)
                else:
                    tcn_out = tcn_out[:, :lstm_out.size(1), :]
            
            # Concatenate LSTM and TCN features
            combined_features = torch.cat([lstm_out, tcn_out], dim=2)
            
            # Fuse features
            features = self.fusion(combined_features)
        else:
            features = lstm_out
        
        # Apply attention if enabled
        if self.use_attention:
            # Multi-head attention
            attn_output, attention_weights = self.attention(
                features, features, features
            )
            context = attn_output[:, -1, :]  # Use last timestep
        else:
            # Use last hidden state
            context = features[:, -1, :]
            attention_weights = None
        
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
        
        # Calibrate uncertainty
        uncertainty_params = self.uncertainty_calibration(context)
        scale = F.softplus(uncertainty_params[:, 0:1])  # Ensure positive
        shift = uncertainty_params[:, 1:2]
        
        # Apply calibration
        uncertainty_range = price_upper - price_lower
        calibrated_lower = price_median - scale * (price_median - price_lower) - shift
        calibrated_upper = price_median + scale * (price_upper - price_median) + shift
        
        # Volatility prediction
        volatility = self.volatility_predictor(context)
        
        # Trading signal prediction
        signal_logits = self.signal_predictor(context)
        signal_probs = F.softmax(signal_logits, dim=1)
        
        return {
            'price': {
                'lower': calibrated_lower,
                'median': price_median,
                'upper': calibrated_upper
            },
            'volatility': volatility,
            'signal': signal_probs,
            'attention_weights': attention_weights
        }

class EnhancedMultiTaskLoss(MultiTaskLoss):
    """
    Enhanced multi-task loss function with:
    
    1. Adaptive task weighting
    2. Focal loss for imbalanced signal classes
    3. Uncertainty-aware loss components
    4. Adversarial regularization
    """
    def __init__(self, 
                price_weight: float = 1.0,
                volatility_weight: float = 0.5,
                signal_weight: float = 1.0,
                focal_gamma: float = 2.0,
                adaptive_weights: bool = True,
                adversarial_lambda: float = 0.1):
        """
        Initialize enhanced loss function
        
        Args:
            price_weight: Initial weight for price prediction loss
            volatility_weight: Initial weight for volatility prediction loss
            signal_weight: Initial weight for signal prediction loss
            focal_gamma: Gamma parameter for focal loss
            adaptive_weights: Whether to use adaptive task weighting
            adversarial_lambda: Weight for adversarial regularization
        """
        super().__init__(
            price_weight=price_weight,
            volatility_weight=volatility_weight,
            signal_weight=signal_weight
        )
        
        self.focal_gamma = focal_gamma
        self.adaptive_weights = adaptive_weights
        self.adversarial_lambda = adversarial_lambda
        
        # Initialize log task weights for adaptive weighting
        if adaptive_weights:
            self.log_price_weight = nn.Parameter(torch.tensor(np.log(price_weight)))
            self.log_volatility_weight = nn.Parameter(torch.tensor(np.log(volatility_weight)))
            self.log_signal_weight = nn.Parameter(torch.tensor(np.log(signal_weight)))
    
    def focal_loss(self, 
                  probs: torch.Tensor, 
                  targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss for imbalanced classification
        
        Args:
            probs: Class probabilities
            targets: Target class indices
            
        Returns:
            Focal loss
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
        
        # Calculate focal weights
        p_t = (targets_one_hot * probs).sum(dim=1)
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(probs, targets, reduction='none')
        
        # Apply focal weights
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def uncertainty_aware_mse(self, 
                             preds: torch.Tensor, 
                             targets: torch.Tensor,
                             uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Uncertainty-aware MSE loss
        
        Args:
            preds: Predictions
            targets: Target values
            uncertainty: Uncertainty estimates
            
        Returns:
            Uncertainty-aware MSE loss
        """
        # Ensure uncertainty is positive
        uncertainty = F.softplus(uncertainty) + 1e-6
        
        # Calculate squared error
        squared_error = (preds - targets) ** 2
        
        # Weight by inverse uncertainty
        weighted_error = squared_error / uncertainty
        
        # Add log uncertainty term (prevents uncertainty from growing too large)
        loss = weighted_error + torch.log(uncertainty)
        
        return loss.mean()
    
    def forward(self, 
               outputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], 
               targets: Dict[str, torch.Tensor],
               model=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass
        
        Args:
            outputs: Dictionary with model outputs
            targets: Dictionary with target values
            model: Model for adversarial regularization (optional)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Price prediction loss
        price_loss = self.quantile_loss(outputs['price'], targets['price'])
        
        # Volatility prediction loss with uncertainty awareness
        volatility_range = outputs['price']['upper'] - outputs['price']['lower']
        volatility_loss = self.uncertainty_aware_mse(
            outputs['volatility'], 
            targets['volatility'],
            volatility_range
        )
        
        # Trading signal prediction loss with focal loss
        if self.focal_gamma > 0:
            signal_loss = self.focal_loss(outputs['signal'], targets['signal'])
        else:
            signal_loss = self.signal_loss_fn(outputs['signal'], targets['signal'])
        
        # Get task weights
        if self.adaptive_weights:
            price_weight = torch.exp(self.log_price_weight)
            volatility_weight = torch.exp(self.log_volatility_weight)
            signal_weight = torch.exp(self.log_signal_weight)
        else:
            price_weight = self.price_weight
            volatility_weight = self.volatility_weight
            signal_weight = self.signal_weight
        
        # Combine losses
        total_loss = (
            price_weight * price_loss +
            volatility_weight * volatility_loss +
            signal_weight * signal_loss
        )
        
        # Add adversarial regularization if model is provided
        if model is not None and self.adversarial_lambda > 0:
            # L2 regularization on attention weights
            if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
                attn_weights = outputs['attention_weights']
                # Encourage sparse attention
                attn_reg = torch.norm(attn_weights, p=1)
                total_loss = total_loss + self.adversarial_lambda * attn_reg
        
        # Return total loss and individual losses
        loss_dict = {
            'price_loss': price_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'signal_loss': signal_loss.item()
        }
        
        if self.adaptive_weights:
            loss_dict.update({
                'price_weight': price_weight.item(),
                'volatility_weight': volatility_weight.item(),
                'signal_weight': signal_weight.item()
            })
        
        return total_loss, loss_dict