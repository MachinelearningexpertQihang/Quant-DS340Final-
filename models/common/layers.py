import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Union

class TimeSeriesEmbedding(nn.Module):
    """
    Embedding layer for time series data
    """
    def __init__(self, input_dim: int, embedding_dim: int):
        """
        Initialize embedding layer
        
        Args:
            input_dim: Input dimension
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_length, embedding_dim)
        """
        return self.projection(x)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class TimeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention with time awareness
    """
    def __init__(self, 
                embed_dim: int, 
                num_heads: int, 
                dropout: float = 0.1, 
                time_aware: bool = True):
        """
        Initialize time-aware multi-head attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            time_aware: Whether to use time-aware attention
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.time_aware = time_aware
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Time decay parameters (if time-aware)
        if time_aware:
            self.time_decay = nn.Parameter(torch.ones(num_heads, 1, 1))
            self.time_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
               query: torch.Tensor, 
               key: torch.Tensor, 
               value: torch.Tensor, 
               time_intervals: Optional[torch.Tensor] = None,
               attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor of shape (batch_size, seq_length, embed_dim)
            key: Key tensor of shape (batch_size, seq_length, embed_dim)
            value: Value tensor of shape (batch_size, seq_length, embed_dim)
            time_intervals: Time intervals tensor of shape (batch_size, seq_length, 1)
            attn_mask: Attention mask tensor
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply time-aware attention if enabled and time intervals are provided
        if self.time_aware and time_intervals is not None:
            # Reshape time intervals for broadcasting
            time_intervals = time_intervals.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Apply time decay and bias
            time_decay = torch.exp(-self.time_decay * time_intervals.abs())
            scores = scores * time_decay + self.time_bias
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

class GatedResidualNetwork(nn.Module):
    """
    Gated residual network for feature selection and transformation
    """
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                output_dim: Optional[int] = None, 
                dropout: float = 0.1):
        """
        Initialize gated residual network
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension (if None, use input_dim)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.hidden_dim = hidden_dim
        
        # Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.gate = nn.Linear(input_dim, self.output_dim)
        
        # Skip connection (if input and output dimensions differ)
        self.skip_connection = (
            nn.Linear(input_dim, self.output_dim) 
            if input_dim != self.output_dim else nn.Identity()
        )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, ..., input_dim)
            
        Returns:
            Output tensor of shape (batch_size, ..., output_dim)
        """
        # Main branch
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        
        # Gate branch
        gate = torch.sigmoid(self.gate(x))
        
        # Skip connection
        skip = self.skip_connection(x)
        
        # Combine with gating and skip connection
        output = gate * h + (1 - gate) * skip
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        return output

class VariableSelectionNetwork(nn.Module):
    """
    Variable selection network for selecting relevant features
    """
    def __init__(self, 
                input_dims: List[int], 
                hidden_dim: int, 
                dropout: float = 0.1):
        """
        Initialize variable selection network
        
        Args:
            input_dims: List of input dimensions for each feature
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dims = input_dims
        self.num_vars = len(input_dims)
        self.hidden_dim = hidden_dim
        
        # GRNs for each variable
        self.grns = nn.ModuleList([
            GatedResidualNetwork(dim, hidden_dim, hidden_dim, dropout)
            for dim in input_dims
        ])
        
        # GRN for variable selection
        self.selection_grn = GatedResidualNetwork(
            sum(input_dims), hidden_dim, self.num_vars, dropout
        )
        
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: List of input tensors, each of shape (batch_size, ..., input_dim_i)
            
        Returns:
            Tuple of (selected features, selection weights)
        """
        # Process each variable with its own GRN
        processed_vars = [grn(var) for grn, var in zip(self.grns, x)]
        
        # Concatenate all variables for selection
        concat_vars = torch.cat(x, dim=-1)
        
        # Get selection weights
        selection_weights = self.selection_grn(concat_vars)
        selection_weights = F.softmax(selection_weights, dim=-1)
        
        # Apply selection weights
        selected_features = torch.zeros_like(processed_vars[0])
        for i, processed_var in enumerate(processed_vars):
            selected_features += selection_weights[..., i:i+1] * processed_var
        
        return selected_features, selection_weights

class QuantileLoss(nn.Module):
    """
    Quantile loss for quantile regression
    """
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initialize quantile loss
        
        Args:
            quantiles: List of quantiles
        """
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            preds: Predictions tensor of shape (batch_size, num_quantiles)
            target: Target tensor of shape (batch_size, 1)
            
        Returns:
            Loss tensor
        """
        assert preds.size(1) == len(self.quantiles), "Number of predictions must match number of quantiles"
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        # Sum losses across quantiles
        loss = torch.cat(losses, dim=1).mean()
        return loss
