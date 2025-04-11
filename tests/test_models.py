import unittest
import torch
import numpy as np
from models.base import FinancialModel
from models.enhanced import EnhancedFinancialModel
from models.common.layers import TemporalAttention, GatedResidualNetwork
from uncertainty import uncertainty
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uncertainty import UncertaintyEstimator, EnhancedUncertaintyEstimator

class TestLayers(unittest.TestCase):
    """Test cases for neural network layers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 32
        self.seq_len = 60
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_temporal_attention(self):
        """Test TemporalAttention layer"""
        # Create layer
        layer = TemporalAttention(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim).to(self.device)
        
        # Forward pass
        output, attention = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))
        
        # Check attention shape
        self.assertEqual(attention.shape, (self.batch_size, self.seq_len, self.seq_len))
        
        # Check attention sums to 1 along dim 2
        attention_sum = attention.sum(dim=2)
        self.assertTrue(torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5))
    
    def test_gated_residual_network(self):
        """Test GatedResidualNetwork layer"""
        # Create layer
        layer = GatedResidualNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=0.1
        ).to(self.device)
        
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim).to(self.device)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))

class TestBaseModel(unittest.TestCase):
    """Test cases for base financial model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 32
        self.seq_len = 60
        self.input_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
        self.dropout = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = FinancialModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
    def test_forward(self):
        """Test forward pass"""
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim).to(self.device)
        
        # Forward pass
        outputs = self.model(x)
        
        # Check output types
        self.assertIsInstance(outputs, dict)
        self.assertIn('price', outputs)
        self.assertIn('volatility', outputs)
        self.assertIn('signal', outputs)
        
        # Check price output
        self.assertIsInstance(outputs['price'], dict)
        self.assertIn('median', outputs['price'])
        self.assertIn('lower', outputs['price'])
        self.assertIn('upper', outputs['price'])
        
        # Check output shapes
        self.assertEqual(outputs['price']['median'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['price']['lower'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['price']['upper'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['volatility'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['signal'].shape, (self.batch_size, 3))
        
        # Check signal sums to 1
        signal_sum = outputs['signal'].sum(dim=1)
        self.assertTrue(torch.allclose(signal_sum, torch.ones_like(signal_sum), atol=1e-5))

class TestEnhancedModel(unittest.TestCase):
    """Test cases for enhanced financial model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 32
        self.seq_len = 60
        self.input_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
        self.num_heads = 4
        self.dropout = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = EnhancedFinancialModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        ).to(self.device)
        
    def test_forward(self):
        """Test forward pass"""
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim).to(self.device)
        
        # Forward pass
        outputs = self.model(x)
        
        # Check output types
        self.assertIsInstance(outputs, dict)
        self.assertIn('price', outputs)
        self.assertIn('volatility', outputs)
        self.assertIn('signal', outputs)
        self.assertIn('attention', outputs)
        
        # Check price output
        self.assertIsInstance(outputs['price'], dict)
        self.assertIn('median', outputs['price'])
        self.assertIn('lower', outputs['price'])
        self.assertIn('upper', outputs['price'])
        
        # Check output shapes
        self.assertEqual(outputs['price']['median'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['price']['lower'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['price']['upper'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['volatility'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['signal'].shape, (self.batch_size, 3))
        self.assertEqual(outputs['attention'].shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check signal sums to 1
        signal_sum = outputs['signal'].sum(dim=1)
        self.assertTrue(torch.allclose(signal_sum, torch.ones_like(signal_sum), atol=1e-5))
        
        # Check attention sums to 1 along dim 3
        attention_sum = outputs['attention'].sum(dim=3)
        self.assertTrue(torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5))

class TestUncertainty(unittest.TestCase):
    """Test cases for uncertainty estimation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 32
        self.seq_len = 60
        self.input_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
        self.dropout = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = FinancialModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Create uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            model=self.model,
            num_samples=10,
            confidence_level=0.95,
            device=self.device
        )
        
        # Create enhanced uncertainty estimator
        self.enhanced_uncertainty_estimator = EnhancedUncertaintyEstimator(
            model=self.model,
            num_samples=10,
            confidence_level=0.95,
            device=self.device,
            use_conformal=False
        )
        
    def test_monte_carlo_dropout(self):
        """Test Monte Carlo dropout"""
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim).to(self.device)
        
        # Get uncertainty estimates
        uncertainty = self.uncertainty_estimator.monte_carlo_dropout(x)
        
        # Check output types
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('price', uncertainty)
        self.assertIn('volatility', uncertainty)
        self.assertIn('signal', uncertainty)
        
        # Check price output
        self.assertIsInstance(uncertainty['price'], dict)
        self.assertIn('mean', uncertainty['price'])
        self.assertIn('std', uncertainty['price'])
        self.assertIn('lower', uncertainty['price'])
        self.assertIn('upper', uncertainty['price'])
        
        # Check output shapes
        self.assertEqual(uncertainty['price']['mean'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['price']['std'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['price']['lower'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['price']['upper'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['volatility']['mean'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['volatility']['std'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['signal']['mean'].shape, (self.batch_size, 3))
        self.assertEqual(uncertainty['signal']['std'].shape, (self.batch_size, 3))
        self.assertEqual(uncertainty['signal']['entropy'].shape, (self.batch_size, 1))
        
        # Check signal sums to 1
        signal_sum = uncertainty['signal']['mean'].sum(dim=1)
        self.assertTrue(torch.allclose(signal_sum, torch.ones_like(signal_sum), atol=1e-5))
    
    def test_quantile_based_intervals(self):
        """Test quantile-based intervals"""
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim).to(self.device)
        
        # Get uncertainty estimates
        uncertainty = self.uncertainty_estimator.quantile_based_intervals(x)
        
        # Check output types
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('q5', uncertainty)
        self.assertIn('q50', uncertainty)
        self.assertIn('q95', uncertainty)
        
        # Check output shapes
        self.assertEqual(uncertainty['q5'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['q50'].shape, (self.batch_size, 1))
        self.assertEqual(uncertainty['q95'].shape, (self.batch_size, 1))
        
        # Check q5 <= q50 <= q95
        self.assertTrue(torch.all(uncertainty['q5'] <= uncertainty['q50']))
        self.assertTrue(torch.all(uncertainty['q50'] <= uncertainty['q95']))
    
    def test_decompose_uncertainty(self):
        """Test uncertainty decomposition"""
        # Create input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim).to(self.device)
        
        # Get uncertainty estimates
        uncertainty = self.enhanced_uncertainty_estimator.decompose_uncertainty(x)
        
        # Check output types
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('total_uncertainty', uncertainty)
        self.assertIn('aleatoric_uncertainty', uncertainty)
        self.assertIn('epistemic_uncertainty', uncertainty)
        
        # Check output shapes
        self.assertEqual(uncertainty['total_uncertainty'].shape, (self.batch_size,))
        self.assertEqual(uncertainty['aleatoric_uncertainty'].shape, (self.batch_size,))
        self.assertEqual(uncertainty['epistemic_uncertainty'].shape, (self.batch_size,))
        
        # Check total_uncertainty >= aleatoric_uncertainty
        self.assertTrue(torch.all(uncertainty['total_uncertainty'] >= uncertainty['aleatoric_uncertainty']))
        
        # Check total_uncertainty >= epistemic_uncertainty
        self.assertTrue(torch.all(uncertainty['total_uncertainty'] >= uncertainty['epistemic_uncertainty']))
        
        # Check total_uncertainty ~= aleatoric_uncertainty + epistemic_uncertainty
        self.assertTrue(torch.allclose(
            uncertainty['total_uncertainty'],
            uncertainty['aleatoric_uncertainty'] + uncertainty['epistemic_uncertainty'],
            atol=1e-5
        ))

if __name__ == '__main__':
    unittest.main()
