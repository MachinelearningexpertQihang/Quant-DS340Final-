import unittest
import torch
import numpy as np
import pandas as pd
from data.loader import FinancialDataLoader
from data.dataset import FinancialDataset, TimeSeriesSampler
from torch.utils.data import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for financial data loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_path = 'data/sample_data.csv'
        self.start_date = '2010-01-01'
        self.end_date = '2020-12-31'
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Create sample data
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        n_dates = len(dates)
        
        data = []
        for symbol in self.symbols:
            # Generate random prices
            prices = np.random.randn(n_dates).cumsum() + 100
            
            # Generate random volumes
            volumes = np.random.randint(1000000, 10000000, size=n_dates)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Symbol': symbol,
                'Open': prices,
                'High': prices * (1 + np.random.rand(n_dates) * 0.02),
                'Low': prices * (1 - np.random.rand(n_dates) * 0.02),
                'Close': prices * (1 + np.random.randn(n_dates) * 0.01),
                'Volume': volumes
            })
            
            data.append(df)
        
        # Concatenate DataFrames
        self.sample_data = pd.concat(data, ignore_index=True)
        
        # Save sample data
        self.sample_data.to_csv(self.data_path, index=False)
        
        # Create data loader
        self.data_loader = FinancialDataLoader(
            data_path=self.data_path,
            start_date=self.start_date,
            end_date=self.end_date,
            symbols=self.symbols
        )
    
    def test_load_data(self):
        """Test loading data"""
        # Load data
        data = self.data_loader.load_data()
        
        # Check data type
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check columns
        expected_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.assertListEqual(list(data.columns), expected_columns)
        
        # Check symbols
        self.assertSetEqual(set(data['Symbol'].unique()), set(self.symbols))
        
        # Check date range
        self.assertTrue(data['Date'].min() >= pd.Timestamp(self.start_date))
        self.assertTrue(data['Date'].max() <= pd.Timestamp(self.end_date))
    
    def test_calculate_returns(self):
        """Test calculating returns"""
        # Load data
        data = self.data_loader.load_data()
        
        # Calculate returns
        returns = self.data_loader.calculate_returns(data)
        
        # Check returns type
        self.assertIsInstance(returns, pd.DataFrame)
        
        # Check returns columns
        expected_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
        self.assertListEqual(list(returns.columns), expected_columns)
        
        # Check returns values
        self.assertTrue(np.isfinite(returns['Returns']).all())
    
    def test_calculate_features(self):
        """Test calculating features"""
        # Load data
        data = self.data_loader.load_data()
        
        # Calculate returns
        returns = self.data_loader.calculate_returns(data)
        
        # Calculate features
        features = self.data_loader.calculate_features(returns)
        
        # Check features type
        self.assertIsInstance(features, pd.DataFrame)
        
        # Check features columns
        expected_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns',
                           'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50',
                           'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                           'Upper_Band', 'Middle_Band', 'Lower_Band']
        self.assertListEqual(list(features.columns), expected_columns)
        
        # Check features values
        self.assertTrue(np.isfinite(features.drop(['Date', 'Symbol'], axis=1)).all().all())
    
    def test_split_data(self):
        """Test splitting data"""
        # Load data
        data = self.data_loader.load_data()
        
        # Calculate returns
        returns = self.data_loader.calculate_returns(data)
        
        # Calculate features
        features = self.data_loader.calculate_features(returns)
        
        # Split data
        train_data, val_data, test_data = self.data_loader.split_data(features)
        
        # Check data types
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(val_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        
        # Check data sizes
        self.assertTrue(len(train_data) > 0)
        self.assertTrue(len(val_data) > 0)
        self.assertTrue(len(test_data) > 0)
        
        # Check data dates
        self.assertTrue(train_data['Date'].max() < val_data['Date'].min())
        self.assertTrue(val_data['Date'].max() < test_data['Date'].min())

class TestDataset(unittest.TestCase):
    """Test cases for financial dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        n_samples = 1000
        n_features = 10
        
        # Create features
        self.features = np.random.randn(n_samples, n_features)
        
        # Create targets
        self.prices = np.random.randn(n_samples, 1).cumsum(axis=0) + 100
        self.volatility = np.abs(np.random.randn(n_samples, 1))
        self.signals = np.random.randint(0, 3, size=(n_samples, 1))
        
        # Create dataset
        self.dataset = FinancialDataset(
            features=self.features,
            prices=self.prices,
            volatility=self.volatility,
            signals=self.signals
        )
        
        # Create time series sampler
        self.seq_len = 60
        self.sampler = TimeSeriesSampler(
            dataset=self.dataset,
            seq_len=self.seq_len,
            batch_size=32,
            shuffle=True
        )
    
    def test_dataset(self):
        """Test dataset"""
        # Check dataset length
        self.assertEqual(len(self.dataset), len(self.features))
        
        # Check dataset getitem
        item = self.dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 2)
        
        # Check features
        x = item[0]
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (self.features.shape[1],))
        
        # Check targets
        y = item[1]
        self.assertIsInstance(y, dict)
        self.assertIn('price', y)
        self.assertIn('volatility', y)
        self.assertIn('signal', y)
        
        # Check target shapes
        self.assertEqual(y['price'].shape, (1,))
        self.assertEqual(y['volatility'].shape, (1,))
        self.assertEqual(y['signal'].shape, (1,))
    
    def test_sampler(self):
        """Test time series sampler"""
        # Check sampler length
        self.assertEqual(len(self.sampler), (len(self.dataset) - self.seq_len) // self.sampler.batch_size)
        
        # Check sampler getitem
        batch = next(iter(self.sampler))
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 2)
        
        # Check batch features
        x_batch = batch[0]
        self.assertIsInstance(x_batch, torch.Tensor)
        self.assertEqual(x_batch.shape, (self.sampler.batch_size, self.seq_len, self.features.shape[1]))
        
        # Check batch targets
        y_batch = batch[1]
        self.assertIsInstance(y_batch, dict)
        self.assertIn('price', y_batch)
        self.assertIn('volatility', y_batch)
        self.assertIn('signal', y_batch)
        
        # Check batch target shapes
        self.assertEqual(y_batch['price'].shape, (self.sampler.batch_size, 1))
        self.assertEqual(y_batch['volatility'].shape, (self.sampler.batch_size, 1))
        self.assertEqual(y_batch['signal'].shape, (self.sampler.batch_size, 1))

if __name__ == '__main__':
    unittest.main()