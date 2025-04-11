import pandas as pd
import numpy as np
import yfinance as yf 
from typing import List, Dict, Tuple, Optional, Union
import os
import logging

logger = logging.getLogger(__name__)

class FinancialDataLoader:
    """
    Data loader for financial time series data
    """
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize the data loader
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_data(self, 
                     tickers: List[str], 
                     start_date: str, 
                     end_date: str, 
                     interval: str = "1d",
                     force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download financial data for the given tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            force_download: Force download even if cached data exists
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        results = {}
        
        for ticker in tickers:
            cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}_{interval}.csv")
            
            # Check if cached data exists
            if os.path.exists(cache_file) and not force_download:
                logger.info(f"Loading cached data for {ticker}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            else:
                logger.info(f"Downloading data for {ticker}")
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
                    # Save to cache
                    df.to_csv(cache_file)
                except Exception as e:
                    logger.error(f"Error downloading data for {ticker}: {e}")
                    continue
            
            results[ticker] = df
            
        return results
    
    def load_market_data(self, 
                        market_index: str = "^GSPC", 
                        start_date: str = None, 
                        end_date: str = None) -> pd.DataFrame:
        """
        Load market index data (e.g., S&P 500)
        
        Args:
            market_index: Market index ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with market data
        """
        result = self.download_data([market_index], start_date, end_date)
        return result.get(market_index, pd.DataFrame())
    
    def load_fundamental_data(self, 
                             tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load fundamental data for the given tickers
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to fundamental data DataFrames
        """
        results = {}
        
        for ticker in tickers:
            try:
                # Get ticker info
                ticker_obj = yf.Ticker(ticker)
                
                # Get financial data
                income_stmt = ticker_obj.income_stmt
                balance_sheet = ticker_obj.balance_sheet
                cash_flow = ticker_obj.cashflow
                
                # Combine into a single DataFrame
                fundamental_data = {
                    'income_stmt': income_stmt,
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow
                }
                
                results[ticker] = fundamental_data
                
            except Exception as e:
                logger.error(f"Error loading fundamental data for {ticker}: {e}")
                
        return results
    
    def merge_data(self, 
                  price_data: Dict[str, pd.DataFrame], 
                  fundamental_data: Optional[Dict[str, Dict]] = None) -> Dict[str, pd.DataFrame]:
        """
        Merge price and fundamental data
        
        Args:
            price_data: Dictionary mapping ticker symbols to price DataFrames
            fundamental_data: Dictionary mapping ticker symbols to fundamental data
            
        Returns:
            Dictionary mapping ticker symbols to merged DataFrames
        """
        results = {}
        
        for ticker, price_df in price_data.items():
            merged_df = price_df.copy()
            
            if fundamental_data and ticker in fundamental_data:
                # Process and merge fundamental data
                # This is a simplified example - in practice, you'd need to handle
                # the different reporting frequencies and dates
                fund_data = fundamental_data[ticker]
                
                # Extract key metrics and resample to match price data frequency
                # This is just a placeholder - actual implementation would be more complex
                if 'income_stmt' in fund_data:
                    income = fund_data['income_stmt'].T
                    if not income.empty:
                        # Resample to daily frequency (forward fill)
                        income = income.resample('D').ffill()
                        
                        # Align with price data
                        common_dates = merged_df.index.intersection(income.index)
                        if not common_dates.empty:
                            for col in ['TotalRevenue', 'NetIncome']:
                                if col in income.columns:
                                    merged_df.loc[common_dates, f'fundamental_{col}'] = income.loc[common_dates, col]
            
            results[ticker] = merged_df
            
        return results

