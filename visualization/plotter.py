import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FinancialPlotter:
    """
    Plotter for financial time series data and model predictions
    """
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize plotter
        
        Args:
            figsize: Figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        
        # Set style
        plt.style.use(style)
        
        # Create custom colormap for attention
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention_cmap', ['#f7fbff', '#08306b']
        )
    
    def plot_price_prediction(self, 
                             dates: np.ndarray, 
                             actual_prices: np.ndarray, 
                             predicted_prices: np.ndarray,
                             uncertainty: Optional[np.ndarray] = None,
                             title: str = 'Price Prediction',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot price prediction
        
        Args:
            dates: Array of dates
            actual_prices: Array of actual prices
            predicted_prices: Array of predicted prices
            uncertainty: Array of prediction uncertainties
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual prices
        ax.plot(dates, actual_prices, label='Actual', color='#1f77b4', linewidth=2)
        
        # Plot predicted prices
        ax.plot(dates, predicted_prices, label='Predicted', color='#ff7f0e', linewidth=2)
        
        # Plot uncertainty if provided
        if uncertainty is not None:
            ax.fill_between(
                dates,
                predicted_prices - 2 * uncertainty,
                predicted_prices + 2 * uncertainty,
                color='#ff7f0e',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volatility_prediction(self, 
                                 dates: np.ndarray, 
                                 actual_volatility: np.ndarray, 
                                 predicted_volatility: np.ndarray,
                                 uncertainty: Optional[np.ndarray] = None,
                                 title: str = 'Volatility Prediction',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot volatility prediction
        
        Args:
            dates: Array of dates
            actual_volatility: Array of actual volatility
            predicted_volatility: Array of predicted volatility
            uncertainty: Array of prediction uncertainties
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual volatility
        ax.plot(dates, actual_volatility, label='Actual', color='#1f77b4', linewidth=2)
        
        # Plot predicted volatility
        ax.plot(dates, predicted_volatility, label='Predicted', color='#ff7f0e', linewidth=2)
        
        # Plot uncertainty if provided
        if uncertainty is not None:
            ax.fill_between(
                dates,
                predicted_volatility - 2 * uncertainty,
                predicted_volatility + 2 * uncertainty,
                color='#ff7f0e',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_signal_prediction(self, 
                              dates: np.ndarray, 
                              actual_signals: np.ndarray, 
                              predicted_signals: np.ndarray,
                              title: str = 'Trading Signal Prediction',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trading signal prediction
        
        Args:
            dates: Array of dates
            actual_signals: Array of actual signals (0: Sell, 1: Hold, 2: Buy)
            predicted_signals: Array of predicted signals
            
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Define signal colors
        signal_colors = {
            0: '#d62728',  # Sell: Red
            1: '#7f7f7f',  # Hold: Gray
            2: '#2ca02c'   # Buy: Green
        }
        
        # Plot actual signals
        for signal in [0, 1, 2]:
            mask = actual_signals == signal
            if np.any(mask):
                ax.scatter(
                    dates[mask],
                    np.ones_like(dates[mask]) * 0.1,
                    color=signal_colors[signal],
                    marker='o',
                    s=50,
                    label=f'Actual {["Sell", "Hold", "Buy"][signal]}'
                )
        
        # Plot predicted signals
        for signal in [0, 1, 2]:
            mask = predicted_signals == signal
            if np.any(mask):
                ax.scatter(
                    dates[mask],
                    np.ones_like(dates[mask]) * 0.2,
                    color=signal_colors[signal],
                    marker='x',
                    s=50,
                    label=f'Predicted {["Sell", "Hold", "Buy"][signal]}'
                )
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_yticks([0.1, 0.2])
        ax.set_yticklabels(['Actual', 'Predicted'])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_decomposition(self,
                                      dates: np.ndarray,
                                      total_uncertainty: np.ndarray,
                                      aleatoric_uncertainty: np.ndarray,
                                      epistemic_uncertainty: np.ndarray,
                                      title: str = 'Uncertainty Decomposition',
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot uncertainty decomposition
        
        Args:
            dates: Array of dates
            total_uncertainty: Array of total uncertainties
            aleatoric_uncertainty: Array of aleatoric uncertainties
            epistemic_uncertainty: Array of epistemic uncertainties
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot uncertainties
        ax.plot(dates, total_uncertainty, label='Total Uncertainty', color='#1f77b4', linewidth=2)
        ax.plot(dates, aleatoric_uncertainty, label='Aleatoric Uncertainty', color='#ff7f0e', linewidth=2)
        ax.plot(dates, epistemic_uncertainty, label='Epistemic Uncertainty', color='#2ca02c', linewidth=2)
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str] = ['Sell', 'Hold', 'Buy'],
                             title: str = 'Confusion Matrix',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: Array of true labels
            y_pred: Array of predicted labels
            class_names: List of class names
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Normalized Frequency', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                              ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black")
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(self,
                              confidence_levels: np.ndarray,
                              empirical_coverages: np.ndarray,
                              title: str = 'Calibration Curve',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve
        
        Args:
            confidence_levels: Array of confidence levels
            empirical_coverages: Array of empirical coverages
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot calibration curve
        ax.plot(confidence_levels, empirical_coverages, label='Calibration Curve', color='#1f77b4', linewidth=2, marker='o')
        
        # Plot ideal calibration
        ax.plot([0, 1], [0, 1], label='Ideal Calibration', color='#ff7f0e', linewidth=2, linestyle='--')
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Confidence Level', fontsize=12)
        ax.set_ylabel('Empirical Coverage', fontsize=12)
        
        # Set limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_backtest_results(self,
                             dates: np.ndarray,
                             portfolio_values: np.ndarray,
                             benchmark_values: Optional[np.ndarray] = None,
                             signals: Optional[np.ndarray] = None,
                             title: str = 'Backtest Results',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot backtest results
        
        Args:
            dates: Array of dates
            portfolio_values: Array of portfolio values
            benchmark_values: Array of benchmark values
            signals: Array of trading signals
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Normalize values
        portfolio_values = portfolio_values / portfolio_values[0] * 100
        
        # Plot portfolio values
        ax.plot(dates, portfolio_values, label='Portfolio', color='#1f77b4', linewidth=2)
        
        # Plot benchmark values if provided
        if benchmark_values is not None:
            benchmark_values = benchmark_values / benchmark_values[0] * 100
            ax.plot(dates, benchmark_values, label='Benchmark', color='#ff7f0e', linewidth=2, linestyle='--')
        
        # Plot signals if provided
        if signals is not None:
            # Buy signals
            buy_mask = signals == 2
            if np.any(buy_mask):
                ax.scatter(
                    dates[buy_mask],
                    portfolio_values[buy_mask],
                    color='#2ca02c',
                    marker='^',
                    s=100,
                    label='Buy Signal'
                )
            
            # Sell signals
            sell_mask = signals == 0
            if np.any(sell_mask):
                ax.scatter(
                    dates[sell_mask],
                    portfolio_values[sell_mask],
                    color='#d62728',
                    marker='v',
                    s=100,
                    label='Sell Signal'
                )
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value (normalized to 100)', fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_backtest(self,
                                 dates: np.ndarray,
                                 portfolio_values: np.ndarray,
                                 benchmark_values: Optional[np.ndarray] = None,
                                 signals: Optional[np.ndarray] = None,
                                 title: str = 'Interactive Backtest Results'):
        """
        Plot interactive backtest results using Plotly
        
        Args:
            dates: Array of dates
            portfolio_values: Array of portfolio values
            benchmark_values: Array of benchmark values
            signals: Array of trading signals
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Create figure
        fig = go.Figure()
        
        # Normalize values
        portfolio_values = portfolio_values / portfolio_values[0] * 100
        
        # Add portfolio values
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=2)
            )
        )
        
        # Add benchmark values if provided
        if benchmark_values is not None:
            benchmark_values = benchmark_values / benchmark_values[0] * 100
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=benchmark_values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                )
            )
        
        # Add signals if provided
        if signals is not None:
            # Buy signals
            buy_mask = signals == 2
            if np.any(buy_mask):
                fig.add_trace(
                    go.Scatter(
                        x=dates[buy_mask],
                        y=portfolio_values[buy_mask],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='#2ca02c', size=10, symbol='triangle-up')
                    )
                )
            
            # Sell signals
            sell_mask = signals == 0
            if np.any(sell_mask):
                fig.add_trace(
                    go.Scatter(
                        x=dates[sell_mask],
                        y=portfolio_values[sell_mask],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='#d62728', size=10, symbol='triangle-down')
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value (normalized to 100)',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='x unified'
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importances: np.ndarray,
                               title: str = 'Feature Importance',
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importances: Array of feature importances
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        indices = np.argsort(importances)
        feature_names = np.array(feature_names)[indices]
        importances = importances[indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot feature importance
        ax.barh(range(len(importances)), importances, align='center')
        
        # Set ticks and labels
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(feature_names)
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Importance', fontsize=12)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_weights(self,
                              dates: np.ndarray,
                              feature_names: List[str],
                              attention_weights: np.ndarray,
                              title: str = 'Attention Weights',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot attention weights
        
        Args:
            dates: Array of dates
            feature_names: List of feature names
            attention_weights: Array of attention weights (shape: [n_dates, n_features])
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot attention weights
        im = ax.imshow(attention_weights.T, aspect='auto', cmap=self.attention_cmap)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attention Weight', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(0, len(dates), len(dates) // 10))
        ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates[::len(dates) // 10]])
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_yticklabels(feature_names)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Set title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
