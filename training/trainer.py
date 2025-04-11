import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

class Trainer:
    """
    Unified trainer for financial forecasting models
    Supports both base and enhanced models
    """
    def __init__(self, 
                model, 
                train_loader, 
                val_loader, 
                test_loader=None,
                optimizer=None,
                scheduler=None,
                loss_fn=None,
                device='cuda',
                config=None,
                enhanced=False):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            optimizer: Optimizer (optional)
            scheduler: Learning rate scheduler (optional)
            loss_fn: Loss function (optional)
            device: Device to use for training
            config: Training configuration
            enhanced: Whether to use enhanced training features
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.enhanced = enhanced
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Default configuration
        default_config = {
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'grad_clip_value': 0.0,
            'log_interval': 10,
            'save_dir': 'saved_models',
            'model_name': 'financial_model',
            'save_best_only': True,
            # Enhanced options
            'use_mixed_precision': enhanced,
            'use_ema': enhanced,
            'ema_decay': 0.999,
            'use_swa': enhanced,
            'swa_start': 50,
            'use_lookahead': enhanced,
            'lookahead_k': 5,
            'lookahead_alpha': 0.5,
            'use_online_learning': enhanced,
            'online_learning_interval': 10
        }
        
        # Update with provided config
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Setup optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler if not provided
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = scheduler
        
        # Setup loss function if not provided
        if loss_fn is None:
            if enhanced:
                from models.enhanced import EnhancedMultiTaskLoss
                self.loss_fn = EnhancedMultiTaskLoss()
            else:
                from models.base import MultiTaskLoss
                self.loss_fn = MultiTaskLoss()
        else:
            self.loss_fn = loss_fn
        
        # Create save directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = defaultdict(list)
        
        # Setup enhanced training features
        if enhanced:
            self._setup_enhanced_features()
    
    def _setup_enhanced_features(self):
        """Setup enhanced training features"""
        # Exponential moving average of model parameters
        if self.config['use_ema']:
            self.ema_model = self._create_ema_model()
        
        # Stochastic weight averaging
        if self.config['use_swa']:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer, 
                anneal_strategy="cos", 
                anneal_epochs=5, 
                swa_lr=1e-3
            )
        
        # Lookahead optimizer wrapper
        if self.config['use_lookahead']:
            self.optimizer = self._create_lookahead_optimizer(
                self.optimizer,
                k=self.config['lookahead_k'],
                alpha=self.config['lookahead_alpha']
            )
        
        # Mixed precision training
        if self.config['use_mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _create_ema_model(self):
        """Create exponential moving average model"""
        ema_model = type(self.model)(
            **{k: v for k, v in self.model.__dict__.items() 
               if not k.startswith('_') and not callable(v)}
        ).to(self.device)
        
        # Copy parameters
        for param_ema, param_model in zip(ema_model.parameters(), self.model.parameters()):
            param_ema.data.copy_(param_model.data)
            param_ema.requires_grad_(False)
        
        return ema_model
    
    def _update_ema_model(self):
        """Update exponential moving average model"""
        decay = self.config['ema_decay']
        for param_ema, param_model in zip(self.ema_model.parameters(), self.model.parameters()):
            param_ema.data.mul_(decay).add_(param_model.data, alpha=1 - decay)
    
    def _create_lookahead_optimizer(self, optimizer, k=5, alpha=0.5):
        """Create lookahead optimizer wrapper"""
        class Lookahead:
            def __init__(self, optimizer, k=5, alpha=0.5):
                self.optimizer = optimizer
                self.k = k
                self.alpha = alpha
                self.step_counter = 0
                self.slow_weights = [[p.clone().detach() for p in group['params']]
                                    for group in optimizer.param_groups]
            
            def step(self, closure=None):
                loss = self.optimizer.step(closure)
                self.step_counter += 1
                
                if self.step_counter % self.k == 0:
                    # Update slow weights
                    for group, slow_group in zip(self.optimizer.param_groups, self.slow_weights):
                        for p, slow_p in zip(group['params'], slow_group):
                            if p.grad is not None:
                                slow_p.data.add_(self.alpha * (p.data - slow_p.data))
                                p.data.copy_(slow_p.data)
                
                return loss
            
            def zero_grad(self):
                self.optimizer.zero_grad()
            
            def state_dict(self):
                state = {
                    'optimizer': self.optimizer.state_dict(),
                    'slow_weights': self.slow_weights,
                    'k': self.k,
                    'alpha': self.alpha,
                    'step_counter': self.step_counter
                }
                return state
            
            def load_state_dict(self, state_dict):
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.slow_weights = state_dict['slow_weights']
                self.k = state_dict['k']
                self.alpha = state_dict['alpha']
                self.step_counter = state_dict['step_counter']
            
            @property
            def param_groups(self):
                return self.optimizer.param_groups
        
        return Lookahead(optimizer, k, alpha)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = defaultdict(float)
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            # Move data to device
            x = x.to(self.device)
            y_price = y['price'].to(self.device)
            y_volatility = y['volatility'].to(self.device)
            y_signal = y['signal'].to(self.device)
            
            # Create targets dictionary
            targets = {
                'price': y_price,
                'volatility': y_volatility,
                'signal': y_signal
            }
            
            # Forward pass with mixed precision if enabled
            self.optimizer.zero_grad()
            
            if self.enhanced and self.config['use_mixed_precision']:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x)
                    loss, loss_dict = self.loss_fn(outputs, targets, self.model if self.enhanced else None)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['grad_clip_value'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_value']
                    )
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward pass
                outputs = self.model(x)
                loss, loss_dict = self.loss_fn(outputs, targets, self.model if self.enhanced else None)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['grad_clip_value'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_value']
                    )
                
                # Update weights
                self.optimizer.step()
            
            # Update EMA model if enabled
            if self.enhanced and self.config['use_ema'] and batch_idx % 10 == 0:
                self._update_ema_model()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Track metrics
            for name, value in loss_dict.items():
                epoch_metrics[name] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Update SWA model if enabled
        if self.enhanced and self.config['use_swa'] and epoch >= self.config['swa_start']:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        elif isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # Don't step scheduler here if it's ReduceLROnPlateau
            pass
        else:
            self.scheduler.step()
        
        # Calculate average loss and metrics
        avg_loss = epoch_loss / len(self.train_loader)
        avg_metrics = {name: value / len(self.train_loader) for name, value in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, loader, model=None):
        """Validate model on loader"""
        # Use provided model or default to self.model
        if model is None:
            model = self.model
        
        model.eval()
        val_loss = 0
        val_metrics = defaultdict(float)
        
        with torch.no_grad():
            for x, y in loader:
                # Move data to device
                x = x.to(self.device)
                y_price = y['price'].to(self.device)
                y_volatility = y['volatility'].to(self.device)
                y_signal = y['signal'].to(self.device)
                
                # Create targets dictionary
                targets = {
                    'price': y_price,
                    'volatility': y_volatility,
                    'signal': y_signal
                }
                
                # Forward pass
                outputs = model(x)
                loss, loss_dict = self.loss_fn(outputs, targets)
                
                # Track loss and metrics
                val_loss += loss.item()
                for name, value in loss_dict.items():
                    val_metrics[name] += value
        
        # Calculate average loss and metrics
        avg_loss = val_loss / len(loader)
        avg_metrics = {name: value / len(loader) for name, value in val_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self):
        """Train model for multiple epochs"""
        self.logger.info(f"Starting training for {self.config['num_epochs']} epochs")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate with base model
            val_loss, val_metrics = self.validate(self.val_loader)
            
            # Validate with EMA model if enabled
            if self.enhanced and self.config['use_ema']:
                ema_val_loss, ema_val_metrics = self.validate(self.val_loader, self.ema_model)
                self.logger.info(f"EMA Val Loss: {ema_val_loss:.4f}")
                
                # Use EMA validation loss for early stopping if it's better
                if ema_val_loss < val_loss:
                    val_loss = ema_val_loss
                    val_metrics = ema_val_metrics
                    self.logger.info("Using EMA model for validation")
            
            # Validate with SWA model if enabled
            if self.enhanced and self.config['use_swa'] and epoch >= self.config['swa_start']:
                # Update batch norm statistics
                torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model)
                
                swa_val_loss, swa_val_metrics = self.validate(self.val_loader, self.swa_model)
                self.logger.info(f"SWA Val Loss: {swa_val_loss:.4f}")
                
                # Use SWA validation loss for early stopping if it's better
                if swa_val_loss < val_loss:
                    val_loss = swa_val_loss
                    val_metrics = swa_val_metrics
                    self.logger.info("Using SWA model for validation")
            
            # Update learning rate for ReduceLROnPlateau
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            
            # Log results
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            for name, value in train_metrics.items():
                self.history[f'train_{name}'].append(value)
            for name, value in val_metrics.items():
                self.history[f'val_{name}'].append(value)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                if self.config['save_best_only']:
                    self._save_model(f"{self.config['model_name']}_best.pth")
                    self.logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0 and not self.config['save_best_only']:
                self._save_model(f"{self.config['model_name']}_epoch{epoch+1}.pth")
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Online learning if enabled
            if self.enhanced and self.config['use_online_learning'] and epoch > 0 and epoch % self.config['online_learning_interval'] == 0:
                self._perform_online_learning()
        
        # Final evaluation on test set if provided
        if self.test_loader is not None:
            self.logger.info("Evaluating on test set")
            
            # Test with best model
            best_model = self._load_best_model()
            test_loss, test_metrics = self.validate(self.test_loader, best_model)
            self.logger.info(f"Best Model Test Loss: {test_loss:.4f}")
            
            # Add test metrics to history
            self.history['test_loss'] = test_loss
            for name, value in test_metrics.items():
                self.history[f'test_{name}'] = value
            
            # Test with EMA model if enabled
            if self.enhanced and self.config['use_ema']:
                ema_test_loss, ema_test_metrics = self.validate(self.test_loader, self.ema_model)
                self.logger.info(f"EMA Model Test Loss: {ema_test_loss:.4f}")
                
                # Add EMA test metrics to history
                self.history['ema_test_loss'] = ema_test_loss
                for name, value in ema_test_metrics.items():
                    self.history[f'ema_test_{name}'] = value
            
            # Test with SWA model if enabled
            if self.enhanced and self.config['use_swa'] and epoch >= self.config['swa_start']:
                swa_test_loss, swa_test_metrics = self.validate(self.test_loader, self.swa_model)
                self.logger.info(f"SWA Model Test Loss: {swa_test_loss:.4f}")
                
                # Add SWA test metrics to history
                self.history['swa_test_loss'] = swa_test_loss
                for name, value in swa_test_metrics.items():
                    self.history[f'swa_test_{name}'] = value
        
        # Calculate training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self.history
    
    def _perform_online_learning(self):
        """Perform online learning on recent data"""
        self.logger.info("Performing online learning")
        # Implementation depends on specific online learning strategy
        # This is a placeholder for the actual implementation
    
    def _save_model(self, filename, model=None):
        """Save model checkpoint"""
        if model is None:
            model = self.model
            
        save_path = os.path.join(self.config['save_dir'], filename)
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save EMA model if enabled
        if self.enhanced and self.config['use_ema']:
            checkpoint['ema_model'] = self.ema_model.state_dict()
        
        # Save SWA model if enabled
        if self.enhanced and self.config['use_swa'] and hasattr(self, 'swa_model'):
            checkpoint['swa_model'] = self.swa_model.state_dict()
        
        torch.save(checkpoint, save_path)
    
    def _load_best_model(self):
        """Load best model from checkpoint"""
        load_path = os.path.join(self.config['save_dir'], f"{self.config['model_name']}_best.pth")
        
        if not os.path.exists(load_path):
            self.logger.warning(f"Best model checkpoint not found at {load_path}")
            return self.model
        
        checkpoint = torch.load(load_path)
        
        # Create a new model instance
        from models.enhanced import EnhancedFinancialModel
        from models.base import FinancialForecastingModel
        
        if self.enhanced:
            best_model = EnhancedFinancialModel().to(self.device)
        else:
            best_model = FinancialForecastingModel().to(self.device)
        
        # Load state dict
        best_model.load_state_dict(checkpoint['model'])
        
        return best_model
