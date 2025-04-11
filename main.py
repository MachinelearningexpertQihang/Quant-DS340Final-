#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train(args):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 加载配置
    config = load_config(args.config)
    
    # 导入必要的模块
    from data.loader import FinancialDataLoader
    from data.dataset import FinancialDataset
    from models.base import FinancialLSTM, FinancialCNN
    from training.trainer import Trainer
    
    # 加载数据
    data_loader = FinancialDataLoader()
    # 这里需要根据你的实际数据加载逻辑进行调整
    data = data_loader.load_data(
        tickers=config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval']
    )
    
    # 创建数据集
    train_dataset = FinancialDataset(
        data=data,
        window_size=config['model']['seq_length'],
        features=config['data']['feature_columns'],
        targets=config['data']['target_columns'],
        start_date=config['splits']['train']['start_date'],
        end_date=config['splits']['train']['end_date']
    )
    
    val_dataset = FinancialDataset(
        data=data,
        window_size=config['model']['seq_length'],
        features=config['data']['feature_columns'],
        targets=config['data']['target_columns'],
        start_date=config['splits']['val']['start_date'],
        end_date=config['splits']['val']['end_date']
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # 创建模型
    input_dim = len(config['data']['feature_columns'])
    model_type = config['model'].get('type', 'lstm')
    
    if model_type.lower() == 'lstm':
        model = FinancialLSTM(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    else:
        model = FinancialCNN(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim']
        )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 训练模型
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到 {model_path}")

def predict(args):
    """使用模型进行预测"""
    logger.info("开始进行预测...")
    
    # 加载配置
    config = load_config(args.config)
    
    # 导入必要的模块
    from data.loader import FinancialDataLoader
    from data.dataset import FinancialDataset
    from models.base import FinancialLSTM, FinancialCNN
    from visualization.plotter import FinancialPlotter
    
    # 加载数据
    data_loader = FinancialDataLoader()
    data = data_loader.load_data(
        tickers=[args.ticker],
        start_date=config['splits']['test']['start_date'],
        end_date=config['splits']['test']['end_date'],
        interval=config['data']['interval']
    )
    
    # 创建数据集
    test_dataset = FinancialDataset(
        data=data,
        window_size=config['model']['seq_length'],
        features=config['data']['feature_columns'],
        targets=config['data']['target_columns']
    )
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # 加载模型
    input_dim = len(config['data']['feature_columns'])
    model_type = config['model'].get('type', 'lstm')
    
    if model_type.lower() == 'lstm':
        model = FinancialLSTM(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    else:
        model = FinancialCNN(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim']
        )
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # 进行预测
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['features']
            target = batch['targets']
            
            output = model(inputs)
            
            predictions.append(output['price'].numpy())
            targets.append(target.numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # 保存预测结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{args.ticker}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    results_df = pd.DataFrame({
        'Predicted': predictions.flatten(),
        'Actual': targets.flatten()
    })
    
    results_df.to_csv(results_path, index=False)
    logger.info(f"预测结果已保存到 {results_path}")
    
    # 可视化预测结果
    plotter = FinancialPlotter()
    plot_path = os.path.join(args.output_dir, f"{args.ticker}_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plotter.plot_predictions(predictions, targets, title=f"{args.ticker} Price Predictions", save_path=plot_path)
    logger.info(f"预测图表已保存到 {plot_path}")

def uncertainty(args):
    """使用不确定性估计进行预测"""
    logger.info("开始进行不确定性估计预测...")
    
    # 加载配置
    config = load_config(args.config)
    
    # 导入必要的模块
    from data.loader import FinancialDataLoader
    from data.dataset import FinancialDataset
    from models.base import FinancialLSTM, FinancialCNN
    from uncertainty.uncertainty import UncertaintyEstimator
    from visualization.plotter import FinancialPlotter
    
    # 加载数据
    data_loader = FinancialDataLoader()
    data = data_loader.load_data(
        tickers=[args.ticker],
        start_date=config['splits']['test']['start_date'],
        end_date=config['splits']['test']['end_date'],
        interval=config['data']['interval']
    )
    
    # 创建数据集
    test_dataset = FinancialDataset(
        data=data,
        window_size=config['model']['seq_length'],
        features=config['data']['feature_columns'],
        targets=config['data']['target_columns']
    )
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # 加载模型
    input_dim = len(config['data']['feature_columns'])
    model_type = config['model'].get('type', 'lstm')
    
    if model_type.lower() == 'lstm':
        model = FinancialLSTM(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
    else:
        model = FinancialCNN(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim']
        )
    
    model.load_state_dict(torch.load(args.model_path))
    
    # 创建不确定性估计器
    uncertainty_estimator = UncertaintyEstimator(
        model=model,
        method=args.method,
        num_samples=args.num_samples
    )
    
    # 进行预测
    predictions, uncertainties = uncertainty_estimator.predict(test_loader)
    
    # 获取实际值
    targets = []
    for batch in test_loader:
        targets.append(batch['targets'].numpy())
    
    targets = np.concatenate(targets)
    
    # 保存预测结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{args.ticker}_uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    results_df = pd.DataFrame({
        'Predicted': predictions.flatten(),
        'Lower': (predictions - uncertainties).flatten(),
        'Upper': (predictions + uncertainties).flatten(),
        'Uncertainty': uncertainties.flatten(),
        'Actual': targets.flatten()
    })
    
    results_df.to_csv(results_path, index=False)
    logger.info(f"预测结果已保存到 {results_path}")
    
    # 可视化预测结果
    plotter = FinancialPlotter()
    plot_path = os.path.join(args.output_dir, f"{args.ticker}_uncertainty_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plotter.plot_uncertainty(predictions, uncertainties, targets, title=f"{args.ticker} Price Predictions with Uncertainty", save_path=plot_path)
    logger.info(f"预测图表已保存到 {plot_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="金融预测模型")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", type=str, default="config/settings.yaml", help="配置文件路径")
    train_parser.add_argument("--output_dir", type=str, default="saved_models", help="模型保存目录")
    
    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="使用模型进行预测")
    predict_parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    predict_parser.add_argument("--config", type=str, default="config/settings.yaml", help="配置文件路径")
    predict_parser.add_argument("--ticker", type=str, required=True, help="股票代码")
    predict_parser.add_argument("--output_dir", type=str, default="results", help="结果保存目录")
    
    # 不确定性估计命令
    uncertainty_parser = subparsers.add_parser("uncertainty", help="使用不确定性估计进行预测")
    uncertainty_parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    uncertainty_parser.add_argument("--config", type=str, default="config/settings.yaml", help="配置文件路径")
    uncertainty_parser.add_argument("--ticker", type=str, required=True, help="股票代码")
    uncertainty_parser.add_argument("--method", type=str, default="mc_dropout", choices=["mc_dropout", "ensemble", "quantile"], help="不确定性估计方法")
    uncertainty_parser.add_argument("--num_samples", type=int, default=100, help="采样次数")
    uncertainty_parser.add_argument("--output_dir", type=str, default="results", help="结果保存目录")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "uncertainty":
        uncertainty(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()