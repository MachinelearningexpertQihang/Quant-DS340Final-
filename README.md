# Quant-DS340Final-# Financial Forecasting Model

This repository contains a financial forecasting model with both base and enhanced implementations. The model is designed to predict financial time series data with uncertainty quantification.

## Project Structure

\`\`\`
project/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── settings.yaml
│   └── __init__.py
├── data/
│   ├── loader.py
│   └── dataset.py
├── models/
│   ├── common/
│   │   └── layers.py
│   ├── base.py
│   └── enhanced.py
├── training/
│   └── trainer.py
├── uncertainty/
│   └── uncertainty.py
├── visualization/
│   └── plotter.py
└── tests/
    ├── test_models.py
    └── test_data.py
\`\`\`

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

### Training a model

\`\`\`bash
# Train base model
python -m training.trainer --model base

# Train enhanced model
python -m training.trainer --model enhanced
\`\`\`

### Making predictions with uncertainty

\`\`\`bash
python -m uncertainty.uncertainty --model base
python -m uncertainty.uncertainty --model enhanced
\`\`\`

## Features

- Base financial forecasting model
- Enhanced model with advanced layers and loss functions
- Uncertainty quantification
- Data loading and preprocessing utilities
- Visualization tools
- Comprehensive testing suite
\`\`\`

```text file="requirements.txt"
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.9.0
pytorch-lightning>=1.4.0
pyyaml>=6.0
plotly>=5.3.0
yfinance>=0.1.70
tqdm>=4.62.0
