# QLib Agent Framework

An automated quantitative research orchestration framework built around [Microsoft QLib](https://github.com/microsoft/qlib) and Codex-powered agents for data preparation, model training, strategy implementation, and performance evaluation.

## Core Dependencies

```bash
pip install pyqlib
pip install numpy
pip install --upgrade cython
pip install lightgbm torch
pip install openai  # For Codex integration
```

## Quick Start

```bash
# Clone your repository and change into the project directory
# git clone <your-repo-url>
cd qlib-agent-framework

# Install Python dependencies
pip install -r requirements.txt

# Download the latest QLib community dataset
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1

# Initialize the Codex agents
python agents/orchestrator.py --initialize

# Run the complete workflow
python agents/orchestrator.py --execute-full-pipeline

# Check workflow status
python agents/orchestrator.py --status

# Generate performance reports
python agents/orchestrator.py --generate-report
```

## Expected Outcomes

- Automated download and preprocessing of market data.
- Training and validation of multiple ML models (LightGBM, MLP, LSTM).
- Implementation and backtesting of trading strategies.
- Generation of performance reports and optimization suggestions.
- Continuous experiment tracking, logging, and model versioning.
- End-to-end automation coverage of roughly 85-90% of the quant workflow.

## Automation Workflow Overview

1. **Data Pipeline Management** – automatic data download/validation, feature engineering (Alpha158), preprocessing, and dataset splitting.
2. **Model Development** – baseline LightGBM, PyTorch MLP, LSTM, hyperparameter tuning, and validation.
3. **Strategy Implementation** – TopkDropout strategy, risk management, position sizing, and transaction cost modeling.
4. **Performance Analysis** – backtesting, comprehensive metrics, visualization, and reporting.

## Repository Layout

```
qlib-agent-framework/
├── .github/workflows/qlib-automation.yml
├── agents/
│   ├── data_agent.py
│   ├── model_agent.py
│   ├── strategy_agent.py
│   └── orchestrator.py
├── configs/
│   ├── base_config.yaml
│   ├── models/
│   └── strategies/
├── data/
│   └── .qlib/
├── results/
├── scripts/
│   ├── setup.sh
│   └── install_qlib.py
├── requirements.txt
└── README.md
```

## GitHub Actions Automation

A weekly GitHub Action (`.github/workflows/qlib-automation.yml`) installs dependencies, downloads data, initializes Codex agents, runs QLib experiments, and uploads results.

## License

MIT License (add your specific license text if different).
