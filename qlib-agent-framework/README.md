# QLib Agent Framework

An automated quantitative research orchestration framework built around [Microsoft QLib](https://github.com/microsoft/qlib) and Codex-powered agents for data preparation, model training, strategy implementation, and performance evaluation.

---

## Repository Layout

```
qlib-agent-framework/
├── agents/                  # Codex agent implementations (data/model/strategy/evaluation)
├── configs/                 # Base and advanced workflow configuration
│   ├── base_config.yaml
│   ├── agent_workflow.yaml
│   ├── agent_workflow_advanced.yaml
│   ├── features/sample_features.yaml
│   ├── models/.gitkeep
│   └── strategies/.gitkeep
├── models/                  # Example model configs & stubs
│   ├── custom_model.py
│   ├── eindexing_config.yaml
│   └── lstm_config.yaml
├── scripts/                 # Helper scripts (dataset setup, risk analysis, etc.)
│   ├── install_qlib.py
│   ├── run_risk_analysis.py
│   └── setup.sh
├── requirements.txt         # Python dependencies
├── results/                 # Generated artifacts (ignored from git; created at runtime)
└── README.md
```

---

## Prerequisites

- **Python 3.9+**
- **Operating system:** macOS, Linux, or WSL (Windows Subsystem for Linux)
- **Git** for version control
- **Optional GPU support:** install CUDA & cuDNN if you plan to run PyTorch LSTM models on GPU

### Core Python Dependencies

```bash
pip install -r qlib-agent-framework/requirements.txt
```

Key packages include:
- `pyqlib`, `numpy`, `pandas`, `lightgbm`, `torch`
- `openai` (Codex agent orchestration)
- `pyyaml`, `tqdm`, `cython`

---

## Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-user>/Markets-Analysis.git
   cd Markets-Analysis/qlib-agent-framework
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download QLib community data**
   ```bash
   ./scripts/setup.sh
   ```
   This stores the daily CN market data under `~/.qlib/qlib_data/cn_data`.

---

## Running the Orchestrated Workflow

```bash
# Initialize agents (creates state files, verifies config)
python agents/orchestrator.py --initialize

# Execute the end-to-end pipeline (data → models → strategies → evaluation)
python agents/orchestrator.py --execute-full-pipeline

# Inspect the latest state snapshot
python agents/orchestrator.py --status

# Generate a consolidated text report
python agents/orchestrator.py --generate-report
```

Artifacts are written to `results/`:
- `workflow_results.yaml` – aggregated outputs (data summary, model metrics, strategy reports)
- `data_agent_data_summary.yaml` – dataset coverage
- `model_agent_<model>_predictions.parquet` – model predictions
- `strategy_agent_<strategy>_report_*.csv` – backtest reports
- `evaluation_agent_evaluation_summary.yaml` – strategy comparison results

---

## Configuring the Agents

### Base Orchestration (`configs/base_config.yaml`)

- Defines the agents, their roles, and default QLib dataset/strategy configurations.
- Each agent consumes a portion of the configuration:
  - `data_agent` → QLib init, handler (`Alpha158`), dataset segments
  - `model_agent` → the list of models (LightGBM, PyTorch MLP, LSTM)
  - `strategy_agent` → strategy parameters (TopkDropout, Enhanced Indexing)
  - `evaluation_agent` → how to pick the best strategy (Sharpe/IC logic)

### Declarative Workflows (`configs/agent_workflow*.yaml`)

- `agent_workflow.yaml` – opinionated defaults (torch disabled, enhanced indexing off).
- `agent_workflow_advanced.yaml` – toggles for LSTM (`enabled: true` once torch is available), Enhanced Indexing (`riskmodel_root` path), and artifact directories.

### Feature & Alpha Construction

- `configs/features/sample_features.yaml` demonstrates programmable expressions (e.g. `Ref`, `EMA`, custom formulas).
- Plug these into your handler by extending the `base_config.yaml` or referencing them when creating `DatasetH` instances.

### Custom Model Integration

- `models/custom_model.py` contains a minimal `CustomAlphaModel` stub that adheres to QLib’s `BaseModel` API.
- Register it by adding a model entry to the config:
  ```yaml
  models:
    custom_alpha:
      class: "CustomAlphaModel"
      module_path: "models.custom_model"
      kwargs:
        window: 10
        scale: 1.5
  ```

### Enabling LSTM / Enhanced Indexing

- **LSTM:** ensure `torch` is installed, set `enabled: true` in `agent_workflow_advanced.yaml`, adjust `input_size` in `models/lstm_config.yaml`.
- **Enhanced Indexing:** populate `models/eindexing_config.yaml` & `agent_workflow_advanced.yaml` with a valid `riskmodel_root` (see QLib docs for risk model preparation).

---

## Experiment Management & Logging

The agents already log checkpoints to `results/`, but you can run ad-hoc experiments using QLib’s recorder API:

```python
import qlib
from qlib.workflow import R

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

with R.start(experiment_name="exp_alpha", recorder_name="run_baseline"):
    model.fit(dataset)
    preds = model.predict(dataset)
    R.log_params(model="LGBM", window=5)
    R.log_metrics(train_loss=0.24, ic=0.03, step=1)
    R.save_objects(**{"pred.pkl": preds}, artifact_path="predictions")
```

Useful management commands:
- `R.search_records(['exp_alpha'], order_by=['metrics.ic DESC'])`
- `R.list_experiments()`, `R.list_recorders(experiment_name='exp_alpha')`
- `R.delete_recorder(recorder_id=<id>)` (cleanup)

---

## Risk Analysis & Reporting

- **Automated backtest stats** are logged during the workflow (`strategy_agent_*` files).
- **Ad-hoc risk summary** from a report CSV:
  ```bash
  python scripts/run_risk_analysis.py results/strategy_agent_topk_dropout_report_1day.csv
  ```
- **Graphical reports**:
  ```python
  from qlib.contrib.report.analysis_position import report as position_report
  position_report.report_graph(recorder_id_or_path)
  ```
  See `qlib.contrib.report` for more visualization helpers (IC curves, cumulative returns, risk analysis graphs, etc.).

---

## GitHub Actions CI/CD

Workflow file: `.github/workflows/qlib-automation.yml`

1. Set the `OPENAI_API_KEY` secret in the repository settings (`Settings → Secrets → Actions`).
2. Optionally customize schedule or add push triggers.
3. Trigger manually via the **Actions** tab (“Run workflow” button).
4. Outputs (logs + artifacts) are uploaded for download and review.

The CI pipeline:
- Checks out the repo
- Installs dependencies (`pip install -r requirements.txt`)
- Runs `scripts/setup.sh` to acquire datasets
- Executes the orchestrator workflow
- Uploads the `results/` directory as an artifact

---

## Advanced Guides

- [Nested Decision Execution & Formulaic Alpha Guide](docs/nested_decision_workflow.md)

## Troubleshooting & Tips

- **Dataset not found:** ensure `~/.qlib/qlib_data/cn_data` exists; re-run `scripts/setup.sh`.
- **Torch missing:** install `torch` (CPU or GPU version) before enabling the LSTM configuration.
- **Enhanced Indexing errors:** double-check `riskmodel_root` path and the required risk model files (see QLib docs).
- **Large runs:** adjust `kernels` in `base_config.yaml` based on CPU cores; consider caching or reducing date ranges.
- **Reset state:** delete `results/workflow_state.yaml` if you want to restart agent state tracking.

---

## License

MIT License (add your specific license text if different).
