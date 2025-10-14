# Nested Decision Execution & Formulaic Alpha Guide

This guide walks through extending the QLib agent workflow to support high-frequency, multi-level execution while integrating custom (formulaic) alpha factors.

## 1. Initialize QLib with Multi-Frequency Data

```python
import qlib
from qlib.constant import REG_CN

qlib.init(provider_uri={
    "day": "~/.qlib/qlib_data/cn_data",
    "1min": "~/.qlib/qlib_data/cn_data/1min"
}, region=REG_CN)
```

If the minute-level data is missing, fetch it via `qlib.tests.data.GetData()` (see the `NestedDecisionExecutionWorkflow._init_qlib` helper for a scripted example).

## 2. Define Formulaic Alpha Factors

Use QLib’s expression language to craft custom factors. The example below defines a MACD-style signal and pairs it with a forward-return label.

```python
from qlib.data.dataset.loader import QlibDataLoader

MACD_EXP = (
    "(EMA($close, 12) - EMA($close, 26))/$close"
    "- EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close"
)
fields = [MACD_EXP]
names = ["MACD"]
labels = ["Ref($close, -2)/Ref($close, -1) - 1"]
label_names = ["LABEL"]

loader = QlibDataLoader(config={
    "feature": (fields, names),
    "label": (labels, label_names),
})
df = loader.load(
    instruments="csi300",
    start_time="2010-01-01",
    end_time="2017-12-31",
)
print(df.head())
```

## 3. Plug Formulaic Alphas into Nested Decision Agents

Each Trading Agent layer should:

1. **Extract information** – consume the features produced by your loader.
2. **Forecast** – call a model (LightGBM, custom model, etc.) trained on those features.
3. **Generate decisions** – convert forecasts into orders or signals.

Nested workflows can chain daily → intraday → execution layers. A simplified agent skeleton:

```python
class MyTradingAgent:
    def __init__(self, data, forecast_model, decision_generator):
        self.data = data
        self.forecast_model = forecast_model
        self.decision_generator = decision_generator

    def run(self):
        features = self.prepare_features(self.data)
        pred = self.forecast_model.predict(features)
        decisions = self.decision_generator.generate(pred)
        return decisions
```

## 4. Use QLib’s Nested Executors

Refer to the `NestedDecisionExecutionWorkflow` class (included below) for a ready-to-run example that:

- Initializes day + minute data sets.
- Trains a baseline LightGBM model.
- Runs a three-level NestedExecutor stack (day → 30 min → 5 min).
- Logs results using `SignalRecord` and `PortAnaRecord`.

```python
from copy import deepcopy
import qlib
import fire
import pandas as pd
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.backtest import collect_data


class NestedDecisionExecutionWorkflow:
    market = "csi300"
    benchmark = "SH000300"
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2021-05-31",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2007-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2020-01-01", "2021-05-31"),
                },
            },
        },
    }

    exp_name = "nested"

    port_analysis_config = {
        "executor": {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "inner_executor": {
                    "class": "NestedExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "30min",
                        "inner_executor": {
                            "class": "SimulatorExecutor",
                            "module_path": "qlib.backtest.executor",
                            "kwargs": {
                                "time_per_step": "5min",
                                "generate_portfolio_metrics": True,
                                "verbose": True,
                                "indicator_config": {"show_indicator": True},
                            },
                        },
                        "inner_strategy": {
                            "class": "TWAPStrategy",
                            "module_path": "qlib.contrib.strategy.rule_strategy",
                        },
                        "generate_portfolio_metrics": True,
                        "indicator_config": {"show_indicator": True},
                    },
                },
                "inner_strategy": {
                    "class": "SBBStrategyEMA",
                    "module_path": "qlib.contrib.strategy.rule_strategy",
                    "kwargs": {"instruments": market, "freq": "1min"},
                },
                "track_data": True,
                "generate_portfolio_metrics": True,
                "indicator_config": {"show_indicator": True},
            },
        },
        "backtest": {
            "start_time": "2020-09-20",
            "end_time": "2021-05-20",
            "account": 100000000,
            "exchange_kwargs": {
                "freq": "1min",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    def _init_qlib(self):
        provider_uri_day = "~/.qlib/qlib_data/cn_data"
        GetData().qlib_data(target_dir=provider_uri_day, region=REG_CN, version="v2", exists_skip=True)
        provider_uri_1min = HIGH_FREQ_CONFIG.get("provider_uri")
        GetData().qlib_data(target_dir=provider_uri_1min, interval="1min", region=REG_CN, version="v2", exists_skip=True)
        provider_uri_map = {"1min": provider_uri_1min, "day": provider_uri_day}
        qlib.init(provider_uri=provider_uri_map, dataset_cache=None, expression_cache=None)

    def _train_model(self, model, dataset):
        with R.start(experiment_name=self.exp_name):
            R.log_params(**flatten_dict(self.task))
            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})
            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

    def backtest(self):
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"signal": (model, dataset), "topk": 50, "n_drop": 5},
        }
        self.port_analysis_config["strategy"] = strategy_config
        self.port_analysis_config["backtest"]["benchmark"] = self.benchmark

        with R.start(experiment_name=self.exp_name, resume=True):
            recorder = R.get_recorder()
            par = PortAnaRecord(
                recorder,
                self.port_analysis_config,
                indicator_analysis_method="value_weighted",
            )
            par.generate()

    def collect_data(self):
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        executor_config = self.port_analysis_config["executor"]
        backtest_config = self.port_analysis_config["backtest"]
        backtest_config["benchmark"] = self.benchmark
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"signal": (model, dataset), "topk": 50, "n_drop": 5},
        }
        data_generator = collect_data(executor=executor_config, strategy=strategy_config, **backtest_config)
        for trade_decision in data_generator:
            print(trade_decision)

    def backtest_only_daily(self):
        self._init_qlib()
        model = init_instance_by_config(self.task["model"])
        dataset = init_instance_by_config(self.task["dataset"])
        self._train_model(model, dataset)
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"signal": (model, dataset), "topk": 50, "n_drop": 5},
        }
        pa_conf = deepcopy(self.port_analysis_config)
        pa_conf["strategy"] = strategy_config
        pa_conf["executor"] = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True, "verbose": True},
        }
        pa_conf["backtest"]["benchmark"] = self.benchmark

        with R.start(experiment_name=self.exp_name, resume=True):
            recorder = R.get_recorder()
            par = PortAnaRecord(recorder, pa_conf)
            par.generate()


if __name__ == "__main__":
    fire.Fire(NestedDecisionExecutionWorkflow)
```

import qlib
import fire
import pandas as pd
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG
from qlib.tests.data import GetData
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.backtest import collect_data

# ... full NestedDecisionExecutionWorkflow definition ...

if __name__ == "__main__":
    fire.Fire(NestedDecisionExecutionWorkflow)
```

> **Tip:** The complete class (with backtest, data collection, and validation helpers) lives in this document for reference. Embed it in `scripts/` if you wish to run the high-frequency workflow locally.

## 5. Risk & Indicator Analysis Outputs

When running nested backtests, QLib produces multiple frequency reports. Example outputs:

```
'The following are analysis results of benchmark return(1day).'
                       risk
mean               0.000651
std                0.012472
annualized_return  0.154967
information_ratio  0.805422
max_drawdown      -0.160445

'The following are analysis results of the excess return with cost(30minute).'
                       risk
mean               0.000155
std                0.003343
annualized_return  0.294536
information_ratio  2.018860
max_drawdown      -0.075579

'The following are analysis results of indicators(5minute).'
        value
ffr  0.991017
pa   0.000000
pos  0.000000
```

These logs are stored as artifacts (`port_analysis_*.pkl`, `indicator_analysis_*.pkl`) and can be visualized using `qlib.contrib.report.analysis_position.*` helpers.

## 6. Extend to Reinforcement Learning & Advanced Execution

- Use `qlib.backtest.collect_data` for replaying nested decisions.
- Integrate with [QLibRL](https://qlib.readthedocs.io/en/latest/component/qlib_rl.html) to optimize policies across levels.
- Replace baseline strategies with rule-based (`TWAPStrategy`, `SBBStrategyEMA`) or learned policies.

## 7. Summary Checklist

- [ ] Download day + minute market data (`GetData().qlib_data(...)`).
- [ ] Define formulaic alphas using QLib expressions (stored in `configs/features/`).
- [ ] Wire alphas into dataset handlers and models.
- [ ] Configure nested executors/strategies for each trading layer.
- [ ] Log experiments with `R.start` / `PortAnaRecord` for reproducibility.
- [ ] Analyze multi-frequency risk reports & iterate.

For additional automation guidance, see the main [README](../README.md) or request tailored scripts.
