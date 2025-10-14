"""Codex model training agent for QLib workflows."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from qlib.contrib.evaluate import risk_analysis
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config

from .base_agent import BaseAgent


class ModelAgent(BaseAgent):
    def __init__(self, agent_config: Dict[str, Any], results_dir: Path):
        super().__init__(agent_config, results_dir)
        self.trained_models: Dict[str, Dict[str, Any]] = {}
        self.model_objects: Dict[str, Any] = {}
        self.predictions: Dict[str, pd.DataFrame] = {}

    async def initialize(self) -> None:
        await super().initialize()
        self.logger.info("Model agent supporting capabilities: %s", self.agent_config.get("capabilities", []))

    async def train_model(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        dataset: DatasetH,
    ) -> Dict[str, Any]:
        if not model_config.get("enabled", True):
            self.logger.info("Model %s disabled via configuration; skipping.", model_name)
            result = {"model": model_name, "status": "skipped", "reason": "disabled"}
            self.trained_models[model_name] = result
            return result
        if dataset is None:
            raise ValueError("Dataset must be prepared before training models.")

        self.logger.info("Training model %s with params: %s", model_name, model_config.get("kwargs", {}))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._train_model_sync,
            model_name,
            model_config,
            dataset,
        )

    # --- Internal helpers -----------------------------------------------------------------

    def _train_model_sync(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        dataset: DatasetH,
    ) -> Dict[str, Any]:
        try:
            model = init_instance_by_config(model_config)
        except (ModuleNotFoundError, ImportError) as err:
            self.logger.warning("Skipping model %s due to missing dependency: %s", model_name, err)
            result = {"model": model_name, "status": "skipped", "reason": str(err)}
            self.trained_models[model_name] = result
            return result
        model.fit(dataset=dataset)

        raw_pred = model.predict(dataset=dataset)
        if isinstance(raw_pred, pd.Series):
            pred_df = raw_pred.to_frame("score")
        elif isinstance(raw_pred, pd.DataFrame):
            if "score" not in raw_pred.columns:
                pred_df = raw_pred.rename(columns={raw_pred.columns[0]: "score"})
            else:
                pred_df = raw_pred.rename(columns={"score": "score"})
        else:
            raise TypeError(f"Unsupported prediction type: {type(raw_pred)}")

        pred_path = self.results_dir / f"{self.name}_{model_name}_predictions.parquet"
        pred_df.to_parquet(pred_path)

        segment_metrics = self._compute_segment_metrics(dataset, pred_df)
        evaluation = self._aggregate_risk_metrics(pred_df, dataset)

        result = {
            "model": model_name,
            "metrics": segment_metrics,
            "evaluation": evaluation,
            "artifacts": {"predictions": pred_path.name},
        }

        self.trained_models[model_name] = result
        self.model_objects[model_name] = model
        self.predictions[model_name] = pred_df
        self.logger.info("Model %s trained. Validation IC: %s", model_name, segment_metrics.get("valid", {}).get("ic"))
        return result

    def _compute_segment_metrics(self, dataset: DatasetH, pred_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        segments = getattr(dataset, "segments", {})
        for segment_name in segments:
            label_df = dataset.prepare(segment_name, col_set="label")
            if label_df is None or label_df.empty:
                continue

            label_series = label_df.iloc[:, 0]
            common_index = label_series.index.intersection(pred_df.index)
            if common_index.empty:
                continue

            aligned_label = label_series.loc[common_index]
            aligned_pred = pred_df.loc[common_index, "score"]

            diff = aligned_pred - aligned_label
            mse = float(np.mean(np.square(diff)))
            mae = float(np.mean(np.abs(diff)))
            pearson = self._safe_corr(aligned_pred, aligned_label, method="pearson")
            spearman = self._safe_corr(aligned_pred, aligned_label, method="spearman")

            metrics[segment_name] = {
                "samples": int(common_index.size),
                "mse": mse,
                "mae": mae,
                "ic": spearman,
                "pearson": pearson,
                "label_mean": float(aligned_label.mean()),
                "prediction_mean": float(aligned_pred.mean()),
            }
        return metrics

    def _aggregate_risk_metrics(self, pred_df: pd.DataFrame, dataset: DatasetH) -> Dict[str, Any]:
        """Compute simple risk analysis using predictions scaled by realized returns."""
        try:
            label_df = dataset.prepare("test", col_set="label")
        except Exception:
            label_df = None

        if label_df is None or label_df.empty:
            return {}

        label_series = label_df.iloc[:, 0]
        common_index = label_series.index.intersection(pred_df.index)
        if common_index.empty:
            return {}

        pnl_series = label_series.loc[common_index] * pred_df.loc[common_index, "score"]
        analysis = risk_analysis(pnl_series, freq="day")
        if isinstance(analysis, pd.Series):
            return {k: float(v) for k, v in analysis.items()}
        if isinstance(analysis, pd.DataFrame):
            column = analysis.columns[0]
            return {k: float(v) for k, v in analysis[column].to_dict().items()}
        return {}

    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
        if a.size == 0 or b.size == 0:
            return math.nan
        value = a.corr(b, method=method)
        return float(value) if value is not None else math.nan
