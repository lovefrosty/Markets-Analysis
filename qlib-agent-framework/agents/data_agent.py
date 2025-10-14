"""Codex data preparation agent for QLib workflows."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import qlib
import yaml
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.utils import init_instance_by_config

from .base_agent import BaseAgent

_QLIB_LOCK = threading.Lock()
_QLIB_INITIALIZED = False


class DataAgent(BaseAgent):
    def __init__(self, agent_config: Dict[str, Any], results_dir: Path):
        super().__init__(agent_config, results_dir)
        self.data_ready = False
        self.dataset: Optional[DatasetH] = None
        self.data_handler: Optional[DataHandler] = None
        self.dataset_segments: Dict[str, Any] = {}

    async def initialize(self) -> None:
        await super().initialize()
        self.logger.info("Data agent ready with capabilities: %s", self.agent_config.get("capabilities", []))

    async def prepare_data(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load QLib data, build the dataset, and summarize available segments."""
        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(None, self._prepare_data_sync, data_config)
        self.data_ready = True
        self.logger.info("Data preparation finished")
        return summary

    # --- Internal helpers -----------------------------------------------------------------

    def _prepare_data_sync(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        qlib_init_cfg = data_config.get("qlib_init", {})
        handler_cfg = data_config.get("handler")
        dataset_cfg = data_config.get("dataset")

        if handler_cfg is None or dataset_cfg is None:
            raise ValueError("Data configuration must include 'handler' and 'dataset' definitions.")

        self._ensure_qlib_initialized(qlib_init_cfg)

        self.logger.info("Initializing data handler via config: %s", handler_cfg.get("class"))
        self.data_handler = init_instance_by_config(handler_cfg, accept_types=DataHandler)
        if isinstance(self.data_handler, DataHandlerLP):
            # Ensure processors are fit before dataset construction.
            self.data_handler.setup_data()

        dataset_kwargs = dict(dataset_cfg.get("kwargs", {}))
        dataset_kwargs["handler"] = self.data_handler
        self.logger.info("Creating dataset %s", dataset_cfg.get("class", "DatasetH"))
        self.dataset = DatasetH(**dataset_kwargs)
        self.dataset_segments = dict(self.dataset.segments)

        summary = self._summarize_dataset(qlib_init_cfg)
        summary_yaml = yaml.safe_dump(summary, sort_keys=False)
        self.log_artifact("data_summary.yaml", summary_yaml)
        return summary

    def _ensure_qlib_initialized(self, qlib_init_cfg: Dict[str, Any]) -> None:
        """Initialize QLib only once per process."""
        global _QLIB_INITIALIZED
        with _QLIB_LOCK:
            if _QLIB_INITIALIZED:
                return

            provider_uri = qlib_init_cfg.get("provider_uri")
            if provider_uri is None:
                raise ValueError("QLib initialization requires 'provider_uri'.")

            init_kwargs = dict(qlib_init_cfg)
            init_kwargs.setdefault("skip_if_reg", True)
            self.logger.info("Initializing QLib with provider: %s", provider_uri)
            qlib.init(**init_kwargs)
            _QLIB_INITIALIZED = True

    def _summarize_dataset(self, qlib_init_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset is None or self.data_handler is None:
            raise RuntimeError("Dataset has not been prepared yet.")

        segment_stats: Dict[str, Dict[str, Any]] = {}
        for segment_name in self.dataset_segments:
            df = self.dataset.prepare(segment_name, col_set="label")
            segment_stats[segment_name] = self._segment_summary(df)

        feature_cols = self.data_handler.get_cols("feature")
        label_cols = self.data_handler.get_cols("label")

        summary = {
            "status": "completed",
            "qlib_provider": qlib_init_cfg.get("provider_uri"),
            "region": qlib_init_cfg.get("region"),
            "feature_count": len(feature_cols),
            "label_columns": list(label_cols),
            "segments": segment_stats,
        }
        return summary

    @staticmethod
    def _segment_summary(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"rows": 0}

        datetimes = df.index.get_level_values("datetime")
        instruments = df.index.get_level_values("instrument")
        return {
            "rows": int(df.shape[0]),
            "instruments": int(instruments.unique().size),
            "start": str(datetimes.min()),
            "end": str(datetimes.max()),
        }
