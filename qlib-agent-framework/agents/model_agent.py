"""Codex model training agent for QLib workflows."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent


class ModelAgent(BaseAgent):
    def __init__(self, agent_config: Dict[str, Any], results_dir: Path):
        super().__init__(agent_config, results_dir)
        self.trained_models: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> None:
        await super().initialize()
        self.logger.info("Model agent supporting capabilities: %s", self.agent_config.get("capabilities", []))

    async def train_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Training model %s with params: %s", model_name, model_config.get("kwargs", {}))
        await asyncio.sleep(0)

        metrics = self._simulate_training_metrics(model_name)
        artifact = f"model={model_name}, metrics={metrics}"
        self.log_artifact(f"{model_name}_metrics.txt", artifact)

        result = {"model": model_name, "metrics": metrics}
        self.trained_models[model_name] = result
        self.logger.info("Model %s trained with metrics %s", model_name, metrics)
        return result

    def _simulate_training_metrics(self, model_name: str) -> Dict[str, float]:
        random.seed(model_name)
        return {
            "loss": round(random.uniform(0.01, 0.05), 4),
            "sharpe_ratio": round(random.uniform(1.0, 2.5), 3),
            "information_ratio": round(random.uniform(0.5, 1.5), 3),
        }
