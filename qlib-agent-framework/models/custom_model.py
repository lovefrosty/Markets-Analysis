"""
Example custom model stub for QLib integration.

Drop this file into your workflow and update the agent/model configuration
to point at `models.custom_model.CustomAlphaModel`.  The class demonstrates
the minimal API expected by QLib's training/execution pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from qlib.model.base import BaseModel


@dataclass
class CustomModelConfig:
    """Configuration container for the stub model."""

    window: int = 5
    scale: float = 1.0


class CustomAlphaModel(BaseModel):
    """
    Lightweight example model that averages recent labels and scales them.

    The intent is to provide a clear integration point; swap the logic inside
    :meth:`fit` and :meth:`predict` with your actual modelling code.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.cfg = CustomModelConfig(**kwargs)
        self._fitted_mean: Optional[pd.Series] = None

    def fit(self, dataset: Any) -> None:  # dataset: DatasetH or alike
        df = dataset.prepare("train", col_set="label")
        if df is None or df.empty:
            raise ValueError("Training dataset produced an empty label frame.")
        label = df.iloc[:, 0]
        self._fitted_mean = label.groupby("instrument").rolling(self.cfg.window).mean()

    def predict(self, dataset: Any, segment: str = "test") -> pd.Series:
        if self._fitted_mean is None:
            raise RuntimeError("Call fit() before predict().")
        df = dataset.prepare(segment, col_set="label")
        label = df.iloc[:, 0]
        joined = label.to_frame("label").join(self._fitted_mean.rename("mean"), how="left")
        signal = joined["mean"].fillna(joined["label"].mean())
        return signal * self.cfg.scale

    def score(self, pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
        diff = pred - label
        return {"mse": float(np.mean(diff**2))}
