from __future__ import annotations

import numpy as np
import torch


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Return 1 - Var[y_true - y_pred] / Var[y_true]."""

    y_pred_np = y_pred.detach().float().cpu().numpy().reshape(-1)
    y_true_np = y_true.detach().float().cpu().numpy().reshape(-1)
    var_y = np.var(y_true_np)
    if var_y == 0:
        return float("nan")
    return float(1.0 - np.var(y_true_np - y_pred_np) / var_y)


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))
