"""Class percentages and month-to-month change, in percentage points.

Design choices (state these in the report):
- Percentages are shares of VALID pixels (not IGNORE_INDEX / no-data).
- By default all months are evaluated on the SAME valid-pixel mask (the
  intersection across months). Otherwise a pixel that is missing in one
  month would show up as fake land-use change.
- Change is in percentage points (pp): delta = pct[m] - pct[m-1].
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .classes import CLASS_NAMES, IGNORE_INDEX


def class_percentages(
    pred: np.ndarray,
    n_classes: int,
    ignore_index: int = IGNORE_INDEX,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Percent of valid pixels per class, shape (n_classes,). NaN if no valid pixels."""
    pred = np.asarray(pred)
    mask = pred != ignore_index
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != pred.shape:
            raise ValueError(f"valid mask shape {valid.shape} != pred shape {pred.shape}")
        mask &= valid
    values = pred[mask]
    if values.size == 0:
        return np.full(n_classes, np.nan)
    if values.max() >= n_classes:
        raise ValueError(f"prediction contains class {int(values.max())} >= n_classes={n_classes}")
    counts = np.bincount(values.astype(np.int64), minlength=n_classes)
    return counts / values.size * 100.0


@dataclass(frozen=True)
class MonthlyStats:
    percent: pd.DataFrame        # rows = months, cols = classes, values in %
    delta: pd.DataFrame          # month-to-month change in pp (first row NaN)
    total_change: pd.Series      # last month minus first month, in pp
    valid_fraction: pd.Series    # share of pixels used per month (0..1)


def monthly_change_table(
    preds: Sequence[np.ndarray],
    labels: Sequence[str],
    class_names: Sequence[str] = CLASS_NAMES,
    ignore_index: int = IGNORE_INDEX,
    common_mask: bool = True,
) -> MonthlyStats:
    """Per-month class shares plus month-to-month and total change in pp."""
    if len(preds) != len(labels):
        raise ValueError("preds and labels must have the same length")
    if len(preds) < 2:
        raise ValueError("need at least two months to compute change")
    shape = np.asarray(preds[0]).shape
    for p in preds:
        if np.asarray(p).shape != shape:
            raise ValueError("all monthly predictions must have the same shape")

    stack = np.stack([np.asarray(p) for p in preds])
    per_month_valid = stack != ignore_index
    shared = per_month_valid.all(axis=0) if common_mask else None

    n = len(class_names)
    rows = [class_percentages(p, n, ignore_index, valid=shared) for p in stack]
    percent = pd.DataFrame(rows, index=list(labels), columns=list(class_names))
    delta = percent.diff()
    total = percent.iloc[-1] - percent.iloc[0]
    if shared is not None:
        frac = pd.Series(shared.mean(), index=list(labels))
    else:
        frac = pd.Series(per_month_valid.reshape(len(labels), -1).mean(axis=1), index=list(labels))
    return MonthlyStats(percent=percent, delta=delta, total_change=total, valid_fraction=frac)
