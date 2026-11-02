"""Is the monthly change real? Measure the change our maps report where nothing changed.

Idea: pixels where two independent annual maps (WorldCover and Esri) agree are treated as
stable land cover. Any month-to-month class change a model reports there is seasonal
noise or classification error, not land-use change. We measure it per model and after a
temporal majority vote, and compare with the same number for Dynamic World itself.
"""

from __future__ import annotations

import numpy as np

from .classes import CLASS_NAMES, IGNORE_INDEX, N_CLASSES
from .stats import class_percentages, monthly_change_table


def stable_mask(map_a: np.ndarray, map_b: np.ndarray, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """True where both annual maps give the same valid class."""
    return (map_a == map_b) & (map_a != ignore)


def temporal_majority(preds: np.ndarray, window: int = 3, n_classes: int = N_CLASSES,
                      ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Per-pixel majority class over a window of months, shifted inward at the ends. Ties go to the lower class id."""
    if window % 2 != 1:
        raise ValueError("window must be odd")
    t, h, w = preds.shape
    half = window // 2
    out = np.full(preds.shape, ignore, dtype=np.uint8)
    for i in range(t):
        # Centred window; at the ends the window is shifted inward (Jan uses Jan-Mar, Dec uses Oct-Dec).
        # Repeating the end month would give it two votes of three and leave it unsmoothed.
        start = min(max(i - half, 0), max(t - window, 0))
        idx = list(range(start, min(start + window, t)))
        counts = np.zeros((n_classes, h, w), dtype=np.uint8)
        for j in idx:
            p = preds[j]
            for c in range(n_classes):
                counts[c] += (p == c)
        best = counts.argmax(0).astype(np.uint8)
        best[counts.sum(0) == 0] = ignore
        out[i] = best
    return out


def probability_smooth(probs: np.ndarray, window: int = 3, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Average the class probabilities (T,C,H,W) over the same inward-shifted window as temporal_majority, then take the argmax.

    Keeps the confidence of each month, which a majority vote of hard classes throws away. Pixels with no valid month stay ignore
    (a month with no radar must be marked by the caller: its probabilities are NaN).
    """
    if window % 2 != 1:
        raise ValueError("window must be odd")
    t = probs.shape[0]
    half = window // 2
    out = np.zeros((t, *probs.shape[2:]), np.uint8)
    for i in range(t):
        start = min(max(i - half, 0), max(t - window, 0))
        block = probs[start:min(start + window, t)].astype(np.float32)
        mean = np.nanmean(block, 0)
        best = np.nan_to_num(mean, nan=-1).argmax(0).astype(np.uint8)
        best[np.isnan(mean).all(0)] = ignore
        out[i] = best
    return out


def false_change(preds: np.ndarray, mask: np.ndarray) -> dict:
    """Change reported on stable pixels only.

    flip_rate_pct         mean share of stable pixels that change class between consecutive months
    monthly_churn_pp      mean over month steps of (sum of |class share change|) / 2, in pp
    net_total_change_pp   (sum of |last month - first month class share|) / 2, in pp
    n_pixels              stable pixels used (valid in all months)
    """
    p = np.where(mask[None], preds, IGNORE_INDEX).astype(np.uint8)
    valid_all = (p != IGNORE_INDEX).all(0)
    if not valid_all.any():
        return {"flip_rate_pct": float("nan"), "monthly_churn_pp": float("nan"),
                "net_total_change_pp": float("nan"), "n_pixels": 0}
    flips = [(p[i][valid_all] != p[i - 1][valid_all]).mean() * 100 for i in range(1, len(p))]
    st = monthly_change_table(list(p), [str(i) for i in range(len(p))], CLASS_NAMES, IGNORE_INDEX, True)
    churn = st.delta.iloc[1:].abs().sum(axis=1).mean() / 2
    return {"flip_rate_pct": float(np.mean(flips)), "monthly_churn_pp": float(churn),
            "net_total_change_pp": float(st.total_change.abs().sum() / 2), "n_pixels": int(valid_all.sum())}


def member_spread(pred_stack: np.ndarray, n_classes: int = N_CLASSES) -> dict:
    """Spread of class percentages across ensemble members.

    pred_stack (K, T, H, W). Returns mean and std, each shape (T, n_classes), in percent.
    The std is a model-uncertainty bar for every monthly share and every change in pp.
    """
    k, t = pred_stack.shape[:2]
    shared = (pred_stack != IGNORE_INDEX).all(axis=(0, 1))
    tab = np.array([[class_percentages(pred_stack[m, i], n_classes, valid=shared) for i in range(t)]
                    for m in range(k)])
    return {"mean": tab.mean(0), "std": tab.std(0, ddof=1) if k > 1 else np.zeros_like(tab[0])}
