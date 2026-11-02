"""Patch index and normalisation statistics.

Patches are not copied to disk. We keep an index of (region, month, row, col) and
cut the patch from the region array when it is needed. A patch is kept only when
enough of it has a usable label and nearly all of it has Sentinel-1 data. The index
idea (instead of copied patches, which would multiply the dataset size) and the
validity thresholds follow a reference implementation, re-implemented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def window_starts(size: int, patch: int, stride: int) -> list[int]:
    """Start offsets covering [0, size). The last window is moved back to touch the edge."""
    if patch > size or patch <= 0 or stride <= 0:
        raise ValueError(f"bad sizes: size={size} patch={patch} stride={stride}")
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def window_fractions(mask: np.ndarray, rows: list[int], cols: list[int], patch: int) -> np.ndarray:
    """Share of True pixels in every patch window, via a summed-area table. Shape (len(rows), len(cols))."""
    sat = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.float64)
    sat[1:, 1:] = mask.astype(np.float64).cumsum(0).cumsum(1)
    r = np.asarray(rows)[:, None]
    c = np.asarray(cols)[None, :]
    total = sat[r + patch, c + patch] - sat[r, c + patch] - sat[r + patch, c] + sat[r, c]
    return total / (patch * patch)


def select_patches(
    label_valid: np.ndarray,
    s1_valid: np.ndarray,
    months: list[int],
    patch: int = 256,
    stride: int = 256,
    min_label: float = 0.4,
    min_s1: float = 0.99,
) -> pd.DataFrame:
    """Index of usable patches for one region.

    label_valid, s1_valid: bool arrays (T, H, W), one slice per entry of `months`.
    A patch-month is kept when at least `min_label` of its pixels have a label and at
    least `min_s1` have Sentinel-1 data.
    """
    if label_valid.shape != s1_valid.shape or label_valid.shape[0] != len(months):
        raise ValueError("label_valid, s1_valid and months must agree")
    h, w = label_valid.shape[1:]
    rows, cols = window_starts(h, patch, stride), window_starts(w, patch, stride)
    out = []
    for t, month in enumerate(months):
        lf = window_fractions(label_valid[t], rows, cols, patch)
        sf = window_fractions(s1_valid[t], rows, cols, patch)
        ok = (lf >= min_label) & (sf >= min_s1)
        for i, j in zip(*np.nonzero(ok)):
            out.append((month, rows[i], cols[j], float(lf[i, j]), float(sf[i, j])))
    return pd.DataFrame(out, columns=["month", "row", "col", "label_valid", "s1_valid"])


class RunningStats:
    """Mean and standard deviation over many arrays, ignoring NaN. Use train data only."""

    def __init__(self):
        self.n = 0
        self._s = 0.0
        self._ss = 0.0

    def update(self, arr) -> None:
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        self.n += a.size
        self._s += float(a.sum())
        self._ss += float((a * a).sum())

    @property
    def mean(self) -> float:
        if self.n == 0:
            raise ValueError("no values seen")
        return self._s / self.n

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self._ss / self.n - self.mean ** 2, 0.0)))
