"""Yearly summary channels from the 12 monthly composites (order-independent).

Five per-pixel statistics of the monthly VV/VH dB series. They do not depend on which month
comes first, so they transfer across hemispheres, unlike a calendar-tied decomposition (PCA).
Statistic set and the -18 dB water threshold follow a reference implementation (re-implemented).

    st0  VV minimum            st3  VH spread: 2nd highest - 2nd lowest month
    st1  VV maximum            st4  share of months with VV < -18 dB (open water, smooth surfaces)
    st2  VH standard deviation
"""

from __future__ import annotations

import warnings

import numpy as np

from .quant import dequantize_db

STAT_NAMES = ("st0", "st1", "st2", "st3", "st4")
WATER_DB = -18.0
MIN_MONTHS = 4          # the 2nd-highest minus 2nd-lowest needs at least four months


def yearly_stats(s1: np.ndarray, present: np.ndarray | None = None, rows: int = 256) -> np.ndarray:
    """s1 uint8 (T, 2, H, W) codes -> float16 (5, H, W). NaN where fewer than MIN_MONTHS months have data."""
    t, _, h, w = s1.shape
    use = np.flatnonzero(present) if present is not None else np.arange(t)
    out = np.full((len(STAT_NAMES), h, w), np.nan, np.float32)
    for r0 in range(0, h, rows):
        sl = slice(r0, min(r0 + rows, h))
        vv = dequantize_db(s1[use][:, 0, sl])
        vh = dequantize_db(s1[use][:, 1, sl])
        n = np.isfinite(vv).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vh_sorted = np.sort(vh, axis=0)                      # NaN last
            nv = np.isfinite(vh).sum(0)
            idx_hi = np.clip(nv - 2, 0, None)[None]
            second_hi = np.take_along_axis(vh_sorted, idx_hi, 0)[0]
            block = np.stack([np.nanmin(vv, 0), np.nanmax(vv, 0), np.nanstd(vh, 0),
                              second_hi - vh_sorted[1] if len(use) > 1 else np.full(n.shape, np.nan),
                              (vv < WATER_DB).sum(0) / np.maximum(n, 1)])
        block[:, n < MIN_MONTHS] = np.nan
        out[:, sl] = block
    return out.astype(np.float16)
