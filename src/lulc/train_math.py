"""Small numeric helpers for training, kept free of torch so they can be unit-tested.

Class weights: square-root median-frequency balancing. Learning rate: linear warmup
then cosine decay on a fixed schedule, so every variant gets identical training
and differences between variants are not caused by a reactive scheduler.
"""

from __future__ import annotations

import math

import numpy as np


def class_weights(freq, power: float = 0.5) -> np.ndarray:
    """w_c = (median(freq) / freq_c) ** power over classes that occur, scaled to mean 1.

    Classes with zero frequency get weight 0 (they never appear, so they cannot matter).
    """
    f = np.asarray(freq, dtype=np.float64)
    present = f > 0
    if not present.any():
        raise ValueError("no class has any pixels")
    w = np.zeros_like(f)
    w[present] = (np.median(f[present]) / f[present]) ** power
    w[present] /= w[present].mean()
    return w


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    """Learning rate at an optimiser step: linear 0 -> base_lr, then cosine to min_lr."""
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("need 0 <= warmup_steps < total_steps")
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps - 1, 1), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def sampling_weights(regions, mode: str = "none"):
    """Per-patch sampling weight so that large regions do not dominate. None = uniform over patches.

    "equal": every region gets the same total weight (a 24 km city and a 10 km rural tile are drawn equally often);
    "sqrt": a region's total weight grows with the square root of its patch count (a compromise).
    """
    import numpy as np

    if mode == "none":
        return None
    if mode not in ("equal", "sqrt"):
        raise ValueError(f"region_balance must be none, equal or sqrt, got {mode!r}")
    r = np.asarray(regions)
    _, inv, counts = np.unique(r, return_inverse=True, return_counts=True)
    n = counts[inv].astype(float)
    return n ** (-1.0 if mode == "equal" else -0.5)
