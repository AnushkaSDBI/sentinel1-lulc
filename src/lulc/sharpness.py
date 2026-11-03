"""Boundary F1: how well predicted class borders line up with reference class borders.

mIoU rewards getting the area right and barely notices a border that is a few pixels soft.
This metric looks only at borders. A border pixel is a pixel whose right or lower neighbour
has a different valid class. A predicted border pixel is correct if a reference border pixel
lies within `tol` pixels, and the other way round for recall.
"""

from __future__ import annotations

import numpy as np

IGNORE = 255


def edges(m, ignore: int = IGNORE) -> np.ndarray:
    m = np.asarray(m)
    e = np.zeros(m.shape, bool)
    for ax in (-2, -1):
        a = np.moveaxis(m, ax, 0)
        d = (a[1:] != a[:-1]) & (a[1:] != ignore) & (a[:-1] != ignore)
        ee = np.moveaxis(e, ax, 0)
        ee[1:] |= d
        ee[:-1] |= d
    return e


def dilate(e, tol: int) -> np.ndarray:
    out = np.asarray(e).copy()
    h, w = out.shape[-2:]
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.zeros_like(e)
            sh[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = e[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            out |= sh
    return out


def boundary_counts(pred, ref, tol: int = 1, ignore: int = IGNORE) -> tuple[int, int, int, int]:
    """(pred border pixels hit, pred border pixels, ref border pixels hit, ref border pixels); sums over (T,H,W) or (H,W).

    Pixels where the reference is ignored are left out on both sides.
    """
    pred, ref = np.asarray(pred), np.asarray(ref)
    ok = ref != ignore
    pe = edges(np.where(ok, pred, ignore), ignore)
    re = edges(ref, ignore)
    return (int((pe & dilate(re, tol)).sum()), int(pe.sum()), int((re & dilate(pe, tol)).sum()), int(re.sum()))


def f1_from_counts(c) -> float:
    p = c[0] / c[1] if c[1] else 0.0
    r = c[2] / c[3] if c[3] else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0
