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


def passes_border_rule(base: dict, other: dict, min_bf1: float = 0.02, max_miou_drop: float = 0.005,
                       max_class_drop: float = 0.05) -> tuple[bool, list[str]]:
    """Decision rule for any border-sharpening step (CRF, graph head). Set before the results.

    base / other: {'miou': float, 'bf1': float, 'iou': per-class IoU array}. Passes if boundary F1 rises by at least
    `min_bf1`, mIoU does not fall by more than `max_miou_drop`, and no class IoU falls by more than `max_class_drop`.
    Returns (passes, list of reasons for failing)."""
    why = []
    if other["bf1"] - base["bf1"] < min_bf1:
        why.append(f"boundary F1 {other['bf1'] - base['bf1']:+.3f} (needs +{min_bf1})")
    if other["miou"] - base["miou"] < -max_miou_drop:
        why.append(f"mIoU {other['miou'] - base['miou']:+.3f} (limit -{max_miou_drop})")
    d = np.asarray(other["iou"], float) - np.asarray(base["iou"], float)
    if np.isfinite(d).any() and np.nanmin(d) < -max_class_drop:
        why.append(f"a class IoU fell by {-np.nanmin(d):.3f} (limit {max_class_drop})")
    return (not why), why
