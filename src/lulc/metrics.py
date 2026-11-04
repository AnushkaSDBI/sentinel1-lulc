"""Confusion-matrix metrics: OA, per-class IoU and F1, mIoU, bootstrap over cities.

mIoU is the mean over classes that occur in the reference (support > 0). A class that
never occurs in the reference has IoU NaN and is skipped, so absent classes do not
drag the mean to zero. Report per-class IoU next to every mIoU.

On a small tile, a class with a handful of reference pixels counts as much as the dominant
class. miou_major() averages only classes above a minimum share of the reference, for
comparing regions. It is a companion to miou(), not a replacement: predictions of a dropped
class still lower the IoU of the classes kept, but the dropped class's own IoU is not averaged.
"""

from __future__ import annotations

import numpy as np

from .classes import IGNORE_INDEX


class ConfusionMatrix:
    """rows = reference, columns = prediction."""

    def __init__(self, n_classes: int, ignore_index: int = IGNORE_INDEX):
        self.n = n_classes
        self.ignore = ignore_index
        self.m = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred).ravel().astype(np.int64)
        target = np.asarray(target).ravel().astype(np.int64)
        keep = (target != self.ignore) & (target < self.n) & (pred < self.n) & (pred >= 0)
        self.m += np.bincount(target[keep] * self.n + pred[keep], minlength=self.n ** 2).reshape(self.n, self.n)

    def merge(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        out = ConfusionMatrix(self.n, self.ignore)
        out.m = self.m + other.m
        return out

    @property
    def support(self) -> np.ndarray:
        return self.m.sum(1)

    @property
    def oa(self) -> float:
        tot = self.m.sum()
        return float(np.trace(self.m) / tot) if tot else float("nan")

    def iou(self) -> np.ndarray:
        tp = np.diag(self.m).astype(float)
        union = self.m.sum(0) + self.m.sum(1) - tp
        out = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        out[self.support == 0] = np.nan
        return out

    def f1(self) -> np.ndarray:
        tp = np.diag(self.m).astype(float)
        den = self.m.sum(0) + self.m.sum(1)
        out = np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)
        out[self.support == 0] = np.nan
        return out

    def precision(self) -> np.ndarray:
        tp = np.diag(self.m).astype(float)
        col = self.m.sum(0)
        return np.where(col > 0, tp / np.maximum(col, 1), np.nan)

    def recall(self) -> np.ndarray:
        tp = np.diag(self.m).astype(float)
        out = np.where(self.support > 0, tp / np.maximum(self.support, 1), np.nan)
        return out

    @property
    def kappa(self) -> float:
        tot = self.m.sum()
        if not tot:
            return float("nan")
        po = np.trace(self.m) / tot
        pe = float((self.m.sum(0) * self.m.sum(1)).sum()) / tot ** 2
        return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

    def miou(self) -> float:
        v = self.iou()
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    def major_classes(self, min_share: float = 0.01) -> np.ndarray:
        """Boolean mask of classes holding at least min_share of the reference pixels (never an absent class)."""
        sup = self.support
        return (sup > 0) & (sup >= min_share * sup.sum())

    def miou_major(self, min_share: float = 0.01) -> float:
        """Mean IoU over major_classes(min_share). NaN if the matrix is empty."""
        keep = self.major_classes(min_share)
        return float(self.iou()[keep].mean()) if keep.any() else float("nan")


def bootstrap_miou(per_city: dict[str, ConfusionMatrix], n_boot: int = 1000, seed: int = 0) -> dict:
    """Resample CITIES with replacement and recompute pooled mIoU and OA.

    Cities are the independent units, so this is the honest interval for 'how would the
    score change with other cities'. With few cities the interval is wide, which is the point.
    """
    keys = list(per_city)
    if not keys:
        raise ValueError("no cities")
    rng = np.random.default_rng(seed)
    mious, oas = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        cm = per_city[keys[pick[0]]]
        for i in pick[1:]:
            cm = cm.merge(per_city[keys[i]])
        mious.append(cm.miou())
        oas.append(cm.oa)
    pooled = per_city[keys[0]]
    for k in keys[1:]:
        pooled = pooled.merge(per_city[k])
    lo, hi = np.nanpercentile(mious, [2.5, 97.5])
    olo, ohi = np.nanpercentile(oas, [2.5, 97.5])
    return {"miou": pooled.miou(), "miou_lo": float(lo), "miou_hi": float(hi),
            "oa": pooled.oa, "oa_lo": float(olo), "oa_hi": float(ohi), "n_cities": len(keys)}
