"""Warning for areas the model is probably not reliable on.

No labels are needed at prediction time, so the check uses two label-free signals per area:
mean ensemble confidence (max class probability) and the share of pixels where the seeds disagree.
The reference values come from the held-out test regions (scripts/evaluate.py writes
ood_reference.json next to the weights). With only a handful of reference regions this is a coarse
flag, not a calibrated probability, and the report must say so.
"""

from __future__ import annotations

import numpy as np

from .classes import IGNORE_INDEX


def area_signature(conf: np.ndarray, members: np.ndarray, ignore: int = IGNORE_INDEX) -> dict:
    """conf (T,H,W) in 0..1 (float16 ok); members (K,T,H,W) class ids. Pixels with ignore in any member are skipped."""
    valid = (members != ignore).all(0)
    if not valid.any():
        return {"mean_conf": float("nan"), "disagree": float("nan")}
    c = np.asarray(conf, np.float32)[valid]
    disagree = (members != members[:1]).any(0)[valid].mean() if members.shape[0] > 1 else 0.0
    return {"mean_conf": float(np.nanmean(c)), "disagree": float(disagree)}


def ood_check(sig: dict, reference: dict | None) -> dict:
    """level: 'ok' | 'low' | 'unknown'. Low = confidence below every in-distribution reference city,
    or seed disagreement above every one of them."""
    if not reference or not reference.get("in_dist") or not np.isfinite(sig.get("mean_conf", np.nan)):
        return {"level": "unknown", "message": "No reference statistics available for this model."}
    ref = reference["in_dist"]
    lo = min(r["mean_conf"] for r in ref)
    hi = max(r["disagree"] for r in ref)
    low_conf, high_dis = sig["mean_conf"] < lo, sig["disagree"] > hi
    msg = (f"Mean confidence {sig['mean_conf']:.2f} (held-out cities: lowest {lo:.2f}); "
           f"seed disagreement {100 * sig['disagree']:.1f}% (held-out cities: highest {100 * hi:.1f}%).")
    if low_conf or high_dis:
        return {"level": "low", "message": "This area looks unlike the held-out test areas. " + msg}
    return {"level": "ok", "message": msg}
