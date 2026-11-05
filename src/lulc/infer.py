"""Sliding-window inference for whole regions and for the prototype."""

from __future__ import annotations

import numpy as np
import torch

from .classes import IGNORE_INDEX
from .data import RegionStore, fill_window_gaps, window_months
from .patches import window_starts
from .torch_data import extra_patch, extra_spec, prepare


BLEND_MODES = ("uniform", "gaussian")


def blend_weights(patch: int, mode: str = "uniform") -> torch.Tensor:
    """Per-pixel weight of one tile when overlapping tiles are averaged, shape (patch, patch).

    "gaussian" trusts the tile centre over its edges, where the network sees less context (sigma = patch / 8).
    The floor is per axis, so tiles keep their relative weight along the region's outer rows, where a floor on the
    product would make all tiles equal and leave the seams there. It also keeps corners from dividing by almost nothing."""
    if mode not in BLEND_MODES:
        raise ValueError(f"blend must be one of {BLEND_MODES}, got {mode!r}")
    if mode == "uniform":
        return torch.ones(patch, patch)
    d = torch.arange(patch, dtype=torch.float32) - (patch - 1) / 2
    g = torch.exp(-d ** 2 / (2 * (patch / 8) ** 2)).clamp_min(1e-3)
    return torch.outer(g, g)


def softmax_tta(model, xin: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Softmax for a batch (B,C,H,W). With tta, the mean over the 8 flips and 90-degree rotations of a square input,
    each turned back first. Training augments with the same 8 (torch_data.PatchDataset), so none is new to the model."""
    if not tta:
        return torch.softmax(model(xin).float(), 1)
    total = 0.0
    for k in range(4):
        for flip in (False, True):
            t = torch.rot90(xin, k, (2, 3))
            p = torch.softmax(model(t.flip(3) if flip else t).float(), 1)
            total = total + torch.rot90(p.flip(3) if flip else p, -k, (2, 3))
    return total / 8


@torch.no_grad()
def predict_month(models, store: RegionStore, month: int, norm: dict, window: int, device,
                  patch: int = 512, overlap: int = 64, use_coh: bool = False, amp: bool = True,
                  use_stats: bool = False, return_probs: bool = False, tta: bool = False, blend: str = "uniform",
                  fill_gaps: bool = True):
    """Mean softmax over `models` for one month. Returns (pred uint8 (H,W), maxprob float16 (H,W),
    member_preds list of uint8 (H,W)). No-data Sentinel-1 pixels (VV code 0) get IGNORE_INDEX.
    tta (8 times slower) and blend are the optional border refinements; their defaults leave the maps as they were.
    fill_gaps fills no-data neighbour months from the nearest month in the window (data.fill_window_gaps); False feeds them as blanks."""
    h, w = store.size
    patch = min(patch, h, w)
    stride = max(patch - overlap, 1)
    weight = blend_weights(patch, blend)
    idx = window_months(month, window, store.s1.shape[0])
    stack = store.s1[idx]
    if fill_gaps:
        stack = fill_window_gaps(stack)
    full = torch.from_numpy(stack.reshape(window * 2, h, w))
    n_cls = None
    spec = extra_spec(store.coh.shape[0] if use_coh else 0, use_stats)
    acc = None
    members = []
    rows, cols = window_starts(h, patch, stride), window_starts(w, patch, stride)
    cnt = torch.zeros(h, w)
    for model in models:
        model.eval()
        probs = None
        for r in rows:
            for c in cols:
                x = full[None, :, r:r + patch, c:c + patch].to(device)
                coh = (torch.from_numpy(extra_patch(store, r, c, patch, use_coh, use_stats).astype(np.float32))[None].to(device)
                       if (use_coh or use_stats) else None)
                xin = prepare(x, coh, norm, window, spec)
                with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    p = softmax_tta(model, xin, tta)[0].cpu()
                if probs is None:
                    n_cls = p.shape[0]
                    probs = torch.zeros(n_cls, h, w)
                probs[:, r:r + patch, c:c + patch] += p * weight
                if model is models[0]:
                    cnt[r:r + patch, c:c + patch] += weight
        probs /= cnt
        members.append(probs.argmax(0).numpy().astype(np.uint8))
        acc = probs if acc is None else acc + probs
    acc /= len(models)
    pred = acc.argmax(0).numpy().astype(np.uint8)
    conf = acc.max(0).values.numpy().astype(np.float16)
    nodata = store.s1[month, 0] == 0
    pred[nodata] = IGNORE_INDEX
    for m in members:
        m[nodata] = IGNORE_INDEX
    if return_probs:
        p = acc.numpy().astype(np.float16)
        p[:, nodata] = np.nan
        return pred, conf, members, p
    return pred, conf, members


def predict_region(models, store, norm, window, device, months=None, return_probs: bool = False, **kw):
    """All present months. Returns preds (T,H,W), conf (T,H,W), members (K,T,H,W), months [, probs (T,C,H,W) float16]."""
    months = list(months) if months is not None else [i for i in range(store.s1.shape[0]) if store.present[i]]
    preds, confs, mem, probs = [], [], [], []
    for m in months:
        res = predict_month(models, store, m, norm, window, device, return_probs=return_probs, **kw)
        p, c, ms = res[:3]
        preds.append(p); confs.append(c); mem.append(np.stack(ms))
        if return_probs:
            probs.append(res[3])
    out = (np.stack(preds), np.stack(confs), np.stack(mem, axis=1), months)
    return out + (np.stack(probs),) if return_probs else out
