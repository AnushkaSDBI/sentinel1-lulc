"""Sliding-window inference for whole regions and for the prototype."""

from __future__ import annotations

import numpy as np
import torch

from .classes import IGNORE_INDEX
from .data import RegionStore, window_months
from .patches import window_starts
from .torch_data import extra_patch, extra_spec, prepare


@torch.no_grad()
def predict_month(models, store: RegionStore, month: int, norm: dict, window: int, device,
                  patch: int = 512, overlap: int = 64, use_coh: bool = False, amp: bool = True,
                  use_stats: bool = False, return_probs: bool = False):
    """Mean softmax over `models` for one month. Returns (pred uint8 (H,W), maxprob float16 (H,W),
    member_preds list of uint8 (H,W)). No-data Sentinel-1 pixels (VV code 0) get IGNORE_INDEX."""
    h, w = store.size
    patch = min(patch, h, w)
    stride = max(patch - overlap, 1)
    idx = window_months(month, window, store.s1.shape[0])
    full = torch.from_numpy(store.s1[idx].reshape(window * 2, h, w))
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
                    p = torch.softmax(model(xin).float(), 1)[0].cpu()
                if probs is None:
                    n_cls = p.shape[0]
                    probs = torch.zeros(n_cls, h, w)
                probs[:, r:r + patch, c:c + patch] += p
                if model is models[0]:
                    cnt[r:r + patch, c:c + patch] += 1
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
