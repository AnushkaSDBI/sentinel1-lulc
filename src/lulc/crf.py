"""Convolutional CRF: a post-processing step that sharpens class borders, no training.

Idea (Teichmann and Cipolla 2018, "Convolutional CRFs for Semantic Segmentation"): mean-field inference of a
fully connected CRF, restricted to a k x k window so it runs as shifted array operations. Written from the paper's
idea in plain PyTorch; no code copied.

Each pixel's class probabilities are pulled towards the neighbours' probabilities. The pull between two pixels is
strong when they are close and when a GUIDE image looks the same at both. At a border in the guide the pull is weak,
so the border survives. A single month of radar is full of speckle, so the guide here is the yearly median of the
12 months (in dB), where speckle is averaged out and real borders (fields, shores, roads) remain.

Settings are fixed in advance (not tuned on validation):
    mild    k=7,  theta_xy=3, theta_i=2.0 dB, strength=2
    strong  k=11, theta_xy=5, theta_i=2.0 dB, strength=4
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .quant import dequantize_db

CRF_SETTINGS = {
    "mild": dict(k=7, theta_xy=3.0, theta_i=2.0, strength=2.0, iters=5),
    "strong": dict(k=11, theta_xy=5.0, theta_i=2.0, strength=4.0, iters=5),
}


def yearly_median_db(s1: np.ndarray, present: np.ndarray | None = None, rows: int = 256) -> np.ndarray:
    """s1 uint8 (T,2,H,W) codes -> float32 (2,H,W) per-pixel median dB over the present months (NaN where none)."""
    t, _, h, w = s1.shape
    use = np.flatnonzero(present) if present is not None else np.arange(t)
    out = np.full((2, h, w), np.nan, np.float32)
    import warnings
    for r0 in range(0, h, rows):
        sl = slice(r0, min(r0 + rows, h))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for b in (0, 1):
                out[b, sl] = np.nanmedian(dequantize_db(s1[use][:, b, sl]), axis=0)
    return out


def _offsets(k: int):
    half = k // 2
    return [(dy, dx) for dy in range(-half, half + 1) for dx in range(-half, half + 1) if (dy, dx) != (0, 0)]


def _shift(x: torch.Tensor, dy: int, dx: int, half: int, H: int, W: int) -> torch.Tensor:
    """x padded by `half` (replicate) -> the array shifted so out[y,x] = x[y+dy, x+dx]."""
    return x[..., half + dy:half + dy + H, half + dx:half + dx + W]


@torch.no_grad()
def conv_crf(probs: torch.Tensor, guide: torch.Tensor, k: int = 7, theta_xy: float = 3.0, theta_i: float = 2.0,
             strength: float = 2.0, iters: int = 5, w_app: float = 1.0, w_smooth: float = 1.0) -> torch.Tensor:
    """probs (C,H,W) softmax probabilities, guide (G,H,W) in dB -> refined probabilities (C,H,W).

    NaN probabilities (no radar) become uniform and NaN guide values take the channel median, so they never break the
    maths; the caller must put no-data pixels back to ignore afterwards.
    """
    if k % 2 != 1:
        raise ValueError("k must be odd")
    C, H, W = probs.shape
    half = k // 2
    p = torch.nan_to_num(probs.float(), nan=1.0 / C)
    g = guide.float().clone()
    for b in range(g.shape[0]):
        bad = torch.isnan(g[b])
        if bad.any():
            g[b][bad] = torch.nanmedian(g[b][~bad]) if (~bad).any() else 0.0
    logu = torch.log(p.clamp_min(1e-6))
    gp = F.pad(g[None], (half,) * 4, mode="replicate")[0]
    offs = _offsets(k)
    wts = []
    for dy, dx in offs:
        d2 = float(dy * dy + dx * dx)
        spatial = float(np.exp(-d2 / (2 * theta_xy ** 2)))
        diff2 = ((g - _shift(gp, dy, dx, half, H, W)) ** 2).sum(0)
        wts.append(spatial * (w_app * torch.exp(-diff2 / (2 * theta_i ** 2)) + w_smooth))
    wts = torch.stack(wts)
    wts = wts / wts.sum(0, keepdim=True)
    q = p
    for _ in range(iters):
        qp = F.pad(q[None], (half,) * 4, mode="replicate")[0]
        msg = torch.zeros_like(q)
        for wi, (dy, dx) in zip(wts, offs):
            msg += wi * _shift(qp, dy, dx, half, H, W)
        q = torch.softmax(logu + strength * msg, 0)
    return q


@torch.no_grad()
def conv_crf_tiled(probs: torch.Tensor, guide: torch.Tensor, tile: int = 512, **kw) -> torch.Tensor:
    """Same result as conv_crf on the whole image, in tiles with a halo of iters*(k//2) pixels (keeps memory small)."""
    C, H, W = probs.shape
    halo = kw.get("iters", 5) * (kw.get("k", 7) // 2)
    if H <= tile + 2 * halo and W <= tile + 2 * halo:
        return conv_crf(probs, guide, **kw)
    pf = torch.nan_to_num(probs.float(), nan=1.0 / C)
    gf = guide.float()
    out = torch.empty(C, H, W)
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            r1, c1 = min(r + tile, H), min(c + tile, W)
            ra, rb, ca, cb = max(r - halo, 0), min(r1 + halo, H), max(c - halo, 0), min(c1 + halo, W)   # no padding at the true image edge: conv_crf pads it itself
            sub = conv_crf(pf[:, ra:rb, ca:cb], gf[:, ra:rb, ca:cb], **kw)
            out[:, r:r1, c:c1] = sub[:, r - ra:r1 - ra, c - ca:c1 - ca]
    return out
