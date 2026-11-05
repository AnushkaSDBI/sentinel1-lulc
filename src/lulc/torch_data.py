"""Torch dataset and GPU-side input preparation.

Samples leave the dataset as raw uint8 dB codes (cheap to move); normalisation happens
on the GPU in `prepare`, so workers only slice and flip arrays.
Channel order: for each month in the window VV, VH; then the extras in this order:
coherence bands (optional, one shared normalisation "coh"), then the five yearly
statistics (optional, "st0".."st4", one normalisation each).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import RegionStore, coherence_patch, window_months
from .quant import DB_MIN, DB_STEP
from .yearly import STAT_NAMES


def extra_spec(n_coh: int = 0, stats: bool = False) -> list[tuple[str, int]]:
    """[(norm key, number of channels)] in channel order."""
    spec = [("coh", n_coh)] if n_coh else []
    if stats:
        spec += [(name, 1) for name in STAT_NAMES]
    return spec


def n_extra(spec) -> int:
    return sum(c for _, c in spec)


def n_input_channels(window: int, n_extra_ch: int) -> int:
    return 2 * window + n_extra_ch


def month_stack(store: RegionStore, month: int, window: int, r: int, c: int, patch: int) -> np.ndarray:
    idx = window_months(month, window, store.s1.shape[0])
    return store.s1[idx, :, r:r + patch, c:c + patch].reshape(window * 2, patch, patch)


def extra_patch(store: RegionStore, r: int, c: int, patch: int, use_coh: bool, use_stats: bool) -> np.ndarray:
    parts = []
    if use_coh:
        parts.append(coherence_patch(store.coh, r, c, patch, store.coh_factor).astype(np.float16))
    if use_stats:
        parts.append(store.stats[:, r:r + patch, c:c + patch].astype(np.float16))
    return np.concatenate(parts) if parts else np.zeros((0, patch, patch), np.float16)


class PatchDataset(Dataset):
    """index: DataFrame(region, month, row, col). stores: {region: RegionStore}."""

    def __init__(self, stores: dict, index, window: int = 1, patch: int = 256,
                 augment: bool = False, coherence: bool = False, seed: int = 0, stats: bool = False, jitter: int = 0):
        self.stores, self.window, self.patch = stores, window, patch
        self.augment, self.coherence, self.stats = augment, coherence, stats
        self.rows = index[["region", "month", "row", "col"]].to_records(index=False)
        self.seed, self.jitter = seed, jitter
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        key, m, r, c = self.rows[i]
        s = self.stores[key]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.augment else None
        if rng is not None and self.jitter > 0:                      # shift the window a little, stay inside the tile
            h, w = s.label.shape[-2:]
            r = int(np.clip(r + rng.integers(-self.jitter, self.jitter + 1), 0, h - self.patch))
            c = int(np.clip(c + rng.integers(-self.jitter, self.jitter + 1), 0, w - self.patch))
        x = month_stack(s, int(m), self.window, int(r), int(c), self.patch)
        y = s.label[int(m), r:r + self.patch, c:c + self.patch]
        coh = extra_patch(s, int(r), int(c), self.patch, self.coherence, self.stats)
        if self.augment:
            k = int(rng.integers(4))
            flip = bool(rng.integers(2))
            def tf(a):
                a = np.rot90(a, k, axes=(-2, -1))
                return a[..., ::-1] if flip else a
            x, y, coh = tf(x), tf(y), tf(coh)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(coh)),
                torch.from_numpy(np.ascontiguousarray(y)).long())


def add_speckle(db: torch.Tensor, valid: torch.Tensor, looks: float, p: float) -> torch.Tensor:
    """Extra multiplicative speckle on a fraction p of the samples, in the dB domain.

    Linear-power noise ~ Gamma(shape=L, scale=1/L) has mean 1, so each pixel gets
    10*log10(noise) added. A composite with ~10 effective looks plus L=10 extra noise
    behaves like ~5 looks, which is roughly a one-satellite year (2022-2024).
    """
    if p <= 0 or looks <= 0:
        return db
    pick = (torch.rand(db.shape[0], 1, 1, 1, device=db.device) < p).float()
    g = torch.distributions.Gamma(torch.tensor(float(looks), device=db.device),
                                  torch.tensor(float(looks), device=db.device)).sample(db.shape)
    noisy = db + 10.0 * torch.log10(g.clamp_min(1e-6))
    return torch.where((pick > 0) & valid, noisy, db)


def prepare(x_u8: torch.Tensor, extra: torch.Tensor, norm: dict, window: int, spec=None,
            speckle_looks: float = 0.0, speckle_p: float = 0.0, db_shift: float = 0.0,
            month_dropout: float = 0.0) -> torch.Tensor:
    """uint8 codes (B,2W,H,W) (+ extras (B,E,H,W)) -> normalised float; no-data and NaN -> 0.

    Training-only options (all off by default): db_shift moves VV and VH of a sample by a random offset of up to
    +-db_shift dB (gain and moisture differences between scenes); month_dropout blanks whole months of the window
    (never all of them), so the model cannot lean on one month; speckle as in add_speckle."""
    x = x_u8.float()
    b = x.shape[0]
    if month_dropout > 0 and window > 1:
        keep = torch.rand(b, window, device=x.device) >= month_dropout
        keep[~keep.any(1), 0] = True
        x = x * keep.repeat_interleave(2, dim=1)[:, :, None, None].float()
    nodata = x == 0
    db = x * DB_STEP + DB_MIN
    if db_shift > 0:
        off = (torch.rand(b, 2, device=x.device) * 2 - 1) * db_shift
        db = torch.where(nodata, db, db + off.repeat(1, window)[:, :, None, None])
    db = add_speckle(db, ~nodata, speckle_looks, speckle_p)
    mean = torch.tensor([norm["vv"][0], norm["vh"][0]] * window, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor([norm["vv"][1], norm["vh"][1]] * window, device=x.device).view(1, -1, 1, 1)
    out = (db - mean) / std
    out = torch.where(nodata, torch.zeros_like(out), out)
    if extra is not None and extra.shape[1] > 0:
        spec = spec or [("coh", extra.shape[1])]
        em = torch.tensor([norm[k][0] for k, c in spec for _ in range(c)], device=x.device).view(1, -1, 1, 1)
        es = torch.tensor([norm[k][1] for k, c in spec for _ in range(c)], device=x.device).view(1, -1, 1, 1)
        c = extra.float()
        bad = torch.isnan(c)
        c = (c - em) / es
        out = torch.cat([out, torch.where(bad, torch.zeros_like(c), c)], 1)
    return out
