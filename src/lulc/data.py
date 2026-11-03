"""Load exported region-months into compact in-memory stores and build patch indexes.

A store holds one region for the 12 months of the study year:
  s1     uint8 (T, 2, H, W)   VV and VH as 8-bit dB codes (0 = no data)
  label  uint8 (T, H, W)      6-class label, IGNORE_INDEX where unusable
  coh    float16 (C, h, w)    optional coarse coherence (100 m), NaN = no data
Labels are cut here: Dynamic World pixels with a mean winning-class probability
below `conf_cutoff` percent become IGNORE_INDEX. The raw export is never changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .classes import IGNORE_INDEX, remap_dynamic_world
from .patches import RunningStats, select_patches
from .quant import dequantize_db, quantize_db
from .yearly import STAT_NAMES, yearly_stats

N_MONTHS = 12


@dataclass
class RegionStore:
    key: str
    split: str
    s1: np.ndarray
    label: np.ndarray
    present: np.ndarray               # (T,) bool, False if the month file is missing
    coh: np.ndarray | None = None
    coh_factor: int = 10              # coherence pixels are this many 10 m pixels wide (100 m Earth Engine: 10, HyP3 20 m: 2)
    stats: np.ndarray | None = None   # (5, H, W) float16 yearly statistics, raw units
    year: int = 2021
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.s1.shape[2], self.s1.shape[3]


def month_file(out_dir: Path, key: str, year: int, month: int) -> Path:
    return Path(out_dir) / key / f"{year}-{month:02d}.npz"


def load_store_ym(out_dir, key: str, ym: list, split: str = "case_study", conf_cutoff: float = 0.0) -> RegionStore:
    """Read a list of (year, month) files of one region, in the given order.

    Missing months become no-data and present=False. Used for the training years and
    for the prototype, whose 12 months can cross a year boundary.
    """
    s1 = label = None
    present = np.zeros(len(ym), bool)
    scenes = [float("nan")] * len(ym)
    for i, (y, m) in enumerate(ym):
        path = month_file(out_dir, key, y, m)
        if not path.exists():
            continue
        with np.load(path) as z:
            vv, vh, lab, conf = z["vv"], z["vh"], z["dw_label"], z["dw_conf"]
            if "s1_n" in z.files:
                n = z["s1_n"][vv > -100]          # scenes per pixel, over pixels with data
                scenes[i] = float(np.median(n)) if n.size else float("nan")
        if s1 is None:
            h, w = vv.shape
            s1 = np.zeros((len(ym), 2, h, w), np.uint8)
            label = np.full((len(ym), h, w), IGNORE_INDEX, np.uint8)
        s1[i, 0] = quantize_db(vv.astype(np.float32))
        s1[i, 1] = quantize_db(vh.astype(np.float32))
        lab6 = remap_dynamic_world(lab)
        if conf_cutoff > 0:
            lab6 = np.where(conf >= conf_cutoff, lab6, IGNORE_INDEX).astype(np.uint8)
        label[i] = lab6
        present[i] = True
    if s1 is None:
        raise FileNotFoundError(f"no month files for {key} {ym[0]}..{ym[-1]} in {out_dir}")
    return RegionStore(key=key, split=split, s1=s1, label=label, present=present, year=ym[0][0],
                       meta={"s1_scenes": scenes})


def attach_stats(store: RegionStore) -> RegionStore:
    """Compute the five yearly statistics from the store's months (all present months)."""
    store.stats = yearly_stats(store.s1, store.present)
    return store


def load_store(out_dir, key: str, split: str, year: int = 2021, conf_cutoff: float = 0.0,
               coherence: bool = False, months=range(1, N_MONTHS + 1), stats: bool = False) -> RegionStore:
    """Read all months of one region and year."""
    store = load_store_ym(out_dir, key, [(year, m) for m in months], split, conf_cutoff)
    if stats:
        attach_stats(store)
    if coherence:
        cpath = Path(out_dir) / key / ("coherence_hyp3.npz" if coherence == "hyp3" else "coherence.npz")
        if not cpath.exists():
            raise FileNotFoundError(f"coherence requested but {cpath} is missing")
        with np.load(cpath) as z:
            store.coh = z["coherence"].astype(np.float16)
            store.coh_factor = int(z["factor"]) if "factor" in z.files else 10
    return store


def build_index(store: RegionStore, patch: int = 256, stride: int = 256,
                min_label: float = 0.4, min_s1: float = 0.99) -> pd.DataFrame:
    """Usable patch-months of a region: enough label and nearly complete Sentinel-1."""
    months = [i for i in range(store.s1.shape[0]) if store.present[i]]
    if not months:
        return pd.DataFrame(columns=["region", "month", "row", "col", "label_valid", "s1_valid"])
    label_valid = store.label[months] != IGNORE_INDEX
    s1_valid = store.s1[months, 0] > 0
    idx = select_patches(label_valid, s1_valid, months, patch, stride, min_label, min_s1)
    idx.insert(0, "region", store.key)
    return idx


def compute_norm(stores: list[RegionStore], step: int = 7) -> dict[str, tuple[float, float]]:
    """Mean and std of VV, VH, coherence and yearly statistics over TRAIN stores only (pixel subsample)."""
    vv, vh, coh = RunningStats(), RunningStats(), RunningStats()
    st = [RunningStats() for _ in STAT_NAMES]
    for s in stores:
        for i in range(s.s1.shape[0]):
            if not s.present[i]:
                continue
            vv.update(dequantize_db(s.s1[i, 0, ::step, ::step]))
            vh.update(dequantize_db(s.s1[i, 1, ::step, ::step]))
        if s.coh is not None:
            coh.update(s.coh.astype(np.float32))
        if s.stats is not None:
            for k, acc in enumerate(st):
                acc.update(s.stats[k, ::step, ::step].astype(np.float32))
    out = {"vv": (vv.mean, vv.std), "vh": (vh.mean, vh.std)}
    if coh.n:
        out["coh"] = (coh.mean, coh.std)
    if all(a.n for a in st):
        out.update({name: (a.mean, max(a.std, 1e-3)) for name, a in zip(STAT_NAMES, st)})  # constant channel must not give std 0
    return out


def window_months(month: int, window: int, n: int = N_MONTHS) -> list[int]:
    """Month indices of a centred window; the ends repeat the first or last month."""
    if window % 2 != 1 or window < 1:
        raise ValueError("window must be odd")
    half = window // 2
    return [min(max(month + d, 0), n - 1) for d in range(-half, half + 1)]


def coherence_patch(coh: np.ndarray, row: int, col: int, patch: int, factor: int = 10) -> np.ndarray:
    """Cut a patch from the coarse coherence array and upsample it (nearest) to the 10 m grid."""
    r0, c0 = row // factor, col // factor
    r1, c1 = -(-(row + patch) // factor), -(-(col + patch) // factor)
    block = coh[:, r0:r1, c0:c1]
    up = np.repeat(np.repeat(block, factor, axis=1), factor, axis=2)
    ro, co = row - r0 * factor, col - c0 * factor
    out = up[:, ro:ro + patch, co:co + patch]
    if out.shape[1:] != (patch, patch):  # patch reaches past the coarse edge
        pad = np.full((coh.shape[0], patch, patch), np.nan, dtype=coh.dtype)
        pad[:, :out.shape[1], :out.shape[2]] = out
        out = pad
    return out


def apply_agree_filter(store: RegionStore, data_dir) -> int:
    """Keep a training label only where WorldCover 2021 gives the same class (needs reference.npz of the region).

    Idea from reference implementations A and B (label quality filter, consensus labels). Training labels only: validation and
    test labels are never filtered. After this, WorldCover is no longer an independent check for that model, so
    scores against it must be read as agreement with part of the training signal; use Esri instead.
    Returns the share of valid labels that were removed (0..1)."""
    from pathlib import Path
    from .classes import remap_worldcover
    f = Path(data_dir) / store.key / "reference.npz"
    if not f.exists():
        raise FileNotFoundError(f"agree_filter needs {f} (run scripts/export_reference.py for the training regions)")
    with np.load(f) as z:
        wc = remap_worldcover(z["wc"])
    if wc.shape != store.label.shape[-2:]:
        raise ValueError(f"reference grid {wc.shape} does not match the labels {store.label.shape[-2:]} of {store.key}")
    before = store.label != IGNORE_INDEX
    store.label = np.where(store.label == wc[None], store.label, IGNORE_INDEX).astype(store.label.dtype)
    return float(1 - (store.label != IGNORE_INDEX).sum() / max(before.sum(), 1))


def apply_worldcover_labels(store: RegionStore, data_dir) -> None:
    """Replace the TRAINING labels with ESA WorldCover 2021 (annual, same label in every month). Needs reference.npz.
    After this, WorldCover is no longer an independent check for that model; score against Esri instead."""
    from pathlib import Path
    from .classes import remap_worldcover
    f = Path(data_dir) / store.key / "reference.npz"
    if not f.exists():
        raise FileNotFoundError(f"label_source=worldcover needs {f} (run scripts/export_reference.py for the training regions)")
    with np.load(f) as z:
        wc = remap_worldcover(z["wc"]).astype(store.label.dtype)
    if wc.shape != store.label.shape[-2:]:
        raise ValueError(f"reference grid {wc.shape} does not match the labels {store.label.shape[-2:]} of {store.key}")
    store.label = np.where(store.present[:, None, None], wc[None], IGNORE_INDEX).astype(store.label.dtype)
