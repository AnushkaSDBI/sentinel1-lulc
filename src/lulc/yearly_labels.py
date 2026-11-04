"""Yearly Dynamic World label ("yearly fill") and how to use it.

Monthly Dynamic World labels have holes: in cloudy months there is no optical scene, so the label is missing even
though Sentinel-1 has data. A yearly label averages the class probabilities over the whole year, so it exists almost
everywhere. Two uses, both off by default:

  * fill the holes of the monthly labels with the yearly label (`fill_from_yearly`): more labelled pixels, but a
    pixel that changed during the year gets a wrong label in the months before or after the change. The monthly
    maps have no ground truth either way; the fill is a training convenience only.
  * score a model only where monthly and yearly labels agree (`agreement_mask`): a cleaner reference, no retraining.

Stored per region and year as <data>/<region>/yearly_label_<year>.npz with the raw Dynamic World class (0..8, 255 = none)
and the mean probability of the winning class in percent, on the same grid as the monthly files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .classes import IGNORE_INDEX, remap_dynamic_world
from .ee_export import LABEL_NODATA, TILE_PX, write_atomic
from .ee_stats import DW_COLLECTION
from .classes import DW_CLASS_NAMES
from .grid import assemble, split_tiles

BANDS = ("label", "conf")


def yearly_label_path(out_dir, key: str, year: int) -> Path:
    return Path(out_dir) / key / f"yearly_label_{year}.npz"


def build_yearly_label_image(ee, geom, year: int):
    """Earth Engine image with bands label (argmax of the year's mean class probability) and conf (percent)."""
    col = ee.ImageCollection(DW_COLLECTION).filterBounds(geom).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    names = list(DW_CLASS_NAMES)
    mean = col.select(names).mean()
    label = mean.toArray().arrayArgmax().arrayGet([0]).rename("label").unmask(LABEL_NODATA)
    conf = mean.reduce(ee.Reducer.max()).multiply(100).rename("conf").unmask(0)
    return ee.Image.cat(label, conf).select(list(BANDS)).toFloat()


def decode_yearly(stack: np.ndarray) -> dict[str, np.ndarray]:
    if stack.shape[0] != len(BANDS):
        raise ValueError(f"expected {len(BANDS)} bands, got {stack.shape[0]}")
    label, conf = stack
    label = np.where(np.isfinite(label), label, LABEL_NODATA)
    return {"label": np.clip(label, 0, 255).astype(np.uint8), "conf": np.clip(np.nan_to_num(conf), 0, 100).astype(np.uint8)}


def export_yearly_label_region(grid, fetch, path: Path, meta: dict, tile_px: int = TILE_PX) -> dict:
    tiles = split_tiles(grid, tile_px)
    arrays = decode_yearly(assemble(grid, tiles, [fetch(t) for t in tiles]))
    write_atomic(Path(path), arrays, {**meta, "epsg": grid.epsg, "x0": grid.x0, "y0": grid.y0, "res": grid.res})
    return {"labelled_share": float((arrays["label"] != LABEL_NODATA).mean())}


def load_yearly_label(out_dir, key: str, year: int, conf_cutoff: float = 0.0) -> np.ndarray:
    """Yearly label in the 6-class scheme (IGNORE_INDEX where missing or below conf_cutoff percent)."""
    f = yearly_label_path(out_dir, key, year)
    if not f.exists():
        raise FileNotFoundError(f"{f} is missing; run scripts/export_yearly_labels.py")
    with np.load(f) as z:
        lab, conf = z["label"], z["conf"]
    out = remap_dynamic_world(lab)
    if conf_cutoff > 0:
        out = np.where(conf >= conf_cutoff, out, IGNORE_INDEX).astype(np.uint8)
    return out


def fill_from_yearly(monthly: np.ndarray, yearly: np.ndarray, present: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Fill ignored monthly labels (T,H,W) with the yearly label (H,W). Months without radar stay ignored.

    Returns (filled labels, share of the labelled pixels that came from the fill)."""
    if monthly.shape[-2:] != yearly.shape:
        raise ValueError(f"grid mismatch: monthly {monthly.shape[-2:]} vs yearly {yearly.shape}")
    holes = monthly == IGNORE_INDEX
    if present is not None:
        holes &= np.asarray(present, bool)[:, None, None]
    out = np.where(holes, yearly[None], monthly).astype(monthly.dtype)
    filled = holes & (yearly[None] != IGNORE_INDEX)
    total = (out != IGNORE_INDEX).sum()
    return out, float(filled.sum() / max(total, 1))


def agreement_mask(monthly: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Labels (T,H,W) kept only where they equal `other` (H,W; yearly label or WorldCover), else ignore. For SCORING."""
    if monthly.shape[-2:] != other.shape:
        raise ValueError(f"grid mismatch: {monthly.shape[-2:]} vs {other.shape}")
    keep = (monthly == other[None]) & (monthly != IGNORE_INDEX)
    return np.where(keep, monthly, IGNORE_INDEX).astype(np.uint8)
