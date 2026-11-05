"""Export monthly Sentinel-1 composites and Dynamic World labels on one shared grid.

One Earth Engine image per region and month carries every band (S1 and labels),
and it is requested on the grid from `grid.py`. Because a single request returns
all bands on the same pixel grid, input and label cannot drift against each other.

Stored per region-month (compressed .npz, shape H x W = 2400 x 2400):
    vv, vh    float16  dB, NaN where the month has no Sentinel-1 data
    s1_n      uint8    number of acquisitions behind each pixel
    dw_label  uint8    raw Dynamic World class 0..8, 255 = no label
    dw_conf   uint8    mean probability of the winning class, in percent
    dw_n      uint8    number of Dynamic World observations behind each pixel
Labels stay raw so the class scheme can change without a new export.

Sentinel-1 composite: all IW VV+VH acquisitions of the region's dominant orbit
direction in the month, per-pixel median in dB. Median over time is the first
speckle reduction. One direction only, because ascending and descending views
differ in layover, shadow and backscatter.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from pyproj import Transformer

from .classes import DW_CLASS_NAMES
from .config import Region
from .dates import month_windows
from .ee_stats import DIRECTIONS, DW_COLLECTION, S1_COLLECTION, acquisitions_per_month
from .grid import Grid, Tile, affine_transform, assemble, split_tiles

BANDS = ("VV", "VH", "s1_n", "label", "conf", "dw_n")
S1_NODATA = -9999.0
LABEL_NODATA = 255
TILE_PX = 1200  # 6 float32 bands x 1200^2 = 35 MB, below the 48 MB request limit


# ---------------------------------------------------------------- pure helpers
def dominant_direction(hist: dict[str, int]) -> str:
    """Orbit direction with more acquisitions over the year (ties go to DESCENDING)."""
    per = acquisitions_per_month(hist)
    totals = {d: sum(n for (_, dd), n in per.items() if dd == d) for d in DIRECTIONS}
    if not any(totals.values()):
        raise ValueError("no Sentinel-1 acquisitions in histogram")
    return max(DIRECTIONS, key=lambda d: (totals[d], d == "DESCENDING"))


def decode(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Turn the (6, H, W) float32 band stack into the stored arrays."""
    if stack.shape[0] != len(BANDS):
        raise ValueError(f"expected {len(BANDS)} bands, got {stack.shape[0]}")
    vv, vh, s1_n, label, conf, dw_n = stack
    out = {}
    for name, band in (("vv", vv), ("vh", vh)):
        arr = band.astype(np.float32)
        arr[arr <= S1_NODATA + 1] = np.nan
        out[name] = arr.astype(np.float16)
    out["s1_n"] = np.clip(np.nan_to_num(s1_n), 0, 255).astype(np.uint8)
    out["dw_label"] = np.where(np.isnan(label), LABEL_NODATA, np.clip(label, 0, 255)).astype(np.uint8)
    out["dw_conf"] = np.clip(np.rint(np.nan_to_num(conf) * 100), 0, 100).astype(np.uint8)
    out["dw_n"] = np.clip(np.nan_to_num(dw_n), 0, 255).astype(np.uint8)
    return out


def quality_report(arrays: dict[str, np.ndarray]) -> dict:
    """Numbers that show whether a region-month is usable."""
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"arrays have different shapes: {shapes}")
    vv_ok = ~np.isnan(arrays["vv"].astype(np.float32))
    label_ok = arrays["dw_label"] != LABEL_NODATA
    both = vv_ok & label_ok
    return {
        "s1_valid": float(vv_ok.mean()),
        "label_valid": float(label_ok.mean()),
        "both_valid": float(both.mean()),
        "s1_scenes_median": float(np.median(arrays["s1_n"][vv_ok])) if vv_ok.any() else 0.0,
        "label_conf_mean": float(arrays["dw_conf"][label_ok].mean()) if label_ok.any() else 0.0,
    }


def check_grid_probe(grid: Grid, probe_lonlat: np.ndarray, tile: Tile, tol_deg: float = 2e-5) -> float:
    """Compare Earth Engine's pixel-centre lon/lat with our own computation.

    probe_lonlat has shape (2, h, w): lon and lat of every pixel of `tile` as
    reported by ee.Image.pixelLonLat(). Returns the largest deviation in degrees
    and raises if it exceeds tol_deg (about 2 m).
    """
    _, h, w = probe_lonlat.shape
    to_ll = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    cols = tile.x0 + (np.arange(w) + 0.5) * grid.res
    rows = tile.y0 - (np.arange(h) + 0.5) * grid.res
    xx, yy = np.meshgrid(cols, rows)
    lon, lat = to_ll.transform(xx, yy)
    dev = max(np.abs(probe_lonlat[0] - lon).max(), np.abs(probe_lonlat[1] - lat).max())
    if dev > tol_deg:
        raise RuntimeError(f"grid mismatch: Earth Engine pixel centres differ by {dev:.2e} deg")
    return float(dev)


_TRANSIENT = ("429", "500", "502", "503", "504", "too many", "timed out", "timeout",
              "deadline", "internal error", "unavailable", "connection", "quota")


def is_transient(exc: Exception) -> bool:
    """True for rate limits, timeouts and server hiccups worth retrying."""
    return any(t in str(exc).lower() for t in _TRANSIENT) or isinstance(exc, (ConnectionError, TimeoutError))


def month_path(out_dir: Path, region_key: str, year: int, month: int) -> Path:
    return Path(out_dir) / region_key / f"{year}-{month:02d}.npz"


def write_atomic(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """Write npz via a temp file so an interrupted run never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def export_region_month(
    grid: Grid,
    fetch: Callable[[Tile], np.ndarray],
    path: Path,
    meta: dict,
    tile_px: int = TILE_PX,
    workers: int = 2,
    overwrite: bool = False,
) -> dict | None:
    """Fetch all tiles, assemble, check and save one region-month.

    `fetch(tile)` must return a (6, h, w) float32 array. Returns the quality
    report, or None if the file already exists (resumable runs).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        return None
    tiles = split_tiles(grid, tile_px)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch, tiles))
    stack = assemble(grid, tiles, parts)
    arrays = decode(stack)
    report = quality_report(arrays)
    write_atomic(path, arrays, {**meta, "quality": report, "epsg": grid.epsg, "x0": grid.x0, "y0": grid.y0, "res": grid.res})
    return report


# ------------------------------------------------------------ Earth Engine calls
@dataclass(frozen=True)
class CompositeSettings:
    """How the monthly Sentinel-1 composite is built. Stored in every file's meta.

    method       'median_db' (default): per-pixel median of the dB values.
                 'mean_linear': mean in linear power, then back to dB (classic
                 temporal multilooking).
    angle_range  None (default) keeps all pixels. (lo, hi) masks pixels whose
                 incidence angle is outside the range, which removes the noisy
                 swath edges. The EE 'angle' band is a coarse (about 20 km)
                 interpolated grid, so the mask edge is smooth.
    """

    method: str = "median_db"
    angle_range: tuple[float, float] | None = None

    def __post_init__(self):
        if self.method not in ("median_db", "mean_linear"):
            raise ValueError(f"unknown composite method {self.method!r}")
        if self.angle_range is not None and not (0 <= self.angle_range[0] < self.angle_range[1] <= 90):
            raise ValueError(f"bad angle_range {self.angle_range}")

    def as_dict(self) -> dict:
        return {"method": self.method,
                "angle_range": list(self.angle_range) if self.angle_range else None}


def settings_conflict(path: Path, settings: CompositeSettings) -> bool:
    """True if an existing file was written with different composite settings.

    Files from before settings were recorded count as the defaults, which is what
    they were built with. This stops a run from silently mixing two recipes.
    """
    path = Path(path)
    if not path.exists():
        return False
    with np.load(path) as z:
        old = json.loads(str(z["meta"])).get("composite") or CompositeSettings().as_dict()
    return old != settings.as_dict()


def build_image(ee, geom, year: int, month: int, direction: str,
                settings: CompositeSettings = CompositeSettings()):
    """One image with the six BANDS for a month, ready to be sampled on any grid."""
    ((_, first, nxt),) = month_windows(f"{year}-{month:02d}", 1)
    start, end = first.isoformat(), nxt.isoformat()
    s1_col = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", direction))
    )
    if settings.angle_range:
        lo, hi = settings.angle_range
        s1_col = s1_col.map(
            lambda im: im.updateMask(im.select("angle").gt(lo).And(im.select("angle").lt(hi)))
        )
    s1_col = s1_col.select(["VV", "VH"])

    def s1_composite(c):
        if settings.method == "mean_linear":
            lin = c.map(lambda im: ee.Image(10).pow(im.divide(10)))
            return lin.mean().log10().multiply(10)
        return c.median()

    # An empty collection reduces to an image with 0 bands, which breaks later
    # steps. Every reduction therefore falls back to a fully masked image with
    # the right band names, so a month without data becomes "no data" pixels.
    def safe(col, fn, names):
        empty = ee.Image.constant([0] * len(names)).rename(names).updateMask(0)
        return ee.Image(ee.Algorithms.If(col.size().gt(0), fn(col), empty))

    s1 = safe(s1_col, s1_composite, ["VV", "VH"]).unmask(S1_NODATA)
    s1_n = safe(s1_col, lambda c: c.select("VV").count(), ["s1_n"]).rename("s1_n").unmask(0)

    dw_col = ee.ImageCollection(DW_COLLECTION).filterBounds(geom).filterDate(start, end)
    names = list(DW_CLASS_NAMES)
    mean = safe(dw_col, lambda c: c.select(names).mean(), names)
    label = mean.toArray().arrayArgmax().arrayGet([0]).rename("label").unmask(LABEL_NODATA)
    conf = mean.reduce(ee.Reducer.max()).rename("conf").unmask(0)
    dw_n = safe(dw_col, lambda c: c.select("label").count(), ["dw_n"]).rename("dw_n").unmask(0)

    return ee.Image.cat(s1, s1_n, label, conf, dw_n).select(list(BANDS)).toFloat()


def make_fetcher(ee, image, grid: Grid, retries: int = 4) -> Callable[[Tile], np.ndarray]:
    """Return fetch(tile) -> (bands, h, w) float32 using ee.data.computePixels."""

    def fetch(tile: Tile) -> np.ndarray:
        request = {
            "expression": image,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": tile.width, "height": tile.height},
                "affineTransform": affine_transform(grid, tile),
                "crsCode": grid.crs,
            },
        }
        for attempt in range(retries):
            try:
                data = ee.data.computePixels(request)
                names = data.dtype.names
                return np.stack([data[n].astype(np.float32) for n in names])
            except Exception as e:
                # Retry only transient trouble. A bad request will fail every time.
                if attempt == retries - 1 or not is_transient(e):
                    raise
                time.sleep(2 ** attempt * 3)

    return fetch


def verify_grid_with_ee(ee, grid: Grid, size: int = 8) -> float:
    """Ask Earth Engine for pixel-centre lon/lat of a small block and compare with ours."""
    tile = Tile(row=0, col=0, height=size, width=size, x0=grid.x0, y0=grid.y0)
    probe = make_fetcher(ee, ee.Image.pixelLonLat(), grid)(tile)
    return check_grid_probe(grid, probe, tile)


# ------------------------------------------------- independent reference maps
# Annual maps used only for validation, never for training:
#   wc    ESA WorldCover v200 (2021)
#   esri  Esri / Impact Observatory 10 m annual LULC (community catalog asset,
#         CC BY 4.0, trained on human-labelled pixels, so independent of Dynamic World)
WORLDCOVER = "ESA/WorldCover/v200"
ESRI_LULC = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
REF_BANDS = ("wc", "esri")


def build_esri_image(ee, year: int):
    """Esri / Impact Observatory 10 m annual LULC for `year` (available ~2017-2023), as a single 'esri' band; no-data 0.
    This is the independent check for years other than 2021 (WorldCover only exists for 2020-2021)."""
    return (ee.ImageCollection(ESRI_LULC).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic().select([0]).rename("esri").unmask(0).toFloat())


def export_esri_region(grid: Grid, fetch, path: Path, meta: dict, tile_px: int = TILE_PX) -> dict:
    """Fetch the single Esri band for a region and save it (esri-only reference for a non-2021 year)."""
    tiles = split_tiles(grid, tile_px)
    stack = assemble(grid, tiles, [fetch(t) for t in tiles])
    arr = {"esri": np.clip(np.nan_to_num(stack[0]), 0, 255).astype(np.uint8)}
    write_atomic(Path(path), arr, {**meta, "esri_only": True, "epsg": grid.epsg, "x0": grid.x0, "y0": grid.y0, "res": grid.res})
    return {"esri": value_shares(arr["esri"])}


def build_reference_image(ee, year: int):
    """WorldCover (only exists for 2021) and Esri LULC of `year`, no-data = 0."""
    if year != 2021:
        raise ValueError("WorldCover v200 is the 2021 map; use year 2021")
    wc = ee.ImageCollection(WORLDCOVER).first().select("Map").rename("wc")
    esri = (
        ee.ImageCollection(ESRI_LULC)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select([0])  # band name differs between catalog versions, so select by position
        .rename("esri")
    )
    return ee.Image.cat(wc.unmask(0), esri.unmask(0)).select(list(REF_BANDS)).toFloat()


def decode_reference(stack: np.ndarray) -> dict[str, np.ndarray]:
    if stack.shape[0] != len(REF_BANDS):
        raise ValueError(f"expected {len(REF_BANDS)} bands, got {stack.shape[0]}")
    return {n: np.clip(np.nan_to_num(b), 0, 255).astype(np.uint8) for n, b in zip(REF_BANDS, stack)}


def value_shares(arr: np.ndarray) -> dict[int, float]:
    """Share of pixels per raw value, to check which class scheme a map uses."""
    vals, counts = np.unique(arr, return_counts=True)
    return {int(v): round(float(c) / arr.size, 4) for v, c in zip(vals, counts)}


def export_reference_region(grid: Grid, fetch: Callable[[Tile], np.ndarray], path: Path,
                            meta: dict, tile_px: int = TILE_PX) -> dict:
    """Fetch WorldCover and Esri for a region, save them, return value shares per map."""
    tiles = split_tiles(grid, tile_px)
    arrays = decode_reference(assemble(grid, tiles, [fetch(t) for t in tiles]))
    write_atomic(Path(path), arrays, {**meta, "epsg": grid.epsg, "x0": grid.x0, "y0": grid.y0, "res": grid.res})
    return {k: value_shares(v) for k, v in arrays.items()}


# ------------------------------------------------------------ coherence (optional)
# Global seasonal Sentinel-1 coherence (Kellndorfer et al. 2022, CC BY 4.0). One climatology
# for Dec 2019 - Nov 2020, 92.7 m pixels, four seasons. It is a static prior per place, not
# a monthly series, and its year does not match the 2021 labels. Proof-of-concept data only.
COHERENCE = "Earth_Big_Data/GLOBAL_SEASONAL_S1/V2019/COHERENCE"
COH_SEASONS = ("winter", "spring", "summer", "fall")
COH_BASELINES = (6, 12, 24)  # days between the two acquisitions
COH_BANDS = tuple(f"{s}_vv_COH{d:02d}" for s in COH_SEASONS for d in COH_BASELINES)
COH_FACTOR = 10  # coherence is exported at 100 m, exactly 10 pixels of our 10 m grid


def coarse_grid(grid: Grid, factor: int = COH_FACTOR) -> Grid:
    """Same origin as `grid`, pixels `factor` times larger. Covers at least the whole grid."""
    return Grid(epsg=grid.epsg, x0=grid.x0, y0=grid.y0,
                size_px=-(-grid.size_px // factor), res=grid.res * factor)


def build_coherence_image(ee):
    """VV coherence bands on any grid; no-data becomes -1."""
    img = ee.ImageCollection(COHERENCE).select(list(COH_BANDS)).mosaic()
    return img.resample("bilinear").unmask(-1).toFloat()


def decode_coherence(stack: np.ndarray) -> dict[str, np.ndarray]:
    if stack.shape[0] != len(COH_BANDS):
        raise ValueError(f"expected {len(COH_BANDS)} bands, got {stack.shape[0]}")
    arr = stack.astype(np.float32)
    arr[arr < 0] = np.nan
    return {"coherence": arr.astype(np.float16), "bands": np.array(COH_BANDS)}


def export_coherence_region(grid: Grid, fetch: Callable[[Tile], np.ndarray], path: Path, meta: dict) -> dict:
    """Fetch coherence on the 100 m grid aligned with `grid` and save it. Returns shares of valid pixels."""
    cg = coarse_grid(grid)
    tiles = split_tiles(cg, TILE_PX)
    arrays = decode_coherence(assemble(cg, tiles, [fetch(t) for t in tiles]))
    write_atomic(Path(path), arrays, {**meta, "epsg": cg.epsg, "x0": cg.x0, "y0": cg.y0, "res": cg.res,
                                       "factor": COH_FACTOR, "source_pixel_m": 92.7})
    c = arrays["coherence"].astype(np.float32)
    return {"valid": float(np.isfinite(c).mean()), "mean_vv_coh06_summer": float(np.nanmean(c[COH_BANDS.index("summer_vv_COH06")]))}


def region_direction(ee, region: Region, year: int) -> str:
    from .ee_stats import region_geometry, s1_histogram

    return dominant_direction(s1_histogram(ee, region_geometry(ee, region), year))
