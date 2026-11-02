"""Earth Engine data-availability statistics, used to choose the study year.

Two questions, answered per region, year and month:
  1. How many Sentinel-1 acquisitions (IW, VV+VH) exist, per orbit direction?
  2. How much of the region has a Dynamic World label, and from how many images?

The Earth Engine calls are thin and need a registered project. Everything that
turns their raw answers into tables is plain Python and is unit-tested.

Counting rule: scene products are slices of one pass, and a region can touch two
slices of the same pass. So we count distinct (date, direction) pairs, not
products.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .config import Region

S1_COLLECTION = "COPERNICUS/S1_GRD"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DIRECTIONS = ("ASCENDING", "DESCENDING")


# ---------------------------------------------------------------- pure helpers
def acquisitions_per_month(hist: dict[str, int]) -> dict[tuple[int, str], int]:
    """Turn {'YYYY-MM-DD|DIRECTION': n_products} into {(month, DIRECTION): n_acquisitions}."""
    days: dict[tuple[int, str], set[str]] = defaultdict(set)
    for key in hist:
        day, direction = key.split("|")
        days[(int(day[5:7]), direction)].add(day)
    return {k: len(v) for k, v in days.items()}


def scene_table(region: str, year: int, hist: dict[str, int]) -> pd.DataFrame:
    """One row per month with acquisitions per direction."""
    per = acquisitions_per_month(hist)
    rows = []
    for m in range(1, 13):
        asc, desc = per.get((m, "ASCENDING"), 0), per.get((m, "DESCENDING"), 0)
        rows.append(
            {"region": region, "year": year, "month": m,
             "s1_asc": asc, "s1_desc": desc, "s1_best": max(asc, desc)}
        )
    return pd.DataFrame(rows)


def dw_table(region: str, year: int, per_month: list[dict]) -> pd.DataFrame:
    """One row per month from Dynamic World coverage answers (missing months become NaN)."""
    by_month = {d["month"]: d for d in per_month if d}
    rows = []
    for m in range(1, 13):
        d = by_month.get(m, {})
        rows.append(
            {"region": region, "year": year, "month": m,
             "dw_images": d.get("images", np.nan),
             "dw_cov1": d.get("cov1", np.nan),
             "dw_cov2": d.get("cov2", np.nan),
             "dw_mean_obs": d.get("n_obs", np.nan)}
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, winter=(11, 12, 1, 2)) -> pd.DataFrame:
    """One row per year: Sentinel-1 density and Dynamic World label coverage."""
    out = []
    for year, g in df.groupby("year"):
        w = g[g["month"].isin(winter)]
        out.append(
            {
                "year": year,
                "s1_median_per_month": g["s1_best"].median(),
                "s1_min_month": g["s1_best"].min(),
                "dw_cov1_mean": g["dw_cov1"].mean(),
                "dw_cov1_winter": w["dw_cov1"].mean(),
                "dw_cov1_worst_month": g["dw_cov1"].min(),
                "dw_cov2_mean": g["dw_cov2"].mean(),
                "region_months_cov1_below_80pct": int((g["dw_cov1"] < 0.8).sum()),
                "region_months": len(g),
            }
        )
    return pd.DataFrame(out).set_index("year")


# ------------------------------------------------------------ Earth Engine calls
def region_geometry(ee, region: Region):
    w, s, e, n = region.bbox
    return ee.Geometry.Rectangle([w, s, e, n])


def s1_histogram(ee, geom, year: int) -> dict[str, int]:
    """Server-side count of Sentinel-1 IW VV+VH products by 'date|direction'."""
    col = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(geom)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    def tag(img):
        key = (
            img.date().format("YYYY-MM-dd")
            .cat("|")
            .cat(ee.String(img.get("orbitProperties_pass")))
        )
        return img.set("acq_key", key)

    return col.map(tag).aggregate_histogram("acq_key").getInfo() or {}


def dw_month(ee, geom, year: int, month: int, scale: int = 100) -> dict:
    """Dynamic World coverage of one month over the region, computed at coarse scale."""
    start = ee.Date.fromYMD(year, month, 1)
    col = (
        ee.ImageCollection(DW_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, start.advance(1, "month"))
        .select("label")
    )
    n_images = col.size().getInfo()
    if n_images == 0:
        return {"month": month, "images": 0, "cov1": 0.0, "cov2": 0.0, "n_obs": 0.0}
    cnt = col.count()
    stats = (
        ee.Image.cat(cnt.gt(0).rename("cov1"), cnt.gte(2).rename("cov2"), cnt.rename("n_obs"))
        .reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=int(1e9))
        .getInfo()
    )
    return {"month": month, "images": n_images, **stats}


def collect(ee, regions: list[Region], years: list[int], workers: int = 4) -> pd.DataFrame:
    """Full table: one row per region, year and month."""

    def one(job):
        region, year = job
        geom = region_geometry(ee, region)
        scenes = scene_table(region.key, year, s1_histogram(ee, geom, year))
        months = []
        for m in range(1, 13):
            try:
                months.append(dw_month(ee, geom, year, m))
            except Exception as e:  # keep going, the month shows as NaN
                print(f"  {region.key} {year}-{m:02d}: Dynamic World query failed ({type(e).__name__})")
                months.append({})
        return scenes.merge(dw_table(region.key, year, months), on=["region", "year", "month"])

    jobs = [(r, y) for r in regions for y in years]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = []
        for i, frame in enumerate(pool.map(one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {frame['region'].iloc[0]} {frame['year'].iloc[0]} done")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)
