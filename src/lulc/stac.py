"""Planetary Computer STAC helpers: Sentinel-1 RTC scene counts and bbox coverage."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import date
from typing import TypeVar

import pandas as pd
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .config import BBox, Region
from .dates import month_windows, stac_interval

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-rtc"
WORLDCOVER_COLLECTION = "esa-worldcover"

T = TypeVar("T")


def open_catalog():
    import pystac_client

    return pystac_client.Client.open(STAC_URL)


def _retry(fn: Callable[[], T], tries: int = 3, base_delay: float = 2.0) -> T:
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(base_delay * 2**attempt)
    raise RuntimeError("unreachable")


def bbox_coverage(geometries: Iterable[dict], bbox: BBox) -> float:
    """Fraction (0..1) of bbox area covered by the union of GeoJSON geometries."""
    target = box(*bbox)
    geoms = [shape(g) for g in geometries if g]
    if not geoms:
        return 0.0
    covered = unary_union(geoms).intersection(target)
    return float(covered.area / target.area)


def scene_stats(catalog, bbox: BBox, first: date, next_first: date) -> dict:
    """Scene count, orbit split and bbox coverage for one month."""
    search = catalog.search(
        collections=[S1_COLLECTION],
        bbox=bbox,
        datetime=stac_interval(first, next_first),
    )
    items = _retry(lambda: list(search.items()))
    orbits = Counter(i.properties.get("sat:orbit_state", "unknown") for i in items)
    return {
        "n_scenes": len(items),
        "ascending": orbits.get("ascending", 0),
        "descending": orbits.get("descending", 0),
        "coverage": bbox_coverage((i.geometry for i in items), bbox),
    }


def worldcover_tiles(catalog, bbox: BBox, year: int) -> int:
    """Number of WorldCover items for the given year that intersect bbox."""
    search = catalog.search(
        collections=[WORLDCOVER_COLLECTION],
        bbox=bbox,
        datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
    )
    return len(_retry(lambda: list(search.items())))


def check_all(
    catalog,
    regions: list[Region],
    year: int,
    min_scenes: int = 2,
    min_coverage: float = 0.99,
) -> pd.DataFrame:
    """One row per region and month with scene stats and an ok flag."""
    rows = []
    for r in regions:
        wc = worldcover_tiles(catalog, r.bbox, year)
        for label, first, next_first in month_windows(f"{year}-01", 12):
            s = scene_stats(catalog, r.bbox, first, next_first)
            rows.append(
                {
                    "region": r.key,
                    "split": r.split,
                    "month": label,
                    **s,
                    "worldcover_tiles": wc,
                    "ok": s["n_scenes"] >= min_scenes
                    and s["coverage"] >= min_coverage
                    and wc > 0,
                }
            )
    return pd.DataFrame(rows)
