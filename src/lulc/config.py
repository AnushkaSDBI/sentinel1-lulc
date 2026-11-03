"""Region config with spatial-leakage checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test", "test_ood", "case_study")
KM_PER_DEG_LAT = 111.32

BBox = tuple[float, float, float, float]  # west, south, east, north


def bbox_from_center(lat: float, lon: float, half_size_km: float) -> BBox:
    """Square box of +-half_size_km around a point, in degrees."""
    dlat = half_size_km / KM_PER_DEG_LAT
    dlon = half_size_km / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def bboxes_intersect(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    lat: float
    lon: float
    split: str
    half_size_km: float
    climate: str = ""          # coarse climate group, used to report scores per climate

    @property
    def bbox(self) -> BBox:
        return bbox_from_center(self.lat, self.lon, self.half_size_km)


def validate_no_overlap(regions: list[Region]) -> None:
    """Raise if two regions in different splits overlap (would leak labels)."""
    for a, b in combinations(regions, 2):
        if a.split != b.split and bboxes_intersect(a.bbox, b.bbox):
            raise ValueError(
                f"Regions overlap across splits: {a.key} ({a.split}) and {b.key} ({b.split})"
            )


def load_config(path: str | Path) -> tuple[int, list[Region]]:
    """Return (label_year, regions) from a YAML file."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    default_half = float(cfg.get("half_size_km", 12))
    climate_of = {k: g for g, keys in (cfg.get("climate_groups") or {}).items() for k in keys}
    unknown = set(climate_of) - set(cfg["regions"])
    if unknown:
        raise ValueError(f"climate_groups lists unknown regions: {sorted(unknown)}")
    regions = []
    for key, r in cfg["regions"].items():
        if r["split"] not in SPLITS:
            raise ValueError(f"{key}: unknown split {r['split']!r}, expected one of {SPLITS}")
        regions.append(
            Region(
                key=key,
                name=r["name"],
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                split=r["split"],
                half_size_km=float(r.get("half_size_km", default_half)),
                climate=climate_of.get(key, ""),
            )
        )
    validate_no_overlap(regions)
    return int(cfg["year"]), regions


def load_years(path: str | Path) -> dict[str, list[int]]:
    """Years used per role: {'train': [...], 'val': [...], 'test': [...]}.

    Read from the optional `years:` block of the YAML. Without it, every role uses the
    single label year, which is the old one-year behaviour. A year may not be both a
    training year and a test year: that would leak the imagery of a test year into training.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = int(cfg["year"])
    y = cfg.get("years") or {}
    out = {k: [int(v) for v in y.get(k, [base])] for k in ("train", "val", "test")}
    clash = set(out["train"]) & set(out["test"])
    if clash and (cfg.get("years") or {}).get("test"):
        raise ValueError(f"years in both train and test: {sorted(clash)}")
    return out
