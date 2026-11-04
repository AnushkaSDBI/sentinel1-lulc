"""Orbit direction check for an exported dataset (no Earth Engine call).

    %run scripts/check_orbit.py --data data/raw
    %run scripts/check_orbit.py --data data/raw --data data/raw_2025 --data data/raw_2026

Why: ascending and descending passes look at the ground from opposite sides, so layover, shadow and backscatter differ.
export_dataset.py fixes ONE direction per region (the dominant direction of the config year) for every year. This script
reads the metadata of every exported month file and reports, per region and year:
  * the direction(s) actually used (more than one in a region = mixed geometry = a bug in the export),
  * the median number of scenes per pixel and the share of months with no radar,
so you can see where the fixed direction gives fewer scenes in later years (Sentinel-1B stopped in Dec 2021).
Writes orbit_check.csv next to the first --data folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def scan(data_dirs) -> pd.DataFrame:
    rows = []
    for d in map(Path, data_dirs):
        for f in sorted(d.glob("*/*-*.npz")):
            if not f.stem[:4].isdigit():
                continue
            try:
                with np.load(f) as z:
                    meta = json.loads(str(z["meta"]))
                    vv = z["vv"]
                    n = z["s1_n"][vv > -100] if "s1_n" in z.files else np.array([])
            except Exception as e:                               # unreadable file: report, keep going
                rows.append({"region": f.parent.name, "year": int(f.stem[:4]), "month": int(f.stem[5:7]), "direction": f"ERROR {type(e).__name__}",
                             "scenes_median": np.nan, "has_radar": False, "folder": str(d)})
                continue
            rows.append({"region": f.parent.name, "year": int(meta.get("year", f.stem[:4])), "month": int(meta.get("month", f.stem[5:7])),
                         "direction": meta.get("direction", "unknown"), "scenes_median": float(np.median(n)) if n.size else 0.0,
                         "has_radar": bool(n.size), "folder": str(d)})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    """Per region-year table and the list of problems."""
    ry = (df.groupby(["region", "year"]).agg(direction=("direction", lambda s: "/".join(sorted(set(s)))), months=("month", "count"),
                                              no_radar_months=("has_radar", lambda s: int((~s).sum())),
                                              scenes_median=("scenes_median", "median")).reset_index())
    mixed_in_year = ry[ry.direction.str.contains("/")]
    per_region = df.groupby("region").direction.agg(lambda s: sorted(set(s)))
    changes = per_region[per_region.map(len) > 1]
    low = ry[(ry.scenes_median < 3) | (ry.no_radar_months >= 3)]
    return {"table": ry, "mixed_in_year": mixed_in_year, "direction_changes": changes, "low_coverage": low}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None, help="export folder (repeat for several years)")
    args = ap.parse_args(argv)
    dirs = args.data or ["data/raw"]
    df = scan(dirs)
    if df.empty:
        raise SystemExit(f"no month files found in {dirs}")
    s = summarise(df)
    df.to_csv(Path(dirs[0]) / "orbit_check.csv", index=False)
    dc = df.drop_duplicates(["region", "direction"]).groupby("direction").region.nunique()
    print("Regions per orbit direction:", dc.to_dict())
    print(f"\nRegions with MORE THAN ONE direction across all exported months: {len(s['direction_changes'])}")
    for k, v in s["direction_changes"].items():
        print(f"  {k}: {v}   <- re-export with one direction")
    print(f"\nRegion-years with mixed directions inside one year: {len(s['mixed_in_year'])}")
    print(f"\nRegion-years with a median under 3 scenes per pixel or 3+ months without radar: {len(s['low_coverage'])}")
    if len(s["low_coverage"]):
        print(s["low_coverage"].to_string(index=False))
    print("\nMedian scenes per pixel by year:")
    print(s["table"].groupby("year").scenes_median.median().round(1).to_string())
    print("\nSaved", Path(dirs[0]) / "orbit_check.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
