"""Export the independent reference maps (ESA WorldCover 2021, Esri LULC 2017-2025).

    %run scripts/export_reference.py --project YOUR_ID --out /content/drive/MyDrive/sar-lulc/raw

One small file per region: <out>/<region>/reference.npz with raw class values on the
same grid as the Sentinel-1 data. These maps are for validation only, never training.
The script prints the value shares of each map, so you can see which class scheme the
Esri asset uses (the catalog page is ambiguous). Send me that output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.config import load_config  # noqa: E402
from lulc.ee_export import build_reference_image, export_reference_region, make_fetcher  # noqa: E402
from lulc.grid import grid_for_region  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--year", type=int, help="2021 (default): WorldCover + Esri in reference.npz. Another year (Esri 2017-2025 only): reference_esri_<year>.npz")
    args = ap.parse_args(argv)
    if not args.project:
        ap.error("pass --project or set EE_PROJECT")

    import ee

    ee.Initialize(project=args.project)
    year, regions = load_config(args.config)
    if args.regions:
        regions = [r for r in regions if r.key in args.regions]
    ref_year = args.year or year
    if ref_year == 2021:
        image, fname = build_reference_image(ee, 2021), "reference.npz"
    else:
        from lulc.ee_export import build_esri_image, export_esri_region
        image, fname = build_esri_image(ee, ref_year), f"reference_esri_{ref_year}.npz"
    failed = []
    for r in regions:
        path = Path(args.out) / r.key / fname
        if path.exists() and not args.overwrite:
            print(f"{r.key}: exists, skipped")
            continue
        grid = grid_for_region(r)
        try:
            export = export_reference_region if ref_year == 2021 else export_esri_region
            shares = export(grid, make_fetcher(ee, image, grid), path, {"region": r.key, "year": ref_year})
        except Exception as e:
            failed.append(r.key)
            print(f"{r.key}: FAILED {type(e).__name__}: {str(e)[:150]}")
            continue
        if "wc" in shares:
            print(f"{r.key}: wc={json.dumps(shares['wc'])}")
        print(f"{r.key if 'wc' not in shares else ' ' * len(r.key)}  esri={json.dumps(shares['esri'])}")
    if failed:
        print("FAILED (rerun to retry):", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
