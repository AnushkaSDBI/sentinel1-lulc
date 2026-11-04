"""Export the yearly Dynamic World label ("yearly fill") for every region and year.

    %run scripts/export_yearly_labels.py --project YOUR_ID --out data/raw --years 2021 2022 2023 2024
    %run scripts/export_yearly_labels.py --project YOUR_ID --out data/raw_2025 --years 2025

One small file per region and year: <out>/<region>/yearly_label_<year>.npz (resumable: existing files are skipped).
Same grid as the monthly files. Used by lulc.yearly_labels (fill holes in monthly labels, or score only where
monthly and yearly labels agree). Nothing in training reads these files unless you ask for it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.config import load_config  # noqa: E402
from lulc.ee_export import make_fetcher, verify_grid_with_ee  # noqa: E402
from lulc.ee_stats import region_geometry  # noqa: E402
from lulc.grid import grid_for_region  # noqa: E402
from lulc.yearly_labels import build_yearly_label_image, export_yearly_label_region, yearly_label_path  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if not args.project:
        ap.error("pass --project or set EE_PROJECT")
    import ee

    ee.Initialize(project=args.project)
    _, regions = load_config(args.config)
    if args.regions:
        regions = [r for r in regions if r.key in args.regions]
    failed = []
    for r in regions:
        grid = grid_for_region(r)
        geom = region_geometry(ee, r)
        checked = False
        for y in args.years:
            path = yearly_label_path(args.out, r.key, y)
            if path.exists() and not args.overwrite:
                continue
            try:
                if not checked:
                    verify_grid_with_ee(ee, grid)
                    checked = True
                rep = export_yearly_label_region(grid, make_fetcher(ee, build_yearly_label_image(ee, geom, y), grid), path,
                                                 {"region": r.key, "year": y})
            except Exception as e:
                failed.append((r.key, y))
                print(f"{r.key} {y}: FAILED {type(e).__name__}: {str(e)[:150]}")
                continue
            print(f"{r.key} {y}: labelled share {rep['labelled_share']:.3f}", flush=True)
    if failed:
        print("FAILED (rerun to retry):", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
