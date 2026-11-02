"""Export monthly Sentinel-1 composites and Dynamic World labels for all regions.

    # Plan only, no Earth Engine call, prints grids and request counts
    python scripts/export_dataset.py --dry-run

    # Smallest real test: one region, one month
    %run scripts/export_dataset.py --project YOUR_ID --regions heidelberg --months 7

    # Everything (resumable: existing files are skipped)
    %run scripts/export_dataset.py --project YOUR_ID

Files land in data/raw/<region>/<year>-<month>.npz. Re-running continues where
it stopped. Before the first download of each region, the script checks that
Earth Engine's pixel positions match our grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.config import load_config  # noqa: E402
from lulc.ee_export import (  # noqa: E402
    TILE_PX, CompositeSettings, build_image, settings_conflict, export_region_month, make_fetcher, month_path,
    region_direction, verify_grid_with_ee,
)
from lulc.ee_stats import region_geometry  # noqa: E402
from lulc.grid import grid_for_region, split_tiles  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--regions", nargs="*", help="region keys; default is all")
    ap.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)))
    ap.add_argument("--workers", type=int, default=4, help="parallel tile requests")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--year", type=int, default=None,
                    help="export another year (extra training years). Default: the config year. "
                         "Orbit direction is always taken from the config year, so geometry matches.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--composite", choices=["median_db", "mean_linear"], default="median_db",
                    help="monthly composite: median of dB (default) or mean in linear power")
    ap.add_argument("--angle-mask", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="mask pixels with incidence angle outside LO..HI degrees, e.g. 31 45")
    args = ap.parse_args(argv)
    settings = CompositeSettings(args.composite, tuple(args.angle_mask) if args.angle_mask else None)

    config_year, regions = load_config(args.config)
    year = args.year or config_year
    if args.regions:
        regions = [r for r in regions if r.key in args.regions]
        if not regions:
            ap.error(f"no region matches {args.regions}")
    out = Path(args.out)

    if not args.overwrite:
        clash = [f"{r.key} {year}-{m:02d}" for r in regions for m in args.months
                 if settings_conflict(month_path(out, r.key, year, m), settings)]
        if clash:
            ap.error(f"{len(clash)} existing files were built with different composite settings "
                     f"(first: {clash[0]}). Use --overwrite, another --out, or the same settings.")

    total = len(regions) * len(args.months)
    n_requests = sum(len(split_tiles(grid_for_region(r), TILE_PX)) for r in regions) * len(args.months)
    if args.dry_run:
        for r in regions:
            g = grid_for_region(r)
            todo = [m for m in args.months if not month_path(out, r.key, year, m).exists()]
            print(f"{r.key:10s} {r.split:10s} EPSG:{g.epsg} origin=({g.x0:.0f},{g.y0:.0f}) "
                  f"{g.size_px}px, {len(todo)}/{len(args.months)} months to fetch")
        print(f"\n{total} region-months, {n_requests} tile requests (upper bound)")
        return 0

    if not args.project:
        ap.error("pass --project or set EE_PROJECT")
    import ee

    ee.Initialize(project=args.project)

    pending = sum(1 for r in regions for m in args.months
                  if args.overwrite or not month_path(out, r.key, year, m).exists())
    t_start = time.time()
    done = 0
    failed: list[tuple[str, int, str]] = []
    for r in regions:
        grid = grid_for_region(r)
        todo = [m for m in args.months if args.overwrite or not month_path(out, r.key, year, m).exists()]
        if not todo:
            print(f"{r.key}: all requested months exist")
            continue
        dev = verify_grid_with_ee(ee, grid)
        direction = region_direction(ee, r, config_year)
        print(f"{r.key}: grid ok (max pixel-centre deviation {dev:.1e} deg), orbit {direction}")
        geom = region_geometry(ee, r)
        for m in todo:
            t0 = time.time()
            try:
                image = build_image(ee, geom, year, m, direction, settings)
                fetch = make_fetcher(ee, image, grid)
                meta = {"region": r.key, "year": year, "month": m, "direction": direction,
                        "composite": settings.as_dict()}
                rep = export_region_month(grid, fetch, month_path(out, r.key, year, m), meta,
                                          workers=args.workers, overwrite=args.overwrite)
            except Exception as e:  # keep going; failed months are listed at the end
                failed.append((r.key, m, f"{type(e).__name__}: {str(e)[:150]}"))
                print(f"  {year}-{m:02d} FAILED {type(e).__name__}: {str(e)[:150]}")
                continue
            done += 1
            avg = (time.time() - t_start) / done
            eta_min = avg * (pending - done - len(failed)) / 60
            print(f"  {year}-{m:02d} {time.time() - t0:.0f}s, left ~{eta_min:.0f} min | "
                  f"{json.dumps({k: round(v, 3) for k, v in rep.items()})}", flush=True)
    print(f"\nExported {done} region-months to {out}/")
    if failed:
        print(f"{len(failed)} FAILED (rerun the same command to retry them):")
        for key, m, msg in failed:
            print(f"  {key} {year}-{m:02d}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
