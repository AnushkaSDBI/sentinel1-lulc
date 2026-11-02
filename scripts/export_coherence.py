"""Export Sentinel-1 seasonal coherence (VV, 6/12/24 day baselines) for each region.

    %run scripts/export_coherence.py --project YOUR_ID --out /content/drive/MyDrive/sar-lulc/raw

Writes <out>/<region>/coherence.npz at 100 m, aligned to the region's 10 m grid (one coherence
pixel = 10 x 10 grid pixels). Source: Earth_Big_Data/GLOBAL_SEASONAL_S1/V2019/COHERENCE
(Dec 2019 - Nov 2020, 92.7 m native). Small and quick: about one request per region.
This is proof-of-concept data, see docs/REFERENCES.md.
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
from lulc.ee_export import build_coherence_image, coarse_grid, export_coherence_region, make_fetcher  # noqa: E402
from lulc.grid import grid_for_region  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--out", default="data/raw")
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
    image = build_coherence_image(ee)
    failed = []
    for r in regions:
        path = Path(args.out) / r.key / "coherence.npz"
        if path.exists() and not args.overwrite:
            print(f"{r.key}: exists, skipped")
            continue
        grid = grid_for_region(r)
        try:
            rep = export_coherence_region(grid, make_fetcher(ee, image, coarse_grid(grid)), path, {"region": r.key})
        except Exception as e:
            failed.append(r.key)
            print(f"{r.key}: FAILED {type(e).__name__}: {str(e)[:150]}")
            continue
        print(f"{r.key}: {json.dumps({k: round(v, 3) for k, v in rep.items()})}", flush=True)
    if failed:
        print("FAILED (rerun to retry):", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
