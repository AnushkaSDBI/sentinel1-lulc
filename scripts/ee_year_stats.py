"""Compare candidate years: Sentinel-1 density and Dynamic World label coverage.

Run it inside a Colab notebook where you already ran ee.Authenticate():

    %run scripts/ee_year_stats.py --project YOUR_PROJECT_ID

Outputs reports/ee_year_stats.csv (every region and month) and prints a summary.
Takes a few minutes. Reads data only; nothing is exported.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.config import load_config  # noqa: E402
from lulc.ee_stats import collect, summarize  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2024])
    ap.add_argument("--regions", nargs="*", help="region keys; default is all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="reports/ee_year_stats.csv")
    args = ap.parse_args(argv)
    if not args.project:
        ap.error("pass --project or set EE_PROJECT")

    import ee

    ee.Initialize(project=args.project)

    _, regions = load_config(args.config)
    if args.regions:
        regions = [r for r in regions if r.key in args.regions]
        if not regions:
            ap.error(f"no region matches {args.regions}")

    df = collect(ee, regions, args.years, args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\nPer year, all regions pooled")
    print(summarize(df).round(2).to_string())
    print("\nPer region and year")
    for key, g in df.groupby("region"):
        print(f"\n{key}")
        print(summarize(g).round(2)[
            ["s1_median_per_month", "s1_min_month", "dw_cov1_mean", "dw_cov1_winter", "dw_cov1_worst_month"]
        ].to_string())
    print(f"\nFull table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
