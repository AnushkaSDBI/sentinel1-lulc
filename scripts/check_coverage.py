"""Check Sentinel-1 RTC and WorldCover availability for every region and month.

Usage:
    python scripts/check_coverage.py --config configs/regions.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lulc.config import load_config  # noqa: E402
from lulc.stac import check_all, open_catalog  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/regions.yaml")
    ap.add_argument("--min-scenes", type=int, default=2)
    ap.add_argument("--min-coverage", type=float, default=0.99)
    ap.add_argument("--out", default="reports/coverage.csv")
    args = ap.parse_args()

    year, regions = load_config(args.config)
    df = check_all(open_catalog(), regions, year, args.min_scenes, args.min_coverage)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(df.pivot(index="region", columns="month", values="n_scenes").to_string())
    bad = df[~df["ok"]]
    if bad.empty:
        print("\nAll regions and months pass.")
        return 0
    print(f"\n{len(bad)} region-months fail (scenes < {args.min_scenes}, "
          f"coverage < {args.min_coverage:.0%}, or no WorldCover):")
    print(bad[["region", "month", "n_scenes", "coverage", "worldcover_tiles"]].to_string(index=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
