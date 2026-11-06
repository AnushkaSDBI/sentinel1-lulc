"""Copy the small result files of finished runs into the repository, so the report numbers can be traced.

    python scripts/collect_results.py --runs /home/jovyan/vault/lulc/runs /home/jovyan/vault/lulc/runs_multiyear --out results

Copied, per run folder:
  <variant>/seed*/   history.csv, summary.json, model_card.json, config.json, years.json  (never best.pt)
  eval_*/            every .csv and .json (metrics, per-region scores, boundary, change, by-month, ood_reference)
  crf_*/, graph_head_*/, sharpen_*/   the same, for the border experiments
  val_summary.csv

Weights (*.pt) and anything larger than --max-mb are skipped and listed. Run it on Data Lab, then commit results/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SEED_FILES = ("history.csv", "summary.json", "model_card.json", "config.json", "years.json")
EVAL_SUFFIXES = (".csv", ".json")
EVAL_PREFIXES = ("eval_", "crf_", "graph_head_", "sharpen_")


def collect(runs: Path, out: Path, max_mb: float = 5.0) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Copy result files of one runs folder to out/<runs folder name>/. Returns (copied, skipped)."""
    copied, skipped = [], []
    dest_root = out / runs.name

    def take(src: Path) -> None:
        if src.suffix in (".pt", ".pth"):
            skipped.append((src, "weights"))
        elif src.stat().st_size > max_mb * 1e6:
            skipped.append((src, f"larger than {max_mb} MB"))
        else:
            dst = dest_root / src.relative_to(runs)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)

    for seed_dir in sorted(runs.glob("*/seed*")):
        if seed_dir.is_dir() and not seed_dir.parent.name.startswith("eval_"):
            for name in SEED_FILES:
                if (seed_dir / name).exists():
                    take(seed_dir / name)
    for ev in sorted(d for d in runs.iterdir() if d.is_dir() and d.name.startswith(EVAL_PREFIXES)):
        for f in sorted(ev.iterdir()):
            if f.is_file() and f.suffix in EVAL_SUFFIXES:
                take(f)
    if (runs / "val_summary.csv").exists():
        take(runs / "val_summary.csv")
    return copied, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="one or more runs folders")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-mb", type=float, default=5.0)
    args = ap.parse_args(argv)
    total, n = 0, 0
    for r in map(Path, args.runs):
        if not r.is_dir():
            raise SystemExit(f"not a folder: {r}")
        copied, skipped = collect(r, Path(args.out), args.max_mb)
        n += len(copied)
        total += sum(p.stat().st_size for p in copied)
        print(f"{r}: copied {len(copied)} files; skipped {len(skipped)}")
        for s, why in skipped[:10]:
            print(f"   skipped {s} ({why})")
    print(f"Done: {n} files, {total / 1e6:.1f} MB in {args.out}/")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
