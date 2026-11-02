"""Bundle the chosen variant's seeds into ONE weights file for the hand-in and the prototype.

    python scripts/package_model.py --runs RUNS --variant V2_window3 --out weights

Writes <out>/final_model.pt (all seeds, model cards and normalisation inside) and copies ood_reference.json from the
evaluation folder if there is one. The prototype loads final_model.pt when it is present.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--eval-tag", default="", help="suffix of the eval_<variant> folder that holds ood_reference.json")
    a = ap.parse_args(argv)
    from lulc.train import FINAL_NAME, save_final

    seeds = sorted(p.parent for p in Path(a.runs).glob(f"{a.variant}/seed*/best.pt"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    card = save_final(seeds, out / FINAL_NAME)
    ood = Path(a.runs) / f"eval_{a.variant}{a.eval_tag}" / "ood_reference.json"
    if ood.exists():
        shutil.copy(ood, out / "ood_reference.json")
    size = (out / FINAL_NAME).stat().st_size / 1e6
    print(f"{len(seeds)} seeds of {a.variant} -> {out / FINAL_NAME} ({size:.0f} MB), channels {card['channels']}, "
          f"ood_reference.json {'copied' if ood.exists() else 'NOT found (run evaluate.py first)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
