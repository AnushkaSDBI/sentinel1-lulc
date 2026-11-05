"""Do test-time augmentation, Gaussian tile stitching or a retrained variant sharpen borders without hurting mIoU?

    %run scripts/sharpen_eval.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --year 2024 --settings tta
    %run scripts/sharpen_eval.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --year 2024 --settings none --compare-variant V17_window3_fullres

Every row is scored on the same regions and months against Dynamic World, pooled: mIoU, boundary F1 (1 px tolerance),
predicted / reference border pixels and per-class IoU. The first row, "none", is the --variant model as it runs today.

Decision rule (set before the results, the same as the CRF and the graph head): use a setting only if, on VALIDATION
regions, boundary F1 rises by at least 0.02, pooled mIoU does not fall by more than 0.005, and no class IoU falls by more
than 0.05. Then confirm once on the test regions. Picking the best of several rows on validation is mildly optimistic.
Four months (Jan, Apr, Jul, Oct) by default, as in crf_eval.py, so the rows can be compared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

SHARPEN_MODES = {"none": {}, "tta": {"tta": True}, "gaussian": {"blend": "gaussian"}, "tta+gaussian": {"tta": True, "blend": "gaussian"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--settings", nargs="*", choices=list(SHARPEN_MODES), default=["tta", "gaussian"], help="the baseline row 'none' is always scored")
    ap.add_argument("--compare-variant", help="also score this trained variant with plain inference, against the baseline row")
    ap.add_argument("--compare-runs", help="folder of --compare-variant (default: --runs)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--months", type=int, nargs="*", default=[1, 4, 7, 10], help="calendar months 1..12")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--skip-missing", action="store_true", help="skip regions that have no month files at all for --year, and say which")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    from lulc.classes import CLASS_NAMES, N_CLASSES
    from lulc.config import load_config
    from lulc.data import load_store
    from lulc.infer import predict_month
    from lulc.metrics import ConfusionMatrix
    from lulc.sharpness import boundary_counts, f1_from_counts, passes_border_rule
    from lulc.train import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_year, regions = load_config(args.config)
    year = args.year or cfg_year

    def ensemble(runs, variant):
        seeds = sorted(Path(runs).glob(f"{variant}/seed*/best.pt"))
        if not seeds:
            raise SystemExit(f"no trained seeds for {variant} in {runs}")
        models, cards = zip(*[load_model(p.parent, dev) for p in seeds])
        return dict(models=list(models), norm={k: (v["mean"], v["std"]) for k, v in cards[0]["norm"].items()},
                    window=cards[0]["window_months"], use_stats="st0" in cards[0]["channels"])

    rows_to_score = [("none", ensemble(args.runs, args.variant), {})]
    rows_to_score += [(s, rows_to_score[0][1], SHARPEN_MODES[s]) for s in dict.fromkeys(args.settings) if s != "none"]
    if args.compare_variant:
        rows_to_score.append((args.compare_variant, ensemble(args.compare_runs or args.runs, args.compare_variant), {}))
    use_stats = any(e["use_stats"] for _, e, _ in rows_to_score)
    names = [n for n, _, _ in rows_to_score]
    cms = {n: ConfusionMatrix(N_CLASSES) for n in names}
    bnd = {n: np.zeros(4) for n in names}
    skipped, scored = [], 0
    for r in regions:
        if r.split not in args.splits or not (Path(args.data) / r.key).exists():
            continue
        if args.skip_missing and not list((Path(args.data) / r.key).glob(f"{year}-*.npz")):
            skipped.append(r.key)
            continue
        store = load_store(args.data, r.key, r.split, year, 0.0, False, stats=use_stats)
        for m in [m - 1 for m in args.months]:
            if not store.present[m]:
                continue
            for n, e, kw in rows_to_score:
                pred = predict_month(e["models"], store, m, e["norm"], e["window"], dev, patch=args.patch, overlap=args.overlap,
                                     use_stats=e["use_stats"], **kw)[0]
                cms[n].update(pred, store.label[m])
                bnd[n] += boundary_counts(pred, store.label[m], 1)
        scored += 1
        print(f"{r.key:12s} done", flush=True)
    if not scored:
        raise SystemExit(f"no regions scored: check --data, --splits {args.splits} and --year {year}")
    res = {n: {"miou": cms[n].miou(), "bf1": f1_from_counts(bnd[n]), "iou": cms[n].iou(), "ratio": bnd[n][1] / max(bnd[n][3], 1)} for n in names}
    tab = pd.DataFrame([{"setting": n, "mIoU": res[n]["miou"], "boundary_F1": res[n]["bf1"], "pred/ref border px": res[n]["ratio"],
                         **{f"iou_{c}": v for c, v in zip(CLASS_NAMES, res[n]["iou"])}} for n in names]).round(3)
    out = Path(args.runs) / f"sharpen_{args.variant}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "sharpen.csv", index=False)
    print(f"\n{scored} regions, year {year}, splits {args.splits}, months {args.months}")
    print(tab.to_string(index=False))
    for n in names[1:]:
        ok, why = passes_border_rule(res["none"], res[n])
        print(f"\n{n}: {'PASSES the rule' if ok else 'fails the rule: ' + '; '.join(why)}")
    if skipped:
        print(f"\nSkipped {len(skipped)} region(s) with no {year} month files: {', '.join(skipped)}.")
    print(f"\nSaved to {out}/sharpen.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
