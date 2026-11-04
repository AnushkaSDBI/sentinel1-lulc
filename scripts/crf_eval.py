"""Does a convolutional CRF after the model sharpen borders without hurting mIoU? (no training)

    %run scripts/crf_eval.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --splits val
    %run scripts/crf_eval.py --data data/raw_2025 --runs runs_multiyear --variant V12_window3_balanced_stats --splits test test_ood --year 2025

Scores the model's probability maps with and without each CRF setting (see lulc.crf, two settings fixed in advance),
pooled over the regions: mIoU, boundary F1 (1 px tolerance) and per-class IoU, all against Dynamic World.

Decision rule (set before the results): use the CRF setting in the prototype only if, on VALIDATION regions,
  boundary F1 rises by at least 0.02, pooled mIoU does not fall by more than 0.005, and no class IoU falls by more than 0.05.
Then confirm once on the test regions. Choosing the better of two settings on validation is mildly optimistic; the test run shows if it holds.
Four months (Jan, Apr, Jul, Oct) by default to keep the run short.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--months", type=int, nargs="*", default=[1, 4, 7, 10], help="calendar months 1..12")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    from lulc.classes import CLASS_NAMES, N_CLASSES
    from lulc.config import load_config
    from lulc.crf import CRF_SETTINGS, conv_crf_tiled, yearly_median_db
    from lulc.data import load_store
    from lulc.infer import predict_month
    from lulc.metrics import ConfusionMatrix
    from lulc.sharpness import boundary_counts, f1_from_counts, passes_border_rule
    from lulc.train import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_year, regions = load_config(args.config)
    year = args.year or cfg_year
    seeds = sorted(Path(args.runs).glob(f"{args.variant}/seed*/best.pt"))
    if not seeds:
        raise SystemExit(f"no trained seeds for {args.variant} in {args.runs}")
    models, cards = zip(*[load_model(p.parent, dev) for p in seeds])
    card = cards[0]
    norm = {k: (v["mean"], v["std"]) for k, v in card["norm"].items()}
    window, use_stats = card["window_months"], "st0" in card["channels"]
    names = ["none", *CRF_SETTINGS]
    cms = {n: ConfusionMatrix(N_CLASSES) for n in names}
    bnd = {n: np.zeros(4) for n in names}
    for r in regions:
        if r.split not in args.splits or not (Path(args.data) / r.key).exists():
            continue
        store = load_store(args.data, r.key, r.split, year, 0.0, False, stats=use_stats)
        guide = torch.from_numpy(yearly_median_db(store.s1, store.present))
        for m in [m - 1 for m in args.months]:
            if not store.present[m]:
                continue
            pred, _, _, probs = predict_month(list(models), store, m, norm, window, dev, use_stats=use_stats, return_probs=True)
            lab = store.label[m]
            outs = {"none": pred}
            pt = torch.from_numpy(probs.astype(np.float32))
            for n, kw in CRF_SETTINGS.items():
                q = conv_crf_tiled(pt, guide, **kw).argmax(0).numpy().astype(np.uint8)
                q[pred == 255] = 255                               # no-data stays ignored
                outs[n] = q
            for n, q in outs.items():
                cms[n].update(q, lab)
                bnd[n] += boundary_counts(q, lab, 1)
        print(f"{r.key:12s} done", flush=True)
    rows = []
    res = {n: {"miou": cms[n].miou(), "bf1": f1_from_counts(bnd[n]), "iou": cms[n].iou(), "ratio": bnd[n][1] / max(bnd[n][3], 1)} for n in names}
    for n in names:
        rows.append({"setting": n, "mIoU": res[n]["miou"], "boundary_F1": res[n]["bf1"], "pred/ref border px": res[n]["ratio"],
                     **{f"iou_{c}": v for c, v in zip(CLASS_NAMES, res[n]["iou"])}})
    tab = pd.DataFrame(rows).round(3)
    out = Path(args.runs) / f"crf_{args.variant}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "crf.csv", index=False)
    print(tab.to_string(index=False))
    for n in CRF_SETTINGS:
        ok, why = passes_border_rule(res["none"], res[n])
        print(f"\n{n}: {'PASSES the rule' if ok else 'fails the rule: ' + '; '.join(why)}")
    print(f"\nSaved to {out}/crf.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
