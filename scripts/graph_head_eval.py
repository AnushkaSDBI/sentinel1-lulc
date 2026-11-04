"""Experimental: superpixel graph head on the frozen model's probabilities (see lulc.graph_head). No change to the shipped model.

    %run scripts/graph_head_eval.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats \
         --train-splits val --eval-splits test test_ood --year 2024 --eval-year 2025

Steps: (1) the frozen model predicts probabilities for a few months of every region; (2) superpixels are cut on the
yearly median radar; (3) the head is trained on the --train-splits regions only; (4) three maps are scored on the
--eval-splits regions: the model itself, the untrained object vote (mean probability per superpixel), and the trained head.
Decision rule (set before the results): sharpness.passes_border_rule (boundary F1 +0.02, mIoU not down more than 0.005,
no class down more than 0.05). Judge on the eval regions once. Expect a small effect: the blur is mostly the radar's
20 m resolution and the noisy labels.
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
    ap.add_argument("--eval-data", default=None, help="folder of the evaluation year (default: --data)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--train-splits", nargs="*", default=["val"])
    ap.add_argument("--eval-splits", nargs="*", default=["test", "test_ood"])
    ap.add_argument("--year", type=int, help="year of the training (validation) regions")
    ap.add_argument("--eval-year", type=int, help="year of the evaluation regions (default: --year)")
    ap.add_argument("--months", type=int, nargs="*", default=[1, 4, 7, 10])
    ap.add_argument("--segments", type=int, default=3000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    import pandas as pd
    import torch
    from lulc.classes import CLASS_NAMES, N_CLASSES
    from lulc.config import load_config
    from lulc.crf import yearly_median_db
    from lulc.data import load_store
    from lulc.graph_head import (build_graph, node_labels, object_vote, pixels_from_nodes, predict_nodes, superpixels, train_head)
    from lulc.infer import predict_month
    from lulc.metrics import ConfusionMatrix
    from lulc.sharpness import boundary_counts, f1_from_counts, passes_border_rule
    from lulc.train import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_year, regions = load_config(args.config)
    year = args.year or cfg_year
    eval_year = args.eval_year or year
    seeds = sorted(Path(args.runs).glob(f"{args.variant}/seed*/best.pt"))
    if not seeds:
        raise SystemExit(f"no trained seeds for {args.variant} in {args.runs}")
    models, cards = zip(*[load_model(p.parent, dev) for p in seeds])
    card = cards[0]
    norm = {k: (v["mean"], v["std"]) for k, v in card["norm"].items()}
    window, use_stats = card["window_months"], "st0" in card["channels"]

    def region_items(splits, data, yr):
        for r in regions:
            if r.split not in splits or not (Path(data) / r.key).exists():
                continue
            store = load_store(data, r.key, r.split, yr, 0.0, False, stats=use_stats)
            guide = yearly_median_db(store.s1, store.present)
            seg = superpixels(guide, args.segments)
            for m in [m - 1 for m in args.months]:
                if not store.present[m]:
                    continue
                pred, _, _, probs = predict_month(list(models), store, m, norm, window, dev, use_stats=use_stats, return_probs=True)
                yield r.key, store.label[m], pred, probs.astype(np.float32), guide, seg

    graphs, labels = [], []
    for key, lab, pred, probs, guide, seg in region_items(args.train_splits, args.data, year):
        graphs.append(build_graph(seg, probs, guide))
        labels.append(node_labels(seg, lab, N_CLASSES))
    if not graphs:
        raise SystemExit("no training regions found for --train-splits")
    head = train_head(graphs, labels, N_CLASSES)
    print(f"head trained on {len(graphs)} region-months")

    names = ["model", "object_vote", "graph_head"]
    cms = {n: ConfusionMatrix(N_CLASSES) for n in names}
    bnd = {n: np.zeros(4) for n in names}
    for key, lab, pred, probs, guide, seg in region_items(args.eval_splits, args.eval_data or args.data, eval_year):
        g = build_graph(seg, probs, guide)
        outs = {"model": pred, "object_vote": pixels_from_nodes(seg, object_vote(g)), "graph_head": pixels_from_nodes(seg, predict_nodes(head, g))}
        for n, q in outs.items():
            q = np.where(pred == 255, 255, q).astype(np.uint8)
            cms[n].update(q, lab)
            bnd[n] += boundary_counts(q, lab, 1)
    res = {n: {"miou": cms[n].miou(), "bf1": f1_from_counts(bnd[n]), "iou": cms[n].iou()} for n in names}
    tab = pd.DataFrame([{"map": n, "mIoU": res[n]["miou"], "boundary_F1": res[n]["bf1"],
                         **{f"iou_{c}": v for c, v in zip(CLASS_NAMES, res[n]["iou"])}} for n in names]).round(3)
    out = Path(args.runs) / f"graph_head_{args.variant}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "graph_head.csv", index=False)
    print(tab.to_string(index=False))
    for n in names[1:]:
        ok, why = passes_border_rule(res["model"], res[n])
        print(f"\n{n}: {'PASSES the rule' if ok else 'fails the rule: ' + '; '.join(why)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
