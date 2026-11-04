"""What does the shipped model rely on? Input ablation table (validation or test regions).

    %run scripts/ablate_inputs.py --data data/raw --runs runs --variant V12_window3_balanced_stats --splits val
    %run scripts/ablate_inputs.py --data data/raw --runs runs_multiyear --variant V12_window3_balanced_stats --splits val --year 2024

Replaces one part of the input at a time (average value, blur, or the neighbours' value) and prints how much the IoU
of each class drops. Writes <runs>/ablation_<variant><tag>/ablation.csv and ablation.md (paste the table into the report).
Four months (Jan, Apr, Jul, Oct) are used by default to keep the run short; use --months to change.
Use validation regions for this: it explains the model, it does not select anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    import torch
    from lulc.ablation import ablation_confusions, ablation_table, channel_groups, default_ablations
    from lulc.classes import CLASS_NAMES
    from lulc.config import load_config
    from lulc.data import load_store
    from lulc.torch_data import extra_spec
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
    window, use_coh, use_stats = card["window_months"], "coh" in card["channels"], "st0" in card["channels"]
    spec = extra_spec(0, use_stats) if not use_coh else None
    if use_coh:
        raise SystemExit("this script covers radar + yearly-statistics models; the coherence variant is a research model")
    abl = default_ablations(window, spec)
    print(f"{len(models)} seeds, window {window}, yearly stats {use_stats}, year {year}, groups: {list(channel_groups(window, spec))}")
    pooled = {}
    for r in regions:
        if r.split not in args.splits or not (Path(args.data) / r.key).exists():
            continue
        store = load_store(args.data, r.key, r.split, year, 0.0, False, stats=use_stats)
        cms = ablation_confusions(list(models), store, norm, window, dev, abl, [m - 1 for m in args.months], use_stats)
        for k, cm in cms.items():
            pooled[k] = cm if k not in pooled else pooled[k].merge(cm)
        print(f"{r.key:12s} baseline mIoU {cms['full input'].miou():.3f}", flush=True)
    if not pooled:
        raise SystemExit("no regions found for those splits in --data")
    tab = ablation_table(pooled, CLASS_NAMES)
    out = Path(args.runs) / f"ablation_{args.variant}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    tab.round(3).to_csv(out / "ablation.csv", index=False)
    (out / "ablation.md").write_text(tab.round(3).to_markdown(index=False) if hasattr(tab, "to_markdown") else tab.round(3).to_string(index=False))
    print("\nChange in IoU against the full input (more negative = relied on more):")
    print(tab.round(3).to_string(index=False))
    print(f"\nSaved to {out}/ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
