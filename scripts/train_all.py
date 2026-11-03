"""Train variants x seeds. Resumable: a run with best.pt and history.csv is skipped.

    %run scripts/train_all.py --data data/raw --out runs                  # everything
    %run scripts/train_all.py --data data/raw --out runs --variants V0_scratch_1m --seeds 0 --epochs 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.config import load_config  # noqa: E402


def load_splits(data, regions, splits, year, conf_cutoff, coherence, stats=False, agree=False, label_source="dw"):
    """`year` is one year or a list of years. With several years every region-year is its own store,
    keyed 'region@year'. Each year gets its own yearly statistics, because stats are computed per store."""
    from lulc.data import apply_agree_filter, apply_worldcover_labels, load_store
    years = [year] if isinstance(year, int) else list(year)
    if len(years) > 1 and (agree or label_source == "worldcover"):
        raise SystemExit("WorldCover/agree labels exist for 2021 only; use label_source dw for several years")
    out = {}
    for r in regions:
        if r.split in splits and (Path(data) / r.key).exists():
            for y in years:
                try:
                    st = load_store(data, r.key, r.split, y, conf_cutoff, coherence, stats=stats)
                except FileNotFoundError as e:
                    print("skip", r.key, y, e)
                    continue
                if label_source == "worldcover":   # train on WorldCover instead of Dynamic World (needs reference.npz)
                    apply_worldcover_labels(st, data)
                if agree:                    # outside the try: a missing reference must stop the run, not silently drop or keep a region
                    print(f"{r.key}: agree filter removed {100 * apply_agree_filter(st, data):.0f}% of the labels")
                name = r.key if len(years) == 1 else f"{r.key}@{y}"
                st.key = name
                out[name] = st
    return out


def best_scores(out, variant):
    f = [json.loads(p.read_text())["best_val_miou"] for p in sorted(Path(out).glob(f"{variant}/seed*/summary.json"))]
    return float(np.mean(f)) if f else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--experiments", default=str(ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--train-years", type=int, nargs="*", help="training years (default: years.train in the config)")
    ap.add_argument("--val-years", type=int, nargs="*", help="validation years (default: years.val in the config)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=[], help="train and validate on these region keys only (e.g. the rural tiles that have HyP3 coherence)")
    ap.add_argument("--exclude", nargs="*", default=[], help="region keys to leave out, e.g. the five new southern/arid tiles, so a variant is compared on the old data")
    args = ap.parse_args(argv)

    from lulc.train import run
    exp = yaml.safe_load(Path(args.experiments).read_text())
    year, regions = load_config(args.config)
    from lulc.config import load_years
    yrs = load_years(args.config)
    train_years = args.train_years or yrs["train"]
    val_years = args.val_years or yrs["val"]
    if set(train_years) & set(yrs["test"]) and yrs["test"] != [year]:
        raise SystemExit(f"train years {train_years} overlap test years {yrs['test']}")
    print(f"training years {train_years}, validation years {val_years}")
    unknown = set(args.exclude) - {r.key for r in regions}
    if unknown:
        raise SystemExit(f"--exclude names unknown regions: {sorted(unknown)}")
    regions = [r for r in regions if r.key not in set(args.exclude)]
    if args.only:
        bad = set(args.only) - {r.key for r in regions}
        if bad:
            raise SystemExit(f"--only names unknown or excluded regions: {sorted(bad)}")
        regions = [r for r in regions if r.key in set(args.only)]
    seeds = args.seeds if args.seeds is not None else exp["seeds"]
    unknown = [v for v in (args.variants or []) if v not in exp["variants"]]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; configured: {list(exp['variants'])}")
    cache = {}
    for name, v in exp["variants"].items():
        if args.variants and name not in args.variants:
            continue
        cfg = {**exp["common"], **v, "workers": args.workers}
        if args.epochs:
            cfg["epochs"] = args.epochs
        if cfg["weights"] == "auto":
            s0, s1 = best_scores(args.out, "V0_scratch_1m"), best_scores(args.out, "V1_imagenet_1m")
            if np.isnan(s0) and np.isnan(s1):
                raise SystemExit("V2-V4 need V0 and V1 first (they decide the encoder weights)")
            cfg["weights"] = "imagenet" if (np.nan_to_num(s1, nan=-1) > np.nan_to_num(s0, nan=-1)) else None
            print(f"{name}: encoder weights = {cfg['weights']} (V0 {s0:.3f} vs V1 {s1:.3f})")
        k = (cfg.get("conf_cutoff", 0.0), cfg.get("coherence", False), cfg.get("yearly_stats", False), cfg.get("agree_filter", False), cfg.get("label_source", "dw"))
        if k not in cache:
            cache.clear()
            # Validation labels are never cut: every variant is scored on the same pixels, so the
            # validation mIoU is comparable across variants. Only the training labels use the cutoff.
            cache[k] = (load_splits(args.data, regions, {"train"}, train_years, k[0], k[1], k[2], k[3], k[4]),
                        load_splits(args.data, regions, {"val"}, val_years, 0.0, k[1], k[2]))
            print(f"loaded {len(cache[k][0])} train and {len(cache[k][1])} val regions")
            if not cache[k][0] or not cache[k][1]:
                raise SystemExit(f"{name}: no {'train' if not cache[k][0] else 'validation'} regions could be loaded. Look at the 'skip' lines above: "
                                 "a region is skipped when its folder or its coherence file is missing "
                                 "(for coherence variants run hyp3_coherence.py collect until every tile says done).")
        tr, va = cache[k]
        for seed in seeds:
            d = Path(args.out) / name / f"seed{seed}"
            if (d / "summary.json").exists():
                print("skip", d); continue
            print(f"=== {name} seed {seed}")
            res = run({**cfg, "seed": seed}, tr, va, d)
            (d / "years.json").write_text(json.dumps({"train": train_years, "val": val_years}))
            (d / "summary.json").write_text(json.dumps({"best_val_miou": res["best_val_miou"], "variant": name, "seed": seed}))
    rows = []
    for p in sorted(Path(args.out).glob("*/seed*/summary.json")):
        j = json.loads(p.read_text()); rows.append(j)
    if rows:
        df = pd.DataFrame(rows).groupby("variant").best_val_miou.agg(["mean", "std", "count"])
        print("\nValidation mIoU (gaps under 0.05 are inside seed noise):\n", df.round(3))
        df.to_csv(Path(args.out) / "val_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
