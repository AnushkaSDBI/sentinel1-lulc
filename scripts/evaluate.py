"""Test evaluation of one variant (seed ensemble), plus the false-change test.

    %run scripts/evaluate.py --data data/raw --runs runs --variant V3_window3_conf
    %run scripts/evaluate.py --data data/raw_2024 --year 2024 --splits test --variant V3_window3_conf --tag y2024

Needs reference.npz (export_reference.py) per city for WorldCover/Esri scoring and the stable-pixel mask.
Writes <runs>/eval_<variant><tag>/ with metrics.json, per_city.csv, change.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.classes import CLASS_NAMES, IGNORE_INDEX, N_CLASSES, remap_esri, remap_worldcover  # noqa: E402
from lulc.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "regions.yaml"))
    ap.add_argument("--splits", nargs="*", default=["test", "test_ood"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prob-smooth", action="store_true",
                    help="also score smoothing in probability space (mean softmax over the window) against the majority vote; needs about 1 GB extra for a 24 km city")
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--esri-scheme", choices=["original", "remapped"], help="default: detected from all selected cities pooled")
    ap.add_argument("--patch", type=int, default=512, help="sliding-window size in pixels for inference (models are trained at 256)")
    ap.add_argument("--overlap", type=int, default=64, help="sliding-window overlap in pixels")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip regions that have no month files at all for --year (e.g. a region exported for 2025 and 2026 only) and say which")
    ap.add_argument("--min-class-share", type=float, default=0.01,
                    help="per-region 'major class' mIoU counts only classes with at least this share of the region's reference pixels")
    ap.add_argument("--no-fill-gaps", action="store_true",
                    help="feed no-data neighbour months to the model as blanks, as before, instead of filling them from the nearest month in the window (for a before/after comparison)")
    args = ap.parse_args(argv)
    if not 0 <= args.overlap < args.patch:
        raise SystemExit(f"--overlap must be at least 0 and smaller than --patch, got overlap {args.overlap}, patch {args.patch}")
    if not 0 <= args.min_class_share < 1:
        raise SystemExit(f"--min-class-share must be at least 0 and below 1, got {args.min_class_share}")

    import torch
    from lulc.change_eval import false_change, member_spread, probability_smooth, stable_mask, temporal_majority
    from lulc.data import load_store
    from lulc.infer import predict_region
    from lulc.metrics import ConfusionMatrix, bootstrap_miou, map_agreement_counts, tolerant_target
    from lulc.ood import area_signature
    from lulc.sharpness import boundary_counts, f1_from_counts
    from lulc.train import load_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_year, regions = load_config(args.config)
    year = args.year or cfg_year
    seeds = sorted(Path(args.runs).glob(f"{args.variant}/seed*/best.pt"))
    if not seeds:
        raise SystemExit(f"no trained seeds for {args.variant} in {args.runs}")
    models, card = zip(*[load_model(p.parent, dev) for p in seeds])
    card = card[0]
    norm = {k: (v["mean"], v["std"]) for k, v in card["norm"].items()}
    window, use_coh = card["window_months"], "coh" in card["channels"]
    use_stats = "st0" in card["channels"]
    if (card.get("config") or {}).get("agree_filter"):
        print("NOTE: this model was trained on labels filtered by WorldCover. Scores against WorldCover are no longer independent; "
              "report Esri as the independent check.")
    out = Path(args.runs) / f"eval_{args.variant}{args.tag}"; out.mkdir(parents=True, exist_ok=True)
    print(f"{len(models)} seeds, window {window}, coherence {use_coh}, yearly stats {use_stats}, year {year}, patch {args.patch}, overlap {args.overlap}, "
          f"fill gaps {not args.no_fill_gaps}")
    if year != cfg_year:
        print(f"NOTE: year {year} differs from the label year {cfg_year}. Only Dynamic World of {year} is used; the WorldCover/Esri rows "
              "and the stable-land flicker test (2021 maps) are skipped. Esri of the same year is scored when reference_esri_<year>.npz exists.")

    scheme = args.esri_scheme
    if scheme is None:
        from lulc.classes import detect_esri_scheme
        vals = set()
        for r in regions:
            f = Path(args.data) / r.key / "reference.npz"
            if r.split in args.splits and f.exists():
                with np.load(f) as z:
                    vals |= set(np.unique(z["esri"]).tolist())
        scheme = detect_esri_scheme(np.array(sorted(v for v in vals if v > 0))) if vals else None
        print("Esri scheme:", scheme)
    per = {s: {} for s in ("dw", "wc", "esri", "agree", "dw_wc_agree", "tolerant", *(("dw_majority", "dw_probmean") if args.prob_smooth else ()))}
    agreement = np.zeros((N_CLASSES, 4), np.int64)   # per Dynamic World class: pixels, WorldCover agrees, Esri agrees, both agree
    change_rows, city_rows = [], []
    group_of = {}
    ood_rows = []                       # label-free signature per region, for the prototype warning
    by_month = {}                       # calendar month -> pooled confusion matrix vs Dynamic World
    bnd = {"raw": np.zeros(4), "majority": np.zeros(4), "label_vs_esri": np.zeros(4)}   # pooled boundary counts vs Dynamic World
    bnd_rows = []
    skipped = []
    for r in regions:
        if r.split not in args.splits or not (Path(args.data) / r.key).exists():
            continue
        if args.skip_missing and not list((Path(args.data) / r.key).glob(f"{year}-*.npz")):
            skipped.append(r.key)
            continue
        group_of[r.key] = "cities" if r.half_size_km > 6 else "rural"
        store = load_store(args.data, r.key, r.split, year, 0.0, ((card.get("config") or {}).get("coherence") or use_coh) if use_coh else False, stats=use_stats)
        res = predict_region(list(models), store, norm, window, dev, use_coh=use_coh, use_stats=use_stats, return_probs=args.prob_smooth,
                             patch=args.patch, overlap=args.overlap, fill_gaps=not args.no_fill_gaps)
        preds, conf, members, months = res[:4]
        lab = store.label[months]
        cms = {k: ConfusionMatrix(N_CLASSES) for k in per}
        cms["dw"].update(preds, lab)
        c_raw = boundary_counts(preds, lab, 1)                         # border sharpness of the raw model maps
        c_maj = boundary_counts(temporal_majority(preds, args.smooth), lab, 1) if len(months) >= 3 else c_raw
        bnd["raw"] += c_raw; bnd["majority"] += c_maj
        bnd_rows.append({"region": r.key, "group": group_of[r.key], "bf1_raw": f1_from_counts(c_raw), "bf1_majority": f1_from_counts(c_maj),
                         "pred_border_px": c_raw[1], "ref_border_px": c_raw[3]})
        if args.prob_smooth and len(months) >= 3:
            ps = probability_smooth(res[4], args.smooth)
            cms["dw_majority"].update(temporal_majority(preds, args.smooth), lab)
            cms["dw_probmean"].update(ps, lab)
            del res
        for t, mi in enumerate(months):
            by_month.setdefault(mi + 1, ConfusionMatrix(N_CLASSES)).update(preds[t], lab[t])
        ref = Path(args.data) / r.key / "reference.npz"
        stable = None
        if ref.exists() and year == cfg_year:     # WorldCover/Esri here are 2021 maps; never score another year against them
            with np.load(ref) as z:
                wc, es = remap_worldcover(z["wc"]), remap_esri(z["esri"], scheme)
            stable = stable_mask(wc, es)
            for t in range(len(months)):
                cms["wc"].update(preds[t], wc); cms["esri"].update(preds[t], es)
                cms["agree"].update(preds[t], np.where(stable, wc, IGNORE_INDEX))
                cms["dw_wc_agree"].update(preds[t], np.where(lab[t] == wc, wc, IGNORE_INDEX))   # Dynamic World scored only where WorldCover agrees
                cms["tolerant"].update(preds[t], tolerant_target(preds[t], lab[t], wc, es))     # correct if it matches any of the three maps
                agreement += map_agreement_counts(lab[t], wc, es)
        elif year != cfg_year:     # another year: only Esri has an annual map (reference_esri_<year>.npz from export_reference.py --year)
            eref = Path(args.data) / r.key / f"reference_esri_{year}.npz"
            if eref.exists():
                with np.load(eref) as z:
                    es = remap_esri(z["esri"], "original")
                for t in range(len(months)):
                    cms["esri"].update(preds[t], es)
        for k in per:
            per[k][r.key] = cms[k]
        sig = area_signature(conf, members)
        major = {"dw_miou_major": cms["dw"].miou_major(args.min_class_share), "n_major_classes": int(cms["dw"].major_classes(args.min_class_share).sum())}
        ood_rows.append({"region": r.key, "split": r.split, "group": group_of[r.key], **sig, "miou_dw": cms["dw"].miou(), **major})
        city_rows.append({"region": r.key, "split": r.split, "climate": r.climate, **sig, **{f"{k}_miou": cms[k].miou() for k in cms}, **major,
                          **{f"{k}_oa": cms[k].oa for k in cms}, **{f"iou_{c}": v for c, v in zip(CLASS_NAMES, cms["dw"].iou())}})
        if stable is not None and len(months) >= 3:
            sm = temporal_majority(preds, args.smooth)
            spread = member_spread(members)
            variants = [("model", preds), (f"model+majority{args.smooth}", sm), ("dynamic_world", lab)]
            if args.prob_smooth:
                variants.insert(2, (f"model+probmean{args.smooth}", ps))
            for name, p in variants:
                change_rows.append({"region": r.key, "maps": name, **false_change(p, stable)})
            change_rows.append({"region": r.key, "maps": "member_spread_pp_mean", "monthly_churn_pp": float(spread["std"].mean())})
        print(f"{r.key:12s} mIoU vs DW {cms['dw'].miou():.3f}  OA {cms['dw'].oa:.3f}  major-class mIoU {major['dw_miou_major']:.3f} ({major['n_major_classes']} classes)")

    if bnd_rows:
        pd.DataFrame(bnd_rows).to_csv(out / "boundary.csv", index=False)
        print(f"\nBoundary F1 vs Dynamic World (1 px tolerance, pooled): raw {f1_from_counts(bnd['raw']):.3f}, "
              f"after {args.smooth}-month majority {f1_from_counts(bnd['majority']):.3f}. "
              f"Predicted border pixels / reference border pixels: {bnd['raw'][1] / max(bnd['raw'][3], 1):.2f} "
              "(above 1 = speckly, noisy borders; below 1 = soft, merged borders)")
        (out / "boundary_pooled.json").write_text(json.dumps({
            "bf1_raw": f1_from_counts(bnd["raw"]), "bf1_majority": f1_from_counts(bnd["majority"]),
            "pred_over_ref_border_px": float(bnd["raw"][1] / max(bnd["raw"][3], 1))}, indent=2))
    pd.DataFrame(city_rows).to_csv(out / "per_city.csv", index=False)
    # in_dist = held-out cities of the normal test split; OOD cities (Milan, Valencia) and rural tiles listed separately
    ref = {"in_dist": [{k: v for k, v in o.items() if k != "split"} for o in ood_rows if o["split"] == "test" and o["group"] == "cities"],
           "other": [o for o in ood_rows if not (o["split"] == "test" and o["group"] == "cities")]}
    (out / "ood_reference.json").write_text(json.dumps(ref, indent=2))
    if len(ood_rows) >= 4:
        from scipy.stats import spearmanr
        rho = spearmanr([o["mean_conf"] for o in ood_rows], [o["miou_dw"] for o in ood_rows])[0]
        print(f"Spearman(mean confidence, mIoU vs DW) over {len(ood_rows)} regions: {rho:.2f}  (the OOD warning is only useful if this is clearly positive)")
        multi = [o for o in ood_rows if o["n_major_classes"] >= 2]
        if len(multi) >= 4:
            rho_major = spearmanr([o["mean_conf"] for o in multi], [o["dw_miou_major"] for o in multi])[0]
            print(f"Spearman(mean confidence, major-class mIoU) over {len(multi)} regions with at least 2 major classes: {rho_major:.2f}")
    if city_rows:
        all_regions = pd.DataFrame(city_rows)
        # A one-class region has no class to confuse, so its score says little: left out of these means, flagged in per_city.csv (n_major_classes).
        all_regions["miou_major_multi"] = all_regions.dw_miou_major.where(all_regions.n_major_classes >= 2)
        single = all_regions.loc[all_regions.n_major_classes < 2, "region"].tolist()
        print(f"\nPer-region major-class mIoU vs Dynamic World (classes with at least {args.min_class_share:.1%} of a region's reference pixels; "
              f"one-class regions left out: {', '.join(single) or 'none'}):")
        for grp in ("cities", "rural"):
            v = all_regions.loc[[group_of[k] == grp for k in all_regions.region], "miou_major_multi"].dropna()
            if len(v):
                print(f"  {grp:6s} mean {v.mean():.3f}  median {v.median():.3f}  ({len(v)} regions)")
        cdf = all_regions[all_regions.climate != ""]
        if len(cdf):
            clim = cdf.groupby("climate")[["dw_miou", "miou_major_multi", "dw_oa", "mean_conf"]].agg(["mean", "count"]).round(3)
            clim.to_csv(out / "per_climate.csv")
            print("\nPer climate group (mean over regions; one tile can dominate a group, read with the region table; mIoU_major leaves out one-class regions):")
            print(cdf.groupby("climate").agg(n=("region", "count"), mIoU=("dw_miou", "mean"), mIoU_major=("miou_major_multi", "mean"),
                                             OA=("dw_oa", "mean"), conf=("mean_conf", "mean")).round(3))
    pd.DataFrame(change_rows).to_csv(out / "change.csv", index=False)
    metrics, class_rows = {}, []
    for k, per_region in per.items():
        for grp in ("all", "cities", "rural"):
            d = {c: m for c, m in per_region.items() if m.m.sum() > 0 and (grp == "all" or group_of[c] == grp)}
            if not d:
                continue
            pooled = next(iter(d.values()))
            for m in list(d.values())[1:]:
                pooled = pooled.merge(m)
            name = k if grp == "all" else f"{k}_{grp}"
            class_rows.extend({"set": name, "class": c, "iou": i, "f1": f, "precision": pr, "recall": rc, "support_px": int(sp)}
                              for c, i, f, pr, rc, sp in zip(CLASS_NAMES, pooled.iou(), pooled.f1(), pooled.precision(), pooled.recall(), pooled.support))
            metrics[name] = {**bootstrap_miou(d), "kappa": round(pooled.kappa, 4), "per_class_iou": dict(zip(CLASS_NAMES, np.round(pooled.iou(), 4).tolist())),
                             "per_class_f1": dict(zip(CLASS_NAMES, np.round(pooled.f1(), 4).tolist())),
                             "confusion": pooled.m.tolist()}
    month_tab = pd.DataFrame({mm: {"OA": c.oa, "mIoU": c.miou(), **dict(zip(CLASS_NAMES, c.iou()))}
                              for mm, c in sorted(by_month.items())}).T.round(3)
    month_tab.to_csv(out / "by_month.csv")
    print("\nScore by calendar month (vs Dynamic World, all test regions):")
    print(month_tab)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pc = pd.DataFrame(class_rows)
    pc.to_csv(out / "per_class.csv", index=False)
    if len(pc):
        print("\nPer class, pooled, vs Dynamic World (IoU, F1, precision, recall; rural = tiles of 10 km, cities = 24 km squares):")
        for s in ("dw", "dw_rural", "dw_cities"):
            t = pc[pc.set == s]
            if len(t):
                print(f"-- {s}"); print(t.drop(columns="set").round(3).to_string(index=False))
    if agreement[:, 0].sum():
        share = 100 * agreement[:, 1:] / np.maximum(agreement[:, [0]], 1)
        agree_tab = pd.DataFrame(share, index=CLASS_NAMES, columns=["worldcover_agrees_%", "esri_agrees_%", "both_agree_%"]).round(1)
        agree_tab.insert(0, "dw_pixels", agreement[:, 0])
        agree_tab.to_csv(out / "map_agreement.csv")
        print("\nHow often the other maps agree with Dynamic World, per Dynamic World class (no model involved):")
        print(agree_tab)
    for k, v in metrics.items():
        print(f"{k:12s} mIoU {v['miou']:.3f} [{v['miou_lo']:.3f}, {v['miou_hi']:.3f}]  OA {v['oa']:.3f}  ({v['n_cities']} cities)")
    if "tolerant" in metrics:
        same = pd.DataFrame(city_rows).dropna(subset=["tolerant_miou"])
        print("tolerant: a prediction counts as correct if it matches Dynamic World, WorldCover or Esri. It flatters any model, and it covers only the regions "
              "with WorldCover and Esri maps, so compare it with dw on those same regions:")
        print(f"  mean over {len(same)} regions: dw mIoU {same.dw_miou.mean():.3f}, tolerant mIoU {same.tolerant_miou.mean():.3f}")
    if change_rows:
        print(pd.DataFrame(change_rows).groupby("maps")[["flip_rate_pct", "monthly_churn_pp", "net_total_change_pp"]].mean().round(2))
    if skipped:
        print(f"\nSkipped {len(skipped)} region(s) with no {year} month files: {', '.join(skipped)}. Scored {len(group_of)} regions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
