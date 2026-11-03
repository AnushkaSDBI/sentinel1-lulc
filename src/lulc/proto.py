"""Logic of the prototype, kept out of the Streamlit file so it can be tested.

Flow: lat/lon + start month -> 12 monthly composites (Earth Engine, same recipe as the
training data) -> seed-ensemble U-Net -> optional 3-month majority smoothing ->
class shares, month-to-month change in pp, total change, uncertainty, noise floor.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .change_eval import member_spread, temporal_majority
from .classes import CLASS_NAMES, IGNORE_INDEX, N_CLASSES, remap_esri, remap_worldcover
from .config import Region
from .data import RegionStore, load_store_ym
from .ee_export import (CompositeSettings, build_image, build_reference_image, export_reference_region,
                        export_region_month, make_fetcher, month_path, region_direction, verify_grid_with_ee)
from .grid import grid_bounds_lonlat, grid_for_region
from .metrics import ConfusionMatrix
from .stats import monthly_change_table

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# False change measured on pixels that WorldCover and Esri both call stable (9 test regions, V2,
# 3 seeds, 2021). Total class churn per month = sum of |change in class share| / 2, in pp.
NOISE_FLOOR = {
    "monthly_churn_raw_pp": 6.3,
    "monthly_churn_smoothed_pp": 4.2,
    "net_first_to_last_pp": 7.9,
    "seed_spread_pp": 1.2,
    "flip_rate_raw_pct": 8.3,
    "flip_rate_smoothed_pct": 5.1,
    "source": "false-change test on 9 test regions, 2021 (see report)",
}


def month_sequence(year: int, month: int, n: int = 12) -> list[tuple[int, int]]:
    i = year * 12 + month - 1
    return [((i + k) // 12, (i + k) % 12 + 1) for k in range(n)]


def month_labels(ym) -> list[str]:
    return [f"{MONTH_NAMES[m - 1]} {y}" for y, m in ym]


def make_region(lat: float, lon: float, size_km: float) -> Region:
    if not (-80 <= lat <= 84 and -180 <= lon < 180):
        raise ValueError("lat must be in -80..84 (the UTM grid ends there) and lon in -180..180")
    return Region(f"p{lat:+.4f}_{lon:+.4f}_{size_km:g}", "prototype area", lat, lon, "case_study", size_km / 2)


FIRST_YEAR = 2015                      # first full year of Sentinel-1 IW archive in Earth Engine (S1A only until 2016)


def check_period(ym, today=None) -> list[str]:
    """Raise if the period cannot exist; return warnings about what the radar archive looks like then."""
    import datetime as dt

    today = today or dt.date.today()
    (y0, m0), (y1, m1) = ym[0], ym[-1]
    if (y0, m0) < (FIRST_YEAR, 1):
        raise ValueError(f"Sentinel-1 data in Earth Engine starts reliably in {FIRST_YEAR}; choose a later start.")
    if (y1, m1) >= (today.year, today.month):
        raise ValueError(f"{MONTH_NAMES[m1 - 1]} {y1} is not finished yet. Choose an earlier start month or year.")
    warn = []
    if (y0, m0) < (2017, 1):
        warn.append("Before 2017 Sentinel-1B was new or missing and many places have only one satellite: fewer scenes, "
                    "noisier composites, and some areas outside Europe have gaps.")
    if (y1, m1) >= (2022, 1):
        warn.append("Sentinel-1B stopped in December 2021, so months from 2022 have about half the scenes of 2021. "
                    "The model was trained on 2021, so the noise in these months is higher than our noise floor.")
    if y0 != 2021 or y1 != 2021:
        warn.append("The model was trained on 2021 labels only. Other years run through the same model without any check.")
    return warn


def yearly_summary(preds_by_year: dict) -> dict:
    """{year: monthly predictions (T,H,W)} -> annual map (most frequent class), and shares in percent per class."""
    maps = {y: annual_mode(p) for y, p in preds_by_year.items()}
    rows = {}
    for y, a in maps.items():
        valid = a != IGNORE_INDEX
        rows[y] = [100.0 * (a[valid] == k).sum() / max(valid.sum(), 1) for k in range(N_CLASSES)]
    return {"maps": maps, "shares": pd.DataFrame(rows, index=CLASS_NAMES).T.rename_axis("year")}


def transition_matrix(a, b, ignore: int = IGNORE_INDEX) -> pd.DataFrame:
    """Share of the valid area (percent) that moved from class a (rows) to class b (columns)."""
    a, b = np.asarray(a), np.asarray(b)
    ok = (a != ignore) & (b != ignore)
    m = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(m, (a[ok].astype(int), b[ok].astype(int)), 1)
    return pd.DataFrame(100 * m / max(ok.sum(), 1), index=CLASS_NAMES, columns=CLASS_NAMES)


# ------------------------------------------------------------------ data
def fetch_months(ee, region: Region, ym, workdir, progress=None, workers: int = 4):
    """Download the monthly composites into workdir (npz per month, resumable). Returns the grid."""
    from .ee_stats import region_geometry

    grid = grid_for_region(region)
    verify_grid_with_ee(ee, grid)
    direction = region_direction(ee, region, ym[0][0])
    geom = region_geometry(ee, region)
    for i, (y, m) in enumerate(ym):
        path = month_path(Path(workdir), region.key, y, m)
        if not path.exists():
            image = build_image(ee, geom, y, m, direction, CompositeSettings())
            export_region_month(grid, make_fetcher(ee, image, grid), path,
                                {"region": region.key, "year": y, "month": m, "direction": direction}, workers=workers)
        if progress:
            progress((i + 1) / len(ym), f"downloaded {MONTH_NAMES[m - 1]} {y}")
    return grid


def fetch_reference(ee, region: Region, workdir) -> dict | None:
    """WorldCover and Esri 2021 on the same grid, mapped to our 6 classes (Esri scheme confirmed 'original')."""
    path = Path(workdir) / region.key / "reference.npz"
    if not path.exists():
        grid = grid_for_region(region)
        export_reference_region(grid, make_fetcher(ee, build_reference_image(ee, 2021), grid), path, {"region": region.key})
    return load_reference(path)


def load_reference(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as z:
        wc, es = remap_worldcover(z["wc"]), remap_esri(z["esri"], "original")
    return {"wc": wc, "esri": es, "agree": np.where(wc == es, wc, IGNORE_INDEX).astype(np.uint8)}


def load_local(data_dir, region_key: str, year: int = 2021):
    """Use an already exported region (offline test). Calendar year of `year`."""
    ym = month_sequence(year, 1)
    return load_store_ym(data_dir, region_key, ym, "case_study"), ym


# ------------------------------------------------------------------ model
def load_ensemble(weights_dir, device):
    """Every sub-folder of weights_dir (or weights_dir itself) that holds best.pt + model_card.json."""
    from .train import FINAL_NAME, load_final, load_model

    root = Path(weights_dir)
    if (root / FINAL_NAME).exists():                  # one bundled file (scripts/package_model.py)
        models, cards = load_final(root / FINAL_NAME, device)
        names = [f"seed{i}" for i in range(len(models))]
    else:
        names = None
    dirs = [d for d in sorted([root, *[p for p in root.iterdir() if p.is_dir()]])
            if (d / "best.pt").exists() and (d / "model_card.json").exists()] if root.exists() else []
    if names is None:
        if not dirs:
            raise FileNotFoundError(f"no final_model.pt, and no best.pt + model_card.json in {root} or its sub-folders")
        models, cards = zip(*[load_model(d, device) for d in dirs])
        names = [d.name for d in dirs]
    c0 = cards[0]
    for c in cards[1:]:
        if (c["window_months"], c["channels"], c["norm"]) != (c0["window_months"], c0["channels"], c0["norm"]):
            raise ValueError("the weight folders come from different training setups (window, channels or normalisation differ)")
    norm = {k: (v["mean"], v["std"]) for k, v in c0["norm"].items()}
    return list(models), c0, norm, names


def uses_stats(card: dict) -> bool:
    """True when the model was trained with the five yearly-statistic channels."""
    return "st0" in card["channels"]


def confidence_mask(preds, conf, floor: float = 0.4, ignore: int = IGNORE_INDEX):
    """Post-process: set pixels whose ensemble confidence is below `floor` to no-data. Cleaner maps, fewer low-confidence flips.
    Does not change the model; a display/report knob. Returns a copy."""
    out = np.asarray(preds).copy()
    out[np.asarray(conf) < floor] = ignore
    return out


def uses_coh(card: dict) -> bool:
    """True when the model also reads interferometric coherence channels (research variant, not the Sentinel-1-only model)."""
    return "coh" in card["channels"]


def attach_coherence(store: RegionStore, data_dir, region_key: str) -> bool:
    """Load a coherence file (HyP3 preferred, else the Earth Engine layer) for the region into the store. Returns False if none."""
    folder = Path(data_dir) / region_key
    for name, factor in (("coherence_hyp3.npz", 2), ("coherence.npz", 10)):
        f = folder / name
        if f.exists():
            with np.load(f) as z:
                store.coh = z["coherence"].astype(np.float16)
                store.coh_factor = int(z["factor"]) if "factor" in z.files else factor
            return True
    return False


def load_ood_reference(weights_dir) -> dict | None:
    """ood_reference.json written by scripts/evaluate.py (copy it into the weights folder)."""
    import json
    f = Path(weights_dir) / "ood_reference.json"
    return json.loads(f.read_text()) if f.exists() else None


def run_models(models, store: RegionStore, norm, window, device, progress=None, **kw):
    from .data import attach_stats
    from .infer import predict_month

    if kw.get("use_stats") and store.stats is None:
        attach_stats(store)             # order-independent, so a 12-month span over two calendar years is fine
    if kw.get("use_coh") and store.coh is None:
        raise ValueError("this model needs a coherence file for the area, and none was found "
                         "(coherence_hyp3.npz in the region folder). Use the Sentinel-1-only model, or a region that has coherence.")

    preds, confs, members = [], [], []
    months = [i for i in range(store.s1.shape[0]) if store.present[i]]
    for n, m in enumerate(months):
        p, c, ms = predict_month(models, store, m, norm, window, device, **kw)
        preds.append(p); confs.append(c); members.append(np.stack(ms))
        if progress:
            progress((n + 1) / len(months), f"predicted month {n + 1}/{len(months)}")
    return np.stack(preds), np.stack(confs), np.stack(members, axis=1), months


# ------------------------------------------------------------------ results
def analyse(preds, members, labels, smooth: bool) -> dict:
    """Shares, change, uncertainty and noise-floor flags. preds (T,H,W), members (K,T,H,W)."""
    if smooth and len(preds) >= 3:
        preds = temporal_majority(preds, 3)
        members = np.stack([temporal_majority(m, 3) for m in members])
    st = monthly_change_table(list(preds), labels, CLASS_NAMES, IGNORE_INDEX, True)
    churn = st.delta.iloc[1:].abs().sum(axis=1) / 2
    floor = NOISE_FLOOR["monthly_churn_smoothed_pp" if smooth else "monthly_churn_raw_pp"]
    spread = member_spread(members) if members.shape[0] > 1 else None
    net = float(st.total_change.abs().sum() / 2)
    return {
        "preds": preds, "stats": st, "churn": churn, "floor": floor,
        "within_noise": (churn < floor), "net": net, "net_within_noise": net < NOISE_FLOOR["net_first_to_last_pp"],
        "spread": None if spread is None else pd.DataFrame(spread["std"], index=labels, columns=CLASS_NAMES),
    }


def scene_flags(scenes) -> list[bool]:
    """True for months with fewer scenes than half the median month: their composite is noisier."""
    a = np.asarray(scenes, float)
    if not np.isfinite(a).any():
        return [False] * len(a)
    return [bool(np.isfinite(x) and x < 0.5 * np.nanmedian(a)) for x in a]


def compare_with_reference(preds, reference: dict, ignore: int = IGNORE_INDEX) -> pd.DataFrame:
    """Most frequent class over the months against each reference map (pixel agreement, per-class IoU)."""
    counts = np.stack([(preds == k).sum(0) for k in range(N_CLASSES)])
    annual = counts.argmax(0).astype(np.uint8)
    annual[(preds != ignore).sum(0) == 0] = ignore
    rows = {}
    for name, ref in reference.items():
        cm = ConfusionMatrix(N_CLASSES)
        cm.update(annual, ref)
        rows[name] = {"pixels": int(cm.m.sum()), "agreement_%": cm.oa * 100, "mIoU": cm.miou(),
                      **{f"IoU {c}": v for c, v in zip(CLASS_NAMES, cm.iou())}}
    return pd.DataFrame(rows).T


def annual_mode(stack, n_classes: int = N_CLASSES, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Most frequent class per pixel over the months (T,H,W); ignore where no month has a valid class."""
    stack = np.asarray(stack)
    counts = np.stack([(stack == k).sum(0) for k in range(n_classes)])
    out = counts.argmax(0).astype(np.uint8)
    out[counts.sum(0) == 0] = ignore
    return out


def _scores(pred, ref, ignore: int = IGNORE_INDEX) -> dict:
    cm = ConfusionMatrix(N_CLASSES)
    cm.update(pred, ref)
    return {"pixels": int(cm.m.sum()), "agreement_%": cm.oa * 100, "mIoU": cm.miou(),
            **{f"IoU {c}": v for c, v in zip(CLASS_NAMES, cm.iou())}}


def comparison_table(preds, reference: dict, labels=None, ignore: int = IGNORE_INDEX) -> pd.DataFrame:
    """Model, label-source ceiling and majority-class baseline against every reference map.

    labels (T,H,W): the Dynamic World labels of the same months. Dynamic World is what the model was trained to
    imitate, so its own score against WorldCover or Esri is the ceiling for how well the model can match them.
    The baseline paints the whole area with the reference's most frequent class (what a lazy map would score).
    """
    annual = annual_mode(preds, ignore=ignore)
    annual[(np.asarray(preds) != ignore).sum(0) == 0] = ignore
    dw = annual_mode(labels, ignore=ignore) if labels is not None else None
    rows = {}
    for name, ref in reference.items():
        rows[(name, "model")] = _scores(annual, ref)
        if dw is not None:
            rows[(name, "Dynamic World (ceiling)")] = _scores(dw, ref)
        valid = ref[ref != ignore]
        if valid.size:
            top = int(np.bincount(valid.astype(np.int64), minlength=N_CLASSES).argmax())
            rows[(name, f"all {CLASS_NAMES[top]} (baseline)")] = _scores(np.full_like(annual, top), ref)
    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["reference", "map"])
    if "agree" in reference:
        df["share_of_area_%"] = [100 * (reference[n] != ignore).mean() for n, _ in df.index]
    return df


def confusion_vs(preds, ref, ignore: int = IGNORE_INDEX) -> pd.DataFrame:
    """Row-normalised confusion matrix (% of each reference class predicted as each class)."""
    cm = ConfusionMatrix(N_CLASSES)
    cm.update(annual_mode(preds, ignore=ignore), ref)
    tot = cm.m.sum(1, keepdims=True)
    pct = np.where(tot > 0, 100 * cm.m / np.maximum(tot, 1), np.nan)
    return pd.DataFrame(pct, index=[f"{c} (ref)" for c in CLASS_NAMES], columns=[f"as {c}" for c in CLASS_NAMES])


# ------------------------------------------------------------------ map helpers (basemap overlay, clicks)
def bounds_latlon(region: Region):
    """[[south, west], [north, east]] of the region's grid, the form folium wants."""
    w, s, e, n = grid_bounds_lonlat(grid_for_region(region))
    return [[s, w], [n, e]]


def overlay_rgba(pred, colors, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """class ids (H,W) -> uint8 RGBA (H,W,4); no-data pixels are transparent."""
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] + [255] for c in colors], np.uint8)
    pred = np.asarray(pred)
    out = np.zeros((*pred.shape, 4), np.uint8)
    ok = (pred != ignore) & (pred < len(colors))
    out[ok] = pal[pred[ok].astype(int)]
    return out


def highlight_rgba(pred, prev, colors, dim: float = 0.25, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Like overlay_rgba, but pixels whose class equals the previous month are faded; changed pixels stay solid."""
    out = overlay_rgba(pred, colors, ignore)
    same = (np.asarray(pred) == np.asarray(prev)) | (np.asarray(prev) == ignore)
    out[same, 3] = (out[same, 3] * dim).astype(np.uint8)
    return out


def uncovered_share(s1_month) -> float:
    """Share of pixels with no radar value (code 0) in one month (2,H,W uint8)."""
    return float((np.asarray(s1_month)[0] == 0).mean())


def click_to_pixel(lat: float, lon: float, bounds, shape):
    """Map click (lat, lon) to (row, col) of an array drawn into bounds, or None if outside."""
    (s, w), (n, e) = bounds
    if not (s <= lat <= n and w <= lon <= e):
        return None
    h, wd = shape[-2:]
    row = min(int((n - lat) / (n - s) * h), h - 1)
    col = min(int((lon - w) / (e - w) * wd), wd - 1)
    return row, col


def pixel_series(preds, row: int, col: int, labels) -> pd.DataFrame:
    """Class of one pixel in every month."""
    ids = np.asarray(preds)[:, row, col]
    return pd.DataFrame({"month": list(labels),
                         "class": [CLASS_NAMES[i] if i != IGNORE_INDEX else "no data" for i in ids]})


def change_map(first, last, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Class of the last month where it differs from the first month, ignore elsewhere."""
    first, last = np.asarray(first), np.asarray(last)
    out = last.copy()
    out[(first == last) | (first == ignore) | (last == ignore)] = ignore
    return out
