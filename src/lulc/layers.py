"""Extra map layers, the month-by-month animation and the model-card summary for the prototype.

Everything here is plain numpy / Pillow so it can be tested without Streamlit or Earth Engine.
Class ids are 0..5, 255 = no radar (IGNORE_INDEX), as everywhere else.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .classes import CLASS_NAMES, IGNORE_INDEX

# Change-frequency legend: 0 changes is transparent, then 1, 2, 3, 4 or more (light to dark, one hue, readable in grayscale).
FREQ_COLORS = ["#fde8c8", "#f9b872", "#ec7a2e", "#a63603"]
FREQ_LABELS = ["1 change", "2 changes", "3 changes", "4 or more"]
FREQ_CAPTION = ("Counts how often a pixel switches class between consecutive months. Repeated switching is mostly a seasonal "
                "radar effect (soil moisture, snow, harvest, speckle), not real land-use change. Sentinel-1 VV, no ground truth.")

LAYER_NAMES = ["Land cover", "12-month majority", "Model confidence", "Change since first month",
               "Change frequency (year)", "Sentinel-1 VV (dB)"]


def _hex_rgb(c: str) -> tuple[int, int, int]:
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def majority_map(preds, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Most frequent class per pixel over all months (T,H,W) -> (H,W); ignore where no month has radar."""
    preds = np.asarray(preds)
    counts = np.stack([(preds == k).sum(0) for k in range(len(CLASS_NAMES))])
    out = counts.argmax(0).astype(np.uint8)
    out[counts.sum(0) == 0] = ignore
    return out


def switch_count(preds, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Number of class switches per pixel between consecutive months with radar (T,H,W) -> (H,W) int.

    A month with no radar is skipped, not counted as a change: the pixel is compared with its last observed class."""
    preds = np.asarray(preds)
    last = np.full(preds.shape[1:], ignore, np.uint8)
    n = np.zeros(preds.shape[1:], np.int16)
    for p in preds:
        seen = p != ignore
        n += (seen & (last != ignore) & (p != last)).astype(np.int16)
        last = np.where(seen, p, last)
    return n


def change_since_first(preds, first: int = 0, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Per month: the class of the pixel where it differs from the first month, ignore elsewhere. (T,H,W)."""
    preds = np.asarray(preds)
    base = preds[first]
    out = preds.copy()
    out[(preds == base[None]) | (base[None] == ignore)] = ignore
    return out


def agreement_confidence(members, preds, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Share of ensemble members that give the final class, (K,T,H,W) + (T,H,W) -> (T,H,W) in 0..1.

    Used when the softmax confidence was not saved (older saved runs). It is a vote share, not a probability."""
    members, preds = np.asarray(members), np.asarray(preds)
    conf = (members == preds[None]).mean(0).astype(np.float32)
    conf[preds == ignore] = np.nan
    return conf


def vote_confidence(members, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Share of members that agree with the most common class per pixel, (K,T,H,W) -> (T,H,W). With one member it is 1 everywhere."""
    members = np.asarray(members)
    counts = np.stack([(members == k).sum(0) for k in range(len(CLASS_NAMES))])
    top = counts.max(0).astype(np.float32) / members.shape[0]
    top[(members == ignore).all(0)] = np.nan
    return top


def confidence_rgba(conf, ignore_mask=None, lo: float = 0.4, hi: float = 1.0) -> np.ndarray:
    """Confidence (H,W) in 0..1 -> RGBA. Dark = unsure, light = sure; no-data transparent."""
    conf = np.asarray(conf, np.float32)
    t = np.nan_to_num(np.clip((conf - lo) / (hi - lo), 0, 1))
    shade = (40 + 200 * t).astype(np.uint8)
    out = np.zeros((*conf.shape, 4), np.uint8)
    ok = np.isfinite(conf)
    if ignore_mask is not None:
        ok &= ~np.asarray(ignore_mask)
    out[..., :3] = shade[..., None]
    out[..., 3] = np.where(ok, 255, 0)
    return out


def frequency_rgba(n) -> np.ndarray:
    """Switch count (H,W) -> RGBA: 0 transparent, 1..3 as is, 4+ share the darkest colour."""
    n = np.asarray(n)
    out = np.zeros((*n.shape, 4), np.uint8)
    for k, c in enumerate(FREQ_COLORS, start=1):
        m = (n == k) if k < len(FREQ_COLORS) else (n >= k)
        out[m] = (*_hex_rgb(c), 255)
    return out


def frequency_shares(n, valid) -> dict:
    """Share of valid pixels per legend row (0, 1, 2, 3, 4+), in percent."""
    n, valid = np.asarray(n), np.asarray(valid)
    tot = max(int(valid.sum()), 1)
    rows = {"0 changes": 100 * float(((n == 0) & valid).sum()) / tot}
    for k, lab in enumerate(FREQ_LABELS, start=1):
        m = (n == k) if k < len(FREQ_LABELS) else (n >= k)
        rows[lab] = 100 * float((m & valid).sum()) / tot
    return rows


def vv_rgba(s1_month, lo: float = -25.0, hi: float = 0.0) -> np.ndarray:
    """VV backscatter of one month (2,H,W uint8 codes) -> grey RGBA in dB. Code 0 (no radar) is transparent."""
    from .quant import dequantize_db
    db = dequantize_db(np.asarray(s1_month)[0])
    t = np.clip((db - lo) / (hi - lo), 0, 1)
    g = np.nan_to_num(t * 255).astype(np.uint8)
    out = np.zeros((*db.shape, 4), np.uint8)
    out[..., :3] = g[..., None]
    out[..., 3] = np.where(np.isfinite(db), 255, 0)
    return out


# ------------------------------------------------------------------ animation
def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(pred, prev, colors, label: str, mode: str = "Land cover", scale: int = 2, no_data=(200, 200, 200)):
    """One animation frame as a PIL image: header with the month, the map, a one-line legend.

    mode "Land cover" paints every class; mode "Changes" shows only pixels whose class differs from the previous month
    (all others grey), so the flicker is visible instead of hidden."""
    from PIL import Image, ImageDraw

    from .proto import overlay_rgba
    pred = np.asarray(pred)
    if mode == "Changes" and prev is not None:
        rgba = overlay_rgba(pred, colors)
        same = (pred == np.asarray(prev)) | (np.asarray(prev) == IGNORE_INDEX) | (pred == IGNORE_INDEX)
        rgba[same] = (*no_data, 255)
    else:
        rgba = overlay_rgba(pred, colors)
        rgba[pred == IGNORE_INDEX] = (*no_data, 255)
    img = Image.fromarray(rgba[..., :3]).resize((pred.shape[1] * scale, pred.shape[0] * scale), Image.NEAREST)
    pad_top, pad_bot = 34, 30
    canvas = Image.new("RGB", (img.width, img.height + pad_top + pad_bot), (255, 255, 255))
    canvas.paste(img, (0, pad_top))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), label + ("  (pixels that changed since the previous month)" if mode == "Changes" and prev is not None else ""),
           fill=(20, 20, 20), font=_font(15))
    x, y = 8, img.height + pad_top + 8
    small = _font(11)
    for c, n in zip(colors, CLASS_NAMES):
        d.rectangle([x, y, x + 11, y + 11], fill=_hex_rgb(c), outline=(120, 120, 120))
        d.text((x + 15, y - 1), n, fill=(40, 40, 40), font=small)
        x += 15 + int(d.textlength(n, font=small)) + 12
    return canvas


def animation_gif(preds, labels, colors, mode: str = "Land cover", fps: float = 2.0, scale: int = 2,
                  max_side: int = 640) -> bytes:
    """Animated GIF with one frame per month. fps 0.5..8. Frames are shrunk so the longest side stays under max_side px."""
    preds = np.asarray(preds)
    if len(preds) < 2:
        raise ValueError("an animation needs at least two months")
    if len(labels) != len(preds):
        raise ValueError("one label per month is needed")
    if mode not in ("Land cover", "Changes"):
        raise ValueError(f"unknown mode {mode!r}")
    scale = max(1, min(scale, max_side // max(preds.shape[1:])))
    frames = [render_frame(p, None if i == 0 else preds[i - 1], colors, lab, mode, scale)
              for i, (p, lab) in enumerate(zip(preds, labels))]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=int(1000 / max(fps, 0.1)), loop=0, optimize=False)
    return buf.getvalue()


# ------------------------------------------------------------------ model card
def card_summary(card: dict, names: list[str] | None = None) -> dict:
    """Plain facts from model_card.json for display. Missing keys come back as None, never an error."""
    cfg = card.get("config") or {}
    comp = card.get("composite") or {}
    ch = card.get("channels") or []
    return {
        "architecture": f"U-Net, {cfg.get('encoder', 'resnet34')} encoder (segmentation_models_pytorch)",
        "channels": ch,
        "n_channels": len(ch),
        "window_months": card.get("window_months"),
        "patch": card.get("patch"),
        "classes": card.get("class_names") or list(CLASS_NAMES),
        "ignore_index": card.get("ignore_index"),
        "composite": comp,
        "ensemble": names or [],
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch"),
        "seed": cfg.get("seed"),
        "train_years": cfg.get("train_years") or card.get("train_years"),
        "norm": card.get("norm") or {},
        "uses_stats": "st0" in ch,
        "uses_coherence": "coh" in ch,
    }


def read_train_years(weights_dir) -> dict | None:
    """{"train": [...], "val": [...]} from the first years.json in the weights folder or its sub-folders, else None."""
    root = Path(weights_dir)
    if not root.exists():
        return None
    for f in [root / "years.json", *sorted(root.glob("*/years.json"))]:
        if f.exists():
            try:
                return json.loads(f.read_text())
            except ValueError:
                return None
    return None


def find_eval_dir(results_dir, variant: str = "V12_window3_balanced_stats") -> Path | None:
    """results/runs/eval_<variant>, or the first results/**/eval_<variant>* that has metrics.json."""
    root = Path(results_dir)
    if not root.exists():
        return None
    for cand in [root / "runs" / f"eval_{variant}", *sorted(root.glob(f"**/eval_{variant}"))]:
        if (cand / "metrics.json").exists():
            return cand
    return None


def load_scores(eval_dir) -> dict:
    """Headline scores from an evaluate.py output folder: {key: {miou, oa, per_class_iou}}; empty if nothing usable."""
    f = Path(eval_dir) / "metrics.json"
    if not f.exists():
        return {}
    raw = json.loads(f.read_text())
    out = {}
    for k in ("dw", "dw_cities", "dw_rural", "wc", "esri"):
        v = raw.get(k)
        if isinstance(v, dict) and "miou" in v:
            out[k] = {"miou": v["miou"], "miou_lo": v.get("miou_lo"), "miou_hi": v.get("miou_hi"), "oa": v.get("oa"),
                      "per_class_iou": v.get("per_class_iou") or {}}
    return out
