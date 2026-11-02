"""Visual alignment check: label edges drawn over the Sentinel-1 VV image.

    python scripts/qa_overlay.py data/raw/heidelberg/2021-07.npz --out reports/qa

Writes a PNG with three panels (VV, label, VV with label edges) and a second
one zoomed into the centre. Open them and check that water edges, rivers and
the urban fringe in the label sit on the matching structures in the VV image.
Also reports the pixel shift (-3..+3) at which label edges best match VV edges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from lulc.classes import CLASS_COLORS, IGNORE_INDEX, remap_dynamic_world  # noqa: E402


def label_edges(lbl: np.ndarray) -> np.ndarray:
    e = np.zeros(lbl.shape, bool)
    e[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    e[1:, :] |= lbl[1:, :] != lbl[:-1, :]
    return e


def best_shift(vv: np.ndarray, edges: np.ndarray, max_shift: int = 3) -> tuple[int, int]:
    """Shift of the label edges that maximises mean VV gradient under them."""
    gy, gx = np.gradient(np.nan_to_num(vv.astype(np.float32)))
    grad = np.hypot(gx, gy)
    best, arg = -1.0, (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            sh = np.roll(np.roll(edges, dy, 0), dx, 1)
            score = float(grad[sh].mean()) if sh.any() else 0.0
            if score > best:
                best, arg = score, (dy, dx)
    return arg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--out", default="reports/qa")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    z = np.load(args.npz)
    vv = z["vv"].astype(np.float32)
    lbl = remap_dynamic_world(z["dw_label"])
    cmap = ListedColormap(CLASS_COLORS)
    shown = np.ma.masked_where(lbl == IGNORE_INDEX, lbl)
    edges = label_edges(np.where(lbl == IGNORE_INDEX, 0, lbl))
    dy, dx = best_shift(vv, edges)
    print(f"best edge shift (rows, cols): ({dy}, {dx})  -> expect (0, 0) or within +-1")

    n = vv.shape[0]
    c = n // 2
    for tag, sl in (("full", slice(None)), ("zoom", slice(c - 300, c + 300))):
        v, s, e = vv[sl, sl], shown[sl, sl], edges[sl, sl]
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(v, cmap="gray", vmin=-25, vmax=0); ax[0].set_title("Sentinel-1 VV (dB)")
        ax[1].imshow(s, cmap=cmap, vmin=0, vmax=5, interpolation="nearest"); ax[1].set_title("Dynamic World label")
        ax[2].imshow(v, cmap="gray", vmin=-25, vmax=0)
        ov = np.zeros((*e.shape, 4)); ov[e] = (1, 0.1, 0.1, 0.9)
        ax[2].imshow(ov, interpolation="nearest"); ax[2].set_title("VV + label edges")
        for a in ax:
            a.axis("off")
        out = Path(args.out) / f"{Path(args.npz).parent.name}_{Path(args.npz).stem}_{tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
