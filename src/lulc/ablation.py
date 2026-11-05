"""Input ablations: what does the trained model rely on?

Each ablation changes part of the normalised model input and measures how much IoU drops against the full input.
A channel set to 0 is set to the training mean (inputs are z-scored), so the model sees "an average pixel" there.
More negative change = the model relies more on that input.

Channel layout of the prepared input (see torch_data): for each month of the window VV, VH; then the extras
(coherence, then st0..st4). `channel_groups` turns that into named groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .classes import IGNORE_INDEX, N_CLASSES
from .data import RegionStore
from .metrics import ConfusionMatrix
from .patches import window_starts
from .torch_data import extra_patch, extra_spec, month_stack, prepare

if TYPE_CHECKING:
    import pandas as pd


def channel_groups(window: int, spec) -> dict[str, list[int]]:
    """Named channel groups of the prepared input. `spec` is torch_data.extra_spec(...)."""
    c = window // 2
    groups = {"centre month VV": [2 * c], "centre month VH": [2 * c + 1], "centre month VV+VH": [2 * c, 2 * c + 1]}
    nb = [i for m in range(window) if m != c for i in (2 * m, 2 * m + 1)]
    if nb:
        groups["neighbour months"] = nb
    off = 2 * window
    stats = []
    for name, n in spec:
        idx = list(range(off, off + n))
        off += n
        if name.startswith("st"):
            stats += idx
            groups[f"yearly statistic {name}"] = idx
        else:
            groups[name] = idx
    if stats:
        groups["all yearly statistics"] = stats
    return groups


def ablate(x: torch.Tensor, kind: str, idx: list[int], window: int, blur_px: int = 2) -> torch.Tensor:
    """Return a copy of the prepared input x (B,C,H,W) with one ablation applied.

    mean        channels idx -> 0 (training mean)
    blur        channels idx -> box blur with radius blur_px (2 px = about 20 m, the true radar resolution)
    from_nb     centre month channels -> mean of the neighbour months' channels (does it read THIS month?)
    """
    out = x.clone()
    if kind == "mean":
        out[:, idx] = 0
    elif kind == "blur":
        k = 2 * blur_px + 1
        out[:, idx] = F.avg_pool2d(x[:, idx], k, stride=1, padding=blur_px, count_include_pad=False)
    elif kind == "from_nb":
        c = window // 2
        nb = [m for m in range(window) if m != c]
        if not nb:
            raise ValueError("from_nb needs a window of 3 or more months")
        for ch in (0, 1):
            out[:, 2 * c + ch] = torch.stack([x[:, 2 * m + ch] for m in nb]).mean(0)
    else:
        raise ValueError(f"unknown ablation kind {kind!r}")
    return out


def default_ablations(window: int, spec) -> dict[str, tuple[str, list[int]]]:
    """name -> (kind, channel indices). 'full input' is the baseline."""
    g = channel_groups(window, spec)
    abl: dict[str, tuple[str, list[int]]] = {"full input": ("mean", [])}
    for name in ("centre month VV", "centre month VH", "centre month VV+VH", "neighbour months", "all yearly statistics"):
        if name in g:
            abl[f"{name} -> average"] = ("mean", g[name])
    if window >= 3:
        abl["centre month -> mean of neighbours"] = ("from_nb", g["centre month VV+VH"])
    abl["centre month blurred (5x5)"] = ("blur", g["centre month VV+VH"])
    for name, idx in g.items():
        if name.startswith("yearly statistic "):
            abl[f"{name} -> average"] = ("mean", idx)
    return abl


@torch.no_grad()
def ablation_confusions(models, store: RegionStore, norm: dict, window: int, device, ablations: dict,
                        months, use_stats: bool, use_coh: bool = False, patch: int = 512, amp: bool = True) -> dict:
    """Confusion matrix per ablation for one region (vs the store's Dynamic World labels), tiles without overlap."""
    h, w = store.size
    patch = min(patch, h, w)
    spec = extra_spec(store.coh.shape[0] if use_coh else 0, use_stats)
    cms = {name: ConfusionMatrix(N_CLASSES) for name in ablations}
    rows, cols = window_starts(h, patch, patch), window_starts(w, patch, patch)
    for m in months:
        if not store.present[m]:
            continue
        for r in rows:
            for c in cols:
                x = torch.from_numpy(month_stack(store, m, window, r, c, patch).copy())[None].to(device)
                ex = (torch.from_numpy(extra_patch(store, r, c, patch, use_coh, use_stats).astype(np.float32))[None].to(device)
                      if (use_coh or use_stats) else None)
                xin = prepare(x, ex, norm, window, spec)
                y = store.label[m, r:r + patch, c:c + patch].copy()
                y[store.s1[m, 0, r:r + patch, c:c + patch] == 0] = IGNORE_INDEX
                if (y != IGNORE_INDEX).sum() == 0:
                    continue
                for name, (kind, idx) in ablations.items():
                    xa = xin if not idx else ablate(xin, kind, idx, window)
                    p = 0
                    for model in models:
                        model.eval()
                        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                            p = p + torch.softmax(model(xa).float(), 1)
                    cms[name].update(p.argmax(1)[0].cpu().numpy(), y)
    return cms


def ablation_table(pooled: dict, class_names) -> pd.DataFrame:
    """Rows = ablations; columns = mIoU, change in mIoU, change in IoU per class, hit hardest."""
    import pandas as pd
    base = pooled["full input"].iou()
    base_m = pooled["full input"].miou()
    rows = []
    for name, cm in pooled.items():
        d = cm.iou() - base
        row = {"taken away": name if name != "full input" else "nothing (baseline)", "mIoU": cm.miou(),
               "d_mIoU": cm.miou() - base_m, **{f"d_{c}": v for c, v in zip(class_names, d)}}
        if name != "full input" and np.isfinite(d).any() and np.nanmin(d) < -0.005:
            k = int(np.nanargmin(d))
            row["hit hardest"] = f"{class_names[k]} {d[k]:+.2f}"
        else:
            row["hit hardest"] = "-"
        rows.append(row)
    return pd.DataFrame(rows)
