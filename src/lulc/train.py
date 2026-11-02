"""Train one U-Net run. Everything that varies between variants is in the config dict."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Subset

from .classes import CLASS_NAMES, IGNORE_INDEX, N_CLASSES
from .data import build_index, compute_norm
from .metrics import ConfusionMatrix
from .modelcard import build_model_card, write_model_card
from .torch_data import PatchDataset, extra_spec, n_extra, n_input_channels, prepare
from .yearly import STAT_NAMES
from .train_math import class_weights, cosine_warmup_lr, sampling_weights

DEFAULTS = dict(region_balance="none", encoder="resnet34", weights=None, window=1, coherence=False, conf_cutoff=0.0,
                epochs=12, epoch_samples=3000, batch=16, lr=5e-4, weight_decay=1e-4, warmup_frac=0.05,
                patch=256, ce_weight=1.0, dice_weight=0.5, lovasz_weight=0.0, ema=0.999,
                val_patches=600, patience=5, workers=2, amp=True, seed=0, class_weight_power=0.5,
                yearly_stats=False, speckle_p=0.0, speckle_looks=10.0,
                label_smoothing=0.0, db_shift=0.0, month_dropout=0.0, jitter=0, agree_filter=False)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, n_extra_ch: int = 0) -> nn.Module:
    return smp.Unet(encoder_name=cfg["encoder"], encoder_weights=cfg["weights"],
                    in_channels=n_input_channels(cfg["window"], n_extra_ch), classes=N_CLASSES)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.m, self.decay, self.n = copy.deepcopy(model).eval(), decay, 0
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))      # warm start of the average
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def label_frequency(stores: dict, index, step: int = 8) -> np.ndarray:
    counts = np.zeros(N_CLASSES)
    for key, g in index.groupby("region"):
        s = stores[key]
        for m in g.month.unique():
            lab = s.label[int(m), ::step, ::step]
            counts += np.bincount(lab[lab != IGNORE_INDEX], minlength=N_CLASSES)[:N_CLASSES]
    return counts / max(counts.sum(), 1)


@torch.no_grad()
def evaluate(model, loader, norm, window, device, amp=True, spec=None) -> ConfusionMatrix:
    cm = ConfusionMatrix(N_CLASSES)
    model.eval()
    for xb, cb, yb in loader:
        x = prepare(xb.to(device), cb.to(device), norm, window, spec)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = model(x).argmax(1)
        cm.update(pred.cpu().numpy(), yb.numpy())
    return cm


def run(cfg: dict, train_stores: dict, val_stores: dict, out_dir, device=None, log=print) -> dict:
    cfg = {**DEFAULTS, **cfg}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["workers"] > 0:
        # DataLoader workers pass tensors through /dev/shm by default; on the Data Lab that is tiny and the workers
        # die with "Bus error ... out of shared memory". file_system passes them through temp files instead, so
        # workers (and GPU prefetch) work without a big /dev/shm.
        torch.multiprocessing.set_sharing_strategy("file_system")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    tr_idx = pd.concat([build_index(s, cfg["patch"], cfg["patch"]) for s in train_stores.values()], ignore_index=True)
    va_idx = pd.concat([build_index(s, cfg["patch"], cfg["patch"]) for s in val_stores.values()], ignore_index=True)
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError(f"empty index: train {len(tr_idx)}, val {len(va_idx)}")
    va_idx = va_idx.sample(min(cfg["val_patches"], len(va_idx)), random_state=0).sort_values(["region", "month", "row", "col"])
    norm = compute_norm(list(train_stores.values()))
    n_coh = next(iter(train_stores.values())).coh.shape[0] if cfg["coherence"] else 0
    spec = extra_spec(n_coh, cfg["yearly_stats"])
    if cfg["yearly_stats"] and any(s.stats is None for s in [*train_stores.values(), *val_stores.values()]):
        raise RuntimeError("yearly_stats requested but a store has no stats (load with stats=True)")
    freq = label_frequency(train_stores, tr_idx)
    cw = torch.tensor(class_weights(freq, cfg["class_weight_power"]), dtype=torch.float32, device=device)
    log(f"train patches {len(tr_idx)}, val patches {len(va_idx)}, class freq {np.round(freq, 3).tolist()}")

    model = build_model(cfg, n_extra(spec)).to(device)
    ema = EMA(model, cfg["ema"])
    ds_tr = PatchDataset(train_stores, tr_idx, cfg["window"], cfg["patch"], True, cfg["coherence"], cfg["seed"], cfg["yearly_stats"], cfg["jitter"])
    ds_va = PatchDataset({**val_stores}, va_idx, cfg["window"], cfg["patch"], False, cfg["coherence"], 0, cfg["yearly_stats"])
    va_loader = DataLoader(ds_va, batch_size=cfg["batch"] * 2, shuffle=False, num_workers=cfg["workers"])
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=IGNORE_INDEX, label_smoothing=cfg["label_smoothing"])
    dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE_INDEX)
    lov = smp.losses.LovaszLoss("multiclass", ignore_index=IGNORE_INDEX) if cfg["lovasz_weight"] > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")
    steps_per_epoch = max(cfg["epoch_samples"] // cfg["batch"], 1)
    total = steps_per_epoch * cfg["epochs"]
    warm = max(int(total * cfg["warmup_frac"]), 1)

    sw = sampling_weights(tr_idx["region"].tolist(), cfg["region_balance"])
    sw = None if sw is None else torch.as_tensor(sw, dtype=torch.double)
    hist, best, bad, step = [], -1.0, 0, 0
    gen = torch.Generator().manual_seed(cfg["seed"])
    for ep in range(cfg["epochs"]):
        t0 = time.time(); ds_tr.set_epoch(ep)
        if sw is None:
            pick = torch.randperm(len(ds_tr), generator=gen)[:steps_per_epoch * cfg["batch"]].tolist()
        else:
            pick = torch.multinomial(sw, steps_per_epoch * cfg["batch"], replacement=True, generator=gen).tolist()
        loader = DataLoader(Subset(ds_tr, pick), batch_size=cfg["batch"], shuffle=False, num_workers=cfg["workers"],
                            drop_last=True, persistent_workers=cfg["workers"] > 0, prefetch_factor=4 if cfg["workers"] > 0 else None,
                            pin_memory=device.type == "cuda")
        model.train(); run_loss = 0.0; n = 0
        for xb, cb, yb in loader:
            for g in opt.param_groups:
                g["lr"] = cosine_warmup_lr(step, total, warm, cfg["lr"])
            x = prepare(xb.to(device, non_blocking=True), cb.to(device), norm, cfg["window"], spec,
                        cfg["speckle_looks"], cfg["speckle_p"], cfg["db_shift"], cfg["month_dropout"]); y = yb.to(device)
            with torch.autocast(device_type=device.type, enabled=cfg["amp"] and device.type == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss = cfg["ce_weight"] * ce(logits, y)
            if cfg["dice_weight"] > 0: loss = loss + cfg["dice_weight"] * dice(logits, y)
            if lov is not None: loss = loss + cfg["lovasz_weight"] * lov(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); ema.update(model)
            run_loss += loss.item(); n += 1; step += 1
        cm = evaluate(ema.m, va_loader, norm, cfg["window"], device, cfg["amp"], spec)
        row = dict(epoch=ep + 1, train_loss=run_loss / max(n, 1), val_miou=cm.miou(), val_oa=cm.oa,
                   **{f"iou_{c}": v for c, v in zip(CLASS_NAMES, cm.iou())}, seconds=time.time() - t0)
        hist.append(row)
        log(f"ep {ep + 1}/{cfg['epochs']} loss {row['train_loss']:.3f} val mIoU {row['val_miou']:.3f} OA {row['val_oa']:.3f} ({row['seconds']:.0f}s)")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if row["val_miou"] > best:
            best, bad = row["val_miou"], 0
            torch.save(ema.m.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log("early stop"); break
    channels = ["vv", "vh"] + (["coh"] if n_coh else []) + (list(STAT_NAMES) if cfg["yearly_stats"] else [])
    write_model_card(out / "model_card.json", build_model_card(
        list(CLASS_NAMES), channels, {k: norm[k] for k in channels},
        composite={"method": "median_db"}, window_months=cfg["window"], patch=cfg["patch"], ignore_index=IGNORE_INDEX,
        extra={"config": cfg, "n_coh_bands": n_coh, "n_stat": len(STAT_NAMES) if cfg["yearly_stats"] else 0,
               "best_val_miou": best}))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return {"best_val_miou": best, "history": hist, "norm": norm, "out_dir": str(out)}


def model_from_state(card: dict, state: dict, device) -> nn.Module:
    cfg = {**DEFAULTS, **card["config"]}
    cfg["weights"] = None
    model = build_model(cfg, card.get("n_coh_bands", 0) + card.get("n_stat", 0)).to(device)
    model.load_state_dict(state)
    return model.eval()


def load_model(out_dir, device) -> tuple[nn.Module, dict]:
    card = json.loads((Path(out_dir) / "model_card.json").read_text())
    state = torch.load(Path(out_dir) / "best.pt", map_location=device)
    return model_from_state(card, state, device), card


FINAL_NAME = "final_model.pt"


def save_final(seed_dirs, out_path) -> dict:
    """Bundle the best.pt + model card of every seed into ONE file (the weights to hand in)."""
    cards, states = [], []
    for d in seed_dirs:
        cards.append(json.loads((Path(d) / "model_card.json").read_text()))
        states.append(torch.load(Path(d) / "best.pt", map_location="cpu"))
    if not cards:
        raise ValueError("no seed folders to package")
    c0 = cards[0]
    for c in cards[1:]:
        if (c["window_months"], c["channels"], c["norm"]) != (c0["window_months"], c0["channels"], c0["norm"]):
            raise ValueError("seeds come from different training setups")
    torch.save({"format": 1, "cards": cards, "states": states}, out_path)
    return c0


def load_final(path, device):
    bundle = torch.load(path, map_location=device)
    if bundle.get("format") != 1:
        raise ValueError(f"{path} is not a final_model.pt bundle")
    models = [model_from_state(c, s, device) for c, s in zip(bundle["cards"], bundle["states"])]
    return models, bundle["cards"]
