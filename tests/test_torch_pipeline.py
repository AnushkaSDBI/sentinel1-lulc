import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from lulc.change_eval import false_change
from lulc.data import load_store
from lulc.ee_export import write_atomic
from lulc.infer import predict_region
from lulc.train import load_model, run


def _make(tmp, key, seed, h=96):
    rng = np.random.default_rng(seed)
    for m in range(1, 5):
        yy, xx = np.mgrid[:h, :h]
        lab = np.where(xx < h // 2, 1, 6).astype(np.uint8)          # DW trees | built
        vv = np.where(lab == 1, -14.0, -6.0) + rng.normal(0, 1, (h, h))
        vh = vv - 7
        write_atomic(tmp / key / f"2021-{m:02d}.npz", {
            "vv": vv.astype(np.float16), "vh": vh.astype(np.float16), "s1_n": np.full((h, h), 5, np.uint8),
            "dw_label": lab, "dw_conf": np.full((h, h), 90, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}, {})


def test_end_to_end_learns_trivial_task(tmp_path):
    for k, s in (("a", 0), ("b", 1), ("v", 2)):
        _make(tmp_path, k, s)
    tr = {k: load_store(tmp_path, k, "train") for k in "ab"}
    va = {"v": load_store(tmp_path, "v", "val")}
    res = run(dict(patch=32, epochs=3, epoch_samples=64, batch=8, workers=0, val_patches=20, amp=False,
                   encoder="resnet18", lr=2e-3, window=3), tr, va, tmp_path / "run", device=torch.device("cpu"), log=lambda *_: None)
    assert res["best_val_miou"] > 0.8
    model, card = load_model(tmp_path / "run", torch.device("cpu"))
    preds, conf, members, months = predict_region([model], va["v"], res["norm"], 3, torch.device("cpu"), patch=64, overlap=16)
    assert preds.shape == (4, 96, 96) and members.shape == (1, 4, 96, 96) and months == [0, 1, 2, 3]
    assert (preds[0][:, :40] == 0).mean() > 0.9                     # trees side
    assert false_change(preds, np.ones((96, 96), bool))["flip_rate_pct"] < 5
