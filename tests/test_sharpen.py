import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from torch import nn

from lulc.data import RegionStore, load_store
from lulc.ee_export import write_atomic
from lulc.infer import blend_weights, predict_month, softmax_tta
from lulc.train import FullResUnet, build_model, load_model, model_from_state, run

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
NORM = {"vv": (-12.0, 4.0), "vh": (-18.0, 4.0)}


def _store(t=3, h=96):
    s1 = np.random.default_rng(0).integers(1, 255, (t, 2, h, h)).astype(np.uint8)
    return RegionStore(key="a", split="test", s1=s1, label=np.zeros((t, h, h), np.uint8), present=np.ones(t, bool))


def test_tta_output_turns_with_the_input():
    torch.manual_seed(0)
    model = nn.Conv2d(2, 6, 3, padding=1)                       # direction-sensitive, so one pass does not turn with the input
    x = torch.randn(1, 2, 16, 16)
    plain = torch.softmax(model(x), 1)
    assert not torch.allclose(torch.softmax(model(torch.rot90(x, 1, (2, 3))), 1), torch.rot90(plain, 1, (2, 3)), atol=1e-3)
    tta = softmax_tta(model, x, True)
    assert torch.allclose(softmax_tta(model, torch.rot90(x, 1, (2, 3)), True), torch.rot90(tta, 1, (2, 3)), atol=1e-5)
    assert torch.allclose(softmax_tta(model, x.flip(3), True), tta.flip(3), atol=1e-5)
    assert torch.allclose(tta.sum(1), torch.ones(1, 16, 16), atol=1e-5)
    assert torch.equal(softmax_tta(model, x, False), plain)


def test_tta_changes_nothing_for_a_per_pixel_model():
    torch.manual_seed(0)
    model = nn.Conv2d(2, 6, 1)
    x = torch.randn(1, 2, 16, 16)
    assert torch.allclose(softmax_tta(model, x, True), torch.softmax(model(x), 1), atol=1e-5)


def test_blend_weights():
    assert torch.equal(blend_weights(8), torch.ones(8, 8))
    w = blend_weights(64, "gaussian")
    assert w.shape == (64, 64) and w.max() <= 1 and w.min() >= 1e-6
    assert w[32, 32] > w[32, 5] > w[32, 0] > 0 and w[0, 0] == w.min()
    assert torch.allclose(w, w.flip(0)) and torch.allclose(w, w.T)
    with pytest.raises(ValueError):
        blend_weights(8, "cubic")


class _TileOffset(nn.Module):
    """Every tile gets a different flat answer: any jump between neighbouring pixels is a tile seam."""
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        out = torch.zeros(x.shape[0], 6, *x.shape[2:])
        out[:, 0] = float(self.calls % 2) * 2
        return out


def test_gaussian_stitching_softens_tile_seams():
    store = _store()

    def seam(blend):
        p = predict_month([_TileOffset()], store, 1, NORM, 3, torch.device("cpu"), blend=blend, patch=64, overlap=32,
                          return_probs=True, amp=False)[3][0].astype(np.float32)
        return np.abs(np.diff(p, axis=1)).max()

    assert seam("gaussian") < seam("uniform") / 2


def test_default_stitching_matches_a_plain_tile_average():
    """The defaults must leave existing maps as they were: the mean of the tiles covering each pixel."""
    torch.manual_seed(0)
    store = _store()
    model = nn.Conv2d(6, 6, 3, padding=1).eval()
    probs = predict_month([model], store, 1, NORM, 3, torch.device("cpu"), patch=64, overlap=32, return_probs=True, amp=False)[3]
    from lulc.torch_data import extra_spec, prepare
    full = torch.from_numpy(store.s1[[0, 1, 2]].reshape(6, 96, 96))
    acc, cnt = torch.zeros(6, 96, 96), torch.zeros(96, 96)
    with torch.no_grad():
        for r, c in [(0, 0), (0, 32), (32, 0), (32, 32)]:
            x = prepare(full[None, :, r:r + 64, c:c + 64], None, NORM, 3, extra_spec(0, False))
            acc[:, r:r + 64, c:c + 64] += torch.softmax(model(x), 1)[0]
            cnt[r:r + 64, c:c + 64] += 1
    assert np.allclose(probs.astype(np.float32), (acc / cnt).numpy(), atol=1e-3)


def test_predict_month_with_tta_and_blend_keeps_nodata():
    torch.manual_seed(0)
    store = _store()
    store.s1[1, 0, :4, :4] = 0
    pred, conf, members = predict_month([nn.Conv2d(6, 6, 3, padding=1)], store, 1, NORM, 3, torch.device("cpu"), patch=64, overlap=16,
                                        amp=False, tta=True, blend="gaussian")
    assert pred.shape == (96, 96) and (pred[:4, :4] == 255).all() and (pred[4:, 4:] < 6).all() and 0 < conf.max() <= 1
    assert (members[0][:4, :4] == 255).all()


def _cfg(**kw):
    return dict(encoder="resnet18", weights=None, window=3, **kw)


def test_full_res_branch_shape_and_reach():
    torch.manual_seed(0)
    model = build_model(_cfg(full_res=True), 5).eval()
    assert isinstance(model, FullResUnet)
    x = torch.randn(1, 11, 64, 64)
    with torch.no_grad():
        assert model(x).shape == (1, 6, 64, 64)
        hit = x.clone()
        hit[0, 0, 32, 32] += 5
        moved = (model.branch(x) - model.branch(hit)).abs().sum(1)[0] > 1e-6
    # one changed input pixel moves the branch output, and only within two 3x3 convolutions of that pixel: a pixel-level path
    rows, cols = np.nonzero(moved.numpy())
    assert moved[32, 32] and rows.min() >= 30 and rows.max() <= 34 and cols.min() >= 30 and cols.max() <= 34


def test_old_cards_build_the_plain_unet_and_full_res_round_trips():
    plain = build_model(_cfg(), 5)
    assert not isinstance(plain, FullResUnet) and "segmentation_head.0.weight" in plain.state_dict()
    card = {"config": _cfg(), "n_stat": 5}                        # a model card from before full_res existed
    assert not isinstance(model_from_state(card, plain.state_dict(), torch.device("cpu")), FullResUnet)
    fr = build_model(_cfg(full_res=True), 5).eval()
    back = model_from_state({"config": _cfg(full_res=True), "n_stat": 5}, fr.state_dict(), torch.device("cpu"))
    x = torch.randn(1, 11, 32, 32)
    with torch.no_grad():
        assert torch.equal(back(x), fr(x))


def test_v17_differs_from_v12_only_by_the_branch():
    yaml = pytest.importorskip("yaml")
    v = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())["variants"]
    extra = {k: x for k, x in v["V17_window3_fullres"].items() if v["V12_window3_balanced_stats"].get(k) != x}
    assert extra == {"full_res": True}


def _make(tmp, key, seed, h=96):
    rng = np.random.default_rng(seed)
    for m in range(1, 7):
        _, xx = np.mgrid[:h, :h]
        lab = np.where(xx < h // 2, 1, 6).astype(np.uint8)
        vv = np.where(lab == 1, -14.0, -6.0) + rng.normal(0, 1, (h, h))
        write_atomic(tmp / key / f"2021-{m:02d}.npz", {
            "vv": vv.astype(np.float16), "vh": (vv - 7).astype(np.float16), "s1_n": np.full((h, h), 5, np.uint8),
            "dw_label": lab, "dw_conf": np.full((h, h), 90, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}, {})


def test_sharpen_eval_end_to_end(tmp_path, capsys):
    import pandas as pd
    for k, s in (("a", 0), ("v", 2), ("t1", 3)):
        _make(tmp_path / "raw", k, s)
    (tmp_path / "raw" / "gone").mkdir()
    tr = {"a": load_store(tmp_path / "raw", "a", "train", stats=True)}
    va = {"v": load_store(tmp_path / "raw", "v", "val", stats=True)}
    cfg = dict(patch=32, epochs=2, epoch_samples=48, batch=8, workers=0, val_patches=10, amp=False, encoder="resnet18",
               lr=2e-3, window=3, yearly_stats=True, seed=0)
    run(cfg, tr, va, tmp_path / "runs" / "VA" / "seed0", device=torch.device("cpu"), log=lambda *_: None)
    run({**cfg, "full_res": True}, tr, va, tmp_path / "runs" / "VB" / "seed0", device=torch.device("cpu"), log=lambda *_: None)
    assert isinstance(load_model(tmp_path / "runs" / "VB" / "seed0", torch.device("cpu"))[0], FullResUnet)
    yml = tmp_path / "regions.yaml"
    yml.write_text("year: 2021\nhalf_size_km: 12\nregions:\n"
                   "  t1: {name: t1, lat: 13, lon: 10, split: val, half_size_km: 12}\n"
                   "  gone: {name: gone, lat: 40, lon: 10, split: val, half_size_km: 12}\n")
    import sharpen_eval
    base = ["--data", str(tmp_path / "raw"), "--runs", str(tmp_path / "runs"), "--variant", "VA", "--config", str(yml),
            "--months", "1", "4", "--patch", "64", "--overlap", "16"]
    with pytest.raises(FileNotFoundError):
        sharpen_eval.main(base)
    capsys.readouterr()
    assert sharpen_eval.main([*base, "--skip-missing", "--settings", "tta", "gaussian", "tta+gaussian", "--compare-variant", "VB"]) == 0
    text = capsys.readouterr().out
    tab = pd.read_csv(tmp_path / "runs" / "sharpen_VA" / "sharpen.csv")
    assert tab.setting.tolist() == ["none", "tta", "gaussian", "tta+gaussian", "VB"]
    assert {"mIoU", "boundary_F1", "pred/ref border px", "iou_Water"} <= set(tab.columns) and tab.mIoU.notna().all()
    assert "VB: " in text and "Skipped 1 region(s) with no 2021 month files: gone." in text
    with pytest.raises(SystemExit, match="no trained seeds"):
        sharpen_eval.main([*base, "--compare-variant", "nope"])


def test_notebook_variables_survive_percent_run():
    """%run copies a script's globals into the notebook, so a shared name would overwrite the notebook's variable."""
    import json
    import re
    import sharpen_eval
    nb = json.loads((ROOT / "notebooks" / "sharper_borders.ipynb").read_text(encoding="utf8"))
    used = set()
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            used |= set(re.findall(r"^\s*([A-Za-z_]\w*)\s*=", "".join(c["source"]), re.M))
    assert {"BASELINE", "PASSED_STEP", "PASSED_COMPARE"} <= used
    clashes = used & set(vars(sharpen_eval))
    assert clashes <= {"ROOT"}, clashes                    # ROOT, if ever used, is the same repo path on both sides


class _Recorder(nn.Module):
    """Keeps the input it was given and predicts a constant, so a test can read what reached the model."""

    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 6, 1)
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return self.head(x)


def _partial_year(h=64, last=9):
    """Twelve-month store whose months after `last` (0-based) have not happened: their codes are all 0."""
    s1 = np.random.default_rng(0).integers(1, 250, (12, 2, h, h), dtype=np.uint8)
    s1[last + 1:] = 0
    present = np.arange(12) <= last
    return RegionStore(key="a", split="test", s1=s1, label=np.zeros((12, h, h), np.uint8), present=present)


def test_a_future_month_reaches_the_model_as_the_last_real_month():
    store, before = _partial_year(), None
    before = store.s1.copy()
    seen = {}
    for fill in (True, False):
        model = _Recorder(6)
        predict_month([model], store, 9, NORM, 3, torch.device("cpu"), patch=64, overlap=0, amp=False, fill_gaps=fill)
        seen[fill] = model.seen[0]
    assert torch.equal(seen[True][4:6], seen[True][2:4])          # next month = this month, as December has at a year's end
    assert (seen[False][4:6] == 0).all() and (seen[False][2:4] != 0).any()
    assert np.array_equal(store.s1, before)                        # the store is left alone


def test_five_month_window_fills_every_future_month():
    store = _partial_year()
    model = _Recorder(10)
    predict_month([model], store, 9, NORM, 5, torch.device("cpu"), patch=64, overlap=0, amp=False)
    x = model.seen[0]
    assert torch.equal(x[6:8], x[4:6]) and torch.equal(x[8:10], x[4:6]) and not torch.equal(x[0:2], x[4:6])


def test_fill_is_per_pixel_and_never_invents_the_centre_month():
    from lulc.data import fill_window_gaps
    s = np.full((3, 2, 4, 4), 100, np.uint8)
    s[1] = 50                  # centre
    s[2, 0, 0, 0] = 0          # one pixel missing in the next month, VV only
    s[0, :, 1, 1] = 0          # one pixel missing in the previous month
    s[1, :, 2, 2] = 0          # centre has no data here
    s[0, :, 2, 2] = 0          # and neither does the previous month
    out = fill_window_gaps(s)
    assert out[2, 0, 0, 0] == 50 and out[2, 1, 0, 0] == 100 and out[2, 0, 0, 1] == 100
    assert (out[0, :, 1, 1] == 50).all()
    assert (out[1] == s[1]).all() and (out[0, :, 2, 2] == 0).all()
    assert s[2, 0, 0, 0] == 0  # the input is not changed
    one = np.full((1, 2, 4, 4), 7, np.uint8)
    assert np.array_equal(fill_window_gaps(one), one)
