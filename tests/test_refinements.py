
import numpy as np
import pytest

from lulc.classes import IGNORE_INDEX, N_CLASSES
from lulc.sharpness import passes_border_rule
from lulc.yearly_labels import agreement_mask, decode_yearly, fill_from_yearly, load_yearly_label, yearly_label_path
from lulc.ee_export import write_atomic

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- ablation
def test_channel_groups_and_ablate():
    from lulc.ablation import ablate, channel_groups
    from lulc.torch_data import extra_spec
    g = channel_groups(3, extra_spec(0, True))
    assert g["centre month VV"] == [2] and g["centre month VH"] == [3]
    assert g["neighbour months"] == [0, 1, 4, 5]
    assert g["all yearly statistics"] == [6, 7, 8, 9, 10] and g["yearly statistic st2"] == [8]
    x = torch.randn(2, 11, 16, 16)
    y = ablate(x, "mean", g["all yearly statistics"], 3)
    assert (y[:, 6:] == 0).all() and torch.equal(y[:, :6], x[:, :6])
    z = ablate(x, "from_nb", g["centre month VV+VH"], 3)
    assert torch.allclose(z[:, 2], (x[:, 0] + x[:, 4]) / 2) and torch.allclose(z[:, 3], (x[:, 1] + x[:, 5]) / 2)
    b = ablate(x, "blur", [2], 3)
    assert b[:, 2].var() < x[:, 2].var() and torch.equal(b[:, 3], x[:, 3])
    assert x.shape == (2, 11, 16, 16) and not torch.equal(y, x)          # the input itself is never changed in place
    with pytest.raises(ValueError):
        ablate(x, "nope", [0], 3)


def test_ablation_confusions_tiny_model():
    from lulc.ablation import ablation_confusions, ablation_table, default_ablations
    from lulc.data import RegionStore
    from lulc.quant import quantize_db
    from lulc.torch_data import extra_spec
    rng = np.random.default_rng(0)
    s1 = quantize_db(rng.uniform(-25, -5, (4, 2, 64, 64)).astype(np.float32))
    label = rng.integers(0, 6, (4, 64, 64)).astype(np.uint8)
    store = RegionStore("r", "val", s1, label, np.ones(4, bool))
    store.stats = rng.normal(size=(5, 64, 64)).astype(np.float16)
    norm = {"vv": (-12, 4), "vh": (-18, 4), **{f"st{i}": (0.0, 1.0) for i in range(5)}}
    model = torch.nn.Conv2d(2 * 3 + 5, N_CLASSES, 1)
    abl = default_ablations(3, extra_spec(0, True))
    cms = ablation_confusions([model], store, norm, 3, torch.device("cpu"), abl, [0, 1, 2], True, patch=64)
    assert set(cms) == set(abl) and all(cm.m.sum() > 0 for cm in cms.values())
    tab = ablation_table(cms, [f"c{i}" for i in range(N_CLASSES)])
    assert tab.loc[0, "d_mIoU"] == 0 and "hit hardest" in tab.columns and len(tab) == len(abl)


# ---------------------------------------------------------------- CRF
def _scene(n=64, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.zeros((n, n), int)
    truth[:, n // 2:] = 1
    probs = np.zeros((2, n, n), np.float32)
    p1 = np.clip(0.5 + (truth - 0.5) * 0.4 + rng.normal(0, noise, (n, n)), 0.02, 0.98)
    probs[1], probs[0] = p1, 1 - p1
    guide = (np.where(truth == 1, -8.0, -16.0) + rng.normal(0, 0.3, (n, n)))[None].astype(np.float32)
    return truth, probs, guide


def test_crf_denoises_and_keeps_a_real_border():
    from lulc.crf import conv_crf
    truth, probs, guide = _scene()
    base = (probs.argmax(0) == truth).mean()
    out = conv_crf(torch.from_numpy(probs), torch.from_numpy(guide), k=7, theta_xy=3.0, theta_i=2.0, strength=4.0).numpy()
    acc = (out.argmax(0) == truth).mean()
    assert acc > base + 0.05 and acc > 0.95
    assert np.allclose(out.sum(0), 1, atol=1e-4)
    # border stays within 1 px: columns well inside each side are right, the jump happens at the true column
    pred = out.argmax(0)
    assert (pred[:, :28] == 0).mean() > 0.95 and (pred[:, 36:] == 1).mean() > 0.95
    assert pred[:, 30].mean() < 0.15 and pred[:, 32].mean() > 0.85          # the jump stays at the true column (32)


def test_crf_with_flat_guide_blurs_a_false_border_more_than_with_an_edge():
    from lulc.crf import conv_crf
    truth, probs, guide = _scene(noise=0.0)
    flat = np.full_like(guide, -12.0)
    kw = dict(k=7, theta_xy=3.0, theta_i=2.0, strength=4.0)
    edge_out = conv_crf(torch.from_numpy(probs), torch.from_numpy(guide), **kw).numpy()[1]
    flat_out = conv_crf(torch.from_numpy(probs), torch.from_numpy(flat), **kw).numpy()[1]
    col = 32                                                        # first column of class 1
    assert edge_out[:, col].mean() > flat_out[:, col].mean()       # the guide's edge protects the border


def test_crf_tiled_equals_full():
    from lulc.crf import conv_crf, conv_crf_tiled
    _, probs, guide = _scene(n=48, seed=3)
    kw = dict(k=5, theta_xy=2.0, theta_i=2.0, strength=3.0, iters=3)
    a = conv_crf(torch.from_numpy(probs), torch.from_numpy(guide), **kw)
    b = conv_crf_tiled(torch.from_numpy(probs), torch.from_numpy(guide), tile=16, **kw)
    assert torch.allclose(a, b, atol=1e-5)


def test_crf_handles_nan_and_bad_k():
    from lulc.crf import conv_crf, yearly_median_db
    from lulc.quant import quantize_db
    _, probs, guide = _scene(n=32)
    probs[:, :4, :4] = np.nan
    guide[:, :4, :4] = np.nan
    out = conv_crf(torch.from_numpy(probs), torch.from_numpy(guide), k=5)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        conv_crf(torch.from_numpy(probs), torch.from_numpy(guide), k=4)
    s1 = quantize_db(np.random.default_rng(0).uniform(-20, -8, (6, 2, 8, 8)).astype(np.float32))
    med = yearly_median_db(s1, np.array([1, 1, 1, 1, 0, 0], bool))
    assert med.shape == (2, 8, 8) and np.isfinite(med).all()


def test_border_rule():
    base = {"miou": 0.5, "bf1": 0.30, "iou": np.array([0.5, 0.5])}
    ok, why = passes_border_rule(base, {"miou": 0.499, "bf1": 0.33, "iou": np.array([0.52, 0.48])})
    assert ok and not why
    ok, why = passes_border_rule(base, {"miou": 0.49, "bf1": 0.31, "iou": np.array([0.5, 0.40])})
    assert not ok and len(why) == 3


# ---------------------------------------------------------------- orbit check
def _month_file(root, region, year, month, direction, n_scenes, has_radar=True):
    vv = np.full((8, 8), -12.0 if has_radar else -9999.0, np.float32)
    arrays = {"vv": vv, "vh": vv - 6, "s1_n": np.full((8, 8), n_scenes if has_radar else 0, np.uint8)}
    write_atomic(root / region / f"{year}-{month:02d}.npz", arrays, {"region": region, "year": year, "month": month, "direction": direction})


def test_orbit_scan_and_summary(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_orbit", "scripts/check_orbit.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for m in range(1, 5):
        _month_file(tmp_path, "good", 2021, m, "DESCENDING", 10)
        _month_file(tmp_path, "good", 2024, m, "DESCENDING", 5)
        _month_file(tmp_path, "bad", 2021, m, "DESCENDING", 10)
        _month_file(tmp_path, "bad", 2024, m, "ASCENDING" if m == 4 else "DESCENDING", 1, has_radar=m != 3)
    (tmp_path / "bad" / "reference.npz").write_bytes(b"not a month file")
    df = mod.scan([tmp_path])
    assert len(df) == 16 and set(df.region) == {"good", "bad"}
    s = mod.summarise(df)
    assert list(s["direction_changes"].index) == ["bad"]
    assert len(s["mixed_in_year"]) == 1 and s["mixed_in_year"].iloc[0].region == "bad"
    assert set(s["low_coverage"].region) == {"bad"}


# ---------------------------------------------------------------- yearly labels / agreement
def test_yearly_label_fill_and_agreement(tmp_path):
    monthly = np.full((3, 4, 4), IGNORE_INDEX, np.uint8)
    monthly[0] = 2
    monthly[1, :2] = 1
    yearly = np.full((4, 4), 2, np.uint8)
    yearly[0, 0] = IGNORE_INDEX
    present = np.array([True, True, False])
    filled, share = fill_from_yearly(monthly, yearly, present)
    assert (filled[2] == IGNORE_INDEX).all()                       # a month without radar is not filled
    assert filled[1, 0, 0] == 1 and filled[1, 3, 3] == 2 and filled[1, 0, 1] == 1
    assert 0 < share < 1
    kept = agreement_mask(monthly, yearly)
    assert (kept[0] == np.where(yearly == 2, 2, IGNORE_INDEX)).all() and (kept[1] == IGNORE_INDEX).all()
    with pytest.raises(ValueError):
        fill_from_yearly(monthly, yearly[:2])
    stack = np.stack([np.where(np.eye(4) > 0, 6.0, np.nan), np.full((4, 4), 87.4)])
    d = decode_yearly(stack)
    assert d["label"].dtype == np.uint8 and d["label"][0, 1] == 255 and d["label"][0, 0] == 6 and d["conf"][0, 0] == 87
    write_atomic(yearly_label_path(tmp_path, "r", 2022), {"label": np.array([[0, 6], [255, 1]], np.uint8), "conf": np.array([[90, 80], [0, 30]], np.uint8)}, {})
    lab = load_yearly_label(tmp_path, "r", 2022)
    assert lab.tolist() == [[5, 3], [255, 0]]                       # DW water->Water, built->Built-up, trees->Tree
    assert load_yearly_label(tmp_path, "r", 2022, conf_cutoff=50).tolist() == [[5, 3], [255, 255]]
    with pytest.raises(FileNotFoundError):
        load_yearly_label(tmp_path, "r", 1999)


# ---------------------------------------------------------------- graph head
def test_graph_head_on_a_synthetic_scene():
    pytest.importorskip("skimage")
    from lulc.graph_head import (build_graph, node_labels, object_vote, pixels_from_nodes, predict_nodes, superpixels, train_head)
    truth, probs, guide = _scene(n=64, noise=0.3, seed=1)
    seg = superpixels(guide, n_segments=60)
    assert seg.shape == truth.shape and seg.min() == 0 and len(np.unique(seg)) == seg.max() + 1
    g = build_graph(seg, probs, guide)
    n = seg.max() + 1
    assert g["A"].shape == (n, n) and torch.allclose(g["A"].sum(1), torch.ones(n), atol=1e-5)
    base = (probs.argmax(0) == truth).mean()
    vote = pixels_from_nodes(seg, object_vote(g))
    assert (vote == truth).mean() > base
    y = node_labels(seg, truth.astype(np.uint8), 2)
    head = train_head([g], [y], 2, epochs=60)
    out = pixels_from_nodes(seg, predict_nodes(head, g))
    assert (out == truth).mean() >= (vote == truth).mean() - 0.02
    assert node_labels(seg, np.full(truth.shape, 255, np.uint8), 2).tolist() == [255] * n
