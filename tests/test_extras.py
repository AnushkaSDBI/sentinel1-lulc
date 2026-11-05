import numpy as np
import pytest

from lulc.quant import quantize_db
from lulc.stats import monthly_change_table
from lulc.yearly import yearly_stats

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from lulc.data import load_store
from lulc.ee_export import write_atomic
from lulc.infer import predict_region
from lulc.torch_data import add_speckle, extra_spec, n_extra
from lulc.train import load_model, run


def test_yearly_stats_values_and_order_independence():
    rng = np.random.default_rng(0)
    db = rng.uniform(-25, -5, (12, 2, 8, 8)).astype(np.float32)
    s1 = quantize_db(db)
    a = yearly_stats(s1)
    perm = rng.permutation(12)
    b = yearly_stats(s1[perm])
    assert a.shape == (5, 8, 8)
    np.testing.assert_allclose(a.astype(np.float32), b.astype(np.float32), atol=1e-3)
    vv = db[:, 0]
    assert np.allclose(a[0], vv.min(0), atol=0.3) and np.allclose(a[1], vv.max(0), atol=0.3)
    assert np.allclose(a[4], (vv < -18).mean(0), atol=0.09)          # 8-bit steps shift a few borderline months


def test_yearly_stats_nan_with_too_few_months():
    s1 = quantize_db(np.full((12, 2, 4, 4), -12.0, np.float32))
    present = np.zeros(12, bool); present[:3] = True
    assert np.isnan(yearly_stats(s1, present)).all()


def test_speckle_statistics():
    torch.manual_seed(0)
    db = torch.full((2, 2, 64, 64), -12.0)
    valid = torch.ones_like(db, dtype=torch.bool)
    out = add_speckle(db, valid, looks=10.0, p=1.0)
    d = (out - db).flatten()
    # Gamma(L, 1/L) in dB: mean slightly negative (~ -0.2 dB at L=10), std ~ 1.3 dB
    assert -0.5 < d.mean() < 0.0 and 1.0 < d.std() < 1.7
    assert torch.equal(add_speckle(db, valid, 10.0, 0.0), db)         # p=0 leaves data alone
    nan = db.clone(); nan[0, 0, 0, 0] = float("nan")
    v = torch.isfinite(nan)
    assert torch.isnan(add_speckle(nan, v, 10.0, 1.0)[0, 0, 0, 0])    # no data stays no data


def test_extra_spec():
    assert n_extra(extra_spec(0, False)) == 0
    assert n_extra(extra_spec(2, True)) == 7


def _make(tmp, key, seed, h=96):
    rng = np.random.default_rng(seed)
    for m in range(1, 7):
        yy, xx = np.mgrid[:h, :h]
        lab = np.where(xx < h // 2, 1, 6).astype(np.uint8)
        vv = np.where(lab == 1, -14.0, -6.0) + rng.normal(0, 1, (h, h))
        write_atomic(tmp / key / f"2021-{m:02d}.npz", {
            "vv": vv.astype(np.float16), "vh": (vv - 7).astype(np.float16), "s1_n": np.full((h, h), 5, np.uint8),
            "dw_label": lab, "dw_conf": np.full((h, h), 90, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}, {})


def test_train_and_predict_with_stats_speckle_lovasz(tmp_path):
    for k, s in (("a", 0), ("b", 1), ("v", 2)):
        _make(tmp_path, k, s)
    tr = {k: load_store(tmp_path, k, "train", stats=True) for k in "ab"}
    va = {"v": load_store(tmp_path, "v", "val", stats=True)}
    cfg = dict(patch=32, epochs=3, epoch_samples=64, batch=8, workers=0, val_patches=20, amp=False, encoder="resnet18",
               lr=2e-3, window=3, yearly_stats=True, speckle_p=0.5, speckle_looks=10.0,
               class_weight_power=0.75, lovasz_weight=0.5)
    res = run(cfg, tr, va, tmp_path / "run", device=torch.device("cpu"), log=lambda *_: None)
    assert res["best_val_miou"] > 0.7
    model, card = load_model(tmp_path / "run", torch.device("cpu"))
    assert card["n_stat"] == 5
    preds, conf, members, months = predict_region([model], va["v"], res["norm"], 3, torch.device("cpu"),
                                                  patch=64, overlap=16, use_stats=True)
    assert preds.shape[1:] == (96, 96) and (preds[0][:, :40] == 0).mean() > 0.85


def test_change_table_invariants():
    rng = np.random.default_rng(1)
    preds = rng.integers(0, 6, (12, 40, 40))
    preds[3, :5] = 255                                               # a month with missing pixels
    t = monthly_change_table(list(preds), [f"m{i}" for i in range(12)])
    assert np.allclose(t.percent.sum(1), 100, atol=1e-6)
    assert np.allclose(t.delta.iloc[1:].sum(1), 0, atol=1e-6)
    assert np.allclose(t.delta.iloc[1:].sum(0), t.total_change, atol=1e-6)   # steps add up to the total
    assert (t.valid_fraction < 1).all()                              # common mask applied to every month


def test_ood_signature_and_check():
    from lulc.ood import area_signature, ood_check
    members = np.zeros((3, 2, 10, 10), int)
    members[1, :, :5] = 1                                           # one seed disagrees on half the pixels
    sig = area_signature(np.full((2, 10, 10), 0.8, np.float16), members)
    assert abs(sig["mean_conf"] - 0.8) < 1e-3 and abs(sig["disagree"] - 0.5) < 1e-6
    ref = {"in_dist": [{"mean_conf": 0.7, "disagree": 0.6}, {"mean_conf": 0.9, "disagree": 0.4}]}
    assert ood_check(sig, ref)["level"] == "ok"
    assert ood_check({"mean_conf": 0.5, "disagree": 0.1}, ref)["level"] == "low"
    assert ood_check({"mean_conf": 0.9, "disagree": 0.7}, ref)["level"] == "low"
    assert ood_check(sig, None)["level"] == "unknown"


def test_regions_yaml_climate_and_new_tiles():
    from pathlib import Path
    from lulc.config import load_config
    _, regions = load_config(Path(__file__).resolve().parent.parent / "configs" / "regions.yaml")
    by = {r.key: r for r in regions}
    assert all(r.climate for r in regions)                                   # every region has a group
    new = ["pampas", "nsw_pasture", "kenya_savanna", "mongolia_steppe", "arabia_desert"]
    assert all(by[k].split == "train" for k in new)
    assert sum(by[k].lat < 0 for k in new) == 3                              # southern hemisphere present in training


def test_comparison_table_ceiling_and_baseline():
    from lulc.proto import comparison_table, confusion_vs
    h = 20
    ref = np.zeros((h, h), np.uint8); ref[:, 10:] = 3                       # half tree, half built
    preds = np.stack([ref] * 3)
    labels = np.stack([ref.copy()] * 3); labels[:, :2] = 1                  # label source differs on two rows
    t = comparison_table(preds, {"wc": ref, "agree": ref}, labels)
    assert t.loc[("wc", "model"), "agreement_%"] == 100
    assert 80 < t.loc[("wc", "Dynamic World (ceiling)"), "agreement_%"] < 100
    base = [i for i in t.index if i[0] == "wc" and "baseline" in i[1]]
    assert len(base) == 1 and abs(t.loc[base[0], "agreement_%"] - 50) < 1e-6
    assert t.loc[("agree", "model"), "share_of_area_%"] == 100
    cm = confusion_vs(preds, ref)
    assert abs(cm.loc["Tree cover (ref)", "as Tree cover"] - 100) < 1e-9


def test_map_helpers():
    from lulc.classes import CLASS_COLORS
    from lulc.proto import change_map, click_to_pixel, overlay_rgba, pixel_series
    pred = np.array([[0, 3], [255, 5]], np.uint8)
    rgba = overlay_rgba(pred, CLASS_COLORS)
    assert rgba.shape == (2, 2, 4) and rgba[1, 0, 3] == 0 and rgba[0, 0, 3] == 255
    assert tuple(rgba[0, 1, :3]) == (0x4d, 0x4d, 0x4d)                 # built-up is dark grey
    b = [[49.0, 8.0], [49.1, 8.2]]
    assert click_to_pixel(49.05, 8.1, b, (100, 100)) == (50, 50)
    assert click_to_pixel(49.2, 8.1, b, (100, 100)) is None
    assert click_to_pixel(49.1, 8.2, b, (100, 100)) == (0, 99)
    ps = pixel_series(np.stack([pred, pred]), 1, 0, ["a", "b"])
    assert list(ps["class"]) == ["no data", "no data"]
    cm = change_map(np.array([[0, 3]]), np.array([[0, 5]]))
    assert list(cm[0]) == [255, 5]


def test_prepare_db_shift_and_month_dropout():
    from lulc.torch_data import prepare
    torch.manual_seed(0)
    norm = {"vv": (-10.0, 5.0), "vh": (-17.0, 5.0)}
    x = torch.full((64, 6, 8, 8), 100, dtype=torch.uint8)                # 3 months, all valid
    base = prepare(x, torch.zeros(64, 0, 8, 8), norm, 3)
    drop = prepare(x, torch.zeros(64, 0, 8, 8), norm, 3, month_dropout=0.6)
    blanked = (drop != base).view(64, 3, 2, 8, 8).any((2, 3, 4))             # True where a month was blanked
    assert blanked.sum() > 0
    assert blanked.sum(1).max() <= 2                                        # at least one month always stays
    shifted = prepare(x, torch.zeros(64, 0, 8, 8), norm, 3, db_shift=2.0)
    delta = (shifted - base) * 5.0                                          # back to dB
    assert delta.abs().max() <= 2.0 + 1e-4 and delta.abs().max() > 0.5
    assert torch.allclose(delta[:, 0], delta[:, 2]) and torch.allclose(delta[:, 1], delta[:, 3])   # same offset in every month
    nodata = torch.zeros(4, 6, 8, 8, dtype=torch.uint8)
    assert prepare(nodata, torch.zeros(4, 0, 8, 8), norm, 3, db_shift=2.0).abs().max() == 0   # no-data stays no-data


def test_jitter_stays_inside_tile(tmp_path):
    import pandas as pd
    from lulc.data import RegionStore
    from lulc.torch_data import PatchDataset
    h = 96
    st = RegionStore(key="a", split="train", s1=np.full((2, 2, h, h), 50, np.uint8), label=np.zeros((2, h, h), np.uint8),
                     present=np.array([True, True]), year=2021, meta={})
    idx = pd.DataFrame({"region": ["a"], "month": [0], "row": [64], "col": [0]})
    ds = PatchDataset({"a": st}, idx, window=1, patch=32, augment=True, jitter=40)
    for e in range(30):
        ds.set_epoch(e)
        x, _, y = ds[0]
        assert x.shape[-2:] == (32, 32) and y.shape == (32, 32)


def test_agree_filter(tmp_path):
    from lulc.data import RegionStore, apply_agree_filter
    h = 16
    wc = np.full((h, h), 10, np.uint8); wc[:, 8:] = 50            # WorldCover tree | built
    np.savez(tmp_path / "a" / "reference.npz", wc=wc, esri=wc) if (tmp_path / "a").mkdir() is None else None
    lab = np.zeros((3, h, h), np.uint8); lab[:, :, 12:] = 3        # model label: tree | built only from col 12
    st = RegionStore(key="a", split="train", s1=np.zeros((3, 2, h, h), np.uint8), label=lab, present=np.ones(3, bool), year=2021, meta={})
    removed = apply_agree_filter(st, tmp_path)
    assert (st.label[:, :, 8:12] == 255).all() and (st.label[:, :, :8] == 0).all() and (st.label[:, :, 12:] == 3).all()
    assert abs(removed - 0.25) < 1e-9
    with pytest.raises(FileNotFoundError):
        apply_agree_filter(RegionStore(key="zz", split="train", s1=st.s1, label=lab, present=st.present, year=2021, meta={}), tmp_path)


def test_train_with_regularisation_options(tmp_path):
    for k, s in (("a", 0), ("v", 2)):
        _make(tmp_path, k, s)
    tr = {"a": load_store(tmp_path, "a", "train")}
    va = {"v": load_store(tmp_path, "v", "val")}
    cfg = dict(patch=32, epochs=2, epoch_samples=48, batch=8, workers=0, val_patches=10, amp=False, encoder="resnet18",
               lr=2e-3, window=3, db_shift=1.5, month_dropout=0.15, jitter=16, label_smoothing=0.1, weight_decay=0.05)
    res = run(cfg, tr, va, tmp_path / "run", device=torch.device("cpu"), log=lambda *_: None)
    assert np.isfinite(res["best_val_miou"])


def test_train_with_region_balance(tmp_path):
    for k, s in (("a", 0), ("b", 1), ("v", 2)):
        _make(tmp_path, k, s)
    tr = {k: load_store(tmp_path, k, "train") for k in "ab"}
    va = {"v": load_store(tmp_path, "v", "val")}
    cfg = dict(patch=32, epochs=2, epoch_samples=48, batch=8, workers=0, val_patches=10, amp=False, encoder="resnet18",
               lr=2e-3, window=3, region_balance="equal")
    res = run(cfg, tr, va, tmp_path / "run", device=torch.device("cpu"), log=lambda *_: None)
    assert np.isfinite(res["best_val_miou"])


def test_check_period_and_years():
    import datetime as dt
    import numpy as np
    import pytest
    from lulc import proto

    today = dt.date(2026, 10, 29)
    assert proto.check_period(proto.month_sequence(2021, 1), today) == []
    w = proto.check_period(proto.month_sequence(2024, 1), today)
    assert any("1B" in x for x in w)
    with pytest.raises(ValueError):
        proto.check_period(proto.month_sequence(2014, 6), today)
    with pytest.raises(ValueError):
        proto.check_period(proto.month_sequence(2026, 1), today)       # runs into Dec 2026
    # the current year runs up to the last finished month, and a full past period is untouched
    now = dt.date(2026, 11, 4)
    done = proto.finished_months(proto.month_sequence(2026, 1), now)
    assert done[0] == (2026, 1) and done[-1] == (2026, 10) and len(done) == 10
    assert proto.check_period(done, now)                               # passes the period check (with its usual warnings)
    assert proto.finished_months(proto.month_sequence(2021, 1), now) == proto.month_sequence(2021, 1)
    assert proto.latest_full_start(now) == (2025, 11)
    assert len(proto.finished_months(proto.month_sequence(*proto.latest_full_start(now)), now)) == 12
    assert len(proto.finished_months(proto.month_sequence(2026, 7), now)) == 4     # the smallest period allowed
    with pytest.raises(ValueError, match="Nov 2025"):
        proto.finished_months(proto.month_sequence(2026, 8), now)      # only Aug, Sep, Oct have finished
    a = np.zeros((2, 4, 4), np.uint8); a[:, :2] = 3
    b = a.copy(); b[:, :1] = 0
    s = proto.yearly_summary({2021: a, 2022: b})
    assert abs(s["shares"].loc[2021, "Built-up"] - 50) < 1e-9 and abs(s["shares"].loc[2022].sum() - 100) < 1e-9
    t = proto.transition_matrix(s["maps"][2021], s["maps"][2022])
    assert abs(t.loc["Built-up", "Tree cover"] - 25) < 1e-9 and abs(t.values.sum() - 100) < 1e-9
    assert proto.make_region(-33.9, 151.2, 5.12)


def test_southern_hemisphere_and_highlight():
    import numpy as np
    from lulc import proto
    from lulc.grid import grid_for_region, utm_epsg

    assert utm_epsg(-33.9, 18.4) == 32734
    r = proto.make_region(-33.9, 18.4, 5.12)
    g = grid_for_region(r)
    assert g.epsg == "EPSG:32734" or g.epsg == 32734
    lon0, lat0, lon1, lat1 = proto.grid_bounds_lonlat(g) if hasattr(proto, "grid_bounds_lonlat") else (0, -34, 0, -33)
    assert lat1 < -33.5
    a = np.array([[0, 1], [2, 255]], np.uint8); b = np.array([[0, 2], [2, 255]], np.uint8)
    out = proto.highlight_rgba(a, b, ["#006400", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#0064c8"])
    assert out[0, 1, 3] == 255 and out[0, 0, 3] < 100 and out[1, 1, 3] == 0
    assert proto.uncovered_share(np.zeros((2, 4, 4), np.uint8)) == 1.0


def test_sampling_weights_balance_regions():
    import pytest
    from lulc.train_math import sampling_weights

    regs = ["city"] * 90 + ["rural"] * 10
    assert sampling_weights(regs, "none") is None
    for mode in ("equal", "sqrt"):
        w = sampling_weights(regs, mode)
        assert w[0] < w[-1]
    w = sampling_weights(regs, "equal")
    assert abs(w[:90].sum() - w[90:].sum()) < 1e-9
    with pytest.raises(ValueError):
        sampling_weights(regs, "x")


def test_hyp3_coherence_helpers_and_loader(tmp_path):
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("hc", "scripts/hyp3_coherence.py")
    hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)
    acq = {"b1": {("2021-01-01", "a"), ("2021-01-13", "b"), ("2021-02-06", "c"), ("2021-12-27", "d"), ("2022-01-08", "e")},
           "b2": {("2020-12-26", "x"), ("2021-01-07", "y")}}
    got = [(b, s1, s2) for b, _, s1, s2 in hc.twelve_day_pairs(acq, 2021)]
    assert got == [("b1", "a", "b"), ("b1", "d", "e")]
    assert hc.scene_month("S1_325416_IW3_20250626T031920_VV_5876-BURST") == "2025-06-26"
    assert [hc.season_index(d) for d in ("2021-01-05", "2021-04-01", "2021-07-09", "2021-10-31", "2021-12-01")] == [0, 1, 2, 3, 0]
    m = hc.seasonal_mean([(0, np.full((4, 4), 0.2, np.float32)), (0, np.full((4, 4), 0.4, np.float32)), (2, np.full((4, 4), 0.9, np.float32))])
    assert np.isclose(m[0, 0, 0], 0.3) and np.isnan(m[1]).all() and np.isclose(m[2, 1, 1], 0.9)
    # a 20 m coherence file is read with factor 2 and cut to the right size
    from lulc.data import coherence_patch, load_store
    _make(tmp_path, "a", 0)
    np.savez(tmp_path / "a" / "coherence_hyp3.npz", coherence=np.full((4, 48, 48), 0.5, np.float16), factor=np.array(2))
    st = load_store(tmp_path, "a", "train", coherence="hyp3")
    assert st.coh.shape == (4, 48, 48) and st.coh_factor == 2
    assert coherence_patch(st.coh, 10, 10, 32, st.coh_factor).shape == (4, 32, 32)


def test_prob_smoothing_precision_recall_and_bundle(tmp_path):
    import numpy as np
    from lulc.change_eval import probability_smooth, temporal_majority
    from lulc.metrics import ConfusionMatrix

    # two weak votes for class 0 and one confident vote for class 1: the majority says 0, the mean probability says 1
    p = np.zeros((3, 2, 1, 1), np.float32)
    p[0, :, 0, 0] = [0.55, 0.45]; p[1, :, 0, 0] = [0.55, 0.45]; p[2, :, 0, 0] = [0.05, 0.95]
    hard = p.argmax(1).astype(np.uint8)
    assert (temporal_majority(hard, 3) == 0).all()
    assert (probability_smooth(p, 3) == 1).all()
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 1, 2, 0]), np.array([0, 1, 1, 1, 2, 2]))
    assert np.isclose(cm.precision()[0], 1 / 3) and np.isclose(cm.recall()[2], 1 / 2)
    assert -1 <= cm.kappa <= 1
    nan = np.full((3, 2, 1, 1), np.nan, np.float32)
    assert (probability_smooth(nan, 3) == 255).all()


def test_final_bundle_roundtrip(tmp_path):
    import torch
    from lulc.train import FINAL_NAME, load_final, save_final
    for k, sd in (("a", 0), ("v", 2)):
        _make(tmp_path, k, sd)
    tr = {"a": load_store(tmp_path, "a", "train")}
    va = {"v": load_store(tmp_path, "v", "val")}
    cfg = dict(patch=32, epochs=1, epoch_samples=16, batch=8, workers=0, val_patches=8, amp=False, encoder="resnet18", lr=2e-3, window=3)
    run(cfg, tr, va, tmp_path / "r" / "seed0", device=torch.device("cpu"), log=lambda *_: None)
    save_final([tmp_path / "r" / "seed0"], tmp_path / FINAL_NAME)
    models, cards = load_final(tmp_path / FINAL_NAME, torch.device("cpu"))
    assert len(models) == 1 and cards[0]["window_months"] == 3
    from lulc import proto
    ms, card, norm, names = proto.load_ensemble(tmp_path, torch.device("cpu"))
    assert len(ms) == 1 and names == ["seed0"]


def test_prototype_coherence_wiring(tmp_path):
    import numpy as np
    from lulc import proto
    from lulc.data import RegionStore

    assert proto.uses_coh({"channels": ["vv", "vh", "coh"]}) is True
    assert proto.uses_coh({"channels": ["vv", "vh"]}) is False
    # attach_coherence finds the HyP3 file and sets factor 2
    d = tmp_path / "blackforest"; d.mkdir()
    np.savez(d / "coherence_hyp3.npz", coherence=np.full((4, 20, 20), 0.5, np.float16), factor=np.array(2))
    st = RegionStore(key="blackforest", split="test", s1=np.zeros((12, 2, 40, 40), np.uint8),
                     label=None, present=np.ones(12, bool))
    assert proto.attach_coherence(st, tmp_path, "blackforest") and st.coh.shape == (4, 20, 20) and st.coh_factor == 2
    # run_models refuses a coherence model when no coherence is attached
    st2 = RegionStore(key="x", split="test", s1=np.zeros((1, 2, 8, 8), np.uint8), label=None, present=np.ones(1, bool))
    import pytest
    with pytest.raises(ValueError, match="coherence file"):
        proto.run_models([object()], st2, {}, 3, "cpu", use_coh=True)
    assert proto.attach_coherence(st2, tmp_path, "missing") is False


def test_worldcover_labels_and_confidence_mask(tmp_path):
    import numpy as np
    from lulc.data import apply_worldcover_labels, load_store
    from lulc import proto
    _make(tmp_path, "a", 0)
    st = load_store(tmp_path, "a", "train")
    h, w = st.label.shape[-2:]
    wc = np.random.randint(10, 96, (h, w)).astype(np.int32)
    np.savez(tmp_path / "a" / "reference.npz", wc=wc, esri=wc)
    before = st.label.copy()
    apply_worldcover_labels(st, tmp_path)
    assert st.label.shape == before.shape and st.label.dtype == before.dtype
    # confidence mask sets low-confidence pixels to ignore
    preds = np.zeros((3, 4, 4), np.uint8); conf = np.full((3, 4, 4), 0.9, np.float32); conf[0, 0, 0] = 0.1
    m = proto.confidence_mask(preds, conf, floor=0.4)
    assert m[0, 0, 0] == 255 and m[1, 1, 1] == 0
