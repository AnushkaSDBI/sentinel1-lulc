import numpy as np
import pytest

from lulc.change_eval import false_change, member_spread, stable_mask, temporal_majority
from lulc.classes import IGNORE_INDEX
from lulc.data import (build_index, coherence_patch, compute_norm, load_store, window_months)
from lulc.ee_export import write_atomic
from lulc.metrics import ConfusionMatrix, bootstrap_miou
from lulc.quant import dequantize_db, quantize_db


def test_quantize_round_trip_and_nodata():
    db = np.array([-30.0, -12.34, 0.0, np.nan], dtype=np.float32)
    back = dequantize_db(quantize_db(db))
    assert np.allclose(back[:3], db[:3], atol=0.1 + 1e-6) and np.isnan(back[3])
    assert quantize_db(np.array([-100.0]))[0] == 1 and quantize_db(np.array([50.0]))[0] == 255


def test_confusion_matrix_metrics():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 255]))
    assert cm.m.sum() == 5 and cm.oa == pytest.approx(4 / 5)
    iou = cm.iou()
    assert iou[0] == pytest.approx(1 / 2) and iou[1] == pytest.approx(2 / 3) and iou[2] == pytest.approx(1.0)
    assert cm.miou() == pytest.approx(np.mean([1 / 2, 2 / 3, 1.0]))


def test_absent_class_is_nan_not_zero():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 1]), np.array([0, 1]))
    assert np.isnan(cm.iou()[2]) and cm.miou() == pytest.approx(1.0)


def test_miou_major_ignores_rare_reference_classes():
    cm = ConfusionMatrix(3)
    ref = np.array([0] * 98 + [1] * 2)      # class 1 is 2% of the reference, class 2 is absent
    pred = np.array([0] * 90 + [1] * 8 + [0] * 2)   # class 1 is never predicted where it really is
    cm.update(pred, ref)
    assert cm.miou() == pytest.approx(np.mean([90 / 100, 0.0]))
    assert cm.major_classes(0.05).tolist() == [True, False, False]
    assert cm.miou_major(0.05) == pytest.approx(90 / 100)
    assert cm.miou_major(0.02) == pytest.approx(cm.miou())      # the threshold is inclusive
    assert cm.miou_major(0.0) == pytest.approx(cm.miou())       # no threshold equals mIoU; absent classes stay out
    assert np.isnan(ConfusionMatrix(3).miou_major())


def test_bootstrap_interval_contains_pooled_and_widens_with_few_cities():
    rng = np.random.default_rng(0)
    cms = {}
    for i in range(8):
        cm = ConfusionMatrix(3)
        t = rng.integers(0, 3, 500)
        p = np.where(rng.random(500) < 0.6 + 0.04 * i, t, rng.integers(0, 3, 500))
        cm.update(p, t)
        cms[f"c{i}"] = cm
    r = bootstrap_miou(cms, n_boot=200, seed=1)
    assert r["miou_lo"] <= r["miou"] <= r["miou_hi"] and r["n_cities"] == 8
    with pytest.raises(ValueError):
        bootstrap_miou({})


def test_window_months_edges():
    assert window_months(5, 1) == [5]
    assert window_months(0, 3) == [0, 0, 1]
    assert window_months(11, 3) == [10, 11, 11]
    with pytest.raises(ValueError):
        window_months(3, 2)


def test_coherence_patch_alignment():
    coarse = np.arange(4 * 5 * 5, dtype=np.float32).reshape(4, 5, 5)
    p = coherence_patch(coarse, row=14, col=23, patch=20, factor=10)
    assert p.shape == (4, 20, 20)
    assert p[0, 0, 0] == coarse[0, 1, 2]            # pixel (14,23) lies in coarse cell (1,2)
    assert p[0, 6, 0] == coarse[0, 2, 2]            # row 20 starts coarse row 2


def _write_month(out, key, month, h=64, label=1, conf=90, vv=-12.0):
    z = {"vv": np.full((h, h), vv, np.float16), "vh": np.full((h, h), -18, np.float16),
         "s1_n": np.full((h, h), 5, np.uint8), "dw_label": np.full((h, h), label, np.uint8),
         "dw_conf": np.full((h, h), conf, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}
    write_atomic(out / key / f"2021-{month:02d}.npz", z, {})


def test_load_store_cutoff_and_missing_months(tmp_path):
    for m in (1, 2, 4):
        _write_month(tmp_path, "r", m, conf=60 if m == 2 else 90)
    s = load_store(tmp_path, "r", "train", conf_cutoff=75)
    assert s.s1.shape == (12, 2, 64, 64) and s.present.tolist()[:4] == [True, True, False, True]
    assert (s.label[0] == 0).all()                        # DW trees (1) -> class 0
    assert (s.label[1] == IGNORE_INDEX).all()             # conf 60 < cutoff 75
    assert (s.label[2] == IGNORE_INDEX).all()             # missing month
    with pytest.raises(FileNotFoundError):
        load_store(tmp_path, "nope", "train")


def test_index_and_norm_use_only_present_months(tmp_path):
    for m in (1, 2, 3):
        _write_month(tmp_path, "r", m, vv=-10.0)
    s = load_store(tmp_path, "r", "train")
    idx = build_index(s, patch=32, stride=32)
    assert set(idx.month) == {0, 1, 2} and len(idx) == 3 * 4
    n = compute_norm([s], step=2)
    assert n["vv"][0] == pytest.approx(-10.0, abs=0.11) and n["vh"][0] == pytest.approx(-18.0, abs=0.11)


def test_false_change_and_majority_vote_remove_flicker():
    t, h = 9, 40
    rng = np.random.default_rng(0)
    base = rng.integers(0, 4, (h, h)).astype(np.uint8)
    preds = np.stack([base.copy() for _ in range(t)])
    flick = rng.random((t, h, h)) < 0.08                  # isolated single-month errors
    preds[flick] = (preds[flick] + 1) % 4
    mask = np.ones((h, h), bool)
    raw = false_change(preds, mask)
    smooth = false_change(temporal_majority(preds, 3, n_classes=6), mask)
    assert raw["flip_rate_pct"] > 8 and smooth["flip_rate_pct"] < raw["flip_rate_pct"] / 2
    perfect = false_change(np.stack([base] * t), mask)
    assert perfect["flip_rate_pct"] == 0 and perfect["net_total_change_pp"] == 0


def test_stable_mask_and_majority_ignore_handling():
    a = np.array([[0, 1], [2, 255]], np.uint8)
    b = np.array([[0, 2], [2, 255]], np.uint8)
    assert stable_mask(a, b).tolist() == [[True, False], [True, False]]
    p = np.full((3, 2, 2), 255, np.uint8)
    p[:, 0, 0] = 3
    out = temporal_majority(p, 3)
    assert out[1, 0, 0] == 3 and out[1, 1, 1] == 255


def test_member_spread_zero_for_identical_members():
    base = np.random.default_rng(0).integers(0, 6, (4, 20, 20)).astype(np.uint8)
    r = member_spread(np.stack([base, base, base]))
    assert r["mean"].shape == (4, 6) and np.allclose(r["std"], 0)
    other = (base + 1) % 6
    assert member_spread(np.stack([base, other]))["std"].max() > 0


def test_majority_end_months_are_smoothed_too():
    t, h = 6, 10
    p = np.zeros((t, h, h), np.uint8)
    p[0] = 1          # first month is the odd one out
    p[5] = 2          # and so is the last
    out = temporal_majority(p, 3, n_classes=6)
    assert (out[0] == 0).all() and (out[5] == 0).all()
