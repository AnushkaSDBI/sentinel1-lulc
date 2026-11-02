import numpy as np
import pytest

from lulc.classes import IGNORE_INDEX
from lulc.proto import (NOISE_FLOOR, analyse, compare_with_reference, load_local, load_reference, make_region,
                        month_labels, month_sequence)
from lulc.ee_export import write_atomic


def test_month_sequence_crosses_year_boundary():
    ym = month_sequence(2021, 11)
    assert ym[0] == (2021, 11) and ym[2] == (2022, 1) and ym[-1] == (2022, 10) and len(set(ym)) == 12
    assert month_labels(ym[:2]) == ["Nov 2021", "Dec 2021"]


def test_make_region_validates():
    assert make_region(49.4, 8.67, 5.12).half_size_km == 2.56
    with pytest.raises(ValueError):
        make_region(88, 0, 5)


def _preds(t=12, h=40, flicker=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 4, (h, h)).astype(np.uint8)
    p = np.stack([base] * t)
    f = rng.random(p.shape) < flicker
    p[f] = (p[f] + 1) % 4
    return p


def test_analyse_flags_noise_and_smoothing_reduces_it():
    labels = [f"m{i}" for i in range(12)]
    p = _preds(flicker=0.1)
    members = np.stack([p, _preds(flicker=0.1, seed=0)])
    raw = analyse(p, members, labels, smooth=False)
    smooth = analyse(p, members, labels, smooth=True)
    assert smooth["churn"].mean() < raw["churn"].mean()
    assert raw["spread"].shape == (12, 6)
    clean = analyse(_preds(), np.stack([_preds()]), labels, smooth=False)
    assert clean["within_noise"].all() and clean["net_within_noise"] and clean["spread"] is None
    big = _preds().copy(); big[6:, :20] = 3      # 50 % of the area switches class in month 7
    r = analyse(big, np.stack([big]), labels, smooth=False)
    assert not r["within_noise"].all() and r["churn"].max() > NOISE_FLOOR["monthly_churn_raw_pp"]


def test_compare_with_reference():
    p = _preds(h=30)
    ref = {"esri": p[0].copy(), "wc": (p[0] + 1) % 4}
    ref["wc"] = ref["wc"].astype(np.uint8)
    df = compare_with_reference(p, ref)
    assert df.loc["esri", "agreement_%"] == pytest.approx(100) and df.loc["wc", "agreement_%"] == pytest.approx(0)


def test_load_local_and_reference(tmp_path):
    h = 32
    for m in range(1, 13):
        write_atomic(tmp_path / "r" / f"2021-{m:02d}.npz", {
            "vv": np.full((h, h), -12, np.float16), "vh": np.full((h, h), -18, np.float16),
            "s1_n": np.full((h, h), 5, np.uint8), "dw_label": np.full((h, h), 1, np.uint8),
            "dw_conf": np.full((h, h), 90, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}, {})
    np.savez(tmp_path / "r" / "reference.npz", wc=np.full((h, h), 10, np.uint8), esri=np.full((h, h), 2, np.uint8))
    store, ym = load_local(tmp_path, "r")
    assert store.s1.shape == (12, 2, h, h) and store.present.all() and ym[0] == (2021, 1)
    ref = load_reference(tmp_path / "r" / "reference.npz")
    assert (ref["wc"] == 0).all() and (ref["esri"] == 0).all() and (ref["agree"] == 0).all()
    assert load_reference(tmp_path / "missing.npz") is None


def test_scene_counts_are_read_and_flagged(tmp_path):
    from lulc.proto import scene_flags
    h = 16
    for m in range(1, 4):
        write_atomic(tmp_path / "r" / f"2021-{m:02d}.npz", {
            "vv": np.full((h, h), -12, np.float16), "vh": np.full((h, h), -18, np.float16),
            "s1_n": np.full((h, h), 2 if m == 3 else 8, np.uint8), "dw_label": np.full((h, h), 1, np.uint8),
            "dw_conf": np.full((h, h), 90, np.uint8), "dw_n": np.full((h, h), 3, np.uint8)}, {})
    from lulc.data import load_store_ym
    s = load_store_ym(tmp_path, "r", [(2021, 1), (2021, 2), (2021, 3)])
    assert s.meta["s1_scenes"] == [8.0, 8.0, 2.0]
    assert scene_flags(s.meta["s1_scenes"]) == [False, False, True]
    assert scene_flags([float("nan")] * 3) == [False] * 3
