import numpy as np
import pytest

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


def test_noise_floor_is_the_v12_measurement():
    assert NOISE_FLOOR["flip_rate_raw_pct"] == 2.61 and NOISE_FLOOR["flip_rate_smoothed_pct"] == 1.00
    assert NOISE_FLOOR["monthly_churn_raw_pp"] == 1.61 and NOISE_FLOOR["monthly_churn_smoothed_pp"] == 0.62
    assert NOISE_FLOOR["net_raw_pp"] == 2.27 and NOISE_FLOOR["net_smoothed_pp"] == 1.48
    assert "V12" in NOISE_FLOOR["source"]


def test_no_demo_is_a_region_the_model_trained_on():
    from pathlib import Path
    from lulc.config import load_config
    from lulc.proto import DEMOS, demo_name
    _, regions = load_config(Path(__file__).resolve().parent.parent / "configs" / "regions.yaml")
    seen = [r for r in regions if r.split in ("train", "val")]
    for name, (lat, lon, *_) in DEMOS.items():
        near = [r.key for r in seen if abs(r.lat - lat) < 0.1 and abs(r.lon - lon) < 0.1]
        assert not near, (name, near)
        assert demo_name(lat, lon) == name
    assert any(n.startswith("Control: Prague") for n in DEMOS) and demo_name(0.0, 0.0) is None


def test_month_without_radar_stays_no_data_after_smoothing():
    labels = [f"m{i}" for i in range(12)]
    p = _preds()
    p[5] = 255                                               # no radar at all in this month
    r = analyse(p, np.stack([p, p]), labels, smooth=True)
    assert (r["preds"][5] == 255).all() and (r["preds"][4] != 255).all() and (r["preds"][6] != 255).all()
    pct = r["stats"].percent
    assert pct.loc["m5"].isna().all() and pct.drop(index="m5").notna().all().all()
    assert r["churn"].loc[["m5", "m6"]].isna().all() and not r["within_noise"].loc[["m5", "m6"]].any()
    assert r["stats"].total_change.notna().all() and r["spread"].drop(index="m5").notna().all().all()


def test_first_and_last_months_without_radar_are_skipped_in_total_change():
    labels = [f"m{i}" for i in range(12)]
    p = _preds()
    p[0] = 255; p[11] = 255
    r = analyse(p, np.stack([p]), labels, smooth=False)
    assert (r["first"], r["last"]) == (1, 10) and r["stats"].total_change.notna().all()


def test_pixel_gap_is_not_filled_by_smoothing_and_too_few_months_is_an_error():
    labels = [f"m{i}" for i in range(12)]
    p = _preds()
    p[3, 0, 0] = 255
    r = analyse(p, np.stack([p]), labels, smooth=True)
    assert r["preds"][3, 0, 0] == 255 and r["preds"][2, 0, 0] != 255
    p = np.full_like(p, 255); p[4] = 1
    with pytest.raises(ValueError, match="at least 2"):
        analyse(p, np.stack([p]), labels, smooth=False)


def test_seed_disagreement_is_per_pixel_and_ignores_no_data():
    from lulc.proto import seed_disagreement
    a = np.array([[[0, 1], [2, 255]]], np.uint8)
    b = np.array([[[0, 3], [2, 255]]], np.uint8)
    assert seed_disagreement(np.stack([a, b])).tolist() == [[[False, True], [False, False]]]
    p = _preds()
    assert analyse(p, np.stack([p]), [f"m{i}" for i in range(12)], smooth=False)["disagree"] is None


def test_no_data_and_seed_mask_never_look_like_a_class():
    from lulc.classes import CLASS_COLORS
    from lulc.proto import highlight_rgba, overlay_rgba
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in CLASS_COLORS])

    def far_from_classes(colour):                            # Built-up and Bare are greys, so an exact-match check is not enough
        return np.linalg.norm(pal - np.array(colour), axis=1).min() > 50

    pred = np.full((32, 32), 255, np.uint8)
    pred[:16] = 4
    uncertain = np.zeros((32, 32), bool); uncertain[:8] = True
    rgba = overlay_rgba(pred, CLASS_COLORS, no_data="hatch", uncertain=uncertain)
    hatch = {tuple(x) for x in rgba[16:, :, :3].reshape(-1, 3)}
    assert len(hatch) == 2 and all(far_from_classes(c) for c in hatch) and (rgba[16:, :, 3] == 255).all()
    grey = {tuple(x) for x in rgba[:8, :, :3].reshape(-1, 3)}
    assert len(grey) == 1 and all(far_from_classes(c) for c in grey) and not grey & hatch
    assert (overlay_rgba(pred, CLASS_COLORS)[16:, :, 3] == 0).all()                  # the default stays transparent
    faded = highlight_rgba(pred, pred, CLASS_COLORS, no_data="hatch")
    assert (faded[16:, :, 3] == 255).all() and (faded[:16, :, 3] < 100).all()        # no-data is not faded as "unchanged"


def test_fetch_months_makes_no_earth_engine_call_when_all_months_are_on_disk(tmp_path):
    from lulc.ee_export import month_path
    from lulc.proto import fetch_months, grid_for_region, month_sequence

    class NoCalls:
        def __getattr__(self, name):
            raise AssertionError(f"Earth Engine was called: {name}")

    region = make_region(50.0755, 14.4378, 2.56)
    ym = month_sequence(2021, 1, 4)
    for y, m in ym:
        f = month_path(tmp_path, region.key, y, m)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    assert fetch_months(NoCalls(), region, ym, tmp_path) == grid_for_region(region)
    month_path(tmp_path, region.key, *ym[2]).unlink()
    with pytest.raises(AssertionError, match="Earth Engine was called"):
        fetch_months(NoCalls(), region, ym, tmp_path)


def test_geotiff_keeps_position_classes_and_no_data():
    pytest.importorskip("rasterio", exc_type=ImportError)
    from rasterio.io import MemoryFile
    from lulc.classes import CLASS_COLORS
    from lulc.proto import geotiff_bytes, grid_for_region
    grid = grid_for_region(make_region(50.0755, 14.4378, 2.56))
    n = grid.size_px
    p = np.random.default_rng(0).integers(0, 6, (3, n, n)).astype(np.uint8)
    p[1, :5, :5] = 255
    with MemoryFile(geotiff_bytes(p, grid, ["a", "b", "c"])) as mem, mem.open() as src:
        assert src.count == 3 and src.nodata == 255 and src.crs.to_epsg() == grid.epsg
        assert src.descriptions == ("a", "b", "c") and "1=" in src.tags()["classes"]
        assert tuple(src.bounds) == pytest.approx(grid.extent) and (src.read() == p).all()
    with MemoryFile(geotiff_bytes(p[0], grid, ["a"], CLASS_COLORS)) as mem, mem.open() as src:
        assert src.count == 1 and src.colormap(1)[4][:3] == (0xe8, 0xe8, 0xe8)
    with pytest.raises(ValueError, match="does not match"):
        geotiff_bytes(p[0, :5, :5], grid, ["a"])
