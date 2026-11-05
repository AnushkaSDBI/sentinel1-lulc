import io
import json

import numpy as np
import pytest
from PIL import Image

from lulc import layers, proto
from lulc.classes import CLASS_COLORS

NO = 255


def stack(*maps):
    return np.stack([np.full((4, 4), m, np.uint8) if np.isscalar(m) else np.asarray(m, np.uint8) for m in maps])


def test_switch_count_counts_changes_and_skips_no_radar():
    p = stack(0, 1, NO, 1, 2)          # 0->1 (1), gap skipped, 1->1 (0), 1->2 (1)
    assert (layers.switch_count(p) == 2).all()
    assert (layers.switch_count(stack(3, 3, 3)) == 0).all()
    assert (layers.switch_count(stack(NO, NO)) == 0).all()


def test_change_since_first_only_keeps_differing_pixels():
    a = np.zeros((4, 4), np.uint8)
    b = a.copy(); b[0, 0] = 2
    out = layers.change_since_first(stack(a, b), 0)
    assert out[1, 0, 0] == 2 and (out[1].ravel()[1:] == NO).all() and (out[0] == NO).all()


def test_majority_map_and_no_data():
    p = stack(1, 1, 2)
    assert (layers.majority_map(p) == 1).all()
    assert (layers.majority_map(stack(NO, NO)) == NO).all()


def test_frequency_rgba_and_shares():
    n = np.array([[0, 1, 2], [3, 4, 9], [0, 0, 1]])
    rgba = layers.frequency_rgba(n)
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 255
    assert (rgba[1, 1] == rgba[1, 2]).all()                      # 4 and 9 share "4 or more"
    sh = layers.frequency_shares(n, np.ones_like(n, bool))
    assert sh["0 changes"] == pytest.approx(300 / 9) and sum(sh.values()) == pytest.approx(100)
    assert list(sh)[-1] == "4 or more"


def test_vote_and_confidence_rgba():
    members = np.stack([stack(1, 1), stack(1, 2), stack(1, 2)])    # (3,2,4,4)
    v = layers.vote_confidence(members)
    assert v[0, 0, 0] == 1.0 and v[1, 0, 0] == pytest.approx(2 / 3)
    assert layers.vote_confidence(stack(1, 1)[None])[0, 0, 0] == 1.0
    rgba = layers.confidence_rgba(np.array([[0.4, 1.0, np.nan]]))
    assert rgba[0, 0, 0] < rgba[0, 1, 0] and rgba[0, 2, 3] == 0


def test_vv_rgba_transparent_without_radar():
    s1 = np.zeros((2, 3, 3), np.uint8)
    s1[0, 1, 1] = 200
    out = layers.vv_rgba(s1)
    assert out[1, 1, 3] == 255 and out[0, 0, 3] == 0


def test_animation_gif_has_one_frame_per_month():
    rng = np.random.default_rng(0)
    preds = rng.integers(0, 6, (5, 20, 20)).astype(np.uint8)
    preds[2, :5] = NO
    labels = [f"M{i}" for i in range(5)]
    for mode in ("Land cover", "Changes"):
        gif = layers.animation_gif(preds, labels, CLASS_COLORS, mode=mode, fps=2)
        im = Image.open(io.BytesIO(gif))
        assert im.format == "GIF" and im.n_frames == 5 and im.info["duration"] == 500
    with pytest.raises(ValueError):
        layers.animation_gif(preds[:1], labels[:1], CLASS_COLORS)
    with pytest.raises(ValueError):
        layers.animation_gif(preds, labels, CLASS_COLORS, mode="nope")
    with pytest.raises(ValueError):
        layers.animation_gif(preds, labels[:3], CLASS_COLORS)


def test_card_summary_tolerates_missing_keys():
    s = layers.card_summary({"channels": ["a", "st0"], "window_months": 3, "config": {"encoder": "resnet34", "epochs": 12, "batch": 16, "seed": 0}}, ["seed0"])
    assert s["n_channels"] == 2 and s["uses_stats"] and s["batch_size"] == 16 and "resnet34" in s["architecture"]
    assert layers.card_summary({})["classes"]


def test_scores_and_years_files(tmp_path):
    d = tmp_path / "runs" / "eval_V12_window3_balanced_stats"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"dw": {"miou": 0.54, "miou_lo": 0.5, "miou_hi": 0.58, "oa": 0.8, "per_class_iou": {"Water": 0.9}},
                                                 "other": 1}))
    assert layers.find_eval_dir(tmp_path) == d
    sc = layers.load_scores(d)
    assert list(sc) == ["dw"] and sc["dw"]["per_class_iou"]["Water"] == 0.9
    assert layers.find_eval_dir(tmp_path / "missing") is None and layers.load_scores(tmp_path) == {}
    (tmp_path / "s0").mkdir()
    (tmp_path / "s0" / "years.json").write_text('{"train": [2021], "val": [2022]}')
    assert layers.read_train_years(tmp_path)["train"] == [2021]
    assert layers.read_train_years(tmp_path / "nope") is None


def months(y, n=12, start=1):
    return [(y + (start - 1 + k) // 12, (start - 1 + k) % 12 + 1) for k in range(n)]


def test_reference_plan_covers_every_year():
    assert proto.reference_plan(months(2021))["kind"] == "worldcover+esri"
    for y in (2017, 2019, 2022, 2024, 2025):
        assert proto.reference_plan(months(y))["kind"] == "esri" and proto.reference_plan(months(y))["year"] == y
    p = proto.reference_plan(months(2026, 9))
    assert p["kind"] is None and "not an independent check" in p["reason"]
    assert proto.reference_plan(months(2016))["kind"] is None
    assert proto.reference_plan(months(2022, 12, 7))["year"] == 2023
    assert proto.dominant_year(months(2022, 12, 7)) == 2023          # 6/6 split: the later year wins


def test_reference_file_and_loading(tmp_path):
    assert proto.reference_file(tmp_path, "r", 2021).name == "reference.npz"
    f = proto.reference_file(tmp_path, "r", 2023)
    assert f.name == "reference_esri_2023.npz"
    f.parent.mkdir()
    np.savez(f, esri=np.array([[1, 2], [5, 11]], np.uint8))
    ref = proto.load_reference(f)
    assert list(ref) == ["esri"] and ref["esri"].shape == (2, 2)
    np.savez(tmp_path / "r" / "reference_esri_2030.npz", esri=np.zeros((2, 2), np.uint8))
    with pytest.raises(ValueError, match="no Esri data"):
        proto.load_reference(tmp_path / "r" / "reference_esri_2030.npz")
    assert proto.load_reference(tmp_path / "none.npz") is None


def test_dynamic_world_fallback():
    class S:
        label = np.zeros((3, 4, 4), np.uint8)
    assert (proto.dynamic_world_fallback(S, [0, 1])["dw"] == 0).all()
    S.label = np.full((3, 4, 4), NO, np.uint8)
    assert proto.dynamic_world_fallback(S, [0, 1]) is None
    S.label = None
    assert proto.dynamic_world_fallback(S, [0]) is None


def test_kpi_summary_and_sankey():
    import pandas as pd
    a = np.zeros((4, 4), np.uint8)
    b = a.copy(); b[0, :] = 2
    preds = np.stack([a, b, b])
    pct = pd.DataFrame({"Tree cover": [100.0, 75.0, 75.0], "Grass/shrub": [0.0, 0.0, 0.0], "Cropland": [0.0, 25.0, 25.0]})
    tc = pd.Series({"Tree cover": -25.0, "Grass/shrub": 0.0, "Cropland": 25.0})
    conf = np.full((3, 4, 4), 0.9, np.float32)
    ref = np.zeros((4, 4), np.uint8)
    k = layers.kpi_summary(preds, pct, tc, [10, 10, 5], conf, 1.2, 2.35, ref)
    assert k["months"]["value"] == "3/3" and "median 10" in k["months"]["note"]
    assert k["dominant"]["value"] == "Tree cover" and k["net"]["beyond_noise"] is True
    assert k["flip"]["value"] == "12.5%" and len(k["flip"]["series"]) == 2          # (25% + 0%) / 2 over the two steps
    assert k["agree"]["value"] == "75.0%" and "90%" in k["agree"]["note"]
    assert layers.kpi_summary(preds, pct, tc, [np.nan] * 3, conf, 1.2, 2.35)["agree"]["value"] == "–"
    s = layers.sankey_data(preds[0], preds[1])
    assert sum(s["value"]) == pytest.approx(100) and len(s["labels"]) == 12 and len(s["source"]) == len(s["value"])
    assert layers.agreement_pct(preds, None) is None


def test_persistent_change_and_verdict():
    a = np.zeros((40, 40), np.uint8)
    b = a.copy(); b[5:15, 5:15] = 2                    # one compact 100 px patch
    noisy = a.copy(); noisy[::7, ::7] = 3               # scattered single pixels
    preds = np.stack([a, a, a, a, a, b, b, b])
    v = layers.change_verdict(preds, min_px=30)
    assert v["level"] == "likely" and v["candidate_pct"] == pytest.approx(100 * 100 / 1600)
    assert (v["candidate"][5:15, 5:15] == 2).all() and (v["candidate"][0] == NO).all()
    flick = np.stack([a, noisy, a, noisy, a, noisy, a, noisy])
    assert layers.change_verdict(flick, min_px=30)["level"] == "none"
    assert not layers.compact_patches(noisy != 0, 30).any()
    assert layers.persistent_change(np.stack([a, a])).max() == NO


def test_player_html_contains_every_frame_and_controls():
    maps = [np.full((8, 8), k % 6, np.uint8) for k in range(4)]
    seq = layers.build_sequence(maps, ["A", "B", "C", "D"], CLASS_COLORS)
    assert len(seq["frames"]["Land cover"]) == 4 and "100.0% of the pixels changed" in seq["caption"][1]
    html = layers.player_html({"Month by month": seq, "Year on year": seq}, CLASS_COLORS, ["a"] * 6)
    assert "id=\"play\"" in html and "id=\"spd\"" in html and html.count("data:image/png;base64,") >= 16 and "__DATA__" not in html
