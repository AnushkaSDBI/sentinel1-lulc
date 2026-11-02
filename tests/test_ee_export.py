import json

import numpy as np
import pytest
from pyproj import Transformer

from lulc.config import Region
from lulc.ee_export import (
    BANDS, LABEL_NODATA, S1_NODATA, check_grid_probe, decode, dominant_direction,
    export_region_month, month_path, quality_report,
)
from lulc.grid import Tile, grid_for_region, split_tiles


def _grid(size=40):
    return grid_for_region(Region("r", "R", 49.4, 8.67, "train", 12.0), size_px=size)


def test_dominant_direction():
    hist = {"2021-01-03|ASCENDING": 1, "2021-01-09|DESCENDING": 1, "2021-02-09|DESCENDING": 1}
    assert dominant_direction(hist) == "DESCENDING"
    assert dominant_direction({"2021-01-03|ASCENDING": 1}) == "ASCENDING"
    with pytest.raises(ValueError):
        dominant_direction({})


def _stack(h=4, w=4):
    s = np.zeros((len(BANDS), h, w), dtype=np.float32)
    s[0], s[1] = -12.5, -19.0           # VV, VH
    s[2] = 5                            # s1_n
    s[3] = 6                            # label: built
    s[4] = 0.734                        # conf
    s[5] = 3
    return s


def test_decode_values_and_nodata():
    s = _stack()
    s[0, 0, 0] = S1_NODATA
    s[3, 1, 1] = LABEL_NODATA
    d = decode(s)
    assert np.isnan(d["vv"][0, 0]) and d["vv"][0, 1] == np.float16(-12.5)
    assert d["dw_label"][1, 1] == 255 and d["dw_label"][0, 0] == 6
    assert d["dw_conf"][0, 0] == 73
    assert d["vv"].dtype == np.float16 and d["dw_label"].dtype == np.uint8


def test_decode_rejects_wrong_band_count():
    with pytest.raises(ValueError):
        decode(np.zeros((5, 2, 2), dtype=np.float32))


def test_quality_report():
    s = _stack()
    s[0, :2, :] = S1_NODATA
    rep = quality_report(decode(s))
    assert rep["s1_valid"] == pytest.approx(0.5)
    assert rep["label_valid"] == 1.0 and rep["both_valid"] == pytest.approx(0.5)
    assert rep["s1_scenes_median"] == 5


def test_probe_check_accepts_correct_and_rejects_shifted():
    g = _grid()
    t = Tile(0, 0, 4, 4, g.x0, g.y0)
    to_ll = Transformer.from_crs(g.epsg, 4326, always_xy=True)
    xx, yy = np.meshgrid(g.x0 + (np.arange(4) + 0.5) * 10, g.y0 - (np.arange(4) + 0.5) * 10)
    lon, lat = to_ll.transform(xx, yy)
    probe = np.stack([lon, lat])
    assert check_grid_probe(g, probe, t) < 1e-9
    shifted = np.stack([lon + 3e-4, lat])  # about 20 m
    with pytest.raises(RuntimeError):
        check_grid_probe(g, shifted, t)


def test_export_round_trip_and_resume(tmp_path):
    g = _grid(40)
    full = np.random.default_rng(0).uniform(-25, 0, (len(BANDS), 40, 40)).astype(np.float32)
    full[2:] = 4
    full[3] = 1
    calls = []

    def fetch(tile):
        calls.append(tile)
        return full[:, tile.row:tile.row + tile.height, tile.col:tile.col + tile.width]

    path = month_path(tmp_path, "r", 2021, 7)
    rep = export_region_month(g, fetch, path, {"region": "r"}, tile_px=20, workers=2)
    assert rep["both_valid"] == 1.0 and len(calls) == 4
    z = np.load(path)
    np.testing.assert_allclose(z["vv"].astype(np.float32), full[0], atol=0.02)
    assert z["dw_label"].shape == (40, 40)
    assert json.loads(str(z["meta"]))["epsg"] == g.epsg
    assert not list(path.parent.glob("*.tmp"))

    assert export_region_month(g, fetch, path, {}, tile_px=20) is None  # resume: skipped
    assert len(calls) == 4


def test_failed_fetch_leaves_no_file(tmp_path):
    g = _grid(40)

    def fetch(tile):
        raise OSError("boom")

    path = month_path(tmp_path, "r", 2021, 1)
    with pytest.raises(OSError):
        export_region_month(g, fetch, path, {}, tile_px=20)
    assert not path.exists()


def test_transient_errors():
    from lulc.ee_export import is_transient

    assert is_transient(RuntimeError("429 Too Many Requests"))
    assert is_transient(TimeoutError())
    assert not is_transient(RuntimeError("Image.reduce: Unable to reduce an image with 0 bands."))


def test_empty_month_decodes_to_nodata():
    s = np.zeros((len(BANDS), 4, 4), dtype=np.float32)
    s[0], s[1] = S1_NODATA, S1_NODATA
    s[3] = LABEL_NODATA
    rep = quality_report(decode(s))
    assert rep["s1_valid"] == 0.0 and rep["label_valid"] == 0.0 and rep["both_valid"] == 0.0


def test_composite_settings_validation_and_conflict(tmp_path):
    from lulc.ee_export import CompositeSettings, settings_conflict

    with pytest.raises(ValueError):
        CompositeSettings(method="nope")
    with pytest.raises(ValueError):
        CompositeSettings(angle_range=(45, 31))
    g = _grid(40)
    full = np.zeros((len(BANDS), 40, 40), dtype=np.float32)
    path = month_path(tmp_path, "r", 2021, 5)
    assert not settings_conflict(path, CompositeSettings())            # no file yet
    export_region_month(g, lambda t: full[:, t.row:t.row + t.height, t.col:t.col + t.width],
                        path, {"composite": CompositeSettings().as_dict()}, tile_px=20)
    assert not settings_conflict(path, CompositeSettings())
    assert settings_conflict(path, CompositeSettings(angle_range=(31, 45)))
    assert settings_conflict(path, CompositeSettings(method="mean_linear"))


def test_old_files_without_settings_count_as_defaults(tmp_path):
    from lulc.ee_export import CompositeSettings, settings_conflict

    g = _grid(40)
    full = np.zeros((len(BANDS), 40, 40), dtype=np.float32)
    path = month_path(tmp_path, "r", 2021, 6)
    export_region_month(g, lambda t: full[:, t.row:t.row + t.height, t.col:t.col + t.width],
                        path, {}, tile_px=20)
    assert not settings_conflict(path, CompositeSettings())
    assert settings_conflict(path, CompositeSettings(angle_range=(31, 45)))


def test_reference_decode_and_export(tmp_path):
    from lulc.ee_export import REF_BANDS, decode_reference, export_reference_region, value_shares

    g = _grid(40)
    full = np.zeros((len(REF_BANDS), 40, 40), dtype=np.float32)
    full[0] = 50
    full[1, :20] = 7
    shares = export_reference_region(
        g, lambda t: full[:, t.row:t.row + t.height, t.col:t.col + t.width],
        tmp_path / "reference.npz", {"region": "r"}, tile_px=20)
    assert shares["wc"] == {50: 1.0} and shares["esri"] == {0: 0.5, 7: 0.5}
    z = np.load(tmp_path / "reference.npz")
    assert z["wc"].dtype == np.uint8 and z["esri"].shape == (40, 40)
    assert value_shares(np.array([1, 1, 2, 2])) == {1: 0.5, 2: 0.5}
    with pytest.raises(ValueError):
        decode_reference(np.zeros((3, 2, 2), dtype=np.float32))


def test_coarse_grid_is_aligned_and_covers():
    from lulc.ee_export import COH_BANDS, coarse_grid, decode_coherence, export_coherence_region

    g = _grid(2400)
    c = coarse_grid(g)
    assert (c.x0, c.y0, c.res, c.size_px) == (g.x0, g.y0, 100.0, 240)
    g2 = _grid(1024)
    assert coarse_grid(g2).size_px == 103 and coarse_grid(g2).extent[2] >= g2.extent[2]


def test_coherence_decode_and_export(tmp_path):
    from lulc.ee_export import COH_BANDS, decode_coherence, export_coherence_region

    g = _grid(200)                       # coarse grid 20 px
    full = np.full((len(COH_BANDS), 20, 20), 0.5, dtype=np.float32)
    full[:, :, 10:] = -1                 # no data on the right half
    rep = export_coherence_region(g, lambda t: full[:, t.row:t.row + t.height, t.col:t.col + t.width],
                                  tmp_path / "coherence.npz", {"region": "r"})
    assert rep["valid"] == pytest.approx(0.5)
    z = np.load(tmp_path / "coherence.npz")
    assert z["coherence"].shape == (len(COH_BANDS), 20, 20) and np.isnan(z["coherence"][0, 0, 15])
    assert "summer_vv_COH06" in list(z["bands"])
    with pytest.raises(ValueError):
        decode_coherence(np.zeros((3, 2, 2), dtype=np.float32))
