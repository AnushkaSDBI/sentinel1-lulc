from pathlib import Path

import numpy as np
import pytest
from pyproj import Transformer

from lulc.config import Region, load_config
from lulc.grid import (
    affine_transform,
    assemble,
    grid_bounds_lonlat,
    grid_for_region,
    split_tiles,
    utm_epsg,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "regions.yaml"


@pytest.mark.parametrize(
    "lat,lon,epsg",
    [
        (49.40, 8.67, 32632),    # Heidelberg
        (52.52, 13.40, 32633),  # Berlin, zone 33
        (39.47, -0.38, 32630),   # Valencia, zone 30
        (45.46, 9.19, 32632),    # Milan
        (-33.9, 18.4, 32734),    # southern hemisphere
    ],
)
def test_utm_epsg(lat, lon, epsg):
    assert utm_epsg(lat, lon) == epsg


def test_utm_rejects_bad_longitude():
    with pytest.raises(ValueError):
        utm_epsg(10, 190)


def _region(lat=49.3988, lon=8.6724, half=12.0):
    return Region("r", "R", lat, lon, "train", half)


def test_grid_size_and_snapping():
    g = grid_for_region(_region())
    assert g.size_px == 2400
    assert g.x0 % 10 == 0 and g.y0 % 10 == 0
    xmin, ymin, xmax, ymax = g.extent
    assert xmax - xmin == pytest.approx(24000)
    assert ymax - ymin == pytest.approx(24000)


def test_grid_is_centred_on_the_region():
    r = _region()
    g = grid_for_region(r)
    xmin, ymin, xmax, ymax = g.extent
    to_utm = Transformer.from_crs(4326, g.epsg, always_xy=True)
    e, n = to_utm.transform(r.lon, r.lat)
    assert abs((xmin + xmax) / 2 - e) <= 10
    assert abs((ymin + ymax) / 2 - n) <= 10


def test_lonlat_bounds_contain_the_centre():
    r = _region()
    w, s, e, n = grid_bounds_lonlat(grid_for_region(r))
    assert w < r.lon < e and s < r.lat < n
    assert (n - s) * 111.32 == pytest.approx(24, rel=0.05)


def test_tiles_cover_grid_exactly_once():
    g = grid_for_region(_region(), size_px=None)
    for tile_px in (1200, 1000, 777):
        tiles = split_tiles(g, tile_px)
        cover = np.zeros((g.size_px, g.size_px), dtype=np.int8)
        for t in tiles:
            cover[t.row:t.row + t.height, t.col:t.col + t.width] += 1
        assert (cover == 1).all()
        assert all(t.height <= tile_px and t.width <= tile_px for t in tiles)


def test_tile_corners_match_pixel_positions():
    g = grid_for_region(_region())
    t = split_tiles(g, 1200)[3]  # row 1200, col 1200
    assert (t.row, t.col) == (1200, 1200)
    assert t.x0 == g.x0 + 1200 * g.res
    assert t.y0 == g.y0 - 1200 * g.res
    a = affine_transform(g, t)
    assert a["scaleX"] == 10 and a["scaleY"] == -10
    assert a["translateX"] == t.x0 and a["translateY"] == t.y0


def test_assemble_round_trip():
    g = grid_for_region(_region(), size_px=50)
    full = np.arange(2 * 50 * 50, dtype=np.float32).reshape(2, 50, 50)
    tiles = split_tiles(g, 20)
    parts = [full[:, t.row:t.row + t.height, t.col:t.col + t.width] for t in tiles]
    np.testing.assert_array_equal(assemble(g, tiles, parts), full)


def test_assemble_rejects_wrong_shape_and_gaps():
    g = grid_for_region(_region(), size_px=40)
    tiles = split_tiles(g, 20)
    parts = [np.zeros((1, 20, 20)) for _ in tiles]
    parts[0] = np.zeros((1, 10, 20))
    with pytest.raises(ValueError):
        assemble(g, tiles, parts)
    with pytest.raises(ValueError, match="cover"):
        assemble(g, tiles[:3], [np.zeros((1, 20, 20))] * 3)


def test_all_configured_regions_get_a_valid_grid():
    _, regions = load_config(CONFIG)
    for r in regions:
        g = grid_for_region(r)
        assert g.size_px == round(2 * r.half_size_km * 100)
        assert g.crs.startswith(("EPSG:326", "EPSG:327"))
