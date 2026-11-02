import pytest

from lulc.config import bbox_from_center
from lulc.stac import bbox_coverage

BOX = bbox_from_center(50.0, 8.0, 10.0)


def _poly(w, s, e, n):
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def test_full_coverage():
    w, s, e, n = BOX
    assert bbox_coverage([_poly(w - 1, s - 1, e + 1, n + 1)], BOX) == pytest.approx(1.0)


def test_half_coverage():
    w, s, e, n = BOX
    mid = (w + e) / 2
    assert bbox_coverage([_poly(w - 1, s - 1, mid, n + 1)], BOX) == pytest.approx(0.5)


def test_two_halves_union_to_full():
    w, s, e, n = BOX
    mid = (w + e) / 2
    geoms = [_poly(w - 1, s - 1, mid, n + 1), _poly(mid, s - 1, e + 1, n + 1)]
    assert bbox_coverage(geoms, BOX) == pytest.approx(1.0)


def test_no_geometries_is_zero():
    assert bbox_coverage([], BOX) == 0.0
