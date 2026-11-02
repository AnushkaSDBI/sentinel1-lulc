from pathlib import Path

import pytest

from lulc.config import (
    Region,
    bbox_from_center,
    bboxes_intersect,
    load_config,
    validate_no_overlap,
)

REGIONS_YAML = Path(__file__).resolve().parents[1] / "configs" / "regions.yaml"


def test_bbox_is_centred_and_sized():
    w, s, e, n = bbox_from_center(50.0, 8.0, 10.0)
    assert (w + e) / 2 == pytest.approx(8.0)
    assert (s + n) / 2 == pytest.approx(50.0)
    assert (n - s) * 111.32 == pytest.approx(20.0, rel=1e-6)


def test_intersection_logic():
    assert bboxes_intersect((0, 0, 2, 2), (1, 1, 3, 3))
    assert not bboxes_intersect((0, 0, 1, 1), (1, 0, 2, 1))  # touching edge only


def _r(key, lat, lon, split):
    return Region(key, key, lat, lon, split, 12.0)


def test_overlap_across_splits_raises():
    # Mannheim and Heidelberg are ~18 km apart: 12 km half-size boxes overlap.
    with pytest.raises(ValueError, match="overlap"):
        validate_no_overlap([_r("mannheim", 49.4875, 8.4660, "train"),
                             _r("heidelberg", 49.3988, 8.6724, "case_study")])


def test_overlap_within_same_split_is_allowed():
    validate_no_overlap([_r("a", 49.4875, 8.4660, "train"), _r("b", 49.3988, 8.6724, "train")])


def test_real_config_loads_and_has_no_leakage():
    year, regions = load_config(REGIONS_YAML)
    assert year == 2021
    splits = {r.split for r in regions}
    assert {"train", "val", "test", "test_ood", "case_study"} <= splits
    assert len({r.key for r in regions}) == len(regions)
