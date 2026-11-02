import numpy as np
import pytest

from lulc.classes import (
    CLASS_NAMES,
    DW_CLASS_NAMES,
    DYNAMIC_WORLD_TO_LULC,
    IGNORE_INDEX,
    N_CLASSES,
    remap_dynamic_world,
    remap_worldcover,
)


def test_six_classes():
    assert N_CLASSES == 6 == len(CLASS_NAMES)


def test_remap_known_values():
    wc = np.array([[10, 20, 30], [40, 50, 60], [80, 100, 0]], dtype=np.uint8)
    out = remap_worldcover(wc)
    expected = np.array([[0, 1, 1], [2, 3, 4], [5, 4, IGNORE_INDEX]], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("value", [0, 70, 90, 95, 255, 15])
def test_dropped_and_unknown_values_are_ignored(value):
    out = remap_worldcover(np.array([value], dtype=np.uint8))
    assert out[0] == IGNORE_INDEX


def test_remap_rejects_out_of_range():
    with pytest.raises(ValueError):
        remap_worldcover(np.array([300]))


def test_dynamic_world_known_values():
    dw = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8)
    out = remap_dynamic_world(dw)
    expected = np.array(
        [[5, 0, 1], [IGNORE_INDEX, 2, 1], [3, 4, IGNORE_INDEX]], dtype=np.uint8
    )
    np.testing.assert_array_equal(out, expected)


def test_dynamic_world_nodata_stays_ignored():
    out = remap_dynamic_world(np.array([255, 9, 200], dtype=np.uint8))
    assert (out == IGNORE_INDEX).all()


def test_dynamic_world_mapping_covers_every_class_once_or_is_dropped():
    assert len(DW_CLASS_NAMES) == 9
    dropped = set(range(9)) - set(DYNAMIC_WORLD_TO_LULC)
    assert dropped == {3, 8}  # flooded vegetation, snow and ice
    # every output class is reachable from Dynamic World labels
    assert set(DYNAMIC_WORLD_TO_LULC.values()) == set(range(N_CLASSES))


def test_both_products_use_the_same_output_classes():
    wc = set(remap_worldcover(np.arange(256, dtype=np.uint8)).tolist()) - {IGNORE_INDEX}
    dw = set(remap_dynamic_world(np.arange(256, dtype=np.uint8)).tolist()) - {IGNORE_INDEX}
    assert wc == dw == set(range(N_CLASSES))


def test_esri_scheme_detection_and_remap():
    import numpy as np
    from lulc.classes import IGNORE_INDEX, detect_esri_scheme, remap_esri

    orig = np.array([[1, 2, 5], [7, 8, 11], [4, 10, 0]], dtype=np.uint8)
    assert detect_esri_scheme(orig[orig > 0]) == "original"
    out = remap_esri(orig)
    assert out[0].tolist() == [5, 0, 2] and out[1].tolist() == [3, 4, 1]
    assert out[2].tolist() == [IGNORE_INDEX] * 3          # flooded veg, clouds, no data

    remapped = np.array([[1, 2, 4], [5, 6, 9], [3, 7, 0]], dtype=np.uint8)
    assert detect_esri_scheme(remapped[remapped > 0]) == "remapped"
    out = remap_esri(remapped)
    assert out[0].tolist() == [5, 0, 2] and out[1].tolist() == [3, 4, 1]
    assert out[2].tolist() == [IGNORE_INDEX] * 3

    with pytest.raises(ValueError):
        detect_esri_scheme([1, 2])
