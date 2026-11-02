import numpy as np
import pytest

from lulc.stats import class_percentages, monthly_change_table

IGN = 255
NAMES = ["a", "b", "c"]


def test_percentages_sum_to_100():
    p = np.array([[0, 0, 1], [1, 2, 2]])
    out = class_percentages(p, 3)
    assert out.sum() == pytest.approx(100.0)
    np.testing.assert_allclose(out, [100 / 3, 100 / 3, 100 / 3])


def test_ignored_pixels_are_excluded():
    p = np.array([0, 0, 1, IGN])
    out = class_percentages(p, 3)
    np.testing.assert_allclose(out, [200 / 3, 100 / 3, 0])


def test_all_ignored_gives_nan():
    assert np.isnan(class_percentages(np.full((2, 2), IGN), 3)).all()


def test_class_out_of_range_raises():
    with pytest.raises(ValueError):
        class_percentages(np.array([0, 3]), 3)


def test_delta_and_total_change():
    m1 = np.array([0, 0, 0, 0])      # 100% a
    m2 = np.array([0, 0, 0, 1])      # 75 / 25
    m3 = np.array([0, 0, 1, 1])      # 50 / 50
    s = monthly_change_table([m1, m2, m3], ["m1", "m2", "m3"], NAMES)
    assert s.percent.loc["m2", "a"] == pytest.approx(75.0)
    assert np.isnan(s.delta.loc["m1", "a"])
    assert s.delta.loc["m2", "a"] == pytest.approx(-25.0)
    assert s.delta.loc["m3", "b"] == pytest.approx(25.0)
    assert s.total_change["a"] == pytest.approx(-50.0)
    assert s.total_change["b"] == pytest.approx(50.0)
    # total change equals the sum of monthly deltas
    np.testing.assert_allclose(s.total_change, s.delta.sum(), atol=1e-9)
    # changes across classes cancel out
    assert s.total_change.sum() == pytest.approx(0.0, abs=1e-9)


def test_common_mask_prevents_fake_change_from_missing_data():
    # Same land in both months, but month 2 has a hole where class 1 sits.
    m1 = np.array([0, 0, 1, 1])
    m2 = np.array([0, 0, IGN, IGN])
    shared = monthly_change_table([m1, m2], ["m1", "m2"], NAMES, common_mask=True)
    assert shared.total_change.abs().max() == pytest.approx(0.0)
    assert shared.valid_fraction.iloc[0] == pytest.approx(0.5)

    naive = monthly_change_table([m1, m2], ["m1", "m2"], NAMES, common_mask=False)
    assert naive.total_change["a"] == pytest.approx(50.0)  # fake change


def test_shape_mismatch_and_too_few_months_raise():
    with pytest.raises(ValueError):
        monthly_change_table([np.zeros((2, 2)), np.zeros((3, 3))], ["a", "b"], NAMES)
    with pytest.raises(ValueError):
        monthly_change_table([np.zeros((2, 2))], ["a"], NAMES)
