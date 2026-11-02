import numpy as np
import pytest

from lulc.patches import RunningStats, select_patches, window_fractions, window_starts


def test_window_starts_touch_the_edge():
    assert window_starts(2400, 256, 256)[-1] == 2400 - 256
    assert window_starts(512, 256, 256) == [0, 256]
    assert window_starts(300, 256, 256) == [0, 44]
    with pytest.raises(ValueError):
        window_starts(100, 256, 256)


def test_window_fractions_match_brute_force():
    rng = np.random.default_rng(1)
    m = rng.random((70, 90)) > 0.4
    rows, cols = window_starts(70, 32, 20), window_starts(90, 32, 20)
    f = window_fractions(m, rows, cols, 32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            assert f[i, j] == pytest.approx(m[r:r + 32, c:c + 32].mean())


def test_select_patches_applies_both_thresholds():
    t, h = 2, 64
    label = np.ones((t, h, h), bool)
    s1 = np.ones((t, h, h), bool)
    label[0, :32, :32] = False            # month 0: top-left patch has no label
    s1[1, 40:, 40:] = False               # month 1: bottom-right patch lost S1
    df = select_patches(label, s1, [3, 4], patch=32, stride=32, min_label=0.4, min_s1=0.99)
    got = set(zip(df.month, df.row, df.col))
    assert (3, 0, 0) not in got and (3, 32, 32) in got
    assert (4, 32, 32) not in got and (4, 0, 0) in got
    assert len(df) == 6


def test_select_patches_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        select_patches(np.ones((2, 8, 8), bool), np.ones((2, 8, 8), bool), [1], patch=4, stride=4)


def test_running_stats_ignore_nan_and_match_numpy():
    a, b = np.random.default_rng(0).normal(-12, 5, (2, 1000))
    b[::7] = np.nan
    rs = RunningStats()
    rs.update(a)
    rs.update(b)
    both = np.concatenate([a, b[np.isfinite(b)]])
    assert rs.mean == pytest.approx(both.mean())
    assert rs.std == pytest.approx(both.std())
    with pytest.raises(ValueError):
        RunningStats().mean
