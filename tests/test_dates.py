from datetime import date

import pytest

from lulc.dates import month_windows, stac_interval


def test_twelve_months_with_year_rollover():
    w = month_windows("2021-11", 12)
    assert [x[0] for x in w][:3] == ["2021-11", "2021-12", "2022-01"]
    assert w[-1][0] == "2022-10"
    for (_, _, end), (_, start, _) in zip(w, w[1:]):
        assert end == start  # contiguous, no gaps or overlaps


def test_december_window_ends_next_january():
    label, first, nxt = month_windows("2021-12", 1)[0]
    assert (label, first, nxt) == ("2021-12", date(2021, 12, 1), date(2022, 1, 1))


def test_leap_february_interval():
    _, first, nxt = month_windows("2024-02", 1)[0]
    assert stac_interval(first, nxt) == "2024-02-01T00:00:00Z/2024-02-29T23:59:59Z"


@pytest.mark.parametrize("bad", ["2021", "2021-13", "abc", "2021-00"])
def test_bad_start_raises(bad):
    with pytest.raises(ValueError):
        month_windows(bad)
