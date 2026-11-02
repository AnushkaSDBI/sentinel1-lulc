import numpy as np
import pandas as pd
import pytest

from lulc.ee_stats import acquisitions_per_month, dw_table, scene_table, summarize


def test_two_slices_of_one_pass_count_once():
    hist = {"2021-01-03|DESCENDING": 2, "2021-01-09|DESCENDING": 1}
    assert acquisitions_per_month(hist) == {(1, "DESCENDING"): 2}


def test_directions_and_months_are_separated():
    hist = {
        "2021-01-03|ASCENDING": 1,
        "2021-01-04|DESCENDING": 1,
        "2021-02-10|ASCENDING": 1,
    }
    per = acquisitions_per_month(hist)
    assert per == {(1, "ASCENDING"): 1, (1, "DESCENDING"): 1, (2, "ASCENDING"): 1}


def test_scene_table_has_twelve_rows_and_best_direction():
    hist = {f"2021-03-{d:02d}|ASCENDING": 1 for d in (1, 7, 13, 19, 25)}
    hist["2021-03-02|DESCENDING"] = 1
    t = scene_table("x", 2021, hist)
    assert len(t) == 12
    march = t[t["month"] == 3].iloc[0]
    assert (march.s1_asc, march.s1_desc, march.s1_best) == (5, 1, 5)
    assert t[t["month"] == 1].iloc[0].s1_best == 0


def test_dw_table_missing_month_is_nan():
    rows = [{"month": 1, "images": 4, "cov1": 0.9, "cov2": 0.5, "n_obs": 2.1}, {}]
    t = dw_table("x", 2021, rows)
    assert len(t) == 12
    assert t[t["month"] == 1].iloc[0].dw_cov1 == pytest.approx(0.9)
    assert np.isnan(t[t["month"] == 2].iloc[0].dw_cov1)


def _frame(year, s1, cov):
    return pd.DataFrame(
        {"year": year, "month": range(1, 13), "s1_best": s1, "dw_cov1": cov, "dw_cov2": cov}
    )


def test_summarize_separates_years_and_winter():
    cov_good = [0.5, 0.5] + [1.0] * 8 + [0.5, 0.5]       # winter months weak
    df = pd.concat([_frame(2021, 5, cov_good), _frame(2024, 2, 1.0)])
    s = summarize(df)
    assert s.loc[2021, "s1_median_per_month"] == 5
    assert s.loc[2024, "s1_median_per_month"] == 2
    assert s.loc[2021, "dw_cov1_winter"] == pytest.approx(0.5)
    assert s.loc[2024, "dw_cov1_winter"] == pytest.approx(1.0)
    assert s.loc[2021, "region_months_cov1_below_80pct"] == 4
    assert s.loc[2021, "dw_cov1_worst_month"] == pytest.approx(0.5)
