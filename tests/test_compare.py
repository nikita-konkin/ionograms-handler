"""Method comparison and reference-series handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from muf import compare


def _frame(a, b, limited_a=None):
    n = len(a)
    return pd.DataFrame({
        "datetime": pd.date_range("2026-02-04 00:00:00", periods=n, freq="5min"),
        "freq_stop": [32.5] * n,
        "muf_algo": a,
        "muf_contour": b,
        "limited_algo": limited_a if limited_a is not None else [False] * n,
        "limited_contour": [False] * n,
        "run_algo": [10] * n,
        "run_contour": [10] * n,
    })


def test_identical_series_agree_perfectly():
    values = np.linspace(10, 30, 20)
    result = compare.agreement(pd.Series(values), pd.Series(values), "a", "b")

    assert result.rmse == pytest.approx(0.0)
    assert result.mae == pytest.approx(0.0)
    assert result.bias == pytest.approx(0.0)
    assert result.r2 == pytest.approx(1.0)


def test_bias_sign_says_which_reads_higher():
    left = pd.Series([11.0, 12.0, 13.0])
    right = pd.Series([10.0, 11.0, 12.0])

    assert compare.agreement(left, right, "a", "b").bias == pytest.approx(1.0)
    assert compare.agreement(right, left, "b", "a").bias == pytest.approx(-1.0)


def test_missing_values_are_dropped_pairwise():
    left = pd.Series([10.0, np.nan, 12.0])
    right = pd.Series([10.0, 11.0, np.nan])

    result = compare.agreement(left, right, "a", "b")
    assert result.n == 1
    assert result.rmse == pytest.approx(0.0)


def test_no_overlap_yields_nan_not_an_error():
    result = compare.agreement(
        pd.Series([np.nan, 1.0]), pd.Series([1.0, np.nan]), "a", "b"
    )
    assert result.n == 0
    assert np.isnan(result.rmse)


def test_band_limited_excluded_from_comparison():
    a = [12.0, 32.5, 14.0]
    frame = _frame(a, [12.0, 20.0, 14.0], limited_a=[False, True, False])

    values = compare.usable(frame, "algo")
    assert values.notna().sum() == 2

    pairwise = compare.compare_methods(frame)
    assert int(pairwise.iloc[0]["n"]) == 2
    assert pairwise.iloc[0]["rmse_mhz"] == pytest.approx(0.0)


def test_summary_counts():
    frame = _frame([12.0, 32.5, np.nan], [12.0, 20.0, 14.0],
                   limited_a=[False, True, False])
    summary = compare.summarise_methods(frame).set_index("method")

    assert summary.loc["algo", "n_picked"] == 1
    assert summary.loc["algo", "n_band_limited"] == 1
    assert summary.loc["contour", "n_picked"] == 3


def test_report_is_markdown():
    text, summary, pairwise = compare.report(
        _frame(np.linspace(10, 20, 10), np.linspace(10.1, 20.1, 10))
    )

    assert text.startswith("# MUF method comparison")
    # The column pads to the widest method name, so match the cell, not its
    # width -- otherwise renaming an estimator breaks a formatting assertion.
    assert any(line.startswith("| method") for line in text.splitlines())
    assert "rmse_mhz" in text
    assert len(summary) == 2
    assert len(pairwise) == 1


def test_report_excludes_time_spans():
    frame = _frame(np.linspace(10, 20, 10), np.linspace(10, 20, 10))
    _, summary, _ = compare.report(
        frame, exclude=[("2026-02-04 00:00:00", "2026-02-04 00:20:00")]
    )
    assert summary.iloc[0]["n_soundings"] == 5


def test_reads_legacy_reference_format(tmp_path):
    """MUF.py:198 writes space-delimited `MUF time vrange`, no header."""
    path = tmp_path / "MUF_cyprus1_20220320.csv"
    path.write_text(
        "11.77 00:00:10 2625.0\n"
        "11.76 00:05:10 2600.0\n"
        "11.79 00:10:10 2600.0\n"
    )

    series = compare.read_reference(path)
    assert len(series) == 3
    assert series.iloc[0] == pytest.approx(11.77)


def test_reads_normal_csv_reference(tmp_path):
    path = tmp_path / "ref.csv"
    path.write_text("datetime,muf\n2026-02-04 00:00:00,11.5\n2026-02-04 00:05:00,11.6\n")

    series = compare.read_reference(path)
    assert len(series) == 2
    assert series.iloc[1] == pytest.approx(11.6)


def test_reference_aligns_by_time_of_day():
    """So a historical day can be compared against today's run."""
    frame = _frame([12.0, 13.0], [12.0, 13.0])
    reference = pd.Series(
        [20.0, 21.0],
        index=pd.to_datetime(["2022-03-20 00:00:00", "2022-03-20 00:05:00"]),
    )

    aligned = compare.align_reference(frame, reference)
    assert aligned.tolist() == [20.0, 21.0]


def test_markdown_table_handles_missing_values():
    table = compare._markdown_table(pd.DataFrame({"a": [1, np.nan], "b": ["x", "y"]}))
    assert table.count("\n") == 3      # header, rule, two rows
    assert "| a" in table
