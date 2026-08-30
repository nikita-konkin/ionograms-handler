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


# --- scoring against an external reference -----------------------------------
#
# The GIRO path is the only thing in the project that can distinguish "the
# extractors agree" from "the extractors are right" (see
# docs/2026-08-30-segmentation-quality.md sec. 6b). It was written but never
# exercised end to end, so an outage at DIDBase and a broken integration looked
# identical -- both surface as a line in `reference_problems`. These drive the
# whole path with the network stubbed out, so only the outage can be the cause.

def _geo_frame(algo, contour_values, freq_stop=32.5):
    """A results frame carrying the path geometry `add_reference_models` needs."""
    n = len(algo)
    frame = _frame(algo, contour_values)
    frame["freq_stop"] = [freq_stop] * n
    frame["tx_lat"] = [35.18557] * n           # Nicosia
    frame["tx_lon"] = [33.38228] * n
    frame["rx_lat"] = [56.38] * n              # Yoshkar-Ola
    frame["rx_lon"] = [47.53] * n
    return frame


#: A DIDBase reply shaped like the real one, covering the frame's timestamps.
#: foF2 near 4 MHz over a 2600 km path lands the oblique MUF around 14-15 MHz.
_DIDB_REPLY = (
    "# GIRO tabulated ionospheric characteristics\n"
    "#Time                     foF2   hmF2\n"
    "2026-02-04T00:00:00.000Z   4.10   295.0\n"
    "2026-02-04T00:05:00.000Z   4.15   297.0\n"
    "2026-02-04T00:10:00.000Z   4.05   296.0\n"
    "2026-02-04T00:15:00.000Z   4.20   298.0\n"
)


def test_giro_is_scored_as_a_method_not_just_fetched(monkeypatch):
    """The reference has to reach the pairwise table, or it explains nothing."""
    from muf.reference import giro

    monkeypatch.setattr(giro, "fetch", lambda *a, **k: _DIDB_REPLY)
    frame = _geo_frame([14.0, 14.5, 14.2, 14.8], [15.0, 15.2, 15.1, 15.4])

    text, summary, pairwise = compare.report(frame, reference_models=["giro"])

    assert "muf_giro" in frame.columns or "giro" in text
    pairs = {(row.a, row.b) for row in pairwise.itertuples()}
    assert any("giro" in pair for pair in pairs), \
        "giro was fetched but never compared against anything"
    # Both extractors get scored against it, which is the entire point: it is
    # the only comparison that is not circular.
    assert {"algo", "contour"} <= {name for pair in pairs for name in pair}


def test_a_giro_outage_is_reported_and_does_not_abort_the_run(monkeypatch):
    """DIDBase goes down; the comparison must still produce its own results.

    Observed 2026-08-30: the `DIDBGetValues` servlet returned Tomcat 404s for
    every query while its neighbours on the same host served normally. That has
    to degrade to a stated absence, never to a crash and never to silence.
    """
    from muf.reference import giro

    def dead(*a, **k):
        raise RuntimeError("DIDBase returned HTTP 404 for https://example/x")

    monkeypatch.setattr(giro, "fetch", dead)
    frame = _geo_frame([14.0, 14.5, 14.2, 14.8], [15.0, 15.2, 15.1, 15.4])

    text, summary, pairwise = compare.report(frame, reference_models=["giro"])

    assert "unavailable" in text and "404" in text
    pairs = {(row.a, row.b) for row in pairwise.itertuples()}
    assert ("algo", "contour") in pairs or ("contour", "algo") in pairs


def test_geometry_missing_says_which_columns_and_how_to_get_them():
    """The failure a user actually hits when comparing an older results table."""
    frame = _frame([14.0, 14.5], [15.0, 15.2])       # no tx_lat/rx_lat

    with pytest.raises(KeyError, match="muf run"):
        compare.add_reference_models(frame, ["giro"])


def test_a_reference_above_the_sweep_marks_the_soundings_as_lower_bounds(monkeypatch):
    """The one thing agreement between extractors can never reveal.

    Both extractors can only report what the sounder swept. When the true MUF
    is above the top of the sweep they agree with each other and are both
    wrong, and nothing internal to the pipeline can say so.
    """
    from muf.reference import giro

    monkeypatch.setattr(giro, "fetch", lambda *a, **k: _DIDB_REPLY)
    # Sweep stops at 8 MHz; the reference puts the MUF near 14.
    frame = _geo_frame([7.9, 7.9, 7.9, 7.9], [8.0, 8.0, 8.0, 8.0], freq_stop=8.0)

    text, _, _ = compare.report(frame, reference_models=["giro"])

    assert "Out-of-band soundings" in text
    assert "lower bounds" in text
