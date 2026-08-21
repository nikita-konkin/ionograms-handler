"""Command-line surface: argument parsing and results-table discovery."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from muf import cli


def _results(n=3, day="2026-02-04"):
    return pd.DataFrame({
        "file": [f"s{i}.lfs" for i in range(n)],
        "datetime": pd.date_range(f"{day} 00:00:00", periods=n, freq="5min"),
        "muf_algo": [12.0 + i for i in range(n)],
        "freq_stop": [32.5] * n,
    })


def _daily_curve(n=3, day="2026-02-04"):
    """What `muf daily` writes -- must not be mistaken for a results table."""
    return pd.DataFrame({
        "datetime": pd.date_range(f"{day} 00:00:00", periods=n, freq="5min"),
        "date": [day] * n,
        "muf": [12.0] * n,
        "muf_smooth": [12.0] * n,
        "method": ["algo"] * n,
    })


# --- parsing ----------------------------------------------------------------

def test_run_accepts_several_targets():
    args = cli.build_parser().parse_args(["run", "a", "b", "c"])
    assert args.target == [Path("a"), Path("b"), Path("c")]


def test_gate_parsing():
    assert cli._gate("2000,5000") == (2000.0, 5000.0)
    assert cli._gate(None) is None


def test_gate_rejects_inverted_and_malformed():
    with pytest.raises(Exception):
        cli._gate("5000,2000")
    with pytest.raises(Exception):
        cli._gate("nonsense")


def test_methods_parsing():
    from muf.extractors import ALL_METHODS, DEFAULT_METHODS

    assert cli._methods(None) == DEFAULT_METHODS
    assert cli._methods("all") == ALL_METHODS
    assert cli._methods("algo,contour") == ("algo", "contour")


def test_methods_rejects_unknown():
    with pytest.raises(Exception, match="unknown method"):
        cli._methods("algo,telepathy")


# --- results-table discovery -------------------------------------------------

def test_recognises_a_results_table():
    assert cli._is_results_table(_results())


def test_rejects_a_daily_curve():
    """`muf_smooth` would otherwise be read as a method called "smooth"."""
    assert not cli._is_results_table(_daily_curve())


def test_directory_discovery_skips_non_results_files(tmp_path):
    _results().to_csv(tmp_path / "2026-02-04.csv", index=False)
    _daily_curve().to_csv(tmp_path / "2026-02-04_daily_algo.csv", index=False)
    (tmp_path / "2026-02-04_compare.md").write_text("# report")

    frame = cli._read_tables([tmp_path])

    assert len(frame) == 3
    assert "muf_smooth" not in frame


def test_directory_discovery_spans_days(tmp_path):
    _results(day="2026-02-04").to_csv(tmp_path / "a.csv", index=False)
    _results(day="2026-02-05").to_csv(tmp_path / "b.csv", index=False)

    frame = cli._read_tables([tmp_path])
    assert len(frame) == 6
    assert frame["datetime"].is_monotonic_increasing


def test_combined_table_does_not_double_count(tmp_path):
    """Per-day tables plus a --combined table must not count twice."""
    day_a, day_b = _results(day="2026-02-04"), _results(day="2026-02-05")
    day_a.to_csv(tmp_path / "2026-02-04.csv", index=False)
    day_b.to_csv(tmp_path / "2026-02-05.csv", index=False)
    pd.concat([day_a, day_b]).to_csv(
        tmp_path / "2026-02-04_2026-02-05.csv", index=False
    )

    assert len(cli._read_tables([tmp_path])) == 6


def test_named_file_that_is_not_a_results_table_errors(tmp_path):
    path = tmp_path / "curve.csv"
    _daily_curve().to_csv(path, index=False)

    with pytest.raises(SystemExit, match="not a per-sounding results table"):
        cli._read_tables([path])


def test_empty_directory_errors(tmp_path):
    with pytest.raises(SystemExit, match="no results tables"):
        cli._read_tables([tmp_path])


def test_missing_table_errors(tmp_path):
    with pytest.raises(SystemExit, match="no such results table"):
        cli._read_tables([tmp_path / "absent.csv"])


# --- track ------------------------------------------------------------------

def _both_ends(n=40, day="2026-02-04"):
    """A results table carrying both ends of the band, as `run` writes it."""
    hour = pd.Series(range(n)) / 12.0
    return pd.DataFrame({
        "file": [f"s{i}.lfs" for i in range(n)],
        "datetime": pd.date_range(f"{day} 00:00:00", periods=n, freq="5min"),
        "muf_algo": 18 + 7 * (hour / 24),
        "lof_algo": 8 + 3 * (hour / 24),
        "snr_algo": [55.0] * n,
        "lofsnr_algo": [40.0] * n,
        "run_algo": [30] * n,
        "limited_algo": [False] * n,
        "loflim_algo": [False] * n,
        "freq_stop": [32.5] * n,
    })


def test_track_defaults_to_muf():
    assert cli.build_parser().parse_args(["track", "t.csv"]).param == "muf"


def test_tracking_lof_writes_its_own_file(tmp_path):
    """Both ends of one archive are two curves, not one overwriting the other."""
    _both_ends().to_csv(tmp_path / "2026-02-04.csv", index=False)
    args = cli.build_parser().parse_args(
        ["track", str(tmp_path / "2026-02-04.csv"), "--param", "muf,lof",
         "--out", str(tmp_path / "out")])
    assert cli.cmd_track(args) == 0

    written = sorted(p.name for p in (tmp_path / "out").glob("*.csv"))
    assert written == ["2026-02-04_track_algo.csv",
                       "2026-02-04_track_lof_algo.csv"]

    # The MUF file keeps the name it has always had: it is referred to by name
    # in notes and scripts, and adding a word that was always implied would
    # break those for nothing.
    lof = pd.read_csv(tmp_path / "out" / "2026-02-04_track_lof_algo.csv")
    assert "lof" in lof.columns and "muf" not in lof.columns
    assert lof["lof"].max() < 12
