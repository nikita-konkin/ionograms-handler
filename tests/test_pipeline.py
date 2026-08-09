"""End-to-end behaviour, including regression against real recordings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from muf import pipeline
from muf.pipeline import Options

from conftest import synth_iq

WINDOW = 512
N_FREQ = 200
HALF_SPAN = 60_000.0


@pytest.fixture
def synthetic_path(make_lfs):
    iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=2700.0,
                  half_span_km=HALF_SPAN, echo_last_bin=120)
    return make_lfs(iq)


@pytest.fixture
def options():
    return Options(window=WINDOW, gate_km=(2000.0, 5000.0))


def test_row_has_a_column_per_method(synthetic_path, options):
    row = pipeline.process_file(synthetic_path, options)

    for name in options.methods:
        for prefix in ("muf", "vrange", "ndet", "run", "snr", "limited"):
            assert f"{prefix}_{name}" in row
    assert row["tx"] == "synthtx"
    assert row["path_type"] == "oblique"
    assert "error" not in row


def test_the_row_names_its_format_and_window(synthetic_path, make_chirp_h5,
                                             options):
    """`sounding.format` and `sounding.window` are read straight off the row.

    They were absent from it for as long as the column existed, so every
    ingested sounding stored NULL for both and the console could not tell a
    recording from a v2 product -- which is the one thing they are for when a
    parallel run puts both in one database (architecture.md sec. 2.4, 3.4).

    The spellings are the loader's, not the headers': a `.lfs` header carries
    the file's magic number "LFSG", and storing that beside v2's "chirp2"
    would be two vocabularies in one column.
    """
    lfs_row = pipeline.process_file(synthetic_path, options)
    assert lfs_row["format"] == "lfs"
    assert lfs_row["window"] == WINDOW

    chirp = make_chirp_h5(np.full((4, 64), 100.0))
    chirp_row = pipeline.process_file(chirp, Options(window=WINDOW))

    assert chirp_row["format"] == "chirp2"
    # Not `options.window`: v2 fixed the window when it wrote the product and
    # the raw IQ is not in the file, so the ionogram cannot be re-derived at
    # the one that was asked for. The row records what it was actually formed
    # at, which is what makes the pair meaningful.
    assert chirp_row["window"] != WINDOW


def test_an_error_row_still_names_the_file_but_claims_no_format(tmp_path, options):
    """A row that never reached a header must not assert a format it did not
    establish -- `ingest` skips it, and a half-filled row would be worse than
    an empty one if that ever changed."""
    bad = tmp_path / "bad.lfs"
    bad.write_bytes(b"\x00" * 512)

    row = pipeline.process_file(bad, options)
    assert row["file"] == "bad.lfs"
    assert "format" not in row


def test_unreadable_file_yields_an_error_row(tmp_path, options):
    bad = tmp_path / "bad.lfs"
    bad.write_bytes(b"\x00" * 512)

    row = pipeline.process_file(bad, options)
    assert "error" in row
    assert "muf_algo" not in row


def test_band_limited_is_flagged(make_lfs, options):
    """An echo reaching the top of the sweep is a lower bound, not a value."""
    iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=2700.0,
                  half_span_km=HALF_SPAN, echo_last_bin=N_FREQ - 1)
    row = pipeline.process_file(make_lfs(iq, name="edge.lfs"), options)

    assert row["limited_algo"] is True


def test_not_band_limited_when_trace_stops_early(synthetic_path, options):
    row = pipeline.process_file(synthetic_path, options)
    assert row["limited_algo"] is False


def test_process_many(tmp_path, make_lfs, options):
    for hour in range(3):
        iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=2700.0,
                      half_span_km=HALF_SPAN, echo_last_bin=100 + hour)
        make_lfs(iq, name=f"s_{hour}.lfs", start_hour=hour)

    frame = pipeline.process_many(tmp_path, options, jobs=1, progress=False)
    assert len(frame) == 3
    assert frame["datetime"].is_monotonic_increasing
    assert pipeline.methods_in(frame) == list(options.methods)


def test_missing_target():
    with pytest.raises(FileNotFoundError):
        pipeline.process_many("no/such/place")


# --- several days at once ----------------------------------------------------

def test_find_lfs_accepts_several_targets(tmp_path):
    from muf.io_lfs import find_lfs

    a, b = tmp_path / "day_a", tmp_path / "day_b"
    for folder in (a, b):
        folder.mkdir()
        (folder / "s.lfs").write_bytes(b"")

    assert len(find_lfs([a, b])) == 2
    assert len(find_lfs(a)) == 1


def test_find_lfs_deduplicates_overlapping_targets(tmp_path):
    """Naming a directory and a file inside it must not process it twice."""
    from muf.io_lfs import find_lfs

    (tmp_path / "s.lfs").write_bytes(b"")
    assert len(find_lfs([tmp_path, tmp_path / "s.lfs"])) == 1


def test_find_lfs_reports_only_when_nothing_resolves(tmp_path):
    from muf.io_lfs import find_lfs

    (tmp_path / "s.lfs").write_bytes(b"")
    # A bad target alongside a good one is tolerated...
    assert len(find_lfs([tmp_path, tmp_path / "nope"])) == 1
    # ...but all-bad is an error.
    with pytest.raises(FileNotFoundError):
        find_lfs([tmp_path / "nope"])


def test_process_many_over_two_days(tmp_path, make_lfs, options):
    for day in (4, 5):
        for hour in range(2):
            iq = synth_iq(n_freq=N_FREQ, window=WINDOW, echo_range_km=2700.0,
                          half_span_km=HALF_SPAN, echo_last_bin=100 + hour)
            make_lfs(iq, name=f"d{day}_{hour}.lfs", start_day=day, start_hour=hour)

    frame = pipeline.process_many(tmp_path, options, jobs=1, progress=False)

    assert len(frame) == 4
    assert len(pipeline.days_in(frame)) == 2
    assert frame["datetime"].is_monotonic_increasing


def test_split_by_day():
    frame = pd.concat([_day_frame(n=4), _day_frame(n=4)], ignore_index=True)
    frame.loc[4:, "datetime"] = frame.loc[4:, "datetime"] + pd.Timedelta(days=1)

    parts = dict(pipeline.split_by_day(frame))
    assert len(parts) == 2
    assert all(len(p) == 4 for p in parts.values())


def test_days_in_empty():
    assert pipeline.days_in(pd.DataFrame()) == []


def test_daily_over_two_days_keeps_them_separate():
    """Each day gets its own grid, and rows carry the date they belong to."""
    first = _day_frame(n=12)
    second = _day_frame(n=12)
    second["datetime"] = second["datetime"] + pd.Timedelta(days=1)
    frame = pd.concat([first, second], ignore_index=True)

    curve = pipeline.daily(frame, method="algo")

    assert len(curve) == 576                 # two full days at 5-minute cadence
    assert "date" in curve
    assert curve["date"].nunique() == 2
    assert curve.groupby("date").size().tolist() == [288, 288]


def test_daily_does_not_interpolate_across_a_missing_day():
    """A gap must stay a gap rather than being bridged."""
    first = _day_frame(n=12)
    third = _day_frame(n=12)
    third["datetime"] = third["datetime"] + pd.Timedelta(days=2)
    curve = pipeline.daily(pd.concat([first, third], ignore_index=True),
                           method="algo")

    dates = sorted(curve["date"].unique())
    assert len(dates) == 2
    assert (dates[1] - dates[0]).days == 2   # the middle day is simply absent


def test_daily_skips_an_unusable_day_and_records_it():
    good = _day_frame(n=12)
    bad = _day_frame(n=12, limited=True)
    bad["datetime"] = bad["datetime"] + pd.Timedelta(days=1)

    curve = pipeline.daily(pd.concat([good, bad], ignore_index=True), method="algo")

    assert curve["date"].nunique() == 1
    assert curve.attrs["skipped_days"]


def test_write_and_read_back(tmp_path, synthetic_path, options):
    frame = pd.DataFrame([pipeline.process_file(synthetic_path, options)])
    path = pipeline.write(frame, tmp_path / "out.csv")
    assert pd.read_csv(path).shape[0] == 1


# --- daily aggregation -------------------------------------------------------

def _day_frame(n=24, muf=None, limited=False):
    times = pd.date_range("2026-02-04 00:00:00", periods=n, freq="1h")
    values = muf if muf is not None else np.linspace(10, 20, n)
    return pd.DataFrame({
        "datetime": times,
        "muf_algo": values,
        "limited_algo": [limited] * n,
    })


def test_daily_lands_on_a_regular_grid():
    curve = pipeline.daily(_day_frame(), method="algo")

    assert len(curve) == 288                      # 24h at 5-minute cadence
    assert "muf" in curve and "muf_smooth" in curve
    deltas = curve["datetime"].diff().dropna().unique()
    assert len(deltas) == 1


def test_daily_interpolates_between_soundings():
    curve = pipeline.daily(_day_frame(), method="algo", smooth=False)
    assert curve["muf"].notna().all()
    assert curve["muf"].is_monotonic_increasing


def test_daily_drops_band_limited():
    """Band-limited values would flatten the midday peak if kept."""
    with pytest.raises(ValueError, match="no usable values"):
        pipeline.daily(_day_frame(limited=True), method="algo")


def test_daily_rejects_unknown_method():
    with pytest.raises(KeyError):
        pipeline.daily(_day_frame(), method="nope")


# --- regression against real recordings --------------------------------------

def test_real_sounding_regression(real_file):
    """Pins the reference sounding measured while designing the pipeline.

    03:00 UTC on 2026-02-04: MUF 12.2 MHz with the echo at 2739 km. The range
    is the sharper check -- it is set by the geometry, and a sign error in the
    range axis moves it to -2732 km.
    """
    row = pipeline.process_file(real_file)

    assert row["muf_algo"] == pytest.approx(12.2, abs=0.3)
    assert row["vrange_algo"] == pytest.approx(2739, abs=30)
    assert row["muf_contour"] == pytest.approx(12.2, abs=0.3)
    assert row["tx"] == "cyprus1"
    assert row["rx"] == "yoshkar-ola"


def test_real_methods_agree(real_file):
    row = pipeline.process_file(real_file)
    values = [row[f"muf_{m}"] for m in ("algo", "kmeans", "contour")]

    assert all(np.isfinite(v) for v in values)
    assert max(values) - min(values) < 1.0


def test_real_day_is_physically_plausible(real_dir):
    """Over a day the MUF must vary, peak in daylight and trough at night."""
    frame = pipeline.process_many(
        real_dir, Options(methods=("algo",)), jobs=0, progress=False
    )
    frame = frame.dropna(subset=["muf_algo"])
    if len(frame) < 100:
        pytest.skip("not enough of the day present to judge the diurnal shape")

    frame["hour"] = pd.to_datetime(frame["datetime"]).dt.hour
    night = frame[frame["hour"].isin([0, 1, 2, 3, 22, 23])]["muf_algo"].median()
    midday = frame[frame["hour"].isin([9, 10, 11, 12, 13])]["muf_algo"].median()

    assert midday > night, "MUF must be higher in daylight than at night"
    assert 5 < night < 25
    assert 10 < midday < 40


def test_the_range_rule_is_on_for_digisonde_and_off_for_lfs(synthetic_path, options,
                                                            make_digisonde_h5):
    """Scoped per format on purpose: switching it on for `.lfs` would move every
    result already published, and leaving it off for a digisonde reception lets
    a crowded band satisfy the consecutive-bins rule by accident."""
    from muf import pick as pick_module

    lfs_opts = pipeline.Options(window=WINDOW, gate_km=(2000.0, 5000.0),
                                methods=("algo",))
    assert lfs_opts.per_method()["algo"].get("max_range_slope") is None

    seen = {}
    real_run = pipeline.extractors.run

    def spy(ion, methods=(), **kwargs):
        seen.update(kwargs)
        return real_run(ion, methods=methods, **kwargs)

    pipeline.extractors.run = spy
    try:
        pipeline.process_file(synthetic_path, lfs_opts)
        assert "max_range_slope" not in seen.get("algo", {}), ".lfs untouched"

        seen.clear()
        digi = make_digisonde_h5(n_range=1000, range_step_m=3e3)
        pipeline.process_file(digi, pipeline.Options(methods=("algo",)))
        assert seen["algo"]["max_range_slope"] == pick_module.DEFAULT_MAX_RANGE_SLOPE
    finally:
        pipeline.extractors.run = real_run


def test_an_explicit_slope_overrides_the_per_format_default(make_digisonde_h5):
    opts = pipeline.Options(methods=("algo",), max_range_slope=999.0)
    assert opts.per_method()["algo"]["max_range_slope"] == 999.0
