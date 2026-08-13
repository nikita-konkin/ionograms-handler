"""External reference models.

Network-dependent tests skip rather than fail: the point of these references is
to be available when they can be, not to make the suite depend on a third
party's uptime.
"""

from __future__ import annotations

import json

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from muf import reference
from muf.geometry import Point
from muf.reference import chapman, giro, indices, iri, minimuf

CYPRUS = Point(35.18557, 33.38228)   # the registry's Nicosia
YOSHKAR_OLA = Point(56.38, 47.53)


@pytest.fixture
def day_times():
    return pd.date_range("2026-02-04 00:00:00", periods=48, freq="30min")


# --- registry ---------------------------------------------------------------

def test_all_references_are_registered():
    for name in reference.ALL_REFERENCES:
        assert callable(reference.get(name))


def test_unknown_reference():
    with pytest.raises(KeyError, match="unknown reference"):
        reference.get("nope")


def test_run_isolates_failures(day_times):
    """One unavailable model must not stop the others."""
    out = reference.run(("chapman", "minimuf"), CYPRUS, YOSHKAR_OLA, day_times,
                        chapman={"scale_mhz": 30.0})

    assert out["chapman"].ok
    assert not out["minimuf"].ok
    assert out["minimuf"].error


# --- minimuf ----------------------------------------------------------------

def test_minimuf_declines_rather_than_guessing():
    """It must stay unimplemented until real coefficients are available.

    A fabricated reference is worse than none: its entire purpose is to be
    trusted against the pipeline.
    """
    series = minimuf.predict(CYPRUS, YOSHKAR_OLA, [dt.datetime(2026, 2, 4)])

    assert not series.ok
    assert "not implemented" in series.error.lower()
    assert len(series.muf) == 0


# --- chapman ----------------------------------------------------------------

def test_chapman_needs_a_scale(day_times):
    """It has no absolute scale, and must say so instead of inventing one."""
    series = chapman.predict(CYPRUS, YOSHKAR_OLA, day_times)

    assert not series.ok
    assert "scale" in series.error.lower()


def test_chapman_peaks_near_local_noon(day_times):
    series = chapman.predict(CYPRUS, YOSHKAR_OLA, day_times, scale_mhz=30.0)
    assert series.ok

    peak_ut = series.muf.idxmax()
    # Control point sits near 39.5E, so local noon is about 09:20 UT.
    assert 8 <= peak_ut.hour <= 11


def test_chapman_night_floor(day_times):
    series = chapman.predict(CYPRUS, YOSHKAR_OLA, day_times,
                             scale_mhz=30.0, night_fraction=0.35)
    assert series.muf.min() == pytest.approx(30.0 * 0.35, rel=0.02)
    assert series.muf.max() == pytest.approx(30.0, rel=0.05)


def test_chapman_fits_its_amplitude_to_observations(day_times):
    observed = pd.Series(np.linspace(10, 30, len(day_times)))
    series = chapman.predict(CYPRUS, YOSHKAR_OLA, day_times, observed=observed)

    assert series.ok
    assert "fitted" in series.source
    assert series.muf.max() <= 32


def test_solar_zenith_is_negative_at_local_midnight():
    point = Point(45.99, 39.09)          # the path's control point
    # Local midnight at 39.09E is about 21:24 UT.
    assert chapman.solar_zenith_cos(dt.datetime(2026, 2, 4, 21, 20), point) < 0
    assert chapman.solar_zenith_cos(dt.datetime(2026, 2, 4, 9, 20), point) > 0


# --- giro -------------------------------------------------------------------

def test_nearest_station_to_the_control_point():
    """Rostov is what makes a GIRO comparison meaningful for this path."""
    from muf.geometry import midpoint

    found = giro.nearest_station(midpoint(CYPRUS, YOSHKAR_OLA))
    assert found is not None
    ursi, name, km = found
    assert ursi == "RV149"
    assert km < 200


def test_no_station_near_a_remote_path():
    assert giro.nearest_station(Point(-40.0, -140.0)) is None


def test_giro_reports_when_no_station_is_near(day_times):
    remote = Point(-40.0, -140.0)
    series = giro.predict(remote, Point(-45.0, -150.0), day_times, offline=True)

    assert not series.ok
    assert "station" in series.error.lower()


def test_giro_url_format():
    url = giro.build_url("RV149", dt.datetime(2026, 2, 4),
                         dt.datetime(2026, 2, 5), 2588.0)

    assert "ursiCode=RV149" in url
    assert "DMUF=2588" in url
    assert "2026%2F02%2F04" in url or "2026/02/04" in url


def test_giro_parses_a_didbase_table():
    text = (
        "# Some preamble\n"
        "#Time                     foF2   hmF2\n"
        "2026-02-04T00:00:00.000Z   3.85   310.2\n"
        "2026-02-04T00:15:00.000Z   3.72   305.1\n"
        "2026-02-04T00:30:00.000Z    ---     ---\n"
    )
    frame = giro.parse(text)

    assert len(frame) == 2
    assert frame["fof2"].iloc[0] == pytest.approx(3.85)
    assert frame["hmf2"].iloc[1] == pytest.approx(305.1)


def test_giro_parse_empty():
    assert giro.parse("# nothing here\n").empty


# --- iri --------------------------------------------------------------------

def test_iri_reports_when_unavailable(day_times, monkeypatch):
    monkeypatch.setattr(iri, "_backend", lambda: (None, None))
    series = iri.predict(CYPRUS, YOSHKAR_OLA, day_times)

    assert not series.ok
    assert "not installed" in series.error


@pytest.mark.skipif(not iri.available(), reason="no IRI backend installed")
def test_iri_produces_a_plausible_diurnal_curve(day_times):
    series = iri.predict(CYPRUS, YOSHKAR_OLA, day_times, f107=136.0)
    assert series.ok

    assert series.muf.notna().all()
    assert 3 < series.muf.min() < 25
    assert 15 < series.muf.max() < 60
    assert series.muf.max() > series.muf.min() * 1.5   # a real diurnal swing

    fof2 = series.detail["fof2"]
    assert 1.0 < fof2.min() < 8.0        # night
    assert 5.0 < fof2.max() < 20.0       # midday


# --- solar indices ----------------------------------------------------------

def test_indices_offline_without_cache(tmp_path):
    with pytest.raises(indices.IndexUnavailable):
        indices.solar_indices(dt.date(2026, 2, 4), cache_dir=tmp_path, offline=True)


def test_centred_mean_needs_enough_months():
    monthly = {(2026, 1): 100.0, (2026, 2): 110.0}
    assert indices._centred_mean(monthly, 2026, 2) is None

    monthly.update({(2026, 3): 120.0, (2026, 4): 130.0})
    assert indices._centred_mean(monthly, 2026, 2) == pytest.approx(115.0)


def test_centred_mean_wraps_year_boundary():
    monthly = {(2025, 11): 100.0, (2025, 12): 100.0,
               (2026, 1): 100.0, (2026, 2): 100.0}
    assert indices._centred_mean(monthly, 2026, 1) == pytest.approx(100.0)


def test_parse_silso_daily_skips_missing():
    text = ("2026;02;04;2026.096; 148; 10.0;  30;1\n"
            "2026;02;05;2026.099;  -1;  0.0;   0;0\n")
    parsed = indices._parse_silso_daily(text)

    assert parsed[dt.date(2026, 2, 4)] == pytest.approx(148.0)
    assert dt.date(2026, 2, 5) not in parsed        # -1 means not determined


def test_parse_apf107_reads_by_column_not_by_whitespace():
    """``ap`` reaches 400 in a severe storm and the fields are three wide.

    Three digits in a three-wide field leave no space between columns, so a
    split on whitespace merges them -- and the line that proves it is a
    storm day, which is exactly when the index matters.
    """
    text = (" 14  4 15  5  9  5  3  3  3  5  4  5-11163.0144.0143.1\n"
            " 03 10 29400300207236179132 94 67236-11274.8180.6135.9\n")
    parsed = indices._parse_apf107(text)

    assert parsed[dt.date(2014, 4, 15)]["f107"] == pytest.approx(163.0)
    assert parsed[dt.date(2014, 4, 15)]["f107_81"] == pytest.approx(144.0)
    assert parsed[dt.date(2014, 4, 15)]["ap"] == pytest.approx(5)
    # 2003-10-29, the Halloween storm: Ap 236, and every three-hourly value
    # runs into its neighbour.
    assert parsed[dt.date(2003, 10, 29)]["ap"] == pytest.approx(236)
    assert parsed[dt.date(2003, 10, 29)]["f107"] == pytest.approx(274.8)


def test_apf107_two_digit_years_split_at_1958():
    """The file starts 1958-01-01, so ``58`` cannot mean 2058."""
    assert indices._apf107_year(58) == 1958
    assert indices._apf107_year(99) == 1999
    assert indices._apf107_year(0) == 2000
    assert indices._apf107_year(26) == 2026


def test_swpc_daily_f107_prefers_the_noon_reading():
    """Three readings a day; only the 20:00 UT one is the quoted daily value.

    Taking the newest instead would mix a morning reading into a series of
    noon ones -- and today only *has* a morning reading, which is why the
    fallback exists rather than the day being dropped.
    """
    text = json.dumps([
        {"time_tag": "2026-08-12T17:00:00", "flux": 99.0},
        {"time_tag": "2026-08-12T20:00:00", "flux": 100.0},
        {"time_tag": "2026-08-12T22:00:00", "flux": 101.0},
        {"time_tag": "2026-08-13T17:00:00", "flux": 91.0},
    ])
    parsed = indices._parse_swpc_f107_daily(text)

    assert parsed[dt.date(2026, 8, 12)] == pytest.approx(100.0)
    assert parsed[dt.date(2026, 8, 13)] == pytest.approx(91.0)


def test_eisn_parses_the_current_month():
    text = ("2026, 08, 01, 2026.582, 111,  12.9,  35,  42,\n"
            "2026, 08, 02, 2026.585,  -1,   0.0,   0,   0,\n")
    parsed = indices._parse_silso_eisn(text)

    assert parsed[dt.date(2026, 8, 1)] == pytest.approx(111.0)
    assert dt.date(2026, 8, 2) not in parsed        # -1 means not determined


def test_daily_window_mean_needs_most_of_its_window():
    """A trailing mean over ten days is not an 81-day mean and must not pose
    as one -- it would swing with a single active region."""
    sparse = {dt.date(2026, 8, 1) + dt.timedelta(days=d): 100.0
              for d in range(10)}
    assert indices._daily_window_mean(sparse, dt.date(2026, 8, 5)) is None

    full = {dt.date(2026, 7, 1) + dt.timedelta(days=d): 100.0
            for d in range(81)}
    assert indices._daily_window_mean(
        full, dt.date(2026, 8, 10)) == pytest.approx(100.0)


def test_the_model_driver_is_smoothed_not_daily():
    """The CCIR maps were fitted on a smoothed index; the daily flux is not it.

    `f107` is what an operator reads and `f107_driver` is what a model is
    given, and this is the whole reason they are separate fields.
    """
    si = indices.SolarIndices(date=dt.date(2026, 8, 13), f107=91.0,
                              f107_81=121.3, f107_monthly=118.0)
    assert si.f107_driver == pytest.approx(121.3)

    # Falls back in order, and never silently returns nothing when it has
    # something: monthly beats a bare daily value.
    assert indices.SolarIndices(date=dt.date(2026, 8, 13), f107=91.0,
                                f107_monthly=118.0).f107_driver == 118.0
    assert indices.SolarIndices(date=dt.date(2026, 8, 13),
                                f107=91.0).f107_driver == 91.0
    assert indices.SolarIndices(date=dt.date(2026, 8, 13)).f107_driver is None


def test_one_unreachable_source_does_not_sink_the_rest(tmp_path, monkeypatch):
    """The redundancy is the point of having six files on three hosts.

    Before they were added, a SILSO outage raised and IRI lost its driver.
    Now only a total outage does.
    """
    reachable = {indices.BY_KEY["iri_apf107"].url:
                 " 26  2  4  8  8  8  8  8  8  8  8  8-11162.5140.3132.0\n"}

    def selective(url, cache_dir, name, offline=False):
        if url in reachable:
            return reachable[url]
        raise indices.IndexUnavailable(f"blocked: {url}")

    monkeypatch.setattr(indices, "_fetch", selective)
    si = indices.solar_indices(dt.date(2026, 2, 4), cache_dir=tmp_path)

    assert si.used == ("iri_apf107",)
    assert si.f107 == pytest.approx(162.5)
    assert si.f107_driver == pytest.approx(140.3)
    assert si.ssn_daily is None                 # SILSO was blocked; say so
    assert "5 source(s) unavailable" in si.source


def test_every_source_blocked_still_raises(tmp_path, monkeypatch):
    def blocked(url, cache_dir, name, offline=False):
        raise indices.IndexUnavailable(f"blocked: {url}")

    monkeypatch.setattr(indices, "_fetch", blocked)
    with pytest.raises(indices.IndexUnavailable):
        indices.solar_indices(dt.date(2026, 2, 4), cache_dir=tmp_path)


@pytest.mark.network
def test_indices_fetch_live():
    """Live check; skips when the network or the source is unavailable."""
    try:
        si = indices.solar_indices(dt.date(2022, 3, 20))
    except indices.IndexUnavailable as exc:
        pytest.skip(f"index sources unreachable: {exc}")

    assert si.ssn_daily is not None
    assert si.is_smoothed          # 2022 is old enough to have a real R12
    assert 0 < si.r12 < 400


@pytest.mark.network
def test_every_source_is_reachable_and_parses():
    """Each file in `SOURCES` answers, and answers with what we think it is.

    A source that 404s or starts serving an error page still caches, and a
    parser that finds nothing in it returns an empty dict rather than
    complaining -- so the failure would show up months later as a model
    running on a stale driver. This is the check that makes it show up now.

    irimodel.org is the one to watch: it runs mod_security and refuses
    urllib's default User-Agent with a 406.
    """
    for source in indices.SOURCES:
        try:
            text = indices._fetch(source.url, indices.DEFAULT_CACHE,
                                  source.filename)
        except indices.IndexUnavailable as exc:
            pytest.skip(f"{source.key} unreachable: {exc}")
        assert text.strip(), f"{source.key} served an empty body"

    si = indices.solar_indices(dt.date(2014, 4, 15))
    # Every source but EISN, which is only consulted for dates the definitive
    # daily series does not yet reach. A set, because `used` records the order
    # they were read in and that is an implementation detail.
    assert set(si.used) == {s.key for s in indices.SOURCES
                            if s.key != "silso_eisn"}
    assert si.f107 == pytest.approx(163.0)      # apf107.dat, that exact day
    assert si.f107_81 == pytest.approx(144.0)
    assert si.ap == pytest.approx(5)
    assert si.ssn_smoothed == pytest.approx(116.4, abs=1.0)


# --------------------------------------------------------------------------
# GIRO history mirror
# --------------------------------------------------------------------------

def _history_payload():
    return json.dumps([
        {"code": "JR055", "name": "Juliusruh", "history": [
            ["2026-08-09 00:03:16", 90, 2.59, 9.1, 332.3],
            ["2026-08-09 09:03:16", 95, 5.28, 17.4, 243.5],
        ]},
        {"code": "RL052", "name": "Chilton", "history": []},
    ])


def test_history_parses_the_undocumented_field_order(tmp_path, monkeypatch):
    """The feed documents neither the order nor the units, so the order was
    verified against the same station's live record before being relied on."""
    from muf.reference import giro

    monkeypatch.setattr(giro, "fetch", lambda *a, **k: _history_payload())
    frame = giro.history("JR055")

    assert list(frame.columns) == ["confidence", "fof2", "mufd", "hmf2"]
    assert frame.fof2.tolist() == [2.59, 5.28]
    assert frame.hmf2.tolist() == [332.3, 243.5]
    assert str(frame.index.tz) == "UTC"


def test_a_station_with_no_feed_returns_empty_not_an_error(tmp_path, monkeypatch):
    """Chilton is in DIDBase but not in the real-time mirror. A caller asking
    for it should get nothing, not an exception -- one missing station must not
    stop a comparison across four."""
    from muf.reference import giro

    monkeypatch.setattr(giro, "fetch", lambda *a, **k: _history_payload())
    assert giro.history("RL052").empty
    assert giro.history("NOSUCH").empty


def test_the_stations_dob_receives_are_in_the_registry():
    """`io_digisonde` resolves these by name; `giro` has to resolve the same
    places by URSI code, or the vertical-versus-oblique comparison cannot be
    made at all."""
    from muf.reference import giro

    for ursi in ("JR055", "RL052", "DB049", "TR169"):
        assert ursi in giro.STATIONS, ursi
