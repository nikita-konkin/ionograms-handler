"""External reference models.

Network-dependent tests skip rather than fail: the point of these references is
to be available when they can be, not to make the suite depend on a third
party's uptime.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from muf import reference
from muf.geometry import Point
from muf.reference import chapman, giro, indices, iri, minimuf

CYPRUS = Point(35.00, 34.00)
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
    point = Point(45.88, 39.45)
    # Local midnight at 39.45E is about 21:20 UT.
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
