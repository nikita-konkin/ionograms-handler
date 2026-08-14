"""Path geometry and the vertical-to-oblique conversion."""

from __future__ import annotations

import math

import pytest

from muf.geometry import (EARTH_RADIUS_KM, MAX_SINGLE_HOP_KM, Point,
                          control_points, describe_path, fof2_to_muf,
                          great_circle_km, hop_count, intermediate, m_factor,
                          midpoint, muf_to_fof2)

# The registry's Nicosia, which is what the loader now hands to this module
# for a `cyprus1` sounding; the header's own round 35.00/34.00 is 59.9 km away
# and 0.6 km longer. See `muf.stations.CYPRUS1_LFS_COORDINATES`.
CYPRUS = Point(35.18557, 33.38228)
YOSHKAR_OLA = Point(56.38, 47.53)
PATH_KM = 2587.8


def test_great_circle_matches_the_circuit():
    assert great_circle_km(CYPRUS, YOSHKAR_OLA) == pytest.approx(PATH_KM, abs=5)


def test_great_circle_degenerate_and_antipodal():
    assert great_circle_km(CYPRUS, CYPRUS) == pytest.approx(0.0, abs=1e-6)
    antipode = Point(-CYPRUS.lat, CYPRUS.lon - 180)
    assert great_circle_km(CYPRUS, antipode) == pytest.approx(
        math.pi * EARTH_RADIUS_KM, rel=1e-3
    )


def test_midpoint_is_equidistant():
    mid = midpoint(CYPRUS, YOSHKAR_OLA)
    a = great_circle_km(CYPRUS, mid)
    b = great_circle_km(mid, YOSHKAR_OLA)

    assert a == pytest.approx(b, rel=1e-6)
    assert a + b == pytest.approx(PATH_KM, abs=5)


def test_control_point_location():
    """Pins the control point -- it decides which ionosonde is relevant."""
    mid = midpoint(CYPRUS, YOSHKAR_OLA)
    assert mid.lat == pytest.approx(45.99, abs=0.05)
    assert mid.lon == pytest.approx(39.09, abs=0.05)


def test_single_hop_uses_one_control_point():
    assert len(control_points(CYPRUS, YOSHKAR_OLA)) == 1


def test_long_path_uses_two_control_points():
    far = Point(-30.0, 150.0)
    points = control_points(CYPRUS, far)

    assert len(points) == 2
    assert great_circle_km(CYPRUS, points[0]) == pytest.approx(2000, abs=20)
    assert great_circle_km(far, points[1]) == pytest.approx(2000, abs=20)


def test_hop_count_follows_the_single_hop_limit():
    assert hop_count(0.0) == 1
    assert hop_count(PATH_KM) == 1
    assert hop_count(MAX_SINGLE_HOP_KM) == 1
    assert hop_count(MAX_SINGLE_HOP_KM + 1) == 2
    assert hop_count(14415.0) == 4


def test_the_whole_path_secant_is_not_merely_less_accurate():
    """Past the single-hop limit it describes a ray that cannot exist.

    `m_factor` peaks at 3840 km and falls away after, because the reflection
    would have to happen below the horizon. So evaluating an 8000 km path
    whole *understates* the MUF -- which shows up as the instrument appearing
    to over-pick, not as the model being asked the wrong question.
    """
    whole = m_factor(8000.0, 300.0)
    per_hop = m_factor(8000.0 / hop_count(8000.0), 300.0)

    assert whole == pytest.approx(2.665, abs=0.01)
    assert per_hop == pytest.approx(3.370, abs=0.01)
    assert per_hop > whole


def test_describe_path_is_one_phrasing_for_both_exporters():
    """The CLI and the server formatted their own midpoint until 2026-08-14."""
    short = describe_path(CYPRUS, YOSHKAR_OLA)
    assert short == "control point 45.99N 39.09E, D=2588km"

    far = describe_path(CYPRUS, Point(-33.87, 151.21))
    assert far.startswith("control points ")
    assert " / " in far and far.endswith(", 4 hops")


def test_intermediate_endpoints():
    assert great_circle_km(intermediate(CYPRUS, YOSHKAR_OLA, 0.0), CYPRUS) < 1
    end = intermediate(CYPRUS, YOSHKAR_OLA, PATH_KM)
    assert great_circle_km(end, YOSHKAR_OLA) < 5


# --- the secant law ---------------------------------------------------------

def test_m_factor_is_unity_at_zero_range():
    """Vertical incidence: MUF is foF2 itself."""
    assert m_factor(0.0) == pytest.approx(1.0)


def test_m3000_matches_the_conventional_range():
    """M(3000)F2 is conventionally about 3.0-3.4; this is the sanity anchor."""
    assert 3.0 <= m_factor(3000.0) <= 3.4


def test_m_factor_grows_with_path_length():
    values = [m_factor(d) for d in (500, 1000, 2000, 3000)]
    assert values == sorted(values)


def test_m_factor_falls_with_reflection_height():
    """A higher layer means a steeper ray and less obliquity gain."""
    assert m_factor(PATH_KM, 250) > m_factor(PATH_KM, 300) > m_factor(PATH_KM, 400)


def test_conversion_round_trip():
    muf = fof2_to_muf(6.0, PATH_KM, 300.0)
    assert muf_to_fof2(muf, PATH_KM, 300.0) == pytest.approx(6.0)


def test_measured_muf_implies_plausible_fof2():
    """Our 2026-02-04 values must map back to sensible vertical frequencies.

    Mid-latitude winter near solar maximum: foF2 runs roughly 3-5 MHz at night
    and 8-12 MHz around midday. A sign or scale error in the conversion shows up
    here immediately.
    """
    night = muf_to_fof2(12.20, PATH_KM, 300.0)
    midday = muf_to_fof2(30.14, PATH_KM, 300.0)

    assert 3.0 < night < 5.0
    assert 8.0 < midday < 12.0
    assert midday > night
