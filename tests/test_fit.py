"""Fitting the nose of a trace.

The fit is a quality signal and a guarded cross-check, not a replacement
estimator, so most of these tests are about it *declining* correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import fit


def synthetic_nose(muf=20.0, vertex_range=2800.0, curvature=-2e-3,
                   n=40, span_km=120.0, noise=0.0, seed=0):
    """Points on a parabola whose vertex is at ``(vertex_range, muf)``."""
    rng = np.random.default_rng(seed)
    heights = np.linspace(vertex_range - span_km, vertex_range, n)
    freqs = muf + curvature * (heights - vertex_range) ** 2
    if noise:
        freqs = freqs + rng.normal(0, noise, n)
    return freqs, heights


def test_recovers_a_clean_vertex():
    freq, vrange = synthetic_nose(muf=20.0, vertex_range=2800.0)
    result = fit.fit_nose(freq, vrange)

    assert result.ok
    assert result.muf_mhz == pytest.approx(20.0, abs=0.05)
    assert result.vrange_km == pytest.approx(2800.0, abs=5)
    assert result.curvature < 0


def test_tolerates_noise():
    freq, vrange = synthetic_nose(muf=20.0, noise=0.05)
    result = fit.fit_nose(freq, vrange)

    assert result.ok
    assert result.muf_mhz == pytest.approx(20.0, abs=0.3)
    assert result.rms_residual_mhz < 0.5


def test_declines_on_too_few_points():
    result = fit.fit_nose(np.array([10.0, 11.0]), np.array([2700.0, 2710.0]))

    assert not result.ok
    assert "points" in result.reason


def test_declines_when_trace_is_flat_in_range():
    """No curvature means no nose, whatever the frequency span."""
    # Densely sampled so the refusal is about the shape, not the point count
    # inside the (narrow, 1 MHz) fitting window.
    freq = np.linspace(10, 20, 400)
    vrange = np.full(400, 2700.0)
    result = fit.fit_nose(freq, vrange)

    assert not result.ok
    assert "flat" in result.reason


def test_declines_when_parabola_opens_upward():
    """A trace curving the wrong way has no nose; no vertex may be reported."""
    # Gentle curvature so the whole curve fits inside the fitting window and
    # the shape, not the point count, is what causes the refusal.
    freq, vrange = synthetic_nose(curvature=+1e-4, n=40, span_km=120.0)
    result = fit.fit_nose(freq, vrange)

    assert not result.ok
    assert np.isnan(result.muf_mhz) or result.muf_mhz != pytest.approx(20.0)


def test_declines_on_a_messy_trace():
    """Scatter that does not describe a nose must be rejected, not averaged."""
    rng = np.random.default_rng(3)
    freq = np.linspace(10, 25, 60)
    vrange = 2700 + rng.normal(0, 60, 60)
    result = fit.fit_nose(freq, vrange)

    assert not result.ok


def test_declines_on_excessive_extrapolation():
    """A vertex far past the data is not evidence, it is wishful thinking."""
    # Points on a wing 100-200 km short of the vertex: the curve reaches only
    # 29 MHz while the vertex sits at 30, so the fit must extrapolate 1 MHz.
    heights = np.linspace(2600.0, 2700.0, 30)
    freqs = 30.0 - 1e-4 * (heights - 2800.0) ** 2

    generous = fit.fit_nose(freqs, heights, max_extrapolation_mhz=5.0)
    assert generous.ok
    assert generous.extrapolation_mhz == pytest.approx(1.0, abs=0.2)

    strict = fit.fit_nose(freqs, heights, max_extrapolation_mhz=0.5)
    assert not strict.ok
    assert "extrapolate" in strict.reason


def test_extrapolation_is_reported():
    freq, vrange = synthetic_nose(muf=20.0, curvature=-1e-3)
    keep = vrange < vrange.max() - 15
    result = fit.fit_nose(freq[keep], vrange[keep], max_extrapolation_mhz=5.0)

    if result.ok:
        assert result.extrapolation_mhz > 0
        assert result.is_extrapolated
        assert result.muf_mhz > result.observed_max_mhz


def test_residual_separates_clean_from_contaminated():
    """The property that makes the fit useful as a quality metric."""
    clean = fit.fit_nose(*synthetic_nose(noise=0.02))
    rng = np.random.default_rng(1)
    freq, vrange = synthetic_nose(noise=0.02)
    freq = freq + rng.normal(0, 2.0, len(freq))      # heavy contamination
    dirty = fit.fit_nose(freq, vrange)

    assert clean.ok
    assert clean.rms_residual_mhz < 0.2
    assert not dirty.ok or dirty.rms_residual_mhz > clean.rms_residual_mhz * 5


def test_no_fit_sentinel():
    assert not fit.NO_FIT.ok
    assert np.isnan(fit.NO_FIT.muf_mhz)


# --- against real soundings --------------------------------------------------

def test_trace_points_from_a_real_sounding(real_file):
    from muf import extractors, spectro

    ion = spectro.compute(real_file)
    result = extractors.get("algo")(ion)
    freq, vrange = fit.trace_points(ion, result)

    assert len(freq) == len(vrange)
    assert len(freq) > 10
    assert np.all(np.isfinite(vrange))
    # Points must lie inside the gate, not outside it.
    assert vrange.min() >= ion.vrange.min() - 1
    assert vrange.max() <= ion.vrange.max() + 1


def test_real_sounding_fit_is_close_to_the_pick(real_file):
    """Not identical -- the vertex is a different estimator -- but not wild."""
    from muf import extractors, spectro

    ion = spectro.compute(real_file)
    result = extractors.get("algo")(ion)
    nose = fit.fit_result(ion, result)

    if not nose.ok:
        pytest.skip(f"fit declined on this sounding: {nose.reason}")
    assert nose.muf_mhz == pytest.approx(result.pick.muf_mhz, abs=2.0)
    assert nose.rms_residual_mhz < 1.0
