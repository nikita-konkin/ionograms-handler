"""Tracking MUF through time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from muf import track


def times(n=100, minutes=5):
    return pd.date_range("2026-02-04 00:00:00", periods=n, freq=f"{minutes}min")


def test_follows_a_smooth_curve():
    t = times(100)
    truth = 12 + 10 * np.sin(np.linspace(0, np.pi, 100))
    result = track.track(t, truth, np.full(100, 0.3))

    assert result.n_measured == 100
    assert result.n_rejected == 0
    assert np.abs(result.frame["muf"].to_numpy() - truth).max() < 1.0


def test_fills_gaps_with_honest_uncertainty():
    t = times(100)
    truth = np.linspace(10, 20, 100)
    values = truth.copy()
    values[40:50] = np.nan

    result = track.track(t, values, np.full(100, 0.3))
    frame = result.frame

    assert result.n_filled == 10
    assert frame["muf"].notna().all()
    # Filled points are still close, but reported as less certain.
    assert np.abs(frame["muf"][40:50].to_numpy() - truth[40:50]).max() < 1.5
    assert frame["sigma"][40:50].mean() > frame["sigma"][:40].mean()


def test_rejects_an_isolated_outlier():
    t = times(60)
    truth = np.full(60, 15.0)
    values = truth.copy()
    values[30] = 40.0                      # a 25 MHz spike

    result = track.track(t, values, np.full(60, 0.3))

    assert result.n_rejected == 1
    assert result.frame["rejected"][30]
    assert result.frame["muf"][30] == pytest.approx(15.0, abs=1.0)


def test_does_not_reject_a_genuine_fast_transition():
    """Sunrise is real and steep; the gate must not smooth it away."""
    t = times(72)
    truth = np.concatenate([np.full(24, 12.0),
                            np.linspace(12, 30, 24),      # ~9 MHz/hour
                            np.full(24, 30.0)])
    result = track.track(t, truth, np.full(72, 0.3))

    assert result.n_rejected <= 2
    assert result.frame["muf"].iloc[-1] == pytest.approx(30.0, abs=1.5)
    assert result.frame["rate_mhz_per_hour"].max() > 4.0


def test_output_is_smoother_than_the_input():
    rng = np.random.default_rng(0)
    t = times(120)
    truth = 15 + 5 * np.sin(np.linspace(0, 2 * np.pi, 120))
    noisy = truth + rng.normal(0, 1.0, 120)

    result = track.track(t, noisy, np.full(120, 1.0))
    raw_step = np.abs(np.diff(noisy)).mean()
    tracked_step = np.abs(np.diff(result.frame["muf"].to_numpy())).mean()

    assert tracked_step < raw_step / 2


def test_weights_by_measurement_uncertainty():
    """A pick declared uncertain must move the state less than a confident one.

    The gate is disabled here so that weighting is what is being measured; with
    it on, a departure this large from a tight-sigma series is rejected outright
    rather than merely down-weighted (see the next test).
    """
    t = times(40)
    values = np.full(40, 15.0)
    values[20] = 18.0
    loose = 1e9      # effectively no gating

    confident = track.track(t, values, np.full(40, 0.2), gate_sigma=loose)
    doubtful = track.track(t, values, np.concatenate(
        [np.full(20, 0.2), [5.0], np.full(19, 0.2)]), gate_sigma=loose)

    assert confident.frame["muf"][20] > doubtful.frame["muf"][20]


def test_a_confident_series_gates_out_a_departure_entirely():
    """With tight uncertainty, a lone 3 MHz jump is rejected, not blended in.

    This is the desirable behaviour: among consistent, well-resolved picks, one
    disagreeing sounding is far more likely to be an extraction failure than a
    real 3 MHz excursion in five minutes.
    """
    t = times(40)
    values = np.full(40, 15.0)
    values[20] = 18.0

    result = track.track(t, values, np.full(40, 0.2))

    assert result.frame["rejected"][20]
    assert result.frame["muf"][20] == pytest.approx(15.0, abs=0.1)


def test_requires_at_least_one_measurement():
    with pytest.raises(ValueError, match="no measurements"):
        track.track(times(10), np.full(10, np.nan), np.full(10, 0.3))


def test_requires_timestamps():
    with pytest.raises(ValueError, match="no timestamps"):
        track.track([], [], [])


def test_handles_unsorted_input():
    t = times(20)
    truth = np.linspace(10, 20, 20)
    order = np.array([5, 0, 12, 3] + [i for i in range(20)
                                      if i not in (5, 0, 12, 3)])

    result = track.track(t[order], truth[order], np.full(20, 0.3))
    assert result.frame["datetime"].is_monotonic_increasing


# --- measurement uncertainty from pick quality -------------------------------

def _results(n=10, run=20, snr=55.0):
    return pd.DataFrame({
        "datetime": times(n),
        "muf_algo": np.full(n, 15.0),
        "run_algo": np.full(n, run),
        "snr_algo": np.full(n, snr),
        "limited_algo": [False] * n,
    })


def test_short_runs_are_treated_as_less_certain():
    long_run = track.measurement_sigma(_results(run=30), "algo")
    short_run = track.measurement_sigma(_results(run=2), "algo")

    assert short_run.mean() > long_run.mean() * 2


def test_weak_signal_is_treated_as_less_certain():
    frame = _results(n=10)
    frame.loc[5, "snr_algo"] = 20.0        # much weaker than the rest

    sigma = track.measurement_sigma(frame, "algo")
    assert sigma[5] > sigma[0]


def test_missing_quality_columns_fall_back_to_base():
    frame = pd.DataFrame({"datetime": times(5), "muf_algo": np.full(5, 15.0)})
    sigma = track.measurement_sigma(frame, "algo")

    assert np.allclose(sigma, track.DEFAULT_BASE_SIGMA_MHZ)


def test_track_results_excludes_band_limited():
    """Band-limited picks are lower bounds; they must not anchor the state."""
    frame = _results(n=20)
    frame.loc[10:14, "muf_algo"] = 32.5
    frame.loc[10:14, "limited_algo"] = True

    result = track.track_results(frame, "algo")
    assert result.n_filled == 5
    assert result.frame["muf"].max() < 20      # not dragged up to 32.5


def test_track_results_adds_the_method_column():
    result = track.track_results(_results(), "algo")
    assert (result.frame["method"] == "algo").all()


# --- against a real day ------------------------------------------------------

def test_real_day_is_smoothed_and_gaps_filled(real_dir):
    from muf.pipeline import Options, process_many

    frame = process_many(real_dir, Options(methods=("algo",)), jobs=0,
                         progress=False)
    result = track.track_results(frame, "algo")

    assert len(result.frame) == len(frame)
    assert result.frame["muf"].notna().all()

    raw = pd.to_numeric(frame["muf_algo"], errors="coerce")
    raw_step = raw.diff().abs().max()
    tracked_step = result.frame["muf"].diff().abs().max()
    assert tracked_step < raw_step / 3


# --------------------------------------------------------------------------
# LOF
# --------------------------------------------------------------------------
#
# The filter never cared which frequency it was following -- a random walk in
# rate describes the LOF as well as the MUF. What was MUF-only was the naming:
# the column it wrote, the censoring flag it dropped, and the signal level it
# weighted by. All three differ at the other end of the band, and getting any
# of them wrong produces a plausible curve of the wrong quantity.

def results_table(n=120, method="algo"):
    """A `muf.pipeline` results table with both ends of the band in it."""
    t = times(n)
    hour = np.arange(n) / 12.0
    return pd.DataFrame({
        "datetime": t,
        f"muf_{method}": 18 + 7 * np.sin(2 * np.pi * hour / 24),
        f"lof_{method}": 8 + 3 * np.sin(2 * np.pi * (hour - 2) / 24),
        f"snr_{method}": np.full(n, 55.0),
        f"lofsnr_{method}": np.full(n, 40.0),
        f"run_{method}": np.full(n, 30),
        f"limited_{method}": np.zeros(n, dtype=bool),
        f"loflim_{method}": np.zeros(n, dtype=bool),
    })


def test_the_value_column_is_named_after_the_parameter():
    """A tracked LOF handed on in a column called `muf` is how the wrong
    quantity ends up in a report."""
    t = times(40)
    result = track.track(t, np.linspace(5, 9, 40), np.full(40, 0.3), param="lof")
    assert "lof" in result.frame.columns
    assert "muf" not in result.frame.columns


def test_tracking_lof_reads_the_lof_column():
    frame = results_table()
    result = track.track_results(frame, method="algo", param="lof")
    assert result.n_measured == len(frame)
    # The LOF curve, not the MUF one that sits beside it in the same table.
    assert result.frame["lof"].max() < 12
    assert result.frame["lof"].min() > 4


def test_a_pick_at_the_band_floor_is_dropped_not_tracked():
    """`loflim` is an upper bound on LOF, exactly as `limited` is a lower
    bound on MUF: letting one anchor the state pulls the curve to the edge it
    ran into."""
    frame = results_table()
    frame.loc[40:59, "loflim_algo"] = True
    result = track.track_results(frame, method="algo", param="lof")
    assert result.n_filled == 20
    assert result.frame["sigma"][40:60].mean() > result.frame["sigma"][:40].mean()


def test_the_muf_censoring_flag_does_not_censor_the_lof():
    """The two ends censor independently -- a sweep that ran out of top has
    said nothing about its bottom."""
    frame = results_table()
    frame.loc[40:59, "limited_algo"] = True
    assert track.track_results(frame, method="algo", param="lof").n_filled == 0
    assert track.track_results(frame, method="algo", param="muf").n_filled == 20


def test_lof_is_weighted_by_the_signal_at_its_own_end_of_the_band():
    """`snr` is measured at the top of the trace. Weighting a LOF by it scales
    the uncertainty by the signal level at the other end of the band."""
    frame = results_table()
    frame.loc[:, "lofsnr_algo"] = 40.0
    frame.loc[50, "lofsnr_algo"] = 5.0          # weak at the bottom only

    sigma = track.measurement_sigma(frame, "algo", param="lof")
    assert sigma[50] > sigma[0] * 1.5
    # The MUF's own weighting is untouched by it.
    assert track.measurement_sigma(frame, "algo", param="muf")[50] == \
        pytest.approx(track.measurement_sigma(frame, "algo", param="muf")[0])


def test_an_unknown_parameter_is_refused():
    with pytest.raises(ValueError, match="unknown parameter"):
        track.track_results(results_table(), method="algo", param="fof2")
