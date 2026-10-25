"""muf.sim.summary: a run's rows summed up, with block-bootstrap intervals."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from muf.sim import forecast, summary
from muf.sim.state import (
    MEASURED,
    MODE_FIELDS,
    PRIOR,
    ChannelState,
    Circuit,
    Mode,
    Provenance,
    Sweep,
)

SHORT = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)
T0 = 1770206410.0


def _ar1(n: int, rho: float, rng) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for k in range(1, n):
        x[k] = rho * x[k - 1] + math.sqrt(1 - rho ** 2) * rng.standard_normal()
    return x


def _interval(x: np.ndarray, block: int, reps: int = 500) -> list[float]:
    m = summary.resamples([(0, x.size)], block=block, reps=reps,
                          rng=np.random.default_rng(0))
    return summary.metrics(x, np.ones(x.size), np.arange(x.size), m,
                           quantiles=False)["ci"]["bias"]


# --- the block bootstrap (T-13) ----------------------------------------------

def test_the_resamples_are_the_same_from_the_same_seed():
    a = summary.resamples([(0, 50), (50, 97)], block=12, reps=40,
                          rng=np.random.default_rng(0))
    b = summary.resamples([(0, 50), (50, 97)], block=12, reps=40,
                          rng=np.random.default_rng(0))
    assert np.array_equal(a, b)


def test_every_resample_has_as_many_pairs_as_the_run():
    m = summary.resamples([(0, 50), (50, 97)], block=12, reps=40,
                          rng=np.random.default_rng(3))
    assert set(m.sum(axis=1)) == {97}


def test_a_gap_breaks_the_run_into_segments():
    assert summary.segments([0.0, 300.0, 600.0, 1200.0, 1500.0], gap_s=330.0) \
        == [(0, 3), (3, 5)]


def test_blocks_widen_the_interval_of_a_correlated_series():
    x = _ar1(360, 0.9, np.random.default_rng(11))
    lo, hi = _interval(x, 12)
    lo1, hi1 = _interval(x, 1)
    assert hi - lo > 1.5 * (hi1 - lo1)


def test_the_block_interval_covers_the_true_mean_about_as_often_as_it_says():
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(200):
        lo, hi = _interval(_ar1(360, 0.3, rng), 12, reps=300)
        hits += lo <= 0.0 <= hi
    assert 0.9 <= hits / 200 <= 0.99, "the block interval misses the mean too often"


# --- the metrics (6.2) --------------------------------------------------------

def _m(errors, weights=None, **kw) -> dict:
    e = np.asarray(errors, dtype=float)
    w = np.ones(e.size) if weights is None else np.asarray(weights, dtype=float)
    m = summary.resamples([(0, e.size)], block=1, reps=20, rng=np.random.default_rng(0))
    return summary.metrics(e, w, np.arange(e.size), m, minimum=1, **kw)


def test_the_bias_is_the_mean_error():
    assert _m([1.0, -3.0, 5.0, 1.0])["bias"] == pytest.approx(1.0)


def test_the_mae_is_the_mean_absolute_error():
    assert _m([1.0, -3.0, 5.0, 1.0])["mae"] == pytest.approx(2.5)


def test_the_rmse_is_the_root_mean_square():
    assert _m([1.0, -3.0, 5.0, 1.0])["rmse"] == pytest.approx(3.0)


def test_the_weights_weigh_the_mean():
    assert _m([1.0, 4.0], [3.0, 1.0])["bias"] == pytest.approx(1.75)


def test_the_median_is_of_the_absolute_errors():
    assert _m([-7.0, 1.0, 2.0])["median"] == pytest.approx(2.0)


def test_the_robust_spread_is_the_scaled_median_deviation():
    assert _m([1.0, 2.0, 4.0, 8.0, 100.0])["robust_sd"] == pytest.approx(1.4826 * 3.0)


def test_a_metric_is_not_given_under_its_minimum_of_pairs():
    e = np.ones(10)
    m = summary.resamples([(0, 10)], block=1, reps=5, rng=np.random.default_rng(0))
    assert "bias" not in summary.metrics(e, e, np.arange(10), m, minimum=30)


def test_a_metric_under_its_minimum_says_why():
    e = np.ones(10)
    m = summary.resamples([(0, 10)], block=1, reps=5, rng=np.random.default_rng(0))
    assert summary.metrics(e, e, np.arange(10), m, minimum=30)["reason"] \
        == "10 pairs, under 30"


def test_the_interval_holds_the_estimate():
    out = _m(list(np.random.default_rng(2).normal(1.0, 1.0, 200)))
    assert out["ci"]["mae"][0] <= out["mae"] <= out["ci"]["mae"][1]


def test_the_quantile_interval_holds_the_estimate():
    out = _m(list(np.random.default_rng(2).normal(1.0, 1.0, 200)))
    assert out["ci"]["p90"][0] <= out["p90"] <= out["ci"]["p90"][1]


# --- a run's rows -------------------------------------------------------------

def _mode(name, delay=9137.0, band=(10.3, 13.7), cn0=41.3) -> Mode:
    f = np.linspace(band[0], band[1], 35)
    origin = {k: Provenance(MEASURED if k in ("delay", "cn0", "band") else PRIOR, "t")
              for k in MODE_FIELDS}
    return Mode(name, f, delay + 23.9 * (f - band[0]), cn0, provenance=origin)


def _state(epoch, delay, top, muf, edge=99.0) -> ChannelState:
    return ChannelState(Circuit("cyprus1", "yoshkar-ola"), epoch,
                        [_mode("1F", delay=delay, band=(10.3, top))], sweep=SHORT,
                        source=f"s{int(epoch - T0):06d}.lfs",
                        metrics={"muf_mhz": muf, "band_edge_mhz": edge})


def _write(run: Path, name: str, rows: list[dict]) -> None:
    with (run / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _run(tmp_path, n=48, seed=1, edge=99.0) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps(
        {"params": {"lead_s": 300.0, "tolerance_s": 30.0}}), encoding="utf-8")
    rng = np.random.default_rng(seed)
    delay = 9137.0 + np.cumsum(rng.normal(0.0, 20.0, n + 1))
    top = 13.0 + 0.4 * np.sin(np.arange(n + 1) / 7.0)
    for k in range(n):
        a = _state(T0 + 300.0 * k, delay[k], top[k], top[k], edge)
        b = _state(T0 + 300.0 * (k + 1), delay[k + 1], top[k + 1], top[k + 1], edge)
        score = forecast.score(forecast.persist(a, 300.0), b)
        score.pair["zenith_deg"] = 60.0 + k
        _write(run, "modes", score.modes)
        _write(run, "carriers", score.carriers)
        _write(run, "pairs", [score.pair])
    return run


def test_the_muf_error_is_summed_over_the_pairs(tmp_path):
    rows = [json.loads(x) for x in
            (_run(tmp_path) / "pairs.jsonl").read_text("utf-8").splitlines()]
    mae = float(np.mean([abs(r["muf_mhz"]["error"]) for r in rows]))
    assert summary.summarise(tmp_path / "run")["all"]["muf_mhz"]["mae"] \
        == pytest.approx(mae)


def test_a_censored_truth_leaves_the_main_mae(tmp_path):
    out = summary.summarise(_run(tmp_path, edge=13.0))["all"]["muf_mhz"]
    assert out["censored"]["truth"] > 0 and out["pairs"] + out["censored"]["truth"] \
        <= 48


def test_the_delay_error_is_summed_over_the_mode_pairs(tmp_path):
    out = summary.summarise(_run(tmp_path))["all"]["modes"]["all"]["delay_rms_us"]
    assert out["rmse"] > 0.0


def test_the_mode_classes_are_summed_apart(tmp_path):
    out = summary.summarise(_run(tmp_path))["all"]["modes"]
    assert set(out) == {"all", "main", "high", "other"}


def test_the_carrier_events_add_up_over_the_run(tmp_path):
    run = _run(tmp_path)
    rows = [json.loads(x) for x in (run / "carriers.jsonl").read_text("utf-8")
            .splitlines()]
    hits = sum(r["events"]["mode"]["forecast"] and r["events"]["mode"]["truth"]
               for r in rows)
    assert summary.summarise(run)["all"]["carriers"]["events"]["mode"]["hits"] == hits


def test_the_slices_are_by_the_hour_of_the_valid_time(tmp_path):
    out = summary.summarise(_run(tmp_path))
    assert sorted(out["by"]["hour"]) == ["12", "13", "14", "15", "16"]


def test_the_sun_slices_the_pairs_by_its_zenith_angle(tmp_path):
    out = summary.summarise(_run(tmp_path))
    assert sorted(out["by"]["sun"]) == ["day", "night", "transition"]


def test_a_run_summed_twice_gives_the_same_summary(tmp_path):
    run = _run(tmp_path)
    assert json.dumps(summary.summarise(run)) == json.dumps(summary.summarise(run))


def test_the_recovery_puts_the_forecasts_error_in_scale(tmp_path):
    run = _run(tmp_path)
    _write(run, "recovery", [{"from": "x", "to": "y", "seed": k, "mode": "1F",
                              "overlap_mhz": 3.0, "delay_rms_us": 10.0,
                              "muf_error_mhz": 0.1} for k in range(4)])
    out = summary.summarise(run)
    rmse = out["all"]["modes"]["all"]["delay_rms_us"]["rmse"]
    assert out["q"]["delay_rms_us"]["q"] == pytest.approx(10.0 / rmse)


def test_a_q_over_its_limit_marks_the_change_unresolved(tmp_path):
    run = _run(tmp_path)
    _write(run, "recovery", [{"from": "x", "to": "y", "seed": 0, "mode": "1F",
                              "overlap_mhz": 3.0, "delay_rms_us": 1e4,
                              "muf_error_mhz": None}])
    assert summary.summarise(run)["q"]["delay_rms_us"]["unresolved"]


def test_the_ionograms_average_in_fishers_z(tmp_path):
    run = _run(tmp_path)
    rows = [{"from": f"s{300 * k:06d}.lfs", "to": f"s{300 * (k + 1):06d}.lfs",
             "copies": 4, "cells": 100,
             "rows": {"ion": 0.5, "state": 0.4 + 0.1 * (k % 2), "expected": 0.6,
                      "ceiling": 0.7, "dz_ion": 0.0, "dz_ceiling": 0.3},
             "smoothed": {"ion": 0.9, "state": 0.8, "expected": 0.85, "ceiling": 0.95,
                          "dz_ion": -0.1, "dz_ceiling": 0.5},
             "level_db": {"bias": 0.0, "rms": 3.0}} for k in range(48)]
    _write(run, "ionogram", rows)
    out = summary.summarise(run)["all"]["ionogram"]["rows"]["state"]
    assert out["mean"] == pytest.approx(math.tanh(np.mean(np.arctanh([0.4, 0.5]))))


def test_the_carriers_left_outside_a_band_are_summed(tmp_path):
    out = summary.summarise(_run(tmp_path, edge=13.0))["all"]
    assert out["carriers_outside_band"] == 48 * 5


def test_a_censored_mode_top_is_counted_apart(tmp_path):
    run = _run(tmp_path, edge=13.0)
    rows = [json.loads(x) for x in
            (run / "modes.jsonl").read_text("utf-8").splitlines()]
    censored = sum(r["status"] == "matched" and (r["f_hi_censored_truth"]
                                                 or r["f_hi_censored_forecast"])
                   for r in rows)
    out = summary.summarise(run)["all"]["modes"]["all"]["f_hi_error_mhz"]
    assert (out["censored"], out["n"]) == (censored, 48 - censored)


def test_rows_in_another_order_sum_up_the_same(tmp_path):
    run = _run(tmp_path)
    before = json.dumps(summary.summarise(run, reps=50))
    for name in ("modes", "carriers", "pairs"):
        path = run / f"{name}.jsonl"
        lines = path.read_text("utf-8").splitlines()
        np.random.default_rng(len(name)).shuffle(lines)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert json.dumps(summary.summarise(run, reps=50)) == before, \
        "the summary hangs on the order the rows were written in"
