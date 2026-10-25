"""muf.sim.forecast: the persistence forecast, and its score against the truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from muf import cli
from muf.sim import forecast, observe, presets, sounding
from muf.sim.state import (
    FORECAST,
    MEASURED,
    MODE_FIELDS,
    PRIOR,
    ChannelState,
    Circuit,
    Mode,
    Provenance,
    Sweep,
    same_curves,
)

#: A short sweep, 10-14 MHz in 40 s, so a sounding is 1.6 M samples.
SHORT = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)

#: What observe measures off a power ionogram; the rest is a prior.
SEEN = ("delay", "cn0", "band")


def _mode(name, delay=9137.0, band=(13.3, 21.7), cn0=41.3, slope=23.9,
          provenance=None, **kw) -> Mode:
    f = np.linspace(band[0], band[1], 43)
    origin = {k: Provenance(MEASURED if k in SEEN else PRIOR, "a test")
              for k in MODE_FIELDS}
    origin.update(provenance or {})
    return Mode(name, f, delay + slope * (f - band[0]), cn0, provenance=origin, **kw)


def _state(*modes, epoch=1770206410.0, **metrics) -> ChannelState:
    return ChannelState(Circuit("cyprus1", "yoshkar-ola"), epoch, list(modes),
                        source="a.lfs", metrics=metrics,
                        provenance={"epoch": Provenance(MEASURED, "sounding header")})


def _pair(epoch=1770206410.0) -> ChannelState:
    return _state(_mode("1F"), _mode("2F", delay=9871.0, band=(15.1, 19.4), cn0=33.7),
                  epoch=epoch)


def _matched(score, name: str) -> dict:
    return next(r for r in score.modes
                if r["status"] == "matched" and r["truth_mode"] == name)


def _unnamed(score) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in ("forecast_mode", "truth_mode")}
            for r in score.modes]


# --- persist (T-1) ------------------------------------------------------------

def test_persist_keeps_the_curves():
    state = _pair()
    assert same_curves(forecast.persist(state, 300.0).modes, state.modes)


def test_persist_moves_the_epoch_by_the_lead():
    assert forecast.persist(_pair(), 300.0).epoch == 1770206710.0


def test_persist_marks_every_mode_field_a_forecast():
    f = forecast.persist(_pair(), 300.0)
    assert {m.origin(k) for m in f.modes for k in MODE_FIELDS} == {FORECAST}


def test_persist_marks_the_states_own_fields_a_forecast():
    f = forecast.persist(_pair(), 300.0)
    assert {p.origin for p in f.provenance.values()} == {FORECAST}


def test_persist_keeps_the_former_origin_in_the_note():
    f = forecast.persist(_pair(), 300.0)
    assert forecast.basis(f.modes[0].provenance["doppler_spread"]) == PRIOR


def test_persist_names_its_source_and_lead_in_the_note():
    note = forecast.persist(_pair(), 300.0).modes[0].provenance["delay"].note
    assert note == "persistence of measured: a test from a.lfs, lead 300 s"


def test_persist_leaves_its_input_alone():
    state = _pair()
    before = state.to_dict()
    forecast.persist(state, 300.0)
    assert state.to_dict() == before


def test_a_forecast_survives_its_json():
    f = forecast.persist(_pair(), 300.0)
    back = ChannelState.from_dict(json.loads(json.dumps(f.to_dict())))
    assert back.to_dict() == f.to_dict()


@pytest.mark.parametrize("lead", [0.0, -300.0, float("nan")])
def test_persist_refuses_a_lead_that_is_not_ahead(lead):
    with pytest.raises(ValueError, match="lead"):
        forecast.persist(_pair(), lead)


# --- a state against itself (T-2) --------------------------------------------

def test_a_state_against_itself_has_no_mode_error():
    state = _pair()
    score = forecast.score(forecast.persist(state, 300.0), state)
    assert {r[k] for r in score.modes for k in forecast.MODE_ERRORS
            if r[k] is not None} == {0.0}


def test_a_state_against_itself_misses_nothing():
    state = _pair()
    counts = forecast.score(forecast.persist(state, 300.0), state).pair["modes"]
    assert (counts["matched"], counts["missed"], counts["false"]) == (2, 0, 0)


def test_a_state_against_itself_has_no_muf_error():
    state = _state(_mode("1F"), muf_mhz=21.7)
    score = forecast.score(forecast.persist(state, 300.0), state)
    assert score.pair["muf_mhz"]["error"] == 0.0


# --- a known change (T-3, T-4) -------------------------------------------------

def test_a_later_forecast_reads_as_a_positive_delay_bias():
    f = forecast.persist(_state(_mode("1F", delay=9147.0)), 300.0)
    score = forecast.score(f, _state(_mode("1F")))
    assert _matched(score, "1F")["delay_bias_us"] == pytest.approx(10.0)


def test_a_constant_delay_offset_is_its_own_rms():
    f = forecast.persist(_state(_mode("1F", delay=9147.0)), 300.0)
    score = forecast.score(f, _state(_mode("1F")))
    assert _matched(score, "1F")["delay_rms_us"] == pytest.approx(10.0)


def test_a_delay_offset_leaves_the_cn0_alone():
    f = forecast.persist(_state(_mode("1F", delay=9147.0)), 300.0)
    score = forecast.score(f, _state(_mode("1F")))
    assert _matched(score, "1F")["cn0_bias_db"] == pytest.approx(0.0)


def test_cn0_is_not_compared_where_the_forecast_is_under_the_floor():
    f = forecast.persist(_state(_mode("1F", cn0=21.4)), 300.0)
    score = forecast.score(f, _state(_mode("1F")))
    assert _matched(score, "1F")["cn0_bias_db"] is None


def test_a_steeper_forecast_reads_as_a_gdd_bias():
    f = forecast.persist(_state(_mode("1F", slope=28.9)), 300.0)
    score = forecast.score(f, _state(_mode("1F")))
    assert _matched(score, "1F")["gdd_bias_us_per_mhz"] == pytest.approx(5.0)


def test_a_lower_top_reads_as_a_negative_muf_error():
    f = forecast.persist(_state(_mode("1F", band=(13.3, 21.2))), 300.0)
    score = forecast.score(f, _state(_mode("1F")))
    assert score.pair["mode_muf_mhz"]["error"] == pytest.approx(-0.5)


def test_the_extractors_muf_is_scored_from_the_metrics():
    f = forecast.persist(_state(_mode("1F"), muf_mhz=30.161), 300.0)
    score = forecast.score(f, _state(_mode("1F"), muf_mhz=30.980))
    assert score.pair["muf_mhz"]["error"] == pytest.approx(-0.819)


def test_a_truth_at_the_band_edge_is_censored():
    f = forecast.persist(_state(_mode("1F", band=(13.3, 21.2))), 300.0)
    score = forecast.score(f, _state(_mode("1F"), band_edge_mhz=21.6))
    assert score.pair["mode_muf_mhz"]["censored_truth"]


def test_a_censored_truth_bounds_the_error_one_way():
    f = forecast.persist(_state(_mode("1F", band=(13.3, 21.2))), 300.0)
    score = forecast.score(f, _state(_mode("1F"), band_edge_mhz=21.6))
    assert score.pair["mode_muf_mhz"]["error_bound"] == pytest.approx(0.5)


def test_a_forecast_from_the_band_edge_is_censored():
    f = forecast.persist(_state(_mode("1F"), muf_mhz=21.7, band_edge_mhz=21.6), 300.0)
    score = forecast.score(f, _state(_mode("1F"), muf_mhz=21.7))
    assert score.pair["muf_mhz"]["censored_forecast"]


def test_a_truth_recorded_short_has_its_muf_censored_below_the_edge():
    f = forecast.persist(_state(_mode("1F"), muf_mhz=21.7), 300.0)
    score = forecast.score(f, _state(_mode("1F", band=(13.3, 15.7)), muf_mhz=15.74,
                                     band_edge_mhz=17.25, sweep_complete=0.0))
    assert score.pair["muf_mhz"]["censored_truth"], "a short sweep's MUF passed as real"


def test_a_forecast_recorded_short_has_its_muf_censored_below_the_edge():
    f = forecast.persist(_state(_mode("1F", band=(13.3, 15.7)), muf_mhz=15.74,
                                band_edge_mhz=17.25, sweep_complete=0.0), 300.0)
    score = forecast.score(f, _state(_mode("1F"), muf_mhz=21.7))
    assert score.pair["muf_mhz"]["censored_forecast"]


def test_a_mode_in_a_truth_recorded_short_has_its_top_censored():
    f = forecast.persist(_state(_mode("1F")), 300.0)
    score = forecast.score(f, _state(_mode("1F", band=(13.3, 15.7)),
                                     band_edge_mhz=17.25, sweep_complete=0.0))
    assert _matched(score, "1F")["f_hi_censored_truth"]


def test_a_truth_at_the_band_floor_censors_the_lof():
    f = forecast.persist(_state(_mode("1F", band=(13.9, 21.7))), 300.0)
    score = forecast.score(f, _state(_mode("1F"), lof_edge_mhz=13.4))
    assert score.pair["lof_mhz"]["censored_truth"]


def test_a_mode_reaching_the_band_edge_has_its_top_censored():
    f = forecast.persist(_state(_mode("1F")), 300.0)
    score = forecast.score(f, _state(_mode("1F"), band_edge_mhz=21.6))
    assert _matched(score, "1F")["f_hi_censored_truth"]


# --- modes that come and go (T-5, T-6) ----------------------------------------

def test_a_mode_gone_from_the_truth_is_a_false_mode():
    score = forecast.score(forecast.persist(_pair(), 300.0), _state(_mode("1F")))
    assert score.pair["modes"]["false"] == 1


def test_a_mode_new_in_the_truth_is_missed():
    score = forecast.score(forecast.persist(_state(_mode("1F")), 300.0), _pair())
    assert score.pair["modes"]["missed"] == 1


def test_a_mode_gone_leaves_the_others_errors_alone():
    truth = _state(_mode("1F", delay=9131.4))
    whole = forecast.score(forecast.persist(_pair(), 300.0), truth)
    alone = forecast.score(forecast.persist(_state(_mode("1F")), 300.0), truth)
    assert _matched(whole, "1F") == _matched(alone, "1F")


def test_renaming_the_truths_modes_changes_no_metric():
    f = forecast.persist(_pair(), 300.0)
    truth = _pair()
    renamed = _state(_mode("F#2"), _mode("F", delay=9871.0, band=(15.1, 19.4),
                                         cn0=33.7))
    assert _unnamed(forecast.score(f, renamed)) == _unnamed(forecast.score(f, truth))


def test_the_strongest_truth_mode_is_the_main_one():
    score = forecast.score(forecast.persist(_pair(), 300.0), _pair())
    assert _matched(score, "1F")["class"] == "main"


def test_a_long_mode_outweighs_a_short_louder_one_as_main():
    state = _state(_mode("1F", band=(16.6, 30.6), cn0=33.0),
                   _mode("F", delay=9871.0, band=(28.2, 31.0), cn0=37.0))
    score = forecast.score(forecast.persist(state, 300.0), state)
    assert _matched(score, "1F")["class"] == "main"


def test_a_high_ray_is_classed_apart():
    state = _state(_mode("1F"), _mode("1F-high", delay=9871.0, band=(15.1, 19.4),
                                      cn0=33.7, branch="high"))
    score = forecast.score(forecast.persist(state, 300.0), state)
    assert _matched(score, "1F-high")["class"] == "high"


def test_a_high_ray_forecast_keeps_the_strongest_truth_mode_out_of_main():
    f = forecast.persist(_state(_mode("1F-high", cn0=46.9, branch="high")), 300.0)
    score = forecast.score(f, _state(_mode("F", cn0=37.0)))
    assert _matched(score, "F")["class"] == "high"


# --- priors (T-7) -------------------------------------------------------------

def test_a_prior_field_gives_no_error():
    truth = _state(_mode("1F", provenance={"cn0": Provenance(PRIOR, "a test")}))
    score = forecast.score(forecast.persist(_state(_mode("1F")), 300.0), truth)
    assert _matched(score, "1F")["cn0_bias_db"] is None


def test_a_prior_field_is_counted_as_excluded():
    truth = _state(_mode("1F", provenance={"cn0": Provenance(PRIOR, "a test")}))
    score = forecast.score(forecast.persist(_state(_mode("1F")), 300.0), truth)
    assert score.pair["excluded_prior_fields"] == len(MODE_FIELDS) - 2


def test_a_delay_spread_measured_in_both_is_scored():
    seen = {"delay_spread": Provenance(MEASURED, "a test")}
    f = forecast.persist(_state(_mode("1F", delay_spread_us=12.0, provenance=seen)),
                         300.0)
    score = forecast.score(f, _state(_mode("1F", delay_spread_us=9.0, provenance=seen)))
    assert _matched(score, "1F")["delay_spread_error_us"] == pytest.approx(3.0)


def test_a_delay_spread_assumed_in_one_is_not_scored():
    f = forecast.persist(_state(_mode("1F", delay_spread_us=12.0)), 300.0)
    score = forecast.score(f, _state(_mode("1F", delay_spread_us=9.0)))
    assert _matched(score, "1F")["delay_spread_error_us"] is None


def test_the_trace_width_is_scored_from_the_metrics():
    f = forecast.persist(_state(_mode("1F"), cross_trace={"1F": {"spread_us": 14.8}}),
                         300.0)
    truth = _state(_mode("F"), cross_trace={"F": {"spread_us": 10.3}})
    score = forecast.score(f, truth)
    assert _matched(score, "F")["width_error_us"] == pytest.approx(4.5)


# --- carriers and events (T-10) -----------------------------------------------

def _table() -> dict:
    return forecast.contingency([(True, True)] * 5 + [(True, False)] * 2
                                + [(False, True)] * 3 + [(False, False)] * 10)


def test_the_probability_of_detection_is_hits_over_events():
    assert _table()["pod"] == pytest.approx(5 / 8)


def test_the_false_alarm_ratio_is_false_alarms_over_forecasts():
    assert _table()["far"] == pytest.approx(2 / 7)


def test_the_critical_success_index_leaves_out_correct_negatives():
    assert _table()["csi"] == pytest.approx(5 / 10)


def test_the_accuracy_counts_both_kinds_of_right():
    assert _table()["accuracy"] == pytest.approx(15 / 20)


def test_the_base_rate_is_how_often_the_truth_had_it():
    assert _table()["base_rate"] == pytest.approx(8 / 20)


def test_a_table_with_no_events_has_no_detection_rate():
    assert forecast.contingency([(False, False)] * 4)["pod"] is None


def _swept(*modes) -> ChannelState:
    state = _state(*modes)
    state.sweep = SHORT
    return state


def test_the_carriers_default_to_the_sweep():
    state = _swept(_mode("1F", band=(10.3, 13.7)))
    score = forecast.score(forecast.persist(state, 300.0), state)
    assert [r["f0_mhz"] for r in score.carriers] == list(np.arange(10.0, 14.01, 0.25))


def test_a_state_against_itself_has_no_carrier_error():
    state = _swept(_mode("1F", band=(10.3, 13.7)),
                   _mode("2F", delay=9871.0, band=(11.1, 12.9), cn0=33.7))
    score = forecast.score(forecast.persist(state, 300.0), state)
    assert {v["error"] for r in score.carriers for v in r["figures"].values()
            if v["error"] is not None} == {0.0}


def test_a_carrier_seen_by_one_side_only_has_no_error():
    f = forecast.persist(_swept(_mode("1F", band=(10.3, 13.7))), 300.0)
    score = forecast.score(f, _swept(_mode("1F", band=(10.3, 12.7))))
    row = next(r for r in score.carriers if r["f0_mhz"] == 13.0)
    assert row["figures"]["first_delay_us"]["error"] is None


def test_a_carrier_seen_by_one_side_only_is_a_false_alarm():
    f = forecast.persist(_swept(_mode("1F", band=(10.3, 13.7))), 300.0)
    score = forecast.score(f, _swept(_mode("1F", band=(10.3, 12.7))))
    assert score.pair["carriers"]["events"]["mode"]["false_alarms"] == 4


def test_a_carrier_past_the_truths_recorded_band_is_left_out():
    f = forecast.persist(_swept(_mode("1F", band=(10.3, 13.7))), 300.0)
    truth = _swept(_mode("1F", band=(10.3, 12.7)))
    truth.metrics["band_edge_mhz"] = 12.8
    score = forecast.score(f, truth)
    assert 13.0 not in [r["f0_mhz"] for r in score.carriers], \
        "a carrier the truth never recorded was scored as no mode"


def test_the_carriers_left_out_are_counted():
    f = forecast.persist(_swept(_mode("1F", band=(10.3, 13.7))), 300.0)
    truth = _swept(_mode("1F", band=(10.3, 12.7)))
    truth.metrics["band_edge_mhz"] = 12.8
    assert forecast.score(f, truth).pair["carriers"]["outside_band"] == 5


def test_a_prior_doppler_spread_leaves_the_coherence_time_unscored():
    state = _swept(_mode("1F", band=(10.3, 13.7), doppler_spread_hz=0.5))
    score = forecast.score(forecast.persist(state, 300.0), state)
    row = next(r for r in score.carriers if r["f0_mhz"] == 12.0)
    assert "coherence_time_s" in row["excluded"]


def test_one_mode_with_a_prior_delay_spread_leaves_the_delay_spread_unscored():
    state = _swept(_mode("1F", band=(10.3, 13.7)))
    score = forecast.score(forecast.persist(state, 300.0), state)
    row = next(r for r in score.carriers if r["f0_mhz"] == 12.0)
    assert row["figures"]["delay_spread_us"]["error"] is None


def test_two_modes_score_their_delay_spread_with_priors_of_their_own():
    state = _swept(_mode("1F", band=(10.3, 13.7)),
                   _mode("2F", delay=9871.0, band=(11.1, 12.9), cn0=33.7))
    score = forecast.score(forecast.persist(state, 300.0), state)
    row = next(r for r in score.carriers if r["f0_mhz"] == 12.0)
    assert row["figures"]["delay_spread_us"]["error"] == 0.0


def test_the_carrier_rows_are_read_against_the_truths_muf():
    state = _swept(_mode("1F", band=(10.3, 13.7)))
    state.metrics["muf_mhz"] = 13.6
    score = forecast.score(forecast.persist(state, 300.0), state)
    row = next(r for r in score.carriers if r["f0_mhz"] == 12.0)
    assert row["f_over_muf"] == pytest.approx(12.0 / 13.6)


# --- the ionogram itself (T-11) ------------------------------------------------

def _law_state(**kw) -> ChannelState:
    law = presets.law_mode(muf_mhz=14.0, f_lo_mhz=10.5, cn0_dbhz=45.0, spread_hz=0.5,
                           **kw)
    return ChannelState(presets.cyprus1_circuit(), 1770206410.0, [law], sweep=SHORT,
                        source="test")


def _moved(state: ChannelState, us: float) -> ChannelState:
    m = state.modes[0]
    return ChannelState(state.circuit, state.epoch,
                        [Mode(m.name, m.freq_mhz, m.delay_us + us, m.cn0_dbhz,
                              doppler_spread_hz=m.doppler_spread_hz)],
                        sweep=state.sweep, source=state.source)


def _image(tmp_path, truth_us: float) -> dict:
    state = _law_state()
    a = forecast.sounded(state, 1, seed=98, workdir=tmp_path)[0]
    b = forecast.sounded(_moved(state, truth_us), 1, seed=99, workdir=tmp_path)[0]
    made = forecast.sounded(state, 4, workdir=tmp_path)
    return forecast.image(a, b, made, state.modes)


def test_a_state_against_its_own_sounding_reaches_the_ceiling(tmp_path):
    assert abs(_image(tmp_path, 0.0)["smoothed"]["dz_ceiling"]) < 0.15


def test_a_trace_three_bins_off_falls_under_the_ceiling(tmp_path):
    assert _image(tmp_path, 150.0)["smoothed"]["dz_ceiling"] > 0.5


def test_a_trace_three_bins_off_lowers_the_states_correlation(tmp_path):
    same, off = _image(tmp_path, 0.0), _image(tmp_path, 150.0)
    assert off["rows"]["state"] < same["rows"]["state"] - 0.1


def test_a_row_without_data_leaves_the_comparison_finite(tmp_path):
    state = _law_state()
    a = forecast.sounded(state, 1, seed=98, workdir=tmp_path)[0]
    b = forecast.sounded(state, 1, seed=99, workdir=tmp_path)[0]
    b.power[len(b.freq) // 2] = np.nan
    made = forecast.sounded(state, 2, workdir=tmp_path)
    assert np.isfinite(forecast.image(a, b, made, state.modes)["smoothed"]["ion"]), \
        "a row without data spread into the comparison"


def test_the_copies_are_deleted_once_read(tmp_path):
    forecast.sounded(_law_state(), 2, workdir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_two_soundings_sharing_their_range_have_it_in_common():
    assert forecast.common_span((1800.0, 3100.0), (1800.0, 3100.0)) == (1800.0, 3100.0)


def test_a_truth_mostly_outside_the_forecasts_range_shares_none():
    assert forecast.common_span((1800.0, 3100.0), (2900.0, 4200.0)) is None


def test_a_truth_mostly_inside_the_forecasts_range_shares_the_overlap():
    assert forecast.common_span((1800.0, 3100.0), (1900.0, 3200.0)) == (1900.0, 3100.0)


# --- the pair's row -----------------------------------------------------------

def test_the_pair_row_is_json():
    score = forecast.score(forecast.persist(_pair(), 300.0), _pair(epoch=1770206711.0))
    assert json.loads(json.dumps(score.to_dict(), allow_nan=False))["lead_actual_s"] \
        == 301.0


# --- the command --------------------------------------------------------------

def _soundings(tmp_path) -> list[str]:
    law = presets.law_mode(muf_mhz=14.0, f_lo_mhz=10.5, cn0_dbhz=45.0, spread_hz=0.0)
    return [str(sounding.write_lfs(
        ChannelState(presets.cyprus1_circuit(), 1770206410.0 + 300.0 * k, [law],
                     sweep=SHORT, source="test"),
        tmp_path / f"s{k}.lfs", seed=k)) for k in range(2)]


def test_the_command_writes_a_forecast(tmp_path):
    a, _ = _soundings(tmp_path)
    cli.main(["sim", "forecast", a, "--out", str(tmp_path / "f.json")])
    state = ChannelState.load(tmp_path / "f.json")
    assert {m.origin("delay") for m in state.modes} == {FORECAST}


def test_the_command_scores_the_next_sounding(tmp_path):
    a, b = _soundings(tmp_path)
    cli.main(["sim", "forecast", a, "--next", b, "--json", str(tmp_path / "p.json")])
    pair = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert pair["modes"]["matched"] >= 1


def _swept_for(tmp_path, name: str, seconds: float) -> str:
    law = presets.law_mode(muf_mhz=14.0, f_lo_mhz=10.5, cn0_dbhz=45.0, spread_hz=0.0)
    sweep = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=seconds,
                  sample_rate_hz=40e3)
    return str(sounding.write_lfs(
        ChannelState(presets.cyprus1_circuit(), 1770206410.0, [law], sweep=sweep,
                     source="test"), tmp_path / name, seed=len(name)))


def test_a_truth_swept_short_is_compared_on_the_band_both_hold(tmp_path):
    a, b = _swept_for(tmp_path, "a.lfs", 40.0), _swept_for(tmp_path, "bb.lfs", 32.0)
    origin = observe.from_ionogram(observe.load(a), source="a.lfs")
    score = forecast.evaluate(forecast.persist(origin, 300.0), b, origin=a, copies=2)
    assert "rows" in score.ionogram, "a short truth sweep left the ionograms apart"


def test_a_forecast_swept_short_of_the_truths_band_skips_the_ionograms(tmp_path):
    a, b = _swept_for(tmp_path, "a.lfs", 16.0), _swept_for(tmp_path, "bb.lfs", 40.0)
    origin = observe.from_ionogram(observe.load(a), source="a.lfs")
    score = forecast.evaluate(forecast.persist(origin, 300.0), b, origin=a, copies=2)
    assert "skipped" in score.ionogram


def test_a_recording_cut_short_is_observed_as_incomplete(tmp_path):
    whole = Path(_swept_for(tmp_path, "a.lfs", 40.0))
    cut = tmp_path / "cut.lfs"
    cut.write_bytes(whole.read_bytes()[:whole.stat().st_size * 3 // 5])
    state = observe.from_ionogram(observe.load(cut), source="cut.lfs")
    assert state.metrics["sweep_complete"] == 0.0


def test_the_command_compares_the_ionograms_with_copies(tmp_path):
    a, b = _soundings(tmp_path)
    cli.main(["sim", "forecast", a, "--next", b, "--copies", "2",
              "--json", str(tmp_path / "p.json")])
    pair = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert pair["ionogram"]["copies"] == 2
