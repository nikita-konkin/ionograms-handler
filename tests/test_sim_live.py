"""muf.sim.live: forecasts issued and scored as the soundings come."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from muf.sim import live, presets, series, sounding
from muf.sim.state import ChannelState, Sweep

#: A short sweep, 10-14 MHz in 40 s, so a sounding is 1.6 M samples.
SHORT = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)

#: 2026-02-04 12:00:10 UTC.
T0 = 1770206410.0


def _state(epoch: float, *, muf: float = 13.6, source: str = "s.lfs",
           traced: bool = True, offset: float | None = None) -> ChannelState:
    law = presets.law_mode(muf_mhz=muf, f_lo_mhz=10.5, cn0_dbhz=45.0, spread_hz=0.5)
    return ChannelState(presets.cyprus1_circuit(), epoch, [law] if traced else [],
                        sweep=SHORT, source=source,
                        metrics={} if offset is None else {"echo_offset_us": offset})


def _turns(epochs, *, leads=None, untraced=(), offsets=None) -> list[dict]:
    ledger, events = live.Ledger(), []
    for k, epoch in enumerate(epochs):
        state = _state(epoch, muf=13.6 + 0.1 * k, source=f"s{k}.lfs",
                       traced=k not in untraced,
                       offset=None if offsets is None else offsets[k])
        ledger, got = live.step(ledger, series.Session(Path(f"s{k}.lfs"), epoch,
                                                       "cyprus1", "yoshkar-ola"),
                                state, lambda pending, s=state: s, leads=leads)
        events.extend(got)
    return events


def _scored(lead_s: float, muf_error: float, *, at: str = "2026-02-04T12:10:10Z",
            censored: bool = False, delays=(), overlaps=None,
            suspect: bool = False) -> dict:
    bounded = {"error": muf_error, "censored_truth": censored,
               "censored_forecast": False}
    return {"event": "scored", "at": at, "lead_s": lead_s,
            "clock": {"suspect": suspect, "jump_us": 1520.0 if suspect else 40.0,
                      "offsets_us": [480.0, 2000.0 if suspect else 520.0],
                      "why": "the echo moved 1520 us between the soundings"
                             if suspect else ""},
            "pair": {"muf_mhz": bounded, "lof_mhz": bounded,
                     "modes": {"matched": len(delays), "missed": 0, "false": 0}},
            "modes": [{"status": "matched", "delay_rms_us": d, "cn0_rms_db": None,
                       "f_hi_error_mhz": None, "f_hi_censored_truth": False,
                       "f_hi_censored_forecast": False, "overlap_mhz": o}
                      for d, o in zip(delays, overlaps or [1.0] * len(delays))]}


def _soundings(folder: Path, epochs) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k, epoch in enumerate(epochs):
        state = _state(epoch, muf=13.6 + 0.07 * k, source="test")
        out.append(sounding.write_lfs(state, folder / sounding.default_name(state),
                                      seed=k))
    return out


def test_the_interval_is_the_median_of_the_last_gaps():
    assert live.cadence([T0, T0 + 600, T0 + 1200, T0 + 2400, T0 + 3000]) == 600.0


def test_one_sounding_gives_no_interval():
    assert live.cadence([T0]) is None


def test_nothing_is_issued_before_the_interval_is_known():
    assert [e for e in _turns([T0]) if e["event"] == "issued"] == [], \
        "a forecast was issued with no interval to lead by"


def test_forecasts_are_issued_one_and_two_intervals_ahead():
    issued = [e["lead_s"] for e in _turns([T0, T0 + 600]) if e["event"] == "issued"]
    assert issued == [600.0, 1200.0], "the leads are not the next two soundings"


def test_chosen_leads_replace_the_intervals():
    issued = [e["lead_s"] for e in _turns([T0], leads=[300.0])
              if e["event"] == "issued"]
    assert issued == [300.0], "the chosen leads were not the ones issued"


def test_a_forecast_is_scored_by_the_sounding_at_its_time():
    scored = [(e["from"], e["to"]) for e in _turns([T0, T0 + 600, T0 + 1200])
              if e["event"] == "scored"]
    assert scored == [("s1.lfs", "s2.lfs")], "the forecast was not scored at its time"


def test_a_forecast_with_no_sounding_at_its_time_lapses():
    lapsed = [e["reason"] for e in _turns([T0, T0 + 600, T0 + 1500])
              if e["event"] == "lapsed"]
    assert lapsed == ["no sounding within 30 s of 2026-02-04T12:20:10Z"], \
        "a forecast past its time did not lapse"


def test_a_sounding_with_no_trace_issues_nothing():
    assert [e for e in _turns([T0, T0 + 600], untraced={1}) if e["event"] == "issued"] \
        == [], "a sounding with no trace issued a forecast"


def test_a_forecast_due_at_a_sounding_with_no_trace_lapses_saying_so():
    lapsed = [e["reason"] for e in _turns([T0, T0 + 600, T0 + 1200], untraced={2})
              if e["event"] == "lapsed"]
    assert lapsed == ["no trace in the truth's sounding"], \
        "a forecast due at an untraced sounding did not lapse"


def test_a_ledger_comes_back_whole_from_its_dict():
    ledger, _ = live.step(live.Ledger(), series.Session(Path("s0.lfs"), T0, "cyprus1",
                                                        "yoshkar-ola"),
                          _state(T0), lambda pending: None, leads=[600.0])
    back = live.Ledger.from_dict(json.loads(json.dumps(ledger.to_dict())))
    assert back.to_dict() == ledger.to_dict(), "the ledger changed on its way back"


def test_the_running_figures_count_each_leads_pairs():
    figures = live.figures(_turns([T0, T0 + 600, T0 + 1200, T0 + 1800]))
    assert {lead: f["n"] for lead, f in figures.items()} == {600.0: 2, 1200.0: 1}


def test_the_muf_error_leaves_out_censored_pairs():
    rows = [_scored(600.0, 0.5), _scored(600.0, -1.5),
            _scored(600.0, 9.0, censored=True)]
    assert live.figures(rows)[600.0]["muf_mae_mhz"] == pytest.approx(1.0)


def test_the_delay_error_is_the_rms_over_the_modes():
    rows = [_scored(600.0, 0.0, delays=[3.0]), _scored(600.0, 0.0, delays=[4.0, None])]
    assert live.figures(rows)[600.0]["delay_rms_us"] == pytest.approx(math.sqrt(12.5))


def test_a_modes_delay_error_weighs_by_its_overlap():
    rows = [_scored(600.0, 0.0, delays=[3.0, 4.0], overlaps=[1.0, 3.0])]
    assert live.figures(rows)[600.0]["delay_rms_us"] == pytest.approx(math.sqrt(14.25))


def test_the_running_figures_start_at_since():
    rows = [_scored(600.0, 0.5, at="2026-02-04T12:00:10Z"),
            _scored(600.0, 1.5, at="2026-02-04T13:00:10Z")]
    assert live.figures(rows, since=T0 + 1800)[600.0]["n"] == 1


def test_a_scored_line_names_its_lead_in_minutes():
    line = live.format_event({**_scored(600.0, 0.5, delays=[3.0]), "from": "a.h5",
                              "to": "b.h5", "issued_at": "2026-02-04T12:00:10Z"})
    assert "+10 min" in line, f"the lead is not in minutes: {line}"


def test_a_pair_across_a_jump_of_the_clock_is_scored_as_clock_suspect():
    a = random.uniform(100.0, 700.0)
    scored = [e for e in _turns([T0, T0 + 300.0, T0 + 600.0], leads=[300.0],
                                offsets=[a, a, a + random.uniform(400.0, 1200.0)])
              if e["event"] == "scored"]
    assert scored[-1]["clock"]["suspect"] is True, \
        "a pair across a jump of the clock was not scored as clock-suspect"


def test_a_pair_on_a_steady_clock_is_not_clock_suspect():
    a = random.uniform(100.0, 700.0)
    scored = [e for e in _turns([T0, T0 + 300.0], leads=[300.0],
                                offsets=[a, a + random.uniform(-150.0, 150.0)])
              if e["event"] == "scored"]
    assert scored[0]["clock"]["suspect"] is False, \
        "a pair on a steady clock was scored as clock-suspect"


def test_an_issued_forecast_says_where_its_echo_begins():
    offset = random.uniform(-900.0, 2000.0)
    issued = [e for e in _turns([T0], leads=[300.0], offsets=[offset])
              if e["event"] == "issued"]
    assert issued[0]["echo_offset_us"] == pytest.approx(offset), \
        "an issued forecast did not say where its echo begins"


def test_a_sounding_with_no_trace_says_where_its_echo_begins():
    offset = random.uniform(-6000.0, -1000.0)
    untraced = [e for e in _turns([T0, T0 + 300.0], leads=[300.0], untraced={1},
                                  offsets=[400.0, offset])
                if e["event"] == "untraced"]
    assert f"{offset / 1e3:+.2f} ms" in live.format_event(untraced[0]), \
        "a sounding with no trace did not say where its echo begins"


def test_a_clock_suspect_line_says_so():
    line = live.format_event({**_scored(600.0, 9.0, delays=[1500.0], suspect=True),
                              "from": "a.h5", "to": "b.h5",
                              "issued_at": "2026-02-04T12:00:10Z"})
    assert "clock-suspect" in line, f"a clock-suspect line did not say so: {line}"


def test_the_running_figures_leave_clock_suspect_pairs_out():
    rows = [_scored(600.0, 0.5), _scored(600.0, 9.0, suspect=True)]
    assert live.figures(rows)[600.0]["n"] == 1, \
        "a clock-suspect pair was counted in the running figures"


def test_the_running_figures_count_the_pairs_set_apart():
    k = random.randint(1, 6)
    rows = [_scored(600.0, 9.0, suspect=True) for _ in range(k)] + [_scored(600.0, 0.5)]
    assert live.figures(rows)[600.0]["set_apart"] == k, \
        "the running figures did not count the pairs set apart"


def test_a_lead_with_only_clock_suspect_pairs_still_has_figures():
    assert live.figures([_scored(600.0, 9.0, suspect=True)])[600.0]["n"] == 0, \
        "a lead with only clock-suspect pairs lost its figures"


def test_one_pairs_own_figures_keep_it_when_asked():
    rows = [_scored(600.0, 9.0, delays=[1500.0], suspect=True)]
    assert live.figures(rows, suspect=True)[600.0]["n"] == 1, \
        "a clock-suspect pair's own figures left it out"


def test_the_figures_line_says_how_many_pairs_were_set_apart():
    rows = [_scored(600.0, 9.0, suspect=True), _scored(600.0, 0.5)]
    assert "1 set apart" in live.format_figures(live.figures(rows), "last hour"), \
        "the figures line did not say how many pairs were set apart"


def test_a_turn_on_an_unreadable_sounding_says_so(tmp_path):
    bad = tmp_path / "cyprus1_20260204_121010.lfs"
    bad.write_bytes(b"not a sounding")
    _, events = live.take(live.Ledger(), series.Session(bad, T0, "cyprus1",
                                                        "yoshkar-ola"), tmp_path / "c")
    assert [e["event"] for e in events] == ["unobserved"], \
        "an unreadable sounding was not reported as unobserved"


def test_a_replay_scores_each_forecast_whose_sounding_came(tmp_path):
    paths = _soundings(tmp_path / "in", [T0 + 300.0 * k for k in range(4)])
    live.replay(paths, out=tmp_path / "events.jsonl", emit=lambda line: None)
    rows = [json.loads(line) for line in
            (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(r["event"] == "scored" for r in rows) == 3, \
        "the replay did not score the three forecasts that met their soundings"


def test_a_replay_waits_out_the_gaps_at_its_pace(tmp_path):
    paths = _soundings(tmp_path / "in", [T0, T0 + 300.0, T0 + 600.0])
    waits = []
    live.replay(paths, pace=60.0, emit=lambda line: None, sleep=waits.append)
    assert waits == [5.0, 5.0], "the replay did not wait the gaps over its pace"
