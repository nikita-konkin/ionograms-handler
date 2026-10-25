"""muf.sim.series: a forecast scored over a run of soundings."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from muf.sim import observe, presets, series, sounding, summary
from muf.sim.state import ChannelState, Sweep

#: A short sweep, 10-14 MHz in 40 s, so a sounding is 1.6 M samples.
SHORT = Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)

T0 = 1770206410.0


def _session(name: str, epoch: float, tx="cyprus1", rx="yoshkar-ola"):
    return series.Session(Path(name), epoch, tx, rx)


def _soundings(folder: Path, epochs) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k, epoch in enumerate(epochs):
        law = presets.law_mode(muf_mhz=13.6 + 0.07 * k, f_lo_mhz=10.5, cn0_dbhz=45.0,
                               spread_hz=0.5)
        state = ChannelState(presets.cyprus1_circuit(), epoch, [law], sweep=SHORT,
                             source="test")
        out.append(sounding.write_lfs(state, folder / sounding.default_name(state),
                                      seed=k))
    return out


def _rows(run: Path, name: str) -> list[dict]:
    path = run / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _timeless(rows: list[dict]) -> list[str]:
    return sorted(json.dumps({k: v for k, v in r.items() if k != "seconds"},
                             sort_keys=True) for r in rows)


def _run(paths, out: Path, **kw) -> Path:
    options = {"copies": 0, "recovery_every": 0, "jobs": 1}
    options.update(kw)
    return series.run_series(paths, out, **options)


# --- pairs (T-12) -------------------------------------------------------------

def test_a_missing_sounding_makes_no_pair_twice_the_lead_apart():
    found, _ = series.pairs([_session("a", T0), _session("b", T0 + 300.0),
                             _session("d", T0 + 900.0)], lead_s=300.0, tolerance_s=30.0)
    assert [(a.path.name, b.path.name) for a, b in found] == [("a", "b")]


def test_a_sounding_with_none_a_lead_later_is_skipped_with_its_reason():
    _, skipped = series.pairs([_session("a", T0), _session("d", T0 + 900.0)],
                              lead_s=300.0, tolerance_s=30.0)
    assert skipped[0]["reason"] == "no sounding 300 +- 30 s later"


def test_a_ten_minute_step_pairs_at_a_ten_minute_lead():
    found, _ = series.pairs([_session("a", T0), _session("b", T0 + 600.0),
                             _session("c", T0 + 1200.0)], lead_s=600.0,
                            tolerance_s=30.0)
    assert len(found) == 2


def test_the_pair_is_the_first_sounding_within_the_tolerance():
    found, _ = series.pairs([_session("a", T0), _session("b", T0 + 290.0),
                             _session("c", T0 + 310.0)], lead_s=300.0, tolerance_s=30.0)
    assert found[0][1].path.name == "b"


def test_two_circuits_need_one_chosen():
    with pytest.raises(ValueError, match="circuit"):
        series.circuit_of([_session("a", T0), _session("b", T0, rx="dobrich")], None)


def test_the_chosen_circuit_is_kept():
    kept = series.circuit_of([_session("a", T0), _session("b", T0, rx="dobrich")],
                             "cyprus1:dobrich")
    assert [s.path.name for s in kept] == ["b"]


# --- a run --------------------------------------------------------------------

def test_a_run_scores_every_pair(tmp_path):
    run = _run(_soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(3)]),
               tmp_path / "run")
    assert len(_rows(run, "pairs")) == 2


def test_an_unreadable_sounding_is_skipped(tmp_path):
    _soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(3)])
    bad = tmp_path / "data" / "cyprus1_20260204_120000.lfs"
    bad.write_bytes(b"not a sounding" * 100)
    run = _run([tmp_path / "data"], tmp_path / "run")
    assert [r["from"] for r in _rows(run, "skipped")
            if r["stage"] == "header"] == [bad.name]


def test_an_unreadable_sounding_leaves_the_others_scored(tmp_path):
    _soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(3)])
    (tmp_path / "data" / "cyprus1_20260204_120000.lfs").write_bytes(b"x" * 1400)
    run = _run([tmp_path / "data"], tmp_path / "run")
    assert len(_rows(run, "pairs")) == 2


def test_a_run_refuses_an_existing_folder_without_resume(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    _run(paths, tmp_path / "run")
    with pytest.raises(ValueError, match="resume"):
        _run(paths, tmp_path / "run")


def test_a_resume_refuses_other_parameters(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    _run(paths, tmp_path / "run")
    with pytest.raises(ValueError, match="parameters"):
        _run(paths, tmp_path / "run", resume=True, lead_s=600.0)


def test_the_run_records_its_parameters_and_files(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    run = json.loads((_run(paths, tmp_path / "run") / "run.json").read_text("utf-8"))
    assert (run["params"]["lead_s"], len(run["files"])) == (300.0, 2)


def test_each_pair_knows_where_the_sun_was(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    pair = _rows(_run(paths, tmp_path / "run"), "pairs")[0]
    assert 0.0 <= pair["zenith_deg"] <= 180.0


def test_a_pair_on_a_steady_clock_is_not_set_apart(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    pair = _rows(_run(paths, tmp_path / "run"), "pairs")[0]
    assert pair["clock"]["suspect"] is False, "a pair on a steady clock was set apart"


# --- resuming (T-14) ----------------------------------------------------------

def test_a_resume_recomputes_nothing(tmp_path):
    paths = _soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(3)])
    run = _run(paths, tmp_path / "run")
    before = (run / "pairs.jsonl").read_bytes()
    _run(paths, run, resume=True)
    assert (run / "pairs.jsonl").read_bytes() == before


def test_an_interrupted_run_resumes_to_the_whole(tmp_path):
    paths = _soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(4)])
    whole = _run(paths, tmp_path / "whole")
    cut = _run(paths, tmp_path / "cut")
    lines = (cut / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    (cut / "pairs.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    _run(paths, cut, resume=True)
    assert all(_timeless(_rows(cut, name)) == _timeless(_rows(whole, name))
               for name in ("pairs", "modes", "carriers"))


# --- states, jobs, copies -----------------------------------------------------

def test_a_cached_state_is_read_not_observed(tmp_path, monkeypatch):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    run = _run(paths, tmp_path / "run")
    (run / "pairs.jsonl").write_text("", encoding="utf-8")

    def refuse(*args, **kw):
        raise AssertionError("observed again")

    monkeypatch.setattr(observe, "from_ionogram", refuse)
    _run(paths, run, resume=True)
    assert len(_rows(run, "pairs")) == 1


def test_a_truth_on_the_forecasts_grid_is_read_not_observed(tmp_path, monkeypatch):
    a, b = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    for path in (a, b):
        series.state_of(path, tmp_path / "states")

    def refuse(*args, **kw):
        raise AssertionError("observed again")

    monkeypatch.setattr(observe, "from_ionogram", refuse)
    assert series.truth_of(a, b, tmp_path / "states").source == b.name


def test_a_state_is_observed_again_when_its_parameters_change(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    _run(paths, tmp_path / "a")
    cache = tmp_path / "a" / "states"
    first = series.state_of(paths[0], cache, {"knot_mhz": 0.2})
    assert len(first.modes[0].freq_mhz) < 30


def test_two_jobs_score_as_one(tmp_path):
    paths = _soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(3)])
    one = _run(paths, tmp_path / "one")
    two = _run(paths, tmp_path / "two", jobs=2)
    assert _timeless(_rows(two, "modes")) == _timeless(_rows(one, "modes"))


def test_the_ionograms_are_compared_with_copies(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    run = _run(paths, tmp_path / "run", copies=2)
    assert _rows(run, "ionogram")[0]["copies"] == 2


def test_the_recovery_reads_the_shared_copies(tmp_path):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    run = _run(paths, tmp_path / "run", copies=2, recovery_every=1, recovery_copies=2)
    assert {r["seed"] for r in _rows(run, "recovery")} == {0, 1}


def test_the_recovery_sounds_no_copy_twice(tmp_path, monkeypatch):
    paths = _soundings(tmp_path / "data", [T0, T0 + 300.0])
    written = []
    real = sounding.write_lfs

    def counted(state, path, **kw):
        written.append(Path(path).name)
        return real(state, path, **kw)

    monkeypatch.setattr(sounding, "write_lfs", counted)
    _run(paths, tmp_path / "run", copies=2, recovery_every=1, recovery_copies=2)
    assert len(written) == 2


def test_the_recovery_runs_on_every_kth_sounding(tmp_path):
    paths = _soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(4)])
    run = _run(paths, tmp_path / "run", recovery_every=2, recovery_copies=1)
    assert sorted({r["from"] for r in _rows(run, "recovery")}) == [paths[0].name,
                                                                   paths[2].name]


def test_an_interrupted_run_sums_up_as_the_whole(tmp_path):
    paths = _soundings(tmp_path / "data", [T0 + 300.0 * k for k in range(4)])
    whole = _run(paths, tmp_path / "whole")
    cut = _run(paths, tmp_path / "cut")
    lines = (cut / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    (cut / "pairs.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    _run(paths, cut, resume=True)
    assert json.dumps(summary.summarise(cut, reps=50)) \
        == json.dumps(summary.summarise(whole, reps=50))


def test_a_failed_run_leaves_the_pairs_not_begun():
    start = time.monotonic()
    with pytest.raises(RuntimeError):
        for _ in series._map(time.sleep, [0.1] * 200, 2):
            raise RuntimeError("the writer failed")
    assert time.monotonic() - start < 5.0, "a failed run went on scoring every pair"
