"""muf.sim.bench: a frozen benchmark of the simulator against real soundings."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from muf import echo_window
from muf.sim import bench, observe, presets, sounding
from muf.sim.state import ChannelState, Sweep


def _sweep() -> Sweep:
    return Sweep(start_mhz=10.0, rate_hz_s=100e3, duration_s=40.0, sample_rate_hz=40e3)


def _state(epoch: float, *, cn0: float = 45.0, modes: int = 1) -> ChannelState:
    found = [presets.law_mode(muf_mhz=13.6, f_lo_mhz=10.5, cn0_dbhz=cn0,
                              spread_hz=0.5)]
    if modes > 1:
        found.append(presets.law_mode("2F", muf_mhz=13.0, tau0_us=9800.0,
                                      f_lo_mhz=11.0, cn0_dbhz=cn0, spread_hz=0.5))
    return ChannelState(presets.cyprus1_circuit(), epoch, found, sweep=_sweep(),
                        source="test")


def _lfs(folder: Path, epoch: float, **kw) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    state = _state(epoch, **kw)
    return sounding.write_lfs(state, folder / sounding.default_name(state), seed=3)


def _epoch(day: int, hour: int) -> float:
    return 1770163200.0 + 86400.0 * day + 3600.0 * hour + 10.0


def _sounded(tmp_path: Path, **kw):
    ion = observe.load(_lfs(tmp_path, _epoch(0, 9), **kw))
    return ion, observe.from_ionogram(ion, source="real")


def _scaled(ion, gain):
    return dataclasses.replace(ion, power=np.asarray(ion.power) * gain, _db=None)


def _blanked(ion, state):
    power = np.array(ion.power)
    mode = state.modes[0]
    rows = np.flatnonzero(np.asarray(ion.freq) < 0.5 * (mode.f_lo + mode.f_hi))
    power[rows] = observe.NOISE_MEAN_POWER
    return dataclasses.replace(ion, power=power, _db=None)


def test_an_lfs_is_listed_as_carrying_all_its_samples(tmp_path):
    path = _lfs(tmp_path, _epoch(0, 9))
    assert bench.entry(path, tmp_path)["phase"] == "lfs", \
        "an .lfs was not listed as phase"


def test_a_product_with_an_echo_window_is_listed_as_carrying_it(tmp_path):
    report = echo_window.write(_lfs(tmp_path / "raw", _epoch(0, 9)), out_dir=tmp_path)
    assert bench.entry(report.target, tmp_path)["phase"] == "echo window", \
        "an echo window was not listed as its phase"


def test_a_product_without_voltage_is_listed_as_power_only(tmp_path, make_chirp_h5):
    power = np.random.default_rng(random.randrange(99)).exponential(size=(16, 512))
    path = make_chirp_h5(power, name="lfm_ionogram-synthtx-synthrx-ch000-007-1.00.h5")
    assert bench.entry(path, tmp_path)["phase"] == "power", \
        "a power-only product was listed as carrying phase"


def test_an_entry_carries_the_digest_of_its_bytes(tmp_path):
    path = _lfs(tmp_path, _epoch(0, 9))
    assert bench.entry(path, tmp_path)["sha256"] \
        == hashlib.sha256(path.read_bytes()).hexdigest(), \
        "the digest is not of the file"


def test_an_entry_names_its_file_against_the_root_with_forward_slashes(tmp_path):
    path = _lfs(tmp_path / "2026.10.17", _epoch(0, 9))
    assert bench.entry(path, tmp_path)["path"] == f"2026.10.17/{path.name}", \
        "the entry does not name its file against the root"


def test_an_unreadable_product_is_listed_with_its_reason(tmp_path):
    path = tmp_path / "lfm_ionogram-NIC2-yoshkar-ola-ch0-000-1792216515.00.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(random.randrange(64, 512)))
    assert "reason" in bench.entry(path, tmp_path), "a broken product gave no reason"


def test_a_day_named_in_two_splits_is_refused():
    with pytest.raises(ValueError, match="2026-10-17"):
        bench.splits_of({"tune": ["2026-10-17"], "validation": ["2026-10-17"]})


def test_a_split_the_benchmark_does_not_know_is_refused():
    with pytest.raises(ValueError, match="practice"):
        bench.splits_of({"practice": ["2026-10-17"]})


def test_a_sounding_takes_the_split_of_its_utc_day(tmp_path):
    path = _lfs(tmp_path, _epoch(1, 23))
    day = bench.entry(path, tmp_path)["day"]
    assert bench.assign([bench.entry(path, tmp_path)],
                        bench.splits_of({"validation": [day]}))[0]["split"] \
        == "validation", "a sounding did not take its day's split"


def test_a_sounding_of_a_day_in_no_split_is_left_unused(tmp_path):
    path = _lfs(tmp_path, _epoch(1, 23))
    assert bench.assign([bench.entry(path, tmp_path)],
                        bench.splits_of({"tune": ["1999-01-01"]}))[0]["split"] \
        is None, "a sounding outside every split was given one"


def test_a_frozen_manifest_lists_every_input_once(tmp_path):
    paths = [_lfs(tmp_path / "in", _epoch(0, h)) for h in (9, 10)]
    manifest = bench.freeze([tmp_path / "in", *paths], tmp_path, tmp_path / "out",
                            splits={"tune": ["2026-02-04"]})
    assert sorted(e["file"] for e in manifest["entries"]) \
        == sorted(p.name for p in paths), "the manifest does not list each input once"


def test_a_diagnostic_sounding_outside_the_manifest_is_refused(tmp_path):
    _lfs(tmp_path / "in", _epoch(0, 9))
    with pytest.raises(ValueError, match="not in the manifest"):
        bench.freeze([tmp_path / "in"], tmp_path, tmp_path / "out", splits={},
                     diagnostic=["elsewhere.lfs"])


def test_an_input_changed_after_freezing_is_refused(tmp_path):
    path = _lfs(tmp_path / "in", _epoch(0, 9))
    manifest = bench.freeze([path], tmp_path, tmp_path / "out", splits={})
    with path.open("r+b") as handle:
        handle.seek(1024)
        handle.write(bytes([random.randrange(1, 255)]) * 8)
    with pytest.raises(ValueError, match="changed"):
        bench.check_inputs(manifest, tmp_path)


def test_a_copy_twice_as_strong_reads_three_db_high_at_the_ridge(tmp_path):
    ion, state = _sounded(tmp_path)
    found = bench.levels(state, ion, [_scaled(ion, 2.0)])
    assert found[state.modes[0].name]["peak_bias_db"] \
        == pytest.approx(3.0103, abs=0.01), \
        "a copy twice as strong did not read 3 dB high"


def test_a_copy_like_the_sounding_holds_its_energy(tmp_path):
    ion, state = _sounded(tmp_path)
    found = bench.levels(state, ion, [_scaled(ion, 1.0)])
    assert found[state.modes[0].name]["energy_bias_db"] \
        == pytest.approx(0.0, abs=1e-9), \
        "a copy like the sounding did not hold its energy"


def test_a_copy_like_the_sounding_misses_no_echo(tmp_path):
    ion, state = _sounded(tmp_path)
    assert bench.coverage(state, ion, [_scaled(ion, 1.0)])["missed"] == 0.0, \
        "a copy like the sounding missed an echo"


def test_an_echo_the_copy_lacks_is_counted_missed(tmp_path):
    ion, state = _sounded(tmp_path)
    assert bench.coverage(state, ion, [_blanked(ion, state)])["missed"] > 0.2, \
        "an echo the copy lacks was not counted missed"


def test_an_echo_the_sounding_lacks_is_counted_false(tmp_path):
    ion, state = _sounded(tmp_path)
    assert bench.coverage(state, _blanked(ion, state), [ion])["false"] > 0.2, \
        "an echo the sounding lacks was not counted false"


def test_an_echo_off_every_mode_of_the_state_is_counted_untraced(tmp_path):
    ion, state = _sounded(tmp_path, modes=2)
    alone = dataclasses.replace(
        state, modes=[min(state.modes, key=lambda m: m.delay_us[0])])
    assert bench.coverage(alone, ion, [ion])["untraced"] > 0.2, \
        "an echo off every mode was not counted untraced"


def test_a_diagnosis_scores_every_mode_of_the_sounding(tmp_path):
    path = _lfs(tmp_path, _epoch(0, 9))
    row = bench.diagnose(path, {**bench.PARAMS, "copies": 2})
    assert sorted(row["reconstruction"]["levels"]) \
        == sorted(m.name for m in observe.from_sounding(path).modes), \
        "the diagnosis did not score every mode"


def test_a_sounding_with_no_trace_cannot_be_diagnosed(tmp_path):
    path = _lfs(tmp_path, _epoch(0, 9), cn0=-20.0)
    with pytest.raises(ValueError, match="no trace"):
        bench.diagnose(path, {**bench.PARAMS, "copies": 2})


def _frozen(tmp_path: Path) -> Path:
    paths = [_lfs(tmp_path / "in", _epoch(0, 9 + h / 6)) for h in range(3)]
    bench.freeze(paths, tmp_path, tmp_path / "bench", splits={"tune": ["2026-02-04"]},
                 diagnostic=[paths[0].name],
                 params={"copies": 2, "forecast_copies": 0, "leads_s": [600.0]})
    return tmp_path / "bench" / "manifest.json"


def test_two_runs_of_one_manifest_give_one_digest(tmp_path):
    manifest = _frozen(tmp_path)
    first = bench.run(manifest, tmp_path / "a", root=tmp_path)
    assert bench.run(manifest, tmp_path / "b", root=tmp_path)["digest"] \
        == first["digest"], "two runs of one manifest gave two results"


def test_a_run_draws_each_diagnostic_sounding(tmp_path):
    manifest = _frozen(tmp_path)
    bench.run(manifest, tmp_path / "run", root=tmp_path, forecast=False)
    assert len(list((tmp_path / "run" / "gallery").glob("*.png"))) == 1, \
        "a diagnostic sounding was not drawn"


def test_a_run_leaves_the_final_split_alone_unless_asked(tmp_path):
    paths = [_lfs(tmp_path / "in", _epoch(0, 9 + h / 6)) for h in range(2)]
    bench.freeze(paths, tmp_path, tmp_path / "bench", splits={"final": ["2026-02-04"]},
                 params={"copies": 2, "forecast_copies": 0, "leads_s": [600.0]})
    bench.run(tmp_path / "bench" / "manifest.json", tmp_path / "run", root=tmp_path,
              diagnostic=False)
    assert not (tmp_path / "run" / "forecast" / "final").exists(), \
        "the run scored the held-out split unasked"


def test_a_run_scores_each_split_and_lead_apart(tmp_path):
    manifest = _frozen(tmp_path)
    bench.run(manifest, tmp_path / "run", root=tmp_path, diagnostic=False)
    assert (tmp_path / "run" / "forecast" / "tune" / "600s"
            / "pairs.jsonl").is_file(), \
        "the split's forecast was not scored at its lead"


def test_a_run_records_the_manifest_it_ran(tmp_path):
    manifest = _frozen(tmp_path)
    record = bench.run(manifest, tmp_path / "run", root=tmp_path, forecast=False)
    assert record["manifest_sha256"] \
        == hashlib.sha256(manifest.read_bytes()).hexdigest(), \
        "the run does not name the manifest it ran"


def test_the_run_record_is_written_beside_its_rows(tmp_path):
    manifest = _frozen(tmp_path)
    record = bench.run(manifest, tmp_path / "run", root=tmp_path, forecast=False)
    assert json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))[
        "digest"] == record["digest"], "the run record on disk is not the one returned"


def test_the_inventory_counts_each_days_soundings_by_their_phase():
    rows = [{"file": f"{k}.h5", "day": "2026-10-17", "phase": "echo window"}
            for k in range(random.randrange(2, 9))]
    assert f"2026-10-17  {len(rows)} soundings: {len(rows)} echo window" \
        in bench.format_inventory(rows), \
        "the inventory did not count the day's soundings"


def test_the_inventory_counts_the_set_aside_apart():
    rows = [{"file": "a.h5", "reason": "OSError: truncated file"},
            {"file": "b.h5", "day": "2026-10-17", "phase": "power"}]
    assert "1 set aside" in bench.format_inventory(rows), \
        "the inventory did not count the set aside apart"


def _row(r: float, ceiling: float) -> dict:
    return {"file": "x.h5", "epoch": 1792831515.0, "modes": ["1F"],
            "reconstruction": {
                "together": {"peak_bias_db": 3.21, "energy_bias_db": 1.04},
                "coverage": {"blocks": 40, "missed": 0.1, "false": 0.2,
                             "untraced": 0.3},
                "correlation": {"rows": {"real": {"mean": r},
                                         "synthetic": {"mean": ceiling}}}}}


def test_a_diagnosis_line_gives_the_ridge_bias_of_all_modes_together():
    assert "peak +3.2 dB, energy +1.0 dB" in bench.format_diagnosis(_row(0.7, 0.9)), \
        "the line does not give the ridge bias"


def test_a_diagnosis_line_sets_r_beside_the_copies_own():
    r, ceiling = random.uniform(0.3, 0.6), random.uniform(0.7, 0.95)
    assert f"r {r:.2f} of {ceiling:.2f}" in bench.format_diagnosis(_row(r, ceiling)), \
        "the line does not set r beside the copies' own"


def test_fragments_of_one_trace_together_hold_their_energy(tmp_path):
    ion, state = _sounded(tmp_path)
    twin = dataclasses.replace(state.modes[0], name="1F#2")
    split = dataclasses.replace(state, modes=[state.modes[0], twin])
    assert bench.together(split, ion, [ion])["energy_bias_db"] \
        == pytest.approx(0.0, abs=1e-9), "fragments of one trace lost their energy"


def test_all_modes_together_read_a_copy_twice_as_strong_three_db_high(tmp_path):
    ion, state = _sounded(tmp_path, modes=2)
    assert bench.together(state, ion, [_scaled(ion, 2.0)])["peak_bias_db"] \
        == pytest.approx(3.0103, abs=0.01), \
        "a copy twice as strong did not read 3 dB high"


def test_two_copies_of_one_file_are_benchmarked_once(tmp_path):
    path = _lfs(tmp_path / "a", _epoch(0, 9))
    twin = tmp_path / "b" / path.name
    twin.parent.mkdir()
    twin.write_bytes(path.read_bytes())
    assert len(bench.freeze([path, twin], tmp_path, tmp_path / "out",
                            splits={})["entries"]) == 1, \
        "two copies of one file were benchmarked twice"


def test_the_leads_follow_the_circuits_usual_gap_not_its_last_few(tmp_path):
    epochs = [_epoch(0, 9) + 600.0 * k for k in range(5)]
    epochs += [epochs[-1] + 10800.0 * k for k in range(1, 4)]
    paths = [_lfs(tmp_path / "in", t) for t in epochs]
    assert bench.freeze(paths, tmp_path, tmp_path / "out", splits={})["params"][
        "leads_s"] == [600.0, 1200.0], "the leads did not follow the usual gap"


def test_a_sweep_cut_short_is_set_aside_with_its_reason(tmp_path):
    path = _lfs(tmp_path, _epoch(0, 9))
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size // random.randrange(3, 9))
    assert "of the sweep" in bench.entry(path, tmp_path).get("reason", ""), \
        "a sweep cut short was not set aside"


def test_each_split_and_lead_is_summed_up_with_its_intervals(tmp_path):
    manifest = _frozen(tmp_path)
    bench.run(manifest, tmp_path / "run", root=tmp_path, diagnostic=False)
    assert (tmp_path / "run" / "forecast" / "tune" / "600s"
            / "summary.json").is_file(), "the split's forecast was not summed up"


def _twins(tmp_path: Path) -> tuple[Path, Path, dict]:
    lfs = _lfs(tmp_path / "in", _epoch(0, 9))
    product = echo_window.write(lfs, out_dir=tmp_path / "in").target
    manifest = bench.freeze([lfs, product], tmp_path, tmp_path / "out", splits={})
    return lfs, product, manifest


def test_a_sounding_kept_twice_is_benchmarked_once_from_its_product(tmp_path):
    _, product, manifest = _twins(tmp_path)
    assert [e["file"] for e in manifest["entries"]] == [product.name], \
        "a sounding kept twice was not benchmarked once from its product"


def test_the_twin_of_a_benchmarked_sounding_is_set_aside_with_its_reason(tmp_path):
    lfs, product, manifest = _twins(tmp_path)
    assert {e["file"]: e["reason"] for e in manifest["set_aside"]}[lfs.name] \
        == f"the same sounding as {product.name}", "the twin was not set aside"
