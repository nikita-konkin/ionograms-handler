"""The benchmark's verdicts: what it raises an alarm about, and what it does not.

Only the judgement is tested here, not the timing -- a test that measured
anything would be asserting how fast the machine running it happens to be. The
judgement is the part that has to keep working, because a benchmark that can
only ever say "fine" is worse than none at all.

Loaded by path: `tools/` is deliberately not one of the shipped packages.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "benchmark", REPO / "tools" / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load()


def run(**over):
    """A healthy run on a ten-core box, which individual tests then spoil."""
    base = {
        "host": {"node": "station", "cpus": 10, "platform": "x", "python": "3"},
        "stages": {"read_ms": 7.0, "spectrogram_ms": 172.0, "total_ms": 265.0,
                   "io_share": 0.03, "methods_ms": {}},
        "extraction": {
            "n_files": 40, "jobs": 8, "pinned": True,
            "serial": {"workers": 1, "seconds": 10.8, "files_per_s": 3.7,
                       "ms_per_file": 271.0, "errors": 0},
            "parallel": {"workers": 8, "seconds": 3.3, "files_per_s": 12.1,
                         "ms_per_file": 82.0, "errors": 0},
            "speedup": 3.27, "efficiency": 0.41, "error_rate": 0.0,
            "rss_mb": 310.0, "rss_growth": 1.02,
        },
        "pages": [{"page": "console", "path": "/ui", "status": 200,
                   "cold_ms": 21.0, "warm_ms": 8.0, "p99_ms": 21.0, "kb": 5.7}],
    }
    base.update(over)
    return base


def levels(found):
    return {what: level for level, what, _ in found}


# --- a healthy box is quiet --------------------------------------------------

def test_a_healthy_run_says_nothing():
    assert benchmark.verdicts(run(), None) == []


# --- the ratios that travel between machines ---------------------------------

def test_workers_fighting_for_threads_are_called_out():
    spoiled = run()
    spoiled["extraction"]["speedup"] = 2.16
    spoiled["extraction"]["pinned"] = False

    found = benchmark.verdicts(spoiled, None)

    assert levels(found)["parallel speed-up"] == "BAD"
    assert "MUF_PIN_THREADS" in found[0][2]


def test_a_sample_too_small_to_judge_scaling_is_not_called_a_fault():
    """Eight files on eight workers reports ~1.2x on a perfectly good box:
    the pool costs more to start than the work costs to do."""
    spoiled = run()
    spoiled["extraction"]["serial"]["seconds"] = 2.1
    spoiled["extraction"]["speedup"] = 1.24

    found = benchmark.verdicts(spoiled, None)

    assert levels(found)["sample too small to judge scaling"] == "INFO"
    assert "parallel speed-up" not in levels(found)


def test_a_slow_archive_mount_is_visible_before_a_page_is():
    spoiled = run()
    spoiled["stages"]["io_share"] = 0.40

    assert levels(benchmark.verdicts(spoiled, None))["archive reads"] == "WARN"


def test_a_worker_that_keeps_growing_is_reported():
    spoiled = run()
    spoiled["extraction"]["rss_growth"] = 1.6

    assert levels(benchmark.verdicts(spoiled, None))["memory growth"] == "WARN"


def test_soundings_that_raised_are_not_reported_as_a_speed_problem():
    spoiled = run()
    spoiled["extraction"]["error_rate"] = 0.1

    found = benchmark.verdicts(spoiled, None)

    assert levels(found)["soundings that failed"] == "BAD"
    assert "correctness" in found[0][2]


# --- this box against itself -------------------------------------------------

def test_extraction_getting_slower_is_measured_against_the_baseline():
    was = run()
    was["extraction"]["parallel"]["ms_per_file"] = 50.0

    found = benchmark.verdicts(run(), was)

    assert levels(found)["extraction slower"] == "BAD"


def test_noise_between_two_runs_is_not_a_regression():
    was = run()
    was["extraction"]["parallel"]["ms_per_file"] = 78.0   # 5 % apart

    assert benchmark.verdicts(run(), was) == []


def test_picks_that_moved_are_the_loudest_thing_it_can_say():
    was = run(picks=[{"datetime": "t", "muf_algo": 12.0}])
    now = run(picks=[{"datetime": "t", "muf_algo": 12.4}])

    found = benchmark.verdicts(now, was)

    assert levels(found)["the picks moved"] == "BAD"


def test_identical_picks_pass_silently():
    picks = [{"datetime": "t", "muf_algo": 12.0}]

    assert benchmark.verdicts(run(picks=picks), run(picks=picks)) == []


def test_a_baseline_from_another_machine_is_refused_rather_than_compared():
    """Timings do not travel; saying so is more useful than a false alarm."""
    was = run()
    was["host"] = dict(was["host"], node="the-dev-mac")
    was["extraction"]["parallel"]["ms_per_file"] = 50.0

    found = benchmark.verdicts(run(), was)

    assert levels(found) == {"different machine": "WARN"}


# --- reporting ---------------------------------------------------------------

def test_a_page_that_failed_is_reported_whatever_its_timing():
    spoiled = run()
    spoiled["pages"][0]["status"] = 500

    assert levels(benchmark.verdicts(spoiled, None))["page console"] == "BAD"


@pytest.mark.parametrize("value,expected", [
    (float("nan"), None),
    (12.5, 12.5),
    ("2026-02-04", "2026-02-04"),
])
def test_picks_are_written_as_json_a_later_run_can_compare(value, expected):
    assert benchmark._plain(value) == expected
