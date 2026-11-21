"""Restored MUF samples: where the public page may show a value nobody measured.

A restored sample is a guess, so these tests pin both how good the guess is
and where it is allowed at all: only strictly inside a gap, only up to the
longest gap that was tested, and never mixed into the measured arrays.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.publish.restore import Restoration

SITE = Path(__file__).resolve().parents[1] / "services" / "publish" / "site"


def _day(rng, days=6, step_min=10, hole=None):
    """A measured series with a daily cycle and noise; `hole` is (start, hours)
    in hours from the first sample, left out of the series."""
    start = datetime(2026, 11, 1, 0, 5, 15) + timedelta(
        minutes=int(rng.integers(0, 60))
    )
    floor, swing, peak = rng.uniform(8, 12), rng.uniform(12, 20), rng.uniform(9, 12)
    t, muf, truth = [], [], []
    for k in range(days * 24 * 60 // step_min):
        when = start + timedelta(minutes=k * step_min)
        hour = k * step_min / 60
        value = (
            floor + swing * max(0.0, np.cos((hour % 24 - peak) / 24 * 2 * np.pi)) ** 1.5
        )
        if hole and hole[0] < hour < hole[0] + hole[1]:
            truth.append((when.strftime("%Y-%m-%dT%H:%M:%S"), value))
            continue
        t.append(when.strftime("%Y-%m-%dT%H:%M:%S"))
        muf.append(round(value + rng.normal(0, 0.3), 3))
    return t, muf, dict(truth)


def test_a_gap_under_an_hour_is_restored_on_the_straight_line_between_its_edges():
    rng = np.random.default_rng()
    t, muf, _ = _day(rng, days=3, hole=(30 + rng.uniform(0, 12), 0.75))
    block = Restoration(t, muf).block()
    edge = [datetime.fromisoformat(x).timestamp() for x in t]
    want = np.interp(
        [datetime.fromisoformat(x).timestamp() for x in block["t"]], edge, muf
    )
    assert np.allclose(block["muf"], want, atol=1e-3), (
        f"a short gap was not restored linearly: {block['muf']} against {want}"
    )


def test_a_gap_of_hours_is_restored_along_the_daily_cycle():
    rng = np.random.default_rng()
    t, muf, truth = _day(rng, hole=(72 + rng.uniform(0, 12), rng.uniform(6, 12)))
    block = Restoration(t, muf).block()
    error = np.mean(
        [abs(v - truth[x]) for x, v in zip(block["t"], block["muf"]) if x in truth]
    )
    assert error < 1.5, (
        f"a gap of hours was not restored along the daily cycle, error {error:.2f} MHz"
    )


def test_a_restored_sample_of_a_long_gap_carries_its_uncertainty():
    rng = np.random.default_rng()
    t, muf, _ = _day(rng, hole=(60 + rng.uniform(0, 12), rng.uniform(2, 10)))
    sigma = Restoration(t, muf).block()["sigma"]
    assert sigma and all(s is not None and s > 0 for s in sigma), (
        f"a sample restored in a long gap did not carry its uncertainty: {sigma}"
    )


def test_a_gap_longer_than_eighteen_hours_stays_empty():
    rng = np.random.default_rng()
    t, muf, _ = _day(rng, days=7, hole=(60 + rng.uniform(0, 12), rng.uniform(18.5, 30)))
    assert not Restoration(t, muf).block()["t"], (
        "a gap longer than eighteen hours was restored"
    )


def test_a_series_sounded_every_twenty_minutes_has_nothing_to_restore():
    rng = np.random.default_rng()
    t, muf, _ = _day(rng, days=2, step_min=20)
    assert not Restoration(t, muf).block()["t"], (
        "a series without gaps had samples restored"
    )


def test_restored_samples_lie_strictly_inside_the_gap():
    rng = np.random.default_rng()
    start = 40 + rng.uniform(0, 12)
    t, muf, truth = _day(rng, days=4, hole=(start, rng.uniform(0.5, 9)))
    restored = Restoration(t, muf).block()["t"]
    assert restored and all(
        x in truth or (min(truth) < x < max(truth)) for x in restored
    ), f"a sample was restored outside the gap: {restored}"


def test_a_missing_value_counts_as_a_gap():
    rng = np.random.default_rng()
    t, muf, _ = _day(rng, days=3)
    hole = int(rng.integers(200, 300))
    muf[hole : hole + 4] = [None] * 4
    assert set(t[hole : hole + 4]) <= set(Restoration(t, muf).block()["t"]), (
        "samples with no measured value were not restored"
    )


def test_the_page_draws_restored_samples_in_a_color_of_their_own():
    style = (SITE / "style.css").read_text(encoding="utf-8")
    assert ".chart .muf-restored" in style and "--restored" in style, (
        "the page has no color of its own for restored samples"
    )


def test_the_restored_samples_are_named_in_each_language():
    strings = json.loads((SITE / "strings.json").read_text(encoding="utf-8"))
    assert all(strings[lang].get("restored") for lang in ("en", "ru")), (
        "restored samples are not named in every language"
    )


def test_a_series_thinned_to_half_an_hour_has_nothing_to_restore():
    rng = np.random.default_rng()
    t, muf, _ = _day(rng, days=3, step_min=30)
    assert not Restoration(t, muf).block()["t"], (
        "a thinned series without gaps had samples restored"
    )


def test_a_gap_whose_estimate_overflows_is_left_empty():
    t = ["2026-11-01T00:00:00", "2026-11-01T02:00:00"]
    assert not Restoration(t, [1e155, -1e155]).block()["t"], (
        "an estimate that overflowed was published"
    )


def test_no_restored_sample_is_below_zero():
    rng = np.random.default_rng()
    low = rng.uniform(2, 4)
    t = [
        "2026-11-01T00:00:00",
        "2026-11-01T00:10:00",
        "2026-11-01T02:00:00",
        "2026-11-01T02:10:00",
    ]
    muf = Restoration(t, [low, low, low, low + rng.uniform(14, 18)]).block()["muf"]
    assert all(v > 0 for v in muf), f"a restored sample fell to or below zero: {muf}"
