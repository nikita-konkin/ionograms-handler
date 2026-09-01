"""The flat-feature survey.

What this tool has to get right is the distinction between *a trace that is
flat here* and *a line that is flat everywhere*. A real oblique trace is nearly
flat along the low-ray leg and only rises at the nose, so flatness alone
convicts nothing; it is flatness sustained over megahertz that cannot be
propagation. These tests are written against that distinction rather than
against the constants, which are expected to move.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "flat_tails", ROOT / "tools" / "flat_tails.py")
flat_tails = importlib.util.module_from_spec(spec)
sys.modules["flat_tails"] = flat_tails
spec.loader.exec_module(flat_tails)

FREQ = np.arange(8.0, 28.0, 0.05)          # the archive's axis: 50 kHz bins
VRANGE = np.arange(3000.0, 2000.0, -2.0)   # descending, 2 km bins, as stored
FLOOR, ECHO = 30.0, 60.0


class _Ion:
    """The three attributes the survey reads."""

    def __init__(self, db):
        self.db, self.freq, self.vrange = db, FREQ, VRANGE


def _blank():
    return np.full((FREQ.size, VRANGE.size), FLOOR)


def _draw(db, f_lo, f_hi, r_start, r_end):
    """Light one cell per frequency bin, range ramping linearly."""
    sel = np.flatnonzero((FREQ >= f_lo) & (FREQ <= f_hi))
    for r, f in zip(np.linspace(r_start, r_end, sel.size), sel):
        db[f, int(np.argmin(np.abs(VRANGE - r)))] = ECHO
    return db


def test_the_range_profile_takes_the_brightest_cell_not_the_first():
    """Two lit cells at one frequency: the survey must follow the stronger."""
    db = _blank()
    db[20, 100] = 50.0                      # dimmer, nearer the top of the axis
    db[20, 300] = 58.0                      # brighter
    has, rng = flat_tails.range_profile(_Ion(db))

    assert has[20]
    assert rng[20] == VRANGE[300]
    assert not has[21]


def test_a_sustained_flat_line_is_separated_from_a_curved_trace():
    """The `020` picture: a real trace, and an artefact above it in frequency.

    Both are found as segments; only one is flat, and the flat one reaches
    higher in frequency, which is exactly how it captures a MUF.
    """
    db = _blank()
    _draw(db, 10.0, 13.0, 2760.0, 2800.0)   # curved: 40 km over 3 MHz
    _draw(db, 14.0, 22.0, 2642.0, 2642.0)   # flat: 0 km over 8 MHz
    segs = flat_tails.segments(_Ion(db))

    flat = [s for s in segs if s["span"] >= flat_tails.MIN_FLAT_MHZ
            and s["spread"] / s["span"] < flat_tails.FLAT_KM_PER_MHZ]
    curved = [s for s in segs
              if s["spread"] / s["span"] >= flat_tails.FLAT_KM_PER_MHZ]

    assert len(flat) == 1 and len(curved) == 1
    assert flat[0]["r_med"] == pytest.approx(2642.0, abs=2.0)
    assert flat[0]["f_hi"] > curved[0]["f_hi"], "the artefact must reach higher"
    assert curved[0]["spread"] == pytest.approx(40.0, abs=4.0)


def test_a_short_flat_stretch_of_a_real_trace_is_not_convicted():
    """The low-ray leg is flat away from the nose, and that is propagation.

    Only sustained flatness is diagnostic, so a trace that runs flat for half a
    megahertz and then climbs must not be counted -- otherwise the survey
    reports the ionosphere as an artefact.
    """
    db = _blank()
    _draw(db, 10.0, 10.5, 2700.0, 2700.0)   # flat, but only 0.5 MHz
    _draw(db, 10.55, 12.0, 2702.0, 2860.0)  # then the nose
    segs = flat_tails.segments(_Ion(db))

    convicted = [s for s in segs if s["span"] >= flat_tails.MIN_FLAT_MHZ
                 and s["spread"] / s["span"] < flat_tails.FLAT_KM_PER_MHZ]
    assert not convicted


def test_features_separated_in_range_are_not_joined_into_one():
    """A jump bigger than `MAX_STEP_KM` is a different feature.

    Without this the flat line and the trace above it merge into one segment
    whose spread is the distance between them, and nothing looks flat at all.
    """
    db = _blank()
    _draw(db, 12.0, 14.0, 2642.0, 2642.0)
    _draw(db, 14.05, 16.0, 2780.0, 2780.0)  # 138 km higher, adjacent in freq
    segs = flat_tails.segments(_Ion(db))

    assert len(segs) == 2
    assert {round(s["r_med"]) for s in segs} == {2642, 2780}


def test_a_fade_of_two_bins_does_not_split_a_feature():
    db = _blank()
    _draw(db, 12.0, 16.0, 2642.0, 2642.0)
    db[np.flatnonzero((FREQ >= 14.0) & (FREQ <= 14.05)), :] = FLOOR
    segs = flat_tails.segments(_Ion(db))

    assert len(segs) == 1
    assert segs[0]["span"] == pytest.approx(4.0, abs=0.1)


def test_pure_noise_yields_no_segments():
    rng = np.random.default_rng(0)
    db = _blank() + rng.normal(0.0, 1.5, (FREQ.size, VRANGE.size))
    assert flat_tails.segments(_Ion(db)) == []
