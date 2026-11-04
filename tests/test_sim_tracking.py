"""Causal branch continuity, including loss of evidence and identity changes."""

import json

import numpy as np
import pytest

from muf import trace
from muf.sim import tracking


def arch(peak=12.0, amplitude=20.0, offset=2800.0):
    f = np.linspace(10.0, 14.0, 161)
    return trace.Segment(f, offset - amplitude * (f - peak) ** 2)


def test_small_peak_movement_keeps_the_boundary_without_accumulating_drift():
    _, history, _ = tracking.split([arch()])
    cut = history[0]["cuts"]
    for peak in [12.2, 11.9, 12.1] * 4:
        pieces, history, _ = tracking.split([arch(peak)], history)
        assert len(pieces) == 2
        assert history[0]["cuts"] == cut
        np.testing.assert_array_equal(np.concatenate([p.freq for p in pieces]),
                                      arch().freq)


def test_hysteresis_keeps_a_weak_hump_but_releases_a_vanished_one():
    _, history, _ = tracking.split([arch()])
    weak = arch(amplitude=3.0)
    assert len(tracking.split([weak])[0]) == 1
    assert len(tracking.split([weak], history)[0]) == 2
    assert len(tracking.split([arch(amplitude=0.0)], history)[0]) == 1


def test_a_shifting_frequency_grid_does_not_ratchet_the_boundary():
    _, history, _ = tracking.split([arch()])
    cut = history[0]["cuts"]
    for k in range(1, 20):
        f = arch().freq + k * 0.001
        item = trace.Segment(f, 2800.0 - 20.0 * (f - 12.0) ** 2)
        _, history, _ = tracking.split([item], history)
        assert history[0]["cuts"] == cut


def test_a_peak_that_really_moves_is_not_locked_to_the_old_cut():
    _, history, _ = tracking.split([arch()])
    _, moved, _ = tracking.split([arch(peak=12.8)], history)
    assert moved[0]["cuts"][0] == pytest.approx(12.8, abs=0.06)


def test_history_follows_geometry_when_track_order_changes():
    _, history, _ = tracking.split([arch(), arch(offset=3000)])
    history[0]["names"] = ["F", "F#2"]
    history[1]["names"] = ["2F", "2F#2"]
    _, _, names = tracking.split([arch(offset=3000), arch()], history)
    assert [item[2] for item in names.values()] == ["2F", "2F#2", "F", "F#2"]


def test_a_distant_or_ambiguous_track_cannot_inherit_a_cut():
    _, history, _ = tracking.split([arch()])
    weak = arch(amplitude=3.0, offset=3100)
    assert len(tracking.split([weak], history)[0]) == 1
    weak = arch(amplitude=3.0)
    assert len(tracking.split([weak], history * 2)[0]) == 1


def test_history_round_trips_as_json_without_changing_the_result():
    _, history, _ = tracking.split([arch()])
    restored = json.loads(json.dumps(history))
    assert tracking.split([arch(peak=12.1)], history)[1] == (
        tracking.split([arch(peak=12.1)], restored)[1])
