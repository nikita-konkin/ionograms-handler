"""Estimator correctness, against synthetic soundings with a known answer.

Real recordings carry no ground truth, so this is where correctness is actually
established: IQ is built with an echo at a chosen virtual range that stops at a
chosen frequency, and every estimator has to find both.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import extractors, spectro
from muf.pick import find_runs, pick_muf

from conftest import snapped_range, synth_iq

# Small enough to keep tests quick; large enough that the picker's continuity
# rule has room to work.
WINDOW = 512
N_FREQ = 200
HALF_SPAN = 60_000.0
ECHO_RANGE = 2700.0
ECHO_LAST_BIN = 120

#: Range bins are coarse at this window (2*60000/512 = 234 km), so the echo is
#: snapped to a bin centre and compared against that.
EXPECTED_RANGE = snapped_range(ECHO_RANGE, HALF_SPAN, WINDOW)
RANGE_TOLERANCE = 2 * HALF_SPAN / WINDOW

METHODS = ["algo", "kmeans", "contour"]


@pytest.fixture
def synthetic(make_lfs):
    """A sounding whose echo sits at 2700 km and stops at bin 120."""
    iq = synth_iq(
        n_freq=N_FREQ, window=WINDOW, echo_range_km=ECHO_RANGE,
        half_span_km=HALF_SPAN, echo_last_bin=ECHO_LAST_BIN,
    )
    path = make_lfs(iq, name="synthetic.lfs")
    return spectro.compute(path, window=WINDOW, gate_km=(2000.0, 5000.0))


def test_synthetic_echo_lands_where_intended(synthetic):
    """Guard the fixture: if the injection is wrong, everything below is noise."""
    row = synthetic.power[ECHO_LAST_BIN // 2]
    assert synthetic.vrange[int(row.argmax())] == pytest.approx(
        EXPECTED_RANGE, abs=RANGE_TOLERANCE
    )


@pytest.mark.parametrize("method", METHODS)
def test_recovers_injected_muf(synthetic, method):
    """The MUF must land on the last frequency bin carrying the echo."""
    result = extractors.get(method)(synthetic)
    expected = synthetic.freq[ECHO_LAST_BIN]

    assert result.ok, f"{method} found nothing"
    # Within a few bins: the picker's continuity rule can trim the very edge,
    # and segmentation dilates by design.
    assert result.pick.muf_mhz == pytest.approx(
        expected, abs=4 * synthetic.cal.freq_step_mhz
    )


@pytest.mark.parametrize("method", METHODS)
def test_localises_echo_in_range(synthetic, method):
    """The detection mask must sit on the echo, mid-trace.

    Checked mid-trace rather than at the MUF: past the cutoff the trace has
    ended by construction, so the strongest cell there is noise, and
    segmentation's dilation can push the pick a bin or two beyond the last
    frequency carrying signal.
    """
    result = extractors.get(method)(synthetic)
    assert result.mask is not None

    mid = ECHO_LAST_BIN // 2
    lit = np.flatnonzero(result.mask[mid])
    assert lit.size, f"{method} found nothing at a frequency carrying the echo"

    ranges = synthetic.vrange[lit]
    assert np.min(np.abs(ranges - EXPECTED_RANGE)) <= RANGE_TOLERANCE


@pytest.mark.parametrize("method", METHODS)
def test_reported_range_is_the_echo_range(synthetic, method):
    """The range reported alongside the MUF must be the echo's, not noise."""
    result = extractors.get(method)(synthetic)
    index = result.pick.freq_index

    if index > ECHO_LAST_BIN:
        pytest.skip(f"{method} picked past the cutoff; covered by the MUF test")
    assert result.pick.vrange_km == pytest.approx(
        EXPECTED_RANGE, abs=RANGE_TOLERANCE
    )


@pytest.mark.parametrize("method", METHODS)
def test_no_pick_on_pure_noise(make_lfs, method):
    """A recording with no echo must yield nothing, not a value from noise.

    This is the failure the kmeans estimator originally had: with no signal
    present it still reported a MUF at the top of the band.
    """
    iq = synth_iq(
        n_freq=N_FREQ, window=WINDOW, echo_range_km=ECHO_RANGE,
        half_span_km=HALF_SPAN, echo_last_bin=-1, amplitude=0.0, seed=7,
    )
    ion = spectro.compute(make_lfs(iq, name="noise.lfs"),
                          window=WINDOW, gate_km=(2000.0, 5000.0))

    result = extractors.get(method)(ion)
    assert not result.ok, f"{method} invented {result.pick.muf_mhz} MHz from noise"


@pytest.mark.parametrize("method", METHODS)
def test_higher_cutoff_gives_higher_muf(make_lfs, method):
    """Move the echo's cutoff up and the reported MUF must follow."""
    picks = []
    for last_bin in (80, 140):
        iq = synth_iq(
            n_freq=N_FREQ, window=WINDOW, echo_range_km=ECHO_RANGE,
            half_span_km=HALF_SPAN, echo_last_bin=last_bin,
        )
        ion = spectro.compute(make_lfs(iq, name=f"cut{last_bin}.lfs"),
                              window=WINDOW, gate_km=(2000.0, 5000.0))
        result = extractors.get(method)(ion)
        assert result.ok
        picks.append(result.pick.muf_mhz)

    assert picks[1] > picks[0]


def test_methods_agree_on_synthetic(synthetic):
    results = extractors.run(synthetic, methods=tuple(METHODS))
    values = [r.pick.muf_mhz for r in results.values() if r.ok]

    assert len(values) == len(METHODS)
    assert max(values) - min(values) < 0.5


def test_runner_isolates_failures(synthetic):
    """One broken estimator must not take the others down."""
    results = extractors.run(synthetic, methods=("algo", "kmeans"),
                             kmeans={"rule": "nonsense"})

    assert results["algo"].ok
    assert results["kmeans"].error is not None
    assert not results["kmeans"].ok


def test_unknown_method():
    with pytest.raises(KeyError, match="unknown method"):
        extractors.get("does-not-exist")


# --- the shared picker -------------------------------------------------------

def test_find_runs():
    presence = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert find_runs(presence, min_run=3) == [(0, 2), (4, 7)]
    assert find_runs(presence, min_run=5) == []
    assert find_runs(np.zeros(5, dtype=bool), min_run=1) == []


def test_picker_ignores_isolated_spike():
    """An isolated detection above the trace must not become the MUF."""
    freq = np.linspace(7.5, 32.5, 100)
    presence = np.zeros(100, dtype=bool)
    presence[10:30] = True     # the trace
    presence[80] = True        # interference

    assert pick_muf(presence, freq, min_run=5).muf_mhz == pytest.approx(freq[29])
    # min_run=1 is the historical behaviour, which takes the spike.
    assert pick_muf(presence, freq, min_run=1).muf_mhz == pytest.approx(freq[80])


def test_picker_reports_no_pick_when_nothing_qualifies():
    freq = np.linspace(7.5, 32.5, 50)
    presence = np.zeros(50, dtype=bool)
    presence[10] = True

    pick = pick_muf(presence, freq, min_run=5)
    assert not pick.ok
    assert np.isnan(pick.muf_mhz)
    assert pick.n_detections == 1


def test_picker_parabolic_interpolation_is_sub_bin():
    """Interpolation must place the peak off-grid when the peak is off-grid."""
    freq = np.array([10.0, 11.0, 12.0])
    presence = np.ones(3, dtype=bool)
    vrange = np.array([3000.0, 2900.0, 2800.0])
    # Asymmetric shoulders: true peak lies between bins 0 and 1.
    power_db = np.tile(np.array([40.0, 50.0, 30.0]), (3, 1))

    off = pick_muf(presence, freq, power_db, vrange, min_run=1, parabolic=False)
    on = pick_muf(presence, freq, power_db, vrange, min_run=1, parabolic=True)

    assert off.vrange_km == 2900.0
    assert on.vrange_km != 2900.0
    assert 2900.0 < on.vrange_km < 3000.0


def test_picker_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="does not match"):
        pick_muf(np.ones(5, dtype=bool), np.linspace(0, 1, 6))


# --- the contour estimator's mask ---------------------------------------------
#
# Morphology decides *which* cells count as trace. It must not add any. Both
# tests below pin defects that shipped: a 3x3 opening that erased one-range-bin
# traces, and cv2.FILLED asserting detections in cells the sounder never lit.

def test_contour_mask_reports_only_cells_above_threshold(synthetic):
    """Every reported cell was genuinely above the detection threshold.

    Dilation and cv2.FILLED are how the contour analysis finds the trace; left
    unintersected they also get to say where it is, and 8-18% of the runs then
    described a gap between two modes rather than either of them.
    """
    from muf.extractors import contour

    result = contour.extract(synthetic)
    assert result.mask is not None
    assert result.mask.any()

    above = synthetic.db >= contour.DEFAULT_THRESHOLD_DB
    invented = result.mask & ~above
    assert not invented.any(), f"{int(invented.sum())} cells reported but never lit"


def test_opening_keeps_a_trace_one_range_bin_tall(synthetic):
    """A flat oblique leg is often one bin tall; a square kernel deletes it.

    Opening removes anything narrower than the kernel in *either* axis. The
    kernel runs along frequency only, which is the axis where a real echo
    persists and noise does not.
    """
    from muf.extractors import contour

    above = synthetic.db.T >= contour.DEFAULT_THRESHOLD_DB
    opened = contour.segment(synthetic.db, dilate_iterations=0) > 0

    assert opened.sum() > 0.5 * above.sum(), (
        f"opening kept only {opened.sum()}/{above.sum()} above-threshold cells"
    )
    assert contour.OPEN_KERNEL.shape[0] == 1, "opening must not span range bins"


def test_thresh_still_resolves_to_the_contour_estimator():
    """The old name keeps working; the result reports the new one."""
    assert extractors.canonical("thresh") == "contour"
    assert extractors.get("thresh") is extractors.get("contour")
