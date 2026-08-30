"""The DSP estimator: a vectorised port of ``stuffr.filter2_np_nb_MUF``.

The original (``stuffr.py:272``) walks every cell of the spectrogram in a
triple-nested Python loop looking for three vertically adjacent above-threshold
cells whose range-neighbours are also lit. Its detection rule is preserved here
exactly; only the implementation changes, from ~110M interpreted iterations to a
handful of array operations.

Three pre-existing defects are fixed. Each shifts the numbers, so ``legacy=True``
is provided to reproduce the historical decision where that is possible:

``stuffr.py:276``
    ``adjacent_spec_num = np.empty(3)`` is never initialised, so the first
    detections compare against uninitialised memory.

``stuffr.py:285``
    ``np.append(adjacent_spec_num, counter)`` discards its return value --
    ``np.append`` does not mutate in place -- so for the first three hits the
    buffer is never actually written.

``stuffr.py:319``
    ``np.amax(MUFs, axis=0)`` takes the maximum of each column *independently*,
    so the returned ``(frequency, range, row)`` triple is assembled from
    different detections and does not describe any single echo. The reported
    virtual range is therefore not the range at the reported MUF. This one
    cannot be faithfully reproduced under ``legacy`` without reproducing the
    defect; ``legacy`` reproduces the frequency, which is unaffected, and
    reports the range coherently.

Additionally, the original reads range-neighbours at ``k-1``/``k+1`` without
bounds checking: at ``k=0`` this wraps to the far end of the array, and at the
last column it raises ``IndexError``, which is caught and printed while the
stale values from the previous iteration are used. Edge columns are excluded
here instead.
"""

from __future__ import annotations

import numpy as np

from ..pick import DEFAULT_BRIDGE, DEFAULT_MIN_RUN, pick_muf
from ..spectro import Ionogram
from . import MufResult

# Historical thresholds, in median-equalized linear power. `filter_func` in
# stuffr.py defaults to k=20; the neighbour test is hardcoded at 10.
DEFAULT_THRESHOLD = 20.0
DEFAULT_NEIGHBOUR_THRESHOLD = 10.0

#: Consecutive frequency bins that must be lit for a detection. Fixed at 3 in
#: the original; exposed here because it is the rule's main sensitivity knob.
RUN_LENGTH = 3

#: How many range-neighbours of an anchor must be lit: 1 or 2.
#:
#: The original demanded **both**, and that quietly requires the trace to be at
#: least three range bins tall. An oblique trace is often not: measured
#: 2026-08-30 on four soundings where this estimator read 2-10 MHz below
#: `contour`, 40-57% of lit frequencies carried a trace two range bins thick or
#: less. Requiring both neighbours discards every one of those cells, and
#: relaxing to either raised detections by 25-70% per sounding.
#:
#: This is the same defect `contour` was fixed for on 2026-02-04, where a 3x3
#: opening was erasing the flat low-ray leg for the same reason -- "opening
#: removes anything narrower than the kernel in *either* axis". Its kernel
#: became frequency-only; this is the equivalent correction here.
#:
#: One neighbour rather than none: an isolated cell with nothing above or below
#: it is what the test is there to reject, and that much of the original rule
#: is right.
DEFAULT_NEIGHBOURS = 1


def detect(
    power: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    neighbour_threshold: float = DEFAULT_NEIGHBOUR_THRESHOLD,
    run_length: int = RUN_LENGTH,
    neighbours: int = DEFAULT_NEIGHBOURS,
) -> np.ndarray:
    """Boolean ``[n_freq, n_range]`` mask of cells that anchor a detection.

    A cell is marked when it is the middle of ``run_length`` consecutive
    above-threshold cells along frequency, and at least ``neighbours`` of its
    two range-neighbours are above ``neighbour_threshold``.

    ``neighbours=2`` is the original's rule and requires a trace three range
    bins tall; see :data:`DEFAULT_NEIGHBOURS` for why that is the wrong test for
    an oblique trace.
    """
    n_freq, n_range = power.shape
    mask = np.zeros_like(power, dtype=bool)
    if n_freq < run_length or n_range < 3:
        return mask

    lit = power > threshold

    # Consecutive run along the frequency axis. run[i] is True when rows
    # i .. i+run_length-1 are all lit at that range bin.
    run = lit[: n_freq - run_length + 1].copy()
    for offset in range(1, run_length):
        run &= lit[offset: n_freq - run_length + 1 + offset]

    # Anchor on the middle row of each run, as the original does.
    middle = run_length // 2
    anchored = np.zeros_like(mask)
    anchored[middle: middle + run.shape[0]] = run

    # Range-neighbours of the anchor. Edge columns have only one neighbour and
    # are excluded, as in the original.
    strong = power > neighbour_threshold
    lit_neighbours = np.zeros_like(mask, dtype=np.uint8)
    lit_neighbours[:, 1:-1] = (strong[:, :-2].astype(np.uint8)
                               + strong[:, 2:].astype(np.uint8))

    return anchored & (lit_neighbours >= neighbours)


def extract(
    ion: Ionogram,
    threshold: float = DEFAULT_THRESHOLD,
    neighbour_threshold: float = DEFAULT_NEIGHBOUR_THRESHOLD,
    run_length: int = RUN_LENGTH,
    min_run: int = DEFAULT_MIN_RUN,
    max_range_slope: float | None = None,
    percentile: float = 100.0,
    neighbours: int = DEFAULT_NEIGHBOURS,
    bridge: int = DEFAULT_BRIDGE,
    legacy: bool = False,
) -> MufResult:
    """Estimate MUF with the historical DSP rule.

    Args:
        ion: the gated ionogram.
        threshold: linear median-equalized power for the main test.
        neighbour_threshold: linear power required of the range-neighbours.
        run_length: consecutive frequency bins forming a detection.
        min_run: consecutive *detections* required by the shared picker. The
            original had no such requirement; ``legacy=True`` forces 1.
        percentile: passed to the picker.
        neighbours: range-neighbours an anchor needs. 2 is the original rule.
        bridge: undetected frequency bins the picker may skip over.
        legacy: reproduce the original decision rule -- no continuity test, no
            gap bridging, and both range-neighbours required.
    """
    if legacy:
        min_run = 1
        bridge = 0
        neighbours = 2

    mask = detect(ion.power, threshold, neighbour_threshold, run_length,
                  neighbours)
    presence = mask.any(axis=1)

    pick = pick_muf(
        presence, ion.freq,
        power_db=ion.db, vrange=ion.vrange,
        min_run=min_run, percentile=percentile,
        max_range_slope=max_range_slope, bridge=bridge,
    )
    return MufResult(
        method="algo", pick=pick, presence=presence, mask=mask,
    )
