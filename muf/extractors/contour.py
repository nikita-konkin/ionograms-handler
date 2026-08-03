"""The contour estimator: threshold, clean up, then take contours.

Ported from ``MUF_clustering/segment_ionogram.py``: threshold in dB,
morphological open to drop speckle, dilate to close the trace, then
``cv2.findContours``. The original derived MUF from the right-hand edge of the
bounding box of the *tallest* contour -- tallest in virtual range, which near
the MUF is the right instinct, since the trace turns sharply upward as it
approaches the critical frequency.

The source applied this to a rendered PNG; here it runs on the dB array, so the
threshold is a real signal level rather than a pixel value recovered from a
colormap.

It was called ``thresh`` until the threshold turned out to be the one thing it
does *not* do differently: it uses the same 43 dB level as the algorithmic
estimator, by construction. What distinguishes it is the contour analysis, so
that is its name.

This is the cheapest of the estimators and makes a good sanity baseline: when it
disagrees sharply with the other two, the sounding is usually genuinely
ambiguous rather than the method being at fault.

**Morphology selects cells; it does not create them.** Two defects came from
losing that distinction, both measured on 2026-02-04:

*The 3x3 opening erased thin traces.* Opening removes anything narrower than the
kernel in *either* axis, and the flat low-ray leg of an oblique trace is often
one range bin tall. At 14:00 it discarded 91% of the above-threshold cells
(827 -> 72); across the day it kept only 9-72%. The kernel is now ``(1, 3)`` --
frequency only -- which still removes the speckle it is there for (8-31% of
cells) while a one-bin-tall trace survives. A real echo persists across
frequency; noise does not, and that is the axis the test belongs on.

*Dilation and ``cv2.FILLED`` invented detections.* Only 32-46% of the cells in
the old mask were ever above threshold; the rest were the dilation skirt and the
filled interior of a contour's outline. 8-18% of the runs handed to
:func:`muf.trace.extract_points` contained no above-threshold cell at all, so
their reported group range was the centroid of a gap -- visibly floating above
the trace it claimed to describe. The retained-contour mask is now intersected
back with the cells that were actually above threshold, so the morphology only
decides *which* detections count.

Together these take the method from 53 to 151 points on the 03:00 sounding with
none of them invented, and widen its reach from 10.42 to 9.44 MHz. The MUF moves
by at most 0.33 MHz over the soundings checked, always outward, so this is a
trace-quality fix rather than a recalibration.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..pick import DEFAULT_MIN_RUN, pick_muf
from ..spectro import Ionogram
from . import MufResult

#: The dB equivalent of the algorithmic estimator's linear threshold of 20,
#: given ``spectro.to_db``'s 1e-3 reference: 10*log10(20/1e-3) = 43 dB. Using
#: the same physical level keeps the two methods comparable. The source scripts
#: used 45-60 dB against pixel values on a different scale.
DEFAULT_THRESHOLD_DB = 43.0

DEFAULT_OPEN_ITERATIONS = 1
DEFAULT_DILATE_ITERATIONS = 2

#: Opening runs along frequency only, so a one-range-bin-tall trace survives it.
#: See the module docstring: a 3x3 kernel here cost up to 91% of the detections.
OPEN_KERNEL = np.ones((1, 3), np.uint8)

#: Dilation stays square: its job is to close the trace into one connected
#: component for ``findContours`` and to give it the height ``min_height``
#: tests. Nothing it adds reaches the output -- ``extract`` intersects the
#: result back with the cells that were genuinely above threshold.
DILATE_KERNEL = np.ones((3, 3), np.uint8)

#: Contours shorter than this in virtual-range bins are discarded as speckle.
DEFAULT_MIN_HEIGHT = 2

#: Registry name.
NAME = "contour"


def segment(
    db: np.ndarray,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    open_iterations: int = DEFAULT_OPEN_ITERATIONS,
    dilate_iterations: int = DEFAULT_DILATE_ITERATIONS,
) -> np.ndarray:
    """Binary trace mask, in image orientation ``[n_range, n_freq]``."""
    # cv2 wants rows=y. Frequency is the x axis here, matching the source.
    binary = ((db.T >= threshold_db) * 255).astype(np.uint8)
    if open_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, OPEN_KERNEL,
                                  iterations=open_iterations)
    if dilate_iterations:
        binary = cv2.dilate(binary, DILATE_KERNEL, iterations=dilate_iterations)
    return binary


def contour_mask(
    binary: np.ndarray,
    select: str = "all",
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> tuple[np.ndarray, int]:
    """Mask of retained contours, plus how many were retained.

    ``select='tallest'`` keeps only the single tallest contour, reproducing
    ``segment_ionogram.py``; ``'all'`` keeps every contour taller than
    ``min_height`` and lets the shared picker arbitrate.
    """
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[-2]        # cv2 3.x returned 3 values, 4.x returns 2

    kept = []
    tallest, tallest_h = None, -1
    for contour in contours:
        _, _, _, h = cv2.boundingRect(contour)
        if h > tallest_h:
            tallest, tallest_h = contour, h
        if h >= min_height:
            kept.append(contour)

    if select == "tallest":
        kept = [tallest] if tallest is not None else []
    elif select != "all":
        raise ValueError(f"unknown contour selection {select!r}")

    mask = np.zeros_like(binary, dtype=np.uint8)
    if kept:
        cv2.drawContours(mask, kept, -1, color=1, thickness=cv2.FILLED)
    return mask.astype(bool), len(kept)


def extract(
    ion: Ionogram,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    open_iterations: int = DEFAULT_OPEN_ITERATIONS,
    dilate_iterations: int = DEFAULT_DILATE_ITERATIONS,
    select: str = "all",
    min_height: int = DEFAULT_MIN_HEIGHT,
    min_run: int = DEFAULT_MIN_RUN,
    percentile: float = 100.0,
) -> MufResult:
    """Estimate MUF by thresholding and contour analysis."""
    binary = segment(ion.db, threshold_db, open_iterations, dilate_iterations)
    mask_img, _ = contour_mask(binary, select=select, min_height=min_height)

    # Back to [n_freq, n_range], intersected with the cells that were actually
    # above threshold: morphology selects detections, it does not create them.
    #
    # Dilation and cv2.FILLED are how the contour analysis *finds* the trace.
    # Left to assert where it is, they put 8-18% of runs on cells the sounder
    # never lit, and a fused run's centroid then lands in the gap between two
    # modes rather than on either.
    #
    # Intersecting the presence array too means frequency continuity is decided
    # in exactly one place -- pick_muf's ``min_run`` -- instead of half here, in
    # a dilation whose reach is invisible to the caller. It costs nothing
    # measurable: over every fourth sounding of 2026-02-04 the two choices
    # differ on one sounding out of 72.
    mask = mask_img.T & (ion.db >= threshold_db)
    presence = mask.any(axis=1)

    pick = pick_muf(
        presence, ion.freq,
        power_db=ion.db, vrange=ion.vrange,
        min_run=min_run, percentile=percentile,
    )
    return MufResult(method=NAME, pick=pick, presence=presence, mask=mask)
