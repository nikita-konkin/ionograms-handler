"""The pairs the receiver's clock broke, set apart from a forecast's figures.

An ionozond receiver whose GPSDO has lost lock free-runs, and every echo
of a sounding moves by its clock's error at that sounding. A forecast
scored across a jump of that error measures the jump: its delay error is the
jump itself, and a truth read on the forecast's range gate cuts off the
modes, and the MUF with them, where the echo has slid out of it. One pair
cannot tell the clock from the ionosphere by its errors, but the echo's
start can. A path's earliest echo begins just past its ground line, and
from one sounding to the next it moves only as fast as the layer it comes
off.

So a pair is **clock-suspect** when its echo began more than
:data:`JUMP_US` apart in its two soundings, or began where no path can put
it in either: earlier than :data:`EARLY_S` before the ground line, or later
than :data:`LATE_HOPS` hops at :data:`LATE_HEIGHT_KM` -- the window the
station's own check (``echo_offset`` in ``services/agent/health.py``)
judges its clock by, and which it keeps itself because it runs without muf.
The limit sits above the largest move between neighbouring soundings, ten
or twenty minutes apart, on a day the receiver held its lock (under
0.2 ms), and below the jumps of the days it did not.
A sounding whose echo was not read (:func:`muf.echo_window.echo_offset_s`)
is evidence neither way.

The jump is not taken out of the truth. It holds the ionosphere's own
change between the soundings as well as the clock's, so a delay error
corrected by it would not be a measurement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..geometry import EARTH_RADIUS_KM
from ..io_chirp import C_M_S
from .state import ChannelState

#: Microseconds the echo's start may move between a pair's soundings.
JUMP_US = 250.0

#: How far before the ground line an echo may begin: the station's
#: ``EPOCH_EARLY_LIMIT_S``.
EARLY_S = 0.5e-3

#: The latest a path's earliest echo can begin: this many hops at this
#: virtual height, the station's ``ECHO_MAX_HOPS`` and ``ECHO_MAX_HEIGHT_KM``.
LATE_HOPS = 2
LATE_HEIGHT_KM = 450.0

#: Where a state keeps its echo's start against the ground line, us.
OFFSET = "echo_offset_us"


def window_us(ground_km: float) -> tuple[float, float]:
    """Where a path's earliest echo can begin against its ground line, us.

    The late edge is the station's ``hop_excess_s``: each hop two equal
    slant legs to a mirror at :data:`LATE_HEIGHT_KM` over a spherical Earth.
    """
    phi = ground_km / LATE_HOPS / 2.0 / EARTH_RADIUS_KM
    r, rh = EARTH_RADIUS_KM, EARTH_RADIUS_KM + LATE_HEIGHT_KM
    leg = math.sqrt(max(0.0, r * r + rh * rh - 2 * r * rh * math.cos(phi)))
    return (-EARLY_S * 1e6,
            max(0.0, 2.0 * LATE_HOPS * leg - ground_km) / (C_M_S / 1e9))


def offset_us(state: ChannelState) -> float | None:
    """Where ``state``'s echo began against the ground line, us; None unread."""
    value = state.metrics.get(OFFSET)
    return None if value is None or not math.isfinite(value) else float(value)


def verdict(before: ChannelState, after: ChannelState, *,
            jump_us: float = JUMP_US) -> dict:
    """Whether the pair of the soundings ``before`` and ``after`` is clock-suspect.

    ``after`` is the truth's sounding's own state, not one read on the
    forecast's range. Returns ``suspect``, the echo's ``jump_us`` between
    them, their ``offsets_us`` and ``why`` in words, '' when not suspect.
    """
    a, b = offset_us(before), offset_us(after)
    jump = None if a is None or b is None else abs(b - a)
    why = [] if jump is None or jump <= jump_us else [
        f"the echo moved {jump:.0f} us between the soundings"]
    for value, state, whose in ((a, before, "forecast's"), (b, after, "truth's")):
        ground = state.circuit.ground_km
        if value is None or not ground:
            continue
        early, late = window_us(ground)
        if not early < value < late:
            why.append(f"the {whose} echo begins {value / 1e3:+.2f} ms from the ground "
                       f"line, outside {early / 1e3:+.1f} to {late / 1e3:+.2f} ms")
    return {"suspect": bool(why), "jump_us": jump, "offsets_us": [a, b],
            "why": "; ".join(why)}


def suspect(row: Mapping) -> bool:
    """Whether a scored row -- a live event, a series' pair -- is clock-suspect;
    a row from before the check is not."""
    return bool((row.get("clock") or {}).get("suspect"))
