"""What a link at one carrier sees of a state: the channel as a modem meets it.

At a working frequency f0 the state's modes are numbers rather than curves --
each one's delay, GDD and C/N0 there (:meth:`ChannelState.at`) -- and a
modem asks a few things of them: is there a mode at all, how far apart in
delay are the modes, over how wide a band and for how long is the channel
coherent, and does the signal clear the noise in the modem's band.

A mode is seen where its C/N0 is over
:data:`muf.sim.check.CN0_COMPARE_ABOVE_DBHZ`: under it the level read off an
ionogram is the noise's, not the echo's. Every figure is drawn from the modes
seen.

* The delay spread is the RMS spread of the modes' delays weighted by their
  power, with each mode's own delay spread added in quadrature. A prior
  delay spread is zero, so where none was measured it is the spread between
  the modes alone.
* The coherence bandwidth, 1/(5 sigma), is where the frequency correlation of
  a channel of RMS delay spread sigma falls to about 0.5; one path bounds
  none.
* The coherence time is the strongest mode's, where the correlation of its
  fading falls to 0.5 (:func:`muf.sim.fading.autocorrelation`); no spread
  bounds none.
"""

from __future__ import annotations

import math

import numpy as np

from . import fading
from .check import CN0_COMPARE_ABOVE_DBHZ
from .state import ChannelState, ModeAt

#: The band a modem's SNR is read in, Hz: an HF voice channel.
BANDWIDTH_HZ = 3000.0

#: SNR in that band a link is usable from, dB.
SNR_MIN_DB = 10.0

#: Spacing of the working frequencies, MHz.
CARRIER_STEP_MHZ = 0.25

#: Correlation the coherence bandwidth and time are read at.
COHERENCE = 0.5

#: Each figure :func:`at` gives, in the order it gives them.
FIGURES = ("modes", "first_delay_us", "multipath_us", "delay_spread_us",
           "coherence_bandwidth_khz", "gdd_us_per_mhz", "cn0_dbhz", "snr_db",
           "coherence_time_s")


def carrier_grid(lo_mhz: float, hi_mhz: float,
                 step_mhz: float = CARRIER_STEP_MHZ) -> np.ndarray:
    """Every ``step_mhz`` from ``lo_mhz``, up to ``hi_mhz`` when it falls on one."""
    n = math.floor((hi_mhz - lo_mhz) / step_mhz + 1e-9)
    return np.round(lo_mhz + step_mhz * np.arange(n + 1), 6)


def present(state: ChannelState, f0_mhz: float) -> list[ModeAt]:
    """The modes a link at ``f0_mhz`` sees, earliest first."""
    return [item for item in state.at(f0_mhz)
            if item.cn0_dbhz >= CN0_COMPARE_ABOVE_DBHZ]


def coherence_time_s(spread_hz: float, shape: str) -> float | None:
    """Lag at which a fading of this spread and shape decorrelates to 0.5.

    sqrt(2 ln 2) / (pi s) = 0.375 / s for a Gaussian spectrum of F.1487's
    spread s; ln 2 / (pi s) = 0.221 / s for a Lorentzian of full width s.
    None for no spread, which never decorrelates.
    """
    if spread_hz <= 0:
        return None
    fading.check_shape(shape)
    level = -math.log(COHERENCE)
    if shape == fading.GAUSSIAN:
        return math.sqrt(2.0 * level) / (math.pi * spread_hz)
    return level / (math.pi * spread_hz)


def at(state: ChannelState, f0_mhz: float, *, bandwidth_hz: float = BANDWIDTH_HZ,
       snr_min_db: float = SNR_MIN_DB) -> dict:
    """The link's :data:`FIGURES` at ``f0_mhz``, and whether it is usable.

    ``has_mode`` and ``usable`` are the events a forecast is scored on: a mode
    seen, and an SNR in ``bandwidth_hz`` of ``snr_min_db`` or more. Figures a
    link without modes, or with one, cannot have are None.
    """
    seen = present(state, f0_mhz)
    out = {"f0_mhz": float(f0_mhz), **dict.fromkeys(FIGURES), "modes": len(seen),
           "strongest": None, "has_mode": bool(seen), "usable": False}
    if not seen:
        return out
    power = np.array([10.0 ** (m.cn0_dbhz / 10.0) for m in seen])
    delay = np.array([m.delay_us for m in seen])
    own = np.array([m.mode.delay_spread_us for m in seen])
    mean = float(np.sum(power * delay) / power.sum())
    spread = math.sqrt(float(np.sum(power * ((delay - mean) ** 2 + own ** 2))
                             / power.sum()))
    strongest = seen[int(np.argmax(power))]
    cn0 = 10.0 * math.log10(float(power.sum()))
    snr = cn0 - 10.0 * math.log10(bandwidth_hz)
    out.update(
        first_delay_us=float(delay.min()),
        multipath_us=float(delay.max() - delay.min()) if len(seen) > 1 else None,
        delay_spread_us=spread,
        coherence_bandwidth_khz=1e3 / (5.0 * spread) if spread > 0 else None,
        gdd_us_per_mhz=strongest.gdd_us_per_mhz, cn0_dbhz=cn0, snr_db=snr,
        coherence_time_s=coherence_time_s(strongest.mode.doppler_spread_hz,
                                          strongest.mode.doppler_shape),
        strongest=strongest.mode.name, usable=snr >= snr_min_db)
    return out
