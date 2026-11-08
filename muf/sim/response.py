"""A mode's frequency response, read subchannel by subchannel off a dechirped record.

After dechirping, an echo of delay tau is a tone at the beat ``beat(tau)``,
and a window of the record T long is an instantaneous delay profile with
bins 1/T of beat apart. The sweep moves ``rate * T`` in that time, so the
window sees the channel through a subchannel B = rate * T wide, and the
complex value at the mode's peak is one sample of its response H(f) there.
This is the playground's ``#fr`` page (``ionozond-playground/js/modules/fr.js``)
in muf, so that it can be checked on synthetic soundings, where the truth is
known, and shown for any stored sounding that keeps the phase (#103).

**The windows.** A Hann window of ``round(B / rate * sr)`` samples starts
every ``hop`` samples: one window after another (no overlap) gives one sample
per subchannel, overlapping windows a sample every ``hop``, down to one at
every record sample. Both series are measured, the stepped one always, since
what the overlap buys is seen only against it.

**The gate.** Each window's peak is looked for within a gate around the
mode's curve, ``tau(f) + shift`` at the window's centre, or around one delay
-- the curve at the stretch's middle -- when the gate is held. Its
half-width is set by hand, or else ``max(GATE_US, GATE_BINS * bin)``: the
curve is the ionogram's trace, good to tens of microseconds, and a narrow
subchannel's bin is wider than that. The peak's delay is refined between bins
by a parabola through the log power of its neighbours. For a Hann window that
parabola is biased by up to a sixtieth of a bin, a pattern that repeats bin
after bin: a curve that crosses less than a bin over the stretch has its
slope tilted by it, and the GDD read off the peaks wants a wide subchannel.

**The phase.** Every window's value is referred to the record's own clock,
then turned by its bin's offset from the beat of one reference delay, the
stepped peaks' median: what is left is the channel's phase against a path of
that delay, given in the channel's convention,
phi(f) = -2 pi * integral of (tau - tau_ref) df, whichever way the record's
axis runs. It is unwrapped run by run; a step of more than pi/2 is counted as
a jump, a turn the unwrapping may have lost. Windows a subchannel apart see
the phase move 2 pi (tau - tau_ref) B between them, so on a wide subchannel
only overlapping windows unwrap it. The stepped phase is put on the sliding
one's turn at its first sample, and every stepped sample more than pi from the
sliding phase nearest it in frequency is marked lost.

**The level.** The peak bin's power against the noise of one bin: the median
over the stepped windows of each window's median bin, divided by ln 2, the
median of an exponential. Most bins of a window are noise, so the reference
needs no knowledge of where the other modes are; with several modes in a
narrow window it reads a little high.

**GDD two ways.** From the phase's curvature, -phi''/(2 pi), and from the
slope of the peaks' delays, both over the sliding samples MIN_LEVEL_DB or more
above the noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from .fast import Record
from .state import Mode

#: The playground page's subchannels, Hz.
SUBCHANNELS_HZ = (5.12e3, 8e3, 10.24e3, 20.48e3, 40.96e3, 61.44e3, 81.92e3, 102.4e3)
SUBCHANNEL_HZ = 20.48e3
OVERLAP = 0.5

#: The automatic gate's half-width: this many us, or this many bins if wider.
GATE_US = 60.0
GATE_BINS = 1.5

#: The shortest window, samples.
MIN_WINDOW = 8

#: Samples weaker than this, dB above the noise, stay out of the GDD fits.
MIN_LEVEL_DB = 6.0


@dataclass(frozen=True)
class Gate:
    """Where a window's peak is looked for: about the mode's curve or at one delay.

    ``shift_us`` moves the gate off the curve; ``half_us`` is its half-width,
    None for the automatic one.
    """

    held: bool = False
    shift_us: float = 0.0
    half_us: float | None = None


def _listed(values) -> list:
    return [round(float(v), 6) if np.isfinite(v) else None for v in values]


def _number(value: float):
    return round(float(value), 6) if math.isfinite(value) else None


@dataclass(frozen=True)
class Series:
    """One sample per window: its peak, its level and the response's phase there.

    ``valid`` is false where the gate had nowhere to be -- outside the mode's
    band -- and the other arrays are NaN there. ``lost`` marks the stepped
    samples a turn off the sliding phase; it is all false on the sliding
    series itself.
    """

    freq_mhz: np.ndarray
    gate_us: np.ndarray
    delay_us: np.ndarray
    level_db: np.ndarray
    phase_rad: np.ndarray
    valid: np.ndarray
    lost: np.ndarray
    jumps: int
    steps: int

    def as_dict(self) -> dict:
        return {"freq_mhz": _listed(self.freq_mhz), "gate_us": _listed(self.gate_us),
                "delay_us": _listed(self.delay_us), "level_db": _listed(self.level_db),
                "phase_rad": _listed(self.phase_rad), "lost": self.lost.tolist(),
                "jumps": self.jumps, "steps": self.steps}


@dataclass(frozen=True)
class Response:
    """A mode's frequency response over a stretch, at one subchannel and overlap."""

    mode: str
    source: str
    f_lo_mhz: float
    f_hi_mhz: float
    subchannel_hz: float
    window: int
    hop: int
    bin_us: float
    gate: Gate
    gate_half_us: float
    gate_auto_us: float
    delay_ref_us: float
    noise: float
    snr_db: float
    gdd_phase_us_per_mhz: float
    gdd_trace_us_per_mhz: float
    stepped: Series = field(repr=False)
    sliding: Series = field(repr=False)

    def as_dict(self) -> dict:
        return {"mode": self.mode, "source": self.source, "f_lo_mhz": self.f_lo_mhz,
                "f_hi_mhz": self.f_hi_mhz, "subchannel_hz": _number(self.subchannel_hz),
                "window": self.window, "hop": self.hop, "bin_us": _number(self.bin_us),
                "gate": {"held": self.gate.held, "shift_us": self.gate.shift_us,
                         "half_us": _number(self.gate_half_us),
                         "auto_us": _number(self.gate_auto_us)},
                "delay_ref_us": _number(self.delay_ref_us),
                "snr_db": _number(self.snr_db),
                "gdd_phase_us_per_mhz": _number(self.gdd_phase_us_per_mhz),
                "gdd_trace_us_per_mhz": _number(self.gdd_trace_us_per_mhz),
                "stepped": self.stepped.as_dict(), "sliding": self.sliding.as_dict()}


@dataclass(frozen=True)
class _Windows:
    """The windows starting at some samples: where they sit, their gates and peaks."""

    time_s: np.ndarray
    freq_mhz: np.ndarray
    gate_us: np.ndarray
    valid: np.ndarray
    bins: np.ndarray
    delay_us: np.ndarray
    power: np.ndarray
    value: np.ndarray


def _beat(record: Record, delay_us) -> np.ndarray:
    """The beat of a delay before any mix-down, Hz."""
    return (np.asarray(delay_us, dtype=np.float64) * record.km_per_us
            - record.range_zero_km) / record.km_per_hz


def _delay(record: Record, beat_hz) -> np.ndarray:
    return (np.asarray(beat_hz, dtype=np.float64) * record.km_per_hz
            + record.range_zero_km) / record.km_per_us


def _taper(size: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1, size + 1) / (size + 1))


def _peaks(record: Record, taper: np.ndarray, starts: np.ndarray, lo: np.ndarray,
           hi: np.ndarray, span_s: float):
    """Each window's strongest bin in [lo, hi], its value and power, the peak's offset.

    One bin at a time over every window that needs it: the record turned by
    the bin's beat once, its windows' sums a convolution with the taper.
    """
    count, size = starts.size, taper.size
    best = np.full(count, -1.0)
    where = lo.copy()
    value = np.zeros(count, dtype=np.complex128)
    below = np.zeros(count)
    above = np.zeros(count)
    previous = np.full(count, np.nan)
    for m in range(int(lo.min()) - 1, int(hi.max()) + 2):
        near = np.nonzero((lo - 1 <= m) & (m <= hi + 1))[0]
        power = np.full(count, np.nan)
        x = np.zeros(count, dtype=np.complex128)
        if near.size:
            a, b = int(starts[near].min()), int(starts[near].max())
            t = record.t_first_s + np.arange(a, b + size) / record.sr_hz
            turned = record.samples[a:b + size] * np.exp(
                -2j * np.pi * (m / span_s - record.shift_hz) * t)
            sums = signal.oaconvolve(turned, taper[::-1], mode="valid")
            x[near] = sums[starts[near] - a]
            power[near] = np.abs(x[near]) ** 2
        after = (where == m - 1) & np.isfinite(power)
        above[after] = power[after]
        better = (lo <= m) & (m <= hi) & (power > best)
        best[better] = power[better]
        where[better] = m
        value[better] = x[better]
        below[better] = previous[better]
        previous = power
    low, mid, high = (np.log(v + 1e-300) for v in (below, best, above))
    den = low - 2.0 * mid + high
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(den < 0, np.clip(0.5 * (low - high) / den, -0.5, 0.5), 0.0)
    return where, value, best, frac


def _windows(record: Record, taper: np.ndarray, starts: np.ndarray, span_s: float,
             centre, half_us: float) -> _Windows:
    size = taper.size
    time = record.t_first_s + (starts + (size - 1) / 2.0) / record.sr_hz
    freq = np.asarray(record.freq_at(time), dtype=np.float64)
    gate = np.asarray(centre(freq), dtype=np.float64)
    valid = np.isfinite(gate)
    bins = np.full(starts.size, np.nan)
    delay = np.full(starts.size, np.nan)
    power = np.full(starts.size, np.nan)
    value = np.full(starts.size, np.nan, dtype=np.complex128)
    if valid.any():
        b1 = _beat(record, gate[valid] - half_us) * span_s
        b2 = _beat(record, gate[valid] + half_us) * span_s
        lo = np.ceil(np.minimum(b1, b2)).astype(np.int64)
        hi = np.floor(np.maximum(b1, b2)).astype(np.int64)
        empty = hi < lo
        lo[empty] = hi[empty] = np.round(_beat(record, gate[valid][empty]) * span_s)
        where, x, p, frac = _peaks(record, taper, starts[valid], lo, hi, span_s)
        bins[valid] = where
        delay[valid] = _delay(record, (where + frac) / span_s)
        power[valid] = p
        value[valid] = x
    return _Windows(time_s=time, freq_mhz=freq, gate_us=np.where(valid, gate, np.nan),
                    valid=valid, bins=bins, delay_us=delay, power=power, value=value)


def _noise(record: Record, taper: np.ndarray, starts: np.ndarray) -> float:
    frames = sliding_window_view(record.samples, taper.size)[starts] * taper
    power = np.abs(np.fft.fft(frames, axis=1)) ** 2
    return float(np.median(np.median(power, axis=1)) / math.log(2.0))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))


def _phase(w: _Windows, span_s: float, beat_ref: float,
           flip: bool) -> tuple[np.ndarray, int, int]:
    """The phase against the reference beat, unwrapped run by run, and its jumps."""
    raw = np.angle(w.value) + 2.0 * np.pi * (w.bins / span_s - beat_ref) * w.time_s
    raw = -raw if flip else raw
    out = np.full(raw.shape, np.nan)
    jumps = steps = 0
    for a, b in _runs(w.valid):
        step = np.angle(np.exp(1j * np.diff(raw[a:b])))
        out[a:b] = raw[a] + np.concatenate([[0.0], np.cumsum(step)])
        jumps += int(np.sum(np.abs(step) > np.pi / 2))
        steps += step.size
    return out, jumps, steps


def _aligned(stepped: _Windows, step_phase: np.ndarray, sliding: _Windows,
             slide_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The stepped phase on the sliding one's turn, and its samples a turn off it."""
    lost = np.zeros(stepped.valid.size, dtype=bool)
    ok = np.nonzero(sliding.valid)[0]
    if ok.size < 2 or not stepped.valid.any():
        return step_phase, lost
    freq = sliding.freq_mhz[ok]
    order = np.argsort(freq)
    freq, ok = freq[order], ok[order]

    def nearest(f):
        i = np.clip(np.searchsorted(freq, f), 1, freq.size - 1)
        pick = np.where(np.abs(freq[i - 1] - f) <= np.abs(freq[i] - f), i - 1, i)
        return slide_phase[ok[pick]]

    first = int(np.nonzero(stepped.valid)[0][0])
    gap = nearest(stepped.freq_mhz[first:first + 1])[0] - step_phase[first]
    step_phase = step_phase + 2.0 * np.pi * np.round(gap / (2.0 * np.pi))
    valid = stepped.valid
    lost[valid] = np.abs(nearest(stepped.freq_mhz[valid]) - step_phase[valid]) > np.pi
    return step_phase, lost


def _series(w: _Windows, noise: float, phase: np.ndarray, jumps: int, steps: int,
            lost: np.ndarray) -> Series:
    return Series(freq_mhz=w.freq_mhz, gate_us=w.gate_us, delay_us=w.delay_us,
                  level_db=10.0 * np.log10(w.power / noise + 1e-300), phase_rad=phase,
                  valid=w.valid, lost=lost, jumps=jumps, steps=steps)


def _fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray | None:
    if x.size <= degree + 1:
        return None
    return np.polyfit(x, y, degree)


def measure(record: Record, mode: Mode, f_lo_mhz: float, f_hi_mhz: float, *,
            subchannel_hz: float = SUBCHANNEL_HZ, overlap: float = OVERLAP,
            gate: Gate = Gate()) -> Response:
    """The mode's response over [f_lo, f_hi] MHz, from the windows centred within it."""
    if not f_hi_mhz > f_lo_mhz:
        raise ValueError(f"the stretch {f_lo_mhz}-{f_hi_mhz} MHz ends before it starts")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap {overlap} is outside 0 to 1")
    if not subchannel_hz > 0:
        raise ValueError(f"subchannel {subchannel_hz} Hz is not positive")
    if gate.half_us is not None and not gate.half_us > 0:
        raise ValueError(f"gate half-width {gate.half_us} us is not positive")
    sr, rate = record.sr_hz, record.rate_hz_s
    size = max(MIN_WINDOW, round(subchannel_hz / abs(rate) * sr))
    span_s = size / sr
    hop = 1 if overlap >= 1.0 else max(1, round(size * (1.0 - overlap)))
    ends = (np.asarray(record.time_at([f_lo_mhz, f_hi_mhz])) - record.t_first_s) * sr
    s0 = max(0, math.ceil(float(ends.min()) - (size - 1) / 2.0))
    s1 = min(record.samples.size - size,
             math.floor(float(ends.max()) - (size - 1) / 2.0))
    if s1 < s0:
        raise ValueError(f"{record.source}: {f_lo_mhz}-{f_hi_mhz} MHz holds no "
                         f"{abs(rate) * span_s / 1e3:.2f} kHz window")
    hz_per_us = record.km_per_us / record.km_per_hz
    bin_us = 1.0 / (span_s * abs(hz_per_us))
    auto = max(GATE_US, GATE_BINS * bin_us)
    half = gate.half_us if gate.half_us is not None else auto
    middle = (f_lo_mhz + f_hi_mhz) / 2.0
    held = float(mode.delay_at(min(max(middle, mode.f_lo), mode.f_hi))) + gate.shift_us

    def centre(freq):
        if gate.held:
            return np.full(freq.shape, held)
        return mode.delay_at(freq) + gate.shift_us

    taper = _taper(size)
    starts = np.arange(s0, s1 + 1, size)
    noise = _noise(record, taper, starts)
    stepped = _windows(record, taper, starts, span_s, centre, half)
    sliding = stepped if hop == size else _windows(
        record, taper, np.arange(s0, s1 + 1, hop), span_s, centre, half)
    ref = float(np.round(np.median(stepped.delay_us[stepped.valid]))) \
        if stepped.valid.any() else held
    beat_ref = float(_beat(record, ref))
    flip = hz_per_us * rate > 0
    step_phase, step_jumps, step_steps = _phase(stepped, span_s, beat_ref, flip)
    if sliding is stepped:
        lost = np.zeros(stepped.valid.size, dtype=bool)
        out_stepped = _series(stepped, noise, step_phase, step_jumps, step_steps, lost)
        out_sliding = out_stepped
    else:
        slide_phase, slide_jumps, slide_steps = _phase(sliding, span_s, beat_ref, flip)
        step_phase, lost = _aligned(stepped, step_phase, sliding, slide_phase)
        out_stepped = _series(stepped, noise, step_phase, step_jumps, step_steps, lost)
        out_sliding = _series(sliding, noise, slide_phase, slide_jumps, slide_steps,
                              np.zeros(sliding.valid.size, dtype=bool))
    good = out_sliding.valid & (out_sliding.level_db >= MIN_LEVEL_DB)
    x = out_sliding.freq_mhz[good] - middle
    curve = _fit(x, out_sliding.phase_rad[good], 2)
    slope = _fit(x, out_sliding.delay_us[good], 1)
    scale = abs(hz_per_us * 1e6 / rate)
    return Response(
        mode=mode.name, source=record.source, f_lo_mhz=float(f_lo_mhz),
        f_hi_mhz=float(f_hi_mhz), subchannel_hz=abs(rate) * span_s, window=size,
        hop=hop,
        bin_us=bin_us, gate=gate, gate_half_us=float(half), gate_auto_us=float(auto),
        delay_ref_us=ref, noise=noise,
        snr_db=float(np.median(out_sliding.level_db[out_sliding.valid]))
        if out_sliding.valid.any() else math.nan,
        gdd_phase_us_per_mhz=float(-curve[0] / (np.pi * scale)) if curve is not None
        else math.nan,
        gdd_trace_us_per_mhz=float(slope[0]) if slope is not None else math.nan,
        stepped=out_stepped, sliding=out_sliding)


def default_band(mode: Mode, width_mhz: float = 1.0) -> tuple[float, float]:
    """A stretch of ``width_mhz`` about the middle of the mode's band, inside it."""
    width = min(width_mhz, mode.f_hi - mode.f_lo)
    lo = min(max((mode.f_lo + mode.f_hi - width) / 2.0, mode.f_lo), mode.f_hi - width)
    return round(lo, 6), round(lo + width, 6)


def format_response(r: Response) -> str:
    gate = "held at one delay" if r.gate.held else "on the curve"
    count = int(r.sliding.valid.sum()), int(r.stepped.valid.sum())
    return "\n".join([
        f"{r.mode} over {r.f_lo_mhz:.2f}-{r.f_hi_mhz:.2f} MHz ({r.source})",
        f"  subchannel {r.subchannel_hz / 1e3:.2f} kHz: a window of {r.window} "
        f"samples, delay bin {r.bin_us:.1f} us",
        f"  overlap: a window every {r.hop} samples, {count[0]} sliding samples "
        f"against {count[1]} stepped",
        f"  gate: {gate}, shift {r.gate.shift_us:+.1f} us, half-width "
        f"{r.gate_half_us:.1f} us (automatic {r.gate_auto_us:.1f} us)",
        f"  level: median {r.snr_db:.1f} dB above the noise",
        f"  GDD: {r.gdd_phase_us_per_mhz:.1f} us/MHz from the phase, "
        f"{r.gdd_trace_us_per_mhz:.1f} us/MHz from the peaks",
        f"  phase: sliding {r.sliding.jumps} jumps in {r.sliding.steps} steps; stepped "
        f"{r.stepped.jumps} in {r.stepped.steps}, {int(r.stepped.lost.sum())} samples "
        f"a turn off"])


def write_csv(r: Response, path) -> Path:
    """Both series, one row per window: the stepped first, then the sliding."""
    path = Path(path)
    lines = ["series,freq_mhz,gate_us,delay_us,level_db,phase_rad,lost"]
    for name, s in (("stepped", r.stepped), ("sliding", r.sliding)):
        for i in np.nonzero(s.valid)[0]:
            lines.append(f"{name},{s.freq_mhz[i]:.6f},{s.gate_us[i]:.2f},"
                         f"{s.delay_us[i]:.3f},{s.level_db[i]:.3f},"
                         f"{s.phase_rad[i]:.5f},{int(s.lost[i])}")
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path
