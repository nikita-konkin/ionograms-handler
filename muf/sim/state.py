"""The channel state: everything the simulator is told about the channel.

A state describes one circuit at one moment as a list of propagation
**modes** -- 1F, 2F, a high ray, an extraordinary wave -- each one a curve
over frequency rather than a number:

* ``delay_us(f)``: the group delay, which is what a chirp sounder measures
  (the ionogram's trace is this curve, scaled to range);
* ``cn0_dbhz(f)``: the mode's strength as carrier-to-noise density, so that
  it means the same at any FFT window and any signal bandwidth;
* a Doppler shift and spread, a fading law, a delay spread within the mode,
  and an optional magneto-ionic split.

Keeping the whole curve is the point. A synthetic sounding samples it across
the sweep; a communication signal at carrier ``fc`` takes its Taylor
expansion there -- delay, GDD = d(tau)/df, TOD = d2(tau)/df2 -- so both
describe the same channel and each can check the other.

Every value carries a :class:`Provenance`: measured off a sounding, derived
from measured values through a law, a prior from the literature, set by a
scenario, or forecast. A result built on a state can then say which of its
inputs were measured and which were assumed.

Units, chosen to be the ones the numbers are quoted in: MHz, microseconds,
dB-Hz, Hz. GDD is microseconds per MHz (1 us/MHz = 1 ps/Hz); TOD is
microseconds per MHz squared.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SCHEMA = "muf.sim.state"
VERSION = 1

# --- provenance ---------------------------------------------------------------

#: Read off a sounding.
MEASURED = "measured"
#: Computed from measured values through a model -- GDD from a fitted law.
DERIVED = "derived"
#: A default from the literature or from other data -- ITU-R F.1487, or
#: February's statistics applied to an August sounding.
PRIOR = "prior"
#: Set by a scenario, with no data behind it.
MODEL = "model"
#: Predicted for a later time.
FORECAST = "forecast"
ORIGINS = (MEASURED, DERIVED, PRIOR, MODEL, FORECAST)

#: The mode fields a provenance is recorded for.
MODE_FIELDS = ("delay", "cn0", "band", "doppler_shift", "doppler_spread",
               "fading", "delay_spread", "magnetoionic")

#: A delay spread's Gaussian profile is cut this many sigma either side.
DELAY_SPREAD_SIGMAS = 4.0


@dataclass(frozen=True)
class Provenance:
    """Where one value came from, and how."""

    origin: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"origin {self.origin!r} is not one of {ORIGINS}")

    def __str__(self) -> str:
        return f"{self.origin}: {self.note}" if self.note else self.origin

    def to_dict(self) -> dict:
        return {"origin": self.origin, "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping | str) -> Provenance:
        if isinstance(data, str):
            origin, _, note = data.partition(":")
            return cls(origin.strip(), note.strip())
        return cls(str(data["origin"]), str(data.get("note", "")))


def _provenance_map(data: Mapping | None) -> dict[str, Provenance]:
    out = {}
    for key, value in (data or {}).items():
        out[str(key)] = (value if isinstance(value, Provenance)
                         else Provenance.from_dict(value))
    return out


def _optional(value) -> float | None:
    return None if value is None else float(value)


def _metric(value):
    """A metric as JSON: numbers as floats, tables of them kept as tables."""
    if isinstance(value, Mapping):
        return {str(k): _metric(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric(v) for v in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    return float(value)


# --- one mode -----------------------------------------------------------------

@dataclass
class Mode:
    """One propagation mode, over the band it exists in.

    ``freq_mhz`` are the knots, strictly ascending; the mode exists from the
    first to the last and nowhere else -- the last is its MUF when the mode is
    the low ray that reaches it. ``delay_us`` and ``cn0_dbhz`` are the values
    at the knots. Between knots the delay is a monotone cubic (PCHIP), so GDD
    is continuous and never overshoots the data; ``cn0`` is linear in dB.

    ``doppler_spread_hz`` is ITU-R F.1487's *frequency spread*: 2 sigma of a
    Gaussian Doppler power spectrum. Zero means a non-fading mode.
    ``doppler_shape`` is ``"gaussian"``, as F.1487 has it, or
    ``"lorentzian"``, Vogler and Hoffmeyer's exponential autocorrelation, for
    which the spread is the full width at half power
    (:mod:`muf.sim.fading`).
    ``rician_k_db`` adds a specular part K dB above the scattered one; None is
    pure Rayleigh, as F.1487's Gaussian-scatter model has it.

    ``delay_spread_us`` spreads the mode in delay about its curve: the RMS of
    a Gaussian power-delay profile centred on it, realised as paths every few
    microseconds that fade independently -- the wide-sense stationary,
    uncorrelated-scattering channel, with the delay spread of Vogler and
    Hoffmeyer's wideband model. Zero is one path, as in F.1487.

    ``ox_split_us`` splits the mode into ordinary and extraordinary waves, the
    x wave that much later, ``x_to_o_db`` stronger or weaker, and shifted by
    ``x_shift_hz`` more; the mode's power is shared between them, since a
    power ionogram measures them together. None leaves the mode whole.
    """

    name: str
    freq_mhz: np.ndarray
    delay_us: np.ndarray
    cn0_dbhz: np.ndarray
    doppler_shift_hz: float = 0.0
    doppler_spread_hz: float = 0.0
    rician_k_db: float | None = None
    ox_split_us: float | None = None
    x_to_o_db: float = 0.0
    x_shift_hz: float = 0.0
    hops: int | None = None
    branch: str | None = None
    doppler_shape: str = "gaussian"
    delay_spread_us: float = 0.0
    provenance: dict[str, Provenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.freq_mhz = np.asarray(self.freq_mhz, dtype=np.float64).ravel()
        self.delay_us = np.asarray(self.delay_us, dtype=np.float64).ravel()
        self.cn0_dbhz = np.broadcast_to(
            np.asarray(self.cn0_dbhz, dtype=np.float64), self.freq_mhz.shape).copy()
        if self.freq_mhz.size < 2:
            raise ValueError(f"mode {self.name}: needs at least two knots")
        if self.delay_us.shape != self.freq_mhz.shape:
            raise ValueError(f"mode {self.name}: {self.delay_us.size} delays "
                             f"for {self.freq_mhz.size} knots")
        if not np.all(np.diff(self.freq_mhz) > 0):
            raise ValueError(f"mode {self.name}: knots must rise strictly")
        for name in ("freq_mhz", "delay_us", "cn0_dbhz"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"mode {self.name}: {name} is not finite")
        if self.doppler_spread_hz < 0:
            raise ValueError(f"mode {self.name}: a negative Doppler spread")
        self.delay_spread_us = float(self.delay_spread_us)
        if not (math.isfinite(self.delay_spread_us) and self.delay_spread_us >= 0):
            raise ValueError(f"mode {self.name}: delay spread {self.delay_spread_us} "
                             f"is not a finite non-negative number")
        if self.doppler_shape not in ("gaussian", "lorentzian"):
            raise ValueError(f"mode {self.name}: Doppler shape "
                             f"{self.doppler_shape!r} is not gaussian or lorentzian")
        self.provenance = _provenance_map(self.provenance)

    # -- where it exists
    @property
    def f_lo(self) -> float:
        return float(self.freq_mhz[0])

    @property
    def f_hi(self) -> float:
        return float(self.freq_mhz[-1])

    def covers(self, f_mhz) -> np.ndarray:
        f = np.asarray(f_mhz, dtype=np.float64)
        return (f >= self.f_lo) & (f <= self.f_hi)

    # -- the curve and its derivatives
    @cached_property
    def _curve(self):
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(self.freq_mhz, self.delay_us, extrapolate=False)

    def delay_at(self, f_mhz) -> np.ndarray:
        """Group delay, us; NaN where the mode does not exist."""
        return self._curve(np.asarray(f_mhz, dtype=np.float64))

    def gdd_at(self, f_mhz) -> np.ndarray:
        """Group-delay dispersion d(tau)/df, us/MHz."""
        return self._curve.derivative(1)(np.asarray(f_mhz, dtype=np.float64))

    def tod_at(self, f_mhz) -> np.ndarray:
        """Third-order dispersion d2(tau)/df2, us/MHz^2.

        The GDD's difference across one knot spacing either side, one-sided
        at the band's edges -- not the interpolant's own second derivative,
        which for PCHIP is discontinuous at every knot and reads a quadratic
        curve 5 % low; the difference reads it to 0.03 %.
        """
        f = np.asarray(f_mhz, dtype=np.float64)
        h = float(np.median(np.diff(self.freq_mhz)))
        lo = np.clip(f - h, self.f_lo, self.f_hi)
        hi = np.clip(f + h, self.f_lo, self.f_hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (self.gdd_at(hi) - self.gdd_at(lo)) / (hi - lo)
        return np.where(self.covers(f), out, np.nan)

    def cn0_at(self, f_mhz) -> np.ndarray:
        """Carrier-to-noise density, dB-Hz; NaN where the mode does not exist."""
        f = np.asarray(f_mhz, dtype=np.float64)
        out = np.interp(f, self.freq_mhz, self.cn0_dbhz)
        return np.where(self.covers(f), out, np.nan)

    def taylor(self, f_mhz: float) -> tuple[float, float, float]:
        """(delay us, GDD us/MHz, TOD us/MHz^2) at one frequency."""
        f = float(f_mhz)
        return (float(self.delay_at(f)), float(self.gdd_at(f)), float(self.tod_at(f)))

    def delay_taps(self, step_us: float) -> list[tuple[float, float, float | None]]:
        """The paths the mode is realised as: (offset us, power share, K dB).

        The Gaussian profile of :attr:`delay_spread_us`, sampled every
        ``step_us`` or half a sigma if that is finer, to
        :data:`DELAY_SPREAD_SIGMAS` either side, the shares summing to one.
        A specular part (:attr:`rician_k_db`) is not spread: it rides the
        path at the curve, whose own K is then the specular power over that
        path's share of the scattered. No spread: the curve alone.
        """
        sigma = self.delay_spread_us
        if sigma == 0.0:
            return [(0.0, 1.0, self.rician_k_db)]
        step = min(float(step_us), sigma / 2.0)
        n = math.ceil(DELAY_SPREAD_SIGMAS * sigma / step)
        offsets = step * np.arange(-n, n + 1, dtype=np.float64)
        shares = np.exp(-0.5 * (offsets / sigma) ** 2)
        shares /= shares.sum()
        if self.rician_k_db is None:
            return [(float(d), float(w), None) for d, w in zip(offsets, shares)]
        k = 10.0 ** (self.rician_k_db / 10.0)
        out = []
        for d, w in zip(offsets, shares):
            if d == 0.0:
                out.append((0.0, float((k + w) / (k + 1.0)),
                            float(10.0 * math.log10(k / w))))
            else:
                out.append((float(d), float(w / (k + 1.0)), None))
        return out

    def origin(self, key: str) -> str:
        """The origin recorded for ``key``, or '' when none was."""
        item = self.provenance.get(key)
        return item.origin if item else ""

    # -- (de)serialisation
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hops": self.hops,
            "branch": self.branch,
            "freq_mhz": self.freq_mhz.tolist(),
            "delay_us": self.delay_us.tolist(),
            "cn0_dbhz": self.cn0_dbhz.tolist(),
            "doppler_shift_hz": float(self.doppler_shift_hz),
            "doppler_spread_hz": float(self.doppler_spread_hz),
            "doppler_shape": self.doppler_shape,
            "rician_k_db": _optional(self.rician_k_db),
            "ox_split_us": _optional(self.ox_split_us),
            "x_to_o_db": float(self.x_to_o_db),
            "x_shift_hz": float(self.x_shift_hz),
            "delay_spread_us": float(self.delay_spread_us),
            "provenance": {k: v.to_dict() for k, v in sorted(self.provenance.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Mode:
        return cls(
            name=str(data["name"]),
            freq_mhz=data["freq_mhz"],
            delay_us=data["delay_us"],
            cn0_dbhz=data["cn0_dbhz"],
            doppler_shift_hz=float(data.get("doppler_shift_hz", 0.0)),
            doppler_spread_hz=float(data.get("doppler_spread_hz", 0.0)),
            doppler_shape=str(data.get("doppler_shape", "gaussian")),
            rician_k_db=data.get("rician_k_db"),
            ox_split_us=data.get("ox_split_us"),
            x_to_o_db=float(data.get("x_to_o_db", 0.0)),
            x_shift_hz=float(data.get("x_shift_hz", 0.0)),
            delay_spread_us=float(data.get("delay_spread_us", 0.0)),
            hops=data.get("hops"),
            branch=data.get("branch"),
            provenance=data.get("provenance") or {},
        )


# --- the circuit and the sweep ------------------------------------------------

EARTH_RADIUS_KM = 6371.0


@dataclass
class Circuit:
    """Transmitter and receiver. The same name at both ends is vertical sounding."""

    tx: str
    rx: str
    tx_lat: float | None = None
    tx_lon: float | None = None
    rx_lat: float | None = None
    rx_lon: float | None = None

    @property
    def vertical(self) -> bool:
        return self.tx == self.rx

    @property
    def ground_km(self) -> float | None:
        """Great-circle distance, or None for vertical sounding or no coordinates."""
        coords = (self.tx_lat, self.tx_lon, self.rx_lat, self.rx_lon)
        if self.vertical or any(c is None for c in coords):
            return None
        p1, p2 = math.radians(self.tx_lat), math.radians(self.rx_lat)
        dp, dl = p2 - p1, math.radians(self.rx_lon - self.tx_lon)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))

    def to_dict(self) -> dict:
        return {"tx": self.tx, "rx": self.rx, "tx_lat": self.tx_lat,
                "tx_lon": self.tx_lon, "rx_lat": self.rx_lat, "rx_lon": self.rx_lon}

    @classmethod
    def from_dict(cls, data: Mapping) -> Circuit:
        return cls(**{k: data.get(k) for k in ("tx", "rx", "tx_lat", "tx_lon",
                                               "rx_lat", "rx_lon")})


@dataclass
class Sweep:
    """The chirp a sounding was made with, or is to be synthesised with.

    ``sample_rate_hz`` is the recorded, decimated rate (40 kHz on cyprus1);
    ``adc_rate_hz`` the rate before decimation that an ``.lfs`` header calls
    ``sample_rate``. The remaining fields only fill an ``.lfs`` header.
    """

    start_mhz: float
    rate_hz_s: float
    duration_s: float
    sample_rate_hz: float
    adc_rate_hz: float = 25e6
    rmin_km: int = 0
    rmax_km: int = 5000
    repetition_s: int = 300
    chirpt: int = 10

    @property
    def stop_mhz(self) -> float:
        return self.start_mhz + self.rate_hz_s * self.duration_s / 1e6

    def freq_at(self, t_s) -> np.ndarray:
        """Sweep frequency, MHz, at time ``t_s`` after the start."""
        return self.start_mhz + self.rate_hz_s * np.asarray(t_s, dtype=np.float64) / 1e6

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "start_mhz", "rate_hz_s", "duration_s", "sample_rate_hz",
            "adc_rate_hz", "rmin_km", "rmax_km", "repetition_s", "chirpt")}

    @classmethod
    def from_dict(cls, data: Mapping) -> Sweep:
        return cls(**dict(data))


# --- the state ----------------------------------------------------------------

@dataclass(frozen=True)
class ModeAt:
    """One mode's Taylor expansion at a frequency."""

    mode: Mode
    delay_us: float
    gdd_us_per_mhz: float
    tod_us_per_mhz2: float
    cn0_dbhz: float


@dataclass
class ChannelState:
    """One circuit at one moment.

    ``epoch`` is unix seconds, UTC. ``sweep`` is needed to synthesise a
    sounding and is otherwise optional. ``metrics`` holds scalar observations
    that are not part of the channel model but travel with it -- the MUF an
    extractor read off the sounding the state came from, for instance.
    """

    circuit: Circuit
    epoch: float
    modes: list[Mode] = field(default_factory=list)
    sweep: Sweep | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provenance = _provenance_map(self.provenance)
        names = [m.name for m in self.modes]
        if len(set(names)) != len(names):
            raise ValueError(f"mode names repeat: {names}")

    @property
    def datetime(self) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(self.epoch, tz=_dt.timezone.utc)

    @property
    def muf_mhz(self) -> float:
        """Highest frequency any mode reaches; NaN with no modes."""
        return max((m.f_hi for m in self.modes), default=float("nan"))

    def mode(self, name: str) -> Mode:
        for m in self.modes:
            if m.name == name:
                return m
        raise KeyError(name)

    def at(self, f_mhz: float) -> list[ModeAt]:
        """The modes present at ``f_mhz``, expanded there, earliest first."""
        out = []
        for m in self.modes:
            if not m.covers(f_mhz):
                continue
            tau, gdd, tod = m.taylor(f_mhz)
            out.append(ModeAt(m, tau, gdd, tod, float(m.cn0_at(f_mhz))))
        return sorted(out, key=lambda item: item.delay_us)

    # -- (de)serialisation
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "source": self.source,
            "epoch": float(self.epoch),
            "utc": self.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "circuit": self.circuit.to_dict(),
            "sweep": None if self.sweep is None else self.sweep.to_dict(),
            "metrics": {k: _metric(v) for k, v in self.metrics.items()},
            "notes": list(self.notes),
            "provenance": {k: v.to_dict() for k, v in sorted(self.provenance.items())},
            "modes": [m.to_dict() for m in self.modes],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> ChannelState:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not a channel state (schema {data.get('schema')!r})")
        if int(data.get("version", 0)) != VERSION:
            raise ValueError(f"channel state version {data.get('version')} -- "
                             f"this reader knows {VERSION}")
        sweep = data.get("sweep")
        return cls(
            circuit=Circuit.from_dict(data["circuit"]),
            epoch=float(data["epoch"]),
            modes=[Mode.from_dict(m) for m in data.get("modes", [])],
            sweep=None if sweep is None else Sweep.from_dict(sweep),
            source=str(data.get("source", "")),
            notes=list(data.get("notes", [])),
            metrics=dict(data.get("metrics", {})),
            provenance=data.get("provenance") or {},
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ChannelState:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def describe(self, fc_mhz: float | None = None) -> str:
        """A few lines for a terminal."""
        lines = [f"{self.circuit.tx} -> {self.circuit.rx}  "
                 f"{self.datetime:%Y-%m-%d %H:%M:%S}Z  ({self.source or 'no source'})"]
        for m in self.modes:
            lines.append(
                f"  {m.name:<10} {m.f_lo:6.2f}-{m.f_hi:6.2f} MHz  "
                f"delay {np.nanmin(m.delay_us):8.1f}-{np.nanmax(m.delay_us):8.1f} us  "
                f"C/N0 {np.max(m.cn0_dbhz):5.1f} dB-Hz  "
                f"spread {m.doppler_spread_hz:.2f} Hz"
                f"{'' if m.doppler_shape == 'gaussian' else ' ' + m.doppler_shape} "
                f"[{m.origin('doppler_spread') or '?'}]"
                + (f"  delay spread {m.delay_spread_us:.1f} us "
                   f"[{m.origin('delay_spread') or '?'}]"
                   if m.delay_spread_us else ""))
        if fc_mhz is not None:
            lines.append(f"  at {fc_mhz:.3f} MHz:")
            present = self.at(fc_mhz)
            for item in present:
                lines.append(
                    f"    {item.mode.name:<10} delay {item.delay_us:9.2f} us  "
                    f"GDD {item.gdd_us_per_mhz:8.2f} us/MHz  "
                    f"TOD {item.tod_us_per_mhz2:8.2f} us/MHz^2  "
                    f"C/N0 {item.cn0_dbhz:5.1f} dB-Hz")
            if not present:
                lines.append("    no mode")
        return "\n".join(lines)


def same_curves(a: Iterable[Mode], b: Iterable[Mode]) -> bool:
    """True when two mode lists carry identical knots -- for tests and checks."""
    a, b = list(a), list(b)
    return len(a) == len(b) and all(
        x.name == y.name and np.array_equal(x.freq_mhz, y.freq_mhz)
        and np.array_equal(x.delay_us, y.delay_us)
        and np.array_equal(x.cn0_dbhz, y.cn0_dbhz) for x, y in zip(a, b))
