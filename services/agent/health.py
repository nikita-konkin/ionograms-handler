"""Health metrics for one station, collected into an atomic JSON document.

``architecture.md`` sec. 2.5, "Health -- read-only, ship first". The disk-full
failure mode is what this exists to prevent, and the rest of the list is what
this particular station has actually been broken by.

Two rules shape every collector here:

**A collector never raises.** A station reporting nine metrics and one error
is far more useful than a station reporting nothing because ``systemctl`` was
missing. Each returns its own failure as a value, and :func:`collect` always
produces a document.

**A metric that cannot be measured is ``None``, not zero.** "Zero soundings in
the last hour" and "could not tell how many soundings" are different
situations with different responses, and a monitoring system that conflates
them pages you for the wrong thing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import StationConfig

#: Anything older than this and the station is not producing, whatever the
#: processes say. Three sounding periods at DOB's 300 s cycle.
STALE_PRODUCT_S = 900.0

#: Ringbuffer occupancy past this is the hour-before-failure signal. Observed
#: at 94 % on 2026-08-05 with `ringbuffer_max_age_min` too high.
RINGBUFFER_WARN_FRACTION = 0.85

#: Free space below this on the data volume and the station has days, not
#: weeks. `save_raw_voltage` turns days into hours -- see sec. 3.4.
DISK_WARN_FRACTION = 0.10

#: Scatter across the reference transmitter's slots, past which the epoch
#: solve is not a measurement. A sound one at DOB agreed to 0.08 ms across
#: four slots eleven hours apart, and the phase drifts 0.19 ms/hour, so 10 ms
#: is generous. Exceeding it means the slots disagree -- an archive holding
#: two eras, or the wrong transmitter named -- and the honest report is
#: "unknown", not a number.
EPOCH_SCATTER_LIMIT_S = 10e-3


@dataclass
class Metric:
    """One measurement, its verdict, and why it could not be taken."""

    name: str
    value: Any = None
    ok: bool | None = None          # None: unknown, not "fine"
    detail: str = ""

    @classmethod
    def unknown(cls, name: str, why: str) -> "Metric":
        return cls(name=name, value=None, ok=None, detail=why)


def _run(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """Run a command, never raise. Returns ``(returncode, output)``."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
        return proc.returncode, (proc.stdout or proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, f"{args[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{args[0]}: timed out after {timeout}s"
    except Exception as exc:                                  # pragma: no cover
        return 1, f"{args[0]}: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Collectors
# --------------------------------------------------------------------------

def unit_states(config: StationConfig) -> list[Metric]:
    """Is each process of ``chirp.target`` running.

    ``systemctl is-active`` rather than a process-table scan: the units are
    the supervision boundary, and a bare ``pgrep`` cannot tell a service that
    exited cleanly from one that was never started.
    """
    out = []
    for unit in config.units:
        code, text = _run(["systemctl", "is-active", unit])
        if text in ("", "unknown") or code == 127:
            out.append(Metric.unknown(f"unit:{unit}", text or "no systemctl"))
        else:
            out.append(Metric(f"unit:{unit}", text, ok=(text == "active")))
    return out


def newest_product_age(config: StationConfig) -> Metric:
    """Seconds since the newest product file. Soundings stopping is not the
    same as a process dying, and this is the metric that separates them."""
    root = Path(config.output_dir)
    if not root.is_dir():
        return Metric.unknown("newest_product_age_s", f"{root}: no such directory")
    newest = None
    try:
        for path in root.rglob("lfm_ionogram-*.h5"):
            mtime = path.stat().st_mtime
            if newest is None or mtime > newest:
                newest = mtime
    except OSError as exc:
        return Metric.unknown("newest_product_age_s", f"{type(exc).__name__}: {exc}")
    if newest is None:
        return Metric("newest_product_age_s", None, ok=False,
                      detail="no products under the output directory at all")
    age = time.time() - newest
    return Metric("newest_product_age_s", round(age, 1), ok=(age < STALE_PRODUCT_S),
                  detail=f"threshold {STALE_PRODUCT_S:.0f}s")


def disk_free(config: StationConfig) -> list[Metric]:
    """Free space on the data volume and on the ringbuffer.

    Two separate failures. A full data volume stops the archive; a full
    ringbuffer stops acquisition within seconds and is usually a
    `ringbuffer_max_age_min` set too high rather than a disk problem.
    """
    out = []
    for name, path, warn in (
        ("disk_free_fraction", Path(config.output_dir), DISK_WARN_FRACTION),
        ("ringbuffer_free_fraction", Path(config.ringbuffer_dir),
         1.0 - RINGBUFFER_WARN_FRACTION),
    ):
        if not path.exists():
            out.append(Metric.unknown(name, f"{path}: no such path"))
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            out.append(Metric.unknown(name, f"{type(exc).__name__}: {exc}"))
            continue
        fraction = usage.free / usage.total if usage.total else 0.0
        out.append(Metric(name, round(fraction, 4), ok=(fraction > warn),
                          detail=f"{usage.free / 1e9:.1f} GB free of "
                                 f"{usage.total / 1e9:.1f} GB"))
    return out


def sample_rate(config: StationConfig) -> Metric:
    """Configured sample rate against what is expected.

    A silent misconfiguration: nothing fails, the products are simply on a
    different range scale than every other day in the archive.
    """
    if config.expected_sample_rate is None:
        return Metric.unknown("sample_rate_hz", "no expectation configured")
    import configparser

    path = Path(config.chirp_config)
    if not path.is_file():
        return Metric.unknown("sample_rate_hz", f"{path}: no such file")
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
        value = float(json.loads(parser.get("config", "sample_rate")))
    except Exception as exc:
        return Metric.unknown("sample_rate_hz", f"{type(exc).__name__}: {exc}")
    ok = abs(value - config.expected_sample_rate) < 1.0
    return Metric("sample_rate_hz", value, ok=ok,
                  detail=f"expected {config.expected_sample_rate:.0f}")


def uptime_s() -> Metric:
    """Seconds since boot, for the startup grace period."""
    try:
        with open("/proc/uptime", "r", encoding="ascii") as fh:
            return Metric("uptime_s", round(float(fh.read().split()[0]), 1), ok=True)
    except Exception as exc:
        return Metric.unknown("uptime_s", f"{type(exc).__name__}: {exc}")


def epoch_offset(config: StationConfig, max_age_s: float = 6 * 3600.0) -> Metric:
    """Receiver clock against a transmitter of known position and schedule.

    The metric this station earned. On 2026-08-05 its epoch was 0.956 s out;
    every product looked perfect -- stable to 0.5 ms, self-consistent
    schedule, plausible ionograms -- and every range was nonsense. Nothing
    internal to the files could reveal it, and nothing did for two days.

    Reported in seconds. Anything past a millisecond is 300 km of range error;
    past half a second the transmit *second* is wrong too and identification
    fails along with ranging.
    """
    spec = config.reference_tx or {}
    if not spec:
        return Metric.unknown("epoch_offset_s", "no reference transmitter configured")
    try:
        from muf import io_detect
    except Exception as exc:
        return Metric.unknown("epoch_offset_s", f"muf unavailable: {exc}")

    root = Path(config.output_dir)
    if not root.is_dir():
        return Metric.unknown("epoch_offset_s", f"{root}: no such directory")

    cutoff = time.time() - max_age_s
    try:
        recent = [p for p in root.rglob("par-*.h5") if p.stat().st_mtime > cutoff]
    except OSError as exc:
        return Metric.unknown("epoch_offset_s", f"{type(exc).__name__}: {exc}")
    if not recent:
        return Metric.unknown(
            "epoch_offset_s",
            f"no timing solutions in the last {max_age_s / 3600:.0f} h")

    try:
        solutions = []
        for path in recent:
            try:
                solutions.append(io_detect.read_timing(path))
            except Exception:
                continue
        offset = io_detect.solve_epoch_offset(
            solutions,
            rate=float(spec["rate"]),
            transmit_seconds=tuple(spec["transmit_seconds"]),
            distance_km=float(spec["distance_km"]),
            cycle_s=float(spec.get("cycle_s", 300.0)),
            reference=str(spec.get("name", "reference")),
            window_s=float(spec.get("window_s", 1.5)),
        )
    except ValueError as exc:
        # Not an error: the reference simply was not transmitting, or was not
        # heard. Saying "unknown" is right; saying "0.0" would be a lie.
        return Metric.unknown("epoch_offset_s", str(exc))
    except Exception as exc:                                  # pragma: no cover
        return Metric.unknown("epoch_offset_s", f"{type(exc).__name__}: {exc}")

    if offset.residual_sd_s > EPOCH_SCATTER_LIMIT_S:
        return Metric.unknown(
            "epoch_offset_s",
            f"slots disagree by +/-{offset.residual_sd_s * 1e3:.0f} ms "
            f"({offset.range_uncertainty_km:.0f} km) across {offset.n_slots} "
            f"slots -- not one transmitter on one clock. Either the archive "
            f"spans a clock change, or {spec.get('name', 'the reference')} is "
            f"not what is being heard.")

    # Split into the two failures it actually causes. Reporting the raw
    # product of offset and c gives 286,440 km for the real DOB fault -- a
    # number larger than the planet, which tells a human nothing. Whole
    # seconds break transmitter *identification*; the remainder breaks range.
    whole = round(offset.seconds)
    remainder = offset.seconds - whole
    parts = []
    if whole:
        parts.append(f"{whole:+d} whole second(s) -- transmit seconds are "
                     f"misidentified")
    parts.append(f"{remainder * 1e3:+.1f} ms = {abs(remainder) * 299792.458:.0f} km "
                 f"of range error")

    return Metric(
        "epoch_offset_s", round(offset.seconds, 6),
        ok=abs(offset.seconds) < 1e-3,
        detail=(f"{spec.get('name', 'reference')}, {offset.n_slots} slots, "
                f"{offset.n_samples} samples, +/-{offset.residual_sd_s * 1e3:.2f} ms "
                f"({offset.range_uncertainty_km:.0f} km); " + "; ".join(parts)),
    )


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

@dataclass
class HealthReport:
    station: str
    timestamp: float
    metrics: list[Metric] = field(default_factory=list)
    agent_version: str = "0.1.0"

    @property
    def in_startup_grace(self) -> bool:
        up = self.metric("uptime_s")
        return bool(up and up.value is not None and up.value < self._grace)

    def metric(self, name: str) -> Metric | None:
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    @property
    def failing(self) -> list[Metric]:
        return [m for m in self.metrics if m.ok is False]

    @property
    def unknown(self) -> list[Metric]:
        return [m for m in self.metrics if m.ok is None]

    @property
    def healthy(self) -> bool:
        """False only on a *definite* failure.

        An unknown metric is not a failure -- it is a gap in observation, and
        treating the two alike means a missing `systemctl` pages someone at
        03:00 for a station that is fine.
        """
        return not self.failing

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps({
            "station": self.station,
            "timestamp": self.timestamp,
            "agent_version": self.agent_version,
            "healthy": self.healthy,
            "metrics": [asdict(m) for m in self.metrics],
        }, indent=indent, sort_keys=False)

    _grace: float = 300.0


def collect(config: StationConfig | None = None, *,
            include_epoch: bool = True) -> HealthReport:
    """Every metric, in one document. Never raises."""
    config = config or StationConfig()
    metrics: list[Metric] = []
    metrics.extend(unit_states(config))
    metrics.append(newest_product_age(config))
    metrics.extend(disk_free(config))
    metrics.append(sample_rate(config))
    metrics.append(uptime_s())
    if include_epoch:
        metrics.append(epoch_offset(config))

    report = HealthReport(station=config.station, timestamp=time.time(),
                          metrics=metrics)
    report._grace = config.startup_grace_s
    return report
