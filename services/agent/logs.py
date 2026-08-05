"""Reading the station's logs, bounded.

``architecture.md`` sec. 2.5, "Logs -- the third thing the agent exists for".
Health says *that* something is wrong; logs say *what*. Every failure this
station has had was diagnosed by reading process output -- dropped-packet
markers, ``Unable to set the thread priority``, ``ref_locked: false``, ``no
DigitalRF data bounds available`` -- and no metric would have carried any of
them.

**Everything here is bounded.** A log endpoint that can return a gigabyte is a
denial of service against your own station, and the recorder in particular
emits a line per dropped packet: the 2026-08-05 logs ran to thousands of
``got no data in recv 0`` in minutes. Line counts are clamped, byte counts are
clamped, and the reader says when it truncated rather than silently returning
a prefix.

**Counts belong in health, text belongs here.** "Dropped packets in the last
five minutes" is a metric that should trend; the surrounding text is what you
fetch once the metric moves. Streaming every line to the server continuously
would recreate the bulk-transfer problem of sec. 5.1.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable

from .config import StationConfig

#: Hard ceiling on lines returned, whatever is asked for.
MAX_LINES = 2000

#: Hard ceiling on bytes returned, because a single line can be long.
MAX_BYTES = 512_000

#: Patterns worth surfacing without being asked, learned from this station's
#: actual failures. Each is a thing that is invisible in every metric and
#: obvious in one line of output.
KNOWN_SIGNATURES = (
    (re.compile(r"Unable to set the thread priority"),
     "receive thread is not real-time; expect continuous packet loss "
     "(LimitRTPRIO in the unit, or setcap on the binary)"),
    (re.compile(r"got no data in recv"),
     "host fell behind the USRP; raise the NIC RX ring and check rtprio"),
    (re.compile(r"ref_locked.*false|One or more devices not locked"),
     "USRP is not locked to its reference; range and frequency both drift"),
    (re.compile(r"No UHD Devices Found|No devices found"),
     "the radio did not answer -- it may be wedged streaming and need power "
     "removed; no software path recovers that state"),
    (re.compile(r"no DigitalRF data bounds available"),
     "the recorder is not writing; everything downstream is idle by "
     "consequence, not by fault"),
    (re.compile(r"gps_locked:\s*false"),
     "GPSDO has no satellite lock; the epoch is free-running"),
)


@dataclass
class LogChunk:
    """Lines from one unit, and whether anything was left out."""

    unit: str
    lines: list[str] = field(default_factory=list)
    truncated: bool = False
    error: str = ""
    #: Signatures matched in these lines, as ``(pattern description, count)``.
    signatures: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"unit": self.unit, "lines": self.lines,
                "truncated": self.truncated, "error": self.error,
                "signatures": [{"meaning": m, "count": c} for m, c in self.signatures]}


def scan_signatures(lines: Iterable[str]) -> list[tuple[str, int]]:
    """Which known failure signatures appear, and how often.

    The count matters as much as the presence: the recorder drops packets
    occasionally under normal load, and pathologically when misconfigured.
    One ``got no data in recv`` is noise; four hundred is a diagnosis.
    """
    counts: dict[str, int] = {}
    for line in lines:
        for pattern, meaning in KNOWN_SIGNATURES:
            if pattern.search(line):
                counts[meaning] = counts.get(meaning, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def read_unit(unit: str, *, lines: int = 200, since: str | None = None,
              priority: str | None = None, grep: str | None = None,
              timeout: float = 15.0, runner=None) -> LogChunk:
    """Recent journal for one unit.

    ``since`` takes systemd's own vocabulary (``"10 min ago"``, ``"today"``),
    ``priority`` its severity names (``"warning"``, ``"err"``). Filtering
    happens in ``journalctl`` rather than here so the volume is never carried
    into this process in the first place.
    """
    n = max(1, min(int(lines), MAX_LINES))
    args = ["journalctl", "-u", unit, "-n", str(n), "--no-pager",
            "--output", "short-iso"]
    if since:
        args += ["--since", since]
    if priority:
        args += ["--priority", priority]
    if grep:
        args += ["--grep", grep]

    runner = runner or subprocess.run
    try:
        proc = runner(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return LogChunk(unit, error="journalctl: not found")
    except subprocess.TimeoutExpired:
        return LogChunk(unit, error=f"journalctl: timed out after {timeout}s")
    except Exception as exc:                                  # pragma: no cover
        return LogChunk(unit, error=f"{type(exc).__name__}: {exc}")

    if proc.returncode != 0:
        return LogChunk(unit, error=(proc.stderr or "").strip() or
                        f"journalctl exited {proc.returncode}")

    text = proc.stdout or ""
    truncated = False
    if len(text.encode("utf-8", "replace")) > MAX_BYTES:
        text = text.encode("utf-8", "replace")[-MAX_BYTES:].decode("utf-8", "replace")
        truncated = True
    out = [ln for ln in text.splitlines() if ln.strip()]
    if len(out) > n:
        out = out[-n:]
        truncated = True

    return LogChunk(unit, lines=out, truncated=truncated,
                    signatures=scan_signatures(out))


def read_all(config: StationConfig, **kw) -> list[LogChunk]:
    """Every unit of ``chirp.target``, same bounds applied to each."""
    return [read_unit(unit, **kw) for unit in config.units]


def triage(config: StationConfig, *, lines: int = 500, **kw) -> dict:
    """Signatures across every unit, without the text.

    What to fetch first when a station goes unhealthy: it answers "which of
    the known failures is this" in one round trip, and only then do you pull
    the lines from the unit that matched.
    """
    found: dict[str, int] = {}
    errors: dict[str, str] = {}
    for chunk in read_all(config, lines=lines, **kw):
        if chunk.error:
            errors[chunk.unit] = chunk.error
        for meaning, count in chunk.signatures:
            found[meaning] = found.get(meaning, 0) + count
    return {
        "station": config.station,
        "signatures": [{"meaning": m, "count": c}
                       for m, c in sorted(found.items(), key=lambda kv: -kv[1])],
        "errors": errors,
    }
