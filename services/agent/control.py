"""Start, stop and reconfigure acquisition.

``architecture.md`` sec. 2.5, "Control -- narrow, authenticated, journaled".
Deliberately small: everything here is either a ``systemctl`` verb on one
target or an edit to one ``.ini``. Anything not on that list is a change made
on the station.

Three properties this module exists to guarantee, each of which the station
has already been bitten by:

**Ordered, graceful shutdown.** ``systemctl stop chirp.target`` takes the
units down in dependency order, and the recorder's unit must send ``SIGINT``.
A USRP killed mid-stream keeps transmitting to a host that is gone and wedges:
it stops answering ARP and discovery, and no software on the host can recover
it -- only removing power. That is a site visit, so it is worth a unit file
getting it right.

**No half-applied configuration.** A parameter change is write-then-restart,
and the write is atomic. A truncated ``.ini`` is not a bad setting, it is an
acquisition outage that needs someone to notice and repair the file.

**No combination that records nothing while reporting healthy.** Switching to
scheduled mode without a schedule, or pointing ``output_dir`` at a path that
does not exist, both produce a station whose processes are all "active" and
whose archive stops growing. Those are validated before the write, and
refused whole.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import StationConfig


class ControlError(RuntimeError):
    """A command that could not be carried out, with a reason for the operator."""


@dataclass
class CommandResult:
    """What happened, in a form that can be acknowledged back to the server."""

    command: str
    ok: bool
    detail: str = ""
    timestamp: float = field(default_factory=time.time)
    #: Populated for parameter changes: what to write into ``config_epoch``.
    journal: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {"command": self.command, "ok": self.ok, "detail": self.detail,
                "timestamp": self.timestamp, "journal": self.journal}


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------

#: Verbs the agent will pass to systemctl. An allow-list, not a filter: the
#: command name arrives over the network, and `systemctl` has verbs like
#: `mask` and `isolate` that have no business being reachable remotely.
ALLOWED_VERBS = ("start", "stop", "restart", "status", "is-active")


#: Set ``AGENT_FAKE_SYSTEMCTL=1`` to log verbs instead of running them.
#:
#: For the Docker test rig, where the agent runs in a container that has no
#: systemd -- ``systemctl`` is simply absent, so every control command would
#: report "not found" and the command path could never be demonstrated end to
#: end. Named for what it is, off unless explicitly asked for, and it prints
#: every call so a fake success can never be mistaken for a real one.
FAKE_SYSTEMCTL_ENV = "AGENT_FAKE_SYSTEMCTL"


def _fake_runner(args, **kwargs):
    import sys

    print(f"agent: FAKE systemctl {' '.join(args[1:])} "
          f"(AGENT_FAKE_SYSTEMCTL is set; nothing was actually done)",
          file=sys.stderr)
    return subprocess.CompletedProcess(args, 0, "", "")


def systemctl(verb: str, target: str, *, timeout: float = 120.0,
              runner=None) -> CommandResult:
    """One systemctl verb against one unit or target.

    ``runner`` exists so a test can drive this without systemd; it defaults to
    :func:`subprocess.run` and is never something a caller supplies in
    production.
    """
    if verb not in ALLOWED_VERBS:
        raise ControlError(
            f"verb {verb!r} is not allowed; choose from {', '.join(ALLOWED_VERBS)}")

    # An empty target means this station is not supervised by systemd -- DOB
    # runs `dombas.sh`, which owns the recorder as its own child. Refusing here
    # is not pedantry: `systemctl restart chirp.target` on such a host does not
    # restart what is running, it *starts a second recorder against the same
    # USRP*, and a USRP with two streamers has to be power-cycled by hand. The
    # honest answer to "restart" on a script-run station is "I cannot", said
    # plainly, rather than a systemctl invocation that appears to work.
    if not target.strip():
        return CommandResult(
            verb, False,
            "no systemd target configured for this station (`target` is empty "
            "in the agent config), so there is nothing this agent may act on. "
            "A station whose acquisition is run by a script must be controlled "
            "the same way it is started.")

    if runner is None and os.environ.get(FAKE_SYSTEMCTL_ENV, "").strip() not in ("", "0"):
        runner = _fake_runner
    runner = runner or subprocess.run
    args = ["systemctl", verb, target]
    try:
        proc = runner(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return CommandResult(f"{verb} {target}", False, "systemctl: not found")
    except subprocess.TimeoutExpired:
        # A stop that times out is the dangerous case: systemd will escalate to
        # SIGKILL and the radio can wedge. Say so rather than reporting a bare
        # timeout.
        extra = (" -- the recorder may have been killed rather than stopped; "
                 "check that the USRP still answers"
                 if verb in ("stop", "restart") else "")
        return CommandResult(f"{verb} {target}", False,
                             f"timed out after {timeout}s{extra}")
    except Exception as exc:                                  # pragma: no cover
        return CommandResult(f"{verb} {target}", False,
                             f"{type(exc).__name__}: {exc}")

    text = (proc.stdout or "") + (proc.stderr or "")
    return CommandResult(f"{verb} {target}", proc.returncode == 0, text.strip())


def start(config: StationConfig, **kw) -> CommandResult:
    return systemctl("start", config.target, **kw)


def stop(config: StationConfig, **kw) -> CommandResult:
    return systemctl("stop", config.target, **kw)


def restart(config: StationConfig, **kw) -> CommandResult:
    return systemctl("restart", config.target, **kw)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Settings the agent will change, mapped to their ``.ini`` location. An
#: allow-list for the same reason as `ALLOWED_VERBS`: `chirp_config.py` reads
#: dozens of keys, most of which would need a considered decision and a site
#: visit if they went wrong.
EDITABLE = {
    "mode": ("lfm", "serendipitous"),
    "sounder_timings": ("lfm", "sounder_timings"),
    "output_dir": ("config", "output_dir"),
    "max_range_extent": ("lfm", "max_range_extent"),
    "save_raw_voltage": ("lfm", "save_raw_voltage"),
}

#: Changes that are not a key at all but an operation over several, each with
#: its cross-checks attached. `set_band` takes ``band_start_mhz`` and
#: optionally ``analysis_min_mhz`` / ``analysis_max_mhz``, and writes the six
#: entries of :data:`BAND_INI`. Kept separate from `EDITABLE` so the five
#: coupled keys have no individual door.
COMPOSITE = ("set_band",)

#: What `set_band` accepts. `band_start_mhz` is the hardware; the other two
#: default to the full digitised band.
BAND_ARGS = ("band_start_mhz", "analysis_min_mhz", "analysis_max_mhz")


def _band_kwargs(request: Any) -> dict:
    """Check the shape of a `set_band` request before any of it is believed."""
    if not isinstance(request, dict):
        raise ControlError(
            f"set_band takes an object with {list(BAND_ARGS)}; "
            f"got {type(request).__name__}")
    unknown = set(request) - set(BAND_ARGS)
    if unknown:
        raise ControlError(
            f"set_band: unknown field(s) {sorted(unknown)}; "
            f"allowed: {list(BAND_ARGS)}. The sample rate is deliberately not "
            f"among them -- it has a hardcoded twin at "
            f"rx_uhd_ext_gps.cpp:173 and must divide the N2x0's 100 MHz "
            f"clock, so it stays compiled in.")
    if "band_start_mhz" not in request:
        raise ControlError("set_band: band_start_mhz is required")
    out = {}
    for name in BAND_ARGS:
        if name not in request or request[name] is None:
            continue
        try:
            out[name] = float(request[name])
        except (TypeError, ValueError):
            raise ControlError(
                f"set_band: {name} = {request[name]!r} is not a number") from None
    return out


#: `mode` is friendlier than a bare boolean, and the mapping is the one place
#: the two vocabularies meet. See architecture.md sec. 2.5.
MODES = {"search": "true", "serendipitous": "true",
         "scheduled": "false", "schedule": "false"}

#: Every key `calc_ionograms.py` reads off a `sounder_timings` entry, each with
#: a bare subscript and no default:
#:
#:     rep      = st[s_idx]["rep"]              # line 444
#:     chirpt   = st[s_idx]["chirpt"]           # 445
#:     rate     = st[s_idx]["chirp-rate"]
#:     cid      = st[s_idx]["id"]               # 446
#:     txname   = st[s_idx]["transmit_name"]    # 447
#:
#: The last two are not decoration. They become the product's file name --
#: `lfm_ionogram-<transmit_name>-<station>-<ch>-<id>-<t0>.h5` -- and
#: `ho["txname"]`, which is the only thing downstream has to identify the
#: transmitter with. Every `.ini` in the clone's `examples/marieluise` carries
#: all five; a schedule composed anywhere else must too.
#:
#: This list is duplicated in `services/api/acquisition.py` on purpose. The
#: server's copy refuses a bad schedule while the operator is looking at the
#: screen; this one is the last line, on the station, and must not depend on
#: the server having been updated.
SCHEDULE_KEYS = ("chirp-rate", "rep", "chirpt", "id", "transmit_name")


# --------------------------------------------------------------------------
# The band, as one operation
# --------------------------------------------------------------------------

#: The five ini values a band change writes, plus the flag that makes two of
#: them bind. Not in :data:`EDITABLE`: they are unreachable individually on
#: purpose, because "five keys that must agree" is precisely the shape that
#: failed on 2026-08-19. The only way in is :func:`plan_band`.
BAND_INI = {
    "center_freq": ("config", "center_freq"),
    "minimum_analysis_frequency": ("lfm", "minimum_analysis_frequency"),
    "maximum_analysis_frequency": ("lfm", "maximum_analysis_frequency"),
    "min_freq": ("lfm", "min_freq"),
    "max_freq": ("lfm", "max_freq"),
    "manual_freq_extent": ("lfm", "manual_freq_extent"),
}

#: How much wall time the ringbuffer holds, in seconds -- `B` in sec. 3's
#: `r * B / (1 - r)`. 14 GB of `/dev/shm` at 25 MS/s sc16 (4 bytes/sample,
#: 100 MB/s) is ~140 s. Measured, not derived: the agent does not know the
#: tmpfs size at validation time and this is a warning, not a refusal.
RINGBUFFER_SPAN_S = 140.0

#: What fraction of realtime the analysis actually achieves -- `r`. **0.64,
#: measured 2026-08-18** from `n=1039 lost=87 (8.37%)`. It moves with load,
#: which is why the span check below warns and never refuses.
CONSUMER_REALTIME_FRACTION = 0.64

#: Patch 0014's option string, looked for in the recorder's bytes.
CENTER_FREQ_OPTION = b"center-freq"


@dataclass
class BandPlan:
    """What a band change would write, and what it would cost."""

    changes: dict = field(default_factory=dict)   #: ini key -> value, as text
    warnings: list = field(default_factory=list)  #: shown, never fatal
    notes: list = field(default_factory=list)     #: what was adjusted, and why
    band_start_hz: float = 0.0
    band_stop_hz: float = 0.0
    analysis_min_hz: float = 0.0
    analysis_max_hz: float = 0.0
    sweep_seconds: dict = field(default_factory=dict)  #: transmit_name -> s
    budget_seconds: float = 0.0

    def summary(self) -> str:
        return (f"digitise {self.band_start_hz / 1e6:.3f}-"
                f"{self.band_stop_hz / 1e6:.3f} MHz, analyse "
                f"{self.analysis_min_hz / 1e6:.3f}-"
                f"{self.analysis_max_hz / 1e6:.3f} MHz")


def recorder_reads_the_ini(binary: str | Path | None) -> tuple[bool, str]:
    """Does the deployed recorder take its LO from the command line?

    Reads the file and looks for patch 0014's option string. A byte scan
    rather than an execution, and that distinction is the whole point:
    ``rx_uhd_ext_gps --help`` **opens the radio**, because the option is
    declared without a ``vm.count("help")`` branch to handle it.

    Returns ``(False, reason)`` when it cannot tell. Refusing on "cannot tell"
    is the safe direction: the failure this guards against is silent, produces
    no error anywhere, and costs every sounding until someone thinks to check
    the LO against the ini by hand.
    """
    if binary is None:
        return False, "no recorder_binary configured"
    path = Path(binary)
    try:
        blob = path.read_bytes()
    except OSError as exc:
        return False, f"{path}: {exc.strerror or exc}"
    if CENTER_FREQ_OPTION not in blob:
        return False, (
            f"{path} has no --center-freq option, so it tunes to whatever "
            f"set_rx_freq was compiled with and ignores center_freq in the "
            f"ini. Every product would be dechirped by the difference, with "
            f"no error in any log. Apply patch 0014 and rebuild first.")
    return True, "ok"


def _float_ini(parser: configparser.ConfigParser, section: str, option: str,
               *, default: float | None = None) -> float:
    """chirpsounder2 writes ``25e6`` and reads it with ``json.loads``."""
    raw = parser.get(section, option, fallback=None)
    if raw is None or not str(raw).strip():
        if default is None:
            raise ControlError(
                f"[{section}] {option} is missing from the config and has no "
                f"safe default; a band change cannot be checked without it")
        return default
    try:
        return float(str(raw).strip().strip('"'))
    except ValueError:
        raise ControlError(
            f"[{section}] {option} = {raw!r} is not a number") from None


def _schedule_entries(parser: configparser.ConfigParser) -> list[dict]:
    """Every scheduled transmitter, flattened across ranks."""
    raw = parser.get("lfm", "sounder_timings", fallback=None)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list) or not parsed:
        return []
    ranks = parsed if all(isinstance(x, list) for x in parsed) else [parsed]
    return [e for rank in ranks if isinstance(rank, list)
            for e in rank if isinstance(e, dict)]


def _alignment_quantum_hz(sample_rate: float, block_samples: float,
                          decimation: float, chirp_rate: float) -> float:
    """The frequency grid patch 0013's skip has to land on.

    0013 starts the downconverter at ``minimum_analysis_frequency`` instead of
    at 0 Hz, and it gets there by skipping whole blocks. The skip is
    ``analysis_min / chirp_rate * sample_rate`` samples, and a block is
    ``downconversion_block_samples * decimation`` of them, so an analysis floor
    that is not a whole number of blocks lands mid-block and shifts the
    frequency axis under every product.

    Today's numbers give exactly 750 blocks: 7.5 MHz / 100 kHz/s = 75 s,
    x 25 MS/s = 1.875e9 samples, / (4000 x 625) = 750.
    """
    block = block_samples * decimation
    if block <= 0 or sample_rate <= 0 or chirp_rate <= 0:
        return 0.0
    return block * chirp_rate / sample_rate


def plan_band(parser: configparser.ConfigParser, *,
              band_start_mhz: float,
              analysis_min_mhz: float | None = None,
              analysis_max_mhz: float | None = None,
              recorder_binary: str | Path | None = None,
              check_recorder: bool = True) -> BandPlan:
    """Turn three numbers into the five the ini needs, or refuse with why.

    ``band_start_mhz`` is the hardware: the bottom of what gets digitised.
    ``sample_rate`` stays compiled in (it has the ``sample_rate_numerator``
    twin at ``rx_uhd_ext_gps.cpp:173`` and a 100 MHz clock divisor to respect),
    so the start alone fixes ``center_freq`` and the passband.

    The analysis window defaults to the whole passband and is usually narrower.
    Conflating the two is what sec. 3 cost a day to.
    """
    plan = BandPlan()

    if check_recorder:
        ok, reason = recorder_reads_the_ini(recorder_binary)
        if not ok:
            raise ControlError(f"refusing to change the band: {reason}")

    sample_rate = _float_ini(parser, "config", "sample_rate")
    if sample_rate <= 0:
        raise ControlError(f"sample_rate = {sample_rate} is not usable")

    band_start = float(band_start_mhz) * 1e6
    band_stop = band_start + sample_rate
    centre = band_start + sample_rate / 2.0

    analysis_min = (band_start if analysis_min_mhz is None
                    else float(analysis_min_mhz) * 1e6)
    analysis_max = (band_stop if analysis_max_mhz is None
                    else float(analysis_max_mhz) * 1e6)

    # 5 -- the arithmetic ones first, so later messages can assume a real span.
    if band_start < 0:
        raise ControlError(
            f"band_start {band_start / 1e6:.3f} MHz is below zero")
    if analysis_min < 0:
        raise ControlError(
            f"analysis_min {analysis_min / 1e6:.3f} MHz is below zero")
    if analysis_max <= analysis_min:
        raise ControlError(
            f"analysis_max {analysis_max / 1e6:.3f} MHz is not above "
            f"analysis_min {analysis_min / 1e6:.3f} MHz")

    # 4 -- snap before the containment check, so a snap cannot push the floor
    # out of the band without being caught.
    block_samples = _float_ini(parser, "lfm", "downconversion_block_samples",
                               default=4000.0)
    decimation = _float_ini(parser, "lfm", "decimation", default=625.0)
    entries = _schedule_entries(parser)
    rates = sorted({float(e["chirp-rate"]) for e in entries
                    if str(e.get("chirp-rate", "")).strip() not in ("", "None")}
                   ) if entries else []

    if rates:
        # Snap on the coarsest grid in the schedule, then report any rate the
        # result does not also suit. One transmitter -- the usual case -- makes
        # this exact.
        quanta = [_alignment_quantum_hz(sample_rate, block_samples,
                                        decimation, r) for r in rates]
        coarsest = max(q for q in quanta) if any(quanta) else 0.0
        if coarsest > 0:
            snapped = round(analysis_min / coarsest) * coarsest
            if abs(snapped - analysis_min) > 1e-6:
                plan.notes.append(
                    f"analysis_min snapped {analysis_min / 1e6:.4f} -> "
                    f"{snapped / 1e6:.4f} MHz, onto the "
                    f"{coarsest / 1e3:.3f} kHz block grid; off-grid the "
                    f"downconverter starts mid-block and the frequency axis "
                    f"shifts under every product")
                analysis_min = snapped
            for rate, quantum in zip(rates, quanta):
                if quantum > 0 and abs(
                        analysis_min / quantum
                        - round(analysis_min / quantum)) > 1e-6:
                    plan.warnings.append(
                        f"analysis_min is not a whole number of blocks at "
                        f"{rate / 1e3:.1f} kHz/s ({quantum / 1e3:.3f} kHz "
                        f"grid); that transmitter's frequency axis will be "
                        f"offset. Schedules mixing chirp rates cannot satisfy "
                        f"every grid at once.")

    # 1 -- you cannot analyse spectrum that was never digitised. The refusal
    # `maximum_analysis_frequency = 30e6` should have produced and did not.
    tol = 1.0  # Hz; snapping and float text both land a hair off.
    if analysis_min < band_start - tol or analysis_max > band_stop + tol:
        raise ControlError(
            f"the analysis window {analysis_min / 1e6:.3f}-"
            f"{analysis_max / 1e6:.3f} MHz reaches outside the digitised band "
            f"{band_start / 1e6:.3f}-{band_stop / 1e6:.3f} MHz "
            f"(center_freq {centre / 1e6:.3f} MHz +/- {sample_rate / 2e6:.3f} "
            f"MHz). The part outside is FFTs over spectrum the radio never "
            f"sampled: no error, no signal, and the sweep time spent anyway.")

    # 2 -- the sweep has to finish inside the cycle it belongs to.
    span = analysis_max - analysis_min
    for entry in entries:
        try:
            rate = float(entry["chirp-rate"])
            rep = float(entry["rep"])
        except (KeyError, TypeError, ValueError):
            continue
        if rate <= 0 or rep <= 0:
            continue
        sweep = span / rate
        plan.sweep_seconds[str(entry.get("transmit_name", "?"))] = sweep
        if sweep >= rep:
            raise ControlError(
                f"{entry.get('transmit_name', '?')}: sweeping "
                f"{span / 1e6:.3f} MHz at {rate / 1e3:.1f} kHz/s takes "
                f"{sweep:.1f} s, which does not fit the {rep:.0f} s "
                f"repetition period. The next sounding starts before this one "
                f"is read out.")

    # 3 -- the ringbuffer budget. Warns: `r` is measured and it moves.
    r = CONSUMER_REALTIME_FRACTION
    plan.budget_seconds = r * RINGBUFFER_SPAN_S / (1.0 - r)
    for name, sweep in plan.sweep_seconds.items():
        if sweep > plan.budget_seconds:
            plan.warnings.append(
                f"{name}: a {sweep:.1f} s sweep is past the "
                f"{plan.budget_seconds:.0f} s ringbuffer budget "
                f"(r={r:.2f}, B={RINGBUFFER_SPAN_S:.0f} s), so the oldest "
                f"samples of a sounding can be pruned before the analysis "
                f"reaches them. That is sec. 3's 8.37% loss; it degrades, it "
                f"does not fail.")

    plan.band_start_hz = band_start
    plan.band_stop_hz = band_stop
    plan.analysis_min_hz = analysis_min
    plan.analysis_max_hz = analysis_max

    # `min_freq`/`max_freq` bind only when `manual_freq_extent` is true
    # (`calc_ionograms.py:326`), so writing them without it stores the whole
    # analysed span and quietly ignores the narrowing that was asked for.
    #
    # They are set *equal* to the analysis bounds, and line 327 selects with
    # strict inequalities -- `freqs > min_freq` -- so the two edge bins are
    # dropped and a requested 7.5-32.5 is stored as 7.55-32.30. That is
    # expected, not drift, and it is why phase 3 prints observed beside
    # requested rather than asserting they match.
    plan.changes = {
        "center_freq": _hz(centre),
        "minimum_analysis_frequency": _hz(analysis_min),
        "maximum_analysis_frequency": _hz(analysis_max),
        "min_freq": _hz(analysis_min),
        "max_freq": _hz(analysis_max),
        "manual_freq_extent": "true",
    }
    return plan


def _hz(value: float) -> str:
    """``12.5e6``, the way chirpsounder2's own configs write frequencies.

    Nine significant digits, not `%g`'s six: a floor snapped onto the grid of
    a 500.0084 kHz/s chirp lands on values like 37.50063 MHz, and six digits
    would round the config away from the frequency that was actually checked.
    """
    return f"{value / 1e6:.9g}e6"


def read_config(path: str | Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    read = parser.read(Path(path))
    if not read:
        raise ControlError(f"{path}: could not be read")
    return parser


#: ``Key=`` at the head of a line. In a systemd unit only ``Exec*`` runs
#: anything; in a shell script this matches a variable assignment, which is
#: likewise not a launch.
_DIRECTIVE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)\s*=")


def _logical_lines(text: str):
    """Lines with trailing-backslash continuations joined, comments left alone.

    systemd and shell both continue a line with ``\\``; a comment continues in
    neither, so a ``#`` line that happens to end in one must not swallow what
    follows it.
    """
    buffered = ""
    for line in text.splitlines():
        stripped = line.strip()
        if buffered or not stripped.startswith("#"):
            if stripped.endswith("\\"):
                buffered += stripped[:-1].strip() + " "
                continue
            if buffered:
                yield (buffered + stripped).strip()
                buffered = ""
                continue
        yield stripped
    if buffered:
        yield buffered.strip()


def _launcher_ranks(path: str | Path | None) -> int | None:
    """How many MPI ranks the launcher starts ``calc_ionograms.py`` with.

    ``None`` means "no answer, do not check": the file is unreadable, does not
    mention `calc_ionograms.py`, or -- the good case -- passes a variable to
    ``-np`` because it derives the count from `sounder_timings` itself, which
    is patch 0009 and makes the mismatch this guards against impossible.

    Deliberately a text scan and not an execution. The launcher is a shell
    script that starts a radio; the agent reads it and never runs it.

    Reads a **systemd unit** as readily as a shell script, because after the
    migration off ``dombas.sh`` the unit is what starts the program and is
    therefore what `launcher` should point at. Two things about unit files that
    a script does not have, both of which this got wrong on 2026-08-16:

    * a unit names the program in ``Description=`` *before* it launches it in
      ``ExecStart=``. That first mention carries no ``-np``, so the naive scan
      took the "not under mpirun" branch and answered 1 for a unit that starts
      two ranks -- which then refused a correct two-transmitter schedule with
      a message describing a mismatch that did not exist. Only ``Exec*=``
      lines are launches; every other ``Key=`` line is prose or configuration.
    * ``ExecStart=`` is routinely wrapped across lines with a trailing
      backslash, which would put ``-np`` and the script name in different
      lines. Shell continues the same way, so joining first is right for both.
    """
    if path is None:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    for stripped in _logical_lines(text):
        if "calc_ionograms.py" not in stripped or stripped.startswith("#"):
            continue
        directive = _DIRECTIVE_RE.match(stripped)
        if directive is not None and not directive.group(1).startswith("Exec"):
            continue
        found = re.search(r"-np\s+(\S+)", stripped)
        if found is None:
            # Not under mpirun at all: one process, hence rank 0 only.
            return 1
        count = found.group(1).strip('"\'')
        return int(count) if count.isdigit() else None
    return None


def _validate(parser: configparser.ConfigParser, changes: dict,
              launcher: str | Path | None = None) -> None:
    """Refuse combinations that record nothing while looking healthy."""
    def value_of(key, section, option):
        if key in changes:
            return changes[key]
        return parser.get(section, option, fallback=None)

    mode = value_of("mode", "lfm", "serendipitous")
    if mode is not None and str(mode).lower() in ("false", "scheduled", "schedule"):
        timings = value_of("sounder_timings", "lfm", "sounder_timings")
        try:
            parsed = json.loads(timings) if timings else []
        except json.JSONDecodeError as exc:
            raise ControlError(f"sounder_timings is not valid JSON: {exc}") from None
        if not parsed:
            raise ControlError(
                "scheduled mode with an empty sounder_timings would record "
                "nothing while every process reported healthy. Supply the "
                "schedule in the same command, or stay in search mode.")
        # The ini stores `sounder_timings` **per MPI rank**, so the real shape
        # is a list of lists -- `[[{...}, {...}]]` for one rank -- and that is
        # what `/ui/sources` builds when you tick rows. This validator only
        # understood the flat form and reached `set(entry)` with a list, which
        # raised an unhandled TypeError rather than a ControlError: the
        # operator saw the command fail with "unhashable type: 'dict'" and no
        # indication of what to change. Both shapes are accepted; a flat list
        # is read as a single rank.
        ranks = parsed if all(isinstance(item, list) for item in parsed) else [parsed]
        for rank in ranks:
            if not isinstance(rank, list) or not rank:
                raise ControlError(
                    f"sounder_timings must be a list of entries, or a list of "
                    f"one such list per MPI rank; got {rank!r}")
            for entry in rank:
                if not isinstance(entry, dict):
                    raise ControlError(
                        f"sounder_timings entry {entry!r} is not an object with "
                        f"{', '.join(SCHEDULE_KEYS)}")
                missing = set(SCHEDULE_KEYS) - set(entry)
                if missing:
                    raise ControlError(
                        f"sounder_timings entry {entry} is missing "
                        f"{sorted(missing)}. calc_ionograms.py reads all of "
                        f"{list(SCHEDULE_KEYS)} with a bare subscript, so an "
                        f"entry short of one is a KeyError on that rank at the "
                        f"first slot -- while the other ranks carry on and the "
                        f"log looks normal.")
                if not str(entry["transmit_name"]).strip():
                    raise ControlError(
                        f"sounder_timings entry {entry} has an empty "
                        f"transmit_name; it is written into the product's file "
                        f"name and read back as the transmitter's identity.")

        # `calc_ionograms.py:452` does `st = conf.sounder_timings[rank]` with
        # no guard, so the rank count and the schedule length must agree.
        # Neither way of disagreeing announces itself: too few ranks and the
        # transmitters past the cut are simply never sounded, too many and one
        # rank dies of IndexError while the others carry on. In both cases the
        # log looks normal, because for the surviving ranks it is.
        started = _launcher_ranks(launcher)
        if started is not None and started != len(ranks):
            raise ControlError(
                f"the schedule has {len(ranks)} rank group(s) but the launcher "
                f"starts calc_ionograms.py with -np {started}. "
                + (f"{len(ranks) - started} transmitter(s) would never be "
                   f"sounded, silently."
                   if started < len(ranks) else
                   f"{started - len(ranks)} rank(s) would die of IndexError "
                   f"while the rest carried on looking healthy.")
                + " Match -np to the schedule, or apply patch 0009 so the "
                  "launcher derives it.")

    if "output_dir" in changes:
        target = Path(str(changes["output_dir"]).strip('"'))
        if not target.parent.exists():
            raise ControlError(
                f"{target}: parent directory does not exist. Acquisition would "
                f"start, report healthy, and write nowhere.")

    if "save_raw_voltage" in changes:
        if str(changes["save_raw_voltage"]).lower() not in ("true", "false"):
            raise ControlError("save_raw_voltage must be true or false")


def _atomic_write(path: Path, parser: configparser.ConfigParser) -> None:
    """Write via a temporary file in the same directory, then replace.

    Same directory so the replace is on one filesystem and therefore atomic.
    A half-written config is an outage, not a misconfiguration.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name,
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            parser.write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply_config(config: StationConfig, changes: dict, *,
                 backup: bool = True) -> CommandResult:
    """Validate, back up, write atomically, and describe what to journal.

    Does **not** restart: :func:`apply_and_restart` does, so that a caller who
    wants to batch several changes into one restart can. v2 reads its config
    only at process start, so nothing takes effect until something restarts.
    """
    unknown = set(changes) - set(EDITABLE) - set(COMPOSITE)
    if unknown:
        raise ControlError(
            f"not editable remotely: {sorted(unknown)}; "
            f"allowed: {sorted(set(EDITABLE) | set(COMPOSITE))}")
    if not changes:
        raise ControlError("no changes given")

    path = Path(config.chirp_config)
    parser = read_config(path)

    normalized = dict(changes)
    band_request = normalized.pop("set_band", None)
    if "mode" in normalized:
        raw = str(normalized["mode"]).lower()
        if raw not in MODES:
            raise ControlError(
                f"mode {normalized['mode']!r} unknown; "
                f"choose from {sorted(set(MODES))}")
        normalized["mode"] = MODES[raw]

    _validate(parser, normalized, launcher=getattr(config, "launcher", None))

    before = {}
    locations = {key: EDITABLE[key] for key in normalized}
    values = dict(normalized)

    # The simple keys go onto the in-memory parser first so that a band change
    # sent in the same command validates against the schedule that command is
    # installing, not the one on disk. Nothing has touched the file yet.
    for key, value in normalized.items():
        section, option = EDITABLE[key]
        if not parser.has_section(section):
            parser.add_section(section)
        before[key] = parser.get(section, option, fallback=None)
        parser.set(section, option, str(value))

    plan = None
    if band_request is not None:
        plan = plan_band(parser, recorder_binary=getattr(
            config, "recorder_binary", None), **_band_kwargs(band_request))
        for key, value in plan.changes.items():
            section, option = BAND_INI[key]
            if not parser.has_section(section):
                parser.add_section(section)
            before[key] = parser.get(section, option, fallback=None)
            parser.set(section, option, str(value))
            locations[key] = BAND_INI[key]
            values[key] = value

    if backup:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        shutil.copy2(path, path.with_suffix(path.suffix + f".{stamp}.bak"))

    _atomic_write(path, parser)

    detail = "; ".join(f"{k}: {before[k]!r} -> {v!r}"
                       for k, v in values.items())
    journal = {
        "station": config.station,
        "config_path": str(path),
        "changes": {k: {"from": before[k], "to": str(v)}
                    for k, v in values.items()},
        "applied_at": time.time(),
        "requires_restart": True,
    }
    if plan is not None:
        journal["band"] = {
            "summary": plan.summary(),
            "band_start_mhz": plan.band_start_hz / 1e6,
            "band_stop_mhz": plan.band_stop_hz / 1e6,
            "analysis_min_mhz": plan.analysis_min_hz / 1e6,
            "analysis_max_mhz": plan.analysis_max_hz / 1e6,
            "sweep_seconds": plan.sweep_seconds,
            "budget_seconds": plan.budget_seconds,
            "warnings": plan.warnings,
            "notes": plan.notes,
        }
        extra = plan.notes + plan.warnings
        if extra:
            detail += " | " + " | ".join(extra)

    return CommandResult(command="apply_config", ok=True, detail=detail,
                         journal=journal)


def apply_and_restart(config: StationConfig, changes: dict, **kw) -> list[CommandResult]:
    """The whole sequence, so a half-applied change cannot exist.

    Stop, write, start -- rather than restart-after-write -- because the
    config is read at process start and stopping first makes the window in
    which the old processes could write products under the new configuration
    exactly zero.
    """
    results = [stop(config, **kw)]
    if not results[0].ok:
        results.append(CommandResult(
            "apply_config", False,
            "not attempted: the stop failed, and editing the config of a "
            "running acquisition would take effect at an unpredictable moment"))
        return results

    try:
        results.append(apply_config(config, changes))
    except ControlError as exc:
        results.append(CommandResult("apply_config", False, str(exc)))
        # Bring it back up on the old config rather than leaving it down.
        results.append(start(config, **kw))
        return results

    results.append(start(config, **kw))
    return results
