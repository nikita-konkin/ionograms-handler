"""What the agent needs to know about the station it runs on.

Kept separate from the collectors so that "where is the data volume" is
answered in one place rather than rediscovered by each metric, and so a test
can describe a whole station without a station.

The agent's own settings are deliberately *not* v2's ``.ini``. That file
belongs to chirpsounder2 and the agent edits it under instruction
(``architecture.md`` sec. 2.5); mixing our settings into it would put our code
in the pinned clone's config, which is the same mistake as putting our code in
the clone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

#: The units of ``chirp.target``, in start order. Stop order is the reverse:
#: the recorder comes up first and goes down last, because its consumers read
#: the ringbuffer it fills.
DEFAULT_UNITS = (
    "chirp-rx.service",
    "chirp-detect.service",
    "chirp-timings.service",
    "chirp-ionograms.service",
    "chirp-metadata.service",
)

#: Units that are watched but whose absence is not a failure, matched as a
#: substring of the unit name.
#:
#: The digisonde receivers are the case this exists for. Each is an oblique
#: reception of a *remote vertical* sounder -- a free extra circuit -- and the
#: station acquires its own path perfectly well with all four stopped. They
#: are also the ones that stop: a template instance whose ini section is
#: missing exits at startup, and Chilton has no feed at all. Reporting four
#: reds for four circuits nobody promised is the "eleven false reds" failure
#: the station config was written to avoid, one migration later.
#:
#: Not the same as leaving them out of ``units``: an optional unit still
#: reports its state, so "Dourbes has been down since Tuesday" is on the page
#: for whoever wants it. It just cannot make the station unhealthy.
DEFAULT_OPTIONAL_UNITS = ("chirp-digisonde@",)

#: Systemd target the control commands act on. One target rather than five
#: units, so ordering is systemd's problem and not the agent's.
DEFAULT_TARGET = "chirp.target"


@dataclass(frozen=True)
class StationConfig:
    """Everything the agent needs that is not discoverable at runtime."""

    #: Station code as it appears in products and in the registry.
    station: str = "DOB"

    #: chirpsounder2's config file. The agent reads it for expected values and
    #: edits it on a parameter change.
    chirp_config: Path = Path("/home/ionouser/chirpsounder2/my_station.ini")

    #: The launcher script, read-only and only to cross-check ``-np`` against
    #: the schedule -- a mismatch silently stops sounding a transmitter, and
    #: the ini alone cannot show it. Never edited: this agent changes
    #: parameters, not how the station is started. A path that does not exist,
    #: or a launcher that derives ``-np`` at runtime, simply disables the
    #: check rather than failing the command.
    #: **The installed unit, not the repo's copy and not dombas.sh.** systemd
    #: reads `/etc/systemd/system/`, so that is the file whose `-np` decides
    #: how many ranks actually start. Pointed at `dombas.sh` this check found
    #: no `calc_ionograms.py` line at all, answered None, and disabled itself
    #: -- which is how a three-transmitter schedule was accepted against
    #: `-np 1` on 2026-08-20 and two transmitters went unsounded for twelve
    #: hours. A guard aimed at a file that no longer launches anything reads
    #: exactly like a guard that passed.
    launcher: Path = Path("/etc/systemd/system/chirp-ionograms.service")

    #: The recorder binary, read-only and only to answer one question: does it
    #: take the LO from the ini, or is it still the build with ``set_rx_freq``
    #: compiled in? A band change against the latter tunes nothing and
    #: dechirps everything by the difference -- which is exactly how this
    #: station went blind twice on 2026-08-19, with every unit green.
    #:
    #: Answered by looking for ``center-freq`` in the file's bytes, patch
    #: 0014's option string. **Never by running it with ``--help``**: the
    #: option is declared in ``add_options`` but there is no
    #: ``vm.count("help")`` branch, so the program falls through to
    #: ``multi_usrp::make`` and opens the radio. ``--help`` starts a recorder,
    #: against a USRP the live one already owns.
    recorder_binary: Path = Path("/home/ionouser/chirpsounder2/rx_uhd_ext_gps")

    #: Where products land. Read from ``chirp_config`` when present; this is
    #: the fallback and the thing disk-free is measured against.
    #:
    #: The boot SSD rather than DATA3: the latter is USB NTFS behind a
    #: ``mount.ntfs`` FUSE daemon that competes with the recorder for CPU. The
    #: archive-sync and archive-prune units must name this same path.
    output_dir: Path = Path("/home/ionouser/ionozond_data2")

    #: The DigitalRF ringbuffer, usually tmpfs. Separate from ``output_dir``
    #: because filling it is a different failure with a different remedy.
    ringbuffer_dir: Path = Path("/dev/shm/hf25")

    units: tuple[str, ...] = DEFAULT_UNITS
    #: Of those, the ones allowed to be inactive. See
    #: :data:`DEFAULT_OPTIONAL_UNITS`; set it to ``[]`` to have every listed
    #: unit count.
    optional_units: tuple[str, ...] = DEFAULT_OPTIONAL_UNITS
    target: str = DEFAULT_TARGET

    #: Where health goes. Push, per sec. 5.4 -- the station never listens.
    server_url: str = ""
    #: Shared secret for the push and the command pull. Absent means the agent
    #: reports to stdout and takes no commands, which is the right behaviour
    #: for a first run and for a test.
    token: str = ""

    #: Send a thumbnail of the newest product per transmitter alongside the
    #: health push. Off makes the station report ages and nothing else, which
    #: is what it did before; see :mod:`services.agent.preview` for what one
    #: costs (2.5 ms and ~3 KB for an ordinary DOB product).
    preview: bool = True
    #: ``(frequencies, ranges)``. Wider than the console shows would only send
    #: pixels the browser throws away.
    preview_size: tuple[int, int] = (128, 96)

    push_interval_s: float = 60.0
    #: How long after boot to suppress alerts. A station coming up looks
    #: exactly like a station that has died.
    startup_grace_s: float = 300.0

    #: Expected acquisition parameters, for the "silent misconfiguration"
    #: metrics. None disables the check rather than inventing an expectation.
    expected_sample_rate: float | None = 25e6

    #: A transmitter whose position and published transmit seconds are known,
    #: used to measure the receiver's epoch offset. This is the check that
    #: would have caught the 0.956 s clock error on the first day instead of
    #: the second; see ``muf.io_detect.solve_epoch_offset``.
    reference_tx: dict = field(default_factory=lambda: {
        "name": "cyprus1",
        "rate": 100e3,
        "transmit_seconds": [235, 240, 245, 300],
        "distance_km": 3436.0,
        "cycle_s": 300.0,
        # Wide enough to admit an epoch error past a whole second, which is
        # exactly the case worth catching.
        "window_s": 2.0,
    })

    @classmethod
    def from_json(cls, path: str | Path) -> "StationConfig":
        """Read the station config, and fail legibly when it is not there.

        Both errors below are re-raised with the path and the environment
        variable that chose it. The agent runs under ``Restart=always``, so a
        config mistake is not a crash you read once -- it repeats every 30
        seconds, and a bare ``FileNotFoundError`` several frames deep buys the
        reader nothing on the twentieth repetition.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(
                "%s: no such file. This is the station config, named by "
                "AGENT_CONFIG. It belongs outside the repository -- normally "
                "~/agent.json -- so that `git pull` cannot overwrite it and it "
                "cannot be committed by accident. Copy "
                "deploy/station-dob.json.example there and edit server_url."
                % path) from None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("%s: not valid JSON (%s)" % (path, exc)) from None
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "StationConfig":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        for key in ("chirp_config", "launcher", "output_dir", "ringbuffer_dir"):
            if key in known:
                known[key] = Path(known[key])
        for key in ("units", "optional_units", "preview_size"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)

    @classmethod
    def from_env(cls) -> "StationConfig":
        """``AGENT_CONFIG`` points at a JSON file; otherwise the defaults.

        ``AGENT_TOKEN`` overrides the file's ``token``, and is how the compose
        rig supplies it. The shared secret is the one field that must not live
        in the config file: ``deploy/station-sim.json`` is committed, so the
        obvious instruction -- paste ``CONTROL_TOKEN`` in as ``token`` -- puts
        a live credential in a tracked file one ``git add`` away from being
        published. Everything else in here is configuration; this is a
        credential, and it arrives the way the api's own ``CONTROL_TOKEN``
        does.

        An unset or empty ``AGENT_TOKEN`` means "not specified" and leaves the
        file's value alone, rather than clearing it. Absent in both places
        still means the agent reports to stdout and takes no commands.
        """
        path = os.environ.get("AGENT_CONFIG")
        config = cls.from_json(path) if path else cls()
        token = os.environ.get("AGENT_TOKEN", "")
        return replace(config, token=token) if token else config

    def as_dict(self) -> dict:
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            out[name] = str(value) if isinstance(value, Path) else value
        out["token"] = "***" if self.token else ""
        return out
