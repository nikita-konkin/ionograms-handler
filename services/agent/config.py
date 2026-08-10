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

    #: Where products land. Read from ``chirp_config`` when present; this is
    #: the fallback and the thing disk-free is measured against.
    output_dir: Path = Path("/media/ionouser/DATA3/ionozond_data2")

    #: The DigitalRF ringbuffer, usually tmpfs. Separate from ``output_dir``
    #: because filling it is a different failure with a different remedy.
    ringbuffer_dir: Path = Path("/dev/shm/hf25")

    units: tuple[str, ...] = DEFAULT_UNITS
    target: str = DEFAULT_TARGET

    #: Where health goes. Push, per sec. 5.4 -- the station never listens.
    server_url: str = ""
    #: Shared secret for the push and the command pull. Absent means the agent
    #: reports to stdout and takes no commands, which is the right behaviour
    #: for a first run and for a test.
    token: str = ""

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
        for key in ("chirp_config", "output_dir", "ringbuffer_dir"):
            if key in known:
                known[key] = Path(known[key])
        if "units" in known:
            known["units"] = tuple(known["units"])
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
