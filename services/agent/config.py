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
from dataclasses import dataclass, field
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
        data = json.loads(Path(path).read_text(encoding="utf-8"))
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
        """``AGENT_CONFIG`` points at a JSON file; otherwise the defaults."""
        path = os.environ.get("AGENT_CONFIG")
        return cls.from_json(path) if path else cls()

    def as_dict(self) -> dict:
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            out[name] = str(value) if isinstance(value, Path) else value
        out["token"] = "***" if self.token else ""
        return out
