"""Station coordinate registry.

``architecture.md`` sec. 2.3 item 3. A chirpsounder2 product carries ``txname``
and ``station_name`` as bare strings -- ``NIC``, ``DOB``, ``unkown`` -- and no
coordinates at all, so without a table ``geometry.path_of`` has nothing to work
with and ``calibrate.default_gate`` falls back to the stored range extent.
``.lfs`` files are unaffected: their 512-byte header carries lat/lon directly,
which is why this did not exist before v2.

**The numbers are copied, not read from the clone.** chirpsounder2 ships a
table at ``examples/marieluise/server.ini`` under ``[stations] station_info``,
and 16 of the entries below come from it. The clone is pinned and disposable
(sec. 2.2), so depending on it at runtime would make our geometry a property of
whichever commit is checked out. Every entry records where its numbers came
from in :attr:`Station.source`, because the one thing this table cannot do is
be silently wrong: a coordinate error does not raise, it produces a path length
that is plausible and a virtual height that is not.

**Nothing here is inferred from measurements.** The temptation is real -- this
archive contains cyprus1 on four slots and could be inverted for a position --
but a range measured through the ionosphere carries a virtual-height excess of
a few percent and, on an uncalibrated receiver, an epoch error of any size at
all (see ``io_detect``). A registry that quietly absorbed those would be a
record of the receiver's faults.

**Two names, one site.** ``cyprus1`` (what the ``.lfs`` archive calls it) is an
alias of ``NIC`` (what v2 calls it), resolving to v2's five-decimal position.
That identification is an operator judgement recorded on 2026-08-05, not
something the data established -- see :data:`CYPRUS1_LFS_COORDINATES` for what
was weighed. It is the one entry here whose position was chosen rather than
copied, which is why it is documented at length and the alternative is kept in
the module rather than deleted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping

#: Provenance tags for :attr:`Station.source`.
V2 = "chirpsounder2 server.ini"
LFS = ".lfs header"
LOCAL = "local"


@dataclass(frozen=True)
class Station:
    """One site. ``code`` is what a product's ``txname``/``station_name`` holds."""

    code: str
    name: str
    latitude: float
    longitude: float
    source: str
    aliases: tuple[str, ...] = ()
    note: str = ""

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)

    def __str__(self) -> str:
        return (f"{self.code:14} {self.latitude:10.5f} {self.longitude:11.5f}  "
                f"{self.name} [{self.source}]")


#: The 16 sites chirpsounder2 carries, transcribed verbatim from
#: ``examples/marieluise/server.ini`` at commit ``0d2712553063``. Do not
#: "tidy" the decimals: several are 14 significant figures because they were
#: computed rather than surveyed, and rounding them would silently move a
#: station by metres to tens of metres for no gain.
_V2_STATIONS = (
    Station("SGO", "Sodankyla Geophysical Observatory",
            67.36369337350563, 26.634311805059543, V2),
    Station("TGO", "Tromso Geophysical Observatory",
            69.66174439007057, 18.939127366530286, V2),
    Station("DOB", "Dombas", 62.073, 9.111, V2),
    Station("KHO", "Kjell Henriksen Observatory", 78.1478, 16.04035, V2),
    Station("W2NAF", "W2NAF", 41.33509166666666, -75.60079, V2),
    # `cyprus1` is the name the .lfs archive gives this transmitter and `NIC`
    # is v2's. Treating them as one site is an operator judgement (2026-08-05),
    # not a measurement -- see CYPRUS1_LFS_COORDINATES for what was given up
    # and why the data could not decide it.
    Station("NIC", "Nicosia, Cyprus", 35.18557, 33.38228, V2,
            aliases=("cyprus1",),
            note="also the .lfs archive's 'cyprus1'; its header says "
                 "35.0/34.0, 59.9 km away, superseded by these five decimals"),
    Station("Ramfjordmoen", "Ramfjordmoen",
            69.58187184247221, 19.220853348827067, V2),
    Station("ROTHR1", "ROTHR Chesapeake Bay",
            37.56561111111111, -76.26387222222223, V2),
    Station("ROTHR2", "ROTHR Culebra", 18.00889, -66.50611, V2),
    Station("ROTHR3", "ROTHR Texas",
            27.527361111111112, -98.71644444444443, V2),
    Station("ROTHR", "ROTHR", 36.11793762278912, -82.5807086169964, V2),
    Station("JORN", "Jindalee Operational Radar Network",
            -23.853424758715892, 125.07620821000198, V2),
    Station("Juliusruh", "Juliusruh",
            54.628740389301186, 13.374299986524846, V2),
    Station("TH76", "Thule", 76.54, -68.44, V2),
    Station("DB049", "Dourbes", 50.1, 4.6, V2),
    Station("Chilton", "Chilton", 51.56932026174434, -1.32, V2),
)

#: Sites this pipeline knows and v2 does not, from ``.lfs`` file headers --
#: the only record of them that exists.
_LOCAL_STATIONS = (
    Station("yoshkar-ola", "Yoshkar-Ola", 56.38, 47.53, LFS,
            note="the .lfs receiver; absent from v2's table"),
)

#: What the ``.lfs`` archive's header gives for ``cyprus1``, superseded on
#: 2026-08-05 by ``NIC``'s five-decimal position, which ``cyprus1`` is now an
#: alias for. Kept because the number is still embedded in every ``.lfs`` file
#: and someone will eventually notice the disagreement.
#:
#: The archive could not settle it. After removing the DOB receiver's 0.9557 s
#: epoch error, cyprus1's four slots gave 3398, 3420, 3422 and 3504 km of
#: *virtual* range -- a 106 km spread, against the 40 km that separates the two
#: candidate positions on that path, and on top of a few percent of
#: ionospheric excess that is not modelled. The decision was made on the shape
#: of the numbers instead: 35.0/34.0 is round to a degree, 35.18557/33.38228 is
#: not.
CYPRUS1_LFS_COORDINATES = (35.0, 34.0)

#: How much the choice moves each path. It is not symmetric: the ``.lfs``
#: baseline barely notices, and DOB moves by 20 range bins at 2 km resolution.
#:
#:     cyprus1 -> yoshkar-ola   2588.4 -> 2587.8 km   (-0.6 km)
#:     cyprus1 -> DOB           3476.3 -> 3435.9 km   (-40.3 km)
#:
#: **The ``.lfs`` path is unaffected in practice**, and not only because 0.6 km
#: is nothing: ``io_lfs`` reads coordinates from each file's own 512-byte
#: header and never consults this registry. A ``.lfs`` sounding of cyprus1 will
#: keep reporting 35.0/34.0 and a 2588.4 km path, which is what
#: ``signal-chain.md`` records as measured. This table governs v2 products,
#: where the file carries a name and nothing else.

#: v2's marker for a transmitter it could not identify
#: (``calc_ionograms.py:189``, upstream's typo). It must never resolve to
#: coordinates -- every unidentified emitter in an archive shares this string,
#: so a match would give them all the same position.
UNIDENTIFIED = "unkown"


class Registry(Mapping):
    """Name to ``(lat, lon)``, case-insensitively, with aliases.

    Implements the ``Mapping[str, tuple[float, float]]`` that
    :func:`muf.io_chirp.read_header` takes, so it can be passed straight
    through. Lookup folds case and strips whitespace because the two formats
    disagree about both: v2 writes ``DOB`` and ``NIC``, ``.lfs`` writes
    ``cyprus1`` and ``yoshkar-ola``, and a station operator editing a config
    by hand writes whatever they like.
    """

    def __init__(self, stations: Iterable[Station] = ()):
        self._by_code: dict[str, Station] = {}
        for station in stations:
            self.add(station)

    @staticmethod
    def _key(name: str) -> str:
        return str(name).strip().lower()

    def add(self, station: Station, replace: bool = True) -> None:
        key = self._key(station.code)
        if key in self._by_code and not replace:
            return
        self._by_code[key] = station
        for alias in station.aliases:
            self._by_code.setdefault(self._key(alias), station)

    def station(self, name: str) -> Station | None:
        """The full record, or None. Never raises on an unknown name."""
        if not name or self._key(name) == self._key(UNIDENTIFIED):
            return None
        return self._by_code.get(self._key(name))

    def __getitem__(self, name: str) -> tuple[float, float]:
        station = self.station(name)
        if station is None:
            raise KeyError(name)
        return station.coordinates

    def __iter__(self) -> Iterator[str]:
        # Deduplicated: aliases point at the same Station and would otherwise
        # be yielded as separate sites.
        seen = set()
        for station in self._by_code.values():
            if station.code not in seen:
                seen.add(station.code)
                yield station.code

    def __len__(self) -> int:
        return len({s.code for s in self._by_code.values()})

    def stations(self) -> list[Station]:
        return sorted({s.code: s for s in self._by_code.values()}.values(),
                      key=lambda s: s.code.lower())

    def merged_with(self, other: "Registry | Iterable[Station]") -> "Registry":
        """A new registry where ``other`` wins every collision.

        Collisions are the point: a station's own config is more authoritative
        about its own coordinates than a table copied out of somebody else's
        example file.
        """
        out = Registry(self.stations())
        for station in (other.stations() if isinstance(other, Registry) else other):
            out.add(station, replace=True)
        return out

    def __repr__(self) -> str:
        return f"Registry({len(self)} stations)"


def default_registry() -> Registry:
    """The built-in table: v2's 16 sites plus the two only this pipeline knows."""
    return Registry(_V2_STATIONS + _LOCAL_STATIONS)


def from_json(path: str | Path) -> Registry:
    """Read a registry from JSON: ``{code: {name, lat, lon, aliases, note}}``.

    ``lat``/``lon`` may also be spelled ``latitude``/``longitude``. Anything
    else in an entry is ignored rather than rejected, so a file carrying extra
    operational fields stays usable here.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Registry(_station_from(code, entry, f"{path.name}")
                    for code, entry in data.items())


def from_server_ini(path: str | Path) -> Registry:
    """Read chirpsounder2's own ``[stations] station_info`` JSON blob.

    For pointing at a station's live ``server.ini`` rather than transcribing
    it. The built-in table already holds this file's contents as of commit
    ``0d2712553063``; this exists for a deployment whose copy has diverged.
    """
    import configparser

    path = Path(path)
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_option("stations", "station_info"):
        raise ValueError(f"{path}: no [stations] station_info")
    data = json.loads(parser.get("stations", "station_info"))
    return Registry(_station_from(code, entry, f"{path.name}")
                    for code, entry in data.items())


def _station_from(code: str, entry: Mapping, source: str) -> Station:
    lat = entry.get("lat", entry.get("latitude"))
    lon = entry.get("lon", entry.get("longitude"))
    if lat is None or lon is None:
        raise ValueError(f"station {code!r} in {source} has no lat/lon: {dict(entry)}")
    return Station(
        code=str(code),
        name=str(entry.get("name", code)),
        latitude=float(lat),
        longitude=float(lon),
        source=source,
        aliases=tuple(str(a) for a in entry.get("aliases", ())),
        note=str(entry.get("note", "")),
    )


def describe(registry: Registry | None = None) -> str:
    """The table, for ``muf stations`` and for a config-check in a log."""
    registry = registry or default_registry()
    lines = [f"{len(registry)} stations",
             f"{'code':>14} {'latitude':>10} {'longitude':>11}  name [source]"]
    for station in registry.stations():
        lines.append(f"{station}")
        if station.note:
            lines.append(f"{'':14} note: {station.note}")
    return "\n".join(lines)
