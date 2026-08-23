"""Independent references to compare the extracted MUF against.

The three estimators in :mod:`muf.extractors` share a spectrogram, a range gate,
a threshold and a picker. When they agree that shows consistency, not accuracy --
a bias common to all three would not show up. These references come from outside
the pipeline entirely.

Availability differs, so each degrades with a clear reason rather than failing
the run:

``giro``
    Real ionosonde measurements from the station nearest the path's control
    point. The strongest reference: an independent instrument measuring the same
    ionosphere. Needs the network and a nearby station.

``iri``
    The International Reference Ionosphere, the standard empirical model. Needs
    an optional package (``PyIRI`` or ``iri2016``).

``chapman``
    A transparent solar-zenith model, computed here with every constant stated.
    Checks the *shape* of the diurnal variation, not its absolute level -- its
    amplitude is fitted, so it cannot corroborate a scale error.

``minimuf``
    Not implemented. See :mod:`muf.reference.minimuf` for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..geometry import Point


@dataclass
class ReferenceSeries:
    """MUF predicted or measured by something outside the pipeline."""

    name: str
    muf: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    detail: pd.DataFrame | None = None      # foF2, hmF2, ... when available
    source: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.muf) > 0

    def __str__(self) -> str:
        if self.error:
            return f"{self.name}: unavailable ({self.error})"
        if not len(self.muf):
            return f"{self.name}: no data"
        return (f"{self.name}: {len(self.muf)} points, "
                f"{self.muf.min():.2f}-{self.muf.max():.2f} MHz  [{self.source}]")


def _registry() -> dict[str, Callable[..., ReferenceSeries]]:
    from .chapman import predict as chapman
    from .giro import predict as giro
    from .iri import predict as iri
    from .minimuf import predict as minimuf

    return {"giro": giro, "iri": iri, "chapman": chapman, "minimuf": minimuf}


ALL_REFERENCES = ("giro", "iri", "chapman", "minimuf")


def get(name: str) -> Callable[..., ReferenceSeries]:
    reg = _registry()
    if name not in reg:
        raise KeyError(
            f"unknown reference {name!r}; available: {', '.join(sorted(reg))}"
        )
    return reg[name]


def run(
    names,
    tx: Point,
    rx: Point,
    times,
    **kwargs,
) -> dict[str, ReferenceSeries]:
    """Evaluate several references over the same timestamps.

    A failure in one is recorded on its own series rather than aborting.
    """
    out: dict[str, ReferenceSeries] = {}
    for name in names:
        try:
            out[name] = get(name)(tx=tx, rx=rx, times=times,
                                  **kwargs.get(name, {}))
        except Exception as exc:
            out[name] = ReferenceSeries(
                name=name, error=f"{type(exc).__name__}: {exc}"
            )
    return out


def as_index(times) -> pd.DatetimeIndex:
    """Normalise assorted time inputs to a naive UTC DatetimeIndex."""
    index = pd.DatetimeIndex(pd.to_datetime(list(times), utc=True))
    return index.tz_localize(None)


__all__ = ["ReferenceSeries", "get", "run", "as_index", "ALL_REFERENCES"]
