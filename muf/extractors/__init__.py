"""MUF estimators.

Each estimator takes the same :class:`~muf.spectro.Ionogram` and returns the
same :class:`MufResult`, so the pipeline can run all of them over one
spectrogram and the results can be compared directly.

Estimators differ only in how they decide *which cells belong to the trace*;
they all hand a per-frequency presence array to :func:`muf.pick.pick_muf` for
the final decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..pick import MufPick
from ..spectro import Ionogram


@dataclass
class MufResult:
    """One estimator's answer for one sounding."""

    method: str
    pick: MufPick
    presence: np.ndarray = field(repr=False)   # bool per frequency bin
    mask: np.ndarray | None = field(default=None, repr=False)  # bool [n_freq, n_range]
    error: str | None = None

    @property
    def muf_mhz(self) -> float:
        return self.pick.muf_mhz

    @property
    def vrange_km(self) -> float:
        return self.pick.vrange_km

    @property
    def ok(self) -> bool:
        return self.error is None and self.pick.ok

    def __str__(self) -> str:
        if self.error:
            return f"{self.method}: failed ({self.error})"
        if not self.pick.ok:
            return f"{self.method}: no detection ({self.pick.n_detections} bins)"
        return (
            f"{self.method}: MUF {self.pick.muf_mhz:.2f} MHz "
            f"@ {self.pick.vrange_km:.0f} km "
            f"(run {self.pick.run_len}, {self.pick.snr_db:.0f} dB)"
        )


class Extractor(Protocol):
    def __call__(self, ion: Ionogram, **kwargs: object) -> MufResult: ...


#: Names that used to identify an estimator, and what they are now. ``thresh``
#: was a misnomer: it uses the same 43 dB threshold as ``algo``, so the
#: threshold was never what set it apart -- the contour analysis is. Old
#: commands and old scripts keep working; old result *tables* keep their
#: ``muf_thresh`` columns, since renaming those retroactively would make two
#: runs of the same data disagree about what they measured.
ALIASES = {"thresh": "contour"}


def _registry() -> dict[str, Callable[..., MufResult]]:
    from .algorithmic import extract as algorithmic
    from .contour import extract as contour
    from .kmeans import extract as kmeans
    from .viterbi import extract as viterbi

    reg: dict[str, Callable[..., MufResult]] = {
        "algo": algorithmic,
        "kmeans": kmeans,
        "contour": contour,
        "dp": viterbi,
    }

    # TensorFlow is optional; importing it costs seconds and it is absent on
    # plenty of machines, so the CNN only joins the registry when asked for.
    try:
        from .cnn import extract as cnn
    except Exception:  # pragma: no cover - depends on local install
        pass
    else:
        reg["cnn"] = cnn
    return reg


#: Estimators run when ``--methods`` is not given. Two are excluded. The CNN
#: needs a model trained on this geometry (see cnn.py). ``dp`` is new on
#: 2026-08-30 and no stored result was produced with it, so making it a default
#: would silently change what a re-run of an old archive means; it has to earn
#: its way in against the others first. See
#: ``docs/2026-08-30-segmentation-quality.md`` sec. 6a.
DEFAULT_METHODS = ("algo", "kmeans", "contour")

ALL_METHODS = ("algo", "kmeans", "contour", "dp", "cnn")


def canonical(name: str) -> str:
    """Resolve a possibly-retired estimator name to its current one."""
    return ALIASES.get(name, name)


def get(name: str) -> Callable[..., MufResult]:
    name = canonical(name)
    reg = _registry()
    if name not in reg:
        if name in ALL_METHODS:
            raise KeyError(
                f"method '{name}' is unavailable -- its optional dependency is "
                f"not installed (see requirements.txt)"
            )
        raise KeyError(f"unknown method '{name}'; available: {', '.join(sorted(reg))}")
    return reg[name]


def available() -> tuple[str, ...]:
    return tuple(sorted(_registry()))


def run(ion: Ionogram, methods=DEFAULT_METHODS, **kwargs) -> dict[str, MufResult]:
    """Run several estimators over one ionogram.

    A failure in one estimator is recorded in its own ``MufResult.error`` rather
    than aborting the others.
    """
    out: dict[str, MufResult] = {}
    for name in methods:
        opts = kwargs.get(name, {}) if isinstance(kwargs.get(name), dict) else {}
        try:
            out[name] = get(name)(ion, **opts)
        except Exception as exc:
            out[name] = MufResult(
                method=name,
                pick=MufPick(np.nan, np.nan, 0, 0, np.nan, -1),
                presence=np.zeros(ion.shape[0], dtype=bool),
                error=f"{type(exc).__name__}: {exc}",
            )
    return out


__all__ = [
    "MufResult", "Extractor", "get", "available", "run", "canonical",
    "DEFAULT_METHODS", "ALL_METHODS", "ALIASES",
]
