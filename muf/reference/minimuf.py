"""MINIMUF -- deliberately not implemented.

MINIMUF-3.5 (Rose & Martin, NOSC TD 201, 1978; DTIC ADA066256) is a compact
empirical MUF algorithm, and a good reference to have: it is published, widely
used, and its error statistics are documented. The inherited
``MUF_plot_day_by_day.py:10`` and ``MUF_spectre.py:10`` both imported a
``mini_muf`` module that was never in this repository; those two files were
themselves removed in ``f94f561``, so nothing here reaches for it any more.

**Why there is no implementation here.** The authoritative coefficients could
not be obtained. The DTIC scan's OCR degrades exactly the lines that matter --
the solar-zenith terms and the sunspot scaling come through as fragments like
``K8»3.82\\*H0+12^O.13`` and ``G2«(1\\*S9s250>``, where neither the operators nor
the constants can be read with confidence.

Writing it from memory would produce a model that looks like MINIMUF, runs, and
returns plausible numbers that are quietly wrong. That is precisely the class of
defect this pipeline was built to remove -- an inverted range axis, a longitude
read from the latitude's offset, a stretched frequency axis: all of them
survived for years because their output stayed plausible. A fabricated reference
would be worse, because its whole purpose is to be trusted.

**To enable it**, obtain the coefficients from a source you can verify:

* the report itself -- DTIC ADA066256, or NOSC TD 201 via a library;
* a published implementation whose provenance you can check (MINIMUF appeared in
  several 1980s amateur-radio packages, and in ``ITS/VOACAP`` documentation);
* the successor, MINIMUF-85 (Sailors & Sprague), which corrected the
  sunspot-number dependence.

Then implement :func:`predict` to the signature the other references use, and
add it to the registry in ``muf/reference/__init__.py``.

Meanwhile :mod:`muf.reference.giro` gives real measurements and
:mod:`muf.reference.iri` a standard model -- both stronger references than
MINIMUF, which is itself an approximation fitted to ionosonde data.
"""

from __future__ import annotations

from ..geometry import Point
from . import ReferenceSeries

NOT_IMPLEMENTED_REASON = (
    "MINIMUF is not implemented: the authoritative coefficients (Rose & Martin, "
    "NOSC TD 201 / DTIC ADA066256) could not be verified, and implementing it "
    "from memory would produce a plausible-but-wrong reference. See "
    "muf/reference/minimuf.py for what is needed to enable it. Use `giro` for "
    "measurements or `iri` for a standard model."
)


def predict(tx: Point, rx: Point, times, **_) -> ReferenceSeries:
    """Always unavailable, with the reason attached."""
    return ReferenceSeries(name="minimuf", error=NOT_IMPLEMENTED_REASON)
