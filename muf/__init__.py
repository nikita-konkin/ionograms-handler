"""MUF extraction pipeline for oblique-incidence chirp ionosonde data.

Reads ``.lfs`` IQ recordings, forms a range-gated ionogram once per sounding,
and runs several independent MUF estimators over that same array so their
results can be compared directly.

See ``README`` for the physics and the CLI.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
