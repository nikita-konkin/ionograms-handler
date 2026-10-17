"""One sounding as the live view's history draws it: an SNR and a PDP strip.

Shared by the server, which summarises a sounding from the live columns it
received, and the agent, which summarises the products it archived before
anyone was watching (issue #12). One computation, so the two halves of a
strip agree where they meet.

``cells`` are one byte per frequency step and range row, a quarter of a dB per
count above the noise floor, as the live columns carry them. The SNR strip is
the strongest cell of each frequency step; the PDP strip is the mean power of
each range row over the steps, back in dB.
"""

from __future__ import annotations

import base64

import numpy as np

#: dB per count, as in the live columns.
DB_STEP = 0.25


def counts(db: np.ndarray) -> np.ndarray:
    """dB above the noise floor as one byte each, clipped to what a byte carries."""
    return np.clip(np.round(np.nan_to_num(db, nan=0.0) / DB_STEP), 0,
                   255).astype(np.uint8)


def snr(cells: np.ndarray) -> str:
    return pack(cells.max(axis=1))


def pdp(cells: np.ndarray) -> str:
    power = np.mean(10.0 ** (cells * DB_STEP / 10.0), axis=0)
    return pack(10.0 * np.log10(power) / DB_STEP)


def pack(values: np.ndarray) -> str:
    cells = np.clip(np.round(values), 0, 255).astype(np.uint8)
    return base64.b64encode(cells.tobytes()).decode("ascii")
