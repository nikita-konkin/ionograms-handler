"""An archived product read in the live view's grid, gated to its path like live."""

from __future__ import annotations

import random

import numpy as np

from muf import io_chirp
from services.agent import product


def stored(make_chirp_h5, tx, rx="yoshkar-ola"):
    return make_chirp_h5(np.random.exponential(1.0, (random.randint(20, 60), 8192)),
                         txname=tx, station_name=rx, max_range_extent_km=8000.0)


def test_a_product_on_a_known_path_starts_where_its_path_allows(make_chirp_h5):
    head, _ = product.read(stored(make_chirp_h5, random.choice(["NIC1", "NIC2"])))
    assert head["range_lo_km"] > 1500.0, (
        "a product on a known path was not gated to the range its path allows")


def test_a_product_on_a_known_path_keeps_its_own_range_step(make_chirp_h5):
    path = stored(make_chirp_h5, "NIC1")
    step = float(np.median(np.abs(np.diff(io_chirp.load(path).vrange))))
    assert product.read(path)[0]["range_step_km"] == step, (
        "a gated product was folded although its gate fits the live rows")


def test_a_product_on_an_unknown_path_keeps_its_whole_range(make_chirp_h5):
    path = stored(make_chirp_h5, "synth%d" % random.randint(0, 999))
    lowest = float(np.min(io_chirp.load(path).vrange))
    assert product.read(path)[0]["range_lo_km"] == lowest, (
        "a product on an unknown path lost range it had no reason to lose")
