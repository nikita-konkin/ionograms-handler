"""The station sweep that cuts each chirpsounder2 product's echo window."""

from __future__ import annotations

import hashlib
import math
import os
import random
import time

import numpy as np
from conftest import ECHO_RANGE_KM, ECHO_T0

from muf import echo_window, io_chirp, loader
from services.agent import echo


def product(factory, *, echo_present=True, age_s=3600.0):
    """A v2 product carrying `z`, aged `age_s`, with or without an echo."""
    n = int(10 * 40_000.0)
    rng = np.random.default_rng(random.randint(0, 10_000))
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    if echo_present:
        beat = (3100.0 - ECHO_RANGE_KM) / (-io_chirp.C_M_S / 100e3 / 1e3)
        z = z + np.exp(2j * np.pi * beat * np.arange(n) / 40_000.0)
    path = factory(z=z, window=2048, step=4096,
                   t0=ECHO_T0 + random.randint(0, 9999))
    then = time.time() - age_s
    os.utime(path, (then, then))
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_old_product_loses_its_raw_voltage(make_chirp_z_h5):
    path = product(make_chirp_z_h5)
    echo.run(path.parent, min_age_s=random.uniform(60, 600))
    assert not loader.read_header(path).has_raw_voltage, \
        "the full voltage survived a sweep that cut its window"


def test_an_old_product_gains_its_echo_window(make_chirp_z_h5):
    path = product(make_chirp_z_h5)
    echo.run(path.parent, min_age_s=random.uniform(60, 600))
    assert loader.read_header(path).has_echo_window, \
        "a swept product was left without an echo window"


def test_a_young_product_is_left_untouched(make_chirp_z_h5):
    path = product(make_chirp_z_h5, age_s=random.uniform(0, 30))
    before = digest(path)
    echo.run(path.parent, min_age_s=random.uniform(120, 600))
    assert digest(path) == before, \
        "a product that may still be mid-write was rewritten"


def test_a_product_stamped_in_the_sweeps_own_tick_is_not_held_back(tmp_path):
    """Windows, 2026-10-25: Python 3.12's clock and NTFS share one coarse tick
    and round an instant to floats differently, so a product written in the
    sweep's own tick read one ulp younger than zero, too young even for
    ``min_age_s=0`` (#51)."""
    path = tmp_path / f"lfm_ionogram-{random.randint(0, 10**9)}.h5"
    path.write_bytes(os.urandom(random.randint(1, 4096)))
    found = echo.candidates(tmp_path, min_age_s=0,
                            now=math.nextafter(path.stat().st_mtime, -math.inf))
    assert found == [path], \
        "a product stamped one ulp after the sweep's clock was held back"


def test_a_product_stamped_ahead_of_the_sweep_still_waits_out_the_minimum_age(
        tmp_path):
    """Reading a negative age as zero must not let it skip `min_age_s`: a
    product stamped ahead of the clock may be the one still being written."""
    path = tmp_path / f"lfm_ionogram-{random.randint(0, 10**9)}.h5"
    path.write_bytes(os.urandom(random.randint(1, 4096)))
    found = echo.candidates(tmp_path, min_age_s=random.uniform(60, 600),
                            now=path.stat().st_mtime - random.uniform(1, 3600))
    assert found == [], \
        "a product stamped ahead of the sweep skipped the minimum age"


def test_a_cut_product_is_left_untouched(make_chirp_z_h5):
    path = product(make_chirp_z_h5)
    echo_window.write(path, drop_raw=True)
    before = digest(path)
    echo.run(path.parent, min_age_s=random.uniform(60, 600))
    assert digest(path) == before, \
        "a product already cut was rewritten by a second sweep"


def test_a_product_with_nothing_to_keep_keeps_its_raw_voltage(make_chirp_z_h5):
    path = product(make_chirp_z_h5, echo_present=False)
    echo.run(path.parent, min_age_s=random.uniform(60, 600))
    assert loader.read_header(path).has_raw_voltage, \
        "a product with no echo lost its only copy of the voltage"


def test_a_product_with_nothing_to_keep_is_counted(make_chirp_z_h5):
    path = product(make_chirp_z_h5, echo_present=False)
    result = echo.run(path.parent, min_age_s=random.uniform(60, 600))
    assert result["kept"] == 1, \
        "a product kept whole for lack of an echo went unreported"


def test_a_cut_reports_the_space_it_freed(make_chirp_z_h5):
    path = product(make_chirp_z_h5)
    size = path.stat().st_size
    result = echo.run(path.parent, min_age_s=random.uniform(60, 600))
    assert result["freed"] == size - path.stat().st_size, \
        "the sweep misreported the space it reclaimed"


def test_a_dry_run_changes_nothing(make_chirp_z_h5):
    path = product(make_chirp_z_h5)
    before = digest(path)
    echo.run(path.parent, min_age_s=random.uniform(60, 600), dry_run=True)
    assert digest(path) == before, "a dry run rewrote a product"


def test_a_failed_verification_fails_the_run(make_chirp_z_h5, monkeypatch):
    path = product(make_chirp_z_h5)

    def refuse(*args, **kwargs):
        raise echo_window.VerificationError("window power differs by 3.10 dB")

    monkeypatch.setattr(echo_window, "write", refuse)
    assert echo.main([str(path.parent), "--min-age-s", "60"]) == 1, \
        "a sweep whose cut did not verify reported success"


def test_a_missing_directory_fails_the_run(tmp_path):
    absent = tmp_path / f"absent-{random.randint(0, 999)}"
    assert echo.main([str(absent)]) == 2, \
        "a sweep of a missing directory reported success"
