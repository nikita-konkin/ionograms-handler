"""Reading another station's digisonde, received here.

The tests worth having are about the three places this format differs from
`io_chirp` and could be got wrong silently: which polarization channel is
being read, what NaN means, and where the range zero comes from.
"""

from __future__ import annotations

import numpy as np
import pytest

from muf import io_chirp, io_digisonde, loader, spectro

C_KM_S = io_digisonde.C_KM_S


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def test_name_decides_between_two_h5_formats(make_digisonde_h5, make_chirp_h5):
    """Both products are `.h5` in one directory, so the extension cannot
    decide. Getting this wrong hands a chirp reader a file whose only shared
    dataset is `t0`."""
    digi = make_digisonde_h5()
    chirp = make_chirp_h5(np.full((4, 64), 100.0))

    assert loader.format_of(digi) == loader.DIGISONDE
    assert loader.format_of(chirp) == loader.CHIRP2


def test_all_three_formats_are_found_in_one_tree(make_digisonde_h5, make_chirp_h5,
                                                 make_lfs, tmp_path):
    from conftest import synth_iq

    make_lfs(synth_iq(n_freq=64, window=256, echo_range_km=2700.0,
                      half_span_km=60_000.0, echo_last_bin=40), name="a.lfs")
    make_chirp_h5(np.full((4, 64), 100.0))
    make_digisonde_h5()

    found = loader.find_soundings(tmp_path)
    assert loader.describe_formats(found) == {
        loader.LFS: 1, loader.CHIRP2: 1, loader.DIGISONDE: 1}


def test_detection_files_are_still_not_soundings(make_digisonde_h5,
                                                 make_detection_h5, tmp_path):
    make_digisonde_h5()
    make_detection_h5("chirp", cycles=3, into=tmp_path)
    make_detection_h5("cdetections", cycles=3, into=tmp_path)

    found = loader.find_soundings(tmp_path)
    assert [p.name.split("-")[0] for p in found] == ["digisonde_ionogram"]


def test_a_file_missing_its_datasets_is_refused(make_digisonde_h5):
    bad = make_digisonde_h5(drop=("SNR",))
    with pytest.raises(ValueError, match="not a digisonde ionogram"):
        io_digisonde.read_header(bad)


def test_a_chirp_product_renamed_as_one_is_refused(make_digisonde_h5):
    """The `type` dataset is the positive identification; the name is only a
    hint, and a hint is not enough to hand a file to the wrong reader."""
    wrong = make_digisonde_h5(kind="lfm_ionogram")
    with pytest.raises(ValueError, match="type is"):
        io_digisonde.read_header(wrong)


# --------------------------------------------------------------------------
# Geometry and the range zero
# --------------------------------------------------------------------------

def test_the_range_offset_is_applied(make_digisonde_h5):
    """`ranges` is stored from zero; the absolute axis is that plus
    `offset_us`, which is how upstream plots it. Forgetting the offset shifts
    every echo by 600 km at the usual setting and still looks plausible."""
    path = make_digisonde_h5(offset_us=2000.0, n_range=64, range_step_m=3e3)
    header = io_digisonde.read_header(path)

    assert header.range_offset_km == pytest.approx(2000e-6 * C_KM_S, rel=1e-9)
    assert header.rmin == pytest.approx(round(2000e-6 * C_KM_S), abs=1)

    ion = io_digisonde.load(path)
    assert ion.cal.vrange.min() == pytest.approx(2000e-6 * C_KM_S, abs=0.5)


def test_the_zero_is_flagged_as_configured_not_measured(make_digisonde_h5):
    """`offset_us` is typed into an ini and cross-checked by nothing. 1 ms is
    300 km, and the products stay self-consistent either way -- the same trap
    as `ChirpHeader.range_is_relative`, reached from the other side."""
    assert io_digisonde.read_header(make_digisonde_h5()).range_is_configured


def test_stations_resolve_from_the_registry(make_digisonde_h5):
    """Unlike a serendipitous chirp product, both ends are named and real."""
    header = loader.read_header(make_digisonde_h5(transmitter="Juliusruh",
                                                  receiver="DOB"))
    assert header.has_coordinates
    assert header.tx_latitude == pytest.approx(54.63, abs=0.1)
    assert header.rx_latitude == pytest.approx(62.07, abs=0.1)
    assert header.is_oblique and header.path_type == "oblique"


def test_an_unknown_station_leaves_the_geometry_unavailable(make_digisonde_h5):
    header = io_digisonde.read_header(make_digisonde_h5(transmitter="nowhere"),
                                      stations={})
    assert not header.has_coordinates
    assert np.isnan(header.tx_latitude)


def test_a_pulsed_sounder_reports_no_chirp_rate(make_digisonde_h5):
    """nan rather than a plausible number: anything reaching for a chirp rate
    here is asking a question this instrument does not answer."""
    assert np.isnan(io_digisonde.read_header(make_digisonde_h5()).rate)


# --------------------------------------------------------------------------
# The array
# --------------------------------------------------------------------------

def _with_echo(n_pol=2, n_freq=32, n_range=64, *, pol=0, freq=10, rng=20,
               strength=500.0):
    power = np.ones((n_pol, n_freq, n_range))
    power[pol, freq, rng] = strength
    return power


def test_the_range_axis_descends(make_digisonde_h5):
    """This pipeline's bin 0 holds the largest virtual range; the stored axis
    ascends. Getting it backwards mirrors the ionogram and still looks like an
    ionogram -- see the `calibrate` module docstring."""
    ion = io_digisonde.load(make_digisonde_h5())
    assert ion.cal.vrange[0] > ion.cal.vrange[-1]


def test_an_echo_survives_the_flip_at_the_right_range(make_digisonde_h5):
    path = make_digisonde_h5(_with_echo(freq=10, rng=20), offset_us=0.0,
                             range_step_m=3e3)
    ion = io_digisonde.load(path)

    i, j = np.unravel_index(np.argmax(ion.power), ion.power.shape)
    assert i == 10, "frequency bin unchanged"
    assert ion.cal.vrange[j] == pytest.approx(20 * 3.0, abs=0.01)


def test_below_threshold_cells_read_as_noise_not_nan(make_digisonde_h5):
    """Roughly 90% of a real array is NaN by construction. Propagating that
    into the estimators would make every one of them special-case it."""
    ion = io_digisonde.load(make_digisonde_h5(_with_echo()))
    assert np.isfinite(ion.power).all()
    assert (ion.power > 0).all(), "noise is a level, not a hole"


def test_the_noise_floor_matches_the_chirp_reader(make_digisonde_h5):
    """The shared 43 dB detection level has to mean the same thing in both
    formats, or the estimators are not comparable across them. Both define SNR
    as (P - median)/median, so both go through `snr_to_power`."""
    ion = io_digisonde.load(make_digisonde_h5(np.ones((2, 16, 32))))
    quiet = spectro.to_db(ion.power)

    expected = spectro.to_db(io_chirp.snr_to_power(np.zeros(1)))[0]
    assert np.median(quiet) == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------
# Polarization
# --------------------------------------------------------------------------

def test_summing_is_the_default_and_keeps_both_channels(make_digisonde_h5):
    """A trace in one channel only is still a trace, and the file does not say
    which channel is O and which is X -- so picking one is picking at random."""
    power = np.ones((2, 32, 64))
    power[0, 5, 10] = 500.0        # channel 0 only
    power[1, 20, 30] = 500.0       # channel 1 only
    ion = io_digisonde.load(make_digisonde_h5(power))

    db = spectro.to_db(ion.power)
    assert db[5].max() > 43, "channel 0 echo present"
    assert db[20].max() > 43, "channel 1 echo present"


def test_a_channel_can_be_selected(make_digisonde_h5):
    power = np.ones((2, 32, 64))
    power[1, 20, 30] = 500.0
    path = make_digisonde_h5(power)

    only_0 = spectro.to_db(io_digisonde.load(path, pol=0).power)
    only_1 = spectro.to_db(io_digisonde.load(path, pol=1).power)

    assert only_0[20].max() < 43, "channel 0 does not carry it"
    assert only_1[20].max() > 43, "channel 1 does"


def test_a_channel_that_is_not_there_is_refused(make_digisonde_h5):
    with pytest.raises(ValueError, match="out of range"):
        io_digisonde.load(make_digisonde_h5(), pol=7)


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------

def test_the_gate_narrows_freely(make_digisonde_h5):
    """Unlike a v2 chirp product, nothing was discarded at acquisition: the
    whole unambiguous window is stored, so there is no wider extent to warn
    about."""
    path = make_digisonde_h5(n_range=64, range_step_m=3e3, offset_us=0.0)
    full = io_digisonde.load(path)
    gated = io_digisonde.load(path, gate_km=(30.0, 90.0))

    assert gated.cal.vrange.max() <= 90.0 and gated.cal.vrange.min() >= 30.0
    assert gated.power.shape[1] < full.power.shape[1]
    assert gated.power.shape[0] == full.power.shape[0], "frequency axis intact"


def test_a_gate_that_misses_entirely_is_an_error(make_digisonde_h5):
    with pytest.raises(ValueError, match="does not overlap"):
        io_digisonde.load(make_digisonde_h5(offset_us=0.0),
                          gate_km=(50_000.0, 60_000.0))


# --------------------------------------------------------------------------
# Through the loader
# --------------------------------------------------------------------------

def test_loader_load_returns_the_same_ionogram(make_digisonde_h5):
    path = make_digisonde_h5(_with_echo())
    direct = io_digisonde.load(path)
    through = loader.load(path)

    assert through.power.shape == direct.power.shape
    assert np.allclose(through.power, direct.power)
    assert through.header.format == "digisonde"


def test_window_and_zero_periods_do_not_warn(make_digisonde_h5, recwarn):
    """They warn for chirp2, where a window genuinely was fixed at archive
    time. A digisonde product is pulse compressed, so there is no window that
    could have been used instead and warning about one would invent a knob."""
    loader.load(make_digisonde_h5(), window=4096, zero_periods=3)
    assert not [w for w in recwarn if "window" in str(w.message)]


def test_the_cache_key_separates_the_formats(make_digisonde_h5, tmp_path):
    """`sounding.h5` and `sounding.h5` under one cache directory must not
    collide just because both are products."""
    path = make_digisonde_h5()
    key = loader.cache_key(path, 8192, 0, None)
    assert loader.DIGISONDE in key and loader.CHIRP2 not in key
