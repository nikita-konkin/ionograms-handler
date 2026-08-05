"""Reading chirpsounder2 ionogram products.

The load-bearing test is :func:`test_db_matches_lfs_convention`: the same power
spectrogram is put through v2's normalization and through this pipeline's, and
the resulting dB must agree. If it ever stops agreeing, the shared 43 dB
detection threshold means something different on the two formats and every
comparison between them is quietly wrong -- the failure `architecture.md` sec. 8
calls the one most likely to produce plausible-looking wrong numbers.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest
from conftest import (ECHO_RANGE_KM, ECHO_T0, chirp_range_gates_m,
                      chirp_range_offset_km, chirp_stored_mask)

from muf import calibrate, io_chirp
from muf.spectro import NOISE_COEF, NOISE_FLOOR_DB, to_db

pytest.importorskip("h5py")

FFTLEN = 1024
N_FREQ = 64
SR = 40_000.0
RATE = 100_000.0

#: The threshold every estimator shares. 43 dB on the scale `to_db` produces.
DETECTION_DB = NOISE_FLOOR_DB + 13.0

#: The columns v2 actually writes -- its default +/-2000 km window around the
#: direct-path delay. Expectations built from the full spectrogram have to be
#: sliced by this to line up with what came out of the file.
STORED = chirp_stored_mask(FFTLEN, SR, RATE)

#: A few bins off the direct path, so a mirrored axis cannot pass by symmetry.
ECHO_BIN = FFTLEN // 2 + 4


def make_power(echo_bin: int = ECHO_BIN, echo_power: float = 4000.0,
               echo_until: int = 48, noise_scale: float = 7.3,
               seed: int = 0) -> np.ndarray:
    """A power spectrogram: exponential noise, one echo at a known range bin.

    Exponential because that is what ``|FFT|^2`` of complex Gaussian noise is,
    and the ``4*ln2`` median-to-mean factor both pipelines carry only means
    what it means for that distribution.
    """
    rng = np.random.default_rng(seed)
    power = rng.exponential(scale=noise_scale, size=(N_FREQ, FFTLEN))
    power[:echo_until, echo_bin] += echo_power
    return power


def muf_normalized(power: np.ndarray) -> np.ndarray:
    """What ``spectro.compute`` would produce from the same spectrogram.

    Normalized over the full row *before* slicing to the stored window, because
    that is what both pipelines do -- v2's ``S0 = S[:, ridx]`` line is commented
    out upstream. Gating first here would hide a real divergence.
    """
    floor = NOISE_COEF * np.median(power, axis=1, keepdims=True)
    return (power / floor)[:, STORED]


def test_db_matches_lfs_convention(make_chirp_h5):
    """v2 and .lfs normalization must land on the same dB scale."""
    power = make_power()
    ion = io_chirp.load(make_chirp_h5(power))

    expected = to_db(muf_normalized(power))[:, ::-1]   # flipped to match vrange
    got = ion.db

    # Only where v2 actually stored a value: cells below its storage threshold
    # were replaced with NaN at acquisition and cannot be recovered.
    kept = expected >= to_db(np.array([(2.0 + 1.0) / NOISE_COEF]))[0]
    assert kept.sum() > 0
    assert np.abs(got[kept] - expected[kept]).max() < 0.05


def test_echo_crosses_the_shared_threshold(make_chirp_h5):
    """An echo above 43 dB in one format is above 43 dB in the other."""
    power = make_power()
    ion = io_chirp.load(make_chirp_h5(power))

    expected = to_db(muf_normalized(power))[:, ::-1]
    assert (expected >= DETECTION_DB).any(), "fixture echo is too weak to test"
    np.testing.assert_array_equal(ion.db >= DETECTION_DB, expected >= DETECTION_DB)


def test_dropped_cells_sit_below_the_threshold(make_chirp_h5):
    """NaN-sparsified cells must not read as detections, or as -inf."""
    # Pure noise: almost every cell falls under the storage threshold.
    ion = io_chirp.load(make_chirp_h5(make_power(echo_power=0.0)))
    assert np.isfinite(ion.db).all()
    assert ion.db.max() < DETECTION_DB


def test_range_axis_descends_and_locates_the_echo(make_chirp_h5):
    """Bin 0 is the largest range, and the echo lands where it was put."""
    echo_bin = ECHO_BIN
    power = make_power(echo_bin=echo_bin)
    path = make_chirp_h5(power)

    ion = io_chirp.load(path)
    assert ion.vrange[0] > ion.vrange[-1], "range axis must descend"

    expected_km = (chirp_range_gates_m(FFTLEN, SR, RATE)[echo_bin] / 1e3
                   + chirp_range_offset_km(ECHO_T0))
    assert STORED[echo_bin], "fixture echo must be inside the stored window"
    found_km = ion.vrange[int(np.argmax(ion.power.sum(axis=0)))]
    assert abs(found_km - expected_km) < ion.cal.range_step


def test_relative_axis_is_completed_from_t0(make_chirp_h5):
    """The stock path stores ranges centred on the direct-path delay.

    Without the `(t0 - floor(t0)) * c` term the axis is centred on 0 km instead
    of on the path length -- an ionogram that plots cleanly and is wrong by
    2710 km. This is the correction `plot_ionograms.py` applies, and the reason
    it cannot be skipped just because `range_offset_applied` is False.
    """
    # The echo sits at the middle range gate, i.e. at raw range ~0.
    path = make_chirp_h5(make_power(echo_bin=FFTLEN // 2))
    ion = io_chirp.load(path)

    found_km = ion.vrange[int(np.argmax(ion.power.sum(axis=0)))]
    assert found_km == pytest.approx(ECHO_RANGE_KM, abs=ion.cal.range_step)
    # and the whole axis moved with it, rather than only the peak
    assert ion.vrange.min() > 0.0


def test_applied_offset_is_not_added_twice(make_chirp_h5):
    """When v2 already wrote display ranges, leave them alone."""
    path = make_chirp_h5(make_power(echo_bin=FFTLEN // 2),
                         range_offset_applied=True)
    ion = io_chirp.load(path)

    found_km = ion.vrange[int(np.argmax(ion.power.sum(axis=0)))]
    assert found_km == pytest.approx(0.0, abs=ion.cal.range_step)


def test_unidentified_transmitter_keeps_the_axis_relative(make_chirp_h5):
    """v2's `unkown` marker means nothing can vouch for the offset.

    Search mode measures `t0` instead of imposing it from `sounder_timings`,
    so the offset is only as good as the receiver's absolute clock. The DOB
    run of 2026-08-05 showed what that costs: a clock 0.956 s slow turned
    cyprus1's 3436 km path into 16,700 km, on products where no station name
    existed to check it against. An axis that is honestly relative is usable;
    one that is confidently absolute and wrong by 13,000 km is not.
    """
    path = make_chirp_h5(make_power(echo_bin=FFTLEN // 2),
                         txname=io_chirp.UNIDENTIFIED_TX)
    with pytest.warns(UserWarning, match="range axis left relative"):
        ion = io_chirp.load(path)

    assert ion.header.range_is_relative
    assert io_chirp.UNIDENTIFIED_TX in ion.header.range_relative_reason
    # centred on zero, exactly as v2 stored it
    found_km = ion.vrange[int(np.argmax(ion.power.sum(axis=0)))]
    assert found_km == pytest.approx(0.0, abs=ion.cal.range_step)
    assert ion.vrange.min() < 0.0


def test_impossible_offset_is_refused_even_for_a_named_transmitter(make_chirp_h5):
    """No terrestrial one-way path is 27,673 km long.

    That is what the 125 kHz/s emitter's `t0` implied at DOB before the
    receiver's clock error was known. A named station does not make the number
    physical, so the bound is checked independently of who the file says
    transmitted -- a scheduled sounding on a broken clock lands here too.
    """
    # 0.1 s past the second -> 29,979 km, past MAX_VIRTUAL_RANGE_KM.
    t0 = float(int(ECHO_T0)) + 0.1
    path = make_chirp_h5(make_power(echo_bin=FFTLEN // 2), t0=t0)
    with pytest.warns(UserWarning, match="past the 22000 km"):
        ion = io_chirp.load(path)

    assert ion.header.range_is_relative
    assert ion.vrange.min() < 0.0


def test_plausible_offset_on_a_named_transmitter_is_still_applied(make_chirp_h5):
    """The guards must not disturb the scheduled path, which is the common one."""
    path = make_chirp_h5(make_power(echo_bin=FFTLEN // 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ion = io_chirp.load(path)

    assert not ion.header.range_is_relative
    assert ion.header.range_relative_reason == ""
    assert ion.vrange.min() > 0.0


def test_relative_axis_keeps_range_differences(make_chirp_h5):
    """What survives is exactly what the extractors consume.

    `pick_muf` returns a frequency and is unaffected; a spacing between two
    echoes is unaffected; only the zero is gone. The test pins that, because
    the whole argument for returning a relative axis is that it stays useful.
    """
    power = make_power(echo_bin=FFTLEN // 2)
    absolute = io_chirp.load(make_chirp_h5(power, name="a.h5"))
    with pytest.warns(UserWarning, match="range axis left relative"):
        relative = io_chirp.load(make_chirp_h5(power, txname=io_chirp.UNIDENTIFIED_TX,
                                               name="b.h5"))

    np.testing.assert_allclose(np.diff(relative.vrange), np.diff(absolute.vrange))
    assert relative.vrange.size == absolute.vrange.size


def test_orientation_is_frequency_by_range(make_chirp_h5):
    """Rows are frequency, columns range -- as v2 stores it and as muf wants."""
    ion = io_chirp.load(make_chirp_h5(make_power()))
    assert ion.shape == (N_FREQ, int(STORED.sum()))
    assert ion.cal.n_freq == N_FREQ
    assert ion.cal.n_range == int(STORED.sum())


def test_header_reconstructs_the_acquisition_scalars(make_chirp_h5):
    """`cf`/`dur`/`sample_rate` are derived so `calibrate` reproduces the axes."""
    header = io_chirp.read_header(make_chirp_h5(make_power()))

    assert header.format == "chirp2"
    assert header.tx_name == "synthtx" and header.rx_name == "synthrx"
    assert header.path_type == "oblique" and header.div_coef == 2.0
    assert header.datetime.year == 2026

    # The calibration's half-span describes the full FFT axis, not the sliver
    # v2 stored, and on an oblique path it must agree with the .lfs formula.
    ion = io_chirp.load(make_chirp_h5(make_power()))
    assert ion.cal.half_span == pytest.approx(
        calibrate.range_half_span(header), rel=1e-3)
    assert ion.cal.n_range < ion.cal.n_range_full

    # `cf` and `dur` are chosen so this inverts exactly, not approximately:
    # sweep_bounds must return the frequency span the file actually covers.
    start, stop = calibrate.sweep_bounds(header)
    assert start == pytest.approx(float(ion.cal.freq[0]), rel=1e-9)
    assert stop == pytest.approx(float(ion.cal.freq[-1]), rel=1e-9)


def test_recovers_the_exact_fft_length(make_chirp_h5):
    """`n_range_full` must come back as v2's real fftlen, not a near miss.

    It is a rounded ratio of two lengths, so it only lands exactly when the
    half-span is rebuilt with the same speed of light v2 used. `calibrate`'s
    C_KM_S is exactly 3e5; scipy's is not.
    """
    # Store a slice, so the count cannot come from the array's own width.
    ion = io_chirp.load(make_chirp_h5(make_power(), keep=slice(600, 800)))
    assert ion.cal.n_range_full == FFTLEN
    assert ion.window == FFTLEN


def test_oblique_half_span_agrees_with_the_lfs_convention(make_chirp_h5):
    """The two range conventions must coincide on an oblique path.

    This is what lets the 2710 km baseline and the default gate carry over to
    v2 unchanged: v2 applies no path divisor, and `div_coef == 2` makes
    `calibrate.range_half_span` the same quantity.
    """
    header = io_chirp.read_header(make_chirp_h5(make_power()))
    assert header.is_oblique
    ours = calibrate.range_half_span(header)
    theirs = io_chirp.chirp_half_span_km(header.rate, header.sample_rate)
    assert theirs == pytest.approx(ours, rel=1e-3)


def test_vertical_path_warns_about_the_missing_divisor(make_chirp_h5):
    """v2 has no div_coef, so a vertical sounding's ranges are 2x ours."""
    path = make_chirp_h5(make_power(), txname="samesite", station_name="samesite")
    header = io_chirp.read_header(path)
    assert header.div_coef == 4.0                     # what .lfs would apply

    with pytest.warns(UserWarning, match="no vertical divisor"):
        ion = io_chirp.load(path)

    # The stored axis is kept as-is, so it stays consistent with n_range_full.
    assert ion.cal.n_range_full == FFTLEN
    assert ion.cal.half_span == pytest.approx(
        2 * calibrate.range_half_span(header), rel=1e-3)


def test_nominal_stop_makes_truncation_visible(make_chirp_h5):
    """Without `maximum_analysis_frequency` a short sweep looks complete."""
    ion = io_chirp.load(make_chirp_h5(make_power()))
    assert ion.cal.sweep_complete and ion.cal.sweep_fraction == 1.0

    # The same file, told what the sweep was supposed to reach.
    ion = io_chirp.load(make_chirp_h5(make_power()), nominal_stop_mhz=25.0)
    assert not ion.cal.sweep_complete
    assert 0.0 < ion.cal.sweep_fraction < 1.0


def test_unknown_station_degrades_to_no_geometry(make_chirp_h5):
    """A transmitter absent from the registry must not stop the extraction."""
    header = io_chirp.read_header(make_chirp_h5(make_power()))
    assert not header.has_coordinates
    assert calibrate.ground_range_km(header) is None
    # and the default gate still returns something usable
    lo, hi = calibrate.default_gate(header)
    assert lo < hi


def test_station_registry_supplies_coordinates(make_chirp_h5):
    stations = {"synthtx": (35.0, 34.0), "synthrx": (56.38, 47.53)}
    header = io_chirp.read_header(make_chirp_h5(make_power()), stations)

    assert header.has_coordinates
    assert calibrate.ground_range_km(header) == pytest.approx(2588.4, abs=1.0)


# --- gating ------------------------------------------------------------------

def test_narrower_gate_is_applied(make_chirp_h5):
    path = make_chirp_h5(make_power())
    full = io_chirp.load(path)

    # Fully inside the 719-4701 km v2 stored, so nothing is clamped.
    lo, hi = 2000.0, 4000.0
    gated = io_chirp.load(path, gate_km=(lo, hi))

    assert gated.cal.n_range < full.cal.n_range
    assert gated.vrange.min() >= lo and gated.vrange.max() <= hi
    assert gated.cal.gate_km == (lo, hi)


def test_partly_wider_gate_is_clamped_to_what_was_stored(make_chirp_h5):
    """A gate overhanging the stored extent records what was applied, not asked."""
    path = make_chirp_h5(make_power())
    with pytest.warns(UserWarning, match="wider than"):
        gated = io_chirp.load(path, gate_km=(2000.0, 99_000.0))

    stored_hi = io_chirp.load(path).vrange.max()
    assert gated.cal.gate_km[0] == 2000.0
    assert gated.cal.gate_km[1] == pytest.approx(stored_hi)


def test_wider_gate_warns_and_does_not_double_gate(make_chirp_h5):
    """v2 already discarded the rest; a wider request cannot recover it."""
    # Store only a slice, as v2 does when it gates at acquisition.
    keep = slice(600, 800)
    path = make_chirp_h5(make_power(), keep=keep)
    stored = io_chirp.load(path)

    with pytest.warns(UserWarning, match="wider than"):
        wide = io_chirp.load(path, gate_km=(-60_000.0, 60_000.0))

    assert wide.cal.n_range == stored.cal.n_range
    np.testing.assert_allclose(wide.vrange, stored.vrange)


def test_non_overlapping_gate_raises(make_chirp_h5):
    path = make_chirp_h5(make_power(), keep=slice(600, 800))
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="does not overlap"):
            io_chirp.load(path, gate_km=(50_000.0, 60_000.0))


# --- provenance and failure modes --------------------------------------------

def test_gate_default_is_absolute_not_relative(make_chirp_h5):
    """`rmin`/`rmax` must be absolute, or the default gate lands around 0 km."""
    header = io_chirp.read_header(make_chirp_h5(make_power()))
    assert header.rmin > 0 and header.rmax > header.rmin
    assert header.range_offset_km == pytest.approx(ECHO_RANGE_KM, abs=1.0)


def test_detection_file_is_rejected_with_its_contents(tmp_path):
    """A chirp-*.h5 must fail loudly, naming what it actually holds."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "chirp-1770163210.00.h5"
    with h5py.File(path, "w") as fh:
        fh["chirp_rate"] = 100_000.0
        fh["f0"] = 7.5e6
        fh["t0"] = 1770163210.0

    with pytest.raises(ValueError, match="not a chirpsounder2 ionogram product"):
        io_chirp.read_header(path)
    # the message must list what is there, so the mistake is obvious
    with pytest.raises(ValueError, match="chirp_rate"):
        io_chirp.read_header(path)


def test_find_h5_ignores_detection_files(tmp_path, make_chirp_h5):
    for name in ("chirp-1770163210.00.h5", "cdetections-1770163210.00.h5",
                 "par-1770163210.00.h5"):
        (tmp_path / name).write_bytes(b"")
    real = make_chirp_h5(make_power())
    (tmp_path / real.name).write_bytes(real.read_bytes())

    found = io_chirp.find_h5(tmp_path)
    assert [p.name for p in found] == [real.name]


def test_carries_the_acquisition_software_provenance(make_chirp_h5):
    """Which code produced the number, not just which configuration.

    A product whose `git_dirty` is True was written by a modified clone, which
    is the one thing architecture.md sec. 2.2 says must never happen -- so it is
    worth being able to see it from the file rather than from the laptop.
    """
    header = io_chirp.read_header(make_chirp_h5(make_power()))
    assert header.software_version == "0.2.0"
    assert header.git_commit == "0d2712553063"
    assert header.git_dirty is False


def test_missing_version_tag_is_not_fatal(make_chirp_h5, tmp_path):
    """Products predating the tagger must still load."""
    h5py = pytest.importorskip("h5py")
    path = make_chirp_h5(make_power())
    with h5py.File(path, "a") as fh:
        for key in ("chirpsounder2_version", "git_commit", "git_dirty"):
            del fh.attrs[key]

    header = io_chirp.read_header(path)
    assert header.software_version == "" and header.git_dirty is None


def test_describe_lists_the_schema(make_chirp_h5):
    schema = io_chirp.describe(make_chirp_h5(make_power()))
    assert schema["SNR"].startswith(f"({N_FREQ}, {int(STORED.sum())})")
    assert "scalar" in schema["rate"]
    assert "txname" in schema


# --------------------------------------------------------------------------
# Against real products. These skip when the recordings are not on the machine,
# so they cannot be the only cover for anything -- each one has a synthetic
# counterpart above. What they add is the half a fixture cannot: that h5py
# hands back what this module expects from a file chirpsounder2 really wrote.
# --------------------------------------------------------------------------


#: The one warning a real product is allowed to raise. Search-mode archives
#: emit it on every file by design; scheduled ones never do. These tests have
#: to pass against either, so they permit this and nothing else.
RELATIVE_RANGE_WARNING = "range axis left relative"


@contextlib.contextmanager
def only_relative_range_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=f".*{RELATIVE_RANGE_WARNING}.*")
        yield


def test_real_product_loads(real_chirp_file):
    """The end-to-end shape of a real file, and that nothing unexpected warns."""
    with only_relative_range_warning():
        header = io_chirp.read_header(real_chirp_file)
        ion = io_chirp.load(real_chirp_file)

    assert header.format == "chirp2"
    assert header.tx_name and header.rx_name       # decoded, not b'...'
    assert isinstance(header.tx_name, str)
    assert header.is_oblique
    assert not header.has_coordinates              # until the registry lands
    assert ion.power.shape == (ion.freq.size, ion.vrange.size)
    assert np.isfinite(ion.db).all()
    # The axis is relative exactly when something declined to vouch for the
    # offset -- never silently, and never for a product that named its
    # transmitter and implied a reachable distance.
    if header.range_is_relative:
        assert (header.tx_name == io_chirp.UNIDENTIFIED_TX
                or header.range_offset_km > io_chirp.MAX_VIRTUAL_RANGE_KM)
    else:
        assert header.range_relative_reason == ""


def test_real_range_axis_descends(real_chirp_file):
    """v2 stores an ascending axis; bin 0 here must be the largest range.

    Mirroring this is the error that plots cleanly and is upside down, so it is
    worth asserting against a file whose axis this module did not construct.
    """
    ion = io_chirp.load(real_chirp_file)
    assert ion.vrange[0] > ion.vrange[-1]
    assert np.all(np.diff(ion.vrange) < 0)


def test_real_noise_floor_lands_on_the_lfs_convention(real_chirp_file):
    """The whole point of the module, checked on a real product.

    v2 sparsifies at `storage_snr_threshold`, so every NaN cell is filled with
    `SNR_v2 = 0` -- the row median. Those cells are the floor of the array, and
    the floor must be exactly where a median-noise cell lands on the `.lfs`
    path. If this drifts, 43 dB no longer means the same thing on the two
    formats. See `test_db_matches_lfs_convention` for the synthetic proof that
    the whole scale, not just the floor, agrees.
    """
    ion = io_chirp.load(real_chirp_file)
    expected = to_db(np.array([1.0 / NOISE_COEF], dtype=np.float32))[0]
    assert float(ion.db.min()) == pytest.approx(expected, abs=1e-4)
    assert float(ion.db.min()) == pytest.approx(25.571, abs=1e-3)


def test_real_noise_floor_may_be_longer_than_the_frequency_axis(real_chirp_file):
    """`noise_floor` is full length while `freqs` is cut by `manual_freq_extent`.

    `calc_ionograms.py` writes `noise_floor` before applying `fidx` and
    `freqs[fidx]` after, so the two legitimately disagree -- 497 against 310 at
    DOB. Reading `noise_floor` positionally against `freqs` would be a silent
    frequency shift, which is why the header only ever takes its median.
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(real_chirp_file, "r") as fh:
        assert fh["noise_floor"].shape[0] >= fh["freqs"].shape[0]
    assert np.isfinite(io_chirp.read_header(real_chirp_file).noise_floor_median)


def test_real_fft_length_is_recovered_exactly(real_chirp_file):
    """`n_range_full` must come back as v2's own `fftlen`.

    Upstream builds it as `int(sr_dec * ds / dr / 2.0) * 2`, so it is even and
    exact. Recovering it is a round trip through `chirp_half_span_km` and the
    stored bin spacing, and it only closes if both use v2's speed of light --
    which is why this module carries `C_M_S` rather than `calibrate.C_KM_S`.
    """
    ion = io_chirp.load(real_chirp_file)
    assert ion.cal.n_range_full % 2 == 0
    assert ion.cal.n_range_full >= ion.vrange.size
    half = io_chirp.chirp_half_span_km(ion.header.rate, ion.header.sample_rate)
    assert ion.cal.n_range_full == pytest.approx(
        2 * half / ion.cal.range_step, abs=0.5)


def test_real_wider_gate_never_widens_the_data(real_chirp_file):
    """Never double-gate: asking for more than v2 stored yields what it stored."""
    with only_relative_range_warning():
        stored = io_chirp.load(real_chirp_file)
    # Derived from the stored extent rather than hard-coded, so this stays a
    # test of "wider" on a relative axis (which starts negative) too.
    wider = (float(stored.vrange.min()) - 1000.0,
             float(stored.vrange.max()) + 1000.0)
    with pytest.warns(UserWarning, match="wider than"):
        wide = io_chirp.load(real_chirp_file, gate_km=wider)
    assert wide.vrange.size == stored.vrange.size
    np.testing.assert_array_equal(wide.vrange, stored.vrange)


def test_real_narrower_gate_is_applied(real_chirp_file):
    with only_relative_range_warning():
        stored = io_chirp.load(real_chirp_file)
    lo = float(stored.vrange.min()) + 100.0
    hi = float(stored.vrange.max()) - 100.0
    with only_relative_range_warning():
        narrow = io_chirp.load(real_chirp_file, gate_km=(lo, hi))
    assert narrow.vrange.size < stored.vrange.size
    assert narrow.vrange.min() >= lo and narrow.vrange.max() <= hi
    assert narrow.cal.gate_km[0] >= lo and narrow.cal.gate_km[1] <= hi


def test_find_h5_ignores_the_other_products_in_a_real_tree(real_chirp_dir):
    """A real v2 tree also holds detection files and, at DOB, digisonde products.

    `digisonde_ionogram-*.h5` is the interesting one: it is a different
    instrument with a 3-D `SNR` and no `rate`, yet it is close enough to the
    ionogram schema to be worth excluding by name as well as by schema.
    """
    found = io_chirp.find_h5(real_chirp_dir)
    assert found, "fixture guarantees at least one product"
    assert all(p.name.startswith("lfm_ionogram-") for p in found)
    assert len(found) < len(list(real_chirp_dir.glob("*.h5")))


@pytest.mark.parametrize("pattern", ["chirp-*.h5", "cdetections-*.h5",
                                     "digisonde_ionogram-*.h5"])
def test_real_other_products_are_rejected_by_schema(real_chirp_dir, pattern):
    """Belt and braces: the name filter is not the only thing keeping them out."""
    others = sorted(real_chirp_dir.glob(pattern))
    if not others:
        pytest.skip(f"no {pattern} alongside the products")
    with pytest.raises(ValueError, match="not a chirpsounder2 ionogram product"):
        io_chirp.read_header(others[0])
