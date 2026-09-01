"""Reading chirpsounder2 detection files.

The load-bearing tests here are about the two ways a schedule reads correctly
and means something wrong: a fractional second that straddles the integer, and
a receiver whose epoch is off by more than half a second. Both produce files
that are internally consistent, and the DOB archive of 2026-08-05 contained the
second one for eleven hours without anything noticing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from muf import io_detect

C_M_S = 299_792_458.0
CYPRUS_KM = 3436.0
CYPRUS_SLOTS = (235, 240, 245)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def test_reads_a_detection(make_detection_h5):
    root = make_detection_h5("chirp", cycles=1, transmit_seconds=(235,))
    paths = io_detect.find_detections(root)
    assert len(paths) == 1

    det = io_detect.read_detection(paths[0])
    assert det.rate == pytest.approx(100e3)
    assert det.channel == "ch0"
    assert det.sample_rate == pytest.approx(25e6)
    # 3436 km one way is 11.46 ms, and that is the whole fractional second
    assert det.fraction_s == pytest.approx(CYPRUS_KM * 1e3 / C_M_S, abs=1e-6)


def test_reads_a_timing_solution(make_detection_h5):
    root = make_detection_h5("par", cycles=1, transmit_seconds=(235,))
    sol = io_detect.read_timing(io_detect.find_timings(root)[0])

    assert sol.num_detections == 3
    assert sol.t0s.size == 3 and sol.f0s.size == 3 and sol.snrs.size == 3
    assert sol.swept_hz == pytest.approx(2e6)
    assert sol.git_commit == "0d2712553063"
    assert sol.git_dirty is True


def test_reads_cdetections(make_detection_h5):
    root = make_detection_h5("cdetections", cycles=2, transmit_seconds=(235, 240))
    data = io_detect.read_cdetections(io_detect.find_cdetections(root)[0])

    assert data.shape == (4, len(io_detect.CDETECTION_COLUMNS))
    rates = data[:, io_detect.CDETECTION_COLUMNS.index("chirp_rate")]
    assert np.allclose(rates, 100e3)


def test_ionogram_products_are_rejected_by_schema(make_chirp_h5):
    """The two readers share a tree; each must refuse the other's files."""
    path = make_chirp_h5(np.full((4, 64), 100.0))
    with pytest.raises(ValueError, match="not a chirpsounder2 detection file"):
        io_detect.read_detection(path)
    with pytest.raises(ValueError, match="not a chirpsounder2 timing file"):
        io_detect.read_timing(path)
    with pytest.raises(ValueError, match="not a chirpsounder2 cdetections file"):
        io_detect.read_cdetections(path)


def test_the_finders_do_not_pick_up_each_other(make_detection_h5, tmp_path):
    chirps = make_detection_h5("chirp", cycles=2, transmit_seconds=(235,))
    pars = make_detection_h5("par", cycles=2, transmit_seconds=(235,))
    assert all(p.name.startswith("chirp-") for p in io_detect.find_detections(chirps))
    assert all(p.name.startswith("par-") for p in io_detect.find_timings(pars))
    # `chirp-*.h5` must not swallow `lfm_ionogram-*.h5`, whose name contains
    # neither prefix, nor vice versa
    assert io_detect.find_timings(chirps) == []


def test_a_truncated_file_does_not_lose_the_census(make_detection_h5):
    """A live tree is written continuously; one bad file must not be fatal."""
    root = make_detection_h5("par", cycles=4, transmit_seconds=CYPRUS_SLOTS)
    (root / "par-ch0-9999999999.0000.h5").write_bytes(b"not an hdf5 file")

    with pytest.warns(UserWarning, match="skipped 1 unreadable"):
        sols = io_detect.load_timings(root)
    assert len(sols) == 12


# --------------------------------------------------------------------------
# Census
# --------------------------------------------------------------------------

def test_census_groups_one_transmitter_into_one_emitter(make_detection_h5):
    root = make_detection_h5("par", cycles=5, transmit_seconds=CYPRUS_SLOTS)
    emitters = io_detect.census(io_detect.load_timings(root))

    assert len(emitters) == 1
    e = emitters[0]
    assert e.count == 15
    assert e.rate == pytest.approx(100e3)
    assert e.observed_seconds == CYPRUS_SLOTS
    assert e.fraction_s == pytest.approx(CYPRUS_KM * 1e3 / C_M_S, abs=1e-5)


def test_census_separates_two_transmitters_sharing_a_slot(make_detection_h5,
                                                          tmp_path):
    """Same rate, same second, different range -- the phase is what separates."""
    tree = tmp_path / "both"
    make_detection_h5("par", cycles=4, transmit_seconds=(235,),
                      distance_km=1000.0, into=tree)
    make_detection_h5("par", cycles=4, transmit_seconds=(235,),
                      distance_km=8000.0, into=tree)
    emitters = io_detect.census(io_detect.load_timings(tree))

    assert len(emitters) == 2
    assert [e.count for e in emitters] == [4, 4]
    ranges = sorted(e.fraction_s * C_M_S / 1e3 for e in emitters)
    assert ranges[0] == pytest.approx(1000.0, abs=1.0)
    assert ranges[1] == pytest.approx(8000.0, abs=1.0)


def test_a_phase_straddling_the_whole_second_stays_one_emitter(make_detection_h5):
    """The bug that splits one transmitter in two and puts it at 150,000 km.

    A transmitter 30 km away arrives 0.1 ms after the second; with a receiver
    1 ms slow, half its detections land at 0.9999 and half at 0.0001. Averaged
    naively those give 0.5 s -- a plausible-looking ionogram at an impossible
    range -- and clustered naively they are two transmitters.
    """
    root = make_detection_h5("par", cycles=12, transmit_seconds=(235,),
                             distance_km=30.0, epoch_offset_s=-1e-4,
                             jitter_s=2e-4, seed=3)
    emitters = io_detect.census(io_detect.load_timings(root))

    assert len(emitters) == 1, "straddling the integer split one emitter in two"
    assert emitters[0].count == 12
    assert emitters[0].fraction_sd_s < 1e-3


def test_census_drops_one_off_false_alarms(make_detection_h5, tmp_path):
    tree = tmp_path / "noisy"
    make_detection_h5("par", cycles=6, transmit_seconds=(235,), into=tree)
    # a single unrepeated detection at an unrelated phase, as a false alarm
    # looks: nothing constrains where in the second it lands
    make_detection_h5("par", cycles=1, transmit_seconds=(77,),
                      distance_km=1234.0, into=tree)
    sols = io_detect.load_timings(tree)
    assert len(sols) == 7

    assert len(io_detect.census(sols)) == 1
    assert len(io_detect.census(sols, min_count=1)) == 2


# The failure the two-stage grouping exists for. Single linkage joins anything
# whose neighbours are closer than the tolerance, so on a busy band a dense
# population chains end to end: over 18-28 Aug 2026 at Yoshkar-Ola, 9,714
# solutions at 100 kHz/s came back as ONE emitter holding 31 slots with 7.6 ms
# of phase scatter. Lowering the tolerance did not help -- from 5 ms down to
# 1 ms the same arrivals stayed welded, because a few scattered slots lay
# across the gaps and bridged them.

def test_a_smeared_slot_no_longer_welds_two_transmitters(make_detection_h5,
                                                         tmp_path):
    """A slot with a wide spread must not bridge the gap between two others.

    Two transmitters 2400 km apart in range: 8 ms of phase, further apart than
    the 5 ms that separates individual arrivals, so on their own they resolve.
    A third slot then sits at the *same* range as the first but smeared by
    2 ms, which is what a slot carrying interference looks like. Its arrivals
    reach most of the way to the second transmitter, and linking raw arrivals
    walks up the smear and out the other side, reporting all three as one.

    Collapsing the slot to a single median is what stops it: the smear is
    centred on the transmitter it belongs with, so it joins that one and
    cannot reach the other.
    """
    tree = tmp_path / "smeared"
    make_detection_h5("par", cycles=60, transmit_seconds=(100,),
                      distance_km=2000.0, jitter_s=3e-4, into=tree, seed=1)
    make_detection_h5("par", cycles=60, transmit_seconds=(200,),
                      distance_km=4400.0, jitter_s=3e-4, into=tree, seed=2)
    make_detection_h5("par", cycles=60, transmit_seconds=(250,),
                      distance_km=2000.0, jitter_s=2e-3, into=tree, seed=3)

    sols = io_detect.load_timings(tree)

    # What the old single-stage rule did with exactly this input, kept here so
    # the test states the defect rather than only the cure: one emitter.
    welded = io_detect._link_on_phase(
        np.array([s.t0 % 1.0 for s in sols]), io_detect.PHASE_TOLERANCE_S)
    assert len(welded) == 1, "the fixture no longer reproduces the bridging"

    emitters = io_detect.census(sols)
    homes = {s: e for e in emitters for s in e.observed_seconds}
    assert homes[100] is not homes[200], "the smear welded them into one"
    assert homes[250] is homes[100], "the smear belongs with its own range"
    assert homes[200].count == 60


def test_every_emitter_has_a_consistent_arrival_phase(make_detection_h5,
                                                      tmp_path):
    """The invariant the two stages buy, and the one worth guarding.

    A census row's slot list is what an operator identifies a transmitter by,
    so every slot in a row has to actually agree about where the signal came
    from. Before the slots were grouped first this did not hold: the worst row
    on the real archive carried 7.6 ms of scatter, ten times what one
    transmitter shows.
    """
    tree = tmp_path / "mixed"
    for i, (second, km) in enumerate(((100, 1500.0), (150, 2400.0),
                                      (200, 3300.0), (250, 4200.0))):
        make_detection_h5("par", cycles=20, transmit_seconds=(second,),
                          distance_km=km, jitter_s=8e-4, into=tree, seed=i)

    emitters = io_detect.census(io_detect.load_timings(tree))
    assert len(emitters) == 4, "four ranges, four transmitters"
    for e in emitters:
        assert e.fraction_sd_s <= io_detect.SLOT_PHASE_TOLERANCE_S, (
            f"{e.observed_seconds} disagree about the arrival phase")


def test_slots_of_one_transmitter_still_come_back_as_one_emitter(
        make_detection_h5):
    """The other direction: grouping by slot must not shatter a schedule.

    Nicosia is heard on five slots of a 300 s cycle at one range. All five
    share a phase, so the second stage has to put them back together -- a
    census that reported five transmitters would be as useless as one that
    reported them all as a single row with everything else.
    """
    root = make_detection_h5("par", cycles=10,
                             transmit_seconds=(0, 235, 240, 245, 280),
                             distance_km=3436.0, jitter_s=3e-4, seed=7)
    emitters = io_detect.census(io_detect.load_timings(root))
    assert len(emitters) == 1
    assert emitters[0].observed_seconds == (0, 235, 240, 245, 280)
    assert emitters[0].count == 50


def test_census_accepts_raw_detections_too(make_detection_h5):
    root = make_detection_h5("chirp", cycles=5, transmit_seconds=CYPRUS_SLOTS)
    emitters = io_detect.census(io_detect.load_detections(root))
    assert len(emitters) == 1 and emitters[0].count == 15


def test_census_rejects_the_wrong_record_type():
    with pytest.raises(TypeError, match="Detection or TimingSolution"):
        io_detect.census([object()])


# --------------------------------------------------------------------------
# Epoch offset -- the DOB fault
# --------------------------------------------------------------------------

@pytest.mark.parametrize("offset_s", [0.0, -0.9557, +0.4, -1.9557])
def test_epoch_offset_is_recovered(make_detection_h5, offset_s):
    """Including the case that actually happened, and one past a whole second."""
    root = make_detection_h5("par", cycles=8, transmit_seconds=CYPRUS_SLOTS,
                             distance_km=CYPRUS_KM, epoch_offset_s=offset_s,
                             jitter_s=3e-4, seed=1)
    found = io_detect.solve_epoch_offset(
        io_detect.load_timings(root), rate=100e3,
        transmit_seconds=CYPRUS_SLOTS, distance_km=CYPRUS_KM,
        reference="cyprus1", window_s=2.5)

    assert found.seconds == pytest.approx(offset_s, abs=1e-3)
    assert found.n_slots == 3
    assert found.residual_sd_s < 1e-3


def test_epoch_offset_recovers_the_true_range(make_detection_h5):
    """The whole point: a 0.956 s clock error made cyprus1 read 16,700 km."""
    root = make_detection_h5("par", cycles=8, transmit_seconds=CYPRUS_SLOTS,
                             distance_km=CYPRUS_KM, epoch_offset_s=-0.9557)
    emitters = io_detect.census(io_detect.load_timings(root))
    e = emitters[0]

    # uncorrected, the number is wrong and gives no sign of it
    assert e.range_km() == pytest.approx(16_697, abs=50)
    assert 0.0 < e.range_km() < io_detect.MAX_VIRTUAL_RANGE_KM

    offset = io_detect.solve_epoch_offset(
        io_detect.load_timings(root), 100e3, CYPRUS_SLOTS, CYPRUS_KM,
        reference="cyprus1")
    assert e.range_km(offset.seconds) == pytest.approx(CYPRUS_KM, abs=1.0)


def test_observed_seconds_are_not_transmit_seconds(make_detection_h5):
    """cyprus1's 900:0 slot arrived during second 299 at DOB.

    Published against the Twente list, 299 matches nothing. The whole-second
    shift is invisible from inside the data.
    """
    root = make_detection_h5("par", cycles=6, transmit_seconds=(0,),
                             distance_km=CYPRUS_KM, epoch_offset_s=-0.9557)
    e = io_detect.census(io_detect.load_timings(root))[0]

    assert e.observed_seconds == (299,)
    assert e.transmit_seconds() == (299,)          # uncorrected: still wrong
    assert e.transmit_seconds(-0.9557) == (0,)     # corrected: matches Twente


def test_impossible_range_warns(make_detection_h5):
    """A phase implying more than half the planet is a clock, not a transmitter."""
    root = make_detection_h5("par", cycles=5, transmit_seconds=(235,),
                             distance_km=CYPRUS_KM, epoch_offset_s=-0.9)
    e = io_detect.census(io_detect.load_timings(root))[0]

    with pytest.warns(UserWarning, match="past the 22000 km"):
        km = e.range_km()
    assert km > io_detect.MAX_VIRTUAL_RANGE_KM


def test_naming_the_wrong_transmitter_raises(make_detection_h5):
    root = make_detection_h5("par", cycles=5, transmit_seconds=(235,))
    with pytest.raises(ValueError, match="does not|lands within"):
        io_detect.solve_epoch_offset(
            io_detect.load_timings(root), 100e3, transmit_seconds=(17, 42),
            distance_km=CYPRUS_KM, reference="not this one", window_s=0.5)


def test_absent_rate_raises(make_detection_h5):
    root = make_detection_h5("par", cycles=5, transmit_seconds=(235,))
    with pytest.raises(ValueError, match="no 125 kHz/s records"):
        io_detect.solve_epoch_offset(
            io_detect.load_timings(root), 125e3, CYPRUS_SLOTS, CYPRUS_KM)


def test_epoch_offset_uncertainty_is_reported_in_km(make_detection_h5):
    root = make_detection_h5("par", cycles=8, transmit_seconds=CYPRUS_SLOTS,
                             jitter_s=1e-3, seed=5)
    off = io_detect.solve_epoch_offset(
        io_detect.load_timings(root), 100e3, CYPRUS_SLOTS, CYPRUS_KM)
    assert off.range_uncertainty_km == pytest.approx(
        off.residual_sd_s * C_M_S / 1e3)
    assert "km" in str(off)


def test_describe_refuses_to_call_phase_a_range(make_detection_h5):
    root = make_detection_h5("par", cycles=5, transmit_seconds=CYPRUS_SLOTS)
    text = io_detect.describe(root)
    assert "AS RECEIVED" in text
    assert "solve_epoch_offset" in text
    assert "100k" in text


def test_empty_census_is_empty_not_an_error():
    assert io_detect.census([]) == []


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------

def test_real_detection_tree_yields_the_known_emitters(real_chirp_dir):
    """The 2026-08-05 DOB archive, whose answer was established by hand.

    Skips silently on the 2026-08-04 tree, which was scheduled rather than
    search mode and holds no timing solutions worth a census.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sols = io_detect.load_timings(real_chirp_dir)
    if len(sols) < 50:
        pytest.skip(f"{real_chirp_dir} holds {len(sols)} timing solutions, "
                    f"not a search-mode archive")

    emitters = io_detect.census(sols)
    assert emitters, "a search-mode archive must contain repeating emitters"
    # phase is the tightest thing in the archive: sub-millisecond over hours
    assert emitters[0].fraction_sd_s < 5e-3
    assert emitters[0].count > 20
    assert emitters[0].span_hours > 1.0


# --------------------------------------------------------------------------
# The consolidated file
# --------------------------------------------------------------------------

def test_cdetections_load_as_detections(make_detection_h5):
    """The 900 s consolidation is what survives on a synced archive.

    The per-detection `chirp-*.h5` are written into the ringbuffer tree and
    rotate away with it, so a census run on the archive volume -- which is
    where anyone actually runs one -- has only these.
    """
    tree = make_detection_h5("cdetections", transmit_seconds=CYPRUS_SLOTS,
                             distance_km=CYPRUS_KM, cycles=5)
    items = io_detect.load_cdetections(tree)

    assert len(items) == 15
    assert all(isinstance(d, io_detect.Detection) for d in items)
    emitters = io_detect.census(items)
    assert len(emitters) == 1
    assert tuple(emitters[0].observed_seconds) == CYPRUS_SLOTS


def test_cdetections_do_not_invent_the_fields_they_lack(make_detection_h5):
    """The (N, 5) array has no columns for these, and a plausible-looking
    zero would be worse than the sentinel `read_detection` already uses."""
    tree = make_detection_h5("cdetections")
    one = io_detect.load_cdetections(tree)[0]

    assert one.channel == ""
    assert one.i0 == -1 and one.n_samples == -1
    assert np.isnan(one.sample_rate)


def test_cdetections_still_carry_the_epoch_error(make_detection_h5):
    """The consolidation loses fields, not timing -- which is the whole point
    of being willing to read it."""
    tree = make_detection_h5("cdetections", transmit_seconds=CYPRUS_SLOTS,
                             distance_km=CYPRUS_KM, epoch_offset_s=-0.9557,
                             cycles=6)
    offset = io_detect.solve_epoch_offset(
        io_detect.load_cdetections(tree), rate=100e3,
        transmit_seconds=CYPRUS_SLOTS, distance_km=CYPRUS_KM, window_s=2.0)

    assert offset.seconds == pytest.approx(-0.9557, abs=1e-6)


def test_a_named_file_of_the_wrong_kind_is_not_offered_to_the_reader(
        make_detection_h5):
    """`muf detect` tries par, then chirp, then cdetections against the same
    target. Without this, naming one cdetections file hands it to the first
    two readers, and the caller sees two "skipped 1 unreadable file" warnings
    before the right one gets it."""
    tree = make_detection_h5("cdetections")
    named = next(tree.glob("cdetections-*.h5"))

    assert io_detect.find_cdetections(named) == [named]
    assert io_detect.find_timings(named) == []
    assert io_detect.find_detections(named) == []

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert io_detect.load_timings(named) == []
        assert io_detect.load_detections(named) == []
