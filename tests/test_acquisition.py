"""Placing a schedule against a clock, and composing one from identifications.

Two kinds of test here, and they fail for different reasons.

The **arithmetic** is tested at fixed instants because the interesting cases
are the boundaries -- the moment a slot starts, the moment it ends, and the
wrap into the next cycle -- and none of them can be reached reliably by
running at "now".

The **composition** is tested against what ``calc_ionograms.py`` actually
reads. Its rank loop subscripts five keys with no default, two of which
nothing in this repository was writing, and a schedule short of them does not
fail loudly: the rank that gets it dies at its first slot while the others
carry on and the log looks normal.
"""

from __future__ import annotations

import json

import pytest

from services.api import acquisition

#: A complete entry, the shape every `.ini` in the pinned clone's
#: `examples/marieluise` carries.
NIC = {"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0,
       "id": 1, "transmit_name": "NIC"}

#: DOB's sweep: 0.0 to 24.825 MHz. At 100 kHz/s that is 248.25 s of sounding
#: in a 300 s cycle -- comfortably less than the cycle, which is what makes
#: "is it sounding right now" a question with two answers.
SPAN_MHZ = 24.825


def at(second_of_cycle: float, cycle: float = 300.0) -> float:
    """A unix time that is exactly ``second_of_cycle`` into its cycle."""
    base = 1785888000.0                       # 2026-08-06T00:00:00Z, on a 300s
    assert base % cycle == 0
    return base + second_of_cycle


# --------------------------------------------------------------------------
# Placing one entry
# --------------------------------------------------------------------------

def test_the_slot_is_found_by_flooring_not_by_scanning():
    slot = acquisition.place(NIC, at(240.0), span_mhz=SPAN_MHZ)

    assert slot.chirpt_s == 235.0
    assert slot.started_at == at(235.0)
    assert slot.next_start == at(535.0)       # 235 of the following cycle
    assert slot.seconds_in == pytest.approx(5.0)
    assert slot.seconds_until == pytest.approx(295.0)


def test_a_slot_just_past_its_start_is_next_due_a_whole_cycle_later():
    """The wrap, which a naive ``chirpt - (now % rep)`` reports as negative."""
    slot = acquisition.place(NIC, at(236.0), span_mhz=SPAN_MHZ)
    assert slot.seconds_until == pytest.approx(299.0)
    assert slot.seconds_until > 0


def test_a_sweep_that_outlasts_its_cycle_is_still_running_in_the_next_one():
    """235 + 248.25 = 483.25, which is 183 seconds into the following cycle.

    So "which slot is it sounding" is not answered by the slot second: on this
    circuit NIC occupies 248 of every 300 seconds, and for 183 of them the
    cycle counter has already rolled over. Anything that decided in-progress
    from ``now % rep`` against ``chirpt`` would call this idle.
    """
    slot = acquisition.place(NIC, at(10.0), span_mhz=SPAN_MHZ)

    assert slot.seconds_until == pytest.approx(225.0)   # the next one
    assert slot.in_progress, "the previous cycle's sweep has 173 s to run"
    assert slot.seconds_in == pytest.approx(75.0)

    # And it does end, 183.25 s into this cycle.
    assert not acquisition.place(NIC, at(184.0), span_mhz=SPAN_MHZ).in_progress


def test_exactly_on_the_slot_second_is_the_start_not_the_end():
    slot = acquisition.place(NIC, at(235.0), span_mhz=SPAN_MHZ)
    assert slot.started_at == at(235.0)
    assert slot.seconds_in == 0.0
    assert slot.in_progress
    assert slot.seconds_until == pytest.approx(300.0)


def test_the_sweep_is_the_band_over_the_rate():
    slot = acquisition.place(NIC, at(300.0), span_mhz=SPAN_MHZ)
    assert slot.sweep_s == pytest.approx(248.25)


@pytest.mark.parametrize("into,running", [
    (235.0, True),                            # the instant it starts
    (400.0, True),                            # 165 s in, mid-sweep
    (483.0, True),                            # 248 s in, the last second
    (484.0, False),                           # 249 s in, the sweep is over
])
def test_in_progress_follows_the_sweep_length(into, running):
    assert acquisition.place(NIC, at(into), span_mhz=SPAN_MHZ).in_progress is running


def test_without_a_band_no_slot_is_called_in_progress():
    """A rig that has ingested nothing cannot time a sweep, and must say so.

    Reporting "sounding now" from the slot second alone would be a guess that
    is wrong for 51 of every 300 seconds on this circuit, and the operator has
    no way to see that it was a guess.
    """
    slot = acquisition.place(NIC, at(240.0), span_mhz=None)
    assert slot.sweep_s is None
    assert slot.in_progress is False
    assert slot.ends_at is None
    # The slot times themselves are still exact.
    assert slot.seconds_until == pytest.approx(295.0)


def test_an_entry_that_cannot_be_placed_is_dropped_not_invented():
    assert acquisition.place({"chirp-rate": 1e5, "rep": 0.0, "chirpt": 1.0},
                             at(0.0)) is None
    assert acquisition.place({"rep": 300.0, "chirpt": 1.0}, at(0.0)) is None
    assert acquisition.place({"chirp-rate": 1e5, "rep": "soon", "chirpt": 1.0},
                             at(0.0)) is None


def test_a_slot_past_the_end_of_its_own_cycle_folds_into_it():
    slot = acquisition.place({"chirp-rate": 1e5, "rep": 60.0, "chirpt": 305.0},
                             at(0.0))
    assert slot.chirpt_s == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Reading a schedule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,ranks", [
    ([NIC], 1),                                          # flat: one rank
    ([[NIC]], 1),
    ([[NIC], [dict(NIC, id=2, transmit_name="SGO")]], 2),
    (json.dumps([[NIC]]), 1),                            # as the ini stores it
    ("", 0),
    ("not json", 0),
    (None, 0),
])
def test_rank_groups_reads_every_shape_the_schedule_arrives_in(value, ranks):
    assert len(acquisition.rank_groups(value)) == ranks


def test_the_rank_grouping_is_not_flattened_away():
    """It is the launcher's ``-np``, and the only copy of that number.

    `calc_ionograms.py:422` does `st = conf.sounder_timings[rank]`. Two groups
    means two ranks; flattening them to four entries in one group would leave a
    schedule that looks the same and sounds half of it.
    """
    groups = acquisition.rank_groups(
        [[NIC, dict(NIC, chirpt=265.0)],
         [dict(NIC, id=2, transmit_name="SGO", chirpt=54.0)]])
    assert [len(g) for g in groups] == [2, 1]
    assert len(acquisition.entries(groups)) == 3


def test_describe_sorts_slots_by_rank_then_by_slot_second():
    state = acquisition.describe(
        "DOB", now=at(0.0), span_mhz=SPAN_MHZ,
        timings=[[dict(NIC, chirpt=265.0), NIC],
                 [dict(NIC, id=2, transmit_name="SGO", chirpt=54.0, rep=60.0)]])
    assert [(s.rank, s.chirpt_s) for s in state.slots] == [
        (0, 235.0), (0, 265.0), (1, 54.0)]


def test_no_schedule_is_reported_as_unknown_not_as_empty():
    """A station configured by hand has no epoch here, and that is not a fault.

    Saying "empty schedule" would read as "this station records nothing",
    which is the one failure the whole control path exists to prevent -- and
    it would be a false alarm.
    """
    state = acquisition.describe("DOB", timings=None, now=at(0.0))
    assert state.slots == []
    assert "no schedule recorded" in state.unknown


def test_the_name_on_the_entry_wins_over_a_match():
    """It is the string the station writes into the file name."""
    verified = [{"code": "SOMEONE_ELSE",
                 "timings": [{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0}]}]
    assert acquisition.name_for(NIC, verified) == "NIC"


def test_an_unnamed_entry_falls_back_to_matching_the_numbers():
    """For a schedule written before any of this existed."""
    verified = [{"code": "NIC",
                 "timings": [{"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0}]}]
    bare = {"chirp-rate": 100e3, "rep": 300.0, "chirpt": 235.0}

    assert acquisition.name_for(bare, verified) == "NIC"
    # Rate and slot both, because neither alone identifies anyone: 100 kHz/s
    # is the common rate, and a second only means something within a cycle.
    assert acquisition.name_for(dict(bare, chirpt=54.0), verified) is None
    assert acquisition.name_for(dict(bare, **{"chirp-rate": 500e3}),
                                verified) is None
    assert acquisition.name_for(bare, []) is None


# --------------------------------------------------------------------------
# Composing one
# --------------------------------------------------------------------------

def test_compose_gives_one_rank_group_per_transmitter():
    """Upstream's own arrangement: "one MPI process for each sounder"."""
    groups = acquisition.compose([
        {"code": "NIC", "timings": [NIC, dict(NIC, chirpt=265.0)]},
        {"code": "SGO", "timings": [dict(NIC, id=2, transmit_name="SGO")]},
    ])
    assert len(groups) == 2
    assert len(groups[0]) == 2, "one site's slots share a rank"
    assert [e["transmit_name"] for e in groups[0]] == ["NIC", "NIC"]


@pytest.mark.parametrize("dropped", acquisition.REQUIRED_ENTRY_KEYS)
def test_every_key_calc_ionograms_subscripts_is_required(dropped):
    entry = {k: v for k, v in NIC.items() if k != dropped}
    faults = acquisition.problems([[entry]])
    assert any(dropped in f for f in faults), faults


def test_two_transmitters_may_not_share_an_id():
    """The id is `%03d` in the product's file name.

    `lfm_ionogram-NIC-DOB-ch000-001-...h5` and
    `lfm_ionogram-SGO-DOB-ch000-001-...h5` differ, so nothing is overwritten --
    but the id is also what `ho["id"]` records, and two transmitters answering
    to one number in one archive is a join waiting to go wrong.
    """
    faults = acquisition.problems([[NIC], [dict(NIC, transmit_name="SGO")]])
    assert any("id 1 is used by both" in f for f in faults), faults


def test_an_empty_schedule_says_what_it_would_do():
    assert any("records nothing" in f for f in acquisition.problems([]))
    assert any("records nothing" in f for f in acquisition.problems([[]]))


def test_an_empty_transmit_name_is_a_fault_not_a_blank():
    faults = acquisition.problems([[dict(NIC, transmit_name="  ")]])
    assert any("transmit_name" in f for f in faults), faults


def test_every_fault_is_reported_not_just_the_first():
    """A form that reports one fault per submission costs one round trip each."""
    faults = acquisition.problems([
        [{"chirp-rate": 1e5}],
        [{"rep": 300.0, "chirpt": 1.0, "id": 9, "transmit_name": ""}],
    ])
    assert len(faults) >= 2
