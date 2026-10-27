"""The schedule checks, case for case with ionozond's tests/test_schedulecheck.cpp."""

from __future__ import annotations

import random
from dataclasses import replace

from services.agent.schedulecheck import (
    ERROR,
    INFO,
    WARNING,
    Entry,
    Settings,
    SimStation,
    between,
    check,
    simulate,
    upcoming,
)


def station(name, active, rep, chirpt, rate, dur, cf, lat, lon, fft=0) -> Entry:
    return Entry(name=name, active=active, rx=False, rep=rep, chirpt=chirpt, rate=rate,
                 dur=dur, cf=cf, lat=lat, lon=lon, fft=fft)


def receiver() -> Entry:
    return replace(station("yoshkar-ola", False, "60", "0", "50", "260", "13500",
                           "56.38", "47.53"), rx=True)


def has(issues, level, who, part) -> bool:
    return any(i.level == level and i.station == who and part in i.text for i in issues)


def errors(issues) -> list:
    return [i for i in issues if i.level == ERROR]


def schedule_of_the_station() -> list:
    return [station("Magadan", True, "600", "8", "500", "52", "16500", "60", "150.7",
                    4096),
            station("NIC1_235", False, "300", "0", "100", "235", "20000", "35", "33"),
            station("Norilsk", True, "600", "188", "500", "52", "16500", "69.2", "88",
                    4096),
            station("Sodankyla", False, "300", "121.0015", "500.0084", "50", "16000",
                    "67.37", "26.63"),
            receiver()]


def test_the_stations_schedule_has_no_errors():
    assert errors(check(schedule_of_the_station(), Settings())) == [], (
        "the station's own schedule produced errors")


def test_52_s_at_500_khz_s_is_past_the_band_for_magadan():
    assert has(check(schedule_of_the_station(), Settings()), WARNING, "Magadan",
               "wider than the"), "52 s at 500 kHz/s was not flagged past the band"


def test_norilsk_is_told_50_s_is_enough():
    assert has(check(schedule_of_the_station(), Settings()), WARNING, "Norilsk",
               "50 s is enough"), "Norilsk was not told how long a sweep fits"


def test_magadan_and_norilsk_do_not_collide():
    issues = check(schedule_of_the_station(), Settings())
    assert has(issues, INFO, "", "do not overlap"), (
        "Magadan and Norilsk were not found clear of each other")


def test_neither_magadan_nor_norilsk_loses_a_session():
    issues = check(schedule_of_the_station(), Settings())
    assert not any(has(issues, WARNING, n, "loses") for n in ("Magadan", "Norilsk")), (
        "a session was reported lost where none is")


def test_4096_points_at_500_khz_s_is_the_right_cell():
    assert not has(check(schedule_of_the_station(), Settings()), INFO, "Magadan",
                   "ionogram cell"), "4096 points at 500 kHz/s drew advice"


def test_sodankylas_odd_rate_is_not_flagged_past_the_band():
    s = schedule_of_the_station()
    s[3] = replace(s[3], active=True)
    s[0] = replace(s[0], active=False)
    s[2] = replace(s[2], active=False)
    assert not has(check(s, Settings()), WARNING, "Sodankyla", "wider than the"), (
        "50 s at 500.0084 kHz/s was flagged past the band")


def inside() -> list:
    return [station("A", True, "300", "0", "100", "250", "12500", "35", "33"),
            station("B", True, "300", "100", "100", "50", "12500", "35", "34"),
            receiver()]


def test_b_loses_every_session_to_a():
    assert has(check(inside(), Settings()), WARNING, "B", "loses 288 of 288"), (
        "B was not found to lose every session to A")


def test_b_is_told_how_short_a_must_be():
    assert has(check(inside(), Settings()), WARNING, "B", "shorten A to 80 s"), (
        "B was not told how short A must be")


def test_a_loses_none():
    assert not has(check(inside(), Settings()), WARNING, "A", "loses"), (
        "A was reported losing sessions")


def five_minutes() -> list:
    return [station("Magadan", True, "300", "0", "500", "52", "16500", "60", "150.7",
                    4096),
            station("NIC_900", True, "600", "315", "100", "240", "20000", "35.18",
                    "33.38"),
            station("Norilsk", True, "300", "180", "500", "52", "16500", "69.2", "88",
                    4096),
            receiver()]


def test_nic_900_falls_inside_magadans_capture():
    assert has(check(five_minutes(), Settings()), WARNING, "NIC_900",
               "loses 144 of 144"), "NIC_900 was not found inside Magadan's capture"


def test_the_advice_is_to_record_magadan_every_other_session():
    assert has(check(five_minutes(), Settings()), WARNING, "NIC_900",
               "every other session (period 600 s, chirptime 0)"), (
        "the advice was not to record Magadan every other session")


def test_the_advice_is_not_to_cut_a_26_mhz_sweep_to_3_mhz():
    assert not has(check(five_minutes(), Settings()), WARNING, "NIC_900", "to 6 s"), (
        "the advice cut a 26 MHz sweep to 3 MHz")


def test_a_tie_goes_to_the_first_listed_and_the_other_is_told():
    s = [station("A", True, "600", "10", "100", "100", "12500", "35", "33"),
         station("B", True, "600", "10", "100", "100", "12500", "35", "34"), receiver()]
    issues = check(s, Settings())
    assert has(issues, WARNING, "B", "loses 144 of 144") and has(
        issues, WARNING, "B", "almost together"), "a tie was not reported to B"


def sidecar() -> list:
    return [station("A", True, "600", "0", "100", "250", "12500", "35", "33"),
            station("B", True, "600", "260", "100", "50", "12500", "35", "34"),
            receiver()]


def test_ten_seconds_apart_is_not_a_certain_loss():
    assert not has(check(sidecar(), Settings()), WARNING, "B", "loses"), (
        "ten seconds apart was reported as a certain loss")


def test_ten_seconds_apart_is_at_risk():
    assert has(check(sidecar(), Settings()), WARNING, "B", "may lose up to 144"), (
        "ten seconds apart was not reported at risk")


def test_a_session_longer_than_its_period_records_every_other_one():
    s = [station("A", True, "60", "0", "100", "60", "12500", "35", "33"), receiver()]
    assert has(check(s, Settings()), WARNING, "A",
               "720 of 1440 sessions a day are recorded"), (
        "a session longer than its period was not reported")


def uncarried() -> list:
    return [station("Норильск", True, "600", "188", "500", "50", "16500", "69.2", "88",
                    4096),
            station("C", True, "0300", "0", "100", "250", "12500", "56,38", "33"),
            station("D", True, "0", "0", "100", "250", "12500", "35", "33"),
            station("E'", True, "300", "0", "100", "250", "12500", "35", "33"),
            receiver()]


def test_a_cyrillic_name_is_an_error():
    assert has(check(uncarried(), Settings()), ERROR, "Норильск", "Latin-1"), (
        "a Cyrillic name was not an error")


def test_a_leading_zero_is_an_error():
    assert has(check(uncarried(), Settings()), ERROR, "C", "leading zero"), (
        "0300 was not an error")


def test_a_decimal_comma_is_an_error():
    assert has(check(uncarried(), Settings()), ERROR, "C", "comma"), (
        "56,38 was not an error")


def test_a_zero_period_is_an_error():
    assert has(check(uncarried(), Settings()), ERROR, "D", "hangs"), (
        "rep 0 was not an error")


def test_a_quote_in_the_name_is_an_error():
    assert has(check(uncarried(), Settings()), ERROR, "E'", "quote"), (
        "a quote in the name was not an error")


def timing() -> list:
    return [station("Fast", True, "600", "0", "1000", "25", "12500", "59.56", "150.80",
                    2048),
            station("Big", True, "600", "200", "500", "50", "16000", "67.37", "26.63",
                    16384),
            station("Odd", True, "7000", "7100", "100", "250", "12500", "35", "33"),
            station("Tiny", True, "600", "400", "100", "30", "12500", "35", "33",
                    65536),
            station("Tinier", True, "600", "500", "100", "10", "12500", "35", "33",
                    65536),
            receiver(), receiver()]


def test_1000_khz_s_cannot_reach_magadans_delay():
    assert has(check(timing(), Settings()), ERROR, "Fast", "delay window"), (
        "1000 kHz/s on Magadan's path was not an error")


def test_16384_points_at_500_khz_s_draws_advice_of_4096():
    assert has(check(timing(), Settings()), INFO, "Big", "FFT of 4096"), (
        "16384 points at 500 kHz/s drew no advice of 4096")


def test_a_chirptime_past_the_period_is_a_warning():
    assert has(check(timing(), Settings()), WARNING, "Odd", "chirptime 7100"), (
        "a chirptime past the period was not a warning")


def test_a_period_that_does_not_divide_a_day_is_a_warning():
    assert has(check(timing(), Settings()), WARNING, "Odd", "does not divide"), (
        "a period that does not divide a day was not a warning")


def test_65536_points_in_30_s_is_coarse():
    assert has(check(timing(), Settings()), WARNING, "Tiny", "spectra"), (
        "65536 points in 30 s was not called coarse")


def test_65536_points_in_10_s_leaves_nothing():
    assert has(check(timing(), Settings()), ERROR, "Tinier", "remove everything"), (
        "65536 points in 10 s was not an error")


def test_two_columns_with_one_name_is_an_error():
    assert has(check(timing(), Settings()), ERROR, "yoshkar-ola", "one name"), (
        "two columns with one name was not an error")


def lone() -> list:
    return [station("A", True, "300", "0", "100", "250", "12500", "35", "33")]


def test_an_object_level_above_9_by_3_is_an_error():
    assert has(check(lone(), Settings(obj_level=28)), ERROR, "",
               "everything would be removed"), "obj_level above 9 x 3 was not an error"


def test_no_receiver_is_a_warning():
    assert has(check(lone(), Settings(obj_level=28)), WARNING, "",
               "no receiving station"), "a schedule with no receiver drew no warning"


def test_an_object_level_of_27_is_allowed():
    assert not has(check(lone(), Settings(obj_level=27)), ERROR, "",
                   "everything would be removed"), "obj_level 27 was refused"


def test_a_sample_rate_and_index_that_disagree_is_a_warning():
    issues = check(lone(), Settings(obj_level=27, sample_rate_index=2))
    assert has(issues, WARNING, "", "differently"), (
        "a sample rate disagreeing with its index drew no warning")


def test_the_3_s_lead_skips_a_chirp_too_close():
    starts = [c.start for c in simulate([SimStation(300, 1, 0, 0)], 0, 1000)]
    assert starts == [301, 601, 901], "the 3 s lead did not skip a chirp too close"


def test_the_next_capture_after_midnight_is_at_its_chirptime():
    midnight = 86400.0 * random.randint(20000, 21000)
    chirpt = random.randint(4, 290)
    first = upcoming([SimStation(300, chirpt, 250, 17)], midnight, 1)[0]
    assert first.start == midnight + chirpt, (
        "the next capture after midnight was not at its chirptime")


def test_the_next_captures_skip_the_one_being_recorded():
    starts = [c.start for c in upcoming([SimStation(300, 10, 250, 17)], 0.0, 3)]
    assert starts == [10.0, 310.0, 610.0], "the next captures were not a period apart"


def test_a_span_holds_every_capture_a_running_sounder_starts_in_it():
    midnight = 86400.0 * random.randint(20000, 21000)
    hours = random.randint(1, 23)
    starts = [c.start for c in between([SimStation(300, 10, 250, 17)], midnight,
                                       midnight + 3600 * hours)]
    assert starts == [midnight + 10 + 300 * k for k in range(12 * hours)], (
        "a span did not hold one capture a period from its chirptime")


def test_a_span_omits_the_capture_already_running_at_its_start():
    midnight = 86400.0 * random.randint(20000, 21000)
    starts = [c.start for c in between([SimStation(300, 10, 250, 17)],
                                       midnight + 100, midnight + 700)]
    assert starts == [midnight + 310, midnight + 610], (
        "a span counted a capture that began before it")


def test_a_span_follows_the_captures_before_it():
    """Two transmitters a few seconds apart: the second is never reached, because
    the first is still being recorded. Replayed from the span's own start, with no
    past, the second would look free at first."""
    midnight = 86400.0 * random.randint(20000, 21000)
    sounder = [SimStation(300, 10, 250, 17), SimStation(300, 20, 50, 5)]
    tx = {c.station for c in between(sounder, midnight + 15, midnight + 3600)}
    assert tx == {0}, "a span ignored the capture running when it began"
