"""The frame every page shares: its header, and the theme it is drawn in."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("fastapi")

PAGES = ["/ui", "/ui/live?station=YOS", "/ui/soundings", "/ui/sources", "/ui/archives"]


THEMES = ("auto", "light", "plots", "grey", "dim", "dark")


def test_every_page_names_every_theme_its_switch_steps_through(client):
    text = client.get(random.choice(PAGES)).text
    assert all("data-label-%s=" % k in text for k in THEMES), (
        "a page does not name every theme its switch steps through")


def test_the_theme_switch_steps_through_the_themes_in_order(client):
    text = client.get(random.choice(PAGES)).text
    assert 'data-themes="%s"' % " ".join(THEMES) in text, (
        "the theme switch does not step through the themes in order")


def test_a_remembered_theme_of_any_kind_is_accepted(client):
    head = client.get(random.choice(PAGES)).text.split("<body>")[0]
    assert "['light', 'plots', 'grey', 'dim', 'dark'].includes(theme)" in head, (
        "a remembered theme other than light or dark is ignored")


def test_the_light_theme_with_dark_plots_keeps_its_plots_dark(client):
    text = client.get(random.choice(PAGES)).text
    assert ':root[data-theme="plots"] { --plot:#14161a;' in text, (
        "the light theme with dark plots does not darken its plots")


def test_a_remembered_theme_is_applied_before_the_page_draws(client):
    head = client.get(random.choice(PAGES)).text.split("<body>")[0]
    assert "localStorage.getItem('theme')" in head, (
        "a remembered theme is not applied before the page draws")


def test_a_chosen_light_theme_wins_over_a_dark_system(client):
    assert ':root[data-theme="light"]' in client.get(random.choice(PAGES)).text, (
        "a chosen light theme does not override the system's dark one")


def test_a_chosen_dark_theme_wins_over_a_light_system(client):
    assert ':root:not([data-theme])' in client.get(random.choice(PAGES)).text, (
        "a chosen dark theme does not override the system's light one")


# --------------------------------------------------------------------------
# Times in UTC and Moscow
# --------------------------------------------------------------------------

def test_a_stamp_is_shown_in_utc_and_moscow():
    from services.api.web_routes import _msk

    assert _msk("2026-10-29 06:45:26", lang="ru") == (
        "2026-10-29 06:45:26 UTC · 09:45:26 МСК")
    assert _msk("2026-10-29 06:45Z", lang="en") == "2026-10-29 06:45 UTC · 09:45 MSK"


def test_moscow_names_its_own_date_after_utc_midnight():
    from services.api.web_routes import _msk

    assert _msk("2026-10-29T22:10:05Z", lang="ru") == (
        "2026-10-29 22:10:05 UTC · 2026-10-30 01:10:05 МСК")


def test_a_sliced_stamp_keeps_its_slice_on_both_sides():
    from services.api.web_routes import _msk

    assert _msk("2026-10-29T06:45:26.5+00:00", 11, 19, "ru") == (
        "06:45:26 UTC · 09:45:26 МСК")
    assert _msk("2026-10-29 06:45:26", 0, 16, "ru") == (
        "2026-10-29 06:45 UTC · 09:45 МСК")


def test_a_stamp_in_another_zone_is_converted_first():
    from services.api.web_routes import _msk

    assert _msk("2026-08-13T21:12:05+03:00", lang="ru") == (
        "2026-08-13 18:12:05 UTC · 21:12:05 МСК")


def test_what_is_not_a_stamp_is_left_as_it_was():
    from services.api.web_routes import _msk

    for value in (None, "", "garbage", "06:45:26", "2026-10-29", 42):
        assert _msk(value) == value
