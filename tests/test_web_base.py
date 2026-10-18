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
