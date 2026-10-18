"""The frame every page shares: its header, and the theme it is drawn in."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("fastapi")

PAGES = ["/ui", "/ui/live?station=YOS", "/ui/soundings", "/ui/sources", "/ui/archives"]


def test_every_page_offers_the_three_themes(client):
    text = client.get(random.choice(PAGES)).text
    assert all('data-theme-set="%s"' % k in text for k in ("auto", "light", "dark")), (
        "a page does not offer the auto, light and dark themes")


def test_a_remembered_theme_is_applied_before_the_page_draws(client):
    head = client.get(random.choice(PAGES)).text.split("<body>")[0]
    assert "localStorage.getItem('theme')" in head, (
        "a remembered theme is not applied before the page draws")


def test_a_chosen_light_theme_wins_over_a_dark_system(client):
    assert ':root[data-theme="light"]' in client.get(random.choice(PAGES)).text, (
        "a chosen light theme does not override the system's dark one")


def test_a_chosen_dark_theme_wins_over_a_light_system(client):
    assert ':root:not([data-theme="dark"])' in client.get(random.choice(PAGES)).text, (
        "a chosen dark theme does not override the system's light one")
