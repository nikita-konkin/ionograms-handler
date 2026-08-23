"""Inline event handlers survive the trip through the HTML parser.

The bug this exists for was silent in the worst way. Jinja's ``tojson`` escapes
``<``, ``>``, ``&`` and ``'`` -- everything needed to sit safely inside a
*single*-quoted attribute -- and deliberately not ``"``. Written into a
double-quoted one it renders as::

    onclick="pick("2026-08-19")"

which the HTML parser reads as the complete attribute ``pick(`` followed by
junk. The button is in the DOM, it is styled, it depresses, and clicking it
raises ``SyntaxError: Unexpected end of input`` into a console nobody has open.
Four buttons across two pages shipped like that, including the one an operator
has to press to register an archive at all.

So this checks the property rather than the four sites: every inline handler
this server renders must still be a complete call after the parser has had it.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from services.api import db

HANDLER = re.compile(r"^on[a-z]+$")

TEMPLATES = Path(__file__).resolve().parents[1] / "services" / "api" / "templates"

#: A double-quoted inline handler, and whatever it contains up to the closing
#: quote. `re.S` because `forecast.html` wraps one across two lines.
DOUBLE_QUOTED_HANDLER = re.compile(r'\son[a-z]+="([^"]*)"', re.S)


class Handlers(HTMLParser):
    """Every inline handler on a page, as the browser will actually see it."""

    def __init__(self) -> None:
        super().__init__()
        self.found: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if HANDLER.match(name) and value:
                self.found.append((tag, name, value))


def handlers_of(markup: str) -> list[tuple[str, str, str]]:
    parser = Handlers()
    parser.feed(markup)
    return parser.found


# --------------------------------------------------------------------------
# The static check, which is the one that would have caught it
# --------------------------------------------------------------------------
#
# The rendered check below can only see buttons that a fixture happens to make
# render, and the button that shipped broken -- `use`, on an *unregistered*
# candidate folder -- needs a real archive tree on disk to appear at all. A
# guard that depends on getting the fixture right is a guard that reports
# "passed" for a page it never looked at.

@pytest.mark.parametrize("template", sorted(TEMPLATES.glob("*.html")),
                         ids=lambda p: p.name)
def test_no_double_quoted_handler_carries_a_tojson_value(template):
    for body in DOUBLE_QUOTED_HANDLER.findall(template.read_text()):
        assert "tojson" not in body, (
            f"{template.name}: `|tojson` inside a double-quoted handler.\n"
            f"  {body.strip()[:90]}\n"
            f"It does not escape `\"`, so the attribute ends at the string's "
            f"own opening quote and the browser gets a syntax error. Use "
            f"single quotes, as `sources.html` does for `data-row`.")



def seed(conn):
    """One archive and one model, so the row-level buttons actually render.

    Names carrying a quote and an apostrophe on purpose: those are the two
    characters that decide which attribute delimiter is safe, and a fixture
    of plain identifiers would pass whichever way the template was written.
    """
    conn.execute(
        "INSERT INTO archive (name, relpath, methods, enabled, added_at) "
        "VALUES (?,?,?,?,?)",
        ('2026-08-19 "night" run', "2026-08-19", "algo", 1, db.utcnow()))
    conn.execute(
        "INSERT INTO model_registry (name, param, tx, rx, origin, framework, "
        "loader, capability, artifact, sha256, features, target_src, "
        "imported_at, active) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("o'brien \"r7\"", "muf", "NIC", "DOB", "trained", "sklearn", "joblib",
         "slim", "/models/x.sav", "abc", '["a"]', "measured", db.utcnow(), 0))
    conn.commit()


PAGES = ["/ui", "/ui/archives", "/ui/forecast", "/ui/series", "/ui/soundings",
         "/ui/sources"]


@pytest.mark.parametrize("page", PAGES)
def test_every_inline_handler_is_a_complete_call(client, page):
    seed(client.app.state.db)
    response = client.get(page)
    assert response.status_code == 200

    for tag, name, value in handlers_of(response.text):
        script = value.strip().rstrip(";")
        assert script.endswith((")", "}")), (
            f"{page}: <{tag} {name}> was truncated by the HTML parser to "
            f"{script!r}. A `|tojson` value in a double-quoted attribute ends "
            f"the attribute at its own opening quote -- use single quotes.")


@pytest.mark.parametrize("page", PAGES)
def test_a_name_with_a_quote_in_it_reaches_the_handler_intact(client, page):
    """The seeded names carry both quote characters. If either delimiter
    choice were wrong, one of them would come back cut in half."""
    seed(client.app.state.db)
    text = client.get(page).text
    for tag, name, value in handlers_of(text):
        assert value.count("(") == value.count(")"), (
            f"{page}: unbalanced parentheses in <{tag} {name}>: {value!r}")


def test_the_check_would_have_caught_the_original_bug():
    """The guard is only worth having if it fails on the shape that shipped."""
    broken = '<button onclick="pick("2026-08-19")">use</button>'
    (tag, name, value), = handlers_of(broken)
    assert value == "pick("
    assert not value.rstrip(";").endswith(")")
