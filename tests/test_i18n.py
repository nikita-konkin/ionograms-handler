"""Two languages, and the two things that keep them honest.

The console's strings live in `services/api/locale/{en,ru}.py` as plain dicts
rather than gettext catalogs, so there is no `pybabel extract` to tell anyone a
string was added and never translated. These tests are what replaces it:

* **parity** -- the two catalogs must hold identical key sets, so a Russian
  translation cannot quietly fall behind an edited English one;
* **no misses** -- rendering every page in every language must not ask for a
  key that does not exist, which catches the strings a scanner would miss
  because they are built inside inline `<script>` blocks.

The third check is the one that protects everything already written: with no
cookie and no ``?lang=``, the console is still English. Around a hundred
assertions elsewhere in this suite read English text off these pages, and they
stay meaningful only while that is true.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient          # noqa: E402

from services.api import auth, db, i18n, main, net  # noqa: E402
from services.api import series as series_mod       # noqa: E402
from services.api.locale import en, ru              # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The api with no archive, no network and no model -- see `test_api.client`.

    Deliberately a copy rather than an import: these tests are about what the
    pages *say*, and pinning them to another module's fixture would mean an
    unrelated change there could turn this file green while the console was
    broken.
    """
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setenv("API_DB", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    net.reset()
    monkeypatch.setattr(series_mod, "MODEL", False)
    series_mod.clear()
    monkeypatch.delenv("UI_LANG", raising=False)
    with TestClient(main.app) as c:
        yield c


#: Every page a browser can reach that renders without any data.
PAGES = ["/ui", "/ui/series", "/ui/soundings", "/ui/sources",
         "/ui/archives", "/ui/forecast"]

TEMPLATES = Path(__file__).resolve().parent.parent / "services/api/templates"


# --------------------------------------------------------------------------
# The catalogs
# --------------------------------------------------------------------------

def test_the_two_catalogs_carry_the_same_keys():
    """A key in one language and not the other is the failure this arrangement
    is most exposed to: nothing on a live console announces it, because the
    missing side silently renders English."""
    missing_ru = sorted(set(en.MESSAGES) - set(ru.MESSAGES))
    missing_en = sorted(set(ru.MESSAGES) - set(en.MESSAGES))
    assert not missing_ru, f"never translated to Russian: {missing_ru}"
    assert not missing_en, f"in ru.py but not in en.py: {missing_en}"


def test_the_plural_tables_carry_the_same_nouns():
    assert set(en.PLURALS) == set(ru.PLURALS)


def test_every_russian_noun_declares_all_three_forms():
    """`few` and `many` are separate words, not a stylistic choice."""
    for noun, forms in ru.PLURALS.items():
        assert set(forms) == {"one", "few", "many"}, noun


#: Russian entries that are deliberately identical to their English.
#:
#: Two legitimate reasons, and nothing else. Anything that lands here without
#: being one of them is a line somebody pasted and forgot, which is invisible
#: on a live console because it renders perfectly well.
DELIBERATELY_SAME = {
    # Latin domain abbreviations, kept Latin by decision so that what is on
    # screen matches the plot axes, the database columns and the SAO records.
    "nav.api", "soundings.filter.tx", "common.col.circuit",
    "sounding.col.muf", "sounding.col.lof", "sounding.show_marks",
    # Statistical symbols, which are symbols in both languages: n is a count,
    # r is a correlation coefficient.
    "series.col.n", "series.col.r", "sources.col.n", "sources.col.snr",
    # Error metrics named by their acronyms on both plot axes and in both
    # languages' literature. Translating them would put a Cyrillic label on a
    # number an operator compares against the leaderboard's Latin one.
    "model.mae", "model.rmse",
    "series.trace.muf", "series.trace.lof", "sources.js.col.id",
    # Nothing in them but placeholders, punctuation and markup.
    "archives.day_folders", "archives.js.result", "archives.js.why",
    "console.js.sending", "series.js.trace.forecast_compare",
}

#: The handful of English messages that legitimately name a plural slot,
#: because English already pluralised them properly by hand before any catalog
#: existed. Everywhere else English keeps its `(s)`, which is what stops this
#: change from altering a single rendered English page.
ENGLISH_PLURALISED = {"archives.day_folders", "archives.js.loaded"}

#: Placeholders a translation may use where English does not.
#:
#: English gets away with `point(s)`; Russian needs one of three words chosen
#: by the number in front of it. The template passes both the count and the
#: plural form to every such message, and each language uses what it needs --
#: which is why the two catalogs legitimately differ in which placeholders they
#: name, and why English keeps rendering byte-for-byte what it rendered before
#: any of this existed.
PLURAL_SLOTS = {"unit", "trace_unit", "point_unit", "model_unit",
                "circuit_unit", "pair_unit", "day_unit", "file_unit"}


def test_no_message_is_left_as_its_english_self_by_accident():
    """A Russian entry identical to the English one is either a real loanword
    or an untranslated line someone pasted. Both exist, so this pins the
    allowed set rather than banning the case."""
    same = {key for key, text in ru.MESSAGES.items()
            if text == en.MESSAGES.get(key)}
    assert same <= DELIBERATELY_SAME, f"looks untranslated: {sorted(same - DELIBERATELY_SAME)}"
    stale = DELIBERATELY_SAME - same
    assert not stale, f"translated after all -- drop from the allowlist: {sorted(stale)}"


def test_a_translation_only_adds_placeholders_the_template_already_passes():
    """The one asymmetry the catalogs are allowed, and its limit.

    A Russian message may reach for a plural form English does not need. It may
    not reach for anything else: every other placeholder has to be one English
    already names, because that is the set the template is known to pass and a
    name outside it is a `KeyError` at render time on whichever page that
    message lives.
    """
    import re
    slot = re.compile(r"\{(\w+)\}")
    for key, text in ru.MESSAGES.items():
        extra = set(slot.findall(text)) - set(slot.findall(en.MESSAGES[key]))
        assert extra <= PLURAL_SLOTS, f"{key}: unknown placeholder {sorted(extra)}"


def test_english_still_needs_no_plural_form_at_the_count_sites():
    """The counterpart of the rule above, stated from the other side.

    If an English message ever starts naming a plural slot, its rendering has
    changed -- and around a dozen assertions elsewhere in this suite grep those
    exact strings, `(s)` and all.
    """
    import re
    slot = re.compile(r"\{(\w+)\}")
    for key, text in en.MESSAGES.items():
        named = set(slot.findall(text)) & PLURAL_SLOTS
        assert not named or key in ENGLISH_PLURALISED, key


def test_a_message_may_carry_markup_but_a_value_substituted_into_it_may_not():
    """The catalog is repository source and is trusted like a template. A
    station name, a folder path or a model name is not, and `Markup.format`
    escapes it -- which is the whole reason `t` returns `Markup` rather than
    the templates marking each message `|safe` at the call site."""
    rendered = i18n.t("sounding.scaling_failed", "en",
                      url='" onerror="alert(1)')
    assert "<a href=" in rendered, "the message's own markup survives"
    assert 'onerror="alert(1)' not in rendered, "the value did not"
    assert "&#34;" in rendered or "&quot;" in rendered


def test_the_strings_javascript_writes_carry_no_markup():
    """`*.js.*` messages are written into the DOM with `textContent`, so a tag
    in one of them shows up on screen as a tag rather than as formatting."""
    for catalog in (en.MESSAGES, ru.MESSAGES):
        for key, text in catalog.items():
            if ".js." in key:
                assert "<" not in text and "&" not in text, key


def test_no_template_shadows_the_translation_helpers():
    """`t` is a one-letter global, and Jinja will happily let a loop bury it.

    `console.html` had `{% for t in s.verified %}` around a row that then
    called `t('console.khz_s')` and got `'dict' object is not callable`. That
    one failed loudly because the row happened to need a string; a loop that
    shadows `t` and calls nothing simply renders, and the next person to add a
    string inside it gets the error instead.
    """
    import re
    declare = re.compile(r"\{%-?\s*(?:for|set)\s+([\w,\s]+?)\s*(?:in|=)")
    clashes = []
    for template in sorted(TEMPLATES.glob("*.html")):
        for names in declare.findall(template.read_text()):
            for name in (n.strip() for n in names.split(",")):
                if name in {"t", "plural", "lang", "locales", "lang_urls"}:
                    clashes.append(f"{template.name}: {name}")
    assert not clashes, f"shadows an i18n helper: {clashes}"


def test_the_russian_catalog_is_actually_russian():
    """Catches the character that got in by accident.

    Writing several thousand words of Cyrillic turns up exactly one class of
    typo that no reviewer reliably sees and no other test would: a character
    from a third script, sitting inside a word, rendering as a box. This found
    a CJK ideograph in `sources.identify_note` on the first run.
    """
    # Greek letters are mathematical notation in either language: σ is the
    # tracker's standard deviation and Δ is a difference, neither is a word.
    # ✗ and ⚠ are status glyphs. (This comment was here twice until
    # 2026-08-27, which is its own small argument for reading test files.)
    allowed = set("—–…‑ «»±≥≤−→σΔ✗⚠")
    strays = {}
    for key, text in ru.MESSAGES.items():
        for ch in text:
            code = ord(ch)
            if code > 0x24F and not (0x400 <= code <= 0x4FF) and ch not in allowed:
                strays.setdefault(key, set()).add(f"{ch!r} U+{code:04X}")
    assert not strays, f"not Cyrillic, not punctuation: {strays}"


def test_no_message_smuggles_a_script_into_a_page():
    """Messages render unescaped, so the catalogs are held to the same rule a
    template is: no script, no handler attribute, no javascript: URL."""
    import re
    bad = re.compile(r"<script|javascript:|\son\w+\s*=", re.I)
    for catalog in (en.MESSAGES, ru.MESSAGES):
        for key, text in catalog.items():
            assert not bad.search(text), key


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------

def test_a_missing_key_renders_english_and_is_recorded():
    """The half-translated state has to be shippable: a key not yet in `ru.py`
    shows the English string, not the key id and not a 500."""
    i18n.MISSES.clear()
    try:
        i18n.CATALOGS["ru"].pop("common.apply")
        assert i18n.t("common.apply", "ru") == "Apply"
        assert ("ru", "common.apply") in i18n.MISSES
    finally:
        i18n.CATALOGS["ru"]["common.apply"] = ru.MESSAGES.get(
            "common.apply", "Применить")
        i18n.MISSES.clear()


def test_an_unknown_key_renders_as_itself():
    i18n.MISSES.clear()
    assert i18n.t("no.such.key", "en") == "no.such.key"
    assert ("en", "no.such.key") in i18n.MISSES
    i18n.MISSES.clear()


def test_placeholders_are_named_so_a_translation_can_move_them():
    assert i18n.t("common.pager", "en", first=1, last=20, total=99) == "1–20 of 99"
    assert i18n.t("common.pager", "ru", first=1, last=20, total=99) == "1–20 из 99"


@pytest.mark.parametrize("n,expected", [
    (1, "зондирование"), (2, "зондирования"), (4, "зондирования"),
    (5, "зондирований"), (11, "зондирований"), (12, "зондирований"),
    (14, "зондирований"), (21, "зондирование"), (22, "зондирования"),
    (25, "зондирований"), (101, "зондирование"), (111, "зондирований"),
    (0, "зондирований"),
])
def test_russian_picks_the_form_from_the_last_two_digits(n, expected):
    """21 goes with the same word as 1; 11 does not. That is the whole rule,
    and it is why `(s)` cannot survive translation."""
    assert i18n.plural(n, "sounding", "ru") == expected


@pytest.mark.parametrize("n,expected", [(1, "sounding"), (0, "soundings"),
                                        (2, "soundings"), (21, "soundings")])
def test_english_picks_between_two(n, expected):
    assert i18n.plural(n, "sounding", "en") == expected


# --------------------------------------------------------------------------
# Choosing a language
# --------------------------------------------------------------------------

def test_a_fresh_browser_gets_english(client):
    """The guard for every other test in this suite that reads English off a
    page. If this fails, around a hundred assertions elsewhere have stopped
    testing what they were written to test."""
    page = client.get("/ui/soundings").text
    assert 'lang="en"' in page
    assert ">Soundings<" in page
    assert "Зондирования" not in page


def test_asking_for_russian_renders_it_and_remembers(client):
    page = client.get("/ui/soundings?lang=ru")
    assert 'lang="ru"' in page.text
    assert "Зондирования" in page.text
    assert client.cookies.get(i18n.COOKIE) == "ru"

    # The cookie alone, with nothing in the query string.
    assert "Зондирования" in client.get("/ui/soundings").text

    # And back again, which must also update the cookie -- a toggle that can
    # only be moved one way is not a toggle.
    back = client.get("/ui/soundings?lang=en")
    assert 'lang="en"' in back.text
    assert client.cookies.get(i18n.COOKIE) == "en"


def test_an_unknown_language_is_ignored_rather_than_refused(client):
    """A pasted link with `?lang=de` should show the console, not an error."""
    page = client.get("/ui/soundings?lang=de")
    assert page.status_code == 200
    assert 'lang="en"' in page.text
    assert i18n.COOKIE not in client.cookies


def test_only_an_explicit_choice_writes_the_cookie(client):
    """Every response carrying a Set-Cookie would include the station agent's
    health posts, which have no browser and no language."""
    assert "set-cookie" not in client.get("/ui/soundings").headers
    assert "set-cookie" in client.get("/ui/soundings?lang=ru").headers


def test_the_deployment_default_comes_from_the_environment(client, monkeypatch):
    """`UI_LANG=ru` in compose, so a station boots Russian-first without anyone
    touching a browser."""
    monkeypatch.setenv("UI_LANG", "ru")
    assert 'lang="ru"' in client.get("/ui/soundings").text
    monkeypatch.setenv("UI_LANG", "klingon")
    assert 'lang="en"' in client.get("/ui/soundings").text


def test_the_toggle_keeps_the_query_string(client):
    """Changing language on a filtered, sorted table must not also reset the
    filter and the sort."""
    page = client.get("/ui/soundings?sort=tx&dir=desc&picks=some").text
    assert "/ui/soundings?sort=tx&amp;dir=desc&amp;picks=some&amp;lang=ru" in page


# --------------------------------------------------------------------------
# The sweep that replaces `pybabel extract`
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lang", i18n.LOCALES)
def test_no_page_asks_for_a_string_that_does_not_exist(client, lang):
    """Run against what the pages actually request, not against what a scanner
    can find in the templates -- which matters here because a third of this
    console's text is built inside inline `<script>` blocks."""
    i18n.MISSES.clear()
    for path in PAGES:
        assert client.get(f"{path}?lang={lang}").status_code == 200, path
    assert not i18n.MISSES, f"missing strings: {sorted(i18n.MISSES)}"


#: Every key named by a literal `t('...')` or `T('...')` call in a template.
def _keys_in_templates() -> dict[str, set[str]]:
    import re
    found: dict[str, set[str]] = {}
    call = re.compile(r"\bT?t\(\s*'([a-z][a-z0-9_.]*)'|\bT\(\s*'([a-z][a-z0-9_.]*)'")
    # Comments first. `base.html` explains the helper by showing a call, and a
    # key named in prose is not a key the page asks for.
    comment = re.compile(r"\{#.*?#\}|/\*.*?\*/", re.S)
    for template in sorted(TEMPLATES.glob("*.html")):
        body = comment.sub(" ", template.read_text())
        for one, two in call.findall(body):
            found.setdefault(one or two, set()).add(template.name)
    return found


def test_every_key_a_template_names_actually_exists():
    """The static half of the guard.

    The render sweep can only see the branches a data-less fixture reaches --
    it never opens a sounding page, and it sees the forecast tables empty. This
    reads the templates instead, so a key misspelled inside an `{% if %}` that
    only fires on real data is caught here rather than on the console.

    Parity does the rest: a key that exists in English exists in Russian.
    """
    unknown = {key: sorted(where) for key, where in _keys_in_templates().items()
               if key not in en.MESSAGES}
    assert not unknown, f"named by a template, missing from the catalog: {unknown}"


def test_no_key_is_carried_that_nothing_uses():
    """Dead strings are worse than missing ones: they get translated, reviewed
    and maintained for a page that stopped showing them."""
    import re
    used = set(_keys_in_templates())
    # Keys the Python side reaches for by name rather than a template: the
    # duration and lead filters compose `unit.*`, and `i18n` itself has none.
    source = (Path("services/api/web_routes.py").read_text()
              + Path("services/api/i18n.py").read_text())
    used |= set(re.findall(r"['\"]([a-z][a-z0-9_.]*\.[a-z0-9_.]+)['\"]", source))
    used |= {f"unit.{u}" for u in ("s", "m", "h", "d")}
    orphans = sorted(set(en.MESSAGES) - used)
    assert not orphans, f"in the catalog, used by nothing: {orphans}"


def test_the_browser_gets_the_javascript_half_of_the_catalog(client):
    """`base.html` ships the `*.js.*` keys and the plural forms as JSON, so the
    strings JavaScript writes come from the same catalog the templates use."""
    catalog = i18n.js_catalog("ru")
    assert catalog["lang"] == "ru"
    assert catalog["plurals"]["sounding"]["few"] == "зондирования"
    assert all(".js." in key for key in catalog["messages"])

    # Inlined into the helper script, not carried in its own JSON block: see
    # the note in `base.html`. A block would shadow `series-frame` and
    # `sao-frame` for anything that reads the first JSON payload on a page.
    page = client.get("/ui/soundings").text
    assert "const I18N = {" in page
    assert '<script id="i18n-catalog"' not in page


#: Keys whose value is appended to something already rendered -- a count, a
#: list, a timestamp -- rather than starting its own label, sentence or cell.
#: They are the one place the sentence-case rule inverts: a capital here lands
#: in the middle of a phrase.
#:
#: This has now bitten three times. `forecast.of` follows
#: `{{ m.features|length }}` and rendered "18 Of muf" on the live page;
#: `sources.js.of_rep` follows a list of chirp offsets and rendered
#: "5s, 10s Of 300s". Both were introduced by the capitalisation sweep, which
#: could not see where a string lands.
#:
#: `console.in` and `console.from_products` are deliberately NOT here: each is
#: the whole contents of its table cell, so it starts a phrase and keeps its
#: capital.
MID_PHRASE = {
    "forecast.of",
    "sources.js.of_rep",
}


@pytest.mark.parametrize("lang", i18n.LOCALES)
@pytest.mark.parametrize("key", sorted(MID_PHRASE))
def test_a_mid_phrase_fragment_does_not_start_with_a_capital(key, lang):
    value = i18n.CATALOGS[lang][key]
    first = value.lstrip()[0]
    assert first == first.lower(), (
        f"{lang}:{key} = {value!r} starts with a capital, but it is appended "
        f"to a value already on the page -- it renders mid-phrase. "
        f"See MID_PHRASE.")


def test_the_mid_phrase_list_has_no_stale_entries():
    """A key removed from the catalog should leave this list, not sit in it."""
    missing = sorted(MID_PHRASE - set(en.MESSAGES))
    assert not missing, f"MID_PHRASE names keys that no longer exist: {missing}"


def test_every_key_the_browser_looks_up_is_shipped_to_the_browser():
    """`T('...')` in a template must name a key `js_catalog` actually sends.

    Two different lookups live in these files and they are not
    interchangeable. `{{ t('k') }}` is resolved by Jinja at render time and may
    name any key; `T('k')` is resolved by JavaScript against the catalog
    inlined in `base.html`, which carries only the `*.js.*` keys -- so a
    non-`js` key handed to `T` silently renders as the key itself.

    This is not hypothetical. `series.trace.forecast` and
    `series.trace.forecast_band` were written that way during the bilingual
    release and shipped broken: the plot legend read
    "series.trace.forecast_band" verbatim. Nobody saw it because no forecast
    had ever been drawn on that page, and the first one that was drew it in
    both languages.
    """
    import re

    shipped = set(i18n.js_catalog("en")["messages"])
    wrong: list[str] = []
    for path in sorted(TEMPLATES.glob("*.html")):
        # Comments first. `base.html` documents the helper with a literal
        # `T('key', ...)` in prose, which is an example and not a call --
        # exactly the false positive the template-key scanner above strips for.
        source = re.sub(r"\{#.*?#\}|/\*.*?\*/|//[^\n]*", " ",
                        path.read_text(encoding="utf-8"), flags=re.S)
        for key in re.findall(r"\bT\(\s*['\"]([a-z][a-z0-9_.]*)['\"]", source):
            if key not in shipped:
                wrong.append(f"{path.name}: T({key!r})")
    assert not wrong, (
        "handed to the browser's T() but not in the js catalog, so it renders "
        "as the bare key: " + ", ".join(sorted(wrong)))
