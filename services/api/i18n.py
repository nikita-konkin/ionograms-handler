"""Two languages for the console, with no compile step between them.

The catalogs are ordinary Python dicts in `services/api/locale/`, not gettext
`.po`/`.mo` files. `web_routes` already explains why this deployment has no
JavaScript toolchain -- "it adds an install, a lockfile and a second thing that
can fail to start" -- and a Babel workflow adds exactly that on the Python
side: a dependency, a `babel.cfg`, and a `pybabel compile` step in the image
build whose failure mode is a silently missing `.mo` and a console that reverts
to English without saying so. A dict is greppable, diffable, and importable by
a test.

What is lost is `pybabel extract`, and `tests/test_i18n.py` replaces it with
two checks that fit this project better: the two catalogs must carry identical
key sets, and rendering every page under every language must produce no lookup
miss. The second runs against what the pages actually ask for rather than what
a scanner can find, which matters here because a third of the console's text
lives inside inline `<script>` blocks.

**English is the fallback, always.** A key missing from a translation renders
the English string and is recorded in `MISSES`; it never renders the key id and
never raises. That is what makes a half-translated catalog shippable.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from markupsafe import Markup
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import Response

from .locale import en, ru

#: The language every unmarked request gets, and the fallback for every
#: missing key. Not negotiable per-request: see `LocaleMiddleware`.
DEFAULT = "en"

#: Every language the console can render, in the order the toggle shows them.
LOCALES = ("en", "ru")

#: Where a chosen language is remembered, and how it is asked for.
COOKIE = "ui_lang"
QUERY = "lang"

#: A year. The choice is a preference, not a session -- an operator who set the
#: console to Russian in March should not find it English in April because a
#: browser was closed.
COOKIE_MAX_AGE_S = 365 * 24 * 3600

CATALOGS: dict[str, dict[str, str]] = {"en": en.MESSAGES, "ru": ru.MESSAGES}
PLURALS: dict[str, dict[str, dict[str, str]]] = {"en": en.PLURALS, "ru": ru.PLURALS}

#: Keys asked for that the requested catalog did not have, as
#: ``{(lang, key): count}``. Recorded rather than raised, so a missing string
#: degrades to English on a live console and fails a test instead.
MISSES: dict[tuple[str, str], int] = {}


def default_lang() -> str:
    """The language a request with no cookie and no ``?lang=`` gets.

    Read from the environment on every call rather than captured at import, so
    a station can be brought up Russian-first with ``UI_LANG=ru`` in its
    compose environment and a test can change it without reloading the module.
    """
    wanted = os.environ.get("UI_LANG", DEFAULT).strip().lower()
    return wanted if wanted in LOCALES else DEFAULT


def resolve(asked: str | None, remembered: str | None) -> str:
    """Which language to render: the query string, then the cookie, then the env.

    An unknown code is ignored rather than refused. ``?lang=de`` on a link
    someone pasted should show the console in English, not an error page.
    """
    for candidate in (asked, remembered):
        if candidate and candidate.strip().lower() in LOCALES:
            return candidate.strip().lower()
    return default_lang()


def t(key: str, lang: str = DEFAULT, /, **values: Any) -> Markup:
    """One string, in `lang`, falling back to English and then to the key.

    ``values`` are substituted with `str.format`, so a message holds its
    placeholders as ``{station}`` rather than being built by concatenation at
    the call site. That is what lets a translation put the number, the station
    name or the unit somewhere other than where English puts it.

    **Messages may contain inline markup, and interpolated values may not.**
    Most of this console's text is explanatory prose carrying a `<b>`, a
    `<code>` or a link, and a message chopped into fragments around them cannot
    be translated at all -- Russian does not put the words in English's order.
    So the return is `Markup`: the catalog is a source file in this repository
    and is trusted exactly as far as the templates are. Everything substituted
    into it is *not* trusted, and `Markup.format` escapes it -- which matters
    because station names and folder paths reach these messages.
    """
    catalog = CATALOGS.get(lang, CATALOGS[DEFAULT])
    text = catalog.get(key)
    if text is None:
        MISSES[(lang, key)] = MISSES.get((lang, key), 0) + 1
        text = CATALOGS[DEFAULT].get(key, key)
    return Markup(text).format(**values) if values else Markup(text)


def plural(n: int | float, key: str, lang: str = DEFAULT) -> str:
    """The noun form that goes with `n`, in `lang`.

    English gets away with ``sounding(s)`` in the templates; Russian does not.
    ``1 зондирование``, ``2 зондирования`` and ``5 зондирований`` are three
    different words, and which one is right depends on the last digit and the
    last two digits of the number in front of it.

    Returns the word alone, so a template writes ``{{ n }} {{ plural(n, ... }}``
    and a message that needs the number elsewhere in the sentence passes it
    through `t` as a placeholder instead.
    """
    forms = PLURALS.get(lang, {}).get(key) or PLURALS[DEFAULT].get(key)
    if not forms:
        MISSES[(lang, f"plural:{key}")] = MISSES.get((lang, f"plural:{key}"), 0) + 1
        return key
    return forms.get(_form(n, lang), forms.get("other", key))


def _form(n: int | float, lang: str) -> str:
    """Which plural category `n` falls into: ``one``, ``few``, ``many``, ``other``."""
    try:
        n = abs(int(n))
    except (TypeError, ValueError):
        return "other"
    if lang != "ru":
        return "one" if n == 1 else "other"
    # The Slavic rule, and the reason `few` and `many` both exist: 2-4 take the
    # genitive singular, everything else the genitive plural, except in the
    # teens, where 11-14 are `many` despite ending in 1-4.
    if n % 10 == 1 and n % 100 != 11:
        return "one"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "few"
    return "many"


def js_catalog(lang: str) -> dict[str, Any]:
    """What the browser needs to build its own strings.

    Every ``*.js.*`` key plus the plural forms, sent as one JSON block by
    `base.html`. The whole set rather than the current page's slice: it is
    about 12 KB of text before the gzip that `main` already applies to
    everything over 1 KB, and a per-page split needs each template to declare
    its own name, which is one more thing to forget when a string moves.
    """
    english = CATALOGS[DEFAULT]
    catalog = CATALOGS.get(lang, english)
    keys = [key for key in english if ".js." in key]
    return {
        "messages": {key: catalog.get(key, english[key]) for key in keys},
        "plurals": {key: PLURALS.get(lang, {}).get(key) or forms
                    for key, forms in PLURALS[DEFAULT].items()},
        "lang": lang,
    }


def lang_for(request: Request) -> str:
    """The language this request is being rendered in.

    Reads what `LocaleMiddleware` decided. Falls back to deciding again from
    the request itself so that rendering a template outside the middleware --
    which is how several tests reach these pages -- still picks a language
    rather than raising.
    """
    found = getattr(request.state, "lang", None)
    if found in LOCALES:
        return found
    return resolve(request.query_params.get(QUERY), request.cookies.get(COOKIE))


def switch_urls(request: Request) -> dict[str, str]:
    """Where each language's toggle link points: this page, in that language.

    The whole query string is carried across, so switching language while
    looking at a filtered and sorted table does not also reset the filter and
    the sort. Path and query only -- an absolute URL built from the request
    would carry a host a reverse proxy may have rewritten.
    """
    out: dict[str, str] = {}
    for code in LOCALES:
        url = request.url.include_query_params(**{QUERY: code})
        out[code] = f"{url.path}?{url.query}" if url.query else url.path
    return out


def context(request: Request) -> dict[str, Any]:
    """Jinja context processor: `t`, `plural` and `lang`, bound to this request.

    Registered on the `Jinja2Templates` instance in `web_routes`, so no route
    function and no context dict has to carry a language around. `t` and
    `plural` arrive already bound, which means a template writes
    ``t('nav.series')`` and cannot accidentally render one page in two
    languages by forgetting an argument.
    """
    lang = lang_for(request)
    bound_t: Callable[..., str] = lambda key, **values: t(key, lang, **values)  # noqa: E731
    bound_plural: Callable[..., str] = lambda n, key: plural(n, key, lang)  # noqa: E731
    return {
        "lang": lang,
        "t": bound_t,
        "plural": bound_plural,
        "js_catalog": js_catalog(lang),
        "locales": LOCALES,
        "lang_urls": switch_urls(request),
    }


class LocaleMiddleware:
    """Decide the language before routing, and remember an explicit choice.

    Plain ASGI rather than `@app.middleware("http")`: that decorator wraps
    `BaseHTTPMiddleware`, which buffers responses through an anyio stream, and
    this app streams rendered ionograms and a 1 MB plotting bundle through the
    same stack.

    **No `Accept-Language`.** The browser's own setting is not consulted, on
    purpose: the same URL has to render the same page for everyone, or a
    screenshot in a bug report stops being evidence of what the operator saw.
    Language is a choice made once on the toggle and kept in a cookie, or a
    deployment-wide `UI_LANG`.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        asked = request.query_params.get(QUERY)
        lang = resolve(asked, request.cookies.get(COOKIE))
        scope.setdefault("state", {})["lang"] = lang

        # The cookie is written only when a language was actually asked for.
        # Setting it on every response would mean every reply from this server
        # carries a Set-Cookie, including the agent's health posts.
        if not (asked and asked.strip().lower() in LOCALES):
            await self.app(scope, receive, send)
            return

        cookie = _cookie_header(lang)

        async def send_with_cookie(message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("set-cookie", cookie)
            await send(message)

        await self.app(scope, receive, send_with_cookie)


def _cookie_header(lang: str) -> str:
    """The `Set-Cookie` value, built by Starlette so the attributes are right."""
    carrier = Response()
    carrier.set_cookie(COOKIE, lang, max_age=COOKIE_MAX_AGE_S,
                       path="/", samesite="lax")
    return carrier.headers["set-cookie"]
