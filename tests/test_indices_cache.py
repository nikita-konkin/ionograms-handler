"""What a failed download is allowed to do to the cache: nothing.

A station lost its daily F10.7 this way. SWPC answered a request with an empty
body -- a 200, not an error, so nothing raised -- and the empty string was
written straight over 42 days of good data. `cache_state` then reported the
file as freshly fetched, because the write that destroyed it had updated the
mtime it reads. The only visible symptom was the IRI trace on the series page
stopping two days short, explained by one line of prose below the chart.

The rule these pin: **a cache entry is only ever replaced by something better
than what is there.**
"""

from __future__ import annotations

import datetime as dt
import urllib.error
from pathlib import Path

import pytest

from muf.reference import indices

GOOD = "a real document\n" * 40          # comfortably over MIN_USEFUL_BYTES
BETTER = "a newer document\n" * 40


@pytest.fixture
def cache(tmp_path) -> Path:
    root = tmp_path / "indices"
    root.mkdir()
    return root


def serve(monkeypatch, body: str | None = None, error: Exception | None = None,
          status_code: int = 200):
    """Make the next fetch return `body`, or raise `error`."""
    class Response:
        status = status_code
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self): return body.encode()
        def getcode(self): return status_code

    def urlopen(request, timeout=None):
        if error is not None:
            raise error
        return Response()

    monkeypatch.setattr(indices.urllib.request, "urlopen", urlopen)


def test_an_empty_response_does_not_destroy_a_good_cache(cache, monkeypatch):
    """The bug, exactly. An empty 200 raises nothing and used to be written."""
    cached = cache / "f.json"
    cached.write_text(GOOD)
    # Old enough that the fetch is attempted rather than served from cache.
    old = dt.datetime.now().timestamp() - 86400 * 30
    import os
    os.utime(cached, (old, old))

    serve(monkeypatch, body="")
    assert indices._fetch("http://x/f.json", cache, "f.json") == GOOD
    assert cached.read_text() == GOOD


def test_an_empty_response_with_no_cache_is_an_error_not_an_empty_file(
        cache, monkeypatch):
    serve(monkeypatch, body="")
    with pytest.raises(indices.IndexUnavailable, match="not a document"):
        indices._fetch("http://x/f.json", cache, "f.json")
    assert not (cache / "f.json").exists()


def test_a_good_response_does_replace_the_cache(cache, monkeypatch):
    """The guard must not also block the thing the cache is for."""
    cached = cache / "f.json"
    cached.write_text(GOOD)
    old = dt.datetime.now().timestamp() - 86400 * 30
    import os
    os.utime(cached, (old, old))

    serve(monkeypatch, body=BETTER)
    assert indices._fetch("http://x/f.json", cache, "f.json") == BETTER
    assert cached.read_text() == BETTER


def test_a_network_error_still_falls_back_to_a_usable_cache(cache, monkeypatch):
    cached = cache / "f.json"
    cached.write_text(GOOD)
    old = dt.datetime.now().timestamp() - 86400 * 30
    import os
    os.utime(cached, (old, old))

    serve(monkeypatch, error=urllib.error.URLError("down"))
    assert indices._fetch("http://x/f.json", cache, "f.json") == GOOD


def test_an_empty_cached_file_is_not_a_cache_hit(cache, monkeypatch):
    """It is the wreckage of an earlier failure, and reading it as a hit is
    what stopped the retry from ever retrying."""
    (cache / "f.json").write_text("")
    serve(monkeypatch, body=GOOD)
    assert indices._fetch("http://x/f.json", cache, "f.json") == GOOD
    assert (cache / "f.json").read_text() == GOOD


def test_an_empty_cached_file_is_not_a_fallback_either(cache, monkeypatch):
    (cache / "f.json").write_text("")
    serve(monkeypatch, error=urllib.error.URLError("down"))
    with pytest.raises(indices.IndexUnavailable, match="could not fetch"):
        indices._fetch("http://x/f.json", cache, "f.json")


def test_offline_will_not_serve_an_empty_file(cache):
    (cache / "f.json").write_text("")
    with pytest.raises(indices.IndexUnavailable, match="offline"):
        indices._fetch("http://x/f.json", cache, "f.json", offline=True)


def test_the_write_is_atomic(cache, monkeypatch):
    """A fetch interrupted half way must leave the previous copy intact, not a
    truncated one that would pass every check above."""
    cached = cache / "f.json"
    cached.write_text(GOOD)
    old = dt.datetime.now().timestamp() - 86400 * 30
    import os
    os.utime(cached, (old, old))

    def die(self, target):
        raise OSError("interrupted")

    monkeypatch.setattr(Path, "replace", die)
    serve(monkeypatch, body=BETTER)
    # Warns rather than raises -- the document was fetched and the caller can
    # have it; only the caching failed. What matters here is the file.
    with pytest.warns(UserWarning, match="could not cache"):
        assert indices._fetch("http://x/f.json", cache, "f.json") == BETTER
    assert cached.read_text() == GOOD


# --------------------------------------------------------------------------
# The indicator has to agree with reality
# --------------------------------------------------------------------------

def test_cache_state_reports_an_empty_file_as_absent(cache):
    """It reported one as *fresh* for two days, because the write that
    destroyed it updated the mtime `cache_state` reads. An operator asking
    "is my cache healthy" got yes."""
    for source in indices.SOURCES:
        (cache / source.filename).write_text("")
    state = indices.cache_state(cache)
    assert set(state) == {s.key for s in indices.SOURCES}
    assert all(age is None for age in state.values()), state


def test_cache_state_still_reports_a_real_file(cache):
    for source in indices.SOURCES:
        (cache / source.filename).write_text(GOOD)
    state = indices.cache_state(cache)
    assert all(age is not None and age >= 0 for age in state.values()), state


# --------------------------------------------------------------------------
# A 2xx that is not 200
#
# `services.swpc.noaa.gov` answered a station **202 Accepted with an empty
# body**, twice over, for both of its sources. NOAA does not send that; a proxy
# intercepting the request does. `urlopen` raises for no 2xx, so the old code
# read it as a successful fetch of an empty document.
# --------------------------------------------------------------------------

def test_a_202_with_no_body_does_not_replace_the_cache(cache, monkeypatch):
    cached = cache / "f.json"
    cached.write_text(GOOD)
    import os
    old = dt.datetime.now().timestamp() - 86400 * 30
    os.utime(cached, (old, old))

    serve(monkeypatch, body="", status_code=202)
    assert indices._fetch("http://x/f.json", cache, "f.json") == GOOD
    assert cached.read_text() == GOOD


def test_a_202_names_interception_rather_than_the_host(cache, monkeypatch):
    """The message has to point at the proxy. "0 bytes" sends an operator to
    NOAA's status page, which is fine, and tells them nothing."""
    serve(monkeypatch, body="", status_code=202)
    with pytest.raises(indices.IndexUnavailable) as caught:
        indices._fetch("http://x/f.json", cache, "f.json")
    detail = str(caught.value)
    assert "202" in detail
    assert "proxy" in detail or "intercepting" in detail


def test_a_200_is_still_accepted(cache, monkeypatch):
    serve(monkeypatch, body=GOOD, status_code=200)
    assert indices._fetch("http://x/f.json", cache, "f.json") == GOOD


# --------------------------------------------------------------------------
# An unwritable cache is not a failed fetch
#
# A `docker cp` left two cache files owned by another uid. Every later refresh
# downloaded the document successfully and then died on the write, so a
# permission problem on a *cache* presented as a model with no solar driver.
# --------------------------------------------------------------------------

def test_a_cache_that_cannot_be_written_still_returns_the_document(
        cache, monkeypatch):
    serve(monkeypatch, body=GOOD)

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", refuse)
    with pytest.warns(UserWarning, match="could not cache"):
        assert indices._fetch("http://x/f.json", cache, "f.json") == GOOD


def test_the_warning_says_the_value_is_still_correct(cache, monkeypatch):
    """The distinction an operator needs: the number is right, the caching is
    broken. Reporting it as a fetch failure sends them to the network."""
    serve(monkeypatch, body=GOOD)

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "write_text", refuse)
    with pytest.warns(UserWarning) as caught:
        indices._fetch("http://x/f.json", cache, "f.json")
    message = str(caught[0].message)
    assert "could not cache" in message
    assert "correct" in message


def test_a_failed_write_leaves_no_partial_behind(cache, monkeypatch):
    """A stray `.partial` would be indistinguishable from a real cache file to
    anyone reading the directory."""
    serve(monkeypatch, body=GOOD)
    real_write = Path.write_text

    def refuse_replace(self, target):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "replace", refuse_replace)
    with pytest.warns(UserWarning):
        indices._fetch("http://x/f.json", cache, "f.json")
    assert list(cache.iterdir()) == []
