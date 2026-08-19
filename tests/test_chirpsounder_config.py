"""The census in ``docs/chirpsounder2-config.md`` against the real source.

The document exists because a key that is parsed and read by nothing looks
exactly like a key that works (BACKLOG sec. 3, sec. 30). That failure is
silent, so the census only helps if it cannot quietly go stale: upstream adds a
key, or moves a reader, and the table still reads as authoritative.

These tests re-derive the mechanical half of the table from
``../chirpsounder2`` and fail when it has drifted. They skip when that
checkout is absent, which is most machines -- it is the station's clone, not a
dependency.

The *mechanical* half is all they can guard. The document is explicit that the
readers column under-reports, for five reasons it lists; a test that demanded
the hand-written classifications match a word-search would be enforcing the
wrong thing, and would have to be wrong in the same direction the search is.
"""
import re
from pathlib import Path

import pytest

CLONE = Path(__file__).resolve().parents[2] / "chirpsounder2"
CENSUS = Path(__file__).resolve().parents[1] / "docs" / "chirpsounder2-config.md"


def _skip_without_clone():
    if not (CLONE / "chirp_config.py").exists():
        pytest.skip(f"station chirpsounder2 clone not present: {CLONE}")


def _declared_keys():
    """``(section, key)`` for every key in ``chirp_config``'s default tables."""
    src = (CLONE / "chirp_config.py").read_text()
    section, out = None, []
    for line in src.splitlines():
        header = re.match(r'\s*cf\["(\w+)"\]\s*=', line)
        if header:
            section = header.group(1)
        key = re.match(r'\s*"([a-z_0-9]+)"\s*:', line)
        if key and section:
            out.append((section, key.group(1)))
    return out


#: The sections ``chirp_config`` defines defaults for. Used to pick the census
#: table out of the document, which holds three other tables with the same
#: pipe-delimited shape.
SECTIONS = ("config", "detection", "lfm", "transfer", "rtf", "stations")


def _census_keys():
    rows = []
    for line in CENSUS.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 7 or cells[1] not in SECTIONS:
            continue
        rows.append((cells[1], cells[2].strip("`")))
    return rows


def test_census_covers_every_declared_key():
    _skip_without_clone()
    missing = sorted(set(_declared_keys()) - set(_census_keys()))
    assert not missing, (
        f"{len(missing)} ini key(s) exist in chirp_config.py and not in the "
        f"census: {missing}. A key absent from the table is the case the "
        f"document was written to prevent -- it reads as complete.")


def test_census_invents_nothing():
    _skip_without_clone()
    extra = sorted(set(_census_keys()) - set(_declared_keys()))
    assert not extra, (
        f"the census lists {extra}, which chirp_config.py no longer declares. "
        f"An upstream removal leaves the panel offering a key that is gone.")


def test_the_only_validated_key_is_still_the_only_one():
    """The census claims exactly one key is validated. That is a load-bearing
    claim: every cross-check in sec. 30 exists because the parser has none."""
    _skip_without_clone()
    src = (CLONE / "chirp_config.py").read_text()
    raises = re.findall(r"raise\s+(\w+)", src)
    assert raises == ["ValueError"], (
        f"chirp_config.py now raises {raises}; the census says it validates "
        f"downconversion_filter and nothing else. Re-check the claim.")


def test_ionowebsync_still_defaults_to_an_unreachable_host():
    """The 8.37% sounding loss traces to a 60 s blocking POST to a host this
    network cannot reach (BACKLOG sec. 3). If upstream changes the default or
    the timeout, the census's remedies stop being the right ones."""
    _skip_without_clone()
    src = (CLONE / "ionowebsync.py").read_text()
    assert "juha.no" in src and "timeout=60" in src, (
        "ionowebsync.py's default URL or timeout changed; the census section "
        "'The upload is configured to one place and goes to another' needs "
        "re-reading before it is quoted again.")


def test_digisonde_section_is_still_outside_chirp_config():
    """The census warns that a panel built from ``chirp_config``'s defaults
    would omit the digisonde receiver entirely."""
    _skip_without_clone()
    defaults = (CLONE / "chirp_config.py").read_text()
    assert 'cf["digisonde"]' not in defaults, (
        "chirp_config.py now defines a [digisonde] section, so it is no "
        "longer a second configuration surface. Simplify the census.")
    reader = (CLONE / "receive_digisonde.py").read_text()
    assert '"digisonde"' in reader, (
        "receive_digisonde.py no longer reads [digisonde] directly; find out "
        "what does before trusting the census on it.")
