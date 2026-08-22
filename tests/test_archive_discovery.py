"""Finding the folders worth registering, rather than listing what is there.

The archives page offered every subdirectory one level under the root. For the
station's own layout -- day directories written straight into the root -- that
meant the choices on offer *were* the days, so an archive was fifteen rows and
every new day the receiver created needed another one by hand.

A folder is a **dataset** when its day-named children hold soundings, or when
it holds sounding files directly. These tests pin that rule against the three
layouts in service, and pin the consequence that makes it worth having: a
registered dataset picks up tomorrow's day folder with nobody doing anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from muf import loader
from services.api import archives

# The api-backed tests below reuse `test_archives`'s client, archive root
# and helpers rather than standing up a second copy of the same fixtures.
from tests.test_archives import (  # noqa: F401
    _add, _candidates, archive_root, client)


def sounding(path: Path, name: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    target = path / name
    target.touch()
    return target


@pytest.fixture
def flat(tmp_path) -> Path:
    """What the receiver writes: day directories straight into the root."""
    root = tmp_path / "flat"
    for day in ("2026-08-04", "2026-08-05", "2026-08-06"):
        sounding(root / day, f"lfm_ionogram-NIC-{day}.h5")
    return root


@pytest.fixture
def nested(tmp_path) -> Path:
    """What the archive server holds: a dataset folder, then the days."""
    root = tmp_path / "nested"
    for day in ("2026-08-04", "2026-08-05"):
        sounding(root / "ionozond_data2" / day, f"lfm_ionogram-NIC-{day}.h5")
        sounding(root / "ionograms" / day, f"rec-{day}.lfs")
    (root / "temp" / "notes").mkdir(parents=True)
    (root / "Test Rusla").mkdir()
    return root


def names(root: Path, found: list[Path]) -> set[str]:
    return {p.relative_to(root).as_posix() for p in found}


# --------------------------------------------------------------------------
# The rule, against each layout
# --------------------------------------------------------------------------

def test_day_folders_at_the_root_make_the_root_the_dataset(flat):
    """One row, not one per day. This is the whole point."""
    assert archives.datasets(flat) == [flat]


def test_the_days_themselves_are_not_also_offered(flat):
    """A folder and the days inside it are never both choices -- offering both
    is what turns a list of choices back into an inventory."""
    assert names(flat, archives.datasets(flat)) == {"."}


def test_a_nested_layout_offers_the_dataset_folders(nested):
    """Not the root, because its children are not days -- and not the days,
    because their parent already covers them. Keeping the two folders separate
    is what lets a `.lfs` archive and a `chirp2` archive carry different
    formats and different estimators."""
    assert names(nested, archives.datasets(nested)) == {
        "ionozond_data2", "ionograms"}


def test_a_folder_of_loose_soundings_is_a_dataset(tmp_path):
    """No day directories at all -- some older archives are just a folder of
    files, and they are still registrable."""
    root = tmp_path / "loose"
    sounding(root / "captures", "rec-1.lfs")
    assert names(root, archives.datasets(root)) == {"captures"}


def test_folders_with_no_soundings_are_not_offered(nested):
    found = names(nested, archives.datasets(nested))
    assert "temp" not in found
    assert "Test Rusla" not in found


def test_a_synology_recycle_bin_is_never_offered(tmp_path):
    """It is full of day folders full of soundings, and it is the one place
    they must not be indexed from."""
    root = tmp_path / "vol"
    sounding(root / "#recycle" / "2026-08-04", "lfm_ionogram-x.h5")
    sounding(root / "@eaDir" / "2026-08-04", "lfm_ionogram-x.h5")
    sounding(root / "data" / "2026-08-04", "lfm_ionogram-x.h5")
    assert names(root, archives.datasets(root)) == {"data"}


def test_discovery_does_not_descend_for_ever(tmp_path):
    """Bounded, because an unbounded search of a 16 TB share looking for a
    folder that is not there is indistinguishable from a hang."""
    root = tmp_path / "deep"
    sounding(root / "a" / "b" / "c" / "d" / "2026-08-04", "lfm_ionogram-x.h5")
    assert archives.datasets(root) == []


def test_an_unreadable_folder_is_skipped_not_raised(nested, monkeypatch):
    """One dead share must not take the whole discovery pass down."""
    real = Path.iterdir

    def iterdir(self):
        if self.name == "ionograms":
            raise OSError(5, "Input/output error")
        return real(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert names(nested, archives.datasets(nested)) == {"ionozond_data2"}


# --------------------------------------------------------------------------
# The consequence
# --------------------------------------------------------------------------

def test_a_new_day_needs_no_registration(flat):
    """The ask behind all of this: the receiver creates tomorrow's folder and
    it is indexed, with nobody touching the archives page.

    `find_soundings` is recursive, so this follows from registering the
    *containing* folder rather than the days -- which is precisely what the
    root refusal used to prevent.
    """
    before = loader.find_soundings(flat)
    sounding(flat / "2026-08-07", "lfm_ionogram-NIC-2026-08-07.h5")
    after = loader.find_soundings(flat)

    assert len(after) == len(before) + 1
    # And the set of registrable folders has not changed -- there is still
    # exactly one, so nothing new is waiting to be added by hand.
    assert archives.datasets(flat) == [flat]


# --------------------------------------------------------------------------
# `has_soundings` is a shortcut, and shortcuts drift
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,name", [
    ("lfs", "recording.lfs"),
    ("chirp2", "lfm_ionogram-NIC-DOB-ch000-007-1770163210.01.h5"),
    ("digisonde", "digisonde_ionogram-Juliusruh-DOB-1786245496.00.h5"),
])
def test_the_probe_agrees_with_the_finder(tmp_path, fmt, name):
    """`has_soundings` reimplements the finders' globs to answer "any?" without
    building the list. Pinned against `find_soundings` itself so the two cannot
    drift apart in a way that makes a real archive undiscoverable."""
    root = tmp_path / fmt
    sounding(root / "2026-08-04", name)

    assert loader.has_soundings(root) is True
    assert loader.has_soundings(root, format=fmt) is True
    assert len(loader.find_soundings(root, format=fmt)) == 1


def test_the_probe_says_no_for_a_tree_of_the_wrong_files(tmp_path):
    """Detection products share the directory and the suffix. Counting them as
    soundings would offer a folder that then indexes nothing."""
    root = tmp_path / "detections"
    sounding(root / "2026-08-04", "cdetections-DOB-1785879000.h5")
    sounding(root / "2026-08-04", "chirp-DOB-1785879000.h5")

    assert loader.has_soundings(root) is False
    assert archives.datasets(root) == []


def test_the_probe_can_be_asked_about_this_folder_only(tmp_path):
    """The distinction the rule turns on: a folder of soundings, versus a
    folder of folders of soundings."""
    root = tmp_path / "root"
    sounding(root / "2026-08-04", "lfm_ionogram-x.h5")

    assert loader.has_soundings(root, recursive=True) is True
    assert loader.has_soundings(root, recursive=False) is False


# --------------------------------------------------------------------------
# A folder beside the days, not under them
#
# The receiver writes day directories into the root. Adding an archive of
# older `.lfs` recordings means putting a folder next to those days -- and
# discovery used to stop at the first dataset it found, which was the root,
# so the new folder was never offered and could only be registered by typing
# its path from memory.
# --------------------------------------------------------------------------

def test_a_sibling_folder_beside_the_days_is_also_offered(flat):
    sounding(flat / "old_lfs" / "2025-03-01", "rec-1.lfs")
    assert names(flat, archives.datasets(flat)) == {".", "old_lfs"}


def test_the_days_are_still_not_offered_alongside_it(flat):
    """The original bug must not come back through the new descent: day
    folders are covered by their parent and are never separate choices."""
    sounding(flat / "old_lfs" / "2025-03-01", "rec-1.lfs")
    offered = names(flat, archives.datasets(flat))
    assert not any(name.startswith("2026-08") for name in offered)
    assert "old_lfs/2025-03-01" not in offered


def test_a_sibling_with_nothing_in_it_is_still_not_offered(flat):
    (flat / "scratch" / "notes").mkdir(parents=True)
    assert names(flat, archives.datasets(flat)) == {"."}


def test_the_list_says_which_offer_contains_which(client, archive_root):
    """Two ways to register one tree -- the root, or the folders in it -- are
    both legitimate, and which is which belongs on the page rather than in the
    409 an operator meets after filling in the form."""
    (archive_root / "good" / "extra").mkdir(parents=True)
    (archive_root / "good" / "extra" / "rec.lfs").write_bytes(b"")

    offered = _candidates(client)
    assert offered["good"]["inside"] is None
    assert offered["good/extra"]["inside"] == "good"


def test_a_covered_folder_can_still_be_registered_on_its_own(client, archive_root):
    """`inside` is a note, not a refusal. Registering the inner folder alone
    is a real choice -- it is how a `.lfs` subtree gets its own estimators --
    and only registering *both* is the mistake."""
    (archive_root / "good" / "extra").mkdir(parents=True)
    (archive_root / "good" / "extra" / "rec.lfs").write_bytes(b"")

    assert _add(client, path="good/extra", format="lfs").status_code == 200
    assert _add(client, path="good").status_code == 409
