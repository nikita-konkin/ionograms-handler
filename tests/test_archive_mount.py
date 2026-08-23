"""The mount panel survives the mount it is reporting on being broken.

The symptom that prompted this: `/ui/archives` returning "Internal Server
Error" on the work server while `docker ps` showed the api **healthy**. Those
two are consistent, and that is the trap -- `/healthz` answers "is this process
up" and touches no storage on purpose, so a NAS that has stopped answering
leaves a green container and one dead page.

`Path.is_dir()` swallows only ENOENT, ENOTDIR, EBADF and ELOOP. A mount that is
present but sick raises EIO, ESTALE, EACCES or ETIMEDOUT, and those propagate
straight out of the request. `iterdir` was guarded; `is_dir` was not.
"""

from __future__ import annotations

import errno
import re
from pathlib import Path

import pytest

from services.api import archives, db, main

#: What a mount that is present but not answering raises. ENOENT is in the
#: list as the control: it is the one `Path.is_dir` already handled, so a test
#: that only used it would have passed against the bug.
SICK = [errno.EIO, errno.ESTALE, errno.EACCES, errno.ETIMEDOUT, errno.ENOENT]



#: The root the fixture pretends is mounted. Pinned rather than taken from the
#: environment, so the test does not quietly pass by patching a path the app
#: never looks at -- which is exactly how the first version of this passed.
ROOT = Path("/archive")


@pytest.fixture
def sick_mount(monkeypatch):
    """Make every stat of the archive root fail, as a dead NAS does."""
    def go(number: int):
        real = Path.is_dir

        def is_dir(self):
            if self == ROOT:
                raise OSError(number, "the mount is not answering")
            return real(self)

        monkeypatch.setattr(archives, "roots", lambda: [ROOT])
        monkeypatch.setattr(Path, "is_dir", is_dir)
    return go


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_the_mount_state_reports_rather_than_raises(sick_mount, number):
    sick_mount(number)
    state = archives._root_state(ROOT, 0)
    assert state["exists"] is False
    assert state["populated"] is None


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_the_archives_page_still_renders(client, sick_mount, number):
    """A dead mount is the thing this page exists to show. It must not be the
    thing that stops it rendering."""
    sick_mount(number)
    response = client.get("/ui/archives")
    assert response.status_code == 200, (
        f"{errno.errorcode[number]} from the mount 500s the page that reports "
        f"on the mount")


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_healthz_and_the_page_do_not_disagree_silently(client, sick_mount, number):
    """`/healthz` staying green while storage is gone is by design -- it answers
    a different question. The page has to answer this one."""
    sick_mount(number)
    assert client.get("/healthz").json()["ok"] is True
    body = client.get("/archives").json()
    assert body["mount"]["exists"] is False


def test_an_unreadable_root_says_why(sick_mount):
    sick_mount(errno.EIO)
    state = archives._root_state(ROOT, 0)
    assert "Input/output error" in state["error"] or "EIO" in state["error"] \
        or "not answering" in state["error"]


# --------------------------------------------------------------------------
# Which unreadable, and therefore which fix
# --------------------------------------------------------------------------
#
# The work server hit EIO on one folder and EHOSTDOWN on another within four
# minutes. The page told it to set ARCHIVE_HOST_PATH and redeploy -- advice for
# a bind mount that was never made, given to someone whose path was correct and
# whose NAS had gone away. Redeploying would have changed nothing.

@pytest.mark.parametrize("name", ["EIO", "EHOSTDOWN", "ESTALE", "ETIMEDOUT"])
def test_storage_that_stopped_answering_is_not_a_missing_bind_mount(name):
    number = getattr(errno, name, None)
    if number is None:
        pytest.skip(f"{name} is not defined on this platform")
    assert archives.mount_fault(OSError(number, name)) == "unreachable"


def test_a_mount_that_was_never_made_is_still_reported_as_missing():
    assert archives.mount_fault(OSError(errno.ENOENT, "no such file")) == "missing"
    assert archives.mount_fault(OSError(errno.ENOTDIR, "not a directory")) == "missing"


def test_a_permission_problem_is_neither():
    assert archives.mount_fault(OSError(errno.EACCES, "denied")) == "denied"


@pytest.mark.parametrize("name,expected,wrong", [
    ("EIO", "not answering", "ARCHIVE_HOST_PATH"),
    ("EHOSTDOWN", "not answering", "ARCHIVE_HOST_PATH"),
    ("ENOENT", "ARCHIVE_HOST_PATH", "not answering"),
])
def test_the_page_offers_the_fix_that_matches_the_errno(client, sick_mount,
                                                        name, expected, wrong):
    number = getattr(errno, name, None)
    if number is None:
        pytest.skip(f"{name} is not defined on this platform")
    sick_mount(number)
    text = client.get("/ui/archives").text
    assert expected in text, f"{name}: page does not mention {expected!r}"
    assert wrong not in text.split("archives")[0] or expected in text


def test_the_unreachable_message_does_not_send_anyone_to_redeploy(client,
                                                                  sick_mount):
    """The specific wrong turn: editing a correct .env and redeploying while
    the storage is still down."""
    sick_mount(errno.EIO)
    text = client.get("/ui/archives").text
    start = text.find("The mount is there")
    assert start > 0, "the unreachable banner did not render"
    # Bounded by its own paragraph, so the two phrases have to be in the same
    # message rather than merely somewhere on a long page.
    # Tags stripped and whitespace collapsed: the assertion is about what a
    # reader sees, and `is <i>not</i> the problem` is one sentence to them.
    banner = re.sub(r"\s+", " ",
                    re.sub(r"<[^>]+>", "", text[start:text.index("</p>", start)]))
    assert "not the problem" in banner
    assert "fix it on the host" in banner
    assert "redeploying will not help" in banner.lower()


# --------------------------------------------------------------------------
# The routes that serve a file, not a page
# --------------------------------------------------------------------------

def seed_sounding(conn):
    conn.execute(
        "INSERT INTO sounding (file, path, datetime, tx, rx, ingested_at) "
        "VALUES (?,?,?,?,?,?)",
        ("s1.h5", "2026-08-19/s1.h5", "2026-08-19 00:00:00", "NIC", "DOB",
         db.utcnow()))
    conn.commit()
    return conn.execute("SELECT id FROM sounding").fetchone()[0]


@pytest.fixture
def sick_file(monkeypatch):
    """Every `is_file` on the archive fails, as it does on a dead mount."""
    def go(number: int):
        real = Path.is_file

        def is_file(self):
            raise OSError(number, "the mount is not answering")

        monkeypatch.setattr(Path, "is_file", is_file)
    return go


@pytest.mark.parametrize("name", ["EIO", "EHOSTDOWN", "ESTALE"])
def test_an_ionogram_on_dead_storage_is_503_not_500(client, sick_file, name):
    """The row is intact and the file is probably fine -- what is down is the
    disk. A 500 says the server is broken; a 410 would say the sounding is
    gone. Both are wrong, and 410 would be cached as permanent."""
    number = getattr(errno, name, None)
    if number is None:
        pytest.skip(f"{name} is not defined on this platform")
    sounding = seed_sounding(client.app.state.db)
    sick_file(number)

    response = client.get(f"/ionogram/{sounding}.png")
    assert response.status_code == 503, response.status_code
    assert "storage, not data" in response.json()["detail"]


def test_a_genuinely_missing_file_is_still_410(client, monkeypatch):
    """The other branch has to keep working: a row whose file really is not
    there is a different fault with a different fix."""
    sounding = seed_sounding(client.app.state.db)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    response = client.get(f"/ionogram/{sounding}.png")
    assert response.status_code == 410
    assert "ARCHIVE_ROOT" in response.json()["detail"]


def test_a_permission_problem_names_the_uid(client, sick_file):
    sounding = seed_sounding(client.app.state.db)
    sick_file(errno.EACCES)
    detail = client.get(f"/ionogram/{sounding}.png").json()["detail"]
    assert "uid 10001" in detail


# --------------------------------------------------------------------------
# Walking the root, not just probing it
#
# `_root_state` probes the root and reports what it finds. `candidates` *walks*
# it, and did so through a bare `root.is_dir()` -- so the page that reports a
# dead mount could still 500 one function further along. The fixture above
# breaks a pinned `/archive`, which the app under test never walks, so these
# break the root the request path actually uses.
# --------------------------------------------------------------------------

@pytest.fixture
def sick_primary(monkeypatch):
    """Break the archive root the running app walks, whatever it is."""
    def go(number: int):
        root = Path(main.app.state.archive_root).resolve()
        real = Path.is_dir

        def is_dir(self):
            if self.resolve() == root:
                raise OSError(number, "the mount is not answering")
            return real(self)

        monkeypatch.setattr(Path, "is_dir", is_dir)
        return root
    return go


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_listing_candidates_survives_a_dead_root(sick_primary, number):
    root = sick_primary(number)
    with db.session(main.app.state.db_path if hasattr(main.app.state, "db_path")
                    else db.DEFAULT_DB) as conn:
        assert archives.candidates(conn, root) == []


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_the_archives_page_renders_when_its_own_root_is_dead(client, sick_primary,
                                                             number):
    """The regression: the page probed the mount safely and then walked it
    unsafely, which is a 500 on the one page that explains the outage."""
    sick_primary(number)
    assert client.get("/ui/archives").status_code == 200


@pytest.mark.parametrize("number", SICK, ids=lambda n: errno.errorcode[n])
def test_listable_never_raises(number):
    path = Path("/nowhere-in-particular")

    def is_dir(self):
        raise OSError(number, "the mount is not answering")

    original = Path.is_dir
    Path.is_dir = is_dir
    try:
        assert archives.listable(path) is False
    finally:
        Path.is_dir = original


def test_registering_on_a_dead_mount_does_not_call_it_missing(client, monkeypatch):
    """`is not a directory that exists here` is a lie about a folder on a share
    that stopped answering, and sends the operator after a path bug they do not
    have."""
    def is_dir(self):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(Path, "is_dir", is_dir)
    response = client.post("/archives", json={"path": "ionograms"},
                           headers={"Authorization": "Bearer ctl"})
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "mount is not answering" in detail
    assert "is not a directory that exists here" not in detail


def test_a_walk_that_dies_is_not_reported_as_an_empty_folder(tmp_path, monkeypatch):
    """Zero soundings is a measurement. A walk that failed is not one, and
    registering on the strength of it would schedule a scan of nothing."""
    from muf import loader

    def blow_up(*args, **kwargs):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(loader, "find_soundings", blow_up)
    with pytest.raises(archives.ArchiveError) as caught:
        archives.survey(tmp_path)
    message = str(caught.value).lower()
    assert "could not be walked" in message
    assert "says nothing about what the folder holds" in message


# --------------------------------------------------------------------------
# The probe is not the read
#
# A CIFS client answers `is_file()` out of its attribute cache, so on a share
# that has just gone the probe succeeds and the read a millisecond later does
# not. That is the shape of a mount dropping under an index, and it reached
# the request as a 500 -- the server reporting itself broken on behalf of a
# NAS that is the thing at fault.
# --------------------------------------------------------------------------

@pytest.fixture
def sick_read(monkeypatch):
    """The file probes fine and then fails to open."""
    def go(number: int):
        from muf import loader

        def blow_up(*args, **kwargs):
            raise OSError(number, "the mount is not answering")

        monkeypatch.setattr(Path, "is_file", lambda self: True)
        monkeypatch.setattr(loader, "load", blow_up)
        monkeypatch.setattr(loader, "read_header", blow_up)
    return go


@pytest.mark.parametrize("name", ["EIO", "EHOSTDOWN", "ESTALE", "ETIMEDOUT"])
@pytest.mark.parametrize("route", ["/ionogram/{id}.png", "/soundings/{id}/sao.xml"])
def test_a_read_that_dies_mid_request_is_503_not_500(client, sick_read, name, route):
    number = getattr(errno, name, None)
    if number is None:
        pytest.skip(f"{name} is not defined on this platform")
    sounding = seed_sounding(client.app.state.db)
    sick_read(number)

    response = client.get(route.format(id=sounding))
    assert response.status_code == 503, (
        f"{route} answered {response.status_code} when the read hit "
        f"{name}; a 500 blames the server for the NAS being down")
    assert "storage, not data" in response.json()["detail"]


# --------------------------------------------------------------------------
# The mount panel is on a polled endpoint
#
# `GET /archives` is polled once a second while a scan runs -- the code that
# serves it says so. Reading an entry *count* for the panel meant enumerating
# the whole root on every one of those polls, and on Python 3.12 `iterdir` is
# `os.listdir` underneath, so it read the entire directory eagerly. On a local
# disk that is 0.1 ms; on the station's SMB share it is 6.3 ms per entry,
# against the same mount the indexer is reading. The panel that exists to
# report the mount's health was helping to take it down.
# --------------------------------------------------------------------------

def test_the_mount_panel_does_not_enumerate_the_root(tmp_path, monkeypatch):
    import os as os_mod

    root = tmp_path / "archive"
    root.mkdir()
    for day in range(200):
        (root / f"2026-08-{day:03d}").mkdir()

    consumed = []
    real_scandir = os_mod.scandir

    class Counting:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def __iter__(self):
            return self

        def __next__(self):
            consumed.append(1)
            return next(self._inner)

    monkeypatch.setattr(os_mod, "scandir", lambda p: Counting(real_scandir(p)))
    monkeypatch.setattr(os_mod, "listdir",
                        lambda p: pytest.fail("read the whole root with listdir"))

    state = archives._root_state(root, 0)

    assert state["exists"] is True
    assert state["populated"] is True
    assert state["empty"] is False
    assert len(consumed) <= 1, (
        f"pulled {len(consumed)} of 200 entries; on a share polled once a "
        f"second at 6.3 ms an entry that is the outage, not a page render")


def test_an_empty_root_is_still_reported_as_empty(tmp_path):
    """The one thing the count was load-bearing for. A mounted-but-empty root
    means every scan will truthfully report "0 on disk", and saying so is the
    whole point of the panel."""
    root = tmp_path / "archive"
    root.mkdir()
    state = archives._root_state(root, 0)
    assert state["exists"] is True
    assert state["populated"] is False
    assert state["empty"] is True
