"""Soundings whose file is gone: counted on request, removed only as counted.

The case that made it: tesla's August rows from the old ``ionozond_data2``
share, whose files were deleted on purpose (2026-10-16). What must hold is
less about the delete than about what it must *not* do -- take a row whose
file is present, take more than the operator was shown, or read a share that
is down as a share whose files are all gone.
"""

from __future__ import annotations

import errno
import os

import pytest

from services.api import auth, db, main
from services.api import missing as missing_mod

CTL = {"Authorization": "Bearer ctl"}


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "archive"
    for day in ("old/2026-08-01", "old/2026-08-02", "new/2026-10-01"):
        (root / day).mkdir(parents=True)
    return root


@pytest.fixture
def client(tmp_path, root, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from services.api import archives as archives_mod
    from services.api import net
    from services.api import series as series_mod

    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setenv("API_DB", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "api.sqlite3")
    monkeypatch.setenv("ARCHIVE_ROOT", str(root))
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    net.reset()
    monkeypatch.setattr(series_mod, "MODEL", False)
    series_mod.clear()
    monkeypatch.setattr(archives_mod, "DEFAULT_INTERVAL_S", 0.0)
    with TestClient(main.app) as c:
        yield c


def _sounding(conn, root, relpath: str, when: str, *, on_disk: bool) -> int:
    if on_disk:
        (root / relpath).write_bytes(b"x")
    name = relpath.rsplit("/", 1)[-1]
    cur = conn.execute(
        "INSERT INTO sounding (file, path, datetime, ingested_at)"
        " VALUES (?, ?, ?, ?)", (name, relpath, when, db.utcnow()))
    sid = int(cur.lastrowid)
    conn.execute("INSERT INTO extraction (sounding_id, method, muf)"
                 " VALUES (?, 'contour', 20.0)", (sid,))
    conn.execute("INSERT INTO reference (sounding_id, source, param, value)"
                 " VALUES (?, 'iri', 'muf', 21.0)", (sid,))
    conn.commit()
    return sid


@pytest.fixture
def seeded(client, root):
    """`old`: two of four files deleted, and one whole day folder gone.
    `new`: a second archive, every file present."""
    conn = main.app.state.db
    ids = {
        "kept": _sounding(conn, root, "old/2026-08-01/a.lfs",
                          "2026-08-01T00:00:00", on_disk=True),
        "gone1": _sounding(conn, root, "old/2026-08-01/b.lfs",
                           "2026-08-01T00:10:00", on_disk=False),
        "gone2": _sounding(conn, root, "old/2026-08-03/c.lfs",
                           "2026-08-03T00:00:00", on_disk=False),
        "kept2": _sounding(conn, root, "old/2026-08-02/d.lfs",
                           "2026-08-02T00:00:00", on_disk=True),
        "other": _sounding(conn, root, "new/2026-10-01/e.lfs",
                           "2026-10-01T00:00:00", on_disk=False),
    }
    old = db.add_archive(conn, name="old", relpath="old", methods="contour")
    new = db.add_archive(conn, name="new", relpath="new", methods="contour")
    return conn, old, new, ids


def _count(conn, table: str) -> int:
    return db.one(conn, f"SELECT COUNT(*) AS n FROM {table}")["n"]


def test_it_counts_exactly_the_rows_whose_file_is_gone(client, seeded):
    _, old, _, _ = seeded
    m = client.get(f"/archives/{old}/missing").json()

    assert m["total"] == 2 and m["of"] == 4
    assert m["folder"] == "present"
    assert (m["first"], m["last"]) == ("2026-08-01T00:10:00", "2026-08-03T00:00:00")
    assert m["examples"] == ["old/2026-08-01/b.lfs", "old/2026-08-03/c.lfs"]


def test_removing_takes_them_with_their_values_and_nothing_else(client, seeded):
    conn, old, _, ids = seeded
    r = client.request("DELETE", f"/archives/{old}/missing?expect=2", headers=CTL)

    assert r.status_code == 200, r.text
    assert r.json()["removed"] == 2
    left = {row["id"] for row in db.rows(conn, "SELECT id FROM sounding")}
    assert left == {ids["kept"], ids["kept2"], ids["other"]}
    # Cascaded, not orphaned.
    assert _count(conn, "extraction") == 3 and _count(conn, "reference") == 3
    assert client.get(f"/archives/{old}/missing").json()["total"] == 0


def test_another_archives_missing_files_are_not_touched(client, seeded):
    conn, old, new, ids = seeded
    client.request("DELETE", f"/archives/{old}/missing?expect=2", headers=CTL)
    assert db.one(conn, "SELECT id FROM sounding WHERE id = ?", (ids["other"],))
    assert client.get(f"/archives/{new}/missing").json()["total"] == 1


def test_a_count_that_changed_is_refused(client, seeded, root):
    """The operator agreed to a number. A file reappearing -- or a share
    dropping -- between the look and the press changes what would go."""
    conn, old, _, _ = seeded
    assert client.get(f"/archives/{old}/missing").json()["total"] == 2
    (root / "old/2026-08-01/b.lfs").write_bytes(b"back")

    r = client.request("DELETE", f"/archives/{old}/missing?expect=2", headers=CTL)
    assert r.status_code == 409
    assert "1 sounding(s)" in r.json()["detail"]
    assert _count(conn, "sounding") == 5


def test_a_share_that_does_not_answer_is_not_a_share_whose_files_are_gone(
        client, seeded, monkeypatch):
    conn, old, _, _ = seeded

    def down(path):
        raise OSError(errno.EHOSTDOWN, "Host is down", str(path))

    monkeypatch.setattr(missing_mod.os, "scandir", down)
    assert client.get(f"/archives/{old}/missing").status_code == 503
    r = client.request("DELETE", f"/archives/{old}/missing?expect=4", headers=CTL)
    assert r.status_code == 503
    assert _count(conn, "sounding") == 5


def test_an_unmounted_folder_says_so(client, seeded, root):
    """An empty mountpoint looks exactly like deleted files. It cannot be told
    apart, so the count says what the folder itself looks like."""
    _, old, _, _ = seeded
    for path in sorted((root / "old").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()

    m = client.get(f"/archives/{old}/missing").json()
    assert m["total"] == 4 and m["folder"] == "empty"
    (root / "old").rmdir()
    assert client.get(f"/archives/{old}/missing").json()["folder"] == "absent"


def test_looking_is_open_and_removing_needs_the_token(client, seeded):
    conn, old, _, _ = seeded
    assert client.get(f"/archives/{old}/missing").status_code == 200
    assert client.request(
        "DELETE", f"/archives/{old}/missing?expect=2").status_code == 401
    assert client.request("DELETE", f"/archives/{old}/missing",
                          headers=CTL).status_code == 422     # no count, no delete
    assert _count(conn, "sounding") == 5


def test_an_unknown_archive_is_404(client, seeded):
    assert client.get("/archives/999/missing").status_code == 404


def test_one_listing_per_folder_not_one_stat_per_file(seeded, root, monkeypatch):
    """On a share every stat is a round trip; a day folder is listed once."""
    conn, old, _, _ = seeded
    for i in range(50):
        _sounding(conn, root, f"old/2026-08-02/x{i}.lfs", "2026-08-02T01:00:00",
                  on_disk=i % 2 == 0)
    listed = []
    real = os.scandir

    def counting(path):
        listed.append(str(path))
        return real(path)

    monkeypatch.setattr(missing_mod.os, "scandir", counting)
    found = missing_mod.find(conn, db.archive(conn, old), root)
    assert found.total == 2 + 25
    # Three day folders, plus the archive's own folder for its state.
    assert len(listed) == 4


def test_the_page_offers_the_check(client, seeded):
    page = client.get("/ui/archives").text
    assert "checkMissing(" in page and "/missing?expect=" in page
