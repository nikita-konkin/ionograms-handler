"""Raw deletion requires a verified replacement and a recoverable journal."""

import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import make_header_bytes

from muf import echo_window, loader
from services.api import conversion, db, watch


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    day = root / "2026.02.04"
    day.mkdir(parents=True)
    monkeypatch.setenv("ARCHIVE_ROOT", str(root))
    monkeypatch.delenv("ARCHIVE_ROOTS", raising=False)
    sr, n = 40000, 200000
    rng = np.random.default_rng(5)
    samples = (
        rng.normal(size=n)
        + 1j * rng.normal(size=n)
        + 30 * np.exp(-2j * np.pi * (2900 / 3) * np.arange(n) / sr)
    )
    source = day / "synth_20260204_000010.lfs"
    source.write_bytes(make_header_bytes() + samples.astype(np.complex64).tobytes())
    old = time.time() - 120
    os.utime(source, (old, old))
    with db.session(tmp_path / "db.sqlite3") as conn:
        archive_id = db.add_archive(
            conn, name="captures", relpath=".", format="lfs", methods="contour"
        )
        yield conn, root, source, archive_id


def reviewed(archive):
    conn, root, source, archive_id = archive
    job = conversion.enqueue(
        conn, archive_id, source.parent.relative_to(root).as_posix(), "admin"
    )
    assert job["mode"] == "survey"
    assert conversion.run_next(conn)["state"] == "ready"
    return conversion.start(conn, job["id"], True)


def index_raw(conn, root, source):
    cur = conn.execute(
        "INSERT INTO sounding(file,path,format,datetime,ingested_at)"
        " VALUES (?,?,'lfs','2026-02-04 00:00:10',?)",
        (source.name, source.relative_to(root).as_posix(), db.utcnow()),
    )
    conn.execute(
        "INSERT INTO extraction(sounding_id,method,muf) VALUES (?,'contour',18.5)",
        (cur.lastrowid,),
    )
    conn.execute(
        "INSERT INTO reference(sounding_id,source,param,value)"
        " VALUES (?,'iri','muf',17.0)",
        (cur.lastrowid,),
    )
    conn.commit()
    return cur.lastrowid


def test_inspection_is_read_only_and_requires_explicit_replacement(archive):
    conn, root, source, archive_id = archive
    checksum = conversion.digest(source)
    job = conversion.enqueue(conn, archive_id, ".", "admin")
    job = conversion.run_next(conn)
    assert job["state"] == "ready" and job["total"] == 1
    assert job["source_bytes"] == source.stat().st_size
    assert conversion.digest(source) == checksum
    assert not list(root.rglob("*.echo.h5"))
    with pytest.raises(conversion.ConversionError, match="confirm"):
        conversion.start(conn, job["id"], False)


def test_verified_replacement_removes_raw_and_preserves_measurements(archive):
    conn, root, source, archive_id = archive
    sounding_id = index_raw(conn, root, source)
    reviewed(archive)
    result = conversion.run_next(conn)
    assert result["state"] == "completed", result
    assert (result["done"], result["failed"]) == (1, 0)
    assert not source.exists()
    target = source.with_suffix(".echo.h5")
    assert target.is_file()
    row = db.one(conn, "SELECT * FROM sounding WHERE id = ?", (sounding_id,))
    assert row["file"] == target.name and row["format"] == "echo_window"
    assert root / row["path"] == target
    assert (
        db.one(
            conn, "SELECT muf FROM extraction WHERE sounding_id = ?", (sounding_id,)
        )["muf"]
        == 18.5
    )
    assert (
        db.one(
            conn, "SELECT value FROM reference WHERE sounding_id = ?", (sounding_id,)
        )["value"]
        == 17.0
    )
    assert loader.load(target).power.size
    assert loader.find_soundings(root) == [target]
    assert db.archive(conn, archive_id)["format"] == "lfs"
    assert db.archive_orphans(conn, archive_id)["total"] == 0
    assert loader.find_soundings(root, format="lfs") == [target]
    assert (
        watch.find_new(
            [root], conn, ("contour",), 0, now=time.time() + 120, archive_root=root
        )[0]
        == []
    )


def test_verification_failure_preserves_raw(archive, monkeypatch):
    conn, _, source, _ = archive
    job = reviewed(archive)
    before = conversion.digest(source)

    def fail(*args, **kwargs):
        raise echo_window.VerificationError("mismatched samples")

    monkeypatch.setattr(echo_window, "write", fail)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and result["failed"] == 1
    assert conversion.digest(source) == before
    assert (
        "mismatched"
        in db.one(
            conn,
            "SELECT detail FROM archive_conversion_file WHERE job_id = ?",
            (job["id"],),
        )["detail"]
    )


def test_changed_capture_is_not_deleted(archive):
    conn, _, source, _ = archive
    reviewed(archive)
    source.write_bytes(source.read_bytes() + b"more samples")
    result = conversion.run_next(conn)
    assert result["failed"] == 1 and source.exists()
    assert not source.with_suffix(".echo.h5").exists()


def test_foreign_existing_output_is_not_overwritten_or_used_for_deletion(archive):
    conn, _, source, _ = archive
    target = source.with_suffix(".echo.h5")
    target.write_bytes(b"unrelated existing file")
    reviewed(archive)
    result = conversion.run_next(conn)
    assert result["failed"] == 1 and source.exists()
    assert target.read_bytes() == b"unrelated existing file"


def test_possible_cropped_echo_keeps_the_raw(archive, monkeypatch):
    conn, _, source, _ = archive
    reviewed(archive)
    read = echo_window.read

    def flagged(*args, **kwargs):
        window = read(*args, **kwargs)
        window.trace_at_edge = True
        return window

    monkeypatch.setattr(echo_window, "read", flagged)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and source.exists()


def test_database_error_before_unlink_keeps_raw(archive, monkeypatch):
    conn, _, source, _ = archive
    reviewed(archive)

    def busy(*args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(conversion, "_relink", busy)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and source.exists()


def test_restart_recovers_after_unlink_before_database_commit(archive, monkeypatch):
    conn, root, source, _ = archive
    sounding_id = index_raw(conn, root, source)
    job = reviewed(archive)
    unlink = Path.unlink

    def interrupted(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if path == source:
            raise OSError("power loss after unlink")

    monkeypatch.setattr(Path, "unlink", interrupted)
    result = conversion.run_next(conn)
    assert result["state"] == "failed" and not source.exists()
    assert (
        db.one(conn, "SELECT file FROM sounding WHERE id = ?", (sounding_id,))["file"]
        == source.name
    )
    monkeypatch.setattr(Path, "unlink", unlink)
    conversion.start(conn, job["id"], True)
    resumed = conversion.run_next(conn)
    assert resumed["state"] == "completed", resumed
    assert (
        db.one(conn, "SELECT file FROM sounding WHERE id = ?", (sounding_id,))["file"]
        == source.with_suffix(".echo.h5").name
    )


def test_cancel_stops_before_the_next_capture(archive, monkeypatch):
    conn, _, source, _ = archive
    second = source.with_name("second.lfs")
    second.write_bytes(source.read_bytes())
    os.utime(second, (time.time() - 120,) * 2)
    job = reviewed(archive)
    replace = conversion.replace_one

    def replace_and_cancel(*args):
        replace(*args)
        conversion.cancel(conn, job["id"])

    monkeypatch.setattr(conversion, "replace_one", replace_and_cancel)
    result = conversion.run_next(conn)
    assert result["state"] == "cancelled" and result["done"] == 1
    assert sum(path.exists() for path in (source, second)) == 1


def test_paths_outside_registered_archive_are_refused(archive):
    conn, root, _, archive_id = archive
    for subpath in ("../elsewhere", str(root.parent)):
        with pytest.raises(conversion.ConversionError):
            conversion.enqueue(conn, archive_id, subpath, "admin")
    assert conversion.listing(conn) == []


def test_only_one_job_can_be_active(archive):
    conn, _, _, archive_id = archive
    conversion.enqueue(conn, archive_id, ".", "admin")
    with pytest.raises(conversion.ConversionError, match="another"):
        conversion.enqueue(conn, archive_id, ".", "admin")


def test_discovery_does_not_double_count_raw_and_derived(archive):
    _, root, source, _ = archive
    target = echo_window.write(source).target
    assert loader.find_soundings(root) == [source]
    assert loader.find_soundings(target) == [target]
    with pytest.raises(FileNotFoundError):
        loader.find_soundings(root, format="echo_window")
    assert not loader.has_soundings(root, format="echo_window")
    source.unlink()
    assert loader.has_soundings(root, format="echo_window")
    assert loader.find_soundings(root) == [target]


def test_lfs_filter_keeps_replacements_without_admitting_other_formats(archive):
    from services.api import archives

    conn, root, source, archive_id = archive
    index_raw(conn, root, source)
    # A non-root registration also exercises the orphan path-prefix check.
    with conn:
        conn.execute(
            "UPDATE archive SET relpath = ? WHERE id = ?",
            (source.parent.relative_to(root).as_posix(), archive_id),
        )
    job = conversion.enqueue(conn, archive_id, ".", "admin")
    conversion.run_next(conn)
    conversion.start(conn, job["id"], True)
    assert conversion.run_next(conn)["state"] == "completed"
    unrelated = source.with_name("lfm_ionogram-other.h5")
    unrelated.touch()
    assert loader.find_soundings(root, format="lfs") == [source.with_suffix(".echo.h5")]
    assert archives.survey(root, format="lfs")["by_format"] == {"echo_window": 1}
    with conn:
        conn.execute(
            "INSERT INTO sounding(file,path,format,datetime,ingested_at)"
            " VALUES (?,?,'chirp2','2026-02-04 00:00:10',?)",
            (unrelated.name, unrelated.relative_to(root).as_posix(), db.utcnow()),
        )
    assert db.archive_orphans(conn, archive_id)["by_format"] == {"chirp2": 1}
    assert db.delete_archive_orphans(conn, archive_id) == 1
    assert db.one(conn, "SELECT COUNT(*) AS n FROM sounding")["n"] == 1


def test_an_unindexed_replacement_can_be_ingested_by_the_archive_pipeline(archive):
    from muf import pipeline
    from services.api import ingest

    conn, root, _, _ = archive
    reviewed(archive)
    assert conversion.run_next(conn)["state"] == "completed"
    result = ingest.ingest(
        [root],
        conn,
        options=pipeline.Options(methods=("contour",)),
        archive_root=root,
        jobs=1,
        progress=False,
    )
    assert result["loaded"] == 1, result
    row = db.one(conn, "SELECT file, format FROM sounding")
    assert row["file"].endswith(".echo.h5") and row["format"] == "echo_window"


def test_existing_hdf5_without_an_echo_group_is_not_modified(archive):
    import h5py

    conn, _, source, _ = archive
    target = source.with_suffix(".echo.h5")
    with h5py.File(target, "w") as handle:
        handle.create_dataset("unrelated", data=[1, 2, 3])
    checksum = conversion.digest(target)
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert conversion.digest(target) == checksum and source.exists()


def test_manual_existing_window_cannot_authorize_raw_deletion(archive):
    conn, _, source, _ = archive
    target = echo_window.write(source, manual_km=(2600, 3300), verify=True).target
    checksum = conversion.digest(target)
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists() and conversion.digest(target) == checksum


def test_echo_with_external_hdf5_samples_cannot_replace_raw(archive, tmp_path):
    import h5py

    conn, _, source, _ = archive
    target = echo_window.write(source).target
    external = tmp_path / "external-samples.h5"
    with h5py.File(target, "r+") as handle, h5py.File(external, "w") as other:
        other.create_dataset("samples", data=handle["echo_window/z"][()])
        del handle["echo_window/z"]
        handle["echo_window/z"] = h5py.ExternalLink(str(external), "/samples")
    checksum = conversion.digest(target)
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists() and conversion.digest(target) == checksum


def test_output_symlink_is_never_followed(archive, tmp_path):
    conn, _, source, _ = archive
    other = tmp_path / "unrelated.h5"
    other.write_bytes(b"other data")
    target = source.with_suffix(".echo.h5")
    try:
        target.symlink_to(other)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    reviewed(archive)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists() and other.read_bytes() == b"other data"


def test_failed_durable_flush_keeps_the_original(archive, monkeypatch):
    conn, _, source, _ = archive
    reviewed(archive)

    def unavailable(*args):
        raise OSError("flush failed")

    monkeypatch.setattr(conversion, "_durable_target", unavailable)
    assert conversion.run_next(conn)["failed"] == 1
    assert source.exists()


def test_watch_skips_an_incomplete_replacement_journal(archive, monkeypatch):
    conn, root, source, _ = archive
    index_raw(conn, root, source)
    reviewed(archive)
    unlink = Path.unlink

    def interrupted(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if path == source:
            raise OSError("interrupted after unlink")

    monkeypatch.setattr(Path, "unlink", interrupted)
    assert conversion.run_next(conn)["state"] == "failed"
    # A concurrent indexer must not create a second sounding during recovery.
    assert watch.find_new([root], conn, ("contour",), 0, archive_root=root)[0] == []


@pytest.fixture
def client(archive, monkeypatch):
    from fastapi.testclient import TestClient

    from services.api import archives, auth, main, net

    conn, _, _, _ = archive
    database_path = conn.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setattr(db, "DEFAULT_DB", Path(database_path))
    monkeypatch.setattr(auth, "READ_TOKEN", "")
    monkeypatch.setattr(auth, "CONTROL_TOKEN", "ctl")
    monkeypatch.setattr(archives, "DEFAULT_INTERVAL_S", 0)
    monkeypatch.setattr(main, "WARM_CENSUS", False)
    monkeypatch.setattr(net, "ENABLED", False)
    with TestClient(main.app) as api:
        yield api


AUTH = {"Authorization": "Bearer ctl"}


def test_conversion_tab_is_linked_and_translated(client):
    english = client.get("/ui/conversion?lang=en")
    russian = client.get("/ui/conversion?lang=ru")
    assert english.status_code == russian.status_code == 200
    assert 'href="/ui/conversion"' in english.text
    assert "LFS to echo archive" in english.text
    assert "Преобразование LFS" in russian.text


def test_job_controls_and_file_lists_require_archive_permission(client, archive):
    _, _, _, archive_id = archive
    assert client.get("/archive-conversions").status_code == 401
    assert (
        client.post("/archive-conversions", json={"archive_id": archive_id}).status_code
        == 401
    )
    teacher = client.post(
        "/principals", headers=AUTH, json={"name": "teacher", "role": "teacher"}
    ).json()["token"]
    assert (
        client.get(
            "/archive-conversions", headers={"Authorization": "Bearer " + teacher}
        ).status_code
        == 403
    )


def test_browser_can_inspect_review_start_and_read_the_result(client, archive):
    conn, _, source, archive_id = archive
    response = client.post(
        "/archive-conversions",
        headers=AUTH,
        json={"archive_id": archive_id, "subpath": "."},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    assert conversion.run_next(conn)["state"] == "ready"
    details = client.get(f"/archive-conversions/{job_id}", headers=AUTH).json()
    assert len(details["files"]) == 1 and source.exists()
    assert (
        client.post(
            f"/archive-conversions/{job_id}/start", headers=AUTH, json={}
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/archive-conversions/{job_id}/start",
            headers=AUTH,
            json={"confirm_delete": True},
        ).status_code
        == 202
    )
    assert conversion.run_next(conn)["state"] == "completed"
    details = client.get(f"/archive-conversions/{job_id}", headers=AUTH).json()
    assert details["files"][0]["state"] == "replaced" and not source.exists()
    assert details["job"]["approved_by"] == "control"


def test_browser_can_cancel_queued_job_without_touching_files(client, archive):
    _, _, source, archive_id = archive
    job = client.post(
        "/archive-conversions", headers=AUTH, json={"archive_id": archive_id}
    ).json()
    response = client.post(f"/archive-conversions/{job['id']}/cancel", headers=AUTH)
    assert response.json()["state"] == "cancelled" and source.exists()


def test_process_lock_excludes_a_second_worker(tmp_path):
    from services.archive_converter import worker_lock

    path = tmp_path / "conversion.lock"
    with worker_lock(path), pytest.raises(OSError), worker_lock(path):
        pytest.fail("second worker acquired the active worker's lock")
    with worker_lock(path):
        pass


def copy_capture(source, folder, day):
    folder.mkdir(parents=True, exist_ok=True)
    capture = folder / f"synth_{day.replace('.', '')}_000010.lfs"
    capture.write_bytes(source.read_bytes())
    os.utime(capture, (time.time() - 120,) * 2)
    return capture


def test_range_inspects_only_captures_in_its_sibling_folders(archive):
    conn, root, source, archive_id = archive
    for day in ("2026.02.03", "2026.02.05", "2026.02.06", "2026.02.07"):
        copy_capture(source, root / day, day)
    nested = copy_capture(source, root / "2026.02.05" / "nested", "2026.02.05")
    recent = source.with_name("recent.lfs")
    recent.write_bytes(source.read_bytes())
    job = conversion.enqueue(
        conn, archive_id, source.parent.name, "admin", last="2026.02.06"
    )
    conversion.run_next(conn)
    files = db.rows(
        conn, "SELECT path FROM archive_conversion_file WHERE job_id = ?", (job["id"],)
    )
    expected = {
        source.relative_to(root).as_posix(),
        "2026.02.05/synth_20260205_000010.lfs",
        "2026.02.06/synth_20260206_000010.lfs",
        nested.relative_to(root).as_posix(),
    }
    assert {file["path"] for file in files} == expected, (
        "Range inspection listed the wrong captures"
    )


def test_range_leaves_out_captures_directly_in_the_parent(archive):
    conn, root, source, archive_id = archive
    capture = copy_capture(source, root, "2026.02.01")
    job = conversion.enqueue(
        conn, archive_id, source.parent.name, "admin", last=source.parent.name
    )
    conversion.run_next(conn)
    files = db.rows(
        conn, "SELECT path FROM archive_conversion_file WHERE job_id = ?", (job["id"],)
    )
    assert capture.name not in {file["path"] for file in files}, (
        "Range inspection listed a capture in the parent"
    )


def test_reversed_range_is_refused(archive):
    conn, root, source, archive_id = archive
    (root / "2026.02.03").mkdir()
    with pytest.raises(conversion.ConversionError) as error:
        conversion.enqueue(
            conn, archive_id, source.parent.name, "admin", last="2026.02.03"
        )
    assert "sorts before" in str(error.value), (
        "Reversed range refusal omitted the ordering rule"
    )


@pytest.mark.parametrize(
    "last", ["2026.02.05", "2026.02.04/synth_20260204_000010.lfs"],
    ids=["missing", "file"],
)
def test_last_folder_must_be_an_existing_directory(archive, last):
    conn, _, source, archive_id = archive
    with pytest.raises(conversion.ConversionError) as error:
        conversion.enqueue(conn, archive_id, source.parent.name, "admin", last=last)
    assert "not a directory" in str(error.value), (
        "Invalid last folder refusal omitted the directory requirement"
    )


def test_first_folder_must_be_an_existing_directory(archive):
    conn, _, source, archive_id = archive
    with pytest.raises(conversion.ConversionError) as error:
        conversion.enqueue(
            conn, archive_id, "2026.02.03", "admin", last=source.parent.name
        )
    assert "not a directory" in str(error.value), (
        "Missing first folder refusal omitted the directory requirement"
    )


def test_range_folders_with_different_parents_are_refused(archive):
    conn, root, source, archive_id = archive
    folder = root / "nested" / "2026.02.05"
    folder.mkdir(parents=True)
    with pytest.raises(conversion.ConversionError) as error:
        conversion.enqueue(
            conn, archive_id, source.parent.name, "admin",
            last=folder.relative_to(root).as_posix(),
        )
    assert "same parent" in str(error.value), (
        "Range refusal omitted the common parent requirement"
    )


@pytest.mark.parametrize("subpath", [".", "./.", ""], ids=["dot", "dots", "empty"])
def test_the_archive_itself_cannot_start_a_range(archive, subpath):
    conn, _, source, archive_id = archive
    with pytest.raises(conversion.ConversionError) as error:
        conversion.enqueue(conn, archive_id, subpath, "admin", last=source.parent.name)
    assert "archive itself" in str(error.value), (
        "Range refusal omitted the archive root restriction"
    )


@pytest.mark.parametrize("endpoint", ["first", "last"])
def test_a_linked_folder_cannot_be_a_range_endpoint(archive, endpoint):
    conn, root, source, archive_id = archive
    link = root / "2026.02.05"
    try:
        link.symlink_to(source.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    first, last = (
        (link.name, source.parent.name)
        if endpoint == "first"
        else (source.parent.name, link.name)
    )
    with pytest.raises(conversion.ConversionError) as error:
        conversion.enqueue(conn, archive_id, first, "admin", last=last)
    assert "linked folders" in str(error.value), (
        "Linked range endpoint refusal omitted the link restriction"
    )


def make_junction(link, target):
    winapi = pytest.importorskip("_winapi")
    try:
        winapi.CreateJunction(str(target), str(link))
    except OSError:
        pytest.skip("junction creation is unavailable")


def test_range_inspection_skips_a_junction_to_an_outside_folder(archive):
    conn, root, source, archive_id = archive
    outside = copy_capture(source, root / "2026.02.07", "2026.02.07")
    (root / "2026.02.06").mkdir()
    make_junction(root / "2026.02.05", outside.parent)
    conversion.enqueue(conn, archive_id, source.parent.name, "admin", last="2026.02.06")
    job = conversion.run_next(conn)
    assert (job["state"], job["total"]) == ("ready", 1), (
        "Range inspection followed a junction outside the range"
    )


def test_a_junction_cannot_redirect_an_approved_range(archive):
    conn, root, source, archive_id = archive
    job = conversion.enqueue(
        conn, archive_id, source.parent.name, "admin", last=source.parent.name
    )
    conversion.run_next(conn)
    conversion.start(conn, job["id"], True)
    outside = source.parent.rename(root / "2026.02.07")
    make_junction(source.parent, outside)
    conversion.run_next(conn)
    assert (outside / source.name).is_file(), (
        "Range replacement followed a junction outside the approved range"
    )


def test_a_junction_cannot_redirect_the_parent_of_an_approved_range(archive):
    conn, root, source, archive_id = archive
    capture = copy_capture(source, root / "batch" / "2026.02.04", "2026.02.04")
    subpath = capture.parent.relative_to(root).as_posix()
    job = conversion.enqueue(conn, archive_id, subpath, "admin", last=subpath)
    conversion.run_next(conn)
    conversion.start(conn, job["id"], True)
    outside = (root / "batch").rename(root / "outside")
    make_junction(root / "batch", outside)
    conversion.run_next(conn)
    assert (outside / capture.relative_to(root / "batch")).is_file(), (
        "Range replacement followed a redirected parent"
    )


def test_an_approved_range_replaces_only_its_reviewed_captures(archive):
    conn, root, source, archive_id = archive
    second = copy_capture(source, root / "2026.02.05", "2026.02.05")
    outside = copy_capture(source, root / "2026.02.06", "2026.02.06")
    parent = copy_capture(source, root, "2026.02.01")
    job = conversion.enqueue(
        conn, archive_id, source.parent.name, "admin", last=second.parent.name
    )
    conversion.run_next(conn)
    conversion.start(conn, job["id"], True)
    conversion.run_next(conn)
    files = set(root.rglob("*.lfs")) | set(root.rglob("*.echo.h5"))
    expected = {
        outside, parent, source.with_suffix(".echo.h5"), second.with_suffix(".echo.h5"),
    }
    assert files == expected, "Range replacement changed the wrong set of captures"


def test_the_route_records_the_sibling_range(client, archive):
    _, root, source, archive_id = archive
    (root / "2026.02.05").mkdir()
    response = client.post(
        "/archive-conversions",
        headers=AUTH,
        json={
            "archive_id": archive_id,
            "subpath": source.parent.name,
            "last": "2026.02.05",
        },
    )
    job = response.json()
    assert (
        response.status_code,
        job.get("folder"),
        job.get("first_folder"),
        job.get("last_folder"),
    ) == (202, ".", source.parent.name, "2026.02.05"), (
        "The route did not record the range on its parent"
    )


@pytest.mark.parametrize("payload", [{}, {"last": None}, {"last": ""}])
def test_an_unspecified_last_folder_keeps_a_single_folder_job(client, archive, payload):
    _, _, source, archive_id = archive
    response = client.post(
        "/archive-conversions",
        headers=AUTH,
        json={"archive_id": archive_id, "subpath": source.parent.name, **payload},
    )
    job = response.json()
    assert (
        response.status_code,
        job.get("folder"),
        job.get("first_folder"),
        job.get("last_folder"),
    ) == (202, source.parent.name, None, None), (
        "An unspecified last folder changed the single folder job"
    )


def test_the_conversion_page_offers_an_optional_last_folder(client):
    page = client.get("/ui/conversion?lang=en")
    assert '<input id="conversion-last">' in page.text, (
        "The page is missing the optional last folder input"
    )


def test_startup_adds_nullable_range_columns_to_existing_jobs(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    conn = db.connect(database)
    try:
        with conn:
            conn.execute(
                "CREATE TABLE archive_conversion"
                " (id INTEGER PRIMARY KEY, folder TEXT NOT NULL, state TEXT NOT NULL)"
            )
            conn.execute("INSERT INTO archive_conversion VALUES (1, '.', 'ready')")
    finally:
        conn.close()
    with db.session(database) as conn:
        job = conversion.get(conn, 1)
    assert (job.get("first_folder", "absent"), job.get("last_folder", "absent")) == (
        None,
        None,
    ), "Startup did not add both nullable endpoints to the existing job"
