"""The public page: what it publishes, and -- the point -- what it does not.

The page is served to the internet from a folder. Everything here pins one of
the reasons that folder is safe to serve: only allow-listed, unmuted circuits;
only allow-listed fields; never half a file; and a bad sounding costs its own
picture, not the page.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.api import db, ingest
from services.publish import publisher

NOW = datetime(2026, 8, 13, 9, 30, tzinfo=timezone.utc)

#: Strings that must never reach the public folder.
SECRET_ARTIFACT = "/models/objects/ab/do-not-publish.joblib"
SECRET_SHA = "f" * 64


def _sounding(conn, archive, name, when, *, tx="NIC", rx="DOB", muf=20.0, lof=8.0):
    (archive / name).write_bytes(b"x")
    ingest.ingest_row(conn, {"file": name, "datetime": when, "tx": tx, "rx": rx,
                             "format": "chirp2", "muf_contour": muf,
                             "lof_contour": lof},
                      archive / name, archive, ("contour",))


def _model(conn, *, tx="NIC", rx="DOB", name="live-model", active=1):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(model_registry)")}
    row = {"name": name, "param": "muf", "origin": "trained",
           "framework": "xgboost", "loader": "joblib",
           "artifact": SECRET_ARTIFACT, "sha256": SECRET_SHA if active else "e" * 64,
           "features": '["x"]', "target_src": "measured" if active else "modelled",
           "imported_at": "2026-08-13T08:00:00Z", "active": active,
           "tx": tx, "rx": rx,
           "trained_from": "2026-07-01T00:00:00Z", "trained_to": "2026-08-12T00:00:00Z"}
    row = {k: v for k, v in row.items() if k in cols}
    conn.execute(f"INSERT INTO model_registry ({','.join(row)})"
                 f" VALUES ({','.join('?' * len(row))})", tuple(row.values()))
    model_id = conn.execute("SELECT id FROM model_registry WHERE name = ?",
                            (name,)).fetchone()[0]
    for hour in range(10, 14):
        conn.execute("INSERT INTO forecast (model_id, param, tx, rx, issued_at,"
                     " valid_at, horizon_s, value, sigma)"
                     " VALUES (?, 'muf', ?, ?, '2026-08-13T09:00:00Z', ?, ?, ?, 1.5)",
                     (model_id, tx, rx, f"2026-08-13T{hour:02d}:00:00Z",
                      (hour - 9) * 3600, 21.0 + hour / 10))
    for subject in (f"model:{model_id}", "baseline:persistence"):
        conn.execute("INSERT OR IGNORE INTO score (subject, param, tx, rx,"
                     " horizon_s, scored_at, n, mae, detail)"
                     " VALUES (?, 'muf', ?, ?, 3600, '2026-08-13T09:00:00Z',"
                     " 40, 1.2, ?)",
                     (subject, tx, rx,
                      json.dumps({"vs_persistence": {"skill": 0.1,
                                                     "distinguishable": True}})))
    conn.commit()
    return model_id


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    return root


@pytest.fixture
def database(tmp_path, archive):
    """A populated api database: two allowed circuits, one not listed, one
    muted after ingest, and an active model with scores on NIC -> DOB."""
    path = tmp_path / "api.sqlite3"
    with db.session(path) as conn:
        for i, hour in enumerate((6, 7, 8, 9)):
            _sounding(conn, archive, f"nic{i}.h5", f"2026-08-13 0{hour}:05:00")
        _sounding(conn, archive, "sgo0.h5", "2026-08-13 09:01:00", tx="SGO")
        _sounding(conn, archive, "jul0.h5", "2026-08-13 09:02:00", tx="Juliusruh")
        _sounding(conn, archive, "chi0.h5", "2026-08-13 09:03:00", tx="Chilton")
        _model(conn)
        _model(conn, name="candidate", active=0)
        # Muted only now: ingest refuses a muted circuit, but rows held from
        # before the rule are still in the database.
        db.mute_circuit(conn, "Chilton", "DOB")
    return path


@pytest.fixture
def drawn(monkeypatch):
    """Replace the renderer with a recorder. Returns the list of paths drawn."""
    calls = []

    def fake(path):
        calls.append(Path(path).name)
        return b"RIFF....WEBP" + Path(path).name.encode()

    monkeypatch.setattr(publisher, "render_webp", fake)
    return calls


def _publish(database, site, archive, circuits="NIC->DOB, sgo->dob, Chilton->DOB",
             now=NOW):
    conn = publisher.connect_ro(database)
    try:
        return publisher.build_site(conn, site,
                                    circuits=publisher.parse_circuits(circuits),
                                    method="contour", days=30,
                                    archive_root=archive, now=now)
    finally:
        conn.close()


def _everything(site: Path) -> str:
    """Every text file in the folder. Images are skipped: the fake renderer
    stamps the product name into its bytes so the tests can tell them apart."""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in site.rglob("*") if p.is_file() and p.suffix != ".webp")


# --------------------------------------------------------------------------
# What appears
# --------------------------------------------------------------------------

def test_only_allow_listed_unmuted_circuits_are_published(database, archive,
                                                          tmp_path, drawn):
    site = tmp_path / "site"
    _publish(database, site, archive)

    status = json.loads((site / "status.json").read_text(encoding="utf-8"))
    assert [c["slug"] for c in status["circuits"]] == ["nic-dob", "sgo-dob"]
    assert sorted(p.name for p in (site / "data").iterdir()) == [
        "nic-dob.json", "sgo-dob.json"]
    # Juliusruh was never listed; Chilton was listed and then muted.
    assert "Juliusruh" not in _everything(site)
    assert "Chilton" not in json.dumps(status)


def test_the_allow_list_ignores_case(database, archive, tmp_path, drawn):
    assert publisher.parse_circuits(" nic->DOB ,bad, ->x, SGO -> dob") == [
        ("nic", "dob"), ("sgo", "dob")]
    _publish(database, tmp_path / "site", archive, circuits="nic->dob")
    assert (tmp_path / "site" / "data" / "nic-dob.json").exists()


def test_status_names_the_circuit_and_its_newest_pick(database, archive,
                                                      tmp_path, drawn):
    site = tmp_path / "site"
    _publish(database, site, archive)
    status = json.loads((site / "status.json").read_text(encoding="utf-8"))

    assert status["generated_at"] == "2026-08-13T09:30:00Z"
    nic = status["circuits"][0]
    assert nic["tx_name"] == "Nicosia, Cyprus"
    assert nic["last_sounding"] == "2026-08-13T09:05:00Z"
    assert nic["latest_pick"]["muf"] == 20.0
    assert [i["t"] for i in nic["images"]] == [
        "2026-08-13T06:05:00Z", "2026-08-13T07:05:00Z",
        "2026-08-13T08:05:00Z", "2026-08-13T09:05:00Z"]


def test_the_forecast_is_the_active_model_and_the_baselines_only(
        database, archive, tmp_path, drawn):
    site = tmp_path / "site"
    _publish(database, site, archive)
    data = json.loads((site / "data" / "nic-dob.json").read_text(encoding="utf-8"))

    assert data["forecast"]["model"]["name"] == "live-model"
    assert data["forecast"]["model"]["issued_at"] == "2026-08-13T09:00:00Z"
    assert {s["name"] for s in data["forecast"]["scores"]} == {"live-model",
                                                               "persistence"}
    assert [f["model"] for f in data["series"]["forecast"]] == ["live-model"]
    assert len(data["series"]["forecast"][0]["value"]) == 4
    assert "candidate" not in _everything(site)


def test_nothing_internal_reaches_the_public_folder(database, archive,
                                                    tmp_path, drawn):
    site = tmp_path / "site"
    _publish(database, site, archive)
    published = _everything(site)

    for secret in (SECRET_ARTIFACT, SECRET_SHA, str(archive), archive.as_posix(),
                   "nic0.h5", "golden", "sha256", "artifact"):
        assert secret not in published, secret
    series = json.loads((site / "data" / "nic-dob.json").read_text(
        encoding="utf-8"))["series"]
    assert "id" not in series


# --------------------------------------------------------------------------
# How it is written
# --------------------------------------------------------------------------

def test_each_sounding_is_drawn_once(database, archive, tmp_path, drawn):
    site = tmp_path / "site"
    first = _publish(database, site, archive)
    assert first["rendered"] == 5 and len(drawn) == 5      # 4 NIC + 1 SGO

    second = _publish(database, site, archive)
    assert second["rendered"] == 0 and len(drawn) == 5


def test_a_sounding_that_cannot_be_drawn_costs_only_its_picture(
        database, archive, tmp_path, monkeypatch):
    def flaky(path):
        if Path(path).name == "nic2.h5":
            raise OSError("Host is down")
        return b"RIFF....WEBP"

    monkeypatch.setattr(publisher, "render_webp", flaky)
    site = tmp_path / "site"
    result = _publish(database, site, archive)

    assert result["skipped"] == 1 and result["rendered"] == 4
    assert "Host is down" in publisher.describe(result)
    status = json.loads((site / "status.json").read_text(encoding="utf-8"))
    assert len(status["circuits"][0]["images"]) == 3


def test_pictures_and_circuits_that_drop_out_are_pruned(database, archive,
                                                        tmp_path, drawn):
    site = tmp_path / "site"
    _publish(database, site, archive)
    assert len(list((site / "img" / "nic-dob").glob("*.webp"))) == 4

    # A day on, nothing is inside the 24 h strip; and SGO is delisted.
    later = datetime(2026, 8, 14, 9, 30, tzinfo=timezone.utc)
    result = _publish(database, site, archive, circuits="NIC->DOB", now=later)

    assert list(site.glob("img/**/*.webp")) == []
    assert not (site / "data" / "sgo-dob.json").exists()
    assert result["pruned"] == 6                         # 5 images + 1 json


def test_no_temporary_files_are_left_behind(database, archive, tmp_path, drawn):
    site = tmp_path / "site"
    site.mkdir()
    (site / f"{publisher.TMP_PREFIX}status.json.999").write_text("half")
    _publish(database, site, archive)
    assert list(site.rglob(f"{publisher.TMP_PREFIX}*")) == []


def test_an_unchanged_file_is_not_rewritten(database, archive, tmp_path, drawn):
    """Rewriting identical bytes would change the mtime, and with it the
    ETag, so every visitor would re-download data that had not changed."""
    site = tmp_path / "site"
    _publish(database, site, archive)
    data = site / "data" / "nic-dob.json"
    page = site / "index.html"
    before = data.stat().st_mtime_ns, page.stat().st_mtime_ns
    os.utime(data, ns=(0, before[0] - 10**9))
    os.utime(page, ns=(0, before[1] - 10**9))
    stamped = data.stat().st_mtime_ns, page.stat().st_mtime_ns

    _publish(database, site, archive)
    assert (data.stat().st_mtime_ns, page.stat().st_mtime_ns) == stamped


def test_the_page_itself_is_copied(database, archive, tmp_path, drawn):
    site = tmp_path / "site"
    _publish(database, site, archive)
    for name in ("index.html", "app.js", "style.css", "strings.json"):
        assert (site / name).read_bytes() == (publisher.SITE_SRC / name).read_bytes()


def test_the_page_loads_nothing_from_elsewhere():
    """The vhost sends `default-src 'self'`; one CDN link breaks the page."""
    for name in ("index.html", "app.js", "style.css"):
        text = (publisher.SITE_SRC / name).read_text(encoding="utf-8")
        assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")
        assert "https://" not in text
    html = (publisher.SITE_SRC / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "style=" not in html


def test_both_languages_carry_the_same_strings():
    strings = json.loads((publisher.SITE_SRC / "strings.json").read_text(
        encoding="utf-8"))
    assert set(strings["en"]) == set(strings["ru"])


# --------------------------------------------------------------------------
# The database and the command line
# --------------------------------------------------------------------------

def test_the_database_is_opened_read_only(database):
    conn = publisher.connect_ro(database)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM sounding")
    finally:
        conn.close()


def test_main_runs_one_pass(database, archive, tmp_path, drawn, capsys):
    site = tmp_path / "site"
    code = publisher.main(["--db", str(database), "--out", str(site),
                           "--circuits", "NIC->DOB", "--archive-root", str(archive)])
    assert code == 0
    assert "published 1 circuit" in capsys.readouterr().out
    assert (site / "status.json").exists()


def test_main_refuses_to_publish_without_an_allow_list(tmp_path, capsys):
    assert publisher.main(["--db", str(tmp_path / "x.sqlite3"),
                           "--out", str(tmp_path / "site"), "--circuits", ""]) == 2
    assert "PUBLISH_CIRCUITS" in capsys.readouterr().err


def test_a_missing_database_fails_the_pass_and_says_so(tmp_path, capsys):
    code = publisher.main(["--db", str(tmp_path / "absent.sqlite3"),
                           "--out", str(tmp_path / "site"), "--circuits", "A->B"])
    assert code == 1
    assert "pass failed" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The real renderer
# --------------------------------------------------------------------------

def test_a_real_product_renders_to_webp(tmp_path, make_chirp_h5):
    pytest.importorskip("PIL")
    power = np.full((64, 256), 100.0)
    power[20:40, 120:124] = 5e4
    path = make_chirp_h5(power)

    image = publisher.render_webp(path)

    assert image[:4] == b"RIFF" and image[8:12] == b"WEBP"
    assert len(image) < 200_000
