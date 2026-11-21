"""Downloading the archive for a range (#162): which windows, and the ZIP.

`services/publish/site/bundle.js` picks the windows a range and a step hold
and writes them into an uncompressed ZIP in the browser. It runs here under
Node, and Python's own `zipfile` reads what it wrote.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.publish import publisher

SITE = publisher.SITE_SRC

HARNESS = """
const fs = require("fs"), vm = require("vm");
const [site, job] = process.argv.slice(2);
vm.runInThisContext(fs.readFileSync(site + "/bundle.js", "utf8"));
const task = JSON.parse(fs.readFileSync(job, "utf8"));
if (task.op === "pick") {
  const kept = bundle.pick(task.soundings, Date.parse(task.from),
                           Date.parse(task.to), task.step);
  process.stdout.write(JSON.stringify(kept.map(function (e) { return e.t; })));
} else {
  const out = fs.openSync(task.out, "w");
  const writer = bundle.zip(function (chunk) { fs.writeSync(out, chunk); });
  let chain = Promise.resolve();
  task.files.forEach(function (f) {
    chain = chain.then(function () {
      return writer.add(f.name, new Uint8Array(fs.readFileSync(f.path)),
                        Date.parse(f.t));
    });
  });
  chain.then(function () { return writer.close(); })
    .then(function () { fs.closeSync(out); })
    .catch(function (error) { console.error(error); process.exit(1); });
}
"""


def _node(tmp_path, task):
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "harness.cjs"
    harness.write_text(HARNESS, encoding="utf-8")
    job = tmp_path / "job.json"
    job.write_text(json.dumps(task), encoding="utf-8")
    ran = subprocess.run(
        ["node", str(harness), str(SITE), str(job)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if ran.returncode:
        raise AssertionError(f"the bundle module failed: {ran.stderr[-800:]}")
    return ran.stdout


def _stamp(when):
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _soundings(rng, start, count, minutes=5, jitter=0):
    out = []
    for k in range(count):
        when = start + timedelta(
            minutes=k * minutes, seconds=int(rng.integers(0, jitter + 1))
        )
        out.append(
            {
                "t": _stamp(when),
                "file": when.strftime("%Y%m%dT%H%M%S.echo.h5"),
                "bytes": int(rng.integers(1_500_000, 1_600_000)),
            }
        )
    return out


def _pick(tmp_path, soundings, start, end, step):
    return json.loads(
        _node(
            tmp_path,
            {
                "op": "pick",
                "soundings": soundings,
                "from": _stamp(start),
                "to": _stamp(end),
                "step": step,
            },
        )
    )


def test_a_step_of_fifteen_minutes_keeps_every_third_five_minute_sounding(tmp_path):
    rng = np.random.default_rng()
    start = datetime(2026, 2, 4, int(rng.integers(0, 12)), 0, 10, tzinfo=timezone.utc)
    day = _soundings(rng, start, 120)
    kept = _pick(tmp_path, day, start, start + timedelta(hours=10), 900)
    assert kept == [e["t"] for e in day[::3]], f"a 15-minute step kept {kept}"


def test_each_step_takes_the_first_sounding_at_or_after_it(tmp_path):
    rng = np.random.default_rng()
    start = datetime(2026, 2, 4, 0, 0, tzinfo=timezone.utc)
    day = _soundings(rng, start + timedelta(seconds=7), 200, jitter=170)
    kept = _pick(tmp_path, day, start, start + timedelta(days=1), 3600)
    want = []
    for hour in range(24):
        mark = start + timedelta(hours=hour)
        first = next((e["t"] for e in day if e["t"] >= _stamp(mark)), None)
        if first and first not in want:
            want.append(first)
    assert kept == want, (
        f"the hourly step did not take the first sounding of each hour: {kept}"
    )


def test_a_sounding_without_a_window_is_never_picked(tmp_path):
    rng = np.random.default_rng()
    start = datetime(2026, 2, 4, 3, 0, 10, tzinfo=timezone.utc)
    day = _soundings(rng, start, 60)
    gone = {int(k) for k in rng.choice(60, 20, replace=False)}
    for k in gone:
        day[k].pop("file")
    kept = _pick(tmp_path, day, start, start + timedelta(hours=6), 0)
    assert not {day[k]["t"] for k in gone} & set(kept), (
        "a sounding without a window was picked"
    )


def test_nothing_outside_the_range_is_picked(tmp_path):
    rng = np.random.default_rng()
    start = datetime(2026, 2, 4, 0, 0, 10, tzinfo=timezone.utc)
    day = _soundings(rng, start, 288)
    low, high = sorted(int(x) for x in rng.choice(288, 2, replace=False))
    kept = _pick(
        tmp_path,
        day,
        start + timedelta(minutes=5 * low),
        start + timedelta(minutes=5 * high),
        0,
    )
    assert kept == [e["t"] for e in day[low : high + 1]], f"the range kept {kept}"


@pytest.fixture
def windows(tmp_path):
    rng = np.random.default_rng()
    files = []
    for k in range(int(rng.integers(3, 7))):
        path = tmp_path / f"w{k}.echo.h5"
        path.write_bytes(rng.bytes(int(rng.integers(1, 300_000))))
        files.append(
            {
                "name": f"cyprus1-yoshkar-ola/2026020{k}T0{k}0510.echo.h5",
                "path": str(path),
                "t": f"2026-02-0{k + 1}T0{k}:05:10Z",
            }
        )
    return files


def _zip(tmp_path, windows):
    out = tmp_path / "range.zip"
    _node(tmp_path, {"op": "zip", "files": windows, "out": str(out)})
    return zipfile.ZipFile(out)


def test_the_zip_holds_each_window_byte_for_byte(tmp_path, windows):
    with _zip(tmp_path, windows) as archive:
        same = [
            archive.read(w["name"]) == Path(w["path"]).read_bytes() for w in windows
        ]
    assert all(same), f"a window in the ZIP differs from its file: {same}"


def test_the_zip_passes_its_own_checksums(tmp_path, windows):
    with _zip(tmp_path, windows) as archive:
        broken = archive.testzip()
    assert broken is None, f"the ZIP failed its checksum at {broken}"


def test_the_zip_names_each_window_in_order(tmp_path, windows):
    with _zip(tmp_path, windows) as archive:
        names = archive.namelist()
    assert names == [w["name"] for w in windows], f"the ZIP named its windows {names}"


def test_the_zip_dates_each_window_by_its_sounding(tmp_path, windows):
    with _zip(tmp_path, windows) as archive:
        dates = [i.date_time for i in archive.infolist()]
    want = [
        datetime.fromisoformat(w["t"].replace("Z", "")).timetuple()[:6] for w in windows
    ]
    assert dates == want, f"the ZIP dated its windows {dates}"


def test_the_page_offers_a_range_download():
    page = (SITE / "index.html").read_text(encoding="utf-8")
    assert all(
        f'id="{name}"' in page
        for name in ("range-from", "range-to", "range-step", "range-list", "range-zip")
    ), "the archive section offers no range download"


def test_the_page_loads_the_bundle_module_before_its_own_script():
    page = (SITE / "index.html").read_text(encoding="utf-8")
    assert 0 <= page.find('src="bundle.js"') < page.find('src="app.js"'), (
        "the page does not load bundle.js before app.js"
    )
