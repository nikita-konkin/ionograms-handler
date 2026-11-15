"""The browser's transform of an echo window (#152, docs/echo-window.md 3.9).

`services/publish/site/echo.js` reads a published window with the vendored
h5wasm and transforms the stored samples as they are. It runs here under
Node against `muf.echo_window.spectrogram`, the same transform in Python.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import make_header_bytes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muf import echo_window, spectro
from services.publish import publisher

SITE = publisher.SITE_SRC

HARNESS = """
const fs = require("fs"), vm = require("vm");
const [site, file, length, pad] = process.argv.slice(2);
vm.runInThisContext(fs.readFileSync(site + "/vendor/h5wasm.js", "utf8"));
vm.runInThisContext(fs.readFileSync(site + "/echo.js", "utf8"));
h5wasm.ready.then(function () {
  const found = echo.spectrogram(
    echo.read(h5wasm, new Uint8Array(fs.readFileSync(file))),
    Number(length), Number(pad));
  process.stdout.write(JSON.stringify({
    rows: found.rows, bins: found.bins, points: found.points,
    power: Array.from(found.power), ranges: Array.from(found.ranges),
    sweep: Array.from(found.sweep)}));
}).catch(function (error) { console.error(error); process.exit(1); });
"""


@pytest.fixture
def window(tmp_path):
    rng = np.random.default_rng(13)
    n = 617_000
    beat = -2971.0 / 3 + 9.0 * np.arange(n) / n
    capture = tmp_path / "synth_20260204_000010.lfs"
    capture.write_bytes(make_header_bytes() + (
        rng.normal(size=n) + 1j * rng.normal(size=n)
        + 27 * np.exp(2j * np.pi * beat * np.arange(n) / 40000)
    ).astype(np.complex64).tobytes())
    return echo_window.read(echo_window.write(capture).target)


def _browser(window, tmp_path, length, pad):
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "harness.cjs"
    harness.write_text(HARNESS, encoding="utf-8")
    ran = subprocess.run(
        ["node", str(harness), str(SITE), str(window.path), str(length), str(pad)],
        capture_output=True, text=True, timeout=120)
    if ran.returncode:
        raise AssertionError(f"the browser transform failed: {ran.stderr[-800:]}")
    return json.loads(ran.stdout)


def _length(window):
    return round(spectro.DEFAULT_WINDOW / window.decimation)


def test_the_browser_draws_what_the_reference_draws(window, tmp_path):
    length = _length(window)
    drawn = _browser(window, tmp_path, length, 0)
    power, _, _ = echo_window.spectrogram(window, length, fft=drawn["points"])
    assert np.abs(spectro.to_db(np.array(drawn["power"]).reshape(power.shape))
                  - spectro.to_db(power)).max() < 1e-3, \
        "the browser does not draw what the reference draws"


def test_the_browser_pads_to_the_next_power_of_two(window, tmp_path):
    length = _length(window)
    assert _browser(window, tmp_path, length, 3)["points"] == \
        2 ** int(np.ceil(np.log2(4 * length))), \
        "the browser does not pad to the next power of two"


def test_the_browser_puts_each_bin_at_its_range(window, tmp_path):
    length = _length(window)
    drawn = _browser(window, tmp_path, length, 1)
    _, ranges, _ = echo_window.spectrogram(window, length, fft=drawn["points"])
    assert drawn["ranges"] == pytest.approx(list(ranges), abs=1e-6), \
        "the browser does not put each bin at its range"


def test_the_browser_labels_each_row_with_its_sweep(window, tmp_path):
    length = _length(window) // 2
    drawn = _browser(window, tmp_path, length, 0)
    _, _, sweep = echo_window.spectrogram(window, length, fft=drawn["points"])
    assert drawn["sweep"] == pytest.approx(list(sweep), abs=1e-3), \
        "the browser does not label each row with its sweep"


def test_the_browser_refuses_a_file_that_is_not_a_window(window, tmp_path):
    window.path.write_bytes(b"not a window at all, 43 bytes of plain text")
    with pytest.raises(AssertionError, match="browser transform failed"):
        _browser(window, tmp_path, 64, 0)


def test_the_reader_is_shipped_with_its_licence():
    assert "NIST" in (SITE / "vendor" / "h5wasm.LICENSE.txt").read_text(
        encoding="utf-8"), "the reader is shipped without its licence"


@pytest.mark.parametrize("name", ("echo.js", "vendor/h5wasm.js"))
def test_the_reader_loads_nothing_from_elsewhere(name):
    text = (SITE / name).read_text(encoding="utf-8")
    assert not any(mark in text for mark in ("https://", "http://", "eval(",
                                             "new Function")), \
        f"{name} reaches outside the page or evaluates text"


def test_the_page_offers_the_archive():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert 'id="archive"' in html and 'src="echo.js"' in html, \
        "the page does not offer the archive"


def test_the_page_draws_the_archive_itself():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert '<canvas id="past-canvas"' in html, \
        "the page has nothing to draw a past sounding on"


def test_the_reader_is_fetched_only_on_demand():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert "h5wasm" not in html and "vendor/h5wasm.js" in (
        SITE / "app.js").read_text(encoding="utf-8"), \
        "every visitor would fetch the reader, wanted or not"


@pytest.mark.parametrize("lang", ("en", "ru"))
def test_the_archive_is_named_in_each_language(lang):
    strings = json.loads((SITE / "strings.json").read_text(encoding="utf-8"))[lang]
    wanted = {"archive", "archive_hint", "archive_circuit", "archive_day",
              "archive_time", "archive_row", "archive_pad", "archive_low",
              "archive_high", "archive_download", "archive_loading",
              "archive_failed", "archive_none", "archive_caption",
              "archive_edge", "archive_axis_f", "archive_axis_r"}
    assert wanted <= set(strings), f"{lang} lacks {sorted(wanted - set(strings))}"


@pytest.mark.parametrize("name", ("app.js", "echo.js"))
def test_the_page_scripts_parse(name):
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    ran = subprocess.run(["node", "--check", str(SITE / name)],
                         capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, f"{name} does not parse: {ran.stderr[-400:]}"


def test_the_vhost_lets_the_reader_run():
    vhost = (Path(__file__).resolve().parents[1] / "deploy"
             / "public-site.nginx.conf").read_text(encoding="utf-8")
    assert "script-src 'self';" not in vhost and \
        "script-src 'self' 'wasm-unsafe-eval';" in vhost, \
        "the vhost's policy stops the reader's WebAssembly"
