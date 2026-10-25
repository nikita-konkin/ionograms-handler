"""services.api.apart: work in a process of its own, stopped if it hangs."""

from __future__ import annotations

import math
import random
import time

import pytest

from services.api import apart


def test_the_work_gives_back_its_value():
    assert apart.run(math.sqrt, 16.0, timeout_s=60.0, what="sqrt") == 4.0


def test_a_failure_comes_back_naming_the_work():
    with pytest.raises(RuntimeError, match="sqrt: ValueError"):
        apart.run(math.sqrt, -1.0, timeout_s=60.0, what="sqrt")


def test_work_past_its_time_is_stopped():
    with pytest.raises(TimeoutError, match=r"sleep ran past 0\.5 s"):
        apart.run(time.sleep, 30.0, timeout_s=0.5, what="sleep")


def test_a_progress_is_read_back_as_it_was_written(tmp_path):
    done, total = random.randint(1, 50), random.randint(51, 99)
    apart.Progress(tmp_path / "x.progress", time.time())(done, total)
    got = apart.progress(tmp_path / "x.progress")
    assert (got["done"], got["total"]) == (done, total), \
        "the progress was not read back as it was written"


def test_a_progress_says_how_long_the_work_has_run(tmp_path):
    apart.Progress(tmp_path / "x.progress", time.time() - 42.0)(1, 2)
    assert apart.progress(tmp_path / "x.progress")["elapsed_s"] >= 42.0, \
        "the progress did not say how long the work has run"


def test_no_progress_reads_as_none(tmp_path):
    assert apart.progress(tmp_path / "never.progress") is None, \
        "a progress never written was read as something"


def _report(progress) -> int:
    progress(7, 11)
    return 0


def test_a_progress_written_in_the_works_own_process_is_read_back(tmp_path):
    apart.run(_report, apart.Progress(tmp_path / "x.progress", time.time()),
              timeout_s=60.0, what="report")
    assert apart.progress(tmp_path / "x.progress")["done"] == 7, \
        "a progress written in the work's own process was not read back"


def test_a_progress_written_while_it_is_read_does_not_stop_the_work(tmp_path):
    total = random.randint(2, 99)
    tell = apart.Progress(tmp_path / "x.progress", time.time())
    tell(1, total)
    with open(tmp_path / "x.progress", encoding="utf-8"):
        tell(2, total)
    assert apart.progress(tmp_path / "x.progress")["total"] == total, \
        "a progress written while it was read stopped the work"
