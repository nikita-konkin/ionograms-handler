"""services.api.apart: work in a process of its own, stopped if it hangs."""

from __future__ import annotations

import math
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
