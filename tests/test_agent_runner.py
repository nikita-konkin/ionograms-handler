"""The agent's pass: the ionozond block goes whole only when it has changed."""

from __future__ import annotations

import json
import random
import time
from types import SimpleNamespace

import pytest

from services.agent import health, ionozond, runner
from services.agent.config import StationConfig


class Reply:
    """A server that takes every push and has no commands."""

    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode()

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def block(**extra) -> dict:
    return dict({"base": "%064x" % random.getrandbits(256),
                 "sounder_hash": "%064x" % random.getrandbits(256),
                 "site_hash": "%064x" % random.getrandbits(256),
                 "pending": False, "drift": False, "problems": [],
                 "document": {"schedule": {}, "general": {
                     "keep_lfs_hours": str(random.randint(1, 48))}}}, **extra)


def pushed(monkeypatch, blocks: list, sounder, refused: int = 0) -> list:
    sent, feed, calls = [], iter(blocks), [0]
    monkeypatch.setattr(health, "collect", lambda config, **kw: health.HealthReport(
        station="yoshkar-ola", timestamp=time.time(), sounder_config=next(feed)))

    def opener(request, timeout=None):
        if request.full_url.endswith("/stations/health"):
            calls[0] += 1
            if calls[0] <= refused:
                raise OSError("connection refused")
            sent.append(json.loads(request.data.decode())["sounder_config"])
        return Reply({"commands": []} if request.full_url.endswith("/commands")
                     else {"ok": True})

    config = StationConfig(station="yoshkar-ola", server_url="http://server/api",
                           preview=False)
    for _ in blocks:
        runner.run_once(config, opener=opener, include_epoch=False, sounder=sounder)
    return sent


def test_an_unchanged_block_goes_short_on_the_next_pass(monkeypatch):
    shown = block()
    sent = pushed(monkeypatch, [shown, dict(shown)], runner.SounderState())
    assert "document" not in sent[1], "an unchanged block was sent whole again"


def test_a_short_report_carries_the_summary(monkeypatch):
    shown = block()
    sent = pushed(monkeypatch, [shown, dict(shown)], runner.SounderState())
    assert sent[1] == ionozond.summary(shown), (
        "a short report did not carry the block's summary")


@pytest.mark.parametrize("key,value", [
    ("base", "0" * 64), ("sounder_hash", "1" * 64), ("site_hash", "2" * 64),
    ("pending", True), ("drift", True), ("problems", ["web.env is unreadable"])])
def test_a_change_in_the_summary_sends_the_block_whole(monkeypatch, key, value):
    shown = block()
    sent = pushed(monkeypatch, [shown, dict(shown, **{key: value})],
                  runner.SounderState())
    assert "document" in sent[1], "a changed %s did not send the block whole" % key


def test_the_block_goes_whole_once_an_hour(monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=lambda: clock[0]))
    state, shown = runner.SounderState(), block()
    pushed(monkeypatch, [shown], state)
    clock[0] += ionozond.WHOLE_S + random.uniform(1.0, 600.0)
    assert "document" in pushed(monkeypatch, [dict(shown)], state)[0], (
        "an hour on, an unchanged block still went short")


def test_a_refused_push_leaves_the_next_block_whole(monkeypatch):
    shown = block()
    sent = pushed(monkeypatch, [shown, dict(shown)], runner.SounderState(), refused=1)
    assert "document" in sent[0], "the block after a refused push went short"


def test_a_pass_without_state_sends_the_block_whole(monkeypatch):
    shown = block()
    sent = pushed(monkeypatch, [shown, dict(shown)], None)
    assert "document" in sent[1], "a pass with no state sent the block short"


def test_the_health_loop_sends_an_unchanged_block_short(monkeypatch):
    shown, sent = block(), []
    monkeypatch.setattr(health, "collect", lambda config, **kw: health.HealthReport(
        station="yoshkar-ola", timestamp=time.time(), sounder_config=dict(shown)))

    def opener(request, timeout=None):
        if request.full_url.endswith("/stations/health"):
            sent.append(json.loads(request.data.decode())["sounder_config"])
        return Reply({"commands": []} if request.full_url.endswith("/commands")
                     else {"ok": True})

    runner.run_forever(StationConfig(station="yoshkar-ola",
                                     server_url="http://server/api", preview=False),
                       opener=opener, sleep=lambda s: None, max_passes=2)
    assert "document" not in sent[1], "the health loop sent an unchanged block whole"
