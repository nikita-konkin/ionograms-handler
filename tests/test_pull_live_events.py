"""Pulling the hub's live forecasts: every event once, none lost, none twice.

The hub deletes its live events after LIVE_KEEP_DAYS, so the copy this pull
keeps is the only lasting one. What matters is that a pull carries on exactly
where the file stops, and pages through a long history without a gap.

Loaded by path: `tools/` is deliberately not one of the shipped packages.
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import threading
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "pull_live_events", REPO / "tools" / "pull_live_events.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pull = _load()


def _events(ids) -> list[dict]:
    return [{"schema": "muf.sim.live.event", "version": 1, "event": "issued",
             "id": k, "tx": "NIC2", "rx": "yoshkar-ola"} for k in ids]


def _hub(held: list[dict], asked: list[str] | None = None):
    def get(url: str) -> list[str]:
        if asked is not None:
            asked.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        after, limit = int(query["after"][0]), int(query["limit"][0])
        return [json.dumps(e) for e in held if e["id"] > after][:limit]
    return get


def _ids(path: Path) -> list[int]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


def test_a_pull_writes_every_event_the_hub_holds(tmp_path):
    path = tmp_path / "NIC2__yoshkar-ola.jsonl"
    pull.pull("http://hub", "NIC2", "yoshkar-ola", path,
              get=_hub(_events([3, 8, 13, 21])))
    assert _ids(path) == [3, 8, 13, 21], "the pull did not write every event"


def test_a_second_pull_adds_only_what_is_new(tmp_path):
    path, held = tmp_path / "NIC2__yoshkar-ola.jsonl", _events([4, 9])
    pull.pull("http://hub", "NIC2", "yoshkar-ola", path, get=_hub(held))
    held.extend(_events([17, 29]))
    pull.pull("http://hub", "NIC2", "yoshkar-ola", path, get=_hub(held))
    assert _ids(path) == [4, 9, 17, 29], "the second pull wrote an event twice"


def test_a_pull_pages_through_a_long_history(tmp_path):
    path = tmp_path / "NIC2__yoshkar-ola.jsonl"
    pull.pull("http://hub", "NIC2", "yoshkar-ola", path,
              get=_hub(_events([2, 5, 6, 11, 12, 30, 31])), page=3)
    assert _ids(path) == [2, 5, 6, 11, 12, 30, 31], "a page of the history was lost"


def test_a_pull_carries_on_after_the_last_id_the_file_holds(tmp_path):
    path, asked = tmp_path / "NIC2__yoshkar-ola.jsonl", []
    path.write_text(json.dumps(_events([57])[0]) + "\n")
    pull.pull("http://hub", "NIC2", "yoshkar-ola", path, get=_hub([], asked))
    assert "after=57" in asked[0], "the pull did not carry on after the file's last id"


def test_a_pull_asks_for_the_circuit_named(tmp_path):
    asked = []
    pull.pull("http://hub/", "NIC2", "yoshkar-ola", tmp_path / "x.jsonl",
              get=_hub([], asked))
    assert asked[0].startswith("http://hub/sim/live/NIC2/yoshkar-ola/events?"), \
        "the pull asked for another route"


def test_a_pull_says_how_many_events_it_added(tmp_path):
    assert pull.pull("http://hub", "NIC2", "yoshkar-ola", tmp_path / "x.jsonl",
                     get=_hub(_events([1, 4, 6]))) == 3, "the pull miscounted"


def test_a_circuit_is_kept_under_its_own_name(tmp_path):
    assert pull.target(tmp_path, "NIC2", "yoshkar-ola").name \
        == "NIC2__yoshkar-ola.jsonl", "the circuit's file is misnamed"


def _serve(body: bytes, seen: dict) -> http.server.HTTPServer:
    class Hub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.update(self.headers)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Hub)
    server.timeout = 10.0
    threading.Thread(target=server.handle_request, daemon=True).start()
    return server


def test_a_page_is_read_off_the_hub_line_by_line():
    server = _serve(b'{"id": 4, "event": "issued"}\n\n'
                    b'{"id": 7, "event": "scored"}\n', {})
    lines = pull.get(f"http://127.0.0.1:{server.server_port}/sim/live/a/b/events")
    server.server_close()
    assert lines == ['{"id": 4, "event": "issued"}', '{"id": 7, "event": "scored"}'], \
        "the page was not read line by line"


def test_a_read_token_goes_to_the_hub(monkeypatch):
    monkeypatch.setenv("READ_TOKEN", "r9-Xq")
    seen = {}
    server = _serve(b"", seen)
    pull.get(f"http://127.0.0.1:{server.server_port}/sim/live/a/b/events")
    server.server_close()
    assert seen.get("Authorization") == "Bearer r9-Xq", "the read token was not sent"
