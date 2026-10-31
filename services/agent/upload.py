"""Push recent, stable HDF5 products directly to the hub (stdlib, Python 3.8).

Runs separately from the health/control agent: a transfer cannot delay either
acquisition or commands. A local receipt records acknowledged uploads. Failed
transfers are retried; the hub also de-duplicates after a lost acknowledgement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import StationConfig


def fingerprint(path: Path) -> list:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def send(config, path: Path, before: list, *, opener=urllib.request.urlopen) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
        if fingerprint(path) != before:
            raise ValueError("file changed while hashing")
        f.seek(0)
        url = (config.server_url.rstrip("/") + "/stations/"
               + urllib.parse.quote(config.station, safe="") + "/files/"
               + urllib.parse.quote(path.name, safe=""))
        request = urllib.request.Request(url, data=f, method="PUT", headers={
            "Authorization": "Bearer " + config.token,
            "Content-Type": "application/octet-stream",
            "Content-Length": str(before[0]),
            "X-Content-SHA256": digest.hexdigest(),
        })
        with opener(request, timeout=60) as response:
            result = json.load(response)
    if fingerprint(path) != before or result.get("sha256") != digest.hexdigest():
        raise ValueError("file changed during upload or invalid acknowledgement")
    return result


def run_once(config, source: Path, state_path: Path, *, min_age=60,
             lookback=3600, batch=8, now=None, upload=send) -> dict:
    now = time.time() if now is None else now
    if not config.server_url or not config.token:
        raise ValueError("server_url and AGENT_TOKEN must be configured")
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    identity = [config.server_url.rstrip("/"), config.station]
    state = (json.loads(state_path.read_text(encoding="utf-8"))
             if state_path.exists() else {})
    if state.get("destination") != identity:
        state = {"destination": identity, "sent": {}, "retry": {}}
    candidates = []
    for path in source.rglob("*.h5"):
        try:
            if path.is_symlink() or not path.name.startswith("lfm_ionogram-"):
                continue
            stamp = fingerprint(path)
            age = now - stamp[1] / 1e9
            if not min_age <= age <= lookback or stamp[0] < 8:
                continue
            candidates.append((stamp[1], path, stamp))
        except OSError:
            continue
    sent, errors = 0, []
    # Oldest first: live forecasts cannot score a sounding that arrives after
    # a newer one has already advanced the circuit's ledger.
    attempted = 0
    for _, path, stamp in sorted(candidates):
        key = str(path.relative_to(source))
        if state["sent"].get(key) == stamp or state["retry"].get(key, 0) > now:
            continue
        if attempted >= batch:
            break
        attempted += 1
        try:
            upload(config, path, stamp)
            state["sent"][key] = stamp
            state["retry"].pop(key, None)
            sent += 1
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            state["retry"][key] = now + 60
        # Keep the retry cursor too, so an immutable-name conflict or a bad
        # file cannot monopolise the first batch on every subsequent pass.
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state_path)
    return {"sent": sent, "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--lookback-hours", type=float, default=1)
    parser.add_argument("--min-age", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.min_age < 0 or args.lookback_hours <= 0:
        parser.error("interval/lookback must be positive and min-age nonnegative")
    config = StationConfig.from_env()
    while True:
        try:
            result = run_once(config, args.source, args.state,
                              min_age=args.min_age, lookback=args.lookback_hours * 3600)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            result = {"errors": [str(exc)]}
            print(json.dumps(result), flush=True)
        if args.once:
            return 1 if result["errors"] else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
