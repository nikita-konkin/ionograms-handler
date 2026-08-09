# Temporary test rig

`api` + web on one host, plus a simulated station, in Docker Compose. Brings
`architecture.md` M3 and M4 forward as a throwaway deployment so the agent loop
can be exercised before either milestone is built properly.

**Not production.** SQLite, plain HTTP, no TLS, no backups, no migrations.

---

## 1. Start it

```bash
cp deploy/.env.example deploy/.env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # → CONTROL_TOKEN
$EDITOR deploy/.env

docker compose -f deploy/docker-compose.yml --env-file deploy/.env up --build
```

Then <http://127.0.0.1:8000/ui>. `/docs` is the generated OpenAPI page.

`CONTROL_TOKEN` is the only value you have to set. The sim reads it from the
same `deploy/.env`, as `AGENT_TOKEN`, so there is nothing to keep in sync by
hand. It deliberately does **not** live in `deploy/station-sim.json`: that file
is committed, and a secret pasted into it is one `git add` from being
published. `AGENT_TOKEN` overrides the file's `token` for the real agent too,
which is what `station-dob.json.example` should be deployed with.

### If the build cannot reach the network

Two symptoms, one cause:

```
failed to resolve source metadata for docker.io/library/python:3.12-slim
  ... lookup registry-1.docker.io: no such host

ERROR: Could not install packages due to an OSError:
  HTTPSConnectionPool(host='files.pythonhosted.org', port=443)
  ... [Errno -3] Temporary failure in name resolution
```

**Containers have no DNS.** Not a Dockerfile problem — the same failure hits
the registry and PyPI, at build time and at run time. Check with:

```bash
docker run --rm python:3.12-slim \
    python -c "import socket; print(socket.gethostbyname('pypi.org'))"
```

On the development machine this was **intermittent** — it resolved once and
then failed again minutes later, which is worth knowing because a single
successful check proves nothing. Run it a few times.

Fixes, in the order worth trying: Docker Desktop → Settings → Resources →
Proxies (a corporate proxy usually has to be entered explicitly, and BuildKit
does not inherit the host's); Settings → Docker Engine, add
`"dns": ["8.8.8.8"]`; on WSL2, a stale `/etc/resolv.conf` inside the VM does
it too, so `wsl --shutdown` then restart Docker Desktop.

### What the first real build found

Both images have now been built and run end to end, on macOS 26.2 / arm64 with
OrbStack. The whole of section 4 passes, including the command round-trip and
the ionogram renderer reading both formats off the mounted archive. Three
defects only a genuine build could surface, all fixed here:

- **`Dockerfile.station` pinned `h5py==3.11.0`**, which publishes no cp312
  aarch64 wheel. On an arm64 host pip fell back to building it from source and
  the image died on a missing `pkg-config` and HDF5 headers. Now ranged like
  `requirements-api.txt` always was, with `--only-binary=:all:` so a future
  resolution needing a compiler says so at the pin.
- **`Dockerfile.station` never chowned `/app`.** `COPY` preserves the source
  mode, so a checkout whose files are `0600` — a synced or restrictively
  umasked working copy — landed root-owned and unreadable to the non-root
  `station` user. The container restart-looped on `PermissionError:
  /app/services/__init__.py`. `Dockerfile.api` had it right already.
- **`requirements-api.txt` omitted `h5py` entirely**, though `pyproject.toml`
  calls it core. Nothing failed: `muf.io_chirp` imports it lazily, so the api
  came up green, passed its health check, and then skipped every `.h5`
  sounding at ingest. An archive that loads 514 soundings on the host loaded
  133 in the container. **A green health check does not mean the image can
  read your data.**

Two of the three were silent. Compare a container ingest against a host ingest
before trusting a new image.

## 2. Load some soundings

The console is empty until an archive is ingested. From the host:

```bash
python -m services.api.ingest F:/MyData/ND/lfs/ionozond_data2/2026-08-05 \
    --db data/ionograms.sqlite3 \
    --archive-root F:/MyData/ND/lfs \
    --methods algo,kmeans,contour
```

Measured on the development machine: **318 of 319 v2 soundings**, one skipped
as unreadable. `.lfs` works the same way and is slower, because it re-derives
the spectrogram rather than reading a stored product.

`--archive-root` matters. `sounding.path` is stored *relative* to it, so the
same database resolves on the host and inside the container, which mount the
archive at different paths. Ingest with the wrong root and the ionogram
endpoint returns `410 Gone` naming the path it tried.

### Keeping up with a growing archive

`ingest` re-derives everything you hand it, which is right for a deliberate
reload and wrong on a timer. `services.api.watch` enumerates the archive, asks
the database what it already holds, and ingests only the difference — so the
usual pass costs a directory scan and one query.

Inside the container, where the database and the archive both live:

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec api \
    python -m services.api.watch /archive \
    --db /data/ionograms.sqlite3 --archive-root /archive \
    --methods algo,kmeans,contour --jobs 0
```

```
2026-08-09T12:34:39Z  623 on disk, 109 new, loaded 109
2026-08-09T12:36:02Z  623 on disk, 0 new
```

Run it from cron for one pass, or add `--interval 900` to leave it resident.
It is idempotent on `(file, method)` — the same key `ingest` upserts on — so a
pass that dies halfway costs nothing but the work it had not reached, and
widening `--methods` brings the older soundings back into scope by itself.

Three flags earn their keep on a live station:

- **`--min-age`** (default 60 s) skips files modified more recently than that.
  A sounding still being written, or still arriving over a sync, reads as a
  short sweep — and a short sweep does not fail. It ingests with
  `sweep_complete` false and stays wrong until someone notices.
- **`--batch N`** caps a pass, so the first run over a large archive does not
  hold the database for hours.
- **`--dry-run`** reports what it would do and changes nothing.

A `SKIPPED` count that never falls is worth chasing: unreadable files are not
recorded, so they are retried every pass, which is right for a half-synced
file and pointless for a corrupt one.

## 3. Connect the real acquisition laptop

This is the part that needs a decision, because it is the only step that makes
the rig reachable from outside the work PC.

Health is **pushed** and commands are **pulled** (arch §5.4), so the laptop
only ever makes outbound connections and never needs an inbound port. The
server does.

### Case A — laptop and work PC on the same LAN

```ini
# deploy/.env
BIND_ADDR=192.168.1.50        # the work PC's LAN address, NOT 0.0.0.0
```

`0.0.0.0` would also publish it on every other interface, including any VPN
adapter. Name the interface you mean.

Then on the laptop:

```bash
scp deploy/station-dob.json.example ionouser@<laptop>:~/agent.json
# edit server_url → http://192.168.1.50:8000 and token → your CONTROL_TOKEN

cd ~/chirpsounder2 && source .venv38/bin/activate
export PYTHONPATH=~/ionograms-handler
AGENT_CONFIG=~/agent.json python -m services.agent health      # local check
AGENT_CONFIG=~/agent.json python -m services.agent run --passes 1   # one push
```

The agent is Python 3.8-clean, so the station's `.venv38` runs it unmodified.

Leave it running under the unit once the single pass works:

```bash
sudo systemctl enable --now chirp-agent.service
```

### Case B — different networks

**Do not port-forward this over the internet.** It is plain HTTP and one
bearer token away from stopping a radio.

Use an SSH tunnel instead. If the laptop can reach the work PC's SSH:

```bash
# on the laptop
ssh -N -L 8000:127.0.0.1:8000 you@work-pc
# agent server_url stays http://127.0.0.1:8000, and BIND_ADDR stays 127.0.0.1
```

If only the reverse is possible — you can reach the laptop but it cannot reach
you — push the tunnel from the work PC:

```bash
# on the work PC
ssh -N -R 8000:127.0.0.1:8000 ionouser@laptop
```

Either way the API stays bound to localhost on both ends and nothing is
exposed. The agent still initiates every connection, so the station is still
not a listening service.

## 4. Verify

| check | expected |
|---|---|
| `curl localhost:8000/healthz` | `{"ok":true,...}` |
| `/ui` after one push interval | `SIM` appears, `HEALTHY`, most metrics grey |
| Queue `restart`, then watch the sim's log | `FAKE systemctl restart chirp.target` |
| `/ui` again | the command shows `acked` |
| Stop the sim container, wait 3 min | `SIM` turns `STALE` |
| `/ui/series?method=kmeans` | a MUF curve, hollow markers for lower bounds |
| `/ui/sounding/1` | an ionogram, auto-gated |

Most metrics being **grey** in the sim is correct, not a fault: a container has
no systemd and no journald, so unit states and logs are genuinely unmeasurable.
Grey is *unknown*, and unknown never makes a station unhealthy.

`AGENT_FAKE_SYSTEMCTL=1` in the sim logs each verb instead of running it,
which is the only way to see the command path complete in a container. It
prints on every call so a fake success cannot be mistaken for a real one. Never
set it on the real station.

## 5. Tear down

```bash
docker compose -f deploy/docker-compose.yml down          # keep the database
docker compose -f deploy/docker-compose.yml down -v       # delete it too
```

---

## What this rig is not

- **No TLS.** Tokens cross the wire in clear. Fine on localhost or a tunnel;
  not fine across a network you do not control.
- **No migrations.** `schema.sql` is applied with `CREATE TABLE IF NOT EXISTS`.
  A schema change means deleting the volume.
- **No parameter editing.** `control.py` implements validated `.ini` edits and
  the web surface deliberately does not route them — `control_routes.QUEUEABLE`
  is `start`, `stop`, `restart` only. A parameter change rewrites the station's
  config and needs a restart to take effect; proving start/stop first is worth
  more than the extra surface.
- **One process, no queue.** The renderer runs inside the API process, so a
  slow render blocks a worker. At ~0.5 s per ionogram on one viewer that is
  fine and anything else would be premature.

## Security posture in one paragraph

`CONTROL_TOKEN` unset **disables** control rather than opening it — a missing
secret must never be the same as a granted one. `READ_TOKEN` unset leaves reads
open, which is right for `127.0.0.1` and wrong the moment `BIND_ADDR` leaves
it; the API says so in its startup banner. The two scopes are separate because
public read of soundings must not share a scope with anything that can stop an
acquisition (§4.3). The web UI holds the control token in `sessionStorage` for
the tab only, never baked into a served page, and `stop` asks for confirmation
naming the consequence.
