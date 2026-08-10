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

Then <http://127.0.0.1:8000/ui>, or whatever `PORT` you set in `deploy/.env` —
the shipped example sets `8002`, because the work server's 8000 is taken.
`/docs` is the generated OpenAPI page.

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

### Sounding mode, and where a schedule comes from

`/ui/sources` lists the transmitters the station has actually heard, from the
detection files under `ARCHIVE_ROOT`, and offers each as a `sounder_timings`
entry. The panel below it switches the station between the two sounding modes.

That pairing is the point. **search** (serendipitous) mode records whatever
sweeps past and infers who was transmitting; **scheduled** mode downconverts a
fixed list at times it is told, giving named products with an absolute range
zero. The output of the first is the input to the second, and `control.py`
enforces it: leaving search mode without a schedule is refused, because a
scheduled station with an empty `sounder_timings` records nothing while every
process reports healthy.

**This widens what the web can do to a radio**, so it is narrow on purpose.
Of the five settings `control.py` can edit, only `mode` and `sounder_timings`
are routed — `output_dir` decides where a week of data lands and a typo is
unrecoverable from here. The allow-list is checked at the server as well as at
the agent, `mode` is validated against `control.MODES`, and the change rewrites
the ini and takes effect on the **next restart**, so it is two deliberate
actions rather than one.

The seconds shown are **as received**, not as transmitted — a slot is the
transmit second plus travel time plus this receiver's epoch offset. For
scheduling that is the number you want; it is not a transmit time and not a
range.

## 2b. Deploying from Docker Hub (the work server)

`docker-compose.yml` **builds** from a checkout. That is right on a development
machine and wrong on a server: building needs the source, a toolchain and ten
minutes, and what comes out can differ from what CI tested.
`docker-compose.hub.yml` pulls instead.

```bash
cp deploy/.env.example deploy/.env      # set CONTROL_TOKEN and IMAGE_NAMESPACE
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env up -d
```

### Fitting alongside what the host already runs

The work server already runs other stacks, and two of them collide with the
defaults:

| already there | collision |
|---|---|
| `tec-backend` on **8000** | the api's old default. `PORT` is now **8002** |
| `watchtower` | ours would be a **second** one on the same Docker socket |

`PORT` is the host side only; the container still listens on 8000 internally,
so nothing else changes.

**Do not start a second watchtower.** Two of them poll, restart and prune the
same containers, which is wasted work at best and a race over one image at
worst. Ours is behind a compose *profile* so it stays out of the way:

```bash
# host already has watchtower -- the normal case here
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env up -d

# only on a host that has none
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env \
    --profile watchtower up -d
```

The existing watchtower needs nothing from us except the labels already on
`api` and `watch` -- **provided it is label-scoped**. Check:

```bash
sudo docker inspect watchtower --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i label
```

`WATCHTOWER_LABEL_ENABLE=true` means it updates only labelled containers, and
ours are labelled, so they are picked up automatically. If that variable is
absent it updates **everything on the host**, including these -- which still
works, but means the opt-in the labels were meant to provide is not actually in
force for anything on that machine.

Three services, and the third is the one to think about:

| service | what it does |
|---|---|
| `api` | the console and read API, as before |
| `watch` | `services.api.watch` on a timer — ingests whatever is new, every `INGEST_INTERVAL_S` |
| `watchtower` | polls Docker Hub and restarts a container when its image digest changes |

`watch` is the same image as `api` because it is the same code, but a separate
container so a long ingest cannot block a request and a crash in one does not
take the other down. Its healthcheck is **disabled** on purpose: the image's
check curls the api's `/healthz`, and this container runs no server, so
inheriting it would leave a permanently-unhealthy container and teach everyone
to ignore the column. Its liveness is the line it logs each pass:

```
2026-08-09T20:46:11Z  1722 on disk, 0 new
```

### The pipeline

`.github/workflows/ci.yml` runs the suite on every push and pull request, and
publishes **only if the suite is green** — `publish` declares `needs: test`,
so a red build cannot produce an image the server would then pull. Pull
requests are tested but never published, so a fork cannot push into your
registry.

Both images are built for `linux/amd64` and `linux/arm64`, so `docker pull`
gives the work server and a development Mac the right one without either
asking. That is most of why this belongs in CI: cross-building scipy and
opencv locally is slow enough that people stop doing it.

Tags:

| tag | when | use |
|---|---|---|
| `sha-<short>` | every build | **what a rollback names.** `latest` does not tell you what is running |
| `latest` | default branch only | what `watchtower` follows |
| `v1.2.3`, `1.2` | on a `v*` git tag | releases |

Two repository secrets are needed, under Settings → Secrets and variables →
Actions:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` — a Docker Hub **access token**, not the account password
  (Docker Hub → Account settings → Personal access tokens), scoped
  Read & Write.

**The namespace defaults to `DOCKERHUB_USERNAME`**, which is right unless you
are pushing into a Docker Hub *organisation*. Only then set
`DOCKERHUB_NAMESPACE`, and it is read from either the Variables or the Secrets
tab -- they sit next to each other on the same settings page, and putting it in
the wrong one used to make it vanish silently.

That is worth spelling out because it is how this first failed. The namespace
was resolved in a **workflow-level** `env`, where GitHub exposes only the
`github`, `vars` and `inputs` contexts -- **not `secrets`**. A namespace added
under Secrets therefore read as empty, fell back to the GitHub owner
(`nikita-konkin`, which is not the Docker Hub account `nikitaikonkin`), and the
push failed with:

```
push access denied, repository does not exist or may require authorization:
server message: insufficient_scope: authorization failed
```

That message names neither the account nor the namespace, so the mismatch is
invisible from the log. The job now resolves the namespace at *job* level,
where `secrets` is available, and prints where it came from:

```
Notice: Pushing as: namespace from DOCKERHUB_USERNAME;
        images nikitaikonkin/ionograms-{api,station}
```

and warns when the namespace differs from the authenticated account, which is
legitimate only for an organisation and otherwise is exactly this bug.

If you see `insufficient_scope` with the namespace and account matching, the
remaining cause is token scope: the Docker Hub access token must be **Read &
Write**. A read-only token authenticates and then cannot push.

**Until those secrets exist the pipeline still runs and still passes.** The
publish job checks for them first and, finding none, writes a job-summary note
naming the secrets and where to add them, then skips the registry steps. The
alternative was `login-action` failing with `Username and password required` --
accurate, but it names neither the secret nor its location, and it turns a
pipeline red for a reason that has nothing to do with the commit. A fork sees
the same thing, which is the other reason it is a skip rather than a failure.

If you see this in the log:

```
Node 20 is being deprecated. This workflow is running with Node 24 by default.
```

nothing is wrong and nothing needs changing. It is a notice about actions that
still bundle the older runtime; the workflow is already on Node 24. Do **not**
set `ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION` -- that pins you *back* to the
deprecated runtime.

### Rolling back

`latest` moving is what makes an automatic update convenient and what makes a
bad one land unattended. To pin a server, or to undo:

```bash
# in deploy/.env
IMAGE_TAG=sha-1a2b3c4
```
```bash
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env up -d
```

A pinned tag also stops watchtower moving it, since the digest behind a `sha-`
tag never changes.

### What is deliberately *not* auto-updated

`watchtower` runs with `WATCHTOWER_LABEL_ENABLE`, so it updates **only**
containers carrying `com.centurylinklabs.watchtower.enable=true`. Adding a
service to the compose file does not silently enrol it.

Nothing that touches a radio is labelled. The `api` is a read surface plus a
command queue; an unattended restart costs a few seconds of uptime and no
data. **The station agent is not on this server at all** — it runs on the
sounding laptop, and it should be updated when somebody is watching, because
a restart there interrupts acquisition.

If you later containerise the agent on the laptop, leave it unlabelled and
update it by hand. An auto-updating process that can stop a radio is a
different risk from an auto-updating web page.

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

**The laptop needs no `deploy/.env`.** That file configures Compose, and the
laptop runs no containers -- it runs the agent under systemd, next to the
radio. It is also per-host by nature (`PORT` is about this server's port
collisions, `ARCHIVE_HOST_PATH` is a path on this server), so copying it across
would carry nothing true. The laptop gets two things instead: `agent.json`, and
the token in the unit's environment.

Exactly one value spans both machines -- `CONTROL_TOKEN` here must equal
`AGENT_TOKEN` there. Everything else is independent.

Then on the laptop:

```bash
scp deploy/station-dob.json.example ionouser@<laptop>:~/agent.json
# edit server_url → http://192.168.1.50:8002   (PORT from deploy/.env)
```

The token does **not** go in that file -- leave `"token": ""`. `agent.json` is
copied, edited and backed up; a secret in it travels with every copy. It goes
in the unit's environment file, root-owned and unreadable by anyone else:

```bash
printf 'AGENT_TOKEN=%s\n' '<the CONTROL_TOKEN>' | sudo tee /etc/default/chirp-agent
sudo chmod 600 /etc/default/chirp-agent
```

`chirp-agent.service` reads it via `EnvironmentFile=`, and `AGENT_TOKEN`
overrides the file's `token`. For the two commands below, which run in your own
shell rather than under the unit, export it by hand:

```bash
cd ~/chirpsounder2 && source .venv38/bin/activate
export PYTHONPATH=~/ionograms-handler
export AGENT_CONFIG=~/agent.json AGENT_TOKEN='<the CONTROL_TOKEN>'
python -m services.agent health          # local check, no server needed
python -m services.agent run --passes 1  # one push
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
ssh -N -L 8002:127.0.0.1:8002 you@work-pc
# agent server_url stays http://127.0.0.1:8002, and BIND_ADDR stays 127.0.0.1
```

If only the reverse is possible — you can reach the laptop but it cannot reach
you — push the tunnel from the work PC:

```bash
# on the work PC
ssh -N -R 8002:127.0.0.1:8002 ionouser@laptop
```

Either way the API stays bound to localhost on both ends and nothing is
exposed. The agent still initiates every connection, so the station is still
not a listening service.

## 4. Verify

Ports below are written as `$PORT`, from `deploy/.env`. Unset it and the two
rigs differ on purpose: the test rig takes 8000, the work server takes 8002,
where 8000 belongs to `tec-backend`.

| check | expected |
|---|---|
| `curl localhost:$PORT/healthz` | `{"ok":true,...}` |
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
