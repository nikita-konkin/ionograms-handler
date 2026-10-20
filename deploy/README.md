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
detection files under `ARCHIVE_ROOT`, and each row can be **identified** —
given a code, which is what turns an anonymous detection into a schedulable
`sounder_timings` entry. Choosing which of those to sound, and the mode, is on
the **console**: the *sounding plan* panel on `/ui` sits beside start / stop /
restart, because who to sound and whether to run it are one decision. The
console's list comes from the database rather than the archive, so it does not
pay for the census.

The census opens at most 2000 detection files, newest day first. On an archive
that is not being pruned it will hit that ceiling — DOB held 172,056 files
across three days on 2026-08-15 — and then the page says so, and its answer is
about the recent end of the archive rather than all of it. Ask for
`?max_days=1` to spend the whole budget on today. The startup log names the
ceiling before the warm-up starts, so a container that is still reading can be
told apart from one whose warm-up died; `CENSUS_WARM=0` skips the read
entirely.

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

**Ask the server what it is running; do not infer it.** Every image carries the
commit it was built from, passed in as a build arg and served at `/healthz`:

```console
$ curl -s http://server:8002/healthz
{"ok":true,"version":"0.1.0","build":"3097398","built_at":"2026-08-13T18:41:02Z"}
```

`version` is hand-edited in `services/api/main.py` and has read `0.1.0` through
every deploy so far, so it answers nothing. `build` is the thing to compare
against `git log`. It reads `"source"` from a checkout run by uvicorn and
`"unknown"` from an image built without the arg.

This matters most when a fix appears not to have worked. A green CI run is
**not** proof of a new image: if `DOCKERHUB_USERNAME` or `DOCKERHUB_TOKEN` is
missing the publish step is skipped, the workflow stays green, and the only
trace is an "Images not published" notice in the run summary. Check `build`
first, then the run summary, then pull.

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
naming the secrets and where to add them, then skips the Docker Hub push. The
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

### GitHub Container Registry

Every published build also goes to `ghcr.io/nikita-konkin/<image>`, with the
same tags, pushed with the workflow's own `GITHUB_TOKEN` -- there is no secret
to set, and it publishes even when the Docker Hub secrets are absent.

It exists because the work server stopped being able to pull from Docker Hub
on 2026-09-27: the registry API answered, but every blob download from its CDN
timed out, so each deploy became a local overlay build. To pull from GitHub
instead, check the server can reach it and switch the namespace:

```bash
curl -sI https://ghcr.io/v2/ | head -1     # expect HTTP 401 -- reachable, wants auth
```
```bash
# in deploy/.env
IMAGE_NAMESPACE=ghcr.io/nikita-konkin
```

A package pushed this way starts **private**. Either make each one public once
(GitHub → your profile → Packages → `ionograms-api` → Package settings →
Change visibility), or log the server in with a personal access token that has
`read:packages`:

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u nikita-konkin --password-stdin
```

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

## 2c. Public page

A public, read-only results page -- latest ionograms, a 24-hour strip, MUF/LOF
series and the forecast -- served by the host's own nginx from a folder of
static files. The `publish` service (`services/publish/publisher.py`) rewrites
that folder every `PUBLISH_INTERVAL_S`.

**The api is never exposed by this, and must not be.** Reads are open while
`READ_TOKEN` is unset, and the same read router answers `/principals`, model
artifacts, station health and `/net`; the control routes share the process.
The page is files, so it shows exactly what the publisher writes: allow-listed
circuits, allow-listed fields, no file paths, ids or model internals. It also
keeps showing the last snapshot while the api or the archive mount is down,
with a banner once it is over an hour old.

**Before going live:** rotate `CONTROL_TOKEN` if it has ever been committed,
and confirm `BIND_ADDR` is unset or `127.0.0.1` -- the host is now reachable
from the internet, and nothing but nginx should answer on it.

```bash
# 1. The folder: written by the containers' uid, read by nginx
sudo mkdir -p /srv/ionograms-public
sudo chown 10001:www-data /srv/ionograms-public
sudo chmod 2755 /srv/ionograms-public

# 2. deploy/.env
#    PUBLISH_DIR=/srv/ionograms-public
#    PUBLISH_CIRCUITS=NIC->DOB          # allow-list; empty refuses to start

# 3. One pass by hand, to see it work (or what stops it)
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env \
  --profile publish run --rm publish \
  python -m services.publish.publisher --db /data/ionograms.sqlite3 \
  --archive-root /archive --out /publish/site
ls /srv/ionograms-public                 # index.html status.json data/ img/ ...

# 4. The vhost (edit server_name and root first), then TLS
sudo cp deploy/public-site.nginx.conf /etc/nginx/sites-available/ionograms-public
sudo ln -s /etc/nginx/sites-available/ionograms-public /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d <domain>

# 5. Stay resident
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env \
  --profile publish up -d publish
```

**Check from outside** -- the page loads, and the api does not answer through
it:

```bash
curl -sI https://<domain>/ | grep -iE "^HTTP|content-security|cache-control"
curl -s -o /dev/null -w "%{http_code}\n" https://<domain>/principals   # 404
curl -s -o /dev/null -w "%{http_code}\n" https://<domain>/healthz      # 404
```

**On Apache** the same vhost is `DocumentRoot` on the folder, `Header always
set` for the three security headers, `Header set Cache-Control` per
`<LocationMatch>` (`^/img/` immutable, `\.json$` `no-cache`), `mod_deflate`
for json/js/css, and `<FilesMatch "^\.">` denied. No `ProxyPass`.

**Costs.** Each sounding is drawn once -- the image's name is its time, so an
existing file is its own cache. `.h5` products render in a fraction of a
second; `.lfs` re-runs the spectrogram, seconds each. The folder holds 24
images and one JSON per circuit, a few hundred KB each.

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

That gets health *out* of the station. Getting commands *in* needs one more
thing, and DOB does not have it yet: acquisition there is run by `dombas.sh`,
so `target` in `agent.json` is empty and the agent refuses every start, stop,
restart and `set_config` — correctly, because `systemctl restart chirp.target`
on a script-run station starts a second recorder against the same USRP rather
than restarting the first. The console's buttons stay dead until systemd is
actually the supervisor.

**[migrate-dob-to-systemd.md](migrate-dob-to-systemd.md) is that procedure**:
the checks to do while still on air (including the polkit one, which decides
whether any of it works), the ordered cutover, what regresses, and how to roll
back. Do not set `target` before following it.

### An ionozond station: Start and Stop from `/ui/live`

The Yoshkar-Ola laptop runs the ionozond sounder, not chirpsounder2. systemd
already supervises it as `ionozond-sounder.service`, so it needs no migration:
copy [station-ionozond.json.example](station-ionozond.json.example) to
`~/agent.json`, edit `server_url` and `live_dir`, and `target` is already
right. Two processes then act on the sounder, both as `samsung`:
`chirp-live.service`, which runs a command within one 2 s interval of its
reply saying one waits, and the health loop, which pulls once a minute. For
the health loop, install `chirp-agent.service` with `User=samsung`, the
`samsung` paths, and `ExecStart=/usr/bin/python3 -m services.agent run`.

Both read the sounder's journal -- the live feed for its `STATUS` lines, the
health loop for `sounder_state` and `sounder_overflows` -- so `samsung` must be
in `systemd-journal` or `adm`. `chirp-live.service` adds the group itself;
`id samsung` shows whether the health loop has it.

Both call `systemctl` bare, with no `sudo` (`services/agent/control.py`). A
non-root `systemctl start` asks polkit, whose default for managing units is an
administrator's password, and a service has nobody to type one: the console's
button fails with `Interactive authentication required`. Ubuntu 24.04 ships
polkit 124, which reads JavaScript rules
and can name one unit and four verbs -- unlike DOB's 0.105, where the only
grant is every unit on the host (see `migrate-dob-to-systemd.md` step 0.6).
The fourth, `try-restart`, is how a schedule change from `/ui/live` takes
effect: it restarts the sounder only when it is running, so one stopped on
purpose stays stopped.
Check the version first, since an older polkit ignores this file silently:

```bash
pkaction --version
```

Then create `/etc/polkit-1/rules.d/50-ionozond-sounder.rules`:

```js
// Lets the ionograms-handler agent start, stop and restart the sounder, and
// restart it only if it is running after a schedule change; nothing else: no
// other unit, and no enable, mask or edit of this one.
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.systemd1.manage-units" &&
        action.lookup("unit") == "ionozond-sounder.service" &&
        subject.user == "samsung") {
        var verb = action.lookup("verb");
        if (verb == "start" || verb == "stop" || verb == "restart" ||
            verb == "try-restart") {
            return polkit.Result.YES;
        }
    }
});
```

The folder is readable only by root and polkit, so write the file with `sudo`.
polkit rereads its rules when the file changes; `journalctl -u polkit` says
"Finished loading" or names the error, since a broken rule fails as silently as
an old polkit.

Prove the grant without touching the radio, and prove it is narrow. Run both
from a shell logged in as `samsung`: polkit refuses `--detail` from anyone but
root, so root asks, and `$$` -- expanded by that shell before `sudo` runs --
makes `samsung` the subject. The first two must print `exit 0`; the third must
not, and prints `exit 2` with "Authorization requires authentication":

```bash
sudo pkcheck --action-id org.freedesktop.systemd1.manage-units --process $$ --detail unit ionozond-sounder.service --detail verb stop; echo "exit $?"
```

```bash
sudo pkcheck --action-id org.freedesktop.systemd1.manage-units --process $$ --detail unit ionozond-sounder.service --detail verb try-restart; echo "exit $?"
```

```bash
sudo pkcheck --action-id org.freedesktop.systemd1.manage-units --process $$ --detail unit ssh.service --detail verb stop; echo "exit $?"
```

The `try-restart` probe passes whatever systemd calls the verb, since `pkcheck`
sends the name it is given. Proof that systemd sends `try-restart` is a
schedule change on `/ui/live` acknowledged with its restart accepted; a
refusal there, with the probe passing, means systemd names it otherwise.

A `sudoers` line naming the four commands would be as narrow, but the agent
would have to run `sudo systemctl`, which it does not; that is a code change,
not a configuration. The definitive test is Start on `/ui/live`: the command
row on `/ui` turns acknowledged within seconds, and the sounder's `waiting`
line appears in the event stream.

**Keep `"no_block": true` in `agent.json`.** The sounder finishes the sounding
in progress on `SIGTERM`, which its unit allows 300 s for, and the agent gives
up on a waiting `systemctl` after 120 s. Without the flag a Stop pressed
mid-sounding was acknowledged as a timeout that systemd then completed, and
the live feed stood still for those 120 s. With it the agent runs
`systemctl --no-block`, acknowledges once systemd accepts the job, and the
sounder's next `STATUS` line and the health pass say how it went. A polkit
refusal is still reported, since systemd refuses before it queues anything.

### An ionozond station: the health loop, under the live feed's station

The schedule panel on `/ui/live` reads the schedule in force from the health
reports, and the server checks an edit against the latest one. The live feed
sends none, so an ionozond laptop needs the health loop as well, and it must
report under the **same station name as the live feed** -- the one Start, Stop
and schedule changes are queued for. Using the live feed's own
`~/agent.json` guarantees that.

First look for an agent already on the laptop. A chirpsounder2 agent reporting
as `Yoshkar-Ola`, with capital letters, is a different station to the server:

```bash
systemctl status chirp-agent.service --no-pager
```

If none exists, install the repo's unit edited for `samsung` under its own
name. If one does, install this copy as `chirp-agent-ionozond.service`
instead, and leave the other alone. It runs from the checkout
`chirp-live.service` uses, which `WD` reads from that unit:

```bash
UNIT=chirp-agent.service
```

```bash
WD=$(systemctl show chirp-live.service -p WorkingDirectory | cut -d= -f2)
```

```bash
sed -e "s#^User=.*#User=samsung#" -e "s#^WorkingDirectory=.*#WorkingDirectory=$WD#" -e "s#^Environment=AGENT_CONFIG=.*#Environment=AGENT_CONFIG=/home/samsung/agent.json#" -e "s#^ExecStart=.*#ExecStart=/usr/bin/python3 -m services.agent run\nSupplementaryGroups=systemd-journal#" "$WD/services/agent/systemd/chirp-agent.service" | sudo tee /etc/systemd/system/$UNIT
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now $UNIT
```

`/usr/bin/python3` is right here, unlike on DOB: Ubuntu 24.04's is 3.12, and
the token comes from the same `/etc/default/chirp-agent` the live feed reads.
To read and change the schedule the agent also reads `/etc/default/ionozond`
(0644) for `IONOZOND_CONFIG`, and edits the console's
`~/.config/dsChirp/schedule.ini` and `config.ini` and the sounder's
`chirp_config.py`, all of which belong to `samsung`. It needs no permission
beyond the polkit rule above.

Check it within a minute: the station's row on `/ui` shows a fresh report for
`yoshkar-ola`, and that report carries `sounder_config`:

```bash
journalctl -u $UNIT -n 20 --no-pager
```

### An ionozond station: retiring the Qt console

Phase 4 (issue #8, `docs/2026-10-17-ionozond-retiring-the-console.md`) moves
what was left of the console onto `/ui/live`: Regenerate after a hand edit of
the ini files, the `.h5` archive switch and its local retention, the
storage-policy keys of `/etc/default/ionozond`, and Run now for the NAS copy
and the pruner. The steps below are run on the laptop, by hand, in this order.
None of them restarts the sounder, since `chirp_config.py` does not change.

**0. Before starting**, record two facts: whether the prune timer is enabled,
and whether a chirpsounder2 agent posting as `Yoshkar-Ola` runs on this
laptop. The health loop must already report `sounder_config` for
`yoshkar-ola` with no drift and no problems.

```bash
systemctl is-enabled ionozond-prune.timer ionozond-nas.timer
```

```bash
systemctl list-units 'chirp-agent*' --no-pager
```

**1. Back up** the console's settings, the sounder's file and the root file
into one dated tarball:

```bash
CFG=$(sed -n 's/^IONOZOND_CONFIG=//p' /etc/default/ionozond)
```

```bash
tar czf ~/ionozond-before-phase4-$(date +%F).tar.gz ~/.config/dsChirp "$CFG" /etc/default/ionozond
```

**2. Deploy** the agent: `git pull` in the checkout `chirp-live.service` runs
from, then restart the health loop (`chirp-agent.service`, or
`chirp-agent-ionozond.service` where it was installed under that name) and
`chirp-live.service`.

**3. Check the three folders agree.** `data_dir` in `config.ini`, `live_dir`
in `~/agent.json` and `IONOZOND_ARCHIVE` must name the same folder; a
trailing slash or a `~` does not matter. Regenerate refuses to run while they
disagree, because the sounder would write where the prune and NAS units never
look.

```bash
grep -E '^IONOZOND_(ARCHIVE|CONFIG)=' /etc/default/ionozond; grep -E '^(data_dir|nas_dest|nas_keep_h5_days)=' ~/.config/dsChirp/config.ini; grep live_dir ~/agent.json
```

**4. Align the NAS fallbacks.** `nas_sync.py` takes the destination and the
retention from `config.ini`, through `chirp_config.py`, and falls back to
`NAS_DEST` and `NAS_KEEP_H5_DAYS` only when those are missing. Set the
fallbacks to what `config.ini` says, so that a `config.ini` restored from an
old backup cannot quietly stop the copy or change the retention. With the
values step 3 printed (on 2026-10-17, `/mnt/ionozond_5tb/ionozond_data` and
21):

```bash
sudo sed -i -e 's#^NAS_DEST=.*#NAS_DEST=/mnt/ionozond_5tb/ionozond_data#' -e 's#^NAS_KEEP_H5_DAYS=.*#NAS_KEEP_H5_DAYS=21#' /etc/default/ionozond
```

```bash
grep -E '^NAS_(DEST|KEEP_H5_DAYS)=' /etc/default/ionozond
```

If the second command prints nothing for a key, the file never had it: add
the line with `sudoedit /etc/default/ionozond`.

**5. Let the prune and NAS units read `web.env`.** The web writes
`PRUNE_KEEP_DAYS`, `PRUNE_FREE_GB` and `NAS_BWLIMIT` into
`~/.config/ionozond/web.env`, which `samsung` owns, and never touches the root
file. A drop-in loads it after `/etc/default/ionozond`, so its keys win. An
environment file can set `PATH` or `LD_PRELOAD`, so it may only be loaded by a
unit that runs as `samsung` with no step run as root. Check both units as
installed first. This must print `User=samsung` twice and nothing else; any
`Exec…=+…`, `Exec…=!…` or `PermissionsStartOnly=` line means stop here and
install no drop-in:

```bash
systemctl cat ionozond-prune.service ionozond-nas.service | grep -E '^(User|PermissionsStartOnly)=|^Exec[A-Za-z]*=[+!]'
```

Then install the drop-in in both, never in `ionozond-sounder.service`, whose
`ExecStartPre=+…ethtool` runs as root:

```bash
for u in ionozond-prune ionozond-nas; do sudo mkdir -p /etc/systemd/system/$u.service.d && printf '%s\n' '# web.env is writable by samsung. Load it only in a unit that runs as samsung' '# with no "+" or "!" step: never in ionozond-sounder.service.' '[Service]' 'EnvironmentFile=-/home/samsung/.config/ionozond/web.env' | sudo tee /etc/systemd/system/$u.service.d/web.conf; done
```

```bash
sudo systemctl daemon-reload
```

```bash
systemctl show ionozond-prune.service ionozond-nas.service -p EnvironmentFiles
```

The last command shows both files for each unit, `web.env` with `(ignore_errors=yes)`:
the leading `-` makes a missing `web.env` harmless, so deleting it undoes every
web edit at the next run.

**6. Extend the polkit rule** so Run now can start the two job units, and
nothing more. Replace `/etc/polkit-1/rules.d/50-ionozond-sounder.rules`
(written with `sudo`, as before) with:

```js
// Lets the ionograms-handler agent start, stop and restart the sounder, and
// restart it only if it is running after a schedule change; and start one run
// of the NAS copy or the pruner. Nothing else: no other unit or verb, and no
// enable, mask or edit of these.
polkit.addRule(function(action, subject) {
    if (action.id != "org.freedesktop.systemd1.manage-units" ||
        subject.user != "samsung") {
        return polkit.Result.NOT_HANDLED;
    }
    var unit = action.lookup("unit"), verb = action.lookup("verb");
    if (unit == "ionozond-sounder.service" &&
        (verb == "start" || verb == "stop" || verb == "restart" ||
         verb == "try-restart")) {
        return polkit.Result.YES;
    }
    if ((unit == "ionozond-nas.service" || unit == "ionozond-prune.service") &&
        verb == "start") {
        return polkit.Result.YES;
    }
    return polkit.Result.NOT_HANDLED;
});
```

Prove it from a shell logged in as `samsung`, as above. The first must print
`exit 0`; the second must print `exit 2`, since the rule grants `start` only:

```bash
sudo pkcheck --action-id org.freedesktop.systemd1.manage-units --process $$ --detail unit ionozond-nas.service --detail verb start; echo "exit $?"
```

```bash
sudo pkcheck --action-id org.freedesktop.systemd1.manage-units --process $$ --detail unit ionozond-nas.service --detail verb stop; echo "exit $?"
```

Repeat the four sounder probes above: the rule was rewritten, not appended to.

**7. Dry-run the pruner, then enable its timer.** Without `--apply` it only
reports what it would delete. Read the list before enabling the timer, which
runs it with `--apply`:

```bash
cd ~/projects/ionozond && set -a && . /etc/default/ionozond && set +a && python3 tools/prune_lfs.py "$IONOZOND_ARCHIVE" --keep-days "$PRUNE_KEEP_DAYS" --free-gb "$PRUNE_FREE_GB"
```

```bash
sudo systemctl enable --now ionozond-prune.timer
```

It is a backstop: with `keep_lfs` off the sounder deletes each capture once
its `.h5` is written, and the pruner finds nothing to do.

**8. Report both jobs.** In `~/agent.json`, set
`"job_units": ["ionozond-nas.service", "ionozond-prune.service"]`, as in
[station-ionozond.json.example](station-ionozond.json.example), and restart
the health loop. `/ui/live` then shows each job's last run beside its Run now
button, and `/ui` turns red on a failed run or a timer that stopped firing.

**9. Retire the console's launcher**, so it is no longer started from habit.
Leave the binary and `~/.cache/ionozond-build` in place for a rollback:

```bash
rm -f ~/.local/share/applications/ionozond.desktop "$(xdg-user-dir DESKTOP)/ionozond.desktop"
```

**10. Verify** from `/ui/live`: a schedule change; a Regenerate after a
deliberate hand edit of an unused key in `config.ini` (such as `whiten_n`),
whose diff names that key alone; a change of `PRUNE_FREE_GB`, which appears in
`~/.config/ionozond/web.env`; and a Run now of the NAS copy that ends green.

**Rolling back.** Delete `~/.config/ionozond/web.env` to undo every web edit
of the site settings; delete the two drop-ins and run `daemon-reload` to stop
reading it. The previous agent answers `regenerate_ionozond` and
`set_ionozond_site` with "unknown command". To bring the console back, run
`tools/host-setup/19-install-desktop.sh` in the ionozond checkout. Its first
dialog OK writes every key the dialog holds into the ini files, so it adds any
the files lack: on 2026-10-17 it added `live_ionogram=true` and
`save_raw_voltage=false`. After such an OK, check the health report before
relying on it: drift false and an unchanged `sounder_hash` mean the sounder
reads what it read before.

### An ionozond station: the live view's history

The SNR and PDP strips on `/ui/live` show three days and scroll back as far
as the server holds summaries (issue #12). The server keeps every summary it
builds from live columns, and the live feed fills in the rest from the
products the station already has: a worker beside the feed reads each
`lfm_ionogram-*.h5` under `live_dir` and `backfill_dirs`, summarises it as the
server summarises a live sounding, and posts a few after each live batch.

Once an hour (`backfill_every_s`) it asks the server how many summaries it
holds for each transmitter and UTC day, and resends every day that holds
fewer than the station has products, newest day first. A live summary always
wins over one read from a file. A product that cannot be read is logged once
and skipped until the feed restarts.

`backfill_dirs` lists folders beyond `live_dir` that hold older products, as
paths on the laptop. The laptop keeps `nas_keep_h5_days` of products itself
and the NAS, mounted at `/mnt/ionozond_5tb` by `ionozond_5tb.mount`, holds
the rest, so name the NAS copy to reach the whole history:

```json
"backfill_dirs": ["/mnt/ionozond_5tb/ionozond_data"]
```

A sounding found in both is read once, from `live_dir`. A NAS that is not
mounted holds nothing to read and costs nothing. Restart
`chirp-live.service` after editing `~/agent.json`, then watch the history
arrive with `journalctl -u chirp-live -f` and on `/ui/live`.

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
| `curl localhost:$PORT/healthz` | `{"ok":true,...}`, and `build` naming the commit that is running |
| `/ui` after one push interval | `SIM` appears, `HEALTHY`, most metrics grey |
| Queue `restart`, then watch the sim's log | `FAKE systemctl restart chirp.target` |
| `/ui` again | the command shows `acked` |
| Stop the sim container, wait 3 min | `SIM` turns `STALE` |
| `/ui/series?method=kmeans` | a MUF curve, hollow markers for lower bounds |
| `/ui/sounding/1` | an ionogram, auto-gated, with the scaled trace drawn over it |
| `/ui/sounding/1`, untick **raster** | the circles alone on an empty field — the overlay is real data, not part of the image |
| `curl -I localhost:$PORT/static/plotly.min.js` | `200`, ~1.0 MB — the plot library ships in the image |
| `curl localhost:$PORT/soundings/1/sao.xml` | `<SAORecordList>` with three `<SAORecord>`, one per estimator |
| `curl localhost:$PORT/net` | `"state":"online"` if this host can reach the solar index servers |

**`/net` is the answer to "does the server have internet?"** — asked of the
three hosts that matter rather than of the internet at large, since a proxy
that blocks `sidc.be` and a working mirror both look identical to a ping. It
reports the last background pass and never probes on request, so it costs
about a millisecond and is safe to poll. The same reading is the **upstream**
panel at the top of `/ui`, with each file's cache age beside its host.

`state` is `unknown` for the first second after startup, and again if the
checker thread ever stops — a dead checker must not leave a green light. Three
knobs, all optional: `NET_CHECK=0` to switch it off on a deliberately isolated
host, `NET_CHECK_INTERVAL_S` (default 600) and `NET_TIMEOUT_S` (default 4).

**The sounding page draws the trace itself.** The raster is served bare
(`/ionogram/{id}.png?bare=true` — no axes, no colourbar, no overlay) and placed
under scatter traces in data coordinates, so only a few hundred scaled points
cross the wire instead of 1.9 million raster cells. If the circles look offset
from the echo by a hair, that is the extent, not the detector: the image covers
half a cell more than the axis at each end.

The panel beside the plot needs **PyIRI**, which the image now installs. It is
not a light dependency — netCDF4, cftime, fortranformat and opt_einsum, about
30 MB — and it needs a solar driver, so on a host with no route out and a cold
`MUF_INDEX_CACHE` the panel says so instead of showing numbers. `SAO_MODEL=0`
turns the whole IRI column off on a host that should make no outbound requests
at all.

The indices themselves cache to `MUF_INDEX_CACHE`, set to `/data/indices` in
both compose files. It has to be on the volume: `$HOME` is an image layer, so
without it every `docker compose pull` discards ~5 MB of index files and the
next model run re-downloads them — on a host with no route out, that is the
only copy gone.

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
  not fine across a network you do not control. **Accounts do not change
  this.** Per-user tokens make access revocable for one person and put a real
  name in the audit columns; they are still bearer tokens in cleartext, and
  anyone who can watch the wire can replay one. The tunnel above is still the
  answer, and it is the reason accounts issue random tokens rather than taking
  passwords: a leaked rig token is a rig token, while a leaked password is very
  often the one its owner uses elsewhere.
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
