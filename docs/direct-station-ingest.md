# Direct station files for live simulation

The laptop pushes completed `lfm_ionogram-*.h5` files to the hub over HTTP.
The hub keeps them on local disk and ingests them independently of archive
scanning. The existing live simulator then finds the new database rows on
its normal pass. NAS copying remains a separate archive operation.

## Enable on the hub

Deploy an API image containing this change and set `DIRECT_INGEST=1` in
`deploy/.env`. Recreate the API container with the updated compose file:

```sh
docker compose --env-file deploy/.env -f deploy/docker-compose.hub.yml up -d api
```

Files, receipts, and the retry queue live in `/data/station-files`, on the
existing persistent `api-data` volume. For a non-container deployment this
defaults to `station-files` beside `API_DB`; `DIRECT_ROOT` overrides it.
It must be **local writable storage**, visible at the same absolute path to
any other workers that read soundings. No NAS mount is used by this queue.

The worker starts every five seconds, uses the contour extractor, and commits
each file separately. It uses a separate connection and a process with a
120-second limit for each extraction. Errors remain queued with backoff up
to one hour; other files can proceed. Existing archive scans can fill other
extraction methods later without moving a sounding's read path off local disk.

## Enable on the station

Install the same code on the laptop. The uploader is stdlib-only and Python
3.8 compatible. It runs separately from the radio and the health/control
agent. It uses `AGENT_CONFIG` and `AGENT_TOKEN` exactly as the existing agent
does; the config must name the correct station and server URL.

Adapt `deploy/ionograms-direct-upload.service` to the **installed agent's**
user, Python executable, checkout, config and secret EnvironmentFile. The
provided source path is `/home/samsung/ionograms`, for NIC2. Then:

```sh
sudo install -m 644 deploy/ionograms-direct-upload.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ionograms-direct-upload.service
journalctl -u ionograms-direct-upload.service -n 30 --no-pager
```

It checks every 15 seconds, sends at most eight files per pass, and ignores
files modified in the last 60 seconds. It streams bytes rather than loading
the capture into memory, and detects modification during hashing or transfer.
Both sides check SHA-256. Transfers land in temporary files and are published
atomically; incomplete or mismatched uploads cannot enter the ingest queue.
The receiver parses the HDF5 in the worker before it becomes a database row.

Only files modified in the last hour are considered, oldest first. This
avoids shipping the entire history when first enabled. Increase
`--lookback-hours` after a long outage if those older files are needed.
Already acknowledged files are recorded in `--state`; failures
are retried after a 60-second backoff. A changed server URL or station resets
the sender's receipt cache. The sender does not delete recordings. A changed
file with an already received name is refused with 409: completed files are immutable.
Run only one uploader per station/state file.

## Verify

Upload: `PUT /stations/{station}/files/{basename}` with the existing agent
Bearer token, binary body, Content-Length, and X-Content-SHA256. Maximum
file size is 256 MiB. A 202 acknowledges **durable receipt**, not extraction.
Repeating the same file is safe, including after a lost HTTP response.

`GET /stations/{station}/files/{basename}` with that token reports `queued`,
`ingested`, or `muted`, the SHA-256, timestamps and any last error/retry time.
Check `sounding_id`, then the circuit's `/ui/sim/live` page. The circuit must
already be followed; enabling transfers does not change that selection.

An existing row with the same filename is updated rather than duplicated.
Its local file can be used even if the NAS is unavailable. Soundings older
than the simulator's current circuit cursor do not replay old forecasts.

Expected delay is recording duration, the 60-second stability wait, transfer
and extraction time, then the simulator's usual polling interval (60 seconds
by default). This does not eliminate the four-minute NIC2 recording time.

Local files are retained; **there is no automatic deletion or NAS fallback**.
Provision and monitor the hub's local volume accordingly. Do not delete these
files while database rows reference them. The sender state also persists
across restarts; deleting it rechecks/reuploads the lookback window safely.

To stop new deliveries, stop the station uploader. Set `DIRECT_INGEST=0` and
recreate the API to stop its worker and reject new uploads; retained files,
receipts, database rows and NAS archiving are unchanged.
