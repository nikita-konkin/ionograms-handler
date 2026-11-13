# Replace LFS captures from the web console

The **Conversion / Конвертация** tab at `/ui/conversion` creates persistent
jobs for a registered archive or a subfolder of it. It is an administrative
console feature, authorized by the existing archive capability. Job APIs,
including file listings, require the same capability.

## Operator workflow

1. Register the source archive on **Archives** if it is not listed already.
2. Open **Conversion**, select the archive and enter a relative subfolder
   (for example `2026.02.04`, or `.` for the entire archive).
3. Click **Inspect folder**. The worker lists eligible captures, their total
   size, and a paginated file manifest. Inspection does not change captures.
   Links and files modified less than a minute ago are excluded.
4. Review the manifest and tick the replacement/deletion acknowledgement.
5. Click **Start or resume replacement**. Progress reports completed files,
   errors and output size. Failed files are listed first in the detail view.

Each `day/name.lfs` becomes `day/name.echo.h5`, in the same directory. The
worker uses the reference converter's adaptive echo selection, including
three-hop geometry and a 150 km margin. It stores complex64 samples, filter
taps, the original 512-byte LFS header, timing/calibration and noise metadata.
It does not store the full-band raw voltage. Signals outside the saved band
cannot be recovered from this derivative; it is not a reversible raw backup.

**Cancel job** stops before the next file. A completed replacement stays in
place. Pending files retain their original capture. Resume retries failures
and remaining files; changed sources require a new folder inspection.
Closing the browser does not stop the worker. Interrupted inspections repeat
their directory walk, while interrupted replacements resume their manifest.

## Replacement and recovery

Before unlinking a capture, the worker requires all of the following:

- Its identity, size and modification time still match the inspected file.
- The output passes the converter's numerical verification against the source.
- The carried LFS header matches byte for byte; the selection uses rule 2
  and has no `trace_at_edge` warning. Old or possibly cropped windows retain
  their raw source for review.
- The echo product can be opened by the normal loader.
- The source checksum remains unchanged, and the replacement is flushed to
  storage. Storage must honor file and directory flushes.

The worker commits both SHA-256 checksums to a per-file verification journal
before deleting anything. It then updates the existing sounding's filename,
path and format and deletes the `.lfs`. Sounding IDs and existing extraction,
reference and dependent records are preserved. A crash after unlink but before
SQLite commit is recovered using the verified output checksum and manifest.
Unrelated, damaged or linked existing outputs are refused without overwriting
them. Other indexing workers skip files during the replacement handoff.
An output can remain alongside its raw capture after a failed check, for
review. Directory scans do not index that derivative while its raw exists.

Standalone echo files are discoverable as `echo_window`. While both a raw
capture and its derivative exist, default directory scans prefer the raw file
to avoid duplicate soundings. The `lfs` archive filter includes its echo
replacements, so a partially converted archive discovers both remaining
captures and converted files without admitting unrelated product formats.
Unindexed replacements are processed by the usual archive scan.

## One-time worker deployment

The API only queues jobs. Deploy the dedicated worker on the same host, with
the same database, registered archive roots and paths. The worker needs create,
rename and delete permission in the selected archive directories. With the
container image's default user, that is uid 10001. The ordinary API and other
archive readers keep their read-only mounts.

For prebuilt hub images containing this change:

```sh
docker compose -f deploy/docker-compose.hub.yml --env-file deploy/.env \
  --profile archive-conversion up -d archive-conversion
```

For the source-build Compose stack:

```sh
docker compose -f deploy/docker-compose.yml --env-file deploy/.env \
  --profile archive-conversion up -d --build archive-conversion
```

Update the API and all archive indexers to the same code version to expose
the tab, new schema and replacement-aware indexing.
After this one-time deployment, inspection, replacement, cancellation and
resume are all controlled through the web tab. No conversion starts merely
by enabling the service. A worker heartbeat is shown on the page.

For a host installation, run `python -m services.archive_converter` with the
same `API_DB`, `ARCHIVE_ROOT` and `ARCHIVE_ROOTS` as the API. `--once` handles
one queued job. Only one worker per database can hold the process lock.

Use completed archive folders for historical conversion. Another sync or
acquisition program must not rewrite captures while they are being replaced,
or restore the deleted raw files from an upstream copy afterward. The worker
does not delete `.lfp` sidecars or other files.
