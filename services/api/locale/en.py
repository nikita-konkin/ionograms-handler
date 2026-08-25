"""English. The source language, and the fallback for every missing key.

Keys are namespaced by the page that uses them, with `common.*` for strings
that appear on more than one and `*.js.*` for the ones inline JavaScript
builds -- `i18n.js_catalog` selects on that middle segment, so a string moving
into or out of a `<script>` block has to be renamed with it.

Placeholders are `str.format` names, never positions: a translation has to be
free to put the number somewhere other than where English puts it. Values
substituted into them are escaped; the message itself may carry inline markup,
because most of this console's text is prose with a `<b>` or a link in it and a
message chopped into fragments around those cannot be translated at all.

**`*.js.*` is for messages the browser assembles from live values** -- a
confirmation naming a station, a refusal naming a folder. A string that is
fixed at render time is not one of them, even when it ends up inside a
`<script>`: Jinja writes it there with `{{ t('key')|tojson }}`, which keeps it
out of every page that does not use it.

Those messages must also stay markup-free. They are written into the DOM with
`textContent`, so a tag in one of them appears on screen as a tag.
"""

from __future__ import annotations

MESSAGES: dict[str, str] = {

    # -- chrome ---------------------------------------------------------
    "app.title": "Ionograms",
    "chrome.language": "Language",

    "nav.console": "Console",
    "nav.series": "Series",
    "nav.soundings": "Soundings",
    "nav.sources": "Sources",
    "nav.archives": "Archives",
    "nav.forecast": "Forecast",
    "nav.api": "API",

    # -- units ------------------------------------------------------------
    # The single letters `_duration` composes into `9h04m`, and the words the
    # tables put after a number.
    "unit.s": "s",
    "unit.m": "m",
    "unit.h": "h",
    "unit.min": "min",
    "unit.d": "d",
    "unit.km": "km",
    "unit.mhz": "MHz",

    # -- shared -----------------------------------------------------------
    "common.from": "From",
    "common.to": "To",
    "common.apply": "Apply",
    "common.clear": "Clear",
    "common.open": "Open",
    "common.yes": "Yes",
    "common.previous": "Previous",
    "common.next": "Next",
    "common.col.circuit": "tx → rx",
    "common.pager": "{first}–{last} of {total}",
    "common.pager.empty": "0 of {total}",

    # -- soundings --------------------------------------------------------
    "soundings.title": "Soundings",
    "soundings.hint": "{n} matching",
    "soundings.filter.tx": "tx",
    "soundings.filter.tx.any": "Any",
    "soundings.filter.format": "Format",
    "soundings.filter.format.any": "Any",
    "soundings.filter.picks": "Picks",
    "soundings.filter.picks.any": "Any",
    "soundings.filter.picks.some": "At least one",
    "soundings.filter.picks.none": "None",
    "soundings.col.time": "Time (UTC)",
    "soundings.col.format": "Format",
    "soundings.col.sweep": "Sweep MHz",
    "soundings.col.gate": "Gate km",
    "soundings.col.complete": "Complete",
    "soundings.col.picks": "Picks",
    "soundings.empty": "Nothing matches these filters.",
    "soundings.empty.clear": "Clear them.",

    # -- one sounding -----------------------------------------------------
    "sounding.notfound": "Not found",
    "sounding.all": "All soundings",
    "sounding.earliest": "Earliest sounding",
    "sounding.latest": "Latest sounding",
    "sounding.arrows": "&larr; and &rarr; step through soundings in time order.",
    "sounding.col.path": "Path",
    "sounding.col.sweep": "Sweep",
    "sounding.col.gate": "Gate",
    "sounding.col.file": "File",
    "sounding.percent_complete": "{pct}% complete",

    "sounding.ionogram": "Ionogram",
    "sounding.gate_label": "Range gate:",
    "sounding.gate.auto": "Auto",
    "sounding.gate.full": "Full",
    "sounding.plot_label": "Plot:",
    "sounding.plot.interactive": "Interactive",
    "sounding.plot.rendered": "Rendered",
    "sounding.scaling_label": "Scaling:",
    "sounding.method_title": "Which estimator's trace is drawn",
    "sounding.sao": "Download SAO.XML",
    "sounding.sao_title": "All three records, SAO.XML 5.0",
    "sounding.auto_note":
        "<b>Auto</b> fits the window to where the echo is. A search-mode v2 "
        "product stores &plusmn;3998 km while the trace occupies a few "
        "hundred, so at full extent it is a hairline in an empty field.",
    "sounding.rendered_note":
        "Rendered on request, not precomputed &mdash; 288 images a day "
        "precomputed is 105k files a year that mostly nobody opens.",
    "sounding.no_scaling": "No scaling",
    "sounding.scaling_failed":
        "The stored extractions below are unaffected &mdash; they were written "
        "at ingest. Try <a href=\"{url}\">rendered</a>, which needs no SAO "
        "record.",
    "sounding.show_points": "Scaled points",
    "sounding.show_raster": "Raster",
    "sounding.show_marks": "MUF / LOF",
    "sounding.legend_hint":
        "&mdash; Click a legend entry to toggle one trace, double-click to "
        "isolate it.",
    "sounding.relative_pill": "Relative range",
    "sounding.relative_note":
        "This record's range zero is not trustworthy: differences along the "
        "trace are right, the origin is not. Do not read the axis as group "
        "range.",
    "sounding.traces_points":
        "{method} &mdash; {traces} trace(s), {points} point(s)",
    "sounding.declined": "This estimator declined &mdash; no pick",
    "sounding.letter_title": "UAG-23A qualifying letter {letter}",
    "sounding.model_note":
        "Modelled rows are marked with the model that asserted them and are "
        "never used to correct the measurement. IRI and the measured MUF "
        "disagree by several MHz at times, and writing them side by side does "
        "not settle which is right.",
    "sounding.scaled_in": "Scaled in {cost}s.",

    "sounding.extractions": "Extractions",
    "sounding.col.method": "Method",
    "sounding.col.muf": "MUF",
    "sounding.col.lof": "LOF",
    "sounding.col.range": "Range km",
    "sounding.col.snr": "SNR dB",
    "sounding.col.run": "Run",
    "sounding.col.hops": "Hops",
    "sounding.col.scatter": "Scatter",
    "sounding.col.flags": "Flags",
    "sounding.no_pick": "No pick",
    "sounding.limited_title":
        "Pick at the top of the sweep: the MUF is a lower bound",
    "sounding.loflim_title": "LOF ran off the bottom of the band",
    "sounding.stored_note":
        "Stored at ingest. Quality columns travel with the value, never "
        "separately &mdash; a <b>limited</b> MUF is a lower bound, not a "
        "measurement. The panel above is scaled live from the product, so it "
        "can differ from these if the detectors have changed since.",

    "sounding.axis.freq": "Frequency (MHz)",
    "sounding.axis.range": "Virtual range (km)",
    "sounding.hop": "Hop",

    # -- forecast ---------------------------------------------------------
    "forecast.title": "Forecast",
    "forecast.hint":
        "{models} model(s) registered, {n} live on {circuits} circuit(s)",
    "forecast.live": "Live",
    "forecast.col.circuit": "Circuit",
    "forecast.col.param": "Param",
    "forecast.col.model": "Model",
    "forecast.col.last_issue": "Last issue",
    "forecast.col.state": "State",
    "forecast.no_model": "No model",
    "forecast.stale": "Stale",
    "forecast.ok": "Ok",
    "forecast.no_circuits":
        "No circuit has enough extracted data to forecast yet.",
    "forecast.overtaken": "Overtaken",
    "forecast.drift":
        "{circuit} / {param}: <b>{model}</b> scores {mae} MHz at {lead} where "
        "<b>{baseline}</b> scores {baseline_mae}, over {n} pairs. "
        "Surfaced, not demoted.",
    "forecast.nothing_live":
        "Nothing is live. That is the normal state of a fresh deployment, not "
        "a fault &mdash; add a model below, or train one on this circuit's own "
        "measurements, then promote it. Until then <code>/forecast</code> "
        "returns nothing rather than something unvouched-for.",

    "forecast.models": "Models",
    "forecast.col.origin": "Origin",
    "forecast.col.inputs": "Inputs",
    "forecast.golden.recorded": "golden recorded",
    "forecast.golden.absent": "golden absent",
    "forecast.unbound": "Unbound",
    "forecast.of": "of",
    "forecast.state.active": "ACTIVE",
    "forecast.state.comparison": "Comparison",
    "forecast.activate": "Activate",
    "forecast.retire": "Retire",
    "forecast.activate.modelled_title":
        "Fitted against a modelled target, so it can be compared for ever and "
        "never promoted. The schema refuses it, not this page.",
    "forecast.activate.unbound_title":
        "Bound to no circuit, so there is no forecast it could be. Re-import "
        "it with --tx and --rx.",
    "forecast.nothing_registered":
        "Nothing registered. Upload an artifact below, or train one. Either "
        "way the file is opened by a worker rather than by this server: "
        "registering a model means running code out of it, and the process "
        "answering this request has no business doing that.",

    # -- forecast: issuing one now ----------------------------------------
    #
    # Activating a model does not produce a forecast; `infer` does, on its next
    # pass. Before this button the only way to say so was a shell command, and
    # an empty "Last issue" column with no way to act on it is where a person
    # gets stuck.
    "forecast.run": "Run now",
    "forecast.run.title":
        "Issue a forecast for this circuit now, instead of waiting for the "
        "next scheduled pass. The model is loaded by `infer`, not by this "
        "server.",
    "forecast.run.no_model_title":
        "No model is live for this circuit, so there is no forecast to issue. "
        "Activate one below first.",
    "forecast.run.recent": "Recent passes",
    "forecast.run.col.requested": "Asked for",
    "forecast.run.col.rows": "Rows",
    "forecast.run.backtest": "backtest",

    # -- forecast: adding a model -----------------------------------------
    #
    # The panel exists because registering a model used to need a shell on the
    # host. What has *not* changed is that the api never opens an artifact:
    # these strings describe a queue, and the wording is deliberate about it.
    "forecast.add": "Add a model",
    "forecast.add.hint":
        "The file is hashed, checked, and put in a holding area. It is not "
        "opened here &mdash; a <code>.sav</code> is a pickle, and loading one "
        "runs code out of it, so a worker with no network surface does that "
        "part. Expect a few seconds between accepted and registered.",
    "forecast.add.file": "Artifact",
    "forecast.add.name": "Name",
    "forecast.add.name.placeholder": "default: the file name",
    "forecast.add.circuit": "Circuit",
    "forecast.add.circuit.any": "Unbound (comparison only)",
    "forecast.add.origin": "Origin",
    "forecast.add.origin.imported": "imported &mdash; dropped in by hand",
    "forecast.add.origin.legacy": "legacy &mdash; from the research archive",
    "forecast.add.origin.trained": "trained &mdash; fitted elsewhere",
    "forecast.add.target_src": "Target",
    "forecast.add.target_src.auto": "default for this origin",
    "forecast.add.target_src.measured": "measured &mdash; promotable",
    "forecast.add.target_src.modelled": "modelled &mdash; comparison only",
    "forecast.add.note": "Note",
    "forecast.add.submit": "Upload",
    "forecast.add.queue": "In the queue",
    "forecast.add.col.file": "File",
    "forecast.add.col.uploaded": "Uploaded",
    "forecast.add.col.detail": "What happened",
    "forecast.add.empty": "Nothing waiting.",
    "forecast.add.forget": "Forget",
    "forecast.state.pending": "Queued",
    "forecast.state.refused": "Refused",

    # -- forecast: training -----------------------------------------------
    "forecast.train": "Train a model",
    "forecast.train.hint":
        "Fits on this circuit's own measured picks &mdash; inputs from the "
        "tracked grid, target from the picks themselves, band-edge bounds "
        "excluded, and the last days held back. The result is registered for "
        "comparison and reports how it did against persistence; promoting it "
        "stays a separate decision.",
    "forecast.train.lead": "Lead",
    "forecast.train.estimator": "Estimator",
    "forecast.train.holdout": "Hold back (days)",
    "forecast.train.submit": "Queue training",
    "forecast.train.jobs": "Training runs",
    "forecast.train.col.requested": "Requested",
    "forecast.train.col.lead": "Lead",
    "forecast.train.empty": "Nothing has been trained on this deployment yet.",
    "forecast.train.cancel": "Cancel",
    "forecast.state.queued": "Queued",
    "forecast.state.running": "Running",
    "forecast.state.done": "Done",
    "forecast.state.failed": "Failed",
    "forecast.state.cancelled": "Cancelled",

    "forecast.leaderboard": "Leaderboard",
    "forecast.leaderboard.sub": "{circuit} &middot; {param} &middot; MAE (MHz)",
    "forecast.circuit_label": "Circuit:",
    "forecast.col.subject": "Subject",
    "forecast.col.pairs": "Pairs",
    "forecast.baselines_divider": "&mdash; Baselines &mdash;",
    "forecast.baseline": "Baseline",
    "forecast.censored_note":
        "Censored hours &mdash; a pick at the top of the sweep or the band "
        "floor &mdash; are scored one-sidedly and counted apart, so a bound "
        "never dilutes the number above. Rescore with <code>python -m "
        "services.prediction.scoring --once</code>.",
    "forecast.nothing_scored":
        "Nothing scored for this circuit yet. Run <code>python -m "
        "services.prediction.scoring --once</code>, or wait for the next "
        "<code>infer</code> pass &mdash; scoring runs behind it. Until then "
        "the table above reports what a model <i>is</i>, never how good it is.",
    "forecast.col.what": "What it is",
    "forecast.baseline.persistence":
        "The value one lead time ago; at a 24 h horizon that is yesterday at "
        "the same UTC minute, which captures the diurnal cycle for free",
    "forecast.baseline.recurrence":
        "One solar rotation ago; the standard operational HF baseline",
    "forecast.baseline.iri":
        "The stored IRI reference at the path control point &mdash; already "
        "built and validated. MUF only: IRI says nothing about the absorption "
        "floor that sets LOF",
    "forecast.baseline.harmonic":
        "Diurnal harmonics plus solar zenith angle, fitted strictly before the "
        "scored window so it cannot become an oracle on its own leaderboard",

    "forecast.js.run_slow":
        "The pass for {name} is still queued. `infer` checks every few "
        "seconds; if that container is not running, nothing will issue a "
        "forecast.",
    "forecast.js.no_file": "Choose an artifact file first.",
    "forecast.js.uploading": "Uploading {name}\u2026",
    "forecast.js.waiting":
        "{name} accepted. Waiting for the registrar to open it\u2026",
    "forecast.js.slow":
        "{name} is still queued. The registrar polls every few seconds; if it "
        "is not running, nothing will open this file.",
    "forecast.js.queued_training":
        "Queued. The trainer picks it up within a minute; a fit takes minutes, "
        "not seconds.",
    "forecast.js.forget_confirm":
        "Forget {name}?\n\nThe uploaded bytes are deleted. Nothing that is "
        "already registered is affected.",
    "forecast.js.cancel_confirm":
        "Cancel training job {id}?\n\nOnly a job that has not started can be "
        "cancelled; a fit that is running goes to its end.",
    "forecast.js.activate_confirm":
        "Make {name} the {param} forecast for {circuit}?\n\n"
        "Whatever is live for that circuit is demoted in the same step.",
    "forecast.js.retire_confirm":
        "Retire {name}?\n\n"
        "Its rows and its forecasts are kept, so this can be undone by "
        "activating it again.",
    # -- archives ---------------------------------------------------------
    "archives.title": "Archives",
    "archives.hint": "{n} folder(s) registered",
    "archives.mounted": "Mounted archive",
    "archives.col.host": "Host folder",
    "archives.col.seen_as": "Seen here as",
    "archives.col.state": "State",
    "archives.not_reported": "Not reported",
    "archives.primary": "Primary",
    "archives.not_readable": "Not readable",
    "archives.mounted_empty": "Mounted but empty",
    "archives.ok": "Ok",
    "archives.in_container":
        "Running in a container, so &ldquo;seen here as&rdquo; is the mount "
        "point and the host folder is what <code>deploy/.env</code> set.",
    "archives.one_root":
        "One root. To index a folder on another disk, give it a "
        "<code>volumes:</code> line of its own <i>and</i> list its container "
        "path in <code>ARCHIVE_ROOTS</code> &mdash; both, then redeploy. A "
        "path cannot be added from this page: a container's filesystem is "
        "fixed when it starts.",
    "archives.fault.unreachable":
        "<b>The mount is there; the storage behind it is not answering.</b> "
        "<code>{root}</code> exists inside the container and every read of it "
        "fails &mdash; {error}. <code>ARCHIVE_HOST_PATH</code> is <i>not</i> "
        "the problem here and redeploying will not help: fix it on the host, "
        "where the volume is mounted, then the scans resume on their own. "
        "Nothing can be registered or indexed until then.",
    "archives.fault.denied":
        "The mount exists but this process may not read it &mdash; {error}. "
        "The api runs as uid 10001; that uid needs read access to the host "
        "folder. No folder can be registered until this is fixed.",
    "archives.fault.missing":
        "Nothing is readable at <code>{root}</code>. Under Docker that means "
        "the bind mount is missing or its source is gone &mdash; set "
        "<code>ARCHIVE_HOST_PATH</code> in <code>deploy/.env</code> and "
        "redeploy. No folder can be registered until this is fixed.",
    "archives.fault.empty":
        "The mount exists but holds nothing. A bind mount whose source was "
        "renamed on the host still appears inside the container, as an empty "
        "directory &mdash; every scan would then report &ldquo;0 on "
        "disk&rdquo; truthfully and forever.",
    "archives.intro":
        "Folders under <code>{root}</code> that this server keeps indexed. A "
        "scan runs the pipeline over whatever is new, so every sounding it "
        "loads arrives with its characteristics already derived &mdash; MUF, "
        "LOF, group range and SNR in the database, and the full scaling "
        "behind each sounding's page and its <code>sao.xml</code>. There is "
        "no separate step to compute them.",
    "archives.rescan_on":
        "Enabled folders are re-scanned automatically every {minutes} min. "
        "Scans run one at a time: they are CPU-bound and they lock the "
        "database, so two would finish later than one after the other.",
    "archives.rescan_off":
        "Automatic re-scanning is off "
        "(<code>ARCHIVE_SCAN_INTERVAL_S=0</code>), so new files appear only "
        "when someone presses scan.",
    "archives.col.name": "Name",
    "archives.col.path": "Path",
    "archives.col.format": "Format",
    "archives.col.methods": "Methods",
    "archives.col.soundings": "Soundings",
    "archives.col.last_scan": "Last scan",
    "archives.col.result": "Result",
    "archives.disabled": "Disabled",
    "archives.any": "Any",
    "archives.set": "Set",
    "archives.scan_now": "Scan now",
    "archives.disable": "Disable",
    "archives.enable": "Enable",
    "archives.remove": "Remove",
    "archives.nothing_registered":
        "Nothing registered yet. Until a folder is here, this server indexes "
        "only what someone ran <code>services.api.ingest</code> over by hand.",

    "archives.add": "Add a folder",
    "archives.add_note":
        "The path is relative to <code>{root}</code>. Only folders under that "
        "root can be indexed &mdash; in the container it is the single path "
        "mounted at <code>/archive</code>, so a folder elsewhere is not merely "
        "unreadable, it is invisible, and a scan of it would report success "
        "having loaded nothing.",
    "archives.cands_note":
        "Folders that hold daily sounding data, and the days inside each of "
        "them. Registering one covers every day it contains and every day the "
        "receiver adds later &mdash; scanning is recursive, so a new day "
        "folder needs no action here.",
    "archives.col.folder": "Folder",
    "archives.col.root": "Root",
    "archives.col.holds": "Holds",
    "archives.mount_itself": "(The mount itself)",
    "archives.day_folders": "{n} {unit}",
    "archives.inside":
        "Inside <code>{path}</code> &mdash; register one or the other, not both",
    "archives.stored_absolute": "Stored absolute",
    "archives.indexed": "{n} indexed",
    "archives.holds": "{n} sounding(s)",
    "archives.unreadable": "Nothing this server can read",
    "archives.registered": "Registered",
    "archives.use": "Use",
    "archives.field.path": "Path",
    "archives.field.name": "Name",
    "archives.name_placeholder": "(Defaults to the path)",
    "archives.field.format": "Format",
    "archives.add_button": "Add",
    "archives.methods": "Methods",
    "archives.all": "All",
    "archives.methods_note":
        "Adding a method later re-scopes the folder's existing soundings on "
        "the next scan &mdash; anything already computed is kept.",
    "archives.methods_note.cnn":
        "A method that cannot run here is disabled rather than merely warned "
        "about: <code>already_done</code> counts a sounding finished only when "
        "it holds a row for <i>every</i> requested method, so choosing one "
        "that never produces a row would re-scan the whole folder on every "
        "pass, forever.",

    # -- series -----------------------------------------------------------
    "series.title": "Series",
    "series.hint": "{n} point(s)",
    "series.h2": "Parameters against time",
    "series.nothing_ingested": "Nothing ingested yet",
    "series.circuit": "Circuit",
    "series.all_overlaid": "All (overlaid)",
    "series.no_picks": "No picks for this method",
    "series.day": "Day",
    "series.all": "All",
    "series.reference": "Reference",
    "series.off": "Off",
    "series.off_note": "Off skips the model, and the network it may need.",
    "series.bare_date": "A bare date covers that whole day at either end.",
    "series.family.forecast": "Forecast",
    "series.forecast": "Model:",
    "series.forecast_note":
        "Draws one model's latest issue on the same axis as the picks it "
        "is meant to predict. Only models that have written rows for this "
        "circuit are offered. A candidate is dotted; the operational "
        "forecast, when there is one, is dashed.",
    "series.family.context": "Sweep top / hmF2",
    "series.drag_hint":
        "&mdash; Drag to zoom, double-click to reset, click a point to open "
        "its ionogram.",
    "series.hue_note":
        "<b>Hue is the parameter, shape is the source.</b> Markers are "
        "measured, lines are modelled &mdash; so a blue marker and the blue "
        "dashed line beside it are this circuit's MUF and IRI's, and the gap "
        "between them is what the residual panel plots.",
    "series.hue_note.multi":
        "With several circuits overlaid, hue is the <b>circuit</b> instead "
        "&mdash; two paths' MUFs in one colour is exactly the comparison this "
        "must not invite &mdash; so only MUF and its model are drawn at first. "
        "Each circuit is modelled at its own control point; the legend groups "
        "them.",
    "series.hollow_note":
        "<b>Hollow markers are bounds, not measurements.</b> A hollow MUF "
        "marker sat at the top of the sweep, so the real MUF is at or above "
        "it; a hollow LOF marker sat at the band floor, so the real LOF is at "
        "or below it. They are drawn rather than filtered out &mdash; dropping "
        "either silently would bend the curve towards the middle of the band. "
        "They are left out of the residual statistics below, which is a "
        "different decision and is stated there.",
    "series.fof2_note":
        "<b>foF2 here is not measured.</b> An oblique sounder never sees "
        "vertical incidence. It is the measured MUF put back through the "
        "secant law at hmF2&nbsp;=&nbsp;{hmf2}&nbsp;km over one hop, which is "
        "what makes it comparable with a model or with a nearby vertical "
        "ionosonde. IRI's own foF2 is its model output, converted the same way "
        "in reverse to get the MUF beside it.",
    "series.lof_note":
        "<b>LOF, not LUF.</b> ITU-R P.533-13 sec. 9 defines the lowest "
        "<i>usable</i> frequency with a required signal-to-noise ratio and a "
        "monthly median &mdash; a property of a service and of a month, "
        "neither of which one sounding has. What an oblique sounder scales is "
        "the lowest <i>observed</i> frequency, and that is what this is. It "
        "tracks D-region absorption, so it follows the sun rather than the F2 "
        "layer: a MUF that moves while the LOF under it does not is worth a "
        "second look.",
    "series.summary": "Summary",
    "series.col.n": "n",
    "series.col.muf_median": "MUF median",
    "series.col.at_ceiling": "At ceiling",
    "series.col.lof_median": "LOF median",
    "series.col.at_floor": "At floor",
    "series.col.fof2_median": "foF2 median",
    "series.col.vs_iri": "Vs IRI: n",
    "series.col.bias": "Bias",
    "series.col.rms": "RMS",
    "series.col.r": "r",
    "series.path_km": "{km} km",
    "series.hops": ", {n} hops",
    "series.limited_title": "Pick at the top of the sweep: a lower bound",
    "series.loflim_title": "LOF ran off the bottom of the band: an upper bound",
    "series.excluded_title": "Lower bounds, left out",
    "series.reference_off": "Reference off",
    "series.no_pair": "No pair to compare",
    "series.bias_note":
        "<b>Bias is a median of measured &minus; IRI, over the pairs that are "
        "measurements.</b> The lower bounds counted in the <i>at ceiling</i> "
        "column are excluded from it: a pick pinned to the top of the sweep "
        "says the ionosphere supported <i>at least</i> that, and scoring it as "
        "a residual would report the recorder's band ceiling as a modelling "
        "error. On a ceiling-limited circuit that is most of the daytime, "
        "which is why the count of excluded points is printed beside the one "
        "used.",
    "series.iri_note":
        "IRI is a climatology driven by a smoothed solar index, and for a "
        "recent month that index does not exist yet &mdash; see "
        "<code>muf/reference/indices.py</code>. Treat a few MHz of "
        "disagreement as normal and the <i>shape</i> of the residual as the "
        "interesting part: a flat offset is a scale question, a diurnal one is "
        "not.",

    # -- sources ----------------------------------------------------------
    "sources.title": "Sources",
    "sources.hint": "{n} emitter(s)",
    "sources.receiver": "Receiver",
    "sources.no_station":
        "No station has reported yet, so there is nothing to identify a "
        "transmitter <i>for</i>.",
    "sources.station": "Station",
    "sources.per_receiver": "Everything on this page is per receiver.",
    "sources.slot_note":
        "A slot second is a <b>reception</b> second &mdash; the transmit "
        "second plus the one-way travel time plus this receiver's own epoch "
        "offset. The same transmitter heard at two receivers has two different "
        "<code>chirpt</code> values, so an identification made here belongs to "
        "this circuit and not to the transmitter alone. (The band ceiling is "
        "keyed the same way, and for the same reason.)",

    "sources.heard": "Transmitters heard",
    "sources.census_note":
        "From <b>{kind}</b> files over the last {days} day(s), grouped by "
        "chirp rate and arrival phase on a {cycle} s cycle. This is what "
        "<b>search</b> mode is for: it finds who is transmitting, and each row "
        "is a candidate to be identified and then sounded on purpose.",
    "sources.building_pill": "First census running",
    "sources.building":
        "<b>No census has finished yet, so this is not &ldquo;no transmitters "
        "heard&rdquo;.</b> Listing one day of this archive costs minutes "
        "before a file is opened, so the census runs in the background and the "
        "page never waits for it. Reload in a few minutes. If it never "
        "completes, the archive mount is the thing to look at, not this page.",
    "sources.as_of": "As of {minutes} min ago",
    "sources.refreshing": "A newer one is being read now.",
    "sources.next_at": "A newer one starts when this passes {minutes} min.",
    "sources.stale_note":
        "The archive is too slow to census while you wait, so the page shows "
        "the last completed one and its age rather than blocking on a new one.",
    "sources.cached": "Cached",
    "sources.read": "Read {files} file(s) from {where} in {seconds} s",
    "sources.archive_root": "The archive root",
    "sources.opened": "&mdash; opened {opened}, {cached} already known",
    "sources.nothing_reopened": "&mdash; nothing re-opened",
    "sources.unreadable":
        "<b>{n} file(s) could not be read</b> and were skipped; a detector "
        "caught mid-write is normal, a steady count is not.",
    "sources.one_day": "1 day",
    "sources.capped_pill": "Capped",
    "sources.capped":
        "<b>This is the newest {files} of {found} file(s), not all of them.</b> "
        "The census stops at {budget} because opening the rest would take "
        "hours on this archive, and a page that never answers is worse than "
        "one that answers about the recent end of it. It trims time, not "
        "quality: the same detection product, newest days first, so today is "
        "whole and the oldest day is the one that lost files. An emitter that "
        "only ever transmitted in the part that was trimmed will be missing "
        "&mdash; ask for <a href=\"{url}\">1 day</a> to spend the whole budget "
        "on today, and prune the archive if this stays capped.",
    "sources.slow":
        "<b>That is slow, and it is the archive, not the arithmetic.</b> A "
        "census opens one HDF5 file per detection, and each open costs a round "
        "trip on a network archive. Every file read here is remembered &mdash; "
        "these products are written once and never change &mdash; so the next "
        "load should open only what has arrived since. If it stays slow, the "
        "process was restarted or the archive is being rewritten under it.",

    "sources.col.rate": "Rate",
    "sources.col.heard_at": "Heard at (s into cycle)",
    "sources.col.n": "n",
    "sources.col.per_slot": "/slot",
    "sources.col.per_slot_title":
        "Detections divided by distinct slots: how often it came back",
    "sources.col.phase": "Phase ms",
    "sources.col.snr": "SNR",
    "sources.col.span": "Span h",
    "sources.col.identified_as": "Identified as",
    "sources.identify": "Identify",
    "sources.khz_s": "{rate} kHz/s",
    "sources.seconds_note":
        "<b>Seconds are as received, not as transmitted.</b> A slot is the "
        "transmit second plus the one-way travel time plus this receiver's "
        "epoch offset. For scheduling that is the number you want &mdash; the "
        "station needs to know when to <i>listen</i> &mdash; but it is not a "
        "transmit time and not a range.",
    "sources.no_emitter":
        "No repeating emitter found in the last {days} day(s) with at least "
        "{min_count} detections. Search mode has to be running, and its "
        "detection files have to be reaching this archive.",
    "sources.rejected_summary": "{n} group(s) rejected as interference",
    "sources.rejected_note":
        "Rejected on <b>shape, not strength</b>. The loudest group in this "
        "archive was the least real: 500 kHz/s at a higher median SNR than "
        "cyprus1, claiming every one of the 300 seconds in the cycle with its "
        "arrival phase scattering &plusmn;274&nbsp;ms. A transmitter is silent "
        "in most seconds and arrives at the same instant within the few it "
        "uses. Shown so the list can be checked &mdash; if the row you came "
        "for is down here, the thresholds are wrong, not the transmitter.",
    "sources.col.rejected_because": "Rejected because",

    "sources.identify_panel": "Identify this emitter",
    "sources.identify_note":
        "Nothing in a detection says who is transmitting. The name you give "
        "here is a <b>judgement</b>, made the way <code>cyprus1</code> was "
        "resolved to <code>NIC</code> &mdash; and it is load-bearing: it is "
        "written into the schedule as <code>transmit_name</code>, which "
        "<code>calc_ionograms.py</code> puts in the product's <i>file name</i> "
        "and in <code>txname</code>. This pipeline reads that back as the "
        "transmitter's identity and looks up its coordinates and band ceiling "
        "by it. An emitter you cannot name can still be scheduled &mdash; call "
        "it what you know it as &mdash; but that string is what every product "
        "from it will be called.",
    "sources.field.code": "Code",
    "sources.field.txname": "Full name",
    "sources.field.rep": "Rep (s)",
    "sources.field.note": "Note",
    "sources.name_rule":
        "Letters, digits, underscore and dot &mdash; <b>no dash</b>. The "
        "product file name is dash-delimited "
        "(<code>lfm_ionogram-&#123;tx&#125;-&#123;rx&#125;-&#123;ch&#125;-"
        "&#123;id&#125;-&#123;t0&#125;.h5</code>), so a dash inside the code "
        "does not fail to parse, it shifts every field after it.",
    "sources.slots_label": "Slots to sound:",
    "sources.slots_note":
        "One entry per ticked slot. Each is another sounding per cycle "
        "competing for the ringbuffer window &mdash; see the README on reading "
        "<code>find_timings.log</code> before ticking many.",
    "sources.note_placeholder": "How you know -- what settled it",
    "sources.save": "Save transmitter",
    "sources.cancel": "Cancel",
    "sources.verified": "Verified transmitters",
    "sources.verified_note":
        "Identified once, kept with the census row that settled it. These are "
        "what a schedule is composed from &mdash; a raw census row cannot be "
        "scheduled, because it has no name and no id, and "
        "<code>calc_ionograms.py</code> reads both without a default. "
        "<b>Choosing which of them to sound, and starting the sounding, is on "
        "the <a href=\"/ui\">console</a></b>, beside the station that would do "
        "it: the schedule and the recorder are one decision, and this page has "
        "to read the archive before it can render at all.",

    # -- console ----------------------------------------------------------
    "console.title": "Station console",
    "console.hint": "Auto-refresh 15 s",
    "console.archive": "Archive",
    "console.ingested": "{n} sounding(s) ingested.",
    "console.method_picks": "{method} &middot; {picks}/{n} picks",
    "console.no_extractions":
        "No extractions yet &mdash; register a folder on "
        "<a href=\"/ui/archives\">archives</a> and scan it",

    "console.upstream": "Upstream",
    "console.net.online": "INTERNET OK",
    "console.net.degraded": "PARTIAL",
    "console.net.offline": "NO INTERNET",
    "console.net.unknown": "INTERNET?",
    "console.checked_ago": "Checked {age} ago",
    "console.net_note":
        "{detail}. These are the servers the solar indices come from, not the "
        "internet at large &mdash; a host that reaches them through a mirror "
        "is green here and a host behind a proxy that blocks "
        "<code>sidc.be</code> is not, which is the distinction that decides "
        "whether the IRI values on a sounding page can be refreshed.",
    "console.col.host": "Host",
    "console.col.rtt": "Rtt",
    "console.col.provides": "Provides",
    "console.col.cached": "Cached",
    "console.col.detail": "Detail",
    "console.never": "Never",
    "console.ok": "Ok",
    "console.fail": "FAIL",
    "console.cache_note":
        "<b>Reachable and cached are different questions.</b> Unreachable with "
        "a fresh cache is a model still answering correctly; reachable with "
        "<span class=\"pill unk\">never</span> is a model that has never had a "
        "driver. A cached file older than seven days is re-fetched on the next "
        "use, and a stale one is served anyway if the fetch fails &mdash; so "
        "an index can be months old without anything else on this console "
        "saying so.",

    "console.no_stations": "No stations",
    "console.no_stations_note":
        "Nothing has pushed a health report yet. Start the agent with "
        "<code>server_url</code> pointing here and it will appear within one "
        "push interval.",
    "console.stale": "STALE",
    "console.healthy": "HEALTHY",
    "console.unhealthy": "UNHEALTHY",
    "console.last_report": "Last report {age} s ago",
    "console.never_reported": "Never reported",
    "console.agent": "Agent {version}",
    "console.stale_note":
        "Nothing has arrived for more than {seconds} s. Health is pushed, so "
        "silence is itself the alert &mdash; the metrics below are the last "
        "ones received, not the current state.",
    "console.forget": "Forget this receiver",
    "console.forget_note":
        "For one that was <b>renamed or retired</b>. Drops its health history "
        "and queued commands, which is what this panel is made of. "
        "Identifications and configuration epochs are kept &mdash; they are "
        "the provenance of products already on disk.",

    "console.acquisition": "Acquisition",
    "console.acquiring": "ACQUIRING",
    "console.not_acquiring": "NOT ACQUIRING",
    "console.no_products": "NO PRODUCTS",
    "console.acquiring_unknown": "ACQUIRING?",
    "console.mode.scheduled": "Scheduled",
    "console.mode.search": "Search",
    "console.mode.unrecorded": "Mode not recorded",
    "console.sounding_pill": "SOUNDING",
    "console.slot_due": "SLOT DUE",
    "console.between_slots": "Between slots",
    "console.oversubscribed": "RANK {ranks} OVERSUBSCRIBED",
    "console.configured_at": "Configured {when}",
    "console.changed_by": "by {who}",
    "console.due_note":
        "<b>The slot marked due is the schedule against the clock, not an "
        "observation:</b> it says a chirp is expected this second, not that "
        "one is being recorded.",
    "console.search_note":
        "The station is in <b>search</b> mode, so this schedule is what the "
        "ini holds, not what it is doing: search records whatever sweeps past "
        "and infers the timing afterwards.",
    "console.col.rank": "Rank",
    "console.col.transmitter": "Transmitter",
    "console.col.rate": "Rate",
    "console.col.slot": "Slot",
    "console.col.state": "State",
    "console.col.next": "Next",
    "console.unidentified": "Unidentified",
    "console.khz_s": "{rate} kHz/s",
    "console.slot_of": "{chirpt}s of {rep}s",
    "console.sounding_state": "Sounding",
    "console.due_state": "Due",
    "console.seconds_in": "{age} in, sweep ends {ends}",
    "console.sweep_unknown": "Sweep length unknown",
    "console.idle": "Idle",
    "console.in": "In {age}",
    "console.contended":
        "<b>Rank {ranks} has more slots in progress than it has processes.</b> "
        "A rank sounds one chirp at a time: it takes the slot with the "
        "shortest wait and is busy for the whole sweep, so the overlapping one "
        "is skipped that cycle without anything saying so. Give the "
        "transmitters their own ranks, or move the slots more than one sweep "
        "apart.",
    "console.span_note":
        "A sweep is timed from the band this receiver's recent products "
        "actually cover ({span} MHz) and the entry's chirp rate &mdash; "
        "measured, not configured.",
    "console.no_span_note":
        "No product has been ingested for this receiver, so the sweep length "
        "is unknown and no slot can be called in progress. The slot times "
        "themselves are still exact.",

    "console.preview_alt": "Newest sounding from {tx}",
    "console.old": "{age} old",
    "console.preview_note":
        "The newest product on <b>this station's own disk</b>, one per "
        "transmitter, encoded there and pushed with the health report &mdash; "
        "so it is current even when the archive transfer is hours behind. Age "
        "is the sounding's own start time. Colours are the same "
        "20&ndash;75 dB <code>jet</code> scale as a full ionogram",
    "console.preview_note.cropped":
        "; an asterisk marks a range axis narrowed to fit the echoes",
    "console.no_preview_note":
        "This station sends no preview. Either its agent predates the feature "
        "or <code>preview</code> is <code>false</code> in its "
        "<code>agent.json</code>; whether it has products to show at all is "
        "what <code>newest_product_age_s</code> below answers.",

    "console.col.arrived": "Arrived",
    "console.col.from": "From",
    "console.col.age": "Age",
    "console.col.sweep": "Sweep",
    "console.arrivals_note":
        "Last products <b>ingested</b> for this receiver. A stalled transfer "
        "shows old arrivals while the recorder is fine &mdash; the "
        "<code>newest_product_age_s</code> metric below is the one that speaks "
        "for the station itself, and it is what the indicator at the top of "
        "this panel reads.",
    "console.nothing_ingested_for":
        "Nothing ingested under the receiver name <code>{name}</code>.",

    "console.col.metric": "Metric",
    "console.col.value": "Value",
    "console.unknown_note":
        "Grey <span class=\"pill unk\">?</span> is <b>unknown</b>, not fine: "
        "the agent could not measure it. Only a definite failure makes a "
        "station unhealthy.",

    "console.plan": "Sounding plan",
    "console.mode": "Mode",
    "console.not_recorded": "&mdash; Not recorded &mdash;",
    "console.apply": "Apply",
    "console.queued_note":
        "Queued now; the station applies it on its next pull and it takes "
        "effect when acquisition restarts",
    "console.col.code": "Code",
    "console.col.name": "Name",
    "console.col.slots": "Slots",
    "console.col.identified": "Identified",
    "console.col.note": "Note",
    "console.slots_of": "{slots}s of {rep}s",
    "console.nothing_identified":
        "Nothing identified at this receiver yet, so there is no schedule to "
        "compose. A census row cannot be scheduled &mdash; it has no name and "
        "no id, and <code>calc_ionograms.py</code> reads both without a "
        "default. Name one on <a href=\"/ui/sources\">sources</a> and it "
        "appears here.",

    "console.band": "Receive band",
    "console.read_only": "Read-only",
    "console.col.digitised": "Digitised",
    "console.col.analysed": "Analysed",
    "console.col.lo": "LO",
    "console.configured": "Configured",
    "console.observed": "Observed",
    "console.band_not_reported":
        "Not reported &mdash; this station has not sent a health report "
        "carrying its band yet",
    "console.from_products": "From {n} product(s)",
    "console.no_products_yet": "No products yet",
    "console.disagree": "<b>Configured and observed disagree:</b> {detail}",
    "console.agree":
        "Configured and observed agree to within {tolerance} MHz. The small "
        "shortfall is expected: <code>calc_ionograms.py</code> selects stored "
        "bins with strict inequalities, so both edge bins are dropped.",
    "console.band_start": "Band start",
    "console.analyse_from": "Analyse from",
    "console.apply_band": "Apply band",
    "console.presets": "Presets:",
    "console.preset_v2": "0&ndash;25 (v2 default)",
    "console.preset_current": "7.5&ndash;32.5 (current)",

    "console.start": "Start",
    "console.stop": "Stop",
    "console.restart": "Restart",
    "console.queued_pull": "Queued now, collected on the station's next pull",
    "console.col.command": "Command",
    "console.col.issued": "Issued",
    "console.col.acked": "Acked",
    "console.col.result": "Result",
    "console.acked": "Acked",
    "console.failed": "Failed",
    "console.delivered": "Delivered",
    "console.pending": "Pending",

    "console.token": "Control token",
    "console.token_note":
        "Held in this tab only (sessionStorage), never baked into the page. "
        "Required for start/stop/restart.",

    # -- shared, browser side ---------------------------------------------
    "common.js.no_token": "No control token. Paste it on the console page first.",

    # -- archives, browser side -------------------------------------------
    "archives.js.path_set":
        "Path set to {path} \u2014 check the methods below, then press add.",
    "archives.js.enter_path": "Enter a path first.",
    "archives.js.checking": "Checking \u2026",
    "archives.js.choose_method": "Choose at least one method.",
    "archives.js.replace_confirm":
        "{detail}\n\nRegister it and drop those rows?",
    "archives.js.registered":
        "{name} registered \u2014 {n} sounding(s) visible{kinds}. "
        "Press scan now to index them.",
    "archives.js.starting": "Starting \u2026",
    "archives.js.methods_set": "Methods set to {methods} \u2014 {note}",
    "archives.js.unrecognised": "Unrecognised",
    "archives.js.out_of_scope": "{what} out of scope",
    "archives.js.remove": "Remove",
    "archives.js.these": "These",
    "archives.js.drop_confirm":
        "Delete {what} sounding(s), with their extractions and modelled "
        "values? The files are not touched \u2014 widening the format reads "
        "them again.",
    "archives.js.press_again": "Press again to confirm",
    "archives.js.unregister_confirm":
        "Unregister {name}?\n\nIts soundings and their characteristics stay "
        "in the database; this only stops the indexing.",
    "archives.js.reading":
        "Reading {name} \u2014 listing files and checking what is already "
        "indexed",
    "archives.js.reading_detail":
        "This step reads the folder before any indexing starts. It is not "
        "stuck.",
    "archives.js.indexing": "Indexing {name} \u2014 {done} of {total} ({pct}%)",
    "archives.js.left": "~{time} left",
    "archives.js.loaded": "{n} {unit} loaded",
    "archives.js.skipped": ", {n} skipped",
    "archives.js.loaded_note":
        " \u2014 characteristics are derived as each file is read, so what is "
        "done is already usable.",
    "archives.js.result": "{name}: {result}",
    "archives.js.looking": "Looking at what is mounted \u2026",
    "archives.js.no_unregistered": "No unregistered folders under the mount.",
    "archives.js.why": "{why}.",

    # -- series, inside the plot ------------------------------------------
    # Fixed at render time, so Jinja writes them into the script rather than
    # `T` looking them up in the browser.
    "series.trace.muf": "MUF",
    "series.trace.muf_bound": "MUF (lower bound)",
    "series.trace.muf_smooth": "MUF smoothed",
    "series.trace.lof": "LOF",
    "series.trace.lof_bound": "LOF (at band floor)",
    "series.trace.fof2": "foF2 (equivalent)",
    "series.trace.iri_muf": "IRI MUF",
    "series.trace.iri_fof2": "IRI foF2",
    "series.trace.iri_hmf2": "IRI hmF2",
    "series.trace.residual": "Measured &minus; IRI",
    "series.trace.residual_bound": "Residual (bound)",
    "series.trace.sweep_top": "Sweep top",
    "series.js.trace.forecast": "{param} forecast",
    "series.js.trace.forecast_band": "{param} forecast \u00b1\u03c3",
    "series.js.trace.forecast_compare": "{param} \u00b7 {model}",
    "series.hover.muf_ceiling": "MUF at the top of the sweep",
    "series.hover.muf_smooth": "MUF, smoothed by track",
    "series.hover.lof_floor": "LOF at the band floor",
    "series.hover.fof2": "foF2 implied by the measured MUF",
    "series.hover.iri_hmf2": "IRI hmF2, right axis",
    "series.hover.residual_bound": "Lower bound: not in the statistics",
    "series.hover.sweep_top": "Top of the sweep",
    "series.axis.time": "Time (UTC)",
    "series.axis.frequency": "Frequency (MHz)",
    "series.axis.residual": "Meas &minus; IRI (MHz)",
    "series.axis.hmf2": "hmF2 (km)",

    # -- sources, browser side --------------------------------------------
    "sources.js.code_required": "A code is required.",
    "sources.js.tick_a_slot": "Tick at least one slot.",
    "sources.js.saving": "Saving...",
    "sources.js.refused": "Refused: {detail}",
    "sources.js.none_identified":
        "Nothing identified at this receiver yet. Pick a row above and name it.",
    "sources.js.khz_s": "{rate} kHz/s",
    "sources.js.of_rep": "of {rep}s",
    "sources.js.forget": "Forget",
    "sources.js.forget_confirm":
        "Forget {code} at {station}?\n\nProducts already recorded keep the "
        "name; only the identification and its evidence are dropped.",
    "sources.js.col.code": "Code",
    "sources.js.col.name": "Name",
    "sources.js.col.id": "id",
    "sources.js.col.rate": "Rate",
    "sources.js.col.slots": "Slots",
    "sources.js.col.verified": "Verified",
    "sources.js.col.note": "Note",

    # -- console, browser side --------------------------------------------
    "console.js.paste_token": "Paste the control token below first.",
    "console.js.stop_confirm":
        "Stop acquisition on {station}? The recorder is stopped with SIGINT so "
        "the USRP is released cleanly, and nothing is recorded until it is "
        "started again. Press stop again to confirm; the page will not refresh "
        "until you do.",
    "console.js.stop_cancelled": "Stop was not confirmed, so nothing was sent.",
    "console.js.sending": "{name} \u2026",
    "console.js.refused": "Refused: {detail}",
    "console.js.queued":
        "{name} queued as {id} \u2014 pending until the station's agent "
        "collects it.",
    "console.js.forget_confirm":
        "Forget {station}? Its health history and queued commands are deleted "
        "-- including when its reports stopped. Identifications and "
        "configuration epochs are kept. Press again to confirm; the page will "
        "not refresh until you do.",
    "console.js.forget_cancelled":
        "Forget was not confirmed, so nothing was deleted.",
    "console.js.forgetting": "Forgetting {station} \u2026",
    "console.js.forgotten": "Forgotten",
    "console.js.live_again": "Nothing was applied, so the page is live again.",
    "console.js.no_mode":
        "This server has no record of the mode this station is in, so nothing "
        "is pre-selected. Choose one to see what would be sent.",
    "console.js.search_no_schedule":
        "Search mode needs no schedule: it records whatever sweeps past and "
        "infers the timing afterwards. The ticks are ignored until the mode is "
        "scheduled.",
    "console.js.plan":
        "{n} transmitter(s): {codes} \u2014 {n} rank group(s), so "
        "calc_ionograms.py must run with -np {n}.",
    "console.js.tick_one":
        "Tick at least one transmitter. Scheduled mode with an empty schedule "
        "records nothing while every process reports healthy.",
    "console.js.choose_mode":
        "Choose a mode first. This server has no record of the one the station "
        "is in, and applying a default would configure it from an assumption "
        "rather than from what you know.",
    "console.js.tick_one_first":
        "Tick at least one transmitter first \u2014 the server refuses "
        "scheduled mode without a schedule, and it is right to.",
    "console.js.schedule_queued":
        "Schedule queued as {id} \u2014 {ranks} rank(s) for {transmitters}. It "
        "reaches the station on its next pull and takes effect when "
        "acquisition restarts, so press restart after it is collected.",
    "console.js.mode_queued":
        "Mode {mode} queued as {id} \u2014 applied on the station's next pull, "
        "effective on restart.",
    "console.js.enter_band": "Enter a band start to see what would be sent.",
    "console.js.enter_band_first": "Enter a band start first.",
    "console.js.no_sample_rate":
        "This station has not reported its sample rate, so the passband cannot "
        "be shown here. The agent still checks it before applying anything.",
    "console.js.band_plan":
        "LO {lo} MHz, digitising {start}\u2013{stop} MHz",
    "console.js.window_inverted": "The analysis window is inverted",
    "console.js.window_outside":
        "\u2717 The analysis window reaches outside the digitised band \u2014 "
        "that part is FFTs over spectrum the radio never sampled, and the "
        "server will refuse it",
    "console.js.sweep_too_long":
        "\u2717 {code}: a {sweep} s sweep does not fit its {rep} s cycle",
    "console.js.sweep_over_budget":
        "\u26a0 {code}: {sweep} s sweep, past the {budget} s ringbuffer budget "
        "\u2014 soundings will be lost, as they are today",
    "console.js.sweep_ok": "{code}: {sweep} s sweep of its {rep} s cycle",
    "console.js.band_sending": "Sending \u2026",
    "console.js.band_queued":
        "Band queued as {id} \u2014 the station applies it on its next pull "
        "and it takes effect when acquisition restarts, so press restart after "
        "it is collected. The observed row above is how you confirm it landed: "
        "it should move to the new window within one sounding cycle.",

}

#: Noun forms selected by `i18n.plural`. English needs two; the `few`/`many`
#: split exists for the languages that do, and is simply absent here.
PLURALS: dict[str, dict[str, str]] = {
    "sounding": {"one": "sounding", "other": "soundings"},
    "trace": {"one": "trace", "other": "traces"},
    "point": {"one": "point", "other": "points"},
    "model": {"one": "model", "other": "models"},
    "circuit": {"one": "circuit", "other": "circuits"},
    "pair": {"one": "pair", "other": "pairs"},
    "folder": {"one": "folder", "other": "folders"},
    "dayfolder": {"one": "day folder", "other": "day folders"},
    "emitter": {"one": "emitter", "other": "emitters"},
    "day": {"one": "day", "other": "days"},
    "file": {"one": "file", "other": "files"},
    "group": {"one": "group", "other": "groups"},
    "product": {"one": "product", "other": "products"},
}
