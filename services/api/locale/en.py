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
    "app.title": "ionograms",
    "chrome.language": "language",

    "nav.console": "console",
    "nav.series": "series",
    "nav.soundings": "soundings",
    "nav.sources": "sources",
    "nav.archives": "archives",
    "nav.forecast": "forecast",
    "nav.api": "api",

    # -- units ------------------------------------------------------------
    # The single letters `_duration` composes into `9h04m`, and the words the
    # tables put after a number.
    "unit.s": "s",
    "unit.m": "m",
    "unit.h": "h",
    "unit.d": "d",
    "unit.km": "km",
    "unit.mhz": "MHz",

    # -- shared -----------------------------------------------------------
    "common.from": "from",
    "common.to": "to",
    "common.apply": "apply",
    "common.clear": "clear",
    "common.open": "open",
    "common.yes": "yes",
    "common.previous": "previous",
    "common.next": "next",
    "common.col.circuit": "tx → rx",
    "common.pager": "{first}–{last} of {total}",
    "common.pager.empty": "0 of {total}",

    # -- soundings --------------------------------------------------------
    "soundings.title": "soundings",
    "soundings.hint": "{n} matching",
    "soundings.filter.tx": "tx",
    "soundings.filter.tx.any": "any",
    "soundings.filter.format": "format",
    "soundings.filter.format.any": "any",
    "soundings.filter.picks": "picks",
    "soundings.filter.picks.any": "any",
    "soundings.filter.picks.some": "at least one",
    "soundings.filter.picks.none": "none",
    "soundings.col.time": "time (UTC)",
    "soundings.col.format": "format",
    "soundings.col.sweep": "sweep MHz",
    "soundings.col.gate": "gate km",
    "soundings.col.complete": "complete",
    "soundings.col.picks": "picks",
    "soundings.empty": "Nothing matches these filters.",
    "soundings.empty.clear": "Clear them.",

    # -- one sounding -----------------------------------------------------
    "sounding.notfound": "not found",
    "sounding.all": "all soundings",
    "sounding.earliest": "earliest sounding",
    "sounding.latest": "latest sounding",
    "sounding.arrows": "&larr; and &rarr; step through soundings in time order.",
    "sounding.col.path": "path",
    "sounding.col.sweep": "sweep",
    "sounding.col.gate": "gate",
    "sounding.col.file": "file",
    "sounding.percent_complete": "{pct}% complete",

    "sounding.ionogram": "ionogram",
    "sounding.gate_label": "range gate:",
    "sounding.gate.auto": "auto",
    "sounding.gate.full": "full",
    "sounding.plot_label": "plot:",
    "sounding.plot.interactive": "interactive",
    "sounding.plot.rendered": "rendered",
    "sounding.scaling_label": "scaling:",
    "sounding.method_title": "which estimator's trace is drawn",
    "sounding.sao": "download SAO.XML",
    "sounding.sao_title": "all three records, SAO.XML 5.0",
    "sounding.auto_note":
        "<b>auto</b> fits the window to where the echo is. A search-mode v2 "
        "product stores &plusmn;3998 km while the trace occupies a few "
        "hundred, so at full extent it is a hairline in an empty field.",
    "sounding.rendered_note":
        "Rendered on request, not precomputed &mdash; 288 images a day "
        "precomputed is 105k files a year that mostly nobody opens.",
    "sounding.no_scaling": "no scaling",
    "sounding.scaling_failed":
        "The stored extractions below are unaffected &mdash; they were written "
        "at ingest. Try <a href=\"{url}\">rendered</a>, which needs no SAO "
        "record.",
    "sounding.show_points": "scaled points",
    "sounding.show_raster": "raster",
    "sounding.show_marks": "MUF / LOF",
    "sounding.legend_hint":
        "&mdash; click a legend entry to toggle one trace, double-click to "
        "isolate it.",
    "sounding.relative_pill": "relative range",
    "sounding.relative_note":
        "This record's range zero is not trustworthy: differences along the "
        "trace are right, the origin is not. Do not read the axis as group "
        "range.",
    "sounding.traces_points":
        "{method} &mdash; {traces} trace(s), {points} point(s)",
    "sounding.declined": "this estimator declined &mdash; no pick",
    "sounding.letter_title": "UAG-23A qualifying letter {letter}",
    "sounding.model_note":
        "Modelled rows are marked with the model that asserted them and are "
        "never used to correct the measurement. IRI and the measured MUF "
        "disagree by several MHz at times, and writing them side by side does "
        "not settle which is right.",
    "sounding.scaled_in": "Scaled in {cost}s.",

    "sounding.extractions": "extractions",
    "sounding.col.method": "method",
    "sounding.col.muf": "MUF",
    "sounding.col.lof": "LOF",
    "sounding.col.range": "range km",
    "sounding.col.snr": "SNR dB",
    "sounding.col.run": "run",
    "sounding.col.hops": "hops",
    "sounding.col.scatter": "scatter",
    "sounding.col.flags": "flags",
    "sounding.no_pick": "no pick",
    "sounding.limited_title":
        "pick at the top of the sweep: the MUF is a lower bound",
    "sounding.loflim_title": "LOF ran off the bottom of the band",
    "sounding.stored_note":
        "Stored at ingest. Quality columns travel with the value, never "
        "separately &mdash; a <b>limited</b> MUF is a lower bound, not a "
        "measurement. The panel above is scaled live from the product, so it "
        "can differ from these if the detectors have changed since.",

    "sounding.axis.freq": "Frequency (MHz)",
    "sounding.axis.range": "Virtual range (km)",
    "sounding.hop": "hop",

    # -- forecast ---------------------------------------------------------
    "forecast.title": "forecast",
    "forecast.hint":
        "{models} model(s) registered, {n} live on {circuits} circuit(s)",
    "forecast.live": "live",
    "forecast.col.circuit": "circuit",
    "forecast.col.param": "param",
    "forecast.col.model": "model",
    "forecast.col.last_issue": "last issue",
    "forecast.col.state": "state",
    "forecast.no_model": "no model",
    "forecast.stale": "stale",
    "forecast.ok": "ok",
    "forecast.no_circuits":
        "No circuit has enough extracted data to forecast yet.",
    "forecast.overtaken": "overtaken",
    "forecast.drift":
        "{circuit} / {param}: <b>{model}</b> scores {mae} MHz at {lead} where "
        "<b>{baseline}</b> scores {baseline_mae}, over {n} pairs. "
        "Surfaced, not demoted.",
    "forecast.nothing_live":
        "Nothing is live. That is the normal state of a fresh deployment, not "
        "a fault &mdash; register a model with <code>python -m "
        "services.prediction.importer</code>, then promote one below. Until "
        "then <code>/forecast</code> returns nothing rather than something "
        "unvouched-for.",

    "forecast.models": "models",
    "forecast.col.origin": "origin",
    "forecast.col.inputs": "inputs",
    "forecast.golden.recorded": "golden recorded",
    "forecast.golden.absent": "golden absent",
    "forecast.unbound": "unbound",
    "forecast.of": "of",
    "forecast.state.active": "ACTIVE",
    "forecast.state.comparison": "comparison",
    "forecast.activate": "activate",
    "forecast.retire": "retire",
    "forecast.activate.modelled_title":
        "Fitted against a modelled target, so it can be compared for ever and "
        "never promoted. The schema refuses it, not this page.",
    "forecast.activate.unbound_title":
        "Bound to no circuit, so there is no forecast it could be. Re-import "
        "it with --tx and --rx.",
    "forecast.nothing_registered":
        "Nothing registered. <code>python -m services.prediction.importer "
        "&lt;file&gt; --param muf</code> registers one; there is deliberately "
        "no route that does it over HTTP, because registering a model means "
        "running code out of a file on a shared volume.",

    "forecast.leaderboard": "leaderboard",
    "forecast.leaderboard.sub": "{circuit} &middot; {param} &middot; MAE (MHz)",
    "forecast.circuit_label": "circuit:",
    "forecast.col.subject": "subject",
    "forecast.col.pairs": "pairs",
    "forecast.baselines_divider": "&mdash; baselines &mdash;",
    "forecast.baseline": "baseline",
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
    "forecast.col.what": "what it is",
    "forecast.baseline.persistence":
        "the value one lead time ago; at a 24 h horizon that is yesterday at "
        "the same UTC minute, which captures the diurnal cycle for free",
    "forecast.baseline.recurrence":
        "one solar rotation ago; the standard operational HF baseline",
    "forecast.baseline.iri":
        "the stored IRI reference at the path control point &mdash; already "
        "built and validated. MUF only: IRI says nothing about the absorption "
        "floor that sets LOF",
    "forecast.baseline.harmonic":
        "diurnal harmonics plus solar zenith angle, fitted strictly before the "
        "scored window so it cannot become an oracle on its own leaderboard",

    "forecast.js.no_token": "No control token. Paste it on the console page first.",
    "forecast.js.activate_confirm":
        "Make {name} the {param} forecast for {circuit}?\n\n"
        "Whatever is live for that circuit is demoted in the same step.",
    "forecast.js.retire_confirm":
        "Retire {name}?\n\n"
        "Its rows and its forecasts are kept, so this can be undone by "
        "activating it again.",
    # -- archives ---------------------------------------------------------
    "archives.title": "archives",
    "archives.hint": "{n} folder(s) registered",
    "archives.mounted": "mounted archive",
    "archives.col.host": "host folder",
    "archives.col.seen_as": "seen here as",
    "archives.col.state": "state",
    "archives.not_reported": "not reported",
    "archives.primary": "primary",
    "archives.not_readable": "not readable",
    "archives.mounted_empty": "mounted but empty",
    "archives.ok": "ok",
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
    "archives.col.name": "name",
    "archives.col.path": "path",
    "archives.col.format": "format",
    "archives.col.methods": "methods",
    "archives.col.soundings": "soundings",
    "archives.col.last_scan": "last scan",
    "archives.col.result": "result",
    "archives.disabled": "disabled",
    "archives.any": "any",
    "archives.set": "set",
    "archives.scan_now": "scan now",
    "archives.disable": "disable",
    "archives.enable": "enable",
    "archives.remove": "remove",
    "archives.nothing_registered":
        "Nothing registered yet. Until a folder is here, this server indexes "
        "only what someone ran <code>services.api.ingest</code> over by hand.",

    "archives.add": "add a folder",
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
    "archives.col.folder": "folder",
    "archives.col.root": "root",
    "archives.col.holds": "holds",
    "archives.mount_itself": "(the mount itself)",
    "archives.day_folders": "{n} {unit}",
    "archives.inside":
        "inside <code>{path}</code> &mdash; register one or the other, not both",
    "archives.stored_absolute": "stored absolute",
    "archives.indexed": "{n} indexed",
    "archives.holds": "{n} sounding(s)",
    "archives.unreadable": "nothing this server can read",
    "archives.registered": "registered",
    "archives.use": "use",
    "archives.field.path": "path",
    "archives.field.name": "name",
    "archives.name_placeholder": "(defaults to the path)",
    "archives.field.format": "format",
    "archives.add_button": "add",
    "archives.methods": "methods",
    "archives.all": "all",
    "archives.methods_note":
        "Adding a method later re-scopes the folder's existing soundings on "
        "the next scan &mdash; anything already computed is kept.",
    "archives.methods_note.cnn":
        "A method that cannot run here is disabled rather than merely warned "
        "about: <code>already_done</code> counts a sounding finished only when "
        "it holds a row for <i>every</i> requested method, so choosing one "
        "that never produces a row would re-scan the whole folder on every "
        "pass, forever.",
    "archives.looking": "looking at what is mounted &hellip;",

    # -- series -----------------------------------------------------------
    "series.title": "series",
    "series.hint": "{n} point(s)",
    "series.h2": "Parameters against time",
    "series.nothing_ingested": "nothing ingested yet",
    "series.circuit": "circuit",
    "series.all_overlaid": "all (overlaid)",
    "series.no_picks": "no picks for this method",
    "series.day": "day",
    "series.all": "all",
    "series.reference": "reference",
    "series.off": "off",
    "series.off_note": "off skips the model, and the network it may need.",
    "series.bare_date": "A bare date covers that whole day at either end.",
    "series.family.forecast": "forecast",
    "series.family.context": "sweep top / hmF2",
    "series.drag_hint":
        "&mdash; drag to zoom, double-click to reset, click a point to open "
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
    "series.summary": "summary",
    "series.col.n": "n",
    "series.col.muf_median": "MUF median",
    "series.col.at_ceiling": "at ceiling",
    "series.col.lof_median": "LOF median",
    "series.col.at_floor": "at floor",
    "series.col.fof2_median": "foF2 median",
    "series.col.vs_iri": "vs IRI: n",
    "series.col.bias": "bias",
    "series.col.rms": "RMS",
    "series.col.r": "r",
    "series.path_km": "{km} km",
    "series.hops": ", {n} hops",
    "series.limited_title": "pick at the top of the sweep: a lower bound",
    "series.loflim_title": "LOF ran off the bottom of the band: an upper bound",
    "series.excluded_title": "lower bounds, left out",
    "series.reference_off": "reference off",
    "series.no_pair": "no pair to compare",
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
}
