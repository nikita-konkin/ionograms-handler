/* The public results page. Reads status.json and data/{circuit}.json written
 * by services/publish/publisher.py; draws everything with plain DOM and SVG so
 * the site carries no third-party code and runs under `default-src 'self'`.
 * All text goes in through textContent, never innerHTML.
 */
(function () {
  "use strict";

  var STALE_S = 3600;            // older than this, the banner says so
  var REFRESH_MS = 5 * 60 * 1000;
  var NOW_MS = 10 * 1000;       // now.json: the sounding in progress
  var NOW_STALE_S = 120;        // older than this, the live feed is delayed
  var RANGES = [1, 7, 30];
  var THEMES = ["auto", "light", "dark"];
  var MIN_SPAN_MS = 2 * 3600000;  // the closest the chart zooms in
  var ZOOM_STEP = 1.5;             // one button press, in or out
  var SVG = "http://www.w3.org/2000/svg";

  var state = {
    lang: "en", strings: {}, status: null, slug: null, range: 7,
    data: {}, shown: null, now: null, link: null, hidden: {},
    // view: the [from, to] a reader scrolled or zoomed to, null for the
    // chosen range; geo: what the last drawing mapped, for the gestures
    view: null, geo: null, theme: "auto"
  };

  // --- helpers -------------------------------------------------------------

  function t(key, vars) {
    var table = state.strings[state.lang] || {};
    var text = table[key] != null ? table[key] : key;
    Object.keys(vars || {}).forEach(function (k) {
      text = text.replace("{" + k + "}", vars[k]);
    });
    return text;
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }

  function svg(tag, attrs) {
    var node = document.createElementNS(SVG, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // Every stamp from the publisher is UTC; some carry no zone suffix.
  function parseT(text) {
    if (!text) return NaN;
    var s = String(text).replace(" ", "T");
    if (!/(Z|[+-]\d\d:?\d\d)$/.test(s)) s += "Z";
    return Date.parse(s);
  }

  function pad(n) { return (n < 10 ? "0" : "") + n; }

  // Moscow has kept UTC+3 all year since 2014, so its zone is a fixed offset.
  var MSK_MS = 3 * 3600 * 1000;

  function ymd(d) {
    return d.getUTCFullYear() + "-" + pad(d.getUTCMonth() + 1) + "-" + pad(d.getUTCDate());
  }

  function hhmm(d) { return pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes()); }

  // `2026-10-29 06:45 UTC · 09:45 МСК`; Moscow's date only when it is not
  // UTC's. `withDate === false` is the bare UTC time an axis tick wants.
  function fmtTime(ms, withDate) {
    var d = new Date(ms), m = new Date(ms + MSK_MS);
    if (withDate === false) return hhmm(d);
    return ymd(d) + " " + hhmm(d) + " " + t("utc") + " · " +
      (ymd(m) === ymd(d) ? "" : ymd(m) + " ") + hhmm(m) + " " + t("msk");
  }

  function fmtDay(ms) {
    var d = new Date(ms);
    return pad(d.getUTCDate()) + "." + pad(d.getUTCMonth() + 1);
  }

  function ago(seconds) {
    if (seconds < 5400) return t("ago_min", { n: Math.max(1, Math.round(seconds / 60)) });
    if (seconds < 172800) return t("ago_h", { n: Math.round(seconds / 3600) });
    return t("ago_d", { n: Math.round(seconds / 86400) });
  }

  function mhz(v) { return v == null ? "—" : v.toFixed(1); }

  function getJSON(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + ": " + r.status);
      return r.json();
    });
  }

  function remember(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }

  function recall(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }

  // --- page ----------------------------------------------------------------

  function circuit() {
    var list = (state.status && state.status.circuits) || [];
    for (var i = 0; i < list.length; i++) if (list[i].slug === state.slug) return list[i];
    return null;
  }

  function renderStatic() {
    document.documentElement.lang = state.lang;
    document.title = t("title");
    Array.prototype.forEach.call(document.querySelectorAll("[data-t]"), function (node) {
      node.textContent = t(node.getAttribute("data-t"));
    });
    document.getElementById("lang").textContent = t("lang");
    applyTheme();
  }

  // Auto leaves the palette to the system; light and dark pin it.
  function applyTheme() {
    var root = document.documentElement;
    if (state.theme === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", state.theme);
    document.getElementById("theme").textContent = t("theme_" + state.theme);
  }

  function renderHeader() {
    var st = state.status;
    var banner = document.getElementById("banner");
    var updated = document.getElementById("updated");
    if (!st) {
      updated.textContent = "";
      banner.textContent = t("unavailable");
      banner.hidden = false;
      return;
    }
    var generated = parseT(st.generated_at);
    var age = (Date.now() - generated) / 1000;
    updated.textContent = t("updated") + ": " + fmtTime(generated);
    banner.hidden = !(age > STALE_S);
    banner.textContent = age > STALE_S ? t("delayed", { age: ago(age) }) : "";
    if (!st.circuits.length) {
      banner.textContent = t("no_circuits");
      banner.hidden = false;
    }
    document.getElementById("build").textContent =
      st.build ? "build " + st.build : "";
  }

  function renderTabs() {
    var nav = document.getElementById("circuits");
    clear(nav);
    ((state.status && state.status.circuits) || []).forEach(function (c) {
      var b = el("button", {
        type: "button", text: c.tx + " → " + c.rx,
        "aria-pressed": String(c.slug === state.slug),
        title: c.tx_name + " → " + c.rx_name
      });
      b.addEventListener("click", function () { select(c.slug); });
      nav.appendChild(b);
    });
  }

  function pointNear(ms) {
    var d = state.data[state.slug];
    if (!d) return null;
    var s = d.series, best = null, bestGap = 120000;   // within 2 minutes
    for (var i = s.t.length - 1; i >= 0; i--) {
      var gap = Math.abs(parseT(s.t[i]) - ms);
      if (gap <= bestGap) { best = i; bestGap = gap; }
      if (parseT(s.t[i]) < ms - bestGap) break;
    }
    return best == null ? null : { muf: s.muf[best], lof: s.lof[best], limited: s.limited[best] };
  }

  function showImage(c, image) {
    state.shown = image ? image.img : null;
    var img = document.getElementById("latest-img");
    var cap = document.getElementById("latest-cap");
    clear(cap);
    if (!image) {
      img.hidden = true;
      cap.textContent = t("no_images");
      return;
    }
    var ms = parseT(image.t);
    img.hidden = false;
    img.src = image.img;
    img.alt = c.tx_name + " → " + c.rx_name + ", " + fmtTime(ms);
    cap.appendChild(document.createTextNode(fmtTime(ms)));
    var p = pointNear(ms);
    if (p) {
      cap.appendChild(document.createTextNode(" · " + t("muf") + " "));
      cap.appendChild(el("b", { text: mhz(p.muf) }));
      if (p.limited) cap.appendChild(document.createTextNode(" (" + t("limited") + ")"));
      cap.appendChild(document.createTextNode(" · " + t("lof") + " "));
      cap.appendChild(el("b", { text: mhz(p.lof) }));
      cap.appendChild(document.createTextNode(" " + t("mhz")));
    }
    Array.prototype.forEach.call(document.querySelectorAll("#strip button"), function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-img") === state.shown));
    });
  }

  function renderImages(c) {
    var strip = document.getElementById("strip");
    clear(strip);
    var images = c.images || [];
    images.slice().reverse().forEach(function (image) {
      var b = el("button", { type: "button", "data-img": image.img }, [
        el("img", { src: image.img, alt: "", loading: "lazy", width: "128", height: "56" }),
        el("span", { text: fmtTime(parseT(image.t), false) })
      ]);
      b.addEventListener("click", function () { showImage(c, image); });
      strip.appendChild(b);
    });
    var keep = images.filter(function (i) { return i.img === state.shown; })[0];
    showImage(c, keep || images[images.length - 1]);
  }

  function renderPath(c) {
    var d = state.data[c.slug];
    var text = t("path") + ": " + c.tx_name + " → " + c.rx_name;
    if (d && d.path_km) text += " · " + Math.round(d.path_km) + " " + t("km");
    document.getElementById("path").textContent = text;
  }

  // --- chart ---------------------------------------------------------------

  var TICKS_H = [0.25, 0.5, 1, 3, 6, 12, 24, 48, 96, 168];

  function niceMax(v) { return Math.max(5, Math.ceil(v / 5) * 5); }

  // The forecast curves a reader can switch on and off: the live model and
  // the baselines it is scored against. The order is the order of the legend.
  var CURVES = ["model", "persistence", "recurrence-27d", "harmonic", "iri"];

  function curveLabel(key, f) {
    return key === "model" ? (f && f.model ? f.model.name : t("model")) : t("m_" + key);
  }

  function curves(d, start, until) {
    var s = d.series, found = [];
    function inside(ts, values, sigmas) {
      var pts = [];
      for (var i = 0; i < ts.length; i++) {
        var x = parseT(ts[i]);
        if (x >= start && x <= until) pts.push([x, values[i], sigmas ? sigmas[i] : null]);
      }
      return pts;
    }
    var fc = (s.forecast || []).filter(function (f) { return f.param === "muf"; })[0];
    if (fc) found.push({ key: "model", pts: inside(fc.t, fc.value, fc.sigma) });
    (s.baselines || []).forEach(function (b) {
      if (CURVES.indexOf(b.name) > 0) found.push({ key: b.name, pts: inside(b.t, b.value) });
    });
    return found.filter(function (c) { return c.pts.length; }).sort(function (a, b) {
      return CURVES.indexOf(a.key) - CURVES.indexOf(b.key);
    });
  }

  function renderChart() {
    var box = document.getElementById("chart");
    clear(box);
    var d = state.data[state.slug];
    if (!d) return;
    var s = d.series;
    var now = parseT(state.status.generated_at);
    var whole = extent(d, now), start, end;
    if (state.view) {
      state.view = clampView(state.view, whole);
      start = state.view[0]; end = state.view[1];
    } else {
      start = Math.max(whole[0], now - state.range * 86400000);
      end = Math.max(now, Math.min(whole[1], now + state.range * 86400000));
    }
    // Curves reach a little past each edge so a line runs to the frame
    // rather than stopping at the last point inside it.
    var margin = (end - start) / 20;

    var muf = [], lof = [];
    for (var i = 0; i < s.t.length; i++) {
      var x = parseT(s.t[i]);
      if (x < start || x > end) continue;
      if (s.muf[i] != null) muf.push([x, s.muf[i], s.limited[i]]);
      if (s.lof[i] != null) lof.push([x, s.lof[i]]);
    }
    var all = curves(d, start - margin, end + margin);
    var shown = all.filter(function (c) { return !state.hidden[c.key]; });
    if (!muf.length && !lof.length && !all.length) {
      box.appendChild(el("p", { "class": "muted", text: t("no_data") }));
      renderLegend([], d);
      return;
    }

    var top = 0;
    muf.forEach(function (p) { top = Math.max(top, p[1]); });
    lof.forEach(function (p) { top = Math.max(top, p[1]); });
    shown.forEach(function (c) {
      c.pts.forEach(function (p) {
        if (p[1] == null || p[0] < start || p[0] > end) return;
        top = Math.max(top, p[1] + (p[2] || 0));
      });
    });
    top = niceMax(top * 1.05);

    var W = Math.max(320, box.clientWidth || 640), H = W < 500 ? 240 : 300;
    var L = 36, R = 10, T = 10, B = 26;
    function X(ms) { return L + (ms - start) / (end - start) * (W - L - R); }
    function Y(v) { return T + (1 - Math.min(v, top) / top) * (H - T - B); }

    var root = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                            "aria-label": t("series") });
    var frame = svg("clipPath", { id: "plot" });
    frame.appendChild(svg("rect", { x: L, y: T, width: W - L - R, height: H - T - B }));
    var defs = svg("defs");
    defs.appendChild(frame);
    root.appendChild(defs);
    var plot = svg("g", { "clip-path": "url(#plot)" });

    // y grid
    var ystep = top > 30 ? 10 : 5;
    for (var v = 0; v <= top; v += ystep) {
      root.appendChild(svg("line", { "class": "grid", x1: L, x2: W - R, y1: Y(v), y2: Y(v) }));
      var lab = svg("text", { "class": "axis", x: L - 6, y: Y(v) + 4, "text-anchor": "end" });
      lab.textContent = String(v);
      root.appendChild(lab);
    }
    // x ticks: the smallest step that leaves at most ~7 labels
    var spanH = (end - start) / 3600000, stepH = TICKS_H[TICKS_H.length - 1];
    var room = Math.max(3, Math.floor((W - L - R) / 90));
    for (var k = 0; k < TICKS_H.length; k++) {
      if (spanH / TICKS_H[k] <= room) { stepH = TICKS_H[k]; break; }
    }
    var stepMs = stepH * 3600000;
    var first = Math.ceil(start / stepMs) * stepMs;
    for (var tx = first; tx <= end; tx += stepMs) {
      root.appendChild(svg("line", { "class": "grid", x1: X(tx), x2: X(tx), y1: T, y2: H - B }));
      var midnight = new Date(tx).getUTCHours() === 0;
      var tl = svg("text", { "class": "axis", x: X(tx), y: H - B + 16, "text-anchor": "middle" });
      tl.textContent = stepH >= 24 || midnight ? fmtDay(tx) : fmtTime(tx, false);
      root.appendChild(tl);
    }

    // "now" marks where measurement ends and forecast begins
    if (now > start && now < end) {
      root.appendChild(svg("line", { "class": "now", x1: X(now), x2: X(now), y1: T, y2: H - B }));
    }
    root.appendChild(plot);

    // each curve, broken wherever a value is missing; the model also has a band
    shown.slice().reverse().forEach(function (c) {
      var runs = [], run = [];
      c.pts.forEach(function (p) {
        if (p[1] == null) { if (run.length) runs.push(run); run = []; } else run.push(p);
      });
      if (run.length) runs.push(run);
      runs.forEach(function (r) {
        var withSigma = r.filter(function (p) { return p[2] != null; });
        if (withSigma.length > 1) {
          var up = withSigma.map(function (p) { return X(p[0]) + "," + Y(p[1] + p[2]); });
          var down = withSigma.slice().reverse().map(function (p) {
            return X(p[0]) + "," + Y(Math.max(0, p[1] - p[2]));
          });
          plot.appendChild(svg("polygon", { "class": "band", points: up.concat(down).join(" ") }));
        }
        plot.appendChild(svg("polyline", { "class": "fc c-" + c.key,
          points: r.map(function (p) { return X(p[0]) + "," + Y(p[1]); }).join(" ") }));
      });
    });

    lof.forEach(function (p) {
      plot.appendChild(svg("circle", { "class": "lof", cx: X(p[0]), cy: Y(p[1]), r: 2 }));
    });
    muf.forEach(function (p) {
      plot.appendChild(svg("circle", { "class": p[2] ? "muf-limited" : "muf",
                                       cx: X(p[0]), cy: Y(p[1]), r: p[2] ? 2.6 : 2.2 }));
    });

    // hover / touch readout: the nearest measurement, or past "now" the
    // nearest forecast instant with every curve that is switched on
    var cursor = svg("line", { "class": "cursor", x1: 0, x2: 0, y1: T, y2: H - B, visibility: "hidden" });
    root.appendChild(cursor);
    var tip = el("div", { "class": "tip", hidden: "" });
    var times = s.t.map(parseT);
    var ahead = {};
    shown.forEach(function (c) {
      c.pts.forEach(function (p) {
        if (p[1] == null || p[0] < start || p[0] > end) return;
        (ahead[p[0]] = ahead[p[0]] || []).push([c.key, p[1]]);
      });
    });
    var aheadTimes = Object.keys(ahead).map(Number);
    function nearest(list, ms, from) {
      var best = -1, gap = Infinity;
      for (var n = 0; n < list.length; n++) {
        if (list[n] < from) continue;
        var g = Math.abs(list[n] - ms);
        if (g < gap) { gap = g; best = n; }
      }
      return [best, gap];
    }
    function readout(evt) {
      var rect = root.getBoundingClientRect();
      var px = ((evt.touches ? evt.touches[0].clientX : evt.clientX) - rect.left) * (W / rect.width);
      var ms = start + (px - L) / (W - L - R) * (end - start);
      var seen = nearest(times, ms, start), told = nearest(aheadTimes, ms, start);
      var forecast = told[1] < seen[1];
      var hit = forecast ? told : seen;
      if (hit[0] < 0 || hit[1] > (end - start) / 40) {
        cursor.setAttribute("visibility", "hidden"); tip.hidden = true; return;
      }
      var at = forecast ? aheadTimes[hit[0]] : times[hit[0]];
      var cx = X(at);
      cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx);
      cursor.setAttribute("visibility", "visible");
      clear(tip);
      tip.appendChild(el("div", { "class": "tip-time", text: fmtTime(at) }));
      if (forecast) {
        ahead[at].forEach(function (pair) {
          tip.appendChild(el("div", {}, [
            el("i", { "class": "swatch c-" + pair[0] }),
            document.createTextNode(curveLabel(pair[0], d.forecast) + " "),
            el("b", { text: mhz(pair[1]) + " " + t("mhz") })]));
        });
      } else {
        tip.appendChild(el("div", { text: t("muf") + " " + mhz(s.muf[hit[0]]) + " · " +
          t("lof") + " " + mhz(s.lof[hit[0]]) + " " + t("mhz") }));
      }
      tip.hidden = false;
      var left = cx * rect.width / W, wide = tip.offsetWidth || 190;
      tip.style.left = Math.min(Math.max(0, left - wide / 2), Math.max(0, rect.width - wide)) + "px";
      tip.style.top = "0px";
    }
    root.addEventListener("mousemove", readout);
    root.addEventListener("touchstart", readout, { passive: true });
    root.addEventListener("mouseleave", function () {
      cursor.setAttribute("visibility", "hidden"); tip.hidden = true;
    });

    box.appendChild(root);
    box.appendChild(tip);
    state.geo = { whole: whole, view: [start, end], W: W, L: L, R: R, root: root };
    renderZoom();
    renderLegend(all, d);
  }

  // --- scrolling and zooming ----------------------------------------------

  // The time the loaded series covers: its first sample to the furthest
  // forecast, and at least a day, so there is always somewhere to scroll.
  function extent(d, now) {
    var first = d.series.t.length ? parseT(d.series.t[0]) : now, last = now;
    curves(d, -Infinity, Infinity).forEach(function (c) {
      c.pts.forEach(function (p) {
        first = Math.min(first, p[0]);
        last = Math.max(last, p[0]);
      });
    });
    return [Math.min(first, last - 86400000), last];
  }

  // A view inside the series, no narrower than MIN_SPAN_MS.
  function clampView(view, whole) {
    var span = Math.min(Math.max(view[1] - view[0], MIN_SPAN_MS), whole[1] - whole[0]);
    var from = Math.min(Math.max(view[0], whole[0]), whole[1] - span);
    return [from, from + span];
  }

  var drawing = 0;
  function setView(view) {
    state.view = clampView(view, state.geo.whole);
    if (!drawing) {
      drawing = window.requestAnimationFrame(function () { drawing = 0; renderChart(); });
    }
  }

  // Zoom by `factor` (below one zooms in) keeping the instant `at` in place.
  function zoom(at, factor) {
    var v = state.view || state.geo.view;
    setView([at - (at - v[0]) * factor, at + (v[1] - at) * factor]);
  }

  function pan(ms) {
    var v = state.view || state.geo.view;
    setView([v[0] + ms, v[1] + ms]);
  }

  function resetView() {
    state.view = null;
    renderChart();
  }

  function renderZoom() {
    var seg = document.getElementById("zoom");
    clear(seg);
    var v = state.geo.view, middle = (v[0] + v[1]) / 2;
    [["−", "zoom_out", function () { zoom(middle, ZOOM_STEP); }],
     ["+", "zoom_in", function () { zoom(middle, 1 / ZOOM_STEP); }],
     [t("zoom_reset"), "zoom_reset", resetView]].forEach(function (spec) {
      var b = el("button", { type: "button", text: spec[0], title: t(spec[1]),
                             "aria-label": t(spec[1]) });
      if (spec[1] === "zoom_reset" && !state.view) b.disabled = true;
      b.addEventListener("click", spec[2]);
      seg.appendChild(b);
    });
  }

  // Gestures land on the chart's box, which outlives every redraw, so a drag
  // keeps its pointer while the SVG under it is replaced.
  function bindChart(box) {
    var pointers = {};
    function ready() { return state.geo && box.contains(state.geo.root); }
    function msPerPx() {
      var g = state.geo, wide = g.root.getBoundingClientRect().width;
      return (g.view[1] - g.view[0]) / ((g.W - g.L - g.R) * wide / g.W);
    }
    function msAt(clientX) {
      var g = state.geo, rect = g.root.getBoundingClientRect();
      return g.view[0] + (clientX - rect.left - g.L * rect.width / g.W) * msPerPx();
    }
    box.addEventListener("wheel", function (evt) {
      if (!ready()) return;
      evt.preventDefault();
      if (Math.abs(evt.deltaX) > Math.abs(evt.deltaY)) pan(evt.deltaX * msPerPx());
      else if (evt.deltaY) zoom(msAt(evt.clientX), evt.deltaY > 0 ? 1.25 : 0.8);
    }, { passive: false });
    box.addEventListener("pointerdown", function (evt) {
      if (!ready() || evt.button > 0 || !state.geo.root.contains(evt.target)) return;
      pointers[evt.pointerId] = evt.clientX;
      box.setPointerCapture(evt.pointerId);
      box.classList.add("dragging");
    });
    box.addEventListener("pointermove", function (evt) {
      if (!(evt.pointerId in pointers) || !ready()) return;
      var ids = Object.keys(pointers);
      if (ids.length === 1) {
        pan((pointers[evt.pointerId] - evt.clientX) * msPerPx());
      } else if (ids.length === 2) {
        var other = pointers[ids[0] === String(evt.pointerId) ? ids[1] : ids[0]];
        var before = Math.abs(pointers[evt.pointerId] - other);
        var after = Math.abs(evt.clientX - other);
        if (before > 10 && after > 10) zoom(msAt((evt.clientX + other) / 2), before / after);
      }
      pointers[evt.pointerId] = evt.clientX;
    });
    function lift(evt) {
      delete pointers[evt.pointerId];
      if (!Object.keys(pointers).length) box.classList.remove("dragging");
    }
    box.addEventListener("pointerup", lift);
    box.addEventListener("pointercancel", lift);
    box.addEventListener("dblclick", resetView);
  }

  function renderLegend(all, d) {
    var legend = document.getElementById("legend");
    clear(legend);
    function key(cls, label) {
      return el("span", {}, [el("i", { "class": cls }), document.createTextNode(label)]);
    }
    legend.appendChild(key("k-muf", t("muf")));
    legend.appendChild(key("k-lim", t("muf") + ", " + t("limited")));
    legend.appendChild(key("k-lof", t("lof")));
    if (!all.length) return;
    var group = el("div", { "class": "curves", role: "group", "aria-label": t("curves") });
    group.appendChild(el("span", { "class": "curves-title", text: t("curves") + ":" }));
    all.forEach(function (c) {
      var on = !state.hidden[c.key];
      var label = curveLabel(c.key, d.forecast) + (c.key === "model" ? " ±σ" : "");
      var b = el("button", { type: "button", "aria-pressed": String(on) }, [
        el("i", { "class": "swatch c-" + c.key }), document.createTextNode(label)]);
      b.addEventListener("click", function () {
        state.hidden[c.key] = on;
        remember("hidden", JSON.stringify(state.hidden));
        renderChart();
      });
      group.appendChild(b);
    });
    legend.appendChild(group);
  }

  function renderRanges() {
    var seg = document.getElementById("ranges");
    clear(seg);
    RANGES.forEach(function (days) {
      var b = el("button", { type: "button", text: t("range_" + days),
                             "aria-pressed": String(days === state.range) });
      b.addEventListener("click", function () {
        state.range = days;
        state.view = null;
        remember("range", String(days));
        renderRanges();
        renderChart();
      });
      seg.appendChild(b);
    });
  }

  // --- forecast table ------------------------------------------------------

  // One table per horizon: a row per method that has a score there, the
  // model first, the best value of each column marked.
  function scoreTable(f, h) {
    var rows = f.scores.filter(function (row) { return row.mae && row.mae[h] != null; });
    if (!rows.length) return null;
    function best(metric, pick) {
      var values = rows.map(function (row) { return row[metric] && row[metric][h]; })
        .filter(function (v) { return v != null; });
      return values.length > 1 ? pick.apply(null, values) : null;
    }
    var leastError = best("mae", Math.min), mostExplained = best("r2", Math.max);
    function number(value, top) {
      return el("td", value != null && value === top ? { "class": "best", text: value.toFixed(2) }
                                                      : { text: value == null ? "—" : value.toFixed(2) });
    }
    var body = el("tbody");
    rows.forEach(function (row) {
      var model = row.kind === "model";
      var key = model ? "model" : row.name;
      var about = model ? t("m_model_about") : t("m_" + row.name + "_about", { h: t("h" + h) });
      var skill = row.skill && row.skill[h];
      if (model && skill != null) {
        about += " · " + t(skill >= 0 ? "better" : "worse", { n: Math.abs(Math.round(skill * 100)) });
      }
      body.appendChild(el("tr", model ? { "class": "live" } : {}, [
        el("td", {}, [
          el("i", { "class": "swatch c-" + key }),
          el("b", { text: curveLabel(key, f) }),
          el("small", { text: about })]),
        number(row.mae[h], leastError),
        number(row.r2 ? row.r2[h] : null, mostExplained),
        el("td", { text: row.n && row.n[h] != null ? String(row.n[h]) : "—" })]));
    });
    var head = el("tr", {}, [
      el("th", { text: t("method") }), el("th", { text: t("mae") }),
      el("th", { text: t("r2") }), el("th", { text: t("pairs") })]);
    return el("div", { "class": "table-wrap" }, [
      el("table", { "class": "scores" }, [
        el("caption", { text: t("lead", { h: t("h" + h) }) }),
        el("thead", {}, [head]), body])]);
  }

  function renderForecast() {
    var box = document.getElementById("forecast");
    clear(box);
    var d = state.data[state.slug];
    var f = d && d.forecast;
    if (!f || !f.model) {
      box.appendChild(el("p", { "class": "muted", text: t("no_forecast") }));
      return;
    }
    var line = t("model") + ": " + f.model.name;
    if (f.model.issued_at) line += " · " + t("issued") + " " + fmtTime(parseT(f.model.issued_at));
    if (f.model.trained_from && f.model.trained_to) {
      line += " · " + t("trained") + " " + fmtDay(parseT(f.model.trained_from)) +
        "–" + fmtDay(parseT(f.model.trained_to));
    }
    box.appendChild(el("p", { "class": "model-line", text: line }));
    var tables = f.horizons.map(function (h) { return scoreTable(f, h); })
      .filter(function (table) { return table; });
    if (!tables.length) return;
    box.appendChild(el("p", { "class": "note", text: t("scores_intro") }));
    tables.forEach(function (table) { box.appendChild(table); });
    box.appendChild(el("ul", { "class": "note how" }, ["mae_about", "r2_about", "pairs_about"]
      .map(function (key) { return el("li", { text: t(key) }); })));
  }

  // --- flow ----------------------------------------------------------------

  function renderCircuit() {
    var c = circuit();
    var section = document.getElementById("circuit");
    if (!c) { section.hidden = true; return; }
    section.hidden = false;
    renderPath(c);
    renderImages(c);
    renderRanges();
    renderChart();
    renderForecast();
  }

  // The sounding in progress, from now.json (services/publish/now.py).
  function renderNow() {
    var now = state.now;
    var line = document.getElementById("now-state");
    var list = document.getElementById("now-list");
    clear(list);
    if (!now) { line.textContent = t("now_unavailable"); return; }
    var age = (Date.now() - parseT(now.generated_at)) / 1000;
    if (age > NOW_STALE_S) { line.textContent = t("now_delayed", { age: ago(age) }); return; }
    if (!now.circuits.length) { line.textContent = t("now_idle"); return; }
    line.textContent = "";
    now.circuits.forEach(function (c) {
      var pct = Math.round(c.progress * 100);
      var bar = el("progress", { max: "100", value: String(pct) });
      list.appendChild(el("figure", { "class": "latest now-item" }, [
        el("img", { src: c.img, alt: t("now_alt", { tx: c.tx_name, rx: c.rx_name }) }),
        el("figcaption", {}, [
          el("b", { text: c.tx_name + " → " + c.rx_name }),
          el("span", { text: " · " + t("now_line", {
            f: c.f_now_mhz.toFixed(2), lo: c.f_lo_mhz.toFixed(1),
            hi: c.f_hi_mhz.toFixed(1), p: pct }) }),
          el("br"),
          el("span", { "class": "muted", text: t("now_started", { time: fmtTime(parseT(c.started)) }) }),
          bar
        ])
      ]));
    });
  }

  // The public host's own word on its link to the hub (link.json, written by
  // its pull after every pass). Absent where the hub serves the page itself.
  function renderLink() {
    var link = state.link;
    var line = document.getElementById("link");
    line.hidden = !link;
    if (!link) return;
    var checked = (Date.now() - parseT(link.checked_at)) / 1000;
    var bad = !link.ok || checked > NOW_STALE_S;
    line.className = "link-state " + (bad ? "down" : "up");
    if (checked > NOW_STALE_S) {
      line.textContent = t("link_stale", { age: ago(checked) });
    } else if (link.ok) {
      line.textContent = t("link_ok");
    } else {
      line.textContent = t("link_lost", {
        reason: t("link_" + link.reason),
        since: link.since ? fmtTime(parseT(link.since), false) : "?" });
    }
  }

  function refreshLink() {
    return getJSON("link.json").then(function (link) { state.link = link; })
      .catch(function () { state.link = null; })
      .then(renderLink);
  }

  function refreshNow() {
    return getJSON("now.json").then(function (now) { state.now = now; })
      .catch(function () { state.now = null; })
      .then(renderNow);
  }

  function renderAll() {
    renderStatic();
    renderHeader();
    renderNow();
    renderLink();
    renderTabs();
    renderCircuit();
  }

  function select(slug, force) {
    if (slug === state.slug && !force && state.data[slug]) return;
    if (slug !== state.slug) state.view = null;
    state.slug = slug;
    if (window.location.hash !== "#" + slug) history.replaceState(null, "", "#" + slug);
    renderTabs();
    var c = circuit();
    if (!c) { renderCircuit(); return; }
    getJSON(c.data).then(function (d) {
      state.data[slug] = { series: d.series, forecast: d.forecast, path_km: d.series.path_km };
      if (slug === state.slug) renderCircuit();
    }).catch(function () {
      state.data[slug] = null;
      renderCircuit();
    });
  }

  function refresh() {
    return getJSON("status.json").then(function (st) {
      var changed = !state.status || st.generated_at !== state.status.generated_at;
      state.status = st;
      var slugs = st.circuits.map(function (c) { return c.slug; });
      var wanted = window.location.hash.slice(1);
      var slug = slugs.indexOf(wanted) >= 0 ? wanted
        : slugs.indexOf(state.slug) >= 0 ? state.slug : slugs[0];
      renderHeader();
      if (changed) { state.data = {}; select(slug, true); } else renderTabs();
    }).catch(function () {
      renderHeader();
    });
  }

  function start() {
    var saved = recall("lang");
    state.lang = saved === "ru" || saved === "en" ? saved
      : (navigator.language || "").toLowerCase().indexOf("ru") === 0 ? "ru" : "en";
    var r = parseInt(recall("range"), 10);
    try { state.hidden = JSON.parse(recall("hidden")) || {}; } catch (e) { state.hidden = {}; }
    if (RANGES.indexOf(r) >= 0) state.range = r;
    var theme = recall("theme");
    if (THEMES.indexOf(theme) >= 0) state.theme = theme;
    applyTheme();
    bindChart(document.getElementById("chart"));
    document.getElementById("theme").addEventListener("click", function () {
      state.theme = THEMES[(THEMES.indexOf(state.theme) + 1) % THEMES.length];
      remember("theme", state.theme);
      applyTheme();
    });

    document.getElementById("lang").addEventListener("click", function () {
      state.lang = state.lang === "ru" ? "en" : "ru";
      remember("lang", state.lang);
      renderAll();
    });
    window.addEventListener("hashchange", function () {
      var slug = window.location.hash.slice(1);
      if (slug && slug !== state.slug) select(slug);
    });
    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(renderChart, 150);
    });

    getJSON("strings.json").then(function (s) { state.strings = s; })
      .catch(function () { state.strings = {}; })
      .then(function () { renderStatic(); renderNow(); return refresh(); });
    setInterval(refresh, REFRESH_MS);
    refreshNow();
    refreshLink();
    setInterval(function () { refreshNow(); refreshLink(); }, NOW_MS);
  }

  start();
}());
