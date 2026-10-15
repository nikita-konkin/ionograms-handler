/* The public results page. Reads status.json and data/{circuit}.json written
 * by services/publish/publisher.py; draws everything with plain DOM and SVG so
 * the site carries no third-party code and runs under `default-src 'self'`.
 * All text goes in through textContent, never innerHTML.
 */
(function () {
  "use strict";

  var STALE_S = 3600;            // older than this, the banner says so
  var REFRESH_MS = 5 * 60 * 1000;
  var RANGES = [1, 7, 30];
  var SVG = "http://www.w3.org/2000/svg";

  var state = {
    lang: "en", strings: {}, status: null, slug: null, range: 7,
    data: {}, shown: null
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

  function fmtTime(ms, withDate) {
    var d = new Date(ms);
    var hm = pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes());
    if (withDate === false) return hm;
    return d.getUTCFullYear() + "-" + pad(d.getUTCMonth() + 1) + "-" +
      pad(d.getUTCDate()) + " " + hm + " " + t("utc");
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

  var TICKS_H = [1, 3, 6, 12, 24, 48, 96, 168];

  function niceMax(v) { return Math.max(5, Math.ceil(v / 5) * 5); }

  function renderChart() {
    var box = document.getElementById("chart");
    clear(box);
    var d = state.data[state.slug];
    if (!d) return;
    var s = d.series;
    var now = parseT(state.status.generated_at);
    var start = now - state.range * 86400000;

    var muf = [], lof = [];
    for (var i = 0; i < s.t.length; i++) {
      var x = parseT(s.t[i]);
      if (x < start) continue;
      if (s.muf[i] != null) muf.push([x, s.muf[i], s.limited[i]]);
      if (s.lof[i] != null) lof.push([x, s.lof[i]]);
    }
    var fc = (s.forecast || []).filter(function (f) { return f.param === "muf"; })[0];
    var fpts = [];
    if (fc) {
      for (var j = 0; j < fc.t.length; j++) {
        var fx = parseT(fc.t[j]);
        if (fx >= start && fx <= now + state.range * 86400000) {
          fpts.push([fx, fc.value[j], fc.sigma[j]]);
        }
      }
    }
    if (!muf.length && !lof.length && !fpts.length) {
      box.appendChild(el("p", { "class": "muted", text: t("no_data") }));
      renderLegend(false);
      return;
    }

    var end = now;
    fpts.forEach(function (p) { if (p[1] != null && p[0] > end) end = p[0]; });
    var top = 0;
    muf.forEach(function (p) { top = Math.max(top, p[1]); });
    lof.forEach(function (p) { top = Math.max(top, p[1]); });
    fpts.forEach(function (p) { if (p[1] != null) top = Math.max(top, p[1] + (p[2] || 0)); });
    top = niceMax(top * 1.05);

    var W = Math.max(320, box.clientWidth || 640), H = W < 500 ? 240 : 300;
    var L = 36, R = 10, T = 10, B = 26;
    function X(ms) { return L + (ms - start) / (end - start) * (W - L - R); }
    function Y(v) { return T + (1 - v / top) * (H - T - B); }

    var root = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                            "aria-label": t("series") });

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
    if (end > now) {
      root.appendChild(svg("line", { "class": "now", x1: X(now), x2: X(now), y1: T, y2: H - B }));
    }

    // forecast band and line, broken wherever a value is missing
    if (fpts.length) {
      var runs = [], run = [];
      fpts.forEach(function (p) {
        if (p[1] == null) { if (run.length) runs.push(run); run = []; } else run.push(p);
      });
      if (run.length) runs.push(run);
      runs.forEach(function (r) {
        var withSigma = r.filter(function (p) { return p[2] != null; });
        if (withSigma.length > 1) {
          var up = withSigma.map(function (p) { return X(p[0]) + "," + Y(Math.min(top, p[1] + p[2])); });
          var down = withSigma.slice().reverse().map(function (p) {
            return X(p[0]) + "," + Y(Math.max(0, p[1] - p[2]));
          });
          root.appendChild(svg("polygon", { "class": "band", points: up.concat(down).join(" ") }));
        }
        root.appendChild(svg("polyline", { "class": "fc",
          points: r.map(function (p) { return X(p[0]) + "," + Y(p[1]); }).join(" ") }));
      });
    }

    lof.forEach(function (p) {
      root.appendChild(svg("circle", { "class": "lof", cx: X(p[0]), cy: Y(p[1]), r: 2 }));
    });
    muf.forEach(function (p) {
      root.appendChild(svg("circle", { "class": p[2] ? "muf-limited" : "muf",
                                       cx: X(p[0]), cy: Y(p[1]), r: p[2] ? 2.6 : 2.2 }));
    });

    // hover / touch readout
    var cursor = svg("line", { "class": "cursor", x1: 0, x2: 0, y1: T, y2: H - B, visibility: "hidden" });
    root.appendChild(cursor);
    var tip = el("div", { "class": "tip", hidden: "" });
    var times = s.t.map(parseT);
    function readout(evt) {
      var rect = root.getBoundingClientRect();
      var px = ((evt.touches ? evt.touches[0].clientX : evt.clientX) - rect.left) * (W / rect.width);
      var ms = start + (px - L) / (W - L - R) * (end - start);
      var best = -1, gap = Infinity;
      for (var n = 0; n < times.length; n++) {
        if (times[n] < start) continue;
        var g = Math.abs(times[n] - ms);
        if (g < gap) { gap = g; best = n; }
      }
      if (best < 0 || gap > (end - start) / 40) { cursor.setAttribute("visibility", "hidden"); tip.hidden = true; return; }
      var cx = X(times[best]);
      cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx);
      cursor.setAttribute("visibility", "visible");
      tip.textContent = fmtTime(times[best]) + " · " + t("muf") + " " + mhz(s.muf[best]) +
        " · " + t("lof") + " " + mhz(s.lof[best]) + " " + t("mhz");
      tip.hidden = false;
      var left = cx * rect.width / W;
      tip.style.left = Math.min(Math.max(0, left - 90), rect.width - 190) + "px";
      tip.style.top = "0px";
    }
    root.addEventListener("mousemove", readout);
    root.addEventListener("touchstart", readout, { passive: true });
    root.addEventListener("mouseleave", function () {
      cursor.setAttribute("visibility", "hidden"); tip.hidden = true;
    });

    box.appendChild(root);
    box.appendChild(tip);
    renderLegend(fpts.length > 0);
  }

  function renderLegend(withForecast) {
    var legend = document.getElementById("legend");
    clear(legend);
    function key(cls, label) {
      return el("span", {}, [el("i", { "class": cls }), document.createTextNode(label)]);
    }
    legend.appendChild(key("k-muf", t("muf")));
    legend.appendChild(key("k-lim", t("muf") + ", " + t("limited")));
    legend.appendChild(key("k-lof", t("lof")));
    if (withForecast) legend.appendChild(key("k-fc", t("forecast_band")));
  }

  function renderRanges() {
    var seg = document.getElementById("ranges");
    clear(seg);
    RANGES.forEach(function (days) {
      var b = el("button", { type: "button", text: t("range_" + days),
                             "aria-pressed": String(days === state.range) });
      b.addEventListener("click", function () {
        state.range = days;
        remember("range", String(days));
        renderRanges();
        renderChart();
      });
      seg.appendChild(b);
    });
  }

  // --- forecast table ------------------------------------------------------

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
    if (!f.scores.length) return;

    var horizons = f.horizons.filter(function (h) {
      return f.scores.some(function (row) { return row.mae[h] != null; });
    });
    var head = el("tr", {}, [el("th", { text: t("mae") })].concat(
      horizons.map(function (h) { return el("th", { text: t("h" + h) }); })));
    var body = el("tbody");
    f.scores.forEach(function (row) {
      var name = row.kind === "baseline" ? row.name + " (" + t("baseline") + ")" : row.name;
      var cells = [el("td", { text: name })];
      horizons.forEach(function (h) {
        var text = row.mae[h] == null ? "—" : row.mae[h].toFixed(2);
        var skill = row.skill && row.skill[h];
        if (row.kind === "model" && skill != null) {
          text += " (" + (skill >= 0 ? "+" : "") + Math.round(skill * 100) + "%)";
        }
        cells.push(el("td", { text: text }));
      });
      body.appendChild(el("tr", {}, cells));
    });
    box.appendChild(el("div", { "class": "table-wrap" }, [
      el("table", {}, [el("thead", {}, [head]), body])
    ]));
    box.appendChild(el("p", { "class": "note", text: "(%) — " + t("skill") }));
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

  function renderAll() {
    renderStatic();
    renderHeader();
    renderTabs();
    renderCircuit();
  }

  function select(slug, force) {
    if (slug === state.slug && !force && state.data[slug]) return;
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
    if (RANGES.indexOf(r) >= 0) state.range = r;

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
      .then(function () { renderStatic(); return refresh(); });
    setInterval(refresh, REFRESH_MS);
  }

  start();
}());
