/* The frequency response page (#103). Every change of a setting asks
   /soundings/{id}/response again -- the hub keeps the sounding in memory, so
   that is a measurement, not a read -- and redraws four plots: the ionogram the
   stretch is picked on, and the level, the phase and the peaks' delay along it.
   The bundle is plotly-basic, which has no heatmap: the ionogram is painted
   into a canvas and laid under the axes as an image, as the sounding page
   lays its raster. */
(function () {
  'use strict';
  const page = document.getElementById('frPage');
  if (!page) return;
  const id = page.dataset.id;
  const SPAN_MHZ = 1.0;
  const STOPS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const st = { mode: null, lo: null, sub: 20.48, overlap: 0.5, held: false, shift: 0, half: 0 };
  const $ = name => document.getElementById(name);
  const say = $('frSay');
  let data = null, busy = false, again = false, timer = 0, painted = null;

  const num = (v, d) => (v === null || v === undefined ? '-' : Number(v).toFixed(d));

  function query() {
    const q = new URLSearchParams({ sub: st.sub, overlap: st.overlap, held: st.held,
                                    shift: st.shift });
    if (st.mode) q.set('mode', st.mode);
    if (st.lo !== null) {
      q.set('lo', st.lo.toFixed(2));
      q.set('hi', (st.lo + SPAN_MHZ).toFixed(2));
    }
    if (st.half > 0) q.set('half', st.half);
    return q;
  }

  async function load() {
    if (busy) { again = true; return; }
    busy = true;
    say.textContent = T('response.js.loading');
    try {
      const r = await fetch('/soundings/' + id + '/response?' + query());
      const body = await r.json().catch(() => ({}));
      if (!r.ok) {
        const why = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || r.status);
        say.textContent = T('response.js.error', { why: why });
        return;
      }
      data = body;
      if (st.mode === null) st.mode = data.response.mode;
      if (st.lo === null) st.lo = data.response.f_lo_mhz;
      say.textContent = '';
      draw();
    } catch (error) {
      say.textContent = T('response.js.error', { why: String(error) });
    } finally {
      busy = false;
      if (again) { again = false; load(); }
    }
  }

  const later = () => { clearTimeout(timer); timer = setTimeout(load, 250); };

  function mark(box, value) {
    box.querySelectorAll('button').forEach(b => b.classList.toggle('ok', b.dataset.v === String(value)));
  }

  function segment(box, key, parse) {
    box.addEventListener('click', event => {
      const b = event.target.closest('button');
      if (!b) return;
      st[key] = parse(b.dataset.v);
      mark(box, b.dataset.v);
      load();
    });
  }

  function slider(input, key, show) {
    input.addEventListener('input', () => { st[key] = Number(input.value); show(); later(); });
  }

  /* ------------------------------------------------------------ controls */

  function modes() {
    const box = $('frModes');
    if (box.childElementCount) return;
    data.modes.forEach(m => {
      const b = document.createElement('button');
      b.className = 'pill';
      b.dataset.v = m.name;
      b.textContent = m.name;
      box.appendChild(b);
    });
    mark(box, st.mode);
    box.addEventListener('click', event => {
      const b = event.target.closest('button');
      if (!b) return;
      const m = data.modes.find(x => x.name === b.dataset.v);
      st.mode = m.name;
      st.lo = Math.max(data.record.f_lo_mhz, Math.min((m.f_lo_mhz + m.f_hi_mhz - SPAN_MHZ) / 2,
                                                      data.record.f_hi_mhz - SPAN_MHZ));
      mark(box, st.mode);
      load();
    });
  }

  function stretch() {
    const input = $('frLo');
    input.min = data.record.f_lo_mhz.toFixed(2);
    input.max = Math.max(data.record.f_lo_mhz, data.record.f_hi_mhz - SPAN_MHZ).toFixed(2);
    input.value = st.lo;
    showLo();
  }

  function showLo() {
    $('frLoText').textContent = T('response.js.card.stretch.v', { lo: num(st.lo, 2), hi: num(st.lo + SPAN_MHZ, 2) });
  }

  function showShift() {
    $('frShiftText').textContent = T('response.js.us', { v: (st.shift > 0 ? '+' : '') + st.shift });
  }

  function showHalf() {
    const auto = data ? data.response.gate.auto_us : null;
    $('frHalfText').textContent = st.half > 0 ? T('response.js.us', { v: st.half })
      : T('response.js.auto', { v: num(auto, 0) });
  }

  function move(lo) {
    const input = $('frLo');
    st.lo = Math.round(Math.max(Number(input.min), Math.min(Number(input.max), lo)) * 20) / 20;
    input.value = st.lo;
    showLo();
    load();
  }

  /* ------------------------------------------------------------ plots */

  function layout(x, y) {
    const ink = getComputedStyle(document.body).color;
    return {
      margin: { l: 62, r: 12, t: 22, b: 44 },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: ink, family: 'ui-monospace,Menlo,monospace', size: 11 },
      xaxis: { title: { text: x }, gridcolor: 'rgba(127,127,127,.25)', zeroline: false },
      yaxis: { title: { text: y }, gridcolor: 'rgba(127,127,127,.25)', zeroline: false },
      legend: { orientation: 'h', y: -0.25, x: 0, font: { size: 10 } },
      hovermode: 'closest',
    };
  }

  const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d'] };

  function colour(v) {
    const x = Math.max(0, Math.min(1, v)) * (STOPS.length - 1), j = Math.min(STOPS.length - 2, Math.floor(x)), u = x - j;
    return STOPS[j].map((c, k) => c * (1 - u) + STOPS[j + 1][k] * u);
  }

  function paint(map) {
    const bytes = Uint8Array.from(atob(map.map), c => c.charCodeAt(0));
    const canvas = document.createElement('canvas');
    canvas.width = map.cols;
    canvas.height = map.rows;
    const g = canvas.getContext('2d'), image = g.createImageData(map.cols, map.rows);
    for (let i = 0; i < map.cols; i++) {
      for (let j = 0; j < map.rows; j++) {
        const c = colour(bytes[i * map.rows + j] / 255), k = ((map.rows - 1 - j) * map.cols + i) * 4;
        image.data[k] = c[0]; image.data[k + 1] = c[1]; image.data[k + 2] = c[2]; image.data[k + 3] = 255;
      }
    }
    g.putImageData(image, 0, 0);
    return canvas.toDataURL();
  }

  function drawIonogram() {
    const m = data.ionogram, r = data.response;
    if (!painted || painted.source !== data.source) painted = { source: data.source, url: paint(m) };
    const curves = data.modes.map(x => ({
      x: x.curve.freq_mhz, y: x.curve.delay_us, type: 'scatter', mode: 'lines', name: x.name,
      line: { color: x.name === r.mode ? '#ffffff' : 'rgba(255,255,255,.5)',
              width: x.name === r.mode ? 2 : 1, dash: x.name === r.mode ? 'solid' : 'dot' },
      hoverinfo: 'name',
    }));
    const shapes = [{ type: 'rect', x0: r.f_lo_mhz, x1: r.f_hi_mhz, y0: m.delay_us[0], y1: m.delay_us[1],
                      line: { color: '#ff8a65', width: 2 } }];
    const notes = [];
    if (data.muf_mhz) {
      shapes.push({ type: 'line', x0: data.muf_mhz, x1: data.muf_mhz, yref: 'paper', y0: 0, y1: 1,
                    line: { color: '#ffd166', width: 1.4, dash: 'dash' } });
      notes.push({ x: data.muf_mhz, y: 1, yref: 'paper', yanchor: 'bottom', xanchor: 'right', showarrow: false,
                   text: T('response.js.muf', { f: num(data.muf_mhz, 2) }), font: { color: '#ffd166', size: 10 },
                   bgcolor: 'rgba(11,13,16,.85)', borderpad: 2 });
    }
    const box = layout(T('response.js.ax.f'), T('response.js.ax.delay'));
    box.xaxis.range = m.freq_mhz;
    box.yaxis.range = m.delay_us;
    box.plot_bgcolor = '#0b0d10';
    box.images = [{ source: painted.url, xref: 'x', yref: 'y', x: m.freq_mhz[0], y: m.delay_us[1],
                    sizex: m.freq_mhz[1] - m.freq_mhz[0], sizey: m.delay_us[1] - m.delay_us[0],
                    xanchor: 'left', yanchor: 'top', sizing: 'stretch', layer: 'below' }];
    box.shapes = shapes;
    box.annotations = notes;
    box.dragmode = false;
    box.showlegend = false;
    Plotly.react($('frIon'), curves, box, CONFIG);
  }

  function picked(series, key, flag) {
    return series[key].map((v, i) => (series[flag][i] ? v : null));
  }

  function drawPlots() {
    const r = data.response, s = r.sliding, p = r.stepped;
    const line = (y, name) => ({ x: s.freq_mhz, y: y, type: 'scatter', mode: 'lines', name: name,
                                 line: { width: 1.4, color: '#58a6ff' } });
    const dots = (y, name, colour, symbol) => ({ x: p.freq_mhz, y: y, type: 'scatter', mode: 'markers', name: name,
                                                 marker: { size: 7, color: colour, symbol: symbol } });
    Plotly.react($('frLevel'), [line(s.level_db, T('response.js.sliding')),
                                dots(p.level_db, T('response.js.stepped'), '#ff8a65', 'circle')],
                 layout(T('response.js.ax.f'), T('response.js.ax.level')), CONFIG);
    Plotly.react($('frPhase'), [line(s.phase_rad, T('response.js.sliding')),
                                dots(p.phase_rad, T('response.js.stepped'), '#ff8a65', 'circle'),
                                dots(picked(p, 'phase_rad', 'lost'), T('response.js.lost'), '#e5534b', 'x')],
                 layout(T('response.js.ax.f'), T('response.js.ax.phase')), CONFIG);
    const half = r.gate.half_us;
    const edge = sign => s.gate_us.map(v => (v === null ? null : v + sign * half));
    Plotly.react($('frDelay'), [
      { x: s.freq_mhz, y: edge(-1), type: 'scatter', mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: s.freq_mhz, y: edge(1), type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty',
        fillcolor: 'rgba(88,166,255,.15)', name: T('response.js.gate'), hoverinfo: 'skip' },
      { x: s.freq_mhz, y: s.gate_us, type: 'scatter', mode: 'lines', name: T('response.js.curve'),
        line: { width: 1, dash: 'dash', color: 'rgba(127,127,127,.9)' } },
      { x: s.freq_mhz, y: s.delay_us, type: 'scatter', mode: 'lines', name: T('response.js.peaks'),
        line: { width: 1.4, color: '#58a6ff' } },
    ], layout(T('response.js.ax.f'), T('response.js.ax.delay')), CONFIG);
  }

  function drawCards() {
    const r = data.response, g = r.gate, lost = r.stepped.lost.filter(Boolean).length;
    const rows = [
      [T('response.js.card.stretch'), T('response.js.card.stretch.v', { lo: num(r.f_lo_mhz, 2), hi: num(r.f_hi_mhz, 2) })
        + (data.muf_mhz ? ' · ' + T('response.js.card.rel', { lo: num(r.f_lo_mhz / data.muf_mhz, 2), hi: num(r.f_hi_mhz / data.muf_mhz, 2) }) : '')],
      [T('response.js.card.sub'), T('response.js.card.sub.v', { sub: num(r.subchannel_hz / 1e3, 2), n: r.window, bin: num(r.bin_us, 1), hop: r.hop })],
      [T('response.js.card.gate'), (g.held ? T('response.js.card.gate.held') : T('response.js.card.gate.curve'))
        + ' · ' + T('response.js.card.gate.v', { shift: num(g.shift_us, 0), half: num(g.half_us, 0), auto: num(g.auto_us, 0) })],
      [T('response.js.card.level'), T('response.js.card.level.v', { db: num(r.snr_db, 1) })],
      [T('response.js.card.gdd'), T('response.js.card.gdd.v', { phase: num(r.gdd_phase_us_per_mhz, 1), peaks: num(r.gdd_trace_us_per_mhz, 1) })],
      [T('response.js.card.phase'), T('response.js.card.phase.v', { j: r.sliding.jumps, n: r.sliding.steps, sj: r.stepped.jumps, sn: r.stepped.steps, lost: lost })],
    ];
    const table = $('frCards');
    table.replaceChildren(...rows.map(([label, value]) => {
      const tr = document.createElement('tr'), th = document.createElement('th'), td = document.createElement('td');
      th.textContent = label;
      td.textContent = value;
      tr.append(th, td);
      return tr;
    }));
  }

  function draw() {
    modes();
    stretch();
    showHalf();
    drawIonogram();
    drawPlots();
    drawCards();
  }

  /* ------------------------------------------------------------ wiring */

  segment($('frSub'), 'sub', Number);
  segment($('frOverlap'), 'overlap', Number);
  segment($('frGate'), 'held', v => v === 'true');
  slider($('frLo'), 'lo', showLo);
  slider($('frShift'), 'shift', showShift);
  slider($('frHalf'), 'half', showHalf);
  $('frIon').addEventListener('click', event => {
    const div = $('frIon'), full = div._fullLayout;
    if (!full || !data) return;
    const x = full.xaxis.p2d(event.clientX - div.getBoundingClientRect().left - full._size.l);
    if (Number.isFinite(x)) move(x - SPAN_MHZ / 2);
  });
  showShift();
  load();
})();
