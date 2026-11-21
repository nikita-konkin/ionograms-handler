/* The archive's echo windows for a range, in one download (#162).
 *
 * The page stays a folder of static files, so a range is put together in the
 * browser: this picks the windows a range and a step hold, and writes them
 * into a ZIP that stores each one byte for byte, uncompressed, since a window
 * barely compresses. The ZIP is the plain format, without its 64-bit
 * extension, so it refuses to grow past 4 GB or 65,535 windows; the page
 * offers the list of addresses for anything larger. tests/test_publish_bundle.py
 * runs it under Node and reads its ZIP with Python's zipfile.
 */
var bundle = (function () {
  "use strict";

  var LARGEST = 0xFFFFFFFF;
  var MOST = 0xFFFF;
  var table = null;

  /* The soundings with a window from `from` to `to`, both in milliseconds and
   * both included: every one when `step` is 0, otherwise the first at or
   * after each step of `step` seconds counted from `from`. */
  function pick(soundings, from, to, step) {
    var kept = [], next = from, span = step * 1000;
    soundings.filter(function (e) { return e.file; })
      .map(function (e) { return [Date.parse(e.t), e]; })
      .sort(function (a, b) { return a[0] - b[0]; })
      .forEach(function (pair) {
        var at = pair[0];
        if (at < from || at > to || at < next) return;
        kept.push(pair[1]);
        next = span ? from + (Math.floor((at - from) / span) + 1) * span : at;
      });
    return kept;
  }

  function crc(bytes) {
    if (!table) {
      table = new Uint32Array(256);
      for (var n = 0; n < 256; n++) {
        var c = n;
        for (var k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c >>> 0;
      }
    }
    var value = 0xFFFFFFFF;
    for (var i = 0; i < bytes.length; i++) value = table[(value ^ bytes[i]) & 0xFF] ^ (value >>> 8);
    return (value ^ 0xFFFFFFFF) >>> 0;
  }

  /* MS-DOS time and date, which is all a plain ZIP entry can carry. */
  function dos(ms) {
    var d = new Date(ms);
    return [(d.getUTCHours() << 11) | (d.getUTCMinutes() << 5) | (d.getUTCSeconds() >> 1),
            ((d.getUTCFullYear() - 1980) << 9) | ((d.getUTCMonth() + 1) << 5) | d.getUTCDate()];
  }

  function record(fields) {
    var size = fields.reduce(function (sum, f) { return sum + f[1]; }, 0);
    var out = new Uint8Array(size), view = new DataView(out.buffer), at = 0;
    fields.forEach(function (f) {
      if (f[1] === 2) view.setUint16(at, f[0], true); else view.setUint32(at, f[0], true);
      at += f[1];
    });
    return out;
  }

  /* A ZIP written to `sink` one piece at a time, as `add` is called; `sink`
   * takes a Uint8Array and may return a promise, which is waited for. */
  function zip(sink) {
    var offset = 0, central = [], chain = Promise.resolve();
    function put(bytes) {
      chain = chain.then(function () { return sink(bytes); });
      offset += bytes.length;
      return chain;
    }
    function add(name, bytes, when) {
      var title = new TextEncoder().encode(name), stamp = dos(when), sum = crc(bytes);
      if (central.length >= MOST || offset + 30 + title.length + bytes.length > LARGEST) {
        return Promise.reject(new Error("the ZIP would pass the plain format's limits"));
      }
      var common = [[20, 2], [0x0800, 2], [0, 2], [stamp[0], 2], [stamp[1], 2],
                    [sum, 4], [bytes.length, 4], [bytes.length, 4], [title.length, 2], [0, 2]];
      central.push({ common: common, title: title, at: offset });
      put(record([[0x04034b50, 4]].concat(common)));
      put(title);
      return put(bytes);
    }
    function close() {
      var start = offset;
      var size = central.reduce(function (sum, e) { return sum + 46 + e.title.length; }, 22);
      if (start + size > LARGEST) {
        return Promise.reject(new Error("the ZIP would pass the plain format's limits"));
      }
      central.forEach(function (entry) {
        put(record([[0x02014b50, 4], [20, 2]].concat(entry.common,
          [[0, 2], [0, 2], [0, 2], [0, 4], [entry.at, 4]])));
        put(entry.title);
      });
      return put(record([[0x06054b50, 4], [0, 2], [0, 2], [central.length, 2],
                         [central.length, 2], [offset - start, 4], [start, 4], [0, 2]]));
    }
    return { add: add, close: close };
  }

  return { pick: pick, zip: zip, LARGEST: LARGEST, MOST: MOST };
})();
