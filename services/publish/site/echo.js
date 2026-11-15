/* An echo window drawn in the browser: docs/echo-window.md, section 3.9.
 *
 * The window keeps the dechirped voltage of a sounding, decimated to the
 * echo's own band. This reads one with the vendored h5wasm and transforms the
 * stored samples as they are, so a reader chooses the row length and the
 * padding instead of taking the one picture the hub drew. It is the transform
 * of muf.echo_window.spectrogram, and tests/test_publish_echo.py holds the
 * two together.
 */
var echo = (function () {
  "use strict";

  var FLOOR = 1e-3;   // spectro.to_db: noise reads 30 dB, detection is 43 dB
  var serial = 0;

  function number(node, key) {
    if (!(key in node.attrs)) throw new Error("the window lacks " + key);
    return Number(node.attrs[key].value);
  }

  /* A published window, bytes to numbers. Refuses anything but a v1 window
   * that carries its row noise, which only a window cut from an .lfs does. */
  function read(h5wasm, bytes) {
    var name = "window-" + (serial += 1) + ".h5";
    h5wasm.FS.writeFile(name, bytes);
    var file = null;
    try {
      file = new h5wasm.File(name, "r");
      var group = file.get("echo_window");
      if (!group || number(group, "version") !== 1) {
        throw new Error("not an echo window of version 1");
      }
      if (group.keys().indexOf("row_noise") < 0) {
        throw new Error("the window carries no row noise");
      }
      var noise = group.get("row_noise");
      var pairs = group.get("z").value;
      var re = new Float64Array(pairs.length);
      var im = new Float64Array(pairs.length);
      for (var k = 0; k < pairs.length; k += 1) {
        re[k] = pairs[k][0];
        im[k] = pairs[k][1];
      }
      var found = { re: re, im: im, row_noise: Float64Array.from(noise.value),
                    noise_window: number(noise, "window") };
      ["decimation", "sr_hz", "shift_hz", "t_first_s", "km_per_hz",
       "range_zero_km", "range_lo_km", "range_hi_km", "source_samples",
       "sweep_hz_at_t_first", "rate_hz_s"].forEach(function (key) {
        found[key] = number(group, key);
      });
      return found;
    } finally {
      if (file) file.close();
      h5wasm.FS.unlink(name);
    }
  }

  /* Transform points for rows of `length` padded `pad` times their length
   * with zeros: the next power of two, which the transform below needs. */
  function points(length, pad) {
    var size = 2;
    while (size < length * (1 + pad)) size *= 2;
    return size;
  }

  /* The discrete Fourier transform of re + j·im, in place. */
  function transform(re, im) {
    var size = re.length, i, j = 0, bit, half, step, held;
    for (i = 1; i < size; i += 1) {
      bit = size >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        held = re[i]; re[i] = re[j]; re[j] = held;
        held = im[i]; im[i] = im[j]; im[j] = held;
      }
    }
    for (half = 1; half < size; half *= 2) {
      step = -Math.PI / half;
      for (i = 0; i < size; i += 2 * half) {
        for (j = 0; j < half; j += 1) {
          var c = Math.cos(step * j), s = Math.sin(step * j);
          var a = i + j, b = a + half;
          var xr = re[b] * c - im[b] * s, xi = re[b] * s + im[b] * c;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  /* The stored row noise at source sample `centre`, held flat past its ends. */
  function floorAt(window, centre) {
    var noise = window.row_noise, at = centre / window.noise_window - 0.5;
    if (at <= 0) return noise[0];
    if (at >= noise.length - 1) return noise[noise.length - 1];
    var low = Math.floor(at);
    return noise[low] + (at - low) * (noise[low + 1] - noise[low]);
  }

  /* Rows of `length` stored samples under a Hann window, on the shared noise
   * scale: power[row * bins + bin], ranges ascending and held to the window,
   * each row's sweep frequency at its centre. */
  function spectrogram(window, length, pad) {
    var span = length * window.decimation;
    var rows = Math.floor(window.source_samples / span);
    if (!(length >= 2) || !(pad >= 0) || rows < 1) {
      throw new Error("rows of " + length + " samples cannot be cut from this window");
    }
    var size = points(length, pad);
    var lead = Math.round(-window.t_first_s * window.sr_hz);
    var kept = [];
    for (var q = 0; q < size; q += 1) {
      var range = window.range_zero_km + window.km_per_hz * (
        (q < size / 2 ? q : q - size) * window.sr_hz / size + window.shift_hz);
      if (range >= window.range_lo_km && range <= window.range_hi_km) {
        kept.push([range, q]);
      }
    }
    kept.sort(function (a, b) { return a[0] - b[0]; });
    var bins = kept.length;
    var hann = new Float64Array(length);
    for (var k = 0; k < length; k += 1) {
      hann[k] = 0.5 * (1 - Math.cos(2 * Math.PI * k / length));
    }
    var power = new Float32Array(rows * bins);
    var sweep = new Float64Array(rows);
    var re = new Float64Array(size), im = new Float64Array(size);
    var gain = window.decimation * window.decimation;
    for (var i = 0; i < rows; i += 1) {
      re.fill(0);
      im.fill(0);
      for (k = 0; k < length; k += 1) {
        re[k] = hann[k] * window.re[lead + i * length + k];
        im[k] = hann[k] * window.im[lead + i * length + k];
      }
      transform(re, im);
      var floor = floorAt(window, (i + 0.5) * span) * span / window.noise_window;
      for (var j = 0; j < bins; j += 1) {
        q = kept[j][1];
        power[i * bins + j] = gain * (re[q] * re[q] + im[q] * im[q]) / floor;
      }
      sweep[i] = window.sweep_hz_at_t_first + window.rate_hz_s * (
        lead + (i + 0.5) * length) / window.sr_hz;
    }
    return { rows: rows, bins: bins, points: size, power: power, sweep: sweep,
             ranges: Float64Array.from(kept, function (pair) { return pair[0]; }) };
  }

  /* Power on the hub's decibel scale. */
  function decibels(power) {
    return 10 * Math.log10(Math.max(power, FLOOR) / FLOOR);
  }

  return { read: read, points: points, spectrogram: spectrogram,
           decibels: decibels };
})();
