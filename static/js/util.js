// Shared helpers: colormaps, number/time formatting, binary step parsing.

const NEG = [0.18, 0.62, 1.0];   // negative values: electric blue
const POS = [1.0, 0.44, 0.16];   // positive values: amber
const BASE = [0.025, 0.03, 0.06];

// Robust per-vector scale: 98th percentile of |v| (sampled for long vectors).
export function robustScale(vals) {
  const n = vals.length;
  if (!n) return 1;
  const stride = Math.max(1, Math.floor(n / 2048));
  const a = [];
  for (let i = 0; i < n; i += stride) a.push(Math.abs(vals[i]));
  a.sort((x, y) => x - y);
  return a[Math.min(a.length - 1, Math.floor(a.length * 0.98))] || 1e-6;
}

export function divColor(t, out, o) {
  const m = Math.min(1, Math.abs(t));
  const g = Math.pow(m, 0.62);
  const c = t < 0 ? NEG : POS;
  let r = BASE[0] + (c[0] - BASE[0]) * g, gg = BASE[1] + (c[1] - BASE[1]) * g, b = BASE[2] + (c[2] - BASE[2]) * g;
  if (m > 0.85) { const w = (m - 0.85) / 0.15 * 0.55; r += (1 - r) * w; gg += (1 - gg) * w; b += (1 - b) * w; }
  out[o] = r * 255; out[o + 1] = gg * 255; out[o + 2] = b * 255; out[o + 3] = 255;
}

export function seqColor(t, out, o) {
  const m = Math.pow(Math.min(1, Math.max(0, t)), 0.5);
  const r = 0.02 + m * (m > 0.7 ? 0.95 : 0.15), g = 0.03 + m * 0.92, b = 0.07 + m * 0.85;
  out[o] = Math.min(1, r) * 255; out[o + 1] = Math.min(1, g) * 255; out[o + 2] = Math.min(1, b) * 255; out[o + 3] = 255;
}

// Fill an RGBA buffer of width w from vals (downsampling by max-|v| when vals is longer).
export function fillRGBA(out, w, vals, kind, scale) {
  const n = vals.length;
  if (kind === 'logit') {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { const v = vals[i]; if (v > -1e8) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    for (let x = 0; x < w; x++) {
      const v = vals[Math.floor(x * n / w)];
      seqColor(v < -1e8 ? 0 : Math.pow((v - lo) / (hi - lo || 1), 3), out, x * 4);
    }
    return;
  }
  const s = scale || robustScale(vals);
  for (let x = 0; x < w; x++) {
    let v;
    if (n <= w) v = vals[x];
    else {
      const a = Math.floor(x * n / w), b = Math.max(a + 1, Math.floor((x + 1) * n / w));
      v = vals[a];
      for (let i = a + 1; i < b; i++) if (Math.abs(vals[i]) > Math.abs(v)) v = vals[i];
    }
    if (kind === 'seq') seqColor(v / s, out, x * 4);
    else divColor(v / s, out, x * 4);
  }
}

export function fmtNum(v, sig = 5) {
  if (v === undefined || v === null || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-4) return v.toExponential(sig - 2);
  return v.toPrecision(sig).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}

export function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }

export function fmtBig(n) {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(Math.round(n));
}

export function fmtTime(sec) {
  if (!isFinite(sec)) return '∞';
  const a = Math.abs(sec);
  if (a < 1e-6) return (sec * 1e9).toFixed(1) + ' ns';
  if (a < 1e-3) return (sec * 1e6).toFixed(1) + ' µs';
  if (a < 1) return (sec * 1e3).toFixed(2) + ' ms';
  if (a < 120) return sec.toFixed(2) + ' s';
  if (a < 7200) return (sec / 60).toFixed(1) + ' min';
  if (a < 172800) return (sec / 3600).toFixed(1) + ' h';
  if (a < 3.15e7 * 2) return (sec / 86400).toFixed(1) + ' days';
  return (sec / 3.15e7).toFixed(1) + ' years';
}

export function showTok(s) {
  return s.replace(/\n/g, '↵').replace(/ /g, '·').replace(/\t/g, '⇥') || '∅';
}

// A step recorded on the GPU: [u32 header length][JSON header][float32 tensors]
export class StepData {
  constructor(buf) {
    const hl = new DataView(buf).getUint32(0, true);
    this.meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    this.f32 = new Float32Array(buf, 4 + hl);
    this.step = this.meta.step;
  }
  get(name) {
    const e = this.meta.index[name];
    return e ? this.f32.subarray(e[0], e[0] + e[1]) : null;
  }
}
