// Dot-product microscope: replays one output number of a matmul, one multiply-add at a time,
// using the real weight row from the model and the real input vector captured on the GPU.
import { MAT_IO } from './ops.js';
import { fmtNum, fmtInt, showTok } from './util.js';

const RATES = [1, 4, 20, 100, 1000, 20000];

export class Micro {
  constructor(el, info, viz) {
    this.el = el; this.info = info; this.viz = viz;
    this.rate = 4;
    el.innerHTML = `
      <div class="mhead"><span class="mtitle"></span><button class="mclose" title="Close">✕</button></div>
      <div class="mformula"></div>
      <canvas class="mcanvas" width="1120" height="220"></canvas>
      <div class="mrow"><div class="mmacs"></div><div class="msum"></div></div>
      <div class="mctrl">
        <span class="mmode"></span>
        <span class="mrates">${RATES.map(r => `<button data-r="${r}">${r >= 1000 ? r / 1000 + 'k' : r}/s</button>`).join('')}<button data-r="inf">all</button></span>
        <button class="mreplay">↺ replay</button>
      </div>
      <div class="mresult"></div>`;
    this.$ = s => el.querySelector(s);
    this.$('.mclose').onclick = () => this.close();
    this.$('.mreplay').onclick = () => { if (this.w) { this.i = 0; this.sync = false; this.render(true); } };
    el.querySelectorAll('.mrates button').forEach(b => b.onclick = () => {
      this.rate = b.dataset.r === 'inf' ? Infinity : +b.dataset.r;
      this.sync = false;
      this.paintRates();
    });
    this.paintRates();
    this.ctx = this.$('.mcanvas').getContext('2d');
    this.acc = 0;
    this.key = null;
  }

  paintRates() {
    this.el.querySelectorAll('.mrates button').forEach(b =>
      b.classList.toggle('on', !this.sync && ((b.dataset.r === 'inf' && this.rate === Infinity) || +b.dataset.r === this.rate)));
  }

  get isOpen() { return !this.el.classList.contains('hidden'); }

  async open({ layer, mat, j, sd, sync = false }) {
    const key = `${sd.step}:${layer}:${mat}:${j}`;
    this.sync = sync;
    this.el.classList.remove('hidden');
    this.paintRates();
    if (key === this.key) return;
    this.key = key;
    const info = this.info, meta = sd.meta;
    const KV = info.n_kv_heads * info.head_dim;
    let x, gpu, label;
    if (mat === 'lm_head') {
      x = sd.get('final_norm');
      const t = meta.topk.find(t => t.id === j) || meta.sampled;
      j = t.id; gpu = t.logit; label = `logit of token ${JSON.stringify(showTok(t.text))} (#${j})`;
    } else {
      const [inName, outName] = MAT_IO[mat];
      x = sd.get(`${layer}.${inName}`);
      const out = sd.get(`${layer}.${outName}`);
      gpu = (mat === 'k' || mat === 'v') ? out[(meta.n_in - 1) * KV + j] : out[j];
      label = `${mat === 'down' ? 'MLP out' : mat} [${j}]`;
    }
    const W = { q: 'W_Q', k: 'W_K', v: 'W_V', o: 'W_O', gate: 'W_gate', up: 'W_up', down: 'W_down', lm_head: 'E (LM head)' }[mat];
    this.$('.mtitle').innerHTML = `🔬 Dot product · ${layer >= 0 ? 'Layer ' + layer + ' · ' : ''}<b>${W}</b> row ${fmtInt(j)} · token step ${sd.step}`;
    this.$('.mformula').textContent = `${label} = Σᵢ ${W}[${j}, i] · x[i]${info.has_bias[mat] ? ' + b' : ''}     i = 0 … ${x.length - 1}`;
    this.$('.mresult').textContent = 'fetching real weights from the model…';
    const buf = await fetch(`/api/weight?mat=${mat}&row=${j}&layer=${layer}`).then(r => r.arrayBuffer());
    if (this.key !== key) return;
    const f = new Float32Array(buf);
    this.b = Number.isNaN(f[0]) ? null : f[0];
    this.w = f.subarray(1);
    this.x = x; this.gpu = gpu; this.layer = layer; this.mat = mat; this.j = j;
    const n = x.length;
    this.cum = new Float64Array(n + 1);
    this.prod = new Float64Array(n);
    for (let i = 0; i < n; i++) { this.prod[i] = this.w[i] * x[i]; this.cum[i + 1] = this.cum[i] + this.prod[i]; }
    let lo = 0, hi = 0, pm = 0;
    for (let i = 0; i <= n; i++) { lo = Math.min(lo, this.cum[i]); hi = Math.max(hi, this.cum[i]); }
    for (let i = 0; i < n; i++) pm = Math.max(pm, Math.abs(this.prod[i]));
    const fin = this.cum[n] + (this.b ?? 0);
    lo = Math.min(lo, fin, gpu); hi = Math.max(hi, fin, gpu);
    this.range = [lo, hi, pm];
    this.i = sync ? 0 : 0;
    this.render(true);
  }

  close() { this.el.classList.add('hidden'); this.key = null; this.w = null; this.viz.setMicro(null, null); }

  syncTo(frac) { if (this.w) { this.i = Math.floor(frac * this.x.length); this.render(); } }

  tick(dt) {
    if (!this.w || this.sync || !this.isOpen) return;
    const n = this.x.length;
    if (this.i >= n) return;
    if (this.rate === Infinity) this.i = n;
    else { this.acc += dt * this.rate; const k = Math.floor(this.acc); this.acc -= k; this.i = Math.min(n, this.i + k); }
    this.render();
  }

  render(force) {
    if (!this.w) return;
    const n = this.x.length, i = this.i;
    if (!force && i === this.lastI) return;
    this.lastI = i;
    this.viz.setMicro(this.layer, this.mat, this.j, i < n ? (i + 0.5) / n : null);
    const c = this.ctx, Wd = c.canvas.width, Hd = c.canvas.height;
    const [lo, hi, pm] = this.range;
    c.clearRect(0, 0, Wd, Hd);
    const top = 14, mid = 130, sy = v => top + (hi - v) / (hi - lo || 1) * (mid - top - 8);
    // GPU result line
    c.strokeStyle = 'rgba(122,229,130,0.7)'; c.setLineDash([6, 6]); c.beginPath(); c.moveTo(0, sy(this.gpu)); c.lineTo(Wd, sy(this.gpu)); c.stroke(); c.setLineDash([]);
    c.strokeStyle = 'rgba(255,255,255,0.12)'; c.beginPath(); c.moveTo(0, sy(0)); c.lineTo(Wd, sy(0)); c.stroke();
    // running sum
    c.strokeStyle = '#4cc9f0'; c.lineWidth = 2; c.beginPath();
    for (let k = 0; k <= i; k += Math.max(1, Math.floor(n / Wd))) { const px = k / n * Wd; k ? c.lineTo(px, sy(this.cum[k])) : c.moveTo(px, sy(this.cum[k])); }
    c.lineTo(i / n * Wd, sy(this.cum[i])); c.stroke(); c.lineWidth = 1;
    // products
    const base = mid + 45;
    for (let k = 0; k < i; k++) {
      const v = this.prod[k] / (pm || 1), px = k / n * Wd;
      c.fillStyle = v < 0 ? '#2e9eff' : '#ff7029';
      c.fillRect(px, v < 0 ? base : base - v * 40, Math.max(1, Wd / n), Math.abs(v) * 40);
    }
    if (i < n) { c.fillStyle = '#fff'; c.fillRect(i / n * Wd - 1, top, 2, Hd - top); }
    c.fillStyle = '#7f9cc4'; c.font = '20px Menlo, monospace';
    c.fillText('running sum', 6, top + 8); c.fillText('each product W·x', 6, base - 46);
    c.fillStyle = 'rgba(122,229,130,0.9)'; c.fillText('GPU result', Wd - 140, sy(this.gpu) - 6);

    const lines = [];
    for (let k = Math.max(0, i - 6); k < Math.min(n, i + 1); k++) {
      const cur = k === i;
      lines.push(`<div class="${cur ? 'cur' : ''}"><span>i=${k}</span> ${fmtNum(this.w[k], 5)} × ${fmtNum(this.x[k], 5)} = <b>${fmtNum(this.prod[k], 5)}</b> <span>Σ ${fmtNum(this.cum[k + 1], 7)}</span></div>`);
    }
    this.$('.mmacs').innerHTML = lines.join('');
    this.$('.msum').innerHTML = `<div class="big">${fmtNum(this.cum[i], 8)}</div><div>${fmtInt(i)} / ${fmtInt(n)} multiply-adds</div>`;
    this.$('.mmode').textContent = this.sync ? '⏱ synchronized to the playback clock' : 'speed:';
    if (i >= n) {
      const ours = this.cum[n] + (this.b ?? 0);
      const d = Math.abs(ours - this.gpu);
      const order = [...this.prod.keys()].sort((a, b) => Math.abs(this.prod[b]) - Math.abs(this.prod[a])).slice(0, 4);
      this.$('.mresult').innerHTML = `${this.b !== null ? `Σ ${fmtNum(this.cum[n], 8)} + bias ${fmtNum(this.b, 6)} = ` : ''}<b>${fmtNum(ours, 8)}</b> (float64 replay) &nbsp;vs&nbsp; <b>${fmtNum(this.gpu, 8)}</b> computed by the GPU
        <span class="${d < 1e-3 * Math.max(1, Math.abs(this.gpu)) ? 'ok' : 'bad'}">Δ ${d.toExponential(1)}${d < 1e-3 * Math.max(1, Math.abs(this.gpu)) ? ' ✓ match (float32 rounding)' : ''}</span>
        <div class="dim">Biggest contributors: ${order.map(k => `i=${k} (${fmtNum(this.prod[k], 3)})`).join(', ')}</div>`;
    } else {
      this.$('.mresult').innerHTML = `<span class="dim">GPU result for comparison: ${fmtNum(this.gpu, 8)}</span>`;
    }
  }
}
