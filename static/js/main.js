// App controller: talks to the GPU server, replays recorded steps on a scaled clock, drives HUD + 3D.
import { Viz, escapeHtml } from './scene.js';
import { Micro } from './micro.js';
import { opInfo, MATMUL, GROUP, GROUP_COLOR, matDims } from './ops.js';
import { StepData, fmtNum, fmtInt, fmtBig, fmtTime, showTok, robustScale } from './util.js';

const $ = s => document.querySelector(s);
const SPEEDS = [
  ['1', 1, '1× (real GPU time)'], ['0.1', 0.1, '1/10×'], ['0.01', 0.01, '1/100×'], ['0.001', 1e-3, '1/1,000×'],
  ['1e-4', 1e-4, '1/10,000×'], ['1e-5', 1e-5, '1/100,000×'], ['1e-6', 1e-6, '1/1,000,000×'],
  ['1e-7', 1e-7, '1/10,000,000×'], ['1e-9', 1e-9, '1/1,000,000,000×'], ['guided', 'guided', 'Guided (~2 s per op)'],
];
const SYSTEM = 'You are a helpful assistant. Answer briefly.';

const info = await fetch('/api/model').then(r => r.json());
const thumbs = new Uint8Array(await fetch('/api/thumbs').then(r => r.arrayBuffer()));
const viz = new Viz($('#stage'), info, thumbs);
const micro = new Micro($('#micro'), info, viz);

const S = {
  genId: 0, messages: [], tokens: [], promptLen: 0,
  server: [], done: false, totalMs: 0,
  data: new Map(), requested: new Set(),
  cur: { step: 0, ev: 0, frac: 0, started: false },
  playing: true, speed: 1e-3, finished: false, visText: '', shownSteps: new Set(),
};

// ------------------------------------------------------------------ header / model card
$('#model').innerHTML = `<b>${info.model_id}</b> · ${(info.params / 1e9).toFixed(2)}B params · ${info.n_layers} layers ·
  d=${fmtInt(info.hidden)} · MLP ${fmtInt(info.intermediate)} · ${info.n_heads} heads (${info.n_kv_heads} KV) · vocab ${fmtInt(info.vocab)} ·
  ${info.device.toUpperCase()} ${info.dtype} · <span title="Plain generation speed without the recording hooks">${info.uninstrumented_ms_per_token.toFixed(0)} ms/token uninstrumented</span>`;

// ------------------------------------------------------------------ websocket
let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => setStatus('connected', 'ok');
  ws.onclose = () => { setStatus('disconnected, retrying…', 'bad'); setTimeout(connect, 1500); };
  ws.onmessage = e => {
    if (e.data instanceof ArrayBuffer) return onStep(new StepData(e.data));
    const m = JSON.parse(e.data);
    if (m.gen_id !== undefined && m.gen_id < S.genId) return;
    if (m.type === 'gen_start') onGenStart(m);
    if (m.type === 'token') onToken(m);
    if (m.type === 'gen_done') onGenDone(m);
    if (m.type === 'error') setStatus('model error: ' + m.message, 'bad');
  };
}
connect();
function setStatus(t, cls) { const el = $('#status'); el.textContent = t; el.className = cls; }
const send = o => ws.readyState === 1 && ws.send(JSON.stringify(o));

function onGenStart(m) {
  S.genId = m.gen_id;
  S.tokens = m.prompt_tokens.slice();
  S.promptLen = S.tokens.length;
  S.server = []; S.done = false; S.data.clear(); S.requested.clear(); S.shownSteps.clear();
  S.cur = { step: 0, ev: 0, frac: 0, started: false };
  S.finished = false; S.visText = '';
  viz.resetTokens(); viz.resetCaches();
  renderTokenization(m.prompt_tokens);
  addBubble('assistant', '');
  $('#ahead').textContent = '';
  request(0);
}

function onToken(m) {
  S.server[m.step] = m;
  const txt = S.server.filter(Boolean).map(t => t.text).join('').replace(/<\|im_end\|>|<\|endoftext\|>/g, '');
  $('#ahead').innerHTML = `<div class="k">Already generated on the GPU (${S.server.length} tokens, ${fmtTime(S.server.reduce((a, t) => a + t.wall_ms, 0) / 1000)} wall)</div>${escapeHtml(txt)}`;
  prefetch();
}

function onGenDone(m) {
  S.done = true; S.totalMs = m.total_ms;
  const txt = S.server.map(t => t.text).join('').replace(/<\|im_end\|>|<\|endoftext\|>/g, '');
  S.messages.push({ role: 'assistant', content: txt });
  $('#send').disabled = false;
  updateEta();
}

function onStep(sd) {
  if (sd.meta.gen_id !== S.genId) return;
  S.data.set(sd.step, sd);
  prefetch();
}

function request(k) {
  if (S.requested.has(k)) return;
  if (k > 0 && !S.server[k]) return;
  S.requested.add(k);
  send({ type: 'get_step', gen_id: S.genId, step: k });
}

function prefetch() {
  const k = S.cur.step;
  for (let i = k; i <= k + 2; i++) if (S.server[i] || i === 0) request(i);
  for (const key of S.data.keys()) if (key < k - 1) S.data.delete(key);
}

// ------------------------------------------------------------------ chat
function addBubble(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  $('#chat').appendChild(d);
  $('#chat').scrollTop = 1e9;
  if (role === 'assistant') S.bubble = d;
  return d;
}

function sendMessage() {
  const text = $('#input').value.trim();
  if (!text || ws.readyState !== 1) return;
  $('#input').value = '';
  if (!S.messages.length) S.messages.push({ role: 'system', content: SYSTEM });
  if (!S.done && S.genId) S.messages.push({ role: 'assistant', content: S.visText });
  S.messages.push({ role: 'user', content: text });
  addBubble('user', text);
  S.genId++;   // ignore stragglers from older generations
  send({ type: 'generate', messages: S.messages, max_new_tokens: +$('#maxtok').value, temperature: +$('#temp').value });
  $('#welcome')?.remove();
}
$('#send').onclick = sendMessage;
$('#input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
$('#tempv').textContent = $('#temp').value;
$('#temp').oninput = () => $('#tempv').textContent = $('#temp').value;

function renderTokenization(toks) {
  $('#toks').innerHTML = toks.map((t, i) => `<span class="tk" title="position ${i} · token id ${t.id}">${escapeHtml(showTok(t.text))}<sub>${t.id}</sub></span>`).join('');
  $('#tokcount').textContent = `${toks.length} tokens (chat template included)`;
}

// ------------------------------------------------------------------ transport
const sel = $('#speed');
sel.innerHTML = SPEEDS.map(([k, , l]) => `<option value="${k}">${l}</option>`).join('');
sel.value = '0.001';
sel.onchange = () => { S.speed = SPEEDS.find(s => s[0] === sel.value)[1]; updateEta(); if (curEv()) viz.chooseFocus(curEv(), playDur(curEv())); };
$('#play').onclick = () => setPlaying(!S.playing);
function setPlaying(p) { S.playing = p; $('#play').textContent = p ? '⏸' : '▶'; }
$('#nextop').onclick = () => { if (curData()) { finishCurrent(); S.cur.frac = 0; } };
$('#nexttok').onclick = () => skipToken();
$('#restart').onclick = () => seek(0);
$('#follow').onclick = () => { viz.setFollow(!viz.follow); paintFollow(); };
$('#tower').onclick = () => { viz.view('tower'); paintFollow(); };
viz.onUserCamera = paintFollow;
function paintFollow() { $('#follow').classList.toggle('on', viz.follow); }
paintFollow();
addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!S.playing); }
  if (e.key === 'ArrowRight') $('#nextop').click();
  if (e.key === 'n') skipToken();
  if (e.key === 'f') $('#follow').click();
  if (e.key === 't') $('#tower').click();
  if (e.key === 'Escape') micro.close();
});

// ------------------------------------------------------------------ playback engine
const curData = () => S.data.get(S.cur.step);
const curEv = () => curData()?.meta.events[S.cur.ev];
function playDur(ev) {
  if (S.speed === 'guided') return MATMUL.has(ev.op) || ev.op === 'attn' ? 2.6 : 1.3;
  return ev.dur / S.speed;
}

function onStepStart(sd) {
  const m = sd.meta;
  // the tokens fed into this step occupy positions kv_len - n_in … kv_len - 1
  viz.beginStep();
  viz.setTokens(S.tokens, m.kv_len - m.n_in, m.kv_len);
  renderLens(sd, -1);
  updateEta();
  drawTimeline();
}

function startEvent() {
  const sd = curData(), ev = curEv();
  if (S.cur.ev === 0) onStepStart(sd);
  S.cur.started = true;
  if (micro.sync && micro.isOpen) micro.close();
  viz.begin(ev, sd, playDur(ev));
  renderOp(ev, sd);
}

function completeEvent(ev, sd, animated) {
  if (animated) viz.end(); else viz.instant(ev, sd);
  if (ev.op === 'add2') renderLens(sd, ev.layer);
  if (ev.op === 'sample' && !S.shownSteps.has(sd.step)) {
    S.shownSteps.add(sd.step);
    const t = sd.meta.sampled;
    const isEnd = /<\|im_end\|>|<\|endoftext\|>/.test(t.text);
    if (!isEnd) { S.visText += t.text; S.bubble.textContent = S.visText; }
    S.tokens.push({ id: t.id, text: t.text });
    if (S.done && sd.step === S.server.length - 1) S.finished = true;
  }
}

function finishCurrent() {
  const sd = curData(), ev = curEv();
  if (!ev) return;
  completeEvent(ev, sd, S.cur.started);
  advance();
}

function advance() {
  S.cur.ev++; S.cur.started = false;
  const sd = curData();
  if (S.cur.ev >= sd.meta.events.length) {
    S.cur.step++; S.cur.ev = 0;
    prefetch();
  }
}

function skipToken() {
  const sd = curData();
  if (!sd) return;
  const k = sd.step;
  while (curData() && S.cur.step === k) finishCurrent();
  S.cur.frac = 0;
}

function seek(k) {
  const sd = curData();
  if (!sd) return;
  viz.end();
  onStepStart(sd);
  for (let i = 0; i < k; i++) completeEvent(sd.meta.events[i], sd, false);
  S.cur.ev = k; S.cur.frac = 0; S.cur.started = false;
}

let hudAcc = 0;
viz.hooks.push(dt => {
  micro.tick(dt);
  hudAcc += dt;
  if (!S.playing || S.finished) return;
  if (!curData()) { if (S.server[S.cur.step] || S.cur.step === 0) prefetch(); $('#clock').textContent = S.genId ? (S.done && S.cur.step >= S.server.length ? 'done' : 'waiting for GPU…') : 'idle'; return; }
  let budget = dt;
  let guard = 0;
  while (curData() && guard++ < 5000) {
    const ev = curEv(), d = playDur(ev);
    // events far shorter than a frame are applied without animation
    if (!S.cur.started && d < 0.02 && budget >= d * (1 - S.cur.frac)) {
      budget -= d * (1 - S.cur.frac);
      completeEvent(ev, curData(), false);
      advance(); S.cur.frac = 0;
      continue;
    }
    if (!S.cur.started) startEvent();
    const need = d * (1 - S.cur.frac);
    if (budget < need) { S.cur.frac += budget / d; budget = 0; break; }
    budget -= need;
    completeEvent(ev, curData(), true);
    advance(); S.cur.frac = 0;
  }
  if (curData() && S.cur.started) {
    viz.progress(S.cur.frac);
    if (hudAcc > 0.06) { hudAcc = 0; updateLive(); }
  }
  if (!curData() && S.finished) $('#clock').textContent = 'done';
});

// ------------------------------------------------------------------ HUD
function vecPreview(v, n = 8) {
  const a = Array.from(v.subarray(0, n), x => fmtNum(x, 4));
  return `[${a.join(', ')}, … ] <span class="dim">(${fmtInt(v.length)})</span>`;
}
function stats(v) {
  let mn = Infinity, mx = -Infinity, ss = 0, mi = 0;
  for (let i = 0; i < v.length; i++) { const x = v[i]; if (x < mn) mn = x; if (x > mx) mx = x; ss += x * x; if (Math.abs(x) > Math.abs(v[mi])) mi = i; }
  const s98 = robustScale(v);
  return { mn, mx, rms: Math.sqrt(ss / v.length), mi, out: Math.abs(v[mi]) / s98 };
}
function statLine(v) {
  const s = stats(v);
  return `min ${fmtNum(s.mn, 4)} · max ${fmtNum(s.mx, 4)} · rms ${fmtNum(s.rms, 4)}${s.out > 8 ? ` · <span class="warn" title="A few residual-stream dimensions carry huge values (a known LLM phenomenon). Colors clip them.">outlier dim #${s.mi} = ${fmtNum(v[s.mi], 4)}</span>` : ''}`;
}

function tensorsFor(ev, sd) {
  const L = ev.layer, g = n => sd.get(`${L}.${n}`);
  const KV = info.n_kv_heads * info.head_dim, n = sd.meta.n_in;
  const lastRow = v => v.subarray((n - 1) * KV, n * KV);
  switch (ev.op) {
    case 'embed': return { out: ['x⁽⁰⁾ = E[id]', sd.get('emb')] };
    case 'norm1': return { in: ['x', g('resid_in')], out: ['RMSNorm(x)', g('norm1')] };
    case 'q': return { in: ['x', g('norm1')], out: ['q', g('q')] };
    case 'k': return { in: ['x', g('norm1')], out: ['k', lastRow(g('k'))] };
    case 'v': return { in: ['x', g('norm1')], out: ['v', lastRow(g('v'))] };
    case 'attn': return { out: ['ctx', g('ctx')] };
    case 'o': return { in: ['ctx', g('ctx')], out: ['attn out', g('o')] };
    case 'add1': return { in: ['x', g('resid_in')], in2: ['attn out', g('o')], out: ['x + attn', g('resid_mid')] };
    case 'norm2': return { in: ['x', g('resid_mid')], out: ['RMSNorm(x)', g('norm2')] };
    case 'gate': return { in: ['x', g('norm2')], out: ['g', g('gate')] };
    case 'act': return { in: ['g', g('gate')], out: ['SiLU(g)', g('act')] };
    case 'up': return { in: ['x', g('norm2')], out: ['u', g('up')] };
    case 'mul': return { in: ['SiLU(g)', g('act')], in2: ['u', g('up')], out: ['h', g('mul')] };
    case 'down': return { in: ['h', g('mul')], out: ['MLP out', g('down')] };
    case 'add2': return { in: ['x', g('resid_mid')], in2: ['MLP out', g('down')], out: ['layer output', g('resid_out')] };
    case 'final_norm': return { in: ['x', sd.get(`${info.n_layers - 1}.resid_out`)], out: ['y', sd.get('final_norm')] };
    case 'lm_head': return { in: ['y', sd.get('final_norm')] };
  }
  return {};
}

function opDetail(ev, sd, T) {
  const m = sd.meta, L = ev.layer;
  const rows = [];
  const pairs = (a, b, c, f) => {
    const out = [];
    for (let i = 0; i < 5; i++) out.push(`<div class="eq">${f(a[i], b?.[i], c[i], i)}</div>`);
    return out.join('');
  };
  if (ev.op === 'embed') {
    const id = m.input_ids.at(-1);
    rows.push(`<div class="eq">token <b>${escapeHtml(showTok(S.tokens[m.kv_len - 1]?.text ?? ''))}</b> → id <b>${id}</b> → row ${fmtInt(id)} of E</div>`);
  }
  if (ev.op === 'norm1' || ev.op === 'norm2' || ev.op === 'final_norm') {
    const x = T.in[1], s = stats(x);
    const r = Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length + info.rms_eps);
    rows.push(`<div class="eq">rms(x) = √(mean(x²)+ε) = <b>${fmtNum(r, 6)}</b> <span class="dim">(recomputed here from the captured x)</span></div>`);
    rows.push(pairs(x, null, T.out[1], (a, _, c, i) => `y[${i}] = ${fmtNum(a, 5)} / ${fmtNum(r, 4)} · g[${i}] = <b>${fmtNum(c, 5)}</b>`));
  }
  if (ev.op === 'add1' || ev.op === 'add2') {
    const a = T.in[1], b = T.in2[1], c = T.out[1];
    let err = 0; for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] + b[i] - c[i]));
    rows.push(pairs(a, b, c, (x, y, z) => `${fmtNum(x, 5)} + ${fmtNum(y, 5)} = <b>${fmtNum(z, 5)}</b>`));
    rows.push(`<div class="dim">check: max |x + a − y| over all ${fmtInt(a.length)} dims = ${err.toExponential(1)}</div>`);
  }
  if (ev.op === 'act') rows.push(pairs(T.in[1], null, T.out[1], (g, _, y) => `SiLU(${fmtNum(g, 5)}) = <b>${fmtNum(y, 5)}</b>`));
  if (ev.op === 'mul') rows.push(pairs(T.in[1], T.in2[1], T.out[1], (a, b, c) => `${fmtNum(a, 5)} × ${fmtNum(b, 5)} = <b>${fmtNum(c, 5)}</b>`));
  if (ev.op === 'attn') {
    const A = sd.get(`${L}.attn`), kv = m.kv_len, nH = info.n_heads;
    const hs = [];
    for (let h = 0; h < nH; h++) {
      const row = A.subarray(h * kv, (h + 1) * kv);
      const top = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, 3);
      hs.push(`<div class="head"><span>h${h}</span>${top.map(p => `<i style="opacity:${0.35 + row[p] * 0.65}">${escapeHtml(showTok(S.tokens[p]?.text ?? '?'))}<sub>${(row[p] * 100).toFixed(0)}%</sub></i>`).join('')}</div>`);
    }
    rows.push(`<div class="dim">Most-attended earlier tokens per head (real softmax weights):</div>${hs.join('')}`);
  }
  if (ev.op === 'sample') {
    rows.push(m.topk.slice(0, 8).map(t => `<div class="eq ${t.id === m.sampled.id ? 'pick' : ''}"><b>${escapeHtml(showTok(t.text))}</b> logit ${fmtNum(t.logit, 5)} → p ${(t.p * 100).toFixed(2)}%</div>`).join(''));
    rows.push(`<div class="eq">sampled → <b class="pick">${escapeHtml(showTok(m.sampled.text))}</b></div>`);
  }
  return rows.join('');
}

let live = null;
function renderOp(ev, sd) {
  const m = sd.meta, I = opInfo(info, ev, m), T = tensorsFor(ev, sd);
  const pd = playDur(ev);
  const slow = isSlow(ev);
  $('#op').innerHTML = `
    <div class="sec" style="--c:${GROUP_COLOR[GROUP[ev.op]]}">${I.section}<span>${I.where}</span><span>token step ${sd.step}${m.n_in > 1 ? ' · prefill' : ''}</span></div>
    <h2>${I.title}</h2>
    <p>${I.desc}</p>
    <div class="formula">${I.formula}</div>
    <div class="grid">
      <div><label>multiply-adds</label>${I.macs ? fmtBig(I.macs) : '—'}</div>
      <div><label>measured GPU time</label>${fmtTime(ev.dur)}${slow ? ' <span class="warn" title="Much slower than the same op in other layers, usually a GPU scheduling hiccup or Python pause during recording">⚠ outlier</span>' : ''}</div>
      <div><label>plays for</label>${fmtTime(pd)}</div>
      <div><label>GPU throughput</label>${I.macs ? fmtBig(I.macs * 2 / ev.dur) + 'FLOP/s' : '—'}</div>
    </div>
    <div class="nums">
      ${T.in ? `<div class="v"><label>${T.in[0]} (input)</label>${vecPreview(T.in[1])}</div>` : ''}
      ${T.in2 ? `<div class="v"><label>${T.in2[0]}</label>${vecPreview(T.in2[1])}</div>` : ''}
      <div id="liveout"></div>
      ${T.out ? `<div class="v"><label>${T.out[0]} (output)</label>${vecPreview(T.out[1])}<div class="dim">${statLine(T.out[1])}</div></div>` : ''}
      ${opDetail(ev, sd, T)}
    </div>
    ${I.mat ? `<button id="inspect">🔬 Watch one dot product, multiply by multiply</button><div class="dim small">or click any cell of a matrix / output vector in 3D</div>` : ''}
    ${MATMUL.has(ev.op) ? `<p class="dim small">The GPU computes these ${fmtInt(matDims(info, ev.op)[1])} dot products in parallel tiles across its cores. The sweep puts them in order so you can follow, and the time is scaled from the measured kernel time.</p>` : ''}`;
  live = { ev, sd, I };
  if (I.mat) $('#inspect').onclick = () => openMicro(ev.layer, I.mat, I.mat === 'lm_head' ? m.sampled.id : currentJ(), sd);
  updateLive();
}

const opDurs = {};
function isSlow(ev) {
  const sd = curData();
  if (!sd) return false;
  const same = sd.meta.events.filter(e => e.op === ev.op).map(e => e.dur).sort((a, b) => a - b);
  return same.length > 4 && ev.dur > 8 * same[same.length >> 1];
}

function currentJ() {
  if (!live || !MATMUL.has(live.ev.op)) return 0;
  const dout = live.ev.op === 'lm_head' ? info.vocab : matDims(info, live.ev.op)[1];
  return Math.min(dout - 1, Math.floor(S.cur.frac * dout));
}

function updateLive() {
  const sd = curData(), ev = curEv();
  if (!sd || !ev) return;
  // clock
  const evs = sd.meta.events;
  let t = 0, tot = 0;
  for (let i = 0; i < evs.length; i++) { if (i < S.cur.ev) t += evs[i].dur; tot += evs[i].dur; }
  t += ev.dur * S.cur.frac;
  $('#clock').innerHTML = `GPU clock <b>${(t * 1000).toFixed(6)} ms</b> / ${(tot * 1000).toFixed(2)} ms · token ${sd.step + 1}${S.server.length ? '/' + S.server.length : ''}`;
  drawPlayhead(t, tot);
  if (!live || live.ev !== ev) return;
  const lo = $('#liveout');
  if (!lo) return;
  if (MATMUL.has(ev.op) && ev.op !== 'lm_head') {
    const [din, dout] = matDims(info, ev.op);
    const T = tensorsFor(ev, sd), out = T.out[1];
    const j = currentJ();
    const recent = [];
    for (let k = Math.max(0, j - 5); k <= j; k++) recent.push(`<span class="${k === j ? 'cur' : ''}">[${k}] ${fmtNum(out[k], 5)}</span>`);
    lo.innerHTML = `<div class="v live"><label>now computing</label><div class="big">${T.out[0]}[${fmtInt(j)}] = ${fmtNum(out[j], 6)}</div>
      <div class="dim">dot product of ${fmtInt(din)} weights × ${fmtInt(din)} inputs · output ${fmtInt(j + 1)} of ${fmtInt(dout)}${sd.meta.n_in > 1 && ev.op !== 'lm_head' ? ` · ×${sd.meta.n_in} rows in prefill` : ''}</div>
      <div class="ticker">${recent.join('')}</div></div>`;
    // at extreme slowdowns each output takes seconds: show its multiply-adds in sync with the clock
    const perCol = playDur(ev) / dout;
    if (perCol > 2.5 && S.speed !== 'guided') {
      const colFrac = S.cur.frac * dout - j;
      openMicro(ev.layer, ev.op, j, sd, true).then(() => micro.syncTo(colFrac));
      micro.syncTo(colFrac);
    }
  } else if (ev.op === 'lm_head') {
    const j = currentJ();
    lo.innerHTML = `<div class="v live"><label>now computing</label><div class="big">logit[${fmtInt(j)}]</div><div class="dim">${fmtInt(j + 1)} of ${fmtInt(info.vocab)} vocabulary scores</div></div>`;
  }
}

async function openMicro(layer, mat, j, sd, sync = false) {
  if (!sync) setPlaying(false);
  await micro.open({ layer, mat, j, sd, sync });
}

viz.renderer.domElement.addEventListener('pointerdown', e => { viz._pd = [e.clientX, e.clientY]; });
viz.renderer.domElement.addEventListener('pointerup', e => {
  if (!viz._pd || Math.hypot(e.clientX - viz._pd[0], e.clientY - viz._pd[1]) > 4) return;
  const hit = viz.pick(e.clientX, e.clientY);
  if (!hit) return;
  // use the step whose numbers are currently shown on that floor
  const sd = curData() || S.data.get(S.cur.step - 1);
  if (!sd) return;
  const evIdx = sd.meta.events.findIndex(x => x.op === hit.mat && x.layer === hit.layer);
  const shown = evIdx >= 0 && evIdx < S.cur.ev ? sd : S.data.get(sd.step - 1);
  if (!shown) return;
  openMicro(hit.layer, hit.mat, hit.mat === 'lm_head' ? shown.meta.sampled.id : hit.j, shown);
});

// ------------------------------------------------------------------ logit lens
function renderLens(sd, upto) {
  const lens = sd.meta.lens;
  $('#lens').innerHTML = lens.map((l, i) => {
    const on = i <= upto;
    const t = l[0];
    return `<div class="lr ${on ? '' : 'off'} ${i === upto ? 'now' : ''}"><span>L${i}</span><b>${on ? escapeHtml(showTok(t.text)) : '·'}</b><i style="width:${on ? t.p * 100 : 0}%"></i><em>${on ? (t.p * 100).toFixed(0) + '%' : ''}</em></div>`;
  }).reverse().join('');
}

// ------------------------------------------------------------------ timeline (real per-op GPU timings)
const tl = $('#timeline'), tctx = tl.getContext('2d');
let tlCache = null;
function drawTimeline() {
  const sd = curData();
  if (!sd) return;
  const W = tl.width = tl.clientWidth * devicePixelRatio, Hh = tl.height = tl.clientHeight * devicePixelRatio;
  const evs = sd.meta.events, tot = evs.reduce((a, e) => a + e.dur, 0);
  tctx.clearRect(0, 0, W, Hh);
  let x = 0;
  const xs = [];
  for (const e of evs) {
    const w = e.dur / tot * W;
    tctx.fillStyle = GROUP_COLOR[GROUP[e.op]];
    tctx.globalAlpha = 0.85;
    tctx.fillRect(x, Hh * 0.25, Math.max(1, w - (w > 3 ? 1 : 0)), Hh * 0.5);
    xs.push([x, w]);
    x += w;
  }
  tctx.globalAlpha = 1;
  tlCache = { img: tctx.getImageData(0, 0, W, Hh), xs, tot, evs, step: sd.step };
}
function drawPlayhead(t, tot) {
  if (!tlCache || tlCache.step !== S.cur.step) drawTimeline();
  if (!tlCache) return;
  tctx.putImageData(tlCache.img, 0, 0);
  const x = t / tot * tl.width;
  tctx.fillStyle = '#fff';
  tctx.fillRect(x - 1, 0, 2 * devicePixelRatio, tl.height);
}
function tlIndexAt(e) {
  if (!tlCache) return -1;
  const x = (e.offsetX / tl.clientWidth) * tl.width;
  return tlCache.xs.findIndex(([a, w]) => x >= a && x < a + w);
}
tl.addEventListener('mousemove', e => {
  const i = tlIndexAt(e), tip = $('#tltip');
  if (i < 0) { tip.style.display = 'none'; return; }
  const ev = tlCache.evs[i];
  tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX, innerWidth - 220) + 'px';
  tip.innerHTML = `<b>${ev.layer >= 0 ? 'L' + ev.layer + ' · ' : ''}${ev.op}</b> ${fmtTime(ev.dur)}<br><span class="dim">click to jump here</span>`;
});
tl.addEventListener('mouseleave', () => $('#tltip').style.display = 'none');
tl.addEventListener('click', e => { const i = tlIndexAt(e); if (i >= 0) seek(i); });
addEventListener('resize', () => { tlCache = null; });

function updateEta() {
  const sd = curData();
  if (!sd) return;
  const tot = sd.meta.events.reduce((a, e) => a + e.dur, 0);
  const play = sd.meta.events.reduce((a, e) => a + playDur(e), 0);
  $('#eta').innerHTML = `this token: <b>${fmtTime(tot)}</b> of GPU work → plays in <b>${fmtTime(play)}</b>`;
}

// ------------------------------------------------------------------ legend toggles
$('#lblbtn').onclick = () => { document.body.classList.toggle('nolabels'); $('#lblbtn').classList.toggle('on'); };
$('#panelbtn').onclick = () => { document.body.classList.toggle('nopanels'); viz.applyViewOffset(); if (viz.follow && curEv()) viz.chooseFocus(curEv(), playDur(curEv())); };

// handle for poking at the app from the devtools console
window.llmviewer = { S, viz, micro, seek, setPlaying, skipToken };
