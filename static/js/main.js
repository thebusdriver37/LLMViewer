// App controller: talks to the GPU server, replays recorded steps on a scaled clock, drives HUD + 3D.
import { Viz, escapeHtml } from './scene.js';
import { Micro } from './micro.js';
import { Sound, Narrator } from './audio.js';
import { opInfo, simpleText, MATMUL, GROUP, GROUP_COLOR, matDims, MAT_NAME } from './ops.js';
import { StepData, fmtNum, fmtInt, fmtBig, fmtTime, showTok, robustScale, divColor } from './util.js';

const $ = s => document.querySelector(s);
const SPEEDS = [
  ['1', 1, '1× (real GPU time)'], ['0.1', 0.1, '1/10×'], ['0.01', 0.01, '1/100×'], ['0.001', 1e-3, '1/1,000×'],
  ['1e-4', 1e-4, '1/10,000×'], ['1e-5', 1e-5, '1/100,000×'], ['1e-6', 1e-6, '1/1,000,000×'],
  ['1e-7', 1e-7, '1/10,000,000×'], ['1e-9', 1e-9, '1/1,000,000,000×'], ['guided', 'guided', 'Guided (~2 s per step)'],
];
const SYSTEM = 'You are a helpful assistant. Answer briefly.';
const PRESETS = [
  ['🗼 A fact', 'What is the capital of France?'],
  ['🔢 Counting', 'Count from 1 to 5.'],
  ['🌍 Translate', 'Translate "good morning" into Spanish.'],
  ['🌹 Rhyme', 'Finish the rhyme: Roses are red, violets are'],
  ['➕ Math', 'What is 7 + 5?'],
  ['✨ Poem', 'Write a haiku about stars.'],
];
const store = {
  get: (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

const info = await fetch('/api/model').then(r => r.json());
const thumbs = new Uint8Array(await fetch('/api/thumbs').then(r => r.arrayBuffer()));
const viz = new Viz($('#stage'), info, thumbs);
const micro = new Micro($('#micro'), info, viz);
const sound = new Sound(), narrator = new Narrator();

const S = {
  genId: 0, messages: [], tokens: [], promptLen: 0,
  server: [], done: false, totalMs: 0,
  data: new Map(), requested: new Set(),
  cur: { step: 0, ev: 0, frac: 0, started: false },
  playing: true, speed: 1e-3, finished: false, visText: '', shownSteps: new Set(),
  ablate: { heads: [], attn: [], mlp: [] }, runAblate: null, lastAnswer: null,
  lastSample: null, attrAlt: false, attr3d: false, tab: 'now', mode: 'simple',
};
const curData = () => S.data.get(S.cur.step);
const curEv = () => curData()?.meta.events[S.cur.ev];

// ------------------------------------------------------------------ header / model card / modes
$('#model').innerHTML = `${(info.params / 1e9).toFixed(2)}B params · ${info.n_layers} layers ·
  d=${fmtInt(info.hidden)} · MLP ${fmtInt(info.intermediate)} · ${info.n_heads} heads (${info.n_kv_heads} KV) · vocab ${fmtInt(info.vocab)} ·
  ${info.device.toUpperCase()} ${info.dtype}${info.gpu_cores ? ` · ${info.gpu_cores}-core GPU` : ''} · <span title="Plain generation speed without the recording hooks">${info.uninstrumented_ms_per_token.toFixed(0)} ms/token normally</span>`;
$('#obmodel').textContent = `${info.model_id.split('/')[1]}, ${(info.params / 1e9).toFixed(1)} billion numbers`;
const msel = $('#modelsel');
msel.innerHTML = [...new Set([info.model_id, ...info.choices])].map(m => `<option value="${m}">${m.split('/')[1]}</option>`).join('');
msel.value = info.model_id;
msel.onchange = async () => {
  const id = msel.value;
  $('#loading').classList.remove('hidden');
  $('#loading .lt').textContent = `Loading ${id}… (the first time downloads it from Hugging Face)`;
  const r = await fetch('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_id: id }) });
  const j = await r.json();
  if (j.ok) location.reload();
  else { $('#loading .lt').textContent = 'Could not load: ' + j.error; setTimeout(() => $('#loading').classList.add('hidden'), 5000); msel.value = info.model_id; }
};

function setMode(m) {
  S.mode = m;
  document.body.classList.toggle('simple', m === 'simple');
  document.body.classList.toggle('expert', m === 'expert');
  document.querySelectorAll('#modeseg button').forEach(b => b.classList.toggle('on', b.dataset.m === m));
  store.set('llmv_mode', m);
  if (curEv() && S.cur.started) renderOp(curEv(), curData());
}
document.querySelectorAll('#modeseg button').forEach(b => b.onclick = () => setMode(b.dataset.m));
setMode(store.get('llmv_mode', 'simple'));

$('#soundbtn').onclick = () => { sound.enable(!sound.on); paintToggles(); };
$('#narrbtn').onclick = () => { narrator.enable(!narrator.on); paintToggles(); };
function paintToggles() {
  $('#soundbtn').classList.toggle('on', sound.on); $('#soundbtn').textContent = sound.on ? '🔊 sound' : '🔈 sound';
  $('#narrbtn').classList.toggle('on', narrator.on);
}
paintToggles();

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
const cleanText = t => t.replace(/<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>/g, '');

function onGenStart(m) {
  S.genId = m.gen_id;
  S.tokens = m.prompt_tokens.slice();
  S.promptLen = S.tokens.length;
  S.server = []; S.done = false; S.data.clear(); S.requested.clear(); S.shownSteps.clear();
  S.cur = { step: 0, ev: 0, frac: 0, started: false };
  S.finished = false; S.visText = ''; S.lastSample = null;
  viz.resetTokens(); viz.resetCaches(); viz.setAttribution(null);
  renderTokenization(m.prompt_tokens);
  const ab = m.ablate || {};
  const nAb = (ab.heads?.length || 0) + (ab.attn?.length || 0) + (ab.mlp?.length || 0);
  addBubble('assistant', '', nAb ? `🧪 with ${nAb} knockout${nAb > 1 ? 's' : ''}: ${ablateLabel(ab)}` : '');
  $('#ahead').textContent = '';
  renderWhy();
  request(0);
}

function onToken(m) {
  S.server[m.step] = m;
  const txt = cleanText(S.server.filter(Boolean).map(t => t.text).join(''));
  $('#ahead').innerHTML = `<div class="k">Already finished on the GPU: ${S.server.length} tokens in ${fmtTime(S.server.reduce((a, t) => a + t.wall_ms, 0) / 1000)}</div>${escapeHtml(txt)}`;
  prefetch();
}

function onGenDone(m) {
  S.done = true; S.totalMs = m.total_ms;
  const txt = cleanText(S.server.map(t => t.text).join(''));
  if (!S.runAblate) { S.messages.push({ role: 'assistant', content: txt }); S.lastAnswer = txt; }
  else if (S.bubbleMeta) S.bubbleMeta.innerHTML += `<br>without knockouts it said: <i>${escapeHtml(S.lastAnswer ?? '')}</i>`;
  S.runAblate = null;
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

// ------------------------------------------------------------------ chat, presets, experiments, recordings
function addBubble(role, text, meta = '') {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  const body = document.createElement('div'); body.textContent = text; d.appendChild(body);
  if (role === 'assistant') S.bubbleMeta = null;
  if (meta) { const m = document.createElement('div'); m.className = 'mmeta'; m.innerHTML = meta; d.appendChild(m); S.bubbleMeta = m; }
  $('#chat').appendChild(d);
  $('#chat').scrollTop = 1e9;
  if (role === 'assistant') S.bubble = body;
  return d;
}

function generate(messages, ablate = null) {
  S.genId++;   // ignore stragglers from older generations
  send({ type: 'generate', messages, max_new_tokens: +$('#maxtok').value, temperature: +$('#temp').value, ablate });
  $('#onboard').classList.add('hidden');
}

function sendMessage(text = $('#input').value.trim()) {
  if (!text || ws.readyState !== 1) return;
  $('#input').value = '';
  if (!S.messages.length) S.messages.push({ role: 'system', content: SYSTEM });
  if (!S.done && S.genId && S.messages.at(-1)?.role === 'user') S.messages.push({ role: 'assistant', content: S.visText });
  S.messages.push({ role: 'user', content: text });
  addBubble('user', text);
  S.runAblate = null;
  generate(S.messages);
}
$('#send').onclick = () => sendMessage();
$('#input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
$('#tempv').textContent = $('#temp').value;
$('#temp').oninput = () => $('#tempv').textContent = $('#temp').value;
$('#presets').innerHTML = PRESETS.map(([l, p]) => `<button data-p="${escapeHtml(p)}">${l}</button>`).join('');
$('#presets').querySelectorAll('button').forEach(b => b.onclick = () => sendMessage(b.dataset.p));

function ablateLabel(ab) {
  return [...(ab.mlp || []).map(l => `MLP L${l}`), ...(ab.attn || []).map(l => `attention L${l}`), ...(ab.heads || []).map(h => `head L${h.layer}.h${h.head}`)].join(', ');
}
const sameAb = (kind, x, val) => kind === 'heads' ? x.layer === val.layer && x.head === val.head : x === val;
function isAblated(kind, val) { return S.ablate[kind].some(x => sameAb(kind, x, val)); }
function toggleAblate(kind, val) {
  const list = S.ablate[kind];
  const i = list.findIndex(x => sameAb(kind, x, val));
  i >= 0 ? list.splice(i, 1) : list.push(val);
  renderAblate();
  if (S.tab === 'why') renderWhy();
  if (curEv() && S.cur.started) renderOp(curEv(), curData());
}
window.llmvAblate = (kind, a, b) => toggleAblate(kind, kind === 'heads' ? { layer: a, head: b } : a);
function renderAblate() {
  const ab = S.ablate, n = ab.heads.length + ab.attn.length + ab.mlp.length;
  $('#expsec').classList.toggle('hidden', !n);
  $('#ablist').innerHTML = [
    ...ab.mlp.map(l => `<span class="abl" onclick="llmvAblate('mlp',${l})">MLP L${l} ✕</span>`),
    ...ab.attn.map(l => `<span class="abl" onclick="llmvAblate('attn',${l})">attention L${l} ✕</span>`),
    ...ab.heads.map(h => `<span class="abl" onclick="llmvAblate('heads',${h.layer},${h.head})">head L${h.layer}.h${h.head} ✕</span>`)].join('');
}
$('#rerun').onclick = () => {
  const msgs = S.messages.slice();
  while (msgs.length && msgs.at(-1).role === 'assistant') msgs.pop();
  if (!msgs.some(m => m.role === 'user')) return;
  S.runAblate = JSON.parse(JSON.stringify(S.ablate));
  generate(msgs, S.runAblate);
};
$('#abclear').onclick = () => { S.ablate = { heads: [], attn: [], mlp: [] }; renderAblate(); if (S.tab === 'why') renderWhy(); };
renderAblate();

$('#savebtn').onclick = async () => {
  const blob = await fetch('/api/recording').then(r => r.blob());
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `llmviewer-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.llmv`;
  a.click();
};
$('#openbtn').onclick = () => $('#openfile').click();
$('#openfile').onchange = async () => {
  const f = $('#openfile').files[0];
  if (!f) return;
  const j = await fetch('/api/recording', { method: 'POST', body: await f.arrayBuffer() }).then(r => r.json());
  $('#openfile').value = '';
  if (j.error) { setStatus(j.error, 'bad'); return; }
  $('#chat').innerHTML = '';
  for (const m of j.messages) if (m.role !== 'system') addBubble(m.role, m.content);
  S.messages = j.messages.slice();
  S.genId = j.gen_id;
  onGenStart({ gen_id: j.gen_id, prompt_tokens: j.prompt_tokens, ablate: j.ablate });
  for (const t of j.tokens) onToken({ ...t, gen_id: j.gen_id });
  onGenDone({ gen_id: j.gen_id, total_ms: j.tokens.reduce((a, t) => a + t.wall_ms, 0) });
};

function renderTokenization(toks) {
  $('#toks').innerHTML = toks.map((t, i) => `<span class="tk" title="position ${i} · token id ${t.id}">${escapeHtml(showTok(t.text))}<sub>${t.id}</sub></span>`).join('');
  $('#tokcount').textContent = `${toks.length} tokens (chat template included)`;
}

// ------------------------------------------------------------------ transport
const sel = $('#speed');
sel.innerHTML = SPEEDS.map(([k, , l]) => `<option value="${k}">${l}</option>`).join('');
function setSpeed(k) { sel.value = k; S.speed = SPEEDS.find(s => s[0] === k)[1]; updateEta(); if (curEv() && S.cur.started) viz.chooseFocus(curEv(), playDur(curEv())); }
sel.onchange = () => { if (story.on) stopTour(); setSpeed(sel.value); };
$('#sweep').onchange = () => { viz.sweep = $('#sweep').value; };
$('#play').onclick = () => setPlaying(!S.playing);
function setPlaying(p) { S.playing = p; $('#play').textContent = p ? '⏸' : '▶'; if (!p) narrator.stop(); }
$('#nextop').onclick = () => { if (curData()) { finishCurrent(); S.cur.frac = 0; } };
$('#nexttok').onclick = () => skipToken();
$('#restart').onclick = () => seek(0);
$('#follow').onclick = () => { viz.setFollow(!viz.follow); paintFollow(); };
$('#tower').onclick = () => { viz.view('tower'); paintFollow(); };
viz.onUserCamera = paintFollow;
function paintFollow() { $('#follow').classList.toggle('on', viz.follow); }
paintFollow();
addEventListener('keydown', e => {
  if (['TEXTAREA', 'INPUT', 'SELECT'].includes(e.target.tagName)) return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!S.playing); }
  if (e.key === 'ArrowRight') $('#nextop').click();
  if (e.key === 'n') skipToken();
  if (e.key === 'f') $('#follow').click();
  if (e.key === 't') $('#tower').click();
  if (e.key === 'Escape') { micro.close(); if (story.on) stopTour(); }
});

// ------------------------------------------------------------------ guided tour
const story = { on: false, said: new Set() };
const textDur = t => Math.max(3.2, Math.min(10, t.split(/\s+/).length / 2.6 + 1.4));
function startTour() {
  $('#onboard').classList.add('hidden');
  setMode('simple');
  story.on = true; story.said = new Set();
  setSpeed('guided');
  viz.setFollow(true); paintFollow();
  setPlaying(true);
  $('#chat').innerHTML = '';
  S.messages = [];
  caption('Let\'s follow one question through a real AI model running on this computer.', `It will read your words, think through ${info.n_layers} layers, and pick an answer one word at a time.`);
  sendMessage(PRESETS[0][1]);
}
function stopTour() { story.on = false; narrator.stop(); }
$('#tourbtn').onclick = () => { if (!narrator.on) { narrator.enable(true); paintToggles(); } startTour(); };
$('#obtour').onclick = () => { if (!narrator.on) { narrator.enable(true); paintToggles(); } startTour(); };
$('#obfree').onclick = () => $('#onboard').classList.add('hidden');
if (!store.get('llmv_seen2', false)) $('#onboard').classList.remove('hidden');
store.set('llmv_seen2', true);

// the tour narrates only the steps that tell the story; the rest pass quickly
const TOUR_TEXT = {
  embed: () => simpleText(info, 'embed'),
  q: () => ['Three questions for every word', 'The word makes a question (what am I looking for?), a label (what do I contain?), and a package of information to share.'],
  softmax: () => ['Decide where to look', 'It compares its question with every earlier word\'s label. The glowing arcs show which words it pays attention to.'],
  add1: () => simpleText(info, 'add1'),
  gate: () => ['Check thousands of feature detectors', `Next, ${fmtInt(info.intermediate)} detectors check what this word is about. Only the strong ones switch on.`],
  down: () => simpleText(info, 'down'),
  add2: () => ['One layer done', 'The glowing word beside the layer is what the model would answer if it stopped right now.'],
  lm_head: () => simpleText(info, 'lm_head'),
  sample: () => simpleText(info, 'sample'),
};
function storyDur(ev, sd) {
  const nL = info.n_layers, L = ev.layer;
  if (!sd || sd.step > 0) return null;
  if (L > 0 && L < nL - 1) return 0.07;
  if (L === nL - 1) return 0.3;
  const tt = TOUR_TEXT[ev.op];
  if (!tt) return 1.1;
  const [t, b] = tt();
  return Math.min(9, textDur(t + ' ' + b)) + (ev.op === 'sample' ? 2 : 0);
}
function storyEvent(ev, sd) {
  const nL = info.n_layers, L = ev.layer;
  if (sd.step > 0) {
    caption('That\'s one word.', 'Now it does all of that again for every next word, much faster. Ask your own question on the left, or hover and click anything to look closer.');
    setSpeed('0.01'); story.on = false;
    return;
  }
  const key = L > 0 && L < nL - 1 ? 'climb' + (L < 8 ? 0 : L < 18 ? 1 : 2) : ev.op + L;
  if (story.said.has(key)) return;
  story.said.add(key);
  if (key === 'climb0') caption('The same steps repeat in every layer.', `There are ${nL} layers. Each one refines the meaning a little. Watch the big glowing word beside each layer: it is the model's best guess so far.`);
  else if (key === 'climb1') caption('Climbing through the middle layers…', 'Early guesses are often nonsense. The real answer usually appears near the top.');
  else if (key === 'climb2') caption('Near the top now.', 'Watch the guess settle on the answer.');
  else if (!key.startsWith('climb') && TOUR_TEXT[ev.op]) { const [t, b] = TOUR_TEXT[ev.op](); caption(t, b); }
}

let capTimer;
function caption(title, body) {
  const c = $('#caption');
  c.innerHTML = `<b>${escapeHtml(title)}</b> ${escapeHtml(body)}`;
  c.classList.add('show');
  narrator.say(`${title} ${body}`);
  clearTimeout(capTimer);
  capTimer = setTimeout(() => c.classList.remove('show'), 14000);
}

// ------------------------------------------------------------------ playback engine
function playDur(ev) {
  if (S.durOverride) { const d = S.durOverride(ev); if (d != null) return d; }
  if (story.on) { const d = storyDur(ev, curData()); if (d !== null) return d; }
  if (S.speed === 'guided') return MATMUL.has(ev.op) || ev.op === 'softmax' ? 2.6 : 1.4;
  const d = ev.dur / S.speed;
  return ev.op === 'sample' && S.speed <= 0.01 ? Math.max(d, 1.5) : d;   // hold the pick on screen
}

function onStepStart(sd) {
  const m = sd.meta;
  // the tokens fed into this step occupy positions kv_len - n_in … kv_len - 1
  viz.setTokens(S.tokens, m.kv_len - m.n_in, m.kv_len);
  viz.beginStep(sd);
  renderLens(sd, -1);
  updateEta();
  drawTimeline();
}

function startEvent() {
  const sd = curData(), ev = curEv();
  if (S.cur.ev === 0) onStepStart(sd);
  S.cur.started = true;
  if (micro.sync && micro.isOpen) micro.close();
  const pd = playDur(ev);
  viz.begin(ev, sd, pd);
  sound.event(ev, sd);
  if (story.on) storyEvent(ev, sd);
  else if (S.mode === 'simple' && pd > 1 && document.body.classList.contains('nopanels')) { const [t, b] = simpleText(info, ev.op); caption(t, b); }
  renderOp(ev, sd);
  drawMinimap(ev.layer);
}

function completeEvent(ev, sd, animated) {
  if (animated) viz.end(); else viz.instant(ev, sd);
  if (ev.op === 'add2') renderLens(sd, ev.layer);
  if (ev.op === 'sample' && !S.shownSteps.has(sd.step)) {
    S.shownSteps.add(sd.step);
    const t = sd.meta.sampled;
    if (cleanText(t.text) === t.text) { S.visText += t.text; S.bubble.textContent = S.visText; }
    S.tokens.push({ id: t.id, text: t.text });
    S.lastSample = sd;
    if (S.tab === 'why') renderWhy();
    if (S.attr3d) viz.setAttribution(attrRows(sd));
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
    const ev = curEv();
    if (MATMUL.has(ev.op) && ev.op !== 'lm_head' && viz.spec?.mat) {
      const st = viz.spec.mat.outStrip, j = Math.min(st.values.length - 1, Math.floor(S.cur.frac * st.values.length));
      sound.value(st.values[j], st.scale, S.cur.frac * 2 - 1);
    }
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
  return { mn, mx, rms: Math.sqrt(ss / v.length), mi, out: Math.abs(v[mi]) / robustScale(v) };
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
    case 'rope': return { in: ['q (before)', g('q')], out: ['q (after RoPE)', g('q_rot')] };
    case 'scores': return { out: ['scores, head 0', g('scores').subarray(0, sd.meta.kv_len)] };
    case 'softmax': return { out: ['weights, head 0', g('attn').subarray(0, sd.meta.kv_len)] };
    case 'mix': return { out: ['ctx', g('ctx')] };
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

// which earlier words the current word reads (mean over heads)
function readingHTML(sd, L) {
  const kv = sd.meta.kv_len, nH = info.n_heads, A = sd.get(`${L}.attn`);
  const w = new Float32Array(kv);
  for (let h = 0; h < nH; h++) for (let p = 0; p < kv; p++) w[p] += A[h * kv + p] / nH;
  const sink = kv > 1 && w[0] > 0.3;
  const mx = Math.max(...(sink ? w.subarray(1) : w)) || 1;
  return `<div class="reading">${[...w].map((v, p) => {
    const t = escapeHtml(showTok(S.tokens[p]?.text ?? '?'));
    const a = p === 0 && sink ? 0.2 : Math.min(1, v / mx);
    return `<span class="${p === kv - 1 ? 'me' : ''}" style="--a:${a.toFixed(3)}" title="${(v * 100).toFixed(1)}% of attention">${t}</span>`;
  }).join('')}</div>
  ${sink ? `<div class="dim small">The first token soaks up ${(w[0] * 100).toFixed(0)}% of attention. It acts as a resting spot (an "attention sink"), so it is dimmed here to let the other words stand out.</div>` : ''}`;
}

function headRows(sd, L, vals, fmt) {
  const kv = sd.meta.kv_len, nH = info.n_heads, hs = [];
  for (let h = 0; h < nH; h++) {
    const row = vals.subarray(h * kv, (h + 1) * kv);
    const top = [...row.keys()].sort((a, b) => row[b] - row[a]).slice(0, 3);
    const off = isAblated('heads', { layer: L, head: h });
    hs.push(`<div class="head ${off ? 'off' : ''}"><span>h${h}</span>${top.map(p => `<i>${escapeHtml(showTok(S.tokens[p]?.text ?? '?'))}<sub>${fmt(row[p])}</sub></i>`).join('')}<button class="ko" title="${off ? 'Restore' : 'Knock out'} this head for the next re-run" onclick="llmvAblate('heads',${L},${h})">${off ? '↺' : '✕'}</button></div>`);
  }
  return hs.join('');
}

function ropeCanvas(sd, L) {
  const q = sd.get(`${L}.q`), qr = sd.get(`${L}.q_rot`), hd = info.head_dim;
  const c = document.createElement('canvas'); c.width = 360; c.height = 152; c.className = 'ropec';
  const x = c.getContext('2d');
  [0, Math.floor(hd / 8), Math.floor(hd / 4)].forEach((i, k) => {
    const cx = 60 + k * 120, cy = 66, a = [q[i], q[i + hd / 2]], b = [qr[i], qr[i + hd / 2]];
    const m = Math.max(1e-6, Math.hypot(...a), Math.hypot(...b));
    x.strokeStyle = 'rgba(120,150,200,0.3)'; x.beginPath(); x.arc(cx, cy, 48, 0, 7); x.stroke();
    const arrow = (v, col) => { x.strokeStyle = col; x.lineWidth = 2.5; x.beginPath(); x.moveTo(cx, cy); x.lineTo(cx + v[0] / m * 46, cy - v[1] / m * 46); x.stroke(); };
    arrow(a, '#7f9cc4'); arrow(b, '#ffd166');
    const ang = (Math.atan2(b[1], b[0]) - Math.atan2(a[1], a[0])) * 180 / Math.PI;
    x.fillStyle = '#9fb0d0'; x.font = '11px Menlo'; x.textAlign = 'center';
    x.fillText(`dims ${i} & ${i + hd / 2}`, cx, 132); x.fillText(`rotated ${(((ang + 540) % 360) - 180).toFixed(1)}°`, cx, 146);
  });
  return c;
}

function fullAttnCanvas(sd, L, head) {
  const A = sd.get(`${L}.attn_full`), n = sd.meta.n_in, kv = sd.meta.kv_len, nH = info.n_heads;
  if (!A || A.length !== nH * n * kv) return null;
  const c = document.createElement('canvas'); c.width = kv; c.height = n; c.className = 'fullc';
  const x = c.getContext('2d'), img = x.createImageData(kv, n), d = img.data;
  for (let i = 0; i < n; i++) for (let p = 0; p < kv; p++) {
    const v = Math.sqrt(A[(head * n + i) * kv + p]), o = (i * kv + p) * 4;
    d[o] = 20 + v * 235; d[o + 1] = 25 + v * 230; d[o + 2] = 50 + v * 205; d[o + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  return c;
}

function opDetail(ev, sd, T) {
  const m = sd.meta, L = ev.layer, rows = [];
  const pairs = (a, b, c, f) => { const out = []; for (let i = 0; i < 5; i++) out.push(`<div class="eq">${f(a[i], b?.[i], c[i], i)}</div>`); return out.join(''); };
  if (ev.op === 'embed') rows.push(`<div class="eq">token <b>${escapeHtml(showTok(S.tokens[m.kv_len - 1]?.text ?? ''))}</b> → id <b>${m.input_ids.at(-1)}</b> → row ${fmtInt(m.input_ids.at(-1))} of E</div>`);
  if (ev.op === 'norm1' || ev.op === 'norm2' || ev.op === 'final_norm') {
    const x = T.in[1];
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
  if (ev.op === 'rope') rows.push(`<div class="dim">Real q for head 0 before (grey) and after (gold) rotation, position ${m.kv_len - 1}. Low dimension pairs spin fast, high ones slowly:</div><div id="ropeslot"></div>`);
  if (ev.op === 'scores') rows.push(`<div class="dim">Highest raw scores per head (before softmax):</div>${headRows(sd, L, sd.get(`${L}.scores`), v => fmtNum(v, 3))}`);
  if (ev.op === 'softmax' || ev.op === 'mix') {
    rows.push(`<div class="dim">Most-attended earlier tokens per head (real softmax weights). ✕ knocks a head out for the next re-run.</div>${headRows(sd, L, sd.get(`${L}.attn`), v => (v * 100).toFixed(0) + '%')}`);
    if (sd.get(`${L}.attn_full`)) rows.push(`<div class="dim">Full prompt attention pattern, head <select id="fullhead">${Array.from({ length: info.n_heads }, (_, h) => `<option>${h}</option>`).join('')}</select> (rows = query token, columns = key token, causal triangle):</div><div id="fullslot"></div>`);
  }
  if (ev.op === 'sample') {
    rows.push(m.topk.slice(0, 8).map(t => `<div class="eq ${t.id === m.sampled.id ? 'pick' : ''}"><b>${escapeHtml(showTok(t.text))}</b> logit ${fmtNum(t.logit, 5)} → p ${(t.p * 100).toFixed(2)}%</div>`).join(''));
    rows.push(`<div class="eq">sampled → <b class="pick">${escapeHtml(showTok(m.sampled.text))}</b> <button class="linkbtn" onclick="llmvTab('why')">why this word? →</button></div>`);
  }
  return rows.join('');
}

let live = null;
function renderOp(ev, sd) {
  const m = sd.meta, I = opInfo(info, ev, m), T = tensorsFor(ev, sd);
  const [st, sb] = simpleText(info, ev.op);
  const pd = playDur(ev);
  const attnOp = ev.op === 'softmax' || ev.op === 'mix';
  const lens = ev.op === 'add2' ? m.lens[ev.layer][0] : null;
  const handTime = I.macs ? `<div class="hand">${fmtBig(I.macs)} multiply-adds in <b>${fmtTime(ev.dur)}</b> on the GPU. By hand, at one per second: <b>${fmtTime(I.macs)}</b>.</div>` : '';
  $('#op').innerHTML = `
    <div class="sec" style="--c:${GROUP_COLOR[GROUP[ev.op]]}">${I.section}<span>${I.where}</span><span>word ${sd.step + 1}${m.n_in > 1 ? ' · reading the prompt' : ''}</span></div>
    <div class="simple-only">
      <h2>${st}</h2><p class="lead">${sb}</p>
      ${attnOp ? `<div class="dim small">What "${escapeHtml(showTok(S.tokens[m.kv_len - 1]?.text ?? ''))}" is paying attention to:</div>${readingHTML(sd, ev.layer)}` : ''}
      ${lens ? `<div class="guess">Right now the model would say <b>${escapeHtml(showTok(lens.text))}</b> <span>(${(lens.p * 100).toFixed(0)}% sure)</span></div>` : ''}
      ${ev.op === 'sample' ? `<div class="picks">${m.topk.slice(0, 5).map(t => `<div class="${t.id === m.sampled.id ? 'on' : ''}"><span>${escapeHtml(showTok(t.text))}</span><i style="width:${Math.max(1, t.p * 100)}%"></i><em>${(t.p * 100).toFixed(1)}%</em></div>`).join('')}</div>
        <button class="linkbtn" onclick="llmvTab('why')">Why did it pick "${escapeHtml(showTok(m.sampled.text))}"? →</button>` : ''}
      ${handTime}
      <p class="dim small">Technical name: ${I.title}</p>
    </div>
    <div class="expert-only">
      <h2>${I.title}</h2>
      <p>${I.desc}</p>
      <div class="formula">${I.formula}</div>
      <div class="grid">
        <div><label>multiply-adds</label>${I.macs ? fmtBig(I.macs) : '—'}</div>
        <div><label>GPU time</label>${fmtTime(ev.dur)}${isSlow(ev) ? ' <span class="warn" title="Much slower than the same op in other layers">⚠ outlier</span>' : ''}</div>
        <div><label>plays for</label>${fmtTime(pd)}</div>
        <div><label>throughput</label>${I.macs ? fmtBig(I.macs * 2 / ev.dur) + 'FLOP/s' : '—'}</div>
        ${ev.isolated !== undefined ? `<div class="wide"><label>timing detail</label><span class="dim small">raw GPU event ${fmtTime(ev.gpu_event)} · isolated (synced) ${fmtTime(ev.isolated)} · shown = the step's total GPU-event time, split by isolated cost</span></div>` : ''}
      </div>
      ${handTime}
      <div class="nums">
        ${T.in ? `<div class="v"><label>${T.in[0]} (input)</label>${vecPreview(T.in[1])}</div>` : ''}
        ${T.in2 ? `<div class="v"><label>${T.in2[0]}</label>${vecPreview(T.in2[1])}</div>` : ''}
        <div id="liveout"></div>
        ${T.out ? `<div class="v"><label>${T.out[0]} (output)</label>${vecPreview(T.out[1])}<div class="dim">${statLine(T.out[1])}</div></div>` : ''}
        ${attnOp ? `<div class="v"><label>reading (mean over heads)</label>${readingHTML(sd, ev.layer)}</div>` : ''}
        ${opDetail(ev, sd, T)}
      </div>
      ${MATMUL.has(ev.op) ? `<p class="dim small">The GPU computes these ${fmtInt(matDims(info, ev.op)[1])} dot products in parallel tiles across its ${info.gpu_cores || ''} cores. Switch the sweep to "parallel tiles" in the bottom bar to see that ordering.</p>` : ''}
    </div>
    ${I.mat ? `<button id="inspect">🔬 Watch one dot product, multiply by multiply</button><div class="dim small">or click any cell of a matrix / output vector in 3D · hover anything to read its real value</div>` : ''}
    ${ev.layer >= 0 ? `<div class="kobar">Experiment: <button onclick="llmvAblate('attn',${ev.layer})">${isAblated('attn', ev.layer) ? '↺ restore' : '✕ knock out'} attention L${ev.layer}</button><button onclick="llmvAblate('mlp',${ev.layer})">${isAblated('mlp', ev.layer) ? '↺ restore' : '✕ knock out'} MLP L${ev.layer}</button></div>` : ''}`;
  live = { ev, sd, I };
  if (I.mat) $('#inspect').onclick = () => openMicro(ev.layer, I.mat, I.mat === 'lm_head' ? m.sampled.id : currentJ(), sd);
  if (ev.op === 'rope') $('#ropeslot')?.appendChild(ropeCanvas(sd, ev.layer));
  const fh = $('#fullhead');
  if (fh) { const draw = () => { const c = fullAttnCanvas(sd, ev.layer, +fh.value); $('#fullslot').replaceChildren(...(c ? [c] : [])); }; fh.onchange = draw; draw(); }
  updateLive();
}

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
  const evs = sd.meta.events;
  let t = 0, tot = 0;
  for (let i = 0; i < evs.length; i++) { if (i < S.cur.ev) t += evs[i].dur; tot += evs[i].dur; }
  t += ev.dur * S.cur.frac;
  $('#clock').innerHTML = `GPU clock <b>${(t * 1000).toFixed(6)} ms</b> / ${(tot * 1000).toFixed(2)} ms · word ${sd.step + 1}${S.server.length ? '/' + S.server.length : ''}`;
  drawPlayhead(t, tot);
  if (!live || live.ev !== ev) return;
  const lo = $('#liveout');
  if (!lo) return;
  if (MATMUL.has(ev.op) && ev.op !== 'lm_head') {
    const [din, dout] = matDims(info, ev.op);
    const out = tensorsFor(ev, sd).out[1], name = tensorsFor(ev, sd).out[0];
    const j = currentJ();
    const recent = [];
    for (let k = Math.max(0, j - 5); k <= j; k++) recent.push(`<span class="${k === j ? 'cur' : ''}">[${k}] ${fmtNum(out[k], 5)}</span>`);
    lo.innerHTML = `<div class="v live"><label>now computing</label><div class="big">${name}[${fmtInt(j)}] = ${fmtNum(out[j], 6)}</div>
      <div class="dim">dot product of ${fmtInt(din)} weights × ${fmtInt(din)} inputs · output ${fmtInt(j + 1)} of ${fmtInt(dout)}${sd.meta.n_in > 1 ? ` · ×${sd.meta.n_in} rows in prefill` : ''}</div>
      <div class="ticker">${recent.join('')}</div></div>`;
    // at extreme slowdowns each output takes seconds: show its multiply-adds in sync with the clock
    const perCol = playDur(ev) / dout;
    if (perCol > 2.5 && S.speed !== 'guided' && viz.sweep === 'human') {
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

// ------------------------------------------------------------------ right-panel tabs
function setTab(t) {
  S.tab = t;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === t));
  $('#op').classList.toggle('hidden', t !== 'now');
  $('#why').classList.toggle('hidden', t !== 'why');
  $('#lenswrap').classList.toggle('hidden', t !== 'lens');
  if (t === 'why') renderWhy();
}
window.llmvTab = setTab;
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => setTab(b.dataset.t));

// ------------------------------------------------------------------ "why this word?" (exact logit attribution)
function attrRows(sd) {
  const [a, b] = sd.meta.attr;
  if (!S.attrAlt) return a;
  const sub = (x, y) => x.map((v, i) => v - y[i]);
  return { ...a, emb: a.emb - b.emb, attn: sub(a.attn, b.attn), mlp: sub(a.mlp, b.mlp), heads: a.heads.map((r, l) => sub(r, b.heads[l])), total: a.total - b.total, logit: a.logit - b.logit };
}
function renderWhy() {
  const el = $('#why');
  const sd = S.lastSample;
  if (!sd) { el.innerHTML = '<h2>Why this word?</h2><p class="dim">After the model picks a word, this shows which parts of the model pushed it toward that word.</p>'; return; }
  const [a, b] = sd.meta.attr, R = attrRows(sd), nL = info.n_layers;
  const word = escapeHtml(showTok(a.text)), alt = escapeHtml(showTok(b.text));
  const items = [];
  for (let l = 0; l < nL; l++) {
    items.push({ k: 'mlp', l, v: R.mlp[l], name: `MLP, layer ${l}` });
    for (let h = 0; h < info.n_heads; h++) items.push({ k: 'heads', l, h, v: R.heads[l][h], name: `attention head ${h}, layer ${l}` });
  }
  items.push({ k: 'emb', v: R.emb, name: 'the input word itself (embedding)' });
  const top = items.filter(x => x.v > 0).sort((x, y) => y.v - x.v).slice(0, 6);
  const against = items.filter(x => x.v < 0).sort((x, y) => x.v - y.v).slice(0, 3);
  const mx = Math.max(1e-6, ...R.attn.map(Math.abs), ...R.mlp.map(Math.abs));
  const bar = (v, cls) => `<i class="${cls} ${v < 0 ? 'neg' : ''}" style="width:${Math.abs(v) / mx * 100}%"></i>`;
  const row = x => {
    const off = x.k === 'heads' ? isAblated('heads', { layer: x.l, head: x.h }) : x.k === 'mlp' ? isAblated('mlp', x.l) : false;
    const btn = x.k === 'emb' ? '' : `<button class="ko" title="${off ? 'Restore' : 'Knock out'} for the next re-run" onclick="llmvAblate('${x.k}',${x.l}${x.k === 'heads' ? ',' + x.h : ''})">${off ? '↺' : '✕'}</button>`;
    return `<div class="contrib ${off ? 'off' : ''}"><span>${x.name}</span><b class="${x.v < 0 ? 'neg' : 'pos'}">${x.v >= 0 ? '+' : ''}${x.v.toFixed(2)}</b>${btn}</div>`;
  };
  el.innerHTML = `
    <h2>Why "${word}"?</h2>
    <div class="seg small"><button class="${S.attrAlt ? '' : 'on'}" onclick="llmvAttr(false)">score for "${word}"</button><button class="${S.attrAlt ? 'on' : ''}" onclick="llmvAttr(true)">"${word}" vs runner-up "${alt}"</button></div>
    <p class="lead">${S.attrAlt ? `How much each part of the model preferred "${word}" over "${alt}".` : `Every part of the model adds a push toward or away from "${word}". The pushes add up exactly to its final score.`}</p>
    <div class="dim small">Biggest pushes toward "${word}":</div>${top.map(row).join('')}
    ${against.length ? `<div class="dim small" style="margin-top:6px">Pushing against it:</div>${against.map(row).join('')}` : ''}
    <div class="dim small" style="margin-top:8px">Per layer, top to bottom (<span style="color:#4cc9f0">attention</span> | <span style="color:#ff5d8f">MLP</span>):</div>
    <div class="layerbars">${Array.from({ length: nL }, (_, i) => nL - 1 - i).map(l => `<div class="lb"><span>L${l}</span><div class="half">${bar(R.attn[l], 'a')}</div><div class="half">${bar(R.mlp[l], 'm')}</div></div>`).join('')}</div>
    <button id="attr3d" class="${S.attr3d ? 'on' : ''}">${S.attr3d ? 'Hide' : 'Show'} these pushes in 3D</button>
    <p class="dim small expert-only">Exact decomposition: final residual = embedding + Σ attention outputs + Σ MLP outputs. With the final RMSNorm scale frozen at its real value, the logit is linear in that sum, so contributions sum to ${R.total.toFixed(4)} = ${S.attrAlt ? 'logit difference' : 'logit'} ${R.logit.toFixed(4)}. Per-head values split W_O into column blocks.</p>
    <p class="dim small">Click ✕ to knock a part out, then press "Re-run with knockouts" on the left to see if the answer changes.</p>`;
  $('#attr3d').onclick = () => { S.attr3d = !S.attr3d; viz.setAttribution(S.attr3d ? attrRows(sd) : null); renderWhy(); };
}
window.llmvAttr = alt => { S.attrAlt = alt; renderWhy(); if (S.attr3d && S.lastSample) viz.setAttribution(attrRows(S.lastSample)); };

// ------------------------------------------------------------------ logit lens
function renderLens(sd, upto) {
  $('#lens').innerHTML = sd.meta.lens.map((l, i) => {
    const on = i <= upto, t = l[0];
    return `<div class="lr ${on ? '' : 'off'} ${i === upto ? 'now' : ''}" onclick="llmvFloor(${i})"><span>L${i}</span><b>${on ? escapeHtml(showTok(t.text)) : '·'}</b><i style="width:${on ? t.p * 100 : 0}%"></i><em>${on ? (t.p * 100).toFixed(0) + '%' : ''}</em></div>`;
  }).reverse().join('');
}
window.llmvFloor = l => { viz.view('floor', l); paintFollow(); };

// ------------------------------------------------------------------ minimap
const mm = $('#minimap'), mctx = mm.getContext('2d');
let mmLayer = -1;
function drawMinimap(layer) {
  mmLayer = layer;
  const nL = info.n_layers, dpr = devicePixelRatio, W = mm.width = mm.clientWidth * dpr, H = mm.height = mm.clientHeight * dpr;
  if (!W || !H) return;
  mctx.clearRect(0, 0, W, H);
  const rows = nL + 2, rh = H / rows, ev = curEv();
  for (let r = 0; r < rows; r++) {
    const l = rows - 2 - r, isOut = r === 0, isIn = r === rows - 1;
    const io = ev && ev.layer < 0;
    const active = (isIn && io && (ev.op === 'embed' || ev.op === 'prep')) || (isOut && io && ev.op !== 'embed' && ev.op !== 'prep') || (!isIn && !isOut && l === layer);
    const done = !isIn && !isOut && layer >= 0 && l < layer;
    mctx.fillStyle = active ? '#4cc9f0' : done ? 'rgba(76,201,240,0.35)' : 'rgba(90,140,220,0.15)';
    mctx.fillRect(4 * dpr, r * rh + 1, W - 8 * dpr, rh - 2);
  }
  mctx.fillStyle = '#b794ff'; mctx.font = `${8 * dpr}px Menlo`; mctx.textAlign = 'center'; mctx.textBaseline = 'middle';
  mctx.fillText('OUT', W / 2, rh / 2); mctx.fillText('IN', W / 2, H - rh / 2);
}
const mmRow = e => Math.floor(e.offsetY / mm.clientHeight * (info.n_layers + 2));
mm.onclick = e => {
  const r = mmRow(e), rows = info.n_layers + 2;
  if (r === 0) viz.view('top'); else if (r === rows - 1) viz.view('bottom'); else viz.view('floor', rows - 2 - r);
  paintFollow();
};
mm.onmousemove = e => { const r = mmRow(e), rows = info.n_layers + 2; mm.title = r === 0 ? 'Output: LM head and probabilities' : r === rows - 1 ? 'Input: tokens and embedding' : `Layer ${rows - 2 - r}`; };
drawMinimap(-1);

// ------------------------------------------------------------------ hover: real values under the cursor
const tip = $('#hover');
let hoverT = 0, hoverKey = '';
const tileCache = new Map(), tokCache = new Map();
viz.renderer.domElement.addEventListener('pointermove', e => {
  if (e.buttons) { tip.style.display = 'none'; return; }
  const now = performance.now();
  if (now - hoverT < 40) return;
  hoverT = now;
  const h = viz.hoverPick(e.clientX, e.clientY);
  if (!h) { tip.style.display = 'none'; hoverKey = ''; return; }
  tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 18, innerWidth - 340) + 'px';
  tip.style.top = Math.min(e.clientY + 14, innerHeight - 280) + 'px';
  renderHover(h);
});
viz.renderer.domElement.addEventListener('pointerleave', () => { tip.style.display = 'none'; hoverKey = ''; });

async function tokenTexts(ids) {
  const miss = ids.filter(i => !tokCache.has(i));
  if (miss.length) {
    const r = await fetch('/api/tokens?ids=' + miss.join(',')).then(r => r.json());
    miss.forEach((id, k) => tokCache.set(id, r[k]));
  }
  return ids.map(i => tokCache.get(i));
}

function renderHover(h) {
  const where = h.layer >= 0 ? `layer ${h.layer} · ` : '';
  if (h.kind !== 'weight') hoverKey = '';
  if (h.kind === 'vector') {
    const lo = Math.max(0, h.index - 5), hi = Math.min(h.n, h.index + 6), cells = [];
    for (let i = lo; i < hi; i++) cells.push(`<span class="${i === h.index ? 'cur' : ''}">${fmtNum(h.values[i], 4)}</span>`);
    tip.innerHTML = `<div class="ht">${where}${escapeHtml(h.name)}</div><div class="hv">[${fmtInt(h.index)}] = <b>${fmtNum(h.values[h.index], 6)}</b>${h.kindv === 'logit' ? ' <span class="dim">(highest logit in this bin)</span>' : ''}</div>
      <div class="hn">${cells.join('')}</div><div class="dim small">${fmtInt(h.n)} real numbers${h.step !== undefined ? ` · from word ${h.step + 1}` : ''}</div>`;
  } else if (h.kind === 'attn') {
    tip.innerHTML = `<div class="ht">${where}attention head ${h.head} → position ${h.pos}</div><div class="hv">"${escapeHtml(showTok(h.token?.text ?? '?'))}" · <b>${h.signed ? 'raw score ' + fmtNum(h.value, 5) : (h.value * 100).toFixed(2) + '% of attention'}</b></div>`;
  } else if (h.kind === 'cache') {
    tip.innerHTML = `<div class="ht">${where}${h.which}</div><div class="hv">position ${h.pos} "${escapeHtml(showTok(h.token?.text ?? '?'))}" · dim ${h.dim}</div><div class="hv"><b>${fmtNum(h.value, 6)}</b></div>`;
  } else if (h.kind === 'token') {
    tip.innerHTML = `<div class="ht">input token, position ${h.pos}</div><div class="hv"><b>${escapeHtml(showTok(h.token?.text ?? '?'))}</b> · id ${h.token?.id}</div>`;
  } else if (h.kind === 'weight') {
    const key = `${h.layer}:${h.mat}:${h.row}:${h.col}`;
    if (key === hoverKey) return;
    hoverKey = key;
    tip.innerHTML = `<div class="ht">${where}${MAT_NAME[h.mat]} weight</div><div class="hv">row ${fmtInt(h.row)} (output) · column ${fmtInt(h.col)} (input)</div><div class="loupe">loading real weights…</div>`;
    const r0 = Math.max(0, h.row - 4), c0 = Math.max(0, h.col - 4), tk = `${h.layer}:${h.mat}:${r0}:${c0}`;
    const get = tileCache.get(tk) || fetch(`/api/tile?mat=${h.mat}&layer=${h.layer}&r0=${r0}&c0=${c0}&h=9&w=9`).then(r => r.arrayBuffer());
    tileCache.set(tk, get);
    if (tileCache.size > 400) tileCache.delete(tileCache.keys().next().value);
    get.then(async buf => {
      if (hoverKey !== key) return;
      const [R0, C0, hh, ww] = new Int32Array(buf, 0, 4), W = new Float32Array(buf, 16);
      let mx = 0; for (const v of W) mx = Math.max(mx, Math.abs(v));
      const px = new Uint8Array(4), cells = [];
      for (let r = 0; r < hh; r++) for (let c = 0; c < ww; c++) {
        const v = W[r * ww + c]; divColor(v / (mx || 1), px, 0);
        const cur = R0 + r === h.row && C0 + c === h.col;
        cells.push(`<span class="${cur ? 'cur' : ''}" style="background:rgb(${px[0]},${px[1]},${px[2]})">${fmtNum(v, 2)}</span>`);
      }
      const val = W[(h.row - R0) * ww + (h.col - C0)];
      const word = h.mat === 'lm_head' ? ` · token "${escapeHtml(showTok((await tokenTexts([h.row]))[0] ?? '?'))}"` : '';
      if (hoverKey !== key) return;
      tip.querySelector('.loupe').outerHTML = `<div class="hv">W[${fmtInt(h.row)}, ${fmtInt(h.col)}] = <b>${fmtNum(val, 6)}</b>${word}</div>
        <div class="loupe" style="grid-template-columns:repeat(${ww},1fr)">${cells.join('')}</div>
        <div class="dim small">rows ${R0}…${R0 + hh - 1} · columns ${C0}…${C0 + ww - 1}, read from the model's real weights</div>`;
    });
  }
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
  const shown = evIdx >= 0 && (evIdx < S.cur.ev || (evIdx === S.cur.ev && S.cur.started)) ? sd : S.data.get(sd.step - 1);
  if (!shown) return;
  openMicro(hit.layer, hit.mat, hit.mat === 'lm_head' ? shown.meta.sampled.id : hit.j, shown);
});

// ------------------------------------------------------------------ timeline (real per-op GPU timings)
const tl = $('#timeline'), tctx = tl.getContext('2d', { willReadFrequently: true });
let tlCache = null;
function drawTimeline() {
  const sd = curData();
  if (!sd) return;
  const W = tl.width = tl.clientWidth * devicePixelRatio, Hh = tl.height = tl.clientHeight * devicePixelRatio;
  if (!W || !Hh) return;
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
  if (!tlCache || !tl.width) return;
  tctx.putImageData(tlCache.img, 0, 0);
  tctx.fillStyle = '#fff';
  tctx.fillRect(t / tot * tl.width - 1, 0, 2 * devicePixelRatio, tl.height);
}
function tlIndexAt(e) {
  if (!tlCache) return -1;
  const x = (e.offsetX / tl.clientWidth) * tl.width;
  return tlCache.xs.findIndex(([a, w]) => x >= a && x < a + w);
}
tl.addEventListener('mousemove', e => {
  const i = tlIndexAt(e), tp = $('#tltip');
  if (i < 0) { tp.style.display = 'none'; return; }
  const ev = tlCache.evs[i];
  tp.style.display = 'block';
  tp.style.left = Math.min(e.clientX, innerWidth - 220) + 'px';
  tp.innerHTML = `<b>${ev.layer >= 0 ? 'L' + ev.layer + ' · ' : ''}${ev.op}</b> ${fmtTime(ev.dur)}<br><span class="dim">click to jump here</span>`;
});
tl.addEventListener('mouseleave', () => $('#tltip').style.display = 'none');
tl.addEventListener('click', e => { const i = tlIndexAt(e); if (i >= 0) { if (story.on) stopTour(); seek(i); } });
addEventListener('resize', () => { tlCache = null; drawMinimap(mmLayer); });

function updateEta() {
  const sd = curData();
  if (!sd) return;
  const tot = sd.meta.events.reduce((a, e) => a + e.dur, 0);
  const play = sd.meta.events.reduce((a, e) => a + playDur(e), 0);
  $('#eta').innerHTML = `this word: <b>${fmtTime(tot)}</b> of GPU work → plays in <b>${fmtTime(play)}</b>`;
}

// ------------------------------------------------------------------ toggles
$('#lblbtn').onclick = () => { document.body.classList.toggle('nolabels'); $('#lblbtn').classList.toggle('on'); };
$('#panelbtn').onclick = () => { document.body.classList.toggle('nopanels'); viz.applyViewOffset(); if (viz.follow && curEv() && S.cur.started) viz.chooseFocus(curEv(), playDur(curEv())); };
$('#digitsbtn').onclick = () => { viz.digitsOn = !viz.digitsOn; $('#digitsbtn').classList.toggle('on', viz.digitsOn); };
setSpeed(S.mode === 'simple' ? 'guided' : '0.001');
setTab('now');

// handle for poking at the app from the devtools console
window.llmviewer = { S, viz, micro, seek, setPlaying, skipToken, setSpeed, startTour, setMode, sound, setTab };
