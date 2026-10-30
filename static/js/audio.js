// Sonification: the computation as sound. Pitches come from the real numbers being computed.
import { GROUP } from './ops.js';

const PENTA = [0, 2, 4, 7, 9];        // major pentatonic, so any combination of values sounds consonant

function noteFreq(step, base = 220) {
  const oct = Math.floor(step / 5), deg = ((step % 5) + 5) % 5;
  return base * Math.pow(2, oct + PENTA[deg] / 12);
}

export class Sound {
  constructor() { this.on = false; this.ctx = null; this.last = 0; this.budget = 0; }

  enable(on) {
    this.on = on;
    if (on && !this.ctx) this.init();
    if (this.ctx) this.ctx[on ? 'resume' : 'suspend']();
  }

  init() {
    const c = this.ctx = new AudioContext();
    this.master = c.createGain(); this.master.gain.value = 0.32;
    const comp = c.createDynamicsCompressor();
    this.master.connect(comp); comp.connect(c.destination);
    // generated impulse response: a soft, roomy reverb
    const len = c.sampleRate * 2.6, ir = c.createBuffer(2, len, c.sampleRate);
    for (let ch = 0; ch < 2; ch++) { const d = ir.getChannelData(ch); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2); }
    this.verb = c.createConvolver(); this.verb.buffer = ir;
    const wet = c.createGain(); wet.gain.value = 0.55;
    this.verb.connect(wet); wet.connect(this.master);
    this.dry = c.createGain(); this.dry.gain.value = 0.6; this.dry.connect(this.master);
  }

  tone(freq, { dur = 0.35, type = 'sine', gain = 0.12, pan = 0, attack = 0.005, when = 0 } = {}) {
    if (!this.on || !this.ctx) return;
    const c = this.ctx, t = c.currentTime + when;
    const o = c.createOscillator(), g = c.createGain(), p = c.createStereoPanner();
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    p.pan.value = Math.max(-1, Math.min(1, pan));
    o.connect(g); g.connect(p); p.connect(this.dry); p.connect(this.verb);
    o.start(t); o.stop(t + dur + 0.05);
  }

  whoosh(gain = 0.08) {
    if (!this.on || !this.ctx) return;
    const c = this.ctx, t = c.currentTime, len = c.sampleRate * 0.5;
    const b = c.createBuffer(1, len, c.sampleRate), d = b.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const s = c.createBufferSource(); s.buffer = b;
    const f = c.createBiquadFilter(); f.type = 'bandpass'; f.Q.value = 2;
    f.frequency.setValueAtTime(300, t); f.frequency.exponentialRampToValueAtTime(2400, t + 0.45);
    const g = c.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + 0.08); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
    s.connect(f); f.connect(g); g.connect(this.dry); g.connect(this.verb);
    s.start(t); s.stop(t + 0.55);
  }

  // throttle so fast playback doesn't turn into noise
  allow(min = 0.07) {
    const now = this.ctx?.currentTime ?? 0;
    if (now - this.last < min) return false;
    this.last = now;
    return true;
  }

  // one event starting
  event(ev, sd) {
    if (!this.on || !this.allow(0.09)) return;
    const g = GROUP[ev.op], L = Math.max(0, ev.layer);
    const base = 110 * Math.pow(2, (L % 12) / 24);     // drifts slowly upward as the layers climb
    if (ev.op === 'embed') [0, 2, 4].forEach((s, i) => this.tone(noteFreq(s + 5, 165), { dur: 1.2, gain: 0.08, when: i * 0.09 }));
    else if (ev.op === 'sample') {
      const p = sd.meta.sampled.p;
      this.tone(noteFreq(10 + Math.round(p * 5), 220), { dur: 2.4, gain: 0.14, type: 'sine' });
      this.tone(noteFreq(10 + Math.round(p * 5), 220) * 2.76, { dur: 1.6, gain: 0.04 });
    } else if (ev.op === 'softmax') {
      // a chord from the three most-attended positions of head 0
      const kv = sd.meta.kv_len, a = sd.get(`${ev.layer}.attn`).subarray(0, kv);
      [...a.keys()].sort((x, y) => a[y] - a[x]).slice(0, 3).forEach((p, i) => this.tone(noteFreq(p % 10, base * 2), { dur: 1.1, gain: 0.05 + a[p] * 0.08, when: i * 0.04, type: 'triangle' }));
    } else if (g === 'add') this.whoosh(0.05);
    else if (g === 'norm') this.tone(base * 4, { dur: 0.4, gain: 0.035, type: 'sine' });
    else if (g === 'attn') this.tone(base * 2, { dur: 0.5, gain: 0.05, type: 'triangle' });
    else if (g === 'mlp') this.tone(base, { dur: 0.6, gain: 0.06, type: 'triangle' });
  }

  // a value being computed during a matmul sweep: pitch follows the real number
  value(v, scale, pan) {
    if (!this.on || !this.allow(0.085)) return;
    const t = Math.tanh(v / (scale || 1));
    this.tone(noteFreq(Math.round(5 + t * 5), 220), { dur: 0.22, gain: 0.04 + Math.abs(t) * 0.05, pan, type: t < 0 ? 'triangle' : 'sine' });
  }
}

// Spoken narration for the guided tour (the browser's built-in speech).
export class Narrator {
  constructor() { this.on = false; this.voice = null; }
  enable(on) {
    this.on = on;
    if (!on) speechSynthesis.cancel();
    if (on && !this.voice) {
      const pick = () => {
        const vs = speechSynthesis.getVoices();
        this.voice = vs.find(v => /Samantha|Ava|Allison|Serena|Google US English/.test(v.name)) || vs.find(v => v.lang?.startsWith('en')) || null;
      };
      pick();
      speechSynthesis.onvoiceschanged = pick;
    }
  }
  say(text) {
    if (!this.on || !text) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    if (this.voice) u.voice = this.voice;
    u.rate = 1.02; u.pitch = 1.0;
    speechSynthesis.speak(u);
  }
  stop() { speechSynthesis.cancel(); }
}
