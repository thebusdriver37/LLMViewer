// The 3D world: a tower with one floor per transformer layer, built to the model's real dimensions.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { fillRGBA, robustScale, divColor, seqColor, fmtInt, fmtNum, showTok } from './util.js';
import { MATMUL, MAT_IO } from './ops.js';

const FLOOR_H = 27;
const A0 = 1.6;            // attention block base height within a floor
const MAXTEX = 16384;
const MAXPOS = 2048;

const VERT = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FRAG = `
uniform sampler2D map; uniform float reveal; uniform float dim; uniform float glow; uniform float axis; uniform float opacity;
varying vec2 vUv;
void main(){
  vec3 c = texture2D(map, vUv).rgb;
  float coord = axis < 0.5 ? vUv.x : 1.0 - vUv.y;
  float shown = step(coord, reveal + 1e-4);
  vec3 col = c * mix(0.05, 1.0, shown) * dim * (1.0 + glow);
  gl_FragColor = vec4(col, opacity);
}`;

function makeTex(data, w, h, nearest = true) {
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
  t.colorSpace = THREE.SRGBColorSpace;
  t.magFilter = nearest ? THREE.NearestFilter : THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

function valueMat(tex, { axis = 0, opacity = 1, dim = 1 } = {}) {
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: tex }, reveal: { value: 1 }, dim: { value: dim }, glow: { value: 0 }, axis: { value: axis }, opacity: { value: opacity } },
    vertexShader: VERT, fragmentShader: FRAG, transparent: opacity < 1,
  });
}

function label(parent, html, cls, x, y, z = 0) {
  const d = document.createElement('div');
  d.className = 'lbl ' + (cls || '');
  d.innerHTML = html;
  const o = new CSS2DObject(d);
  o.position.set(x, y, z);
  parent.add(o);
  return o;
}

// A vector of real numbers drawn as a strip of colored cells.
class Strip {
  constructor(parent, len, { vertical = false, down = false, thick = 0.24, depth = 0.1, kind = 'div' } = {}) {
    Object.assign(this, { len, vertical, thick, depth, kind });
    this.tex = makeTex(new Uint8Array(4), 1, 1);
    this.mat = valueMat(this.tex, { dim: 0.25 });
    this.mesh = new THREE.Mesh(new THREE.BoxGeometry(len, thick, depth), this.mat);
    // vertical strips run bottom→top; `down` strips run top→bottom (index 0 at the top)
    if (vertical) this.mesh.rotation.z = down ? -Math.PI / 2 : Math.PI / 2;
    this.mesh.userData.strip = this;
    parent.add(this.mesh);
    this.values = null;
  }
  at(x, y, z = 0) { this.mesh.position.set(x, y, z); return this; }
  setData(vals, scale) {
    this.values = vals;
    this.scale = scale || (this.kind === 'logit' ? 1 : robustScale(vals));
    const w = Math.min(vals.length, MAXTEX);
    if (this.tex.image.width !== w) {
      this.tex.dispose();
      this.tex = makeTex(new Uint8Array(w * 4), w, 1);
      this.mat.uniforms.map.value = this.tex;
    }
    fillRGBA(this.tex.image.data, w, vals, this.kind, this.scale);
    this.tex.needsUpdate = true;
  }
  set dim(v) { this.mat.uniforms.dim.value = v; }
  set reveal(v) { this.mat.uniforms.reveal.value = v; }
  set glow(v) { this.mat.uniforms.glow.value = v; }
  point(f, out = new THREE.Vector3()) {
    out.set(-this.len / 2 + f * this.len, 0, this.depth / 2);
    return this.mesh.localToWorld(out);
  }
}

// A weight matrix drawn from a grid of its real weights, with its input and output vectors on two edges.
class Matrix {
  constructor(parent, o) {
    Object.assign(this, o);       // key, layer, din, dout, inAxis, w, h, thumbTex
    this.g = new THREE.Group();
    this.g.position.set(o.x, o.y, 0);
    parent.add(this.g);
    this.plane = new THREE.Mesh(new THREE.PlaneGeometry(o.w, o.h), valueMat(o.thumbTex, { opacity: 0.95, dim: 0.5 }));
    this.plane.position.set(o.w / 2, o.h / 2, 0);
    this.plane.userData.matrix = this;
    this.g.add(this.plane);
    const back = new THREE.Mesh(new THREE.BoxGeometry(o.w + 0.12, o.h + 0.12, 0.08),
      new THREE.MeshBasicMaterial({ color: 0x0a1020 }));
    back.position.set(o.w / 2, o.h / 2, -0.06);
    this.g.add(back);
    this.frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(o.w + 0.12, o.h + 0.12, 0.1)),
      new THREE.LineBasicMaterial({ color: 0x2a4a7a, transparent: true, opacity: 0.8 }));
    this.frame.position.copy(back.position);
    this.g.add(this.frame);
    const y = o.inAxis === 'y';
    if (o.inStrip !== null) {
      this.inStrip = o.inStrip || new Strip(this.g, y ? o.h : o.w, { vertical: y });
      if (!o.inStrip) y ? this.inStrip.at(-0.36, o.h / 2) : this.inStrip.at(o.w / 2, -0.36);
    }
    this.outStrip = new Strip(this.g, y ? o.w : o.h, { vertical: !y, down: true, kind: o.outKind || 'div' });
    y ? this.outStrip.at(o.w / 2, o.h + 0.36) : this.outStrip.at(o.w + 0.36, o.h / 2);
    this.outStrip.mesh.userData.matrix = this;
    const sm = new THREE.MeshBasicMaterial({ color: new THREE.Color(2.2, 2.4, 2.8) });
    this.scan = new THREE.Mesh(new THREE.BoxGeometry(y ? 0.05 : o.w + 0.9, y ? o.h + 0.9 : 0.05, 0.05), sm);
    this.scan.visible = false;
    this.g.add(this.scan);
    this.micro = new THREE.Mesh(new THREE.BoxGeometry(y ? 0.04 : o.w + 0.9, y ? o.h + 0.9 : 0.04, 0.07),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.4, 2.5, 1.2) }));
    this.micro.visible = false;
    this.g.add(this.micro);
  }
  setScan(p) {
    this.scan.visible = p >= 0;
    if (p < 0) return;
    if (this.inAxis === 'y') this.scan.position.set(p * this.w, this.h / 2 + 0.36, 0.06);
    else this.scan.position.set(this.w / 2 + 0.36, this.h * (1 - p), 0.06);
  }
  scanPoint(p, f, out = new THREE.Vector3()) {
    if (this.inAxis === 'y') out.set(p * this.w, f * this.h, 0.05);
    else out.set(f * this.w, this.h * (1 - p), 0.05);
    return this.g.localToWorld(out);
  }
  setMicro(j) {
    this.micro.visible = j !== null;
    if (j === null) return;
    const p = (j + 0.5) / this.dout;
    if (this.inAxis === 'y') this.micro.position.set(p * this.w, this.h / 2 + 0.36, 0.08);
    else this.micro.position.set(this.w / 2 + 0.36, this.h * (1 - p), 0.08);
  }
  indexFromUV(uv) {
    const f = this.inAxis === 'y' ? uv.x : 1 - uv.y;
    return Math.min(this.dout - 1, Math.max(0, Math.floor(f * this.dout)));
  }
}

// Instanced 3D bars rising out of the active output vector.
class Bars {
  constructor(scene, max = 6000) {
    this.max = max;
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ toneMapped: false }), max);
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(max * 3), 3);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
    this.n = 0;
    this._m = new THREE.Matrix4(); this._q = new THREE.Quaternion(); this._p = new THREE.Vector3(); this._s = new THREE.Vector3();
    this._c = new Uint8Array(4);
  }
  clear() { this.n = 0; this.mesh.count = 0; }
  fromStrip(strip, vals, kind = 'div', maxBars = 1100) {
    const n = vals.length, nb = Math.min(n, maxBars, this.max);
    const s = kind === 'logit' ? 1 : (strip.scale || robustScale(vals));
    let lo = Infinity, hi = -Infinity;
    if (kind === 'logit') for (const v of vals) if (v > -1e8) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    strip.mesh.updateWorldMatrix(true, false);
    const W = strip.mesh.matrixWorld;
    const bw = strip.len / nb;
    for (let k = 0; k < nb; k++) {
      const a = Math.floor(k * n / nb), b = Math.max(a + 1, Math.floor((k + 1) * n / nb));
      let v = vals[a];
      for (let i = a + 1; i < b; i++) if (Math.abs(vals[i]) > Math.abs(v)) v = vals[i];
      let t = kind === 'logit' ? Math.pow(Math.max(0, (v - lo) / (hi - lo || 1)), 3) : v / s;
      t = Math.max(-1.6, Math.min(1.6, t));
      const hgt = Math.max(0.01, Math.abs(t) * 1.4);
      this._p.set(-strip.len / 2 + (k + 0.5) * bw, 0, (strip.depth / 2 + hgt / 2) * (t < 0 && kind !== 'logit' ? -1 : 1));
      this._s.set(bw * 0.78, strip.thick * 0.8, hgt);
      this._m.compose(this._p, this._q.identity(), this._s).premultiply(W);
      this.mesh.setMatrixAt(k, this._m);
      kind === 'logit' ? seqColor(t, this._c, 0) : divColor(t, this._c, 0);
      this.mesh.instanceColor.setXYZ(k, this._c[0] / 160, this._c[1] / 160, this._c[2] / 160);
    }
    this.n = nb;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor.needsUpdate = true;
    this.mesh.count = 0;
  }
  fromGrid(plane, rows, cols, vals, w, h) {
    plane.updateWorldMatrix(true, false);
    const W = plane.matrixWorld;
    let k = 0;
    const cw = w / cols, rh = h / rows;
    for (let r = 0; r < rows && k < this.max; r++) {
      for (let c = 0; c < cols && k < this.max; c++, k++) {
        const v = vals[r * cols + c];
        const hgt = Math.max(0.005, Math.sqrt(v) * 2.2);
        this._p.set(-w / 2 + (c + 0.5) * cw, h / 2 - (r + 0.5) * rh, hgt / 2);
        this._s.set(cw * 0.8, rh * 0.8, hgt);
        this._m.compose(this._p, this._q.identity(), this._s).premultiply(W);
        this.mesh.setMatrixAt(k, this._m);
        seqColor(v, this._c, 0);
        this.mesh.instanceColor.setXYZ(k, this._c[0] / 150, this._c[1] / 150, this._c[2] / 150);
      }
    }
    this.n = k;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor.needsUpdate = true;
    this.mesh.count = 0;
  }
  reveal(p) { this.mesh.count = Math.min(this.n, Math.ceil(p * this.n)); }
}

// Glowing particles flowing along the data path of the current operation.
class Flows {
  constructor(scene, n = 700) {
    this.n = n;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const x = c.getContext('2d');
    const gr = x.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.25, 'rgba(255,255,255,0.6)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    x.fillStyle = gr; x.fillRect(0, 0, 64, 64);
    this.mat = new THREE.PointsMaterial({ size: 0.2, map: new THREE.CanvasTexture(c), vertexColors: true, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true });
    this.points = new THREE.Points(this.geo, this.mat);
    this.points.frustumCulled = false;
    scene.add(this.points);
    this.p = Array.from({ length: n }, () => ({ t: 1, wait: 0, a: new THREE.Vector3(), b: new THREE.Vector3(), c: new THREE.Vector3(), col: null, sp: 1 }));
    this.flows = [];
    this._v = new THREE.Vector3();
  }
  set(flows) { this.flows = flows; for (const p of this.p) { p.t = 1; p.wait = Math.random() * 0.8; } }
  update(dt) {
    const pos = this.geo.attributes.position.array, col = this.geo.attributes.color.array;
    const nf = this.flows.length;
    for (let i = 0; i < this.n; i++) {
      const p = this.p[i];
      if (p.wait > 0) { p.wait -= dt; col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = 0; continue; }
      p.t += dt * p.sp;
      if (p.t >= 1 && nf) {
        const f = this.flows[(Math.random() * nf) | 0];
        f.src(p.a); f.dst(p.b);
        const d = p.a.distanceTo(p.b);
        p.c.addVectors(p.a, p.b).multiplyScalar(0.5);
        p.c.z += Math.min(6, 0.25 * d + 0.4);
        p.col = f.color; p.sp = (0.9 + Math.random() * 0.6) / Math.max(0.5, Math.min(2.2, d / 8));
        p.t = 0;
      }
      if (p.t >= 1 || !p.col || !nf) { col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = 0; continue; }
      const t = p.t, u = 1 - t;
      pos[i * 3] = u * u * p.a.x + 2 * u * t * p.c.x + t * t * p.b.x;
      pos[i * 3 + 1] = u * u * p.a.y + 2 * u * t * p.c.y + t * t * p.b.y;
      pos[i * 3 + 2] = u * u * p.a.z + 2 * u * t * p.c.z + t * t * p.b.z;
      const fade = Math.sin(Math.PI * t) * 0.32;
      col[i * 3] = p.col.r * fade; col[i * 3 + 1] = p.col.g * fade; col[i * 3 + 2] = p.col.b * fade;
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate = true;
  }
}

function chipTexture(text, sub, w = 256, h = 96) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const x = c.getContext('2d');
  x.fillStyle = '#0d1630'; x.strokeStyle = '#4cc9f0'; x.lineWidth = 4;
  x.beginPath(); x.roundRect(3, 3, w - 6, h - 6, 14); x.fill(); x.stroke();
  x.fillStyle = '#e8f4ff'; x.textAlign = 'center'; x.textBaseline = 'middle';
  let fs = 38; x.font = `600 ${fs}px Menlo, monospace`;
  while (x.measureText(text).width > w - 20 && fs > 14) { fs -= 2; x.font = `600 ${fs}px Menlo, monospace`; }
  x.fillText(text, w / 2, sub ? h * 0.4 : h / 2);
  if (sub) { x.font = '22px Menlo, monospace'; x.fillStyle = '#7f9cc4'; x.fillText(sub, w / 2, h * 0.78); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

export class Viz {
  constructor(el, info, thumbs) {
    this.info = info;
    this.thumbs = thumbs;
    this.el = el;
    this.s = 4 / info.hidden;
    this.D = 4;
    this.DI = info.intermediate * this.s;
    this.KVD = info.n_kv_heads * info.head_dim;
    this.DK = Math.max(0.5, this.KVD * this.s);

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(r.domElement);
    this.css = new CSS2DRenderer();
    this.css.setSize(el.clientWidth, el.clientHeight);
    Object.assign(this.css.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
    el.appendChild(this.css.domElement);

    const sc = this.scene = new THREE.Scene();
    sc.background = new THREE.Color(0x02030a);
    sc.fog = new THREE.FogExp2(0x02030a, 0.0042);
    this.camera = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 0.1, 5000);
    this.camera.position.set(30, -10, 70);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, -8, 0);
    this.follow = true;
    this.controls.addEventListener('start', () => { this.follow = false; this.onUserCamera?.(); });

    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(sc, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(el.clientWidth, el.clientHeight), 0.85, 0.55, 0.42);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.bars = new Bars(sc);
    this.flows = new Flows(sc);
    this.camGoal = { pos: this.camera.position.clone(), target: this.controls.target.clone() };
    this.labelSets = [];
    this.build();
    sc.updateMatrixWorld(true);
    this.focusBox(this.boxOf([this.embSlab, this.chipGroup, this.floors[0].residIn.mesh]), true);

    addEventListener('resize', () => this.resize());
    this.clock = new THREE.Clock();
    this.tick = this.tick.bind(this);
    this.hooks = [];
    requestAnimationFrame(this.tick);
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.renderer.setSize(w, h); this.css.setSize(w, h); this.composer.setSize(w, h);
    this.camera.aspect = w / h; this.applyViewOffset();
  }

  thumbTex(layer, mat, inAxis) {
    const T = this.info.thumb, mats = this.info.mats;
    const k = layer < 0 ? this.info.n_layers * mats.length : layer * mats.length + mats.indexOf(mat);
    const src = this.thumbs.subarray(k * T * T, (k + 1) * T * T);     // [out][in]
    const out = new Uint8Array(T * T * 4);
    for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) {
      // texture row r (bottom→top), column c (left→right)
      const v = inAxis === 'y' ? src[c * T + r] : src[(T - 1 - r) * T + c];
      divColor((v - 127.5) / 127.5, out, (r * T + c) * 4);
    }
    return makeTex(out, T, T);
  }

  build() {
    const { info, D, DI, DK } = this;
    const sc = this.scene;
    const nL = info.n_layers;
    this.floors = [];
    for (let l = 0; l < nL; l++) this.floors.push(this.buildFloor(l));
    this.top = nL * FLOOR_H;

    // residual-stream spine
    const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, this.top + 4, 12),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(0.15, 0.5, 0.9), transparent: true, opacity: 0.35 }));
    spine.position.set(0, this.top / 2 - 1, -0.7);
    sc.add(spine);
    this.pulse = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(1.2, 2.2, 3) }));
    this.pulse.position.set(0, 0, -0.7);
    sc.add(this.pulse);

    // ---- bottom: tokens + embedding table
    const bot = new THREE.Group(); sc.add(bot);
    const labs = [];
    this.embSlab = new THREE.Mesh(new THREE.PlaneGeometry(D, 11), valueMat(this.thumbTex(-1, 'embed', 'e'), { dim: 0.55 }));
    this.embSlab.position.set(0, -9.5, 0);
    bot.add(this.embSlab);
    const ef = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(D + 0.1, 11.1)), new THREE.LineBasicMaterial({ color: 0x6a5acd }));
    ef.position.copy(this.embSlab.position); bot.add(ef);
    this.embRow = new THREE.Mesh(new THREE.BoxGeometry(D + 0.8, 0.09, 0.12), new THREE.MeshBasicMaterial({ color: new THREE.Color(2.5, 2.0, 3.0) }));
    this.embRow.visible = false; bot.add(this.embRow);
    this.embStrip = new Strip(bot, D).at(0, -2.4);
    labs.push(label(bot, `<b>Embedding table</b><br>${fmtInt(info.vocab)} rows × ${fmtInt(info.hidden)}<br><i>rows compressed for display · real sampled weights</i>`, 'mat', -D / 2 - 0.6, -6, 0));
    labs.push(label(bot, 'embedding vector x⁽⁰⁾', 'vec', D / 2 + 0.5, -2.4));
    this.chipGroup = new THREE.Group(); this.chipGroup.position.set(0, -17, 0); bot.add(this.chipGroup);
    labs.push(label(bot, '<b>INPUT TOKENS</b>', 'big', 0, -15.6));
    this.chips = [];
    this.labelSets.push({ y: -10, labels: labs });

    // ---- top: final norm, LM head, distribution, sampled token
    const T = this.top;
    const tg = new THREE.Group(); tg.position.y = T; sc.add(tg);
    const tl = [];
    const lmW = 40;
    this.lm = new Matrix(tg, { key: 'lm_head', layer: -1, din: info.hidden, dout: info.vocab, inAxis: 'y', x: -lmW / 2, y: 1.8, w: lmW, h: D,
      thumbTex: this.thumbTex(-1, 'embed', 'y'), outKind: 'logit' });
    tl.push(label(tg, `<b>LM head</b> ${info.tied ? '(tied = embedding table)' : ''}<br>${fmtInt(info.vocab)} × ${fmtInt(info.hidden)} · ${(info.vocab * info.hidden / 1e6).toFixed(0)}M weights<br><i>vocab axis compressed</i>`, 'mat', -lmW / 2 + 4, 1.8 + D / 2));
    tl.push(label(tg, 'final RMSNorm output', 'vec', -lmW / 2 - 0.4, 1.8 + D + 0.9));
    tl.push(label(tg, 'logits over all tokens (max-pooled)', 'vec', 0, 1.8 + D + 1.0));
    this.topBars = [];
    this.topLabels = [];
    for (let k = 0; k < 20; k++) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1, 1.0), new THREE.MeshBasicMaterial({ color: 0x334 }));
      m.position.set(-20.9 + k * 2.2, T + 9, 0); m.scale.y = 0.01; sc.add(m);
      this.topBars.push(m);
      this.topLabels.push(label(tg, '', 'tok', -20.9 + k * 2.2, 8.3));
    }
    tl.push(label(tg, '<b>NEXT-TOKEN PROBABILITIES</b> (top 20)', 'big', 0, 20.5));
    this.sampledTex = chipTexture('…', 'sampled token', 512, 160);
    this.sampled = new THREE.Mesh(new THREE.PlaneGeometry(8, 2.5), new THREE.MeshBasicMaterial({ map: this.sampledTex, transparent: true, color: new THREE.Color(1.3, 1.3, 1.3) }));
    this.sampled.position.set(0, T + 23.5, 0); sc.add(this.sampled);
    this.labelSets.push({ y: T + 8, labels: [...tl, ...this.topLabels] });
  }

  buildFloor(l) {
    const { info, D, DI, DK, s } = this;
    const H = info.hidden, I = info.intermediate, KV = this.KVD;
    const g = new THREE.Group(); g.position.y = l * FLOOR_H; this.scene.add(g);
    const F = { l, g };
    const labs = [];
    const L = (...a) => labs.push(label(g, ...a));
    F.residIn = new Strip(g, D).at(0, 0.6);
    F.residMid = new Strip(g, D).at(0, 7.0);
    F.residOut = new Strip(g, D).at(0, 25.0);

    // attention block (left), reading left→right
    const G = 1.5, HW = 6.5;
    const attnW = D + DK + DK + HW + D + 4 * G;
    let x = -3.4 - attnW;
    const xStart = x;
    const mk = (key, din, dout, inAxis, x, y, w, h, extra = {}) =>
      new Matrix(g, { key, layer: l, din, dout, inAxis, x, y, w, h, thumbTex: this.thumbTex(l, key, inAxis), ...extra });
    F.Q = mk('q', H, H, 'y', x, A0, D, D); x += D + G;
    F.K = mk('k', H, KV, 'y', x, A0, DK, D); x += DK + G;
    F.V = mk('v', H, KV, 'y', x, A0, DK, D); x += DK + G;
    F.headsX = x; F.headsW = HW;
    F.headsTex = makeTex(new Uint8Array(4), 1, 1);
    F.heads = new THREE.Mesh(new THREE.PlaneGeometry(HW, D), valueMat(F.headsTex, { axis: 1, dim: 0.3 }));
    F.heads.position.set(x + HW / 2, A0 + D / 2, 0); g.add(F.heads);
    const hf = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(HW + 0.1, D + 0.1)), new THREE.LineBasicMaterial({ color: 0x1f6f8b }));
    hf.position.copy(F.heads.position); g.add(hf);
    x += HW + G;
    F.O = mk('o', H, H, 'y', x, A0, D, D);
    // KV caches grow upward above K and V
    for (const [key, M] of [['kc', F.K], ['vc', F.V]]) {
      const tex = makeTex(new Uint8Array(4), 1, 1);
      const m = new THREE.Mesh(new THREE.PlaneGeometry(DK, 1), valueMat(tex, { dim: 0.8 }));
      m.position.set(M.x + DK / 2, A0 + D + 1.2, 0); m.scale.y = 0.01; g.add(m);
      F[key] = { mesh: m, tex, data: new Float32Array(MAXPOS * KV), rows: 0, x: M.x };
    }

    const sh = (a, b) => `${fmtInt(a)}→${fmtInt(b)}`;
    L(`LAYER ${l}`, 'floor', xStart - 3.2, A0 + 2);
    L('<b>W<sub>Q</sub></b> query<br>' + sh(H, H), 'mat', F.Q.x + D / 2, A0 - 0.9);
    L('<b>W<sub>K</sub></b> key<br>' + sh(H, KV), 'mat', F.K.x + DK / 2, A0 - 0.9);
    L('<b>W<sub>V</sub></b> value<br>' + sh(H, KV), 'mat', F.V.x + DK / 2 + 0.6, A0 - 2.0);
    L(`<b>Attention weights</b><br>${info.n_heads} heads × positions`, 'mat', x - G - HW / 2, A0 - 0.9);
    L('<b>W<sub>O</sub></b> output<br>' + sh(H, H), 'mat', F.O.x + D / 2, A0 - 0.9);
    L('RMSNorm(x)', 'vec', F.Q.x - 0.4, A0 + D + 0.7);
    L('q', 'vec', F.Q.x + D / 2, A0 + D + 0.85);
    L('K cache', 'vec', F.K.x + DK / 2, A0 + D + 0.85);
    L('V cache', 'vec', F.V.x + DK / 2 + 0.4, A0 + D + 1.5);
    L('ctx (heads concat)', 'vec', F.O.x - 0.4, A0 - 2.0);
    L('attn out', 'vec', F.O.x + D / 2, A0 + D + 0.85);
    L('<b>residual stream</b> x (layer input)', 'vec', 0, -0.2);
    L('x + attention', 'vec', 0, 6.2);
    L('x + MLP → next layer', 'vec', 0, 25.8);

    // MLP block (right), stacked upward
    const mx = 4.1;
    F.gate = mk('gate', H, I, 'y', mx, 8.4, DI, D);
    F.act = new Strip(g, DI).at(mx + DI / 2, 8.4 + D + 0.78);
    F.up = mk('up', H, I, 'y', mx, 14.2, DI, D);
    F.mul = new Strip(g, DI).at(mx + DI / 2, 14.2 + D + 0.78);
    F.down = mk('down', I, H, 'x', mx, 19.37, DI, D, { inStrip: F.mul });
    L('<b>W<sub>gate</sub></b> ' + sh(H, I), 'mat', mx + DI / 2, 8.4 + D / 2);
    L('<b>W<sub>up</sub></b> ' + sh(H, I), 'mat', mx + DI / 2, 14.2 + D / 2);
    L('<b>W<sub>down</sub></b> ' + sh(I, H), 'mat', mx + DI / 2, 19.37 + D / 2);
    L('RMSNorm(x)', 'vec', mx - 1.3, 8.4 + D / 2);
    L('gate g', 'vec', mx - 1.0, 8.4 + D + 0.36);
    L('SiLU(g)', 'vec', mx - 1.0, 8.4 + D + 0.8);
    L('up u', 'vec', mx - 1.0, 14.2 + D + 0.36);
    L('h = SiLU(g) ⊙ u', 'vec', mx - 1.6, 14.2 + D + 0.8);
    L('MLP out', 'vec', mx + DI + 1.4, 19.37 + D + 0.4);
    F.lens = label(g, '', 'lens', 2.6, 25.0);
    labs.push(F.lens);

    // floor plate
    const x0 = xStart - 1.5, x1 = mx + DI + 2;
    const plate = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, 0.06, 5),
      new THREE.MeshBasicMaterial({ color: 0x0a1428, transparent: true, opacity: 0.55, depthWrite: false }));
    plate.position.set((x0 + x1) / 2, -0.4, -0.5); g.add(plate);
    F.plateEdge = new THREE.LineSegments(new THREE.EdgesGeometry(plate.geometry), new THREE.LineBasicMaterial({ color: 0x1a3a66, transparent: true, opacity: 0.7 }));
    F.plateEdge.position.copy(plate.position); g.add(F.plateEdge);
    F.box = { x0, x1 };

    F.strips = [F.residIn, F.residMid, F.residOut, F.act, F.mul];
    for (const M of [F.Q, F.K, F.V, F.O, F.gate, F.up, F.down]) {
      if (M.inStrip && M.inStrip !== F.mul) F.strips.push(M.inStrip);
      F.strips.push(M.outStrip);
    }
    F.mats = { q: F.Q, k: F.K, v: F.V, o: F.O, gate: F.gate, up: F.up, down: F.down };
    this.labelSets.push({ y: l * FLOOR_H + 12, labels: labs, floor: l });
    return F;
  }

  // ------------------------------------------------------------------ tokens at the bottom
  setTokens(tokens, activeFrom, activeTo) {
    const per = 14, cw = 1.75, ch = 0.66;
    while (this.chips.length < tokens.length) {
      const t = tokens[this.chips.length];
      const m = new THREE.Mesh(new THREE.PlaneGeometry(cw - 0.12, ch), new THREE.MeshBasicMaterial({ map: chipTexture(showTok(t.text), '#' + t.id), transparent: true }));
      this.chipGroup.add(m);
      this.chips.push(m);
    }
    this.chips.forEach((m, i) => {
      const r = Math.floor(i / per), c = i % per;
      m.position.set(-((per - 1) * cw) / 2 + c * cw, -r * (ch + 0.12), 0);
      const on = i >= activeFrom && i < activeTo;
      m.material.color.setScalar(on ? (i === activeTo - 1 ? 2.2 : 1.3) : 0.45);
      m.position.z = on && i === activeTo - 1 ? 0.4 : 0;
    });
    this.activeChip = this.chips[activeTo - 1];
  }
  resetTokens() { for (const m of this.chips) { this.chipGroup.remove(m); m.material.map.dispose(); } this.chips = []; }

  // ------------------------------------------------------------------ per-step state
  resetCaches() {
    for (const F of this.floors) for (const k of ['kc', 'vc']) { F[k].rows = 0; F[k].mesh.scale.y = 0.01; }
  }
  beginStep() {
    for (const F of this.floors) {
      for (const s of F.strips) { s.dim = 0.22; s.reveal = 1; s.glow = 0; }
      F.heads.material.uniforms.dim.value = 0.3;
      F.lens.element.classList.add('stale');
    }
    this.embStrip.dim = 0.22; this.lm.outStrip.dim = 0.22; this.lm.inStrip.dim = 0.22;
    for (const b of this.topBars) b.material.color.multiplyScalar(0.4);
  }

  updateCache(F, key, rowsVals, base, n) {
    const C = F[key], KV = this.KVD;
    if (base + n > MAXPOS) return;
    C.data.set(rowsVals.subarray(0, n * KV), base * KV);
    C.rows = base + n;
    const rows = C.rows;
    const img = new Uint8Array(KV * rows * 4);
    const sc = robustScale(C.data.subarray(0, rows * KV));
    // texture rows go bottom→top, so the oldest position is at the bottom and the newest on top
    for (let i = 0; i < rows * KV; i++) divColor(C.data[i] / sc, img, i * 4);
    C.tex.dispose();
    C.tex = makeTex(img, KV, rows);
    C.mesh.material.uniforms.map.value = C.tex;
    const hgt = Math.min(rows * 0.06, 9);
    C.mesh.scale.y = hgt;
    C.mesh.position.y = A0 + this.D + 1.2 + hgt / 2;
  }

  setHeads(F, attn, kv) {
    const nH = this.info.n_heads;
    const img = new Uint8Array(kv * nH * 4);
    for (let h = 0; h < nH; h++) for (let p = 0; p < kv; p++) seqColor(attn[h * kv + p], img, ((nH - 1 - h) * kv + p) * 4);
    F.headsTex.dispose();
    F.headsTex = makeTex(img, kv, nH);
    F.heads.material.uniforms.map.value = F.headsTex;
  }

  // ------------------------------------------------------------------ events
  // Apply an operation's real output tensors to the scene and return what to animate.
  apply(ev, sd, animate) {
    const L = ev.layer, F = this.floors[L], meta = sd.meta;
    const g = n => sd.get(`${L}.${n}`);
    const KV = this.KVD, n = meta.n_in, base = meta.kv_len - meta.n_in;
    const set = (strip, vals, kind) => { if (kind) strip.kind = kind; strip.setData(vals); strip.dim = 1; strip.reveal = animate ? 0 : 1; return strip; };
    const spec = { reveal: [], glow: [], flows: [], focus: [], bars: null, mat: null };
    const C = (r, gg, b) => new THREE.Color(r, gg, b);
    const cyan = C(0.5, 1.6, 2.4), pink = C(2.4, 0.6, 1.4), gold = C(2.4, 1.8, 0.5), green = C(0.7, 2.2, 0.9), violet = C(1.6, 1.0, 2.6);
    const seg = (strip) => (o) => strip.point(Math.random(), o);
    const pt = (strip, f) => (o) => strip.point(f, o);
    const meshPt = (mesh, w, h) => (o) => { o.set((Math.random() - 0.5) * w, (Math.random() - 0.5) * h, 0.05); return mesh.localToWorld(o); };
    const matmul = (M, color) => {
      spec.mat = M; spec.focus.push(M.g); spec.glow.push(M.inStrip);
      spec.flows.push({ src: seg(M.inStrip), dst: (o) => M.scanPoint(this.scanP ?? 0, Math.random(), o), color });
      spec.flows.push({ src: (o) => M.scanPoint(this.scanP ?? 0, Math.random(), o), dst: (o) => M.outStrip.point(this.scanP ?? 0, o), color: C(2, 2, 2.4) });
    };
    switch (ev.op) {
      case 'embed': {
        const v = sd.get('emb');
        spec.reveal.push(set(this.embStrip, v), set(this.floors[0].residIn, v));
        spec.bars = this.embStrip;
        const id = meta.input_ids ? meta.input_ids.at(-1) : 0;
        this.embRow.position.set(0, -9.5 + 5.5 - 11 * (id / this.info.vocab), 0.08);
        this.embRow.visible = true;
        const rowPt = (o) => o.set((Math.random() - 0.5) * this.D, this.embRow.position.y, 0.1);
        if (this.activeChip) spec.flows.push({ src: (o) => this.activeChip.getWorldPosition(o), dst: rowPt, color: violet });
        spec.flows.push({ src: rowPt, dst: seg(this.embStrip), color: violet });
        spec.focus.push(this.embSlab, this.embStrip.mesh, this.chipGroup);
        break;
      }
      case 'prep':
        spec.focus.push(this.embStrip.mesh, this.floors[0].residIn.mesh, this.embSlab);
        spec.flows.push({ src: seg(this.embStrip), dst: seg(this.floors[0].residIn), color: violet });
        break;
      case 'norm1': {
        const v = g('norm1');
        spec.reveal.push(set(F.Q.inStrip, v), set(F.K.inStrip, v), set(F.V.inStrip, v));
        spec.bars = F.Q.inStrip; spec.glow.push(F.residIn);
        for (const M of [F.Q, F.K, F.V]) spec.flows.push({ src: seg(F.residIn), dst: seg(M.inStrip), color: gold });
        spec.focus.push(F.residIn.mesh, F.Q.g, F.V.g);
        break;
      }
      case 'q': spec.reveal.push(set(F.Q.outStrip, g('q'))); spec.bars = F.Q.outStrip; matmul(F.Q, cyan); break;
      case 'k': case 'v': {
        const M = ev.op === 'k' ? F.K : F.V, all = g(ev.op);
        spec.reveal.push(set(M.outStrip, all.subarray((n - 1) * KV, n * KV)));
        spec.bars = M.outStrip; matmul(M, cyan);
        this.updateCache(F, ev.op === 'k' ? 'kc' : 'vc', all, base, n);
        spec.focus.push(F[ev.op === 'k' ? 'kc' : 'vc'].mesh);
        break;
      }
      case 'attn': {
        this.setHeads(F, g('attn'), meta.kv_len);
        F.heads.material.uniforms.dim.value = 1;
        F.heads.material.uniforms.reveal.value = animate ? 0 : 1;
        spec.headsReveal = F.heads;
        spec.reveal.push(set(F.O.inStrip, g('ctx')));
        spec.grid = [F.heads, this.info.n_heads, meta.kv_len, g('attn'), F.headsW, this.D];
        const hp = meshPt(F.heads, F.headsW, this.D);
        spec.flows.push({ src: seg(F.Q.outStrip), dst: hp, color: cyan });
        spec.flows.push({ src: meshPt(F.kc.mesh, this.DK, 1), dst: hp, color: C(0.6, 2.2, 1.8) });
        spec.flows.push({ src: meshPt(F.vc.mesh, this.DK, 1), dst: hp, color: C(1.6, 1.2, 2.6) });
        spec.flows.push({ src: hp, dst: seg(F.O.inStrip), color: cyan });
        spec.focus.push(F.heads, F.Q.outStrip.mesh, F.kc.mesh, F.vc.mesh, F.O.inStrip.mesh);
        break;
      }
      case 'o': spec.reveal.push(set(F.O.outStrip, g('o'))); spec.bars = F.O.outStrip; matmul(F.O, cyan); break;
      case 'add1':
        spec.reveal.push(set(F.residMid, g('resid_mid'))); spec.bars = F.residMid; spec.glow.push(F.O.outStrip, F.residIn);
        spec.flows.push({ src: seg(F.O.outStrip), dst: seg(F.residMid), color: green }, { src: seg(F.residIn), dst: seg(F.residMid), color: green });
        spec.focus.push(F.residMid.mesh, F.residIn.mesh, F.O.g);
        break;
      case 'norm2': {
        const v = g('norm2');
        spec.reveal.push(set(F.gate.inStrip, v), set(F.up.inStrip, v)); spec.bars = F.gate.inStrip; spec.glow.push(F.residMid);
        spec.flows.push({ src: seg(F.residMid), dst: seg(F.gate.inStrip), color: gold }, { src: seg(F.residMid), dst: seg(F.up.inStrip), color: gold });
        spec.focus.push(F.residMid.mesh, F.gate.inStrip.mesh, F.up.inStrip.mesh);
        break;
      }
      case 'gate': spec.reveal.push(set(F.gate.outStrip, g('gate'))); spec.bars = F.gate.outStrip; matmul(F.gate, pink); break;
      case 'act':
        spec.reveal.push(set(F.act, g('act'))); spec.bars = F.act; spec.glow.push(F.gate.outStrip);
        spec.flows.push({ src: seg(F.gate.outStrip), dst: seg(F.act), color: pink });
        spec.focus.push(F.act.mesh, F.gate.outStrip.mesh);
        break;
      case 'up': spec.reveal.push(set(F.up.outStrip, g('up'))); spec.bars = F.up.outStrip; matmul(F.up, pink); break;
      case 'mul':
        spec.reveal.push(set(F.mul, g('mul'))); spec.bars = F.mul; spec.glow.push(F.act, F.up.outStrip);
        spec.flows.push({ src: seg(F.act), dst: seg(F.mul), color: pink }, { src: seg(F.up.outStrip), dst: seg(F.mul), color: pink });
        spec.focus.push(F.mul.mesh, F.up.outStrip.mesh, F.act.mesh);
        break;
      case 'down': spec.reveal.push(set(F.down.outStrip, g('down'))); spec.bars = F.down.outStrip; matmul(F.down, pink); break;
      case 'add2': {
        const v = g('resid_out');
        spec.reveal.push(set(F.residOut, v)); spec.bars = F.residOut; spec.glow.push(F.down.outStrip, F.residMid);
        if (L + 1 < this.floors.length) spec.reveal.push(set(this.floors[L + 1].residIn, v));
        spec.flows.push({ src: seg(F.down.outStrip), dst: seg(F.residOut), color: green }, { src: seg(F.residMid), dst: seg(F.residOut), color: green });
        spec.focus.push(F.residOut.mesh, F.residMid.mesh, F.down.outStrip.mesh);
        break;
      }
      case 'final_norm': {
        const last = this.floors.at(-1);
        spec.reveal.push(set(this.lm.inStrip, sd.get('final_norm'))); spec.bars = this.lm.inStrip;
        spec.flows.push({ src: seg(last.residOut), dst: seg(this.lm.inStrip), color: gold });
        spec.focus.push(last.residOut.mesh, this.lm.inStrip.mesh);
        break;
      }
      case 'lm_head': spec.reveal.push(set(this.lm.outStrip, sd.get('logits_pooled'), 'logit')); spec.bars = this.lm.outStrip; matmul(this.lm, violet); break;
      case 'sample': {
        this.setTopK(meta, animate);
        spec.flows.push({ src: seg(this.lm.outStrip), dst: (o) => this.topBars[(Math.random() * 20) | 0].getWorldPosition(o), color: violet });
        spec.focus.push(this.topBars[0], this.topBars[19], this.sampled, this.lm.outStrip.mesh);
        spec.topk = true;
        break;
      }
    }
    if (ev.op === 'add2') {
      const top = meta.lens[L][0];
      F.lens.element.innerHTML = `logit lens → <b>${escapeHtml(showTok(top.text))}</b> ${(top.p * 100).toFixed(0)}%`;
      F.lens.element.classList.remove('stale');
    }
    return spec;
  }

  setTopK(meta, animate) {
    const tk = meta.topk;
    const maxp = tk[0].p;
    tk.forEach((t, k) => {
      const b = this.topBars[k];
      const isS = t.id === meta.sampled.id;
      b.userData.h = Math.max(0.02, (t.p / maxp) * 10);
      b.material.color.set(isS ? new THREE.Color(1.5, 1.25, 2.0) : new THREE.Color(0.35 + t.p * 0.6, 0.25 + t.p * 0.7, 0.9 + t.p * 0.5));
      this.topLabels[k].element.innerHTML = `<span>${escapeHtml(showTok(t.text))}</span><em>${(t.p * 100).toFixed(t.p < 0.01 ? 2 : 1)}%</em>`;
      this.topLabels[k].element.classList.toggle('picked', isS);
      if (!animate) { b.scale.y = b.userData.h; b.position.y = this.top + 9 + b.userData.h / 2; }
    });
    const tex = chipTexture(showTok(meta.sampled.text), `sampled · p=${(meta.sampled.p * 100).toFixed(1)}%`, 512, 160);
    this.sampled.material.map.dispose(); this.sampled.material.map = tex;
  }

  begin(ev, sd, playSec) {
    this.end();
    const spec = this.apply(ev, sd, true);
    this.spec = spec;
    this.scanP = 0;
    for (const s of spec.glow) s.glow = 0.7;
    if (spec.mat) spec.mat.setScan(0);
    if (spec.bars) this.bars.fromStrip(spec.bars, spec.bars.values, spec.bars.kind);
    else if (spec.grid) this.bars.fromGrid(...spec.grid);
    else this.bars.clear();
    this.flows.set(spec.flows);
    this.layerFocus = ev.layer;
    this.chooseFocus(ev, playSec);
    if (ev.layer >= 0) this.highlightFloor(ev.layer);
  }

  progress(p) {
    const sp = this.spec;
    if (!sp) return;
    this.scanP = p;
    for (const s of sp.reveal) s.reveal = p;
    if (sp.headsReveal) sp.headsReveal.material.uniforms.reveal.value = p;
    if (sp.mat) sp.mat.setScan(p);
    this.bars.reveal(p);
    if (sp.topk) for (const b of this.topBars) { const h = (b.userData.h || 0.02) * Math.min(1, p * 1.2); b.scale.y = Math.max(0.01, h); b.position.y = this.top + 9 + h / 2; }
  }

  end() {
    const sp = this.spec;
    if (!sp) return;
    this.progress(1);
    for (const s of sp.glow) s.glow = 0;
    if (sp.mat) sp.mat.setScan(-1);
    this.spec = null;
  }

  instant(ev, sd) { this.apply(ev, sd, false); }

  highlightFloor(l) {
    if (this.hlFloor === l) return;
    if (this.hlFloor !== undefined) this.floors[this.hlFloor].plateEdge.material.color.set(0x1a3a66);
    this.floors[l].plateEdge.material.color.set(new THREE.Color(0.4, 1.2, 2.2));
    this.hlFloor = l;
  }

  // ------------------------------------------------------------------ camera
  boxOf(objs) {
    const b = new THREE.Box3();
    for (const o of objs) if (o) b.expandByObject(o);
    return b;
  }
  floorBox(l) {
    const F = this.floors[l];
    return new THREE.Box3(new THREE.Vector3(F.box.x0, l * FLOOR_H - 1, -1), new THREE.Vector3(F.box.x1, l * FLOOR_H + FLOOR_H - 1, 1));
  }
  towerBox() {
    const F = this.floors[0];
    return new THREE.Box3(new THREE.Vector3(F.box.x0, -18, -1), new THREE.Vector3(F.box.x1, this.top + 25, 1));
  }
  chooseFocus(ev, playSec) {
    this.lastEv = ev; this.lastPlay = playSec;
    if (!this.follow) return;
    if (playSec < 0.03) this.towerView();
    else if (playSec < 0.5 && ev.layer >= 0) this.focusBox(this.floorBox(ev.layer));
    else if (this.spec) this.focusBox(this.boxOf(this.spec.focus));
  }
  // Screen area not covered by the HTML panels; the camera centers and fits within it.
  freeArea() {
    const W = this.el.clientWidth, H = this.el.clientHeight;
    const r = id => { const e = document.getElementById(id); return e && getComputedStyle(e).display !== 'none' ? e.getBoundingClientRect() : null; };
    const L = r('left'), R = r('right'), B = r('bar');
    const x0 = L ? L.right : 0, x1 = R ? R.left : W, y0 = 44, y1 = B ? B.top : H;
    return { W, H, x0, x1, y0, y1, w: Math.max(200, x1 - x0), h: Math.max(200, y1 - y0) };
  }
  applyViewOffset() {
    const a = this.freeArea();
    const dx = a.W / 2 - (a.x0 + a.x1) / 2, dy = a.H / 2 - (a.y0 + a.y1) / 2;
    this.camera.setViewOffset(a.W, a.H, dx, dy, a.W, a.H);
    return a;
  }
  focusBox(box, snap = false) {
    const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    s.x = Math.max(s.x, 9); s.y = Math.max(s.y, 7);
    const a = this.applyViewOffset();
    const fov = this.camera.fov * Math.PI / 180;
    const fit = Math.max(s.y * a.H / a.h, (s.x / this.camera.aspect) * a.W / a.w) / (2 * Math.tan(fov / 2));
    const dist = fit * 1.1 + 4;
    const dir = new THREE.Vector3(0.28, 0.2, 1).normalize();
    this.camGoal.target.copy(c);
    this.camGoal.pos.copy(c).addScaledVector(dir, dist);
    if (snap) { this.camera.position.copy(this.camGoal.pos); this.controls.target.copy(c); }
  }
  // whole model, seen from low and to the side so the tall tower reads in perspective
  towerView() {
    const a = this.applyViewOffset();
    const F = this.floors[0];
    const cx = (F.box.x0 + F.box.x1) / 2;
    this.camGoal.target.set(cx, this.top * 0.42, 0);
    this.camGoal.pos.set(cx + 150, -40, 210 * Math.max(1, 1.3 * a.W / a.w / this.camera.aspect));
  }
  setFollow(on) {
    this.follow = on;
    if (on && this.lastEv && this.spec) this.chooseFocus(this.lastEv, this.lastPlay);
  }
  view(name) {
    this.follow = false;
    if (name === 'tower') this.towerView();
    if (name === 'floor') this.focusBox(this.floorBox(this.hlFloor ?? 0));
    this.goTo = true;
  }

  pick(clientX, clientY) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(v, this.camera);
    const targets = [];
    for (const F of this.floors) for (const M of Object.values(F.mats)) targets.push(M.plane, M.outStrip.mesh);
    targets.push(this.lm.plane, this.lm.outStrip.mesh);
    const hit = rc.intersectObjects(targets, false)[0];
    if (!hit) return null;
    const M = hit.object.userData.matrix;
    // output index 0 is at the left (horizontal outputs) or the top (vertical outputs)
    const j = hit.object === M.plane ? M.indexFromUV(hit.uv)
      : Math.min(M.dout - 1, Math.max(0, Math.floor(hit.uv.x * M.dout)));
    return { layer: M.layer, mat: M.key, j };
  }

  setMicro(layer, mat, j, iFrac) {
    if (this.microM) this.microM.setMicro(null);
    if (!this.microDot) {
      this.microDot = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 10), new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 3, 1.5) }));
      this.scene.add(this.microDot);
    }
    if (mat === null) { this.microM = null; this.microDot.visible = false; return; }
    const M = mat === 'lm_head' ? this.lm : this.floors[layer].mats[mat];
    this.microM = M;
    M.setMicro(mat === 'lm_head' ? null : j);
    this.microDot.visible = iFrac !== null;
    if (iFrac !== null) M.inStrip.point(iFrac, this.microDot.position);
  }

  // ------------------------------------------------------------------ render loop
  tick() {
    requestAnimationFrame(this.tick);
    const dt = Math.min(0.1, this.clock.getDelta());
    for (const h of this.hooks) h(dt);
    if (this.follow || this.goTo) {
      const k = 1 - Math.exp(-dt * 2.6);
      this.camera.position.lerp(this.camGoal.pos, k);
      this.controls.target.lerp(this.camGoal.target, k);
      if (this.goTo && this.camera.position.distanceTo(this.camGoal.pos) < 0.5) this.goTo = false;
    }
    this.controls.update();
    this.flows.update(dt);
    if (this.hlFloor !== undefined) {
      const ty = this.spec && this.lastEv?.layer >= 0 ? this.hlFloor * FLOOR_H + (this.lastEv.op === 'add2' ? 25 : this.lastEv.op === 'add1' ? 7 : 0.6) : this.pulse.position.y;
      this.pulse.position.y += (ty - this.pulse.position.y) * (1 - Math.exp(-dt * 6));
    }
    // labels: only the floors near what the camera is looking at
    const ty = this.controls.target.y, dist = this.camera.position.distanceTo(this.controls.target);
    this.scene.fog.density = Math.min(0.0042, 0.55 / dist);
    for (const set of this.labelSets) {
      const vis = Math.abs(set.y - ty) < FLOOR_H * 0.9 && dist < 110;
      if (set.vis !== vis) { set.vis = vis; for (const o of set.labels) o.visible = vis; }
    }
    this.composer.render();
    this.css.render(this.scene, this.camera);
  }
}

function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
export { FLOOR_H, escapeHtml };
