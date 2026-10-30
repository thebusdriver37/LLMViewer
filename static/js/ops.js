// What each recorded GPU operation is, in words and math, with this model's real dimensions.
import { fmtInt } from './util.js';

export const MATMUL = new Set(['q', 'k', 'v', 'o', 'gate', 'up', 'down', 'lm_head']);

export const GROUP = {
  embed: 'io', prep: 'io', final_norm: 'io', lm_head: 'io', sample: 'io',
  norm1: 'norm', norm2: 'norm',
  q: 'attn', k: 'attn', v: 'attn', rope: 'attn', scores: 'attn', softmax: 'attn', mix: 'attn', o: 'attn',
  add1: 'add', add2: 'add',
  gate: 'mlp', act: 'mlp', up: 'mlp', mul: 'mlp', down: 'mlp',
};

export const GROUP_COLOR = { io: '#b794ff', norm: '#ffd166', attn: '#4cc9f0', add: '#7ae582', mlp: '#ff5d8f' };

// Which captured tensor feeds each matmul (the "x" in W·x) and which holds its output.
export const MAT_IO = {
  q: ['norm1', 'q'], k: ['norm1', 'k'], v: ['norm1', 'v'], o: ['ctx', 'o'],
  gate: ['norm2', 'gate'], up: ['norm2', 'up'], down: ['mul', 'down'], lm_head: ['final_norm', null],
};

export const MAT_NAME = { q: 'W_Q', k: 'W_K', v: 'W_V', o: 'W_O', gate: 'W_gate', up: 'W_up', down: 'W_down', lm_head: 'LM head', embed: 'Embedding' };

export function matDims(info, op) {
  const H = info.hidden, I = info.intermediate, KV = info.n_kv_heads * info.head_dim;
  return {
    q: [H, H], k: [H, KV], v: [H, KV], o: [H, H],
    gate: [H, I], up: [H, I], down: [I, H], lm_head: [H, info.vocab],
  }[op];
}

// Everyday-language version of each step, for people who have never seen a neural network.
export function simpleText(info, op) {
  const H = fmtInt(info.hidden), I = fmtInt(info.intermediate), V = fmtInt(info.vocab), nH = info.n_heads;
  return {
    embed: ['Words become numbers', `Each word piece has its own list of ${H} learned numbers, like coordinates for its meaning. The model looks that list up.`],
    prep: ['Where is each word?', 'The model notes the position of every word, so "dog bites man" means something different from "man bites dog".'],
    norm1: ['Steady the volume', 'All the numbers are scaled to a steady level so none of them gets too loud before the next step.'],
    q: ['Ask a question', 'The current word writes a question: "what am I looking for in the words before me?"'],
    k: ['Make a label', 'The word makes a label for itself: "here is what I contain." Every earlier word already has one.'],
    v: ['Pack the information', 'The word packs up the information it will hand over if another word looks at it.'],
    rope: ['Stamp the position', 'The question and the label are rotated by position, so the model can tell how far apart two words are.'],
    scores: ['Compare with every word', 'The question is compared with the label of every earlier word. A higher score means a better match.'],
    softmax: ['Decide where to look', `The scores become percentages that add up to 100%. Each of the ${nH} "heads" looks at the sentence in its own way.`],
    mix: ['Gather information', 'The word collects information from the words it is paying attention to, mostly from the strongest matches.'],
    o: ['Combine the findings', `The ${nH} heads' findings are blended into one update.`],
    add1: ['Add, don\'t replace', 'The update is added to the word\'s running meaning. Nothing is thrown away; it just gets richer.'],
    norm2: ['Steady the volume again', 'Another rescale, ready for the thinking step.'],
    gate: ['Check the feature detectors', `The word is checked against ${I} feature detectors. Which ones should switch on?`],
    act: ['Switch features on or off', 'Weak and negative signals are mostly switched off. Strong ones pass through.'],
    up: ['Prepare what each feature adds', `A second set of ${I} numbers carries what each feature would contribute.`],
    mul: ['Let only active features through', 'Switch × signal: only the features that switched on get through.'],
    down: ['Recall knowledge', 'The active features are squeezed back into an update. This is where stored knowledge, like facts, tends to come out.'],
    add2: ['Layer done, pass it up', 'The update joins the running meaning and goes up to the next layer. The glowing word beside the layer shows what the model would say right now.'],
    final_norm: ['Final tidy-up', 'One last rescale after all the layers.'],
    lm_head: ['Score every word it knows', `The final meaning is compared with all ${V} word pieces in the vocabulary. Every one gets a score.`],
    sample: ['Pick the next word', 'Scores become probabilities and one word is picked. It is added to the answer and fed back in to predict the one after.'],
  }[op] || [op, ''];
}

export function opInfo(info, ev, meta) {
  const H = info.hidden, I = info.intermediate, nH = info.n_heads, nKV = info.n_kv_heads, hd = info.head_dim;
  const KV = nKV * hd, V = info.vocab, n = meta.n_in, kv = meta.kv_len;
  const L = ev.layer;
  const where = L >= 0 ? `Layer ${L} of ${info.n_layers}` : '';
  const bias = m => info.has_bias[m] ? ' + b' : '';
  const T = (title, section, desc, formula, macs, extra = {}) => ({ title, section, where, desc, formula, macs, ...extra });
  const rows = n > 1 ? `${n} tokens × ` : '';
  switch (ev.op) {
    case 'embed': return T('Token embedding lookup', 'Input',
      n > 1
        ? `Prefill: all ${n} prompt tokens are looked up at once. Each token id selects one row of the ${fmtInt(V)} × ${fmtInt(H)} embedding table. No multiplication happens here, just a memory read. The view follows the last token, because it predicts the next word.`
        : `The token just generated is fed back in. Its id selects one row of the ${fmtInt(V)} × ${fmtInt(H)} embedding table: ${fmtInt(H)} learned numbers that represent this token.`,
      `x⁽⁰⁾ = E[token_id]   (${rows}${fmtInt(H)})`, 0);
    case 'prep': return T('Position setup (RoPE tables + causal mask)', 'Input',
      `Computes the rotary position embedding (RoPE) cos/sin tables for position${n > 1 ? 's 0…' + (kv - 1) : ' ' + (kv - 1)}. No learned weights.`,
      `θᵢ = pos · ${info.rope_theta ?? 10000}^(−2i/${hd})`, 0);
    case 'norm1': return T('RMSNorm (before attention)', 'Attention',
      'Rescales the residual stream to unit root-mean-square, then multiplies by a learned gain per dimension. This keeps numbers in a stable range before the matrix multiplies.',
      `y = x / √(mean(x²) + ${info.rms_eps}) ⊙ g`, n * H);
    case 'q': return T('Query projection  W_Q', 'Attention',
      `Every output number is a dot product of a ${fmtInt(H)}-number weight row with the normalized input. The ${fmtInt(H)} outputs split into ${nH} heads × ${hd}.`,
      `q = W_Q · x${bias('q')}   (${fmtInt(H)}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * H, { mat: 'q' });
    case 'k': return T('Key projection  W_K', 'Attention',
      `Produces ${nKV} key heads × ${hd} = ${KV} numbers. Grouped-query attention: each key head is shared by ${nH / nKV} query heads.`,
      `k = W_K · x${bias('k')}   (${KV}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * KV, { mat: 'k' });
    case 'v': return T('Value projection  W_V', 'Attention',
      `Produces ${nKV} value heads × ${hd}: what each token hands over when attended to. Appended to the V cache, so earlier tokens never need recomputing.`,
      `v = W_V · x${bias('v')}   (${KV}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * KV, { mat: 'v' });
    case 'rope': return T('Rotary position embedding (RoPE)', 'Attention',
      `Each pair of dimensions (i, i+${hd / 2}) in every head of q and k is rotated by the angle pos·θᵢ. Dot products between rotated vectors depend on relative position. The rotated keys are written to the K cache (${kv} positions now).`,
      `q' = q·cos(pos·θ) + rotate_half(q)·sin(pos·θ)   (same for k)`, n * (nH + nKV) * hd * 2);
    case 'scores': return T('Attention scores  QKᵀ/√d', 'Attention',
      `Each of the ${nH} query heads takes a dot product with all ${kv} cached keys of its key head, scaled by 1/√${hd}. These are the raw, pre-softmax scores (blue = negative, amber = positive, scaled per head).`,
      `s_h[p] = q_h · k_h[p] / √${hd}     for p = 0…${kv - 1}`, nH * n * kv * hd);
    case 'softmax': return T('Softmax → attention weights', 'Attention',
      `Exponentiate and normalize each head's scores so they sum to 1${n > 1 ? ', with a causal mask so no token sees the future' : ''}. Bright cells show which earlier tokens each head reads. Arcs below show the average over heads.`,
      `a_h[p] = e^(s_h[p]) / Σ_p' e^(s_h[p'])`, nH * n * kv * 3);
    case 'mix': return T('Weighted sum of values', 'Attention',
      `Each head averages the ${kv} cached value vectors using its attention weights, then the ${nH} head outputs are concatenated into ${fmtInt(H)} numbers.`,
      `ctx_h = Σ_p a_h[p] · v_h[p]`, nH * n * kv * hd);
    case 'o': return T('Output projection  W_O', 'Attention',
      `Mixes the ${nH} heads' results back into residual-stream space.`,
      `a = W_O · ctx   (${fmtInt(H)}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * H, { mat: 'o' });
    case 'add1': return T('Residual add (attention)', 'Residual',
      'Attention does not replace the representation, it adds to it. The residual stream carries information up through all layers.',
      'x ← x + Attention(RMSNorm(x))', n * H);
    case 'norm2': return T('RMSNorm (before MLP)', 'MLP',
      'Same normalization again, with its own learned gains, ahead of the feed-forward block.',
      `y = x / √(mean(x²) + ${info.rms_eps}) ⊙ g`, n * H);
    case 'gate': return T('Gate projection  W_gate', 'MLP',
      `Expands ${fmtInt(H)} → ${fmtInt(I)} numbers. After SiLU these act as soft on/off switches for the ${fmtInt(I)} hidden features.`,
      `g = W_gate · x   (${fmtInt(I)}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * I, { mat: 'gate' });
    case 'act': return T('SiLU activation', 'MLP',
      'The non-linearity. Without it, all these layers would collapse into one big linear map.',
      'SiLU(g) = g · σ(g) = g / (1 + e^(−g))', n * I);
    case 'up': return T('Up projection  W_up', 'MLP',
      `A second ${fmtInt(H)} → ${fmtInt(I)} projection, carrying the feature values the gate will scale.`,
      `u = W_up · x   (${fmtInt(I)}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * I, { mat: 'up' });
    case 'mul': return T('Gating (element-wise multiply)', 'MLP',
      `SwiGLU: each of the ${fmtInt(I)} up-values is multiplied by its gate.`,
      'h = SiLU(g) ⊙ u', n * I);
    case 'down': return T('Down projection  W_down', 'MLP',
      `Compresses the ${fmtInt(I)} features back to ${fmtInt(H)} numbers to write into the residual stream.`,
      `m = W_down · h   (${fmtInt(H)}×${fmtInt(I)}) · (${fmtInt(I)})`, n * H * I, { mat: 'down' });
    case 'add2': return T('Residual add (MLP)', 'Residual',
      'The result is this layer\'s output and the next layer\'s input. The logit lens decodes it to show what the model would say if it stopped here.',
      'x ← x + MLP(RMSNorm(x))', n * H);
    case 'final_norm': return T('Final RMSNorm', 'Output',
      'One last normalization of the residual stream after all layers.',
      'y = x / √(mean(x²) + ε) ⊙ g', H);
    case 'lm_head': return T('LM head: logits over the vocabulary', 'Output',
      `Dot product of the final vector with every one of the ${fmtInt(V)} token embeddings${info.tied ? ' (the same table used at the input: tied weights)' : ''}. Only the last position is computed.`,
      `logits = E · y   (${fmtInt(V)}×${fmtInt(H)}) · (${fmtInt(H)})`, H * V, { mat: 'lm_head' });
    case 'sample': return T('Softmax + sampling', 'Output',
      `Logits → probabilities. The next token is drawn (temperature ${meta.temperature}, top-p 0.9). It becomes the input of the next step.`,
      'p = softmax(logits / T);  token ~ p', V);
  }
  return T(ev.op, '', '', '', 0);
}
