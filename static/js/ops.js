// What each recorded GPU operation is, in words and math, with this model's real dimensions.
import { fmtInt } from './util.js';

export const MATMUL = new Set(['q', 'k', 'v', 'o', 'gate', 'up', 'down', 'lm_head']);

export const GROUP = {
  embed: 'io', prep: 'io', final_norm: 'io', lm_head: 'io', sample: 'io',
  norm1: 'norm', norm2: 'norm',
  q: 'attn', k: 'attn', v: 'attn', attn: 'attn', o: 'attn',
  add1: 'add', add2: 'add',
  gate: 'mlp', act: 'mlp', up: 'mlp', mul: 'mlp', down: 'mlp',
};

export const GROUP_COLOR = { io: '#b794ff', norm: '#ffd166', attn: '#4cc9f0', add: '#7ae582', mlp: '#ff5d8f' };

// Which captured tensor feeds each matmul (the "x" in W·x) and which holds its output.
export const MAT_IO = {
  q: ['norm1', 'q'], k: ['norm1', 'k'], v: ['norm1', 'v'], o: ['ctx', 'o'],
  gate: ['norm2', 'gate'], up: ['norm2', 'up'], down: ['mul', 'down'], lm_head: ['final_norm', null],
};

export function matDims(info, op) {
  const H = info.hidden, I = info.intermediate, KV = info.n_kv_heads * info.head_dim;
  return {
    q: [H, H], k: [H, KV], v: [H, KV], o: [H, H],
    gate: [H, I], up: [H, I], down: [I, H], lm_head: [H, info.vocab],
  }[op];
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
    case 'prep': return T('Position setup (RoPE + causal mask)', 'Input',
      `Computes the rotary position embedding (RoPE) cos/sin tables for position${n > 1 ? 's 0…' + (kv - 1) : ' ' + (kv - 1)} and the causal mask that stops tokens seeing the future. No learned weights.`,
      `θᵢ = pos · ${info.rope_theta ?? 10000}^(−2i/${hd})`, 0);
    case 'norm1': return T('RMSNorm (before attention)', 'Attention',
      'Rescales the residual stream to unit root-mean-square, then multiplies by a learned gain per dimension. This keeps numbers in a stable range before the matrix multiplies.',
      `y = x / √(mean(x²) + ${info.rms_eps}) ⊙ g`, n * H);
    case 'q': return T('Query projection  W_Q', 'Attention',
      `Every output number is a dot product of a ${fmtInt(H)}-number weight row with the normalized input. The ${fmtInt(H)} outputs split into ${nH} heads × ${hd}: for each head, "what am I looking for?"`,
      `q = W_Q · x${bias('q')}   (${fmtInt(H)}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * H, { mat: 'q' });
    case 'k': return T('Key projection  W_K', 'Attention',
      `Produces ${nKV} key heads × ${hd} = ${KV} numbers: "what do I contain?" Grouped-query attention: each key head is shared by ${nH / nKV} query heads. Keys are appended to the KV cache, so earlier tokens never need recomputing.`,
      `k = W_K · x${bias('k')}   (${KV}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * KV, { mat: 'k' });
    case 'v': return T('Value projection  W_V', 'Attention',
      `Produces ${nKV} value heads × ${hd}: the information each token hands over when someone attends to it. Also appended to the KV cache.`,
      `v = W_V · x${bias('v')}   (${KV}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * KV, { mat: 'v' });
    case 'attn': return T('Scaled dot-product attention', 'Attention',
      `RoPE rotates q and k by position. Each of the ${nH} heads scores the current token's query against all ${kv} cached keys, softmaxes the scores into weights (heatmap), and takes the weighted average of the values. Bright cells show which earlier tokens each head is reading.`,
      `ctx_h = softmax(q_h · K_hᵀ / √${hd}) · V_h     for h = 0…${nH - 1}`, nH * n * kv * hd * 2);
    case 'o': return T('Output projection  W_O', 'Attention',
      `Concatenates the ${nH} heads' results (${fmtInt(H)} numbers) and mixes them back into residual-stream space.`,
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
      'The non-linearity. Without it, all these layers would collapse into one big linear map. Negative inputs are squashed toward 0, positive ones pass through.',
      'SiLU(g) = g · σ(g) = g / (1 + e^(−g))', n * I);
    case 'up': return T('Up projection  W_up', 'MLP',
      `A second ${fmtInt(H)} → ${fmtInt(I)} projection, carrying the feature values the gate will scale.`,
      `u = W_up · x   (${fmtInt(I)}×${fmtInt(H)}) · (${fmtInt(H)})`, n * H * I, { mat: 'up' });
    case 'mul': return T('Gating (element-wise multiply)', 'MLP',
      `SwiGLU: each of the ${fmtInt(I)} up-values is multiplied by its gate. This is where the MLP decides which features fire.`,
      'h = SiLU(g) ⊙ u', n * I);
    case 'down': return T('Down projection  W_down', 'MLP',
      `Compresses the ${fmtInt(I)} features back to ${fmtInt(H)} numbers to write into the residual stream. This is usually where facts stored in the MLP come out.`,
      `m = W_down · h   (${fmtInt(H)}×${fmtInt(I)}) · (${fmtInt(I)})`, n * H * I, { mat: 'down' });
    case 'add2': return T('Residual add (MLP)', 'Residual',
      'Add the MLP output to the stream. The result is this layer\'s output and the next layer\'s input. The logit lens (right) decodes it to show what the model would say if it stopped here.',
      'x ← x + MLP(RMSNorm(x))', n * H);
    case 'final_norm': return T('Final RMSNorm', 'Output',
      'One last normalization of the residual stream after all layers.',
      'y = x / √(mean(x²) + ε) ⊙ g', H);
    case 'lm_head': return T('LM head: logits over the vocabulary', 'Output',
      `Dot product of the final vector with every one of the ${fmtInt(V)} token embeddings${info.tied ? ' (the same table used at the input: tied weights)' : ''}. One score (logit) per possible next token. Only the last position is computed.`,
      `logits = E · y   (${fmtInt(V)}×${fmtInt(H)}) · (${fmtInt(H)})`, H * V, { mat: 'lm_head' });
    case 'sample': return T('Softmax + sampling', 'Output',
      `Logits → probabilities. The next token is drawn (temperature ${meta.temperature}, top-p 0.9). It becomes the input of the next step.`,
      'p = softmax(logits / T);  token ~ p', V);
  }
  return T(ev.op, '', '', '', 0);
}
