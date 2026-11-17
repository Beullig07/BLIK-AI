// Next-token sampling from the model's real logits.
//
// Transformers.js 4.3.2 applies temperature and top-k itself but ignores
// top_p (its TopPLogitsWarper has no implementation) and does not hand back
// per-step logits. So BLIK samples here, from inside a logits processor, and
// records what the model actually predicted at each step.
//
// Pure functions, no browser APIs: tests/unit.mjs runs them in Node.

/** A small seedable generator, for repeatable tests. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The k largest values, largest first, written into outIdx/outVal.
 * One pass; values rarely beat the current k-th once it has warmed up.
 * Returns how many were written (k, or fewer for a short input).
 */
export function topK(values, k, outIdx, outVal) {
  const n = values.length;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (count === k && !(v > outVal[k - 1])) continue;
    let j = count < k ? count++ : k - 1;
    while (j > 0 && outVal[j - 1] < v) {
      outVal[j] = outVal[j - 1];
      outIdx[j] = outIdx[j - 1];
      j--;
    }
    outVal[j] = v;
    outIdx[j] = i;
  }
  return count;
}

/** Buffers reused across steps, so a step allocates almost nothing. */
export function createWorkspace(vocabSize, show) {
  return {
    work: new Float32Array(vocabSize),
    showIdx: new Int32Array(show),
    showVal: new Float32Array(show),
    candIdx: new Int32Array(vocabSize),
    candVal: new Float32Array(vocabSize),
    candP: new Float64Array(vocabSize),
  };
}

/**
 * One sampling step.
 *
 * @param {Float32Array} logits  the model's logits for the next token (not modified)
 * @param {Iterable<number>} seen  token ids already in the context (for the repetition penalty)
 * @param {{temperature:number, topK:number, topP:number, repetitionPenalty:number, show:number}} p
 * @param {() => number} rng  uniform in [0, 1)
 * @param ws  from createWorkspace
 * @returns {{
 *   chosen: number,        the sampled token id
 *   top: {id:number, p:number}[],  the model's most likely tokens, before any sampling setting
 *   pModel: number,        the chosen token's probability under the model's own distribution
 *   pSample: number,       its probability in the distribution it was actually drawn from
 *   kept: number,          how many tokens that distribution had
 *   entropy: number        entropy of the model's distribution, in bits
 * }}
 */
export function sampleStep(logits, seen, p, rng, ws) {
  const n = logits.length;

  // The model's own distribution (temperature 1, nothing filtered).
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max);
    s1 += e;
    if (e > 0) s2 += e * logits[i];
  }
  const lse = max + Math.log(s1);
  if (!Number.isFinite(lse)) {
    throw new Error('The model returned logits that are not finite numbers.');
  }
  const entropy = (lse - s2 / s1) / Math.LN2;

  const shown = topK(logits, Math.min(p.show, n), ws.showIdx, ws.showVal);
  const top = [];
  for (let i = 0; i < shown; i++) top.push({ id: ws.showIdx[i], p: Math.exp(ws.showVal[i] - lse) });

  // Repetition penalty, as Hugging Face defines it: a token already in the
  // context has its logit divided by the penalty if positive, multiplied if not.
  let w = logits;
  if (p.repetitionPenalty !== 1) {
    w = ws.work;
    w.set(logits);
    for (const id of seen) {
      if (id >= 0 && id < n) w[id] = w[id] < 0 ? w[id] * p.repetitionPenalty : w[id] / p.repetitionPenalty;
    }
  }

  let chosen;
  let pSample = 1;
  let kept = 1;

  if (!(p.temperature > 0)) {
    // Temperature 0: always the most likely token.
    let best = 0;
    for (let i = 1; i < n; i++) if (w[i] > w[best]) best = i;
    chosen = best;
  } else {
    const T = p.temperature;
    const useTopK = p.topK > 0 && p.topK < n;
    const useTopP = p.topP > 0 && p.topP < 1;

    if (!useTopK && !useTopP) {
      // The whole vocabulary: one pass over the cumulative sum, no sorting.
      let wmax = -Infinity;
      for (let i = 0; i < n; i++) if (w[i] > wmax) wmax = w[i];
      let z = 0;
      for (let i = 0; i < n; i++) z += Math.exp((w[i] - wmax) / T);
      let r = rng() * z;
      chosen = n - 1;
      for (let i = 0; i < n; i++) {
        r -= Math.exp((w[i] - wmax) / T);
        if (r < 0) {
          chosen = i;
          break;
        }
      }
      pSample = Math.exp((w[chosen] - wmax) / T) / z;
      kept = n;
    } else {
      // Candidates, most likely first.
      let count;
      let z;
      if (useTopK) {
        // Top-k first; what is left is renormalised before top-p looks at it.
        count = topK(w, p.topK, ws.candIdx, ws.candVal);
        z = 0;
        for (let i = 0; i < count; i++) z += Math.exp((ws.candVal[i] - ws.candVal[0]) / T);
      } else {
        // Top-p alone: take enough of the most likely tokens to cover it,
        // measured against the whole vocabulary.
        let wmax = -Infinity;
        for (let i = 0; i < n; i++) if (w[i] > wmax) wmax = w[i];
        z = 0;
        for (let i = 0; i < n; i++) z += Math.exp((w[i] - wmax) / T);
        let size = 256;
        for (;;) {
          size = Math.min(size, n);
          count = topK(w, size, ws.candIdx, ws.candVal);
          let mass = 0;
          for (let i = 0; i < count; i++) mass += Math.exp((ws.candVal[i] - wmax) / T);
          if (mass / z >= p.topP || size === n) break;
          size *= 4;
        }
      }
      const ref = ws.candVal[0];
      for (let i = 0; i < count; i++) ws.candP[i] = Math.exp((ws.candVal[i] - ref) / T) / z;

      // Top-p: the smallest set of most likely tokens whose probabilities add up to topP.
      kept = count;
      let total = 0;
      for (let i = 0; i < count; i++) total += ws.candP[i];
      if (useTopP) {
        let cum = 0;
        for (let i = 0; i < count; i++) {
          cum += ws.candP[i];
          if (cum >= p.topP) {
            kept = i + 1;
            total = cum;
            break;
          }
        }
      }
      let r = rng() * total;
      let pick = kept - 1;
      for (let i = 0; i < kept; i++) {
        r -= ws.candP[i];
        if (r < 0) {
          pick = i;
          break;
        }
      }
      chosen = ws.candIdx[pick];
      pSample = ws.candP[pick] / total;
    }
  }

  return { chosen, top, pModel: Math.exp(logits[chosen] - lse), pSample, kept, entropy };
}
