// Unit tests for the pure logic: sampling and prefill chunk sizing.
// Run: node tests/unit.mjs

import assert from 'node:assert/strict';
import { sampleStep, createWorkspace, topK, mulberry32 } from '../js/sampler.js';
import { nextChunk, prefill } from '../js/prefill.js';

let passed = 0;
function test(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => {
      passed++;
      console.log(`ok   ${name}`);
    })
    .catch((error) => {
      console.log(`FAIL ${name}\n     ${error.message}`);
      process.exitCode = 1;
    });
}

const N = 65536;
function randomLogits(seed, n = N, spread = 6) {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rng() + rng() + rng() - 1.5) * spread;
  return out;
}
const softmax = (values, T = 1) => {
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  const e = Array.from(values, (v) => Math.exp((v - max) / T));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
};
const base = { temperature: 1, topK: 0, topP: 1, repetitionPenalty: 1, show: 8 };

await test('topK matches a full sort', () => {
  const logits = randomLogits(1);
  const idx = new Int32Array(50);
  const val = new Float32Array(50);
  assert.equal(topK(logits, 50, idx, val), 50);
  const sorted = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, 50);
  assert.deepEqual(Array.from(idx), sorted);
});

await test('shown probabilities are the softmax of the raw logits', () => {
  const logits = randomLogits(2);
  const ws = createWorkspace(N, 8);
  const step = sampleStep(logits, [], { ...base, temperature: 0.1, topK: 50, repetitionPenalty: 1.05 }, mulberry32(1), ws);
  const p = softmax(logits);
  const sorted = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, 8);
  assert.deepEqual(step.top.map((t) => t.id), sorted);
  for (const t of step.top) assert.ok(Math.abs(t.p - p[t.id]) < 1e-6);
  assert.ok(Math.abs(step.pModel - p[step.chosen]) < 1e-6);
  const entropy = -p.reduce((sum, x) => sum + (x > 0 ? x * Math.log2(x) : 0), 0);
  assert.ok(Math.abs(step.entropy - entropy) < 1e-3, `${step.entropy} vs ${entropy}`);
});

await test('temperature 0 picks the most likely token and leaves the logits alone', () => {
  const logits = randomLogits(3);
  const copy = Float32Array.from(logits);
  const ws = createWorkspace(N, 8);
  const step = sampleStep(logits, [], { ...base, temperature: 0 }, () => 0.99, ws);
  let best = 0;
  for (let i = 1; i < N; i++) if (logits[i] > logits[best]) best = i;
  assert.equal(step.chosen, best);
  assert.equal(step.kept, 1);
  assert.deepEqual(logits, copy);
});

await test('top-k keeps exactly k tokens and never samples outside them', () => {
  const logits = randomLogits(4, 1000);
  const ws = createWorkspace(1000, 8);
  const allowed = new Set(Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, 5));
  const rng = mulberry32(7);
  for (let i = 0; i < 500; i++) {
    const step = sampleStep(logits, [], { ...base, temperature: 2, topK: 5 }, rng, ws);
    assert.ok(allowed.has(step.chosen));
    assert.equal(step.kept, 5);
  }
});

await test('top-p keeps the smallest set reaching p (with and without top-k)', () => {
  const logits = new Float32Array(100).fill(-20);
  // Probabilities 0.5, 0.3, 0.15, 0.05 on four tokens.
  [0.5, 0.3, 0.15, 0.05].forEach((p, i) => (logits[10 + i] = Math.log(p)));
  const ws = createWorkspace(100, 8);
  for (const topKSetting of [0, 50]) {
    const cases = [
      // (settings exactly on a boundary, like 0.5, depend on float rounding)
      [0.4, 1],
      [0.45, 1],
      [0.55, 2],
      [0.75, 2],
      [0.85, 3],
      [0.9, 3],
      [0.96, 4],
    ];
    for (const [topP, expected] of cases) {
      const step = sampleStep(logits, [], { ...base, topK: topKSetting, topP }, mulberry32(3), ws);
      assert.equal(step.kept, expected, `topK ${topKSetting} topP ${topP}`);
    }
  }
});

await test('sampling frequencies follow the tempered distribution', () => {
  const logits = new Float32Array(50).fill(-30);
  const probs = [0.6, 0.25, 0.1, 0.05];
  probs.forEach((p, i) => (logits[i] = Math.log(p)));
  const ws = createWorkspace(50, 8);
  for (const settings of [{ topK: 0, topP: 1 }, { topK: 10, topP: 1 }, { topK: 0, topP: 0.999 }]) {
    const T = 0.7;
    const want = softmax(logits, T);
    const counts = new Array(50).fill(0);
    const rng = mulberry32(11);
    const runs = 40000;
    for (let i = 0; i < runs; i++) counts[sampleStep(logits, [], { ...base, temperature: T, ...settings }, rng, ws).chosen]++;
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(counts[i] / runs - want[i]) < 0.012, `${JSON.stringify(settings)} token ${i}: ${counts[i] / runs} vs ${want[i]}`);
    }
  }
});

await test('pSample is the probability in the distribution actually drawn from', () => {
  const logits = new Float32Array(20).fill(-30);
  [0.5, 0.3, 0.2].forEach((p, i) => (logits[i] = Math.log(p)));
  const ws = createWorkspace(20, 8);
  // top-p 0.7 keeps tokens 0 and 1: renormalised to 0.625 and 0.375.
  const step = sampleStep(logits, [], { ...base, topP: 0.7 }, () => 0.9, ws);
  assert.equal(step.chosen, 1);
  assert.ok(Math.abs(step.pSample - 0.375) < 1e-6, String(step.pSample));
  assert.ok(Math.abs(step.pModel - 0.3) < 1e-6);
});

await test('repetition penalty lowers tokens already seen, as Hugging Face defines it', () => {
  const logits = new Float32Array(10).fill(-5);
  logits[1] = 2.0;
  logits[2] = 1.95;
  const ws = createWorkspace(10, 8);
  assert.equal(sampleStep(logits, [], { ...base, temperature: 0 }, Math.random, ws).chosen, 1);
  // 2.0 / 1.05 = 1.905 < 1.95
  assert.equal(sampleStep(logits, [1], { ...base, temperature: 0, repetitionPenalty: 1.05 }, Math.random, ws).chosen, 2);
  // a negative logit is multiplied, so it gets more negative
  const neg = new Float32Array(10).fill(-5);
  neg[3] = -1;
  neg[4] = -1.04;
  assert.equal(sampleStep(neg, [3], { ...base, temperature: 0, repetitionPenalty: 1.05 }, Math.random, ws).chosen, 4);
  // the reported probabilities are still the model's own
  const step = sampleStep(logits, [1], { ...base, temperature: 0, repetitionPenalty: 1.05 }, Math.random, ws);
  assert.equal(step.top[0].id, 1);
});

await test('non-finite logits are an error, not a token', () => {
  const ws = createWorkspace(10, 8);
  assert.throws(() => sampleStep(new Float32Array(10).fill(NaN), [], base, Math.random, ws), /not finite/);
  assert.throws(() => sampleStep(new Float32Array(10).fill(-Infinity), [], base, Math.random, ws), /not finite/);
});

const cfg = { targetMs: 1000, first: 64, min: 16, max: 256 };

await test('chunk size: first chunk, time target, cap, floor, doubling limit', () => {
  assert.equal(nextChunk(undefined, 0, cfg), 64);
  assert.equal(nextChunk(5, 0, cfg), 200); // 1000 ms / 5 ms per token
  assert.equal(nextChunk(1, 0, cfg), 256); // fast GPU: capped
  assert.equal(nextChunk(100, 0, cfg), 16); // slow: floor
  assert.equal(nextChunk(1, 64, cfg), 128); // at most twice the previous chunk
  assert.equal(nextChunk(8, 200, cfg), 125);
});

await test('prefill reads in chunks, never over the cap, and leaves the last part for generate()', async () => {
  const calls = [];
  let clock = 0;
  const result = await prefill({
    total: 2000,
    held: 0,
    msPerToken: undefined,
    cfg,
    stopped: () => false,
    now: () => clock,
    async readTo(end) {
      const from = calls.length ? calls[calls.length - 1] : 0;
      clock += (end - from) * 4; // 4 ms per token
      calls.push(end);
    },
  });
  assert.deepEqual(calls.slice(0, 4), [64, 192, 442, 692]);
  let previous = 0;
  for (const end of calls) {
    assert.ok(end - previous <= cfg.max);
    previous = end;
  }
  assert.equal(result.read, calls[calls.length - 1]);
  assert.ok(2000 - result.read >= 1 && 2000 - result.read <= cfg.max, `left ${2000 - result.read}`);
  assert.equal(result.chunks, calls.length);
  assert.equal(result.msPerToken, 4);
});

await test('prefill shrinks its chunks when the device slows down', async () => {
  const sizes = [];
  let clock = 0;
  let last = 0;
  await prefill({
    total: 3000,
    held: 0,
    msPerToken: 4,
    cfg,
    stopped: () => false,
    now: () => clock,
    async readTo(end) {
      const size = end - last;
      sizes.push(size);
      clock += size * (end > 1000 ? 20 : 4); // five times slower past 1,000 tokens
      last = end;
    },
  });
  assert.equal(sizes[0], 250);
  assert.ok(sizes.at(-1) === 50, `last chunk ${sizes.at(-1)}`);
});

await test('prefill reads nothing when the prompt fits one chunk, or the cache already holds it', async () => {
  const never = async () => assert.fail('should not read');
  let r = await prefill({ total: 60, held: 0, msPerToken: undefined, cfg, stopped: () => false, readTo: never });
  assert.deepEqual([r.read, r.chunks], [0, 0]);
  r = await prefill({ total: 1500, held: 1400, msPerToken: 4, cfg, stopped: () => false, readTo: never });
  assert.deepEqual([r.read, r.chunks], [1400, 0]);
});

await test('prefill stops between chunks when asked to', async () => {
  let reads = 0;
  const r = await prefill({
    total: 5000,
    held: 0,
    msPerToken: 4,
    cfg,
    stopped: () => reads >= 2,
    now: () => reads * 1000,
    async readTo() {
      reads++;
    },
  });
  assert.equal(r.chunks, 2);
});

console.log(`\n${passed} passed${process.exitCode ? ', some FAILED' : ''}`);
