// Measures GPU buffer churn as the context grows past 2,048 tokens, with and
// without the buffer pool (js/gpu.js), and checks the pool does not change
// the model's output.
//
//   node tests/serve.cjs &
//   node tests/pool.cjs [approximate prompt tokens, default 3300]
//
// Background: ONNX Runtime keeps spare GPU buffers per size bucket. One KV
// cache tensor of this model is 1,024 bytes per token, so past 2,048 tokens
// they need the 4 MiB bucket, which keeps fewer spares than a decode step
// uses (microsoft/onnxruntime#33751). This test counts what really happens.
//
// The default prompt lands a little over 3,000 tokens, where a decode step
// starts to need more 4 MiB buffers than ONNX Runtime keeps spare.
// Slow on a software GPU: reading 3,000 tokens takes the better part of an hour.

const { launch, openHarness, mib } = require('./browser.cjs');

const target = Number(process.argv[2]) || 3300;
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) process.exitCode = 1;
};

// About `target` tokens of plain numbered sentences (roughly 13 tokens each).
const colours = ['red', 'green', 'blue', 'black', 'white', 'grey', 'brown'];
const things = ['box', 'jar', 'tin', 'bag', 'crate', 'drawer'];
const lines = [];
for (let i = 0; lines.length * 13.2 < target - 40; i++) {
  lines.push(`Item ${i + 1}: the ${colours[i % 7]} ${things[i % 6]} holds ${((i * 7) % 23) + 2} parts.`);
}
// A question with a long answer, so there are enough decode steps to count.
const messages = [{ role: 'user', content: `${lines.join(' ')}\n\nCopy out items 1 to 4 again, word for word.` }];
const params = { temperature: 0, topK: 50, topP: 1, repetitionPenalty: 1.05, maxNewTokens: 14, contextCap: 4096 };

async function run(context, query) {
  const page = context.pages()[0] || (await context.newPage());
  const h = await openHarness(context, { quiet: true, query });
  const ready = h.waitFor((m) => m.type === 'ready', { timeout: 3600000 });
  await h.send({ type: 'load' });
  const r = await ready;
  const start = r.gpu;
  const chunks = [];
  const steps = [];
  let lastPrint = 0;
  const done = h.waitFor((m) => m.type === 'done', {
    timeout: 6 * 3600000,
    onMessage(m) {
      if (m.type === 'prefill' && m.gpu) {
        chunks.push({ read: m.read, ...m.gpu });
        if (Date.now() - lastPrint > 120000) {
          lastPrint = Date.now();
          console.log(`  ... read ${m.read} of ${m.total} tokens, ${m.gpu.created} buffers created so far, live ${mib(m.gpu.live)}`);
        }
      }
      if (m.type === 'step') steps.push({ id: m.chosen.id, context: m.context, ...m.gpu });
    },
  });
  await h.send({ type: 'generate', id: 1, messages, params });
  const d = await done;
  await page.close();
  return { start, chunks, steps, done: d, pool: r.gpu.pool };
}

function report(name, r) {
  console.log(`\n== ${name} (pool ${r.pool ? 'on' : 'off'})`);
  console.log(`prompt ${r.done.promptTokens} tokens, generated ${r.done.generated}, context ${r.done.context}`);
  // New buffers per chunk of prompt, by where the context was.
  const bands = [
    [0, 1024],
    [1024, 2048],
    [2048, 4096],
  ];
  let previous = { read: 0, created: r.start.created, destroyed: r.start.destroyed, reused: r.start.reused, createdBytes: r.start.createdBytes };
  const perBand = bands.map(() => ({ tokens: 0, created: 0, destroyed: 0, reused: 0, bytes: 0 }));
  for (const c of r.chunks) {
    const band = bands.findIndex(([lo, hi]) => c.read > lo && c.read <= hi);
    if (band >= 0) {
      perBand[band].tokens += c.read - previous.read;
      perBand[band].created += c.created - previous.created;
      perBand[band].destroyed += c.destroyed - previous.destroyed;
      perBand[band].reused += c.reused - previous.reused;
      perBand[band].bytes += c.createdBytes - previous.createdBytes;
    }
    previous = c;
  }
  bands.forEach(([lo, hi], i) => {
    const b = perBand[i];
    if (!b.tokens) return;
    console.log(`reading, context ${lo}-${hi}: ${b.tokens} tokens, ${b.created} buffers created (${mib(b.bytes)}), ${b.destroyed} destroyed, ${b.reused} reused from the pool`);
  });
  const first = r.steps[0];
  const last = r.steps[r.steps.length - 1];
  const n = r.steps.length - 1;
  const perToken = (key) => ((last[key] - first[key]) / n).toFixed(2);
  console.log(`writing, context ${first.context}-${last.context}: per token ${perToken('created')} buffers created (${mib((last.createdBytes - first.createdBytes) / n)}), ${perToken('destroyed')} destroyed, ${perToken('reused')} reused from the pool`);
  console.log(`GPU buffers at the end: live ${mib(last.live)}, peak ${mib(last.peak)}, waiting in the pool ${last.pooled} (${mib(last.pooledBytes)})`);
  return { perBand, createdPerToken: (last.created - first.created) / n, bytesPerToken: (last.createdBytes - first.createdBytes) / n };
}

(async () => {
  const context = await launch();
  try {
    const withPool = await run(context, '?pool');
    const a = report('with the buffer pool', withPool);
    const without = await run(context, '?nopool');
    const b = report('without the buffer pool', without);

    console.log();
    check(withPool.pool && !without.pool, 'the two runs differ only in the pool');
    check(withPool.steps.map((s) => s.id).join() === without.steps.map((s) => s.id).join(), 'the pool does not change the tokens generated');
    check(withPool.done.context > 2048, `the context went past 2,048 tokens (${withPool.done.context})`);
    check(withPool.steps.length >= 6, `enough decode steps to measure (${withPool.steps.length})`);
    check(a.createdPerToken <= b.createdPerToken, `while writing, the pool creates no more buffers per token (${a.createdPerToken.toFixed(2)} with, ${b.createdPerToken.toFixed(2)} without)`);
    console.log(`reply: ${JSON.stringify(withPool.steps.map((s) => s.id).length)} tokens`);
  } catch (error) {
    console.log(`FAIL  ${error.message}`);
    process.exitCode = 1;
  } finally {
    await context.close();
  }
})();
