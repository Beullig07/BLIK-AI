// Runtime test in a real browser: loads the configured model and chats with it.
//
//   node tests/serve.cjs &            (serves the repo on http://127.0.0.1:8378/)
//   node tests/runtime.cjs [steps]    steps: any of  env load chat reuse long stop sample context
//                                     (default: env load chat)
//
// It prints what the page's worker reported. Nothing here is simulated: if the
// model cannot run, the test fails.

const { launch, openHarness, mib } = require('./browser.cjs');

const steps = process.argv.slice(2).length ? process.argv.slice(2) : ['env', 'load', 'chat'];
const has = (name) => steps.includes(name);
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) process.exitCode = 1;
};

const base = { temperature: 0, topK: 50, topP: 1, repetitionPenalty: 1.05, maxNewTokens: 24, contextCap: 2048 };

(async () => {
  const context = await launch();
  const h = await openHarness(context);
  let nextId = 1;

  /** Sends one generate request and collects everything it produced. */
  async function chat(messages, params, { timeout = 600000, onStep } = {}) {
    const id = nextId++;
    const out = { text: '', steps: [], prefill: [], done: null };
    const t0 = Date.now();
    const finished = h.waitFor((m) => m.type === 'done' && m.id === id, {
      timeout,
      onMessage(m) {
        if (m.id !== id) return;
        if (m.type === 'text') out.text += m.text;
        else if (m.type === 'step') {
          out.steps.push(m);
          onStep?.(m);
        } else if (m.type === 'prefill') out.prefill.push(m);
      },
    });
    await h.send({ type: 'generate', id, messages, params: { ...base, ...params } });
    out.done = await finished;
    out.wallMs = Date.now() - t0;
    return out;
  }

  try {
    if (has('env') || has('load')) {
      await h.send({ type: 'hello' });
      const env = await h.waitFor((m) => m.type === 'env');
      const a = env.adapter;
      console.log('\n== environment');
      console.log(`model            ${env.model.repo} @ ${env.model.revision.slice(0, 7)} (${env.model.dtype})`);
      console.log(`transformers.js  ${env.versions.transformers}   onnxruntime-web ${env.versions.onnxruntime}`);
      console.log(`wasm build       ${env.wasmBuild} (JSPI ${env.jspi ? 'yes' : 'no'}), threads ${env.threads}, SharedArrayBuffer ${env.sharedArrayBuffer ? 'yes' : 'no'}`);
      console.log(`OPFS             ${env.opfs ? 'yes' : 'no'}`);
      console.log(`adapter          ${a.info ? `${a.info.vendor} / ${a.info.architecture} / ${a.info.description || a.info.device}` : 'none'}`);
      console.log(`                 software renderer: ${a.software ? 'YES (a CPU standing in for a GPU)' : 'no'}; shader-f16: ${a.f16 ? 'yes' : 'no'}`);
      if (a.limits) console.log(`                 maxBufferSize ${mib(a.limits.maxBufferSize)}, maxStorageBufferBindingSize ${mib(a.limits.maxStorageBufferBindingSize)}`);
      console.log(`stored           ${env.stored.files.length} files, ${mib(env.stored.bytes)}, weights complete: ${env.stored.weightsComplete}, partial: ${env.stored.partial}`);
      if (env.storage) console.log(`storage          ${mib(env.storage.usage)} used of ${mib(env.storage.quota)} quota`);
      check(a.ok, `the adapter can run this model${a.ok ? '' : `: ${a.problem}`}`);
      check(env.opfs, 'OPFS with synchronous access is available in the worker');
    }

    if (has('load')) {
      console.log('\n== load');
      let lastPrint = 0;
      const ready = h.waitFor((m) => m.type === 'ready', {
        timeout: 3 * 3600 * 1000,
        onMessage(m) {
          if (m.type === 'phase') console.log(`phase: ${m.phase}`);
          if (m.type === 'download' && (Date.now() - lastPrint > 15000 || m.loaded === m.total)) {
            lastPrint = Date.now();
            const speed = m.bytesPerSecond ? `, ${(m.bytesPerSecond / 1048576).toFixed(1)} MiB/s` : '';
            const from = m.resumedFrom ? `, resumed from ${mib(m.resumedFrom)}` : '';
            console.log(`  ${m.file}: ${mib(m.loaded)} of ${mib(m.total)}${speed}${from}${m.cached ? ' (already stored)' : ''}`);
          }
        },
      });
      await h.send({ type: 'load' });
      const r = await ready;
      console.log(`timings          total ${(r.timings.totalMs / 1000).toFixed(1)} s = download ${(r.timings.downloadMs / 1000).toFixed(1)} s + session ${(r.timings.sessionMs / 1000).toFixed(1)} s + warm-up ${(r.timings.warmupMs / 1000).toFixed(1)} s`);
      console.log(`download         ${r.download.fromCache ? 'from OPFS (no network)' : `${mib(r.download.downloadedBytes)} downloaded${r.download.resumedFrom ? `, resumed from ${mib(r.download.resumedFrom)}` : ''}`}`);
      console.log(`weights read     ${r.weights.reads} reads, ${mib(r.weights.bytes)} in total, largest ${mib(r.weights.largest)} (the largest buffer JavaScript ever held)`);
      console.log(`session          device ${r.session.device}, dtype ${r.session.dtype}, ${r.session.inputs} inputs, num_logits_to_keep input: ${r.session.numLogitsToKeep}`);
      console.log(`KV cache         ${r.session.kvTensors} tensors, ${r.session.kvType}, ${r.session.kvBytesPerToken} bytes per token each`);
      console.log(`model            ${r.model.modelType}, ${r.model.layers} layers, hidden ${r.model.hiddenSize}, vocab ${r.model.vocabSize}`);
      if (r.device) console.log(`ORT's device     shader-f16 ${r.device.f16}, maxBufferSize ${mib(r.device.maxBufferSize)}${r.device.info ? `, ${r.device.info.vendor}/${r.device.info.architecture}/${r.device.info.description}` : ''}`);
      console.log(`GPU buffers      live ${mib(r.gpu.live)}, peak ${mib(r.gpu.peak)}, ${r.gpu.created} created, ${r.gpu.destroyed} destroyed, pool ${r.gpu.pool ? 'on' : 'off'} (${r.gpu.pooled} waiting)`);
      console.log(`persisted        ${r.persisted}`);
      check(r.session.device === 'webgpu', 'the session runs on the WebGPU execution provider');
      check(r.session.numLogitsToKeep, 'the graph has a num_logits_to_keep input');
      check(r.gpu.live > 500 * 1048576, `the weights are in GPU buffers (${mib(r.gpu.live)} live)`);
      check(r.weights.largest < 300 * 1048576, `no weights read was larger than 300 MiB (largest ${mib(r.weights.largest)})`);
      check(r.stored.weightsComplete, 'the weights are stored complete in OPFS');
    }

    if (has('chat')) {
      console.log('\n== chat (greedy, short)');
      const messages = [{ role: 'user', content: 'What is the capital of France? Answer in one short sentence.' }];
      const r = await chat(messages, { maxNewTokens: 16 }, { onStep: (s) => process.stdout.write(`${JSON.stringify(s.chosen.text)} `) });
      console.log();
      const d = r.done;
      console.log(`reply            ${JSON.stringify(r.text)}`);
      console.log(`tokens           prompt ${d.promptTokens} (cached ${d.cachedTokens}, ${d.prefillChunks} prefill chunks), generated ${d.generated}, reason ${d.reason}`);
      console.log(`timing           first token ${(d.ttftMs / 1000).toFixed(2)} s, ${d.tokensPerSecond?.toFixed(2)} tokens/s, prompt ${d.promptTokensPerSecond?.toFixed(1)} tokens/s`);
      console.log(`context          ${d.context} of ${d.cap}`);
      const s0 = r.steps[0];
      console.log(`step 0 top-${s0.top.length}    ${s0.top.map((c) => `${JSON.stringify(c.text)} ${(c.p * 100).toFixed(1)}%`).join('  ')}`);
      console.log(`step 0           entropy ${s0.entropy.toFixed(3)} bits, chosen p ${(s0.pModel * 100).toFixed(1)}%`);
      check(/paris/i.test(r.text), 'the reply mentions Paris');
      check(r.steps.length === d.generated, 'one logits step per generated token');
      check(r.steps.every((s) => s.top.length > 0 && s.top[0].p > 0 && s.top[0].p <= 1 && s.top.every((c, i) => i === 0 || c.p <= s.top[i - 1].p)), 'top-k probabilities are valid and sorted');
      check(r.steps.every((s) => s.chosen.id === s.top[0].id), 'at temperature 0 every chosen token is the most likely one');
      check(d.context === d.promptTokens + d.generated, 'context used = prompt + generated');

      if (has('reuse')) {
        console.log('\n== second turn (KV cache reuse)');
        const turn2 = [...messages, { role: 'assistant', content: r.text }, { role: 'user', content: 'And of Italy?' }];
        const r2 = await chat(turn2, { maxNewTokens: 12 });
        const d2 = r2.done;
        console.log(`reply            ${JSON.stringify(r2.text)}`);
        console.log(`tokens           prompt ${d2.promptTokens}, reused from cache ${d2.cachedTokens}, read ${d2.promptRead}, generated ${d2.generated}, reason ${d2.reason}`);
        console.log(`timing           first token ${(d2.ttftMs / 1000).toFixed(2)} s, ${d2.tokensPerSecond?.toFixed(2)} tokens/s`);
        check(d2.cachedTokens > 0, `the second turn reused the first turn's cache (${d2.cachedTokens} tokens)`);
        check(/rome|roma/i.test(r2.text), 'the reply mentions Rome');

        console.log('\n== same second turn from an empty cache (must give the same tokens)');
        await h.send({ type: 'reset' });
        await h.waitFor((m) => m.type === 'gpu');
        const r3 = await chat(turn2, { maxNewTokens: 12 });
        console.log(`reply            ${JSON.stringify(r3.text)}  (cached ${r3.done.cachedTokens})`);
        check(r3.done.cachedTokens === 0, 'after "new chat" nothing is reused');
        check(r3.steps.map((s) => s.chosen.id).join() === r2.steps.map((s) => s.chosen.id).join(), 'cache reuse does not change the reply');
      }
    }
    const reset = async () => {
      await h.send({ type: 'reset' });
      await h.waitFor((m) => m.type === 'gpu');
    };
    const ids = (r) => r.steps.map((s) => s.chosen.id).join(' ');

    if (has('long')) {
      console.log('\n== long prompt: read in chunks, and the same reply as reading it in one pass');
      const colours = ['red', 'green', 'blue', 'black', 'white'];
      const items = Array.from({ length: 24 }, (_, i) => `Item ${i + 1}: the ${colours[i % 5]} box holds ${((i * 7) % 23) + 2} parts.`).join(' ');
      const messages = [{ role: 'user', content: `${items}\n\nHow many parts does the box in item 3 hold? Answer with the number only.` }];
      const chunking = { targetMs: 1000, first: 48, min: 32, max: 96 };
      await reset();
      const chunked = await chat(messages, { maxNewTokens: 6, prefill: chunking }, { timeout: 3600000 });
      const sizes = chunked.prefill.filter((m) => m.chunk > 0).map((m) => m.chunk);
      console.log(`chunked          prompt ${chunked.done.promptTokens} tokens in ${chunked.done.prefillChunks} chunks of ${sizes.join(', ')} + ${chunked.done.promptTokens - sizes.reduce((a, b) => a + b, 0)} read by generate()`);
      console.log(`                 chunk times ${chunked.prefill.filter((m) => m.chunk > 0).map((m) => `${(m.ms / 1000).toFixed(1)} s`).join(', ')}; first token ${(chunked.done.ttftMs / 1000).toFixed(1)} s`);
      console.log(`                 reply ${JSON.stringify(chunked.text)}`);
      await reset();
      const single = await chat(messages, { maxNewTokens: 6, prefill: { targetMs: 1000, first: 1e6, min: 1e6, max: 1e6 } }, { timeout: 3600000 });
      console.log(`one pass         ${single.done.prefillChunks} chunks; first token ${(single.done.ttftMs / 1000).toFixed(1)} s; reply ${JSON.stringify(single.text)}`);
      const drift = Math.max(...chunked.steps.map((s, i) => Math.abs(s.top[0].p - (single.steps[i]?.top[0].p ?? 0))));
      console.log(`                 largest difference in the top token's probability: ${drift.toExponential(2)}`);
      check(chunked.done.prefillChunks >= 2, `the prompt was read in ${chunked.done.prefillChunks} chunks`);
      check(sizes.every((n) => n <= chunking.max), `no chunk was larger than ${chunking.max} tokens`);
      check(single.done.prefillChunks === 0, 'the comparison run read the prompt in one pass');
      check(ids(chunked) === ids(single), 'chunked and one-pass reading give the same tokens');
    }

    if (has('stop')) {
      console.log('\n== stop mid-reply, then carry on');
      await reset();
      const messages = [{ role: 'user', content: 'Count from one to thirty in words, separated by commas.' }];
      const r = await chat(messages, { maxNewTokens: 64 }, { onStep: (s) => s.index === 3 && h.send({ type: 'stop' }) });
      console.log(`reply            ${JSON.stringify(r.text)}  (${r.done.generated} tokens, reason ${r.done.reason})`);
      check(r.done.reason === 'stopped', 'the reply ended because it was stopped');
      check(r.done.generated >= 4 && r.done.generated <= 6, `it stopped within two tokens of the request (${r.done.generated} generated)`);
      const r2 = await chat([...messages, { role: 'assistant', content: r.text }, { role: 'user', content: 'Thanks. Say OK.' }], { maxNewTokens: 4 });
      console.log(`next turn        ${JSON.stringify(r2.text)}  (reused ${r2.done.cachedTokens} of ${r2.done.promptTokens} prompt tokens)`);
      check(r2.done.generated > 0, 'the next turn works after a stop');
    }

    if (has('sample')) {
      console.log('\n== sampling: temperature 0.9, top-p 0.9, seeded');
      const messages = [{ role: 'user', content: 'Give me one unusual name for a cat.' }];
      const params = { temperature: 0.9, topP: 0.9, topK: 0, repetitionPenalty: 1, maxNewTokens: 8, seed: 42 };
      await reset();
      const a = await chat(messages, params);
      await reset();
      const b = await chat(messages, params);
      await reset();
      const c = await chat(messages, { ...params, seed: 7 });
      console.log(`seed 42          ${JSON.stringify(a.text)}   kept per step: ${a.steps.map((s) => s.kept).join(' ')}`);
      console.log(`seed 42 again    ${JSON.stringify(b.text)}`);
      console.log(`seed 7           ${JSON.stringify(c.text)}`);
      check(ids(a) === ids(b), 'the same seed gives the same tokens');
      check(a.steps.some((s) => s.kept > 1), 'top-p kept more than one candidate at some step');
      check(a.steps.every((s) => s.kept < 65536), 'top-p cut the vocabulary at every step');
      check(a.steps.every((s) => s.pSample > 0 && s.pSample <= 1 && s.pModel > 0 && s.pModel <= 1), 'probabilities are valid');
      console.log(`(seed 7 ${ids(a) === ids(c) ? 'happened to give the same tokens' : 'gave different tokens'})`);
    }

    if (has('context')) {
      console.log('\n== context cap: old turns dropped, and the reply stops at the cap');
      await reset();
      const turns = [];
      for (let i = 0; i < 6; i++) {
        turns.push({ role: 'user', content: `Note ${i + 1}: remember that drawer ${i + 1} contains ${['pens', 'tape', 'clips', 'stamps', 'string', 'glue'][i]}, ${['cards', 'keys', 'coins', 'maps', 'nails', 'wire'][i]} and a spare ${['lamp', 'lock', 'fuse', 'bulb', 'plug', 'bell'][i]}.` });
        turns.push({ role: 'assistant', content: `Noted: drawer ${i + 1}.` });
      }
      turns.push({ role: 'user', content: 'Write a very long story about a lighthouse keeper. Do not stop.' });
      const r = await chat([{ role: 'system', content: 'You are a helpful assistant.' }, ...turns], { contextCap: 256, maxNewTokens: 512 }, { timeout: 3600000 });
      const d = r.done;
      console.log(`tokens           prompt ${d.promptTokens} after dropping ${d.dropped} old messages; generated ${d.generated}; context ${d.context} of ${d.cap}; reason ${d.reason}`);
      console.log(`reply            ${JSON.stringify(r.text.slice(0, 120))}...`);
      check(d.dropped > 0, `old messages were dropped to fit (${d.dropped})`);
      check(d.promptTokens + 64 <= 256, 'the prompt left room for the reply');
      check(d.context <= d.cap, `context never exceeded the cap (${d.context} of ${d.cap})`);
      check(d.reason === 'context' && d.context === d.cap, 'the reply stopped exactly at the cap');
    }
  } catch (error) {
    console.log(`\nFAIL  ${error.message}`);
    process.exitCode = 1;
  } finally {
    await context.close();
  }
})();
