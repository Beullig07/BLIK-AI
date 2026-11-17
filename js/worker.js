// The model's side: downloads, loads and runs it. A module Web Worker.
//
// Messages in:   hello | load | generate | stop | reset | purge | unload
// Messages out:  env | phase | download | ready | prefill | text | step | done
//                | gpu | gpu-lost | purged | unloaded | error
//
// Every number sent from here is measured here. Nothing is estimated or made up.

import { MODEL, HUB, CONTEXT, PREFILL, SHOW_TOP, GPU_BUFFER_POOL } from './config.js';
import { probeAdapter, installGpuHooks } from './gpu.js';
import * as opfs from './opfs.js';
import { sampleStep, createWorkspace, mulberry32 } from './sampler.js';
import { prefill } from './prefill.js';
import {
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  LogitsProcessor,
  ModelRegistry,
  StoppingCriteriaList,
  TextStreamer,
  env,
  ORT_DIR,
  ORT_VERSION,
} from './lib.js';

const post = (message) => self.postMessage(message);
const now = () => performance.now();
const needsF16 = /f16|fp16/.test(MODEL.dtype);

// ---- GPU hooks: installed before ONNX Runtime creates its device -----------

let ortDevice = null;
const gpu = installGpuHooks({
  // tests/pool.cjs starts the worker with ?nopool to measure the difference.
  pool: GPU_BUFFER_POOL && !new URL(self.location.href).searchParams.has('nopool'),
  onDevice(device) {
    ortDevice = device;
    device.lost?.then((info) => post({ type: 'gpu-lost', reason: info.reason ?? '', message: info.message ?? '' }));
    device.addEventListener?.('uncapturederror', (event) => {
      post({ type: 'error', where: 'gpu', message: `WebGPU error: ${event.error?.message ?? 'unknown'}` });
    });
  },
});
const gpuSnapshot = () => ({
  live: gpu.live,
  peak: gpu.peak,
  created: gpu.created,
  destroyed: gpu.destroyed,
  reused: gpu.reused,
  pooled: gpu.pooled,
  pooledBytes: gpu.pooledBytes,
  createdBytes: gpu.createdBytes,
  pool: gpu.pool,
});

// ---- Transformers.js setup ---------------------------------------------------

const jspi = typeof WebAssembly !== 'undefined' && 'Suspending' in WebAssembly;
// ONNX Runtime's JSPI build where the browser has JSPI, else its Asyncify
// build. Transformers.js always picks Asyncify; JSPI is smaller and, measured
// by others on Safari, peaks about 730 MB lower (huggingface/transformers.js#1835).
const wasmBuild = jspi ? 'jspi' : 'asyncify';

env.remoteHost = HUB;
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = false;
env.useFSCache = false;
env.useCustomCache = true;
env.customCache = opfs.makeCache(HUB);
// Let ONNX Runtime fetch its own WebAssembly from this site (streaming
// compile, HTTP cache) instead of Transformers.js reading it into memory.
env.useWasmCache = false;
env.backends.onnx.wasm.wasmPaths = {
  mjs: `${ORT_DIR}ort-wasm-simd-threaded.${wasmBuild}.mjs`,
  wasm: `${ORT_DIR}ort-wasm-simd-threaded.${wasmBuild}.wasm`,
};
// GitHub Pages cannot send COOP/COEP headers, so there is no
// SharedArrayBuffer and no WebAssembly threads.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

const fileUrl = (path) => `${HUB}${MODEL.repo}/resolve/${encodeURIComponent(MODEL.revision)}/${path}`;
/** The folder this model's files live under in OPFS (as opfs.listFiles reports paths). */
const storedPrefix = `${new URL(HUB).host}/${MODEL.repo}/resolve/${MODEL.revision}/`.replace(/[^\w.@+/-]/g, '_');

// ---- State --------------------------------------------------------------------

/** @type {{tokenizer:any, model:any, config:any, eos:Set<number>} | null} */
let loaded = null;
const stopping = new InterruptableStoppingCriteria();
/** The last generation's KV cache and the tokens it was built from. */
let reuse = null;
/** Prompt reading speed measured so far, for sizing prefill chunks. */
let msPerToken;
let busy = false;
let ws = null;
const tokenText = new Map();

function phase(name, detail) {
  post({ type: 'phase', phase: name, detail: detail ?? null });
}

async function dropReuse() {
  const cache = reuse?.cache;
  reuse = null;
  await cache?.dispose().catch(() => undefined);
}

/** What is stored for this model, without touching the network. */
async function storedState() {
  if (!opfs.opfsSupported()) return { files: [], bytes: 0, weightsComplete: false, partial: false };
  const files = (await opfs.listFiles()).filter((f) => f.path.startsWith(storedPrefix));
  const weights = files.filter((f) => /\.onnx_data(_\d+)?$/.test(f.path));
  return {
    files: files.map((f) => ({ path: f.path.slice(storedPrefix.length), size: f.size, complete: f.complete })),
    bytes: files.reduce((sum, f) => sum + f.size, 0),
    weightsComplete: weights.length > 0 && weights.every((f) => f.complete),
    partial: weights.some((f) => !f.complete),
  };
}

async function hello() {
  post({
    type: 'env',
    model: MODEL,
    versions: { transformers: env.version, onnxruntime: ORT_VERSION },
    wasmBuild,
    jspi,
    threads: 1,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    crossOriginIsolated: !!self.crossOriginIsolated,
    opfs: opfs.opfsSupported(),
    adapter: await probeAdapter({ needsF16 }),
    storage: await opfs.storageEstimate(),
    stored: await storedState(),
    loaded: !!loaded,
  });
}

// ---- Load -----------------------------------------------------------------------

async function load() {
  await unload();
  const t0 = now();

  phase('check');
  if (!opfs.opfsSupported()) {
    throw new Error('This browser cannot read and write files from a worker (OPFS with synchronous access). BLIK needs it to keep the weights out of memory.');
  }
  const adapter = await probeAdapter({ needsF16 });
  if (!adapter.ok) throw new Error(adapter.problem);
  // Ask the browser not to evict the weights under storage pressure. It may say no.
  const persisted = await navigator.storage.persist?.().catch(() => false);

  phase('config');
  const options = { revision: MODEL.revision };
  const config = await AutoConfig.from_pretrained(MODEL.repo, options);
  const files = await ModelRegistry.get_model_files(MODEL.repo, { config, dtype: MODEL.dtype, device: 'webgpu' });
  const dataFiles = files.filter((f) => /\.onnx_data(_\d+)?$/.test(f));

  // The weights: streamed into OPFS, resumed if a part is already there.
  phase('download');
  const tDownload = now();
  const weights = [];
  let downloadedBytes = 0;
  let resumedFrom = 0;
  let fromCache = true;
  let model;
  let tokenizer;
  let tSession;
  try {
    for (const path of dataFiles) {
      let last = 0;
      let mark = { t: now(), loaded: 0 };
      let bps = 0;
      const result = await opfs.ensureFile(fileUrl(path), {
        onProgress({ loaded, total, resumedFrom: from }) {
          const t = now();
          if (mark.loaded === 0) mark = { t, loaded: Math.max(loaded, 1) };
          if (t - mark.t >= 1000) {
            bps = ((loaded - mark.loaded) / (t - mark.t)) * 1000;
            mark = { t, loaded };
          }
          if (t - last < 100 && loaded < total) return;
          last = t;
          post({ type: 'download', file: path, loaded, total, resumedFrom: from, bytesPerSecond: bps });
        },
      });
      if (!result.cached) {
        fromCache = false;
        downloadedBytes += result.size - result.resumedFrom;
        resumedFrom += result.resumedFrom;
      }
      post({ type: 'download', file: path, loaded: result.size, total: result.size, resumedFrom: result.resumedFrom, bytesPerSecond: 0, cached: result.cached });
      const file = new opfs.OnDemandFile(path);
      await file.open(result.handle);
      weights.push(file);
    }
    const downloadMs = now() - tDownload;

    // The session: ONNX Runtime reads each tensor from OPFS and puts it on the GPU.
    phase('session');
    tSession = now();
    tokenizer = await AutoTokenizer.from_pretrained(MODEL.repo, options);
    model = await AutoModelForCausalLM.from_pretrained(MODEL.repo, {
      ...options,
      config,
      dtype: MODEL.dtype,
      device: 'webgpu',
      ...(weights.length > 0 && { use_external_data_format: false }),
      session_options: {
        ...(weights.length > 0 && {
          externalData: weights.map((w) => ({ path: w.path.slice(w.path.lastIndexOf('/') + 1), data: w })),
        }),
        // Constant folding copies the tied embedding table into a second full
        // copy, in WebAssembly memory and on the GPU (microsoft/onnxruntime#33793).
        extra: { 'optimization.disable_specified_optimizers': 'ConstantFolding' },
      },
    });
    const sessionMs = now() - tSession;
    const read = weights.reduce(
      (sum, w) => ({ reads: sum.reads + w.reads, bytes: sum.bytes + w.bytesRead, largest: Math.max(sum.largest, w.largestRead) }),
      { reads: 0, bytes: 0, largest: 0 },
    );
    for (const w of weights) w.close();
    weights.length = 0;

    if (!tokenizer.chat_template) {
      // A repo with its template only in chat_template.jinja.
      const response = await fetch(fileUrl('chat_template.jinja')).catch(() => null);
      if (response?.ok) tokenizer.chat_template = await response.text();
    }
    if (!tokenizer.chat_template) throw new Error('This model has no chat template.');

    const session = model.sessions.model;
    const eosIds = [model.generation_config?.eos_token_id ?? config.eos_token_id].flat().filter((x) => x != null).map(Number);
    loaded = { tokenizer, model, config, eos: new Set(eosIds) };
    ws = createWorkspace(config.vocab_size, SHOW_TOP);
    msPerToken = undefined;

    // A first short reply compiles the GPU's shaders, so the first chat doesn't
    // wait for them, and a GPU that cannot run the model fails here, in "load".
    phase('warmup');
    const tWarm = now();
    const warm = tokenizer.apply_chat_template([{ role: 'user', content: 'Hi' }], { add_generation_prompt: true, return_dict: true });
    await model.generate({ ...warm, max_new_tokens: 2, do_sample: false });
    const warmupMs = now() - tWarm;

    // One KV cache tensor's bytes per token, from the session's own metadata.
    const kv = (session.inputMetadata ?? []).filter((m) => /^past_key_values\..*\.(key|value)$/.test(m.name));
    const bytesPer = { float16: 2, float32: 4 };
    const kvBytesPerToken = kv.length
      ? kv[0].shape.filter((d) => typeof d === 'number').reduce((a, b) => a * b, 1) * (bytesPer[kv[0].type] ?? 0)
      : 0;

    phase('ready');
    post({
      type: 'ready',
      timings: { totalMs: now() - t0, downloadMs, sessionMs, warmupMs },
      download: { fromCache, downloadedBytes, resumedFrom },
      weights: { files: dataFiles, ...read },
      persisted: !!persisted,
      session: {
        device: session.config?.device ?? null,
        dtype: session.config?.dtype ?? null,
        inputs: session.inputNames.length,
        numLogitsToKeep: session.inputNames.includes('num_logits_to_keep'),
        kvTensors: kv.length,
        kvType: kv[0]?.type ?? null,
        kvBytesPerToken,
      },
      model: {
        vocabSize: config.vocab_size,
        hiddenSize: config.hidden_size,
        layers: config.num_hidden_layers,
        layerTypes: config.layer_types ?? null,
        maxPositions: config.max_position_embeddings ?? null,
        modelType: config.model_type,
      },
      device: ortDevice
        ? {
            features: [...ortDevice.features].sort(),
            f16: ortDevice.features.has('shader-f16'),
            maxBufferSize: ortDevice.limits.maxBufferSize,
            maxStorageBufferBindingSize: ortDevice.limits.maxStorageBufferBindingSize,
            info: ortDevice.adapterInfo
              ? {
                  vendor: ortDevice.adapterInfo.vendor || '',
                  architecture: ortDevice.adapterInfo.architecture || '',
                  device: ortDevice.adapterInfo.device || '',
                  description: ortDevice.adapterInfo.description || '',
                }
              : null,
          }
        : null,
      gpu: gpuSnapshot(),
      storage: await opfs.storageEstimate(),
      stored: await storedState(),
    });
  } catch (error) {
    for (const w of weights) w.close();
    await model?.dispose?.().catch(() => undefined);
    loaded = null;
    throw error;
  }
}

async function unload() {
  const previous = loaded;
  loaded = null;
  await dropReuse();
  await previous?.model.dispose().catch(() => undefined);
  gpu.drain?.();
}

// ---- Generate ---------------------------------------------------------------------

const idsOf = (tensor) => Array.from(tensor.data, Number);
const startsWith = (tokens, prefix) => prefix.length <= tokens.length && prefix.every((t, i) => t === tokens[i]);

function textOf(id) {
  let text = tokenText.get(id);
  if (text === undefined) {
    text = loaded.tokenizer.decode([id], { skip_special_tokens: false });
    tokenText.set(id, text);
  }
  return text;
}

/**
 * Samples each token itself (js/sampler.js) from the logits generate() hands
 * it, then leaves only that token possible, so generate()'s greedy pick is it.
 */
class SamplerProcessor extends LogitsProcessor {
  constructor(params, rng, onStep) {
    super();
    this.params = params;
    this.rng = rng;
    this.onStep = onStep;
    this.seen = new Set();
    this.primed = false;
    this.index = 0;
  }

  _call(inputIds, logits) {
    const ids = inputIds[0];
    if (this.primed) this.seen.add(Number(ids[ids.length - 1]));
    else {
      for (const id of ids) this.seen.add(Number(id));
      this.primed = true;
    }
    const data = logits.data;
    const step = sampleStep(data, this.seen, this.params, this.rng, ws);
    data.fill(-Infinity);
    data[step.chosen] = 0;
    this.onStep(step, this.index++);
    return logits;
  }
}

async function generate({ id, messages, params }) {
  if (!loaded) throw new Error('No model is loaded.');
  const { tokenizer, model } = loaded;
  const tStart = now();
  const cap = Math.max(256, Math.min(CONTEXT.max, Math.floor(params.contextCap)));

  // The prompt, with the oldest turns dropped until it fits the context.
  let kept = messages.slice();
  let dropped = 0;
  let inputs;
  let total;
  for (;;) {
    inputs = tokenizer.apply_chat_template(kept, { add_generation_prompt: true, return_dict: true });
    total = inputs.input_ids.dims[1];
    if (total + CONTEXT.replyReserve <= cap) break;
    const first = kept[0]?.role === 'system' ? 1 : 0;
    if (kept.length - first <= 1) {
      throw new Error(`This message is ${total} tokens. With ${CONTEXT.replyReserve} tokens kept for the reply it does not fit a ${cap}-token context.`);
    }
    kept.splice(first, 1);
    dropped++;
    if (kept[first]?.role === 'assistant' && kept.length - first > 1) {
      kept.splice(first, 1);
      dropped++;
    }
  }
  const prompt = idsOf(inputs.input_ids);

  // Reuse the last generation's cache if this prompt continues it.
  if (reuse && !startsWith(prompt, reuse.tokens)) await dropReuse();
  let cache = reuse?.cache;
  reuse = null;
  const cachedTokens = cache ? cache.get_seq_length() : 0;
  post({ type: 'prefill', id, read: cachedTokens, total, cached: cachedTokens, chunk: 0, ms: 0, dropped });

  try {
    // Read the prompt into the cache in time-sized chunks (js/prefill.js).
    phase('prefill');
    const pre = await prefill({
      total,
      held: cachedTokens,
      msPerToken,
      // (tests pass their own chunk sizes to compare chunked and unchunked reading)
      cfg: params.prefill ?? PREFILL,
      stopped: () => stopping.interrupted,
      async readTo(end) {
        const out = await model.generate({
          input_ids: inputs.input_ids.slice(null, [0, end]),
          attention_mask: inputs.attention_mask.slice(null, [0, end]),
          ...(cache && { past_key_values: cache }),
          max_new_tokens: 1,
          do_sample: false,
          return_dict_in_generate: true,
        });
        cache = out.past_key_values;
        const held = cache.get_seq_length();
        if (held !== end) throw new Error(`A prompt chunk left ${held} tokens in the cache, expected ${end}.`);
      },
      onChunk(c) {
        post({ type: 'prefill', id, read: c.to, total, cached: cachedTokens, chunk: c.to - c.from, ms: c.ms, dropped, gpu: gpuSnapshot() });
      },
    });
    if (pre.msPerToken !== undefined) msPerToken = pre.msPerToken;

    if (stopping.interrupted) {
      if (cache) reuse = { cache, tokens: prompt.slice(0, pre.read) };
      phase('idle');
      post({ type: 'done', id, reason: 'stopped', promptTokens: total, cachedTokens, prefillChunks: pre.chunks, generated: 0, context: pre.read, cap, dropped, gpu: gpuSnapshot() });
      return;
    }

    // The rest of the prompt, then the reply.
    phase('generate');
    const maxNew = Math.max(1, Math.min(Math.floor(params.maxNewTokens), cap - total));
    let generated = 0;
    let tFirst = 0;
    let tLast = 0;
    let lastId = -1;
    const rng = params.seed == null ? Math.random : mulberry32(params.seed);
    const sampling = {
      temperature: params.temperature,
      topK: params.topK,
      topP: params.topP,
      repetitionPenalty: params.repetitionPenalty,
      show: SHOW_TOP,
    };
    const processor = new SamplerProcessor(sampling, rng, (step, index) => {
      const t = now();
      if (index === 0) tFirst = t;
      tLast = t;
      generated = index + 1;
      lastId = step.chosen;
      post({
        type: 'step',
        id,
        index,
        t: t - tStart,
        chosen: { id: step.chosen, text: textOf(step.chosen) },
        top: step.top.map((c) => ({ id: c.id, text: textOf(c.id), p: c.p })),
        pModel: step.pModel,
        pSample: step.pSample,
        kept: step.kept,
        entropy: step.entropy,
        context: total + index + 1,
        ...(index % 8 === 0 && { gpu: gpuSnapshot() }),
      });
    });
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => post({ type: 'text', id, text }),
    });
    const criteria = new StoppingCriteriaList();
    criteria.push(stopping);

    const tGenerate = now();
    const out = await model.generate({
      input_ids: inputs.input_ids,
      attention_mask: inputs.attention_mask,
      ...(cache && { past_key_values: cache }),
      max_new_tokens: maxNew,
      do_sample: false,
      logits_processor: [processor],
      streamer,
      stopping_criteria: criteria,
      return_dict_in_generate: true,
    });
    cache = out.past_key_values;
    const sequence = idsOf(out.sequences);
    reuse = { cache, tokens: sequence };
    cache = undefined;

    // Speed of reading what generate() itself read of the prompt, for the next prompt's chunks.
    const tail = total - pre.read;
    if (tFirst && tail >= PREFILL.min) msPerToken = (tFirst - tGenerate) / tail;

    const reason = stopping.interrupted
      ? 'stopped'
      : loaded.eos.has(lastId)
        ? 'eos'
        : total + generated >= cap
          ? 'context'
          : 'length';
    const decodeMs = tLast - tFirst;
    phase('idle');
    post({
      type: 'done',
      id,
      reason,
      promptTokens: total,
      cachedTokens,
      prefillChunks: pre.chunks,
      // Prompt tokens actually read this turn, and how long until the first reply token.
      promptRead: total - cachedTokens,
      ttftMs: tFirst ? tFirst - tStart : null,
      generated,
      decodeMs,
      // Tokens per second over the reply, not counting the wait for the first token.
      tokensPerSecond: generated > 1 && decodeMs > 0 ? ((generated - 1) / decodeMs) * 1000 : null,
      promptTokensPerSecond: tFirst && total - cachedTokens > 0 ? ((total - cachedTokens) / (tFirst - tStart)) * 1000 : null,
      totalMs: now() - tStart,
      context: sequence.length,
      cap,
      dropped,
      gpu: gpuSnapshot(),
    });
  } catch (error) {
    await cache?.dispose?.().catch(() => undefined);
    await dropReuse();
    throw error;
  }
}

// ---- Messages -----------------------------------------------------------------------

async function run(where, task, id) {
  if (busy) {
    post({ type: 'error', where, id, message: 'The model is busy.' });
    return;
  }
  busy = true;
  try {
    await task();
  } catch (error) {
    phase(loaded ? 'idle' : 'unloaded');
    post({ type: 'error', where, id, message: error instanceof Error ? error.message : String(error), gpu: gpuSnapshot() });
  } finally {
    busy = false;
  }
}

self.onmessage = ({ data }) => {
  switch (data.type) {
    case 'hello':
      hello().catch((error) => post({ type: 'error', where: 'hello', message: String(error?.message ?? error) }));
      break;
    case 'load':
      run('load', load);
      break;
    case 'generate':
      stopping.reset();
      run('generate', () => generate(data), data.id);
      break;
    case 'stop':
      stopping.interrupt();
      break;
    case 'reset':
      if (!busy) dropReuse().then(() => post({ type: 'gpu', gpu: gpuSnapshot() }));
      break;
    case 'unload':
      run('unload', async () => {
        await unload();
        phase('unloaded');
        post({ type: 'unloaded', gpu: gpuSnapshot() });
      });
      break;
    case 'purge':
      run('purge', async () => {
        await opfs.purge();
        post({ type: 'purged', storage: await opfs.storageEstimate(), stored: await storedState() });
      });
      break;
    default:
      post({ type: 'error', where: 'worker', message: `Unknown message: ${data.type}` });
  }
};

self.addEventListener('unhandledrejection', (event) => {
  post({ type: 'error', where: 'worker', message: String(event.reason?.message ?? event.reason) });
});

post({ type: 'boot' });
