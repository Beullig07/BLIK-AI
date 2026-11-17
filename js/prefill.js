// Reading a long prompt into the KV cache in chunks.
//
// Safari kills its GPU process when a WebGPU command buffer has not finished
// 3 seconds after it was committed, and ONNX Runtime submits a whole forward
// pass at once. On an iPhone one pass over roughly 500 to 900 prompt tokens
// crosses that line and the page loses its GPU (microsoft/onnxruntime#33753).
//
// A fixed chunk size is not enough: what matters is queued GPU time, which
// depends on the model, the device, how hot it is and how long the context
// already is. So each chunk is sized from how long the last one took, to take
// about `targetMs`, never more than `max` tokens and never more than twice the
// previous chunk. Every chunk ends with the logits being read back, which
// drains the GPU queue before the next chunk is submitted.
//
// The technique follows daveycodez/tanstack-ai-browser (MIT); upstream support
// is requested in huggingface/transformers.js#1836.
//
// Pure logic, no browser APIs: tests/unit.mjs runs it in Node.

/** Tokens for the next chunk, given the speed measured so far. */
export function nextChunk(msPerToken, previous, cfg) {
  if (!(msPerToken > 0)) return Math.max(cfg.min, Math.min(cfg.max, cfg.first));
  let size = Math.floor(cfg.targetMs / msPerToken);
  if (previous > 0) size = Math.min(size, previous * 2);
  return Math.max(cfg.min, Math.min(cfg.max, size));
}

/**
 * Reads prompt tokens [held, total) into the cache until what is left fits in
 * one chunk; the caller's generate() reads that last part and then answers.
 *
 * @param {object} o
 * @param {number} o.total      prompt length in tokens
 * @param {number} o.held       tokens the cache already holds
 * @param {number|undefined} o.msPerToken  speed measured earlier, if any
 * @param {(end:number) => Promise<void>} o.readTo  read the prompt up to `end` into the cache
 * @param {() => boolean} o.stopped
 * @param {(c:{from:number,to:number,ms:number,msPerToken:number}) => void} [o.onChunk]
 * @param {() => number} [o.now]
 * @param {object} o.cfg        PREFILL from config.js
 * @returns {Promise<{read:number, chunks:number, msPerToken:number|undefined}>}
 */
export async function prefill({ total, held, msPerToken, readTo, stopped, onChunk, now = () => performance.now(), cfg }) {
  let read = held;
  let chunks = 0;
  let previous = 0;
  for (;;) {
    const size = nextChunk(msPerToken, previous, cfg);
    if (total - read <= size || stopped()) break;
    const end = read + size;
    const started = now();
    await readTo(end);
    const ms = now() - started;
    msPerToken = ms / size;
    onChunk?.({ from: read, to: end, ms, msPerToken });
    read = end;
    previous = size;
    chunks++;
  }
  return { read, chunks, msPerToken };
}
