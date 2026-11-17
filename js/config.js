// BLIK llm configuration.
//
// To run your own weights, change MODEL below and nothing else:
//   repo      a Hugging Face model repo laid out for Transformers.js
//             (config.json, tokenizer files, onnx/model_<dtype>.onnx + .onnx_data)
//   revision  a commit SHA (recommended: a pinned SHA never changes under you),
//             or a branch or tag name
//   dtype     which ONNX file to load: "q4f16" loads onnx/model_q4f16.onnx
//
// The graph must take a `num_logits_to_keep` input (see train/README.md);
// without it a long prompt returns logits for every token and the tab dies.

export const MODEL = {
  repo: 'daveycodez/LFM2.5-1.2B-Web-ONNX',
  revision: 'f8df717df1ee01ccb899484d7f25cddfa78dd49f',
  dtype: 'q4f16',
};

// Shown in the model card window. Text only: nothing here changes what is loaded.
export const MODEL_CARD = {
  name: 'LFM2.5-1.2B-Instruct',
  maker: 'Liquid AI, Inc.',
  build: 'Unofficial community ONNX build by daveycodez. Same weights as ' +
    'LiquidAI/LFM2.5-1.2B-Instruct-ONNX (q4f16), plus a num_logits_to_keep input.',
  license: 'LFM Open License v1.0',
  licenseNote: 'Not Apache. Commercial use is licensed only for entities with ' +
    'annual revenue under 10 million US dollars.',
  licenseUrl: 'https://huggingface.co/daveycodez/LFM2.5-1.2B-Web-ONNX/blob/main/LICENSE',
};

// Where files are downloaded from. `{repo}/resolve/{revision}/{file}` is appended.
export const HUB = 'https://huggingface.co/';

// Context limits, in tokens (prompt plus reply).
//
// One KV cache tensor of this model takes 1,024 bytes per token. ONNX Runtime
// keeps GPU buffers in size buckets (..., 2 MiB, 4 MiB, 8 MiB): past 2,048
// tokens the cache moves to the 4 MiB bucket, where ONNX Runtime keeps too few
// spares and starts creating and destroying buffers on every token; past 4,096
// it moves to the 8 MiB bucket and live GPU memory jumps
// (microsoft/onnxruntime#33751). So 2,048 is the default and 4,096 the ceiling.
export const CONTEXT = {
  default: 2048,
  max: 4096,
  choices: [1024, 2048, 3072, 4096],
  // Tokens kept free for the reply when old turns are dropped to fit.
  replyReserve: 64,
};

// Long prompts are read into the KV cache in chunks. Safari kills its GPU
// process when a command buffer has not finished 3 s after it was committed
// (microsoft/onnxruntime#33753), so a chunk is sized from the measured speed
// to take about `targetMs`, and is never larger than `max`.
export const PREFILL = {
  targetMs: 1000,
  first: 64,
  min: 16,
  max: 256,
};

// Sampling defaults. Temperature, top-k and repetition penalty are the values
// Liquid AI's model card recommends; the card gives no top-p, so it starts off.
export const SAMPLING = {
  temperature: 0.1,
  topP: 1.0,
  topK: 50,
  repetitionPenalty: 1.05,
  maxNewTokens: 512,
};

// How many next-token candidates the logits window shows.
export const SHOW_TOP = 8;

// Reuse GPU buffers that ONNX Runtime destroys (see CONTEXT above): without
// this, every token past 2,048 creates and destroys several 4 MiB buffers,
// which Safari is slow to give back.
export const GPU_BUFFER_POOL = true;
