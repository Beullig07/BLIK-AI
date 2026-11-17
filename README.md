# BLIK llm

A chat page that runs a 1.2-billion-parameter language model on your phone's
GPU, in the browser, with WebGPU. Static files only: no server, no build step,
no framework. What you type never leaves the device.

- **Model:** LFM2.5-1.2B-Instruct by Liquid AI, q4f16 ONNX, 725 MiB of weights
- **Runtime:** Transformers.js 4.3.2 on ONNX Runtime Web (WebGPU), in a Web Worker, vendored in this repo
- **Target:** iPhone 17 Pro, iOS 27.2, Safari
- **Look:** 1-bit, like a 1980s desktop

> **Not yet run on an iPhone.** Everything was tested in headless Chromium on
> a software WebGPU adapter (a CPU). The parts that only a real iPhone can
> show are listed under [On-phone checklist](#on-phone-checklist).

## Deploy to GitHub Pages

1. **Make the repository public.** On a free GitHub plan, Pages only works
   for public repositories. (Settings > General > Danger Zone > Change
   visibility.)
2. Settings > Pages > Build and deployment > Source: **Deploy from a branch**.
3. Pick the branch that holds these files and the folder **/ (root)**, then Save.
4. Wait for the "pages build and deployment" run under Actions, then open
   `https://<your-user>.github.io/<repo>/` on the phone, in Safari. For this
   repository that should be `https://beullig07.github.io/BLIK-AI/`.

There is nothing to build. `.nojekyll` tells Pages to serve the files as they are.

To try it on a computer first: `node tests/serve.cjs`, then open
`http://127.0.0.1:8378/` in a browser with WebGPU.

## Using it

Tap **Load model**. The first time, 725 MiB of weights download straight from
Hugging Face into the browser's private file storage; after that they load
from the device. Nothing is loaded until you tap.

The load panel shows what the browser's GPU adapter is and offers. If WebGPU
or `shader-f16` is missing, it says so and the button stays off. There is no
CPU fallback. A software adapter is labelled "software, not a GPU".

## Swap in your own model

Model repo, revision and dtype live in one place, the top of
[`js/config.js`](js/config.js):

```js
export const MODEL = {
  repo: 'daveycodez/LFM2.5-1.2B-Web-ONNX',
  revision: 'f8df717df1ee01ccb899484d7f25cddfa78dd49f',
  dtype: 'q4f16',
};
```

The repo must be public on Hugging Face, laid out for Transformers.js
(`config.json`, tokenizer files, `onnx/model_<dtype>.onnx` and its
`.onnx_data`), and its graph must take a `num_logits_to_keep` input.
[`train/`](train/README.md) has the scripts to fine-tune LFM2.5-1.2B, convert
it and check the result.

## License

**The model is not Apache-licensed.** LFM2.5-1.2B-Instruct is under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE).
Commercial use is licensed only for entities with annual revenue under 10
million US dollars; above that it is not licensed. The weights are not in
this repository: the page downloads them from
[daveycodez/LFM2.5-1.2B-Web-ONNX](https://huggingface.co/daveycodez/LFM2.5-1.2B-Web-ONNX),
an unofficial community build with the same weights as Liquid AI's q4f16 file
(same SHA-256) plus a `num_logits_to_keep` input.

Vendored in `vendor/`, each with its license file:

| What | License |
|---|---|
| Transformers.js 4.3.2 | Apache-2.0 |
| ONNX Runtime Web 1.31.0-dev.20260914-8d85527a0 (WebAssembly builds) | MIT |
| Jersey 15 and W95FA fonts | SIL Open Font License 1.1 |

The OPFS weights reader and the time-sized prompt reading follow
[daveycodez/tanstack-ai-browser](https://github.com/daveycodez/tanstack-ai-browser) (MIT).

This repository's own code has no license file yet. Add one before inviting
others to reuse it.

## Known iPhone limits

A Safari tab on an iPhone is killed at roughly 2 GB of memory, GPU buffers
included (2,048 MB was measured by others on an iPhone 16 Pro Max; the 17 Pro
is unmeasured). A kill looks like a page reload, not an error. Stock
Transformers.js does not survive this with a model this size. What BLIK does
about it, and what is left:

| Limit | What BLIK does | Still true |
|---|---|---|
| **Download.** Stock loading holds the weights in memory about twice; the tab dies as the download ends, nothing is cached, and every retry dies the same way. | Streams the weights into the browser's private file system (OPFS) in chunks, and resumes a partial download. ONNX Runtime then reads them from the file one tensor at a time. The largest block held in memory is 64 MiB. | Needs about 730 MiB of free storage. Safari may delete a site's storage after 7 days without a visit; the weights then download again. |
| **Long prompts.** Safari kills its GPU process when a GPU command buffer has not finished 3 s after it was committed. One pass over about 500 to 900 prompt tokens crosses that. | Reads the prompt into the KV cache in chunks sized from the measured speed, about 1 s each, 256 tokens at most. | A phone that is hot or in Low Power Mode is slower; chunks shrink to 16 tokens at the least. Below about 5 tokens per second even that takes over 3 s. |
| **Context.** GPU memory grows in steps as the context crosses 1,024, 2,048 and 4,096 tokens. Past 4,096, by an upstream report, ONNX Runtime also starts creating and destroying multi-megabyte buffers on every token, which Safari gives back slowly. | Caps the context: 2,048 tokens by default, 4,096 at most. Shows context used. When a chat no longer fits it drops the oldest turns, down to three quarters of the context so that it does not happen every turn, and says so. | See the table below. Above 2,048 is not tested on a phone, and 3,117 to 4,096 is not measured anywhere. |
| **Logits.** Without care, a prompt returns logits for every token: 210 MB for 800 tokens. | Uses a graph with a `num_logits_to_keep` input; only the last position is computed. | A model file without that input brings the problem back; the app warns when it loads one. |
| **Silent kills.** | A note is left before a load or a reply starts and removed when it ends. If the page opens with the note still there, it says the last run ended in a tab reload, probably out of memory, and at which step. | It cannot tell an out-of-memory kill from the phone killing a backgrounded tab. |
| **No threads.** GitHub Pages cannot send the headers `SharedArrayBuffer` needs. | Runs ONNX Runtime's WebAssembly single-threaded. The model itself runs on the GPU. | Tokenizing and sampling are on one CPU thread. |

GPU buffer bytes by context size, counted in the page (ONNX Runtime's
allocation, the same on any GPU):

| Context reached | GPU buffers |
|---|---|
| model loaded, no chat | 728 MiB |
| up to 1,024 tokens | 777 MiB |
| up to 2,048 tokens | 827 MiB |
| 2,049 tokens and up (measured to 3,116) | 927 MiB |

This is only the page's GPU buffers. The tab's total also has the WebAssembly
runtime, the tokenizer and Safari's own overhead, which were not measured.

While writing at about 3,100 tokens of context, each token created and
destroyed three small buffers (0.3 MiB in all) and nothing larger. The
buffer pool that [onnxruntime#33751](https://github.com/microsoft/onnxruntime/issues/33751)
suggests for larger churn is in `js/gpu.js` but switched off
(`GPU_BUFFER_POOL` in `js/config.js`): this model gave it nothing to do, so
its effect is untested.

Other things to know:

- Private Browsing may not offer file storage. The load panel then says
  "File storage: no" and loading is off.
- The first load compiles shaders and takes longer than later ones.
- The reply limit is 512 tokens per turn.
- Sampling defaults are Liquid AI's recommendation: temperature 0.1, top-k 50,
  repetition penalty 1.05. Transformers.js 4.3.2 ignores `top_p`, so BLIK does
  its own sampling.

## On-phone checklist

None of this can be checked without the phone. In Safari, on Wi-Fi:

1. **Load panel.** Note the lines: WebGPU, Adapter, shader-f16,
   maxBufferSize. There should be no "Kind: software" line.
2. **Load.** Tap Load model. The bar should reach 725.1 MiB, then "Moving the
   weights to the GPU", then "Compiling shaders", then a note with the times.
   If the page reloads instead, read the note on the load panel: it names the
   step.
3. **Chat.** Ask something short. Text should stream; note tok/s and first token.
4. **Second visit.** Reload the page. "Weights" should say "on this device".
   Load again: no download.
5. **Long prompt.** Paste about 700 words. The state should say Reading while
   CTX climbs in steps, then the reply starts. The GPU must not be lost.
6. **Long chat.** Keep going until CTX nears 2,048. Expect a note that old
   messages were left out, or "context full".
7. **Keyboard.** Tap the message field. It must stay visible above the keyboard.
8. **Stop and New chat** (Chat menu) both work mid-reply and after.
9. **Interrupted download.** Model > Delete downloaded files, start a load,
   switch to airplane mode partway, switch back, Load again: it should say
   "resumed at".
10. **Away and back.** Switch to another app for a minute and return.

Worth sending back: the Adapter line, the speeds, and the exact text of any
note about a tab reload or a lost GPU.

## What was tested, and how

All in headless Chromium 153 with Playwright, against the real weights, with
no mocks. Details and commands: [`tests/README.md`](tests/README.md).

- **Model:** `daveycodez/LFM2.5-1.2B-Web-ONNX` at `f8df717`, `model_q4f16.onnx`,
  on ONNX Runtime's WebGPU execution provider, JSPI WebAssembly build.
- **Adapter:** not a GPU. Mesa's CPU renderer (llvmpipe), reached through a
  test-only shim because Chromium's own software adapter has no `shader-f16`.
  The page labelled it "software, not a GPU". Speeds were about 1.4 tokens per
  second and mean nothing for a phone.
- **Passed:** download (725.1 MiB into OPFS), resume after a killed page,
  load, greedy chat, KV cache reuse between turns, a 316-token prompt read in
  9 chunks with the same tokens as one pass, stop, seeded sampling with
  top-p, the context cap, the note after a killed tab, menus, and the layout
  following a shrinking visible area. On the same prompt, the first-token
  probabilities matched ONNX Runtime on the CPU.
- **Not tested:** any real GPU; Safari; iOS; Metal; the 2 GB limit; the
  3-second GPU watchdog; the on-screen keyboard; heat; Low Power Mode.

## Upstream issues this works around

When these are fixed, the matching workaround can go:

- [huggingface/transformers.js#1827](https://github.com/huggingface/transformers.js/issues/1827): large downloads held in memory twice
- [huggingface/transformers.js#1836](https://github.com/huggingface/transformers.js/issues/1836): chunked prefill in `generate()`
- [huggingface/transformers.js#1835](https://github.com/huggingface/transformers.js/issues/1835): use the JSPI build where available
- [microsoft/onnxruntime#33753](https://github.com/microsoft/onnxruntime/issues/33753): one-pass prefill crashes Safari's GPU process
- [microsoft/onnxruntime#33751](https://github.com/microsoft/onnxruntime/issues/33751): GPU buffer buckets and long contexts
- [microsoft/onnxruntime#33793](https://github.com/microsoft/onnxruntime/issues/33793): constant folding duplicates a tied embedding

All were open, with no released fix, on 17 November 2026.

## Files

```
index.html, css/blik.css     the page
js/config.js                 model repo, revision, dtype; limits; sampling defaults
js/app.js                    the page's logic (menus, chat, status)
js/worker.js                 loads and runs the model
js/opfs.js                   download into OPFS, on-demand weights reader
js/prefill.js                time-sized prompt chunks
js/sampler.js                temperature, top-k, top-p, repetition penalty
js/gpu.js                    adapter probe, GPU buffer counting and pool
vendor/                      Transformers.js, ONNX Runtime WebAssembly, fonts
train/                       fine-tune and convert your own weights
tests/                       unit and browser tests
```
