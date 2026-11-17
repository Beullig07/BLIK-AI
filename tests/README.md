# Tests

Nothing here is needed to run or deploy BLIK llm. These are the checks used
while building it.

## Unit tests (no browser)

```sh
node tests/unit.mjs
```

Sampling (temperature, top-k, top-p, repetition penalty) and prefill chunk
sizing. Needs Node 22 or newer, nothing else.

## Browser tests (the real model)

They drive a real Chromium with Playwright against the real weights. Nothing
is mocked: if the model cannot run, the test fails.

```sh
npm install playwright        # once, anywhere; then: npx playwright install chromium
node tests/serve.cjs &        # serves the repo on http://127.0.0.1:8378/
node tests/runtime.cjs env load chat reuse long stop sample context
node tests/resume.cjs         # kills a download partway, then resumes it
node tests/pool.cjs           # GPU buffer churn past 2,048 tokens (slow)
node tests/ui.cjs shots open menus keyboard load chat crash
```

| Script | What it checks |
|---|---|
| `runtime.cjs` | Adapter probe; load (weights streamed to OPFS, read on demand, on the GPU); a greedy chat; KV cache reuse between turns; a long prompt read in chunks gives the same tokens as one pass; stop; seeded sampling with top-p; context cap. |
| `resume.cjs` | A download interrupted by a killed page resumes from the bytes already stored, and the resumed file loads and answers. |
| `pool.cjs` | Counts GPU buffers created and destroyed per token as the context grows past 2,048, with and without the buffer pool, and checks the pool does not change the output. |
| `ui.cjs` | The page itself: load panel, menus, About box, the shell following the visible area, load, a two-turn chat, new chat, and the note shown after a tab is killed mid-load. Saves screenshots. |

Settings are environment variables, listed at the top of `browser.cjs`. The
weights (725 MiB) are downloaded once into the browser profile named by
`BLIK_PROFILE` and reused.

On a machine with a real GPU, plain `node tests/runtime.cjs` is enough.

## Running them without a GPU

A machine with no GPU still has a WebGPU adapter in Chromium (SwiftShader, a
CPU renderer), but it has no `shader-f16`, which the q4f16 model needs. Mesa's
CPU renderer (lavapipe) does have it, but Chromium refuses CPU adapters other
than SwiftShader. `vkshim/shim.c` is a small Vulkan driver wrapper that
forwards everything to lavapipe and reports SwiftShader's vendor and device
IDs, so Chromium accepts it. It is a test aid only; it is not part of the site.

```sh
apt-get install mesa-vulkan-drivers libvulkan-dev
gcc -shared -fPIC -O2 -o libvkshim.so tests/vkshim/shim.c -ldl
# an ICD file pointing at the wrapper:
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"'$PWD'/libvkshim.so","api_version":"1.4.318"}}' > shim_icd.json
# a copy of Chromium without its own SwiftShader, so the wrapper is the only adapter:
cp -r "$(dirname "$(node -e "console.log(require('playwright').chromium.executablePath())")")" chrome-lvp
rm chrome-lvp/libvk_swiftshader.so chrome-lvp/vk_swiftshader_icd.json
BLIK_CHROME=$PWD/chrome-lvp/chrome BLIK_VK_ICD=$PWD/shim_icd.json node tests/runtime.cjs
```

The page then reports the adapter as `google swiftshader` and as a software
renderer, which is true: it is a CPU. Everything runs, at about 1 token per
second. Speeds measured this way say nothing about a phone.
