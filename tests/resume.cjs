// Download resume test: starts a load, kills the page partway through the
// weights download (as a tab kill would), opens it again and loads.
// The second load must continue from the bytes already in OPFS.
//
//   node tests/serve.cjs &
//   node tests/resume.cjs
//
// Uses its own browser profile, so it does not touch the weights other tests keep.

const os = require('node:os');
const path = require('node:path');
const fs = require('node:fs');
const { launch, openHarness, mib } = require('./browser.cjs');

const profile = path.join(os.tmpdir(), 'blik-test-profile-resume');
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) process.exitCode = 1;
};

(async () => {
  fs.rmSync(profile, { recursive: true, force: true });

  // First visit: start loading, kill the page once some of the weights are down.
  let context = await launch({ profile });
  let h = await openHarness(context, { quiet: true });
  const killAt = 150 * 1048576;
  const partway = h.waitFor((m) => m.type === 'download' && m.loaded >= killAt && m.loaded < m.total, { timeout: 1800000 });
  await h.send({ type: 'load' });
  const seen = await partway;
  console.log(`first visit      killed at ${mib(seen.loaded)} of ${mib(seen.total)}`);
  await context.close();

  // Second visit: what is stored, then load again.
  context = await launch({ profile });
  h = await openHarness(context, { quiet: true });
  await h.send({ type: 'hello' });
  const env = await h.waitFor((m) => m.type === 'env');
  const part = env.stored.files.find((f) => /\.onnx_data$/.test(f.path));
  console.log(`second visit     stored: ${part ? `${mib(part.size)}, complete: ${part.complete}` : 'nothing'}; partial: ${env.stored.partial}`);
  check(!!part && !part.complete && part.size >= killAt, 'a partial weights file survived the kill');

  let first = null;
  const ready = h.waitFor((m) => m.type === 'ready', {
    timeout: 3600000,
    onMessage(m) {
      if (m.type === 'download' && !first) first = m;
    },
  });
  await h.send({ type: 'load' });
  const r = await ready;
  console.log(`second load      first progress at ${mib(first.loaded)}, resumed from ${mib(r.download.resumedFrom)}, downloaded ${mib(r.download.downloadedBytes)} this time`);
  check(r.download.resumedFrom >= killAt, `the download resumed (from ${mib(r.download.resumedFrom)})`);
  check(r.download.resumedFrom + r.download.downloadedBytes === first.total, 'resumed bytes + new bytes = the whole file');
  check(r.stored.weightsComplete, 'the weights are complete after resuming');

  // The resumed file must be the real thing: the model has to answer correctly from it.
  const done = h.waitFor((m) => m.type === 'done', { timeout: 600000 });
  let text = '';
  h.waitFor((m) => (m.type === 'text' && (text += m.text), false), { timeout: 600000, failOnError: false }).catch(() => undefined);
  await h.send({ type: 'generate', id: 1, messages: [{ role: 'user', content: 'What is the capital of France? Answer in one short sentence.' }], params: { temperature: 0, topK: 50, topP: 1, repetitionPenalty: 1.05, maxNewTokens: 12, contextCap: 2048 } });
  await done;
  console.log(`reply            ${JSON.stringify(text)}`);
  check(/paris/i.test(text), 'the model loaded from the resumed file answers correctly');

  await context.close();
  fs.rmSync(profile, { recursive: true, force: true });
})().catch((error) => {
  console.log(`FAIL  ${error.message}`);
  process.exit(1);
});
