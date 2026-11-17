// Launches Chromium for the browser tests and talks to the page's worker.
//
// Environment variables (all optional):
//   BLIK_URL       where the site is served        (default http://127.0.0.1:8378/)
//   BLIK_CHROME    Chromium/Chrome executable      (default: Playwright's Chromium)
//   BLIK_PROFILE   browser profile folder; keeps the downloaded weights between runs
//   BLIK_VK_ICD    a Vulkan ICD json to force (see tests/README.md, software GPU)
//   BLIK_PROXY     proxy for outbound requests     (default: $HTTPS_PROXY)
//   BLIK_HEADED    set to 1 to show the window

const path = require('node:path');
const os = require('node:os');
const { chromium } = require('playwright');

const URL_BASE = process.env.BLIK_URL || 'http://127.0.0.1:8378/';

async function launch({ profile, viewport, deviceScaleFactor, isMobile, hasTouch } = {}) {
  const env = { ...process.env };
  if (process.env.BLIK_VK_ICD) {
    env.VK_ICD_FILENAMES = process.env.BLIK_VK_ICD;
    env.VK_DRIVER_FILES = process.env.BLIK_VK_ICD;
    env.XDG_RUNTIME_DIR ||= os.tmpdir();
  }
  const proxy = process.env.BLIK_PROXY || process.env.HTTPS_PROXY || process.env.https_proxy;
  const context = await chromium.launchPersistentContext(profile || process.env.BLIK_PROFILE || path.join(os.tmpdir(), 'blik-test-profile'), {
    executablePath: process.env.BLIK_CHROME || undefined,
    channel: process.env.BLIK_CHROME ? undefined : 'chromium',
    headless: false, // the real browser in its "new" headless mode, set by the flag below
    args: [
      ...(process.env.BLIK_HEADED ? [] : ['--headless=new']),
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-angle=vulkan',
      '--ignore-gpu-blocklist',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ],
    env,
    ...(proxy && { proxy: { server: proxy, bypass: 'localhost,127.0.0.1' } }),
    viewport: viewport || { width: 402, height: 760 },
    ...(deviceScaleFactor && { deviceScaleFactor }),
    ...(isMobile && { isMobile }),
    ...(hasTouch && { hasTouch }),
  });
  return context;
}

/** Opens the worker harness and returns helpers to send messages and wait for replies. */
async function openHarness(context, { quiet = false } = {}) {
  const page = context.pages()[0] || (await context.newPage());
  const messages = [];
  const waiters = new Set();
  await page.exposeFunction('blikMessage', (message) => {
    messages.push(message);
    for (const waiter of waiters) waiter(message);
  });
  page.on('console', (m) => {
    if (!quiet) console.log(`  [console.${m.type()}] ${m.text().slice(0, 400)}`);
  });
  page.on('pageerror', (error) => console.log(`  [pageerror] ${error.message}`));
  page.on('crash', () => console.log('  [page crashed]'));
  await page.goto(URL_BASE + 'tests/harness.html');

  const send = (message) => page.evaluate((m) => window.blikSend(m), message);
  /** Resolves with the first message matching `test`; rejects on an error message or timeout. */
  const waitFor = (test, { timeout = 60000, onMessage, failOnError = true } = {}) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        waiters.delete(waiter);
        reject(new Error(`Timed out after ${timeout} ms waiting for a worker message.`));
      }, timeout);
      const waiter = (message) => {
        onMessage?.(message);
        const error = failOnError && message.type === 'error';
        if (!error && !test(message)) return;
        clearTimeout(timer);
        waiters.delete(waiter);
        if (error) reject(new Error(`Worker error (${message.where}): ${message.message}`));
        else resolve(message);
      };
      waiters.add(waiter);
    });
  await (messages.some((m) => m.type === 'boot') ? Promise.resolve() : waitFor((m) => m.type === 'boot', { timeout: 30000 }));
  return { page, messages, send, waitFor };
}

const mib = (bytes) => `${(bytes / 1048576).toFixed(1)} MiB`;

module.exports = { launch, openHarness, mib, URL_BASE };
