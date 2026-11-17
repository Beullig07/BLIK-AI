// Drives the real page in a browser and saves screenshots of it.
//
//   node tests/serve.cjs &
//   node tests/ui.cjs <out-dir> [steps]     steps: any of  open menus keyboard load chat crash
//
// The screenshots show whatever the page really did: a real model, real numbers.

const fs = require('node:fs');
const path = require('node:path');
const { launch, URL_BASE } = require('./browser.cjs');

const out = process.argv[2] || 'screenshots';
const steps = process.argv.slice(3).length ? process.argv.slice(3) : ['open'];
const has = (name) => steps.includes(name);
const check = (ok, what) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`);
  if (!ok) process.exitCode = 1;
};

(async () => {
  fs.mkdirSync(out, { recursive: true });
  // iPhone 17 Pro: 402 x 874 points at 3x. Safari's own bars leave roughly 402 x 760.
  const context = await launch({ viewport: { width: 402, height: 760 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true });
  // Keep a copy of everything the worker tells the page, for the checks below.
  await context.addInitScript(() => {
    const Native = window.Worker;
    window.blikLog = [];
    window.Worker = class extends Native {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', (event) => window.blikLog.push(event.data));
      }
    };
  });
  context.setDefaultTimeout(60000);
  let page = context.pages()[0] || (await context.newPage());
  const watch = (p) => {
    p.on('pageerror', (error) => console.log(`  [pageerror] ${error.message}`));
    p.on('console', (m) => m.type() === 'error' && !/favicon/.test(m.text()) && console.log(`  [console.error] ${m.text().slice(0, 300)}`));
  };
  watch(page);
  const opened = async () => {
    await page.waitForFunction(() => document.querySelectorAll('#checks > div').length > 0, null, { timeout: 30000 });
    await page.evaluate(() => document.fonts.ready);
  };
  const shot = async (name) => {
    await page.waitForTimeout(250);
    await page.screenshot({ path: path.join(out, `${name}.png`) });
    console.log(`saved ${path.join(out, `${name}.png`)}`);
  };

  try {
    await page.goto(URL_BASE);
    await opened();
    if (has('open')) {
      await shot('1-open');
      const widths = await page.evaluate(() => ({ doc: document.documentElement.scrollWidth, view: innerWidth }));
      check(widths.doc <= widths.view, `no horizontal overflow (${widths.doc} <= ${widths.view})`);
    }

    if (has('menus')) {
      await page.click('[data-menu="menu-model"]');
      await shot('7-menu');
      const items = await page.evaluate(() => [...document.querySelectorAll('#menu-model button')].map((b) => [b.textContent, !b.disabled]));
      console.log(`Model menu: ${JSON.stringify(items)}`);
      check(await page.isVisible('#menu-model'), 'the Model menu opens');
      await page.mouse.click(200, 500);
      check(!(await page.isVisible('#menu-model')), 'a tap elsewhere closes the menu');
      await page.click('[data-menu="menu-blik"]');
      await page.click('[data-action="about"]');
      await shot('8-about');
      const about = await page.textContent('#dialog-body');
      check(/LFM Open License v1\.0/.test(about) && /10 million/.test(about), 'the About box states the model license and its revenue limit');
      await page.click('#dialog-actions button');
      check(!(await page.isVisible('#dialog-layer')), 'OK closes the About box');
    }

    if (has('keyboard')) {
      // A phone keyboard shrinks the visual viewport. Shrinking the window does the same here.
      const full = page.viewportSize();
      await page.setViewportSize({ width: full.width, height: 420 });
      await page.waitForTimeout(300);
      const fit = await page.evaluate(() => {
        const box = document.getElementById('composer').getBoundingClientRect();
        return { shell: document.getElementById('shell').getBoundingClientRect().height, visible: window.visualViewport.height, composerBottom: box.bottom };
      });
      await shot('9-keyboard');
      check(Math.abs(fit.shell - fit.visible) < 1, `the shell is as tall as the visible area (${fit.shell} vs ${fit.visible})`);
      check(fit.composerBottom <= fit.visible, `the message field is inside the visible area (bottom ${fit.composerBottom} <= ${fit.visible})`);
      await page.setViewportSize(full);
      await page.waitForTimeout(200);
    }

    if (has('load')) {
      await page.click('#load-button');
      await page.waitForFunction(() => !document.getElementById('progress').hidden, null, { timeout: 10000 });
      if (has('open')) await shot('2-loading');
      await page.waitForFunction(() => !document.getElementById('input').disabled, null, { timeout: 3600000 });
      await shot('3-loaded');
      check(await page.evaluate(() => document.getElementById('loader').hidden), 'the load panel is gone once the model is ready');
    }

    if (has('chat')) {
      const ask = async (text, name, { midway } = {}) => {
        await page.fill('#input', text);
        await page.click('#send-button');
        if (midway) {
          await page.waitForFunction(() => document.querySelector('.msg.assistant')?.textContent.length > 12, null, { timeout: 3600000 });
          await shot(midway);
        }
        await page.waitForFunction(() => document.getElementById('send-button').textContent === 'Send', null, { timeout: 3600000 });
        await shot(name);
      };
      await ask('What is the capital of France? One short sentence.', '5-reply', { midway: '4-writing' });
      const reply = await page.evaluate(() => [...document.querySelectorAll('.msg.assistant')].at(-1)?.textContent ?? '');
      console.log(`reply: ${JSON.stringify(reply)}`);
      check(/paris/i.test(reply), 'the reply on screen mentions Paris');
      const status = await page.evaluate(() => ({ ctx: document.getElementById('ctx-text').textContent, speed: document.getElementById('speed-text').textContent, ttft: document.getElementById('ttft-text').textContent }));
      console.log(`status strip: ${JSON.stringify(status)}`);
      check(/^\d[\d,]*\/2,048$/.test(status.ctx) && !status.ctx.startsWith('0/'), 'context used is shown');
      check(/^\d+\.\d tok\/s$/.test(status.speed), 'tokens per second is shown');
      await ask('And of Japan? Then name two things to see there.', '6-second-turn');
      const turns = await page.evaluate(() => window.blikLog.filter((m) => m.type === 'done').map((m) => ({ prompt: m.promptTokens, cached: m.cachedTokens, generated: m.generated, reason: m.reason })));
      console.log(`turns: ${JSON.stringify(turns)}`);
      check(turns.length === 2 && turns[1].cached > 0, `the second turn reused the first turn's cache (${turns[1]?.cached} tokens)`);

      await page.click('[data-menu="menu-chat"]');
      await page.click('[data-action="new-chat"]');
      const after = await page.evaluate(() => ({ messages: document.querySelectorAll('.msg').length, ctx: document.getElementById('ctx-text').textContent }));
      check(after.messages === 0 && after.ctx.startsWith('0/'), 'New chat clears the messages and the context count');
    }

    if (has('crash')) {
      // 1. Reloading on purpose in the middle of a load is not a crash.
      await page.goto(URL_BASE);
      await opened();
      await page.click('#load-button');
      await page.waitForFunction(() => /Moving the weights|Compiling shaders/.test(document.getElementById('progress-text').textContent), null, { timeout: 300000 });
      await page.reload();
      await opened();
      check(!(await page.isVisible('#crash-note')), 'a deliberate reload during a load leaves no crash note');

      // 2. The tab's process dying in the middle of a load is one.
      await page.click('#load-button');
      await page.waitForFunction(() => /Moving the weights|Compiling shaders/.test(document.getElementById('progress-text').textContent), null, { timeout: 300000 });
      const at = await page.textContent('#progress-text');
      // Kill the tab's process, as the phone does when memory runs out. (The
      // command never answers, since the process it was sent to is gone.)
      const cdp = await context.newCDPSession(page);
      const crashed = new Promise((resolve) => page.once('crash', resolve));
      cdp.send('Page.crash').catch(() => undefined);
      await Promise.race([crashed, new Promise((resolve) => setTimeout(resolve, 10000))]);
      page.close().catch(() => undefined);
      page = await context.newPage();
      watch(page);
      await page.goto(URL_BASE);
      await opened();
      const note = (await page.isVisible('#crash-note')) ? await page.textContent('#crash-text') : '';
      console.log(`killed at ${JSON.stringify(at)}; the next visit says: ${JSON.stringify(note)}`);
      await shot('10-after-a-kill');
      check(/last load .* ended in a tab reload, probably out of memory/.test(note), 'after the tab is killed mid-load, the next visit says so');
      await page.reload();
      await opened();
      check(!(await page.isVisible('#crash-note')), 'the note is shown once');
    }
  } catch (error) {
    console.log(`FAIL  ${error.message.split('\n')[0]}`);
    process.exitCode = 1;
    await page.screenshot({ path: path.join(out, 'failure.png') }).catch(() => undefined);
    const state = await page
      .evaluate(() => ({
        problem: document.getElementById('problem').hidden ? null : document.getElementById('problem-text').textContent,
        progress: document.getElementById('progress-text').textContent,
        last: (window.blikLog || []).slice(-4).map((m) => JSON.stringify(m).slice(0, 300)),
      }))
      .catch(() => null);
    console.log(`page state: ${JSON.stringify(state, null, 1)}`);
  } finally {
    await context.close();
  }
})();
