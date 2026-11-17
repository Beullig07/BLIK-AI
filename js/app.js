// The page: menus, the chat window and its status strip. The model runs in
// js/worker.js; this file only shows what the worker reports.

import { MODEL, MODEL_CARD, CONTEXT, SAMPLING } from './config.js';

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};
const root = document.documentElement;

// ---- Numbers, as shown on screen ----------------------------------------------

const mib = (bytes) => `${(bytes / 1048576).toFixed(1)} MiB`;
const seconds = (ms) => `${(ms / 1000).toFixed(ms < 9950 ? 2 : 1)} s`;
const count = (n) => n.toLocaleString('en-US');
const NONE = '--';

// ---- Was the last run cut off by a tab reload? -------------------------------
//
// iPhone Safari kills a tab that uses too much memory, and that shows up as a
// plain page reload: no error reaches JavaScript. So a note is left in
// localStorage before a load or a reply starts and removed when it ends. If
// the page opens and the note is still there, the tab died in the middle.

const RUN_KEY = 'blik.run';
function readRun() {
  try {
    return JSON.parse(localStorage.getItem(RUN_KEY) ?? 'null');
  } catch {
    return null;
  }
}
function markRun(what, phase) {
  try {
    localStorage.setItem(RUN_KEY, JSON.stringify({ what, phase, at: Date.now() }));
  } catch {
    // private browsing with storage off: the note is simply not kept
  }
}
function clearRun() {
  try {
    localStorage.removeItem(RUN_KEY);
  } catch {
    // see markRun
  }
}
const interrupted = readRun();
clearRun();
// Leaving or reloading the page on purpose is not a crash.
window.addEventListener('pagehide', clearRun);

// ---- State --------------------------------------------------------------------

const state = {
  env: null,
  loading: false,
  loaded: false,
  busy: false, // a reply is being generated
  gone: false, // the GPU device was lost
  messages: [], // {role, content}
  system: '',
  current: null, // the reply in progress
  nextId: 1,
  contextCap: CONTEXT.default,
  context: 0,
};

const PHASES = {
  check: 'Checking this device',
  config: 'Reading the model config',
  download: 'Downloading the weights',
  session: 'Moving the weights to the GPU',
  warmup: 'Compiling shaders',
  prefill: 'Reading the prompt',
  generate: 'Writing',
};

// ---- The visual viewport ------------------------------------------------------
//
// On iPhone the keyboard does not resize the page; it covers it. The shell is
// sized and placed to the *visual* viewport, which is the part still visible.

function fitViewport() {
  const vv = window.visualViewport;
  if (!vv) return;
  if (vv.scale > 1.01) return; // pinch-zoomed: leave the layout alone
  root.style.setProperty('--vvh', `${vv.height}px`);
  root.style.setProperty('--vvt', `${vv.offsetTop}px`);
  root.classList.toggle('kb', window.innerHeight - vv.height > 120);
  updateScrollbar();
}
if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', fitViewport);
  window.visualViewport.addEventListener('scroll', fitViewport);
}
window.addEventListener('resize', fitViewport);

// ---- Scrolling ----------------------------------------------------------------

const messagesEl = $('messages');
const scrollbar = $('scrollbar');
const thumb = $('sb-thumb');
let stick = true; // follow new text while the reader is at the bottom

function updateScrollbar() {
  const { scrollTop, scrollHeight, clientHeight } = messagesEl;
  const track = thumb.parentElement.clientHeight;
  const idle = scrollHeight <= clientHeight + 1 || track < 24;
  scrollbar.classList.toggle('idle', idle);
  if (idle) return;
  const size = Math.max(16, Math.round((track * clientHeight) / scrollHeight));
  const top = Math.round(((track - size) * scrollTop) / (scrollHeight - clientHeight));
  thumb.style.height = `${size}px`;
  thumb.style.top = `${top}px`;
}
function scrollToEnd() {
  if (stick) messagesEl.scrollTop = messagesEl.scrollHeight;
  updateScrollbar();
}
messagesEl.addEventListener('scroll', () => {
  stick = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 28;
  updateScrollbar();
});
new ResizeObserver(() => scrollToEnd()).observe(messagesEl);
for (const arrow of scrollbar.querySelectorAll('.sb-arrow')) {
  arrow.addEventListener('click', () => messagesEl.scrollBy({ top: Number(arrow.dataset.dir) * 60 }));
}
thumb.parentElement.addEventListener('pointerdown', (event) => {
  if (event.target === thumb) return;
  const above = event.offsetY < thumb.offsetTop;
  messagesEl.scrollBy({ top: (above ? -1 : 1) * (messagesEl.clientHeight - 40) });
});
thumb.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  thumb.setPointerCapture(event.pointerId);
  const startY = event.clientY;
  const startTop = messagesEl.scrollTop;
  const track = thumb.parentElement.clientHeight - thumb.offsetHeight;
  const range = messagesEl.scrollHeight - messagesEl.clientHeight;
  const move = (e) => {
    messagesEl.scrollTop = startTop + ((e.clientY - startY) * range) / Math.max(1, track);
  };
  const up = () => {
    thumb.removeEventListener('pointermove', move);
    thumb.removeEventListener('pointerup', up);
    thumb.removeEventListener('pointercancel', up);
  };
  thumb.addEventListener('pointermove', move);
  thumb.addEventListener('pointerup', up);
  thumb.addEventListener('pointercancel', up);
});

// ---- Small pieces ----------------------------------------------------------------

const WARNING_ICON = 'M7 1h2v1H7zM6 2h1v2H6zM9 2h1v2H9zM5 4h1v2H5zM10 4h1v2h-1zM4 6h1v2H4zM11 6h1v2h-1zM3 8h1v2H3zM12 8h1v2h-1zM2 10h1v2H2zM13 10h1v2h-1zM1 12h1v2H1zM14 12h1v2h-1zM1 14h14v1H1zM7 5h2v5H7zM7 11h2v2H7z';
const INFO_ICON = 'M2 1h12v1H2zM1 2h1v12H1zM14 2h1v12h-1zM2 14h12v1H2zM7 4h2v2H7zM6 7h3v1H6zM7 8h2v4H7zM6 12h4v1H6z';

function icon(path) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'px');
  svg.setAttribute('viewBox', '0 0 16 16');
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '16');
  svg.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', path);
  svg.append(p);
  return svg;
}

function setLabel(button, text) {
  button.textContent = text;
  button.dataset.label = text;
}

/** A boxed line in the message list: something the app says, not the model. */
function addNote(text, warning = false) {
  const note = el('div', 'note');
  note.append(icon(warning ? WARNING_ICON : INFO_ICON), el('div', null, text));
  messagesEl.append(note);
  scrollToEnd();
  return note;
}

function addMessage(role, text) {
  const node = el('div', `msg ${role}`, text);
  messagesEl.append(node);
  scrollToEnd();
  return node;
}

// ---- Dialogs ----------------------------------------------------------------------

function showDialog({ title, body, actions }) {
  $('dialog-title').textContent = title;
  const bodyEl = $('dialog-body');
  bodyEl.replaceChildren(...body.map((part) => (typeof part === 'string' ? el('p', null, part) : part)));
  const actionsEl = $('dialog-actions');
  actionsEl.replaceChildren();
  for (const action of actions) {
    const button = el('button', `btn${action.primary ? ' btn-default' : ''}`);
    button.type = 'button';
    setLabel(button, action.label);
    button.addEventListener('click', () => {
      $('dialog-layer').hidden = true;
      action.run?.();
    });
    actionsEl.append(button);
  }
  $('dialog-layer').hidden = false;
  actionsEl.lastElementChild?.focus({ preventScroll: true });
}

function link(text, href) {
  const a = el('a', null, text);
  a.href = href;
  a.target = '_blank';
  a.rel = 'noopener';
  return a;
}

function showAbout() {
  const env = state.env;
  const license = el('p');
  license.append(`${MODEL_CARD.name} by ${MODEL_CARD.maker} is under the `, link(MODEL_CARD.license, MODEL_CARD.licenseUrl), `. ${MODEL_CARD.licenseNote}`);
  showDialog({
    title: 'About BLIK llm',
    body: [
      'A language model that runs in this browser tab, on the device\'s GPU through WebGPU. What you type stays on the device. The weights are downloaded once from Hugging Face and kept here.',
      license,
      MODEL_CARD.build,
      env ? `Transformers.js ${env.versions.transformers}, ONNX Runtime Web ${env.versions.onnxruntime} (${env.wasmBuild} build).` : 'The runtime has not started.',
    ],
    actions: [{ label: 'OK', primary: true }],
  });
}

// ---- Menus --------------------------------------------------------------------------

const menuEnabled = {
  'about': () => true,
  'new-chat': () => !state.busy && (state.messages.length > 0 || messagesEl.querySelector('.note')),
  'stop': () => state.busy,
  'load': () => !state.loaded && !state.loading && !!state.env?.adapter.ok && !!state.env?.opfs,
  'purge': () => !state.loading && !state.busy && (state.env?.stored.files.length ?? 0) > 0,
};
const menuRun = {
  'about': showAbout,
  'new-chat': newChat,
  'stop': stop,
  'load': load,
  'purge': confirmPurge,
};
let openMenu = null;

function closeMenu() {
  if (!openMenu) return;
  openMenu.menu.hidden = true;
  openMenu.title.classList.remove('open');
  openMenu = null;
}
for (const title of document.querySelectorAll('.menu-title')) {
  title.addEventListener('click', (event) => {
    event.stopPropagation();
    const menu = $(title.dataset.menu);
    const wasOpen = openMenu?.menu === menu;
    closeMenu();
    if (wasOpen) return;
    for (const item of menu.querySelectorAll('button')) item.disabled = !menuEnabled[item.dataset.action]();
    menu.style.left = `${title.offsetLeft}px`;
    menu.style.top = `${$('menubar').offsetHeight - 1}px`;
    menu.hidden = false;
    title.classList.add('open');
    openMenu = { menu, title };
  });
}
for (const item of document.querySelectorAll('.menu button')) {
  item.addEventListener('click', () => {
    closeMenu();
    menuRun[item.dataset.action]();
  });
}
document.addEventListener('pointerdown', (event) => {
  if (openMenu && !event.target.closest('.menu, .menu-title')) closeMenu();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeMenu();
});

// ---- The load panel -------------------------------------------------------------------

const loadButton = $('load-button');
setLabel(loadButton, 'Load model');
setLabel($('send-button'), 'Send');

function check(label, value, bad = false) {
  const row = el('div', bad ? 'bad' : null);
  row.append(el('dt', null, label), el('span', 'lead'), el('dd', null, value));
  return row;
}

function storedText(stored) {
  const weights = stored.files.filter((f) => /\.onnx_data(_\d+)?$/.test(f.path));
  if (stored.weightsComplete) return `on this device, ${mib(stored.bytes)}`;
  if (stored.partial) return `part downloaded, ${mib(weights.reduce((sum, f) => sum + f.size, 0))}`;
  return 'not downloaded';
}

function renderEnv() {
  const env = state.env;
  const a = env.adapter;
  $('loader-name').textContent = MODEL_CARD.name;
  $('loader-sub').textContent = `${MODEL.dtype} weights, WebGPU only, no CPU fallback`;
  $('info-model').textContent = `${MODEL_CARD.name} ${MODEL.dtype}`;

  const rows = [
    check('WebGPU', a.webgpu ? 'yes' : 'no', !a.webgpu),
  ];
  if (a.info) {
    const name = [a.info.vendor, a.info.architecture, a.info.description].filter(Boolean).join(' ') || 'unnamed';
    rows.push(check('Adapter', name));
    // A software renderer is a CPU doing a GPU's work. Say so; never call it a GPU.
    if (a.software) rows.push(check('Kind', 'software, not a GPU', true));
    rows.push(check('shader-f16', a.f16 ? 'yes' : 'no', !a.f16));
    rows.push(check('maxBufferSize', mib(a.limits.maxBufferSize)));
    rows.push(check('Max binding', mib(a.limits.maxStorageBufferBindingSize)));
  }
  rows.push(check('File storage', env.opfs ? 'yes' : 'no', !env.opfs));
  rows.push(check('Weights', storedText(env.stored)));
  $('checks').replaceChildren(...rows);

  const problem = a.problem ?? (env.opfs ? null : 'This browser cannot read and write files from a worker (OPFS). BLIK needs that to keep the weights out of memory.');
  $('problem').hidden = !problem;
  $('problem-text').textContent = problem ?? '';
  loadButton.disabled = !!problem || state.loading || state.loaded;

  const fine = $('loader-fine');
  fine.replaceChildren();
  fine.append(
    env.stored.weightsComplete ? 'Nothing more to download. ' : 'Loading downloads about 725 MiB once, straight from Hugging Face. ',
    `${MODEL_CARD.name} is under the `,
    link(MODEL_CARD.license, MODEL_CARD.licenseUrl),
    `: ${MODEL_CARD.licenseNote.charAt(0).toLowerCase()}${MODEL_CARD.licenseNote.slice(1)}`,
  );
  setInfoState();
  updateScrollbar();
}

function setInfoState() {
  const a = state.env?.adapter;
  const backend = !a ? '' : a.ok ? (a.software ? 'WebGPU software' : 'WebGPU') : 'cannot run';
  $('info-state').textContent = backend;
  const word = state.gone
    ? 'GPU lost'
    : state.loading
      ? 'Loading'
      : !state.loaded
        ? 'Not loaded'
        : state.busy
          ? state.current?.writing
            ? 'Writing'
            : 'Reading'
          : 'Ready';
  $('bar-state').textContent = word;
}

if (interrupted) {
  const when = new Date(interrupted.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const phase = PHASES[interrupted.phase];
  const during = phase ? ` It was at: ${phase.charAt(0).toLowerCase()}${phase.slice(1)}.` : '';
  $('crash-text').textContent =
    interrupted.what === 'load'
      ? `The last load (${when}) ended in a tab reload, probably out of memory.${during}`
      : `The last reply (${when}) ended in a tab reload, probably out of memory.${during}`;
  $('crash-note').hidden = false;
}

function load() {
  if (state.loading || state.loaded) return;
  state.loading = true;
  loadButton.disabled = true;
  $('problem').hidden = true;
  $('progress').hidden = false;
  $('progress-fill').style.width = '0%';
  $('progress-text').textContent = PHASES.check;
  // Before anything heavy starts: if the tab dies now, the next visit says so.
  markRun('load', 'check');
  setInfoState();
  worker.postMessage({ type: 'load' });
}
loadButton.addEventListener('click', load);

function onDownload(m) {
  if (!m.total) return;
  $('progress-fill').style.width = `${((m.loaded / m.total) * 100).toFixed(2)}%`;
  if (m.cached) {
    $('progress-text').textContent = `Weights already on this device, ${mib(m.total)}`;
    return;
  }
  const speed = m.bytesPerSecond > 0 ? `, ${(m.bytesPerSecond / 1048576).toFixed(1)} MiB/s` : '';
  const resumed = m.resumedFrom > 0 ? ` (resumed at ${mib(m.resumedFrom)})` : '';
  $('progress-text').textContent = `${mib(m.loaded)} of ${mib(m.total)}${speed}${resumed}`;
}

function onReady(m) {
  clearRun();
  state.loading = false;
  state.loaded = true;
  state.env.stored = m.stored;
  $('loader').hidden = true;
  const t = m.timings;
  addNote(
    `Model loaded in ${seconds(t.totalMs)}: weights ${m.download.fromCache ? 'read from this device' : `downloaded in ${seconds(t.downloadMs)}`}, ` +
      `GPU ${seconds(t.sessionMs)}, shaders ${seconds(t.warmupMs)}. ${mib(m.gpu.live)} in GPU buffers.`,
  );
  if (!m.session.numLogitsToKeep) {
    addNote('This model file has no num_logits_to_keep input, so a long prompt computes logits for every token. Long prompts may run out of memory.', true);
  }
  const input = $('input');
  input.disabled = false;
  input.placeholder = 'Message';
  updateComposer();
  setContext(0);
  setInfoState();
}

// ---- Chat ----------------------------------------------------------------------------------

const input = $('input');
const sendButton = $('send-button');

function updateComposer() {
  if (state.busy) {
    setLabel(sendButton, 'Stop');
    sendButton.disabled = false;
  } else {
    setLabel(sendButton, 'Send');
    sendButton.disabled = !state.loaded || state.gone || input.value.trim() === '';
  }
}

function autosize() {
  input.style.height = '32px';
  input.style.height = `${Math.min(112, input.scrollHeight + 2)}px`;
}

function setContext(tokens) {
  state.context = tokens;
  const cap = state.contextCap;
  $('ctx-fill').style.width = `${Math.min(100, (tokens / cap) * 100).toFixed(2)}%`;
  $('ctx-meter').classList.toggle('full', tokens >= cap);
  $('ctx-meter').classList.toggle('at-zero', tokens === 0);
  $('ctx-text').textContent = `${count(tokens)}/${count(cap)}`;
}
function setSpeed(tokensPerSecond, ttftMs) {
  $('speed-text').textContent = `${tokensPerSecond == null ? NONE : tokensPerSecond.toFixed(1)} tok/s`;
  $('ttft-text').textContent = `first token ${ttftMs == null ? NONE : seconds(ttftMs)}`;
}
setContext(0);
setSpeed(null, null);

function send() {
  const text = input.value.trim();
  if (!text || state.busy || !state.loaded || state.gone) return;
  input.value = '';
  autosize();
  state.messages.push({ role: 'user', content: text });
  stick = true;
  addMessage('user', text);

  const node = el('div', 'msg assistant');
  const textNode = document.createTextNode('');
  node.append(textNode, el('span', 'caret'));
  messagesEl.append(node);
  state.current = { id: state.nextId++, node, textNode, text: '', writing: false, firstT: null, ttftMs: null, speed: null };
  state.busy = true;
  updateComposer();
  setSpeed(null, null);
  setInfoState();
  scrollToEnd();

  markRun('reply', 'prefill');
  const messages = state.system ? [{ role: 'system', content: state.system }, ...state.messages] : state.messages;
  worker.postMessage({
    type: 'generate',
    id: state.current.id,
    messages,
    params: { ...SAMPLING, contextCap: state.contextCap },
  });
}

function stop() {
  if (state.busy) worker.postMessage({ type: 'stop' });
}

function newChat() {
  if (state.busy) return;
  state.messages = [];
  for (const node of [...messagesEl.children]) if (node.id !== 'loader') node.remove();
  worker.postMessage({ type: 'reset' });
  setContext(0);
  setSpeed(null, null);
  updateScrollbar();
}

function finishReply(note) {
  const current = state.current;
  state.current = null;
  state.busy = false;
  clearRun();
  if (current) {
    current.node.querySelector('.caret')?.remove();
    if (current.text.trim() === '') current.node.remove();
    else state.messages.push({ role: 'assistant', content: current.text });
  }
  if (note) messagesEl.append(note);
  updateComposer();
  setInfoState();
  scrollToEnd();
}

function onPrefill(m) {
  if (!state.current || m.id !== state.current.id) return;
  setContext(m.read);
}

function onText(m) {
  const current = state.current;
  if (!current || m.id !== current.id) return;
  current.text += m.text;
  current.textNode.data = current.text;
  scrollToEnd();
}

function onStep(m) {
  const current = state.current;
  if (!current || m.id !== current.id) return;
  if (m.index === 0) {
    current.firstT = m.t;
    current.ttftMs = m.t;
    current.writing = true;
    markRun('reply', 'generate');
    setInfoState();
  } else if (m.t > current.firstT) {
    current.speed = (m.index / (m.t - current.firstT)) * 1000;
  }
  setContext(m.context);
  setSpeed(current.speed, current.ttftMs);
}

function onDone(m) {
  if (!state.current || m.id !== state.current.id) return;
  setContext(m.context);
  setSpeed(m.tokensPerSecond ?? null, m.ttftMs ?? null);
  const parts = [];
  if (m.generated > 0) {
    parts.push(`${count(m.generated)} token${m.generated === 1 ? '' : 's'}`);
    if (m.tokensPerSecond != null) parts.push(`${m.tokensPerSecond.toFixed(1)} tok/s`);
    if (m.ttftMs != null) parts.push(`first ${seconds(m.ttftMs)}`);
  }
  if (m.reason === 'stopped') parts.push('stopped');
  else if (m.reason === 'context') parts.push(`context full, ${count(m.context)}/${count(m.cap)}`);
  else if (m.reason === 'length') parts.push(`reply limit, ${count(SAMPLING.maxNewTokens)} tokens`);
  const meta = parts.length ? el('div', 'msg-meta', parts.join(', ')) : null;
  finishReply(meta);
  if (m.dropped > 0) {
    addNote(`The ${m.dropped} oldest message${m.dropped === 1 ? '' : 's'} no longer fit the ${count(m.cap)}-token context and ${m.dropped === 1 ? 'was' : 'were'} left out of this prompt.`);
  }
}

$('composer').addEventListener('submit', (event) => {
  event.preventDefault();
  if (state.busy) stop();
  else send();
});
input.addEventListener('input', () => {
  autosize();
  updateComposer();
});
input.addEventListener('keydown', (event) => {
  // With a hardware keyboard, Enter sends and Shift+Enter breaks the line.
  // On a touch keyboard, Enter breaks the line and the button sends.
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
  if (window.matchMedia('(pointer: coarse)').matches) return;
  event.preventDefault();
  send();
});

// ---- Deleting the downloaded files --------------------------------------------------------------

function confirmPurge() {
  showDialog({
    title: 'Delete downloaded files?',
    body: [
      `This removes the model files BLIK keeps on this device (${mib(state.env.stored.bytes)}). ${state.loaded ? 'The model stays loaded until you close or reload the page. ' : ''}Loading again will download them again.`,
    ],
    actions: [
      { label: 'Cancel' },
      { label: 'Delete', primary: true, run: () => worker.postMessage({ type: 'purge' }) },
    ],
  });
}

// ---- The worker ------------------------------------------------------------------------------------

function fatal(text) {
  state.loading = false;
  clearRun();
  $('problem').hidden = false;
  $('problem-text').textContent = text;
  $('progress').hidden = true;
  loadButton.disabled = true;
  setInfoState();
}

function onError(m) {
  if (m.where === 'load') {
    state.loading = false;
    clearRun();
    $('progress').hidden = true;
    $('problem').hidden = false;
    $('problem-text').textContent = `Loading failed: ${m.message}`;
    setLabel(loadButton, 'Try again');
    loadButton.disabled = !state.env?.adapter.ok;
    setInfoState();
  } else if (m.where === 'generate') {
    finishReply(null);
    addNote(`The reply failed: ${m.message}`, true);
  } else if (!state.env) {
    fatal(`The model runtime failed to start: ${m.message}`);
  } else {
    addNote(m.message, true);
  }
}

function handle(m) {
  switch (m.type) {
    case 'boot':
      worker.postMessage({ type: 'hello' });
      break;
    case 'env':
      state.env = m;
      renderEnv();
      break;
    case 'phase':
      if (state.loading && PHASES[m.phase]) {
        markRun('load', m.phase);
        if (m.phase !== 'download') $('progress-text').textContent = `${PHASES[m.phase]}...`;
      }
      break;
    case 'download':
      onDownload(m);
      break;
    case 'ready':
      onReady(m);
      break;
    case 'prefill':
      onPrefill(m);
      break;
    case 'text':
      onText(m);
      break;
    case 'step':
      onStep(m);
      break;
    case 'done':
      onDone(m);
      break;
    case 'purged':
      state.env.stored = m.stored;
      state.env.storage = m.storage;
      if (!state.loaded) renderEnv();
      addNote('The downloaded model files were deleted from this device.');
      break;
    case 'gpu-lost':
      state.gone = true;
      if (state.busy) finishReply(null);
      addNote(`The GPU device was lost${m.reason ? ` (${m.reason})` : ''}${m.message ? `: ${m.message}` : '.'} The model is no longer on the GPU. Reload the page to load it again.`, true);
      updateComposer();
      setInfoState();
      break;
    case 'error':
      onError(m);
      break;
    default:
      break;
  }
}

let worker;
try {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (event) => handle(event.data);
  worker.onerror = (event) => {
    event.preventDefault?.();
    fatal(`The model runtime failed to start${event.message ? `: ${event.message}` : '.'}`);
  };
} catch (error) {
  fatal(`This browser cannot start the model runtime: ${error.message}`);
}

$('loader-name').textContent = MODEL_CARD.name;
$('loader-sub').textContent = 'Checking this device...';
$('info-model').textContent = `${MODEL_CARD.name} ${MODEL.dtype}`;
setInfoState();
fitViewport();
document.fonts?.ready.then(updateScrollbar);
