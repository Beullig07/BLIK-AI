// Model files in the origin's private file system (OPFS). Runs in the worker.
//
// Why not the stock Transformers.js cache: on a cache miss it reads a whole
// file into memory and then gives the Cache API a second copy, so a 725 MiB
// weights file peaks near 1.5 GB of JavaScript buffers and an iPhone Safari
// tab (about 2 GB, GPU memory included) is killed as the download ends. Nothing
// gets cached, so every retry dies the same way (huggingface/transformers.js#1827).
//
// Here:
//  - ensureFile() streams a download into OPFS chunk by chunk, and resumes a
//    partial file with an HTTP Range request.
//  - OnDemandFile gives ONNX Runtime the weights as a Uint8Array whose
//    subarray() reads that byte range from the file. ONNX Runtime reads
//    external data one tensor at a time through byteLength and subarray()
//    only (checked in onnxruntime-web 1.31's asyncify and JSPI glue), so the
//    weights are never whole in memory.
//  - makeCache() is the `env.customCache` Transformers.js uses for the small
//    files (config, tokenizer, the .onnx graph).
//
// The OPFS reader technique follows daveycodez/tanstack-ai-browser (MIT).

const ROOT = 'blik-models';
const DONE = '.blik-done'; // beside a complete file: {url, size, etag, type}
const PART = '.blik-part'; // beside a partial file:  {url, total, etag}
const FLUSH_EVERY = 32 * 1024 * 1024;

/** Whether this worker can do synchronous OPFS reads and writes. */
export function opfsSupported() {
  return (
    typeof navigator !== 'undefined' &&
    !!navigator.storage?.getDirectory &&
    typeof FileSystemFileHandle !== 'undefined' &&
    'createSyncAccessHandle' in FileSystemFileHandle.prototype
  );
}

const rootDir = async (create = true) => (await navigator.storage.getDirectory()).getDirectoryHandle(ROOT, { create });

/** `https://host/a/b/c.bin` -> directory segments ["host", "a", "b"] and file name "c.bin". */
function locate(url) {
  const u = new URL(url);
  const parts = [u.host, ...u.pathname.split('/').filter(Boolean)].map((s) => decodeURIComponent(s).replace(/[^\w.@+-]/g, '_'));
  return { dirs: parts.slice(0, -1), name: parts[parts.length - 1] };
}

async function dirFor(dirs, create) {
  let dir = await rootDir(create);
  for (const d of dirs) dir = await dir.getDirectoryHandle(d, { create });
  return dir;
}

async function readJson(dir, name) {
  try {
    return JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text());
  } catch {
    return undefined;
  }
}

async function writeJson(dir, name, value) {
  const handle = await dir.getFileHandle(name, { create: true });
  const access = await handle.createSyncAccessHandle();
  try {
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    access.truncate(0);
    access.write(bytes, { at: 0 });
    access.flush();
  } finally {
    access.close();
  }
}

const remove = (dir, name) => dir.removeEntry(name).catch(() => undefined);

/** A complete file's handle and record, or undefined. */
async function complete(url) {
  const { dirs, name } = locate(url);
  let dir;
  try {
    dir = await dirFor(dirs, false);
  } catch {
    return undefined;
  }
  const done = await readJson(dir, name + DONE);
  if (!done) return undefined;
  try {
    const handle = await dir.getFileHandle(name);
    const file = await handle.getFile();
    return file.size === done.size ? { handle, file, done } : undefined;
  } catch {
    return undefined;
  }
}

/** Whether `url` is fully downloaded, and how much of it is there if not. */
export async function fileState(url) {
  const kept = await complete(url);
  if (kept) return { complete: true, size: kept.done.size, total: kept.done.size };
  const { dirs, name } = locate(url);
  try {
    const dir = await dirFor(dirs, false);
    const part = await readJson(dir, name + PART);
    if (!part) return { complete: false, size: 0, total: 0 };
    const file = await (await dir.getFileHandle(name)).getFile();
    return { complete: false, size: Math.min(file.size, part.total || file.size), total: part.total || 0 };
  } catch {
    return { complete: false, size: 0, total: 0 };
  }
}

function writeAll(access, chunk, at) {
  let done = 0;
  while (done < chunk.byteLength) {
    const n = access.write(done ? chunk.subarray(done) : chunk, { at: at + done });
    if (!(n > 0)) throw new Error('Writing to the browser\'s file storage failed (out of space?).');
    done += n;
  }
}

/**
 * Makes sure `url` is fully downloaded into OPFS, resuming a partial download.
 * The body is written as it arrives; it is never whole in memory.
 *
 * @param {string} url
 * @param {object} [o]
 * @param {(p:{loaded:number,total:number,resumedFrom:number}) => void} [o.onProgress]
 * @param {AbortSignal} [o.signal]
 * @returns {Promise<{size:number, cached:boolean, resumedFrom:number, handle:FileSystemFileHandle}>}
 */
export async function ensureFile(url, { onProgress, signal } = {}) {
  const kept = await complete(url);
  if (kept) return { size: kept.done.size, cached: true, resumedFrom: 0, handle: kept.handle };

  const { dirs, name } = locate(url);
  const dir = await dirFor(dirs, true);
  const handle = await dir.getFileHandle(name, { create: true });
  const part = await readJson(dir, name + PART);
  const access = await handle.createSyncAccessHandle();
  try {
    let have = part ? access.getSize() : 0;
    if (part?.total && have > part.total) have = 0;

    // Resume: ask for the rest, and take it only if the server answers with
    // exactly that range of the same file.
    let response;
    let total = 0;
    if (have > 0) {
      try {
        const ranged = await fetch(url, { headers: { Range: `bytes=${have}-` }, signal });
        const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(ranged.headers.get('content-range') ?? '');
        const etag = ranged.headers.get('etag');
        const same = !part.etag || !etag || part.etag === etag;
        if (ranged.status === 206 && range && Number(range[1]) === have && same && (!part.total || Number(range[3]) === part.total)) {
          response = ranged;
          total = Number(range[3]);
        } else {
          await ranged.body?.cancel().catch(() => undefined);
        }
      } catch (error) {
        if (signal?.aborted) throw error;
      }
      if (!response) have = 0;
    }
    const resumedFrom = have;

    if (!response) {
      access.truncate(0);
      response = await fetch(url, { signal });
      if (response.status !== 200) {
        await response.body?.cancel().catch(() => undefined);
        throw new Error(`Download failed: HTTP ${response.status} for ${url}`);
      }
      total = Number(response.headers.get('content-length')) || 0;
    }
    if (!response.body) throw new Error(`Download failed: no response body for ${url}`);

    const etag = response.headers.get('etag') ?? part?.etag ?? null;
    const type = response.headers.get('content-type') ?? 'application/octet-stream';
    // The partial record goes down before the first byte, so a tab killed
    // mid-download leaves a file the next load can resume.
    await writeJson(dir, name + PART, { url, total, etag });
    let written = have;
    let sinceFlush = 0;
    const reader = response.body.getReader();
    onProgress?.({ loaded: written, total, resumedFrom });
    for (let read = await reader.read(); !read.done; read = await reader.read()) {
      writeAll(access, read.value, written);
      written += read.value.byteLength;
      sinceFlush += read.value.byteLength;
      if (sinceFlush >= FLUSH_EVERY) {
        access.flush();
        sinceFlush = 0;
      }
      onProgress?.({ loaded: written, total, resumedFrom });
    }
    access.flush();
    if (total && written !== total) {
      throw new Error(`The download stopped at ${written} of ${total} bytes. Load again to resume it.`);
    }
    await writeJson(dir, name + DONE, { url, size: written, etag, type });
    await remove(dir, name + PART);
    return { size: written, cached: false, resumedFrom, handle };
  } finally {
    // Keep what did arrive, so a failed download resumes instead of restarting.
    try {
      access.flush();
    } catch {
      // nothing more to do
    }
    access.close();
  }
}

/**
 * `env.customCache` for Transformers.js: `match` and `put`, kept in OPFS.
 * Only keys under `hub` are handled; anything else is a miss.
 */
export function makeCache(hub) {
  return {
    async match(key) {
      if (typeof key !== 'string' || !key.startsWith(hub)) return undefined;
      const kept = await complete(key);
      if (!kept) return undefined;
      return new Response(kept.file, {
        headers: { 'content-length': String(kept.done.size), 'content-type': kept.done.type ?? 'application/octet-stream' },
      });
    },
    async put(key, response) {
      if (typeof key !== 'string' || !key.startsWith(hub) || !response.body) return;
      const { dirs, name } = locate(key);
      const dir = await dirFor(dirs, true);
      const handle = await dir.getFileHandle(name, { create: true });
      const access = await handle.createSyncAccessHandle();
      let written = 0;
      try {
        access.truncate(0);
        const reader = response.body.getReader();
        for (let read = await reader.read(); !read.done; read = await reader.read()) {
          writeAll(access, read.value, written);
          written += read.value.byteLength;
        }
        access.flush();
      } finally {
        access.close();
      }
      await writeJson(dir, name + DONE, {
        url: key,
        size: written,
        etag: response.headers.get('etag'),
        type: response.headers.get('content-type') ?? 'application/octet-stream',
      });
    },
  };
}

/** Every stored file: path under the cache folder, size, and whether it is complete. */
export async function listFiles() {
  const out = [];
  async function walk(dir, prefix) {
    const names = new Map();
    for await (const [name, handle] of dir.entries()) names.set(name, handle);
    for (const [name, handle] of names) {
      if (handle.kind === 'directory') await walk(handle, `${prefix}${name}/`);
      else if (!name.endsWith(DONE) && !name.endsWith(PART)) {
        out.push({ path: prefix + name, size: (await handle.getFile()).size, complete: names.has(name + DONE) });
      }
    }
  }
  try {
    await walk(await rootDir(false), '');
  } catch {
    // nothing stored yet
  }
  return out;
}

/** Deletes every stored model file. */
export async function purge() {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(ROOT, { recursive: true }).catch(() => undefined);
}

/** The origin's storage use and quota, as the browser estimates them. */
export async function storageEstimate() {
  try {
    const { usage, quota } = await navigator.storage.estimate();
    return { usage, quota };
  } catch {
    return undefined;
  }
}

// One buffer every OnDemandFile reads into, grown to the largest tensor read.
// ONNX Runtime copies each slice before asking for the next, so sharing is
// safe. It is detached when the last file closes: Safari does not reliably
// garbage-collect a large buffer in a worker.
let scratch = new ArrayBuffer(0);
let openFiles = 0;

const refuse = (what) => () => {
  throw new Error(`On-demand weights were read through ${what}, not subarray().`);
};

/**
 * A weights file in OPFS, passed to ONNX Runtime as external data. It looks
 * like a Uint8Array of the file's length, but holds no bytes: subarray(a, b)
 * reads that range from the file. Any other way of reading it throws, so a
 * runtime that reads external data differently fails loudly instead of
 * silently reading zeros.
 */
export class OnDemandFile extends Uint8Array {
  #access;
  #size = 0;
  /** Counters, for the stats window. */
  reads = 0;
  bytesRead = 0;
  largestRead = 0;

  constructor(path) {
    super(0);
    this.path = path;
  }

  async open(handle) {
    this.#access = await handle.createSyncAccessHandle();
    this.#size = this.#access.getSize();
    openFiles++;
  }

  get byteLength() {
    return this.#size;
  }

  get length() {
    return this.#size;
  }

  subarray(begin = 0, end = this.#size) {
    if (!this.#access) throw new Error(`${this.path} is not open.`);
    if (begin < 0 || end < begin || end > this.#size) {
      throw new RangeError(`${this.path}: bytes ${begin} to ${end} of ${this.#size}`);
    }
    const length = end - begin;
    if (scratch.byteLength < length) {
      scratch.transfer?.(0);
      scratch = new ArrayBuffer(length);
    }
    const bytes = new Uint8Array(scratch, 0, length);
    if (length > 0 && this.#access.read(bytes, { at: begin }) !== length) {
      throw new Error(`${this.path}: short read at byte ${begin}`);
    }
    this.reads++;
    this.bytesRead += length;
    if (length > this.largestRead) this.largestRead = length;
    return bytes;
  }

  close() {
    if (!this.#access) return;
    this.#access.close();
    this.#access = undefined;
    if (--openFiles > 0) return;
    scratch.transfer?.(0);
    scratch = new ArrayBuffer(0);
  }
}
for (const name of ['buffer', 'byteOffset']) {
  Object.defineProperty(OnDemandFile.prototype, name, { get: refuse(name) });
}
for (const name of ['slice', 'set', 'at', 'fill', 'copyWithin', 'values', 'keys', 'entries', 'forEach', 'map', 'filter', 'reduce', 'join', 'indexOf', 'includes']) {
  Object.defineProperty(OnDemandFile.prototype, name, { value: refuse(name) });
}
Object.defineProperty(OnDemandFile.prototype, Symbol.iterator, { value: refuse('its iterator') });
