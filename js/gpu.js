// WebGPU: what the adapter offers, and what the page's GPU buffers add up to.
// Used by the page (probe) and by the worker (probe, hooks).

/**
 * Asks the browser for a GPU adapter and reports what it is and what it can do.
 * Creates no device. `ok` is true only if WebGPU is there and, when the model
 * needs 16-bit floats, the adapter has shader-f16.
 */
export async function probeAdapter({ needsF16 }) {
  const out = {
    ok: false,
    problem: null, // why the model cannot run here, in words for the screen
    webgpu: false,
    f16: false,
    software: false, // a CPU renderer standing in for a GPU
    info: null,
    features: [],
    limits: null,
  };
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    out.problem = 'This browser has no WebGPU. BLIK runs the model on the GPU only; there is no CPU fallback.';
    return out;
  }
  out.webgpu = true;
  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (error) {
    out.problem = `WebGPU is present but asking for a GPU adapter failed: ${error.message}`;
    return out;
  }
  if (!adapter) {
    out.problem = 'WebGPU is present but the browser returned no GPU adapter.';
    return out;
  }
  const info = adapter.info ?? {};
  out.info = {
    vendor: info.vendor || '',
    architecture: info.architecture || '',
    device: info.device || '',
    description: info.description || '',
  };
  out.software = !!(info.isFallbackAdapter ?? adapter.isFallbackAdapter);
  out.features = [...adapter.features].sort();
  out.f16 = adapter.features.has('shader-f16');
  const l = adapter.limits;
  out.limits = {
    maxBufferSize: l.maxBufferSize,
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
  };
  if (needsF16 && !out.f16) {
    out.problem = 'This GPU adapter has no shader-f16 (16-bit floats in shaders), which this model file needs.';
    return out;
  }
  out.ok = true;
  return out;
}

const MIB = 1024 * 1024;

/**
 * Hooks GPUDevice.createBuffer and GPUBuffer.destroy, in the worker, before
 * ONNX Runtime creates its device.
 *
 * It counts the bytes of every live GPU buffer (the numbers in the stats
 * window), reports the device ONNX Runtime actually got, and optionally pools
 * destroyed buffers.
 *
 * The pool: ONNX Runtime keeps spare buffers per size bucket and destroys the
 * rest after each run. Once the KV cache tensors grow into a bucket with too
 * few spares, every token creates and destroys several multi-megabyte buffers,
 * and Safari gives destroyed GPU memory back slowly, so the tab's footprint
 * climbs until it is killed (microsoft/onnxruntime#33751). With the pool, a
 * buffer of 1 MiB or more that ONNX Runtime destroys is kept, and handed back
 * to the next createBuffer of the same size and usage.
 */
export function installGpuHooks({ pool, onDevice }) {
  const stats = {
    live: 0, // bytes in buffers that exist now (pooled ones included)
    peak: 0,
    created: 0, // createBuffer calls that made a new buffer
    destroyed: 0, // destroy calls that really destroyed one
    reused: 0, // createBuffer calls served from the pool
    pooled: 0, // buffers waiting in the pool now
    pooledBytes: 0,
    createdBytes: 0, // total bytes of every buffer ever created
    pool: !!pool,
  };
  if (typeof GPUDevice === 'undefined' || typeof GPUBuffer === 'undefined' || typeof GPUBufferUsage === 'undefined') return stats;

  const POOL_MIN = MIB;
  const POOL_PER_KEY = 12;
  const MAPPABLE = GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE;
  const records = new WeakMap(); // buffer -> {size, usage, state: 'live' | 'pooled' | 'dead'}
  const pools = new Map(); // "size:usage" -> buffers
  const seenDevices = new WeakSet();

  const create = GPUDevice.prototype.createBuffer;
  const destroy = GPUBuffer.prototype.destroy;

  GPUDevice.prototype.createBuffer = function (descriptor) {
    if (!seenDevices.has(this)) {
      seenDevices.add(this);
      onDevice?.(this);
    }
    const { size, usage, mappedAtCreation } = descriptor;
    if (pool && !mappedAtCreation && size >= POOL_MIN && !(usage & MAPPABLE)) {
      const list = pools.get(`${size}:${usage}`);
      const buffer = list?.pop();
      if (buffer) {
        records.get(buffer).state = 'live';
        stats.reused++;
        stats.pooled--;
        stats.pooledBytes -= size;
        return buffer;
      }
    }
    const buffer = create.call(this, descriptor);
    records.set(buffer, { size, usage, state: 'live' });
    stats.created++;
    stats.createdBytes += size;
    stats.live += size;
    if (stats.live > stats.peak) stats.peak = stats.live;
    return buffer;
  };

  GPUBuffer.prototype.destroy = function () {
    const record = records.get(this);
    if (!record) return destroy.call(this);
    if (record.state !== 'live') return undefined; // already given back
    if (pool && record.size >= POOL_MIN && !(record.usage & MAPPABLE)) {
      const key = `${record.size}:${record.usage}`;
      let list = pools.get(key);
      if (!list) pools.set(key, (list = []));
      if (list.length < POOL_PER_KEY) {
        // A mapped buffer cannot be handed out again as it is.
        if (this.mapState === undefined || this.mapState === 'unmapped') {
          list.push(this);
          record.state = 'pooled';
          stats.pooled++;
          stats.pooledBytes += record.size;
          return undefined;
        }
      }
    }
    record.state = 'dead';
    stats.destroyed++;
    stats.live -= record.size;
    return destroy.call(this);
  };

  /** Really destroys everything in the pool (when the model is unloaded). */
  stats.drain = () => {
    for (const list of pools.values()) {
      for (const buffer of list) {
        const record = records.get(buffer);
        record.state = 'dead';
        stats.destroyed++;
        stats.live -= record.size;
        destroy.call(buffer);
      }
    }
    pools.clear();
    stats.pooled = 0;
    stats.pooledBytes = 0;
  };
  return stats;
}
