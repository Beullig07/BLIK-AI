// The pinned, vendored runtime. Nothing is loaded from a CDN.
//
//   Transformers.js   4.3.2                          vendor/transformers-4.3.2/
//   ONNX Runtime Web  1.31.0-dev.20260914-8d85527a0  vendor/onnxruntime-web-.../
//
// The ONNX Runtime version is the one Transformers.js 4.3.2 depends on; its
// JavaScript is bundled inside transformers.min.js and only the WebAssembly
// files are separate. To upgrade, replace both folders together and update
// the two paths below.

export * from '../vendor/transformers-4.3.2/transformers.min.js';

export const ORT_VERSION = '1.31.0-dev.20260914-8d85527a0';
export const ORT_DIR = new URL('../vendor/onnxruntime-web-1.31.0-dev.20260914-8d85527a0/', import.meta.url).href;
