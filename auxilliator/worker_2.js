import initWasm, { run_wasm_assignment } from '../pkg/edge_kmeans_benchmark.js';

let wasmReady = false;
const initPromise = initWasm().then(() => { wasmReady = true; });

self.onmessage = async (e) => {
  await initPromise;
  
  const { sab, centroids, sliceN, d, k, startIdx } = e.data;
  
  const byteOffset = startIdx * d * 4;
  const dataSlice = new Float32Array(sab, byteOffset, sliceN * d);
  
  const assignments = run_wasm_assignment(dataSlice, centroids, sliceN, d, k);
  
  self.postMessage({ startIdx, assignments });
};