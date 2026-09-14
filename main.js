import initWasm from './pkg/edge_kmeans_benchmark.js'
import { setupWasmBenchmark, setupWasmMTBenchmark } from './lib/wasmBenchmark.js';
import { setupJsBenchmark } from './lib/jsBenchmark.js';
import { setupWebGPUBenchmark } from './lib/webGPUBenchmark.js';

const log = (msg) => { document.getElementById('statusLog').innerText = msg; };

initWasm().then(async () => {
  document.getElementById('statusLog').innerText = "Wasm Initialized.";
  document.getElementById('btnRunWasm').disabled = false;
  document.getElementById('btnRunWasmMT').disabled = false;
});

// Javascript benchmark setup
setupJsBenchmark(log);

// WebAssembly benchmark setup
setupWasmBenchmark(log);

// Multithreaded WebAssembly benchmark setup
setupWasmMTBenchmark(log);

// Web GPU benchmark setup
setupWebGPUBenchmark(log);
