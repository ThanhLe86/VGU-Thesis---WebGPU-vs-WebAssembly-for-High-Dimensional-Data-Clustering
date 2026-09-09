import { generateSyntheticData, initializeCentroids } from './generator.js';
import initWasm, {run_wasm_assignment} from './pkg/edge_kmeans_benchmark.js'

let dataset = null;
let datasetSab = null;
let centroids = null;
let currentN = 0, currentD = 0, currentK = 0;

initWasm().then(() => {
  document.getElementById('statusLog').innerText = "Wasm Initialized.";
});

const log = (msg) => { document.getElementById('statusLog').innerText = msg; };

// 1. Data Generation Handler
document.getElementById('btnGenData').addEventListener('click', () => {
  currentN = parseInt(document.getElementById('numPoints').value, 10);
  currentD = parseInt(document.getElementById('numDims').value, 10);
  currentK = parseInt(document.getElementById('numClusters').value, 10);

  log(`Generating ${currentN} points across ${currentD} dimensions...`);
  
  const t0 = performance.now();
  const generated = generateSyntheticData(currentN, currentD);
  dataset = generated.data;
  datasetSab = generated.sab;
  centroids = initializeCentroids(dataset, currentN, currentD, currentK);
  const t1 = performance.now();

  log(`Data generated in ${(t1 - t0).toFixed(2)} ms. Ready to benchmark.`);
  document.getElementById('btnRunJS').disabled = false;
});

// 2. Pure JavaScript K-Means (Euclidean Distance Assignment)
function runJSAssignment(data, centroids, N, D, K) {
  const assignments = new Int32Array(N);

  for (let i = 0; i < N; i++) {
    const pointOffset = i * D;
    let minDistanceSq = Infinity;
    let bestCluster = -1;

    for (let k = 0; k < K; k++) {
      const centroidOffset = k * D;
      let distSq = 0.0;

      for (let d = 0; d < D; d++) {
        const diff = data[pointOffset + d] - centroids[centroidOffset + d];
        distSq += diff * diff;
      }

      if (distSq < minDistanceSq) {
        minDistanceSq = distSq;
        bestCluster = k;
      }
    }
    assignments[i] = bestCluster;
  }
  return assignments;
}

document.getElementById('btnRunJS').addEventListener('click', () => {
  log('Running JavaScript assignment baseline...');
  runJSAssignment(dataset, centroids, currentN, currentD, currentK);

  const t0 = performance.now();
  const assignments = runJSAssignment(dataset, centroids, currentN, currentD, currentK);
  const t1 = performance.now();
  const duration = (t1 - t0).toFixed(2);

  log(`JS execution completed in ${duration} ms.`);

  const row = `
    <tr>
      <td>Pure JavaScript</td>
      <td>${currentN}</td>
      <td>${currentD}</td>
      <td>${currentK}</td>
      <td>${duration}</td>
    </tr>
  `;
  document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
});

// Enable the Wasm button when data is generated
document.getElementById('btnGenData').addEventListener('click', () => {
  document.getElementById('btnRunWasm').disabled = false;
});

// Wasm Benchmark Runner
document.getElementById('btnRunWasm').addEventListener('click', async () => {
  log('Running WebAssembly baseline...');
  const t0 = performance.now();
  
  // Detect CPU cores (usually 8, 12, or 16)
  const numCores = navigator.hardwareConcurrency || 4; 
  const pointsPerWorker = Math.ceil(currentN / numCores);
  
  let completedWorkers = 0;
  const finalAssignments = new Int32Array(currentN);
  
  for (let i = 0; i < numCores; i++) {
    const startIdx = i * pointsPerWorker;
    if (startIdx >= currentN) break; // Catch edge cases
    
    const sliceN = Math.min(pointsPerWorker, currentN - startIdx);
    
    const worker = new Worker('./auxilliator/worker.js', { type: 'module' });
    
    worker.onmessage = (e) => {
      // Stitch the results back together
      finalAssignments.set(e.data.assignments, e.data.startIdx);
      completedWorkers++;
      worker.terminate();
      
      if (completedWorkers === numCores) {
        const t1 = performance.now();
        const duration = (t1 - t0).toFixed(2);
        log(`Wasm (${numCores} Threads) completed in ${duration} ms.`);
        
        const row = `<tr>
          <td>Wasm (${numCores} Threads)</td>
          <td>${currentN}</td><td>${currentD}</td><td>${currentK}</td>
          <td>${duration}</td>
        </tr>`;
        document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
      }
    };
    
    // Dispatch the job
    worker.postMessage({
      sab: datasetSab,
      centroids: centroids,
      sliceN: sliceN,
      d: currentD,
      k: currentK,
      startIdx: startIdx
    });
  }
});

document.getElementById('btnClearData').addEventListener('click', () => {
  dataset = null;
  datasetSab = null;
  centroids = null;
  document.getElementById('btnRunJS').disabled = true;
  document.getElementById('btnRunWasm').disabled = true;
  if (typeof window.gc === 'function') {
    window.gc();
    log('Memory references cleared and Garbage Collection forced.');
  } else {
    log('References cleared. (Launch browser with --js-flags="--expose-gc" to force hard GC).');
  }
});