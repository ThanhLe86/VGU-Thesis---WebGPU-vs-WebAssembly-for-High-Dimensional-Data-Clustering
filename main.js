import initWasm, {run_wasm_assignment} from './pkg/edge_kmeans_benchmark.js'
import { initializeCentroidsRandom, generateBatch, generateSyntheticData } from './generator.js';

initWasm().then(() => {
  document.getElementById('statusLog').innerText = "Wasm Initialized.";
  document.getElementById('btnRunWasm').disabled = false;
  document.getElementById('btnRunWasmMT').disabled = false;
});

const log = (msg) => { document.getElementById('statusLog').innerText = msg; };

function processSliceOnWorker(worker, payload) {
  return new Promise(resolve => {
    worker.onmessage = (e) => resolve(e.data);
    worker.postMessage(payload);
  });
}

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

function updateCentroids(batch, assignments, centroids, batchSize, D, clusterCounts) {
  for (let i = 0; i < batchSize; i++) {
    const cluster = assignments[i];
    clusterCounts[cluster]++;
    
    const learningRate = 1.0 / clusterCounts[cluster];
    const pointOffset = i * D;
    const centroidOffset = cluster * D;

    for (let d = 0; d < D; d++) {
      const pointVal = batch[pointOffset + d];
      centroids[centroidOffset + d] += learningRate * (pointVal - centroids[centroidOffset + d]);
    }
  }
}

document.getElementById('btnRunJS').addEventListener('click', async () => {
  const totalPoints = parseInt(document.getElementById('numPoints').value, 10);
  const D = parseInt(document.getElementById('numDims').value, 10);
  const K = parseInt(document.getElementById('numClusters').value, 10);
  const batchSize = 10000;
  const iterations = Math.ceil(totalPoints / batchSize);
  
  log(`Starting Pure JS Mini-Batch streaming for ${totalPoints} points...`);
  
  let streamingCentroids = initializeCentroidsRandom(D, K);
  const clusterCounts = new Int32Array(K);
  let totalComputeMs = 0;

  for (let iter = 0; iter < iterations; iter++) {
    const batch = generateBatch(batchSize, D);
    
    const t0 = performance.now();
    const assignments = runJSAssignment(batch, streamingCentroids, batchSize, D, K);
    updateCentroids(batch, assignments, streamingCentroids, batchSize, D, clusterCounts);
    totalComputeMs += performance.now() - t0;
    
    if (iter % 100 === 0 && iter > 0) {
      log(`JS Processed ${iter * batchSize} / ${totalPoints} points...`);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  const duration = totalComputeMs.toFixed(2);
  log(`JS Mini-Batch streaming completed in ${duration} ms.`);

  const row = `
    <tr>
      <td>Pure JavaScript (Mini-Batch)</td>
      <td>${totalPoints}</td>
      <td>${batchSize}</td>
      <td>${D}</td>
      <td>${K}</td>
      <td>${duration}</td>
    </tr>
  `;
  document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
});

document.getElementById('btnRunWasm').addEventListener('click', async () => {
  const totalPoints = parseInt(document.getElementById('numPoints').value, 10);
  const D = parseInt(document.getElementById('numDims').value, 10);
  const K = parseInt(document.getElementById('numClusters').value, 10);
  const batchSize = 10000;
  const iterations = Math.ceil(totalPoints / batchSize);
  
  log(`Starting Wasm Mini-Batch streaming for ${totalPoints} points...`);
  
  let streamingCentroids = initializeCentroidsRandom(D, K);
  const clusterCounts = new Int32Array(K);
  let totalComputeMs = 0;

  for (let iter = 0; iter < iterations; iter++) {
    const batch = generateBatch(batchSize, D);
    
    const t0 = performance.now();
    const assignments = run_wasm_assignment(batch, streamingCentroids, batchSize, D, K);
    updateCentroids(batch, assignments, streamingCentroids, batchSize, D, clusterCounts);
    totalComputeMs += performance.now() - t0;
    
    if (iter % 100 === 0 && iter > 0) {
      log(`Wasm Processed ${iter * batchSize} / ${totalPoints} points...`);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  const duration = totalComputeMs.toFixed(2);
  log(`Wasm Mini-Batch streaming completed in ${duration} ms.`);

  const row = `
    <tr>
      <td>WebAssembly (Mini-Batch)</td>
      <td>${totalPoints}</td>
      <td>${batchSize}</td>
      <td>${D}</td>
      <td>${K}</td>
      <td>${duration}</td>
    </tr>
  `;
  document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
});

document.getElementById('btnRunWasmMT').addEventListener('click', async () => {
  const totalPoints = parseInt(document.getElementById('numPoints').value, 10);
  const D = parseInt(document.getElementById('numDims').value, 10);
  const K = parseInt(document.getElementById('numClusters').value, 10);
  
  // MT needs larger batches to overcome thread communication overhead
  const batchSize = 100000; 
  const iterations = Math.ceil(totalPoints / batchSize);
  
  log(`Starting Wasm Multi-Threaded streaming for ${totalPoints} points...`);
  
  // 1. Initialize Persistent Worker Pool
  const numCores = navigator.hardwareConcurrency || 4;
  const workers = Array.from({ length: numCores }, () => new Worker('./auxilliator/worker_2.js', { type: 'module' }));
  
  let streamingCentroids = initializeCentroidsRandom(D, K);
  const clusterCounts = new Int32Array(K);
  let totalComputeMs = 0;

  // 2. Stream Macro-Batches
  for (let iter = 0; iter < iterations; iter++) {
    const { data, sab } = generateSyntheticData(batchSize, D);
    const pointsPerWorker = Math.ceil(batchSize / numCores);
    const promises = [];
    
    const t0 = performance.now();

    for (let i = 0; i < numCores; i++) {
      const startIdx = i * pointsPerWorker;
      if (startIdx >= batchSize) break;
      const sliceN = Math.min(pointsPerWorker, batchSize - startIdx);
      
      promises.push(processSliceOnWorker(workers[i], {
        sab: sab,
        centroids: streamingCentroids,
        sliceN: sliceN,
        d: D,
        k: K,
        startIdx: startIdx
      }));
    }

    const results = await Promise.all(promises);
    
    const batchAssignments = new Int32Array(batchSize);
    for (const res of results) {
      batchAssignments.set(res.assignments, res.startIdx);
    }
    
    updateCentroids(data, batchAssignments, streamingCentroids, batchSize, D, clusterCounts);
    totalComputeMs += performance.now() - t0;

    if (iter % 10 === 0 && iter > 0) {
      log(`MT Wasm Processed ${iter * batchSize} / ${totalPoints} points...`);
    }
  }

  // 3. Cleanup threads to free RAM
  workers.forEach(w => w.terminate());

  const duration = totalComputeMs.toFixed(2);
  log(`Multi-Threaded Wasm completed in ${duration} ms.`);

  const row = `
    <tr>
      <td>Wasm (Multi-Thread Mini-Batch)</td>
      <td>${totalPoints}</td>
      <td>${batchSize}</td>
      <td>${D}</td>
      <td>${K}</td>
      <td>${duration}</td>
    </tr>
  `;
  document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
});