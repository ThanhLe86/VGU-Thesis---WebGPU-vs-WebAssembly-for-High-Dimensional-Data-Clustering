import { initializeCentroidsRandom, generateBatch, calculateBatchSize } from '../auxilliator/generator.js'

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

export function setupJsBenchmark(log) {
    const btnRunJS = document.getElementById('btnRunJS');
    if (!btnRunJS) return;

    btnRunJS.addEventListener('click', async () => {
        const totalPoints = parseInt(document.getElementById('numPoints').value, 10);
        const D = parseInt(document.getElementById('numDims').value, 10);
        const K = parseInt(document.getElementById('numClusters').value, 10);
        const batchSize = calculateBatchSize(totalPoints, D)
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
            log(`Wasm Processed ${iter * batchSize} / ${totalPoints} points...`);
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
    }
)}