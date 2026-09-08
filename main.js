import { generateSyntheticData, initializeCentroids } from './generator.js';

let dataset = null;
let centroids = null;
let currentN = 0, currentD = 0, currentK = 0;

const log = (msg) => { document.getElementById('statusLog').innerText = msg; };

// 1. Data Generation Handler
document.getElementById('btnGenData').addEventListener('click', () => {
  currentN = parseInt(document.getElementById('numPoints').value, 10);
  currentD = parseInt(document.getElementById('numDims').value, 10);
  currentK = parseInt(document.getElementById('numClusters').value, 10);

  log(`Generating ${currentN} points across ${currentD} dimensions...`);
  
  const t0 = performance.now();
  dataset = generateSyntheticData(currentN, currentD);
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