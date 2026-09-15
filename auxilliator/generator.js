export function initializeCentroidsRandom(D, K) {
  const centroids = new Float32Array(K * D);
  for (let i = 0; i < centroids.length; i++) {
    centroids[i] = Math.random();
  }
  return centroids;
}

export function generateBatch(batchSize, D) {
  const data = new Float32Array(batchSize * D);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.random();
  }
  return data;
}

export function calculateBatchSize(totalPoints, D, maxBytes = 1500 * 1024 * 1024) {
  const maxPointsByMemory = Math.floor(maxBytes / (D * 4));
  return Math.max(1, Math.min(totalPoints, maxPointsByMemory));
}