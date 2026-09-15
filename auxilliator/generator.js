/**
 * Generates a flat Float32Array representing N points in D dimensions.
 * Point i, dimension d is at index: (i * D) + d
 */
export function generateSyntheticData(N, D) {
  const totalElements = N * D;
  const buffer = new SharedArrayBuffer(totalElements * 4); 
  const data = new Float32Array(buffer);
  
  for (let i = 0; i < totalElements; i++) {
    data[i] = Math.random();
  }
  return { data, sab: buffer };
}

export function initializeCentroids(data, N, D, K) {
  const centroids = new Float32Array(K * D);
  const usedIndices = new Set();

  for (let k = 0; k < K; k++) {
    let randIdx;
    do {
      randIdx = Math.floor(Math.random() * N);
    } while (usedIndices.has(randIdx));
    usedIndices.add(randIdx);

    const sourceOffset = randIdx * D;
    const destOffset = k * D;
    for (let d = 0; d < D; d++) {
      centroids[destOffset + d] = data[sourceOffset + d];
    }
  }
  return centroids;
}

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