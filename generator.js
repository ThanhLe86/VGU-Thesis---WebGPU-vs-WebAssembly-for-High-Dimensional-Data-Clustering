/**
 * Generates a flat Float32Array representing N points in D dimensions.
 * Point i, dimension d is at index: (i * D) + d
 */
export function generateSyntheticData(N, D) {
  const totalElements = N * D;
  const data = new Float32Array(totalElements);
  
  for (let i = 0; i < totalElements; i++) {
    data[i] = Math.random();
  }
  return data;
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