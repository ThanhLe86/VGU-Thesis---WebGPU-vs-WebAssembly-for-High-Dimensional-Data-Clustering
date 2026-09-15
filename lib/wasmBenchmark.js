import { initializeCentroidsRandom, generateBatch, calculateBatchSize } from "../auxilliator/generator.js";
import { CentroidTracker, run_wasm_assignment } from "../pkg/edge_kmeans_benchmark.js";

export function setupWasmBenchmark(log) {
    const btnRunWasm = document.getElementById('btnRunWasm');
    if (!btnRunWasm) return;

    btnRunWasm.addEventListener('click', async () => {
        const totalPoints = parseInt(document.getElementById('numPoints').value, 10);
        const D = parseInt(document.getElementById('numDims').value, 10);
        const K = parseInt(document.getElementById('numClusters').value, 10);
        const batchSize = calculateBatchSize(totalPoints, D)
        const iterations = Math.ceil(totalPoints / batchSize);

        log(`Starting Wasm Mini-Batch streaming for ${totalPoints} points...`);

        let streamingCentroids = initializeCentroidsRandom(D, K);
        const tracker = new CentroidTracker(streamingCentroids, D, K);
        let totalComputeMs = 0;

        for (let iter = 0; iter < iterations; iter++) {
            const batch = generateBatch(batchSize, D);

            const t0 = performance.now();
            const assignments = run_wasm_assignment(batch, streamingCentroids, batchSize, D, K);
            
            // Update centroids directly within WebAssembly
            tracker.update(batch, assignments, batchSize);
            streamingCentroids = tracker.get_centroids();
            totalComputeMs += performance.now() - t0;

            if (iter % 100 === 0 && iter > 0) {
                log(`Wasm Processed ${iter * batchSize} / ${totalPoints} points...`);
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        // Free the tracker pointer from Wasm linear memory
        tracker.free();

        const duration = totalComputeMs.toFixed(2);
        log(`Wasm Mini-Batch streaming completed in ${duration} ms.`);

        const row = `
        <tr>
            <td>WebAssembly</td>
            <td>${totalPoints}</td>
            <td>${D}</td>
            <td>${K}</td>
            <td>${duration}</td>
        </tr>
        `;
        document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
    })
}

function processSliceOnWorker(worker, payload) {
  return new Promise(resolve => {
    worker.onmessage = (e) => resolve(e.data);
    worker.postMessage(payload);
  });
}

export function setupWasmMTBenchmark(log) {
    const btnRunWasmMT = document.getElementById('btnRunWasmMT');
    if (!btnRunWasmMT) return;

    btnRunWasmMT.addEventListener('click', async () => {
        const totalPoints = parseInt(document.getElementById('numPoints').value, 10);
        const D = parseInt(document.getElementById('numDims').value, 10);
        const K = parseInt(document.getElementById('numClusters').value, 10);

        const batchSize = calculateBatchSize(totalPoints, D);
        const iterations = Math.ceil(totalPoints / batchSize);

        log(`Starting Wasm Multi-Threaded streaming for ${totalPoints} points...`);

        const numCores = navigator.hardwareConcurrency || 4;
        const workers = Array.from(
        { length: numCores },
        () => new Worker('../auxilliator/worker_2.js', { type: 'module' })
        );

        let streamingCentroids = initializeCentroidsRandom(D, K);
        const tracker = new CentroidTracker(streamingCentroids, D, K);

        let totalComputeMs = 0;
        const totalElements = batchSize * D;
        const sharedBuffer = new SharedArrayBuffer(totalElements * 4);
        const sharedDataView = new Float32Array(sharedBuffer);

        for (let iter = 0; iter < iterations; iter++) {
        for (let i = 0; i < totalElements; i++) {
            sharedDataView[i] = Math.random();
        }

        const pointsPerWorker = Math.ceil(batchSize / numCores);
        const promises = [];

        const t0 = performance.now();

        for (let i = 0; i < numCores; i++) {
            const startIdx = i * pointsPerWorker;
            if (startIdx >= batchSize) break;
            const sliceN = Math.min(pointsPerWorker, batchSize - startIdx);

            promises.push(
            processSliceOnWorker(workers[i], {
                sab: sharedBuffer,
                centroids: streamingCentroids,
                sliceN: sliceN,
                d: D,
                k: K,
                startIdx: startIdx
            })
            );
        }

        const results = await Promise.all(promises);

        const batchAssignments = new Int32Array(batchSize);
        for (const res of results) {
            batchAssignments.set(res.assignments, res.startIdx);
        }

        tracker.update(sharedDataView, batchAssignments, batchSize);
        streamingCentroids = tracker.get_centroids();

        totalComputeMs += performance.now() - t0;

        if (iter % 10 === 0 && iter > 0) {
            log(`Wasm Processed ${iter * batchSize} / ${totalPoints} points...`);
        }
        }

        workers.forEach(w => w.terminate());
        tracker.free();

        const duration = totalComputeMs.toFixed(2);
        log(`Multi-Threaded Wasm completed in ${duration} ms.`);

        const row = `
        <tr>
        <td>Wasm (Multi-Thread Mini-Batch)</td>
        <td>${totalPoints}</td>
        <td>${D}</td>
        <td>${K}</td>
        <td>${duration}</td>
        </tr>
        `;
        document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
    });
}