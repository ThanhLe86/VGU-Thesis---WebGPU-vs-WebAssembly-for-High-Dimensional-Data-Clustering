import initWebGPU from '../auxilliator/wgslInitializer.js';
import { CentroidTracker } from '../pkg/edge_kmeans_benchmark.js';
import { initializeCentroidsRandom, generateBatch } from '../auxilliator/generator.js';

let gpuDevice = null;
let gpuPipeline = null;

export function setupWebGPUBenchmark(log) {
  const btnRunWebGPU = document.getElementById('btnRunWebGPU');
  if (!btnRunWebGPU) return;

  btnRunWebGPU.addEventListener('click', async () => {
    const gpu = await initWebGPU(log);
    if (!gpu) return;
    gpuDevice = gpu.device;
    gpuPipeline = gpu.pipeline;

    const totalPoints = parseInt(document.getElementById('numPoints').value, 10);
    const D = parseInt(document.getElementById('numDims').value, 10);
    const K = parseInt(document.getElementById('numClusters').value, 10);

    const batchSize = 1000000;
    const iterations = Math.ceil(totalPoints / batchSize);

    log(`Starting WebGPU streaming for ${totalPoints} points...`);

    let streamingCentroids = initializeCentroidsRandom(D, K);
    const tracker = new CentroidTracker(streamingCentroids, D, K);
    let totalComputeMs = 0;

    const uniformBuffer = gpuDevice.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([D, K]));

    const dataBuffer = gpuDevice.createBuffer({
      size: batchSize * D * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const centroidBuffer = gpuDevice.createBuffer({
      size: K * D * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const assignmentBuffer = gpuDevice.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const readBuffer = gpuDevice.createBuffer({
      size: batchSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = gpuDevice.createBindGroup({
      layout: gpuPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: dataBuffer } },
        { binding: 2, resource: { buffer: centroidBuffer } },
        { binding: 3, resource: { buffer: assignmentBuffer } },
      ],
    });

    for (let iter = 0; iter < iterations; iter++) {
      const batch = generateBatch(batchSize, D);

      const t0 = performance.now();

      gpuDevice.queue.writeBuffer(dataBuffer, 0, batch);
      gpuDevice.queue.writeBuffer(centroidBuffer, 0, streamingCentroids);

      const encoder = gpuDevice.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(gpuPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(batchSize / 64));
      pass.end();

      encoder.copyBufferToBuffer(assignmentBuffer, 0, readBuffer, 0, batchSize * 4);
      gpuDevice.queue.submit([encoder.finish()]);

      await readBuffer.mapAsync(GPUMapMode.READ);
      const assignments = new Int32Array(readBuffer.getMappedRange());

      tracker.update(batch, assignments, batchSize);
      readBuffer.unmap();
      streamingCentroids = tracker.get_centroids();
      totalComputeMs += performance.now() - t0;

      if (iter % 10 === 0 && iter > 0) {
        log(`Wasm Processed ${iter * batchSize} / ${totalPoints} points...`);
      }
    }

    const duration = totalComputeMs.toFixed(2);
    log(`WebGPU streaming completed in ${duration} ms.`);

    const row = `
        <tr>
        <td>WebGPU processing</td>
        <td>${totalPoints}</td>
        <td>${batchSize}</td>
        <td>${D}</td>
        <td>${duration}</td>
        </tr>
    `;

    document.getElementById('resultsTable').insertAdjacentHTML('beforeend', row);
    uniformBuffer.destroy();
    dataBuffer.destroy();
    centroidBuffer.destroy();
    assignmentBuffer.destroy();
    readBuffer.destroy();
    tracker.free();
  });
}