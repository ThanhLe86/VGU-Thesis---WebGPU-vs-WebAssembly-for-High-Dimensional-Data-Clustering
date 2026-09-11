export default async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to acquire GPU adapter.");
  }
  const device = await adapter.requestDevice();

  const wgslCode = `
    struct Uniforms {
      d: u32,
      k: u32,
    };
    @group(0) @binding(0) var uniforms: Uniforms;
    @group(0) @binding(1) var data: array;
    @group(0) @binding(2) var centroids: array;
    @group(0) @binding(3) var assignments: array;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3) {
      let i = global_id.x;
      let num_points = arrayLength(&assignments);
      if (i >= num_points) { return; }

      let D = uniforms.d;
      let K = uniforms.k;
      
      var min_dist = 3.40282347e+38f;
      var best_cluster: i32 = 0;

      for (var c: u32 = 0; c < K; c = c + 1) {
        var dist_sq: f32 = 0.0;
        for (var dim: u32 = 0; dim < D; dim = dim + 1) {
          let diff = data[i * D + dim] - centroids[c * D + dim];
          dist_sq = dist_sq + (diff * diff);
        }
        if (dist_sq < min_dist) {
          min_dist = dist_sq;
          best_cluster = i32(c);
        }
      }
      assignments[i] = best_cluster;
    }
  `;

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: wgslCode }),
      entryPoint: 'main',
    },
  });

  return { device, pipeline };
}