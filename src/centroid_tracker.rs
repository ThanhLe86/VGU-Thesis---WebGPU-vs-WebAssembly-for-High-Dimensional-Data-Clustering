use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CentroidTracker {
    centroids: Vec<f32>,
    cluster_counts: Vec<i32>,
    d: usize,
}

#[wasm_bindgen]
impl CentroidTracker {
    #[wasm_bindgen(constructor)]
    pub fn new(initial_centroids: &[f32], d: usize, k: usize) -> CentroidTracker {
        CentroidTracker {
            centroids: initial_centroids.to_vec(),
            cluster_counts: vec![0; k],
            d,
        }
    }

    pub fn update(&mut self, data: &[f32], assignments: &[i32], n: usize) {
        for i in 0..n {
            let cluster = assignments[i] as usize;
            self.cluster_counts[cluster] += 1;
            let learning_rate = 1.0 / (self.cluster_counts[cluster] as f32);
            
            let point_offset = i * self.d;
            let centroid_offset = cluster * self.d;
            
            for dim in 0..self.d {
                let point_val = data[point_offset + dim];
                self.centroids[centroid_offset + dim] += learning_rate * (point_val - self.centroids[centroid_offset + dim]);
            }
        }
    }

    pub fn get_centroids(&self) -> Vec<f32> {
        self.centroids.clone()
    }
}