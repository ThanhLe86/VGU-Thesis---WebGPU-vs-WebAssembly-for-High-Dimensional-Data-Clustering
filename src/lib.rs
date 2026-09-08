use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init_console() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn run_wasm_assignment(
    data: &[f32], 
    centroids: &[f32], 
    n: usize, 
    d: usize, 
    k: usize
) -> Vec<i32> {
    // initializes the array with zeros
    let mut assignments = vec![0; n];

    for i in 0..n {
        let point_offset = i * d;
        let mut min_dist_sq = f32::INFINITY;
        let mut best_cluster = 0;

        for j in 0..k {
            let centroid_offset = j * d;
            let mut dist_sq = 0.0;

            for dim in 0..d {
                let diff = data[point_offset + dim] - centroids[centroid_offset + dim];
                dist_sq += diff * diff;
            }

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                best_cluster = j as i32;
            }
        }
        assignments[i] = best_cluster;
    }

    assignments
}

// pub fn add(left: u64, right: u64) -> u64 {
//     left + right
// }
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
