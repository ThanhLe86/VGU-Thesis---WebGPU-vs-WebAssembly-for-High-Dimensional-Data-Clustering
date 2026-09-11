# Comparative Edge-Compute Performance: WebGPU vs. WebAssembly for High-Dimensional Data Clustering

A browser-based empirical benchmarking suite comparing the performance characteristics of client-side execution environments (**WebAssembly** on the CPU vs. **WebGPU** via compute shaders) for high-dimensional $K$-means clustering.

---

## 1. Project Overview & Research Context

High-dimensional data analysis in web environments has traditionally relied on either server-side cloud offloading or interpreted runtimes, both of which incur substantial latency and memory overheads.

This research investigates the performance tradeoffs of client-side execution by benchmarking:

* **WebAssembly (Wasm):** Single-threaded and multi-core CPU execution accelerated with SIMD128 vectorization.
* **WebGPU:** Massively parallel execution via WGSL compute shaders.

### Core Research Objective

Identify the **performance crossover point**—the dataset size ($N$) and dimensionality ($D$) threshold where WebGPU's parallel throughput overcomes the latency penalty of system RAM-to-VRAM buffer copies.

---

## 2. Current Implementation Status

* [x] **Benchmarking Test Harness:** Web interface with configurable $N, D, K$ parameters and high-resolution timing via `performance.now()`.
* [x] **Streaming Mini-Batch Architecture:** $O(\text{batch\_size})$ constant memory footprint bypassing browser 32-bit linear address limits.
* [x] **Cross-Origin Isolation Server:** Dedicated Node.js server injecting COOP/COEP headers to unlock `SharedArrayBuffer`.
* [x] **Baseline JavaScript Engine:** Mini-Batch streaming Euclidean distance assignment loop.
* [x] **Single-Threaded Wasm Engine:** Rust kernel compiled to `wasm32-unknown-unknown` via `wasm-bindgen`.


* [x] **Multi-Threaded Wasm Engine:** Persistent Web Worker pool partitioning batch slices over zero-copy `SharedArrayBuffer` memory.
* [x] **SIMD128 Auto-Vectorization:** Build flags enabled targeting 128-bit vector registers.
* [ ] **WebGPU Compute Pipeline:** WGSL distance calculation kernel and buffer binding (Upcoming).

---

## 3. Architecture & Memory Management

### The Streaming Mini-Batch Model

Standard full-batch Lloyd's $K$-means requires allocating the entire dataset contiguously, hitting the 32-bit Wasm memory limit ($2.14\text{ GB}$) and triggering browser `RangeError` / OOM crashes at scale.

This engine adopts a **Streaming Mini-Batch** architecture:

* Batches are dynamically sampled, assigned to centroids, and used to iteratively update cluster centers.
* Memory consumption remains constant ($<10\text{ MB}$) regardless of whether $N = 10^4$ or $N = 10^9$.
* In multi-threaded mode, batches are mapped across a persistent pool of Web Workers via `SharedArrayBuffer` views, eliminating thread instantiation and message serialization overhead.

---

## 4. Prerequisites & Environment Setup

Configured on **EndeavourOS (Arch Linux)** using stable Rust and Node.js.

### System Packages

```bash
sudo pacman -S base-devel rustup nodejs npm wasm-pack wabt

```

### Rust Toolchain & SIMD Configuration

```bash
rustup default stable
rustup target add wasm32-unknown-unknown

```

Ensure `.cargo/config.toml` includes:

```toml
[build]
rustflags = ["-C", "target-feature=+simd128"]

```

---

## 5. Building & Running Locally

### 1. Compile the WebAssembly Module

```bash
wasm-pack build --target web

```

*Optional:* Verify SIMD instructions in the compiled binary:

```bash
wasm2wat pkg/edge_kmeans_benchmark_bg.wasm | grep -E "v128|f32x4" | head -n 10

```

### 2. Run the Cross-Origin Isolated Server

`SharedArrayBuffer` requires strict cross-origin isolation policies:

```bash
node server.js

```

Open `http://localhost:3000` in a Chromium-based browser.