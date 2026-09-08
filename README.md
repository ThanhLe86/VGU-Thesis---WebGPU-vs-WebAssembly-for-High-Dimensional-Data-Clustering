# Comparative Edge-Compute Performance: WebGPU vs. WebAssembly for High-Dimensional Data Clustering

A browser-based empirical benchmarking suite comparing the performance characteristics of client-side execution environments (**WebAssembly** on the CPU vs. **WebGPU** via compute shaders) for high-dimensional $K$-means clustering.

---

## 1. Project Overview & Research Context

High-dimensional data analysis in web environments has traditionally relied on either server-side cloud offloading or client-side interpreted runtimes (e.g., Pyodide/Python runtimes), both of which incur significant latency overheads.

This research investigates the performance tradeoffs of client-side edge computing by bypassing interpreted layers:

* **WebAssembly (Wasm):** Near-native compiled CPU execution utilizing sequential execution and SIMD (Single Instruction, Multiple Data) vectorization.
* **WebGPU:** Massively parallel throughput via WGSL (WebGPU Shading Language) compute pipelines.

### Core Research Objective

Identify the **performance crossover point**—the threshold of dataset size ($N$) and dimensionality ($D$) where WebGPU's parallel throughput overcomes the latency penalty of system RAM-to-VRAM buffer copies.

---

## 2. Current Implementation Status

* [x] **Benchmarking Test Harness:** Web UI with configurable parameters ($N, D, K$) and high-resolution timing via `performance.now()`.
* [x] **Synthetic Data Engine:** Continuous flat `Float32Array` generator preserving cache locality and hardware buffer compatibility.
* [x] **Baseline JavaScript Engine:** Single-threaded Euclidean distance assignment loop for comparative baseline.
* [x] **WebAssembly Engine (Baseline):** Native-speed Rust kernel compiled to `wasm32-unknown-unknown` with `wasm-bindgen` bindings.

---

## 4. Prerequisites & Environment Setup

The development environment is configured on **EndeavourOS (Arch Linux)** using stable Rust and Node.js.

### System Packages

```bash
sudo pacman -S base-devel rustup nodejs npm wasm-pack

```

### Rust Toolchain & Wasm Target

```bash
rustup default stable
rustup target add wasm32-unknown-unknown
```

---

## 5. Building & Running Locally

### 1. Compile the WebAssembly Module

Compile the Rust kernel into web-ready ES6 modules:

```bash
wasm-pack build --target web

```

### 2. Serve the Harness

Because WebAssembly requires fetch/streaming mechanics, serve the root directory over a local HTTP server:

```bash
# Using Node's npx serve
npx serve .

# Or use the VS Code Live Server extension on index.html

```

Open your browser to `http://localhost:3000` (or the port specified by your local server).

---

## 6. Known Architectural Constraints & Edge Cases

### The 32-Bit WebAssembly Address Limit

WebAssembly currently targets a 32-bit linear address space (`wasm32`).

* The maximum indexable single allocation in Rust on `wasm32` is bounded by `isize::MAX` (**2,147,483,647 bytes / ~2.14 GB**).
* Attempting to allocate datasets exceeding this limit (e.g., $N = 10,000,000$ points at $D = 64$, requiring $10,000,000 \times 64 \times 4 \text{ bytes} \approx 2.56 \text{ GB}$) results in an immediate out-of-memory (OOM) capacity overflow:
```text
panicked at library/alloc/src/raw_vec.rs: capacity overflow

```


* **Benchmarking Parameter Scope:** Empirical benchmarks for the single-threaded in-memory Wasm pipeline should maintain $N$ between $10,000$ and $500,000$ points to guarantee allocation stability inside the browser sandbox.