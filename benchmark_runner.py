import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

SERVER_URL = "http://localhost:3000"

EXPERIMENT_MATRIX = [
    {"points": 10_000,"dims": 32,"clusters": 16}, # WebGPU will likely lose here
    {"points": 100_000,"dims": 32,"clusters": 16},
    {"points": 1_000_000,"dims": 32,"clusters": 16},
    {"points": 5_000_000,"dims": 32,"clusters": 16}, 

    # --- SWEEP 2: Dimensionality (D) ---
    {"points": 500_000, "dims": 2,"clusters": 16}, # Spatial data (compute bound)
    {"points": 500_000, "dims": 128,"clusters": 16}, # Standard ML features
    {"points": 500_000, "dims": 768,"clusters": 16}, # LLM Embeddings (memory bandwidth bound)
    {"points": 500_000, "dims": 1536,"clusters": 16}, # High-res embeddings (stress tests Wasm memory)

    # --- SWEEP 3: Cluster Count (K) ---
    {"points": 500_000, "dims": 32,"clusters": 2},
    {"points": 500_000, "dims": 32,"clusters": 64},
    {"points": 500_000, "dims": 32,"clusters": 512},
    {"points": 500_000, "dims": 32,"clusters": 1024},
]

ENGINES = [
    {"name": "Pure JS",       "button_id": "#btnRunJS"},
    {"name": "Wasm (ST)",     "button_id": "#btnRunWasm"},
    {"name": "Wasm (MT)",     "button_id": "#btnRunWasmMT"},
    {"name": "WebGPU Compute","button_id": "#btnRunWebGPU"},
]

def run_benchmarks():
    # Pass discrete GPU offloading environment to headless Chromium
    env = os.environ.copy()
    env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE",
                "--enable-unsafe-webgpu",
                "--use-angle=vulkan",
                "--no-sandbox",
            ],
            env=env,
        )

        page = browser.new_page()
        page.goto(SERVER_URL)

        # Wait for Wasm and WebGPU async initialization
        page.wait_for_selector("#btnRunWasm:not([disabled])", timeout=10000)
        page.wait_for_selector("#btnRunWebGPU:not([disabled])", timeout=10000)
        time.sleep(1)

        print("[+] Test harness initialized. Running parameter sweep...\n")

        for run in EXPERIMENT_MATRIX:
            n, d, k = run["points"], run["dims"], run["clusters"]
            print(f"--- Dataset: N={n:,} | D={d} | K={k} ---")

            # Set inputs
            page.fill("#numPoints", str(n))
            page.fill("#numDims", str(d))
            page.fill("#numClusters", str(k))

            for engine in ENGINES:
                row_count_before = len(page.query_selector_all("#resultsTable tr"))

                # Trigger benchmark run
                page.click(engine["button_id"])

                # Wait until new table row is appended
                page.wait_for_function(
                    f"document.querySelectorAll('#resultsTable tr').length > {row_count_before}",
                    timeout=300000  # 5 min timeout for large runs
                )

                # Extract last inserted row
                latest_row = page.query_selector_all("#resultsTable tr")[-1]
                cells = [c.inner_text().strip() for c in latest_row.query_selector_all("td")]
                
                # The execution duration is always the last column
                exec_time_ms = float(cells[-1])

                print(f"  {engine['name']:<18}: {exec_time_ms:>10.2f} ms")
                results.append({
                    "Engine": engine["name"],
                    "Points": n,
                    "Dims": d,
                    "Clusters": k,
                    "ExecTime_ms": exec_time_ms
                })

        browser.close()

    return pd.DataFrame(results)

def generate_outputs(df):
    # 1. Output tabular data to console and CSV
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 60)
    print(df.to_markdown(index=False))
    df.to_csv("./results/benchmark_results.csv", index=False)
    print("\n[+] Results saved to 'benchmark_results.csv'")

    # 2. Generate publication-ready comparison plot
    plt.figure(figsize=(10, 6))
    for engine_name, group in df.groupby("Engine"):
        plt.plot(
            group["Points"], 
            group["ExecTime_ms"], 
            marker="o", 
            linewidth=2, 
            label=engine_name
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Edge K-Means Benchmark: Compute Scaling across Backends", fontsize=13, pad=12)
    plt.xlabel("Dataset Size (Points N) - Log Scale", fontsize=11)
    plt.ylabel("Execution Time (ms) - Log Scale", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig("./results/benchmark_scaling_graph.png", dpi=300)
    print("[+] Graph saved to 'benchmark_scaling_graph.png'")

if __name__ == "__main__":
    df_results = run_benchmarks()
    generate_outputs(df_results)