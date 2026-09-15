import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

SERVER_URL = "http://localhost:3000"

SWEEPS = {
     "Points": [
        {"points": 10, "dims": 32, "clusters": 16},
        # {"points": 100_000, "dims": 32, "clusters": 16},
        # {"points": 1_000_000, "dims": 32, "clusters": 16},
        # {"points": 5_000_000, "dims": 32, "clusters": 16},
        # {"points": 10_000_000, "dims": 32, "clusters": 16},
        # {"points": 50_000_000, "dims": 32, "clusters": 16},
        # {"points": 500_000_000, "dims": 32, "clusters": 16},
        # {"points": 1_000_000_001, "dims": 32, "clusters": 16},
        # {"points": 5_000_000_000, "dims": 32, "clusters": 16},
        # {"points": 1_000_000_000_000, "dims": 32, "clusters": 16},
    ],
    "Dims": [
        {"points": 500_000, "dims": 2,    "clusters": 16},
        {"points": 500_000, "dims": 32,   "clusters": 16},
        {"points": 500_000, "dims": 128,  "clusters": 16},
        {"points": 500_000, "dims": 150,  "clusters": 16},
        {"points": 500_000, "dims": 200,  "clusters": 16},
        {"points": 500_000, "dims": 256,  "clusters": 16},
        {"points": 500_000, "dims": 367,  "clusters": 16},
        {"points": 500_000, "dims": 512,  "clusters": 16},
        {"points": 500_000, "dims": 1024, "clusters": 16},
    ],
    "Clusters": [
        {"points": 500_000, "dims": 32, "clusters": 2},
        # {"points": 500_000, "dims": 32, "clusters": 16},
        # {"points": 500_000, "dims": 32, "clusters": 64},
        # {"points": 500_000, "dims": 32, "clusters": 256},
        # {"points": 500_000, "dims": 32, "clusters": 512},
        # {"points": 500_000, "dims": 32, "clusters": 1024},
        # {"points": 500_000, "dims": 32, "clusters": 2048},
    ]
}

ENGINES = [
    {"name": "Pure JS",        "button_id": "#btnRunJS"},
    {"name": "Wasm (ST)",      "button_id": "#btnRunWasm"},
    {"name": "Wasm (MT)",      "button_id": "#btnRunWasmMT"},
    {"name": "WebGPU Compute", "button_id": "#btnRunWebGPU"},
]

# Workload execution ceilings (N * D * K distance operations)
JS_MAX_COMPUTE_OPS = 3_000_000_000         
WASM_ST_MAX_COMPUTE_OPS = 15_000_000_000   

def run_benchmarks():
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

        page.wait_for_selector("#btnRunWasm:not([disabled])", timeout=15000)
        page.wait_for_selector("#btnRunWebGPU:not([disabled])", timeout=15000)
        time.sleep(1)

        # Eliminates Cold-Start JIT & Shader Compilation Bias
        print("[+] Running warm-up passes to compile WGSL shaders and prime JIT/Worker pools...")
        page.fill("#numPoints", "5000")
        page.fill("#numDims", "32")
        page.fill("#numClusters", "16")

        for engine in ENGINES:
            row_count_before = len(page.query_selector_all("#resultsTable tr"))
            page.click(engine["button_id"], timeout=0, no_wait_after=True)
            page.wait_for_function(
                f"document.querySelectorAll('#resultsTable tr').length > {row_count_before}",
                timeout=0
            )

        print("[+] Warm-up completed.\n")

        for sweep_name, matrix in SWEEPS.items():
            print(f"\n=== Starting Sweep: {sweep_name} ===")
            for run in matrix:
                n, d, k = run["points"], run["dims"], run["clusters"]
                compute_ops = n * d * k
                print(f"--- Dataset: N={n:,} | D={d} | K={k} ({compute_ops:,.0f} ops) ---")

                page.fill("#numPoints", str(n))
                page.fill("#numDims", str(d))
                page.fill("#numClusters", str(k))

                for engine in ENGINES:
                    # Guard: Prevent Pure JS from hanging when operations exceed practical bounds
                    if engine["name"] == "Pure JS" and compute_ops > JS_MAX_COMPUTE_OPS:
                        print(f"  {engine['name']:<18}: SKIPPED (Exceeds Pure JS runtime limit: {compute_ops:,.0f} > {JS_MAX_COMPUTE_OPS:,.0f})")
                        continue

                    # Guard: Prevent Single-Threaded Wasm from stalling on multi-billion operations
                    if engine["name"] == "Wasm (ST)" and compute_ops > WASM_ST_MAX_COMPUTE_OPS:
                        print(f"  {engine['name']:<18}: SKIPPED (Exceeds Wasm (ST) runtime limit: {compute_ops:,.0f} > {WASM_ST_MAX_COMPUTE_OPS:,.0f})")
                        continue

                    row_count_before = len(page.query_selector_all("#resultsTable tr"))

                    page.click(engine["button_id"], timeout=0, no_wait_after=True)

                    page.wait_for_function(
                        f"document.querySelectorAll('#resultsTable tr').length > {row_count_before}",
                        timeout=0
                    )

                    latest_row = page.query_selector_all("#resultsTable tr")[-1]
                    cells = [c.inner_text().strip() for c in latest_row.query_selector_all("td")]
                    exec_time_ms = float(cells[-1])

                    print(f"  {engine['name']:<18}: {exec_time_ms:>10.2f} ms")
                    
                    results.append({
                        "Sweep_Type": sweep_name,
                        "Engine": engine["name"],
                        "Points": n,
                        "Dims": d,
                        "Clusters": k,
                        "ExecTime_ms": exec_time_ms
                    })

        browser.close()

    return pd.DataFrame(results)

def generate_outputs(df):
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 60)
    print(df.to_markdown(index=False))
    
    os.makedirs("./results", exist_ok=True)
    df.to_csv("./results/benchmark_results.csv", index=False)
    print("\n[+] Results saved to './results/benchmark_results.csv'")

    plot_configs = {
        "Points":   {"x_col": "Points",   "title": "Scaling Dataset Size (Compute vs Bus Transfer)", "x_label": "Dataset Size (N)"},
        "Dims":     {"x_col": "Dims",     "title": "Scaling Dimensionality (Memory Bandwidth Bound)",  "x_label": "Dimensions (D)"},
        "Clusters": {"x_col": "Clusters", "title": "Scaling Clusters (Arithmetic Intensity Bound)", "x_label": "Clusters (K)"}
    }

    colors = {
        "Pure JS": "#7f7f7f",
        "Wasm (ST)": "#1f77b4",
        "Wasm (MT)": "#2ca02c",
        "WebGPU Compute": "#d62728"
    }

    for sweep, config in plot_configs.items():
        sweep_df = df[df["Sweep_Type"] == sweep]
        if sweep_df.empty: 
            continue

        plt.figure(figsize=(9, 5.5))
        for engine_name, group in sweep_df.groupby("Engine"):
            group = group.sort_values(config["x_col"])
            
            plt.plot(
                group[config["x_col"]], 
                group["ExecTime_ms"], 
                marker="o", 
                linewidth=2.2, 
                markersize=6,
                color=colors.get(engine_name, None),
                label=engine_name
            )

        if sweep == "Points":
            plt.xscale("log")
        else:
            plt.xscale("log", base=2)
            
        plt.yscale("log")
        plt.title(f"Edge K-Means: {config['title']}", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel(f"{config['x_label']} (Log Scale)", fontsize=10)
        plt.ylabel("Execution Time (ms) - Log Scale", fontsize=10)
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend(frameon=True, facecolor="white", framealpha=0.9)
        plt.tight_layout()

        filename = f"./results/benchmark_scaling_{sweep.lower()}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[+] Graph saved to '{filename}'")

if __name__ == "__main__":
    df_results = run_benchmarks()
    generate_outputs(df_results)