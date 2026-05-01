import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from cpp_backend_wrapper import run_cpp_fused

def benchmark_all():
    # Setup data (mimicking a layer in ResNet18)
    size = 1024 * 1024 
    x = torch.randn(size)
    w = torch.randn(size)
    b = torch.randn(size)
    
    results = {}

    def run_benchmark(name, func, *args):
        print(f"Benchmarking {name}...")
        # 1. Warm-up (Source: repeated iterations to stabilize) [4]
        for _ in range(3): func(*args)
        
        # 2. Timing
        start = time.perf_counter()
        for _ in range(100): func(*args)
        end = time.perf_counter()
        
        latency = (end - start) / 100 * 1000 # ms
        results[name] = latency

    # Eager Mode
    run_benchmark("Eager Mode", lambda: torch.relu(x * w + b))

    # Python Fused (simulating what you did in Week 3)
    def py_fused(input, weight, bias):
        return torch.nn.functional.relu(input * weight + bias)
    run_benchmark("Python Fused", py_fused, x, w, b)

    # C++ Fused (Your new kernel)
    run_benchmark("C++ Fused (OpenMP)", run_cpp_fused, x, w, b)

    # Save and Plot
    df = pd.DataFrame([results])
    df.to_csv("final_results.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    colors = ['#95a5a6', '#f39c12', '#27ae60']
    bars = plt.bar(results.keys(), results.values(), color=colors)
    plt.ylabel("Latency (ms)")
    plt.title("Final Performance: Eager vs. Python vs. C++ Backend")
    
    # Calculate Speedup
    eager_time = results["Eager Mode"]
    cpp_time = results["C++ Fused (OpenMP)"]
    speedup = eager_time / cpp_time
    
    plt.annotate(f"Final Speedup: {speedup:.2f}x", xy=(2, cpp_time), 
                 xytext=(1.5, eager_time*0.8), arrowprops=dict(arrowstyle="->"))
    
    plt.savefig("final_performance_report.png")
    print(f"\nFinal Speedup over Eager: {speedup:.2f}x")
    print("Report saved as final_performance_report.png")

if __name__ == "__main__":
    benchmark_all()