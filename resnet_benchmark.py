import torch
import torchvision.models as models
import time
import csv
from fused_op_pass import fuse_linear_relu # Your tool from Week 2

def benchmark(model, input_data, iters=100, warmups=3):
    """
    Standard protocol from the paper: 
    3 warm-up iterations followed by 100 trials [5].
    """
    model.eval()
    with torch.no_grad():
        # Warm-up to stabilize CPU clocks and caches
        for _ in range(warmups):
            model(input_data)
        
        # Actual measurement
        start = time.perf_counter()
        for _ in range(iters):
            model(input_data)
        end = time.perf_counter()
    
    return (end - start) / iters * 1000  # Returns average latency in ms

def run_phase_4():
    # 1. Scaling to a real-world model (ResNet18) [6]
    print("Loading ResNet18 (Model Breadth Test)...")
    model = models.resnet18()
    input_data = torch.randn(1, 3, 224, 224)

    # 2. Baseline measurement (Eager Mode) [7, 8]
    print("Measuring Eager Mode...")
    eager_latency = benchmark(model, input_data)

    # 3. Optimization Pass [9, 10]
    print("Applying Linear+ReLU Fusion...")
    optimized_model = fuse_linear_relu(model)

    # 4. Optimized measurement (Fused Mode) [11, 12]
    print("Measuring Optimized Mode...")
    fused_latency = benchmark(optimized_model, input_data)

    # 5. Save results to CSV (as done in the paper's infrastructure [3])
    results = {
        "Model": "ResNet18",
        "Eager_ms": round(eager_latency, 3),
        "Fused_ms": round(fused_latency, 3),
        "Speedup": round(eager_latency / fused_latency, 2)
    }
    
    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    print(f"\nResults saved. Speedup: {results['Speedup']}x")

if __name__ == "__main__":
    run_phase_4()