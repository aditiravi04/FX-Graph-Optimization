import torch
import time
from baseline import SimpleNN
import torch
import time
from baseline import SimpleNN  # Import your model
from fused_op_pass import fuse_linear_relu  # Import your optimization pass

def benchmark(model, x, name, iters=100):
    """
    Standard benchmarking approach as described in the sources:
    Includes a warm-up phase to reduce noise.
    """
    # 1. Warm-up (Source [1, 2] recommends at least 3-10 iterations)
    for _ in range(10):
        model(x)
    
    # 2. Timing the execution
    start_time = time.perf_counter()
    for _ in range(iters):
        model(x)
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / iters
    print(f"{name} Average Latency: {avg_latency * 1000:.3f} ms")
    return avg_latency

def run_verification():
    # Setup hardware and data
    device = "cpu"
    input_data = torch.randn(128, 1024).to(device)

    # Initialize original model in Eager Mode
    model = SimpleNN().eval().to(device)

    # 1. Capture and Optimize (The 'Compiled Function' creation)
    # Note: This returns the GraphModule directly, avoiding the AttributeError
    optimized_model = fuse_linear_relu(model)

    # 2. Soundness Check (Correctness)
    # The paper emphasizes that capture must match Eager results [3, 4]
    with torch.no_grad():
        eager_output = model(input_data)
        optimized_output = optimized_model(input_data)
    
    is_correct = torch.allclose(eager_output, optimized_output, atol=1e-5)
    print(f"--- Correctness Check: {'PASSED' if is_correct else 'FAILED'} ---")

    if not is_correct:
        print("Warning: Optimized model output differs from Eager mode.")
        return

    # 3. Performance Benchmarking
    print("\n--- Performance Comparison ---")
    eager_latency = benchmark(model, input_data, "Eager Mode")
    optimized_latency = benchmark(optimized_model, input_data, "Optimized (Fused) Mode")

    # Calculate Speedup (Source [5, 6] uses geomean speedup over eager)
    speedup = eager_latency / optimized_latency
    print(f"\nFinal Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    run_verification()