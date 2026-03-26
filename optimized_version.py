from baseline import SimpleNN
from fused_op_pass import fuse_linear_relu
import torch

# 1. Initialize original model
model = SimpleNN()
input_data = torch.randn(128, 1024)

# 2. Run your optimization pass
optimized_model = fuse_linear_relu(model)

# 3. Verify the "Systems Concept"
print("\n--- Optimized Graph Code ---")
print(optimized_model.code)

# 4. Ensure correctness [17]
with torch.no_grad():
    original_out = model(input_data)
    optimized_out = optimized_model(input_data)
    
    # The outputs should be identical
    is_correct = torch.allclose(original_out, optimized_out)
    print(f"\nOptimization Soundness Check: {'PASSED' if is_correct else 'FAILED'}")