import torch
from torch.fx import symbolic_trace
import torchvision.models as models
from fused_op_pass import fuse_linear_relu # Your tool

def generate_graph_viz():
    # 1. Load a small version of your model for clarity
    model = models.resnet18()
    
    # 2. Capture the graph BEFORE optimization
    print("Capturing Original Graph...")
    traced_original = symbolic_trace(model)
    
    # 3. Capture the graph AFTER optimization
    print("Capturing Fused Graph...")
    traced_fused = fuse_linear_relu(model)

    # 4. Save the text-based graph representations
    # These look great in a report "Appendix" or "Methods" section
    with open("original_graph_structure.txt", "w") as f:
        f.write(str(traced_original.graph))
    
    with open("fused_graph_structure.txt", "w") as f:
        f.write(str(traced_fused.graph))

    print("Success: Graph structures saved to .txt files.")
    print("\n--- SAMPLE OF FUSED GRAPH ---")
    traced_fused.graph.print_tabular()

if __name__ == "__main__":
    generate_graph_viz()
