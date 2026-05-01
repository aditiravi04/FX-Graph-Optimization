import torch
from torch.fx import symbolic_trace
import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt
from fused_op_pass import fuse_linear_relu # Your optimization pass

def save_graph_as_figure(traced_model, filename="figure1_graph_structure.png"):
    """
    Transforms the FX Graph tabular data into a professional PNG image
    to serve as Figure 1 in the Soundness Verification section.
    """
    node_data = []
    # Iterating through nodes to capture the 'blueprint' [2, 5]
    for node in traced_model.graph.nodes:
        # We capture Opcodes, Targets, and Data Flow (Args) [2]
        node_data.append([node.op, str(node.target), str(node.args), str(node.kwargs)])
    
    # Create a DataFrame for high-quality rendering
    df = pd.DataFrame(node_data, columns=['Opcode', 'Target', 'Args (Data Flow)', 'Kwargs'])
    
    # Use Matplotlib to render the table as a high-resolution image
    # We set the figure size dynamically based on the number of nodes
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.4 + 1))
    ax.axis('off')
    
    # Create the table with clean styling for a research report
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     loc='center', 
                     cellLoc='left',
                     colColours=["#f2f2f2"] * 4) # Light grey header
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Save as a high-DPI image for the paper
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Success: Figure 1 saved to {filename}")

def generate_graph_viz():
    # 1. Load a standard model to verify the 'Breadth First' principle [6, 7]
    model = models.resnet18()
    
    # 2. Capture the graph BEFORE optimization
    print("Capturing Original Graph...")
    traced_original = symbolic_trace(model)
    
    # 3. Capture the graph AFTER optimization (Reimplementing the Fusion Pass) [8, 9]
    print("Capturing Fused Graph...")
    traced_fused = fuse_linear_relu(model)

    # 4. Save the text-based graph representations for your methodology section [2]
    with open("original_graph_structure.txt", "w") as f:
        f.write(str(traced_original.graph))
    
    with open("fused_graph_structure.txt", "w") as f:
        f.write(str(traced_fused.graph))

    # 5. Generate Figure 1 for the Soundness Verification section [4]
    # This visualizes the 'blueprint' before it is lowered to the C++ backend
    save_graph_as_figure(traced_fused, "figure1_graph_structure.png")

    print("\n--- TABULAR PREVIEW OF FUSED GRAPH ---")
    traced_fused.graph.print_tabular()

if __name__ == "__main__":
    generate_graph_viz()