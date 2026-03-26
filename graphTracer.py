import torch.fx
from baseline import SimpleNN

model = SimpleNN()

traced_graph = torch.fx.symbolic_trace(model)

print("Printing FX Graph Nodes!!!!!!!!!!!!!!!!")

for node in traced_graph.graph.nodes:
    print(f"Op: {node.op}, Target: {node.target}, Name: {node.name}")

print("\n\nPrinting FX Python Code!!!!!!!!!!!!!!!!!!!!!")

print(traced_graph.code)

for name, module in traced_graph.named_modules():
    print(name, "->", module)