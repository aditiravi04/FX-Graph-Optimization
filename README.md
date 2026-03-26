# FX-Graph-Optimization

This project implements a simplified version of PyTorch 2’s FX-based compiler optimizations. The goal is to explore graph-level operator fusion to improve performance over standard eager execution. 

Specifically, the project:

1. Builds a baseline PyTorch model composed of sequential nn.Linear and nn.ReLU layers.
2. Captures the model as an FX Graph using torch.fx.symbolic_trace to enable compiler-style analysis.
3. Implements a graph-level optimization pass that identifies Linear→ReLU patterns and fuses them into a single operation, reducing redundant memory operations.

The remaining work focuses on executing fused kernels and benchmarking performance against the baseline model, including scaling to larger networks. This project serves as a hands-on exploration of how compiler techniques like pattern matching and fusion can accelerate neural network inference in PyTorch.
