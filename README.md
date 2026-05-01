# PyTorch 2 Simplified: FX-Based Graph Fusion Compiler

This project implements a simplified version of the PyTorch 2 compiler pipeline, with a focus on dynamic graph capture, operator fusion, and low-level C++ backend execution. It aims to overcome the limitations of standard eager execution by identifying and optimizing common computation patterns, thereby reducing redundant computation and memory traffic.

**Core Features**
The system includes the following key components:

**Graph Capture**
A lightweight analogue of TorchDynamo built using torch.fx.symbolic_trace, which converts PyTorch models into a structured Intermediate Representation (IR). This enables static analysis and transformation of dynamic models.

**Optimization Pass**
A graph-level peephole optimization pass that performs pattern matching on the FX graph. In particular, it identifies and fuses common operator sequences such as Linear + ReLU, reducing intermediate memory writes and improving execution efficiency.

**C++ Backend Lowering**
A custom execution backend that translates the optimized FX graph into parallelized C++ code using OpenMP. This bypasses the Python interpreter during execution and enables significantly faster runtime performance.

**Wrapper Code Generation**
A ctypes-based interface that automatically generates the necessary glue code to invoke compiled C++ kernels directly from Python, allowing seamless integration with existing PyTorch workflows.

**Project Structure**
- fused_op_pass.py: Implements the graph-level fusion pass, including pattern detection and operator fusion logic.
- fused_kernelC.cpp: Contains the high-performance C++ implementation of fused operations, optimized using OpenMP.
- visualize_graphs.py: Utility script for visualizing the captured FX graph, providing insight into the intermediate representation.
- verify.py: Main benchmarking and correctness verification script used to compare eager execution against optimized and compiled pipelines.

**How to Run**
1. Visualize the Graph IR

To inspect the intermediate representation used by the compiler before lowering to C++, run:
python visualize_graphs.py

Expected Output:
figure1_graph_structure.png: A structured visualization of the FX graph showing operations, targets, and data flow.
fused_graph_structure.txt: A textual representation of the optimized graph after fusion passes.

2. Benchmark and Verify Correctness

To evaluate performance improvements and verify numerical correctness, run:
python verify.py

Expected Output:
Numerical Accuracy Check: Confirms that compiled outputs match eager execution using torch.allclose.
Performance Metrics: Reports average latency across eager mode, Python-level fusion, and the C++ backend (with warm-up and repeated trials).
final_performance_report.png: A bar chart showing relative speedups, typically achieving ~1.64× improvement in the C++ backend.
scaling_comparison.png: A visualization demonstrating how performance benefits scale from small MLPs to larger architectures such as ResNet-18.

**Key Results**
*Memory Traffic Reduction*
By fusing operations such as Linear and ReLU, the system eliminates intermediate memory writes and reads between layers, significantly reducing memory bandwidth overhead.

*Hardware-Level Parallelism*
The C++ backend leverages OpenMP (#pragma omp parallel for) to distribute computation across multiple CPU cores, enabling parallel execution that is not achievable in standard Python execution.

*Amortized Compilation Overhead*
Experimental results show that the benefits of compilation increase with model size. Larger models are able to amortize the fixed overhead of graph capture and compilation more effectively, leading to greater relative speedups.
