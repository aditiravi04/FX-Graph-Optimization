import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedLinearReLU(nn.Module):
    """
    Your 'Fused Kernel'. It performs the Linear operation and 
    the ReLU activation in a single step to mimic a fused backend.
    """
    def __init__(self, original_linear):
        super().__init__()
        # We 'lower' the parameters from the original layer into this one
        self.weight = original_linear.weight
        self.bias = original_linear.bias

    def forward(self, x):
        # In a real backend (like C++), this would be a single loop 
        # that doesn't write intermediate results back to main memory.
        return F.relu(F.linear(x, self.weight, self.bias))
