import torch
import torch.nn as nn
import torch.nn.functional as F

class FuseFunc(nn.Module):
    def __init__(self, linear_module):
        super().__init__()
        # get weights & bias for linear layer
        self.weight = nn.Parameter(linear_module.weight.data)
        self.bias = nn.Parameter(linear_module.bias.data)

    def forward(self, x):
        # do both operations in 1 step
            # i.e. combine linear + ReLU in a single fwd pass
        # goal is to avoid writing linear o/p to main mem before reLU reads it -> "fusion"
        return F.relu(F.linear(x, self.weight, self.bias))