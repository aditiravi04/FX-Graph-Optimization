import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import torch
from baseline import SimpleNN

# Eager mode: baseline for performance comparisons
def eagerMode(model, x, iters=100):
    # warm up (paper recommended)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        
        startTime = time.perf_counter()
        for i in range(iters):
            model(x)
        endTime = time.perf_counter()

        avgTime = (endTime-startTime)/iters
        print("Avg time: ", avgTime)
        return avgTime

# initialize model + data
model = SimpleNN()
input_data = torch.randn(128, 1024)

# call eager benchmarking:
eagerMode(model, input_data)