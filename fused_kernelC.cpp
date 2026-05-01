#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h> // Essential for the OpenMP parallelization used in the paper

// This is your "Compiled Function" logic
extern "C" { // This 'extern "C"' allows Python to see and call this function
    void fused_linear_relu(float* input, float* weight, float* bias, float* output, int size) {
        // The paper uses #pragma omp for to parallelize across your CPU cores [5]
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            // Fused Step 1 & 2: Linear Math + ReLU
            // By doing both here, we avoid a "round trip to memory" [6, 7]
            float val = (input[i] * weight[i]) + bias[i];
            output[i] = std::max(0.0f, val);
        }
    }
}