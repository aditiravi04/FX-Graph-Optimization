import torch
import ctypes
import os

# 1. ADD THE MINGW BIN DIRECTORY TO THE DLL SEARCH PATH
# This is the folder where g++ and its libraries (like OpenMP) live
mingw_bin_path = r"C:\msys64\ucrt64\bin" 
if os.path.exists(mingw_bin_path):
    os.add_dll_directory(mingw_bin_path)

# 2. Load the compiled library using an absolute path
lib_path = os.path.abspath("fused_kernel.dll")
try:
    fused_lib = ctypes.CDLL(lib_path)
except Exception as e:
    print(f"Error loading DLL: {e}")
    # If it still fails, check if fused_kernel.dll is actually in this folder
    print(f"Looking for DLL at: {lib_path}")

# Define the argument types for our C++ function:
# (float* input, float* weight, float* bias, float* output, int size)
fused_lib.fused_linear_relu.argtypes = [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int
]

def run_cpp_fused(input_tensor, weight_tensor, bias_tensor):
    """
    Simulates the TorchInductor execution of a fused C++ kernel.
    """
    # Flatten tensors and ensure they are on CPU
    size = input_tensor.numel()
    output_tensor = torch.empty_like(input_tensor)

    # Convert PyTorch tensors to C-compatible pointers
    in_ptr = input_tensor.data_ptr()
    w_ptr = weight_tensor.data_ptr()
    b_ptr = bias_tensor.data_ptr()
    out_ptr = output_tensor.data_ptr()

    # Call the C++ kernel (this uses OpenMP internally)
    fused_lib.fused_linear_relu(
        ctypes.cast(in_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(w_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)),
        size
    )
    return output_tensor
