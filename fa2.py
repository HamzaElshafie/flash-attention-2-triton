import torch
import triton
import triton.language as tl

@triton.jit
def __attn_fwd_inner():
    pass

@triton.jit
def __attn_fwd():
    pass

class TritonAttention(torch.autograd.Function):
    pass

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        pass
        

# Benchmarking
def torch_attention():
    pass

def triton_attention():
    pass

def run_benchmark():
    pass

# Plot TFLOPs vs seq_len

def benchmark_flashattention():
    pass

if __name__ == "__main__":
    benchmark_flashattention.run_benchmark(show_plots=True, print_data=True)
