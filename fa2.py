import math
import torch
import triton
import triton.language as tl
import triton.testing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offsets_q: tl.constexpr,
    offsets_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    ):

    if STAGE == 1:
        # Will work only with elements below main diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q # For ex, if BLOCK_SIZE_Q = 64 and block_index_q = 2, 
                                                # this Q block handles rows 128–191, so thoe rows should see columns 0 up to 127 only
    elif STAGE == 2:
        # On the diagonal
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q 
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # Load K𝑗,V𝑗 from HBM to on-chip SRAM.
        K_block = tl.load(K_block_ptr)
        # On chip, compute S(𝑗)𝑖 = Qi @ K^𝑇
        S = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offsets_q.expand_dims(1) >= (start_kv + offsets_kv.expand_dims(0))
            S = S * softmax_scale + tl.where(mask, 0, -1.0e6)
            # 𝑚(𝑗)𝑖 = max(𝑚(𝑗−1)𝑖, rowmax(S(𝑗)𝑖))
            m_ij = tl.maximum(m_i, tl.max(S, 1) * softmax_scale)
            S -= m_ij.expand_dims(1)
        else:
            m_ij = tl.maximum(m_i, tl.max(S, 1) * softmax_scale)
            S = S * softmax_scale - m_ij.expand_dims(1)

        # P(𝑗)𝑖 = exp(S(𝑗)𝑖 − 𝑚(𝑗)𝑖)
        P = tl.math.exp(S)

        # rowsum(P(𝑗)𝑖) --> For the normalisation factor later
        l_ij = tl.sum(P, 1)

        # Correction factor for previous block's l_i
        correction_factor = tl.math.exp(m_i - m_ij)
        l_i = (correction_factor * l_i) + l_ij

        # Load Vj block
        V_block = tl.load(V_block_ptr)
        
        P = P.to(tl.float16)
        V_block = V_block.to(tl.float16)

        O_block = O_block * correction_factor.expand_dims(1) # Fix previous block
        O_block = tl.dot(P, V_block, O_block)

        # Save new maximum
        m_i = m_ij

        # Advance pointers to next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i

    
@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    O,
    softmax_scale,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, # Queries to group toegther
    BLOCK_SIZE_KV: tl.constexpr, # Keys/Values to group together
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    # Head index changes fastest, then the batch index. This is the same as the remapping trick from 1D to 2D in CUDA. 
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # Offset to the start of the entire [SEQ_LEN, HEAD_DIM] tensor for a given (batch, head)
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        # Advance pointer to the [SEQ_LEN, HEAD_DIM] tensor using offset
        base=Q+qkv_offset, # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        shape=[SEQ_LEN, HEAD_DIM],
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O+qkv_offset, # O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
        shape=[SEQ_LEN, HEAD_DIM],
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offsets_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) # Offsets for the global index of each token in the sequence
    offsets_kv = tl.arange(0, BLOCK_SIZE_KV)

    # Create tensor to store running maximum across BLOCK_SIZE_Q
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
    # Create tensor to store running normalisation factor across BLOCK_SIZE_Q
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0 # check this
    # Create tensor for accumilating output for block
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32) 

    # Loads block from DRAM to SRAM
    Q_block = tl.load(Q_block_ptr)

    if STAGE == 1 or STAGE == 3: # STAGE 3 is for causal (preserving autoregressive property) and stage 1 is for non-causal
        # Runs for non-causal or for blocks to the left of the main diagonal (past tokens)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offsets_q,
            offsets_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offsets_q,
            offsets_kv,
            SEQ_LEN,
        )

    O_block = O_block / l_i.expand_dims(1)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        # Make sure head dimension matches for Q, K and V
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V 

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        O = torch.empty_like(Q) # Nxd
        stage = 3 if causal else 1

        # Number of parallel programs --> BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), # We block only across the rows
            BATCH_SIZE * NUM_HEADS, # Number of parallel programs
            1,
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            O=O,
            softmax_scale=softmax_scale,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        return O

# Benchmarking 
def torch_mha_attention(Q, K, V, softmax_scale, causal):
    Q, K, V = Q.half(), K.half(), V.half()
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale
    )

def torch_attention(Q, K, V, softmax_scale, causal):
    Q, K, V = Q.half(), K.half(), V.half()
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    if causal:
        S = Q.size(-2) # (SEQ_LEN, SEQ_LEN)
        mask = torch.tril(torch.ones((S, S), device=Q.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(Q.dtype)
    return attn_probs @ V

def triton_attention(Q, K, V, softmax_scale, causal):
    return TritonAttention.apply(Q, K, V, causal, softmax_scale)

def run_benchmark(seq_len, provider, batch_size=4, num_heads=8, head_dim=64, causal=True, check_outputs=True):
    dtype = torch.float16

    Q = torch.randn((batch_size, num_heads, seq_len, head_dim), device=DEVICE, dtype=dtype)
    K = torch.randn((batch_size, num_heads, seq_len, head_dim), device=DEVICE, dtype=dtype)
    V = torch.randn((batch_size, num_heads, seq_len, head_dim), device=DEVICE, dtype=dtype)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    def out_triton(): return triton_attention(Q, K, V, softmax_scale, causal)
    def out_torch(): return torch_attention(Q, K, V, softmax_scale, causal)
    def out_mha(): return torch_mha_attention(Q, K, V, softmax_scale, causal)

    if check_outputs and provider == 'triton':
        try:
            torch_out = out_torch()
            triton_out = out_triton()
            triton.testing.assert_close(torch_out, triton_out, atol=2e-2, rtol=2e-3)
            print(f"[seq_len={seq_len}] Triton output match: PASSED")
        except AssertionError:
            print(f"[seq_len={seq_len}] Triton output match: FAILED")
    elif check_outputs and provider == 'mha':
        try:
            ref = out_torch()
            mha_out = out_mha()
            triton.testing.assert_close(ref, mha_out, atol=2e-2, rtol=2e-3)
            print(f"[seq_len={seq_len}] Torch MHA output match: PASSED")
        except AssertionError:
            print(f"[seq_len={seq_len}] Torch MHA output match: FAILED")

    # Warm-up
    for _ in range(10):
        if provider == 'torch':
            out_torch()
        elif provider == 'triton':
            out_triton()
        elif provider == 'mha':
            out_mha()
    torch.cuda.synchronize()

    if provider == 'torch':
        fn = out_torch
    elif provider == 'triton':
        fn = out_triton
    elif provider == 'mha':
        fn = out_mha
    else:
        raise ValueError(f"Unknown provider {provider}")

    runtimes_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    median_runtime_s = runtimes_ms[0] / 1e3

    causal_scale = 0.5 if causal else 1.0
    flops_per_head = 2 * causal_scale * seq_len * seq_len * head_dim
    total_flops = flops_per_head * batch_size * num_heads
    tflops = (total_flops / median_runtime_s) / 1e12

    return tflops

# Plot TFLOPs vs SEQ_LEN
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel='Throughput (TFLOPs/s)', 
        plot_name='flashattention-tflops',
        args={'causal': True},
    )
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['torch', 'mha', 'triton'],
        line_names=['Naive Torch', 'Torch MHA', 'Triton'],
        styles=[('green', '-'), ('red', '-'), ('blue', '-')],
        ylabel='Throughput (TFLOPs/s)',
        plot_name='flashattention-tflops',
        args={'causal': True},
    )
)
def benchmark_flashattention(seq_len, provider, causal):
    return run_benchmark(seq_len=seq_len, provider=provider, causal=causal)

# flashattention-tflops:
#    seq_len  Naive Torch  nn.MultiheadAttention     Triton
# 0    512.0     1.765280              15.887515  20.164923
# 1   1024.0     1.972862              34.952534  36.157792
# 2   2048.0     1.907369              55.553695  52.428801
# 3   4096.0     2.083673              73.103337  67.923954
# 4   8192.0     2.142171              79.231245  61.091366
