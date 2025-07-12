import torch
import triton
import triton.language as tl

class TritonAttention(torch.autograd.Function):
    pass

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale): 
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V  # Make sure head dimension matches for Q, K and V

        O = torch.empty_like(Q) # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), not the Q, K and V of the input to the attention, but its the output
                                # of the W_Q, W_V, W_K because flash attention is not concerned with optimising these matrices.

        stage = 3 if causal else 1

        # Number of parallel programs = BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), device=Q.device, dtype=torch.float32,
        )

        __attn__fwd[grid](
            Q = Q,
            K = K,
            V = V,
            softmax_scale = softmax_scale,
            M = M,
            stride_Q_batch = Q.stride(0),
            stride_Q_head = Q.stride(1),
            stride_Q_seq = Q.stride(2),
            stride_Q_dim = Q.stide(3),
            stride_K_batch = K.stride(0),
            stride_K_head = K.stride(1),
            stride_K_seq = K.stride(2),
            stride_K_dim = K.stide(3),
            stride_V_batch = V.stride(0),
            stride_V_head = V.stride(1),
            stride_V_seq = V.stride(2),
            stride_V_dim = V.stide(3),
            NUM_HEADS = NUM_HEADS,
            SEQ_LEN = SEQ_LEN,
            HEAD_DIM=HEAD_DIM_Q,
            stage=stage,
        )

        


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
    )
    
    K = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
    )

    V = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
    )

    scaling_factor = 1 / (HEAD_DIM**0.5) # QK^T/sqrt(HEAD_DIM)

    # Reference (Vanilla) implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2,3)) * scaling_factor # --> # SEQ_LEN × SEQ_LEN attention pattern
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half() # Convert P first to fp32 for numerical stability and then convert back to half precision to maintain consistency with
                                                # other tensors.

    ref_0 = torch.matmul(P, V) # --> Brings back shape to (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
   
    # Triton attention
    tri_out = TritonAttention.apply(Q, K, V, causal, scaling_factor).half()

    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_0, tri_out, atol=atol, rtol=rtol)