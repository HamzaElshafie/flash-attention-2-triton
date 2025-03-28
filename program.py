import torch
import triton
import triton.language as tl

class TritonAttention(torch.autograd.Function):
    pass

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale): 
        pass

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    
    K = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    V = (
        torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    scaling_factor = 1 / (HEAD_DIM**0.5) # QK^T/sqrt(HEAD_DIM)
    d0 = torch.randn_like(Q) # Needed for the backward pass

    # Reference (Vanilla) implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2,3)) * scaling_factor
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half() # Convert P first to fp32 for numerical stability and then convert back to half precision to maintain consistency with
                                                # other tensors.

    ref_0 = torch.matmul(P, V)
    ref_0.backward(d0)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # Triton attention
    tri_out = TritonAttention.apply(Q, K, V, causal, scaling_factor).half()
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_0, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)