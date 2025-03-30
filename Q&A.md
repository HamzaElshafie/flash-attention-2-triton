# Q&A 

```python
grid = lambda args: (
    triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
    BATCH_SIZE * NUM_HEADS,
    1,
)

**Q: Can we replace BLOCK_SIZE_Q with BLOCK_SIZE_K in the grid and still mean the same thing?**

A: Not quite. While both BLOCK_SIZE_Q and BLOCK_SIZE_K are used in the attention computation, only BLOCK_SIZE_Q is relevant for the grid shape. The grid's X-dimension controls how many queries each kernel processes. Even though the attention calculation involves K^T, we are still tiling over blocks of queries when defining the grid.

The role of BLOCK_SIZE_K typically comes into play inside the kernel, where we loop over key blocks when calculating the dot product. But in terms of the grid — which defines how we divide the work across kernels — it’s BLOCK_SIZE_Q that matters, since each kernel is responsible for processing a chunk of the query sequence.

**Q: In the notes, each block had 2 queries — so does that mean BLOCK_SIZE_Q = 2 or BLOCK_SIZE_Q = 4?**

A: It depends on how many query rows each block contains. In the diagrams, Q had a total of 8 rows (query vectors), and each block was shown as containing 4 queries. That means BLOCK_SIZE_Q = 4 in that case. If each block had only 2 rows, then BLOCK_SIZE_Q = 2.

So the value of BLOCK_SIZE_Q directly reflects how many queries are grouped into one block that is processed by a single kernel instance.

