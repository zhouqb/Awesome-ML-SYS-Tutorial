# Week 5 — Attention Backends, IO, and Data Layout

Reading time: 45–60 minutes. Scheduling decides what runs; the attention backend decides how model-level attention becomes kernels over concrete layouts and cache indices.

## Learning objectives

After this lesson, you should be able to:

- explain why prefill/extend and decode use different attention paths;
- reason about attention’s compute and memory traffic;
- understand shape, stride, tiling, and coalesced access at a practical level;
- describe the contract between the model layer, KV pool, metadata builder, and kernel;
- classify optimizations by the resource they target.

## 1. Attention computation and the KV boundary

For one head, scaled dot-product attention is:

```text
S = Q Kᵀ / √d
P = softmax(S)
O = P V
```

In prefill with sequence length `S`, the score matrix has `S × S` structure (modulo causal masking and optimized algorithms). In decode, a new query of length one attends over cached length `T`, producing a `1 × T` score vector per head/request.

The mathematical operation is related, but the useful parallelism and data movement differ dramatically.

## 2. Why prefill and decode select different kernels

### Prefill/extend

- many query positions are processed;
- matrix operations have larger dimensions and more reuse;
- ragged requests can have different prefix and extension lengths;
- new K/V for multiple positions must be written;
- FlashAttention-style tiling avoids materializing the full score matrix in HBM.

### Decode

- usually one new query position per request;
- all prior K/V must be read through paged/cache indirection;
- batch size supplies parallelism across requests;
- metadata and launch overhead can be significant;
- memory bandwidth and cache layout often dominate.

Using one generic kernel for both leaves performance on the table. The backend chooses wrappers/kernels based on forward mode, batch geometry, architecture, and enabled features.

## 3. FlashAttention’s IO idea

Naive attention conceptually writes the `S × S` score/probability matrix to HBM, then reads it back to multiply by V. FlashAttention tiles Q/K/V so blocks of scores and the online softmax state remain in faster on-chip memory.

The exact FLOP count is not eliminated; the important saving is HBM traffic and intermediate storage. A useful mental model is:

```text
performance ≈ useful arithmetic / bytes moved through the bottleneck memory level
```

Tiling increases reuse: one loaded Q/K/V tile participates in many operations before eviction.

## 4. Shape and stride

A tensor’s shape describes logical coordinates; stride maps those coordinates to linear memory offsets.

For shape `[M, N]` with row-major contiguous storage:

```text
offset(i, j) = i × N + j
stride = [N, 1]
```

Transposing the logical view may change stride to `[1, N]` without moving data. Kernels care because neighboring GPU threads should often access neighboring addresses to coalesce transactions.

Layout is therefore not cosmetic. A mathematically correct view with poor stride can multiply memory transactions or require an explicit transpose.

## 5. Tiling and arithmetic intensity

Assume a matrix multiplication tile computes `C_tile = A_tile × B_tile`. Loading a larger reusable tile can increase operations per byte, but it consumes shared memory/registers and may reduce occupancy. Tile design balances:

- global-memory coalescing;
- shared-memory bank behavior;
- register pressure;
- tensor-core-compatible shapes;
- occupancy and number of resident warps;
- boundary/ragged handling.

The repository’s [data-layout note](../../kernels/data_layout.md) develops shape/stride, tile layout functions, replication, and swizzle. For this week, master shape/stride and the reason for tiling; TMEM and swizzle are stretch topics.

## 6. The attention backend contract

The model layer should express semantic attention; the backend should prepare and execute the efficient implementation.

Inputs typically include:

- Q/K/V tensors for new positions;
- forward mode (extend/decode);
- sequence lengths and prefix lengths;
- request-to-token/cache indices;
- positions and causal/sliding-window settings;
- architecture-specific dimensions.

Persistent state can include workspace buffers, wrapper objects, graph-compatible metadata buffers, and references to the KV pool.

Outputs are attention results shaped for the model layer, plus side effects that write new K/V to cache.

## 7. Metadata is part of performance

Paged attention needs index arrays describing where each request’s cached tokens live. Building those arrays on CPU every step can become visible when GPU kernels are short. A backend therefore separates:

1. metadata that can persist or be captured;
2. values updated per batch;
3. work performed outside versus inside a CUDA Graph;
4. extend-specific and decode-specific wrappers.

Read pinned [`FlashInferAttnBackend.init_forward_metadata`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/flashinfer_backend.py#L916), then compare [`forward_extend`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/flashinfer_backend.py#L1252) with [`forward_decode`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/flashinfer_backend.py#L1410).

For each path, record:

- which new K/V is written;
- how cached K/V is addressed;
- which wrapper/kernel is selected;
- the expected output shape;
- which metadata changes next iteration.

## 8. Worked resource classification

- **FlashAttention tiling:** primarily reduces HBM traffic/intermediate materialization; secondarily changes compute scheduling.
- **Paged KV storage:** primarily improves capacity/allocation flexibility; adds indexed-access complexity.
- **Fused RoPE plus cache write:** removes intermediate traffic and kernel launches.
- **FP8 KV cache:** reduces capacity and bandwidth per cached token; adds conversion/scaling requirements.
- **Precomputed metadata:** reduces CPU and launch-path overhead; may require static buffers.
- **Prefix reuse:** eliminates prefill compute and new KV writes for matched tokens.
- **Tensor-parallel attention:** distributes capacity/compute; introduces communication and possible KV duplication.

This classification prevents vague claims like “kernel optimization.” Always name the resource whose demand is reduced or better utilized.

## 9. A roofline-style intuition

Arithmetic intensity is approximately:

```text
AI = operations / bytes moved
```

Achievable performance is bounded by:

```text
min(peak compute, memory bandwidth × AI)
```

Large prefill tiles can raise AI and approach compute limits. Single-request decode has low reuse of weights/KV per step, so the bandwidth roof may be lower. Batching raises reuse and effective AI, explaining why aggregate decode throughput improves with concurrency until another bottleneck takes over.

## Checkpoint

Explain without notes:

1. Why attention math is similar but prefill/decode kernels differ.
2. Why FlashAttention is primarily an IO optimization.
3. How stride differs from shape.
4. What metadata paged decode needs.
5. Which resource each of paged KV, fused cache write, and FP8 KV targets.

## Go deeper

- [SGLang Code Walk Through: AttentionBackend](../../sglang/code-walk-through/readme.md#attentionbackend-accelerate-model-forward).
- [Data layout and notation](../../kernels/data_layout.md).
- [Flash Attention notes](../../transformers/flash_attn/readme.md) focus on backward algebra; useful for math, not as the serving-kernel explanation.
- [Lab 5 attention contract](../LABS.md#lab-5--attention-backend-contract).

