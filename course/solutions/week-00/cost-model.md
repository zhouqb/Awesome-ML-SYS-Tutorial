# Week 0 Reference Solution: LLM Inference Cost Model

This is a source-based reference solution for Lab 0. It establishes predictions that should later be tested on a specified model, server configuration, and workload; it does not present unmeasured estimates as benchmark results.

## Initial prediction

Before building the full model, my naive prediction would have been:

1. GPU model execution accounts for nearly all request latency.
2. Model weights account for nearly all persistent GPU memory.
3. Increasing concurrency should increase throughput almost linearly until GPU compute is saturated.

These statements are incomplete. They ignore queueing and CPU work, treat prefill and decode as the same workload, omit KV cache and runtime allocations, and do not specify whether “throughput” describes one request or the whole server.

## Part A: performance vocabulary

| Metric | Definition used here | Unit | Starts at | Ends at |
|---|---|---|---|---|
| Time to first token (TTFT) | Time visible to the caller before the first generated token or text fragment becomes available | ms or s | Request arrival at the serving boundary | First output token/fragment is available to the caller |
| Inter-token latency (ITL) | Time between successive generated output tokens for one request; report a distribution or percentiles rather than only a mean | ms/token | Output token `j - 1` becomes available | Output token `j` becomes available |
| End-to-end latency | Total caller-visible completion time | ms or s/request | Request arrival | Final output or terminal response is available |
| Per-request output throughput | Decode rate for one request, defined here as `(O - 1) / (t_last - t_first)` for `O > 1` | output tokens/s/request | First output token | Last output token |
| Aggregate output throughput | Total output tokens completed by all requests divided by benchmark wall-clock duration | output tokens/s/server | First measured request arrival | Last measured request completion |
| Goodput | Completed requests satisfying a declared service-level objective divided by the observation window | SLO-satisfying requests/s | Start of measurement window | End of measurement window |

Other definitions can be valid, but the benchmark must state them. For example, some tools include the first output token in “per-request tokens/s,” while others approximate it as the inverse of mean ITL. Those numbers should not be compared without normalization.

### Why “the server achieved 120 tokens/s” is uninterpretable

The statement omits at least six necessary qualifiers:

1. Are these input, output, or total tokens?
2. Is this one request’s decode rate or aggregate server throughput?
3. What were prompt and output lengths?
4. What was the request concurrency or arrival pattern?
5. Were TTFT, ITL, or latency SLOs satisfied?
6. Were warmup, prefix-cache hits, and failed requests included?

A server can report higher aggregate tokens/s while each user receives tokens more slowly. It can also obtain an excellent number by processing long prefills even when interactive decode latency is poor. The metric name alone does not distinguish those cases.

## Part B: KV-cache arithmetic

For a conventional MHA/GQA decoder:

```text
KV bytes per request
= layers × sequence tokens × 2 (K and V)
  × KV heads × head dimension × bytes per element
```

Assumptions:

- BF16 uses 2 bytes per element and FP8 uses 1 byte.
- The formula counts both prompt and generated tokens currently retained for the request.
- It describes raw tensor payload only. It excludes allocator/page slack, metadata, alignment, fragmentation, and non-KV runtime memory.
- Results below use binary units: `1 MiB = 2^20 bytes`, `1 GiB = 2^30 bytes`.

### Case A: 32-layer MHA, 4,096 tokens, BF16

```text
32 × 4,096 × 2 × 32 × 128 × 2 bytes
= 2,147,483,648 bytes
= 2,048 MiB
= 2 GiB per request
```

The per-token KV cost is `512 KiB`.

### Case B: 32-layer GQA, 4,096 tokens, BF16

```text
32 × 4,096 × 2 × 8 × 128 × 2 bytes
= 536,870,912 bytes
= 512 MiB
= 0.5 GiB per request
```

The per-token KV cost is `128 KiB`.

### Case C: 32-layer GQA, 32,768 tokens, BF16

```text
32 × 32,768 × 2 × 8 × 128 × 2 bytes
= 4,294,967,296 bytes
= 4,096 MiB
= 4 GiB per request
```

This is eight times Case B because the retained sequence is eight times longer.

### Case D: 32-layer GQA, 32,768 tokens, FP8 KV

```text
32 × 32,768 × 2 × 8 × 128 × 1 byte
= 2,147,483,648 bytes
= 2,048 MiB
= 2 GiB per request
```

This is half of Case C because each KV element uses one byte instead of two.

### Results

| Case | KV bytes/token | KV per request | Relative to A |
|---|---:|---:|---:|
| A: 32 KV heads, 4K, BF16 | 512 KiB | 2 GiB | 1× |
| B: 8 KV heads, 4K, BF16 | 128 KiB | 0.5 GiB | 0.25× |
| C: 8 KV heads, 32K, BF16 | 128 KiB | 4 GiB | 2× |
| D: 8 KV heads, 32K, FP8 | 64 KiB | 2 GiB | 1× |

### Questions

#### 1. What factor does GQA change relative to MHA?

In these examples, GQA reduces the number of stored KV heads from 32 to 8 while the query-head count can remain 32. KV capacity changes linearly with `kv_heads`, so the cache becomes `8 / 32 = 1/4` as large. This arithmetic does not say that GQA is free: model quality, supported kernels, and how query heads share KV heads remain design considerations.

#### 2. What factor does KV FP8 change?

It changes `bytes_per_element` from 2 to 1, halving raw KV payload. A real implementation may add scale metadata and alignment overhead, and numerical behavior must be evaluated; therefore observed savings can be slightly smaller than the ideal 2× ratio.

#### 3. What runtime memory is absent?

The equation omits:

- model weights and any dequantization/packing metadata;
- activations and temporary tensors;
- attention and sampling workspaces;
- CUDA Graph private pools and persistent input/output buffers;
- CUDA context, communication/NCCL buffers, and allocator bookkeeping;
- prefix-cache tree, request/token mappings, and other framework state;
- memory used by other processes on the same GPU.

The repository’s [SGLang OOM analysis](../../../sglang/kvcache-code-walk-through/mem-fraction-static-en.md#what-does-mem_fraction_static-actually-allocate) separates a static region for weights and the KV pool from a dynamic reserve for activations, CUDA Graph buffers, and other inference overhead.

#### 4. Why can page-based allocation exceed live-token arithmetic?

Physical allocation happens in page-sized units. A partially filled final page still occupies a whole page. Internal slack grows when many sequences end at different positions. Reserved-but-not-currently-live pages, alignment, fragmentation, duplicated metadata, cache entries preserved for prefix reuse, and delayed reclamation can widen the difference further.

### Concrete memory budget: representative 8B GQA decoder

Consider a representative configuration with 32 layers, 8 KV heads, head dimension 128, and BF16 KV—the same KV geometry as Case B. Assume:

- approximately 8 billion BF16 parameters: `8B × 2 bytes ≈ 16 GB ≈ 14.9 GiB` raw weights;
- `pre_model_load_memory = 78 GiB` on an nominal 80 GiB GPU after context/other reservations;
- observed weight and load-time allocation of `15 GiB`, making `post_model_load_memory = 63 GiB`;
- `mem_fraction_static = 0.85`.

Following the repository’s SGLang budget equation:

```text
memory_for_kv
= y × pre_model_load_memory
  - (pre_model_load_memory - post_model_load_memory)

= 0.85 × 78 GiB - (78 GiB - 63 GiB)
= 66.3 GiB - 15 GiB
= 51.3 GiB
```

The complementary dynamic reserve is:

```text
(1 - 0.85) × 78 GiB = 11.7 GiB
```

At an idealized `128 KiB/token`, the KV region could hold:

```text
51.3 GiB ÷ 128 KiB/token ≈ 420,250 tokens
```

This implies ideal upper bounds of roughly:

- `51.3 / 0.5 ≈ 102` simultaneous 4,096-token sequences;
- `51.3 / 4 ≈ 12` simultaneous 32,768-token sequences.

These are capacity sanity checks, not safe production limits. Page slack, prefix-cache retention, runtime workspaces, model-specific state, uneven TP-rank memory, and safety margin reduce usable concurrency. Moreover, fitting the cache does not prove that latency or compute throughput is acceptable.

## Part C: latency and throughput model

For request `i` with `O_i` output tokens:

```text
T_request,i
= T_queue
  + T_preprocess
  + T_prefill
  + T_first_sample_and_stream
  + Σ(j=2..O_i) T_inter_token,j
  + T_postprocess
  + T_transport_tail
```

This separates the first-token path from repeated decode. A common approximation, when ITL is stable, is:

```text
T_request ≈ TTFT + (O - 1) × mean_ITL + completion_tail
```

Aggregate server output throughput over a measurement window is:

```text
throughput_output
= Σ_i O_i / (time_of_last_completion - time_of_first_arrival)
```

Goodput under an end-to-end latency SLO `L` is:

```text
goodput_L
= Σ_i 1[T_request,i ≤ L] / measurement_window
```

If the SLO also constrains TTFT or ITL, the indicator must require all those conditions. Reporting aggregate throughput alongside TTFT/ITL distributions prevents a batching policy from appearing better merely because it makes individual users wait longer.

### Bottleneck hypotheses by concurrency

The hypotheses assume a decode-heavy online workload with a warm model. Long-prompt, prefill-heavy traffic can produce a different result.

#### Low concurrency

**Prediction:** GPU execution dominates service time, but the GPU is not fully utilized. Decode repeatedly streams a large fraction of model weights for only one or a few tokens, so HBM bandwidth and fixed launch/CPU overhead are stronger candidates than tensor-core compute. Queueing should be near zero.

**Primary observation:** low queue time, modest batch size, weak compute utilization, and a large share of step time in weight-moving kernels or gaps between kernels.

**Would falsify it:** sustained high compute-pipeline utilization with low memory-bandwidth pressure, or measured preprocessing/network time larger than model execution.

#### Medium concurrency

**Prediction:** continuous batching raises arithmetic intensity and aggregate throughput because one weight read serves more request tokens. GPU execution remains dominant, but fixed overhead is amortized. TTFT and ITL may rise moderately as requests share iterations.

**Primary observation:** larger effective decode batches, higher aggregate output throughput, and better GPU utilization without a large queue.

**Would falsify it:** throughput remains flat while CPU scheduling gaps grow, or queue time dominates before GPU utilization improves. That would indicate a host/control-plane or admission bottleneck rather than insufficient batching.

#### High concurrency

**Prediction:** queueing, KV-cache capacity, and scheduling policy become first-order constraints. Aggregate throughput approaches a plateau while TTFT and tail latency worsen. Very long live sequences can pressure the KV pool and cause delayed admission or decode retraction.

**Primary observation:** growing waiting time and p95/p99 TTFT, near-flat aggregate throughput, falling free KV capacity, or retraction/admission events.

**Would falsify it:** queue time and tail latency remain stable, KV capacity remains ample, and throughput continues scaling. In that case the tested load has not reached the high-concurrency regime; more load or a different workload is required before drawing a saturation conclusion.

### What these hypotheses do not establish

No measurements were run, so the hypotheses do not establish the bottleneck of a particular deployment. The answer depends on model architecture and precision, GPU, attention backend, prompt/output distribution, prefix-cache hit rate, batch policy, tensor-parallel degree, CPU allocation, and network path. The purpose of this model is to decide what to measure, not to replace measurement.

## Corrected model after reading

The corrected model is:

1. A request’s cost is a pipeline, not a single GPU call. Queueing, tokenization, scheduling, sampling, detokenization, and transport can all become visible.
2. Prefill and decode must be modeled separately. Prefill scales mainly with new prompt tokens and often exposes parallel matrix work; decode is an iterative loop with growing cached context and comparatively little new work per request.
3. GPU memory is divided among fixed and dynamic consumers. Weights are largely fixed per loaded replica, while KV cache grows with retained tokens and directly limits active context/concurrency. Activations, workspaces, CUDA Graph pools, communication buffers, and safety margin compete with both.
4. Concurrency trades individual latency for server utilization. Moderate batching can amortize weight reads and overhead; excessive concurrency creates queueing and cache pressure after throughput saturates.
5. A performance claim requires workload and metric semantics. At minimum, report model/version, hardware, input/output lengths, concurrency, aggregate output throughput, and TTFT/ITL distributions.

## Largest change in understanding

The largest correction is that “the GPU is fast or slow” is not a useful model of serving performance. The scheduler converts irregular requests into a sequence of changing batches, and that decision couples compute efficiency, queueing latency, and KV-memory capacity. More concurrency can improve aggregate throughput by amortizing model-weight traffic while simultaneously worsening each user’s TTFT or ITL; more static memory can admit more cached tokens while leaving too little dynamic headroom. The right question is therefore not “How many tokens per second does the model achieve?” but “For this request distribution and SLO, which resource constrains goodput, and what measurement would disprove that diagnosis?”

## Week 0 readiness check

Before continuing to Week 1, I should be able to explain without notes:

- the difference between TTFT, ITL, per-request throughput, aggregate throughput, and goodput;
- why prefill and decode have different cost shapes;
- why GQA and KV precision change cache capacity linearly;
- why raw KV arithmetic is below observed GPU memory use;
- why throughput can rise while user-visible latency worsens.

The next empirical step is to select an actual model/hardware environment, record it using the workbook template, and test the three concurrency hypotheses rather than treating them as conclusions.
