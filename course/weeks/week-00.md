# Week 0 — The Cost Model of LLM Inference

Reading time: 35–45 minutes. This lesson is the foundation for every later optimization. The question is not “Is the model fast?” but “Where do time and memory go for this workload?”

## Learning objectives

After this lesson, you should be able to:

- separate prefill from decode;
- define TTFT, ITL, throughput, and goodput;
- estimate model-weight and KV-cache memory;
- explain why batching can improve server throughput while hurting a user’s latency;
- turn a performance guess into a falsifiable hypothesis.

## 1. The minimum serving loop

A model maps token IDs to logits. A serving engine must surround that operation with request admission, tokenization, scheduling, persistent attention state, sampling, streaming, and cleanup.

```text
request → tokenize → schedule → model forward → sample token
                         ↑                        ↓
                         └──── update state ←─────┘
                                      ↓
                               detokenize/stream
```

The loop repeats because an autoregressive model normally produces one new token per request per decode step. New requests can arrive while old requests are still generating, so the set of requests in the next model invocation changes over time.

SGLang’s scheduler can initially be understood as four verbs:

1. **Receive** new requests.
2. **Select** work that fits the policy and available KV memory.
3. **Run** a model batch.
4. **Update** tokens, stopping state, queues, and cache ownership.

That simple loop will later expand into `TokenizerManager`, `Scheduler`, `TpModelWorker`, `ModelRunner`, attention backends, and the detokenizer.

## 2. Prefill and decode are different workloads

Suppose a prompt contains `P` tokens and the model generates `O` tokens.

During **prefill**, all `P` prompt positions are processed. Large matrix multiplications see many rows at once, exposing substantial parallel work. Prefill creates the key/value state needed by future attention. Its caller-visible effect is a large part of time to first token.

During **decode**, the engine normally advances each active request by one token per iteration. Earlier keys and values are reused, but the model weights and growing attention state must still be accessed. For small batches, there may be too little arithmetic per byte moved, so decode is often limited by memory bandwidth or fixed launch/CPU overhead rather than peak tensor-core FLOPS.

This gives the first useful approximation:

```text
request latency
≈ queue time
  + preprocessing
  + prefill
  + first-token sampling/streaming
  + (output_tokens - 1) × average ITL
  + completion/transport tail
```

Do not universalize “prefill is compute-bound and decode is bandwidth-bound.” It is a starting hypothesis. Model architecture, batch size, sequence length, quantization, kernels, and hardware can change the result.

## 3. The metrics answer different questions

### Time to first token

**TTFT** is the caller-visible time from request arrival until the first generated token or text fragment becomes available. It includes queueing and prompt processing, not merely a GPU kernel.

```text
TTFT = queue + preprocess + prefill + first sample + first stream
```

Long prompts, busy queues, and prefill scheduling policy strongly affect TTFT.

### Inter-token latency

**ITL** is the time between consecutive generated tokens for one request. Report a distribution or percentiles, because an average hides stalls.

For output token timestamps `t1, t2, …, tO`:

```text
ITL_j = t_j - t_(j-1), for j = 2…O
```

The inverse of mean ITL roughly describes one request’s steady decode rate.

### Aggregate throughput

Server output throughput asks how much total output the deployment completes:

```text
aggregate output throughput
= sum(output tokens across requests) / benchmark wall time
```

It can rise even when individual ITL worsens, because batching lets one model-weight read serve more request tokens.

### Goodput

**Goodput** counts only work that meets a declared service-level objective (SLO):

```text
goodput
= count(requests satisfying TTFT/ITL/E2E SLO) / observation time
```

Goodput prevents a system from claiming victory by accumulating a huge batch and making every interactive user wait.

### Why “120 tokens/s” says almost nothing

You still need to know whether tokens are input or output, per-request or aggregate, what concurrency and lengths were used, whether prefix-cache hits were allowed, whether warmup was excluded, and whether latency SLOs were met.

## 4. Weight memory and KV-cache memory

Weights are mostly fixed after a model replica loads:

```text
weight bytes ≈ parameter count × bytes per stored parameter
```

An 8-billion-parameter model stored entirely in BF16 has a raw payload near `16 GB`, or about `14.9 GiB`. Runtime footprint can differ due to packing, scales, tied parameters, allocator behavior, and auxiliary tensors.

KV cache grows with retained sequence tokens. For a conventional MHA/GQA decoder:

```text
KV bytes
= layers × retained tokens × 2(K,V)
  × KV heads × head dimension × bytes per element
```

Worked example: 32 layers, 8 KV heads, head dimension 128, BF16.

```text
bytes per token
= 32 × 2 × 8 × 128 × 2
= 131,072 bytes
= 128 KiB/token

4,096-token request = 512 MiB
32,768-token request = 4 GiB
```

Two linear levers are now obvious:

- reducing KV heads from 32 to 8 reduces raw KV storage to one quarter;
- changing KV from BF16 to FP8 halves raw KV storage.

Neither is “free”: architecture, quality, scale metadata, kernel support, and alignment still matter.

## 5. Why observed memory exceeds the formula

The KV equation counts payload, not the entire process. GPU memory also holds:

- activations and temporary tensors;
- attention and sampling workspaces;
- CUDA Graph private pools and persistent buffers;
- CUDA context and NCCL/communication buffers;
- page slack, allocator metadata, and fragmentation;
- prefix-cache entries retained after requests finish;
- other processes’ allocations.

SGLang’s `mem_fraction_static = y` divides available pre-load memory conceptually into a static region for weights plus KV and a dynamic reserve. Using the notation in the repository’s OOM analysis:

```text
memory_for_kv
= y × pre_model_load_memory
  - (pre_model_load_memory - post_model_load_memory)
```

The parenthesized term approximates memory consumed by model loading and related allocation. Raising `y` can rescue startup when weights leave no room for a KV pool; lowering `y` can rescue runtime when activations or dynamic workspaces lack headroom. “OOM” is a symptom, not a diagnosis.

## 6. Concurrency changes arithmetic intensity and queueing

At low concurrency, a decode step may process only one or two tokens while reading large weights. Latency is low because there is little queueing, but hardware utilization can be poor.

At moderate concurrency, continuous batching processes more tokens per model invocation. Fixed overhead and weight traffic are amortized, so aggregate throughput improves. Each request may wait slightly longer between turns.

At excessive concurrency, throughput approaches a resource ceiling while queueing and tail latency grow. KV capacity can delay admission or force requests to be retracted. The useful curve is therefore not throughput alone; it is goodput versus offered load under explicit latency constraints.

## 7. Three hypotheses to test later

For a warm, decode-heavy workload:

1. **Low concurrency:** decode is limited by weight bandwidth or fixed launch/CPU overhead. Falsify this with high compute utilization and low bandwidth pressure.
2. **Medium concurrency:** batching raises aggregate throughput faster than it worsens latency. Falsify this if throughput stays flat while scheduler gaps or queue time grow.
3. **High concurrency:** queueing and KV capacity dominate tail behavior. Falsify this if free KV remains high, queue time stays near zero, and throughput continues scaling.

These are predictions, not findings. Your hardware/model experiment in Week 10 will determine which is true.

## 8. Source connection

The first source-level memory equation appears in SGLang’s pinned [`_profile_available_bytes`](https://github.com/sgl-project/sglang/blob/4fa3482/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L56-L74). It measures post-load availability and subtracts the dynamic-memory reserve. Read it only after the conceptual equation above is clear.

The scheduling loop you will later trace begins at pinned [`Scheduler.event_loop_normal`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L1520). For now, look only for the receive/select/run/update rhythm; do not follow every branch.

## Checkpoint

You are ready for Week 1 if you can answer without notes:

1. Why is TTFT not the same as prefill kernel time?
2. Why can server throughput improve while per-request ITL worsens?
3. How many KV bytes does one token require for 32 layers, 8 KV heads, dimension 128, and BF16?
4. Why is that KV calculation below real GPU memory consumption?
5. What measurement would falsify the claim that low-concurrency decode is bandwidth-bound?

## Go deeper

- [SGLang Before the Scheduler](../SGLANG_PRIMER.md) provides a slower introduction to the serving loop.
- [Completed Week 0 cost-model solution](../solutions/week-00/cost-model.md) contains all four KV arithmetic cases and a concrete 80 GiB memory budget.
- [When SGLang OOMs](../../sglang/kvcache-code-walk-through/mem-fraction-static-en.md) develops the static/dynamic memory split from a production failure.

