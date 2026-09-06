# Week 10 — Profiling and the Capstone

## Why this week matters

Inference optimization is not a collection of tricks. It is a disciplined loop:

```text
observe a user-visible symptom
        ↓
form a falsifiable systems hypothesis
        ↓
measure the suspected resource or code path
        ↓
change one thing
        ↓
measure again under the same workload
```

The capstone is to apply that loop to the repository and produce an evidence-backed explanation or improvement. A small, well-measured finding is more valuable than a large patch supported only by intuition.

## Learning objectives

By the end of this week, you should be able to:

1. Translate latency or throughput symptoms into testable hypotheses.
2. Design a benchmark that isolates one serving mechanism.
3. distinguish queueing delay, CPU orchestration, communication, and GPU execution.
4. Read a timeline without confusing correlation for causation.
5. Produce a reproducible capstone report tied to concrete source code.

## 1. Start from the metric the user feels

The main serving metrics answer different questions:

- **TTFT**: how long until useful output begins?
- **Inter-token latency (ITL)**: how smooth is generation after it begins?
- **End-to-end latency**: how long until the request finishes?
- **Throughput**: how many tokens or requests finish per unit time?
- **Goodput**: how much work completes while meeting a service-level objective?

A throughput improvement can make tail TTFT worse. An ITL improvement can reduce throughput if it reserves too much GPU capacity for decode. Always state the primary metric and the guardrail metrics before running an experiment.

## 2. Decompose request time

A useful first model is:

```text
request latency
  = queueing
  + scheduling and preprocessing
  + model execution
  + sampling and postprocessing
  + streaming or network delay
```

Model execution can be decomposed again:

```text
model execution
  = compute
  + memory traffic
  + communication
  + launch and synchronization overhead
```

This hierarchy prevents a common mistake: optimizing a GPU kernel when requests are actually waiting in a queue or on CPU-side batch preparation.

## 3. Queueing changes abruptly near saturation

Little's Law relates average concurrency \(L\), arrival rate \(\lambda\), and average time in the system \(W\):

```text
L = lambda * W
```

It does not predict the entire latency distribution, but it is an excellent sanity check. If arrival rate stays constant while average latency doubles, average in-system concurrency should also roughly double.

As utilization approaches capacity, queueing delay often rises nonlinearly. A benchmark at a single, comfortably low request rate may hide the most important production behavior. Sweep offered load through and beyond the saturation knee.

## 4. Build a benchmark matrix

At minimum, control these dimensions:

- input length;
- requested output length;
- request arrival pattern;
- concurrency or request rate;
- model and precision;
- scheduler configuration;
- cache warmth and prefix reuse;
- hardware and software revision.

Do not vary every dimension in one giant experiment. Start with a small baseline matrix, then expand the dimension connected to your hypothesis.

### A practical baseline

```text
prompt lengths:       128, 2048, 8192
output lengths:       32, 256
concurrency:          1, 8, 32
arrival modes:        closed-loop and fixed-rate
repetitions:          enough for stable p50 and p99
```

Record both aggregate metrics and per-request traces. Aggregates tell you whether something changed; traces help explain why.

## 5. Warmup is part of the experiment

Early iterations may include:

- model loading;
- memory-pool creation;
- kernel autotuning or JIT compilation;
- CUDA graph capture;
- allocator growth;
- prefix-cache population.

Separate cold-start results from steady-state results. Do not silently discard warmup; state exactly which requests were excluded and why.

## 6. Use the cheapest tool that can falsify the hypothesis

You do not need a GPU trace for every question.

### Application metrics and logs

Use these first for queue length, batch composition, cache hit rate, token budgets, request states, and scheduler decisions.

### CPU profiler

Use it when batch construction, tokenization, grammar processing, serialization, or Python overhead is suspected.

### Framework profiler

Use it to connect model operations to CPU and CUDA activity, tensor shapes, and memory allocations.

### System timeline

Use Nsight Systems or an equivalent timeline when you need to see kernel launches, CPU gaps, streams, collectives, and synchronization.

### Kernel profiler

Use Nsight Compute or focused microbenchmarks only after identifying a kernel that materially affects the end-to-end metric.

## 7. How to read a timeline

Ask these questions in order:

1. Is the GPU busy during the interval that matters?
2. Are there unexplained gaps between kernels?
3. Which CPU thread launches the work, and is it late?
4. Are compute, copies, and communication overlapping?
5. Does a synchronization point serialize otherwise independent work?
6. Are kernel shapes stable or changing across iterations?

A long kernel is not automatically the optimization target. It may be doing most of the useful work efficiently. An apparently small CPU gap repeated between every decode token can dominate interactive latency.

## 8. Add instrumentation with a question in mind

Avoid logging everything. Instrument the exact boundary needed to test the hypothesis:

```python
record(
    request_id=request.id,
    event="scheduled",
    timestamp=now(),
    batch_tokens=batch.num_tokens,
    running_requests=len(batch.requests),
    free_kv_tokens=kv_pool.available_size(),
)
```

For a scheduling study, useful events might include:

- request arrival;
- admission to a batch;
- prefill start and end;
- each decode batch membership;
- retraction or preemption;
- completion;
- KV allocation and release.

Use stable request identifiers so events can be joined into one trajectory.

## 9. Capstone options

Choose one mechanism and one primary claim.

### Option A: Explain a performance cliff

Find a configuration where p99 latency or throughput changes sharply. Trace the cliff to queueing, KV pressure, batch shape, communication, or a kernel transition.

### Option B: Reproduce a serving technique

Reproduce a claim about chunked prefill, prefix caching, CUDA graphs, quantization, or speculative decoding under a documented workload.

### Option C: Make a small optimization

Implement a narrowly scoped improvement, such as eliminating repeated host work, caching metadata, reducing a synchronization, or improving an admission heuristic. Include a correctness test and an end-to-end benchmark.

### Option D: Build an observability tool

Create a trace or visualization that makes request state, batch composition, and KV occupancy understandable over time. Use it to explain at least one real behavior.

## 10. Example capstone: chunked-prefill fairness

### Question

Does chunking long prefills improve decode ITL for short interactive requests without unacceptable throughput loss?

### Hypothesis

Without chunking, a long prefill monopolizes a batch and creates an ITL spike. With a moderate chunk size, decode work can interleave, reducing tail ITL. Very small chunks add scheduling overhead and reduce throughput.

### Independent variable

Prefill chunk size.

### Controlled variables

Model, hardware, prompt/output distribution, arrival trace, precision, scheduler settings, and random seed.

### Primary metric and guardrails

- primary: p99 ITL of short requests;
- guardrails: total throughput, long-request completion time, and GPU utilization.

### Evidence to collect

- per-request latency trace;
- batch token counts;
- prefill/decode batch composition;
- GPU timeline for representative runs;
- exact commit and configuration.

### Falsification

The hypothesis is wrong if chunking changes neither batch interleaving nor tail ITL, or if the apparent improvement disappears under repeated controlled runs.

## 11. Read the repository in this order

1. Start with the benchmark and profiling utilities to understand how requests are generated and measured.
2. Locate the metric emission points in the scheduler and server.
3. Trace your chosen configuration from command-line argument to runtime behavior.
4. Identify the exact state transition or kernel boundary connected to your hypothesis.
5. Read tests to understand intended behavior before changing code.

Repository entry points:

- [Course code map](../CODE_MAP.md)
- [Labs and capstone rubric](../LABS.md)
- [SGLang scheduler evolution](../../sglang/scheduler-evolution/SGLang Scheduler Evolution.md)
- [SGLang source tree](../../sglang/)
- [Repository root](../../)

Useful source-search terms:

```text
benchmark, profiler, metrics, ttft, itl, queue, cuda graph,
chunked prefill, retract, available_size, cache hit, trace
```

## 12. The capstone report

Your final report should contain:

1. **Question** — one sentence.
2. **Why it matters** — connect it to a user-visible metric.
3. **System model** — the relevant request, scheduler, memory, and execution path.
4. **Hypothesis** — specific and falsifiable.
5. **Method** — hardware, software revision, workload, configuration, and repetitions.
6. **Results** — distributions and guardrail metrics, not only averages.
7. **Evidence** — traces, profiles, logs, and source links.
8. **Interpretation** — why the evidence supports the claim.
9. **Limitations** — what the study does not establish.
10. **Next experiment** — the most valuable unresolved question.

## 13. Reproducibility checklist

- Record the exact git commit.
- Preserve the full launch command and environment.
- State hardware model, GPU count, and topology.
- Save raw results, not just charts.
- Separate warmup from measurement.
- Repeat runs and report variance or confidence intervals.
- Keep correctness checks enabled.
- Compare against a clearly named baseline.
- Explain excluded data.
- Make the claim no broader than the evidence.

## Checkpoint

You have completed the course when you can:

1. Draw one request from arrival through scheduling, KV allocation, model execution, sampling, and streaming.
2. Explain which resource limits prefill and decode in your chosen workload.
3. Defend a benchmark design against confounders.
4. Connect a user-visible metric change to source code and measured runtime evidence.
5. State what would falsify your capstone conclusion.

## Optional references

- [SGLang documentation](https://docs.sglang.ai/)
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)
- [Little's Law](https://en.wikipedia.org/wiki/Little%27s_law)
