# Week 6 — CUDA Graphs and Overlap Scheduling

Reading time: 45–55 minutes. Once kernels are fast, CPU scheduling, kernel launches, allocation, and synchronization can leave visible gaps. CUDA Graphs remove repeated launch work; overlap scheduling hides CPU work behind GPU work. Both require stricter correctness contracts.

## Learning objectives

After this lesson, you should be able to:

- explain CUDA Graph capture and replay;
- identify pointer, shape, control-flow, RNG, and synchronization hazards;
- distinguish persistent buffers from replacement tensors;
- draw normal versus overlapped CPU/GPU timelines;
- distinguish work eliminated from work hidden.

## 1. The launch-overhead problem

A decode iteration may launch many small kernels. In ordinary eager execution, Python/framework code dispatches each kernel separately. When kernels are short, gaps and CPU launch cost occupy a meaningful share of the step.

CUDA Graph capture records a sequence of GPU operations and their dependencies once. Replay launches the recorded graph with much lower CPU overhead.

```text
Eager:  CPU launch K1 → launch K2 → launch K3 → ...
Graph:  CPU graph replay ───────────────────────→
GPU:             K1 → K2 → K3 → ...
```

The graph does not make the kernels’ arithmetic disappear. It makes repeated submission cheaper and can preserve an optimized dependency schedule.

## 2. Why replay imposes constraints

Captured kernels refer to memory addresses and a recorded control/dependency structure. Safe replay therefore needs invariants.

### Stable addresses

Replacing a tensor object may allocate a new address that captured kernels do not reference. Updating values in a persistent tensor with `copy_`, `fill_`, or `zero_` preserves its address.

```python
# graph-friendly pattern
persistent_input.copy_(new_values)
replay_graph()

# dangerous if the graph captured the old address
persistent_input = new_values.clone()
replay_graph()
```

### Compatible shapes

The captured kernels encode launch dimensions and buffer sizes. Serving engines often capture graphs for a set of batch-size buckets and pad/select the nearest compatible graph.

### Static control flow

Python branches are resolved during capture. Changing a Python flag later does not insert new kernels into an already-recorded graph.

### Allocation discipline

Dynamic allocation inside capture/replay can violate address assumptions or use graph-private pools unexpectedly. Required buffers should normally be allocated before capture.

### Host-device synchronization

Operations that force the CPU to wait for a GPU scalar or copy break asynchronous pipelines. Examples include `.item()` in a hot path or data-dependent Python branching on a CUDA tensor.

### Randomness

Random sampling needs replay-safe RNG-state handling. Greedy `argmax` is easier because it is deterministic and does not advance random state.

## 3. A production case: Dual AR TTS

The repository’s CUDA Graph article studies an S2-Pro model with a slow autoregressive head and an inner fast codebook loop. Capturing only the slow head leaves launch overhead in the inner loop. The optimization moves codebook decoding into the model forward and captures both as one graph.

The initialization order is essential:

```text
temporarily disable graph capture
→ initialize worker/model
→ apply precision fixes
→ allocate fast-head KV/cache buffers
→ attach decoder and persistent buffers
→ capture the complete forward path
```

If capture happens before the decoder is attached, the recorded graph omits that branch forever. Setting a flag afterward cannot modify the graph.

Pinned case-study anchors:

- [`create_s2pro_sglang_engine`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/factory.py)
- [`setup_vq_decode`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/sglang_model.py#L196)
- [`S2ProSGLangModelRunner`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py)

## 4. Persistent-buffer protocol

Dynamic request values can still change between replays if they are copied into captured addresses:

```text
outside graph: copy current request values into persistent input buffers
inside graph:  kernels read/write those captured addresses
outside graph: consume persistent output buffers after completion
```

This protocol separates **dynamic values** from **static addresses**. Correctness requires synchronization: the CPU must not overwrite an input buffer before the graph has consumed it or read an output before the graph has produced it.

## 5. Normal scheduler timeline

Without overlap:

```text
time ─────────────────────────────────────────────→
CPU: schedule n | wait/launch | process n | schedule n+1
GPU:            | execute n  |           | execute n+1
```

CPU scheduling and result processing can create bubbles between GPU steps.

## 6. Overlap scheduling

Overlap attempts:

```text
time ─────────────────────────────────────────────→
CPU: schedule n | prepare/process n+1 | prepare n+2
GPU:            | execute n           | execute n+1
```

The CPU uses GPU execution time to prepare future work. This does not remove scheduling work; it hides it behind a longer GPU critical path when dependencies allow.

The difficulty is that the output token from step `n` is normally an input to step `n+1`. An overlap design can use placeholder/future indices and GPU-resident result paths so the CPU does not immediately synchronize just to obtain token IDs.

Synchronization is deferred until a consumer truly needs host-visible values, such as streaming or stop-condition handling that cannot remain on device.

## 7. Multi-step versus overlap

**Multi-step scheduling** may schedule several decode steps together under an assumption that batch membership/control state remains valid. It amortizes scheduling but delays reacting to stop tokens or new requests.

**Overlap scheduling** pipelines consecutive steps and preserves more frequent scheduling decisions, but it must manage futures and cross-step dependencies safely.

Both trade control responsiveness and implementation complexity for lower exposed CPU overhead.

## 8. Graph-safety audit method

For each hot-path source line, classify:

1. Does it allocate?
2. Does it replace a captured address?
3. Can shape or launch dimensions change?
4. Is control flow data-dependent?
5. Does it synchronize host and device?
6. Does it use mutable RNG state?

Then assign one verdict:

- safe as written;
- safe under a stated invariant;
- must execute outside capture;
- incompatible with this graph design.

The invariant is part of the implementation, not a footnote. For example, “batch size belongs to captured bucket 8 and buffers are preallocated for 8” is a testable condition.

## 9. Measurement design

Compare graph disabled/enabled only after warmup. Use at least two batch sizes because launch overhead’s fraction changes with GPU work. Report:

- median and p95 step/ITL;
- aggregate output throughput;
- GPU timeline gaps;
- memory change from graph pools;
- output-equivalence guardrail under fixed sampling.

If kernels dominate, graphs may provide little benefit. If CPU gaps are large, benefit can be substantial. The profiler timeline should explain the result.

## Checkpoint

Explain without notes:

1. Why graph replay needs stable addresses but can accept new values.
2. Why a late Python flag change cannot alter captured operations.
3. What `.item()` can do to an asynchronous pipeline.
4. The difference between removing CPU work and hiding it.
5. Why output token dependencies complicate overlap scheduling.

## Go deeper

- [Revisiting CUDA Graph](../../torch/cuda-graph/readme-2-en.md).
- [SGLang scheduler: overlap scheduling](../../sglang/scheduler/readme-en.md#overlap-scheduler-hiding-scheduling-overhead-behind-operators).
- [Commit-pinned CUDA Graph case study](../CODE_MAP.md#cuda-graph-case-study).
- [Lab 6 graph and overlap audit](../LABS.md#lab-6--cuda-graph-and-overlap-audit).

