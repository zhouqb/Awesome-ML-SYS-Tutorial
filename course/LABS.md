# LLM Inference Systems Lab Workbook

This workbook is deliberately evidence-first. Keep your answers in a separate copy of each template so your predictions remain visible after you inspect the source or run the experiment.

## Working conventions

Create one folder outside the course materials for your outputs:

```text
inference-course-work/
  environment.md
  week-00/
  week-01/
  ...
  capstone/
```

Record the following in `environment.md` before any measurement:

- date and Git commit;
- installation method and container/package version;
- GPU model/count and driver/CUDA version, if applicable;
- CPU model/count, memory, and relevant topology;
- model repository and exact revision;
- command-line arguments and environment variables;
- benchmark command, prompt distribution, input/output lengths, concurrency, and number of requests.

Never compare two results if one of those fields changed unintentionally.

## Standard experiment protocol

Use this protocol whenever a lab asks for measurements:

1. State one falsifiable hypothesis.
2. Select one primary metric and no more than three guardrail metrics.
3. Hold model, prompt set, output length, sampling configuration, and software commit constant.
4. Separate cold start and warmup from steady state.
5. Run enough repetitions to report variation, not only the best result.
6. Save raw output before summarizing it.
7. Report negative and unexpected results.
8. End with “what this evidence does not show.”

## Lab 0 — Cost model before code

### Part A: vocabulary

Define each metric in your own words and state its unit:

| Metric | Definition | Unit | Starts at | Ends at |
|---|---|---|---|---|
| TTFT | | | | |
| ITL | | | | |
| End-to-end latency | | | | |
| Per-request output throughput | | | | |
| Aggregate output throughput | | | | |
| Goodput | | | | |

Then explain why the following report is uninterpretable: “The server achieved 120 tokens/s.”

### Part B: KV-cache arithmetic

For a conventional MHA/GQA decoder, begin with:

```text
KV bytes per request
= layers × sequence_tokens × 2(K,V) × kv_heads × head_dim × bytes_per_element
```

Calculate the KV cache for all four cases. Show units and assumptions.

| Case | Layers | Tokens | KV heads | Head dim | Dtype |
|---|---:|---:|---:|---:|---|
| A | 32 | 4,096 | 32 | 128 | BF16 |
| B | 32 | 4,096 | 8 | 128 | BF16 |
| C | 32 | 32,768 | 8 | 128 | BF16 |
| D | 32 | 32,768 | 8 | 128 | FP8 |

Answer:

1. What factor does GQA change relative to MHA?
2. What factor does KV FP8 change?
3. What important runtime memory is absent from this equation?
4. If physical allocation is page based, why may observed memory exceed live-token arithmetic?

### Part C: latency model

Write a simple decomposition:

```text
request latency = queue + preprocess + prefill + repeated decode + postprocess + transport
```

For low, medium, and high concurrency, predict the dominant term and explain what evidence would disprove your prediction.

### Completion artifact

Save `week-00/cost-model.md`. Include your original prediction, corrected model after reading, and one paragraph on the largest change in your thinking.

After completing your own attempt, compare it with the [Week 0 reference solution](./solutions/week-00/cost-model.md).

## Lab 1 — Token-to-logit trace

### Part A: render and tokenize

Choose one instruct model and one two-message conversation. Record:

1. The original role/content structure.
2. The exact rendered chat-template string.
3. Token IDs, including special tokens.
4. Decoded text with and without skipping special tokens.
5. Whether a generation prompt marker is appended.

If you cannot execute the tokenizer, inspect its `tokenizer_config.json` and write a source-only prediction.

### Part B: annotate five boundaries

Trace and label shapes at:

1. `input_ids`;
2. token embeddings;
3. one decoder layer input/output;
4. final hidden state used for generation;
5. logits and selected token ID.

For every boundary, record dtype, device, batch dimension, sequence dimension, and sharding assumption.

### Part C: failure analysis

Explain the likely symptom, detection method, and owner for:

- missing assistant-generation marker;
- wrong EOS token;
- weights interpreted with the wrong quantization config;
- tied embeddings loaded twice or inconsistently;
- TP shard shape mismatch.

### Completion artifact

Save `week-01/token-to-logit.md`. The trace must cite `DefaultModelLoader`, one model `forward`, one `load_weights`, and `Sampler.forward` from [CODE_MAP.md](./CODE_MAP.md).

## Lab 2 — Request lifecycle

### Part A: source trace table

Fill one row each time control or data crosses a meaningful component boundary:

| Step | Process/component | Function | Input object | Output object | State mutated | Sync/async |
|---:|---|---|---|---|---|---|
| 1 | | | | | | |

Your trace must include HTTP entry, TokenizerManager, Scheduler, TpModelWorker, ModelRunner, attention backend, sampler, DetokenizerManager, and the streaming response path.

### Part B: sequence diagram

Draw a sequence diagram from request arrival through two generated tokens. Mark:

- process boundaries;
- queues/sockets;
- the repeated decode loop;
- GPU execution;
- the point at which token IDs become text;
- any point where backpressure could accumulate.

### Part C: version archaeology

Compare one lifecycle symbol between v0.4.0 and v0.5.16 using the worksheet in [CODE_MAP.md](./CODE_MAP.md). Decide whether its responsibility stayed stable, split, merged, or moved.

### Completion artifact

Save `week-02/request-lifecycle.md` and an image/SVG/PDF of the diagram. A reader should be able to place a breakpoint for each hop from your table.

## Lab 3 — Hand-simulate continuous batching

Use this workload:

| Request | Arrival iteration | Uncached prompt tokens | Desired output tokens |
|---|---:|---:|---:|
| A | 0 | 6 | 4 |
| B | 1 | 2 | 3 |
| C | 2 | 7 | 2 |

Assume:

- the iteration token budget is 5;
- chunked prefill is allowed;
- decode consumes one token per running request per iteration;
- output length is known only for simulation;
- no prefix cache hit;
- enough KV memory exists.

### Part A: define your policy

State how you prioritize new prefill versus running decode and how unused token budget is filled. The policy must be deterministic.

### Part B: simulate

Fill at least six iterations:

| Iteration | Arrivals | Prefill work | Decode work | Waiting state | Running state | Tokens used |
|---:|---|---|---|---|---|---:|
| 0 | | | | | | |

Calculate approximate TTFT and completion iteration for each request.

### Part C: compare policy changes

Repeat with one change:

- smaller prefill chunks;
- decode-first priority;
- strict first-come-first-served;
- one request with a reusable cached prefix.

Explain which request benefits, which suffers, and why aggregate utilization may not capture fairness.

### Completion artifact

Save `week-03/scheduler-simulation.md` with both schedules and a 300-word policy comparison anchored to the real scheduler functions.

## Lab 4 — KV-cache ledger and lifecycle

### Part A: memory ledger

Choose a real model config. Record:

| Allocation | Formula or source | Predicted bytes | Lifetime | Scales with |
|---|---|---:|---|---|
| Model weights | | | | |
| KV cache | | | | |
| Activations | | | | |
| CUDA Graph pool | | | | |
| Attention/runtime workspace | | | | |
| Framework/non-torch overhead | | | | |

Track uncertainty explicitly. A range with justified assumptions is better than false precision.

### Part B: logical versus physical mapping

For these token sequences:

```text
R1: [10, 11, 12, 20, 21]
R2: [10, 11, 12, 30]
R3: [10, 11, 40]
```

Draw:

- request slots;
- token positions;
- physical KV slots;
- a radix tree after R1 completes, after R2 begins, and after R3 begins;
- reference/lock state sufficient to explain which cache entries are evictable.

Do not assume token ID equals physical slot ID.

### Part C: source audit

Use the pinned links to answer:

1. What does `ReqToTokenPool.alloc` allocate?
2. What does `RadixCache.match_prefix` return and update?
3. What changes when a request finishes?
4. Which component ultimately owns physical KV storage?

### Track B/C experiment

With fixed prompts and output lengths, vary only concurrency or static-memory fraction. Record predicted versus observed maximum concurrency and explain the residual using the ledger.

### Completion artifact

Save `week-04/kv-ledger.md` and `week-04/kv-lifecycle.*`.

## Lab 5 — Attention backend contract

### Part A: contract table

Create separate rows for extend/prefill and decode:

| Mode | Q input | K/V source | Cache writes | Dynamic metadata | Kernel/wrapper | Output |
|---|---|---|---|---|---|---|
| Extend | | | | | | |
| Decode | | | | | | |

For each tensor, state logical shape and physical-layout questions you would need to answer before writing a kernel.

### Part B: resource classification

Classify each change by its first-order target:

- tiling Q/K/V;
- paged KV storage;
- fewer host kernel launches;
- quantized KV values;
- fused RoPE/cache write;
- prefix reuse;
- tensor parallel attention;
- precomputed metadata.

Allowed targets: arithmetic, HBM traffic, cache capacity, launch overhead, inter-GPU communication, or CPU scheduling. Some changes have a secondary target; name it separately.

### Part C: source trace

Trace metadata creation into both `forward_extend` and `forward_decode`. Identify which values can be reused and which must change every iteration.

### Track B/C experiment

Profile one short-prefill/long-decode and one long-prefill/short-decode workload. Compare the dominant operators and memory/compute utilization. Do not compare throughput unless total input and output tokens are reported.

### Completion artifact

Save `week-05/attention-contract.md` with a prediction of when decode becomes memory-bandwidth limited.

## Lab 6 — CUDA Graph and overlap audit

### Part A: graph-safety checklist

Audit one hot-path function. Fill one row for every suspicious operation:

| Source line | Operation | Allocation? | Stable address? | Static shape/control? | Host sync? | RNG state? | Verdict |
|---|---|---|---|---|---|---|---|

Your verdict must be one of: safe as written, safe under stated invariant, needs capture outside, or graph-incompatible.

### Part B: timeline

Draw two timelines with CPU scheduling above GPU execution:

1. normal scheduling where CPU work precedes each GPU step;
2. overlap scheduling where CPU prepares step `n+1` while GPU executes step `n`.

Mark dependencies that prevent arbitrary reordering. State whether each improvement removes work or hides it.

### Part C: persistent-buffer proof

Using the SGLang-Omni case study in [CODE_MAP.md](./CODE_MAP.md), explain:

1. why graph capture is deferred;
2. why `copy_`, `fill_`, and `zero_` differ from allocating replacement tensors;
3. which values are dynamic across replays;
4. which Python decisions are fixed at capture time.

### Track B/C experiment

Compare graph disabled/enabled for at least two batch sizes after warmup. Report median and variation for latency plus one throughput metric. Verify that outputs remain equivalent under the chosen sampling mode.

### Completion artifact

Save `week-06/graph-overlap-audit.md` and one timeline diagram.

## Lab 7 — Parallel deployment design

### Part A: linear-layer derivation

For `Y = XW`, derive both:

- column-parallel `W`, including the shape on each rank and when outputs need gathering;
- row-parallel `W`, including the partial result shape and why reduction is needed.

Use concrete dimensions: batch tokens 32, hidden size 4,096, output size 11,008, TP size 4, BF16.

Calculate the per-rank weight size and the principal communicated tensor size for each arrangement.

### Part B: topology scenarios

Design placement for:

1. eight GPUs with full high-bandwidth GPU interconnect;
2. eight GPUs split into two four-GPU PCIe/NUMA islands;
3. two nodes of eight GPUs each with slower inter-node networking.

For each, decide where TP, DP, and optionally PP/EP boundaries go. State what you optimize: latency, throughput, model fit, or resilience.

### Part C: collective mapping

Map each need to a collective and justify it:

- combine partial matrix multiplication outputs;
- collect distinct shards on every rank;
- distribute a reduced shard to each rank;
- route tokens to different experts;
- synchronize one sampled token ID.

### Track C experiment

Measure TP=1 versus TP=2 (and more if available) at low and high batch/concurrency. Record communication time separately if the profiler permits. Explain any regime where more GPUs lose.

### Completion artifact

Save `week-07/parallel-deployment.md` with one topology diagram and a table of expected communication operations.

## Lab 8 — Quantization integration review

Assume you must add a fictional format, `W4A8Block`, with 4-bit block-scaled weights and FP8 activations.

### Part A: lifecycle design

Specify:

| Phase | Required behavior | New metadata/tensors | Failure if wrong |
|---|---|---|---|
| Config detection | | | |
| Weight creation | | | |
| Checkpoint mapping | | | |
| Post-load processing | | | |
| Kernel selection | | | |
| Apply/forward | | | |
| Fallback | | | |

### Part B: compatibility matrix

Cover at least:

- GPU capability;
- activation dtype;
- dense linear versus MoE;
- tensor-parallel shard alignment;
- checkpoint already quantized versus quantize-on-load;
- tied embedding/head weights;
- unsupported shape fallback.

### Part C: test plan

Propose tests at four levels:

1. config and metadata unit tests;
2. weight-load and shard tests;
3. numerical comparison against a higher-precision reference;
4. end-to-end memory and throughput benchmark.

For numerical tests, state tolerances and why a single exact-output generation test is insufficient.

### Completion artifact

Save `week-08/quantization-review.md`. Cite the base interface and FP8 implementation phases from [CODE_MAP.md](./CODE_MAP.md).

## Lab 9 — Decoding optimization memo

Evaluate these workloads:

| Workload | Input | Output | Concurrency | Constraint |
|---|---|---|---:|---|
| JSON extraction | Medium | Short | High | Strict schema |
| Interactive chat | Medium | Medium | Low | Low ITL |
| RL rollout | Short/medium | Long | High | Sampling diversity |

### Part A: structured decoding

For each workload, estimate:

- fraction of vocabulary permitted at a typical step;
- grammar state/update overhead;
- benefit from preventing invalid tokens/retries;
- interaction with batching and CUDA Graphs.

### Part B: speculative decoding

Use this conceptual model:

```text
benefit depends on
= target passes avoided
- draft cost
- verification overhead
- rejection/recovery cost
- added scheduler/control cost
```

State how acceptance length and batch size affect the break-even point. Avoid claiming a theoretical `K×` speedup unless every omitted cost is explicitly zero.

### Part C: recommendation

Recommend structured decoding, speculative decoding, both, or neither for each workload. Every recommendation needs:

- one primary metric;
- one correctness guardrail;
- one workload variable to sweep;
- one result that would falsify the recommendation.

### Completion artifact

Save `week-09/decoding-memo.md`, limited to 1,200 words plus figures/tables.

## Lab 10 — Profiling drill

Before the capstone, perform one short drill on a slow or memory-limited workload.

### Investigation log

| Time/order | Observation | Hypothesis | Next measurement | Result | Updated belief |
|---:|---|---|---|---|---|
| 1 | | | | | |

Rules:

- Do not change code before the first measurement identifies a component.
- Do not use a system-wide average to explain a per-request tail without distribution evidence.
- Do not infer GPU idleness from low aggregate throughput alone.
- Stop after one hypothesis is strongly supported or three are falsified.

### Completion artifact

Save `week-10/profiling-drill.md` with the smallest source region consistent with the evidence.

## Capstone protocol

### 1. Proposal

Write no more than one page:

- question;
- mechanism and source location;
- falsifiable hypothesis;
- workload matrix;
- primary and guardrail metrics;
- controls;
- expected failure modes;
- hardware track and time budget.

### 2. Reproducibility package

Include:

- exact commits and model revision;
- environment record;
- launch and benchmark commands;
- fixed prompt/input dataset or generator seed;
- raw measurements;
- analysis steps;
- patch or patch sketch with pinned source links.

### 3. Required result views

At minimum, report:

1. primary metric versus the independent variable;
2. one latency distribution or percentile view;
3. memory/resource use;
4. correctness or output-quality guardrail;
5. predicted versus observed result.

### 4. Final report outline

```text
Title
Question and motivation
System and source path
Hypothesis
Experimental setup
Results
Mechanism: why the result occurred
Threats to validity
Decision: adopt, reject, or investigate
Reproduction instructions
```

### Rubric (30 points)

| Criterion | Points | Full-credit standard |
|---|---:|---|
| Question and hypothesis | 4 | Specific, falsifiable, and tied to a mechanism |
| Source understanding | 5 | Correct control/data flow with commit-pinned anchors |
| Experimental control | 6 | Reproducible workload; confounders controlled |
| Metrics and correctness | 4 | Primary metric plus relevant latency/resource/quality guards |
| Analysis | 6 | Explains mechanism; separates fact from inference; treats variance honestly |
| Communication | 3 | Clear figures/tables, units, assumptions, and commands |
| Limitations and decision | 2 | States what is unknown and makes an evidence-proportional decision |

## Progress tracker

| Week | Prediction saved | Reading complete | Code trace complete | Lab complete | Can explain from memory |
|---:|:---:|:---:|:---:|:---:|:---:|
| 0 | ☐ | ☐ | — | ☐ | ☐ |
| 1 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | ☐ | ☐ | ☐ | ☐ | ☐ |

The last column is the important one: close the notes and teach the mechanism aloud. If you cannot reconstruct the control flow or resource equation, revisit the trace rather than rereading passively.
