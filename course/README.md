# LLM Inference Systems: A Code-First Course

## Structured weekly lessons

The syllabus below gives the learning order. Each lesson is a self-contained chapter with the required concepts, equations, worked examples, source-reading guidance, checkpoints, and optional references.

1. [Week 0 — The Cost Model of LLM Inference](weeks/week-00.md)
2. [Week 1 — From Chat Messages to Logits](weeks/week-01.md)
3. [Week 2 — One Request Through SGLang](weeks/week-02.md)
4. [Week 3 — Continuous Batching and the Scheduler](weeks/week-03.md)
5. [Week 4 — KV Cache as a Memory System](weeks/week-04.md)
6. [Week 5 — Attention Backends, IO, and Data Layout](weeks/week-05.md)
7. [Week 6 — CUDA Graphs and Overlap Scheduling](weeks/week-06.md)
8. [Week 7 — Parallel Inference and Communication](weeks/week-07.md)
9. [Week 8 — Quantization as a Serving Lifecycle](weeks/week-08.md)
10. [Week 9 — Structured and Speculative Decoding](weeks/week-09.md)
11. [Week 10 — Profiling and the Capstone](weeks/week-10.md)

This course turns the repository into a guided path from “a prompt goes in” to “a production inference system schedules, executes, and returns tokens.” It uses the repository as the textbook and commit-pinned SGLang source as the laboratory.

The organizing question is:

> Where does each millisecond and each byte go when an LLM serves a request?

The course is designed for self-study over 10 core weeks at 6–8 hours per week, plus an optional Week 0 foundation. It assumes that you can read Python and know the high-level Transformer block. CUDA experience and access to a multi-GPU server are not required. Every lab has a reading-only path; GPU work progressively deepens the same mental model.

## What you will be able to do

By the end, you should be able to:

1. Trace a request through tokenization, scheduling, model execution, sampling, and detokenization.
2. Explain why prefill and decode have different bottlenecks and how continuous batching exploits that difference.
3. Account for model weights, activations, CUDA Graph pools, and KV cache memory instead of treating OOMs as mysteries.
4. Read a paged/radix KV-cache implementation and predict the effect of prefix reuse, eviction, and chunked prefill.
5. Connect attention kernels, CUDA Graphs, quantization, and parallelism to latency and throughput.
6. Design a benchmark that separates TTFT, inter-token latency, throughput, cache-hit rate, and concurrency effects.
7. Make and defend one inference-system design change using source evidence and measurements.

## How to use the course

Each week follows the same loop:

1. **Predict:** write down what you expect before reading or measuring.
2. **Build the model:** read the assigned repository notes and draw the data/control flow from memory.
3. **Trace real code:** follow the commit-pinned symbols in [CODE_MAP.md](./CODE_MAP.md).
4. **Measure or inspect:** complete the lab in [LABS.md](./LABS.md).
5. **Explain:** produce a short artifact that would let another engineer reproduce your reasoning.

Do not pass a week by finishing the reading alone. The deliverable and checkpoint are the real completion criteria.

## Choose your lab track

| Track | Hardware | What you do |
|---|---|---|
| A: Source | Any laptop | Static traces, memory arithmetic, call graphs, design reviews, and benchmark design |
| B: Single GPU | One NVIDIA GPU | Everything in A plus serving, profiling, cache and quantization experiments |
| C: Multi-GPU | 2+ NVIDIA GPUs | Everything in B plus tensor parallelism, topology, and communication experiments |

Track A is a complete course. Track B or C adds empirical evidence; it does not replace source reading.

## Version policy

This repository spans several generations of SGLang. That is useful if handled deliberately:

- The original end-to-end walkthrough uses SGLang v0.4.0 at commit [`f8b0326`](https://github.com/sgl-project/sglang/tree/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751). Treat it as the clean historical model.
- Practical labs pin SGLang v0.5.16 at commit [`fdebc93`](https://github.com/sgl-project/sglang/tree/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1). The tag was the latest release when this course was assembled on 2026-09-03.
- Never substitute `main` in a source link. If you intentionally upgrade the lab version, record the new commit and repair the code map first.
- When the historical walkthrough and current source differ, write down whether the change is a rename, a moved responsibility, or a changed design. This “version archaeology” is part of the course.

Use the [official installation guide](https://docs.sglang.ai/get_started/install.html) for the pinned release and the repository’s [Docker guide](../engineer/how-to-use-docker/readme_en.md) for reproducible environments. The SGLang documentation notes that `latest` and `dev` container tags are mutable, so use an immutable release tag.

## Entry diagnostic

Spend at most 45 minutes. Answer without searching first:

1. For a decoder-only Transformer, what work happens once per prompt token and what work happens once per generated token?
2. For a model with 32 layers, 32 KV heads, head dimension 128, BF16 KV values, and a 4,096-token sequence, estimate KV-cache bytes for one request. Count K and V.
3. Why can higher request concurrency increase throughput while worsening inter-token latency?
4. What problem does tensor parallelism solve, and what communication does it introduce?
5. Name two operations that make a captured CUDA Graph hard to replay safely.

Score one point for each answer you can justify. A score of 0–2 means you should do Week 0 in full. A score of 3–4 means skim Week 0 but complete its deliverable. A score of 5 means begin at Week 1.

## Course map

| Week | System question | Core concept | Deliverable |
|---:|---|---|---|
| 0 | What are we optimizing? | Latency, throughput, arithmetic intensity, memory accounting | One-page inference cost model |
| 1 | What exactly enters the engine? | Tokens, chat templates, model structure and weights | Token-to-logit trace |
| 2 | Where does a request travel? | Processes, queues, data objects, control flow | End-to-end request sequence |
| 3 | How are many requests made into work? | Prefill/decode, continuous batching, scheduling | Scheduler state-machine explanation |
| 4 | Why is KV cache a system? | Paged allocation, radix reuse, eviction, OOM accounting | KV-cache ledger and lifecycle trace |
| 5 | Where does GPU speed come from? | Attention IO, layouts, kernel selection | Attention backend trace |
| 6 | How is overhead hidden? | CUDA Graphs, overlap scheduling, synchronization | CPU/GPU timeline and graph-safety audit |
| 7 | How does inference scale out? | TP, DP, collectives, topology | Parallel deployment proposal |
| 8 | How do we trade precision for capacity? | Quantization lifecycle and kernels | Quantization integration review |
| 9 | How can decoding do less work? | Structured and speculative decoding | Optimization decision memo |
| 10 | Can you reason from evidence? | Profiling, experimental design, synthesis | Capstone report |

## Week 0 — Build an inference cost model

### Motivation

Optimizations are meaningless until you know which resource is limiting. This week builds a vocabulary for time and memory so later claims are testable.

### Learn

- Distinguish offline inference from online serving.
- Define time to first token (TTFT), inter-token latency (ITL), end-to-end latency, output-token throughput, and goodput.
- Explain why prefill tends toward compute-bound matrix-matrix work while decode often tends toward bandwidth-bound matrix-vector work.
- Account separately for weights, KV cache, activations, runtime workspaces, and graph pools.

### Read

- The repository [overview](../README.md), especially “SGLang Learning Notes” and “ML System Fundamentals.”
- [SGLang Before the Scheduler](./SGLANG_PRIMER.md) in full. This is the required system-context primer.
- [When SGLang OOMs](../sglang/kvcache-code-walk-through/mem-fraction-static-en.md), only through the memory-budget equation. Stop when the article enters SGLang-specific allocation details.

Do **not** read the scheduler deep-dive in Week 0. It assumes the request lifecycle, batching model, and KV-cache vocabulary introduced by the primer. You will read it in Week 3.

### Lab and deliverable

Complete Lab 0 in [LABS.md](./LABS.md). Submit a one-page cost model containing:

- a latency decomposition for one request;
- a server throughput equation;
- weight and KV-cache estimates for one model configuration;
- three hypotheses about what will bottleneck at low, medium, and high concurrency.

### Checkpoint

You pass when you can explain why “tokens per second” is ambiguous without asking whether it is per request or aggregate and whether prefill tokens are counted.

## Week 1 — From conversation to logits

### Motivation

Many apparent model bugs begin before the first layer: a wrong chat template, special token, dtype, shard mapping, or output head changes the computation.

### Learn

- Explain the boundary between a chat message, rendered prompt text, token IDs, embeddings, hidden states, logits, and sampled IDs.
- Recognize tied embeddings, model shards, and post-load weight processing.
- Trace a decoder layer without getting lost in model-specific branches.

### Read

- [Special tokens and chat templates](../transformers/special_tokens/special_tokens_en.md).
- The “Model Load Weights and Perform Forward” section of [SGLang Code Walk Through](../sglang/code-walk-through/readme.md#model-load-weights-and-perform-forward).
- Optional, explicitly **Pending Review** in the repository: [How a model is loaded](../sglang/how-model-is-loaded/readme.md). Use it as an exploration prompt, not as an authority.

### Code trace

Follow the current source links for `DefaultModelLoader`, `Qwen3ForCausalLM.forward`, and `Qwen3ForCausalLM.load_weights` in [CODE_MAP.md](./CODE_MAP.md).

### Lab and deliverable

Complete Lab 1. Produce a token-to-logit trace with shapes at five boundaries. Include one failure scenario caused by a template mismatch and one caused by a weight/dtype mismatch.

### Checkpoint

Given a prompt and model config, you can identify which component owns rendering, tokenization, embedding lookup, forward execution, logit projection, and sampling.

## Week 2 — Trace one request end to end

### Motivation

An inference engine is not one model call. It is a distributed control plane around repeated model calls. This week gives you the map used by every later optimization.

### Learn

- Identify the process boundary and communication role of the HTTP server, TokenizerManager, Scheduler, TpModelWorker, ModelRunner, attention backend, and DetokenizerManager.
- Separate request-level objects from batch-level and model-execution objects.
- Distinguish streaming control flow from non-streaming response assembly.

### Read

- [SGLang Code Walk Through](../sglang/code-walk-through/readme.md) in full.
- Revisit the architecture SVG inside that article after reading; redraw it without looking.

### Code trace

Start at `http_server.generate_request`, then follow `TokenizerManager.generate_request`, the scheduler loop, `TpModelWorker.forward_batch_generation`, and `ModelRunner.forward`. Use the current-source column in [CODE_MAP.md](./CODE_MAP.md), but compare it with the historical links embedded in the article.

### Lab and deliverable

Complete Lab 2. Produce a sequence diagram with:

- process boundaries;
- the type/name of the payload at each hop;
- the loop that emits each additional token;
- one likely backpressure point.

### Checkpoint

Starting from an OpenAI-compatible request, you can narrate the path to the first streamed text fragment and name the component that owns each transition.

## Week 3 — Scheduling prefill and decode

### Motivation

The engine must continually decide which work deserves scarce GPU tokens and KV slots. Throughput, fairness, and latency meet in this loop.

### Learn

- Contrast static batching with continuous batching.
- Explain `waiting_queue`, `running_batch`, `last_batch`, and why batches change every iteration.
- Explain chunked prefill as both a latency/fairness mechanism and a memory constraint.
- Derive how a scheduling decision changes TTFT and ITL for two competing requests.

### Read

- Revisit [SGLang Before the Scheduler](./SGLANG_PRIMER.md), especially the name translation and three-pass reading guide.
- [From KV Cache to Zero Overhead Scheduling](../sglang/scheduler/readme-en.md) in full.
- Re-read the Scheduler section of [SGLang Code Walk Through](../sglang/code-walk-through/readme.md#scheduler-receive-requests-and-process-batches).

### Code trace

Trace `Scheduler.event_loop_normal`, `process_input_requests`, `get_next_batch_to_run`, `run_batch`, and `process_batch_result` at the pinned current commit.

### Lab and deliverable

Complete Lab 3. Hand-simulate at least six scheduler iterations for three requests with different prompt and output lengths. Then write a 300-word explanation of the fairness/throughput trade-off you observed.

### Checkpoint

You can predict when a new long prompt delays decode work and describe how chunk size changes that outcome.

## Week 4 — KV cache, prefix reuse, and memory

### Motivation

KV cache turns autoregressive generation from repeated full-sequence computation into incremental decoding, but it also becomes the engine’s dynamic memory allocator and admission-control constraint.

### Learn

- Derive KV-cache size from layers, KV heads, head dimension, dtype, and token count.
- Distinguish request-to-token mapping from token-to-physical-KV mapping.
- Explain page allocation, prefix matching, cache insertion, lock references, eviction, and reclamation.
- Explain why `--mem-fraction-static` budgets more than model weights and KV tensors.

### Read

- [When SGLang OOMs](../sglang/kvcache-code-walk-through/mem-fraction-static-en.md) in full.
- The KV-cache section of [From KV Cache to Zero Overhead Scheduling](../sglang/scheduler/readme-en.md#kv-cache-management).
- Optional, explicitly **Pending Review**: [KV Cache Code Walkthrough](../sglang/kvcache-code-walk-through/readme.md). Verify every claim against the pinned source.

### Code trace

Follow `ReqToTokenPool`, the token-to-KV pool implementations, `RadixCache.match_prefix`, and `RadixCache.cache_finished_req` in [CODE_MAP.md](./CODE_MAP.md).

### Lab and deliverable

Complete Lab 4. Produce a memory ledger and a token-lifecycle diagram. Explain exactly what is logical, what is physical, and when each resource can be reused.

### Checkpoint

Given two requests sharing a prefix and a fixed KV budget, you can show which tokens need new compute and which physical slots remain live.

## Week 5 — Attention backends and data layout

### Motivation

Scheduling decides *what* runs; kernels and layouts decide how efficiently bytes reach tensor cores. The abstraction boundary between the model and attention backend is where these worlds meet.

### Learn

- Explain why attention is tiled and why layout/stride determines coalescing and reuse.
- Distinguish extend/prefill attention metadata from decode metadata.
- Connect paged KV indices to the attention kernel’s inputs.
- Recognize when an optimization targets arithmetic, memory traffic, launch overhead, or synchronization.

### Read

- The AttentionBackend section of [SGLang Code Walk Through](../sglang/code-walk-through/readme.md#attentionbackend-accelerate-model-forward).
- [Data layout and notation](../kernels/data_layout.md), through “Tile And Its Layout”; treat the later TMEM/swizzle sections as a stretch reading.
- Optional mathematical background: [Flash Attention notes](../transformers/flash_attn/readme.md). This note focuses on backward-pass derivation, so use it to strengthen attention algebra rather than as the inference kernel explanation.

### Code trace

Trace `FlashInferAttnBackend.init_forward_metadata`, `forward_extend`, and `forward_decode`. At each function boundary, record tensor purpose and layout, not just names.

### Lab and deliverable

Complete Lab 5. Produce an “attention backend contract”: inputs, outputs, persistent state, dynamic metadata, and mode-specific behavior.

### Checkpoint

You can explain why decode attention and prefill/extend attention often select different kernels even though both compute attention.

## Week 6 — CUDA Graphs and overlap scheduling

### Motivation

Once GPU kernels are fast, Python scheduling, kernel launch, allocation, and synchronization become visible. Removing overhead requires stricter execution contracts.

### Learn

- Explain capture/replay and the requirements for stable addresses, static control flow, and controlled shapes.
- Identify accidental host-device synchronization and allocation inside a hot path.
- Compare normal, multi-step, and overlap scheduling.
- Distinguish latency hidden by overlap from work actually eliminated.

### Read

- [Revisiting CUDA Graph](../torch/cuda-graph/readme-2-en.md) in full.
- The “Overlap Scheduler” section of [SGLang scheduler](../sglang/scheduler/readme-en.md#overlap-scheduler-hiding-scheduling-overhead-behind-operators).
- Optional earlier perspective: [CUDA Graph based on torch-memory-savor](../torch/cuda-graph/readme_en.md).
- Optional, explicitly **Pending Review**: [Zero-Overhead Batch Scheduler](../sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md).

### Lab and deliverable

Complete Lab 6. Draw a before/after CPU–GPU timeline and audit one source path for graph safety. Every flagged operation must be classified as allocation, address change, shape/control-flow variation, RNG state, or synchronization.

### Checkpoint

You can explain why replacing a tensor object can break replay while copying new values into a persistent tensor can be safe.

## Week 7 — Parallel inference and communication

### Motivation

A model may not fit or meet latency targets on one GPU. Partitioning solves one constraint by introducing communication and topology constraints.

### Learn

- Compare data, tensor, pipeline, and expert parallelism for inference.
- Derive the communication points of column- and row-parallel linear layers.
- Explain AllReduce, AllGather, ReduceScatter, and AllToAll by their data movement.
- Read `nvidia-smi topo -m` and propose a topology-aware rank layout.

### Read

- [NCCL in SGLang](../torch/nccl/readme_en.md).
- [PyTorch Distributed](../torch/torch-distributed/readme.md), focusing on process groups and collectives.
- Optional, explicitly **Pending Review**: [Tensor Parallelism from scratch](../torch/tensor-parallelism/readme.md) and [DP Attention](../sglang/dp-attention/readme.md). Validate them with your own shape/communication derivation.

### Lab and deliverable

Complete Lab 7. Propose placements for a model on two hypothetical 8-GPU machines: one fully connected by NVLink and one split across PCIe/NUMA domains. State the objective, communication volume, and expected bottleneck.

### Checkpoint

You can explain why using more GPUs can reduce capacity pressure yet worsen latency for a small batch.

## Week 8 — Quantization as a lifecycle

### Motivation

Quantization is not merely changing a dtype. A serving framework must create parameters, load checkpoints, transform scales/packing, select kernels, and preserve model-specific semantics.

### Learn

- Distinguish weight-only, weight-and-activation, and KV-cache quantization.
- Explain per-tensor, per-channel, and block scales as accuracy/kernel trade-offs.
- Trace SGLang’s `create_weights → process_weights_after_loading → apply` lifecycle.
- Identify the integration points needed for a new quantization method.

### Read

- [General quantization background](../sglang/quantization/general_background.md).
- [SGLang quantization architecture](../sglang/quantization/quantization_architecture_en.md).

### Code trace

Compare the abstract quantization interfaces with `Fp8Config` and `Fp8LinearMethod` at the pinned current commit. Record where format-specific behavior enters model loading and execution.

### Lab and deliverable

Complete Lab 8. Write a design review for adding a hypothetical block-wise weight format. Include parameter metadata, checkpoint mapping, post-load transformation, kernel dispatch, fallback, and tests.

### Checkpoint

You can name the failure produced when each lifecycle phase is implemented incorrectly: load failure, silent accuracy loss, excessive memory, or slow fallback.

## Week 9 — Structured and speculative decoding

### Motivation

Not every speedup comes from making the same forward pass faster. We can constrain the token space or verify several candidate tokens per target-model pass—but both techniques add control-plane work and have workload-dependent break-even points.

### Learn

- Explain how grammar state masks the vocabulary and where grammar work enters the decode loop.
- Explain draft, verify, accept/reject, and expected accepted length in speculative decoding.
- Derive a qualitative break-even condition that includes draft cost, verification cost, acceptance rate, and batch size.
- Recognize interactions with scheduling, CUDA Graphs, and sampling.

### Read

- [Constraint Decoding](../sglang/constraint-decoding/readme_en.md).
- Optional, explicitly **Pending Review**: [Speculative Decoding](../sglang/speculative-decoding/speculative-decoding.md).
- For a reviewed production application, read [Power Up Speculative Decoding in Reinforcement Learning](../rlhf/slime/spec/readme-en.md), focusing on why acceptance changes over training.

### Lab and deliverable

Complete Lab 9. Write an optimization decision memo for three workloads: JSON extraction, low-concurrency chat, and high-throughput rollout generation. Recommend structured decoding, speculative decoding, both, or neither, and state how you would falsify each recommendation.

### Checkpoint

You can explain why a higher theoretical token acceptance rate does not automatically improve end-to-end throughput.

## Week 10 — Profiling and capstone

### Motivation

Inference engineering ends in evidence. A persuasive optimization isolates a bottleneck, changes one mechanism, and reports the workload and trade-offs clearly.

### Learn

- Move from symptom to hypothesis to measurement to source location.
- Separate warmup, compilation, and steady-state behavior.
- Avoid benchmark traps: changing prompts, uncontrolled output length, cache reuse, mixed concurrency, and unreported versions.
- Communicate a result that another engineer can reproduce.

### Read

- [Interactive profiling](../sglang/interactive-profiling/Interactive_Profiling_en.md).
- Revisit [When SGLang OOMs](../sglang/kvcache-code-walk-through/mem-fraction-static-en.md) as an example of resource accounting.
- Stretch case study: [Optimizing TTS inference](../sglang/sglang-omni/tts-optimization.md), which connects profiling, caching, CUDA Graphs, CPU/GPU overlap, and batching in a multi-stage system.

### Capstone

Choose one:

1. **Scheduler study:** evaluate one chunked-prefill or admission-control policy across at least three concurrency levels.
2. **Memory study:** predict and measure maximum concurrency as prompt/output lengths and KV dtype vary.
3. **Execution study:** identify one CPU, launch, synchronization, or layout overhead and evaluate a source-level change or credible patch design.
4. **Scaling study:** evaluate or model tensor-parallel scaling across two placements/topologies.

Use the capstone protocol and rubric in [LABS.md](./LABS.md). The final report must link every source claim to the pinned commit and distinguish measured facts from inference.

### Checkpoint

You pass when a skeptical engineer can reproduce the workload, audit the source path, and tell whether the result supports your original claim.

## Optional extension — Beyond text-only autoregressive models

After the core course, study how the same principles change for multi-stage and non-autoregressive generation:

- [Omni model inference](../transformers/omni/readme-en.md): codecs, RVQ, Dual AR, and Thinker–Talker structure.
- [Why SGLang Omni](../sglang/sglang-omni/why-sglang-omni-en.md): stage isolation, scheduling, communication, and memory boundaries.
- [SGLang Diffusion walkthrough](../sglang/code-walk-through/sgl_diffusion_en.md): request flow for diffusion generation.
- [CPU as a first-class resource](../sglang/sglang-omni/cpu-first-class-citizen.md): a case study in diagnosing host contention rather than blaming GPU code.

Extension deliverable: compare text AR, Dual AR, and diffusion inference along the axes of state, scheduling unit, cache lifetime, graphability, and streaming.

## Assessment

| Component | Weight | Evidence |
|---|---:|---|
| Weekly diagrams and traces | 25% | Correct ownership, order, state, and source anchors |
| Memory/performance derivations | 20% | Units, assumptions, arithmetic, and sanity checks |
| Labs or source audits | 25% | Reproducibility and quality of evidence |
| Capstone | 30% | Question, experimental control, interpretation, and communication |

Use this standard throughout:

- **Can name it:** insufficient.
- **Can explain it:** developing.
- **Can predict behavior before running:** proficient.
- **Can locate, change, and evaluate the mechanism:** advanced.

## A note on repository quality signals

The repository’s README marks some articles **Pending Review**, and several walkthroughs intentionally document older SGLang versions. This course never hides those facts. Required readings emphasize reviewed material; pending material is optional and must be checked against commit-pinned source. Historical code is used to build a clean mental model, while the current pinned release is used to learn how real systems evolve.

That tension is a feature of the course: production inference knowledge is not memorizing today’s filenames. It is learning to recover design intent from measurements, interfaces, and source history.
