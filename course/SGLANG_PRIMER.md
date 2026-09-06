# SGLang Before the Scheduler

Read this before the SGLang scheduler article. The scheduler article begins in the middle of the story: it assumes you already know what an inference server does, why generation is iterative, what prefill and decode mean, and why KV cache is scarce. This primer supplies that missing story.

Your goal is not to memorize class names. By the end, you should be able to explain this sentence:

> SGLang repeatedly chooses a changing batch of requests, reserves KV-cache space for their next tokens, runs the model, records the results, and repeats until every request finishes.

## 1. What is SGLang?

A Hugging Face model can turn token IDs into logits. That is necessary, but it is not yet a serving system.

A serving system must also:

- accept many requests arriving at different times;
- render chat messages and tokenize them;
- decide which requests run together;
- store attention state between generated tokens;
- run the model efficiently on one or more GPUs;
- sample tokens and detect stopping conditions;
- stream partial text back to each user;
- reclaim memory when requests finish.

SGLang is the system around the model that performs those jobs. Think of the model as an engine and SGLang as the traffic control, fuel system, transmission, and dashboard around it.

For this course, reduce SGLang to five conceptual components:

1. **Frontend:** accepts an HTTP or offline inference request.
2. **TokenizerManager:** turns text/chat into token IDs and later matches results back to the caller.
3. **Scheduler:** decides which requests and tokens should run in the next GPU step.
4. **Model worker:** executes the model and samples the next token.
5. **Detokenizer:** turns output token IDs into streamable text.

The real implementation contains more processes and abstractions, but this five-part model is enough to begin.

```text
request
  ↓
Frontend → TokenizerManager → Scheduler → Model worker
              ↑                  ↓             ↓
              └── Detokenizer ← token IDs ← logits/sample
```

The arrow from the model worker back toward the scheduler repeats once for every generation step.

## 2. Why generation is a loop

Suppose the tokenized prompt is:

```text
[The] [capital] [of] [France] [is]
```

The model produces a probability distribution for the next token. Assume the sampler chooses `[Paris]`. To generate another token, the model now needs to process the sequence ending in `[Paris]`, then choose the next token. Generation is therefore iterative:

```text
prompt → logits → Paris
prompt + Paris → logits → .
prompt + Paris + . → logits → EOS
```

The scheduler is not invoked only once per request. It participates in this repeated loop while new requests may arrive and old requests may finish.

## 3. Prefill and decode

The first model execution for a request and later executions have different shapes.

### Prefill

During **prefill**, the model processes the prompt tokens. A 1,000-token prompt presents substantial parallel work: many token positions pass through large matrix multiplications together. The system also creates the attention keys and values that later steps will reuse.

Prefill answers:

> What state must the model build from the prompt before it can generate?

### Decode

During **decode**, the system normally generates one new token per active request per iteration. The model reuses the earlier keys and values and computes only the new token’s representation.

Decode answers:

> Given all cached history, what is the next token?

The important contrast is:

- Prefill processes many new tokens for a request at once.
- Decode processes one new token per active request per step.
- Prefill often has abundant parallel arithmetic.
- Decode repeatedly reads model weights and growing attention state for little new work.

This is why TTFT is strongly affected by prompt processing, while inter-token latency is strongly affected by the repeated decode loop.

## 4. KV cache: the state carried between steps

Self-attention needs keys and values for earlier tokens. Recomputing all earlier layers for every new token would be wasteful, so the engine saves those keys and values in a **KV cache**.

For a conventional decoder, the rough storage is:

```text
layers × tokens × 2 (K and V) × KV heads × head dimension × bytes per element
```

KV cache changes two things at once:

1. It makes decode incremental instead of recomputing the whole prefix.
2. It turns active sequence length into a memory-capacity constraint.

If the server admits too many tokens, KV cache can run out even when the model weights fit. The scheduler must therefore reason about both compute work and available cache slots.

Do not confuse three identities:

- a **request** belongs to one user operation;
- a **token position** belongs to a request’s sequence;
- a **physical KV slot** is memory allocated by the engine.

SGLang uses mappings and pools so token positions do not need to live at fixed contiguous GPU addresses.

## 5. Why batching is continuous

Traditional static batching assumes every item enters and leaves together. LLM requests do not behave that way:

- prompts have different lengths;
- outputs have different lengths;
- requests arrive continuously;
- some stop early;
- some reuse cached prefixes.

If requests A and B start together but A finishes first, waiting for B would waste capacity. SGLang uses **continuous batching**: after each iteration, it removes finished work, admits eligible new work, and constructs the next batch.

Consider this simplified history:

```text
Iteration 0: prefill A
Iteration 1: decode A + prefill B
Iteration 2: decode A + decode B
Iteration 3: A finishes; decode B + prefill C
Iteration 4: decode B + decode C
```

Real policies may not mix work exactly this way, but the central idea is stable: the batch is a temporary scheduling decision, not a permanent group of requests.

## 6. What the Scheduler owns

At a conceptual level, the Scheduler repeats four operations:

1. **Receive:** accept newly tokenized requests and control messages.
2. **Select:** choose prefill or decode work that fits policy and memory constraints.
3. **Run:** ask the model worker to execute the selected batch.
4. **Update:** record output tokens, finish or requeue requests, stream results, and update/release cache state.

```text
receive → select → run → update
   ↑                         ↓
   └──────── repeat ─────────┘
```

The Scheduler does not implement the Transformer. It prepares and manages the work that the model worker executes.

## 7. Translate the scheduler article’s names

The article introduces several queues and batch objects quickly. On the first pass, translate them into plain language:

- `waiting_queue`: requests not currently ready to perform ordinary decode; commonly new/prefill work or retracted work.
- `new_batch`: prefill/extend work selected for the current iteration.
- `running_batch`: requests already in the repeated decode phase.
- `cur_batch`: whichever batch the Scheduler chose to execute now.
- `last_batch`: the batch from the preceding iteration whose state may need to be merged or resolved.
- `chunked_req`: a long prompt whose prefill has been split across iterations.
- `ScheduleBatch`: the scheduler’s CPU-side description of the work.
- `ModelWorkerBatch`: the execution-oriented representation sent toward the model worker.
- `ForwardBatch`: the tensor-level representation consumed by model execution.
- `GenerationBatchResult`: logits/token results and synchronization metadata returned after execution.

For your first reading, focus on `waiting_queue`, `new_batch`, `running_batch`, and `cur_batch`. Treat the other types as refinements of the boundary between scheduling and execution.

## 8. One request’s minimum lifecycle

Before reading any source, be able to narrate this lifecycle:

1. Text/chat arrives at the frontend.
2. The TokenizerManager renders and tokenizes it.
3. The Scheduler places the request in waiting state.
4. The Scheduler selects its prompt tokens for prefill.
5. The model worker runs prefill, creates KV state, and selects the first output token.
6. The Scheduler records that token and moves the unfinished request into decode state.
7. Each decode iteration reserves space for one more token, runs the model, and records the result.
8. The detokenizer converts token IDs to text for streaming.
9. EOS, a stop sequence, or a token limit finishes the request.
10. The Scheduler releases request-owned resources or preserves reusable prefix-cache state.

New requests can enter between almost any two iterations, which is why the Scheduler needs a policy rather than a simple per-request loop.

## 9. How to read the scheduler article without drowning

Read [From KV Cache to Zero Overhead Scheduling](../sglang/scheduler/readme-en.md) in three passes, during Week 3—not Week 0.

### Pass 1: control flow only

Read the introduction, the descriptions of `waiting_queue`, `new_batch`, `running_batch`, and `cur_batch`, and the “Normal Scheduler” section. Skip class fields, code, speculative decoding, and overlap details.

Answer only:

1. Where can a request be before prefill, during prefill, and during decode?
2. What determines whether the next batch is prefill or decode?
3. What state changes after a token is produced?

### Pass 2: data boundaries

Read `ScheduleBatch → ModelWorkerBatch → ForwardBatch → GenerationBatchResult`.

For each boundary, ask:

- Is the object mainly controlled by CPU scheduling or GPU execution?
- Which fields describe requests, and which fields are tensors?
- What must return to the scheduler after the GPU step?

### Pass 3: optimizations

Only after the normal loop is clear, read KV-cache management and overlap scheduling. For each optimization, ask:

- What baseline cost or idle period exists?
- Is work removed, or merely overlapped?
- What new correctness constraint is introduced?

## 10. Your readiness check

Do not continue to the scheduler article until you can answer these without notes:

1. Why does generation require repeated model execution?
2. What work happens during prefill versus decode?
3. Why does KV cache improve computation but constrain concurrency?
4. Why does the batch membership change after every iteration?
5. What four operations does the Scheduler repeat?

If you can answer four of five, you have enough context. You do not need to know SGLang internals yet; the scheduler chapter is where you begin learning them.
