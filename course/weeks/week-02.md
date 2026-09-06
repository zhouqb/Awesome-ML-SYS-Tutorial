# Week 2 — One Request Through SGLang

Reading time: 40–50 minutes. This week turns “the server runs a model” into a concrete control/data-flow map. Once this map is stable, scheduler, cache, and kernel optimizations have somewhere to attach.

## Learning objectives

After this lesson, you should be able to:

- name the major SGLang components and their ownership boundaries;
- trace a request to the first streamed token and through repeated decode;
- distinguish request, scheduling-batch, worker-batch, and forward-batch representations;
- explain why tokenization and detokenization are separated from GPU execution;
- identify likely queueing and synchronization boundaries.

## 1. Architecture in one picture

```text
Client
  │ HTTP / Engine API
  ▼
Frontend + TokenizerManager
  │ tokenized request objects
  ▼
Scheduler ── selects batch, owns request/cache control state
  │ ModelWorkerBatch
  ▼
TpModelWorker → ModelRunner → Model → AttentionBackend
  │                          logits → sampler
  └──────── GenerationBatchResult ◄──────────────┘
  │
  ├─ update request, cache, stopping, queues
  ▼
DetokenizerManager → TokenizerManager/frontend → Client stream
```

Several components can live in separate processes and communicate through queues/sockets. The point is isolation: CPU-facing protocol and text work should not dictate the GPU execution loop, and GPU work should not block accepting or streaming unrelated requests.

## 2. Frontend and TokenizerManager

The frontend validates an HTTP/OpenAI-compatible request and converts it to an internal generation request. Streaming and non-streaming responses share most of the backend path but assemble results differently.

`TokenizerManager` owns the text/token boundary. Its responsibilities include:

- rendering chat templates;
- tokenizing text or accepting token IDs;
- normalizing single, batch, and parallel-sampling requests;
- associating backend responses with the waiting caller;
- handling control operations such as weight-update coordination.

Current source anchors:

- [`http_server.generate_request`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/entrypoints/http_server.py#L833)
- [`TokenizerManager.generate_request`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/tokenizer_manager.py#L624)

The key question is not every branch. Find the moment a caller-facing request becomes a scheduler-facing object and the mechanism used to await matching responses.

## 3. Scheduler: control-plane owner

The scheduler owns active-request lifecycle and the decision for the next model step. It receives tokenized requests, maintains waiting/running state, consults policy and memory availability, builds a batch, runs it through the worker, then processes results.

Conceptually:

```python
while True:
    receive_and_admit_requests()
    batch = select_next_work()
    result = run_batch_on_worker(batch)
    update_requests_cache_and_streams(batch, result)
```

This is pseudocode. The production loop handles overlap, distributed coordination, prefill/decode modes, speculative decoding, embeddings, grammar state, and asynchronous copies.

Start at pinned [`Scheduler.event_loop_normal`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L1520). Locate calls corresponding to receive, select, run, and update. Stop there on the first pass.

## 4. Why the request changes representation

One giant request object would mix protocol fields, scheduling policy, CPU metadata, device tensors, and kernel metadata. SGLang narrows the representation as work approaches the GPU:

### Request-level state

A request retains input IDs, generated IDs, sampling/stopping state, cache association, and client identity. It lives across many iterations.

### `ScheduleBatch`

The scheduler’s CPU-oriented view groups selected requests and includes sequence lengths, prefix/extend lengths, cache allocators, and forward mode. It answers: “What should run now?”

### `ModelWorkerBatch`

The worker-oriented view carries what device execution needs: input IDs, sequence lengths, request-pool indices, output cache locations, and mode. It answers: “What must the worker transfer/prepare?”

### `ForwardBatch`

The model runner’s tensor-level view supplies positions and metadata used by the model and attention backend. It answers: “What tensors and mode does this forward consume?”

### `GenerationBatchResult`

The result carries sampled token IDs, logits-related outputs, accepted-token information for speculative decoding, and synchronization metadata. It answers: “What happened, and when is it safe for CPU logic to consume it?”

These are not copies of the same idea. They are contracts at successive ownership boundaries.

## 5. Model execution path

`TpModelWorker` represents one tensor-parallel worker’s execution interface. `ModelRunner` owns device initialization, memory pools, attention backend, graph handling, and model forward dispatch.

The conceptual path is:

```text
ScheduleBatch
→ ModelWorkerBatch
→ ForwardBatch
→ ModelRunner.forward
→ Qwen3ForCausalLM.forward
→ attention backend + MLP layers
→ logits
→ Sampler.forward
→ next_token_ids
→ GenerationBatchResult
```

Pinned anchors:

- [`TpModelWorker.forward_batch_generation`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/tp_worker.py#L529)
- [`ModelRunner.forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_executor/model_runner.py#L1232)
- [`Qwen3ForCausalLM.forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/qwen3.py#L512)
- [`Sampler.forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/sampler.py#L95)

## 6. First token versus later tokens

For the first token, the request goes through admission and prefill. The model processes prompt positions, creates KV state, produces logits for the generation position, and samples the first output token.

If the request is unfinished, the scheduler keeps its lifecycle state. On the next iteration, decode supplies the newly generated token/position while attention reads prior KV. After every result:

1. append output token IDs;
2. update logprob or grammar state if requested;
3. test EOS, stop strings, and maximum length;
4. stream eligible results;
5. keep unfinished requests running;
6. cache or release resources for finished requests.

The model does not directly “send a string.” Text appears only after token IDs return to the detokenization/frontend path.

## 7. Streaming and backpressure

Streaming shortens perceived latency but introduces flow-control boundaries. Potential accumulation points include:

- incoming requests waiting for scheduler admission;
- long prefills delaying decode work;
- CPU scheduling lagging GPU completion;
- GPU-to-CPU token copies and synchronization;
- detokenization or network clients consuming slowly.

When diagnosing latency, timestamp boundaries rather than blaming the nearest visible component. A slow first chunk can originate in queueing, prefill, sampling synchronization, detokenization, or transport.

## 8. Historical versus current architecture

The repository walkthrough uses SGLang v0.4.0. Current v0.5.16 has more features and moved code, but the durable abstractions remain recognizable: tokenization/control, scheduling, worker/model execution, attention backend, sampling, and detokenization.

This is the purpose of version archaeology. Do not memorize the old line number. Compare the responsibility:

- Did the same component retain it?
- Was it split into a mixin/helper?
- Did an optimization introduce a new result or synchronization field?
- Did a frontend path move while the backend contract remain stable?

If you can recover the lifecycle after a refactor, you understand the system rather than a snapshot.

## 9. A two-token trace

For a prompt that generates `Paris` and then `.`, narrate:

```text
1. Client messages become rendered text and input IDs.
2. Scheduler admits the request and selects prefill.
3. Worker/model processes prompt, writes KV, samples Paris.
4. Scheduler appends Paris, sees request unfinished, streams text.
5. Scheduler selects request for decode and reserves one KV slot.
6. Worker/model processes the new token position, samples period.
7. Scheduler appends period, evaluates stopping, and streams.
8. On EOS/stop, request state is removed and cache ownership is resolved.
```

That story is the backbone for Weeks 3–6.

## Checkpoint

Without notes, draw the components and answer:

1. Which component owns text/token conversion?
2. Which component decides the next batch?
3. Why do `ScheduleBatch` and `ForwardBatch` both exist?
4. Where are logits turned into token IDs?
5. Why can the first streamed token be slow even if decode ITL is good?

## Go deeper

- [SGLang Code Walk Through](../../sglang/code-walk-through/readme.md) traces the historical v0.4.0 request lifecycle.
- [Commit-pinned request lifecycle map](../CODE_MAP.md#request-lifecycle) pairs historical and current symbols.
- [SGLang architecture SVG](../../sglang/code-walk-through/sglang-architecture.svg) provides the detailed component diagram.
- Week 3 explains how the scheduler selects batches; Week 4 explains the cache state it manages.

