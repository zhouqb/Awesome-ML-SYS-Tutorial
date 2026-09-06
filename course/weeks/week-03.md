# Week 3 — Continuous Batching and the Scheduler

Reading time: 45–55 minutes. The scheduler is where irregular user traffic becomes efficient GPU work. It balances throughput, TTFT, ITL, fairness, and KV-memory capacity.

## Learning objectives

After this lesson, you should be able to:

- explain continuous batching and why batch membership changes;
- track requests through waiting, prefill, and decode states;
- distinguish token budget from request count;
- explain chunked prefill and decode retraction;
- read the normal scheduler loop before studying overlap optimizations.

## 1. The scheduling problem

Requests differ in arrival time, prompt length, output length, cached prefix, and sampling/grammar requirements. A static batch waits for every member to finish, wasting capacity when short outputs end early. Continuous batching rebuilds the runnable batch each iteration.

The scheduler must answer:

1. Which new requests should begin or continue prefill?
2. Which running requests should decode one more token?
3. Does the selected work fit the token and KV-cache budgets?
4. Which result/state updates are needed before the next iteration?

The output of scheduling is not text. It is a batch description and a reservation of execution/cache resources.

## 2. The four control states to learn first

Use plain-language names before source names:

- **Waiting (`waiting_queue`)**: eligible work not executing ordinary decode. It includes new requests and work returned after retraction.
- **Selected prefill (`new_batch`)**: prompt/extend work chosen for this iteration.
- **Running decode (`running_batch`)**: requests whose prompts are ready and which can generate another token.
- **Current execution (`cur_batch`)**: the batch actually passed to `run_batch` now.

`last_batch` represents the preceding iteration’s selected batch when its state still matters. A long prompt split by chunked prefill can also be held as `chunked_req` between chunks.

For a normal transition:

```text
new request
→ waiting_queue
→ new_batch (prefill/extend)
→ last_batch
→ running_batch (decode)
→ finished/cache/release
```

Running decode can loop many times:

```text
running_batch → cur_batch → result update → running_batch
```

## 3. Prefill-first selection

The simplified policy in the repository’s scheduler article gives prefill priority:

```python
if last_batch_was_extend:
    running_batch.merge(last_batch)

new_batch = get_new_batch_prefill()
if new_batch:
    return new_batch
return update_running_batch(running_batch)
```

This is illustrative pseudocode, not a substitute for current source. The key decision is that new prefill work can take the current iteration; decode runs when no new prefill batch is selected.

Why prioritize prefill? New requests cannot receive a first token until their prompt state exists. But an unbounded long prefill can delay tokens for already-running users. That conflict motivates chunking and token budgets.

## 4. Token budget, not merely batch size

A request count is a poor measure of prefill work. One request may contribute 20 tokens and another 20,000. A scheduler therefore tracks how many new tokens it can process in an iteration.

Conceptually:

```text
selected new tokens
= sum(uncached prompt/extend tokens admitted this iteration)
≤ prefill token budget
```

Decode usually contributes one new token per running request per step, although speculative methods can accept multiple.

The scheduler also needs space for each selected token’s KV state. A batch can fit a compute token budget but fail the cache-capacity constraint.

## 5. Chunked prefill

Chunked prefill splits one long prompt across multiple iterations:

```text
12,000 uncached prompt tokens, chunk size 4,000
→ iteration A: positions 0–3,999
→ iteration B: positions 4,000–7,999
→ iteration C: positions 8,000–11,999
→ then ordinary decode
```

Benefits:

- bounds one iteration’s prefill work;
- prevents a single long prompt from monopolizing the GPU for too long;
- provides more scheduling opportunities between chunks;
- helps fit temporary activation/workspace constraints.

Costs:

- more scheduling and launch boundaries;
- potentially lower prefill efficiency for chunks that are too small;
- intermediate state is more complex because prefill has started but decode is not ready;
- policy choices determine fairness between chunks and other requests.

Chunk size is therefore a latency/throughput trade-off, not a universal constant.

## 6. Decode memory pressure and retraction

Every unfinished decode request normally needs space for its next token. If the active batch cannot reserve sufficient KV slots, the scheduler can retract requests from decode back to waiting state.

Retraction preserves system progress but damages the retracted request’s latency and may require recomputation depending on retained cache state. A sensible victim policy considers priority, generated length, and recomputation cost.

The important distinction is:

- **GPU OOM** is an allocator/runtime failure.
- **KV admission failure** is a scheduling condition that a well-designed engine can handle by delaying/retracting work.

## 7. A worked scheduling example

Assume an iteration budget of five new tokens:

- A arrives at iteration 0 with six prompt tokens and needs four output tokens.
- B arrives at iteration 1 with two prompt tokens and needs three output tokens.
- C arrives at iteration 2 with seven prompt tokens and needs two output tokens.

Under deterministic prefill-first policy with chunking:

```text
Iteration 0: prefill first 5 tokens of A
Iteration 1: finish A prefill (1); use remaining 4 budget for B prefill (2)
             A and B become decode-ready after their respective prefill results
Iteration 2: C has waiting prefill; select first 5 tokens of C
Iteration 3: finish C prefill (2)
Iteration 4: decode A, B, C (one token each)
Iteration 5: decode A, B, C
Iteration 6: C completes; decode A, B
Iteration 7: B completes; decode A
Iteration 8: A completes
```

This deliberately simple policy produces poor ITL for A because later prefills take priority. An overlap-capable or mixed policy may schedule differently. The exercise is to expose the trade-off: prefill-first improves new-request readiness while delaying established decode.

## 8. Read the normal loop in source

Use current pinned SGLang in this order:

1. [`Scheduler.event_loop_normal`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L1520): locate one full iteration.
2. [`process_input_requests`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L1663): see how incoming types reach handlers.
3. [`get_next_batch_to_run`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L2687): identify merge, prefill selection, and decode update.
4. [`run_batch`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L3290): find the worker boundary.
5. [`process_batch_result`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L3562): locate mode-specific result handling.

On the first pass, ignore speculative decoding and overlap branches. Write one sentence for the state mutated by each call.

## 9. What to measure

Scheduling changes multiple metrics simultaneously:

- TTFT by controlling prefill admission;
- ITL by deciding how often running requests decode;
- throughput by forming efficient batches;
- fairness through priority and victim selection;
- maximum live workload through KV allocation.

A scheduler benchmark must report the arrival process and prompt/output length distribution. Fixed-concurrency synthetic traffic and open-loop arrival traffic answer different questions.

## Checkpoint

You understand the scheduler’s big picture if you can answer:

1. Why is `running_batch` not the batch currently executing in every iteration?
2. What state holds a partially prefetched long request?
3. Why is a token budget more meaningful than request count?
4. How can chunked prefill improve ITL but reduce prefill efficiency?
5. Why is retraction preferable to an uncontrolled OOM?

## Go deeper

- [From KV Cache to Zero Overhead Scheduling](../../sglang/scheduler/readme-en.md) is the main deep dive.
- [SGLang Before the Scheduler](../SGLANG_PRIMER.md) provides the name translation and three-pass strategy.
- [Lab 3 scheduler simulation](../LABS.md#lab-3--hand-simulate-continuous-batching) forces an explicit policy.
- Week 4 explains the cache pools and radix state used by these decisions.

