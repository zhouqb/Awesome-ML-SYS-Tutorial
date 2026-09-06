# Week 9 — Structured and Speculative Decoding

## Why this week matters

Ordinary decoding asks the model for one unconstrained token at a time. Two important serving techniques change that loop:

- **Structured decoding** restricts the legal next tokens so the output obeys a grammar, schema, or regular expression.
- **Speculative decoding** asks a cheaper process to propose several tokens, then verifies them with the target model in parallel.

They solve different problems—correct shape versus lower latency—but both live directly inside the token-generation loop. That makes them unusually sensitive to scheduler policy, batching, kernel shape, and synchronization overhead.

## Learning objectives

By the end of this week, you should be able to:

1. Explain how a grammar becomes a per-step token mask.
2. Distinguish regular constraints from constraints that require a stack-like parser.
3. Describe draft, verify, accept, and correction in speculative decoding.
4. Estimate when either technique helps or hurts a serving workload.
5. Identify the integration points between decoding logic and the scheduler.

## 1. Structured decoding is stateful masking

Suppose the model produces logits \(z \in \mathbb{R}^{|V|}\) over vocabulary \(V\). An unconstrained sampler converts those logits into a probability distribution. A constrained sampler first computes the set of legal tokens \(A(s)\) for parser state \(s\), then applies a mask:

```text
masked_logit[token] = logit[token]       if token in A(state)
                      -infinity          otherwise
```

After a token is chosen, the parser advances:

```text
state_next = transition(state, token)
```

The important point is that the grammar engine is not merely validating the final string. It participates in every decoding step.

### The tokenizer complication

Grammars operate on characters or bytes, while a model samples tokenizer tokens. One token may contain several characters, a partial word, whitespace, or escaped bytes. The constraint engine therefore needs a mapping from parser state to all tokenizer tokens whose byte sequences preserve a valid parse prefix.

This mapping can be expensive to compute repeatedly. Practical systems precompute token transitions, cache masks, compress accepted-token sets, or move mask application onto the GPU.

## 2. Finite state machines and pushdown behavior

A regular expression can usually be represented by a finite state machine (FSM). JSON-like formats are trickier because nested arrays and objects require remembering an unbounded nesting depth. That behavior resembles a pushdown automaton with a stack.

In practice, schema-constrained decoders often combine:

- finite-state transitions for local lexical rules;
- a parser stack for nested structure;
- semantic checks for schema fields, enums, or numeric ranges;
- tokenizer-aware transition tables.

The serving cost is not just parser execution. Different requests may be in different grammar states, producing different masks and making a batch less uniform.

## 3. Where constraints enter the serving loop

A simplified constrained decode step looks like this:

```python
logits = model.forward(batch)

for request, row in zip(batch.requests, logits):
    allowed = grammar_backend.allowed_tokens(request.grammar_state)
    row.masked_fill_(~allowed, float("-inf"))

tokens = sampler.sample(logits)

for request, token in zip(batch.requests, tokens):
    request.grammar_state = grammar_backend.advance(
        request.grammar_state, token
    )
```

Read this as a systems checklist:

1. Is mask generation on the CPU or GPU?
2. Does applying a different mask per row break a fused sampling path?
3. Can requests with the same grammar state share cached work?
4. What happens when a grammar admits no token?
5. Is parser state included when a request is paused, retracted, or migrated?

## 4. Structured decoding failure modes

### Correct but slow

The output is valid, but inter-token latency rises because mask construction or host-to-device transfer sits on the critical path.

### Fast but accidentally permissive

The implementation accepts token prefixes that later become impossible to complete. A final validator catches the error, but the serving system has already wasted generation work.

### Tokenizer mismatch

A grammar compiled against one tokenizer is reused with another. Token IDs may exist in both vocabularies while representing different byte strings.

### Empty legal set

An inconsistent schema, parser bug, or invalid prompt continuation leaves no legal token. Production code needs an explicit error path rather than sampling from all \(-\infty\) logits.

## 5. Speculative decoding: propose, verify, accept

Autoregressive target decoding normally performs one expensive target-model step per emitted token. Speculative decoding introduces a cheaper **draft** mechanism that proposes \(K\) tokens:

```text
draft:   d1, d2, d3, ... dK
target:  verify all K positions in one forward pass
result:  accept a prefix; correct at the first rejection
```

The target model remains authoritative. The draft changes how much target work can be parallelized, not the distribution we intend to sample from.

### Greedy intuition

For greedy decoding, accept consecutive draft tokens while each equals the target model's argmax. At the first mismatch, emit the target token and stop accepting the remaining proposal.

### Sampling intuition

For sampling, let \(q(x)\) be the draft distribution and \(p(x)\) the target distribution. A proposed token \(x\) is accepted with probability:

```text
alpha(x) = min(1, p(x) / q(x))
```

On rejection, the correction distribution is derived from the positive part of \(p-q\). This correction is what preserves exact target-model sampling rather than merely approximating it.

## 6. A useful speed model

Let:

- \(K\): number of proposed draft tokens;
- \(A\): number of accepted draft tokens;
- \(T_d(K)\): time to draft \(K\) tokens;
- \(T_v(K)\): time for the target to verify them;
- \(T_o\): orchestration and sampling overhead;
- \(T_t\): time for one ordinary target decode step.

A rough latency speedup is:

```text
speedup ≈ E[A + 1] * T_t / (T_d(K) + T_v(K) + T_o)
```

The extra \(+1\) represents the target token produced at the verification boundary. This model is deliberately incomplete, but it exposes the central tradeoff: a larger \(K\) creates more possible accepted work while making drafting and verification more expensive.

### Worked example

Assume:

- ordinary target decode: 8 ms per token;
- draft 4 tokens: 5 ms;
- target verification: 10 ms;
- orchestration: 1 ms;
- mean accepted draft prefix: 2.5 tokens.

Then:

```text
ordinary time for 3.5 tokens = 3.5 * 8 = 28 ms
speculative iteration time  = 5 + 10 + 1 = 16 ms
estimated speedup           = 28 / 16 = 1.75x
```

If the mean accepted prefix falls to 0.5, the same system becomes slower:

```text
speedup = 1.5 * 8 / 16 = 0.75x
```

Acceptance rate alone is not enough; you must include draft, verify, and orchestration costs.

## 7. Draft mechanisms

The draft source does not have to be a smaller independent model. Common families include:

- a smaller model trained or selected to match the target;
- extra prediction heads attached to the target model;
- prompt or prefix lookup that proposes repeated n-grams;
- tree-based proposals that verify multiple candidate continuations;
- feature-level methods that reuse intermediate target representations.

Each family trades model memory, implementation complexity, acceptance rate, and batchability differently.

## 8. Scheduler interactions

Speculative decoding changes a decode request from “one token of work” into a variable-sized verification job. That affects:

- token-budget accounting;
- KV-cache reservation for proposed positions;
- rollback after rejected tokens;
- CUDA graph shapes;
- fairness between ordinary and speculative requests;
- batching requests with different proposal lengths.

Structured decoding also adds request-local state and variable sampling work. Combining both techniques means the verifier must apply grammar legality while determining which proposed tokens can be accepted.

## 9. Read the repository in this order

Use GitHub search within the fork because these features evolve quickly across SGLang versions:

1. Search for `grammar`, `xgrammar`, and `constraint` to locate structured-output integration.
2. Search for `speculative`, `draft`, `verify`, and `accept_length` to find proposal and verification code.
3. Trace fields from request objects into schedule batches and then into the model worker.
4. Find where KV slots are committed or released after verification.
5. Find metrics that expose acceptance length, grammar overhead, or speculative throughput.

Repository entry points:

- [SGLang source tree](../../sglang/)
- [SGLang scheduler evolution](../../sglang/scheduler-evolution/SGLang Scheduler Evolution.md)
- [Course code map](../CODE_MAP.md)
- [Course SGLang primer](../SGLANG_PRIMER.md)

## 10. Measurement plan

For structured decoding, compare:

- unconstrained generation;
- a small enum or regular expression;
- a shallow JSON schema;
- a deeply nested schema.

Measure TTFT, inter-token latency, throughput, CPU time, GPU utilization, and invalid-output rate.

For speculative decoding, sweep:

- proposal length \(K\);
- prompt and output lengths;
- batch size;
- draft/target model pairing;
- temperature and sampling settings.

Always report mean accepted length alongside end-to-end latency and throughput.

## Checkpoint

You are ready to continue when you can answer these without notes:

1. Why is final JSON validation not equivalent to structured decoding?
2. Why does tokenization make grammar masking harder than character parsing?
3. What makes speculative sampling exact rather than approximate?
4. Give one workload where speculative decoding will probably lose.
5. Name three scheduler resources affected by speculative verification.

## Optional references

- [SGLang documentation](https://docs.sglang.ai/)
- [XGrammar](https://github.com/mlc-ai/xgrammar)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
