# Week 4 — KV Cache as a Memory System

Reading time: 45–55 minutes. KV cache is both the mechanism that makes autoregressive decode practical and the scarce dynamic resource that shapes admission, scheduling, and long-context capacity.

## Learning objectives

After this lesson, you should be able to:

- derive KV size for MHA, GQA, and reduced-precision cache;
- separate logical request/token positions from physical KV slots;
- explain page allocation and radix prefix reuse;
- trace prefix match, cache insertion, reference protection, eviction, and reclamation;
- diagnose startup versus runtime memory pressure.

## 1. Why cache K and V?

At layer `l`, self-attention for a new query needs keys and values from all retained earlier positions. Without caching, generating token `t` would recompute K/V for positions `1…t-1` at every layer. KV cache stores those projected tensors once.

For one new token:

```text
Q_new attends to [K_1, K_2, …, K_new]
and combines     [V_1, V_2, …, V_new]
```

Only `K_new` and `V_new` are appended, while earlier entries are read. This converts repeated full-prefix projection into incremental decode, at the cost of memory that grows with sequence length.

## 2. Capacity equation

For a conventional attention layer:

```text
KV bytes/token
= layers × 2(K,V) × KV heads × head dimension × bytes/element
```

Total live payload is this value times retained tokens across all requests, after accounting for shared prefix entries only once where the implementation truly reuses them.

Example: 32 layers, 8 KV heads, head dimension 128, BF16:

```text
32 × 2 × 8 × 128 × 2 = 131,072 bytes = 128 KiB/token
```

At 100,000 retained tokens, raw payload is about `12.2 GiB`. This is why “number of concurrent requests” is incomplete; total retained tokens and architecture determine memory.

## 3. Logical positions versus physical slots

An engine should not require each request’s cache to occupy one growing contiguous allocation. Instead it uses indirection:

```text
request slot + sequence position
        ↓ ReqToTokenPool mapping
physical KV slot index
        ↓ TokenToKVPool storage
K/V tensors for every layer
```

`ReqToTokenPool` answers: “For request R at logical position p, which physical KV slot holds the data?”

The token-to-KV pool answers: “At physical slot s, where are this token’s K/V tensors for each layer?”

This resembles virtual memory: logical sequence layout is decoupled from physical placement. Requests can grow without relocating all previous KV tensors.

## 4. Page allocation and slack

Physical slots are commonly managed in pages/blocks. If page size is `B` tokens, a sequence of length `T` needs:

```text
pages(T) = ceil(T / B)
allocated token slots = B × ceil(T / B)
internal slack = allocated slots - T
```

For page size 16 and sequences of 17, 31, and 33 tokens:

```text
17 → 2 pages → 32 slots → 15 slack
31 → 2 pages → 32 slots → 1 slack
33 → 3 pages → 48 slots → 15 slack
```

Across many short or uneven sequences, slack and fragmentation explain why measured capacity is below raw token arithmetic.

## 5. Prefix reuse with a radix tree

Requests often share system prompts, few-shot examples, or conversation prefixes. Radix caching indexes token sequences so a new request can reuse previously computed KV state.

Suppose:

```text
R1: [10, 11, 12, 20, 21]
R2: [10, 11, 12, 30]
R3: [10, 11, 40]
```

After R1 is cached, R2 can match `[10,11,12]` and prefill only `[30]`. R3 can match `[10,11]` and compute `[40]`. The radix tree represents shared token prefixes; its values reference physical KV slots.

Important: a prefix-cache entry is not merely text. Tokenization, model revision, adapter/weights, and relevant execution identity must be compatible. A textually identical prefix with different token IDs or model state cannot safely share KV.

## 6. Ownership and eviction

Cache management needs a distinction between “present” and “evictable.” A cached prefix may be protected because an active request references it. Reference/lock counts prevent those slots from being reclaimed.

A simplified lifecycle is:

```text
match prefix
→ protect matched nodes/slots
→ allocate slots for uncached suffix
→ execute and append KV
→ request finishes
→ insert/merge reusable prefix
→ release active reference
→ later evict unreferenced entries under pressure
```

Eviction policy often favors least-recently-used unprotected leaves, but exact behavior belongs to the implementation. The invariant is stronger: never reuse a physical slot while active computation can still read it.

## 7. Current SGLang source path

Read these pinned anchors:

1. [`ReqToTokenPool`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L244), especially [`alloc`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L279) and [`free`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L307).
2. [`MHATokenToKVPool`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L1683): identify layer/slot storage.
3. [`RadixCache.match_prefix`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/radix_cache.py#L355): record returned match information and reference changes.
4. [`RadixCache.cache_finished_req`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/radix_cache.py#L437): see how finished-request state becomes reusable cache state.

Draw two columns: logical metadata on CPU/control side and physical tensors on GPU. Place each field you encounter into one column.

## 8. MHA versus MLA

The earlier formula assumes explicit K and V heads. Multi-head latent attention (MLA) stores a compressed latent representation and reconstructs/provides the needed attention inputs differently. The capacity formula must follow actual stored state, not blindly reuse `2 × kv_heads × head_dim`.

SGLang therefore has a distinct pinned [`MLATokenToKVPool`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L3800). The lesson is general: derive memory from the architecture and concrete pool layout.

## 9. OOM diagnosis

Using `mem_fraction_static = y`:

```text
memory_for_kv
= y × pre_model_load_memory
  - measured model/load allocation
```

Two failures point in opposite directions:

- **Startup KV-pool allocation failure:** the static region cannot contain weights plus a viable KV pool; increasing `y` may help.
- **Runtime dynamic OOM:** activations, graph pools, or other dynamic work exceed reserved headroom; decreasing `y` may help.

Other processes or uneven tensor-parallel ranks can invalidate a clean calculation. Measure free memory per rank; do not average away the tightest GPU.

## Checkpoint

Explain without notes:

1. Why token position is not a physical KV address.
2. The raw BF16 KV bytes/token for the worked configuration.
3. Why prefix reuse saves both compute and memory.
4. Why cached data can be present but not evictable.
5. Why increasing static-memory fraction can fix one OOM and cause another.

## Go deeper

- [When SGLang OOMs](../../sglang/kvcache-code-walk-through/mem-fraction-static-en.md).
- [SGLang scheduler: KV-cache management](../../sglang/scheduler/readme-en.md#kv-cache-management).
- Optional and marked Pending Review: [KV Cache Code Walkthrough](../../sglang/kvcache-code-walk-through/readme.md); verify claims against the pinned source above.
- [Lab 4](../LABS.md#lab-4--kv-cache-ledger-and-lifecycle) asks you to draw the mapping and memory ledger.

