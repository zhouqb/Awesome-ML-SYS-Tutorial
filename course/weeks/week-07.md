# Week 7 — Parallel Inference and Communication

Reading time: 45–60 minutes. Multiple GPUs solve model-fit or latency/throughput constraints by partitioning computation and state. Every partition introduces communication and topology assumptions.

## Learning objectives

After this lesson, you should be able to:

- distinguish data, tensor, pipeline, and expert parallelism;
- derive row- and column-parallel linear layers;
- map data movement to NCCL collectives;
- reason about topology-aware placement;
- explain why adding GPUs can make a small workload slower.

## 1. Four kinds of parallelism

### Data parallelism (DP)

Each replica holds the model and serves different requests. For inference, replicas normally do not AllReduce activations; a router distributes traffic. DP increases aggregate capacity but does not help one model fit if a replica is too large.

### Tensor parallelism (TP)

Each layer’s tensors/computation are partitioned across ranks. TP reduces per-GPU weight/KV burden and can reduce one request’s compute time, but introduces communication inside many layers.

### Pipeline parallelism (PP)

Consecutive layer groups reside on different stages. Activations move stage to stage. PP helps model fit across nodes but introduces pipeline bubbles and per-request stage latency.

### Expert parallelism (EP)

MoE experts are distributed. Tokens are routed to owning ranks, typically requiring AllToAll-style communication. Load imbalance becomes as important as total FLOPs.

These dimensions can be combined: DP replicas, each using TP within a fast interconnect domain, with PP across nodes and EP for expert layers.

## 2. Column-parallel linear algebra

For `Y = XW`, let `X` have shape `[M, K]` and `W` have `[K, N]`. With TP size `p`, column partitioning splits output features:

```text
W = [W₁ W₂ ... Wₚ], each Wᵣ: [K, N/p]
Yᵣ = X Wᵣ, each Yᵣ: [M, N/p]
```

If the next operation can consume partitioned output, no immediate gather is required. Otherwise ranks AllGather `Yᵣ` to reconstruct `[M, N]`.

## 3. Row-parallel linear algebra

Row partitioning splits input features:

```text
X = [X₁ X₂ ... Xₚ], each Xᵣ: [M, K/p]
W vertically split, Wᵣ: [K/p, N]
partial Yᵣ = Xᵣ Wᵣ: [M, N]
Y = sumᵣ partial Yᵣ
```

Ranks must sum partial outputs, usually with AllReduce (replicated result) or ReduceScatter (partitioned result).

Transformer TP pairs column- and row-parallel projections so one layer’s partitioned output becomes the next operation’s expected input, minimizing unnecessary gathers.

## 4. Concrete dimensions

Let `M=32` tokens, `K=4096`, `N=11008`, TP=4, BF16.

Full weight elements:

```text
4096 × 11008 = 45,088,768 elements
≈ 86 MiB in BF16
```

Each rank stores one quarter, about `21.5 MiB` raw weight.

Column-parallel local output is `[32, 2752]`, or `88,064` BF16 elements (`172 KiB`). An AllGather would reconstruct four shards.

Row-parallel each rank computes a partial `[32,11008]`, `352,256` BF16 elements (`688 KiB`), which must be reduced. The weight saving is clear, but communication repeats for every invocation/layer boundary that needs it.

## 5. Collective operations by meaning

- **AllReduce:** sum/reduce contributions and give the full result to every rank.
- **AllGather:** collect distinct shards and give the concatenated collection to every rank.
- **ReduceScatter:** reduce contributions while leaving each rank only its result shard.
- **Broadcast:** send one rank’s value to all ranks.
- **AllToAll:** each rank sends distinct partitions to every other rank; central to expert/token routing.
- **Send/Recv:** explicit point-to-point transfer, useful for pipeline stages.

Choose a collective from the desired data layout after communication, not from habit.

## 6. Communication cost model

A rough collective time is:

```text
T_comm ≈ latency_term + bytes_transferred / effective_bandwidth
```

The latency term dominates small messages; bandwidth dominates large messages. Effective bandwidth depends on algorithm, topology, contention, protocol, and whether transfer overlaps compute.

For ring AllReduce, each rank conceptually participates in reduce-scatter plus all-gather phases. The exact traffic factor depends on implementation, but it approaches a multiple of tensor bytes that grows toward `2(p-1)/p` per rank. This is why small decode batches can be communication-latency sensitive even if tensors are not huge.

## 7. Topology-aware placement

Read `nvidia-smi topo -m` as a cost map:

- NVLink/NVSwitch paths generally provide higher GPU-GPU bandwidth and lower latency.
- PCIe paths crossing a host bridge cost more.
- `SYS` paths crossing CPU sockets/NUMA interconnect are usually less desirable for chatty TP collectives.
- Inter-node networking is commonly slower/higher latency than intra-node fabrics.

A typical policy is to keep TP within the fastest connected group, use DP across independent groups, and use PP across slower boundaries when model fit demands it. MoE/EP placement additionally considers expert routing volume and load balance.

## 8. DP attention and KV duplication

For architectures with very few KV heads or a latent KV representation, ordinary TP can duplicate KV state across ranks because the attention partition has less head-level state to divide. DP attention approaches distribute request batches across attention ranks while using TP for other model parts, reducing duplicated KV capacity.

This saves memory but adds layout transitions/communication and scheduling constraints. The correct comparison is capacity plus end-to-end latency—not KV bytes alone.

The repository’s DP Attention article is marked Pending Review, so treat it as a hypothesis guide and verify against current source/design.

## 9. Why more GPUs can be slower

At small batch/concurrency:

- local matrix multiplications shrink;
- collective latency does not shrink proportionally;
- synchronization exposes the slowest rank;
- CPU/process overhead increases;
- communication may interrupt kernel fusion or graph paths.

More GPUs help when weight/model fit, KV capacity, or sufficiently large computation outweighs those costs. Always sweep batch/concurrency rather than report one TP point.

## Checkpoint

Explain without notes:

1. Why DP improves server capacity but not single-replica model fit.
2. The output shape and communication need of column-parallel linear.
3. Why row-parallel partial outputs require reduction.
4. Why TP should normally remain inside the fastest fabric domain.
5. Why MLA/few-KV-head models motivate alternative attention parallelism.

## Go deeper

- [NCCL in SGLang](../../torch/nccl/readme_en.md).
- [PyTorch Distributed](../../torch/torch-distributed/readme.md).
- Optional and marked Pending Review: [Tensor Parallelism from scratch](../../torch/tensor-parallelism/readme.md) and [DP Attention](../../sglang/dp-attention/readme.md).
- [Lab 7 parallel deployment design](../LABS.md#lab-7--parallel-deployment-design).

