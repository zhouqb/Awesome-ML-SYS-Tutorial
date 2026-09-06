# Week 8 — Quantization as a Serving Lifecycle

Reading time: 45–55 minutes. Quantization is not “cast weights to fewer bits.” A serving framework must understand checkpoint metadata, create correctly shaped storage, load and shard values, transform/pack them, select compatible kernels, and protect numerical quality.

## Learning objectives

After this lesson, you should be able to:

- distinguish weight, activation, and KV-cache quantization;
- explain scale granularity and low-precision range trade-offs;
- trace `create_weights → process_weights_after_loading → apply`;
- identify integration requirements for a new format;
- design correctness, compatibility, memory, and performance tests.

## 1. Quantization model

A common affine/symmetric approximation maps a real tensor `x` to quantized values `q` using scale `s`:

```text
q = clamp(round(x / s), q_min, q_max)
x̂ = s × q
```

The scale trades range against resolution. If `s` is too small, values saturate; if too large, representable steps are coarse.

Floating-point formats such as FP8 encode exponent/mantissa directly but still commonly use external scaling to fit tensor distributions into useful range.

## 2. What can be quantized?

### Weight-only

Weights use fewer bits; activations remain higher precision. This reduces model storage and weight bandwidth. Kernels may dequantize during matrix multiplication or use native low-precision paths.

### Weight-and-activation

Both operands use reduced precision. This can unlock faster tensor-core kernels and reduce traffic, but activation distributions vary by input and require runtime or calibrated scaling.

### KV cache

Stored keys/values use fewer bytes. This increases context/concurrency capacity and reduces attention bandwidth. It affects a repeatedly read state, so accumulated numerical effects and kernel support matter.

### Accumulation/output precision

Low-precision inputs often accumulate in FP16/BF16/FP32-like precision. Input storage format and accumulation format must be stated separately.

## 3. Scale granularity

- **Per-tensor:** one scale; cheap metadata and simple kernels, but outliers determine range for everything.
- **Per-channel:** separate scales along an output/input channel; better fit, more metadata and kernel complexity.
- **Per-group/block:** scales for fixed blocks; balances locality, quality, and hardware-friendly packing.
- **Dynamic/token-wise:** scales depend on runtime activations; adapts to inputs but adds reduction/conversion overhead.

Granularity must align with TP sharding and kernel tile layout. A scale block crossing rank or packing boundaries complicates loading and execution.

## 4. Why three lifecycle phases exist

SGLang’s quantization method abstraction separates:

### `create_weights`

Runs while constructing layers. It declares parameter shapes, storage dtypes, scale tensors, packed dimensions, and attributes needed by loading.

### `process_weights_after_loading`

Runs after checkpoint values exist. It can repack, transpose, fuse scales, quantize-on-load, validate shapes, and allocate kernel-specific metadata.

### `apply`

Runs on every forward. It chooses/executes the compatible low-precision kernel and returns outputs in the expected dtype/layout.

The separation follows information availability: construction knows intended structure; post-load knows values; apply sees runtime activations and execution context.

## 5. End-to-end loading flow

```text
server/model config
→ detect quantization config
→ construct model and quantized parameter placeholders
→ iterate checkpoint tensors
→ map names and TP shards
→ load weights/scales
→ post-load packing/transformation
→ warmup/select kernel path
→ apply during forward
```

A mismatch at each phase has a different symptom:

- config detection: wrong method or unsupported hardware;
- creation: shape/storage errors;
- checkpoint mapping: missing or misplaced weights/scales;
- post-load: silent layout/scale corruption or wasted memory;
- apply: slow fallback, kernel error, or numerical failure.

## 6. Pinned source walkthrough

Start at [`QuantizeMethodBase`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/base_config.py#L20) and [`QuantizationConfig`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/base_config.py#L126).

Then follow FP8:

1. [`Fp8Config`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L220): hardware/dtype/config rules.
2. [`Fp8LinearMethod.create_weights`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L504): weight and scale representation.
3. [`process_weights_after_loading`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L781): value-dependent transforms.
4. [`apply`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L893): runtime dispatch.

For each function, list inputs, newly created state, invariants, and fallback behavior.

## 7. Worked design: W4A8Block

Suppose a new format stores 4-bit block-scaled weights and uses FP8 activations.

It needs:

- config keys for block size, weight format, activation format, and supported layers;
- packed weight parameters plus scale tensors;
- shard alignment rules so TP partitions do not split unsupported blocks;
- checkpoint mapping for prequantized and quantize-on-load cases;
- post-load packing into the target GEMM layout;
- runtime activation scaling and compatible kernel selection;
- a higher-precision fallback for unsupported shapes;
- explicit behavior for embeddings, output head, MoE experts, and tied weights.

The design is incomplete until it states supported GPU capability and activation dtype.

## 8. Accuracy and performance validation

Use four layers of tests:

1. **Metadata/unit:** config parsing, parameter shapes, scale/block mapping.
2. **Loading/sharding:** known tensors load identically across TP partitions and reconstruct correctly.
3. **Numerical:** compare layer/model outputs against BF16 across representative and adversarial distributions; report tolerances and task metrics.
4. **System:** measure load memory, steady memory, TTFT, ITL, throughput, warmup/JIT time, and fallback frequency.

One exact generated string is a weak numerical test: sampling can hide logit differences, and identical greedy tokens can coexist with large distribution error.

## 9. Memory arithmetic

Ideal raw weight compression from BF16 to 4-bit is:

```text
2 bytes / 0.5 bytes = 4×
```

Actual savings are smaller due to scales, zero points, packing alignment, unquantized layers, and runtime transformed copies. If both packed and unpacked weights remain resident, the implementation may save little despite a compact checkpoint.

Always measure peak during loading as well as steady state; post-load repacking can temporarily hold both representations.

## Checkpoint

Explain without notes:

1. Why per-tensor scales struggle with outliers.
2. Why creation cannot perform every post-load transform.
3. How TP shard alignment constrains block quantization.
4. Why a smaller checkpoint does not guarantee lower steady GPU memory.
5. What tests distinguish load correctness, numerical quality, and speed.

## Go deeper

- [General quantization background](../../sglang/quantization/general_background.md).
- [SGLang quantization architecture](../../sglang/quantization/quantization_architecture_en.md).
- [Commit-pinned quantization lifecycle](../CODE_MAP.md#quantization-lifecycle).
- [Lab 8 integration review](../LABS.md#lab-8--quantization-integration-review).

