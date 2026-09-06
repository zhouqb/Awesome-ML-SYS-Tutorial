# Commit-Pinned SGLang Code Map

This is the course’s source-reading spine. Every external source link is pinned to a commit so the symbol and line number do not drift.

Two snapshots are intentionally used:

- **Historical model:** SGLang v0.4.0, commit [`f8b0326934bacb7a7d4eba68fb6eddebaa6ff751`](https://github.com/sgl-project/sglang/tree/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751). It matches the repository’s end-to-end walkthrough.
- **Current lab model:** SGLang v0.5.16, commit [`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`](https://github.com/sgl-project/sglang/tree/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1). It is the reproducible implementation used by the course.

Do not try to read every line. For each symbol, answer four questions:

1. Who calls it?
2. What state does it read or mutate?
3. What object or tensor crosses the next boundary?
4. Which performance resource does it control: compute, memory capacity, memory bandwidth, communication, launch overhead, or CPU time?

## Request lifecycle

| Stage | Historical anchor used by the article | Current lab anchor | Reading question |
|---|---|---|---|
| HTTP entry | [`server.py: launch_server`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/server.py#L507) | [`http_server.py: generate_request`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/entrypoints/http_server.py#L833) | What work belongs to the protocol layer, and what is delegated? |
| Tokenization/control | [`TokenizerManager`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tokenizer_manager.py#L88) | [`TokenizerManager`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/tokenizer_manager.py#L265) and [`generate_request`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/tokenizer_manager.py#L624) | Which request fields exist before and after tokenization? |
| Scheduler construction | [`Scheduler`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L97) | [`Scheduler`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L303) | Which policies, pools, and workers are initialized here? |
| Scheduler loop | [`event_loop_normal`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L376) | [`event_loop_normal`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L1520) | Which steps repeat once per scheduling iteration? |
| Request admission | [`process_input_requests`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L508) | [`process_input_requests`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L1663) | How are request types dispatched and generation requests admitted? |
| Batch selection | [`get_next_batch_to_run`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L768) | [`get_next_batch_to_run`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L2687) | Where do prefill priority, memory availability, and running requests meet? |
| Batch execution | [`run_batch`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L956) | [`run_batch`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L3290) | What is the boundary between scheduling and device execution? |
| Result/state update | [`process_batch_result`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/scheduler.py#L987) | [`process_batch_result`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/scheduler.py#L3562) | What makes a request finished, cached, requeued, or streamed? |

## Model execution and sampling

| Mechanism | Historical anchor | Current lab anchor | Reading question |
|---|---|---|---|
| TP worker | [`TpModelWorker`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/managers/tp_worker.py#L40) | [`TpModelWorker`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/tp_worker.py#L273) and [`forward_batch_generation`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/tp_worker.py#L529) | Which work happens once per rank, and which outputs must agree across ranks? |
| Model runner | [`ModelRunner`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/model_executor/model_runner.py#L66) | [`ModelRunner`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_executor/model_runner.py#L235) and [`forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_executor/model_runner.py#L1232) | How does forward mode select the execution path? |
| Model body | [`Qwen2ForCausalLM.forward`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/models/qwen2.py#L289) | [`Qwen3ForCausalLM`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/qwen3.py#L452), [`forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/qwen3.py#L512), and [`load_weights`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/qwen3.py#L596) | Where do embeddings become hidden states, and hidden states become logits? |
| Sampling | Historical flow is summarized in the [walkthrough](../sglang/code-walk-through/readme.md#tpmodelworker-manage-forward-and-token-sampling) | [`Sampler.forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/sampler.py#L95) | Which transforms happen before multinomial/greedy choice, and how is TP consistency maintained? |

## Model loading

| Mechanism | Current lab anchor | Reading question |
|---|---|---|
| Loader abstraction | [`BaseModelLoader`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_loader/loader.py#L333) | Which responsibilities are common across load formats? |
| Default loader | [`DefaultModelLoader`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_loader/loader.py#L355) and [`load_model`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_loader/loader.py#L772) | In what order are model construction, checkpoint iteration, device placement, and post-processing performed? |
| Quantization config selection | [`_get_quantization_config`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_loader/loader.py#L198) | Which config comes from the checkpoint and which from server arguments? |

## KV-cache memory and prefix reuse

| Mechanism | Current lab anchor | Reading question |
|---|---|---|
| Logical request map | [`ReqToTokenPool`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L244), [`alloc`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L279), and [`free`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L307) | Why is a request slot different from a KV slot? |
| MHA physical KV | [`MHATokenToKVPool`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L1683) | Which dimensions determine storage, and where can dtype/layout vary? |
| MLA physical KV | [`MLATokenToKVPool`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/memory_pool.py#L3800) | How does the latent representation change the memory account? |
| Prefix lookup | [`RadixCache`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/radix_cache.py#L280) and [`match_prefix`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/radix_cache.py#L355) | What is returned by a prefix match besides a length? |
| Cache insertion | [`cache_finished_req`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/mem_cache/radix_cache.py#L437) | Which ownership/reference changes allow finished-request KV to survive? |

## Attention backend

| Mechanism | Historical anchor | Current lab anchor | Reading question |
|---|---|---|---|
| Backend construction | [`FlashInferBackend`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L48) | [`FlashInferAttnBackend`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/flashinfer_backend.py#L300) | Which workspaces and wrappers persist across requests? |
| Forward metadata | [`init_forward_metadata`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L130) | [`init_forward_metadata`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/flashinfer_backend.py#L916) | Which metadata depends on batch shape and cache indices? |
| Extend/prefill path | [`forward_extend`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L223) | [`forward_extend`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/flashinfer_backend.py#L1252) | When is ragged work preferable to paged work? |
| Decode path | [`forward_decode`](https://github.com/sgl-project/sglang/blob/f8b0326934bacb7a7d4eba68fb6eddebaa6ff751/python/sglang/srt/layers/attention/flashinfer_backend.py#L277) | [`forward_decode`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/attention/flashinfer_backend.py#L1410) | What is reused from previous steps, and what changes for the next token? |

## Quantization lifecycle

| Mechanism | Current lab anchor | Reading question |
|---|---|---|
| Method contract | [`QuantizeMethodBase`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/base_config.py#L20) | Why are creation, post-load processing, and application separate phases? |
| Config contract | [`QuantizationConfig`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/base_config.py#L126) | How does a model layer obtain its quantization method? |
| FP8 configuration | [`Fp8Config`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L220) | Which hardware, activation dtype, and checkpoint constraints are checked? |
| FP8 linear weights | [`Fp8LinearMethod.create_weights`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L504) | What extra scale/metadata parameters are created? |
| FP8 post-load | [`Fp8LinearMethod.process_weights_after_loading`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L781) | What transforms cannot be performed until checkpoint values exist? |
| FP8 execution | [`Fp8LinearMethod.apply`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L893) | What selects the optimized kernel versus a fallback? |

## CUDA Graph case study

The core CUDA Graph article analyzes SGLang-Omni commit [`cd9aaf3`](https://github.com/sgl-project/sglang-omni/tree/cd9aaf3). Use these links as one coherent case study:

- [`create_s2pro_sglang_engine`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/factory.py): delayed capture and initialization ordering.
- [`setup_vq_decode`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/sglang_model.py#L196): persistent buffer allocation.
- [`S2ProSGLangModelRunner`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py): write-before-replay and read-after-replay protocol.
- [`forward`](https://github.com/sgl-project/sglang-omni/blob/cd9aaf3/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py#L919): the forward boundary pulled inside the graph.

Reading question: which values may change between replays, which addresses must not change, and which control-flow choices have been frozen?

## Version-archaeology worksheet

For each week, pick one historical/current pair and fill this table:

| Field | Your observation |
|---|---|
| Symbol pair | |
| Same responsibility preserved | |
| Responsibility moved in or out | |
| Data structure/type change | |
| New optimization or constraint | |
| Evidence (two pinned links) | |
| Consequence for your mental model | |

The goal is not to catalog renames. The goal is to learn which abstractions survived two years of production evolution and which were replaced.
