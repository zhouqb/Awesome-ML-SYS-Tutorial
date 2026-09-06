# Week 1 — From Chat Messages to Logits

Reading time: 35–45 minutes. This week establishes what enters the serving engine and what the model actually computes. Many “model bugs” are really boundary mismatches in templates, tokens, weights, or shapes.

## Learning objectives

After this lesson, you should be able to:

- trace chat messages into token IDs and token IDs into logits;
- explain special tokens and the assistant-generation marker;
- distinguish model construction from weight loading;
- identify the major tensor shapes through a decoder-only model;
- locate sampling as a separate operation after logits.

## 1. The representation ladder

An OpenAI-style request does not enter the Transformer as JSON. It passes through several representations:

```text
messages
→ rendered prompt string
→ token IDs
→ token embeddings
→ hidden states through decoder layers
→ vocabulary logits
→ sampled token ID
→ decoded text
```

Each arrow has an owner and a contract. Debugging becomes easier when you ask at which arrow the output first becomes wrong.

## 2. Chat templates are executable formatting rules

A base language model consumes a token sequence; it does not intrinsically understand roles such as `system`, `user`, and `assistant`. A chat template serializes structured messages into the text and special tokens expected during instruction tuning.

A conceptual rendering might be:

```text
<BOS><system>You are helpful.</system>
<user>What is 2+2?</user>
<assistant>
```

The exact tokens are model-specific. The final assistant marker is important: it tells the model that the next tokens belong to the assistant. Omitting it can make the model continue the user turn, emit role markers, or stop unexpectedly.

Special tokens commonly mark beginning/end of sequence, role boundaries, tool calls, or multimodal placeholders. `skip_special_tokens=True` affects decoded presentation; it does not mean those tokens were absent from model input.

The practical rule is: use the tokenizer and chat template shipped with the model revision unless you have deliberately validated a replacement.

## 3. Tensor shapes through a decoder

Let:

- `B` = batch size;
- `S` = number of input positions processed in this forward;
- `H` = hidden size;
- `V` = vocabulary size;
- `L` = number of decoder layers.

The core shape path is:

```text
input_ids:        [B, S]
embedding output: [B, S, H]
layer output:     [B, S, H]  repeated L times
final hidden:     [B, S, H]
logits:           [B, S, V]  or selected positions [N, V]
sampled IDs:      [N]
```

Inference engines avoid materializing logits for positions that do not need them. During ordinary decode, `S` is effectively one new position per active request, while attention reads cached history.

## 4. Inside one decoder layer

A simplified pre-norm decoder layer is:

```python
residual = hidden_states
x = input_layernorm(hidden_states)
x = self_attention(x, positions, kv_cache)
hidden_states = residual + x

residual = hidden_states
x = post_attention_layernorm(hidden_states)
x = mlp(x)
hidden_states = residual + x
```

This is explanatory pseudocode. The production implementation handles tensor parallelism, fused operators, quantization, rotary position encoding, attention backends, and model-specific variants.

At the top of the model, an `lm_head` projects hidden dimension `H` to vocabulary dimension `V`. Some models tie this matrix to the input embedding weights; others keep distinct parameters.

## 5. Construction, loading, and post-load processing

“Load the model” hides multiple phases:

1. Parse model and quantization configuration.
2. Construct module objects and parameter placeholders.
3. Iterate checkpoint tensors.
4. Map checkpoint names to framework/model parameter names.
5. Shard or transform tensors for the local rank.
6. Perform post-load packing, scale adjustment, or other optimization.
7. Move the model to an execution-ready state.

The distinction matters because a quantized kernel may require a storage layout that is not identical to the checkpoint layout. Model construction knows intended shapes; post-load processing has actual tensor values; execution needs the final packed representation.

In current pinned SGLang, start at [`DefaultModelLoader.load_model`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_loader/loader.py#L772). Then follow the model-specific [`Qwen3ForCausalLM.load_weights`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/qwen3.py#L596).

## 6. Forward and sampling are separate

The model’s forward method produces hidden states/logits; a sampler turns the relevant logits into token IDs. Greedy decoding chooses `argmax`. Temperature rescales logits before normalization. Top-k and top-p restrict candidate sets. Penalties and grammar masks can modify scores before selection.

Conceptually:

```python
logits = model(input_ids, positions, forward_batch)
scores = apply_bias_penalties_and_constraints(logits)
probs = softmax(scores / temperature)
next_token = sample(probs)
```

SGLang’s pinned [`Sampler.forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/sampler.py#L95) is the production boundary. Separating sampling lets the engine batch different sampling parameters and coordinate token IDs across tensor-parallel ranks.

## 7. Worked failure diagnoses

### Missing generation marker

Symptom: the output repeats the user, emits a role header, or behaves like continuation rather than response. First check the exact rendered string and token IDs—not the Transformer kernels.

### Wrong EOS token

Symptom: requests stop too early or continue until `max_new_tokens`. Compare tokenizer special-token IDs with server stopping configuration.

### Quantization/config mismatch

Symptom: load-time shape errors, unsupported-kernel errors, excessive memory, or silent quality collapse. Trace config selection, parameter creation, checkpoint mapping, and post-load processing.

### TP shard mismatch

Symptom: a local parameter has an incompatible dimension or ranks disagree. Write down global shape, partition dimension, TP size, and expected local shape before inspecting communication.

## 8. Source-reading path

Read these symbols in order:

1. [`TokenizerManager.generate_request`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/managers/tokenizer_manager.py#L624): find normalization and tokenization boundaries.
2. [`DefaultModelLoader`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/model_loader/loader.py#L355): identify shared loader responsibilities.
3. [`Qwen3ForCausalLM.forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/qwen3.py#L512): locate model, head, and output boundary.
4. [`Sampler.forward`](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/sampler.py#L95): locate score processing and token selection.

For each, answer: who calls it, what representation enters, what leaves, and what state changes?

## Checkpoint

Explain without notes:

1. Why a chat template is part of model correctness.
2. The shapes from `input_ids` to logits.
3. Why weight creation and post-load processing are separate.
4. Why the sampler is not part of the Transformer layer stack.
5. Where you would debug repeated role markers versus a TP shape error.

## Go deeper

- [Special tokens and chat templates](../../transformers/special_tokens/special_tokens_en.md).
- [SGLang Code Walk Through: model execution](../../sglang/code-walk-through/readme.md#model-load-weights-and-perform-forward).
- [Commit-pinned code map](../CODE_MAP.md#model-execution-and-sampling).
- Optional and marked Pending Review in the source repository: [How a Model Is Loaded](../../sglang/how-model-is-loaded/readme.md).

