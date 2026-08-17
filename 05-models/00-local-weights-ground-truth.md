# GLM-5.2 local weights: ground truth from the checkpoints on this box

**Research date:** 2026-08-17
**Scope:** everything in this document was read out of a file on this machine.
Nothing here comes from memory, and nothing comes from a web page. Where a
number is derived rather than read, it is labelled `[inferred]` and the
arithmetic is shown.

**Label convention used throughout**

| label | meaning |
|---|---|
| `[verified]` | read directly from a file on this box; the path is given |
| `[reported]` | stated in a model card / README shipped *inside* the weights directory (i.e. read locally, but authored by the model publisher) |
| `[inferred]` | computed from `[verified]` numbers; the arithmetic is shown |
| `[unverified]` | not checked here — flagged for someone to measure |

---

## Status

Three directories were assigned. Two contain a real checkpoint; one is empty.

| directory | state | shards | on-disk (safetensors only) |
|---|---|---|---|
| `/home/aman/code/weights/GLM-5.2-FP8` | complete | 141 | 755,632,050,320 B = **755.63 GB / 703.72 GiB** |
| `/home/aman/code/weights/GLM-5.2-NVFP4` | complete | 47 | 464,823,042,096 B = **464.82 GB / 432.87 GiB** |
| `/home/aman/code/weights/GLM-5.2-FP8-TileRT` | **EMPTY** — 0 files, `du -sb` = 6 bytes (the directory inode) | 0 | 0 |

`[verified]` — `ls -l`, `du -sb`, and the safetensors headers of every shard.

**The TileRT build does not exist on this box.** The directory was created
2026-08-16 15:35 and never populated. Every statement about "the TileRT build"
in any other document is therefore about something that is not here. The diff
table in §5 has a TileRT column that is honestly empty.

**Provenance of the two real checkpoints** `[verified]` from
`.cache/huggingface/trees/*.json` and the shipped `README.md` front-matter:

- `GLM-5.2-FP8` is a Hugging Face snapshot of **`zai-org/GLM-5.2-FP8`**
  (README links to `https://huggingface.co/zai-org/GLM-5.2-FP8`, license MIT,
  `library_name: transformers`). HF tree manifest:
  `GLM-5.2-FP8/.cache/huggingface/trees/ba978f7d347eaf65d22f1a86833408afdb953541.json`.
- `GLM-5.2-NVFP4` is a snapshot of **`nvidia/GLM-5.2-NVFP4`** `[reported]`
  (README: "Release Date: Hugging Face 06/25/2026 via
  `https://huggingface.co/nvidia/GLM-5.2-NVFP4`", `base_model: zai-org/GLM-5.2`,
  `library_name: Model Optimizer`). HF tree manifest:
  `GLM-5.2-NVFP4/.cache/huggingface/trees/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa.json`.

Both ship the **same tokenizer and the same chat template** `[verified]` by md5:

```
752f6cd2e6a4a2ea824d1b513530e0b0  GLM-5.2-FP8/tokenizer.json
752f6cd2e6a4a2ea824d1b513530e0b0  GLM-5.2-NVFP4/tokenizer.json
42994f78b64752fe472149dd7e20410d  GLM-5.2-FP8/chat_template.jinja
42994f78b64752fe472149dd7e20410d  GLM-5.2-NVFP4/chat_template.jinja
```

**The single most important fact in this document:** the two checkpoints are
the *same model* — identical architecture, identical parameter count to the
byte — but they are **quantized on completely different sets of modules**. The
FP8 build quantizes essentially every GEMM. The NVFP4 build quantizes **only
the routed experts**, and leaves all attention, all shared experts, the three
dense layers, and the whole MTP layer in BF16. See §5. This is not a detail;
it changes the roofline of every non-MoE kernel by 2x.

---

## 0. What GLM-5.2 is, in one table

Derived only from the checkpoints. Every row is reproducible from the files.

| quantity | value | how established |
|---|---|---|
| total parameters | **753,329,921,024** (753.33 B) | `[inferred]`, exact — see §4.1 |
| activated parameters / token | **40,297,758,720** (40.30 B) | `[inferred]` — see §4.2 |
| architecture class | `GlmMoeDsaForCausalLM` | `[verified]` `config.json:architectures` |
| HF `model_type` | `glm_moe_dsa` | `[verified]` |
| attention | MLA (DeepSeek-style latent) + **DeepSeek Sparse Attention** indexer, top-2048 | `[verified]` `config.json`, `layer_types` all `deepseek_sparse_attention` |
| context | 1,048,576 tokens | `[verified]` `max_position_embeddings` |
| MoE | 256 routed experts, top-8, 1 shared, 3 leading dense layers | `[verified]` |
| MTP | 1 next-token-prediction layer (index 78) | `[verified]` `num_nextn_predict_layers` + layer-78 tensors |

The NVFP4 model card independently states **"753B in total and 40B activated"**
`[reported]` (`GLM-5.2-NVFP4/README.md`, "Model Architecture"). The
checkpoint-derived numbers above agree with it, which is a useful independent
check on the arithmetic in §4.

---

## 1. `GLM-5.2-FP8` — full architecture as read from `config.json`

Path: `/home/aman/code/weights/GLM-5.2-FP8/config.json` (29,464 B).
All rows `[verified]`. Exact JSON key names are given so a kernel author or
config-parser can grep for them.

### 1.1 Core shape

| JSON key | value | note |
|---|---|---|
| `architectures` | `["GlmMoeDsaForCausalLM"]` | |
| `model_type` | `"glm_moe_dsa"` | |
| `dtype` | `"bfloat16"` | the *base* dtype; quantization is layered on top |
| `num_hidden_layers` | `78` | decoder layers, 0..77 |
| `hidden_size` | `6144` | |
| `intermediate_size` | `12288` | dense-MLP FFN width (only layers 0-2) |
| `hidden_act` | `"silu"` | SwiGLU (gate/up/down) |
| `rms_norm_eps` | `1e-05` | |
| `initializer_range` | `0.02` | training-only |
| `vocab_size` | `154880` | |
| `tie_word_embeddings` | `false` | `lm_head` is a separate 154880x6144 tensor |
| `max_position_embeddings` | `1048576` | 1 M |
| `attention_bias` | `false` | no bias tensors anywhere in attention |
| `attention_dropout` | `0.0` | |
| `pretraining_tp` | `1` | |
| `use_cache` | `true` | |
| `transformers_version` | `"5.12.0"` | |

### 1.2 Attention (MLA)

| JSON key | value | note |
|---|---|---|
| `num_attention_heads` | `64` | query heads |
| `num_key_value_heads` | `64` | nominal; **MLA stores one 576-wide latent**, not 64 KV heads |
| `q_lora_rank` | `2048` | Q down-projection rank |
| `kv_lora_rank` | `512` | KV latent rank |
| `qk_nope_head_dim` | `192` | non-positional part of QK |
| `qk_rope_head_dim` | `64` | RoPE part of QK |
| `qk_head_dim` | `256` | = 192 + 64 |
| `v_head_dim` | `256` | **note: != qk_nope_head_dim**, unlike DeepSeek-V3 |
| `head_dim` | `192` | **inconsistent with `qk_head_dim`/`v_head_dim`** — see §9.1 |
| `rope_parameters` | `{"rope_theta": 8000000, "rope_type": "default"}` | **no YARN, no scaling** — plain RoPE at theta 8e6 for 1 M context |
| `rope_interleave` | `true` | => SGLang sets `is_neox_style=False` for the main RoPE |

The MLA latent is `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` wide —
**identical to DeepSeek-V3/V3.2 and Kimi K2/K3**, so any KV-traffic model built
for those transfers unchanged. The head count does not: 64 heads here against
DeepSeek-V3.2's 128, with 1.5x the nope dim (192 vs 128) and 2x the V dim
(256 vs 128).

### 1.3 DSA indexer (the "lightning indexer")

| JSON key | value | note |
|---|---|---|
| `index_n_heads` | `32` | indexer heads (vs 64 attention heads) |
| `index_head_dim` | `128` | |
| `index_topk` | `2048` | keys kept per query |
| `index_topk_freq` | `4` | one indexer serves 4 consecutive layers |
| `index_skip_topk_offset` | `3` | phase offset of the freq-4 pattern |
| `index_topk_pattern` | `null` | pattern is generated from freq+offset |
| `index_share_for_mtp_iteration` | `true` | MTP iterations reuse the target's top-k |
| `indexer_rope_interleave` | `true` | => `is_neox_style=False` => SGLang enables its fused indexer path |
| `indexer_types` | 78-entry list of `"full"` / `"shared"` | 21 `"full"`, 57 `"shared"` |

**The indexer only exists on 21 of 78 layers.** `[verified]` two independent
ways:

1. `indexer_types` has `"full"` at exactly layers
   `0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74`.
2. `model.safetensors.index.json` contains `self_attn.indexer.*` tensors for
   exactly layers `0, 1, 2, 6, 10, ..., 74` **plus layer 78** (the MTP layer),
   22 sets in total. The other 57 layers carry no indexer weights at all.

This matches the engine's rule exactly. In
`/home/aman/code/NotSglang/python/sglang/srt/configs/model_config.py:170`,
`dsa_layer_skips_topk()` computes
`max(layer_id - offset + 1, 0) % freq != 0` = `max(layer_id - 2, 0) % 4 != 0`,
which is False (i.e. the layer runs its own indexer) for
`{0,1,2,6,10,...,74}` and True for all others. `[verified]`

This is the **IndexShare** mechanism the FP8 README describes `[reported]`:
"reuses the same indexer across every four sparse attention layers, reducing
per-token FLOPs by 2.9x at a 1M context length"
(`GLM-5.2-FP8/README.md`, "Improved Architecture").

### 1.4 MoE

| JSON key | value | note |
|---|---|---|
| `n_routed_experts` | `256` | |
| `num_experts_per_tok` | `8` | top-8 |
| `n_shared_experts` | `1` | always-on |
| `moe_intermediate_size` | `2048` | per-expert FFN width (also the shared expert's) |
| `first_k_dense_replace` | `3` | layers 0,1,2 are dense MLP |
| `moe_layer_freq` | `1` | every layer from 3 onward is sparse |
| `mlp_layer_types` | 78-entry list: 3x `"dense"`, 75x `"sparse"` | redundant with `first_k_dense_replace` |
| `scoring_func` | `"sigmoid"` | not softmax |
| `topk_method` | `"noaux_tc"` | DeepSeek-V3 aux-loss-free routing with bias correction |
| `norm_topk_prob` | `true` | |
| `routed_scaling_factor` | `2.5` | |
| `n_group` / `topk_group` | `1` / `1` | **no group-limited routing** — a token may hit any of the 256 experts |
| `moe_router_dtype` | `"float32"` | gate GEMM in fp32 (FP8 config only; absent in NVFP4) |
| `ep_size` | `1` | checkpoint-side hint, ignored by the engine |

`n_group = 1` is worth noting for an EP plan: DeepSeek-V3 restricts a token to
`topk_group` of `n_group` device groups, which bounds all-to-all fan-out.
GLM-5.2 does not. `[inferred]` a token's 8 experts can land on all 8 ranks.

### 1.5 MTP / speculative head

| JSON key | value | note |
|---|---|---|
| `num_nextn_predict_layers` | `1` | one MTP layer, checkpoint index **78** |

Layer 78 is a **complete decoder layer** — full MLA attention, its own indexer,
its own 256-expert MoE and shared expert — plus four MTP-specific tensors
(`enorm`, `hnorm`, `eh_proj`, `shared_head.norm`). `[verified]` from the
weight map: layer 78 has 1,569 tensors. It is *not* a lightweight EAGLE head;
it is 10.03 GB (FP8) / 19.91 GB (NVFP4) of extra weights. See §4.3.

### 1.6 Tokenizer / generation

| file / key | value |
|---|---|
| `tokenizer_config.json:tokenizer_class` | `"TokenizersBackend"` (backend `tokenizers`) |
| `tokenizer_config.json:model_max_length` | `1048576` |
| `config.json:eos_token_id` | `[154820, 154827, 154829]` — three stop ids |
| `config.json:pad_token_id` | `154820` |
| `generation_config.json` | `temperature: 1.0`, `top_p: 0.95`, same eos/pad |
| special tokens | `<|endoftext|>`, `[MASK]`, `[gMASK]`, `[sMASK]`, `<sop>`, `<eop>`, `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|observation\|>`, and image/video/audio/transcription begin-end pairs |
| `chat_template.jinja` | 5,076 B, identical across both builds |

The image/video/audio special tokens exist in the tokenizer but **there is no
vision or audio tower in either checkpoint** `[verified]` — the weight map has
no `vision`, `visual`, or `audio` tensors. The tokenizer is shared with the
multimodal members of the family.

### 1.7 Quantization block (FP8)

`config.json:quantization_config` `[verified]`:

| key | value |
|---|---|
| `quant_method` | `"fp8"` |
| `fmt` | `"e4m3"` |
| `activation_scheme` | `"dynamic"` |
| `weight_block_size` | `[128, 128]` — **block-wise**, not per-tensor, not per-channel |
| `modules_to_not_convert` | 541 entries |

The 541 exclusions normalize to (count x pattern):

```
79  model.layers.N.input_layernorm
79  model.layers.N.post_attention_layernorm
79  model.layers.N.self_attn.q_a_layernorm
79  model.layers.N.self_attn.kv_a_layernorm
76  model.layers.N.mlp.gate
76  model.layers.N.mlp.gate.e_score_correction_bias
22  model.layers.N.self_attn.indexers_proj          <-- see §9.2, name is wrong
22  model.layers.N.self_attn.indexer.k_norm
22  model.layers.N.self_attn.indexer.k_norm.bias
 1  model.layers.78.hnorm / .enorm / .eh_proj / .shared_head.norm
 1  model.norm
 1  lm_head
 1  model.embed_tokens
```

Everything else — all attention projections, all dense MLPs, all 19,456 routed
experts, all 76 shared experts, and the indexer's `wk` and `wq_b` — is
FP8 E4M3 with a `[ceil(out/128), ceil(in/128)]` fp32 `weight_scale_inv`.

---

## 2. `GLM-5.2-NVFP4` — full architecture as read from `config.json`

Path: `/home/aman/code/weights/GLM-5.2-NVFP4/config.json` (15,517 B).

**The architecture block is byte-for-byte equivalent to the FP8 build.** A
programmatic key/value diff of the two files (excluding `quantization_config`)
`[verified]` returns:

| difference | FP8 | NVFP4 |
|---|---|---|
| keys only in FP8 | `moe_router_dtype: "float32"` | — |
| keys only in NVFP4 | — | `bos_token_id: 0`, `mlp_bias: false`, `num_experts: 256`, `layer_types: [78x "deepseek_sparse_attention"]` |
| differing values | `transformers_version: "5.12.0"` | `transformers_version: "5.11.0"` |

Nothing else differs. Every number in §1.1-§1.6 applies verbatim to the NVFP4
build. Two notes:

- `num_experts: 256` duplicates `n_routed_experts: 256`. Harmless, but a
  parser that reads `num_experts` and adds `n_shared_experts` would get 257.
- **`moe_router_dtype` is missing from the NVFP4 config.** In the FP8 build
  the router GEMM is pinned to fp32. `[unverified]` whether SGLang's default
  for a missing `moe_router_dtype` is also fp32 on this path; if it is not,
  the two builds route in different precision, which would show up as a
  routing-divergence between the checkpoints and nowhere else. Worth 10
  minutes to check.

### 2.1 Quantization block (NVFP4)

Two files carry it and they agree `[verified]`:
`config.json:quantization_config` and `hf_quant_config.json`.

| key | value |
|---|---|
| `quant_method` | `"modelopt"` |
| `quant_algo` | `"NVFP4"` |
| `producer` | `modelopt` `0.46.0.dev65+g977d34dc3` |
| `config_groups.group_0.targets` | `["Linear"]` |
| `...weights` | `num_bits: 4`, `type: float`, `group_size: 16`, `dynamic: false` |
| `...input_activations` | `num_bits: 4`, `type: float`, `group_size: 16`, `dynamic: false` |
| `kv_cache_scheme` | `num_bits: 8`, `type: float`, `dynamic: false` (i.e. **FP8 KV cache**) |
| `hf_quant_config.json:kv_cache_quant_algo` | `"FP8"` |
| `ignore` / `exclude_modules` | 156 entries |

The 156 exclusions normalize to:

```
75  model.layers.{3..77}.mlp.shared_experts*
75  model.layers.{3..77}.self_attn*
 1  model.layers.0*        <-- whole layer
 1  model.layers.1.*       <-- whole layer
 1  model.layers.2.*       <-- whole layer
 1  model.layers.78*       <-- whole MTP layer
 1  lm_head
 1  model.embed_tokens
```

**Net effect: only `model.layers.{3..77}.mlp.experts.*` are quantized.**
That is 75 layers x 256 experts x 3 projections = 57,600 quantized Linears.
`[verified]` by tensor count: the index has 19,456 expert `*_proj.weight`
tensors (76 layers x 256) but only 19,200 `weight_scale` tensors (75 x 256) —
the missing 256 are layer 78's, which stayed BF16.

The NVFP4 README says the same thing in prose `[reported]`: "Only the weights
and activations of the linear operators within transformer blocks in MoE
experts are quantized. The shared expert is not quantized."

`.quant_summary.txt` (8.8 MB, 40,972 lines) is ModelOpt's per-quantizer dump
and confirms it tensor by tensor `[verified]`, e.g.:

```
model.layers.0.self_attn.q_a_proj.weight_quantizer      TensorQuantizer(disabled)
model.layers.0.self_attn.k_bmm_quantizer   TensorQuantizer((4, 3) bit fake per-tensor amax=dynamic calibrator=MaxCalibrator quant)
model.layers.0.self_attn.v_bmm_quantizer   TensorQuantizer((4, 3) bit fake per-tensor amax=dynamic calibrator=MaxCalibrator quant)
```

`(4, 3) bit` = E4M3 = the FP8 **KV cache** quantizer, and it is `amax=dynamic`,
i.e. **no static k_scale/v_scale tensors are shipped** (none appear in the
weight map `[verified]`). The KV cache scales are computed at runtime.

---

## 3. `GLM-5.2-FP8-TileRT` — nothing here

```
/home/aman/code/weights/GLM-5.2-FP8-TileRT:  0 files, created 2026-08-16 15:35
```

`[verified]`. No `config.json`, no shards, no README. There is nothing to
report and nothing to compare against. Any TileRT column below is empty by
necessity, not by omission.

`[unverified]` — whether a TileRT build of GLM-5.2 exists anywhere. This
document makes no claim about it.

---

## 4. Sizes: on disk, and per GPU under TP8

### 4.1 Parameter count, derived and cross-checked

Reading every safetensors header (`struct`-unpacked 8-byte length + JSON) and
summing element counts by dtype `[verified]`:

| build | dtype | elements | bytes |
|---|---|---:|---:|
| **FP8** | `F8_E4M3` | 751,226,191,872 | 751,226,191,872 |
| | `BF16` | 2,103,729,152 | 4,207,458,304 |
| | `F32` (scales) | 45,872,560 | 183,490,240 |
| | **total** | 753,375,793,584 | **755,617,140,416** |
| **NVFP4** | `U8` (packed FP4, 2/byte) | 362,387,865,600 B => **724,775,731,200 params** | 362,387,865,600 |
| | `F8_E4M3` (block scales) | 45,298,483,200 | 45,298,483,200 |
| | `BF16` | 28,554,189,824 | 57,108,379,648 |
| | `F32` | 134,656 | 538,624 |
| | **total** | | **464,795,267,072** |

**Both builds hold exactly 753,329,921,024 real parameters** `[inferred]`,
and it checks three ways:

```
FP8   real params = 751,226,191,872 (fp8) + 2,103,729,152 (bf16) = 753,329,921,024
NVFP4 real params = 724,775,731,200 (fp4) + 28,554,189,824 (bf16) = 753,329,921,024
```

and from the config alone, module by module:

```
routed experts   3 * 6144 * 2048 * 256 * 76 layers = 734,439,407,616
attention        (6144*2048 + 2048*16384 + 6144*576 + 512*28672 + 16384*6144) * 79
                 = 165,019,648 * 79                =  13,036,552,192
shared experts   3 * 6144 * 2048 * 76              =   2,868,903,936
embed + lm_head  2 * 154880 * 6144                 =   1,903,165,440
dense MLP        3 * 6144 * 12288 * 3 layers       =     679,477,248
indexers         (2048*4096 + 6144*128 + 6144*32) * 22
                 = 9,371,648 * 22                  =     206,176,256
MoE routers      256 * 6144 * 76                   =     119,537,664
MTP eh_proj      6144 * 12288                      =      75,497,472
norms + k_norm                                     =       1,203,200
                                                     ---------------
                                                     753,329,921,024   EXACT MATCH
```

The `F32` buckets are pure scale metadata: FP8's 45,872,560 fp32 elements are
the block `weight_scale_inv` tensors plus 76x256 `e_score_correction_bias`;
NVFP4's 134,656 = 19,456 (`e_score_correction_bias`) + 75x256x3x2
(`weight_scale_2` + `input_scale` scalars) = 115,200. `[verified]` — both
reconcile to the byte.

> **Watch out:** `GLM-5.2-NVFP4/model.safetensors.index.json` advertises
> `"total_parameters": 380989135104`. That is **wrong by 2x for the expert
> weights**: it counts each `uint8` container as one parameter when it holds
> two FP4 values. Do not quote 381 B. `[verified]`

### 4.2 Activated parameters per token

`[inferred]`, excluding the MTP layer, including `lm_head`:

```
attention        165,019,648 * 78 layers            = 12,871,532,544
MoE layers 3-77  (8 routed + 1 shared) * 37,748,736
                 + 256*6144 router, per layer       = 341,311,488 * 75
                                                    = 25,598,361,600
dense MLP        226,492,416 * 3                    =    679,477,248
indexers         9,371,648 * 21 (layer 78 excluded) =    196,804,608
lm_head          154880 * 6144                      =    951,582,720
                                                      --------------
                                                      40,297,758,720   = 40.30 B
```

Matches the NVFP4 model card's "40B activated" `[reported]`.

**Ratio that matters for the roofline:** 40.30 B activated of 753.33 B total =
**5.35%**. Of the 40.30 B activated, 25.60 B (63.5%) is MoE and 12.87 B
(31.9%) is attention projections.

### 4.3 On-disk bytes by module role

`[verified]` — every tensor classified by name and summed. GB = 10^9 bytes.

| role | FP8 | NVFP4 | NVFP4 / FP8 |
|---|---:|---:|---:|
| routed experts (layers 3-77) | 724.953 GB | 407.687 GB | 0.56x |
| `o_proj` (row-TP) | 7.854 | 15.703 | **2.00x** |
| shared experts | 2.832 | 5.662 | **2.00x** |
| `q_b_proj` (col-TP) | 2.618 | 5.234 | **2.00x** |
| `lm_head` | 1.903 | 1.903 | 1.00x |
| `embed_tokens` | 1.903 | 1.903 | 1.00x |
| `q_a`+`kv_a` fused (replicated) | 1.258 | 2.515 | **2.00x** |
| `kv_b_proj` (col-TP) | 1.145 | 2.290 | **2.00x** |
| dense MLP (layers 0-2) | 0.680 | 1.359 | **2.00x** |
| MoE routers | 0.236 | 0.236 | 1.00x |
| indexers | 0.201 | 0.394 | 1.96x |
| norms / misc | 0.002 | 0.002 | 1.00x |
| **target model total** | **745.585 GB** | **444.889 GB** | 0.60x |
| MTP layer 78 (experts) | 9.666 | 19.327 | **2.00x** |
| MTP layer 78 (everything else) | 0.367 | 0.579 | |
| **MTP total** | **10.033 GB** | **19.906 GB** | **1.98x** |
| **checkpoint total** | **755.617 GB** | **464.795 GB** | |

Read the `2.00x` column: **every module the NVFP4 build declines to quantize
is exactly twice the size in the NVFP4 checkpoint, because it is BF16 there
and FP8 in the other build.** That is 33.4 GB of attention + shared-expert +
dense weights running at BF16 in the "FP4" build, versus 16.7 GB at FP8.

### 4.4 Per-expert cost, three ways

`[inferred]` from the shapes in §6:

| format | bytes / expert | bits / param | contents |
|---|---:|---:|---|
| BF16 (reference) | 75,497,472 | 16.00 | — |
| FP8 block 128x128 | 37,757,952 | **8.002** | 3 weights + 3 fp32 `[16,48]`/`[48,16]` scales |
| NVFP4 group-16 | 21,233,688 | **4.500** | 3 packed-u8 weights + 3 E4M3 block-scale planes + 6 fp32 scalars |

NVFP4's 0.5 bits/param of scale overhead is the E4M3 plane at group size 16
(1 byte per 16 values = 0.5 bit/value). FP8's block-128x128 scale overhead is
0.002 bits/param — negligible.

### 4.5 Per-GPU weight footprint under TP8

To split this correctly you have to know which modules SGLang replicates.
`[verified]` from
`/home/aman/code/NotSglang/python/sglang/srt/models/deepseek_v2.py`:

| module | class | line | TP behaviour |
|---|---|---|---|
| `fused_qkv_a_proj_with_mqa` (= `q_a_proj` ⊕ `kv_a_proj_with_mqa`) | `ReplicatedLinear` | 1777 | **replicated on every rank** |
| `q_b_proj` | `ColumnParallelLinear` | 1785 | sharded on heads |
| `kv_b_proj` | `ColumnParallelLinear` | 1850 | sharded on heads |
| `o_proj` | `RowParallelLinear` | 1860 | sharded on input |
| `indexer.wq_b` / `.wk` / `.weights_proj` | `ReplicatedLinear` | `dsa_indexer.py:416,433,440` | **replicated on every rank** |
| MoE router `mlp.gate` | (replicated) | — | replicated |
| norms | — | — | replicated |
| experts, shared experts, dense MLP, embed, lm_head | — | — | sharded |

Summing accordingly `[inferred]` (GiB = 2^30):

| build | replicated bytes | sharded bytes | **per GPU @ TP8** |
|---|---:|---:|---:|
| FP8, target only | 1,697,145,408 (1.581 GiB) | 743,887,362,048 (692.80 GiB) | **88.18 GiB = 94.68 GB** |
| FP8, target + MTP | +28,910,208 | +10,003,722,752 | **89.37 GiB = 95.96 GB** |
| NVFP4, target only | 3,146,964,480 (2.931 GiB) | 441,742,460,928 (411.41 GiB) | **54.36 GiB = 58.37 GB** |
| NVFP4, target + MTP | +54,200,832 | +19,851,640,832 | **56.72 GiB = 60.90 GB** |

Against 183 GB (170.5 GiB) per B200:

| build | weights | **headroom for KV + activations + graphs** |
|---|---:|---:|
| FP8 + MTP | 89.37 GiB | **81.1 GiB** |
| NVFP4 + MTP | 56.72 GiB | **113.8 GiB** |

The replicated tail is small but not free: **1.58 GiB (FP8) / 2.93 GiB
(NVFP4) per rank is duplicated 8 times**, i.e. 12.6 / 23.4 GiB of aggregate
HBM spent on the same bytes. The NVFP4 build pays nearly double because its
`q_a`+`kv_a` fused projection and its indexers are BF16.

---

## 5. Diff table: what actually differs between the builds

Per-module quantization state. `[verified]` from the tensor inventory (a
module is "quantized" iff its weight tensor is not BF16 and a scale tensor
exists beside it).

| module | GLM-5.2-FP8 | GLM-5.2-NVFP4 | GLM-5.2-FP8-TileRT |
|---|---|---|---|
| `embed_tokens` | BF16 | BF16 | *(directory empty)* |
| `lm_head` | BF16 | BF16 | — |
| `layers.0-2` dense `gate/up/down_proj` | **FP8 E4M3**, block `[128,128]` | BF16 | — |
| `self_attn.q_a_proj` | **FP8**, scale `[16,48]` | BF16 | — |
| `self_attn.q_b_proj` | **FP8**, scale `[128,16]` | BF16 | — |
| `self_attn.kv_a_proj_with_mqa` | **FP8**, scale `[5,48]` | BF16 | — |
| `self_attn.kv_b_proj` | **FP8**, scale `[224,4]` | BF16 | — |
| `self_attn.o_proj` | **FP8**, scale `[48,128]` | BF16 | — |
| `self_attn.indexer.wq_b` | **FP8**, scale `[32,16]` | BF16 | — |
| `self_attn.indexer.wk` | **FP8**, scale `[1,48]` | BF16 | — |
| `self_attn.indexer.weights_proj` | BF16 | BF16 | — |
| `self_attn.indexer.k_norm` (.weight/.bias) | BF16 | BF16 | — |
| `mlp.gate.weight` (router) | BF16 | BF16 | — |
| `mlp.gate.e_score_correction_bias` | F32 | F32 | — |
| `mlp.shared_experts.*` | **FP8**, scale `[16,48]`/`[48,16]` | BF16 | — |
| `mlp.experts.{0..255}.*`, layers 3-77 | **FP8**, scale `[16,48]`/`[48,16]` | **NVFP4** u8 + E4M3 `weight_scale` + F32 `weight_scale_2` + F32 `input_scale` | — |
| `mlp.experts.*`, **layer 78 (MTP)** | **FP8** | **BF16 — not quantized** | — |
| all layernorms / `model.norm` | BF16 | BF16 | — |
| MTP `enorm`/`hnorm`/`eh_proj`/`shared_head.norm` | BF16 | BF16 | — |

### 5.1 Scale-factor tensors present

| build | scale tensors | dtype | shape rule | granularity |
|---|---|---|---|---|
| FP8 | `<w>.weight_scale_inv` | F32 | `[ceil(out/128), ceil(in/128)]` | 128x128 weight block; **activations dynamic per-token-group-128** (`activation_scheme: "dynamic"`) |
| NVFP4 | `<w>.weight_scale` | F8_E4M3 | `[out, in/16]` | one E4M3 scale per 16 contiguous input values |
| NVFP4 | `<w>.weight_scale_2` | F32 | `[]` scalar | global per-tensor scale for the E4M3 plane |
| NVFP4 | `<w>.input_scale` | F32 | `[]` scalar | static (calibrated) global activation scale; per-16 block scales computed at runtime |
| both | KV cache scales | *none shipped* | — | FP8 KV amax is dynamic (§2.1) |

### 5.2 Block sizes side by side

| | FP8 | NVFP4 |
|---|---|---|
| weight block | 128 x 128 (2-D) | 1 x 16 (1-D, along K) |
| weight scale dtype | FP32 | E4M3 |
| activation quant | dynamic, group 128 | group 16, static global scale |
| scale overhead | 0.002 bit/param | 0.500 bit/param |
| effective width | 8.002 bit/param | 4.500 bit/param |

### 5.3 Practical consequences

1. `[inferred]` **In the NVFP4 build, every dense (non-MoE) GEMM runs BF16
   unless the engine requantizes.** SGLang has an explicit escape hatch for
   exactly this: `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN`
   (`deepseek_v2.py:2234-2243`, `_get_q_b_proj_quant_config`) substitutes an
   `Fp8Config(weight_block_size=[128,128])` for `q_b_proj` when the checkpoint
   is NVFP4. It only covers `q_b_proj`. Whether it is on by default and what
   it does to the measured 37.1% dense-GEMM share is **the single highest-value
   thing to measure on this box.**
2. `[inferred]` The NVFP4 MTP layer is **19.91 GB vs FP8's 10.03 GB** — a
   2.36 GiB/GPU penalty under TP8 for running speculative decoding on the
   NVFP4 build, for zero quality reason (the target's experts were fine to
   quantize; the draft's were excluded).
3. `[reported]` NVFP4 quality delta vs the FP8 baseline, from
   `GLM-5.2-NVFP4/README.md`: GPQA-Diamond 89.39 vs 89.52; SciCode 49.04 vs
   49.85; IFBench 75.81 vs 74.95; AA-LCR 70.13 vs 69.38; tau^2-Bench Telecom
   98.25 vs 97.90. Two of five *improve*, which is within run-to-run noise at
   these sample sizes — treat as "no measurable regression", not as a win.

---

## 6. Annotated tensor inventory for one layer

What follows is every tensor a kernel author will see for **layer 10** — a
"full"-indexer sparse layer, i.e. the most complete case. `[verified]` shapes
from the safetensors headers. `H=6144`, `TP=8` unless noted.

### 6.1 FP8 build, `model.layers.10.*`

```
--- pre-attention -----------------------------------------------------------
input_layernorm.weight                     BF16 [6144]        RMSNorm on the residual

--- MLA: down-projections (SGLang FUSES these two at load) ------------------
self_attn.q_a_proj.weight                  F8   [2048, 6144]  x -> q_lora (2048)
self_attn.q_a_proj.weight_scale_inv        F32  [16, 48]
self_attn.kv_a_proj_with_mqa.weight        F8   [576, 6144]   x -> [kv_latent 512 | k_pe 64]
self_attn.kv_a_proj_with_mqa.weight_scale_inv F32 [5, 48]
   => runtime module `fused_qkv_a_proj_with_mqa`, ReplicatedLinear [2624, 6144]
      (2048 + 512 + 64). One GEMM, output split 3 ways. NOT TP-sharded.

--- MLA: norms on the two latents ------------------------------------------
self_attn.q_a_layernorm.weight             BF16 [2048]        RMSNorm(q_lora)
self_attn.kv_a_layernorm.weight            BF16 [512]         RMSNorm(kv_latent)

--- MLA: up-projections ----------------------------------------------------
self_attn.q_b_proj.weight                  F8   [16384, 2048] q_lora -> 64 heads x 256
self_attn.q_b_proj.weight_scale_inv        F32  [128, 16]     ColumnParallel: [2048,2048] per rank @TP8
self_attn.kv_b_proj.weight                 F8   [28672, 512]  kv_latent -> 64 x (192 nope + 256 v)
self_attn.kv_b_proj.weight_scale_inv       F32  [224, 4]      ColumnParallel: [3584,512] per rank
   NOTE: in the "absorbed" MLA decode path kv_b_proj is not run as a GEMM;
   it is split into w_kc / w_vc and folded into two BMMs around the attention.

--- DSA indexer (present only on layers 0,1,2,6,10,...,74,78) ---------------
self_attn.indexer.wq_b.weight              F8   [4096, 2048]  q_lora -> 32 idx heads x 128
self_attn.indexer.wq_b.weight_scale_inv    F32  [32, 16]      ReplicatedLinear
self_attn.indexer.wk.weight                F8   [128, 6144]   x -> single 128-d index key
self_attn.indexer.wk.weight_scale_inv      F32  [1, 48]
self_attn.indexer.weights_proj.weight      BF16 [32, 6144]    x -> per-head gate (32)
self_attn.indexer.k_norm.weight            BF16 [128]         LayerNorm (has a bias => not RMS)
self_attn.indexer.k_norm.bias              BF16 [128]
   => on CUDA these become ONE bf16 module `indexer.wk_weights_proj` of shape
      [160, 6144] (128 wk rows + 32 weights_proj rows). The FP8 wk is
      DEQUANTIZED to bf16 at load. See §7.3 and §9.3.

--- attention output -------------------------------------------------------
self_attn.o_proj.weight                    F8   [6144, 16384] 64 heads x 256 v -> 6144
self_attn.o_proj.weight_scale_inv          F32  [48, 128]     RowParallel: [6144,2048] per rank

--- pre-MLP ----------------------------------------------------------------
post_attention_layernorm.weight            BF16 [6144]

--- MoE router -------------------------------------------------------------
mlp.gate.weight                            BF16 [256, 6144]   sigmoid scores, fp32 GEMM
mlp.gate.e_score_correction_bias           F32  [256]         noaux_tc routing bias

--- shared expert (always on) ----------------------------------------------
mlp.shared_experts.gate_proj.weight        F8   [2048, 6144]  + weight_scale_inv [16,48]
mlp.shared_experts.up_proj.weight          F8   [2048, 6144]  + weight_scale_inv [16,48]
mlp.shared_experts.down_proj.weight        F8   [6144, 2048]  + weight_scale_inv [48,16]

--- 256 routed experts, top-8 selected -------------------------------------
mlp.experts.{0..255}.gate_proj.weight      F8   [2048, 6144]  + weight_scale_inv [16,48]
mlp.experts.{0..255}.up_proj.weight        F8   [2048, 6144]  + weight_scale_inv [16,48]
mlp.experts.{0..255}.down_proj.weight      F8   [6144, 2048]  + weight_scale_inv [48,16]
```

Total for one sparse FP8 layer: **1,569 tensors**, 9.83 GB.

### 6.2 NVFP4 build, what changes in the same layer

Every `self_attn.*`, `mlp.shared_experts.*`, `mlp.gate.*` tensor above becomes
plain BF16 with **no scale tensor**. Only the routed experts change form:

```
mlp.experts.E.gate_proj.weight          U8      [2048, 3072]   <- 2048x6144 FP4, 2 per byte
mlp.experts.E.gate_proj.weight_scale    F8_E4M3 [2048, 384]    <- 6144/16 block scales
mlp.experts.E.gate_proj.weight_scale_2  F32     []             <- global scale
mlp.experts.E.gate_proj.input_scale     F32     []             <- static activation scale
mlp.experts.E.up_proj.*                 (identical shapes)
mlp.experts.E.down_proj.weight          U8      [6144, 1024]   <- 6144x2048 FP4
mlp.experts.E.down_proj.weight_scale    F8_E4M3 [6144, 128]    <- 2048/16
mlp.experts.E.down_proj.weight_scale_2  F32     []
mlp.experts.E.down_proj.input_scale     F32     []
```

### 6.3 The MTP layer, `model.layers.78.*`

Everything in §6.1 (including its own indexer) **plus**:

```
enorm.weight              BF16 [6144]         RMSNorm on the embedding branch
hnorm.weight              BF16 [6144]         RMSNorm on the hidden-state branch
eh_proj.weight            BF16 [6144, 12288]  concat(enorm(e), hnorm(h)) -> 6144
shared_head.norm.weight   BF16 [6144]         final norm before the shared lm_head
```

`[verified]` — 1,569 layer tensors + these 4. The MTP layer **reuses the
target's `embed_tokens` and `lm_head`** (there is no second copy in the
checkpoint), which is what `get_embed_and_head` / `set_embed_and_head` in
`deepseek_v2.py:3105-3115` wire up.

### 6.4 Per-layer GEMM shape table for a kernel author

`[inferred]` — the actual `(M, N, K)` a single decode token sees at TP8.
`M = 1` at concurrency 1 (or `= num_draft_tokens` under EAGLE verify).

| GEMM | N (per rank) | K | replicated? |
|---|---:|---:|---|
| `fused_qkv_a_proj_with_mqa` | 2624 | 6144 | **yes, all 8 ranks** |
| `q_b_proj` | 2048 | 2048 | no |
| `indexer.wq_b` | 4096 | 2048 | **yes** |
| `indexer.wk_weights_proj` | 160 | 6144 | **yes** |
| `kv_b` absorb (BMM `w_kc`) | 8 heads x 192 x 512 | | no |
| `kv_b` absorb (BMM `w_vc`) | 8 heads x 512 x 256 | | no |
| `o_proj` | 6144 | 2048 | no |
| `mlp.gate` (router) | 256 | 6144 | **yes** |
| shared expert gate/up | 2 x 256 | 6144 | no |
| shared expert down | 6144 | 256 | no |
| routed expert gate/up (x8 tokens' worth) | 2 x 256 | 6144 | no |
| routed expert down | 6144 | 256 | no |

The two replicated GEMMs with large N — `fused_qkv_a_proj_with_mqa`
(2624x6144) and `indexer.wq_b` (4096x2048) — are run **identically on all 8
ranks**, 78 and 22 times per token respectively. At M=1 they are pure
bandwidth: 16.1 MB (FP8) or 32.3 MB (BF16, NVFP4 build) per token per rank
just for the a-projections. `[inferred]` This is a plausible contributor to
the 37.1% dense-GEMM share in the C1 profile and is worth isolating.

---

## 7. What the engine does per forward pass, in module order

The fork routes GLM-5.2 through the **DeepSeek-V2/V3 code path**:

```python
# /home/aman/code/NotSglang/python/sglang/srt/models/glm4_moe.py:1466
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")

class GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN):
    ...
EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM, GlmMoeDsaForCausalLMNextN]
```

`[verified]`. **There is no `glm5.py` and no GLM-specific decoder layer.** All
kernel names in a profile map back to `deepseek_v2.py`, `dsa_indexer.py` and
`dsa_backend.py`. The draft model is selected by rewriting the architecture
string (`model_config.py:580-581`: `GlmMoeDsaForCausalLM` ->
`GlmMoeDsaForCausalLMNextN` when `is_draft_model`).

### 7.1 Top level — `DeepseekV2Model.forward` (`deepseek_v2.py:2706`)

1. `embed_tokens(input_ids)` -> `hidden_states`; `residual = None`
2. allocate `BumpAllocator` zero-buffers (`total_num_layers * 2` fp32)
3. loop layers `start_layer..end_layer`, threading **`prev_topk_indices`**
   from layer to layer — this is the IndexShare carrier
4. `model.norm` -> `logits_processor(lm_head)`

Aux-hidden-state capture for EAGLE3 taps layers
`[2, num_hidden_layers // 2, num_hidden_layers - 3]` = **`[2, 39, 75]`**
`[verified]` (`deepseek_v2.py:3131`, the inherited implementation; the same
expression appears at `glm4_moe.py:1458` for `Glm4MoeForCausalLM`).

### 7.2 Per layer — `DeepseekV2DecoderLayer.forward` (`deepseek_v2.py:2399`)

| # | module / op | notes |
|---|---|---|
| 1 | `layer_communicator.prepare_attn_and_capture_last_layer_outputs` | fused RMSNorm+residual, plus the DP/TP gather; **this is where rank-arrival skew is paid** |
| 2 | `self_attn(...)` with `maybe_use_decode_attn_tp` ctx | returns `(hidden, topk_indices)` |
| 3 | `get_attn_tp_context().clear_attn_inputs()` | |
| 4 | `maybe_prefetch_next_full_attention_kv(next_full_attention_layer_id)` | DSA-specific KV prefetch for the next "full" indexer layer |
| 5 | `layer_communicator.prepare_mlp` | all-gather / reduce-scatter decision |
| 6 | `self.mlp(...)` — `DeepseekV2MLP` (layers 0-2) or `DeepseekV2MoE` (3+) | |
| 7 | `layer_communicator.postprocess_layer` **or** set `_sglang_needs_allreduce_fusion` | when `should_fuse_mlp_allreduce_with_next_layer`, the all-reduce is deferred into the next layer's norm (this is the `tllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` in the profile) |

### 7.3 Inside attention — `forward_absorb_prepare` (`attention_forward_methods/forward_mla.py:329`)

Decode path, `AttnForwardMethod.MLA`:

1. `fetch_qkv_latent()` -> one fused GEMM output, `.split([2048, 576])`
2. `q_a_layernorm(q)` on the main stream **||** `kv_a_layernorm(k_nope)` on
   `alt_stream` (only during CUDA-graph capture)
3. **then, also on two streams:**
   - alt stream: `q_b_proj_forward(q)` -> `[-1, num_local_heads=8, 256]`
   - main stream: `if should_run_indexer(prev_topk_indices): topk_indices = self.indexer(...)`
     else `topk_indices = maybe_capture_indexer_topk(layer_id, prev_topk_indices)`
4. RoPE on `q_pe` / `k_pe` (interleaved style — `rope_interleave: true`)
5. absorbed BMMs `w_kc` / `w_vc` around `attn_mqa` (a `RadixAttention` with
   `num_kv_heads=1`, `head_dim=576`, `v_head_dim=512`)
6. `o_proj` (RowParallel; `reduce_results=False`, the all-reduce is deferred)

`should_run_indexer` (`forward_mla.py:173`) is `not self.skip_topk or (is_nextn
and prev_topk_indices is None)`. With `skip_topk = dsa_layer_skips_topk(...)`,
**57 of 78 layers never touch the indexer at all** and simply forward the
carried `topk_indices`. `[verified]`

### 7.4 Inside the indexer — `Indexer.forward` (`dsa_indexer.py:357`)

Constructed with `scale_fmt="ue8m0"`, `block_size=128`,
`is_neox_style = not indexer_rope_interleave = False` `[verified]`
(`deepseek_v2.py:1816-1841`). Because `is_neox_style` is False and
`_is_cuda`, `use_dsa_indexer_fusion` is **True**, so:

- `wk` [128,6144] and `weights_proj` [32,6144] are concatenated into one
  **BF16** `wk_weights_proj` [160,6144] at load time. If the checkpoint stored
  `wk` in block-FP8 (the FP8 build), it is **dequantized to BF16**
  (`deepseek_weight_loader.py:80-123`, `block_quant_dequant`). `[verified]`
- `wq_b` [4096,2048] keeps the checkpoint's quantization.
- top-2048 selection runs through
  `DSAPagedMQALogitsBackend` (`dsa_backend.py`) — the profile's
  `parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen` and
  `cutedsl_paged_mqa_logits` kernels.

### 7.5 Inside the MoE — `DeepseekV2MoE.forward_normal` (`deepseek_v2.py:1050`)

Order: optional quantize-once of the MoE input -> shared expert (or deferred)
-> `self.gate(hidden)` (router GEMM) -> `self.topk(...)` (biased grouped top-k
with `e_score_correction_bias`) -> `self.experts(...)` -> scale by
`routed_scaling_factor` -> add shared output -> `tensor_model_parallel_all_reduce`.

`determine_num_fused_shared_experts` (`deepseek_v2.py:2964`) gates on
`architectures[0] == "GlmMoeDsaForCausalLM"`, `n_routed_experts in (256, 384)`
and `n_shared_experts == 1` — GLM-5.2 satisfies all three. The disable list is
`--disable-shared-experts-fusion`, SBO, TBO, a DeepEP a2a backend,
`moe_ep_size > 1` on CUDA, and W4AFP8/W4A16 `[verified]`. **Absent those,
`num_fused_shared_experts = 1` and the fused MoE kernel runs 257 experts, not
256.** A profile that reports "256 experts" is reporting the config, not the
kernel. Note this also means **plain TP8 fuses the shared expert, but the
moment you turn on EP the fusion silently disappears** — a shape change the
MoE kernel sees but no flag announces.

### 7.6 KV cache layout — `DSATokenToKVPool` (`mem_cache/memory_pool.py:4289`)

Two buffers per layer `[verified]`:

- **MLA latent**: `[size + page_size, 1, kv_cache_dim]`. With FP8 KV,
  `calculate_mla_kv_cache_dim` (`mem_cache/kv_cache_configurator.py:1963-1973`)
  returns
  `kv_lora_rank + kv_lora_rank//128*4 + qk_rope_head_dim*2`
  = `512 + 16 + 128` = **656 bytes/token/layer**
  (nope in FP8, per-128 fp32 block scales, **rope kept in BF16**).
- **Indexer K**: `[num_pages, page_size * (128 + 128//128*4)]` uint8
  = **132 bytes/token/layer**.

`page_size` is **asserted to be exactly 64** on CUDA
(`memory_pool.py:4347`). `[verified]`

`[inferred]` per-token KV cost, all 78 layers, FP8 KV, **replicated on every
TP rank** (MLA has one KV head, so TP does not shard it):

```
MLA latent : 656 B x 78 layers  = 51,168 B/token
indexer K  : 132 B x 78 layers  = 10,296 B/token
                                  -----------------
                                  61,464 B/token   = 61.46 GB per 1M tokens per GPU

with BF16 KV instead:  576 x 2 x 78 + 10,296 = 100,152 B/token = 100.15 GB / 1M
```

Cross-check against the headroom in §4.5: an NVFP4 + MTP deployment has
113.8 GiB free per GPU, so a single 1M-token sequence's KV (57.2 GiB) fits
with room for graphs and activations. An FP8 + MTP deployment has 81.1 GiB
free — also enough for one 1M sequence, but only just, and not at
`--mem-fraction-static 0.85`.

---

## 8. Engine-side config facts worth pinning

All `[verified]` from `/home/aman/code/NotSglang`.

| fact | value | source |
|---|---|---|
| default attention backend | `dsa` | `arg_groups/overrides.py:529` |
| DSA page size | must be 64 (assert) | `mem_cache/memory_pool.py:4347` |
| default speculative config for `GlmMoeDsaForCausalLM` | `(num_steps=3, eagle_topk=1, num_draft_tokens=4)` — i.e. **"3-1-4"** | `arg_groups/speculative_hook.py:800-822` |
| draft model path | defaults to the target path (MTP is in-checkpoint) | `speculative_hook.py:576-578` |
| FlashInfer all-reduce fusion | explicitly supported for this arch | `overrides.py:1573` |
| EPLB / expert-location support | registered | `eplb/lplb_solver.py:43` |
| decode attention-TP + CP | registered | `layers/cp/cp_decode_attn_tp.py:35` |
| registered 8-GPU test | `test/registered/8-gpu-models/test_glm52_fp8.py`, `zai-org/GLM-5.2-FP8`, TP8 / TP8+DP8 / TP8+DP8+MTP, gsm8k baseline **0.92** | that file |

The registered test's flags, in full `[verified]`:

```
--trust-remote-code --reasoning-parser=glm45 --tool-call-parser=glm47
--mem-fraction-static=0.85 --enable-metrics
  [+ --dp=8 --enable-dp-attention]
  [+ --speculative-algorithm=EAGLE --speculative-num-steps=3
     --speculative-eagle-topk=1 --speculative-num-draft-tokens=4]
```

And the NVFP4 vendor's published launch line `[reported]`
(`GLM-5.2-NVFP4/README.md`):

```sh
pip install -U "transformers>=5.3.0" && \
python3 -m sglang.launch_server \
    --model nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 8 \
    --quantization modelopt_fp4 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --trust-remote-code \
    --chunked-prefill-size 16384 \
    --mem-fraction-static 0.80
```

```sh
vllm serve nvidia/GLM-5.2-NVFP4 \
    --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --kv-cache-dtype fp8_e4m3 --host 0.0.0.0 --port 8000
```

Note the NVFP4 card claims `transformers>=5.3.0` while the shipped configs
declare `transformers_version` 5.11.0 / 5.12.0 `[verified]`, and the FP8 card
says "Transformers (v0.5.12+)" `[reported]` — three mutually inconsistent
version strings. Trust the config's own `transformers_version`.

---

## 9. Surprising or wrong-looking

### 9.1 `head_dim: 192` contradicts `qk_head_dim: 256` and `v_head_dim: 256`

`[verified]`. `head_dim` equals `qk_nope_head_dim`, not the real per-head QK
width (256) or V width (256). Any generic HF-config consumer that computes
`num_attention_heads * head_dim` gets `64 * 192 = 12288`, but the true
`o_proj` input is `64 * 256 = 16384` (confirmed by the tensor shape
`[6144, 16384]`). SGLang is immune because the MLA path reads
`qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` directly
(`deepseek_v2.py:2282-2286`). A KV-size estimator or a TRT-LLM importer that
trusts `head_dim` will be wrong by 1.33x. **Do not use `head_dim`.**

### 9.2 The FP8 exclusion list names a module that does not exist

`[verified]`. `quantization_config.modules_to_not_convert` contains 22 entries
of the form `model.layers.<N>.self_attn.indexers_proj`. **No tensor by that
name exists in the checkpoint.** The real tensor is
`self_attn.indexer.weights_proj`, which is *not* in the exclusion list — yet
it ships in BF16 anyway. So the shipped weights match the evident intent, but
the exclusion list does not describe them. A loader that applies
`modules_to_not_convert` literally would quantize `indexer.weights_proj`
(there is no scale for it, so it would fail loudly) or, worse, a
re-quantization tool that round-trips this config would silently produce a
different model. Also note `mlp.gate` is excluded by prefix; it does not
collide with `mlp.gate_proj` only because sparse layers have no `gate_proj`
and dense layers have no `mlp.gate` — luck, not design.

### 9.3 The FP8 indexer's `wk` is dequantized back to BF16 at load

`[verified]` (`deepseek_common/deepseek_weight_loader.py:105-122`). On CUDA the
fused `wk_weights_proj` param is created BF16
(`dsa_indexer.py:425-431`, `params_dtype=torch.bfloat16`), and the loader calls
`block_quant_dequant(weight, scale, [128,128], torch.bfloat16)` to fill it. So
the effort spent FP8-quantizing `indexer.wk` in the FP8 checkpoint is thrown
away at runtime, and the FP8 and NVFP4 builds execute the *identical* BF16
indexer `wk_weights_proj` GEMM. Small in bytes (1.97 MB/layer x 22), but it
means "the FP8 build has a quantized indexer" is false in practice.

### 9.4 The indexer KV cache is allocated for all 78 layers; only 22 use it

`[verified]`. `DSATokenToKVPool._create_index_buffers`
(`memory_pool.py:4358-4383`) allocates one `index_k_with_scale_buffer` per
layer, `for _ in range(self.layer_num)`, and `_compute_cell_size`
(`model_executor/pool_configurator.py:224-246`) charges
`indexer_size_per_token * effective_num_layers` for the budget. But only 21
non-MTP layers carry indexer weights, and SGLang's own comment says so:

> "shared layers' cache is never read, so filling it is dead work."
> — `forward_mla.py:189-190`

`[inferred]` cost of the dead allocation:

```
allocated : 132 B x 78 layers = 10,296 B/token
needed    : 132 B x 22 layers =  2,904 B/token
wasted    :                      7,392 B/token   = 7.39 GB per 1M tokens per GPU
```

That is 12% of the entire FP8-KV per-token cost, and 59 GB of aggregate HBM
across 8 ranks at a 1M-token cache. Nothing in the pool code is aware of
`skip_topk` — grepping `python/sglang/srt/mem_cache/` for `skip_topk` or
`indexer_layer` returns nothing. **This looks like a straightforward
several-GB-per-GPU win** and is the cheapest item on this list to verify: log
`token_to_kv_pool` sizes at startup and compare against 61,464 B/token.

### 9.5 The NVFP4 build leaves the MTP layer entirely in BF16

`[verified]` — `model.layers.78*` is in `exclude_modules`, and the tensor
count confirms it (19,456 expert weights, only 19,200 scale sets). The MTP
layer is therefore **19.91 GB in the NVFP4 build vs 10.03 GB in the FP8
build** — the *quantized* checkpoint's draft head is twice the size of the
unquantized-format one's. Under TP8 that is +2.36 GiB/GPU for speculative
decoding, and the draft's expert GEMMs run BF16 while the target's run FP4, so
draft and target take different kernel paths for the same operation.
`[unverified]` whether this was deliberate (draft-quality protection) or an
artifact of ModelOpt not knowing what layer 78 is. The evaluation table in the
NVFP4 README does not mention speculative decoding at all.

### 9.6 `layers.0`, `layers.1`, `layers.2` are excluded from NVFP4 quantization

`[verified]` — `model.layers.0*`, `model.layers.1.*`, `model.layers.2.*`. These
are exactly the three `first_k_dense_replace` dense layers, which have no
routed experts to quantize, so the exclusion is a no-op given
`targets: ["Linear"]` would have hit their dense MLPs. Effect: **the three
dense MLP layers are BF16 in the NVFP4 build** (1.359 GB vs FP8's 0.680 GB).
Consistent with the rest of the policy, but worth knowing when reading a
profile: the first three layers are a different GEMM shape *and* a different
dtype from everything after them.

### 9.7 The NVFP4 index metadata under-reports parameters by ~2x

`[verified]` — `"total_parameters": 380989135104` vs the true 753,329,921,024.
See §4.1. Any tool that reads that field will report GLM-5.2 as a 381 B model.

### 9.8 1M context on plain RoPE, theta 8e6, no scaling

`[verified]` — `rope_parameters: {"rope_theta": 8000000, "rope_type":
"default"}`. There is no YARN block, no `factor`, no
`original_max_position_embeddings`. SGLang handles the transformers-v5
`rope_parameters` key explicitly (`deepseek_v2.py:2262-2268`) and correctly
sets `rope_scaling = None` when `rope_type == "default"`, so `rope_theta`
does **not** silently fall back to the 10000 default in the function
signature. Worth knowing because that fallback is a classic silent-wrong-output
bug: `deepseek_v2.py:1727` declares `rope_theta: float = 10000`, and a
config-loading path that missed the `rope_parameters` branch would produce a
model that works at 4k and degrades at 100k.

### 9.9 `glm-kernels/` is a scaffold with zero kernels

`[verified]` — `/home/aman/code/NotSglang/glm-kernels/` contains
`CMakeLists.txt`, `README.md`, `include/glm/glm_abi.h`, and `src/common/`.
Its README says so explicitly: *"Status: scaffold only. **No kernels yet, and
that is deliberate.**"* By contrast `k3-kernels/` has real `.cu` files
(`src/attn/{kda_decode,mla_decode,mla_decode_mma,mla_prefill}.cu`,
`src/moe/moe_{decode,prefill}_mxfp4.cu`, `src/quant/act_quant_mxfp8.cu`) —
but those are Kimi-K3 MXFP4 kernels and, per the glm-kernels README, do not
port: *"different quant format, different attention structure, different head
counts, no linear attention."* **No hand-written kernel currently runs for
GLM-5.2.** Every kernel in a GLM-5.2 profile is SGLang's, cuBLAS's, TRT-LLM's,
or FlashInfer's.

---

## 10. Files read for this document

Weights:

- `/home/aman/code/weights/GLM-5.2-FP8/config.json`
- `/home/aman/code/weights/GLM-5.2-FP8/generation_config.json`
- `/home/aman/code/weights/GLM-5.2-FP8/tokenizer_config.json`
- `/home/aman/code/weights/GLM-5.2-FP8/README.md`
- `/home/aman/code/weights/GLM-5.2-FP8/model.safetensors.index.json`
- all 141 `/home/aman/code/weights/GLM-5.2-FP8/model-*.safetensors` headers
- `/home/aman/code/weights/GLM-5.2-FP8/.cache/huggingface/trees/ba978f7d347eaf65d22f1a86833408afdb953541.json`
- `/home/aman/code/weights/GLM-5.2-NVFP4/config.json`
- `/home/aman/code/weights/GLM-5.2-NVFP4/hf_quant_config.json`
- `/home/aman/code/weights/GLM-5.2-NVFP4/generation_config.json`
- `/home/aman/code/weights/GLM-5.2-NVFP4/tokenizer_config.json`
- `/home/aman/code/weights/GLM-5.2-NVFP4/README.md`
- `/home/aman/code/weights/GLM-5.2-NVFP4/.quant_summary.txt`
- `/home/aman/code/weights/GLM-5.2-NVFP4/model.safetensors.index.json`
- all 47 `/home/aman/code/weights/GLM-5.2-NVFP4/model-*.safetensors` headers
- `/home/aman/code/weights/GLM-5.2-NVFP4/.cache/huggingface/trees/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa.json`
- `/home/aman/code/weights/GLM-5.2-FP8-TileRT/` — confirmed empty

Engine (`/home/aman/code/NotSglang`):

- `python/sglang/srt/models/glm4_moe.py` (:1466-1539)
- `python/sglang/srt/models/deepseek_v2.py` (:252-3214; MLA init :1717-1900, decoder forward :2399-2496, model forward :2706-2883, MoE :549-1690, shared-expert fusion :2964-3035)
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` (:165-500)
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` (:80-300)
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` (:357-530)
- `python/sglang/srt/layers/attention/dsa_backend.py` (structure)
- `python/sglang/srt/configs/model_config.py` (:110-240, :575-690)
- `python/sglang/srt/mem_cache/memory_pool.py` (:3879-4400)
- `python/sglang/srt/mem_cache/kv_cache_configurator.py` (:1923-1974)
- `python/sglang/srt/mem_cache/dsa_cache_layer_split.py`
- `python/sglang/srt/model_executor/pool_configurator.py` (:180-300)
- `python/sglang/srt/arg_groups/overrides.py` (:500-600, :1495-1880)
- `python/sglang/srt/arg_groups/speculative_hook.py` (:555-840)
- `test/registered/8-gpu-models/test_glm52_fp8.py`
- `glm-kernels/README.md`, `k3-kernels/README.md`
- `personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` (cross-check only; its numbers are measurements, not checkpoint facts)

---

## Appendix A: reproducing the byte accounting

```python
import json, struct, glob, re, collections
def header(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))

for d in ['GLM-5.2-FP8', 'GLM-5.2-NVFP4']:
    tot = collections.Counter(); nbytes = collections.Counter()
    for p in sorted(glob.glob(d + '/*.safetensors')):
        for k, v in header(p).items():
            if k == '__metadata__': continue
            ne = 1
            for s in v['shape']: ne *= s
            tot[v['dtype']] += ne
            nbytes[v['dtype']] += v['data_offsets'][1] - v['data_offsets'][0]
    print(d, dict(tot), dict(nbytes))
```

Expected output (`[verified]`, this is what was run):

```
GLM-5.2-FP8   F32 45,872,560   BF16 2,103,729,152   F8_E4M3 751,226,191,872
GLM-5.2-NVFP4 F32 134,656  BF16 28,554,189,824  F8_E4M3 45,298,483,200  U8 362,387,865,600
```
