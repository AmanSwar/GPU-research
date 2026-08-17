# Qwen 3.x on 8xB200

Research date: 2026-08-17. Target hardware: 8x B200 SXM (183 GB HBM3e each, 1.464 TB
aggregate, NV18 NVLink5 via NVSwitch, CUDA 13.2). Engine: SGLang fork at
`/home/aman/code/NotSglang`.

---

## Status

**"Qwen3.8" is real.** It is not a typo, not a rumor, and not a placeholder. As of
2026-08-17 the Qwen organization on HuggingFace publishes exactly four official
Qwen3.8 repositories [verified — fetched `huggingface.co/Qwen` and
`huggingface.co/models?search=Qwen3.8`]:

| Repo | Params | Modality | Created | Last modified |
|---|---|---|---|---|
| `Qwen/Qwen3.8-2.4T-A95B` | 2,446,182,725,504 | text-only | 2026-08-08 | 2026-08-12 |
| `Qwen/Qwen3.8-2.4T-A95B-FP8` | same, FP8 | text-only | 2026-08-08 | 2026-08-12 |
| `Qwen/Qwen3.8-27B` | 27,781,427,952 | image-text-to-text | 2026-08-05 | 2026-08-14 |
| `Qwen/Qwen3.8-27B-FP8` | same, FP8 | image-text-to-text | 2026-08-14 | 2026-08-14 |

[verified — HF API `/api/models/...`, `safetensors.parameters` and `createdAt` fields]

The naming convention changed from the `Qwen3-<size>-A<active>B` style. `2.4T-A95B`
means 2.4 trillion total parameters, 95 billion active. The 27B is a *dense*
multimodal model, not an MoE.

### Two things that are easy to get wrong

1. **The internal architecture name is `Qwen3_5`, not `Qwen3_8`.** The flagship's
   `config.json` declares `"architectures": ["Qwen3_5MoeForCausalLM"]` and
   `"model_type": "qwen3_5_moe_text"`; the 27B declares
   `"Qwen3_5ForConditionalGeneration"` / `"qwen3_5"` [verified — both config.json].
   Qwen3.8 is a *weights* refresh on the Qwen3.5 architecture. Every engine flag,
   every model-registry entry, and every PR title says "qwen3.5". If you grep your
   engine for `qwen3_8` you will find nothing.

2. **The public blog is stale.** `qwenlm.github.io/blog/` still tops out at
   Qwen3Guard (2025-09-23) [verified — fetched]. `qwen.ai/blog` returned an empty
   SPA shell to WebFetch. There is **no arXiv technical report** for Qwen3.5/3.6/3.8
   that I could find; `github.com/QwenLM/Qwen3.6` says documentation is "coming
   soon" [reported — fetched README]. `github.com/QwenLM/Qwen3.8` is 404
   [verified]. **All architectural numbers below come from config.json files and
   model cards I actually fetched, never from a blog.**

### The real lineup

[verified — HF search pages + `github.com/QwenLM/Qwen3.6` README, which carries the
release dates]

| Generation | Models | Released |
|---|---|---|
| Qwen3-Next | `Qwen3-Next-80B-A3B-Instruct/Thinking` (80B total / 3B active) | 2025 |
| Qwen3.5 | `397B-A17B` | 2026-02-16 |
| Qwen3.5 | `122B-A10B`, `35B-A3B`, `27B` | 2026-02-24 |
| Qwen3.5 | `9B`, `4B`, `2B`, `0.8B` (dense) | 2026-03-02 |
| Qwen3.6 | `35B-A3B` | 2026-04-16 |
| Qwen3.6 | `27B` (dense) | 2026-04-22 |
| Qwen3.8 | `27B` (dense multimodal) | 2026-08-05 |
| Qwen3.8 | `2.4T-A95B` (text-only MoE) | 2026-08-08 |

There is no Qwen3.7 in the public lineup [inferred — absent from HF search and from
the Qwen3.6 repo's model list]. Qwen3-Next-80B-A3B is the direct architectural
ancestor: it introduced the Gated DeltaNet + Gated Attention hybrid with 512 experts
and top-10 routing that Qwen3.5/3.6/3.8 all inherit [verified — Qwen3-Next model
card].

### Bottom line on existence

Everything the assignment asked me to look for — the `-A**B` naming, hybrid linear
attention, MTP heads, YaRN extension, FP8/NVFP4 releases — is real and documented.
What does **not** exist is a technical report. Nothing below is extrapolated from
"Qwen3" of 2025; it is all read off 2026 configs.

---

## What this is

Qwen3.5/3.6/3.8 are a single architecture family. Every member is:

- **Hybrid-attention.** `full_attention_interval: 4`. Three Gated DeltaNet (linear
  attention) layers, then one Gated Attention (softmax, RoPE, KV-cached) layer,
  repeating. 75% of layers have no KV cache at all.
- **Gated attention.** `attn_output_gate: true` — the Q projection emits both queries
  and a sigmoid gate, so `q_proj` is 2x width. `output_gate_type: "swish"` on the
  DeltaNet side.
- **Partial RoPE.** `partial_rotary_factor: 0.25` with `head_dim: 256` → only 64 of
  256 dims are rotated. `rope_theta: 10000000`.
- **MTP-native.** `mtp_num_hidden_layers: 1` on every single config I read, including
  the 0.8B.
- **262,144 native context**, extensible to ~1,010,000 with YaRN.
- **`vocab_size: 248320` on every model in the family**, 0.8B through 2.4T. This is
  the single most useful fact for speculative decoding (see §6).

The 2.4T flagship is text-only; everything else in 3.5/3.6/3.8 ships a SigLIP-style
vision tower (`vision_config.depth: 27`, `patch_size: 16`, `spatial_merge_size: 2`)
and mRoPE (`mrope_section: [11, 11, 10]`) [verified — configs].

---

## Bottom line for serving it on 8xB200

Ranked, concrete, most important first.

**1. `Qwen3.8-2.4T-A95B` does not fit on 8xB200 in any published quantization. Stop
planning around it.** The tightest checkpoint anywhere is NVFP4 at **1,447 GB**
against **1,464 GB** of HBM — 17 GB of headroom across the whole node, 2.1 GB per
GPU, before KV cache, DeltaNet state, activations, CUDA graphs, or NCCL buffers
[verified — arithmetic in §2 reproduces HF `usedStorage` to within 0.01%]. vLLM's own
recipe agrees: it lists NVFP4 W4A4 as **"8 GPUs (B300)"** — 288 GB parts, not 183 GB
parts — or 16x H200 [verified — recipes.vllm.ai/Qwen/Qwen3.8-2.4T-A95B]. On our box
this model needs 16 GPUs, i.e. two nodes at TP16.

**2. The model you should actually run is `Qwen3.5-397B-A17B-FP8`: 406 GB, leaving
1,058 GB free on the node.** It is the largest official checkpoint that fits with
room to serve. At FP8 it reads ~16.3 GB/token at batch 1 → a 3,925 tok/s HBM ceiling
(§3). It has a published latency recipe (TP8 + MTP-1) and a published throughput
recipe (DP8 + EP) [verified — recipes].

**3. This family is collective-*latency* bound at low concurrency, not
bandwidth-bound.** Qwen3.5-397B at TP8 issues **120 all-reduces per decoded token**
(15 full-attn + 45 DeltaNet + 60 MoE); the 2.4T issues **184**. At the ~3 µs per
all-reduce implied by our own GLM-5.2 measurements, that is 0.36 ms and 0.55 ms of
pure collective latency per token — against an FP8 weight-read floor of 0.26 ms for
the 397B. **Collectives cost more than the GEMMs.** Reducing collective *count* (not
bytes) is the single highest-leverage change. Our existing finding that 47% of
collective time is rank-arrival skew applies directly and is worse here because there
are more of them.

**4. NVFP4 is a capacity win, not a speed win, on this family — and for the 397B it
is actively slower than FP8.** NVIDIA's own NVFP4 recipe quantizes "only the weights
and activations of the linear operators within transformer blocks in MoE" [verified —
`nvidia/Qwen3.5-397B-A17B-NVFP4` card]; attention and Gated DeltaNet stay BF16. The
official FP8 checkpoint quantizes *everything* (400.7B of 403.4B params are F8_E4M3
[verified — HF API]). Result: FP8 reads ~16.3 GB/token, NVFP4 reads ~20.5 GB/token
(§3). **Use FP8 on B200 unless you need the capacity.**

**5. Verify the MTP draft head actually loads before you trust any speculative
number.** vLLM PR #52013 (opened 2026-08-12, Qwen-tagged) fixes a bug where a
dedicated `mtp.lm_head` was silently remapped to `model.lm_head.weight`, a module that
does not exist, so the draft head was **never loaded** and appeared only as an
"unexpected weight" warning. With the fix, MTP reaches **~96.8% draft acceptance**
[verified — PR #52013 description]. This almost certainly explains the otherwise
incoherent published numbers in §6.

**6. Budget for the DeltaNet recurrent state as a per-*sequence*, not per-*token*,
cost.** The 2.4T carries **552 MiB of fp32 SSM state per sequence** (69 layers x 128
value heads x 128 x 128 x 4 B), constant regardless of context length. At 8K context
the state (0.57 GiB) is *larger* than the FP8 KV cache (0.39 GiB). Any capacity model
that only counts KV will be wrong at high concurrency.

**7. Tree speculation is the wrong shape for this architecture; use chain drafts.**
You cannot cheaply fork and roll back a recurrent SSM state the way you can a KV
cache. Every published SGLang config for this family uses
`--speculative-eagle-topk 1` — a chain, not a tree [verified — Qwen3.5-397B and
Qwen3-Next model cards]. Our EAGLE 3-1-4 topology from GLM-5.2 does not port over as-is.

**8. `TP` must divide 64 attention heads → only 1/2/4/8/16/32 are legal** for the 2.4T
[verified — vLLM recipe]. The 397B has 32 heads but only **2 KV heads**, so TP>2
replicates KV rather than sharding it — a real reason to prefer DP-attention + EP for
throughput on that model.

**9. At batch 64 you still touch 72% of the 512 experts.** Expected distinct experts =
512·(1−(1−10/512)^B): B=64 → 367 experts; B=256 → 509. **Fine-grained MoE with 512
experts needs batch ≥256 to amortize.** Our GLM-5.2 economics at C64 will not transfer.

**10. TensorRT-LLM is not ready.** Qwen3.8-27B NVFP4 cannot load on the PyTorch
backend (issue #17723, open, five independent blockers). The perf PR #17700 is still
draft. Use SGLang or vLLM.

---

## 1. Architecture table

All values verbatim from `config.json` unless noted. `[cfg-2.4T]` =
`Qwen/Qwen3.8-2.4T-A95B/config.json`; `[cfg-397B]` = `Qwen/Qwen3.5-397B-A17B`
`text_config`; `[cfg-27B]` = `Qwen/Qwen3.8-27B` `text_config`; `[cfg-122B]` =
`Qwen/Qwen3.5-122B-A10B` `text_config`; `[cfg-0.8B]` = `Qwen/Qwen3.5-0.8B`
`text_config`. `[card]` = HF model card.

| Field | Qwen3.8-2.4T-A95B | Qwen3.5-397B-A17B | Qwen3.5-122B-A10B | Qwen3.8-27B | Qwen3.5-0.8B | Source |
|---|---|---|---|---|---|---|
| Architecture class | `Qwen3_5MoeForCausalLM` | `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` | cfg |
| `model_type` | `qwen3_5_moe_text` | `qwen3_5_moe` | `qwen3_5_moe` | `qwen3_5` | `qwen3_5` | cfg |
| Total params | 2,446,182,725,504 | 403,397,928,944 | 125,086,497,008 | 27,781,427,952 | ~0.8B | HF API |
| Active params (derived) | 93.10B (card: 95B) | 16.31B (card: 17B) | ~10B (card) | dense | dense | §3 derivation |
| `num_hidden_layers` | 92 | 60 | 48 | 64 | 24 | cfg |
| `hidden_size` | 8192 | 4096 | 3072 | 5120 | 1024 | cfg |
| `full_attention_interval` | 4 | 4 | 4 | 4 | 4 | cfg |
| Full-attn / linear layers | 23 / 69 | 15 / 45 | 12 / 36 | 16 / 48 | 6 / 18 | cfg `layer_types` |
| `num_attention_heads` | 64 | 32 | 32 | 24 | 8 | cfg |
| `num_key_value_heads` | 4 | 2 | 2 | 4 | 2 | cfg |
| `head_dim` | 256 | 256 | 256 | 256 | 256 | cfg |
| `attn_output_gate` | true | true | true | true | true | cfg |
| `linear_num_key_heads` | 16 | 16 | 16 | 16 | 16 | cfg |
| `linear_num_value_heads` | 128 | 64 | 64 | 48 | 16 | cfg |
| `linear_key_head_dim` | 128 | 128 | 128 | 128 | 128 | cfg |
| `linear_value_head_dim` | 128 | 128 | 128 | 128 | 128 | cfg |
| `linear_conv_kernel_dim` | 4 | 4 | 4 | 4 | 4 | cfg |
| `mamba_ssm_dtype` | float32 | float32 | float32 | float32 | float32 | cfg |
| `output_gate_type` | swish | swish | swish | swish | swish | cfg |
| `num_experts` | 512 | 512 | 256 | — (dense) | — (dense) | cfg |
| `num_experts_per_tok` | 10 | 10 | 8 | — | — | cfg |
| `moe_intermediate_size` | 2048 | 1024 | 1024 | — | — | cfg |
| `shared_expert_intermediate_size` | 2048 | 1024 | 1024 | — | — | cfg |
| Shared experts | 1 | 1 | 1 | — | — | card + cfg |
| Dense `intermediate_size` | — | — | — | 17408 | 3584 | cfg |
| `vocab_size` | 248320 | 248320 | 248320 | 248320 | 248320 | cfg |
| `max_position_embeddings` | 262144 | 262144 | 262144 | 262144 | 262144 | cfg |
| Extended context | 1,010,000 (YaRN) | 1,010,000 | 1,010,000 | 1,000,000 | — | card |
| `rope_theta` | 10,000,000 | 10,000,000 | 10,000,000 | 10,000,000 | 10,000,000 | cfg |
| `partial_rotary_factor` | 0.25 (→64 of 256 dims) | 0.25 | 0.25 | 0.25 | 0.25 | cfg |
| mRoPE | no (text-only) | `[11,11,10]` interleaved | `[11,11,10]` | `[11,11,10]` | `[11,11,10]` | cfg |
| `mtp_num_hidden_layers` | 1 | 1 | 1 | 1 | 1 | cfg |
| `mtp_use_dedicated_embeddings` | false | false | false | false | false | cfg |
| `tie_word_embeddings` | false | false | false | false | **true** | cfg |
| `rms_norm_eps` | 1e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 | cfg |
| Vision tower | none | depth 27, h1152 | depth 27, h1152 | depth 27, h1152 | depth 12, h768 | cfg |
| Tokenizer | shared across family, 248,320 entries, `bos=eos=248044` | same | same | same | same | cfg |

**Layer layout, stated on the model cards** [verified]:
- 2.4T: `23 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`
- 397B: `15 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`
- 27B: `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`

These are exactly consistent with the `layer_types` arrays I read. Good.

---

## 2. Memory arithmetic on 8xB200

Node budget: **8 × 183 GB = 1,464 GB = 1.464 TB**.

### 2.1 Weight bytes

**Qwen3.8-2.4T-A95B.** Total 2,446,182,725,504 params.

- **BF16**: 2,446,182,725,504 × 2 B = **4,892 GB**. HF reports ~4.89 TB [verified].
  3.34x over node capacity.
- **FP8 official** (`Qwen/Qwen3.8-2.4T-A95B-FP8`): 2,396.59B params in F8_E4M3 +
  49.59B in BF16 [verified — HF API dtype breakdown].
  `2,396.59e9 × 1 + 49.59e9 × 2 = 2,396.6 + 99.2 = 2,496 GB`. 1.70x over.
- **NVFP4** (`Inferact/Qwen3.8-2.4T-A95B-NVFP4`) [verified — HF API]:
  - packed FP4 stored as U8: `1,198,295,875,584 B = 1,198.30 GB`
  - E4M3 block scales at `group_size: 16`: `2,396.59e9 / 16 × 1 B = 149.79 GB`
  - unquantized BF16: `49,590,974,336 × 2 = 99.18 GB`
  - **total = 1,198.30 + 149.79 + 99.18 = 1,447.27 GB**
  - HF `usedStorage` = 1,447,406,629,396 B. **My arithmetic matches to 0.01%.**
    vLLM's recipe independently calls it "1.32 TiB" = 1,451.5 GB [verified].
  - Headroom on 8xB200: `1,464 − 1,447 = 17 GB` total, **2.1 GB/GPU. Not servable.**
- `RadixArk/Qwen3.8-2.4T-A95B-NVFP4` is *worse*: it leaves 75.36B params in BF16,
  `usedStorage = 1,484 GB` — **larger than the node** [verified — HF API].

**Everything else, on the same node budget:**

| Checkpoint | Weight bytes | Fits? | Free for cache |
|---|---|---|---|
| Qwen3.8-2.4T-A95B BF16 | 4,892 GB | no (3.3x) | — |
| Qwen3.8-2.4T-A95B FP8 | 2,496 GB | no (1.7x) | — |
| Qwen3.8-2.4T-A95B NVFP4 | 1,447 GB | technically | 17 GB — **no** |
| Qwen3.5-397B-A17B BF16 | 807 GB [HF usedStorage] | yes | 657 GB |
| **Qwen3.5-397B-A17B-FP8** | **406 GB** [HF usedStorage] | **yes** | **1,058 GB** |
| nvidia/Qwen3.5-397B-A17B-NVFP4 | ~751 GB reported [see note] | yes | ~713 GB |
| Qwen3.5-122B-A10B BF16 | 250 GB [HF usedStorage] | yes | 1,214 GB |
| Qwen3.6-35B-A3B BF16 | 71.9 GB [HF usedStorage] | yes | 1,392 GB |
| Qwen3.8-27B BF16 | 55.6 GB (27.78e9 × 2) | yes | 1,408 GB |
| Qwen3.8-27B-FP8 | 30.9 GB [HF usedStorage] | yes | 1,433 GB |

> **Note on the NVIDIA NVFP4 397B**: HF reports `usedStorage = 751.45 GB`, which is
> ~3.7x what NVFP4-on-MoE-only should cost (~215–250 GB derived). That is
> inconsistent with 11 safetensors shards. Most likely `usedStorage` is counting LFS
> history or a superseded revision. **[unverified — do not plan against 751 GB;
> measure the actual shard sum before committing.]** Listed as an open question in §9.

### 2.2 KV cache bytes per token per sequence

Only full-attention layers have a KV cache. This is the family's structural advantage.

`bytes/token = n_full_layers × 2 (K,V) × num_key_value_heads × head_dim × dtype_bytes`

| Model | full layers | KV heads | BF16 B/tok | FP8 B/tok | FP8 @ 262,144 ctx |
|---|---|---|---|---|---|
| Qwen3.8-2.4T | 23 | 4 | 94,208 (92 KiB) | 47,104 (46 KiB) | **12.35 GB/seq** |
| Qwen3.5-397B | 15 | 2 | 61,440 (60 KiB) | 30,720 (30 KiB) | **8.05 GB/seq** |
| Qwen3.5-122B | 12 | 2 | 49,152 (48 KiB) | 24,576 (24 KiB) | 6.44 GB/seq |
| Qwen3.8-27B | 16 | 4 | 65,536 (64 KiB) | 32,768 (32 KiB) | 8.59 GB/seq |

Worked example, 2.4T FP8: `23 × 2 × 4 × 256 × 1 B = 47,104 B/token`;
`× 262,144 = 12,348,030,976 B = 12.35 GB`.

For scale: a 92-layer model with *full* attention on every layer and 8 KV heads would
be `92 × 2 × 8 × 256 = 376,832 B/token` at FP8 — **8x** what this model costs.

### 2.3 Gated DeltaNet recurrent state — the term people forget

Per linear-attention layer, the DeltaNet carries a matrix-valued state
`S ∈ R^{H_v × d_k × d_v}` plus a short conv window. `mamba_ssm_dtype: float32`.

`state_bytes/layer = linear_num_value_heads × linear_key_head_dim × linear_value_head_dim × 4`

**Qwen3.8-2.4T**: `128 × 128 × 128 × 4 = 8,388,608 B = 8 MiB/layer`; `× 69 layers =
578,813,952 B = 552 MiB/seq`.
Conv state: channels `= 2×(16×128) + (128×128) = 4,096 + 16,384 = 20,480`;
`20,480 × 4 × 4 B = 327,680 B/layer × 69 = 21.6 MiB`.
**Total ≈ 574 MiB per sequence, independent of context length.**

**Qwen3.5-397B**: `64 × 128 × 128 × 4 = 4 MiB/layer × 45 = 180 MiB`; conv
`(4,096 + 8,192) × 4 × 4 = 196,608 B × 45 = 8.4 MiB`. **Total ≈ 188 MiB/seq.**

The crossover where state exceeds KV, at FP8:
- 2.4T: `574 MiB / 47,104 B/tok ≈ 12,780 tokens`. **Below ~12.8K context the
  recurrent state costs more than the KV cache.**
- 397B: `188 MiB / 30,720 B ≈ 6,420 tokens`.

At concurrency 64 the 2.4T's state alone is `64 × 574 MiB = 35.9 GiB`. This is why
vLLM's recipe warns to lower `--max-cudagraph-capture-size` on "Mamba cache" assertion
errors [verified — recipe] — the state pool is sized per captured graph batch.

### 2.4 What actually fits: Qwen3.5-397B-A17B-FP8 on 8xB200

Weights 406 GB. Reserve ~15% of the remainder for activations, CUDA graphs, NCCL
buffers and fragmentation → usable cache ≈ `(1,464 − 406) × 0.85 ≈ 899 GB`.

Per-sequence cost at FP8 KV = `8.05 GB (262K ctx) + 0.188 GB (state) = 8.24 GB`.

| Context | KV/seq (FP8) | + state | Concurrent sequences |
|---|---|---|---|
| 262,144 | 8.05 GB | 8.24 GB | **109** |
| 131,072 | 4.03 GB | 4.21 GB | 213 |
| 32,768 | 1.01 GB | 1.19 GB | 753 |
| 8,192 | 0.25 GB | 0.44 GB | 2,029 |

[inferred — arithmetic from verified config values]

**109 concurrent sequences at the full 262K context on one node** is an unusually
comfortable position, and it is a direct consequence of 45 of 60 layers having no KV
cache and the other 15 having only 2 KV heads.

### 2.5 TP8 vs EP vs hybrid, per GPU

For `Qwen3.5-397B-A17B-FP8` (406 GB):

- **TP8**: 50.75 GB of weights per GPU. MoE `moe_intermediate_size` 1024 shards to 128
  per rank — small, but workable. **KV does not shard: `num_key_value_heads = 2` <
  TP 8**, so K/V heads are replicated 4x. KV cost becomes `4 × 8.05 = 32.2 GB/seq` at
  262K unless the engine uses DP-attention. This is a serious TP8 penalty.
- **DP8 + EP8** (vLLM's throughput recipe): each rank holds all attention weights
  (~7.9B params ≈ 7.9 GB at FP8) plus 1/8 of the experts (`386.5B/8 = 48.3B ≈ 48.3
  GB`) = **~56 GB/GPU**. KV is *not* replicated — each rank owns whole sequences. This
  is why the published throughput recipe is `-dp 8 --enable-expert-parallel`.
- **Hybrid TP2 + DP4 + EP8**: KV shards exactly across the 2 KV heads with no
  replication, MoE goes wide. **This is the configuration I would test first for
  cost-per-user** (§4).

For the 2.4T at TP16 across two nodes: `2,496/16 = 156 GB/GPU` at FP8, `1,447/16 = 90
GB/GPU` at NVFP4. `moe_intermediate_size` 2048 / 16 = 128 per rank. Note SGLang PR
#34795 added a tuned MoE config for `E=512, N=256` — i.e. `2048/8`, confirming **TP8
sharding of the expert intermediate** is the shape being tuned [verified].

---

## 3. Roofline

**Bandwidth**: B200 SXM HBM3e = **8 TB/s per GPU**, **64 TB/s aggregate** across 8.
**Interconnect**: NVLink5, 1.8 TB/s bidirectional per GPU.

### 3.1 Deriving active parameters from config (not from the model name)

Everything here is `[inferred]` but from `[verified]` config values. Weight matrices
only; norms and biases are <1%.

**Qwen3.8-2.4T-A95B**, `H = 8192`, `moe_inter = 2048`, top-10 + 1 shared, 512 experts.

Per MoE block (all 92 layers):
```
routed   = 10 experts × 3 mats × 8192 × 2048 =  503,316,480
shared   =             3 mats × 8192 × 2048 =   50,331,648
router   =                       8192 × 512 =    4,194,304
                                       total =  557,842,432
× 92 layers                                  = 51,321,503,744
```

Per Gated Attention layer (23 layers). `attn_output_gate` makes `q_proj` 2x wide:
```
q_proj = 8192 × (64 × 256 × 2) = 8192 × 32768 = 268,435,456
k_proj = 8192 × (4 × 256)                     =   8,388,608
v_proj = 8192 × (4 × 256)                     =   8,388,608
o_proj = (64 × 256) × 8192                    = 134,217,728
                                        total = 419,430,400
× 23 layers                                   = 9,646,899,200
```

Per Gated DeltaNet layer (69 layers):
```
q_proj    = 8192 × (16 × 128)  =  16,777,216
k_proj    = 8192 × (16 × 128)  =  16,777,216
v_proj    = 8192 × (128 × 128) = 134,217,728
gate_proj = 8192 × (128 × 128) = 134,217,728
o_proj    = (128 × 128) × 8192 = 134,217,728
                          total = 436,207,616
× 69 layers                     = 30,098,325,504
```

`lm_head = 248,320 × 8192 = 2,034,237,440`

```
TOTAL ACTIVE = 51,321,503,744 + 9,646,899,200 + 30,098,325,504 + 2,034,237,440
             = 93,100,965,888  ≈ 93.10B
```
Model card says 95B. The 1.9B gap is norms, the MTP layer, and decay/beta projections
I did not enumerate. **The derivation validates the config.** [inferred, cross-checked]

**Qwen3.5-397B-A17B**, same method, `H = 4096`, `moe_inter = 1024`:
```
MoE:      140,509,184 × 60 =  8,430,551,040
Full attn:104,857,600 × 15 =  1,572,864,000
DeltaNet: 117,440,512 × 45 =  5,284,823,040
lm_head                    =  1,017,118,720
TOTAL ACTIVE               = 16,305,356,800 ≈ 16.31B   (card: 17B) ✓
```

### 3.2 Which parameters are in which precision

From the HF dtype breakdowns [verified] the 2.4T splits **2,396.59B quantized /
49.59B BF16**. Reconciling against my component totals:

```
routed experts  512 × 92 × 3 × 8192 × 2048 = 2,370.82B   → quantized
shared experts             92 × 50,331,648 =     4.63B   → quantized
routers                     92 ×  4,194,304 =     0.39B   → quantized
                                    subtotal = 2,375.84B
full attention                              =     9.65B   → BF16
Gated DeltaNet                              =    30.10B   → BF16
embed + lm_head (untied)                    =     4.07B   → BF16
norms + MTP layer                           =    ~5.8B    → BF16
                                    subtotal =   ~49.6B   ✓ matches 49.59B
```

So: **experts are FP4/FP8; both attention mechanisms and the embeddings stay BF16.**
This is confirmed independently by the NVFP4 `quantization_config` ignore list, which
exempts "linear attention and shared expert components, all self-attention modules,
embed_tokens, norm, lm_head, MTP layers" [verified — `RadixArk/...-NVFP4/config.json`],
and by NVIDIA's statement that "only the weights and activations of the linear
operators within transformer blocks in MoE are quantized" [verified].

### 3.3 Bytes read per decoded token, and the HBM ceiling

NVFP4 effective cost = `0.5 B/param + 1/16 B/param scale = 0.5625 B/param`.

**Qwen3.8-2.4T-A95B, batch 1:**

| Precision | Quantized active | BF16 active | Bytes/token | Time @64 TB/s | tok/s ceiling |
|---|---|---|---|---|---|
| BF16 | — | 93.10B × 2 | 186.2 GB | 2.909 ms | 344 |
| FP8 | 51.32B × 1 = 51.32 GB | 41.78B × 2 = 83.56 GB | **134.9 GB** | 2.108 ms | **474** |
| NVFP4 | 51.32B × 0.5625 = 28.87 GB | 41.78B × 2 = 83.56 GB | **112.4 GB** | 1.756 ms | **569** |

Quantized-active = MoE routed (46.31B) + shared (4.63B) + routers (0.39B) = 51.32B.
BF16-active = 9.65 + 30.10 + 2.03 = 41.78B.

> **The headline of this table**: going FP8 → NVFP4 only buys 17% on bandwidth,
> because **74% of the bytes read at batch 1 are the BF16 attention and DeltaNet
> weights**, which neither format touches. NVFP4 on this model is a *capacity* play.

**Reality check against published measurement**: vLLM's verification run for NVFP4
TP8 (on B300, 8K in / 1K out) reports **C1 = 101 tok/s, TPOT 9.9 ms** [verified —
recipe]. Against a 569 tok/s / 1.76 ms roofline that is **18% memory-bandwidth
utilization**. The missing 82% is §3.4.

**Qwen3.5-397B-A17B, batch 1** — the model that fits:

| Precision | Bytes/token | Time @64 TB/s | tok/s ceiling |
|---|---|---|---|
| BF16 | 16.31B × 2 = **32.6 GB** | 0.509 ms | 1,964 |
| **FP8 (everything quantized)** | ~16.3B × 1 = **16.3 GB** | **0.255 ms** | **3,925** |
| NVFP4 (MoE only) | 8.43B×0.5625 + 7.87B×2 = 4.74 + 15.74 = **20.5 GB** | 0.320 ms | 3,124 |

**NVFP4 is 26% slower than FP8 here** because the official FP8 checkpoint quantizes
400.7B of 403.4B params (everything), while NVIDIA's NVFP4 quantizes only the MoE and
leaves 7.87B of active attention/DeltaNet weights in BF16. [inferred from verified
dtype breakdowns.] Actionable: **on B200, FP8 beats NVFP4 for the 397B on both latency
and simplicity.**

### 3.4 Collective bytes and count per token — the actual bottleneck

Under plain TP, every layer needs one all-reduce of the hidden state after its output
projection, and MoE blocks need one after the expert down-projection.

`collectives/token = n_full_attn + n_deltanet + n_moe_blocks`

| Model | full | deltanet | MoE | **all-reduces/token** | payload (BF16) |
|---|---|---|---|---|---|
| Qwen3.8-2.4T | 23 | 69 | 92 | **184** | 8192 × 2 = 16 KB |
| Qwen3.5-397B | 15 | 45 | 60 | **120** | 4096 × 2 = 8 KB |
| Qwen3.5-122B | 12 | 36 | 48 | **96** | 3072 × 2 = 6 KB |

**Bandwidth cost is trivial.** Ring all-reduce moves `2(N−1)/N × S` = `1.75 × S`:
- 2.4T: `184 × 1.75 × 16 KB = 5.15 MB/token`. At 1.8 TB/s → **2.9 µs**. Nothing.

**Latency cost is not.** These are 8–16 KB messages: pure fixed-overhead territory.
Calibrating against our own box — GLM-5.2 at C1 shows collectives at 19.6% of a
2.74 ms TPOT = 0.537 ms; at ~184 collectives that is **~2.9 µs per all-reduce**, of
which 47% is rank-arrival skew. Using 3 µs:

| Model | all-reduces | collective latency/token | compute floor (FP8) | ratio |
|---|---|---|---|---|
| Qwen3.8-2.4T | 184 | **0.552 ms** | 2.108 ms | 0.26x |
| Qwen3.5-397B | 120 | **0.360 ms** | 0.255 ms | **1.41x** |
| Qwen3.5-122B | 96 | 0.288 ms | ~0.10 ms | **2.9x** |

**For the 397B and 122B, collective latency exceeds the entire weight-read time.**
Serialized floor for the 397B at TP8/FP8 = `0.255 + 0.360 = 0.615 ms` → **1,626
tok/s**. With good overlap (collective hidden behind the next layer's GEMM) you
approach 0.36 ms → 2,780 tok/s. With none, and at a realistic 35% MBU on the GEMMs,
expect 700–1,100 tok/s [inferred].

The 2.4T is the opposite: it is genuinely compute/bandwidth-heavy per token, so its
184 collectives are only 26% of the floor. Its problem is that it needs 16 GPUs.

### 3.5 Latency floor summary, 8xB200

| Config | weight read | collectives | serialized floor | tok/s | TPOT |
|---|---|---|---|---|---|
| 397B FP8 TP8 | 0.255 ms | 0.360 ms | 0.615 ms | 1,626 | 0.62 ms |
| 397B FP8 TP8 + MTP-1 @ 96.8% accept | — | — | — | **~2,760** | **~0.36 ms** |
| 397B FP8 TP2·DP4 + EP8 | 1.02 ms | 0.090 ms | 1.11 ms | 901 | 1.11 ms |
| 2.4T NVFP4 TP16 (2 nodes) | 0.878 ms | ~0.65 ms | 1.53 ms | 654 | 1.53 ms |
| 2.4T NVFP4 TP8 (B300, published) | — | — | measured | **101** | **9.9 ms** |

MTP-1 gain = `1 + 0.968 = 1.968` accepted tokens/step, minus ~15% draft overhead →
1.70x. [inferred from verified acceptance rate.]

The gap between the 397B's 0.62 ms theoretical TPOT and our GLM-5.2's measured 2.74 ms
is the entire prize here. **Our GLM-5.2 hotspot profile — dense GEMM 37.1%,
collectives 19.6%, MoE GEMMs 19.4% — will invert on Qwen3.5: there is very little
dense GEMM (no dense FFN at all in the MoE variants), the MoE is far sparser
(10/512 vs GLM's routing), and 75% of layers run a linear-attention kernel we have not
written.** Budget engineering time for a Gated DeltaNet kernel, not for GEMM tuning.

### 3.6 Throughput roofline: the 512-expert amortization problem

At batch `B`, expected distinct experts touched per layer:
`E(B) = 512 × (1 − (1 − 10/512)^B)`

| B | experts touched | of 512 | expert bytes/step (397B FP8) | bytes/token | agg tok/s @64 TB/s |
|---|---|---|---|---|---|
| 1 | 10 | 2.0% | 0.126 GB | 16.30 GB | 3,925 |
| 16 | 145 | 28.4% | 1.83 GB | 8.31 GB | 7,701 |
| 64 | 367 | 71.7% | 4.62 GB | 4.45 GB | 14,382 |
| 256 | 509 | 99.4% | 6.40 GB | 1.54 GB | 41,558 |
| 1024 | 512 | 100% | 6.44 GB | 0.44 GB | 145,000 |

Worked: B=64 → `(1−10/512)^64 = e^(64 × ln 0.98047) = e^(−1.2623) = 0.2830` →
`512 × 0.717 = 367 experts`. Expert bytes = `60 layers × 367 × 3 × 4096 × 1024 × 1 B
= 277.0 GB`... per *step*, i.e. `277.0/64 = 4.33 GB/token`, plus 0.12 GB/token of
attention → 4.45 GB/token.

**Our GLM-5.2 baseline is 40.8k tok/s at C64. Qwen3.5-397B would deliver ~14.4k tok/s
at C64 and only reaches GLM-parity at C≈256.** With 512 fine-grained experts and
top-10, the sparsity that makes this model cheap at batch 1 makes it expensive at
moderate batch. **If cost-per-user is the objective, run this model at C≥256** — which
§2.4 says the memory comfortably allows (753 sequences at 32K context).

---

## 4. Parallelism recommendation

### For minimum single-stream latency

**`Qwen3.5-397B-A17B-FP8`, TP8, MTP chain depth 1–2, FP8 KV.**

Reasoning:
1. TP8 minimizes per-GPU weight bytes (50.75 GB), which is the term the 0.255 ms
   compute floor is made of.
2. The KV replication penalty of TP8 (`num_key_value_heads=2 < 8`, so 4x replication)
   is irrelevant at concurrency 1 — you have 1,058 GB free and one sequence.
3. It is the published latency recipe [verified — vLLM recipe: `--tensor-parallel-size
   8 --speculative-config '{"method":"mtp","num_speculative_tokens":1}'`].
4. **Do not go past TP8 on one node.** The collective count is fixed at 120 by layer
   count; more ranks only raises per-collective skew, which our profiling already
   flags as 47% of collective time.

The lever that matters is **not** parallelism, it is collective count. Concretely:
fuse the DeltaNet `o_proj` all-reduce with the following MoE block's input (the
DeltaNet output and the MoE input are separated only by an RMSNorm), which would cut
45 of 120 collectives on the 397B. FlashInfer already ships an all-reduce+RMSNorm
fusion — SGLang exposes `--enforce-disable-flashinfer-allreduce-fusion` as a B300
escape hatch [verified — SGLang docs], implying the fusion is on by default elsewhere.
**Confirm it is active on B200 before optimizing anything else.**

### For minimum cost per user

**`Qwen3.5-397B-A17B-FP8`, DP8 + EP8, `--language-model-only`, prefix caching, target
concurrency ≥256.**

Reasoning:
1. §3.6: you need B≥256 to amortize 512 experts. DP8 gives 8 independent replicas of
   attention each running its own batch, so the *global* batch reaching each expert
   shard is 8x the per-rank batch — exactly the amortization you need.
2. EP8 puts 64 experts on each GPU (48.3 GB at FP8) with no expert-weight replication.
3. DP-attention means **no KV replication** — the TP8 4x penalty disappears, which is
   what makes 109 concurrent 262K sequences possible.
4. `--language-model-only` drops the vision tower [verified — vLLM documents this for
   "hybrid-only models such as ... Qwen-3.5"]. For a text workload this is free memory.
5. It is the published throughput recipe verbatim.

**The configuration I would actually test first and is not published anywhere:
TP2 · DP4 + EP8.** TP2 shards the 2 KV heads *exactly* (no replication, unlike TP8),
halves per-rank attention weight bytes versus pure DP, and cuts collective payload —
while DP4 still gives 4x batch aggregation into the experts. This is the natural fit
for a model with `num_key_value_heads = 2`, and no published recipe explores it.

### Do not attempt

- **2.4T on 8xB200 at any precision.** §2.1. 16 GPUs or nothing.
- **PP within the node.** vLLM verified TP4×PP3 for the 2.4T as a "fallback for
  non-power-of-two GPU counts" [verified], not as a latency config. Pipeline bubbles
  at concurrency 1 are pure loss and NVLink5 makes TP cheap.

---

## 5. Quantization

### What exists

| Format | Repo | Scope | Size | Status |
|---|---|---|---|---|
| BF16 | `Qwen/Qwen3.8-2.4T-A95B` | — | 4,892 GB | official |
| FP8 | `Qwen/Qwen3.8-2.4T-A95B-FP8` | experts + most linears | 2,496 GB | official |
| FP8 | `Qwen/Qwen3.5-397B-A17B-FP8` | 400.7B of 403.4B params | 406 GB | official |
| FP8 | `Qwen/Qwen3.8-27B-FP8`, `Qwen/Qwen3.6-35B-A3B-FP8`, `Qwen/Qwen3.5-122B-A10B-FP8` | — | — | official |
| NVFP4 | `nvidia/Qwen3.5-397B-A17B-NVFP4` | **MoE linears only** | see §2.1 note | NVIDIA ModelOpt |
| NVFP4 | `nvidia/Qwen3.6-35B-A3B-NVFP4` | MoE only | — | NVIDIA ModelOpt |
| NVFP4 | `Inferact/Qwen3.8-2.4T-A95B-NVFP4` | experts | 1,447 GB | community, **used by vLLM's own recipe** |
| NVFP4 | `Inferact/Qwen3.8-27B-NVFP4`, `unsloth/Qwen3.8-27B-NVFP4` | — | — | community |
| MXFP4 | `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` | — | ~1.2T params | AMD Quark |
| GPTQ-Int4 | `Qwen/Qwen3.5-122B-A10B` GPTQ variants | — | fits 1x80GB | community |
| AWQ-Int4 | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | — | — | community |
| INT8 W8A16 | `lued/Qwen3.8-27B-INT8-W8A16-MTP` | — | — | community |

[all verified — HF search pages and API]

**MXFP4 does not load on NVIDIA devices** — "missing linear method support"
[verified — vLLM Qwen3.8-27B recipe]. Ignore the AMD Quark checkpoints on this box.

### Official FP8 recipe details

`Qwen/Qwen3.8-2.4T-A95B-FP8` `quantization_config` [verified — fetched]:
- method: fp8, **dynamic activation quantization**
- **weight block size 128×128** (block-wise, not per-tensor: `per_tensor: false`)
- ignore list: embeddings, lm_head, MTP layers, and named attention/MLP components
  across layers 0–91

The 128×128 block shape matters for us: SGLang PR #34795 added a tuned Triton MoE
config keyed `E=512,N=256,dtype=fp8_w8a8,block_shape=[128,128]` — `N=256` is
`moe_intermediate_size 2048 / TP8`. Our `glm-kernels/` FP8 block-scaled GEMMs likely
assume a different block shape; **check before reusing them.**

### Published quality deltas

The only head-to-head table published by anyone
[verified — `nvidia/Qwen3.5-397B-A17B-NVFP4` model card]:

| Precision | MMLU-Pro | GPQA-Diamond | LiveCodeBench V6 | SciCode | AIME 2025 | AA-LCR | IFBench |
|---|---|---|---|---|---|---|---|
| FP8 | 0.883 | 0.871 | 0.837 | 0.467 | 0.918 | 0.696 | 0.761 |
| NVFP4 | 0.880 | 0.871 | 0.843 | 0.479 | 0.922 | 0.701 | 0.756 |

NVFP4 is within noise of FP8 (−0.3pp on MMLU-Pro, +1.2pp on SciCode, +0.5pp on
IFBench the other way). Calibration used CNN/DailyMail + Nemotron Post-Training
Dataset v2 [verified].

**No BF16 baseline is published**, so the BF16→FP8 delta is unmeasured by the vendor.
[unverified.] Given that the official FP8 checkpoint quantizes 99.3% of parameters,
this is the delta that actually matters and nobody has published it.

### What to keep in higher precision

Both NVIDIA and the community converge on the same answer, and the configs agree:
- **Keep BF16**: embeddings, `lm_head`, all RMSNorms, **all self-attention modules**,
  **all Gated DeltaNet modules**, and the **MTP layer**.
- **Quantize**: routed experts, shared expert, routers.

The DeltaNet exclusion is not arbitrary — `mamba_ssm_dtype: float32` says the state
recurrence itself is fp32. A linear-attention recurrence accumulates over the whole
sequence; low-precision error compounds in a way it does not in softmax attention.
**Do not quantize the DeltaNet state or its projections.**

### KV cache format

`--kv-cache-dtype fp8` appears in every published low-latency recipe for the 2.4T and
the 27B [verified]. Halves the KV term in §2.2. Given only 15–23 of 60–92 layers have
KV at all, and the quality-sensitive part of this architecture is the fp32 recurrence
rather than the KV, FP8 KV looks safe. **The DeltaNet state must stay fp32 regardless
of KV dtype** — they are separate pools.

---

## 6. Speculative decoding

### Does it ship an MTP head? Yes — on every model in the family.

`mtp_num_hidden_layers: 1` and `mtp_use_dedicated_embeddings: false` appear in the
config of the 2.4T, 397B, 122B, 27B **and the 0.8B** [verified — all five configs].
One MTP transformer layer, sharing the base model's input embeddings.

### Published acceptance and speedups

| Source | Config | Number |
|---|---|---|
| vLLM PR #52013 | MTP on an NVFP4 ckpt with dedicated draft head | **~96.8% draft acceptance** [verified] |
| vLLM recipe, 2.4T | MTP-3 | **"roughly 2.3x on per-user output rate"** [verified] |
| vLLM recipe, 2.4T | MTP-1 | **"only +3.4% at concurrency 1, with losses at higher loads"** [verified] |
| vLLM recipe, 2.4T | NVFP4 TP8 + MTP-3 | 304 tok/s per user |
| vLLM recipe, 2.4T | FP8 TP16 + MTP-3 | 307 tok/s per user |
| vLLM recipe, 2.4T | NVFP4 TP8, no MTP, C1 | 101 tok/s, TPOT 9.9 ms |

### These numbers are mutually inconsistent, and I think I know why

At 96.8% per-token acceptance, expected accepted tokens per step:
- depth 1: `1 + 0.968 = 1.97` → ~1.97x before overhead
- depth 3: `1 + 0.968 + 0.968² + 0.968³ = 1 + 0.968 + 0.937 + 0.907 = 3.81` → ~3.81x
  before overhead, and 2.3x realized is a plausible 60% efficiency

Depth 3 lines up. **Depth 1 giving +3.4% does not** — it should give ~1.7x.

The most likely explanation is sitting in the PR that fixed it. vLLM PR #52013
(2026-08-12) describes exactly this failure: when a checkpoint ships a dedicated
`mtp.lm_head`, the generic `mtp.` → `model.` remap rewrote it to
`model.lm_head.weight`, which `Qwen3_5MultiTokenPredictor` does not have, so **the
draft head was silently never loaded** and surfaced only as an "unexpected weight"
[verified — PR description]. A draft head running on an unloaded/base head would
produce near-zero acceptance, and MTP-1 would degenerate to base speed minus draft
overhead — i.e. roughly +0%. The recipe's MTP-1 measurement predates the fix; the
96.8% figure is from the fix's own end-to-end validation, where the model loaded "with
zero missing or unexpected weights."

**[inferred, but strongly.] Action: on our box, log the MTP weight-load result and
measure acceptance directly before believing any depth-1 number.**

### Recommended configuration

**Latency, 397B on 8xB200:**
```
--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```
Start at 2, sweep 1→4. The published 397B latency recipe uses 1; the 2.4T recipe found
3 optimal. Given ~96.8% acceptance, depth 3 is where the geometric series still pays
(0.907 marginal acceptance at step 3, 0.878 at step 4).

**SGLang equivalent** [verified — Qwen3.5-397B model card]:
```
--speculative-algo NEXTN --speculative-num-steps 3 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

### The architectural constraint on speculation: chains, not trees

**`--speculative-eagle-topk 1` in every published config is not an accident.** Tree
speculation requires cheaply forking and rolling back model state per tree branch.
For a KV cache that is a pointer copy. For a **Gated DeltaNet recurrent state it is a
552 MiB memcpy per branch** on the 2.4T (188 MiB on the 397B), and rollback requires
either re-running the recurrence or checkpointing state per node.

**Our GLM-5.2 EAGLE 3-1-4 topology (a tree) does not port to this family.** Expect to
run chain drafts. If we want trees, the engineering task is a DeltaNet state
checkpoint/restore path with branch-shared prefixes — that is a real project, and it
is the single biggest speculative-decoding difference between GLM-5.2 and Qwen3.5.

### Small dense Qwen3.5 models as draft models

This is a real, documented use. vLLM's own recipe for `Qwen3.5-0.8B` says it is sized
"for edge devices **or as a draft model for speculative decoding with larger Qwen3.5
checkpoints**" [verified].

Why it works: **the entire family shares `vocab_size: 248320` and the same
tokenizer** — 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B, and the 2.4T
[verified — every config I read]. Vocabulary identity is the hard prerequisite for
cross-model speculation, and Qwen satisfies it across three generations.

Candidates, cheapest first:

| Draft | Params | Layers | Hidden | Notes |
|---|---|---|---|---|
| `Qwen3.5-0.8B` | ~0.8B | 24 | 1024 | tied embeddings; explicitly recommended as draft |
| `Qwen3.5-2B` | ~2B | — | — | |
| `Qwen3.5-4B` | ~4B | — | — | |

Two caveats that matter and that no source mentions:

1. **The drafts are themselves hybrid linear-attention models.** `Qwen3.5-0.8B` has 18
   DeltaNet + 6 full-attention layers and `mamba_ssm_dtype: float32`. Running it as a
   draft means maintaining a *second* recurrent state pool (18 layers × 16 value
   heads × 128 × 128 × 4 B = 18 MiB/seq — small, but it must exist and must roll back
   in lockstep with the target on rejection). A pure-dense-attention draft would be
   operationally simpler; none exists in this family.
2. **The 0.8B carries a vision tower** (`vision_config.depth: 12`) and `mrope_section
   [11,11,10]`, while the 2.4T target is text-only with plain RoPE. For text drafting
   this is dead weight to be stripped, and the positional encodings differ between
   draft and target — a correctness question, not just an efficiency one.

**Given the built-in MTP head reaches ~96.8% acceptance at a fraction of 0.8B's cost,
use MTP first.** The separate-draft path is worth exploring only for the 2.4T (where
the MTP head is 1 layer of a 92-layer model and may under-serve depth >3) or for
cross-generation drafting.

---

## 7. Engine support status

### vLLM — best support, actively churning

- Model registry entry is **`Qwen3NextForCausalLM`** for Qwen3-Next and the `qwen3_5`
  / `qwen3_5_moe` types for 3.5/3.6/3.8 [verified — vLLM supported-models docs list
  `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`; the 3.5 family is
  documented via the hybrid-model note].
- **`--language-model-only`** is documented for "hybrid-only models such as Llama-4,
  Step3, Mistral-3 and Qwen-3.5" to zero out multimodal modalities and reclaim GPU
  memory [verified — vLLM docs, quoted].
- Version floor: **vLLM ≥ 0.17.0** for the 3.5 family; **≥ 0.24.0** for NVFP4 on some
  paths [verified — recipes].
- MoE backends: `--moe-backend flashinfer_trtllm` (FP8) and `flashinfer_cutedsl`
  (NVFP4); all-to-all via `--all2all-backend flashinfer_nvlink_one_sided` [verified].
- Linear-attention backend: **`--linear-backend flashinfer_cutedsl` is Docker-only
  until a FlashInfer version bump** — omit it on pip installs [verified — recipe].

**Open/recent PRs and issues:**

| PR/issue | Title | State |
|---|---|---|
| [#52013](https://github.com/vllm-project/vllm/pull/52013) | `[Bugfix][Qwen3.5-MTP] Load a dedicated mtp.lm_head draft head` | opened 2026-08-12 |
| [#52007](https://github.com/vllm-project/vllm/pull/52007) | `[CI Bug] Fix ci qwen3.5` | merged 2026-08-12 |
| [#52004](https://github.com/vllm-project/vllm/pull/52004) | `Add per-layer sliding window support for Qwen3.5 hybrid attention` | closed |
| [#51990–51992](https://github.com/vllm-project/vllm/pull/51992) | `Add TriangleMix attention acceleration for Qwen3.5 hybrid models` | **closed — spam, see below** |

> **Ignore TriangleMix.** PRs #51990/#51991/#51992 claim 2.15x TTFT and 1.94x
> throughput on Qwen3.5-2B by sliding-windowing "the deepest three full_attention
> layers." They were closed by the author within hours and a maintainer commented "ok
> it's a scam lol, this guy is spamming" [verified — fetched PR #51992]. The numbers
> are not credible and the technique is not in vLLM. Flagging explicitly because the
> claimed speedups are exactly what someone searching for hybrid-attention
> optimizations would latch onto.

### SGLang — supported, MTP via NEXTN, active kernel work

- Launch via `--model-path Qwen/Qwen3.5-...` with `--reasoning-parser qwen3` and
  `--tool-call-parser qwen3_coder` [verified — model cards].
- MTP via `--speculative-algo NEXTN` [verified].
- B300-specific flags exist in the docs — `--attention-backend flashinfer` and
  `--enforce-disable-flashinfer-allreduce-fusion` [verified — SGLang docs], which
  tells us **the FlashInfer all-reduce+RMSNorm fusion is default-on** and is the
  mechanism §4 wants us to confirm on B200.

| PR | Title | State |
|---|---|---|
| [#34795](https://github.com/sgl-project/sglang/pull/34795) | `[MoE] Add H20 fp8_w8a8 tuned configs for Qwen3.8` | merged 2026-08-17 |
| [#34934](https://github.com/sgl-project/sglang/pull/34934) | `[Perf] Fuse prefill norm/act quantization for NVFP4 W4A4` | open 2026-08-15 |
| [#34970](https://github.com/sgl-project/sglang/pull/34970) | `[AMD] Add Qwen3.5-397B-A17B MXFP4 recipes` | closed 2026-08-15 |
| [#34560](https://github.com/sgl-project/sglang/pull/34560) | `[Fix] Fix Qwen3.5 MTP startup with HiCache` | merged 2026-08-14 |
| [#34571](https://github.com/sgl-project/sglang/pull/34571) | `Fix: resolve Qwen3.5 NEXTN HiCache startup failure` | closed 2026-08-13 |
| [#34622](https://github.com/sgl-project/sglang/pull/34622) | `Prevent Qwen3.5 MTP draft from inheriting GPTQ quantization` | open 2026-08-12 |

**Known bug pattern: MTP + HiCache startup failures** (#34560, #34571) and **MTP draft
inheriting the target's quantization config** (#34622). Both are directly relevant if
we enable MTP on a quantized checkpoint in our SGLang fork — #34622 in particular
would silently quantize a draft head that should stay BF16.

PR #34795 is the most useful single datapoint: tuned-vs-heuristic Triton MoE speedups
of **1.59x at M=1, 1.87x at M=16, 1.84x at M=64, 1.79x at M=256**, and end-to-end on
8xH20 TP8·PP4: **BS1 52.38 → 60.79 tok/s (+16.1%), BS16 366.28 → 513.30 tok/s
(+40.1%)** [verified]. **We will need to autotune MoE configs for
`E=512,N=256,dtype=fp8_w8a8,block_shape=[128,128]` on SM100 — no B200 config exists
upstream.** A 1.6–1.9x kernel-level gain is on the table for free.

### TensorRT-LLM — not ready

| Issue/PR | Title | State |
|---|---|---|
| [#17723](https://github.com/NVIDIA/TensorRT-LLM/issues/17723) | `[Bug]: Qwen3.8-27B NVFP4 cannot load with the PyTorch backend` | **open** |
| [#17700](https://github.com/NVIDIA/TensorRT-LLM/pull/17700) | Perf: Qwen3.5/3.8 MoE, attention-DP, weight loading | **draft** |
| #17786 | Normalize Qwen3.8 28B FP8 VLM quantization config | draft |
| #17724/#17725 | Fixes for Qwen3.8 dense VLM and compressed-tensors parsing | draft |
| #17649 | Remove stale Qwen3.5 and DeepSeekV32 waivers | — |

Issue #17723 lists five independent blockers including "the explicit FP8 `lm_head`
entry is dropped, which would cast the FP8 weight to bf16 and discard its scale"
[verified]. No workaround published.

PR #17700 is worth reading even though it is draft — it contains **Gated DeltaNet
kernel work directly applicable to `k3-kernels/`**: consolidating multiple PyTorch
kernel launches into a single Triton kernel for GDN "replay work item preparation",
and **"specialized tuning for 8:1 value-head shape ratios with pipelined all-layer
commit"** [verified]. 8:1 is exactly `linear_num_value_heads 128 : linear_num_key_heads
16` on the 2.4T (and 64:16 = 4:1 on the 397B). It also adds
`TRT_LLM_PAGEOUT_WEIGHTS_AFTER_MOE_LOAD` for "multi-hundred-GiB checkpoint scenarios."

### TileRT

No published Qwen3.5/3.8 support found. Our local `GLM-5.2-FP8-TileRT` directory is
empty [verified — `ls /home/aman/code/weights/GLM-5.2-FP8-TileRT`]. Nothing to report.

---

## 8. Known-good serving configs, in full

All verbatim from `recipes.vllm.ai` and HF model cards [verified].

### Qwen3.8-2.4T-A95B — hardware requirements as published

```
BF16          : 24 GPUs (B300/MI355X) or 48 GPUs (H200)
FP8           : 16 GPUs (B300/MI355X) or 32 GPUs (H200); 4 GB300 trays (TP16)
MXFP4 (AMD)   : 8 GPUs (MI355X) or 16 GPUs (H200)
NVFP4 W4A4    : 8 GPUs (B300) or 16 GPUs (H200); 2 GB300 trays (TP8)

Constraint: "TP must divide the 64 attention heads, so only 1/2/4/8/16/32 are legal."
```

**Note the NVFP4 row says B300 (288 GB), not B200 (183 GB).** This is the published
confirmation of §2.1.

### Qwen3.8-2.4T-A95B — low latency, NVFP4 TP8

```bash
vllm serve Inferact/Qwen3.8-2.4T-A95B-NVFP4 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### Qwen3.8-2.4T-A95B — multi-node FP8, TP16 across 2 nodes

```bash
vllm serve Qwen/Qwen3.8-2.4T-A95B-FP8 \
  --tensor-parallel-size 16 \
  --nnodes 2 --node-rank 0 --master-addr $HEAD_ADDR \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3
# rank > 0 adds: --headless
```

**This is the one that would run on our hardware, if we had a second node.**

### Qwen3.8-2.4T-A95B — high throughput, expert parallel

```bash
# FP8, TP4 · DP4 + EP16
vllm serve Qwen/Qwen3.8-2.4T-A95B-FP8 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --moe-backend flashinfer_trtllm \
  --enable-expert-parallel \
  --all2all-backend flashinfer_nvlink_one_sided

# NVFP4, DEP16
vllm serve Inferact/Qwen3.8-2.4T-A95B-NVFP4 \
  --moe-backend flashinfer_cutedsl \
  --enable-expert-parallel \
  --all2all-backend flashinfer_nvlink_one_sided
```

### Qwen3.8-2.4T-A95B — environment and load tuning

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

# cut weight load 545 s -> 306 s on the 1.32 TiB checkpoint:
--load-format fastsafetensors --safetensors-load-strategy lazy
```

### Qwen3.8-2.4T-A95B — published measurements (NVFP4 TP8 on B300, 8K in / 1K out)

| Concurrency | TPS/user | Total TPS/GPU | TPOT |
|---|---|---|---|
| 1 | 101 | 108 | 9.9 ms |
| 32 | 41.9 | 1,338 | 23.9 ms |
| 64 | 26.9 | 1,715 | 37.2 ms |
| 128 | 18.3 | 2,170 | 54.7 ms |

Headline claims: NVFP4 TP8 **304 tok/s/user with MTP-3**; FP8 TP16 **307 tok/s/user
with MTP-3**; throughput up to **4,300 tok/s/GPU (NVFP4 DEP16)** and **3,200
tok/s/GPU (FP8 TP4·DP4+EP16)**.

> Per-stream falls **5.5x from C1 to C128** here (101 → 18.3). Our GLM-5.2 falls 4.7x
> from C1 to C16. Different model, but the same disease, and worse.

### Qwen3.5-397B-A17B — the recipes that matter for our box

```bash
# THROUGHPUT, text-only  (recommended cost-per-user starting point)
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  -dp 8 --enable-expert-parallel \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching

# THROUGHPUT, multimodal
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  -dp 8 --enable-expert-parallel \
  --mm-encoder-tp-mode data --mm-processor-cache-type shm \
  --reasoning-parser qwen3 --enable-prefix-caching

# LATENCY  (recommended single-stream starting point)
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser qwen3

# Tool calling: append
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
# Disable thinking: append
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

SGLang equivalents [verified — model card]:
```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-397B-A17B \
  --port 8000 --tp-size 8 --mem-fraction-static 0.8 \
  --context-length 262144 --reasoning-parser qwen3

# with MTP
python -m sglang.launch_server --model-path Qwen/Qwen3.5-397B-A17B \
  --port 8000 --tp-size 8 --mem-fraction-static 0.8 \
  --context-length 262144 --reasoning-parser qwen3 \
  --speculative-algo NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4

# NVIDIA NVFP4 checkpoint
python3 -m sglang.launch_server --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tensor-parallel-size 4 --quantization modelopt_fp4 --trust-remote-code
```

### Long context (YaRN) — 262K → 1.01M

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --hf-overrides '{"text_config": {"rope_parameters": {
      "mrope_interleaved": true, "mrope_section": [11, 11, 10],
      "rope_type": "yarn", "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --max-model-len 1010000
```

`factor: 2.0` → ~524K, `factor: 4.0` → ~1.01M. **YaRN here is static scaling: it
applies at all lengths and degrades short-context quality.** Both Qwen model cards warn
about this explicitly [verified]. Do not enable it on a general endpoint.

### Sampling parameters (published)

| Model / mode | temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Qwen3.8-2.4T | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Qwen3.8-27B thinking (default) | 1.0 | 0.95 | 20 | — | 0.0 |
| Qwen3.8-27B instruct/non-thinking | 0.7 | 0.80 | 20 | — | **1.5** |
| Qwen3.5-397B thinking | 0.6 | 0.95 | 20 | 0.0 | — |
| Qwen3.5-397B non-thinking | 0.7 | 0.80 | 20 | 0.0 | — |

Output-length guidance for the 397B: 32,768 tokens standard, 81,920 for complex
reasoning [verified — model card]. Qwen3.8 adds a `reasoning_effort` control
(`xhigh`/`medium`/`low`) and "preserved thinking" which retains reasoning blocks
across turns **by default** — that inflates prefill on multi-turn agentic workloads
and interacts badly with prefix caching. Worth measuring.

### Troubleshooting, published

- **Mamba/CUDA-graph cache assertion**: lower `--max-cudagraph-capture-size` (default
  512) or set `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
- **Engine ready timeout on huge checkpoints**: `VLLM_ENGINE_READY_TIMEOUT_S=3600`,
  and probe `/v1/chat/completions` rather than `/health`.
- **MXFP4 will not load on NVIDIA** — missing linear method.

---

## 9. Open questions and what to measure on our box

1. **Confirm the 397B-FP8 resident footprint.** Predicted 406 GB weights, 50.75 GB/GPU
   at TP8. Measure `torch.cuda.memory_allocated` after load. If it exceeds 60 GB/GPU,
   something is replicating that should not be.

2. **Resolve the `nvidia/Qwen3.5-397B-A17B-NVFP4` size discrepancy.** HF reports
   `usedStorage` 751 GB against a derived ~215–250 GB across 11 shards. Sum the actual
   shard sizes before allocating. [unverified — the one number in this document I do
   not trust.]

3. **Measure per-all-reduce latency and skew at 8 KB on our NV18 topology.** My entire
   §3.4 argument rests on ~3 µs, back-derived from GLM-5.2's 19.6%/2.74 ms. Measure it
   directly with an 8 KB all-reduce microbenchmark, and separate fixed cost from
   arrival skew. **If it is 3 µs, the 397B is collective-bound and kernel work on GEMMs
   is wasted effort.**

4. **Is the FlashInfer all-reduce + RMSNorm fusion active on SM100?** SGLang exposes
   `--enforce-disable-flashinfer-allreduce-fusion` as a B300 flag, implying default-on
   elsewhere. If it is off on B200, turning it on is the cheapest available win.

5. **Count collectives empirically.** Predicted 120/token for the 397B at TP8. Verify
   with an NCCL trace. Then test whether the DeltaNet `o_proj` all-reduce can fuse into
   the following MoE block — worth 45 of 120.

6. **Autotune the MoE Triton config for SM100.** Upstream has H20 configs only for
   `E=512,N=256,fp8_w8a8,block_shape=[128,128]` (SGLang #34795). That PR got 1.59–1.87x
   kernel-level over the heuristic on H20. **Nothing exists for B200. This is probably
   the largest single free win available.**

7. **Verify the MTP head loads, then measure acceptance directly.** Check for
   "unexpected weight" warnings mentioning `mtp.lm_head`. Then measure accept length at
   depths 1/2/3/4 on real data. The published +3.4% for MTP-1 vs 96.8% acceptance
   cannot both be right (§6); find out which applies to our checkpoint.

8. **Confirm SGLang #34622 does not bite us**: does our fork let the MTP draft inherit
   the target's quantization config? On an FP8 checkpoint that would quantize a head
   the official ignore-list keeps in BF16.

9. **Measure the DeltaNet state pool.** Predicted 188 MiB/seq for the 397B, 574 MiB/seq
   for the 2.4T. Confirm the engine allocates fp32 and that the pool is sized by
   `max_num_seqs`, not by `max_model_len`. Then find the concurrency at which state,
   not KV, is the binding constraint.

10. **Write/port a Gated DeltaNet kernel and profile it.** 45 of 60 layers on the 397B
    are DeltaNet. Our measured GLM-5.2 profile (attention 10.9%, DSA indexer 5.8%) has
    no analogue for this. **Profile first — I would not be surprised if DeltaNet is
    30–40% of decode time on an untuned stack.** TensorRT-LLM PR #17700's "8:1
    value-head ratio tuning with pipelined all-layer commit" is the reference to steal.

11. **Find the expert-amortization crossover empirically.** §3.6 predicts 72% of 512
    experts touched at B=64 and 99% at B=256. Measure actual expert-touch counts and
    the resulting tok/s curve. **If the model does not reach GLM-5.2's 40.8k tok/s at
    any concurrency the memory allows, that is a decisive argument against switching.**

12. **Test TP2 · DP4 + EP8.** Unpublished, but the natural fit for
    `num_key_value_heads = 2`. Compare against the published DP8+EP8 at C=256.

13. **Quantify the BF16 → FP8 quality delta.** Nobody has published it, and the official
    FP8 checkpoint quantizes 99.3% of parameters. Only the FP8 → NVFP4 delta is public.

14. **Measure "preserved thinking" prefill cost** on Qwen3.8-27B and its interaction
    with prefix caching on multi-turn agentic traces.

---

## 10. Sources

Everything below was fetched during this session on 2026-08-17.

**Ground-truth configs (config.json / HF API):**
- [Qwen/Qwen3.8-2.4T-A95B config.json](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/config.json)
- [Qwen/Qwen3.8-27B config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
- [Qwen/Qwen3.8-2.4T-A95B-FP8 config.json](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8/raw/main/config.json)
- [Qwen/Qwen3.5-397B-A17B config.json](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)
- [Qwen/Qwen3.5-122B-A10B config.json](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/raw/main/config.json)
- [Qwen/Qwen3.5-0.8B config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json)
- [RadixArk/Qwen3.8-2.4T-A95B-NVFP4 config.json](https://huggingface.co/RadixArk/Qwen3.8-2.4T-A95B-NVFP4/raw/main/config.json)
- HF API: [2.4T](https://huggingface.co/api/models/Qwen/Qwen3.8-2.4T-A95B), [2.4T-FP8](https://huggingface.co/api/models/Qwen/Qwen3.8-2.4T-A95B-FP8), [Inferact NVFP4](https://huggingface.co/api/models/Inferact/Qwen3.8-2.4T-A95B-NVFP4), [RadixArk NVFP4](https://huggingface.co/api/models/RadixArk/Qwen3.8-2.4T-A95B-NVFP4), [27B](https://huggingface.co/api/models/Qwen/Qwen3.8-27B), [27B-FP8](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8), [397B](https://huggingface.co/api/models/Qwen/Qwen3.5-397B-A17B), [397B-FP8](https://huggingface.co/api/models/Qwen/Qwen3.5-397B-A17B-FP8), [122B](https://huggingface.co/api/models/Qwen/Qwen3.5-122B-A10B), [35B-A3B](https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B), [nvidia NVFP4](https://huggingface.co/api/models/nvidia/Qwen3.5-397B-A17B-NVFP4)

**Model cards:**
- [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4) — the only published quantization quality table
- [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) — architectural ancestor

**Serving recipes:**
- [recipes.vllm.ai/Qwen/Qwen3.8-2.4T-A95B](https://recipes.vllm.ai/Qwen/Qwen3.8-2.4T-A95B) — the single most useful source in this document
- [recipes.vllm.ai/Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [recipes.vllm.ai/Qwen/Qwen3.5-397B-A17B](https://recipes.vllm.ai/Qwen/Qwen3.5-397B-A17B)
- [recipes.vllm.ai/Qwen/Qwen3.5-122B-A10B](https://recipes.vllm.ai/Qwen/Qwen3.5-122B-A10B)
- [recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
- [recipes.vllm.ai/Qwen/Qwen3.5-0.8B](https://recipes.vllm.ai/Qwen/Qwen3.5-0.8B) — draft-model statement
- [recipes.vllm.ai/Qwen](https://recipes.vllm.ai/Qwen) — all 25 Qwen recipes
- [docs.vllm.ai recipes (Qwen3.5)](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [SGLang generative models](https://docs.sglang.io/supported_models/generative_models.html), [SGLang Qwen3 guide](https://docs.sglang.io/basic_usage/qwen3.html)

**Engine PRs and issues:**
- [vLLM #52013 — MTP draft head loading (96.8% acceptance)](https://github.com/vllm-project/vllm/pull/52013)
- [vLLM #51992 — TriangleMix (closed, spam)](https://github.com/vllm-project/vllm/pull/51992)
- [SGLang #34795 — Qwen3.8 MoE tuned configs + benchmarks](https://github.com/sgl-project/sglang/pull/34795)
- [SGLang PR search: qwen3.5](https://github.com/sgl-project/sglang/pulls?q=qwen3.5+OR+qwen3_5)
- [vLLM PR search: qwen3.5](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+qwen3.5+OR+qwen3_5)
- [TensorRT-LLM #17723 — Qwen3.8-27B NVFP4 load failure](https://github.com/NVIDIA/TensorRT-LLM/issues/17723)
- [TensorRT-LLM #17700 — Qwen3.5/3.8 perf, GDN kernels](https://github.com/NVIDIA/TensorRT-LLM/pull/17700)
- [TensorRT-LLM issue search](https://github.com/NVIDIA/TensorRT-LLM/issues?q=qwen3.5+OR+qwen3_5+OR+qwen3.8)

**Lineup and dates:**
- [github.com/QwenLM/Qwen3.6 README](https://github.com/QwenLM/Qwen3.6/blob/main/README.md) — release dates for 3.5 and 3.6
- [github.com/QwenLM](https://github.com/QwenLM)
- [huggingface.co/Qwen](https://huggingface.co/Qwen), [search: Qwen3.8](https://huggingface.co/models?search=Qwen3.8), [search: Qwen3.6](https://huggingface.co/models?search=Qwen3.6), [search: Qwen3.5](https://huggingface.co/models?search=Qwen3.5), [search: Qwen3.8-2.4T](https://huggingface.co/models?search=Qwen3.8-2.4T)
- [qwenlm.github.io/blog](https://qwenlm.github.io/blog/) — stale, last post 2025-09-23

**Not found / does not exist:** arXiv technical report for Qwen3.5/3.6/3.8;
`github.com/QwenLM/Qwen3.8` (404); `docs.vllm.ai/.../Qwen/Qwen3.8.html` (404);
readable content at `qwen.ai/blog`.

**Note on method:** WebSearch quota was exhausted before this task began (200/200
calls used by earlier work in the session), so every finding here was obtained by
fetching primary URLs directly — HF configs and API endpoints, vLLM/SGLang recipe
pages, and GitHub PR/issue pages. That biases toward primary sources, which is the
right bias for this document, but it means I could not sweep third-party benchmark
blogs. If someone has published independent B200 numbers for this family, I did not
see them.
