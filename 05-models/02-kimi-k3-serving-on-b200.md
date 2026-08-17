# Kimi K3 on 8xB200

**Research date:** 2026-08-17 · **Target box:** 8x NVIDIA B200 SXM (sm_100), 183,359 MiB/GPU,
NVLink5/NVSwitch, CUDA 13.2 · **Engine:** `/home/aman/code/NotSglang` (SGLang fork)

**Evidence labels:** `[verified]` read by me from a config.json, model card, repo file, or local
source file · `[reported]` published by a vendor/paper/blog I fetched but did not independently
check · `[inferred]` arithmetic I did from `[verified]` inputs, with the arithmetic shown ·
`[unverified]` no source found.

**Method caveat, stated up front.** This session's WebSearch budget was exhausted before I
started (200/200 used), so every external fact below comes from a *direct WebFetch of a known
URL*, not from search. I fetched the HuggingFace model card and config.json, the HF model API,
the GitHub repo README, the arXiv abstract, the SGLang and vLLM day-0 blogs, the vLLM recipes
page, the Mooncake repo, the DSpark draft-model card, the K2 config.json, the SGLang and
TensorRT-LLM PR lists, and the NVFP4 checkpoint config. What I could **not** do is
keyword-search for issues I do not already know the URL of — so "I found no evidence of X"
in this document means "no evidence in the ~18 pages I fetched", not "X does not exist".
The Chinese-language search the brief asked for did not happen for the same reason.

---

## Status

**Kimi K3 exists and is publicly released.** This is not a speculative document.

| Fact | Value | Source |
|---|---|---|
| Real public name | **Kimi K3** (`moonshotai/Kimi-K3`) | HF model card `[verified]` |
| HF repo created | 2026-06-13T06:42:57Z | HF model API `[verified]` |
| HF repo last modified | 2026-07-27T16:29:18Z | HF model API `[verified]` |
| Announced | 2026-07-16 | local `kimi-k3-onprem-serving.md` §13, citing Fortune `[reported]` |
| Open weights landed | 2026-07-26 | local `kimi-k3-onprem-serving.md` freshness note `[reported]` |
| Tech report | arXiv **2607.24653**, *"Kimi K3: Open Frontier Intelligence"*, v1 2026-07-27, v2 2026-08-07, Kimi Team (400+ authors) | arXiv abstract `[verified]` |
| Downloads | 2,136,775 | HF model API `[verified]` |
| License | Kimi K3 License (code and weights) | GitHub README `[reported]` |
| Weight files | 96 safetensors shards | HF model API `[verified]` |
| Engines with day-0 support | vLLM, SGLang, TokenSpeed | GitHub README `[reported]` |

Also real and relevant:

- **`RadixArk/Kimi-K3-DSpark`** — 2.25 B BF16 speculative draft head, published acceptance
  lengths `[verified]` from its model card. Also mirrored as `Inferact/Kimi-K3-DSpark` (vLLM's
  recipe points there) and `RedHatAI/Kimi-K3-speculator.dspark`. `[verified]` via HF listing.
- **`nvidia/Kimi-K3-NVFP4`** — an NVFP4 requantization by NVIDIA. `[verified]` (I read its
  quantization_config).
- Community derivatives: `unsloth/Kimi-K3-GGUF`, `nota-ai/Kimi-K3-Nota-Global-Pruned-50`,
  `pipenetwork/Kimi-K3-REAP73-MLX-mxfp4-q8` (131 B REAP-pruned). `[verified]` via HF listing.

### What the local prior work establishes

`/home/aman/code/NotSglang/personal_docs/kimi-k3/` contains five documents from a **paused** K3
program. They are the best grounding in this repo and I have used them throughout:

| File | What it establishes |
|---|---|
| `AGENT-HANDOFF-sm100.md` | The K3 kernel effort ran on a **single RTX PRO 6000 (sm_120)**, was written to resume on SM100/SM103, and **was never executed on this B200 box** — the effort was redirected to GLM-5.2. §7 item 7 states outright: *"8× B200 is 1440 GB against ~1514 GB of weights and does **not** fit."* Also carries the open **E8M0 scale-bias** question. |
| `k3-kernel-optimization-log.md` | 40 correctness checks, 18 spill-free kernel instantiations for `sm_100a`/`sm_103a`/`sm_120a`, and per-kernel measured results on sm_120. Critically: §1a says every headline gain is **against the log's own baseline, not SGLang**, and the one case checked (KDA decode) was a **1.02× wash** against SGLang's Triton kernel. |
| `k3-sglang-kernel-map-and-rtxpro6000-plan.md` | The SGLang kernel inventory for K3, the `KDAKernelDispatcher` arch-gate table, and the corrected expert-residency roofline (`E(T) = 896·(1−(1−1/896)^(16T))`) that replaced an 8×-wrong MBU figure. |
| `kimi-k3-onprem-serving.md` | Economics and hardware gate. Lists **8× B200 as blocked** ("weights don't fit at 8 GPUs"), 16× B200 as viable. |
| `kimi-k3-b300-mi355x-onprem-serving.md` | Exact 256K session-memory math for B300 DCP8; C56 feasibility analysis. |

Also local and load-bearing: `/home/aman/code/NotSglang/docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx`
(1,933 lines) is the **SGLang cookbook recipe source**, and it is the authority for §8 below —
better than the rendered docs page, which is a React configurator that WebFetch cannot read.

`k3-kernels/` builds and passes its 40 checks, but **has never run on this box, never run inside
SGLang, and never run on more than one GPU** (`AGENT-HANDOFF-sm100.md` §9).

---

## What this is

Kimi K3 is a **2.8 T-parameter, 104 B-active, natively multimodal, hybrid-attention MoE** with a
1 M-token context window, released as **MXFP4 weights** (quantization-aware trained from the SFT
stage onward). `[reported]`, model card + GitHub README.

Four things make it structurally unlike anything on this box today:

1. **Hybrid linear attention.** 93 layers = **69 KDA** (Kimi Delta Attention, a gated delta-rule
   linear attention) + **24 gated MLA**. `[verified]` from config.json's `linear_attn_config`.
   KDA carries a *fixed-size* recurrent state that does not grow with context; only the 24 MLA
   layers hold a KV cache. This is why 1 M context is affordable at all.
2. **Latent MoE.** The 896 routed experts do not run in the 7168-d hidden space — they run in a
   **3584-d latent space** (`routed_expert_hidden_size: 3584`), with a down-projection into the
   latent space and an up-projection out. `[verified]`.
3. **Attention residual bank** (`attn_res_block_size: 12`) — every 12th layer snapshots the
   pre-attention stream into a bank and each layer mixes a learned projection of it back in.
   `[verified]` from config; mechanism described in the local kernel map §2.1 `[reported]`.
4. **MXFP4 native.** The released checkpoint is `mxfp4-pack-quantized`, group 32, E8M0 (uint8)
   scales, and **only the routed experts are quantized** — the ignore list explicitly exempts
   `self_attn`, `shared_experts`, the dense MLP, `lm_head`, and the vision tower. `[verified]`.

The lineage is **K2 → Kimi Linear → K3**. K2 (`moonshotai/Kimi-K2-Instruct`) was a
DeepSeek-V3-architecture 1 T MoE: 61 layers, 384 routed experts, top-8, 1 shared, moe_intermediate
2048, **pure MLA with 64 heads**, YaRN RoPE (theta 50000, factor 32), 131 K context, FP8
block-128 quantization, `num_nextn_predict_layers: 0`. `[verified]` from its config.json.
Kimi Linear (arXiv 2510.26692, 2025-10-30) introduced **KDA** as a finer-grained-gating refinement
of Gated DeltaNet using DPLR transition matrices, in a layerwise KDA+MLA hybrid, claiming up to
**75 % KV-cache reduction and up to 6× decoding throughput at 1 M context** on a 48 B-A3B model.
`[reported]` from the arXiv abstract. K3 is that architecture scaled 58× in total parameters,
with NoPE MLA, latent MoE, the attention-residual bank, and a vision tower added.

The MuonClip/optimizer story matters here only in one respect that touches the released weights:
the K3 blog and tech report describe **MXFP4 weights / MXFP8 activations applied as
quantization-aware training from the SFT stage onward** `[reported]`, and SGLang's day-0 post
reports RL rollout/train KL holding flat at *"the ~2e-3 MXFP4 floor"* `[reported]`. The practical
consequence for serving: **MXFP4 is the native numeric format of this checkpoint, not a
post-hoc compression.** Converting away from it is a downgrade, not an upgrade — see §5.

---

## Bottom line for serving it on 8xB200

Ranked, concrete, and the first item dominates everything else.

1. **Kimi K3 does not fit on this node, at any published quantization. Do not plan around it.**
   Released MXFP4+BF16 weights are **1,449.6 GiB**; this box has **1,432.5 GiB** of physical HBM
   (8 × 183,359 MiB, read from `nvidia-smi`). That is a 17.1 GiB deficit against *raw* HBM with
   zero bytes for KV, activations, CUDA graphs, or NCCL buffers — and a **232 GiB deficit**
   against the `--mem-fraction-static 0.85` budget every published recipe uses. `[inferred]`,
   arithmetic in §2. Both upstream engines agree independently: vLLM's recipe says *"At least
   8x GB300"* `[reported]`; SGLang's cookbook has **no single-node B200 cell at all** — every
   `hw: "b200"` recipe is `nnodes: 2` `[verified]` from the local jsx.

2. **NVFP4 makes it worse, not better.** NVFP4 is group-16 with E4M3 scales = 0.5625 B/param
   against MXFP4's group-32 E8M0 = 0.53125 B/param. On 2.72 T routed-expert parameters that is
   **+85 GB**. `nvidia/Kimi-K3-NVFP4` claws some back by also quantizing `self_attn` to FP8, but
   nets out at **1,479.8 GiB — 30 GiB worse than the released MXFP4 checkpoint.** `[inferred]`,
   §5. This is the opposite of the GLM-5.2 situation on this box and it is the single most
   counterintuitive result in this document.

3. **The realistic path is 2 nodes, and then the choice is TP16 vs TP8×PP2 — a real trade.**
   TP16 spans both nodes, which puts all **186 collectives per decode step** (93 layers × 2)
   on the inter-node fabric. At a 20 µs small-message all-reduce that is **3.72 ms of pure
   latency per token**, against a 1.04 ms HBM roofline — the network becomes 78 % of the step.
   TP8×PP2 keeps every collective on NVLink and crosses the network once per token, but
   **DSPARK requires `pp_size == 1`** `[verified]` from the SGLang recipe source, so PP2 forfeits
   the 3× speculative win. Measure both; do not assume. §3, §4.

4. **If you run K3 on 8 GPUs at all, run plain TP — never EP — for latency.** The shared experts
   are 12.155 B **BF16, unquantized** parameters (24.31 GB), and SGLang **replicates them at
   tp1 on every rank under EP all-to-all** (`self._shared_experts_tp1 = self._ep_a2a`)
   `[verified]` from `models/kimi_k3.py:519`, with the code's own comment confirming
   "~264 MB per layer per rank". At C1 that turns 3.04 GB/GPU of shared-expert traffic into
   24.31 GB/GPU — **it takes the per-GPU decode roofline from 483 tok/s to 211 tok/s.**
   `[inferred]`, §3. SGLang's own B200 and B300 *low-latency* cells carry no `--moe-a2a-backend`
   flag, which is consistent.

5. **Speculative decoding is the largest single lever, and K3 ships a real one.** There is **no
   MTP head** — `num_nextn_predict_layers: 0` `[verified]`. Instead there is DSPARK, a separate
   2.25 B draft model with 5 Qwen3-style GQA layers reading auxiliary target layers
   [7, 23, 51, 67, 83], block size 7. Published acceptance lengths: **5.51 (HumanEval), 5.42
   (GSM8K), 4.26 (RULER-V2 1M), 2.99 (AIME26)** `[reported]` from the DSpark card; SGLang reports
   ~2.7 on chat and ~5.0 on few-shot math `[reported]`. At accept-5.4 the roofline gain at C1 is
   **2.4×** `[inferred]`, §6 — and vLLM measures 111 → 331 tok/s (3.0×) in practice `[reported]`.

6. **The KV story is unusually cheap and the *state* story is unusually expensive.** Only 24 of
   93 layers hold KV: **13.5 KiB/token at FP8** — 3.375 GiB for a full 256 K sequence, 13.5 GiB
   at 1 M `[inferred]`. But the KDA recurrent state is **context-independent and per-slot**:
   232.3 MB per slot at BF16 unsharded, and SGLang reserves 4–5 slots per request. At TP8 with
   `--mamba-radix-cache-strategy extra_buffer_lazy` that is **116.2 MB/GPU/request before a
   single token of context.** `[inferred]` from the recipe's own ratio formula. At short context
   the state pool, not KV, sets the concurrency ceiling.

7. **`--enable-linear-replayssm-spec` is not optional once you turn on DSPARK.** Without it, a
   block-7 draft adds 8 more state slots per request — 116 MB → **348.5 MB/GPU/request at TP8,
   a 3× jump.** `[inferred]`. SGLang reports ReplaySSM cutting draft-window memory ~32×
   `[reported]`, and every DSPARK recipe in the cookbook layers it on except the PD-prefill role
   `[verified]`.

8. **Your existing hotspot profile transfers with one large correction.** The GLM-5.2 C1 split
   (dense GEMM 37 %, collectives 20 %, MoE 19 %, attention 11 %) will not hold. K3 at C1 is
   **62 % attention-and-shared-expert weight streaming** (KDA linears 61.2 GB + shared experts
   24.3 GB node-wide, both BF16) and only **19 % routed-expert traffic** — because at T=1 only
   15.9 of 896 experts are touched. `[inferred]`, §3. The bottleneck moves from "MoE" to "the
   BF16 parts nobody quantized".

9. **For our two objectives specifically:** minimum single-stream latency on K3 is a
   *collective-count* problem (186/step over 93 layers) far more than a bandwidth problem —
   the HBM roofline at TP16 is 1.04 ms/token but published bs=1 is ~8.5 ms, i.e. **12 % MBU**
   `[inferred]` from vLLM's 111–118 tok/s `[reported]`. Minimum cost/user is a
   *distinct-experts-touched* problem: at C64 the model reads 68 % of all 896 experts every
   step, so marginal tokens get nearly free past ~C128. §3, §4.

10. **What to actually do on this box:** keep GLM-5.2 in production, and use the box to settle
    the three K3 questions that only need *one* GPU or *one* node and that the paused program
    left open — the E8M0 scale bias, `deep_gemm.fp8_fp4_mega_moe` vs `k3-kernels`' MoE decode,
    and the 186-collective cost at TP8 on a byte-equivalent block stack. §9.

---

## 1. Architecture table

Everything in this table is `[verified]` from `https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`
unless the Source column says otherwise. The text config is nested under `text_config`; the
inner `model_type` is `kimi_linear` and the inner architecture is `KimiLinearForCausalLM`,
wrapped by `KimiK3ForConditionalGeneration`.

| Property | Value | Source |
|---|---|---|
| Total parameters | **2.78 T** (card says 2.8 T) | HF API `safetensors.total`; my block-structure derivation gives 2.7778 T — agrees to 0.8 % `[inferred]` |
| Active parameters/token | **104 B** | model card `[reported]` |
| Parameter dtype split | BF16 57.2 B · U8 (packed MXFP4) · F32 11.1 M | HF API `[verified]` |
| Layers | **93** | `num_hidden_layers: 93` |
| — KDA (linear attn) | **69**, layers 1,2,3,5,6,7,9,… | `linear_attn_config.kda_layers` (69 entries) |
| — full attention (MLA) | **24**, layers 4,8,12,…,88,92,93 | `linear_attn_config.full_attn_layers` (24 entries) |
| Hidden size | **7168** | `hidden_size` |
| Dense MLP intermediate (layer 0 only) | **33792** | `intermediate_size` + `first_k_dense_replace: 1` |
| Activation | **`situ`**, β=4.0, linear β=25.0 | `hidden_act`, `activation_situ_beta`, `activation_situ_linear_beta` |
| **MoE — routed experts** | **896** | `num_experts` |
| **MoE — top-k** | **16** | `num_experts_per_token` |
| **MoE — shared experts** | **2** | `num_shared_experts` |
| MoE intermediate | **3072** | `moe_intermediate_size` |
| **Latent MoE width** | **3584** | `routed_expert_hidden_size` — experts run here, not at 7168 |
| Router | sigmoid, `noaux_tc`, grouped top-k, 1 group, renormalize, scaling 1.0 | `moe_router_activation_func`, `topk_method`, `num_expert_group: 1`, `topk_group: 1`, `moe_renormalize`, `routed_scaling_factor` |
| Latent norm | on | `latent_moe_use_norm: true` |
| **MLA — q_lora_rank** | **1536** | `q_lora_rank` |
| **MLA — kv_lora_rank** | **512** | `kv_lora_rank` |
| MLA — qk_nope / qk_rope / v head dim | **128 / 64 / 128** | `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` |
| MLA heads | **96** | `num_attention_heads` (= `num_key_value_heads`) |
| **MLA uses NoPE** | **true** | `mla_use_nope: true` — see RoPE note below |
| MLA output gate | **true** | `mla_use_output_gate: true` |
| **KDA — heads × head_dim** | **96 × 128** | `linear_attn_config.num_heads`, `.head_dim` |
| KDA — short conv kernel | **4** | `.short_conv_kernel_size` |
| KDA — full-rank gate | **true** | `.use_full_rank_gate` |
| KDA — gate lower bound | **−5.0** | `.gate_lower_bound` |
| **Attention residual block size** | **12** | `attn_res_block_size` |
| Vocab | **163,840** | `vocab_size` (card rounds to "160K") |
| **Context length** | **1,048,576** | `max_position_embeddings` |
| **RoPE config** | **none present** — no `rope_theta`, no `rope_scaling`, and `mla_use_nope: true` | absence in config.json `[verified]`; that `qk_rope_head_dim: 64` still occupies head width while NoPE is on is `[unverified]` — resolve by reading `modeling_kimi_linear.py` |
| **MTP / draft heads in-model** | **`num_nextn_predict_layers: 0` — none** | `[verified]`. Speculation ships as a *separate* DSPARK model, §6 |
| Quantization | `mxfp4-pack-quantized`, num_bits 4, group_size 32, symmetric, `scale_dtype: torch.uint8` (E8M0), `quant_method: compressed-tensors` | `quantization_config` |
| Quantization **ignore** list | `self_attn`, `shared_experts`, `mlp.(gate\|up\|gate_up\|down)_proj`, `lm_head`, `vision_tower`, `mm_projector` | `[verified]` — **only routed experts are 4-bit** |
| KV cache scheme in checkpoint | `null` (no shipped KV quant) | `quantization_config.kv_cache_scheme` |
| Tokenizer | tiktoken-based (`tiktoken.model`, `tokenization_kimi.py`, `encoding_k3.py`); BOS 163584, EOS 163586, PAD 163839, image placeholder `<\|kimi_image_placeholder\|>` id 163605 | HF file list + config `[verified]` |
| Vision tower | MoonViT-V2, 27 layers, hidden 1024, intermediate 4096, 12 heads, patch 14, 2×2 `sd2_tpool` merge, `patchmergerv2` projector, **401 M params** | `vision_config` `[verified]`; 401 M from README `[reported]` |

**Derived per-layer parameter budget** `[inferred]` — this is the table the roofline runs on.

| Component | Params | Bytes at released precision |
|---|---:|---:|
| One routed expert (3 × 3584 × 3072) | 33.03 M | 17.55 MB MXFP4 |
| One MoE layer, all 896 experts | 29.595 B | 15.72 GB MXFP4 |
| All 92 MoE layers, routed experts | **2.7227 T** | **1,446.5 GB** |
| One KDA block (q,k,v,g,o @ 7168×12288 + full-rank gate + conv) | 443.7 M | 887 MB BF16 |
| 69 KDA blocks | 30.617 B | 61.2 GB BF16 |
| One MLA block | 144.1 M | 288 MB BF16 |
| 24 MLA blocks | 3.459 B | 6.9 GB BF16 |
| **Shared experts, all 92 layers (7168→6144, BF16)** | **12.155 B** | **24.3 GB BF16** |
| Latent up/down projections, 92 layers | 4.727 B | 9.5 GB BF16 |
| Router gates, 92 layers | 0.591 B | 1.2 GB BF16 |
| Dense MLP (layer 0) | 0.727 B | 1.5 GB BF16 |
| Embedding + lm_head (untied) | 2.349 B | 4.7 GB BF16 |
| MoonViT-V2 | 0.401 B | 0.8 GB BF16 |
| **Sum of non-expert** | **55.03 B** | **110.1 GB** |

Cross-check: HF reports 57.2 B BF16 parameters; my structural sum is 55.03 B. The 2.2 B gap is
norms, per-head KDA betas, the attention-residual bank projections, and conv weights I did not
enumerate. **The model is validated to ~4 %.** The 443.7 M/KDA-layer and 144.1 M/MLA-layer
figures reproduce the local kernel map §2.2 exactly, which is an independent derivation.

---

## 2. Memory arithmetic on 8xB200

### 2.1 The box, measured

```
$ nvidia-smi --query-gpu=memory.total --format=csv
183,359 MiB  x 8  =  1,466,872 MiB
             =  1,432.5 GiB
             =  1,538.1 GB (decimal)
             =    179.1 GiB per GPU
```
`[verified]` — run on this machine. Note the brief's "183 GB each" is MiB read as GB; the
decimal figure is 192.3 GB/GPU. **I use GiB throughout §2 to avoid this trap.**

### 2.2 Weight bytes, four ways

MXFP4 = 4 data bits + one 8-bit E8M0 scale per 32 elements = **0.53125 B/param**.
NVFP4 = 4 data bits + one E4M3 scale per 16 elements = **0.5625 B/param**.

| Configuration | Expert bytes | Non-expert bytes | **Total** | /8 GPUs | /16 GPUs |
|---|---:|---:|---:|---:|---:|
| **Released: MXFP4 experts + BF16 rest** | 1,446.5 GB | 110.1 GB | **1,556.5 GB = 1,449.6 GiB** | **181.2 GiB** | 90.6 GiB |
| NVFP4 experts + BF16 rest | 1,531.5 GB | 110.1 GB | 1,641.6 GB = 1,528.9 GiB | 191.1 GiB | 95.6 GiB |
| **`nvidia/Kimi-K3-NVFP4`: NVFP4 experts + FP8 attn/rest** | 1,531.5 GB | 57.4 GB | 1,588.9 GB = **1,479.8 GiB** | 185.0 GiB | 92.5 GiB |
| *Hypothetical* MXFP4 experts + FP8 everything-but-embeddings | 1,446.5 GB | 57.4 GB | 1,503.8 GB = **1,400.6 GiB** | 175.1 GiB | 87.5 GiB |
| *Hypothetical* BF16 everything | 5,555 GB | 110.1 GB | 5,665 GB = 5,276 GiB | — | — |

`[inferred]` from §1's parameter table. Cross-check: AMD measured **190.974 GiB/GPU at TP8**
for the loader-aware text-only checkpoint `[reported]` via the local B300 doc = 1,527.8 GiB
node-wide, against my 1,449.6 GiB pure-weight figure. The 78 GiB gap is loader replication and
the vision tower/alignment padding — consistent with the local doc's "~5 % loader replication"
`[reported]`.

### 2.3 Does it fit? No.

```
Released weights                                1,449.6 GiB
8x B200 physical HBM                            1,432.5 GiB
                                                -----------
Deficit against RAW HBM                           -17.1 GiB   (-1.2%)
Deficit at --mem-fraction-static 0.85            -232.0 GiB
Deficit at --mem-fraction-static 0.92            -131.7 GiB
```

`[inferred]`. **The 1.2 % raw-HBM deficit is the cruelest number in this document** — K3 misses
by almost nothing, and then misses by 232 GiB once you need anywhere to put the KV cache, the
KDA state pool, CUDA graph pools, NCCL/symmetric-memory buffers, and chunked-prefill scratch.
A servable configuration needs roughly **25–45 GiB/GPU of non-weight memory** (the local B300
doc budgets 24 GiB for runtime/graphs/workspaces plus ~30 GiB of session state at C56
`[reported]`), i.e. **200–360 GiB node-wide**, so weights must come in under ~1,100 GiB. From
1,449.6 GiB that is a **24 % reduction** — and no published checkpoint achieves it:

- FP8 everything non-expert: → 1,400.6 GiB. Still 32 GiB over raw HBM. **No.**
- NVFP4 (either variant): **worse.** See §5.
- Both engines' minimums confirm this independently: vLLM *"At least 8x GB300"* `[reported]`;
  SGLang has zero single-node B200 cells `[verified]`.

### 2.4 What *does* fit, and at what parallelism

| Config | GPUs | HBM | Weights/GPU | Free/GPU @0.85 | Verdict |
|---|---:|---:|---:|---:|---|
| **8× B200, TP8** | 8 | 1,432.5 GiB | 181.2 GiB | **negative** | ❌ does not load |
| **16× B200, TP16** (2 nodes) | 16 | 2,865 GiB | **90.6 GiB** | **~61 GiB** | ✅ SGLang's published shape |
| 16× B200, TP8×PP2 | 16 | 2,865 GiB | 90.6 GiB | ~61 GiB | ✅ SGLang long-context cell |
| 8× B300 | 8 | 2,146 GiB | 181.2 GiB | ~47 GiB | ✅ the reference platform |
| 8× MI355X | 8 | 2,146 GiB | 181.2 GiB | ~47 GiB | ✅ (ROCm feature gaps) |
| 8× B200 + REAP-50 expert prune | 8 | 1,432.5 GiB | ~97 GiB `[inferred]` | ~85 GiB | ⚠️ fits, quality delta unpublished |

At **TP16** the released weights land at 90.6 GiB/GPU, leaving ~61 GiB/GPU at
`--mem-fraction-static 0.85` after weights — comfortable. At **TP8×PP2** the per-GPU footprint
is identical (PP splits layers, TP splits within a stage), which is why the SGLang long-context
cell can afford `--context-length 131072` with `extra_buffer` rather than the lazy strategy.

### 2.5 Per-request session memory

**MLA KV** — only 24 of 93 layers cache anything: `24 × (kv_lora_rank 512 + qk_rope 64)` elements
per token `[verified]` from config, matching the cookbook calculator's `kvBytesPerToken = 24 * (512 + 64) * kvBytes`
`[verified]` from the local jsx.

```
per token:  BF16  27,648 B = 27.0 KiB     FP8  13,824 B = 13.5 KiB
```

| Context | FP8/seq | ÷DCP8 | ÷DCP16 |
|---:|---:|---:|---:|
| 32 K | 0.422 GiB | 0.053 GiB | 0.026 GiB |
| 128 K | 1.688 GiB | 0.211 GiB | 0.105 GiB |
| 256 K | 3.375 GiB | 0.422 GiB | 0.211 GiB |
| 1 M | 13.500 GiB | 1.688 GiB | 0.844 GiB |

**KDA state** — context-independent, per slot, sharded by attention-TP:
`69 × (96/attn_tp × 128 × 128 × ssm_bytes + 3 × 3 × 96/attn_tp × 128 × 2)` `[verified]`, the
cookbook's own formula.

```
unsharded per slot:   FP32  449.4 MB      BF16  232.3 MB
TP8 per slot:         FP32   56.2 MB      BF16   29.0 MB
TP16 per slot:        FP32   28.1 MB      BF16   14.5 MB
```

Slot counts, from the cookbook's `_calculate_mamba_ratio` mirror `[verified]`:

| Strategy | Slots | TP8 BF16 | TP16 BF16 |
|---|---:|---:|---:|
| `extra_buffer` + overlap scheduler (default) | 5 | 145.2 MB | 72.6 MB |
| `extra_buffer_lazy` | 4 | **116.2 MB** | 58.1 MB |
| `extra_buffer_lazy` + DSPARK-7 **without** ReplaySSM | 12 | **348.5 MB** | 174.2 MB |

The 116.2 MB/GPU/request at TP8 reproduces the local B300 doc's 0.108 GiB figure exactly
`[reported]`. **At any context below ~90 K, the state pool costs more per request than the KV
cache does** — 116 MB of state vs 0.422 GiB/8 = 54 MB of DCP8 KV at 128 K. That inversion is why
`--mamba-full-memory-ratio` exists and why SGLang ships a calculator for it.

---

## 3. Roofline — show the arithmetic

This is the section that decides everything. All of it is `[inferred]` from §1's parameter
table, with B200 HBM3e taken at **8.0 TB/s per GPU** and NVLink5 latency figures labelled where
used.

### 3.1 The expert-residency function — why per-token cost is not batch-independent

With top-16 routing over 896 experts, the number of **distinct** experts a step touches is

```
E(T) = 896 · (1 − (1 − 1/896)^(16·T))          T = tokens in the step
```

`[reported]` from the local kernel map §4.1, and it is the single most important correction in
the K3 prior work — the earlier analysis treated expert traffic as batch-independent and was
wrong by ~8×.

| T (tokens/step) | distinct experts | of 896 | expert bytes/step, node-wide |
|---:|---:|---:|---:|
| 1 | 15.9 | 1.8 % | 25.61 GB |
| 2 | 31.5 | 3.5 % | 50.78 GB |
| 4 | 61.8 | 6.9 % | 99.77 GB |
| 8 | 119.3 | 13.3 % | 192.66 GB |
| 16 | 222.8 | 24.9 % | 359.65 GB |
| 32 | 390.2 | 43.5 % | 629.88 GB |
| 64 | 610.4 | 68.1 % | 985.47 GB |
| 128 | 805.0 | 89.8 % | 1,299.54 GB |
| 256 | 886.8 | 99.0 % | 1,431.53 GB |

(node-wide bytes = `92 layers × E(T) × 17.55 MB/expert`.)

Past ~T=128 essentially the entire model is read every step and marginal tokens are nearly free.
**Interactive latency, not memory, is what stops you getting there.**

### 3.2 Bytes read per decoded token at C1 — and why EP is a trap

Node-wide weight traffic for one decode step at T=1:

```
routed experts   92 × 15.9 × 17.55 MB   =  25.61 GB
shared experts   12.155 B × 2 B         =  24.31 GB    <-- BF16, unquantized
KDA linears      30.617 B × 2 B         =  61.24 GB    <-- BF16, unquantized
latent + gate     5.318 B × 2 B         =  10.64 GB
MLA linears       3.459 B × 2 B         =   6.92 GB
dense MLP L0      0.727 B × 2 B         =   1.45 GB
lm_head           1.174 B × 2 B         =   2.35 GB
                                          ---------
                                          132.52 GB node-wide
```

Now divide by parallelism. The critical distinction is whether the shared experts are sharded:

| Mode | experts | **shared** | rest | **total/GPU** | ms @ 8 TB/s | **tok/s ceiling** |
|---|---:|---:|---:|---:|---:|---:|
| **TP8, all sharded** | 3.20 GB | **3.04 GB** | 10.32 GB | **16.56 GB** | 2.071 | **483** |
| TP8 + EP a2a (shared replicated) | 3.20 GB | **24.31 GB** | 10.32 GB | 37.84 GB | 4.729 | **211** |
| **TP16, all sharded** | 1.60 GB | **1.52 GB** | 5.16 GB | **8.28 GB** | 1.035 | **966** |
| TP16 + EP a2a (shared replicated) | 1.60 GB | **24.31 GB** | 5.16 GB | 31.07 GB | 3.884 | **257** |

**EP costs 2.3× at TP8 and 3.8× at TP16 in single-stream latency, entirely because of one
unquantized 24.3 GB weight slab that SGLang replicates on every rank.** `[verified]` that the
replication happens (`models/kimi_k3.py:519`, comment at :532 confirms "~264 MB per layer per
rank"); `[inferred]` for the roofline consequence.

SGLang partially mitigates this with **SBO (single-batch overlap)**: the shared experts are
issued on a side stream concurrently with the a2a-latency-bound routed path, measured at
**+4–5 % output tok/s and −5 % ITL over bs 1–32** on 2×4 GB300 `[reported]` from the code
comment. That reclaims overlap, not bytes.

Notice also the composition at C1, TP8: **KDA linears alone are 7.66 GB/GPU = 46 % of the step**,
and routed experts only 19 %. K3's C1 decode is dominated by the parts of the model *nobody
quantized*.

### 3.3 Collective bytes and count per token

**186 collectives per decode step** = 93 layers × 2 (attention output reduce + MoE reduce).
`[reported]` from the local handoff §7 and kernel map §3.1; consistent with the architecture.

Payload per collective at C1: `hidden_size 7168 × 2 B = 14,336 B` per token per rank.
**This is a latency problem, not a bandwidth problem** — 14 KB is far below any NVLink5
message-size knee.

| Fabric | small-msg AR latency | 186 × latency | + 2.07 ms HBM (TP8) | tok/s |
|---|---:|---:|---:|---:|
| NVLink5, symm-mem one-shot | ~4 µs `[unverified]` | 0.74 ms | 2.81 ms | **355** |
| NVLink5, stock NCCL | ~8 µs `[unverified]` | 1.49 ms | 3.56 ms | **281** |
| **Cross-node IB/RoCE 400 G** | ~20 µs `[unverified]` | **3.72 ms** | 5.79 ms | **173** |

The latency constants are `[unverified]` — I could not fetch a measured NVLink5 all-reduce
latency curve, and **measuring them on this box is the single cheapest high-value experiment**
(§9). But the *shape* of the conclusion is robust to a 2× error in any of them:
**at TP16 across two nodes, the fabric is 3–4× the HBM roofline and owns the step.**

This is why SGLang's B200 long-context cell is `--tp-size 8 --pp-size 2`, not TP16: PP crosses
the network **once per token** with a 14 KB activation instead of 186 times.

### 3.4 The resulting latency floor, and how far reality is from it

| Config | HBM floor | + collectives | **floor** | published bs=1 | **MBU** |
|---|---:|---:|---:|---:|---:|
| TP8 (needs B300/MI355X to fit) | 2.07 ms | +0.74 ms | 2.81 ms → 355 tok/s | **111** (vLLM), **113** (SGLang) `[reported]` | ~12 % |
| TP16, 2 nodes | 1.04 ms | +3.72 ms | 4.76 ms → 210 tok/s | **118** (vLLM) `[reported]` | ~7 % |

`[inferred]` against `[reported]` measurements. **The gap at C1 is ~3× and it is not in any
single kernel** — the local kernel map §5 makes exactly this point, and the fact that TP16
buys only +6 % over TP8 in vLLM's own numbers despite 2× the bandwidth is direct evidence that
bandwidth is not what binds.

Per-layer budget at 8.5 ms/step: **91 µs per layer** against an 11 µs roofline. For comparison,
this box's GLM-5.2 NVFP4 TP8 runs TPOT 2.74 ms — K3's *floor* at TP16 is worse than GLM-5.2's
achieved latency.

### 3.5 The throughput curve — cost per user

Per-GPU bytes at TP8, DCP8, FP8 KV, full 256 K contexts, no EP replication:

| C | experts | shared | rest | MLA KV | **total/GPU** | ms | tok/s/user | **aggregate** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.20 | 3.04 | 10.32 | 0.45 | **17.02 GB** | 2.13 | 470 | 470 |
| 4 | 12.47 | 3.04 | 10.32 | 1.81 | 27.65 | 3.46 | 289 | 1,158 |
| 8 | 24.08 | 3.04 | 10.32 | 3.62 | 41.07 | 5.13 | 195 | 1,558 |
| 16 | 44.96 | 3.04 | 10.32 | 7.25 | 65.57 | 8.20 | 122 | 1,952 |
| 32 | 78.73 | 3.04 | 10.32 | 14.50 | 106.59 | 13.32 | 75 | 2,402 |
| **64** | 123.18 | 3.04 | 10.32 | 28.99 | **165.54** | 20.69 | **48** | **3,093** |
| 128 | 162.44 | 3.04 | 10.32 | 57.98 | 233.79 | 29.22 | 34 | 4,380 |

At TP16/DCP16: C1 → 940 tok/s, C16 → 244/user (3,904 agg), C64 → 97/user (6,186 agg).

`[inferred]`. Compare the one real anchor: **541 aggregate tok/s at 48 sessions on 2×4 GB300**
`[reported]` from SGLang's day-0 post — against a ~2,800 tok/s roofline at C48, i.e. **~19 %
MBU**, which independently reproduces the local kernel map's corrected 17.4 % figure `[reported]`.
Two unrelated derivations agreeing is the strongest single piece of evidence in this document.

**Sublinearity check:** aggregate goes 470 → 3,093 from C1 to C64, only **6.6×** for 64× the
concurrency, because expert bytes grow as `E(T)`. Per-stream speed falls **9.7×** over that
range — roughly twice as steep as the 4.7× you measure on GLM-5.2 from C1 to C16, and the cause
is 896-way top-16 routing decorrelating across the batch.

### 3.6 Prefill

MLA is the only quadratic term (69 KDA layers are linear in context). Materialized MLA prefill
costs 640 FLOP per (query, key, head) — `2×(qk_nope 128 + qk_rope 64)` for QK plus
`2×(v_head_dim 128)` for PV `[verified]` from config dims, and matching the kernel log's stated
"640 materialized" `[reported]`.

```
MLA prefill FLOP = 24 layers × 96 heads × 640 × ctx² / 2   (causal)
MoE/linear FLOP  = 2 × 104e9 × ctx
```

| Context | MoE/linear | MLA quadratic | Total | attn share |
|---:|---:|---:|---:|---:|
| 32 K | 6.8 PF | 0.8 PF | 7.6 PF | 10 % |
| 128 K | 27.3 PF | 12.7 PF | 39.9 PF | 32 % |
| **256 K** | 54.5 PF | **50.7 PF** | **105.2 PF** | 48 % |
| 1 M | 218.1 PF | **810.6 PF** | 1,028.8 PF | **79 %** |

`[inferred]`. The 256 K MLA figure of 50.7 PFLOP reproduces the local kernel log §10a's
50.7 PFLOP exactly — independent confirmation of the constant. At 1 M, **79 % of prefill is
attention, which does not run in FP4**, so the model spends four fifths of a cold 1 M prefill in
the slow numeric path. This is the root cause of K3's 1 M TTFT problem and no amount of MoE
kernel work touches it.

---

## 4. Parallelism recommendation

### For minimum single-stream latency

**On 8× B200: not achievable — the model does not load.** The honest recommendation is a second
node.

**On 16× B200 (2 nodes), for latency:** the two candidates are genuinely close and the answer
depends on a number nobody has published.

| | TP16 (SGLang low-latency cell) | TP8 × PP2 (SGLang long-context cell) |
|---|---|---|
| Weights/GPU | 90.6 GiB | 90.6 GiB |
| Collectives crossing the fabric | **186/token** | **1/token** |
| Fabric cost/token @20 µs | 3.72 ms | 0.02 ms |
| HBM floor/token | 1.04 ms | 2.07 ms (TP8 within stage) |
| **Modeled floor** | **4.76 ms → 210 tok/s** | **2.09 ms → 478 tok/s** |
| DSPARK speculation | ✅ | ❌ — requires `pp_size == 1` `[verified]` |
| Effective with spec @accept-5.4 | ~2.4× → ~500 tok/s | n/a |
| Pipeline bubbles at C1 | none | none (single request, no pipelining to lose) |

`[inferred]`. **The two land in the same place**, which is exactly why this needs measurement
rather than argument. My recommendation: **start at TP8×PP2 non-speculative** because its floor
does not depend on an unmeasured fabric constant, get a clean number, then A/B TP16+DSPARK.
If your inter-node fabric is better than 10 µs for a 14 KB all-reduce, TP16+DSPARK wins.

**Never enable EP for latency.** §3.2: replicating 24.31 GB of BF16 shared experts on every rank
costs 2.3–3.8× on the roofline. SGLang's own low-latency cells carry no `--moe-a2a-backend`.

**Always enable DCP** (`--dcp-size 16`) once context is long: it shards the MLA latent KV by
token position for one extra all-to-all per MLA layer (24, not 93), and SGLang reports
**~7.9× logical KV capacity on K3 at DCP8** `[reported]`. But note the compatibility cliffs
`[verified]` from the recipe source: HiCache L3 (Mooncake) storage keys are not `dcp_rank`-aware
and **drop DCP**, and HiCache-under-DCP **rejects speculative decoding outright**.

### For minimum cost per user

**16× B200, TP16 + DCP16 + EP16 + DSPARK + ReplaySSM, `--mem-fraction-static 0.92`.**

Reasoning: at high concurrency the shared-expert replication penalty amortizes — 24.31 GB/GPU
is a fixed cost per *step*, and at C64 the routed experts alone are 123 GB/GPU, so the
replication goes from 64 % of traffic at C1 to 16 % at C64. Meanwhile EP's real payoff (bigger
grouped GEMMs per expert, a2a instead of full all-reduce) grows with batch. The crossover is
around **C8–C16** `[inferred]` — measure it.

Add PD disaggregation only when you have ≥2 decode replicas to amortize another full weight
copy; at 1,449.6 GiB per copy, a prefill pool is a very expensive thing to stand up. SGLang's
own K3 PD cells exist for B200 (`--disaggregation-mode prefill|decode`,
`--disaggregation-transfer-backend nixl`, `--disaggregation-decode-extra-slots 16`) `[verified]`.

---

## 5. Quantization

### What exists

| Format | Where | Expert bytes/param | Total | Verdict for this box |
|---|---|---:|---:|---|
| **MXFP4** (group 32, E8M0 uint8 scale) | **`moonshotai/Kimi-K3`** — the native checkpoint, QAT'd from SFT onward | 0.53125 | 1,449.6 GiB | Reference. Still doesn't fit 8 GPUs |
| **NVFP4** (group 16, E4M3 scale) + FP8 attn | `nvidia/Kimi-K3-NVFP4` | 0.5625 | **1,479.8 GiB** | **Worse by 30 GiB** |
| BF16 | not published | 2.0 | 5,276 GiB | Irrelevant |
| GGUF Q2_K / IQ1_S | `unsloth/Kimi-K3-GGUF`, others | — | — | Wrong stack (llama.cpp), not SGLang/vLLM |
| REAP-pruned | `nota-ai/...Global-Pruned-50` (1.4 T), `pipenetwork/...REAP73` (131 B) | — | ~97 GiB/GPU at 8 | ⚠️ **the only thing that fits 8× B200**; quality delta unpublished |

`[verified]` for the checkpoints' existence and the NVFP4 config's quantization groups (group_0:
`experts`, 4-bit float, group_size 16; group_1: `self_attn` projections, `FP8_PB_WO`).
`[inferred]` for the byte totals.

### The NVFP4 result, stated plainly

**On K3, NVFP4 is a memory regression.** This is the opposite of what NVFP4 does for GLM-5.2 on
this box, and the reason is that K3's checkpoint is *already 4-bit*. NVFP4's finer group (16 vs
32) costs 0.03125 B/param more in scale bytes, which on 2.72 T expert parameters is **+85 GB**.
NVIDIA's checkpoint offsets that by taking `self_attn` from BF16 to FP8 (−52.7 GB), but the net
is **+30 GiB**. `[inferred]`.

The local handoff already said this and it is worth repeating verbatim `[reported]`:

> Weights are MXFP4 (e2m1, group 32, E8M0 scale), *not* NVFP4 (group 16, E4M3). **Do not
> convert:** it costs +86 GB of capacity, +5.9 % decode traffic, and accuracy, since the
> released weights are already MXFP4.

My independent computation gives +85 GB. The two agree.

There is one narrow case where NVFP4 is right: if your MoE kernel path is materially faster on
NVFP4 than MXFP4 on SM100 and you have the memory to spare (i.e. you are on B300 or 16× B200),
the +30 GiB may buy throughput. SGLang PR **#35077** — *"[Fix] Correct Kimi-K3 ModelOpt NVFP4
accuracy with TRT-LLM MoE"* (draft, opened 2026-08-17) `[verified]` — suggests that path is
being actively debugged for accuracy and is **not yet trustworthy**.

### What to keep in higher precision

The checkpoint has already made these choices for you, and they are correct `[verified]` from
the `ignore` list:

- **`self_attn`** — all MLA and KDA linears stay BF16. 34.1 B params, 68.2 GB. The delta-rule
  recurrence and the MLA latent projections are numerically delicate, and 69 KDA layers compound
  any error 69 times.
- **`shared_experts`** — BF16. These see *every* token; error here is not averaged away by
  routing.
- **`lm_head`**, the dense MLP at layer 0, the vision tower, and `mm_projector` — BF16.

**If you must find bytes**, the ranked order is: (1) FP8 the KDA/MLA linears (−34 GB, the
`nvidia` checkpoint already does this and it is the least-risky 4-bit-adjacent move);
(2) FP8 the shared experts (−12 GB, higher risk); (3) do not touch the routed experts — they are
QAT'd at MXFP4 and going lower has no published checkpoint or evaluation.

### KV cache format

- `quantization_config.kv_cache_scheme: null` — **the checkpoint ships no KV quantization**
  `[verified]`.
- Every SGLang B300/GB300 high-throughput recipe sets **`--kv-cache-dtype fp8_e4m3`**
  `[verified]`; it halves KV from 27.0 to 13.5 KiB/token.
- vLLM's FP8 KV requires the extra
  `--attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"flashinfer"}'`
  `[reported]`.
- **KDA state dtype is a separate knob:** `--mamba-ssm-dtype bfloat16` halves the state pool
  (449.4 → 232.3 MB/slot unsharded) and the local kernel log measures the BF16-state KDA decode
  kernel at **1.95× wall clock** for half the bytes `[reported]`. The B300 high-throughput recipe
  sets it `[verified]`. Default is FP32 and is warned-but-accepted for drift.
- **FP4/INT4 latent KV is not worth chasing.** KV is 17 % of decode traffic at C56 and halving it
  buys ≤8 % `[reported]` from the local kernel map §4.3. Ranked #5 there and #9 on AMD's own
  roadmap; both are right.

---

## 6. Speculative decoding

**K3 ships no MTP head.** `num_nextn_predict_layers: 0` `[verified]` — same as K2, which also
had 0. Anyone telling you K3 has an MTP layer is confusing it with DeepSeek-V3.

Speculation ships as **DSPARK**, a separately-published draft model.

### `RadixArk/Kimi-K3-DSpark` `[verified]` from its model card

| Property | Value |
|---|---|
| Parameters | **2.25 B**, BF16 |
| Draft body | 5 full-attention Qwen3-style GQA layers, hidden 7168, **64 Q heads / 16 KV heads** |
| Auxiliary target layers read | **[7, 23, 51, 67, 83]** |
| Block size | **7** tokens/block |
| Verify window | 1 current + 7 draft = **8 tokens** |
| Context | up to 1 M; trained at 65,536 with **YaRN-16** |
| Embedding/unembedding | **omitted** (shares the target's) |

**Published acceptance lengths** `[reported]`:

| Benchmark | acc_len | Source |
|---|---:|---|
| **HumanEval** | **5.51** | DSpark card |
| GSM8K | 5.42 | DSpark card |
| RULER-V2 @1M | 4.26 | DSpark card |
| AIME26 | 2.99 | DSpark card |
| chat workload | ~2.7 | SGLang day-0 blog |
| few-shot math | ~5.0 | SGLang day-0 blog |
| low-entropy tasks | 4.73 | vLLM blog |
| high-entropy tasks | 2.61 | vLLM blog |

**For a coding workload, HumanEval's 5.51 is the number to plan on.**

### Roofline gain at C1, TP8 `[inferred]`

Verifying 8 tokens/step raises distinct experts touched from 15.9 to 119.3, so the step gets
*more expensive* — the win is that you emit `acc_len` tokens for it.

```
base C1 :  T=1, 17.02 GB/GPU, 2.13 ms  ->  470 tok/s
spec-7  :  T=8, 37.44 GB/GPU, 4.68 ms  ->  acc_len / 4.68 ms
```

| acc_len | tok/s roofline | speedup vs C1 base |
|---:|---:|---:|
| 2.70 (chat) | 577 | 1.19× |
| 4.26 (1M) | 910 | 1.88× |
| 5.42 (GSM8K) | 1,158 | **2.40×** |
| 5.51 (HumanEval) | 1,177 | **2.44×** |

Measured reality is *better* than this roofline, which tells you the base case is not
bandwidth-bound: vLLM reports **111 → 331 tok/s at TP8 (3.0×)** and **118 → 370 at TP16 (3.14×)**
`[reported]`; SGLang reports **~113 → ~423** `[reported]`. Speculation is attacking the
collective-latency and launch-overhead floor, not just bytes — which is consistent with the 12 %
MBU in §3.4.

**One caution, from the local prior work** `[reported]`: *"DSPARK's 3.14× is a batch-1 latency
ratio; published aggregate effects are +18 % to +51 %. Use 1.2–1.5× for cost modeling."* SGLang
separately reports verify-trimming gains of **+68 % and +24 % at bs 256** `[reported]`.

### Recommended configuration

```
--speculative-algorithm DSPARK
--speculative-draft-model-path RadixArk/Kimi-K3-DSpark
--speculative-dspark-block-size 7
--enable-linear-replayssm-spec
```

`[verified]` — this is exactly what SGLang's cookbook emits for every DSPARK cell except the
PD-prefill role. vLLM's equivalent `[reported]`:

```
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark",
  "num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA",
  "draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

Hard constraints, all `[verified]` from the recipe source:

- **`pp_size == 1`.** DSPARK and pipeline parallelism are mutually exclusive. This is the single
  most consequential constraint for a 2-node B200 deployment (§4).
- **`--enable-linear-replayssm-spec` is effectively mandatory** — without it, block-7 adds 8
  state slots per request (116 → 348 MB/GPU at TP8). It needs the Triton linear-attn decode
  backend, which is K3's default. Rejected at startup on a PD-prefill server.
- **Ragged verify `compact` mode is silently inert** without
  `--speculative-dspark-sps-table-path`, and **fails fast with ReplaySSM or DCP > 1**.
- **HiCache under DCP rejects speculation at startup** (the draft host pool has no DCP index
  translation), so those recipes drop `--dcp-size`.
- **DFLASH is listed but not selectable** — no K3 DFLASH draft checkpoint has been published.
- EAGLE is not offered for K3; the cookbook's `--speculative-num-steps` strip-prefix exists only
  so switching families does not leave a stale flag.

---

## 7. Engine support status

### SGLang — the most complete path

`[verified]` from the local fork, which carries the K3 branch merged to `main`
(292 files, +51,956/−1,538):

- `srt/models/kimi_k3.py`, `kimi_k3_vl.py`, `srt/configs/kimi_k3.py`
- Fused JIT kernels under `kernels/jit/csrc/kimi_k3/`: `comm/gemm_ar.cuh` (GEMM+all-reduce, 70 KB),
  `comm/gemm_ag.cuh`, `comm/ar_fusion.cuh` (MNNVL AR + residual, 41 KB), `comm/sp_collective.cuh`,
  `attn_res/fused_tma.cuh` (43 KB), `situ_and_mul{,_masked_post_quant}.cuh`, `mla_output_gate.cuh`
- KDA suite: `kda_blackwell/kernel_h.py`, `cutedsl_kda.py`, `kda_fused_decode.py`,
  `kda_packed_decode.py`, `fla/kda.py`, `kimi_k3/kda_decode_mtp.py` (48 KB CuTe DSL)
- Function calling: `kimik3_detector.py`, `kimik3_format.py`, `kimik3_structural_tag.py`
- A hand-written **PTX KDA prefill kernel** (commit `f37baeb52`, PR **#32819**) `[reported]`

**Kernel arch gating — the B200-relevant part.** `KDAKernelDispatcher` selects among six
backends `[verified]` from the local kernel map §2.4:

| Backend | Gate | On our B200 (sm_100, cc 10.0) |
|---|---|---|
| `triton` | none | ✅ always available |
| `cutedsl` | `major >= 10` | ✅ |
| `flashinfer` | `major >= 10` | ✅ |
| `nvidia_kda` | `major == 10` | ✅ |
| `ptx_kda` | `== (10, 3)` | ❌ **GB300 only — B200 is 10.0** |
| `flashkda` | none visible | `[unverified]` |

**Good news for this box:** all four of the arch-gate bugs the local program fixed
(`kda_cutedsl.py`, `kda_flashinfer.py`, `gdn_cutedsl.py`, `kimi_k3/attn_res.py`, all the same
`major >= 10` accepting consumer SM120) are **irrelevant here** — B200 reports 10.0 and takes
the fast path either way. The one remaining flagged-but-unfixed item,
`gdn_flashinfer.py`'s `use_state_pool = sm_major >= 10`, is likewise moot on 10.0.

**The one real SM100-vs-SM103 gap:** the PTX KDA prefill kernel is gated `== (10, 3)`. On B200
you get the CuTe-DSL path instead. The local map notes this proves 10.3 has instructions 10.0
lacks `[reported]`. Long-context TTFT on B200 will be worse than on GB300 for this reason.

**Open/recent SGLang PRs touching K3** `[verified]` from the PR list (89 open, 154 closed
matching "kimi k3" as of 2026-08-17):

| PR | Title | State |
|---|---|---|
| **#35077** | [Fix] Correct Kimi-K3 ModelOpt **NVFP4** accuracy with TRT-LLM MoE | draft, 2026-08-17 |
| #34987 | [multimodal] Bound tokenizer-side GPU memory in Kimi GPU image preprocessing | open, 2026-08-16 |
| #34985 | [AMD] Add the Kimi-K3 MI35x perf sweep inside the accuracy job | draft |
| #34944 | [NPU] fix(dspark): preserve parity in folded NPU paths | open |
| **#34842** | Revert "[Fix] Disable `--enable-symm-mem` under CUDA graphs on Kimi hybrid models" | draft, 2026-08-14 |
| **#34760** | Fix Mamba state donation misalignment in unified radix cache **under DCP** | open, 2026-08-13 |
| #34883 | [Kimi-K3] Use explicit SiTU activation for MegaMoE | **merged** 2026-08-15 |
| #34766 | [Fix] Carry the backend on Kimi-K3 deferred preprocessing configs | **merged** |
| #34984 | [AMD] Make the Kimi-K3 MI35x nightly accuracy-only | **merged** |

Two of these matter for us. **#34842** is a revert of a symm-mem/CUDA-graph disable on Kimi
hybrid models — and the B300 low-latency recipe carries the comment *"No `--enable-symm-mem`:
it makes the fused all-reduce auto-probe skip"* `[verified]`, so symmetric memory and the fused
all-reduce path are in active flux. **#34760** is a Mamba-state donation bug **under DCP**,
which is exactly the configuration the balanced recipes use.

**Verification status:** every K3 cell in the cookbook carries
`verified: false, verificationStatus: "in-progress"` — including all B300 and B200 cells.
`[verified]`. Treat every recipe as "runs" and not "validated".

### vLLM — day-0, GB300 minimum

`[reported]` from the day-0 blog and recipes page:

- Docker `vllm/vllm-openai:kimi-k3` (CUDA), `vllm/vllm-openai_rocm:kimi-k3` (ROCm).
  **"Only Docker images usable currently due to complicated dependencies."**
- Hardware minimum: **"At least 8x GB300. Multi-node for real production traffic."**
  16× B200 is listed as a supported configuration.
- FlashKDA CUTLASS prefill; fused CUDA decode kernel folding causal conv + recurrent update +
  RMSNorm into one launch; dedicated K3 KDA metadata builder replacing generic GDN —
  **870 µs → 34 µs at bs=1, a 96 % reduction**.
- Custom reduce-scatter/all-gather **1.7×–4.5× faster than NCCL** at prefill message sizes.
- Known limitations: prefix caching **off by default** for K3; the vision encoder has
  `head_size=12` and **cannot be evenly sharded at TP=8**, so it runs data-parallel; K3
  occasionally emits tool-call formats its own parser rejects.
- PRs: **#45845** (interval-based retention for hybrid models), **#47782** (Marconi-style
  selective retention), **#49502** (partial block cache hits + KV offloading).

### TensorRT-LLM — real and active, less mature

`[verified]` from the PR list:

| PR | Title | State |
|---|---|---|
| #17784 | [TRTLLM-15177] Kimi K3: inline model modules, drop dead MoE comm plumbing | open, 2026-08-16 |
| #17741 | feat/k3 weight pipeline opt | open, 2026-08-15 |
| #17684 | [feat] Remove padding in Kimi K3 MLA module | open, 2026-08-14 |
| **#17624** | [TRTLLM-15284][feat] add Kimi K3 **SiTU MegaMoE** support | **merged** 2026-08-15 |
| #17500 | [feat] support Kimi K3 with breakable CUDA graphs | closed, 2026-08-11 |

MLA, SiTU MegaMoE, KVCacheManagerV2 and CUDA-graph support are landing. I found **no** TRT-LLM
PR referencing DSPARK, MXFP4, or a published K3 recipe. `[unverified]` whether TRT-LLM can serve
K3 end-to-end today.

### TileRT

**No evidence found.** TileRT is not mentioned in any K3 source I fetched, and there is no K3
TileRT checkpoint in `/home/aman/code/weights/` (only GLM-5.2 variants). `[unverified]`.

### The E8M0 scale-bias question — still open, and cheap to settle here

The local handoff calls this the one correctness question that outranks all benchmarking
`[reported]`:

> `k3-kernels` emits strict OCP bias 127 (byte 127 = 2^0), matching SGLang, CUTLASS's
> `float_ue8m0_t`, CUTLASS's host reference (`gett.hpp:560`), the OCP spec and the released
> checkpoints — all verified directly. But driving `mma.sync...block_scale` on **sm_120**
> measured a neutral byte of **128**, a factor of 4.

**Status:** I found no ecosystem resolution in anything I fetched, and could not search.
`[unverified]`. But three things narrow it substantially:

1. The observation was made on **`mma.sync...kind::mxf4`**, an SM120 instruction. **B200 uses
   `tcgen05.mma...block_scale` instead** — a different instruction with a different operand
   path. The sm_120 result may simply not apply here.
2. Five independent authorities (OCP spec, CUTLASS `float_ue8m0_t`, CUTLASS host reference,
   SGLang, and the released checkpoint's `scale_dtype: torch.uint8` under a
   `compressed-tensors` mxfp4 config `[verified]`) all say bias 127. A lone disagreeing
   measurement from a hand-written PTX harness is far more likely to be a harness bug.
3. **The decisive experiment needs one GPU and about half an hour**, and this box has eight:
   dequantize one real K3 expert tile through SGLang's bias-127 path and compare against the
   same tile through `tcgen05.mma` block-scaled. The checkpoint is the authority. §9.

---

## 8. Known-good serving configs, in full

Every command below is transcribed **verbatim** from
`/home/aman/code/NotSglang/docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` `[verified]`,
which is the SGLang cookbook's source of truth. Model path is `moonshotai/Kimi-K3`. Docker image
for B200/GB200 is **`lmsysorg/sglang:kimi-k3`**. **Every cell is `verified: false /
verificationStatus: "in-progress"`.**

### There is no single-node 8×B200 recipe

Confirmed by the config header `[verified]`:

```
// B300 (1x8 TP8), GB300 (2x4 TP8 MNNVL), B200 (2x8 TP16, or TP8/PP2 for
// Long-Context), GB200 (4x4 TP16 MNNVL), H200 (2x8 TP16/EP16, or 4x8 TP32/EP32
// for High-Throughput), H100 (4x8 TP32/EP32), and MI350X/MI355X (1x8 TP8)
```

Every `hw: "b200"` cell carries `nnodes: 2`.

### B200 2×8, TP16 — Low-Latency (unified)

```bash
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 16 \
  --mem-fraction-static 0.85 \
  --disable-flashinfer-autotune \
  --watchdog-timeout 3600 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --host 0.0.0.0 --port 30000
```

### B200 2×8, TP16 + DCP16 — Balanced (unified)

```bash
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 16 \
  --dcp-size 16 \
  --mem-fraction-static 0.85 \
  --disable-flashinfer-autotune \
  --watchdog-timeout 3600 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --host 0.0.0.0 --port 30000
```

High-Throughput is the same command with `--mem-fraction-static 0.92` plus a large-scale DP×EP
overlay.

### B200 2×8, TP8 × PP2 — Long-Context (unified)

```bash
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 8 \
  --pp-size 2 \
  --mem-fraction-static 0.85 \
  --context-length 131072 \
  --chunked-prefill-size 8192 \
  --mamba-radix-cache-strategy extra_buffer \
  --disable-flashinfer-autotune \
  --watchdog-timeout 3600 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --host 0.0.0.0 --port 30000
```

Note `--context-length 131072`, not 1 M — even at 16 GPUs the long-context cell caps at 128 K.

### B200 2×8 — PD-disaggregated decode role (balanced)

```bash
  --tp-size 16 --dcp-size 16 --mem-fraction-static 0.85 \
  --disaggregation-decode-extra-slots 16 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend nixl
```

### Required cross-node environment for B200 `[verified]`

```bash
GLOO_SOCKET_IFNAME=<your-nic>   # bootstrap interface
NCCL_SOCKET_IFNAME=<your-nic>   # force NCCL off kube-ipvs0
SGLANG_HOST_IP=<this-node-ip>
NCCL_IB_HCA=<hca0,hca1,...>     # RDMA fabrics only
```

### Reference single-node: B300 1×8 (the platform K3 was tuned on)

```bash
# Low-Latency  (note: no --enable-symm-mem, it makes the fused AR auto-probe skip)
  --trust-remote-code --model-path moonshotai/Kimi-K3 \
  --tp-size 8 --mem-fraction-static 0.85 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3

# Balanced
  --tp-size 8 --dcp-size 8 --disable-custom-all-reduce --mem-fraction-static 0.85

# High-Throughput overlay
  --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy --mem-fraction-static 0.92
```

### Speculative overlay (any `pp_size == 1` cell)

```bash
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7 \
  --enable-linear-replayssm-spec
```

### vLLM `[reported]`

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
# env: VLLM_USE_RUST_FRONTEND=1  VLLM_USE_V2_MODEL_RUNNER=1
# --all2all-backend flashinfer_nvlink_one_sided  (NVLink) | deepep_v2 (RDMA)
# --moe-backend deep_gemm_mega_moe               (GB200/GB300, disaggregated)
```

### AMD MI350X/MI355X 1×8 `[verified]`, for contrast — it fits in 8 GPUs

```bash
SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1 \
python -m sglang.launch_server --model-path moonshotai/Kimi-K3 --trust-remote-code \
  --tp-size 8 --attention-backend triton --dtype bfloat16 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 256 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

### Published performance anchors — all of them

| Number | Config | Source |
|---|---|---|
| **111 tok/s** bs=1 | TP8 | vLLM blog `[reported]` |
| **118 tok/s** bs=1 | TP16 | vLLM blog `[reported]` |
| **331 tok/s** bs=1 + DSpark | TP8 (3.0×) | vLLM blog `[reported]` |
| **370 tok/s** bs=1 + DSpark | TP16 (3.14×) | vLLM blog `[reported]` |
| **~113 tok/s** bs=1 | GB300, no spec | SGLang blog `[reported]` |
| **~423 tok/s** bs=1 + DSpark | GB300 | SGLang blog `[reported]` |
| **2,808 tok/s per GPU** | PP8 prefill + TP8 decode, 2×4 GB300, MXFP4 | SGLang blog `[reported]` |
| **2,633 tok/s per GPU** | PP8 + DCP8 | SGLang blog `[reported]` |
| **541 tok/s aggregate @ 48 sessions** | DCP8 agentic workload, 2×4 GB300 (= 11.3/user) | SGLang blog `[reported]` |
| DCP8 logical KV capacity | **~7.9×** | SGLang blog `[reported]` |
| ReplaySSM draft-window memory | **~32× cut** (512 KB → 16 KB/req) | SGLang blog `[reported]` |
| KDA metadata builder | **870 µs → 34 µs** at bs=1 | vLLM blog `[reported]` |
| PP8×TP1 prefill | **1.7× TEP8's ceiling** at 8K | SGLang blog `[reported]` |
| GSM8K / GPQA-D / OCRBench / MMMU-Pro-Vision | 0.976 / 0.939 / 0.889 / 0.818 | vLLM blog `[reported]` |
| GPQA Diamond / DeepSWE / BrowseComp / Video-MME | 93.5 / 67.5 / 91.2 / 90.0 | model card `[reported]` |

**No published number exists for 8× or 16× B200.** Every anchor above is GB300, B300, or
unspecified.

### Moonshot's own serving stack: Mooncake

The brief asks how Mooncake maps onto one node vs a cluster. `[reported]` from the repo README:

- **KVCache-centric disaggregated architecture** separating prefill and decode clusters, with a
  distributed KV pool built from otherwise-idle CPU, DRAM, and SSD.
- **Transfer Engine** — topology-aware multi-NIC RDMA, measured at **87 GB/s on 4×200 Gbps and
  190 GB/s on 8×400 Gbps RoCE, ~2.4–4.6× faster than TCP**.
- **Mooncake Store** — distributed KV/weight cache with replication and eviction.
- **Mooncake EP / Process Group** — DeepEP-style expert-parallel dispatch with fault tolerance.
- Production result: **"enables Kimi to handle 75 % more requests while adhering to SLOs."**
- **Best Paper, FAST 2025** (usenix.org/system/files/fast25-qin.pdf); production traces released.
- Also used for RL weight sync: **1 T-param Kimi-K2 weight updates 53 s → 7.2 s (7×)** via
  zero-copy RDMA.

**Mapping onto our 8× B200 node:** essentially none of the distributed machinery applies —
chunked pipeline parallelism, early-rejection scheduling, and cross-node KV pooling are all
answers to problems that appear at cluster scale. The **one** component that is directly useful
on a single node (or a 2-node pair) is **Mooncake as SGLang's HiCache L3 backend**, via
`--enable-hierarchical-cache --hicache-storage-backend mooncake` plus
`SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` and a `mooncake_master` process on rank 0 `[verified]`.
That gives you host-DRAM and NVMe prefix tiering, which for agentic K3 traffic (>90 % prefix-hit
rates are reported `[reported]`) is worth more than any kernel. **Caveat, `[verified]` from the
recipe source: L3 storage keys are not `dcp_rank`-aware, so enabling it drops `--dcp-size` and
runs plain TP — which costs you most of the ~7.9× KV capacity.** That is a genuine, currently
unresolved conflict between long context and prefix tiering.

---

## 9. Open questions and what to measure on our box

Ranked by value per hour of machine time. Items 1–4 need **no second node** and settle questions
the paused K3 program left open.

1. **Settle the E8M0 scale bias. ~30 minutes, one GPU.** Dequantize one real K3 expert tile
   through SGLang's bias-127 path and through `tcgen05.mma...block_scale` on sm_100a; compare.
   The checkpoint is the authority. A wrong answer is a silent 4× on every expert weight in a
   model that still runs and still benchmarks well. Note the sm_120 observation used
   `mma.sync`, a different instruction — this may resolve immediately. (`AGENT-HANDOFF-sm100.md` §4)

2. **Measure the NVLink5 small-message all-reduce latency curve at TP8.** 14 KB payload,
   8 ranks, with and without `--enable-symm-mem`, with and without `--disable-custom-all-reduce`.
   **Three of the most load-bearing numbers in §3 and §4 rest on an unverified ~4–20 µs
   constant.** You already have the tooling — the GLM-5.2 profile attributes 19.6 % to
   collectives with 47 % of that being rank-arrival skew, so the harness exists. This also
   directly serves the GLM-5.2 latency objective.

3. **Baseline `k3-kernels` against production, which was never possible on the old box.**
   `deep_gemm.fp8_fp4_mega_moe` vs `moe_decode_mxfp4`/`moe_prefill_mxfp4`; FlashInfer/FlashMLA
   vs `mla_decode`/`mla_prefill`. The sm_120 blocker (sgl-kernel published against CUDA 12) is
   gone here. **The one comparison that was made came out 1.02× — a wash.** Assume the same
   until measured. (`k3-kernel-optimization-log.md` §1a)

4. **Re-tune every tile/buffer macro for 227 KB shared memory.** `K3_MLA_KV_PAD`,
   `K3_MLA_RAW_BUFS`, `K3_MLA_PREFILL_{PAD,PPAD,SPAD}`, `K3_MOE_PREFILL_PAD`,
   `K3_MOE_DECODE_UNROLL`. Several current choices are known artifacts of sm_120's 99 KB cap —
   `mla_decode` single-buffers its FP8 landing zone specifically to fit. **Do not sweep
   `K3_MLA_TILE_S` or `K3_MLA_PREFILL_BM`** — `static_assert`s now block it and a sweep would
   return plausible numbers attached to wrong output. Re-record `bench_matrix.sh --record`
   rather than trusting the sm_120 baseline.

5. **Build the byte-equivalent block stack and measure the 186-collective cost at TP8.** Five to
   twenty K3 blocks with the true **112 experts/rank** (896/8), real shapes, MXFP4 experts +
   BF16 attention. This is the only way to get a real K3 collective number on a node that cannot
   hold K3, and per the local map §3.1 it is the harness worth building. Report bytes/token and
   % of measured stream — never wall clock.

6. **Validate `E(T)` against real routing.** K3 uses `noaux_tc` grouped top-k with
   `num_expert_group: 1`, which should mean no group constraint — but if routing correlates
   across tokens, distinct-experts-touched is *lower* than `E(T)` and the whole byte model in §3
   is conservative. Instrument the router. `[unverified]`, and it moves every number in §3.5.

7. **Confirm the shared-expert replication penalty empirically.** §3.2 predicts 2.3–3.8×; the
   mechanism is `[verified]` from source but the roofline consequence is `[inferred]`. Run the
   block stack with and without `--moe-a2a-backend megamoe` at C1 and C64 and find the crossover.

8. **Resolve whether `qk_rope_head_dim: 64` is live under `mla_use_nope: true`.** Read
   `modeling_kimi_linear.py` from the HF repo. If the 64 RoPE dims are dead weight, MLA KV drops
   from 13.5 to 12.0 KiB/token — an 11 % KV saving that nobody has claimed. `[unverified]`.

9. **Two-node questions, if a second node ever appears:** (a) TP16 vs TP8×PP2 at C1 — §4 says
   they model to the same place and the answer is a measurement; (b) whether DSPARK+TP16 beats
   non-spec TP8×PP2; (c) per-GPU inter-node bandwidth against the ~100 GB/s the 186 collectives
   need; (d) whether DCP16 works across the node boundary and at what ITL cost.

10. **Things I could not verify and someone should search for:** whether the ecosystem has
    resolved E8M0 on SM100; whether SGLang's `flashkda` backend has an arch gate; whether any
    Chinese-language source publishes an 8×B200 or 16×B200 K3 number; the quality delta of
    `nota-ai/Kimi-K3-Nota-Global-Pruned-50`, which is **the only published artifact that would
    fit this node**; and whether TileRT has any K3 path at all.

**And the standing recommendation:** none of this changes the fact that Kimi K3 cannot be served
on this box. If the goal is a frontier model on 8× B200 today, the answer remains GLM-5.2 — and
items 2 and 4 above pay for themselves on that workload regardless of whether K3 ever resumes.

---

## 10. Sources

**Fetched directly for this document (2026-08-17)**

- [`moonshotai/Kimi-K3` model card](https://huggingface.co/moonshotai/Kimi-K3)
- [`moonshotai/Kimi-K3` config.json](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json) — **ground truth for §1**
- [`moonshotai/Kimi-K3` HF API](https://huggingface.co/api/models/moonshotai/Kimi-K3) — dates, file list, dtype split
- [MoonshotAI/Kimi-K3 GitHub](https://github.com/MoonshotAI/Kimi-K3)
- [arXiv 2607.24653 — *Kimi K3: Open Frontier Intelligence*](https://arxiv.org/abs/2607.24653)
- [`RadixArk/Kimi-K3-DSpark`](https://huggingface.co/RadixArk/Kimi-K3-DSpark) — draft-model card, acceptance lengths
- [`nvidia/Kimi-K3-NVFP4` config.json](https://huggingface.co/nvidia/Kimi-K3-NVFP4/raw/main/config.json)
- [`moonshotai/Kimi-K2-Instruct` config.json](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json) — K2 baseline
- [HF model search: Kimi-K3](https://huggingface.co/models?search=Kimi-K3) — derivative checkpoints
- [LMSYS — SGLang day-0 Kimi K3 support](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support/)
- [vLLM — Kimi K3 day-0](https://vllm.ai/blog/2026-07-27-k3)
- [vLLM recipes — Kimi K3](https://recipes.vllm.ai/moonshotai/Kimi-K3)
- [Mooncake (kvcache-ai)](https://github.com/kvcache-ai/Mooncake) · [FAST '25 paper](https://www.usenix.org/system/files/fast25-qin.pdf)
- [arXiv 2510.26692 — *Kimi Linear*](https://arxiv.org/abs/2510.26692) — KDA origin
- [SGLang PRs matching "kimi k3"](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+kimi+k3)
- [TensorRT-LLM PRs matching "kimi k3"](https://github.com/NVIDIA/TensorRT-LLM/pulls?q=is%3Apr+kimi+k3)

**Local, and authoritative where cited**

- `/home/aman/code/NotSglang/personal_docs/kimi-k3/AGENT-HANDOFF-sm100.md`
- `/home/aman/code/NotSglang/personal_docs/kimi-k3/k3-kernel-optimization-log.md`
- `/home/aman/code/NotSglang/personal_docs/kimi-k3/k3-sglang-kernel-map-and-rtxpro6000-plan.md`
- `/home/aman/code/NotSglang/personal_docs/kimi-k3/kimi-k3-onprem-serving.md`
- `/home/aman/code/NotSglang/personal_docs/kimi-k3/kimi-k3-b300-mi355x-onprem-serving.md`
- `/home/aman/code/NotSglang/docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` — **the recipe source of truth for §8**
- `/home/aman/code/NotSglang/docs_new/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx` — KV/state byte formulas
- `/home/aman/code/NotSglang/python/sglang/srt/models/kimi_k3.py` — shared-expert tp1 replication under EP (line 519), quantization comment (line ~1280)
- `nvidia-smi` on this box — 8 × 183,359 MiB, compute capability 10.0

**Referenced by local docs, not re-fetched here** (`[reported]` at one remove): AMD's K3
deployment guide (190.974 GiB/GPU at TP8), SGLang PR #32819 (PTX KDA prefill),
the SGLang K3 cookbook rendered page.
