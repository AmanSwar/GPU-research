# DeepSeek V3.2 and V4 on 8xB200

## Status

**The brief for this document asked me to check whether a DeepSeek "V4" exists publicly, and
to say so at the top if it were unreleased or only rumoured. It is released. The premise needs
correcting, so I am correcting it first.**

`[verified]` As of 2026-08-17 DeepSeek-V4 is fully public: open weights on HuggingFace under
MIT, a live first-party API, an arXiv technical report, a reference implementation, and
day-0 support in both vLLM and SGLang — including in *our own fork* at
`/home/aman/code/NotSglang`, which ships `python/sglang/srt/models/deepseek_v4.py`, a `dsv4`
attention backend and a complete DSpark speculative-decoding subsystem.

| Model | Released | HF repo | Total params | Activated | Context | Precision as shipped |
|---|---|---|---|---:|---:|---|
| DeepSeek-V3 | 2024-12-26 | `deepseek-ai/DeepSeek-V3` | 671 B | 37 B | 128 K | FP8 |
| DeepSeek-R1 | 2025-01-20 | `deepseek-ai/DeepSeek-R1` | 671 B | 37 B | 128 K | FP8 |
| DeepSeek-V3-0324 | 2025-03-24 | — | 671 B | 37 B | 128 K | FP8 |
| DeepSeek-R1-0528 | 2025-05-28 | — | 671 B | 37 B | 128 K | FP8 |
| DeepSeek-V3.1 | 2025-08-21 | `deepseek-ai/DeepSeek-V3.1` | 685 B | 37 B | 128 K | FP8 |
| DeepSeek-V3.1-Terminus | 2025-09-22 | — | 685 B | 37 B | 128 K | FP8 |
| **DeepSeek-V3.2-Exp** | 2025-09-29 | `deepseek-ai/DeepSeek-V3.2-Exp` | 685 B | 37 B | 163,840 | FP8 |
| DeepSeek-V3.2 (GA) | 2025-12-01 | `deepseek-ai/DeepSeek-V3.2` | 685 B | 37 B | 163,840 | FP8 |
| **DeepSeek-V4-Flash** | 2026-04-24 | `deepseek-ai/DeepSeek-V4-Flash` | 284 B | 13 B | 1,048,576 | FP4 experts + FP8 |
| **DeepSeek-V4-Pro** | 2026-04-24 | `deepseek-ai/DeepSeek-V4-Pro` | 1.6 T | 49 B | 1,048,576 | FP4 experts + FP8 |
| **DeepSeek-V4-Flash-0731** | 2026-07-31 | `deepseek-ai/DeepSeek-V4-Flash-0731` | 1.7 T*/304 B | 13 B | 1,048,576 | FP4 experts + FP8 |
| **DeepSeek-V4-Pro-0813** | 2026-08-13 | `deepseek-ai/DeepSeek-V4-Pro-0813` | 1.7 T | 49 B | 1,048,576 | FP4 experts + FP8 |

`[verified]` Change log at https://api-docs.deepseek.com/updates/ ; org listing at
https://huggingface.co/deepseek-ai ; every `config.json` and `model.safetensors.index.json`
in this document was fetched and parsed by me, not read from a blog post.

Technical report: arXiv **2606.19348**, *"DeepSeek-V4: Towards Highly Efficient Million-Token
Context Intelligence"* `[verified — title and abstract fetched from arxiv.org/abs/2606.19348]`.
319 authors. Pre-trained on >32 T tokens.

**What is *not* released**, stated plainly so nobody chases it:
- `[verified]` There is **no V4 "Base" checkpoint of the 0731/0813 production builds** — only
  the April `-Base` repos exist (`DeepSeek-V4-Pro-Base`, `DeepSeek-V4-Flash-Base`, FP8 mixed,
  described as for continued pre-training only).
- `[unverified]` No public V5/V4.5. Nothing in the change log, the HF org listing, or the API
  docs suggests one. I searched in English and Chinese and found only SEO-farm speculation,
  which I have deliberately not cited.
- `[verified]` **`deepseek-chat` and `deepseek-reasoner` were deprecated after 2026-07-24.**
  The legacy V3-lineage API endpoints are gone; `deepseek-v4-pro` and `deepseek-v4-flash`
  are the live models.

**Research date: 2026-08-17.** Confidence labels: `[verified]` = I fetched the artifact and
read it, or it is measured on this box; `[reported]` = a vendor asserts it and I read the
assertion; `[inferred]` = my arithmetic or reasoning; `[unverified]` = could not source.

---

## What this is

The serving-side companion to `04-industry/04-deepseek-open-infrastructure.md`. That document
covers DeepSeek's open-source *releases* — DeepEP, DeepGEMM, FlashMLA, EPLB, the Open Source
Week posts, the DSpark paper, the V4 report's §3 systems sections. **This document does not
repeat any of it.** It does four things that document does not:

1. **Does the KV arithmetic**, from first principles, to the byte — MLA's low-rank compression,
   the exact per-token cost, and the direct comparison to GLM-5.2's architecturally identical
   576-wide latent.
2. **Reads the checkpoints, not the model cards.** Every shard header of
   `DeepSeek-V4-Pro-0813` (66 shards, 149,782 tensors) was fetched and accounted. This produced
   several findings that are not in any DeepSeek document, including the exact FP4 block size,
   the true parameter split, and hard evidence of what changed between the April and August
   production builds.
3. **Reads the kernel repos, not the READMEs.** The FlashMLA SM100 support matrix in the README
   is not what is in `csrc/`. The difference matters to us.
4. **Fills the gap the industry document flags.** That file references a section called
   *"What DeepSeek actually ships"* that does not exist in it. The production serving
   configurations are reconstructed here from the model cards and from the SGLang cookbook
   source that is sitting in our own fork.

**Why we care about a model we are not serving.** Three reasons, in order of value:

- GLM-5.2's attention is a **re-implementation of DeepSeek's**. Same 512+64 latent, same
  lightning indexer, same top-2048. Everything DeepSeek publishes about MLA and DSA kernels is
  directly, mechanically applicable to our hot path.
- DeepSeek-V4 is the **only model our fork implements two-batch overlap for**
  `[verified, our repo]`. TBO is refused for GLM-5.2. The V4 code path is the reference
  implementation to port.
- V4 shows where the architecture is going, and **the SGLang cookbook has first-party
  8xB200 benchmarks for it** at exactly the Artificial-Analysis shape. That gives us a
  calibrated competitor on identical hardware, which is rare.

---

## Bottom line for serving on 8xB200

### The five numbers that matter

| # | Fact | Consequence |
|---|---|---|
| 1 | **V4-Pro decodes at 235 tok/s on 8xB200 at C1; V4-Flash at 344.** `[verified, SGLang cookbook benchmark data in our fork]` | **We are already faster.** GLM-5.2 at 365.5 tok/s beats both. DeepSeek is not our leaderboard threat — TileRT is. |
| 2 | **V4-Flash's C1→C16 falloff is 3.0x; ours is 4.7x.** `[verified]` TPOT 2.91 → 8.76 ms. | This is the one place V4 clearly beats us, and it is our known weakness. The mechanism is DSpark plus DP-attention, both of which our fork already has code for. |
| 3 | **V4-Pro hits only 15.2% of its own HBM roofline at C1; GLM-5.2 hits 41.7%.** `[inferred]` | V4's architecture is **not bandwidth-bound at C1** — it is op-count-bound. 61 layers x (compressor + indexer + Sinkhorn + SWA branch + compressed branch) is a launch-latency problem. Do not copy V4's architecture expecting speed. |
| 4 | **V4-Pro's KV is 5,031 B/token vs GLM-5.2's 61,464 — 12.2x cheaper.** `[inferred, arithmetic below]` | This is a *capacity and cost* win, not a latency win. It buys concurrency, not TPOT. |
| 5 | **DeepSeek's own API serves V4-Pro-0813 at ~84 tok/s and V4-Flash at ~111 tok/s on Artificial Analysis; third parties reach 267.6.** `[verified, artificialanalysis.ai]` | DeepSeek optimises cost, not interactivity. Their published engineering is throughput engineering. Filter accordingly. |

### Ranked steals, specific to our fork

**Tier 0 — this week**

| # | Steal | Change | Attacks | Evidence |
|---|---|---|---|---|
| 1 | **`--enable-deepseek-v4-fp4-indexer` equivalent for GLM-5.2's DSA** | The flag already exists in our fork for V4 and routes to the DeepGEMM FP4 indexer on SM100. The kernel (`sm100_mqa_logits.cuh`) is dtype-generic — it takes `qk_dtype_t` and dispatches `SM100_MMA_MXF4_SS` for `float_e2m1_t`. GLM-5.2 uses the same `index_head_dim: 128`, which the kernel asserts as supported. | DSA indexer 5.8% of C1 | `[verified]` — I read the kernel: `kNumQKBytesPerToken = kIsFP4 ? kHeadDim/2 : kHeadDim` → **64 B/token instead of 128**, halving indexer cache bandwidth. |
| 2 | **Adopt `--page-size 256`** | Our GLM-5.2 pool asserts page_size 64. DeepSeek ship `--block-size 256` (vLLM) and our own V4 tests use `--page-size 256`. | attention 10.9%, TMA efficiency | `[verified]` both the DeepSeek model cards and `test/registered/amd/test_deepseek_v4_pro_fp4.py` in our fork. |
| 3 | **Stop allocating indexer KV on all 78 GLM-5.2 layers** | V4 allocates the indexer cache *only on CSA layers* — 30 of 61 for Pro, 21 of 43 for Flash — because the `Indexer` module only exists where `compress_ratio == 4`. Our engine allocates on all 78 where only 22 are real. | 7.39 GB per 1M tokens per GPU | `[verified]` — I enumerated the tensor namespace: exactly 30 of 61 layers carry `layers.N.attn.indexer.*` in V4-Pro-0813. DeepSeek do not make this mistake. |

**Tier 1 — the structural wins**

| # | Steal | Change | Attacks | Evidence |
|---|---|---|---|---|
| 4 | **Port V4's two-batch-overlap op path to GLM-5.2** | Our fork's own test docstring says TBO "only DeepSeek-V4 implements". The V4 path is non-EP DP TBO (`moe_a2a_backend='none'`) overlapping one microbatch's `all_gatherv`/`reduce_scatterv` with the other's attention + expert compute, driven by `op_gather`/`op_moe`/`op_combine`. | collectives 19.6%, of which 47% is rank-arrival skew | `[verified, our repo]` `test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py`. The blocker for GLM-5.2 was believed to be `index_topk_freq=4`; the *other* blocker is that no GLM model implements the op path at all. |
| 5 | **DP-attention as the default, not the exception** | DeepSeek ship `--tp 4 --dp 4 --enable-dp-attention` for V4-Flash balanced on B200, and `--tp 8 --dp 8 --enable-dp-attention` for V3.2. Under MLA/CSA the KV has **one head**, so TP cannot shard it anyway — TP only splits the query heads and buys nothing but an extra all-reduce. | attention + collectives | `[verified]` cookbook cell + V3.2-Exp README. And see §2.5: TP8 is strictly worse than DP8 for a 1-KV-head model on memory grounds *as well as* kernel grounds. |
| 6 | **DSpark-style load-adaptive verification length** | Our fork already contains `python/sglang/srt/speculative/dspark_components/` with `DEFAULT_DSPARK_GAMMA = 7`, a block-accept estimator, and Markov head types `("vanilla","gated","rnn")`. It is wired for V4 checkpoints that carry `dspark_*` config keys. The *scheduler* half — expanding verification length at low load and contracting at high load — is model-agnostic. | the 4.7x C1→C16 falloff | `[verified, our repo]` + `[reported]` +60–85% per-user speed at matched throughput in V4 production. |

**Explicitly not worth it**

- `[inferred]` **Do not copy CSA/HCA hybrid attention.** It requires retraining and it is why
  V4-Pro sits at 15% of its roofline. It is a 1M-context capacity technique, and Artificial
  Analysis uses ~10 k inputs.
- `[inferred]` **Do not copy mHC.** `hc_mult=4` residual copies plus a 20-iteration Sinkhorn
  per sub-layer, on a GEMM whose output dimension is 24. DeepSeek themselves name it as the
  source of a determinism problem. It costs serving time and buys training quality.
- `[inferred]` **Do not chase V4's KV compression for the leaderboard.** 12.2x less KV is worth
  a great deal at C256 and nothing at C1 with 10 k inputs, where GLM-5.2's whole KV is 586 MiB.
- `[verified]` **Do not use `--moe-a2a-backend megamoe` at low latency.** The cookbook states
  MegaMoE "is only wired into the `high-throughput` recipe on Blackwell" and is hidden on
  low-latency and balanced. The DeepGEMM Mega-MoE 1.96x-at-BS=1 number in the industry doc is
  a *kernel* microbenchmark; the engine integration is not there for C1.

---

## 1. The architectural trajectory, stated as a sequence

`[verified]` from the configs, which I fetched and parsed for every generation:

| | V3 / V3.1 | V3.2 | **V4-Flash** | **V4-Pro** |
|---|---|---|---|---|
| `num_hidden_layers` | 61 | 61 | 43 | 61 |
| `hidden_size` | 7168 | 7168 | 4096 | 7168 |
| attention | MLA dense | MLA + DSA | **CSA/HCA/SWA hybrid** | **CSA/HCA hybrid** |
| `kv_lora_rank` | 512 | 512 | *(absent)* | *(absent)* |
| `head_dim` | — | — | **512** | **512** |
| `num_key_value_heads` | 128 | 128 | **1** | **1** |
| `qk_nope_head_dim` / `v_head_dim` | 128 / 128 | 128 / 128 | *(absent)* | *(absent)* |
| `qk_rope_head_dim` | 64 | 64 | 64 | 64 |
| `num_attention_heads` | 128 | 128 | 64 | 128 |
| `q_lora_rank` | 1536 | 1536 | 1024 | 1536 |
| `o_lora_rank` / `o_groups` | — | — | **1024 / 8** | **1024 / 16** |
| `index_n_heads`/`index_head_dim`/`index_topk` | — | 64 / 128 / **2048** | 64 / 128 / **512** | 64 / 128 / **1024** |
| `sliding_window` | — | — | **128** | **128** |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 8 | 256 / **8** | 256 / **6** | 384 / **6** |
| `n_shared_experts` | 1 | 1 | 1 | 1 |
| `moe_intermediate_size` | 2048 | 2048 | 2048 | 3072 |
| `first_k_dense_replace` | 3 | 3 | *(absent)* | *(absent)* |
| `num_hash_layers` | — | — | **3** | **3** |
| `n_group` / `topk_group` | 8 / 4 | 8 / 4 | *(absent)* | *(absent)* |
| `scoring_func` | sigmoid | sigmoid | **sqrtsoftplus** | **sqrtsoftplus** |
| `topk_method` | noaux_tc | noaux_tc | noaux_tc | noaux_tc |
| `num_nextn_predict_layers` | 1 | 1 | 1 | 1 |
| `hc_mult` / `hc_sinkhorn_iters` | — | — | 4 / 20 | 4 / 20 |
| `max_position_embeddings` | 163840 | 163840 | **1,048,576** | **1,048,576** |
| `rope_theta` / scaling | 10000 / YaRN×40 | 10000 / YaRN×40 | 10000 / YaRN×16 | 10000 / YaRN×16 |
| `compress_rope_theta` | — | — | 160000 | 160000 |
| `vocab_size` | 129280 | 129280 | 129280 | 129280 |
| `swiglu_limit` | — | — | 10.0 | 10.0 |
| `expert_dtype` | — | — | **fp4** | **fp4** |
| `scale_fmt` | ue8m0 | ue8m0 | ue8m0 | ue8m0 |

Four trends a serving stack has to be ready for, independent of which model it runs:

1. **KV cache stops being linear in sequence length.** V3.2 pays 576 values per token per
   layer. V4 pays 576/4 on CSA layers and 576/128 on HCA layers, plus a fixed 128-token window.
   `[inferred]` **Any engine whose block allocator assumes `tokens → blocks` is a fixed ratio
   is architecturally obsolete.** DeepSeek say this in the report themselves: *"The hybrid
   attention mechanism violates fundamental assumptions behind PagedAttention and its
   variants."* `[verified, §3.5]`
2. **Node-limited routing is dead.** `n_group`/`topk_group` are gone in V4. `[inferred]` That
   is a bet on large scale-up NVLink domains — the regime we are in — and it means a token's 6
   experts can land on any of the 8 ranks with no cap.
3. **Weights go FP4, activations follow.** `expert_dtype: fp4` is a *config key*, not a
   post-hoc quantization. The cookbook exposes a W4A4 MegaMoE variant. `[reported]` "~89.5
   GPQA on Pro" with FP4 activations.
4. **The draft model is part of the checkpoint.** See §4 — this is the single most consequential
   change for a serving engine, and I have hard checkpoint evidence for it.

### 1.1 What V4's attention actually is, read from the reference implementation

`[verified]` `inference/model.py` from `DeepSeek-V4-Pro-0813`, read in full. DeepSeek's own
class docstring still says *"Multi-head Latent Attention (MLA) with sliding window + optional
KV compression"* — so they consider it MLA's descendant, and structurally it is.

The mechanism, per layer, keyed off `config.compress_ratios[layer_id]`:

```python
# Attention.__init__, verbatim structure
self.wkv  = Linear(self.dim, self.head_dim)          # 7168 -> 512.  ONE KV head.
self.kv_norm = RMSNorm(self.head_dim, self.eps)
if self.compress_ratio:
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    self.indexer = Indexer(args, self.compress_ratio) if self.compress_ratio == 4 else None
kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim))
```

- **`ratio == 4` → CSA layer.** Overlapping learned gated pooling down to 1/4 the entries,
  **plus** a lightning indexer that top-k selects *compressed blocks*.
- **`ratio == 128` → HCA layer.** Non-overlapping pooling to 1/128, **no indexer** — dense
  attention over `seq/128` entries.
- **`ratio == 0` → pure sliding-window-128 layer**, with YaRN disabled and base `rope_theta`.
- **Every** layer also runs the 128-token sliding-window branch, concatenated into the same
  core attention.

`[verified]` The per-layer composition, counted from `compress_ratios` and cross-checked against
which layers actually carry `attn.indexer.*` and `attn.compressor.*` tensors in the checkpoint:

| | CSA (r=4) | HCA (r=128) | SWA-only (r=0) | total main layers |
|---|---:|---:|---:|---:|
| V4-Pro / V4-Pro-0813 | **30** | **31** | 0 | 61 |
| V4-Flash / V4-Flash-0731 | **21** | **20** | **2** (layers 0–1) | 43 |

The two counts agree exactly: 30 layers in V4-Pro-0813 carry `layers.N.attn.indexer.*`, and
those are precisely layers 2, 4, 6, …, 60. `[verified]` **This is a two-way cross-check between
the config and the tensor inventory, and it passed.**

`[inferred]` **This is the single most transferable idea in V4 for our DSA indexer.** Our
`index_topk_freq=4` amortizes indexer cost *temporally* — run it every 4th step. DeepSeek
amortize it *spatially* — run it on every CSA layer but against `seq/4` compressed keys, and
put it on only half the layers. The two are orthogonal and compose. And note the direction of
travel on `index_topk`: 2048 (V3.2) → 1024 (Pro) → 512 (Flash), justified in §2.3.4 as
*"thereby improving model efficiency on short- and medium-length texts"* `[verified]` — which
is exactly the Artificial-Analysis regime.

---

## 2. MLA in depth, and the exact KV bytes

This is the section the brief asked for, and it is the reason DeepSeek decode is cheap.

### 2.1 The mechanism

`[verified]` from the V3 report (arXiv 2412.19437) and the ISCA'25 co-design paper
(arXiv 2505.09343), restated with V3.2's actual config values.

Standard MHA caches, per token per layer, one K vector and one V vector **per head**. MLA
instead projects the hidden state down to a single low-rank latent and caches *that*:

```
c^KV_t = W^DKV · h_t                    # [7168] -> [512]     the compressed latent
k^R_t  = RoPE(W^KR · h_t)               # [7168] -> [64]      the decoupled RoPE key

cache per token per layer = c^KV_t (512) ++ k^R_t (64) = 576 values
```

Keys and values are then *reconstructed* on the fly:

```
[k^C_{t,1}; …; k^C_{t,n_h}] = W^UK · c^KV_t          # 512 -> 128 heads x 128 nope dims
[v^C_{t,1}; …; v^C_{t,n_h}] = W^UV · c^KV_t          # 512 -> 128 heads x 128 dims
k_{t,i} = [k^C_{t,i} ; k^R_t]                        # rope key SHARED across all heads
```

**The decoupled RoPE is the load-bearing design choice.** RoPE is position-dependent and does
not commute with the up-projection `W^UK`, so if the rotary part lived inside the latent you
could not absorb `W^UK` into `W^UQ` (see below) — you would have to materialize keys per
position. DeepSeek's answer is to carve out a separate 64-dim rotary key that is
**MQA-shared: one `k^R_t` for all 128 heads.** The nope part stays inside the compressible
latent and never gets rotated. `[verified, V3 report §2.1.1]`

### 2.2 Weight absorption, and its numerical caveats

`[verified]` The trick, stated in the V3 report: because

```
q_{t,i}^T · k^C_{s,i} = (W^UQ_i c^Q_t)^T (W^UK_i c^KV_s) = c^Q_t^T (W^UQ_i^T W^UK_i) c^KV_s
```

you can pre-multiply `W^UQ_i^T W^UK_i` once, offline, and **never materialize `k^C` at all**.
The same absorption works on the output side: `W^UV` folds into `W^O`. The decode kernel then
reads only the 576-wide latent and does a single MQA-shaped attention against it. This is why
FlashMLA's decode kernel has `head_dim_k = 576, head_dim_v = 512` and one KV head `[verified,
FlashMLA API]` — the absorbed form *is* MQA.

**The caveats, which are real and under-discussed:**

- `[inferred]` **Absorption converts a rank-128 product into a rank-512 dense matrix.**
  `W^UQ_i^T W^UK_i` is `[q_head_dim x 512]`. You have traded a small GEMM plus a small KV read
  for a larger GEMM against a much smaller KV. That is a *good* trade at low batch and a
  progressively worse one at high batch, because the absorbed GEMM's FLOPs scale with tokens
  while the KV read does not. This is exactly the "seesaw" that makes MLA decode compute-bound
  at `h_q · s_q >= 128` and memory-bound below it.
- `[inferred]` **Absorption changes the accumulation order and therefore the numerics.**
  `(A^T B) c` and `A^T (B c)` are not bitwise equal in floating point. Any engine that switches
  between an absorbed decode path and a non-absorbed prefill path has two numerically different
  attention implementations for the same weights. DeepSeek treat this as a first-class problem —
  the V4 report §3.3 is entirely about *"bitwise batch-invariant and deterministic kernels"* and
  states they *"carefully design the calculation path of the second kernel to ensure its
  accumulation order is the same as that of the first"* `[verified]`.
- `[verified]` **The rope tail must not be quantized.** FlashMLA's FP8 KV format keeps the
  64 rope dims in BF16 while quantizing the 512 nope dims — *"left unquantized for accuracy"*.
  This is not optional folklore; it is baked into the kernel's memory layout.
- `[verified]` **There is a published, silent-corruption RoPE layout bug.** The V3.2-Exp README
  (2025-11-17) records that the indexer's RoPE input requires a **non-interleaved** layout while
  MLA's RoPE expects an **interleaved** one. Getting it wrong degrades quality without
  crashing. `[inferred]` If our GLM-5.2 DSA port was written by analogy to pre-Nov-2025
  reference code, **check this**; it is a 20-minute check and a silent quality regression.

### 2.3 The exact KV bytes — do the arithmetic

`[verified]` The FP8 layout is not a matter of opinion. It appears identically in three
independent places, which is the strongest form of confirmation available:

1. FlashMLA's Hopper FP8 deep-dive doc: 512 B of `e4m3` + 16 B of FP32 scales (one per 128
   values) + 128 B of BF16 rope = **656 bytes**.
2. Our SGLang fork's GLM-5.2 KV configurator: `kv_lora_rank + kv_lora_rank/128*4 +
   qk_rope_head_dim*2 = 512 + 16 + 128 = 656`.
3. **FlashMLA's SM100 decode kernel `config.h`, which I read directly** — a comment stating,
   verbatim: *"So we set this to 656 for V32 and 576 for MODEL1."* `[verified]`

Now the per-model totals.

```
per token per layer:
  BF16 latent            = (512 + 64) x 2                       = 1,152 B
  FP8 latent (656 fmt)   = 512x1 + (512/128)x4 + 64x2           =   656 B
  DSA indexer key, FP8   = 128x1 + (128/128)x4                  =   132 B
  DSA indexer key, FP4   = 128/2 + 128/32                       =    68 B   (MXFP4, 1x32 SF)
```

| | layers | naive MHA cache/tok/layer | MLA latent | compression | BF16 B/tok | **FP8 B/tok** | indexer B/tok | **total FP8 B/tok** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DeepSeek-V3 / V3.2** | 61 | 128·(128+64) + 128·128 = **40,960** | 576 | **71.1x** | 70,272 | 40,016 | 132×61 = 8,052 | **48,068** |
| **GLM-5.2** | 78 | 64·(192+64) + 64·256 = **32,768** | 576 | **56.9x** | 89,856 | 51,168 | 132×78 = 10,296 | **61,464** |
| GLM-5.2, indexer on the 22 real layers | 78 | — | 576 | — | — | 51,168 | 132×22 = 2,904 | 54,072 |

**Two cross-checks, both of which pass** `[verified]`:

- The ISCA'25 paper's Table 1 gives DeepSeek-V3 MLA at **70.272 KB/token** BF16. My arithmetic
  gives **70,272 bytes**. Exact — note they are using decimal kB, and anyone who divides by
  1024 and reports "68.6 KB" has silently changed units.
- Our corpus's GLM-5.2 figure of **61,464 B/token/GPU** falls straight out of
  `656×78 + 132×78`. Exact.

**And the comparison the brief asked for, stated bluntly** `[inferred]`:

> GLM-5.2 and DeepSeek-V3.2 have **the same 576-wide latent and the same 656-byte FP8 layout**.
> GLM-5.2 nonetheless costs **1.279x more KV per token** — +13,396 B/token — for two reasons
> that have nothing to do with MLA: it has **78 layers instead of 61** (+17 layers × 656 =
> +11,152 B) and it **allocates the indexer cache on all 78 layers when only 22 use it**
> (+7,392 B of pure waste). Fix the second and the gap halves.

`[inferred]` Note also that GLM-5.2's MLA is *less* efficient per layer in the abstract sense:
it compresses 32,768 values to 576 (56.9x) where V3.2 compresses 40,960 to 576 (71.1x). That is
because GLM-5.2 has fewer, fatter heads (64 heads × 256-dim values) against V3.2's 128 × 128.
The *cached* cost is identical — 576 either way — but the reconstruction GEMM is 20% smaller,
which is a small win for GLM-5.2's compute and irrelevant to its memory.

`[verified]` One consequence people get wrong constantly, so it is worth stating: **MLA has one
KV head, so tensor parallelism does not shard the KV cache.** Every rank under TP8 holds a full
copy of all 61 (or 78) layers of latent. The KV numbers above are *per GPU*, not per node.
Under DP-attention they are also per GPU — meaning **DP-attention costs nothing extra in KV
memory for an MLA model.** That is a much stronger argument for DP than the usual one, and it
is specific to 1-KV-head architectures.

### 2.4 What V4 replaced it with, in bytes

`[verified]` Read from the FlashMLA SM100 head64 decode kernel's `config.h`, which contains a
compile-time description of both models' KV layouts. This is primary source of the best kind —
it is the code that reads the bytes:

```cpp
template<ModelType MODEL_TYPE>
struct KernelTemplate {
static constexpr int D_Q    = MODEL_TYPE == ModelType::V32 ? 576 : 512;
static constexpr int D_V    = 512;
static constexpr int D_NOPE = MODEL_TYPE == ModelType::V32 ? 512 : 448;
static constexpr int D_ROPE = 64;
static constexpr int QUANT_TILE_SIZE      = MODEL_TYPE == ModelType::V32 ? 128 : 64;
static constexpr bool V_HAVE_ROPE         = MODEL_TYPE == ModelType::V32 ? false : true;
static constexpr int NUM_SCALES_EACH_TOKEN = MODEL_TYPE == ModelType::V32 ? 4 : 8;
static constexpr int TMA_K_STRIDE = MODEL_TYPE == ModelType::V32
        ? D_NOPE+2*D_ROPE+4*(D_NOPE/QUANT_TILE_SIZE) : D_NOPE+2*D_ROPE;
```

Reading that off: for V4 (`MODEL1`) the KV entry is **512 wide, of which 448 nope in FP8 and 64
rope in BF16**, tile-quantized at **1x64** (not 1x128) giving 8 scale values per token, and
`V_HAVE_ROPE = true` — the entry serves as **both key and value**, so the value carries the
rope dims too. That last flag is the code-level confirmation of report §2.3.3's statement that
*"the KV entry serves as both key and value"* and that RoPE with position `−i` is applied to the
core-attention output. `[verified]`

```
V4 KV entry = 448 B (FP8 nope) + 128 B (BF16 rope) = 576 B in the TMA-mapped tensor
            + 8 scale values stored separately            [inferred: UE8M0, 1 B each]
            = 584 B per compressed entry
V4 indexer entry = 128 dims in MXFP4 = 64 B + 4 B of 1x32 MX scale = 68 B
```

The marginal cost per *token* is then the entries a token adds, weighted by layer type:

```
V4-Pro:    (30/4  + 31/128) x 584 B  +  (30/4) x 68 B  =  4,521.4 + 510.0 =  5,031.4 B/token
V4-Flash:  (21/4  + 20/128) x 584 B  +  (21/4) x 68 B  =  3,157.3 + 357.0 =  3,514.2 B/token
```

| | 10 k ctx | 128 k ctx | 1 M ctx | vs V3.2 (48,068 B/tok) | vs GLM-5.2 (61,464) |
|---|---:|---:|---:|---:|---:|
| **V4-Pro** | 48.0 MiB | 628.9 MiB | **4.91 GiB** | **10.5%** (9.6x cheaper) | 12.2x cheaper |
| **V4-Flash** | 33.5 MiB | 439.3 MiB | **3.43 GiB** | 7.3% (13.7x) | 17.5x cheaper |
| V3.2 (MLA+DSA) | 458 MiB | 5.87 GiB | 46.94 GiB | 1.00x | — |
| GLM-5.2 as shipped | 586 MiB | 7.50 GiB | 60.02 GiB | 1.28x | 1.00x |

**`[inferred]` My independent arithmetic gives 10.5% and the paper's abstract claims "10% of KV
cache compared with DeepSeek-V3.2" at 1M context. That reproduces to within half a point.**
This is the strongest available evidence that the compress_ratios reading is correct, because
the two derivations are completely independent: theirs from the model, mine from a config array
and a kernel header.

`[inferred]` **The honest caveat on all of this:** at 10 k input — the Artificial-Analysis shape
— GLM-5.2's entire KV is 586 MiB and V4-Pro's is 48 MiB. Both are noise against 178 GiB. **KV
compression is worth zero on the leaderboard.** It is worth a great deal at C256 and at 1M
context, which is a different business.

### 2.5 The compute-bound crossover, redone for our box

`[verified]` FlashMLA's deep-dive derives: MLA decode FLOPs ≈ `2·h_q·s_q·s_k·(d_k+d_v)`, bytes
≈ `2·s_k·d_k`, so arithmetic intensity ≈ `2·h_q·s_q`. Compute-bound iff
`h_q·s_q ≥ ½ · (peak FLOPS / HBM BW)`.

Our numbers, with the driver-probed 7.672 TB/s rather than the rounded 8:

```
B200 dense BF16 peak       ~ 2.25 PFLOPS      [reported]
B200 HBM (driver-derived)    7.672 TB/s       [verified, 00-this-machine-ground-truth.md]
crossover  h_q·s_q  >=  0.5 x 2250/7.672  =  147
```

| configuration | `h_q` per rank | `s_q` needed | reachable? |
|---|---:|---:|---|
| GLM-5.2, TP8, 64 heads | 8 | **18** | No. EAGLE 3-1-4 gives `s_q ≈ 4`. |
| GLM-5.2, **DP8**, 64 heads | 64 | **2.3** | **Yes, trivially** — EAGLE already gives 4. |
| V3.2, TP8, 128 heads | 16 | 9 | No. |
| V3.2, **DP8**, 128 heads | 128 | 1.15 | Yes, even without speculation. |

`[verified]` DeepSeek state the consequence outright: *"we don't use Tensor Parallel for
decoding instances, meaning h_q is 128 and the kernel is compute-bound."* `[inferred]`
**Our TP8 configuration puts GLM-5.2's MLA decode on the memory-bound side of a line DeepSeek
deliberately architected to be on the other side of, and it costs us KV memory nothing to
cross it, because MLA's single KV head is replicated under TP anyway.** This is the same
conclusion the industry doc reaches; I am restating it because the memory argument — that DP
costs *nothing* in KV for a 1-KV-head model — is the part that usually kills the proposal, and
it is wrong.

---

## 3. Fine-grained MoE, shared experts, and load skew at inference

### 3.1 The three design decisions

`[verified]` from the V3 report and confirmed in every config since:

1. **Fine granularity.** 256 (or 384) small experts at `moe_intermediate 2048`/`3072`, top-8
   (V3) or top-6 (V4), rather than 8–16 fat experts. More experts × smaller each = the same
   activated FLOPs with far more routing combinations.
2. **One always-on shared expert.** `n_shared_experts: 1` in every generation. It absorbs the
   knowledge every token needs, so the routed experts specialize.
3. **Auxiliary-loss-free load balancing.** A per-expert bias `b_i` added to affinity scores
   **for top-K selection only**. The gate weight multiplied into the FFN output still comes
   from the unbiased score. This is visible verbatim in V4's reference `Gate.forward`, with
   DeepSeek's own comment `[verified]`:

```python
original_scores = scores
# Bias shifts scores for expert selection (topk) but does not affect routing weights.
if self.bias is not None:
    scores = scores + self.bias
indices = self.tid2eid[input_ids] if self.hash else scores.topk(self.topk, dim=-1)[1]
weights = original_scores.gather(1, indices)
```

`[verified]` V4 changed the affinity function from `sigmoid` (V3/V3.2) to
`scoring_func: "sqrtsoftplus"` — implemented as `F.softplus(scores).sqrt()`. `topk_method`
stays `noaux_tc`. `[inferred]` Softplus is unbounded above where sigmoid saturates at 1, so
the routed-expert weights have more dynamic range before the `norm_topk_prob` renormalization;
`routed_scaling_factor` moved 2.5 → 1.5 for Flash, consistent with that.

### 3.2 What it means for expert load skew at inference

`[inferred]` This is where the theory and our operating point diverge sharply, and the
distinction is worth being precise about because it is routinely muddled.

**Aux-loss-free balancing flattens the *expected* load, not the *instantaneous* load.** The
bias `b_i` is updated once per training step against a whole training batch. It makes the
marginal probability of each expert roughly uniform *in aggregate*. It says nothing about what
happens when you route 6 tokens.

Consequently there are two completely different skew problems:

| | statistical skew | structural skew |
|---|---|---|
| When | large batch (C64, C256, prefill) | small batch (C1 decode) |
| Cause | some experts are genuinely more popular on this workload | 6 experts drawn from 384 cannot cover 8 ranks |
| Magnitude | DeepSeek get away with **32 redundant experts** across EP144 | at C1, ≥2 of 8 ranks receive **zero** tokens, always |
| Fix | EPLB / LPLB / `--enable-waterfill` | **none exists** |

`[inferred]` **At C1 the skew is arithmetic, not statistical.** With top-6 over EP8, at most 6
ranks have work; at least 2 idle. If two chosen experts share a rank, three idle. No load
balancer can help, because there is nothing to balance — the imbalance is a property of
`6 < 8`. This is why the industry doc is right to exclude EPLB from the C1 work list, and it is
worth noting that **V4 made this worse**, going from top-8 (≥8 ranks coverable) to top-6.

The roofline consequence is concrete. From §7's accounting, one V4-Pro expert is 35.09 MB:

| busiest rank holds | bytes read that rank | ceiling |
|---|---:|---:|
| 1 of 6 active experts/layer (ideal) | 5.50 GB | 1,396 tok/s |
| 2 of 6 (typical) | 7.64 GB | 1,005 tok/s |
| 6 of 6 (worst case) | 16.20 GB | 474 tok/s |

`[inferred]` **A 2.9x spread in the decode ceiling purely from expert placement luck, at C1.**
This, not bandwidth, is a substantial part of what our profile records as 47%-of-collectives
rank-arrival skew: ranks arrive at the combine barrier at different times because they did
different amounts of work.

`[inferred]` The only structural fixes at C1 are (a) **replicate hot experts so every rank has
work** — which is what redundant experts do, used for a purpose DeepSeek did not intend; or
(b) **fuse dispatch/GEMM/combine so there is no barrier to arrive at**, which is what
Mega-MoE does. Note that our fork exposes `--moe-a2a-backend megamoe --enable-waterfill`
together, and that the cookbook says Waterfill *"applies Waterfill to the fused shared expert
slot"* under MegaMoE `[verified]`.

### 3.3 Hash routing — a free win nobody talks about

`[verified]` `num_hash_layers: 3` in both V4 models, and the checkpoint carries
`layers.{0,1,2}.ffn.gate.tid2eid` — an `int32[vocab_size, num_experts_per_tok]` lookup table.
For the first three MoE layers, expert indices come from **the input token IDs**, not from a
router GEMM:

```python
self.hash = layer_id < args.n_hash_layers
indices = self.tid2eid[input_ids]     # frozen, requires_grad=False
```

`[inferred]` **For a serving engine this is a gift.** Routing for three layers is known before
the forward pass begins. Dispatch metadata can be computed on the host, or prefetched, entirely
off the critical path — removing three router GEMMs, three top-k selections, and three
routing-dependent stalls per token. At 4.25 ms TPOT over 61 layers, three layers is ~5% of the
step. `[inferred]` Note the memory cost: `129280 × 6 × 4 B = 3.10 MB` per layer, replicated on
every rank — trivial. Compare to GLM-5.2, which uses three *dense* leading layers for the same
purpose (avoiding router instability early in the stack) at a cost of 226 M params per layer.
**Hash routing is strictly cheaper than dense-layer replacement and does the same job.**

---

## 4. MTP, and the checkpoint evidence that DSpark replaced it

### 4.1 MTP as published

`[verified, V3 report]` V3 predicts the next 2 tokens through MTP. *"the acceptance rate of the
second token prediction ranges between **85% and 90%** across various generation topics… This
high acceptance rate enables DeepSeek-V3 to achieve a significantly improved decoding speed,
delivering **1.8 times TPS**."* One MTP module (`num_nextn_predict_layers: 1`), used at
inference as a speculative drafter.

`[verified]` The throughput framing matters as much as the latency one: *"by predicting multiple
tokens per step, MTP increases the inference batch size, which is crucial for boosting EP
computational intensity and hardware utilization."* — MTP raises `s_q`, which is the same term
in the compute-bound condition of §2.5. Speculation and arithmetic intensity are the same lever.

`[verified]` `num_nextn_predict_layers: 1` persists through V3.2 and into V4's config.

### 4.2 What actually changed in the production checkpoints

**This is a finding, not a restatement.** I enumerated the tensor namespace of four checkpoints.
`[verified]`

| checkpoint | main layers | **`mtp.N` blocks** | indexer layers | `compress_ratios` length | trailing zeros |
|---|---:|---:|---:|---:|---:|
| `DeepSeek-V4-Pro` (Apr) | 61 | **1** (`mtp.0`) | 30 | 62 | 1 |
| `DeepSeek-V4-Pro-0813` | 61 | **3** (`mtp.0,1,2`) | 30 | **64** | **3** |
| `DeepSeek-V4-Flash` (Apr) | 43 | **1** | 21 | 44 | 1 |
| `DeepSeek-V4-Flash-0731` | 43 | **3** | 21 | **46** | **3** |

The `compress_ratios` array is `main_layers + drafter_layers` long, and the trailing entries are
`0` — pure sliding-window-128 attention. In the production checkpoints there are **three** of
them, and none of the `mtp.N` blocks carries a `compressor` or `indexer` tensor `[verified]`.

Set that against the DSpark paper's description of the V4-co-deployed drafter: *"three MoE
layers with mHC and a sliding window attention of 128"* `[reported]`.

> `[inferred, but the evidence is about as direct as checkpoint archaeology gets]`
> **The 0731/0813 production checkpoints replaced the single MTP-1 block with a three-block
> DSpark drafter, and you can see it in the tensor names.** Each `mtp.N` block is a full-width
> MoE layer — 384 experts, a shared expert, a gate, mHC parameters — with SWA-128 attention and
> no compression. `mtp.0` additionally carries `main_proj`/`main_norm`, which projects the
> target model's hidden state into the drafter; `mtp.1` and `mtp.2` do not. That is a
> semi-autoregressive chain fed once from the target, exactly as the paper describes.

**The serving cost of that drafter, measured from the shard headers** `[verified]`:

| | params | bytes | share of checkpoint |
|---|---:|---:|---:|
| V4-Pro-0813 main model + embeddings | 1,573.0 B | 850.75 GB | 95.3% |
| **V4-Pro-0813 DSpark drafter (`mtp.0–2`)** | **77.5 B** | **41.98 GB** | **4.7%** |
| total | 1,650.5 B | 892.73 GB | 100% |

`[inferred]` **42 GB of HBM — 5.25 GB per GPU at TP8 — is the drafter.** That is a real, and
rarely stated, tax. It also explains the HuggingFace "1.7T" badge on the 0813 repo against the
paper's "1.6T": the badge counts the drafter and the paper does not.

### 4.3 What our engine already supports

`[verified, our fork]` `python/sglang/srt/speculative/dspark_components/` exists and contains:

- `dspark_config.py` — `DEFAULT_DSPARK_GAMMA = 7`,
  `SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla", "gated", "rnn")`, and
  `checkpoint_bundles_dspark_draft()` which detects the drafter by looking for the
  `dspark_block_size` / `dspark_markov_rank` / `dspark_noise_token_id` /
  `dspark_target_layer_ids` keys on the target config — **exactly the keys I found in the
  0731/0813 configs and not in the April ones.** Independent confirmation of §4.2.
- `dspark_block_accept_estimator.py` (825+ lines) with online logging via
  `SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH` and `SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL`.
- `dspark_kv_inject.py`, `dspark_disaggregation.py`, and a `dspark_verify_window` kernel.
- `base_spec_worker.py` comment: *"dflash / dspark drive the draft model through a plain
  TpModelWorker"*.

`[verified]` The `dspark_target_layer_ids` values are `[58, 59, 60]` for Pro and `[40, 41, 42]`
for Flash — i.e. **the last three main layers**, whose hidden states feed the drafter.

`[inferred]` **The half of DSpark that helps us needs no DSpark checkpoint.** The
block-accept estimator plus a load-adaptive verification-length scheduler is model-agnostic:
profile steps-per-second against batch size once at init, then admit draft tokens by cumulative
confidence against a cost table. EAGLE's existing draft probabilities are a usable confidence
proxy for v1. Our fork has the estimator already; what it does not have is that scheduler wired
to GLM-5.2's EAGLE path. **That is the single highest-value speculative-decoding change
available to us**, because it attacks the 4.7x C1→C16 falloff directly, and because it sidesteps
the eagle_worker_v2 IMA entirely — it does not deepen the draft tree, it varies how much of a
fixed tree gets verified.

`[reported]` The published payoff, for calibration: **+60–85% per-user tok/s at matched
throughput** (Flash) and +57–78% (Pro) versus static MTP-1. And DeepSeek's stated reason MTP-1
was the baseline at all: *"deploying a static multi-token drafter (e.g. MTP-3/5) strictly
degrades aggregate throughput under high concurrency due to excessive verification overhead."*
`[inferred]` That is a precise description of our C16 problem.

---

## 5. DSA in V3.2-Exp, and index sharing across layers

### 5.1 The lightning indexer

`[verified]` from `DeepSeek_V3_2.pdf` / arXiv 2512.02556 §2:

```
I_{t,s} = Σ_{j=1..H_I}  w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )
u_t     = Attn( h_t, { c_s | I_{t,s} ∈ Top-k(I_{t,:}) } )
```

`H_I = 64` heads of 128 dims, top-k = 2048. ReLU *"for throughput consideration"*, and the
whole thing runs in FP8. The critical kernel-level justification, quoted `[verified]`:

> *"At the kernel level, each key-value entry must be shared across multiple queries for
> computational efficiency. Therefore, we implement DSA based on the MQA mode of MLA, where
> each latent vector will be shared across all query heads of the query token."*

`[inferred]` That sentence is why DSA and MLA are inseparable. Sparse attention over a
per-head KV cache would gather 64 different key sets per token; over an MQA latent it gathers
one. GLM-5.2 inherits this property for free, since its latent is also MQA-shared.

`[verified]` Cost claim: complexity drops O(L²) → O(Lk), with the honest note that *"the
lightning indexer still has a complexity of O(L²)"* — it is cheaper per unit, not
asymptotically better. And the published crossover from FlashMLA's own benchmarking: at
bs=128/heads=128/`s_q`=2/topk=2048 the sparse kernel's runtime equals the dense kernel's at
**sequence length ≈ 3000**; below that, dense wins `[reported]`.

`[inferred]` **At the Artificial-Analysis input length of ~10 k this crossover is uncomfortably
close.** DeepSeek themselves ship two prefill paths and switch on length: *"for short-sequence
prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher
efficiency under short-context conditions"* `[verified]`. **We should measure a dense-masked
prefill path at 10 k against our DSA path.** Our TTFT is 189 ms and TTFT is scored by AA.

### 5.2 Index sharing across layers — what is published, and what it means for us

This is the part of the brief with the highest leverage, because GLM-5.2's `index_topk_freq=4`
is precisely an index-sharing scheme.

**What is published, precisely** `[verified]`:

- V3.2 runs the indexer on **every layer**, against **raw** keys. No sharing.
- V4 runs the indexer on **half the layers** (CSA only: 30 of 61, 21 of 43), against
  **compressed** keys (`seq/4` of them). The indexer's *queries* come from the **same latent
  `c^Q_t`** as the attention queries — `W^DQ` is shared, then `W^IUQ` vs `W^UQ` diverge
  `[verified, report §2.3.1 and confirmed in `Indexer.__init__`: `self.wq_b =
  ColumnParallelLinear(self.q_lora_rank, n_heads*head_dim)` reads `qr`, the q-lora activation].
- The indexer has its **own** `Compressor` constructed with `rotate=True`, meaning a Hadamard
  transform (`fast_hadamard_transform` is in `requirements.txt`) is applied before
  `fp4_act_quant` `[verified, `Indexer.forward`]`.
- DeepSeek's own inline comment: `# use fp4 simulation for q and kv in indexer` and
  `# We performed QAT here, kv could also use fp8 format, though current implementation uses bf16`
  `[verified]`.

**What is *not* published, stated as a gap** `[unverified]`: I found no DeepSeek publication
that shares a single computed top-k index set across multiple layers the way
`index_topk_freq=4` does. Their sharing is spatial (fewer keys, fewer layers with indexers),
not temporal (reuse the same indices across steps). **So GLM-5.2's temporal amortization is,
as far as I can source, a Zhipu invention with no DeepSeek analogue.** Do not go looking for a
paper that validates it.

`[inferred]` **But the two compose, and that is the actionable point.** Our indexer costs 5.8%
of C1. Three independent multipliers are available:

| lever | source | effect on indexer cost | our status |
|---|---|---|---|
| temporal: run every 4th step | GLM-5.2 `index_topk_freq=4` | ÷4 | **already have it** |
| spatial: indexer on half the layers | V4 CSA/HCA split | ÷2 | needs retraining — not available |
| **precision: FP4 index QK** | V4 §5.2.1 + DeepGEMM SM100 kernel | **÷2 on cache bandwidth** | **available now**, flag exists in our fork for V4 |
| **precision: FP32→BF16 index scores before top-k** | V4 §5.2.1 | **2x top-k selector at 99.7% recall** | one dtype change |

The last two are the ones we can take today, and the second is a one-line change with a
published measurement behind it: `[reported]` *"we further quantize the index scores `I_{:,:}`
from FP32 to BF16 during this QAT process. This optimization achieves a **2x speedup for the
top-k selector**, while preserving a **99.7% recall rate** of KV entries."*

`[verified]` And the FP4 indexer is not theoretical — I read the kernel. DeepGEMM's
`sm100_mqa_logits.cuh` (593 lines) is a *unified* implementation:

```cpp
constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;
static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or kHeadDim == 64 or kHeadDim == 128, "Invalid head dim");
// "MXFP4 / MXFP8 folds its scale into the MX SF UMMA, no extra scale"
```

with **both** a contiguous-KV entry (*"Unified contiguous-KV entry for FP8 / MXFP4 / MXFP8"*)
and a **paged** entry (`sm100_paged_mqa_logits`, taking `block_table`, `PAGE_KV`,
`kSplitsPerChunk`). `[inferred]` **`kHeadDim == 128` is supported and that is GLM-5.2's
`index_head_dim`. This kernel is MIT-licensed, SM100-native, paged, and FP4-capable, and it
is the exact shape of our indexer.** The industry doc records this kernel as SM90-only from
PR #200; **that is out of date — the current `main` has a full SM100 implementation.**

### 5.3 The RoPE interleaving trap, restated because it is cheap to check

`[verified]` V3.2-Exp README, 2025-11-17: *"the input tensor to RoPE in the indexer module
requires a **non-interleaved** layout, whereas RoPE in the MLA module expects an
**interleaved** layout."* `[inferred]` Silent quality regression, no crash. Check our GLM-5.2
indexer. Note V4 changed the geometry again: indexer RoPE uses `compress_rope_theta = 160000`
where the model's base is `rope_theta = 10000`, and pure-SWA layers disable YaRN entirely
`[verified, `Attention.__init__`]` — three different rotary configurations in one model.

---

## 6. DeepSeek's inference system, and what survives on one node

### 6.1 What they published

`[verified]` Recapping only the load-bearing facts (full detail is in
`04-industry/04-deepseek-open-infrastructure.md`, which I am not repeating):

| | Prefill | Decode |
|---|---|---|
| Routed experts | **EP32** | **EP144** |
| MLA / shared expert | **DP32** | **DP144** |
| Deployment unit | 4 nodes (32 GPUs) | 18 nodes (144 GPUs) |
| Redundant experts | 32 | 32 |
| Per-GPU experts | 9 routed + 1 shared | 2 routed + 1 shared |
| Overlap | dual-microbatch | **5-stage pipeline**, attention split in two |

Measured production, 24 h: 226.75 average nodes, $87,072/day at $2/H800-h, 608 B input tokens
(56.3% on-disk KV hits), 168 B output, **average output speed 20–22 tok/s/user**, theoretical
margin 545%. Precision split: *"matrix multiplications and dispatch transmissions adopt the
FP8 format… while core MLA computations and combine transmissions use the BF16 format."*

### 6.2 What changes on one 8xB200 node — item by item

`[inferred]` throughout. This is the adaptation the brief asked for, and the honest answer is
that **most of it does not survive**.

| Their design | Survives on 8xB200? | Why |
|---|---|---|
| **PD disaggregation** | **No, for the leaderboard. Maybe, for cost.** | AA measures single-prompt and 10-parallel P50 with TTFT counted to first token. Splitting prefill and decode across a 8-GPU box means each phase gets 4 GPUs and a KV transfer hop in the middle. At C1 that is strictly worse. At C256 it is arguable and our fork supports it (mooncake). |
| **EP144 / EP32** | **No.** | We have EP8, full stop. Their entire sparsity argument — *"the model's high sparsity necessitates an extremely large overall batch size"* — assumes each expert sees enough tokens. At EP8 with 384 experts, each rank owns 48 experts and at C1 activates ≤6 across the whole node. |
| **DP144 / DP32 attention** | **Yes, as DP8. This is the piece that transfers.** | The reason is architectural, not scale-dependent: 1 KV head means TP shards nothing, and `h_q` per rank determines whether the decode kernel is compute-bound (§2.5). Both hold at DP8. |
| **Node-limited routing** | **No, and V4 deleted it too.** | It exists to deduplicate InfiniBand traffic across a 4:1 scale-up:scale-out bandwidth ratio. We have one NVLink domain at 956.25 GB/s per GPU through NVSwitch and no inter-node hop. V4's configs have no `n_group`/`topk_group`. |
| **Redundant experts (32) + EPLB** | **Partially, and for a different reason.** | Their purpose is statistical balancing at EP144. At EP8/C1 the skew is structural (§3.2). But redundancy repurposed as *coverage* — ensuring every rank has work at top-6 — is a genuinely different and untested idea. Our fork has `--ep-num-redundant-experts`, `--eplb-algorithm`, `--init-expert-location`, `--enable-waterfill`. |
| **Dual-microbatch overlap (prefill)** | **Yes — and it is implemented for V4 in our fork.** | See §9.2. This is the highest-value transfer in the table. |
| **5-stage decode pipeline** | **Unclear, probably not.** | It exists because their all-to-all is RDMA and *"does not occupy GPU SMs: after RDMA messages are issued, all GPU SMs are freed."* On NVLink our combine *does* occupy SMs, so the overlap structure is different. Their prefill-side trick — *"the same prompt may be split between them"* to balance attention load across microbatches — does transfer. |
| **On-disk KV cache (56.3% hit)** | **No, for AA. Enormous, for cost.** | AA prompts are fresh. Our measured 54% prefix-cache hit on a coding workload at C64 says the opposite is true for real traffic. |
| **Diurnal capacity reallocation** | **N/A at 8 GPUs**, but the productized form is visible in their price sheet: off-peak = half price, peak hours 01:00–04:00 and 06:00–10:00 UTC `[verified]`. |
| **The 545% margin arithmetic** | **Reproduces exactly**, and is worth redoing for us. | 226.75 × 8 × 24 × $2 = $87,072 ✓. Revenue 342B×$0.14/M + 266B×$0.55/M + 168B×$2.19/M = $562,100 ≈ their $562,027 ✓. |

### 6.3 The comm-bound ceiling, redone for us

`[verified]` The ISCA paper's calculation, which is worth reproducing because the answer flips:

```
H800 + CX7 400Gb (~50 GB/s):
  per layer, 2 all-to-alls  = (1 B dispatch FP8 + 2 B combine BF16) x 32 tok x 9 experts x 7168 hidden / 50 GB/s
                            = 120.96 us
  dual-microbatch overlap   -> 2 x 120.96 = 241.92 us/layer x 61 layers = 14.76 ms TPOT ≈ 67 tok/s
```

`[inferred]` Our version, using DeepEP V2's own first-party SM100 EP8 NVLink measurement of
**726 GB/s dispatch / 740 GB/s combine** `[reported]` and GLM-5.2's hidden 6144, top-8+shared:

```
per layer = (1 + 2) B x 32 tok x 9 x 6144 / 726e9  =  7.31 us   ...at 32 tokens
at batch 1 (our C1):  (1+2) x 1 x 9 x 6144 / 726e9  =  0.23 us per layer
                                        x 78 layers =  17.8 us per token
```

`[inferred]` **17.8 µs against a 2.74 ms TPOT budget is 0.65%. Communication bandwidth is not
remotely our ceiling.** Our collectives are 19.6% of C1 and 47% of that is rank-arrival skew.
**The problem is synchronization and launch cost, not bytes on the wire.** That reframing is
the most important thing the DeepSeek corpus does for us, and it is why §3.2's expert-placement
spread and §7's fused-kernel work matter more than any bandwidth optimization.

For calibration, the counterfactual DeepSeek themselves computed `[verified]`: on GB200 NVL72 at
900 GB/s, `3 × 32 × 9 × 7168 / 900e9 = 6.88 µs`, giving *"a theoretical upper limit of over
0.82 ms TPOT, approximately 1200 tokens per second. While this figure is purely theoretical and
has not been empirically validated…"*

---

## 7. FlashMLA, DeepEP, DeepGEMM — SM100 status, read from the source

The brief said to verify by reading the repos rather than the READMEs. I did, and the READMEs
are wrong in at least one materially important way.

### 7.1 FlashMLA — what is actually in `csrc/`

`[verified]` `github.com/deepseek-ai/FlashMLA`, `main`, last push 2026-07-28, 130 files,
branches `main` and `nv_dev`. Full recursive tree read. What exists:

| path | size | what it is |
|---|---:|---|
| `csrc/sm90/decode/dense/splitkv_mla.cuh` | 58,333 B | Hopper dense MLA decode, BF16/FP16 |
| `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` | 39,360 B | Hopper sparse FP8 decode + `components/dequant.h` (the "Crossover" path) |
| `csrc/sm90/decode/sparse_fp8/instantiations/` | — | `v32_persistent_h64.cu`, `v32_persistent_h128.cu`, `model1_persistent_h64.cu`, `model1_persistent_h128.cu` |
| **`csrc/sm100/decode/head64/kernel.cuh`** | **51,887 B** | **the only native SM100 decode kernel**; instantiations `v32.cu` and `model1.cu` |
| **`csrc/sm100/decode/head128/README.md`** | **185 B** | **a README and nothing else** |
| `csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh` | 28,868 B | SM100 sparse prefill, k512 + k576 |
| `csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh` | 30,554 B | SM100 sparse prefill, k512 + k576 |
| `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.cuh` | 56,963 B | **new**: small-topk family, instantiations `phase1_prefill_k512.cu` **and `phase1_decode_k512.cu`** |
| `csrc/sm100/prefill/dense/…` | ~20 files | CUTLASS FMHA fwd + bwd (NVIDIA-contributed, PR #76) |

**The finding.** `csrc/sm100/decode/head128/README.md` contains, in full `[verified]`:

> *"Head128 decoding kernels are located at
> `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu`
> (for k_dim = 512) or simulated using 2x head64 kernel"*

`[inferred]` So on Blackwell there is **no purpose-built 128-query-head sparse MLA decode
kernel**. You either borrow a kernel from the *prefill* small-topk family — which only
instantiates `k_dim = 512`, i.e. **V4's geometry, not V3.2's 576** — or you run the head64
kernel twice. That is a real performance cliff, and it is consistent with the README's own
admission that B200 sparse decode is *"not really optimized yet"* (350 TFLOPS on B200 vs 410
on H800) `[reported]`.

**What this means for GLM-5.2 specifically** `[inferred]`:

- GLM-5.2 has **64 query heads** and a **576-wide** latent. Under **DP-attention**, `h_q = 64`
  per rank and the **native SM100 `head64` decode kernel is the exact match**, with `k_dim=576`
  available via `ModelType::V32`.
- Under **TP8**, `h_q = 8` per rank — neither kernel's shape. You are padding to 64 and wasting
  87.5% of the tile.
- **This is a second, independent argument for DP-attention**, alongside §2.5's compute-bound
  crossover and the observation that DP costs no KV memory for a 1-KV-head model. Three
  arguments, three mechanisms, one conclusion.

### 7.2 DeepGEMM — SM100 is the primary target now

`[verified]` `main`, last push 2026-08-11, 136 files. SM100-specific implementations present:

```
sm100_bf16_gemm            sm100_fp8_fp4_gemm_1d1d      sm100_mqa_logits
sm100_bf16_mega_moe        sm100_fp8_fp4_mega_moe       sm100_tf32_hc_prenorm_gemm
sm100_bmk_bnk_mn           epilogue/sm100_store_cd{,_swap_ab}
scheduler/sm100_mqa_logits scheduler/sm100_paged_mqa_logits    mma/sm100.cuh
```

`[verified]` Three observations from the file list that are not in the README:

1. **`sm100_tf32_hc_prenorm_gemm`** exists on both SM90 and SM100 — that is the mHC
   hyper-connection pre-norm GEMM. `[inferred]` DeepSeek shipped a production kernel for a
   24-wide-output GEMM, which tells you how much that architecture choice cost them.
2. **SM90 has `fp8_gemm_1d1d` *and* `fp8_gemm_1d2d`; SM100 has only `fp8_fp4_gemm_1d1d`.**
   `[inferred]` The 1D-2D scaling variant was not ported. If our stack assumes 1x128 activation
   scaling against 128x128 weight scaling on Blackwell, check which path it lands on.
3. **The indexer kernel is unified and paged on SM100** (§5.2). The `sm90_fp8_paged_mqa_logits`
   / `sm100_paged_mqa_logits` split shows SM100 got a single kernel covering FP8/MXFP4/MXFP8
   where SM90 needed separate ones.

`[verified]` The porting trap the README does state, and which is real: **SM90 requires FP32
scale factors, SM100 requires packed UE8M0** (*"which packs 4 UE8M0 into a single
`torch.int`"*). Both GLM-5.2's FP8 build and every DeepSeek config since V3.2 declare
`"scale_fmt": "ue8m0"`. The ecosystem has standardized. Helpers:
`get_mn_major_tma_aligned_packed_ue8m0_tensor`, `transform_sf_into_required_layout`.

`[verified]` Mega-MoE is explicitly a Blackwell win, from PR #316's own notes: *"The code is
specifically optimized for Blackwell (SM100) with TMEM features; SM90 adaptation shows
diminished gains without TMEM."*

### 7.3 DeepEP — the branch that matters is not `main`

`[verified]` 15 branches: `main`, `epv2-release`, **`hybrid-ep`**, `nvDev`, `antgroup-opt`,
`tencent-zcopy`, `mori-ep`, `eager-rdma`, `elect-one-sync`, and several compat branches.

`[verified]` The `hybrid-ep` branch tree (78 files) contains
`csrc/hybrid_ep/{allocator,backend,buffer,executor,extension,jit}/`,
`deep_ep/hybrid_ep_buffer.py`, `tests/test_low_latency_nvfp4.py`, and
`tests/test_graphed_hybrid_ep.py`. **The presence of `test_low_latency_nvfp4.py` and
`test_graphed_hybrid_ep.py` is the useful signal** — NVFP4 low-latency dispatch and CUDA-graph
capture are both tested there, and both are requirements for our C1 path.

`[reported]` The B200 numbers from that branch's `docs/README_Hybrid-EP.md` (4096 tokens,
hidden 7168, topk 8, 8 ranks):

| impl | measurement | SMs | dispatch FP8 | dispatch BF16 | combine |
|---|---|---:|---:|---:|---:|
| DeepEP | Torch API | 16 | 246 | 348 | 302 |
| DeepEP | Torch API | 48 | 554 | 646 | 586 |
| **Hybrid-EP** | Torch API | **16** | **409.71** | 535.94 | 530.86 |
| **Hybrid-EP** | kernel only | **16** | **599.27** | 734.95 | 673.84 |

`[inferred]` Two readings. First, **naive low-SM DeepEP on B200 is bad** (246 GB/s at 16 SMs)
and the naive fix costs 48 of 148 SMs. Second, and more useful: the **32% gap between "Torch
API" (409) and "kernel only" (599) at 16 SMs is host/framework overhead, not network** — which
independently corroborates the V4 report's Host Codegen claim that CPU-side validation drops
*"from tens or hundreds of microseconds to less than one microsecond per invocation"*
`[reported]`.

`[verified]` Worth flagging against our own cookbook config: SGLang's V4-Flash *balanced* recipe
on B200 passes `--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'`
— **96 of 148 SMs, 65% of the GPU, on comms.** `[inferred]` That is a throughput-recipe choice
and would be catastrophic at C1. If we ever enable DeepEP, do not inherit that number.

---

## 8. Memory arithmetic and roofline on 8xB200

Hardware constants, all `[verified]` from `00-hardware/00-this-machine-ground-truth.md`:
178.34 GiB usable per GPU, **HBM 7.672 TB/s** (7680-bit bus × 3996 MHz × 2), 148 SMs,
126.50 MiB L2. **I use 7.67 TB/s throughout; note that `05-models/01-glm-5.2` used a rounded,
unverified 8.0 TB/s, so its ceilings are ~4% optimistic and I restate them below.**

### 8.1 Checkpoint sizes, measured not estimated

`[verified]` Summed from the HuggingFace API's per-file blob sizes — not from the
`total_size` field, which for `DeepSeek-V3.2-Exp` reports 1,370,793,842,752 B (exactly 2x the
truth, an FP8-dtype accounting bug worth knowing about).

| checkpoint | shards | bytes | GB | GiB | **GiB/GPU at TP8** | free of 178.34 GiB |
|---|---:|---:|---:|---:|---:|---:|
| `DeepSeek-V4-Pro-0813` | 66 | 892,744,322,880 | 892.74 | 831.43 | **103.93** | 74.41 |
| `DeepSeek-V4-Pro` (Apr) | 64 | 864,721,029,744 | 864.72 | 805.33 | 100.67 | 77.67 |
| `DeepSeek-V4-Flash-0731` | 48 | 166,886,535,336 | 166.89 | 155.43 | **19.43** | 158.91 |
| `DeepSeek-V4-Flash` (Apr) | 46 | 159,617,149,040 | 159.62 | 148.66 | 18.58 | 159.76 |
| `DeepSeek-V3.2-Exp` (FP8) | 163 | 689,483,049,129 | 689.48 | 642.13 | **80.27** | 98.07 |
| GLM-5.2 FP8 (ours) | — | — | 755.62 | — | **89.37** | 88.97 |
| GLM-5.2 NVFP4 (ours) | — | — | 464.80 | — | **56.72** | 121.62 |

`[inferred]` **Three things fall out immediately.**

1. **V4-Pro-0813 fits on 8xB200 with 74 GiB/GPU to spare** despite being 1.65 T parameters —
   because its experts are genuinely FP4. GLM-5.2's FP8 build, at less than half the parameter
   count, uses *more* per-GPU memory than V4-Flash by a factor of 4.6.
2. **V4-Flash-0731 is 155.43 GiB total.** It fits on **one** B200 (178.34 GiB) with ~23 GiB
   left. SGLang's own cookbook nonetheless recommends `--tp 4` for it on B200 — `[inferred]`
   presumably for KV headroom and because TP4 halves the per-rank weight read. But an 8-GPU box
   could host **two TP4 replicas or eight TP1 replicas**, and at C1 that is a very different
   throughput/cost point than one TP8 instance.
3. **The delta between the April and August checkpoints is the drafter**: 892.74 − 864.72 =
   **28.02 GB** for Pro, 166.89 − 159.62 = **7.27 GB** for Flash. (Not equal to the 41.98 GB
   drafter total, because the April checkpoint's single `mtp.0` block was itself ~14 GB.)

### 8.2 Bytes per parameter, read from the tensors

`[verified]` I read the safetensors header of every one of the 66 shards of
`DeepSeek-V4-Pro-0813` — 149,782 tensors — and summed. My total is 892,727,580,904 B against
the API's 892,744,322,880 B, a ratio of **0.99998** (the difference is the 66 header blocks
themselves). The accounting is therefore trustworthy.

Actual dtypes, by example tensor:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `layers.32.ffn.experts.0.w1.weight` | **I8** | [3072, 3584] | FP4, **2 values packed per byte** → logical [3072, 7168] |
| `layers.32.ffn.experts.0.w1.scale` | **F8_E8M0** | [3072, 224] | 7168/224 = **32** → **1x32 MXFP4 block scaling** |
| `layers.32.attn.wq_b.weight` | F8_E4M3 | [65536, 1536] | = 128 heads × 512 head_dim |
| `layers.32.attn.wq_b.scale` | F8_E8M0 | [512, 12] | 65536/512 = 1536/12 = **128 → 128x128 blocks** |
| `layers.32.attn.wkv.weight` | F8_E4M3 | [512, 7168] | the single MQA KV projection |
| `layers.32.attn.compressor.wkv.weight` | **BF16** | [1024, 7168] | `coff=2` (C^a ++ C^b) for a CSA layer |
| `layers.32.attn.compressor.ape` | F32 | [4, 1024] | learnable positional bias, `[ratio, coff*head_dim]` |
| `layers.32.attn.indexer.wq_b.weight` | F8_E4M3 | [8192, 1536] | = 64 index heads × 128 dims |
| `layers.32.attn.attn_sink` | **F32** | [128] | per-head learnable softmax sink |

Giving, exactly:

```
FP4 expert weights  = 0.5 + 1/32     = 0.53125   B/param   [verified from shapes]
FP8 weights         = 1 + 1/(128*128) = 1.000061 B/param   [verified from shapes]
BF16                = 2.0            B/param
F32 (mHC, sinks, ape) = 4.0          B/param
```

`[inferred]` **The 1x32 MXFP4 block scaling is the checkable property that matters to us.** The
V4 report's lossless FP4→FP8 dequantization trick requires *"the ratio between the maximum and
minimum scale factors of the FP4 sub-blocks (1x32 tiles) within each FP8 quantization block
(128x128 tiles) does not exceed a certain threshold"* `[verified]`. That is a property you can
test on our NVFP4 GLM-5.2 weights with a short script, and if it holds, an NVFP4 build can
reuse an FP8 pipeline end to end. Note our build uses NVFP4 **group-16** (0.5625 B/param) where
DeepSeek use MXFP4 group-32 (0.53125) — finer scaling, 6% more bytes.

### 8.3 The full parameter and byte breakdown

`[verified]`, summed from the shard headers, scale tensors excluded from parameter counts:

| scope | module | bytes (GB) | params (B) | B/param |
|---|---|---:|---:|---:|
| **MAIN (61 layers)** | routed_experts (384/layer) | 822.05 | 1,547.396 | 0.5312 |
| | attention | 18.29 | 18.294 | 1.0001 |
| | shared_expert | 4.03 | 4.030 | 1.0001 |
| | compressor | 1.34 | 0.670 | 2.0064 |
| | indexer (30 layers) | 0.63 | 0.501 | 1.2473 |
| | router (+ 3× `tid2eid`) | 0.35 | 0.170 | 2.0823 |
| | mHC | 0.34 | 0.084 | 4.0000 |
| | **subtotal** | **847.04** | **1,571.146** | |
| **DRAFTER (`mtp.0–2`)** | all | **41.98** | **77.498** | |
| **GLOBAL** | embed (BF16) | 1.85 | 0.927 | 2.0000 |
| | lm_head (BF16, untied) | 1.85 | 0.927 | 2.0000 |
| **TOTAL** | | **892.73** | **1,650.5** | |

`[inferred]` Main model + embeddings = **1,573.0 B params → the "1.6T" badge**; with the
drafter, 1,650.5 B → the "1.7T" HF badge. Both numbers are honest; they count different things.

### 8.4 Active parameters and bytes per decoded token

`[inferred]`, main model only (drafter excluded — it runs on its own schedule):

| module | active params | B/param | active bytes |
|---|---:|---:|---:|
| attention | 18.294 B | 1.0001 | **18.295 GB** |
| routed experts (6 of 384) | 24.178 B | 0.5312 | 12.845 GB |
| shared expert | 4.030 B | 1.0001 | 4.030 GB |
| lm_head | 0.927 B | 2.0000 | 1.853 GB |
| compressor | 0.670 B | 2.0064 | 1.345 GB |
| indexer (30 layers) | 0.501 B | 1.2473 | 0.625 GB |
| router | 0.170 B | 2.0823 | 0.355 GB |
| mHC | 0.084 B | 4.0000 | 0.336 GB |
| **total** | **48.854 B** | | **39.683 GB** |

**The model card says "49 B activated". My independent sum says 48.854 B.** `[verified]` That
closes to 0.3%, which validates the whole accounting.

`[inferred]` **And it produces the most surprising result in this document: attention is 18.295
GB of the 39.683 GB active read — 46% — which is more than the routed experts (32%).** V4 did
not just move cost out of the KV cache; it moved cost *into the attention weights*. `wq_b` is
1536×65,536 and `wo_b` is 16,384×7,168 per layer, because 128 heads × 512 head_dim = 65,536.
The grouped output projection (`o_groups=16`, `o_lora_rank=1024`) is what stops this being
worse: a dense 65,536×7,168 `o_proj` would be 470 M params/layer where the factored version is
67 M + 117 M. `[inferred]` **A model whose attention weights dominate its decode bandwidth is a
different optimization target from every MoE model we have profiled**, and it is the reason
DP-attention matters even more for V4 than for V3.2.

### 8.5 The HBM roofline

```
weight read per decoded token, perfectly sharded over 8 GPUs:
    39.683 GB / (8 x 7.672 TB/s) = 39.683e9 / 61.376e12 = 646.6 us  ->  1,547 tok/s
```

| model / build | MB/rank/token | ceiling @100% BW | @85% BW | **measured C1** | % of roofline |
|---|---:|---:|---:|---:|---:|
| **V4-Pro-0813 (FP4+FP8)** | 4,960 | **1,547** | 1,315 | **235 tok/s** | **15.2%** |
| GLM-5.2 NVFP4 as shipped | 8,756 | 876 | 745 | **365.5 tok/s** | **41.7%** |
| GLM-5.2 FP8 | 6,673 | 1,150 | 977 | — | — |
| GLM-5.2, all non-expert GEMMs at FP8 | 6,048 | 1,268 | 1,078 | — | — |

`[inferred]` **Two conclusions, and they point in opposite directions.**

1. **V4-Pro is 1.77x lighter per token than our GLM-5.2 NVFP4 build despite having 2.2x the
   parameters.** 4,960 MB/rank vs 8,756 MB/rank. The entire difference is the exclude list:
   GLM-5.2's NVFP4 checkpoint keeps `fused_qkv_a_proj_with_mqa` (2,515 MB/token/rank, 28.7%),
   `o_proj` (1,962 MB), the shared expert (708 MB), `q_b_proj` (654 MB) and all 21 indexers in
   **BF16**. DeepSeek keep the equivalent tensors in **FP8**. **This is the single largest
   uncontested win available on GLM-5.2 and it needs no new kernel — it needs a requantization
   pass.** Moving GLM-5.2's non-expert GEMMs to FP8 takes 8,756 → 6,048 MB and the ceiling from
   876 → 1,268 tok/s.
2. **V4-Pro achieves only 15.2% of its roofline where GLM-5.2 achieves 41.7%.** V4 at C1 is
   **not bandwidth-bound**. 61 layers × (wq_a, wq_b, wkv, compressor wkv, compressor wgate,
   Sinkhorn-normalized mHC pre/post, indexer wq_b + weights_proj + its own compressor, wo_a,
   wo_b, router, 6 expert GEMMs, shared expert, plus the SWA branch and the compressed branch
   concatenated inside one attention) is an enormous number of small dependent kernels.
   `[inferred]` **Do not read V4's low KV cost as a speed advantage. It is a capacity
   advantage bought with op count, and op count is exactly what hurts at 2.7–4.3 ms TPOT.**

### 8.6 Context capacity

`[inferred]` KV pool GiB = `178.34 × mem_fraction_static − weights_GiB`; tokens = pool / marginal
bytes from §2.4. Ignores activation and CUDA-graph memory, so these are upper bounds.

| build | weights GiB/GPU | mfs | KV pool GiB | B/token | **tokens** |
|---|---:|---:|---:|---:|---:|
| V4-Pro-0813 | 103.93 | 0.85 | 47.66 | 5,031 | **10.2 M** |
| V4-Pro-0813 | 103.93 | 0.90 | 56.57 | 5,031 | 12.1 M |
| V4-Flash-0731 | 19.43 | 0.85 | 132.16 | 3,514 | **40.4 M** |
| V3.2-Exp FP8 | 80.27 | 0.85 | 71.32 | 48,068 | 1.59 M |
| GLM-5.2 NVFP4 | 56.72 | 0.85 | 94.87 | 61,464 | 1.66 M |
| GLM-5.2 NVFP4, indexer fix | 56.72 | 0.85 | 94.87 | 54,072 | 1.88 M |

`[inferred]` **V4-Flash on 8xB200 can hold 40 million tokens of KV.** At the AA shape (10 k in,
1.5 k out) that is ~3,500 concurrent sequences of KV — the engine will run out of compute,
scheduler, or CUDA-graph batch buckets long before it runs out of KV. `[inferred]` **This is
what "1M context routinely" actually means operationally: the KV cache stops being the binding
constraint, and every capacity-planning heuristic built around KV becomes wrong.**

---

## 9. Engine support status, and known-good configs

### 9.1 First-party, from the model cards

`[verified]` — fetched from both `DeepSeek-V4-Pro-0813` and `DeepSeek-V4-Flash-0731` model
cards, which agree in structure. **This is the section the industry doc references and does not
contain.**

```bash
# vLLM — DeepSeek's own recommended line, single 4xGB300 node
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'

# SGLang — DeepSeek's own recommended line
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.1
```

`[verified]` Notes DeepSeek attach: *"do not set a separate `--speculative-draft-model-path` as
the target and draft weights therefore come from the same checkpoint"* — direct confirmation of
§4.2. Recommended sampling: `temperature = 1.0`, `top_p = 0.95` for agentic, `1.0` otherwise;
max output 384 K for high/max reasoning effort.

`[inferred]` Four flags worth transplanting to GLM-5.2 experiments regardless: **`--block-size
256` / `--page-size 256`**, **`--mem-fraction-static 0.90`** (we should be testing above 0.85),
**`--swa-full-tokens-ratio 0.1`**, and **`num_speculative_tokens: 7`** — a considerably longer
draft block than our 3-1-4, made safe by DSpark's adaptive verification.

### 9.2 What our own fork already has

`[verified]` in `/home/aman/code/NotSglang`:

| artifact | path |
|---|---|
| V4 model | `python/sglang/srt/models/deepseek_v4.py` |
| V4 DSpark drafter | `python/sglang/srt/models/deepseek_v4_dspark.py` |
| V4 NextN/MTP | `python/sglang/srt/models/deepseek_v4_nextn.py` |
| V4 attention backend | `python/sglang/srt/layers/attention/dsv4/` — `compressor.py`, `compressor_v2.py`, `compress_hip.py`, `indexer.py`, `metadata.py`, `sparse_prefill_utils.py` |
| DSpark speculative | `python/sglang/srt/speculative/dspark_components/` (config, draft, KV inject, block-accept estimator), `dspark_disaggregation.py` |
| B200/B300/GB200/GB300 recipes | `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` + `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4{,-benchmarks}.jsx` |
| 8-GPU V4 tests | `test/registered/amd/test_deepseek_v4_{pro,flash}_fp{4,8}*.py` |

`[verified]` The V4 attention implementation uses **multi-stream overlap for the indexer** —
`self.alt_streams_indexer = alt_streams[-2:]`, with `stream_indexer.wait_stream(...)` /
`current_stream.wait_stream(stream_indexer)` around the `C4Indexer` call, and a separate
`stream_indexer_compressor`. `[inferred]` **That is a directly portable pattern for GLM-5.2's
indexer**: run the indexer on a side stream concurrent with the main attention path rather than
serially. Our indexer is 5.8% of C1 and is on the critical path today.

**The two-batch-overlap finding.** `[verified]` From
`test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py`'s own docstring:

> *"TBO here is the DP-attention TP-MoE variant (`moe_a2a_backend='none'`): it overlaps one
> micro-batch's DP `all_gatherv` (pre-MoE gather) + `reduce_scatterv` (post-MoE combine) with
> the other micro-batch's attention + expert compute (prefill only). Enabled purely via
> `--enable-dp-attention` + `--enable-two-batch-overlap` (no opt-in env). … this runs the TBO
> forward on the real model — **which only DeepSeek-V4 implements** …"*

with env `SGLANG_DP_USE_GATHERV=1`, `SGLANG_DP_USE_REDUCE_SCATTER=1`,
`SGLANG_SHARED_EXPERT_TP1=1`, `SGLANG_DP_SHARED_EXPERT_LOCAL=1`.

`[inferred]` **Our corpus records TBO as "refused for GLM-5.2 because `index_topk_freq=4` means
the TBO op path cannot propagate topk indices." That is true but incomplete: the deeper reason
is that no GLM model implements the `op_gather`/`op_moe`/`op_combine` decomposition at all.**
`deepseek_v4.py` is the working reference. Note also that this TBO variant is **prefill-only**,
so it attacks TTFT (189 ms) rather than TPOT — which is still worth having, since AA counts
TTFT to first token including reasoning tokens.

### 9.3 First-party B200 benchmarks

`[verified]` Extracted from `docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`
in our fork. SGLang **0.5.15**, random dataset, **isl 8192 / osl 1024** — close to the AA shape.
All single-node, 8xB200.

| variant | quant | strategy | TP/DP | conc. | TTFT ms | **TPOT ms** | **implied tok/s/stream** |
|---|---|---|---|---:|---:|---:|---:|
| Flash | FP4 | low-latency | TP4 | 1 | 302 | **2.91** | **343.6** |
| Flash | FP4 | low-latency | TP4 | 16 | 454 | 8.76 | 114.2 |
| Flash | NVFP4 | low-latency | TP4 | 1 | 308 | **2.88** | **347.2** |
| Flash | NVFP4 | low-latency | TP4 | 16 | 466 | 8.67 | 115.3 |
| Flash | FP4 | balanced | TP4+DP4 | 64 | 642 | 23.2 | 43.1 |
| Flash | FP4 | balanced | TP4+DP4 | 256 | 3,147 | 64.0 | 15.6 |
| Flash | FP4 | high-throughput | TP4+DP4, megamoe | 1024 | 104,109 | 70.25 | 14.2 |
| **Pro** | FP4 | low-latency | **TP8** | 1 | **230** | **4.25** | **235.3** |
| Pro | FP4 | low-latency | TP8 | 16 | 446 | 11.56 | 86.5 |
| Pro | NVFP4 | low-latency | TP8 | 1 | 223 | **4.19** | **238.7** |
| Pro | FP4 | balanced | — | 64 | 1,081 | 36.23 | 27.6 |
| Pro | FP4 | high-throughput | megamoe | 1024 | 107,158 | 44.45 | 22.5 |

`[verified]` The launch cells behind those rows, verbatim from `deepseek-v4.jsx` (`verified: true`):

```bash
# B200 · Flash · FP4 · low-latency · single node
--trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Flash \
--tp 4 --moe-runner-backend flashinfer_mxfp4 \
--speculative-algorithm EAGLE --speculative-num-steps 3 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--chunked-prefill-size 4096 --disable-flashinfer-autotune --swa-full-tokens-ratio 0.1

# B200 · Pro · FP4 · low-latency · single node
--trust-remote-code --model-path deepseek-ai/DeepSeek-V4-Pro \
--tp 8 --moe-runner-backend flashinfer_mxfp4 \
--speculative-algorithm EAGLE --speculative-num-steps 3 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
--chunked-prefill-size 8192 --disable-flashinfer-autotune --swa-full-tokens-ratio 0.1 \
--mem-fraction-static 0.90

# B200 · Flash · FP4 · balanced · single node
env SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
--tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend deepep \
--speculative-algorithm EAGLE --speculative-num-steps 1 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

# B200 · Flash · FP4 · high-throughput · single node
env SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
--tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend megamoe
```

**Six things to take from this table** `[inferred]`:

1. **`--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` is
   the low-latency recipe — identical to our GLM-5.2 EAGLE 3-1-4.** Independent convergence on
   the same draft shape from a different team on the same hardware. Our shape is not the
   problem.
2. **We beat both at C1.** 365.5 tok/s vs Flash's 343.6 and Pro's 235.3.
3. **They beat us at C16.** Flash 114.2 tok/s/stream vs our 78–95. Flash's C1→C16 falloff is
   **3.01x**; ours is **4.7x**. That gap is the concrete target.
4. **NVFP4 (`nvidia/DeepSeek-V4-*-NVFP4`) is 1% faster than DeepSeek's MXFP4 at C1** —
   2.88 vs 2.91 ms, 4.19 vs 4.25 ms. Real but marginal; not a reason to requantize on its own.
5. **The MTP ladder is explicit:** low-latency 3/1/4 → balanced 1/1/2 → high-throughput
   speculation **off**, because *"at saturation the verify step costs more than it saves"*
   `[verified, cookbook]`. That is a static version of what DSpark makes dynamic.
6. **A DeepEP sizing invariant we should adopt:** `max-running-requests × MTP_draft_tokens ≤
   SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, or DeepEP's dispatch buffer blows at
   steady state (`deep_ep.cpp:1105`) `[verified, cookbook]`.

`[inferred]` One honesty note on that table: the `tokens_per_sec_per_gpu` column in the source
(677 for Flash at C1) does not reconcile with `1/TPOT` under the file's own stated formula
unless you fold in TTFT and the TP4 replica count. Reconstructing: TTFT 302 ms + 1023 × 2.91 ms
= 3.28 s wall, (8192+1024)/3.28 = 2,810 tok/s per replica, × 2 replicas / 8 GPUs = **702**,
against the recorded 677 — a 4% gap. **I therefore report TTFT and TPOT, which are unambiguous,
and treat the throughput column as approximate.**

### 9.4 Other flags worth knowing

`[verified]`, all from the cookbook source in our fork:

- `--enable-deepseek-v4-fp4-indexer` — *"the experimental FP4 C4 indexer on SM100 GPUs with
  DeepGEMM FP4 indexer support… intended for decode-heavy long-context workloads."*
- `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16` (default `float32`) — halves the C4/C128 compressed
  attention **state** pools without touching weights or the main KV dtype.
- `--moe-a2a-backend megamoe` (or `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`); W4A4 adds
  `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1` and
  `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1`, *"higher throughput with negligible accuracy
  drop (~89.5 GPQA on Pro)"*. Blackwell-only; **only wired into `high-throughput`**
  (sgl-project/sglang#26451).
- `--enable-waterfill` with `--init-expert-location /path/to/expert_distribution_recorder_*.pt`;
  works with MegaMoE, *"applying Waterfill to the fused shared expert slot."*
- `--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000` for TP+DP at high concurrency.
- `nvidia/DeepSeek-V4-{Pro,Flash}-NVFP4` require `--moe-runner-backend flashinfer_trtllm_routed`
  (auto-selected), SM100+, with the MTP layer staying MXFP4-packed via
  `Mxfp4FlashinferTrtllmMoEMethod`.
- Our own 8-GPU V4-Pro test uses `--attention-backend dsv4 --page-size 256
  --mem-fraction-static 0.90 --swa-full-tokens-ratio 0.1 --disable-radix-cache
  --disable-shared-experts-fusion --chunked-prefill-size 8192` with
  `SGLANG_DSV4_FP4_EXPERTS=true`.

---

## 10. Competitive position

`[verified]` Artificial Analysis, fetched 2026-08-17.

| model / provider | output tok/s | TTFT s | $/1M |
|---|---:|---:|---:|
| GLM-5.2 board leaders | **330–336** | — | — |
| V4-Flash-0731 · Nebius | **267.6** | 9.35 | $0.20 |
| V4-Flash-0731 · Makora | 261.5 | 8.28 | $0.20 |
| V4-Flash-0731 · Baseten | 257.5 | 8.49 | $0.14 |
| V4-Flash-0731 · Wafer (FAST) | 214.2 | 11.55 | $0.17 |
| V4-Flash-0731 · CoreWeave | 156.1 | 14.49 | $0.12 |
| V4-Flash-0731 · Fireworks | 126 | 0.97 | $0.08 |
| **V4-Flash-0731 · DeepSeek (first-party)** | **111** | 1.23 | $0.11 |
| V4-Pro-0813 (max) · DeepSeek | **~84** | — | — |
| V4-Pro (Apr, max) · DeepSeek | ~69 | — | — |
| V4-Pro (non-reasoning) · DeepSeek | ~32.2 | — | — |

`[inferred]` **Three readings.**

1. **DeepSeek is not the leaderboard threat.** Their own API serves their flagship at ~84
   tok/s. They are optimizing $/token — and their price sheet shows it: V4-Flash output at
   $0.66 off-peak / $1.32 peak per 1M, against R1's $2.19 in Feb 2025, for a model that went
   from 671 B/37 B-active at 128 K to 284 B/13 B-active at 1 M.
2. **The 2.4x spread between DeepSeek's own 111 tok/s and Nebius's 267.6 on the same weights
   is the value of the serving stack, isolated.** That is the clearest available measurement of
   how much engine engineering is worth, and it is roughly the same multiple as the gap between
   our GLM-5.2 and TileRT's claimed 500 tok/s on GLM-5 FP8.
3. **Note the TTFT/throughput inversion.** The fast providers (Nebius 267.6 tok/s) have TTFT of
   8–14 s; the slow ones (Fireworks 126 tok/s) have TTFT under 1 s. `[inferred]` That is the
   signature of large-batch, high-queueing deployments trading TTFT for TPOT. **AA scores both**,
   and TTFT is counted to first token including reasoning tokens — so a deployment tuned purely
   for TPOT can lose. Our 189 ms TTFT is a genuine competitive asset and should not be traded
   away casually.

---

## 11. Open questions, and what to measure on our box

Ranked by expected value. Each is a specific experiment, not a topic.

1. **`[inferred]` Requantize GLM-5.2's non-expert GEMMs from BF16 to FP8.** §8.5 puts the
   ceiling at 876 → 1,268 tok/s. DeepSeek keep exactly these tensors in FP8 and lose nothing.
   **Measure:** GSM8K + our eval set on an FP8-everything-but-experts build. Highest-value item
   in this document.
2. **`[verified]` Does the SM100 `head64` FlashMLA decode kernel outperform our current path
   for GLM-5.2 under DP-attention?** GLM-5.2 has 64 query heads and a 576 latent — an exact
   match for `ModelType::V32` in that kernel. Under TP8 we have `h_q = 8` and no matching tile.
   **Measure:** kernel-level, both configurations, at our real sequence lengths.
3. **`[verified]` Wire the DeepGEMM SM100 FP4 paged MQA-logits kernel to GLM-5.2's indexer.**
   `kHeadDim == 128` is supported; halves indexer cache bytes. Our fork already has the
   equivalent flag for V4. **Measure:** indexer time (5.8% of C1) and top-2048 recall against
   the FP8 path.
4. **`[reported]` FP32→BF16 index scores before top-k.** One dtype change, published at 2x
   top-k selector speed and 99.7% recall. **Measure:** recall on our own prompts first.
5. **`[verified]` Port `deepseek_v4.py`'s `op_gather`/`op_moe`/`op_combine` TBO decomposition to
   GLM-5.2.** Prefill-only, so it attacks the 189 ms TTFT. **Measure:** TTFT at 10 k input, and
   whether the `index_topk_freq=4` topk-propagation problem is actually load-bearing or just
   the first error encountered.
6. **`[verified]` Run the indexer on a side CUDA stream**, as `deepseek_v4.py` does with
   `alt_streams_indexer`. **Measure:** whether the indexer can be hidden behind the compressed
   attention branch at C1.
7. **`[inferred]` Stop allocating GLM-5.2's indexer KV on all 78 layers.** 7.39 GB per 1M
   tokens per GPU, and V4 demonstrates the correct pattern. **Measure:** just do it and check
   the pool size in the startup log.
8. **`[inferred]` Test the 1x32-scale-ratio condition on our NVFP4 weights.** If the max/min
   FP4 sub-block scale ratio within each 128x128 FP8 block is bounded, the FP4→FP8 lossless
   dequantization path opens and an NVFP4 build can reuse an FP8 kernel pipeline. **Measure:**
   a 30-line script over the safetensors.
9. **`[inferred]` Quantify the C1 expert-placement spread.** §3.2 predicts a 2.9x ceiling
   spread from which of the 8 ranks own the top-k experts. **Measure:** per-rank MoE kernel time
   at C1, correlated with the routing decision. If this is a large fraction of the 47%
   rank-arrival skew, redundant-experts-for-coverage becomes a real idea rather than a
   speculative one.
10. **`[verified]` Try `--mem-fraction-static 0.90` and `--page-size 256`.** DeepSeek ship both.
    We run 0.85 and page 64. **Measure:** KV pool size, TMA efficiency, and whether anything
    breaks.
11. **`[unverified]` Does a dense-masked prefill path beat DSA at 10 k input?** DeepSeek ship
    two prefill implementations and switch on length; their published sparse/dense crossover is
    ~3,000 tokens at topk=2048, which is uncomfortably close to AA's 10 k. **Measure:** TTFT
    both ways at 10 k.
12. **`[unverified]` What is our actual achievable HBM fraction?** Every roofline here assumes
    80–100% of 7.672 TB/s. `00-hardware` flags this as unestablished. Until it is measured,
    every ceiling in §8.5 carries a ±15% band.

**Two things I could not source** `[unverified]`, stated so nobody assumes I checked:
- The exact dtype of V4's 8 per-token KV scale values. `NUM_SCALES_EACH_TOKEN = 8` with
  `QUANT_TILE_SIZE = 64` over 448 nope dims, and the kernel defines
  `using e8m0 = __nv_fp8_e8m0`, so 1 byte each is the natural reading — but the TMA stride
  comment excludes them, so they live in a separate tensor whose dtype I did not confirm. My
  584 B/entry figure would become 608 B if they are FP32, changing §2.4's totals by ~4%.
- Whether DeepSeek's production V4 deployment still uses PD disaggregation and at what EP
  degree. The Feb-2025 EP32/EP144 figures are for V3/R1. No equivalent disclosure exists for V4.

---

## 12. Sources

All fetched 2026-08-17 unless noted. `[verified]` means I read the artifact at this URL.

**Models and configs** — every `config.json` and `model.safetensors.index.json` below was
downloaded and parsed:
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 (card, config, index, all 66 shard headers, `inference/model.py`, `inference/kernel.py`)
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 (card, config, index)
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro · https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash (April configs + indices)
- https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp (config, index, shard header)
- https://huggingface.co/deepseek-ai (org listing)

**Papers and docs:**
- arXiv **2606.19348** — *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence* (title + abstract verified at arxiv.org/abs/2606.19348)
- arXiv **2412.19437** — DeepSeek-V3 technical report (MLA, aux-loss-free balancing, MTP)
- arXiv **2505.09343** — *Insights into DeepSeek-V3* (ISCA'25); Table 1 KV cache, the TPOT ceiling calculation
- arXiv **2512.02556** / `DeepSeek_V3_2.pdf` — DSA / lightning indexer
- arXiv **2607.05147** — *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*
- https://api-docs.deepseek.com/updates/ — full change log
- https://api-docs.deepseek.com/quick_start/pricing
- https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md

**Kernel repos — trees and source files read, not just READMEs:**
- https://github.com/deepseek-ai/FlashMLA — full recursive tree; `csrc/sm100/decode/head64/config.h`; `csrc/sm100/decode/head64/instantiations/{v32,model1}.cu`; `csrc/sm100/decode/head128/README.md`
- https://github.com/deepseek-ai/DeepGEMM — full recursive tree; `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh` (read in full)
- https://github.com/deepseek-ai/DeepEP — branch list; `hybrid-ep` recursive tree; `docs/README_Hybrid-EP.md`

**Benchmarks:**
- https://artificialanalysis.ai/models/deepseek-v4-flash/providers (13-provider table)
- https://artificialanalysis.ai/models/deepseek-v4-pro (V4-Pro-0813 speed)

**Local, in our own repositories:**
- `/home/aman/code/NotSglang/docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`
- `/home/aman/code/NotSglang/docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx`
- `/home/aman/code/NotSglang/docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx`
- `/home/aman/code/NotSglang/python/sglang/srt/models/deepseek_v4{,_dspark,_nextn}.py`
- `/home/aman/code/NotSglang/python/sglang/srt/layers/attention/dsv4/`
- `/home/aman/code/NotSglang/python/sglang/srt/speculative/dspark_components/`
- `/home/aman/code/NotSglang/test/registered/amd/test_deepseek_v4_*.py`
- `/home/aman/code/research/04-industry/04-deepseek-open-infrastructure.md`
- `/home/aman/code/research/05-models/01-glm-5.2-serving-on-b200.md`
- `/home/aman/code/research/00-hardware/00-this-machine-ground-truth.md`
