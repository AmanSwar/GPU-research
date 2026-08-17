# Cross-model patterns, and a serving stack that is fast on all four

**Research date:** 2026-08-17
**Inputs:** `00-local-weights-ground-truth.md`, `01-glm-5.2-serving-on-b200.md`,
`02-kimi-k3-serving-on-b200.md`, `03-qwen3-serving-on-b200.md`,
`04-deepseek-v3.2-v4-serving-on-b200.md`, `06-open-source-kernels-and-implementations-to-steal.md`,
`00-hardware/00-this-machine-ground-truth.md`,
`NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`.
**Target:** 8x NVIDIA B200 SXM, 148 SMs, HBM **7.67 TB/s** driver-derived, 178.34 GiB
usable/GPU, 956.25 GB/s NVLink5 per GPU through NVSwitch.

**Label convention** (same as the rest of the series)

| label | meaning |
|---|---|
| `[verified]` | measured on this box, or read out of a config/checkpoint/source file by one of the five input documents, which state the path |
| `[reported]` | a vendor/paper asserts it and an input document read the assertion |
| `[inferred]` | arithmetic done here from `[verified]` inputs; the arithmetic is shown |
| `[unverified]` | not established; flagged |

**Method note.** This document does no new fetching. It is a *reconciliation*: it puts
six models on one bandwidth denominator, one collective-cost model, and one expert-residency
model, and it reports where the five input documents disagree with each other. Three such
disagreements were found and are corrected in §4.1. Every restated number is traceable to
the input document that established it.

**One correction applied everywhere.** `01-glm-5.2`, `02-kimi-k3` and `03-qwen3` all use a
rounded **8.0 TB/s** HBM figure; `04-deepseek` uses the driver-derived **7.672 TB/s** from
`00-hardware/00-this-machine-ground-truth.md`. The driver figure is the tiebreaker, so every
roofline in this document is **~4.3% lower** than the same quantity in the first three
documents. This is stated once and not repeated.

---

## Status: the thesis, adjudicated

The thesis put to this document was that the 2025-2026 frontier open models have converged
on a common shape: **very large sparse MoE with fine-grained experts plus a shared expert; a
compressed or sparse attention mechanism that collapses KV bandwidth; a multi-token-prediction
head for speculative decoding; FP8-native weights with NVFP4 builds appearing; and long
context in a cheap-attention regime.** And that GLM-5.2's 576-wide MLA latent being
byte-identical in shape to DeepSeek's and K3's is evidence of how deep that convergence runs.

**Verdict: four of six clauses hold, one is refuted as stated, and one holds in outcome while
the mechanisms underneath it have split into two families that share no kernels.** The
576-latent observation is true and is the weakest piece of evidence in the set, because it is
a *trailing* convergence — DeepSeek, who invented it, has already left it behind.

| clause | verdict | the correction |
|---|---|---|
| very large sparse MoE, fine-grained experts + shared expert | **HOLDS**, and tighter than claimed | activated/total is **3.11%-5.35%** across six models from three labs. But per-expert size spans **5.2x** (12.6 M to 66.1 M params) and `topk/E` spans **2x**, so "fine-grained" is not one granularity. §1.1 |
| compressed or sparse attention that collapses KV | **HOLDS in outcome, SPLIT in mechanism** | two disjoint families: low-rank latent + top-k sparsity (GLM-5.2, DeepSeek) vs hybrid linear attention (K3, Qwen). They share **no attention kernel**, and the second family adds a per-*sequence* recurrent state pool that inverts the memory model and breaks tree speculation. §1.3 |
| the 576-wide MLA latent is common | **HOLDS for three of four families, and is being abandoned** | byte-identical (512+64, 656 B/token/layer FP8) in GLM-5.2, DeepSeek <=V3.2, and Kimi K2/K3's MLA layers `[verified]`. **DeepSeek-V4 dropped it** (head_dim 512, no `kv_lora_rank` key, entry serves as both K and V). **Qwen never used MLA.** §1.2 |
| a multi-token-prediction head | **HOLDS 3/4** | Kimi K3 ships `num_nextn_predict_layers: 0` and speculates via a separately published 2.25 B model. DeepSeek's *production* checkpoints already replaced MTP-1 with a 3-block drafter. But the **draft shape converges harder than the head does**: four teams independently ship `steps=3, topk=1, draft_tokens=4`. §1.4 |
| FP8-native with NVFP4 appearing | **REFUTED as stated** | the frontier is past FP8: DeepSeek-V4 has `expert_dtype: fp4` as a *config key* and K3 is MXFP4 **QAT'd from the SFT stage**. And NVFP4 is a **regression** on K3 (+30 GiB) and a **26% slowdown** on Qwen3.5-397B. The live axis is not the format, it is the exclusion list. §1.5 |
| long context in a cheap-attention regime | **HOLDS on context, ZERO convergence on position** | 3 of 4 ship 1,048,576 natively. Positional encoding has **four mutually incompatible answers** (plain RoPE theta 8e6 / YaRN x16 with three rotary configs in one model / NoPE / partial-RoPE 0.25 + mRoPE + static YaRN). Every failure mode here is silent. §1.6 |

---

## Bottom line

Ranked by expected value, most important first. All of these are cross-model conclusions;
none of them is visible from a single model's document.

**1. Three of the four "4-bit" frontier checkpoints read 74-82% of their C1 decode bytes as
BF16. DeepSeek reads 9.8%.** `[inferred]` This is the sharpest cross-model finding in the
corpus and it is a *checkpoint policy* problem, not a kernel problem:

| checkpoint | BF16 share of C1 decode bytes | what is BF16 |
|---|---:|---|
| `nvidia/GLM-5.2-NVFP4` | **81.8%** | all attention, shared experts, 3 dense layers, whole MTP layer, lm_head |
| Kimi K3 MXFP4 (native) at TP8 | **~80.7%** | all KDA + MLA linears, both shared experts, dense MLP, lm_head |
| `Inferact/Qwen3.8-2.4T-NVFP4` | **74.3%** | all gated attention, all Gated DeltaNet, embeddings, lm_head, MTP |
| `deepseek-ai/DeepSeek-V4-Pro-0813` | **9.8%** | only compressor, router, mHC, lm_head |

**2. But the fix is only principled for GLM-5.2.** `[inferred]` GLM-5.2's BF16 tail is plain
softmax-attention projections and shared experts — exactly the tensors DeepSeek keeps in FP8
and loses nothing on. K3's and Qwen's BF16 tails are *recurrent*-attention projections, where
error compounds over 69 layers of sequence-length recurrence and where the checkpoint's own
ignore list keeps them BF16 deliberately (`mamba_ssm_dtype: float32` says so out loud). So the
one-line recommendation "requantize the non-expert GEMMs" is right for GLM-5.2, worth ~12% for
Qwen, and a genuine quality risk for K3. §1.5, §4.

**3. Every architectural improvement in this set makes the collective problem relatively
worse, and requantizing GLM-5.2 makes it 45% worse.** `[inferred]` Collective *latency* per
token is `layers x 2 x ~3.44 us` and is unchanged by quantization, sparsity or expert count.
The HBM floor shrinks with all three. The ratio:

| configuration | collective latency | HBM floor | ratio |
|---|---:|---:|---:|
| GLM-5.2 NVFP4 as shipped, TP8 | 537 us | 1,141 us | **0.47** |
| GLM-5.2 FP8, TP8 | 537 us | 870 us | 0.62 |
| **GLM-5.2 with non-expert GEMMs at FP8, TP8** | 537 us | 690-788 us | **0.68-0.78** |
| DeepSeek-V4-Pro, TP8 | 420 us | 647 us | 0.65 |
| Qwen3.5-397B-A17B FP8, TP8 | 413 us | 266 us | **1.55** |

The +45% to +65% roofline win from requantizing GLM-5.2 (876 -> 1,268-1,449 tok/s; the corpus
does not settle which, see §4.1 correction 4) will not be realized end-to-end unless the
collective term is attacked in the same change. §3.2.

**4. Pure TP at C1, DP-attention + EP at throughput, crossover between C16 and C64.**
`[verified]` Four teams, four models, one answer — read off the published recipe cells, not
argued: GLM-5.2 low-latency B200 is TP8 with no `--dp`; DeepSeek-V4-Pro low-latency B200 is
TP8, V4-Flash is TP4; Kimi K3's low-latency B200 cell is TP16 with an explicit "never enable
EP for latency"; Qwen3.5-397B's latency recipe is `--tensor-parallel-size 8`. Every
`balanced`/`high-throughput` cell across all four adds DP-attention and EP. The published cell
boundaries put the switch at C16-C64. §3.6.

**5. Nobody's C1 decode is dominated by the MoE.** `[inferred]` Routed experts are 18-32% of
decode bytes at C1 on every model in the set (GLM-5.2 18.2%, K3 19.3%, Qwen3.8 25.7%,
V4-Pro 32.4%). The 68-82% remainder is attention projections, shared experts, the lm_head, and
whatever the quantizer skipped. **A stack tuned on the assumption that "MoE is the hot spot"
is tuned for the wrong regime at C1.** It becomes true only past ~C64. §3.4.

**6. GLM-5.2 has the smallest roofline gap of any model in the set.** `[inferred]` 365.5 of
876 tok/s = **41.7%** of its HBM roofline, against DeepSeek-V4-Pro's 15.2%, V4-Flash's ~12.7%,
Kimi K3's 12.7% at TP16, and Qwen3.8-2.4T's ~28% at TP16. We are already the closest to the
wall, which is good news competitively and bad news for easy wins: the remaining headroom is
2.40x, not 6.6x. §7.

**7. Five kernels are shared and worth hand-optimizing once; one is not shared and should
not be built.** `[inferred]` Ranked by breadth x cost: grouped blockscaled expert GEMM (4/4
models, top-3 by cost on every one of them), fused allreduce+residual+norm+quant (4/4, and the
only cost that does not amortize), the MTP verify step (4/4, and the largest measured win in
the corpus at 3.09x), MQA-shaped latent decode attention (3/4), the top-k indexer (2/4, but
the kernel already exists MIT-licensed and SM100-native — it needs wiring, not writing). The
linear-attention decode kernel (KDA/GDN) serves **only** K3 and Qwen and is a separate
investment, not a shared one. §2.

**8. One engine serves all four with a two-registry decomposition, and per-model code is
unavoidable in exactly six places.** `[inferred]` The unavoidable six are: the KV/state pool
shape function (three fundamentally different models, and this is where SGLang's current
GLM-5.2 waste lives), positional encoding, the routing pre-pass, the draft-model forward,
the checkpoint quantization-override policy, and the chat/reasoning/tool-call parsers.
Everything else — scheduler, batcher, sampler, verify loop, communicator, allocator interface
— can be model-agnostic. §5.

**9. The corpus contains five internal numeric inconsistencies and one unresolved contradiction
that blocks every collective measurement.** `[inferred]` Corrected in §4.1 and §6.4. The most
consequential is that `01-glm-5.2` gives two irreconcilable figures for the requantized GLM-5.2
build (6,048 vs 5,293 MB/rank/token), so the headline requant win is a **band of +45% to +65%**,
not a point estimate. The contradiction: the ledger states FlashInfer allreduce fusion was off
in every run,
yet `tllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` (8.2%) and `twoshotAllreduceKernel`
(4.3%) are the #2 and #6 kernels in the profile. Until that is resolved, no collective number
in any of the five documents means anything.

---

## 1. The convergence, tested clause by clause

### 1.1 Fine-grained sparse MoE with a shared expert — HOLDS, tighter than claimed

`[verified]` from the config values established in the input documents:

| model | total params | activated | **activated/total** | routed E | top-k | shared | `moe_intermediate` | hidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.2 | 753.33 B | 40.30 B | **5.35%** | 256 | 8 | 1 | 2048 | 6144 |
| DeepSeek-V3.2 | 685 B | 37 B | 5.40% | 256 | 8 | 1 | 2048 | 7168 |
| DeepSeek-V4-Pro-0813 | 1,573.0 B (main) | 48.85 B | **3.11%** | 384 | 6 | 1 | 3072 | 7168 |
| DeepSeek-V4-Flash | 284 B | 13 B | 4.58% | 256 | 6 | 1 | 2048 | 4096 |
| Kimi K3 | 2,780 B | 104 B | **3.74%** | 896 | 16 | **2** | 3072 | 7168 (experts at **3584**) |
| Qwen3.8-2.4T-A95B | 2,446.2 B | 93.10 B | 3.81% | 512 | 10 | 1 | 2048 | 8192 |
| Qwen3.5-397B-A17B | 403.4 B | 16.31 B | 4.04% | 512 | 10 | 1 | 1024 | 4096 |

**Seven models, three labs, one release year, and the activation ratio lives in a 1.7x band
between 3.11% and 5.35%.** `[inferred]` That is a real convergence and it is the strongest
clause in the thesis. Every one of them has at least one always-on shared expert; every one
of them uses many small experts rather than few fat ones.

**Where "fine-grained" stops being one thing** `[inferred]`:

| model | params/expert | topk/E | routed-active params | expert width / hidden |
|---|---:|---:|---:|---:|
| Qwen3.5-397B | 3x4096x1024 = **12.58 M** | 1.95% | 7.55 B | 0.250 |
| DeepSeek-V4-Flash | 3x4096x2048 = 25.17 M | 2.34% | — | 0.500 |
| Kimi K3 | 3x3584x3072 = 33.03 M | 1.79% | 48.62 B | 0.857 (of latent) |
| GLM-5.2 | 3x6144x2048 = 37.75 M | **3.13%** | 22.65 B | 0.333 |
| Qwen3.8-2.4T | 3x8192x2048 = 50.33 M | 1.95% | 46.31 B | 0.250 |
| DeepSeek-V4-Pro | 3x7168x3072 = **66.06 M** | **1.56%** | 24.18 B | 0.429 |

A 5.2x spread in per-expert size and a 2x spread in `topk/E`. `[inferred]` The `topk/E` column
is the one that matters operationally, because it sets how fast the expert bank saturates as
batch grows — see §3.4. **GLM-5.2 has the highest `topk/E` in the set, which is why it
amortizes at lower concurrency than anything else here** (87% of experts read per step at
C64 against Qwen's 72% and V4-Pro's 64%) and why its measured 40,794 tok/s at C64 does not
transfer as an expectation to the others.

**Router convergence** `[verified]`: aux-loss-free `noaux_tc` bias-corrected top-k in
GLM-5.2, DeepSeek V3/V3.2/V4, and Kimi K3 — four independent implementations of the same
mechanism, with the bias applied to *selection only* and the unbiased score used as the gate
weight. Scoring function diverges: sigmoid (GLM-5.2, DSv3.2, K3), `sqrtsoftplus` (V4). Qwen's
scoring function is `[unverified]` — not established by `03-qwen3`.

**Group-limited routing is dead** `[verified]`: GLM-5.2 `n_group = 1`, Kimi K3
`num_expert_group: 1`, DeepSeek-V4 has no `n_group`/`topk_group` key at all (V3 had 8/4).
`[inferred]` That is a collective bet on large scale-up NVLink domains — the regime this box
is in — and it means a token's experts can land on any rank with no cap under EP.

**Four genuine breaks in this clause** `[verified]`:

1. **Kimi K3 runs its experts in a 3584-d latent space, not the 7168-d hidden space**
   (`routed_expert_hidden_size: 3584`), with a down-projection in and an up-projection out.
   Nobody else does this. It changes the grouped-GEMM `K` dimension and adds 4.727 B of
   projection weights.
2. **K3 has two shared experts**; everyone else has one.
3. **DeepSeek-V4 hash-routes the first three MoE layers** from a frozen
   `int32[vocab, topk]` `tid2eid` table — routing for three layers is known before the forward
   pass begins. GLM-5.2 solves the same early-layer-router-instability problem with three
   *dense* layers at 226 M params each; V4's answer costs 3.10 MB. `[inferred]` Hash routing
   is strictly cheaper and does the same job.
4. **The shared expert is sometimes fused into the routed kernel and sometimes not, silently.**
   `[verified]` SGLang's `determine_num_fused_shared_experts` fires for GLM-5.2 under plain
   TP8, so the MoE kernel runs **257 experts, not 256** — and the fusion turns off with no log
   line the moment EP, DeepEP, SBO or TBO is enabled. K3 is worse: under EP a2a SGLang
   *replicates* the shared experts at tp1 on every rank (`kimi_k3.py:519`), turning 3.04 GB/GPU
   of shared-expert traffic into 24.31 GB/GPU. `[inferred]` **The shared expert's placement is
   the single most consequential undocumented shape change in the set.**

### 1.2 The 576-wide MLA latent — HOLDS for three, and its inventor has left

`[verified]` The latent is byte-identical in shape across three model families:

| model | `kv_lora_rank` | `qk_rope_head_dim` | latent | FP8 on-disk B/token/layer |
|---|---:|---:|---:|---:|
| DeepSeek-V3 / V3.1 / V3.2 | 512 | 64 | **576** | 656 |
| Kimi K2 | 512 | 64 | **576** | 656 |
| Kimi K3 (its 24 MLA layers) | 512 | 64 | **576** | 656 |
| GLM-5.2 | 512 | 64 | **576** | 656 |

`[verified]` And the 656-byte FP8 layout (512 B e4m3 nope + 16 B fp32 per-128 block scales +
128 B **BF16 rope tail, deliberately unquantized**) is confirmed three independent ways in the
corpus: FlashMLA's Hopper FP8 deep-dive, SGLang's own
`kv_cache_configurator.calculate_mla_kv_cache_dim`, and a verbatim comment in FlashMLA's
SM100 decode `config.h`: *"So we set this to 656 for V32 and 576 for MODEL1."*

**This is a real convergence, and it is the shallowest one in the thesis.** Four things break
it, in increasing severity `[verified]`:

**(a) The cache is byte-identical; the geometry above it is not.**

| | q heads | `q_lora_rank` | `qk_nope` | `v_head_dim` | extras |
|---|---:|---:|---:|---:|---|
| DeepSeek-V3.2 | 128 | 1536 | 128 | 128 | — |
| Kimi K3 | 96 | 1536 | 128 | 128 | output gate, **NoPE** |
| GLM-5.2 | **64** | **2048** | **192** | **256** | — |

`[inferred]` **`v_head_dim=256 != qk_nope_head_dim=192` is the quiet killer.** DeepSeek's case
has them equal at 128, so a kernel templated on that equality — which is the natural way to
write it — will not compile for GLM-5.2 or will be silently wrong. `06-open-source-kernels`
flags this as the top risk for adopting FlashMLA, and it is unresolved `[unverified]`. The
absorbed-decode BMM shapes at TP8 are `w_kc [8 heads, 192, 512]` and `w_vc [8, 512, 256]` for
GLM-5.2 against `[16, 128, 512]` / `[16, 512, 128]` for V3.2. **Same cache, different kernel.**

**(b) DeepSeek-V4 left the family.** `[verified]` No `kv_lora_rank` key, no `qk_nope_head_dim`,
no `v_head_dim`. Instead `head_dim: 512`, `num_key_value_heads: 1`, and — read off FlashMLA's
`config.h` — 448 nope dims in FP8 plus 64 rope dims in BF16, tile-quantized at 1x64 giving 8
scales per token, with `V_HAVE_ROPE = true` meaning **the entry serves as both key and value**.
The marginal KV cost fell from 48,068 B/token (V3.2) to 5,031 B/token (V4-Pro), a 9.6x
reduction, and it was achieved by abandoning the 576 latent, not by compressing it further.

**(c) Qwen never joined.** `[verified]` `Qwen3.8-2.4T-A95B` is gated GQA: 64 query heads,
**4** KV heads, `head_dim: 256`, `attn_output_gate: true` (so `q_proj` is 2x wide),
`partial_rotary_factor: 0.25`. `Qwen3.5-397B-A17B` has 32 query heads and **2** KV heads. No
low-rank latent anywhere in the family.

**(d) What actually generalizes is not 576 — it is "TP cannot shard the KV cache."**
`[verified]/[inferred]` This holds for **all four families** and for two different reasons:

| model | KV heads | TP8 consequence |
|---|---:|---|
| GLM-5.2, DeepSeek V3.2/V4, K3's MLA layers | **1** | MLA/MQA has one KV head; every rank holds a full copy. TP shards nothing. |
| Qwen3.8-2.4T | 4 | `4 < 8`, so K/V heads are **replicated 2x** under TP8 |
| Qwen3.5-397B | 2 | `2 < 8`, so K/V heads are **replicated 4x** — the doc computes the KV cost rising from 8.05 to 32.2 GB/seq at 262 K |

`[inferred]` **Therefore DP-attention costs nothing in KV memory on any model in this set, and
on Qwen it actively *saves* 2-4x.** That is a far stronger and far more actionable convergence
than the 576 latent, and it is the memory half of the DP-attention argument in §3.5 — the half
that usually kills the proposal, and it is wrong.

### 1.3 Attention that collapses KV — HOLDS in outcome, SPLIT in mechanism

All four families reduced KV bandwidth by a large factor. They did it two incompatible ways.

| | Family A: latent + top-k sparsity | Family B: hybrid linear attention |
|---|---|---|
| members | GLM-5.2, DeepSeek V3.2, DeepSeek V4 | Kimi K3, Qwen3.5/3.6/3.8 |
| mechanism | low-rank MQA latent, then select a top-k subset of keys per query (or pool keys into compressed blocks) | replace 75% of layers with a gated delta-rule recurrence carrying a fixed-size state; keep softmax attention on the rest |
| layer split | 78/78 sparse (GLM); 30 CSA + 31 HCA (V4-Pro) | 69 KDA + 24 MLA (K3); 69 GDN + 23 GQA (Qwen 2.4T) |
| per-token KV | 61,464 B (GLM as shipped); 5,031 B (V4-Pro) | 13,824 B (K3, 24 layers FP8); 47,104 B (Qwen 2.4T, 23 layers FP8) |
| per-**sequence** state | none | **232.3 MB/slot BF16 unsharded** (K3); **574 MiB/seq fp32** (Qwen 2.4T) |
| context-independent cost | no | **yes** — this is the whole point, and the whole problem |
| decode kernel | sparse MQA MLA + top-k selector | KDA / Gated DeltaNet recurrence |
| kernel sharing across the split | **none** | **none** |

`[verified]` from configs. Three consequences that no single-model document can see:

**1. The memory model inverts.** `[inferred]` For Family B the recurrent state is per-*sequence*
and context-independent, so below a crossover length the state costs more than the KV cache:
Qwen3.8-2.4T crosses at **~12,780 tokens**, Qwen3.5-397B at **~6,420**, Kimi K3 at
**~90,000** (116.2 MB/GPU/request of state at TP8 against 54 MB of DCP8 KV at 128 K). At the
Artificial-Analysis input length of ~10 k, **Qwen's state pool is the binding constraint and
its KV cache is not.** Every capacity heuristic built on `tokens -> blocks` is wrong for
Family B.

**2. Tree speculation dies.** `[inferred]` Forking a KV cache is a pointer copy. Forking a
552 MiB fp32 recurrent state is a 552 MiB memcpy per branch, and rollback needs either
re-running the recurrence or checkpointing per tree node. `03-qwen3` observes that every
published Qwen config uses `--speculative-eagle-topk 1` and reads it as architectural
necessity — correctly. `[verified]` K3's DSPARK adds 8 state slots per request at block 7
without `--enable-linear-replayssm-spec` (116 -> 348.5 MB/GPU/request, a 3x jump), which is
the same problem wearing different clothes.

**3. The two families do not share an attention kernel, and pretending otherwise costs a
quarter.** `[verified]` SGLang's own tree admits this: `kda_blackwell/`, `cutedsl_kda.py`,
`kda_fused_decode.py`, `kda_packed_decode.py`, `fla/kda.py`, `kda_decode_mtp.py` for K3, and
`gdn_cutedsl.py` / `gdn_flashinfer.py` for Gated DeltaNet — a completely separate kernel suite
from `dsa_backend.py` / `flashmla_backend.py` / `tokenspeed_mla_backend.py`. `[inferred]` KDA
and GDN *are* siblings (both 128-dim heads, conv kernel 4, gated, and SGLang files them side
by side) so **one** linear-attention investment covers both; but zero of it covers GLM-5.2 or
DeepSeek. §2.6.

**One thing the split does share, and it is worth taking.** `[verified]` DeepSeek's own
justification for why DSA and MLA are inseparable: *"each key-value entry must be shared across
multiple queries for computational efficiency. Therefore, we implement DSA based on the MQA
mode of MLA, where each latent vector will be shared across all query heads."* `[inferred]`
Sparse attention over a per-head KV cache would gather `h` different key sets per token; over
an MQA latent it gathers one. **GLM-5.2 inherits this for free. Qwen, with 2-4 KV heads,
does not — which is an independent structural reason Qwen went linear rather than sparse.**

### 1.4 An MTP head — HOLDS 3/4, and the draft *shape* converges harder

| model | in-checkpoint speculation | what it actually is |
|---|---|---|
| GLM-5.2 | `num_nextn_predict_layers: 1`, layer index 78 | **a complete decoder layer** — full MLA, its own indexer, its own 256-expert MoE + shared expert, plus `enorm`/`hnorm`/`eh_proj`/`shared_head.norm`. 1,573 tensors. **10.03 GB (FP8) / 19.91 GB (NVFP4).** Not a lightweight head. |
| DeepSeek-V3 / V3.2 / V4-April | `num_nextn_predict_layers: 1` | one MTP block; 85-90% second-token acceptance, 1.8x TPS `[reported]` |
| **DeepSeek-V4-0731/0813** | **3 `mtp.N` blocks** | `04-deepseek` established by tensor-namespace enumeration that the production checkpoints replaced MTP-1 with a **3-block semi-autoregressive DSpark drafter** — full-width MoE layers with SWA-128 and no compression, `mtp.0` carrying `main_proj`/`main_norm`, fed from target layers [58,59,60]. **41.98 GB = 4.7% of the checkpoint.** |
| Kimi K3 | **`num_nextn_predict_layers: 0`** | none. Speculation is `RadixArk/Kimi-K3-DSpark`, a **separately published 2.25 B BF16 model** with 5 Qwen3-style GQA layers reading target layers [7,23,51,67,83] at block 7 |
| Qwen3.5/3.6/3.8 | `mtp_num_hidden_layers: 1` on **every** member, 0.8 B through 2.4 T | one MTP transformer layer, `mtp_use_dedicated_embeddings: false` |

`[verified]` So the clause is 3/4 true, and the exception is instructive: Moonshot's answer is
a separate artifact, and DeepSeek's *published* answer (MTP-1) is not what they ship in
production. **The generalizable statement is "speculation is part of the release," not "there
is an MTP-1 head."**

**What converges much harder is the draft shape.** `[verified]` Four independent teams:

| model | published low-latency draft shape |
|---|---|
| GLM-5.2 (SGLang cookbook, B200 NVFP4) | `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` |
| GLM-5.2 (our fork's arch default) | `(num_steps=3, eagle_topk=1, num_draft_tokens=4)` |
| DeepSeek-V4-Pro / V4-Flash (SGLang cookbook, B200) | `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |
| Qwen3.5-397B (model card, SGLang) | `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |
| Kimi K3 (SGLang cookbook) | `--speculative-dspark-block-size 7` (a block, i.e. topk=1 by construction) |

`[inferred]` **`eagle_topk = 1` in every published config for every model. Nobody ships a
tree.** For Family B that is architectural necessity (§1.3). For Family A it is an empirical
finding, and `04-deepseek` notes the independent convergence on 3-1-4 explicitly: *"our shape
is not the problem."*

**And the ladder converges.** `[verified]` Both cookbooks encode the same three-rung structure
with the same stated reason:

| regime | GLM-5.2 B200 NVFP4 | DeepSeek-V4 B200 | cookbook's own justification |
|---|---|---|---|
| low-latency (C1-C16) | 5-1-6 | 3-1-4 | — |
| balanced (C64-C256) | 2-1-3 | 1-1-2 | *"at this concurrency the verify overhead of a long draft outweighs the accept-length gain"* |
| high-throughput (C1024+) | **off** | **off** | *"at saturation the verify step costs more than it saves"* |

`[inferred]` DSpark's contribution is to make that static ladder *dynamic* — a load-adaptive
verification length driven by a block-accept estimator — and the published payoff is
**+60-85% per-user tok/s at matched throughput**. `[verified]` Our fork already contains
`speculative/dspark_components/` including a 825-line block-accept estimator with online
logging; what it does not have is that estimator wired to a scheduler on GLM-5.2's EAGLE path.
**That is the one speculative change in the corpus that is model-agnostic, attacks the 4.7x
C1->C16 falloff directly, and sidesteps the `eagle_worker_v2` IMA entirely** — it does not
deepen the draft tree, it varies how much of a fixed tree gets verified.

**The measured value of speculation on this box, for calibration** `[verified]` from the
ledger's "Tried" table: off -> 2-1-3 is **2.50x** (144.8 -> 362.4 tok/s); 2-1-3 -> 3-1-4 is
**1.23x** (362.4 -> 446.8). Compounded **3.09x**. That is the largest single measured win in
the entire corpus, on any model, and it is the reason the verify step is a shared kernel worth
hand-optimizing (§2.5).

### 1.5 FP8-native with NVFP4 appearing — REFUTED as stated

The clause describes 2025. In 2026 the frontier is native 4-bit and the interesting axis has
moved.

| model | shipped format | is 4-bit *native*? | NVFP4 build | net effect of NVFP4 |
|---|---|---|---|---|
| GLM-5.2 | FP8 E4M3 block 128x128 (Z.ai) | no | `nvidia/GLM-5.2-NVFP4`, experts only | 755.62 -> 464.80 GB. **A real win** — 89.37 -> 56.72 GiB/GPU at TP8 |
| DeepSeek-V3.2 | FP8 block-128, `scale_fmt: ue8m0` | no | — | — |
| DeepSeek-V4 | **`expert_dtype: fp4` is a config key**; MXFP4 group-32 (I8 packed + F8_E8M0 scales) + FP8 attention | **yes** | `nvidia/DeepSeek-V4-*-NVFP4` | **1% faster at C1** (TPOT 2.88 vs 2.91 ms Flash; 4.19 vs 4.25 Pro). Marginal |
| Kimi K3 | **MXFP4, QAT'd from the SFT stage onward**, group-32 E8M0 | **yes** | `nvidia/Kimi-K3-NVFP4` | **+30 GiB — a regression.** Group-16 E4M3 costs 0.5625 B/param against MXFP4's 0.53125, which on 2.72 T expert params is **+85 GB**; the FP8 attention only claws back 52.7 GB |
| Qwen3.8-2.4T | BF16 official **and** FP8 official (2,396.6 B of 2,446.2 B quantized) | no | `Inferact/...-NVFP4` (community, used by vLLM's own recipe) | 2,496 -> 1,447 GB, but only **17% faster at C1** because 74% of C1 bytes are BF16 either way |
| Qwen3.5-397B | FP8 official, **400.7 B of 403.4 B quantized** | no | `nvidia/...-NVFP4`, MoE only | **26% SLOWER than FP8** (20.5 vs 16.3 GB/token) because the official FP8 quantizes everything and NVFP4 quantizes only the MoE |

`[inferred]` **Two of six models get *worse* under NVFP4 and a third gets 1%.** The clause is
refuted. Bits per parameter for the record `[verified]`:

```
BF16                      = 16.000 bit/param
FP8 block 128x128         =  8.002 bit/param   (scale overhead 0.002)
MXFP4 group-32 E8M0       =  4.250 bit/param   (0.53125 B/param)
NVFP4 group-16 E4M3       =  4.500 bit/param   (0.5625 B/param)
```

**The axis that is live is the exclusion list, and it is where 74-82% of C1 decode bytes go.**
`[verified]` from the four checkpoints' `ignore` / `modules_to_not_convert` / `exclude_modules`:

| checkpoint | quantized | left alone | left alone at |
|---|---|---|---|
| `nvidia/GLM-5.2-NVFP4` | `layers.{3..77}.mlp.experts.*` **only** (156 exclude entries) | all attention, all shared experts, 3 dense layers, whole MTP layer, lm_head, embed | **BF16** |
| Kimi K3 (native MXFP4) | routed experts only | `self_attn` (KDA+MLA), `shared_experts`, dense MLP, `lm_head`, vision tower | **BF16** |
| `Inferact/Qwen3.8-2.4T-NVFP4` | experts | all self-attention, all Gated DeltaNet, embed, norm, lm_head, MTP layers | **BF16** |
| `deepseek-ai/DeepSeek-V4-Pro-0813` | routed experts (MXFP4 1x32) | attention, shared expert, indexer -> **FP8**; compressor, router, mHC, embed, lm_head -> BF16/F32 | **FP8 mostly** |

And the resulting C1 decode byte composition `[inferred]`:

| checkpoint | 4-bit share | FP8 share | **BF16 share** |
|---|---:|---:|---:|
| GLM-5.2 NVFP4 as shipped, TP8 | 18.2% | 0% | **81.8%** |
| Kimi K3 MXFP4, TP8, all sharded | 19.3% | 0% | **~80.7%** |
| Qwen3.8-2.4T NVFP4, TP8 | 25.7% | 0% | **74.3%** |
| DeepSeek-V4-Pro-0813, TP8 | 32.4% | 57.8% | **9.8%** |

`[inferred]` **DeepSeek is the only lab in the set that finished the job**, and the gap it buys
is enormous: V4-Pro reads **4,960 MB/rank/token against GLM-5.2 NVFP4's 8,756** despite having
**2.1x the parameters and 1.2x the activated parameters**. §4.

**But the fix generalizes only partly, and for a principled reason** `[inferred]`:

- **GLM-5.2: take it.** The BF16 tail is `fused_qkv_a_proj_with_mqa` (2,515 MB/rank/token,
  28.7%), `o_proj` (1,963 MB), the shared expert (708 MB), `q_b_proj` (654 MB), the indexers
  (394 MB), `kv_b_proj` (286 MB). These are plain softmax-attention projections and dense
  FFNs — exactly the tensors DeepSeek keeps in FP8 with no reported loss. Requantizing takes
  8,756 -> **6,048 MB (roofline 1,268 tok/s) or 5,293 MB (1,449 tok/s)**, depending on which of
  two mutually inconsistent figures in `01-glm-5.2` §3.2 you take — see §4.1 correction 4.
  `[verified]` SGLang already has the escape hatch, `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN`, and it
  covers **only** `q_b_proj` (654 of 6,463 fixable MB). Widening it is a loader change, not a
  kernel.
- **Qwen: worth ~12%, no more.** Of the 41.78 B BF16-active params, **30.10 B is Gated
  DeltaNet** and must stay high precision (`mamba_ssm_dtype: float32` is the checkpoint saying
  so). Only the 9.65 B of gated attention plus the 2.03 B lm_head are safely fixable:
  112.4 -> 100.7 GB node-wide, 546 -> **609 tok/s** at TP8. Modest, and the right answer.
- **Kimi K3: a real quality risk.** The BF16 tail is 61.2 GB of KDA linears and 24.3 GB of
  shared experts. Taking the KDA/MLA linears to FP8 would move the TP8 step from 16.56 to
  ~12.30 GB/rank and the roofline from 463 to ~**624 tok/s** — but a delta-rule recurrence
  compounds error across 69 layers, and `nvidia/Kimi-K3-NVFP4` does exactly this and is under
  active accuracy debugging (SGLang PR #35077, draft, *"Correct Kimi-K3 ModelOpt NVFP4 accuracy
  with TRT-LLM MoE"*). Do not assume it is free.

`[verified]` One shared quantization fact worth pinning because the ecosystem standardized on
it: **`scale_fmt: "ue8m0"` appears in GLM-5.2's FP8 build and every DeepSeek config since
V3.2**, and DeepGEMM's SM100 path requires packed UE8M0 (4 per `torch.int`) where SM90 required
FP32 scales. Any FP8 GEMM adopted here must land on the packed-UE8M0 path.

### 1.6 Long context in a cheap-attention regime — HOLDS on context, ZERO on position

| model | native context | extension | KV per 1 M tokens per GPU |
|---|---:|---|---:|
| GLM-5.2 | **1,048,576** | none needed | 61.46 GB (as shipped) / 54.07 GB (indexer fix) |
| DeepSeek-V3.2 | 163,840 | YaRN x40 | 48.07 GB |
| DeepSeek-V4-Pro | **1,048,576** | YaRN x16 | **5.03 GB** |
| DeepSeek-V4-Flash | **1,048,576** | YaRN x16 | 3.51 GB |
| Kimi K3 | **1,048,576** | none (NoPE) | 13.50 GB (24 layers) + 116.2 MB/GPU/request state |
| Qwen3.8-2.4T | 262,144 | **static YaRN x4 -> 1,010,000** | 47.10 GB + 574 MiB/seq state |

`[verified]` Three of four ship 1 M natively. Qwen is the outlier and its extension is *static*
scaling that applies at all lengths and degrades short-context quality — both Qwen model cards
warn about this explicitly. `[inferred]` **Do not enable it on a general endpoint**, which
means Qwen's real serving context is 262 K unless you run a second replica.

**Positional encoding has four mutually incompatible answers and no common parameterization:**

| model | scheme `[verified]` |
|---|---|
| GLM-5.2 | plain RoPE, `rope_theta: 8000000`, `rope_type: "default"`, **no YaRN block at all**, `rope_interleave: true` (so `is_neox_style=False`) — **and the indexer's RoPE needs a non-interleaved layout while MLA's needs interleaved** |
| DeepSeek-V4 | **three rotary configs in one model**: base `rope_theta: 10000` + YaRN x16 for main layers, `compress_rope_theta: 160000` for the indexer, YaRN **disabled** on pure-SWA layers, plus RoPE at position `-i` applied to the core-attention *output* |
| Kimi K3 | **no RoPE at all** — `mla_use_nope: true`, no `rope_theta` key, no `rope_scaling`. And `qk_rope_head_dim: 64` still occupies head width; whether those 64 dims are live is `[unverified]` |
| Qwen3.5/3.6/3.8 | `partial_rotary_factor: 0.25` (64 of 256 dims rotated), `rope_theta: 10000000`, mRoPE `mrope_section: [11,11,10]` interleaved, optional static YaRN `factor: 4.0` |

`[inferred]` A shared RoPE kernel must be parameterized over: interleaved vs neox, partial vs
full rotation fraction, theta, YaRN on/off/per-layer, mRoPE sectioning, and application to
input vs output. **This is the single highest-risk area of per-model code in the whole stack,
because every failure mode is silent.** The corpus contains one published instance:
`[verified]` the DeepSeek-V3.2-Exp README (2025-11-17) records that the indexer's RoPE input
requires non-interleaved while MLA's expects interleaved, and getting it wrong *degrades
quality without crashing*. `[inferred]` If our GLM-5.2 DSA port was written by analogy to
pre-November-2025 reference code, this is a 20-minute check that nobody has done.

`[verified]` A second silent trap worth stating because it also generalizes: GLM-5.2's
`config.json` declares `head_dim: 192` while `qk_head_dim` and `v_head_dim` are both 256. A
generic consumer computing `num_attention_heads * head_dim = 12288` is wrong by 1.33x; the true
`o_proj` input is 16384. GLM-5/5.1 declare `head_dim: 64` for the same architecture. **The
field is not maintained anywhere in this family; a shared config reader must refuse to read
it.**

### 1.7 Scorecard: what a serving stack can actually assume

`[inferred]` Assumptions that hold for all four families, and are therefore safe to build on:

1. **MoE with an always-on shared expert and aux-loss-free bias-corrected top-k routing.**
2. **Activated/total between 3% and 6%**, so 94-97% of weights are idle per token.
3. **No group-limited routing** — a token's experts can reach any rank.
4. **TP does not shard the KV cache**, so DP-attention is KV-free and sometimes KV-saving.
5. **A blockscaled 4-bit or FP8 grouped expert GEMM at `M <= 8` per expert** at every
   concurrency this box can hold (§3.4).
6. **Speculation ships with the model, at `eagle_topk = 1`, with a low/balanced/off ladder.**
7. **1 M-class context is the design point** (Qwen at 262 K native is the exception).
8. **The `ignore` list is where the bytes are.** Never trust a checkpoint's precision label.

Assumptions that do **not** hold and must be parameterized:

1. The attention mechanism (two disjoint families, no shared kernel).
2. Whether a per-sequence recurrent state pool exists.
3. Whether `tokens -> KV blocks` is a fixed ratio (DeepSeek say V4 *"violates fundamental
   assumptions behind PagedAttention"*).
4. Positional encoding, in every dimension.
5. Expert granularity, expert width relative to hidden, and shared-expert count.
6. Where the draft model comes from and what it reads.

---

## 2. The five kernels worth hand-optimizing once

Ranked by (models served) x (measured cost). "Measured" means measured on this box for
GLM-5.2; for the other models the share is derived from the byte accounting in their documents
and is `[inferred]`.

### 2.1 Grouped blockscaled expert GEMM — serves 4/4, top-3 by cost on every one

**Who it serves and at what shape.** `[verified]` from the configs; `N` is per rank.

| model | format | K | N/rank | experts/token | fused shared? |
|---|---|---:|---:|---:|---|
| GLM-5.2 | NVFP4 g16 (E4M3 plane + F32 global + F32 input scale) | 6144 | 2048/8 = **256** | 8 routed + 1 shared | **yes -> 257 experts** under plain TP8 |
| DeepSeek-V4-Pro | MXFP4 g32 (I8 packed + F8_E8M0) | 7168 | 3072/8 = **384** | 6 + 1 | no |
| DeepSeek-V4-Flash | MXFP4 g32 | 4096 | 2048/4 = **512** (TP4) | 6 + 1 | no |
| Kimi K3 | MXFP4 g32 | **3584** (latent!) | 3072/8 = **384** | 16 + 2 | no; **shared replicated at tp1 under EP** |
| Qwen3.8-2.4T | NVFP4 g16 or FP8 block-128x128 | 8192 | 2048/8 = **256** | 10 + 1 | `[unverified]` |
| Qwen3.5-397B | FP8 block-128x128 | 4096 | 1024/8 = **128** | 10 + 1 | `[unverified]` |

`[inferred]` The template space is **`M in 1..8`, `N in {128, 256, 384, 512}`,
`K in {3584, 4096, 6144, 7168, 8192}`, format in {NVFP4 g16, MXFP4 g32, FP8 block-128x128}**.
That is one kernel family with roughly 60 live instantiations — entirely tractable. Note
`M <= 8` is not an assumption about C1: §3.4 shows the mean tokens-per-resident-expert stays
below 8 even at C256 on every model.

**What it is worth.**

| model | share of decode cost |
|---|---|
| GLM-5.2 | **19.4% of C1, 30.8% of C64** GPU time `[verified]`, nsys, all 8 ranks. Two `Bmm_E2m1_*` cubins are 23.5% of C64 GPU time |
| DeepSeek-V4-Pro | **32.4%** of active decode bytes `[inferred]` |
| Kimi K3 | **19.3%** of the TP8 C1 step `[inferred]` |
| Qwen3.8/3.5 | **25.7%** of C1 bytes `[inferred]`; and SGLang PR #34795 measured tuned-vs-heuristic Triton MoE at **1.59x at M=1, 1.87x at M=16, 1.84x at M=64, 1.79x at M=256** on H20, end-to-end +16.1% at BS1 / +40.1% at BS16 `[reported]`. **No B200 config exists upstream for `E=512,N=256`** |

`[inferred]` **It is the only kernel that is top-3 by cost on every model in the set, and the
only one with a published tuning headroom in the 1.6-1.9x range.** That makes it the
unambiguous #1.

**What exists.** `[verified]` The incumbent on this box is a trtllm-gen cubin delivered via
FlashInfer:
`Bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_..._swiGlu_dynB_sm100f`
— tile 128x8x512, 5-stage, MMA 128x8x64, fused SwiGLU, dynamic batch. 394 `sm100f` cubins sit
in `~/.cache/flashinfer/cubins/`. Alternatives: DeepGEMM's `sm100_fp8_fp4_gemm_1d1d` and
`sm100_fp8_fp4_mega_moe` (MIT, already installed as `sgl-deep-gemm`, **no published SM100
performance table**), CUTLASS examples 72 / 75 / 92. `[unverified]` **Nobody, anywhere, has
published a fraction-of-B200-NVFP4-peak for any implementation.** That measurement does not
exist and would be worth having.

**The trap.** `[verified]` SM90 has both `fp8_gemm_1d1d` and `fp8_gemm_1d2d`; SM100 has only
`fp8_fp4_gemm_1d1d`. If the stack assumes 1x128 activation scaling against 128x128 weight
scaling on Blackwell, check which path it lands on.

### 2.2 Fused allreduce + residual-add + RMSNorm + quantize — serves 4/4, and never amortizes

**Who it serves.** All four, at every layer, unconditionally. It is the only kernel in the set
with no model-dependent gate.

**What it is worth.** `[verified]` on this box: collectives are **19.6% of a 2.74 ms TPOT at
C1** and **25.4% at C64**, and `observed 14,097 ms = transfer 7,505 ms + waiting 6,599 ms` —
i.e. **47% of collective time is rank-arrival skew**, mean 9.2 us, max 4,897 us, rank 0 last in
24% of 114,171 instances. 47% of 537 us is **252 us/token of pure waiting = 9.2% of TPOT.**

`[inferred]` **Two different projects hide inside this one line item.** Fusion attacks the 53%
that is transfer and per-call fixed cost — it collapses two collectives per layer into one by
deferring the MLP all-reduce into the next layer's norm, which is what
`should_fuse_mlp_allreduce_with_next_layer` already does and what
`tllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` already is. **Overlap** (TBO/SBO,
tile-granularity scheduling, or removing the barrier entirely via Mega-MoE) attacks the 47%
that is skew. **Neither addresses the other.** §3.2 shows why this matters more than any other
kernel in the set: it is the only cost that does not shrink with quantization, sparsity or
batch.

**What exists.** `[verified]` **Five** allreduce implementations are already in our tree:
FlashInfer's `trtllm_mnnvl_fused_allreduce_add_rmsnorm{,_quant}` (with an `AllReduceFusionOp`
covering RESIDUAL_RMS_NORM plus FP8/NVFP4/per-token-group-blockwise quant),
`trtllm_custom_all_reduce`, torch symmetric memory (multimem / two-shot),
`custom_all_reduce_v2` (three algorithms x three pull sources, and per its docstring the most
carefully built for CUDA-graph capture), and MSCCL++ at `sglang-v0.9.1`. `[inferred]` **A
four-way bake-off at our exact 8-16 KB message sizes is configuration, not code**, and it is
the cheapest experiment in this document.

**The blocking contradiction.** `[verified]` The ledger states allreduce fusion was **off** in
every measurement, yet `oneshotAllreduceFusionKernel` is 783 ms / 8.2% and
`twoshotAllreduceKernel` is 407 ms / 4.3% of GPU time. `[inferred]` The MNNVL *transport* is
active independently of the *fusion* flag. **Resolve this before interpreting any collective
number, on any model.** Also `[verified]`: AllReduce is the **only** top-10 kernel at 0% CUDA
graph capture while AllGather and ReduceScatter are at 100% — which is either the whole
explanation or a large part of it.

### 2.3 The MTP/draft verify step — serves 4/4, largest measured win in the corpus

**Who it serves.** All four, including K3 — the *draft* differs (in-checkpoint MTP layer vs
3-block DSpark vs separate 2.25 B model) but the **verify** is the same operation everywhere:
run the target forward at `M = num_draft_tokens`, compare, accept a prefix, roll back.

**What it is worth.** `[verified]` on this box: **3.09x compounded** (2.50x from off to 2-1-3,
1.23x from 2-1-3 to 3-1-4). `[reported]` elsewhere: DeepSeek-V3 1.8x TPS at 85-90% second-token
acceptance; Qwen ~96.8% draft acceptance after vLLM PR #52013 and *"roughly 2.3x on per-user
output rate"* at MTP-3; K3 111 -> 331 tok/s at TP8 (3.0x) and 118 -> 370 at TP16 (3.14x) with
DSPARK.

**The shared mechanism nobody states plainly.** `[verified]` DeepSeek: *"by predicting multiple
tokens per step, MTP increases the inference batch size, which is crucial for boosting EP
computational intensity and hardware utilization."* `[inferred]` **Speculation and
DP-attention are the same lever — both raise `s_q`, and `h_q * s_q` is what decides whether MLA
decode is compute-bound or memory-bound.** With the crossover at
`h_q * s_q >= 0.5 * 2250/7.672 = 147`:

| configuration | `h_q`/rank | `s_q` needed | reachable? |
|---|---:|---:|---|
| GLM-5.2 TP8 | 8 | **18** | No — 3-1-4 gives `s_q ~ 4` |
| GLM-5.2 DP8 | 64 | **2.3** | **Yes, already** |
| DeepSeek-V3.2 TP8 | 16 | 9 | No |
| DeepSeek-V3.2 DP8 | 128 | 1.15 | Yes, even without speculation |
| Kimi K3 MLA layers TP8 | 12 | 12.25 | No, just barely |
| Kimi K3 MLA layers DP8 | 96 | 1.53 | Yes |

`[verified]` DeepSeek say the consequence outright: *"we don't use Tensor Parallel for decoding
instances, meaning `h_q` is 128 and the kernel is compute-bound."* `[inferred]` **All three MLA
models sit on the memory-bound side of that line under TP8 and on the compute-bound side under
DP8, and DP costs nothing in KV memory (§1.2d).** One argument, three models.

**Where per-model code is unavoidable.** The draft *forward* — GLM-5.2's MTP layer is a
complete decoder layer with its own indexer that reuses the target's top-k
(`index_share_for_mtp_iteration: true`); V4's DSpark chain is fed once from target layers
[58,59,60]; K3's DSPARK reads [7,23,51,67,83] and runs GQA. §5.3.

**The trap that generalizes.** `[verified]` Three separate published bugs, one pattern: the
draft head inheriting or losing the wrong quantization. vLLM #52013 — a dedicated `mtp.lm_head`
silently remapped to a nonexistent module, so the draft head was **never loaded** and appeared
only as an "unexpected weight" warning. SGLang #34622 — "Prevent Qwen3.5 MTP draft from
inheriting GPTQ quantization." And on GLM-5.2 `[verified]`: the NVFP4 build leaves layer 78
entirely BF16, so the *target* MoE runs an NVFP4 kernel while the *draft* MoE runs
`trtllm_bf16_moe` — which is the exact kernel in SGLang issue #32377's stack trace, and a
clean mechanistic story for why 3-1-4 survives and 5-1-6 does not. **Any shared verify path
must assert that the draft's dtype is what the loader intended, and log it.**

### 2.4 MQA-shaped decode attention against a <=576-wide latent — serves 3/4

**Who it serves.** GLM-5.2, DeepSeek V3.2, DeepSeek V4 (at 512), Kimi K3's 24 MLA layers.
**Not Qwen** (gated GQA).

**The template space** `[verified]`: `latent in {512, 576}`, `q heads in {64, 96, 128}`,
`qk_nope in {128, 192, 448}`, `v_head_dim in {128, 256, 512}`, plus an optional output gate
(K3) and optional NoPE (K3). Page size: **64 asserted** for GLM-5.2's DSA pool on CUDA, **256**
shipped by DeepSeek for V4.

**What it is worth — and this is where a single-model view misleads.**

| model | attention share |
|---|---|
| GLM-5.2 | **10.9% of C1 but only 4.6% of C64** GPU time `[verified]`. The ledger explicitly deprioritizes it: *"It amortises across the batch. Optimizing it would be chasing a shrinking target."* |
| DeepSeek-V4-Pro | **46% of active decode bytes — more than the routed experts (32%)** `[inferred]`. `wq_b` is 1536x65,536 because 128 heads x 512 head_dim |

`[inferred]` **Do not generalize GLM-5.2's 10.9% to V4.** V4 did not just move cost out of the
KV cache, it moved cost *into the attention weights*, and a model whose attention weights
dominate its decode bandwidth is a different optimization target from every MoE model profiled
here.

**What exists, and the tile-shape argument for DP-attention.** `[verified]` FlashMLA's SM100
tree has exactly **one** native decode kernel, `csrc/sm100/decode/head64/kernel.cuh`, and
`csrc/sm100/decode/head128/` contains a README and nothing else, which states you must either
borrow from the prefill small-topk family (which only instantiates `k_dim = 512`, i.e. V4's
geometry, not V3.2's 576) or run head64 twice. `[inferred]` **GLM-5.2 has 64 query heads and a
576 latent, so under DP-attention `h_q = 64` is an exact match for the native SM100 kernel
with `ModelType::V32`; under TP8 `h_q = 8` and you pad to 64, wasting 87.5% of the tile.**
That is a third independent argument for DP-attention, alongside the compute-bound crossover
and the KV-memory argument. Three mechanisms, one conclusion.

Also on the shelf and unexploited `[verified]`: `tokenspeed_mla==0.1.8` is a **pinned
dependency already installed** ("speed-of-light MLA kernels for Blackwell SM100/SM103", MIT,
upstream at 0.2.5), with a `tokenspeed_mla_backend.py` in-tree gated to fire only for Kimi-K3
DCP. `06-open-source-kernels` ranks bumping the pin and forcing the backend as the highest
value-per-minute item in the whole shelf, at 0.5-1 engineer-day. `[unverified]` whether it
handles `v_head_dim=256`.

### 2.5 The top-k sparse indexer (MQA logits) — serves 2/4, and already written

**Who it serves.** GLM-5.2 (32 heads x 128, top-2048, 21 of 78 layers), DeepSeek-V3.2
(64 x 128, top-2048, all 61 layers), DeepSeek-V4 (64 x 128, top-1024 Pro / top-512 Flash,
30 of 61 / 21 of 43 layers, scored against **compressed** keys). Not K3, not Qwen.

**What it is worth.** `[verified]` GLM-5.2: **5.8% of C1, 2.4% of C64**. Small and shrinking,
and it already only runs on 22 of 79 layers.

**Why it is on this list anyway: the kernel exists and is a drop-in.** `[verified]`
`04-deepseek` read DeepGEMM's `sm100_mqa_logits.cuh` (593 lines) in full. It is a **unified**
implementation: `kIsFP4` dispatches `SM100_MMA_MXF4_SS` vs `SM100_MMA_MXF8F6F4_SS`,
`kNumQKBytesPerToken = kIsFP4 ? kHeadDim/2 : kHeadDim`, a static assert accepting
`kHeadDim == 128` (GLM-5.2's exact `index_head_dim`), a contiguous-KV entry *and* a paged entry
(`sm100_paged_mqa_logits` taking a `block_table`). MIT, SM100-native, FP4-capable, paged.
`[inferred]` **Take it for the cache-bandwidth halving (132 -> 68 B/token/layer), not for the
FLOPs.** `[verified]` Our fork already has `--enable-deepseek-v4-fp4-indexer` wired for V4;
the equivalent for GLM-5.2's DSA does not exist. And `[reported]` a second one-line win:
quantizing the index scores from FP32 to BF16 before top-k is published at **2x top-k selector
speed at 99.7% recall**.

`[inferred]` **The most interesting thing about this kernel is what it reveals about
amortization strategy.** GLM-5.2 amortizes indexer cost **temporally** — `index_topk_freq=4`,
one indexer serving 4 consecutive layers, 21 of 78 layers carrying weights. DeepSeek amortizes
**spatially** — the indexer runs on every CSA layer but against `seq/4` compressed keys, on
only half the layers. `[unverified]` **`04-deepseek` found no DeepSeek publication that shares
a computed top-k index set across layers**, so GLM-5.2's temporal sharing appears to be a Zhipu
invention with no DeepSeek analogue. The two are orthogonal and compose — but spatial sharing
needs retraining and is not available to us.

`[verified]` **And the shared allocator bug lives here.** GLM-5.2's `DSATokenToKVPool`
allocates one indexer buffer per layer `for _ in range(self.layer_num)` — 78 buffers — while
only 22 layers carry indexer weights, and SGLang's own comment says *"shared layers' cache is
never read, so filling it is dead work."* Cost: 7,392 B/token = **7.39 GB per 1 M tokens per
GPU**, 12% of the entire FP8-KV per-token cost, 59 GB aggregate at a 1 M cache. DeepSeek-V4
allocates the indexer cache **only on CSA layers** — 30 of 61 — and does not make this mistake.
§5.3 item 1.

### 2.6 The kernel that is NOT shared: linear-attention decode

`[verified]` Serves Kimi K3 (69 of 93 layers, KDA) and Qwen3.5/3.6/3.8 (69 of 92 on the 2.4 T,
45 of 60 on the 397 B, Gated DeltaNet). Serves **neither GLM-5.2 nor DeepSeek**, at all.

`[inferred]` KDA and GDN are siblings — both 128-dim heads, `conv_kernel 4`, gated, and SGLang
files `kda_*.py` and `gdn_*.py` side by side — so **one** investment covers both families, with
two instantiations for the head geometry (K3: 96 key heads x 128 = 96 value heads x 128,
symmetric; Qwen: 16 key : 128 value, an **8:1 ratio** that TensorRT-LLM PR #17700 is
specifically tuning for `[reported]`).

`[inferred]` **The honest recommendation is not to build it.** It serves half the set, neither
half is our target, and the corpus contains a specific warning about the outcome of exactly
this kind of effort: `k3-kernels/` built and passed 40 correctness checks and 18 spill-free
instantiations on sm_120, and the one head-to-head against production came out **1.02x — a
wash** against SGLang's Triton kernel. `[verified]` Meanwhile the *free* wins in this space are
large and elsewhere: vLLM's dedicated K3 KDA metadata builder took **870 us -> 34 us at bs=1, a
96% reduction** `[reported]` — a host-side change, not a kernel.

`[verified]` One hardware fact to record: SGLang's `ptx_kda` prefill backend is gated
`== (10, 3)`, i.e. **GB300 only — this box reports 10.0** and takes the CuTe-DSL path.
`[reported]` The local kernel map reads this as proof that 10.3 has instructions 10.0 lacks.
Long-context TTFT for K3 on B200 will be worse than on GB300 for this reason.

### 2.7 Summary: the ranked shared-kernel table

| # | kernel | models | measured share (GLM-5.2, this box) | what it is worth | status |
|---|---|:---:|---|---|---|
| 1 | **grouped blockscaled expert GEMM** | 4/4 | 19.4% C1 / **30.8% C64** | only top-3 kernel on every model; 1.6-1.9x tuning headroom published on H20, **nothing tuned for B200** | incumbent is a trtllm-gen cubin; no fraction-of-peak known anywhere |
| 2 | **fused allreduce + residual + norm + quant** | 4/4 | 19.6% C1 / 25.4% C64, **47% of it skew** | the only cost that does not amortize; ratio to HBM floor rises 0.47 -> 1.55 across the set | **five** implementations in-tree, unmeasured; a blocking contradiction (§2.2) |
| 3 | **MTP verify step** | 4/4 | not separately profiled | **3.09x measured**, the largest win in the corpus; DSpark-style adaptive verification is +60-85% `[reported]` | draft-shape ladder converged; adaptive scheduler not wired |
| 4 | **MQA latent decode attention** | 3/4 | 10.9% C1 / 4.6% C64 | shrinking on GLM-5.2, **46% of V4-Pro's bytes** — do not generalize | native SM100 head64 kernel exists and matches GLM-5.2 **only under DP-attention** |
| 5 | **top-k indexer (MQA logits)** | 2/4 | 5.8% C1 / 2.4% C64 | halves indexer cache bytes at FP4; 2x top-k selector at BF16 scores | **already written**: DeepGEMM `sm100_mqa_logits.cuh`, MIT, SM100, `kHeadDim==128`, paged, FP4 |
| — | linear-attention (KDA/GDN) decode | 2/4, **neither ours** | n/a | do not build; the one head-to-head was a 1.02x wash | separate suite already in-tree |

---

## 3. Parallelism: which strategy generalizes, and the collective arithmetic

### 3.1 Collective count and payload, per model

`[verified]` from layer counts and hidden sizes. Under pure TP with no DP-attention, each
decoder layer needs one all-reduce after the attention output projection and one after the MoE
down-projection. Qwen's hybrid layers each need one too, on top of the MoE reduce.

| model | layers | **collectives / decoded token** | payload BF16 at M=1 | total payload/token |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 43 | **86** | 4096 x 2 = 8,192 B | 0.70 MB |
| Qwen3.5-397B-A17B | 60 (15 GQA + 45 GDN + 60 MoE) | **120** | 4096 x 2 = 8,192 B | 0.98 MB |
| DeepSeek-V4-Pro | 61 | **122** | 7168 x 2 = 14,336 B | 1.75 MB |
| **GLM-5.2** | 78 | **156** | 6144 x 2 = 12,288 B | **1.92 MB** |
| Qwen3.8-2.4T-A95B | 92 (23 GQA + 69 GDN + 92 MoE) | **184** | 8192 x 2 = 16,384 B | 3.01 MB |
| Kimi K3 | 93 | **186** | 7168 x 2 = 14,336 B | 2.67 MB |

`[inferred]` **Every payload is 8-16 KB — pure fixed-overhead territory, far below any NVLink5
message-size knee.** The bandwidth term is arithmetically negligible: for GLM-5.2, a ring
all-reduce moves `2(8-1)/8 x 1.92 MB = 3.36 MB`, which at 0.9 TB/s per GPU unidirectional is
**3.7 us per token** — about **1%** of the observed 537 us of collective time.

`[verified]` So the cost is per-call fixed overhead plus arrival skew, and this box measures it
at:

```
observed collective share at C1 = 19.6% of 2.74 ms          = 537 us / token
                                / 156 collectives           = 3.44 us each
of which 47% is rank-arrival skew                           = 252 us = 9.2% of TPOT
```

`[inferred]` Applying that same 3.44 us to the other models (TP8, one NVLink domain, same
communicator):

| model | collectives | **collective latency / token** |
|---|---:|---:|
| DeepSeek-V4-Flash | 86 | 296 us |
| Qwen3.5-397B | 120 | 413 us |
| DeepSeek-V4-Pro | 122 | 420 us |
| GLM-5.2 | 156 | **537 us** |
| Qwen3.8-2.4T | 184 | 633 us |
| Kimi K3 | 186 | 640 us |

`[unverified]` The 3.44 us constant is back-derived from one measurement on one model at TP8.
It is the most load-bearing unmeasured number in this document, and **three of the five input
documents rest on it.** Measuring an 8-16 KB all-reduce latency curve at TP8, with and without
`--enable-symm-mem`, with and without `--disable-custom-all-reduce`, separating fixed cost from
skew, is the single cheapest high-value experiment available. At TP4 (V4-Flash's shape) the
constant should be lower; the 296 us above is therefore an upper bound.

### 3.2 The ratio that decides everything

`[inferred]` Put the collective latency next to the HBM weight-read floor. On this box, with
TBO, SBO and allreduce fusion all disabled, these are serialized, so the ratio is the honest
model of the current configuration — and it is also exactly the measure of what overlap is
worth.

| configuration | collectives | coll. latency | HBM floor | **ratio** |
|---|---:|---:|---:|---:|
| Kimi K3, TP8 (does not fit) | 186 | 640 us | 2,158 us | **0.30** |
| Qwen3.8-2.4T NVFP4, TP8 (does not fit) | 184 | 633 us | 1,831 us | **0.35** |
| **GLM-5.2 NVFP4 as shipped, TP8** | 156 | 537 us | 1,141 us | **0.47** |
| **GLM-5.2 FP8, TP8** | 156 | 537 us | 870 us | **0.62** |
| DeepSeek-V4-Pro FP4, TP8 | 122 | 420 us | 647 us | **0.65** |
| **GLM-5.2, non-expert GEMMs at FP8, TP8** | 156 | 537 us | 690-788 us | **0.68-0.78** |
| DeepSeek-V4-Flash FP4, TP4 | 86 | 296 us | 369 us | **0.80** |
| **Qwen3.5-397B-A17B FP8, TP8** | 120 | 413 us | 266 us | **1.55** |
| Kimi K3, TP16 across 2 nodes (@20 us inter-node) | 186 | 3,720 us | 1,079 us | **3.45** |
| Qwen3.8-2.4T, TP16 across 2 nodes (@20 us) | 184 | 3,680 us | 916 us | **4.02** |

**Three conclusions, and the first is the most important thing in this document.**

**1. `[inferred]` Every improvement that shrinks the byte term makes the collective problem
relatively worse, because the collective term does not move.** Read the GLM-5.2 rows in
sequence: 0.47 as shipped, 0.62 at FP8, **0.68-0.78** if you requantize the non-expert GEMMs.
The requant is worth +45% to +65% on the roofline (876 -> 1,268-1,449 tok/s) and it takes
collectives from 47% to roughly 70-78% of the byte floor. **The requant and the collective work
are the same project.** Doing the requant alone will produce a disappointing end-to-end number
and the disappointment will be misattributed.

**2. `[inferred]` Qwen3.5-397B-A17B is already past 1.0 — its collectives cost more than its
entire weight read.** A serialized floor of `266 + 413 = 679 us -> 1,473 tok/s`, against a
pure-bandwidth 3,765. This is what a well-quantized, genuinely sparse, hybrid-attention model
looks like on 8 ranks: **a latency problem in the interconnect, not the memory system.** It is
also where GLM-5.2 is heading.

**3. `[inferred]` Crossing a node boundary at TP16 is catastrophic and both non-fitting models
hit it.** 186 collectives x ~20 us of inter-node all-reduce is 3.72 ms against a 1.08 ms HBM
floor. `[reported]` The direct evidence: vLLM's own K3 numbers go 111 tok/s at TP8 to 118 at
TP16 — **+6% for 2x the bandwidth.** And by our consistent arithmetic, K3's bandwidth
utilization *halves* from 24.0% at TP8 to 12.7% at TP16, which is precisely the signature of a
collective-count-bound workload. `[verified]` This is why SGLang's B200 long-context K3 cell is
`--tp-size 8 --pp-size 2`, not TP16: PP crosses the network **once per token** with a 14 KB
activation instead of 186 times. And it is why `--pp-size 2` costs you DSPARK, which
`[verified]` requires `pp_size == 1`.

### 3.3 TP vs EP at C1 — the arithmetic

`[inferred]` Per decoder layer:

| | pure TP8 | EP8 (experts whole, attention still TP8) |
|---|---|---|
| collectives / layer | **2** (o_proj AR, MoE AR) | **3** (o_proj AR, dispatch a2a, combine a2a) |
| MoE weight bytes / rank | `(topk+1) x 3 x hidden x moe_inter / 8` — every rank reads 1/8 of every touched expert | whole experts on whichever ranks own them |
| shared-expert fusion | **on** (GLM-5.2 runs 257 experts) | **off, silently** |
| K3-specific | shared experts sharded, 3.04 GB/rank | shared experts **replicated at tp1**, 24.31 GB/rank |
| rank coverage at C1 | all 8 ranks work | at most `topk` ranks work; **V4's top-6 guarantees >= 2 idle ranks, always** |

**The a2a bandwidth cost at C1, using DeepEP V2's own first-party SM100 EP8 NVLink measurement
of 726 GB/s dispatch** `[reported]`, with dispatch FP8 (1 B) and combine BF16 (2 B):

```
per layer = 3 B x M x (topk + 1) x expert_input_width / 726e9

GLM-5.2      : 3 x 1 x  9 x 6144 = 165,888 B -> 0.229 us x 78 layers = 17.8 us / token
V4-Pro       : 3 x 1 x  7 x 7168 = 150,528 B -> 0.207 us x 61        = 12.7 us / token
Kimi K3      : 3 x 1 x 18 x 3584 = 193,536 B -> 0.267 us x 92        = 24.5 us / token
Qwen3.8-2.4T : 3 x 1 x 11 x 8192 = 270,336 B -> 0.372 us x 92        = 34.3 us / token
```

`[inferred]` **Under 35 us/token on every model. EP's bandwidth cost at C1 is 0.6-1.3% of
TPOT. Its real cost is +50% on the collective *count* — the term that already costs 296-640
us — to save exactly zero bytes.** That is the general result, and it holds for all four
models by the same arithmetic.

`[verified]` And on K3 the EP penalty is not marginal, it is catastrophic: replicating 24.31 GB
of BF16 shared experts on every rank takes the TP8 roofline from **463 to 203 tok/s (2.3x)** and
the TP16 roofline from 927 to 247 (3.8x). `[inferred]` SGLang's own K3 low-latency cells carry
no `--moe-a2a-backend` flag, which is consistent.

`[verified]` One number to *not* inherit: SGLang's V4-Flash **balanced** B200 recipe passes
`--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'` —
**96 of 148 SMs, 65% of the GPU, on comms.** That is a throughput choice and would be
catastrophic at C1.

### 3.4 Expert residency E(T), and where the answer changes

`[inferred]` With top-k routing over `E` experts and `T` tokens in a step, the expected number
of **distinct** experts touched is `E(T) = E * (1 - (1 - k/E)^T)`. This is the function that
decides when EP starts paying and when marginal tokens become free.

| model | E | k | T=1 | T=16 | T=64 | T=256 |
|---|---:|---:|---:|---:|---:|---:|
| **GLM-5.2** | 256 | 8 | 8.0 (3.1%) | **102.0 (39.8%)** | **222.4 (86.9%)** | 255.9 (100.0%) |
| DeepSeek-V4-Flash | 256 | 6 | 6.0 (2.3%) | 80.8 (31.6%) | 199.9 (78.1%) | 255.4 (99.8%) |
| Qwen3.8 / Qwen3.5 | 512 | 10 | 10.0 (2.0%) | 138.6 (27.1%) | 367.1 (71.7%) | 508.7 (99.4%) |
| Kimi K3 | 896 | 16 | 16.0 (1.8%) | 224.4 (25.0%) | 613.2 (68.4%) | 887.1 (99.0%) |
| DeepSeek-V4-Pro | 384 | 6 | 6.0 (1.6%) | 85.5 (22.3%) | 243.8 (63.5%) | 377.2 (98.2%) |

`[inferred]` **GLM-5.2 saturates its expert bank faster than anything else in the set** — 87%
of experts read per step at C64 against V4-Pro's 64% — purely because its `topk/E` is 3.13%,
the highest in the group. **That is the direct explanation for why GLM-5.2 measures 40,794
tok/s at C64 on a coding workload and `03-qwen3` models Qwen3.5-397B at only ~14.4k at the same
concurrency, reaching GLM-parity around C256.** It is a genuine design trade: GLM-5.2 bought
efficiency at moderate concurrency and gave up peak-throughput headroom.

**And the number that decides the MoE kernel's shape.** `[inferred]` Under EP8 a rank owns
`E/8` experts and receives `T*k/8` tokens spread over `E(T)/8` resident experts, so the mean
tokens per resident expert is `T*k / E(T)`:

| T | GLM-5.2 | V4-Flash | Qwen | K3 | V4-Pro |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16 | 1.26 | 1.19 | 1.15 | 1.14 | 1.12 |
| 64 | **2.30** | 1.92 | 1.74 | 1.67 | 1.57 |
| 256 | **8.00** | 6.01 | 5.03 | 4.62 | 4.07 |

`[inferred]` **`M <= 8` per expert at every concurrency this box can hold, on every model.**
That is why the expert GEMM is the same kernel problem everywhere: it is a bandwidth-bound
weight-streaming problem, not a matmul, and it is why the incumbent cubin on this box has an
N-tile of 8 (`t128x8x512u2`) against a per-rank expert N of 256.

**Where the parallelism answer changes.** `[verified]` The published cell boundaries, across
three vendors and four models, put the switch from `low-latency` (pure TP, long draft) to
`balanced` (DP-attention + EP, short draft) **between C16 and C64**:

| model | low-latency cells cover | balanced cells cover |
|---|---|---|
| GLM-5.2 B200 NVFP4 | C1-C16, TP8 no DP, 5-1-6 | C64-C256, `--tp 8 --dp 8 --enable-dp-attention`, 2-1-3 |
| DeepSeek-V4-Flash B200 | C1, C16, TP4 no DP, 3-1-4 | C64, C256, `--tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend deepep`, 1-1-2 |
| DeepSeek-V4-Pro B200 | C1, C16, TP8 no DP, 3-1-4 | C64, DP-attention |
| Kimi K3 B200 (2 nodes) | TP16 no DP, no EP | TP16 + DCP16; HT adds DP x EP |
| Qwen3.5-397B | TP8 + MTP-1 | DP8 + EP8, target C>=256 |

`[inferred]` `02-kimi-k3` independently models the crossover at **C8-C16** from the
shared-expert amortization curve. Two derivations, same neighbourhood. **Call it C16-C32 and
measure it.**

### 3.5 DP-attention: three arguments for, one against, and how to settle it

`[inferred]` The corpus contains a genuine, unresolved conflict and it is worth stating rather
than papering over.

**Three independent arguments FOR DP-attention at C1 on any MLA model:**

1. **Arithmetic intensity.** `h_q * s_q >= 147` is the compute-bound condition; TP8 gives
   `h_q = 8` and needs `s_q = 18`, DP8 gives `h_q = 64` and needs `s_q = 2.3`, which 3-1-4
   already exceeds. §2.3.
2. **Kernel tile match.** The only native SM100 MLA decode kernel is `head64`; GLM-5.2 has
   64 query heads. TP8 pads 8 to 64 and wastes 87.5% of the tile. §2.4.
3. **KV memory is free.** MLA has one KV head, so TP already replicates the whole cache on
   every rank. DP costs nothing extra. §1.2d. This is the argument that usually kills the
   proposal and it is wrong.

**The argument AGAINST, from `01-glm-5.2`:** at C1 there is exactly one sequence. SGLang's
DP-attention assigns whole sequences to DP ranks, so DP8 leaves **seven ranks idle in
attention** and adds a gather. `[verified]` Every published `low-latency` cookbook cell for
every model omits `--dp` / `--enable-dp-attention`; every `balanced` and `high-throughput` cell
includes them.

`[inferred]` **Both are right about different things, and the disagreement is really about what
"DP-attention" names.** Arguments 1-3 are about *not sharding the query heads* — i.e.
attention-TP = 1 — which at C1 means one rank does the attention with a compute-bound
`h_q = 64` kernel while seven wait. The counter-argument prices the waiting. Since attention is
only 10.9% of C1 GPU time on GLM-5.2, the trade is: pay up to 7/8 of 10.9% in idle time to
convert the remaining work from a memory-bound `h_q=8` kernel with a 12.5%-occupied tile into a
compute-bound `h_q=64` kernel with a full tile. **Whether that nets out positive is a
measurement, not an argument**, and it is cheap: run the published low-latency cell with and
without `--enable-dp-attention` at C1 and read the attention share off an nsys trace.

`[verified]` And note that the four published low-latency recipes are not neutral evidence
here — three of them (GLM-5.2 B200, V4-Pro B200, Qwen3.5-397B) are `verified: true` cells that
someone measured. `[inferred]` The prior should be that pure TP wins at C1 and the experiment
is worth running anyway, because two of the three supporting arguments (tile match, arithmetic
intensity) were only articulated after those cells were published.

### 3.6 The rule that generalizes

`[inferred]` One rule, four models, derived from the arithmetic above and confirmed by the
published cells:

```
C1 - C16      : pure TP, largest draft that survives capture, no EP, no DP-attention.
                Reason: EP adds +50% collective count for 0 bytes saved (§3.3); expert
                residency is 2-4% so there is nothing to amortize (§3.4); the shared-expert
                fusion is live only under plain TP (§1.1).

C16 - C256    : DP-attention + EP, short draft (1-1-2 / 2-1-3).
                Reason: expert residency crosses 25-40% at C16 and 64-87% at C64, so EP's
                bigger per-expert GEMMs start paying; DP-attention costs no KV on any model
                in the set and saves 2-4x on Qwen; verify overhead starts exceeding
                accept-length gain.

C256+         : DP-attention + EP, speculation OFF, mem-fraction 0.90-0.92.
                Reason: >=98% of experts are read every step so marginal tokens are nearly
                free; the verify step is pure overhead at saturation.

Never         : EP at C1 (any model). TP16 across a node boundary (186 collectives x 20 us
                = 3.7 ms against a 1.1 ms floor). PP inside one node at C1 (pure bubble;
                and GLM-5.2 has two open PP bugs). EP on Kimi K3 at any concurrency below
                ~C16 (shared-expert replication is 2.3-3.8x).
```

`[inferred]` The one place this rule needs a model-specific override: **Kimi K3 and
Qwen3.8-2.4T do not fit on 8 GPUs**, so their C1 answer is a two-node question, and there the
choice is TP16 (all 186 collectives on the inter-node fabric, but DSPARK works) against
TP8 x PP2 (one crossing per token, but `pp_size == 1` is required for DSPARK). `[inferred]`
`02-kimi-k3` models both at ~the same place (210 vs 478 tok/s before speculation, ~500 vs 478
after) and concludes it is a measurement. Correct.

---

## 4. The memory-bandwidth roofline, side by side

All rows on **7.672 TB/s per GPU** (driver-derived: 960 B/clk x 3.996e9 clk/s x 2 DDR), which
is 61.376 TB/s aggregate over 8 GPUs. Bytes-per-parameter by format:
BF16 = 2.0; FP8 block-128x128 = 1.00006; MXFP4 g32 = 0.53125; NVFP4 g16 = 0.5625.

```
tok/s ceiling = 7.672e12 / (bytes read per rank per decoded token)
```

| model / build | fits 8xB200? | parallelism | active params | **MB/rank/token** | **ms** | **roofline tok/s** | 4-bit / FP8 / BF16 byte split |
|---|:--:|---|---:|---:|---:|---:|---|
| **GLM-5.2 NVFP4 as shipped** | **yes** (56.72 GiB/GPU) | TP8 | 40.30 B | **8,756** | 1.141 | **876** | 18% / 0% / **82%** |
| GLM-5.2 FP8 | **yes** (89.37 GiB/GPU) | TP8 | 40.30 B | 6,674 | 0.870 | **1,150** | 0% / 92% / 8% |
| *GLM-5.2, non-expert GEMMs requantized to FP8* | yes | TP8 | 40.30 B | *6,048 or 5,293* | *0.788 / 0.690* | ***1,268 - 1,449*** | *~26-30% / ~55-63% / ~8%* |
| **DeepSeek-V4-Pro-0813 FP4** | **yes** (103.93 GiB/GPU) | TP8 | 48.85 B | **4,960** | 0.647 | **1,547** | 32% / 58% / **10%** |
| **DeepSeek-V4-Flash-0731 FP4** | **yes** (19.43 GiB/GPU) | TP4 | ~13.2 B | ~2,828 | 0.369 | **~2,713** | ~30% / ~52% / ~18% |
| DeepSeek-V3.2-Exp FP8 | yes (80.27 GiB/GPU) | TP8 | 37 B | `[unverified]` | — | — | 0% / ~92% / ~8% |
| **Kimi K3 MXFP4** | **NO** (181.2 GiB/GPU vs 178.34 usable) | TP8 *(hypothetical)* | 104 B | 16,560 | 2.158 | **463** | 19% / 0% / **81%** |
| Kimi K3 MXFP4, TP16 (2 nodes) | n/a | TP16 | 104 B | 8,280 | 1.079 | 927 | 19% / 0% / 81% |
| Kimi K3 MXFP4 + EP a2a (shared replicated) | NO | TP8+EP | 104 B | 37,840 | 4.932 | **203** | 8% / 0% / 92% |
| **Qwen3.8-2.4T-A95B NVFP4** | **NO** (1,447 of 1,464 GB — 2.1 GB/GPU spare) | TP8 *(hypothetical)* | 93.10 B | 14,050 | 1.831 | **546** | 26% / 0% / **74%** |
| Qwen3.8-2.4T-A95B NVFP4, TP16 | n/a | TP16 | 93.10 B | 7,025 | 0.916 | 1,092 | 26% / 0% / 74% |
| Qwen3.8-2.4T-A95B FP8, TP16 | n/a | TP16 | 93.10 B | 8,431 | 1.099 | 910 | 0% / 38% / 62% |
| **Qwen3.5-397B-A17B FP8** | **yes** (50.75 GiB/GPU) | TP8 | 16.31 B | **2,038** | 0.266 | **3,765** | 0% / **99%** / 1% |
| Qwen3.5-397B-A17B NVFP4 (MoE only) | yes | TP8 | 16.31 B | 2,563 | 0.334 | 2,994 | 23% / 0% / 77% |

**Worked arithmetic for the four flagship rows** `[inferred]`:

```
GLM-5.2 NVFP4:  8,756.1e6 B / 7.672e12 B/s = 1.1413e-3 s -> 876.2 tok/s
   composition (MB/rank/token, from 01-glm-5.2 §3.2):
     fused_qkv_a_proj (REPLICATED, BF16) 2,515.0 = 28.7%   <- 8 ranks do identical work
     o_proj            (row-TP,   BF16) 1,962.9 = 22.4%
     routed experts    (        NVFP4) 1,592.5 = 18.2%
     shared expert     (         BF16)   707.8 =  8.1%
     q_b_proj          (col-TP,  BF16)   654.3 =  7.5%
     indexer  (REPLICATED,       BF16)   393.6 =  4.5%
     kv_b_proj         (col-TP,  BF16)   286.3 =  3.3%
     lm_head           (         BF16)   237.9 =  2.7%
     router   (REPLICATED, BF16/fp32)    235.9 =  2.7%
     dense MLP L0-2    (         BF16)   169.9 =  1.9%

DeepSeek-V4-Pro-0813:  39.683 GB node-wide / 8 = 4,960 MB/rank
   4.960e9 / 7.672e12 = 6.466e-4 s -> 1,546.7 tok/s
   composition: attention 18.295 GB (46%!), routed experts 12.845 (32%),
                shared 4.030, lm_head 1.853, compressor 1.345, indexer 0.625,
                router 0.355, mHC 0.336

Kimi K3 TP8:  16.56 GB/rank  ->  2.1585e-3 s -> 463.3 tok/s
   composition: KDA linears 7.66 GB (46%, BF16), routed experts 3.20 (19%, MXFP4),
                shared experts 3.04 (18%, BF16), latent+gate 1.33, MLA 0.87, rest 0.46

Qwen3.8-2.4T NVFP4 TP8:  112.4 GB node-wide / 8 = 14.05 GB/rank
   1.405e10 / 7.672e12 = 1.8313e-3 s -> 546.0 tok/s
   composition: Gated DeltaNet 30.10 B x 2 B = 60.20 GB (54%, BF16),
                MoE 51.32 B x 0.5625 = 28.87 (26%, NVFP4),
                gated attention 9.65 B x 2 = 19.30 (17%, BF16), lm_head 4.07 (4%, BF16)
```

**Two derivations that deserve their own caveats:**

`[inferred]` **DeepSeek-V4-Flash's active bytes are derived here, not read from a document.**
Using the module structure `04-deepseek` read out of V4-Pro's shard headers, scaled to Flash's
config (43 layers, hidden 4096, 64 heads x head_dim 512, `q_lora_rank` 1024,
`o_lora_rank` 1024 / `o_groups` 8, 256 experts top-6, `moe_intermediate` 2048, 21 CSA + 20 HCA
+ 2 SWA, 3 hash layers):

```
attention  (wq_a 4.19M + wq_b 33.55M + wkv 2.10M + wo_a 33.55M + wo_b 33.55M) x 43 = 4.60 B  FP8
experts    6 x 3 x 4096 x 2048 x 43                                                 = 6.49 B  FP4
shared     3 x 4096 x 2048 x 43                                                     = 1.08 B  FP8
lm_head    129280 x 4096                                                            = 0.53 B  BF16
compressor, indexer, router, mHC  (scaled from Pro)                                ~= 0.52 B  mixed
                                                                              total ~= 13.22 B
```
against the model card's "13 B activated" — closes to **1.7%**, which validates the structure.
Bytes: `4.598 + 3.450 + 1.082 + 1.058 + ~1.12 = ~11.31 GB` node-wide. **Carry a +/-10% band on
this row**, concentrated in the compressor/indexer/mHC terms; the attention and expert terms
are solid.

`[unverified]` **DeepSeek-V3.2-Exp's active bytes are not established anywhere in the corpus.**
`04-deepseek` gives its checkpoint size (689.48 GB, 80.27 GiB/GPU at TP8) and its KV cost
(48,068 B/token) but not a per-token weight-read figure. Row left empty rather than guessed.

### 4.1 Five corrections to the input documents

`[inferred]` Reconciling six models on one denominator surfaced five internal inconsistencies.
Recording them so nobody re-derives them. Corrections 1, 2 and 4 are single-cell slips;
correction 3 is a units choice; correction 5 is systematic.

**1. `02-kimi-k3` §3.4 reports "~12% MBU" for both the TP8 and TP16 rows.** The 12.2% is
correct for TP16 only (1.04 ms floor against an 8.5 ms measured step). At TP8 the same
arithmetic gives `111 / 483 = 23.0%` at 8.0 TB/s, or **24.0% at 7.672 TB/s**. The TP8 row's
"~12%" appears to be a copy of the TP16 cell. `[inferred]` The corrected pair — **24.0% at
TP8 falling to 12.7% at TP16** — is a stronger result than the document's own, because
bandwidth utilization *halving* as you double the ranks is the clean signature of a
collective-count-bound workload.

**2. `03-qwen3` §3.6 gives `E(16) = 145` for 512 experts at top-10.** Recomputing:
`512 x (1 - 0.98046875^16) = 512 x (1 - e^(-0.315582)) = 512 x 0.270734 = 138.6`. The
document's own `E(64) = 367` and `E(256) = 509` reproduce exactly, so this is a single-cell
slip. §3.4 uses 138.6.

**4. `01-glm-5.2` §3.2 gives two mutually inconsistent figures for the requantized GLM-5.2
build.** The prose states *"With every non-expert, non-router GEMM at FP8: 6,048.2 MB"* (and
`04-deepseek` §8.5 independently repeats 6,048 -> 1,268 tok/s). But the same table's own
"MB saved if FP8" column sums to **3,463.0**, which gives `8,756.1 - 3,463.0 = 5,293.1 MB` and
**1,449 tok/s**. The two differ by 755 MB and I cannot reconcile them from what is in the
corpus — the 6,048 figure would require leaving ~1.5 GB of BF16 that the savings column says is
fixable. `[unverified]` **Both are reported here as a band.** The direction of the finding in
§3.2 is unaffected and is in fact strengthened by the lower number: at 5,293 MB the collective
ratio rises to 0.78, not 0.68. **Settle this by summing the actual per-module bytes before
committing to a requantization plan** — it is a 20-line script over the two `config.json`
exclusion lists.

**5. Every roofline in `01-glm-5.2`, `02-kimi-k3` and `03-qwen3` is ~4.3% optimistic** because
they use 8.0 TB/s rather than the driver-derived 7.672. `01-glm-5.2` says so itself
(*"B200 SXM HBM3e nominal 8.0 TB/s `[unverified]` — I did not measure it on this box, and you
should"*) and `04-deepseek` flags it. Restated throughout here.

`[unverified]` And the standing caveat that applies to every row in §4: **what fraction of
7.672 TB/s a real decode access pattern achieves on this box has never been measured.**
`00-hardware` names it as unestablished. Until it is, every ceiling above carries a ±15% band,
and the honest reading of the "% of roofline" column in §7 is *relative* comparison between
models, not absolute MBU.

---

## 5. One engine, four models

### 5.1 The common skeleton

`[inferred]` All four families decompose into the same per-layer sequence. This is not a
generalization from one model — it is what the four sets of configs and the four SGLang model
files have in common:

```
for layer in layers:
    1. pre_norm(h, residual)                 # + optional fused all-reduce carried from
                                             #   the previous layer, + optional quantize
    2. TOKEN MIXER  <-- variation point A
    3. post_norm(h, residual)
    4. CHANNEL MIXER <-- variation point B
    5. reduce (all-reduce | reduce-scatter | deferred into the next pre_norm)
final_norm -> lm_head -> logits_processor
```

Variation point **A** has exactly four implementations across the whole set:

| | mixer | models | layers |
|---|---|---|---|
| **A1** | absorbed MLA/MQA decode against a <=576 latent | GLM-5.2, DSv3.2, K3, DSv4 | 78 / 61 / 24 / 61 |
| **A2** | top-k sparse selector feeding A1 (lightning indexer) | GLM-5.2, DSv3.2, DSv4 | 21 of 78 / 61 / 30 of 61 |
| **A3** | gated linear-attention recurrence (KDA / Gated DeltaNet) | K3, Qwen | 69 of 93 / 69 of 92 |
| **A4** | gated GQA softmax + partial RoPE | Qwen | 23 of 92 |

Variation point **B** has exactly three:

| | mixer | models |
|---|---|---|
| **B1** | grouped blockscaled expert GEMM + shared expert (+ optional fusion) | all four |
| **B2** | dense SwiGLU MLP | GLM-5.2 layers 0-2, K3 layer 0, all dense Qwen models |
| **B3** | latent down-projection around B1, then up | **K3 only** |

`[inferred]` **Seven live cells in a 4x3 grid.** An engine that implements a *token-mixer
registry* and a *channel-mixer registry* over one shared communicator, one shared allocator
interface, one scheduler, one batcher, one sampler and one verify loop serves all four with no
per-model code in any of those six subsystems.

`[verified]` This is not hypothetical — SGLang already does most of it. GLM-5.2 has **no
`glm5.py` and no GLM-specific decoder layer**: `GlmMoeDsaForCausalLM` subclasses
`DeepseekV2ForCausalLM` and overrides one method. Every kernel in a GLM-5.2 profile maps back
to `deepseek_v2.py`, `dsa_indexer.py` and `dsa_backend.py`. `[inferred]` **The registry design
is already the de facto architecture; what is missing is that the *memory* layer did not get
the same treatment**, which is exactly where the bugs are (§5.3 item 1).

### 5.2 The two things the skeleton must get right that it currently does not

`[inferred]` **1. The reduce in step 5 must be a *deferred, fusable* operation, not a call.**
`[verified]` SGLang already does this correctly in structure: `o_proj` is constructed with
`reduce_results=False` and the MLP all-reduce is folded into the next layer's norm when
`should_fuse_mlp_allreduce_with_next_layer`. §3.2 says this is the highest-value shared kernel
in the set. But `[verified]` AllReduce is the only top-10 kernel at **0% CUDA-graph capture**
while AllGather and ReduceScatter are at 100%, and the fusion flag's relationship to the
observed MNNVL kernels is unresolved. **A deferred reduce that is not graph-captured pays the
launch cost 156 times per token.**

`[inferred]` **2. The per-layer `bytes_per_token` function must be allowed to return zero.**
Three of the four families have layers that cache nothing:

| model | layers that need no KV | current SGLang behaviour |
|---|---|---|
| GLM-5.2 | 57 of 78 need no *indexer* KV | **allocates all 78** — 7,392 B/token wasted = 7.39 GB per 1 M tokens per GPU `[verified]` |
| DeepSeek-V4 | 31 of 61 (HCA) need no indexer KV; and CSA layers add 1 entry per 4 tokens, HCA 1 per 128 | **correct** — indexer only on CSA layers `[verified]` |
| Kimi K3 | 69 of 93 need no KV at all (KDA) | correct (state pool is separate) |
| Qwen3.5+ | 69 of 92 need no KV at all (GDN) | correct |

`[verified]` SGLang's own comment admits the GLM-5.2 case — *"shared layers' cache is never
read, so filling it is dead work"* — and grepping `mem_cache/` for `skip_topk` or
`indexer_layer` returns nothing. **The bug and the abstraction are the same thing: the pool
configurator charges `indexer_size_per_token * effective_num_layers` because it has no
per-layer callback.**

### 5.3 The six places per-model code is unavoidable

`[inferred]` Each of these is a place where the models genuinely differ in kind, not degree.
Ranked by risk.

**1. The KV / state pool shape function.** Three fundamentally different models, and no
abstraction over all three that I can defend:

- **Fixed ratio (GLM-5.2, DSv3.2):** `656 B + 132 B` per token per layer, `page_size` asserted
  to 64. Classic paged allocation.
- **Not a fixed ratio (DeepSeek-V4):** a CSA layer adds one entry per 4 tokens, an HCA layer
  one per 128, plus a fixed 128-token sliding window per layer. `[verified]` DeepSeek state the
  consequence themselves in the V4 report §3.5: *"The hybrid attention mechanism violates
  fundamental assumptions behind PagedAttention and its variants."*
- **Per-sequence recurrent state (K3, Qwen):** context-independent, sized by `max_num_seqs`
  rather than `max_model_len`, with a slot count that depends on the *speculation strategy* —
  K3 goes 4 slots (`extra_buffer_lazy`) to 5 (default) to **12** (DSPARK-7 without ReplaySSM).
  `[verified]` And a separate dtype knob (`--mamba-ssm-dtype`) that trades 2x memory for
  1.95x kernel wall clock.

`[inferred]` The honest design is a `KVLayout` interface with three implementations plus a
per-layer `bytes_per_token(layer_id, ctx_len) -> int` callback that may return 0, and a
separate `StatePool` with `slots_per_request(spec_config) -> int`. **This is the single highest
-value abstraction in the document because it is simultaneously the fix for a measured 7.39
GB/GPU waste and the prerequisite for ever serving V4 or a hybrid model correctly.**

**2. Positional encoding.** Four incompatible schemes (§1.6), with per-*module* variation
inside one model (GLM-5.2's indexer needs non-interleaved while its MLA needs interleaved;
V4 has three rotary configs). `[inferred]` **Highest-risk per-model code in the stack because
every failure mode is silent** — wrong layout degrades quality without crashing, and there is
one published instance of exactly that.

**3. The routing pre-pass.** Mostly common (`noaux_tc`, bias on selection only, unbiased gate
weight) but with four irreducible per-model facts: V4's `tid2eid` hash routing makes the first
three layers' routing known before the forward pass — a scheduler-visible property nothing else
has; GLM-5.2's shared-expert fusion changes the kernel's expert count 256 <-> 257 depending on
flags with no log line; K3's experts live at a different width than hidden (3584 vs 7168); K3
replicates its shared experts under EP.

**4. The draft-model forward.** Four shapes: GLM-5.2's in-checkpoint MTP layer (a full decoder
layer with its own indexer, reusing the target's top-k via
`index_share_for_mtp_iteration: true`); V4's 3-block DSpark chain fed once from target layers
[58,59,60]; Qwen's MTP-1 sharing input embeddings; K3's separate 2.25 B DSPARK reading target
layers [7,23,51,67,83]. `[inferred]` The **verify** half is common (§2.3); the draft half is
not, and it must carry an explicit dtype assertion (§2.3's three published bugs).

**5. The checkpoint quantization-override policy.** `[inferred]` Every checkpoint's exclusion
list is different and three of four are wrong in ways that cost 74-82% of decode bytes (§1.5).
The loader needs a declarative per-checkpoint override — a mapping from module glob to target
dtype, applied after the checkpoint's own list. `[verified]` SGLang has exactly one such hook,
`SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN`, and it covers one module (`q_b_proj`, 654 of 6,463
fixable MB on GLM-5.2). **This is a config schema, not a kernel.**

**6. Chat template, reasoning parser, tool-call parser.** Trivially per-model, and worth
naming because the corpus shows real operational cost: GLM-5.2's default reasoning effort is
"max" with only one alternative; Qwen3.8's "preserved thinking" **retains reasoning blocks
across turns by default**, which inflates prefill on multi-turn agentic traffic and interacts
badly with prefix caching; K3 *"occasionally emits tool-call formats its own parser rejects"*
`[reported]`.

`[inferred]` **And six places where per-model code is NOT needed and is sometimes written
anyway:** the scheduler, the continuous batcher, the sampler, the verify/accept loop, the
communicator, and the CUDA-graph capture policy. `[verified]` The corpus contains evidence for
the last one being a live problem: GLM-5.2's B200 low-latency cell omits the
`--max-running-requests 16 --cuda-graph-max-bs 16 --max-prefill-tokens 8192` triple that the
B300 and GB300 cells carry, and the 5-1-6 crash is *in capture*. A capture policy derived from
`(spec_shape, max_running_requests)` rather than hand-written per platform would have caught it.

---

## 6. A shared benchmark and accuracy harness

`[inferred]` Every requirement below is derived from a specific failure recorded in the corpus,
not from good practice in the abstract.

### 6.1 The seven rules, each with the measurement that produced it

| # | rule | the evidence |
|---|---|---|
| 1 | **Real data only. Synthetic input is not a workload.** | `[verified]` on this box: `random` gives accept 4.00/4 and 447.8 tok/s; real sharegpt gives 3.16/4 and 364.9. **Synthetic inflates decode by 22%.** SGLang's own cookbook refuses to publish acceptance on `random` because *"random-dataset acceptance is unrealistic"* |
| 2 | **Report TTFT and TPOT. Never report a derived throughput.** | `[verified]` `04-deepseek` found SGLang's own first-party benchmark file's `tokens_per_sec_per_gpu` column does not reconcile with `1/TPOT` under the file's own stated formula — a 4% gap — and concluded to report TTFT and TPOT only |
| 3 | **Never `--flush-cache` on a task workload.** | `[verified]` prefix caching is worth **1.54x** on real coding traffic at 54.2% hit (26,454 -> 40,794 tok/s at C64). Flushing measures a workload nobody runs |
| 4 | **Accept length must be measured, never simulated.** | `[verified]` the cookbook pins `SGLANG_SIMULATE_ACC_LEN=3.5` for its 5-1-6 cells, `=2` for 1-1-2, `=2.99` for the MI355X 3-1-4 cell. Every number derived from a pinned AL is a model output. The real metric is `generation_tokens_total / spec_verify_calls_total` from `/metrics`. `[inferred]` our own AL is ~2.07 of a max 4 against a published 4.5-5.2 of 6 — **this gap was found by arithmetic, not by measurement, and it is still unmeasured** |
| 5 | **Log the pools and the weights at startup; assert against the model.** | `[verified]` three separate memory bugs in the corpus would have been caught by one startup assertion: GLM-5.2's 78-vs-22 indexer allocation, the NVFP4 index metadata reporting `total_parameters` **wrong by 2x**, and `DeepSeek-V3.2-Exp`'s `total_size` field reporting exactly 2x the truth |
| 6 | **>=3 replicates; refuse to report a delta under 8%.** | `[verified]` run-to-run variance on this box is **~4%** |
| 7 | **No speed number without a paired quality number.** | `[verified]` GLM-5.2's registered 8-GPU test has a gsm8k baseline of **0.92**. `[verified]` And the corpus contains two *silent* quality regressions that a speed-only harness would have shipped: SGLang #32176 (EAGLE acceptance silently collapses after HiCache load-back) and #32459 (EAGLE defeats radix prefix reuse for multi-turn traffic — no crash, silent 97% -> 40-53% reuse collapse) |

### 6.2 The record

`[inferred]` One row per (model, build, parallelism, spec shape, concurrency, dataset). Fixed
schema so rows are comparable across models — which is the whole point, and which no existing
artifact in the corpus achieves.

```
identity     : model, checkpoint_sha256, engine_git_sha, launch_argv_hash
config       : parallelism (tp/dp/ep/pp/dcp), quant, kv_dtype, state_dtype,
               spec_algorithm, spec_shape, page_size, mem_fraction_static
workload     : dataset, isl_p50, osl_p50, concurrency, prefix_hit_rate
latency      : ttft_p50_ms, ttft_p95_ms, tpot_p50_ms, tpot_p95_ms
speculation  : accept_len_measured, spec_verify_calls, generation_tokens
memory       : weights_bytes_logged, kv_pool_bytes_logged, state_pool_bytes_logged,
               graph_pool_bytes_logged
roofline     : bytes_per_rank_per_token_modelled, roofline_tok_s, pct_of_roofline
quality      : gsm8k, <model-specific>, eval_git_sha
hygiene      : replicates, cv, wall_clock_utc, sm_clock_locked_mhz, power_cap_w
```

`[inferred]` **`pct_of_roofline` is the only column comparable across models**, and today the
corpus cannot fill it for a single *kernel* on this box — the ledger says so explicitly
(*"no measured fraction-of-roofline"*). `bytes_per_rank_per_token_modelled` must come from a
checked-in per-model byte model (the arithmetic in §4), so that a change to the exclusion list
moves the denominator in the same commit that moves the numerator.

`[verified]` Two hygiene fields with specific justification: this box's max SM clock is 1965
MHz but benchmarks lock to 1597 (81.3%), and there is **90.9 s of accumulated SW power
capping** against a 1000 W cap. Any record without the clock and cap is not reproducible.

### 6.3 Matching the leaderboard, and saying when you are not

`[verified]` Artificial Analysis methodology: ~10 k input, >=1500 output, single-prompt **and**
10-parallel, P50 over a trailing 72 hours, from GCP us-central1-a over the public internet,
reasoning models at temperature 0.6, **TTFT counted to first token including reasoning tokens.**

`[inferred]` Three consequences for the harness:

1. **Run `isl≈10000 / osl>=1500` at C1 and C10 as a first-class cell**, not just C1/C16/C64.
   SGLang's first-party B200 V4 benchmarks use `isl 8192 / osl 1024` — close, and not the same.
   Our own C1 numbers are on sharegpt, which is neither.
2. **TTFT is scored and we are good at it.** `[verified]` Our 189 ms against the AA provider
   table's 0.97-14.49 s spread. `[inferred]` And the table shows a clear inversion — Nebius at
   267.6 tok/s has 9.35 s TTFT, Fireworks at 126 tok/s has 0.97 s — which is the signature of
   large-batch high-queueing deployments trading TTFT for TPOT. **A deployment tuned purely for
   TPOT can lose on TTFT. Our 189 ms is a competitive asset and should not be traded away
   casually.**
3. **Temperature 0.6, not greedy.** `[inferred]` Every acceptance-length number in the corpus
   that is labelled "temp 0 greedy" (including SGLang PR #29787's 4.3-5.2 table) is an upper
   bound on what AA will see. The harness must sweep temperature and report accept length at
   0.6.

### 6.4 The four things to measure before trusting any of it

`[inferred]` Ranked. Each blocks a large fraction of the corpus.

1. **Resolve the MNNVL-allreduce contradiction** (§2.2). 0.5 engineer-days. Until it is
   resolved, no collective number in any of the five input documents means anything — and
   §3.2 says collectives are the dominant non-bandwidth cost on all four models.
2. **Measure the 8-16 KB all-reduce latency curve at TP8**, separating fixed cost from arrival
   skew, with and without symmetric memory and custom all-reduce. `[unverified]` The 3.44 us
   constant carries §3.1, §3.2, §3.3, and equivalent sections in `02-kimi-k3` and `03-qwen3`.
3. **Measure achieved HBM bandwidth** under a decode-shaped access pattern. `00-hardware`
   flags it as unestablished; every roofline in §4 and §7 carries a ±15% band until it is done.
4. **Measure our own accept length.** `[inferred]` ~2.07 of a max 4 is an arithmetic inference
   from a chain of three modelled quantities. It is also the single largest unexplained gap in
   the GLM-5.2 story. `/metrics`, `generation_tokens_total / spec_verify_calls_total`, on real
   sharegpt at temperature 0.6.

---

## 7. Headline table

All roofline figures at **7.672 TB/s per GPU**. "Fits" means the weights load on 8 x 178.34 GiB
with room for KV, state, CUDA graphs and NCCL buffers.

| model / build | fits 8xB200? | best parallelism (C1) | **roofline tok/s @ C1** | **measured / published C1** | **gap** | source of the measurement |
|---|:--:|---|---:|---:|---:|---|
| **GLM-5.2 NVFP4, EAGLE 3-1-4** | **yes**, 56.72 GiB/GPU | **TP8**, no DP/EP/TBO | **876** | **365.5** | **2.40x** | `[verified]` this box, real sharegpt, accept 3.16/4 |
| GLM-5.2 NVFP4, EAGLE 5-1-6 | yes | TP8 | 876 | 541 `[reported]` | 1.62x | SGLang cookbook B200 cell, **AL pinned to 3.5** — not a measurement of acceptance |
| GLM-5.2 FP8, EAGLE 5-1-6 | yes, 89.37 GiB/GPU | TP8 | 1,150 | 311 `[reported]` | 3.70x | same, AL pinned 3.5 |
| *GLM-5.2, non-expert GEMMs at FP8* | *yes* | *TP8* | ***1,268 - 1,449*** | *—* | *—* | *`[inferred]` — the requant does not exist yet; band per §4.1 correction 4* |
| **DeepSeek-V4-Pro-0813 FP4** | **yes**, 103.93 GiB/GPU | **TP8**, EAGLE 3-1-4 | **1,547** | **235.3** | **6.57x** | `[verified]` SGLang 0.5.15 first-party 8xB200 data in our fork, isl 8192 / osl 1024 |
| **DeepSeek-V4-Flash-0731 FP4** | **yes**, 19.43 GiB/GPU (fits on **one** GPU) | **TP4**, EAGLE 3-1-4 | **~2,713** `[inferred, ±10%]` | **343.6** | **~7.9x** | same source, TP4 |
| DeepSeek-V3.2-Exp FP8 | yes, 80.27 GiB/GPU | TP8 (DP8 for attention) | `[unverified]` | — | — | no per-token byte model in the corpus |
| **Kimi K3 MXFP4** | **NO** — 181.2 GiB/GPU against 178.34 usable; **−17.1 GiB against raw HBM, −232 GiB at mfs 0.85** | **TP16 across 2 nodes** (or TP8xPP2, which forfeits DSPARK) | **927** (TP16) / 463 (TP8, hypothetical) | **118** (TP16, vLLM) / 111 (TP8) `[reported]` | **7.86x** (TP16) / 4.17x (TP8) | vLLM day-0 blog; **no published 8x or 16x B200 number exists** |
| **Qwen3.8-2.4T-A95B NVFP4** | **NO** — 1,447 GB against 1,464 GB, i.e. **2.1 GB/GPU** before KV, DeltaNet state, activations or graphs | **TP16 across 2 nodes** | **1,092** (TP16) / 546 (TP8, hypothetical) | **101** (TP8 on **B300**, no MTP); **307** (FP8 TP16 + MTP-3) `[reported]` | 3.56x (TP16 vs MTP-3) | vLLM recipe; **B300 not B200 — the % column is not directly comparable** `[unverified]` |
| **Qwen3.5-397B-A17B FP8** | **yes**, 50.75 GiB/GPU | **TP8** + MTP chain depth 1-2 | **3,765** | **none published, anywhere** | — | `[inferred]` serialized floor with collectives is 679 us -> **1,473 tok/s**; the honest headroom is unknown |
| Qwen3.5-397B-A17B NVFP4 | yes | TP8 | 2,994 | none | — | **26% slower than FP8** — do not use on B200 |

**Reading the table.**

`[inferred]` **1. Four of the six flagship builds fit on this box, and the two that do not miss
by almost nothing.** Kimi K3 misses raw HBM by **1.2%** and then misses a servable
configuration by 232 GiB. Qwen3.8-2.4T has 2.1 GB/GPU of headroom, and vLLM's own recipe lists
NVFP4 W4A4 as "8 GPUs **(B300)**" — 288 GB parts, not 183 GB parts. **Both are 16-GPU models
and no amount of quantization changes that**: for K3, NVFP4 is +30 GiB *worse* than the shipped
MXFP4, and the only published artifact that would fit is a REAP-50 expert prune with no
published quality delta.

`[inferred]` **2. GLM-5.2 has the smallest gap in the table (2.40x) and DeepSeek has the
largest (6.6-7.9x).** That inverts the usual reading. V4's low KV cost is a *capacity*
advantage bought with **op count** — 61 layers of (compressor + indexer + Sinkhorn-normalized
mHC + SWA branch + compressed branch + 6 expert GEMMs) is an enormous number of small dependent
kernels — and op count is exactly what hurts at 2.7-4.3 ms TPOT. `[inferred]` **Do not read a
large roofline gap as available headroom. Read it as evidence the model is not
bandwidth-bound.**

`[inferred]` **3. We already beat DeepSeek at C1 on their own first-party B200 numbers**
(365.5 against V4-Flash's 343.6 and V4-Pro's 235.3) **and lose at C16** (78-95 tok/s/stream
against Flash's 114.2). Our C1->C16 falloff is **4.7x**; Flash's is **3.01x**. That falloff, not
the C1 number, is the concrete gap — and the mechanism (DSpark-style adaptive verification
length plus DP-attention) is code our fork already partly has.

`[inferred]` **4. Against the board.** Artificial Analysis GLM-5.2 leaders are 330-336 tok/s
and TileRT publishes ~500 on GLM-5-FP8 on identical hardware. Our 365.5 already clears the
board. The 876 shipped-roofline and 1,268-1,449 requantized-roofline say the target is not
unreasonable — but §3.2 says the requant alone will not deliver it, because it takes collectives
from 47% to 68% of the byte floor.

`[verified]` **5. The single clearest measurement of what a serving stack is worth**, from the
AA provider table: DeepSeek's own API serves V4-Flash-0731 at **111 tok/s** while Nebius serves
**the same weights at 267.6** — a **2.4x spread on identical parameters.** That is roughly the
same multiple as the gap between our 365.5 and TileRT's claimed 500, and it is the best
available answer to "how much is engine engineering worth."

---

## 8. Open questions

`[inferred]` Ranked by how much of this document they would change. Items 1-4 block large
fractions of the corpus and need no new hardware.

1. **Resolve the MNNVL-allreduce contradiction.** 0.5 d. §2.2, §6.4. Blocks every collective
   number in five documents.
2. **Measure the 8-16 KB all-reduce latency and skew at TP8 and TP4.** 1 d. The 3.44 us
   constant carries §3.1, §3.2, §3.3 and the equivalent sections of `02-kimi-k3` and
   `03-qwen3`. If it is really 3.4 us, Qwen3.5-397B is collective-bound and GEMM work on it is
   wasted effort; if it is 1 us, half of §3.2 relaxes.
3. **Measure achieved HBM bandwidth under a decode access pattern.** 1 d. Every roofline in
   §4 and §7 carries a ±15% band. `nvbandwidth` plus an ncu pass on the top three kernels.
4. **Measure our own accept length on real data at temperature 0.6.** Hours. `[inferred]` ~2.07
   of 4 against a published 4.5-5.2 of 6 is the largest unexplained gap in the GLM-5.2 story
   and it was derived, not measured.
5. **Widen `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN` past `q_b_proj`, and measure the collective
   ratio at the same time.** 1-2 d. §1.5, §3.2. Roofline 876 -> 1,268-1,449 but the collective
   share goes 47% -> 68-78%. **Do not measure one without the other.** Precede it with the
   20-line script in §4.1 correction 4 that settles which of the two corpus figures for the
   requantized byte count is right — the answer changes the expected win by 14%.
6. **Fix the per-layer KV callback so it can return zero.** 1 d. §5.2, §5.3 item 1. 7.39 GB per
   1 M tokens per GPU on GLM-5.2 today, and the prerequisite for ever serving V4 or a hybrid
   model correctly. DeepSeek-V4 demonstrates the correct pattern.
7. **Wire DeepGEMM's `sm100_paged_mqa_logits` to GLM-5.2's DSA indexer.** 1-2 d. §2.5. The
   kernel is MIT, SM100-native, paged, FP4-capable, and asserts `kHeadDim == 128` which is
   GLM-5.2's exact `index_head_dim`. Halves indexer cache bytes.
8. **A/B pure TP8 against `--enable-dp-attention` at C1 on GLM-5.2.** 1 d. §3.5 contains a
   real unresolved conflict between three arguments for and one measured recipe against.
9. **Check the indexer RoPE interleaving.** 20 minutes. §1.6. `[verified]` there is a published
   silent-corruption instance and our port may have been written by analogy to pre-Nov-2025
   reference code.
10. **Autotune the expert GEMM for SM100.** 2-3 d. §2.1. `[reported]` 1.59-1.87x kernel-level
    on H20 from tuned-vs-heuristic; **nothing exists for B200 for any of these models' shapes.**
    Probably the largest single free win in the set, and it serves all four.
11. **Wire DSpark's block-accept estimator to GLM-5.2's EAGLE path.** 3-5 d. §1.4. Attacks the
    4.7x C1->C16 falloff, is model-agnostic, sidesteps the `eagle_worker_v2` IMA, and the
    estimator is already in our fork.
12. **`[unverified]` Does FlashMLA's SM100 sparse kernel instantiate `v_head_dim = 256`?**
    Read the template instantiation list. §1.2a. It is the gate on the whole
    FlashMLA/TokenSpeed line of work and it is a 10-minute answer.
13. **`[unverified]` Test the 1x32-scale-ratio condition on GLM-5.2's NVFP4 weights.** A
    30-line script. `[verified]` V4's lossless FP4->FP8 dequantization requires the max/min FP4
    sub-block scale ratio within each 128x128 FP8 block to be bounded. If it holds on our
    weights, an NVFP4 build can reuse an FP8 kernel pipeline end to end.
14. **`[unverified]` Does a dense-masked prefill path beat DSA at 10 k input?** `[verified]`
    DeepSeek ship two prefill implementations and switch on length, and their published
    sparse/dense crossover is ~3,000 tokens at topk=2048 — uncomfortably close to AA's 10 k.
    TTFT is scored.
15. **`[unverified]` Is Qwen's `scoring_func` sigmoid or something else?** Not established by
    `03-qwen3`. The only gap in the §1.1 router-convergence table.
16. **`[unverified]` What fraction of B200 NVFP4 peak does any expert GEMM achieve?** Nobody
    has published this for any implementation, on any model. §2.1. It is the number that would
    tell us whether kernel work on the #1 shared kernel has headroom at all.

---

## 9. Sources

Everything in this document is a reconciliation of five corpus documents plus the hardware
ground truth and the optimization ledger. No new external fetching was done. Each input
document states its own primary sources; this section records which document established which
fact, so a reader can go one level down without re-searching.

**Corpus inputs, all local `[verified]` as documents:**

- `/home/aman/code/research/05-models/00-local-weights-ground-truth.md` — GLM-5.2 checkpoint
  facts read from the safetensors headers on this box: 753,329,921,024 params exact, the 156
  NVFP4 exclusions, per-module byte table, per-GPU TP8 footprints, the 78-vs-22 indexer
  allocation, the 656 B/token/layer KV layout, the 1,569-tensor MTP layer.
- `/home/aman/code/research/05-models/01-glm-5.2-serving-on-b200.md` — the per-module
  MB/rank/token table in §3.2 (the basis for every GLM-5.2 roofline here), the collectives
  arithmetic in §3.4, the published accept-length tables, the 5-1-6 crash dossier, the
  cookbook cells.
- `/home/aman/code/research/05-models/02-kimi-k3-serving-on-b200.md` — K3's config, the
  1,449.6 GiB / 1,432.5 GiB fit failure, the `E(T)` residency function, the shared-expert
  replication penalty from `kimi_k3.py:519`, the DSPARK acceptance table, the TP16-vs-TP8xPP2
  trade, the state-pool slot counts.
- `/home/aman/code/research/05-models/03-qwen3-serving-on-b200.md` — the Qwen3.5/3.6/3.8
  configs, the 1,447-vs-1,464 GB fit failure, the DeltaNet state arithmetic, the active-param
  derivations (93.10 B and 16.31 B), the 120/184 collective counts, the MTP #52013 story, the
  512-expert amortization curve.
- `/home/aman/code/research/05-models/04-deepseek-v3.2-v4-serving-on-b200.md` — the V4 config
  and shard-header accounting, the CSA/HCA/SWA layer split, the 5,031 B/token KV derivation,
  the `compress_ratios`/`mtp.N` drafter archaeology, the FlashMLA `config.h` and DeepGEMM
  `sm100_mqa_logits.cuh` readings, the first-party 8xB200 benchmark table, the 7.672 TB/s
  discipline, the `h_q * s_q >= 147` crossover.
- `/home/aman/code/research/05-models/06-open-source-kernels-and-implementations-to-steal.md` —
  the installed-package inventory, the 394 `sm100f` cubins and the incumbent MoE cubin's tile
  shape, the FlashMLA / DeepGEMM / TokenSpeed / TileRT / vLLM-CMakeLists arch-gate audit, the
  five in-tree allreduce implementations, TileRT's model list and published numbers.
- `/home/aman/code/research/00-hardware/00-this-machine-ground-truth.md` — driver-probed
  148 SMs, 178.34 GiB usable, **7.672 TB/s** derived from 960 B/clk x 3.996 GHz x 2, 1965 MHz
  max SM clock, 90.9 s of accumulated SW power capping, and the standing note that achievable
  HBM fraction is unmeasured.
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` — the
  C1 and C64 hotspot splits, the arrival-skew decomposition
  (`14,097 = 7,505 transfer + 6,599 waiting`), the "Tried, with verdicts" table (3.09x
  speculative, 1.54x prefix caching, the 4-1-5/5-1-6 IMA, the 22% synthetic inflation), the
  0%-CUDA-graph AllReduce finding, and the standing statement that no kernel on this box has a
  measured fraction-of-roofline.

**New arithmetic introduced by this document** (all `[inferred]`, all shown inline):

- §1.1 activated/total across seven models; per-expert size and `topk/E` spreads.
- §1.5 the 4-bit / FP8 / BF16 byte-composition split for four checkpoints — the 81.8% / 80.7%
  / 74.3% / 9.8% result.
- §1.5 the model-specific requant verdicts (GLM-5.2 take it; Qwen ~12%; K3 a quality risk).
- §3.1 collective latency per token for six models at a common 3.44 us.
- §3.2 the collective-latency-to-HBM-floor ratio table — the 0.30 to 4.02 span and the
  observation that requantizing GLM-5.2 moves its ratio from 0.47 to 0.68-0.78.
- §3.3 the EP a2a bandwidth cost per model at C1 (12.7-34.3 us/token).
- §3.4 `E(T)` for five models at T ∈ {1,16,64,256} and the mean-tokens-per-resident-expert
  table showing `M <= 8` everywhere.
- §4 the whole side-by-side roofline restated at 7.672 TB/s, including the V4-Flash active-byte
  derivation (±10%).
- §4.1 the five corrections to the input documents, including the irreconcilable pair of
  requantized-GLM-5.2 byte figures (6,048 vs 5,293 MB/rank/token).
- §5.1 the 4x3 token-mixer / channel-mixer grid with seven live cells.
- §7 the headline table, gaps, and the reading that a large roofline gap is evidence of *not*
  being bandwidth-bound.

**Deliberately not claimed** `[unverified]`, recorded so nobody assumes it was checked:
DeepSeek-V3.2-Exp's per-token weight-read bytes; Qwen's MoE `scoring_func`; whether FlashMLA's
SM100 sparse kernel instantiates `v_head_dim = 256`; B300's HBM bandwidth (which makes the
Qwen3.8 measured row not directly comparable); the achievable fraction of 7.672 TB/s on this
box; and whether the 3.44 us per-collective constant holds at TP4 or on any model other than
GLM-5.2.
