# DeepSeek: the most detailed public LLM inference infrastructure disclosure there is

## What this is

A mine of DeepSeek's published inference-systems engineering, read from primary sources
(GitHub READMEs and raw source, arXiv HTML/PDF, HuggingFace model cards and reference
implementations, GitHub PR bodies via the API). Every technical claim below is tagged:

- **[verified]** — I fetched the artifact at the URL given and read the text myself.
- **[reported]** — DeepSeek claims it; I read the claim but nobody outside DeepSeek has
  reproduced it. All performance tables in this document are of this kind unless noted.
- **[inferred]** — my own reasoning applied to their disclosures, aimed at our 8×B200 stack.
- **[unverified]** — I could not source it.

Research date: 2026-08-17. **This document has been through a second, adversarial verification
pass**: every URL below was re-fetched, every number re-checked against its primary source, and
claims that could not be confirmed were deleted or downgraded. Corrections made in that pass are
marked inline with **[CORRECTED]**. Scope covers Open Source Week (Feb 2025) through the
DeepSeek-V4 series (Apr–Aug 2026): **DeepEP V2**, **DeepGEMM Mega MoE**, **TileKernels**,
**LPLB**, **Engram**, **DeepSpec/DSpark**, the **DeepSeek-V4 technical report**, and the
**V4 production serving configuration**.

Three headline findings:

1. **DeepSeek-V4 exists, is released, and is fully documented.** [verified] `DeepSeek-V4-Pro`
   (1.6T total / 49B active) and `DeepSeek-V4-Flash` (284B / 13B), both 1M context, MIT
   licensed, on HuggingFace with a reference implementation, plus an arXiv technical report
   (2606.19348, title confirmed: *"DeepSeek-V4: Towards Highly Efficient Million-Token Context
   Intelligence"*). The current production checkpoints are `DeepSeek-V4-Pro-0813` (2026-08-13)
   and `DeepSeek-V4-Flash-0731` (2026-08-01).
2. **DeepSeek publishes its own production serving command lines, for both vLLM and SGLang, and
   they are Blackwell-generation configs.** [verified, NEW in this pass] This is the single most
   actionable thing in the corpus and the first pass missed it entirely. It names
   `--moe-backend deep_gemm_mega_moe`, `--speculative-algorithm DSPARK`, FP4 indexer cache,
   `--block-size 256`, and a 4×GB300 target. Full detail in
   *"What DeepSeek actually ships"* below.
3. **DeepSeek's public reference kernels have moved from CUDA/CUTLASS to TileLang.** [verified]
   They released a dedicated TileLang kernel library (`TileKernels`, SM90/SM100, CUDA 13.1+),
   and the V4 report §3.2 describes replacing "hundreds of fine-grained Torch ATen operators"
   with TileLang fused kernels. This is directly relevant to the TileRT threat model — the
   "tile-level DSL beats hand-written CUTLASS for iteration speed" bet is being made by
   DeepSeek too.

A caution on the whole corpus: **DeepSeek's published operating point is the opposite of
ours.** Their flagship disclosure serves users at 20–22 output tok/s at enormous concurrency
across 226 H800 nodes. We want 365+ tok/s single-stream on 8 B200s. Much of their throughput
engineering (EP144, huge batches, cross-node all-to-all) is *anti-transferable*. What
transfers is their kernel work, their arithmetic-intensity reasoning, their speculative
decoding scheduler, and their hardware bottleneck analysis. I flag this per technique.

---

## Bottom line for our system

Ranked by expected value on our two objectives (min single-stream latency at C1; min cost/user
at C64). Every row names the **specific change to an SGLang-derived engine on 8×B200**, so this
is a work list, not a reading list.

### Tier 0 — do this week, low risk

| # | Steal | Concrete change to our SGLang fork | Attacks | Expected effect | Difficulty |
|---|---|---|---|---|---|
| 1 | **Quantize index scores FP32→BF16 before top-k** | One dtype change in the DSA top-k selector kernel, before the sort/select. Validate recall against the FP32 path on our own prompts. | DSA indexer 5.8% | DeepSeek report **2× top-k selector speedup at 99.7% KV recall** [reported, V4 §5.2.1]. Upper bound on our end-to-end gain ≈ 2–3% of C1. | **Low** |
| 2 | **Turn PDL on between dependent kernels** | `deep_gemm.set_pdl(True)` if we use DeepGEMM; otherwise add `cudaGridDependencySynchronize`/`cudaTriggerProgrammaticLaunchCompletion` to our splitkv→combine and GEMM→norm pairs. | dense GEMM 37.1%, attention 10.9% | DeepSeek use it to overlap `splitkv_mla` with `combine`. Cheap launch-tail recovery at 2.74 ms TPOT. | **Low** |
| 3 | **`EVICT_FIRST` TMA cache hints + split the K-block TMA into many small copies** | In our MLA/DSA decode kernel: issue a 64×576 K block as **9× 64×64 TMA copies** and start the first GEMM on the first arrival; tag the copies `EVICT_FIRST`. | attention 10.9% | Part of the 580→660 TFLOPS H800 dense-MLA gain; DeepSeek say the cache hint improves L2 hit rate "as shown by experiments". | **Low–Medium** |
| 4 | **Adopt DeepSeek's own published SGLang flag set as a baseline** | `--moe-runner-backend flashinfer_mxfp4`, `--speculative-algorithm DSPARK`, `--mem-fraction-static 0.90`, `--chunked-prefill-size 4096`, `--swa-full-tokens-ratio 0.1` — see *"What DeepSeek actually ships"*. | everything | Free calibration: these are the flags the model author chose. Note their config is TP4 on GB300, not TP8 on B200 SXM — treat as a starting point, not a target. | **Low** |

### Tier 1 — the structural wins

| # | Steal | Concrete change to our SGLang fork | Attacks | Expected effect | Difficulty |
|---|---|---|---|---|---|
| 5 | **DP attention instead of TP8 for the DSA/MLA decode path** | `--enable-dp-attention` (SGLang has it; DeepSeek ship it in their own V3.2 launch line). Costs KV-cache replication across ranks — budget it against 183 GB/GPU. | attention 10.9% + collectives 19.6% | Restores per-rank `h_q = 128`, moving MLA decode from memory-bound to compute-bound, and deletes a per-layer all-reduce. See the crossover arithmetic below — this is the load-bearing argument in the whole corpus. | **High** (engine surgery + memory budget) |
| 6 | **Mega-MoE fused dispatch→GEMM→SwiGLU→GEMM→combine megakernel** | Replace our DeepEP-dispatch + grouped-GEMM + combine sequence with `deep_gemm.fp8_fp4_mega_moe`. Requires symmetric memory + multi-process launch + PyTorch ≥2.9. vLLM already exposes it as `--moe-backend deep_gemm_mega_moe`, so a port target exists. | MoE GEMMs 19.4% + collectives 19.6% | **1.96× at batch size 1** on a 256-expert/top-6 EP8 config [reported]. BS=1 *is* our C1 regime. Also removes the collective *boundary*, which is where our rank-arrival skew lives. | **High** |
| 7 | **Confidence-scheduled, load-adaptive verification length** (DSpark) | Profile `SPS(B)` once at engine init into a cost table; implement Algorithm 1 as a dynamic top-K admission over per-request cumulative confidences; run it **asynchronously on 2-steps-prior confidences** so CUDA-graph replay and zero-overhead scheduling still work. Start with EAGLE's existing draft probabilities as a crude confidence proxy — no retraining needed for v1. | the 4.7× per-stream falloff C1→C16 | **+60–85% per-user tok/s at matched throughput** vs static MTP-1 in V4 production [reported]. Exactly our C1↔C16 Pareto problem. | **Medium** (scheduler) / **High** (trained confidence head) |
| 8 | **Kill per-launch CPU overhead (host codegen)** | Audit every Python-side op wrapper on the decode path; move shape/dtype/stride validation to a generated host launcher, or pre-bake it into CUDA graphs. | everything, at TPOT 2.74 ms | DeepSeek report host-side validation dropping **from tens-to-hundreds of µs to <1 µs per invocation** [reported]. Independently corroborated by Hybrid-EP's 409 vs 599 GB/s Torch-API-vs-kernel gap on B200. | **Medium** |

### Tier 2 — worth a spike

| # | Steal | Concrete change to our SGLang fork | Attacks | Expected effect | Difficulty |
|---|---|---|---|---|---|
| 9 | **"Crossover": share dequantized KV between CTAs via distributed shared memory** | Launch DSA decode with cluster size 2; each CTA `__ldg`s and dequantizes half the KV, then `st.async`-writes it into the peer CTA's SMEM, synchronized by a cluster transaction barrier. | attention 10.9% under NVFP4/FP8 KV | 250→410 TFLOPS on H800 [reported]. **Check first**: B200 has native FP4/FP8 conversion paths H800 lacks, so our dequant cycle count may already be below the MMA count — measure before porting. | **Medium–High** |
| 10 | **DeepEP V2 analytical SM budgeting / Hybrid-EP TMA path** | Call `get_theoretical_num_sms(num_experts, num_topk)` instead of auto-tuning; evaluate the `hybrid-ep` branch (it has NVFP4 support and explicitly targets "single-batch scenarios"). | collectives 19.6% | On B200 EP8: Hybrid-EP hits 409.71 GB/s FP8 dispatch at **16 SMs**, where baseline DeepEP needs 44–48 SMs for 544–554 GB/s. ~30 SMs returned to GEMM. | **Medium** |
| 11 | **Batch-invariant dual-kernel decode (wave-quantization fix)** | Two bitwise-identical attention kernels: one sequence-per-SM for full waves, a multi-SM-per-sequence variant using thread-block-cluster DSM for the ragged tail wave. | collectives 19.6% (47% rank skew) | Rank-arrival skew at C1 *is* tail-wave latency variance. Closest published analogue to our problem, and B200 has clusters. Reproducibility comes free. | **High** |
| 12 | **Masked-MHA dense prefill path for short contexts** | Ship two prefill implementations and switch on sequence length, as DeepSeek do for V3.2. | TTFT 189 ms | AA uses ~10k input. At 10k with topk≈2048 the gather+indexer overhead may not pay for itself vs a dense masked kernel. Cheap experiment, possibly large TTFT win. | **Medium** |

### Explicitly *not* worth it, and why

- **EPLB / LPLB at C1.** With 8 active experts routed across 8 ranks at batch 1, the load is 8 tokens over 8 ranks — some ranks get zero work. That imbalance is *structural*, not statistical, and no load balancer fixes it. LPLB's own README states the solver takes **~100 µs**, which is 36× our entire per-token budget. EPLB (130 lines) is still worth wiring up for C64.
- **On-disk / hierarchical KV cache** for the leaderboard. DeepSeek measured 56.3% of input tokens hitting it in production, which is enormous for cost/user — but **zero effect on Artificial Analysis**, whose prompts are fresh.
- **Node-limited routing, DualPipe, EP144-style disaggregation.** Anti-transferable: we are one NVLink domain, no PP, and our operating point (365 tok/s single-stream) is the opposite of theirs (20–22 tok/s/user at massive scale).
- **The `ld.global.nc.L1::no_allocate` PTX hack.** Its correctness argument is explicitly Hopper-specific. Do not port to SM100 without validation.
- **LogFMT-style log-domain communication compression.** DeepSeek built it, measured it, and killed it: 50–100% encode/decode overhead when fused with all-to-all.
- **SASS FFMA-interleaving post-processing.** Obsoleted by NVCC 12.9, which does it automatically. DeepGEMM disabled it.

**The single most important number in this whole corpus for us**, from the FlashMLA deep-dive
[verified]: MLA decode is compute-bound iff `h_q · s_q ≥ 128` on H800-class hardware, and
DeepSeek explicitly say *"we don't use Tensor Parallel for decoding instances, meaning h_q is
128 and the kernel is compute-bound."* Under our TP8 with 128 query heads, per-rank `h_q = 16`;
even with EAGLE giving `s_q ≈ 4`, `h_q·s_q ≈ 64` — **we are on the memory-bound side of the
line, and DeepSeek deliberately architected to be on the other side.** [inferred]

**The single most important number in this whole corpus for us**, from the FlashMLA deep-dive
[verified]: MLA decode is compute-bound iff `h_q · s_q ≥ 128` on H800-class hardware, and
DeepSeek explicitly says *"we don't use Tensor Parallel for decoding instances, meaning h_q is
128 and the kernel is compute-bound."* Under our TP8 with 128 query heads, per-rank `h_q = 16`;
even with EAGLE giving `s_q ≈ 4`, `h_q·s_q ≈ 64` — **we are on the memory-bound side of the
line, and DeepSeek deliberately architected to be on the other side.** [inferred]

---

## The DeepSeek-V3/R1 Inference System Overview (Open Source Week, Day 6)

The masterclass in inference economics. Source is a single markdown file; I read it verbatim.
[verified] https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md

### Stated objectives and the EP argument

Two objectives, quoted: *"higher throughput and lower latency."* The mechanism is cross-node
Expert Parallelism, justified two ways:

- EP **scales the batch size**, improving GPU matrix-computation efficiency → throughput.
- EP **distributes experts across GPUs** so each GPU holds only a small subset, *reducing
  memory-access demands* → latency.

They name the two costs EP imposes: cross-node communication (needs compute/comm overlap) and
mandatory DP across instances (needs load balancing). The sparsity argument is explicit: with
only 8 of 256 experts active per layer, *"the model's high sparsity necessitates an extremely
large overall batch size"* to give each expert enough tokens.

### The exact parallelism degrees

Prefill/decode disaggregated, with **different EP degrees per phase**:

| Phase | Routed experts | MLA / shared expert | Deployment unit | Redundant experts | Per-GPU experts |
|---|---|---|---|---|---|
| Prefill | **EP32** | **DP32** | 4 nodes (32 GPUs) | 32 | 9 routed + 1 shared |
| Decode | **EP144** | **DP144** | 18 nodes (144 GPUs) | 32 | 2 routed + 1 shared |

Note **DP, not TP, for the attention and shared-expert path in both phases.** This is the
architectural decision that makes their MLA kernel compute-bound (see FlashMLA below).

This differs from the V3 technical report's stated deployment (prefill 4 nodes / 32 GPUs,
attention TP4+SP with DP8, EP32; decode 40 nodes / 320 GPUs, attention TP4+SP with DP80,
EP320) [verified, arXiv 2412.19437]. The Feb-2025 blog reflects the later, TP-free production
configuration. [inferred] The trajectory is TP4 → TP1 for attention.

### Compute–communication overlap

- **Prefill**: dual-batch overlap. Batch split into two microbatches executed alternately;
  *"the communication cost of one microbatch is hide behind the computation of the other."*
- **Decode**: the stage durations are unbalanced, so they **subdivide the attention layer into
  two steps and use a 5-stage pipeline** for seamless overlap.

The profile-data repo [verified] adds the crucial decode detail: *"unlike in prefilling, the
all-to-all communication during decoding does not occupy GPU SMs: after RDMA messages are
issued, all GPU SMs are freed, and the system waits for the all-to-all communication to
complete after the computation has finished."*

The prefill profile also notes something subtle and worth copying: they ensure *"the attention
computation load is balanced across the two micro-batches — meaning that the same prompt may
be split between them."* [verified]

### The three load balancers

| Balancer | Problem | Objective |
|---|---|---|
| **Prefill LB** | varying request counts / seq lengths across DP instances | balance core-attention compute; equalize **input token counts** per GPU (dispatch-send balance) |
| **Decode LB** | uneven request counts / seq lengths | balance **KVCache usage** per GPU; equalize **request counts** per GPU |
| **Expert-Parallel LB** | inherently high-load experts | minimize the **maximum dispatch-receive load** across GPUs |

### The precision split (important and often misquoted)

Quoted [verified]: *"matrix multiplications and dispatch transmissions adopt the FP8 format
aligned with training, while core MLA computations and combine transmissions use the BF16
format."* So: FP8 GEMM + FP8 dispatch; BF16 core attention + BF16 combine.

### The measured production statistics (24 h, UTC+8 2025-02-27 12:00 → 02-28 12:00)

| Metric | Value |
|---|---|
| Peak node occupancy (V3+R1) | 278 nodes (8× H800 each) |
| Average node occupancy | 226.75 nodes |
| Assumed GPU rental | $2 / H800-hour |
| **Total daily cost** | **$87,072** |
| Total input tokens | 608 B |
| — of which on-disk KV cache hits | 342 B (**56.3%**) |
| Total output tokens | 168 B |
| Average output speed | **20–22 tokens/second** |
| Average KV-cache length per output token | 4,989 tokens |
| Per-node prefill throughput | ~73.7 k tokens/s input (incl. cache hits) |
| Per-node decode throughput | ~14.8 k tokens/s output |
| Theoretical daily revenue at R1 pricing | $562,027 |
| **Cost profit margin** | **545%** |

R1 pricing used: $0.14/M input (cache hit), $0.55/M input (cache miss), $2.19/M output.

**I reproduced both calculations** [inferred, arithmetic]:
- Cost: 226.75 nodes × 8 GPUs × 24 h × $2 = **$87,072** ✓ exact.
- Revenue: 342B × $0.14/M + 266B × $0.55/M + 168B × $2.19/M = $47,880 + $146,300 + $367,920 =
  **$562,100** ≈ their $562,027 (difference is rounding on the token counts) ✓.
- Margin: (562,027 − 87,072)/87,072 = **545%**, i.e. revenue ≈ **6.45× cost**.

They then knock down their own number honestly, listing why actual revenue is *"substantially
lower"*: V3 is priced far below R1; web and app access are free; nighttime discounts apply
automatically.

### Two operational lessons that are cheap for us to copy

1. **Diurnal capacity reallocation.** [verified] *"we implemented a mechanism to deploy
   inference services across all nodes during peak daytime hours. During low-load nighttime
   periods, we reduce inference nodes and allocate resources to research and training."* Peak
   278 vs average 226.75 nodes is an ~18% utilization gain that costs nothing but scheduling.
2. **Per-GPU throughput normalization.** 73.7k/8 ≈ **9.2k prefill tok/s/GPU**; 14.8k/8 ≈
   **1.85k decode tok/s/GPU** on H800. [inferred] Our 40.8k tok/s aggregate at C64 on 8 B200
   is ~5.1k tok/s/GPU — but at radically different per-stream speed (they run 20–22 tok/s/user,
   we run far higher), and B200 vs H800. The two numbers are not directly comparable; I flag
   this because the comparison is frequently made badly.

---

## FlashMLA — the MLA/DSA attention kernel library

Repo: https://github.com/deepseek-ai/FlashMLA (MIT). [verified] README, both deep-dive blogs,
and the commit log read directly.

### What is actually in the repo now

It is no longer "the Hopper MLA decode kernel." Current support matrix [verified]:

| Kernel | GPU arch | MLA mode | KV format |
|---|---|---|---|
| Dense decoding | SM90 | MQA (`head_dim_k`=576, `head_dim_v`=512) | BF16 |
| **Sparse decoding** | **SM90 & SM100** | MQA | **FP8** |
| Dense prefill | **SM100** | MHA (`head_dim_k`=192/128, `d_v`=128) | — |
| **Sparse prefill** | **SM90 & SM100** | MQA | — |

Requirements: CUDA 12.8+, **CUDA 12.9+ required for SM100 kernels**; PyTorch 2.0+.

### Published performance (all [reported], DeepSeek's own benchmarks)

| Kernel | Hardware | Number | Config notes |
|---|---|---|---|
| Dense MLA decode | H800 SXM5, CUDA 12.8 | 3000 GB/s memory-bound; **660 TFLOPS** compute-bound | up from 580 TFLOPS pre-2025.04.22 |
| Sparse MLA decode (FP8 KV, BF16 math) | H800 SXM5 | **410 TFLOPS** | bs=128, heads=128, `s_q`=2, topk=2048 |
| Sparse MLA decode | **B200** | **350 TFLOPS** | README states *"not really optimized yet"* |
| Sparse MLA prefill | H800 SXM5, CUDA 12.8 | 640 TFLOPS | |
| Sparse MLA prefill | **B200, CUDA 12.9** | **1450 TFLOPS** | |
| Dense MHA prefill | **B200** | 1460 TFLOPS fwd / 1000 TFLOPS bwd | *"as reported by NVIDIA"* |

**Blackwell status, plainly**: sparse prefill on B200 is excellent (1450 TFLOPS). **Sparse
decode on B200 is explicitly un-tuned (350 TFLOPS, below the H800's 410).** [verified] The
SM100 MHA forward/backward kernels came from an NVIDIA PR (#76, 2025-08-01), not DeepSeek.
[inferred] If we adopt FlashMLA for our DSA decode path on B200 we are adopting a kernel its
authors say is not optimized for our hardware — that is an opportunity, not a blocker.

### The compute-bound argument (deep dive, 2025-04-22)

[verified] https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md

FLOPs ≈ `2·h_q·s_q·s_k·(d_k+d_v)`; bytes ≈ `2·s_k·d_k`. Ratio ≈ `h_q·s_q·(d_k+d_v)/d_k ≈ 2·h_q·s_q`.
H800: 3.35 TB/s, 990 TFLOPS peak but **~865 TFLOPS practical after clock throttling to ~1600 MHz**.
Crossover: `h_q·s_q ≥ ½ · 865/3.35 = 128`.

**This is the load-bearing sentence for our stack** [verified quote]: *"According to the
overview of DeepSeek's Online Inference System, we don't use Tensor Parallel for decoding
instances, meaning h_q is 128 and the kernel is compute-bound."*

[inferred] Redo it for 8×B200: B200 HBM3e ≈ 8 TB/s, dense BF16 ≈ 2.25 PFLOPS. Crossover
`h_q·s_q ≈ ½ · 2250/8 ≈ 140` — roughly the same threshold. Under TP8 with 128 heads we have
`h_q = 16` per rank. We need `s_q ≈ 9` accepted-and-verified query tokens per step to reach
the crossover. EAGLE 3-1-4 does not get us there. **Either go DP-attention (h_q=128 → crossover
at s_q=2, trivially satisfied), or accept a memory-bound decode kernel.** This is
recommendation #1 in the bottom line.

### "Seesaw" scheduling — the actual kernel trick

The constraint: a 64×512 FP32 output tile needs 32,768 32-bit registers; an SM has 65,536. So
**only one output matrix fits per SM**, which kills FlashAttention-3's ping-pong scheduling
(which needs two). DeepSeek's answer [verified]:

Split the output vertically into `O_L`, `O_R` (each 64×256), owned by warpgroup 0 and 1
respectively; split `V_0`, `V_1` likewise. Then interleave, with a *shared* running max `m`:

```
0. shared running max m = -inf; o_L = o_R = 0
1. [wg0] p0 = q·K0ᵀ / qk_scale
2. [wg1] p1 = q·K1ᵀ / qk_scale
3. [wg0] mp0=max(p0); m_new0=max(m,mp0); scale0=exp(m_new0-m); m ← m_new0
4. [wg0] p0 ← exp(p0 - m_new0)
5. [wg0] o_L ← o_L·scale0 + p0·V_0L
6. [wg1] mp1=max(p1); m_new1=max(m,mp1); scale1=exp(m_new1-m); m ← m_new1
7. [wg1] p1 ← exp(p1 - m_new1)
8. [wg1] o_R ← o_R·(scale0·scale1) + p1·V_1R
9. [wg0] p0 ← p0·scale1
10.[wg1] o_R ← o_R + p0·V_0R
11.[wg0] o_L ← o_L·scale1 + p1·V_1L
```

Mathematically equivalent to FlashAttention online softmax, but it lets two warpgroups
alternate CUDA-core and Tensor-core work with a *single* output matrix. They call it a
"ping-pong variant using one output matrix."

Supporting techniques [verified]:
- **Fine-grained TMA-copy ↔ GEMM pipelining**: a 64×576 K block is fetched as **9 separate TMA
  copies of 64×64**; the first GEMM starts as soon as the first copy lands.
- **Cache hints**: `cute::TMA::CacheHintSm90::EVICT_FIRST` on TMA copies *"improves L2 cache
  hit rates, as shown by experiments."*
- **Programmatic Dependent Launch (PDL)** to overlap the `splitkv_mla` and `combine` kernels.
- **Tile scheduler** allocating (request, block) jobs to SMs for balance.

Result: *"up to 80% Tensor Core utilization (of the throttled theoretical peak) and 3 TB/s
memory bandwidth"* — with an honest **negative note: ~2% slower than the old ping-pong-buffer
version in memory-bound settings**, which they accept. [verified]

### "Crossover" — the FP8 sparse decode trick (deep dive, 2025-09-29)

[verified] https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md

Motivation: at 128K context a single request's MLA KV cache is
`576 × 2 × 62 × 128 × 1024 = 8.72 GiB`. Hence FP8 KV.

**FP8 KV cache format — 656 bytes per token** [verified]:
- first 512 B: 512× `float8_e4m3` — the quantized NoPE part, tile-quantized at **1×128**
- next 16 B: 4× `float32` scale factors (one per 128 values)
- last 128 B: 64× `bfloat16` — the RoPE part, **left unquantized for accuracy**

The bottleneck they found is *not* memory and *not* tensor cores — it is **dequantization on
CUDA cores**. Their cycle accounting per K/V token, per SM, with 64 query heads per CTA:
- MMA: `64 × (576+512) × 2 / 4096 ≈ 34 cycles` (4096 MMA FLOP/clk = 989 TFLOPS/1830 MHz/132 SMs)
- Dequant: H800 cannot cast `e4m3`→`bf16` directly, so it is e4m3→half→float32→bf16→×scale,
  costing `(1/64 + 1/64 + 1/16 + 1/256) × 512 ≈ 50 cycles`

**The kernel is dequantization-bound, 50 cycles vs 34.** The fix exploits MQA: every query head
in a token attends the *same* key head, so two CTAs handling different halves of the 128 query
heads can share dequantized KV. They launch **clusters of size 2**; each CTA (a) `__ldg`-loads
half the quantized KV with **128-bit wide loads**, (b) dequantizes its half on CUDA cores,
(c) stores to its own shared memory, and (d) simultaneously `st.async`-writes it into the peer
CTA's shared memory via **Hopper Distributed Shared Memory**, synchronized with the **cluster
transaction barrier**. Each CTA then has the full dequantized KV.

Result [reported]: **250 → 410 TFLOPS** (bs=128, heads=128, `s_q`=2, topk=2048, H800). With
topk=32768 it reaches **460 TFLOPS**. They also give the honest crossover: this sparse kernel's
runtime at that config equals the dense kernel's at **sequence length ≈ 3000**; beyond that
sparse wins.

[inferred] **This is highly transferable to us.** Our NVFP4 build will hit exactly the same
dequant-bound wall on the DSA decode path, and B200 has DSM (thread block clusters) too. The
`index_topk_freq=4` in our config means the indexer runs every 4 steps — the *attention* still
dequantizes every step, so crossover applies unchanged. Note B200 *does* have native FP4/FP8
conversion paths that H800 lacks, which may shift where the bound sits — measure before porting.

### API details worth having

`get_mla_metadata(cache_seqlens, s_q*h_q//h_kv, h_kv, h_q, is_fp8, topk)` is called **once
before the decode loop**, then `flash_mla_with_kvcache(...)` per layer. `s_q` is *"the number
of q tokens per q sequence. If MTP (speculative decoding) is disabled, it should be 1"* —
i.e. the kernel is speculative-decoding-aware by design. Sparse `indices` is
`(batch, seq_len_q, topk)` with values pre-multiplied into
`page_block_index * page_block_size + offset` **so the kernel needs no `block_table`**;
invalid entries are `-1`. Paged KV block size 64. [verified]

Recent commits show continued tuning: *"Extend decode-combine num_splits buckets to 256"*
(2026-07-28), *"Swap FlashMLA combine grid dimensions"* (2026-04-29), and a fix for grid-dim
overflow *"when sequence length is very large (>1M)"* (2026-03-31). [verified]

---

## DeepEP — expert-parallel all-to-all

Repo: https://github.com/deepseek-ai/DeepEP (MIT). [verified] Current README, `docs/legacy.md`
(the V1 docs), PR #605, and the `hybrid-ep` branch docs read directly.

### V1 (the version everyone knows) and its two mechanisms

**Normal kernels** — asymmetric-domain bandwidth forwarding. Because V3 uses group-limited
gating, tokens destined for several GPUs on the same node are sent **once over RDMA** and then
**forwarded over NVLink**, deduplicating IB traffic. SM-count controllable via
`Buffer.set_num_sms(24)`. [verified]

**Low-latency kernels** — pure RDMA, for decode, with the **hook-based overlap** that is the
genuinely clever part [verified]:

```python
recv_hidden_states, recv_expert_count, handle, event, hook = \
    _buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank,
                                 num_experts, async_finish=False, return_recv_hook=True)
# "the actual tensor will not be received only if you call hook()",
# "it is useful for double-batch overlapping, but **without any SM occupation**"
```

The kernel issues the RDMA operations and returns; the receive is deferred into `hook()`. Between
issue and hook, **zero SMs are consumed by communication** — the network runs in the background
while all SMs do GEMM. Requirements: `num_qps_per_rank = num_experts // group.size()` (*"for the
best performance, the QP number must be equal to the number of local experts"*), and
`num_max_dispatch_tokens_per_rank` should be **< 256**. CUDA-graph compatible. [verified]

V1 low-latency measurements, H800 + CX7 400 Gb IB, 128 tokens/batch, 7168 hidden, top-8, FP8
dispatch / BF16 combine [reported]:

| EP | Dispatch latency | Dispatch BW | Combine latency | Combine BW |
|---|---|---|---|---|
| 8 | 77 µs | 98 GB/s | 114 µs | 127 GB/s |
| 16 | 118 µs | 63 GB/s | 195 µs | 74 GB/s |
| 32 | 155 µs | 48 GB/s | 273 µs | 53 GB/s |
| 64 | 173 µs | 43 GB/s | 314 µs | 46 GB/s |
| 128 | 192 µs | 39 GB/s | 369 µs | 39 GB/s |
| 256 | 194 µs | 39 GB/s | 360 µs | 40 GB/s |

### The undocumented PTX behaviour they relied on

Quoted verbatim from `docs/legacy.md` [verified]:

> *"For extreme performance, we discover and use an undefined-behavior PTX usage: using
> read-only PTX `ld.global.nc.L1::no_allocate.L2::256B` to **read volatile data**. The PTX
> modifier `.nc` indicates that a non-coherent cache is used. But the correctness is tested to
> be guaranteed with `.L1::no_allocate` on Hopper architectures, and performance will be much
> better. The reason we guess may be: the non-coherent cache is unified with L1, and the L1
> modifier is not just a hint but a strong option, so that the correctness can be guaranteed by
> no dirty data in L1."*

And the negative-result history [verified]: *"Initially, because NVCC could not automatically
unroll volatile read PTX, we tried using `__ldg` (i.e., `ld.nc`). Even compared to manually
unrolled volatile reads, it was significantly faster (likely due to additional compiler
optimizations). However, the results could be incorrect or dirty."*

Escape hatch: `DISABLE_AGGRESSIVE_PTX_INSTRS=1`. Their V1 roadmap listed *"Fully remove
undefined-behavior PTX instructions"* as **unchecked** — i.e. never done in V1. [verified]

[inferred] **Do not port this to SM100 without validation.** The correctness argument is
explicitly Hopper-specific ("unified with L1 on Hopper"). Blackwell's cache hierarchy differs.
V2 moving to NCCL Gin plausibly retires the hack, but the `DISABLE_AGGRESSIVE_PTX_INSTRS`
build flag still exists in V2's env-var list for "legacy methods."

### DeepEP V2 (public release 2026-04-29) — the important rewrite

[verified] PR #605 and the current README. Headline: *"a complete refactoring of Expert
Parallelism — achieving extreme performance with several times fewer SM resources compared to
V1, while supporting significantly larger scale-up and scale-out domains."*

Changes that matter to us:

- **Backend switched from NVSHMEM to the NCCL Gin backend** — header-only, lightweight, *"able
  to reuse existing NCCL communicators."* Requires **NCCL ≥ 2.30.4** and **PyTorch ≥ 2.10**.
- **High-throughput and low-latency APIs unified** into a single `ElasticBuffer`, with a new
  GEMM layout.
- **Analytical SM and QP count calculation — no more auto-tuning.**
  `_buffer.get_theoretical_num_sms(num_experts, num_topk)`.
- **Up to EP2048** scale-up/scale-out domains.
- **For V3-like legacy training, SM usage reduced from 24 to 4–6** at equal or better perf.
- 0-SM Engram (RDMA), 0-SM PP (RDMA), 0-SM CP (Copy Engine) — experimental.
- **Handle caching for decode**: pass `cached_handle` to `dispatch()` to *"skip layout
  recomputation and CPU sync"* when gating is unchanged. [verified]

**Negative results / regressions they state plainly** [verified]:
- *"Buffer size consumption is larger than V1"*
- ***"0 SM RDMA low-latency EP is no longer supported"*** — the hook trick above is gone in V2.
- Engram, PP, CP are experimental.

**V2 performance — and this table contains the only first-party SM100 EP8 NVLink numbers**
[reported]. Config: 8K tokens/batch, 7168 hidden, top-8, FP8 dispatch, BF16 combine.

| Arch | NIC | Topo | Dispatch BW | Combine BW | #SMs |
|---|---|---|---|---|---|
| SM90 | CX7 | EP 8×2 | 90 GB/s (RDMA) | 81 GB/s | 12 |
| SM90 | CX7 | EP 8×4 | 61 GB/s (RDMA) | 61 GB/s | 6 |
| SM100 | CX7 | EP 8×2 | 90 GB/s (RDMA) | 91 GB/s | 12 |
| **SM100** | **N/A** | **EP 8 (NVLink)** | **726 GB/s** | **740 GB/s** | **64 (max perf)** |
| **SM100** | **N/A** | **EP 8 (NVLink)** | **643 GB/s** | **675 GB/s** | **24 (min #SM)** |

*"Comparing with V1, V2 achieves up to 1.3× peak performance, while saving up to 4× SM count."*
Bandwidths are **logical** (include local-rank traffic).

[inferred] **This is our exact topology** (single node, EP8, NVLink5). It tells us the SM/BW
tradeoff curve: 24 SMs buys 88% of the 64-SM bandwidth. With 148 SMs on a B200, burning 64
(43%) on comms is unacceptable at C1; 24 (16%) is arguable; Hybrid-EP's 16 is better still.

### Network configuration guidance (directly actionable)

[verified] From the V2 README:
- **Traffic isolation** via InfiniBand Virtual Lanes; V2 exposes `sl_idx` argument or
  `EP_OVERRIDE_RDMA_SL` env var. Segregate EP workloads from other workloads.
- **Adaptive routing**: V2 says *"Even though adaptive routing introduces additional latency, we
  still recommend enabling it under all network load conditions."* — note this **reverses V1's
  advice**, which was "enable AR under heavy load, use static routing under light load." A
  documented change of mind. [verified]
- **Congestion control**: *"disabled because it hurts maximum bandwidth."* (V1 wording was
  "we have not observed significant congestion in our production environment.")
- **PCI atomics**: `sudo mlxconfig -y -d mlx5_$i set PCI_ATOMIC_MODE=4` to improve RDMA atomic
  performance.

### Environment variables worth knowing (V2) [verified]

`EP_BUFFER_DEBUG`, `EP_AVOID_RECORD_STREAM`, `EP_NUM_TOPK_IDX_BITS`, `EP_NIC_NAME` (default
`mlx5_0`), `EP_OVERRIDE_RDMA_SL`, `EP_DISABLE_GIN`, `EP_JIT_CACHE_DIR` (default `$HOME/.deep_ep`),
`EP_JIT_PTXAS_CHECK` (*"assert no local memory usage in compiled kernels"* — a nice CI gate),
`EP_JIT_DUMP_SASS`, `EP_USE_NVIDIA_TOOLS`. Four are **persistent** (baked in at build time):
`EP_JIT_CACHE_DIR`, `EP_JIT_PRINT_COMPILER_COMMAND`, `EP_NUM_TOPK_IDX_BITS`, `EP_NCCL_ROOT_DIR`.

### The experimental branches — where the B200 work lives

The main README lists them [verified]:

- **`hybrid-ep`** — *"A new backend implementation using TMA instructions for minimal SM usage
  and larger NVLink domain support; fine-grained communication-computation overlap for
  single-batch scenarios; PCIe kernel support for non-NVLink environments; **NVFP4 data type
  support**."* The phrase *"single-batch scenarios"* is aimed at exactly our C1 problem.
- **`nvDev`** — *"V2-based branch with the latest CUDA features, such as **Compute Fabric
  Transport (CFT)** that brings better latency on small token sizes."*
- **Zero-copy** (PR #453, by Tencent Network Platform Dept) — removes the tensor↔comm-buffer copy.
- **AntGroup-Opt**: `Normal-SMFree` (#347) removes SMs from the RDMA path entirely;
  `LL-SBO` (#483) overlaps the down-GEMM with combine-send via a signalling mechanism;
  `LL-Layered` (#500) rail-optimized forwarding.

**Hybrid-EP B200 numbers** [reported], from `docs/README_Hybrid-EP.md` on that branch. Config:
B200, 4096 tokens, hidden 7168, topk 8, uniform-random router, 8 local experts, 8 ranks:

| Impl | Measurement | SMs | Dispatch (FP8) | Dispatch (BF16) | Combine |
|---|---|---|---|---|---|
| DeepEP | Torch API | 16 | 246 | 348 | 302 |
| DeepEP | Torch API | 24 | 349 | 494 | 420 |
| DeepEP | Torch API | 32 | 443 | 619 | 524 |
| DeepEP | Torch API | 44 | 544 | 640 | 577 |
| DeepEP | Torch API | 48 | 554 | 646 | 586 |
| **Hybrid-EP** | **Torch API** | **16** | **409.71** | **535.94** | **530.86** |
| **Hybrid-EP** | **kernel time only** | **16** | **599.27** | **734.95** | **673.84** |

[inferred] The gap between "Torch API" and "kernel only" at 16 SMs (409 vs 599 GB/s FP8) is
**~32% lost to host/framework overhead**, not to the network. At C1 that overhead is the thing
to kill, and it corroborates the V4 report's Host Codegen work. Also note DeepEP-baseline at
16 SMs is only 246 GB/s FP8 — so a naive low-SM DeepEP config on B200 is *bad*, and the naive
fix (throw 48 SMs at it) costs a third of the GPU.

Chunk-size tuning knobs on that branch [verified]: `NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API`,
`..._COMBINE_API`, `..._PREPROCESSING_API`, default 64, *"when the fused path is not used, you
can try increasing all three chunk-size variables to 128."*

---

## DeepGEMM — FP8/FP4 GEMM, and now Mega MoE

Repo: https://github.com/deepseek-ai/DeepGEMM (MIT). [verified] README, PR #304, PR #316.

### Scope creep, in a good way

It is no longer "a 300-line FP8 GEMM." Current self-description [verified]: *"GEMMs (FP8, FP4,
BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning
indexer, HyperConnection (HC), and more."* Fully JIT — *"no CUDA compilation during
installation."*

### Layouts (unchanged and still the right design)

- **Grouped, contiguous layout** — groups only the **M axis**; N and K fixed. *"This design is
  tailored for scenarios where experts in an MoE model share the same shape."* For training
  forward / inference **prefill**, tokens for all experts are concatenated into one tensor; each
  expert segment must be aligned to the GEMM M block size
  (`get_mk_alignment_for_contiguous_layout()`). [verified]
- **Grouped, masked layout** — for **decode with CUDA graphs**, *"when the CPU is unaware of
  the number of tokens each expert receives."* You pass a mask tensor; the kernel computes only
  valid portions. *"An example usage is to use the output of low-latency kernels from DeepEP as
  input."* [verified] This masked-grouped-GEMM + DeepEP-LL pairing is the canonical CUDA-graph-
  safe MoE decode path and we should be using it.
- K-axis-grouped API for MoE weight backward (`k_grouped_fp8_gemm_tn_contiguous`).

### Blackwell / SM100 status — good, and specific

[verified] Requirements: SM90 or SM100; **CUDA 12.9+ for SM100**; CUTLASS 4.0+; C++20.
Layout support differs by arch: *"the SM90 implementation supports only the NT memory layout
(row-major, col-major), the SM100 implementation supports all memory layouts (NT, TN, NN, TT)."*

**Scale-factor format differs by arch and this is a real porting trap** [verified]:
- SM90 requires scaling factors in **FP32**.
- SM100 requires scaling factors in **packed UE8M0** format, *"which packs 4 UE8M0 into a single
  `torch.int`."*

Helpers exist: `get_mn_major_tma_aligned_packed_ue8m0_tensor`,
`transform_sf_into_required_layout`, `get_tma_aligned_size`. Note our GLM-5.2 FP8 build and
DeepSeek's own V3.2/V4 configs both declare `"scale_fmt": "ue8m0"` — the ecosystem has
standardized on this for Blackwell. [verified from the HF configs]

### Mega MoE (released 2026-04-16, PR #304) — the headline

[verified] *"Mega MoE fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2
(FP8xFP4), and EP combine into a single mega-kernel, overlapping NVLink communication and
tensor core computation. It requires multi-process launch with symmetric memory."*

API [verified]:
```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts,
             num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)  # PyTorch >= 2.9
t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)  # FP4 + UE8M0 SF
buffer.x[:n].copy_(x_fp8); buffer.x_sf[:n].copy_(x_sf)
buffer.topk_idx[:n].copy_(topk_idx); buffer.topk_weights[:n].copy_(topk_weights)
deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, buffer)
```
*"Only FP8 × FP4 MoE is supported."* `DG_COMM_KERNEL_DEBUG=1` zeroes the symmetric buffer before
each call for debugging.

**Benchmarks** [reported], PR #316, EP8, averaged over 8 ranks. The model names are the giveaway
that these are V4 shapes:

DeepSeek-V4-Flash (256 experts, top-6, hidden 4096, intermediate 2048):

| Batch (tokens/rank) | Time (µs) | TFLOPS | Global mem (GB/s) | Interconnect (GB/s) | Speedup vs legacy |
|---|---|---|---|---|---|
| **1** | **56.5** | 5 | 1311 | 1 | **1.96×** |
| 512 | 146.5 | 1056 | 3192 | 266 | 1.73× |
| 8192 | 1283.1 | 1928 | 998 | 499 | 1.56× |
| 32768 | 4855.5 | 2038 | 794 | 529 | 1.62× |

DeepSeek-V4-Pro (384 experts, top-6, hidden 7168, intermediate 3072):

| Batch | Time (µs) | TFLOPS | Global mem (GB/s) | Interconnect (GB/s) | Speedup vs legacy |
|---|---|---|---|---|---|
| **1** | **108.1** | 7 | 1758 | 1 | **1.61×** |
| 512 | 369.6 | 1098 | 4619 | 182 | 1.54× |
| 8192 | 2818.5 | 2304 | 1094 | 393 | 1.50× |
| 32768 | 10655.2 | 2438 | 692 | 417 | 1.54× |

**Config asymmetries to state honestly**: "speedup vs legacy" is against **DeepSeek's own prior
DeepEP+DeepGEMM path**, not against SGLang/TRT-LLM/vLLM. It is FP8×FP4 only. It requires
symmetric memory and multi-process launch. And DeepSeek label it *"still under development."*

Critically, from PR #316's own notes [verified]: *"The code is specifically optimized for
Blackwell (SM100) with TMEM features; SM90 adaptation shows diminished gains without TMEM,
requiring alternative approaches like FP4 EP V2 + FP8 DeepGEMM + PDL instead."* — **the fusion
win is a Blackwell win.** That is our hardware.

[inferred] The batch-size-1 rows are the most valuable numbers in this document for objective
one. 1.96×/1.61× at BS=1 on a 256/384-expert top-6 MoE, EP8, on SM100. Our MoE expert GEMMs are
19.4% of C1 and our collectives 19.6% — a fused megakernel attacks both simultaneously, because
the dispatch and combine stop being separate collectives with their own rank-arrival barriers.
The 47%-of-collectives rank-arrival skew is partly an artifact of *having discrete collective
boundaries at all*; fusing them removes the boundary.

### Other DeepGEMM items

- **V3.2 indexer kernels** (2025-09-28, PR #200): `fp8_mqa_logits` (non-paged, prefill) and
  `fp8_paged_mqa_logits` (paged, decode). Semantics [verified]:
  `out[i,j] = sum_h relu(q[i,h,:] · (kv[j,:]·sf[j])) * weights[i,h]`, iterating
  `j ∈ [cu_seq_len_k_start[i], cu_seq_len_k_end[i])`, with `clean_logits` controlling whether
  unfilled logits become `-inf`. **This is exactly our DSA indexer kernel** (5.8% of C1) and it
  is MIT-licensed and SM100-capable. PR #304 adds an **FP4 Indexer with larger MTP support.**
- **PDL** is now a first-class toggle: `deep_gemm.set_pdl(True)`.
- **Dynamic swap A/B** — *"much faster MoE GEMM"* by dynamically choosing matrix orientation.
- Tuning knobs [verified]: `set_num_sms` / `get_num_sms`, `set_tc_util` (an *approximated tensor
  core utilization ratio* — useful for leaving headroom for a concurrent comm kernel),
  `set_block_size_multiple_of`, `set_ignore_compile_dims`.
- **NVRTC negative result** [verified]: added 2025-05-07 for 10× compile speedup
  (`DG_JIT_USE_NVRTC=1`), then **disabled in the 2025-07-20 refactor**. Also: *"As NVCC 12.9 will
  automatically do the FFMA interleaving, all post optimizations will be no longer supported"* —
  i.e. their famous SASS post-processing FFMA-interleave hack is **dead**, obsoleted by the
  compiler. Do not reimplement it.
- Peak claim: **1550 TFLOPS on H800** (2025-04-18). [reported]

---

## DualPipe, EPLB, LPLB — scheduling and load balancing

### DualPipe [verified] https://github.com/deepseek-ai/DualPipe

Training-only (bidirectional PP with full fwd/bwd overlap). Bubble comparison table:

| Method | Bubble | Params/device | Activations/device | #Devices |
|---|---|---|---|---|
| 1F1B | (PP−1)(F+B) | 1× | PP | PP |
| ZB1P | (PP−1)(F+B−2W) | 1× | PP | PP |
| DualPipe | (PP/2−1)(F&B+B−3W) | **2×** | PP+1 | PP |
| DualPipeV | (PP/2−1)(F&B+B−3W) | 2× | PP+1 | **PP/2** |

DualPipeV is credited to Sea AI Lab's "Cut-in-half" blog post. **Not transferable to us** —
we have no pipeline parallelism at TP8 single-node. Listed for completeness.

### EPLB — the redundant-experts placement algorithm [verified]

https://github.com/deepseek-ai/eplb — `eplb.py` read in full; it is ~130 lines.

Two policies:
- **Hierarchical** (when `num_nodes | num_groups`): pack expert *groups* to nodes evenly →
  replicate experts *within* each node → pack replicas to GPUs. *"can be used in prefilling
  stage with a smaller expert-parallel size."* Exploits V3's group-limited routing to keep
  same-group experts on one node and cut inter-node traffic.
- **Global**: replicate globally ignoring groups, then pack. *"can be adopted in decoding stage
  with a larger expert-parallel size."*

The two primitives, read from source:

```python
def replicate_experts(weight, num_phy):        # minimize max load per replica
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices   # greedy argmax of load/replicas
        phy2log[:, i] = redundant_indices
        logcnt[arangen, redundant_indices] += 1

def balanced_packing(weight, num_packs):       # LPT greedy, fixed cardinality
    indices = weight.sort(-1, descending=True).indices
    for group in indices[i]:
        pack = min((p for p in range(num_packs) if pack_items[p] < groups_per_pack),
                   key=pack_weights.__getitem__)
```

So: **replication is greedy on `load/replica_count`** (a classic minimize-makespan heuristic),
and **packing is longest-processing-time-first with a cardinality constraint**. Load estimation
is explicitly out of scope — *"A common method is to use moving average of historical
statistics."* In production the redundant-expert set is recomputed **every ~10 minutes**
[verified, V3 tech report].

Interface: `phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)`.

### LPLB — the successor nobody talks about [verified]

https://github.com/deepseek-ai/LPLB (Nov 2025). *"Linear-Programming-Based Load Balancer."*

Where EPLB handles **static** imbalance (consistently hot experts), LPLB targets **per-batch
fluctuation from small-batch randomness**. Mechanism: each redundant expert forms an **edge**
between two GPUs; the edge's capacity is the token count assigned to that redundant expert this
batch; an **LP redistributes tokens along edges to minimize imbalance within the EP group**.
The embedded solver is a **single-SM Interior Point Method** built on NVIDIA **cuSolverDx** and
**cuBLASDx**. Expert *selection* for replication still comes from EPLB (reordering only).
Real-time load stats are synchronized over **NVLink/NVSHMEM instead of
`torch.distributed.all_reduce`**.

Topologies: **Cube** (subset of GPUs, cube graph with diagonals, ≥2 experts/GPU, *"ideal for
balancing within an 8-GPU EP subgroup"*), **Hypercube** (no diagonals, needs 16 GPUs),
**Torus** (one replica on a neighbour GPU in-node, one on a neighbour node).

**Stated limitations, verbatim** [verified] — unusually candid and directly relevant:
- *"The current planner balances only total token count, not accounting for non-linearity in
  grouped matrix multiplication time costs, which may lead to suboptimal performance."*
- *"The solver takes ~100 µs for intra-node optimization (longer for inter-node), which may be
  non-negligible for small batches."*
- *"Under extreme global load imbalance, LPLB may perform worse than EPLB."*
- *"LPLB is currently in the early research stage, and performance improvements are still under
  evaluation."*

[inferred] **Cube is designed for exactly our shape** (8-GPU EP subgroup). But the 100 µs solver
latency is **36× our entire 2.74 ms TPOT budget divided across layers** — unusable inline at C1.
At C64 with a 40.8k tok/s aggregate it may amortize. And the honest read on C1: with 8 active
experts routed across 8 ranks and batch size 1, **the load is 8 tokens spread over 8 ranks —
some ranks get zero work.** No load balancer fixes that; only fusing the MoE into a megakernel
(Mega MoE) or accepting the imbalance does. Our 47%-of-collectives rank-arrival skew at C1 is
substantially *this*, and EPLB/LPLB are the wrong tool for it.

---

## 3FS and smallpond — storage

[verified] https://github.com/deepseek-ai/3fs

Fire-Flyer File System: disaggregated, **CRAQ** (Chain Replication with Apportioned Queries) for
strong consistency, stateless metadata over **FoundationDB**, FUSE + a zero-copy **USRBIO** API.

Numbers [reported]:
- **6.6 TiB/s aggregate read** on a 180-storage-node cluster (each 2×200 Gb IB, 16×14 TiB NVMe),
  ~500+ clients at 1×200 Gb each, *"with background traffic from training jobs."*
- **GraySort**: 110.5 TiB across 8,192 partitions in **30 min 14 s** = **3.66 TiB/min**, on 25
  storage + 50 compute nodes.
- **KVCache**: peak read throughput **up to 40 GiB/s** for KV-cache clients (1×400 Gb NIC/node).

The KVCache use case is the one that matters: *"Provides a cost-effective alternative to
DRAM-based caching, offering high throughput and significantly larger capacity."* This is the
substrate behind the **56.3% on-disk KV cache hit rate** in the Day-6 economics.

[inferred] Not transferable as a system (we are single-node), but the *idea* is: 56.3% of input
tokens never got prefilled. For cost-per-user that dwarfs any kernel optimization. It does
nothing for the Artificial Analysis leaderboard, whose prompts are fresh.

---

## profile-data — the published traces

[verified] https://github.com/deepseek-ai/profile-data — PyTorch Profiler traces, viewable in
`chrome://tracing`.

| Trace | Config |
|---|---|
| Training | EP64, TP1, 4K seqlen, DualPipe fwd/bwd chunk pair, 4 MoE layers/chunk, PP comm excluded |
| Prefill | **EP32, TP1**, 4K prompt, **16K tokens/GPU batch**, two micro-batches overlapping compute and all-to-all, attention load balanced across the two micro-batches |
| Decode | **EP128, TP1**, 4K prompt, **128 requests/GPU**, two micro-batches; **all-to-all does not occupy SMs** |

Caveat they state up front [verified]: *"we simulate an absolutely balanced MoE routing strategy
for profiling."* So the traces show the *achievable* overlap, not the load-imbalanced reality
their three load balancers exist to fix. Read them as an upper bound.

---

## DeepSeek-V2 / V3 / R1 / V3.1 / V3.2 — the model-side techniques that are inference techniques

**Coverage note, stated honestly.** I did *not* fetch the DeepSeek-V2 paper (arXiv 2405.04434)
or the DeepSeek-R1 paper directly. MLA's inference-relevant numbers below come from the V3
report and the ISCA'25 paper, which restate and quantify them; that is sufficient for our
purposes and I flag the provenance rather than implying I read V2. **DeepSeek-R1 contains
essentially no inference-systems content** — it is an RL/reasoning-training paper (published in
*Nature* 645(8081):633–638, 2025) and its serving story is entirely the V3 infrastructure, which
is why the Day-6 economics post covers "V3 and R1" as one deployment. Anyone citing "the R1
paper" for inference engineering is citing the wrong document. **DeepSeek-V3.1 / V3.1-Terminus**
have no standalone technical report I could find; they exist as HF checkpoints and are described
only as the baseline that V3.2-Exp continued-trained from.

### MLA and the KV-cache table (V2 → V3, quantified in the ISCA'25 paper)

[verified] arXiv 2505.09343, Table 1 — KV cache per token at BF16:

| Model | KV cache/token | Multiplier |
|---|---|---|
| **DeepSeek-V3 (MLA)** | **70.272 KB** | 1× |
| Qwen-2.5 72B (GQA) | 327.680 KB | 4.66× |
| LLaMA-3.1 405B (GQA) | 516.096 KB | 7.28× |

And Table 2, training cost per token at 4096 seqlen: DeepSeek-V2 MoE 155 GFLOPS, **DeepSeek-V3
MoE 250 GFLOPS**, Qwen-72B dense 394, LLaMa-405B dense 2448.

### Auxiliary-loss-free load balancing (V3) [verified, arXiv 2412.19437]

A per-expert **bias term `b_i` added to affinity scores for top-K selection only**:

```
g'_{i,t} = s_{i,t}   if  s_{i,t} + b_i ∈ TopK({s_{j,t}+b_j}, K_r)
         = 0         otherwise
```

*"Note that the bias term is only used for routing. The gating value, which will be multiplied
with the FFN output, is still derived from the original affinity score."* At the end of each
training step, `b_i` decreases by γ if the expert is overloaded and increases by γ if
underloaded (γ = "bias update speed"). A **complementary sequence-wise auxiliary loss** with
small α prevents extreme within-sequence imbalance. V3 switched from V2's softmax to
**sigmoid** affinity with normalization over selected scores.

This survives into V3.2 (`"topk_method": "noaux_tc"`) and **into V4** (`"topk_method":
"noaux_tc"`, but with `"scoring_func": "sqrtsoftplus"` replacing sigmoid). [verified from HF configs]

[inferred] The inference-side consequence: a model trained this way has **flatter expert load at
serving time**, which is why DeepSeek can get away with only 32 redundant experts. If GLM-5.2
was trained with an aux-loss instead, our expert load skew may be structurally worse and EPLB
buys correspondingly more.

### Node-limited routing (V3) — a communication-driven *architecture* decision [verified]

From the ISCA paper: H800 scale-up:scale-out bandwidth ratio is **~4:1** (NVLink 200 GB/s, of
which ~160 GB/s achievable; each 400 Gb IB NIC 50 GB/s nominal, ~40 GB/s effective). So they
**group 256 experts into 8 groups of 32, deploy one group per node, and algorithmically cap each
token at 4 nodes.** Tokens for the same node are sent once over IB and NVLink-forwarded,
deduplicating IB traffic from `8t` to `Mt` with `M < 8`.

[inferred] **Anti-transferable to us and worth saying so.** On 8×B200 with NV18 all-to-all
NVLink5 and no inter-node hop, node-limited routing is meaningless — the whole EP domain is one
NVLink domain. Interestingly, **V4 appears to have dropped it**: the V4 configs have no `n_group`
or `topk_group` keys, unlike V3.2's `n_group: 8, topk_group: 4`. [verified from HF configs]
That is consistent with a bet on large scale-up domains (NVL72-class), which is exactly the
regime we are in.

### MTP as speculative decoding (V3) [verified]

*"DeepSeek-V3 predicts the next 2 tokens through the MTP technique... the acceptance rate of the
second token prediction ranges between **85% and 90%** across various generation topics... This
high acceptance rate enables DeepSeek-V3 to achieve a significantly improved decoding speed,
delivering **1.8 times TPS**."* (ISCA paper restates it as 80–90% / 1.8×.)

Also, the throughput argument for MTP [verified]: *"by predicting multiple tokens per step, MTP
increases the inference batch size, which is crucial for boosting EP computational intensity and
hardware utilization."* — MTP is not only a latency trick; it *raises arithmetic intensity*.
Which is precisely the `s_q` term in the FlashMLA compute-bound condition.

`num_nextn_predict_layers: 1` persists in V3.2 **and V4**. [verified from HF configs]

### FP8 training details that constrain inference numerics [verified]

- Tile-wise **1×128** quantization for activations; block-wise **128×128** for weights.
- Hopper FP8 accumulation: *"After aligning 32 mantissa products by right-shifting based on the
  maximum exponent, the Tensor Core only maintains their highest 13 fraction bits for addition...
  Addition results are accumulated to FP22 registers (1 sign, 8 exponent, 13 mantissa)."* (The V3
  report says 14 bits; the ISCA paper says 13. **The two DeepSeek papers disagree**; I report
  both rather than pick.) Their mitigation: promote to FP32 in CUDA cores every `N_C = 128`
  accumulation steps.
- A **customized E5M6 format** for the activations feeding the Linear after attention, with
  **round-scaled (power-of-2) scaling factors** to avoid extra quantization error on the
  1×128 → 128×1 transposition in backward.
- Relative accuracy loss vs BF16 **below 0.25%**, validated on 16B and 230B V2 models first.

### LogFMT — a fully-documented negative result [verified]

Logarithmic Floating-Point Format, n bits, leading bit = sign. Per 1×128 tile: take
`log(abs(x))`, find min/max, encode min as `S.00…01`, max as `S.11…11`, step
`= (max−min)/(2^{n−1}−2)`, zero as `S.00…00`. Two findings: *"it is important to round in the
original Linear space, instead of the Log space, for the unbiased activation quantization"*, and
they constrain `min > max − log(2^32)` (≈E5 range).

Result: **LogFMT-8Bit beats E4M3 and E5M2 at the same bit width; LogFMT-10Bit ≈ BF16 combine.**

Why they killed it [verified]: *"Due to insufficient GPU bandwidth for log/exp operations and
excessive register pressure during encode/decode, if encode/decode operations are fused with
all-to-all communication, the overhead can be substantial (50%~100%). Therefore, although
experimental results validate the effectiveness of this format, we do not employ it eventually."*

[inferred] Do not chase log-domain combine compression on B200 without first measuring
SFU throughput for `ex2`/`lg2` against your combine volume. DeepSeek did the experiment and it
lost by 50–100%.

### DeepSeek Sparse Attention (V3.2-Exp / V3.2) — the mechanism, exactly

[verified] `DeepSeek_V3_2.pdf` in the V3.2-Exp repo, and arXiv 2512.02556 §2.

**Lightning indexer**:

```
I_{t,s} = Σ_{j=1..H_I}  w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )
```

*"We choose ReLU as the activation function for throughput consideration. Given that the
lightning indexer has a small number of heads and can be implemented in FP8, its computational
efficiency is remarkable."*

**Fine-grained token selection**: `u_t = Attn(h_t, {c_s | I_{t,s} ∈ Top-k(I_{t,:})})`.

**Instantiated under MLA in MQA mode** — critical kernel-level reason given [verified]: *"At the
kernel level, each key-value entry must be shared across multiple queries for computational
efficiency. Therefore, we implement DSA based on the MQA mode of MLA, where each latent vector
will be shared across all query heads of the query token."*

V3.2 config values [verified]: `index_head_dim: 128`, `index_n_heads: 64`, `index_topk: 2048`,
`num_attention_heads: 128`, `kv_lora_rank: 512`, `qk_rope_head_dim: 64`, `num_hidden_layers: 61`,
`max_position_embeddings: 163840`.

**The training recipe** (this is the part almost nobody reproduces correctly) [verified]:

| Stage | What is trained | LR | Steps | Batch | Tokens |
|---|---|---|---|---|---|
| **Dense warm-up** | indexer only; **all other params frozen**; attention stays **dense** | 1e-3 | 1,000 | 16 seqs × 128K | **2.1 B** |
| **Sparse training** | all params; top-k selection on, k=2048 | 7.3e-6 | 15,000 | 480 seqs × 128K | **943.7 B** |

Indexer target: sum the main attention scores across all heads, **L1-normalize along the
sequence dimension** → target distribution `p_{t,:}`; train with
`L_I = Σ_t D_KL(p_{t,:} ‖ Softmax(I_{t,:}))`. In the sparse stage the KL is restricted to the
selected set `S_t`.

**And the key isolation trick** [verified]: *"we detach the indexer input from the computational
graph for separate optimization. The training signal of the indexer is from only `L_I`, while
the optimization of the main model is according to only the language modeling loss."*

Cost claims [verified]: complexity drops from O(L²) to O(Lk); *"Although the lightning indexer
still has a complexity of O(L²), it requires much less computation compared with MLA."* Costs
were *"estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price
of 2 USD per GPU hour"* — figures only, no table; the published price cut is the harder number.
Parity table shows essentially no benchmark regression vs V3.1-Terminus.

**A short-context optimization that is easy to miss** [verified]: *"for short-sequence
prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher
efficiency under short-context conditions."* — i.e. they run **two different prefill
implementations** and switch on sequence length. Given the AA methodology's ~10k input tokens,
[inferred] this is directly relevant: at 10k tokens with topk=2048, sparsity buys ~5×, but the
gather/indexer overhead may not pay for itself. **Measure a dense-masked prefill path at 10k.**

### The RoPE interleaving bug — a real, published gotcha

[verified] V3.2-Exp README, 2025.11.17: *"We have identified that previous versions of the
inference demo code contained an implementation discrepancy in Rotary Position Embedding (RoPE)
within the indexer module, potentially leading to degraded model performance. Specifically, the
input tensor to RoPE in the indexer module requires a **non-interleaved** layout, whereas RoPE
in the MLA module expects an **interleaved** layout."*

[inferred] If our GLM-5.2 DSA port was written against pre-Nov-2025 DeepSeek reference code, or
by analogy to it, **check this**. It is a silent quality regression, not a crash.

### Deployment guidance DeepSeek themselves publish

[verified] V3.2-Exp README SGLang launch line:
```
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention
```
`--tp 8 --dp 8 --enable-dp-attention` on an 8-GPU node. **DeepSeek's own recommended SGLang
configuration for a DSA-MLA MoE on 8 GPUs is DP attention, not TP attention.** Docker images
`lmsysorg/sglang:dsv32` (H200), `dsv32-rocm` (MI350), `dsv32-a2`/`dsv32-a3` (NPUs).
vLLM had day-0 support.

Kernel sourcing guidance [verified]: *"For TileLang kernels with better readability and
research-purpose design, refer to TileLang `examples/deepseek_v32`. For high-performance CUDA
kernels, indexer logit kernels (including paged versions) are available in DeepGEMM PR #200.
Sparse attention kernels are released in FlashMLA PR #98."*

---

## The hardware co-design paper (ISCA'25): "Insights into DeepSeek-V3"

[verified] arXiv 2505.09343, full HTML read. This is the single densest source of bottleneck
analysis and it is written as an explicit wishlist to NVIDIA.

### The theoretical TPOT ceiling calculation — reproduce this for our hardware

[verified] Assume each device holds one expert and processes ~32 tokens. Per-layer, two
all-to-alls:

```
Comm time = (1 Byte + 2 Bytes) × 32 × 9 × 7K / 50 GB/s = 120.96 µs
```
(1 B dispatch FP8 + 2 B combine BF16; 9 = 8 routed + 1 shared; 7K hidden; CX7 400 Gb ≈ 50 GB/s.)

Under dual-microbatch overlap, per layer `2 × 120.96 = 241.92 µs`; × 61 layers = **14.76 ms
TPOT ≈ 67 tok/s** — the theoretical ceiling for an IB-connected EP deployment.

Then the counterfactual [verified]: with GB200 NVL72 at 900 GB/s unidirectional,
`3 × 32 × 9 × 7K / 900 = 6.72 µs`, giving *"a theoretical upper limit of over 0.82 ms TPOT,
approximately **1200 tokens per second**. While this figure is purely theoretical and has not
been empirically validated..."*

[inferred] **Run this arithmetic for us.** Single-node 8×B200, NVLink5 all-to-all. Using DeepEP
V2's own measured SM100 EP8 NVLink figure of ~726 GB/s dispatch / 740 GB/s combine and our
model's shape (256 experts, top-8 → 9 with shared, hidden dim per GLM-5.2), the per-layer comm
term is on the order of a few µs, i.e. **communication is not our TPOT ceiling; it is our
19.6% because of skew and launch overhead, not bandwidth.** That reframes the collectives
problem entirely: chase *arrival synchronization and kernel-launch cost*, not bandwidth.

### The SM tax, stated in numbers

[verified] *"during training, up to 20 of the SMs on the H800 GPU are allocated for
communication-related operations"* (of 132). The five SM tasks they enumerate:
1. Forwarding data between IB and NVLink domains (aggregating IB traffic per node)
2. Transporting data between RDMA buffers and I/O buffers
3. **Reduce operations for EP all-to-all combine**
4. Managing fine-grained memory layouts for chunked transfers
5. **Data type casts before/after all-to-all**

And the production countermeasure [verified]: *"To maximize throughput in online inference, we
perform EP all-to-all communication entirely through NIC RDMA, avoiding SM resource contention
and improving compute efficiency."*

### IBGDA [verified]

*"IBGDA addresses this issue by allowing the GPU to directly fill the WR content and write to
the RDMA doorbell MMIO address. By managing the entire control plane within the GPU, IBGDA
eliminates the significant latency overhead associated with GPU-CPU communication. Moreover,
when sending a large number of small packets, the control plane processor can easily become a
bottleneck."* Used by DeepEP and named in the V3 report's decode section.

### Measured latency table (64 B transfer, CPU-side end-to-end) [reported]

| Link layer | Same leaf | Cross leaf |
|---|---|---|
| RoCE | 3.6 µs | 5.6 µs |
| InfiniBand | 2.8 µs | 3.7 µs |
| NVLink | 3.33 µs | — |

[inferred] NVLink at 3.33 µs is *slower than same-leaf IB* for a 64 B message. Small-message
latency on the scale-up fabric is not free. At C1 our collectives are small messages.

### Multi-Plane Fat-Tree (MPFT) [reported]

Training-metric comparison MPFT vs multi-rail fat-tree (MRFT): tokens/day 272.80 B vs 272.52 B;
time/step 19.926 s vs 19.946 s; **TFLOPS 432 non-causal / 385 causal, identical for both**;
MFU 43.73%/38.94% vs 43.68%/38.90%. I.e. **MPFT costs nothing in performance while being
cheaper**. Not transferable single-node; included because the MFU numbers (43.7% non-causal on
H800 FP8 training) are a useful public calibration point.

### The hardware wishlist (what they asked NVIDIA for, and what Blackwell delivered)

| Ask | Status on B200 | [label] |
|---|---|---|
| **Higher FP8 accumulation precision** (FP32 or configurable) | Improved on Blackwell; still not user-configurable | [inferred] |
| **Native fine-grained quantization** — Tensor Cores receive scaling factors, dequant inside TC | **Delivered**: they explicitly name *"NVIDIA Blackwell's support for microscaling data format"* as *"a notable industrial implementation of this approach"* | [verified] |
| **Fused FP8 cast + TMA access**, warp-level cast instruction, near-memory cast on HBM read | Partially | [inferred] |
| **Transposed GEMM** / direct transposed reads from shared memory before MMA | Partially | [inferred] |
| **Unified scale-up/scale-out network adapter**, single LID/IP with policy routing | Not in B200 | [inferred] |
| **Dedicated communication co-processor / I/O die** to offload the five SM tasks | Not in B200 | [inferred] |
| **Hardware forwarding/broadcast/reduce across scale-up+scale-out** | Partially (SHARP) | [inferred] |
| **Hardware synchronization primitives** — the **Region Acquire/Release (RAR)** proposal: receiver hardware keeps a bitmap tracking RNR memory-region state; acquire/release scoped to the RAR address range, eliminating sender-side fences | Not in B200 | [verified as a proposal] |
| **Dynamic NVLink/PCIe traffic prioritization** (expose PCIe traffic class to userspace) | Not exposed | [verified as an ask] |
| **CPU–GPU NVLink** instead of PCIe | Delivered on GB200/GB300 (not on our 8×B200 SXM) | [inferred] |
| **In-network multicast for dispatch, in-network aggregation for combine** | *"due to the small reduction scope and imbalanced workload in EP combine, implementing in-network aggregation in a flexible manner is challenging"* — their own caveat | [verified] |
| **Native LogFMT compress/decompress units** | Not present | [verified as an ask] |
| **DRAM-stacked accelerators (SeDRAM-style), System-on-Wafer** | Not present | [verified as an ask] |

Also a concrete CPU-side requirement worth checking on our two NUMA nodes [verified]:
*"latency-sensitive tasks such as kernel launches and network processing demand high single-core
CPU performance, typically requiring base frequencies **above 4 GHz**. Furthermore, modern AI
workloads require sufficient CPU cores per GPU to prevent control-side bottlenecks. For
chiplet-based architectures, additional cores are needed to support **cache-aware workload
partitioning and isolation**."* [inferred] At 2.74 ms TPOT on a 2-NUMA box, CPU pinning and NUMA
affinity for the launch threads is not optional.

### Robustness, stated as a limitation [verified]

*"High-performance interconnects (e.g., IB and NVLink) are prone to intermittent
disconnections... This is especially harmful in communication-heavy workloads like EP, where
even brief interruptions may lead to significant performance drops or job failures."* Plus
silent data corruption: *"Current mitigation strategies rely on application-level heuristics,
which are insufficient."* They ask vendors for diagnostic toolkits shipped as standard.

---

## DeepSeek-V4 — released, documented, and full of inference engineering

**It exists.** [verified]

| Model | Total | Active | Context | Precision | HF |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash-Base | 284 B | 13 B | 1 M | FP8 Mixed | `deepseek-ai/DeepSeek-V4-Flash-Base` |
| **DeepSeek-V4-Flash** | 284 B | 13 B | 1 M | **FP4 + FP8 Mixed** | `deepseek-ai/DeepSeek-V4-Flash` |
| DeepSeek-V4-Pro-Base | 1.6 T | 49 B | 1 M | FP8 Mixed | `deepseek-ai/DeepSeek-V4-Pro-Base` |
| **DeepSeek-V4-Pro** | 1.6 T | 49 B | 1 M | **FP4 + FP8 Mixed** | `deepseek-ai/DeepSeek-V4-Pro` |

*"FP4 + FP8 Mixed: MoE expert parameters use FP4 precision; most other parameters use FP8."*
MIT licensed. Tech report: arXiv **2606.19348**, *"DeepSeek-V4: Towards Highly Efficient
Million-Token Context Intelligence."* Pre-trained on **>32T tokens**. Later HF checkpoints exist
(`DeepSeek-V4-Flash-0731`, `DeepSeek-V4-Pro-0813`) plus `-DSpark` variants.

Headline efficiency claim [verified, abstract]: *"In the one-million-token context setting,
DeepSeek-V4-Pro requires only **27% of single-token inference FLOPs** and **10% of KV cache**
compared with DeepSeek-V3.2."* And in §2.3.4: KV cache reduced *"to approximately **2%** of"*
a BF16 GQA8 head-dim-128 baseline at 1M context.

### The architecture, read from `config.json` and the reference implementation

[verified] Both configs and `inference/model.py` + `inference/kernel.py` read in full.

| Key | V4-Flash | V4-Pro |
|---|---|---|
| `num_hidden_layers` | 43 | 61 |
| `hidden_size` | 4096 | 7168 |
| `n_routed_experts` / `num_experts_per_tok` | 256 / **6** | 384 / **6** |
| `moe_intermediate_size` | 2048 | 3072 |
| `num_attention_heads` | 64 | 128 |
| `num_key_value_heads` | **1** (MQA) | **1** (MQA) |
| `head_dim` / `qk_rope_head_dim` | 512 / 64 | 512 / 64 |
| `sliding_window` | **128** | **128** |
| `index_n_heads` / `index_head_dim` / `index_topk` | 64 / 128 / **512** | 64 / 128 / **1024** |
| `o_groups` / `o_lora_rank` | 8 / 1024 | 16 / 1024 |
| `expert_dtype` | **`fp4`** | **`fp4`** |
| `quantization_config.scale_fmt` | `ue8m0` | `ue8m0` |
| `scoring_func` / `topk_method` | `sqrtsoftplus` / `noaux_tc` | same |
| `hc_mult` / `hc_sinkhorn_iters` | 4 / 20 | 4 / 20 |
| `num_hash_layers` | 3 | 3 |
| `num_nextn_predict_layers` | 1 | 1 |
| `max_position_embeddings` | 1,048,576 | 1,048,576 |
| `swiglu_limit` | 10.0 | 10.0 |

**`compress_ratios`** is the whole story. It is a per-layer list of length `L+1` (the extra entry
is the MTP block). Counted [verified]: Flash = `{4: 21, 128: 20, 0: 3}`; Pro = `{128: 31, 4: 30, 0: 1}`.
Reading `model.py`:

- **`ratio == 4` → CSA layer**: a `Compressor` with `overlap=True` **plus** an `Indexer` that
  selects `index_topk` compressed blocks.
- **`ratio == 128` → HCA layer**: a `Compressor` with `overlap=False` and **no indexer** — it
  attends to *all* compressed blocks (there are only seq/128 of them).
- **`ratio == 0` → pure sliding-window layer** (window 128; YaRN disabled, base `rope_theta`).

Every layer additionally runs a **128-token sliding window branch**, and the per-layer KV cache
is sized `window_size + max_seq_len // compress_ratio`.

### CSA — Compressed Sparse Attention [verified, tech report §2.3.1]

Two projections `C^a = H·W^{aKV}`, `C^b = H·W^{bKV}` and two gate projections `Z^a`, `Z^b`, plus
learnable positional biases `B^a, B^b ∈ R^{m×c}`. Each compressed entry pools **2m** source
entries with a softmax over all 2m gate scores:

```
[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}] = Softmax_row([Z^a + B^a ; Z^b + B^b])
C^Comp_i = Σ S^a_j ⊙ C^a_j  +  Σ S^b_j ⊙ C^b_j
```

The `b` window is offset by one block, so consecutive compressed entries **overlap** — smoother
boundaries — while still compressing sequence length to `1/m`. The reference `Compressor.forward`
implements exactly this, in **fp32** (*"compression need fp32"*), with an incremental decode path
using `kv_state` / `score_state` ring buffers and `should_compress = (start_pos+1) % ratio == 0`.

**The indexer now runs on compressed keys.** [verified] `K^IComp` is produced by *the same
compression operation*, then
`I_{t,s} = Σ_h w^I_{t,h} · ReLU(q^I_{t,h} · K^IComp_s)` over blocks `s < floor(t/m)`, and top-k
selects **compressed blocks**. Indexer queries come from the **same latent `c^Q_t`** used for the
attention queries (`W^{DQ}` shared, then `W^{IUQ}` vs `W^{UQ}`).

[inferred] **This is the biggest single idea for our DSA indexer (5.8% of C1).** Running the
indexer against `seq/4` compressed keys instead of `seq` raw keys cuts both the QK-score FLOPs
*and* the top-k candidate set by 4×. Our `index_topk_freq=4` is a *temporal* amortization of the
same cost; DeepSeek's is a *spatial* one, and the two compose.

The reference `Indexer` also shows two quantization details [verified]: its own `Compressor` is
constructed with `rotate=True`, meaning `rotate_activation()` (a **Hadamard transform**;
`fast_hadamard_transform` is in `requirements.txt`) is applied before `fp4_act_quant` — the
standard rotation-before-FP4 recipe to tame outliers. And the comment: *"use fp4 simulation for
q and kv in indexer."*

### HCA — Heavily Compressed Attention [verified §2.3.2]

Same pooling but with a **much larger `m'` (128), no overlap, and no sparse selection** — full
attention over the `seq/m'` compressed entries. At 1M context that is ~8k entries, cheap.

### The other four attention details [verified §2.3.3]

1. **Q and KV entry RMSNorm** per head immediately before core attention (*"avoids exploding
   attention logits"*). Visible in `model.py` as `q *= rsqrt(q.square().mean(-1)+eps)`.
2. **Partial RoPE on the last 64 dims** of queries and KV entries — **and, because the KV entry
   serves as both key and value, RoPE with position `−i` is applied to the core-attention
   output** so the output carries relative rather than absolute position. In code:
   `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` (the `inverse=True` flag).
3. **Sliding-window branch** of `n_win` uncompressed KV entries, concatenated with the compressed
   entries inside the same core attention, *"In order to strictly preserve causality... a query
   cannot access information from other tokens within its own compressed block."*
4. **Attention sink**: per-head learnable `z'_h`, with `exp(z'_h)` added to the softmax
   denominator, *"allows each query head to adjust its total attention scores to be not equal to
   1, and even to be near 0."* In code: `self.attn_sink = nn.Parameter(torch.empty(n_local_heads))`.

### Grouped output projection [verified §2.3.1]

Because `c·n_h` is huge (Pro: 512×128 = 65,536), the output projection is factored: split `n_h`
into `g` groups, project each `c·n_h/g` group down to `d_g < c·n_h/g`, then project the
concatenated `d_g·g` to `d`. Pro: g=16, d_g=1024 → 16×4096 → 16×1024 = 16,384 → 7,168. In code:
`einsum("bsgd,grd->bsgr", o, wo_a)` then `wo_b`.

[inferred] A dense 65,536×7,168 output projection would be ~470M params/layer; the grouped
low-rank version is ~67M+117M. **This is a dense-GEMM reduction technique** and our dense GEMM is
37.1% of C1. Worth evaluating whether GLM-5.2's o_proj could be similarly factored (requires
retraining — so a note for the next model, not a serving change).

### Efficiency levers, enumerated by DeepSeek [verified §2.3.4]

1. **Mixed KV storage: BF16 for RoPE dims, FP8 for the rest** — *"reduces the KV cache size by
   nearly half compared with pure BF16 storage."* (Same structure as FlashMLA's 656-byte format.)
2. **Indexer attention in FP4.**
3. *"relative to DeepSeek-V3.2, a **smaller attention top-k** is chosen"* — 512/1024 vs 2048 —
   *"thereby improving model efficiency on short- and medium-length texts."*
4. Compression itself.

### The V4 inference framework §3.5 — KV cache design for hybrid attention

[verified] *"Our inference framework largely inherits from that of DeepSeek-V3, with some
differences in KV Cache management."*

They state plainly that **hybrid attention breaks PagedAttention**: *"The hybrid attention
mechanism violates fundamental assumptions behind PagedAttention and its variants."* Two obstacles:
diverse cache policies (SWA), and alignment constraints imposed by high-performance kernels.
They name Jenga and Hymba as prior art that does not suffice.

Their two-part solution:

- **State cache** for SWA entries *and* uncompressed tail tokens. Rationale: *"Since SWA is
  designed to enhance performance under a limited KV cache size, it is reasonable to treat it,
  along with the uncompressed tail tokens from the compression branch, as a **state-space
  model**. The corresponding KV cache can thus be regarded as a sequence-specific state that
  depends solely on the current position."* Pre-allocated fixed-size pool, one block per request.
- **Classical KV cache** for compressed CSA/HCA entries, with a block covering
  **`lcm(m, m')` original tokens**, yielding `k1 = lcm/m` CSA tokens and `k2 = lcm/m'` HCA tokens
  per block. *"the number of original tokens per block can be any multiple of `lcm(m, m')`."*
  Plus explicit kernel co-design: *"padding blocks to align with cache lines can improve
  performance."*

### On-disk KV cache for V4 §3.5.2 — three strategies, with the tradeoff stated

[verified] Compressed CSA/HCA entries are stored wholesale; on a prefix hit they are read back
*"until the last complete compression block"*, and the tail incomplete block is recomputed.
SWA entries are the problem — *"their volume is approximately **8 times larger** than the
compressed CSA and HCA KV entries."* Three options:

| Strategy | Mechanism | Cost |
|---|---|---|
| **Full SWA caching** | store SWA KV for all tokens; on hit read only the last `n_win` | zero recompute, but *"inefficient for modern SSD-based storage systems — only a small subset of the stored SWA KV cache will be accessed for each hitting request, which leads to an unbalanced write-intensive access pattern"* |
| **Periodic checkpointing** | checkpoint last `n_win` every `p` tokens; recompute the tail | tunable storage↔compute via `p` |
| **Zero SWA caching** | store nothing; recompute | needs only the last `n_win · L` tokens recomputed for an L-layer model, because *"the SWA KV entry of each token depends on the SWA KV entries of only the most recent `n_win` tokens from the previous layer"* |

[inferred] That last observation — SWA state has a **cone of dependence of `n_win · L` tokens** —
is a neat, generalizable result. For V4-Pro that is 128 × 61 ≈ 7,808 tokens of recompute to
restore *all* SWA state. Cheap.

### Batch-invariant and deterministic kernels §3.3 — the most under-appreciated section

[verified] Goal: *"bitwise batch-invariant and deterministic kernels with minimal performance
overhead"* to guarantee *"bitwise alignment among pre-training, post-training, and inference."*

**The attention problem and their fix** [verified]: batch invariance forbids split-KV, but
abandoning split-KV causes *"severe wave-quantization problems."* Their answer is a **dual-kernel
strategy**: kernel 1 computes a whole sequence's attention **within a single SM** (high
throughput for fully-occupied waves); kernel 2 uses **multiple SMs for a single sequence** to
minimize the latency of the final partially-filled wave — *"For the bitwise identity of these two
kernels, we carefully design the calculation path of the second kernel to ensure its accumulation
order is the same as that of the first kernel. Additionally, the second kernel utilizes
**distributed shared memory within thread-block clusters**, enabling high-speed data exchange
across SMs. This dual-kernel method effectively confines the overhead of batch-invariant decoding
to be negligible."*

**GEMM**: *"Traditional cuBLAS library cannot achieve batch invariance. Therefore, we **replace
it end-to-end with DeepGEMM**."* And a negative result made good: *"for very small batch sizes,
conventional implementation usually employs split-k... split-k techniques cannot guarantee batch
invariance... Therefore, we abandon split-k in most scenarios, which, however, may cause
performance degradation. To address this, we introduce a set of optimizations that enable our
implementation of matrix multiplication to match or even surpass the performance of standard
split-k in most major scenarios."*

**Determinism** sources and fixes [verified]:
- Sparse attention backward `atomicAdd` → **per-SM accumulation buffers + global deterministic sum**
- MoE backward cross-rank writes → **token-order pre-processing per rank + buffer isolation across ranks**
- mHC's tiny GEMM (**output dimension only 24**) forces split-k → **output each split separately,
  deterministic reduction in a following kernel**

[inferred] The dual-kernel wave-quantization fix is our closest published analogue to the
**47%-of-collectives rank-arrival skew** problem. Rank arrival skew at C1 *is* tail-wave latency
variance across ranks. Their fix — a second kernel specialized for the ragged tail wave that
spreads one sequence over multiple SMs via DSM — is exactly the right shape of solution, and it
is a Blackwell-capable technique (B200 has thread-block clusters).

### TileLang and Host Codegen §3.2 [verified]

*"our elaborate model architecture would have resulted in hundreds of fine-grained Torch ATen
operators. We adopt TileLang to develop a set of fused kernels to replace the vast majority of
them."*

Three concrete contributions:
1. **Host Codegen** — co-generate the device kernel and a lightweight host launcher at the IR
   level, embedding dtype/rank/shape/stride metadata, lowered onto **TVM-FFI**. *"Our measurements
   show that CPU-side validation overhead drops from **tens or hundreds of microseconds to less
   than one microsecond per invocation**."*
2. **Z3 SMT solver integrated into TileLang's algebraic system**, translating integer expressions
   into **QF_NIA**, to strengthen layout inference, memory-hazard detection and bound analysis.
   *"Under reasonable resource limits, Z3 elevates overall optimization performance while
   restricting compilation time overhead to just a few seconds."* Improves vectorization, barrier
   insertion, code simplification.
3. **Numerical policy**: fast-math **disabled by default**; approximations opt-in via
   `T.__exp`, `T.__log`, `T.__sin`; IEEE intrinsics `T.ieee_fsqrt`, `T.ieee_fdiv`, `T.ieee_add`
   with explicit rounding; `T.annotate_layout` to pin lowering decisions for **bit-identical
   agreement with a reference CUDA implementation**.

[inferred] Item 1 is a top-5 steal for us. At 2.74 ms TPOT over 60+ layers with many small fused
ops, "tens to hundreds of µs of Python-side validation per launch" is catastrophic; if any of our
hot path goes through Python-side op wrappers, this is free milliseconds. And Hybrid-EP's 409 vs
599 GB/s (Torch API vs kernel-only) is independent evidence that framework overhead is ~32% at
low SM counts on B200.

### FP4 quantization-aware training §5.2.1 [verified]

**MXFP4** applied to (1) MoE expert weights and (2) *"the Query-Key (QK) path in the indexer of
CSA, where QK activations are **cached, loaded, and multiplied entirely in FP4**."*

Plus a distinct, very cheap win [verified]: *"we further quantize the index scores `I_{:,:}` from
FP32 to BF16 during this QAT process. This optimization achieves a **2× speedup for the top-k
selector**, while preserving a **99.7% recall rate** of KV entries."*

**The lossless FP4→FP8 dequantization trick** [verified] — this is elegant and reusable: FP32
master weights are quantized to FP4 then dequantized back to FP8 for compute. Because E4M3 has
2 more exponent bits than E2M1, *"as long as the ratio between the maximum and minimum scale
factors of the FP4 sub-blocks (**1×32** tiles) within each FP8 quantization block (**128×128**
tiles) does not exceed a certain threshold, the fine-grained scale information can be fully
absorbed by the extended dynamic range of FP8. We empirically verify that current weights satisfy
this condition. This allows the entire QAT pipeline to fully reuse the existing FP8 training
framework without any modification."* Backward uses the straight-through estimator against the
same FP8 weights, *"This also avoids the need to re-quantize transposed weights."*

For RL rollout and inference: *"we directly use **native FP4** quantized weights instead of
simulated quantization. This ensures that model behavior during sampling is fully consistent with
online deployment."*

[inferred] Directly relevant to our NVFP4 build. The 1×32-within-128×128 scale-ratio condition is
a **checkable property of our weights** and would let an NVFP4 build reuse an FP8 pipeline
end-to-end. And the FP32→BF16 index-score quantization (2× top-k, 99.7% recall) is a one-line
change to our indexer with a measurable ceiling.

### mHC — Manifold-Constrained Hyper-Connections [verified §2.2, `model.py`]

Replaces the residual stream with `hc_mult = 4` parallel copies. Per sub-layer:
`hc_pre` reduces 4→1 by a learned weighted sum; `hc_post` expands 1→4 using learned `post`
weights plus a `comb` mixing matrix. The weights come from a projection of the (flattened,
RMS-scaled) hyper-state through `hc_*_fn ∈ R^{(2+hc)·hc × hc·d}`, split into `pre`/`post`/`comb`
and passed through **`hc_split_sinkhorn`** — a TileLang kernel that sigmoids `pre`, doubles the
sigmoid for `post`, and Sinkhorn-normalizes `comb` (row softmax, then column normalize) for
**20 iterations**.

[inferred] This is a training-quality technique, not a serving one, and it *costs* us: it adds a
`(2+4)·4 = 24`-wide GEMM and a Sinkhorn kernel per sub-layer. DeepSeek call this out themselves
as the source of a determinism problem (output dim 24 forces split-k). **Relevant to us only
because Kimi K3 / Qwen3.8 / DeepSeek V4 may all ship hyper-connections**, in which case our
engine needs a fast fused Sinkhorn — and `TileKernels` has one (`tile_kernels/mhc/`), MIT
licensed.

### Hash routing — a small oddity worth flagging

[verified] `num_hash_layers: 3` and `Gate.hash = layer_id < args.n_hash_layers`. For the first 3
layers, expert indices are **looked up from a frozen `tid2eid` table indexed by token ID** rather
than computed from a router:

```python
self.tid2eid = nn.Parameter(torch.empty(vocab_size, n_activated_experts, dtype=torch.int32),
                            requires_grad=False)
indices = self.tid2eid[input_ids]
```

[inferred] For a serving engine this is a gift: for the first 3 MoE layers, **routing is known
from the input token IDs before the forward pass begins**. Dispatch metadata for those layers can
be computed on the host, or prefetched, entirely off the critical path — and at C1 that removes
three router GEMMs plus three top-k selections plus three routing-dependent stalls.

---

## DSpark / DeepSpec — speculative decoding, and the most transferable paper in the corpus

Paper: arXiv **2607.05147**, *"DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation"*, Peking University + DeepSeek-AI (author list includes Jiashi Li,
Damai Dai, Chengqi Deng, Zhean Xu, Runxin Xu, Yu Wu, Wenfeng Liang). [verified]
Code: https://github.com/deepseek-ai/DeepSpec (MIT). Checkpoints on HF.

**Why this is the single most relevant DeepSeek publication to objective two:** it is about
exactly our failure mode — *per-stream speed collapsing as concurrency rises* — and it was
deployed in **DeepSeek-V4 production, replacing MTP-1 two weeks after the V4-preview release.**

### Framing (equation 1)

`L = (T_draft + T_verify) / τ`. Three levers: lower `T_draft`, raise `τ`, or **reduce effective
`T_verify`**. Almost all prior work targets the first two; DSpark targets the third.

### Mechanism 1 — semi-autoregressive drafting

A **parallel backbone** (DFlash-style: 5 layers, target hidden states from layers
`[1, 9, 17, 25, 33]` concatenated, RMSNorm'd, projected, and **injected as extra K/V into every
draft layer**) produces all γ logits in one pass, so `T_draft` is ~independent of γ. Then a
**lightweight sequential head** adds a prefix-dependent transition bias:

```
p_k(v | x_0, x_<k) ∝ exp( U_k(v) + B_k(x_0, x_<k, v) )
```

The cheap instantiation is a **Markov head**: `B` depends only on the immediately preceding token
(`markov_rank: 256` in the released configs). A minor but useful backbone change: *"instead of
feeding an anchor token plus γ mask tokens and predicting only the mask positions, we treat the
anchor itself as the first prediction position, so γ input tokens (anchor + γ−1 masks) yield γ
draft logits. This reduces draft computation while maintaining similar draft quality."*

### Mechanism 2 — confidence head + hardware-aware prefix scheduler

**Confidence head**: `c_k = σ(wᵀ[h_k ; W_1[x_{k-1}]])`, trained against the *analytical* per-step
acceptance rate `c*_k = 1 − ½‖p^d_k − p^t_k‖_1` with BCE. Then **Sequential Temperature Scaling
(STS)**: because neural confidences are overconfident and the scheduler needs *absolute*
magnitudes (not just rankings), they calibrate the **cumulative product** `∏_{i≤k} c_i` left to
right by 1-D grid search on temperature to minimize Expected Calibration Error, freezing already-
calibrated positions. Temperature scaling is order-preserving, so rankings survive.

**The scheduler (Algorithm 1)**, which is the part we can implement without retraining anything:

```
for each request r:  a_{r,j} = ∏_{i≤j} c_{r,i}          # prefix survival probability
E = {(r,j) | a_{r,j} > 0}, sorted descending by a_{r,j}
B = R;  τ* = R;  Θ_best = R · SPS(R)
for (r,j) in E:
    ℓ_r = j;  B += 1;  τ* += a_{r,j}
    Θ = τ* · SPS(B)
    if Θ > Θ_best:  Θ_best = Θ; record ℓ*
    else: break                                          # early stop enforces causality
```

`SPS(B)` is the engine's **steps-per-second as a function of forward-pass batch size**,
*"profiled once during engine initialization and stored as a lightweight cost table."*
Objective `Θ = τ · SPS(B)`. Greedy is optimal-along-the-admission-path because `a_{r,j}` is
monotonically non-increasing in `j`, so global sorting automatically respects prefix dependencies.

**The losslessness argument, and why it matters**: strict speculative decoding requires the
**non-anticipating property** — admission must not depend on future candidates. Because the
confidence head uses the previously *sampled* token as a Markov feature, computing `a_{r,k+1}`
requires having instantiated `x_{r,k}`; a retrospective global search would leak it. The
`break` on throughput decrease confines the decision to the already-processed prefix.

### The production adaptations (§5.2) — where theory meets CUDA graphs

[verified] Two conflicts with real infrastructure:
1. `SPS(B)` is **not smooth** — *"the true hardware capacity SPS(B) is inherently discrete,
   exhibiting a jagged, step-wise degradation"* — so the unimodality assumption breaks and
   early-stopping gets trapped in local minima.
2. Dynamic per-step token counts clash with **CUDA graph replay** and **Zero-Overhead
   Scheduling** (ZOS needs next-step batch size *before* the current step finishes).

Their fix is elegant: run the scheduler **asynchronously, using confidence outputs from two steps
prior** to set the truncation length `K` (the batch capacity limit), while the *ordering* of
candidates uses current, up-to-date cumulative confidences. This casts admission as **dynamic
top-K selection**. And because the capacity decision now depends only on 2-steps-ago information,
the causal barrier is preserved *by the asynchrony itself* — so they can **remove the
early-stopping break** and do an unconstrained global search that jumps over the SPS cliffs,
while still being provably lossless. *"asynchronous design forms a causal barrier, maximizing
physical throughput across hardware cliffs while preserving the exact target distribution."*

### Variable-length verification at the kernel layer (§5.3)

[verified] *"Standard decode kernels are heavily optimized for fixed query lengths; naively
processing variable-length verified prefixes leads to severe GPU under-utilization due to padding
and uneven workload distribution. We resolve this by **decoupling physical execution from logical
sequence tracking**. In our compute kernels, all tokens across different requests are flattened
and processed identically as independent elements. The complex intra-sequence dependencies are
then strictly conveyed via a **marker tensor** integrated into our sparse attention
implementation. Specifically on the DeepSeek-V4 architecture, **only the index-attention and
compress kernels require modification** to support this variable-length routing."*

And a framing that matches our situation exactly [verified]: *"In our deployment setting... the
effective batch size persistently remains well below the GPU's compute-saturating threshold.
Under this regime, the traditional trade-off simplifies: given a fixed concurrency limit,
maximizing per-GPU total token throughput and maximizing the generation speed per user become
**highly correlated objectives rather than competing ones**."*

### Results

**Offline** [reported], accepted length τ per round, temperature 1.0, chain drafting, block size
7, 5 draft layers (1 for Eagle3, TTT horizon 7), same training data (Open-PerfectBlend, 10 epochs):

| Target | Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | Eagle3 | 5.30 | 4.77 | 3.91 | 3.96 | 4.33 | 4.17 | 2.66 | 2.54 | 2.54 |
| Qwen3-8B | DFlash | 5.33 | 4.91 | 4.07 | 4.36 | 4.64 | 4.39 | 3.11 | 2.98 | 2.81 |
| Qwen3-8B | **DSpark** | **6.17** | **5.78** | **5.01** | **5.16** | **5.52** | **5.17** | **3.72** | **3.58** | **3.21** |

Macro-average improvement over Eagle3: **+30.9% / +26.7% / +30.0%** for Qwen3-4B/8B/14B;
over DFlash: **+16.3% / +18.4% / +18.3%**.

**A finding that should change how we think about EAGLE** [verified]: parallel drafters *beat*
autoregressive ones, and the reason is position 1. *"autoregressive models like Eagle3 are
constrained to shallow networks due to their O(γ) latency, whereas O(1) parallel drafters can
afford much deeper networks. This structural gap yields a substantial accuracy margin at position
1, with DFlash starting noticeably higher than Eagle3 (e.g., 0.88 vs 0.81 on Math, and **0.72 vs
0.53 on Chat**). Because speculative decoding operates as a strict prefix-matching survival
process, the first token carries the highest leverage — a rejection here immediately invalidates
the entire block."* Position-wise conditional acceptance shows Eagle3 flat-or-rising with
position while DFlash decays — DSpark combines DFlash's position-1 capacity with autoregressive
suffix coherence.

**Production on DeepSeek-V4** [reported]. DSpark-5 (γ=5) vs MTP-1. The V4-co-deployed drafter is
*"three MoE layers with mHC and a sliding window attention of 128"*, γ=5, Markov head.

| Engine | SLA (tok/s/user) | Aggregate throughput vs MTP-1 |
|---|---|---|
| V4-Flash | 80 | **+51%** |
| V4-Flash | 120 | +661% *(nominal; baseline near its operational boundary)* |
| V4-Pro | 35 | **+52%** |
| V4-Pro | 50 | +406% *(nominal)* |

At **matched throughput**: **+60–85% per-user speed (Flash)**, **+57–78% (Pro)**.

They are honest about the big numbers [verified]: *"We therefore interpret this high-SLA point
primarily as evidence that DSpark extends the feasible interactivity frontier, rather than as a
representative multiplicative speedup over a well-utilized baseline."*

**The load-adaptive mechanism, measured** [verified]: below ~200 concurrent requests (Flash) /
~150 (Pro) the scheduler expands verification *"from MTP-1's static 2 tokens to roughly 4–6
tokens per request"*; as concurrency rises, *"the average verification length decreases smoothly
with load."*

**Why MTP-1 was the baseline at all** [verified]: *"This single-token setup was historically
maintained in production because deploying a static multi-token drafter (e.g., MTP-3/5) strictly
degrades aggregate throughput under high concurrency due to excessive verification overhead."*

**Stated limitation** [verified]: *"DSpark still incurs a fixed draft-side cost to generate the
initial γ-token block via the parallel backbone. For complex queries with inherently low
acceptance rates, this upfront drafting compute is unrecoverable."* Proposed future fix:
difficulty-aware early exit in the drafter.

### Training-side systems tricks worth copying [verified §5.1]

- **Hidden-state communication instead of logits.** Rather than shipping full-vocabulary logits
  (`V ≈ 10^5`) between workers, *"we temporarily cache the target model's forward-pass activations
  and communicate only the hidden states immediately preceding the LM head. The LM head
  projection is then executed locally on the draft model's workers only for the sampled target
  positions. This reduces the per-token communication complexity to O(d)."*
- **Anchor-bounded sequence packing** via **token-level attention indices rather than 2D masks**,
  to keep exact causal masking across independent packed sequences without padding overhead.

### The released artifacts

[verified] `DeepSpec` ships configs, training and eval for **Eagle3, DFlash and DSpark**, with
checkpoints for Qwen3-{4B,8B,14B} and Gemma4-12B. Config knobs visible in the HF configs:
`block_size: 7`, `num_anchors: 512`, `markov_rank: 256`, `enable_confidence_head`,
`confidence_head_with_markov`, `target_layer_ids: [1,9,17,25,33]`, `num_hidden_layers: 5`
(DSpark/DFlash) vs `1` (Eagle3), `ttt_length: 7` (Eagle3). Training loss weights
`α_ce=0.1, α_tv=0.9, α_conf=1.0` with position weights `w_k = exp(−(k−1)/γ)`.
Storage warning worth knowing: the target cache is *"roughly 38 TB for the default Qwen/Qwen3-4B
setting."* Built on SGLang's **SpecForge**.

---

## TileKernels and Engram — the two newest repos

**TileKernels** [verified] https://github.com/deepseek-ai/TileKernels (MIT, Apr 2026).
*"Optimized GPU kernels for LLM operations, built with TileLang."* Requires **SM90 or SM100**,
PyTorch ≥2.10, TileLang ≥0.1.9, **CUDA 13.1+**. Modules: `moe/` (top-k gating, token-to-expert
mapping, fused expansion/reduction, weight normalization), `quant/` (per-token / per-block /
per-channel FP8/FP4/**E5M6** casting, **fused SwiGLU+quantization**), `transpose/`, `engram/`,
`mhc/` (Sinkhorn + mix split/apply), `modeling/`. Honest disclaimer: *"they do not represent best
practices and we are actively working on improving the code quality and documentation."*

[inferred] The **fused SwiGLU+quantization** kernels and the MoE routing kernels are drop-in
useful for any SM100 MoE engine, independent of DeepSeek's model. This repo is the least-known
and among the more immediately usable artifacts in the whole set.

**Engram** [verified] https://github.com/deepseek-ai/Engram (Jan 2026). *"Conditional Memory via
Scalable Lookup: A New Axis of Sparsity."* Modernized N-gram embeddings for O(1) lookup as a
complement to MoE. Claims a **U-shaped scaling law** trading neural computation (MoE) against
static memory (Engram), and iso-param/iso-FLOP gains for an Engram-27B model. The serving-relevant
claim: *"The module employs **deterministic addressing**, enabling the offloading of massive
embedding tables to host memory with minimal inference overhead."* Note that DeepEP V2 ships an
**"0 SM Engram (with RDMA)"** primitive — so this is being built into their communication stack.
The repo ships a demo (`engram_demo_v1.py`) that *"mocks standard components (like
Attention/MoE/mHC)"* — **not a production implementation**.

[inferred] Not in our path today, but if Kimi K3 / DeepSeek V4-next ship Engram-style lookup
tables, an engine needs host-memory embedding offload with deterministic prefetch. Worth watching.

---

## The current DeepSeek API pricing (inference economics, 2026-08)

[verified] https://api-docs.deepseek.com/quick_start/pricing — models `deepseek-v4-flash` and
`deepseek-v4-pro`, 1M context, max 384K output, per 1M tokens:

| Model | Cache hit in | Cache miss in | Output |
|---|---|---|---|
| deepseek-v4-flash | $0.007 off-peak / $0.014 peak | $0.22 / $0.44 | $0.66 / $1.32 |
| deepseek-v4-pro | $0.022 / $0.044 | $0.66 / $1.32 | $1.98 / $3.96 |

*"Off-peak rates are half of the peak rates. Peak hours are 01:00–04:00 and 06:00–10:00 UTC."*

[inferred] Compare to the Feb-2025 R1 prices used in the 545% calculation ($0.14 / $0.55 / $2.19):
V4-Flash output is **$1.32 vs $2.19 peak — a 40% cut** while the model went from 671B/37B-active
at 128K to 284B/13B-active at 1M context. The compounding of architectural efficiency (27% of
V3.2's FLOPs, 10% of its KV) and serving efficiency (Mega MoE, DSpark) is visible directly in the
price sheet. Also note the **structural** discount mechanism: off-peak = half price, which is the
productized version of the "reduce inference nodes at night" policy from the Day-6 post.

---

## Techniques ranked by transferability to our stack

| Technique | Source | Attacks | Transferability | Difficulty | Notes / caveats |
|---|---|---|---|---|---|
| **DP attention over TP for MLA/DSA decode** | FlashMLA deep dive + Day-6 + V3.2 launch cmd | attention 10.9%, collectives 19.6% | **Very high** | High | The `h_q·s_q ≥ 128` crossover is the argument. `--enable-dp-attention` exists in SGLang. Costs KV-cache replication across ranks — check 183 GB budget |
| **Mega-MoE fused dispatch/GEMM/SwiGLU/GEMM/combine megakernel** | DeepGEMM PR #304/#316 | MoE 19.4%, collectives 19.6% | **Very high** | High | 1.96× at BS=1, SM100+TMEM only, FP8×FP4 only, needs symmetric memory + multi-process. "vs legacy" = their own baseline |
| **Confidence-scheduled dynamic verification length** | DSpark §3.2, §5.2 | C1→C16 falloff | **Very high** | Medium (scheduler) / High (confidence head) | Algorithm 1 + `SPS(B)` cost table is implementable on top of our EAGLE today. The async 2-steps-prior trick is what makes it CUDA-graph/ZOS compatible |
| **Host codegen / sub-µs launch validation** | V4 §3.2 | all, at 2.74 ms TPOT | **Very high** | Medium | "tens/hundreds of µs → <1 µs". Corroborated by Hybrid-EP's 32% Torch-API-vs-kernel gap |
| **FP32→BF16 index scores for the top-k selector** | V4 §5.2.1 | DSA indexer 5.8% | **Very high** | Low | 2× top-k speedup, 99.7% KV recall. Cheapest item on this list |
| **`EVICT_FIRST` TMA cache hints + fine-grained TMA↔GEMM pipelining** | FlashMLA deep dive | attention 10.9% | High | Medium | 9× 64×64 TMA copies per 64×576 K block; start GEMM on first arrival |
| **PDL between dependent kernels** | FlashMLA + DeepGEMM `set_pdl` | dense GEMM 37.1%, attention | High | Low-Medium | First-class in DeepGEMM; overlaps splitkv↔combine |
| **"Crossover": DSM sharing of dequantized KV across CTAs in a cluster** | FlashMLA FP8 deep dive | attention under NVFP4/FP8 KV | High | Medium-High | 250→410 TFLOPS on H800. B200 has clusters + DSM. Verify B200's native FP4/FP8 cast doesn't already fix the dequant bound |
| **Indexer over compressed blocks (CSA)** | V4 §2.3.1 | DSA indexer 5.8% | High (idea) / Low (as-is) | High | Composes with our `index_topk_freq=4`. Requires retraining — a next-model note |
| **Masked-MHA dense path for short prefill** | V3.2 paper | TTFT 189 ms at ~10k tokens | High | Medium | They ship two prefill implementations and switch on length. AA uses ~10k input |
| **Batch-invariant dual-kernel decode (one-SM + multi-SM tail wave, bitwise identical)** | V4 §3.3 | collectives 19.6% / rank skew | High | High | Closest published analogue to rank-arrival skew. Also gives reproducibility for free |
| **DeepEP V2 analytical SM budgeting + Hybrid-EP TMA path** | DeepEP V2 README, hybrid-ep docs | collectives 19.6% | High | Medium | 16 SMs → 409 GB/s FP8 dispatch on B200 EP8, vs 44–48 SMs for DeepEP baseline parity. NVFP4 support on that branch |
| **`nvDev` branch / Compute Fabric Transport** | DeepEP README | collectives at C1 | Medium-High | Medium | *"better latency on small token sizes"* — our exact regime. Experimental |
| **Hash-routed first-N MoE layers → host-side dispatch prefetch** | V4 config + `model.py` | MoE 19.4% at C1 | Medium | Low (if model has it) | Only applies if the model uses hash routing. GLM-5.2 likely does not |
| **Grouped low-rank output projection** | V4 §2.3.1 | dense GEMM 37.1% | Medium | Very High | Requires retraining. Note for the next model generation |
| **Lossless FP4→FP8 dequant (1×32 scales absorbed by 128×128 E4M3 range)** | V4 §5.2.1 | NVFP4 build pipeline | Medium | Medium | Lets an NVFP4 model reuse an FP8 kernel pipeline unchanged. Checkable precondition |
| **On-disk / hierarchical KV cache** | Day-6 + 3FS + V4 §3.5.2 | cost per user | Medium | Medium-High | 56.3% hit rate in production. **Zero effect on the AA leaderboard** (fresh prompts) |
| **EPLB redundant experts** | eplb.py | collectives at C16+ | Medium | Low | 130 lines. Useless at C1 with 8 experts over 8 ranks |
| **LPLB per-batch LP rebalancing** | LPLB README | collectives at C64 | Low-Medium | Medium | ~100 µs solver — 36× our per-token budget. Their own README calls it early-stage |
| **Parallel drafter architecture (DFlash) over autoregressive EAGLE** | DSpark §4.3.1 | acceptance rate | Medium | High | Position-1 acceptance 0.72 vs 0.53 on chat. Requires training a new drafter |
| **`ld.global.nc.L1::no_allocate.L2::256B` volatile-read hack** | DeepEP legacy docs | comm kernels | **Low — do not port blindly** | Low | Correctness argument is explicitly Hopper-specific |
| **Node-limited routing / group-limited gating** | V3 paper, ISCA §4.3 | — | **Anti-transferable** | — | Meaningless in a single NVLink domain. V4 appears to have dropped it |
| **DualPipe / DualPipeV** | DualPipe repo | — | **Anti-transferable** | — | Training-only, requires PP |
| **EP32/EP144 cross-node disaggregation** | Day-6 | — | **Anti-transferable** | — | Their operating point is 20–22 tok/s/user at massive scale; ours is the opposite |
| **LogFMT log-domain communication compression** | ISCA §3.2 | — | **Negative result** | — | They validated the format then abandoned it: 50–100% encode/decode overhead |
| **SASS post-processing FFMA interleaving** | DeepGEMM news | — | **Obsolete** | — | *"NVCC 12.9 will automatically do the FFMA interleaving"* |
| **NVRTC JIT backend** | DeepGEMM news | — | **Withdrawn** | — | Added for 10× compile speed, disabled in the 2025-07 refactor |

---

## Things I could not source

- **[unverified]** Any DeepSeek statement of end-to-end tokens/s for V4 on **B200/GB200 hardware
  specifically.** Their published serving numbers are H800-based (Day-6, V3.2 cost figures), and
  the DSpark production figures give tok/s/user SLAs (80/120 for Flash, 35/50 for Pro) without
  naming the hardware. DeepGEMM's Mega MoE and DeepEP V2 numbers are SM100, but they are
  component microbenchmarks, not end-to-end serving.
- **[unverified]** An independent, third-party reproduction of *any* of the performance tables in
  this document. Every number tagged [reported] is DeepSeek's own measurement.
- **[unverified]** The standalone DeepSeek mHC paper (cited in DSpark as "Xie et al. 2026"). I
  found many third-party mHC papers on arXiv but not a DeepSeek-authored one; the mechanism is
  documented in the V4 report §2.2 and in `TileKernels/mhc/`, which is sufficient.
- **[unverified]** Whether the `ld.global.nc.L1::no_allocate` hack is still present in DeepEP V2's
  Gin path. The V2 env-var list retains `DISABLE_AGGRESSIVE_PTX_INSTRS` but scopes it to "legacy
  methods"; I did not read the V2 CUDA sources.
- **[unverified]** DeepSeek's actual (as opposed to theoretical) revenue or margin. They
  explicitly decline to state it.
- **[unverified]** Anything about `deepseek-harness` beyond its one-line description
  (*"Everything is a Plugin"*, 135k stars, pushed 2026-08-13) — its README is a stub and it
  appears to be an agent framework, not inference infrastructure.

---

## Sources

All fetched and read on 2026-08-17.

**Open Infra Week / repos**
- Inference System Overview (Day 6): https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- open-infra-index: https://github.com/deepseek-ai/open-infra-index
- FlashMLA: https://github.com/deepseek-ai/FlashMLA
- FlashMLA deep dive (new kernel, 2025-04-22): https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
- FlashMLA deep dive (FP8 sparse decode, 2025-09-29): https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md
- DeepEP: https://github.com/deepseek-ai/DeepEP
- DeepEP V1 legacy docs (PTX hack, hook overlap): https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md
- DeepEP V2 release PR #605: https://github.com/deepseek-ai/DeepEP/pull/605
- DeepEP Hybrid-EP branch docs (B200 tables): https://github.com/deepseek-ai/DeepEP/blob/hybrid-ep/docs/README_Hybrid-EP.md
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- DeepGEMM PR #304 (Mega MoE, FP4 Indexer, PDL): https://github.com/deepseek-ai/DeepGEMM/pull/304
- DeepGEMM PR #316 (Mega MoE benchmarks): https://github.com/deepseek-ai/DeepGEMM/pull/316
- DualPipe: https://github.com/deepseek-ai/DualPipe
- EPLB (algorithm source read in full): https://github.com/deepseek-ai/eplb
- LPLB: https://github.com/deepseek-ai/LPLB
- profile-data: https://github.com/deepseek-ai/profile-data
- 3FS: https://github.com/deepseek-ai/3fs
- smallpond: https://github.com/deepseek-ai/smallpond
- TileKernels: https://github.com/deepseek-ai/TileKernels
- Engram: https://github.com/deepseek-ai/Engram
- DeepSpec: https://github.com/deepseek-ai/DeepSpec

**Papers**
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437 (HTML v2 read)
- Insights into DeepSeek-V3 (ISCA'25 hardware co-design): https://arxiv.org/abs/2505.09343 (HTML v1 read in full)
- DeepSeek-V3.2-Exp (DSA): https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf (PDF text extracted)
- DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
- **DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**: https://arxiv.org/abs/2606.19348 (HTML read; §2.3 CSA/HCA, §3.2 TileLang, §3.3 batch-invariance, §3.5 inference framework, §5.2.1 FP4 QAT)
- **DSpark**: https://arxiv.org/abs/2607.05147 (HTML read in full)
- DFlash (referenced baseline): https://arxiv.org/abs/2602.06036 — *cited by DeepSpec; I did not read it*

**Models / configs / reference code**
- DeepSeek-V4-Pro card + config + `inference/model.py` + `inference/kernel.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V4-Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- DeepSeek-V3.2 config: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- DeepSpec checkpoints (dspark/dflash/eagle3 configs): https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7 and siblings
- Model index (HF API, all deepseek-ai models by lastModified): https://huggingface.co/api/models?author=deepseek-ai

**Other**
- DeepSeek API pricing: https://api-docs.deepseek.com/quick_start/pricing
- DeepSeek GitHub org repo listing (via API): https://api.github.com/orgs/deepseek-ai/repos
