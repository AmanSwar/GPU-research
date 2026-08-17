# MoE at inference speed: grouped GEMM, NVFP4 expert kernels, routing and expert parallelism

**What this is.** A kernel-engineer's map of the MoE decode path on 8×B200 for GLM-5.2
(256 routed experts, top-8, NVFP4, TP8). Every stage — router, top-k, permutation,
grouped GEMM, unpermute, combine — with the kernel that actually runs on our box,
taken out of the `sweep-latency-3-1-4` nsys capture and out of the trtllm-gen kernel
metadata on this filesystem, not from memory. Then the batch-1 arithmetic (bytes,
arithmetic intensity, achieved fraction of DRAM peak), the EP-vs-TP decision with the
imbalance math, and a concrete recommendation at C1 and C64.

---

## Bottom line for our system

- **The MoE family is 19% of C1 kernel time and it decomposes into four kernels, not one.**
  Measured on device 0: expert GEMMs 1088.9 ms (12.1%), routing 233.4 ms (2.6%),
  finalize/combine 198.3 ms (2.2%), gated-activation 177.8 ms (2.0%). [verified —
  `/home/aman/code/benchmark/runs/sweep-latency-3-1-4/trace.sqlite`]
- **The down-projection GEMM is the weak one, and the reason is structural.** At TP8 the
  MoE `K` per rank is exactly 256, which is exactly one `tileK` — so the 9-stage software
  pipeline in `bmm_Bfloat16_E2m1E2m1_..._t128x8x256_s9_...` never fills. It moves half the
  bytes of GEMM1 in 85% of the time: **≤2.81 TB/s vs GEMM1's ≤4.80 TB/s** (36.7% vs 62.6%
  of the 7.672 TB/s CUPTI-reported peak). [verified, with the caveat below on distinct-expert
  count]
- **We are not HBM-bound at C1.** Average DRAM read over the steady window is **17.2% of
  peak** on all eight GPUs (max sample 84–87%), Tensor Active **4.1%**, SMs Active 55%,
  GR Active 95%. Independent arithmetic (12.83 GB of weights per 4-token verify forward ÷
  7.672 TB/s = 1.672 ms floor against an ~8.96 ms measured forward) predicts **18.7%**.
  The counter and the arithmetic agree, which means essentially all DRAM traffic is weight
  traffic and the step is ~5× longer than the bandwidth floor. [verified]
- **Speculative decoding amortises dense weights but *not* expert weights.** The 4-token
  EAGLE verify reads attention/dense weights once but touches up to 4×8 = 32 expert slots.
  Grid is literally `(4, 32)` for GEMM1 in every decode launch. Deeper speculation grows
  the MoE share of the step; it does not shrink it. [verified — grid distribution query]
- **GEMM1 runs 128 CTAs on a 148-SM GPU** — under one wave, so its ceiling is 86.5% of peak
  bandwidth before any inefficiency. Any fix that raises CTA count (smaller `tileM`, CGA
  split-K `c1x1x2`, or fusing GEMM1+GEMM2 persistently) is attacking the right thing.
  [verified grid; [inferred] on the fix]
- **EP is the wrong parallelism at C1 and the right one at C1024.** 32 expert slots into 8
  EP ranks gives E[max]/mean = **1.76×** imbalance under *uniform* routing (worse in
  practice); TP8 is perfectly balanced by construction. At 1536 slots (C64-ish) imbalance
  falls to 1.10×. [verified by simulation; uniform-routing assumption is optimistic]
- **Cheapest un-taken win in this area:** the shared expert is stored **BF16, unquantized**
  (`model.layers.N.mlp.shared_experts.*_proj.weight`, `BF16 [2048,6144]`), costing 9.44 MB
  per rank per layer against 21.2 MB for all eight routed experts. Quantising it to NVFP4
  removes ~0.63 GB per forward — about 82 µs of pure bandwidth at peak. [verified shapes;
  [inferred] saving]
- **Expert offloading is dead on this box.** PCIe Gen5 x16 = ~63 GB/s; one token's routed
  experts are 1.593 GB/GPU → 25 ms of PCIe per token against a 2.24 ms TPOT. Not viable at
  any concurrency. [verified link gen/width; arithmetic]

---

## 1. The pipeline, stage by stage

A routed-MoE layer at decode is six distinct pieces of work. Naming them separately matters
because in our profile they are six separate kernels with six separate costs, and three of
them are not GEMMs.

```
  x[T,H] ──► router GEMM  ──► logits[T,E]
                              │
                              ▼
                          score + top-k + norm  ──► topk_ids[T,K], topk_w[T,K]
                              │
                              ▼
                          histogram + prefix-scan over experts
                              │   (expandedIdx → permutedIdx, ctaIdxXyToBatchIdx,
                              │    numNonExitingCtas, totalNumPaddedTokens)
                              ▼
                          permute (gather rows by expert)  [fused into the GEMM on our path]
                              │
                              ▼
    ┌──────── grouped GEMM 1: [tokens_e, H] × [2I_r, H]ᵀ → [tokens_e, 2I_r] ────────┐
    │                    + fused SwiGLU epilogue + NVFP4 requant                     │
    └───────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────── grouped GEMM 2: [tokens_e, I_r] × [H, I_r]ᵀ → [tokens_e, H] ──────────┐
    └───────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                          unpermute + weighted combine (finalize)  ──► y[T,H]
```

### 1.1 Router GEMM

`[T, H] × [E, H]ᵀ → [T, E]`. For GLM-5.2: `H=6144`, `E=256`, and the gate weight is
`model.layers.N.mlp.gate.weight  BF16 [256, 6144]` — **replicated, not TP-sharded**, and
computed in fp32 (`"moe_router_dtype": "float32"`). [verified —
`/home/aman/code/weights/GLM-5.2-NVFP4/model-00006-of-00047.safetensors` header,
`config.json`]

At `T=4` this is a 4×6144×256 GEMM = 12.6 MFLOP reading 3.1 MB. It is a rounding error in
FLOPs and it never shows up as its own kernel in our trace, because **trtllm-gen fuses the
router into the routing kernel** when you pass `routing_logits` rather than pre-computed
`topk_ids` — the `FromLogits` path. [verified — `flashinfer/fused_moe/core.py`
`trtllm_fp4_block_scale_moe(routing_logits=...)`, and
`/home/aman/code/NotSglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
lines ~1086–1135]

Design note: the router is the one GEMM whose *latency* matters more than its throughput,
because everything downstream depends on it. On a 78-layer model at C1 it is on the critical
path 76 times per forward. Keeping it fused with scoring+top-k is unambiguously right.

### 1.2 Scoring, top-k, normalisation

GLM-5.2's config: `scoring_func: "sigmoid"`, `topk_method: "noaux_tc"`,
`n_group: 1`, `topk_group: 1`, `norm_topk_prob: true`, `routed_scaling_factor: 2.5`,
plus a per-expert `e_score_correction_bias`. [verified — `config.json`]

Because `n_group == 1` the DeepSeek-V3 group-limited path degenerates and the runtime lands
on trtllm-gen's **custom routing policy**. The exact template instantiation in our trace:

```
moe::dev::routing::routingCustom::routingIndicesBlockKernel<
  moe::dev::routing::routingCustom::KernelParams<
      float,               // input logits dtype
      __nv_bfloat16,       // output expert-weight dtype (always bf16)
      256,                 // MaxNumExperts
      8,                   // MaxNumTopExperts
      moe::dev::routing::TopKExpertSelect<
          moe::dev::routing::SigmoidBiasPreprocess,
          moe::dev::routing::ScaledSumNormalizePostprocess>>>
```
[verified — trace, device 0; 41,925 launches, 220.2 ms, **avg 5.25 µs**, grid `(1,1,1)`,
block 256]

`SigmoidBiasPreprocess` = sigmoid then add `e_score_correction_bias`;
`ScaledSumNormalizePostprocess` = divide by the sum of the selected weights then multiply by
`routed_scaling_factor`. That is exactly GLM-5.2's `noaux_tc` + `norm_topk_prob` +
`routed_scaling_factor=2.5`. The dtype pair `(float, __nv_bfloat16)` confirms fp32 logits in,
bf16 expert weights out — the latter is hard-coded regardless of logit dtype
(`routingData.mDtypeOutput` is bf16 for every routing method). [verified — flashinfer
`core.py` comment at line ~2341, and FlashInfer issue 3595 referenced in our tree]

**The cost is worth staring at.** 5.25 µs, on **one CTA**, 147 SMs idle, 76 times per forward
= ~400 µs per forward. That is 2.6% of all GPU time at C1 spent on a single-block kernel doing
a 256-bin histogram and an exclusive scan for four tokens.

### 1.3 Permutation: histogram, scan, and the descriptor arrays

This is the stage most write-ups skip, and it is where the grouped-GEMM contract is actually
built. trtllm-gen produces, on device:

| output | shape | meaning |
|---|---|---|
| `mPtrExpandedIdxToPermutedIdx` | `[T*K]` | where token `t`'s `k`-th copy lands in the permuted buffer; `-1` if the expert is not local |
| `mPtrCtaIdxXyToBatchIdx` | `[divUp(T*K + E*(tileN-1), tileN)]` | which expert each CTA in the token dim works on |
| `mPtrCtaIdxXyToMnLimit` | same | per-CTA end offset, so a CTA knows its valid row range |
| `mPtrNumNonExitingCtas` | `[1]` | how many CTAs are real; the rest early-exit |
| `mPtrTotalNumPaddedTokens` | `[1]` | `Σ_b divUpMul(N[b], tileN)` |
| `mPtrPermutedIdxSize` | `[1]` | same quantity, used for buffer sizing |

[verified — `BatchedGemmInterface.h`, lines ~404–475, in
`/home/aman/.cache/flashinfer/cubins/b368d003e8fdfe4b271bff7c788ac52ef789a81b/batched_gemm-da58956-b4ac80e/include/trtllmGen_bmm_export/`]

The header's own example is the clearest statement of the padding contract:

```
// There are 3 tokens [0, 1, 2] such that [0, 1] belong to batch [B0] and [2] to batch [B1].
// Let's assume that the padded size is 4.
//   expandedIdx[0] = 0
//   expandedIdx[1] = 1
//   expandedIdx[2] = divUpMul(2, 4) + 0 = 4
// The route map is [B0, B0, X, X, B1, X, X, X] where X could be any value.
```

and for the CTA→batch map:

```
// E.g. with listM = 128,255,32 and tileM = 128, should be equal to
// ctaIdxXyToBatchIdx = [0, 1, 1, 2]
// with numTokens = [128,255,32] and tileM = 128:
// ptrCtaIdxXyToMnLimit = [128, 256, 383, 416]
```

**Three routing-kernel shapes exist and the launcher picks between them by token count**
[verified — `flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu`
lines ~580–620]:

```cpp
int numThreadsPerCluster = numThreadsHist * NumBlocksPerCluster;   // NumBlocksPerCluster = 8
bool const useSingleCluster =
    data.mNumTokens <= 1024 && data.mNumTokens * topK <= numThreadsPerCluster;

if (useSingleCluster)              launchClusterKernel(...);        // distributed-smem cluster
else if (numTokens <= maxTokensCoop) launchCoopKernel(...);         // cooperative grid
else { launchHistogramKernel(...); launchOffsetsKernel(...); }      // two-pass
```

For GLM-5.2, `numThreadsHist = getMaxNumExperts(256) = 256`, so
`numThreadsPerCluster = 2048` and the single-cluster path is taken for any
`T ≤ 256` at top-8. Our trace shows the **block** variant at decode (grid `(1,1,1)`) and the
**cluster** variant (grid `(8,1,1)`, 684 launches) elsewhere; the histogram/offsets pair
appears only in prefill (`routingIndicesHistogramScoresKernel`, grid `(1024,1,1)`, 304
launches). [verified — trace]

The cluster kernel is worth reading because it is a nice piece of Hopper/Blackwell code:
`smemExpertCount` is accumulated per-CTA with `atomicAdd` into shared memory, then combined
across the cluster with `cg::cluster_group::map_shared_rank(smemExpertCount, rank)` — i.e. a
histogram reduction over **distributed shared memory**, no global atomics — followed by a
CUB `ExclusiveSum` to get per-expert offsets and `numNonExitingCtas` in one pass. [verified —
`flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh` lines 264–510]

### 1.4 The expert GEMMs

Covered in depth in §2 and §3.

### 1.5 Unpermute + weighted combine ("finalize")

```
moe::dev::finalize::finalizeKernel<KernelParams<bfloat16_t, bfloat16_t, 4, true>>
```
grid `(24,16,1)`, block 256, 42,703 launches, **167.8 ms, avg 3.93 µs**. The prefill sibling
`finalizeKernelVecLoad<...,4,true>` runs at grid `(112,1,1)`, avg 73.4 µs. [verified — trace]

The kernel body is the obvious thing done carefully: for each output element, loop `k` over
top-K, follow `expandedIdxToPermutedIdx`, skip `-1`, accumulate
`expertWeightsPtr[expandedIdx] * inPtr[permutedIdx * hiddenDimPadded + hiddenIdx]` in fp32,
store bf16. The `VecLoad` variant packs `TopKUnrollFactor` indices and scales into a single
vector load (`FinalizeTraits<1|2|4>` selects `int`/`int2`/`int4` packed index types) and
stages them in shared memory before the reduction — that is where the 4 in the template
comes from. [verified — `trtllm_fused_moe_dev_kernel.cu` lines 626–900]

There is also a **`do_finalize=False`** mode that returns
`(gemm2_output, expert_weights, expanded_idx_to_permuted_idx)` and lets the caller fuse the
combine into a downstream kernel — SGLang wires this up as
`flashinfer_trtllm_deferred_finalize_context` / `finalize_flashinfer_trtllm_deferred_output`.
[verified — `flashinfer_trtllm.py` lines 63–90, 1120–1160]. Also present in the ABI: a
**direct register-to-gmem finalize** inside the GEMM epilogue via `mPtrExpertWeightsPtr` +
`mPtrPermutedIdxToExpandedIdx` + `mUseCMultiCast`, with a self-resetting multicast completion
barrier (`mPtrMulticastCompletionBarUc/Mc`). [verified — `BatchedGemmInterface.h` lines
~478–505]. That is the fully-fused combine; we are not using it.

### 1.6 What our box actually runs — measured

Device 0, `runs/sweep-latency-3-1-4/trace.sqlite`, C1, TP8, NVFP4, EAGLE 3-1-4.
Total device-0 kernel time in the capture: **8966.1 ms** over a 7599.0 ms span,
**1,183,520 kernels**. [verified]

| stage | kernel | launches | total ms | avg µs | grid | block |
|---|---|---:|---:|---:|---|---:|
| GEMM1 + SwiGLU | `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f` | 40,312 | 475.5 | **11.79** | (4,32,1) | 640 |
| GEMM2 (down) | `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f` | 40,312 | 405.6 | **10.06** | (48,32,1) | 512 |
| routing | `routingCustom::routingIndicesBlockKernel<...SigmoidBias, ScaledSumNormalize>` | 41,925 | 220.2 | 5.25 | (1,1,1) | 256 |
| combine | `finalize::finalizeKernel<bf16,bf16,4,true>` | 42,703 | 167.8 | 3.93 | (24,16,1) | 256 |
| gated act (non-MoE) | `act_and_mul_kernel<bf16, ActivationKind0, true, false>` | 44,742 | 177.8 | 3.97 | (6,1,1) | 256 |
| all `bmm_*` | — | 86,166 | **1088.9** | — | — | — |
| all `*routing*` | — | 44,451 | 233.4 | — | — | — |
| all `*finalize*` | — | 44,177 | 198.3 | — | — | — |

Sum of the four MoE families = 1698.4 ms = **18.9%** of device-0 kernel time — matching the
ledger's 19.4% (the small delta is classifier scope, and the ledger's total of 9572 ms vs my
8966 ms). [verified; consistent with
`/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` §2]

Whole-GPU counters over the middle 60% of the window, all eight GPUs (nsys GPU metrics,
100 µs sample period):

| metric | avg | max |
|---|---:|---:|
| GR Active [%] | 92.3 – 95.8 | 100 |
| SMs Active [%] | 52.2 – 56.0 | 100 |
| **Tensor Active [%]** | **4.07 – 4.11** | 100 |
| **DRAM Read BW [%]** | **17.16 – 17.20** | 80 – 87 |
| DRAM Write BW [%] | 2.14 – 2.15 | 53 – 55 |
| NVLink RX/TX user data [%] | 2.24 – 2.41 | 66 – 68 |

[verified — `GPU_METRICS` table, `TARGET_INFO_GPU_METRICS` metric ids 2/3/5/9/10/12/16]

Device constants straight out of the trace's CUPTI device record: **148 SMs**, L2 =
132,644,864 B (126.5 MiB), total memory 191,495,471,104 B, **memoryBandwidth
7,672,320,000,000 B/s**, clockRate 1.965 GHz, `maxShmemPerSm` 232,448 B,
`maxShmemPerBlockOptin` 233,472 B, chip `GB100`, SM 10.0. [verified —
`TARGET_INFO_GPU`]

---

## 2. Grouped GEMM on SM100

### 2.1 The problem

A grouped GEMM is one launch executing a list of independent GEMMs with different `M` (and
possibly `N`, `K`). For MoE, group `e` is expert `e` and `M_e` is however many tokens routed
to it. Three properties make it hard:

1. **`M_e` is not known on the host.** It comes out of the routing kernel. Any design that
   needs host-visible sizes forces a device→host sync, which at 76 layers × 2 GEMMs × a
   2.2 ms step is unaffordable.
2. **`M_e` is ragged.** Tile quantisation wastes `Σ_e (ceil(M_e/tileM)·tileM − M_e)` rows.
3. **The tail wave.** With variable group sizes the CTA count varies per launch, so a static
   grid either over-launches (and must early-exit) or under-launches.

### 2.2 CUTLASS: `GroupProblemShape`, `MoEProblemShape`, device-side TMA

CUTLASS 3.x/4.x ships `examples/75_blackwell_grouped_gemm/` with both a plain and a
block-scaled variant. Key structure [verified — the example source and CUTLASS 4.x changelog
via https://github.com/NVIDIA/cutlass/blob/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu]:

- Problem shapes are `GroupProblemShape<Shape<int,int,int>>`, constructed as
  `{groups, problem_sizes_device_ptr, nullptr}` — the host pointer is allowed to be null,
  i.e. **the kernel reads shapes from device memory**.
- A newer `MoEProblemShape` takes `max_m, max_n, max_k` plus a counts vector and derives
  per-group shapes internally, removing the need to materialise a shape array at all.
- Kernel schedules are the pointer-array variants:
  `KernelPtrArrayTmaWarpSpecialized1SmSm100` / `...2SmSm100` with
  `EpiloguePtrArrayTmaWarpSpecialized1Sm` / `2Sm`. MMA tile `<128,256,K>` for 1SM,
  `<256,256,K>` for 2SM; 2SM requires `cluster_dim.x >= 2`.
- **Device-side TMA descriptor modification** is the enabling feature: the kernel rewrites
  the tensormap's global address as it moves between groups, rather than the host encoding
  one descriptor per group. The PTX for this is `tensormap.replace` + a
  `tensormap.cp_fenceproxy` — both are present as first-class intrinsics in our toolkit at
  `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/tensormap_replace.h`
  and `.../tensormap_cp_fenceproxy.h`. [verified — files exist in the local CUDA 13.3 tree]
- Alignment: 128-bit TMA alignment, i.e. 16 elements for FP8, 8 for FP16.

The mufeezamjad NVFP4 grouped-GEMM worklog is the best public account of what this costs in
practice on SM100 [reported — https://mufeezamjad.com/blog/nvfp4-group-gemm]: the accumulator
plus scale factors fill the 512-column TMEM partition so **only one CTA fits per SM**
(12.5% theoretical occupancy); 128×128 output tiles support 6 pipeline stages while 128×256
tiles fall to 2–4 because of smem pressure; the winning design is a **persistent kernel with a
global atomic work counter** over a flattened tile list, plus warp specialisation
(separate A/SFA and B/SFB producer warps + an MMA consumer warp) and cluster multicast of A.
Their v4 reaches 23.8 µs geomean against a 238 µs reference on mixed-`M` MoE shapes, at 50%
tensor utilisation, 27% DRAM throughput, 26% L2 hit rate, 12.4% achieved occupancy. Those
last three numbers are the useful ones: **a good SM100 grouped GEMM at MoE decode shapes is
not near either roof.**

CUTLASS also exposes `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16` for the
NVFP4 path with `block16` = one scale per 16 elements, matching NVFP4 (MXFP4 uses `block32`).
[reported — same worklog; the instruction family is documented in the PTX ISA, and the
`tcgen05_mma.h` / `tcgen05_alloc.h` / `tcgen05_cp.h` headers are present locally]

### 2.3 Why variable group sizes hurt, quantitatively

Two separate costs, and it is worth not conflating them.

**Padding waste.** With `tileN = 8` (our decode tile) and 256 local experts, the padded token
count is `Σ_e ceil(M_e/8)·8`. At C64 (≈192 tokens/forward, top-8 → 1536 slots, mean 6 per
expert) that is 256×8 = 2048 padded slots vs 1536 real = **33% padding**. But padding only
inflates *activation* traffic and MMA work, and at decode both are negligible next to weight
traffic — so this is a second-order cost for us. [inferred from the tile-selection code +
config]

**Wave quantisation.** This one is first-order. The CTA count is
`gridM · Σ_e ceil(M_e/tileN)`. For our GEMM1 that is `4 × 32 = 128` CTAs on 148 SMs — under
one wave, permanently. A kernel that cannot fill a wave cannot reach peak bandwidth no matter
how good its pipeline: its ceiling is `128/148 = 86.5%` of DRAM peak. [verified grid;
arithmetic]

**Scheduler cost.** trtllm-gen's answer is `mIsStaticBatch = 0` (the `dynB` suffix): launch a
*static, worst-case* grid and have the surplus CTAs early-exit by comparing their index against
`*mPtrNumNonExitingCtas`. This keeps the launch CUDA-graph-capturable at the price of launching
CTAs that do nothing. The relevant option fields in the cubin metadata are
`mEnablesEarlyExit = 1`, `mEnablesDelayedEarlyExit = 0`,
`mGridWaitForPrimaryEarlyExit = 1`. [verified — `flashinferMetaInfo.h`]

### 2.4 The alternative: one padded batched GEMM

Pad every expert to a fixed `M_pad` and issue a plain strided-batched GEMM. Cost model:
work scales as `E · M_pad` instead of `Σ_e M_e`. At decode with `E=256`, `T·K = 32` slots and
`M_pad = 8`, that is 2048 token-slots of work for 32 real tokens — **64× waste**. Dead on
arrival for decode. It is defensible only when `T·K/E` approaches `M_pad`, i.e. prefill.

The honest version of this idea for decode is what our kernel already does: `tileN = 8` and
*skip empty experts entirely* via the CTA→batch map. That is a padded batched GEMM where the
padding is per-*touched*-expert, not per-expert.

### 2.5 The trtllm-gen formulation we run: batch-**N**

Our kernel names carry `bN` and `transOut`. Decoded against the header:

- `mBatchMode = BatchMode::BatchN (1)` → **the batched dimension is N, and N is tokens**.
  Therefore **A is the weight matrix and B is the activation matrix**. [verified —
  `BatchedGemmOptions.h` line 82 `enum class BatchMode { BatchM, BatchN };`; metadata dump
  shows `mBatchMode = BatchedGemmOptions::BatchMode(1)`]
- `mTransposeMmaOutput = 1` (`transOut`) → the MMA computes `Wᵀ·xᵀ` and the epilogue writes
  the token-major result.
- Consequence: `tileM` tiles the *weight output channels*, `tileN` tiles the *tokens*. That
  is why our decode tile is `t128x8x512`: **128 output channels × 8 tokens × 512 K per CTA.**
- `mUseShuffledMatrix = 1` → the weights are pre-shuffled offline to match the epilogue's
  TMEM readback order (see §3.4).

GEMM1: `M = 2·I_r = 512` → `gridX = 512/128 = 4` ✔ matches the observed grid.
GEMM2: `M = H = 6144` → `gridX = 6144/128 = 48` ✔.
Both: `gridY = 32 = T·K = 4·8` ✔. The grid is a direct, verifiable readout of the model shape.

---

## 3. The NVFP4 expert GEMM path we actually run

### 3.1 Decoding the kernel name

```
bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16
   _t128x8x512_s5_et128x8_m128x8x64_c1x1x1
   _rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN
   _tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f
```

| token | option field(s) | value on our kernel |
|---|---|---|
| `bmm` | — | batched GEMM (trtllm-gen `BatchedGemm`) |
| `E2m1` (1st) | `mDtypeC` | output C is NVFP4 (GEMM1 re-quantises for GEMM2) |
| `E2m1E2m1` | `mDtypeA`, `mDtypeB` | weights NVFP4, activations NVFP4 (W4A4) |
| `Fp32` | `mDtypeAcc` | fp32 accumulate |
| `Ab16 Bb16 Cb16` | `mSfBlockSizeA/B/C` | **block size 16** for all three → NVFP4, not MXFP4 (which would be 32) |
| `t128x8x512` | `mTileM/N/K` | 128 out-channels × 8 tokens × 512 K |
| `u2` (when present) | `mUseUnrollLoop2xForMma` | 2× MMA loop unroll |
| `s5` | `mNumStagesA/B` | 5-stage A/B pipeline |
| `et128x8` | `mEpilogueTileM/N` | epilogue tile |
| `m128x8x64` | `mMmaM/N/K` | UMMA instruction shape |
| `c1x1x1` | `mClusterDimX/Y/Z` | no CGA; `c2x1x1` = CTA-pair MMA, `c1x1x2..4` = CGA split-K |
| `rM` | raster order | raster along M |
| `TN` | `mLayoutA/B` | both MajorK |
| `transOut` | `mTransposeMmaOutput=1` | transposed MMA output |
| `schedS` / `schPd2x1x2x3` | `mTileScheduler` / persistent-CGA schedule | ours is the persistent variant |
| `biasFp32M` | `mBiasType = BiasType(1)`, `mBiasDtype = Fp32` | per-`M` fp32 bias (the dequant scale carrier) |
| `bN` | `mBatchMode = BatchN` | tokens are the batched dim |
| `tma` / `ldgsts` | `mRouteImpl` | **how the gather (permutation) is done**: `RouteImpl::Tma` = `UTMALDG.GATHER4`, `RouteImpl::Ldgsts` = LDGSTS, `LdgPlusSts` = LDG+STS |
| `tmaSf` / `ldgstsSf` | `mRouteSfsImpl` | same, for the scale-factor gather |
| `rgTma` | `mUseTmaStore` | TMA store in the epilogue |
| `clmp` | `mClampBeforeAct = 1` | clamp the accumulator before the activation |
| `swiGlu` | `mActType`, `mFusedAct = 1` | **fused SwiGLU epilogue** |
| **`dynB`** | **`mIsStaticBatch = 0`** | **dynamic batch: per-expert token counts are read from device pointers at run time** |
| `sm100f` | `gemm::SmVersion::Sm100f` | SM100 "family" cubin |

[verified — every mapping above is read off the option dump in
`flashinferMetaInfo.h` (10.8 MB, 2,516 kernels) next to the kernel name, cross-checked
against `BatchedGemmOptions.h`, `BatchedGemmEnums.h`, `GemmGatedActOptions.h`,
`SfLayoutDecl.h` in the same include tree]

**`dynB` proven, not guessed.** Of 2,516 exported kernels, **2,462 carry `dynB` and 54 do
not**. Dumping both variants of otherwise-identical names:

```
bmm_E2m1_E2m1E2m1_..._t128x8x512_s5_..._swiGlu_dynB_sm100f   →  mIsStaticBatch = 0,
                                                                mRouteImpl = RouteImpl(1) = Ldgsts,
                                                                mNumBatches = 128, mNumTokens = 2
bmm_E2m1_E2m1E2m1_..._t128x8x512_s4_..._clmp_sm100f          →  mIsStaticBatch = 1,
                                                                mRouteImpl = RouteImpl(0) = NoRoute,
                                                                mNumBatches = 2,   mNumTokens = 0
```
[verified — direct extraction from `flashinferMetaInfo.h`]

So `dynB` ⟺ *the per-batch token counts and the CTA→batch map live in device memory and the
grid is a static worst case with early exit*. `mNumTokens > 0` in the metadata is also the
flag that `routeAct` is on — i.e. the gather is fused into the GEMM's load path, which is why
there is no separate permute kernel in our trace.

### 3.2 The fused SwiGLU epilogue, exactly

The ABI documents the arithmetic, including the clamp/dequant algebra, which is worth
reproducing because getting the scale placement wrong is a silent accuracy bug:

```
// gatedActivation <- (x0 + beta) * activation(x1, alpha)
//   out_glu = x_glu * sigmoid(alpha * x_glu) * (x_linear + beta)
//
// with clamping, applied BEFORE dequantisation by folding dqAb into the limit:
//   x0 = clamp(x0, none, limit / dqAb)
//   x0 = x0 * dqAb;  x0 = x0 * sigmoid(alpha * x0)
//   x1 = clamp(x1, -limit / dqAb, +limit / dqAb)
//   scaleC = dqAb * qC;   beta' = beta / dqAb
//   out = scaleC * (x1 + beta') * x0
```
[verified — `BatchedGemmInterface.h`, `mPtrClampLimit` / `mPtrGatedActAlpha` /
`mPtrGatedActBeta` comments]

And the scale definitions:

```
scaleC    = dequantA * dequantB * quantC     // [B]  mPtrScaleC
scaleGate = dequantA * dequantB              // [B]  mPtrScaleGate
// for NvFp4, dequant maps [-448*6, 448*6] -> [-amax, amax]
```
[verified — same header]

SGLang builds exactly these three per-expert vectors: `g1_alphas` (gate half),
`g1_scale_c = w2_input_scale_quant * g1_alphas_up` (up half, folding GEMM2's input requant),
and `g2_alphas`. The comment in our tree explains why there are two: *"TRT-LLM dequantizes the
two halves of the fused GEMM1 separately."* [verified —
`flashinfer_trtllm.py::_compute_g1_scale_c`]

The epilogue also **re-quantises to NVFP4** (`mDtypeC = E2m1`, `mSfLayoutC = R8c4`), so GEMM2
consumes FP4 directly with no bf16 round trip. That is the single biggest structural advantage
of this path over a naive two-GEMM + separate-activation + separate-quant pipeline: it removes
one full write + one full read of `[padded_tokens, 2·I_r]` plus a quantisation kernel.

Supported activations in the exported cubins: `swiGlu`, `geGlu`, `relu2`, and identity;
FlashInfer maps `activation_type` 3/4/6/7 → Swiglu/Geglu/Relu2/Identity. [verified — kernel
name census + `flashinfer/fused_moe/core.py` docstring]

### 3.3 Block-scale layouts

The four layouts, verbatim from the generated header:

```
Linear : SF buffer is [m, ceil(n/b)];         SF(i,j) at (i, j/b)
R8c4   : [ceil(m/8),   ceil(n/b/4),  8, 4];   SF(i,j) at (i/8,  j/b/4,  i%8,        (j/b)%4)
R8c16  : [ceil(m/8),   ceil(n/b/16), 8,16];   niche: LowLatency FP4 weights, needs n % 256 == 0
R128c4 : [ceil(m/128), ceil(n/b/4), 32,4,4];  SF(i,j) at (i/128, j/b/4, i%32, (i%128)/32, (j/b)%4)
         // rows 0-31, 32-63, 64-95, 96-127 interleaved
```
[verified — `trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h`]

On our GEMM1 the metadata reads `mSfLayoutA = SfLayout(3) = R128c4`,
`mSfLayoutB = SfLayout(0) = Linear`, `mSfLayoutC = SfLayout(1) = R8c4`. Since `bN` makes A the
weights and B the activations, that means **weights use R128c4, activations use Linear, the
re-quantised GEMM1 output uses R8c4**. [verified]

This is precisely what SGLang does at load time: it quantises activations with
`nvfp4_quantize(..., sfLayout=SfLayout.layout_linear, per_token_activation=True,
backend="cute-dsl")` and passes them straight through. [verified — `flashinfer_trtllm.py`
`fused_experts_none_to_flashinfer_trtllm_fp4`]

The ABI's shape rules for the weight SF tensor are strict and worth writing down before you
build one:

```
R128c4:  paddedN % 128 == 0,  K % 64 == 0
         layout [paddedN/128, K/P/4, 512]; TMA view [paddedN/128, K/P/4, 2, 256]
R8c4:    paddedN % 8   == 0,  K % 64 == 0
         layout [paddedN/8, K/P/4, 32];    TMA view [paddedN/8, K/P/4/repeats, repeats*32]
                                            repeats = min(tileK/P/4, 8)
where P is the scaling block size (16 for NVFP4, 32 for MXFP4)
```
[verified — `BatchedGemmInterface.h`, `mPtrSfB` comments]

### 3.4 Weight preparation (offline, once)

`align_fp4_moe_weights_for_flashinfer_trtllm` does four things, in this order [verified —
`flashinfer_trtllm.py` lines 462–600, `quantization/utils.py::prepare_static_weights_for_trtllm_fp4_moe`]:

1. **Pad `intermediate_size`** to a multiple of 16 (gated) or 128 (non-gated), zero-filling
   both weights and scales. For GLM-5.2 at TP8, `I_r = 2048/8 = 256`, already aligned — no
   padding. [verified — `min_alignment = 16 if is_gated`, and `2048 % 8 == 0`]
2. **Reorder rows for the gated-act GEMM** (`reorder_rows_for_gated_act_gemm`) so gate and up
   halves interleave the way the epilogue expects.
3. **Shuffle for the transposed MMA output** with `epilogue_tile_m = 128` — the source
   carries the honest comment `# FIXME: this depends on the kernel internals`. This is the
   `mUseShuffledMatrix = 1` contract.
4. **Interleave the scales into R128c4** with `nvfp4_block_scale_interleave`.

Expected input shapes (FlashInfer's own docstring) [verified —
`flashinfer/fused_moe/core.py`]:

| tensor | shape | dtype |
|---|---|---|
| `hidden_states` | `[T, H/2]` | uint8 (packed FP4) |
| `hidden_states_scale` | `[T, H/16]` | float8_e4m3 |
| `gemm1_weights` | `[E, 2·I, H/2]` | uint8 |
| `gemm1_weights_scale` | `[E, 2·I, H/16]` | float8_e4m3 |
| `gemm2_weights` | `[E, H, I/2]` | uint8 |
| `gemm2_weights_scale` | `[E, H, I/16]` | float8_e4m3 |

### 3.5 How the kernel gets selected

Two-level. First a **tile ladder + heuristic**, then an **autotuner "tactic"**.

```cpp
static constexpr std::array<int32_t,6> mBaseSupportedTileNums = {8, 16, 32, 64, 128, 256};

float avg_tokens_per_expert = float(num_tokens * top_k) / num_local_experts;
int32_t tile = clamp(nextPowerOfTwo(avg_tokens_per_expert), front, back);
// candidate set = {tile-1, tile, tile+1, tile+2} steps on the ladder
```
[verified — `computeSelectedTileN` in
`python/sglang/kernels/ops/moe/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu`
lines 127–170]

The source itself flags the divergence that bites: *"AutoTuner maps raw `num_tokens` with
`last_positive_power_of_2` (round-**down**). Here we map derived `avg_tokens_per_expert` and
use `nextPowerOfTwo` (round-**up**). Because they round different quantities in different
directions, cache bucket and runtime tile candidates can diverge."* [verified — same file]

For GLM-5.2 at TP8, `num_local_experts = 256` (no EP), so
`avg_tokens_per_expert = T·8/256 = T/32`:

| tokens per forward `T` | avg/expert | selected `tileN` |
|---:|---:|---:|
| 4 (C1, EAGLE 3-1-4) | 0.125 | **8** (clamped to ladder front) |
| 192 (C64, MTP 2-1-3) | 6 | **8** |
| 256 | 8 | 8 |
| 768 (C256) | 24 | **32** |
| 4096 (prefill chunk) | 128 | 128 |

[verified formula; T values [inferred] from the benchmark modes]

That confirms what the trace shows: **`tileN = 8` for everything from C1 through C64**, and
our two decode kernels are the only MoE GEMMs that matter for the latency story.

The exported cubin pool gives 96 `E2m1_E2m1E2m1 … swiGlu … dynB` kernels across tiles
`t128x{8,16,32,64,128}x{256,512}` (± `u2`), of which the `tileN=8` rows have **only
`c1x1x1`** — no CTA-pair (`c2x1x1`) variant. CGA split-K variants `c1x1x2/3/4` do exist for
`t128x8x512`. [verified — name census of `flashinferMetaInfo.h`]

Finally the launcher asks the runner for a tactic:
`moe_runner->getDefaultValidConfigIndex(...)` when the autotuner has no cached entry, else the
cached tactic. SGLang's wrapper notes *"kernel tile config ('tactic') defaults to the runner's
built-in heuristic — pass an explicit one for tuned setups."* [verified —
`trtllm_fused_moe_kernel_launcher.cu` line 521; `trtllm_gen_moe.py` module docstring]

---

## 4. DeepGEMM

DeepGEMM is installed in our venv as `sgl_deep_gemm 0.1.5.post1`
(`/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/deep_gemm`), so this section is
read off the package, not a blog post.

### 4.1 Layouts and API

| layout | API (FP8/FP4) | when |
|---|---|---|
| **M-grouped contiguous** | `m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, ...)`, `..._nn_contiguous` | tokens already sorted by expert; each segment starts on an alignment boundary |
| **M-grouped masked** | `m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, ...)` | CUDA-graph decode: per-expert counts unknown at capture, passed as a device tensor `masked_m`; `expected_m` is the tuning hint |
| **K-grouped contiguous** | `k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, grouped_layout, ...)`, `..._nt_...` | MoE weight-gradient (training) |
| BF16 equivalents | `m_grouped_bf16_gemm_{nt,nn}_contiguous`, `m_grouped_bf16_gemm_nt_masked`, `k_grouped_bf16_gemm_tn_contiguous` | — |

Alignment is queryable, not hard-coded: `get_mk_alignment_for_contiguous_layout()` and
`get_theoretical_mk_alignment_for_contiguous_layout()`. Scale-factor relayout is
`transform_sf_into_required_layout(sf, mn, k, recipe, num_groups, is_sfa, disable_ue8m0_cast)`.
[verified — `deep_gemm/__init__.py` lines 172–262]

Newer knobs visible in the signatures and not widely documented: `use_psum_layout`,
`ensure_zero_padding`, `expected_m_for_psum_layout`, `compiled_dims='nk'`,
`disable_ue8m0_cast`. [verified — same file]

### 4.2 Blackwell / NVFP4 status

**Yes, and more than "supported".** The package exports `fp8_fp4_gemm_{nt,nn,tn,tt}` and
`m_grouped_fp8_fp4_gemm_*` as the *primary* names, with `m_grouped_fp8_gemm_nt_contiguous`
merely aliased to the FP4-capable entry point. [verified — `deep_gemm/__init__.py` lines
180–181: `m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous`]

Upstream README [reported — https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md]:
SM90 + SM100; SM90 needs CUDA 12.3+ (12.9+ recommended), SM100 needs 12.9+; SM100 uses UMMA
and supports all of NT/TN/NN/TT while SM90 is NT-centric; scale factors are fp32 on SM90 and
**packed UE8M0 (4 per `int`) on SM100**; JIT via NVCC or NVRTC (`DG_JIT_USE_NVRTC`,
`DG_JIT_CACHE_DIR`, `DG_JIT_DEBUG`). Note the UE8M0 detail: that is DeepSeek's own FP8
recipe, *not* NVFP4's E4M3-per-16 scales. Mixing them up is the classic error.

### 4.3 Mega MoE — the interesting part for us

`fp8_fp4_mega_moe` fuses **EP dispatch → linear1 (FP8×FP4) → SwiGLU → linear2 (FP8×FP4) →
EP combine** into one kernel over a symmetric-memory buffer. [verified — `deep_gemm/mega/__init__.py`]

```python
SymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
           hidden, intermediate_hidden, num_ring_tokens,
           mma_type='fp8xfp4', activation='swiglu')

fp8_fp4_mega_moe(y, l1_weights=(data, sf), l2_weights=(data, sf), sym_buffer,
                 cumulative_local_expert_recv_stats=None,
                 recipe=(1, 1, 32), activation='swiglu',
                 activation_clamp=None, fast_math=True)
```

Weight prep is `transform_weights_for_mega_moe`, which does two things:
`_interleave_weights(gran=8)` — reorder to `[gate 0..7 | up 0..7 | gate 8..15 | up 8..15 | …]`
instead of `[gate | up]` — and `_transpose_sf_for_utccp`, a
`(num_groups, mn/128, 4, 32, packed_k) → transpose(2,3)` shuffle so the scale factors can be
moved with **`tcgen05.cp` (UTCCP)** into TMEM. [verified — same file]

The buffer is a **ring**: `get_ring_limit_for_mega_moe(...)` sizes it, with an explicit
prefill/decode split at `num_max_tokens_per_rank >= 6144` and a decode budget the source
describes as "roughly ~18 GB". `activation='swiglu'` only. [verified]

SGLang already wires this up for DeepSeek V2/V4
(`python/sglang/srt/layers/moe/mega_moe.py`) behind
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS` → `DG_USE_FP4_ACTS=1` and
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND` → `DG_USE_MXF4_KIND=1`. [verified]

**Does it apply to GLM-5.2?** Not as-is. Mega MoE is an *EP* kernel — it fuses the dispatch and
combine, which only exist under expert parallelism. Our GLM-5.2 runs `ep_size: 1` with TP8.
The `mma_type='fp8xfp4'` recipe also wants **FP8 activations against FP4 weights**, whereas our
checkpoint is W4A4 NVFP4 with E4M3-per-16 scales, and the `recipe=(1,1,32)` default is a
group-32 (MX-style) contract. Porting it would mean adopting EP *and* changing the activation
quantisation. Worth doing only if §6's EP analysis says EP is right for the operating point.
[inferred from the API contracts above]

---

## 5. The batch-1 problem, with the arithmetic

### 5.1 Shapes and bytes

GLM-5.2, TP8: `H = 6144`, `I = 2048`, `I_r = I/TP = 256`, `E = 256`, `K = 8`,
75 MoE layers (`num_hidden_layers 78` − `first_k_dense_replace 3`) + 1 MTP layer.
NVFP4 = 4 bits data + one E4M3 scale per 16 elements = **0.5625 bytes/element**.

| per expert, per rank | elements | bytes |
|---|---:|---:|
| gate+up (`2·I_r × H`) | 3,145,728 | 1,769,472 (1.769 MB) |
| down (`H × I_r`) | 1,572,864 | 884,736 (0.885 MB) |
| **total** | 4,718,592 | **2,654,208 (2.654 MB)** |

| scenario | experts touched | bytes/layer | bytes/forward (75 layers) | floor @7.672 TB/s |
|---|---:|---:|---:|---:|
| 1 token, top-8 | 8 | 21.23 MB | **1.593 GB** | 207.6 µs → 4818 fwd/s |
| 4-token EAGLE verify | ≤32 | 84.93 MB | **6.370 GB** | 830.3 µs → 1204 fwd/s |
| C64+ (all experts hit) | 256 | 679.5 MB | **50.96 GB** | 6.64 ms → 151 fwd/s |

[verified arithmetic from the config; peak from the CUPTI device record]

Whole-model cross-check: `total_size` in the safetensors index is **464,795,267,072 B**;
computed routed-expert bytes are **413.1 GB**, leaving **51.7 GB** of non-expert weights
(= 6.46 GB per GPU: attention, dense layers 0–2, 76 bf16 shared experts, gates, lm_head).
[verified — `model.safetensors.index.json` metadata, `du -sh` = 433 GiB]

### 5.2 Arithmetic intensity — it is not close

GEMM1 at C1 does `2·T·K·(2 I_r H)` = 201.3 MFLOP per layer against 56.6 MB of weights.

| quantity | value |
|---|---:|
| AI of the expert GEMMs at C1 (`T=4`) | **3.56 FLOP/byte** |
| AI at C64 (`T=192`, all 256 experts) | **21.3 FLOP/byte** |
| Machine balance, dense NVFP4 (9 PFLOPS ÷ 7.672 TB/s) | **1173 FLOP/byte** |
| Tokens/expert needed to reach the ridge | ≈ 586 |
| Batch size that implies (`E/K · 586`) | **≈ 18,750 tokens per forward** |

[verified arithmetic; 9 PFLOPS/GPU dense FP4 is 72 PFLOPS ÷ 8 from the DGX B200 page —
[reported]]

So MoE decode is memory-bound by a factor of ~330× at C1 and ~55× at C64, and only prefill-scale
batches reach the compute roof. **There is no operating point in serving where MoE expert GEMMs
are compute-bound.** Every optimisation must be a bytes-moved or a bytes/second optimisation.

### 5.3 What fraction of DRAM peak we actually get

The honest caveat first: I know the CTA-tile count (32) but not the number of *distinct*
experts those 32 slots hit. If the four EAGLE tokens route identically, distinct experts could
be as low as 8 and L2 would absorb the rest. So the table is a range, and the top row is an
upper bound.

| assumed distinct experts | GEMM1 bytes | GEMM1 @11.79 µs | GEMM2 bytes | GEMM2 @10.06 µs |
|---:|---:|---:|---:|---:|
| 32 (no overlap) | 56.6 MB | **4.80 TB/s (62.6%)** | 28.3 MB | **2.81 TB/s (36.7%)** |
| 24 | 42.5 MB | 3.60 TB/s (46.9%) | 21.2 MB | 2.11 TB/s (27.5%) |
| 16 | 28.3 MB | 2.40 TB/s (31.3%) | 14.2 MB | 1.41 TB/s (18.4%) |

The structural ceilings, which do not depend on that unknown:

- GEMM1 launches **128 CTAs** with 202,640 B dynamic smem. `maxShmemPerSm = 232,448 B`, so
  **one CTA per SM**. 128 of 148 SMs → **86.5% of peak is the hard ceiling**, before any
  inefficiency. [verified — trace grid + smem, CUPTI device record]
- GEMM2 launches **1536 CTAs** with 184,464 B smem → also 1 CTA/SM, ~10.4 waves, so wave
  quantisation is *not* its problem.
- GEMM2's problem is `K`. `mTileK = 512` for GEMM1 with `K = H = 6144` → 12 K-iterations, and a
  5-stage pipeline has something to do. GEMM2 has `mTileK = 256` with `K = I_r = 256` → **one
  K-iteration**, and the kernel it selects advertises `s9` (9 stages). A 9-stage pipeline over
  a single iteration is pure prologue + epilogue. Per CTA it moves 128×256 FP4 + scales =
  **18,432 bytes**. [verified — kernel names, config, arithmetic]

**That is the single clearest kernel-level finding in this document: at TP8 the MoE down-projection
has exactly one K-tile, so it cannot pipeline, and it runs at roughly half GEMM1's bandwidth
while moving half the bytes.**

### 5.4 Is the whole thing pure HBM bandwidth? No.

Bytes per 4-token verify forward, per GPU:

```
  routed experts (≤32 distinct)        6.37 GB
  everything else (attn, dense, shared,
   gates, lm_head — read once)         6.46 GB
  ------------------------------------------
  total                               12.83 GB
  floor at 7.672 TB/s                  1.672 ms
  measured forward (TPOT 2.24 ms × accept 4.00)   ≈ 8.96 ms
  → fraction of the bandwidth roofline           18.7%
```

Measured DRAM Read counter over the same window: **17.2%**. The two numbers agree to within a
point and a half, which is the strongest statement in this section:

> **Essentially all DRAM traffic at C1 is weight traffic, and the decode step is ~5× longer
> than the weight-bandwidth floor.** GLM-5.2 at C1 on this box is latency-bound, not
> bandwidth-bound.

Supporting evidence in the same direction: 1,183,520 kernels on device 0 in a 7599 ms span =
**156 kernel launches per millisecond**, average kernel 7.6 µs, Tensor Active 4.1%.

Single-token, no-speculation ceiling for completeness: 8.05 GB/token → 1049 µs →
**953 tok/s** absolute upper bound from weight bandwidth alone. With EAGLE 4-token verify at
accept 3.16 the same arithmetic gives ~1890 accepted tok/s. Measured is 365–447 tok/s.
[verified arithmetic]

### 5.5 The speculative-decoding asymmetry

This deserves its own heading because it inverts a common intuition.

Speculative decoding amortises a weight read across the verified tokens **only if the tokens
share the weight**. Attention and dense weights are shared by construction: one read serves all
4 tokens. Expert weights are *not*: token `t` routes to its own 8 experts, so a 4-token verify
touches up to 32 expert slots and reads up to 4× the expert bytes.

Our grid confirms there is no dedup at the kernel level — GEMM1's grid is `(4, 32)` in
**every** one of the 40,312 decode launches, i.e. `gridY = T·K` exactly. [verified — grid
distribution query]

Consequences:

1. **The MoE share of the step grows with speculation depth.** Going 3-1-4 → 5-1-6 makes MoE a
   larger fraction, not a smaller one. The published GLM-5.2 recipe uses MTP 5-1-6 at C1
   (527 tok/s/GPU) and drops to 2-1-3 at C64 — and MoE weight traffic is one reason that
   direction is right.
2. **Expert-set overlap across speculative tokens is a free, unexploited win.** Adjacent tokens
   in a sequence plausibly share experts. If they do, the weights are already in L2 (the
   32-expert working set is 84.9 MB against a 126.5 MB L2) and the achieved bandwidth number in
   §5.3 is lower than the top row suggests. **This is measurable and has not been measured.**
   [inferred; measurement listed in the open questions]

---

## 6. Expert parallelism vs tensor parallelism

### 6.1 What changes

| | TP8 (what we run) | EP8 |
|---|---|---|
| expert sharding | every rank holds `1/8` of **every** expert (`I_r = 256`) | each rank holds **32 whole** experts (`I = 2048`) |
| GEMM1 `M` | `2·I_r = 512` → `gridX = 4` | `2·I = 4096` → `gridX = 32` |
| GEMM2 `K` | **256 — one tile** | **2048 — 8 tiles** |
| collective | all-reduce of `[T, H]` after GEMM2 | all-to-all dispatch before + all-to-all combine after |
| balance | perfect by construction | depends on routing |
| bytes read per rank at C1 | 84.9 MB (32 experts × 2.654 MB) | 84.9 MB (≈4 experts × 21.2 MB) *if balanced* |

Note the last row: **the total expert bytes are identical.** EP does not reduce weight traffic
at fixed batch; it redistributes it. What EP buys is (a) a fatter `K` and `M` per GEMM, and
(b) fewer experts resident per rank, which is what makes very large models fit. What it costs
is imbalance and all-to-all.

### 6.2 The imbalance math

Balls-in-bins with `T·K` slots into `EP` ranks, uniform routing (optimistic — real routing is
skewed and the expert→rank map is *fixed*, so a hot expert lands on the same rank every step):

| slots (`T·K`) | EP ranks | mean/rank | E[max] | imbalance |
|---:|---:|---:|---:|---:|
| 8 (`T=1`) | 8 | 1.00 | 2.60 | **2.60×** |
| 32 (`T=4`, our C1) | 8 | 4.00 | 7.05 | **1.76×** |
| 32 | 16 | 2.00 | 4.83 | 2.41× |
| 1536 (`T=192`, C64) | 8 | 192 | 211.9 | 1.10× |
| 1536 | 16 | 96 | 113.7 | 1.18× |
| 1536 | 32 | 48 | 62.9 | 1.31× |

[verified by Monte-Carlo, 20k trials for small cases]

Since the step waits for the slowest rank, EP8 at C1 costs ~1.76× on the MoE GEMM *before*
counting all-to-all. TP8 costs 1.00×. **This is the quantitative reason EP is wrong at C1.**

It also interacts with our existing worst problem: the ledger records that **47% of collective
time is rank-arrival skew** with a persistent pattern (rank 0 last in 24% of instances). EP
would *add* a structurally-imbalanced phase to a system already losing half its collective time
to imbalance. [verified —
`personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` §2a]

### 6.3 The collective profile change

Payload per MoE layer:

| | `T=4` (C1) | `T=192` (C64) |
|---|---:|---:|
| TP all-reduce payload (`T·H` bf16) | 49.2 KB | 2359 KB |
| EP dispatch (`T·K·H` NVFP4+sf) | 110.6 KB | 5308 KB |
| EP combine (`T·K·H` bf16) | 393.2 KB | 18,874 KB |

[verified arithmetic]

At C1, EP moves **~10× more bytes** than the TP all-reduce it replaces. Even on NVLink5 at
1.8 TB/s/GPU, the fixed latency of two extra dependent kernels per layer × 76 layers is the
dominant term, not the bytes. Our NVLink counters currently sit at 2.2–2.4% average
utilisation, so bandwidth is not the constraint either way. [verified]

### 6.4 DeepEP

Primary source: the DeepEP README's own tables.
[reported — https://raw.githubusercontent.com/deepseek-ai/DeepEP/main/README.md]

| Arch | NIC | Topo | Dispatch | Combine | #SMs |
|---|---|---|---:|---:|---:|
| SM90 | CX7 | EP 8×2 | 90 GB/s (RDMA) | 81 GB/s | 12 |
| SM90 | CX7 | EP 8×4 | 61 GB/s | 61 GB/s | 6 |
| **SM100** | CX7 | EP 8×2 | 90 GB/s (RDMA) | 91 GB/s | 12 |
| **SM100** | — | **EP 8 (NVLink)** | **726 GB/s** | **740 GB/s** | 64 (max perf) |
| **SM100** | — | EP 8 (NVLink) | 643 GB/s | 675 GB/s | **24 (min SMs)** |

V2 claims "up to 1.3× peak performance while saving up to 4× SM count" over V1, replaces the
NVSHMEM backend with an "NCCL Gin" header-only backend reusing existing NCCL communicators, and
adds analytical SM sizing instead of autotuning. Notably, **"0 SM RDMA low-latency EP is no
longer supported"** in V2, while three "0 SM" features remain (Engram/RDMA, PP/RDMA, CP/Copy
Engine). [reported — same README]

The V1 low-latency numbers, which are the ones relevant to decode
[reported — https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md; **H800 + CX7 IB
400 Gb/s, 128 tokens/batch, hidden 7168, top-8, FP8 dispatch, BF16 combine** — *not* B200,
*not* our hidden size]:

| #EP | dispatch latency | dispatch BW | combine latency | combine BW |
|---:|---:|---:|---:|---:|
| 8 | 77 µs | 98 GB/s | 114 µs | 127 GB/s |
| 16 | 118 µs | 63 GB/s | 195 µs | 74 GB/s |
| 32 | 155 µs | 48 GB/s | 273 µs | 53 GB/s |
| 64 | 173 µs | 43 GB/s | 314 µs | 46 GB/s |
| 128 | 192 µs | 39 GB/s | 369 µs | 39 GB/s |
| 256 | 194 µs | 39 GB/s | 360 µs | 40 GB/s |

Those are **inter-node RDMA** numbers. Our 8×B200 is a single NVLink domain, so the relevant row
is the SM100/NVLink one above, and the RDMA latencies do not apply. Still: 77+114 = 191 µs per
layer at EP8 over IB would be catastrophic at 76 layers; the intranode NVLink path is the only
one worth considering here.

Other verified mechanics: hook-based overlap — *"the actual tensor will not be received only if
you call `hook()`"* — lets the combine's receive be deferred without occupying SMs;
`num_max_dispatch_tokens_per_rank` should be < 256 for low-latency mode; SGLang asserts
`≤ 1024` because DeepEP's internode LL dispatch uses `FINISHED_SUM_TAG = 1024`.
[reported for the first two; verified for the last —
`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` line 382]

### 6.5 EPLB and redundant experts

DeepSeek's EPLB is vendored verbatim into our tree at
`python/sglang/srt/eplb/eplb_algorithms/deepseek.py` (header comment: *"copied from
https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package"*),
alongside `deepseek_vec.py`, `elasticity_aware.py`, an `lplb_solver.py`, an
`expert_distribution.py` recorder, an `expert_location_updater.py`, and an `eplb_simulator/`.
[verified — directory listing + file header]

The algorithm is two primitives:
`replicate_experts(weight, num_phy)` — replicate `num_log` logical experts to `num_phy`
physical replicas minimising the max replica load — and `balanced_packing(weight, num_packs)` —
pack `n` weighted items into `m` packs of exactly `n/m` items minimising max pack weight, by
greedy descending-weight assignment to the currently-lightest pack. Two policies:
*hierarchical* (when node count divides group count: balance groups across nodes, then
replicate within a node) and *global* (replicate globally, then distribute). [verified —
`deepseek.py`]

NVIDIA's wide-EP writeup adds the online variant: *static EPLB* uses precomputed mappings,
*online EPLB* "redistributes hot experts alongside cold experts" with "weight updates in a
non-blocking fashion by scheduling them between forward passes", and reports **up to 1.8×
higher per-GPU throughput for EP32 vs EP8 on DeepSeek-R1 at 100 tok/s/user** on GB200 NVL72.
[reported —
https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/]

**Relevance to us: none at present.** With `ep_size = 1` there is no expert→rank map to
rebalance. EPLB becomes relevant only if we adopt EP, which §6.2 argues against at C1 and §8
argues about at C64.

---

## 7. Expert offloading and prefetching

Node facts: PCIe **Gen5 ×16** on every GPU (`pcie.link.gen.max 5`, `pcie.link.width.max 16`,
current gen 5), **3905 GB** of host RAM, 2 NUMA nodes, Xeon Platinum 8581C. [verified —
`nvidia-smi`, `free -g`, `lscpu`]

Gen5 ×16 ≈ **63–64 GB/s** per direction per GPU. [reported — PCIe spec; not measured on this box]

| workload | expert bytes needed per forward per GPU | PCIe time | actual budget |
|---|---:|---:|---:|
| C1, 1 token, top-8 | 1.593 GB | **24.9 ms** | 2.24 ms TPOT |
| C1, 4-token verify | 6.370 GB | **99.5 ms** | ~8.96 ms |
| C64, all experts | 50.96 GB | **796 ms** | ~51.9 ms |

Offloading is **11×–15× too slow at every operating point**, and it gets worse with batch
because more distinct experts are touched. There is no prefetch schedule that fixes an order-of-
magnitude bandwidth deficit: to hide 24.9 ms of transfer behind a 2.24 ms step you would need to
prefetch ~11 layers ahead, and the router for layer `L+11` does not exist until layer `L+10`
has run. Predicting routing 11 layers ahead is a research problem, not an engineering one.

Where offloading *is* viable — and this is the honest counter-case — is when the model does not
fit at all and you are willing to pay it. On a Grace-Hopper/Grace-Blackwell part with NVLink-C2C
(~900 GB/s [reported], ~14× PCIe Gen5) the C1 single-token number becomes ~1.8 ms, which is the
same order as a decode step. **We do not have C2C**; our B200s are SXM modules in an x86 host.
So for this box: not viable, and the 433 GB NVFP4 checkpoint fits in 8×183 GiB anyway at
54 GB/GPU with ~129 GiB free for KV cache. [verified — `du`, `nvidia-smi`]

A cheaper cousin *is* worth doing: **L2-resident prefetch of the next layer's experts.** The
32-expert working set at C1 is 84.9 MB against a **126.5 MB L2** (verified from the CUPTI
device record). The routing decision for layer `L+1` is not available early, but the *shared*
expert's weights and the router gate for `L+1` are known unconditionally and are small. This is
speculative and unmeasured. [inferred]

---

## 8. Recommendation for GLM-5.2 (256 experts, top-8) on 8×B200

### 8.1 Keep TP8 for the MoE. Do not adopt EP at C1 or C64.

The arithmetic, in one table:

| | TP8 | EP8 |
|---|---|---|
| MoE weight bytes/rank, C1 4-token | 84.9 MB | 84.9 MB (**identical**) |
| load imbalance, C1 (32 slots / 8 ranks) | 1.00× | **1.76×** (uniform routing; worse in reality) |
| load imbalance, C64 (1536 slots / 8 ranks) | 1.00× | 1.10× |
| extra collective bytes/layer, C1 | 0 | +454 KB (dispatch+combine − allreduce) |
| GEMM2 `K` per rank | **256 = 1 tile (bad)** | 2048 = 8 tiles (good) |
| GEMM1 `gridX` | 4 (128 CTAs total — under one wave) | 32 (but only ~4 expert-tiles → ~128 CTAs too) |

EP fixes the one real kernel defect (GEMM2's single K-tile) and breaks two things that are
currently perfect (balance, and no extra collective). At C1 that trade is clearly bad — 1.76×
on the dominant MoE kernel plus two more dependent kernels per layer against a step that is
already latency-bound. At C64 the imbalance drops to 1.10× and the trade becomes arguable, but
C64 is not where our headline number lives.

**Verdict: TP8. Revisit EP only at C256+ or if we ever run a model that does not fit.**

### 8.2 Fix GEMM2's single-K-tile problem without changing parallelism

Ranked by expected value ÷ effort:

1. **Force a different tile for GEMM2 via the tactic.** The heuristic picks `tileN = 8` for both
   GEMMs. GEMM2 has `M = 6144` (48 M-tiles) and `K = 256`; a `t128x8x256` with `s9` wastes its
   pipeline. Sweep the four candidates the launcher already produces
   (`{tile-1, tile, tile+1, tile+2}` on the `{8,16,32,64,128,256}` ladder) and the
   `u2` / `schedS` / `schPd2x1x2x3` variants via the autotuner, measuring GEMM2 alone.
   **Expected:** if GEMM2 goes from 36.7% → 60% of peak, it drops 10.06 → 6.15 µs, saving
   **293 µs per forward** ≈ 3.3% of step time. Cost: an autotune run. [inferred; the tile
   ladder and candidate-set logic are verified]
2. **Quantise the shared expert.** It is BF16 `[2048,6144]`×2 + `[6144,2048]` per layer, i.e.
   **9.44 MB per rank per layer** unquantised against 21.2 MB for all eight routed experts.
   NVFP4 would cut it to 2.65 MB. Saving: 6.79 MB × 76 layers = **516 MB per forward** ≈
   **67 µs at peak bandwidth**, plus it moves the shared expert onto the same fused kernel
   family. Cost: a re-quantisation pass and an accuracy check. [verified shapes; [inferred]
   saving]
3. **Attack the 5.25 µs single-CTA routing kernel.** 76 launches/forward × 5.25 µs =
   **399 µs per forward, 2.6% of all GPU time**, on one SM. At `T=4` the whole job is a
   256-bin histogram over 32 entries plus a scan. Options: (a) fuse routing into the tail of
   the preceding norm/quant kernel with PDL; (b) use the cluster variant unconditionally at
   decode (it is already compiled — 684 launches in our trace at grid `(8,1,1)`, same 5.16 µs,
   so this alone will not help); (c) write an SGLang-native fused
   `sigmoid+bias→top8→scaled-sum-norm` for `E=256, K=8, T≤8` that finishes in ~1–2 µs.
   **Expected:** 200–300 µs/forward ≈ 2.5% of step. [verified cost; [inferred] fix]
4. **Fuse the combine into the GEMM2 epilogue.** The ABI already supports it
   (`mPtrExpertWeightsPtr`, `mPtrPermutedIdxToExpandedIdx`, `mUseCMultiCast`, multicast
   completion barriers). Our `finalizeKernel` costs 3.93 µs × 76 = **299 µs/forward** (2.2% of
   GPU time). SGLang's `do_finalize=False` + deferred-finalize path is already wired; check
   whether it is on. **Expected:** most of 299 µs. [verified header + SGLang plumbing;
   [inferred] saving]
5. **Raise GEMM1's CTA count.** 128 CTAs on 148 SMs caps it at 86.5%. The `c1x1x2/3/4` CGA
   split-K variants of `t128x8x512` exist in the cubin pool. Split-K 2 would give 256 CTAs
   across two waves — the same 86.5% arithmetic, so this is probably *not* the win; the real
   win would be a `tileM = 64` variant giving 8×32 = 256 CTAs, which **does not exist in the
   exported pool** (`mTileM` is 128 for every `E2m1` bmm kernel). Writing one is a CUTLASS job.
   [verified the pool census; [inferred] the analysis]

Aggregate of items 1–4: `293 + 67 + 250 + 299 ≈ **909 µs per forward** out of ~8.96 ms, i.e.
**~10%**, without touching parallelism or the checkpoint format. At C1 that maps to roughly
365 → **406 tok/s** on sharegpt. Every term is an estimate built on a measured cost and an
assumed efficiency gain; none is a promise. [inferred, arithmetic shown]

### 8.3 At C64

At `T ≈ 192` tokens/forward the picture changes in three ways, all verifiable from the same
formulas:

- **Every expert is hit.** `E[distinct] = 256·(1 − (1 − 8/256)^192) = 255.4` of 256. The MoE
  reads its entire 50.96 GB per forward, at a 6.64 ms floor. [verified]
- **The tile stays at 8.** `avg_tokens_per_expert = 192·8/256 = 6 → nextPow2 = 8`. The tile only
  moves to 32 above ~512 tokens/forward. So the same two kernels run, with
  `gridY = Σ_e ceil(M_e/8) ≈ 256` (every expert gets one 8-token tile), i.e. GEMM1 goes from
  128 to ≈ `4 × 256 = 1024` CTAs and finally clears one wave; GEMM2 goes to ≈ `48 × 256 =
  12,288` CTAs. [verified formula; CTA counts [inferred] from it]
- **Padding waste appears:** 256 experts × 8-token tiles = 2048 padded slots for 1536 real =
  33%. It costs activation traffic and MMA, both negligible; it does **not** cost weight
  traffic. [verified]

Weight-bandwidth floor at C64: 50.96 GB (experts) + 6.46 GB (rest) = **57.4 GB → 7.48 ms**
against a measured forward of ~51.9 ms (TPOT 17.41 ms × accept 2.98, coding workload). That is
**14.4% of the roofline** — the same story as C1. [verified arithmetic; measured numbers from
`/home/aman/code/benchmark/RESULTS.md`]

Recommendation at C64: same list, same order, plus **take the C64 profile**. Every number in
§5–§8 is derived from a C1 capture; the ledger explicitly flags that the C64 profile has not
been taken and will redistribute the table. Two things specifically worth measuring there:
whether GEMM2's efficiency improves once it has 10× the CTAs (it should not — the K problem is
unchanged), and whether the padding waste starts to matter once tokens/expert crosses the tile
boundary.

### 8.4 What not to do

- **Do not adopt DeepGEMM Mega MoE for GLM-5.2 as it stands.** It is an EP kernel
  (fuses dispatch/combine) with an FP8×FP4 recipe and group-32 scales; we run EP1 and W4A4
  NVFP4 with group-16 E4M3. The port is a parallelism change plus a quantisation change, and
  §6 says the parallelism change is wrong at our operating points. [inferred from the verified
  API contracts]
- **Do not offload experts.** §7.
- **Do not chase the MoE GEMMs' tensor-core utilisation.** At 3.56 FLOP/byte against a 1173
  FLOP/byte machine balance, tensor cores are irrelevant here; every microsecond is memory
  and launch latency.
- **Do not assume "MoE at batch 1 is pure HBM bandwidth."** On this box, with TP8 and a 4-token
  EAGLE verify, the measured average DRAM read is 17.2% and the arithmetic agrees. The widely
  repeated framing is right about the *arithmetic intensity* and wrong about the *bottleneck*.

---

## 9. Open questions (each needs a measurement, not an argument)

1. **How many distinct experts do the 4 EAGLE tokens actually touch?** Everything in §5.3 is a
   range because of this. Measure with `routing_replay_out` — FlashInfer's
   `trtllm_fp4_block_scale_moe` takes an optional `int16[num_tokens, top_k]` buffer that
   captures the selected expert ids at zero cost when null. [verified — the parameter exists]
2. **What is `dram__bytes_read.sum` for the two decode GEMM kernels?** An `ncu` run on
   `bmm_E2m1_..._swiGlu_dynB_sm100f` and `bmm_Bfloat16_E2m1E2m1_..._s9_...` settles §5.3
   outright and gives a defensible fraction-of-roofline per the ledger's rule 2.
3. **Does the autotuner ever pick a non-default tactic for GEMM2 at `tileN=8`?** The cache-bucket
   vs runtime-candidate rounding mismatch is documented in the launcher's own comment; whether
   it costs us anything is unmeasured.
4. **Is `do_finalize=False` / deferred finalize enabled in our config?** If not, item 4 in §8.2
   may be a config flag rather than a kernel.
5. **What does the C64 MoE profile look like?** Not captured.
6. **Exact per-rank weight-byte accounting.** I derived non-expert bytes as
   (total_size − routed) / 8 = 6.46 GB/GPU, which assumes clean TP8 splitting of everything
   non-expert. Which attention tensors are replicated vs sharded in SGLang's GLM path
   (`q_a_proj` in particular) was not verified.
7. **PCIe Gen5 ×16 achieved bandwidth on this box** — quoted from spec, not measured (§7).
8. **Whether the trtllm-gen `mUseCMultiCast` fused-finalize path is reachable from FlashInfer's
   Python API at all.** The ABI has it; the exported wrapper may not expose it.

---

## Sources

### Read on this machine (primary)

- `/home/aman/code/weights/GLM-5.2-NVFP4/config.json` — model geometry, routing config
- `/home/aman/code/weights/GLM-5.2-FP8/config.json` — MoE fields (`n_routed_experts`,
  `moe_intermediate_size`, `topk_method`, `routed_scaling_factor`, …)
- `/home/aman/code/weights/GLM-5.2-NVFP4/model.safetensors.index.json` — tensor census,
  `total_parameters` 380,989,135,104, `total_size` 464,795,267,072
- `/home/aman/code/weights/GLM-5.2-NVFP4/model-00006-of-00047.safetensors` (header) — shapes and
  dtypes of `mlp.gate.weight`, `mlp.shared_experts.*`, `self_attn.*`
- `/home/aman/code/benchmark/runs/sweep-latency-3-1-4/trace.sqlite` — kernel names, launch
  counts, durations, grids, shared memory, registers; `GPU_METRICS`; `TARGET_INFO_GPU`
- `/home/aman/code/benchmark/RESULTS.md` — measured TPOT / accept length / tok/s
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/glm-5.2-optimization-log.md`
- `/home/aman/.cache/flashinfer/cubins/b368d003e8fdfe4b271bff7c788ac52ef789a81b/batched_gemm-da58956-b4ac80e/include/flashinferMetaInfo.h`
  — 2,516 exported trtllm-gen bmm kernels with full option dumps
- `…/trtllmGen_bmm_export/BatchedGemmInterface.h` — the grouped-GEMM ABI (route map, CTA→batch
  map, `numNonExitingCtas`, SF layouts, SwiGLU/clamp algebra, MoE-finalize params)
- `…/trtllmGen_bmm_export/BatchedGemmOptions.h` — `enum class BatchMode { BatchM, BatchN }`
- `…/trtllmGen_bmm_export/BatchedGemmEnums.h` — `enum class RouteImpl { NoRoute, Ldgsts, Tma, LdgPlusSts }`
- `…/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h` — Linear / R8c4 / R8c16 / R128c4
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flashinfer/fused_moe/core.py`
  — `trtllm_fp4_block_scale_moe` signature + docstring, routing-method table
- `…/flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu`
  — routing-kernel selection thresholds
- `…/flashinfer/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu`
  — activation / permute / finalize kernels
- `…/flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh`
  — cluster histogram via `cg::cluster_group::map_shared_rank`, CUB `ExclusiveSum`
- `/home/aman/code/NotSglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- `/home/aman/code/NotSglang/python/sglang/srt/layers/quantization/utils.py`
  (`prepare_static_weights_for_trtllm_fp4_moe`)
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/moe/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu`
  (`computeSelectedTileN`, tile ladder, tactic resolution)
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/moe/trtllm_gen_moe.py`
- `/home/aman/code/NotSglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `/home/aman/code/NotSglang/python/sglang/srt/eplb/eplb_algorithms/deepseek.py`
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/deep_gemm/__init__.py`
  and `deep_gemm/mega/__init__.py` (version 0.1.5.post1)
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/`
  — `tcgen05_{mma,mma_ws,alloc,cp,ld,st,commit,wait,fence,shift}.h`,
  `tensormap_replace.h`, `tensormap_cp_fenceproxy.h`, `cp_async_bulk_tensor.h`
- `nvidia-smi`, `lscpu`, `free -g` on the node

### Web (fetched and read)

- https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md
- https://raw.githubusercontent.com/deepseek-ai/DeepEP/main/README.md
- https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md
- https://github.com/NVIDIA/cutlass/blob/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu
- https://mufeezamjad.com/blog/nvfp4-group-gemm
- https://deepwiki.com/deepseek-ai/DeepGEMM
- https://www.nvidia.com/en-us/data-center/dgx-b200/
- https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/
- https://github.com/deepseek-ai/EPLB/blob/main/README.md (via search summary)
- https://docs.nvidia.com/cuda/cublas/index.html — confirmed that
  `cublasGemmGroupedBatchedEx()` and `cublasLtGroupedMatrixLayoutCreate()` /
  `cublasLtGroupedMatrixLayoutInit()` exist in the CUDA 13.x cuBLAS API. **Their signatures,
  host-vs-device shape residency, and whether they support FP4 block scaling could not be
  retrieved** — the rendered pages returned only the table of contents. Not sourced; do not
  assume cuBLAS grouped GEMM is usable for MoE decode until someone reads the real header.

### Explicitly not sourced

- Per-GPU dense FP4 TFLOPS is derived by dividing the DGX B200 system figure by 8
  (72 PFLOPS ÷ 8 = 9 PFLOPS/GPU). NVIDIA's per-GPU datasheet was not read.
- PCIe Gen5 ×16 = 63–64 GB/s is a spec figure, not measured on this node.
- NVLink-C2C ≈ 900 GB/s (§7 counter-case) is recalled, not sourced, and is irrelevant to this
  box in any case.
- No paper is cited for expert offloading; the section is pure arithmetic against our measured
  link bandwidth, which is stronger than a citation would have been.
