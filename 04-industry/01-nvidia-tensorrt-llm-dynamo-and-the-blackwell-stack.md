# NVIDIA: TensorRT-LLM, Dynamo, trtllm-gen kernels and the official Blackwell playbook

## What this is

A mining pass over everything NVIDIA has published about LLM inference engineering on
Blackwell, read for mechanisms we can port into our SGLang-derived engine on 8×B200
running GLM-5.2 (MoE 256/8, DSA sparse MLA, NVFP4/FP8, TP8, EAGLE 3-1-4).

Sources actually read (not summarised from memory): the 26 engineering write-ups in
`docs/source/blogs/tech_blog/` in the TensorRT-LLM repo, the repo's feature docs and
per-model deployment guides, the C++/Python source for the AllReduce, trtllm-gen FMHA
and batched-GEMM kernel selectors, `envUtils.cpp`, the AutoTuner, the NVIDIA Dynamo
documentation tree (including its GLM-5.2, GLM-5, Kimi-K3 and Qwen3.8 recipes), several
NVIDIA developer-blog posts, the MLPerf Inference v6.0 NVIDIA submission tree, the CUDA
Blackwell tuning guide, CUTLASS changelog, and CUDA/Nsight release notes.

Evidence labels used throughout: **[verified]** = I read it at the URL given.
**[reported]** = the vendor claims it and I read the claim but it is not independently
reproduced. **[inferred]** = my reasoning on top of verified facts. **[unverified]** =
I could not source it.

A standing caveat on every number below: all of it is NVIDIA-produced benchmarking of
NVIDIA hardware, mostly with the config chosen to flatter the result. Where the config
is asymmetric (different TP, different ISL/OSL, spec decoding on in one arm and off in
the other) I say so.

---

## Bottom line for our system

Ranked by expected effect on our two objectives (C1 single-stream latency; cost/user at
C64), with difficulty in our SGLang fork.

| # | What to steal | Expected effect | Difficulty | Why |
|---|---|---|---|---|
| 1 | **Test TP4 (2 replicas) instead of TP8 for the min-latency build** | Potentially large on C1; unknown sign | Medium (config + memory fit) | Two independent NVIDIA sources put their *own* min-latency DSA/GLM configs at 4 GPUs, not 8: TRT-LLM's DeepSeek-V3.2 min-latency benchmark is `TP=4` with MTP-3, and Dynamo's GLM-5.2 B200 aggregated recipe is `DTP4` on 4×B200. Our collectives are 19.6% of C1 and 47% of that is rank-arrival skew — halving the participant count attacks both terms. Counter-force: per-rank weight traffic doubles. Must be measured, not assumed. |
| 2 | **Fuse_A-style weight-prefetch GEMM + custom RouterGEMM for tiny-M** | Attacks the 37.1% dense-GEMM slice head-on | Medium-High (kernel work) | At C1 with EAGLE 3-1-4 we are at M≈4–5 tokens. NVIDIA reports a custom fused QKV/KV-down GEMM that prefetches most of its weights into shared memory under PDL, "substantial improvements over default GEMM when num_tokens < 16", and a RouterGEMM that beats cuBLAS for num_tokens ≤ 30. On DeepSeek-V4's 256-expert router they took the router GEMM from ~8 µs to ~3 µs at 1–16 rows — our exact shape. |
| 3 | **Collapse AllReduce count per layer via one-shot AR + RMSNorm + NVFP4-quant fusion** | Directly on the 19.6% collectives | Medium (we already run their kernels) | Our profile already shows `oneshotAllreduceFusionKernel`. The published pattern set goes further: `kARResidualRMSNormFP4Quant` and `MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM` collapse the MoE local-reduce + AR + residual + norm + quantize into one kernel. Fewer kernels = fewer sync points = less arrival skew. |
| 4 | **`TLLM_NUMA_AWARE_WORKER_AFFINITY=1` equivalent — pin ranks to the right NUMA node** | Plausibly several % of the skew term | Low | We have 2 NUMA nodes. NVIDIA ships an explicit doc on this and warns that memory touched before affinity is set lands on what becomes the remote node. Cheap to test. |
| 5 | **`nvidia-smi boost-slider --vboost 4` + persistence-mode cycle** | +3.4% measured on their C1 run | Trivial | Blog01 shows 204 → 211 tok/s from clock/power tuning alone on 8×B200. |
| 6 | **Suffix-automaton speculation stacked on EAGLE-3 (`use_sa_spec`, `sa_spec_threshold=4`)** | Free acceptance-length on prompt-echoing content | Low-Medium | AA's harness is ~10k input; suffix-automaton drafting mines the prompt for free draft tokens and TRT-LLM explicitly supports combining it with EAGLE-3 and MTP. |
| 7 | **Go *exact* on the DSA index instead of `index_topk_freq=4`** | Removes an accuracy risk while keeping speed | Medium | NVIDIA's GVR reuses the previous step's top-k as a *warm start* and then verifies + refines, so the output is bit-equivalent to `torch.topk`, at 1.57–2.02× the radix-select baseline. That is strictly better than periodic index reuse, which is an approximation. Caveat: their kernel only engages above 12,288 tokens by default and only at `index_topk=2048`. |
| 8 | **MoE-as-dense-GEMM at C16–C64, not at C1** | Up to 1.12× on the MoE module in-band | Medium | Their measured window on B200/NVFP4/hidden=7168/256 experts/top-8/TP8 is 64–208 tokens, peaking at 128, and it *loses* below 32 and above 272. At C1 (4–8 tokens) it is a trap. At C16 with 3-1-4 spec decode we are at ~80 tokens — dead centre. |
| 9 | **CUDA-graph batch-size buckets at +64 granularity above 128** | up to 1.3× agg, 1.5× disagg at high concurrency | Low | Default doubling leaves up to 50% padding waste between 128 and 2048. Cost is ~260 MB and 1.17× startup. Purely a C64 lever. |
| 10 | **Piecewise CUDA graph over the prefill** | Attacks our 189 ms TTFT | Medium | `prefill_cuda_graph_backend: piecewise` runs attention eagerly and graph-captures everything else, removing host launch overhead from a 10k-token prefill. |
| 11 | **Mixed MoE parallel: attention TP8 but MoE TP4×EP2** | +18% measured on their C1 run | Medium | Blog01: EP4TP2 → EP2TP4 moved 253 → 299 tok/s at C1 on 8×B200. This is a pure sharding change, not a kernel change. |
| 12 | **Relaxed acceptance during the thinking phase** | +8–9% acceptance length | Low | `relaxed_topk: 10, relaxed_delta: 0.6` gave 2.82 → 3.10 acceptance on MTP-3. But it costs MMLU-Pro 84.0 → 81.2. Only defensible if the leaderboard's accuracy run is separate from its speed run. |
| 13 | **DFlash block-diffusion drafting to replace EAGLE-3** | 1.5× over EAGLE-3 at equal interactivity (their claim) | High (needs a trained draft head) | Already integrated into SGLang, and NVIDIA says it swaps for EAGLE-3 by config change. But the 20 released checkpoints cover Qwen/Kimi/Llama/Gemma/gpt-oss — **not GLM**. We would have to train it. |
| 14 | **`TRTLLM_GEN_ENABLE_TILE_SIZE_KV64` — try the KV64 attention tile** | Small, but cheap to test | Trivial | Our decode kernel is `...Q8Kv128...`. There is a documented env flag to switch the trtllm-gen generation tile to KV=64, which is the right shape for very short decode batches. |
| 15 | **Attention-DP arrival-skew balancing (`enable_balance`, `timeout_iters`, `batching_wait_iters`)** | 1.33× throughput, balance 54% → 88% | Medium | Only applies if/when we run attention-DP at C64. It costs TTFT. |

Two things explicitly **not** worth our time, on the evidence:

- **MNNVL AllReduce.** TRT-LLM gates it on `aarch64` (or a test override) *and* multi-node.
  We are x86, single node, one NVLink domain. Not applicable.
- **Wide-EP / EPLB.** NVIDIA's own authors write that large-scale EP only pays when MoE
  GroupGEMM dominates end-to-end time. Our MoE GEMMs are 19.4% at C1, and we have 8 GPUs
  in one node. The gains they publish (up to 6.17× per-GPU throughput) are EP4/EP8 → EP16/EP32
  on GB200 NVL72, a different machine.

---

## TensorRT-LLM: the engine

### What they run, and the PyTorch rewrite

TensorRT-LLM is no longer a TensorRT graph-compilation product. **[verified]** The 1.2
release notes record it as a breaking change: *"PyTorch is now the sole execution backend"*
with the TensorRT backend completely removed. 1.0 had already designated the PyTorch
architecture as the stable default.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/release-notes.md)

The architecture doc **[verified]** describes the resulting runtime as `PyExecutor`
(orchestrator) → `Scheduler` (a `CapacityScheduler` deciding resource availability plus a
`MicroBatchScheduler` deciding admission) → `PyTorchModelEngine` (forward pass) →
`ResourceManager` / `KVCacheManager` (C++ core, Python-overridable). The stated rationale
is extensibility — "customization is possible" through Python interfaces — rather than a
performance argument.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/torch/arch_overview.md)

**[inferred]** The honest read is that NVIDIA converged on the same architecture SGLang and
vLLM already had — eager PyTorch dispatch, custom ops for the hot paths, CUDA graphs for
the decode step, torch.compile for peephole fusion — because ahead-of-time whole-graph
compilation could not keep up with model churn (MTP, MLA, DSA, mHC all arrived inside 18
months). For us this is reassuring: nearly everything below is a *kernel or a scheduling
policy*, not an engine-architecture assumption, so it ports.

### The min-latency playbook (this is the closest published analogue of our C1 problem)

**[verified]** `blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md`
is a step-by-step ledger from 67 to 368 tok/s on 8×B200, DeepSeek-R1, ISL 1024 / OSL 2048,
concurrency 1. Every row is an attributable optimization:

| Step | tok/s | Mechanism |
|---|---|---|
| Baseline (CUDA Graph + EP8TP8) | 67 | — |
| Multi-stream expert overlap | 73 | shared expert on a second stream vs sparse experts |
| MLA kernel optimization | 80 | custom MLA gen kernel; MLA at TP8 behaves as MQA with 16 q-heads |
| TopK kernel optimization | 84 | `RoutingKernelTopK.cuh`, `noAuxTcKernels.cu` |
| Fuse_A GEMM optimization | 89 | weight prefetch into smem, enabled by PDL |
| MTP3 vanilla | 154 | 3 draft tokens |
| MTP3 autoregressive + RouterGEMM | 164 | |
| Fused AllReduce + RMSNorm | 168 | `allReduceFusionKernels.cu` |
| PDL | 173 | `export TRTLLM_ENABLE_PDL=1` |
| Multi-stream RMSNorm overlap | 180 | two RMSNorms concurrently |
| MTP3 autoregressive | 204 | |
| Clock/power tuning | 211 | `nvidia-smi boost-slider --vboost 4` |
| CUTLASS grouped GEMM opt | 236 | not open-sourced at time of writing |
| Sparse experts as GEMMs | 249 | send all tokens to every activated expert, mask outputs |
| EP4TP2 | 253 | `--tp 8 --ep 4` |
| TRTLLM backend, EP2TP4 | 299 | `--tp 8 --ep 2`, PR #4280 |
| Fuse_A + RouterGEMM opt | 340 | WIP at time of writing |
| Relaxed acceptance | 368 | |

(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md)

Mechanisms worth extracting in detail:

- **Mixed parallel strategy.** Attention TP8; MoE sparse experts TP4×EP2; MoE shared
  experts TP8; Fuse_A GEMM DP8; RouterGEMM DP8. Precision is *mixed*: attention BF16,
  dense FFN NVFP4, MoE FFN NVFP4, MTP layers BF16, RouterGEMM BF16 in / FP32 out.
  **[inferred]** Keeping attention and the MTP/draft layers in BF16 at C1 is a deliberate
  accuracy-for-free trade: at batch 1 those modules are launch- and bandwidth-bound, not
  tensor-core-bound, so FP4 buys nothing there but costs acceptance rate.
- **Fuse_A GEMM.** Concatenate `[W_DQ, W_DKV, W_KR]` into one weight, then use a custom
  kernel that *prefetches the majority of its weights into shared memory* under PDL. Beats
  the default GEMM for `num_tokens < 16`. This is the single most directly transferable
  idea for our 37.1% dense-GEMM slice.
- **RouterGEMM.** Auto-generated by an internal code generator, BF16 in / FP32 out for
  numerical stability, beats default GEMM for `num_tokens <= 30`.
- **Fusion patterns named:** two overlapped RMSNorms → one GroupedRMSNorm;
  `(LocalReduction) + AllReduce + RMSNorm + DynamicQuant(bf16→nvfp4)` → one kernel;
  grouped GEMM FC1 + gated activation → one kernel when `moe_backend=TRTLLM`.
- **13-module decomposition of one MoE layer** is published (Fuse_A GEMM → 2×RMSNorm →
  UQ_QR_GEMM → UK_BGEMM → concat KV + RoPE → genAttention → UV_GEMM → WO_GEMM → fused
  AR/norm/quant → RouterGEMM+topK → shared expert → sparse experts → final fused
  reduce/AR/norm). Useful as a checklist against our own profile.
- **Negative results, stated:** MTP beyond 3 layers regresses (MTP4 245 vs MTP3 253);
  weight absorption is deliberately avoided in some module pairs to prevent weight-size
  inflation; two of the biggest wins (CUTLASS grouped GEMM, sparse-experts-as-GEMMs) were
  not open-sourced at publication.

**Reproduction config** is published separately **[verified]**: `TRTLLM_ENABLE_PDL=1`,
`moe_config.backend: TRTLLM`, `speculative_config: {decoding_type: MTP, num_nextn_predict_layers: 3}`,
`trtllm-bench --model nvidia/DeepSeek-R1-FP4 throughput --num_requests 10 --concurrency 1
--max_batch_size 1 --tp 8 --ep 2`, reporting **274.7 tok/s/user**. Note this is a *lower*
number than blog01's 368 — different TRT-LLM version and dataset. That discrepancy is
itself a useful reminder that these figures are not stable across releases.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md)

### The other min-latency record: Llama-4 Maverick, >1000 tok/s/user on 8×B200

**[verified]** `blog06` publishes the complete config for the ">1000 TPS/user on 8×B200"
claim. It is worth reading purely for which knobs they turn *off*:

```yaml
enable_autotuner: false
enable_attention_dp: false
enable_min_latency: true
cuda_graph_config:
  max_batch_size: 8
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 3
  speculative_model_dir: /config/models/eagle
  eagle3_one_model: true
kv_cache_config:
  enable_block_reuse: false
```
plus `tp_size 8`, `ep_size 1`, `TRTLLM_ENABLE_PDL=1`,
`TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=True`, KV fraction 0.75. Target
Llama-4-Maverick-17B-128E-Instruct-FP8 with a BF16 Eagle3 draft.
NVIDIA's own caveat is quoted in the doc: the setup "is designed to maximize single-user
performance rather than high-concurrency throughput," with rapid degradation as
concurrency rises.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog06_Llama4_maverick_eagle_guide.md)

**[inferred]** Three transferable signals: (a) `enable_min_latency: true` is a distinct
engine mode, not just a batch size; (b) `eagle3_one_model: true` — fusing the draft head
into the target engine rather than running two engines removes a launch/sync boundary per
step, which matters far more at 2.7 ms TPOT than at 30 ms; (c) they turn the **autotuner
off** for min latency, because tactic search at tiny M is noise-dominated and the fallback
tactic is often the right one. Worth A/B-ing in our stack.

---

## trtllm-gen: the kernel generator whose kernels we are already running

Our profile shows `oneshotAllreduceFusionKernel`, `twoshotAllreduceKernel`,
`bmm_E2m1_E2m1E2m1_Fp32_..._swiGlu_dynB_sm100f`, and
`parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen`. Here is what those actually are.

### Where they live

**[verified]** `cpp/tensorrt_llm/kernels/trtllmGenKernels/` contains five kernel families:
`batchedGemm`, `blockScaleMoe`, `fmha`, `gemm`, `gemmGatedAct`. Each ships as an
auto-generated *export* directory containing option/enum headers, a `KernelMetaInfo.h`, a
`config.json`, and a `cubins/` tree of pre-compiled binaries. There is no CUDA source for
the kernel bodies in the open repo — trtllm-gen is a closed generator whose output is
shipped as cubins plus a runtime selector.
(https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/trtllmGenKernels)

The same cubins are re-exported by FlashInfer, which is how SGLang-derived stacks end up
running them.

### FMHA kernel naming and selection — decoded

**[verified]** `fmha/fmhaKernels.h` builds a 64-bit hash key over exactly these fields:

```
Bit  0-3 : QKV layout          Bit 34-41: headDimV >> 3
Bit  4-7 : mask type           Bit 42-43: tileSizeKv >> 6
Bit  8-11: kernel type         Bit 44-48: log2(numTokensPerPage)
Bit 12-15: tile scheduler      Bit 49-52: log2(tileSizeQ)
Bit 16-17: MultiCtasKvMode     Bit 53   : reuseSmemKForV
Bit 18-25: headDimPerCtaV >> 3 Bit 54   : uses2CtaMma
Bit 26-33: headDimQk >> 3      Bit 55-56: sparseAttention
                               Bit 57   : skipsSoftmax
                               Bit 58   : fusesDsv4InvRopeFp8Quant
```

Selection path: `parseOptionsFromRunnerParams()` → `checkFmhaOptions()` →
`updateFmhaOptions()` → `computeNumCtas()` → cubin lookup by hash, with an NVRTC JIT
fallback when no cubin exists. There is an `FmhaAutoTuner` whose decision is "relative to
number of numCtas/numCtasPerSeqKv, which translate to batch size, seqLenQ and seqLenKv",
and whose warm-up grid is dense over batch 1–24 and sparse thereafter — the code comments
that the autotuner "is sensitive to numCtas/numCtasPerSeqKv in the range of 1-24, but
became much insensitive later."
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h)

**[inferred]** Decoding our kernel `...MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen`
against those verified fields:
- `MultiCtasKv` — multi-CTA split over the KV axis with global-memory reduction (the
  Blackwell equivalent of flash-decoding split-K). Requires the scratch/counter pointers.
- `VarSeq` — variable sequence lengths across the batch (`mSupportsVarSeqLens`).
- `Q8` / `Kv128` — `tileSizeQ = 8`, `tileSizeKv = 128`. The header constrains
  `tileSizeKv ∈ {64, 128}`.
- `Static` — static tile scheduler (as opposed to persistent/CLC dynamic scheduling).
- `SwapsAb` — the MMA A/B operand swap. Blog15 explains why: when TP partitions the Q
  heads, the M dimension collapses and the tensor core is underutilized, so they "swap the
  A and B operands during matrix multiplication to improve hardware utilization."
- `ForGen` — generation (decode) kernel, `FmhaKernelType::Generation`.

**Actionable knobs on this path [verified]** from `envUtils.cpp`:
`TRTLLM_GEN_ENABLE_TILE_SIZE_KV64` (default 0) forces the KV=64 tile;
`FORCE_ATTENTION_KERNEL_DETERMINISTIC` (default 0);
`TRTLLM_DISABLE_CHUNKED_ATTENTION_IN_GEN_PHASE` (default 0);
`TRTLLM_XQA_BLOCKS_PER_SEQUENCE` and `TRTLLM_MMHA_BLOCKS_PER_SEQUENCE` tune split depth on
the legacy paths.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/common/envUtils.cpp)

### Batched-GEMM kernel naming — decoded

**[verified]** `batchedGemm/trtllmGen_bmm_export/BatchedGemmOptions.h` exposes the full
option surface: `mTileM/mTileN/mTileK`, `mEpilogueTileM/N`, `mDtypeA/B/C`, `mDtypeAcc`,
`mDtypeMmaA/mDtypeMmaB`, `mBatchMode ∈ {BatchM, BatchN}`, `mIsStaticBatch`
("whether the batch size is static (i.e. known at kernel launch time)"),
`mIsUniformNumTokensPerBatch`, `mNumStagesA/B/Mma`, `mFusedAct`
("whether to perform a fused gated activation"), `mRouteImpl` ("whether load the input
tokens and do routing"), `mRouteSfsImpl`, `mGridWaitForPrimaryRouting`, `mUseTmaStore`,
`mPrefetchB`, `mUseTmaOobOpt`, `mUseCMultiCast`, `mClusterDimX/Y/Z`,
`mUseFlexibleClusterDims`, `mMmaKind`, `mMmaM/N/K`, `mNumRegsPerThreadLoadA/B`, and
`mNumWarpsLoadA/B`. Validation in `checkAndUpdateBatchedGemmOptions()` enforces e.g.
"N must be divisible by TileN" in BatchM mode and that DeepSeek-FP8 requires dims
divisible by 128.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/BatchedGemmOptions.h)

**[inferred]** Our `bmm_E2m1_E2m1E2m1_Fp32_..._swiGlu_dynB_sm100f` therefore reads as:
batched GEMM, `dtypeA = E2m1` (NVFP4), `dtypeB = E2m1` with `E2m1` MMA operand, FP32
accumulate, `mFusedAct` on with SwiGLU (i.e. the MoE FC1 with gate fused into the
epilogue), `dynB` = dynamic batch (`mIsStaticBatch == false`, batch sizes read from GPU
memory so it stays CUDA-graph-safe), `sm100f` = the SM100 "family" cubin. This is the MoE
FC1 grouped GEMM.

**Actionable:** `mFusedAct` means the gate multiply is already free. The remaining
epilogue fusion opportunity in the MoE block is FC2's alpha and the finalize/reduce — see
`TRTLLM_MOE_DISABLE_FINALIZE_FUSION` (default 0, i.e. finalize fusion is *on* by default)
and the `TRTLLM_MOE_FUSED_FC2_ALPHA` flag discussed under MoE-as-dense-GEMM below.

### XQA — largely historical on Blackwell

**[verified]** The XQA blog in the repo documents a generation-phase kernel for MQA/GQA
that reduces data loading/conversion and uses tensor cores, with numbers *only* on
H100/H200 (Llama-70B FP8, ISL 128 / OSL 2048, batch 1–256: 1,227 → 2,941 tok/s/GPU
single-GPU; 13,232 → 25,300 on 8×H100). It is enabled/forced with `TRTLLM_FORCE_XQA=1`
and is picked by a heuristic against the masked-MHA kernel.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/XQA-kernel.md,
https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/attention.md)

**[inferred]** On Blackwell the generation path in TRT-LLM is trtllm-gen FMHA, not XQA —
blog16 says as much when it lists backends per architecture ("Hopper decode: XQA kernels;
Blackwell: trtllmGenKernels"). XQA is not a lever for us. Recording it because the
assignment asked and because the absence is the finding.

---

## Collectives: what the AllReduce kernels in our profile actually do

### Strategy set and the one-shot threshold

**[verified]** `customAllReduceKernels.h`:

```cpp
enum class AllReduceStrategyType : int8_t
{ NCCL=0, MIN_LATENCY=1, UB=2, AUTO=3, ONESHOT=4, TWOSHOT=5,
  LOWPRECISION=6, MNNVL=7, NCCL_SYMMETRIC=8 };

enum class AllReduceFusionOp : int8_t
{ NONE=0, RESIDUAL_RMS_NORM=1, LAST_PROCESS_FOR_UB=2, RESIDUAL_RMS_PREPOST_NORM=3,
  RESIDUAL_RMS_NORM_QUANT_FP8=4, RESIDUAL_RMS_NORM_QUANT_NVFP4=5,
  RESIDUAL_RMS_NORM_OUT_QUANT_FP8=6, RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4=7,
  MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM=8, RMS_NORM=9 };
```
with `MAX_ALL_REDUCE_BLOCKS = 24`, `MAX_RANKS_PER_NODE = 16`, `DEFAULT_BLOCK_SIZE = 512`,
`kLamportTokenNumThreshold = 16`, `kLamportHiddenSizeThreshold = 256`.

**[verified]** `allReduceFusionKernels.h` sets `static constexpr int kOneShotMaxToken = 128;`
and `kBarrierFlagCount = 256`, with the fusion-pattern enum
`kAllReduce, kARResidualRMSNorm, kARResidualRMSNormFP8Quant, kARResidualRMSNormFP4Quant,
kARResidualRMSNormOutFP8Quant, kARResidualRMSNormOutFP4Quant, kARRMSNorm` and a
`use_oneshot` boolean plus `trigger_completion_at_end` in the params struct.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h)

**[verified]** On the Python side, `_torch/distributed/ops.py` documents AUTO as "chooses
the best available strategy. Will try MNNVL, then choose between NCCL and MIN_LATENCY",
sets `_MNNVL_ONE_SHOT_THRESHOLD_BYTES = 64 * 1024 * 8 * 2` (1 MiB) with the comment "For
one-shot, each rank needs to store num_tokens * group_size tokens; For two-shot, each rank
stores slices of tokens", grows workspaces "in 8 MiB granularity to avoid frequently
scaling the buffer", and caps the MoE AR path at `self.max_token = 128` with the comment
"Pls keep this value in sync with the kOneShotMaxToken in moeAllReduceFusionKernels.h".
Env vars: `TLLM_NCCL_SYMMETRIC_ZERO_COPY` (default "1"), `TRTLLM_FORCE_MNNVL_AR`,
`TLLM_TEST_MNNVL`, `TLLM_DISABLE_ALLREDUCE_AUTOTUNE`.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/tensorrt_llm/_torch/distributed/ops.py)

**MNNVL is gated on aarch64 + multi-node + dtype ∈ {fp16, bf16, fp32} + no context
parallelism.** On our x86 DGX-class 8×B200 it will never be selected. **[verified]**

**[inferred]** What this means for us concretely:
1. At C1 with EAGLE 3-1-4 (≤5 tokens/step) we are two orders of magnitude below
   `kOneShotMaxToken = 128`, so one-shot is correct and `twoshotAllreduceKernel` in our
   profile is probably firing on the *prefill* path (10k tokens ≫ 128), not decode. Worth
   confirming — if two-shot is being selected during decode, that is a bug-shaped win.
2. `TLLM_NCCL_SYMMETRIC_ZERO_COPY` describes a real technique worth copying:
   *"writes GEMM output directly into the window buffer so the allreduce needs no extra
   copy."* Eliminating the pre-AR copy is a pure latency win at small message sizes.
3. `MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM` (fusion op 8) collapses the MoE finalize
   into the same kernel as the AR + residual + norm. That is one fewer kernel boundary per
   MoE layer × 92-ish layers × every decode step.
4. `SYMM_MEM` (PyTorch symmetric memory with MULTIMEM instructions) is present but
   "currently supports only `NONE` fusion operations; fused operations fall back". So NVIDIA
   themselves have not yet fused into the multimem path — a gap we could exploit.

### One-sided all-to-all over NVLink (the MoE collective)

**[verified]** `blog18` describes `NVLinkOneSided`, now the default MoE communication
strategy within a single NVLink domain in TensorRT-LLM and also shipped in FlashInfer.

Mechanism: CUDA Virtual Memory Management creates *symmetric memory* — a shared virtual
address space across GPUs. Each kernel then only reads **or** only writes peer memory,
"eliminating cooperative send/recv". Dispatch is a push (load token once locally, store to
up to `top_k` remote symmetric buffers, atomic counters for slot assignment). Combine is a
pull (read peer outputs, tree-reduce, store locally, reusing dispatch routing metadata).
Two producer-consumer barriers: release membar before flag writes on dispatch; acquire
membar after polling on combine.

The buffer layout is the key trick: **rank-major** `[ep_size, max_tokens_per_rank, ...]`
rather than expert-major. A token routed to two experts on the same rank appears **once**,
which shrinks the pre-allocated buffer by `1/num_experts_per_rank` versus expert-major.
NVIDIA explicitly contrasts this with DeepEP low-latency, which uses expert-major and
therefore duplicates.

Measured on GB200 NVL72, `ep_size=8`, batch 2048, BF16 (peak uni-directional 900 GB/s):
dispatch 311.8 µs / 753.3 GB/s (~84% of peak); combine 322.6 µs / 728.2 GB/s (~81%).
Post-quantization dispatch speedups at bsz=2048: MXFP8 1.81×, NVFP4 3.06× (bytes/token
0.52× and 0.28× of BF16). Combine defaults to BF16 with an optional low-precision path.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md)

**[verified]** Tunables: `TLLM_MOE_A2A_DISPATCH_BLOCK_SIZE` and
`TLLM_MOE_A2A_COMBINE_BLOCK_SIZE`, both default 256, clamped to [32, 1024].

**[inferred]** For us at TP8/EP-within-node this is directly portable and the rank-major
layout is the reusable insight. Note the measured numbers are at batch 2048 — at C1 this
kernel is latency-bound, not bandwidth-bound, and the published bandwidth figures tell us
nothing about our regime. They also note bandwidth degrades with EP size due to
synchronization overhead, and that atomic counter contention may limit very small batches.

### Rank arrival skew: the ADP Balance strategy

Our profile says 47% of collective time is rank arrival skew. **[verified]** `blog10` is
NVIDIA's treatment of exactly this problem for attention-DP + MoE:

> "The execution time for the Attention module in any given iteration is bounded by the
> rank with the highest workload."

Two mechanisms:
- **Context synchronization (`timeout_iters`)** — a rank that becomes free to run context
  while peers are still generating waits up to N iterations before scheduling context work,
  so contexts land on the same iteration across ranks.
- **Batch equilibration (`batching_wait_iters`)** — ranks with fewer accumulated context
  batches wait up to M more iterations to accumulate.

Production config:
```yaml
attention_dp_config:
    enable_balance: true
    batching_wait_iters: 10
    timeout_iters: 50
```
Measured on 8×GB200, DeepSeek-V3, 16,000 requests, 803-token avg input, 3,653-token avg
output, DP8 attention / EP8 MoE:

| | Baseline | Context wait | Full strategy |
|---|---|---|---|
| Actual TPS | 25,664 | 33,499 | 34,140 |
| Balance ratio | 54.11% | 84.33% | **87.70%** |
| Speedup | 1.00× | 1.31× | **1.33×** |

Stated trade-off: TTFT increases because of the strategic waiting. Round-robin by token
count (their baseline) "fails to guarantee per-iteration load balance."
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog10_ADP_Balance_Strategy.md)

**[inferred]** This only helps if the skew is *scheduling* skew across DP ranks. At C1
with TP8 there is no DP, so our 47% must be coming from somewhere else: per-rank kernel
time variance (different expert counts per rank under EP), CPU launch jitter, or NUMA
asymmetry. The three candidate fixes in priority order are (a) NUMA pinning, (b) a single
CUDA graph covering the whole decode step so launch jitter cannot accumulate, and (c)
fusing away sync points. See the CPU-affinity section.

### CPU affinity / NUMA

**[verified]** TRT-LLM ships a dedicated guide. `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` forces
NUMA-aware autoconfiguration; `=0` preserves user settings; unset means auto-configure if
unconstrained. It warns that if a worker process runs on a suboptimal node before
autoconfiguration triggers, "some CPU memory may have been allocated/touched on what will
become a remote NUMA node" — though it claims "minimal performance impact" in practice.
The launch-time alternatives are `bindpcie` (needs
`OMPI_MCA_hwloc_base_binding_policy=none`, `OMPI_MCA_rmaps_base_inherit=1`, or
`SLURM_CPU_BIND=none` / `srun --cpu-bind=none`), `numactl --physcpubind=... --membind=0`,
or an OpenMPI rankfile with explicit `rank N=localhost slot=...`. No measured numbers are
given.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/deployment-guide/configuring-cpu-affinity.md)

**[inferred]** Two NUMA nodes and 8 GPUs means 4 GPUs per node. If our launcher does not
bind rank→NUMA→GPU consistently, half the ranks pay remote-DRAM latency on every host-side
operation in the critical path (graph launch, sampling, spec-decode bookkeeping). At 2.74 ms
TPOT this is not a rounding error. Cheapest experiment on this whole list.

---

## MoE optimisation

### MoE as dense GEMM

**[verified]** `blog24` argues that in the memory-bound regime you should stop doing
grouped GEMM over selected experts and instead run a **dense** FC1/FC2 over *all* routed
experts, masking with per-token alphas. The arithmetic intensity of FC1 is ≈ `4M`
FLOPs/byte independent of expert count or hidden size, and the B200 ridge point is
~1125–1250 FLOPs/byte, so redundant compute is free while `4M < ~1125` ⟹ `M < ~281–312`.
Measured FC1 crossover lands at M ≈ 336.

Measured on B200 (SM100), DeepSeek-V3-shaped: hidden 7168, 256 experts, top-8, SwiGLU,
NVFP4, TP8, per-rank MoE module latency excluding the flanking AllReduces:

| num_tokens | DENSEGEMM µs | TRTLLM-Gen µs | Speedup |
|---|---|---|---|
| 64 | 140.59 | 141.68 | 1.01× |
| **128** | **144.50** | **161.87** | **1.12×** |
| 192 | 166.61 | 168.23 | 1.01× |
| 208 | 168.38 | 169.34 | 1.01× |
| 272 | 241.80 | 176.10 | 0.73× |

Kernels: `cute_dsl_nvfp4_dense_gemm_swiglu_blackwell` (FC1) and
`cute_dsl_nvfp4_dense_gemm_fc2_blackwell` (FC2). Flag `TRTLLM_MOE_FUSED_FC2_ALPHA=1`
pre-multiplies FC2's alpha into FC1's output, disabled by default pending numerical
closure. Weight layouts: FC1 `[num_experts × 2 × intermediate, hidden]` with W1/W3
interleaved; FC2 `[hidden, num_experts × intermediate]` hidden-major for aligned K-blocks
of 256. Stated gaps: shared expert still standalone; router GEMM and top-K still
serialized before the backend.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.md)

**[inferred]** Our model shape (hidden and expert count) is essentially the shape they
benchmarked. Our C1 token count with 3-1-4 speculation is far below the 32-token floor, so
this is a *C16–C64* technique for us, not a C1 one. At C16 × ~5 tokens ≈ 80 tokens we sit
in the window; at C64 × 5 ≈ 320 we are past the crossover. So the interesting deployment
is a *dynamic* switch keyed on the actual per-step token count, which is exactly the kind
of thing a dispatcher in front of the MoE op can do cheaply.

### DeepGEMM MegaMoE (the newest MoE path)

**[verified]** `blog26` (DeepSeek-V4) describes an MXFP8 × MXFP4 backend with **fused
dispatch, expert GEMMs, SwiGLU, and combine via symmetric memory** — i.e. the whole MoE
block including both collectives in one mega-kernel. Measured: 15.3% throughput increase
and 12.7% latency reduction on a 500-request DeepSeek-V4-Flash test on B200 with TP4, EP4,
1K in / 1K out. Input preparation (quantization + buffer writes) is folded into a single
CUDA kernel. NVFP4 MegaMoE is called out as *not yet integrated*: "Integration and
validation" ongoing, excluded from the measured curves — a useful negative.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md)

**[inferred]** This is the direction of travel: the MoE block stops being
`dispatch → grouped GEMM → activation → grouped GEMM → combine → allreduce` and becomes one
kernel over symmetric memory. That is also what TileRT's tile-level runtime is presumably
doing. If we want to close a 500-vs-365 gap, this is the class of change that does it, not
another flag.

### Router GEMM and top-K

**[verified]** Three separate published results on the router path:
- blog03: mixed BF16-in/FP32-out router GEMM = 4% E2E; top-K kernel fusion = 7.4% E2E,
  reducing 18 PyTorch ops to 2 fused kernels and **252 µs → 15 µs** on B200. Implementation
  pointer given: `Deepseekv3RoutingImpl` in `modeling_deepseekv3.py`.
- blog26: custom router GEMM for a 256-expert model at 1–16 token rows: **~8 µs → ~3 µs**.
- blog01: `RoutingKernelTopK.cuh` / `noAuxTcKernels.cu`, +4 tok/s at C1.

**[inferred]** 256 experts, 8 active, rows 1–16 is *exactly* our C1 shape. If our router +
top-K is anywhere near 18 dispatched ops, this is a several-percent C1 win for a
self-contained kernel.

### Wide-EP and EPLB (context, low applicability to us)

**[verified]** blog04/blog08 document the Expert-Parallel Load Balancer. Core abstraction:
an **Expert Slot** decoupled from expert identity, with a routing table mapping ExpertId →
SlotId; hot experts occupy multiple slots. Two equivalent ways to add 32 redundant experts
on DeepSeek-R1: 9 slots/rank at EP32 (288 slots), or 8 slots/rank at EP36 (288 slots).
Offline results on the machine-translation dataset, layer 36:

| Config | Std dev (by rank) | Imbalance ratio (by rank) |
|---|---|---|
| No EPLB, 8 slots/rank, EP32 | 491.6 | 1.564 |
| EPLB, 9 slots/rank, EP32 | 52.0 | **0.109** |
| EPLB, 8 slots/rank, EP36 | 53.9 | 0.115 |

(imbalance ratio = (max − mean)/mean.)

Notable engineering detail **[verified]**: they went through four implementations of the
live weight update before it worked — `cudaMemcpyAsync` deadlocked against CUDA Graph's
mutexes; multithreaded CPU copy with managed memory caused page migration thrash;
`numa_alloc_onnode` + `cudaHostRegister` caused TLB thrashing; the final version uses
**512 MB huge pages** via `mmap` + `mbind` + `madvise(MADV_HUGEPAGE)`. Weight updates are
in-place (not pointer swaps) to stay CUDA-graph-safe, driven by a background C++ thread
with lock/unlock kernels bracketing the MoE computation.

They also explain why they did **not** use DeepEP on GB200: *"DeepEP does not provide CUDA
graph compatibility for all the scenarios."* Their replacement reads communication sizes
from GPU memory (so sizes can vary without CPU sync), uses an LL128-like primitive, and
divides the workspace into channels acting as FIFO write buffers.

Author caveat, quoted: *"there are no magical solutions when doing system design and
optimization, such as large-scale EP"* — evaluate whether MoE GroupGEMM actually dominates
first. Reported end-to-end: up to 6.17× per-GPU output throughput for EP16/EP32 vs EP4/EP8
on GB200 NVL72 with DeepSeek-R1-FP4 across 1k-1k, 4k-1k, 8k-1k.
(blog04, blog08 in the same directory)

**[inferred]** The huge-page/`MADV_HUGEPAGE` and "in-place update to stay graph-safe"
lessons transfer even if wide-EP does not. Any live weight or routing-table mutation we do
under CUDA graphs must be in-place.

### DWDP — a negative-results-rich alternative to EP

**[verified]** `blog19` proposes Distributed Weight Data Parallelism: keep data-parallel
execution, replicate attention weights, partition experts, and **prefetch missing remote
experts via copy-engine `cudaMemcpyAsync`** (no SM consumption) overlapped with the compute
window of MoE layer L and attention layer L+1, with ping-pong buffers. Components:
`DwdpPrefetchBuffer`, `DwdpLayerHandleCollector`, `DwdpManager`.

Measured (DeepSeek-R1, GB200×4, ISL 8K, max_num_tokens 32768): iteration latency DEP4
1319.85 µs → DWDP4 1131.58 µs (14.26%); 161.85 µs of synchronization eliminated; 126.74 µs
of NCCL replaced by a 429 µs P2P copy that is off the critical path. End-to-end on 8K/1K
SemiAnalysis data: 1.10× at 20–30 TPS/user, 1.08× at 40–50, "8.8% at comparable TPS/user
in the 20–100 TPS/user serving range". Roofline crossover vs DEP at ~16K input tokens at
batch 1.

The limitations list is unusually honest and worth reading in full: CuteDSL MoE backend +
NVFP4 only; `TP = 1` within each DWDP group; MPI worker launch only; **incompatible with
the overlap scheduler and with EPLB**; needs a large compute window (16K tokens minimum at
batch 1); CUDA IPC handles prevent cross-node; TTFT rises from 2538 ms to 8314 ms at
20–30 TPS/user; and "communication-computation interference causes frequency throttling,
reducing net gains despite eliminating synchronization overhead."
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.md)

**[inferred]** Not for us — it is a prefill/context-server technique needing ≥16K tokens.
But "copy-engine prefetch overlapped with the previous layer's compute, off the SMs" is a
reusable primitive, and the frequency-throttling observation is a warning about any scheme
that runs heavy copies concurrently with tensor-core work on B200.

---

## Attention, MLA and DSA — the closest match to GLM-5.2

### DeepSeek-V3.2 / DSA on Blackwell (blog15) — read this one twice

This is the single most directly applicable document to our model, because DSV3.2's DSA is
the same mechanism as GLM-5.2's, and NVIDIA published the full optimization ledger.
**[verified]** throughout, from
https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md

**Indexer precision recipe.** Low-rank linear layers BF16; the weight-projection layer
**FP32** (explicitly for accuracy); indexer K cache blockwise FP8; MQA math blockwise FP8;
top-K in FP32. Indexer K cache stored separately and reused across iterations.

**Sparse MLA kernel.** Uses the **`TMALDG.Gather4`** instruction on Blackwell, which
"loads four rows from a source 2D tensor and coalesces them into a single destination
tensor" — this is the hardware primitive that makes gathered sparse KV cheap. Per-tensor
FP8 for both Q and KV, FP8 arithmetic throughout: up to **47.03% throughput increase**.
Plus the SwapsMmaAb operand swap for the TP-partitioned-Q-heads case.

**Top-K.** Non-deterministic radix-select with partitioning, chosen over priority queues
because "the runtime of existing priority-queue approaches grows rapidly as K increases".
4 radix iterations for 32-bit values with 8-bit digits; once the candidate set drops below
2048 it switches to CUB parallel sort, and for very small candidate sets uses a "low-overhead
naive O(N²) comparison-based ranking". PR-9255 adds a bin-distribution preprocessing step
using the **leading 11 bits**, halving the final candidate set. Measured on B200 with input
[64, 9295] selecting top-2048: **7.41× average vs `torch.topk`** (range 6.859×–7.702×).
Determinism ablation on GPQA-Diamond: FP8 model DE 79.8 vs non-DE 79.9; NVFP4 DE 80.3 vs
non-DE 79.4 — they shipped non-deterministic.

**Fusions with measured deltas:**
- PR-8701: fuse blockwise FP8 quantization into the indexer-K-cache write → **32.64–64.20%**
  throughput.
- PR-8960: fuse indexer K tensor stores → **3.5–13.4%** E2E.
- PR-8988: weight-scale consolidation via multi-stream → ~2.53% latency.
- PR-9052: LayerNorm fusion → 1.42% low-latency / 1.90% throughput.
- Overall top-K work: "25%~40% e2e speedup for the low latency and throughput scenarios."

**Multi-stream schedule:** overlap indexer weight scaling with K-cache updates; concurrent
Q and K quantization; overlap the FP32 weight projection against the low-rank Q projection
+ LayerNorm + RoPE.

**Fast path (PR-9524):** when `N ≤ index_topk` (2048), select all past KV and skip MQA and
top-K entirely, with **separate CUDA graphs for short and long sequences**. ~1.03× at 1K/1K.

**MQA/DeepGEMM kernel:** larger MMA tile → up to 10% on Blackwell; paged KV block size
generalized from 64 to any B with `64 % B == 0`; native MTP-3 support (previously MTP>1
needed the PR-9045 workaround of flattening seq-len into the batch dimension).

**Published B200 benchmarks:**

| Scenario | Config | Result |
|---|---|---|
| Min latency | **TP=4**, MTP-3, batch 1, ISL 8K, OSL 1K, 10 requests, `max_num_tokens 8384` | TTFT 425.99 ms, **TPOT 3.2344 ms**, 68.54 tok/s/GPU |
| Max throughput | TP=8, EP=8, MTP-1, batch 256, ISL 8K, OSL 1K, 768 requests, `max_num_tokens 8576`, `enable_attention_dp: true` | TTFT 19,537.8 ms, TPOT 98.52 ms, 1077.28 tok/s/GPU |

CUDA graph config used: `enable_padding: true`,
`batch_sizes: [1..16, 32, 64, 128]`.

NVFP4 accuracy for DSV3.2 vs original: GSM8k 95.26 vs 95.91; MMLU 87.54 vs 87.84;
GPQA-Diamond 84.85 vs 84.34.

**[inferred] Calibration for us.** Their min-latency TPOT of 3.2344 ms ⇒ ~309 tok/s. Our
2.74 ms ⇒ 365 tok/s. We are ahead of TRT-LLM's published DSA min-latency number — but
their run is **TP=4**, ISL 8K, MTP-3, on a 671B model, and ours is TP8 on GLM-5.2. The TP=4
choice is the striking part and is corroborated independently by Dynamo's GLM-5.2 B200
recipe (`DTP4`, 4×B200). **This is the highest-information single fact in this document.**

### DSA index reuse: NVIDIA's answer is *exact*, not periodic

Our stack uses `index_topk_freq=4` — recompute the DSA index every 4 decode steps and reuse
in between. **[verified]** `blog21` describes GVR (Guess-Verify-Refine), which attacks the
same cost but keeps exactness:

Mechanism: consecutive decode steps query highly similar KV neighbourhoods because RoPE/YaRN
make the score landscape "Toeplitz-like", so advancing one step translates rather than
reshuffles it. On real DeepSeek-V3.2 traces they measure **35–50% raw overlap between
consecutive steps in most layers**. The kernel then: (1) **Guess** — compute min/max/mean of
the previous step's indices, use mean as a threshold estimate; (2) **Verify** — secant-style
threshold search to land the candidate count in `[K, C]`; (3) ballot-free candidate
collection reusing phase-2 counts; (4) exact refinement via shared-memory histogram if the
candidate count exceeds K. NVIDIA is explicit: *"GVR is not an approximate pruning
method... exactness is preserved by the verification step and the final shared-memory
refinement."*

Config:
```yaml
sparse_attention_config:
  algorithm: dsa
  index_topk: 2048
  enable_heuristic_topk: true
```
Kernels `gvrTopKJob` / `gvrTopKKernel`. Debug env: `TRTLLM_HEURISTIC_NMIN` (default
**12288**, range [1024, 200000]) and `TRTLLM_SCHEMEX_DEBUG=1`.

Measured: single-operator on a 68,665-token SWE-bench prompt, 17 sampled steps across 9
layers — **1.88× average, 1.57×–2.02× per layer, 2.42× best case**. Ablation on prediction
quality: no preIdx 1.00×, random indices 1.44×, prev-step top-K in high-correlation layers
(L20–60) 1.94×, low-correlation layers (L0–1) 1.65×.

End-to-end TPOT reduction, DeepSeek-V3.2-Exp NVFP4, 8×B200, TEP8, batch 1, OSL 1K:

| Context | MTP=0 | MTP=1 | MTP=3 |
|---|---|---|---|
| 64K | 5.47% | 4.36% | 2.40% |
| 100K | **7.52%** | 6.30% | 3.45% |

Accuracy: no measurable regression on MMLU, GSM8K, GPQA-Diamond, LongBench V1; outputs
match `torch.topk`.

Fallbacks to radix-select: prefill, missing/invalid preIdx, non-contiguous layouts,
unsupported `index_topk`, **sequences below 12,288 tokens**, or batch sizes above a
hardware-aware threshold.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog21_Temporal_Correlation_Meets_Sparse_Attention.md)

**[inferred] Two important consequences for us.**
1. At AA's ~10k input we are *below* their 12,288-token default gate, so GVR would not
   even engage. Our DSA indexer at 5.8% of C1 reflects a 10k context; the payoff curve
   they publish (2.4% at 64K with MTP-3) suggests the ceiling here is small for us.
   Lowering `TRTLLM_HEURISTIC_NMIN` is a one-line experiment but the upside is bounded.
2. The strategic point is different and more valuable: **`index_topk_freq=4` is buying
   speed with an approximation that GVR gets for free with exactness.** If our
   leaderboard runs are ever accuracy-audited, periodic index reuse is a liability GVR
   does not have. Worth reimplementing on principle even at neutral speed.

### DeepSeek-V4's attention, for forward planning

**[verified]** blog26 documents V4's three interleaved attention modes — Sliding Window
(dense over 128 raw tokens, ratio 0), Compressed Sparse (latest 128 raw + 4× compressed
selected by an indexer top-K, ratio 4), Heavily Compressed (latest 128 raw + all 128×
compressed, no indexer, ratio 128) — served by a **dual-pool MLA operator**. Plus an online
compressor (token-level softmax-gated pooling, FP32 state buffers persisted across chunked
prefill steps) and mHC (manifold-constrained hyper-connections) widening the residual
stream 4× with a doubly-stochastic mixing matrix built by 20 Sinkhorn-Knopp iterations.
Two model scales: Flash 284B total / 13B active, 256 routed + 1 shared; Pro 1.6T / 49B
active, 384 routed + 1 shared. First 3 layers use **hash routing** (token-ID → expert
table); later layers use learned sqrt-softplus affinity with score-correction bias.

Kernel/perf deltas on B200/B300: sparse MLA 1.31× faster CSA prefill / 1.21× faster HCA
generation with FP8 KV; standalone inverse-RoPE + FP8 quant 1.6–2.3× faster than Triton;
FMHA epilogue fusion up to 1.62× prefill / 1.34× generation for Pro with FP8 KV; a
device-aware top-K dispatcher (insertion selection / single-CTA radix / multi-CTA
split-and-merge) taking K=512 at batch 148 from **384 µs → 112 µs**; GVR 1.40–2.17× vs
radix baseline for Flash/Pro shapes on B300 and 6.4% higher end-to-end throughput on a
65K-token Flash test (8×B300 SXM6, batch 4, MTP-3).

KV cache is managed by **KVCacheManagerV2** with a sliding-window pool and persistent
compressed pools sharing indices at the compression ratio; page allocation in raw-token
coordinates while compressed buffers store `tokens_per_block / r` entries; knobs
`kv_cache_config.pool_ratio`, `kv_cache_config.avg_seq_len`, `enable_kv_pool_rebalance`
(beta, off by default), `cache_salt`, `swiglu_limit`.

Milestones on GB300 NVL72, DeepSeek-V4-Pro, MXFP4, disaggregated, 8K in / 1K out:
984 → 1,167 (+18.6%) → 1,505 (+29.0%) → 1,618 tok/s/GPU (+64.5% cumulative), with
TEP4/TEP8 for latency-oriented points and DEP8/DEP16/DEP32 for throughput, MTP-3 at most
points and MTP-1 at the highest concurrency.

Explicit negatives: host offloading and incremental transfer produced no meaningful
end-to-end improvement; NVFP4 MegaMoE not yet integrated; DSpark speculative-decoding path
tuning "remains in progress".

### Other sparse-attention machinery

**[verified]** `blog17` gives the abstraction: `sparse_kv_indices` (what to keep in cache
after context) and `sparse_attn_indices` (what to attend to), an `AttentionBackend` holding
the prediction logic and a unified `AttentionOp`, a `gatherKvPageOffsetsKernel` for
page-level and direct token-level kernel support, and `updateSparseKvCacheAfterFmha` to
rewrite the KV cache in place (requires sorted indices). Three algorithms ship: RocketKV
(training-free eviction + dynamic top-K, default `budget=2048, window_size=32,
kt_page_size=4, kt_cache_dtype=fp8, topk=64`; on B200 Llama3.1-8B at 32k/4k, 2.26× tok/s/GPU
and 3.40× tok/s/user with LongBench accuracy 48.15 vs 48.70), DSA, and Skip-Softmax.

**[verified]** `blog16` Skip-Softmax / BLASST skips softmax and BMM2 for a KV block when
`exp(local_max − global_max) < λ`, with `λ = threshold_scale_factor / context_length`.
Calibrated factors for Qwen3-30B: 50% sparsity → prefill 587.18 / decode 16.52; 80% →
7799.91 / 317.99. **BMM1 is never skipped**, capping kernel speedup at 1.8×. Accuracy on
LongBench V1: negligible to 40% sparsity, −0.92% at 50%, −2.23% at 70%, −7.95% at 90%.
End-to-end TTFT on H200: 10k avg 9420 → 8130 ms (−13.7%) at 90% sparsity; 130k avg
16487 → 12508 ms (−24.1%). Config:
```yaml
sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: {prefill: 1000.0, decode: 500.0}
```

**[inferred]** Skip-softmax is orthogonal to DSA and applies to the dense part of our
attention. At AA's 10k input, a −13.7% TTFT at 90% sparsity is interesting for our 189 ms
TTFT — but the accuracy cost at 90% is −7.95%, which is not acceptable, and at the safe
50% setting the TTFT gain will be a fraction of that. Probably not worth it for us.

### Architectural guidance for attention (co-design post)

**[verified]** NVIDIA's July 2026 post gives four guidelines: (1) push the GQA group size
G high — prefill runtime is flat in G (<1% from G=1 to G=64) while decode arithmetic
intensity ≈ 2G so decode time halves per doubling of G; (2) use head dim 128 or 256 to
match 128-wide tiles and 128-byte transfers (Hsz=64 still pays for a full 128-wide tile;
Hsz≥512 approaches TMEM capacity); (3) reduce effective KV state; (4) **keep TP ≤ number of
KV heads**, otherwise use attention-DP or KV-parallelism. They quantify why long context
matters: on DeepSeek-R1, attention is 18% of prefill time at 4K context and **85% at 128K**.
(https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference/)

**[inferred]** Guideline 4 is another argument on the TP4-vs-TP8 question: MLA at TP8 has
128/8 = 16 q-heads per rank, which blog01 explicitly calls "like an MQA with
num_q_heads = 16". That is where SwapsMmaAb comes from — it exists because TP8 leaves the M
dimension too small. TP4 would double it.

---

## Speculative decoding

**[verified]** TRT-LLM supports six methods with these config surfaces:

| Method | Required fields | Notable options | Engine |
|---|---|---|---|
| Draft/Target | `max_draft_len`, `speculative_model` | `disable_overlap_scheduler=True` | two-model |
| EAGLE-3 | `max_draft_len`, `speculative_model` | `use_dynamic_tree`, `dynamic_tree_max_topK`, `max_total_draft_tokens`, `eagle3_one_model` | two-model or one-model |
| NGram | `max_draft_len`, `max_matching_ngram_size` | `is_public_pool`, `is_keep_all`, `is_use_oldest` | single-model |
| MTP | `max_draft_len`, `num_nextn_predict_layers` | `use_relaxed_acceptance_for_thinking`, `relaxed_topk`, `relaxed_delta` | single-model |
| PARD | `max_draft_len`, `speculative_model` | `mask_token_id` | two-model |
| DFlash | `max_draft_len`, `speculative_model` | `mask_token_id`, `target_layer_ids` | two-model |

Constraint: `max_draft_len <= max_total_draft_tokens <= dynamic_tree_max_topK * max_draft_len`.
Dynamic tree is EAGLE-3 only and is **not supported for sliding-window attention or MLA** —
which rules it out for GLM-5.2. And: *"Speed ups are only observable at low batch sizes."*

**Suffix automaton** can be layered on EAGLE-3, MTP and PARD via `use_sa_spec=True` with
`sa_spec_threshold` (default 4 = minimum suffix match length).
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/speculative-decoding.md)

**MTP internals [verified]** (blog02): *MTP Vanilla* runs K distinct modules each with its
own KV cache and needs the past K input IDs and hidden states; *MTP Eagle* reuses one
module K times with a shared KV cache and needs only the last token ID and hidden state.
Verification accepts the longest matching prefix and evicts the KV of rejected drafts.
Chain-based only in the PyTorch backend — no tree. On DeepSeek-R1-FP4 671B on B200:
MTP nextn=3 strict 2.16×, +relaxed acceptance 2.33×; CUDA graph 7.22×; overlap scheduler
1.03×.

**Relaxed acceptance [verified]** (blog01/blog02): during the thinking phase, sample top-N
from the logits, drop candidates with probability below `top1_prob − delta`, and accept the
draft if it matches *any* survivor.

| Config | Acceptance | Speedup |
|---|---|---|
| MTP3 top1, delta 0.0 | 2.82 | 1.00× |
| MTP3 top10, delta 0.5 | 3.06 | 1.08× |
| **MTP3 top10, delta 0.6** | **3.10** | **1.09×** |
| MTP3 top15, delta 0.5 | 3.07 | 1.08× |

Accuracy cost is real and published: MMLU-Pro 84.0 → 81.2 (n=12,032); GPQA Diamond
71.0 → 69.2; but MATH-500 96.0 → 96.2, AIME24 68 → 74, SciCode 36 → 39, LiveCodeBench
62 → 66 (small n on the last four). Requires `add_generation_prompt = True`.

MTP depth sweep on 8×B200 **[verified]**:

| Config | Acceptance | tok/s | Speedup |
|---|---|---|---|
| no MTP | 1.00 | 111 | 1.00× |
| MTP1 | 1.92 | 198 | 1.78× |
| MTP2 | 2.58 | 250 | 2.25× |
| **MTP3** | **2.82** | **253** | **2.28×** |
| MTP4 | 2.99 | 245 | 2.21× |
| MTP5 | 3.01 | 239 | 2.15× |

**[inferred]** Note the shape: acceptance keeps rising past 3 but throughput peaks at 3.
Our 3-1-4 configuration sits at the same place on the curve. The lesson is that the
optimum is set by draft *cost*, not draft *quality* — which is exactly the premise DFlash
attacks.

### DFlash — the newest and potentially biggest lever

**[verified]** NVIDIA's June 2026 post introduces DFlash, "an open source lightweight block
diffusion model designed for speculative decoding". Mechanism: replace autoregressive
drafting with **block-diffusion drafting** — "predicts a block of masked future tokens in a
single forward pass" — combined with target hidden-state conditioning and **KV injection**
(target context features injected into the draft model's key/value projections across
layers). The explicit argument against EAGLE-3: it "still generate[s] tokens sequentially,
so drafting cost increases as the number of speculative tokens increases."

Reported numbers (all vendor, none independently reproduced):
- gpt-oss-120b on 8× DGX B300 with TensorRT-LLM: up to **15× throughput at 500–600 tok/s
  interactivity** vs autoregressive, and **1.5× over EAGLE-3 at the same interactivity**;
  per-dataset 1.5–2.6×.
- Llama 3.1 8B: nearly doubles interactivity vs EAGLE-3 at the same concurrency; 2.8×
  average across Speed-Bench.
- Gemma 4 31B on vLLM, 1× DGX B300: up to 5.8× (Math500), 5.3% GSM8K→5.3×, 5.6× HumanEval.
- Qwen3-8B on **SGLang, 1× B200**: 5.1× Math500, 4.2× HumanEval.

Availability: 20 checkpoints on Hugging Face for Blackwell and Hopper; integrated into
**SGLang**, vLLM (via Speculators) and TensorRT-LLM; "no code changes required — swaps with
EAGLE-3 via configuration updates". Families covered: Qwen, Kimi K2.6, Llama, Gemma,
gpt-oss.
(https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)

**[inferred]** Config asymmetry warning: the "15×" is throughput at a *fixed interactivity
target*, i.e. it is a Pareto-frontier comparison at 500–600 tok/s/user, not a single-stream
latency number. The honest headline is the 1.5× over EAGLE-3. For us: SGLang integration
already exists, so the engine work is small — but **no GLM checkpoint is published**, so the
cost is training a DFlash head for GLM-5.2. Given that our EAGLE 3-1-4 is already on the
flat part of the depth curve, this is the most plausible path to a step change in C1.

---

## Runtime: CUDA graphs, PDL, torch.compile, autotuner

### Programmatic Dependent Launch

**[verified]** `TRTLLM_ENABLE_PDL` is now **default 1** on SM90+ (`envUtils.cpp`). blog01
measures it at +5 tok/s (173 from 168) standalone, but it is also the enabler for the
Fuse_A weight-prefetch GEMM (+5 there too). blog26 lists where it is applied in the newest
code: DeepGEMM, packed 1×128 FP8 quantization, MLA RoPE, sparse-index conversion, sparse
FMHA, and mHC — "consumer grids launch before producer retires; explicit dependency
points." The trtllm-gen FMHA runner queries `getEnvEnablePDL()` to enable grid
serialization. CUDA 13's cuBLAS also supports PDL for sm_90+, "decreas[ing] kernel launch
latencies when executed alongside other PDL kernels."

**[inferred]** PDL is the mechanism that makes a long chain of small kernels — exactly our
C1 decode step — behave less like a chain of round trips. If our fork does not set it
everywhere, that is free.

### CUDA graphs

**[verified]** blog03 measures CUDA graph at **+22% E2E** and the overlap scheduler at +4%
on the throughput path. blog02 measures CUDA graph at **7.22×** on the min-latency MTP path.
Padding is on by default (`cuda_graph_config.enable_padding`), opt out with
`enable_padding: False`.

**[verified]** blog20 is a dedicated study of bucket granularity. Default "x2" set:
`1, 2, 4, 8, 16, 24, ..., 128, 256, 512, 1024, 2048` (23 graphs), allowing up to 50%
padding waste between 128 and 2048. `+64` adds 64-step increments (49 graphs), reducing
max per-step padding from 1024 to 63 tokens. `+8` gives 259 graphs and 7-token max padding.
Measured on GB200 with TRT-LLM v1.3.0rc8, four NVFP4 models, ISL 500 / OSL 2000, aggregated
(4×GB200 TP) and disaggregated (4–8 prefill GPUs, 4 decode, EP):
- `+64`: up to **1.3×** aggregated, up to **1.5×** disaggregated (DeepSeek-R1 at
  concurrency 500–600). Cost: ~260 MB (26 extra graphs × ~10 MB) and 1.17× server startup
  (graph capture 2.4× longer).
- `+8`: DeepSeek-V3.2's 210 extra graphs cost 4.86 GB and **17.2% of KV cache**, producing
  a ~5% *regression* at concurrency 500. DeepSeek-R1's cost 1.99 GB / 8.6% KV, giving up
  to 1.15×.

**[inferred]** Clean result with a clean failure mode: graph metadata competes with KV
cache. `+64` is the sweet spot; `+8` is over-fitting. Directly applicable to our C64 target.

### torch.compile and piecewise CUDA graph

**[verified]** TRT-LLM implements a custom torch.compile backend that runs
"cudagraph-unsupported components (primarily attention) in eager mode while capturing and
replaying the supported parts with CUDA Graph". Attention, MoE (because chunked MoE has a
data-dependent iteration count), output projections and MTP are wrapped as opaque custom
ops. Config:

```yaml
prefill_cuda_graph_backend: piecewise     # or "breakable" (experimental)
prefill_capture_num_tokens: [ ... ]
torch_compile_config:
  enable_userbuffers: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1024
```

Custom passes registered at the ATen IR level: AllReduce fusion (AR + residual + RMSNorm +
optional FP8/FP4 quant), AllReduce with user buffers (removes an extra copy), re-inplace
optimization, and **auto multi-stream scheduling** that builds a dependency DAG and
schedules ops across streams via custom scheduling operators.

Limitations: cannot work with multi-ModelEngine config; incompatible with two-model
speculative decoding (MTP Eagle, Eagle3); unsupported with multimodal. Guidance: "Larger
token counts reduce exposure to host-side overhead" but cost memory; and even with
piecewise graphs "you may still observe bubbles in the context (prefill) phase, primarily
due to the attention operator's substantial host-side overhead."
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/torch_compile_and_piecewise_cuda_graph.md)

**[inferred]** The "auto multi-stream scheduling from a dependency DAG" pass is the
generalization of all the hand-written multi-stream overlaps in blog01/blog15. That is the
right way to build it once rather than per-model. Also note the incompatibility with
two-model Eagle3 — another argument for `eagle3_one_model: true`.

### AutoTuner

**[verified]** `_torch/autotuner.py` profiles candidate "tactics" at warmup and caches
winners to a JSON on disk, keyed by shape buckets (`gen_tuning_buckets` /
`map_to_tuning_buckets`). Defaults: `warmup=2`, `repeat=10`,
`stream_delay_micro_secs=1000`. Distributed strategies: `BROADCAST` (rank 0 only),
`INDEPENDENT` (per-rank cache under `rank_{N}`), `MERGE`, `PARALLEL`. Options include
`use_cold_l2_cache` (cycles through buffer copies to simulate a cold L2),
`use_cuda_graph` for stable timing, `tune_max_num_tokens`, `inputs_pre_hook`, and
`exclude_from_cache` for JIT-driven ops. Every runner must implement tactic `-1` as a
universal fallback; a cache miss at inference falls back with a warning. Short-circuit: if
warmup exceeds 1 ms, it drops to `profile_fewer_repeat=2`. Env: `TLLM_PROFILING_TIMER`
(`globaltimer` | `cuda_event`), `TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE`,
`TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO`. Cache writes use POSIX byte-range locks plus
atomic replace.

**[inferred]** Two things to steal even without the tuner: (a) `use_cold_l2_cache` — if we
benchmark GEMM tactics with a hot L2 we will pick the wrong tactic for the real decode
loop; (b) the min-latency Llama-4 recipe sets `enable_autotuner: false`, so NVIDIA's own
best C1 config distrusts its tactic search at tiny M. There is a known
FlashInfer issue that the `trtllm_fp4_block_scale_moe` autotuner "can pick
slower-than-default tactics and is not EP/DP-aware" on SM100 — worth checking whether our
NVFP4 MoE is being mis-tuned.
(https://github.com/flashinfer-ai/flashinfer/issues/3537 — issue title read via search
result listing only; **[unverified]** contents.)

### Full environment-variable surface

**[verified]** from `envUtils.cpp`, the ones that matter to us:

| Var | Default | Effect |
|---|---|---|
| `TRTLLM_ENABLE_PDL` | **1** on SM90+ | programmatic dependent launch |
| `TRTLLM_GEN_ENABLE_TILE_SIZE_KV64` | 0 | trtllm-gen attention KV tile 64 |
| `TRTLLM_FORCE_XQA` | 0 | force XQA generation kernel |
| `TRTLLM_XQA_BLOCKS_PER_SEQUENCE` | unset | split-KV depth for XQA |
| `TRTLLM_MMHA_BLOCKS_PER_SEQUENCE` / `TRTLLM_MMHA_KERNEL_BLOCK_SIZE` | 0 | MMHA tuning |
| `TRTLLM_DISABLE_CHUNKED_ATTENTION_IN_GEN_PHASE` | 0 | |
| `TRTLLM_ENABLE_CASCADE_MMHA` | 0 | |
| `TRTLLM_ENABLE_TRTLLMGEN_MOE_ROUTING_RENORM_PDL` | 0 | PDL for MoE routing renorm |
| `TRTLLM_MOE_DISABLE_FINALIZE_FUSION` | 0 | finalize fusion is on by default |
| `TLLM_MOE_A2A_DISPATCH_BLOCK_SIZE` / `..._COMBINE_BLOCK_SIZE` | 256 (clamp 32–1024) | one-sided A2A block size |
| `FORCE_ALLREDUCE_KERNEL_WORKSPACE_SIZE` | 1e9 | AR workspace |
| `FORCE_DETERMINISTIC`, `FORCE_MOE_KERNEL_DETERMINISTIC`, `FORCE_ALL_REDUCE_DETERMINISTIC`, `FORCE_ATTENTION_KERNEL_DETERMINISTIC` | 0 | determinism escape hatches |
| `TLLM_NUMA_AWARE_WORKER_AFFINITY` | auto | NUMA binding |
| `TRTLLM_PRINT_SKIP_SOFTMAX_STAT` | 0 | skip-softmax sparsity telemetry |
| `TLLM_NVTX_DEBUG`, `TLLM_PROFILE_RECORD_GC`, `TLLM_PROFILE_START_STOP` | | profiling |
| `TRTLLM_USE_UCX_KVCACHE` / `_MPI_` / `_NIXL_` / `_MOONCAKE_` | 0 | disagg transport |
| `TRTLLM_KVCACHE_TRANSFER_BUFFER_SIZE` | 512 MB | |
| `TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM` | 1 | |
| `TRTLLM_KVCACHE_RECV_BUFFER_COUNT` | 2 | |
| `TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY` | 0 | |
| `TRTLLM_EPLB_FORCE_GDRCOPY` | 0 | |

---

## Disaggregated serving in TensorRT-LLM

**[verified]** blog05 documents three deployment routes (`trtllm-serve` with a
disaggregated orchestrator passing `ctx_params`; NVIDIA Dynamo; Triton with a BLS
orchestrator) and three KV transports (MPI, UCX, NIXL, all over RDMA/NVLink), with the
recommendation: *"we recommend using UCX and NIXL backends, as we are adding a dynamic
scaling mechanism on top of them."* The KV-exchange module is decoupled from both the KV
cache manager and the comms library, so context and generation can run different
parallelism (e.g. TP in context, PP in generation) with layout conversion handled during
transfer. Transfers overlap with computation *across* requests: "While one request is
sending or receiving its KV cache blocks, other requests can proceed with computation."

Their **rate-matching methodology** is the reusable part: measure context throughput
(req/s/GPU) under a TTFT constraint across TP/EP/DP/PP; measure generation throughput and
per-user latency across mappings, concurrency and speculative settings; solve for the
context:generation worker ratio; then compute
`per-GPU throughput = TotalOutputTokens/s ÷ [(NumCtxGPUs × GenReqRate)/CtxReqRate + NumGenGPUs]`.

Reported speedups vs aggregated: DeepSeek-R1 ISL 4400 / OSL 1200, 1.4–1.8× without MTP and
1.6–2.5× with MTP; ISL 8192 / OSL 256, up to 1.73× with 4-GPU generation and up to 2× with
8-GPU; Qwen3 ISL 8192 / OSL 1024, 1.7–6.11×. Example topologies: `CTX=1xTEP-4 | GEN=2xDEP-8`,
`ctx1dep4-gen2dep4-mtp3`.
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog05_Disaggregated_Serving_in_TensorRT-LLM.md)

**[inferred]** Disaggregation is a cost-per-user technique, not a single-stream one — it
buys you the freedom to run the decode pool at a different (usually smaller) TP. For our
C64 objective it is the structurally right answer; for C1 it adds a KV transfer to TTFT.
Dynamo's own GLM-5.2 numbers below show exactly this trade: disagg roughly doubles
tok/s/GPU but takes TTFT P50 from 356 ms to 1938 ms.

---

## NVIDIA Dynamo

**[verified]** Dynamo positions itself as "the orchestration layer above inference engines",
Rust core + Python extensibility, integrating with **SGLang, TensorRT-LLM and vLLM** (all
three supporting disaggregated serving, KV-aware routing and SLA-based planning).
Components: KV-aware router; KV Block Manager (KVBM) offloading GPU → CPU → SSD → remote
(S3/Azure); ModelExpress/NIXL streaming weights GPU-to-GPU; Planner (SLA-driven
autoscaler); Grove (topology-aware gang scheduling on Kubernetes).
Headline claims **[reported]**: 7× higher throughput per GPU (DeepSeek-R1 on GB200 NVL72),
7× faster model startup (ModelExpress, DeepSeek-V3 on H200), 2× faster TTFT from KV-aware
routing (Qwen3-Coder 480B), 80% fewer SLA breaches at 5% lower TCO, 750× throughput
increase (DeepSeek-R1 on GB300 NVL72). None of these carry a full config in the README and
none are independently reproduced; treat as marketing until a config appears.
(https://raw.githubusercontent.com/ai-dynamo/dynamo/main/README.md)

### KV-aware routing: the actual cost function

**[verified]** This is the part with real content:

```
cost = prefill_load_scale * adjusted_prefill_blocks
     + potential_decode_blocks
     + active_request_blocks

adjusted_prefill_blocks = max(0, raw_prefill_blocks - overlap_credit_blocks)
potential_decode_blocks = active_decode_blocks + incoming_active_blocks
active_request_blocks   = decode_active_request_weight * active_requests
```
The router picks the lowest-cost eligible worker. Overlap credits combine device, host,
disk and shared-cache credits, clamped non-negative, with `overlap_score_credit_decay` to
discount device-local credit when a cache-rich worker is already prefill-saturated. A
conditional-disaggregation variant uses
`cost = max(0, potential_decode_blocks − overlap_credit_blocks) + active_request_blocks`.
Knobs: `--router-prefill-load-model aic` (query AIConfigurator for expected prefill
duration, with time-based decay of active load), `--router-track-output-blocks`,
`--router-decode-active-request-weight` (default 0), `router_temperature` (softmax sampling
over normalized cost logits), `--router-queue-threshold`, `overlap_score_weight` (example
value 2.0 for contexts > 8192 tokens).
(https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md)

**[verified]** The agentic-inference post adds: a **Flash Indexer** at "170M ops/s" for the
routing index; an `nvext.agent_hints` request extension carrying `priority`, `osl`
(harness's own output-length estimate), and `speculative_prefill`; `cache_control.type:
ephemeral` with `ttl` (e.g. "1h") for subagent/reasoning KV that should not be retained;
and measured agent cache behaviour — Claude Code 85–97% cache hit rate per call with an
11.7× read/write ratio, 97.2% aggregate across a 4-agent team, teammate agents at 79.4%
vs lead 91.3% because of cold-start writes on first call. Thompson-sampling router in NeMo
Agent Toolkit: 4× reduction in P50 TTFT, 1.5× increase in P50 tok/s, 63% P50 TTFT reduction
under memory pressure with priority tagging. Models tested include **GLM-5** and MiniMax 2.5.
(https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/)

### Dynamo's published GLM-5.2 recipe — the most directly relevant artifact in this document

**[verified]** `docs/fern/pages/recipes/model-recipes/glm-5-2.mdx`. Four targets, **engine
is SGLang** under Dynamo:

| Target | GPUs | Checkpoint | Precision | Parallelism | Max ctx |
|---|---|---|---|---|---|
| B200 aggregated | 4× B200, 1 worker | `nvidia/GLM-5.2-NVFP4` | NVFP4 + FP8 KV | **DTP4** | 500K w/ HiCache CPU offload |
| B200 disaggregated | 4× B200 prefill + 8× B200 decode | `nvidia/GLM-5.2-NVFP4` | NVFP4 + FP8 KV | **DEP4 prefill / DTP8 decode**, NIXL/UCX over IB | 500K |
| H200 aggregated | 8× H200 | `zai-org/GLM-5.2-FP8` | FP8 + FP8 KV | TP8/EP8 | 250K |
| H200 disaggregated | 8+8 × H200 | `zai-org/GLM-5.2-FP8` | FP8 + FP8 KV | TP8/EP8 prefill / TP8/DP8/EP1 decode | 250K |

Features: KV-aware routing at the frontend; **EAGLE-style MTP speculative decoding with
draft length 3 and a measured acceptance length of 2.69**.

Published agentic-trace results:

| Target | Concurrency | tok/s/GPU | User tok/s P50 | TTFT P50 (ms) |
|---|---|---|---|---|
| B200 agg | 64 | 176.42 | 57.49 | 355.6 |
| B200 disagg | 128 | 320.91 | 65.11 | 1938.1 |
| H200 agg | 32 | 54.55 | 52.37 | 1790 |
| H200 disagg | 24 | 68.86 | 53.88 | 1874 |

Limitations stated: B200 tops out at 500K context (1M unsupported); structured decoding
requires reasoning disabled; disaggregated targets do not support `n>1`.

**[inferred]** Three things jump out. (1) **NVIDIA publishes an `nvidia/GLM-5.2-NVFP4`
checkpoint** — if we are quantizing our own, comparing against theirs is a free accuracy
and layout check. (2) Their B200 aggregated shape is **4 GPUs, DTP4** — a second
independent vote for TP4 on this model class. (3) Their numbers are an *agentic trace* at
concurrency 64/128, so 57–65 tok/s/user is not comparable to our 365 tok/s at C1; nothing
here contradicts our position, and NVIDIA has published no single-stream GLM-5.2 figure I
could find.

**[verified]** The related GLM-5 NVFP4 recipe (20× GB200 across 5 nodes, SGLang runtime
image `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13`) is worth reading for its
SGLang env vars, which are directly usable in our fork:
`SGLANG_ENABLE_SPEC_V2=1` (EAGLEWorkerV2 with overlap scheduler),
`SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=1` (quantize the MTP layer to FP8 at load),
`FLASHINFER_WORKSPACE_BASE=/model-store` (persist JIT artifacts),
`--kv-cache-dtype fp8_e4m3`, `UCX_TLS=cuda_copy,cuda_ipc,tcp`,
`SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` on EFA. Prefill TP4 on one node, decode
TP16/DP16/EP16 across four; EAGLE MTP with a reported **85–95% accept rate**. Published:
ISL 1K / OSL 8K / concurrency 512 → 16,824 tok/s output, TTFT P50 15,423 ms, ITL 23.31 ms
(UCX) vs 19,131 tok/s, TTFT P50 621 ms, ITL 24.5 ms (AWS EFA).
(https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/recipes/model-recipes/glm-5-nvfp4.mdx)

**[verified]** A GLM-5-FP8 Pareto sweep run under Dynamo's DynoSim on B200 with a 72-GPU
budget, `zai-org/GLM-5-FP8`, synthetic 1024-in/1024-out, disaggregated static: 204
Pareto-optimal points out of 640 evaluations. Max throughput 1,788.4 tok/s/GPU at 23.56
tok/s/user using `2xDEP8 + 6xDTP8` across 64 GPUs at 11,940 concurrency. Lowest-concurrency
point: **87.79 tok/s/user with 1 request in flight using `1xTEP8 + 2xTP8` on 24 GPUs.**
The sweep reached 94.6% of measured SemiAnalysis results.
(https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/glm-5-fp8-pareto-sweep.md)

**[inferred]** That 87.79 tok/s/user single-request figure is a *simulator* output for
GLM-5 FP8 in a disaggregated topology, not a measured min-latency run, and it is far below
what a latency-optimized aggregated deployment achieves. It is not a competitive datapoint;
it is evidence that Dynamo's sweeps are not optimizing for our C1 objective at all.

### Roadmap models

**[verified]** Dynamo also publishes recipes for our next targets:
- **Kimi-K3** (`moonshotai/Kimi-K3`): vLLM engine, **FlashInfer MLA** attention, FlashInfer
  TRT-LLM MoE backend with **MXFP4-packed routed experts** + BF16 dense, FP8 KV, 1M context.
  GB200: 16 GPUs TP16 over NVLink (aggregated) or 16+16 across 8 nodes. GB300: TP8 per
  replica with KV-aware cross-replica routing; NIXL/MNNVL transfer. TP over MNNVL uses
  Kubernetes ComputeDomains. **No performance numbers published.**
- **Qwen3.8-2.4T-A95B** (`Qwen/Qwen3.8-2.4T-A95B-FP8`): FP8 weights + FP8 KV, hybrid
  gated-delta-net + MoE with **512 experts**, 262,144 context. SGLang on both aggregated and
  disaggregated, vLLM aggregated only. TP16 over MNNVL. Notable flags:
  `VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl` with `VLLM_USE_NCCL_SYMM_MEM=0`, and
  `--no-async-scheduling`. NIXL over `cuda_ipc + MNNVL`.

**[verified]** NVIDIA's Qwen3.8 launch post reports "over 4K tokens per second per GPU" in
FP8 and "over 350 tokens per second per user" on GB300 NVL72 (72 GPUs), with NVFP4 named as
future work. No config flags in the post.
(https://developer.nvidia.com/blog/serve-qwen3-8-2-4t-a95b-a-2-4t-parameter-model-with-configurable-reasoning-on-nvidia-gb300-nvl72/)

---

## NVFP4, quantization recipes, and the Model Optimizer

**[verified]** NVFP4 is E2M1 (1 sign, 2 exponent, 1 mantissa), range roughly −6..+6, with
**one shared FP8 (E4M3) scale per 16-value block** plus one FP32 per-tensor scale.
Versus MXFP4: block 16 vs 32, and E4M3 fractional scales vs E8M0 power-of-two scales.
Effective storage ≈ 4.5 bits/value. Memory: 3.5× vs FP16, 1.8× vs FP8. Accuracy on
DeepSeek-R1-0528 FP8 → NVFP4: MMLU-PRO 85 → 84, GPQA Diamond 81 → 80, Math-500 98 → 98,
AIME 2024 89 → 91.
(https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

**[verified]** Published mixed-precision recipes, which is the more useful part:

- *DeepSeek-R1 min latency (blog01)*: attention BF16; dense FFN NVFP4; MoE FFN NVFP4; MTP
  layers BF16; RouterGEMM BF16 in / FP32 out.
- *DeepSeek-V3.2 (blog15)*: attention QKV projection BF16; output projection NVFP4; sparse
  MLA KV cache per-tensor FP8; sparse MLA math per-tensor FP8; MoE NVFP4. Indexer: low-rank
  linears BF16, **weight projection FP32**, K cache blockwise FP8, MQA math blockwise FP8,
  top-K FP32.
- *DeepSeek-V4 (blog26)*: three recipes — FP8 base (blockwise FP8 expert/attention GEMMs,
  BF16 projections, blockwise FP8 indexer-K, **FP32 compressor state**); MXFP4 instruct
  (MXFP8 activations × MXFP4 weights for routed experts, MXFP8 shared expert, BF16
  projections); NVFP4 requantized (NVFP4 routed experts in the main transformer, MTP subtree
  keeps the instruct recipe).

**[verified]** blog03 measures the FP8 KV cache path: static per-tensor FP8 with a default
1.0 scale (or calibrated), +6% throughput at fixed concurrency and 50% cache compression,
with GSM8K unchanged at 0.9613 (FP8 ckpt) and 0.9606 (FP4 ckpt) whether MLA/cache is BF16
or FP8. And FP4 AllGather instead of BF16 AllGather: ~3× on the kernel, 4% E2E, with no
accuracy impact because the downstream op wants FP4 anyway.

**[inferred]** The consistent pattern across three models: **keep the small, numerically
sensitive, latency-irrelevant modules in high precision** (router GEMM output FP32, indexer
weight projection FP32, compressor state FP32, MTP/draft layers BF16) and spend the FP4
budget only on the big GEMMs. If our NVFP4 build quantizes the router or the EAGLE head, we
are paying acceptance rate for nothing.

**[unverified]** I could not retrieve concrete `modelopt` PTQ commands or config files at
any URL I tried; the NVFP4 blog names TensorRT Model Optimizer and LLM Compressor but
contains no commands. Do not cite a ModelOpt recipe from this document.

---

## MLPerf Inference on Blackwell

**[verified]** NVIDIA's inference results page for MLPerf Inference v6.0 lists, in Offline:
DeepSeek-R1 2,494,310 tok/s on 288× GB300 and 486,141 tok/s on 72× GB200; GPT-OSS-120B
1,046,150 tok/s on 72× GB300 and 879,542 on 72× GB200; Llama3.1-405B 19,512 (72× GB300) and
15,462 (72× GB200); Llama2-70B 1,126,850 (72× GB300) and 888,054 (72× GB200); and 8× B300
achieving 70,326 tok/s on DeepSeek-R1. Benchmarks in the round: DeepSeek-R1, GPT-OSS-120B,
Qwen3-VL-235B, Llama3.1-405B, Llama2-70B, Llama3.1-8B, Wan2.2, DLRMv3, Whisper.
(https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference)

**[verified]** The submission tree discloses systems `B200-SXM-180GBx8` (DGX B200),
`B300-SXM-288GBx8` (DGX B300), and GB200/GB300 NVL72 variants (aarch64, 4-node and 72-node),
with `code/`, `configs/` (by system name), `systems/`, `scaleout/` (SLURM/Pyxis multi-node),
and a documented autotune script at `scripts/autotune`. DeepSeek-R1 runs through a
`trtllm_endpoint` harness started by `run_llm_server`, using the `nvidia/deepseek-r1-fp4`
checkpoint, with an **FP4-v2** variant where "WO_GEMM weight [is] additionally quantized to
FP4". IFB (`run_scaleout.sh`) is used for Offline and disaggregated prefill/decode
(`run_scaleout_disagg.py`) for Server/Interactive because of "better tail latency
characteristics due to workload isolation".
(https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA;
https://raw.githubusercontent.com/mlcommons/inference_results_v6.0/main/closed/NVIDIA/code/deepseek-r1/tensorrt/README.md)

**[verified]** Their tuning guide's concrete rules: `max_num_tokens = (1 + isl/osl) * max_batch_size`
as a rule of thumb; `kvcache_free_gpu_mem_frac > 0.9` for isolated runs and **up to 0.95 on
B200/B300**, backing off in 0.05 steps on OOM; `kv_cache_type: paged`;
`batch_scheduler_policy: max_util` for Offline; `enable_chunked_context` with
`first_come_first_served` or `equal_progress`; DeepSeek-R1 at TP8 with
`moe_expert_parallelism=8`.
(https://raw.githubusercontent.com/mlcommons/inference_results_v6.0/main/closed/NVIDIA/documentation/performance_tuning_guide.md)

**[inferred]** MLPerf's LLM benchmarks are all throughput or 99th-percentile-latency
scenarios; none of them is our C1 single-stream case, and the SingleStream scenario is not
used for the LLM benchmarks. The genuinely useful disclosures are the FP4-v2 detail (they
quantize the attention output projection to FP4 too) and the 0.95 KV-fraction guidance for
183 GB parts.

---

## Blackwell architecture, CUDA, CUTLASS, Nsight

**[verified]** CUDA Blackwell tuning guide, compute capability 10.0 (B200): 64 concurrent
warps per SM; 64K 32-bit registers per SM; max 255 registers per thread; max 32 thread
blocks per SM; **228 KB shared memory per SM**, 227 KB addressable per block (CUDA reserves
1 KB); selectable carveouts of 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB via
`cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout)`; static allocations
still capped at 48 KB, dynamic requires opt-in. Thread-block clusters with distributed
shared memory: portable max 8, **B200 allows a non-portable 16** with
`cudaFuncAttributeNonPortableClusterSizeAllowed`. Use `cudaOccupancyMaxActiveClusters` for
cluster kernels. L2: the guide states GB200 increases L2 capacity to 126 MB. NVLink5
traffic between peers is routed automatically once `cudaDeviceEnablePeerAccess()` is set.
(https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

**[verified]** CUTLASS: 3.8.0 (Jan 2025) established SM100 support — TCGen05 5th-gen tensor
core MMA atoms, **tensor memory (tmem) as a first-class data locale**, block-scaled NVFP4 /
MXFP4 / MXFP6 / MXFP8, and Cluster Launch Control for dynamic persistent scheduling. 4.0.0
(Jun 2025) introduced the **CuTe DSL** (Python). 4.1.0 added aarch64. 4.3.0 added
"Blackwell SM100 kernels for MoEs focusing on Low-Latency inference performance" (example
92) using **TMA for weights and CPASYNC for token loads** — an expert-dispatch-specific
design, not a general grouped GEMM. 4.4.0 added CUDA 13.1 and SM103 (GB300) plus
`cute.experimental` and AoT compilation. 4.5.0 added 2-SM MMA to mixed TMA+CpAsync SM100
GEMMs, a 2-kernel backward FMHA ~25% faster than the 1-kernel version at FP8 on SM103, and
MoE kernels showing "mxfp8_2dx3d: avg 1.29 speedup" over PyTorch including helper-kernel
overhead. 4.7.0 (Aug 2026) added a lower-level Primitives API, a task-scheduling framework
for warp-specialized kernels, and **compile-time register-spill detection**.
(https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md)

**[verified]** CUDA 13.x release notes, inference-relevant: cuBLAS improved FP4 matmul on
Blackwell Ultra by a geomean of **5%** and TF32 by **27%**; PDL is "now supported in some
cuBLAS kernels for architectures sm_90 and above", "decreas[ing] kernel launch latencies
when executed alongside other PDL kernels"; cuBLAS 13.3 added **CUDA Green Contexts**
support; TMA-based SYMV acceleration for Hopper/Blackwell/Blackwell-Ultra.
The section I read does **not** document cluster launch control APIs, tile IR, NVLink
multicast APIs, or graph improvements — **[unverified]** for those.
(https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

**[verified]** Nsight Compute: Blackwell support landed in 2024.4. 2025.3 "added or improved
support for Blackwell chips" and added `launch__persisting_l2_cache_size` to Memory Workload
Analysis. 2025.4 added C2C link information collection on Blackwell. 2026.2 extended support
to more Blackwell SKUs with reduced metric coverage. The release notes I read do **not**
document tcgen05 or tensor-memory-specific metrics or a Blackwell roofline change —
**[unverified]** for those.
(https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)

**[inferred]** For our profiling: the 228 KB smem per SM is what makes the Fuse_A
"prefetch most of the weights into shared memory" trick possible at all, and the
non-portable cluster size of 16 plus 2-SM MMA are the two Blackwell features that change
GEMM tiling decisions relative to Hopper. `launch__persisting_l2_cache_size` is the metric
to watch if we try L2-persisting the router/expert-gate weights.

## AutoDeploy / ADP

**[verified]** AutoDeploy (beta) exports a HuggingFace PyTorch model via `torch.export` to an
ATen graph with custom attention ops, then applies automated transforms — graph sharding for
multi-GPU, KV-cache insertion, GEMM and MHA fusion, quantization, CUDA-graph capture — and
compiles with a backend such as `torch-opt` before running on the TRT-LLM runtime. Entry
point `build_and_run_ad.py`. Docs: `auto-deploy.md`, `pipeline_cache_design.md`,
`support_matrix.md`, `transforms.rst`. **No performance claims are made in the docs.**
(https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/auto_deploy/auto-deploy.md)

Note the acronym collision the assignment flagged: in TRT-LLM, **ADP** in the performance
literature means *Attention Data Parallel* (blog10, the balance strategy above), not
AutoDeploy. They are unrelated.

**[inferred]** AutoDeploy is a time-to-first-deployment tool, not a performance tool. Nothing
in it beats a hand-tuned path, and it has no bearing on our targets.

## NIM

**[unverified].** I could not retrieve `docs.nvidia.com/nim/large-language-models/latest/`
pages — both the supported-models and profiles URLs returned effectively empty bodies.
The only NIM datapoint I could verify is a passing mention in the Qwen3.8 launch post that
"a model-free NVIDIA NIM container deployment option is available via NGC". **I have no
sourced NIM B200 profile, precision or TP recommendation and will not invent one.**

---

## Techniques ranked by transferability to our stack

| Technique | Source | Mechanism in one line | Evidence quality | Effort in our fork | C1 latency | C64 cost/user |
|---|---|---|---|---|---|---|
| NUMA-pin ranks (`TLLM_NUMA_AWARE_WORKER_AFFINITY`) | TRT-LLM deployment guide | bind rank→NUMA node→GPU before any allocation | doc only, no numbers | trivial | ●●○ | ●○○ |
| `vboost 4` + persistence cycle | blog01 | raise sustained clocks | measured +3.4% | trivial | ●●○ | ●○○ |
| PDL everywhere (`TRTLLM_ENABLE_PDL`) | blog01, blog26, CUDA RN | consumer grid starts before producer retires | measured +3% standalone, enabler for more | trivial | ●●○ | ●○○ |
| `enable_min_latency` mode + autotuner off | blog06 | distinct engine path for batch≤8 | config published, no ablation | low | ●●○ | ○○○ |
| One-model EAGLE-3 (`eagle3_one_model`) | blog06 | draft head inside the target engine | config published | low-med | ●●● | ●○○ |
| Suffix automaton on top of EAGLE-3 | features/speculative-decoding | mine the prompt for free drafts | feature doc, no numbers | low-med | ●●○ | ●●○ |
| CUDA graph buckets at +64 above 128 | blog20 | cut padding waste from 1024 to 63 tokens | 1.3–1.5× measured | low | ○○○ | ●●● |
| AR+RMSNorm+NVFP4-quant single-kernel fusion | blog01, allReduceFusionKernels.h | collapse local-reduce/AR/residual/norm/quant | measured +5 tok/s, enum verified | medium | ●●● | ●●○ |
| `MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM` fusion | customAllReduceKernels.h | fold MoE finalize into the AR kernel | enum verified, no isolated number | medium | ●●○ | ●●○ |
| GEMM→window zero-copy (`TLLM_NCCL_SYMMETRIC_ZERO_COPY`) | ops.py | GEMM writes straight into the AR window | comment verified | medium | ●●○ | ●○○ |
| Fuse_A-style smem-prefetch GEMM for M<16 | blog01 | concat down-projections, prefetch weights to smem under PDL | measured +5 tok/s, plus later WIP 299→340 | high | ●●● | ○○○ |
| Custom RouterGEMM for M≤30 / 256 experts | blog01, blog03, blog26 | BF16-in/FP32-out fused router | 252→15 µs, 8→3 µs | medium-high | ●●● | ●●○ |
| Fused top-K (2 kernels instead of 18 ops) | blog03 | one grouped-then-topk kernel with bias/scales | 7.4% E2E | medium | ●●● | ●●○ |
| Mixed MoE parallel (attn TP8, MoE TP4×EP2) | blog01 | rebalance expert sharding vs comms | 253→299 measured | low | ●●● | ●○○ |
| **TP4 instead of TP8** | blog15 (TP=4 min-latency), Dynamo GLM-5.2 (DTP4) | fewer AR participants, bigger per-rank M | config verified twice, no ablation published | medium | ●●● | ●●○ |
| MoE-as-dense-GEMM, gated on token count | blog24 | dense FC1/FC2 + alpha mask in the 64–208 window | measured table on our exact shape | medium | ○○○ | ●●● |
| DeepGEMM MegaMoE (dispatch+GEMM+act+combine fused) | blog26 | whole MoE block as one symmetric-memory kernel | 15.3% tput / 12.7% latency | very high | ●●● | ●●● |
| NVLinkOneSided A2A, rank-major buffers | blog18 | one-sided push/pull over symmetric memory | 81–84% of NVLink peak at bs2048 | high | ●○○ | ●●● |
| GVR exact top-K warm-started from prev step | blog21 | guess/verify/refine, exact output | 1.88× on op; 2.4–7.5% E2E at 64–100K | medium | ●○○ (10k ctx) | ●○○ |
| Fused blockwise-FP8 quant into indexer-K write | blog15 | one kernel for quant + cache write | 32.6–64.2% throughput | medium | ●●○ | ●●○ |
| Sparse-MLA `TMALDG.Gather4` + per-tensor FP8 | blog15 | hardware 4-row gather for sparse KV | up to 47% throughput | high (kernel) | ●●● | ●●● |
| Short-sequence fast path (N ≤ index_topk) | blog15 | skip MQA+topK entirely, separate CUDA graphs | 1.03× at 1K/1K | low | ●○○ | ●○○ |
| Piecewise CUDA graph for prefill | features doc | attention eager, rest graphed | no numbers published | medium | ●●○ (TTFT) | ●○○ |
| ADP balance (`enable_balance`, wait iters) | blog10 | align context scheduling across DP ranks | 1.33×, balance 54→88% | medium | ○○○ | ●●● |
| Relaxed acceptance in thinking phase | blog01/blog02 | accept if draft ∈ top-N within delta of top-1 | +9% acceptance, −2.8 MMLU-Pro | low | ●●● | ●●○ |
| DFlash block-diffusion drafting | NVIDIA blog | draft a whole block in one forward pass | 1.5× over EAGLE-3 (vendor) | high (train a head) | ●●● | ●●● |
| KV-aware routing cost function | Dynamo router docs | overlap credits minus load, explicit formula | formula verified, no numbers | medium | ○○○ | ●●● |
| Wide-EP / EPLB | blog04/08 | expert slots + replication + live weight update | 6.17× on NVL72 EP4→EP32 | very high | ○○○ | ●○○ (8 GPUs) |
| DWDP copy-engine expert prefetch | blog19 | pull remote experts off-SM, overlapped | 14.3% iteration, 8.8% E2E; many caveats | very high | ○○○ | ○○○ |
| MNNVL AllReduce | ops.py | multi-node NVLink AR | gated on aarch64 + multinode | N/A | ○○○ | ○○○ |
| Skip-softmax / BLASST | blog16 | skip softmax+BMM2 for low-max blocks | −13.7% TTFT at 90% sparsity, −7.95% acc | medium | ●○○ | ●○○ |

(● = expected impact for us, ○ = little or none. All impact ratings are **[inferred]**.)

---

## Honest gaps

- **NIM**: could not fetch any NIM documentation. No sourced B200 profile guidance.
- **ModelOpt PTQ recipes**: named in the NVFP4 post but no commands or config files were
  retrievable at any URL I tried. Do not cite commands from this document.
- **trtllm-gen kernel *bodies***: closed. Only options, enums, hash layout and cubins are
  public. Our decode of the specific kernel-name strings is `[inferred]` from verified
  option fields, not from reading a name-construction function.
- **AllReduce one-shot/two-shot selection heuristic**: `kOneShotMaxToken = 128` and
  `_MNNVL_ONE_SHOT_THRESHOLD_BYTES = 1 MiB` are verified constants, but I did not find a
  comment block spelling out the full ONESHOT vs TWOSHOT vs NCCL decision tree in the
  headers I read.
- **Nsight Compute Blackwell metrics** (tcgen05, tmem, roofline changes): not documented in
  the release notes I read.
- **CUDA 13 cluster launch control / tile IR / multicast APIs**: not present in the release
  notes section I read.
- **No NVIDIA single-stream GLM-5.2 tok/s number exists** that I could find. The Dynamo
  recipe reports 57–65 tok/s/user at C64/C128 on an agentic trace; the TRT-LLM GLM-5
  deployment guide presents a Pareto curve as a figure with no tabulated values. Our 365
  tok/s at C1 cannot be compared to either.
- **Web search was unavailable** for this session (budget exhausted at the first call), so
  Chinese-language sources (Zhihu, WeChat) and any NVIDIA blog post not reachable by
  constructing a URL or by walking the blog index pages were out of reach. The
  developer-blog index pages I walked covered roughly April–August 2026.

---

## Sources

All URLs below were fetched and read for this document.

**TensorRT-LLM engineering blogs (`docs/source/blogs/tech_blog/`)**
- blog01 Pushing Latency Boundaries — DeepSeek-R1 on B200: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md
- blog02 DeepSeek-R1 MTP: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog02_DeepSeek_R1_MTP_Implementation_and_Optimization.md
- blog03 DeepSeek-R1 throughput on Blackwell: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog03_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.md
- blog04 Scaling Expert Parallelism: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.md
- blog05 Disaggregated Serving: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog05_Disaggregated_Serving_in_TensorRT-LLM.md
- blog06 Llama4 Maverick + Eagle3: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog06_Llama4_maverick_eagle_guide.md
- blog08 Scaling Expert Parallelism part 2: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog08_Scaling_Expert_Parallelism_in_TensorRT-LLM_part2.md
- blog10 ADP Balance Strategy: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog10_ADP_Balance_Strategy.md
- blog11 GPT-OSS Eagle3: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md
- blog15 DeepSeek V3.2 on Blackwell (DSA): https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md
- blog16 Skip Softmax Attention: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md
- blog17 Sparse Attention in TensorRT-LLM: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md
- blog18 One-Sided AlltoAll over NVLink: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md
- blog19 DWDP: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.md
- blog20 Tuning CUDA Graph Batch Sizes: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md
- blog21 Temporal Correlation Meets Sparse Attention (GVR): https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog21_Temporal_Correlation_Meets_Sparse_Attention.md
- blog24 MoE as Dense GEMM: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.md
- blog26 DeepSeek-V4 on Blackwell: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md
- Directory index of all 26: https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog

**TensorRT-LLM docs and source**
- XQA kernel: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/XQA-kernel.md
- Best perf practice, DeepSeek-R1: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
- Release notes: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/release-notes.md
- PyTorch backend architecture: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/torch/arch_overview.md
- Attention feature doc: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/attention.md
- Speculative decoding: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/speculative-decoding.md
- torch.compile + piecewise CUDA graph: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/torch_compile_and_piecewise_cuda_graph.md
- Parallel strategy: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/parallel-strategy.md
- AutoDeploy: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/features/auto_deploy/auto-deploy.md
- GLM-5 deployment guide (8×B200): https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/deployment-guide/deployment-guide-for-glm-5-on-trtllm.md
- CPU affinity / NUMA: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/docs/source/deployment-guide/configuring-cpu-affinity.md
- `envUtils.cpp`: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/common/envUtils.cpp
- `customAllReduceKernels.h`: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/customAllReduceKernels.h
- `allReduceFusionKernels.h`: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h
- communicationKernels directory: https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/communicationKernels
- `_torch/distributed/ops.py`: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/tensorrt_llm/_torch/distributed/ops.py
- `_torch/autotuner.py`: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/tensorrt_llm/_torch/autotuner.py
- trtllm-gen kernel families: https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/trtllmGenKernels
- trtllm-gen FMHA selector: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h
- trtllm-gen batched GEMM options: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/BatchedGemmOptions.h

**NVIDIA Dynamo**
- README: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/README.md
- Docs navigation: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/index.yml
- GLM-5.2 recipe: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/recipes/model-recipes/glm-5-2.mdx
- GLM-5 NVFP4 recipe: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/recipes/model-recipes/glm-5-nvfp4.mdx
- Kimi-K3 recipe: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/recipes/model-recipes/kimi-k3.mdx
- Qwen3.8 recipe: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/recipes/model-recipes/qwen-3-8-2-4t-a95b-fp8.mdx
- Router routing concepts: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md
- GLM-5-FP8 Pareto sweep: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/ai-simulate-experimental/sweeper-experimental/glm-5-fp8-pareto-sweep.md

**NVIDIA developer blog**
- DFlash speculative decoding: https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/
- Full-stack agentic inference with Dynamo: https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/
- Co-designing attention for long-context inference: https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference/
- Introducing NVFP4: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Qwen3.8-2.4T-A95B on GB300 NVL72: https://developer.nvidia.com/blog/serve-qwen3-8-2-4t-a95b-a-2-4t-parameter-model-with-configurable-reasoning-on-nvidia-gb300-nvl72/
- Blog index pages walked: https://developer.nvidia.com/blog/ , /page/2/ , /page/3/

**MLPerf**
- NVIDIA inference results: https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference
- v6.0 NVIDIA submission tree: https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA
- Submission code directories: https://github.com/mlcommons/inference_results_v6.0/tree/main/closed/NVIDIA/code
- Performance tuning guide: https://raw.githubusercontent.com/mlcommons/inference_results_v6.0/main/closed/NVIDIA/documentation/performance_tuning_guide.md
- DeepSeek-R1 harness README: https://raw.githubusercontent.com/mlcommons/inference_results_v6.0/main/closed/NVIDIA/code/deepseek-r1/tensorrt/README.md

**Platform documentation**
- CUDA Blackwell tuning guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CUDA toolkit release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUTLASS changelog: https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md
- Nsight Compute release notes: https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html

**Attempted and failed (recorded for honesty)**
- https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html — empty body
- https://docs.nvidia.com/nim/large-language-models/latest/profiles.html — empty body
- https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html — 404
- https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/performance — 404
- https://developer.nvidia.com/blog/tag/tensorrt-llm/ — 404
- https://developer.nvidia.com/blog/wp-json/wp/v2/posts?search=... — empty result set
