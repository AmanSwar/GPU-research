# Kernel authoring for Blackwell: Triton, CuTe DSL, TileLang, ThunderKittens, and compilers

## What this is

A survey of the tools you can actually write a fast SM100 kernel in, and the
compiler/search literature behind them, assembled for a team that hand-writes
kernels (`glm-kernels/`, `k3-kernels/`) for GLM-5.2 / Kimi K3 / Qwen3.8 /
DeepSeek V4 inference on 8x B200 SXM.

Every tool below is assessed on the same six axes:

1. **Abstraction** — what unit of work you program against.
2. **Blackwell surface** — does it expose `tcgen05` MMA, TMEM, `mbarrier`, TMA,
   thread-block clusters / 2-CTA MMA, and Cluster Launch Control (CLC)?
3. **Block-scaled FP4/FP8** — can it emit `tcgen05.mma...block_scale` with the
   required scale-factor layouts in TMEM?
4. **Measured performance vs hand-written CUTLASS on Blackwell** — real numbers
   where they exist, and an explicit "no public number" where they don't.
5. **Compile-time cost.**
6. **What it cannot express.**

Results are labelled `[verified]` (I read the number in the primary source),
`[reported]` (the authors claim it in an abstract/README/blog I read), or
`[inferred]` (my arithmetic or reasoning). Hardware and model are stated for
every number, because a 2x on A100 with a 7B dense model tells you almost
nothing about a 400B-class MoE on B200.

> **Audit note (2026-08-17).** Every citation in this document was
> re-verified by fetching the primary source: arXiv abstract pages, the arXiv
> API, GitHub tree/raw endpoints, and vendor documentation. No fabricated
> citations were found. Several *numbers and attributions* were wrong and have
> been corrected in place — the most serious were Mirage's speedup range, the
> HipKittens figure/table reference, the Hidet baseline attribution,
> KernelEvolve's author count, and the L2-flushing benchmarking advice, which
> contradicted NVIDIA's own (newer) measurement methodology guide. Eleven
> systems that were missing — most importantly **Event Tensor** (MLSys 2026, the
> only published 8xB200 MoE megakernel result), **TokenWeave**,
> **ParallelKittens**, **Guess-Verify-Refine**, **AVO** and **CAKE** — have been
> added.

**Reference peaks used throughout (B200 SXM, dense, no sparsity):** BF16/FP16
2.25 PFLOP/s, FP8/FP6 4.5 PFLOP/s, FP4 9 PFLOP/s. The BF16 figure is confirmed
by the FlashAttention-4 paper, which states Blackwell delivers "2.25 PFLOPS vs
1 PFLOPS" for FP16/BF16 against Hopper, and derives its 71%-of-peak claim from it
`[verified]`.

---

## Bottom line for our system

Ranked by expected effect on our two objectives (single-stream latency; tok/s
per GPU at concurrency). Effect estimates are `[inferred]` unless stated; the
evidence behind each is `[verified]`/`[reported]` as marked.

1. **Fuse the decode step into a persistent megakernel. Largest single lever,
   and it is a kernel-authoring decision, not a scheduling one.**
   *Expected effect: most of the 365 → ~500 tok/s gap to TileRT.* Collectives are
   19.6% of our C1 profile and 47% of that is rank-arrival skew — bubbles, not
   bytes. Skew is what per-operator kernel launches manufacture. **Four
   independent groups now measure the same fix**, and one of them measured it on
   our exact hardware:
   - **Event Tensor / ETC** (MLSys 2026): on **8x B200 with NVLink**, running
     **Qwen3-30B-A3B** (128 experts, top-8) — the closest published analogue to
     GLM-5.2 on our box — **1.48x over vLLM 0.11.0rc2 and 1.20x over SGLang
     0.5.3rc0 at batch size 1**, plus 1.40x over cuBLAS+NCCL on fused
     GEMM+ReduceScatter and AllGather+GEMM `[verified]`.
   - **DeepGEMM Mega MoE**: 1.50–1.96x over a DeepEP + TileLang baseline at EP8
     across batch 1–32768, largest at batch 1 `[verified]`.
   - **ThunderKittens megakernel**: Llama-1B bf16 forward in <680 us on B200,
     "over 3.5x" vs vLLM and "more than 1.5x" vs SGLang `[reported]`.
   - **Mirage Persistent Kernel**: 12.5 ms/token vs 14.5 for vLLM/SGLang on
     A100 40GB `[reported]`.

   Event Tensor also cuts warmup from vLLM's 123 s / SGLang's 583 s to **35 s
   with zero JIT graph captures** `[verified]`, which matters for our deploy loop.

2. **Make CuTe DSL (CUTLASS 4.x Python) the default language for any new
   tensor-core-bound Blackwell kernel.** *Expected effect: enables items 1, 3, 4;
   no standalone number.* It is the only Python-level tool exposing the full
   SM100 surface *and* carrying production proof: FlashAttention-4 is written
   entirely in CuTe DSL, hits **1613 TFLOP/s (71% of BF16 peak) on B200 180GB
   SXM6 1000W**, and beats cuDNN 9.13.0 by 1.1–1.3x `[verified]` — while compiling
   in **2.5 s forward / 1.4 s backward vs FA3's C++ templates at 55 s / 45 s**
   `[verified, FA4 Table 4]`. Compile time matters to us specifically because
   fixed shapes mean we recompile a config space many times.

3. **Do not hand-roll the NVFP4 grouped MoE GEMM scale-factor path.** *Expected
   effect: weeks of engineering avoided; correctness risk removed.* The hard part
   is not the MMA — it is getting SFA/SFB into TMEM in the exact `32x16B`-tile,
   `warpx4`-multicast layout that `tcgen05.mma...block_scale` demands, plus the
   `tcgen05.cp`/`tcgen05.mma` implicit-pipeline ordering. CUTLASS ships
   `blockscaled_grouped_gemm/grouped_blockscaled_gemm.py` (and an SM103 variant)
   and `blockwise_gemm/{contiguous,masked}_grouped_gemm.py` `[verified, repo
   tree]`; DeepGEMM ships M-axis contiguous and masked grouped FP8/FP4 GEMMs with
   runtime JIT `[verified, README]`. Fork one.

4. **Write the fused allreduce+norm+quant kernel with `multimem` over NVSHMEM
   symmetric memory, not a library allreduce plus a separate norm.** *Expected
   effect: on the order of 1.2x on the collective-bound portion of decode, i.e.
   ~3–4% end-to-end at C1, more at concurrency.* Evidence: **TokenWeave**
   (MLSys 2026) fuses AllReduce+RMSNorm via NVSHARP/multimem using **only 2–8
   SMs** and gets up to **1.28x latency / 1.19x throughput on an 8xH100 DGX**,
   and explicitly notes such fusions "are not turned on by default" in vLLM,
   SGLang or TensorRT-LLM `[reported]`. **ParallelKittens** reports up to 2.33x
   on data/tensor-parallel and 1.22x on expert-parallel workloads across Hopper
   and Blackwell from the same overlap discipline `[reported]`. CUTLASS 4.x ships
   this exact shape as CuTe DSL examples (`all_reduce_two_shot_multimem.py`,
   `all_reduce_one_shot_lamport.py`,
   `distributed_gemm_blockscaled_all_reduce_ldmcxstmc_blackwell.py`) `[verified,
   repo tree]`. `multimem.ld_reduce` reduces inside the NVSwitch (NVLink SHARP)
   and supports `.acc::f32` for FP16/BF16 operands and `.acc::f16` for FP8
   `[verified, CUTLASS distributed README]` — which is what you need for the norm
   to be numerically defensible.

5. **Steal Guess-Verify-Refine for the DSA indexer top-k.** *Expected effect: up
   to ~7.5% TPOT at long context; less at short.* NVIDIA measured a data-aware
   top-k that reuses the previous decode step's top-k as a prediction signal at
   **1.88x average (up to 2.42x) over the production radix-select kernel** on
   Blackwell for DeepSeek-V3.2's DSA indexer, with **up to 7.52% end-to-end TPOT
   improvement at 100K context in a min-latency deployment**; it is integrated
   into TensorRT-LLM `[reported, arXiv 2604.22312]`. Our indexer is 5.8% of the
   C1 profile and `index_topk_freq=4` means we run selection every 4th step —
   temporal correlation is exactly the exploitable structure.

6. **Stop trying to make Triton hit peak on tensor-core-bound Blackwell kernels;
   use Gluon there, keep Triton for the memory-bound tail.** *Expected effect:
   1.6–3.2x on any attention kernel currently in Triton.* PyTorch's own
   FlexAttention team states flatly: "On Blackwell, high-performance attention
   requires a deeply pipelined, warp-specialized kernel. These techniques aren't
   expressible in our Triton-based implementation" — and measured **1.6–3.2x
   forward / 1.85–2.3x backward** switching FlexAttention's backend from Triton
   to CuTe-DSL FA4 on **GB200 at 1000W** `[verified]`. FA4 independently reports
   2.1–2.7x over Triton on B200 `[verified]`. The CUDA Tile evaluation puts
   Triton at **~62% of cuBLAS for square GEMM on B200** specifically `[verified]`.
   Our DSA indexer, top-k, quantise, and RoPE kernels are a different story —
   Triton is fine and fast there.

7. **Our shapes are fixed, so autotune once, exhaustively, over a small space,
   and freeze the config into source.** *Expected effect: 5–20% per kernel,
   one-off cost.* The legal SM100 space is small: `mma_tiler_mn` x
   `cluster_shape_mn` x `use_2cta_instrs` x `use_tma_store` x pipeline stages x
   raster/swizzle — exactly what CUTLASS's own CuTe DSL autotuning guide searches
   by exhaustive `itertools.product()` `[verified]`. Seed with
   `nvMatmulHeuristics` (Inductor `NVGEMM` backend,
   `nvgemm_max_profiling_configs` default 10) or `cutlass_profiler
   --enable-best-kernel-for-fixed-shape` `[verified]`, then *measure* per §11.
   Do not ship a cost model.

8. **Attack rank-arrival skew with CLC + persistent scheduling + PDL, in the
   kernel.** *Expected effect: recovers part of the 47%-of-collectives skew that
   survives fusion.* Cluster Launch Control lets a persistent worker query for
   the next tile instead of taking a static slice — the documented fix for
   load-imbalanced persistent kernels. CuTeDSL 4.4 added CLC-based persistent
   scheduling `[verified, PyTorch blog]`; CUTLASS 4.7 added a **Task Scheduling
   framework** that statically analyses warp-specialised schedules and *stops
   compilation* on known concurrency issues, with a full NVFP4 tutorial series
   under `examples/python/CuTeDSL/experimental/task_scheduling/blackwell/`
   `[verified, CHANGELOG + repo tree]`. Triton 3.7 exposes CLC in Gluon
   (`gluon/language/nvidia/blackwell/clc.py`) `[verified, repo tree]`. TileLang
   has `gemm_tcgen5mma_ws_clc.py` and now also persistent and Stream-K variants
   `[verified]`. DeepGEMM exposes PDL via `deep_gemm.set_pdl` `[verified]`.

9. **Treat LLM-driven kernel work as a search driver over a *typed template
   space*, not as a source of kernels — but note this changed in 2026.**
   *Expected effect: 0–10% on already-good kernels, at high compute cost.* The
   pessimistic reading is still mostly right: on B200, FlashInfer-Bench measured
   frontier agents at **<50% of SOTA on more than half** of GEMM and GQA
   workloads, with **30 of 32 correctness failures being plain compile errors**
   `[verified]`; the best RL-trained kernel model (Kevin, QwQ-32B base) reaches a
   **mean 1.10x over PyTorch *eager*** `[verified]`. But two 2026 results break
   the pattern by constraining the agent: **AVO** (NVIDIA/UW) ran 7 days of
   agentic evolutionary search on B200 and found attention kernels **up to 10.5%
   faster than FlashAttention-4 and up to 3.5% faster than cuDNN** `[reported]`;
   **CAKE** gives agents a typed, hardware-explicit schedule IR and gets **2.05x
   geomean over the official FlashKDA** on Kimi Delta Attention on B200, while
   noting that letting agents write **direct CUDA/PTX scored only 0.928x**
   `[reported]`. The lesson is the constraint, not the agent.

10. **Keep Helion / `torch.compile` for the long tail, and only there.** *Expected
    effect: 1.85x geomean over hand-written Triton on tail kernels.* Helion posts
    a **3.27x geomean over eager on B200 vs 2.7x for `torch.compile`
    max-autotune and 1.76x for hand-written Triton** on the Liger-Kernel suite
    `[verified]` — but it compiles *to Triton*, so it inherits Triton's Blackwell
    ceiling on anything tensor-core-bound.

11. **Keep a TileLang track for research prototyping of DSA variants, not for
    production.** This is DeepSeek's own posture: for V3.2 they point at TileLang
    for kernels with "better readability and research-purpose design" and at
    DeepGEMM/FlashMLA for "high-performance CUDA kernels" `[verified,
    DeepSeek-V3.2-Exp README]`.

12. **Know that upstream sparse-MLA *decode* on B200 is unoptimised, and budget
    for it.** FlashMLA's own README: the token-level sparse MLA decoding kernel
    hits 410 TFLOP/s on H800 SXM5 but only **"up to 350 TFlops on B200 (which is
    not really optimized yet)"** `[verified]`. Sparse MLA *prefill* forward, by
    contrast, is 1450 TFLOP/s on B200 `[verified]`. Our decode path is 10.9% of
    the C1 profile and there is no good upstream kernel to fork — this is the one
    hot family where we genuinely have to author.

---

## Ground rules: what SM100 demands of a kernel language

Everything below is a consequence of five hardware changes. A tool that does not
expose all five cannot reach peak on B200.

- **`tcgen05` replaces WGMMA.** Instruction tiles for `cta_group::1` are
  M ∈ {64,128} x N ∈ {64,128,192,256} x (4*MMA-K); for `cta_group::2` they are
  M ∈ {128,256} x N ∈ {64,128,192,256} x (4*MMA-K) `[verified, CUTLASS Blackwell
  functionality docs]`. Issue semantics are *single-thread*: one elected lane in
  one warp initiates the whole MMA `[verified, TileLang `mxfp8_illustrated.md`]`.
- **TMEM.** 256 KB per SM, addressed by 32-bit logical addresses in a separate
  address space from SMEM/GMEM, allocated in 32-bit-wide *columns*
  `[reported, Blackwell GPU wiki; CuTe DSL exposes this as `TmemAllocator()`]`.
  ThunderKittens describes the geometry as "128x512 tensor memory per SM …
  divided into two slots: one accessed by tensor cores, the other accessed by
  epilogue threads simultaneously" `[verified, TK 2.0 blog]`. Sub-32-bit
  accumulators pack: the CuTe DSL guide states "TMEM columns are 32-bit wide, so
  scale to element offset for narrower types (e.g. Float16: scale = 32 // 16 =
  2)" `[verified]`. When A is sourced from TMEM it is "laid out after the
  accumulator's TMEM columns" `[verified]`. TMEM is per-SM and not addressable by
  other CTAs outside the pair.
- **Everything is async.** MMA and TMA both complete via `mbarrier`. You own a
  phase-parity protocol. Get it wrong and you get a hang, not a wrong answer.
- **2-CTA MMA + clusters.** SM100 supports multicast clusters (e.g.
  `Shape<_2,[_1|_2|_4],_1>` and `Shape<_4,[_1|_2|_4],_1>` with 2SM instructions);
  SM120 (workstation/GeForce Blackwell) has **no multicast, so cluster shape is
  fixed to 1x1x1** and it is **TN-only**, therefore it cannot use CTA-pair MMA at
  all `[verified, CUTLASS Blackwell functionality docs]`. Cluster size is not
  free: ThunderKittens documents that with cluster size 4 on B200's 148 SMs
  "only 132 SMs are active at a time" `[verified, TK 2.0 Table 1]`.
- **Block scaling is a first-class MMA mode.**
  `tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}` takes
  `[scale-A-tmem]` and `[scale-B-tmem]` operands. `mxf8/mxf6/mxf4` use 32-element
  blocks with UE8M0 scales; `nvf4` uses 16-element blocks with UE4M3 scales (max
  representable 448) `[verified, Colfax]`. Scale factors sit in TMEM in
  32x16-byte tiles, duplicated to all 32 lane partitions, delivered by
  `tcgen05.cp` — CUTLASS wraps this as `cute.nvgpu.tcgen05.Cp4x32x128bOp` with
  `.shape=.32x128b` and `.multicast=.warpx4` `[verified, Colfax]`. Crucially,
  "the `tcgen05.cp` and `tcgen05.mma` form an implicit *tcgen05 pipeline*, which
  guarantees execution in the same order as instruction issuance. This also
  explains why no circular buffer is used for the scale factor tiles in TMEM"
  `[verified, Colfax]`. ThunderKittens found the same thing empirically and
  "recovered the missing ~500 TFLOP/s, roughly a 10% improvement for NVFP4"
  `[verified, TK 2.0]`.

This is the bar. Read the tables below against it.

---

## Family 1: Kernel authoring languages and DSLs

| Tool | Lab | Venue/date | Blackwell surface | Headline result (hardware, model) | Production? |
|---|---|---|---|---|---|
| CUTLASS 3.x/4.x C++ + CuTe | NVIDIA | continuous; 4.0.0 2025-06-03 → 4.7.0 2026-08-04 `[verified, CHANGELOG]` | full: tcgen05, TMEM, TMA, clusters, CLC, block-scale, 2SM, EVT | reference implementation; **1570 TFLOP/s** BF16 GEMM M=N=K=8192 on B200 `[verified, HipKittens Table 2]` | yes — everywhere |
| **CuTe DSL** (CUTLASS Python) | NVIDIA | CUTLASS 4.0, 2025-06-03 | full, same as C++; EFC epilogue fusion since 4.4 | FA4: **1613 TFLOP/s (71% of BF16 peak)**, 1.1–1.3x over cuDNN 9.13.0, 2.1–2.7x over Triton, B200 180GB SXM6 1000W, BF16, hd 64/128/(192,128) `[verified]`; compile 2.5 s vs FA3 55 s `[verified]` | yes — FA4, QuACK, FlexAttention FLASH backend, FlashInfer |
| Triton (`tl.*`) | OpenAI | MAPL'19; 3.4.0 2025-07-30 → 3.7.1 2026-06-18 `[verified, GitHub releases API]` | partial: TMA descriptors, `dot_scaled`→tcgen05, TMEM/WS via compiler passes; no explicit TMEM/mbarrier control | **62–101% of cuBLAS** across H100 NVL / B200 / RTX PRO 6000, but **~62% for square GEMM on B200** `[verified, arXiv 2604.23466]`; 2.1–2.7x *slower* than FA4 on B200 attention `[verified]` | yes — vLLM/SGLang glue, `triton_kernels` MoE |
| **Gluon** (Triton's low-level dialect) | OpenAI | shipped in Triton 3.4+ | full: `allocate_tensor_memory`, `tcgen05_mma_scaled/copy/commit`, `tma.async_load/store`, `mbarrier.*`, `gl.warp_specialize`, CLC, 2CTA | 4870.97 TFLOP/s mxfp4, 4846.83 nvfp4, 2378.49 mxfp8 at M=N=K=8192 `[verified, official tutorial table]` — ≈54% of B200 FP4 peak `[inferred]`; GPU model unstated in tutorial | emerging (Triton internals, some prod MoE) |
| TileLang | Lei Wang, Yu Cheng, …, Lingxiao Ma, Jilong Xue, Fan Yang, Zhi Yang (MSRA/PKU) | arXiv 2504.17577, 2025-04-24 (v2 2025-04-27) | preview: `T.alloc_tmem`, `T.tcgen05_gemm`, 2SM, CLC, persistent, Stream-K; MXFP8 block-scale on SM100 | H100: 1.36x FA3, 1.41x Triton (attention); 98% of FlashMLA in ~70 LOC `[reported]`. **No SM100-vs-CUTLASS number published.** | research; DeepSeek ships TileLang DSA kernels as the *readable* reference |
| ThunderKittens 2.0 | HazyResearch (Stanford) | blog 2026-02-19 | full: tcgen05.mma/cp/alloc, TMEM, TMA, DSMEM, clusters | B200 BF16 GEMM **1538 TFLOP/s** vs CUTLASS 1570 `[verified, HipKittens Table 2]`; "at or near cuBLAS speeds", attention "near-cuDNN speeds on B200" `[verified, Together blog]` | partial (Together, internal forks) |
| **ParallelKittens** | Sul, Arora, Spector, Ré (Stanford) | arXiv 2511.13940, 2025-11-17 | TK-based, multi-GPU: 8 primitives + unified template | Hopper + Blackwell: **2.33x** DP/TP, **4.08x** sequence-parallel, **1.22x** expert-parallel; "<50 lines of device code" `[reported]` | research; TK ecosystem |
| Mosaic GPU / Pallas | Google (JAX) | continuous | `plgpu.tcgen05_mma`, `plgpu.TMEM`, `copy_gmem_to_smem`, `async_{load,store}_tmem`, `Barrier(orders_tensor_core=True)`, `ClusterBarrier`, `tcgen05_commit`, `cluster=`, `dynamic_scheduling_loop` — all confirmed present `[verified, JAX docs]` | **no public B200 vs CUTLASS number found** | JAX/TPU-first shops only |
| Helion | Meta (Jason Ansel, Oguz Ulgen, Will Feng et al.) | PyTorch blog 2025-10-23; beta 2025-10-22 | inherits Triton's | B200 geomean **3.27x** over eager vs **2.7x** `torch.compile` max-autotune, **1.76x** hand Triton, Liger-Kernel suite `[verified]`; matches QuACK CuTe DSL RMSNorm-bwd on H100 `[verified]` | beta |
| CUDA Tile / cuTile | NVIDIA | independent eval: Yadav, Zhao, Kumar, arXiv 2604.23466, 2026-04-25 (v2 2026-06-03) | tile-level, compiler-managed | B200 attention **1007 TFLOP/s, 2.5x FA2**; GEMM **52–79% of cuBLAS** (B200 8192³: 875.8 vs cuBLAS 1671.8) in **22 LOC** vs 123 WMMA / 53 Triton `[verified]`. Falls to 53% of FA2 on RTX PRO 6000. | no |
| Hidet | Ding, Yu, Zheng, Liu, Wang, Pekhimenko (Toronto/AWS) | ASPLOS 2023, arXiv 2210.09603 | pre-Blackwell | **up to 1.48x (1.22x average) over ONNX Runtime and TVM**; tuning 20x faster than AutoTVM, 11x faster than Ansor `[verified, abstract]` | no (absorbed into CentML) |
| torch.compile / Inductor | Meta | continuous | Triton templates + `CUTLASS`/`CUTEDSL`/`NVGEMM` autotune backends | gpt-fast: 25.5 → 244.7 tok/s Llama-7B, A100-80GB **power-limited to 330W** `[verified]` | yes — everywhere, as glue |

---

## Family 2: Autotuning and search

| System | Lab | Venue/date | Approach | Result | Cost |
|---|---|---|---|---|---|
| CUTLASS profiler | NVIDIA | continuous | enumerate + measure | `--enable-best-kernel-for-fixed-shape`, `--sort-results-flops-per-sec`, `--raster_order`, `--swizzle_size`, `--workspace-count`, `--llc-capacity` `[verified]` | minutes–hours |
| CUTLASS GEMM measurement methodology guide | NVIDIA | CUTLASS 4.6, 2026-07-01 | how to benchmark correctly (see §11) | buffer rotation ≥2x L2; warm up until power stabilises; e.g. 10,000 warmups / 4,000 profiling iters for large Blackwell GEMMs `[verified]` | — |
| CuTe DSL autotuning guide | NVIDIA | CUTLASS 4.x | exhaustive `itertools.product()` over `use_2cta_instrs`, `use_tma_store`, `mma_tiler_mn`, `cluster_shape_mn`; two-level `config_kernel_dict`/`input_kernel_dict` cache `[verified]` | — | compile+bench per config |
| `nvMatmulHeuristics` | NVIDIA | shipped; surfaced as Inductor `NVGEMM` backend | heuristic → top-N configs (`nvgemm_max_profiling_configs=10`) `[verified, `torch/_inductor/config.py`]` | — | ~zero |
| cuBLASLt heuristics | NVIDIA | continuous | `cublasLtMatmulAlgoGetHeuristic` returns ranked candidates | — | ~zero |
| Ansor | Zheng, Jia, Sun, Wu, Yu, Haj-Ali, Wang, Yang, Zhuo, Sen, Gonzalez, Stoica (Berkeley) | OSDI 2020, arXiv 2006.06762 | hierarchical sketch + random annotation + evolutionary search + learned cost model | up to 3.8x Intel CPU, 2.6x ARM CPU, **1.7x NVIDIA GPU** vs SOTA `[verified, abstract]` | hours–days |
| Roller | Zhu, Wu, Diao, Ke, Li, Zhang, Xue, Ma, Xia, Cui, Yang, Yang, Zhou, Cidon, Pekhimenko (MSRA) | OSDI 2022 (July 2022) | **construction, not search**: rTile aligned to hardware, micro-performance model | "generate efficient kernels in *seconds*"; "comparable performance to the state-of-the-art solutions on popular accelerators like GPUs" `[verified]` | seconds |
| Hidet | Toronto/AWS | ASPLOS 2023 | task-mapping in the program + post-scheduling fusion | 11x faster tuning than Ansor, 20x than AutoTVM `[verified]` | minutes |
| Helion autotuner | Meta | 2025-10 | implicit search space, differential evolution / pattern search | "Autotuning complete in 586.6s after searching 1520 configs" `[verified]` ≈ 0.39 s/config | ~10 min/kernel |
| WaveTune | Zhang, Ding, Qian, Wang, Cao, Xue, Huang, Yang, Zhang (SJTU et al.) | arXiv 2604.10187, 2026-04-11 | wave-aware bilinear latency model + sparse sampling + dual-table retrieval | up to **1.83x** kernel, **1.33x** TTFT; decision overhead 5 orders of magnitude below exhaustive; 3 kernels, 5 GPU architectures `[verified, abstract]` | near-zero at runtime |
| Mirage | Wu, Cheng, Liu, Shi, Ji, Ao, Velliengiri, Miao, Padon, Jia (CMU) | OSDI 2025, arXiv 2405.05751 (v3 2025-06-06) | μGraph superoptimisation + abstraction pruning + probabilistic equivalence check | **up to 3.3x** over best baseline on A100 40GB / H100: GatedMLP 2.7–3.3x, GQA up to 2.2x, LoRA 1.1–2.4x, QKNorm 1.4x `[verified]` | **up to 4 hours per Lax program** `[verified]` |
| Tawa | Chen, Fan, Collins, Hagedorn, Gaburov, Masuda, Brookhart, Sullivan, Knight, Zhang, Grover (NVIDIA + Cornell) | CGO 2026, arXiv 2510.14719 | automatic warp specialisation via `aref` IR | **H100 only**: up to 1.1x cuBLAS GEMM, 1.2x Triton attention, matches CUTLASS FA3 `[verified, abstract]` | compiler pass |
| **RaMP** | Sharma, Datta | arXiv 2604.26039, 2026-04-28 | runtime expert-histogram-aware kernel dispatch, 4-parameter wave cost model | claims 1.22x over static dispatch, 1.30x vs Triton, **1.41x vs DeepGEMM**, 1.13x vs FlashInfer CUTLASS; 10–24 min one-time profiling; 0.93% mean regret vs exhaustive `[reported]` — **hardware not named in the abstract; treat as thin evidence** | ~zero at runtime |

---

## Family 3: Fusion, collectives and megakernels

| Work | Lab | Venue/date | Hardware | Result | Production? |
|---|---|---|---|---|---|
| **Event Tensor / ETC** | Jin, Hou, Wang, Lai, Chen, Ye, … Grover, Mowry, Jia, T. Chen (CMU/NVIDIA/UW) | **MLSys 2026**, arXiv 2604.13327, 2026-04-14 | **8x B200 NVLink** | Qwen3-30B-A3B (128 experts, top-8): **1.48x vs vLLM, 1.20x vs SGLang at BS=1**; Qwen3-32B dense 1.15x vs vLLM; fused GEMM+RS / AG+GEMM **1.40x vs cuBLAS+NCCL**; MoE layer 1.23x at 1024 tokens; warmup **35 s (0 captures)** vs vLLM 123 s (67) / SGLang 583 s (51) `[verified]` | research; ahead-of-time compiled |
| **TokenWeave** | Gond, Kwatra, Ramjee (Microsoft Research India) | **MLSys 2026**, arXiv 2505.11329 (v5 2026-05-01) | 8x H100 DGX | fused AllReduce+RMSNorm via NVSHARP/Multimem in **2–8 SMs**: up to **1.28x latency, 1.19x throughput**; beats "all communication removed" in some settings `[reported]` | code released; explicitly *not on by default* in vLLM/SGLang/TRT-LLM |
| **ParallelKittens** | Sul, Arora, Spector, Ré (Stanford) | arXiv 2511.13940, 2025-11-17 | Hopper + Blackwell | 2.33x DP/TP, 4.08x sequence-parallel, 1.22x expert-parallel; 8 primitives, <50 LOC device code `[reported]` | research |
| DeepGEMM Mega MoE | DeepSeek | released 2026-04-16 (PR #304/#316) | SM100 `[inferred: FP8xFP4 requires it]`; GPU not stated in PR | EP8, V4-Flash/V4-Pro: **1.50–1.96x** vs DeepEP+TileLang legacy, largest at batch 1 `[verified]` | yes — DeepSeek production library |
| TK megakernel ("No bubbles") | HazyResearch | blog 2025-05-27 | H100, B200 | Llama-1B bf16: <1 ms on H100 ("almost 2.5x vs vLLM, over 1.5x vs SGLang"); **<680 us on B200, "over 3.5x" vs vLLM, "more than 1.5x" vs SGLang** `[verified]` | research |
| Mirage Persistent Kernel (MPK) | CMU | blog 2025 | A100 40GB | 12.5 ms/token vs 14.5 vLLM/SGLang; floor ~10 ms `[reported]` | research |
| AutoMegaKernel | Jaber, Jaber | arXiv 2606.09682, 2026-06-08 | — | agent harness compiling a HF Llama-family forward into one persistent cooperative kernel `[reported]` — no verified perf number | no |
| **SonicMoE** | Guo, Mishra, Cheng, Stoica, Dao | arXiv 2512.14080, 2025-12-16 (v2 2026-03-26) | H100 + Blackwell | vs DeepGEMM on Blackwell: **25% forward / 15% backward** speedup; 1.86x vs ScatterMoE BF16 on Hopper; 45% activation-memory reduction; tile-aware token rounding 1.16x `[reported]` | research |

---

## Family 4: LLM-driven kernel generation

| Work | Lab | Venue/date | Hardware | Claim | What survives scrutiny |
|---|---|---|---|---|---|
| KernelBench | Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini (Stanford) | arXiv 2502.10517, 2025-02-14 | not stated in abstract | 250 tasks, `fast_p` metric | frontier reasoning models match the PyTorch **eager** baseline in **<20%** of cases `[verified]` |
| Stanford CRFM "fast kernels" blog | Stanford | 2025-05-28 | **L40S**, FP32 | LayerNorm 484.4%, Conv2D 179.9%, Softmax 111.8%, matmul 101.3% of torch | authors' own caveat: FP16 matmul **52%** of `torch.matmul`, FP16 flash attention **9%** of SDPA; "FP32 is less common in modern ML workloads and often less optimized on recent hardware" `[verified]` |
| Sakana "AI CUDA Engineer" | Sakana AI | Feb 2025 | — | "up to 100x" | **retracted**: reward-hacked the eval harness (bypassed accuracy validation); one user measured a 3x *slowdown*; Sakana rebuilt the harness and revised the paper `[verified, TechCrunch 2025-02-21]` |
| Kevin | Baronio, Marsella, Pan, Guo, Alberti | arXiv 2507.11948, 2025-07-16 | not stated in abstract | multi-turn RL, QwQ-32B base | correctness 56%→82%; **mean 0.53x → 1.10x of PyTorch *eager***, beating o4-mini's 0.78x `[verified]` |
| FlashInfer-Bench | Xing, Zhai, Jiang, Dong, Wu, Ye, Ruan, Huang, Zhang, Yin, Bayyapu, Ceze, T. Chen (UW/NVIDIA) | arXiv 2601.00227, 2026-01-01 | **B200** | 1,600 workloads across 41 kernel configurations, 240 solutions, 9,600 results | GPT-5 83.9% / o3 71.3% / Gemini 2.5 Pro 48.8% pass; **<50% of SOTA on more than half** of GEMM and GQA workloads; **30 of 32 correctness errors are compile errors** `[verified]` |
| SOL-ExecBench | E. Lin, Modi, Hari, … Kozyrakis, Shi (NVIDIA + multi-lab, 35 authors) | arXiv 2603.19173, 2026-03-19 | **Blackwell** | 235 problems from 124 production/emerging models; SOLAR pipeline computes hardware speed-of-light bounds; sandboxed anti-reward-hack evaluation `[verified]` | the right methodology |
| KernelEvolve | Liao, Qin, Wang, Golden, Kuchnik, Yetim et al. (Meta, **36 authors**) | arXiv 2512.23236, 2025-12-29 (v4 2026-07-06) | NVIDIA / AMD / MTIA | 100% pass on 250 KernelBench problems + 160 ATen ops, targeting Triton and CuTe DSL `[verified]` | correctness ≠ speed; abstract gives **no numeric speedup** |
| **AVO** | T. Chen, Ye, Xu, Ye, Liu, Hassani, T. Chen, Kerr, Wu, … Krashinsky, Grover, Ceze, Shi (NVIDIA/UW) | arXiv 2603.24517, 2026-03-25 | **B200** | agentic variation operators driving evolutionary search, 7 days continuous | **up to 10.5% over FlashAttention-4** and **up to 3.5% over cuDNN** on MHA; GQA up to 9.3% over FA4, 7.0% over cuDNN `[reported]` — the first credible agent-beats-SOTA-on-B200 result |
| **CAKE** | Ye, Huang, Jin, Hou, Shao, Yu, Chen, Cowan, Cao, Xing, H. Chen, Grover, T. Chen, Ceze | arXiv 2608.12629, 2026-08-12 | Ampere → **B200** | typed, hardware-explicit schedule IR ("CAKE IR") exposing warp roles, memory movement, sync, pipelines; agents author the IR | **2.05x geomean over official FlashKDA** (Kimi Delta Attention); Flash-KMeans 1.144x vs tuned FlashML; dispatcher-backed 1.42–2.12x across 400+ shapes; **direct CUDA/PTX only 0.928x** `[reported]` |
| Harness Engineering | Shui, Ma, Xu, Wen, Wang | arXiv 2607.17979, 2026-07-20 | **B200** | MLSys 2026 FlashInfer AI Kernel Generation Contest: mean-latency speedups 1.62x / 18.05x / 29.68x / 1.12x / 13.70x over **supplied contest baselines** `[reported]` | baselines are contest-provided, not necessarily production; the durable finding is that **agent-assisted beat full-agent** |

---

# Deep sections

## 1. CuTe DSL (CUTLASS 4.x Python) — the default choice

**Abstraction.** Identical to CUTLASS C++ CuTe: `Layout` (shape:stride algebra),
`Tensor`, `TiledCopy`, `TiledMma`, `Atom`. You write `@cute.jit` host code and
`@cute.kernel` device code in Python; it lowers through MLIR to PTX. The mental
model is unchanged from C++ — which is the point: everything you already know
about CuTe layout algebra transfers.

**Blackwell surface: complete.** The canonical GEMM shape is:

```python
op = tcgen05.MmaF16BF16Op(
    cutlass.Float16, cutlass.Float32,
    (128, 256, 16),                 # instruction shape M,N,K
    tcgen05.CtaGroup.ONE,           # or TWO for 2-SM MMA
    tcgen05.OperandSource.SMEM,     # A from SMEM or TMEM
    OperandMajorMode.K, OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(op)
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
tCtAcc    = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stages))
tmem_ptr  = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc    = cute.make_tensor(tmem_ptr, tCtAcc.layout)
...
tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
```

`[verified, CUTLASS tcgen05 programming guide]`. When A is sourced from TMEM
rather than SMEM, "a fragment in TMEM is laid out after the accumulator's TMEM
columns", offset using `column_to_element_scale = 32 // dtype.width` `[verified]`.

**What ships for our exact kernel families** (verified 2026-08-17 by walking the
GitHub tree API on `NVIDIA/cutlass@main`, path
`examples/python/CuTeDSL/cute/blackwell/`):

- `kernel/dense_gemm/` — 6 variants: `dense_gemm.py`, `_persistent`,
  `_persistent_dynamic`, `_persistent_prefetch`, `_alpha_beta_persistent`,
  `_software_pipeline`.
- `kernel/blockscaled_gemm/` — `dense_blockscaled_gemm_persistent{,_amax,_prefetch}.py`
  plus `sm103_dense_blockscaled_gemm_persistent.py` (B300).
- `kernel/blockscaled_grouped_gemm/` — `grouped_blockscaled_gemm.py` and
  `sm103_grouped_blockscaled_gemm.py` — **NVFP4 grouped MoE GEMM.**
- `kernel/blockwise_gemm/` — `blockwise_gemm.py`, `contiguous_grouped_gemm.py`,
  `masked_grouped_gemm.py` — the DeepSeek-style contiguous (prefill) and masked
  (CUDA-graph decode) layouts.
- `kernel/moe/` — `moe_persistent_scheduler.py`, `moe_sched_extension.py`,
  `moe_utils.py`, `torch_grouped_mm.py`, `torch_scaled_grouped_mm.py`.
- `kernel/attention/mla/` — `mla_decode_fp8.py`, `mla_decode_fp16.py`,
  `mla_helpers.py` — **MLA decode.**
- `kernel/attention/fmha/` — `fmha.py`, `fmha_bwd.py`;
  `mixed_input_fmha/` — `prefill_d256`, `prefill_d512`, `decode`;
  `attention/mamba2_ssd/` — Mamba-2 SSD with its own tile scheduler.
- `kernel/mixed_input_gemm/` — `mixed_input_gemm.py`,
  `grouped_mixed_input_gemm{,_acc_scale}.py` (the acc_scale variant was added in
  4.4 specifically "to deliver better performance for decoding cases").
- `kernel/distributed/` — `all_reduce_simple.py`, `all_reduce_one_shot_lamport.py`,
  `all_reduce_two_shot_multimem.py`, `all_reduce_tma.py`,
  `distributed_gemm_all_reduce_{,lamport_,ldxstmc_}blackwell.py`,
  `distributed_gemm_blockscaled_all_reduce_ldmcxstmc_blackwell.py`,
  `distributed_gemm_reduce_scatter_blackwell.py`,
  `distributed_all_gather_gemm_blackwell.py`.
- `kernel/rmsnorm/rmsnorm.py`, `kernel/reduce/reduce.py`,
  `kernel/grouped_gemm/grouped_gemm.py`.
- `efc/` — **Epilogue Fusion Configuration**: `custom_epilogue_dense_gemm.py`,
  `activation_custom_epilogue_dense_gemm.py`,
  `synthetic_custom_epilogue_dense_gemm.py`.
- Tutorials: `tutorial/tutorial_gemm/{fp16_gemm_0..6, nvfp4_gemm_0, nvfp4_gemm_1}.py`,
  `tutorial/tutorial_tma/`.
- `experimental/task_scheduling/blackwell/tutorial/` — 7-part series including
  `03_persistent_scheduling_dynamic_domain_ts`, `05_gemm_nvfp4_ts`,
  `06_gemm_split_k_fp16_ts`, and `07_group_pipeline_ts` (which contains
  `03_merge_gemm_resadd.py` and `04_fork_load_gemm_act.py` — i.e. worked examples
  of exactly the kind of cross-operator fusion recommendation #1 needs).

That list covers all five of our hot kernel families. This is the single
strongest argument for CuTe DSL: you are forking, not writing.

**Compile time.** FA4's Table 4 is definitive: **2.5 s forward / 1.4 s backward
in CuTe DSL vs 55 s / 45 s for FA3's C++ templates** `[verified]`; the paper
summarises this as "20-30× faster compile times". JIT results are cached in
memory and on disk; the cache key hashes the generated MLIR bytecode, all CuTe
DSL Python sources, all shared libraries, and all CuTe DSL env vars `[verified,
JIT caching docs]`. Override with `CUTE_DSL_CACHE_DIR`; disable file caching with
`CUTE_DSL_DISABLE_FILE_CACHING=True`. AoT compilation landed in **4.4**
(2026-02-14, examples under `cute/export/`); `cute.compile_to` for custom
compile-execute pipelines in **4.6** (2026-07-01), which also ships standalone
compiler binaries so you can build a pipeline outside Python `[verified,
CHANGELOG]`.

**What it cannot express** `[verified, official limitations page]`:
- No `global`; capturing outer-scope variables outside the JIT context raises at
  runtime.
- Functions can only return `constexpr` values — "returning dynamic values is not
  yet supported."
- Lists/tuples/dicts are compile-time metaprogramming only — no structural
  mutation in-kernel, no dynamic list indexing.
- **CuTe layout algebra is restricted to 32-bit shapes and strides.** "64bit or
  arbitrary width support is planned for future releases." For a paged KV cache
  spanning >2^31 elements you will be doing pointer arithmetic outside the layout
  system.
- No early `return` out of if/else; variables must keep consistent types through
  loops and conditionals.
- **~2–3 us per-tensor DLPack conversion overhead on the host** — this matters at
  concurrency 1 where a decode step is ~2.7 ms. Mitigate with the TVM-FFI path
  (added 4.3) or AoT.
- Debugging is materially worse than C++: no single-step through JIT code.
- Not supported: convolutions, preferred clusters, Windows, and — note —
  **"Task Scheduling support for existing CuTe DSL and extension kernels"**. The
  4.7 Task Scheduling framework does not retrofit onto kernels you already wrote.

**Version note.** Things we care about landed late.
- **4.3** (2025-11-21): TVM-FFI support for reduced host runtime overhead.
- **4.4** (2026-02-14): AoT compilation; SM103/GB300 FP4 "Ultra" blockscaled GEMM;
  JAX support; `cute.experimental` fragment-free layer with automatic TMA
  descriptor generation; **EFC (Epilogue Fusion Configuration)** — "customized
  epilogue fusion for persistent dense GEMM through a Python Epilogue Fusion
  Configuration (EFC) function, somewhat similar to CUTLASS C++ EVT";
  `CopyDsmemStoreOp` for distributed shared memory.
- **4.5** (2026-05-01): `block_copy()` to simplify TMA and S2T (SMEM → TMEM)
  copies; **MXF8 x MXF4 / MXF8 x MXF6 mixed block-scaled MMA**.
- **4.6** (2026-07-01): `cute.compile_to`; **IKET** (In-Kernel-Event-Tracing)
  profiler for intra-kernel activity tracing of persistent warp-specialised
  kernels; distributed compiler binaries; the GEMM performance measurement
  methodology guide (§11).
- **4.7** (2026-08-04): the **Primitives API** — "a lower-level abstraction
  beneath CuTe enabling Tensor Core programming through SIMT … a stable, thin
  wrapper over NVVM operations to use where CuTe abstractions reduce development
  velocity" (experimental); the **Task Scheduling framework** (static analysis of
  warp-specialised schedules, compilation stops on known concurrency issues, plus
  dependency visualisation); **compile-time register-spill and local-memory
  reporting with source line numbers**.

`[all verified, CHANGELOG]`. Pin ≥4.6 for IKET alone; ≥4.7 if you want the spill
reporting and schedule checker, which for a megakernel is worth the upgrade
risk.

## 2. CUTLASS 3.x/4.x C++ — still the floor and the ceiling

Everything CuTe DSL does, C++ does first, plus: EVT (Epilogue Visitor Trees) with
the widest fusion vocabulary, sparse GEMM `(2/4*MMA-K)` variants, preferred
clusters, convolutions, and the widest dtype/layout matrix. SM100 supports
TN/NN/NT/TT and multicast clusters; SM120 is **TN-only, cluster shape fixed to
1x1x1 (no multicast)**, with pingpong and cooperative schedules `[verified,
Blackwell functionality docs]`.

**Correction to earlier advice in this document:** "use C++ when you need EVT" is
now too strong. CuTe DSL got EFC in 4.4, with three worked examples under
`cute/blackwell/efc/`, and 4.7 added scalar reductions and per-operand data
movement strategy to the CUTLASS Operator API's custom epilogue fusions
`[verified, CHANGELOG]`. Use C++ when you need **sparsity, convolutions,
preferred clusters, or an AoT binary with no Python**; CuTe DSL otherwise.

The cost of C++ is compile time (the 55 s FA3 figure is representative) and the
template error surface. Alignment requirements are a real trap: for
narrow-precision block-scaled GEMMs with mixed input datatypes (e.g.
`mx_float4_t` x `mx_float6_t`) both A and B must be **128-element aligned**
`[verified]` — this bites when your MoE intermediate dimension is not a nice
multiple.

## 3. Triton and Gluon — one repo, two languages, very different ceilings

### Triton proper

Triton's Blackwell backend matured across five releases, dates verified against
the GitHub releases API: **3.4.0 (2025-07-30)**, **3.5.0 (2025-10-21)**, 3.5.1
(2025-11-12, SM103/GB300 fixes), **3.6.0 (2026-01-21)**, **3.7.0 (2026-05-07)**,
3.7.1 (2026-06-18) `[verified]`.

- 3.5.0: `tcgen05.commit`, generic lowering for `tcgen05.ld/st`, subtile QK TMEM
  load, warp specialisation for Hopper/Blackwell, ragged TMA.
- 3.6.0: **`tcgen05.mma.scaled` support, native FP4 scaled dot, native MXFP-FP8
  scaled dot**, TMEM bitwidth encoding, TMEM layout broadcasting, `aref`-based
  end-to-end warp specialisation, TMA gather4 on sm_120.
- 3.7.0: **2-CTA mode end-to-end**, Gluon multi-CTA + 2CTA, Blackwell scale
  swizzling for batched matmul, fine-grained cluster barrier, `tcgen05.ld.red`
  on sm_103, block-scaled matmul baselining for mxfp8/nvfp4.

So: yes, Triton can emit `tcgen05` and block-scaled FP4 today, via `tl.dot_scaled`.
The question is whether it hits peak, and the honest answer is *for GEMM on
Hopper, close; for GEMM on Blackwell, no; for attention on Blackwell, badly no*.

**GEMM.** The only independent measurement covering B200 is the CUDA Tile
evaluation (arXiv 2604.23466; Yadav, Zhao, Kumar; 2026-04-25, v2 2026-06-03),
which summarises Triton as sustaining **"62–101% of cuBLAS performance across all
tested platforms without architecture-specific tuning"** — but the band is not
uniformly distributed: **98% of cuBLAS at 4096² on H100 NVL, and ~62% for square
GEMM on B200** `[verified]`. The bottom of that band is on our newest hardware
and on the kernel family that is 37.1% of our profile.

**Attention.** Two independent sources put Triton 2–3x behind on Blackwell:
FA4 measures **2.1–2.7x over Triton on B200 SXM6** `[verified]`;
PyTorch's FlexAttention team measures **1.6–3.2x forward / 1.85–2.3x backward**
by swapping Triton for the CuTe-DSL FA4 backend on **GB200 at 1000W**
`[verified]`. They also disclose that FlexAttention-on-Triton has *regressed*
relative to FA3 on Hopper — "roughly 60% of FlashAttention-3's throughput" today,
down from ~80% at the original blog post `[verified]`. Their diagnosis
generalises to our sparse MLA kernel:

> "On Blackwell, high-performance attention requires a deeply pipelined,
> warp-specialized kernel. These techniques aren't expressible in our
> Triton-based implementation."

The mechanism: "The tensor cores got bigger and faster, but the special-function
unit (SFU), which handles operations like exponentials, didn't keep pace. For
forward attention, this shifts the bottleneck: softmax's exp() is now as
expensive as the matrix multiplies. To keep the GPU fully saturated, you need to
ping-pong between two tiles, overlapping one tile's matrix multiplies with the
other's exponentiation" `[verified]`. That is a multi-role warp-specialised
schedule a general-purpose scheduler will not discover.

**One constraint we must design around.** Because FA4 processes two M-tiles per
CTA (`q_stage=2`), **the minimum sparse block size on Blackwell is 256x128, up
from 128x128 on the Triton path** `[verified, PyTorch blog]`. Our DSA block-mask
granularity has to match, and paged-KV page sizes that assume 128-row blocks will
not line up.

### Gluon — Triton's escape hatch, and it is genuinely good

Gluon is a second frontend in the Triton repo that drops the auto-layout,
auto-pipeline machinery and hands you the hardware. From the Gluon intro, quoted
by the PyTorch team `[verified]`:

> "While the Triton compiler does a good job of generating efficient code for a
> wide range of kernels, it can be beaten by hand-tuned low-level code. When
> this happens, there is little the user can do to significantly improve
> performance since all the details are hidden."

The Blackwell surface is complete: `allocate_tensor_memory`,
`TensorMemoryLayout`, `TensorMemoryScalesLayout`, `tensor_memory_descriptor`,
`tcgen05_mma_scaled`, `tcgen05_copy`, `tcgen05_commit`, `tma.async_load/store`,
`tma.store_wait`, `mbarrier.{init,expect,wait,arrive,invalidate}`,
`fence_async_shared`, `gl.warp_specialize`, `gl.AutoLayout`, plus
`python/triton/experimental/gluon/language/nvidia/blackwell/clc.py` for Cluster
Launch Control `[verified, repo tree + tutorial]`.

**The block-scaled Gluon tutorial is the single best public Blackwell FP4
optimisation walkthrough that exists.** Its measured table (M=N=K=8192; the page
requires "a Blackwell NVIDIA GPU" but does not name the model) `[verified]`:

| Step | mxfp8 x mxfp8 | mxfp4 x mxfp4 | mxfp8 x mxfp4 | nvfp4 x nvfp4 |
|---|---|---|---|---|
| `simple_mma_scaled` | 33.41 | 67.02 | 34.60 | 70.84 |
| `mma_scaled_contig` | 663.28 | 1435.05 | 741.82 | 1303.69 |
| `mma_scaled_packed_block` | 900.97 | 2081.76 | 1000.48 | 2002.05 |
| `mma_scaled_tcgen05_copy` | 929.07 | 2147.76 | 1035.60 | 2092.39 |
| `mma_scaled_pipelined` | 2018.58 | 3916.62 | 2144.05 | 3842.19 |
| `mma_scaled_warp_specialized` | **2378.49** | **4870.97** | **2615.73** | **4846.83** |

TFLOP/s. Read this as an optimisation curriculum: naive → contiguous SF layout →
packed SF blocks → `tcgen05.cp` for SF delivery → software pipelining → warp
specialisation, each roughly doubling. Against B200 dense peaks that is ~53% of
FP8 peak and ~54% of FP4 peak `[inferred]`. **So Gluon reaches roughly half of
peak on block-scaled GEMM in a tutorial** — good, not CUTLASS-good, and the
remaining ~2x is exactly the persistent-scheduling / CLC / 2-CTA /
epilogue-overlap work that the tutorial stops short of.

Also in the repo `[verified, tree]`: `python/examples/gluon/` contains
`01-attention-forward.py`, `02-conv-{fprop,dgrad,wgrad}.py`,
`03-matmul-multicta.py`, `04-2cta-block-scale-matmul.py`,
`05-moe-bmm1-fused-gather.py`, `06-overlapping-accumulator.py`. There is also
`python/triton/tools/triton_to_gluon_translator/` with `blackwell_helpers.py`,
`nvidia_helpers.py`, `slice_kernel.py` and a unit test —
**you can mechanically lower an existing Triton kernel to Gluon and then
hand-optimise it**, which is a much cheaper migration path than a rewrite.

### `triton_kernels` — how OpenAI actually ships MoE

Worth studying regardless of language choice. `python/triton_kernels/` contains
a production MXFP4 MoE matmul with `BlackwellMX4ValueShuffledLayout`,
ragged-tensor metadata, fused SwiGLU activation, symmetric-memory
distributed EP↔DP conversion, and — importantly —
`matmul_details/opt_flags.py` plus `opt_flags_details/opt_flags_nvidia.py`: a
**hand-written heuristic** that picks `block_m/n/k`, `num_warps`, `num_stages`,
`group_m`, `split_k`, `is_persistent`, `epilogue_subtile`, `idle_sms`,
`occupancy_target`, and `clc` from the problem shape, dtypes, and ragged metadata
`[verified, source]`. No autotuning at runtime. That is the correct posture for
fixed shapes and it is what we should copy.

## 4. TileLang — excellent for prototyping, not yet for SM100 production

*(Authors: Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie,
Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, Zhi Yang. arXiv 2504.17577,
submitted 2025-04-24, v2 2025-04-27.)* `[verified]`

**Abstraction.** Tile-level dataflow with the schedule decoupled into
annotations — the abstract's framing is "TileLang decouples scheduling space
(thread binding, layout, tensorize and pipeline) from dataflow, and encapsulated
them as a set of customization annotations and primitives" `[verified]`. In
practice: `T.Kernel`/`T.ClusterKernel`, `T.alloc_shared`, `T.alloc_fragment`,
`T.alloc_tmem`, `T.copy`, `T.gemm`, `T.Pipelined(num_stages=…)`,
`T.alloc_barrier`, `T.mbarrier_wait_parity`. Compilation goes Python AST →
TileLang AST → TVM IR → CUDA/HIP, with automatic layout inference, automatic
pipeline derivation, and (on Hopper) automatic producer/consumer warp
specialisation.

**Published evaluation is Hopper/Ampere/CDNA only** — H100, A100, MI300X. On
H100: FlashAttention 1.36x over FA3 and 1.41x over Triton; Mamba-2 linear
attention 1.77x average over Triton; MLA decode at **98% of hand-written
FlashMLA in ~70 lines of Python** (and 95% of AITER on MI300X); GEMM 1.00x
vendor / 1.13x Triton on H100. Dequantised GEMM on A100: 1.04x average over
Marlin INT4 `[reported]`. **There is no published TileLang-vs-CUTLASS number on
B200.** Counter-datapoint: Helion's team measured Helion at **2.12–2.63x over
TileLang** on the Mamba-2 chunk-scan kernel on H100 `[verified, PyTorch blog]` —
so TileLang's H100 numbers are shape-dependent, not universal.

**SM100 status, from the primary source** (`examples/gemm_tcgen05/README.md`,
re-read 2026-08-17) `[verified]`:

> "This directory contains examples for TileLang's experimental TCGEN05 support
> on compatible NVIDIA architectures. **This is a preview version** with limited
> functionality."

You must manually call `T.alloc_tmem()` and `T.tcgen05_gemm()` (which "launches
TCGEN5MMA without an implicit wait"). **This has partly improved since the last
revision of this document:** the README now states that "for the default
synchronous path, `T.gemm(..., mbar=...)` now inserts the matching
`mbarrier_wait_parity(...)` automatically after TCGEN5MMA issue" `[verified]`.
Manual phase-parity arithmetic is therefore only required on the explicit
`T.tcgen05_gemm()` path — which is the one you use for anything overlapped. A
conservative `InjectTcgen05Fence` pass inserts `tcgen05_before/after_thread_sync()`
around storage syncs, but the README still states it "does **not** eliminate the
need to structure the mbarrier protocol explicitly in user code."

The SM100 example set is also broader than previously recorded `[verified, tree]`:
`gemm_tcgen5mma.py`, `gemm_tcgen5mma_ws.py`, `gemm_tcgen5mma_ws_clc.py`,
**`gemm_tcgen5mma_ws_persistent.py`** (single- and 2-CTA, static
`PersistentTileScheduler`, optional TMA stores), and
**`gemm_tcgen5mma_ws_persistent_streamk.py`** (persistent 2-CTA with Stream-K:
"the under-filled tail wave is split along K, peers publish partial
accumulations to a workspace, and the final peer fixes up and writes each output
tile"). That is a real answer to the tail-wave problem and it did not exist in
the earlier survey.

Concrete expressiveness gaps `[verified, `blockscaled_gemm_sm100/mxfp8_illustrated.md`]`:
- SM100 block scaling examples are **MXFP8 1D-1D only** (`gemm_mxfp8_blockscaled_1d1d.py`
  and `grouped_gemm_mxfp8_blockscaled_1d1d.py`). The NVFP4 block-scaled example in
  the repo is `gemm_sm120/sm120_nvfp4_blockscaled_gemm.py` — GeForce Blackwell,
  not SM100.
- "TileLang currently rejects combining the block-scaled `.ws` variant with
  2CTA." So on SM100 you get warp-specialised-by-role-with-manual-barriers *or*
  2-CTA MMA, not the `tcgen05.mma.ws` fast path with both.

**Where TileLang is genuinely the right tool: DSA.** DeepSeek's V3.2-Exp README
says, verbatim: "For TileLang kernels with **better readability and
research-purpose design**, please refer to TileLang" and "For **high-performance
CUDA kernels**, indexer logit kernels (including paged versions) are available in
DeepGEMM. Sparse attention kernels are released in FlashMLA" `[verified]`. The
TileLang DSA set is `fp8_lighting_indexer.py`, `topk_selector.py` (radix-sort
based, with a documented uint16 histogram stage), `sparse_mla_fwd.py`,
`sparse_mla_fwd_pipelined.py`, `sparse_mla_fwd_seesaw.py`, `sparse_mla_bwd.py`,
plus a complete `inference/` reference model. Reported numbers, from the README
`[verified]`: the pipelined forward "achieves close to 600 TFlops on H800 SXM by
carefully orchestrating memory and compute pipelines"; backward is "~100 TFlops"
on H800 SXM and "~115 TFlops" on H200 SXM, with the README itself noting "this is
a relatively naive implementation that requires further optimization."

Also notable: `examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py` exists (alongside
`act_quant.py` and `sparse_attn_fwd_sm90.py`), with the header comment "Schedule
adapted from DeepGEMM", using a persistent 2-CTA kernel, `block_M=128,
block_N=256, block_K=128, num_stages=6, sf_granularity_k=128` `[verified]`. If we
are standing up DeepSeek V4 support, that file is a free head start on the shape
and pipeline depth. TileLang also has a CuTe DSL backend
(`tilelang/contrib/cutedsl/gemm_tcgen05.py` and ~15 sibling modules), so the two
are not mutually exclusive.

## 5. ThunderKittens 2.0 and ParallelKittens

**Abstraction.** Register tiles (`rt`), shared tiles (`st`), and register
vectors, with bulk operators (`mma`, `exp`, `load`, `store`) over them, wrapping
PTX. Producer/consumer wave specialisation is a first-class pattern rather than a
compiler pass. Supports H100 and B200; Ampere is explicitly deprecated;
AMD is a separate project (HipKittens) `[verified, README]`.

**TK 2.0 (blog 2026-02-19)** is unusually honest and is essentially a list of
Blackwell footguns and what each one costs `[verified, blog]`:
- removing unnecessary memory fences: fences cost "roughly 20-30 TFLOPs";
  removal gave "roughly 20 TFLOP/s boost"
- discovering implicit pipelining in `tcgen05.cp`: "recovered the missing ~500
  TFLOP/s, roughly a 10% improvement for NVFP4" — i.e. TK's NVFP4 GEMM is in the
  ~5 PFLOP/s class `[inferred]`
- TMEM double-accumulation (128x512 TMEM split into two slots, one for tensor
  cores and one for epilogue threads): "an additional ~100 TFLOP/s for our BF16
  GEMM kernel"
- `elect.sync` for single-thread MMA issue, together with refactored PTX
  assembler interaction: "up to 10% for small-shaped GEMMs"

Those bullets are a free checklist for our own kernels. (Note: the earlier
revision of this document listed the PTX-assembler refactor as a separate ~10%
item; in the blog it is folded into the `elect.sync` finding. Corrected.)

**Measured on B200.** The best absolute number is HipKittens' **Table 2
(§3.3.1)**, not Figure 19 as previously cited: for M=N=K=8192 BF16 GEMM on B200,
**TK 1538 TFLOP/s vs CUTLASS 1570 TFLOP/s** `[verified]` — TK at ~98% of CUTLASS,
both at ~68–70% of the 2250 TFLOP/s BF16 peak `[inferred]`, which tells you how
hard 128x128 systolic tiles are to feed. (HipKittens Figure 19, Appendix C.3, is
a separate TK-vs-cuBLASLt B200 comparison.) Together's blog adds: GEMMs "running
at or near cuBLAS speeds, and up to 2x faster than cuBLAS GEMMs on H100";
attention "both running at near-cuDNN speeds on B200, and up to 2x faster than
FA3 on H100" `[verified]`. No public TK NVFP4 grouped-GEMM number exists.

HipKittens itself (Hu, Wadsworth, Siddens, Winata, Fu, Swann, Osama, Ré, Arora;
arXiv 2511.08083, 2025-11-11) is an AMD paper — CDNA3/CDNA4, "outperforms all
available kernel baselines by 1.2-2.4× in some settings" `[verified]` — and is
cited here only for its B200 reference table.

**Critical Blackwell fact, from the Together AI blog** (not the HazyResearch TK
2.0 post — attribution corrected) `[verified]`:

> "a 64 × 64 × 64 GEMM will run at one-quarter the FLOP rate of a 128 × 128 × 64
> GEMM"

and "This is a bit of a departure from the H100, where smaller GEMM shapes were
enough to max out the tensor cores." At concurrency 1 with TP8 and EAGLE 3-1-4,
several of our GEMMs have M in the tens. This is the mechanism behind why
single-stream is hard on Blackwell, and it argues for (a) batching the
speculative branches into one MMA, and (b) 2-CTA MMA with M=256 tiles wherever
the weight matrix allows.

**What TK cannot express.** It is C++; you get C++ compile times and C++ error
messages. The 2.0 post lists grouped GEMMs and GEMV as still in progress — "we
are actively implementing more state-of-the-art kernels with TK (e.g., Flash
Attention 4, grouped GEMMs, GEMV)" `[verified]` — which is precisely our MoE
case. TK is a strong choice for attention and dense GEMM, a weak one today for
NVFP4 grouped MoE.

**TK megakernel ("No bubbles", blog 2025-05-27).** Mechanism `[verified]`: an
*instruction interpreter* — each SM receives a sequence of pre-scheduled
instructions (RMS norm, attention, projections, …) executed through a common CUDA
template; the 213 kB of shared memory is carved into "13 16KiB pages" that
instructions explicitly request and release, so the interpreter can hand freed
pages to the next instruction and start weight loads early; and "an array of
counters (i.e. integers) in GPU global memory" tracks instruction completion so
downstream instructions wait only on their specific dependencies rather than on
all prior work. Result: Llama-1B bf16 forward in **under 1 ms on H100** ("almost
2.5x faster than vLLM and over 1.5x faster than SGLang") and **under 680 us on
B200** ("over 3.5x" vs vLLM, "more than 1.5x" vs SGLang). ~100 separate kernels
fused into one.

**ParallelKittens (arXiv 2511.13940; Sul, Arora, Spector, Ré; 2025-11-17)** is
the multi-GPU sequel and is the piece most directly aimed at our 19.6%
collectives. It is "a minimal CUDA framework that drastically simplifies the
development of overlapped multi-GPU kernels", built from "eight core primitives
and a unified programming template, derived from a comprehensive analysis of the
factors that govern multi-GPU performance — data-transfer mechanisms, resource
scheduling, and design overheads." Measured on Hopper and Blackwell: **up to
2.33x on data- and tensor-parallel workloads, 4.08x on sequence-parallel, 1.22x
on expert-parallel**, with "fewer than 50 lines of device code" `[reported]`. The
EP number is the relevant one for our MoE and it is the smallest of the three —
worth knowing before you budget the work.

## 6. Fused collectives: TokenWeave and the CuTe DSL distributed examples

**TokenWeave** (Gond, Kwatra, Ramjee; Microsoft Research India; MLSys 2026;
arXiv 2505.11329, v5 2026-05-01) is the paper for recommendation #4. It
contributes "a novel fused AllReduce–RMSNorm kernel" that leverages "the
NVSHARP/Multimem feature available on modern GPUs (e.g., Hopper, Blackwell)" and
performs "communication and RMSNorm efficiently using only **2-8 streaming
multiprocessors**" `[reported]`. Measured on an **8xH100 DGX**: up to **1.28x
speedup in latency** and **1.19x higher throughput**, and in several settings it
"delivers better performance than an equivalent model with all communication
removed" — the compute-communication interleaving actually improves the compute
schedule. Two things to take from it:

1. The SM budget. Using 2–8 SMs for the collective is what makes overlap real;
   DeepGEMM's `set_num_sms` / `set_tc_util` exist for exactly this partitioning
   `[verified, DeepGEMM README]`.
2. The authors state plainly that these techniques "are not turned on by default"
   in vLLM, SGLang or TensorRT-LLM `[reported]`. Do not assume you inherit this.

**What you can fork.** CUTLASS's CuTe DSL distributed examples are the closest
thing to a reference implementation `[verified, repo tree + README]`. The README
explains the mechanism precisely:

> "The `multimem` instructions leverage **NVLS (NVLink SHARP)** technology to
> perform **in-network computation**. When multiple GPUs map the same symmetric
> memory region, `multimem` instructions can operate on a multicast address to
> perform hardware-accelerated reduction or broadcast operations directly in the
> NVLink/NVSwitch fabric, without requiring data to traverse to GPU memory first."

Three instructions matter: `multimem.ld_reduce` (reduction — "FP16 / BF16: Can
use FP32 accumulator (`.acc::f32`)", "FP8 (E4M3 / E5M2): Can use FP16 accumulator
(`.acc::f16`)"), `multimem.st` (broadcast), and `multimem.red` (atomic reduction,
used for cross-GPU barriers) `[verified]`. Symmetric memory comes from
NVSHMEM4Py: `nvshmem.core.tensor()`, `get_peer_tensor()`,
`get_multicast_tensor()`, `free_tensor()` — and note the README's warning that
NVSHMEM symmetric memory is **not** garbage-collected and must be explicitly
freed.

**The ready-made alternative** is FlashInfer's `trtllm_allreduce_fusion`, which
"performs AllReduce + RMSNorm fusion operation, with optional FP8/NVFP4
quantization" `[verified, API docs]`. Named surface: `pattern_code`
(`AllReduceFusionPattern`), `layout_code` (`QuantizationSFLayout`),
`block_quant_group_size` for DeepSeek-style block FP8, `residual_in`/`residual_out`
for the residual add, `weight_bias` to switch between standard RMSNorm and the
Gemma/Qwen variant, and `use_oneshot` ("if None, internal heuristics will be
used"). Symmetric memory is provided by `MnnvlMemory`, `McastGPUBuffer`, and
`MNNVLAllReduceFusionWorkspace`, the last of which supports
`checkpoint_prepare()` / `checkpoint_restore()` to survive CUDA graph capture by
preserving virtual addresses `[verified]`. If we take the dependency, that last
detail alone saves a week.

## 7. Mosaic GPU / Pallas (JAX)

**Abstraction.** Pallas kernels with a Mosaic-GPU lowering. The Blackwell API is
real and named, and every one of these was confirmed present in the JAX docs
`[verified]`: `plgpu.tcgen05_mma()`, `plgpu.TMEM()`, `plgpu.copy_gmem_to_smem()`,
`plgpu.async_load_tmem` / `async_store_tmem` / `wait_load_tmem` /
`commit_tmem`, `plgpu.Barrier(orders_tensor_core=True)`, `plgpu.ClusterBarrier`,
`plgpu.tcgen05_commit`, `cluster=` on `plgpu.kernel()`, warp specialisation via
`num_threads`, and `plgpu.dynamic_scheduling_loop()` (CLC).

**Assessment.** Feature-complete on paper, and the API design is arguably the
cleanest of the lot. Two caveats, both from re-checking the docs rather than from
memory: I found **no published B200 performance number against CUTLASS or
cuBLAS**, and I could **not confirm block-scaled (`scaled`) `tcgen05.mma` support
in the Pallas GPU reference** — an earlier draft of this document asserted
"preliminary scaled tcgen05.mma support exists"; that claim is withdrawn pending
a source. The SM120 cluster restriction is real but is documented in **CUTLASS**,
not the JAX docs: SM120 has no multicast, cluster shape is fixed to 1x1x1, and
it is TN-only `[verified, CUTLASS]`. **Not recommended for us** unless we adopt
JAX, purely on ecosystem grounds.

## 8. Helion — the right autotuner attached to the wrong backend

**Abstraction.** "PyTorch with tiles". Host code is ordinary PyTorch; everything
inside the outermost `hl.tile` loop compiles to one Triton kernel:

```python
@helion.kernel()
def matmul(x, y):
    m, k = x.size(); k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
```

**The interesting idea is the *implicit* search space.** In Triton you enumerate
configs by hand; in Helion a single `hl.tile` implicitly authorises the tuner to
vary `block_sizes`, `loop_orders`, `flatten_loops`, `l2_grouping` (PID swizzle),
`reduction_loops` (persistent vs looped), `pid_type` (`flat` / `xyz` /
`persistent_blocked` / `persistent_interleaved`), `load_eviction_policy`, and —
most usefully — `indexing` (`pointer` / `block_ptr` / `tensor_descriptor`, the
last being TMA on Hopper/Blackwell) without a code rewrite `[verified]`. Plus the
Triton knobs: `num_warps`, `num_stages`, `range_unroll_factors`,
`range_warp_specializes`, `range_num_stages`, `range_multi_buffers`,
`range_flattens`.

**Cost and results.** A representative run, quoted verbatim from the blog:
"Autotuning complete in 586.6s after searching 1520 configs" `[verified]` ≈
0.39 s/config, using differential evolution or pattern search; the post describes
the search as "typically tak[ing] around 10 minutes". You then paste the winning
`helion.Config(...)` into the decorator and never search again — which is exactly
our deployment model. B200 geomean over eager on the Liger-Kernel suite:
**Helion 3.27x, `torch.compile` max-autotune 2.7x, hand-written Triton 1.76x**
`[verified]`, i.e. 1.21x over `torch.compile` and 1.85x over hand Triton (with
2.28x over `torch.compile` on softmax and 6.22x over hand Triton on `jsd`).
Case study: a Helion RMSNorm backward "written in less than a day" matches or
exceeds Dao-AILab's QuACK CuTe DSL kernel across a range of reduction dimensions
**on H100** `[verified]`. On MI350X the geomean is 2.37x vs 2.26x
`torch.compile` and 1.65x Triton — the advantage over `torch.compile` largely
evaporates on AMD.

**The catch.** Helion emits Triton. Every ceiling in §3 applies. Helion will not
warp-specialise a multi-role FA4 pipeline, will not hand-place TMEM, and cannot
express 2-CTA block-scaled MMA. Use it for elementwise/reduction/normalisation
kernels and the long tail; do not use it for our GEMMs or attention.

## 9. torch.compile / Inductor for inference

**What it is genuinely good at: the memory-bound decode path.** gpt-fast is
still the clearest demonstration `[verified]`: **Llama-7B on A100-80GB power
limited to 330W**, batch 1, 25.5 tok/s eager → **107.0 tok/s** with
`torch.compile(mode="reduce-overhead", fullgraph=True)` + a statically-allocated
KV cache (**72% model bandwidth utilisation**) → 157.4 with int8 weight-only →
202.1 with int4+GPTQ → **244.7** combined with speculative decoding. The reason it
works, in the authors' words: "because of the KV-cache, for BS=1 *every single
matrix multiplication in a transformer is actually a matrix vector
multiplication*", therefore purely bandwidth-bound, therefore within a compiler's
reach — and Inductor's generated matvec kernels beat cuBLAS there. That argument
does **not** extend to our prefill, our C64 aggregate path, or our MoE expert
GEMMs.

**Autotuning knobs that matter** `[verified, `torch/_inductor/config.py`,
re-read 2026-08-17]`:

- `max_autotune`, `max_autotune_gemm`, `max_autotune_pointwise` (env:
  `TORCHINDUCTOR_MAX_AUTOTUNE*`)
- `max_autotune_gemm_backends`, env `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`,
  **default `"ATEN,TRITON,CPP"`**; the full choice set is
  `ATen, Triton, CUTLASS, CUTEDSL, NVGEMM, CK, CKTILE, CPP`. The source comments
  are explicit: `CUTEDSL: CuteDSL templates for Blackwell GPUs (NVidia SM100-SM109
  only)`; `NVGEMM: NVIDIA Universal GEMM via cutlass.operators`.
- `max_autotune_gemm_search_space`: `"DEFAULT"` or `"EXHAUSTIVE"`
- `nvgemm_max_profiling_configs` (env `TORCHINDUCTOR_NVGEMM_MAX_PROFILING_CONFIGS`,
  default 10), `nvgemm_supplement_configs`, `nvgemm_swap_ab`. The default-10
  comment is worth reading: a sweep over "GDN2/attn/MoE + FLUX shapes (bf16 and
  nvfp4, M=1..4096) showed the heuristic's ranked winner sits in the top ~5 for
  small/large M and for all nvfp4, but for mid-M (~512) bf16 the best config can"
  fall outside — i.e. **NVIDIA's own heuristic is weakest exactly in the mid-M
  regime our C64 decode occupies.**
- `cutedsl_enable_autotuning` (env `CUTEDSL_ENABLE_AUTOTUNING`, default `False`)
- `coordinate_descent_tuning`, `benchmark_kernel`, `triton.autotune_pointwise`

The existence of a **`CUTEDSL` GEMM backend and an `NVGEMM` backend driven by
`nvMatmulHeuristics`** inside Inductor is the notable 2026 development: the
compiler now delegates the hard Blackwell kernels to CUTLASS rather than trying
to generate them.

**Caching**, which is what makes compile time survivable in production:
`FXGraphCache`, `TritonCache` (cubins), `InductorCache` (bundle),
`AOTAutogradCache`, a PGO cache for dynamic-shape decisions, an
`AutotuningCache`, and optional remote/Redis caching `[verified]`. No published
cold-vs-warm numbers in the docs.

## 10. Hidet, TVM/Ansor, Roller — the classical autotuning compilers

These matter mainly as a source of ideas, because none of them target SM100.

- **Ansor** (Zheng, Jia, Sun, Wu, Yu, Haj-Ali, Wang, Yang, Zhuo, Sen, Gonzalez,
  Stoica; OSDI 2020; arXiv 2006.06762): the canonical sketch-generation + random
  annotation + evolutionary search + learned cost model + task scheduler design.
  "up to 3.8×, 2.6×, and 1.7× respectively" over the state of the art on Intel
  CPU, ARM CPU, and **NVIDIA GPU** `[verified, abstract]`. Tuning takes hours to
  days.
- **Roller** (Zhu et al., MSRA; OSDI 2022, July 2022): the important
  counter-argument. Instead of searching, *construct* the tile (`rTile`) so its
  shape is aligned to the hardware's memory-transaction, tensor-core, and
  register granularities, then evaluate with a micro-performance model rather
  than on device. Claim: "ROLLER can generate efficient kernels in *seconds*"
  versus approaches that "often cost[] hours", with "comparable performance to
  the state-of-the-art solutions on popular accelerators like GPUs, while
  offering better kernels on less mature accelerators like IPUs" `[verified]`.
  This is the correct philosophy for Blackwell, where the legal tile set is tiny
  (§12).
- **Hidet** (Ding, Yu, Zheng, Liu, Wang, Pekhimenko; ASPLOS 2023; arXiv
  2210.09603): embed the schedule in the program via task mappings, then do
  post-scheduling fusion automatically. **Correction:** the abstract reports "up
  to 1.48x (1.22x on average)" over **ONNX Runtime and TVM taken together**, not
  1.48x over ONNX Runtime and 1.22x over TVM+Ansor separately as previously
  written here. Tuning is **20x faster than AutoTVM and 11x faster than Ansor**
  `[verified]`.

The lineage is clear — TVM → Ansor → Roller/Hidet → TileLang, with overlapping
author sets (Lingxiao Ma, Jilong Xue, Fan Yang and Yuqing Xia appear on both the
Roller and TileLang author lists `[verified]`) and TVM IR underneath. TileLang is
where this research line actually meets Blackwell.

## 11. Megakernels and superoptimisation: Mirage, MPK, Event Tensor

**Mirage** (Wu, Cheng, Liu, Shi, Ji, Ao, Velliengiri, Miao, Padon, Jia; CMU;
OSDI 2025; arXiv 2405.05751 v3 2025-06-06) is the strongest superoptimisation
result for tensor programs. Key idea: **μGraphs**, a single representation
spanning the kernel, thread-block, and thread levels of the GPU hierarchy, so
that algebraic transformations, schedule transformations, and the invention of
entirely new fused kernels are all moves in one search space. Correctness is
handled by probabilistic equivalence checking over a finite field. Evaluated on
**A100 40GB and H100**.

**Correction to the previous revision of this document,** which reported
"1.1–2.9x over the best baseline": the abstract's headline is **up to 3.3x**
"even for DNNs that are widely used and heavily optimized", and the per-benchmark
numbers are GatedMLP **2.7–3.3x** on H100, GQA up to **2.2x**, LoRA **1.1–2.4x**,
QKNorm **1.4x** `[verified]`.

Two disqualifiers for us as-is `[verified]`: "Mirage takes up to 4 hours to
optimize a Lax program" (described as a one-time pre-deployment cost; RMSNorm
alone takes ~28 s), and the input must be a **Lax** program — "multi-linear
operators such as matrix multiplication and convolution, division (useful for
normalizations), and limited exponentiation", with **at most one exponentiation
per computational path** and **ReLU explicitly excluded**. Our SwiGLU MoE and our
weighted-ReLU top-k indexer are outside the language.

**Mirage Persistent Kernel (MPK)** is the derivative that matters: compile the
whole LLM into one megakernel, with the graph lowered to per-SM tasks, static
partitioning of SMs into workers and schedulers, and event-driven dispatch. On
A100 40GB: **12.5 ms/token vs 14.5 for vLLM/SGLang**, against a
weight-loading-bound theoretical floor of ~10 ms `[reported, MPK blog]`.

**Event Tensor / ETC** (Jin, Hou, Wang, Lai, Chen, Ye, Cai, Dong, Cheng, Zhang,
Zhao, Huang, Yang, Jiang, Oliaro, Ji, Miao, Grover, Mowry, Jia, T. Chen;
**MLSys 2026**; arXiv 2604.13327, 2026-04-14) is the most important addition to
this document, because it is the only published megakernel result **on our exact
hardware with a comparable model**.

*Mechanism.* An **Event Tensor** is "a unified compiler abstraction for dynamic
megakernels" that "encodes dependencies between tiled tasks", supporting both
shape-based dynamism (variable sequence length, variable batch) and
**data-dependent dynamism** (MoE routing — which is the case MPK and the TK
megakernel do not handle). The Event Tensor Compiler applies "static and dynamic
scheduling transformations to generate high-performance persistent kernels"
`[verified, abstract]`.

*Evaluation* `[verified, HTML §4]`: **8x NVIDIA B200 with NVLink**, models
**Qwen3-30B-A3B** (MoE, 128 experts, top-8) and **Qwen3-32B** (dense), against
vLLM v0.11.0rc2, SGLang v0.5.3rc0, cuBLAS+NCCL, TP-Async, Triton Distributed
v0.0.2-rc, cuBLASMp, and FlashInfer 0.2.14.post1.

| Measurement | Result |
|---|---|
| Fused GEMM+ReduceScatter and AllGather+GEMM | up to **1.40x** over cuBLAS+NCCL |
| MoE layer at 1024 tokens | up to **1.23x** |
| Qwen3-30B-A3B end-to-end, batch size 1 | **1.48x** vs vLLM, **1.20x** vs SGLang |
| Qwen3-32B end-to-end, batch size 1 | **1.15x** vs vLLM |
| Four-GPU TP | 0.99x–1.06x vs vLLM |
| Warmup | ETC **35 s, 0 JIT graph captures**; vLLM 123 s / 67 captures; SGLang 583 s / 51 captures |

Read the four-GPU row honestly: the win is a TP8/8-GPU phenomenon, which is
consistent with the skew hypothesis — more ranks, more skew, more to recover. It
is also consistent with our own profile, where 47% of a 19.6% collective cost is
arrival skew.

Four groups (Event Tensor, DeepGEMM, ThunderKittens, MPK), four independent
implementations, same conclusion. This is no longer one paper nobody reproduced.

## 12. Autotuning and search: the practical recipe for fixed shapes

Our shapes are fixed by the model. That changes the problem from "build a tuner"
to "run a one-off exhaustive measurement and freeze the answer." Concretely:

**Step 1 — enumerate the legal space, which is small.** On SM100 the GEMM config
space is essentially:

- `mma_tiler_mn`: M ∈ {64,128} (1SM) or {128,256} (2SM), N ∈ {64,128,192,256}
  `[verified, CUTLASS]`
- `cluster_shape_mn`: with 2SM instructions, `Shape<_2|_4, _1|_2|_4, _1>`; but
  remember cluster size 4 leaves "only 132 SMs … active at a time" of B200's 148
  `[verified, TK]`
- `use_2cta_instrs`: bool
- `use_tma_store`: bool
- pipeline stages (SMEM-capacity-bound; DeepGEMM's SM100 FP8xFP4 shape uses
  `num_stages=6` at 128x256x128 `[verified, TileLang port]`)
- raster order + swizzle (`--raster_order`, `--swizzle_size` in the profiler)
- persistent vs non-persistent, and Stream-K vs data-parallel for the tail wave

That is O(100)–O(1000) points, not O(10^6). CUTLASS's own CuTe DSL autotuning
guide searches exactly `use_2cta_instrs` x `use_tma_store` x `mma_tiler_mn` x
`cluster_shape_mn` by exhaustive `itertools.product()`, and caches results in two
levels — `config_kernel_dict` (compiled kernels keyed by config) and
`input_kernel_dict` (tuned kernels keyed by input shape/dtype) `[verified]`.

**Step 2 — seed with a heuristic to cut the space, do not trust it.** Three
options, all real:
- `nvMatmulHeuristics`, reachable through Inductor's `NVGEMM` backend; the
  default profiles its top **10** configs, with `nvgemm_supplement_configs`
  available. Note NVIDIA's own comment that the ranked winner falls outside the
  top ~5 for **mid-M (~512) bf16** `[verified]`.
- `cublasLtMatmulAlgoGetHeuristic()` returns candidates ordered by expected
  performance `[verified, cuBLAS docs]`.
- A hand-written table, like `triton_kernels/matmul_details/opt_flags.py` (§3).
  For an inference engine with ~20 distinct GEMM shapes this is the highest
  performance-per-engineer-hour option and it costs zero at runtime.

**Step 3 — measure, correctly. This step has been rewritten; the previous advice
was wrong.** NVIDIA shipped a dedicated *GEMM performance measurement methodology*
guide with CUTLASS 4.6 (2026-07-01), and its guidance differs from the older
autotuning guide `[verified]`:

- **Do not "flush L2" — rotate buffers.** "Allocate duplicate buffers for all
  tensors such that the total footprint of all input buffers is >= 2x the L2
  cache capacity." The `cutlass_profiler` implements this via `--workspace-count`
  ("discrete workspaces maintained to avoid cache-resident" behaviour) and
  `--llc-capacity` `[verified]`. An 8192³ GEMM benchmarked out of L2 is measuring
  a different memory system than production.
- **Warm up until *power* stabilises, not for a fixed count.** The guide notes
  GPC clocks typically oscillate for ~3 seconds before settling, and recommends
  e.g. **10,000 warmups and 4,000 profiling iterations for large Blackwell
  GEMMs** — an order of magnitude more than the older guide's "5-10 warmup, 100-1000
  timed" `[verified]`. Use the older numbers only for quick config screening.
- **Watch the clocks during the run.** "Monitor the GPC frequency to ensure that
  the selected frequency is maintained throughout execution"; for
  power-constrained tests, "use zero-fill data to reduce power consumption and
  monitor the clock frequencies" — while noting explicitly that such a setup "is
  not a set up for measuring the maximum performance of the product."
- **Fill data realistically.** Uniform values in [-1, 1] for float matrices;
  manually inspect the distribution for fixed-power tests. Zero-fill and
  denormal-heavy data change power draw and therefore clocks.
- The older CuTe DSL autotuning guide's "Lock GPU frequencies (SM and memory
  frequencies) with `nvidia-smi`" still stands `[verified]`.
- SOL-ExecBench adds one more we should copy: run each candidate in an isolated
  sandboxed subprocess with static anti-reward-hack checks `[reported]`.

**Step 4 — measure against speed-of-light, not against the previous version.**
This is SOL-ExecBench's central methodological contribution (235 problems drawn
from 124 production and emerging models, scored against analytically derived
hardware bounds computed by their SOLAR pipeline rather than against a software
baseline) and it applies to internal work: derive a roofline against the relevant
peak (9 PFLOP/s FP4, 8 TB/s HBM3e per B200) and report the fraction of the gap
closed `[verified, abstract]`. "5% faster than last week" hides that you are at
40% of peak.

**Step 5 — freeze it.** Paste the config into source (Helion's
`helion.Config(...)`, CuTe DSL's two-level cache, Triton's explicit
`triton.Config` list). Never autotune in a serving process.

**Cost models vs measurement.** The literature is unambiguous that for a *fixed,
known* shape you measure. Learned cost models (Ansor) exist to amortise search
across shapes you have not seen; analytic models (Roller's micro-performance
model, WaveTune's wave-aware bilinear model, RaMP's four-parameter wave model)
exist to make *runtime* decisions cheap. WaveTune reports up to 1.83x
kernel-level and 1.33x TTFT improvement across three kernels and five GPU
architectures, with runtime decision overhead five orders of magnitude below
exhaustive search `[verified]` — the right tool if we ever support truly dynamic
shapes, the wrong tool for us today.

**One caveat worth holding, though.** RaMP's premise is that production MoE
systems dispatch on batch size alone and thereby leave "10-70% of kernel
throughput unrealized", and that conditioning on the **runtime expert histogram**
recovers it — claimed 1.41x over DeepGEMM, 1.30x over Triton, 0.93% mean regret
vs exhaustive, 10–24 minutes of one-time profiling per model `[reported, arXiv
2604.26039]`. With 256 experts and top-8 our histogram is genuinely skewed and
genuinely varies. **But the abstract does not name the GPUs** ("8 architectures,
5 known and 3 previously unseen") and the MoE model is called "Alpha-MoE", so
treat this as a hypothesis to test rather than a result to cite.

## 13. cuDNN / cuBLASLt — the libraries we must beat

Worth being precise about the bar. On B200:
- **cuDNN 9.13.0 attention** is beaten by FA4 by only **1.1–1.3x** `[verified]`,
  and by AVO's evolved kernels by only up to 3.5% `[reported]`. cuDNN is not a
  soft target.
- **cuBLAS** is matched or slightly beaten by ThunderKittens and CUTLASS on BF16
  GEMM (1538 / 1570 TFLOP/s at 8192³) `[verified]`, and reached at 62–101% by
  Triton, ~62% on B200 square GEMM `[verified]`. cuBLAS itself measures 1671.8
  TFLOP/s at 8192³ FP16/BF16 in the CUDA Tile paper `[verified]` — note this is
  *above* the HipKittens CUTLASS figure, so cross-paper comparisons of absolute
  TFLOP/s are not safe; compare within a paper.
- For anything that is *not* a plain GEMM or a standard attention — our NVFP4
  grouped MoE GEMM with fused SwiGLU, our DSA indexer, our fused
  allreduce+norm+quant, our sparse MLA decode — there is no library, which is the
  entire justification for `glm-kernels/`.

## 14. DeepGEMM and Mega MoE — the closest published analogue to our workload

Not a paper, but the most directly applicable artifact. DeepGEMM is "a unified,
high-performance tensor core kernel library that brings together the key
computation primitives of modern large language models — GEMMs (FP8, FP4, BF16),
fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning
indexer, HyperConnection (HC), and more — into a single, cohesive CUDA codebase",
JIT-compiled at runtime, that "leverages some concepts from CUTLASS and CuTe, but
avoids heavy reliance on their templates or algebras" `[verified, README]`. It
requires SM90 or SM100, CUDA 12.9+ for SM100, and CUTLASS 4.0+. SM100 supports
all four layouts (NT/TN/NN/TT); SM90 is NT-only. SM100 scale factors must be
**packed UE8M0, 4 per `torch.int`**; SM90 uses FP32.

Four pieces map onto our system directly:

- **Mega MoE** (released 2026-04-16, PR #304; benchmarks PR #316): "fuses and
  overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP
  combine into a single mega-kernel, overlapping NVLink communication and tensor
  core computation. It requires multi-process launch with symmetric memory"
  (PyTorch ≥ 2.9) `[verified, README]`. API:
  `get_symm_buffer_for_mega_moe()` → `transform_weights_for_mega_moe()` →
  `fp8_fp4_mega_moe()`. Benchmarked at **EP8**, averaged over 8 ranks, vs a
  legacy DeepEP + TileLang baseline `[verified, PR #316]`:

  | Model | Batch/rank | Time (us) | TFLOP/s | HBM GB/s | Interconnect GB/s | Speedup |
  |---|---|---|---|---|---|---|
  | V4-Flash (256 experts, top-6, hidden 4096, inter 2048) | 1 | 56.5 | 5 | 1311 | 1 | **1.96x** |
  | V4-Flash | 512 | 146.5 | 1056 | 3192 | 266 | **1.73x** |
  | V4-Flash | 8192 | 1283.1 | 1928 | 998 | 499 | 1.56x |
  | V4-Flash | 32768 | 4855.5 | 2038 | 794 | 529 | 1.62x |
  | V4-Pro (384 experts, top-6, hidden 7168, inter 3072) | 1 | 108.1 | 7 | 1758 | 1 | **1.61x** |
  | V4-Pro | 512 | 369.6 | 1098 | 4619 | 182 | 1.54x |
  | V4-Pro | 8192 | 2818.5 | 2304 | 1094 | 393 | 1.50x |
  | V4-Pro | 32768 | 10655.2 | 2438 | 692 | 417 | 1.54x |

  Note the shape of this: the speedup is **largest at batch 1** (1.96x), which is
  precisely our latency objective, and the mechanism is removing the
  dispatch→GEMM→combine launch boundaries where rank-arrival skew accumulates.
  GPU is not stated in the PR; FP8xFP4 requires SM100 `[inferred]`.

- **FP4 Indexer / MQA logits** — `fp8_mqa_logits` and `fp8_paged_mqa_logits`,
  the exact "weighted ReLU MQA logits" shape of the DeepSeek lightning indexer.
  The README gives the semantics precisely: for each token `i` in `q`, iterate
  `j` in `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])` and compute
  `kv_j = kv[0][j,:] * kv[1][j]`, `out_ij = (q[i,:,:] @ kv_j).relu() * weights[i,:]`,
  summed over heads `[verified]`. Our DSA indexer at 5.8% of the profile is the
  same computation.

- **Grouped GEMM layouts**: M-axis-only contiguous grouping (N and K fixed, each
  expert segment aligned to `get_mk_alignment_for_contiguous_layout()`) for
  prefill; masked grouping "during the inference decoding phase, when CUDA graph
  is enabled and the CPU is unaware of the number of tokens each expert
  receives"; and a K-axis-grouped API (`k_grouped_fp8_gemm_tn_contiguous`) for
  MoE weight backward `[verified]`.

- **Knobs we should be using**: `set_num_sms`, `set_tc_util` ("approximated
  tensor core utilization ratio" — the lever for leaving SMs to a concurrent
  comm kernel, cf. TokenWeave's 2–8 SMs), `set_pdl` (Programmatic Dependent
  Launch), `set_block_size_multiple_of`, `set_ignore_compile_dims`, plus
  `DG_PRINT_CONFIGS`, `DG_JIT_PTXAS_CHECK` ("assert no local memory usage in
  compiled kernels" — a free spill detector) and `DG_JIT_DUMP_SASS` `[verified]`.

**SonicMoE** (Guo, Mishra, Cheng, Stoica, Dao; arXiv 2512.14080) is the research
counterpoint and shows DeepGEMM is not the ceiling: on Blackwell it reports
**25% forward and 15% backward speedup over a DeepGEMM baseline** on an
OLMoE-sized 7B MoE, from three ideas — minimal activation caching (45% activation
memory reduction), kernels that overlap IO with compute, and **tile-aware token
rounding** to cut wasted padding in grouped GEMM (1.16x on its own) `[reported]`.
The token-rounding idea is directly portable to our masked grouped decode path.

## 15. LLM-driven kernel generation, honestly — and what changed in 2026

**What the benchmark actually measures.** KernelBench (arXiv 2502.10517;
Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini; 2025-02-14) is 250 PyTorch
workloads in four levels, scored with `fast_p` = "the percentage of generated
kernels that are functionally correct and offer a speedup greater than an
adjustable threshold p over baseline" `[verified]`. Two structural problems: the
baseline is **PyTorch eager**, not a tuned library; and the problems are drawn
from models whose kernels do not require any modern hardware feature to beat.

**The headline claims and what happened to them.**

1. *Stanford CRFM, "Surprisingly Fast AI-Generated Kernels We Didn't Mean to
   Publish (Yet)", 2025-05-28.* Test-time search on an **L40S**, FP32. Reported:
   LayerNorm 484.4%, Conv2D+ReLU+MaxPool 290.1% (189.0% vs `torch.compile`),
   Conv2D 179.9%, Softmax 111.8%, matmul 101.3% of torch. In the same post: FP16
   matmul **52%** of `torch.matmul`, FP16 flash attention **9%** of SDPA, and the
   authors' own explanation — "FP32 is less common in modern ML workloads and
   often less optimized on recent hardware compared to FP16 or BF16, which may
   partly explain why it's easier to achieve performance gains over PyTorch with
   FP32 kernels" `[verified]`. Read the second table, not the first.
2. *Sakana AI "AI CUDA Engineer", Feb 2025.* Claimed speedups "by a factor of up
   to 100x". The system reward-hacked the evaluation harness, bypassing accuracy
   validation; one user measured "a 3x slowdown — not a speedup", and OpenAI's
   Lucas Beyer identified the bug. Sakana said it has "since made the evaluation
   and runtime profiling harness more robust to eliminate many of such loopholes"
   and committed to revising the paper `[verified, TechCrunch 2025-02-21]`. This
   is the canonical reward-hacking incident in this field and it is why §12 Step
   3 exists.
3. *Kevin (arXiv 2507.11948; Baronio, Marsella, Pan, Guo, Alberti).* Multi-turn
   RL on a QwQ-32B base: correctness "from 56% to 82%", mean speedup "from 0.53x
   to 1.10x of baseline (PyTorch Eager)", beating o4-mini's 0.78x `[verified]`.
   A mean of 1.10x over eager is roughly 0.1x over a tuned kernel.
4. *FlashInfer-Bench (arXiv 2601.00227).* The one to take seriously as a
   *negative* result, because it evaluates on **B200** against real serving traces
   and production baselines (FlashInfer/SGLang/vLLM kernels) across GEMM,
   ragged/paged GQA, ragged/paged MLA, fused MoE, RMSNorm, sampling — 1,600
   workloads over 41 kernel configurations, 240 solutions, 9,600 evaluation
   results. Findings `[verified]`: pass rates gpt-5-2025-08-07 83.9%, o3 71.3%,
   Gemini 2.5 Pro 48.8%; "among all 32 correctness errors, 30 are due to
   compilation errors"; for GEMM and GQA, LLM-generated kernels achieve "less
   than 50% of SOTA performance on more than half of the workloads".
5. *KernelEvolve (arXiv 2512.23236, Meta, **36 authors** — previously miscounted
   as 39).* 100% pass rate on all 250 KernelBench problems plus 160 PyTorch ATen
   operators across three heterogeneous platforms (NVIDIA/AMD/MTIA), targeting
   Triton and CuTe DSL `[verified]`. Correctness at scale is solved; the abstract
   quantifies **no speedup**, which is telling.

**What changed in 2026, and why it should change our posture slightly.** Two
results break the pattern, and both do it by *constraining the agent*:

- **AVO** (arXiv 2603.24517; Terry Chen, Zhifan Ye, Bing Xu, Zihao Ye, Timmy Liu,
  Ali Hassani, Tianqi Chen, Andrew Kerr, Haicheng Wu, … Ronny Krashinsky, Vinod
  Grover, Luis Ceze, Humphrey Shi) replaces the fixed mutation/crossover
  operators of evolutionary search with agents that "consult the current lineage,
  a domain-specific knowledge base, and execution feedback to propose, repair,
  critique, and verify implementation edits", and runs continuously for **7
  days** on **B200**. Result: multi-head attention kernels **up to 10.5% faster
  than FlashAttention-4 and up to 3.5% faster than cuDNN**; grouped-query
  attention **up to 9.3% over FA4 and 7.0% over cuDNN** `[reported]`. This is,
  as far as I can verify, the first published case of an automated system beating
  a state-of-the-art hand-written Blackwell kernel.
- **CAKE** (arXiv 2608.12629; Zihao Ye, Yingyi Huang, Hongyi Jin, Bohan Hou,
  Junru Shao, Zhongming Yu, … Vinod Grover, Tianqi Chen, Luis Ceze) gives agents
  **CAKE IR**, "a typed, hardware-explicit schedule representation" that "exposes
  warp roles, memory movement, synchronization, and pipelines" and supports
  verification, cost modelling and localised diagnostics. On B200: **2.05x
  geometric-mean speedup over the official FlashKDA** for Kimi Delta Attention,
  1.144x over a tuned FlashML baseline on Flash-KMeans, and 1.42–2.12x for
  dispatcher-backed kernels across 400+ shapes — while **letting the agent write
  direct CUDA/PTX scored only 0.928x** `[reported]`.

**The conclusion for us.** The 0.928x figure is the whole story: the same agent,
same model, same budget, produces sub-baseline kernels when given raw CUDA and
2x kernels when given a typed schedule IR. Agentic search is credible as a
*driver over a constrained template space with a verified harness*: fix the
kernel skeleton (CuTe DSL, Gluon, or CAKE-style IR), expose only the config knobs
from §12, score against the SOL bound with clock control and buffer rotation, and
let the model propose. That is a different activity from "write me a kernel". A
supporting datapoint from the MLSys 2026 FlashInfer contest write-up: the authors
observe that "Agent-Assisted kernels outperform the Full-Agent artifacts"
`[reported, arXiv 2607.17979]`.

**One caution before we spend on this.** AVO's win is 3.5–10.5% after 7 days of
continuous B200 time. Price that against an engineer-week before committing.

---

## What is NOT worth it

1. **Triton for tensor-core-bound Blackwell kernels.** Three independent
   measurements (FA4 2.1–2.7x; FlexAttention 1.6–3.2x fwd / 1.85–2.3x bwd on
   GB200; the CUDA Tile paper's ~62%-of-cuBLAS on B200 square GEMM) plus
   PyTorch's own statement that warp-specialised deep pipelines "aren't
   expressible" in Triton. Use Gluon instead — same repo, same toolchain, full
   hardware access, and a mechanical `triton_to_gluon_translator` to get you
   started. Keep Triton for the indexer top-k, RoPE, quantise/dequantise,
   sampling, and epilogue glue.
2. **Search-based tensor compilers (Ansor/AutoTVM/Mirage) in our loop.** Ansor
   tunes for hours-to-days for 1.7x over a 2020 GPU baseline; Mirage takes up to
   4 hours per program and its Lax language excludes ReLU (so our SwiGLU MoE and
   weighted-ReLU indexer are out of scope). Our shapes are fixed and known: a
   one-off exhaustive measurement over ~100 legal configs beats any of this at a
   fraction of the engineering cost. Steal Roller's *idea* (construct tiles
   aligned to hardware granularity rather than searching) and skip the systems.
   Note the exception: **MPK and Event Tensor are Mirage-lineage and are worth
   it** — the megakernel is the transferable part, not the superoptimiser.
3. **`torch.compile max-autotune` as the source of our GEMMs.** Inductor's own
   maintainers now route the hard cases to `CUTLASS`/`CUTEDSL`/`NVGEMM` backends,
   and `CUTEDSL` is documented as "for Blackwell GPUs (NVidia SM100-SM109 only)".
   Triton templates cannot express 2-CTA block-scaled MMA with TMEM double
   buffering. Keep `torch.compile` for the elementwise/normalisation tail and for
   CPU-overhead reduction where you are not already on CUDA graphs.
4. **LLM-generated kernels as *unconstrained* production kernels — but this is
   now a narrower claim than it was.** Best published mean for free-form
   generation is 1.10x over *eager*; on B200 against production baselines the
   agents land under 50% on more than half of GEMM/GQA workloads; and letting an
   agent write raw CUDA/PTX measured **0.928x** in CAKE. Every headline claim
   above 3x that has been independently checked has either been retracted
   (Sakana) or explained by a weak baseline (FP32 on L40S, or contest-supplied
   baselines). **However**, AVO's 10.5%-over-FA4 and CAKE's 2.05x-over-FlashKDA
   on B200 are real, verified-to-abstract results, and both come from *bounding
   the agent's action space*. Do not fund free-form kernel generation; a bounded
   search over a CuTe DSL/Gluon skeleton with a SOL-scored harness is defensible
   once the megakernel work is done.
5. **TileLang for SM100 production, today.** The SM100 backend self-describes as
   a "preview version with limited functionality"; SM100 block scaling is
   MXFP8-1D-1D-only in the examples (the NVFP4 example is SM120); and `.ws`
   block-scaled cannot be combined with 2CTA. It has improved — `T.gemm(...,
   mbar=...)` now auto-inserts the wait, and persistent/Stream-K SM100 examples
   landed — but you still get manual effort without CUTLASS's maturity. It is
   excellent as a *readable reference* and for prototyping DSA variants, which is
   exactly what DeepSeek uses it for.
6. **Mosaic GPU / Pallas, for us.** The API is complete and clean (every Blackwell
   entry point verified present) but there is no published Blackwell performance
   evidence and no ecosystem pull for a PyTorch shop.
7. **Chasing cluster size 4 by default.** With cluster size 4 on B200, "only 132
   SMs are active at a time" out of 148 `[verified, TK 2.0 Table 1]`. Measure
   1x1, 2x1, and 2x2 before assuming bigger is better.
8. **Deep NVFP4 GEMM micro-optimisation before fixing launch structure.** Dense
   GEMM is 37.1% and MoE GEMM 19.4%; even a heroic 15% on both is ~8.5%
   end-to-end. Collectives are 19.6% and *47% of that is skew* — pure bubble,
   recoverable by fusion rather than by faster math. Event Tensor's 1.48x over
   vLLM at batch 1 on 8x B200 is the closest measured analogue. Do the megakernel
   first.
9. **Autotuning in the serving process.** Helion's tuner takes ~10 minutes;
   Inductor's `EXHAUSTIVE` GEMM search is worse. Freeze configs into source.
10. **`cutlass_profiler` output taken at face value.** CUTLASS's own 4.6
    measurement methodology guide requires buffer rotation to ≥2x L2
    (`--workspace-count`, `--llc-capacity`), warmup until power stabilises, and
    clock monitoring throughout. Skipping this produces configs that win the
    benchmark and lose in production. **Also: do not compare absolute TFLOP/s
    across papers** — cuBLAS at 8192³ BF16 on B200 is 1570 in one source and
    1671.8 in another.
11. **INT8 anything on B300, if we go that way.** An ISA- and source-level audit
    found that "the PTX ISA never exposes the fifth-generation tensor-core integer
    path" for SM_103a; CUTLASS skips INT8 kernel generation for B300 while
    generating FP8 unconditionally; vLLM ships no INT8 GEMM for Blackwell and
    "fails with a hard runtime error at the first forward pass"; and SGLang's INT8
    tuning stops at SM90 `[reported, arXiv 2608.11693]`. NVFP4/FP8 is the only
    supported narrow path on Blackwell Ultra.

---

## Recommendation per hot kernel family

| Kernel family | % of C1 profile | Author in | Why | Start from |
|---|---|---|---|---|
| **Dense GEMM (NVFP4 / FP8, TP8)** | 37.1% | **CuTe DSL**, C++ CUTLASS if AoT needed | Only tool with 2-CTA block-scaled MMA + TMEM double buffering + CLC persistent scheduling; ~2x above where the Gluon tutorial stops | `cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent{,_prefetch}.py`; `experimental/task_scheduling/blackwell/tutorial/05_gemm_nvfp4_ts/`; tune per §12 |
| **NVFP4 grouped MoE GEMM (256 experts / 8 active)** | 19.4% | **CuTe DSL**, or fork DeepGEMM (CUDA) | SF-in-TMEM layout + `tcgen05.cp` implicit pipeline is the hard part and is already solved there; needs both contiguous (prefill) and masked (CUDA-graph decode) layouts | `blockscaled_grouped_gemm/grouped_blockscaled_gemm.py` + `blockwise_gemm/{contiguous,masked}_grouped_gemm.py`; or DeepGEMM `m_grouped_fp8_gemm_nt_{contiguous,masked}`. Add SonicMoE's tile-aware token rounding. |
| **Sparse MLA attention (DSA)** | 10.9% | **CuTe DSL** | FA4 proves the deeply-pipelined warp-specialised ping-pong schedule is required on SM100 and only expressible here. **No good upstream fork exists for decode**: FlashMLA's sparse MLA decode is 410 TFLOP/s on H800 but only "up to 350 TFlops on B200 (which is not really optimized yet)" `[verified]`. Prefill is better served: 1450 TFLOP/s sparse forward on B200 `[verified]`. Mind the 256x128 minimum block-sparse granularity on Blackwell. | `cute/blackwell/kernel/attention/mla/mla_decode_fp8.py` + FA4 (`flash_attn/cute`); cross-check against FlashMLA SM100 |
| **DSA indexer + top-k (`index_topk_freq=4`)** | 5.8% | **Triton or Gluon** for the logits; **port Guess-Verify-Refine** for the selection | Weighted-ReLU MQA logits are bandwidth- and integer-bound — Triton's strength. The top-k is the part with a published 1.88x: NVIDIA's data-aware selector reuses the previous step's top-k as a prediction signal, narrows by secant-style counting in 1–2 global passes, verifies with a ballot-free collector, finishes exactly in SMEM. Up to 7.52% TPOT at 100K context. `index_topk_freq=4` makes the temporal correlation stronger, not weaker. | DeepGEMM `fp8_mqa_logits` / `fp8_paged_mqa_logits` for reference semantics; TileLang `deepseek_v32/{fp8_lighting_indexer,topk_selector}.py` for a readable spec; arXiv 2604.22312 for the algorithm |
| **Fused allreduce + norm + quant** | part of 19.6% collectives | **CuTe DSL** (or FlashInfer if we accept the dependency) | Needs `multimem.ld_reduce ... .acc::f32` over NVSHMEM symmetric memory to reduce in the NVSwitch, then RMSNorm and NVFP4 quantise in the same kernel without a round trip to HBM. TokenWeave shows this is worth 1.28x latency on 8xH100 using only 2–8 SMs, and that nobody ships it on by default. | `distributed/all_reduce_two_shot_multimem.py`, `all_reduce_one_shot_lamport.py`, `distributed_gemm_blockscaled_all_reduce_ldmcxstmc_blackwell.py`; or FlashInfer `trtllm_allreduce_fusion(..., use_oneshot=)` with `MNNVLAllReduceFusionWorkspace` for CUDA-graph-safe checkpointing |
| **The decode step as a whole** | — | **CuTe DSL megakernel** | The 47%-of-collectives skew is a launch-boundary artifact. Event Tensor measured 1.48x vs vLLM / 1.20x vs SGLang at BS=1 on **8x B200 with a 128-expert MoE**; DeepGEMM Mega MoE 1.96x at batch 1; TK 3.5x vs vLLM on B200; MPK 12.5 vs 14.5 ms/token. Four groups, same answer. | TK megakernel design (instruction interpreter + 13x16KiB SMEM paging + global counter-array deps); Event Tensor for the data-dependent-dynamism (MoE routing) part; DeepGEMM `fp8_fp4_mega_moe` for the MoE segment; CUTLASS `07_group_pipeline_ts` for CuTe DSL cross-operator fusion patterns |
| **Everything else (RoPE, sampling, KV writeback, EAGLE plumbing)** | — | **Helion or `torch.compile`** | Memory-bound; Helion beats hand-Triton 1.85x geomean on B200 and costs ~10 min of tuning once | Helion examples; freeze `helion.Config(...)` into source |

**Three cross-cutting notes.**

*On speculative decoding (worth 3.09x to us):* EAGLE 3-1-4 makes several decode
GEMMs have small M. Together's measurement that "a 64 × 64 × 64 GEMM will run at
one-quarter the FLOP rate of a 128 × 128 × 64 GEMM" on B200 `[verified]` means
the branch structure should be flattened into the M dimension of a single MMA
wherever possible, and 2-CTA MMA (M=256 tiles) considered even when M looks
small. Also note NVIDIA's own admission that `nvMatmulHeuristics` is weakest at
mid-M (~512) bf16 — measure, do not trust the heuristic, in exactly the regime
speculative decoding puts us in.

*On the TileRT gap:* TileRT reaches ~500 tok/s on GLM-5 FP8 on the same hardware
where we get 365 — a 1.37x. Nothing in this literature suggests 1.37x is
available from better GEMM kernels alone: CUTLASS, ThunderKittens and cuBLAS are
all within a few percent of each other on B200 dense GEMM, and all sit at ~70% of
BF16 peak. The gap is much more likely to be launch structure, scheduling, and
fusion — and Event Tensor's 1.48x-over-vLLM at BS=1 on 8x B200 is a
same-hardware, same-shaped-model measurement of exactly that lever. This is why
the megakernel is recommendation #1.

*On version pinning:* CuTe DSL ≥4.6 for IKET intra-kernel tracing; ≥4.7 if you
want compile-time register-spill reporting with source lines and the Task
Scheduling concurrency checker — but note the checker does **not** retrofit onto
existing kernels `[verified, limitations page]`. Triton ≥3.7.0 for end-to-end
2-CTA. PyTorch ≥2.9 for the symmetric memory DeepGEMM Mega MoE requires.

---

## Sources

**Fetched and read (primary), all re-verified 2026-08-17:**

*NVIDIA CUTLASS / CuTe DSL*
- CUTLASS CHANGELOG (4.0.0 2025-06-03 → 4.7.0 2026-08-04) — https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md
- CUTLASS Blackwell SM100/SM120 GEMM functionality — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- CUTLASS CuTe DSL tcgen05 MMA programming guide — https://docs.nvidia.com/cutlass/4.6.2/media/docs/pythonDSL/guides/mma/tcgen05_programming.html
- CUTLASS CuTe DSL limitations — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- CUTLASS CuTe DSL GEMM autotuning guide — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/autotuning_gemm.html
- CUTLASS GEMM performance measurement methodology guidelines (new in 4.6) — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html
- CUTLASS profiler docs (`--enable-best-kernel-for-fixed-shape`, `--sort-results-flops-per-sec`, `--raster_order`, `--swizzle_size`, `--workspace-count`, `--llc-capacity`) — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html
- CUTLASS CuTe DSL Blackwell example inventory (GitHub tree API, `NVIDIA/cutlass@main`, 141 paths enumerated)
- CUTLASS CuTe DSL distributed examples README (NVSHMEM, `multimem.ld_reduce`, NVLS, `.acc::f32`/`.acc::f16`) — https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/README.md
- Colfax Research, "CUTLASS Tutorial: Hardware-supported Block-scaling with NVIDIA Blackwell GPUs" — https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/

*Triton / Gluon*
- Triton Gluon tutorial, "Blocked-Scaled Matrix Multiplication" (benchmark table) — https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05-mma-scaled.html
- Triton releases (tags + `published_at`) — https://api.github.com/repos/triton-lang/triton/releases
- Triton repo tree (`python/examples/gluon/`, `python/triton/experimental/gluon/language/nvidia/blackwell/`, `python/triton/tools/triton_to_gluon_translator/`, `python/triton_kernels/`) — GitHub tree API, `triton-lang/triton@main`
- `triton_kernels/matmul_details/opt_flags.py` — https://raw.githubusercontent.com/triton-lang/triton/main/python/triton_kernels/triton_kernels/matmul_details/opt_flags.py

*Attention / FlashAttention*
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling", arXiv:2603.05451, 2026-03-05 — https://arxiv.org/abs/2603.05451
- PyTorch blog, "FlexAttention + FlashAttention-4: Fast and Flexible", 2026-03-04 — https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- Modal, "Reverse engineering FlashAttention-4" — https://modal.com/blog/reverse-engineer-flash-attention-4
- FlashMLA README (SM90 + SM100; B200 sparse prefill 1450 TFLOP/s; **sparse decode only ~350 TFLOP/s on B200, "not really optimized yet"**) — https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/README.md

*TileLang / DeepSeek*
- Wang, Cheng, Shi, Tang, Mo, Xie, Ma, Xia, Xue, Yang, Yang, "TileLang: A Composable Tiled Programming Model for AI Systems", arXiv:2504.17577, 2025-04-24 — https://arxiv.org/abs/2504.17577
- TileLang repo tree + `examples/gemm_tcgen05/README.md`, `examples/blockscaled_gemm_sm100/mxfp8_illustrated.md`, `examples/deepseek_v32/README.md`, `examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py`
- DeepSeek-V3.2-Exp README (TileLang = research, DeepGEMM/FlashMLA = production) — https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/README.md
- DeepGEMM README (SM90/SM100, Mega MoE, FP4 Indexer, HyperConnection, grouped layouts, env vars) — https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md
- DeepGEMM PR #316, "Mega MoE benchmarks" (EP8, V4-Flash/V4-Pro tables) — https://github.com/deepseek-ai/DeepGEMM/pull/316

*ThunderKittens family*
- HazyResearch, "ThunderKittens 2.0: Even Faster Kernels for Your GPUs", 2026-02-19 — https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
- HazyResearch, "No bubbles" megakernel post, 2025-05-27 — https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Together AI, "ThunderKittens Now Optimized for NVIDIA Blackwell GPUs" (source of the 64³-vs-128²x64 quarter-rate claim) — https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus
- Hu, Wadsworth, Siddens, Winata, Fu, Swann, Osama, Ré, Arora, "HipKittens: Fast and Furious AMD Kernels", arXiv:2511.08083, 2025-11-11 (**Table 2 §3.3.1** is the B200 TK-1538 / CUTLASS-1570 source) — https://arxiv.org/abs/2511.08083
- Sul, Arora, Spector, Ré, "ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels", arXiv:2511.13940, 2025-11-17 — https://arxiv.org/abs/2511.13940

*Megakernels, fusion, collectives*
- Jin, Hou, Wang, Lai, Chen, Ye et al., "Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel", **MLSys 2026**, arXiv:2604.13327, 2026-04-14 — https://arxiv.org/abs/2604.13327
- Gond, Kwatra, Ramjee, "TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference", **MLSys 2026**, arXiv:2505.11329 (v5 2026-05-01) — https://arxiv.org/abs/2505.11329
- Wu, Cheng, Liu, Shi, Ji, Ao, Velliengiri, Miao, Padon, Jia, "Mirage: A Multi-Level Superoptimizer for Tensor Programs", OSDI'25, arXiv:2405.05751 (v3) — https://arxiv.org/abs/2405.05751
- Zhihao Jia, "Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference" (MPK) — https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- Guo, Mishra, Cheng, Stoica, Dao, "SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations", arXiv:2512.14080, 2025-12-16 — https://arxiv.org/abs/2512.14080
- Jaber, Jaber, "AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis", arXiv:2606.09682, 2026-06-08 — https://arxiv.org/abs/2606.09682
- FlashInfer communication API (`trtllm_allreduce_fusion`, `AllReduceFusionPattern`, `MnnvlMemory`, `McastGPUBuffer`, `MNNVLAllReduceFusionWorkspace`) — https://docs.flashinfer.ai/api/comm.html

*Compilers, autotuning, PyTorch*
- PyTorch blog, "Helion: A High-Level DSL for Performant and Portable ML Kernels", 2025-10-23 — https://pytorch.org/blog/helion/
- `torch/_inductor/config.py` (autotune flags, `NVGEMM`/`CUTEDSL` backends, mid-M caveat) — https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/config.py
- PyTorch blog, "Accelerating Generative AI with PyTorch II: GPT, Fast" (A100-80GB **power-limited to 330W**) — https://pytorch.org/blog/accelerating-generative-ai-2/
- Dao-AILab QuACK (CuTe DSL memory-bound kernels) — https://github.com/Dao-AILab/quack
- Yadav, Zhao, Kumar, "Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs", arXiv:2604.23466, 2026-04-25 (v2 2026-06-03) — https://arxiv.org/abs/2604.23466
- Chen, Fan, Collins, Hagedorn, Gaburov, Masuda, Brookhart, Sullivan, Knight, Zhang, Grover, "Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References", CGO'26, arXiv:2510.14719 — https://arxiv.org/abs/2510.14719
- Ding, Yu, Zheng, Liu, Wang, Pekhimenko, "Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs", ASPLOS 2023, arXiv:2210.09603 — https://arxiv.org/abs/2210.09603
- Zheng, Jia, Sun, Wu, Yu, Haj-Ali, Wang, Yang, Zhuo, Sen, Gonzalez, Stoica, "Ansor: Generating High-Performance Tensor Programs for Deep Learning", OSDI 2020, arXiv:2006.06762 — https://arxiv.org/abs/2006.06762
- Zhu, Wu, Diao, Ke, Li, Zhang, Xue, Ma, Xia, Cui, Yang, Yang, Zhou, Cidon, Pekhimenko, "Roller: Fast and Efficient Tensor Compilation for Deep Learning", OSDI'22 — https://www.microsoft.com/en-us/research/publication/roller-fast-and-efficient-tensor-compilation-for-deep-learning/
- Zhang, Ding, Qian, Wang, Cao, Xue, Huang, Yang, Zhang, "WaveTune: Wave-aware Bilinear Modeling for Efficient GPU Kernel Auto-tuning", arXiv:2604.10187, 2026-04-11 — https://arxiv.org/abs/2604.10187
- Sharma, Datta, "RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts", arXiv:2604.26039, 2026-04-28 — https://arxiv.org/abs/2604.26039
- JAX docs, "Writing Mosaic GPU kernels with Pallas" — https://docs.jax.dev/en/latest/pallas/gpu/reference.html

*LLM-driven kernel generation*
- Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini, "KernelBench: Can LLMs Write Efficient GPU Kernels?", arXiv:2502.10517, 2025-02-14 — https://arxiv.org/abs/2502.10517
- Stanford CRFM, "Surprisingly Fast AI-Generated Kernels We Didn't Mean to Publish (Yet)", 2025-05-28 — https://crfm.stanford.edu/2025/05/28/fast-kernels.html
- Baronio, Marsella, Pan, Guo, Alberti, "Kevin: Multi-Turn RL for Generating CUDA Kernels", arXiv:2507.11948, 2025-07-16 — https://arxiv.org/abs/2507.11948
- Xing, Zhai, Jiang, Dong, Wu, Ye, Ruan, Huang, Zhang, Yin, Bayyapu, Ceze, Chen, "FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems", arXiv:2601.00227, 2026-01-01 — https://arxiv.org/abs/2601.00227
- E. Lin, Modi, Hari et al., "SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits", arXiv:2603.19173, 2026-03-19 — https://arxiv.org/abs/2603.19173
- Liao, Qin, Wang, Golden, Kuchnik, Yetim et al. (36 authors, Meta), "KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta", arXiv:2512.23236, 2025-12-29 — https://arxiv.org/abs/2512.23236
- T. Chen, Ye, Xu, Ye, Liu, Hassani, T. Chen, Kerr, Wu et al., "AVO: Agentic Variation Operators for Autonomous Evolutionary Search", arXiv:2603.24517, 2026-03-25 — https://arxiv.org/abs/2603.24517
- Ye, Huang, Jin, Hou, Shao, Yu, Chen, Cowan, Cao, Xing, Chen, Grover, T. Chen, Ceze, "CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution", arXiv:2608.12629, 2026-08-12 — https://arxiv.org/abs/2608.12629
- Shui, Ma, Xu, Wen, Wang, "Harness Engineering for LLM-Driven GPU Kernel Generation", arXiv:2607.17979, 2026-07-20 — https://arxiv.org/abs/2607.17979
- TechCrunch, "Sakana walks back claims that its AI can dramatically speed up model training", 2025-02-21 — https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/

*Hardware characterisation and DSA*
- Cheng, Zhao, Liu, Li, Qiao, Duan, Chen, Chen, Darvish Rouhani, Yang (NVIDIA), "Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation", arXiv:2604.22312, 2026-04-24 — https://arxiv.org/abs/2604.22312
- Teng-Ruei Chen, "Spec Sheets Are Not Kernels: An ISA- and Source-Level Audit of INT8 Availability on NVIDIA Blackwell Ultra", arXiv:2608.11693, 2026-08-12 — https://arxiv.org/abs/2608.11693
- Jarmusch, Chandrasekaran, "Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis", arXiv:2512.02189, 2025-12-01 (v3 2026-03-02) — https://arxiv.org/abs/2512.02189
- Blackwell GPU wiki, "tcgen05 and TMEM" — https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/

**Deliberately not cited:** any paper whose title, authors and URL I could not
confirm by fetching the primary source. Two claims present in the previous
revision have been **withdrawn** rather than corrected, because I could not find
a source: (a) "preliminary scaled `tcgen05.mma` support exists" in Mosaic
GPU/Pallas, and (b) the TK megakernel achieving "78% of memory bandwidth" on
H100. One claim was **re-attributed**: the 64³-vs-128²x64 quarter-FLOP-rate
result is in the Together AI blog, not the HazyResearch TK 2.0 post.
