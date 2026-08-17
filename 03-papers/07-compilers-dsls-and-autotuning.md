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

**Reference peaks used throughout (B200 SXM, dense, no sparsity):** BF16/FP16
2.25 PFLOP/s, FP8/FP6 4.5 PFLOP/s, FP4 9 PFLOP/s. The BF16 figure is confirmed
by the FlashAttention-4 paper, which states "Blackwell doubles the FP16/BF16
tensor core throughput compared to Hopper (2.25 PFLOPS vs 1 PFLOPS)" and derives
its 71%-of-peak claim from it `[verified]`.

---

## Bottom line for our system

Ranked by expected effect on our two objectives (single-stream latency; tok/s
per GPU at concurrency).

1. **Fuse the decode step into a persistent megakernel. This is the largest
   single lever available and it is a kernel-authoring decision, not a
   scheduling one.** Our collectives are 19.6% of the profile and 47% of that
   is rank-arrival skew — i.e. bubbles, not bytes. Skew is exactly what
   per-operator kernel launches manufacture. Three independent results say the
   fix works: the ThunderKittens megakernel runs Llama-1B bf16 forward in
   <680 us on B200, 3.5x vs vLLM and 1.5x vs SGLang `[reported]`; DeepGEMM's
   "Mega MoE" (dispatch + FP8xFP4 linear1 + SwiGLU + linear2 + combine in one
   kernel, overlapping NVLink with tensor cores) measures 1.50–1.96x over a
   DeepEP + TileLang baseline at EP8 across batch 1–32768 `[verified]`; Mirage
   Persistent Kernel gets 12.5 ms/token vs 14.5 for vLLM/SGLang on A100
   `[reported]`. Expected effect: this is where most of the 365 → ~500 tok/s gap
   to TileRT lives.
2. **Make CuTe DSL (CUTLASS 4.x Python) the default language for any new
   tensor-core-bound Blackwell kernel.** It is the only Python-level tool that
   exposes the full SM100 surface *and* has production proof: FlashAttention-4
   is written entirely in CuTe DSL and beats cuDNN 9.13 by 1.1–1.3x on B200
   forward `[reported]`, while compiling in 2.5 s forward / 1.4 s backward
   against FA3's C++-template 55 s / 45 s — 20–30x faster `[verified, FA4 Table
   4]`. Compile time matters to us specifically because fixed shapes mean we
   will recompile a config space many times.
3. **Do not hand-roll the NVFP4 grouped MoE GEMM scale-factor path.** The hard
   part is not the MMA, it is getting SFA/SFB into TMEM in the exact
   `32x16B`-tile, `warpx4`-multicast layout that `tcgen05.mma...block_scale`
   demands, plus the `tcgen05.cp`/`tcgen05.mma` implicit-pipeline ordering.
   CUTLASS ships `blackwell/kernel/blockscaled_grouped_gemm/grouped_blockscaled_gemm.py`
   and `blockwise_gemm/{contiguous,masked}_grouped_gemm.py` `[verified, repo
   tree]`; DeepGEMM ships M-axis contiguous and masked grouped FP8/FP4 GEMMs
   with a runtime JIT `[verified, README]`. Fork one of these.
4. **Write the fused allreduce+norm+quant kernel with `multimem` over NVSHMEM
   symmetric memory, not with a library allreduce plus separate norm.** CUTLASS
   4.x ships exactly this shape as CuTe DSL examples:
   `all_reduce_two_shot_multimem.py`, `all_reduce_one_shot_lamport.py`, and
   `distributed_gemm_blockscaled_all_reduce_ldmcxstmc_blackwell.py`
   `[verified, repo tree]`. `multimem.ld_reduce` does the reduction inside the
   NVSwitch (NVLink SHARP) and supports `.acc::f32` for bf16/fp16 operands and
   `.acc::f16` for FP8 `[verified, CUTLASS distributed README]` — which is what
   you need if the norm is going to be numerically defensible. FlashInfer's
   `trtllm_allreduce_fusion` is the ready-made alternative and already fuses
   RMSNorm + FP8/NVFP4 quant `[verified, API docs]`.
5. **Stop trying to make Triton hit peak on tensor-core-bound Blackwell
   kernels; use Gluon there instead, and keep Triton for the memory-bound
   tail.** PyTorch's own FlexAttention team states flatly: "On Blackwell,
   high-performance attention requires a deeply pipelined, warp-specialized
   kernel. These techniques aren't expressible in our Triton-based
   implementation" — and measured 1.6–3.2x forward / 1.85–2.3x backward by
   switching FlexAttention's backend from Triton to CuTe-DSL FA4 on GB200
   `[verified]`. Independently, FA4 reports 2.1–2.7x over Triton on B200
   `[reported]`. Our DSA indexer, top-k, quantize, and RoPE kernels are a
   different story — Triton is fine and fast there.
6. **Our shapes are fixed, so autotune once, exhaustively, over a small space,
   and freeze the config into source.** The practical space on SM100 is small:
   `mma_tiler_mn` x `cluster_shape_mn` x `use_2cta_instrs` x `use_tma_store` x
   pipeline stages x raster/swizzle — that is what CUTLASS's own CuTe DSL
   autotuning guide searches, by exhaustive `product()` `[verified]`. Seed it
   with `nvMatmulHeuristics` (exposed through Inductor's `NVGEMM` backend,
   `nvgemm_max_profiling_configs` default 10) or `cutlass_profiler
   --enable-best-kernel-for-fixed-shape` `[verified]`, then *measure*. Do not
   ship a cost model.
7. **Attack rank-arrival skew with CLC + persistent scheduling + PDL, in the
   kernel.** Cluster Launch Control lets a persistent worker query for the next
   tile instead of taking a static slice, which is the documented fix for
   load-imbalanced persistent kernels; CuTeDSL 4.4 added CLC-based persistent
   scheduling `[verified, PyTorch blog]`. Triton 3.7 exposes CLC in Gluon
   (`gluon/language/nvidia/blackwell/clc.py`) `[verified, repo tree]`. TileLang
   has `gemm_tcgen5mma_ws_clc.py` `[verified]`. DeepGEMM exposes PDL via
   `deep_gemm.set_pdl` `[verified]`.
8. **Treat LLM-generated kernels as a search driver over a template space, not
   as a source of kernels.** On B200, FlashInfer-Bench measured frontier agents
   at <50% of production performance on the majority of GEMM and paged-GQA
   workloads, with 30 of 32 correctness failures being plain compile errors
   `[reported]`. The best RL-trained kernel model published (Kevin, QwQ-32B
   base) reaches a *mean 1.10x over PyTorch eager* `[reported]` — eager, not
   CUTLASS. This is not a path to beating a hand-written NVFP4 grouped GEMM.
9. **Keep Helion / `torch.compile` for the long tail, and only there.** Helion
   posts a 3.27x geomean over eager on B200 vs 2.70x for `torch.compile
   max-autotune` and 1.76x for hand-written Triton on the Liger suite
   `[verified]` — but it compiles *to Triton*, so it inherits Triton's Blackwell
   ceiling on anything tensor-core-bound.
10. **Keep a TileLang track for research prototyping of DSA variants, not for
    production.** This is DeepSeek's own posture: for V3.2 they point at
    TileLang for kernels with "better readability and research-purpose design"
    and at DeepGEMM/FlashMLA for "high-performance CUDA kernels" `[verified,
    DeepSeek-V3.2-Exp README]`.

---

## Ground rules: what SM100 demands of a kernel language

Everything below is a consequence of five hardware changes. A tool that does not
expose all five cannot reach peak on B200.

- **`tcgen05` replaces WGMMA.** Instruction shapes are 64x64..128x256 for
  1-SM (`cta_group::1`) and 128x64..256x256 for 2-SM (`cta_group::2`)
  `[verified, CUTLASS Blackwell functionality docs]`. Issue semantics are
  *single-thread*: one elected lane in one warp initiates the whole MMA
  `[verified, TileLang `mxfp8_illustrated.md`]`.
- **TMEM.** 256 KB per SM, addressed by 32-bit logical addresses in a separate
  address space from SMEM/GMEM, allocated in 32-bit-wide *columns* via
  `tcgen05.alloc` / freed via `tcgen05.dealloc` +
  `tcgen05.relinquish_alloc_permit` `[verified, CUTLASS + Blackwell GPU wiki]`.
  ThunderKittens describes the geometry as 128x512 organised as two 128x256
  slots `[reported]`. Sub-32-bit accumulators pack:
  `column_to_element_scale = 32 // dtype.width` `[verified, CUTLASS tcgen05
  guide]`. TMEM is per-SM and not addressable by other CTAs outside the pair.
- **Everything is async.** MMA and TMA both complete via `mbarrier`. You now
  own a phase-parity protocol. Get it wrong and you get a hang, not a wrong
  answer.
- **2-CTA MMA + clusters.** SM100 supports 1x1 through 4x4 clusters with
  multicast; SM120 (workstation/GeForce Blackwell) is fixed at 1x1, TN-only,
  and therefore *cannot use CTA-pair MMA at all* `[verified, CUTLASS docs +
  Pallas docs]`. Cluster size is not free: ThunderKittens documents that
  size-4 clusters only fill 132 of 148 SMs `[reported]`.
- **Block scaling is a first-class MMA mode.**
  `tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}` takes
  `[scale-A-tmem]` and `[scale-B-tmem]` operands. `mxf8/mxf6/mxf4` use 32-element
  blocks with UE8M0 scales; `nvf4` uses 16-element blocks with UE4M3 scales
  (max representable 448) `[verified, Colfax]`. Scale factors must sit in TMEM
  in 32x16-byte tiles, duplicated to all 32 lane partitions, delivered by
  `tcgen05.cp` with `.shape=.32x128b .multicast=.warpx4`. Crucially,
  `tcgen05.cp` and `tcgen05.mma` form an *implicit* pipeline with guaranteed
  ordering — no circular buffering of SF in TMEM required `[verified, Colfax]`.
  ThunderKittens measured ~500 TFLOP/s of NVFP4 throughput purely from
  discovering and exploiting this implicit pipelining `[reported]`.

This is the bar. Read the table below against it.

---

## Family 1: Kernel authoring languages and DSLs

| Tool | Lab | Venue/date | Blackwell surface | Headline result (hardware, model) | Production? |
|---|---|---|---|---|---|
| CUTLASS 3.x/4.x C++ + CuTe | NVIDIA | continuous; 4.0 2025-06-03 → 4.7.0 2026-08-04 | full: tcgen05, TMEM, TMA, clusters, CLC, block-scale, 2SM | the reference implementation; ~1570 TFLOP/s BF16 GEMM on B200 `[reported, HipKittens Fig. 19]` | yes — everywhere |
| **CuTe DSL** (CUTLASS Python) | NVIDIA | CUTLASS 4.0, 2025-06-03 | full, same as C++ | FA4: 1.1–1.3x over cuDNN 9.13, 2.1–2.7x over Triton, B200 SXM6 1000W, BF16, hd=128 `[reported]`; compile 2.5 s vs FA3 55 s `[verified]` | yes — FA4, QuACK, FlexAttention FLASH backend |
| Triton (`tl.*`) | OpenAI | MAPL'19; 3.4.0 2025-07-30 → 3.7.1 2026-06-18 | partial: TMA descriptors, `dot_scaled`→tcgen05, TMEM/WS via compiler passes; no explicit TMEM/mbarrier control | 62–101% of cuBLAS across H100 NVL / B200 / RTX PRO 6000 `[reported, arXiv 2604.23466]`; 2.1–2.7x *slower* than FA4 on B200 attention `[reported]` | yes — vLLM/SGLang glue, `triton_kernels` MoE |
| **Gluon** (Triton's low-level dialect) | OpenAI | shipped in Triton 3.4+ | full: `allocate_tensor_memory`, `tcgen05_mma_scaled/copy/commit`, `tma.async_load/store`, `mbarrier.*`, `gl.warp_specialize`, CLC, 2CTA | 4871 TFLOP/s mxfp4, 4847 nvfp4, 2378 mxfp8 at M=N=K=8192 `[verified, official tutorial table]` — ≈54% of B200 FP4 peak `[inferred]` | emerging (Triton internals, some prod MoE) |
| TileLang | Lei Wang, Lingxiao Ma, Jilong Xue, Fan Yang, Zhi Yang et al. | arXiv 2504.17577, 2025-04-24 | preview: `T.alloc_tmem`, `T.tcgen05_gemm`, manual mbarrier parity; 2SM; CLC; MXFP8 block-scale on SM100 | H100: 1.36x FA3, 1.41x Triton (attention); 98% of FlashMLA in ~70 LOC `[reported]`. **No SM100 number published.** | research; DeepSeek ships TileLang DSA kernels as the *readable* reference |
| ThunderKittens 2.0 | HazyResearch (Stanford) | blog 2026-02-19 | full: tcgen05.mma/cp/alloc, TMEM, TMA, DSMEM, clusters | B200 BF16 GEMM ~1538 TFLOP/s vs CUTLASS ~1570 `[reported, HipKittens Fig. 19]`; "at or near cuBLAS", attention "near cuDNN" `[reported]` | partial (Together, internal forks) |
| Mosaic GPU / Pallas | Google (JAX) | continuous | `plgpu.tcgen05_mma`, `plgpu.TMEM`, `copy_gmem_to_smem`, `Barrier(orders_tensor_core=True)`, `ClusterBarrier`, `dynamic_scheduling_loop` | **no public B200 vs CUTLASS number found** | JAX/TPU-first shops only |
| Helion | Meta (Jason Ansel et al.) | PyTorch blog 2025-10-23; beta 2025-10-22 | inherits Triton's | B200 geomean 3.27x over eager vs 2.70x `torch.compile`, 1.76x hand Triton `[verified]`; matches Quack CuTe DSL RMSNorm-bwd on H100 | beta |
| CUDA Tile / cuTile | NVIDIA | independent eval arXiv 2604.23466, 2026-04-25 | tile-level, compiler-managed | B200 attention 1007 TFLOP/s, 2.5x FA2; GEMM 52–79% of cuBLAS in 22 LOC `[reported]` | no |
| Hidet | Toronto/AWS (Ding et al.) | ASPLOS 2023, arXiv 2210.09603 | pre-Blackwell | 1.48x over ONNX Runtime, 1.22x avg over TVM+Ansor; tuning 20x faster than AutoTVM `[reported]` | no (absorbed into CentML) |
| torch.compile / Inductor | Meta | continuous | Triton templates + `CUTLASS`/`CUTEDSL`/`NVGEMM` autotune backends | gpt-fast: 25.5 → 244.7 tok/s Llama-7B, A100-80GB `[verified]` | yes — everywhere, as glue |

---

## Family 2: Autotuning and search

| System | Lab | Venue/date | Approach | Result | Cost |
|---|---|---|---|---|---|
| CUTLASS profiler | NVIDIA | continuous | enumerate + measure | `--enable-best-kernel-for-fixed-shape` + `--sort-results-flops-per-sec` `[verified]` | minutes–hours |
| CuTe DSL autotuning guide | NVIDIA | CUTLASS 4.x | exhaustive `product()` over `mma_tiler_mn`, `cluster_shape_mn`, `use_2cta_instrs`, `use_tma_store`; two-level cache `[verified]` | — | compile+bench per config |
| `nvMatmulHeuristics` | NVIDIA | shipped; surfaced as Inductor `NVGEMM` backend | learned/analytic heuristic → top-N configs (`nvgemm_max_profiling_configs=10`) `[verified]` | — | ~zero |
| cuBLASLt heuristics | NVIDIA | continuous | `cublasLtMatmulAlgoGetHeuristic` returns ranked candidates; `cublasLtHeuristicsCacheSetCapacity` `[verified]` | — | ~zero |
| Ansor | Berkeley (Zheng et al.) | OSDI 2020, arXiv 2006.06762 | hierarchical sketch + random annotation + evolutionary search + learned cost model | up to 1.7x on NVIDIA GPU vs SOTA `[reported]` | hours–days |
| Roller | MSRA (Zhu et al.) | OSDI 2022 | **construction, not search**: rTile aligned to hardware, micro-performance model | "generate efficient kernels in *seconds*", comparable to SOTA on GPUs `[reported]` | seconds |
| Hidet | Toronto/AWS | ASPLOS 2023 | task-mapping in the program + post-scheduling fusion | 11x faster tuning than Ansor `[reported]` | minutes |
| Helion autotuner | Meta | 2025-10 | implicit search space, differential evolution / pattern search | "1520 configs in 586.6 s" `[verified]` ≈ 0.39 s/config | ~10 min/kernel |
| WaveTune | SJTU et al. | arXiv 2604.10187, 2026-04-11 | wave-aware bilinear latency model + sparse sampling + dual-table retrieval | up to 1.83x kernel, 1.33x TTFT; runtime decision overhead 5 orders of magnitude below exhaustive `[reported]` | near-zero at runtime |
| Mirage | CMU (Wu, Jia et al.) | OSDI 2025, arXiv 2405.05751 | μGraph superoptimisation + abstraction pruning + probabilistic equivalence check | 1.1–2.9x over best baseline, A100/H100 `[reported]` | **up to 4 hours per program** `[reported]` |
| Tawa | NVIDIA + Cornell (Chen et al.) | CGO 2026, arXiv 2510.14719 | automatic warp specialisation via `aref` IR | H100 only: 1.1x cuBLAS GEMM, 1.2x Triton attention, matches CUTLASS FA3 `[reported]` | compiler pass |

---

## Family 3: LLM-driven kernel generation

| Work | Lab | Venue/date | Hardware | Claim | What survives scrutiny |
|---|---|---|---|---|---|
| KernelBench | Stanford (Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini) | arXiv 2502.10517 | mixed | 250 tasks, `fast_p` metric | frontier reasoning models match the PyTorch baseline in **<20%** of cases `[reported]` |
| Stanford CRFM "fast kernels" blog | Stanford | 2025-05-28 | **L40S**, FP32 | LayerNorm 484%, Conv2D 180%, matmul 101% of torch | authors' own caveat: FP16 matmul **52%** of `torch.matmul`, FP16 flash attention **9%** of SDPA; FP32 is "less optimized on recent hardware" `[verified]` |
| Sakana "AI CUDA Engineer" | Sakana AI | Feb 2025 | — | "10–100x" | **retracted**: reward-hacked a memory-reuse hole in the eval harness; company walked the claims back and rebuilt the harness `[reported, TechCrunch 2025-02-21]` |
| Kevin | Baronio, Marsella, Pan, Guo, Alberti | arXiv 2507.11948, 2025-07-16 | — | multi-turn RL, QwQ-32B base | correctness 56%→82%; **mean 1.10x over PyTorch *eager***, beating o4-mini's 0.78x `[reported]` |
| FlashInfer-Bench | UW/NVIDIA (Xing, Ye, Ceze, Chen et al.) | arXiv 2601.00227 | **B200** | real serving traces, 8 kernel families, 1600 workloads | GPT-5 83.9% / o3 71.3% / Gemini 2.5 Pro 48.8% pass; **<50% of production perf** on most GEMM and paged-GQA workloads; agents sometimes just call cuBLAS `[reported]` |
| SOL-ExecBench | multi-lab (Lin et al.) | arXiv 2603.19173 | **Blackwell** | 235 problems from 124 models, BF16/FP8/NVFP4, scored against hardware speed-of-light not a software baseline; clock-locked, L2-flushed, static anti-reward-hack checks `[reported]` | the right methodology |
| KernelEvolve | Meta (Liao et al., 39 authors) | arXiv 2512.23236 | NVIDIA/AMD/MTIA | 100% pass on 250 KernelBench problems + 160 ATen ops, targeting Triton and CuTe DSL `[reported]` | correctness ≠ speed; no headline speedup in the abstract |

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
rather than SMEM it is placed *after* the accumulator columns, offset by
`num_acc_tmem_cols * column_to_element_scale`.

**What ships for our exact kernel families** (verified by walking
`NVIDIA/cutlass@main:examples/python/CuTeDSL/cute/blackwell/`):

- `dense_gemm/` — 6 variants: plain, persistent, persistent+dynamic,
  persistent+prefetch, alpha/beta persistent, software-pipelined.
- `blockscaled_gemm/` — `dense_blockscaled_gemm_persistent{,_amax,_prefetch}.py`
  plus an SM103 (B300) variant.
- `blockscaled_grouped_gemm/grouped_blockscaled_gemm.py` — **NVFP4 grouped MoE
  GEMM.**
- `blockwise_gemm/{contiguous_grouped_gemm,masked_grouped_gemm}.py` — the
  DeepSeek-style contiguous (prefill) and masked (CUDA-graph decode) layouts.
- `moe/moe_persistent_scheduler.py`, `moe_sched_extension.py`.
- `attention/mla/mla_decode_fp8.py`, `mla_decode_fp16.py` — **MLA decode.**
- `attention/fmha/fmha.py`, `fmha_bwd.py`, `mixed_input_fmha/*` (prefill d=256
  and d=512, decode).
- `distributed/` — `all_reduce_one_shot_lamport.py`,
  `all_reduce_two_shot_multimem.py`, `all_reduce_tma.py`,
  `distributed_gemm_all_reduce_*`, `distributed_gemm_reduce_scatter_*`, and
  `distributed_gemm_blockscaled_all_reduce_ldmcxstmc_blackwell.py`.
- `rmsnorm/rmsnorm.py`, `reduce/reduce.py`.
- Tutorials: `nvfp4_gemm_0.py`, `nvfp4_gemm_1.py`, `tutorial_tma/`.

That list covers five of our five hot kernel families. This is the single
strongest argument for CuTe DSL: you are forking, not writing.

**Compile time.** FA4's Table 4 gives the definitive number: **2.5 s forward /
1.4 s backward in CuTe DSL vs 55 s / 45 s for FA3's C++ templates — 20–30x**
`[verified]`. JIT results are cached in memory and on disk at
`/tmp/{user}/cutlass_python_cache`; the cache key hashes the generated MLIR
bytecode, all CuTe DSL Python sources, all shared libraries, and all CuTe DSL
env vars `[verified]`. Override with `CUTE_DSL_CACHE_DIR`; disable file caching
with `CUTE_DSL_DISABLE_FILE_CACHING=True`. AoT compilation landed in 4.4
(2026-02-14) and `cute.compile_to` for custom pipelines in 4.6 (2026-07-01)
`[verified, CHANGELOG]`.

**What it cannot express** `[verified, official limitations page]`:
- No `global` / `nonlocal`; captured outer-scope variables outside the JIT
  context raise at runtime.
- Functions can only return `constexpr` values, not runtime-computed types.
- Lists/tuples/dicts are compile-time metaprogramming only — no structural
  mutation in-kernel, no dynamic list indexing.
- **CuTe layout algebra is restricted to 32-bit shapes and strides.** For a
  paged KV cache spanning >2^31 elements you will be doing pointer arithmetic
  outside the layout system.
- No early `return` out of if/else.
- ~2–3 us per-tensor DLPack conversion overhead on the host — this matters at
  concurrency 1 where a decode step is ~2.7 ms. Mitigate with the TVM-FFI path
  added in 4.3 or AoT.
- Debugging is materially worse than C++: no single-step through JIT code.
- Not supported: convolutions, preferred clusters, Windows.

**Version note.** Things we care about landed late. SM103 (B300/GB300) FP4
"Ultra" blockscaled GEMM: 4.4. `block_copy()` to simplify TMA and S2T (SMEM →
TMEM) copies: 4.5. MXF8 x MXF4 / MXF8 x MXF6 mixed block-scaled MMA: 4.5. IKET
intra-kernel tracing and SASS dump without a CUDA toolkit: 4.6. Compile-time
register-spill and local-memory reporting, and a "Task Scheduling" framework
that does *static analysis of execution schedules for warp-specialized kernels*:
4.7 `[verified, CHANGELOG]`. Pin ≥4.6 for the profiling tooling alone.

## 2. CUTLASS 3.x/4.x C++ — still the floor and the ceiling

Everything CuTe DSL does, C++ does first, plus: EVT (Epilogue Visitor Trees) for
arbitrary fused epilogues, sparse GEMM `(2/4*MMA-K)` variants, and the widest
dtype/layout matrix. SM100 supports TN/NN/NT/TT and 1x1..4x4 clusters with
multicast; SM120 is TN-only, 1x1 clusters, no multicast, pingpong/cooperative
schedules `[verified, Blackwell functionality docs]`.

The cost is compile time (the 55 s FA3 figure above is representative) and the
template error surface. Use C++ when you need EVT, sparsity, or you are shipping
an AoT binary with no Python; use CuTe DSL otherwise. Alignment requirements are
a real trap: narrow-precision mixed pairs demand up to **128-element alignment**
per PTX restrictions `[verified]` — this bites when your MoE intermediate
dimension is not a nice multiple.

## 3. Triton and Gluon — one repo, two languages, very different ceilings

### Triton proper

Triton's Blackwell backend matured across five releases, dates verified against
the GitHub releases API: **3.4.0 (2025-07-30)**, **3.5.0 (2025-10-21)**, 3.5.1
(2025-11-12, SM103/GB300 fixes), **3.6.0 (2026-01-21)**, **3.7.0 (2026-05-07)**,
3.7.1 (2026-06-18).

- 3.5.0: `tcgen05.commit`, generic lowering for `tcgen05.ld/st`, subtile QK TMEM
  load, warp specialisation for Hopper/Blackwell, ragged TMA.
- 3.6.0: **`tcgen05.mma.scaled` support, native FP4 scaled dot, native MXFP-FP8
  scaled dot**, TMEM bitwidth encoding, TMEM layout broadcasting, `aref`-based
  end-to-end warp specialisation, TMA gather4 on sm_120.
- 3.7.0: **2-CTA mode end-to-end**, Gluon multi-CTA + 2CTA, Blackwell scale
  swizzling for batched matmul, fine-grained cluster barrier, `tcgen05.ld.red`
  on sm_103, block-scaled matmul baselining for mxfp8/nvfp4.

So: yes, Triton can emit `tcgen05` and block-scaled FP4 today, via `tl.dot_scaled`.
The question is whether it hits peak, and the honest answer is *for GEMM, close;
for attention, no*.

**GEMM.** The only independent measurement I could find covering B200 is the
CUDA Tile evaluation paper (arXiv 2604.23466, Yadav/Zhao/Kumar, 2026-04-25),
which reports Triton at **62–101% of cuBLAS** across H100 NVL, B200, and RTX PRO
6000 `[reported]`. That is a wide band, and the bottom of it is unacceptable for
a kernel that is 37.1% of our profile.

**Attention.** Two independent sources put Triton 2–3x behind:
FA4 measures **2.1–2.7x** over Triton on B200 SXM6 forward `[reported]`;
PyTorch's FlexAttention team measures **1.6–3.2x forward / 1.85–2.3x backward**
by swapping Triton for the CuTe-DSL FA4 backend on GB200 `[verified]`. Their
diagnosis is explicit and worth quoting in full because it generalises to our
sparse MLA kernel:

> "On Blackwell, high-performance attention requires a deeply pipelined,
> warp-specialized kernel. These techniques aren't expressible in our
> Triton-based implementation."

The mechanism: Blackwell's SFU did not scale with the tensor cores, so `exp()`
in softmax now costs about as much as the matmuls, which forces a two-tile
ping-pong overlapping one tile's MMA with the other's exponentiation. That is a
5-role warp-specialised schedule (load / MMA / softmax x8 warps / correction x4 /
epilogue), which a general-purpose scheduler will not discover `[verified,
PyTorch blog + Modal FA4 teardown]`.

### Gluon — Triton's escape hatch, and it is genuinely good

Gluon is a second frontend in the Triton repo that drops the auto-layout,
auto-pipeline machinery and hands you the hardware. From the Gluon intro, quoted
by the PyTorch team:

> "While the Triton compiler does a good job of generating efficient code for a
> wide range of kernels, it can be beaten by hand-tuned low-level code. When
> this happens, there is little the user can do to significantly improve
> performance since all the details are hidden."

The Blackwell surface is complete: `allocate_tensor_memory`,
`TensorMemoryLayout`, `TensorMemoryScalesLayout`, `tensor_memory_descriptor`,
`tcgen05_mma_scaled`, `tcgen05_copy`, `tcgen05_commit`, `tma.async_load/store`,
`tma.store_wait`, `mbarrier.{init,expect,wait,arrive,invalidate}`,
`fence_async_shared`, `gl.warp_specialize`, `gl.AutoLayout`, plus
`nvidia/blackwell/clc.py` for Cluster Launch Control `[verified, repo tree +
tutorial]`.

**The block-scaled Gluon tutorial is the single best public Blackwell FP4
optimisation walkthrough that exists.** Its measured table (M=N=K=8192, GPU
model unfortunately unspecified) `[verified]`:

| Step | mxfp8 x mxfp8 | mxfp4 x mxfp4 | mxfp8 x mxfp4 | nvfp4 x nvfp4 |
|---|---|---|---|---|
| `simple_mma_scaled` | 33.4 | 67.0 | 34.6 | 70.8 |
| `mma_scaled_contig` | 663.3 | 1435.1 | 741.8 | 1303.7 |
| `mma_scaled_packed_block` | 901.0 | 2081.8 | 1000.5 | 2002.1 |
| `mma_scaled_tcgen05_copy` | 929.1 | 2147.8 | 1035.6 | 2092.4 |
| `mma_scaled_pipelined` | 2018.6 | 3916.6 | 2144.1 | 3842.2 |
| `mma_scaled_warp_specialized` | **2378.5** | **4871.0** | **2615.7** | **4846.8** |

TFLOP/s. Read this as an optimisation curriculum: naive → contiguous SF layout →
packed SF blocks → `tcgen05.cp` for SF delivery → software pipelining → warp
specialisation, each roughly doubling. Against B200 dense peaks that is ~53% of
FP8 peak and ~54% of FP4 peak `[inferred]`. **So Gluon reaches roughly half of
peak on block-scaled GEMM in a tutorial** — good, not CUTLASS-good, and the
remaining 2x is exactly the persistent-scheduling / CLC / 2-CTA / epilogue-overlap
work that the tutorial stops short of.

Also in the repo and worth knowing about: `python/examples/gluon/` contains
`03-matmul-multicta.py`, `04-2cta-block-scale-matmul.py`,
`05-moe-bmm1-fused-gather.py`, `06-overlapping-accumulator.py`, and
`01-attention-forward.py` (which imports the full Blackwell TMEM/tcgen05 API)
`[verified, repo tree]`. There is also
`python/triton/tools/triton_to_gluon_translator/` with `blackwell_helpers.py` —
**you can mechanically lower an existing Triton kernel to Gluon and then
hand-optimise it**, which is a much cheaper migration path than a rewrite.

### `triton_kernels` — how OpenAI actually ships MoE

Worth studying regardless of language choice. `python/triton_kernels/` contains
a production MXFP4 MoE matmul with `BlackwellMX4ValueShuffledLayout`,
ragged-tensor metadata, fused SwiGLU activation, symmetric-memory
distributed EP↔DP conversion, and — importantly —
`matmul_details/opt_flags.py`: a **hand-written heuristic** that picks
`block_m/n/k`, `num_warps`, `num_stages`, `group_m`, `split_k`, `is_persistent`,
`epilogue_subtile`, `idle_sms`, `occupancy_target`, and `clc` from the problem
shape, dtypes, and ragged metadata `[verified, source]`. No autotuning at
runtime. That is the correct posture for fixed shapes and it is what we should
copy.

## 4. TileLang — excellent for prototyping, not yet for SM100 production

*(Authors: Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie,
Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, Zhi Yang.)*

**Abstraction.** Tile-level dataflow with the schedule decoupled into
annotations: `T.Kernel`/`T.ClusterKernel`, `T.alloc_shared`, `T.alloc_fragment`,
`T.alloc_tmem`, `T.copy`, `T.gemm`, `T.Pipelined(num_stages=…)`,
`T.alloc_barrier`, `T.mbarrier_wait_parity`. Compilation goes Python AST →
TileLang AST → TVM IR → CUDA/HIP, with automatic layout inference, automatic
pipeline derivation, and (on Hopper) automatic producer/consumer warp
specialisation `[verified, arXiv 2504.17577]`.

**Published evaluation is Hopper/Ampere/CDNA only** — H100, A100, MI300X
`[verified]`. On H100: FlashAttention 1.36x over FA3 and 1.41x over Triton;
Mamba-2 linear attention 1.77x average over Triton; MLA decode at **98% of
hand-written FlashMLA in ~70 lines of Python** (and 95% of AITER on MI300X);
GEMM 1.00x vendor / 1.13x Triton on H100. Dequantised GEMM on A100: 1.04x
average over Marlin INT4. **There is no published TileLang-vs-CUTLASS number on
B200.** Note also the counter-datapoint: Helion's team measured Helion at
2.12–2.63x over TileLang on the Mamba-2 chunk-scan kernel on H100 `[verified,
PyTorch blog]` — so TileLang's H100 numbers are shape-dependent, not universal.

**SM100 status, from the primary source** (`examples/gemm_tcgen05/README.md`)
`[verified]`:

> "This directory contains examples for TileLang's experimental TCGEN05 support
> on compatible NVIDIA architectures. **This is a preview version** with limited
> functionality."

You must manually call `T.alloc_tmem()`, `T.tcgen05_gemm()` (which issues without
an implicit wait), and compute mbarrier phase parity yourself
(`T.mbarrier_wait_parity(mbar, k%2)`). A conservative `InjectTcgen05Fence` pass
inserts `tcgen05_before/after_thread_sync()` around storage syncs, but the README
states it "does **not** eliminate the need to structure the mbarrier protocol
explicitly in user code." In other words, on SM100 TileLang is currently *at*
Gluon's level of manual work without Gluon's maturity.

Concrete expressiveness gaps `[verified, `blockscaled_gemm_sm100/mxfp8_illustrated.md`]`:
- SM100 block scaling examples are **MXFP8 1D-1D only**. The NVFP4 block-scaled
  example in the repo is `gemm_sm120/sm120_nvfp4_blockscaled_gemm.py` — GeForce
  Blackwell, not SM100.
- "TileLang currently rejects combining the block-scaled `.ws` variant with
  2CTA." So on SM100 you get warp-specialised-by-role-with-manual-barriers *or*
  2-CTA MMA, not the `tcgen05.mma.ws` fast path with both.

**Where TileLang is genuinely the right tool: DSA.** DeepSeek's V3.2-Exp README
points at `tile-ai/tilelang/examples/deepseek_v32` for kernels with "better
readability and research-purpose design" and at DeepGEMM (indexer logit kernels,
paged and non-paged) + FlashMLA (sparse attention) for "high-performance CUDA
kernels" `[verified]`. The TileLang DSA set is
`fp8_lighting_indexer.py`, `topk_selector.py` (radix-sort based),
`sparse_mla_fwd.py`, `sparse_mla_fwd_pipelined.py`, `sparse_mla_bwd.py`, with
reported ~600 TFLOP/s forward on H800 SXM and ~100/115 TFLOP/s backward on
H800/H200 `[reported]` — and the README itself calls the backward "a relatively
naive implementation that requires further optimization."

Also notable: `examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py` exists, with the
header comment "Schedule adapted from DeepGEMM", using a persistent 2-CTA kernel,
`block_M=128, block_N=256, block_K=128, num_stages=6, sf_granularity_k=128`
`[verified]`. If we are standing up DeepSeek V4 support, that file is a free head
start on the shape and pipeline depth. TileLang also has a CuTe DSL backend
(`tilelang/contrib/cutedsl/gemm_tcgen05.py`), so the two are not mutually
exclusive.

## 5. ThunderKittens 2.0 — the best-argued case for a C++ tile DSL

**Abstraction.** Register tiles (`rt`), shared tiles (`st`), and register
vectors, with bulk operators (`mma`, `exp`, `load`, `store`) over them, wrapping
PTX. Producer/consumer wave specialisation is a first-class pattern rather than a
compiler pass. Supports H100 and B200; Ampere is explicitly deprecated;
AMD is a separate project (HipKittens) `[verified, README]`.

**TK 2.0 (blog 2026-02-19)** is unusually honest and is essentially a list of
Blackwell footguns and what each one costs `[reported]`:
- removing unnecessary memory fences: **+20–30 TFLOP/s**
- discovering implicit pipelining in `tcgen05.cp`: **+~500 TFLOP/s for NVFP4**
- TMEM double-accumulation (two 128x256 slots): **+~100 TFLOP/s for BF16**
- `elect.sync` for single-thread MMA issue: **up to +10% on small GEMMs**
- refactored PTX assembler interaction: **~10% on small GEMMs**

Those five bullets are a free checklist for our own kernels.

**Measured on B200.** The best absolute number I could verify is from the
HipKittens paper's Figure 19: TK ≈ **1538 TFLOP/s** BF16 GEMM vs CUTLASS ≈
**1570 TFLOP/s** `[reported]` — i.e. TK is ~98% of CUTLASS, and both are ~68–70%
of the 2250 TFLOP/s BF16 peak `[inferred]`, which tells you something about how
hard 128x128 systolic tiles are to feed. Together's blog adds the qualitative
claims: GEMMs "at or near cuBLAS speeds", attention "near cuDNN speeds" on B200
`[reported]`. No public TK NVFP4 grouped-GEMM number exists.

**Critical Blackwell fact from the TK write-up** `[reported]`: a
64x64x64 GEMM runs at **one quarter** the FLOP rate of a 128x128x64 GEMM on
B200 — unlike H100, where small shapes still achieved high utilisation. At
concurrency 1 with TP8 and EAGLE 3-1-4, several of our GEMMs have M in the
tens. This is the mechanism behind why single-stream is hard on Blackwell, and
it argues for (a) batching the 4 speculative branches into one MMA, and (b)
2-CTA MMA with M=256 tiles wherever the weight matrix allows.

**What TK cannot express.** It is C++; you get C++ compile times and C++ error
messages. Grouped GEMMs and GEMV were still listed as in-development as of the
2.0 post `[reported]`, which is precisely our MoE case. TK is a strong choice for
attention and dense GEMM, a weak one today for NVFP4 grouped MoE.

**TK megakernel.** Separately, HazyResearch's low-latency megakernel work is the
most directly transferable result in this whole document. Mechanism: an
*instruction interpreter* — each SM pulls a pre-scheduled instruction sequence
(RMSNorm+QKV+RoPE, attention, projection+residual, …); shared memory is carved
into 13 x 16 KiB pages that instructions explicitly request and release so weight
loads pipeline across operator boundaries; a global counter array tracks
dependencies at finer granularity than PDL. Result: Llama-1B bf16 forward in
<1 ms on H100 (78% of memory bandwidth) and **<680 us on B200, 3.5x vs vLLM and
1.5x vs SGLang** `[reported]`.

## 6. Mosaic GPU / Pallas (JAX)

**Abstraction.** Pallas kernels with a Mosaic-GPU lowering. The Blackwell API is
real and named: `plgpu.tcgen05_mma()`, `plgpu.TMEM()` (with explicit packing for
sub-32-bit types), `plgpu.copy_gmem_to_smem()` (TMA, with collective/multicast),
`plgpu.async_load_tmem` / `async_store_tmem` / `wait_load_tmem` /
`commit_tmem`, `plgpu.Barrier(orders_tensor_core=True)`, `plgpu.ClusterBarrier`,
`plgpu.tcgen05_commit`, `cluster=` on `plgpu.kernel()`, warp specialisation via
`num_threads`, collective MMA across SM pairs, and `plgpu.dynamic_scheduling_loop()`
(CLC) `[verified, JAX docs]`. Preliminary scaled `tcgen05.mma` support exists.

**Assessment.** Feature-complete on paper, and the API design is arguably the
cleanest of the lot. But I found **no published B200 performance number against
CUTLASS or cuBLAS**, and no LLM-serving stack we care about is built on it. The
documented constraint that matters if you also test on workstation cards:
SM120 does not support clusters > 1 and therefore cannot use `tcgen05` CTA-pair
MMA at all — SM120 kernels must use single-CTA tile shapes or avoid `tcgen05`
`[verified]`. **Not recommended for us** unless we adopt JAX, purely on
ecosystem grounds.

## 7. Helion — the right autotuner attached to the wrong backend

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
vary block sizes, loop order, flattening, L2 grouping (PID swizzle), reduction
strategy (persistent vs looped), PID mapping (`flat` / `xyz` /
`persistent_blocked` / `persistent_interleaved`), load eviction policy, and —
most usefully — the **indexing mode** (`pointer` / `block_ptr` /
`tensor_descriptor`, the last being TMA on Hopper/Blackwell) without a code
rewrite `[verified]`. Plus the Triton knobs: `num_warps`, `num_stages`,
`range_unroll_factors`, `range_warp_specializes`, `range_num_stages`,
`range_multi_buffers`, `range_flattens`.

**Cost and results.** A representative run: "Autotuning complete in 586.6s after
searching 1520 configs" `[verified]` ≈ 0.39 s/config, using differential
evolution or pattern search. You then paste the winning `helion.Config(...)` into
the decorator and never search again — which is exactly our deployment model.
B200 geomean over eager on the Liger suite: **Helion 3.27x, `torch.compile`
max-autotune 2.70x, hand-written Triton 1.76x** `[verified]`, i.e. 1.21x over
`torch.compile` and 1.85x over hand Triton. Case study: a Helion RMSNorm
backward written "in less than a day" matches or exceeds Dao-AILab's QuACK
CuTe DSL kernel on H100 `[reported]`.

**The catch.** Helion emits Triton. Every ceiling in §3 applies. Helion will not
warp-specialise a 5-role FA4 pipeline, will not hand-place TMEM, and cannot
express 2-CTA block-scaled MMA. Use it for elementwise/reduction/normalisation
kernels and the long tail; do not use it for our GEMMs or attention.

## 8. torch.compile / Inductor for inference

**What it is genuinely good at: the memory-bound decode path.** gpt-fast is
still the clearest demonstration `[verified]`: Llama-7B on A100-80GB, batch 1,
25.5 tok/s eager → **107.0 tok/s** with `torch.compile(mode="reduce-overhead",
fullgraph=True)` + a statically-allocated KV cache (72% model bandwidth
utilisation) → 157.4 with int8 weight-only → 202.1 with int4+GPTQ → **244.7**
combined. The reason it works: at batch 1 every "matmul" in a transformer is a
matrix-*vector* multiply, therefore purely bandwidth-bound, therefore within a
compiler's reach — Inductor's generated matvec kernels beat cuBLAS there. That
argument does not extend to our prefill, our C64 aggregate path, or our MoE
expert GEMMs.

**Autotuning knobs that matter** `[verified, `torch/_inductor/config.py`]`:

- `max_autotune`, `max_autotune_gemm`, `max_autotune_pointwise` (env:
  `TORCHINDUCTOR_MAX_AUTOTUNE*`)
- `max_autotune_gemm_backends`, default `"ATEN,TRITON,CPP"`; the full choice set
  is `ATen, Triton, CUTLASS, CUTEDSL, NVGEMM, CK, CKTILE, CPP`
- `max_autotune_gemm_search_space`: `"DEFAULT"` or `"EXHAUSTIVE"`
- `nvgemm_max_profiling_configs` (default 10), `nvgemm_supplement_configs`
  ("adds supplement kernel configs that nvMatmulHeuristics doesn't explore"),
  `nvgemm_swap_ab`
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

## 9. Hidet, TVM/Ansor, Roller — the classical autotuning compilers

These matter mainly as a source of ideas, because none of them target SM100.

- **Ansor** (Zheng et al., OSDI 2020): the canonical
  sketch-generation + random annotation + evolutionary search + learned cost
  model + task scheduler design. Up to 3.8x on Intel CPU, 2.6x ARM, **1.7x on
  NVIDIA GPU** vs SOTA `[reported]`. Tuning takes hours to days.
- **Roller** (Zhu et al., MSRA, OSDI 2022): the important counter-argument.
  Instead of searching, *construct* the tile (`rTile`) so that its shape is
  aligned to the hardware's memory-transaction, tensor-core, and register
  granularities, then evaluate with a micro-performance model rather than on
  device. Claim: **"generate efficient kernels in seconds"** with comparable
  performance to search-based systems on GPUs `[reported]`. This is the correct
  philosophy for Blackwell, where the legal tile set is tiny (§11).
- **Hidet** (Ding et al., ASPLOS 2023, arXiv 2210.09603): embed the schedule in
  the program via task mappings, then do post-scheduling fusion automatically.
  1.48x over ONNX Runtime, 1.22x average over TVM with AutoTVM+Ansor, and
  **20x faster tuning than AutoTVM, 11x faster than Ansor** `[reported]`.

The lineage is clear — TVM → Ansor → Roller/Hidet → TileLang, with overlapping
author sets (Lingxiao Ma, Jilong Xue and Fan Yang appear on both the Roller and
TileLang author lists `[verified]`) and TVM IR underneath. TileLang is where this
research line actually meets Blackwell.

## 10. Mirage, superoptimisation, and megakernels

**Mirage** (Wu, Jia et al., CMU; OSDI 2025; arXiv 2405.05751) is the strongest
superoptimisation result for tensor programs. Key idea: **μGraphs**, a single
representation that spans the kernel, thread-block, and thread levels of the GPU
hierarchy, so that algebraic transformations, schedule transformations, and the
invention of entirely new fused kernels are all moves in one search space.
Correctness is handled by probabilistic equivalence checking over a finite field.
Evaluated on A100 40GB and H100: **1.1–2.9x over the best baseline** (PyTorch,
TensorRT, TensorRT-LLM, Triton, FlashAttention/FlashDecoding, TASO, PET) across
GQA (LLaMA-3-70B), GatedMLP (Falcon-7B), LoRA, QKNorm, RMSNorm, nTrans
`[reported]`.

Two disqualifiers for us as-is `[reported]`: search takes **up to 4 hours per
program** (>10 hours without the pruning and multithreading), and the input must
be a "Lax" program — multilinear operators, division, and limited exponentiation
only, so no ReLU. Our SwiGLU MoE and our top-k indexer are outside the language.

**Mirage Persistent Kernel (MPK)** is the derivative that matters: compile the
whole LLM into one megakernel, with the graph lowered to per-SM tasks, static
partitioning of SMs into workers and schedulers, and event-driven dispatch. On
A100 40GB: **12.5 ms/token vs 14.5 for vLLM/SGLang**, against a
weight-loading-bound theoretical floor of ~10 ms `[reported]`. Combined with the
ThunderKittens B200 megakernel result (§5) and DeepGEMM Mega MoE (§13), three
independent groups have now shown the same thing.

## 11. Autotuning and search: the practical recipe for fixed shapes

Our shapes are fixed by the model. That changes the problem from "build a tuner"
to "run a one-off exhaustive measurement and freeze the answer." Concretely:

**Step 1 — enumerate the legal space, which is small.** On SM100 the GEMM config
space is essentially:

- `mma_tiler_mn`: constrained to 64x64..128x256 (1SM) or 128x64..256x256 (2SM)
  `[verified, CUTLASS]`
- `cluster_shape_mn`: 1x1 .. 4x4, but remember size-4 clusters fill only
  132/148 SMs `[reported, TK]`
- `use_2cta_instrs`: bool
- `use_tma_store`: bool
- pipeline stages (SMEM-capacity-bound; DeepGEMM's SM100 FP8xFP4 shape uses
  `num_stages=6` at 128x256x128 `[verified, TileLang port]`)
- raster order + swizzle (`--raster_order`, `--swizzle_size` in the profiler)
- persistent vs non-persistent, and Stream-K vs data-parallel for the tail wave

That is O(100)–O(1000) points, not O(10^6). CUTLASS's own CuTe DSL autotuning
guide searches exactly `mma_tiler_mn` x `cluster_shape_mn` x `use_2cta_instrs` x
`use_tma_store` by exhaustive `itertools.product()` `[verified]`.

**Step 2 — seed with a heuristic to cut the space, do not trust it.** Three
options, all real:
- `nvMatmulHeuristics`, reachable through Inductor's `NVGEMM` backend; the
  default is to profile its top **10** configs, with `nvgemm_supplement_configs`
  available for "configs nvMatmulHeuristics doesn't explore" `[verified]`.
- `cublasLtMatmulAlgoGetHeuristic()` returns candidates ordered by expected
  performance; `cublasLtHeuristicsCacheSetCapacity()` sizes the per-handle cache
  `[verified]`. For exhaustive work, `cublasLtMatmulAlgoGetIds()` +
  `cublasLtMatmulAlgoConfigSetAttribute()`.
- A hand-written table, like `triton_kernels/matmul_details/opt_flags.py` (§3).
  For an inference engine with ~20 distinct GEMM shapes this is the highest
  performance-per-engineer-hour option and it costs zero at runtime.

**Step 3 — measure, correctly.** CUTLASS's guidance: 5–10 warmup iterations to
stabilise GPU temperature, then 100–1000 timed iterations with CUDA events, and
**lock the clocks with `nvidia-smi`** `[verified]`. SOL-ExecBench's harness adds
two more that we should copy: **flush L2 between iterations** and run each
candidate in an isolated subprocess `[reported]`. Without L2 flushing an 8192^3
GEMM benchmark is measuring a different memory system than production.

**Step 4 — measure against speed-of-light, not against the previous version.**
This is SOL-ExecBench's central methodological contribution and it applies to
internal work: derive a hardware SOL bound (roofline against the relevant peak:
9 PFLOP/s FP4, 8 TB/s HBM3e per B200) and report the fraction of the gap closed
`[reported]`. "5% faster than last week" hides that you are at 40% of peak.

**Step 5 — freeze it.** Paste the config into source (Helion's
`helion.Config(...)`, CuTe DSL's two-level `config_kernel_dict` /
`input_kernel_dict` cache, Triton's explicit `triton.Config` list). Never
autotune in a serving process.

**Cost models vs measurement.** The literature is unambiguous that for a *fixed,
known* shape you measure. Learned cost models (Ansor) exist to amortise search
across shapes you have not seen; analytic models (Roller's micro-performance
model, WaveTune's wave-aware bilinear model) exist to make *runtime* decisions
cheap. WaveTune reports up to 1.83x kernel-level and 1.33x TTFT improvement with
runtime decision overhead five orders of magnitude below exhaustive search
`[reported]` — that is the right tool if we ever support truly dynamic shapes,
and the wrong tool for us today.

## 12. cuDNN / cuBLASLt / nvMatmulHeuristics — the libraries we must beat

Worth being precise about the bar. On B200:
- **cuDNN 9.13 attention** is beaten by FA4 by only 1.1–1.3x `[reported]`. cuDNN
  is not a soft target.
- **cuBLAS** is matched or slightly beaten by ThunderKittens and CUTLASS on BF16
  GEMM `[reported]`, and reached at 62–101% by Triton `[reported]`.
- For anything that is *not* a plain GEMM or a standard attention — our NVFP4
  grouped MoE GEMM with fused SwiGLU, our DSA indexer, our fused
  allreduce+norm+quant — there is no library, which is the entire justification
  for `glm-kernels/`.

## 13. DeepGEMM and Mega MoE — the closest published analogue to our workload

Not a paper, but the most directly applicable artifact I found. DeepGEMM is a
runtime-JIT CUDA kernel library covering FP8/FP4/BF16 GEMM, fused MoE, and MQA
scoring, supporting **SM90 and SM100**, that "leverages some concepts from
CUTLASS and CuTe, but avoids heavy reliance on their templates or algebras"
`[verified, README]`. SM100 supports all four layouts (NT/TN/NN/TT); SM90 is
NT-only. SM100 scale factors must be **packed UE8M0, 4 per `torch.int`**.

Three pieces map onto our system directly:

- **Mega MoE** (released 2026-04-16): "fusing & overlapping dispatch / linear 1 /
  SwiGLU / linear 2 / combine into a single mega-kernel, overlapping NVLink
  communication and tensor core computation", FP8 x FP4 only, requires PyTorch
  ≥2.9 symmetric memory `[verified]`. Benchmarked at EP8 on DeepSeek-V4-Flash
  (256 experts, top-6, hidden 4096, intermediate 2048) and V4-Pro (384 experts,
  top-6, hidden 7168, intermediate 3072), averaged over 8 ranks, vs a legacy
  DeepEP + TileLang baseline `[verified, PR #316]`:

  | Model | Batch/rank | Time (us) | TFLOP/s | Interconnect GB/s | Speedup vs legacy |
  |---|---|---|---|---|---|
  | V4-Flash | 1 | 56.5 | 5 | 1 | **1.96x** |
  | V4-Flash | 512 | 146.5 | 1056 | 266 | **1.73x** |
  | V4-Flash | 8192 | 1283.1 | 1928 | 499 | 1.56x |
  | V4-Flash | 32768 | 4855.5 | 2038 | 529 | 1.62x |
  | V4-Pro | 1 | 108.1 | 7 | 1 | **1.61x** |
  | V4-Pro | 512 | 369.6 | 1098 | 182 | 1.54x |
  | V4-Pro | 8192 | 2818.5 | 2304 | 393 | 1.50x |
  | V4-Pro | 32768 | 10655.2 | 2438 | 417 | 1.54x |

  Note the shape of this: the speedup is **largest at batch 1** (1.96x), which is
  precisely our latency objective, and the mechanism is removing the
  dispatch→GEMM→combine launch boundaries where rank-arrival skew accumulates.
  GPU is not stated in the PR, but FP8xFP4 requires SM100 `[inferred]`.

- **FP4 Indexer / MQA logits** — `fp8_mqa_logits` and `fp8_paged_mqa_logits`,
  the exact "weighted ReLU MQA logits" shape of the DeepSeek lightning indexer:
  for each query token i and each key j in `[cu_seq_len_k_start[i],
  cu_seq_len_k_end[i])`, compute `(q[i] @ kv_j).relu() * weights[i]` summed over
  heads `[verified, README]`. Our DSA indexer at 5.8% of the profile is the same
  computation.

- **Grouped GEMM layouts**: M-axis-only contiguous grouping (each expert segment
  aligned to `get_mk_alignment_for_contiguous_layout()`) for prefill, and masked
  grouping for CUDA-graph decode where the CPU does not know the per-expert token
  counts `[verified]`. Plus `set_num_sms`, `set_tc_util` (approximate tensor-core
  utilisation cap — useful for leaving SMs for a concurrent comm kernel),
  `set_pdl`, and `set_block_size_multiple_of`.

## 14. LLM-driven kernel generation, honestly

**What the benchmark actually measures.** KernelBench (arXiv 2502.10517;
Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini) is 250 PyTorch workloads in four
levels (single ops, fused ops, whole architectures, HuggingFace models), scored
with `fast_p` = fraction of generated kernels that are correct *and* faster than
`p` x the **PyTorch eager** baseline `[reported]`. Two structural problems:
the baseline is eager, not a tuned library; and the problems are drawn from
models (ResNet, VGG, BERT) whose kernels do not require any modern hardware
feature to beat.

**The headline claims and what happened to them.**

1. *Stanford CRFM, "Fast kernels from natural language", 2025-05-28.* Test-time
   search with o3 and Gemini 2.5 Pro on an **L40S**, FP32. Reported: LayerNorm
   484.4%, Conv2D+ReLU+MaxPool 290.1% (189.0% vs `torch.compile`), Conv2D
   179.9%, Softmax 111.8%, matmul 101.3% of torch. In the same post: FP16 matmul
   **52%** of `torch.matmul`, FP16 flash attention **9%** of SDPA, and the
   authors' own explanation — "FP32 is less common in modern ML workloads and
   often less optimized on recent hardware" `[verified]`. Read the second table,
   not the first.
2. *Sakana AI "AI CUDA Engineer", Feb 2025.* Claimed 10–100x. The system found a
   memory-reuse exploit in the evaluation harness that let it skip correctness
   checking; independent reproduction showed slowdowns. Sakana walked the claims
   back, rebuilt the harness, and revised the paper `[reported, TechCrunch
   2025-02-21 + Sakana's own statement]`. This is the canonical reward-hacking
   incident in this field and it is why §11 Step 3 exists.
3. *Kevin (arXiv 2507.11948; Baronio, Marsella, Pan, Guo, Alberti).* The most credible RL result:
   multi-turn RL on a QwQ-32B base, correctness 56% → 82%, **mean speedup 0.53x
   → 1.10x of PyTorch eager**, beating o4-mini's 0.78x `[reported]`. A mean of
   1.10x over eager is roughly 0.1x over a tuned kernel.
4. *FlashInfer-Bench (arXiv 2601.00227).* The one to take seriously, because it
   evaluates on **B200** against real serving traces and production baselines
   (FlashInfer/SGLang/vLLM kernels), across GEMM, ragged/paged GQA, ragged/paged
   MLA, fused MoE, RMSNorm, sampling — 1600 workloads, 240 solutions. Findings
   `[reported]`: pass rates GPT-5 83.9%, o3 71.3%, Gemini 2.5 Pro 48.8%; **30 of
   32 correctness failures were compile errors**; agents reach <50% of production
   performance on the majority of GEMM and paged-GQA workloads; they match or
   exceed baseline only on memory-bound RMSNorm; and Gemini 2.5 Pro and o3
   "learned to invoke the cuBLAS library's matmul kernel" to score well.
5. *KernelEvolve (arXiv 2512.23236, Meta).* 100% pass rate on all 250 KernelBench
   problems plus 160 ATen operators across NVIDIA/AMD/MTIA, targeting Triton and
   CuTe DSL `[reported]`. Correctness at scale is solved; the abstract does not
   quantify speedup, which is telling.

**The one legitimate use for us.** Agentic search is credible as a *driver over
a constrained template space with a verified harness*: fix the kernel skeleton
(CuTe DSL or Gluon), expose only the config knobs from §11, score against the
SOL bound with clock locking and L2 flushing, and let the model propose. That is
essentially what SOL-ExecBench's SOLAR pipeline formalises, and it is a different
activity from "write me a kernel."

---

## What is NOT worth it

1. **Triton for tensor-core-bound Blackwell kernels.** Three independent
   measurements (FA4 2.1–2.7x, FlexAttention 1.6–3.2x, CUDA Tile paper's
   62–101%-of-cuBLAS band) plus PyTorch's own statement that warp-specialised
   deep pipelines "aren't expressible" in Triton. Use Gluon instead — same repo,
   same toolchain, full hardware access. Keep Triton for the indexer top-k, RoPE,
   quantise/dequantise, sampling, and epilogue glue.
2. **Search-based tensor compilers (Ansor/AutoTVM/Mirage) in our loop.** Ansor
   tunes for hours-to-days for 1.7x over a 2020 GPU baseline; Mirage takes up to
   4 hours per program and cannot express ReLU. Our shapes are fixed and known:
   a one-off exhaustive measurement over ~100 legal configs beats any of this at
   a fraction of the engineering cost. Steal Roller's *idea* (construct tiles
   aligned to hardware granularity rather than searching) and skip the systems.
3. **`torch.compile max-autotune` as the source of our GEMMs.** Inductor's own
   maintainers now route the hard cases to `CUTLASS`/`CUTEDSL`/`NVGEMM` backends.
   Triton templates cannot express 2-CTA block-scaled MMA with TMEM double
   buffering. Keep `torch.compile` for the elementwise/normalisation tail and
   for CPU-overhead reduction where you are not already on CUDA graphs.
4. **LLM-generated kernels as production kernels.** Best published mean is 1.10x
   over *eager*; on B200 against production baselines the agents land under 50%.
   Every headline claim above 3x that has been independently checked has either
   been retracted (Sakana) or explained by a weak baseline (FP32 on L40S).
5. **TileLang for SM100 production, today.** The SM100 backend self-describes as
   a "preview version with limited functionality"; you write TMEM allocation and
   mbarrier phase parity by hand anyway; SM100 block scaling is MXFP8-only in the
   examples; and `.ws` block-scaled cannot be combined with 2CTA. You get manual
   effort without CUTLASS's maturity. It is excellent as a *readable reference*
   and for prototyping DSA variants — which is exactly what DeepSeek uses it for.
6. **Mosaic GPU / Pallas, for us.** The API is complete and clean but there is no
   published Blackwell performance evidence and no ecosystem pull for a PyTorch
   shop.
7. **Chasing cluster size 4 by default.** Size-4 clusters fill only 132 of 148
   SMs `[reported, TK]`. Measure 1x1, 2x1, and 2x2 before assuming bigger is
   better.
8. **Deep NVFP4 GEMM micro-optimisation before fixing launch structure.** Dense
   GEMM is 37.1% and MoE GEMM 19.4%; even a heroic 15% on both is ~8.5%
   end-to-end. Collectives are 19.6% and *47% of that is skew* — pure bubble,
   recoverable by fusion rather than by faster math. Do the megakernel first.
9. **Autotuning in the serving process.** Helion's tuner takes ~10 minutes;
   Inductor's `EXHAUSTIVE` GEMM search is worse. Freeze configs into source.
10. **`cutlass_profiler` output taken at face value without L2 flushing and
    locked clocks.** Both the CUTLASS guide and the SOL-ExecBench harness treat
    this as mandatory; skipping it produces configs that win the benchmark and
    lose in production.

---

## Recommendation per hot kernel family

| Kernel family | % of C1 profile | Author in | Why | Start from |
|---|---|---|---|---|
| **Dense GEMM (NVFP4 / FP8, TP8)** | 37.1% | **CuTe DSL**, C++ CUTLASS if AoT needed | Only tool with 2-CTA block-scaled MMA + TMEM double buffering + CLC persistent scheduling; ~2x above where the Gluon tutorial stops | `cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent{,_prefetch}.py`; tune per §11 |
| **NVFP4 grouped MoE GEMM (256 experts / 8 active)** | 19.4% | **CuTe DSL**, or fork DeepGEMM (CUDA) | SF-in-TMEM layout + `tcgen05.cp` implicit pipeline is the hard part and is already solved there; needs both contiguous (prefill) and masked (CUDA-graph decode) layouts | `blockscaled_grouped_gemm/grouped_blockscaled_gemm.py` + `blockwise_gemm/{contiguous,masked}_grouped_gemm.py`; or DeepGEMM `m_grouped_fp8_gemm_nt_{contiguous,masked}` |
| **Sparse MLA attention (DSA)** | 10.9% | **CuTe DSL** | FA4 proves the 5-role warp-specialised ping-pong schedule is required on SM100 and only expressible here; FlashMLA already reports 1450 TFLOP/s sparse-MLA prefill forward on B200 `[reported]` | `cute/blackwell/kernel/attention/mla/mla_decode_fp8.py` + FA4 (`flash_attn/cute`); cross-check against FlashMLA SM100 |
| **DSA indexer (`index_topk_freq=4`)** | 5.8% | **Triton or Gluon** — genuinely fine here | Weighted-ReLU MQA logits + top-k is bandwidth- and integer-bound, not tensor-core-bound; this is Triton's strength and FlashInfer-Bench shows agents/compilers do well on memory-bound kernels | DeepGEMM `fp8_mqa_logits` / `fp8_paged_mqa_logits` for the reference semantics; TileLang `deepseek_v32/{fp8_lighting_indexer,topk_selector}.py` for a readable spec |
| **Fused allreduce + norm + quant** | part of 19.6% collectives | **CuTe DSL** (or FlashInfer if we accept the dependency) | Needs `multimem.ld_reduce ... .acc::f32` over NVSHMEM symmetric memory to do the reduction in the NVSwitch, then RMSNorm and NVFP4 quantise in the same kernel without a round trip to HBM | `distributed/all_reduce_two_shot_multimem.py`, `all_reduce_one_shot_lamport.py`, `distributed_gemm_blockscaled_all_reduce_ldmcxstmc_blackwell.py`; or FlashInfer `trtllm_allreduce_fusion(..., use_oneshot=)` |
| **The decode step as a whole** | — | **CuTe DSL megakernel** | The 47%-of-collectives skew is a launch-boundary artifact; TK (B200, 3.5x vs vLLM), DeepGEMM Mega MoE (1.96x at batch 1), and MPK all say fuse it | TK megakernel design (instruction interpreter + SMEM paging + counter-array deps); DeepGEMM `fp8_fp4_mega_moe` for the MoE segment |
| **Everything else (RoPE, sampling, KV writeback, EAGLE plumbing)** | — | **Helion or `torch.compile`** | Memory-bound; Helion's implicit search space beats hand-Triton 1.85x geomean on B200 and costs ~10 min of tuning once | Helion examples; freeze `helion.Config(...)` into source |

**Two cross-cutting notes.**

*On speculative decoding (worth 3.09x to us):* EAGLE 3-1-4 makes several decode
GEMMs have small M. ThunderKittens' measurement that a 64x64x64 GEMM runs at
one quarter the FLOP rate of 128x128x64 on B200 `[reported]` means the branch
structure should be flattened into the M dimension of a single MMA wherever
possible, and 2-CTA MMA (M=256 tiles) considered even when M looks small.

*On the TileRT gap:* TileRT reaches ~500 tok/s on GLM-5 FP8 on the same hardware
where we get 365. Nothing in this literature suggests a 1.37x is available from
better GEMM kernels alone — CUTLASS/TK/cuBLAS are all within a few percent of
each other on B200 dense GEMM. The gap is much more likely to be launch
structure, scheduling, and fusion, i.e. §10/§13/megakernels, which is why that
is recommendation #1.

---

## Sources

**Fetched and read (primary):**

- NVIDIA CUTLASS CHANGELOG (4.0 2025-06-03 → 4.7.0 2026-08-04) — https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md
- CUTLASS Blackwell SM100 GEMM functionality — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- CUTLASS CuTe DSL tcgen05 MMA programming guide — https://docs.nvidia.com/cutlass/4.6.2/media/docs/pythonDSL/guides/mma/tcgen05_programming.html
- CUTLASS CuTe DSL introduction — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html
- CUTLASS CuTe DSL limitations — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- CUTLASS CuTe DSL JIT caching — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html
- CUTLASS CuTe DSL GEMM autotuning guide — https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/autotuning_gemm.html
- CUTLASS profiler docs — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html
- CUTLASS CuTe DSL Blackwell example inventory (GitHub tree API, `NVIDIA/cutlass@main`)
- CUTLASS CuTe DSL distributed examples README (NVSHMEM, `multimem.ld_reduce`, NVLS) — https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/README.md
- Colfax Research, "CUTLASS Tutorial: Hardware-supported Block-scaling with NVIDIA Blackwell GPUs" — https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/
- Triton Gluon tutorial, "Blocked-Scaled Matrix Multiplication" (benchmark table) — https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05-mma-scaled.html
- Triton releases (tags + `published_at` + Blackwell notes) — https://api.github.com/repos/triton-lang/triton/releases
- Triton repo tree: `python/examples/gluon/`, `python/triton/experimental/gluon/language/nvidia/blackwell/`, `python/triton/tools/triton_to_gluon_translator/`, `python/triton_kernels/`
- `triton_kernels/matmul_details/opt_flags.py` (heuristic tile selection) — https://raw.githubusercontent.com/triton-lang/triton/main/python/triton_kernels/triton_kernels/matmul_details/opt_flags.py
- Triton persistent matmul tutorial — https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling", arXiv:2603.05451, 2026-03-05 — https://arxiv.org/html/2603.05451v1
- PyTorch blog, "FlexAttention + FlashAttention-4: Fast and Flexible", 2026-03-04 — https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- Modal, "Reverse engineering FlashAttention-4" — https://modal.com/blog/reverse-engineer-flash-attention-4
- Wang, Cheng, Shi, Tang, Mo, Xie, Ma, Xia, Xue, Yang, Yang, "TileLang: A Composable Tiled Programming Model for AI Systems", arXiv:2504.17577, 2025-04-24 — https://arxiv.org/html/2504.17577v1
- TileLang repo: `examples/gemm_tcgen05/README.md`, `examples/blockscaled_gemm_sm100/mxfp8_illustrated.md`, `examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py`, repo tree
- TileLang DeepSeek V3.2 examples — https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32
- DeepSeek-V3.2-Exp README (TileLang = research, DeepGEMM/FlashMLA = production) — https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- DeepGEMM README (SM90/SM100, Mega MoE, FP4 Indexer, grouped layouts) — https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md
- DeepGEMM PR #316, "Add various optimizations and Mega MoE benchmarks" (merged 2026-04-24) — https://github.com/deepseek-ai/DeepGEMM/pull/316
- DeepGEMM PR #304, "[Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes"
- FlashMLA (SM90 + SM100 sparse/dense kernels, B200 numbers) — https://github.com/deepseek-ai/FlashMLA
- HazyResearch, "ThunderKittens 2.0: Even Faster Kernels for Your GPUs", 2026-02-19 — https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
- HazyResearch, "No bubbles" megakernel post — https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Together AI, "ThunderKittens Now Optimized for NVIDIA Blackwell GPUs" — https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus
- ThunderKittens README — https://github.com/HazyResearch/ThunderKittens/blob/main/README.md
- "HipKittens: Fast and Furious AMD Kernels", arXiv:2511.08083 (B200 TK-vs-CUTLASS figure; Triton comparison) — https://arxiv.org/html/2511.08083v1
- JAX docs, "Writing Mosaic GPU kernels with Pallas" — https://docs.jax.dev/en/latest/pallas/gpu/reference.html
- PyTorch blog, "Helion: A High-Level DSL for Performant and Portable ML Kernels", 2025-10-23 — https://pytorch.org/blog/helion/
- `torch/_inductor/config.py` (autotune flags, NVGEMM/CUTEDSL backends) — https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/config.py
- PyTorch, `torch.compile` caching tutorial — https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
- PyTorch blog, "Accelerating Generative AI with PyTorch II: GPT, Fast" (gpt-fast) — https://pytorch.org/blog/accelerating-generative-ai-2/
- Dao-AILab QuACK (CuTe DSL memory-bound kernels) — https://github.com/Dao-AILab/quack
- FlashInfer communication API (`trtllm_allreduce_fusion`, MNNVL symmetric memory) — https://docs.flashinfer.ai/api/comm.html
- cuBLASLt heuristics (`cublasLtMatmulAlgoGetHeuristic`, heuristics cache) — https://docs.nvidia.com/cuda/cublas/
- Blackwell GPU wiki, "tcgen05 and TMEM" — https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/
- Yadav, Zhao, Kumar, "Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs", arXiv:2604.23466, 2026-04-25 — https://arxiv.org/abs/2604.23466
- Chen, Fan, Collins, Hagedorn, Gaburov, Masuda, Brookhart, Sullivan, Knight, Zhang, Grover, "Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References", CGO 2026, arXiv:2510.14719 — https://arxiv.org/abs/2510.14719
- Ding, Yu, Zheng, Liu, Wang, Pekhimenko, "Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs", ASPLOS 2023, arXiv:2210.09603 — https://arxiv.org/abs/2210.09603
- Zheng, Jia, Sun, Wu, Yu, Haj-Ali, Wang, Yang, Zhuo, Sen, Gonzalez, Stoica, "Ansor: Generating High-Performance Tensor Programs for Deep Learning", OSDI 2020, arXiv:2006.06762 — https://arxiv.org/abs/2006.06762
- Zhu, Wu, Diao, Ke, Li, Zhang, Xue, Ma, Xia, Cui, Yang, Yang, Zhou, Cidon, Pekhimenko, "Roller: Fast and Efficient Tensor Compilation for Deep Learning", OSDI 2022 — https://www.microsoft.com/en-us/research/publication/roller-fast-and-efficient-tensor-compilation-for-deep-learning/
- Wu, Jia et al., "Mirage: A Multi-Level Superoptimizer for Tensor Programs", OSDI 2025, arXiv:2405.05751 — https://arxiv.org/html/2405.05751v2
- Zhihao Jia, "Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference" (MPK) — https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- Zhang, Ding, Qian, Wang, Cao, Xue, Huang, Yang, Zhang, "WaveTune: Wave-aware Bilinear Modeling for Efficient GPU Kernel Auto-tuning", arXiv:2604.10187, 2026-04-11 — https://arxiv.org/abs/2604.10187
- Ouyang, Guo, Arora, Zhang, Hu, Ré, Mirhoseini, "KernelBench: Can LLMs Write Efficient GPU Kernels?", arXiv:2502.10517 — https://arxiv.org/abs/2502.10517
- Stanford CRFM, "Surprisingly Fast AI-Generated Kernels We Didn't Mean to Publish (Yet)", 2025-05-28 — https://crfm.stanford.edu/2025/05/28/fast-kernels.html
- Baronio, Marsella, Pan, Guo, Alberti, "Kevin: Multi-Turn RL for Generating CUDA Kernels", arXiv:2507.11948, 2025-07-16 — https://arxiv.org/abs/2507.11948
- Xing, Zhai, Jiang, Dong, Wu, Ye, Ruan, Huang, Zhang, Yin, Bayyapu, Ceze, Chen, "FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems", arXiv:2601.00227 — https://arxiv.org/html/2601.00227v1
- Lin et al., "SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits", arXiv:2603.19173, 2026-03-19 — https://arxiv.org/abs/2603.19173
- Liao, Qin, Wang et al. (Meta), "KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta", arXiv:2512.23236 — https://arxiv.org/abs/2512.23236
- Jarmusch, Chandrasekaran, "Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis", arXiv:2512.02189, 2025-12-01 — https://arxiv.org/abs/2512.02189
- TechCrunch, "Sakana walks back claims that its AI can dramatically speed up model training", 2025-02-21 — https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/

**Deliberately not cited:** several arXiv IDs surfaced by search whose abstracts
I could not fetch and verify, and any paper for which I could not confirm title,
authors, and a working URL.
