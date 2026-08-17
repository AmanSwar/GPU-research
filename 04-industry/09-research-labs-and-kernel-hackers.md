# HazyResearch, Colfax, and the kernel-hacker community: megakernels, ThunderKittens and low-level craft

## What this is

A mining pass over the research groups and individuals who publish *mechanism* rather than
marketing: Stanford HazyResearch (ThunderKittens / Megakernels / ParallelKittens), Colfax
Research (the de-facto public documentation for CUTLASS on Blackwell), CMU Catalyst and its
orbit (Mirage/MPK, Event Tensor, FlashInfer), plus the microbenchmark literature on SM100 and
the individual bloggers and lecture series that the field actually reads.

Everything below was fetched and read during this session. Labels are used throughout:

- **[verified]** — I fetched the URL given and read the content.
- **[reported]** — the authors claim it; I read the claim but nobody independent has reproduced it.
- **[inferred]** — my own arithmetic or reasoning, marked as such.
- **[unverified]** — I could not source it.

Two important caveats before the content:

1. This session's WebSearch budget was exhausted early (200/200 calls consumed before my first
   search returned), so everything here came from direct WebFetch against URLs I could name or
   navigate to, plus the arXiv API. Coverage of the *long tail* of individual bloggers is
   therefore thinner than coverage of the primary labs. Where I could not source something, I
   say so rather than padding.
2. Almost every number here is a vendor-or-author self-report on their own kernel against a
   baseline they chose. Config asymmetries are noted inline. Nothing here has been reproduced
   on our hardware.

---

## Bottom line for our system

Ranked by (expected effect on our two objectives) × (confidence) ÷ (difficulty). Our measured
C1 hotspots — dense GEMM 37.1%, collectives 19.6% of which 47% is rank arrival skew, MoE expert
GEMMs 19.4%, attention 10.9%, DSA indexer 5.8% — drive the ordering.

| # | Steal | Source | Expected effect | Difficulty |
|---|---|---|---|---|
| 1 | **Replace NCCL for the TP all-reduce in decode with a device-initiated one-shot kernel** using `multimem.ld_reduce`/`multimem.red` on pre-allocated destination buffers, no intermediate staging, no two-way handshake | ParallelKittens §3.1.4 + PGL post | PK measures **1.79× over NCCL** on 8×B200 BF16 all-reduce at their *smallest* tested size and the curve is still rising as size falls; our decode all-reduce is ~3 orders of magnitude smaller than their smallest point. Collectives are 19.6% of C1 | Medium — needs VMM multicast setup (`cuMemCreate` + FD exchange), ~100 lines of device code |
| 2 | **Kill rank arrival skew with fine-grained counter semaphores instead of collective barriers** — per-tile `notify()`/`wait()` on global counters, so a rank that arrives early starts consuming instead of blocking | HazyResearch megakernel; ETC static scheduler | Skew is 47% of collectives = **9.2% of C1 wall clock**. PK measures inter-SM sync through HBM at **832 ns** vs **64 ns** for an intra-SM mbarrier — the barrier itself is not free either | Medium |
| 3 | **Make the O-projection data-parallel (replicated) instead of tensor-parallel**, eliminating the post-attention reduce-scatter entirely | HazyResearch TP megakernel | They report **8× reduction in network traffic** for ~9 GB extra weights per GPU. We have 183 GB/GPU — we can afford it. Removes one collective per layer from the C1 critical path | Low-Medium — a parallelism change, not a kernel change |
| 4 | **Three free wins inside our NVFP4/FP8 GEMM kernels**: (a) issue `tcgen05.cp` from the same thread as `tcgen05.mma` and rely on the *implicit* tcgen05 pipeline instead of an explicit barrier; (b) delete `fence.proxy` where PTX causality ordering already covers you; (c) use `elect.sync` rather than a lane-id compare for single-thread instructions | ThunderKittens 2.0 | Authors measured **~500 TFLOP/s (≈10% of NVFP4 GEMM)** for (a), **~20 TFLOP/s** for (b), **up to 10% on small-shape GEMMs** for (c). Dense GEMM is 37.1% of C1 and small-shape is exactly our decode regime | Low, *if* we own the GEMM kernel |
| 5 | **Audit our cluster launch dims** — on B200 (148 SMs) `__cluster_dims__` of 4 strands the grid to 132 SMs, 8 → 120, 16 → 112, unless you configure via `cudaLaunchKernelEx` | ThunderKittens 2.0 | Up to **24% of the machine** silently idle. Trivially checkable with one nsys run | Trivial |
| 6 | **Software-emulate `exp2` for attention softmax** — Cody-Waite range reduction + degree-3 Horner polynomial on the FMA units, splitting the exponential work between `MUFU.EX2` and CUDA cores | FlashAttention-4 | On B200 the exp unit is **16 ops/cycle/SM against 8192 tensor-core ops/cycle** and did not scale from Hopper; FA4 reaches **1605–1613 TFLOP/s (71% util)** forward, 1.1–1.3× cuDNN 9.13. Attention is 10.9% of C1 and DSA indexer another 5.8% | Medium |
| 7 | **Adopt the FlexAttention-in-CuTe-DSL block-sparse plumbing for the DSA `index_topk` mask** — `mask_block_idx` / `full_block_idx` + counts, so fully-masked tiles are never loaded and fully-unmasked tiles skip `mask_mod` | Colfax FlexAttention guide | Reported **95% of FA3 forward performance** with arbitrary `score_mod`/`mask_mod`, ~50% faster than Triton FlexAttention. Directly matches our `index_topk_freq=4` structure | Medium |
| 8 | **Wave-quantization discipline on decode-shape and MoE grouped GEMMs**: Stream-K or Blackwell CLC (`clusterlaunchcontrol.try_cancel`) | Colfax persistent/Stream-K + CLC posts | Static persistent scheduling shows **30–40% throughput cliffs at wave boundaries**; CLC wins big on *imbalanced* work (grouped GEMM with varying K) and roughly ties on balanced work. Our MoE 256/8 routing is inherently imbalanced | Medium |
| 9 | **Warp-specialized epilogue store warp + scale-factor SMEM bank-conflict audit** on the NVFP4 GEMM | Colfax SM120 NVFP4 walkthrough | Their SFA layout change from stride `(32,4):(16,4)` to `(32,4):(4,128)` removed **8.39M excessive wavefronts**; threadblock swizzle=16 was worth **+387 TFLOP/s** at 32k via L2 hit-rate 76%→90%+ | Low |
| 10 | **Megakernel the decode step** (or at minimum fuse the per-layer chain behind one persistent launch with an on-GPU instruction queue) | HazyResearch Megakernels, MPK, ETC | ETC measures **1.48× vs vLLM / 1.20× vs SGLang TPOT at batch 1 on B200** for Qwen3-30B-A3B, and isolates **6–8%** as coming purely from fine-grained pipelining with *identical operator code*. Kernel launch is 5–10 µs against 2 µs kernels | High |
| 11 | **Use a static schedule, not a dynamic work queue, at low batch** | ETC Table 3 | Explicit negative result: on Qwen3-32B TP=4 the dynamic scheduler is **0.83× at BS=1** while static is 1.09×. Dynamic only wins on MoE at ≥128 tokens. This contradicts HazyResearch's global-work-queue result *at their batch 8192* — the regimes differ | Low (a config choice) |
| 12 | **ThunderMLA-style fused decode attention with a CPU-side scheduler reused across layers** — partial + reduction in one kernel, k-way reduction tree synthesized on the host | ThunderMLA | 20–36% over FlashMLA on H100 across batch/seqlen mixes, largest win at small batch × short seq (**36.3%** at B64/S512/Q2). Scheduler cost 1–2 ms amortized over hundreds of forward passes | Medium |

**One derived number worth having.** ParallelKittens gives a closed form for when
communication is fully hidden by compute in a fused GEMM+reduce-scatter: `K ≥ sR/(2B)` where
`s` = bytes/element, `R` = sustained tensor-core FLOP/s, `B` = per-GPU NVLink bytes/s. They
verify `K ≳ 2197` for BF16 on H100. **[inferred]** For B200 with `B = 900 GB/s`: BF16
(`s=2, R≈2.25e15`) → K ≥ 2500; FP8 (`s=1, R≈4.5e15`) → K ≥ 2500; NVFP4 (`s=0.5, R≈9e15`) →
K ≥ 2500. The threshold is **dtype-invariant at ~2500** because tensor-core rate scales
inversely with element size. Practical consequence for us: in *prefill* at ~10k tokens our
K dimensions comfortably exceed 2500, so tile-level intra-SM overlap should hide the
collective almost entirely. In *decode at C1* `M=1`, there is no compute to hide behind —
the collective is pure latency, which is precisely why steal #1 (one-shot / multimem, no
handshake) matters more than any overlap scheme.

---

## Stanford HazyResearch

### What they run

An academic group (Chris Ré's lab) that ships production-grade CUDA and writes it up in
unusual detail. Their stack is: **ThunderKittens** (a tile-based embedded C++ DSL),
**Megakernels** (an on-GPU interpreter for whole-model fusion), **ParallelKittens** (multi-GPU
extensions), and **HipKittens** (the AMD port). Everything is open source under the
HazyResearch GitHub org. ParallelKittens states it "is currently being adopted at Cursor for
large-scale in-house training" [verified, arXiv:2511.13940].

### Technique 1 — ThunderKittens: tile primitives, and what it costs you

**Mechanism.** Three levels of abstraction, all built on a minimum 16×16 tile:

- Register tiles (`rt_bf<32,16>` etc.) — warp-level objects split across thread registers
- Shared tiles (`st_bf<128,64>`) — block-level, with swizzling baked in for bank-conflict avoidance
- Warpgroups — 4-warp collectives for async MMA and collective memory ops

Blackwell adds a tensor-memory tile type. The published API for a 2-SM UMMA is literally:

```cpp
tt<float, 128, 128> d;              // 128x128 fp32 tile living in TMEM
__shared__ st_bf<128, 64> a, b;     // 128x64 bf16 shared tiles
mma<transpose::N, transpose::T>(d, a, b, sem);   // ncta=2 template param triggers CTA-pair
```

**Evidence.** README claims ~855 TFLOPs BF16 matmul on H100 (86% of theoretical max)
[reported, https://github.com/HazyResearch/ThunderKittens]. Requires CUDA 12.8+, C++20,
gcc-11/clang-11 minimum. Kernel directories present: `attention`, `gemm`, `based`,
`linear_attention`, `mamba2`, `layernorm`, `rotary`, `fftconv`, `flux`, `parallel`
[verified, /tree/main/kernels]. Ampere is explicitly no longer maintained.

### Technique 2 — Blackwell-specific findings (TK 1.0 Blackwell post, Mar 2025)

**[verified, https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell]**

- **The 5th-gen tensor core behaves like a 128×128 systolic array.** M and N should be 128 or
  multiples thereof; 64×64 gets proportionally reduced throughput. This is the single most
  important shape rule for porting Hopper kernels.
- TMEM is 256 KB **in addition to** the 256 KB register file and up to 227 KB SMEM — deeper
  pipelines are possible without SRAM pressure.
- CTA pairs: two CTAs on paired SMs can jointly drive the tensor core and access each other's TMEM.
- Reported: BF16 and FP8 GEMM at cuBLAS parity on B200 (2× H100 cuBLAS); attention fwd/bwd near
  cuDNN on B200, up to 2× FA3-on-H100 [reported — no shapes given, treat as directional].

### Technique 3 — ThunderKittens 2.0 (Feb 2026): the highest-density list of free wins I found

**[verified, https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2]**

This post is essentially a changelog of micro-optimizations with the TFLOP/s attached to each.
It is the single most directly stealable document in this entire report.

| Finding | Mechanism | Measured |
|---|---|---|
| **`tcgen05.cp` is implicitly pipelined with `tcgen05.mma`** when issued from the same thread — no explicit sync needed. The authors note the official docs obscured this with a typo referring to `tcgen05.copy` | Remove the barrier between scale-factor copy and MMA | **~500 TFLOP/s, ≈10% of NVFP4 GEMM** |
| **Unnecessary memory fences** — PTX *causality ordering* via TMA's implicit mbarrier ops and mbarrier try-wait acquire semantics already establishes the needed ordering; the explicit `fence.proxy` was redundant | Delete the fence | **~20 TFLOP/s** |
| **PTX assembler serialization** — for single-thread instructions on SM90+, a lane-id compare makes ptxas conservatively emit a loop executed across all 32 lanes. `elect.sync` (TK's `elect_leader()`) does not | Swap the predicate | **up to 10% on small-shape GEMMs** |
| **Tensor-memory double accumulation pattern** | (mechanism not detailed in the post) | **~100 TFLOP/s** |
| **Cluster size strands SMs** — B200 has 148 SMs; with `__cluster_dims__`, cluster 4 → 132 SMs used, 8 → 120, 16 → 112. Needs `cudaLaunchKernelEx` configuration to recover | Check launch config | up to **24% of the GPU** |
| **TMEM hard-limits occupancy to 1 block/SM** regardless of shared-memory allocation | Do not plan for multi-block occupancy on Blackwell GEMM | — |

Also: 2.0 adds MXFP8/NVFP4 blockscaled GEMM, CLC scheduling support, and controllable tensor
memory, and claims BF16/MXFP8/NVFP4 GEMM matching-or-exceeding cuBLAS on B200 [reported].
Their stated benchmarking protocol — bitwise-identical random inputs, multiple input groups to
force natural L2 eviction, 500 warmup / 100 profiling iterations, CUDA events with no
intermediate sync — is worth copying verbatim for our own A/B harness.

**[inferred]** If 500 TFLOP/s is "roughly 10%" of their NVFP4 GEMM, their NVFP4 GEMM runs at
roughly **5 PFLOP/s dense on B200**. Against a dense NVFP4 peak in the ~9 PFLOP/s range that is
~55% of peak, which is a plausible real number rather than a marketing one. Treat the
arithmetic as mine, not theirs.

### Technique 4 — The low-latency Llama megakernel (May 2025)

**[verified, https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles]**
Repo: https://github.com/HazyResearch/Megakernels

This is the reference design for batch-1 latency and maps closely onto our C1 objective.

**Mechanism — an on-GPU interpreter.** Each SM runs a loop that pops instructions from a
pre-computed per-SM queue. All instructions share one CUDA template. The schedule is computed
in **Python on the host** and *reused across hundreds of forward passes*, so scheduling cost is
amortized to nothing. Llama-1B decomposes into **seven** fused instruction types:

1. RMSNorm + QKV + RoPE
2. Attention
3. Attention reduction (`ThunderGQA`)
4. O-projection + residual
5. RMSNorm + up/gate + SiLU
6. Down-projection + residual
7. RMSNorm + LM head

**Mechanism — page-based shared memory.** The first 213 kB of H100 SMEM is carved into **13
pages of 16 KiB**. Instructions explicitly request and release pages. Critically: *the
interpreter hands a released page to the next instruction immediately*, so the next instruction
can begin issuing loads while the previous one is still storing results. This is the
cross-instruction pipelining that removes the bubble.

**Mechanism — counter semaphores in global memory.** Completion increments a counter; a
starting instruction waits for target values. They chunk the MLP intermediate into **four**
pieces with individual counters, so a down-projection instruction waits only on *its* input
chunk rather than the whole MLP.

**Evidence and numbers** (Llama-3.2-1B, bf16, batch size 1, single GPU):

| Hardware | Megakernel forward pass | vs vLLM | vs SGLang | Mem BW util |
|---|---|---|---|---|
| H100 | ~1.00 ms | ~2.5× faster | >1.5× faster | 78% |
| B200 | ~680 µs | >3.5× faster | >1.5× faster | — |

Their published B200 breakdown of a ~600 µs pass: 250 µs store/load activations +
synchronization; 200 µs RMSNorm and matrix-vector compute (95% of which is the matvec); 30 µs
weight loading; 40 µs synchronization overhead; 80 µs setup/misc.

**What this tells us.** Even in a fully fused megakernel at batch 1 on B200, **~48% of the time
is activation movement and synchronization, not compute**. That is the same shape as our
collectives+skew line item. The lesson is not "fuse everything and compute goes to 100%" — it
is "fusing exposes that your real enemy is data movement and barriers."

Repo build knobs: `THUNDERKITTENS_ROOT`, `MEGAKERNELS_ROOT`, `PYTHON_VERSION`, and
`GPU={H100,B200}` with **B200 as the default** [verified, README]. Entry points
`megakernels/scripts/llama_repl.py` and `megakernels/scripts/generate.py`.

### Technique 5 — The tensor-parallel megakernel (Sep 2025)

**[verified, https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main]** and the
companion intro post `2025-09-28-tp-llama-intro`.

8×H100, Llama-70B, 16-bit. Nine instruction types. Three levels of overlap, each ablated:

| Mechanism | With | Without | Delta |
|---|---|---|---|
| Cross-instruction pipelining (loader/consumer/storer warp specialization) | 31,516 TPS | 29,607 TPS | **+6.1%** |
| Global work queue (dynamic instruction assignment via atomic increment, vs static per-SM queues) | 31,516 TPS | 27,033 TPS | **+14.2%** |
| Instruction-type interleaving in the queue (early-finished Down-proj tokens trigger RMSNorm on other SMs) | 31,516 TPS | 29,492 TPS | **+6.4%** |

All at batch 8,192 tokens. End-to-end, integrated into Tokasaurus against SGLang on 65,536
ShareGPT prompts: **23,468 vs 19,170 total tok/s, +22.3%** [reported].

**The architectural decision worth stealing.** They run the **O-projection data-parallel
(replicated across GPUs) rather than tensor-parallel**, which eliminates the post-attention
reduce-scatter. They report this "reduces the network traffic by a factor of 8" at a cost of
~9 GB extra memory per GPU. On 183 GB B200s with a 256-expert MoE this is a cheap trade if the
same restructuring applies to GLM-5.2's attention output projection.

**They removed NCCL entirely.** Communication is unified-memory writes issued from inside the
kernel by dedicated storer threads, using the PGL abstraction, rather than copy engines.
Instruction-pipeline timing overhead measured at 0.39% average across 32 experiments.

**Caution.** The global-work-queue result (+14.2%) is at batch 8,192. ETC's independent result
(below) is that dynamic scheduling *loses* 17% at batch 1 with TP=4. Both can be true; the
regimes are different. At C1 we should expect the *static* schedule to win.

### Technique 6 — ThunderMLA (Mar 2025)

**[verified, https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla]**

A fused MLA-decode megakernel in **250 lines of device code**, replacing FlashMLA's separate
partial and reduction kernels. Attacks three costs: kernel setup/teardown, single-wave tail
effects, and loss of data reuse across launches.

Two schedulers, both run **asynchronously on the CPU** while the previous kernel executes:
a static scheduler (divides jobs, synthesizes k-way reduction trees, heap-based priority queue
allocation to SMs) and a "makespan backwards" scheduler that works backwards from the final
reduction with heuristic rollouts, worth ~10% more. Scheduling cost 1–2 ms, amortized because
schedules are reused across layers and regenerated infrequently.

All on H100 SXM:

| Workload | FlashMLA | ThunderMLA | Speedup |
|---|---|---|---|
| B4, seqs [4641,45118,1730,1696], Q4 | 52 µs / 144 TFLOPS / 1199 GB/s | 41 µs / 183 TFLOPS / 1520 GB/s | 20.6% |
| B1, seq 64k, Q1 | 55 µs | 44.5 µs | 23.6% |
| B64, seqs 256–1024 random, Q4 | 47 µs | 39.5 µs | 19.0% |
| B64, seq 512, Q2 | 39 µs | 28.6 µs | 36.3% |
| B132, seq 4k, Q4 | 226 µs | 210 µs | 7.6% |

Note the pattern: **the win is largest at small batch and short sequence** — i.e. exactly the
tail-effect-dominated regime — and smallest at B132/4k where the GPU is already saturated.
That is our C1 regime.

### Technique 7 — PGL and ParallelKittens: the multi-GPU numbers that matter most to us

**[verified, https://hazyresearch.stanford.edu/blog/2025-09-22-pgl]**,
**[verified, https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens]**,
**[verified, arXiv:2511.13940 full PDF]**

**Mechanism — PGL (Parallel Global Layout).** A ThunderKittens Global Layout that additionally
carries peer memory addresses, multicast addresses, and their TMA descriptors. Three ways to
get cross-GPU addressability, with an explicit ranking:

1. CUDA UVA — simplest, single-process only, not production-viable
2. IPC handles (`cudaIpcGetMemHandle`/`OpenMemHandle`) — multiprocess, **but cannot use NVSwitch multicast**
3. Manual VMM — `cuMemCreate`, export FD over a Unix socket, `cuMemImportFromShareableHandle`.
   **This is the only path that unlocks NVSwitch in-fabric acceleration.** Costs 2 MB allocation granularity.

The three multicast PTX families: `multimem.ld_reduce` (reduce from all devices into registers),
`multimem.red` (reduce from registers to all devices' memory), `multimem.st` (synchronous broadcast).

**The transfer-mechanism table — the most load-bearing measurement in this report.**
Observed NVLink bandwidth moving 1 GB using all SMs (theoretical max 450 GB/s H100, 900 GB/s B200):

| Mechanism | H100 GB/s (ratio) | B200 GB/s (ratio) |
|---|---|---|
| Copy engine | 368.82 (82%) | **726.13 (81%)** |
| TMA op | 350.01 (78%) | **669.12 (74%)** |
| Register op | 342.68 (76%) | **628.35 (70%)** |

*(The earlier Sept-2025 PGL blog post reported 541 GB/s / 60% for register ops on B200; the
Nov-2025 paper reports 628.35 / 70%. Assume the paper supersedes.)*

**Granularity and SM cost — this is where NCCL loses:**

- Copy engine needs **≥256 MB** transfers to sustain >80% utilization. Device-side (TMA) hits
  comparable utilization at **2 KB**.
- TMA's maximum message size is **227 KB**.
- **TMA needs only ~15 SMs to saturate NVLink on B200.** Register ops need **3.2–5.1× more SMs**
  (~76) because they are synchronous and need full occupancy. So register ops are worth it *only*
  when you need in-fabric reduction, which is the one thing TMA cannot do.

Functionality matrix [verified, Table 2]:

| Capability | Copy engine | TMA | Register |
|---|---|---|---|
| P2P transfer | ✓ | ✓ | ✓ |
| In-fabric broadcast | ✓ | ✓ | ✓ |
| P2P reduction | ✗ | ✓ | ✓ |
| **In-fabric reduction** | ✗ | ✗ | **✓** |
| Element-wise transfer | ✗ | ✗ | ✓ |

**Synchronization costs:** a single intra-SM `mbarrier` sync ≈ **64 ns**; inter-SM
synchronization through HBM ≈ **832 ns**. A 13× gap that should shape where we put our
rank-arrival signalling.

**Why NCCL and NVSHMEM are slow at small sizes (the mechanism, not the vibe):**

- NCCL enforces **two-way synchronization on every operation** — sender and receiver must
  handshake before data moves, even for point-to-point — and stages through small pre-allocated
  channel buffers, adding a copy. PK uses pre-allocated *destination* buffers and one-way
  transfers. Measured: **up to 1.79× on pure all-reduce**.
- NVSHMEM's public API does a global load (`ldg`) to fetch the peer address on every remote
  access plus a group sync (`__syncthreads`). Keeping peer addresses in registers and dropping
  the sync gave PK **4.5× lower element-wise NVLink access latency and ~20 GB/s more bandwidth**.

**Collective results vs NCCL, BF16 (relative performance, PK/NCCL):**

| Collective | Size axis | H100 | B200 |
|---|---|---|---|
| All-reduce sum | 2048 / 4096 / 8192 / 16384 / 32768 | 1.28 / 1.32 / 1.12 / 1.07 / 1.02 | **1.79 / 1.66 / 1.30 / 1.18 / 1.04** |
| All-gather along **tensor** dim | same | 2.91 / 2.52 / 2.51 / 2.55 / 2.51 | **3.25 / 2.49 / 2.61 / 2.60 / 2.57** |
| Reduce-scatter along **tensor** dim | same | 2.62 / 2.44 / 2.45 / 2.54 / 2.47 | **2.82 / 2.46 / 2.65 / 2.55 / 2.59** |
| 4-D (B,S,H,D) all-to-all | S = 16384…524288 | 2.03 → 1.82 | **2.38 → 2.30** |

Read the all-reduce row carefully: **the advantage grows monotonically as the message shrinks**,
and the smallest point they measured is far larger than a C1 decode all-reduce. The 2.5–3.2×
all-gather/reduce-scatter numbers come from operating on the *last* dimension directly — NCCL
only does contiguous partitions, so it needs a reshape+copy first.

**Scheduling — intra-SM vs inter-SM overlap, with a genuine surprise.** 8×H100, local GEMM
N×N×N/8, N=32768, BF16:

| Kernel | No overlap | Intra-SM overlap | Inter-SM overlap |
|---|---|---|---|
| GEMM + reduce-scatter | 510.1 TFLOP/s | **743.7** | 618.1 |
| GEMM + all-reduce | 450.9 TFLOP/s | 172.3 | **623.9** |

Intra-SM wins RS by 1.20×; inter-SM wins AR by **3.62×**. The reason: intra-SM AR issues N
atomic writes to N destinations per output tile and they *serialize at the destination port*.
Inter-SM AR instead accumulates locally in HBM, signals, and dedicates a few SMs to a single
in-network (NVSwitch) reduction — cutting `T_comm` by ~N. **If we currently overlap all-reduce
the intra-SM way, we may be 3.6× worse than doing nothing clever.**

**Remote L2 is far-sided.** Data fetched from a peer GPU is cached only on the *source* device's
L2, never the requester's. So repeated remote reads of the same tile re-cross NVLink every time.
The fix is a bulk prefetch into local HBM by communication-dedicated SMs.

**The overlap threshold** (already discussed in Bottom Line): communication is hidden when
`K ≥ sR/(2B)`. Verified empirically on H100 BF16:

| M=N | K | GEMM (ms) | GEMM+RS (ms) | Non-overlapped comm |
|---|---|---|---|---|
| 32768 | 512 | 2.071 | 6.483 | 68% |
| 32768 | 1024 | 2.918 | 6.613 | 56% |
| 32768 | 2048 | 5.567 | 7.531 | 26% |
| 32768 | 4096 | 11.78 | 11.828 | **<1%** |
| 32768 | 8192 | 23.285 | 25.325 | 8% |

**The eight PK primitives** (the whole API surface):

```
store_async(dst, src, coord)              // TMA store of a shared tile to multicast memory
store_add_async(dst, src, coord)          // TMA atomic-add of a shared tile to multicast memory
reduce(dst, dst_coord, src, src_coord)    // in-network reduce from multicast mem -> local HBM
all_reduce(dst_and_src, coord)            // in-network reduce, write back to multicast mem
signal(bar, coord, dev_idx, val)
signal_all(bar, coord, val)
wait(bar, coord, dev_idx, expected)
barrier(bar, coord, dev_idx)
```

P2P primitives are async and single-threaded (fusable with tensor-core work); network-accelerated
primitives need ≥ warp participation. Tiles range 16×16 up to ~256×256.

**Program template — four roles:** `loader` (local or peer HBM reads), `storer` (local or peer
HBM writes), `consumer` (tensor/CUDA-core compute), `communicator` (occupies whole SMs for
dedicated inter-SM-overlap communication). The template auto-tunes the SM/warp partition.

**B200 end-to-end (Appendix A, 8×B200, CUDA 12.8, PyTorch 2.8, BF16/FP32 accum):**

GEMM + reduce-scatter, local shape N×N×N/8, TFLOP/s:

| N | cuBLAS + NCCL | PK | Speedup |
|---|---|---|---|
| 2048 | 24 | 31 | 1.29× |
| 4096 | 137 | 223 | 1.63× |
| 8192 | 424 | 578 | 1.36× |
| 16384 | 729 | 1101 | **1.51×** |
| 32768 | 960 | 1409 | 1.47× |

DeepSpeed-Ulysses attention layer (B=16, H=128, D=128) on 8×B200 vs YunChang: 672 vs 392 at
S=12288, converging to 1336 vs 1297 at S=393216.

**Headline claims** [reported]: ≤2.33× for data/tensor-parallel, ≤4.08× sequence-parallel,
1.22× expert-parallel; matches Flux/Comet/CUTLASS hand-tuned kernels; beats Triton-Distributed
1.07–5.63×; <50 lines of added device code per kernel. Honest caveats they themselves state:
Triton-Distributed was tuned for H800 and sometimes loses to the non-overlapped baseline on
H100; their expert-parallel win over Comet is 0.92–1.22× (i.e. they sometimes lose).

### Negative results and honest limits from HazyResearch

- The B200 megakernel breakdown shows **~48% of a fused batch-1 forward pass is activation
  movement + sync**, not compute. Fusion does not fix data movement.
- TP megakernel authors state register spills and "simple scheduling heuristics" remain
  unoptimized.
- ParallelKittens is intra-node only; inter-node is explicitly future work.
- The O-projection replication trick costs ~9 GB/GPU.
- ThunderMLA's win collapses to 7.6% at large batch × long sequence.

### Blackwell/B200 statements, consolidated

B200 is the **default** build target in the Megakernels repo. TK 2.0 is B200-first (MXFP8/NVFP4,
CLC, TMEM control). ParallelKittens validates on both Hopper and Blackwell with a dedicated
B200 appendix. The 148-SM / cluster-size interaction and the 128×128 systolic shape rule are
the two Blackwell facts they repeat most.

### Based / linear attention

The lab's Based, LoLCATs, JRT and Hedgehog lines are architecture research on the
recall-throughput tradeoff of linear attention, with a one-year retrospective post
(`2025-03-24-based-retro`). Kernels exist in `kernels/based` and `kernels/linear_attention`.
**Honest assessment: not transferable to GLM-5.2 / DSA sparse MLA.** The kernel craft is in
ThunderKittens itself, which we can use without adopting the architectures.

---

## Colfax Research

### What they run

Colfax International is an HPC systems integrator whose research arm publishes what is,
in practice, **the primary public documentation for CUTLASS/CuTe on Blackwell**. Several authors
(Jay Shah in particular) are co-authors on FlashAttention-3 and FlashAttention-4 alongside
NVIDIA's Vijay Thakkar and Tri Dao. These are primary-quality sources — closer to spec than to
blog.

### The tutorial series (complete index as of Aug 2026) [verified]

| Date | Article |
|---|---|
| 2026-08-09 | Optimizing an NVFP4 Blockscaled GEMM on RTX PRO 6000 Blackwell (SM120) |
| 2026-06-20 | NVFP4 Blockscaled GEMM on RTX Pro Blackwell (SM12x) |
| 2026-05-09 | Dynamic persistent tile scheduling with Cluster Launch Control (CLC) |
| 2026-03-10 | FlexAttention + FlashAttention-4 (external) |
| 2026-03-05 | CUTLASS Tutorial: Hardware-supported Block-scaling on Blackwell |
| 2026-03-05 | **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling** |
| 2025-11-14 | A User's Guide to FlexAttention in FlashAttention CuTe DSL |
| 2025-09-21 | Categorical Foundations for CuTe Layouts |
| 2025-07-19 | CUTLASS 3.x APIs (external) |
| 2025-06-07 | CUTLASS Tutorial: Sub-byte GEMM on Blackwell |
| 2025-05-10 | CUTLASS Tutorial: GEMM with Thread Block Clusters on Blackwell |
| 2025-04-19 | CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory (TMEM) |
| 2024-12-19 | CUTLASS Tutorial: Persistent Kernels and Stream-K |
| 2024-10-25 | Epilogue Fusion in CUTLASS with Epilogue Visitor Trees |
| 2024-09-22 | CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining |

### Technique 1 — TMEM: the complete mechanical spec

**[verified, .../cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/]**

- **256 KB per SM**, organized as **512 columns × 128 lanes** of 32-bit cells.
- Address encoding: **bits 31–16 = lane ID, bits 15–0 = column**. So a CuTe tensor over TMEM has
  a lane stride of `1<<16 = 65536`. Lane 1 / column 1 is `0x00010001`.
- `tcgen05.alloc` allocates **whole columns** (all 128 lanes of a column at once), count must be
  a power of 2 with minimum 32. **Alloc and dealloc must both be issued from a single warp, and
  the same warp.** CUTLASS wraps this as `cute::TMEM::Allocator1Sm`.
- `tcgen05.relinquish_alloc_permit` (CUTLASS `release_allocation_lock()`) tells the hardware no
  further allocation will happen so other CTAs can queue for the SM. Forgetting this serializes CTAs.
- `tcgen05.ld.sync.aligned.{shape}.{num}.b32` — shapes `.16x64b`, `.16x128b`, `.16x256b`,
  `.32x32b`; `.num` from `.x1` to `.x128`, up to **16 KB per instruction**.
- **Hard constraint: each warp in a warpgroup can access only 32 lanes.** Warp 0 → lanes 0–31,
  warp 1 → 32–63, etc. Consequence: **any epilogue needs a full warpgroup** to drain TMEM, which
  serializes post-processing. This is the main structural cost of TMEM.
- The largest UMMA atom (128×256×16) occupies **exactly half of TMEM**, so exactly two can be
  pipelined.
- UMMA needs **no registers for operand data** (unlike WGMMA) and is issued by **one thread**,
  which is what makes deep async pipelining practical.
- Completion is signalled by `umma_arrive` → `tcgen05.commit` on an mbarrier; this fences both
  the TMEM writes and the SMEM reads. Reusing SMEM before `umma_arrive` is a correctness bug.
- Debug flag: **`nvcc --g-tensor-memory-access-check`** enables runtime TMEM bounds/init validation.

Listed pitfalls: warpgroup epilogue bottleneck; alloc-lock contention; wrong `.shape`/`.num`
for `tcgen05.ld`; premature SMEM reuse; cross-warp 32-lane boundary violations.

### Technique 2 — 2-SM UMMA and thread block clusters

**[verified, .../cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/]**

- CTAs pair by the **0th bit** of the cluster index (0↔1, 2↔3, …). The **even CTA is the leader**
  and issues `tcgen05.mma` with `cta_group::2`.
- For a 256×256×16 MMA, each CTA loads 128×16 slices of A and B and holds a 128×256 accumulator
  in its own TMEM. **Same FLOPs, half the operand traffic** compared to two independent 128×256
  MMAs — a free arithmetic-intensity doubling.
- Clusters are guaranteed co-scheduled on SMs in the same GPC and can read each other's SMEM
  (DSMEM). Max portable cluster is 8; **B200 supports 16 with opt-in**.
- TMA multicast participation is a **16-bit `ctaMask`** in `cp.async.bulk.tensor`, bit *i* = CTA
  with cluster index *i*, column-major within the cluster. For a 4×4 cluster, CTA 0's A-mask is
  `0x1111` (same row) and B-mask is `0x000f` (same column).
- **The DSMEM address trick:** bit 24 of a cluster-unified SMEM address encodes bit 0 of the CTA
  ID. An odd CTA reaches its leader's mbarrier by clearing bit 24 of its own mbarrier address —
  CUTLASS masks with `0xFEFFFFFF`.
- Post-UMMA arrival count equals the number of **MMA tiles**, not CTAs.
- Pair-UMMA supports M ∈ {128, 256} only, and the accumulator is always split along M — this
  constrains cluster shape choice.

No performance table is given; the guidance is qualitative (halved operand traffic vs added
synchronization complexity).

### Technique 3 — Hardware block-scaling (this is our NVFP4 path)

**[verified, .../cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/]**

Format table:

| Format | Operand type | Vector length | Scale type | Max value |
|---|---|---|---|---|
| mxf8 | E5M2, E4M3 | 32 | UE8M0 | — |
| mxf6 | E3M2, E2M3 | 32 | UE8M0 | — |
| mxf4 | E2M1 | 32 | UE8M0 | — |
| **nvf4** | **E2M1** | **16** | **UE4M3** | **2688** |

Instruction:

```
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
    [d-tmem], a-desc, b-desc, idesc,
    [scale-A-tmem], [scale-B-tmem], enable-input-d;
```

`.kind` ∈ {`mxf8f6f4`, `mxf4`, `mxf4nvf4`}. `atom_K` is 32 bytes for mx*, **64 bytes for nvf4**.
`.scale_vec::1X` (mxf8f6f4 only), `::2X` (mxf4), `::4X` (nvf4).

**Scale factors always live in TMEM and eat TMEM columns.** With bM=128:

| Mode | SFA columns | SFB columns | Total |
|---|---|---|---|
| block32/1X (mxf8) | 4 | up to 8 | ~12 |
| block32/2X (mxf4) | 8 | up to 16 | ~24 |
| **block16/4X (nvf4)** | **16** | **up to 32** | **up to 48** |

So NVFP4 costs ~4× the TMEM for scale factors versus MXFP8. On a 512-column TMEM that is ~9% —
worth budgeting explicitly when picking tile shapes.

**The GMEM layout recipe** — store scale factors *already interleaved the way TMEM wants them*,
so the transfer is coalesced and conflict-free. A 512-byte tile is tiled across the tensor:

```python
def interleave_sf_tensor(sf: torch.Tensor) -> torch.Tensor:
    M, SF_K = sf.shape
    REST_M = M // 128
    REST_K = SF_K // 4
    out = sf.reshape(REST_M, 4, 32, REST_K, 4)
    out = out.permute(0, 3, 2, 1, 4).contiguous()
    out = out.permute(2, 3, 0, 4, 1)
    return out
```

**SMEM → TMEM** is `tcgen05.cp`, wrapped by CUTLASS as `Cp4x32x128bOp` with `.shape=.32x128b`
and `.multicast=.warpx4` — scale factors are **multicast to all 32-lane partitions**.
`tcgen05.cp` and `tcgen05.mma` form an **implicit pipeline; no explicit sync is needed and no
circular buffer is used for the TMEM scale factors**. (This is the same fact ThunderKittens 2.0
measured at ~500 TFLOP/s.)

In the mainloop, SF tensors are not gemm arguments — you set them on the MMA object:

```python
tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

For 2-CTA: **SFA is split between the pair; SFB is multicast identically to both.**

Non-128-divisible N handling (relevant if our MoE expert N is 192 or 64): at bN=192, even/odd
work tiles share half the middle SFB tile and the odd tile shifts the TMEM SFB pointer forward
by 2 columns (32 bytes); at bN=64, both tiles load the same full SFB tile and the odd tile uses
the second half.

### Technique 4 — Sub-byte GEMM: the FP4 layout constraints

**[verified, .../cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/]**

- **FP4/FP6 are padded to 8-bit containers in SMEM.** 16 consecutive 4- or 6-bit elements are
  packed contiguously then padded to a 16-byte boundary. You lose the SMEM footprint benefit;
  you keep the GMEM/HBM benefit.
- GMEM can stay compactly packed; **TMA unpacks during the load**:
  `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` (16 packed 4-bit → 16-byte-aligned padded, +8 bytes)
  and `..._16U6_ALIGN16B` (+4 bytes).
- **Stricter TMA constraints for sub-byte:** base address 32 B aligned (not 16 B); leading
  dimension a multiple of **128 elements**; **only 128 B swizzle or none**. CUTLASS asserts 4-bit
  data at 64 B alignment and 6-bit at 96 B.
- The K extent of a dense MMA tile is **always 32**; operand tiles are 32 bytes wide in K with
  1 byte per padded element.
- Use `float_e2m1_unpacksmem_t` / `float_e2m3_unpacksmem_t` to trigger the right TMA formatting.
  Runtime dtype selection is via 3-bit format fields in the MMA instruction descriptor.

No performance numbers in this article.

### Technique 5 — Cluster Launch Control (CLC): hardware work-stealing

**[verified, .../dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/]**

**Mechanism.** Launch a grid sized as if you were doing one-tile-per-cluster. The first wave of
clusters then *cancels* not-yet-launched clusters and inherits their coordinates. Two PTX
instructions:

- `clusterlaunchcontrol.try_cancel` — one thread per cluster atomically requests cancellation of
  an unlaunched cluster; the 16-byte response is written to shared memory.
- `clusterlaunchcontrol.query_cancel` — decodes it: `.is_canceled` and `.get_first_ctaid`.

**Structure.** A dedicated **scheduler warp** in the cluster's first CTA runs the try_cancel loop
and publishes tile coordinates through a CLC pipeline; TMA / MMA / epilogue warps consume via
mbarrier. The loop terminates when `try_cancel` fails.

**Measured on B200** (74 clusters of size 2, FP8 E4M3 in / FP32 out, MMA tile 256×256,
cluster (2,1), K ∈ {2048, 8192}, M=N up to 32768):

- **Imbalanced** GEMM (varying K per problem, e.g. K ∈ {128, 2048} giving 16× compute disparity
  across clusters): CLC gives a significant speedup and near-uniform FLOP distribution.
- **Balanced** GEMM: CLC ≈ static persistent, and static is *sometimes faster*.
- L2 hit rate: static persistent 52–60%; CLC 35–60% — work stealing costs locality.
- Tile counts per SM under CLC: 54–59 vs 55–56 static. The authors' own conclusion: "forcing all
  SMs to compute exactly the same number of tiles, even if balanced, may be slightly suboptimal."

**Read for us:** CLC is for the *MoE grouped GEMM* (data-dependent expert token counts), not for
the dense GEMMs. Applying it uniformly would cost L2 hit rate for nothing.

### Technique 6 — Persistent kernels and Stream-K

**[verified, .../cutlass-tutorial-persistent-kernels-and-stream-k/]**

- Wave quantization, concretely: on an H100 PCIe (114 SMs) a **115-tile** workload takes the same
  two waves as a **228-tile** workload — half the device wasted.
- Tile scheduler API: `get_initial_tile()`, `is_valid(worktile)`, `get_next_tile()`, returning
  `(m_block, n_block, k_block_start, k_block_stop)`.
- Stream-K splits **only along K**, assigning fractional tiles so every SM gets
  `(M/bM × N/bN)/num_SMs` work. Two reduction modes: **turnstile** (deterministic — CTA₀ writes
  workspace, CTA₁ waits at a barrier and reduces in, last CTA reduces into accumulators and runs
  the epilogue) and **nondeterministic** (CTAs 1..n-1 atomically reduce unordered).
- **Hybrid is what CUTLASS actually ships:** 1–2 waves of Stream-K for partial tiles, then plain
  data-parallel for the rest, which preserves L2 hit rate by aligning K-offsets across SMs.
- Classes/params: `PersistentTileSchedulerSm90StreamKParams`; decomposition modes
  `DataParallel` / `SplitK` / `StreamK` / `Heuristic`; args `{splits, swizzle, raster_order,
  decomposition_mode}`; raster `AlongM`/`AlongN`; swizzle ∈ {1,2,4,8}.
- H100 PCIe, M=1024, K=4096, varying N: DataParallel shows **30–40% drops at wave boundaries**.
  At N=7296 (evenly divisible) DP and Stream-K tie at ~2500 TFLOPs/s; at N=7488 (incomplete last
  wave) Stream-K ~2450 vs DP ~1800. `Heuristic` lands within ~2% of optimal by switching when the
  tail wave is under ~50% full.
- Split-K "rarely outperforms other modes" — an explicit negative result.

### Technique 7 — Pipelining, warp specialization, `setmaxnreg`

**[verified, .../cutlass-tutorial-design-of-a-gemm-kernel/]**

The `Pipeline` state machine: `full_barrier[]` / `empty_barrier[]` mbarrier arrays sized to the
stage count, phase bits, and a thread-local `PipelineState`. Four calls: `producer_acquire()`,
`producer_commit()` (a no-op for TMA, since TMA signals the barrier itself), `consumer_wait()`,
`consumer_release()`.

**Register reallocation** via the `setmaxnreg` PTX instruction at warpgroup granularity: producer
warps drop to ~24–40 registers (they only drive TMA), consumers take 240–256. A published split
is **24/240/240** for 1 producer + 2 consumer warpgroups, against a 64K-register-per-SM budget.

H100 PCIe FP16 (peak 750 TFLOP/s dense):

| Kernel | TFLOP/s | % peak | Config |
|---|---|---|---|
| Multistage | 531 | ~71% | bM=256, bN=256, bK=96, 2 stages, 4 MMA groups |
| Warp-specialized | 536 | ~71% | same |
| **CUTLASS best (WS persistent cooperative)** | **630** | **~84%** | — |

The gap between a good hand-rolled warp-specialized kernel (71%) and CUTLASS's tuned persistent
cooperative kernel (84%) is a useful calibration: **the last 13 points come from persistence and
scheduling, not from the inner loop.**

### Technique 8 — FlashAttention-4

**[verified, .../flashattention-4-algorithm-and-kernel-pipelining-co-design.../ and arXiv:2603.05451]**
Authors: Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao.

**The framing — asymmetric hardware scaling.** H100 → B200: BF16 tensor-core throughput goes
from ~1 to ~2.25 PFLOP/s, while **SFU count and shared-memory bandwidth are unchanged**.
Per-SM feeds and speeds they publish for M=N=D=128:

| Unit | Throughput | Forward cycles | Backward cycles (1-CTA) |
|---|---|---|---|
| Tensor cores | 8192 ops/cycle | 1024 | 2560 |
| Exponential (MUFU.EX2) | **16 ops/cycle** | 1024 | 1024 |
| Shared memory | 128 B/cycle | 768 | **3328** |

So the **forward pass is co-limited by the exponential unit**, and the **backward pass is
shared-memory-bandwidth bound**. Two different problems, two different fixes.

**Fix 1 — software exponential emulation (forward).** Split `2^x` work between the hardware
`MUFU.EX2` and an FMA-based approximation, moving load onto otherwise-idle CUDA cores:

- Cody-Waite range reduction: `2^x = 2^n · 2^f`
- Degree-3 Horner polynomial on the fractional part with published coefficients
  **p₀ = 1.0, p₁ ≈ 0.6951, p₂ ≈ 0.2276, p₃ ≈ 0.0771**
- Exponent bit shifting and mantissa recombination to reassemble

Three softmax-side warpgroups per CTA: **two softmax warpgroups** (one per Q tile) that
*explicitly synchronize with each other to avoid simultaneous MUFU contention*, plus **one
correction warpgroup** that does the rescaling off the critical path. The MUFU-vs-software split
ratio is tunable.

**Fix 2 — conditional online-softmax rescaling.** Only rescale the running output when the max
jumps by more than a threshold τ:

```
O_j = exp(m_{j-1} - m_j)·O_{j-1} + exp(S_j - m_j)·V_j    if m_j - m_{j-1} > τ
    = O_{j-1} + exp(S_j - m_{j-1})·V_j                    otherwise
```

Decided at **warp granularity** to avoid divergence.

**Fix 3 — dual query tile ping-pong.** Each CTA processes two 128-token Q tiles (Q^H, Q^L)
alternately rather than one, so softmax on one tile overlaps MMA on the other.

**Fix 4 — keep P and dS in TMEM, not SMEM.** This is the direct answer to the SMEM-bandwidth
bottleneck; accumulators stay in TMEM (unlike FA3, which held them in registers), so multiple
MMAs are in flight simultaneously.

**Fix 5 — 2-CTA backward.** Compute and store S and P **transposed** (Sᵀ, Pᵀ) so they already
match the operand-A layout for the subsequent dV and dK MMAs. With M=256, N=K=128, operand-B
staging splits across the pair, **halving SMEM traffic**. The dQ reduction axis collides with the
CTA-pair split, resolved by a **DSMEM exchange** that repacks dS so each CTA owns M/2 rows with a
full 2N reduction — which also **halves the global atomic adds for dQ**.

**Deterministic mode** (semaphore locks serializing dQ atomics, plus CTA swizzling and
shortest-processing-time-first ordering) reaches **85–90% of nondeterministic throughput**.

**Numbers (BF16, B200, head_dim=128, causal and non-causal, varying seqlen):**

| | Value |
|---|---|
| FA4 forward | **1605 TFLOP/s** (blog) / **1613 TFLOP/s** (arXiv), **71% utilization** |
| vs cuDNN 9.13 | 1.1–1.3× |
| vs Triton | 2.1–2.7× |
| Backward | outperforms baselines at large seqlen; cuDNN 9.13+ has since absorbed FA4 techniques |

**Implementation:** written in **CuTe-DSL (CUTLASS Python DSL)**, lowered to PTX. Claimed
**20–30× faster compile times** than C++ templates. That matters operationally: it makes
per-shape specialization affordable at deploy time.

**Honest gap for us:** the paper is entirely training-shaped (forward + backward, long sequences).
**There is no decode/KV-cache/token-by-token content.** The transferable pieces are the
exponential emulation, the conditional rescaling, the TMEM-for-intermediates rule, and the
feeds-and-speeds methodology — not the kernel.

### Technique 9 — FlexAttention in the FlashAttention CuTe DSL

**[verified, .../a-users-guide-to-flexattention-in-flash-attention-cute-dsl/]**

Built with Meta and Princeton/Together AI. Targets **SM90 and SM100, forward and backward**.

`FlexAttention(Q,K,V) = Softmax(mask_mod(score_mod(QKᵀ)))V`, with two user hooks:

```python
def alibi_score_mod(score, batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
    slope = exp2(-(head_idx + 1))
    return score - slope * abs(q_idx - kv_idx)

def causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    return kv_idx <= q_idx + offset
```

`score_mod` bodies must use the `TensorSSA` abstraction and vectorize in groups of a tunable
`vec_size` — **vectorization is not possible when `aux_tensors` are used.**

**The block-sparsity machinery — this is what maps onto DSA `index_topk`.** Two index tensors
plus their counts:

- `mask_block_idx[B, H, num_q_blocks, num_kv_blocks]` — partially-masked blocks (run `mask_mod`)
- `full_block_idx[B, H, num_q_blocks, num_kv_blocks]` — fully-live blocks (**skip `mask_mod` entirely**)

Everything not in either list is never loaded. Worked example: seqlen_q=768, seqlen_kv=896,
128×128 tiles, causal → 6 diagonal blocks masked, 21 blocks full, **15 blocks skipped**.

**Hard constraint: the tile size used to compute block sparsity must equal the kernel tile size.**

`create_block_mask()` from PyTorch produces a `BlockMask` that converts to CuTe
`BlockSparseTensors`. Sparsity computation is expensive but amortized across all layers.

```python
out, _ = _flash_attn_fwd(q, k, v, score_mod=..., mask_mod=...,
                         block_sparse_tensors_torch=block_sparse_tensors,
                         aux_tensors=aux_tensors)
# or through PyTorch:
out = torch.compile(flex_attention)(q, k, v, score_mod=..., 
                                    kernel_options={"force_flash": True})
```

Tunables exposed at `cute.compile` time: `num_stages`, `num_threads`, `intra_wg_overlap`.

**Performance:** 95% of FA3 forward with arbitrary mods; ~50% faster than the Triton FlexAttention
[reported].

### Technique 10 — The NVFP4 optimization walkthrough (SM120, but the method transfers)

**[verified, .../optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/]**

Caveat up front: **this is RTX PRO 6000 (SM120), not B200 (SM100)**. SM120 uses warp-level
blockscaled MMA rather than tcgen05 2-SM UMMA, so the *kernel* does not port. The **optimization
sequence and its attribution of gains does**.

| Version | Change | Effect |
|---|---|---|
| v1 baseline | 1 TMA load warp + 8 MMA warps, warp-specialized producer/consumer | 1476 TFLOP/s @8k (73% of 2015.2 peak); L2 hit only 76.31% at 32k |
| v2 | **Threadblock swizzle = 16** so CTAs in a wave touch the same region (distinct tiles per wave 40+ → 28 at 32k) | **+387 TFLOP/s at 32k**; L2 hit 76.31% → 90%+ |
| v3 | Move store-pipeline setup out of the work-tile loop; delay `producer_acquire()` | ~1% |
| v4 | **Dedicated store warp** with two named barriers (`epilog_free_barrier`, `epilog_ready_barrier`), 16 hardware barriers | ~1%, but unblocks tensor cores from epilogue |
| v5 | **SFA SMEM bank-conflict fix**: stride `(32,4):(16,4)` → `(32,4):(4,128)`, i.e. byte_offset `16m₀+4m₁+b` → `4m+b` | removed **8.39M excessive wavefronts**; SFA load now 1 wavefront not 2 |
| v6 | CTA tile 128×128 → **192×128**, MMA warps 8 → 12 (6×2×1); work tiles at 2k drop 256 → 176 (one wave not two); pipeline stages forced 4 → 2 by SMEM | **+186 TFLOP/s at 2k**, +40 at 16k/32k |
| v7 | Autotune tile size, stages, MMA warp count, swizzle, epilogue config, register allocation | +12–13 at 16k/32k |

Final: **1666 TFLOP/s at 16k = 83% of peak**, ~cuBLAS 13.6 parity at 16k and 32k. Progression by
shape: 2k 1258→1623 (+29%), 4k 1387→1472 (+6%), 8k 1476→1536 (+4%), 16k 1436→1666 (+16%),
32k 1169→1637 (+40%).

**Limiters they name:** wave quantization at 2k (256 tiles / 188 SMs ≈ 1.4 waves); L2 thrashing
at 16–32k (288–1152 MB footprint vs 128 MB L2); **thermal throttling from 2.43 → 2.15 GHz,
dropping effective peak from 2015 to ~1782 TFLOP/s**; SMEM bank conflicts.

**Negative results they report:** restricting SFA/SFB loads to only the threads that need them
added branching overhead for no gain; loading extra SFA/SFB for alternate MMA instructions via
thread-id toggling yielded nothing.

### Colfax open-source artifacts

Colfax's articles are documentation, not a library — the code they describe lives in
NVIDIA/cutlass and Dao-AILab/flash-attention. The directly usable artifacts are: CUTLASS's
Blackwell blockscaled GEMM collectives, the CuTe DSL, and the FlashAttention CuTe DSL kernels
(`flash_attn.cute.interface._flash_attn_fwd`, `flash_attn.cute.flash_fwd.FlashAttentionForwardSm90`
and its SM100 sibling).

---

## CMU Catalyst and its orbit

### Mirage / MPK (Mirage Persistent Kernel)

**[verified, https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17]**
**[verified, https://github.com/mirage-project/mirage]**

**Mechanism.** MPK compiles an LLM into a task graph where **tasks** are units assigned to one
SM and **events** are counters. A task's completion increments its triggering event's counter;
when the counter hits a threshold the event is *activated* and pushed to a scheduler's queue.
SMs are **statically partitioned into workers and schedulers**; schedulers run on a **single warp**
and there can be **up to four schedulers per SM**. Workers loop: fetch task → execute → notify.

This buys fine-grained software pipelining across layer boundaries: "matmul tasks can execute in
parallel with attention tasks from different layers; allreduce communication can begin as soon as
partial matmul results are available."

**Evidence.** Single A100 40 GB: **12.5 ms per-token decode vs 14.5 ms for vLLM/SGLang**, against
a stated theoretical lower bound of 10 ms. Headline claim: **1.2–6.7× latency reduction**
[reported]. Multi-GPU gains reportedly grow with GPU count.

**Honest limits.** The blog does not state the model for the 12.5 ms figure. The repo README
demonstrates Qwen3-8B and **does not mention Blackwell, B200 or SM100 anywhere** — the only
hardware-shaped requirement is that the worker/scheduler counts "must match the number of physical
SMs." **[unverified] whether MPK has working SM100 support.** Install:
`git clone --recursive --branch mpk ...; pip install -e . -v; export MIRAGE_HOME=$(pwd)`.
GPU MODE Lecture 79 ("Mirage (MPK): Compiling LLMs into Mega Kernels", Mengdi Wu & Xinhao Cheng)
covers this.

### Event Tensor / ETC — the most directly relevant paper I found

**[verified, arXiv:2604.13327v2, MLSys 2026]** — Jin, Hou, Wang, Lai, Chen, **Zihao Ye**, Cai,
Dong, Cheng, Zhang, Zhao, Huang, Yang, Jiang, Oliaro, Ji, **Xupeng Miao**, **Vinod Grover**,
Mowry, **Zhihao Jia**, **Tianqi Chen** (CMU / NVIDIA / SJTU / Berkeley / Princeton / Tsinghua / PKU).
Implemented as compiler passes over Apache TVM. **"ETC has been incorporated into a major
open-source system"** — unnamed in the paper.

**Why this one matters most:** it is evaluated on **8×B200** (Ubuntu 24.04, PyTorch 2.8.0,
**CUDA 13.0, driver 580.82.07**) against **vLLM v0.11.0rc2 and SGLang v0.5.3rc0 with CUDA Graphs,
PDL and torch.compile already enabled** — i.e. against our actual baseline, on our actual hardware.

**The problem statement, quantified:** "Each kernel launch typically incurs **5–10 µs** of
latency, while the fastest kernels may complete in **2 µs**." CUDA Graphs remove the launch gap
but **preserve kernel boundaries**, so they cannot expose inter-kernel parallelism.

**Mechanism — Event Tensor.** A tensor of event counters that encodes dependencies between
*tiled tasks* with index-mapping strings (`"h->bh"`, `"i->topk[i,:]"`, …). It gives first-class
support for both **shape dynamism** and **data-dependent dynamism** (MoE routing, variable KV
length, speculative decoding branches).

**Mechanism — static scheduling transformation** (Algorithm 1): (1) build per-SM execution queues
on the host; (2) emit a persistent main loop; (3) lower Event Tensor edges into explicit
`notify()` / `wait()`. Worked example for GEMM+RS: each RS task depends on two MM tasks, so the
event counter starts at 2; at T1 SM0's MM finishes and decrements to 1, the RS task
statically scheduled next on SM0 spins; SM1 keeps working; at T2 SM1's MM decrements to 0 and
releases the RS.
Shape dynamism is handled by sampling representative shapes and reusing the next-larger queue;
data dependence by conservatively rewriting notify/wait to the worst case.

**Mechanism — dynamic scheduling.** On event activation, consumer tasks are atomically *pushed*
to an on-GPU scheduler; any idle SM atomically *pops*. No host precomputation.

**Results — TP=8 fused collectives on 8×B200, 8192 tokens, MLP shapes from 8 real models:**
up to **1.40×** execution-time speedup over cuBLAS+NCCL for both GEMM+ReduceScatter (multimem
PTX, dynamic scheduler) and AllGather+GEMM (ring via copy engine, static scheduler), also beating
TP-Async, Triton-Distributed v0.0.2-rc and cuBLASMp. They note Triton-Distributed's **B200
support is experimental and its GEMM is not Blackwell-tuned**, and that the unfused
cuBLAS+NCCL baseline is sometimes competitive with fused approaches.

**Results — MoE layer, Qwen3-30B-A3B (128 experts, top-8), single B200:** up to **1.23×** over
the better of Triton 3.4.0 and FlashInfer 0.2.14.post1, peaking at 1024 tokens. Mechanism: data-
dependent Event Tensors break the global barrier between the two GroupGEMM stages, creating a
pipeline that **smooths SM allocation across fused operators and reduces wave quantization**;
plus on-chip dynamic load balancing for irregular routing.

**Results — end-to-end TPOT, prefill 512 / 100 output tokens:**

| Setting | ETC vs vLLM | ETC vs SGLang |
|---|---|---|
| Qwen3-30B-A3B, TP=1, **BS=1** | **1.48×** | **1.20×** |
| Qwen3-32B, TP=1, BS=1 | 1.15× | — |
| Qwen3-32B, TP=1, BS=64 | — | 1.09× |
| Qwen3-32B, **TP=4** | 0.99–1.06× | **slower** |

They attribute the TP=4 loss to **SGLang's highly optimized CPU scheduler incurring lower
distributed runtime overhead** — a direct, useful acknowledgement that at TP>1 and low batch,
host-side scheduling is a first-order term. They also concede their compiler-generated GEMM tiles
are "less tuned than cuBLAS in certain configurations."

The megakernel covers the **full decode pipeline** (Attention, RoPE, KV-cache append, Norm, MLP,
MoE), and they name the specific wins: Q's Norm+RoPE running concurrently with K's
Norm+RoPE+CacheAppend; pipelining GroupGEMMs and MLP GEMMs to cut wave quantization; and
**prefetching model weights before input activations are ready**.

**Warmup (Qwen3-32B):**

| System | Warmup (s) | # JIT graph captures |
|---|---|---|
| SGLang (JIT) | **583** | 51 |
| vLLM (JIT) | 123 | 67 |
| ETC (AOT) | **35** | 0 |

Offline compile of the single shape-generic megakernel: 107 s.

**The scheduling ablation — a clean negative result we should heed.**
Relative to an *unfused megakernel* using identical operator code (single event = global barrier
between stages), so the delta is purely inter-kernel parallelism:

| Qwen3-32B, TP=4 | BS=1 | BS=16 | BS=32 | BS=128 |
|---|---|---|---|---|
| ETC static | **1.09** | 1.06 | 1.07 | 1.06 |
| ETC dynamic | **0.83** | 0.82 | 0.85 | 0.89 |

| MoE layer | 1 tok | 128 | 1024 | 4096 |
|---|---|---|---|---|
| ETC static | 1.03 | 1.02 | 1.04 | 1.02 |
| ETC dynamic | **0.95** | 1.06 | **1.08** | 1.03 |

Conclusions, in their words: dynamic scheduling's "overhead becomes very large on distributed
setting, especially when trying to push tasks to remote task queue"; and there is a
"consistent **6–8% speedup of ETC-static over ETC-unfused**, a gain purely from fine-grained
pipelining." **For our C1 objective: static schedule, fine-grained events, no work queue.**

### FlashInfer

**[verified, https://github.com/flashinfer-ai/flashinfer README]** — originated in the same
CMU/UW orbit (Zihao Ye; GPU MODE Lecture 40).

Unified APIs over attention / GEMM / MoE with **automatic backend selection** across FA2, FA3,
cuDNN, CUTLASS and TensorRT-LLM kernels. Coverage: prefill, decode, mixed batching (POD-Attention),
**MLA for DeepSeek-style models**, cascade attention with hierarchical KV-cache, **block and
variable-block sparse attention**, paged and ragged KV-cache. Precisions: FP8 per-tensor and
groupwise, **FP4 (NVFP4 and MXFP4) for Blackwell**, BF16. Hardware SM7.5 → SM12.1;
**Blackwell support landed in v0.4.0**. JIT compilation with optional `flashinfer-cubin` and
`flashinfer-jit-cache` wheels to trade flexibility against startup time. Already a dependency of
SGLang, vLLM, TensorRT-LLM and TGI.

**Relevance:** its sparse/block-sparse attention path and its NVFP4 MoE kernels are the closest
open-source analogue to what GLM-5.2's DSA + 256-expert MoE needs. ETC's own benchmark shows
FlashInfer's GroupGEMM is the stronger MoE baseline at large token counts and Triton's
gather/scatter-fused variant wins at small counts — worth A/B-ing both at our C-levels.

### Other megakernel work found on arXiv

**[verified abstracts via arXiv API; I did not read the full PDFs]**

- **Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference**
  (arXiv:2605.11581, May 2026, Baidu-affiliated author list). Compile-time megakernel search with
  a **three-dimensional shared-memory constraint model plus K-dimension splitting**, MLIR-based
  offline search to eliminate runtime branching. **+23.6% throughput — but on an NVIDIA L20**, a
  small inference card. Method may transfer; numbers do not.
- **Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs**
  (arXiv:2604.15379, Apr 2026; authors include Ryan Swann, Muhammad Osama, Sean Siddens — AMD +
  the HipKittens crowd). **Per-chiplet scheduling** and coordinated cache reuse for persistent
  kernels. Claims **1.3–1.5× decode latency reduction and up to 37% less HBM traffic**.
  **Directly interesting for B200, which is itself a dual-die part** — chiplet-aware task placement
  is a lever we have not considered.

---

## Microarchitecture, SASS, and the "what does SM100 actually do" literature

The assignment asked specifically for tcgen05/TMEM reverse-engineering and SM100 SASS analysis.
Honest answer: **the Colfax tutorials plus NVIDIA's PTX ISA are the reverse-engineering.** There
is no public SASS-level teardown of tcgen05 that I could find. What does exist:

| Paper | What it actually measures | Caveat |
|---|---|---|
| **Microbenchmarking NVIDIA's Blackwell Architecture** (arXiv:2512.02189v3, Jarmusch & Chandrasekaran, Dec 2025 / rev Mar 2026) | Open-source suite on **B200**: memory subsystem, tensor-core pipeline, FP32/FP16/FP8/FP6/FP4, dense+sparse GEMM, the Decompression Engine | **Read with care.** I extracted the full text; the TMEM section is correct on structure (256 KB, 512 cols × 128 lanes) and gives a usable rule — **TMEM is most efficient at 64×64 element tiles (4 KB at FP8); <32×32 underutilizes the 1024-bit interface; >128×128 triggers multi-phase transfers** — and a **16 TB/s TMEM read bandwidth** figure. But the same section parenthetically cites `wgmma` matrix shapes while discussing Blackwell, which is a Hopper instruction. Treat the tile-size guidance as a hypothesis to verify, not as spec. |
| **Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks** (arXiv:2507.10789v2, Jul 2025) | Memory hierarchy, SM pipelines, sub-core units, FP4/FP6 | **The comparison is GeForce RTX 5080 vs H100 PCIe** — consumer Blackwell, not B200. Do not transfer numbers. |
| **Microbenchmark-Driven Analytical Performance Modeling** (arXiv:2605.04178, May 2026) | Analytical model for B200 capturing TMEM, TMA and 5th-gen tensor cores; **1.31% MAE across 21 kernels** vs >95% error for naive roofline | Validated on Rodinia/SPEChpc, not LLM kernels |
| **Characterizing Warp Divergence from Pascal to Blackwell** (arXiv:2607.23402, Jul 2026) | **This is the SASS analysis.** Cycle-accurate microbenchmarks + hardware counters + static analysis of compiler-generated SASS. Finds divergent paths serialize *linearly*, `T(k) ≈ sk`, with no super-linear reconvergence penalty on any generation. Blackwell introduces a **two-tier convergence-barrier classification, uniform-branch instructions, and explicit partial-mask warp synchronization** — but controlled bit-flip experiments show the new barrier classes are **static compiler classifications with no observable runtime effect** | A clean negative result: don't over-engineer around Blackwell's new branch encodings |
| **Spec Sheets Are Not Kernels: an ISA- and Source-Level Audit of INT8 Availability on Blackwell Ultra** (arXiv:2608.11693v2, Aug 2026) | **The PTX ISA never exposes `tcgen05.mma` with `.kind::i8` on sm_103a.** CUTLASS skips INT8 UMMA generation for 103a; vLLM ships no INT8 GEMM for Blackwell; SGLang stops at Sm90 | **sm_103a is Blackwell Ultra (B300), not our sm_100a B200.** Still: a useful methodology (audit every layer of the stack, not the datasheet) and a warning that dtype availability is a whole-stack property |

**Practical takeaway:** if we want authoritative tcgen05 timing on our own B200s, nobody has
published it and we should microbenchmark it ourselves. The 2512.02189 suite is open source and
is the closest starting point.

---

## Adjacent DSLs and compilers worth tracking

- **CUDA Tile (CuTile)** — NVIDIA's Python tile DSL. First independent cross-architecture
  evaluation [verified, arXiv:2604.23466v2]: on **B200, up to 1007 TFLOP/s for fused attention,
  2.5× FlashAttention-2, in 60 lines of Python**. GEMM reaches only **52–79% of cuBLAS in 22 lines**
  (WMMA needed 123 lines for comparison). The same attention kernel gets only **53% of FA-2
  throughput on sm_120**, exposing a big cross-architecture tuning gap. By contrast **Triton
  sustained 62–101% of cuBLAS across every platform with no architecture-specific tuning** — the
  portability story is Triton's, the peak story is not.
- **TileLang** (tile-ai/tilelang) [verified README] — Pythonic DSL on TVM. Supports **SM70–SM120
  including SM100**, with explicit **TMA, WGMMA and TMEM** support, "TCGEN5 MMA tensor-shared path",
  "two-SM Blackwell kernels", **MXFP8 block-scaled matmul on SM100**, SM120 NVF4 block-scaled MMA,
  and FlashAttention on SM100. Example kernels include MLA decoding, block-sparse attention,
  dequantization GEMM. Published benchmark charts are H100/A100/MI300X-era; **no B200 numbers in
  the README**.
- **Gluon** — Triton's new tile-based layer with explicit low-level control, presented as GPU MODE
  Lecture 104 by Peter Bell, Mario Lezcano and Keren Zhou. **[verified] the lecture exists;
  [unverified] its Blackwell/TMEM capabilities — I could not fetch the content.** Worth chasing:
  if it exposes tcgen05/TMEM from Triton, it is the cheapest path to Blackwell-native kernels
  inside SGLang, which is already Triton-heavy.
- **DeepGEMM** (deepseek-ai) [verified README] — SM90 **and SM100**. FP8 E4M3, **FP4 with UE8M0
  scaling**, BF16. SM100 supports NT/TN/NN/TT layouts (SM90 is NT-only) and requires scale factors
  in **packed UE8M0 (4 packed into one int32)** with TMA-aligned, transposed SF layouts. API
  surface includes `fp8_gemm_{nt,nn,tn,tt}`, `m_grouped_fp8_gemm_{nt,nn}_contiguous`,
  `m_grouped_fp8_gemm_nt_masked`, **`fp8_fp4_mega_moe`**, and — directly relevant to our DSA
  indexer — **`fp8_mqa_logits` and `fp8_paged_mqa_logits`**. The README's only headline number is
  "up to 1550 TFLOPS on H800"; **no B200 numbers published**.

### "Which open-source project has actually hit near-peak on Blackwell blockscaled FP4 GEMM?"

The honest answer, with what each actually publishes:

| Repo | Claim | Hardware | Quality of evidence |
|---|---|---|---|
| **HazyResearch/ThunderKittens 2.0** | NVFP4 and MXFP8 GEMM "surpasses or matches cuBLAS" | **B200 (SM100)** | [reported] — **no absolute TFLOP/s or shapes published**. The strongest circumstantial number is the ~500 TFLOP/s ≈ 10% delta from `tcgen05.cp` pipelining, which **[inferred]** puts them near ~5 PFLOP/s |
| **Colfax SM120 walkthrough** | 1666 TFLOP/s = **83% of peak**, ~cuBLAS 13.6 parity at 16k/32k | **RTX PRO 6000 (SM120)** — *not* B200 | [verified] full per-version numbers, best-documented FP4 GEMM optimization anywhere |
| **NVIDIA CUTLASS** | Blackwell blockscaled collectives ship in-tree | SM100 | The reference implementation; Colfax's tutorials describe it |
| **TileLang** | MXFP8 block-scaled on SM100, SM120 NVF4 | SM100/SM120 | [verified] feature claim, **no numbers** |
| **DeepGEMM** | FP4 with UE8M0 on SM100 | SM100 | [verified] feature claim, **no B200 numbers** |

**Nobody has published an absolute, shape-annotated, near-peak NVFP4 GEMM number on B200 that I
could verify.** The closest verifiable near-peak FP4 number in public is Colfax's 83% on SM120.
This is a genuine gap — and an opportunity, since it means our own NVFP4 GEMM has no published
bar to clear.

---

## The community layer: lectures, glossaries, and individual writers

### GPU MODE (formerly CUDA MODE) — https://github.com/gpu-mode/lectures

**[verified]** Complete lecture list retrieved. The ones relevant to us:

| # | Title | Speaker |
|---|---|---|
| 8 | CUDA Performance Checklist | Mark Saroufim |
| 12 | Flash Attention | Thomas Viehmann |
| 15 | CUTLASS | Eric Auld |
| 17 | GPU Collective Communication (NCCL) | Dan Johnson |
| 22 | Hacker's Guide to Speculative Decoding in vLLM | Cade Daniel |
| 23 | **Tensor Cores** | **Vijay Thakkar & Pradeep Ramani** (CUTLASS maintainers) |
| 35 | **SGLang Performance Optimization** | **Yineng Zhang** |
| 36 | CUTLASS and FlashAttention 3 | Jay Shah |
| 37 | **Introduction to SASS & GPU Microarchitecture** | Arun Demeure |
| 40 | FlashInfer | Zihao Ye |
| 57 | CuTe | **Cris Cecka** |
| 67 | NCCL & NVSHMEM | Jeff Hammond |
| 75 | GPU Programming Fundamentals + **ThunderKittens** | William Brandon & Simran Arora |
| 78 | Iris: Multi-GPU Programming in Triton | Awad, Osama, Potter |
| 79 | **Mirage (MPK): Compiling LLMs into Mega Kernels** | Mengdi Wu & Xinhao Cheng |
| 84 | Numerics and AI | **Paulius Micikevicius** |
| 86 | Introduction to CuTeDSL | Vicki Wang |
| 103 | Fundamentals of CuTe Layout Algebra | Jack Carlisle & Jay Shah |
| 104 | **Gluon: Tile-Based GPU Programming with Low-Level Control** | Peter Bell, Mario Lezcano, Keren Zhou |

Lectures 23, 57, 84 and 103 are the CUTLASS/NVIDIA maintainer talks the assignment asked about.
**Honest note: I listed these from the repo index; I did not fetch the individual slide decks or
notebooks, so I am not asserting their contents.** Nothing in the index covers Blackwell,
tcgen05 or TMEM directly — that gap is currently filled by Colfax.

### Modal GPU Glossary — https://modal.com/gpu-glossary

**[verified]** Four sections: Device Hardware, Device Software, Host Software, Performance. It is
a well-cross-linked *reference* (e.g. the SM page explains warp context switching at
single-cycle cost, ~1000× faster than a CPU context switch). **Honest assessment: excellent for
onboarding, no novel technique.** Not a source of transferable optimization.

### Simon Boehm — "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"
**[verified, https://siboehm.com/articles/22/CUDA-MMM]** — RTX A6000, **FP32**, cuBLAS = 23,249.6 GFLOPs:

| Kernel | Optimization | GFLOPs | % cuBLAS |
|---|---|---|---|
| 1 | Naive | 309.0 | 1.3% |
| 2 | GMEM coalescing | 1,986.5 | 8.5% |
| 3 | SMEM caching | 2,980.3 | 12.8% |
| 4 | 1-D blocktiling (8 results/thread) | 8,474.7 | 36.5% |
| 5 | 2-D blocktiling (8×8/thread) | 15,971.7 | 68.7% |
| 6 | Vectorized access (transpose As, `float4`) | 18,237.3 | 78.4% |
| 9 | Autotuning BM/BN/BK/TM/TN | 19,721.0 | 84.8% |
| 10 | Warptiling | 21,779.3 | 93.7% |

**Why it still matters in 2026:** the shape of the curve. 80% took "two weekends"; 94% took "four
additional weekends." He explicitly notes double buffering, bank-conflict elimination and
Hopper-era warp specialization are *not* in this list. **Direct relevance to B200 is low** — the
whole hierarchy changed with TMEM and UMMA — but it remains the best pedagogical on-ramp and its
message ("returns follow a power law") is a real budgeting lesson.

### Horace He — "Making Deep Learning Go Brrrr From First Principles"
**[verified, https://horace.io/brrr_intro.html]** — the three-regime framework: **compute-bound**,
**memory-bandwidth-bound**, **overhead-bound**. His diagnostic is arithmetic intensity in
FLOPs/byte; his A100 example: 1.5 TB/s and 19.5 TFLOP/s non-tensor means "you can load 400 billion
numbers in the same time the GPU can perform 20 trillion operations" — so a unary op needs ~100
operations per element before it stops being memory-bound. Operator fusion is "the most important
optimization in deep learning compilers" precisely because a fused `x.cos().cos()` costs
essentially the same as one `cos()`.

On overhead: PyTorch's async queueing hides framework cost **only if the CPU can run ahead of the
GPU** — which fails exactly when tensors are tiny. That is our C1 regime, and it is the same
argument the megakernel papers make quantitatively (5–10 µs launch vs 2 µs kernels).
**Old but structurally correct; the diagnostic discipline is what to steal.**

### Aleksa Gordic — vLLM internals deep dive
**[verified, https://www.aleksagordic.com/blog/vllm]** — a genuinely detailed source-level walk of
vLLM V1. Useful concrete details: KV block size defaults to 16 tokens with block memory
`2 * block_size * num_kv_heads * head_size * dtype_bytes`; prefix caching hashes complete 16-token
blocks via `hash_request_tokens` into `cached_block_hash_to_block` and resolves with
`find_longest_cache_hit` by **linear search**; `MultiProcExecutor` communicates through
`rpc_broadcast_mq` shared-memory queues; DP load balancing scores engines as
`len(waiting)*4 + len(running)`; DP replicas run in **lockstep with dummy steps for idle replicas**
because MoE expert layers need synchronization. Speculative decoding covers n-gram, EAGLE and
Medusa draft methods.
**Relevance:** it is the best public map of what a competitor engine does at the *scheduler* level.
Given ETC's finding that SGLang's CPU scheduler is what keeps it ahead at TP=4/low batch, this
layer deserves as much attention as our kernels.

### Lei Mao — https://leimao.github.io/blog/
**[verified] the blog exists and is active.** The index page I fetched surfaced recent posts on
`torch._assert_async`, predicated-vs-conditional execution across CUDA/TensorRT/XLA, vector and
product quantization, and PyTorch custom C++/CUDA ops. **Honest finding: nothing on CUTLASS, CuTe,
Blackwell, TMA, tcgen05, warp specialization, NCCL or SASS surfaced on the index I could reach.**
His older CUDA fundamentals writing has value as reference, but he is not a Blackwell source.

### Berkeley Sky Computing Lab
**[unverified] — I could not source this.** `https://sky.cs.berkeley.edu/blog/` returned HTTP 404,
and I had no search budget left to locate the correct URL. The lab's headline systems artifacts
(vLLM, SkyPilot, Chatbot Arena/LMSYS lineage) are widely known, but **I am not going to assert
technical content I did not read.** This is a real gap in this report and should be re-run with
search available.

---

## Techniques ranked by transferability to our stack

| Technique | Source | Mechanism in one line | Targets which of our hotspots | Evidence strength | Effort | Risk |
|---|---|---|---|---|---|---|
| Device-initiated one-shot all-reduce on pre-allocated dest buffers (`multimem.*` + VMM multicast) | PK / PGL | Skip NCCL's two-way handshake and channel staging | Collectives 19.6% | **Strong** — 1.79× at their smallest B200 point, mechanism fully explained | Medium | VMM/IPC setup fragility; multi-process correctness |
| Per-tile counter semaphores replacing collective barriers | HazyResearch MK, ETC | `notify()`/`wait()` on global counters; early ranks proceed | Rank arrival skew (9.2% of C1) | **Strong** — 832 ns vs 64 ns sync gap measured | Medium | Deadlock surface; needs careful worst-case rewriting for MoE |
| Data-parallel (replicated) O-projection | HazyResearch TP MK | Removes the post-attention reduce-scatter | Collectives | Medium — 8× traffic claim, no isolated ablation | Low-Med | +~9 GB/GPU; only helps if GLM-5.2's attn-out shape allows |
| `tcgen05.cp` implicit pipeline; delete redundant `fence.proxy`; `elect.sync` | TK 2.0, Colfax block-scaling | Three independent PTX-level fixes | Dense GEMM 37.1%, MoE GEMM 19.4% | **Strong** — individually measured (500 / 20 / 10%) | Low | Requires owning the GEMM kernel, not calling cuBLAS |
| Cluster-dims audit (148 → 132/120/112 SMs) | TK 2.0 | Use `cudaLaunchKernelEx` or cap cluster at 2 | All GEMM | **Strong** — exact SM counts published | Trivial | None |
| Software `exp2` emulation + conditional rescaling | FA4 | Move exponential work from 16 ops/cyc SFU to FMA units | Attention 10.9%, DSA indexer 5.8% | **Strong** — coefficients and feeds/speeds published | Medium | Numerics validation vs reference softmax |
| TMEM for attention intermediates (P, dS) instead of SMEM | FA4, Colfax TMEM | SMEM bandwidth is the unscaled resource on B200 | Attention | Strong | Medium-High | TMEM occupancy limit is 1 block/SM |
| Block-sparse tile skipping (`mask_block_idx`/`full_block_idx`) for DSA | Colfax FlexAttention | Never load fully-masked tiles; skip `mask_mod` on full tiles | DSA indexer + attention | Strong | Medium | Sparsity tile size must equal kernel tile size |
| CLC work stealing for MoE grouped GEMM only | Colfax CLC | Hardware cancellation of unlaunched clusters | MoE GEMM 19.4% | Medium — big on imbalanced, neutral-to-negative on balanced | Medium | Costs L2 hit rate; don't apply to dense GEMM |
| Stream-K / hybrid tail wave for decode-shape GEMMs | Colfax persistent | Split K across SMs for the partial wave only | Dense GEMM | Strong (H100 data) | Medium | Reduction overhead; deterministic mode costs 10–15% |
| Threadblock swizzle for L2 locality | Colfax SM120 | Reorder work-tile assignment within a wave | Prefill GEMM (TTFT 189 ms) | Strong (+387 TFLOP/s at 32k) | Trivial | Shape-dependent; must autotune |
| Warp-specialized epilogue store warp + SF bank-conflict fix | Colfax SM120 | Two named barriers; SMEM stride `(32,4):(4,128)` | NVFP4 GEMM | Strong (8.39M wavefronts removed) | Low | — |
| Static (not dynamic) task schedule at C1 | ETC Table 3 | Precompute per-SM queues on host | Whole decode step | **Strong** — dynamic is 0.83× at BS=1 TP=4 | Low | Contradicts HazyResearch at batch 8192 — regime-specific |
| Full-decode megakernel (persistent, one launch) | ETC, MPK, HazyResearch | Break kernel boundaries entirely | Everything; kills 5–10 µs × N launches | Strong — 1.48× vs vLLM, 1.20× vs SGLang at BS=1 on B200 | **High** | Large engineering; ETC lost to SGLang at TP=4 due to CPU scheduling |
| Inter-SM (not intra-SM) overlap for all-reduce | PK Figure 4 | Accumulate in HBM, then a few SMs do one in-network AR | Collectives | **Strong** — intra-SM AR is 3.62× *worse* | Medium | Must pick per-collective; RS wants the opposite |
| Bulk-prefetch remote KV to local HBM | PK "remote L2 is far-sided" | Peer reads never populate requester L2 | Any cross-GPU reuse | Strong (mechanism) | Low | Only matters if we do cross-GPU KV reads |
| CPU-side scheduler for fused decode attention, reused across layers | ThunderMLA | Async host scheduling, 1–2 ms amortized | Attention, MLA/DSA decode | Strong (H100 data, 20–36%) | Medium | H100 numbers; needs B200 re-derivation |
| CuTe DSL for per-shape kernel specialization | FA4 | 20–30× faster compiles than C++ templates | Build/deploy velocity | Medium | Medium | New toolchain dependency |
| Chiplet-aware task placement on the dual-die B200 | Fleet (arXiv:2604.15379) | Per-chiplet scheduling + cache reuse | Decode latency | Weak — abstract only, AMD-focused | High | Unexplored; may not apply to B200's HBI-unified dies |
| AOT shape-generic compilation to kill warmup | ETC | One shape-generic megakernel compiled offline | Ops, not leaderboard | Strong (583 s → 35 s) | High | No effect on P50 latency once warm |

---

## Sources

All URLs below were fetched and read during this session.

**Stanford HazyResearch**
- https://hazyresearch.stanford.edu/blog — full post index
- https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles — low-latency Llama-1B megakernel
- https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main — tensor-parallel megakernel
- https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro — companion intro
- https://hazyresearch.stanford.edu/blog/2025-09-22-pgl — Parallel Global Layout
- https://hazyresearch.stanford.edu/blog/2025-11-17-pk — ParallelKittens intro
- https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens — multi-GPU kernel principles
- https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla — ThunderMLA
- https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell — ThunderKittens on Blackwell
- https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2 — ThunderKittens 2.0
- https://github.com/HazyResearch/ThunderKittens (+ /tree/main/kernels)
- https://github.com/HazyResearch/Megakernels (+ raw README)
- arXiv:2511.13940 — ParallelKittens (full PDF read)
- arXiv:2410.20399 — ThunderKittens (abstract)
- arXiv:2511.08083 — HipKittens (abstract)

**Colfax Research**
- https://research.colfax-intl.com/blog/ and /category/papers/tutorials/ — indexes
- .../flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/
- .../dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/
- .../cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/
- .../cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
- .../cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/
- .../cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/
- .../cutlass-tutorial-persistent-kernels-and-stream-k/
- .../cutlass-tutorial-design-of-a-gemm-kernel/
- .../a-users-guide-to-flexattention-in-flash-attention-cute-dsl/
- .../optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/
- arXiv:2603.05451 — FlashAttention-4 (abstract)

**CMU Catalyst and orbit**
- arXiv:2604.13327v2 — Event Tensor / ETC, MLSys 2026 (full PDF read)
- https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- https://github.com/mirage-project/mirage (+ raw README)
- https://mirage-project.readthedocs.io/en/latest/
- https://github.com/flashinfer-ai/flashinfer (raw README)

**Other megakernel / microarchitecture papers (arXiv API)**
- arXiv:2605.11581 — Ada-MK
- arXiv:2604.15379 — Fleet
- arXiv:2512.02189v3 — Microbenchmarking Blackwell (full PDF read)
- arXiv:2507.10789v2 — Dissecting Blackwell
- arXiv:2605.04178 — Microbenchmark-driven analytical modeling
- arXiv:2607.23402 — Characterizing Warp Divergence Pascal→Blackwell (SASS)
- arXiv:2608.11693v2 — Spec Sheets Are Not Kernels (INT8 on sm_103a)
- arXiv:2604.23466v2 — Evaluating CUDA Tile on Hopper and Blackwell

**Community / tooling**
- https://github.com/gpu-mode/lectures (+ raw README) — full lecture index
- https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor
- https://siboehm.com/articles/22/CUDA-MMM
- https://horace.io/brrr_intro.html
- https://www.aleksagordic.com/blog/vllm
- https://leimao.github.io/blog/
- https://raw.githubusercontent.com/tile-ai/tilelang/main/README.md
- https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md

**Could not source**
- Berkeley Sky Computing Lab blog (`sky.cs.berkeley.edu/blog/` → HTTP 404; no search budget remaining)
- Gluon documentation / GPU MODE Lecture 104 content
- Any public SASS-level teardown of `tcgen05` specifically
- Any absolute, shape-annotated near-peak NVFP4 GEMM number on B200 from an open-source project
