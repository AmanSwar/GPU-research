# The batch-1 problem: literature on minimum-latency single-stream decoding

## What this is

A survey of everything published that bears on the question *"how fast can one
request go?"* — a regime almost all systems research treats as a footnote on the
way to throughput. The corpus here was assembled by fetching and reading the
papers, blogs and docs cited; every number carries the hardware and model it was
measured on, and a confidence label:

- `[verified]` — I read the number in the paper/blog body.
- `[reported]` — the authors claim it in an abstract or release note I read, but
  I did not see the underlying table.
- `[inferred]` — my own arithmetic from stated hardware specs or our own profile.
  Reproducible; not a citation.

The organising claim of the document, which almost every source independently
confirms: **at batch 1 on a fast GPU you are not bandwidth-bound.** The
"tok/s ≈ HBM_bandwidth / bytes_per_token" model is a *ceiling*, and real engines
sit at 20–35% of it. The gap is launch overhead, kernel-boundary
serialisation, memory-pipeline warm-up, collective latency and rank skew — none
of which appear in the roofline. Everything in this document is about closing
that gap.

Our system: 8× B200 SXM (8 TB/s HBM each → 64 TB/s aggregate, 1.8 TB/s NVLink5
per GPU, NV18 all-to-all) `[verified — NVIDIA DGX B200 product page]`, GLM-5.2
MoE (256 experts / 8 active, DSA sparse MLA, `index_topk_freq=4`), TP8, EAGLE
3-1-4, 365 tok/s single-stream, 78–95 tok/s/stream at C16. The engine to beat is
TileRT at ~500 tok/s on GLM-5 FP8, same hardware `[reported — TileRT README]`.

---

## Bottom line for our system

Ranked by expected effect on single-stream tok/s, with the reasoning.

1. **We are at ~23% of the HBM roofline; the headroom is 4x, not 40%.** At 365
   tok/s the step is 2.74 ms, in which 8× B200 could read 175 GB. GLM-5.2's
   active-parameter footprint in FP8 is nowhere near that. Every published
   whole-model-kernel result says the missing 77% is *structural overhead*, not
   memory. The single highest-value experiment is to instrument one decode step
   for GPU-idle-between-kernels and publish that number. `[inferred]`
   Corroboration: the same measurement on H100 with a 7–8B model found **27% of
   peak HBM bandwidth achieved**, versus **81% on an L4**, with the difference
   attributed entirely to per-kernel launch overhead becoming visible once memory
   stops being the binding constraint (Chen, arXiv:2605.30571) `[verified]`.

2. **Kill the ~9.2% of step time that is pure rank-arrival skew in collectives.**
   19.6% of our step is collectives and 47% of that is ranks waiting for each
   other = **252 µs per token, 9.2% of the step**. `[inferred, from our profile]`
   Our per-allreduce cost works out to ~2.9 µs across ~184 collectives/token,
   against a measured NVLink speed-of-light of **1.404 µs** on GB200 and a
   near-SoL one-shot kernel at ~1.5 µs (Shen et al., ETH Zürich + NVIDIA,
   arXiv:2607.16100) `[verified]`. Halving collective time is worth ~10% of
   single-stream speed. The levers, in order: barrier-free / sentinel
   synchronisation instead of a global barrier per collective; one-shot push-mode
   allreduce for small messages; fusing AllReduce+RMSNorm so the reduction and
   the norm are one kernel on 2–8 SMs (TokenWeave, MSR India, MLSys 2026)
   `[reported]`.

3. **TP8 is the right choice for latency on one node. Do not go to TP16.** The
   arithmetic below (§4) puts TP8 at the optimum: the weight-read term falls as
   1/n while the collective term is roughly flat inside an NVLink domain, but
   crossing to a second node adds ~5 µs per collective × 184 collectives = ~0.9 ms,
   which swamps the ~0.3 ms of memory-term saving. Independently measured:
   TPOT went from **0.86 ms at TP4 (intra-node) to 11.56 ms at TP8 (inter-node)**
   on Llama-3.1-8B, 4× H100/node + IB NDR400 (Xu et al., Hot Interconnects 2025)
   `[verified]`.

4. **A megakernel / persistent-kernel decode path is the single biggest known
   structural win, and it is now a real technique, not one paper.** Six
   independent implementations, three of them industrial. Reference points:
   MPK reduces Qwen3-8B per-token decode on A100 from **14.5 ms (vLLM/SGLang) to
   12.5 ms**, against a 10 ms bandwidth bound `[verified]`; ETC (Event Tensor) on
   **8× B200** hits **1.48× over vLLM and 1.20× over SGLang on Qwen3-30B-A3B MoE
   at batch 1** `[verified]`; TileRT — the engine we are chasing — is a persistent
   tile-scheduled runtime `[reported]`. Caveat that matters to us: ETC's TP4
   result was only **0.99–1.06× vs vLLM, and SGLang still won**, because SGLang's
   CPU scheduling was better `[verified]`. Megakernels pay most when you have
   already fixed the host side.

5. **Speculative decoding is our best-in-class lever and is already working
   (3.09×).** Published ceilings are higher: EAGLE-3 reports mean acceptance
   length **τ = 5.88–6.62** and **4.12–5.51× speedup at T=0** on H100 for
   Vicuna-13B / Llama-3.1-8B / Llama-3.3-70B `[verified]`. Our 3-1-4 config
   (3 draft steps, top-k 1 chain, 4 draft tokens) caps accepted length at 4 by
   construction. Moving to a tree (top-k > 1) with a longer draft and an EAGLE-3
   style multi-layer-fusion head is the obvious next step; the risk is that tree
   verification widens the MoE expert footprint (see §8).

6. **Attention + DSA indexer are 16.7% of our C1 step and they are what makes
   concurrency expensive, not the MoE.** In the C1→C16 decomposition (§8) they
   contribute 2.67 of the 4.74× step-time growth, versus 1.43 from MoE expert
   coverage. Anything that shrinks per-sequence KV bytes (more aggressive DSA
   top-k, cross-layer index reuse) buys per-stream speed *at concurrency*, which
   is where the money is.

7. **Eliminate every dynamic shape and every host round-trip on the decode path.**
   ETC reports vLLM taking **123 s of warmup with 67 JIT captures** and SGLang
   **583 s with 51 captures**, versus **35 s and zero** for an AOT-compiled
   megakernel `[verified]` — that is the cost of shape polymorphism showing up as
   graph recapture. At batch 1 with speculative decoding you have exactly one
   shape; there is no excuse for a recapture, a `.item()`, or a host-side
   branch inside the step.

8. **Spin, don't block; pin the cores.** `cudaDeviceScheduleSpin` vs
   `cudaDeviceScheduleBlockingSync` is a real microsecond-scale lever on the host
   wait `[CUDA Runtime API docs — flags verified to exist; semantics standard]`.
   Empirical support that host single-thread speed matters: TaxBreak measured a
   **10–29% reduction in orchestration overhead and 14% end-to-end latency
   improvement** purely from a faster host CPU (Emerald Rapids vs Sapphire
   Rapids) with the *same class* of GPU `[verified]`.

9. **Accept that batch-1 economics are terrible, and price it that way.**
   SemiAnalysis InferenceX puts DeepSeek-R1 FP4 at **~$0.56/M output tokens at
   50 tok/s/user and ~$4/M at 125 tok/s/user** — 2.5× the speed for ~7× the cost
   `[reported]`. TileRT's decode node **serves exactly one in-flight request**
   `[verified — SemiAnalysis]`. There is no configuration that is simultaneously
   the latency champion and the cost champion; the literature is unanimous. Sell
   two SKUs.

10. **Our 4.7× per-stream decay from C1 to C16 is arithmetically normal for this
    model, not a scheduler bug.** A five-term decomposition of our own C1 hotspot
    profile predicts **4.74×** (§8). The number to be unhappy about is not the
    decay, it is the C1 starting point being only 23% of roofline. Also: our
    "40.8k tok/s at C64" is inconsistent with 78–95 tok/s/stream at C16
    (40800/64 = 637 tok/s/user > the C1 number) — that figure is almost certainly
    counting input+output tokens or a different workload and should be
    re-measured before it is used in any comparison. `[inferred]`

---

## 0. The arithmetic: what batch-1 decode is actually bound by

### 0.1 The naive roofline

At batch 1 with no speculation, one decode step reads every byte the forward pass
touches and does ~2 FLOPs per byte. Arithmetic intensity ≈ 2, versus a B200's
FP8 dense compute/bandwidth ratio in the thousands. So:

```
tok/s_ceiling  =  (N_gpu × BW_per_gpu) / bytes_read_per_token
```

For 8× B200 at 8 TB/s each = **64 TB/s aggregate**, with weights sharded TP8 so
every GPU reads 1/8 of the bytes in parallel `[inferred]`:

| active bytes/token (FP8) | ceiling tok/s | step time |
|---|---|---|
| 20 GB | 3200 | 0.31 ms |
| 30 GB | 2133 | 0.47 ms |
| 40 GB | 1600 | 0.63 ms |
| 50 GB | 1280 | 0.78 ms |
| 60 GB | 1067 | 0.94 ms |
| 80 GB | 800 | 1.25 ms |

Our 365 tok/s = 2.74 ms/step, during which the machine *could* have read 175 GB.
TileRT's ~500 tok/s = 2.00 ms/step = 128 GB readable. Both are far below the
ceiling for any plausible GLM-5.2 active footprint. `[inferred]`

### 0.2 The five things that break the roofline

Every source in this document is attacking one of these.

**(a) Kernel launch overhead.** Measurements, all on real hardware, disagree by
an order of magnitude — and the disagreement is itself informative:

| measurement | value | context | source |
|---|---|---|---|
| dummy kernel, CUDA stream | ~2.1 µs | H100 | Hazy Research `[verified]` |
| dummy kernel, CUDA graph | ~1.3 µs | H100 | Hazy Research `[verified]` |
| per launch, eager | 3.8 µs | Qwen3-8B, 293 launches/token | MPK `[verified]` |
| per launch, CUDA Graphs | 0.8 µs (0.2 ms/token total) | same | MPK `[verified]` |
| null-kernel dispatch→start floor | ~4.7 µs | H100 | TaxBreak `[verified]` |
| cuBLAS GEMM launch | 6.63 µs | H100 | TaxBreak `[verified]` |
| launch band as % of step | 20.6% (3.05 ms of 14.83 ms) | H100, HF eager, ~281 kernels/step | arXiv:2605.30571 `[verified]` |
| launch overhead as % of e2e | 14.6% | L20, production ad-serving | Ada-MK `[reported]` |

The reconciliation: the ~30 µs/kernel implied by arXiv:2605.30571 is *eager
PyTorch/HuggingFace*, not a serving engine. The MPK numbers (3.8 µs eager, 0.8 µs
graphed) are the ones that apply to us. But note MPK's other number: **293 kernel
launches per token for a dense 8B model.** Our MoE is far worse — TaxBreak
measured **9,305 kernels per output token for OLMoE-1B/7B vs 848 for dense
Llama-3.2-1B, i.e. 8–11× more kernel dispatches for MoE** `[verified]`.

**(b) Kernel-boundary serialisation.** Independent of launch cost, CUDA's
ordering guarantee means *no* block of kernel *k+1* may start until *every* block
of kernel *k* has finished. Hazy Research's framing: a kernel with 512 blocks on
a 148-SM B200 runs 4 waves, and at the end of the last wave 80 SMs are idle
waiting for the tail `[verified]`. At batch 1 the tail is a large fraction of a
short kernel.

**(c) Memory-pipeline warm-up and drain.** Each kernel must issue its loads,
wait out HBM latency (hundreds of ns to low µs), compute, and store — and it
cannot prefetch the *next* kernel's weights because it does not know about them.
Hazy's B200 breakdown of a 600 µs Llama-1B forward pass `[verified]`:

| component | µs |
|---|---|
| activation store + sync + reload | 250 |
| RMSNorm + matvec (95% of which is actual matvec) | 200 |
| waiting on weights from global memory | 30 |
| low-level warp sync | 40 |
| setup / misc | 80 |

**A quarter of the step is storing activations and reading them back** — pure
kernel-boundary tax.

**(d) Collective latency.** ~184 allreduces per token at TP8 with ~92 layers,
each with a hard latency floor (§3).

**(e) Host-side scheduling.** Python dispatch, scheduler, sampling, detokenise
(§2).

### 0.3 What the numbers say about us

Plugging our own hotspot profile into the collective budget `[inferred]`:

```
step at 365 tok/s                = 2.74 ms
collectives (19.6%)              = 537 µs
  ÷ ~184 allreduces/token        = 2.92 µs each
  of which rank-arrival skew 47% = 252 µs = 9.2% of the step
```

2.92 µs per small allreduce on NVLink5 is *already better than NCCL ring*
(11.0 µs on 4× GB200 `[verified]`) — so a custom allreduce is evidently in play —
but ~2× the near-SoL 1.5 µs achievable `[verified]`. And the 252 µs of skew is
not communication at all; it is load imbalance and jitter across ranks.

---

## 1. Whole-model kernels: megakernels and persistent runtimes

The dominant idea in the batch-1 literature since mid-2025. Instead of launching
N kernels per layer, launch *one* persistent kernel for the entire forward pass
and run an interpreter/scheduler inside it.

| paper / system | lab | venue+year | hardware | headline result | in production? |
|---|---|---|---|---|---|
| Look Ma, No Bubbles! (Llama-1B megakernel) | Hazy Research, Stanford | blog, 2025-05-27 | H100, B200 | 78% of H100 memory bandwidth at bs=1; 1350 fwd/s vs vLLM 540, SGLang 870 `[verified]` | research (ThunderKittens) |
| MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs (arXiv:2512.22219) | CMU / UW / Berkeley / NVIDIA / Tsinghua (Z. Jia) | arXiv Dec 2025 | A100, H100, B200 | Qwen3-8B A100 bs=1: 14.5 ms → 12.5 ms (bound 10 ms); 1.0–1.7× vs SGLang/vLLM; 1.1–1.4× on 8×H100 TP `[verified]` | open source (mirage-project) |
| Event Tensor / ETC (arXiv:2604.13327) | CMU / NVIDIA / OctoAI-lineage (Jin, Jia, T. Chen) | arXiv Apr 2026 | **8× B200** | Qwen3-30B-A3B **MoE** bs=1: **1.48× vs vLLM, 1.20× vs SGLang**; TP4 only 0.99–1.06× `[verified]` | open source |
| FlashFormer: Whole-Model Kernels for Efficient Low-Batch Inference (arXiv:2505.22758) | MIT / MIT-IBM (Nrusimha, Brandon, Ragan-Kelley, Kim) | arXiv May 2025 | H100 SXM, 1 GPU | Llama-3.1-8B BF16 bs=1: 8–20% over GPTFast, up to 61% over vLLM; 184 vs 168 tok/s at seqlen 128 `[verified]` | research |
| Ada-MK (arXiv:2605.11581) | Baidu (ad-serving) | arXiv May 2026 | NVIDIA L20 | +23.6% single-batch throughput vs TensorRT-LLM, +50.2% vs vLLM; "launch overhead alone can account for 14.6% of e2e" `[reported]` | **yes — commercial ad system** |
| Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs (arXiv:2604.15379) | AMD | arXiv Apr 2026 | MI350 | Qwen3-8B bs=1–8: **1.3–1.5× lower decode latency than vLLM**; L2 hit 12%→54% at bs=32 `[reported]` | vendor research |
| TileRT | tile-ai / Zhipu | GitHub + SemiAnalysis, 2026 | **8× B200** | GLM-5 FP8 ~500 tok/s single-token; DeepSeek-V3.2 FP8 ~600 tok/s; MiMo-V2.5-Pro (1T) >1000 tok/s `[reported]`; 494.2 tok/s/user at 1k/1k `[reported]` | **yes — Zhipu GLM-5.1-HighSpeed, Xiaomi MiMo UltraSpeed** |
| ClusterFusion (arXiv:2508.18850) | SJTU | arXiv Aug 2025 | H100 | 1.61× average e2e latency via cluster-level ClusterReduce/ClusterGather `[reported]` | research |
| Perseus: Eliminating Hidden Serialization in Multi-Node Megakernel Communication (arXiv:2605.00686) | Cornell (Oh, Singh) | arXiv May 2026 | multi-node RDMA | MoE megakernels **regress up to 10× on 8 nodes**; Perseus recovers up to 10.3× `[reported]` | research |
| RaMP (arXiv:2604.26039) | — | arXiv Apr 2026 | 8 architectures | 1.22× kernel, 1.30× e2e in vLLM over Triton `[reported]` | research |
| AutoMegaKernel (arXiv:2606.09682) | Jaber & Jaber | arXiv Jun 2026 | L4/L40S/A10G/5090 | W8A16 megakernel beats CUDA-graphed cuBLAS bf16 at bs=1 on *inference-class* GPUs (L4 1.33×) but **trails on A100/H100** `[reported]` | research |
| GPUOS (arXiv:2604.17861) | UCSC | arXiv Apr 2026 | — | persistent kernel + NVRTC runtime operator injection, up to 15.3× on small-op workloads `[reported]` | research |

### 1.1 The mechanism, in enough detail to build it

All of these share four pieces. Hazy Research's blog is the clearest exposition;
MPK and ETC are the compiler-automated versions.

**Piece 1 — an on-GPU instruction interpreter.** The kernel launches once with a
grid sized to the SM count and never exits. Each SM (or CTA) reads a
pre-computed instruction stream from global memory and dispatches to a
template. Hazy's Llama-1B decode uses **seven fused instruction types**
`[verified]`:

```
1. RMSNorm + QKV projection + RoPE
2. attention
3. attention reduction (long-sequence split-K combine)
4. output projection + residual
5. RMSNorm + up/gate + SiLU
6. down projection + residual
7. RMSNorm + LM head
```

The schedule (which SM runs which instruction in which order) is computed **once,
in Python, and reused across hundreds of forward passes** `[verified]`. This is
only possible because at batch 1 the shapes never change — the technique is
*structurally* a batch-1 technique.

**Piece 2 — paged shared memory.** SMEM is the scarce resource and the whole
point is to start loading instruction *k+1*'s weights while instruction *k* is
still storing. Hazy divides the first **213 kB of H100 shared memory into
thirteen 16 KiB pages**; instructions request and release pages explicitly, and
the interpreter hands a freed page to the next waiter immediately `[verified]`.
That handoff is what removes the memory-pipeline bubble.

**Piece 3 — counter-based fine-grained synchronisation.** Kernel boundaries gave
you a free global barrier; inside a megakernel you must build one. The pattern:
an array of counters in global memory, initialised to zero. On completing a unit
of work an instruction `atomicAdd`s its counter; a dependent instruction spins
until the counter reaches its target. Crucially this is **finer than a barrier** —
Hazy splits the MLP hidden state into **four chunks with independent counters**
so the down-projection can start on chunk 0 while chunk 3 is still being
computed `[verified]`. This is the specific thing Programmatic Dependent Launch
(PDL) *cannot* do: PDL's dependency is whole-kernel-to-whole-kernel, so
attention must wait for *all* of Q, K and V rather than starting per-head
`[verified — Hazy]`.

FlashFormer uses the simpler variant: **atomic-based global synchronisation plus
per-thread global memory fences** to make writes visible before dependents
proceed `[verified]`.

**Piece 4 — an in-kernel scheduler.** MPK partitions SMs into **workers** (run
tasks, signal completion events) and **schedulers** (distributed across warps,
dequeue activated events, enqueue dependent tasks). Queues are **circular buffers
in device memory manipulated with `atomicAdd`** `[verified]`. Measured cost: the
in-kernel scheduler is **0.28% of total runtime**, per-task overhead **1–2 µs**
`[verified]`.

### 1.2 What megakernels do *not* fix, and where they break

Three honest limits, all documented:

- **Dynamic shapes.** The static schedule is the source of the win and the source
  of the fragility. ETC's entire contribution is a symbolic-shape "Event Tensor"
  abstraction so one AOT-compiled template covers variable batch sizes without
  recapture; the payoff is measured in warmup, not steady state: **35 s / 0
  JIT captures vs vLLM 123 s / 67 and SGLang 583 s / 51** `[verified]`.
- **Tensor parallelism dilutes the win.** ETC on 8× B200: 1.48× over vLLM at
  bs=1 single-GPU, but at **TP4 only 0.99–1.06×, and SGLang beat it** because
  SGLang's CPU scheduling was better `[verified]`. Once collectives and host
  scheduling dominate, removing launch gaps does less.
- **Multi-node megakernels are actively worse without transport work.** Perseus
  found MoE megakernels **regress up to 10× on 8 nodes** because proxy-based RDMA
  fences drain the NIC pipeline once per tile transfer `[reported]`. Relevant to
  us only if we ever leave the node — an additional argument for staying at TP8
  in one NVSwitch domain.
- **Bandwidth class matters.** AutoMegaKernel's honest negative result: a
  search-found megakernel beat CUDA-graphed cuBLAS on L4/L40S/A10G/5090 but
  **trailed on A100/H100**, and the authors localise the bottleneck to cross-SM
  synchronisation `[reported]`. The faster the memory, the more the global
  counter/fence traffic costs relative to the work.

**Implication for us:** the megakernel win on 8× B200 with a big MoE at TP8 is
real but is gated on the collective and host paths being fixed first. ETC's TP4
result is the warning.

---

## 2. Host-side overhead: launch, Python, CUDA graphs, and the move off Python

| paper / system | lab | venue+year | hardware | headline result | in production? |
|---|---|---|---|---|---|
| TaxBreak: Unmasking the Hidden Costs of LLM Inference (arXiv:2603.12465) | CMU (Vellaisamy, Blanton, Shen) | arXiv Mar 2026 | H100, H200 | MoE decode is **host-bound**: HDBI 0.10 (OLMoE) vs 0.23 (dense); GPU idle 73–81% even at bs=16 `[verified]` | analysis |
| Characterizing and Optimizing LLM Inference on CPU-GPU Coupled Architectures (arXiv:2504.11750) | CMU + Intel | ISPASS 2025 | GH200 vs PCIe H100 | GH200 stays CPU-bound to **4× larger batch sizes** than loosely-coupled `[reported]` | analysis |
| Memory-Bound but Not Bandwidth-Limited (arXiv:2605.30571) | Josef Chen | arXiv May 2026 | H100/A100/L40S/L4 | achieved HBM fraction **27% H100 vs 81% L4**; CUDA graphs **1.259× H100 vs 1.028× L4** `[verified]` | analysis |
| SGLang v0.4 zero-overhead batch scheduler | LMSYS | blog, 2024-12-04 | (Llama-3.2-3B) | "unoptimized engine can spend **as much as half its time on CPU overhead**"; 1.1× vs v0.3, 1.3× vs baselines; Nsight shows **no GPU idle across 5 decode batches** `[verified]` | **yes — SGLang** |
| vLLM V1 architecture | vLLM | blog, 2025-01-27 | H100 | isolated `EngineCore` process; persistent batch (diff-only input tensors); zero-overhead prefix caching (<1% loss at 0% hit rate); **up to 1.7× throughput vs V0** `[verified]` | **yes — vLLM** |
| Hybrid JIT-CUDA Graph Optimization (arXiv:2604.23467) | Yadav & Zhao | arXiv Apr 2026 | Llama-2 7B | TTFT −66.0% vs TensorRT-LLM in single-batch `[reported]` | research |
| NanoFlow (arXiv:2408.12757) | UW SyFI (K. Zhu, Kasikci) | **OSDI 2025** | NVIDIA GPUs | nano-batching + operation-level pipeline; 1.91× throughput, 59–72% of optimal `[reported]` | ideas adopted in SGLang |
| FastUSP (arXiv:2602.10940) | Guandong Li | arXiv Feb 2026 | FLUX/Qwen-Image | "**kernel launch overhead — rather than communication latency — is the primary bottleneck on modern high-bandwidth GPU interconnects**" `[reported]` | research |

### 2.1 The measured cost of Python

TaxBreak decomposes host orchestration into three mutually exclusive components
and this taxonomy is worth stealing `[verified]`:

- **ΔFT (framework translation)** — Python dispatch + ATen operator resolution.
- **ΔCT (CUDA-library translation)** — cuBLAS/cuDNN descriptor setup, heuristic
  selection. Zero for hand-written/Inductor kernels.
- **ΔKT (kernel launch)** — `cudaLaunchKernel` call to GPU kernel start; the
  hardware floor.

Their H100 numbers: **null-kernel floor 4.7 µs; scan/elementwise 5.1–5.3 µs
(7–12% above floor); cuBLAS GEMM 6.63 µs (40% above floor), of which 1.88 µs is
framework residual** `[verified]`.

The MoE result is the one that should worry us. On H100 at bs=4/seqlen 2048,
decoding 10 tokens `[verified]`:

| model | kernels total | kernels/output token |
|---|---|---|
| OLMoE-1B/7B | 93,053 | 9,305 |
| Qwen1.5-MoE-A2.7B | 66,951 | 6,695 |
| Llama-3.2-1B (dense) | 8,475 | 848 |

and the Host-Device Balance Index (fraction of step that is GPU work) for MoE
decode is **0.10–0.15 vs 0.23 dense, with GPU idle fraction 73–81% persisting to
batch 16** `[verified]`. These are eager-PyTorch measurements — a CUDA-graphed
SGLang/vLLM path is much better — but the *ratio* between MoE and dense kernel
counts is architectural and carries over. **An MoE at batch 1 is the single
worst case for host-side overhead in all of LLM serving.**

### 2.2 CUDA graphs: how much they actually buy, and where they stop

- **1.259× on H100 (95% CI [1.253, 1.267]) vs 1.028× on L4**, same model, same
  context (arXiv:2605.30571) `[verified]`. The intervention *only pays on fast
  memory* — exactly our regime.
- Per-launch 3.8 µs → 0.8 µs; total launch cost for Qwen3-8B 293 launches/token
  → **0.2 ms/token even with graphs** (MPK) `[verified]`.
- Hazy: dummy kernel 2.1 µs on a stream → **1.3 µs with graphs**; the residual is
  irreducible under the kernel abstraction `[verified]`.

That residual is the whole argument for megakernels. Hazy's projection: 7 kernels
per layer × 16 layers with an optimistic 5 µs stall each puts Llama-1B at
**~770 forward passes/s against a 1350/s memory-bound limit** `[verified]`.

**Costs of graphs that are real for us:** every distinct shape needs a capture
(ETC measured 51–67 captures and 123–583 s of warmup in vLLM/SGLang
`[verified]`); graphs freeze pointer values so KV-cache and buffer addresses must
be stable; and a graph cannot contain a host round-trip, which is a feature.

### 2.3 Overlapping the host entirely

SGLang v0.4's zero-overhead batch scheduler, credited to NanoFlow, is the design
to copy: **run the scheduler one batch ahead**, preparing all metadata for step
*k+1* while step *k* executes on the GPU, and resolve the dependency on the
not-yet-sampled token via *future tokens* plus careful CUDA event coordination
`[verified]`. Nsight profiles show **no GPU idle across 5 consecutive decode
batches** with the Triton attention backend (minor gaps remain with FlashInfer)
`[verified]`.

vLLM V1's complements: an isolated `EngineCore` process so tokenisation,
multimodal preprocessing, detokenisation and streaming overlap the model loop;
and **persistent batch** — cache the input tensors and apply only diffs each
step, rather than rebuilding them `[verified]`.

**Flags to check on our stack** `[verified — SGLang server-arguments docs]`:
`--enable-torch-compile` (documented as helping "small models on small batch
sizes", but noted as out of maintenance), `--attention-backend`,
`--speculative-algorithm EAGLE3`, `--enable-custom-allreduce`,
`--enable-symm-mem`, `--stream-interval` (larger = higher throughput, smaller =
smoother streaming — at batch 1 this is a pure latency/jitter knob),
`--cuda-graph-max-bs`. vLLM's equivalent async-scheduling flag is
`--async-scheduling` `[verified — vLLM InferenceMAX blog]`.

---

## 3. The collective latency floor at batch 1

| paper / system | lab | venue+year | hardware | headline result | in production? |
|---|---|---|---|---|---|
| Every µs Matters: Achieving Near Speed-of-Light Latency in GPU Collectives (arXiv:2607.16100) | ETH Zürich + NVIDIA (Shen, Jeaugey, Hoefler) | arXiv Jul 2026 | **GB200** | SoL = **1.404 µs**; one-shot at **~1.50 µs (7% over SoL)**; NCCL ring 11.0 µs → 2.37 µs on 4 GPUs; **vLLM ITL −8.7% at TP4, −9–11% at TP8** `[verified]` | NCCL device-side APIs upstreaming |
| TokenWeave (arXiv:2505.11329) | Microsoft Research India | **MLSys 2026** | 8× H100 | fused AllReduce+RMSNorm on **2–8 SMs** via NVSHARP/multimem; up to **1.28× latency, 1.19× throughput**; overlap works down to 1024 tokens `[reported]` | open source (Microsoft) |
| Characterizing Communication Patterns in Distributed LLM Inference (arXiv:2507.14392) | Ohio State (Panda group) | Hot Interconnects 2025 | 4–8× H100, NVLink + IB NDR400 | decode allreduce shapes **[1, 4096]**; TPOT **0.86 ms TP4 intra-node → 11.56 ms TP8 inter-node** `[verified]` | analysis |

### 3.1 What the floor actually is

Shen et al. derive the hardware lower bound for a 2-GPU NVLink allreduce on
GB200 `[verified]`:

```
L_SoL = 2 × L2_RTT + L_remote_store
      = 2 × 0.306 µs + 0.792 µs
      = 1.404 µs
```

That is the *physics*. Everything above it is software. Their design principles,
which are directly implementable:

1. **Barrier-free synchronisation.** A global memory barrier costs **>1 µs each**
   — comparable to the entire SoL budget `[verified]`. Replace it with:
   - **LL (low-latency) packing**: embed an 8-byte flag alongside 8 bytes of data
     in a single 16-byte atomic store, so arrival and payload are one operation.
   - **Sentinel synchronisation**: pre-fill the buffer with an impossible value
     (e.g. `-NaN`) and poll for change, rather than a separate flag.
   - **Double buffering**: in bidirectional exchange, each *receive* is implicit
     permission for the next *send* — no explicit handshake at all.
2. **One-shot, push mode, for small messages.** Each GPU pushes its buffer to all
   peers and reduces locally: one phase, and *push* costs half an RTT versus
   *pull*'s full RTT `[verified]`. Two-shot (reduce-scatter + all-gather) cuts
   volume from O(N·M) to O(M) but adds a synchronisation phase — correct only
   once messages get large.
3. **Symmetric memory + NVLink SHARP multicast** (`multimem.ld_reduce`) for
   larger rank counts: at 64 GPUs their multicast one-shot stays within ~70% over
   SoL (~2.3 µs) `[verified]`.
4. **LL128 atomic two-shot**: cache-line-granularity atomic adds over NVLink,
   needing only D/N scratch instead of 2D — but restricted to FP32/FP16 addition
   because of vectorised-atomic constraints `[verified]`.

End-to-end payoff on real serving: **ITL −8.7% at TP4 and −9–11% at TP8** on
Llama-3.1-70B, with DeepSeek-V3 and Qwen3-Next also tested; they quantify it as
**~0.9% cost saving per microsecond removed from AllReduce latency**
`[verified]`.

### 3.2 Fusing the collective with the norm

TokenWeave's observation is that RMSNorm sits immediately after the allreduce in
every transformer block and is itself a reduction. Fusing them into **one kernel
that uses only 2–8 SMs** means (a) one launch instead of two, (b) the normalised
output never round-trips HBM, and (c) the remaining 140+ SMs are free for
overlapping compute `[reported]`. They report up to **1.28× latency improvement
on 8× H100**, and the striking claim that in several settings TokenWeave **beats
an equivalent model with all communication removed** — i.e. the fused kernel's
better memory behaviour more than pays for the communication `[reported]`.
vLLM's Blackwell work lists `AllReduce + RMSNorm + Quant` as a shipped
torch.compile fusion `[verified]`.

### 3.3 Rank-arrival skew — our specific problem

**None of the collective papers measure this.** They benchmark allreduce in
isolation, where all ranks arrive together by construction. Our profile says
**47% of collective time is ranks waiting for other ranks** — which means the
allreduce kernel is not the problem; the *dispersion in arrival time* is.

Causes to check, in order of likelihood `[inferred]`:

- **MoE routing imbalance.** With EP or with TP-sharded experts, different ranks
  do different amounts of expert work per step. At batch 1 with 8 active experts
  over 8 ranks the variance is maximal — a rank may get 0 or 3 experts. This is
  the leading hypothesis and it is testable: correlate per-rank pre-allreduce
  arrival time with per-rank expert-token count.
- **Clock/power skew.** B200s under different SM occupancy boost differently.
- **Host jitter**, if any rank's launch is host-driven.
- **DSA indexer imbalance** — `index_topk` work varies with content.

Fixes that follow: redundant/replicated hot experts to flatten per-rank work
(DeepSeek runs **32 redundant experts in prefill** for exactly this reason
`[verified — DeepSeek-V3 tech report]`); padding expert work to a fixed shape so
every rank does identical work regardless of routing; or moving to a
communication pattern that does not require all ranks to arrive (one-shot push
with sentinel polling degrades gracefully — a late rank delays only its own
consumers).

---

## 4. Is more tensor parallelism always better at batch 1? Do the arithmetic.

**Short answer: no, there is an optimum, and on 8× B200 in one NVSwitch domain it
is TP8.**

### 4.1 The model

```
T_step(n) = W/(n·B)            weight + KV reads, falls as 1/n
          + L·c·A(n)           collectives: L layers × c per layer × allreduce latency
          + F                  fixed host/launch/sync overhead, independent of n
```

`W` = active bytes/token, `B` = 8 TB/s per-GPU HBM, `L` ≈ 92 layers, `c` = 2
(one after attention output projection, one after the MLP/MoE), `A(n)` = small
allreduce latency at n ranks, `F` = everything the roofline ignores.

Using `A` values grounded in the ETH/NVIDIA measurements (≈1.5 µs at 2–4 ranks
near-SoL, our measured 2.9 µs at 8 ranks, ≈8 µs once a collective crosses to a
second node over IB) and `W` = 40 GB, `F` = 0.5 ms `[inferred]`:

| TP | memory term | collective term | fixed | total | tok/s |
|---|---|---|---|---|---|
| 1 | 5000 µs | 0 | 500 µs | 5.50 ms | 182 |
| 2 | 2500 µs | 276 µs | 500 µs | 3.28 ms | 305 |
| 4 | 1250 µs | 460 µs | 500 µs | 2.21 ms | 452 |
| **8** | **625 µs** | **534 µs** | **500 µs** | **1.66 ms** | **603** |
| 16 (2 nodes) | 312 µs | 1472 µs | 500 µs | 2.28 ms | 438 |

### 4.2 The crossover condition

Doubling TP from *n* to *2n* is worth it iff the memory saving exceeds the added
collective cost:

```
W/(2·n·B)  >  L · c · (A(2n) − A(n))
```

At n=8 → 16 crossing a node boundary: LHS = 40e9/(2·8·8e12) = **312 µs**;
RHS = 92 × 2 × 5 µs = **920 µs**. Not worth it, by 3×. `[inferred]`

Within an NVLink domain, `A(2n) − A(n)` is small — Shen et al. measure one-shot
allreduce at ~1.5 µs for 2–4 GPUs rising to only ~2.3 µs at 64 GPUs with
multicast `[verified]` — so inside the domain, more TP essentially always wins at
batch 1. The break is at the domain boundary. This is why 8 GPUs on one NVSwitch
is the sweet spot and why NVL72 racks change the answer.

### 4.3 What the literature says

- **Xu et al. (Hot Interconnects 2025)**, the cleanest direct measurement:
  Llama-3.1-8B, 4× H100/node + IB NDR400. TP2 → TP4 **intra-node**: e2e latency
  310 ms → 210 ms, TPOT 1.17 ms → 0.86 ms. TP4 → TP8 **crossing nodes**: TPOT
  0.86 ms → **11.56 ms**, "decode stage becomes communication-bound"
  `[verified]`. A 13× regression from one extra doubling of TP.
- **Pope et al. (Google, arXiv:2211.05102, MLSys 2023)** — still the definitive
  analytical treatment. Their derivation for a feed-forward layer:
  - **1D weight-stationary** (classic Megatron): `T_comm = 2·B·L·E / bandwidth` —
    *independent of chip count*. "As the number of chips grows larger,
    communication becomes a bottleneck." `[verified]`
  - **2D weight-stationary**: shard each weight matrix on both axes, alternate
    which axis you aggregate on. `T_comm = 8·B·L·E / (√n_chips · bandwidth)` —
    scales as **O(1/√n)**, so "even if the 2D layout is communication-limited at
    a certain chip count and batch size, we can continue to reduce latency by
    adding more chips." Crossover: 2D wins when `√n_chips > d_ff/d_model`, i.e.
    **n_chips > 16** for the usual `d_ff = 4·d_model` `[verified]`.
  - **Weight-gathered** (`T_comm = 4E√(B·L·F)/(√n·bw)`, linear in √BL rather than
    BL): only for large batch — prefill, not decode. Their decision rule: "During
    the generate phase, we select the 2D weight-stationary layout because the
    batch size in tokens is always small." `[verified]`
  - Empirically on PaLM 540B, 64 TPU v4: both 1D and 2D become
    communication-limited as chip count rises to 256, but 2D degrades more
    slowly `[verified]`.

  **Direct implication for us:** at TP8 with `d_ff/d_model = 4`, `√8 = 2.83 < 4`,
  so Pope's condition says 1D (Megatron) is still the right layout at 8 chips —
  2D only pays past 16. Our TP8 Megatron-style sharding is correct by their
  criterion.
- **Shift Parallelism (Snowflake AI Research, arXiv:2509.16495)** attacks the
  same tension from the serving side: run **sequence parallel at low traffic**
  and **tensor parallel at high traffic**, made switchable by keeping the KV
  cache layout *invariant* between the two. On 8× H200 with vLLM +
  ArcticInference: TTFT up to **6.97× better than DP and 1.56× better than TP**;
  TPOT **9.34 ms (Llama-70B FP8)**, **8.68 ms (Qwen-32B FP8)**; median TTFT
  **148 ms vs 3930 ms** for TP-only under bursty load; **50% higher throughput**
  than TP at high traffic `[verified]`. Their stated reason TP is good at low
  batch and bad at high batch is the standard one: communication is amortised
  over the batch, so the comm-to-compute ratio is worst when the batch is small —
  *but the alternative (replication) is worse still at batch 1 because it does
  not shard the weight read.*

### 4.4 The verdict for us

TP8 on one node. Do not attempt TP16. If we ever want *more* single-stream speed
from parallelism, the only remaining axis inside the node is to make the
collective cheaper (§3), not to add ranks. If we get NVL72-class hardware, redo
this table with `A(16)`, `A(32)` measured inside the rack's NVLink domain — the
crossover moves.

---

## 5. Speculative decoding as a latency technique

At batch 1 speculative decoding is nearly free: the GPU is idle anyway, so
verifying k tokens costs the same memory traffic as verifying 1 (for the dense
weights) and the only marginal cost is the draft model's own passes. This is why
it is worth 3.09× to us and why every latency-focused system ships it.

| paper / system | lab | venue+year | hardware | headline result | in production? |
|---|---|---|---|---|---|
| Medusa (arXiv:2401.10774) | Princeton/Together (Cai, Dao) | 2024 | — | Medusa-1 >2.2×; Medusa-2 2.3–3.6× `[reported]` | superseded, still in TRT-LLM |
| EAGLE-3 (arXiv:2503.01840) | Li, Wei, Zhang, Zhang | 2025 (NeurIPS'25 per citations) | H100 (SGLang), RTX3090 (vLLM) | τ **5.84–6.62**; speedup **4.12–5.51× at T=0**, 3.45–4.65× at T=1; **1.38–1.39× throughput at batch 24/48/64** in SGLang `[verified]` | **yes — SGLang `--speculative-algorithm EAGLE3`, vLLM, TRT-LLM** |
| MagicDec (arXiv:2408.11049) | CMU (Beidi Chen) | 2024 | various | speculation still wins at **batch 32–256** for long contexts: up to **2.51× on Llama3.1-8B** with a sparse-KV draft `[reported]` | ideas adopted |
| SPIRe (arXiv:2504.06419) | Google (Pope et al.) | 2025 | — | static sparse attention + pruned init + feedback memory draft: **>100% modelled throughput** over small drafters `[reported]` | research |
| Scaling Laws for Speculative Decoding (arXiv:2505.07858) | Scylla | 2025 | — | log-linear acceptance scaling in pretrain tokens, draft capacity, **and batch size**; 1.5–2.2× acceptance over EAGLE-2, 0.3× over EAGLE-3; **2× decode throughput in industrial deployment** `[reported]` | industrial |
| Nightjar (arXiv:2512.22420) | NUDT | 2025 | — | MAB planner **disables speculation** when compute-bound; +14.76% throughput, −20.18% latency under dynamic rates `[reported]` | research |
| Speculative Verification (arXiv:2509.24328) | Hanyang | 2025 | 13B–72B targets | wins across **all** batch sizes 4–80; avg 1.4× in large-batch (32–80) `[reported]` | research |
| Batch Speculative Decoding Done Right (arXiv:2510.22876) | eBay + PSU | 2025 | — | existing batch spec implementations **violate output equivalence**; alignment overhead up to **40% of compute** `[reported]` | correctness warning |
| DSpark (arXiv:2607.05147) | **DeepSeek** | 2026 | production (V4-Flash/Pro) | **+60–85% per-user generation speed at matched throughput** vs MTP-1; +51% aggregate throughput at an **80 tok/s/user SLA** `[verified]` | **yes — DeepSeek production** |

### 5.1 What EAGLE-3 changed and what it means for our 3-1-4 config

EAGLE-1/2 autoregress on the target model's *top-layer feature vector* and
predict the next feature. EAGLE-3 **abandons feature prediction for direct token
prediction** and fuses features from **multiple layers** (low, mid, high) rather
than only the top, via a "training-time test" procedure that simulates
multi-step drafting during training so the head learns to be robust to its own
errors `[verified]`. The point of the change is that feature-prediction imposed a
ceiling on how much training data helped; direct token prediction removes it.

Measured, temperature 0, mean over MT-bench / HumanEval / GSM8K / Alpaca /
CNN-DM `[verified]`:

| model | mean τ (accepted length) | speedup |
|---|---|---|
| Vicuna 13B | 6.62 | 5.51× |
| Llama-3.1-Instruct 8B | 6.23 | 4.44× |
| Llama-3.3-Instruct 70B | 5.88 | 4.12× |
| DeepSeek-R1-Distill-Llama 8B | 5.84 | 4.16× |

**Our 3-1-4 config caps τ at 4 by construction** (`speculative-num-draft-tokens
= 4`, `eagle-topk = 1` meaning a chain, not a tree). Getting 3.09× out of a
τ ≤ 4 chain implies an accepted length around 3.1 — the head is doing well and
the constraint is the config, not the head. The published τ ≈ 6 numbers come from
longer drafts with tree verification.

**But raising the draft length has a cost specific to MoE** `[inferred]`: every
extra draft token is another token routed through the MoE, and expert coverage
grows with token count (§8.2). Going from 4 to 8 draft tokens roughly doubles
the selections, taking distinct-expert coverage from ~30/256 to ~58/256 —
nearly doubling the expert-weight bytes read per verification pass. **On a dense
model longer drafts are almost free; on a 256-expert MoE they are not.** This is
the correct way to think about the 3-1-4 setting, and it means the optimum draft
length for GLM-5.2 is genuinely lower than for a dense model of the same active
size. Measure the τ-vs-expert-bytes curve before lengthening the draft.

### 5.2 Speculation at concurrency

The old folklore ("speculation only helps at small batch") is now clearly wrong
but the mechanism matters:

- EAGLE-3 keeps **1.38–1.39× throughput at batch 24/48/64** in SGLang on a single
  H100 with Llama-3.1-8B `[verified]` — a much smaller multiplier than the
  4.44× latency win at batch 1, but positive.
- MagicDec's contribution is the *analysis*: as batch and sequence length grow,
  the bottleneck shifts from weight loading to **KV loading**, and once KV
  dominates, a draft model with a **sparse KV cache** is cheap relative to the
  target — so speculation keeps paying. Up to **2.51× at batch 32–256** for
  Llama3.1-8B on moderate-to-long sequences `[reported]`.
- DSpark is the most relevant production data point because it is DeepSeek on a
  large MoE: **+60–85% per-user generation speed at matched throughput** for
  V4-Flash, **+57–78%** for V4-Pro, and at a fixed **80 tok/s/user SLA** it
  delivers **+51% aggregate throughput** `[verified]`. Their baseline is MTP-1.
  Offline, DSpark's accepted length beats EAGLE-3 by **26.7–30.9%** on Qwen3
  4B/8B/14B `[verified]`.
- **Correctness warning:** "Batch Speculative Decoding Done Right" documents that
  common batch speculative implementations produce **outputs that differ from
  autoregressive decoding** because sequences accepting different draft counts
  desynchronise position IDs, attention masks and KV state (the "ragged tensor"
  problem), and that fixing it properly costs **up to 40% of compute** unless you
  batch same-length sequences together `[reported]`. Worth auditing our own
  implementation for output equivalence at C>1.

---

## 6. Non-GPU silicon: why Groq and Cerebras win at batch 1

The honest version of this analysis is short, because the mechanism is simple and
the public numbers are mostly vendor-published.

**The mechanism.** At batch 1 the arithmetic intensity of decode is ~2 FLOP/byte.
Every architecture is therefore bandwidth-limited, and the only question is what
memory you read the weights from. HBM3e on a B200 gives 8 TB/s per package. SRAM
on-die gives one to two orders of magnitude more, at the cost of capacity — which
forces you to spread the model over many chips and pay interconnect instead.

- **Groq LPU**: ~**230 MB SRAM per chip**; "hundreds of chips are needed to
  support a 70B parameter model" `[reported — Cerebras blog]`. Deterministic,
  statically scheduled pipeline: no dynamic scheduling means no launch overhead,
  no cache misses, no arrival skew — the *entire* class of problems this document
  is about simply does not exist.
- **Cerebras WSE-3 / CS-3**: a wafer-scale part holding the model in on-wafer
  SRAM so weights never leave the die `[reported]`.

**Published comparative speeds** (vendor page citing Artificial Analysis
measurements) `[reported — treat with appropriate scepticism, this is a
competitor comparison published by one of the competitors]`:

| model | Cerebras tok/s | Groq tok/s |
|---|---|---|
| gpt-oss-120B | ~3,000 | ~493 |
| Llama 4 Maverick | >2,500 | ~497 |
| Llama 3.3 70B | >2,500 | ~403 |

**What a GPU would have to do to match.** The useful framing is not "GPUs need
more bandwidth." It is:

1. **The GPU's bandwidth roofline is already 4–8× above what engines achieve**
   (§0.1). Closing the gap to the roofline is worth more than any plausible
   bandwidth increase, and does not require new silicon.
2. **The specific things Groq gets for free are exactly the megakernel agenda**:
   static schedule computed at compile time, one "launch" for the whole model,
   no dynamic shapes, no host in the loop, deterministic arrival at every
   synchronisation point. TileRT's design — "statically schedules the entire
   computation graph into a persistent GPU Engine Kernel at compile time, one
   kernel launch for the whole inference, intermediates passed through registers,
   shared memory and L2 without writing back to global memory" `[reported]` — is
   literally a software reimplementation of the Groq execution model on a GPU.
   That is why it is at ~500 tok/s and we are at 365.
3. **What a GPU cannot replicate** is the SRAM residency: a B200's on-die caches
   cannot hold a frontier model's active weights, so weights are re-read from
   HBM every token, forever. The ceiling is `64 TB/s / bytes_per_token` and
   nothing in software moves it. Only quantisation moves it — which is why NVFP4
   matters more for latency than for cost (NVFP4 roughly halves `bytes_per_token`
   versus FP8, hence roughly doubles the roofline).

**The concrete GPU data point to hold against them:** NVIDIA reports **250
tokens/s/user on DeepSeek-R1 671B FP4 on a single DGX B200 (8 GPUs) at
concurrency 1**, 1024 in / 2048 out, with TensorRT-LLM v0.17 and speculative
decoding `[reported]`. TileRT reports **~500 tok/s on GLM-5 FP8** and
**494.2 tok/s/user at 1k/1k** on the same class of machine `[reported]`, and
**>1000 tok/s on the 1T-parameter MiMo-V2.5-Pro-UltraSpeed** `[reported]`. So a
well-engineered 8× B200 is now within ~2–5× of Groq on comparable models — the
gap is software, and it is closing.

---

## 7. Techniques that only make sense at batch 1

These are individually small and collectively decisive. Grouped by what they
attack.

### 7.1 Keeping weights close: L2 residency and prefetch

At batch 1 you re-read the same weight bytes every token, ~365 times a second.
Anything that survives in L2 between tokens is free bandwidth. CUDA exposes this
directly `[verified — CUDA C++ Best Practices Guide]`:

```cuda
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize);
// then per-stream:
cudaAccessPolicyWindow w;
w.base_ptr  = weights_ptr;
w.num_bytes = N;              // must be <= the set-aside size or you thrash
w.hitRatio  = r;              // fraction of accesses to treat as persisting
w.hitProp   = cudaAccessPropertyPersisting;
w.missProp  = cudaAccessPropertyStreaming;
```

The guide's warning is the operationally important part: **if the persisting
region exceeds the set-aside L2 portion you get thrashing**, and the fix is to
shrink `num_bytes` and scale `hitRatio` proportionally — their worked example
recovers a 10% regression by keeping only 20 MB resident and letting the rest
stream `[verified]`.

Evidence this is worth real money: **Fleet** on AMD MI350 raises **L2 hit rate
from 12% to 54% at batch 32 and 39% to 61% at batch 64** through cooperative
weight tiling within a chiplet, **reducing HBM traffic by up to 37%** and giving
**1.27–1.30× over a chiplet-unaware megakernel** `[reported]`. Their batch-1–8
result is **1.3–1.5× lower decode latency than vLLM** `[reported]`. B200 is also
a two-die part; the chiplet-locality argument transfers.

Candidates for pinning on our system, in priority order `[inferred]`:
the DSA indexer's projection weights (small, hit every layer every token, 5.8% of
our step), RMSNorm weights, router weights (256×d per layer — small and touched
every token), and the KV-cache pages for the most recent tokens.

### 7.2 Keeping the hot path resident and shape-static

- **One shape, forever.** At batch 1 with a fixed draft length, the decode step
  has exactly one shape. Every dynamic dimension you allow forces either a graph
  recapture (ETC: 51–67 captures, 123–583 s warmup in vLLM/SGLang `[verified]`)
  or a runtime branch inside the kernel. Ada-MK's core insight is precisely this:
  "**under a fixed deployment configuration, the optimal execution path of a
  MegaKernel is uniquely determined, and runtime dynamic decision-making can be
  entirely hoisted to compile time**", and they measure the payoff as **+23.6%
  over TensorRT-LLM, +50.2% over vLLM** at single-batch on an L20 `[reported]`.
- **Pad rather than branch.** Variable expert counts per rank cause the arrival
  skew in §3.3. Padding every rank's expert GEMM to a fixed shape trades a little
  wasted bandwidth for the removal of a synchronisation stall — at batch 1 the
  stall is worth more.
- **No host round-trip.** Sampling, acceptance checking for speculative decoding,
  and stop-token detection must all stay on device. A single `.item()` in the
  decode loop costs a full round-trip (µs to tens of µs) *and* breaks CUDA-graph
  capture.

### 7.3 Spinning rather than blocking, and CPU isolation

- `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` makes the host poll rather than
  sleep while waiting on the device; `cudaDeviceScheduleBlockingSync` sleeps.
  Spinning burns a core to save the wake-up latency; at 365 tok/s you do this
  365 times a second and a 50 µs wake-up is 1.8% of your throughput. `[CUDA
  Runtime API — flags verified to exist; standard semantics]`
- **Pin the ranks.** Our box has **2 NUMA nodes**; a rank whose host thread
  migrates across the NUMA boundary pays for every pinned-memory access and every
  doorbell write. Pin each rank's launch thread to a core on the NUMA node local
  to its GPU's PCIe root complex, and isolate those cores from the general
  scheduler (`isolcpus` / `cgroup` cpuset).
- **Host single-thread speed is measurably load-bearing.** TaxBreak's
  cross-platform result: the same class of GPU with a faster host CPU (Emerald
  Rapids 2.2/4.0 GHz vs Sapphire Rapids 2.0/3.8 GHz) gave **10–29% lower
  orchestration overhead and 14% lower end-to-end latency** for Llama-3.2-1B
  decode, and **13–14% end-to-end improvement for Qwen1.5-MoE-A2.7B despite an
  8% slower GPU** `[verified]`.
- **CPU-GPU coupling matters at low batch.** GH200 (NVLink-C2C) "remains
  CPU-bound up to **4× larger batch sizes**" than PCIe-attached systems
  `[reported]`, which is a statement that low-batch decode is a host-latency
  workload.

### 7.4 Programmatic Dependent Launch — and why it is not enough

PDL (`cudaLaunchAttributeProgrammaticStreamSerialization`,
`cudaTriggerProgrammaticLaunchCompletion`, `cudaGridDependencySynchronize`) lets a
dependent kernel launch and run a *preamble* — typically loading its own weights —
before the producer kernel has finished, then block at
`cudaGridDependencySynchronize()` until the producer's data is ready. It is the
sanctioned CUDA answer to inter-kernel bubbles. `[CUDA Programming Guide §4.5 —
the section exists and is titled "Programmatic Dependent Launch and
Synchronization"; I was unable to fetch the section body, so treat the API
semantics here as standard knowledge rather than verified quotation.]`

Hazy Research's argument against relying on it is the one that decided the
megakernel design, and I did verify that: **PDL's granularity is
whole-kernel-to-whole-kernel**, so attention must wait for *all* of Q, K and V to
be complete rather than starting on head 0 as soon as head 0's QKV is done. Their
counter-based scheme splits the MLP hidden state into **four independently
counted chunks** so down-projection starts on chunk 0 while chunk 3 is in flight
`[verified]`. PDL cannot express that.

**Practical read:** PDL is worth turning on wherever the kernel structure survives
(it is essentially free), but it is a 5–15% technique, not a 2× technique. The
2× is the megakernel.

---

## 8. Serving many streams while each stays fast

### 8.1 The published frontier

| system | hardware | model | interactivity point | throughput point | source |
|---|---|---|---|---|---|
| TensorRT-LLM v0.17 | 1× DGX B200 (8 GPU) | DeepSeek-R1 671B FP4 | **250 tok/s/user @ C1** (1k in / 2k out) | 30,000+ tok/s aggregate at max concurrency | NVIDIA `[reported]` |
| TileRT | 8× B200 | GLM-5 FP8 + MTP | **494.2 tok/s/user @ 1k/1k**; 340 @ 8k/1k | n/a — **one in-flight request per decode node** | SemiAnalysis InferenceX `[verified]` |
| conventional engines (best FP8) | 8× B200 class | GLM-5 | 136.3 tok/s/user @ 1k/1k | — | SemiAnalysis `[reported]` |
| conventional engines (best FP4) | 8× B200 class | GLM-5 | 256.3 tok/s/user @ 1k/1k | — | SemiAnalysis `[reported]` |
| GB300 NVL72 NVFP4+MTP | rack | GLM-5-class | 181.4 tok/s/user @ 8k/1k | — | SemiAnalysis `[reported]` |
| SGLang large-scale EP | 96× H100 (12 nodes) | DeepSeek-V3 | (throughput-optimised) | **22,282 output tok/s per node** decode; 52.3k input tok/s | LMSYS `[verified]` |
| DeepSeek production | H800, ~226–278 nodes | V3/R1 | **20–22 tok/s/user** | 14.8k output tok/s per node (~1,850/GPU) | DeepSeek `[verified]` |
| DeepSeek DSpark | production | V4-Flash / V4-Pro | SLA points at **80** and **35 tok/s/user** | +51% / +52% aggregate at those SLAs | arXiv:2607.05147 `[verified]` |
| Pope et al. | 64× TPU v4 | PaLM 540B int8 | 28.5 ms/token @ batch 64 (≈35 tok/s/user), 14% MFU | 6.0 s / 64 tokens @ batch 512, 33% MFU | MLSys 2023 `[verified]` |

Two structural facts fall out:

**(a) The spread between the interactivity end and the throughput end is roughly
10–25×.** DeepSeek runs production at 20–22 tok/s/user and gets ~1,850 output
tok/s/GPU. TileRT gets ~494 tok/s/user and serves one request per node. NVIDIA's
own B200 numbers bracket it: 250 tok/s/user at C1, 30,000+ tok/s at max
concurrency (i.e. ~3,750 tok/s/GPU).

**(b) The price of speed is superlinear.** SemiAnalysis prices DeepSeek-R1 FP4 at
**~$0.56/M output tokens at 50 tok/s/user and ~$4/M at 125 tok/s/user** — 2.5×
the speed for ~7× the cost `[reported]`. They use this to explain "fast mode"
API pricing: ~2.5× speed for 6–12× the price `[reported]`.

**Pope et al.'s cleanest single sentence on this:** "The minimum latency for
generation is **3 times lower** than the batch-512 latency." And, importantly for
us: "batch size 1 achieves best latency in the prefill phase, but **for the
generate phase we can increase the batch size up to 64 with negligible latency
impact**, and doing so is dramatically better for generate MFU" `[verified]`.
That is a *dense* 540B model on 64 chips — the batch-1-to-64 flatness is a
property of dense decode, and it does **not** transfer to a 256-expert MoE, for
the reason in §8.2.

### 8.2 Why our per-stream speed falls 4.7× from C1 to C16 — and why that is normal

Two effects compound, and only one of them is in the folklore.

**Effect 1: attention and the DSA indexer scale linearly with concurrency.**
KV bytes read per step are proportional to the number of active sequences,
period. At C1 attention is 10.9% and the indexer 5.8% of our step; at C16 those
terms are 16× larger. Nothing in the parallelism strategy changes this — TP8
shards KV heads across ranks, DP-attention shards sequences across ranks, and
either way the *total system* KV bytes are the same.

**Effect 2: MoE distinct-expert coverage grows sub-linearly but fast, saturating
at 256.** At batch 1 you touch 8 experts per layer; at batch 64 you touch nearly
all 256. Modelling top-k selections as independent and uniform over N=256 experts,
the expected number of distinct experts touched by S selections is
`N·(1 − (1−1/N)^S)`. With EAGLE 3-1-4 there are ~4 tokens per verification pass
per stream, so S = 4·C·8 `[inferred]`:

| concurrency | tokens/step | selections | distinct experts | % of 256 |
|---|---|---|---|---|
| C1 (no spec) | 1 | 8 | 7.9 | 3% |
| **C1 (spec ×4)** | 4 | 32 | **30.1** | 12% |
| C4 | 16 | 128 | 100.9 | 39% |
| C8 | 32 | 256 | 162.0 | 63% |
| **C16** | 64 | 512 | **221.5** | 87% |
| C64 | 256 | 2048 | 255.9 | 100% |

Expert-weight read amplification C1 → C16 = **221.5 / 30.1 = 7.35×**. (Real
routing is skewed toward hot experts, which raises coverage at small S and
saturates sooner, so the true ratio is somewhat lower — call it 5–7×.)

**Putting the two together against our own C1 hotspot profile** `[inferred]`:

| component | % of C1 step | growth C1→C16 | contribution |
|---|---|---|---|
| dense GEMM | 37.1% | ×1.1 (GEMV → thin GEMM, still memory-bound) | 0.408 |
| collectives | 19.6% | ×1.2 (message grows 16× but stays latency-bound) | 0.235 |
| MoE expert GEMMs | 19.4% | ×7.35 (expert coverage) | 1.426 |
| attention | 10.9% | ×16 (linear in sequences) | 1.744 |
| DSA indexer | 5.8% | ×16 (linear in sequences) | 0.928 |
| **total step-time growth** | | | **4.74×** |

**Predicted per-stream speed at C16: 365 / 4.74 = 77 tok/s. Measured: 78–95
tok/s.** The model reproduces the measurement.

Three conclusions follow, and they are the most actionable things in this
document:

1. **The 4.7× decay is not a scheduler defect and there is no quick fix for it.**
   It is 56% attention+indexer (linear, irreducible without shrinking per-token
   KV bytes) and 30% MoE expert coverage (a property of 256/8 routing).
2. **The number to attack is the C1 starting point, not the decay ratio.** If C1
   went from 365 to 700 tok/s by closing the roofline gap, C16 would go to ~148
   tok/s by the same arithmetic, because the fixed overheads that dominate C1
   barely move the C16 term.
3. **Aggregate throughput gain C1→C16 is only 16/4.74 = 3.4× predicted**
   (measured 3.7–4.7×). That is the honest cost-scaling curve, and it is much
   worse than a dense model would give — which is the real price of a
   256-expert MoE at moderate concurrency. It gets *better* past C64 once expert
   coverage saturates: beyond full coverage, extra concurrency is nearly free in
   expert bytes, so throughput scales much better from C64 upward. **The
   economically correct operating point for this model is well above C64, and the
   latency-optimal point is C1. There is very little good territory in between**
   — which is an argument for two separate deployments, not one blended one.

*(Data-quality flag: our recorded "40.8k tok/s aggregate at C64" implies 637
tok/s/user, above the C1 figure, so it cannot be output-token throughput at C64
on the same workload. Re-measure with the same definition as the C1 and C16
numbers before comparing to anything in the table above.)*

### 8.3 Techniques for getting both

- **Prefill/decode disaggregation** so a prefill never delays a decode step.
  DeepSeek runs prefill at EP32 and decode at EP144 as separate pools
  `[verified]`. TileRT ships PD disaggregation with a vLLM integration
  `[reported]`.
- **Attention-FFN disaggregation (AFD)** is the next level: separate the
  attention pool from the FFN/MoE pool so each is sized to its own bottleneck.
  Wu et al. report that under strict TTFT/TPOT SLOs, **AFD sustains ~4k tokens/s
  of system throughput on DeepSeek-V3.2 across chat/coding/agentic workloads,
  where non-AFD configurations were infeasible** `[reported]`.
- **Shift Parallelism** — switch parallelism strategy with load, keeping the KV
  layout invariant so no data moves (§4.3) `[verified]`.
- **Load-aware speculation.** Nightjar disables speculation via a multi-armed
  bandit when the system goes compute-bound `[reported]`; DSpark schedules
  verification length from a system throughput profile and confidence estimates,
  which is why it holds **+60–85% per-user speed at matched throughput** in
  DeepSeek production `[verified]`. At C1 you want the longest draft the MoE
  budget allows; at C64 you want MTP-1 or nothing. This should be a runtime
  policy, not a launch flag.
- **Sparse attention to cut the linear term.** Since attention+indexer are 56% of
  our concurrency penalty, the DSA follow-up literature is directly relevant:
  IndexCache removes **75% of indexer computations for 1.82× prefill / 1.48×
  decode speedup** via cross-layer index reuse `[reported]`; MISA treats indexer
  heads as an MoE and reports **3.82× on H200** `[reported]`; HISA adds a
  block-level coarse filter before token-level refinement, validated on
  DeepSeek-V3.2 and GLM-5 `[reported]`; "Guess-Verify-Refine" exploits temporal
  correlation across consecutive decode steps for **1.88× single-operator, up to
  7.52% end-to-end on Blackwell** `[reported]`; LiteTopK reports **1.35× GLM-5.2
  prefill** `[reported]`. These target exactly our 5.8% indexer slice and the KV
  reads behind our 10.9% attention slice.

---

## What is NOT worth it

Techniques that look good in the literature and either do not transfer to 8×
B200 + a large MoE at batch 1, or actively hurt.

1. **TP16 / any cross-node tensor parallelism for latency.** ~5 µs added per
   collective × ~184 collectives = ~0.9 ms/token, against ~0.3 ms of memory-term
   saving `[inferred]`. Directly measured elsewhere as TPOT 0.86 ms → 11.56 ms
   `[verified — Xu et al.]`. Do not do this.

2. **Pipeline parallelism for single-stream latency.** At batch 1 there is
   nothing to fill the pipeline with; PP converts a latency problem into a
   bubble problem. Xu et al.: "pipeline parallelism minimises data transfer
   requirements while **increasing total latency**" `[verified]`.

3. **2D weight-stationary sharding at 8 chips.** Pope et al.'s own crossover
   condition (`√n_chips > d_ff/d_model`) says 2D only beats 1D past ~16 chips for
   the usual `d_ff = 4·d_model` `[verified]`. At TP8, `√8 = 2.83 < 4`. Standard
   Megatron 1D is correct for us.

4. **Weight-gathered layouts during decode.** They are linear in `√(B·L)` rather
   than `B·L`, which is a win only once the batch in tokens is huge. Pope et
   al.'s explicit decision rule: "During the generate phase, we select the 2D
   weight-stationary layout because the batch size in tokens is always small"
   `[verified]`.

5. **Chasing the HBM roofline with faster memory or more GPUs.** The evidence is
   unambiguous that on fast GPUs the roofline is not the binding constraint: 27%
   of peak achieved on H100 vs 81% on L4, with the difference attributed to
   launch-side overhead `[verified]`. FastUSP states it flatly for
   high-bandwidth interconnects: "**kernel launch overhead — rather than
   communication latency — is the primary bottleneck**" `[reported]`.

6. **Megakernels as the *first* optimisation on a TP8 MoE.** ETC's own numbers:
   1.48× at bs=1 on a single GPU, but **0.99–1.06× at TP4, where SGLang's better
   CPU scheduling won** `[verified]`. And AutoMegaKernel trails cuBLAS on
   A100/H100 (training-class, high-bandwidth) while winning on L4/L40S
   `[reported]`. Fix the host path and the collectives first; then the megakernel
   is worth 1.2–1.5×.

7. **Multi-node megakernels.** Perseus documents **up to 10× regression on 8
   nodes** for communication-bound MoE megakernels, because proxy-based RDMA
   fences drain the NIC pipeline once per tile transfer `[reported]`. If the
   megakernel ever tempts us across the node boundary, this is the paper to read
   first.

8. **Naively lengthening the speculative draft on a 256-expert MoE.** On a dense
   model the marginal verification token is nearly free. On GLM-5.2, doubling
   draft tokens roughly doubles top-k selections and pushes expert coverage from
   ~12% to ~22% of 256 `[inferred]` — the extra tokens cost real bandwidth. The
   published τ ≈ 6 EAGLE-3 numbers are on dense models (Vicuna, Llama)
   `[verified]`; do not assume they transfer.

9. **Batch speculative decoding without an equivalence audit.** Common
   implementations desynchronise position IDs / masks / KV state across
   sequences accepting different draft counts, producing outputs that differ
   from autoregressive decoding; correct synchronisation costs **up to 40% of
   compute** unless you group same-length sequences `[reported]`. This is a
   correctness bug that shows up as a quality regression nobody attributes to the
   serving stack.

10. **Persisting large weight regions in L2 without measuring.** The CUDA best
    practices guide's own worked example is a **10% regression from thrashing**
    when the persisting region exceeds the set-aside size `[verified]`. Pin small
    hot things (router, norms, indexer projections), not expert weights.

11. **Optimising the C1→C16 decay ratio as a goal in itself.** It is 4.74×
    because of attention linearity and MoE expert coverage `[inferred, model
    reproduces measurement]`. The decay is a property of the model architecture.
    Optimise C1 absolute speed and C64+ absolute throughput; the middle is
    structurally bad territory for a 256/8 MoE.

12. **Blocking synchronisation and unpinned host threads on a latency SKU.** Free
    to fix, and TaxBreak's cross-CPU result (**14% end-to-end from a faster host
    alone** `[verified]`) shows the host is on the critical path at low batch.

---

## Sources

Every URL below was fetched and read during the preparation of this document.

**Batch-1 analysis and roofline**
- Josef Chen, *Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode*, arXiv:2605.30571, 28 May 2026. https://arxiv.org/abs/2605.30571 · https://arxiv.org/html/2605.30571v1
- Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean (Google), *Efficiently Scaling Transformer Inference*, arXiv:2211.05102, MLSys 2023. https://arxiv.org/abs/2211.05102 · https://arxiv.org/pdf/2211.05102
- NVIDIA, DGX B200 product specifications. https://www.nvidia.com/en-us/data-center/dgx-b200/

**Megakernels and persistent runtimes**
- Benjamin Spector, Jordan Juravsky, Stuart Sul, Owen Dugan, Dylan Lim, Dan Fu, Simran Arora, Chris Ré (Hazy Research, Stanford), *Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B*, 27 May 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Xinhao Cheng et al. (CMU/UW/Berkeley/NVIDIA/Tsinghua), *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*, arXiv:2512.22219v2, Dec 2025. https://arxiv.org/abs/2512.22219 · https://arxiv.org/html/2512.22219v2
- Zhihao Jia, *Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference*. https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- Hongyi Jin et al., *Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel*, arXiv:2604.13327v2, Apr 2026. https://arxiv.org/html/2604.13327v2
- Aniruddha Nrusimha, William Brandon, Mayank Mishra, Yikang Shen, Rameswar Panda, Jonathan Ragan-Kelley, Yoon Kim, *FlashFormer: Whole-Model Kernels for Efficient Low-Batch Inference*, arXiv:2505.22758. https://arxiv.org/abs/2505.22758 · https://arxiv.org/html/2505.22758v2
- Wenxin Dong et al., *Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference*, arXiv:2605.11581v1, May 2026.
- Sangeeta Chowdhary et al. (AMD), *Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs*, arXiv:2604.15379v1, Apr 2026.
- Byungsoo Oh, Rachee Singh, *Eliminating Hidden Serialization in Multi-Node Megakernel Communication* (Perseus), arXiv:2605.00686v1, May 2026.
- Vyom Sharma, Debajyoti Datta, *RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts*, arXiv:2604.26039v1, Apr 2026.
- Jaber Jaber, Osama Jaber, *AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis*, arXiv:2606.09682v1, Jun 2026.
- Xinhao Luo et al., *ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive*, arXiv:2508.18850, Aug 2025. https://arxiv.org/abs/2508.18850
- Yiwei Yang et al., *GPUOS: A GPU Operating System Primitive for Transparent Operation Fusion*, arXiv:2604.17861v1, Apr 2026.
- TileRT, *Tile-Based Runtime for Ultra-Low-Latency LLM Inference*. https://github.com/tile-ai/TileRT · https://github.com/tile-ai/TileRT/releases
- SemiAnalysis InferenceX, *Ultra-High Interactivity on NVIDIA GPUs? TileRT on InferenceX*. https://inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia

**Host-side overhead**
- Prabhu Vellaisamy, Shreesh Tripathi, Vignesh Natarajan, Surya Santhan Thenarasu, Shawn Blanton, John P. Shen (CMU), *TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition*, arXiv:2603.12465, Mar 2026. https://arxiv.org/pdf/2603.12465
- Prabhu Vellaisamy et al., *Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures*, ISPASS 2025, arXiv:2504.11750. https://arxiv.org/abs/2504.11750
- LMSYS, *SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs*, 4 Dec 2024. https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- vLLM, *vLLM V1: A Major Upgrade to vLLM's Core Architecture*, 27 Jan 2025. https://vllm.ai/blog/2025-01-27-v1-alpha-release
- vLLM, *SemiAnalysis InferenceMAX: vLLM and NVIDIA Accelerate Blackwell Inference*, 9 Oct 2025. https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
- Kan Zhu et al. (UW SyFI), *NanoFlow: Towards Optimal Large Language Model Serving Throughput*, OSDI 2025, arXiv:2408.12757. https://arxiv.org/abs/2408.12757
- Divakar Kumar Yadav, Tian Zhao, *Hybrid JIT-CUDA Graph Optimization for Low-Latency Large Language Model Inference*, arXiv:2604.23467v1, Apr 2026.
- Guandong Li, *FastUSP: A Multi-Level Collaborative Acceleration Framework for Distributed Diffusion Model Inference*, arXiv:2602.10940v1, Feb 2026.
- NVIDIA, *CUDA C++ Best Practices Guide* (L2 persistence, `cudaLimitPersistingL2CacheSize`, `cudaAccessPolicyWindow`). https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA, *CUDA Runtime API — Device Management* (`cudaSetDeviceFlags`). https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
- NVIDIA, *CUDA C++ Programming Guide* §4.5 Programmatic Dependent Launch and Synchronization. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- SGLang server arguments. https://docs.sglang.io/advanced_features/server_arguments.html
- vLLM parallelism and scaling docs. https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html

**Collectives**
- Siyuan Shen, Anton Korzh, John Bachan, Tiancheng Chen, Arnav Goel, Ludwig Schneider, Pouya Kousha, Zhenhao He, Sylvain Jeaugey, Kamil Iskra, Nishank Chandawala, Jeff R. Hammond, Torsten Hoefler (ETH Zürich + NVIDIA), *Every µs Matters: Achieving Near Speed-of-Light Latency in GPU Collectives*, arXiv:2607.16100v1, 17 Jul 2026. https://arxiv.org/html/2607.16100v1
- Raja Gond, Nipun Kwatra, Ramachandran Ramjee (Microsoft Research India), *TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference*, MLSys 2026, arXiv:2505.11329. https://arxiv.org/abs/2505.11329
- Lang Xu, Kaushik Kandadi Suresh, Quentin Anthony, Nawras Alnaasan, Dhabaleswar K. Panda (Ohio State), *Characterizing Communication Patterns in Distributed Large Language Model Inference*, Hot Interconnects 2025, arXiv:2507.14392. https://arxiv.org/abs/2507.14392 · https://arxiv.org/html/2507.14392v1

**Parallelism strategy**
- Mert Hidayetoglu, Aurick Qiao, Michael Wyatt, Jeff Rasley, Yuxiong He, Samyam Rajbhandari (Snowflake AI Research), *Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads*, arXiv:2509.16495v2. https://arxiv.org/html/2509.16495
- Hanjiang Wu et al. (Georgia Tech / Google / Intel), *How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving*, arXiv:2605.28302, May 2026. https://arxiv.org/abs/2605.28302
- Long Zhao et al., *Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism* (PAT), arXiv:2605.23945, May 2026. https://arxiv.org/abs/2605.23945

**Speculative decoding**
- Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang, *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*, arXiv:2503.01840. https://arxiv.org/abs/2503.01840 · https://arxiv.org/html/2503.01840v3
- Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao, *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, arXiv:2401.10774. https://arxiv.org/abs/2401.10774
- Ranajoy Sadhukhan et al. (CMU), *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding*, arXiv:2408.11049. https://arxiv.org/abs/2408.11049
- Xin Cheng et al. (DeepSeek), *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*, arXiv:2607.05147, Jul 2026. https://arxiv.org/abs/2607.05147 · https://arxiv.org/html/2607.05147v1
- Sanjit Neelam, Daniel Heinlein, Vaclav Cvicek, Akshay Mishra, Reiner Pope, *SPIRe: Boosting LLM Inference Throughput with Speculative Decoding*, arXiv:2504.06419, Apr 2025.
- Siyuan Yan et al., *Scaling Laws for Speculative Decoding* (Scylla), arXiv:2505.07858, May 2025.
- Rui Li et al., *Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving*, arXiv:2512.22420v5, Dec 2025.
- Sungkyun Kim et al., *Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding*, arXiv:2509.24328v2, Sep 2025.
- Ranran Haoran Zhang et al. (eBay + PSU), *Batch Speculative Decoding Done Right*, arXiv:2510.22876v3, Oct 2025.

**Attention, MoE and sparse attention**
- Zihao Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*, MLSys 2025, arXiv:2501.01005. https://arxiv.org/abs/2501.01005
- Lianmin Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention), arXiv:2312.07104. https://arxiv.org/abs/2312.07104
- Zewen Jin et al., *DeaMoE: Efficient MoE Structure for Fast Small-Batch Decoding*, arXiv:2608.14385, Aug 2026. https://arxiv.org/abs/2608.14385 · https://arxiv.org/html/2608.14385v1
- Krishna Teja Chitty-Venkata et al. (Argonne + Cerebras), *MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models*, arXiv:2508.17467. https://arxiv.org/html/2508.17467v1
- DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv:2512.02556, Dec 2025.
- Yushi Bai et al. (Tsinghua/Zhipu), *IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse*, arXiv:2603.12201, Mar 2026.
- Ruijie Zhou et al., *MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference*, arXiv:2605.07363, May 2026.
- Yufei Xu et al., *HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention*, arXiv:2603.28458v3, Mar 2026.
- Long Cheng et al. (NVIDIA/Microsoft), *Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation*, arXiv:2604.22312, Apr 2026.
- Ziqi Yin et al., *LiteTopK: Exploiting the Curse of Dimensionality for a Fused Indexer-TopK Kernel in Long-Context Sparse Attention*, arXiv:2607.11976v3, Jul 2026.

**Production systems and economics**
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437. https://arxiv.org/html/2412.19437v2
- DeepSeek, *DeepSeek-V3/R1 Inference System Overview* (Open Source Week day 6). https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- LMSYS, *Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism*, 5 May 2025. https://lmsys.org/blog/2025-05-05-large-scale-ep/
- NVIDIA, *NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance*. https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/
- SemiAnalysis, *InferenceMAX: Open Source Inference Benchmarking*. https://inferencex.semianalysis.com/blog/inferencemax-open-source-inference-benchmarking
- SemiAnalysis, *InferenceX v2: NVIDIA Blackwell vs AMD vs Hopper*. https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs
- Cerebras, *Cerebras CS-3 vs. Groq LPU*. https://www.cerebras.ai/blog/cerebras-cs-3-vs-groq-lpu
