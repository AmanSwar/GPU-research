# Megakernels, persistent kernels and the war on launch overhead

**What this is.** A survey of every technique for removing per-kernel launch and
dependency cost from an LLM decode step — Stanford HazyResearch's megakernels,
CMU's Mirage Persistent Kernel, Blackwell Cluster Launch Control, Programmatic
Dependent Launch, CUDA Graphs, and tile-level fusion (TileRT) — read from primary
sources, plus **a first-hand gap analysis of our own nsys trace**. The trace
analysis produces a negative result that redirects the whole topic: on our box,
inter-kernel gaps are already gone. What is left is a different bubble, and it is
bigger. Everything is labelled `[verified] [reported] [inferred] [unverified]`.

---

## Bottom line for our system

- **Launch gaps are already dead here.** In the steady-state C1 decode window of
  `runs/sweep-latency-3-1-4/trace.sqlite`, device 0 spends **1.90 % of wall clock
  with no kernel resident** (68.7 ms of 3621.5 ms), across 804,823 kernels. Gap
  p50 = **0.22 µs**, p90 = 0.54 µs. `[verified — measured, §2]`
- **Because 100 % of decode kernels are already CUDA-graph launched.** 1,132,105
  of 1,183,520 device-0 kernels in the trace carry a non-null `graphId`; in the
  decode window it is 100 %. The eager remainder is prefill (graph disabled for
  MLA). `[verified — measured, §2.2]`
- **The real bubble is grid width, not gaps.** Time-sliced across the decode
  window, only **65.4 % of the 148 SMs hold a resident CTA on average**. A quarter
  of decode wall clock (26.4 %) runs on **≤ 36 of 148 SMs**. `[verified, §2.4]`
- **10.7 % of decode GPU time runs on a single SM.** 152,052 launches with
  `gridDim == 1` (rmsnorm, `act_and_mul`, `routingIndicesBlockKernel`,
  `fused_k_indexer_norm_rope_store`) burn 530.9 ms with 147 SMs idle. `[verified]`
- **Perfect packing ceiling = 1.53×.** If all the SM-work in the window were
  packed 148 wide, wall drops 3621.5 → 2368.7 ms. That maps 365 → **558 tok/s**,
  which brackets TileRT's published 494.2 tok/s on GLM-5-FP8. This is the size of
  the megakernel prize on our hardware — and it is *not* a launch-overhead
  prize. `[inferred from verified measurement, §2.5]`
- **Cheapest intervention is not a megakernel.** In descending value-per-hour:
  (1) turn on the fusion flags that are off (`enable_fused_moe_sum_all_reduce`,
  `enable_flashinfer_allreduce_fusion`, `enable_fused_qk_norm_rope`), (2) set
  `TRTLLM_ENABLE_PDL=1` — PDL is compiled into our tree and gated OFF by default,
  (3) widen or fuse the `gridDim == 1` kernels, (4) overlap the 32-CTA
  `oneshotAllreduceFusionKernel` (12.0 % of decode time, 116 SMs idle
  underneath). `[verified config, §10]`
- **A full megakernel is blocked by registers before anything else.** Our B200
  reports **65,536 registers per SM and per block**; `nvjet_sm100_*` runs at
  **255 registers/thread × 256 threads = 65,280**. A megakernel must size
  registers at the max over all instructions, so it inherits 1 block/SM and
  cannot co-schedule two instruction types on an SM by occupancy. `[verified]`

---

## 1. The problem as it is usually stated

The canonical framing, from the HazyResearch post that started the current wave:

> "existing systems break down a model forward pass into around a hundred
> separate kernels" — each pays launch/teardown, and each forces a full barrier
> at its end because the *slowest* block must finish before the next kernel's
> first block starts.
> — [Look Ma, No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) `[verified]`

Their measured launch costs, on H100:

| path | cost per launch | source |
|---|---:|---|
| CUDA stream, dummy kernel, H100 | **2.1 µs** | HazyResearch `[reported]` |
| CUDA graph replay, dummy kernel, H100 | **1.3 µs** | HazyResearch `[reported]` |

MPK measured the same quantity on their stack (Qwen3-8B, 293 kernel launches per
token):

| path | cost per launch | per token | source |
|---|---:|---:|---|
| eager | **3.8 µs** | 1.1 ms | MPK arXiv 2512.22219 `[reported]` |
| CUDA Graphs | **0.8 µs** | 0.2 ms | MPK arXiv 2512.22219 `[reported]` |

Ada-MK states the industrial version: *"kernel launch overhead alone can account
for 14.6 % of end-to-end inference time"* on an NVIDIA L20 in an online-ads
serving stack `[reported —` [arXiv 2605.11581](https://arxiv.org/abs/2605.11581)`]`.
Note the hardware: L20 is a small inference-class part, and 14.6 % there says
nothing about a 148-SM B200 running graphs.

**The arithmetic on our system.** GLM-5.2, 79 layers, TP8, EAGLE 3-1-4. Measured
in the decode window: **2135 kernels per target-forward group** (804,823 kernels
/ 377 `scheduler.run_batch` ranges) — roughly 27 kernels per layer per forward.
At an eager 3.8 µs/launch that would be **8.1 ms of pure launch cost per forward
group**, against a 2.74 ms TPOT (365 tok/s) and ~3.16 accepted tokens per group.
Eager execution is arithmetically impossible for this model. This is why nobody
runs decode eager, and why the interesting question is what remains *after*
graphs.

---

## 2. Measured on our box: where the decode step actually goes

Everything in this section is first-hand, computed against
`/home/aman/code/benchmark/runs/sweep-latency-3-1-4/trace.sqlite` (3.28 GB nsys
export, GLM-5.2-NVFP4, TP8, EAGLE 3-1-4, C1, SM clock locked 1597 MHz, nsys
2025.6.3). All figures are **device 0**; the other seven ranks have identical
kernel counts (1,183,520 each), confirming lockstep TP8.

### 2.1 Method

The nsys SQLite export gives `CUPTI_ACTIVITY_KIND_KERNEL(start, end, deviceId,
streamId, graphId, gridX/Y/Z, blockX/Y/Z, registersPerThread, staticSharedMemory,
dynamicSharedMemory, clusterX/Y/Z)`. Gap analysis is a sweep line over
`(start, end)` sorted by `start`, tracking a running `max(end)`:

```python
busy = 0; gaps = []; cur_end = rows[0][0]
for s, e, g in rows:                 # rows ordered by start
    if s > cur_end:                  # union gap: nothing resident on the GPU
        gaps.append(s - cur_end); cur_end = s
    if e > cur_end:
        busy += e - cur_end; cur_end = e
```

The whole trace contains prefill, warm-up, several sweep phases and long idle
stretches, so raw whole-trace idle (17.2 %) is meaningless. I split the kernel
stream at union-gaps > 1 ms, which yields 20 contiguous runs, and analysed the
longest one. Cross-checks that it is genuinely C1 decode: 100 % graph-launched;
`routingIndicesBlockKernel` fires **77.8× per `run_batch`** (consistent with a
79-layer model with one dense layer) and the DSA indexer kernels fire **21.9× per
`run_batch`** (consistent with `index_topk_freq = 4` → 22 indexing layers); mean
kernel duration 4.41 µs.

### 2.2 CUDA graphs are already on, and already work

```
=== graph launch breakdown, device 0, whole trace ===
  cudagraph        launches = 1,132,105   gpu_ms = 6770.3
  stream(eager)    launches =    51,415   gpu_ms = 2195.8
```

95.7 % of launches are graph replays. The 51,415 eager launches average **42.7 µs**
each — they are prefill kernels, on stream 124. From `server.log`:

```
Breakable CUDA graph is incompatible with MLA attention; disabling prefill CUDA graph.
cuda_graph_config = CudaGraphConfig(
    decode  = PhaseConfig(backend='full', max_bs=512, bs=[1,2,3,...,512]),   # 67 buckets
    prefill = PhaseConfig(backend='disabled', ...))
Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=4,
    bs=[1,2,3,4,5,6,7,8,10,...,48], avail mem=23.69 GB
```
`[verified — /home/aman/code/benchmark/runs/sweep-latency-3-1-4/server.log]`

So decode is **full-graph**, 67 batch-size buckets up to 512, plus a separate
target-verify graph at `num_tokens_per_req = 4` (EAGLE 3-1-4) for bs ≤ 48.

### 2.3 The gap histogram — the negative result

Longest contiguous run: **3621.5 ms wall, 804,823 kernels**.

| quantity | value |
|---|---:|
| union busy | 3552.8 ms (**98.10 %**) |
| union idle (all inter-kernel gaps) | **68.7 ms (1.90 %)** |
| number of gaps | 119,092 |
| gap p50 | **0.22 µs** |
| gap p90 | 0.54 µs |
| gap p99 | 7.78 µs |
| gap max | 640.7 µs |
| mean kernel duration | 4.41 µs |
| Σ kernel durations / wall | 4962.8 / 3621.5 = **1.37× stream overlap** |

Per-`run_batch` decomposition of the same trace (442 batches, middle 80 %):

```
wall 1862.2 ms   busy 1850.4 ms (99.4 %)   idle 14.0 ms (0.8 %)
kernels 461,310  (461,266 from cuda graphs = 100.0 %)  mean kernel 4.01 us
```

**Inside a forward pass the GPU is 99.4 % busy.** A 0.22 µs median gap is below
the 0.5 µs quantisation floor of a graph replay dispatch and is essentially the
hardware's block-drain time. There is no launch-overhead recovery available here.
Anyone who tells you a megakernel will save the launch overhead on this workload
is describing a bubble that our graphs already closed.

### 2.4 The real bubble: grid width

The GPU being "busy" means *at least one CTA is resident somewhere*. It says
nothing about how many of the 148 SMs are doing anything. Grid sizes in the decode
window:

| grid blocks | launches | share of launches | GPU ms | share of kernel time |
|---|---:|---:|---:|---:|
| **== 1** | 152,052 | 18.9 % | **530.9** | **10.7 %** |
| ≤ 8 | 237,127 | 29.5 % | 767.0 | 15.5 % |
| ≤ 16 | 267,214 | 33.2 % | 874.3 | 17.6 % |
| ≤ 74 (half the SMs) | 452,626 | **56.2 %** | 2075.9 | **41.8 %** |
| ≤ 148 (one wave) | 734,564 | 91.3 % | 4172.0 | 84.1 % |
| > 148 | 70,259 | 8.7 % | 790.8 | 15.9 % |

`[verified — measured]`

The narrow kernels, by GPU time:

| kernel | grid | block | launches | GPU ms | % kernel time | mean µs |
|---|---:|---:|---:|---:|---:|---:|
| `oneshotAllreduceFusionKernel` | **32** | 96 | 58,656 | 593.8 | **12.0 %** | 10.12 |
| `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | 48 | 256 | 29,328 | 172.8 | 3.5 % | 5.89 |
| `routingIndicesBlockKernel` | **1** | 256 | 29,328 | 154.4 | 3.1 % | 5.27 |
| `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` | 64 | 256 | 30,456 | 122.1 | 2.5 % | 4.01 |
| `act_and_mul_kernel` | **1** | 256 | 29,328 | 116.0 | 2.3 % | 3.95 |
| `flashinfer …rmsnorm RMSNorm` (a) | **1** | 128 | 29,704 | 112.8 | 2.3 % | 3.80 |
| `splitKreduce_kernel` | 16 | 32 | 29,328 | 101.4 | 2.0 % | 3.46 |
| `topk_small_batch_kernel` | 32 | 1024 | 8,272 | 88.5 | 1.8 % | 10.70 |
| `set_mla_kv_buffer_kernel` | 4 | 128 | 29,704 | 62.1 | 1.3 % | 2.09 |
| `flashinfer …rmsnorm RMSNorm` (b) | **1** | 128 | 29,704 | 51.1 | 1.0 % | 1.72 |
| `vectorized_elementwise_kernel` | 24 | 128 | 28,576 | 48.9 | 1.0 % | 1.71 |
| `fused_k_indexer_norm_rope_store` | **1** | 128 | 8,272 | 22.5 | 0.5 % | 2.72 |

`[verified — measured]`

Read the `grid == 1` rows together: `routingIndicesBlockKernel` + `act_and_mul` +
two rmsnorms + `fused_k_indexer_norm_rope_store` = **456.8 ms**, 9.2 % of decode
kernel time, executing on **one SM out of 148** with a 3–5 µs duration each. That
is the exact shape a megakernel eliminates: not by removing a launch, but by
letting the other 147 SMs run the *next* instruction while one of them does the
routing scan.

`oneshotAllreduceFusionKernel` deserves its own line. 32 CTAs, 12.0 % of decode
kernel time, 10.1 µs mean. HazyResearch measured on 8×B200 that *"only 8–16 SMs
out of 148 are needed to nearly saturate NVLink bandwidth"*
`[reported —` [PGL post](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)`]`,
so 32 CTAs is a defensible width for the *communication*. The waste is that
nothing else is scheduled underneath it. 593.8 ms × 116/148 ≈ **465 ms of idle
SM-time** hiding behind the all-reduce, which is 9.4 % of total decode kernel
time. This is the same finding as the ledger's "47 % of collective time is
rank-arrival skew" seen from the SM side rather than the rank side.

Cluster usage, for reference: 239,748 launches (1913.2 ms, **38.5 %** of kernel
time) run with `clusterDim > 1`, i.e. the tcgen05 GEMM and FMHA paths already use
thread-block clusters. `[verified — measured]`

### 2.5 The ceiling

A rigorous sweep-line over kernel start/end events, accumulating
`min(Σ resident gridDim, 148)` at each instant:

| instantaneous SM coverage | wall ms | % of wall |
|---|---:|---:|
| 0 SMs (true launch gap) | 68.7 | **1.9 %** |
| 1–15 SMs | 246.0 | 6.8 % |
| 16–36 SMs | 640.5 | 17.7 % |
| 37–73 SMs | 208.3 | 5.8 % |
| 74–110 SMs | 542.6 | 15.0 % |
| 111–147 SMs | 681.8 | 18.8 % |
| **148 SMs (full)** | 1233.7 | **34.1 %** |

**Mean SM coverage = 65.4 %.** Only a third of decode wall clock has every SM
holding a CTA; a quarter of it runs on a quarter of the machine or less.

Packing all the SM-work in the window into a 148-wide machine gives an ideal wall
of **2368.7 ms**, a **1.53× ceiling**. `[inferred from verified measurement]`

Caveats, stated plainly:

- `min(grid, 148)` **over**-counts for `grid > 148` (it assumes a perfect wave
  with no quantisation tail) and **under**-counts nothing. So 65.4 % is an
  *optimistic* estimate of SM coverage; true coverage is lower and the ceiling is
  ≥ 1.53×.
- "A CTA is resident" ≠ "the SM is doing useful math". Occupancy within a CTA,
  memory stalls and tensor-core idle are invisible here. An ncu pass is required
  to convert this into a fraction-of-roofline. `[not measured]`
- Part of the 1.53× is unreachable: some of the narrow work is a genuine
  dependency chain (you cannot start the down-projection before the router picks
  experts), and decode is partly HBM-bound so widening a kernel does not always
  shorten it.
- Under nsys the observed pace was 9.61 ms per target-forward group vs ~8.66 ms
  implied by the un-profiled 365 tok/s — about **10 % profiler overhead**, which
  is consistent and does not change any ratio above.

---

## 3. The HazyResearch megakernel line of work

### 3.1 "Look Ma, No Bubbles" (May 2025) — the founding result

Model **Llama-3.2-1B** (1.24 B params), bf16, **batch size 1**, 32-token prompt /
128 generated tokens, no speculation. The entire forward pass is one kernel.
`[verified —` [post](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)`]`

| metric | H100 | B200 |
|---|---:|---:|
| forward-pass latency | < 1 ms | **~680 µs** |
| memory-bandwidth utilisation | **78 %** | not stated |
| speedup vs vLLM | 2.5× | **> 3.5×** |
| speedup vs SGLang | > 1.5× | > 1.5× |
| baseline systems' bandwidth utilisation | ≤ 50 % | — |

Their **B200** breakdown of a ~600 µs forward pass `[reported]`:

| component | µs |
|---|---:|
| storing activations, awaiting consistency, re-loading them | **250** |
| RMS norm + matrix-vector compute | 200 |
| awaiting weights from global memory | 30 |
| low-level cross-warp synchronisation | 40 |
| setup and other | 80 |

Note what this says: even inside a megakernel on B200, **250 of 600 µs is
activation round-tripping and memory-consistency waiting**, not compute. The
megakernel is not free; it relocates the cost.

The motivating arithmetic they give: 7 kernels/layer × 16 layers, "even with an
optimistic 5 µs of stalling per kernel … generation would run at just ~770
forward passes per second." `[reported]`

### 3.2 The design, read from the source

`github.com/HazyResearch/Megakernels`, `include/config.cuh` — this is the whole
resource contract in 50 lines `[verified — file read]`:

```cpp
struct default_config {
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;
    static constexpr int INSTRUCTION_WIDTH = 32;   // 128 bytes per instruction
    using instruction_t = int[INSTRUCTION_WIDTH];
    static constexpr int TIMING_WIDTH = 128;
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, one mma (launcher) warp.
    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS   = 4 + NUM_CONSUMER_WARPS;   // 20 warps = 640 threads
    static constexpr int NUM_BLOCKS  = 1;                        // 1 CTA per SM
    static constexpr int CLUSTER_BLOCKS = 1;

    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
              (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int PAGE_SIZE = 16384;                      // 16 KiB
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");      // on H100

    static constexpr int CONSUMER_REGISTERS     = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};
```

Every design decision in the family is visible here:

1. **Fixed 128-byte instruction word** (32 ints). The whole program is a
   `[num_sms][queue_len][32]` int tensor in global memory, built on the Python
   side and reused across hundreds of forward passes.
2. **One CTA per SM** (`NUM_BLOCKS = 1`, `__launch_bounds__(NUM_THREADS, 1)`).
   The persistent kernel *is* the occupancy.
3. **Warp specialisation with 5 roles.** From `megakernel.cuh`:
   ```cpp
   if (warpid() < config::NUM_CONSUMER_WARPS) {
       warpgroup::increase_registers<config::CONSUMER_REGISTERS>();      // 104
       consumer::main_loop<config, globals, ops...>(g, mks);
   } else {
       warpgroup::decrease_registers<config::NON_CONSUMER_REGISTERS>();  // 64
       switch (warpgroup::warpid()) {
         case 0: loader::main_loop  (g, mks); break;   // TMA / global loads
         case 1: storer::main_loop  (g, mks); break;   // stores, cross-GPU sends
         case 2: launcher::main_loop(g, mks); break;   // MMA issue
         case 3: controller::main_loop(g, mks); break; // instruction fetch + page alloc
         default: asm volatile("trap;");
       }
   }
   ```
   `setmaxnreg` moves registers from the 4 scheduling warps to the 16 consumer
   warps. This is the same producer/consumer split as a warp-specialised GEMM,
   lifted to whole-model scope.
4. **The interpreter loop is a macro-generated dispatch over opcodes**
   (`MAKE_WORKER` in `util.cuh`):
   ```cpp
   int num_iters = g.instructions.rows();
   for (mks.instruction_index = 0, mks.instruction_ring = 0;
        mks.instruction_index < num_iters; mks.next_instruction()) {
       mks.await_instruction();
       dispatch_op<...>::run(mks.instruction()[0], g, mks);   // instruction[0] == opcode
   }
   ```
   Each of the 5 roles runs its *own* copy of this loop over the same instruction
   stream, and `op::loader::run` / `op::consumer::run` / … are the per-opcode
   specialisations. That is why the code is a template pack `typename... ops`:
   the compiler emits one switch with one arm per instruction type, per role.
5. **A 2-stage instruction pipeline** (`INSTRUCTION_PIPELINE_STAGES = 2`). While
   instruction *n* executes, the controller warp is already fetching *n+1* into
   the other ring slot (`controller/instruction_fetch.cuh`), and the page
   allocator is already deciding which shared-memory pages instruction *n+1*
   inherits from *n* (`controller/page_allocator.cuh`). This is the "tail of one
   op overlaps the head of the next" mechanism, and it is where the interpreter
   earns its keep: an instruction can begin issuing TMA loads the moment the
   *previous* instruction releases a page, not when the previous instruction
   ends.
6. **Page-based shared-memory management.** 13 × 16 KiB pages on H100. Ops call
   `wait_page_ready(pid)` and `finish_page(pid, count)`; the controller's
   `page_allocator_loop` computes `release_lid()` per op and threads freed pages
   to the next instruction. The `page_finished` semaphores are indexed
   `[NUM_PAGES][INSTRUCTION_PIPELINE_STAGES_BITS]`.
7. **Dependencies are counters in global memory.** From the post: *"an array of
   counters (i.e. integers) in GPU global memory"*, starting at zero; an
   instruction increments on completion, dependents spin until the counter hits a
   target. Chunked MLP uses **four counters per intermediate state** so the
   down-projection starts on the first chunk rather than waiting for the whole
   hidden state. `[reported]` The spin loop sleeps
   `GMEM_SPIN_LOOP_SLEEP_NANOS = 20` ns between polls. `[verified — config.cuh]`
8. **Blackwell path exists in the source.** `#ifdef KITTENS_BLACKWELL` adds a
   `tensor_finished` semaphore and a `tensor_allocator_t` (TMEM allocator) to the
   VM state, with `wait_tensor_ready()` gating tcgen05 use. `[verified —
   megakernel.cuh, util.cuh]`

The Python-side scheduler (`megakernels/scheduler.py`) mirrors the constants
(`INTS_PER_INSTRUCTION = 32`, `TIMING_SLOTS = 128`) and offers five SM-assignment
policies: round-robin, zig-zag, wave (group by opcode, assign by cost), **dag**
(heap-based greedy over a dependency graph with priorities), and **pool**
(partition SMs between memory-bound and compute-bound work). `[verified — file read]`

The 7 instruction opcodes for Llama-1B `[reported]`: fused RMSNorm+QKV+RoPE;
attention partial; attention reduction (ThunderGQA); O-projection + residual;
fused RMSNorm + up/gate + SiLU; down-projection + residual; RMSNorm + LM head.

### 3.3 The tensor-parallel megakernel (Sept 2025) — throughput, 8×H100

["We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU"](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main),
**Llama-70B, 8×H100**, sequence-parallel TP. `[reported]`

| metric | megakernel | SGLang |
|---|---:|---:|
| input tok/s (ShareGPT) | 14,425 | 11,783 |
| output tok/s | 9,043 | 7,387 |
| total tok/s | **23,468** | 19,170 (**+22 %**) |

Design deltas over the batch-1 version, all directly relevant to us:

- **9 instructions**, now including `RMSNorm + all-gather` and
  `down-projection + reduce-scatter + residual` — the collectives are *inside*
  the instruction set, not between kernels.
- **Dedicated storer warps** perform the cross-GPU sends in the background while
  loader/consumer warps advance to the next instruction. Worth **14.2 %** at
  bs 8192 (global work queue), **6.4 %** (interleaving compute-bound and
  comm-bound instructions across SMs), **2–6 %** (inter-instruction pipelining).
- **Post-attention "distributed transpose"** replaces reduce-scatter for an
  **8× network-traffic reduction**; the O-projection is *replicated* (data
  parallel) rather than tensor-parallel, costing ~9 GB/GPU and ~15 % of max batch
  size, to make the communication hideable.
- A per-SM profiler with **0.39 % mean overhead (1.07 % max)**.

### 3.4 PGL — "One Kernel for All Your GPUs" (Sept 2025), measured on 8×B200

`[reported —` [post](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)`]`

| finding | value |
|---|---|
| NVLink BW via copy engine | 726 GB/s (81 % of 900 GB/s) |
| NVLink BW via TMA | 669 GB/s (74 %) |
| NVLink BW via register ld/st | 541 GB/s (60 %) |
| SMs needed to saturate NVLink | **8–16 of 148** |
| all-gather vs NCCL (tensor dim) | **2.6×** |
| recommended sharing mechanism | **VMM** (`cuMemCreate` + FD over Unix socket) — IPC handles cannot use the NVSwitch accelerator |

The PGL abstraction wraps a global layout with peer addresses, multicast
addresses and TMA descriptors, exposing `tma::load_async` (P2P),
`tma::store_async` (NVSwitch broadcast), `multimem::ld_reduce` and `multimem::st`.

**Direct relevance:** our `oneshotAllreduceFusionKernel` uses 32 CTAs. PGL says 8–16
suffice. The other ~130 SMs are recoverable if the reduce is expressed as an
*instruction* co-resident with compute rather than as a kernel.

### 3.5 ThunderKittens 2.0 (Feb 2026) — the Blackwell numbers

`[reported —` [post](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)`]`
Released 2026-02-19. B200 BF16 / MXFP8 / NVFP4 GEMMs at or above cuBLAS
(500 warm-up + 100 profiled iterations, L2 evicted between). Findings that matter
for anyone writing SM100 kernels:

| finding | effect |
|---|---|
| `tcgen05.cp` is **implicitly pipelined** w.r.t. `tcgen05.mma` — no explicit barrier needed | **+~500 TFLOP/s (~10 %)** on NVFP4 GEMM |
| removing `tcgen05.fence::after_thread_sync` and `fence.proxy.async.shared::cta` where unnecessary | +~20 TFLOP/s |
| `elect.sync` instead of manual lane selection (avoids a ptxas serialisation loop) | up to +10 % on small GEMMs |
| double-accumulation TMEM buffering vs split-slot | +~100 TFLOP/s on large BF16 |
| NVFP4 scales: 2048 values/operand/MMA-stage/CTA → **12 `tcgen05.cp` per stage** (MXFP8 needs 1) | explains NVFP4 kernel complexity |
| **kernels touching TMEM are hard-limited to 1 block/SM** even using ¼ of TMEM | occupancy constraint |
| cluster size vs usable SMs on 148-SM B200 | see below |

| cluster size | active SMs |
|---:|---:|
| 2 | 148 |
| 4 | 132 |
| 8 | 120 |
| 16 | 112 |

Fix: `cudaLaunchKernelEx` with both a *preferred* and a *minimum* cluster size.
`[reported]` **This is a free correctness check for our own kernels** — 38.5 % of
our decode kernel time runs with clusters, and any of those at cluster size 8
would be leaving 28 SMs unusable.

On megakernels the post says work is ongoing and **gives no B200 megakernel
numbers**. There is, as of this writing, **no published Blackwell end-to-end
megakernel result from HazyResearch.** `[verified — absence in primary source]`

### 3.6 Together AI — the production data point

`[reported —` [Inside the Together AI kernels team](https://www.together.ai/blog/inside-the-together-ai-kernels-team)`]`

| workload | before | after | ratio |
|---|---:|---:|---:|
| Llama-3.2-1B, voice agent, H100 | 281 ms | **77 ms** | 3.6× |
| Qwen-2.5-1.5B (B200 baseline) | 292 ms | **127 ms** | 2.3× |
| unit economics | — | — | 7.2× |

Also: ThunderKittens reduces "1,000+ lines of CUDA to 100–200"; Blackwell support
landed in about a week; "up to 2× speedups over cuBLAS on H100s" for FP4/FP8 GEMM.

**Caveat that matters to us:** every published megakernel end-to-end number is on
a **1–1.5 B dense model**, where the forward pass is a pure HBM-bandwidth race and
a handful of instruction types cover everything. GLM-5.2 is a 79-layer MoE with
256 experts, sparse MLA and a speculative draft. Nothing in the literature
demonstrates a megakernel at that scale on Blackwell.

---

## 4. Mirage Persistent Kernel (MPK) — the compiler version

[arXiv 2512.22219](https://arxiv.org/abs/2512.22219), CMU/UIUC/Berkeley/NVIDIA/Tsinghua.
The claim is the first *automatic* multi-GPU mega-kernelisation. `[reported]`

**Representation.** A `ttGraph` alternating **tasks** (a unit of compute or
communication on one SM) and **events** (activated when all producer tasks
finish). Fine-grained dependencies let a MatMul→AllReduce chain overlap: each
AllReduce task depends on exactly one MatMul task.

**Runtime.** SMs are statically partitioned:

| GPU | SMs | workers | schedulers |
|---|---:|---:|---:|
| A100 | 108 | 104 | 16 |
| H100 | 132 | 128 | 16 |
| **B200** | **148** | **144** | **16** |

4 scheduler *warps* per scheduler SM. Workers loop: dequeue → execute → trigger
event. Two queues per worker (JIT and AOT); JIT drains first. Synchronisation is
device-memory semaphores plus `atomicAdd`. Multi-GPU uses NVSHMEM
(`nvshmem_signal_wait_until`) with transfer tasks in the *same* ttGraph as
compute, so overlap is automatic.

**Resource handling** — the two problems every megakernel has:

- **Shared memory:** paged, **32 KiB pages**, acquired/released monotonically.
- **Registers:** *"per-thread register usage fixed at maximum required across all
  task types"* — i.e. every task pays the worst task's register cost. This is the
  fundamental megakernel tax, stated honestly by the authors. `[reported]`

**Results** `[reported]`:

| setting | MPK | baseline | ratio |
|---|---:|---:|---:|
| Qwen3-8B, A100, per-token | 12.5 ms | 14.5 ms (vLLM/SGLang) | 1.16× (floor ~10 ms) |
| Qwen3-30B-A3B (MoE), **B200** | ~1200 µs | ~1900 µs (SGLang) | **1.58×** |
| Qwen3-1.7B, 8×H100 | — | SGLang/vLLM | 1.1–1.4× |
| overall range | — | — | 1.0–1.7× |

Ablations: cross-task pipelining 1.2–1.3× on the final linear layer (B200);
fine-grained compute/comm overlap 1.1× (4×H100); scheduler overhead **0.28 %** of
runtime; task descriptor **352 bytes**, prefetched to shared memory.

**Compile-stage scale** (Table 2, B200):

| model | operators | tasks/op | events | event-fusion reduction |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 229 | 35.6 | 1,870 | 37× |
| Qwen3-8B | 293 | 47.3 | 2,366 | 68× |
| Qwen3-30B-A3B | 533 | 32.2 | 1,142 | **118×** |

Without event fusion the graphs would carry 69k–162k producer/consumer pairs.
Implementation size: 44k C++ + 42k CUDA + 10k Python. `[reported]`

**The number that matters for us:** MPK's own B200 MoE result is **1.58× on a
30 B MoE**, which is within shouting distance of the 1.53× SM-packing ceiling I
measured on GLM-5.2. Two independent routes to the same magnitude is weak
corroboration that the prize is real and is about **packing**, not launches.

---

## 5. Other megakernel work (2026)

| work | venue | hardware | claim |
|---|---|---|---|
| **Ada-MK** | [arXiv 2605.11581](https://arxiv.org/abs/2605.11581), May 2026 | NVIDIA **L20** | +23.6 % over TensorRT-LLM, +50.2 % over vLLM, single-batch; MegaKernel embedded as a **TensorRT-LLM plugin** for decode only, TRT-LLM keeps prefill. Hoists all runtime scheduling to compile time via MLIR DAG search ("under a fixed deployment configuration the optimal execution path is uniquely determined"), eliminating runtime branching. 3-D shared-memory constraint model + K-splitting cuts peak smem 50 %. Claims first industrial MegaKernel deployment (online advertising). `[reported]` |
| **AutoMegaKernel (AMK)** | [arXiv 2606.09682](https://arxiv.org/abs/2606.09682), Jun 2026 | L4, L40S, A10G, RTX 5090, A100, H100 | Agent harness that compiles a HF Llama-family model to one persistent cooperative kernel. Static schedule-IR validator certifies deadlock/race freedom (7,160 adversarial schedules, 6,091 unsafe, **zero false accepts**). Beats CUDA-graphed cuBLAS bf16 at batch-1 decode on *inference-class* parts (L4 1.33×, L40S 1.25–1.27×, RTX 5090 1.19–1.23×) but **loses on A100/H100**, which the authors attribute to a cross-SM-sync bottleneck. Precision-asymmetric (W8A16 vs bf16). `[reported]` |
| **Fleet** | [arXiv 2604.15379](https://arxiv.org/abs/2604.15379), Apr 2026 | multi-die / chiplet GPUs | "Chiplet-tasks" binding work+data to a chiplet and coordinating via its shared L2; persistent execution with per-chiplet scheduling. 1.3–1.5× lower decode latency at small batch, up to 37 % less memory traffic at large batch. `[reported]` |

**The AMK negative result is the most useful datum in this table.** A generic
megakernel *loses* to CUDA-graphed cuBLAS on H100/A100 — the high-bandwidth,
high-SM-count parts — and wins only on small inference parts. B200 is further up
that same axis than H100. Anyone proposing a from-scratch megakernel for GLM-5.2
on B200 must explain why they will not land where AMK landed.

---

## 6. The general machinery

### 6.1 Persistent kernels

The classic form: launch exactly `num_SMs × blocks_per_SM` CTAs, loop over work
items pulled from a global atomic counter. Benefits: one launch, warm L2/registers
across tiles, no re-entry cost. Costs: no hardware load balancing, no preemption,
occupancy is fixed by the worst-case resource footprint, and correctness depends
on *all* CTAs being co-resident if you use a grid barrier.

Our B200, queried directly `[verified — `cuDeviceGetAttribute`, driver 595.71.05]`:

| attribute | value |
|---|---:|
| `MULTIPROCESSOR_COUNT` | **148** |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | 233,472 B (**228 KiB**) |
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | 232,448 B (**227 KiB**) |
| `MAX_REGISTERS_PER_MULTIPROCESSOR` | **65,536** |
| `MAX_REGISTERS_PER_BLOCK` | **65,536** |
| `MAX_THREADS_PER_MULTIPROCESSOR` | 2,048 |
| `MAX_BLOCKS_PER_MULTIPROCESSOR` | 32 |
| `COOPERATIVE_LAUNCH` | **1** (supported) |
| `CLUSTER_LAUNCH` | **1** (supported) |
| compute capability | **10.0** |
| `L2_CACHE_SIZE` | 132,644,864 B (126.5 MiB) |
| `CLOCK_RATE` | 1,965,000 kHz |

### 6.2 Cluster Launch Control (CLC) — Blackwell's answer

CLC (compute capability 10.0) combines persistent-CTA efficiency with hardware
load balancing: **launch a grid sized to the work, and let already-running
clusters "steal" the launches of clusters that have not started yet.**
`[verified —` [CUDA Programming Guide §4.12](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)`]`

PTX surface `[verified — CUDA PG + CUTLASS docs]`:

```
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes[.multicast::cluster::all].b128
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
clusterlaunchcontrol.query_cancel.get_first_ctaid::dimension.b32.b128
```

`try_cancel` atomically requests cancellation of an *unlaunched* cluster and
writes a **16-byte opaque response** into shared memory, completed via an mbarrier.
One thread per cluster issues it. Querying a failed response is **undefined
behaviour** — a failure means "exit the cluster". libcu++ exposes it as
`cuda::ptx::clusterlaunchcontrol_try_cancel(&result, &bar)` and
`..._query_cancel_is_canceled(result)` / `..._get_first_ctaid_x/y/z(result)`.

CUTLASS implements it as `PipelineClcFetchAsync`: one **scheduler warp per
cluster** produces tile coordinates into a pipeline; TMA/MMA/epilogue warps
consume them. The response gives the coordinates of the cancelled cluster's
*first* CTA, so each CTA offsets by its own position within the cluster. A shared
async-proxy fence guards the next `try_cancel` from clobbering the response before
consumers read it. `[verified —` [CUTLASS Blackwell CLC doc](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)`,`
[Colfax](https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/)`]`

Colfax's measurements on B200 (148 SMs → 74 two-CTA clusters) `[reported]`:

| workload | single-tile | static persistent | CLC |
|---|---:|---:|---:|
| balanced GEMM, mxfp4, 256→32768 sq., K ∈ {2048, 8192} | ~50–70 TFLOP/s | ~160–190 | ~150–185 |
| grouped GEMM with growing K imbalance | — | degrades badly | holds ~80 % of peak |

Two rules from that work worth writing down:

1. **Use a single-stage CLC pipeline for imbalanced problems.** Depth > 1 queues
   unequal work and degenerates toward static scheduling.
2. CLC occasionally *loses* to static persistent on balanced problems because it
   scrambles L2 locality — measured L2 hit rate 35 % (CLC) vs 52 % (static) at
   (32768, 32768, 2048). `[reported]`

CLC also enables **preemption**: a failed `try_cancel` can mean "a higher-priority
kernel wants the machine", so the persistent kernel exits and resumes later. That
is the one hardware answer to the classic persistent-kernel objection.

**Relevance to us:** CLC is the *right* mechanism for the megakernel's global work
queue on Blackwell — it replaces the `atomicAdd`-on-a-global-counter scheme that
the HazyResearch TP megakernel and MPK both use with a hardware path. Neither of
those systems uses CLC in their published code. `[verified — absence]`

### 6.3 Grid-wide sync (cooperative groups) and its cost

`COOPERATIVE_LAUNCH = 1` on our B200, so `cudaLaunchCooperativeKernel` +
`cooperative_groups::this_grid().sync()` is available. The constraints are the
problem, not the barrier:

- The grid must fit **co-resident** on the device. With one 640-thread CTA per SM
  that is 148 CTAs — fine. But it means occupancy is capped by the *most*
  resource-hungry instruction in the kernel.
- **Cooperative kernels cannot be used inside CUDA graph conditional-node
  bodies.** `[verified —` [NVIDIA conditional-nodes blog](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes)`:
  "Kernel nodes (CNP, cooperative not currently supported)"]` So you cannot mix
  "megakernel with grid.sync" and "graph conditional loop for variable
  speculation depth".
- I could not source a measured `grid.sync()` latency for SM100. NVIDIA's docs
  state only that barriers cost and that you should minimise them. **Not
  sourced** — this needs a microbenchmark on our box.

**Which is why the megakernels do not use it.** HazyResearch synchronises with
global-memory counters and per-SM instruction queues; MPK uses device-memory
semaphores and `atomicAdd`. A grid barrier is a *global* rendezvous — it
reintroduces exactly the straggler problem that motivated the megakernel. The
whole point is *point-to-point* dependencies, not device-wide ones.

### 6.4 On-device scheduling and dependency resolution — three tiers

| mechanism | who signals | cost | flexibility |
|---|---|---|---|
| kernel boundary (stream) | driver + hardware scheduler | full grid drain + membar | none |
| **PDL** (`griddepcontrol`) | last CTA of kernel *n* | overlaps launch + prologue of *n+1* | pairwise, same stream |
| **megakernel counters** (L2 atomics / mbarrier) | any CTA, any granularity | spin + memory fence | arbitrary DAG |

Yifan Yang (NVIDIA) frames these exactly this way: baseline = "full hardware
synchronisation", PDL = "software-assisted hardware synchronisation", megakernel =
"software-only synchronisation via L2 atomics — most flexible but least
efficient". `[reported —` [PDL blog](https://yang-yifan.github.io/blogs/pdl/pdl.html)`]`
That last clause is the honest cost: a spin on an L2 atomic is slower *per
dependency* than a hardware barrier; the megakernel wins by needing far fewer of
them on the critical path, not by making each one cheap.

### 6.5 Register and shared-memory pressure — the hard limit

The megakernel tax, stated by MPK: *"per-thread register usage fixed at maximum
required across all task types."* `[reported]` Now put our measured kernels
against our measured device limits. (This table groups by
`(name, reg/thread, smem, block)` and so sums *all* launch shapes of a kernel;
the §2.4 table groups by `(name, grid, block)` and shows one shape — hence
`oneshotAllreduceFusionKernel` reads 602.5 ms here and 593.8 ms there.)

| decode kernel | reg/thread | block threads | reg/block | smem bytes | GPU ms |
|---|---:|---:|---:|---:|---:|
| `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` | **255** | 256 | **65,280** | 156,168 | 688.6 |
| `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN` | **255** | 256 | 65,280 | **205,184** | 228.0 |
| `fmhaSm100fKernel_QkvE4m3O…HVPerCta128Paged` | 128 | 512 | 65,536 | 166,688 | 333.6 |
| `bmm_E2m1_E2m1E2m1_Fp32_…t128x8x512` | 96 | 640 | 61,440 | **202,640** | 331.9 |
| `bmm_Bfloat16_E2m1E2m1_Fp32_…t128x8x256` | 128 | 512 | 65,536 | 184,464 | 284.4 |
| `fused_a_gemm_kernel` | 56 | 256 | 14,336 | 185,344 | 263.0 |
| `oneshotAllreduceFusionKernel` | 62 | 96 | 5,952 | 288 | 602.5 |
| `routingIndicesBlockKernel` | 32 | 256 | 8,192 | 3,232 | 154.4 |
| `splitKreduce_kernel` | 32 | 512 | 16,384 | 0 | 146.9 |

`[verified — measured from the trace]`

Three conclusions, all forced by the numbers:

1. **Registers pin you to 1 block/SM.** 65,280 of 65,536 available. There is no
   second CTA. A megakernel containing `nvjet` must run at
   `CONSUMER_REGISTERS ≈ 255`-equivalent for the MMA path; the HazyResearch config
   gets away with 104 because Llama-1B batch-1 matvecs need nothing like a
   tcgen05 GEMM's accumulator state.
2. **Shared memory is nearly exhausted by a single instruction.** 205,184 B of
   the 232,448 B opt-in maximum. A page-based allocator with 16 KiB pages gets
   **14 pages** on B200 (227 KiB / 16 KiB), and one GEMM instruction wants 13 of
   them. Software pipelining across instructions — the mechanism that produces
   the overlap — has almost no room to work at these tile sizes.
3. **TMEM caps occupancy independently.** ThunderKittens 2.0 reports kernels
   touching tensor memory are hard-limited to 1 block/SM even at ¼ TMEM
   allocation. `[reported]` Every NVFP4/FP8 GEMM in our decode uses tcgen05.

**Therefore:** a GLM-5.2 megakernel on B200 cannot recover the 1.53× by
*co-residency*. It has to recover it by **time-multiplexing one CTA/SM across
instructions with a deep enough instruction pipeline that the narrow instructions
(`grid == 1` rmsnorm, routing, `act_and_mul`) run on SMs that would otherwise
idle.** That is a scheduling win, not an occupancy win, and it is exactly what the
HazyResearch "global work queue + interleaving" ablations measured (14.2 % + 6.4 %
at bs 8192). `[inferred]`

### 6.6 Instruction-cache pressure

Blackwell's instruction hierarchy `[reported —` [Chips and Cheese](https://old.chipsandcheese.com/2025/06/28/blackwell-nvidias-massive-gpu/)`]`:

| level | size | note |
|---|---:|---|
| L0i | 32 KiB per SM sub-partition (4/SM) | one instruction/cycle |
| L1i | ~128 KiB per SM (inferred from unrolled-loop tests) | ~8K instructions at 16 B each |
| instruction width | **128 bit / 16 B**, fixed | |

The measured failure mode: *"L1 instruction bandwidth can be a visible limitation
if two waves on different partitions spill out of L1 and run different code
sections"* — dropping to **one instruction per two cycles**. `[reported]`

That is precisely a megakernel's steady state: a switch over ~10 opcodes × 5 warp
roles, with different sub-partitions executing different arms. A megakernel with
10 instruction types averaging 4 KB of SASS each is 40 KB — within 128 KiB L1i,
but the 5-role structure multiplies the resident footprint because loader, storer,
launcher, controller and 16 consumer warps are each in a different arm of a
different switch at the same time. This is a real and under-discussed risk.
**Mitigation used in practice:** Ada-MK's whole thesis — hoist the dispatch to
compile time so there is no runtime branching at all
(*"completely eliminating runtime branching"*). `[reported]`

I could not find a published measurement of L1i miss rate in a megakernel.
**Not sourced.** It is directly measurable with `ncu --metrics
smsp__inst_executed`, `sm__inst_issued`, and the frontend stall reasons.

---

## 7. Programmatic Dependent Launch — the cheap 80 % of a megakernel

PDL (Hopper+, cc ≥ 9.0) lets kernel *n+1* start its **launch and prologue** while
kernel *n* is still draining. It removes `launch overhead + prologue` from the
critical path without any of the megakernel's resource unification.
`[verified —` [PDL blog](https://yang-yifan.github.io/blogs/pdl/pdl.html)`,` [CUDA PG §3.1](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)`]`

Anatomy of a kernel under PDL:

```
[ CTA launch overhead ][ prologue: const loads, mbarrier init, smem alloc ][ mainloop ][ grid-ending membar ]
                          ^ independent of predecessor output              ^ dependent
```

Two PTX instructions:

| instruction | issued by | meaning |
|---|---|---|
| `griddepcontrol.launch_dependents` | last CTA of kernel *n*, mid-execution | "you may launch my dependents now" |
| `griddepcontrol.wait` | kernel *n+1*, after its prologue | "block until *n*'s grid-ending membar has landed" |

CUDA C++ wrappers: `cudaTriggerProgrammaticLaunchCompletion()` and
`cudaGridDependencySynchronize()`; enabled per-launch via
`cudaLaunchAttributeProgrammaticStreamSerialization` through `cudaLaunchKernelEx`.
Our local CUDA 13.3 headers confirm all three
`[verified — /home/aman/code/cuda-13.3/nvidia/cu13/include/cuda_device_runtime_api.h:464,
driver_types.h:4211]`. Requirements: same stream; the SM must have room for both
kernels' CTAs concurrently (registers, smem, threads, **TMEM**).

**PDL is already in our tree and it is switched OFF.** `[verified — code read]`

```cpp
// python/sglang/kernels/aot/include/utils.h:285
inline bool getEnvEnablePDL() {
  static bool enablePDL = false;
  std::call_once(flag, [&]() {
    if (getSMVersion() >= 90) {
      // PDL will be enabled by setting the env variables `TRTLLM_ENABLE_PDL` to `1`
      enablePDL = getBoolEnv("TRTLLM_ENABLE_PDL");
    }});
  return enablePDL;
}
```

- The **AOT C++ kernels** (`rmsnorm`, `fused_add_rmsnorm`, `gemma_rmsnorm`,
  `per_token_group_quant_8bit_v2`, `pos_enc`) gate PDL on `TRTLLM_ENABLE_PDL=1`,
  **default false**. `TRTLLM_ENABLE_PDL` does not appear anywhere in
  `runs/sweep-latency-3-1-4/server.log`. `[verified]`
- The **Triton ops** enable it unconditionally on sm_90+:
  `_ENABLE_PDL = _pdl_supported()` in
  `ops/attention/fused_qk_rmsnorm_rope_gate.py`, emitting
  `tl.extra.cuda.gdc_launch_dependents()` with the comment *"PDL: signal dependent
  kernels (attention/allreduce) can start early."* `[verified]`

So we have an inconsistent PDL story: Triton paths signal, C++ paths do not.
Setting `TRTLLM_ENABLE_PDL=1` is a **one-environment-variable experiment**. Given
the measured 0.22 µs median gap, PDL will not buy much *gap*; what it buys is
prologue overlap on the 4 µs kernels, where a 1 µs prologue is 25 % of the kernel.
`[inferred]`

One documented hazard: vLLM issue #40742 reports CUDA-graph capture crashing when
PDL is enabled together with Inductor autotuning (a `torch.cuda.synchronize()`
inside capture) under `FULL_DECODE_ONLY` + MLA + FP8. `[reported]` We run
FULL decode graphs + MLA + FP8 KV, so this is the exact configuration to watch.

---

## 8. CUDA Graphs — what they do and do not fix

### 8.1 What they eliminate

Graphs move per-launch CPU work (argument marshalling, driver validation, stream
dependency bookkeeping) to instantiation time: *"pay once for the entire graph
during instantiation, and the graph itself can then be launched repeatedly with
very little overhead."* `[verified —` [CUDA PG §4.2](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)`]`

What they do **not** eliminate:

- The **grid drain** between dependent nodes. A graph edge is still a full
  completion barrier for the producer node. Our 0.22 µs median gap *is* that
  drain, and it is already near-optimal.
- **Narrow grids.** A graph replays the same `gridDim` you captured.
- **Wave quantisation and tail effects** inside each node.
- **Host-side work between graph replays** — sampling, detokenisation, the
  scheduler loop. (`num_continuous_decode_steps` exists specifically to amortise
  this; ours is `1`.)

### 8.2 Capture cost and memory

Capture requires running the model once per bucket, with a private memory pool.
Our decode config captures 67 batch-size buckets up to 512 plus a target-verify
graph at `num_tokens_per_req=4` for 24 buckets ≤ 48. The log reports
`avail mem=23.69 GB` at capture start. vLLM's docs note that combining full and
piecewise modes *"require[s] the most memory and takes the longest to capture."*
`[reported]`

### 8.3 Updating a graph without recapturing

| API | when to use |
|---|---|
| `cudaGraphExecUpdate(exec, graph, ...)` | many params changed; topology must be identical. Fails with `cudaGraphExecUpdateErrorTopologyChanged` / `…NodeTypeChanged` / `…ParametersChanged` / `…UnsupportedFunctionChange` `[verified — driver_types.h:3819]` |
| `cudaGraphExecKernelNodeSetParams()` | few nodes changed — cheaper |
| `cudaGraphNodeSetEnabled()` | turn a node on/off without touching params |
| **device-updatable nodes** | instantiate with the device-update flag, get a `cudaGraphDeviceNode_t`, then apply `cudaGraphKernelNodeUpdate` records from *device code* via `cudaGraphKernelNodeUpdatesApply` `[verified — driver_types.h:3879–3896, 4268]` |

Device-updatable nodes are the underused one: they let a kernel running *inside*
the graph rewrite a later node's parameters (e.g. a grid dimension derived from
the number of accepted speculative tokens) without a host round-trip.

### 8.4 Conditional nodes and device graph launch

`cudaGraphNodeTypeConditional = 0x0d` with three types
`[verified — /home/aman/code/cuda-13.3/nvidia/cu13/include/driver_types.h:3605]`:

```c
enum cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf     = 0,  // body[0] if cond != 0; if size==2, body[1] if cond == 0
    cudaGraphCondTypeWhile  = 1,  // body[0] repeatedly while cond != 0
    cudaGraphCondTypeSwitch = 2,  // body[n] once, where n == cond; no body if n >= size
};
```

Availability: IF and WHILE from CUDA 12.4; IF/ELSE and SWITCH from CUDA 12.8.
`[reported — NVIDIA blog]` Handles come from `cudaGraphConditionalHandleCreate()`
(optionally with `cudaGraphCondAssignDefault`); device code sets the value with
`cudaGraphSetConditional(handle, value)`.

Restrictions from the header `[verified]`:

- Body graphs may contain **only** kernel, empty, child-graph, memset, memcpy and
  conditional nodes — recursively.
- **All kernels must be in the same CUDA context.**
- A graph containing conditionals **cannot be used as a child node, cannot be
  cloned, and only one instantiation may exist at a time.**
- **Cooperative kernels are not supported in conditional bodies.** `[reported]`

Device graph launch: instantiate with `cudaGraphInstantiateFlagDeviceLaunch`, then
from device code launch into `cudaStreamGraphFireAndForget`,
`cudaStreamGraphTailLaunch` (serial after the parent completes), or
`cudaStreamGraphFireAndForgetAsSibling`; `cudaGetCurrentGraphExec()` lets a graph
relaunch itself. `[verified — CUDA PG §4.2]`

**This is the "poor man's megakernel" for our decode loop** `[inferred]`: a WHILE
conditional whose body is one decode step, with the loop condition set on-device
from the EOS check, plus a SWITCH on accepted-token count. It keeps the CPU out of
the loop entirely without unifying registers or shared memory. Two blockers to
check: (a) our all-reduce is a custom kernel, not NCCL, so the same-context rule
is satisfiable; (b) no node in the body may be cooperative.

### 8.5 Padding, bucketing and the shape problem

Both major engines bucket and pad:

- **Ours** (SGLang lineage): 67 decode buckets `[1,2,…,32,40,44,48,…,512]`,
  `disable_cuda_graph_padding=False`, so a real batch replays the smallest graph
  ≥ its size with the remainder as inert padding slots. `[verified — server.log]`
- **vLLM v1** dispatches on a `BatchDescriptor(num_tokens, num_reqs, uniform,
  has_lora)` through five modes: `NONE`, `PIECEWISE`, `FULL`,
  `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE` (default). Attention backends declare
  `AttentionCGSupport ∈ {ALWAYS, UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE,
  NEVER}`; the dispatcher downgrades to the closest supported mode and falls back
  to eager when no key matches. `[reported —` [vLLM cuda_graphs design doc](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md)`]`
- **TensorRT-LLM** uses `enable_piecewise_cuda_graph` with
  `capture_num_tokens: [1,2,4,…,3072]` and `cuda_graph_config.enable_padding`,
  keeping attention eager because of its *"substantial host-side overhead"*.
  `[reported —` [TRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/features/torch_compile_and_piecewise_cuda_graph.html)`]`

Padding waste is real but bounded at C1 (bucket 1 is exact). It bites at C64 where
the bucket ladder jumps 32 → 40 → 44 → 48.

### 8.6 Speculative decoding × graphs — the variable-acceptance problem

EAGLE breaks the fixed-shape assumption twice: the draft tree has a fixed *width*
(`num_draft_tokens = 4` here) but a variable number of *accepted* tokens, and the
next step's KV layout depends on that count.

How it is handled:

- **Fixed the shape, vary the content.** The verify graph is captured at
  `num_tokens_per_req = 4` unconditionally; rejected positions are written to a
  padding slot. vLLM does the same: rejected tokens map to `PADDING_SLOT_ID (-1)`
  to suppress spurious cache writes, and the capture range extends by
  `K × max_num_seqs`. `[reported]` vLLM's `uniform` flag explicitly covers
  "query length = `1 + num_spec_tokens`". `[reported]`
- **Cost:** you always pay for 4 draft tokens even when 1 is accepted. Our
  measured accept is 3.16/4 on real ShareGPT data (4.00/4 on synthetic — the
  synthetic number inflates decode 22 %). So ~21 % of verify work is discarded.
  `[verified — ledger §4 #8]`
- **`num_speculative_steps` is fixed at init** and one graph is captured for it;
  changing depth at runtime is an open request upstream. `[reported —
  sgl-project/sglang issue #21459]`
- Our attempts at depth 4-1-5 and 5-1-6 crash with an IMA in
  `eagle_worker_v2.py:366` **during draft CUDA-graph capture** — i.e. the graph
  machinery is the thing blocking a known 1.21× (published 540 tok/s at 5-1-6).
  `[verified — ledger §4 #3–6]`

**Conditional SWITCH nodes are the principled fix** `[inferred]`: capture one body
per accepted-length class, switch on a device-set condition, and stop padding.
Nobody has published this for EAGLE. It is a genuinely open engineering item.

### 8.7 NCCL and collectives inside graphs

NCCL supports capture: *"NCCL operations can be launched on a stream captured by a
CUDA graph for each rank, and mixing graph-captured and non-graph-captured NCCL
operations is supported."* `[reported —` [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)`]`
NVLS user-buffer registration requires `ncclMemAlloc` **and** graph capture per
rank; NCCL 2.27 symmetric memory claims **up to 9× lower latency for small
messages**. `[reported]` One reported regression: a large gap between an NCCL
AllReduce and the following kernel under graphs in NCCL 2.27 that did not exist in
2.21. `[reported — NVIDIA/nccl issue #1692; not reproduced here]`

**We are not on that path.** Our config has `enable_nccl_nvls=False`,
`enable_symm_mem=False`, `disable_custom_all_reduce=False` — the all-reduce is
TRT-LLM's MNNVL `oneshotAllreduceFusionKernel`, captured directly into our graphs.
`[verified — server.log + trace]`

---

## 9. Fusion, more broadly

### 9.1 What our own engine can fuse and currently does not

Straight from `server_args` in the profiled run `[verified — server.log]`:

| flag | value | what it does |
|---|---|---|
| `enable_fused_moe_sum_all_reduce` | **False** | "Enable fused moe triton and sum all reduce" — folds the MoE combine into the all-reduce. Directly attacks the 12.0 % `oneshotAllreduceFusionKernel`. |
| `enable_flashinfer_allreduce_fusion` | **False** | folds the all-reduce into RMSNorm |
| `enable_fused_qk_norm_rope` | **False** | one kernel for QK-norm + RoPE (we have `fused_qk_rmsnorm_rope_gate.py`, which is also the PDL-enabled Triton op) |
| `disable_shared_experts_fusion` | **True** | shared-expert fusion off |
| `enable_two_batch_overlap` | **False** | blocked on this model by `index_topk_freq=4` |
| `enable_single_batch_overlap` | **False** | |
| `enable_torch_compile` | **False** | `torch_compile_max_bs=32` |
| `num_continuous_decode_steps` | **1** | "Run multiple continuous decoding steps to reduce scheduling overhead" |
| `moe_runner_backend` | `flashinfer_trtllm` | source of `routingIndicesBlockKernel` (grid 1) |

That is at least three fusion switches that are off, two of which target the
exact kernels my grid-width analysis flagged. The ledger already records that
every measurement was taken with the overlap machinery disabled; this extends
that to the fusion machinery.

Fused kernels the tree already ships (`python/sglang/kernels/ops/`):
`fused_qk_norm_rope_store`, `fused_qk_rmsnorm_rope_gate`, `fused_qknorm_rope`,
`fused_store_index_cache`, `fused_a_gemm` (+ CuTe-DSL variant),
`moe_fused_gate`, `moe_route_quant_fused`, `moe_finalize_fuse_shared`,
`moe_fused_mul_sum`, `inkling_ar_fused`, `fused_fp8_qkv_kv_cache`,
`fused_kv_materialize`, `fused_eh_norm`. `[verified — directory listing]`

### 9.2 torch.compile / Inductor for inference

- **vLLM**: piecewise compilation exists *to enable* piecewise graph capture —
  split the FX graph at attention, capture the token-wise pieces, run attention
  eager. Full graphs became possible only as attention backends grew
  cudagraph-safe metadata (FA3 first). Inductor-level graph partitioning
  (`use_inductor_graph_partition=True`) needs torch ≥ 2.9 and is experimental.
  `[reported]`
- **TensorRT-LLM**: the torch.compile backend fuses at ATen IR level —
  **AllReduce + residual + RMSNorm + optional FP8/FP4 quantisation into one op**,
  and converts AllReduce to user buffers to avoid a copy. `[reported]`
- **Ours**: `enable_torch_compile=False`, and the `tc_piecewise` graph backend
  exists but decode uses `full`.

The relevant lesson is TRT-LLM's: the highest-value compiler fusion in a TP
decode is **collective + norm + quant**, which is the same target as
`enable_flashinfer_allreduce_fusion`.

### 9.3 Authoring tools for fused ops on Blackwell

| tool | status | Blackwell notes |
|---|---|---|
| **Triton** | ubiquitous | `tl.extra.cuda.gdc_launch_dependents()` exposes PDL; TLX work adds CLC `[reported — PyTorch blog "Enabling Cluster Launch Control with TLX"]` |
| **ThunderKittens 2.0** | Feb 2026 | full SM100: tcgen05, TMEM, MXFP8, NVFP4, at/above cuBLAS; megakernel interpreter template `[reported]` |
| **CUTLASS / CuTe DSL** | production | the reference CLC implementation (`PipelineClcFetchAsync`); our tree already uses CuTe-DSL (`cutedsl_dsv3_fused_a_gemm.py`) |
| **TileLang** | production via TileRT | the compiler layer under TileRT; techniques "gradually shared … as they are integrated into TileLang and TileScale" `[verified — TileRT README]` |
| **Helion** (pytorch/helion) | 1.0 GA April 2026 | dedicated Blackwell backend via Triton-TileIR; autotunes ~10 min over hundreds of Triton candidates; claimed 2.12–2.63× over TileLang and 1.2–1.85× over Triton **on H100** `[reported]` — no B200 comparison sourced |
| **Mosaic GPU** | JAX-side | **not sourced** for Blackwell LLM inference |

### 9.4 TileRT — the competitor's thesis, read from the local checkout

`third_party/TileRT` v0.1.5 is installed on this box. Reading it is more
informative than reading their blog.

**The thesis** `[verified — README]`: *"LLM operators are decomposed into
fine-grained tile-level tasks, while the runtime dynamically reschedules
computation, I/O, and communication across multiple devices in a highly
overlapped manner."* And, from their site: *"TileRT statically compiles the whole
model ahead of time into a persistent Engine Kernel"*, with warp specialisation
and **GPU specialisation** (in GLM-5.1 attention, *"GPU 0 becomes a Sparse Indexer
worker"* while the others run MLA). `[reported]`

**The Python surface proves the granularity.** `tilert/models/glm_5/_dsa_v32/ops/`
contains exactly 23 ops, and the names are the design document:

```
rmsnorm_projx_wqkva      rmsnorm_projq_wqb      rmsnorm_projx_wqakis
rmsnorm_expert_proj      rmsnorm_up_gate_silu   rmsnorm_head_proj
rmsnorm_kv               rmsnorm_quant          layernorm_rope_rotate
expert_sel_up_gate_silu  expert_down_allreduce  down_allreduce
unproj_o_allreduce       projo_wkvb             projq_wqb
projx_wis                projx_wqaki            projx_wqkva
head_proj_w16a16_hmma    rotate
broadcast_selected_token_ids   receive_selected_token_ids
```

`[verified — directory listing]` Note `down_allreduce`, `expert_down_allreduce`
and `unproj_o_allreduce`: **the collective is inside the GEMM op**, not after it.
`down_allreduce.py` is a thin wrapper over `torch.ops.tilert.down_allreduce_op(
vec_in, mat_in, mat_scale, x_in, flag, vec_out, model_arch, compute_kernel_type)`.
That is the tile-level compute/communication overlap, made concrete.

**The decode loop is one op that runs N steps.** `generator.py` calls
`self.decode_layer.show_hands(prev_draft, ar_steps)` — described in the source as
*"MTP decode (unified single-op, unified-style)"* — which dispatches
`torch.ops.tilert.dsa_mtp_show_hands_glm5(token_id, ar_steps)`. The host does not
re-enter per token; it reads back `ar_accepted_tokens` / `ar_num_accepted`
afterwards. `[verified — tilert/models/glm_5/generator.py:365,
modules/end2end.py:130–137]`

**And they still use CUDA graphs underneath.** `update_sampling_config()` prints
`"Recapturing CUDA graphs: temperature=…, top_p=…"`. `[verified —
modules/end2end.py:290–297]` So TileRT is not a pure megakernel: it is a
persistent engine kernel + graphs + on-device autoregressive stepping.

**Their numbers on our hardware** `[reported — SemiAnalysis InferenceX,
tilert.ai]`:

| workload | TileRT, 8×B200 | comparison |
|---|---:|---|
| GLM-5 FP8 (744 B), 1k/1k | **494.2 tok/s/user** | 1.9× GB300 NVL72 FP4 (256.3); 3.6× best conventional FP8 (136.3) |
| GLM-5 FP8, 8k/1k | 340 tok/s/user | vs GB300 154 tok/s/user — but GB300 does 240 tok/s/GPU aggregate vs TileRT's 160.4 |
| DeepSeek-V3.2 | ~600 tok/s | |
| MiMo-V2.5-Pro (1 T params) | > 1000 tok/s | with FP4 experts + FP8 elsewhere, DFlash spec |

The throughput column is the honest counterweight: TileRT trades aggregate
tok/s/GPU for per-user latency. That is the same trade our latency mode makes.

---

## 10. Concrete assessment for GLM-5.2 decode on 8×B200

### 10.1 How much of the 2.74 ms TPOT is launch/dependency overhead?

**Direct answer: at most ~52 µs, and realistically ~0.**

| component | measured | per 2.74 ms TPOT |
|---|---:|---:|
| inter-kernel gaps (nothing resident) | 1.90 % of wall | **52 µs** |
| gaps per forward-pass group | 119,092 / 377 = **316 gaps**, mean 0.577 µs | 182 µs per group of ~3.16 tokens = **58 µs/token** |
| cross-check: 1.90 % × 2.74 ms | — | 52 µs — agrees with the line above to within the profiler's ~10 % pacing overhead |

Even zeroing every gap yields **1.019×**: 365 → 372 tok/s. That closes 2 % of the
gap to TileRT's 494.

**The dependency-serialisation cost, however, is large and is a different number.**
Time-sliced SM coverage is 65.4 %; perfect packing gives **1.53×** (365 → 558
tok/s). Decomposing where that 34.6 % of unused SM-time sits:

| source | wall share | recoverable by |
|---|---:|---|
| all-reduce running 32-wide (116 SMs idle) | ~9.4 % of kernel time | overlap comm with compute (SBO / allreduce fusion / tile-level) |
| `gridDim == 1` kernels (147 SMs idle) | 10.7 % of kernel time | fusion into neighbours, or a work queue |
| `gridDim ≤ 16` beyond that | ~7 % of kernel time | widening / fusion |
| wave quantisation in `grid` 17–147 kernels | remainder | tile-shape tuning, CLC |
| true launch gaps | **1.9 % of wall** | PDL, graphs (already done) |

`[verified measurement + inferred attribution]`

**So: the honest answer to "how much of TPOT is launch overhead" is under 2 %, and
the reason the megakernel literature reports 1.2–1.7× is that they are measuring
the packing win and calling it a launch win.** That reframing is the single most
useful thing in this document.

### 10.2 How to measure it — the exact recipe

**(a) Gap analysis from the nsys SQLite export.** Reproduce §2 with:

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
     -o trace --capture-range=cudaProfilerApi --duration=20 <cmd>
nsys export --type sqlite --output trace.sqlite trace.nsys-rep
```

`--cuda-graph-trace=node` is essential — the default (`graph`) reports one row per
*graph launch*, hiding every node. Then the sweep line in §2.1, plus:

```sql
-- launches by provenance
SELECT CASE WHEN graphId IS NULL OR graphId=0 THEN 'eager' ELSE 'graph' END AS k,
       COUNT(*), SUM(end-start)/1e6 AS gpu_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=0 GROUP BY k;

-- SM coverage per kernel class (148 SMs)
SELECT s.value, gridX*gridY*gridZ AS grid, COUNT(*), SUM(k.end-k.start)/1e6 AS ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
WHERE k.deviceId=0 AND k.start>=:A AND k.end<=:B
GROUP BY 1,2 ORDER BY ms DESC;
```

Split the stream at union-gaps > 1 ms first, or prefill and phase boundaries will
dominate and you will "discover" 17 % idle that is really the harness.

**(b) Built-in recipes** (nsys 2025.6.3 on this box has them):
`nsys recipe gpu_gaps`, `cuda_gpu_kern_pace`, `cuda_gpu_time_util_map`,
`cuda_gpu_kern_hist`. `gpu_gaps` does exactly the union-gap analysis; use it as a
cross-check on the SQL.

**(c) The SM-coverage sweep line** (§2.5) is the metric that actually predicts the
megakernel prize. It is 20 lines of Python over `(start, end, gridDim)` and should
become a standing metric in the benchmark harness — call it *SM-coverage* and
track it next to tok/s.

**(d) What is still missing.** Everything above is a *scheduling* metric. It
cannot distinguish "SM has a CTA and is at 90 % of roofline" from "SM has a CTA
and is stalled on HBM". That requires `ncu --set full` on the top 6 kernels
(`nvjet_…_TNT`, `oneshotAllreduceFusionKernel`, `fmhaSm100f…`, the two `bmm_E2m1`,
`fused_a_gemm`) — candidate E in the ledger, still not done, and it is the
gate on any claim that widening a kernel would actually speed it up.

### 10.3 The cheapest intervention, ranked

| # | intervention | cost | expected | how to falsify |
|---|---|---|---|---|
| 1 | `TRTLLM_ENABLE_PDL=1` | one env var, one restart | prologue overlap on ~1000 sub-8 µs kernels/forward. `[unverified magnitude]` | A/B the same benchmark; watch for the vLLM #40742 capture crash with FULL+MLA+FP8 |
| 2 | `enable_flashinfer_allreduce_fusion` + `enable_fused_moe_sum_all_reduce` | two flags | attacks the 12.0 %-of-time, 32-CTA all-reduce and the MoE combine | already queued as ledger §4 #9; measure SM-coverage before/after, not just tok/s |
| 3 | `enable_fused_qk_norm_rope` | one flag | removes narrow norm/RoPE kernels; also the PDL-enabled Triton path | check accuracy — it changes the numerics order |
| 4 | Fuse or widen the `gridDim == 1` kernels | days | 10.7 % of kernel time at 1/148 occupancy; `act_and_mul` into the up/gate GEMM epilogue is textbook | ncu first — if `routingIndicesBlockKernel` is latency-bound on a serial scan, widening does nothing |
| 5 | `num_continuous_decode_steps > 1` | one flag | amortises the host scheduler across steps; costs TTFT | measure at C1; the docstring warns about TTFT |
| 6 | Audit cluster sizes in our kernels | hours | TK 2.0: cluster 8 → only 120 of 148 SMs usable; use `cudaLaunchKernelEx` with preferred **and minimum** cluster size | 38.5 % of decode time already runs clustered — check which sizes |
| 7 | Conditional-node decode loop (WHILE + SWITCH on accept length) | weeks | removes host round-trip per step and the ≥ 21 % verify padding waste | blockers: no cooperative kernels in bodies; single instantiation; same context |
| 8 | Tile-level comm/compute overlap in `down_proj` (TileRT's `down_allreduce`) | weeks | the 9.4 % of SM-time idle under the all-reduce | this is the one part of TileRT the ledger already flags as "a rewrite rather than a flag" |
| 9 | Full megakernel for GLM-5.2 | months | ≤ 1.53× ceiling | AMK lost to CUDA-graphed cuBLAS on H100/A100; registers pin us to 1 CTA/SM; no published Blackwell megakernel at this model scale exists |

**Recommendation.** Items 1–3 are environment variables and should be measured
this week; together they target ~15 % of decode kernel time and cost nothing. Item
4 is the first real engineering and has a clean measurement gate (ncu). Item 9 is
not the next thing to do, and the measurement in §2 is why: the launch overhead a
megakernel removes is already gone, and the packing win it delivers is available
in cheaper increments.

---

## Sources

**Read in full (primary):**

- HazyResearch, *Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B* — https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- HazyResearch, *We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU* — https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
- HazyResearch, *One Kernel for All Your GPUs* (PGL) — https://hazyresearch.stanford.edu/blog/2025-09-22-pgl
- HazyResearch, *ThunderKittens 2.0: Even Faster Kernels for Your GPUs* — https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
- HazyResearch/Megakernels source — https://github.com/HazyResearch/Megakernels
  - `include/config.cuh`, `include/megakernel.cuh`, `include/util.cuh`,
    `include/controller/instruction_fetch.cuh`, `include/controller/page_allocator.cuh`,
    `megakernels/scheduler.py`, `megakernels/instructions.py`
- *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*, arXiv 2512.22219 — https://arxiv.org/abs/2512.22219 (HTML read: https://arxiv.org/html/2512.22219)
- Zhihao Jia, *Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference* — https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- *Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference*, arXiv 2605.11581 — https://arxiv.org/abs/2605.11581
- *AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis*, arXiv 2606.09682 — https://arxiv.org/abs/2606.09682
- *Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs*, arXiv 2604.15379 — https://arxiv.org/abs/2604.15379
- NVIDIA, *CUDA Programming Guide §4.12 Work Stealing with Cluster Launch Control* — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html
- NVIDIA, *CUDA Programming Guide §4.2 CUDA Graphs* — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- NVIDIA, *Dynamic Control Flow in CUDA Graphs with Conditional Nodes* — https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes
- NVIDIA CUTLASS, *Blackwell Cluster Launch Control* — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html
- Colfax Research, *Dynamic Persistent Tile Scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs* — https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/
- Yifan Yang (NVIDIA), *Programmatic Dependent Launch* — https://yang-yifan.github.io/blogs/pdl/pdl.html
- NVIDIA NCCL, *Using NCCL with CUDA Graphs* — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
- vLLM, *CUDA Graphs* design doc — https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md
- TensorRT-LLM, *Torch Compile & Piecewise CUDA Graph* — https://nvidia.github.io/TensorRT-LLM/features/torch_compile_and_piecewise_cuda_graph.html
- Together AI, *Inside the Together AI kernels team* — https://www.together.ai/blog/inside-the-together-ai-kernels-team
- SemiAnalysis InferenceX, *Ultra-High Interactivity on NVIDIA GPUs? TileRT on InferenceX* — https://inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia
- TileRT, *Breaking 1000 TPS on a 1T Model* — https://www.tilert.ai/blog/breaking-1000-tps.html
- Chips and Cheese, *Blackwell: Nvidia's Massive GPU* — https://old.chipsandcheese.com/2025/06/28/blackwell-nvidias-massive-gpu/
- PyTorch, *Helion: A High-Level DSL for Performant and Portable ML Kernels* — https://pytorch.org/blog/helion/

**Local primary sources read:**

- `/home/aman/code/cuda-13.3/nvidia/cu13/include/driver_types.h` — `cudaGraphConditionalNodeType` (:3605), `cudaGraphNodeTypeConditional` (:3659), `cudaGraphExecUpdateResult` (:3819), `cudaGraphDeviceNode_t` / `cudaGraphKernelNodeUpdate` (:3879–3896), launch attributes (:4211–4226)
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda_device_runtime_api.h` — `cudaGridDependencySynchronize` (:464), programmatic-launch docs (:327–459)
- `/home/aman/code/benchmark/runs/sweep-latency-3-1-4/trace.sqlite` — 1,183,520 device-0 kernels; all gap, grid-width and SM-coverage numbers in §2
- `/home/aman/code/benchmark/runs/sweep-latency-3-1-4/server.log` — full `ServerArgs`, CUDA-graph capture config
- `/home/aman/code/NotSglang/python/sglang/kernels/aot/include/utils.h:285` — `getEnvEnablePDL()` / `TRTLLM_ENABLE_PDL`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py` — Triton `gdc_launch_dependents()`
- `/home/aman/code/NotSglang/python/sglang/srt/model_executor/cuda_graph_config.py` — `full` / `breakable` / `tc_piecewise` / `disabled` backends
- `/home/aman/code/NotSglang/python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py` — segmented-graph capture with eager break points
- `/home/aman/code/NotSglang/python/sglang/srt/server_args.py` — `enable_fused_moe_sum_all_reduce` (:1929), `enable_fused_qk_norm_rope` (:1919), `num_continuous_decode_steps` (:956)
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`, `glm-5.2-optimization-log.md`
- `/home/aman/code/third_party/TileRT/README.md`, `tilert/models/glm_5/generator.py`, `tilert/models/glm_5/modules/end2end.py`, `tilert/models/glm_5/_dsa_v32/ops/`
- Our B200 via `cuInit` + `cuDeviceGetAttribute` (no context created): 148 SMs, 233,472 B smem/SM, 65,536 regs/SM, `COOPERATIVE_LAUNCH=1`, `CLUSTER_LAUNCH=1`, cc 10.0

**Consulted but not used as evidence:** *Dissecting the NVIDIA Blackwell
Architecture with Microbenchmarks* (arXiv 2507.10789) — the automated extraction
returned internally inconsistent figures (e.g. "DSMEM 128 MB per GPU", "L0 128 KB"
against a measured 233,472 B smem/SM on our device), so **no number from it is
cited here**. The paper may be fine; the extraction was not, and I did not verify
it by hand.

**Not sourced (explicitly):** a measured `grid.sync()` / cooperative-launch
barrier latency on SM100; a published L1i miss-rate measurement inside a
megakernel; any Blackwell end-to-end megakernel result from HazyResearch; Mosaic
GPU status for Blackwell LLM inference.
