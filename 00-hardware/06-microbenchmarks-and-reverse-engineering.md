# Reverse-engineered Blackwell: what microbenchmark papers have measured

## What this is

A survey of every published microbenchmark / reverse-engineering result I could find on NVIDIA
Blackwell (SM100 datacenter and SM120 consumer), cross-checked against Hopper baselines where
Blackwell data does not exist — and, because we have eight idle B200s, cross-checked against
measurements I took on **this box** while writing it. Provenance is labelled per claim:
`[verified]` = read in a primary source or measured here (path/URL given), `[reported]` = a vendor
or company asserts it, `[inferred]` = my reasoning from architecture, `[unverified]` = plausible,
unsourced. Several widely-cited numbers turn out to be wrong; those are called out explicitly,
because a wrong number that everyone repeats is worse than no number.

## Bottom line for our system

- **Our SM clock is pinned at 1597 MHz against a 1965 MHz ceiling — 81% of max — and nothing is
  throttling.** Under a full-SM FP32 load the box draws 380–420 W of a 1000 W limit at 36 °C with
  `clocks_event_reasons.active = 0x0`. `[verified, measured here]` Dense GEMM is 37.1% of our
  profile; if this lock is a benchmarking convenience rather than a requirement, there is up to
  ~23% of clock sitting unused on the single largest slice of our runtime. TileRT's ~500 tok/s
  may well have been measured at boost. **Check this before optimising anything else.**
- **Random-access HBM latency is ~840 cycles (~530 ns), and blows up to ~1163 cycles (~733 ns)
  once the working set passes somewhere between 2 GiB and 4 GiB** — a TLB reach limit.
  `[verified, measured here]` Our DSA sparse-MLA KV gather and our MoE expert gather are both
  random access over multi-GB regions. This is a real, unmodelled tax on 10.9% + 19.4% of the
  profile, and it is attackable (see §11).
- **L2 is 126.5 MB and delivers a measured 21.2 TB/s read** — 3.1× our measured HBM read of
  6.75 TB/s. `[verified, measured here]` Anything we can keep L2-resident across a decode step
  (per-layer weights at NVFP4, the index_topk_freq=4 indexer state) is worth ~3× on bandwidth.
- **Measured NVLink5 peer bandwidth is 771 GB/s read / 686 GB/s write**, 86% of the 900 GB/s
  unidirectional spec. `[verified, measured here]` Since 47% of our 19.6% collective time is
  *rank arrival skew* rather than transfer, the fabric is not our problem — scheduling is.
- **`tcgen05.mma` accumulates only into TMEM, and TMEM is a hard 512-column budget per SM.**
  The largest single-CTA accumulator (m128n256) eats 256 of 512 columns, so at most two such
  tiles are in flight per SM. `[verified: local CCCL headers + Colfax]` This is the real
  occupancy limiter for our NVFP4 MoE GEMMs, not registers.
- **MMA needs M=128 (1SM) or M=256 (2SM/CTA-pair) to reach full datapath utilisation; M=64 caps
  at ~50%.** `[reported: SemiAnalysis]` Worth auditing what M our
  `bmm_E2m1_E2m1E2m1_Fp32_swiGlu_dynB_sm100f` (6.0% of runtime) actually issues.
- **Shared-memory bank conflicts cost exactly 2.0 cycles per conflicting way on SM100**, so a
  32-way conflict is 66.6 cycles vs 6.45 conflict-free — 10.3×. `[verified, measured here]`
  Classic +1 padding (stride 33) fully restores throughput.
- **Treat the two arXiv "Blackwell microbenchmark" papers with suspicion.** Both contain
  numbers that are internally inconsistent or contradicted by the CUDA 13.3 toolkit on this
  machine (§2). The trustworthy public sources are Chips and Cheese, SemiAnalysis, Colfax, and
  the Hopper paper by Luo et al.

---

## 1. The literature: what actually exists

| # | Source | Silicon actually tested | Type | Trust |
|---|---|---|---|---|
| S1 | Jarmusch & Chandrasekaran, *Microbenchmarking NVIDIA's Blackwell Architecture*, arXiv:2512.02189v3 (Mar 2026) | **B200** (cloud) | academic | **Low** — see §2 |
| S2 | Jarmusch, Graddon & Chandrasekaran, *Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks*, arXiv:2507.10789v2 (Jul 2025) | **RTX 5080 (GB203, SM120)** + H100 PCIe | academic | **Low** — see §2 |
| S3 | Luo, Fan, Li, Du, Liu, Wang & Chu, *Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis*, arXiv:2501.12084 | H800 PCIe, A100 PCIe, RTX 4090 | academic | **High** — best Hopper baseline |
| S4 | Chips and Cheese, *Nvidia's B200: Keeping the CUDA Juggernaut Rolling* | **B200** (via Verda/DataCrunch) | industry | **High** — confirmed here |
| S5 | SemiAnalysis, *Dissecting Nvidia Blackwell — Tensor Cores, PTX Instructions, SASS, Floorsweep, Yield* (31 Mar 2026) | **B200 (SM100)** | industry | **High**, mostly paywalled |
| S6 | Colfax Research, *CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory for NVIDIA Blackwell GPUs* | SM100 (architectural) | vendor-adjacent | **High** for ISA semantics |
| S7 | The Software Frontier, *How Blackwell's Tensor Memory Actually Works* | SASS/PTX disassembly, no GPU | blog | **Medium-High** — reproducible from wheels |
| S8 | SemiAnalysis, *NVIDIA Tensor Core Evolution: From Volta To Blackwell* | architectural | industry | **Medium** — narrative, few numbers |
| S9 | This document | **our 8×B200**, driver 595.71.05, nvcc 13.3 | measured | numbers below |

The classic Citadel lineage (Jia et al. on Volta arXiv:1804.06826 and Turing T4 arXiv:1903.07486)
**has no Blackwell successor.** Nobody has published a Citadel-grade SM100 teardown — no SASS
control-word decode, no register-bank-conflict study, no instruction-cache hierarchy, no TLB
structure. The Blackwell papers that do exist are much thinner than the Volta/Turing work they
cite. **That gap is an opportunity for us, not just a nuisance.**

### Things I looked for and could not source

- Any published **register bank conflict** study for SM100. Not sourced.
- Any published **instruction cache (L0i/L1i) hierarchy** measurement for SM100. Not sourced.
- Any published **TLB structure** (entries, levels, page sizes) for SM100. Not sourced — I
  measured a reach limit myself (§11) but not the structure.
- Any published **`tcgen05` issue-to-issue throughput** table by shape from an independent party
  with reproducible code. S5 has the closest thing, paywalled and expressed as % of peak.
- **Measured NVLink5 latency-vs-message-size curves.** Only scattered nccl-tests anecdotes.

---

## 2. Credibility audit: why I do not trust the two arXiv Blackwell papers

This matters more than any single number, so it goes first.

### S2 (arXiv:2507.10789) is not about our silicon, and its headline numbers are broken

The paper's title says "Blackwell", but the GPU is a **GeForce RTX 5080 / GB203 / SM120**
`[verified: paper §III]` — a consumer part with GDDR7, 2 FP64 units per SM, no NVLink, and, by
the paper's own admission, *no `tcgen05` support at all*: "TCGEN05 IS YET TO BE SUPPORTED FOR THE
ARCH. SM120A" `[verified: paper Table IV caption]`. Nothing in it transfers to SM100.

Two numbers in it are impossible:

- "GH100 achieves a peak read bandwidth of **15.8 TB/s**" `[verified: paper §VI.D]`. An H100's
  HBM2e peak is ~3.35 TB/s. A 15.8 TB/s *global memory* read is off by ~5×; it is presumably an
  L2- or L1-resident measurement mislabelled as global memory.
- FP8 D-GEMM at 8192³ reaching "**0.887 TFLOP/s**" on H100 and "0.233 TFLOP/s" on RTX 5080
  `[verified: paper Table VII]`. H100 FP8 dense peak is ~1979 TFLOPS. The reported figure is
  ~2000× low — almost certainly a unit error (GFLOP/s vs TFLOP/s, or seconds vs milliseconds).
  The paper nonetheless draws the conclusion that "Hopper consistently outperforms Blackwell"
  on FP8 GEMM from it.

Do not cite this paper for anything.

### S1 (arXiv:2512.02189) is about a real B200 but contradicts itself and the toolkit

This one did run on a B200 `[verified: paper abstract, §VII.A — CUDA 12.6, driver 560.x]`, and
some of it is fine (the decompression-engine tables are plausible and are the only public DE
numbers I found). But:

**It contradicts the CUDA 13.3 toolkit sitting on this machine.** The paper's Table IV maps FP6
to `kind::mxf6`:

> "FP6 | kind::mxf6 | QMMA" `[verified: paper Table IV]`

There is no such kind. The complete set of `tcgen05.mma` kinds, extracted from the local headers:

```
$ grep -ohE "kind::[a-z0-9_]+" \
    /home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_mma.h \
  | sort -u
kind::f16
kind::f8f6f4
kind::i8
kind::mxf4
kind::mxf4nvf4
kind::mxf8f6f4
kind::tf32
```
`[verified: local file, CUDA 13.3]`. FP6 is reached through `kind::f8f6f4` / `kind::mxf8f6f4`,
not a dedicated FP6 kind. The paper invented an API name.

**It contradicts itself on throughput.** Table VI reports FP16 in/FP16 out at 964.8 TFLOPS and
FP16 in/FP32 out at 482.4 TFLOPS, and builds an argument that "FP32 accumulation halves
throughput… applications requiring high numerical precision must sacrifice 50% performance."
Table VII, two pages later, reports FP16 at **1929.6** TFLOPS `[verified: paper Tables VI, VII]`
— 2× and 4× the Table VI figures. Both cannot be right, and the "FP32 accumulate halves
throughput" claim is exactly the kind of thing a kernel author would act on. It is unsupported.

**Its wgmma baseline is suspiciously identical to S3's.** Table V lists Hopper `wgmma` SI-latency
as 32.0 / 64.0 / 128.0 cycles for n64 / n128 / n256. Luo et al. (S3) measured 32.0 / 64.0 / 128.0
for exactly those shapes on an H800 `[verified: S3 Table 9]`. Same numbers to the decimal. That is
either an uncredited reuse or a remarkable coincidence; either way, the "2.9–11.2× lower latency
than Hopper" headline is a comparison against someone else's H800 numbers, not a controlled
measurement.

**Its shapes are not legal.** Table V measures `tcgen05.mma m256n256k16` at "Warp" scope. M=256
requires `.cta_group::2` (a CTA pair spanning two SMs); it is not a single-CTA warp-scope shape
`[verified: Colfax S6 — single-CTA shapes are 64×N×16 and 128×N×16, N ≤ 256]`.

**What I would still take from S1:** the decompression-engine characterisation (Tables I–III),
which is self-consistent, has a sensible physical story (output-bandwidth-bound, input rate
scaling as 1/C with compression ratio), and is not contradicted by anything else. And the
observation that the DE exists at all.

### The lesson

The reliable public Blackwell numbers come from **Chips and Cheese** and **SemiAnalysis**, not
from academia. Below, I mark which is which, and where our own box agrees.

---

## 3. Memory hierarchy latency — measured on our B200

Method: single-thread dependent pointer chase over a **randomised** permutation cycle (a
sequential chain lets the L2 prefetcher help and understates HBM by ~80 cycles — I made that
mistake first, see §11), 128 B stride, timed loop = exactly one full lap so no residual caching.
Clock measured *in situ* at **1587 MHz** via a dependent-FMA chain against a CUDA event
(nvidia-smi reports 1597 MHz).

| Working set | cycles/access | ns @1587 MHz | Level |
|---|---:|---:|---|
| 128 KiB | 39.6 | 25.0 | L1 hit |
| 192 KiB | 39.5 | 24.9 | L1 hit (still) |
| 256 KiB | 216.3 | 136.3 | L1→L2 transition |
| 1 MiB | 305.1 | 192.3 | L2 hit |
| 8 MiB | 305.0 | 192.2 | L2 hit |
| 32 MiB | 305.0 | 192.2 | L2 hit |
| 64 MiB | 445.8 | 280.9 | exceeds one L2 partition (63.25 MB) |
| 128 MiB | 763.4 | 481.1 | ≈ whole L2 (126.5 MB) |
| 256 MiB | 832.6 | 524.6 | HBM |
| 512 MiB | 838.9 | 528.6 | HBM |
| 1 GiB | 841.2 | 530.1 | HBM |
| 2 GiB | 843.1 | 531.3 | HBM |
| **4 GiB** | **1163.5** | **733.1** | **HBM + TLB miss** |
| shared memory (any size) | 29.0 | 18.3 | SMEM |

`[verified, measured here: /tmp/.../scratchpad/rnd.cu, lat2.cu, smem.cu]`

### Cross-validation against Chips and Cheese

This is the useful part — two independent measurements on different B200s, different APIs.

| Quantity | Ours (CUDA, 1587 MHz) | Chips and Cheese (OpenCL/Vulkan) | Agreement |
|---|---|---|---|
| L1 hit | 39.5–39.6 cyc / 25.0 ns | "19.6 ns, or 39 cycles" `[verified: S4]` | **cycles match exactly** |
| L2 hit (near) | 305 cyc / 192 ns | "~150 ns" to attached partition `[verified: S4]` | ≈298 cyc at their clock |
| L2 read BW | 21.24 TB/s peak | "21 TB/s" same-partition `[verified: S4]` | **match** |
| L2 capacity | 126.5 MB `[verified: cudaDeviceProp]` | "126 MB" `[verified: S4]` | **match** |

The cycle counts agree to within measurement noise; the **nanoseconds do not, and that is the
story**. C&C's 39 cycles at 19.6 ns implies their B200 ran at ~1.99 GHz. Ours runs at 1.587 GHz.
Same silicon, same cache, **25% more wall-clock latency on our box purely from clock** (§10).

### Hopper baseline for comparison (S3, H800 PCIe)

| Level | H800 (S3) | B200 (ours) | Δ |
|---|---:|---:|---|
| L1 cache | 32.0 cyc `[verified: S3 Table 3]` | 39.6 cyc | +24% cycles |
| Shared memory | 29.0 cyc `[verified: S3 Table 3]` | 29.0 cyc | **identical** |
| L2 near hit | 258.0 cyc `[verified: S3 Table 4]` | ~279–293 cyc (§4) | +8–13% |
| L2 far hit | 414.1 cyc `[verified: S3 Table 4]` | not resolved | — |
| L2 near miss | 555.5 cyc `[verified: S3 Table 4]` | — | — |
| L2 far miss | 743.7 cyc `[verified: S3 Table 4]` | — | — |
| Global memory | 656 cyc `[verified: S3 Table 3]` | 833–843 cyc | +27–28% cycles |

Blackwell's memory hierarchy is **slower in cycles at every level except shared memory**. That is
the expected consequence of a much bigger L2 (126.5 vs 50 MB) spread across two dies. Shared
memory latency being bit-for-bit identical across two generations is a strong hint the SMEM
datapath is unchanged silicon `[inferred]`.

The claim circulating that Blackwell delivers "a 58% reduction in memory access latency in
cache-misses compared to H200" is **contradicted by our measurement** — we see global-memory
latency *up* 27% in cycles versus Hopper. I could not find a primary source for the 58% claim.
**Not sourced; treat as false.**

---

## 4. L2 geometry: near/far, dies, and SM pairs

Blackwell B200 is two reticle-sized dies joined by NV-HBI, presenting as one GPU
`[reported: NVIDIA Blackwell architecture brief, via S1 §III.A; corroborated S4]`. 148 SMs are
enabled (74 per die) `[verified: cudaDeviceProp.multiProcessorCount = 148; S4 reports 74/die]`.

I measured **uncontended per-SM L2 hit latency** for all 148 SMs: 148 blocks resident (forced one
per SM by a 100 KB shared-memory hog), only the block whose `%smid` matches the target performs
the timed chase, repeated 148 times.

| Statistic | 8 MiB buffer |
|---|---:|
| min | 279 cycles |
| p10 | 283 cycles |
| median | 293 cycles |
| p90 | 304 cycles |
| max | 324 cycles |
| spread | 16% |
| **SM 0–73 mean** | **287.8 cycles** |
| **SM 74–147 mean** | **293.9 cycles** |

`[verified, measured here: scratchpad/nf2.cu]`

Two findings:

1. **SM ids pair up exactly.** `0:282 1:281`, `2:292 3:292`, `4:286 5:286`, `48:298 49:298`,
   `104:309 105:309` — adjacent even/odd SM ids report the same latency throughout. This is the
   TPC pairing, and it is the same adjacency that `tcgen05.mma.cta_group::2` uses for CTA pairs
   `[verified, measured here; semantics per S6/S8]`. If we ever hand-place CTAs for 2SM MMA, SM
   id `n` and `n^1` are the pair.
2. **The die split is only ~6 cycles on average, not ~300.** With an 8 MiB buffer interleaved
   across all L2 slices on both dies, every SM averages near and far accesses, so this test
   cannot isolate the far-partition penalty — it bounds the *average* asymmetry at ~2%.

The "**~300 cycle die-to-die penalty**" figure `[reported: S5, restated in jianyuh.github.io]`
therefore does **not** describe average L2 hit latency. It must refer to a specific
cross-die path (most plausibly a remote-SMEM or cross-die L2 *miss*). Our measurement neither
confirms nor refutes it. **Open question — see §13.**

For reference, Hopper's near/far L2 gap is large and well-measured: 258.0 vs 414.1 cycles, a
**+156 cycle far-partition penalty** `[verified: S3 Table 4]`. Blackwell almost certainly has an
analogous structure; we just have not isolated it.

### Capacity facts from `cudaDeviceProp` on this box

| Property | Value |
|---|---|
| `multiProcessorCount` | 148 |
| `l2CacheSize` | 132,644,864 B = **126.5 MiB** |
| `persistingL2CacheMaxSize` | 82,903,040 B (79 MiB) |
| `accessPolicyMaxWindowSize` | 134,217,728 B (128 MiB) |
| `sharedMemPerMultiprocessor` | 233,472 B (228 KiB) |
| `sharedMemPerBlockOptin` | 232,448 B (227 KiB) |
| `sharedMemPerBlock` (default) | 49,152 B (48 KiB) |
| `reservedSharedMemPerBlock` | 1,024 B |
| `regsPerMultiprocessor` | 65,536 |
| `maxThreadsPerMultiProcessor` | 2,048 |
| `maxBlocksPerMultiProcessor` | 32 |
| `memoryBusWidth` | **7,680 bits** |
| `totalGlobalMem` | 191,495,471,104 B (178.3 GiB) |
| `asyncEngineCount` | 4 |
| `clusterLaunch` | 1 |

`[verified, measured here: scratchpad/dq.cu]`

**`persistingL2CacheMaxSize` = 79 MiB is worth remembering.** We can pin up to 79 MB of L2 with
`cudaAccessPolicyWindow` — enough for a large slice of an NVFP4 expert's weights, or the whole
DSA indexer working set.

### The L1 carveout is bigger than the literature says

Our latency stays flat at 39.5 cycles out to **192–208 KiB**, breaking at 224 KiB. With no shared
memory requested, the unified L1/SMEM array (228 KiB usable) is nearly all available as L1 data
cache. S2 claims "L1 cache Size 256 (unified)" for GH100 and "128 (unified)" for GB203
`[verified: S2 Table II]` — the 128 KB figure is a consumer-SM120 number and does **not** apply to
SM100. C&C's "256 KB, SM-private, shared with Shared Memory" `[verified: S4]` matches what we see.

---

## 5. Bandwidth at each level vs. spec sheet

All measured here with grid-stride `float4` kernels, 148×8 to 148×16 blocks × 256 threads.

| Level | Measured | Notes |
|---|---:|---|
| L2 read, 8 MiB buffer | 8,738 GB/s | too small to fill the machine |
| L2 read, 32 MiB | 17,638 GB/s | |
| **L2 read, 64 MiB** | **21,237 GB/s** | peak; 13,300 B/clk @1587 MHz |
| L2 read, 96 MiB | 18,990 GB/s | approaching capacity |
| L2 read, 110 MiB | 17,506 GB/s | capacity misses |
| HBM read, 512 MiB | 6,415 GB/s | |
| HBM read, 2 GiB | 6,633 GB/s | |
| **HBM read, 4 GiB** | **6,753 GB/s** | **88% of theoretical** |
| HBM copy (r+w), 4 GiB | 6,059 GB/s | counted as 2× bytes |
| NVLink5 peer read (0→1) | 771 GB/s | 86% of 900 GB/s unidir |
| NVLink5 peer write (0→1) | 686 GB/s | |
| `cudaMemcpyPeer` (0→1) | 771 GB/s | copy-engine path, same as SM path |

`[verified, measured here: scratchpad/mb.cu, smem.cu, p2p.cu]`

### The HBM bandwidth number everyone quotes is wrong for this part

`cudaDeviceProp.memoryBusWidth` on our B200 is **7,680 bits**, and `nvidia-smi` reports a memory
clock of **3,996 MHz** `[verified, measured here]`. HBM3e is DDR, so:

```
7680 bits × 2 × 3.996 GHz ÷ 8 = 7,672 GB/s theoretical
```

Not 8 TB/s. Our measured 6,753 GB/s read is **88.0% of 7.67 TB/s**, which is a healthy efficiency
— but only 84% of the marketing figure. Any roofline we build against "8 TB/s" is ~4% optimistic
before we start.

Chips and Cheese quote **9.2 TB/s** for B200 HBM3e `[verified: S4]`, and describe it as the rated
specification rather than something they measured. I cannot reconcile 9.2 TB/s with a 7,680-bit
bus at 8 Gbps/pin on this hardware, and I could not source it to an NVIDIA document.
**Treat 9.2 TB/s as unsourced.** Note also that our own L2-resident read of 21.2 TB/s and
64 MiB-buffer figure of 9.59 TB/s show how easy it is to produce a >8 TB/s "memory bandwidth"
number by accident — a 64 MiB buffer is entirely L2-resident on this chip.

For Hopper context, S3 measured H800 global memory at 1,861.5 GB/s, **91% of theoretical**
`[verified: S3 Table 5]`, and L2 at 4,472 B/clk. Blackwell's L2 at ~13,300 B/clk is a **3.0×
increase in bytes per clock** over Hopper's L2 `[inferred from S3 Table 5 + our measurement]` —
the single largest generational jump in the hierarchy.

---

## 6. Shared memory: banks and conflicts

32 banks × 4 B, unchanged. Measured with one warp, lane *i* reading `s[i*stride]`, 20,000
iterations, `__syncwarp()` per iteration.

| Stride (u32) | Conflict degree | cycles/iter | vs conflict-free |
|---:|---:|---:|---:|
| 1 | 1-way | 6.45 | 1.00× |
| 2 | 2-way | 6.63 | 1.03× |
| 4 | 4-way | 10.57 | 1.64× |
| 8 | 8-way | 18.57 | 2.88× |
| 16 | 16-way | 34.57 | 5.36× |
| 32 | 32-way | 66.57 | 10.32× |
| **33** | 1-way (padded) | **6.45** | **1.00×** |

`[verified, measured here: scratchpad/smem.cu]`

The degrees ≥4 fit a clean linear model:

```
cycles = 2.00 × (conflict degree) + 2.57       (R² ≈ 1.000 for degree 4,8,16,32)
```

**Serialisation costs exactly 2.0 cycles per conflicting way.** Degrees 1 and 2 are floored at
6.45 cycles by the loop's issue rate, which is why the 2-way case looks free — it is not, it is
just hidden. The classic `+1` padding (stride 33) fully restores conflict-free throughput,
confirming 32 banks of 4 bytes with no Blackwell-specific swizzle in the plain-load path.

Practical: 128 B per conflict-free warp access. S3 measured Hopper shared-memory throughput at
127.4–127.9 B/clk/SM, i.e. 99.8% of the 128 B/clk theoretical `[verified: S3 Table 5]`, and our
identical 29-cycle SMEM latency suggests the same datapath.

### Distributed shared memory (DSMEM) — Hopper numbers only

No published SM100 DSMEM measurements exist. Hopper baseline from S3:

| Quantity | H800 | Source |
|---|---:|---|
| Local SMEM load | 29 cycles | `[verified: S3 Table 3]` |
| Own SMEM *via* the DSM interface | 33 cycles | `[verified: S3 §7.1]` — +4 cycle interface tax |
| Remote SMEM, cluster size 2 | 181 cycles | `[verified: S3 §7.1]` |
| Remote SMEM, cluster size 4–16 | 184–213 cycles | `[verified: S3 §7.1]` |
| Round-trip via global memory | 1,110 cycles | `[verified: S3 §7.1]` |
| DSM speedup vs global round-trip | 6.13× | `[verified: S3 §7.1]` (NVIDIA claims 7×) |
| Peak SM-to-SM throughput | 3.28 TB/s (cluster 2) | `[verified: S3 §7.2]` |
| … cluster 4 | 2.78 TB/s | `[verified: S3 §7.2]` |
| Own SMEM via DSM interface | 205 GB/s (80% of 225 GB/s peak) | `[verified: S3 §7.2]` |

Note the pattern S3 found: **larger clusters reduce DSM throughput** (3.28 → 2.78 TB/s going from
cluster 2 to 4) because blocks contend for the SM-to-SM network, and broadcast patterns degrade
much worse than ring or pair patterns `[verified: S3 §7.2]`. If we use clusters for the MoE
dispatch, cluster 2 is likely the sweet spot. **Unverified on Blackwell.**

For SM100, S5 reports remote shared-memory load bandwidth of **~21 B/clk naive** vs **~32 B/clk
using `cp.async.bulk`** against ~128 B/clk local `[reported: S5 via jianyuh.github.io]` — i.e.
remote SMEM is a 4–6× bandwidth penalty and you must use the bulk path.

---

## 7. Atomics

Global `atomicAdd` on u32, 592 blocks × 256 threads, 200 iterations each.

| Address distribution | G atomics/s | atomics/clk @1587 MHz |
|---|---:|---:|
| all threads → same address | 39.2 | 24.7 |
| spread over 1,024 addresses | 95.2 | 60.0 |
| **spread over 1 M addresses** | **565.2** | **356** |
| spread over 16 M addresses | 544.7 | 343 |

`[verified, measured here: scratchpad/smem.cu]`

Chips and Cheese report "just short of **512** operations per cycle" GPU-wide for global
`atomic_add` `[verified: S4]`. We measure 356/clk. The gap is plausibly methodology (OpenCL vs
CUDA, different address spread and contention) — 1 M distinct addresses = 4 MB is L2-resident for
us, 16 M = 64 MB starts to stress L2. **Discrepancy noted, not resolved.**

The 14× penalty from full contention (39 G/s) to well-spread (565 G/s) is the actionable number.
C&C also measured shared-memory `atomic_cmpxchg` at ~39 ns and **32 atomic ops/cycle per SM**
`[verified: S4]`.

---

## 8. Instruction latency and throughput

### Scalar FMA

Dependent `fmaf` chain, one warp: **4.44 cycles/FMA** `[verified, measured here: scratchpad/lat2.cu]`.
The residual over 4.0 is loop overhead, so **FP32 FMA dependent latency is 4 cycles** — unchanged
from Volta through Blackwell `[inferred]`. S2 measured 4 cycles true latency for both pure INT32
and pure FP32 on GB203 and H100 `[verified: S2 Table III]`, which is the one result in that paper
I am willing to believe because it agrees with ours and with a decade of prior art.

### Hopper `wgmma` — the solid baseline (S3, H800)

This is the best-measured MMA table in the literature and the right yardstick.

| Shape | LAT (SS) | LAT (RS) | TFLOPS (SS, zeros) | TFLOPS (SS, random) |
|---|---:|---:|---:|---:|
| FP16→FP16 m64n256k16 | 128.0 | 128.0 | 729.3 | 704.5 |
| FP16→FP32 m64n256k16 | 128.0 | 128.0 | 728.5 | 665.4 |
| TF32→FP32 m64n256k8 | 128.0 | 128.0 | 364.4 | 357.1 |
| FP8→FP16 m64n256k32 | 128.0 | 128.0 | 1448.4 | 1439.2 |
| FP8→FP32 m64n256k32 | 128.0 | 128.0 | 1447.5 | 1417.2 |
| INT8→INT32 m64n256k32 | 128.0 | 128.0 | 1448.7 | 1442.3 |

`[verified: S3 Table 8]`

Latency scales linearly with N `[verified: S3 Table 9]`:

| N | LAT dense (SS) | LAT dense (RS) | Tput (SS, rand) | Tput (RS, rand) |
|---:|---:|---:|---:|---:|
| 256 | 128.0 | 128.0 | 665.4 | 667.5 |
| 128 | 64.0 | 64.0 | 659.8 | 661.7 |
| 64 | 32.0 | 32.0 | 648.3 | 649.9 |
| 32 | 24.0 | 16.0 | 471.5 | 634.4 |
| 16 | 20.0 | 13.0 | 283.5 | 426.2 |
| 8 | 18.0 | 13.0 | 157.6 | 215.2 |

**S3's rule: use N ≥ 64 with `wgmma`, below which SMEM latency cannot be hidden**
`[verified: S3 §6.2]`. Note also that FP16→FP32 accumulate costs essentially nothing versus
FP16→FP16 on Hopper (728.5 vs 729.3 TFLOPS) — which is more evidence against S1's "FP32
accumulation halves throughput" claim for Blackwell.

Also from S3: `mma.sync` (the portable warp-level path) reaches only **62.9% of peak on Hopper on
average**, versus >95% for `wgmma` `[verified: S3 §6.2]`. Since `mma.sync` is the *only*
instruction portable across Hopper / SM100 / SM120 `[verified: S7]`, portability costs ~35%.

### Blackwell `tcgen05.mma` — what is actually known

| Claim | Value | Provenance |
|---|---|---|
| Single-instruction latency, flat across shapes | 11.0–11.4 cycles | `[reported: S1 Table V]` — **suspect, see §2** |
| Latency by precision | 11.2 (FP16) → 12.6 (FP4) cycles | `[reported: S1 Table VI]` — suspect |
| 1SM MMA, M=64 | ~50% of peak datapath | `[reported: S5]` |
| **1SM MMA, M=128** | **near 100%** | `[reported: S5]` |
| 2SM MMA, M=128, N=64 | ~90% of peak | `[reported: S5]` |
| **2SM MMA, M=256** | **near 100%** | `[reported: S5]` |
| In-flight MMAs needed for SoL | 256–1024 | `[reported: S5]` |
| Typical kernels have in flight | 1–4 → caps at 78–80% | `[reported: S5]` |
| Latency ordering by dtype | `S8 < BF16 = E4M3 = F4 < MXF8 = MXF4` | `[reported: S5]` |

The **M=128 (1SM) / M=256 (2SM)** rule is the single most actionable published Blackwell tensor-core
result, and the dtype ordering says **block-scaled MX formats are slower than plain FP4/FP8** —
relevant to us because we run NVFP4 with block-16 scaling.

The "flat 11 cycles regardless of tile size" claim would be genuinely interesting if true (it
implies a spatial rather than temporally-pipelined array), but given §2 I would not build on it
without re-measuring.

---

## 9. TMEM and `tcgen05` — ground truth from the local toolkit

Everything here is `[verified]` from files on this machine
(`/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/`) unless
marked otherwise.

### Geometry

- **256 KB per SM**, organised as **128 lanes × 512 columns of 32-bit cells**
  `[verified: S6; corroborated S1 §V.A, S7, S8]`.
- Address is a 32-bit word: **bits 31–16 = lane, bits 15–0 = column** `[verified: S6]`.
- Allocation is **by column** (a column spans all 128 lanes), must be a **power of two, minimum
  32 columns** `[verified: S6]`.
- `tcgen05.alloc` / `tcgen05.dealloc` must be issued **by a single warp** `[verified: S6]`.
- **Each warp of a warpgroup can only reach 32 lanes** — warp 0 → lanes 0–31, warp 1 → 32–63,
  etc. One warp cannot read a full 128-lane accumulator `[verified: S6]`.

### Instruction inventory (from local headers)

`tcgen05.ld` / `tcgen05.st` shapes:
```
tcgen05.ld.sync.aligned.32x32b
tcgen05.ld.sync.aligned.16x64b
tcgen05.ld.sync.aligned.16x128b
tcgen05.ld.sync.aligned.16x256b
tcgen05.ld.sync.aligned.16x32bx2
```

`tcgen05.mma` kinds: `f16, tf32, f8f6f4, mxf8f6f4, mxf4, mxf4nvf4, i8`
Weight-stationary (`tcgen05.mma_ws`) kinds: `f16, tf32, f8f6f4, i8` (no MX formats)
Scale-vector modifiers: `scale_vec::1X`, `scale_vec::2X`, `scale_vec::4X`
CTA groups: `cta_group::1` (549 occurrences), `cta_group::2` (229)

`tcgen05.cp` shapes — note the built-in sub-byte decompression:
```
tcgen05.cp.cta_group.128x256b
tcgen05.cp.cta_group.128x128b
tcgen05.cp.cta_group.32x128b.warpx4
tcgen05.cp.cta_group.4x256b
  ... each also available as .b8x16.b4x16_p64   (FP4  -> 8-bit on the fly)
  ... and                  .b8x16.b6x16_p32   (FP6  -> 8-bit on the fly)
```

**`kind::mxf4nvf4` is NVFP4 (block-16, E4M3 scales) and takes `scale_vec::4X`; `kind::mxf4` is
MXFP4 (block-32, E8M0 scales) and takes `scale_vec::2X`** `[verified: header names +
S7 for the block sizes]`. There is **no FP64 kind** — B200 FP64 goes through a separate DMMA path
`[verified: header inventory; corroborated S1 Table IV footnote]`.

The SASS opcode string `UTCHMMA` is present in the local `ptxas` binary
(`strings /home/aman/code/cuda-13.3/nvidia/cu13/bin/ptxas | grep UTC`) `[verified, measured here]`.
S7 reports the full family as **`UTCHMMA`** (dense), **`UTCHMMA.2CTA`** (CTA pair),
**`UTCOMMA`** (block-scaled), and **`UTCATOMSWS.FIND_AND_SET.ALIGN`** for TMEM allocation
`[reported: S7]`. **This contradicts S1 Table IV**, which claims `tcgen05.mma` compiles to plain
`HMMA`/`QMMA`/`OMMA` — the Hopper-era opcodes. S7 is consistent with what is in our ptxas; S1 is
not.

### Why TMEM exists, in one number

The largest single-CTA accumulator (m128n256, FP32) holds **32,768 values**. Spread over a
128-thread warpgroup that is **256 registers per thread**, against an architectural ceiling of
255 `[verified: S7]`. The largest Blackwell MMA literally cannot fit in the register file. The
register file has been 65,536 registers per SM since Volta while tensor throughput per SM per
clock roughly doubled four times; register bytes per FP16 FLOP/clk fell 256 → 128 → 64 → 32 across
Volta → Ampere → Hopper → Blackwell `[reported: S7]`.

### The occupancy constraint that matters for us

512 columns per SM, shared by **all resident CTAs**. The largest UMMA accumulator takes **256
columns**. So:

| Allocation | Concurrent allocations per SM |
|---:|---:|
| 32 columns | 16 |
| 128 columns | 4 |
| 256 columns (m128n256) | **2** |
| 512 columns | 1 |

`[verified: S6, S7]`

And block scaling **competes for the same pool** — NVFP4's block-16 E4M3 scale factors occupy
dedicated TMEM columns alongside the accumulator, and cost **twice** the footprint of MXFP4's
block-32 E8M0 scales `[reported: S7]`. For our NVFP4 MoE GEMMs this is a direct, quantifiable
occupancy tax that has nothing to do with registers or shared memory.

S7 also notes the allocator is genuinely contended: the failure path emits `NANOSLEEP 0x64` and
an indefinite spin-retry loop, and a trivial single-MMA-plus-drain kernel compiles to **152 SASS
instructions at -O3, of which 2 are the actual MMA** `[reported: S7]`. TMEM bookkeeping is not
free.

### TMEM claims I do not believe

S1 states TMEM has "**16 TB/s read bandwidth**" and that keeping intermediates resident avoids
"≈12 TB/s of data movement per SM" `[verified: S1 §V.A]`. Neither is dimensionally coherent as
stated (12 TB/s *per SM* × 148 SMs would be 1.8 PB/s), and no method is given. S7 derives a
different figure from first principles — accumulator traffic of ~562.5 TB/s device-wide, ~3.8
TB/s per SM `[reported: S7]` — which at least has an arithmetic derivation behind it
(`4 bytes / K × peak math throughput`). **Neither is measured. Open question.**

---

## 10. Power and clock behaviour

### Our box is clock-locked, and that is the biggest single finding here

| Observation | Value |
|---|---|
| Max SM clock | 1,965 MHz |
| Max customer boost | 1,965 MHz |
| **Actual SM clock, idle** | **1,597 MHz** |
| **Actual SM clock, full-SM FP32 FMA load** | **1,597 MHz** |
| `clock64` rate measured in-kernel | 1,587 MHz |
| Power, idle | ~249 W |
| Power, full-SM FP32 FMA load | **380–420 W** |
| Enforced power limit | 1,000 W |
| Temperature under load | 35–36 °C |
| `clocks_event_reasons.active` under load | **0x0000000000000000** |
| Performance state | P0 |
| `SW Power Capping` counter (cumulative) | 90,936,966 µs |

`[verified, measured here: nvidia-smi + scratchpad/load.cu, sustained ~30 s]`

Nothing is throttling. Temperature is 36 °C against a 1000 W budget with 400 W drawn. The clock
simply never moves off 1597 MHz, on any GPU, at any load. `nvidia-smi -q -d SUPPORTED_CLOCKS`
lists the full ladder up to 1965 MHz, so the silicon is capable.

Corroboration from our own benchmark corpus: `/home/aman/code/benchmark/RESULTS.md` records
`SM clocks | 1597 MHz` in its provenance table, and its identity hash "covers … SM clock"
`[verified: local file]`. So this is a **deliberate clock lock for run-to-run reproducibility**,
not a fault. That is good methodology — but it means:

- Every latency figure in this document is ~24% worse in nanoseconds than the same silicon at boost.
- Our 365 tok/s single-stream and 40.8k tok/s aggregate were measured at **81% of max clock**.
- Chips and Cheese's B200 ran at ~1.99 GHz `[inferred: their 39 cycles = 19.6 ns]`.
- If TileRT's ~500 tok/s was measured on an unlocked box, **part of the gap we are chasing is a
  clock setting, not a kernel.** 500/365 = 1.37×; 1965/1597 = 1.23×.

This needs to be checked before any further optimisation work, and it is a one-command
experiment (§13).

### Sustained tensor-core load: only Hopper data exists

S3 measured the Hopper case carefully and the result is a warning:

- Running `wgmma` continuously, H800-PCIe core frequency **drops below the 1620 MHz whitepaper
  figure** as it hits the 350 W power limit `[verified: S3 §6.2]`.
- **Input data matters enormously.** Zero-filled matrices keep power under 200 W; random inputs
  hit the 350 W cap immediately. FP16→FP32 `wgmma` throughput falls from 728.5 TFLOPS (zeros) to
  665.4 TFLOPS (random) — a **9% real-data penalty purely from power** `[verified: S3 Table 8]`.
- Sparse `wgmma` causes "a substantial reduction in frequency", which is why sparse never reaches
  2× dense `[verified: S3 §6.2]`.
- `wgmma` achieves only **0.67×** the energy efficiency of `mma` `[verified: S3 §6.2]`. Peak
  performance and peak efficiency are different operating points.

**Implication for us: any tensor-core microbenchmark run on zero-filled matrices overstates real
throughput.** Our 8.2%-of-runtime allreduce and 37.1% dense GEMM run on real data. If we
benchmark kernels with zeros we will mispredict.

Blackwell equivalents of all of this: **not sourced.** Nobody has published a B200 frequency-vs-
sustained-tcgen05 curve. Our box cannot even produce one while clock-locked — which is another
reason to unlock it for a controlled experiment.

### Precision vs power, consumer Blackwell only

S2 measured, on RTX 5080/GB203 `mma` (not `tcgen05`) `[verified: S2 Table VI]`:
FP4 e2m1 **16.75 W**, FP6 e2m3 39.38 W, FP6 e3m2 46.72 W, FP8 e4m3 46.66 W, FP8 e5m2 46.81 W;
H100 FP8 ~55.8 W. Direction (lower precision → lower power) is believable; magnitudes on a
consumer part tell us nothing about a 1000 W B200.

---

## 11. TLB and page behaviour

**This is the one place where I found something the literature does not have.**

No published source gives SM100 TLB structure. My first attempt to probe it failed instructively,
and the failure is worth recording so nobody repeats it:

- **Failed method 1 — sequential pointer chain.** A chain built as `i → i+1` walks memory in
  address order. Prefetchers and page-walkers love it. It reported HBM latency as ~763 cycles and
  showed *zero* stride sensitivity out to a 4 GiB footprint. Both were artifacts.
- **Failed method 2 — per-access `clock64` bracketing** (the "fine-grained P-chase" of S3). On
  SM100 the `clock64` read overhead swamps the signal: I got an identical bimodal 500/870-cycle
  distribution for 8 MiB, 64 MiB and 200 MiB working sets. It was measuring the timer, not the
  memory.
- **Failed method 3 — stride sweep at fixed region size.** Varying stride while holding the
  region at 4 GiB keeps the number of *unique cache lines* small, so everything stayed an L2 hit
  (flat 303 cycles for strides 128 B → 1 MiB). Cache residency, not TLB.

The method that worked: **randomised permutation cycle covering every node exactly once**, timed
loop equal to one full lap.

| Working set | cycles/access | ns |
|---:|---:|---:|
| 256 MiB | 832.6 | 524.6 |
| 512 MiB | 838.9 | 528.6 |
| 1 GiB | 841.2 | 530.1 |
| 2 GiB | 843.1 | 531.3 |
| **4 GiB** | **1163.5** | **733.1** |

`[verified, measured here: scratchpad/rnd.cu]`

Flat at ~840 cycles from 256 MiB through 2 GiB, then a **+320 cycle (+200 ns) step** at 4 GiB.
That step is a page-translation cost, not a cache effect — the data is already fully
uncacheable at 256 MiB.

**Conclusion: GPU TLB reach on B200 is somewhere between 2 GiB and 4 GiB, and exceeding it costs
roughly +320 cycles per access — a 38% latency increase on top of an already-expensive HBM
miss.** `[verified, measured here]` If the backing pages are 2 MiB, a 2 GiB reach implies ~1024
entries `[inferred]`. I did not localise the knee (a 12 GiB sweep timed out building the chain
host-side) or determine the page size, so treat 2–4 GiB as a bracket, not a number.

**Why this matters for us specifically.** Our resident footprint is ~165 GB per GPU. Two of our
hot paths are random gathers over regions far larger than 4 GiB:
- **MoE expert weight gather** (256 experts, 8 active) — 19.4% of runtime.
- **DSA sparse MLA KV gather** with `index_topk_freq=4` — 10.9% + 5.8% indexer.

Both should be paying this tax, and neither the roofline nor any profiler counter we currently
read would show it as anything but "memory latency". Mitigations worth testing: larger backing
pages, `cudaMemAdvise` / pool configuration, and reordering gathers to increase page locality
(sorting expert-token assignments by expert already helps bandwidth; it should help TLB more).

---

## 12. Decompression Engine — the one genuinely new B200 unit

B200 has a hardware decompression engine with no H100 equivalent `[reported: NVIDIA via S1 §V.B]`.
S1's characterisation is the only public measurement I found. 100 MB datasets, 64 KB chunks,
nvCOMP with hardware DE enabled:

| Format | Compression ratio | Input GB/s | Output GB/s | Latency (ms) |
|---|---:|---:|---:|---:|
| LZ4 | 1.00× | 173.2 | 172.6 | 0.608 |
| Snappy | 1.91× | 61.4 | 117.2 | 0.894 |
| Zstd | 2.00× | 77.5 | 154.9 | 0.677 |
| GZIP | 2.00× | 42.0 | 83.8 | 1.251 |
| Cascaded | — | — | 213.4 | 0.491 |
| Bitcomp | 3.00× | 154.0 | **462.4** | **0.227** |
| ANS | — | — | **539.2** | **0.194** |

`[reported: S1 Table I]`

Key structural finding, which is self-consistent and I believe: **the DE is output-bandwidth
bound, not compute bound.** Decompressed output stays in a 160–220 GB/s band regardless of data
entropy, while compressed *input* rate falls as 1/C with the compression ratio (173 GB/s at 1.00×
down to 0.85 GB/s at 245×) `[verified: S1 Table II and §V.B]`.

Chunk-size scaling `[reported: S1 Table III]`: peak throughput 55.8 GB/s (32 KB chunks) → 112.1
GB/s (256 KB chunks); optimal concurrency depth 1–2 for 32–64 KB chunks, 8 for 128–256 KB.

**Honest assessment for us: 170–220 GB/s of decompressed output is ~3% of our 6.75 TB/s HBM read
bandwidth.** The DE is interesting for dataset loading, not for the inference hot path. It is not
a lever on our profile. Note separately that `tcgen05.cp`'s `.b4x16_p64` / `.b6x16_p32` modifiers
are a *different* mechanism — sub-byte-to-8-bit expansion inside the SMEM→TMEM copy — and that
one **is** on our hot path for NVFP4.

---

## 13. What we should microbenchmark ourselves

We have eight idle B200s and the published literature is thin and partly wrong. Ordered by
expected value to our system.

1. **Settle the clock lock.** Highest value, lowest effort. Record `nvidia-smi -q -d CLOCK`,
   then with explicit approval try `nvidia-smi -rgc` (reset locked clocks) on one GPU, rerun the
   FMA load, and see whether it boosts to 1965 MHz and what power it draws. Then rerun the
   single-stream GLM-5.2 benchmark unlocked. **If we get 1.23× clock we should see a large
   fraction of that on the 37.1% dense-GEMM slice.** This is a settings change on a shared box,
   so it needs a human decision — but the *measurement* of what the box would do is cheap.

2. **Localise the TLB knee and find the page size.** Extend §11's randomised chase to a
   2/2.5/3/3.5/4 GiB sweep (build the chain on-device to avoid the host-memory limit that timed
   out my run). Then test whether `CU_MEM_ALLOCATION_TYPE_PINNED` with large granularity, or
   pool configuration, moves the knee. **Directly attacks 30% of our runtime.**

3. **Measure the real MoE and DSA gather latency distribution**, not the synthetic one. Instrument
   `bmm_E2m1_E2m1E2m1_Fp32_swiGlu_dynB_sm100f` and the DSA indexer with in-kernel `clock64`
   deltas around the gather, histogrammed. Compare against the §11 curve to see how much of the
   gap is TLB.

4. **Isolate the near/far L2 and die-to-die penalty.** Method: allocate a small buffer, use
   `cudaAccessPolicyWindow` or physical-address probing to confine it, and run the §4 one-SM-at-a-time
   chase. Goal is the Blackwell analogue of S3 Table 4 (near hit / far hit / near miss / far
   miss). **Settles whether SemiAnalysis's "~300 cycle die-to-die penalty" applies to L2 at all**,
   and tells us whether CTA placement across the die boundary matters for our allreduce.

5. **`tcgen05.mma` issue-to-issue throughput by shape, on real data.** Build the table S1 should
   have: for each of `kind::f16`, `kind::f8f6f4`, `kind::mxf4nvf4` (our format), sweep
   M ∈ {64,128} × N ∈ {8..256} × `cta_group::{1,2}`, with (a) an accumulator-carried dependency
   for latency and (b) 1/2/4/8/256 independent MMAs in flight for throughput. **Measure with
   random inputs, not zeros** — S3 showed zeros understate power by 40% and overstate throughput
   by 9% on Hopper. Verify S5's M=128/M=256 rule on our silicon and find where our kernels sit.

6. **TMEM allocation contention.** Instrument the `UTCATOMSWS.FIND_AND_SET.ALIGN` retry path: how
   often do our MoE kernels spin? Sweep column allocation size (32/64/128/256) against achieved
   occupancy and TFLOPS. **This is the occupancy limiter S7 identifies and nobody has measured.**

7. **NVFP4 vs MXFP4 scale-factor cost.** S7 says NVFP4's block-16 scales take 2× the TMEM
   columns of MXFP4's block-32. We run NVFP4. Measure the actual throughput delta on an identical
   GEMM — if it is large, it is an argument for MXFP4 on some layers.

8. **NVLink5 latency-vs-size curve.** We have the fabric; nobody has published the curve. Sweep
   `cudaMemcpyPeer` and SM-driven remote loads from 8 B to 1 GB, and separately measure the
   *arrival skew* that is 47% of our collective time — i.e. per-rank timestamps at the entry of
   `oneshotAllreduceFusionKernel`. **The skew, not the fabric, is our 9.2%-of-runtime problem**,
   and it is entirely unmeasured in the literature because it is a scheduling property, not a
   hardware one.

9. **Register bank conflicts on SM100.** Completely unstudied. Volta-era method: dependent
   `FFMA Rd, Ra, Rb, Rc` sequences with operands deliberately placed in the same bank, timed.
   Would tell us whether our hand-written kernels are losing issue slots.

10. **A proper DSMEM/cluster study on SM100.** Reproduce S3 §7 (ring / pair / broadcast patterns ×
    cluster size 2/4/8/16 × the three scheduling policies) on Blackwell. Relevant if we want to
    use clusters for MoE dispatch.

11. **Sustained tensor-core frequency curve.** Once unlocked: run `tcgen05` continuously with
    random inputs at each precision and log frequency and power at 10 ms resolution. Produces the
    Blackwell version of S3 Table 11, which does not exist publicly.

### Methodological warnings, learned the hard way here

- **Randomise the pointer chain.** A sequential chain understated HBM latency by ~80 cycles and
  hid the TLB effect entirely.
- **Do not bracket individual accesses with `clock64`** on SM100 — overhead dominates. Time a
  full lap.
- **Guard against dead-code elimination.** Three of my kernels silently reported 0 cycles because
  the chase result fed only a dead branch. Always write the final pointer to a global sink
  unconditionally, and sanity-check that the result is not zero.
- **Sweep the *unique cache line* count, not the region size.** A 4 GiB region touched at 1 MiB
  stride is an L2 benchmark, not a DRAM benchmark.
- **Watch for L2-resident buffers masquerading as HBM.** With 126.5 MB of L2, any buffer under
  ~110 MB gives you a "memory bandwidth" number 3× too high. This is almost certainly the origin
  of at least one published >8 TB/s B200 figure.

---

## Sources

Read in full:

- Aaron Jarmusch, Sunita Chandrasekaran, *Microbenchmarking NVIDIA's Blackwell Architecture: An
  in-depth Architectural Analysis*, arXiv:2512.02189v3 (2 Mar 2026).
  https://arxiv.org/abs/2512.02189 · PDF: https://arxiv.org/pdf/2512.02189
- Aaron Jarmusch, Nathan Graddon, Sunita Chandrasekaran, *Dissecting the NVIDIA Blackwell
  Architecture with Microbenchmarks*, arXiv:2507.10789v2 (21 Jul 2025).
  https://arxiv.org/abs/2507.10789 · PDF: https://arxiv.org/pdf/2507.10789
- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Hongyuan Liu, Qiang Wang, Xiaowen Chu, *Dissecting the
  NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis*,
  arXiv:2501.12084. https://arxiv.org/abs/2501.12084 · PDF: https://arxiv.org/pdf/2501.12084
  (code: https://github.com/HPMLL/NVIDIA-Hopper-Benchmark)
- Chips and Cheese, *Nvidia's B200: Keeping the CUDA Juggernaut Rolling ft. Verda (formerly
  DataCrunch)*. https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut
- Colfax Research, *CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA
  Blackwell GPUs*.
  https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
- The Software Frontier, *How Blackwell's Tensor Memory Actually Works*.
  https://www.thesoftwarefrontier.com/p/how-blackwells-tensor-memory-actually

Read in part (paywalled or summary only):

- SemiAnalysis, *Dissecting Nvidia Blackwell — Tensor Cores, PTX Instructions, SASS, Floorsweep,
  Yield* (31 Mar 2026). https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor
  — free preview only; most tables paywalled.
- Jianyu Huang, *NVIDIA Blackwell SM100: TMEM, TMA, and the New Tensor Core Roofline* (12 Apr
  2026). https://jianyuh.github.io/cuda/2026/04/12/blackwell-sm100.html — restates SemiAnalysis
  figures; used as a secondary channel for the paywalled numbers, and labelled as such above.
- SemiAnalysis, *NVIDIA Tensor Core Evolution: From Volta To Blackwell*.
  https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
- NVIDIA Developer Forums, *Inter-GPU Latency on B200 Higher Than on Hopper*.
  https://forums.developer.nvidia.com/t/inter-gpu-latency-on-b200-higher-than-on-hopper/352473
  — user report: ~3 µs 8-byte all_reduce on 8×B200 vs ~2.3 µs on 8×H800/H200 via nccl-tests.
  No NVIDIA response in the thread. `[reported, single user, unconfirmed]`

Local primary sources read on this machine:

- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_mma.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_ld.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_cp.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_mma_ws.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/bin/ptxas` (opcode strings)
- `/home/aman/code/benchmark/RESULTS.md` (SM clock provenance: 1597 MHz)

Measurements taken for this document (8×B200, driver 595.71.05, nvcc 13.3, `-arch=sm_100a`,
SM clock locked at 1597 MHz, 2026-08-17). Sources in
`/tmp/claude-1000/-home-aman-code/930438ff-5f3c-49e6-a3d9-2663231246c6/scratchpad/`:

| File | Measures |
|---|---|
| `dq.cu` | device properties |
| `lat2.cu` | clock rate, FMA latency, sequential-chain latency sweep |
| `rnd.cu` | randomised-chain latency sweep, TLB knee |
| `mb.cu` | L1/L2/HBM latency and streaming bandwidth |
| `smem.cu` | SMEM latency, bank conflicts, atomics, L2 bandwidth |
| `p2p.cu` | NVLink5 peer read/write/memcpy bandwidth |
| `nf2.cu` | per-SM uncontended L2 latency (all 148 SMs) |
| `fine.cu` | failed fine-grained P-chase (recorded as a negative result) |

These are scratch files in a session-scoped directory; if the measurements matter beyond this
document they should be moved into `/home/aman/code/benchmark/`.
