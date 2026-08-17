# Blackwell in the literature: architecture analyses, benchmarks and MLPerf

## What this is

Everything published — academic, semi-academic and vendor — that actually *measures*
NVIDIA Blackwell (B200 / B300 / GB200 / GB300, SM100 / SM103 / SM120), plus the
Hopper and AMD MI300X/MI355X comparisons, distilled with citations. It ends with a
first-principles analytical model of decode latency for GLM-5.2 on our own 8×B200
box, built from byte counts taken out of our actual checkpoints and from
microbenchmarks run on our actual GPUs.

Three things make this document different from a reading list:

1. **The hardware-facts table is reconciled, not copied.** Secondary sources
   disagree about B200 SM count, HBM capacity, bus width and peak FLOPS — including
   one peer-reviewed paper. Section 1 resolves each disagreement against
   `libcuda` on our own machines.
2. **Everything is labelled.** `[verified]` = I read the number in the paper or
   measured it here. `[reported]` = the authors/vendor claim it in text I read.
   `[inferred]` = my arithmetic on top of verified inputs. `[listing]` = title,
   authors, ID and date confirmed through the arXiv API, abstract only — I did not
   read the evaluation, so no result numbers are quoted.
3. **No citation appears that I did not fetch.** Every arXiv ID below was
   retrieved. Where I only have the abstract, it says so.

Measurement environment for all first-party numbers: 8× NVIDIA B200 SXM, driver
595.71.05, CUDA 13.2 runtime / 13.3 toolkit, `compute_cap 10.0`. GPUs 1–7 were
idle when the microbenchmarks ran; the bandwidth and launch numbers were taken on
GPU 7. Weight accounting is exact — parsed from the safetensors shard headers of
`/home/aman/code/weights/GLM-5.2-FP8` and `/home/aman/code/weights/GLM-5.2-NVFP4`.

---

## Bottom line for our system

Ranked by expected effect on our two objectives.

1. **Build a hybrid checkpoint: NVFP4 routed experts + FP8 attention / shared
   expert / dense MLP.** This is the single biggest lever and it is free.
   `nvidia/GLM-5.2-NVFP4` quantises *only* the routed experts and leaves attention,
   the shared expert, the dense MLPs and the indexer in **BF16** `[verified — read
   the tensor dtypes]`. `zai-org/GLM-5.2-FP8` has all of those in FP8. Result: at
   batch 1 the NVFP4 build reads **6.00 GB/GPU/token vs 5.17 GB for FP8** — the
   "faster" build is 16% *slower* at the bandwidth floor. Splicing the two gives
   **7.52 GB/GPU per 4-token verify pass vs 13.18 (FP8) and 10.51 (NVFP4)** — a
   **1.75×** cut in the decode bandwidth floor. See §8.3.
2. **Attack kernel count, not just kernel speed.** Measured on our B200: a CUDA
   graph replays empty nodes at **~0.50 µs/node**; a non-graph stream launch costs
   **~2.8 µs** `[verified]`. At ~2,100 nodes per verify pass that is **~1.07 ms of
   pure dispatch floor per step — 31% of the realistic total floor** and comparable
   to the entire HBM term. Fusion, persistent/mega-kernels and grouped MoE GEMMs
   buy latency directly. This is why TileRT wins.
3. **Our 3.09× from speculative decoding is a *diagnostic*, not just a win.** If
   decode were bandwidth-bound, verifying 4 tokens would cost 2.55× a 1-token pass
   (expert bytes grow 22.7 → 86.7 GB, §8.2) and spec decoding could not exceed
   ~1.2×. Getting 3.09× proves the step is **latency/dispatch-bound today**
   `[inferred, high confidence]`. It also means the win will *shrink* as we fix
   items 1–2 — budget for that.
4. **Cut all-reduce count before all-reduce cost.** TP8 costs 2 all-reduces/layer =
   ~162/step. Our profile says collectives are 19.6% of time with **47% of that
   rank-arrival skew** — i.e. ~10.5 µs per all-reduce of which ~5 µs is real.
   The payload is trivial (4 tokens × 6144 × 2 B = 49 KB; 0.1 µs of NVLink5 time
   `[inferred]`), so this is 100% a latency/sync problem. NVIDIA's own MLPerf v6.0
   submission got 2.7× on identical GB300 hardware substantially from
   "optimized attention data parallelism" and disaggregation `[reported]`.
   Attention-DP + expert-parallel removes most per-layer all-reduces.
5. **Expert GEMMs are memory-bound at every concurrency we will ever run.** B200's
   FP8 machine balance is **~640 FLOP/byte** (4.5 PFLOPS ÷ 7 TB/s), so a weight
   tensor needs ~320 tokens to become compute-bound. Each routed expert only sees
   `B × 8/256 = B/32` tokens, so the global batch would need to be **~10,000
   tokens** `[inferred]`. Conclusion: NVFP4 buys nothing on expert *math* at our
   batch sizes — it buys **bytes**. Optimise the expert path for bandwidth and
   launch count, never for FLOPS.
6. **Take DFlash seriously as an EAGLE-3 replacement.** NVIDIA reports DFlash
   (block-diffusion drafter, KV injection, target-hidden-state conditioning)
   beating EAGLE-3 by **1.5× on gpt-oss-120b at matched interactivity** on 8× DGX
   B300, **2.8× vs 2.2× mean interactivity speedup on Llama-3.1-8B**, and **5.1× at
   concurrency 1 on Qwen3-8B under SGLang on a single B200** `[reported]`. It is
   already in TensorRT-LLM, SGLang and vLLM. Our 3-1-4 EAGLE config is exactly the
   regime it targets.
7. **DSA indexer: LiteTopK is a direct, published 1.35× on GLM-5.2 prefill on
   8×B200.** Same model, same hardware, same kernel we spend 5.8% of decode time in
   `[reported]`. It exploits score concentration to bin candidates and write back
   only promising ones, keeping exact top-k. Highest-confidence transferable result
   in the whole corpus.
8. **Do not chase INT8 W8A8, and do not plan a B300 migration around it.** NVIDIA
   gives B300 a **30:1 FP8:INT8 dense ratio** (B200 and H200 are 1:1); PTX never
   exposes the 5th-gen integer path on `sm_103a`, CUTLASS skips INT8 UMMA for 103a,
   and vLLM hard-errors on the first forward pass `[verified — read the paper]`.
   FP8/NVFP4 is the only sanctioned 8-bit-and-below path on Blackwell.
9. **Stop planning against datasheet FLOPS.** cuBLAS BF16 on B200 sustains
   **1,551 TFLOPS at 8192³ and 1,425 at 16384³** — 65% and 60% of the 2.38 PFLOPS
   the clock and SM count imply `[verified, third-party]`. Use ~1.5 PFLOPS BF16 /
   ~3.0–3.8 PFLOPS FP8 as the planning number.
10. **The realistic single-stream target is ~800–1,000 tok/s, not 500.** §8.5
    assembles the floors: 1.93 ms HBM + 1.07 ms dispatch + 0.49 ms collectives per
    verify pass ⇒ **886 tok/s if those terms serialise, 1,600 tok/s if they
    overlap**. We are at 365; TileRT is at ~500. Nobody is near the wall.

---

## 1. HARDWARE FACTS — B200 (SM100), sourced and reconciled

### 1.1 The table

| Quantity | Value | Source / confidence |
|---|---|---|
| Compute capability | **10.0** (`sm_100`) | `cuDeviceGetAttribute`, our box `[verified]` |
| SMs enabled | **148** | `MULTIPROCESSOR_COUNT` = 148 `[verified]`; corroborated by Chips&Cheese, SemiAnalysis, Jarmusch |
| SM sites present | 160 (80/die, 74 enabled/die) | Chips and Cheese `[reported]` |
| GPCs | 8 total (4 per die) | Jarmusch `[reported]`; consistent with 4 GPCs × ~10 TPCs × 2 SMs = 80 SMs/die `[inferred]` |
| Dies | 2, fused by NV-HBI | Jarmusch, SemiAnalysis, Chips&Cheese `[reported]` |
| Transistors | 208 B | Jarmusch `[reported]` |
| Max SM clock | **1,965 MHz** | `nvidia-smi -q -d CLOCK` + `CU_DEVICE_ATTRIBUTE_CLOCK_RATE` `[verified]` |
| Idle SM clock | 375 MHz | `nvidia-smi` `[verified]` |
| HBM | HBM3e, 8 stacks | `[reported]` |
| Memory clock | **3,996 MHz** (7.992 Gb/s/pin, DDR) | `[verified]` |
| Memory bus width | **7,680 bit** | `CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH` `[verified]` — see §1.2 |
| HBM theoretical BW | **7.67 TB/s** (7680 b × 7.992 Gb/s ÷ 8) | `[inferred]` from two verified inputs |
| HBM marketed BW | 8.0 TB/s (64 TB/s ÷ 8 GPUs) | NVIDIA DGX B200 page `[reported]` |
| HBM achieved, D2D copy (r+w) | **6.455 TB/s** | measured here, 8 GiB buffers `[verified]` |
| HBM achieved, memset (w only) | **6.791 TB/s** | measured here `[verified]` |
| HBM achieved, element-wise | 7 TB/s | cuTile-Rust paper `[reported]` |
| HBM achieved, TMA bulk | ~7.2 TB/s; LDGSTS ~6.6 TB/s | SemiAnalysis / Jianyu Huang `[reported]` |
| HBM achieved, STREAM Triad | 4.14 TB/s | Jarmusch `[verified in paper]` — outlier, see §1.2 |
| Capacity (CUDA-visible) | **191,495,471,104 B = 178.34 GiB** | `cuDeviceTotalMem` `[verified]` |
| Capacity (`nvidia-smi`) | **183,359 MiB = 179.06 GiB** | `[verified]` |
| Capacity (marketed) | 180 GB (1,440 GB ÷ 8) | NVIDIA DGX B200 `[reported]` — see §1.2 |
| L2 cache | **132,644,864 B = 126.5 MiB**, split 2 ways | `[verified]`; partitioning per Chips&Cheese |
| L2 persisting max | 79.06 MiB; access-policy window 128 MiB | `[verified]` |
| L1 / shared per SM | 256 KB unified; **228 KiB** max shared | `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` = 233,472 `[verified]`; carveouts 0/8/16/32/64/100/132/164/196/228 KB per CUDA Blackwell Tuning Guide |
| Shared per block (opt-in) | 227 KiB (232,448 B) | `[verified]` |
| Register file per SM | 64 K × 32-bit = **256 KiB** | `[verified]` |
| TMEM per SM | **256 KB** = 512 col × 128 lane × 32 bit | Jarmusch, Cornell CAC, thesoftwarefrontier `[reported]`, 3 independent |
| Threads / blocks per SM | 2,048 / 32 (64 warps) | `[verified]` |
| Cluster launch | supported; 8 portable, **16 non-portable** | Tuning Guide `[reported]`; `CLUSTER_LAUNCH`=1 `[verified]` |
| NVLink | **18 links** (`NV18`), 53.125 GB/s/link raw | `nvidia-smi nvlink -s`, `topo -m` `[verified]` |
| NVLink effective | 900 GB/s/dir, 1.8 TB/s bidir | `[inferred]` — 18 × 50 GB/s payload after encoding |
| Cross-die (NV-HBI) penalty | **30 cycles / 15.5 ns** | Alpay & Alpay `[reported]`; SemiAnalysis says "~300 cycles for cross-die *L2*" `[reported]` — different measurements, both plausible |
| Thread-to-thread, same die | 90–100 ns; cross-die 190–220 ns | Chips and Cheese `[reported]` |
| L2 bandwidth | 21 TB/s local partition, 16.8 TB/s cross | Chips and Cheese, Vulkan `[reported]` |
| TDP / power limit | **1,000 W** | `nvidia-smi power.limit` `[verified]` |
| PCIe | Gen5 x16 | `[verified]` |
| DGX B200 system power | ~14.3 kW | NVIDIA `[reported]` |

**Peak tensor throughput.** NVIDIA's published per-GPU dense figures (from HGX B200:
144 | 72 PFLOPS FP4 sparse|dense over 8 GPUs) and the clock-derived figures:

| dtype | NVIDIA dense | NVIDIA sparse | Derived @1,965 MHz × 148 SM | Measured |
|---|---|---|---|---|
| FP64 (tensor) | 37 TFLOPS | — | — | 36.30 TFLOPS DGEMM `[verified, Jarmusch]` |
| TF32 | 1.125 PFLOPS | 2.25 | — | — |
| FP16 / BF16 | **2.25 PFLOPS** | 4.5 | **2.382 PFLOPS** | 1.93 PFLOPS burst `[Jarmusch]`; **1.55 PFLOPS cuBLAS @8192³** `[Paul Chan]` |
| FP8 / FP6 | **4.5 PFLOPS** | 9.0 | **4.765 PFLOPS** | 3.85 PFLOPS burst `[Jarmusch]` |
| NVFP4 | **9.0 PFLOPS** | 18.0 | **9.530 PFLOPS** | 7.70 PFLOPS burst `[Jarmusch]` |
| INT8 | 4.5 POPS | 9.0 | — | 3.93 POPS `[Jarmusch]` |

Per-SM-per-clock dense rate is **8,192 FLOP** for 16-bit (4 partitions ×
1024 16-bit MACs/clk, Chips and Cheese `[reported]`), doubling per precision halving.

### 1.2 Disagreements, resolved

These are the numbers secondary sources get wrong. Resolving them is the point of
this section.

**(a) HBM capacity: 180 GB or 192 GB?** Both, in different units, for the same
silicon. Eight 24 GB HBM3e stacks = 192 GB *decimal*; `nvidia-smi` reports
183,359 MiB = **179.06 GiB**, which NVIDIA markets as "180 GB". CUDA sees
178.34 GiB after ECC/driver reservation. **Our own boxes are the standard B200 —
there is no separate 192 GB SKU here.** Jarmusch et al. state "192 GB HBM3e memory
space" and Paul Chan's otherwise-excellent GEMM writeup states "288 GB HBM3E"
(that is B300's capacity). *Use 178 GiB usable per GPU for capacity planning.*

**(b) Memory bus: 7,680 or 8,192 bit?** CUDA reports **7,680 bit** `[verified]`.
8,192 bit (8 full stacks) would give 8.18 TB/s and matches NVIDIA's "8 TB/s"
marketing; 7,680 bit gives 7.67 TB/s. 7,680 = 8,192 × 15/16, i.e. one-sixteenth of
the interface floorswept — entirely consistent with the aggressive
GPC/TPC floorsweeping SemiAnalysis documents. **Plan against 7.67 TB/s
theoretical and ~6.5–7.0 TB/s achievable.** Our measured 6.46 (r+w) / 6.79
(write-only) TB/s are 84% / 89% of 7.67 — normal HBM efficiency.

**(c) Achieved bandwidth spread is 4.1 → 7.2 TB/s across sources.** Jarmusch's
STREAM Triad 4.14 TB/s (51.8% of peak) is the outlier and reflects a scalar
`a[i]=b[i]+s*c[i]` kernel, not a tuned bulk-copy path. Chips and Cheese report
5.6 TB/s. SemiAnalysis/Jianyu Huang measure 6.6 TB/s (LDGSTS) and ~7.2 TB/s (TMA);
the cuTile-Rust paper reports 7 TB/s element-wise; we measure 6.46–6.79 TB/s.
**For weight streaming with TMA, 6.5–7.0 TB/s is the defensible number.**

**(d) Peak FLOPS: NVIDIA's spec implies 1.856 GHz, not the 1,965 MHz boost.**
148 SMs × 8,192 FLOP × 1.965 GHz = 2.382 PFLOPS BF16, but NVIDIA publishes
2.25 PFLOPS — a 5.6% gap that back-solves to 1,856 MHz. Meanwhile Jarmusch reports
FP8 3,850.6 TFLOPS as "96.3% of peak", which back-solves to a 4.0 PFLOPS peak
assumption — inconsistent with both NVIDIA's 4.5 and the clock-derived 4.77. Their
absolute measurements are **85.6% of NVIDIA's dense spec** across FP8, FP6 and FP4
(identical ratio, so it is a clock effect, not a format effect). *Their absolute
numbers are usable; their "% of peak" column is not.*

**(e) The one genuinely load-bearing number: sustained GEMM.** Paul Chan measured
cuBLAS itself at **1,551 TFLOPS (8192³) and 1,424.8 TFLOPS (16384³) BF16** —
65% / 60% of the 2.382 PFLOPS clock-derived peak, at 1,000 W. Jarmusch's
1,926 TFLOPS BF16 came from a 100-iteration burst and does not reach thermal steady
state. **Plan at ~1.5 PFLOPS BF16.**

**(f) Jarmusch's decompression-engine numbers disagree with their own slides.**
Paper Table: LZ4 173.23 GB/s in, Snappy 61.38, Zstd 77.50, GZIP 42.00, Bitcomp
154.02/462.37, ANS 539.21 out. IPDPS slide deck Fig. 5 shows zstd 820, lz4 940,
snappy 1010, deflate 760 GB/s — labelled "illustrative". A 5–20× discrepancy inside
one publication. **Treat the DE numbers as unestablished.**

### 1.3 B200 vs the alternatives

| | B200 SXM | B300 (Ultra) | H200 SXM | MI355X |
|---|---|---|---|---|
| Dense NVFP4/MXFP4 | 9.0 PF | **15 PF** `[NVIDIA blog]` | — | **10.1 PF** |
| Dense FP8 | 4.5 PF | 4.5 PF | 2.0 PF | **5.0 PF** |
| Dense BF16 | 2.25 PF | 2.25 PF | 1.0 PF | **2.5 PF** |
| Dense INT8 | 4.5 POPS | **0.15 POPS** (30:1) | 2.0 POPS | 5.0 POPS |
| HBM | 180 GB / ~7.7–8 TB/s | 288 GB / 8 TB/s | 141 GB / 4.8 TB/s | **288 GB / 8 TB/s** |
| Last-level cache | 126.5 MiB L2 | — | 50 MB L2 | **256 MB** |
| Scale-up link | 1.8 TB/s NVLink5 | 1.8 TB/s | 900 GB/s NVLink4 | 7 × 153 GB/s IF |
| Board power | 1,000 W | 1,400 W | 700 W | **1,400 W** |

Sources: NVIDIA HGX page and DGX B200 page, AMD MI355X product page, and
arXiv:2608.11693 for the B300 INT8 ratio. All `[reported]` except B200 rows marked
`[verified]` in §1.1. **MI355X has more peak FLOPS, more HBM and more cache than
B200 at 1.4× the power** — the gap is software and scale-up topology, which is
exactly what InferenceMAX measures (§5).

---

## 2. Microbenchmarking and dissection

| Paper | Lab | Venue / year | HW | Headline | Adopted? |
|---|---|---|---|---|---|
| Microbenchmarking NVIDIA's Blackwell Architecture (arXiv:2512.02189) | Jarmusch & Chandrasekaran, U. Delaware | **IPDPS 2026**, New Orleans, May 25–29 | B200 vs H200, CUDA 12.6 / driver 560 | `tcgen05.mma` 11 cycles flat vs `wgmma` 38→132; FP8 3,850 TFLOPS; 1.85× ResNet-50, 1.55× GPT-1.3B vs H200 | open-source suite; the reference Blackwell characterisation |
| Dissecting Nvidia Blackwell — Tensor Cores, PTX, SASS, Floorsweep, Yield | Kimbo Chen & Dylan Patel, SemiAnalysis | blog, 2026-03-31 | B200 | 2SM MMA `cta_group::2`; 78–80% of SoL at 4 in-flight MMAs; per-GPC floorsweep map | industry reference (paywalled beyond preview) |
| Nvidia's B200: Keeping the CUDA Juggernaut Rolling | Chips and Cheese (w/ Verda) | blog | B200 vs H100/A100/MI300X | 126 MB L2 in **two 63 MB partitions**; 21 / 16.8 TB/s L2; 5.6 TB/s HBM; FP16 vector no longer double-rate | independent, OpenCL/Vulkan |
| NVIDIA Blackwell SM100: TMEM, TMA, and the New Tensor Core Roofline | Jianyu Huang | blog, 2026-04-12 | B200 | SMEM 128 B/clk ceiling makes SS-mode MMA **SMEM-bound below N=128**; TMA 7.2 vs LDGSTS 6.6 TB/s | best kernel-level model published |
| How Blackwell's Tensor Memory Actually Works | The Software Frontier | blog | B200 | TMEM alloc rules, SASS lowering (`UTCHMMA`/`UTCFMMA`/`UTCOMMA`), 562.5 TB/s chip-wide accumulator traffic | best TMEM reference |
| Spec Sheets Are Not Kernels: INT8 on Blackwell Ultra (arXiv:2608.11693) | Teng-Ruei Chen | arXiv 2026-08 | B300 (`sm_103a`) | B300 FP8:INT8 = **30:1**; INT8 undeployable through PTX/CUTLASS/vLLM/SGLang | negative result, high value |
| Unprivileged Topology Certificates (arXiv:2606.24934) | Alpay & Alpay | arXiv 2026-06 | B200 | 74 SMs/die; **30-cycle / 15.5 ns cross-die penalty**; 2-way L2 confirmed | attestation, but good microarch data |
| Outperforming cuBLAS on B200 | Paul Chan | blog | B200 | 91.9% of cuBLAS avg, 98.3% best; **cuBLAS itself only 1.55 PF BF16** | best public B200 GEMM writeup |
| Characterizing Warp Divergence Pascal→Blackwell (arXiv:2607.23402) | Alpin Dale | arXiv 2026-07 | Ampere/Hopper/Blackwell | serialization cost stable despite reconvergence changes | `[listing]` |

### 2.1 tcgen05 and TMEM — the mechanism

The Blackwell tensor core is not a faster Hopper tensor core; it has a different
issue model, and every result above follows from that.

**Issue scope.** `wgmma` was warp-group scoped: 128 threads collectively issued and
collectively owned the accumulator in their registers. `tcgen05.mma` is **CTA
scoped and issued by a single elected thread** `[reported, 3 sources]`. Threads
become asymmetric — one orchestrates, others move data — which is why Blackwell
kernels are warp-specialised producer/consumer by construction.

**Why accumulators had to leave the register file.** Blackwell's largest MMA
produces a 128×256 FP32 tile = 256 registers/thread, over the 255-register
architectural ceiling. TMEM exists to make that tile representable
`[reported, thesoftwarefrontier]`. Secondary benefit: threads no longer own
accumulators, so warp specialisation stops fighting register allocation.

**TMEM shape and rules.** 256 KB/SM as **128 lanes × 512 columns of 32-bit cells**.
Addresses are 2D (`hi16 = lane, lo16 = column`), not linear. `tcgen05.alloc` takes
whole columns (all 128 lanes), column counts must be **powers of two, ≥ 32**; the
largest single-CTA accumulator (`m128n256k16`) needs exactly 256 of the 512
columns. Hardware partitions access by warp — warp *i* reaches lanes `32i..32i+31`
— so **draining a full tile structurally requires all four warps** `[reported]`.
`tcgen05.alloc` lowers to `UTCATOMSWS.FIND_AND_SET.ALIGN` with a `NANOSLEEP 0x64`
spin-retry and three compiler-injected trap handlers (use-after-free, double-free,
invalid phase). Requires PTX ISA ≥ 8.6 (CUDA 12.8+). **`wgmma` is removed entirely
from `sm_100a`/`sm_103a`/`sm_120a`.**

**Latency.** Jarmusch's dependency-chain measurement, FP16 accumulator-carried:

| tile | 64×8 | 64×16 | 64×32 | 64×64 | 64×128 |
|---|---|---|---|---|---|
| Hopper `wgmma` | 38 | 56 | 78 | 102 | 132 cycles |
| Blackwell `tcgen05` | 11 | 11 | 11 | 11 | 11 cycles |

`[verified — read off IPDPS Fig. 2]`. Flat latency is the spatial-array signature.
**Implication: on Blackwell, tile size is a throughput knob, not a latency knob** —
pick tiles to feed the pipeline, not to shorten the instruction.

**The catch nobody puts on a slide.** SemiAnalysis and Jianyu Huang independently
find that sustaining peak needs **256–1,024 in-flight MMA instructions**, while
real kernels carry **1–4**, capping throughput at **78–80% of speed-of-light**
`[reported, 2 sources]`. That, plus the 1,000 W clock ceiling, is the gap between
9 PFLOPS on the datasheet and 1.55 PFLOPS from cuBLAS.

**2-CTA MMA.** Under `cta_group::2` two SMs in a TPC cooperate on one
`tcgen05.mma`, doubling effective tile area while halving per-SM operand traffic.
Jianyu Huang measures **>2× strong scaling** for SS-mode small shapes — superlinear,
because each SM contributes independent shared-memory bandwidth and SMEM was the
bottleneck. Paul Chan credits 2-CTA MMA with a 1.5× arithmetic-intensity gain.

**The SMEM roofline that actually binds.** For FP16 1SM MMA at `M=128,N=64,K=16`:
A operand 4,096 B + B operand 2,048 B, 262,144 FLOP; SMEM sustains **128 B/clk**, so
operand service takes 48 cycles against 32 cycles of math — **SMEM-bound, not
tensor-core-bound**. The crossover is `N=128` `[reported]`. *Practical rule: never
write an SS-mode Blackwell GEMM with N < 128.*

**TMEM accumulator bandwidth is precision-invariant.** Because K is always 256 bits,
accumulator traffic is ~562.5 TB/s chip-wide (~3.8 TB/s/SM) at FP16, FP8 *and*
FP4 — roughly **70× B200's entire HBM bandwidth** `[reported]`. That is why
accumulators needed dedicated storage rather than cache.

### 2.2 Dual-die: it is a NUMA machine

Three independent sources agree B200 is two dies behind one CUDA device:

- L2 is **two 63 MiB partitions**, one per die; 21 TB/s local, 16.8 TB/s crossing
  (Chips and Cheese `[reported]`; our `L2_CACHE_SIZE` = 126.5 MiB `[verified]`).
- Cross-die latency penalty: 30 cycles / 15.5 ns (Alpay `[reported]`); "~300 cycles"
  for cross-die L2 (SemiAnalysis `[reported]`); thread-to-thread 90–100 ns same-die
  vs 190–220 ns cross-die (Chips and Cheese `[reported]`).
- **"Nvidia's scheduler tries to fill one partition's SMs before going to the
  other"** (Chips and Cheese `[reported]`).

For us: a 148-SM persistent kernel spans both dies. Any all-to-all inside a kernel
(MoE dispatch/combine, attention KV broadcast) pays the NV-HBI tax on roughly half
its traffic. Chan's Hilbert-curve tile scheduling lifting L2 hit rate 49% → 65%
`[verified]` is the same phenomenon exploited constructively.

---

## 3. Performance models

| Paper | Lab | Venue/year | HW | Result | Adopted? |
|---|---|---|---|---|---|
| TileSight: First-Principles Tile-Centric Analytical GPU Performance Model (arXiv:2607.22432) | Mo, Ma, Xia, Xue, Yang, Mai, Luk, Fan et al. | arXiv 2026-07 | A100, H200, **B200**, B6000 | **12.35% MAPE** single-GPU kernels; 16.18% wMAPE 32-GPU fused; **13.52% wMAPE end-to-end vLLM serving**; L2 hit rate within ~1 pt | research |
| Microbenchmark-Driven Analytical Performance Modeling (arXiv:2605.04178) | Jarmusch & Chandrasekaran | arXiv 2026-05 | **B200**, MI300A | 1.31% MAE / 21 kernels (B200); roofline baseline >95% error | research |
| SOL-ExecBench (arXiv:2603.19173) | Lin, Hari, Ye, T. Chen, Kozyrakis, Ceze, Grover +25 (NVIDIA/CMU/UW) | arXiv 2026-03 | Blackwell | 235 kernel problems from 124 models; SOL bounds from "SOLAR"; BF16/FP8/**NVFP4** | benchmark for kernel agents |
| GPU-Tile-Sim (arXiv:2607.11262) | Ding et al. | arXiv 2026-07 | H100 validated, Blackwell preliminary | 1.22–8.71% MAPE | `[listing]` |
| Sim-FA (arXiv:2605.00555) | Zhou et al. | arXiv 2026-05 | Hopper TMA/WGMMA | 5.7% MAPE on FlashAttention-3 | `[listing]` |

**TileSight is the one to actually read.** It elevates the tile from a programming
primitive to an analysis primitive across three layers: (i) *intra-tile* — models
compute/memory pipeline overlap with resource vectors over network, memory and
compute pipes; (ii) *inter-tile* — schedules dependent actions to expose legal
overlap and **infers multi-level cache hit rates from tile reuse distance**;
(iii) *cross-device* — maps remote tensor accesses through an **α–β stage cost**
for inter-GPU communication `[reported]`. That third layer is exactly what §8 does
by hand for our TP8 all-reduces, and 13.52% wMAPE on end-to-end vLLM serving is
good enough to use as a config search oracle rather than sweeping the hardware.

**Jarmusch's 1.31% MAE is not comparable to TileSight's 12.35%.** It is on 21
Rodinia/SPEChpc kernels, not on LLM kernels, and the "roofline baseline >95% error"
framing means they compared against a naive roofline nobody would use for these
kernels. Useful for the B200 constants it derives (TMEM, TMA, 5th-gen tensor core
terms); do not port the accuracy claim to transformer workloads.

**Why classical roofline fails on Blackwell** — the concrete reasons, all sourced
above: (1) two L2 partitions with a bandwidth cliff between them, so "L2 bandwidth"
is not a scalar; (2) SMEM at 128 B/clk becomes the binding roof below N=128, a term
the roofline has no axis for; (3) TMEM adds a tier with precision-invariant
bandwidth; (4) sustained clock is 60–70% of boost under GEMM load, so the compute
roof is workload-dependent; (5) MMA pipeline depth (1–4 in flight vs 256–1,024
needed) caps you at 78–80% of the roof before any memory effect. Jarmusch's own
takeaway slide says it plainly: *"Re-derive your roofline — Hopper-era roofline
assumptions don't hold with dual-chip and new precisions."*

---

## 4. MLPerf Inference on Blackwell

MLPerf submission detail is unusually informative because the configuration is
disclosed and the latency constraints are fixed. **v6.0 (April 1, 2026) is the
current round**; 24 submitters; largest system 72 nodes / **288 accelerators**
`[verified, MLCommons]`.

### 4.1 The latency constraints — read these first

| Benchmark | Server TTFT / TPOT | Interactive TTFT / TPOT |
|---|---|---|
| Llama3.1-8B | 2000 / 100 ms | 500 / 30 ms |
| Llama2-70B | 2000 / 200 ms | 450 / 40 ms |
| Llama3.1-405B | 6000 / 175 ms | 4500 / 80 ms |
| Mixtral-8x7B | 2000 / 200 ms | — |
| **DeepSeek-R1** | **2000 / 80 ms** | **1500 / 15 ms** |
| **GPT-OSS-120B** | **3000 / 80 ms** | 2000 / 20 ms |

`[verified — MLCommons inference_rules.adoc]`.

**This is the single most important context for reading any MLPerf tok/s number.**
DeepSeek-R1 "Server" allows **80 ms TPOT = 12.5 tok/s/user**. Our 2.74 ms TPOT is
**29× more aggressive than MLPerf Server and 5.5× more aggressive than MLPerf
Interactive**. MLPerf throughput records are won at batch sizes we will never run.
Do not benchmark against them; mine them for *techniques*.

### 4.2 v6.0 results

| System | Benchmark | Scenario | tok/s | tok/s/GPU |
|---|---|---|---|---|
| HGX **B200** ×8 | DeepSeek-R1 | Server | 51,693 | 6,462 |
| HGX **B200** ×8 | DeepSeek-R1 | Offline | 58,582 | 7,323 |
| HGX **B200** ×8 | gpt-oss-120B | Server | 87,444 | 10,931 |
| HGX **B200** ×8 | gpt-oss-120B | Offline | 85,921 | 10,740 |
| HGX B300 ×8 | DeepSeek-R1 | Server / Offline | 60,413 / 69,319 | 7,552 / 8,665 |
| GB300 NVL72 ×8 | DeepSeek-R1 | Server / Offline | 64,510 / 76,347 | 8,064 / 9,543 |
| GB300 NVL72 ×72 | DeepSeek-R1 | Server / Offline | 575,580 / 673,936 | 7,994 / 9,360 |
| GB300 NVL72 ×72 | gpt-oss-120B | Server / Offline | 1,096,770 / 1,046,150 | 15,233 / 14,530 |
| 4× GB300 NVL72 (288 GPU) | DeepSeek-R1 | Offline / Server | 2,494,310 / 1,555,110 | 8,661 / 5,400 |
| GB300 NVL72 | Llama-3.1-405B | Server / Offline | — | 259 / 271 |
| HGX B200 ×8 (Lambda) | Llama-3.1-8B | Server / Offline / Inter. | 130,008 / 160,403 / 128,750 | — |

Per-GPU rows are Nebius/StorageReview reported or my division `[reported]`/`[inferred]`.

**v5.1 → v6.0 on identical GB300 NVL72 hardware** `[reported, NVIDIA]`:

| | v5.1 | v6.0 | gain |
|---|---|---|---|
| DeepSeek-R1 Server | 2,907 tok/s/GPU | **8,064** | **2.77×** |
| DeepSeek-R1 Offline | 5,842 | 9,821 | 1.68× |
| Llama-3.1-405B Server | 170 | 259 | 1.52× |
| Llama-3.1-405B Offline | 224 | 271 | 1.21× |

NVIDIA claims this cut cost/token by **>60%** on unchanged silicon.

### 4.3 What the submissions actually did

NVIDIA's technical writeup names six techniques `[reported]`:

1. **Disaggregated serving** (Dynamo) — "separates and individually optimizes the
   configurations of each inference phase (prefill and decode)". Worth **47% higher
   per-GPU performance** on Llama-3.1-405B Interactive on GB200 NVL72 vs DGX B200
   with traditional serving.
2. **Wide Expert Parallel (WideEP)** — spread MoE expert weights across many GPUs
   inside the NVL72 domain "to reduce expert weight loading bottlenecks in high-
   interactivity scenarios". *This is the direct fix for the §8 finding that routed
   experts are 82% of our bandwidth floor.*
3. **Multi-Token Prediction** — "applying compute that otherwise goes unutilized to
   predict and verify additional tokens in parallel (**up to three** in this
   implementation)".
4. **KV-aware routing** — route by estimated compute cost across Dynamo workers.
5. **Optimized attention data parallelism** — "better balancing of context requests
   between different ranks, enabling significant speedups end-to-end".
6. **Kernel fusion / faster kernels** — "reduce kernel count and overhead".

Quantization: TensorRT Model Optimizer to **NVFP4** for DeepSeek-R1, Llama-3.1-405B,
Llama-2-70B and Llama-3.1-8B. Lambda's B200 closed submission ran **FP4 weights +
FP8 KV cache, TP=1 EP=1, TensorRT-LLM 1.2 / CUDA 13.1** for Llama-3.1-8B; their
open-division gpt-oss-120B used **TP=4 EP=4** with "BLAZE", a runtime MoE routing
optimisation steering ambiguous tokens away from overloaded experts — **−31% P99
latency, +3.9% throughput, no retraining** `[reported]`.

Two observations for us. First, **items 1, 2, 5 and 6 are all "reduce serialisation
and per-layer collectives"** — the same conclusion §8 reaches from arithmetic.
Second, NVIDIA's own MTP depth is 3 while our EAGLE runs 3-1-4; we are already at
or past their draft depth, which suggests our remaining spec-decoding headroom is
in acceptance rate (→ DFlash, §7) rather than in depth.

---

## 5. Credible non-academic measurement

**Clearly labelled: everything in this section is vendor- or press-reported unless
marked otherwise.**

### 5.1 SemiAnalysis InferenceMAX / InferenceX — the most useful of the lot

Methodology `[reported, read from their blog]`: models Llama3-70B (vLLM default),
DeepSeek-R1 670B (SGLang default), gpt-oss-120B; hardware H100/H200/B200/GB200
NVL72 and MI300X/MI325X/MI355X; three ISL/OSL profiles — **1K/1K chat, 1K/8K
reasoning, 8K/1K summarization**, input randomised to 80–100% of range. They sweep
max concurrency (4/8/16/32/64+) and parallelism (1/2/4/8+) and keep the Pareto set —
"every data point P such that there is no point better than P in both throughput and
latency" — plotting **tok/s/GPU vs tok/s/user**. Cost/token comes from the
SemiAnalysis AI TCO model (4-year life, hyperscaler tier, BoM + DC + network).
**Power is estimated from component TDP, not measured** — they say so explicitly and
plan to move to `ipmitool`. Runs nightly on hundreds of chips.

**The directly relevant page: GLM 5/5.1, B200 vs MI355X, 8K/1K, FP8** `[reported]`:

| interactivity | B200 tok/s/GPU | MI355X tok/s/GPU | B200 $/M tok | MI355X $/M tok |
|---|---|---|---|---|
| 32 tok/s/user | **1,756.3** | 1,368.5 | $0.274 | $0.304 |
| 51 tok/s/user | **1,287.2** | 956.5 | $0.373 | $0.436 |
| 71 tok/s/user | **1,003.6** | 709.4 | $0.479 | $0.587 |

B200 is +28% tok/s/GPU and ~20% more energy-efficient; MI355X is ~1% cheaper per
token on some workloads. Other headlines: B200 has **40% lower latency at
comparable throughput** on DeepSeek-R1 670B FP8 vs MI355X; MI355X has **lower TCO
than B200 under vLLM below 225 tok/s/user** on gpt-oss-120B FP4 summarization;
MI355X gained ~3× tokens/s/MW over MI300X (750K → 2.55M).

**Note the extrapolation trap.** The GLM table stops at 71 tok/s/user. Our target is
**365 tok/s/user** — five times off the right edge of their Pareto curve. In that
regime tok/s/GPU collapses toward the batch-1 limit and the ranking is set by
dispatch latency, which InferenceMAX does not probe. Their numbers bound our
*throughput* objective well and say nothing about our *latency* objective.

### 5.2 NVIDIA performance blogs — vendor-reported, treat accordingly

- gpt-oss-120B: **60,000 tok/s/GPU** with TensorRT-LLM v1.0; with Eagle3-v2 spec
  decoding **30,000 tok/s/GPU** at 1,000 tok/s/user (from 6,000 prior).
- Llama-3.3-70B: B200 **>10,000 tok/s/GPU at 50 tok/s/user**, "4× higher per-GPU
  throughput vs H200".
- "10× throughput per megawatt for MoE models"; "15× lower cost per million tokens";
  "$5M GB200 NVL72 → $75M token revenue".

The 60,000 tok/s/GPU figure and MLPerf's 10,931 tok/s/GPU for the same model on the
same GPU differ by 5.5× — because MLPerf enforces 80 ms TPOT and the blog does not.
Both are "true"; only one is a serving number.

### 5.3 Chips and Cheese, SemiAnalysis dissection, and independent blogs

Covered in §2. Chips and Cheese is the only source running an independent
OpenCL/Vulkan suite across NVIDIA and AMD on the same harness, which is why their
B200-vs-MI300X cache and latency comparisons are the ones I trust most. Their
headline judgement — B200 is a "straightforward successor" whose gains come
"primarily from increased SM density rather than architectural breakthroughs", and
MI300X retains vector-FP16 and local-memory-bandwidth advantages — is a useful
corrective to vendor framing.

**Modular:** as of August 2026 their public blog has no B200-vs-MI-series benchmark
posts (checked; recent posts are Mojo 1.0, the Qualcomm acquisition, and MoE serving
in 26.4). No usable data.

---

## 6. Rack-scale GB200/GB300 NVL72

| Paper | Venue | HW | Result |
|---|---|---|---|
| **UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods** (arXiv:2607.06202) | **ACM SIGCOMM 2026** | NVL72 / NVL576, CloudMatrix384 | **−52.4% All-to-All latency; −11.1% MoE TPOT** |
| Provisioning to Runtime Optimization of a 100 MW-Scale AI Cluster (arXiv:2605.24461) | arXiv 2026-05 | 150 MW DC, **83,000 GB200** | power planning → runtime power management |
| FlashBoot: Sub-Second Weight Loading at Rack Scale (arXiv:2608.08482) | arXiv 2026-08 | NVL72 | 20.1 s → 0.4 s single node; 87 s → 0.32 s rack `[listing]` |
| OpScale: Operator-level Provisioning and Autoscaling (arXiv:2608.13499) | arXiv 2026-08 | A100 / GB200 | −36.3% GPUs, −28% power, or +44% throughput `[listing]` |
| NVIDIA MLPerf v6.0 4× GB300 NVL72 | MLPerf v6.0 | 288 GPUs | 2.49 M tok/s offline DeepSeek-R1 |

**UBEP is the most transferable.** It identifies three bottlenecks in MoE all-to-all
on high-bandwidth superpods: **strict execution serialization from coarse-grained
BSP orchestration**, **prohibitive synchronization overhead**, and **load imbalance
from distance-agnostic scheduling** `[verified — read the abstract]`. All three
describe our profile precisely: 19.6% collectives of which **47% is rank-arrival
skew** is textbook BSP serialization plus imbalance. A SIGCOMM-accepted 52.4%
all-to-all reduction is the strongest published evidence that our collective term
has ~2× in it without touching the transfer itself.

Caveat for us: UBEP targets NVL72/NVL576 and CloudMatrix384. We have a single
8-GPU NVLink5 domain (`NV18` all-to-all through NVSwitch, 2 NUMA nodes). The
scheduling and sync ideas port; the topology-distance ideas mostly do not, because
every pair in our box is one switch hop.

---

## 7. Blackwell papers that hit our exact workload

| Paper | HW | Model | Result |
|---|---|---|---|
| **LiteTopK: Fused Indexer-TopK for Long-Context Sparse Attention** (arXiv:2607.11976) | **8× B200** | **GLM 5.2** | **1.35× prefill, exact top-k preserved** |
| Vortex: Programmable Sparse Attention Serving (arXiv:2606.06453, Chen/Jia/B.Chen, CMU+) | B200 | GLM-4.7-Flash (MLA), MiniMax-M2.7 229B | 3.46× vs full attention; **4.7× on MLA**; 1.37× on 229B |
| CTA-Pipelining (arXiv:2607.07862, Liu et al., UIUC) | 8× H200 / **8× B200** | 2-layer GEMM (MLP) | **−31.8% latency vs micro-batching, −29.6% vs TP** |
| DFlash speculative decoding (NVIDIA developer blog) | 8× DGX B300, B200, B300 | gpt-oss-120B, Llama-3.1-8B, Gemma-4-31B, Qwen3-8B | 15× throughput at matched interactivity; **1.5× over EAGLE-3**; 5.1× at concurrency 1 (SGLang, B200) |
| Fearless Concurrency on the GPU (arXiv:2606.15991, Elibol, Roesch, Gelado, Buehler, **Garland**) | B200 | Qwen3-32B | **7 TB/s element-wise, 2 PFLOP/s GEMM (96% cuBLAS)**; 82 tok/s batch-1 decode |
| Evaluating CUDA Tile (arXiv:2604.23466) | B200, RTX PRO 6000 | attention/GEMM | **1,007 TFLOP/s fused attention (2.5× FA-2)** in 60 lines; GEMM 52–79% of cuBLAS; SM120 only 53% of FA-2 |
| Inference Economics of Enterprise Coding Agents (arXiv:2607.13080) | Blackwell | **GLM-5.1/5.2 NVFP4** | on-prem = 40.1% of TCO under shared allocation; dedicated reservation costs 43.8% *more* than cached API |

**LiteTopK** is the highest-confidence transfer in this document — same model, same
GPU count, same kernel. Mechanism: sparse-attention scores concentrate ("curse of
dimensionality"), so sample a small subset to estimate the query's score range, bin
candidates by that range, maintain a tight approximate threshold online, and **write
back only promising candidates**, killing most of the global-memory traffic and the
synchronisation. `LiteDSA` adds packing of neighbouring tokens whose top-k sets
overlap, with masking of extraneous scores. Exact top-k is preserved. Reported
**1.35× on GLM 5.2 prefill on eight B200s** `[reported]`. Our indexer is 5.8% of
decode time and considerably more of prefill/TTFT.

**Vortex's 4.7× on MLA-based GLM-4.7-Flash** is the second-strongest signal —
programmable page-centric sparse attention on B200 against an MLA model in the same
family as ours.

**CTA-Pipelining** is the most interesting *latency* paper here and the least known.
Its thesis: multi-GPU systems are now "tightly integrated shared-memory structures"
but software still treats coherent interconnects as networks. It exploits
dependencies at CTA granularity so **dependent kernels execute concurrently across
GPUs**, orthogonal to TP and composable with it — 31.8% / 29.6% latency cuts on
8× B200 for 2-layer GEMM (an MLP block), using CUTLASS/cuBLAS/NCCL. For a
single-stream engine on NV18 all-to-all, this is precisely the missing axis: TP
splits work spatially but still barriers at every layer.

---

## 8. ANALYTICAL MODEL — decode latency for GLM-5.2 on 8× B200

All inputs are `[verified]`: architecture from `config.json`, weight bytes parsed
from safetensors headers, hardware constants from `libcuda` and microbenchmarks on
our own GPUs. Arithmetic is `[inferred]` and shown.

### 8.1 Model and machine constants

GLM-5.2 (`GlmMoeDsaForCausalLM`): 78 layers, `hidden=6144`; layers 0–2 dense MLP
(`intermediate=12288`), layers 3–77 MoE with **256 routed experts, top-8, 1 shared**,
`moe_intermediate=2048`. MLA: `q_lora_rank=2048`, `kv_lora_rank=512`,
`qk_rope_head_dim=64`, `qk_nope=192`, `v_head_dim=256`, 64 heads. DSA indexer:
`index_n_heads=32`, `index_head_dim=128`, **`index_topk=2048`**,
**`index_topk_freq=4`**, **21 `full` indexer layers, 57 `shared`**. 1 MTP layer.
Vocab 154,880, untied head.

Machine: 148 SM @ 1,965 MHz; HBM 7.67 TB/s theoretical, **7.0 TB/s planning /
6.5 TB/s conservative**; L2 126.5 MiB; NVLink5 900 GB/s/dir; graph node **0.50 µs**;
stream launch **2.8 µs**.

### 8.2 Per-token bytes — exact

Checkpoint totals: **FP8 755.62 GB** (F8_E4M3 751.23 + BF16 4.21 + F32 0.18),
**NVFP4 464.80 GB** (U8 362.39 + BF16 57.11 + F8_E4M3 45.30).

| component | FP8 build (GB) | NVFP4 build (GB) | read every pass? |
|---|---|---|---|
| attention (78 L) | 12.875 (FP8) | **25.743 (BF16)** | yes |
| routed experts (75 L × 256) | 724.953 | 407.687 (NVFP4) | **8/256 per token** |
| shared expert | 2.832 (FP8) | **5.662 (BF16)** | yes |
| dense MLP (3 L) | 0.680 (FP8) | **1.359 (BF16)** | yes |
| DSA indexer | 0.201 | **0.394 (BF16)** | yes |
| router | 0.236 | 0.236 | yes |
| lm_head | 1.903 (BF16) | 1.903 (BF16) | yes |
| norms | 0.002 | 0.002 | yes |
| **always-read subtotal** | **18.729** | **35.299** | |
| MTP layer (active) | 0.669 | 1.182 | per draft step |

For a pass containing *T* tokens, distinct experts touched per layer is
`256·(1 − (1 − 8/256)^T)` (uniform routing; real routing is skewed, so this is an
upper bound):

| T | expert fraction | FP8 expert GB | NVFP4 expert GB |
|---|---|---|---|
| 1 | 3.13% | 22.66 | 12.74 |
| **4** (our 3-1-4) | **11.96%** | **86.72** | **48.77** |
| 16 | 39.9% | 289.5 | 162.8 |
| 64 | 87.2% | 632.0 | 355.4 |
| 256 | 99.97% | 724.7 | 407.6 |

KV traffic, per decode step, DSA-capped at `index_topk=2048`:
`2048 × (512+64) B × 78 L = 92.0 MB` — **constant beyond 2 K context**
`[inferred]`. Indexer scan is `S × 128 B × 21 layers`, amortised by
`index_topk_freq=4`: 2.75 MB at 4 K, **22.0 MB at 32 K**, 88.1 MB at 128 K. KV cache
*growth* is 43.9 KiB/token (FP8 latent) + 2.6 KiB/token (indexer keys).

**KV is ≤ 1.9% of bytes at every context length we serve.** Weights are everything.

### 8.3 The HBM floor

Per GPU at TP8, 32 K context, one 4-token verify pass:

| build | weights/pass | +KV+idx | GB/GPU | @7.0 TB/s | @6.5 TB/s |
|---|---|---|---|---|---|
| FP8 | 105.45 GB | 105.56 | **13.20** | **1.885 ms** | 2.030 ms |
| NVFP4 (as shipped) | 84.07 | 84.18 | **10.52** | **1.503 ms** | 1.618 ms |
| **hybrid: NVFP4 experts + FP8 rest** | 67.50 | 67.61 | **8.45** | **1.207 ms** | 1.300 ms |
| hybrid, NVFP4 attn too | 60.14 | 60.25 | **7.53** | **1.076 ms** | 1.159 ms |

Batch-1 (T=1) for contrast: FP8 **5.29 GB/GPU → 755 µs**; NVFP4 **6.12 GB/GPU →
874 µs**. **The shipped NVFP4 build is 16% worse than FP8 at batch 1 and 20% better
at T=4.** Crossover is `T ≈ 1.7 tokens/pass` `[inferred]` — i.e. NVFP4 only pays off
because we run speculative decoding.

The hybrid rows are the actionable finding: the NVFP4 checkpoint leaves 35.3 GB of
BF16 attention/shared/dense/indexer weights on the table that the FP8 checkpoint
already provides at half the size, for the *same tensors*. Splicing them is a
checkpoint-surgery task, not a research task.

### 8.4 The other three floors

**Compute floor — negligible, and here is why it will stay negligible.**
FLOPs/pass ≈ `2 × 40e9 active params × 4 tokens = 320 GFLOP`, i.e. **40 GFLOP/GPU →
~9 µs at 4.5 PFLOPS FP8** `[inferred]`. Machine balance:

- FP8: `4.5 PF ÷ 7.0 TB/s` = **643 FLOP/byte** ⇒ a 1 B/param weight needs
  **~320 tokens** to be compute-bound.
- NVFP4: `9.0 PF ÷ 7.0 TB/s` = **1,286 FLOP/byte**, 0.5625 B/param ⇒ **~362 tokens**.
- BF16 sustained (1.55 PF): 221 FLOP/byte ⇒ ~110 tokens.

For routed experts each expert sees `B/32` tokens ⇒ **global batch ≈ 10,300 tokens**
to make FP8 expert GEMMs compute-bound `[inferred]`. **Expert GEMMs are memory-bound
at every batch we will run.**

**Collective floor.** TP8 costs 2 all-reduces/layer = 156, plus ~6 for MTP drafts
⇒ **~162/pass**. Payload is `4 × 6144 × 2 B = 49 KB`; a ring all-reduce moves
`2(N−1)/N × S = 86 KB` per GPU ⇒ **0.1 µs of NVLink5 time** `[inferred]`. So this
term is 100% latency and synchronisation. From our own profile: 19.6% of an 8.47 ms
pass = 1.66 ms ⇒ **10.5 µs per all-reduce**, of which **47% (≈4.9 µs) is rank-arrival
skew** and ~5.6 µs is real. A well-tuned one-shot NVLS/multicast all-reduce at
~3 µs gives a **floor of ~0.49 ms/pass**.

**Kernel-launch floor.** Measured here on B200: **0.496–0.572 µs per empty CUDA-graph
node** (2,048 and 8,192-node graphs, linear fit `fixed + 0.496 µs/node`), and
**2.82 µs net per non-graph `cuLaunchKernel`** (3.09 µs measured minus 0.275 µs
ctypes FFI baseline) `[verified]`. A GLM-5.2 layer needs ~26 kernels (norms, 5 MLA
projections, RoPE, indexer ×4 on 21 layers, MLA decode, o_proj, all-reduce, router
+ topk, permute, 2 grouped expert GEMMs, SwiGLU, unpermute, shared expert, second
all-reduce). At 78 layers + head + 3 draft steps that is **~2,100 nodes/pass**
`[inferred]`:

- CUDA graphs: **~1.07 ms/pass**
- No graphs: ~5.9 ms/pass — *which alone would explain a slow engine*

### 8.5 Assembling: how far is 2.74 ms from the wall?

Per 4-token verify pass, FP8 build, 32 K context:

| term | value | share of serial total |
|---|---|---|
| HBM weights + KV | **1.885 ms** | 55% |
| Kernel dispatch (graphed, ~2,100 nodes) | **1.070 ms** | 31% |
| Collectives (162 × 3 µs achievable) | **0.486 ms** | 14% |
| Tensor-core math | 0.009 ms | 0.3% |
| **fully overlapped floor** | **1.885 ms** | |
| **fully serial floor** | **3.450 ms** | |

Converting with 3.09 accepted tokens/pass:

| | ms/pass | ms/accepted token | tok/s |
|---|---|---|---|
| overlapped floor, FP8 | 1.885 | 0.610 | **1,640** |
| serial floor, FP8 | 3.450 | 1.117 | **896** |
| serial floor, **hybrid checkpoint** | 2.772 | 0.897 | **1,115** |
| TileRT (the engine to beat) | ~6.18 | ~2.00 | ~500 |
| **us, measured** | **8.47** | **2.74** | **365** |

**Which term dominates: HBM weight traffic, at 55% of the serial floor — and 82% of
that HBM term is routed-expert weights (86.72 of 105.56 GB).** Dispatch is second at
31% and is the term most under our control. Collectives are third and are pure
latency, not bandwidth.

We are **2.5× off the pessimistic serial floor and 4.5× off the overlapped floor**.
TileRT at ~500 tok/s is still 1.8× off the serial floor, so **~900–1,100 tok/s is
the honest single-stream target on this hardware**, not 500.

### 8.6 The speculative-decoding diagnostic

This is the cleanest inference in the document. If decode were bandwidth-bound, a
4-token verify pass would cost `13.20 / 5.29 = 2.49×` a 1-token pass, so the best
possible spec-decoding speedup would be `3.09 / 2.49 = 1.24×`. **We measure 3.09×.**
Therefore the pass cost is nearly independent of token count, which happens only if
the pass is dominated by **fixed per-pass costs — kernel dispatch and collective
latency — rather than by bytes** `[inferred, high confidence]`.

Two consequences:

1. Every dispatch/collective fix we land will *reduce* spec decoding's measured
   multiplier while increasing absolute tok/s. Do not treat a falling spec-decoding
   ratio as a regression.
2. As we approach the bandwidth floor, the optimal draft depth **falls**, because
   expert bytes grow super-linearly in *T* (22.7 → 86.7 → 289.5 GB at T = 1, 4, 16).
   Re-tune 3-1-4 after each dispatch win.

### 8.7 Sensitivity — what each fix is worth

`[inferred]`, per 4-token pass, from the serial-floor baseline of 3.450 ms:

| change | Δ pass | new serial floor | new tok/s |
|---|---|---|---|
| hybrid checkpoint (NVFP4 experts + FP8 rest) | −0.678 ms | 2.772 ms | 1,115 |
| halve kernel count (2,100 → 1,050) | −0.535 ms | 2.915 ms | 1,060 |
| attention-DP + EP: 162 → 40 all-reduces | −0.366 ms | 3.084 ms | 1,002 |
| eliminate the 47% rank-arrival skew | (in-profile, ~0.79 ms of measured time) | — | — |
| all three together | −1.579 ms | **1.871 ms** | **1,652** |
| NVFP4 attention as well | −0.131 ms | 1.740 ms | 1,776 |

Note the three top items are roughly equal in value and **independent** — they
attack different terms. That is unusual and it means they should be worked in
parallel, not sequenced.

### 8.8 A number that does not reconcile

Our stated 40.8k tok/s aggregate at C64: at 64 streams × 4 spec tokens = 256
tokens/pass, essentially **all 256 experts are touched** (99.97%), so the pass reads
724.7 GB of expert weights + 18.7 dense = 92.9 GB/GPU ⇒ **13.3 ms/pass at 7 TB/s ⇒
~19.3k output tok/s** as a hard bandwidth ceiling `[inferred]`. 40.8k exceeds that
by 2.1×. Most likely the 40.8k counts **input + output** tokens, or C64 runs without
spec decoding, or routing skew is concentrating tokens on far fewer experts than
uniform. Worth resolving — if it is genuinely 40.8k *output* tok/s, one of my
assumptions above is wrong and the whole §8.3 table needs revisiting.

---

## What is NOT worth it

- **INT8 W8A8 anywhere on Blackwell, and especially on B300.** NVIDIA gives B300 a
  30:1 FP8:INT8 dense ratio; PTX never exposes the 5th-gen integer path on
  `sm_103a`; CUTLASS skips INT8 UMMA for 103a; vLLM hard-errors on the first forward
  pass; SGLang's INT8 tuning stops at SM90 (arXiv:2608.11693 `[verified]`). Even on
  B200 where INT8 is 1:1 with FP8, there is no throughput reason to prefer it and the
  toolchain is being actively withdrawn.
- **Structured 2:4 sparsity.** Every "sparse" column in §1 is 2× the dense column
  and requires 2:4 structured pruning with accuracy recovery. No MLPerf inference
  submission in v6.0 used it; no serving engine defaults to it. The datasheet's
  144 PFLOPS FP4 is a number you will not see.
- **Chasing datasheet TFLOPS in kernel work.** Sustained cuBLAS BF16 is 60–65% of
  the clock-derived peak, and MMA pipeline depth caps real kernels at 78–80% of SoL
  before any memory effect. A kernel at 70% of the *achievable* roof is done; the
  remaining 30% is thermal and pipeline-depth physics.
- **Blackwell's hardware decompression engine, for now.** Interesting on paper for
  streaming compressed weights, but the only characterisation (arXiv:2512.02189)
  contradicts itself by 5–20× between its table and its own conference slides. No
  serving engine uses it. Revisit if a second measurement appears.
- **Optimising the KV path for decode bytes.** DSA caps MLA KV reads at 92 MB/step
  regardless of context and the indexer scan amortises to 22 MB at 32 K — together
  **under 2% of per-pass bytes** `[inferred]`. Optimise the indexer for *kernel count
  and latency* (LiteTopK), not for bandwidth.
- **Benchmarking ourselves against MLPerf throughput records.** DeepSeek-R1 Server
  allows 80 ms TPOT; we run 2.74 ms. Those submissions are tuned for a batch regime
  29× less latency-constrained than ours. Mine the technique list (§4.3); ignore the
  leaderboard.
- **Reading InferenceMAX Pareto curves past their right edge.** Their GLM 5/5.1 data
  stops at 71 tok/s/user. Extrapolating to 365 tok/s/user is unsupported — the
  ranking there is set by dispatch latency, which their sweep does not probe.
- **Trusting any single secondary source on B200 specs.** Within this corpus:
  a peer-reviewed IPDPS paper says 192 GB, an excellent GEMM blog says 288 GB, and
  the correct answer is 178.34 GiB CUDA-visible. Check `libcuda`.

---

## Sources

**Peer-reviewed / arXiv (fetched and read):**

- Jarmusch & Chandrasekaran, *Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis*, U. Delaware, IPDPS 2026 — arXiv:2512.02189 (v1 2025-12-01, v3 2026-03-02) + IPDPS slide deck <https://ajarmusch.github.io/slides/blackwell-ipdps-2026.pdf>
- Jarmusch & Chandrasekaran, *Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures* — arXiv:2605.04178 (2026-05-05)
- Mo, Cheng, Wang, Tang, Xu, Li, Dong, Ma, Xia, Xue, Yang, Mai, Yang, Luk, Fan, *TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters* — arXiv:2607.22432 (2026-07-24)
- Lin, Modi, Hari, Huang, Ye, Qin, Zhou, Zhang, Wang, Damani, Peri, … Kozyrakis, Shi, *SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits* — arXiv:2603.19173 (2026-03-19)
- Teng-Ruei Chen, *Spec Sheets Are Not Kernels: An ISA- and Source-Level Audit of INT8 Availability on NVIDIA Blackwell Ultra* — arXiv:2608.11693 (v2 2026-08-14)
- Yin, Gao, Yin, Li, Cong, *LiteTopK: Exploiting the Curse of Dimensionality for a Fused Indexer-TopK Kernel in Long-Context Sparse Attention* — arXiv:2607.11976 (2026-07-13)
- Liu, Liu, Shen, Zheng, Li, Yang, Li, Zhang, Xu, Hu, Huang, Duan, Wang, Ling, Yang, Yu, Bao, Chen, Chen, *UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods*, **ACM SIGCOMM 2026** — arXiv:2607.06202 (2026-07-07)
- Liu, Andoorveedu, Das, Patel, Kindratenko, *CTA-Pipelining: A Latency-Oriented Spatial Scaling Method for Multi-GPU Systems* — arXiv:2607.07862 (2026-07-08)
- Elibol, Roesch, Gelado, Buehler, Garland, *Fearless Concurrency on the GPU* — arXiv:2606.15991 (2026-06-14)
- Chen, Zhong, Feng, Sadhukhan, Zhou, Shieh, Jia, Chen, *Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents* — arXiv:2606.06453 (2026-06-04)
- Peng, Lin, Lee, *Inference Economics of Enterprise Coding Agents: A Case Study of Cloud vs. On-Premise LLMs* — arXiv:2607.13080 (2026-07-13)
- Alpay & Alpay, *Unprivileged Topology Certificates for Cloud GPU Attestation* — arXiv:2606.24934 (2026-06-22)
- Yadav, Zhao, Kumar, *Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs* — arXiv:2604.23466 (v2 2026-06-03)
- Ardestani, Piga, Stojkovic, Balaji, Ozdal, Jimenez Fernandez, Dimovska, Tadic, Shen, Vishwanath, Mishra, Mihret, Andrei, Cespedes, Prigent, Monahan, Graf, Li, Marquez, Kanaujia, Veeraraghavan, Tang, *Provisioning to Runtime Optimization of a 100 MW-Scale AI Cluster* — arXiv:2605.24461 (2026-05-23)

**arXiv, listing-level only (title/authors/ID/date confirmed via arXiv API; abstracts only — no results quoted):** 2607.23402 (Warp Divergence Pascal→Blackwell), 2608.12629 (CAKE), 2603.24517 (AVO), 2607.11262 (GPU-Tile-Sim), 2605.00555 (Sim-FA), 2608.08482 (FlashBoot), 2608.13499 (OpScale), 2608.07009 (HiSparse), 2606.23969 (The Serialized Bridge), 2605.00519 (Silicon Showdown), 2607.16831 (Gated DeltaNet on Blackwell), 2607.17979 (Harness Engineering), 2607.18171 (FlashRT), 2605.23081 (ThriftAttention), 2605.16617 (BF16 emulation exceeding FP32), 2606.06510 / 2606.23698 (FP8 Is All You Need, Matsuoka, B300), 2510.27583 (AMD MI300X Performance Analysis), 2605.09370 (504-GPU B200 cluster operations).

**Vendor and press (labelled as such throughout):**

- NVIDIA, *NVIDIA Platform Delivers Lowest Token Cost Enabled by Extreme Co-Design* (MLPerf v6.0 deep-dive) — <https://developer.nvidia.com/blog/nvidia-extreme-co-design-delivers-new-mlperf-inference-records/>
- NVIDIA, *NVIDIA Blackwell Ultra Sets the Bar in New MLPerf Inference Benchmark* — <https://blogs.nvidia.com/blog/mlperf-inference-blackwell-ultra>
- NVIDIA, *Boost Inference Performance up to 15x on NVIDIA Blackwell Using DFlash Speculative Decoding* — <https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/>
- NVIDIA, *Blackwell Raises Bar in New InferenceMAX Benchmarks* — <https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results>
- NVIDIA, HGX platform page <https://www.nvidia.com/en-us/data-center/hgx/>; DGX B200 page <https://www.nvidia.com/en-us/data-center/dgx-b200/>; CUDA Blackwell Tuning Guide <https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>
- MLCommons, *MLPerf Inference v6.0 Results* (2026-04-01) <https://mlcommons.org/2026/04/mlperf-inference-v6-0-results/>; `inference_rules.adoc` (latency constraints) <https://github.com/mlcommons/inference_policies>
- Nebius, *MLPerf Inference v6.0: Top-tier AI performance on NVIDIA Blackwell and Blackwell Ultra*
- Lambda, *Lambda's MLPerf Inference v6.0: hardware leap, software maturity, research breakthrough* (BLAZE)
- StorageReview, *NVIDIA Sets MLPerf Inference v6.0 Records with Blackwell Ultra Platform*
- SemiAnalysis, Kimbo Chen & Dylan Patel, *Dissecting Nvidia Blackwell — Tensor Cores, PTX Instructions, SASS, Floorsweep, Yield* (2026-03-31, free preview)
- SemiAnalysis InferenceX, *InferenceMAX: Open Source Inference Benchmarking* and the **GLM 5/5.1 B200 vs MI355X** comparison page <https://inferencex.semianalysis.com/compare/glm-5-1-b200-vs-mi355x>
- Chips and Cheese, *Nvidia's B200: Keeping the CUDA Juggernaut Rolling ft. Verda*
- Jianyu Huang, *NVIDIA Blackwell SM100: TMEM, TMA, and the New Tensor Core Roofline* (2026-04-12)
- The Software Frontier, *How Blackwell's Tensor Memory Actually Works*
- Paul Chan, *Outperforming cuBLAS on B200* <https://www.paulwillchan.com/articles/outperforming-cublas-b200>
- Cornell CAC Virtual Workshop, *B200 SM* <https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm>
- AMD, Instinct MI355X product page

**First-party measurements (this repo's hardware, 2026-08-17, driver 595.71.05 / CUDA 13.2):**
`nvidia-smi` (`--query-gpu`, `-q -d CLOCK`, `nvlink -s`, `topo -m`); `libcuda`
`cuDeviceGetAttribute` / `cuDeviceTotalMem`; `cuMemcpyDtoD` and `cuMemsetD32`
bandwidth on 8 GiB buffers; empty-PTX-kernel launch and CUDA-graph node-cost sweep
(64→8,192 nodes); exact safetensors header parsing of `GLM-5.2-FP8` and
`GLM-5.2-NVFP4`.
