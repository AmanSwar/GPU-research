# GEMM at peak on SM100: CUTLASS 4, CuTe DSL, nvjet, and decode-shaped matrices

**What this is.** Dense GEMM is 37.1% of our GPU time at C1 — the single largest
kernel family — and the top kernel by time is a cuBLAS `nvjet_sm100_tst_64x8_...`.
This document is the reference for that family: what the SM100 tensor core
actually is at the PTX level, how CUTLASS 3.x/4.x and the CuTe DSL expose it,
how cuBLASLt picks kernels and how to override it, and — the part that matters
most for us — what "GEMM" even means when M is 4 tokens instead of 4096. Every
claim is labelled; the tcgen05 PTX and the cuBLASLt internals below were read
out of this machine's own toolkit and libraries, not recalled.

---

## Bottom line for our system

- **Our top kernel is BF16, not FP8 or NVFP4.** `nvjet_sm100_tst_...`: the `tst`
  tag decodes as *A/B = bfloat16, compute = fp32, D = bfloat16*, confirmed by the
  exported symbol `cublasLtTSTMatmul` in `libcublasLt.so.13` [verified, §6.2].
  12.6% of all GPU time is a BF16 GEMM in an "NVFP4 build". Quantizing the dense
  projections (q_a/kv_a/q_b/kv_b/o_proj/shared-expert/MLP) to FP8 halves the
  bytes those kernels stream, and at decode **bytes are the entire cost**.
- **Every decode GEMM we run is 30-100× below the FP8 roofline in arithmetic
  intensity.** At M=4, AI = 2M = 8 FLOP/byte against a B200 FP8 machine balance
  of ~562 FLOP/byte [inferred, §1]. Tensor-core peak is irrelevant; the only
  numbers that matter are weight bytes, CTA count, and launch count.
- **Wave quantization, not bandwidth, is the binding constraint.** At tile
  64×8 the per-rank dense GEMMs launch 4-96 CTAs onto 148 SMs — the router
  projection launches **4 CTAs** and the shared-expert gate+up launches **8**
  [inferred from measured tile shape + config.json, §7.3]. Split-K / stream-K
  and horizontal fusion are the levers, not a better mainloop.
- **The tile cuBLAS chose is already the right shape.** 64×8 = 64 output
  features × 8 tokens, matching the hand-written SM100 low-latency GEMM already
  in our tree (`cutedsl_bf16_gemm.py`, `cta_m=64, cta_n=8, cta_k=128`)
  [verified, §5.3]. We are not going to beat cuBLAS on mainloop quality at these
  shapes; we beat it by launching fewer, bigger, fused kernels.
- **At C64 with MTP 3-1-4 the target model sees M = 256** — right at the FP8
  compute-bound crossover (M ≈ 281). The capacity mode and the latency mode need
  *different kernels*, and the C64 profile has not been taken [§8].
- **`tcgen05.mma.ws` (weight-stationary, 4-slot collector, `cta_group::1` only)
  exists in PTX and is unused by anything we run** [verified from local CCCL
  headers, §2.4]. It is the instruction designed for exactly our access pattern:
  one weight matrix, many small activation batches.
- Concrete overrides that exist today and cost nothing to try: `CUBLASLT_LOG_LEVEL`
  / `CUBLASLT_HEURISTICS_LUT_FILE` (both present in `libcublasLt.so.13`
  [verified, §6.5]), a 32 MiB cuBLASLt workspace, and `nvidia-matmul-heuristics`
  for CUTLASS tile selection.

---

## 1. The only equation that matters at decode

A GEMM `C[M,N] = A[M,K] · B[K,N]` with weights resident in HBM does
`2·M·N·K` FLOP and moves `bytes_per_elem · N · K` weight bytes (activations are
negligible when M is small). Arithmetic intensity is therefore

```
AI = 2·M·N·K / (b·N·K) = 2M / b      FLOP per byte
```

— independent of N and K. It depends **only on M and the weight dtype**.

Machine balance on this box, using the max SM clock and HBM clock read from the
node itself [verified: `nvidia-smi`, B200, 183359 MiB, max SM 1965 MHz, max mem
3996 MHz, driver 595.71.05] and NVIDIA's dense peak figures [reported]:

| dtype | dense peak (B200) | HBM3e BW | machine balance | M needed to be compute-bound |
|---|---:|---:|---:|---:|
| BF16/FP16 | 2.25 PFLOP/s | 8.0 TB/s | 281 FLOP/B | M ≈ 141 |
| FP8 (e4m3) | 4.5 PFLOP/s | 8.0 TB/s | 562 FLOP/B | M ≈ 281 |
| NVFP4 (e2m1) | 9 PFLOP/s | 8.0 TB/s | 1125 FLOP/B | M ≈ 281 |

[inferred from the AI equation; peak numbers reported by NVIDIA, not measured
here.] Note the FP8 and NVFP4 crossovers coincide: halving the weight bytes and
doubling the FLOP rate cancel exactly, so the *crossover M* is a dtype-independent
property of the machine (~280 for the narrow types, ~140 for BF16).

An independent microbenchmark measured B200 tensor-core peaks of 1929.6 TFLOP/s
FP16, 3850.6 TFLOP/s FP8 and 7700.2 TFLOP/s FP4 at 96.2-96.5% of *their* stated
peak, and STREAM-triad HBM at 4.141 TB/s = 51.8% of the 8 TB/s figure
[reported: arXiv 2512.02189v2]. **That 51.8% STREAM number is the more honest
denominator for a decode roofline than 8 TB/s** — if a simple triad only reaches
4.1 TB/s, a weight-streaming GEMM will not do better without near-perfect TMA
pipelining. Treat 8 TB/s as the ceiling and ~5-6 TB/s as the realistic target.

### 1.1 GLM-5.2 decode shapes, TP8, per rank

From `/home/aman/code/weights/GLM-5.2-FP8/config.json` [verified]: `hidden_size
6144`, `q_lora_rank 2048`, `kv_lora_rank 512`, `num_attention_heads 64`,
`qk_nope_head_dim 192`, `qk_rope_head_dim 64`, `v_head_dim 256`,
`intermediate_size 12288`, `moe_intermediate_size 2048`, `n_routed_experts 256`,
`num_experts_per_tok 8`, `n_shared_experts 1`, `num_hidden_layers 78`,
`first_k_dense_replace 3`, `vocab_size 154880`, `index_topk_freq 4`.

Per MoE layer, per rank, at TP8 (M is the token count; weight bytes at 1 B/param
= FP8):

| GEMM | K | N | shard | weight bytes/rank (FP8) | CTAs at tile 64×8, M≤8 |
|---|---:|---:|---|---:|---:|
| `q_a_proj` | 6144 | 2048 | replicated | 12.58 MB | 32 |
| `kv_a_proj_with_mqa` | 6144 | 576 | replicated | 3.54 MB | 9 |
| → fused `qkv_a_proj` | 6144 | 2624 | replicated | 16.12 MB | **41** |
| `q_b_proj` | 2048 | 16384 | column | 4.19 MB | 32 |
| `kv_b_proj` | 512 | 28672 | column | 1.84 MB | 56 |
| `o_proj` | 16384 | 6144 | row (K sharded) | 12.58 MB | 96 |
| shared expert gate+up | 6144 | 4096 | column | 3.15 MB | 8 |
| shared expert down | 2048 | 6144 | row (K sharded) | 1.57 MB | 96 |
| router (BF16) | 6144 | 256 | replicated | 3.15 MB | **4** |
| **dense subtotal / layer** | | | | **42.6 MB** | ~333 CTAs in 7 launches |
| × 78 layers | | | | **3.32 GB** | |
| `lm_head` (BF16, once) | 6144 | 154880 | column | 237.9 MB | 303 |
| 8 routed experts (NVFP4) | | | | 18.87 MB/layer → 1.47 GB | |

[inferred: arithmetic from config.json; TP sharding follows the standard
DeepSeek-MLA column/row split, not read from our engine source.]

**Floor per target forward, per rank: 3.32 GB (FP8) / 6.65 GB (BF16) of dense
weights + 1.47 GB of NVFP4 experts.** At 8 TB/s that is 415 µs / 830 µs + 184 µs.
Measured GEMM family time in the trace is 3547 ms over ~979 target forwards
= **3.6 ms per forward** [inferred from the ledger's 20 s window and 365 tok/s].
Even generously, that is ~4× off the BF16 bandwidth floor and ~8× off the FP8
floor. The headroom is real and it is not in the mainloop.

---

## 2. The SM100 GEMM machine, from the PTX up

All of §2 is [verified] by reading this box's CUDA 13.3 CCCL headers under
`/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/`.
Those headers carry the exact inline-asm strings and the target lists.

### 2.1 The instruction family

`generated/tcgen05_mma.h` (6513 lines) emits, verbatim:

```
tcgen05.mma.cta_group::1.kind::f16      [%0], %1, %2, %3, PRED_enable_input_d;
tcgen05.mma.cta_group::2.kind::f16      [%0], %1, %2, %3, PRED_enable_input_d;
tcgen05.mma.cta_group::1.kind::tf32     [...]
tcgen05.mma.cta_group::1.kind::f8f6f4   [...]
tcgen05.mma.cta_group::1.kind::i8       [...]
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X  [%0], %1, %2, %3, [%4], [%5], ...
tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X      [...]
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X  [...]
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X  [...]
tcgen05.mma.cta_group::{1,2}.kind::mxf4nvf4.block_scale.block16    [...]
tcgen05.mma.cta_group::{1,2}.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale.block32 [...]
```

Operand form: `[d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem],
enable_input_d`. There is also a `tcgen05_mma_tmem_a` family taking `[a_tmem]`
instead of `a_desc` — **A can live in TMEM, B cannot**; B is always an SMEM
descriptor. The accumulator D is always TMEM.

Guard-function names in the headers give the exact target coverage:

- non-block-scaled: `SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f`
- `block_scale.scale_vec::{1X,2X,4X}`: `SM_100a, SM_103a, SM_110a` only
- `block_scale.block16 / block32`: `SM_100a/100f, SM_103a/103f, SM_110a/110f`

Two consequences for us: (a) the newer `.block16`/`.block32` spellings are the
portable ones — the `.scale_vec::` spellings are `*a` (arch-specific) only;
(b) `SM_110` is already in CUDA 13.3's headers, so kernels written today should
be templated on arch rather than hardcoding `sm_100a`.

A `disable_output_lane` variant exists in both 4-register (`cta_group::1`) and
8-register (`cta_group::2`) forms — a per-lane write mask on the accumulator.
That is the hook for masked/ragged epilogues (MoE, padded batches) without a
separate predicate pass. [verified; unused by anything in our tree.]

### 2.2 TMEM: allocation, geometry, and the load path

`generated/tcgen05_alloc.h`:

```
tcgen05.alloc.cta_group.sync.aligned.shared::cta.b32 [dst], nCols;
tcgen05.dealloc.cta_group.sync.aligned.b32 taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
```

TMEM is **256 KB per SM, addressed as 128 lanes × 512 columns of 32-bit cells**;
the 32-bit TMEM address packs lane in bits 31:16 and column in bits 15:0.
Allocation granularity is a power of two, minimum 32 columns; alloc and dealloc
must be issued by the same single warp [reported: Colfax tensor-memory tutorial;
consistent with the header signatures].

`generated/tcgen05_ld.h` (11437 lines) exposes exactly four shapes, each with
`.x{1,2,4,8,16,32,64,128}` repeat counts and an optional `.pack::16b`:

```
tcgen05.ld.sync.aligned.32x32b.x{N}[.pack::16b].b32
tcgen05.ld.sync.aligned.16x64b.x{N}[.pack::16b].b32
tcgen05.ld.sync.aligned.16x128b.x{N}[.pack::16b].b32
tcgen05.ld.sync.aligned.16x256b.x{N}[.pack::16b].b32
tcgen05.ld.sync.aligned.16x32bx2.x{N}[.pack::16b].b32
```

`.pack::16b` is the free half-precision pack on the way out of TMEM — use it
when the epilogue writes BF16/FP16 and you would otherwise burn `cvt` on the
accumulators. Each warp can only touch 32 of the 128 lanes (warp *w* sees lanes
32w..32w+31), which is why SM100 epilogues are warpgroup-shaped.

`generated/tcgen05_cp.h` — SMEM→TMEM copy, used for scale factors and for
staging A:

```
tcgen05.cp.cta_group.128x256b        [taddr], s_desc;
tcgen05.cp.cta_group.128x128b        [taddr], s_desc;
tcgen05.cp.cta_group.4x256b          [taddr], s_desc;
tcgen05.cp.cta_group.64x128b.warpx2::02_13   [taddr], s_desc;
tcgen05.cp.cta_group.64x128b.warpx2::01_23   [taddr], s_desc;
tcgen05.cp.cta_group.32x128b.warpx4          [taddr], s_desc;
```

plus `.b8x16.b6x16_p32` and `.b8x16.b4x16_p64` decompressing variants — **FP6 and
FP4 are unpacked to 8-bit lanes by the copy engine on the way into TMEM, for
free**. That is why an FP4 mainloop does not need an unpack pass.

Sync primitives: `tcgen05.fence::before_thread_sync`,
`tcgen05.fence::after_thread_sync`, `tcgen05.wait::ld.sync.aligned`,
`tcgen05.wait::st.sync.aligned`,
`tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster[.multicast::cluster].b64`,
and `tcgen05.shift.cta_group.down [taddr]` (a TMEM row shift, `SM_100a/103a/110a`
only).

### 2.3 CTA pairs (the 2-SM MMA)

`.cta_group::2` makes two adjacent CTAs execute one MMA cooperatively: CTAs
differing only in bit 0 of the cluster rank are a pair, the even one is the
leader that issues the instruction, and **each CTA holds half the accumulator in
its own TMEM and loads half of each operand tile** [reported: Colfax thread-block
clusters tutorial]. This is what makes a 256×256 MMA tile possible: 256 rows of
FP32 accumulator would not fit one SM's 256 KB of TMEM.

Cost model: 2-SM doubles the tile without doubling the operand traffic (TMA
multicast serves both), so it is the right choice whenever there are ≥2×
`cluster_m` output tiles to go around. At decode there are not — see §7.

### 2.4 `tcgen05.mma.ws` — the weight-stationary instruction nobody uses

`generated/tcgen05_mma_ws.h` [verified]:

```
tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill  [%0], %1, %2, %3, PRED_enable_input_d;
tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b0::fill  [...]
tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::fill [...]
tcgen05.mma.ws.cta_group::1.kind::i8.collector::b0::fill     [...]
```

Qualifier space: `collector::b{0,1,2,3}::{fill, use, lastuse, discard}` — a
**four-slot collector buffer** for operand B, with explicit lifetime control.
There is also a `tcgen05_mma_ws_tmem_a` form. Only `cta_group::1` exists; there
is no 2-SM weight-stationary MMA.

NVIDIA describes the use case as "repeatedly multiplying the same matrix A with
a varying matrix B" [reported: NVIDIA/cccl discussion #5669]. That is *exactly*
the decode inner loop: one weight tile, a stream of tiny activation batches. The
`fill` / `use` / `lastuse` / `discard` protocol lets you load a weight tile into
the collector once and issue several MMAs against it.

**This instruction is not used by cuBLAS's nvjet kernels we observe, nor by any
CUTLASS SM100 collective we found, nor by any kernel in our tree.** [verified by
absence in our tree; the cuBLAS claim is [inferred] — we could not disassemble
nvjet, see §6.6.] Whether it wins at M=4 is unmeasured and is the single most
interesting open question in this document.

### 2.5 Narrow types and block scaling

`cuda_fp4.h` / `cuda_fp8.h` in the toolkit define `__NV_E2M1` (the only FP4
interpretation) and `__NV_E4M3`/`__NV_E5M2` [verified].

CUTLASS's table of block-scaled types [verified: `media/docs/cpp/blackwell_functionality.md`]:

| type | SF type | SF vector (dense) | SF vector (sparse) | OCP |
|---|---|---:|---:|---|
| `mx_float8_t<any F8>` | `float_ue8m0_t` | 32 | 64 | yes |
| `mx_float6_t<any F6>` | `float_ue8m0_t` | 32 | 64 | yes |
| `mx_float4_t` | `float_ue8m0_t` | 32 | 64 | yes |
| `nv_float4_t` | `float_ue4m3_t` | 16 | 32 | **no** |

NVFP4 = E2M1 data + UE4M3 scale every 16 elements. UE4M3 max is 448 and E2M1 max
is 6, so the largest representable NVFP4 magnitude is 6 × 448 = 2688 [reported:
Colfax block-scaling tutorial]. Scale factors live in TMEM, duplicated to all 32
lane partitions, delivered by `tcgen05.cp`; the GMEM layout is a **512 B basic
block = 128 M/N × 4 K scale factors**, with M0SF0..M0SF3, M32SF0..M32SF3,
M64SF0.., M96SF0.. stored consecutively [verified: CUTLASS doc, "Scale Factor
Layouts"]. If you hand-roll an NVFP4 quantizer, that interleave is the thing you
will get wrong.

Throughput table from the same CUTLASS doc [verified as NVIDIA's claim]:

| instruction | throughput vs Hopper |
|---|---|
| `tcgen05.mma.kind::{tf32,f16,i8,f8f6f4}` | 2× Hopper equivalent |
| `tcgen05.mma.kind::mxf8f6f4.block_scale` | 2× Hopper FP8 |
| `tcgen05.mma.kind::mxf4.block_scale` | **4× Hopper FP8** |
| `tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::{2X,4X}` | **4× Hopper FP8** |

Note what this says: **block-scaled MXFP8 buys you nothing in rate over plain
FP8** — its value is purely accuracy-per-byte. Only the 4-bit kinds move the
FLOP rate.

---

## 3. CUTLASS 3.x/4.x for SM100

### 3.1 Dispatch policies (the kernel schedule)

[verified: `media/docs/cpp/blackwell_functionality.md`]

| family | 1SM policy | 2SM policy |
|---|---|---|
| legacy (tf32/f16/bf16/i8) | `KernelTmaWarpSpecialized1SmSm100` | `KernelTmaWarpSpecialized2SmSm100` |
| narrow, no block scale | same as legacy | same as legacy |
| NVFP4 | `KernelTmaWarpSpecialized1SmNvf4Sm100` | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| MXFP4 | `KernelTmaWarpSpecialized1SmMxf4Sm100` | `KernelTmaWarpSpecialized2SmMxf4Sm100` |
| MXFP8/6/4 | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` | `KernelTmaWarpSpecialized2SmMxf8f6f4Sm100` |
| sparse variants | `KernelSparseTmaWarpSpecialized1Sm*Sm100` | `KernelSparseTmaWarpSpecialized2Sm*Sm100` |
| auto | `KernelTmaWarpSpecialized1SmBlockScaledSm100` | `KernelTmaWarpSpecialized2SmBlockScaledSm100` |

Epilogue policies: `TmaWarpSpecialized{1,2}Sm`, `NoSmemWarpSpecialized{1,2}Sm`,
and for sparse narrow precision `TmaWarpSpecialized{1,2}Sm{Nvf4,Mxf4,Mxf8f6f4}`.

**`NoSmemWarpSpecialized1Sm` is the one to reach for at decode.** With M ≤ 8 the
output tile is 64×8×4 B = 2 KB; staging it through SMEM and TMA-storing it costs
more than a direct `st.global` from registers, and it frees the whole SMEM budget
for mainloop stages.

**Pingpong vs cooperative does not exist on SM100.** That distinction is a
Hopper (SM90) / SM120 concept: "Similar to Hopper's warp-group GEMM, SM120 GEMMs
support both pingpong and cooperative kernel schedules... If `KernelScheduleAuto`
is specified, `KernelTmaWarpSpecializedCooperative` will be selected by default"
[verified: CUTLASS doc, SM120 section]. On SM100 the MMA is issued by a *single
thread* into TMEM and the epilogue is a separate warpgroup reading TMEM, so the
pingpong/cooperative split is replaced by the TMEM double-buffer depth. If you
read "use pingpong on Blackwell" anywhere, it is wrong for SM100 datacenter parts.

### 3.2 Legal MMA tile shapes

[verified: CUTLASS doc Tables 4-13. Reproduced here for the cases we care about.]

Legacy types (BF16/FP16/TF32/I8), all four layouts TN/TT/NT/NN legal:

| 1SM | 2SM |
|---|---|
| 64×{64,128,192,256}×(4·MMA-K) | 128×{64,128,192,256}×(4·MMA-K) |
| 128×{64,128,192,256}×(4·MMA-K) | 256×{64,128,192,256}×(4·MMA-K) |

FP8×FP8 (`f8f6f4`), all four layouts legal at every tile: same M/N grid with
K = 128.

NVFP4×NVFP4 (`mxf4nvf4`), **TN only**:

| 1SM | 2SM |
|---|---|
| 128×128×256, 128×192×256, 128×256×256 | 256×128×256, 256×192×256, 256×256×256 |

MXFP4×MXFP4 via `mxf8f6f4` (the slower 2× path) relaxes to all four layouts at
128×{128,192,256}×128 / 256×{128,192,256}×128.

Alignment (in elements): legacy 8 (16 for i8); FP8 16; FP4/FP6 **128** for
`f8f6f4`/`mxf8f6f4`, **32** for `nvf4`/`mxf4`. Alignment is what usually kills a
tile choice, not the tile table.

Cluster rule: for a 2SM instruction the **first cluster mode must be a multiple
of 2** — `Shape<_2,[_1|_2|_4],_1>` or `Shape<_4,[_1|_2|_4],_1>` [verified].

Epilogue `PerSmTileShape_MNK` is derived: 1SM → equals the MMA tile; 2SM → MMA
tile with M halved (a 256×256 2SM MMA gives a 128×256 per-SM epilogue tile)
[verified: Table 15].

Two facts that bite at decode:

1. **The smallest legal NVFP4 MMA tile is 128×128×256.** There is no skinny
   block-scaled tile in mainline CUTLASS for SM100 — for tiny M you either pad,
   or you use a non-block-scaled `f8f6f4` tile at 64×64×128, or you write your
   own atom. (CUTLASS 4.6.0 added "tileN = 8,16 for Blackwell **SM120**
   blockscale GEMM kernels" [verified: CHANGELOG] — SM120, not SM100.)
2. Legacy BF16 does go down to 64×64, and the raw `tcgen05.mma` atom for BF16
   supports **N as any multiple of 8 up to 256** [reported: Colfax; corroborated
   by our own kernel's assertion "cta_n ∈ [8,256] step 8 (bf16 tcgen05.mma atom
   limit)" — §5.3]. The CUTLASS *collective* tile table is more restrictive than
   the hardware.

### 3.3 The collective builder, end to end

[verified: CUTLASS doc, "Building a Block Scaled Kernel"]

```cpp
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, GmemLayoutC, AlignC,
    ElementD, GmemLayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecialized2Sm,
    FusionOperation                       // optional
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, GmemLayoutA, AlignA,
    ElementB, GmemLayoutB, AlignB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelMainloopPolicy
  >::CollectiveOp;
```

Order matters: **build the epilogue first**, because the mainloop's
`StageCountAutoCarveout` needs `sizeof(CollectiveEpilogue::SharedStorage)` to
compute how many mainloop stages fit. Getting this backwards is the classic
"my kernel silently has 2 stages" bug.

`ElementAccumulator` is "always float for block scaled `tcgen05.mma`
instructions" [verified].

Block-scaled output fusion — this is how you get a GEMM that *emits* NVFP4 plus
its scale factors, which is what an MoE first-GEMM wants:

```cpp
constexpr int SFDVectorSize = 16;
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    SFDVectorSize, ElementD, ElementCompute, ElementSFD, GmemLayoutSFD, ElementC>;
```

### 3.4 Examples worth reading, by number

[verified: directory listing of `NVIDIA/cutlass/examples` + CHANGELOG entries]

| # | directory | why we care |
|---|---|---|
| 70 | `70_blackwell_gemm` | minimal SM100 FP16 GEMM — the skeleton |
| 71 | `71_blackwell_gemm_with_collective_builder` | epilogue fusion / EVT on SM100 |
| **72** | `72_blackwell_narrow_precision_gemm` | **the NVFP4 reference**; `72a_blackwell_nvfp4_bf16_gemm.cu` (FP4 in, BF16 out) and `72b_blackwell_nvfp4_nvfp4_gemm.cu` (FP4 in, FP4+SF out) |
| 73 | `73_blackwell_gemm_preferred_cluster` | preferred/fallback dynamic cluster — the fix for grids that do not divide by the cluster |
| **74** | `74_blackwell_gemm_streamk` | stream-K on SM100 — §7.4 |
| 75 | `75_blackwell_grouped_gemm` | MoE-shaped grouped GEMM |
| 78 | `78_blackwell_emulated_bf16x9_gemm` | FP32 emulation via 9 BF16 products |
| **81** | `81_blackwell_gemm_blockwise` | blockwise (DeepSeek-style 128×128 FP8 scales) |
| 82 | `82_blackwell_distributed_gemm` | GEMM split across NVLink peers |
| 86 | `86_blackwell_mixed_dtype_gemm` | W4A16-style mixed input |
| 89/90 | `89_sm103_fp4_ultra_gemm`, `90_sm103_fp4_ultra_grouped_gemm` | B300 "ultra" FP4 path |
| **91** | `91_fp4_gemv` | **FP4 GEMV** — added in CUTLASS 4.2.0 as "Blackwell SM100 fp4 gemv kernels" |
| **92** | `92_blackwell_moe_gemm` | SM100 MoE kernels using TMA + `cp.async`; 4.3.0 added a "Ragged Contiguous Grouped gemm kernel" using TMA 3D load |
| 93 | `93_blackwell_low_latency_gqa` | flash-decoding-shaped attention, paged KV (4.5.1) |
| 95 | `95_blackwell_gemm_green_context` | GEMM confined to an SM partition — relevant for compute/comm overlap |

We have not read the source of 91 or 92; the descriptions above are from the
CUTLASS CHANGELOG and directory listing [verified], not from the code.

---

## 4. CuTe, the minimum you need to write a tile

CuTe is one idea applied recursively: a **Layout** is a pair (Shape, Stride) and
*is a function from coordinates to a linear index*. Everything else is algebra on
that function. [verified: CUTLASS `media/docs/cpp/cute/01_layout.md`]

- Notation `Shape:Stride`, hierarchical: `(4,(3,6)):(1,(4,12))`. Underscore-prefixed
  integers (`_4`) are compile-time.
- `size` = product of the shape; `rank` = number of modes; `depth` = nesting
  depth; `cosize` = size of the codomain (how many distinct indices it can hit —
  this is what tells you how much SMEM a layout actually needs).
- Coordinate → index goes through `idx2crd` (colexicographic, right-to-left) then
  `crd2idx` (inner product of natural coordinate with stride).
- Sublayout / concatenation / grouping:
  ```cpp
  Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{};  // (4,(3,6)):(1,(4,12))
  Layout a1  = layout<1>(a);                      // (3,6):(4,12)
  Layout a10 = layout<1,0>(a);                    // 3:4
  Layout row = make_layout(Layout<_3,_1>{}, Layout<_4,_3>{}); // (3,4):(1,3)
  Layout b   = group<0,2>(Layout<Shape<_2,_3,_5,_7>>{});      // ((_2,_3),_5,_7):((_1,_2),_6,_30)
  ```

The four operations you actually use when writing a GEMM:

| operation | what it gives you |
|---|---|
| `composition(A,B)` | "index through B, then through A" — the basis of every re-indexing |
| `complement(A,M)` | the layout of everything A *doesn't* cover in a space of size M — how tiling is derived rather than hand-written |
| `logical_divide` / `zipped_divide` / `tiled_divide` | **tiling**: split a tensor into (tile, rest-of-tensor). `local_tile` is the sugar you call |
| `logical_product` | **replication**: lay one layout out according to another — how a thread-value (TV) layout is built |

**TV layouts** are the whole trick of CuTe-style kernels: instead of writing
`for (i...) data[thread_index_math(i)]`, you build one layout `(thread, value) →
index`, `partition` the tensor with it, and then loop over a dense local tensor.
`tiled_mma.get_slice(tid).partition_A(sA)` returns the fragment *this* thread
owns; the index math is in the layout, not in the loop body. Every SM100
epilogue in CUTLASS is `tiled_copy.partition_S/D` over a TMEM-load atom.

**Swizzles** are layouts too — `Swizzle<B,M,S>` XORs bits of the offset to break
SMEM bank conflicts. On SM100 you mostly do not hand-pick them: the TMA
descriptor carries the swizzle mode (`SWIZZLE_32B/64B/128B`), and
`cutlass::gemm::collective::detail::sm100_smem_selector` (or the DSL's
`sm100_utils.make_smem_layout_{a,b}`) picks the largest legal one. The thing to
know is the *constraint*: the swizzle atom width must divide the tile's K extent
in bytes, which is why K-tiles are 64 B/128 B multiples and why an odd N-tile
(like cuBLAS's 136) is fine but an odd K-tile is not.

---

## 5. The Python CuTe DSL

### 5.1 What it is

CUTLASS 4.x ships `cutlass.cute` — a Python-embedded DSL that JIT-compiles
through MLIR/NVVM to a cubin, with the same layout algebra and the same atoms as
C++ CuTe. It is not "CUTLASS bindings"; it is a kernel-authoring language. The
legacy Python API was renamed to `cutlass_cppgen` in 4.2.0 to make room for it
[verified: CHANGELOG].

Release timeline relevant to us [verified: CUTLASS CHANGELOG]:

| version | date | what landed |
|---|---|---|
| 4.2.0 | 2025-09-15 | SM103 (B300); example 91 FP4 GEMV; example 92 SM100 MoE; heuristics-based kernel filtering and autotuning via `nvidia-matmul-heuristics` |
| 4.3.0 | 2025-11-21 | TVM-FFI host path (lower launch overhead); PTX/CUBIN dumping; SM100 persistent dense GEMM with static scheduling; SM100 blockwise + contiguous/masked grouped blockwise; tutorial hitting "84% SOL at MNK 8K" |
| 4.3.4 | 2025-12-22 | **PDL support** (programmatic dependent launch) in the DSL |
| 4.4.0 | 2026-02-14 | CTK 13.1; AoT compilation; JAX; Python epilogue fusion for persistent dense GEMM; example 93 low-latency GQA; example 94 Ada FP8 blockwise |
| 4.5.0 | 2026-05-01 | `block_copy()` for TMA and S2T; example 95 green-context SM partitioning; 2SM MMA in mixed TMA+`cp.async` SM100 kernels |
| 4.6.0 | 2026-07-01 | `cute.compile_to`; IKET in-kernel event tracing; self-contained SASS dumping; `preferred_smem_carveout` |
| 4.7.0 | 2026-08-04 | **Primitives API** (a layer below CuTe); **Task Scheduling framework** for warp-specialized kernels; register-spill reporting and NVVM hazard detection |

The 4.7.0 additions are the ones to watch: register-spill reporting and a task
scheduler are precisely what you need to hand-write a warp-specialized decode
GEMM without reading SASS by hand.

Environment: `CUTE_DSL_CACHE_DIR` sets the JIT cache path [verified: 4.3.2].

### 5.2 The reference DSL examples

[verified: file paths exist in `NVIDIA/cutlass`]

```
examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py
examples/python/CuTeDSL/blackwell/grouped_gemm.py
examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py
examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py
examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py
```

### 5.3 What we already run: the TGV low-latency GEMM

`/home/aman/code/NotSglang/python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py`
(1455 lines) is a CuTe DSL rewrite of FlashInfer's
`include/flashinfer/gemm/tgv_gemm.cuh`, and it is the single most instructive
file in our tree for this topic. From its docstring [verified]:

```
CuTe DSL TGV BF16 GEMM (low-latency Blackwell GEMM, SM100/SM103 only).
Computes out[M, N] = x[M, K] @ weight[N, K].T (+ bias[N]) for bf16 inputs,
fp32 accumulation, bf16 output. The kernel writes M-contiguous output, so the
runner swaps A and B (gemm_fn(b.t(), a.t(), ...)) to land row-major (M, N).

  use_2cta=False -> 1x1 cluster, 1-CTA tcgen05.mma, cta_n in [8, 256] step 8
  use_2cta=True  -> 2x1 cluster, 2-CTA tcgen05.mma, cta_n in [16, 256] step 16

Warp specialization (8 warps, 256 threads/CTA; warp 3 idle):
  Warp 0    DMA_A   TMA-loads A tiles
  Warp 1    DMA_B   TMA-loads B tiles; PDL griddepcontrol.wait
  Warp 2    MMA     tcgen05.mma into TMEM; owns alloc/dealloc
  Warps 4-7 EPILOG  TMEM -> RMEM -> bf16 cast -> st.global
```

Its tactic table (`cta_m, cta_n, num_ab_stage, use_2cta`; `cta_k` fixed at 128),
29 entries, default = index 1:

```
1-CTA:  (64,8,6) (64,8,8)* (64,8,10) (64,8,12) (64,16,6) (64,16,8) (64,16,11)
        (64,32,6) (64,32,9) (64,64,7) (64,128,4)
        (128,8,6) (128,16,6) (128,32,5) (128,64,4) (128,128,3)
2-CTA:  (64,16,6) (64,16,8) (64,16,12) (64,32,6) (64,32,8) (64,32,11)
        (64,64,6) (64,64,9) (64,128,7) (128,16,6) (128,32,6) (128,64,5) (128,128,4)
                                                       (* = default tactic 1)
```

Three things to take from this [verified by reading the file]:

1. **The default production tile for a low-latency SM100 BF16 GEMM is
   `cta_m=64, cta_n=8, cta_k=128, 8 stages, 1-CTA, cluster 1×1`** — and after the
   A/B swap, `cta_m` indexes *output features* and `cta_n` indexes *tokens*.
   cuBLAS independently chose the same 64×8 (§6.3). That is convergent evidence
   that 64×8 is the right shape and that our remaining headroom is elsewhere.
2. The file's comment `1-CTA bf16 (64,8): Mma_M=(16,4)=64, Mma_N=8, Mma_K=16`
   pins the underlying atom: a 64×8×16 BF16 `tcgen05.mma`.
3. The epilogue atom it selects is `SM100_TMEM_LOAD_16dp256b1x`, and TMEM budget
   is reasoned about explicitly ("TMEM on SM100 has 128 lanes × 512 columns × 4B
   = 256KB total").

Also present and directly relevant:
`python/sglang/kernels/ops/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py` — a
CuTe DSL SM100 **persistent block-scaled NVFP4 GEMM with a fused SwiGLU +
NVFP4-requantize epilogue** (imports `cutlass.utils.blockscaled_layout`,
`cutlass.cute.nvgpu.tcgen05`, `flashinfer.fused_moe.cute_dsl.blackwell.utils`).
That is the MoE first-GEMM fusion already written; §9 discusses where the
equivalent dense-path fusion is missing.

---

## 6. cuBLAS / cuBLASLt on Blackwell, and what `nvjet` names mean

### 6.1 What nvjet is

nvJet is NVIDIA's proprietary GEMM kernel generator inside cuBLASLt, "selected
for most well-aligned shapes that can leverage TMA" and covering a very large
space of tile counts, precisions and fused epilogues [reported: NVIDIA CUDA 13.1
blog]. It replaced the older `cutlass_*` and `xmma_*` families for the common
cases; the legacy CUTLASS kernels are still in the binary (we found
`cutlass_100_simt_cgemm_64x128_16x6_*` symbols in `libcublasLt.so.13`) but only
for corner cases like complex SIMT.

### 6.2 Decoding `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`

This is normally guesswork. It does not have to be: `libcublasLt.so.13` contains
the `printf` format strings that *generate* these names.

**[verified]** — `strings` on
`/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13`
at file offsets 157651640 and 157651704:

```
nvjet_sm90_%s_%dx%d_%dx%d_%dx%d_%c_%s%s%s%s%s%s%s%s%s%s_%c%c%c
nvjet_sm%d_%s_%dx%d_%dx%d_%dx%d%s_%c_%s%s%s%s%s%s%s%s_%c%c%c
```

So the grammar is **fixed and verified**:

```
nvjet_sm<ARCH>_<TYPETAG>_<A>x<B>_<C>x<D>_<E>x<F>[<CTASUFFIX>]_<c>_<flags...>_<L1><L2><L3>
```

Field-by-field for our kernel:

| field | value | meaning | confidence |
|---|---|---|---|
| `sm%d` | `sm100` | target arch | [verified] |
| `%s` (type tag) | `tst` | **A/B = bfloat16, compute = fp32, D = bfloat16** | [verified — see below] |
| `%dx%d` #1 | `64x8` | CTA output tile, **column-major (m, n)** = 64 output features × 8 tokens | [inferred, strong] |
| `%dx%d` #2 | `64x16` | K-block 64, 16 pipeline stages | [inferred] |
| `%dx%d` #3 | `4x1` | cluster shape (x, y) | [inferred, strong] |
| `%s` (CTA suffix) | *(empty)* | `_2cta` here ⇒ 2-SM `tcgen05.mma`; empty ⇒ 1-SM | [verified: `_2cta` is a literal in the binary at offset 157651632] |
| `%c` | `v` | unknown | [unverified] |
| 8× `%s` | `bz` | option flags | [unverified — see below] |
| `%c%c%c` | `TNT` | transpose/layout triple (op(A), op(B), + one more) | [inferred] |

**The type tag is verified, not guessed.** Two independent facts from the same
binary:

1. The tag pool is a contiguous string table at offsets 157650801-157650992:
   `bsb bii bss hss tss sss_tf32` (3-char, A=B) and
   `qqhsh qqhhh qqhsq qqtst qqtsq qqsss qrhsh qrhsq qrhsr qrtst qrtsq qrtsr
   rqhsh rqhsq rqhsr rqtst rqtsq rqtsr rqsss oosss oohsh ootst ootso oohso
   rrrsr qqqsq` (5-char, mixed A/B).
2. `libcublasLt.so.13` **exports type-specialized entry points** whose names are
   the uppercase tags:
   `cublasLtTSTMatmul`, `cublasLtTSSMatmul`, `cublasLtHSHMatmul`,
   `cublasLtHSSMatmul`, `cublasLtHHHMatmul`, `cublasLtSSSMatmul`,
   `cublasLtDDDMatmul`, `cublasLtBIIMatmul`, `cublasLtBSBMatmul`,
   `cublasLtBSSMatmul`, `cublasLtCCCMatmul`, `cublasLtZZZMatmul`,
   `cublasLtKCCMatmul`, `cublasLtKCKMatmul`, `cublasLtVCCMatmul`,
   `cublasLtVCVMatmul`, `cublasLtACCMatmul` (each with matching
   `...AlgoGetHeuristic / AlgoGetIds / AlgoInit / AlgoCheck / AlgoCapGetAttribute`).

Reading the letters with cuBLAS's own conventions (`S`=fp32, `D`=fp64, `H`=fp16,
`C`/`Z`=complex, `I`=int32) and matching the 5-char forms against the flag pool
(`_Avec16UE4M3`, `_Dvec32UE8M0`, …):

| letter | type | evidence |
|---|---|---|
| `s` | fp32 | `SSS`, `sss_tf32` |
| `d` | fp64 | `DDD` |
| `h` | fp16 | `HSH`, `HHH` |
| **`t`** | **bfloat16** | `TST`/`TSS` exported; `qqtst` = FP8 in → bf16 out is the standard FP8 linear |
| `b` | int8 (byte) | `BII` (int8→int32), `BSB`, `BSS` |
| `i` | int32 | `BII` |
| `q` | fp8 e4m3 | `qq*`, and `qr`/`rq` mixed forms |
| `r` | fp8 e5m2 | `qr*`, `rq*`, `rrrsr` |
| `o` | fp4 e2m1 | `oo*`, `ootso` (fp4 in **and** fp4 out) |

⇒ **`tst` = BF16 × BF16, FP32 accumulate, BF16 output.** [verified]

The `_2cta` position is corroborated externally: a public vLLM issue reports the
observed names `nvjet_tst_128x136_64x6_4x2_v_bz_TNT` and
`nvjet_tst_192x144_64x6_2x2_2cta_v_bz_TNT` for a 272×4096×14336 BF16 GEMM
[reported: vllm-project/vllm#35467]. That case is also the best single piece of
evidence for the *first pair being (output-features, tokens)*: **M was 272 tokens
and the winning tiles were 136 and 144 — 272 = 2 × 136 exactly.** cuBLAS picks
weird N-tiles (136, 144, 160) precisely so that they divide the token count with
no remainder. That only makes sense if the second number is the batch dimension.

What we could **not** resolve: the single `%c` (`v`) and the eight `%s` flags
(`bz`). The flag string pool that *is* in the binary (offsets 157651348-157651632)
is all underscore-prefixed:

```
_auxh _auxt _auxq _auxr _relubias _gelubias _reluaux _reluauxbias _geluaux
_geluauxbias _splitK _ptrBatch _Bblk128 _Bvec128 _ovscale _Avec128 _Ablk128
_algo2 _coopA _coopB ssched_ _Dvec32UE8M0 _Dvec16UE4M3 _Cvec32UE8M0
_Cvec16UE4M3 _Bvec32UE8M0 _Bvec16UE4M3 _Avec32UE8M0 _Avec16UE4M3 _2cta
```

`bz` is not in it and the literal `\0bz\0` does not appear in the file, so `b`
and `z` come from a different (unlocated) table. **Do not trust anyone's
explanation of `_v_bz_`, including a future version of this document, unless it
cites a source.** [unverified]

That flag pool is itself useful: it tells you cuBLAS *can* fuse `relu`/`gelu` +
bias + aux output, can do `_splitK`, can emit block-scaled A/B/C/D with UE8M0 or
UE4M3 scale vectors, and marks `_coopA`/`_coopB` (cooperative operand loads) and
`_algo2`. All of that is reachable through cuBLASLt epilogue descriptors.

### 6.3 What the name tells us about our own workload

`nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, 1206 ms = 12.6% of GPU time:

- BF16 in/out ⇒ **the dense projections in our NVFP4 build are not quantized.**
- tile 64×8, 1-SM, cluster 4×1 ⇒ grid = `ceil(N_rank/64) × ceil(M/8)`. With M ≤ 8
  that is `ceil(N_rank/64)` CTAs — see the last column of the §1.1 table.
- 16 stages at K-block 64 with BF16: per stage `64×64×2 = 8 KB` (A) + `8×64×2 =
  1 KB` (B) = 9 KB, × 16 = 144 KB, inside the 227 KB opt-in SMEM. Self-consistent
  [inferred].

### 6.4 Heuristics: query, inspect, override

The public flow [verified: `cublasLt.h` from this box's CUDA 13 wheel]:

```c
cublasLtMatmulPreferenceCreate(&pref);
cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
cublasLtMatmulAlgoGetHeuristic(h, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref,
                               requestedAlgoCount, results, &returnedResults);
```

`cublasLtMatmulPreferenceAttributes_t` [verified, with enum values]:

| attribute | value | use |
|---|---:|---|
| `CUBLASLT_MATMUL_PREF_SEARCH_MODE` | 0 | `CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID` restricts the search to one algo id |
| `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` | 1 | **the single most impactful knob**; split-K kernels are excluded if it is 0 |
| `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` | 3 | allow/forbid split-K reduction schemes |
| `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_{A,B,C,D}_BYTES` | 5-8 | |
| `CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT` | 9 | cap on `wavesCount` — **directly the wave-quantization knob** |
| `CUBLASLT_MATMUL_PREF_IMPL_MASK` | 12 | restrict implementation families |

`cublasLtMatmulHeuristicResult_t` [verified] returns `{algo, workspaceSize,
state, wavesCount}` where "`wavesCount` value of 1.0f suggests that when kernel
is launched it will fully occupy the GPU". **Log `wavesCount` for every decode
GEMM in the model — anything ≪ 1.0 is a kernel that is leaving SMs idle**, and
per §1.1 most of ours will be 0.03-0.65.

Explicit algorithm construction (bypassing or auditing the heuristic):
`cublasLtMatmulAlgoGetIds` → `cublasLtMatmulAlgoInit` →
`cublasLtMatmulAlgoCapGetAttribute` (what this algo supports) →
`cublasLtMatmulAlgoConfigSetAttribute` → `cublasLtMatmulAlgoCheck` (validity +
predicted workspace) → `cublasLtMatmul`.

`cublasLtMatmulAlgoConfigAttributes_t` [verified, with values]:

```
CUBLASLT_ALGO_CONFIG_ID = 0
CUBLASLT_ALGO_CONFIG_TILE_ID = 1              // CUBLASLT_MATMUL_TILE_*
CUBLASLT_ALGO_CONFIG_SPLITK_NUM = 2
CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME = 3
CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING = 4
CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION = 5
CUBLASLT_ALGO_CONFIG_STAGES_ID = 6            // CUBLASLT_MATMUL_STAGES_*
CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID = 7       // MMA884/1684/1688/16816
CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID = 8     // CUBLASLT_CLUSTER_SHAPE_*
```

The enums confirm the nvjet-name reading [verified: `cublasLt.h`]:

- `CUBLASLT_MATMUL_TILE_*` is `<m>x<n>`, and it includes **`8x8`, `8x16`, `16x8`,
  `64x8` (=10), `8x128`, `8x192` … `8x768`, `16x64` … `16x768`, `24x64` …,
  `32x…`, `40x…`, `48x…`, `56x…`, `72x…`, `80x…`** — i.e. cuBLASLt has a
  first-class notion of tiles whose **second dimension is the small one**,
  exactly the decode shape.
- `CUBLASLT_MATMUL_STAGES_*` is `<k-block>x<stages>`: `16x1..16x6, 32x1..32x6,
  64x1..64x6, 128x1..128x6, 32x10, 8x{3,4,5}, 16x10`, plus `{8,16,32,64,128,256,
  768}xAUTO`. This is the convention that makes `64x16` in the nvjet name read as
  "K-block 64, 16 stages".
- `CUBLASLT_CLUSTER_SHAPE_*` is `<x>x<y>x<z>`, including `4x1x1` (=4), `2x2x1`,
  `4x2x1`, `4x4x1`, and odd sizes `3x1x1`, `5x1x1`, `7x1x1`, up to `16x1x1`.
- `CUBLASLT_MATMUL_INNER_SHAPE_*` only has `MMA884/MMA1684/MMA1688/MMA16816` —
  i.e. this enum was never extended for `tcgen05`, which is one more sign that
  nvjet kernels are outside the classic algo-id space.

### 6.5 Environment variables that actually exist in CUDA 13.3

[verified: `strings libcublasLt.so.13`, exact names]

| variable | notes |
|---|---|
| `CUBLASLT_LOG_LEVEL` | 0-5; with `CUBLASLT_LOG_FILE` this prints the selected algo per call — **the cheapest way to see which nvjet kernel a shape gets** |
| `CUBLASLT_LOG_MASK` | bitmask of message classes |
| `CUBLASLT_LOG_FILE` | supports `%i` pid substitution in practice |
| `CUBLASLT_HEURISTICS_CACHE_CAPACITY` | entries in the heuristic memo cache; also settable via `cublasLtHeuristicsCacheSetCapacity()` |
| **`CUBLASLT_HEURISTICS_LUT_FILE`** | a lookup-table file that overrides heuristic choices; the library exports `cublasLtHeuristicLutSerializeEntry` |
| `CUBLASLT_NVTX_LEVEL` | NVTX ranges around cuBLASLt calls — useful for the nsys pipeline we already have |
| `CUBLASLT_DISABLE_CPU_INSTRUCTIONS_MASK` | host-side ISA gating |
| `CUBLAS_AUTOTUNING_CACHE_CAPACITY` | pairs with the documented experimental `CUBLAS_GEMM_AUTOTUNE`, which "benchmarks a number of available algorithms and chooses the optimal one... cached in the cublas handle" [reported: cuBLAS docs] |
| `CUBLAS_WORKSPACE_CONFIG` | `:16:8` / `:4096:8` for deterministic multi-stream behaviour [reported: cuBLAS docs] |
| `NVIDIA_TF32_OVERRIDE=0` | kills TF32 |
| `CUBLAS_EMULATION_STRATEGY`, `CUBLAS_EMULATE_SINGLE_PRECISION`, `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT` | BF16x9 / fixed-point FP32/FP64 emulation |
| `CUBLAS_FORCE_XMMA_KERNEL_INIT` | forces the legacy xmma family (debug) |

`CUBLASLT_HEURISTICS_LUT_FILE` is the interesting one: it is the sanctioned way
to pin a kernel choice per shape without patching the framework. We have not
found documentation of its file format — **the format is not sourced**; discovering
it via `cublasLtHeuristicLutSerializeEntry` is a bounded experiment.

**Workspace.** NVIDIA's recommendation is **32 MiB for Hopper (sm90) and both
Blackwell families (sm10x, sm12x)**, 4 MiB otherwise [reported: cuBLAS docs
table]. PyTorch's default `CUBLAS_WORKSPACE_CONFIG` is smaller than that; if the
workspace is too small, split-K algorithms are simply *not returned by the
heuristic*, which is one silent way to lose a decode GEMM.

### 6.6 The heuristic is not trustworthy at our shapes

Publicly measured on B200, BF16, M=272, N=4096, K=14336 [reported:
vllm-project/vllm#35467]:

| kernel | time | vs best |
|---|---:|---:|
| `nvjet_tst_256x144_*` (TN) | 35.9 µs | best TN |
| best (NN layout) | 34.1 µs | best overall |
| `nvjet_tst_128x136_64x6_4x2_v_bz_TNT` (heuristic pick) | 42.6 µs | +25% |
| `nvjet_tst_192x144_64x6_2x2_2cta_v_bz_TNT` (what vLLM ran) | 48.6 µs | **+43%** |
| theoretical minimum | 16.6 µs | — |

The report notes the heuristic ranked the 128×136 tile **7th** in its own list
and then selected it anyway, and that the 192×144_2cta kernel vLLM ended up with
"doesn't appear in the documented heuristic pool". Two lessons: (1) always
enumerate `requestedAlgoCount > 1` and time them rather than taking result[0];
(2) **layout matters as much as tile** — the NN variant was ~2 µs faster purely
by avoiding a transpose.

For reference on the compute-bound end, cuBLAS on B200 reaches 1671.8 TFLOP/s at
8192³ and 1517.5 at 16384³ [reported: arXiv 2604.23466v1], and a hand-written
BF16 kernel reached 106.3% of cuBLAS at 8192³ (1648.9 vs 1551.0 TFLOP/s) using
cluster launch control and Hilbert-curve tile ordering [reported:
paulwillchan.com]. So cuBLAS is *not* an unbeatable ceiling even at square
shapes — but the margin there is ~6%, versus ~4× at our shapes.

---

## 7. The decode regime: M = 1..8 (for us, M = batch × 4)

### 7.1 What M actually is for us

EAGLE `3-1-4` = `--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4` [verified: `server_args.py` field names +
personal_docs]. Per decode iteration:

| forward | count | M per sequence | M at C1 | M at C64 |
|---|---:|---:|---:|---:|
| draft (MTP layer only) | 3 | 1 | 1 | 64 |
| target (78 layers) | 1 | 4 | **4** | **256** |

So the latency mode lives at M=4 and the capacity mode at M=256. Those are on
opposite sides of the FP8 crossover (§1). Any "which GEMM should we use"
statement that does not name the mode is meaningless.

### 7.2 The three things that cost time at M=4

1. **Weight bytes.** Irreducible except by quantizing (§1.1: 6.65 GB BF16 →
   3.32 GB FP8 → 1.66 GB NVFP4 per rank per forward).
2. **CTA count / wave quantization.** A GEMM that launches 32 CTAs uses 22% of
   the machine no matter how good its mainloop is.
3. **Launch count and dependency chain.** ~608 dense-GEMM launches per target
   forward per rank [inferred from the trace: 596,088 launches / ~979 forwards],
   averaging ~6 µs each. Even inside a CUDA graph, each is a separate
   dependency-ordered kernel with its own ramp-up and drain.

### 7.3 Wave quantization, quantified

At tile 64×8 the number of CTAs is `ceil(N_rank/64)` for M ≤ 8. Against 148 SMs:

| GEMM | CTAs | waves | SMs idle |
|---|---:|---:|---:|
| router | 4 | 0.03 | 97% |
| shared gate+up | 8 | 0.05 | 95% |
| `q_b_proj` | 32 | 0.22 | 78% |
| fused `qkv_a_proj` | 41 | 0.28 | 72% |
| `kv_b_proj` | 56 | 0.38 | 62% |
| `o_proj`, shared down | 96 | 0.65 | 35% |
| `lm_head` | 303 | 2.05 | (tail wave 5%) |

[inferred: arithmetic from the measured tile shape and config.json.]

The per-SM HBM read rate needed to hit the roofline with only 32 CTAs is
8 TB/s ÷ 32 = 250 GB/s per SM. A single SM cannot sustain that — the fair share
at full occupancy is 8000/148 ≈ 54 GB/s, and per-SM outstanding-request limits
put the practical single-SM ceiling in the same order. **So at 32 CTAs the
achievable bandwidth is roughly 32 × 54 GB/s ≈ 1.7 TB/s, i.e. ~21% of HBM
peak.** 8.4 MB of BF16 `q_b_proj` weights ÷ 1.7 TB/s ≈ 4.9 µs — which lands on
top of the ~6 µs average we measure. [inferred; the per-SM bandwidth number is
the weak link and is the first thing to measure — see §11.]

That is the whole story: **our decode GEMMs are CTA-starved, not
bandwidth-starved, and the fix is more CTAs per GEMM or fewer, larger GEMMs.**

### 7.4 Split-K and stream-K

Split-K: partition the K loop across `s` CTAs and reduce. Multiplies CTA count by
`s` at the cost of a reduction (deterministic via a workspace + fixup kernel, or
non-deterministic via atomics). For our shapes K is large (6144, 16384/8, 2048)
so split-K is nearly free in efficiency terms:

| GEMM | K | CTAs now | split-K needed for ≥148 CTAs |
|---|---:|---:|---:|
| router | 6144 | 4 | 37 (K-chunk 166) |
| shared gate+up | 6144 | 8 | 19 (K-chunk 323) |
| `q_b_proj` | 2048 | 32 | 5 (K-chunk 410) |
| fused `qkv_a_proj` | 6144 | 41 | 4 (K-chunk 1536) |

[inferred.] In cuBLASLt this is reachable *without writing a kernel*:
`CUBLASLT_ALGO_CONFIG_SPLITK_NUM` + `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME`, and
the nvjet flag pool contains `_splitK` so the kernels exist. It requires a
non-zero workspace (§6.5).

Stream-K [verified title/authors: Osama, Merrill, Cecka, Garland, Owens,
"Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix
Multiplication on the GPU", arXiv:2301.03598] partitions "an even share of the
aggregate inner loop iterations among physical processing elements" instead of
partitioning output tiles. It is the principled fix for wave quantization:
one tile configuration per precision, near-perfect load balance at any geometry.
Reported peak speedups 14× and 6.7× over CUTLASS/cuBLAS across 32,824 problem
geometries. CUTLASS ships `74_blackwell_gemm_streamk` for SM100, and 4.7.0
included "StreamK heuristic improvements" [verified: CHANGELOG].

**Caveat for us:** stream-K's advantage is largest when the tile count is a
*non-multiple* of the SM count with a large remainder. Ours is not a remainder
problem — it is a "there are only 32 tiles total" problem. Stream-K with a
64×8 tile over N=2048, K=6144 would give each of 148 CTAs a K-slice, which is
exactly split-K with automatic sizing. Worth trying via example 74 rather than
hand-rolling.

### 7.5 When to leave the tensor cores entirely

At M=1..8 the MMA is a rounding error in the time budget; what matters is issuing
loads. A CUDA-core GEMV with 32-byte vector loads can be strictly better because
it has no TMEM allocation, no mbarrier, no epilogue warpgroup, and no tile
quantization in N.

We already ship one. `python/sglang/kernels/ops/gemm/tiny_gemm.py` +
`python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh` [verified]:

```cpp
// bf16 x bf16 -> fp32 fused multiply-add, Blackwell path
SGL_DEVICE float fma_f32_bf16(bf16_t a, bf16_t b, float acc) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
  asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(result) : "h"(a_bits), "h"(b_bits), "f"(acc));
#else
  return fmaf(cast<fp32_t>(a), cast<fp32_t>(b), acc);
#endif
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t N_SPLIT, typename OutT, bool kUsePDL>
__global__ __launch_bounds__(K / kTinyNGemmVecSize, 1)  // 1 block per SM
void tiny_n_gemm_kernel(OutT* out, const bf16_t* x, const bf16_t* w) { ... }
```

Design points worth stealing [verified by reading the file]:

- **One block per SM** (`__launch_bounds__(..., 1)`), block size = `K /
  vec_elems`, i.e. **the K dimension is mapped onto threads** and each thread
  owns one vector of the reduction. No tiling in K at all.
- `vec_elems = 16` bf16 = **32 bytes per load** on Blackwell with CUDA ≥ 12.9,
  8 otherwise. CUDA 13.3's `vector_types.h` defines the 32-byte-aligned types
  `long4_32a`, `ulonglong4_32a`, `double4_32a` [verified] — 256-bit global loads
  are a real Blackwell feature and this kernel is built on them.
- **Weights are prefetched before the PDL wait**: `wv[n].load(...)` then
  `PDLWaitPrimary<kUsePDL>()`. The weight address does not depend on the previous
  kernel's output, so the weight loads start during the *predecessor* kernel.
  This is the single best trick in the file and it generalizes to every
  weight-stationary decode kernel.
- `split_n` is chosen as "the smallest divisor of N whose `N / split_n` blocks
  fit in one wave" — **explicit wave-quantization-aware tuning**, with an
  `M * split_n <= blockSize` constraint so the final reduce has one thread per
  output element.
- A second variant `tiny_k_gemm` handles small-K/large-N (K/8 lanes of one warp
  reduce one output column; requires K/8 a power of two ≤ 32, i.e. K = 128/256) —
  that is the DSA-indexer and router shape.
- `_MAX_M_DEFAULT = 16`: the intended validity window.

Related upstream kernels that do the same thing for DeepSeek-shaped models, all
present in our tree as JIT wrappers [verified]:
`dsv3_fused_a_gemm.py` ("min-latency" fused QKV-A projection, "hd_in a multiple
of 256, hd_out a multiple of 16, **num_tokens 1-16**, bfloat16") and
`dsv3_router_gemm.py` ("num_experts in {256, 384}, hidden_dim a multiple of 1024,
**num_tokens 1-16**"), plus a CuTe DSL variant `cutedsl_dsv3_fused_a_gemm.py`.
**GLM-5.2 is MLA-shaped with 256 experts and hidden 6144 — both of those
constraints are satisfied by our model.** Whether these paths are wired up for
`glm_moe_dsa` is not verified here and is a cheap thing to check.

### 7.6 Weight-stationary vs activation-stationary

At decode the weights are ~10⁴× larger than the activations, so **everything
should be weight-stationary**: stream the weight tile through the mainloop
exactly once and keep the (tiny) activation resident in registers/SMEM/TMEM for
the whole kernel. Concretely:

- Load `x[M,K]` once per CTA into registers (M ≤ 8, K-slice) — `tiny_gemm` does
  exactly this.
- Never re-read weights across the M dimension: with M ≤ 8 there is only one
  n-tile, so this is automatic at tile 64×8.
- Across *consecutive GEMMs on the same weights* (the 3 draft steps + 1 target
  step of a 3-1-4 iteration all touch the MTP layer's weights) there is reuse to
  be had in L2 (126 MB on B200 [unverified — L2 size not sourced here]) but not
  across the 78 target layers, whose working set (3.3-6.7 GB) dwarfs any cache.
- `tcgen05.mma.ws` with `collector::b0::fill` / `::use` / `::lastuse` is the
  hardware expression of this idea (§2.4) and is untried.

Activation-stationary only makes sense in prefill, where the activation tile is
the reused object.

### 7.7 Fusing consecutive GEMMs

Two distinct opportunities:

- **Horizontal (independent GEMMs, same input).** `q_a_proj` and
  `kv_a_proj_with_mqa` both read the same post-norm hidden state; concatenating
  their weights into one `[6144, 2624]` matrix turns 2 launches × (32, 9) CTAs
  into 1 launch × 41 CTAs. Same for `gate_proj`/`up_proj` → `gate_up_proj`
  ([6144, 4096] per layer, [6144, 512] per rank). SGLang already names both
  (`fused_qkv_a_proj_with_mqa`, gate+up packing); the win is launch count and
  wave utilization, not FLOPs.
- **Vertical (dependent GEMMs).** `gate_up → SwiGLU → down` cannot be fused into
  one GEMM, but the *activation and requantization* between them can be folded
  into the first GEMM's epilogue — which is exactly what
  `nvfp4_gemm_swiglu_nvfp4_quant.py` does for the MoE path. The dense/shared-expert
  path has no such kernel in our tree.

A third, more aggressive option is a **persistent megakernel** that keeps one
CTA-per-SM alive across the whole layer and walks the GEMM sequence with
`griddepcontrol`-style handoffs, eliminating 7 launches per layer. That is what
TileRT's "tile-level overlap inside individual operators" amounts to
[reported: TileRT positioning; the mechanism is [inferred]]. CUTLASS's PDL
support (`4.3.4`) and green contexts (example 95) are the building blocks.

---

## 8. Prefill, and where the crossover really is

Prefill with chunked-prefill at, say, 2048 tokens per chunk gives M = 2048 per
rank — 7× past the FP8 crossover. There the rules invert completely:

| | decode (M ≤ 8) | prefill (M ≥ 512) |
|---|---|---|
| bound by | weight bytes, CTA count, launch count | tensor-core FLOPs |
| tile | 64×8 (or 64×16), 1-SM | 128×256 / 256×256, **2-SM** |
| cluster | 1×1 or 4×1 | 2×1×1, 2×2×1, 4×4×1 with TMA multicast |
| epilogue | `NoSmemWarpSpecialized1Sm`, direct `st.global` | `TmaWarpSpecialized2Sm` |
| scheduler | static, one wave | persistent + stream-K, tile swizzle for L2 |
| dtype leverage | quantize weights (bytes) | quantize both (FLOPs) |
| split-K | often essential | harmful |

**Where the crossover is, precisely:** M ≈ 141 for BF16 and M ≈ 281 for
FP8/NVFP4 on the roofline. In practice the switch should happen earlier, because
a 2-SM 256×256 tile needs ≥ 296 output tiles to fill the machine
(`ceil(N/256) × ceil(M/256) ≥ 148` CTA-pairs); for N_rank = 2048 that means
M ≥ 256×148/8 — unreachable. **For our N values, the 2-SM large-tile regime is
only entered by `lm_head` and by prefill.** [inferred]

The practical consequence: our C1 latency mode and our C64 capacity mode should
not be running the same GEMM configuration, and today they do, because both go
through cuBLASLt's per-shape heuristic — which at M=256, N=2048 will pick
something in between and be wrong for both. That is a measurement we do not have
(candidate H in the ledger: "profile at C64").

The compute-bound end also has an emulation escape hatch worth knowing:
`CUBLAS_COMPUTE_32F_EMULATED_16BFX9` [verified: symbol in `libcublas.so.13`] and
CUTLASS example 78 implement FP32 as 9 BF16 products — irrelevant for LLM
inference but relevant if we ever need FP32 accuracy at tensor-core rates.

---

## 9. Fusion opportunities in a GLM-5.2 block

Ranked by what the profile says is expensive.

### 9.1 Already fused (do not reinvent)

From `NotSglang/.claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md`
[verified: local file], the following families already exist upstream and should
be checked as "off/unsupported" before being proposed as new work:

- `fused_add_rmsnorm` (residual + RMSNorm), `silu_and_mul` (SwiGLU),
  `fused_qkv_a_proj_with_mqa` (MLA A-projection packing).
- **FlashInfer unified `allreduce_fusion(..., pattern =
  AllReduceFusionPattern.kARResidualRMSNorm, ...)`** — wired in
  `python/sglang/srt/layers/flashinfer_comm_fusion.py` and
  `layernorm.py::forward_with_allreduce_fusion`. Our ledger records this as
  **off in every measurement so far**.
- FlashInfer norm/quant epilogues: `rmsnorm_quant`, `fused_add_rmsnorm_quant`,
  `rmsnorm_fp4quant`, `add_rmsnorm_fp4quant`
  (`flashinfer/cute_dsl/{rmsnorm_fp4quant,add_rmsnorm_fp4quant}.py`) — these
  produce NVFP4 **plus its scale factors** directly out of the norm, removing a
  separate quant pass. Our profile shows `quant` at 2.4% and `norm` at 0.5%; the
  bigger win is that they remove a round-trip of the activation through HBM.
- MoE: `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4` and
  `blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4` (FlashInfer CuTe
  DSL) fuse gather+GEMM1+SwiGLU and finalize+unpermute+scatter-reduce.

### 9.2 The allreduce + RMSNorm + quantize fusion

Our TP8 layer emits, per layer, two all-reduces (attention output, MLP/MoE
output), each followed by a residual add, an RMSNorm, and (in a quantized build)
a quantize. Unfused that is 4 kernels and 3 extra HBM round-trips of the
`[M, 6144]` activation per all-reduce. At M=4 the tensor is 48 KB — trivially
small, so the cost is **entirely launch overhead and dependency latency**, which
is precisely what a 19.6%-collective / 47%-skew profile is made of.

`kARResidualRMSNorm` collapses all-reduce + residual + norm into one kernel. The
NVFP4 variants (`add_rmsnorm_fp4quant`) add the quantize. Turning this on is
candidate B1 in the ledger and is free.

### 9.3 What is missing on the dense path

The MoE path has a fused NVFP4 GEMM+SwiGLU+quant kernel; the **shared expert and
the 3 dense MLP layers do not**. Same for the attention block: `q_b_proj` and
`kv_b_proj` are separate launches from the same `q_a`/`kv_a` output.

Concretely missing, in expected-value order [inferred]:

1. **Quantize the dense projections.** The `tst` finding says they are BF16.
   Going to FP8 e4m3 halves 3.3 GB/forward/rank of traffic; going to NVFP4
   quarters it. This is a config/checkpoint change, not a kernel.
2. **A dense `gate_up → SwiGLU → quant` epilogue fusion**, mirroring the MoE one.
3. **A fused `q_b_proj ‖ kv_b_proj`**: both read the same 2048/512-dim latents;
   they are not concatenable directly (different K), but they can be issued as a
   grouped GEMM (CUTLASS example 75 / 92 shapes) in one launch.
4. **Residual-add fusion into the GEMM epilogue** (`C` operand with `beta=1`)
   rather than a separate elementwise kernel — elementwise is 3.7% of GPU time
   across 146,293 launches, i.e. ~2.4 µs each, essentially all launch overhead.

---

## 10. Decision table

Given `(M, N, K, dtype, arch=sm100)`, per rank. "N" is the per-rank output width.

| M | N | dtype | choose | why |
|---|---|---|---|---|
| 1 | any | BF16/FP8 | CUDA-core GEMV: `tiny_n_gemm` (large N) or `tiny_k_gemm` (K ≤ 256) | no tensor core can help at AI = 2 FLOP/B; avoid TMEM/mbarrier setup entirely |
| 1-16 | ≤ 1024 | BF16 | `tiny_n_gemm_bf16` with wave-aware `split_n`; or `dsv3_router_gemm` for the router | router at N=256 is 4 CTAs with a tiled kernel |
| 1-16 | 1k-8k | BF16 | cuBLASLt `tst` 64×8 **with split-K enabled** (non-zero workspace, `CUBLASLT_ALGO_CONFIG_SPLITK_NUM`), or the TGV CuTe DSL kernel tactic 1 | 64×8 is already the right tile; the missing ingredient is CTA count |
| 1-16 | 1k-8k | FP8 | cuBLASLt `qqtst`/`qqtsq` 64×8, or CUTLASS `KernelTmaWarpSpecialized1SmSm100` at 64×64×128 + `NoSmemWarpSpecialized1Sm` | FP8 halves the bytes; `f8f6f4` allows a 64-wide tile, block-scaled does not |
| 1-16 | any | NVFP4 | **no mainline SM100 skinny block-scaled tile exists** — smallest is 128×128×256. Either pad M to 128 (wasting 97% of the MMA, which is free since you are bandwidth-bound) or use `f8f6f4` on upconverted data | see §3.2 |
| 1-16 | ≥ 16k (`lm_head`) | BF16 | persistent GEMM with tile 64×128 or 128×128; 303+ CTAs so wave quantization is a non-issue | this is the one decode GEMM that fills the machine |
| 17-128 | any | any | cuBLASLt heuristic, but **enumerate ≥ 8 candidates and time them**; log `wavesCount` | this is where the heuristic is least reliable (§6.6) |
| 128-512 | any | FP8/NVFP4 | CUTLASS 1-SM 128×{128,256} (`f8f6f4`) or 128×128×256 (`nvf4`); stream-K if tiles don't fill waves | crossing into compute-bound |
| ≥ 512 | ≥ 2048 | FP8/NVFP4 | CUTLASS **2-SM** 256×256, cluster `2×2×1`/`4×4×1`, `TmaWarpSpecialized2Sm`, persistent + tile swizzle | full multicast and TMEM utilization |
| ≥ 512 | any | BF16 | cuBLASLt; hand-written only if you also implement CLC + L2-aware tile order (worth ~6%) | [reported: paulwillchan.com] |
| grouped/MoE, any M | — | NVFP4 | FlashInfer CuTe DSL `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4`, or CUTLASS example 92 | fused gather+GEMM+SwiGLU beats any unfused chain |
| M ≡ batch·4 at C64 = 256 | 1k-8k | FP8 | **re-tune**: this is the crossover; neither the decode nor the prefill config is right | unmeasured — §11 |

Layout rules that override all of the above:
- NVFP4/MXFP4 via `mxf4`/`mxf4nvf4` is **TN only** (A row-major, B column-major).
  If your weights are stored the other way you pay a transpose or fall back to
  `mxf8f6f4` at half the rate.
- FP4/FP6 operands need **128-element alignment** for `f8f6f4`/`mxf8f6f4`, 32 for
  `nvf4`/`mxf4`.
- NN layout can be ~2 µs faster than TN at medium M purely by avoiding a
  transpose [reported: vllm#35467] — always test both.

---

## 11. Open questions / what to measure on the box

1. **Per-SM HBM read bandwidth on B200.** The whole §7.3 argument rests on
   ~54 GB/s/SM. Measure it: a grid-stride copy kernel swept over CTA count,
   1 → 148, reporting GB/s. One afternoon, and it converts §7.3 from [inferred]
   to [verified].
2. **Is `tcgen05.mma.ws` faster at M ≤ 8?** Write two 64×8 BF16 kernels differing
   only in `tcgen05.mma` vs `tcgen05.mma.ws` + `collector::b0::{fill,use}` and
   time them. No source anywhere reports this comparison.
3. **What are `_v_` and `bz` in the nvjet name?** Not sourced. Recoverable by
   `CUBLASLT_LOG_LEVEL=5` over a sweep of shapes/epilogues and diffing names.
4. **The `CUBLASLT_HEURISTICS_LUT_FILE` format.** Undocumented; the export
   `cublasLtHeuristicLutSerializeEntry` suggests it is generatable.
5. **Does split-K actually get selected for our shapes if the workspace is
   32 MiB?** Set `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` to 32 MiB, enumerate
   16 heuristic results for `(M=4, N=2048, K=6144, bf16)`, and print
   `wavesCount` + `algo` for each.
6. **Are `dsv3_fused_a_gemm` / `dsv3_router_gemm` / `tiny_gemm` reachable for
   `glm_moe_dsa`?** The shape constraints fit GLM-5.2 exactly (hidden 6144,
   256 experts, tokens 1-16), but the dispatch was not traced.
7. **C64 profile.** Everything about the M=256 regime in §8 is unmeasured.
8. **B200 L2 size and per-partition bandwidth.** Cited nowhere in this document
   because we could not source it; it matters for whether draft-step weight reuse
   is real.
9. **Nsight Compute on `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`.** We have its
   cost (12.6%) and its shape but no `dram__bytes.sum`, no achieved occupancy,
   no `sm__throughput`. Candidate E in the ledger. Without it, "4× headroom" is
   arithmetic, not evidence.

---

## Sources

### Read on this machine (primary)

- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_mma.h`
- `.../generated/tcgen05_mma_ws.h` — weight-stationary MMA, collector qualifiers
- `.../generated/tcgen05_ld.h`, `.../generated/tcgen05_st.h`
- `.../generated/tcgen05_alloc.h`, `.../generated/tcgen05_cp.h`
- `.../generated/tcgen05_fence.h`, `.../generated/tcgen05_wait.h`, `.../generated/tcgen05_commit.h`, `.../generated/tcgen05_shift.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda_fp4.h`, `cuda_fp8.h`, `vector_types.h`
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/cu13/include/cublasLt.h` — tile/stages/cluster/inner-shape enums, preference and algo-config attributes, `cublasLtMatmulHeuristicResult_t`
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13` — nvjet name format strings (offsets 157651640, 157651704), type-tag pool (157650801-157650992), epilogue flag pool (157651348-157651632), exported `cublasLt<TAG>Matmul*` symbols, environment-variable names
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublas.so.13` — `CUBLAS_COMPUTE_*` enum strings
- `/home/aman/code/weights/GLM-5.2-FP8/config.json`, `/home/aman/code/weights/GLM-5.2-NVFP4/config.json`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py` (TGV CuTe DSL low-latency GEMM)
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/gemm/tiny_gemm.py` and `/home/aman/code/NotSglang/python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/gemm/dsv3_fused_a_gemm.py`, `dsv3_router_gemm.py`, `cutedsl_dsv3_fused_a_gemm.py`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`
- `/home/aman/code/NotSglang/python/sglang/srt/layers/flashinfer_comm_fusion.py` (`AllReduceFusionPattern.kARResidualRMSNorm`)
- `/home/aman/code/NotSglang/.claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `nvidia-smi` on this node (B200, 183359 MiB, max SM 1965 MHz, mem 3996 MHz, driver 595.71.05)

### Web (primary/vendor)

- https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md — SM100 dispatch policies, Tables 1-15 (types, alignments, tile shapes, epilogues, `PerSmTileShape`), collective-builder walkthrough, scale-factor GMEM layout
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md — CUTLASS 4.0-4.7 release contents
- https://github.com/NVIDIA/cutlass/tree/main/examples — example directory names/numbers
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/cute/01_layout.md — CuTe layout algebra
- https://docs.nvidia.com/cuda/cublas/index.html — workspace recommendations per architecture, `CUBLAS_GEMM_AUTOTUNE`, emulation env vars
- https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/index.html — `nvidia-matmul-heuristics`, `NvMatmulHeuristicsTarget.CUTLASS3`, `get_with_mnk`
- https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/ — TMEM geometry, `tcgen05.alloc` rules, MMA atom naming
- https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/ — NVFP4/MX formats, UE8M0/UE4M3, SF layouts in TMEM/SMEM
- https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/ — CTA pairs, TMA multicast
- https://arxiv.org/abs/2301.03598 — Stream-K (Osama, Merrill, Cecka, Garland, Owens)
- https://github.com/NVIDIA/cccl/discussions/5669 — `tcgen05.mma.ws` = weight-stationary
- https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/ — what nvJet is

### Web (reported measurements, not verified here)

- https://arxiv.org/html/2512.02189v2 — B200 microbenchmarks: tcgen05 latencies (11.0-12.6 cycles), FP8 3850.6 / FP4 7700.2 TFLOP/s, STREAM triad 4.141 TB/s
- https://arxiv.org/html/2604.23466v1 — cuBLAS on B200: 1671.8 TFLOP/s at 8192³, 1517.5 at 16384³
- https://www.paulwillchan.com/articles/outperforming-cublas-b200 — hand-written BF16 SM100 GEMM reaching 106.3% of cuBLAS at 8192³; CLC and Hilbert tile ordering
- https://github.com/vllm-project/vllm/issues/35467 — observed nvjet names, heuristic mis-selection (+25-43%) at M=272 on B200
- https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction — 2-SM 256×256×16 framing
