# Fifth-generation tensor cores: tcgen05, MMA shapes, and the block-scaled formats

## What this is

A working reference for the SM100 (B200) 5th-gen tensor core, written against primary
sources: the `tcgen05.*` PTX intrinsic headers shipped in our local CUDA 13.3 toolkit
(`/home/aman/code/cuda-13.3/.../cccl/cuda/__ptx/instructions/`), NVIDIA's CUTLASS source
and CuTe-DSL docs, the Colfax tutorials, and device attributes read off this box.
Every substantive claim is tagged `[verified]` (I read it in a primary source, path/URL
given), `[reported]` (a vendor or company asserts it), `[inferred]` (my arithmetic or
reasoning from architecture), or `[unverified]` (plausible, not sourced).
Where a widely-repeated claim is wrong or unsourceable, it is called out as such.

---

## Bottom line for our system

- **Our dense GEMM at C1 is not FLOP-limited and tcgen05 peak is irrelevant to it.**
  `nvjet_sm100_tst_64x8_...` (12.6% of all GPU time) has an 8-wide N tile. With `T`
  tokens per forward, decode GEMM arithmetic intensity is `2T` FLOP/byte for FP8 weights;
  B200 machine balance is `4.5 PFLOP/s ÷ 7.7 TB/s = 584` FLOP/byte, so you need
  **T ≳ 292 tokens** to be compute-bound in FP8 and **T ≳ 329** in NVFP4. At C1 with
  EAGLE 3-1-4 we have T ≈ 8. We are ~40× off the tensor-core roofline. [inferred, arithmetic below]
- **`bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x5...swiGlu_dynB_sm100f` is almost
  certainly `tcgen05.mma.cta_group::?.kind::mxf4nvf4.block_scale.block16`** — the `b16`
  triple is the NVFP4 scale-vector size 16, which only `.kind::mxf4nvf4` + `.block16`
  supports. The `Cb16` says the epilogue re-quantizes the output to NVFP4 in-kernel.
  [inferred, corroborated by `sf_vec_size = mDtypeWeights == MxE2m1 ? 32 : 16` at
  `trtllm_fused_moe_kernel_launcher.cu:1988`]
- **The "2× throughput" of the FP4 path is `.kind::mxf4`/`.kind::mxf4nvf4` vs
  `.kind::mxf8f6f4`, NOT `.block16` vs `.block32`.** Both block16 and block32 forms of
  `mxf4nvf4` run at the same rate. What buys the 2× is that instruction-K doubles from
  32 to 64, and the price is: A and B **both** E2M1, **both K-major** (no MN-major /
  transposed operand), and M fixed at 128 for `cta_group::1`. [verified: CuTeDSL
  `MmaMXF4Op`/`MmaMXF4NVF4Op` `instruction_k = 64` vs `MmaMXF8F6F4Op` `instruction_k = 32`]
- **Block-scaled MMA has no M=64 mode on `cta_group::1`** — M must be 128 (or 128/256 for
  `cta_group::2`). For a token-minor decode GEMM this means you either waste 120/128 rows
  or you swap A and B so tokens land on N (which has an 8 granularity). The
  `...SwapsAbForGen` in our `parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen`
  attention kernel is exactly this trick. [verified for the shape constraint; the kernel
  naming reading is inferred]
- **TMEM is the hard occupancy limit for FP4 GEMM.** A 128×256 FP32 accumulator is 256 of
  512 TMEM columns; add SFA (16) + SFB (32) for block16 and you are at 304, which rounds
  up to a 512-column allocation → **one CTA per SM**. 128×128 fits in 256 columns → two
  CTAs/SM. That is a design fork, not a tuning knob. [inferred from verified TMEM geometry
  + Colfax SF column counts]
- **A full-size 1-SM `tcgen05.mma` (M=128, N=256) costs ≈128 SM clocks for every kind** —
  NVIDIA sized K per kind so the instruction duration is constant. Your mainloop must
  supply 96 B/clk of A+B from SMEM (plus ~12 B/clk of scale factors for NVFP4) to keep the
  pipe full. [inferred, arithmetic below]
- **SM120 (consumer Blackwell) has no tcgen05 and no TMEM.** Our gates must key on
  SM100/SM103, not `major >= 10` — this is already noted in
  `/home/aman/code/NotSglang/glm-kernels/include/glm/glm_abi.h:40`. [verified]
- **Open, and it matters: we have not verified whether SM100 FP8/FP4 MMA needs
  DeepGEMM-style FP32 promotion.** On Hopper the FP8 wgmma accumulator was effectively
  narrower than FP32 and DeepSeek promoted every 128 K-elements. I could not source
  whether SM100 fixed this. See §11 for a measurement recipe.

---

## 1. What actually changed from Hopper

| | `mma.sync` (SM80) | `wgmma` (SM90) | `tcgen05.mma` (SM100) |
|---|---|---|---|
| Issued by | all 32 threads of a warp | all 128 threads of a warpgroup | **one thread** |
| A operand | RMEM | RMEM or SMEM descriptor | **SMEM descriptor or TMEM** |
| B operand | RMEM | SMEM descriptor | **SMEM descriptor** |
| C/D accumulator | RMEM | RMEM | **TMEM** |
| Max shape | m16n8k16 | m64n256k16 | m128n256k16 (1 SM), m256n256k16 (2 SM) |
| Completion | synchronous | `wgmma.wait_group` | **mbarrier via `tcgen05.commit`** |
| Cooperating SMs | 1 | 1 | **1 or 2 (CTA pair)** |

Sources: shape/memory row [reported] from
<https://gau-nernst.github.io/tcgen05/>; issue granularity and operand sources [verified]
from the local CCCL headers (`tcgen05_mma.h` takes `uint32_t d_tmem`, `uint64_t a_desc`
*or* `uint32_t a_tmem`, `uint64_t b_desc`) and from
<https://docs.nvidia.com/cutlass/4.6.2/media/docs/pythonDSL/guides/mma/tcgen05_programming.html>
("issue granularity being single-thread (for `.cta_group::1`) or single-thread in a CTA
pair (for `.cta_group::2`)").

### Why this changes kernel structure

Three consequences, all structural:

1. **The MMA is a one-thread, fire-and-forget async op.** There is no warpgroup-wide
   register dependency to schedule around. In practice CUTLASS elects one warp and one
   thread in it, issues the whole K-loop of MMAs back to back, and signals a single
   mbarrier at the end via `tcgen05.commit`. The rest of the CTA's warps are free for TMA,
   epilogue, or a second problem. [verified: the Colfax mainloop listing at
   <https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/>]

2. **Accumulators no longer consume registers.** A 128×256 FP32 accumulator is 128 KB.
   On Hopper that is 256 registers/thread across a warpgroup, i.e. the entire register
   budget. On Blackwell it is 256 TMEM columns and **zero** registers until the epilogue
   pulls it out with `tcgen05.ld`. This is what makes 128×256 (and 256×256 on a CTA pair)
   practical.

3. **Reading the result is now an explicit, non-trivial data movement.** `tcgen05.ld` is
   warp-scoped and each warp can only see 32 of the 128 TMEM lanes, so the epilogue needs
   ≥4 warps to drain a 128-row accumulator, and you must `tcgen05.wait::ld.sync.aligned`
   before touching the destination registers. [verified: `tcgen05_wait.h`; warp-to-lane
   restriction [reported] from Colfax's TMEM tutorial, "each warp within a warpgroup can
   only access 32 lanes"]

**Why TMEM exists at all — the arithmetic.** For M=128, N=256 the accumulator is
131,072 B. The MMA read-modify-writes it once per instruction, i.e. 262,144 B of traffic
per ~128 clocks = **2,048 B/clk per SM** (≈4.0 TB/s/SM at our measured 1965 MHz, ≈596 TB/s
per GPU). Note this number is *independent of the input dtype* — so as inputs narrow from
FP16 to FP4, accumulator bandwidth per useful FLOP rises 4×. A register file cannot serve
that; a small dedicated SRAM adjacent to the tensor core can. [inferred]

---

## 2. Tensor Memory (TMEM)

### Geometry (all [verified])

| property | value | source |
|---|---|---|
| Size | 256 KB per SM | Colfax TMEM tutorial |
| Shape | 128 lanes ("data path" rows) × 512 columns × 32-bit cell | Colfax; CUTLASS `tmem_allocator_sm100.hpp` `MAX_CAPACITY_BITS = 128*512*32` |
| Address | 32-bit: **bits 31:16 = lane, bits 15:0 = column** | CUTLASS + the engine's own `_tcgen05.py`: `tmem = (row << 16) \| col` |
| Allocation unit | **columns** (a column allocates all 128 lanes) | Colfax |
| Allocation granularity | 32 columns (`ColumnsPerAllocationSlice`) | CUTLASS `tmem_allocator_sm100.hpp` |
| Allocation constraint | `32 <= nCols <= 512` and a **power of two** | CUTLASS |
| Who may allocate | a single fully-active warp; the same warp for all allocations | CUTLASS |
| Visibility | the allocating CTA (or CTA pair); not addressable by other CTAs | [reported] |

Local corroboration: `/home/aman/code/NotSglang/python/sglang/kernels/ops/attention/cute_utils/_tcgen05.py`
allocates a flat `Uint32(512)` columns and builds TMEM pointers as
`(Int32(row) << Int32(16)) | Int32(col)`. [verified]

### The three management instructions

Exact syntax, from `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_alloc.h` [verified]:

```ptx
tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32 [dst], nCols;
tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32            taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned;
```

Notes that bite:

- `alloc` writes the **base TMEM address into shared memory** at `[dst]`, it does not
  return it in a register. The warp then reads it back from SMEM. [verified from the
  `shared::cta` qualifier and the `__as_ptr_smem(__dst)` constraint in the header]
- `dealloc` is mandatory before the CTA exits; TMEM is not reclaimed automatically.
  [reported, Colfax]
- `relinquish_alloc_permit` is the "I will never allocate again" promise that lets the
  next CTA in the queue start allocating — without it, a persistent-kernel CTA holds the
  allocator lock and blocks SM turnover. [reported, Colfax]
- `.cta_group::2` alloc requires **both CTAs of the pair to pass the same `dst` pointer**.
  [reported, CUTLASS `Allocator2Sm` preconditions]

### TMEM budget arithmetic (the real occupancy constraint)

Columns needed for a `cta_group::1` block-scaled GEMM, per CTA:

| MMA tile M×N | FP32 acc columns | SFA cols (block16) | SFB cols (block16) | sum | power-of-2 alloc | CTAs/SM |
|---|---:|---:|---:|---:|---:|---:|
| 128×256 | 256 | 16 | 32 | 304 | **512** | 1 |
| 128×192 | 192 | 16 | 24 | 232 | 256 | 2 |
| 128×128 | 128 | 16 | 16 | 160 | 256 | 2 |
| 128×64  | 64  | 16 |  8 |  88 | 128 | 4 |

Accumulator columns = N (one 32-bit cell per (m,n), lane = m) [inferred from the verified
addressing]. SFA/SFB column counts for the 128×256 block16 case are [reported] from Colfax
("16 columns for SFA, and up to 32 for SFB, for a max of 48 total"); the smaller-N rows
scale SFB proportionally [inferred]. Double-buffering the accumulator doubles the
accumulator term.

This matches an independent report: an NVFP4 grouped-GEMM worklog measured "only **one CTA
fits per SM**, which caps theoretical occupancy at 12.5% by construction" for a 128×256
tile, with achieved occupancy 12.4% and tensor-core utilization 50%.
[reported, <https://mufeezamjad.com/blog/nvfp4-group-gemm>]

**Consequence for us:** at 128×256 you get one CTA per SM and therefore *no* occupancy-based
latency hiding — the pipeline depth must come entirely from software (4+ TMA stages). At
128×128 you get two CTAs and a cheaper pipeline but half the MMA work per accumulator
read. For MoE expert GEMMs with small per-expert token counts, 128×128 with 2 CTAs/SM is
the more forgiving point. [inferred]

---

## 3. Complete tcgen05 instruction inventory

Everything below is transcribed from the CCCL generated headers in our local CUDA 13.3
toolkit. These headers are machine-generated from the PTX ISA spec and carry the ISA
version and target list in a comment on every entry. All `[verified]`.
Path prefix: `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/`.

| instruction | PTX ISA | targets |
|---|---|---|
| `tcgen05.alloc / .dealloc / .relinquish_alloc_permit` | 8.6 | SM_100a/f, SM_103a/f, SM_110a/f |
| `tcgen05.mma` (dense + block_scale) | 8.6 | SM_100a/f, SM_103a/f |
| `tcgen05.mma.ws` | 8.6 | SM_100a/f, SM_103a/f |
| `tcgen05.cp` | 8.6 | SM_100a/f, SM_103a/f, SM_110a/f |
| `tcgen05.ld` / `tcgen05.st` | 8.6 | SM_100a/f, SM_103a/f, SM_110a/f |
| `tcgen05.wait::ld` / `tcgen05.wait::st` | 8.6 | SM_100a/f, SM_103a/f, SM_110a/f |
| `tcgen05.fence::before_thread_sync` / `::after_thread_sync` | 8.6 | SM_100a/f, SM_103a/f, SM_110a/f |
| `tcgen05.commit` (+ `.multicast::cluster`) | 8.6 | SM_100a/f, SM_103a/f, SM_110a/f |
| `tcgen05.shift.cta_group.down` | 8.6 | **SM_100a, SM_103a, SM_110a only** (no `f` family variants) |

Note the two asymmetries: `tcgen05.mma` is **not** listed for SM_110, and `tcgen05.shift`
is only available on the arch-*specific* (`a`) targets, not the arch-*family* (`f`)
targets. If we ever compile with `-arch=sm_100f` for forward compatibility, `tcgen05.shift`
is off the table. [verified]

### Completion and ordering

```ptx
tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one.shared::cluster.b64 [smem_bar];
tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem_bar], ctaMask;
tcgen05.wait::ld.sync.aligned;
tcgen05.wait::st.sync.aligned;
tcgen05.fence::before_thread_sync;
tcgen05.fence::after_thread_sync;
```

The model, as used in CUTLASS and in our engine's `_tcgen05.py`:

- **`tcgen05.mma` completion** is tracked by mbarrier. You issue N MMAs, then one
  `tcgen05.commit`, which arrives on the mbarrier once **all prior** tcgen05 async ops from
  this thread have retired. Colfax: "`umma_arrive` on the `mma_barrier` fences both the
  TMEM write and the SMEM read" — i.e. one barrier wait tells you both that D is readable
  *and* that the A/B SMEM stage can be refilled. [reported]
- **`.multicast::cluster` with `ctaMask`** lets a `cta_group::2` MMA signal both CTAs of the
  pair (mask `0b11`) so the non-leader CTA also learns its SMEM stage is free. [verified
  syntax; mask value reported by DeepWiki/gau-nernst]
- **`tcgen05.ld` / `tcgen05.st` completion** is *not* mbarrier-based — it is
  `tcgen05.wait::ld` / `tcgen05.wait::st`, a warp-scoped drain. Forgetting `wait::ld`
  before reading the destination registers is the classic silent-corruption bug.
- **`tcgen05.fence::before_thread_sync` / `::after_thread_sync`** bracket the async tcgen05
  pipeline against ordinary thread-level synchronization (`bar.sync`, mbarrier waits). Our
  engine wraps both as `fence_before_thread_sync()` / `fence_after_thread_sync()` in
  `_tcgen05.py`. Precise ordering rules are in PTX ISA §9.7.17.6 which I could not extract
  verbatim — **not sourced** beyond the names and the CUTLASS/engine usage pattern.

### `tcgen05.cp` — SMEM → TMEM (one direction only)

```ptx
tcgen05.cp.cta_group::{1,2}.<shape>[.b8x16.{b4x16_p64,b6x16_p32}] [taddr], s_desc;
```
`<shape> ∈ { 128x256b, 128x128b, 4x256b, 32x128b.warpx4,
             64x128b.warpx2::01_23, 64x128b.warpx2::02_13 }` [verified, `tcgen05_cp.h`]

The signature is `(uint32_t taddr, uint64_t s_desc)` — destination is TMEM, source is an
SMEM descriptor. **There is no `tcgen05.cp` from TMEM to SMEM.** A widely-cited community
wiki (<https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/>) lists
`tcgen05.cp.tmem.shared::cta.b64` as "Copy from TMEM to SMEM"; **that instruction does not
exist in the CUDA 13.3 headers** and I could not find it anywhere else. The same page also
describes `tcgen05.alloc` as allocating *bytes* and lists a nonexistent
`tcgen05.wait.cta_group::N %sema`. Treat that page as unreliable. [verified negative]

Uses:
- `.warpx4` / `.warpx2::` multicast a small SMEM tile into all four (or two) 32-lane TMEM
  partitions. This is exactly how scale factors get replicated: Colfax describes the
  block-scaled path as using `Cp4x32x128bOp` with `.shape = .32x128b` and
  `.multicast = .warpx4`, because "scale factors for A and B matrices need to be duplicated
  to all 32 lane partitions of tensor memory." [reported]
- `.b8x16.b4x16_p64` / `.b8x16.b6x16_p32` **decompress while copying**: 16 packed 4-bit
  values (8 B) padded out to 16 B, or 16 packed 6-bit values (12 B) padded to 16 B. The
  `_pNN` suffix is the pad width in bits. [inferred from the names; corroborated by Colfax's
  description of the matching TMA types `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` ("adding 8
  bytes of padding") and `16U6_ALIGN16B` ("adding 4 bytes of padding")]
- **Critical ordering rule:** the `tcgen05.cp` for scale factors must be issued by the same
  warp that issues `tcgen05.mma`, "since both … are asynchronous instructions that are
  ordered on the same internal pipeline." That in-order pipeline is why CUTLASS uses **no
  circular buffer for scale factors in TMEM**. [reported, Colfax block-scaling tutorial]

### `tcgen05.ld` / `tcgen05.st` — TMEM ↔ registers

```ptx
tcgen05.ld.sync.aligned.<shape>.<num>[.pack::16b].b32   out, [taddr];
tcgen05.st.sync.aligned.<shape>.<num>[.unpack::16b].b32 [taddr], values;
tcgen05.ld.sync.aligned.16x32bx2.<num>[.pack::16b].b32  out, [taddr], immHalfSplitoff;
```

Register counts, extracted programmatically from the header's generated signatures
(`B32 (&out)[N]`). All [verified].

| `.shape` | x1 | x2 | x4 | x8 | x16 | x32 | x64 | x128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `.32x32b`   | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| `.16x64b`   | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| `.16x128b`  | 2 | 4 | 8 | 16 | 32 | 64 | 128 | — |
| `.16x256b`  | 4 | 8 | 16 | 32 | 64 | 128 | — | — |
| `.16x32bx2` | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |

`tcgen05.st` has an identical table. The ceiling is **128 32-bit registers per thread**
(= 16 KB per warp), which is why `.16x256b` stops at `.x32` and `.16x128b` at `.x64`.
[verified by the absence of larger variants in the header]

Reading the table: the shape names `<lanes>x<bits>` describe the TMEM footprint one warp
touches. `.32x32b` = 32 lanes × 32 bits (the natural layout for a 128-row accumulator split
across 4 warps); `.16x256b` = 16 lanes × 256 bits, i.e. a wider-but-shallower access.
`.16x32bx2` takes an extra immediate `immHalfSplitoff` and addresses **two disjoint
16-lane halves** — this is the shape you want when the accumulator itself is split across
TMEM halves (the M=64 layouts). [inferred from the signature; semantics not sourced]

`.pack::16b` / `.unpack::16b` convert between a 32-bit register holding two packed 16-bit
values and TMEM's 32-bit cells — used when the accumulator or a TMEM-resident A operand is
FP16/BF16 rather than FP32. [inferred from the naming and the `b32` register type]

Concrete epilogue pattern from our engine's own code path (`_tcgen05.py`, `LDST_MAP`):
only `32x32b` (1 reg/num), `16x128b` (2 regs/num) and `16x256b` (4 regs/num) are wired up.
[verified]

### `tcgen05.shift`

```ptx
tcgen05.shift.cta_group::{1,2}.down [taddr];
```
Syntax verified; semantics **not sourced**. The name and the `.down` qualifier suggest a
lane-wise shift of a TMEM allocation, of the kind you would use to slide an accumulator
window (convolution / attention streaming). Do not build on it without reading PTX ISA
§9.7.17.9.3.

---

## 4. `tcgen05.mma` anatomy

The four operand forms, verbatim from `tcgen05_mma.h` [verified]:

```ptx
tcgen05.mma.cta_group.kind [d_tmem], a_desc,   b_desc, idesc, enable_input_d;
tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d;
tcgen05.mma.cta_group.kind [d_tmem], a_desc,   b_desc, idesc, disable_output_lane, enable_input_d;
tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
```
each also with a trailing `, scale_input_d` immediate variant.

- **A can live in SMEM (`a_desc`, "SS" atoms) or in TMEM (`[a_tmem]`, "TS" atoms). B is
  always an SMEM descriptor.** The TS form exists for mixed-input GEMM and for FMHA, where
  A is itself a previous MMA's output already sitting in TMEM. [verified from the header;
  purpose [reported] by CuTeDSL docs: "A operand source: Can come from SMEM (default) or
  TMEM (for mixed-input GEMM/FMHA)"]
- **`enable_input_d`** is a *predicate register*, not an immediate. The CCCL header
  materializes it as `setp.ne.b32 PRED_enable_input_d, %8, 0;`. False = overwrite D
  (first K-tile), true = accumulate. CUTLASS spells this `UMMA::ScaleOut::Zero` /
  `::One` and flips it after the first `gemm()` in the mainloop. [verified header;
  CUTLASS usage reported by Colfax]
- **`disable_output_lane`** is a `.b32` mask array — 4 words (128 bits, one bit per TMEM
  lane) for `cta_group::1`, **8 words (256 bits)** for `cta_group::2`. It suppresses
  writeback to selected accumulator lanes. This is the "zero-column mask descriptor" of
  PTX ISA §9.7.17.4. [verified from the header's `const uint32_t (&disable_output_lane)[4]`
  vs `[8]`]
- **`scale_input_d`** is an `n32_t<N32>` compile-time immediate (`"n"` asm constraint), so
  it is a *static* scale on the incoming accumulator, not a runtime value. [verified]

### `.kind` values and their operand-type static asserts

From the `static_assert`s in `tcgen05_mma.h` [verified]:

| form | permitted `.kind` |
|---|---|
| plain (no block_scale) | `.kind::f16`, `.kind::tf32`, `.kind::f8f6f4`, `.kind::i8` |
| plain, `disable_output_lane` + `scale_input_d` variant | `.kind::f16`, `.kind::tf32` only |
| `.block_scale.block32` / `.scale_vec::1X` | `.kind::mxf8f6f4` |
| `.block_scale.block32` / `.scale_vec::2X` | `.kind::mxf8f6f4`, `.kind::mxf4`, `.kind::mxf4nvf4` |
| `.block_scale.block16` / `.scale_vec::4X` | `.kind::mxf4`, `.kind::mxf4nvf4` |

Note the first row's restriction: the combined
`disable_output_lane` + `scale_input_d` signature only static-asserts `f16`/`tf32`. If you
need lane masking on an FP8 MMA you get `disable_output_lane` but not `scale_input_d` in
the same instruction. [verified]

### `.collector::a::{fill,use,lastuse,discard}`

Every block-scaled `tcgen05.mma` form in the header comes in five spellings: bare, and
with `.collector::a::fill`, `::use`, `::lastuse`, `::discard`. [verified — these are real
qualifiers in CUDA 13.3]

What they *mean* — **not sourced**. The naming is a standard cache-hint idiom
(`fill` = load into the collector, `use` = read from it, `lastuse` = read and mark dead,
`discard` = invalidate), and a "collector buffer" holding the A operand across a run of
MMAs that share A would let the hardware skip re-reading A from SMEM. That reading is
`[inferred]` and I could not confirm it in any primary source; PTX ISA §9.7.17.10 was
truncated in every fetch I attempted. **Do not put a number on the savings without
measuring.**

### `tcgen05.mma.ws` — weight-stationary

```ptx
tcgen05.mma.ws.cta_group.kind.collector::b{0,1,2,3}::{fill,use,lastuse,discard}
    [d_tmem], {a_desc | [a_tmem]}, b_desc, idesc, enable_input_d [, zero_column_mask_desc];
```
`.kind ∈ { f16, tf32, f8f6f4, i8 }` — **no block-scaled kinds**. Four independent
collector slots `b0..b3` on the **B** side. [verified, `tcgen05_mma_ws.h`]

Reading: `.ws` is the mirror of the normal form — it pins B (weights) in collectors across
multiple MMAs with different A. Four slots means four weight tiles can be resident. There
is no FP4/FP8 block-scaled `.ws`, so **this path is unavailable for our NVFP4 MoE GEMMs**.
The last argument in some variants is documented in the header signature as a
`zero_column_mask_desc` (a `uint64_t`), distinct from the `disable_output_lane` array.
[verified syntax; strategic reading inferred]

### CTA pair (`cta_group::2`)

- CTAs pair by the **0th bit of the cluster rank**: 0↔1, 2↔3, … The even CTA is the
  leader and the only one that issues the MMA. [reported, Colfax]
- "Each CTA in the pair loads half of each MMA operand tile, and holds half of the
  accumulator in its TMEM." For a 256×256×16 MMA, "each CTA loads 128×16 slices from both A
  and B, and holds a 128×256 accumulator matrix in TMEM." [reported, Colfax]
- Everything doubles on the M axis: M ∈ {128, 256} instead of {64, 128}, and
  `disable_output_lane` goes from 4 to 8 words. [verified]
- `tcgen05.alloc.cta_group::2` requires both CTAs to pass the same destination SMEM
  pointer. [reported, CUTLASS]

---

## 5. Descriptors

### Shared-memory descriptor (64-bit)

Exact bitfield union from NVIDIA's own CUTLASS `include/cute/arch/mma_sm100_desc.hpp`
[verified via <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm100_desc.hpp>]:

```cpp
union SmemDescriptor {
  uint64_t desc_ = 0;
  struct {
    uint16_t start_address_       : 14, : 2;   // bits [0:14)
    uint16_t leading_byte_offset_ : 14, : 2;   // bits [16:30)
    uint16_t stride_byte_offset_  : 14, version_ : 2;  // [32:46), version [46:48)
    uint8_t  : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;   // base_offset [49:52), lbo_mode [52:53)
    uint8_t  : 5, layout_type_ : 3;            // layout_type [61:64)
  };
};
```

| field | bits | units / meaning |
|---|---|---|
| `start_address_` | 0–13 | SMEM byte address **>> 4** (16 B granularity) → 18-bit reach = 256 KB |
| `leading_byte_offset_` (LBO) | 16–29 | byte offset **>> 4** |
| `stride_byte_offset_` (SBO) | 32–45 | byte offset **>> 4** |
| `version_` | 46–47 | CUTLASS always writes **1** |
| `base_offset_` | 49–51 | CUTLASS always writes 0 |
| `lbo_mode_` | 52 | CUTLASS always writes 0 |
| `layout_type_` | 61–63 | swizzle mode |

`layout_type_` encoding [verified, same header]:

```cpp
enum class LayoutType : uint8_t {
  SWIZZLE_NONE = 0, SWIZZLE_128B_BASE32B = 1, SWIZZLE_128B = 2,
  SWIZZLE_64B = 4,  SWIZZLE_32B = 6
};
```
Values 3, 5, 7 are reserved/illegal. Note the non-obvious ordering — 128B is **2**, not 3.
Our own port at
`/home/aman/code/NotSglang/python/sglang/kernels/ops/attention/flash_attn/cute/mma_sm100_desc.py:187`
carries the same table and additionally asserts **`SWIZZLE_128B_BASE32B` is invalid for
Major-K** operands. [verified]

The 14-bit `>>4` fields mean every offset is expressed in 16-byte units and caps at
`2^14 × 16 = 256 KB` — which matches this box's `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN =
232448 B` (227 KB) with room to spare. [verified: measured via `cuDeviceGetAttribute`]

Minimal, real constructor from our engine (`_tcgen05.py`), for a 128B-swizzled BF16
operand [verified]:

```python
def make_sdesc_128B_swizzle(LBO: int):
    SBO = 8 * 128
    return Uint64((LBO >> 4 << 16) | (SBO >> 4 << 32) | (1 << 46) | (2 << 61))
    #               LBO@16           SBO@32              version=1    SWIZZLE_128B
```
The full descriptor is this base OR'd with `(smem_addr & 0x3FFFF) >> 4` in bits 0–13.

### Instruction descriptor (32-bit), non-block-scaled

```cpp
union InstrDescriptor {
  uint32_t desc_;
  struct {
    uint16_t sparse_id2_ : 2, sparse_flag_ : 1, saturate_ : 1, c_format_ : 2, : 1,
             a_format_ : 3, b_format_ : 3, a_negate_ : 1, b_negate_ : 1, a_major_ : 1;
    uint16_t b_major_ : 1, n_dim_ : 6, : 1, m_dim_ : 5, : 1, max_shift_ : 2;
  };
};
```
[verified, CUTLASS `mma_sm100_desc.hpp`]

| bits | field | encoding |
|---|---|---|
| 0–1 | `sparse_id2` | sparsity selector |
| 2 | `sparse_flag` | 1 = `.sp` variant |
| 3 | `saturate` | integer saturating accumulate |
| 4–5 | `c_format` | `F16=0, F32=1, S32=2` |
| 7–9 | `a_format` | see dtype table below |
| 10–12 | `b_format` | ditto |
| 13 | `a_negate` | negate A |
| 14 | `b_negate` | negate B |
| 15 | `a_major` | `K=0, MN=1` |
| 16 | `b_major` | `K=0, MN=1` |
| 17–22 | `n_dim` | **N >> 3** (6 bits → N ≤ 504, capped at 256) |
| 24–28 | `m_dim` | **M >> 4** (5 bits) |
| 30–31 | `max_shift` | `NoShift=0, 8=1, 16=2, 32=3` |

Operand format encodings [verified, same header]:

| A/B dtype | `a_format`/`b_format` |
|---|---|
| F16 | 0 |
| BF16 | 1 |
| TF32 | 2 |
| UINT8 | 0 |
| INT8 | 1 |
| E4M3 (fp8) | 0 |
| E5M2 (fp8) | 1 |
| E2M3 (fp6) | 3 |
| E3M2 (fp6) | 4 |
| E2M1 (fp4) | 5 |
| (invalid) | 7 |

The encodings collide across families (F16=0 and E4M3=0 and UINT8=0) — the `.kind`
qualifier disambiguates. This is exactly why `.kind::f8f6f4` allows a **runtime** choice of
A and B dtype: CUTLASS writes `tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;`
[reported, Colfax sub-byte tutorial].

Our engine's BF16 idesc builder, which you can read as a worked example [verified,
`_tcgen05.py`]:

```python
idesc = (1 << 4)                 # c_format = F32
      | (1 << 7)                 # a_format = BF16
      | (1 << 10)                # b_format = BF16
      | ((MMA_N >> 3) << 17)     # n_dim
      | ((MMA_M >> 4) << 24)     # m_dim
idesc |= negate_A << 13; idesc |= negate_B << 14
idesc |= transpose_A << 15; idesc |= transpose_B << 16   # a_major/b_major
```

### Instruction descriptor, **block-scaled** variant

```cpp
union InstrDescriptorBlockScaled {
  uint32_t desc_;
  struct {
    uint16_t sparse_id2_ : 2, sparse_flag_ : 1, : 1, b_sf_id_ : 2, : 1,
             a_format_ : 3, b_format_ : 3, a_negate_ : 1, b_negate_ : 1, a_major_ : 1;
    uint16_t b_major_ : 1, n_dim_ : 6, scale_format_ : 1, m_dim_ : 5,
             a_sf_id_ : 2, k_size_ : 1;
  };
};
```
[verified, CUTLASS `mma_sm100_desc.hpp`]

The differences from the dense form are the whole story of block scaling:

- **`c_format` and `saturate` are gone.** Bits 3–5 are reoccupied by `b_sf_id_`. There is
  no accumulator-format choice: **block-scaled MMA always accumulates in FP32.**
  [verified by absence of the field; corroborated [reported] by CuTeDSL docs:
  "Accumulator type: Always FP32 for block-scaled ops"]
- **`scale_format_` (1 bit, position 23)** selects the scale dtype:
  `ScaleFormat { UE4M3 = 0, UE8M0 = 1 }`. So MXFP4 (E8M0) and NVFP4 (E4M3) both go through
  `.kind::mxf4nvf4` and differ by **one runtime bit**, not by instruction. [verified]
- **`a_sf_id_` (2 bits, 29–30) / `b_sf_id_` (2 bits, 4–5)** select *which* of the up-to-4
  scale-factor sub-columns in TMEM this MMA consumes. Colfax: for block32/2X "two UMMAs use
  `SFA_ID` values 00 and 10"; for block16/4X "the only valid SFA_ID is 00". [reported]
- **`k_size_` (1 bit, 31)** — no CUTLASS comment. Given that `mxf8f6f4` has instruction-K
  32 and `mxf4`/`mxf4nvf4` have instruction-K 64, this bit almost certainly selects between
  them. [inferred, not confirmed]

---

## 6. MMA shapes and dtypes: the master table

Instruction-K, M and N constraints, from the CuTeDSL `MmaOp` validation logic
[verified via <https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py>],
cross-checked against the `static_assert`s in CUTLASS
`include/cute/arch/mma_sm100_umma.hpp`:

| `.kind` | A/B dtypes | acc | instr **K** | M (`cta_group::1`) | M (`cta_group::2`) | N | rel. throughput |
|---|---|---|---:|---|---|---|---|
| `.tf32` | TF32 × TF32 | F32 | **8** | 64, 128 | 128, 256 | 8…256 step 8 (1SM) / 16…256 step 16 (2SM) | 2× Hopper TF32 |
| `.f16` | {F16, BF16}² | F16 or F32 | **16** | 64, 128 | 128, 256 | same | 2× Hopper FP16 |
| `.i8` | {S8, U8}² | S32 | **32** | 64, 128 | 128, 256 | N=8 or 16…256 step 16 | 2× Hopper INT8 |
| `.f8f6f4` | any of {E5M2, E4M3, E3M2, E2M3, E2M1}, A and B chosen independently | F16 or F32 | **32** | 64, 128 | 128, 256 | same | 2× Hopper FP8 |
| `.mxf8f6f4` (block_scale) | same set, + UE8M0 scales, vec 32 | **F32 only** | **32** | **128 only** | 128, 256 | same | 2× Hopper FP8 |
| `.mxf4` (block_scale) | E2M1 × E2M1, UE8M0 scales, vec 32 | **F32 only** | **64** | **128 only** | 128, 256 | same | **4× Hopper FP8** |
| `.mxf4nvf4` (block_scale) | E2M1 × E2M1, UE8M0 **or** UE4M3 scales, vec 16 or 32 | **F32 only** | **64** | **128 only** | 128, 256 | same | **4× Hopper FP8** |

Relative-throughput column [verified] from CUTLASS
`media/docs/cpp/blackwell_functionality.md`, which lists exactly these seven kinds with
these labels.

Additional constraints [verified, CuTeDSL docstrings]:

- `.mxf4` and `.mxf4nvf4`: "**Transpose (MN-major) is not supported. Both A and B must be
  K-major.**" — `a_major_`/`b_major_` are forced to K.
- `.mxf8f6f4` A-from-TMEM (`_TS` atom) *does* allow M=64, with N step 8 at M=64 and N step
  16 at M=128 [verified, CUTLASS `SM100_MMA_MXF8F6F4_TS` static asserts]. The SMEM-source
  (`_SS`) form is M=128 only.

### CUTLASS-supported *tile* shapes (CTA tile, not instruction shape)

[verified, `media/docs/cpp/blackwell_functionality.md`]

| operands | layouts | 1SM tiles | 2SM tiles |
|---|---|---|---|
| `nv_float4_t × nv_float4_t` | **TN only** | 128×128×256, 128×192×256, 128×256×256 | 256×128×256, 256×192×256, 256×256×256 |
| `mx_float4_t × mx_float4_t` (mxf4) | **TN only** | same as above | same as above |
| `mx_float4_t × mx_float4_t` via mxf8f6f4 | all (TN/TT/NT/NN) | 128×128×128, 128×256×128 | 256×256×128 |
| mixed mx types | all | 128×128×128, 128×192×128, 128×256×128 | 256×128×128, 256×192×128, 256×256×128 |
| legacy (tf32/f16/bf16/s8/u8) | all | 64×{64,128,192,256}, 128×{64,128,192,256} | 128×{…}, 256×{…} |

Read the FP4 rows carefully: the pure-FP4 path is **TN only** (A row-major/K-major,
B column-major/K-major) — the same K-major constraint restated at the tile level.

### Derived: why every full-size MMA takes ~128 clocks

B200 datasheet peaks per GPU [verified, Lenovo ThinkSystem HGX B200 product guide]:
FP4 9/18 PFLOPS (dense/sparse), FP8 4.5/9, INT8 4.5/9 POPS, BF16 2.25/4.5,
FP16 2.25/4.5, TF32 1.1/2.2, FP32 75 TFLOPS, FP64 tensor 37 TFLOPS,
180 GB HBM3e @ 7.7 TB/s.

Measured on this box: 148 SMs, max SM clock 1965 MHz, compute capability 10.0,
183359 MiB, L2 = 132,644,864 B (126.5 MiB), 65536 regs/SM.
[verified: `nvidia-smi` + `cuDeviceGetAttribute`]

Solving `9e15 = 148 × FLOPs_per_SM_per_clk × f` for the datasheet clock gives
FLOPs/SM/clk = 32768 at f = 1.856 GHz. So:

| kind | MAC/SM/clk | FLOP/SM/clk | peak @ 1.856 GHz | peak @ 1.965 GHz (max clock) |
|---|---:|---:|---:|---:|
| FP4 (mxf4, mxf4nvf4) | 16384 | 32768 | 9.0 PFLOPS | 9.53 PFLOPS |
| FP8 / INT8 / f8f6f4 / mxf8f6f4 | 8192 | 16384 | 4.5 PFLOPS | 4.77 PFLOPS |
| FP16 / BF16 | 4096 | 8192 | 2.25 PFLOPS | 2.38 PFLOPS |
| TF32 | 2048 | 4096 | 1.1 PFLOPS | 1.19 PFLOPS |

[inferred from the verified datasheet peaks + measured SM count]

Now the punchline. A full-size `cta_group::1` MMA at M=128, N=256:

| kind | K | MACs | MAC/clk | **clocks** |
|---|---:|---:|---:|---:|
| `.tf32` | 8 | 262,144 | 2048 | **128** |
| `.f16` | 16 | 524,288 | 4096 | **128** |
| `.i8` / `.f8f6f4` / `.mxf8f6f4` | 32 | 1,048,576 | 8192 | **128** |
| `.mxf4` / `.mxf4nvf4` | 64 | 2,097,152 | 16384 | **128** |

**Every kind's maximum-size instruction occupies the tensor core for the same ~128
clocks.** NVIDIA chose instruction-K per kind precisely to normalize this. [inferred —
the arithmetic is mine, the inputs are verified]

Consequences you can design against:

- **Operand feed rate is also constant**: A+B bytes per instruction =
  `(128 + 256) × K × bytes_per_elem` = **12,288 B** for every kind → **96 B/clk/SM**.
  A 2-SM MMA doubles both work and SMs, so 96 B/clk/SM still.
- **NVFP4 adds scale-factor traffic on top**: SFA = 128 rows × (64/16) = 512 B,
  SFB = 256 × 4 = 1024 B → 1536 B per instruction → **+12 B/clk**, a 12.5% surcharge on
  operand bandwidth. Using `.block32` instead halves that to +6 B/clk.
- **Accumulator traffic is 2048 B/clk/SM** regardless of dtype (§1).
- **Instruction issue is trivially cheap**: one thread issuing one instruction every ~128
  clocks. If your MMA warp is ever the bottleneck, you have a barrier/ordering problem,
  not an issue-rate problem. This is consistent with the NVFP4 worklog's finding that the
  dominant stall was "barrier wait (≈16 inst)". [reported]

---

## 7. Block scaling in detail

### The two formats

| | MXFP8 / MXFP6 / MXFP4 (OCP "MX") | NVFP4 |
|---|---|---|
| element dtype | E4M3, E5M2 / E3M2, E2M3 / E2M1 | E2M1 |
| scale block (along K) | **32** | **16** |
| scale dtype | **UE8M0** (pure power of two, 2^x, −127 ≤ x ≤ 127) | **UE4M3** (max 448) |
| second-level scale | none | **per-tensor FP32** |
| `.kind` | `.mxf8f6f4` / `.mxf4` / `.mxf4nvf4` | `.mxf4nvf4` |
| `.scale_vectorsize` | `.block32` | `.block16` |

MX scale semantics and range [reported, Colfax block-scaling tutorial: "the possible values
of a `UE8M0` scale factor are 2^x with −127 ≤ x ≤ 127" and UE4M3 "maximum possible value is
just 448"].
NVFP4's two-level scheme [verified, <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>]:
"Each 16-value micro-block shares a single E4M3 FP8 scale factor" plus "a second-level FP32
(E8M23) scalar normalizes the entire tensor." E2M1 representable magnitudes:
0, 0.5, 1, 1.5, 2, 3, 4, 6.

The math the hardware performs: `D = (SFA · A) × (SFB · B) + C`, i.e. the scale multiplies
the *operand*, not the product. [reported, CuTeDSL tcgen05 guide]

### `.scale_vec::NX` and `.blockNN` are two names for the same knob

This trips people up. The CCCL header exposes **both** spellings for the same instruction
family. The relation is:

```
scale_vec::NX  where  N = instruction_K / sf_vec_size
blockNN        where  NN = sf_vec_size
```

| `.kind` | instr K | sf_vec_size | `.blockNN` | `.scale_vec::NX` | SFs per MMA row |
|---|---:|---:|---|---|---:|
| `.mxf8f6f4` | 32 | 32 | `.block32` | `.scale_vec::1X` | 1 |
| `.mxf4` | 64 | 32 | `.block32` | `.scale_vec::2X` | 2 |
| `.mxf4nvf4` | 64 | 32 | `.block32` | `.scale_vec::2X` | 2 |
| `.mxf4nvf4` | 64 | **16** | `.block16` | `.scale_vec::4X` | **4** |

[inferred — but the inference is forced: the local header lists exactly
`{mxf8f6f4: block32, scale_vec::1X}`, `{mxf4: block32, scale_vec::2X}`,
`{mxf4nvf4: block16, block32, scale_vec::2X, scale_vec::4X}`, which is only consistent with
this mapping. Corroborated [reported] by Colfax: "`atom_SFK = atom_K / sf_vec_size`" and
"mxf8f6f4 and mxf4 types support only block32 (1X for mxf8, 2X for mxf4). The mxf4nvf4
supports block16 (4X) or block32 (2X)."]

CUTLASS's own emitter makes the same choice by `VS` (vector size) [verified,
`SM100_MMA_MXF4_SS` in `include/cute/arch/mma_sm100_umma.hpp`]:

```cpp
static_assert(M == 128);
static_assert((N % 8 == 0) && (8 <= N) && (N <= 256));
static_assert((VS == 16) || (VS == 32));
// VS == 16:
"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;"
// VS == 32:
"tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32   [%0], %1, %2, %3, [%5], [%6], p;"
```

Note the operand form: the scale factors are **TMEM addresses** (`[%5]`, `[%6]`), not
descriptors — SFA and SFB must already be in TMEM before the MMA issues.

### Scale-factor path: GMEM → SMEM → TMEM → MMA

[reported, Colfax block-scaling tutorial + CuTeDSL guide]

1. **TMA** loads SFA/SFB tiles from GMEM to SMEM. The GMEM layout is *not* plain K-major —
   it is an interleaved "swizzled" layout so the SMEM→TMEM copy is vectorized and
   bank-conflict-free. Colfax gives the exact CuTe layouts, e.g. for mxf8 128×256:
   ```
   sfa_smem_layout_staged: ((((32,4),1),(32,1)),1,4,4) : ((((16,4),0),(0,0)),0,1,512)
   sfb_smem_layout_staged: ((((32,4),2),(32,1)),1,4,4) : ((((16,4),500),(0,0)),0,1,1024)
   nvf4 (block16) sfa:     ((((32,4),1),(16,4)),1,4,3) : ((((16,4),0),(0,1)),0,512,2048)
   ```
   and a reference host-side interleaver:
   ```python
   def interleave_sf_tensor(sf):            # (M, SF_K) -> interleaved
       M, SF_K = sf.shape
       out = sf.reshape(M // 128, 4, 32, SF_K // 4, 4)
       out = out.permute(0, 3, 2, 1, 4).contiguous()
       return out.permute(2, 3, 0, 4, 1)
   ```
2. **`tcgen05.cp` with `.32x128b.warpx4`** replicates the SF tile to **all four 32-lane
   TMEM partitions**, because every lane partition's MMA sub-unit needs its own copy.
3. **`tcgen05.mma …block_scale…` reads SFA/SFB from TMEM** via the `[scale_A_tmem]`,
   `[scale_B_tmem]` operands, selecting the sub-column with `a_sf_id_`/`b_sf_id_`.

For a CTA pair: "Each CTA in the pair has a distinct half of SFA in its TMEM (multicast 4
times across its 4 groups of 32 lanes), and both CTAs have the identical tile of SFB."
[reported, Colfax]

### TMEM cost of scale factors [reported, Colfax]

| mode | SFA columns | SFB columns (N=256) | total |
|---|---:|---:|---:|
| block32 / 1X (mxf8f6f4) | ≤4 | ≤8 | ≤12 |
| block32 / 2X (mxf4, mxf4nvf4) | 8 | ≤16 | ≤24 |
| block16 / 4X (mxf4nvf4, NVFP4) | 16 | ≤32 | **≤48** |

NVFP4's finer 16-element blocks cost **4× the TMEM** of the mxf8 case for scale factors.
At a 128×256 tile that is 48 of 512 columns — 9% — which is what pushes 304 over the
256-column power-of-2 boundary and forces the 512-column allocation. [inferred]

---

## 8. The "2× FP4 path": exactly what buys it

**Claim as usually stated:** "`.kind::mxf4nvf4` is 2× faster."
**What is actually true:**

- `.kind::mxf4` and `.kind::mxf4nvf4` are both **4× Hopper FP8**; `.kind::mxf8f6f4` and
  `.kind::f8f6f4` are both **2× Hopper FP8**. The 2× is FP4-path vs FP8-path.
  [verified, CUTLASS `blackwell_functionality.md` table]
- The mechanism is instruction-K: **64 for mxf4/mxf4nvf4, 32 for mxf8f6f4**, with the same
  ~128-clock instruction duration. [verified K values; duration inferred in §6]
- **`.block16` is not faster than `.block32`.** Both are `mxf4nvf4` at K=64. `.block16`
  costs *more* — 2× the SF bytes and 2× the TMEM columns — and buys accuracy, not speed.
  If anyone claims NVFP4-block16 is 2× MXFP4-block32 in throughput, that is wrong.
  [inferred from verified instruction-K equality; I found no source claiming otherwise, and
  CUTLASS lists both at "4× Hopper Fp8"]

**The price list for the 2× — every one of these is a hard constraint:**

| constraint | source |
|---|---|
| A and B **both** E2M1 (no mixed FP4×FP8, no FP6) | [verified] CuTeDSL `MmaMXF4Op`/`MmaMXF4NVF4Op` fix both dtypes to `Float4E2M1FN` |
| A and B **both K-major**; MN-major/transpose unsupported | [verified] "Transpose (MN-major) is not supported. Both A and B must be K-major." |
| CTA tile layout **TN only** | [verified] CUTLASS tile table |
| M = **128** exactly for `cta_group::1` (no M=64) | [verified] CuTeDSL + `SM100_MMA_MXF4_SS` `static_assert(M == 128)` |
| accumulator is **FP32, not selectable** | [verified] `InstrDescriptorBlockScaled` has no `c_format` |
| scale factors must be **pre-staged in TMEM** via `tcgen05.cp`, issued by the MMA warp | [reported] Colfax |
| GMEM scale layout must be the **interleaved** layout, not K-major | [reported] Colfax |
| **no `.ws` (weight-stationary) variant exists** for any block-scaled kind | [verified] `tcgen05_mma_ws.h` `.kind ∈ {f16, tf32, f8f6f4, i8}` |

**If you cannot meet all of them, you fall back to `.kind::mxf8f6f4` and pay 2×.** In
particular, mxf8f6f4 *does* accept E2M1 operands — so a kernel that needs mixed
FP4×FP8, or an MN-major operand, or M=64, silently runs FP4 data at the FP8 rate. This is a
real and easy-to-miss performance cliff. [inferred from the verified dtype sets: mxf8f6f4's
A/B set includes `Float4E2M1FN`]

---

## 9. Accumulation precision, and where error actually comes from

### What is verified

- **Block-scaled MMA (`mxf8f6f4`, `mxf4`, `mxf4nvf4`) accumulates in FP32, always.** The
  block-scaled instruction descriptor has **no `c_format` field** — those bits are
  reallocated to `b_sf_id_`. [verified, CUTLASS `InstrDescriptorBlockScaled`] Corroborated
  [reported] by CuTeDSL: "Accumulator type: Always FP32 for block-scaled ops."
- **Non-block-scaled float MMA lets you pick**: `c_format ∈ {F16=0, F32=1}` for
  `.kind::f16` and `.kind::f8f6f4`; `S32=2` for `.kind::i8`. [verified]
  Choosing F16 halves accumulator TMEM (and therefore doubles CTAs/SM) at obvious cost.
- **Scales multiply the operands, not the product**: `D = (SFA·A) × (SFB·B) + C`.
  [reported, CuTeDSL]

### What is not sourced

- **The internal accumulation width of a single instruction's K-reduction.** For
  `.kind::mxf4nvf4` one instruction reduces 64 products per output element. Whether the
  hardware uses an exact wide adder tree, or a sequence of FP32 adds, or something in
  between, is **not documented in any source I could read**. Do not assume.
- **Whether SM100 needs DeepGEMM-style FP32 promotion for FP8.** On Hopper, DeepSeek found
  that `wgmma` FP8 accumulation was effectively narrower than FP32 and promoted partial
  sums to FP32 CUDA cores every 128 K-elements. I could not source whether SM100 fixed
  this. **This is the single most consequential unknown in this document for us**, because
  GLM-5.2's K dimensions are in the thousands. See §12 for the measurement.

### The error model you *can* reason about

For a GEMM with reduction length `K` and instruction-K `K_i`, the FP32 TMEM accumulator
performs `K/K_i` sequential accumulate steps:

| kind | `K_i` | steps at K=7168 | steps at K=2048 |
|---|---:|---:|---:|
| `.f16` | 16 | 448 | 128 |
| `.f8f6f4` / `.mxf8f6f4` | 32 | 224 | 64 |
| `.mxf4` / `.mxf4nvf4` | 64 | **112** | 32 |

If the intra-instruction reduction is exact (unverified), FP32 rounding error over the full
K accumulates as a random walk over `K/K_i` steps: relative error ≈ `sqrt(K/K_i) · 2^-24` ×
(sum magnitude / result magnitude). At K=7168 with FP4 that is `sqrt(112) ≈ 10.6` ulps of
FP32 — negligible. **So FP32 accumulator rounding is not your problem.** [inferred]

Your problem is quantization:

- E2M1 has **one mantissa bit**. Representable magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6};
  worst-case relative rounding error inside a block is ~1/6 ≈ 17% for a single element, and
  the block's dynamic range is entirely carried by the scale.
- **NVFP4's block-16 + E4M3 scale is a direct attack on this**, versus MXFP4's block-32 +
  E8M0. Halving the block localizes the range; E4M3's 3 mantissa bits mean the scale itself
  can land between powers of two, so the block's amax maps closer to 6.0 instead of being
  rounded down to the next power of two (up to 2× wasted range with E8M0). NVIDIA reports
  "1% or less accuracy degradation on key language modeling tasks" going FP8 → NVFP4 on
  DeepSeek-R1-0528. [reported/verified, NVIDIA NVFP4 blog]
- **E4M3's max is 448**, so the per-tensor FP32 scale must be chosen such that every block's
  `amax / 6 ≤ 448`. Getting this wrong clips whole blocks. [inferred from the verified
  format bounds]

**Practical rule for us:** if a long-K NVFP4 GEMM disagrees with BF16 by more than a few
1e-3 relative, suspect the scale-factor layout or the per-tensor scale, not the FP32
accumulator.

---

## 10. Operand layout, swizzle, and the sub-byte packing rules

### The core matrix

SMEM operands are described to the tensor core in units of an **8 × 16 B core matrix**
(128 B). LBO and SBO in the SMEM descriptor are strides between these units, expressed in
16 B granules. [reported, gau-nernst; consistent with the verified 14-bit `>>4` fields]

Our own CUTLASS port validates canonical layouts explicitly [verified,
`mma_sm100_desc.py:224-305`]:

- **Major-K path**: requires MN-size a multiple of 8; `logical_divide(layout, (8, 2))`;
  for swizzled layouts the inner K stride must be 1 and the MN stride must equal the
  swizzle atom MN size (`SWIZZLE_NONE`→1, `32B`→2, `64B`→4, `128B`→8, all in 16 B units).
- **Major-MN path**: `logical_divide(layout, (swizzle_atom_mn_size, swizzle_atom_k_size))`
  with `swizzle_atom_k_size = 4` for `SWIZZLE_128B_BASE32B`, else 8; for swizzled layouts
  the innermost MN stride must be 1.
- **`SWIZZLE_128B_BASE32B` raises for Major-K** — it is an MN-major-only mode.

Swizzle-mode ↔ CuTe `Swizzle<B,M,S>` mapping [verified, same file]:

| `Swizzle<B,M,S>` | LayoutType |
|---|---|
| `Swizzle<0,4,3>` | `SWIZZLE_NONE` (0) |
| `Swizzle<1,4,3>` | `SWIZZLE_32B` (6) |
| `Swizzle<2,4,3>` | `SWIZZLE_64B` (4) |
| `Swizzle<3,4,3>` | `SWIZZLE_128B` (2) |
| `Swizzle<2,5,2>` | `SWIZZLE_128B_BASE32B` (1) |

The CuTeDSL equivalent names are `SmemLayoutAtomKind ∈ {MN_INTER, K_INTER, MN_SW32,
MN_SW64, MN_SW128, K_SW32, K_SW64, K_SW128, MN_SW128_32B}`. [reported, CUTLASS API docs]

### Sub-byte (FP4/FP6) SMEM packing — the padded format

For `.kind::f8f6f4`, 4- and 6-bit operands are **not** stored fully packed
[reported, Colfax sub-byte tutorial]:

- "16 consecutive 4-bit or 6-bit elements are packed contiguously and then padded to 16-byte
  boundaries."
- "Fully compressed contiguous data in SMEM is **not** supported with the `.kind::f8f6f4`
  qualifier."
- Space is budgeted as if the data were byte-width.

TMA does the unpadding→padding conversion on the fly with two special tensor-map types:

| TMA data type | `tcgen05.cp` counterpart | effect |
|---|---|---|
| `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` | `.b8x16.b4x16_p64` | 16 × 4b (8 B) → 16 B, +8 B pad |
| `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B` | `.b8x16.b6x16_p32` | 16 × 6b (12 B) → 16 B, +4 B pad |

Alignment rules for these TMA types [reported, Colfax]:
- base address 32 B aligned;
- leading-dimension extent a multiple of **128 elements**;
- only **128 B swizzle or no swizzle**;
- CUTLASS additionally asserts 4-bit data 64 B aligned, 6-bit data 96 B aligned.

**This is the hidden 2× SMEM cost of the `f8f6f4` path**, and it is a second reason the
pure-`mxf4` path is preferable: Colfax notes block-scaled mxf4 "reduces SMEM usage by a
factor of two compared to using 4-bit data types with the `mxf8f6f4` qualifier." [reported]

---

## 11. Sparsity (2:4) on SM100 — yes, it exists

`tcgen05.mma.sp` is real on SM100. CUTLASS emits it directly
[verified, <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm100_umma.hpp>]:

```ptx
tcgen05.mma.sp.cta_group::1.kind::tf32    [%0], %1, %2, [%9], %3, {%5,%6,%7,%8}, p;
tcgen05.mma.sp.cta_group::1.kind::f16     [%0], %1, %2, [%9], %3, {%5,%6,%7,%8}, p;
tcgen05.mma.sp.cta_group::1.kind::i8      [%0], %1, %2, [%9], %3, {%5,%6,%7,%8}, p;
tcgen05.mma.sp.cta_group::1.kind::f8f6f4  [%0], %1, %2, [%9], %3, {%5,%6,%7,%8}, p;
```

Structs: `SM100_MMA_{TF32,F16BF16,S8,F8F6F4}_SS_SPARSE` and their `_2x1SM_SS_SPARSE`
counterparts; block-scaled sparse exists too as
`SM100_MMA_MXF8F6F4_SS_SPARSE`, `SM100_MMA_MXF4NVF4_SS_SPARSE` and their 2×1SM forms
[verified].

Key facts:

- **The metadata operand is a TMEM address** (`[%9]`), inserted *between* `b_desc` and
  `idesc`. Sparse metadata lives in TMEM, not registers. [verified from the operand order]
- The instruction descriptor's `sparse_flag_` (bit 2) and `sparse_id2_` (bits 0–1) select
  the sparse mode and metadata sub-selector. [verified, `InstrDescriptor`]
- **K doubles** under sparsity, and so does the scale-factor vector size:
  "Scale factors are applied to GEMM-K dimension such that every 16 or 32 elements … have
  an associated scale factor (**32 or 64 elements for sparse** as sparse gemm compress 2×
  along k-dim)." So sparse NVFP4 uses SFVecSize 32, sparse MX uses 64. [verified, CUTLASS
  `blackwell_functionality.md`]
- CUTLASS dispatch policies: `KernelSparseTmaWarpSpecialized1SmSm100` / `2SmSm100`, and
  `KernelSparseTmaWarpSpecialized2SmBlockScaledSm100`. [reported]
- Datasheet sparse peaks: FP4 18 PFLOPS, FP8 9 PFLOPS, BF16 4.5 PFLOPS. [verified]

**Relevance to us: currently none.** GLM-5.2 is not 2:4-sparse, and MoE + 2:4 is a research
problem, not a deployment one. Worth knowing that the 18 PFLOPS headline number requires
2:4-structured weights and is unreachable for us.

**`tcgen05.mma.sp` is NOT exposed in the CCCL `cuda::ptx` headers** shipped with CUDA 13.3 —
I grepped the entire include tree and found no `.sp` variant. You must write the asm
yourself or go through CUTLASS. [verified negative]

---

## 12. What a peak-rate NVFP4 GEMM inner loop looks like

Structure, assembled from the verified instruction set and the CUTLASS/Colfax mainloop
patterns. `[inferred]` as a whole; each individual instruction is `[verified]`.

**Setup (once per CTA):**

```
warp 0, thread 0:
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_tmem_base], 512;
  // 512 columns: 256 for the 128x256 FP32 accumulator, 48 for SFA/SFB, rest slack
  tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;   // let the next CTA queue
  read tmem_base back from SMEM
  build idesc  (block-scaled form: a_format=b_format=E2M1(5), a_major=b_major=K(0),
                n_dim=256>>3=32, m_dim=128>>4=8, scale_format=UE4M3(0), a_sf_id=b_sf_id=0)
  build sdesc bases for A and B (version=1, layout_type=SWIZZLE_128B(2), LBO/SBO >>4)
```

**Steady state**, 4+ stage pipeline, warp-specialized:

```
producer warps (2 of them: one for A/SFA, one for B/SFB)
  loop over k_tile:
     wait empty[stage]                             // mbarrier, phase parity
     cp.async.bulk.tensor.2d ... [sA[stage]], [tma_A], [full[stage]]     // TMA
     cp.async.bulk.tensor.2d ... [sSFA[stage]], [tma_SFA], [full[stage]]
     mbarrier.arrive.expect_tx  full[stage], bytes

MMA warp, thread 0 only:
  loop over k_tile:
     mbarrier.try_wait.parity  full[stage]
     tcgen05.cp.cta_group::1.32x128b.warpx4  [sfa_tmem], sfa_sdesc;   // SMEM->TMEM, all 4 lane groups
     tcgen05.cp.cta_group::1.32x128b.warpx4  [sfb_tmem], sfb_sdesc;
     tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
          [acc_tmem], a_desc(stage), b_desc(stage), idesc,
          [sfa_tmem], [sfb_tmem], PRED_accum;     // PRED_accum = (k_tile != 0)
     tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [empty[stage]];
```

**Epilogue (4 warps, after the final commit):**

```
  mbarrier.try_wait.parity  mma_done
  tcgen05.fence::after_thread_sync;
  tcgen05.ld.sync.aligned.32x32b.x8.b32 {r0..r7}, [acc_tmem + (warp_id*32 << 16) + n*8];
  tcgen05.wait::ld.sync.aligned;
  ... scale, activation, re-quantize to NVFP4, TMA store ...
  tcgen05.dealloc.cta_group::1.sync.aligned.b32  tmem_base, 512;
```

**The five things that make or break the rate:**

1. **Never re-read A/B from L2.** Each MMA needs 12,288 B of A+B in 128 clocks (96 B/clk).
   At 1.965 GHz that is 189 GB/s per SM, 27.9 TB/s per GPU — **3.6× HBM bandwidth**. Only
   SMEM reuse across the N (and M) tiles closes that gap. [inferred]
2. **Pipeline depth ≥ 4.** With one CTA per SM there is no occupancy fallback. The NVFP4
   worklog measured a 4-stage pipeline as the point where "a small delay in TMA completion
   or barrier arrival" stopped stalling the tensor core immediately. [reported]
3. **Issue `tcgen05.cp` for scale factors from the MMA warp**, never from a producer warp —
   they share the same in-order async pipeline, which is what gives you ordering for free
   and is why no SF circular buffer is needed. [reported, Colfax]
4. **Split the producers.** One producer warp issuing both A and B TMAs becomes the
   descriptor-issue bottleneck; the worklog's v4 split it into "Warp 4: producer for A and
   SFA" / "Warp 6: producer for B and SFB". [reported]
5. **Multicast A across the cluster** when the same A tile feeds several N tiles —
   "the master CTA (rank 0) issues the TMA load, and the hardware multicasts the data."
   [reported]

For calibration: that worklog's tuned NVFP4 grouped GEMM reached ~50% tensor-core
utilization end-to-end (~80% in steady-state mainloop), 27% DRAM throughput, 12.4%
occupancy. [reported] Treat 50–60% of FP4 peak as a realistic target for a well-tuned
grouped GEMM, not 90%.

---

## 13. Which of our measured kernels is on which path

### `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x5...swiGlu_dynB_sm100f` (6.0% at C1, 10.4% at C64)

**Verdict: `tcgen05.mma.…kind::mxf4nvf4.block_scale.block16`, with high confidence.**
[inferred, strongly constrained]

Name decode:

| token | reading | confidence |
|---|---|---|
| `bmm` | batched GEMM (trtllm-gen "batched matmul", the MoE grouped-GEMM family) | high |
| `E2m1` (1st) | **output** dtype = NVFP4 | high — the pair form `bmm_Bfloat16_E2m1E2m1_...` also appears, and its output must be BF16 |
| `E2m1E2m1` | A dtype × B dtype = E2M1 × E2M1 | high |
| `Fp32` | accumulator = FP32 | high (and forced: block-scaled MMA has no other option) |
| `Ab16`, `Bb16`, `Cb16` | scale **block size 16** for A, B, and the re-quantized C | high — `sf_vec_size = MxE2m1 ? 32 : 16` at `trtllm_fused_moe_kernel_launcher.cu:1988` [verified] |
| `t128x8x5` | tile M=128, N=8, 5 pipeline stages | medium |
| `swiGlu` | fused SwiGLU epilogue → this is the **gate/up** GEMM | high (`ActType::SwiGlu` at `trtllm_fused_moe_runner.cu:351` [verified]) |
| `dynB` | dynamic batch (per-expert token counts known only at runtime) | medium |
| `sm100f` | compiled for the **`sm_100f` arch family** target | high — matches the `SM_100f` family-conditional guards in the CCCL headers [verified] |

Because block size 16 is only reachable via `.kind::mxf4nvf4` + `.block16` (§7), and both
operands are E2M1, this kernel is on the **4×-Hopper-FP8 FP4 path** with UE4M3 scales.
`Cb16` means the epilogue re-quantizes to NVFP4 in-kernel — that output feeds the second
(down) GEMM, which explains why the companion kernel `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_B...`
(5.2% at C1) has a BF16 output: it is the **down** projection, terminating the expert.

**Implication:** the MoE expert GEMMs are already on the fastest available instruction path.
Any win there is in tiling, pipelining, and the routing/gather overhead
(`finalizeKernelVecLoad` is 6.2% at C64 on its own) — **not** in switching MMA kind.
The `t128x8x5` N=8 tile at C1 is the tell: with 8 active experts and 1 token, each expert
GEMM has ~1 row of work, so this kernel is entirely weight-bandwidth-bound.

### `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` (12.6% at C1)

**Verdict: a `cta_group::1` tcgen05 MMA at M=64, N=8 — the minimum legal N.** [inferred]

cuBLAS `nvjet` naming is **not publicly documented**; I could not source it and will not
pretend otherwise. What the corpus of names we observe supports:

| observed name | reading |
|---|---|
| `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` (C1) | tile 64×8, K-tile 64, 16 stages, cluster 4×1 |
| `nvjet_sm100_tst_32x64_64x16_4x1_v` (C1) | tile 32×64, K-tile 64, 16 stages, cluster 4×1 |
| `nvjet_sm100_tst_256x256_64x4_2x2_` (C64) | tile 256×256, K-tile 64, 4 stages, cluster 2×2 |
| `nvjet_sm100_tst_128x256_64x6_2x1_` (C64) | tile 128×256, K-tile 64, 6 stages, cluster 2×1 |
| `nvjet_sm100_tst_256x128_64x5_2x2_` (C64) | tile 256×128, K-tile 64, 5 stages, cluster 2×2 |

The pattern `<M>x<N>_<Ktile>x<stages>_<clusterM>x<clusterN>` is consistent across all five,
and the C64 tiles (256×256, 128×256, 256×128) are **exactly** the CUTLASS-listed legal 2SM
and 1SM tile shapes. [inferred, but the C64 shapes matching the verified legal-shape table
is strong evidence.] `TNT` is almost certainly the A/B/C transpose triple in cuBLAS's
convention; `bz` reads as beta-zero (C not read). Both **[unverified]**.

M=64 rules out block-scaled kinds (M must be 128), so `tst` is a legacy kind —
`.kind::f16` (BF16) or `.kind::f8f6f4`. **Which one is not established** and matters: it is
the difference between 2.25 and 4.5 PFLOPS of headroom. Measure it with `ncu`
(`sm__inst_executed_pipe_tensor_op_hmma` vs `..._imma`/dtype-specific counters) before
assuming.

**The important reading is not the dtype — it is N=8.** At C1 the dense GEMM is running
tcgen05 with the *smallest legal N tile*, meaning the tensor core is doing 8 of a possible
256 output columns per instruction. That is not a tuning failure; it is the correct choice
for a memory-bound problem. The roofline:

| regime | tokens `T` | weight bytes/elem | AI = 2T/bytes | B200 balance | fraction of peak |
|---|---:|---:|---:|---:|---:|
| C1, EAGLE 3-1-4, FP8 weights | 8 | 1.0 | 16 | 584 | **2.7%** |
| C1, EAGLE 3-1-4, NVFP4 weights (+E4M3 scales) | 8 | 0.5625 | 28.4 | 1169 | **2.4%** |
| C64, EAGLE 3-1-4, FP8 weights | 512 | 1.0 | 1024 | 584 | compute-bound |

[inferred; machine balance = verified datasheet peak ÷ verified 7.7 TB/s]

That single table explains the whole C1→C64 kernel-shape shift we measured: 64×8 tiles at
C1, 256×256 tiles at C64. **At C1, no amount of tcgen05 cleverness in the dense GEMM will
help — the win is in reducing weight traffic (bigger speculative batch, better weight
residency in the 126.5 MiB L2) or in overlapping the GEMM with something else.**

### `parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen` (6.0% at C1)

Not a GEMM kernel per se, but the name carries a tcgen05 lesson: **`SwapsAb`**. Because
`tcgen05.mma` requires M ∈ {64, 128} (or 128 only for block-scaled) while N goes down to 8,
a decode-shaped problem must put the *small* dimension on N. Swapping A and B is how you do
that. [inferred] `Q8` = FP8 queries, `Kv128` = head dim 128, `ForGen` = generation phase.

### `fmhaSm100fKernel_QkvE4m3OBfloat16...` (6.3% at C1, 9.1% at C64)

`sm100f` again — the arch-family target. `QkvE4m3` = FP8 E4M3 QKV, `OBfloat16` = BF16
output. FMHA on SM100 is the canonical user of the **TS operand form**
(`tcgen05.mma [d_tmem], [a_tmem], b_desc, …`), because the second GEMM's A operand is the
softmax output of the first GEMM, already in TMEM. [inferred]

---

## 14. Open questions and measurements to run on this box

1. **Does SM100 FP8/FP4 MMA need DeepGEMM-style FP32 promotion?**
   Recipe: build a `.kind::f8f6f4` GEMM with K = 8192, all A and B entries = 1.0 in E4M3
   (exactly representable). Exact answer is 8192.0. Compare the TMEM accumulator against a
   BF16 reference. Repeat with values that stress the mantissa (e.g. `1.0 + 2^-9`). If the
   error scales like K rather than sqrt(K/32), the internal accumulate is narrow. Same test
   for `.kind::mxf4nvf4` with unit scales. **This is the highest-value experiment in this
   document.**

2. **What is `tst` in `nvjet_sm100_tst_*`?** Run the C1 workload under `ncu` and read the
   per-dtype tensor pipe counters on that kernel. Answer decides whether the dense GEMM's
   theoretical ceiling is 2.25 or 4.5 PFLOPS — though §13 shows we are 40× below either.

3. **What does `.collector::a::*` actually save?** Microbenchmark: a chain of `N` MMAs
   sharing one A descriptor, with `collector::a::fill` on the first and `::use` on the rest,
   versus all-bare. Measure clocks with `%globaltimer` / `clock64()`. If the collector
   elides A re-reads, the 96 B/clk operand feed drops toward 64 B/clk for the B-only part,
   which would matter for SMEM-bound tiles.

4. **Is `tcgen05.mma.ws` usable anywhere in our stack?** It has no block-scaled kinds, so
   not for the MoE FP4 GEMMs. But if the dense path turns out to be `.kind::f8f6f4`, a
   weight-stationary variant with B pinned in 4 collector slots is exactly the shape of a
   decode GEMM. Nothing in CUTLASS uses `.ws` as far as I can tell — needs hand-written PTX.

5. **`tcgen05.shift` semantics.** Syntax verified, meaning unknown. Requires reading PTX ISA
   §9.7.17.9.3 from the PDF (the HTML page truncates before it in every fetch).

6. **Exact SF TMEM column counts for N ≠ 256.** Colfax gives 16/32 for a 128×256 block16
   tile; I extrapolated linearly for smaller N. Confirm by reading CUTLASS's
   `Sm100BlockScaledConfig` before sizing a `tcgen05.alloc`.

7. **PTX ISA §9.7.17.10 verbatim.** Everything about MMA shapes, collector, and the sparse
   metadata layout in this document is second-hand (CUTLASS, CuTeDSL, Colfax). The
   authoritative tables live in a section that `docs.nvidia.com`'s HTML page truncates
   before. Get the PTX ISA **PDF** onto this box and re-verify §4, §6, and §11.

---

## Sources

### Local (read directly)

- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_mma.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_mma_ws.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_alloc.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_cp.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_ld.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_st.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_commit.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_wait.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_fence.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_shift.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/ptx_dot_variants.h`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/attention/flash_attn/cute/mma_sm100_desc.py`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/attention/cute_utils/_tcgen05.py`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/moe/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu` (line 1988: `sf_vec_size`)
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/moe/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu` (line 351: `ActType::SwiGlu`)
- `/home/aman/code/NotSglang/glm-kernels/include/glm/glm_abi.h` (SM120 has no tcgen05)
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/benchmark/runs/latency-3-1-4-repo-baseline-c1-7b4dd908fae6fa2f/report.txt`
- `/home/aman/code/benchmark/runs/capacity-overlap-repo-baseline-c64-69822ec843c1a994/report.txt`
- `/home/aman/code/benchmark/runs/latency-3-1-4-aa-10k-single-c1-f65465a719d104a4/report.txt`
- Device attributes measured via `nvidia-smi` and `cuDeviceGetAttribute` on this node.

### Web (fetched and read)

- <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm100_desc.hpp> — `SmemDescriptor`, `InstrDescriptor`, `InstrDescriptorBlockScaled`, all format enums
- <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm100_umma.hpp> — UMMA structs, PTX strings, static asserts, sparse variants
- <https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py> — per-kind M/N/K validation
- <https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md> — throughput table, tile-shape tables, sparse SFVecSize
- <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/tmem_allocator_sm100.hpp> — TMEM capacity and allocation constraints
- <https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/copy_sm100.hpp> — TMEM load/store op names
- <https://docs.nvidia.com/cutlass/4.6.2/media/docs/pythonDSL/guides/mma/tcgen05_programming.html> — CTA-pair semantics, block-scaled staging, accumulator-in-TMEM
- <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html> — Field/CtaGroup/OperandSource/SmemLayoutAtomKind enums
- <https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/> — TMEM geometry, alloc rules, ld shapes
- <https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/> — scale-factor formats, layouts, TMEM columns, `tcgen05.cp` ordering
- <https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/> — sub-byte SMEM padding, TMA types, alignment
- <https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/> — CTA-pair leader election, mainloop listing, `umma_arrive`
- <https://gau-nernst.github.io/tcgen05/> — descriptor field walkthrough, MMA shape history
- <https://deepwiki.com/gau-nernst/learn-cuda/8.1-tcgen05-instructions-and-tensor-memory> — TMEM addressing, accumulator warp/lane mapping
- <https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tensor_cores/index.html> — single-thread issue, M=64 layout note, SF data flow
- <https://mufeezamjad.com/blog/nvfp4-group-gemm> — measured NVFP4 GEMM: `block16` PTX in the wild, occupancy/TMEM constraint, pipeline depth findings
- <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/> — NVFP4 definition, two-level scaling, MXFP4 comparison, accuracy
- <https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu> — B200 SXM dense/sparse peak table
- <https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html> — PTX ISA §9.7.17 section structure (the section bodies themselves truncate on fetch)

### Explicitly flagged unreliable

- <https://0xsero.github.io/blackwell-gpu-wiki/blackwell/tcgen05-and-tmem/> — lists a
  nonexistent `tcgen05.cp.tmem.shared::cta.b64` (TMEM→SMEM), a nonexistent
  `tcgen05.wait.cta_group::N`, describes `tcgen05.alloc` as byte-granular, and gives
  "single-CTA maximum m128n128k64". None of these survive contact with the CUDA 13.3
  headers. Do not cite it.
