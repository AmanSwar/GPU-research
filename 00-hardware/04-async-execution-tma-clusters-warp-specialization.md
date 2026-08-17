# Asynchrony on Blackwell: TMA, mbarriers, clusters, CLC, and warp specialization

**What this is.** A working reference for how data and work actually get pipelined on
SM100 (B200), written from the PTX ISA 9.3 spec that ships with our CUDA 13.3 toolkit,
the `cuda.h` / `cuda/__ptx` headers on this box, CUTLASS's SM100 sources, and a set of
microbenchmarks I compiled and ran on **this** 8xB200 node. Every substantive claim is
labelled `[verified]` (read in a primary source, path/URL given), `[measured]` (I ran it
here), `[reported]` (a vendor asserts it), `[inferred]`, or `[unverified]`. The last
section applies all of it to the batch-1 decode regime our GLM-5.2 profile actually sits in.

---

## Bottom line for our system

- **A TMA request costs ~288 SM cycles of issue+completion bookkeeping regardless of
  transfer size from 128 B to 16 KB** `[measured]`. Below ~8 KB per request you are
  request-rate-bound, not bandwidth-bound. Size every `cp.async.bulk.tensor` box to
  **≥8 KB**, and prefer 16 KB where SMEM allows.
- **A serialized TMA round trip (issue → mbarrier → observe in SMEM) is ~1000 cycles
  (~510 ns)** `[measured]`, versus ~489 cycles for a dependent `ld.global` from a
  1 MiB working set and ~690–810 cycles from 256 MiB `[measured]`. Asynchrony does not
  make a single hop cheaper; it only lets you have several in flight.
- Our C1 profile is **1,411,149 kernel launches on rank 0 in an 8473 ms window ≈ 456
  launches per output token, mean kernel 6.78 µs ≈ 13,300 SM cycles** `[measured, derived
  from the profile in the task brief]`. A 4-deep TMA pipeline fill (~1000 + 3×288 ≈ 1860
  cycles) is **~14 % of an average kernel** and **~39 % of an average elementwise kernel
  (2.43 µs)**. At batch 1 the pipeline prologue *is* the kernel.
- **CLC (`clusterlaunchcontrol.try_cancel`) does almost nothing for us at batch 1** and is
  not a wave-quantisation fix in the regime we run: it load-balances *many* tiles across
  *fewer* SMs. With decode-shaped GEMMs the tile count is at or below 148, so there is no
  second wave to steal from. Its real value here is **coexistence** — surviving a
  higher-priority kernel occupying SMs — and it becomes worth having at C64+ `[inferred
  from the CLC design docs, verified below]`.
- **2-SM UMMA (`cta_group::2`) is actively wrong at batch 1.** It doubles the MMA's M
  granularity (M ∈ {128, 256}) at exactly the moment our M is 1–4 rows `[verified: PTX
  §9.7.17; CUTLASS blackwell docs]`. Keep `cta_group::1` for decode, switch at prefill/C64.
- **`cp.async.bulk.tensor.2d.tile::gather4` loads four arbitrary rows of a 2-D tensor in
  one instruction** (coords `{col, row0, row1, row2, row3}`) `[verified: PTX ISA 9.3
  §9.7.9.26.5.2]`. That is precisely the access shape of DSA sparse MLA's top-k KV gather
  and of a paged KV cache. Our DSA attention (10.9 %) + indexer (5.8 %) = 16.7 % of GPU
  time is the best-shaped target for it on the box.
- **`multimem.cp.async.bulk` / `multimem.cp.reduce.async.bulk` exist and require only
  sm_90+** (PTX ISA 9.1, `.cp_mask` needs sm_100) `[verified]`. SMEM → multimem global
  with `.add.bf16.noftz` is a one-instruction fused reduce-scatter store. Collectives are
  19.6 % of our time and 47 % of that is rank skew, so this is not the first fix — but it
  is the right primitive if we ever write our own AR.
- **`nvcc -arch=sm_100a` does NOT silently degrade to `.target sm_100` on CUDA 13.3**
  `[measured]` — see "Claims that did not survive" below. Our own `AGENT-HANDOFF-sm100.md`
  says it does; that was true of an older toolkit or is simply wrong here.

---

## 1. The model: proxies, async operations, completion mechanisms

Everything asynchronous on SM100 is built from three ideas.

**Async proxy.** TMA copies, CLC responses, and `tcgen05` writes are performed by hardware
units that are *not* the issuing thread, and their memory effects are ordered in a separate
"proxy" from ordinary loads/stores. Crossing between proxies requires an explicit fence:
`fence.proxy.async`, `fence.proxy.async::generic.{acquire,release}.sync_restrict…`,
`fence.proxy.tensormap::generic.{acquire,release}` `[verified: PTX ISA 9.3 §9.7.9.26.2 and
§9.7.14; wrappers in `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/fence*.h`]`.
Forgetting these is the #1 silent-corruption bug in hand-written TMA code.

**Two completion mechanisms.** PTX names them explicitly `[verified: §9.7.9.26.1]`:

| mechanism | qualifier | used by | who can wait |
|---|---|---|---|
| mbarrier-based | `.mbarrier::complete_tx::bytes` | global→shared TMA loads, `cp.async.bulk` in, CLC, `st.async`, `tcgen05.commit` (as `.mbarrier::arrive::one`) | any thread that can read the mbarrier |
| bulk async-group | `.bulk_group` | shared→global TMA stores, `cp.reduce.async.bulk` to global, `multimem.cp.*.async.bulk` | **only the issuing thread** |

The asymmetry matters: an *outbound* TMA store cannot be waited on by another warp. You
must `cp.async.bulk.commit_group; cp.async.bulk.wait_group[.read] N;` in the same thread
that issued it `[verified: §9.7.9.26.6]`.

**Directions supported.** From the PTX syntax blocks `[verified: §9.7.9.26.4.1, §9.7.9.26.5.2]`:

```
global      -> shared::cta       cp.async.bulk / cp.async.bulk.tensor   (mbarrier)
global      -> shared::cluster   cp.async.bulk / cp.async.bulk.tensor   (mbarrier, +multicast)
shared::cta -> shared::cluster   cp.async.bulk                          (mbarrier)   <- DSMEM push
shared::cta -> global            cp.async.bulk / cp.async.bulk.tensor   (bulk_group)
shared::cta -> global            cp.reduce.async.bulk[.tensor]          (bulk_group)
shared::cta -> multimem global   multimem.cp.[reduce.]async.bulk        (bulk_group)
registers   -> shared::cluster   st.async / red.async                   (mbarrier)   <- "STAS"
global      -> L2                cp.async.bulk.prefetch[.tensor]        (none)
```

There is **no** global→global and no shared→shared-within-CTA bulk copy.

---

## 2. TMA in depth

### 2.1 The tensor map (descriptor)

A tensor map is **a 128-byte opaque object** living in `.const`, `.param`, or `.global`
space `[verified: PTX ISA 9.3 §5.5.8, exact wording "The tensor-map is a 128-byte opaque
object"]`.

Two numbers get confused constantly, so pin them down:

| thing | value | source |
|---|---|---|
| `sizeof(CUtensorMap)` | **128** | `[measured]` — printed on this box |
| `alignof(CUtensorMap)` | **128** | `[measured]`; `cuda.h:3746` comment reads *"Requires compiler support for aligning to 128 bytes"*, struct is `alignas(128)` over `cuuint64_t opaque[16]` |
| documented API requirement | *"`tensorMap` address must be aligned to 64 bytes"* | `[verified]` `cuda.h:24806` and `:24975` |
| **SMEM destination of a bulk-tensor copy** | **128-byte aligned** | `[verified]` CUDA Programming Guide 13.3, §4.11.2.2 Table 23 |

So: the *type* is over-aligned to 128, the *API* only demands 64, and the thing that
genuinely needs 128 is the shared-memory landing buffer. Aligning the descriptor to 128
costs nothing and matches the type — do that and stop thinking about it.

**Creation** is host-side via the driver API `[verified: cuda.h:24961, 25146, 25333]`:

```c
CUresult cuTensorMapEncodeTiled(CUtensorMap*, CUtensorMapDataType, cuuint32_t tensorRank,
    void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim, const cuuint32_t* elementStrides,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
CUresult cuTensorMapEncodeIm2col(...);       // adds pixelBoxLower/UpperCorner, channelsPerPixel, pixelsPerColumn
CUresult cuTensorMapEncodeIm2colWide(...);   // adds CUtensorMapIm2ColWideMode {W, W128}
```

The recommended handoff is `const __grid_constant__ CUtensorMap` as a kernel parameter;
alternatives are `__constant__` + `cudaMemcpyToSymbol`, or a pointer into global memory
which then **requires `fence.proxy.tensormap::generic.acquire` per block before first use**
`[verified: CUDA Programming Guide 13.3 §4.11.2.2]`.

### 2.2 Descriptor constraints — documented and re-verified on this box

I encoded 18 descriptors against the real driver (595.71.05) and recorded the return codes
`[measured]`:

| configuration | result |
|---|---|
| `boxDim = {128, 8}`, uint8, no swizzle | `CUDA_SUCCESS` |
| `boxDim[0] = 256` | `CUDA_SUCCESS` (256 is the documented max) |
| `boxDim[0] = 257` | `CUDA_ERROR_INVALID_VALUE` |
| `boxDim[1] = 257` | `CUDA_ERROR_INVALID_VALUE` |
| `boxDim[0] = 8` (8-byte row, uint8) | `CUDA_ERROR_INVALID_VALUE` — inner box must be a multiple of 16 B |
| `boxDim[0] = 16` | `CUDA_SUCCESS` |
| `globalStrides[0] = 1000` (not ×16) | `CUDA_ERROR_INVALID_VALUE` |
| `globalAddress + 8` (not 16 B aligned) | `CUDA_ERROR_INVALID_VALUE` |
| `globalAddress + 16` | `CUDA_SUCCESS` |
| `SWIZZLE_128B` with `boxDim[0]=128` B | `CUDA_SUCCESS` |
| `SWIZZLE_128B` with `boxDim[0]=256` B | `CUDA_ERROR_INVALID_VALUE` — inner box must be ≤ swizzle span |
| `SWIZZLE_64B` with `boxDim[0]=128` B | `CUDA_ERROR_INVALID_VALUE` |
| `SWIZZLE_64B` with `boxDim[0]=64` B | `CUDA_SUCCESS` |
| `SWIZZLE_32B` with `boxDim[0]=32` B | `CUDA_SUCCESS` |
| `SWIZZLE_128B_ATOM_32B` / `_ATOM_64B` / `_ATOM_32B_FLIP_8B`, box 128 B | all `CUDA_SUCCESS` on sm_100 |
| `globalDim[0] = 1` | `CUDA_SUCCESS` |

Two notes. First, the three Blackwell-new swizzle sub-modes
(`CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B`, `_128B_ATOM_32B_FLIP_8B`, `_128B_ATOM_64B`,
`cuda.h:3796-3798`) are all accepted by this driver `[measured]` — they exist to match the
SMEM layouts `tcgen05.mma` wants for the narrow types. Second, the Programming Guide text
says *"All sizes must be greater than one"*; `globalDim[0] = 1` in fact encodes fine
`[measured]`. Treat the prose as advisory and the driver as authority.

Additional documented constraints worth having in front of you when writing a descriptor
`[verified: cuda.h:24833-24934, PTX §5.5.3]`:

- `tensorRank ∈ [1,5]`; ≥3 if interleaved; **3, 4, or 5 only** for im2col.
- `globalDim[i] ≤ 2^32`; `globalStrides[i]` multiple of 16 and `< 2^40`.
- `elementStrides[i] ∈ [1,8]`; `elementStrides[0]` is *ignored* when interleave is NONE
  (TMA has no stride on dimension 0).
- Bounding box size must be a multiple of 16 B and the bounding-box *address* must be
  16 B aligned (PTX §5.5.3.1).
- OOB fill: `ZERO` (default) or `NAN_REQUEST_ZERO_FMA`; the NaN mode is float-only and
  unavailable for the packed sub-byte types.
- L2 promotion: `NONE | 64B | 128B | 256B` — widens the DRAM fill granularity. Free knob,
  worth sweeping; nothing in our stack sets it today `[measured: no `cuTensorMapEncode*`
  call exists anywhere under `/home/aman/code/NotSglang/{glm,k3}-kernels`]`.

### 2.3 The instruction family, exactly

From PTX ISA 9.3 §9.7.9.26.5.2 `[verified]`:

```
// global -> shared::cta
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
    [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}{, cache_policy};
    .dst = {.shared::cta}   .src = {.global}   .dim = {.1d ... .5d}
    .completion_mechanism = {.mbarrier::complete_tx::bytes}
    .cta_group = {.cta_group::1, .cta_group::2}
    .load_mode = {.tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128}

// global -> shared::cluster  (adds .multicast::cluster + ctaMask)
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
    [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}{, ctaMask}{, cache_policy};

// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
    [tensorMap, tensorCoords], [srcMem]{, cache_policy};
    .completion_mechanism = {.bulk_group}
    .load_mode = {.tile, .tile::scatter4, .im2col_no_offs}
```

Version/arch gating `[verified: same section, plus the SM guards in
`cccl/cuda/__ptx/instructions/generated/cp_async_bulk_tensor*.h`]`:

| qualifier | PTX ISA | target |
|---|---|---|
| base instruction | 8.0 | sm_90+ |
| `.dst = .shared::cta` | 8.6 | sm_90+ |
| `.tile::gather4` | 8.6 | sm_100+ for `.shared::cta` dst; **sm_100a** for `.shared::cluster` dst (sm_100f from 8.8) |
| `.tile::scatter4` | 8.6 | sm_100a / sm_100f+ |
| `.im2col::w`, `.im2col::w::128` | 8.6 | sm_100a / sm_100f+ (`.im2col::w::128` is a/f only) |
| `.cta_group::{1,2}` | 8.6 | sm_100a / sm_100f+ |
| `.multicast::cluster` | 8.0 | works on sm_90+, but PTX warns it "may have substantially reduced performance" outside sm_90a/100a/100f/103a/103f/110a/110f |

**Our toolkit's reach.** `nvcc` 13.3.73 emits `.version 9.3` and honours `.target sm_100a`
`[measured]`. But `__cccl_ptx_isa` is **920** under CUDA 13.3
(`cccl/cuda/std/__cccl/ptx_isa.h:39-42` maps ≥13.2 → 920 and has no 13.3 branch)
`[verified]`. Consequence: any `cuda::ptx::` wrapper gated `#if __cccl_ptx_isa >= 930`
is compiled out even though ptxas would accept it. If you need a PTX-9.3-only form
(e.g. `mbarrier.init.layout::v1`), write inline asm.

### 2.4 Swizzle modes

`[verified: PTX ISA 9.3 §5.5.7; cuda.h:3791-3799]`

| mode | pattern | required starting alignment | max inner box |
|---|---|---|---|
| `NONE` | identity | — | — |
| `32B` | swap 16 B chunks within a 32 B span | 256 B | 32 B |
| `64B` | swap 16 B chunks within a 64 B span | 512 B | 64 B |
| `128B` | swap 16 B chunks within a 128 B span | — | 128 B |
| `128B_ATOM_32B` | swap 32 B chunks within 128 B | — | 128 B |
| `128B_ATOM_32B_FLIP_8B` | as above, plus swap low/high 8 B of every 16 B on alternate rows | — | 128 B |
| `128B_ATOM_64B` | swap 64 B chunks within 128 B | — | 128 B |
| `96B` (PTX-level only) | 16 B atomicity, no interleave, box ≤ 96 B, not with `.im2col::w*` | — | 96 B |

The pattern for 64 B swizzle, each cell = 16 B `[verified, §5.5.7]`:

```
row0: 0 1 2 3 4 5 6 7
row1: 1 0 3 2 5 4 7 6
row2: 2 3 0 1 6 7 4 5
row3: 3 2 1 0 7 6 5 4
```

The `_ATOM_*` variants are the ones that matter on SM100: `tcgen05` reads operands
straight out of SMEM via a matrix descriptor, and the atomicity has to match the MMA's
K-major fragment granularity for the narrow types. `[inferred]` — the PTX doc states the
patterns but does not spell out the tcgen05 pairing; CUTLASS's SM100 layout selectors are
the practical reference.

### 2.5 `.tile::gather4` / `.tile::scatter4` — the sparse-KV primitive

`[verified: PTX ISA 9.3 §5.5.3.4 and §9.7.9.26.5.2]`

- 2-D tensors only.
- `gather4` combines **four rows of the source into one destination 2-D tile**;
  `scatter4` does the reverse.
- `tensorCoords` becomes a fixed 5-vector: `{col_idx, row_idx0, row_idx1, row_idx2, row_idx3}`.
- Bounding-box size in dimension 1 **must be 1**; dimension 0 is the row length.
- Interleave layout is not supported.
- All other tile-mode rules apply (16 B multiples, OOB fill, etc.).

Example from the spec:

```
cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes
    [sMem5], [tensorMap6, {x0, y0, y1, y2, y3}], [mbar5];
```

This is the single most under-used instruction for our workload. DSA sparse MLA picks a
top-k set of KV positions; a paged KV cache stores rows at arbitrary page offsets. Both
are "gather N arbitrary rows of a 2-D [seq, head_dim] tensor". Today that is done with
per-thread `ld.global` (our engine's `mla_decode.cu` uses no bulk copies at all
`[measured: grep found no `cp.async.bulk` / `tensormap` in glm-kernels or k3-kernels]`).
`gather4` moves the address arithmetic into the TMA unit and gets the mbarrier completion
model for free.

### 2.6 im2col modes

`[verified: PTX ISA 9.3 §5.5.4, §5.5.5; cuda.h:25146, 25333]`

- `.im2col` treats the tensor as NWC (3-D) / NHWC (4-D) / NDHWC (5-D). The bounding box
  lives in DHW space and has **two fewer dimensions than the tensor**.
- Corner offsets are 16-bit signed with dimension-dependent ranges: 3-D `[-2^15, 2^15-1]`,
  4-D `[-2^7, 2^7-1]`, 5-D `[-2^4, 2^4-1]`.
- `im2colInfo` operand: `{i2cOffW, i2cOffH, i2cOffD}` for 5-D (size = `.dim` − 2), added
  to the filter-base coordinates.
- `Pixels-per-Column` (≤ 1024) sets how many NDHW elements are fetched;
  `Channels-per-Pixel` (≤ 256) sets the C extent.
- `.im2col::w` / `.im2col::w::128` restrict traversal to the W dimension with D and H box
  size 1. `w::128` **always loads exactly 128 pixels** and ignores `pixelsPerColumn`.
- `wHalo` (16-bit): `im2col::w` range `[0,512)`, `im2col::w::128` range `[0,32)` — halo
  pixels are loaded after every 32 elements in `w::128`. `wOffset` range `[0,32)` shifts
  the box along W so several SMEM buffers can hold different filter-tap alignments.
- `.im2col::w*` **requires a real swizzle**: no-swizzle and `128B_ATOM_32B_FLIP_8B` are
  both illegal for these modes `[verified: §5.5.5]`.

For a transformer decoder this is dead weight — we have no convolutions. It is listed here
so nobody spends a day trying to bend `im2col` into a KV gather; `tile::gather4` is the
instruction for that.

### 2.7 Multicast

`[verified: §9.7.9.26.5.2]`

`.multicast::cluster` takes a **16-bit `ctaMask`**; bit *i* selects the CTA with
`%cluster_ctarank == i`. The data lands at the *same SMEM offset* in every selected CTA.
Under `.cta_group::1` the mbarrier signal goes to the same offset in each destination CTA;
under `.cta_group::2` it goes to either the destination CTA or its peer, chosen by the
`%cluster_ctarank` parity of where `mbar` lives.

CUTLASS builds the masks from the cluster layout `[verified: sources]`:

```cpp
uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);
```
(`include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:540-541`)

For a 4×4×1 cluster, CTA 0's A-mask is `0x1111` (its column of the cluster) and B-mask is
`0x000f` (its row) `[reported: Colfax CUTLASS Blackwell cluster tutorial]`. Four CTAs that
would otherwise issue 8 tile loads issue 4.

**Multicast pays exactly when several CTAs need the same tile.** In a decode GEMM
`A[M=1..4, K] × B[K, N]` split over N, every CTA needs all of A — but A is a few KB total,
so there is nothing to save. Split over K, every CTA needs a *different* B. So at batch 1
multicast buys ~nothing `[inferred]`. At batch 64 (or 256 rows with EAGLE 3-1-4 × 64), A
tiles become 64–256 rows and are genuinely shared down the N axis, and multicast starts to
halve A traffic.

### 2.8 Stores and reductions

`[verified: §9.7.9.26.4.2, §9.7.9.26.5.3, §9.7.9.26.4.4, §9.7.9.26.4.5]`

```
cp.reduce.async.bulk.tensor.dim.global.shared::cta.redOp{.load_mode}.bulk_group
    [tensorMap, tensorCoords], [srcMem]{, cache_policy};
    .redOp = {.add, .min, .max, .inc, .dec, .and, .or, .xor}
```

Valid `redOp` × element type for the **tensor** form:

| redOp | element types |
|---|---|
| `.add` | `.u32 .s32 .u64 .f32 .f16 .bf16` |
| `.min .max` | `.u32 .s32 .u64 .s64 .f16 .bf16` |
| `.inc .dec` | `.u32` |
| `.and .or .xor` | `.b32 .b64` |

Note: no `.f64`, and no `.add.noftz` in the *tensor* form. The **non-tensor**
`cp.reduce.async.bulk` (shared::cta → global) is wider — it adds `.f64`, `.s64`, and the
`.add.noftz.{f16,bf16}` variant that preserves subnormals. Each reduction is element-wise
atomic with `.relaxed.gpu` ordering.

`multimem.cp.async.bulk` (PTX ISA 9.1, sm_90+; `.cp_mask` needs sm_100) and
`multimem.cp.reduce.async.bulk` (sm_90+) copy or reduce SMEM into a **multimem** address —
i.e. into every GPU's copy behind an NVLink multicast handle — in one bulk_group operation
`[verified]`. `size` must be a multiple of 16 and both addresses 16 B aligned. `.cp_mask`
supplies a 16-bit byte mask applied to every 16 B chunk.

Also SM100-only and cheap: **`st.bulk.weak.shared::cta [addr], size, initval;`**
(PTX ISA 8.6, sm_100) bulk-initialises a shared region to an immediate
`[verified: `cccl/.../generated/st_bulk.h`; PTX §9.7.9.14]`. Use it to zero accumulators,
masks, and mbarrier scratch instead of a strided `st.shared` loop.

### 2.9 Modifying a descriptor from the device

Needed whenever the tile shape or base pointer varies per launch — batched/grouped GEMM,
per-expert MoE weight pointers, variable-length KV. The sanctioned sequence
`[verified: CUDA Programming Guide 13.3 §4.11.2.2.2]`:

1. Host builds one `template_tensor_map` with `cuTensorMapEncodeTiled`.
2. Kernel copies it into `__shared__ alignas(128) CUtensorMap smem_tmap`.
3. Modify fields with `tensormap.replace.tile.<field>.shared::cta.b1024.{b32,b64}`.
4. `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.<scope>.sync.aligned
   [gbl], [smem], 128;` — copies 128 B out **and** establishes the proxy release.
5. Consumer does `fence.proxy.tensormap::generic.acquire.<scope> [gbl], 128;` **once per
   block**, then any thread in that block may use it after `__syncthreads()`.

Replaceable fields `[verified: PTX §9.7.9.27]`:
`global_address`, `rank`, `box_dim[0..4]`, `global_dim[0..4]`, `global_stride[0..3]`,
`element_stride[0..4]`, `elemtype`, `interleave_layout`, `swizzle_mode`,
**`swizzle_atomicity` (PTX 8.6, sm_100a+)**, `fill_mode`.

Gotchas, all `[verified]`:
- `.rank`'s `new_val` is **rank − 1** (zero-based).
- `tensormap.replace` is marked **arch-specific (`sm_90a`/`sm_100a`/…)** — you must build
  with an `a` or `f` target, not plain `compute_100`.
- **Only tiled-type maps can be modified on device.** im2col maps cannot.
- `size` for `cp_fenceproxy` must be exactly 128.
- Scope: `.gpu` if producer and consumer are on the same device; `.sys` if the map came
  from a host `cudaMemcpy`.
- The release/acquire is **per thread block** and does not propagate via
  `cluster.sync()` / `grid.sync()` / stream ordering — every consuming block must acquire.

### 2.10 What TMA actually costs on this box

Microbenchmark: one CTA, thread 0 issues `cp.async.bulk.tensor.2d.shared::cta.global.tile`
into a staged SMEM ring, `mbarrier.arrive.expect_tx` with the exact byte count, and
spins on `mbarrier.try_wait.parity`. `uint8` tensor, 128 B rows, no swizzle. Cycles from
`clock64()`; SM clock 1.965 GHz `[measured]`.

**Serialized (dependent) round trip** — next tile's coordinate depends on a byte of the
tile just landed, so no overlap is possible:

| tile | bytes | cycles/tile |
|---|---:|---:|
| 128×1 | 128 | 1034 |
| 128×2 | 256 | 1018 |
| 128×4 | 512 | 1030 |
| 128×8 | 1024 | 1061 |
| 128×16 | 2048 | 1080 |
| 128×32 | 4096 | 1140 |
| 128×64 | 8192 | 992 |
| 128×128 | 16384 | 887 |

**~1000 cycles ≈ 510 ns for one TMA hop, essentially flat from 128 B to 16 KB.** For
calibration, on the same GPU a dependent `ld.global` pointer chase costs 489 cycles over a
1 MiB working set and 690–810 cycles over 256 MiB `[measured]`; `__syncthreads()` with 128
threads is 21 cycles and an `mbarrier.arrive` + `try_wait.parity` round trip with 128
participants is 69 cycles `[measured]`.

**Pipelined (N stages in flight, single issuing thread, no block sync in the loop):**

| tile bytes | stages | cycles/tile | B/cycle | GB/s per SM |
|---:|---:|---:|---:|---:|
| 128 | 8 or 16 | 288.0 | 0.4 | 0.9 |
| 1024 | 4, 8, 16 | 288.0 | 3.6 | 7.0 |
| 8192 | 8 | 288.0 | 28.4 | 55.9 |
| 16384 | 4 | 288.0 | 56.9 | 111.8 |
| 8192 | 2 | 320 (L2) / 565 (2 GiB) | — | — |
| 16384 | 2 | 303 (L2) / 600 (2 GiB) | — | — |

The **288.0 cycles/tile floor is exact and does not move with stage count (8 → 16) or
transfer size (128 B → 16 KB)** `[measured]`. Two stages is not enough to cover the
latency; **four is** for everything I measured.

Aggregate over 148 CTAs (one per SM), 8 KB tiles, 8 stages: **8.06–8.09 TB/s**
`[measured]`. Honesty caveat: all CTAs read the *same* tile stream in that test, so L2
broadcast makes this an L2-bandwidth figure, not HBM; and
`148 × 8192 B / (288 cyc / 1.965 GHz) = 8.27 TB/s` is the issue-rate ceiling, so the
measurement is **98 % issue-bound**. What it does establish, and the part that matters:

> One TMA request costs ~288 SM cycles of issue+completion bookkeeping in the exact pattern
> CUTLASS uses (one `elect.sync` thread in a DMA warp issuing all loads). To be
> bandwidth-limited rather than request-limited you need **≥8 KB per request**.

`cp.async` (LDGSTS) comparison, same byte counts, 128 threads, `cp.async.cg` 16 B/thread,
`commit_group`/`wait_group 0` each iteration `[measured]`: 1117 cyc @ 2 KB, 822 cyc @ 8 KB,
802 cyc @ 16 KB. TMA wins on both fronts — it needs one thread instead of 128 and it
pipelines to 288 cyc/tile — but only once the tile is big enough that the fixed per-request
cost amortises.

---

## 3. mbarrier, precisely

`[verified: PTX ISA 9.3 §9.7.14.16 in full]`

### 3.1 State

An mbarrier is a **`.b64` object in `.shared`, 8-byte aligned**. It tracks:

1. current **primary** and **conditional** phases,
2. **pending arrival count** for the current phase,
3. **expected arrival count** for the next phase,
4. **tx-count** — outstanding asynchronous transaction bytes.

PTX ISA 9.3 introduces `.layout::v0` (default) and `.layout::v1`. Counts differ:

| layout | expected arrivals | pending arrivals | tx-count |
|---|---|---|---|
| `.layout::v0` | 1 … 2²⁰−1 | 0 … 2²⁰−1 | −(2²⁰−1) … 2²⁰−1 |
| `.layout::v1` | 1 … 2⁹−1 | 0 … 2⁹−1 | −(2²⁰−1) … 2²⁰−1 |

`.layout::v1` additionally carries a **payload report** per primary phase and supports a
"report-on" operation, which today only `fabric.try_*` instructions use. Under `v0` the
primary and conditional phases advance in unison. `.layout` requires PTX ISA 9.3 and
sm_90+; since `__cccl_ptx_isa` is 920 on our toolkit, the `cuda::ptx` wrappers will not
emit it — irrelevant for us, we want `v0`.

**tx-count can go negative.** That is not a bug in the spec; it means a `complete_tx`
arrived before the matching `expect_tx`. The phase completes only when pending arrivals
*and* tx-count are both exactly zero.

### 3.2 The operations

```
mbarrier.init{.layout}{.shared{::cta}}.b64 [addr], count;
mbarrier.inval{.shared{::cta}}.b64 [addr];

mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64          state, [addr]{, count};
mbarrier.arrive{.sem.scope}{.shared::cluster}.b64        _,     [addr]{, count};
mbarrier.arrive.expect_tx{.sem.scope}{.shared{::cta}}.b64 state, [addr], txCount;
mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64 state, [addr], count;
mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount;
mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;

mbarrier.test_wait{.parity}{.phase_type}{.sem.scope}{.ss}.b64 waitComplete, [addr], state|phaseParity;
mbarrier.try_wait {.parity}{.phase_type}{.sem.scope}{.ss}.b64 waitComplete, [addr], state|phaseParity {, timeHint};
    .sem   = {.acquire, .relaxed}      (default .acquire)
    .scope = {.cta, .cluster}          (default .cta)
```

Semantics to internalise `[verified]`:

- `expect_tx(n)` **increases** tx-count by n. `complete_tx(n)` **decreases** it by n. TMA's
  `.mbarrier::complete_tx::bytes` performs `complete_tx(bytes actually copied)`.
- With `.arrive.expect_tx`, the arrive count is **implicitly 1** — you cannot combine a
  count and an expect_tx in one instruction.
- `.release` is the default `.sem` for `arrive`; `.acquire` is the default for the waits.
  `.sem` and `.scope` **must be specified together**.
- On a `.shared::cluster` (remote) barrier, `mbarrier.arrive` **cannot return state** —
  the sink `_` is mandatory. Only `arrive`, `arrive_drop`, `expect_tx`, `complete_tx` work
  remotely; `init`, `inval`, and all waits do **not**.
- `test_wait` is non-blocking (spin yourself). `try_wait` may **suspend the thread** and
  resume either on completion or after a system-defined timeout; the optional `timeHint`
  is in nanoseconds.
- `.parity` takes 0 for even phases and 1 for odd. **Only the current incomplete phase
  (returns False) and the immediately preceding phase (returns True) are valid to test.**
- Per primary phase, **at least one wait must return True before any thread arrives in the
  next phase.** This is the rule that makes N-stage ring buffers correct and 1-stage ones
  broken.

### 3.3 The bugs, in order of how often they bite

1. **Wrong `expect_tx` byte count.** The barrier never completes (too many expected) or
   completes early and you read garbage (too few). For a multicast load, the byte count is
   per-destination-CTA, and for `cta_group::2` CUTLASS multiplies by the CTA-pair size
   (`sm100_mma_warpspecialized.hpp:238-240` computes `TmaTransactionBytes` as
   `size(AtomThrShapeMNK) * cosize(...)` for A **and** B) `[verified]`.
2. **Forgetting `fence.proxy.async` before a shared→global TMA store.** Ordinary
   `st.shared` writes are in the generic proxy; the TMA engine reads in the async proxy.
   Without the fence the store can race the writes. The Programming Guide's own example
   does `ptx::fence_proxy_async(ptx::space_shared); __syncthreads();` before issuing
   `[verified]`.
3. **Phase-parity drift.** The parity is *your* bookkeeping; hardware does not tell you.
   Every stage in a ring needs its own parity bit, flipped on every wait that returns True.
   Off-by-one here deadlocks or silently reads the previous iteration's data.
4. **Waiting on a phase more than one behind.** Undefined. If a consumer can fall two
   phases behind, your pipeline depth is wrong, not your barrier.
5. **`mbarrier.init` on a live barrier.** Explicitly undefined; you must `mbarrier.inval`
   first before repurposing the memory.
6. **Missing init fence.** After `mbarrier.init` you need `fence.mbarrier_init.release.cluster`
   (or a `__syncthreads()` within a CTA) before any other CTA in the cluster touches it.
   CUTLASS calls `cutlass::arch::fence_barrier_init()` in every pipeline constructor
   `[verified: `sm100_pipeline.hpp:150, 990`]`.
7. **`try_wait` timeout treated as completion.** `try_wait` returning False is normal —
   it means "resumed early". Loop.
8. **`arrive` with the wrong participant count for warp-specialised kernels.** With a
   1-thread producer and a 4-warp consumer the counts are asymmetric. CUTLASS's CLC
   pipeline sets `producer_arv_count = 1` and
   `consumer_arv_count = NumSchedThreads + cluster_size * (…)`
   `[verified: `sm100_gemm_tma_warpspecialized.hpp:509-514`]`.
9. **`setmaxnreg` inside a divergent branch.** `.aligned` requires all warps of the
   warpgroup to execute the same instruction; violating it is undefined (see §6.2).

---

## 4. Thread block clusters (CGA)

### 4.1 Launch and limits — measured here

Clusters group thread blocks that are **guaranteed co-resident on one GPC**, which is what
makes distributed shared memory and cross-CTA mbarriers possible `[verified: CUDA
Programming Guide 13.3 §1.2.2.1.1]`.

Measured on this node with a trivial kernel (8 registers, 0 static SMEM) `[measured]`:

| query | portable (default) | `cudaFuncAttributeNonPortableClusterSizeAllowed = 1` |
|---|---:|---:|
| `cudaOccupancyMaxPotentialClusterSize` (block 128 or 256) | **8** | **16** |
| `cudaOccupancyMaxActiveClusters` at that size | 142 | 66 |
| actual launch, `clusterDim.x` = 2 / 4 / 8 | OK / OK / OK | OK / OK / OK |
| actual launch, `clusterDim.x` = 16 | **`cudaErrorInvalidClusterSize`** | OK |
| actual launch, `clusterDim.x` = 32 | error | error |

So B200 does support 16-CTA clusters, but only after opting out of portability. `cuda.h`
states the portable size for sm_90 is 8 and "may increase for future compute capabilities"
— it has not, at least not portably `[verified: cuda.h:1142-1161]`.

Related launch knobs `[verified: cuda.h / driver_types.h]`:
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` / `cudaLaunchAttributeClusterDimension`
- `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE` with
  `CU_CLUSTER_SCHEDULING_POLICY_{DEFAULT, SPREAD, LOAD_BALANCING}` (cuda.h:2148-2150)
- `CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION` / `cudaLaunchAttributePreferredClusterDimension`
  (= 11) — the **preferred/fallback cluster** pair: each preferred dim must be a multiple
  of the corresponding `clusterDim` and divide the grid dim; `z` must equal `clusterDim.z`
  (cuda.h:2448-2465). This lets the driver run a big cluster when the GPC has room and fall
  back to the smaller one otherwise. `__cluster_dims__(x,y,z)` sets it at compile time.
- `cudaOccupancyMaxPotentialClusterSize` / `cudaOccupancyMaxActiveClusters`.

Other measured device facts on this box `[measured via `cuDeviceGetAttribute`]`:
148 SMs · 233,472 B shared per SM · 232,448 B opt-in shared per block · 65,536 32-bit
registers per SM · 2048 threads per SM · 32 blocks per SM · L2 = 132,644,864 B (126.5 MiB)
· 7680-bit bus · 3996 MHz memory clock · 1965 MHz SM clock · CC 10.0.

### 4.2 Distributed shared memory

Every CTA in a cluster can read, write, and do atomics in every other CTA's shared memory.
Total DSMEM = cluster size × per-block SMEM `[verified: Programming Guide §2.3 DSMEM]`.

Address translation `[verified: PTX §9.7.9.24, §9.7.9.25]`:

```
mapa{.shared::cluster}.{u32,u64}  d, a, b;   // address `a` as seen in CTA of rank `b`
getctarank{.shared::cluster}.{u32,u64} d, a; // which CTA rank owns address `a`
```

Cooperative Groups exposes these as `cluster.map_shared_rank(ptr, rank)` and
`cluster.block_rank()`. Synchronisation is `barrier.cluster.arrive[.{release,relaxed}]` /
`barrier.cluster.wait[.acquire]`, i.e. `cg::cluster_group::sync()`
`[verified: `cccl/.../generated/barrier_cluster.h`]`.

Hard rule: **DSMEM access requires all blocks to still exist.** You must `cluster.sync()`
before first remote access and guarantee no CTA exits while another is reading it. This is
why CUTLASS's mainloop has a `load_tail` that drains all pipeline stages purely "to avoid
early exit of ctas in Cluster" `[verified: `sm100_mma_warpspecialized.hpp:652-660`]`.

Data movement into DSMEM: `cp.async.bulk.shared::cluster.shared::cta` (bulk, mbarrier
completion), `st.async` / `red.async` for 4/8/16-byte register→DSMEM writes (CUDA calls
this **STAS**), and TMA loads with `.dst = .shared::cluster` `[verified]`.

### 4.3 The CTA pair and what `cta_group::2` means

`[verified: PTX ISA 9.3 §9.7.17.5.1-5.2]`

> "Any 2 CTAs within the cluster whose `%cluster_ctarank` differs by the last bit only is
> said to form a CTA pair."

Even-ranked CTA = leader. `tcgen05` operations run at either single-CTA (`cta_group::1`) or
CTA-pair (`cta_group::2`) granularity; at pair granularity the **Tensor Memory of both CTAs
is accessed**, and the accumulator splits along M. Issue granularity `[verified: Table 49]`:

| operation | `cta_group::1` | `cta_group::2` |
|---|---|---|
| `mma`, `cp`, `shift`, `commit` | one thread in the current CTA | one thread **of the pair**; the peer must be live and not exited |
| `alloc`, `dealloc`, `relinquish_alloc_permit` | one warp in the current CTA | one warp in **each** of the peer CTAs, collectively |

Consequences for us:

- **All `tcgen05` instructions in a kernel must use the same `.cta_group`** `[verified:
  tcgen05.commit description]`. You cannot mix 1-SM and 2-SM MMA in one kernel.
- 2-SM MMA supports M ∈ {128, 256} `[reported: CUTLASS Blackwell tutorial; the PTX shape
  tables are consistent]`. At decode, M = 1–4 rows. The tensor core is idle either way;
  `cta_group::2` just doubles the wasted granularity and adds a cross-CTA dependency.
- `cp.async.bulk.tensor…cta_group::2` lets a TMA load signal the **peer's** mbarrier. The
  address trick CUTLASS uses to let an odd CTA arrive at the even CTA's barrier is to
  clear bit 24 of the SMEM address: `smem_int_mbar & 0xFEFFFFFF` — i.e. the CTA-pair index
  lives in bit 24 of the cluster SMEM address `[reported: Colfax; not stated in the PTX
  doc, and I did not verify it on hardware]`.
- `tcgen05.commit.cta_group::N.mbarrier::arrive::one.b64 [mbar]{, ctaMask}` is how MMA
  completion signals an mbarrier — count 1, **cluster scope** `[verified]`.

---

## 5. Cluster Launch Control

New on Blackwell (CC 10.0). It is a hardware work queue you can steal from.

### 5.1 The instructions

`[verified: PTX ISA 9.3 §9.7.14.18-19]`

```
clusterlaunchcontrol.try_cancel.async{.space}.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128
    [addr], [mbar];                                 // addr: 16-byte, naturally aligned, .shared::cta

clusterlaunchcontrol.query_cancel.is_canceled.pred.b128            p, handle;
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x,y,z,_}, handle;
clusterlaunchcontrol.query_cancel.get_first_ctaid::{x,y,z}.b32.b128 r, handle;
```

- `try_cancel` **atomically requests cancellation of a cluster that has not started running
  yet**, and writes a **16-byte opaque response** to `addr` asynchronously, tracked by
  `mbar` via `complete_tx` at **cluster scope**.
- On success the response carries the `ctaid` of the first CTA of the cancelled cluster,
  and **no other successful `try_cancel` in the grid will ever return that id**.
- `.multicast::cluster::all` writes the same response into the same local SMEM address of
  *every* CTA in the requesting cluster, each signalled on its own mbarrier. Requires
  sm_100a/sm_100f (plain `try_cancel` requires only sm_100).
- Behaviour is **undefined** if any CTA in the cluster has exited when a multicast
  `try_cancel` is issued.
- Behaviour is **undefined** if you issue another `try_cancel` after having *observed* a
  previous one fail. Issuing two before observing either is fine — the CUDA guide spells
  out both cases `[verified: Programming Guide §4.12.1.2]`.
- `get_first_ctaid` on a failed response is undefined.

### 5.2 The CUDA-level pattern

`[verified: CUDA Programming Guide 13.3 §4.12.2.1, quoted structure]`

```cpp
__shared__ uint4 result; __shared__ uint64_t bar; int phase = 0;
if (cg::thread_block::thread_rank() == 0) ptx::mbarrier_init(&bar, 1);

int bx = blockIdx.x;                        // first tile is free — it is your own blockIdx
while (true) {
  __syncthreads();                          // protects `result` from the next iteration
  if (cg::thread_block::thread_rank() == 0) {
    ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
    cg::invoke_one(cg::coalesced_threads(), [&]{ ptx::clusterlaunchcontrol_try_cancel(&result, &bar); });
    ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
  }
  /* ---- do the current tile's work here, concurrently with the query ---- */
  while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase)) {}
  phase ^= 1;
  if (!ptx::clusterlaunchcontrol_query_cancel_is_canceled(result)) break;
  bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
  ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
}
```

The transaction size is **16 bytes** (`sizeof(uint4)`). In the cluster case the query is
issued by **one thread of the whole cluster** with
`clusterlaunchcontrol_try_cancel_multicast`, each CTA arrives on its **own** barrier with
`ptx::scope_cluster`, and each CTA must add its local block index to the returned base
ctaid. `cg::cluster_group::sync()` is required first so that all CTAs are known live
`[verified]`.

### 5.3 How CUTLASS wires it

`[verified: `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`,
`sm100_tile_scheduler.hpp`, `pipeline/sm100_pipeline.hpp`, plus
docs.nvidia.com/cutlass Blackwell CLC page]`

- Grid is launched with **as many CTAs as output tiles** — not persistent-sized. Each
  worker's own `blockIdx` is its first tile, free and static; every subsequent tile comes
  from a CLC query.
- `PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>` with
  `transaction_bytes = 16`, `producer_arv_count = 1`, `producer_blockid = 0`,
  `consumer_arv_count = NumSchedThreads + cluster_size * (...)`.
- **Depth 3** — "to overlap the CLC operations of multiple waves for latency hiding". The
  source comment reads: *"CLC pipeline depth — determines how many waves (stages-1) a warp
  can race ahead"* (`sm100_gemm_tma_warpspecialized.hpp:116-118`).
- There is a second pipeline, `CLCThrottlePipeline` (`PipelineAsync<Stages>`), whose only
  job is to **throttle CLC queries "to mitigate workload imbalance caused by skews among
  persistent workers"** (`sm100_gemm_tma_warpspecialized.hpp:692`). Producer is the
  Mainloop Load warp, consumer is the Scheduler warp. Worth knowing that NVIDIA needed a
  brake on the work-stealing rate.
- The raw query, verbatim from `sm100_tile_scheduler.hpp:394-407`:

```cpp
asm volatile(
  "clusterlaunchcontrol.try_cancel.async.shared::cta"
  ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
  :: "r"(result_addr), "r"(mbarrier_addr));
```

and the decode at `:415-430` uses `ld.shared.b128` + `query_cancel.is_canceled.pred` +
`@p1 query_cancel.get_first_ctaid.v4`, followed by `fence_view_async_shared()`.

### 5.4 What CLC does and does not fix

The CUDA guide frames CLC as combining the two classic scheduling strategies `[verified:
§4.12, table reproduced]`:

| | fixed work/block | fixed #blocks (grid-stride) | CLC |
|---|---|---|---|
| reduced launch/prologue overhead | ✗ | ✓ | ✓ |
| preemption by higher-priority kernel | ✓ | ✗ | ✓ |
| load balancing | ✓ | ✗ | ✓ |

**Wave quantisation.** CLC helps when tile count ≫ SM count and per-tile cost varies, or
when some SMs are stolen by another kernel. It cannot manufacture parallelism. If a decode
GEMM produces, say, 96 output tiles on a 148-SM GPU, there is exactly one wave, 52 SMs are
idle, and no amount of work stealing changes that — the fix there is smaller tiles
(split-K, or narrower N tiles), not CLC `[inferred, but directly implied by the CLC rules
in §5.1: `try_cancel` only ever returns ClcIDs that exist in the launched grid]`.

Where CLC genuinely helps us: we run **many concurrent short kernels** (456 launches per
token on one rank) and multiple CUDA streams. A persistent kernel with a static scheduler
will stall behind whatever else occupies SMs; CLC lets it exit gracefully and lets the
other kernel in `[inferred]`.

---

## 6. Warp specialization

### 6.1 The SM100 role assignment

From CUTLASS's Blackwell dense GEMM `[verified: `sm100_gemm_tma_warpspecialized.hpp:234-240`
and the CUTLASS Blackwell CLC doc]`:

```cpp
enum class WarpCategory : int32_t {
  MMA = 0, Sched = 1, MainloopLoad = 2, EpilogueLoad = 3, Epilogue = 4
};
static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp; // 1 warp
static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp; // 1 warp
static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp; // 1 warp
static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp; // 1 warp
static constexpr uint32_t MaxThreadsPerBlock = Sched + MainloopLoad + MMA + EpilogueLoad + Epilogue;
static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
```

| warp | role |
|---:|---|
| 0 | MMA (issues `tcgen05.mma`, one thread) |
| 1 | Scheduler (CLC producer *and* consumer) |
| 2 | Mainloop Load (TMA producer) |
| 3 | Epilogue Load (C matrix) |
| 4–7 | Epilogue (consume TMEM accumulators, quantise, store) |

Five pipelines coordinate them `[verified: `sm100_gemm_tma_warpspecialized.hpp:187-200`]`:
`MainloopPipeline`, `EpiLoadPipeline`, `LoadOrderBarrier`, `CLCPipeline`,
`AccumulatorPipeline`, `CLCThrottlePipeline`, plus an `arch::ClusterBarrier tmem_dealloc`.

The structural change from Hopper is worth stating plainly. On SM90 a *warpgroup* (128
threads) issues `wgmma` and the accumulator lives in **registers**, so the consumer
warpgroup needs a huge register budget and the producer needs almost none — hence
`setmaxnreg`. On SM100 a **single thread** issues `tcgen05.mma` and the accumulator lives
in **TMEM** (256 KB/SM, 512 columns × 128 lanes of 32-bit cells) `[reported: Jarmusch &
Chandrasekaran, arXiv:2512.02189 §V.A]`. The register pressure asymmetry largely disappears.

**Verified consequence:** `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`
(964 lines) contains **no `setmaxnreg`, no `warpgroup_reg_alloc`, no `warpgroup_reg_dealloc`**
`[measured: grep over the fetched file]`. The SM90 pingpong kernel does:

```cpp
// include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp:147-148
static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40  : 24;
static constexpr uint32_t MmaRegisterRequirement  = !HeavyRegisterPressure ? 232 : 240;
```

So: **do not port Hopper's register-reallocation recipe to SM100 reflexively.** With
MaxThreadsPerBlock = 256 (4 single warps + 4 epilogue warps) and 65,536 registers per SM,
uniform allocation already gives 256 registers/thread at one block per SM `[measured device
attribute + arithmetic]`.

### 6.2 `setmaxnreg`, when you do need it

`[verified: PTX ISA 9.3 §9.7.20.5]`

```
setmaxnreg.inc.sync.aligned.u32 imm-reg-count;
setmaxnreg.dec.sync.aligned.u32 imm-reg-count;
```

- `imm-reg-count` ∈ **[24, 256], multiple of 8**.
- A **per-CTA pool** of spare registers is maintained; `.dec` releases into it, `.inc`
  takes from it and **blocks until enough are available**.
- Registers obtained by `.inc` have **undefined contents**.
- **All warps of the warpgroup must execute the same `setmaxnreg`**, and must synchronise
  explicitly before the next one. Divergent execution is undefined.
- `.dec` requires the current max ≥ the new value; `.inc` requires current max ≤ new value.
- Changes always happen at the **tail end** of the register file.
- **Requires the kernel to have been launched with a valid max-registers-per-thread**
  (`-maxrregcount` / `__launch_bounds__`); otherwise the instruction "may have no effect."
  This is a silent failure mode: no error, just no benefit.
- Arch-specific: sm_90a, sm_100a/f, sm_103a/f, sm_110a/f, sm_120a/f, sm_121a/f — **not**
  plain `sm_100`.

### 6.3 Producer idioms

**Elect one thread per warp** rather than branching on `threadIdx.x == 0`
`[verified: `cccl/.../generated/elect_sync.h`]`:

```
{ .reg .pred P_OUT; elect.sync _|P_OUT, %membermask; selp.b32 %0, 1, 0, P_OUT; }
```
CUTLASS uses `cute::elect_one_sync()` around every TMA issue and every CLC query
`[verified: `sm100_mma_warpspecialized.hpp:619`, `sm100_tile_scheduler.hpp:445`]`. The
CUDA guide's `is_elected()` helper broadcasts warp 0's id with `__shfl_sync` then calls
`elect.sync`, so exactly one thread in the whole block issues `[verified]`.

**The DMA warp loop**, distilled from `sm100_mma_warpspecialized.hpp:597-650` `[verified]`:

```cpp
auto tok = pipeline.producer_try_acquire(state);
while (k_tile_count > 0) {
  pipeline.producer_acquire(state, tok);              // wait for empty barrier, stage `state.index()`
  auto* tma_bar = pipeline.producer_get_barrier(state);
  int stage = state.index();
  ++state;
  tok = pipeline.producer_try_acquire(state);         // start the *next* acquire before issuing
  if (cute::elect_one_sync()) {
    copy(tma_load_a.with(*tma_bar, mcast_mask_a), gA(_, *k_tile), sA(_, stage));
    copy(tma_load_b.with(*tma_bar, mcast_mask_b), gB(_, *k_tile), sB(_, stage));
  }
  --k_tile_count; ++k_tile;
}
```

Two details that are easy to miss: the *next* stage's `try_acquire` is started before the
current issue (so the acquire latency overlaps the TMA), and the loop is
`CUTLASS_PRAGMA_NO_UNROLL` — unrolling it would multiply live barrier state.

**Stage counts.** `DispatchPolicy::Stages >= 2` is `static_assert`ed
(`sm100_mma_warpspecialized.hpp:191`), CLC uses 3, and the SM100 dispatch policies carry a
separate `AccumulatorPipelineStageCount` for TMEM double-buffering `[verified]`. My
measurement says the mainloop wants **≥4** to hide TMA latency at 8–16 KB tiles
`[measured, §2.10]`.

---

## 7. Applying it: batch-1 decode vs batch 64+

### 7.1 The arithmetic of our C1 profile

Derived from the numbers in the brief (nsys, 8473 ms window, all 8 ranks, device 0 shown)
`[measured/derived]`:

| quantity | value |
|---|---:|
| kernel launches on device 0 | 1,411,149 |
| tokens in the window (365 tok/s) | ~3093 |
| **launches per output token, per rank** | **~456** |
| total kernel time / launches | **6.78 µs mean** = ~13,300 SM cycles |
| device-0 busy | 7090 ms / 8473 ms = 83.7 % |
| busy time per token | 2.29 ms (wall 2.74 ms) |

Per-family means and per-token launch counts:

| family | mean kernel | launches/token |
|---|---:|---:|
| dense GEMM (`nvjet_sm100`) | 5.95 µs | 193 |
| MoE NVFP4 expert GEMMs | 7.35 µs | 82 |
| collectives | 16.4 µs | 37 |
| DSA sparse MLA attention | 9.70 µs | 35 |
| DSA indexer | 6.17 µs | 29 |
| elementwise | 2.43 µs | 47 |
| quant | 2.98 µs | 25 |
| norm | 2.90 µs | 5 |

### 7.2 What asynchrony buys at batch 1

**It does not shorten a single hop.** A TMA round trip is ~1000 cycles; a dependent
`ld.global` is 489–810 `[measured]`. If a kernel's critical path is *one* load followed by
*one* small MMA followed by *one* store, TMA is a small loss, not a win.

**It buys overlap, and overlap needs depth.** From §2.10: 2 stages leaves 320–600
cycles/tile on the table; 4 stages reaches the 288-cycle floor. But a decode-shaped GEMM
often has only 2–8 K-tiles total, so a 4-stage pipeline **never reaches steady state** —
you pay the fill and then the drain, and the "steady state" is one or two iterations long
`[inferred from the measured fill cost and the tile arithmetic]`.

**Quantified prologue tax.** Pipeline fill ≈ one full latency plus (stages−1) issues:
1000 + 3 × 288 ≈ **1860 cycles ≈ 0.95 µs**.

| kernel family | mean | fill as % of kernel |
|---|---:|---:|
| elementwise (2.43 µs) | 4,775 cyc | **39 %** |
| quant (2.98 µs) | 5,856 cyc | 32 % |
| dense GEMM (5.95 µs) | 11,691 cyc | 16 % |
| **all kernels (6.78 µs)** | 13,323 cyc | **14 %** |
| collectives (16.4 µs) | 32,226 cyc | 6 % |

That is the honest ceiling on what better pipelining can return at C1: roughly the fill
cost of whatever you replace, and only if the kernel is currently *serial* on its loads.

**What is worth doing at batch 1, in order:**

1. **`tile::gather4` for the DSA sparse-MLA KV gather and the indexer** (16.7 % of GPU
   time combined). This is not about overlap — it is about replacing per-thread scattered
   `ld.global` with 1 instruction per 4 rows and getting coalesced, descriptor-driven,
   OOB-filled loads. `[inferred, but the instruction's shape is an exact match]`
2. **Make every TMA request ≥8 KB.** With 227 KB of SMEM per block on this device
   `[measured]` there is no reason to use 1 KB boxes. A 128×128 NVFP4 weight tile is 8 KB.
3. **`cp.async.bulk.prefetch.tensor` the *next* layer's weights into L2 while the current
   layer computes.** L2 is 126.5 MiB `[measured]` — large enough to hold a meaningful slice
   of per-GPU expert weights. At batch 1 every token re-reads the weights from HBM; this is
   the one place where the workload is genuinely bandwidth-bound. `[inferred — I have not
   measured our per-layer weight footprint and will not guess it]`
4. **`st.bulk` for SMEM initialisation** in the elementwise/quant kernels (47 + 25
   launches/token, 2.4–3.0 µs each — these are small enough that a strided zeroing loop is
   a visible fraction).
5. **Fuse.** 456 kernels/token at 6.78 µs mean is the dominant structural cost. No amount
   of intra-kernel asynchrony beats deleting a launch. CUTLASS's SM100 kernels already
   plumb Programmatic Dependent Launch (`cutlass::arch::IsGdcGloballyEnabled`,
   `IsGdcEnabled` in `sm100_gemm_tma_warpspecialized.hpp:135`) `[verified]`, and CUDA 13.3
   documents PDL in §4.5 of the Programming Guide `[verified: TOC]` — that is the mechanism
   for overlapping the tail of kernel *n* with the prologue of kernel *n+1* without fusing
   them.

**What is not worth doing at batch 1:**

- `cta_group::2` / 2-SM UMMA — wrong granularity (§4.3).
- TMA multicast — nothing is shared across CTAs when M ≈ 1 (§2.7).
- CLC-based persistent scheduling for load balance — one wave, nothing to steal (§5.4).
- Deep (8+) pipelines — you pay the fill and never amortise it.

### 7.3 What changes at batch 64+

Our C64 aggregate is 40.8k tok/s, so the same kernels are being amortised over ~64×
more rows. Concretely:

- **M grows to 64–256 rows** (×4 with EAGLE 3-1-4 draft tokens). `tcgen05.mma` with
  M = 128 becomes fully utilised, and **`cta_group::2` with M = 256 becomes the right
  choice**, doubling arithmetic intensity per CTA pair because each CTA loads only half the
  operand tile `[reported: CUTLASS Blackwell tutorial; consistent with the PTX pair
  semantics]`.
- **TMA multicast starts paying**: A tiles are now large and shared across the N-axis CTAs
  of a cluster. A 4×4 cluster halves operand traffic (8 loads → 4) `[reported]`.
- **Tile counts exceed 148**, so there are multiple waves — and *now* CLC's work stealing
  and the preferred/fallback cluster mechanism address real quantisation. This is where
  a persistent CLC scheduler is worth the complexity.
- **Pipeline depth pays off**: K-loops get long enough that steady state dominates the fill,
  so 4–6 stages amortise properly and the 288-cycle issue floor becomes the real limit.
- **The bottleneck flips from latency to bandwidth**, which is exactly the regime TMA was
  designed for. Expect the ranking of optimisations to invert: at C1 fuse and gather; at
  C64 widen tiles, multicast, and go 2-SM.

### 7.4 Against TileRT (~500 tok/s vs our 365)

TileRT's stated thesis is tile-granular overlap inside operators rather than micro-batch
overlap `[reported: our own `hotspots-and-optimization-ledger.md` §2c]`. Read through this
document, "tile-level overlap" on SM100 means precisely: a warp-specialised kernel where a
DMA warp keeps ≥4 TMA requests of ≥8 KB in flight against an mbarrier ring, MMA is issued
by one thread into TMEM, and the epilogue warps drain TMEM while the next tile loads —
i.e., the CUTLASS SM100 structure, applied across operator boundaries instead of within
one GEMM. The 456-launches-per-token figure is the gap that structure closes: each fused
mega-kernel removes both a launch and a pipeline fill. That reframes the target as
**"reduce launches per token"**, which is measurable today, rather than "make kernels
faster", which we have no roofline for yet.

---

## 8. Claims that did not survive

Recording these because a wrong belief costs more than a missing one.

1. **"`nvcc -arch=sm_100a` silently degrades to `.target sm_100` and rejects tcgen05."**
   Stated in `/home/aman/code/NotSglang/personal_docs/kimi-k3/AGENT-HANDOFF-sm100.md:311`.
   **Does not reproduce on CUDA 13.3** `[measured]`:
   ```
   nvcc -ptx -arch=sm_100a           ->  .version 9.3 / .target sm_100a
   nvcc -ptx -gencode arch=compute_100a,code=sm_100a -> .version 9.3 / .target sm_100a
   nvcc -ptx -arch=sm_100            ->  .version 9.3 / .target sm_100
   ```
   Either it was true of an earlier toolkit or it was never true. Use whichever form you
   like on 13.3; verify with `-ptx` if in doubt.
2. **"The tensor map must be 128-byte aligned."** The *type* is `alignas(128)` and the
   object is 128 bytes, but the documented API requirement is **64** bytes
   `[verified: cuda.h:24806]`. The thing that must be 128-byte aligned is the **shared
   memory destination** `[verified: Programming Guide Table 23]`. Both statements circulate
   as "the 128-byte alignment rule" and they are different rules.
3. **"All tensor dimensions must be greater than one."** Programming Guide prose; the
   driver accepts `globalDim[0] = 1` `[measured]`.
4. **"`__cccl_ptx_isa` tracks the toolkit."** It is **920** under CUDA 13.3 while ptxas
   emits **9.3** `[verified + measured]`. PTX-9.3-only wrappers are compiled out.
5. **`multimem.cp.async.bulk` does not appear in the currently-served
   `docs.nvidia.com/cuda/parallel-thread-execution/index.html`** but *is* in
   `docs.nvidia.com/cuda/archive/13.3.0/parallel-thread-execution/` `[measured: 0 vs 16
   occurrences in the two downloads, taken minutes apart]`. The live doc URL is not stable;
   pin the archive URL for the toolkit version you build against.

---

## 9. Quick-reference gotcha checklist

- [ ] SMEM destination of every bulk-tensor copy is `alignas(128)`.
- [ ] `expect_tx` byte count exactly matches what the hardware will `complete_tx`
      (per destination CTA for multicast; ×2 for `cta_group::2` operand pairs).
- [ ] Every mbarrier ring stage has its own parity bit, flipped only on a True wait.
- [ ] `fence.proxy.async` + `__syncthreads()` before any shared→global TMA store.
- [ ] Outbound TMA is waited on by the **issuing thread** with
      `cp.async.bulk.commit_group` / `wait_group[.read]`, never by another warp.
- [ ] `fence.mbarrier_init.release.cluster` (or `__syncthreads`) after `mbarrier.init`.
- [ ] `mbarrier.inval` before reusing barrier memory.
- [ ] `setmaxnreg` only under a launch with an explicit register bound, never in
      divergent code, value in [24,256] and a multiple of 8.
- [ ] Producer warp starts the *next* `producer_try_acquire` before issuing the current copy.
- [ ] `load_tail` / `producer_tail` drains all stages so no CTA exits while a cluster peer
      is still reading its DSMEM.
- [ ] `cluster.sync()` before any DSMEM or multicast CLC access.
- [ ] Never issue a second `try_cancel` after observing a failed one.
- [ ] All `tcgen05` in a kernel share one `.cta_group`.
- [ ] Build with an `a`/`f` target for `tensormap.replace`, `setmaxnreg`, `cta_group`,
      `tile::gather4` to `.shared::cluster`, `im2col::w::128`, and
      `try_cancel.multicast::cluster::all`.

---

## Sources

**Local files read**
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda.h` — `CUtensorMap` (3743-3760),
  swizzle/interleave/L2/OOB enums (3791-3830), `cuTensorMapEncodeTiled` doc + signature
  (24794-24961), `cuTensorMapEncodeIm2col` (24963-25146), `cuTensorMapEncodeIm2colWide`
  (25148-25333), cluster function attributes (1140-1175), launch attributes (2148-2150,
  2224, 2271, 2440-2465)
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/`
  — `cp_async_bulk_tensor.h`, `cp_async_bulk_tensor_multicast.h`,
  `cp_async_bulk_tensor_gather_scatter.h`, `clusterlaunchcontrol.h`, `mbarrier_*.h`,
  `setmaxnreg.h`, `elect_sync.h`, `st_bulk.h`, `barrier_cluster.h`, `tensormap_replace.h`,
  `cp_reduce_async_bulk_tensor.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/std/__cccl/ptx_isa.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/bin/{nvcc,ptxas}` (13.3.73, built 2026-06-09)
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/NotSglang/personal_docs/kimi-k3/AGENT-HANDOFF-sm100.md`
- `/home/aman/code/NotSglang/{glm-kernels,k3-kernels}/` (grepped: no TMA/mbarrier/cluster
  usage present)

**Web (fetched and read, not recalled)**
- PTX ISA 9.3 — https://docs.nvidia.com/cuda/archive/13.3.0/parallel-thread-execution/index.html
  §5.5 (Tensors, im2col, swizzling, tensor-map), §9.7.9.24-25 (mapa, getctarank),
  §9.7.9.26 (async copy, bulk, tensor copy, prefetch, multimem bulk),
  §9.7.9.27 (tensormap.replace), §9.7.14.16 (mbarrier),
  §9.7.14.17 (tensormap.cp_fenceproxy), §9.7.14.18-19 (clusterlaunchcontrol),
  §9.7.17.5 (CTA pair), §9.7.17.12.1 (tcgen05.commit), §9.7.20.5 (setmaxnreg)
- PTX ISA 9.2 — https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html
  (version comparison only)
- CUDA Programming Guide 13.3 —
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html ,
  .../04-special-topics/async-copies.html ,
  .../04-special-topics/async-barriers.html ,
  .../03-advanced/advanced-kernel-programming.html ,
  .../02-basics/writing-cuda-kernels.html ,
  .../01-introduction/programming-model.html ,
  .../05-appendices/compute-capabilities.html ,
  .../05-appendices/cpp-language-extensions.html
- CUTLASS docs —
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html ,
  https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- CUTLASS source (main) —
  https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp ,
  .../gemm/kernel/sm100_tile_scheduler.hpp ,
  .../gemm/collective/sm100_mma_warpspecialized.hpp ,
  .../pipeline/sm100_pipeline.hpp ,
  .../arch/reg_reconfig.h ,
  .../gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp
- Colfax Research, "CUTLASS Tutorial: GEMM with Thread Block Clusters on NVIDIA Blackwell GPUs" —
  https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/
  (source for the ctaMask examples and the bit-24 peer-address trick; both labelled `[reported]`)
- Jarmusch & Chandrasekaran, "Microbenchmarking NVIDIA's Blackwell Architecture: An
  in-depth Architectural Analysis", arXiv:2512.02189 — https://arxiv.org/pdf/2512.02189
  (TMEM = 256 KB/SM, 512 cols × 128 lanes; `tcgen05.mma` single-instruction latency
  11.0–12.6 cycles vs `wgmma` 32–128; all labelled `[reported]`. The paper contains no TMA,
  mbarrier, cluster, or CLC measurements.)

**Measured on this node** (8×B200, driver 595.71.05, CUDA 13.3 toolkit, `-arch=sm_100a`).
Benchmark sources live in this session's scratchpad
(`/tmp/claude-1000/-home-aman-code/930438ff-5f3c-49e6-a3d9-2663231246c6/scratchpad/`):
`clu.cu` (cluster size limits), `tmcheck.cu` (descriptor constraint probe),
`tma_lat.cu` / `tma2.cu` / `tma3.cu` / `tma4.cu` (TMA latency, stage scaling, aggregate
bandwidth, cp.async comparison, mbarrier and `__syncthreads` round trips). Device
attributes via `cuDeviceGetAttribute` through ctypes. Re-run before trusting any number
here on a differently-clocked box; `nvidia-smi -lgc` was **not** used, so clocks were free
to boost and all cycle counts carry that caveat.
