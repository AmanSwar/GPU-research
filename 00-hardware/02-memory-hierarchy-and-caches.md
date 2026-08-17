# Memory hierarchy on B200: HBM3e, L2, SMEM, TMEM, and how to control every level

## What this is

Every level of the B200 (SM100) memory hierarchy, with numbers I measured on **this node**
(8x B200, driver 595.71.05, CUDA 13.3 toolkit at `/home/aman/code/cuda-13.3`) rather than
recalled, plus the exact PTX/CUDA mechanism to control each level. Microbenchmark sources are
in `/tmp/claude-1000/-home-aman-code/930438ff-5f3c-49e6-a3d9-2663231246c6/scratchpad/`
(`mem2.cu`, `mem4.cu`, `mem5.cu`, `carve.cu`, `l2part.cu`, `l2share.cu`, `tmem_bench.cu`).
Claims are labelled `[verified]` (I read it in a primary source or measured it here),
`[reported]` (a vendor/third party asserts it), `[inferred]`, `[unverified]`.
Where a widely-repeated number turned out to be wrong or unsourceable, I say so.

## Bottom line for our system

- **Effective L2 for data shared by all 148 SMs is ~63 MiB, not 126.5 MiB.** Two independent
  measurements (single-SM pointer chase knee at 48-63 MiB; all-SM shared-read bandwidth knee
  between 56 and 80 MiB) agree. A private-slice access pattern (each block owns its lines)
  gets the full ~126 MiB. `[verified, measured]` This is the single most important L2 fact for
  GLM-5.2: the 8 hot experts per layer per rank are **20.25 MiB in NVFP4**, which fits
  comfortably in 63 MiB — but 64-way concurrent DSA KV reads (72 MiB per layer) do not.
- **HBM3e sustained read = 6.98-7.28 TB/s** measured, against a **7.67 TB/s** pin-rate peak
  (3996 MHz x 2 x 7680 bits). That is 91-95% of peak. `[verified, measured]` Not 8 TB/s.
- **L2 hit = 303 cycles / 154 ns; HBM = 762 cycles / 388 ns; L1 hit = 39 cycles / 20 ns;
  SMEM = 29 cycles / 15 ns.** At concurrency 1 our per-forward-pass weight stream is only
  ~2.5 GB/rank (~325 us at 7.67 TB/s) out of an ~11 ms pass, so **decode at C1 is
  latency-bound, not HBM-bandwidth-bound.** The lever is the 154 ns vs 388 ns gap and
  keeping the L1 fat, not raw bandwidth. `[verified, measured] + [inferred]`
- **`.L2::evict_first` / `.L2::evict_last` on `ld`/`st` require a 256-bit vector type
  (`.v8.b32` or `.v4.b64`) on sm_100.** Every narrower form is rejected by ptxas 13.3. If you
  want per-instruction L2 policy on a `.v4.f32` load you must go through
  `createpolicy` + `.L2::cache_hint`. `[verified, ptxas 13.3.73]` This is not in any blog post
  I found and it silently blocks the obvious way to protect weights from KV traffic.
- **`cudaAccessPolicyWindow` bought +3% at best and *hurt* by 5% when the carve-out was
  oversized to 79 MiB.** The B200 default L2 replacement already keeps a 48 MiB hot set
  resident against a 1 GiB stream. Prefer the per-instruction PTX hints over the stream
  attribute. `[verified, measured]`
- **TMEM alloc+dealloc costs 442 cycles (225 ns) per round trip regardless of column count.**
  A kernel containing `tcgen05.alloc` is reported at **1 CTA/SM** by
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (vs 16 for the same shape without it).
  Allocate once per persistent kernel, never per tile. `[verified, measured]`
- **L1 holds >= 224 KiB of global data when you ask for no shared memory** (39-cycle hits at a
  224 KiB working set). Every KiB of dynamic SMEM you request comes straight out of L1.
  For small-M decode GEMMs that never tile, an SMEM-free kernel gets a 224 KiB L1 for free.
  `[verified, measured]`
- **Remote (DSMEM) shared-memory load = 204 cycles vs 29 local; `cluster.sync()` over 8 CTAs =
  362 cycles vs 20 for `__syncthreads()`.** DSMEM is ~7x local SMEM but still 1.5x *faster*
  than an L2 hit. `[verified, measured]`

---

## 1. The levels, at a glance

All latencies are single-thread dependent-load (pointer chase over a random permutation of
128 B lines), so they are pure loaded-nothing-else latency. All ns figures use the
**measured-under-load SM clock of 1.965 GHz** (confirmed with `nvidia-smi
--query-gpu=clocks.sm` during a saturating kernel; the reported max is also 1965 MHz).

| Level | Capacity | Latency (cycles) | Latency (ns @1.965 GHz) | Measured read BW | How measured |
|---|---|---|---|---|---|
| Register file | 64 K x 32-bit per SM (256 KiB), max 255 regs/thread | ~0 (operand collector) | - | - | `[reported]` Blackwell Tuning Guide |
| TMEM | 512 cols x 128 lanes x 32 bit = 256 KiB per SM | `tcgen05.ld.32x32b.x1` + `wait::ld` = 13.5 cy | 6.9 ns | ~585 B/cycle/SM (4 warps, `.x32`) | measured, `tmem_bench.cu` |
| SMEM (unified L1/SMEM) | 228 KiB/SM usable, 227 KiB max per CTA | 28.7 | 14.6 | ~66 B/cy/SM measured; 128 B/cy/SM theoretical | measured, `mem2.cu`/`mem4.cu` |
| DSMEM (remote CTA in cluster) | up to 8 x 228 KiB (portable) | 204 | 103.8 | not measured | measured, `mem4.cu` |
| L1 data (global) | >= 224 KiB with 0 SMEM requested | 39.4 | 20.1 | ~27 TB/s aggregate | measured, `carve.cu`, `mem3.cu` |
| L2 (own partition) | 126.5 MiB total; ~63 MiB effective for shared data | 303 | 154.2 | ~24-25 TB/s (private slices) | measured, `mem2.cu`, `mem3.cu` |
| L2 (capacity-pressured, 96 MiB WS from 1 SM) | - | 610 | 310.5 | - | measured, `mem2.cu` |
| HBM3e | 183359 MiB reported / 178.34 GiB `totalGlobalMem` | 762 | 387.8 | 6.98-7.28 TB/s | measured, `mem2.cu`, `mem3.cu` |

Cross-checks: chipsandcheese measured **19.6 ns / 39 cycles** for B200 L1 and **~150 ns** for
L2 `[reported]` — my 39.4 cycles and 154 ns match to within noise, which is a good sanity
check on the whole methodology.

### Device properties, straight from `cudaGetDeviceProperties` on this box

```
name=NVIDIA B200                       cc=10.0
totalGlobalMem      = 191495471104 B (178.34 GiB)     [nvidia-smi reports 183359 MiB]
multiProcessorCount = 148
clockRate           = 1965000 kHz      memoryClockRate = 3996000 kHz
memoryBusWidth      = 7680 bits
l2CacheSize                = 132644864 B  (126.50 MiB)
persistingL2CacheMaxSize   =  82903040 B  ( 79.06 MiB)   <- exactly 62.5% of L2
accessPolicyMaxWindowSize  = 134217728 B  (128.00 MiB)
sharedMemPerBlock          =     49152    sharedMemPerBlockOptin = 232448 (227 KiB)
sharedMemPerMultiprocessor =    233472 (228 KiB)   reservedSharedMemPerBlock = 1024
regsPerBlock = 65536   regsPerMultiprocessor = 65536   warpSize = 32
maxThreadsPerMultiProcessor = 2048   maxBlocksPerMultiProcessor = 32
clusterLaunch = 1   globalL1CacheSupported = 1   localL1CacheSupported = 1
driverVer = 13020   runtimeVer = 13030
```
`[verified]` — `devq.cu`, run on device 0.

---

## 2. HBM3e

### Peak vs achievable

`memoryClockRate = 3996000 kHz` and `memoryBusWidth = 7680 bits`. HBM3e is DDR, so:

```
peak = 3.996e9 * 2 * 7680 / 8 bytes/s = 7.672e12 B/s = 7.67 TB/s
```
`[verified, computed from measured device attributes]`

The frequently repeated "8 TB/s per B200" is a rounded marketing figure; the bus this card
actually exposes is 7680 bits (not the 8192 bits a fully-enabled 8-stack HBM3e config would
give), which is consistent with the HGX B200 aggregate figure of ~62 TB/s across 8 GPUs.
I could not fetch the NVIDIA datasheet to confirm the 62 TB/s number, so treat that as
`[unverified]`; the 7.67 TB/s pin rate is `[verified]` because it comes from this card.

Measured sustained read (1184 blocks x 512 threads, `ld.global.nc.v8.b32` 256-bit loads):

| Buffer | plain GB/s | `.L2::evict_first` GB/s | `.L2::evict_last` GB/s |
|---|---|---|---|
| 256 MiB | 9858 | 7277 | 9655 |
| 512 MiB | 7269 | 7379 | 7417 |
| 1024 MiB | 7136 | 6577 | 7029 |
| 2048 MiB | 6947 | 6622 | 6980 |
| 8192 MiB | 6977 | 6536 | 6916 |

`[verified, measured]` — `mem3.cu`. Sustained HBM read tops out at **~6.98 TB/s = 91% of
7.67 TB/s peak**; the best number I saw anywhere was **7.46 TB/s implied** in the hot/cold test
(section 4), i.e. **97% of peak** when part of the traffic is L2-served.

Two things to note:

1. **The 256 MiB row (9.86 TB/s) exceeds HBM peak.** That is L2 doing work: at 256 MiB with
   148x8 blocks each block's private slice is ~221 KiB, so a large fraction stays L1/L2
   resident across the `reps` loop. Any "HBM bandwidth" measurement above ~7.7 TB/s is
   measuring cache, not DRAM.
2. **`.L2::evict_first` on a pure stream is not free.** It costs 4-7% at 1-8 GiB. It only pays
   when there is something else you are protecting (section 4).

### Line and sector granularity

The CUDA cache line is 128 B and L2 is sectored at 32 B `[reported]`, consistent with CCCL's
own code, which walks a region 128 B at a time when applying an access property:

```cpp
// cuda/__annotated_ptr/apply_access_property.h
constexpr size_t __line_size = 128;
for (size_t __i = 0; __i < __nbytes; __i += __line_size) {
  asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(__p + __i) :);
}
```
`[verified]` — `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__annotated_ptr/apply_access_property.h:50`

There is also a **256 B sector-promotion bit** in the L2 descriptor (`__l2_sector_promote_256B`,
bit 62 of the interleaved/block descriptor), which CCCL declares but never sets. `[verified]`
(see section 4.3 for the full bit layout). I did not find a PTX-level way to set it.

My stride sweep in `mem5.cu` produced numbers that exceeded HBM peak and were therefore
dominated by kernel-launch overhead — **not sourced**, do not use them. Measuring sector
granularity properly needs a much longer-running kernel.

---

## 3. L2: 126.5 MiB, but you probably get 63

### Size and partitioning

`l2CacheSize = 132644864 B = 126.50 MiB`. `[verified]` The Blackwell Tuning Guide says
"The GB200 GPU increases L2 cache to 126 MB" `[reported]`.

chipsandcheese reports **two L2 partitions, one per die, ~63 MB each, ~150 ns to the local
partition and a modest cross-partition penalty**, and 21 TB/s local / 16.8 TB/s cross
partition bandwidth `[reported]`. The arXiv microbenchmarking paper (2512.02189) claims
**four** L2 partitions on B200 `[reported]` — these disagree, and that paper has a public
critique (girl.surgery/bad_paper) documenting several outright errors in it (an impossible
tcgen05 FLOP rate derived from its 11.4-cycle MMA latency; a "32x32 optimal wgmma tile"
that is not a legal shape since M >= 64; the claim that TMEM can load from global memory).
**I would not cite 2512.02189 for anything without re-measuring it.** `[verified — I read both]`

Here is what I could establish directly.

#### Measurement A: single-SM pointer chase, capacity knee at ~48-63 MiB

One thread, one SM, random permutation of 128 B lines over a buffer of size S:

| Working set | `ld.global.ca` cycles | ns | `ld.global.cg` cycles |
|---|---|---|---|
| 0.192 MiB | 39.4 | 20.1 | 302 |
| 0.5 - 40 MiB | 303 | 154 | 303 |
| 48 MiB | 311 | 158 | 302 |
| 56 MiB | 389 | 198 | 320 |
| 60 MiB | 393 | 200 | 377 |
| 63 MiB | 471 | 240 | 452 |
| 64 MiB | 461 | 235 | 455 |
| 72 MiB | 604 | 307 | 603 |
| 96 MiB | 610 | 311 | 612 |
| 126 MiB | 699 | 355 | 687 |
| 144 MiB | 763 | 388 | 764 |
| 160 MiB - 1 GiB | 759-764 | 386-389 | 759-770 |

`[verified, measured]` — `mem2.cu`. The knee is at **~48-56 MiB**, and the curve has fully
reached HBM latency by 144 MiB. If a single SM could use all 126.5 MiB, the knee would be at
~126 MiB. It is at half that.

#### Measurement B: all-SM shared read, bandwidth knee between 56 and 80 MiB

148 blocks x 512 threads, **every block reads the whole buffer** (so SMs on both dies request
the same lines):

| Buffer | all-blocks-read-all GB/s |
|---|---|
| 16-56 MiB | 5305-5403 |
| 64 MiB | 4239 |
| 80 MiB | 2604 |
| 96-512 MiB | 2573-2606 |

`[verified, measured]` — `l2share.cu`. Same knee, ~63 MiB.

#### Measurement C: private-slice read reaches the full 126 MiB

The same grid but each block owning a private grid-strided slice (so the union covers the
buffer once and each die's SMs touch roughly their own half):

| Buffer | plain GB/s | `.L2::evict_first` | `.L2::evict_last` |
|---|---|---|---|
| 16 MiB | 27117 | 26787 | 27164 |
| 48 MiB | 25405 | 25427 | 25268 |
| 96 MiB | 24637 | 22889 | 24377 |
| 112 MiB | 24633 | 22553 | 24416 |
| 126 MiB | 22473 | 21035 | 22854 |
| 144 MiB | 18672 | 20851 | 19475 |
| 160 MiB | 17708 | 19057 | 16471 |
| 192 MiB | 15557 | 15698 | 15566 |
| 256 MiB | 9858 | 7277 | 9655 |

`[verified, measured]` — `mem3.cu`. Here the knee is at **126-144 MiB**, i.e. the full L2, and
peak L2 read bandwidth is **~25 TB/s** (which brackets chipsandcheese's 21 TB/s from above).

**Model that fits all three:** each die's L2 partition (63.25 MiB) services its own SMs, and a
line requested by SMs on both dies occupies capacity in **both** partitions. Data that is
private to a die gets the full 126.5 MiB; data shared by the whole GPU gets ~63 MiB.
`[inferred, from three consistent measurements]`

#### What I could *not* confirm: a bimodal near/far latency

I ran a per-SM chase over eight distinct 8 KiB buffers from all 148 SMs and got a spread of
**268 - 304 cycles (136 - 155 ns), a 13% range with no bimodality.** `[verified, measured]` —
`l2part.cu`. So on this part I cannot reproduce a clean "local vs remote L2 partition" latency
split for L2 *hits*. The cross-partition penalty chipsandcheese describes may only show up for
misses/atomics, or my 8 KiB buffers may interleave across both partitions. Treat
"cross-partition L2 hit is much slower" as `[unverified]`.

### Practical L2 budget for GLM-5.2 (per rank, TP8)

From `/home/aman/code/weights/GLM-5.2-NVFP4/config.json` and its safetensors index
`[verified, read on this box]`:

```
hidden_size 6144   moe_intermediate_size 2048   n_routed_experts 256   num_experts_per_tok 8
n_shared_experts 1  num_hidden_layers 78  first_k_dense_replace 3  -> 75 MoE layers
kv_lora_rank 512   qk_rope_head_dim 64  -> 576 values per token per layer of MLA KV
index_topk 2048    index_topk_freq 4
NVFP4 checkpoint total_size = 464,795,267,072 B (433 GiB)
```

Derived (NVFP4 = 4 bits/weight + 1 byte E4M3 block scale per 16 values = 4.5 bits effective):

| Quantity | Value | Fits in 63 MiB? |
|---|---|---|
| One expert (gate+up+down), all ranks | 3 x 2048 x 6144 = 37.75 M params -> 21.2 MB | - |
| One expert, per rank (TP8) | 2.65 MB / 2.53 MiB | yes |
| **8 active experts, one layer, per rank** | **21.2 MB / 20.25 MiB** | **yes, comfortably** |
| All 256 experts, one layer, per rank | 679 MB / 648 MiB | no (10x over) |
| MLA KV for one query, one layer (DSA top-2048, FP8) | 2048 x 576 B = 1.18 MB / 1.125 MiB | yes |
| Same at concurrency 64 | 75.5 MB / 72 MiB | **no - alone it exceeds 63 MiB** |
| 8 experts + C64 DSA KV, one layer | 96.7 MB / 92 MiB | **no** |

`[verified arithmetic from config.json]` `[inferred]` for the fit conclusions.

The actionable read: **at C1 everything you need for a layer fits in L2 with room to spare.
At C64 the DSA KV working set alone (72 MiB) blows past the ~63 MiB shared-data budget and
will evict expert weights unless you tag it.** That is exactly the case section 4.4 addresses.

---

## 4. Controlling L2

There are four distinct mechanisms and they compose badly if you do not know which is which.

### 4.1 The legality matrix (this is the part that bites)

I probed ptxas 13.3.73 (`/home/aman/code/cuda-13.3/nvidia/cu13/bin/ptxas`) directly:

| PTX form | sm_100 | Notes |
|---|---|---|
| `ld.global.nc.L1::evict_last.v4.f32` | **OK** | L1 eviction priority: any width |
| `ld.global.nc.L1::no_allocate.v4.f32` | **OK** | streaming load, does not pollute L1 |
| `ld.global.nc.L1::evict_first.L2::256B.v4.f32` | **OK** | prefetch-size hint composes with L1 policy |
| `ld.global.nc.L2::128B / L2::256B` | **OK** | prefetch size |
| `ld.global.nc.L1::evict_first.L2::cache_hint.v4.f32 ..., %policy` | **OK** | descriptor path, any width |
| `ld.global.nc.L2::evict_first.v4.f32` | **FAIL** | `Instruction 'ld' requires '.v8.b32/.v4.b64' type with '.L2::evict_first' modifier` |
| `ld.global.nc.L2::evict_first.f32` (scalar) | **FAIL** | same error |
| `ld.global.nc.L2::evict_last.v4.f32` | **FAIL** | same error |
| `ld.global.nc.L2::evict_first.v8.b32` | **OK** | 256-bit only |
| `ld.global.nc.L2::evict_last.v4.b64` | **OK** | 256-bit only |
| `ld.global.nc.L2::evict_normal.v8.b32` | **OK** | |
| `ld.global.nc.L2::evict_unchanged.v8.b32` | **FAIL** | `Illegal modifier '.L2::evict_unchanged' for instruction 'ld'` |
| `ld.global.nc.L2::no_allocate.v8.b32` | **FAIL** | `Illegal modifier` |
| `ld.global.nc.L1::no_allocate.L2::evict_first.v8.b32` | **OK** | composes |
| `st.global.L2::evict_first.v8.b32` | **OK** | |
| `st.global.L2::evict_first.v4.f32` | **FAIL** | same 256-bit restriction |
| `st.global.L1::no_allocate.v4.f32` | **OK** | |
| `discard.global.L2 [p], 128;` | **OK** | |
| `applypriority.global.L2::evict_normal [p], 128;` | **OK** | |
| `applypriority.global.L2::evict_last [p], 128;` | **FAIL** | `Unsupported modifier '.evict_last' for instruction 'applypriority'` |
| `prefetch.global.L2::evict_last [p];` | **OK** | **this** is how you promote a line |
| `prefetchu.L1 [p];` | **OK** | |
| `createpolicy.fractional.L2::evict_last.b64 %p, 0.75;` | **OK** | |
| `createpolicy.range.global.L2::evict_last.L2::evict_first.b64 %p,[ptr],pb,tb;` | **OK** | |
| `ld.global.nc.v8.b32` (256-bit load itself) | **OK on sm_100+ only** | `Feature '256 bit wide load/store' requires .target sm_100 or higher` on sm_80/sm_90 |

`[verified, all probed with ptxas 13.3.73]` — `ldtest.cu`, `ldtest2.cu`, `ldtest3.cu`.

Two consequences that matter:

- **Direct `.L2::evict_*` on `ld`/`st` is a Blackwell-only feature and only on 256-bit
  accesses.** It is not that Hopper had it and Blackwell removed it — `.v8.b32` does not exist
  before sm_100, so this instruction form is new. If your loads are `float4`/`uint4` you have
  to widen them to 32 B or use the descriptor path.
- **`applypriority` can only *demote* to `evict_normal`.** To *promote* a line to persistent
  you use `prefetch.global.L2::evict_last`. This is what CCCL's
  `cuda::apply_access_property(ptr, shape, access_property::persisting{})` does.
  `[verified]` — `apply_access_property.h`.

### 4.2 SASS mapping

Disassembled with `cuobjdump -sass` (from the local Triton wheel, matching driver 595):

| PTX | SASS on sm_100 |
|---|---|
| `ld.global.nc.v8.b32` | `LDG.E.ENL2.256.CONSTANT ..., 0x3` |
| `ld.global.nc.L2::evict_first.v8.b32` | `LDG.E.EFL2.256.CONSTANT ..., 0x5` |
| `ld.global.nc.L2::evict_last.v8.b32` | `LDG.E.ELL2.256.CONSTANT ..., 0x9` |
| `ld.global.nc.L1::no_allocate.v4.b32` | `LDG.E.NA.CONSTANT` |
| `discard.global.L2 [p],128` | `CCTL.E.DML2 [R+off]` |
| `applypriority.global.L2::evict_normal [p],128` | `CCTL.E.RML2 [R+off]` |

`[verified, my own disassembly]` — `ld256.cu` / `ld256.cubin`. The `.256` suffix confirms
these are genuine 32 B-per-thread loads (a warp moves 1 KiB per instruction).

### 4.3 The 64-bit L2 cache-policy descriptor

`createpolicy` produces a 64-bit descriptor consumed by `.L2::cache_hint`. CCCL documents the
exact bit layout — this is the most precise description of the mechanism I found anywhere:

```c
// cuda/__annotated_ptr/access_property_encoding.h   (CUDA 13.3 CCCL)
struct __block_desc_t {              // "range" form, descriptor mode 3
  uint64_t __reserved1       : 37;
  uint32_t __block_count     :  7;   // bits 37..43, clamped to [1,127]
  uint32_t __block_start     :  7;   // bits 44..50, ptr >> log2(block_size)
  uint32_t __reserved2       :  1;
  uint32_t __block_size_enum :  4;   // bits 52..55, log2(total)-19, min block 4 KiB
  uint32_t __l2_cop_off      :  1;   // bit 56   secondary policy
  uint32_t __l2_cop_on       :  2;   // bits 57..58 primary policy
  uint32_t __l2_descriptor_mode : 2; // bits 59..60  0=implicit 2=interleaved 3=block
  uint32_t __l1_inv_dont_allocate : 1;   // bit 61
  uint32_t __l2_sector_promote_256B : 1; // bit 62
};
struct __interleaved_desc_t {        // "fractional" form, descriptor mode 2
  uint64_t : 52;  uint32_t __fraction : 4;   // fraction = num/16
  /* same policy/mode/L1/sector bits at 56..62 */
};
enum class __l2_evict_t : uint32_t {
  _L2_Evict_Unchanged = 0, _L2_Evict_First = 1, _L2_Evict_Last = 2, _L2_Evict_Normal_Demote = 3
};
```
`[verified]` — `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__annotated_ptr/access_property_encoding.h`
and `createpolicy.h`.

Notes worth having: the secondary policy is restricted to `evict_first` or `evict_unchanged`
(asserted in the header); the fractional form quantises to 16ths; the minimum block size is
4 KiB; and CCCL carries a comment recording **a hardware/PTX bug in `createpolicy` when
`__block_size_enum == 13`** (i.e. a 32 MiB block size), where the block count is not clamped
correctly. `[verified]` — that is a real trap if you build a range descriptor over a
~4 GB region.

Practical PTX, straight through:

```ptx
// Promote a 48 MiB weight window to evict_last for 75% of its lines, demote the rest.
createpolicy.fractional.L2::evict_last.L2::evict_first.b64 %policy, 0.75;
ld.global.nc.L2::cache_hint.v4.f32 {%f0,%f1,%f2,%f3}, [%weights], %policy;

// Or, if you can afford 32 B loads, skip the descriptor entirely:
ld.global.nc.L2::evict_last.v8.b32 {%r0,...,%r7}, [%weights];   // weights: keep
ld.global.nc.L2::evict_first.v8.b32 {%r8,...,%r15}, [%kv];      // KV: dump on first pressure
```

The C++ equivalents (all in `cuda/annotated_ptr`, CUDA 13.3):

```cpp
#include <cuda/annotated_ptr>
cuda::annotated_ptr<const float, cuda::access_property::persisting> W{w_ptr};
cuda::annotated_ptr<const float, cuda::access_property::streaming>  K{kv_ptr};
// or a mixed descriptor built once on the host and reused:
cuda::access_property ap{w_ptr, /*primary_bytes*/48u<<20, /*total*/48u<<20,
                         cuda::access_property::persisting{},
                         cuda::access_property::streaming{}};
auto* Wp = cuda::associate_access_property(w_ptr, ap);   // -> __nv_associate_access_property
cuda::apply_access_property(w_ptr, 48u<<20, cuda::access_property::persisting{}); // prefetch loop
```
`[verified]` — `access_property.h`, `associate_access_property.h`, `apply_access_property.h`.

### 4.4 L2 set-aside (`cudaAccessPolicyWindow`) — measured, and mostly not worth it

The runtime API `[verified, CUDA Programming Guide 13.3, "L2 Cache Control"]`:

```cpp
size_t carve = std::min<size_t>(prop.l2CacheSize * 0.75, prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve);   // <= 79.06 MiB on B200

cudaStreamAttrValue av{};
av.accessPolicyWindow.base_ptr  = W;                 // must be global memory
av.accessPolicyWindow.num_bytes = ws;                // <= accessPolicyMaxWindowSize = 128 MiB
av.accessPolicyWindow.hitRatio  = 1.0f;              // fraction of lines given hitProp
av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // only Normal or Streaming legal
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &av);
...
cudaCtxResetPersistingL2Cache();                     // or set hitProp=Normal on the same window
```

Documented semantics I confirmed by reading the guide: `hitRatio` is the *fraction of window
lines* that get `hitProp` (the hardware picks them pseudo-randomly), and with concurrent
kernels "the net utilization of this set-aside cache portion is the sum of all the concurrent
kernels' individual use" — i.e. **two streams each claiming 48 MiB will thrash**. `[verified]`

Now the measurement. Kernel: `W` = 48 MiB re-read 8x per iteration, `KV` = 1024 MiB streamed
once per iteration, 4 iterations, 1184 blocks x 512 threads. "min-HBM" assumes W stays fully
resident, so `implied HBM` is the DRAM rate you would need if the policy worked perfectly.

| Configuration | time | logical GB/s | implied HBM GB/s |
|---|---|---|---|
| plain / plain | 0.622 ms | 9501 | 7233 |
| W `.L2::evict_last` / KV `.L2::evict_first` | **0.603 ms** | **9795** | **7457** |
| W `.L2::evict_last` / KV plain | 0.629 ms | 9386 | 7146 |
| W plain / KV `.L2::evict_first` | **0.603 ms** | **9800** | **7462** |
| plain + `accessPolicyWindow(W, persist/stream)`, 48 MiB carve-out | 0.606 ms | 9748 | 7422 |
| plain + `accessPolicyWindow(W, persist/stream)`, 79 MiB carve-out | 0.652 ms | 9059 | 6897 |

`[verified, measured]` — `mem4.cu`.

Read this carefully:

- **Logical bandwidth 9.5-9.8 TB/s exceeds HBM peak (7.67).** W really is being served from L2
  in every configuration, including the baseline. The B200's default replacement policy
  already protects a 48 MiB re-read set against a 1 GiB stream.
- The best result (0.603 ms) is **97% of the theoretical floor** (4.19 GiB of unavoidable HBM
  traffic at 7.67 TB/s = 0.587 ms). There is essentially nothing left on the table.
- The winning knob is **marking the streaming side `evict_first`**, not marking the hot side
  `evict_last`. `W evict_last / KV plain` was the *worst* configuration (0.629 ms) — promoting
  the hot set without demoting the stream just makes L2 replacement work harder.
- **Oversizing the carve-out actively hurts: 79 MiB cost 5% versus baseline.** The set-aside
  is stolen from the normal L2 that the KV stream needs.

Recommendation for our engine: **use `.L2::evict_first` (or a `createpolicy` streaming
descriptor) on KV-cache reads and leave weights alone.** Only reach for
`cudaLimitPersistingL2CacheSize` if you can measure a win, and never set it above ~48 MiB
while other streams are live.

### 4.5 The other L2 verbs

```ptx
prefetch.global.L2      [p];                 // pull into L2
prefetch.global.L2::evict_last [p];          // pull in AND mark persistent  (the "promote" verb)
prefetchu.L1            [p];                 // pull into L1 (uniform datapath)
applypriority.global.L2::evict_normal [p], 128;   // demote back to normal; only evict_normal legal
discard.global.L2 [p], 128;                  // invalidate 128 B without writeback -> undefined data
```
`[verified, all assemble on sm_100 with ptxas 13.3.73]`

`discard.global.L2` is the one people forget. For a scratch buffer you are about to fully
overwrite (e.g. an activation staging buffer between MoE stages), discarding avoids the
read-for-ownership traffic. The size operand must be 128 (the line size). Data becomes
undefined, so only use it on memory you will write completely.

---

## 5. L1 / shared memory: one 256 KiB array, and the split is yours to choose

### Sizes

- `sharedMemPerMultiprocessor = 233472 B = 228 KiB` `[verified]`
- `sharedMemPerBlockOptin = 232448 B = 227 KiB` (228 KiB minus the 1 KiB
  `reservedSharedMemPerBlock`) `[verified]`
- Static `__shared__` is still capped at 48 KiB; above that you must
  `cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, N)` and pass N as the
  third launch parameter `[verified, Blackwell Tuning Guide]`
- Legal carve-outs: **0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KiB per SM**
  `[reported, Blackwell Tuning Guide]`
- Max 32 CTAs/SM, 64 warps/SM (2048 threads) for cc 10.0 `[verified via deviceProp + Tuning Guide]`

### L1 capacity for *global* data, measured

Single-thread `ld.global.ca` chase, varying the working set (columns) against the dynamic SMEM
request and `cudaFuncAttributePreferredSharedMemoryCarveout` (rows). Value = cycles per
dependent load; **39 = L1 hit, ~305 = L2**.

```
carveout% | requested smem |   16K   32K   48K   64K   96K  128K  160K  192K  224K
       0% |         0 KiB  |    39    39    39    39    39    39    39    39    39
       0% |         8 KiB  |    39    39    39    39    39    39    39    39    39
       0% |        64 KiB  |    39    39    39    39    39    39   102   223   272
       0% |       128 KiB  |    39    39    39    39    39   108   230   288   295
       0% |       227 KiB  |    39   227   305   303   304   304   304   304   304
      50% |         8 KiB  |    39    39    39    39    39   108   230   288   295
      75% |         8 KiB  |    39    39    39   132   267   302   304   304   304
     100% |     any        |    42   228   305   303   304   304   304   304   304
```
`[verified, measured]` — `carve.cu`.

Reading: with **carve-out preference 0 (prefer L1) and no dynamic SMEM, the L1 caches at least
224 KiB of global data at 39-cycle latency.** Every KiB of SMEM you request is subtracted
(rounded up to the carve-out granularity). Pinning `PreferredSharedMemoryCarveout = 100`
leaves only 16-32 KiB of L1.

**Why this matters for GLM-5.2 decode:** at concurrency 1 every GEMM is M=1..8 (or M=~4 with
EAGLE 3-1-4 speculation). Those kernels do not need big SMEM tiles. A decode GEMM that asks
for zero dynamic shared memory gets a **224 KiB L1 per SM x 148 SMs = 32 MiB of aggregate L1**,
which is enough to hold an entire NVFP4 expert (2.53 MiB/rank) many times over and cuts its
re-read latency from 154 ns to 20 ns. Conversely, a prefill-shaped kernel that grabs 227 KiB
of SMEM leaves nothing. If the same kernel binary serves both regimes, consider two
specialisations. `[inferred, from measured L1 behaviour + config.json sizes]`

### Bank conflicts: exactly N-way = N times slower

32 banks x 4 B. 1024 threads/CTA (32 resident warps), 16 shared loads per rep, cycles reported
by thread 0:

| Index stride (words) | cycles / 16 warp-loads | slowdown vs stride 1 |
|---|---|---|
| 1 (conflict-free) | 987.7 | 1.00x |
| 2 | 2185.1 | 2.21x |
| 4 | 4370.4 | 4.42x |
| 8 | 8741.2 | 8.85x |
| 16 | 17482.6 | 17.7x |
| 32 (all one bank) | 34964.8 | 35.4x |
| **33 (classic +1 pad)** | **986.9** | **1.00x** |

`[verified, measured]` — `mem4.cu`. The penalty is exactly linear in the conflict degree, and
padding the row stride to 33 words removes it completely. Conflict-free throughput here was
**~66 B/cycle/SM**, about half the 128 B/cycle/SM the 32x4 B bank array can theoretically
deliver — my loop pays for address arithmetic per load, so treat 66 as a floor and 128 as the
ceiling `[verified floor, inferred ceiling]`.

Single-warp (latency-bound) shared load latency is **28.7 cycles / 14.6 ns**, flat from 8 KiB
to 224 KiB of allocated SMEM `[verified, measured]` — `mem2.cu`.

For anything laid out as tensor-core operand tiles you should be using **TMA swizzle modes**
rather than manual padding: the 128 B swizzle is what `tcgen05.mma`'s shared-memory descriptor
expects (`(2ULL << 61)` in the descriptor's swizzle field) `[reported, gau-nernst]`. Padding
and swizzling are alternatives, not complements — a swizzled tile has no bank conflicts *and*
no wasted SMEM.

### Filling SMEM: TMA vs cp.async

Single CTA of 128 threads, `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`
vs `cp.async.cg.shared::cta.global [.], [.], 16`:

| Transfer | TMA cycles | TMA GB/s per CTA | cp.async cycles | cp.async GB/s per CTA |
|---|---|---|---|---|
| 16 KiB | 1391 | 23.1 | 612 | 52.6 |
| 32 KiB | 1087 | 59.2 | 747 | 86.2 |
| 64 KiB | 1361 | 94.6 | 1008 | 127.8 |
| 128 KiB | 1843 | 139.8 | 1531 | 168.2 |
| 224 KiB | 2601 | 173.3 | 2399 | 187.9 |

`[verified, measured]` — `mem5.cu`. Caveat: one CTA, so this is a **latency** measurement, not
a bandwidth one; both would scale with more CTAs.

The useful shape of it: TMA has a **fixed cost of roughly 1000-1400 cycles** for a single bulk
copy issued by one thread, and a marginal rate of ~130 B/cycle. `cp.async` with all 128 threads
issuing is *faster in wall-clock* at every size I measured — but it burns 128 threads' issue
slots and address registers, whereas TMA burns one `elect`ed thread and no address math. For a
warp-specialised producer/consumer decode kernel TMA is still the right call; for a small
one-shot staging copy `cp.async` is not obviously worse.

---

## 6. Distributed shared memory (DSMEM) and clusters

- Portable max cluster size 8; **B200 supports a non-portable cluster size of 16** with
  `cudaFuncAttributeNonPortableClusterSizeAllowed` `[reported, Blackwell Tuning Guide]`
- `clusterLaunch = 1` on this device `[verified]`

Measured, cluster of 8 CTAs x 128 threads, pointer chase into a neighbour CTA's SMEM via
`cg::cluster_group::map_shared_rank`:

| Operation | cycles | ns @1.965 GHz |
|---|---|---|
| local SMEM dependent load | 28.6 | 14.6 |
| **remote (DSMEM) dependent load** | **204.0** | **103.8** |
| `__syncthreads()` (128 threads) | 20.0 | 10.2 |
| `cluster.sync()` (8 CTAs) | 361.7 | 184.1 |

`[verified, measured]` — `mem4.cu`.

Two things follow:

1. **DSMEM (104 ns) is faster than an L2 hit (154 ns)** and 3.7x faster than HBM (388 ns). A
   cluster is a legitimate way to build a 8 x 228 KiB = 1.78 MiB software-managed cache that
   beats L2 on latency.
2. **`cluster.sync()` costs 18x a `__syncthreads()`.** At 184 ns each, a decode layer that
   cluster-syncs 10 times per layer x 78 layers spends 143 us per forward pass on cluster
   barriers alone. Given that our profile already shows **47% of the 19.6% collectives bucket
   is rank arrival skew**, adding intra-GPU cluster barriers to the critical path is exactly
   the wrong direction unless the cluster buys a real reduction in HBM traffic.
   `[inferred from measured barrier cost + the profile in the brief]`

For `tcgen05` 2-SM MMA (`cta_group::2`) a cluster is mandatory anyway — the pair of CTAs shares
one TMEM allocation and one MMA issue. That is a different use of clusters from DSMEM caching,
and its barrier cost is the `mbarrier`/`tcgen05.commit` path, not `cluster.sync()`.

---

## 7. TMEM: the new level

### What it is

A **separate 256 KiB per-SM memory that only `tcgen05.*` instructions can address.** It is
not part of the unified L1/SMEM 256 KiB — it is additive. `[reported, consistent across
Colfax, gau-nernst, thesoftwarefrontier; not found in the Blackwell Tuning Guide, which does
not mention TMEM at all]`

Geometry: **512 columns x 128 lanes x 32 bits**. `[reported, three independent sources agree]`
Address encoding: a 32-bit value where **bits 31..16 = lane, bits 15..0 = column**, so one lane
step is `1 << 16`. `[reported]` My measured base address from `tcgen05.alloc` on an otherwise
idle SM is `0x00000000` (lane 0, column 0) for every column count and every CTA
`[verified, measured]`.

Access is **warp-partitioned in hardware**: warp 0 sees lanes 0-31, warp 1 lanes 32-63, and so
on, so touching all 128 lanes requires a full warpgroup. `[reported]`

Why it exists: the largest single-CTA UMMA (`m128n256`) produces a 128x256 FP32 accumulator =
32768 values = 256 registers per thread across a warpgroup, above the 255-register ceiling.
The accumulator physically cannot live in the RF. `[reported, thesoftwarefrontier — and the
arithmetic checks out]`

### The instruction set, verbatim from the CUDA 13.3 headers

```
tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32 [dst], nCols;
tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32 taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned;
```
PTX ISA 8.6, targets SM_100a/100f, SM_103a/103f, SM_110a/110f.
`[verified]` — `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_alloc.h`

Load/store shapes that exist in CUDA 13.3 (from the generated headers, `[verified]`):

| `tcgen05.ld` shape | repeat factors `.x{N}` | extra modifier |
|---|---|---|
| `.32x32b` | 1,2,4,8,16,32,64,128 | `.pack::16b` |
| `.16x64b` | 1,2,4,8,16,32,64,128 | `.pack::16b` |
| `.16x128b` | 1,2,4,8,16,32,64 | `.pack::16b` |
| `.16x256b` | 1,2,4,8,16,32 | `.pack::16b` |
| `.16x32bx2` | 1,2,4,8,16,32,64,128 | `.pack::16b` |

`tcgen05.st` mirrors these exactly with `.unpack::16b` instead of `.pack::16b`.
`.pack::16b` packs two 16-bit values per 32-bit register on the way out (and `.unpack::16b`
splits on the way in) — that is how you get BF16/FP16 epilogues without a separate convert.

`tcgen05.cp` (SMEM -> TMEM, no registers touched) supports:
`.128x256b`, `.128x128b`, `.64x128b.warpx2::{01_23,02_13}`, `.32x128b.warpx4`, `.4x256b`,
each optionally with the sub-byte decompression forms `.b8x16.b4x16_p64` (4-bit source,
64 pad bits) and `.b8x16.b6x16_p32` (6-bit source). `[verified]` — `tcgen05_cp.h`.
**That `.b4x16_p64` form is the FP4 path**: it lets you stage NVFP4 operands into SMEM and have
the copy engine expand them into TMEM. Relevant to
`bmm_E2m1_E2m1E2m1_Fp32_swiGlu_dynB_sm100f` (6.0% of our profile).

Synchronisation: `tcgen05.wait::ld` / `tcgen05.wait::st`, `tcgen05.fence::before_thread_sync`
/ `::after_thread_sync`, and `tcgen05.commit.cta_group::{1,2}.mbarrier::arrive::one` for MMA
completion. `[verified]` — `tcgen05_wait.h`, `tcgen05_fence.h`, `tcgen05_commit.h`.

### Allocation rules

- `nCols` must be a **power of two, minimum 32, maximum 512** `[reported]`. I verified the
  minimum the hard way: `nCols = 16` produces `an illegal instruction was encountered` at
  runtime and kills the context. `[verified, measured]`
- `nCols = 512` (the whole 256 KiB) allocates successfully from a single CTA
  `[verified, measured]`. Note that thesoftwarefrontier states a "maximum single allocation of
  256 columns" — **that is wrong on this hardware**, 512 works.
- Allocation is **whole columns**: all 128 lanes of a column are allocated together.
  `[reported, consistent across sources]`
- The **same warp** must allocate and deallocate. `[reported]`
- You **must** deallocate before the CTA exits. `[reported]`

### Cost, measured

Single warp, `cta_group::1`, 1000 iterations of alloc-then-dealloc:

| `nCols` | TMEM bytes | alloc+dealloc (cycles) | ns @1.965 GHz | `relinquish_alloc_permit` (cycles) | base taddr |
|---|---|---|---|---|---|
| 32 | 16384 | 442.0 | 224.9 | 24 | 0x00000000 |
| 64 | 32768 | 442.0 | 224.9 | 24 | 0x00000000 |
| 128 | 65536 | 442.0 | 224.9 | 24 | 0x00000000 |
| 256 | 131072 | 442.0 | 224.9 | 24 | 0x00000000 |
| 512 | 262144 | 442.0 | 224.9 | 24 | 0x00000000 |

`[verified, measured]` — `tmem_bench.cu`. A first allocation in a fresh CTA (uncontended)
takes **272-296 cycles**; all 148 CTAs (1 per SM) allocate successfully at every size.

**442 cycles / 225 ns per alloc+dealloc pair, flat in size.** For context, that is 1.5 L2 round
trips, or roughly the time to issue 40 `tcgen05.ld.32x32b.x1`s. In a decode kernel doing
78 layers x several GEMMs, allocating TMEM per GEMM would cost tens of microseconds per
forward pass. **Allocate once per persistent kernel and index into it.**

### Why alloc is expensive: the SASS

```
DEPBAR.LE SB0, 0x36 ;
UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR8 ;   // uniform-datapath atomic bitmap search
BRA.U UP0, <success> ;
NANOSLEEP 0x64 ;                                 // 100 ns backoff on failure
DEPBAR.LE SB0, 0x36 ;
UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR4 ;   // retry
...
ATOMS.OR RZ, [UR6+0x14], R3 ;                    // CTA-local bookkeeping bitmap in SMEM
ATOMS.OR RZ, [UR6+0x18], R0 ;
STS [UR7], R2 ;                                  // write the taddr to the shared::cta slot
```
and

```
UVIRTCOUNT.DEALLOC.SMPOOL 0x80 ;   // tcgen05.relinquish_alloc_permit
...
LDS R3,[UR6+0x14] ; ISETP.NE ... @P0 BRA <trap>   // dealloc validity checks (use-after-free,
LDS R2,[UR6+0x18] ; ISETP.NE ... @P0 BRA <trap>   //  double-free, phase errors)
UTCATOMSWS.AND URZ, UR4 ;                          // tcgen05.dealloc
```
`[verified, my own `cuobjdump -sass` of a locally compiled sm_100a cubin]`. This confirms the
mechanism thesoftwarefrontier described `[reported]`: `tcgen05.alloc` is a spin-on-contention
atomic bitmap search with a 100 ns nanosleep backoff, and ptxas injects runtime traps that
catch freeing unallocated or already-freed columns.

### Occupancy consequence

```
cudaOccupancyMaxActiveBlocksPerMultiprocessor(plain 128-thread kernel, 128, 0) = 16
cudaOccupancyMaxActiveBlocksPerMultiprocessor(same shape + tcgen05.alloc, 128, 0) =  1
```
`[verified, measured]` — `tmem_bench.cu`. The occupancy model treats a TMEM-using kernel as
consuming the whole per-SM TMEM pool, so it reports **1 CTA/SM** regardless of how many columns
you actually ask for. In practice all 148 CTAs of a 1-per-SM grid did allocate simultaneously
(one per SM, so no contention), and the hardware bitmap will let multiple CTAs on the same SM
share TMEM if their column counts fit. But **do not rely on the occupancy API to plan a
multi-CTA-per-SM TMEM kernel** — it will tell you 1 and the launcher may believe it.

### Throughput of getting data out

`tcgen05.ld.sync.aligned.32x32b.x{N}.b32` followed by `tcgen05.wait::ld`, in a dependent loop,
4 warps resident:

| `.x{N}` | regs/thread | bytes/warp/instr | cycles/iter | ns | implied B/cycle/SM (4 warps) |
|---|---|---|---|---|---|
| x1 | 1 | 128 | 13.5 | 6.9 | 37.8 |
| x8 | 8 | 1024 | 16.3 | 8.3 | 251.4 |
| x32 | 32 | 4096 | 28.0 | 14.3 | 584.2 |

`[verified, measured]` — `tmem_bench.cu`.

Interpretation: the **fixed latency of `tcgen05.ld` + `wait::ld` is ~13.5 cycles**, and the
marginal cost is ~0.47 cycles per additional 32-bit register per thread (about 2.1 registers
per thread per cycle). At `.x32` this is **584 B/cycle/SM = 1.15 TB/s per SM = ~170 TB/s
aggregate**. The often-quoted "TMEM read bandwidth 16 TB/s" traces back to the arXiv paper via
thesoftwarefrontier `[reported]`; it is an order of magnitude below what I measure and I cannot
reconcile the two — given that paper's documented errors, **I would not use 16 TB/s.**
Treat my 584 B/cycle/SM as a measured lower bound on a dependent loop.

The practical constraint is not TMEM at all — it is **registers**. `.x32` costs 32 registers per
thread just to hold the result; `.x128` would cost 128+ (thesoftwarefrontier measures
~134 regs/thread including overhead, ~26% of the RF for a warpgroup `[reported]`). Plan the
epilogue register budget before the tile shape.

### TMEM in a GEMM, sketched

```cpp
__shared__ alignas(16) uint32_t tmem_slot;
// ONE warp allocates, once, for the life of the (persistent) kernel:
if (warp_id == 0) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(smem_u32(&tmem_slot)), "r"(256u) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__syncthreads();
uint32_t tmem = tmem_slot;                       // lane<<16 | col
...
// ONE thread issues the MMA; completion lands on an mbarrier:
// tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem], a_desc, b_desc, idesc, enable_input_d;
// tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [bar];
...
// epilogue: 4 warps, each reading its own 32 lanes
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
uint32_t taddr = tmem + ((warp_id) << 21);       // 32 lanes per warp, lane in bits 31..16
asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {...}, [%0];" :: "r"(taddr) : "memory");
asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
...
if (warp_id == 0)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem),"r"(256u):"memory");
```
`[verified — this is the structure that compiles and runs in `tmem_bench.cu`]`.
For a public end-to-end reference, gau-nernst's tcgen05 GEMM walk-through reaches 98% of cuBLAS
(1476 vs 1507 TFLOPS BF16) with exactly this skeleton plus warp specialisation, 2-SM MMA and a
persistent grid `[reported]`.

---

## 8. Worked example: 256-expert MoE where 8 are hot

Numbers are per rank (TP8), NVFP4, one MoE layer of GLM-5.2. `[verified arithmetic]`

```
per expert, all ranks : 3 x 2048 x 6144 = 37,748,736 params
                        NVFP4 payload   = 18,874,368 B
                        E4M3 block scales (1 per 16 vals) = 2,359,296 B
                        total           = 21,233,664 B  (20.25 MiB)
per expert, per rank  : 2,654,208 B  (2.53 MiB)
8 hot experts / rank  : 21,233,664 B (20.25 MiB)   <-- fits in the ~63 MiB shared-L2 budget
256 experts / rank    : 679,477,248 B (648 MiB)    <-- 10x over L2, must stream
```

### The decode-time question

At concurrency 1 with EAGLE 3-1-4, a forward pass touches 8 experts per MoE layer, and the
router picks a *different* 8 next token. Over 75 MoE layers that is
`75 x 21,233,664 = 1,592,524,800 B = 1.59 GB (1.48 GiB)` **of expert weights per forward pass
per rank**, plus roughly 0.9 GB of attention/dense/shared-expert weights (estimated from the
projection shapes in `config.json`, so `[inferred]` rather than verified). Total ~2.5 GB.
At 7.67 TB/s that is **~325 us**. Our measured 365 tok/s with ~4 tokens per accepted pass means
~11 ms per pass, so **weight streaming is under 3% of the pass.** `[inferred]`

Conclusion: **do not spend effort on L2 residency for expert weights at C1.** They are not the
bottleneck; the profile's 37.1% dense GEMM + 19.4% MoE GEMM is small-M compute and launch
latency, not DRAM. What *does* pay at C1:

1. **Latency, not bandwidth.** An expert's 2.53 MiB fits in 148 x 224 KiB = 32 MiB of aggregate
   L1 if the kernel asks for no SMEM. Turning a 388 ns HBM read into a 20 ns L1 read on the
   second and later passes over the same expert matrix is worth more than any L2 policy.
2. **Prefetch the selected experts as soon as the router decides.** The router output is
   available ~2 GEMMs before the expert GEMM needs the weights. Issue
   `prefetch.global.L2::evict_last` over the 8 selected expert slabs (128 B stride,
   20.25 MiB / 128 B = 166k prefetches per layer per rank — too many for one warp; better to
   have the router kernel's tail warps issue a strided subset, or issue
   `cp.async.bulk.prefetch` at TMA granularity). `[inferred; the PTX is verified to assemble]`
3. **Keep the 256-expert table itself out of L2.** The router reads a `[256, 6144]` gate matrix
   (1.5 M params, ~1.6 MB NVFP4 all-ranks) — that is small and hot, leave it normal. But the
   *unselected* 248 experts must never be touched; if any prefetch or speculative read walks
   them, 648 MiB blows the whole cache.

### The concurrency-64 question: KV vs weights

At C64 with DSA `index_topk = 2048`:

```
KV read per query per layer  : 2048 tokens x 576 values x 1 B (FP8) = 1,179,648 B (1.125 MiB)
KV read per layer at C64     : 75,497,472 B (72 MiB)
8 hot experts per layer      : 21,233,664 B (20.25 MiB)
                        total  92.25 MiB   vs ~63 MiB usable shared L2
```

**These now collide.** With `index_topk_freq = 4` the indexer only re-selects every 4th token,
so 3 of every 4 steps re-read the *same* 2048 KV entries — that KV is genuinely reusable and
worth caching. But at 72 MiB it will evict every expert weight if left at default priority,
and the experts *change* every token, so they are the ones that should be marked streaming.

Recommended policy split (this is the inversion of the usual intuition):

| Buffer | Reuse pattern at C64 | Policy | PTX |
|---|---|---|---|
| DSA-selected KV pages | re-read 4x (index_topk_freq=4) | **persist** | `ld.global.nc.L2::evict_last.v8.b32` or `createpolicy.fractional.L2::evict_last` + `.L2::cache_hint` |
| Expert weights (8 of 256, change every token) | read once, then dead | **stream** | `ld.global.nc.L2::evict_first.v8.b32` |
| Attention/dense weights (every token, every step) | re-read every token | persist-ish; leave normal | default |
| Activations / staging buffers | write-once, read-once | `L1::no_allocate` on the read; `discard.global.L2` before overwrite | `ld.global.nc.L1::no_allocate...` / `discard.global.L2 [p],128` |
| Full 256-expert table | never | never touch | - |

`[inferred from measured L2 capacity + config.json + the measured policy results in section 4.4]`

Note the section 4.4 finding applies here too: **marking the streaming side `evict_first` is
what wins; marking the hot side `evict_last` alone can lose.** So the highest-value single
change is `evict_first` on the expert-weight loads, not `evict_last` on KV.

And the 256-bit constraint bites: to use `.L2::evict_first` directly your expert-weight loads
must be `.v8.b32`/`.v4.b64`. NVFP4 weights at 32 B/thread is 64 values per thread, which is a
natural tile granularity anyway. If your loader is `uint4`-shaped, either widen it or build a
`createpolicy.fractional.L2::evict_first` descriptor once in a uniform register and use
`.L2::cache_hint` on the existing width.

---

## 9. Quick reference: every control, one line each

| Goal | Mechanism | Scope |
|---|---|---|
| Keep a global region in L2 across kernels | `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, n)` + `cudaStreamAttributeAccessPolicyWindow` | stream, <= 79.06 MiB set-aside, window <= 128 MiB |
| Keep specific lines in L2 | `prefetch.global.L2::evict_last [p];` | 128 B line, one-shot |
| Undo the above | `applypriority.global.L2::evict_normal [p], 128;` or `cudaCtxResetPersistingL2Cache()` | line / whole context |
| Per-load L2 policy, 256-bit access | `ld.global.nc.L2::evict_{first,last,normal}.v8.b32` | instruction |
| Per-load L2 policy, any width | `createpolicy.{fractional,range}...b64 %p, ...` + `.L2::cache_hint` | instruction |
| Don't pollute L1 | `.L1::no_allocate` (any width) | instruction |
| L1 eviction priority | `.L1::evict_{normal,unchanged,first,last}` | instruction |
| Bigger L2 prefetch on a miss | `.L2::{64B,128B,256B}` | instruction |
| Read-only path (no L1 write-back tracking) | `ld.global.nc` / `__ldg()` | instruction |
| Bypass L1 entirely | `ld.global.cg` (measured identical latency to `.ca` once out of L1) | instruction |
| Throw away a line without writeback | `discard.global.L2 [p], 128;` | 128 B, data becomes undefined |
| Bulk global -> SMEM | `cp.async.bulk...mbarrier::complete_tx::bytes` (TMA) or `cp.async.cg` | CTA |
| Bulk SMEM -> TMEM | `tcgen05.cp.cta_group::{1,2}.<shape>[.b8x16.b4x16_p64]` | CTA |
| Give L1 the whole 228 KiB | request 0 dynamic SMEM, `cudaFuncAttributePreferredSharedMemoryCarveout = 0` | kernel |
| Give SMEM the whole 228 KiB | `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 232448)` | kernel |
| Cross-SM SMEM | cluster + `cg::cluster_group::map_shared_rank` | cluster (<= 8 portable, 16 on B200) |

---

## 10. Open questions / things I could not source

- **Cross-partition L2 latency.** chipsandcheese reports a penalty; my per-SM probe found only
  a 268-304 cycle spread with no bimodality. Needs an experiment that can control which die a
  physical page lands on.
- **Whether an L2 line can be resident in both partitions simultaneously.** My three
  measurements are consistent with replication, but I did not prove it — an alternative model
  (address-homed lines, with a single SM only able to index half the sets) fits measurement A
  and B equally well and would have different implications for write traffic.
- **256 B sector promotion** (`__l2_sector_promote_256B`, descriptor bit 62). CCCL declares it
  and never sets it; I found no PTX syntax to request it.
- **True TMEM bandwidth.** My 584 B/cycle/SM is a dependent-loop lower bound; the widely-cited
  16 TB/s is from a paper with documented errors. Needs an independent-issue benchmark with
  more warps and `.x64`/`.x128` shapes.
- **`tcgen05` occupancy with multiple CTAs per SM.** The occupancy API says 1; the hardware
  bitmap suggests more is possible. Needs a launch with 2+ CTAs/SM each allocating 128 columns.
- **TMA bandwidth at scale.** My numbers are single-CTA latency, not aggregate throughput.
- The exact PTX ISA wording for `tcgen05` constraints — `docs.nvidia.com/cuda/parallel-thread-execution`
  is a single enormous page and every fetch returned only the table of contents. All
  `tcgen05` semantics here come from the CUDA 13.3 CCCL headers (authoritative for syntax and
  target list) plus third-party write-ups (for the geometry and rules).
- HBM3e per-stack organisation on this SKU. The 7680-bit bus is not 8 x 1024, so at least one
  stack is partially disabled; I could not source the exact configuration.

---

## Sources

### Read on this machine

- `/home/aman/code/cuda-13.3/nvidia/cu13/include/driver_types.h` — `cudaAccessPolicyWindow`,
  `cudaAccessProperty`, `cudaLimitPersistingL2CacheSize`, `cudaDeviceProp` fields
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__annotated_ptr/access_property.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__annotated_ptr/access_property_encoding.h`
  — the 64-bit L2 descriptor bit layout and the `createpolicy` block-size-13 bug note
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__annotated_ptr/createpolicy.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__annotated_ptr/apply_access_property.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__annotated_ptr/associate_access_property.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_alloc.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_ld.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_st.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_cp.h`
- `/home/aman/code/cuda-13.3/nvidia/cu13/bin/ptxas` (13.3.73) — cache-modifier legality matrix
- `/home/aman/code/weights/GLM-5.2-NVFP4/config.json` and `model.safetensors.index.json`
- `/home/aman/code/weights/GLM-5.2-FP8/config.json`

### Measured on this node (scratchpad sources)

`devq.cu`, `mem.cu`, `mem2.cu`, `mem3.cu`, `mem4.cu`, `mem5.cu`, `carve.cu`, `l2part.cu`,
`l2share.cu`, `ldtest.cu`, `ldtest2.cu`, `ldtest3.cu`, `ld256.cu`, `tmem.cu`, `tmem_bench.cu`
in `/tmp/claude-1000/-home-aman-code/930438ff-5f3c-49e6-a3d9-2663231246c6/scratchpad/`.
Disassembly via
`/mnt/persistent/app-data/NotSglang/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump`.

### Web

- https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html — SMEM 228 KiB (cc 10.0),
  carve-out list, 126 MB L2, 64K registers/SM, 64 warps/SM, 32 CTAs/SM, cluster 8 portable /
  16 non-portable on B200. **Does not mention TMEM.**
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html —
  `cudaAccessPolicyWindow` semantics, `hitRatio`, the `0.75 * l2CacheSize` sizing
  recommendation, the "sum of all concurrent kernels" warning, `cudaCtxResetPersistingL2Cache`
- https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut — B200 L1 19.6 ns /
  39 cycles, L2 126 MB in two partitions ~150 ns, L2 BW 21 / 16.8 TB/s, atomics latency
- https://arxiv.org/abs/2512.02189 and https://arxiv.org/html/2512.02189v3 — Jarmusch &
  Chandrasekaran, "Microbenchmarking NVIDIA's Blackwell Architecture". Claims 4 L2 partitions,
  TMEM 16 TB/s read, 11.0-11.4 cycle `tcgen05.mma` latency, 4.14 TB/s STREAM Triad.
  **Treat with suspicion — see next entry.**
- https://girl.surgery/bad_paper — Sophia Wisdom's critique of the above: impossible FLOP rate
  implied by the 11.4-cycle MMA latency, a "32x32 wgmma tile" that is not a legal shape,
  the false claim that TMEM can load from global memory
- https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
  — TMEM 512x128x32b, column allocation, power-of-2 / min 32 / max 512, single-warp
  alloc+dealloc, `tcgen05.ld/st` shapes, warpgroup epilogue requirement
- https://gau-nernst.github.io/tcgen05/ — TMEM address encoding (lane<<16 | col), `tcgen05.ld`
  register counts, `tcgen05.mma` instruction descriptor fields, 128 B swizzle descriptor,
  end-to-end kernel reaching 98% of cuBLAS
- https://www.thesoftwarefrontier.com/p/how-blackwells-tensor-memory-actually — hardware warp
  partitioning of lanes, `UTCATOMSWS.FIND_AND_SET.ALIGN` + `NANOSLEEP` backoff (which I
  independently confirmed by disassembly), dealloc trap handlers, register cost of `.x128`,
  the m128n256 accumulator argument for why TMEM exists. Its "max single allocation 256
  columns" claim is contradicted by my measurement.
- https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/tcgen05_alloc.html — PTX
  ISA 8.6, target list SM_100a/100f/103a/103f/110a/110f
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md
  — `cta_group::1` vs `::2`, MMA tile shapes 64x64 to 256x256, 2-SM dispatch policies.
  Contains no TMEM capacity information.
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html — fetched, but only the
  table of contents was retrievable; section 9.7.17 (`tcgen05`) content could not be read
