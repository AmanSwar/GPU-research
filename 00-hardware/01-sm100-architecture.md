# The B200 SM: what one streaming multiprocessor actually is on SM100

**What this is.** A ground-truth description of one SM100 streaming multiprocessor and of
the GB100 chip it sits in, written for people writing kernels on our 8×B200 node rather
than for people buying one. Every number is labelled `[verified]` (read in a primary doc
or measured on this box), `[measured]` (I ran it here, command included), `[reported]`
(a vendor or a third party asserts it), `[inferred]` (my arithmetic, shown), or
`[unverified]`. Where a widely-repeated claim turned out to be wrong or unsourceable I
say so. Roughly half the numbers below were produced by running something on GPU 4 of
this node on 2026-08-17 with the SM clock locked at 1597 MHz.

---

## Bottom line for our system

- **The SM's non-tensor datapath did not change from Hopper.** 128 FP32 / 64 FP64 /
  64 INT32 cores, 16 SFUs, 4 warp schedulers, 64K registers, 256 KB unified L1/SMEM —
  identical spec lines to SM90 in NVIDIA's own text `[verified]`. Everything Blackwell
  gained is in the tensor path (tcgen05 + Tensor Memory) and in +12% SM count. Do not
  expect Blackwell to make our elementwise/norm/quant kernels (6.6% of profile) faster
  per clock; they are on Hopper hardware.
- **Any kernel that touches Tensor Memory is capped at 1 CTA per SM** `[measured]`.
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns 1 for a tcgen05 kernel at
  every block size, versus 16 for the same kernel without it. All latency hiding in a
  GEMM must happen *inside* one CTA (warp specialization), never by co-residency. This
  is the single biggest occupancy difference from Hopper.
- **NVFP4 MMA has no M-tile smaller than 128** `[verified, PTX ISA Table 42]`. FP8
  (`.kind::f8f6f4`) goes down to M=64, and to M=32 with the `.ws` weight-stationary
  variants; block-scaled FP4 has neither. At concurrency 1 our GEMMs see M≈4–8 rows, so
  the NVFP4 expert GEMMs are running at ~3% M-utilisation of the tensor core. NVFP4's
  win over FP8 at C1 is **entirely bandwidth**, zero tensor throughput.
- **The roofline ridge point is ~230–250 rows and is almost dtype-independent**
  `[inferred from measurements]` — because each precision step doubles FLOP/s and halves
  bytes/weight simultaneously. Concretely: BF16 216, FP8 235, NVFP4 249 rows. C1 with
  EAGLE 3-1-4 gives M≈4; C64 gives M≈256, right at the ridge. That is why C64 gets 91×
  the aggregate tokens of C1, not 64×.
- **Measured: an FP8 GEMM at N=6144,K=2048 takes 3.7–6.0 µs for *every* M from 1 to 256**
  `[measured]`. Our profile's mean dense-GEMM duration is 3547 ms / 596,088 launches =
  **5.95 µs** — i.e. the dense GEMM family at C1 is sitting exactly on that floor. The
  37.1% of GPU time in `nvjet_sm100_*` is not slow math, it is 596k trips through a fixed
  per-launch cost. Fewer, larger, fused GEMMs is the lever; a faster GEMM kernel is not.
- **Peak at our locked 1597 MHz**: 7.74 PFLOP/s dense NVFP4, 3.87 dense FP8, 1.94 dense
  BF16, 60.5 TFLOP/s FP32 `[inferred]`. Measured attainment: 77.4% / 82.1% / 75.2% /
  97.4% `[measured]`. HBM read 6.76 TB/s of 7.68 theoretical `[measured]`.
- **Build with `-gencode arch=compute_100a,code=sm_100a`, never `-arch=sm_100a`.** In
  CUDA 13.3.73 the `-arch=sm_100a` shorthand silently emits `.target sm_100` and ptxas
  rejects every tcgen05 instruction `[measured, reproduced 3 ways]`. `glm-kernels`
  already documents this; it is real and it bites.
- **Cluster tax, measured**: cluster size 1–2 costs nothing, 4–8 costs 4% of resident CTA
  slots, 16 costs 11%, and 20 fails to launch. A cluster-launched kernel is additionally
  capped at 8 CTAs/SM regardless of block size `[measured]`.

---

## 1. The chip: GB100, two dies, one GPU

### 1.1 Physical organisation

| item | value | label |
|---|---|---|
| Transistors | 208 billion, custom TSMC 4NP | `[verified]` nvidia.com Blackwell architecture page |
| Dies | 2 reticle-limited dies | `[verified]` same page |
| Die-to-die link | 10 TB/s chip-to-chip interconnect ("NV-HBI" in NVIDIA marketing) | `[verified]` for the 10 TB/s figure; the *name* NV-HBI I could not source on an NVIDIA page — treat the name as `[unverified]` |
| Presentation | "in a unified single GPU" | `[verified]` same page |
| SMs physically present per die | 80 | `[reported]` Chips and Cheese |
| SMs enabled per die on B200 | 74 | `[reported]` Chips and Cheese |
| SMs enabled, whole GPU | **148** | `[measured]` `cudaDeviceProp::multiProcessorCount` = 148 on all 8 GPUs here |
| L2 | 132,644,864 B = **126.5 MiB** | `[measured]` `cudaDevAttrL2CacheSize`; NVIDIA's tuning guide says "126 MB" `[verified]` |
| L2 organisation | partitioned, ~63 MiB per die | `[reported]` Chips and Cheese |
| HBM bus width | **7680 bit** | `[measured]` `cudaDevAttrGlobalMemoryBusWidth` |
| HBM clock | 3996 MHz (≈7.99 GT/s) | `[measured]` `nvidia-smi -q` |
| HBM capacity reported | 183,359 MiB | `[measured]` `nvidia-smi`; `cudaDeviceProp::totalGlobalMem` = 191,495,471,104 B |
| Max power | 1000 W, min 200 W | `[measured]` `nvidia-smi -q` |

Two of those measured numbers reconcile the datasheet cleanly and are worth writing down:

```
7680 bit × 7.992 GT/s / 8   = 7.67 TB/s   ← datasheet HGX B200 says 7.7 TB/s  ✓
8192 bit (a full 8-stack config) × 15/16 = 7680 bit
192 GiB (8 × 24 GiB HBM3e)              × 15/16 = 180 GiB = 184,320 MiB
                        observed 183,359 MiB  (961 MiB held back for ECC/row-remap)
```

`[inferred]` **B200 is a 15/16-harvested memory configuration of GB100**: one 512-bit
slice of the 8192-bit interface and 12 GiB of the 192 GiB are fused off. This is why the
datasheet says 180 GB / 7.7 TB/s for HGX B200 while GB200 gets 186 GB / 8 TB/s. It also
means the widely-quoted "192 GB B200" is wrong for our part and "180 GB" means 180 *GiB*.

### 1.2 Is it one logical GPU, and is there NUMA inside it?

Yes to the first, and a qualified yes to the second.

- **One logical GPU.** `cudaGetDeviceCount()` = 8 for 8 modules; `isMultiGpuBoard` = 0;
  `cudaDevAttrNumaConfig` = 0 and `cudaDevAttrNumaId` = -1 `[measured]`. There is no CUDA
  API, no MIG partition boundary, and no `nvidia-smi` field that exposes the die split.
  You cannot pin a CTA to a die, you cannot allocate "near" memory, and there is no
  affinity hint in the launch API. From the programming model the die seam does not exist.
- **But it is measurably there in the memory system.** `[reported]`, Chips and Cheese
  measurements on B200:

  | path | latency | bandwidth |
  |---|---|---|
  | L2, local partition | ~150 ns | 21 TB/s |
  | L2, cross partition | higher, "only slightly so" | 16.8 TB/s |
  | global atomic, same partition | 90–100 ns | — |
  | global atomic, cross partition | 190–220 ns | — |
  | L1 hit | 19.6 ns (39 cycles) | — |

  So the cross-die tax is ~20% on L2 bandwidth and ~2× on contended atomic latency.
  Same source reports the block scheduler tends to fill one partition before the other.

**What this means for us.** Our `oneshotAllreduceFusionKernel` (8.2% of GPU time) and
`twoshotAllreduceKernel` (4.3%) do cross-rank reduction through global atomics / L2. If
those land on the "far" L2 partition the atomic path is 2× the latency. We have no lever
to control it — but it is a plausible contributor to the 9.2 µs mean rank-arrival skew in
`hotspots-and-optimization-ledger.md` §2a, and it is worth an ncu `l2_...sector` split
before assuming the skew is purely upstream compute imbalance. `[inferred]`

### 1.3 SM counts across the Blackwell datacenter family

| part | CC | arch string | SMs | note | label |
|---|---|---|---|---|---|
| B200 (HGX / ours) | 10.0 | `sm_100` | **148** | `[measured]` here | |
| B100 | 10.0 | `sm_100` | not sourced | same CC | `[unverified]` |
| GB200 (Grace-Blackwell superchip) | 10.0 | `sm_100` | not sourced; same CC and same 126 MB L2 per NVIDIA's tuning guide, higher power (1200 W) and higher quoted FLOPS | | `[unverified]` on SM count |
| B300 / GB300 "Blackwell Ultra" | 10.3 | `sm_103` | not sourced | separate minor CC, in the same *family* as 10.0 | `[verified]` that 10.3 exists and is family-compatible with 10.0 |
| — | 11.0 | `sm_110` | — | exists in CUDA 13.3; **48 warps/SM, 24 blocks/SM, 1536 threads/SM, 228 KB SMEM** — a different SM shape from 10.x. PTX ISA notes `sm_101a` was *renamed* to `sm_110a` from PTX ISA 9.0 | `[verified]` from CUDA 13.3 programming guide Table 30 and PTX ISA |
| RTX PRO 6000 / consumer | 12.0 / 12.1 | `sm_120` / `sm_121` | — | 48 warps/SM, 128 KB unified cache, 100 KB max SMEM, **no tcgen05** | `[verified]` |

I could not source per-part SM counts for B100/GB200 anywhere primary. Do not quote a
number for those. Our part is 148, measured, on all eight modules.

---

## 2. Inside one SM100 SM

### 2.1 The block diagram, in numbers

NVIDIA's own wording, CUDA 13.0 Programming Guide §20.9.1 "Compute Capability 10.0 —
Architecture" `[verified]`:

> A Streaming Multiprocessor (SM) consists of:
> 128 FP32 cores …, 64 FP64 cores …, 64 INT32 cores …, 4 mixed-precision
> fifth-generation Tensor Cores …, 16 special function units …, 4 warp schedulers.
> An SM statically distributes its warps among its schedulers. Then, at every
> instruction issue time, each scheduler issues one instruction for one of its assigned
> warps that is ready to execute, if any.

Divided by the 4 processing blocks (partitions):

| per processing block (×4 per SM) | count | label |
|---|---|---|
| warp scheduler | 1 | `[verified]` |
| dispatch | 1 instruction / clock / scheduler | `[verified]` from the "issues one instruction" wording |
| FP32 lanes | 32 | `[inferred]` 128/4 |
| INT32 lanes | 16 | `[inferred]` 64/4 |
| FP64 lanes | 16 | `[inferred]` 64/4 |
| SFU lanes | 4 | `[inferred]` 16/4 |
| 5th-gen Tensor Core | 1 | `[verified]` |
| Tensor Core rate | 1024 16-bit MAC / clock | `[reported]` Chips and Cheese |
| register file | 16,384 × 32-bit = 64 KiB | `[inferred]` 65536/4 |
| Tensor Memory | 128 lanes × 128 columns × 32 bit = 64 KiB | `[inferred]` from PTX 512 cols/SM ÷ 4; Chips and Cheese independently reports "64 KB" per partition |

A 32-lane FP32 block issuing one warp instruction per clock means an FP32 FMA warp
instruction retires in 1 clock per scheduler; INT32 at 16 lanes takes 2 clocks per warp.
Note the consequence: **an SM100 SM cannot co-issue FP32 and INT32 at full rate** the way
GA100 could not either — with 64 INT32 lanes vs 128 FP32, integer-heavy index math in a
GEMM epilogue costs 2 slots per warp-instruction. `[inferred]`

**Measured check on the FP32 lane count** (16 independent FMA chains, fully unrolled,
2048 threads/SM, 148 SMs, 100k iterations, GPU 4, clock pinned 1597 MHz):

```
FP32 FMA sustained: 58.96 TFLOPS  →  58.96e12 / (148 × 1.597e9 × 2) = 124.7 lanes/SM
```

97.4% of 128. `[measured]` — confirms both the 128-lane figure and that the 1597 MHz lock
is real (nvidia-smi read 1597 MHz throughout, power 250–460 W against a 1000 W cap, no
throttle reason asserted).

### 2.2 Three 256 KB SRAMs

This is the cleanest way to hold SM100 in your head:

| SRAM | size per SM | addressing | label |
|---|---|---|---|
| Register file | 65,536 × 32 bit = **256 KiB** | per-thread, static | `[measured]` `cudaDevAttrMaxRegistersPerMultiprocessor` = 65536 |
| Unified L1 / texture / shared | **256 KiB** | shared carved out at 0/8/16/32/64/100/132/164/196/**228** KB, remainder is L1 | `[verified]` PG Table 32; `[measured]` `cudaDevAttrMaxSharedMemoryPerMultiprocessor` = 233472 = 228 KiB |
| Tensor Memory (TMEM) | 512 columns × 128 lanes × 32 bit = **256 KiB** | `tcgen05.alloc`, dynamic, 32-column granularity | `[verified]` PTX ISA §9.7.17.1 |

148 × 768 KiB = **111 MiB of SM-level SRAM**, sitting in front of 126.5 MiB of L2.
Practically: the whole active-expert working set of one GLM-5.2 MoE layer at TP8
(8 experts × 3 × 6144 × 2048 / 8 ranks × 0.5625 B/elem ≈ **21.2 MiB**) fits in L2 six
times over. It is *streamed* from HBM anyway because a different 8 of 256 experts are
picked each token. `[inferred]`

Notes that matter when you allocate:

- CUDA reserves **1 KiB** of shared memory per thread block; max per block is therefore
  227 KiB, not 228 `[verified]` tuning guide; `[measured]`
  `cudaDevAttrMaxSharedMemoryPerBlockOptin` = 232448 = 227 KiB,
  `cudaDevAttrReservedSharedMemoryPerBlock` = 1024.
- **Static** shared allocations are still capped at 48 KiB for architectural
  compatibility. Anything above that must be dynamic + `cudaFuncSetAttribute(...,
  cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` `[verified]`. In PTX, only
  architecture-specific targets (`sm_100a`, `sm_103a`) get the 228 KB static allowance
  `[verified]` PTX ISA §5.1.7 table.
- 32 shared memory banks, unchanged `[verified]` PG Table 31.

### 2.3 Tensor Memory in detail

TMEM is the actual new thing in the SM100 datapath. From the PTX ISA `[verified]`:

- Organised as **lanes** (rows) × **columns**; on `sm_100a`/`sm_100f` it is
  **512 columns × 128 rows per CTA, 32-bit cells**.
- A TMEM address is 32 bits: `[31:16] = lane index`, `[15:0] = column index`.
- Allocated with `tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32 [dst], nCols;`
  by **one warp**. `nCols` must be a power of two in **[32, 512]**; allocating a column
  allocates all 128 lanes of it. The instruction **blocks** until the request can be
  satisfied. All TMEM must be explicitly `tcgen05.dealloc`'d before kernel exit, and every
  `tcgen05` instruction in a kernel must use the same `.cta_group`.
- With `.cta_group::2`, one warp from **each** of the two peer CTAs must issue the alloc
  collectively, and the issuing warp must guarantee the peer CTA is launched and live.

A **CTA pair** is any two CTAs in the cluster whose `%cluster_ctarank` differ only in the
last bit; even = rank bit 0 clear, odd = set. A `.cta_group::2` tcgen05 op touches the
TMEM of both CTAs. `[verified]`

Working PTX, compiled and run on this box `[measured]`:

```cuda
__global__ void tmem_probe(unsigned* out, unsigned ncols){
  __shared__ unsigned smem_addr;
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "l"(__cvta_generic_to_shared(&smem_addr)), "r"(ncols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  __syncthreads();
  unsigned taddr = smem_addr;                       // [31:16]=lane, [15:0]=column
  __syncthreads();
  if (threadIdx.x < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :: "r"(taddr), "r"(ncols));
}
```

`nCols` = 32, 128, 256 and 512 all succeed with a 148-block grid; the returned address is
`0x00000000` (lane 0, column 0) — i.e. with one CTA per SM every CTA gets the base of
TMEM. `numRegs` = 10, `sharedSizeBytes` = 4, `ptxVersion` = 100, `binaryVersion` = 100.

### 2.4 What changed from SM90 to SM100 in the SM datapath

Put the two Programming Guide sections side by side and the answer is stark `[verified]`:

| | SM90 (H100) | SM100 (B200) |
|---|---|---|
| FP32 cores | 128 | 128 |
| FP64 cores | 64 | 64 |
| INT32 cores | 64 | 64 |
| SFUs | 16 | 16 |
| warp schedulers | 4 | 4 |
| max warps / SM | 64 | 64 |
| max blocks / SM | 32 | 32 |
| registers / SM | 64K × 32-bit | 64K × 32-bit |
| max regs / thread | 255 | 255 |
| unified L1+SMEM | 256 KB | 256 KB |
| SMEM carveouts | 0/8/16/32/64/100/132/164/196/228 KB | identical |
| max SMEM / block | 227 KB | 227 KB |
| Tensor Cores | 4 × 4th gen | 4 × 5th gen |
| Tensor Memory | — | **256 KB, `tcgen05`** |
| MMA issue | `wgmma` (warpgroup, 128 threads) | `tcgen05.mma` (single warp issues, 1 or 2 CTAs) |
| accumulator lives in | registers | **TMEM** |
| operand A from | SMEM or registers | SMEM or **TMEM** |
| MMA M-tile floor | 64 | 64 (f16/f8f6f4), 32 (`.ws`), **128 (block-scaled f4)** |
| new dtypes | FP8 e4m3/e5m2 | + e2m3, e3m2, e2m1 (FP6/FP4), ue8m0 / ue4m3 scales |
| max cluster | 8 | 8 portable, **16 non-portable with opt-in** |
| dynamic work stealing | — | **Cluster Launch Control** (`clusterlaunchcontrol.try_cancel`) |
| L2 | 50 MB | 126 MB |
| SMs per GPU | 132 (H100 SXM) | 148 |

The whole generational delta is: +12% SMs, 2× tensor MAC/clock at the same dtype, a new
4× FP4 path, a 256 KB accumulator store that gets the accumulator out of the register
file, and a work-stealing primitive. Nothing else.

`[verified]` CUTLASS states the throughput multipliers explicitly:

| `tcgen05.mma` kind | throughput | layouts |
|---|---|---|
| `.kind::tf32` | 2× Hopper TF32 | TN NT TT NN |
| `.kind::f16` | 2× Hopper FP16 | TN NT TT NN |
| `.kind::i8` | 2× Hopper INT8 | TN NT TT NN |
| `.kind::f8f6f4` | 2× Hopper FP8 | TN NT TT NN |
| `.kind::mxf8f6f4.block_scale` | 2× Hopper FP8 | TN NT TT NN |
| `.kind::mxf4.block_scale` | **4× Hopper FP8** | **TN only** |
| `.kind::mxf4nvf4.block_scale.scale_vec_size::{2X,4X}` | **4× Hopper FP8** | **TN only** |

**Trap worth naming:** feeding 4-bit operands to `.kind::f8f6f4` or `.kind::mxf8f6f4`
gets you FP8 rate, not FP4 rate. The 4× path exists only under `.kind::mxf4` /
`.kind::mxf4nvf4`, and only with TN layout (row-major A, column-major B). If a kernel of
ours quantises to NVFP4 and then dispatches an `mxf8f6f4` kernel, it pays FP4's accuracy
cost for FP8's speed.

NVFP4 vs MXFP4, per CUTLASS `[verified]`:

| type | scale type | SF vector size (dense) | OCP compliant |
|---|---|---|---|
| `mx_float4_t` | `float_ue8m0_t` | 32 | yes |
| `nv_float4_t` | `float_ue4m3_t` | **16** | no |

GLM-5.2's `quantization_config` is `num_bits: 4, type: float, group_size: 16`
`[verified, /home/aman/code/weights/GLM-5.2-NVFP4/config.json]` — i.e. NVFP4, so the
`mxf4nvf4` path with `scale_vec_size::4X` and `ue4m3` scales.

---

## 3. The arch target taxonomy: sm_100, sm_100a, sm_100f, sm_103

### 3.1 The three feature sets

CUDA 13.3 Programming Guide §5.1.2.3 `[verified]`:

- **Baseline** (`compute_100`, no suffix): features intended to persist into later
  architectures. Compatible with all devices of CC 10.0 *and later*.
- **Architecture-specific** (`a` suffix, introduced with CC 9.0): "a small and highly
  specialized set … not guaranteed to be available or might change significantly on
  subsequent compute architectures". `compute_100a` runs on **CC 10.0 and nothing else**.
- **Family-specific** (`f` suffix, **new with CC 10.0**): the subset of architecture-
  specific features common to a whole family. `compute_100f` runs on CC 10.0 **and 10.3**.

Strict superset chain: `compute_100` ⊂ `compute_100f` ⊂ `compute_100a`.

Family compatibility table, CUDA 13.3 PG Table 28 `[verified]`:

| target | runs on CC |
|---|---|
| `compute_100f` | 10.0, 10.3 |
| `compute_103f` | 10.3 (family currently has one member) |
| `compute_110f` | 11.0 (one member) |
| `compute_120f` | 12.0, 12.1 |
| `compute_121f` | 12.1 (one member) |

And the ptxas rule, quoted from `ptxas --help` in **our** toolkit `[verified]`:

```
PTX for .target sm_XY can be compiled to all GPU targets sm_MN, sm_MNa, sm_MNf where MN >= XY.
PTX for .target sm_XYf can be compiled to GPU targets sm_XZ, sm_XZf, sm_XZa where Z >= Y and
sm_XY and sm_XZ belong in same family.
PTX with .target sm_XYa can only be compiled to GPU target sm_XYa.
```

### 3.2 What our toolkit actually accepts

`/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc`, release 13.3, V13.3.73, built
2026-06-09 `[measured]`.

```
$ nvcc --list-gpu-code
sm_75 sm_80 sm_86 sm_87 sm_88 sm_89 sm_90 sm_100 sm_110 sm_103 sm_120 sm_121
```

`--list-gpu-arch` / `--list-gpu-code` show only the *base* names; `ptxas --help` and
`nvcc --help` show the full set, which is the authoritative list:

```
compute_100 compute_100a compute_100f   sm_100 sm_100a sm_100f
compute_103 compute_103a compute_103f   sm_103 sm_103a sm_103f
compute_110 compute_110a compute_110f   sm_110 sm_110a sm_110f
compute_120 compute_120a compute_120f   sm_120 sm_120a sm_120f
compute_121 compute_121a compute_121f   sm_121 sm_121a sm_121f
compute_90  compute_90a                 sm_90  sm_90a
(75/80/86/87/88/89 have no a/f variants)
plus lto_* for every one of the above
```

Max PTX ISA version accepted: **9.3** `[measured, ptxas --list-version]`.
`__nvcc_device_query` on this box returns `100` `[measured]`.

Note there is **no `sm_100a` in the `--list-gpu-code` output but it is a legal
`-gencode` target** — do not use `--list-gpu-code` to decide what you can build.

### 3.3 The `-arch=sm_100a` trap — verified, reproducible

`nvcc --help` says `-arch=sm_100` is shorthand for
`--gpu-architecture=compute_100 --gpu-code=sm_100,compute_100`. You would expect
`-arch=sm_100a` to expand analogously. **It does not do the right thing in 13.3.73.**

Compiling the `tmem_probe` kernel above `[measured]`:

| command | result |
|---|---|
| `nvcc -arch=sm_100  -c tmem.cu` | fails (expected — base target has no tcgen05) |
| `nvcc -arch=sm_100a -c tmem.cu` | **fails**: `ptxas …/tmem.compute_100.ptx: error: Instruction 'tcgen05.alloc' not supported on .target 'sm_100'` |
| `nvcc -arch=sm_100a -o tmem tmem.cu` | **fails**, same error |
| `nvcc -arch=compute_100a -code=sm_100a -c tmem.cu` | **OK** |
| `nvcc -gencode arch=compute_100a,code=sm_100a -o tmem tmem.cu` | **OK**, runs |
| `nvcc -gencode arch=compute_100f,code=sm_100f -c tmem.cu` | **OK** |
| `nvcc -arch=sm_100a --ptx tmem.cu` | emits `.version 9.3 / .target sm_100a` — the PTX-only path *is* correct, which is what makes this so easy to miss |

The failing intermediate is literally named `tmem.compute_100.ptx`: nvcc generated a
non-`a` virtual PTX for the embedded-PTX slot and handed *that* to ptxas. So the shorthand
is fine right up until ptxas sees the fallback PTX.

Our tree already knows: `glm-kernels/CMakeLists.txt` sets
`GLM_CUDA_ARCHS "100a-real"` with the comment "`-arch=sm_100a` shorthand silently
degrades to `.target sm_100` and rejects…" `[verified, local file]`. CMake's `100a-real`
expands to `-gencode arch=compute_100a,code=sm_100a`, which is the working form.

### 3.4 What to actually put in a build

| you are building | flag | why |
|---|---|---|
| a kernel using `tcgen05` / TMEM / 228 KB static SMEM / TMA `.multicast::cluster`, B200 only | `-gencode arch=compute_100a,code=sm_100a` | full arch-specific set, smallest cubin, no PTX JIT at load |
| the same kernel, must also run on B300 | `-gencode arch=compute_100f,code=sm_100f` | `tcgen05.alloc/dealloc/mma` are all available from `sm_100f` (PTX ISA 8.8+); loses only the handful of `sm_100a`-exclusive shapes |
| B200 + B300 with maximum per-part tuning | `-gencode arch=compute_100a,code=sm_100a -gencode arch=compute_103a,code=sm_103a` | e.g. `K=96` for `.kind::mxf4`/`mxf4nvf4` is **`sm_103a` only** `[verified, PTX Table 42 note]` |
| portable / no arch-specific instructions | `-gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100` | keeps a PTX slot for forward compat, per the compatibility guide |
| our node, today | `100a-real` in `CMAKE_CUDA_ARCHITECTURES` | what `glm-kernels` does |

Things that are gated on `a`/`f` and quietly slow or illegal otherwise `[verified, PTX ISA]`:

- all `tcgen05.*` — `sm_100a` / `sm_101a`(→`sm_110a`) from PTX 8.6, `sm_100f` from PTX 8.8
- `cp.async.bulk … .multicast::cluster` — legal on `sm_90`+ but "optimized for target
  architecture sm_90a / sm_100f / sm_100a / sm_103f / sm_103a / sm_110f / sm_110a and may
  have **substantially reduced performance** on other targets"
- static shared memory > 48 KB — needs `sm_90a`/`sm_100a`/`sm_103a`/`sm_110a` (228 KB) or
  `sm_120a`/`sm_121a` (100 KB)
- `.b6p2x16` type restrictions specific to `sm_103a`

---

## 4. The occupancy limits that actually bind on inference kernels

### 4.1 The five budgets

All measured on this box (`cudaDeviceGetAttribute`, GPU 0) `[measured]`, and all
agreeing with CUDA 13.3 PG Tables 30/31 `[verified]`:

| budget | SM100 value | Hopper SM90 | consumer SM120 |
|---|---|---|---|
| threads / SM | 2048 (64 warps) | 2048 (64) | 1536 (48) |
| blocks / SM | 32 | 32 | 24 |
| 32-bit registers / SM | 65,536 | 65,536 | 65,536 |
| registers / thread | 255 | 255 | 255 |
| shared memory / SM | 228 KiB | 228 KiB | 100 KiB |
| shared memory / block | 227 KiB | 227 KiB | 99 KiB |
| **TMEM columns / SM** | **512** | n/a | n/a |

### 4.2 Register-limited occupancy

`max_warps = min(64, floor(65536 / (32 × R)))` `[inferred]`:

| regs/thread R | max warps/SM | occupancy | typical inhabitant |
|---:|---:|---:|---|
| ≤32 | 64 | 100% | elementwise, quant, norm |
| 40 | 51 | 80% | |
| 48 | 42 | 66% | |
| 64 | 32 | 50% | RMSNorm+AR fusion |
| 80 | 25 | 39% | |
| 96 | 21 | 33% | attention softmax |
| 128 | 16 | 25% | DSA indexer top-k |
| 160 | 12 | 19% | |
| 192 | 10 | 16% | |
| 224 | 9 | 14% | |
| 255 | 8 | 12.5% | GEMM epilogue warps |

For a memory-bound kernel (our 3.7% elementwise + 2.4% quant + 0.5% norm), the target is
R ≤ 32 and 64 resident warps, because you need ~64 warps × ~4 outstanding loads to cover
a ~600 ns HBM round trip on 148 SMs. For a `tcgen05` GEMM, occupancy is irrelevant (§4.4).

### 4.3 Shared-memory-limited occupancy — measured

`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, 128-thread kernel, dynamic SMEM
`[measured]`:

| dynamic SMEM per block | +1 KiB reserved | blocks/SM | warps/SM | occupancy |
|---:|---:|---:|---:|---:|
| 0 | 1 KiB | 16 | 64 | 100% |
| 16 KiB | 17 KiB | 13 | 52 | 81% |
| 32 KiB | 33 KiB | 6 | 24 | 38% |
| 48 KiB | 49 KiB | 4 | 16 | 25% |
| 64 KiB | 65 KiB | 3 | 12 | 19% |
| 112 KiB | 113 KiB | 2 | 8 | 12.5% |
| 227 KiB | 228 KiB | 1 | 4 | 6% |

The model is exactly `floor(233472 / (smem + 1024))`. The carveout quantisation
(0/8/16/32/64/100/132/164/196/228 KB) is handled by the driver picking the smallest
carveout that admits the requested amount; it did not cost occupancy in any of the cases
above.

### 4.4 The TMEM cliff: tcgen05 ⇒ 1 CTA per SM

The single most consequential occupancy fact on SM100, and I could not find it stated in
any NVIDIA doc — so I measured it `[measured]`:

```
plain    T= 128 blocks/SM=16   tcgen05 T= 128 blocks/SM= 1   (4 warps/SM)
plain    T= 256 blocks/SM= 8   tcgen05 T= 256 blocks/SM= 1   (8 warps/SM)
plain    T= 512 blocks/SM= 4   tcgen05 T= 512 blocks/SM= 1   (16 warps/SM)
plain    T=1024 blocks/SM= 2   tcgen05 T=1024 blocks/SM= 1   (32 warps/SM)
```

The two kernels are byte-identical apart from the inline `tcgen05.alloc / relinquish /
dealloc`. Registers: 10. Shared: 4 bytes. Neither budget explains the drop. The toolchain
marks the cubin as 1-CTA-per-SM the moment TMEM is touched.

Mechanically this is consistent with the PTX spec: TMEM is a 512-column SM-level resource,
`tcgen05.alloc` is *blocking*, and `nCols` may be as large as 512 — so a second CTA
requesting TMEM on the same SM could deadlock behind the first. The compiler resolves it
by refusing co-residency.

**Consequences for our kernels:**

1. A GEMM/attention kernel using `tcgen05` gets **one CTA and at most 32 warps** of the
   SM's 64. Tail effects cannot be hidden by a second CTA; they must be hidden by
   warp-specialised producer/consumer stages inside the one CTA, or by persistence +
   Cluster Launch Control.
2. Grid sizing changes: the useful grid for a GEMM is **≤148 CTAs** (or ≤74 clusters of 2
   for `cta_group::2`), not "as many as fit". Anything beyond that serialises into a
   second wave and the wave quantisation is brutal — one extra tile costs a full wave.
3. You cannot overlap a `tcgen05` GEMM with anything else on the same SM. Kernel-level
   overlap (separate streams) is the only overlap available, and it is subject to the
   same 148-SM budget.

### 4.5 CTA size, cluster size, and SM residency — measured

Cluster launch on this box, `cudaOccupancyMaxActiveClusters` plus real
`cudaLaunchKernelEx` `[measured, GPU 4]`:

| cluster size | max active clusters | ⇒ resident CTAs | vs cluster=1 | launch |
|---:|---:|---:|---:|---|
| 1 | 1184 | 1184 | 100% | ok |
| 2 | 592 | 1184 | 100% | ok |
| 4 | 284 | 1136 | 95.9% | ok |
| 8 | 142 | 1136 | 95.9% | ok |
| 12 | 87 | 1044 | 88.2% | ok |
| 16 | 66 | 1056 | 89.2% | ok (needs opt-in) |
| 20 | 24 (reported) | — | — | **`cluster misconfiguration`** |
| 32 | 0 | — | — | **`cluster misconfiguration`** |

Two independent facts fall out:

- **Max cluster size on B200 is 16.** The tuning guide says max portable is 8 and B200
  allows a non-portable 16 with `cudaFuncAttributeNonPortableClusterSizeAllowed`
  `[verified]`; 20 and 32 are rejected at launch `[measured]`. Since a cluster must be
  co-scheduled within one GPC, GB100 GPCs must contain ≥16 SMs. The exact GPC count is
  **not sourced** — 80 physical SMs per die ÷ ≥16 = ≤5 GPCs/die is the bound, but I have
  no primary source and will not guess.
- **A cluster-launched kernel is capped at 8 CTAs/SM**, independent of block size:
  1184 = 148 × 8 for T=64 (which allows 32 blocks/SM without clusters), T=128 (16), and
  T=256 (8). At T=512 the warp budget binds first (592 = 148 × 4). `[measured]`

**Rules of thumb for our kernels** `[inferred from the above]`:

- Cluster 2 is free. If you want `cta_group::2` MMA (M-tile 256), take it — it costs
  nothing in residency and doubles the MMA M-tile.
- Cluster 4 and 8 cost 4%. Worth it if DSM multicast saves more than 4% of traffic — for a
  weight-stationary decode GEMM broadcasting B to 4–8 CTAs, it does.
- Cluster 16 costs 11% and is non-portable (won't run on B300 without re-testing). Only
  take it for a measured win.
- For a `tcgen05` kernel the 8-CTA/SM cluster cap is moot; you are at 1 anyway.

### 4.6 Cluster Launch Control

SM100 adds `clusterlaunchcontrol.try_cancel`, which lets a persistent worker cluster
*steal* not-yet-launched grid coordinates from the hardware work distributor
`[verified, CUTLASS blackwell_cluster_launch_control.md]`. Rules: launch a grid with one
CTA per output tile (not a persistent-sized grid); each worker starts on its own
`blockIdx` and then queries for more; every ClcID is guaranteed to be processed either by
being launched or by being cancelled-and-claimed; the query operates at *cluster*
granularity (a 2×2 worker consumes 2×2 ClcIDs at once).

This is the correct answer to "some SMs are busy with another kernel". For us — where the
scheduler is co-running collectives, attention and GEMM on the same device — a statically
persistent GEMM would be exactly the imbalanced case CLC exists to fix. Worth checking
whether the `nvjet_sm100_*` kernels cuBLAS picks use it; I did not verify that they do.

---

## 5. Clocks and peak FLOPS

### 5.1 The clock domain on our box

`[measured]`, all 8 GPUs:

| | value |
|---|---|
| current SM/graphics clock | **1597 MHz** on all 8, at idle and under full FP32/tensor load |
| max SM clock ("Max Customer Boost") | 1965 MHz |
| supported graphics clocks | 247 discrete points, 1965 down to 120 MHz, ~7.5 MHz apart; **1597 MHz is one of them** (between 1605 and 1590) |
| memory clock | 3996 MHz (single point) |
| video clock | 1530 MHz |
| power limit | 1000 W (min 200, max 1000) |
| power during a saturating FP8 GEMM | 423–462 W |
| throttle reasons asserted | none |

1597 / 1965 = **81.3% of maximum boost**. `benchmark/RESULTS.md` records
`SM clocks | 1597 MHz` in its provenance block `[verified, local file]`, so every number
in our benchmark corpus is at this clock. Everything below is quoted at 1597 MHz first.

Because the clock never moved under a 460 W tensor load against a 1000 W cap, the lock is
real and there is ~2× thermal/power headroom. **If we ever unlock to 1965 MHz we get a
free 23% on every compute-bound kernel** — but the memory clock is fixed at 3996 MHz, so
memory-bound kernels (which is most of C1 decode) would gain nothing, and run-to-run
comparability with the existing corpus would break. `[inferred]`

### 5.2 FLOP per SM per clock

| operation | FLOP (or OP) / SM / clock | label |
|---|---:|---|
| FP32 FMA (vector) | 256 | `[verified]` 128 cores × 2 |
| FP64 FMA (vector) | 128 | `[verified]` 64 cores × 2 |
| INT32 add/mul | 64 | `[verified]` 64 cores |
| SFU transcendental | 16 | `[verified]` 16 SFUs |
| TF32 tensor, dense | 4,096 | `[inferred]` from datasheet ratio (TF32 = ½ FP16) |
| FP16 / BF16 tensor, dense | 8,192 | `[inferred]`; corroborated by "1024 16-bit MAC/clock/partition" × 4 partitions × 2 `[reported]` |
| FP8 / FP6 tensor (`f8f6f4`, `mxf8f6f4`), dense | 16,384 | `[inferred]` |
| INT8 tensor, dense | 16,384 OP | `[inferred]` |
| FP4 via `mxf4` / `mxf4nvf4`, dense | 32,768 | `[inferred]`; CUTLASS's "4× Hopper FP8" corroborates `[verified]` |
| FP4 via `f8f6f4` / `mxf8f6f4`, dense | 16,384 | `[verified]` CUTLASS "2× Hopper FP8" |
| any of the above with `.sp` sparsity | ×2 | `[verified]` datasheet footnote |

### 5.3 Peak table

148 SMs. "NVIDIA ref" is the clock implied by NVIDIA's own HGX B200 datasheet numbers
(see §5.4).

| dtype (dense unless noted) | @1597 MHz (ours) | @1856 MHz (NVIDIA tensor ref) | @1965 MHz (max boost) | datasheet HGX B200 |
|---|---:|---:|---:|---:|
| FP32 vector | 60.5 TFLOP/s | — | 74.5 TFLOP/s | 75 TFLOP/s |
| FP64 vector / tensor | 30.3 TFLOP/s | — | 37.2 TFLOP/s | 37 TFLOP/s |
| TF32 tensor | 968 TFLOP/s | 1,110 | 1,191 | 1,100 dense (2.2 PF sparse) |
| BF16 / FP16 tensor | **1,936 TFLOP/s** | 2,251 | 2,382 | 2,250 dense (4.5 PF sparse) |
| FP8 / FP6 tensor | **3,872 TFLOP/s** | 4,502 | 4,765 | 4,500 dense (9 PF sparse) |
| INT8 tensor | 3,872 TOP/s | 4,502 | 4,765 | 4,500 dense (9 POPS sparse) |
| NVFP4 / MXFP4 tensor | **7,745 TFLOP/s** | 9,004 | 9,530 | 9,000 dense (18 PF sparse) |
| NVFP4 sparse | 15,490 TFLOP/s | 18,008 | 19,059 | 18,000 |

Per-node (×8): **62.0 PFLOP/s dense NVFP4, 31.0 dense FP8, 15.5 dense BF16 at 1597 MHz.**

### 5.4 A discrepancy worth naming

NVIDIA's HGX B200 datasheet numbers are not all at the same clock `[inferred, arithmetic
shown]`:

```
FP32   75 TFLOP/s / (148 × 256)   = 1.979 GHz
FP64   37 TFLOP/s / (148 × 128)   = 1.952 GHz
FP16 2250 TFLOP/s / (148 × 8192)  = 1.856 GHz
FP8  4500 TFLOP/s / (148 × 16384) = 1.856 GHz
FP4  9000 TFLOP/s / (148 × 32768) = 1.856 GHz
```

The vector numbers are quoted at ~boost (1965 MHz, rounded); every tensor number is quoted
at **1.856 GHz, 94.5% of boost**. So the datasheet tensor peaks already bake in a ~5.5%
de-rate that the FP32/FP64 rows do not. When someone says "B200 is 9 PFLOPS dense FP4",
that is 1.856 GHz. At true boost it would be 9.53; at our lock it is 7.74.

Also, note the datasheet's headline "**144 PFLOPS FP4**" for HGX B200 is *sparse*, per its
own footnote 2 ("Specifications in sparse. Dense is one-half of the sparse spec shown.")
`[verified]`. The dense node number is 72 PFLOPS at NVIDIA's reference clock, 62 at ours.
Nothing in our stack uses 2:4 structured sparsity, so 144 and 72 are both irrelevant to us.

### 5.5 Measured attainment on this box

All at 1597 MHz on GPU 4, torch 2.11.0+cu130 `[measured]`:

| kernel | shape | measured | % of dense peak @1597 |
|---|---|---:|---:|
| FP32 FMA loop | 148×2048 threads | 58.96 TFLOP/s | 97.4% |
| BF16 GEMM (`F.linear`) | 8192³ | 1,456.9 TFLOP/s | 75.2% |
| FP8 e4m3 GEMM (`torch._scaled_mm`) | 8192³ | 3,178.6 TFLOP/s | 82.1% |
| NVFP4 GEMM (`torch._scaled_mm`, 1×16 block scales) | 16384×8192×8192 | **5,993.9 TFLOP/s** | 77.4% |
| HBM read (float4 grid-stride, 8 GiB) | — | **6.76 TB/s** | 88.0% of 7.68 |
| HBM copy (read+write, 8 GiB each way) | — | 6.10 TB/s | 79.4% |

Note FP8 attains a *higher* fraction of peak than BF16 (82% vs 75%), which is the opposite
of the usual pattern and suggests the BF16 kernel selection has room. Also note NVFP4 at
77% — the 4× tensor path is real and reachable from stock PyTorch.

---

## 6. Roofline: ridge points and the minimum M that matters

### 6.1 The two bandwidths

- HBM: **6.76 TB/s** measured read, 7.68 TB/s theoretical (7680 bit × 7.992 GT/s).
- L2: **21 TB/s** local partition, 16.8 TB/s cross-partition `[reported]`, not verified
  here. My own L2 sweep was overhead-dominated below 128 MB and is not worth quoting.

### 6.2 Ridge points at 1597 MHz

Ridge point = peak FLOP/s ÷ bandwidth, in FLOP per byte.

| dtype | theoretical peak ÷ 7.68 TB/s | measured peak ÷ 6.76 TB/s | vs L2 (21 TB/s, theoretical peak) |
|---|---:|---:|---:|
| BF16 | 252 FLOP/B | 216 | 92 |
| FP8 | 504 | 470 | 184 |
| NVFP4 | 1,008 | 887 | 369 |

### 6.3 Translating to a minimum batch size

For a weight-stationary decode GEMM `C[M,N] = A[M,K] · B[K,N]` with M small, the traffic
is dominated by streaming B once:

```
FLOPs = 2·M·N·K            bytes ≈ bpe(B)·N·K
arithmetic intensity  AI = 2M / bpe(B)
```

with `bpe(BF16)=2`, `bpe(FP8)=1`, `bpe(NVFP4)=0.5 + 1/16 = 0.5625` (4-bit data plus one
E4M3 scale per 16 elements).

| dtype | AI(M) | M* at measured ridge | M* at theoretical ridge | M* if B is L2-resident |
|---|---|---:|---:|---:|
| BF16 | M | **216** | 252 | 92 |
| FP8 | 2M | **235** | 252 | 92 |
| NVFP4 | 3.556M | **249** | 284 | 104 |

**The ridge point is essentially dtype-invariant at M ≈ 230 ± 20 rows.** Each precision
step doubles the flop rate and halves the bytes per weight, so the two effects cancel
exactly. Quantising further does not move the batch size at which you become compute
bound; it only lowers the *time* on the memory-bound side of the ridge.

And 230–250 rows is, to within rounding, the `cta_group::2` NVFP4 MMA M-tile of **256**.
Blackwell's tile geometry and its roofline agree with each other.

### 6.4 The measured plateau — this is the finding

FP8 e4m3 → BF16, GPU 4, 1597 MHz, `torch._scaled_mm`, GLM-5.2 TP8 shapes, microseconds
`[measured]`:

| shape (N,K) | M=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 192 | 256 | 384 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| o_proj (6144, 2048) | 4.60 | 4.25 | 3.94 | **3.67** | 4.10 | 4.29 | 4.25 | 4.75 | 5.16 | 6.00 | 6.32 | 8.20 | 27.3 |
| q_b (2048, 2048) | 5.38 | 5.13 | 4.65 | 4.65 | 3.45 | 3.66 | 3.62 | 3.91 | 4.13 | 3.99 | 4.79 | 5.11 | 6.27 |
| MoE w13, 1 expert (512, 6144) | 6.76 | 6.70 | 6.41 | 6.89 | 6.49 | 7.14 | 7.31 | 6.54 | 6.61 | 6.52 | 6.74 | 6.65 | 7.17 |
| MoE w2, 1 expert (6144, 512) | 2.70 | 2.50 | 2.67 | 2.50 | 2.71 | 2.70 | 3.04 | 3.26 | 3.49 | 4.11 | 3.93 | 5.28 | 7.49 |

NVFP4 at 8192×8192 shows the same shape more starkly `[measured]`: 15.6 µs at M=8,
15.4 at M=16, 15.5 at M=32, 15.8 at M=64, 12.7 at M=128, 12.5 at M=256, 13.5 at M=512 —
then finally 24.1 µs at M=1024. **Sixty-four times more work for free.**

Three things follow:

1. **Below M≈256 the GEMM is a fixed-cost event.** Measured launch-to-launch floor on this
   box: **5.97 µs** eager, **1.475 µs** inside a CUDA graph `[measured, 100 tiny kernels
   in one graph]`. Add weight streaming and you land on the 4–6 µs plateau above.
2. **Our profile is sitting exactly on that plateau.** 3547 ms of dense GEMM over 596,088
   launches = 5.95 µs mean `[verified, local ledger]`. That is the floor, not a slow
   kernel. The dense-GEMM 37.1% is a *count* problem.
3. **Therefore the levers are, in order:** (a) fuse adjacent GEMMs so there are fewer of
   them; (b) make sure every one of the 596k launches is inside a CUDA graph (1.5 µs, not
   6 µs); (c) raise M — spec-decode width, or batching. Writing a faster GEMM kernel is
   the one lever that provably does nothing here.

### 6.5 Why NVFP4 still wins at M=4, and what it wins

At C1 with EAGLE 3-1-4 the target model sees ~4 rows per GEMM. Compare the two
quantisations at M=4 `[inferred, from verified tile tables]`:

| | FP8 (`f8f6f4`) | NVFP4 (`mxf4nvf4`) |
|---|---|---|
| smallest MMA M-tile | 64 (32 with `.ws`) | **128, no `.ws` variant** |
| M-utilisation at M=4 | 6.25% (12.5% with `.ws`) | 3.1% |
| raw tensor rate | 1× | 2× |
| **effective tensor rate at M=4** | 1× | **1×** — the 2× rate is exactly cancelled by the 2× coarser tile |
| bytes per weight | 1.0 | 0.5625 |
| **weight-streaming time** | 1× | **0.56×** |

So at decode batch 1, NVFP4 buys us a 1.78× reduction in weight traffic and **nothing at
all** from the tensor cores. That is fine — we are 30–60× below the ridge, so bandwidth is
all that matters. But it means: do not expect NVFP4 to help a compute-bound kernel at C1,
and do expect the FP8 build to be competitive on any op whose M is between 32 and 64,
where FP8's finer M-tile (and the `.ws` M=32 variants) actually fit the work.

PTX ISA Table 42, the M-shapes, verbatim `[verified]`:

| `.kind` | `.ws` | CTA group | M shapes | N | K |
|---|---|---|---|---|---|
| `f16` | no | 1 | 64, 128 | 8…256 step 8 | 16 |
| `f16` | no | 2 | 128, 256 | 16…256 step 16 | 16 |
| `f16` | yes | 1 | **32**, 64, 128 | 64/128/256 | 16 |
| `f8f6f4` | no | 1 | 64, 128 | 8…256 | 32 |
| `f8f6f4` | yes | 1 | **32**, 64, 128 | 64/128/256 | 32 |
| `mxf8f6f4` | no | 1 | **128 only** | 8…256 step 8 | 32 |
| `mxf4` / `mxf4nvf4` | no | 1 | **128 only** | 8…256 step 8 | 64 |
| `mxf4` / `mxf4nvf4` | no | 2 | 128, 256 (+256 with K=96 on `sm_103a`) | 16…256 step 16 | 64 |
| `mxf4` / `mxf4nvf4` | yes | 1 or 2 | **Invalid** | — | — |

And CUTLASS's tile shapes for `nv_float4_t × nv_float4_t` `[verified]`, which is what any
NVFP4 kernel we write will use:

| 1/2 SM | MmaTileShape M×N×K | layouts |
|---|---|---|
| 1SM | 128×128×256, 128×192×256, 128×256×256 | **TN only** |
| 2SM | 256×128×256, 256×192×256, 256×256×256 | **TN only** |

There is no 64-row and no 32-row NVFP4 tile anywhere in the ISA. If a decode kernel wants
finer M granularity in FP4, the only route is to batch more rows into the tile — which for
MoE means grouping tokens by expert so each expert's tile is full, not padding each
expert's 1–4 tokens up to 128.

### 6.6 The MoE arithmetic for GLM-5.2 at TP8

`[inferred]` from `config.json` `[verified]`: hidden 6144, `moe_intermediate_size` 2048,
256 routed experts, top-8, 1 shared expert, 78 layers, `first_k_dense_replace` 3.

Per rank, per layer, NVFP4 (0.5625 B/element):

| tensor | params/rank | bytes |
|---|---:|---:|
| one expert (w1+w3+w2) = 3 × 6144 × 2048 / 8 | 4.72 M | 2.65 MiB |
| 8 activated experts | 37.7 M | **21.2 MiB** |
| attention (q_a, q_b, kv_a, kv_b, o) | 34.7 M | 19.5 MiB |
| shared expert | 4.72 M | 2.65 MiB |
| **per layer** | | **≈43.4 MiB** |
| **× 78 layers** | | **≈3.3 GiB per target forward** |

At the measured 6.76 TB/s that is **0.49 ms** of pure weight streaming per target forward
— the hard memory-roofline floor for one decode step at C1. Measured decode is 447 tok/s
at accept-length 4, i.e. ~112 target forwards/s = 8.9 ms each. **We are at ~5.5% of the
memory roofline at C1.** The gap is launch count, collectives (19.6%, of which 47% is
arrival skew), and M-tile waste — not bandwidth, and definitely not tensor throughput.

Per-expert M at C1: 8 experts share ~4 tokens, so **M ≈ 0.5 per expert**, padded to the
128-row NVFP4 tile. That is the 19.4% of GPU time in `bmm_E2m1_*`. The single highest-
leverage change for that family is anything that raises tokens-per-expert: wider spec
decode, larger batch, or a grouped-GEMM layout that packs multiple experts into one
128-row tile. `[inferred]`

---

## 7. A note on the `nvjet_sm100_*` names

Our top kernel is `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`. `strings` on
`libcublasLt.so.13` (CUDA 13.x, in the NotSglang venv) shows the name is built at runtime
from a printf format `[measured]`:

```
nvjet_sm%d_%s_%dx%d_%dx%d_%dx%d%s_%c_%s%s%s%s%s%s%s%s_%c%c%c
nvjet_sm90_%s_%dx%d_%dx%d_%dx%d_%c_%s%s%s%s%s%s%s%s%s%s_%c%c%c
```

So the decomposition of our kernel is `sm=100`, `kind="tst"`, three `%dx%d` pairs
`64x8`, `64x16`, `4x1`, a char `v`, flag string `bz`, and layout triple `TNT`.
`[verified]` that the fields exist and split that way. The *meaning* of the three pairs —
plausibly (CTA tile M×N, K-tile × pipeline stages, cluster M×N) — is `[inferred]` from the
sm_90 naming convention and **not** sourced. Do not build an argument on the tile
interpretation without confirming it with ncu's `launch__grid_size` and
`launch__shared_mem_per_block` for that kernel.

What is safe to say: `TNT` means the A/B/C layout triple, so cuBLAS is running these in a
transposed-A form, and cuBLAS is choosing an extremely narrow N tile (8 or 16) — which is
the signature of a GEMM whose N is small or whose M is 1-ish, i.e. exactly the decode
regime described in §6.4.

---

## 8. Open questions and what to measure next

1. **GPC count and SM-per-GPC on GB100.** Bounded below by 16 (cluster 16 launches) and
   above by 80/16 = 5 GPCs per die, but not sourced. A `%smid` histogram from a
   cluster-16 kernel would settle it in one run.
2. **Whether the 1-CTA/SM TMEM limit is hardware or a ptxas conservatism.** The occupancy
   API says 1; whether the hardware would actually admit two CTAs each allocating 128
   columns is untested. Test: launch 2 CTAs pinned to one SM via `%smid` spin, each
   allocating 128 columns, and see whether both make progress.
3. **The local/remote L2 split on our workload.** Chips and Cheese measured 21 vs 16.8
   TB/s and 90–100 vs 190–220 ns for atomics. We have never measured which side our
   allreduce lands on. ncu `lts__t_sectors` broken down by partition would say.
4. **cuBLAS nvjet tile decoding.** §7 — resolve with ncu rather than inference.
5. **Whether the `nvjet` kernels use Cluster Launch Control.** If they use a static
   persistent scheduler, they are exactly the imbalanced case CLC exists to fix on a
   device that is also running collectives.
6. **NVFP4 GEMM at small M through the trtllm/flashinfer low-latency path.** My NVFP4
   numbers came from `torch._scaled_mm`; flashinfer exposes `trtllm_low_latency_gemm`,
   `prepare_low_latency_gemm_weights`, and shape-specialised entry points including
   `mm_M1_16_K6144_N256` — literally GLM-5.2's router GEMM (6144→256 experts, M≤16).
   Those could beat the 12 µs plateau I measured. `flashinfer` JIT needs `ninja`, which is
   not installed on this box; that is the blocker.
7. **The `.ws` (weight-stationary) tcgen05 variants at M=32.** They are the only sub-64
   M-tile in the ISA and they exist for `f16` and `f8f6f4` but not for block-scaled FP4.
   Nobody in our stack appears to use them. For a 4-row decode GEMM, M=32 vs M=128 is a
   4× reduction in wasted MMA rows — potentially the argument for keeping an FP8 build.
8. **Unlocking to 1965 MHz.** +23% on everything compute-bound and 0% on everything
   memory-bound, at the cost of comparability with the whole existing benchmark corpus.
   Worth one controlled A/B, not a default.

---

## Sources

Primary, read in full or in the relevant sections:

- CUDA 13.3 Programming Guide, §5.1 Compute Capabilities (Tables 28–33) —
  https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- CUDA 13.0 Programming Guide (archived), §20.8 / §20.9 / §20.10 "Compute Capability
  9.0 / 10.0 / 12.0 — Architecture" —
  https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html
- NVIDIA PTX ISA (CUDA 13.3), §5.1.7 Shared State Space, §9.7.16.5 CTA Pair / Peer CTA,
  §9.7.17 TensorCore 5th Generation Family Instructions (Tensor Memory, addressing,
  allocation, Table 42 matrix shapes), `cp.async.bulk` target notes —
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- NVIDIA Blackwell Tuning Guide 13.3 —
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- NVIDIA Hopper Tuning Guide (for the SM90 occupancy baseline) —
  https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Blackwell Compatibility Guide for CUDA Applications —
  https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html
- NVIDIA Technical Blog, "NVIDIA Blackwell and NVIDIA CUDA 12.9 Introduce Family-Specific
  Architecture Features" —
  https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- NVIDIA Blackwell Datasheet (GB200 NVL72 / GB200 NVL4 / **HGX B200** technical
  specifications table, p. 8) —
  https://nor-tech.com/wp-content/uploads/2026/03/blackwell-datasheet-B200.pdf
- NVIDIA Blackwell Architecture product page (208 B transistors, two reticle-limited
  dies, 10 TB/s die-to-die, unified single GPU) —
  https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- CUTLASS `media/docs/cpp/blackwell_functionality.md` — tcgen05 instruction throughput
  table, narrow-precision types, alignment/layout tables, MmaTileShape tables 4–12 —
  https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md
- CUTLASS `media/docs/cpp/blackwell_cluster_launch_control.md` —
  https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_cluster_launch_control.md
- Chips and Cheese, "Nvidia's B200: Keeping the CUDA Juggernaut Rolling" — measured L1/L2
  latency and bandwidth, cross-partition penalties, atomic throughput, 80 physical / 74
  enabled SMs per die, TMEM organisation —
  https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut

Local primary sources:

- `/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc` — `--version`, `--help`,
  `--list-gpu-arch`, `--list-gpu-code`
- `/home/aman/code/cuda-13.3/nvidia/cu13/bin/ptxas` — `--help` (the `-arch` compatibility
  rules quoted in §3.1), `--list-version`
- `/home/aman/code/cuda-13.3/nvidia/cu13/bin/__nvcc_device_query`
- `/home/aman/code/NotSglang/glm-kernels/CMakeLists.txt` (the `100a-real` comment)
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/benchmark/RESULTS.md` (SM clock provenance, measured tok/s)
- `/home/aman/code/weights/GLM-5.2-NVFP4/config.json`
- `/home/aman/code/NotSglang/.venv/.../nvidia/cu13/lib/libcublasLt.so.13` (`strings`, the
  `nvjet_sm%d_...` format)

Measurements taken on this node, 2026-08-17, GPU 4, SM clock 1597 MHz, driver 595.71.05,
CUDA 13.2 runtime, toolkit 13.3.73, torch 2.11.0+cu130. Programs written to
`/tmp/claude-1000/-home-aman-code/930438ff-5f3c-49e6-a3d9-2663231246c6/scratchpad/`:
`dq.cu` / `dq2.cu` (device attributes), `bw.cu` / `l2.cu` (bandwidth, FP32 FMA peak),
`tmem.cu` (tcgen05 alloc + arch-target matrix), `occ.cu` (occupancy incl. the TMEM cliff),
`clus.cu` / `clus2.cu` (cluster residency), `gemm.py` / `peak.py` / `glmshape.py` /
`nv6.py` (BF16 / FP8 / NVFP4 GEMM sweeps), `nvfp4.py` (launch-overhead floor).
