# The 8-GPU fabric: NVLink5, NVSwitch, NVLS multicast, and collective algorithms

## What this is

Everything about moving bytes between the eight B200s in this node: the physical
NVLink 5 / NVSwitch fabric as the box actually reports it, the NVLS
multicast+reduce hardware path and how it is exposed (CUDA multicast objects,
`multimem` PTX, NCCL NVLS, symmetric memory), the algorithm space for a TP=8
allreduce of a *small* tensor, a line-by-line reading of the two TensorRT-LLM
MNNVL kernels that dominate our profile, the NCCL knobs that matter, and a long
section on **rank arrival skew** — the 47% of our collective time that is not
transfer. Claims are labelled `[verified]` (I read it in a primary source, path
or URL given), `[reported]`, `[inferred]`, `[unverified]`.
Revised 2026-08-17 by an adversarial audit pass; every number below was
re-checked against the cited source, several were wrong, and the corrections are
called out inline as **[AUDIT]**. The multimem PTX/SASS matrix (§2.3) is new and
was produced by assembling probe kernels with the local `ptxas` 13.3.73.

## Bottom line for our system

- **Our allreduces are not bandwidth-bound, they are not even close.** At C1 the
  message is 4 tokens x 6144 x bf16 = **48 KiB per rank**. The NVLink data
  movement floor for a TP8 one-shot allreduce of that is **~0.4 µs**; the
  measured per-rank cost is **~15.4 µs**. >95% of the kernel is fixed overhead
  plus waiting. Tuning NCCL bandwidth knobs cannot help. [inferred from measured
  profile + `nvidia-smi` link rates]
- **We run `oneshotAllreduceFusionKernel` because `numTokens <= 10`.** FlashInfer
  picks one-shot iff `num_tokens * hidden * tp_size * elem_size <= 1 MiB`; at
  `hidden=6144, tp=8, bf16` that is exactly **numTokens <= 10**. EAGLE 3-1-4
  gives 4 tokens/verify step, so C1 is always one-shot and C64 (256 tokens) is
  always two-shot. [verified: `flashinfer/comm/trtllm_mnnvl_ar.py:44,51`]
- **The one-shot kernel asks for only 32 CTAs x 96 threads on a 148-SM GPU** for
  our shape (derived below from `adjustGridConfig`, and re-verified line by line
  against the source in this audit). The AR kernel itself can therefore never
  occupy more than ~32 SMs; 116 SMs' worth of issue slots are *available* for the
  whole ~15 µs including the skew window. **[AUDIT]** The previous wording ("116
  SMs sit idle") overclaims: the same trace shows kernel-time shares summing to
  135% of the busy window, i.e. there is real cross-stream concurrency, so some
  of those SMs are occupied by other kernels. The correct claim is that the
  collective *cannot* be the thing saturating the machine, so the lever is
  overlap and fusion, not NCCL tuning. [inferred from verified source + ledger]
- **NCCL NVLS is switched OFF on this box by default.** SGLang sets
  `NCCL_NVLS_ENABLE=int(enable_nccl_nvls or enable_symm_mem)` and both default
  `False`, and it also sets `NCCL_CUMEM_ENABLE=0`, which NVLS requires.
  [verified: `entrypoints/engine.py:1508-1517`, `server_args.py:1885-1897`]
  This mostly does not matter (the mnnvl kernels bypass NCCL) but it does mean
  any NCCL fallback path is running Ring/Tree, not NVLS.
- **NCCL 2.28 halved the Blackwell CTA cap from 32 to 16** — good for us (less
  SM theft) and reversible with `NCCL_MIN_CTAS=32 NCCL_MAX_CTAS=32`. NCCL 2.28
  also adds `NCCL_CTA_POLICY_ZERO` (copy-engine collectives, zero SMs) for
  **alltoall / scatter / gather / allgather** within an (MN)NVL domain — note
  *not* allreduce. [verified: NCCL v2.28.3-1 release notes; `nccl.h:64-66`
  locally] **[AUDIT]** It is set through `ncclConfig_t.CTAPolicy`, not through
  the `NCCL_CTA_POLICY` env var, whose documented tokens are only
  `DEFAULT`/`EFFICIENCY`.
- **The two NUMA domains do not touch NVLink traffic, but they very likely touch
  the skew.** GPUs 0-3 are on NUMA 0 (cores 0-55,112-167), 4-7 on NUMA 1. All
  GPU-GPU traffic is NV18 through NVSwitch and never crosses UPI. But the host
  threads that launch work, and TP-rank-0's extra scheduler/IO duties, live on
  NUMA 0 — and our straggler ranks are 0, then 2 and 3, all NUMA 0.
  [verified topology; skew attribution is a hypothesis, see §6]
- **Persistent skew is NOT expert-routing imbalance here.** We run `--tp 8` with
  `ep_size=1`, so the MoE is tensor-sharded: every rank computes every activated
  expert on a 1/8 slice of `moe_intermediate_size`. Per-rank work is identical
  by construction. The standard MoE-straggler explanation is *excluded* for this
  configuration, which is itself the finding. [inferred from config; see §6.2]
- **TileRT's 500 tok/s comes from fusing the allreduce into the producing GEMM.**
  `third_party/TileRT` ships `down_allreduce`, `unproj_o_allreduce` and
  `expert_down_allreduce` ops — the allreduce is the epilogue of the down/o
  projection, not a separate kernel. That removes the launch, removes the
  round-trip through HBM, and lets the reduction start on the first finished
  tile. [verified: `tilert/models/glm_5/_dsa_v32/ops/*_allreduce.py`]

---

## 1. The physical fabric

### 1.1 What this box reports

`nvidia-smi topo -m` on this node [verified, run 2026-08-17]:

```
     GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity      NUMA
GPU0  X    NV18  NV18  NV18  NV18  NV18  NV18  NV18  0-55,112-167      0
GPU1 NV18   X    NV18  NV18  NV18  NV18  NV18  NV18  0-55,112-167      0
GPU2 NV18  NV18   X    NV18  NV18  NV18  NV18  NV18  0-55,112-167      0
GPU3 NV18  NV18  NV18   X    NV18  NV18  NV18  NV18  0-55,112-167      0
GPU4 NV18  NV18  NV18  NV18   X    NV18  NV18  NV18  56-111,168-223    1
GPU5 NV18  NV18  NV18  NV18  NV18   X    NV18  NV18  56-111,168-223    1
GPU6 NV18  NV18  NV18  NV18  NV18  NV18   X    NV18  56-111,168-223    1
GPU7 NV18  NV18  NV18  NV18  NV18  NV18  NV18   X    56-111,168-223    1
```

`nvidia-smi nvlink -s`: every GPU has **Link 0..17, each 53.125 GB/s**
[verified]. `lscpu`: 2x Intel Xeon Platinum 8581C, 56 cores/socket, 2 NUMA
nodes [verified].

**"NV18" does not mean 18 dedicated links to each peer.** It is the count of
NVLink lanes the GPU has *into the fabric*; because every GPU reaches every other
through NVSwitch, `nvidia-smi` prints the same NV18 in every off-diagonal cell.
There are 8x18 = 144 GPU-side links total, not 8x7x18. [inferred; consistent
with the switch port count below]

### 1.2 NVLink 5 link math

| quantity | value | source |
|---|---|---|
| Links per B200 | 18 | nvidia.com/en-us/data-center/nvlink [verified] |
| Per-link rate reported by driver | 53.125 GB/s | `nvidia-smi nvlink -s` [verified] |
| Aggregate raw, one direction | 18 x 53.125 = **956.25 GB/s** | arithmetic |
| Per-GPU bandwidth, NVIDIA spec | **1,800 GB/s** (bidirectional) | nvidia.com NVLink page [verified] |
| Payload rate, one direction | **900 GB/s** | 1800/2 |
| Payload / raw | 900 / 956.25 = 0.9412 = 16/17 | arithmetic |
| Per-link payload, one direction | 50 GB/s | 900/18 |

The 16/17 ratio is exactly the same one H100/NVLink4 shows (26.5625 GB/s raw per
link, 25 GB/s payload), so it is the NVLink framing overhead, not a Blackwell
quirk. [inferred] The specific lane organisation (how 53.125 GB/s decomposes
into lanes and baud rate) — **not sourced**; do not repeat a lane count you
cannot cite.

HGX B200 board-level, from nvidia.com/en-us/data-center/hgx [verified]:
8 Blackwell SXM, **1.4 TB HBM3e**, **14.4 TB/s** total NVLink bandwidth
(= 8 x 1.8 TB/s), 1.8 TB/s GPU-to-GPU.

### 1.3 NVSwitch on an HGX B200 baseboard

- The HGX B200 baseboard carries **two NVSwitch ASICs**, with **9 of each GPU's
  18 links going to each switch**. [reported: ServeTheHome / FiberMall coverage
  of the HGX B200 baseboard; I could not get this out of an NVIDIA primary doc]
- NVLink Switch (Blackwell generation) is quoted at **72 NVLink 5 ports** per
  chip and **14.4 TB/s** non-blocking switching. [reported, same class of
  sources]
- Consequence for us: **every pair is one switch hop, and the fabric is
  non-blocking.** There is no "near" and "far" GPU. Bisection is the full
  8 x 900 GB/s = 7.2 TB/s each way. A single pair doing a plain P2P copy can in
  principle drive the full 900 GB/s one-way because all 18 of the source's links
  can be striped through the switches to the same destination. [inferred from
  non-blocking crossbar + 18 links/GPU]

### 1.4 Peak vs achievable — the only measured numbers I trust

Hazy Research measured, on **8x B200**, the per-GPU NVLink read bandwidth
achievable by three different data paths [verified:
hazyresearch.stanford.edu/blog/2025-09-22-pgl]:

| path | GB/s | % of 900 GB/s |
|---|---:|---:|
| theoretical peak | 900 | 100% |
| copy engine (`cudaMemcpyPeerAsync` class) | 726 | 81% |
| intra-SM TMA (`cp.async.bulk` to peer) | 669 | 74% |
| register ops (`ld.global` / `multimem.ld_reduce`) | 541 | 60% |

And the number that matters most for kernel design, from the same source:
**"only 8-16 SMs out of 148 are needed to nearly saturate NVLink bandwidth."**
[verified]

Two corollaries:
1. A collective that wants peak bytes/s should use the copy engine (which is
   what NCCL's new CE collectives do, §5.3) or TMA, not scalar/vector register
   loads.
2. A collective that uses 32+ SMs on a B200 is spending SMs on latency hiding or
   on fusion, not on bandwidth.

### 1.5 Latency — the number that dominates at batch 1

I could **not** source an authoritative NVLink 5 point-to-point latency figure
from NVIDIA. Treat any specific nanosecond number you see repeated on blogs as
unverified. What can be said:

- nccl-tests `all_reduce_perf -b 4 -e 8G` with `NCCL_ALGO=NVLS` on 8 GPUs shows
  **~24 µs for an 8-byte message** [reported, from NVIDIA/nccl issue #2077
  transcript]. That is a *floor for NCCL*, not for the fabric: it includes NCCL
  kernel launch, the LL/NVLS protocol handshake and the test harness.
- Our own measurement is the better anchor: the trtllm MNNVL one-shot kernel,
  48 KiB per rank, TP8, **~8.2 µs of "transfer" (= duration of the last-arriving
  rank) and ~7.2 µs of waiting**, per rank per allreduce. [verified: our nsys
  decomposition, `personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`]
- The pure data-movement component of that 8.2 µs is ~0.4 µs (below). So the
  fabric's small-message latency plus the kernel's fixed cost is on the order of
  **6-8 µs** on this software stack. [inferred]

**Data-movement floor for our C1 allreduce** [inferred, arithmetic]:

```
message per rank            = 4 tokens x 6144 x 2 B      = 49,152 B = 48 KiB
one-shot, multicast publish:
  NVLink egress per rank    = 48 KiB                     (switch replicates)
  NVLink ingress per rank   = 7 x 48 KiB = 336 KiB
  ingress time @ 900 GB/s   = 336 KiB / 900e9            = 0.37 µs
local reduce read           = 8 x 48 KiB = 384 KiB from HBM
  @ ~7.5 TB/s STREAM-triad  = 384 KiB / 7.5e12           = 0.05 µs
--------------------------------------------------------------------
floor                                                    ~= 0.42 µs
observed                                                 ~= 15.4 µs
```

(The 7.48 TB/s STREAM triad figure for B200 is [verified] from the Blackwell
microbenchmark paper, arXiv 2512.02189 overview.)

So **97% of the allreduce cost at C1 is not data movement.** Everything in §6 is
about the other 97%.

### 1.6 The two NUMA domains — does it matter?

**For GPU-to-GPU data: no.** Every pair is NV18 through NVSwitch. Nothing in a
TP allreduce crosses the UPI link between the two Xeon sockets. `nvidia-smi topo`
would print `SYS` if it did; it prints `NV18` for all 28 pairs. [verified]

**For everything else: yes, and this is under-appreciated.**

1. **Launch path.** Each rank's host process issues `cuLaunchKernel` /
   `cudaGraphLaunch` from a thread on its own NUMA node. Ranks 0-3 share the 56
   physical cores of NUMA 0 with each other *and* with SGLang's tokenizer
   manager, detokenizer, HTTP server and ZMQ threads. Ranks 4-7 share NUMA 1 with
   nothing but each other. That is an asymmetry of exactly the shape our skew
   data shows (stragglers 0, 2, 3 — all NUMA 0). [inferred; testable, see §6.4]
2. **Pinned-memory locality.** Host-side staging buffers (sampling outputs,
   logits copies, `cudaHostAlloc` regions) allocated on the wrong node cost an
   extra UPI hop per DMA. `NCCL_PROXY_CPUSET` exists specifically to pin NCCL's
   proxy threads. [verified: env var present in our `libnccl.so.2`]
3. **Interrupt / PCIe locality.** GPU 0-3's PCIe root complexes hang off socket
   0. Any host-driven copy or doorbell write from a socket-1 core to a socket-0
   GPU crosses UPI.

Practical rule for us: **bind rank i's process to its GPU's NUMA node**
(`numactl --cpunodebind=$((i/4)) --membind=$((i/4))`) and keep the non-model
SGLang processes (tokenizer/detokenizer/HTTP) off the cores used by ranks 0-3.
`NCCL_IGNORE_CPU_AFFINITY` exists to *undo* NCCL's automatic affinity setting if
it fights your binding [verified: env var present].

---

## 2. NVLS: hardware multicast and in-switch reduction

### 2.1 What it is

NVSwitch (NVLink 3 switches onward, i.e. Hopper and Blackwell) contains
**SHARP** engines — NVIDIA calls the NCCL-visible feature **NVLS** / "NVLink
SHARP". Two hardware capabilities:

- **Multicast**: a store to a *multicast address* is replicated by the switch to
  every GPU that has bound memory to that multicast object. One egress packet
  becomes N ingress packets inside the switch.
- **In-network reduction**: a load-reduce from a multicast address makes the
  switch gather the corresponding line from every bound GPU, **sum them in the
  switch ASIC**, and return one value.

[verified conceptually: nvidia.com NVLink page — "Each NVLink Switch has engines
for NVIDIA SHARP for in-network reductions and multicast acceleration"; and the
NCCL 2.27 blog]

The bandwidth argument: a naive 8-rank allreduce moves 7S bytes into each GPU.
With switch-side reduction, the GPU issues one load-reduce and receives S bytes
of *already reduced* data. The 7x fan-in happens in the switch, on links that
were going to be idle anyway.

### 2.2 The CUDA API, verified on this box

Multicast is a first-class CUDA VMM object (`cuda.h`, CUDA 13.3,
`/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda.h`) [verified]:

```c
typedef struct CUmulticastObjectProp_st {
    unsigned int numDevices;      /* devices in the multicast team          */
    size_t       size;            /* max memory bound per device            */
    unsigned long long handleTypes; /* CUmemAllocationHandleType bitmask    */
    unsigned long long flags;      /* must be zero                          */
} CUmulticastObjectProp;

CUresult cuMulticastCreate(CUmemGenericAllocationHandle *mcHandle,
                           const CUmulticastObjectProp *prop);       /* :14233 */
CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle,
                              CUdevice dev);                          /* :14267 */
CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset,
                            CUmemGenericAllocationHandle memHandle, size_t memOffset,
                            size_t size, unsigned long long flags);   /* :14325 */
CUresult cuMulticastBindAddr(...);  CUresult cuMulticastUnbind(...);
CUresult cuMulticastGetGranularity(size_t*, const CUmulticastObjectProp*,
                                   CUmulticastGranularity_flags);
```

Sequence: create the object with `numDevices` set -> `cuMulticastAddDevice` for
**all** devices (must complete before any bind) -> each device allocates normal
`cuMemCreate` memory and `cuMulticastBindMem`s it -> `cuMemMap` +
`cuMemSetAccess` the mc handle to get a **multicast VA** distinct from the
unicast VA. Support is queried with
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`.

Measured on GPU 0 of this box [verified, via `cuda.bindings.driver`]:

| query | value |
|---|---|
| `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` | **1** |
| `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` | **1** |
| `CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH` | 1 |
| `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` | **148** |
| `cuMulticastGetGranularity` MINIMUM, numDevices=2/4/8 | **2,097,152 B (2 MiB)** |
| `cuMulticastGetGranularity` RECOMMENDED, numDevices=2/4/8 | **2,097,152 B (2 MiB)** |
| `cuMemGetAllocationGranularity` MIN/RECOMMENDED | 2,097,152 B |

**2 MiB is the multicast quantum.** Every multicast-backed buffer is rounded up
to 2 MiB; FlashInfer's workspace sizer does exactly this
(`ceil_align(size, 1 << 21)` then align again to the mc granularity)
[verified: `sglang/srt/layers/flashinfer_comm_fusion.py`,
`_flashinfer_trtllm_workspace_allocation_sizes`].

Also verified from `nvidia-smi -q`: every GPU reports `Fabric State: Completed`,
`Status: Success`, `CliqueId: 0` — one clique, fabric trained, multicast usable.
`nvidia-fabricmanager` shows `inactive` under systemd on this host but the fabric
is trained, so FM is being run some other way (container/other unit). No
`/dev/nvidia-caps-imex-channels` exists — expected, IMEX is multi-node only.

### 2.3 The PTX

Three instructions, all taking a **multicast** address in `.global`:

| instruction | what it does |
|---|---|
| `multimem.st` | broadcast a store to all bound GPUs |
| `multimem.red` | reduce-into-all (no return value) |
| `multimem.ld_reduce` | gather+reduce in the switch, return one value |

I could not get the PTX ISA section 9.7.9.15 grammar to render through
WebFetch — **the qualifier/type matrix is not sourced from the ISA doc**. But
the exact forms in production use are verified locally:

`torch/include/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h`
[verified]:
```
multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];
multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v2.bf16x2 {%0,%1}, [%2];
multimem.ld_reduce.relaxed.sys.global.add        .f32        %0, [%1];
multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};
multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};
multimem.st.relaxed.sys.global.f32    [%0], %1;
```

`flashinfer/data/include/flashinfer/comm/mixed_comm.cuh` [verified]:
```
multimem.red.release.sys.global.add.u32 [%0], %1;                      // signal
multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2  {...}, [%4];
multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {...}, [%4];
multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};
```

And in our own tree, `sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py`
does a Triton-inlined `multimem.st.relaxed.sys.global.v4.f32` all-gather for the
LM head, 1024 threads/block, 8 bf16 per thread, grid clamped to **4..32 blocks**
[verified].

Two design facts worth internalising:
- `.acc::f32` on a `bf16x2`/`f16x2` ld_reduce means the **switch accumulates in
  fp32** and returns bf16/fp16. That is why NVLS allreduce is not bit-identical
  to a local fp32 tree reduce but is *better* than naive bf16 accumulation.
- `multimem.red` with `.add.u32` is the canonical **multicast barrier/signal**:
  one instruction increments a counter on all peers.

### 2.4 Where NVLS wins, and where it doesn't

- NVLS's advantage is **bytes**, and bytes only matter above the latency floor.
  For an 8-rank allreduce it turns 7S of ingress into ~1.75S (two-shot with
  `RSxLDMC_AGxSTMC`) or 1S of egress (one-shot broadcast). Below ~100 KiB the
  fixed cost swamps that.
- NVLS register-path bandwidth is the *worst* of the three data paths (541 GB/s,
  60%) per the Hazy measurement. NVLS wins on *volume moved*, not on link
  utilisation.
- Reported field data: on an NVLink testbed with NCCL 2.29.7, NVLS was the
  default at all sizes but **Ring with 32 channels beat NVLS by 5-27% in the
  4-128 MiB range**, with NVLS only clearly ahead at >=256 MiB. [reported:
  NVIDIA/nccl issue #2077 thread] Do not assume NVLS is free.
- NVIDIA's own claim for NCCL 2.27 symmetric-memory kernels: **up to 9x lower
  latency at small sizes**, and on an NVL8 domain (= our box) **up to 2.5x for
  small-to-medium messages**. [reported: developer.nvidia.com NCCL 2.27 blog]

### 2.5 Slot limits, Fabric Manager, IMEX

- **NVSwitch has a hard limit of 128 multicast slots.** Exceeding it makes
  `cuMulticastBindMem` fail with `CUDA error 2 (out of memory)`. NCCL **2.28.9**
  (our version) degrades gracefully; **2.29.7 regressed** into a fatal error.
  [reported/verified: NVIDIA/nccl issue #2077]
  Practical: every NCCL sub-communicator that enables NVLS, every FlashInfer
  workspace, every torch symmetric-memory group consumes slots. If you create
  many TP/EP/DP sub-groups you can run out.
- **Fabric Manager** configures NVSwitch routing and port maps and trains the
  links; it is required for any NVSwitch system. [verified:
  docs.nvidia.com fabric-manager-user-guide] It does not appear to document the
  multicast group count.
- **IMEX** (Internode Memory Exchange) is **multi-node only** — it establishes
  the VA-to-FA mapping across OS domains for `CU_MEM_HANDLE_TYPE_FABRIC`.
  "NVLink multi-node jobs will fail if the IMEX service is not properly
  initialized." For a single HGX B200 node it is not needed. [verified:
  docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html]
  Our box has no `/dev/nvidia-caps-imex-channels`, consistent with that.
  FlashInfer still *prefers* `CU_MEM_HANDLE_TYPE_FABRIC` when
  `is_mnnvl_fabric_supported()` and falls back to
  `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` otherwise [verified:
  `flashinfer_comm_fusion.py:_make_flashinfer_workspace_allocation_prop`].

### 2.6 Interaction with symmetric memory

"Symmetric memory" = every rank allocates the *same size* buffer at a
*rank-consistent offset*, so a peer address can be computed rather than looked
up. Three stacks on this box, all present:

| stack | API | notes |
|---|---|---|
| NCCL | `ncclCommWindowRegister(comm, buff, size, &win, NCCL_WIN_COLL_SYMMETRIC)` | buffer must come from the CUDA VMM API; collective+blocking; `NCCL_WIN_ENABLE=0` disables [reported: NCCL 2.27 blog] |
| PyTorch | `_SymmetricMemory.empty_strided_p2p(...)`, `torch.ops.symm_mem.{one_shot_all_reduce, multimem_all_reduce, multimem_all_gather_out}` | alpha; sum-only [verified locally in torch 2.11.0+cu130] |
| FlashInfer / TRT-LLM | `MNNVLAllReduceFusionWorkspace`, `SymmDeviceMemory` | 3 rotating Lamport buffers, mc + uc pointers [verified] |

The key coupling: **NVLS requires cuMem/VMM allocation.** NCCL's NVLS transport
allocates through `cuMemCreate` + `cuMulticastBindMem`; if `NCCL_CUMEM_ENABLE=0`
you cannot have NVLS. SGLang forces `NCCL_CUMEM_ENABLE=0` unless
`--enable-symm-mem` [verified: `engine.py:1508-1509`]. So on our box:
**NCCL = no cuMem = no NVLS.** The mnnvl FlashInfer path builds its own
multicast objects directly and is unaffected.

`NCCL_GRAPH_MIXING_SUPPORT=0` is set by SGLang only when `dcp_size > 1`
[verified: `engine.py:1518-1522`], with the comment that it "can help improve
performance for symmetric kernels" (referencing nccl-tests issue #333). Worth
testing unconditionally if we ever enable NCCL symmetric memory.

---

## 3. Collective algorithms for a TP=8 allreduce of a small tensor

### 3.1 The families

Let `S` = bytes of the tensor, `n = 8` ranks, `B` = per-GPU per-direction link
bandwidth, `L` = one "round trip" of end-to-end latency (store visible on peer +
peer's poll notices).

| algorithm | rounds (latency terms) | ingress bytes per rank | notes |
|---|---:|---:|---|
| **Ring allreduce** | `2(n-1) = 14` | `2(n-1)/n · S = 1.75 S` | bandwidth-optimal, latency-terrible |
| **Tree (double-binary)** | `2·log2(n) = 6` | ~`2S` | NCCL's small-message inter-node algo |
| **One-shot (all-gather + local reduce)** | **1** | `(n-1)S = 7S` | every rank gets all shards, reduces locally |
| **Two-shot (reduce-scatter + all-gather)** | **2** | `2(n-1)/n · S = 1.75 S` | same volume as ring, 2 rounds not 14 |
| **NVLS one-shot** (`AGxLLMC_R`) | 1 | `7S` ingress, **`S` egress** | switch replicates the broadcast |
| **NVLS two-shot** (`RSxLDMC_AGxSTMC`) | 2 | `1.75 S` | switch does the sum |

Ring's 14 sequential dependencies are fatal below ~1 MiB on NVLink: 14 x ~2 µs
= 28 µs of pure latency before any bandwidth argument starts. Tree is better
(6 rounds) but still loses to a 1-round algorithm inside a single NVLink domain
where every rank can reach every other rank directly. **Inside NVL8, ring and
tree are only interesting when S is large enough that `S/B` dominates `L`.**

### 3.2 Why one-shot wins below a threshold, and where the threshold is

```
t_oneshot = L + 7·S/B
t_twoshot = 2·L + 1.75·S/B
crossover:  L = 5.25·S/B    =>    S* = L·B / 5.25
```

With `B = 900 GB/s`:

| assumed L | S* (per-rank message) |
|---:|---:|
| 0.5 µs | 86 KB |
| 1.0 µs | 171 KB |
| 2.0 µs | 343 KB |
| 4.0 µs | 686 KB |

FlashInfer's shipped MNNVL threshold is `S* = 1 MiB / tp_size = 128 KiB`
[verified], which back-solves to an **effective marginal `L ≈ 0.75 µs`**
[inferred]. That is much smaller than the ~8 µs the kernel actually takes,
because most of that 8 µs is a *fixed* cost paid by both algorithms (launch,
Lamport buffer clear, flag update) and cancels out of the comparison.

Note the two-shot algorithm also *scales better with n*: its volume is
`2(n-1)/n · S -> 2S`, while one-shot's is `(n-1)S -> nS`. At n=8 the ratio is
already 4x, which is why one-shot's window shrinks fast as world size grows —
visible directly in FlashInfer's own table (§3.3).

### 3.3 Measured/tuned crossovers on B200, TP8

Three independent tuning tables in our tree, all measured on B200. They do not
agree, because they are three different kernels.

**(a) FlashInfer MNNVL (what we run)** — `trtllm_mnnvl_ar.py:44,51` [verified]:
```python
MNNVL_ONE_SHOT_THRESHOLD = 64 * 1024 * 8 * 2          # 1,048,576 B = 1 MiB
if num_tokens * hidden_dim * tp_size * elem_size <= MNNVL_ONE_SHOT_THRESHOLD:
    return ONESHOT
else:
    return TWOSHOT
```
For GLM-5.2 (`hidden=6144`, bf16, tp=8): **one-shot iff `num_tokens <= 10`**
(10 x 6144 x 8 x 2 = 983,040 <= 1,048,576; 11 gives 1,081,344 > threshold).
Per-rank message at the boundary: 10 x 12,288 = **120 KiB**.

**(b) FlashInfer TRTLLM/IPC-Lamport** — `trtllm_ar.py:968` [verified]:
```python
# Heuristics based on all configs of trtllm_allreduce_fusion on B200.
_use_oneshot_heuristics = {2: 512, 4: 64, 8: 42}   # comm_size in MiB
comm_size_mb = token_num * hidden_dim * 2 * world_size * dtype.itemsize / 1MiB
```
=> per-rank message threshold `= 42 MiB / (2·8) =` **2.625 MiB** at TP8
(TP4: 8 MiB, TP2: 128 MiB). Also `kOneShotMaxToken = 128` in
`trtllm_allreduce_fusion.cuh:32` as a hard cap [verified].

**(c) SGLang's own JIT custom all-reduce v2** —
`distributed/device_communicators/configs/custom_all_reduce_v2.py`, "tuned on
B200 (148 SMs)" [verified]:

| world | context | 1shot_push | 1shot_pull | mc 2shot_pull range | 2shot_pull |
|---:|---|---:|---:|---|---:|
| 8 | CUDA graph | <= 0.500 MiB | <= 0.500 MiB | 8 MiB .. 128 MiB | <= 128 MiB |
| 8 | eager | <= 0.750 MiB | <= 0.750 MiB | 0 .. 128 MiB | <= 128 MiB |
| 4 | CUDA graph | <= 2.250 MiB | <= 2.250 MiB | (disabled) | <= 128 MiB |
| 2 | CUDA graph | <= 8.000 MiB | <= 32.00 MiB | (disabled) | <= 128 MiB |

with `num_push_blocks = 148` (all SMs), `num_pull_blocks = 96`, and
`num_mc_blocks = 32` at world 8. Above these it falls back to NCCL.

**The spread between (a) 120 KiB and (c) 512 KiB for the same hardware and world
size is a real, unexplained disagreement** and is worth one afternoon of
measurement — see open questions. It is plausible that (a) is conservative
because its one-shot workspace is `num_tokens·hidden·tp·elem` (it scales with
world size) while the two-shot workspace does not, so a lower threshold saves
memory.

---

## 4. The kernels we actually run

Our profile's top-2 kernel is `tllm_mnnvl_allreduce::oneshotAllreduceFusionKernel`
at **8.2% of all GPU time (783 ms)**, plus
`trtllm_mnnvl_allreduce::twoshotAllreduceKernel` at **4.3% (407 ms)**. Both come
from `flashinfer/data/include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh`
(FlashInfer 0.6.15.post1, JIT-cached under `~/.cache/flashinfer/0.6.15.post1/100a/`).

Path: `--flashinfer-allreduce-fusion-backend auto` resolves to **`mnnvl` on
SM100** [verified: `flashinfer_comm_fusion.py:_resolve_backend`, and the
resolved-args table in `personal_docs/glm-5.2/glm-5.2-optimization-log.md`].
Note the kernel name says "Fusion" even when `rmsNormFusion=false`; seeing this
kernel in the trace does **not** mean RMSNorm fusion is on.

### 4.1 `oneshotAllreduceFusionKernel` — anatomy

Signature [verified, line 915]:
```cpp
template <uint8_t WorldSize, typename T, bool RMSNormFusion = false,
          QuantType QType = QuantType::kNone, typename PackedType = float4>
__global__ void __launch_bounds__(1024)
oneshotAllreduceFusionKernel(AllReduceKernelParams<T> params)
```

Phases:

1. `cudaGridDependencySynchronize()` — **PDL**: the kernel starts issuing before
   the previous kernel has fully drained. SGLang sets `TRTLLM_ENABLE_PDL=1`
   [verified: `engine.py`].
2. **Publish.** Each thread loads its 16-byte `float4` (8 bf16) of the local
   shard, runs `sanitizeLamportPayload` (any word equal to the dirty sentinel
   `0x80000000` = fp32 negative zero is nudged), and stores it through the
   **multicast pointer**:
   ```cpp
   stagePtrMcast[token*tokenDim*WorldSize + rank*tokenDim][packedIdx] = val.packed;
   ```
   One store; the switch replicates it into all 8 ranks' staging buffers.
3. `flag.ctaArrive()` — a named CTA barrier
   (`barrier.cta.sync 1` / `barrier.cta.arrive 1`) then, on sm_100+,
   `red.async.release.global.gpu.add.u32 [flag], 1;` [verified, line ~340].
4. `clearDirtyLamportBuf(...)` — zero out the *previous* generation's buffer
   (3-deep rotation, `NUM_LAMPORT_BUFFERS = 3`, index advances `(cur+1)%3`).
   This is real work that runs *while* waiting for peers, which is the point.
5. **Wait = spin on the data itself.** `waitOneshotRemoteRanks` polls all 8 slots
   with `ld.volatile.global.v4.u32` and re-reads until **no word equals the
   dirty sentinel**. There is no separate barrier: arrival is detected by the
   payload no longer being negative-zero. `isLamportDirty` checks every 32-bit
   word because "ptx memory model only guarantees atomicity for 64-bit
   granularity". [verified]
6. **Reduce**, in a **compile-time-fixed rank order** starting at the local rank
   (`reduceOneshotDeterministic`, `RUN_ONESHOT_LOCAL_RANK(0..7)` switch), fp32
   accumulator. Comment: "Fully deterministic: every rank uses the exact same
   reduction order." That is what makes TP8 bitwise-reproducible.
7. `cudaTriggerProgrammaticLaunchCompletion()` — lets the *next* kernel start.
8. Optional fusion, in this order: `+= residualIn` -> optional write of the
   pre-norm residual -> **RMSNorm** with `blockReduceSum` and, if
   `cluster.num_blocks() > 1`, a **cluster-wide** sum through
   `cluster.map_shared_rank()` + `cluster.barrier_arrive/wait` -> write
   `outputPtr` -> optional quantisation.

Quantisation patterns supported [verified]: `kFP8` (static scale),
`kFP4` (NVFP4, `cvt_warp_fp16_to_fp4`, requires `tokenDim % 16 == 0`, CUDA>=12.8,
fp16/bf16 input only), `kDynamicFP8` (per-token amax via
`blockReduceMax` + cluster reduce). **Quantisation requires RMSNorm fusion** —
`static_assert(QType == kNone || RMSNormFusion)`.

`weightBias` exists so the same kernel serves Gemma/Qwen-style
`(1 + gamma)` RMSNorm [verified].

**Grid shape for GLM-5.2 at C1** — derived by hand-running `adjustGridConfig`
(line 790) with `numTokens=4, dim=6144, eltsPerThread=8, useCluster=true`,
`kMaxClusterSize=8`, `smCount=148` [inferred from verified source]:

```
threadsNeeded = 6144/8 = 768
step1: clusterSize=8, blockSize=768/8=96
step2: 768 % 8 == 0                 -> clusterSize stays 8, maxDivisible=8
step3: blockSize 96 < 128           -> blockSize=192, clusterSize=4
step4: 4*4=16 <= 148                -> no change
step5: candidate cluster 8 -> blockSize 96 >= 64 and 4*8=32 <= 148
                                    -> clusterSize=8, blockSize=96
final: blockDim=96, clusterDim=(1,8,1), grid=(numTokens=4, 8, 1)
       => 32 CTAs, 3,072 threads, cluster launch
```

**32 CTAs on a 148-SM GPU.** The 8-CTA cluster must be co-resident (one GPC), so
we occupy roughly 32 SMs and leave ~116 idle for the entire 15 µs. That is the
headline structural fact for §6: skew here does not contend for SMs, it wastes
them.

### 4.2 `twoshotAllreduceKernel` — anatomy

[verified, line 1269] `__launch_bounds__(128)`, two Lamport stages
(`SCATTER=0`, `BROADCAST=1`):

1. `destRank = token % WorldSize`, `destTokenOffset = token / WorldSize` — tokens
   are round-robined to owner ranks.
2. **Scatter**: each rank stores its slice of token *t* directly into
   `inputPtrs[destRank]` — a **unicast** peer write, not multicast.
3. `clearDirtyLamportBuf(SCATTER)`.
4. **Reduce**: only the owner rank (`destRank == rank`) runs
   `reduceLamportRanksChunked<WorldSize, kRankChunk>` over the 8 contributions
   for its tokens (chunked at `min(WorldSize,16)` to avoid register spills).
5. **Broadcast**: the owner writes the reduced token through `mcastPtr`
   (`broadcastBufW`) — one multicast store fans out to all 8.
6. Consumers spin on `broadcastBufR` until not dirty, then write `outputPtr`.
7. `flag.waitAndUpdate({...})` rotates the Lamport generation; thread 0 of the
   grid spins until the CTA arrival counter reaches `gridDim/clusterBlocks`, then
   publishes the new `{currentIndex, dirtyIndex, bytesPerBuffer, numStages}`.

Grid: `gridDim.x = numTokens`, `gridDim.y = ceil(threadsNeeded / 128)`. For
`hidden=6144`: `gridDim.y = 768/128 = 6`. At C64 with 4 draft tokens (256
tokens) that is **1,536 CTAs of 128 threads** — full GPU, and it *does* contend
with compute.

Two-shot also has a separate `rmsNormLamport` kernel (line 1342) for the fused
path, since the norm needs the fully-broadcast result.

### 4.3 What FlashInfer's fusion actually folds

`AllReduceFusionPattern` values reachable on the MNNVL backend [verified:
`flashinfer/comm/allreduce.py:1014-1058`]:
`kAllReduce`, `kARResidualRMSNorm`, `kARResidualRMSNormFP8Quant`,
`kARResidualRMSNormFP4Quant`, `kARResidualRMSNormOutFP8Quant`,
`kARResidualRMSNormOutFP4Quant`, `kARResidualRMSNormDynamicFP8Quant`,
`kARResidualRMSNormOutDynamicFP8Quant`.
Anything else must use the `trtllm` backend.

For us the interesting one is
`kARResidualRMSNormFP4Quant`: **allreduce + residual add + RMSNorm + NVFP4
quantisation, one kernel, one pass over the tensor.** With 78 layers x 2
allreduces that eliminates 156 elementwise+norm+quant kernel launches and 156
round trips of the hidden state through HBM per forward. Our ledger records
`norm` at 0.5% and `quant` at 2.4% of GPU time — so the ceiling on this fusion
alone is ~3% of GPU time plus 156 launches, before counting the removed HBM
traffic.

### 4.4 Lamport, and why it costs what it costs

The Lamport scheme trades memory for synchronisation: instead of a flag+fence
handshake it poisons the buffer with a sentinel and lets consumers detect
arrival from the data. Costs:

- **3x buffer footprint** (`NUM_LAMPORT_BUFFERS = 3`), each rounded to 2 MiB, and
  for one-shot the buffer scales with `tp_size`
  (`num_tokens · hidden · tp_size · elem`) [verified].
- **A clear pass every call** over the generation-3-ago buffer.
- **One value is unrepresentable**: fp32 negative zero. Payloads are sanitized
  going in and reduced results are re-sanitized before broadcast.
- **The spin is unbounded and burns SM residency.** A rank that arrives 5 µs
  early spins for 5 µs with its 32 CTAs resident. This is exactly the
  `wait[i] = duration[i] - transfer` term our harness measures.

---

## 5. NCCL tuning that matters (and mostly doesn't, for us)

### 5.1 What is installed and what SGLang sets

- NCCL **2.28.9** (`nccl.h`: `NCCL_MAJOR 2, NCCL_MINOR 28, NCCL_PATCH 9`,
  `NCCL_VERSION_CODE 22809`) [verified].
- SGLang sets, at engine start [verified: `entrypoints/engine.py:1500-1524`]:
  ```
  NCCL_CUMEM_ENABLE      = int(enable_symm_mem)          # default 0
  NCCL_NVLS_ENABLE       = int(enable_nccl_nvls or enable_symm_mem)  # default 0
  NCCL_GRAPH_MIXING_SUPPORT = 0   (only when dcp_size > 1)
  CUDA_DEVICE_MAX_CONNECTIONS = 8
  CUDA_MODULE_LOADING    = AUTO
  TRTLLM_ENABLE_PDL      = 1
  ```
  and, only when `nnodes > 1 and is_mnnvl_fabric_device()`,
  `NCCL_CUMEM_ENABLE=1`, `NCCL_MNNVL_ENABLE=1`.
- There is exactly one place SGLang touches `NCCL_ALGO`:
  `server_args.py:7808` sets `NCCL_ALGO="allreduce:tree"` in a specific branch
  [verified] — worth knowing it exists so you do not chase a phantom.

### 5.2 Algorithms and protocols

Verified from `strings libnccl.so.2` on this box, the algorithm tokens the
library actually contains: **`Ring`, `Tree`, `CollNetDirect`, `CollNetChain`,
`NVLS`, `NVLSTree`, `PAT`** (PAT appears via `NCCL_PAT_ENABLE`), and the
protocol tokens **`Simple`, `LL`, `LL128`**.

`NCCL_ALGO` / `NCCL_PROTO` accept a global value or per-collective syntax
`<collective>:<algo>` [verified: NCCL env docs].

Protocol characteristics [reported: arXiv 2507.04786 "Demystifying NCCL",
Table IV — this is the most complete numeric source I found, but it is a
third-party paper, not NVIDIA]:

| protocol | payload framing | channel buffer | per slot | effective/slot | BW efficiency | per-hop latency |
|---|---|---:|---:|---:|---:|---:|
| Simple | raw + memory fences | 4 MiB | 512 KiB | 512 KiB | ~peak | ~6 µs |
| LL | 4 B data + 4 B flag (8 B atomic) | 256 KiB | 32 KiB | 16 KiB | 25-50% | ~1 µs |
| LL128 | 120 B data + 8 B flag (128 B atomic) | ~4800 KiB | 600 KiB | 562.5 KiB | ~95% | ~2 µs |

`NCCL_STEPS` (pipeline slots per channel) defaults to 8. Crossover Simple vs
LL/LL128 lands empirically around **64 KiB**. Intra-node over NVLink, LL128 is
consistently good across all sizes.

For our 48 KiB TP8 allreduce, if we were on NCCL it would be **LL128 + Tree or
NVLS**, and would be slower than the mnnvl one-shot kernel because it still pays
2 or more dependent rounds.

### 5.3 CTAs, channels, and SM theft

This is the NCCL knob that actually matters for an inference engine.

- `NCCL_MAX_CTAS` / `NCCL_MIN_CTAS` supersede the deprecated
  `NCCL_MAX_NCHANNELS` / `NCCL_MIN_NCHANNELS`. Documented defaults:
  **`minCTAs = 1`, `maxCTAs = 32`**, "up to 32" [verified: NCCL API types doc].
- **NCCL 2.28 decreased the max CTA count from 32 to 16 on Blackwell**,
  described as "SM overhead is decreased by 50%", restorable with
  `NCCL_MIN_CTAS=32 NCCL_MAX_CTAS=32`. [verified: NCCL v2.28.3-1 release notes]
- `NCCL_CTA_POLICY` (new in 2.28), values verified in our local `nccl.h:64-66`:
  ```c
  #define NCCL_CTA_POLICY_DEFAULT     0x00
  #define NCCL_CTA_POLICY_EFFICIENCY  0x01
  #define NCCL_CTA_POLICY_ZERO        0x02
  ```
  Documented meanings [verified: NCCL flags doc]:
  - `DEFAULT` — "automatically adjust resource usage and achieve maximal
    performance."
  - `EFFICIENCY` — "use minimal number of CTAs to achieve the decent
    performance when possible"; recommended when the app "needs better
    compute-communication overlap."
  - `ZERO` — "use zero CTA whenever it can, even when that choice may sacrifice
    some performance." This is the **copy-engine (CE) collectives** path, new in
    2.28, covering alltoall/scatter/gather/allgather within an (MN)NVL domain,
    and it requires buffers registered into symmetric windows.
- The arithmetic of SM theft: an NCCL collective at 16 CTAs on a 148-SM B200
  takes ~11% of the machine for its entire duration. If it runs concurrently
  with a GEMM that wanted all 148 SMs, the GEMM gets 89%. For our fused
  epilogue GEMM (`nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, 12.6% of time) that
  is a direct 11% tax on the largest kernel we have. Hence: **if we ever put
  collectives on a side stream to overlap them, cap CTAs hard.**

### 5.4 Buffers

`NCCL_BUFFSIZE` default **4194304 (4 MiB)** per channel per peer [verified:
NCCL env doc]. `NCCL_LL_BUFFSIZE`, `NCCL_LL128_BUFFSIZE`,
`NCCL_P2P_NVL_CHUNKSIZE`, `NCCL_NVLS_CHUNKSIZE`, `NCCL_NVLSTREE_MAX_CHUNKSIZE`
all exist in our 2.28.9 build [verified: `strings libnccl.so.2`]. For 48 KiB
messages none of these are the binding constraint — the message does not fill
one slot of one channel.

### 5.5 CUDA graph capture of collectives

- All of SGLang's decode path is CUDA-graph captured, which is why the v2
  all-reduce config has separate `graph` and `eager` threshold tables
  [verified].
- NCCL collectives are capturable, but NCCL must know whether graphs may be
  *mixed* with non-captured work on the same communicator;
  `NCCL_GRAPH_MIXING_SUPPORT=1` (default) adds serialisation to make that safe.
  Setting it to 0 is a measurable win for symmetric kernels [reported:
  nccl-tests issue #333, cited in SGLang's own comment].
- FlashInfer's MNNVL setup deliberately **broadcasts workspace handles over the
  GLOO cpu_group, not NCCL**, precisely "to avoid NCCL collectives that
  interfere with CUDA graph capture" [verified:
  `flashinfer_comm_fusion.py:_FixedTorchDistBackend`]. If you write your own
  collective, copy that pattern.
- SGLang's v2 all-reduce exchanges peer pointers *after* capture into a
  device-side `graph_params` table that the captured kernel dereferences at
  replay [verified: module docstring]. This is the general trick for making a
  peer-pointer kernel graph-safe.

### 5.6 NVSHMEM and device-initiated communication

- **NVSHMEM is not installed on this box** (`find / -iname "*nvshmem*"` returns
  nothing outside FlashInfer's Python shims) [verified]. FlashInfer ships
  `comm/nvshmem.py` and `comm/nvshmem_allreduce.py` bindings, so enabling it is a
  packaging problem, not a code problem.
- **NCCL 2.28 added an experimental device API** with three transports: **LSA**
  (CUDA P2P), **Multimem** (hardware multicast via NVLink SHARP), and **GIN**
  (GPU-initiated networking, early). [verified: NCCL v2.28.3-1 release notes]
  Our build has the symbols: `NCCL_GIN_ENABLE`, `NCCL_GIN_TYPE`,
  `NCCL_GIN_NCONTEXTS`, `ncclSymkPickKernel`, `ncclSymkGetKernelPtr`, etc.
  [verified: `strings libnccl.so.2`]
- The NCCL **symmetric kernel** names in our binary map exactly onto §3.1:
  ```
  AllReduce_AGxLL_R              one-shot, LL, no multicast
  AllReduce_AGxLLMC_R            one-shot, LL, MULTICAST broadcast
  AllReduce_RSxLD_AGxST          two-shot, no multicast
  AllReduce_RSxLDMC_AGxSTMC      two-shot, multimem.ld_reduce + multimem.st
  AllReduce_RSxNet_ARxMC_AGxNet  multi-node
  AllGather_{LL,LLMC,ST,STMC}
  ```
  `NCCL_SYM_KERNEL` and `NCCL_SYM_CTAS` let you force one and cap its CTAs
  [verified: env var names present]. This is the closest thing to a
  drop-in replacement for the FlashInfer mnnvl kernel if we ever want NCCL back.

---

## 6. Rank arrival skew — the 47%

### 6.1 What we measured, and exactly what the number means

From `personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` [verified],
20 s steady-state nsys window, all 8 ranks, TP8/NVFP4/EAGLE-3-1-4, concurrency 1:

```
observed 14,097 ms  =  transfer 7,505 ms  +  waiting 6,599 ms
arrival skew: mean 9.2 µs, max 4,897 µs
rank 0 is the last to arrive in 26,929 of 114,171 instances (24%),
then ranks 2 and 3 — a persistent pattern, not jitter.
```

The methodology is in `benchmark/gpubench/analysis/multigpu.py` [verified]:

```
skew     = last_arrival - first_arrival          (kernel START timestamps)
transfer = duration observed by the LAST rank to arrive
wait[i]  = duration[i] - transfer
```

Collectives are matched **by per-rank ordinal**, not by timestamp — the Nth
collective on rank A is the Nth on rank B — which is robust and needs no NCCL
instrumentation.

Derived per-instance figures [inferred, arithmetic on the above]:

| quantity | value |
|---|---:|
| collective instances (per rank) | 114,171 |
| aggregate rank-time | 14,097 ms |
| per-rank per-instance total | 14,097/8/114,171 = **15.4 µs** |
| of which "transfer" | 7,505/8/114,171 = **8.2 µs** |
| of which waiting | 6,599/8/114,171 = **7.2 µs** |
| data-movement floor (§1.5) | ~0.4 µs |
| allreduces per forward (78 layers x 2) | 156 |
| collective cost per forward | 156 x 15.4 µs = **2.4 ms** |
| forwards in the window | 114,171 / 156 ≈ **732** |

At 365 tok/s and ~4.2 accepted tokens per verify step, 732 forwards over the
~8.5 s span is 86 forwards/s = **11.6 ms per forward** — which cross-checks.
So collectives are ~21% of the step, matching the 19.6% family share.

**Read the "transfer" term honestly.** It is *the duration of the
last-arriving rank*, which is `fixed kernel cost + real transfer`, not real
transfer. Since real transfer is ~0.4 µs, essentially all 8.2 µs is fixed cost:
kernel launch/teardown, the Lamport clear pass, the arrival-counter round trip,
the final flag update, and the tail latency of the multicast fan-out. **The
correct decomposition for us is 0.4 µs transfer + ~7.8 µs fixed + 7.2 µs skew.**

### 6.2 Why the usual explanation is excluded here

`multigpu.py`'s own docstring names the prime suspect: "uneven expert routing at
EP8 is the prime suspect for a 256-expert top-8 MoE". **That suspect does not
apply to this run.** The resolved server args are `--tp 8` with no `--ep-size`,
so `ep_size = 1` and the MoE runs tensor-parallel through
`moe_runner_backend=flashinfer_trtllm`: every rank computes every activated
expert against a 1/8 slice of `moe_intermediate_size` (2048/8 = 256).
Per-rank FLOPs are identical *by construction*, independent of routing.
[inferred from `server_args.py` semantics + the resolved-args table in
`personal_docs/glm-5.2/glm-5.2-optimization-log.md`]

Same for the other candidates:
- **Attention**: 64 heads / 8 = 8 heads per rank, uniform. DSA `index_topk=2048`
  is a fixed budget, not data-dependent in size.
- **LM head**: vocab 154,880 / 8 = 19,360 per rank, uniform.
- **Sampling**: replicated on all TP ranks after the logits all-gather.

So the skew is **not** GPU-side work imbalance from the model. It has to come
from outside the model graph. That is a much more actionable conclusion.

### 6.3 Catalogue of causes, ranked by how well they fit our data

**(1) TP-rank-0 host-side duties.** In SGLang every TP rank runs a Scheduler,
but `tp_rank == 0` is gated for extra work in at least six places in
`model_runner.py` and `scheduler.py` [verified: `model_runner.py:410,633,1747,
1755,1821,1854,1880,1896`; `scheduler.py:998,4548`]. Rank 0 owns the ZMQ
connection to the tokenizer/detokenizer managers and output streaming. Every one
of those is CPU work on the critical path *between* forwards, and it delays rank
0's next graph launch. **Fits the data best: rank 0 is the straggler in 24% of
instances.**

**(2) NUMA-0 host contention.** Ranks 0-3 share NUMA node 0 (56 cores) with each
other and with all the auxiliary SGLang processes; ranks 4-7 have NUMA 1 mostly
to themselves. Our straggler order is 0, then 2 and 3 — **all on NUMA 0**. The
mechanism is core oversubscription delaying the launch thread, plus remote
memory for pinned staging buffers. [inferred; strong circumstantial fit]

**(3) CPU-side launch jitter.** With CUDA graphs the per-kernel launch cost is
amortised, but the *graph* launch, the pre-forward Python (batch assembly,
sampling metadata, KV bookkeeping) and any allocator activity are per-step and
per-rank. A single Python GC pause or a `cudaMalloc` on one rank explains the
**max skew of 4,897 µs** — that is 4.9 ms, roughly half a forward, and is
certainly not a GPU phenomenon. [inferred]

**(4) Clock and power variation between GPUs.** Currently *not* a factor:
sampled 5x over ~10 s while serving, **all 8 GPUs pinned at 1597 MHz SM clock**
(max 1965 MHz), 450-482 W against a 1000 W limit, no active throttle reason on
any GPU [verified]. But the *cumulative* SW-power-capping counters differ by
2.1x across GPUs [verified: `nvidia-smi -q -d PERFORMANCE`]:

| GPU | PCI | SW power capping (µs) |
|---:|---|---:|
| 0 | 8F:00.0 | 90,936,966 |
| 1 | 90:00.0 | 172,049,375 |
| 2 | 96:00.0 | 152,435,377 |
| 3 | 97:00.0 | 165,263,802 |
| 4 | C4:00.0 | 113,146,100 |
| 5 | C5:00.0 | 192,932,653 |
| 6 | CB:00.0 | 138,468,962 |
| 7 | CC:00.0 | 91,311,282 |

So this box *has* historically clock-capped unevenly. It correlates poorly with
our straggler ranking (GPU 0 has the *least* capping and is the *most* frequent
straggler), which is further evidence the skew is host-side, not silicon-lottery.
Under a heavier workload than C1 this could flip — re-check at C64.

**(5) Skew self-propagation.** In principle an allreduce resets skew: every rank
leaves at about the same time. In practice the one-shot kernel's exit is not
simultaneous — each rank exits when *its* poll notices all 8 payloads, and the
multicast fan-out plus poll granularity stagger that by hundreds of ns. Over 156
chained collectives per forward that can accumulate if one rank is
systematically slower to observe arrivals. [unverified — would need in-kernel
`%globaltimer` instrumentation to confirm]

**(6) PCIe / host interference.** Weight streaming, NVMe, or NIC DMA hitting a
GPU's root complex. Not observed here (nothing else is running), but this is the
usual cause in production. `nvidia-smi nvlink -e` shows **zero** Rx/Tx errors,
zero link recovery events, symbol errors 0 on GPU 0's links, with FEC error
histogram essentially all in bucket 0 [verified] — the fabric itself is clean.

**(7) Genuinely data-dependent GPU work.** With `ep_size=1` there is none in the
model. The one exception to check is the **EAGLE draft acceptance path**: if
accepted-token counts drive any rank-local control flow, ranks could diverge.
[unverified — worth a look]

### 6.4 How to measure each cause

| hypothesis | measurement | cost |
|---|---|---|
| rank-0 duties | run with the TP group's rank order rotated (make GPU 3 be tp_rank 0) and see if the straggler follows the *rank*, not the *device* | 1 run |
| NUMA-0 contention | `numactl --cpunodebind`/`--membind` per rank + move tokenizer/detokenizer/HTTP to specific cores; re-measure `mean_skew_us` | 1 run |
| launch jitter | compare skew with `--disable-cuda-graph` vs captured; and correlate max-skew events with Python GC (`gc.callbacks`) timestamps | 2 runs |
| clock variation | log `clocks.sm` per GPU at 100 ms during the profile window; correlate per-instance straggler identity with instantaneous clock | free, read-only |
| skew propagation | in-kernel `%globaltimer` written to a debug buffer at `ctaArrive` and at spin-exit; plot arrival vs exit per rank | kernel patch |
| draft divergence | log accepted-token count per rank per step; assert identical | cheap |
| fabric | `nvidia-smi nvlink -e` before/after; watch `Link recovery` and `Effective BER` | free |

The single most informative one is the **rank-rotation test**: it separates
"rank 0 is slow because of what rank 0 *does*" from "GPU 0 is slow because of
where GPU 0 *is*". Both have fixes, but different ones.

### 6.5 Mitigations, from cheapest to hardest

**A. Make the skew not matter (overlap).** This is the right answer, because at
C1 we have 116 idle SMs during the entire 15 µs.
1. **Two-batch overlap (TBO)** — currently blocked: SGLang refuses TBO with DSA
   index-topk sharing (`index_topk_freq=4`), because the TBO op path does not
   propagate topk indices across layers [verified: the error text in the
   ledger]. Unblocking this is ledger item B2 and is the single highest-value
   change identified so far.
2. **Fuse the allreduce into the producing GEMM's epilogue.** This is what
   TileRT does and it is in-tree to read:
   `third_party/TileRT/tilert/models/glm_5/_dsa_v32/ops/down_allreduce.py`,
   `unproj_o_allreduce.py`, `expert_down_allreduce.py`, dispatched as
   `torch.ops.tilert.down_allreduce_op(vec_in, mat_in, mat_scale, x_in, flag,
   vec_out, ...)` with algorithms `general | bf16mma | bf16mma_v2` [verified].
   The `flag` argument is the rotating Lamport generation. Benefits: no separate
   launch, no HBM round trip for the GEMM output, and the publish of tile *k*
   can start while tile *k+1* is still being computed — which *hides* skew
   instead of waiting through it.
3. **FlashInfer allreduce+RMSNorm(+quant) fusion** — one kernel instead of four,
   already supported on our exact path (`kARResidualRMSNormFP4Quant`). Note the
   ledger records that "SBO + allreduce fusion" measured **-12.5% throughput**
   and was reverted because SBO's overlap is gated off for
   `flashinfer_trtllm` — so test allreduce fusion **without** SBO.

**B. Reduce the fixed cost (attack the 7.8 µs, not just the 7.2 µs).**
4. Confirm PDL is actually active on the AR kernels
   (`cudaLaunchAttributeProgrammaticStreamSerialization` is set from
   `params.launchWithPdl` [verified]) — PDL lets the AR start its address
   computation and Lamport clear before the previous kernel drains.
5. Re-examine the one-shot/two-shot threshold. Our C1 shape (4 tokens) is 12x
   below the one-shot threshold; a smaller, latency-tuned one-shot variant with
   fewer CTAs (the cluster is 8 CTAs wide only to get 96-thread blocks) might
   cut the fixed cost.
6. Reduce the *number* of allreduces: 156/forward is a lot. Sequence-parallel /
   reduce-scatter+allgather restructuring (SGLang has `--enable-scattered-sconv`
   doing exactly this shape for conv layers [verified]) halves the volume but
   not the count.

**C. Reduce the skew itself.**
7. NUMA-bind every rank process to its GPU's node; move auxiliary processes off
   NUMA 0's first 56 cores.
8. Rotate or balance rank-0's duties (or accept a dedicated "slow" rank and give
   it *less* model work — not currently expressible in SGLang).
9. Lock clocks (`nvidia-smi -lgc`) to eliminate the power-cap variance as a
   variable during measurement. Not a production fix, but it removes a
   confound.
10. If we ever move to EP for K3/V4: **deterministic expert assignment + EPLB
    with redundant experts** becomes the dominant lever, since routing imbalance
    then does map onto rank imbalance. [reported: DeepSeek EPLB; LMSYS
    large-scale EP blog]

**D. What will NOT help** (state this explicitly so nobody spends a week on it):
`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_BUFFSIZE`, channel counts — we do not run NCCL
on this path at all; and raw NVLink bandwidth, which is 97% unused during the
allreduce.

---

## 7. Ordered actions for this box

1. **Rank-rotation experiment** to attribute the persistent straggler (1 run,
   answers §6.3's top-2 hypotheses). Free.
2. **NUMA-bind ranks and auxiliary processes**; re-measure `mean_skew_us`. Free.
3. **Turn on FlashInfer allreduce+RMSNorm+FP4 fusion alone** (not with SBO).
   Expected: -156 launches/forward, -2.9% of GPU time in norm+quant, plus HBM
   traffic. Low risk.
4. **Unblock TBO** by propagating DSA topk indices across the micro-batch split
   (ledger B2). Highest value, highest effort.
5. **Port TileRT's fused down/o-proj allreduce epilogue** for the two hottest
   allreduce sites. This is the mechanism behind the 500 tok/s target.
6. Only after 1-5: measure `nccl-tests` on an idle box to get a real
   latency/bandwidth baseline, and evaluate NCCL 2.28 symmetric kernels
   (`NCCL_SYM_KERNEL=AllReduce_AGxLLMC_R`, `NCCL_CTA_POLICY_EFFICIENCY`,
   `NCCL_CUMEM_ENABLE=1`, `NCCL_NVLS_ENABLE=1`) as an alternative to the
   FlashInfer path.

---

## 8. Open questions / things needing a measurement on this box

1. **NVLink 5 point-to-point small-message latency is unsourced.** Needs a
   direct microbenchmark (ping-pong on a 16 B flag through NVSwitch, with
   `%globaltimer`) on an idle box. Everything in §3.2 hinges on `L`.
2. **The 120 KiB (FlashInfer MNNVL) vs 512 KiB (SGLang v2) one-shot crossover
   disagreement** for the same hardware and world size. Sweep both kernels.
3. **Where does the 8.2 µs "transfer" actually go?** Needs ncu or in-kernel
   timestamps to split launch / Lamport-clear / publish / spin-exit / flag-update.
4. **Is the 2 MiB multicast granularity costing us?** Our C1 one-shot payload is
   384 KiB; the Lamport workspace is 3 x 2 MiB minimum. Check whether TLB/L2
   behaviour on a 2 MiB-granular multicast mapping differs from unicast.
5. **Does skew propagate across the 156 chained collectives within one forward?**
   Plot per-instance skew vs layer index. If it grows monotonically, the fix is
   different (insert a resync) than if it is flat.
6. **C64 profile has not been taken.** Everything here is C1. At C64 the kernel
   flips to two-shot with ~1,536 CTAs, which *will* contend with GEMMs, and the
   skew story may invert.
7. **Does the EAGLE draft path introduce rank-local control flow?** Log accepted
   counts per rank.
8. **NVSwitch-side counters.** `nvidia-smi nvlink -e` gives GPU-side counters
   only; whether NVSwitch congestion/latency counters are readable on this host
   (via NSCQ / DCGM) is unknown.
9. **NVSHMEM is not installed.** Whether a device-initiated NVSHMEM or NCCL
   device-API (LSA/multimem) allreduce beats the mnnvl kernel at 48 KiB is
   untested.
10. **PTX ISA multimem qualifier matrix** could not be extracted from
    docs.nvidia.com. The forms in §2.3 are the ones in production use, not a
    complete grammar; check §9.7.9.15 of the PTX ISA before writing a new
    multimem kernel.

---

## Sources

### Local files read
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda.h` (lines 4490-4530,
  13190-13430, 14170-14400) — multicast object API, granularity flags
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh`
  (1814 lines; read 1-120, 241-345, 440-590, 780-1350, 1540-1700)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh` (lines 1-120, 1490-1850)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/comm/mixed_comm.cuh` (multimem PTX)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flashinfer/comm/trtllm_mnnvl_ar.py` (lines 30-240)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flashinfer/comm/trtllm_ar.py` (lines 955-1000)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flashinfer/comm/allreduce.py` (lines 985-1060)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h` (lines 155-250)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/nccl/include/nccl.h` (version, `ncclConfig_t`, CTA policy flags)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2` (`strings`: env var list, algorithm/protocol tokens, symmetric kernel names)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `/home/aman/code/NotSglang/python/sglang/srt/entrypoints/engine.py` (lines 1480-1540)
- `/home/aman/code/NotSglang/python/sglang/srt/server_args.py` (lines 1880-1965, 7808)
- `/home/aman/code/NotSglang/python/sglang/srt/distributed/parallel_state.py`
- `/home/aman/code/NotSglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- `/home/aman/code/NotSglang/python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py`
- `/home/aman/code/NotSglang/python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/glm-5.2-optimization-log.md`
- `/home/aman/code/benchmark/gpubench/analysis/multigpu.py` (skew methodology)
- `/home/aman/code/benchmark/SCORECARD.md`
- `/home/aman/code/third_party/TileRT/tilert/models/glm_5/_dsa_v32/ops/down_allreduce.py`

### Live queries on this box (2026-08-17)
- `nvidia-smi topo -m`, `nvidia-smi nvlink -s`, `nvidia-smi nvlink -e`
- `nvidia-smi -q -d PERFORMANCE` (SW power-capping counters), `nvidia-smi -q` (Fabric state/CliqueId)
- `nvidia-smi --query-gpu=clocks.sm,power.draw,...` sampled 5x
- `lscpu` (NUMA layout)
- `cuda.bindings.driver`: `cuDeviceGetAttribute(MULTICAST_SUPPORTED / HANDLE_TYPE_FABRIC_SUPPORTED / CLUSTER_LAUNCH / MULTIPROCESSOR_COUNT)`, `cuMulticastGetGranularity`, `cuMemGetAllocationGranularity`
- `torch.cuda.get_device_properties(0)` (148 SMs, 132,644,864 B L2, 1965 MHz max)

### URLs fetched and read
- https://www.nvidia.com/en-us/data-center/nvlink/ — NVLink 5 generation table
- https://www.nvidia.com/en-us/data-center/hgx/ — HGX B200 specs
- https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/ — `ncclCommWindowRegister`, symmetric memory, 9x/2.5x latency claims
- https://github.com/NVIDIA/nccl/releases/tag/v2.28.3-1 — device API, CE collectives, Blackwell CTA 32->16
- https://github.com/NVIDIA/nccl/discussions/1869 — NCCL 2.28.3 release discussion
- https://github.com/NVIDIA/nccl/issues/2077 — 128 multicast slot hard limit; NVLS vs Ring field data
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html — env var defaults
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html — `ncclConfig_t` minCTAs/maxCTAs defaults
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html — CTA policy flag semantics
- https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md — algbw/busbw correction factors
- https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html — NVL tuning guidance
- https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html — IMEX purpose and multi-node-only requirement
- https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html — Fabric Manager role
- https://hazyresearch.stanford.edu/blog/2025-09-22-pgl — measured 8xB200 NVLink bandwidth by data path; 8-16 SMs saturate NVLink
- https://arxiv.org/html/2507.04786v1 — "Demystifying NCCL": protocol buffer/slot sizes, LL/LL128/Simple characteristics
- https://arxiv.org/abs/2505.11329 — TokenWeave: fused AllReduce+RMSNorm on 2-8 SMs, 1.28x/1.19x
- https://www.alphaxiv.org/overview/2512.02189v1 — B200 STREAM triad 7.48 TB/s
- https://raw.githubusercontent.com/NVIDIA/nccl/v2.28.3-1/src/init.cc — `NCCL_PARAM(MaxCTAs/MinCTAs/CTAPolicy)`, MAXCHANNELS capping
- https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/multimem_ld_reduce.html — multimem.ld_reduce sem/scope/op list

### Attempted and failed (recorded so nobody repeats it)
- PTX ISA §9.7.9.15 multimem grammar — docs.nvidia.com renders only the ToC through WebFetch
- NVIDIA Blackwell architecture technical brief — returns a cookie notice
- TokenWeave / MSCCL++ full PDFs — binary, not text-extractable
- NVSHMEM API docs page — returned empty
