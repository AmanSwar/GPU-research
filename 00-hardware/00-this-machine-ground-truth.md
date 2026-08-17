# This machine, measured

**What this is.** Every other document in `00-hardware/` is researched — read
off whitepapers, PTX manuals and third-party teardowns. This one is *queried
from the driver on the actual box*, on 2026-08-17. Where it disagrees with a
spec sheet, believe this file. Reproduce it with
[`scripts/device_probe.py`](../scripts/device_probe.py), which talks to
`libcuda.so.1` through `ctypes` so it runs without a working torch or a matching
CUDA runtime.

Everything below is `[verified]` unless marked otherwise.

---

## Bottom line for our system

- **The benchmark runs at 81% of peak clock.** Max SM clock is **1965 MHz**; our
  runs lock to **1597 MHz** for reproducibility. That is a standing **19% haircut
  on every compute-bound number we publish** — including the 365 tok/s headline.
  Before chasing kernel-level percentages, quantify what unlocking costs in
  variance and what it buys in throughput.
- **L2 is 126.5 MiB and up to 79.06 MiB of it can be pinned.** This is a much
  bigger lever on Blackwell than on Hopper (50 MiB L2, no comparably large
  persisting window). At batch 1 our decode is bandwidth-bound on weights that
  stream from HBM every token — the persisting-L2 window is the mechanism for
  keeping a hot working set (router weights, shared expert, attention
  projections, the DSA indexer's weights) off HBM entirely.
- **The 8 GPUs are property-identical.** The probe diffs every device attribute
  against GPU 0 and finds *no* differences. So the persistent rank-arrival skew
  in our profile (rank 0 last to arrive 24% of the time) is **not** explained by
  device capability differences. It is workload, placement, clock/thermal, or
  host-side — see [`07-power-clocks…`](07-power-clocks-thermals-and-determinism.md).
- **`MULTICAST_SUPPORTED = 1`.** NVLink SHARP multicast/reduction is available on
  this fabric, which is what makes NVLS collectives and CUDA multicast objects
  (`cuMulticastCreate`) usable. Collectives are 19.6% of our GPU time; this is
  the hardware path worth measuring against the trtllm one-shot/two-shot kernels
  we currently run.
- **HBM bandwidth works out to ~7.67 TB/s** from the bus width and clock (below),
  and that number sets the hard ceiling on batch-1 decode. Every roofline in this
  corpus should use it rather than a rounded "8 TB/s".
- **BAR1 is 256 GiB — larger than the 183 GiB of HBM.** The whole framebuffer is
  host-addressable, so peer mapping and symmetric-memory schemes are not
  BAR-limited.

---

## 1. Device identity

| property | value |
|---|---|
| product | NVIDIA B200 (SXM), Blackwell |
| count | 8 |
| compute capability | **sm_100** (major 10, minor 0) |
| board part number | 692-2G525-0220-000 |
| GPU part number | 2901-886-A1 |
| VBIOS | 97.00.FA.00.09 |
| driver | 595.71.05 |
| driver CUDA version | 13.2 (`cuDriverGetVersion` → 13020) |
| local toolkit | CUDA 13.3 at `/home/aman/code/cuda-13.3` (nvcc 13.3.73) |
| MIG | disabled |
| persistence mode | enabled |
| virtualization | pass-through |
| ECC | enabled (ECC object 7.16) |

`sm_100` is the important line. Kernels must be built for `sm_100a` (or
`sm_100f`) to reach the architecture-specific instructions — `tcgen05`, the
block-scaled MMA kinds — that this corpus spends most of its time on.

## 2. Compute resources

| property | driver attribute | value |
|---|---|---|
| SMs | `MULTIPROCESSOR_COUNT` | **148** |
| max SM clock | `CLOCK_RATE_KHZ` | **1,965,000 kHz = 1965 MHz** |
| clock during our benchmarks | (locked) | **1597 MHz — 81.3% of max** |
| warp size | `WARP_SIZE` | 32 |
| max threads / block | `MAX_THREADS_PER_BLOCK` | 1024 |
| max threads / SM | `MAX_THREADS_PER_MULTIPROCESSOR` | 2048 (= 64 warps) |
| max blocks / SM | `MAX_BLOCKS_PER_MULTIPROCESSOR` | 32 |
| registers / SM | `MAX_REGISTERS_PER_MULTIPROCESSOR` | 65,536 |
| registers / block | `MAX_REGISTERS_PER_BLOCK` | 65,536 |
| cluster launch | `CLUSTER_LAUNCH` | **supported** |
| async engines (copy) | `ASYNC_ENGINE_COUNT` | 4 |
| compute preemption | `COMPUTE_PREEMPTION_SUPPORTED` | yes |

### The occupancy budget, stated once

2048 threads/SM and 65,536 registers/SM means **32 registers per thread at full
occupancy**. A tensor-core kernel holding a large accumulator tile in registers
routinely wants 128–255, which caps it at 8–16 warps/SM. That is expected and
usually correct: on SM100 accumulators live in *tensor memory* rather than the
register file (see
[`03-tensor-cores…`](03-tensor-cores-tcgen05-and-mma.md)), and deep async
pipelines hide latency with in-flight work rather than with warp count. Do not
read low occupancy as a defect on this architecture.

### Clock headroom, quantified

The clock lock is a measurement decision with a performance price. At the same
SM count and per-SM issue rate, peak compute scales linearly with clock:

```
1597 / 1965 = 0.8127
```

So every compute-bound kernel in our profile — dense GEMM at 37.1%, MoE expert
GEMMs at 19.4% — runs **at most 81.3% as fast as this silicon can go**, by our
own configuration choice. Memory-bound kernels are unaffected (HBM clock is
independent and already at its 3996 MHz maximum). The open question is what boost
clocks actually sustain under a 1000 W sustained NVFP4 load, which is a
measurement, not a lookup: `SW Power Capping` has already accumulated **90.9
seconds** on this box, so the part does hit its power cap in practice.

## 3. Memory hierarchy, as the driver reports it

| level | driver attribute | value |
|---|---|---|
| HBM3e total | `cuDeviceTotalMem` | 178.34 GiB (183,359 MiB reported by nvidia-smi) |
| memory bus width | `GLOBAL_MEMORY_BUS_WIDTH` | **7680 bits = 960 bytes** |
| memory clock | `MEMORY_CLOCK_RATE_KHZ` | 3,996,000 kHz = 3996 MHz |
| **L2 cache** | `L2_CACHE_SIZE` | **126.50 MiB** (132,644,864 B) |
| **max persisting L2** | `MAX_PERSISTING_L2_CACHE_SIZE` | **79.06 MiB** |
| max access-policy window | `MAX_ACCESS_POLICY_WINDOW_SIZE` | 128.00 MiB |
| shared memory / SM | `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | **228.00 KiB** |
| shared memory / block (opt-in) | `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | **227.00 KiB** |
| shared memory / block (default) | `MAX_SHARED_MEMORY_PER_BLOCK` | 48.00 KiB |
| reserved shared / block | `RESERVED_SHARED_MEMORY_PER_BLOCK` | 1.00 KiB |
| constant memory | `TOTAL_CONSTANT_MEMORY` | 64.00 KiB |
| BAR1 | (nvidia-smi) | **256 GiB**, i.e. larger than HBM |

### Derived HBM bandwidth

```
960 bytes/clk  ×  3.996e9 clk/s  ×  2 (DDR)   =  7.672e12 B/s  =  7.67 TB/s
```

`[inferred]` — the ×2 for double-data-rate is the standard convention for how
`MEMORY_CLOCK_RATE_KHZ` is reported, not something the driver states. It lands
within 4% of the commonly quoted 8 TB/s for B200, which is the expected
agreement. **Use 7.67 TB/s as the roofline denominator**, and treat achievable
bandwidth as meaningfully lower again (a STREAM-style measurement on this box is
the obvious next step and has not been taken).

### Why 126.5 MiB of L2 changes the decode strategy

At batch 1 the decode step is not short of FLOPs; it is short of bytes. Every
token drags the active weights across HBM. Two consequences:

1. **79.06 MiB can be pinned.** `cudaAccessPolicyWindow` with a persisting
   carve-out lets a chosen address range hold L2 residency against streaming
   traffic. 79 MiB is enough for a genuinely useful hot set — at NVFP4 (0.5 B/param
   plus scales) that is on the order of 100M+ parameters held off HBM.
2. **The KV cache will evict everything if left alone.** KV traffic is streaming
   and single-use; weights are reused every token. Without an eviction policy
   (`evict_first` / `evict_last` / `no_allocate` on the KV path) the cache
   optimises for exactly the wrong tenant. Whether SGLang/FlashInfer set any
   policy on the KV loads is an open question worth checking in the source.

The 126.5 MiB is `[inferred]` to be physically split across the two dies of the
B200 package, with an access-locality consequence, but the driver reports a single
aggregate figure and does not expose the split — flagged for the microbenchmark
list rather than asserted.

## 4. Interconnect

| property | value |
|---|---|
| NVLink links per GPU | **18** |
| per-link bandwidth | **53.125 GB/s** |
| **aggregate per GPU** | **956.25 GB/s** per direction |
| topology | `NV18` to *every* peer — all-to-all through NVSwitch |
| fabric state | Completed / Success / **Healthy**, CliqueId 0 |
| P2P access | **all 16 ordered pairs supported, every one at performance rank 0** |
| P2P atomics | supported on every link |
| multicast | **`MULTICAST_SUPPORTED = 1`** |
| GPUDirect RDMA | supported |
| PCIe | Gen5 ×16 (host attach) |
| DMA-BUF | supported |

Two things follow. First, **the fabric is uniform**: every GPU pair has the same
18 bonded links and the same performance rank, so there is no "good pair / bad
pair" structure for a collective to exploit or trip over — a TP8 all-reduce sees
a symmetric fabric. Second, `MULTICAST_SUPPORTED` means the NVSwitch can perform
reduction in the switch rather than at the endpoints, which is the hardware
foundation for NVLS-algorithm NCCL and for `cuMulticastCreate`-based custom
collectives. Given that collectives are 19.6% of our GPU time and 47% of that is
*waiting*, the switch-side path deserves a direct A/B against the trtllm
one-shot/two-shot kernels currently in the profile.

## 5. Host

| property | value |
|---|---|
| CPU | 2× Intel Xeon Platinum 8581C @ 2.10 GHz |
| cores / threads | 56 cores/socket, 2 threads/core, **224 logical CPUs** |
| NUMA nodes | 2 — node 0 = CPUs 0–55,112–167; node 1 = CPUs 56–111,168–223 |
| GPU NUMA affinity | GPUs 0–3 → node 0; GPUs 4–7 → node 1 |
| pageable memory access | **not** supported (`PAGEABLE_MEMORY_ACCESS = 0`) |
| managed memory | supported, with concurrent access |
| memory pools | supported |

The GPU-to-NUMA split matters for the *host* side of the loop, not the device
side: all GPU-to-GPU traffic is NVLink and never crosses the CPU interconnect.
But the scheduler process, the Python/C++ launch threads and any pinned host
buffers do live on a socket, and a rank whose driver threads run on the far
socket pays for it. Given that our collective analysis attributes 47% of
collective time to arrival skew with a *persistent per-rank pattern*, host-side
placement (thread affinity, IRQ affinity, pinned-memory NUMA node) is a
first-class suspect and is cheap to test: pin each rank's process to the socket
matching its GPU and re-measure the skew.

`PAGEABLE_MEMORY_ACCESS = 0` is a reminder that host memory must be pinned to be
accessed efficiently — relevant to any KV-offload or expert-offload design.

## 6. Power and thermal state (idle snapshot)

| property | value |
|---|---|
| power limit | **1000 W** (default = current = max; min settable 200 W) |
| draw at idle-with-model-loaded | ~248 W |
| memory power | ~27 W |
| temperature | 31–35 °C |
| performance state | P0 |
| **accumulated SW power capping** | **90,936,966 µs = 90.9 s** |
| accumulated HW slowdown / thermal | **0 s** |

No thermal or hardware slowdown has ever been recorded on this box, but the
software power cap *has* engaged for 90.9 seconds cumulatively. That is the
signature of a part that reaches its 1000 W envelope under load. Any
uncapped-clock experiment must therefore sample `Clocks Event Reasons`
continuously, or it will report an average over a mixture of clock states without
knowing it.

## 7. Memory occupancy at capture time

| | |
|---|---|
| total | 183,359 MiB |
| used | 166,357 MiB |
| free | 16,268 MiB |
| reserved | 735 MiB |

A model was resident on all 8 GPUs when this probe ran, at ~90.7% of framebuffer.
For reference, our serving runs use `--mem-fraction-static 0.85` while published
configs for this model class use 0.92 — a gap of ~13 GiB per GPU of KV cache,
which is candidate C in the optimization ledger.

## 8. What this file does *not* establish

Everything here is a *reported capability*, not a *measured rate*. The driver
will happily tell you the bus is 7680 bits wide; it will not tell you what
fraction of 7.67 TB/s a real access pattern achieves. Not established, and worth
measuring on these otherwise-idle GPUs:

- achieved HBM bandwidth (STREAM-style, and under the strided/gather pattern an
  MoE expert GEMM actually issues)
- L1/L2/HBM/peer-NVLink **latencies** (pointer-chase)
- whether the 126.5 MiB L2 behaves as one pool or as two per-die pools
- small-message all-reduce latency across 8 ranks, versus the one-shot threshold
- real `tcgen05` MMA issue rates for each block-scaled kind
- sustained clock under a 1000 W NVFP4 load, and the boost-vs-locked delta
- whether per-rank host thread placement changes the measured arrival skew

Each is a candidate for a microbenchmark; see
[`06-microbenchmarks…`](06-microbenchmarks-and-reverse-engineering.md).

## Sources

- `scripts/device_probe.py` against `libcuda.so.1`, driver 595.71.05, 2026-08-17
- `nvidia-smi -q`, `nvidia-smi topo -m`, `nvidia-smi nvlink -s -c`, same date
- `lscpu`
- `/home/aman/code/cuda-13.3/activate.sh` (toolchain provenance)
- Profile and configuration context: `NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`,
  `benchmark/SCORECARD.md`
