# Instruction scheduling on SM100: SASS, stalls, and hiding latency at batch 1

## What this is

The lowest level of the stack: how a Blackwell SM actually issues instructions, where cycles
are lost, and what you can do about it when every GEMM in your model has become a GEMV.
Most of the SASS-level content here was **measured on this box** — the control-bit encoding
was decoded and validated against real `sm_100a` disassembly, and the stall taxonomy was
pulled out of the Nsight Compute 2026.1.1 metric database for chip `gb100` rather than
recalled. Where I could not source something I say so. Claims are tagged
`[verified]` / `[reported]` / `[inferred]` / `[unverified]`.

> **Audit note (second pass).** Every measured number in §3, §4 and §6 was independently
> re-derived from the same cubins with a fresh decoder and reproduces exactly. Three claims
> did **not** survive and have been corrected in place: the tcgen05 write-barrier statistic
> (§4.2 — it is 100%, not 97.6%, once you stop counting non-MMA `UTC*` opcodes), the SM clock
> claim (§1 Table 1 — 1597 MHz is an *idle* clock on this node, not a load clock), and the
> attribution of the FP4 throughput number to the wrong table of arXiv:2512.02189 (§9.1).
> Missing material was added: the PTX-ISA-sourced tcgen05 completion model (§4.4), the
> measured LDGSTS/TMA bytes-in-flight curve (§9.2.3), and the GPC/cluster quantisation trap
> for persistent kernels (§9.2.5).

## Bottom line for our system

- **Every scheduler on SM100 issues at most one instruction per cycle, for one warp.** There
  is no dual issue. 4 schedulers × 1 inst/cycle × 148 SMs is the hard ceiling on instruction
  throughput, and at batch 1 we are nowhere near it — we are latency-bound, so the lever is
  *independent work per warp*, not more warps. [verified, CUDA PG §20.9.1]
- **The per-warp scoreboard has exactly 6 barrier slots (indices 0–5).** Verified across
  89,584 real SM100 production instructions from `sgl_kernel/sm100`. That is the hardware
  budget for tracking outstanding variable-latency operations in one warp — but a barrier is
  a *counter*, so many loads can share one slot. This is why deeper unrolling still buys MLP.
- **`tcgen05` MMA is invisible to the warp scoreboard.** All **592** `UTCHMMA.2CTA` /
  `UTCQMMA.2CTA` instructions in a real production cubin allocate **no write barrier** —
  100%, not "most" — only a *read* barrier protecting the descriptor registers. Completion is
  tracked exclusively through `UTCBAR` (`tcgen05.commit`) → mbarrier →
  `SYNCS.PHASECHK.TRANS64.TRYWAIT`, which is exactly the mechanism the PTX ISA specifies.
  Consequence: **an ncu profile of an SM100 tensor-core kernel will never show a "waiting for
  MMA" stall reason** — it shows up as `short_scoreboard` or `barrier`. Interpret accordingly.
  [verified, measured + PTX ISA §9.7.17.6.2.1.1]
- **Nsight Compute on Blackwell dropped two stall reasons and added none.** `imc_miss` and
  `gmma` exist for `ga100`/`gh100` and are absent from `gb100`. There is no instruction-cache
  stall counter on Blackwell; i-cache pressure is only observable through
  `stall_no_instruction`, which conflates i-cache miss with fetch arbitration.
- **Do not hard-code an SM clock.** This node DVFS's across at least **1050–1965 MHz**;
  `nvidia-smi` reports 1597 MHz at 0% utilisation with no throttle reason active, 1965 MHz on
  a genuinely busy GPU, and ~1050–1215 MHz on power-capped ones. Read the clock at measurement
  time (or take `gpc__cycles_elapsed` from ncu); any cycle↔time conversion done against a
  fixed number is wrong by up to 1.9×. [verified, `nvidia-smi` sampled repeatedly this session]
- **Low occupancy is correct for our dense GEMM (37.1%) and MoE expert GEMMs (19.4%), and
  wrong for our DSA indexer (5.8%) and elementwise (3.7%).** The decision rule is Little's Law
  per pipe, not the occupancy percentage. Section 8 works the numbers.
- **At batch 1 the fix list is: outstanding loads, not FLOPs.** Dense GEMM at 37.1% of kernel
  time has arithmetic intensity 2 (FP8) to 4 (NVFP4) FLOP/byte against a machine balance of
  ~1125 — it is `long_scoreboard`, not tensor-core, time. The levers are `LDG.E.128` +
  `.CONSTANT`, unrolled software pipelines with distinct barrier indices, **32 KiB of bytes in
  flight per SM** (§9.2.3, now a measured target rather than a guess), persistent kernels +
  CUDA graphs to kill the launch tax, and `griddepcontrol` to overlap tail/head of dependent
  kernels. Section 9.
- **The MoE expert working set does not fit L2, but the draft/verify pair might.** L2 on this
  part is **126 MB** [verified, `cudaDeviceProp::l2CacheSize` = 132,644,864 B and the Blackwell
  Tuning Guide]. With 256 experts / 8 active, per-token routed weights are a fraction of that;
  EAGLE 3-1-4 hits the *same* expert set twice per accepted token. Measure
  `lts__t_sectors_lookup_hit` on the expert GEMMs across a draft/verify pair before doing
  anything else to the MoE path — it is the cheapest 19.4% you own.
- **47% of our collective time is rank arrival skew** — ~9.2% of *all* kernel time spent
  waiting. That is a scheduling problem at the grid level with the same shape as a scoreboard
  stall at the warp level: the fix is to give the waiting party independent work (PDL, §9.2.2),
  i.e. overlap the pre-AR compute of rank *i* with the AR wait, not to make the AR faster.
  Note that `stall_sleeping` and `stall_membar` are the two counters that will show you the
  spin side of that wait, and both exist on `gb100`.

---

## 1. What the SM100 issue engine actually is

From the CUDA C++ Programming Guide, "Compute Capability 10.0 → Architecture" [verified]:

> A Streaming Multiprocessor (SM) consists of: 128 FP32 cores…, 64 FP64 cores…, 64 INT32
> cores…, 4 mixed-precision fifth-generation Tensor Cores…, 16 special function units…,
> **4 warp schedulers**.
> An SM statically distributes its warps among its schedulers. Then, at every instruction
> issue time, **each scheduler issues one instruction for one of its assigned warps that is
> ready to execute**, if any.

Two things fall out of that sentence and they matter a lot:

1. **No dual issue.** Since Volta, NVIDIA datacenter SMs are single-dispatch per scheduler.
   Anything you read about "dual issue" applies to Kepler/Maxwell/Pascal, not SM100. The only
   "two things at once" on SM100 are (a) different schedulers issuing in the same cycle, and
   (b) async engines (TMA, tcgen05, LDGSTS) running in the background after a single issue.
2. **Warps are statically bound to a scheduler.** With 64 warps/SM and 4 schedulers, each
   scheduler owns up to 16 warps. If you launch 4 warps per CTA and one CTA per SM you get
   *one warp per scheduler* — each scheduler has exactly one warp to choose from, and any
   stall in that warp is a dead issue slot. This is the single most important occupancy fact
   for low-occupancy tensor-core designs.

### Table 1 — SM100 resources (compute capability 10.0)

| Resource | Value | Source |
|---|---|---|
| Warp schedulers per SM | 4 | [verified] CUDA PG §20.9.1 |
| Instructions issued per scheduler per cycle | 1 | [verified] CUDA PG §20.9.1 |
| FP32 cores / SM | 128 | [verified] CUDA PG §20.9.1 |
| INT32 cores / SM | 64 | [verified] CUDA PG §20.9.1 |
| FP64 cores / SM | 64 | [verified] CUDA PG §20.9.1 |
| SFU / SM | 16 | [verified] CUDA PG §20.9.1 |
| 5th-gen Tensor Cores / SM | 4 | [verified] CUDA PG §20.9.1 |
| Max warps / SM | 64 | [verified] Blackwell Tuning Guide |
| Max thread blocks / SM | 32 | [verified] Blackwell Tuning Guide |
| 32-bit registers / SM | 65536 (256 KiB) | [verified] Blackwell Tuning Guide |
| Max registers / thread | 255 | [verified] Blackwell Tuning Guide |
| Unified L1 + shared memory / SM | 256 KB | [verified] CUDA PG §20.9.1 |
| Shared memory carveouts | 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB | [verified] CUDA PG §20.9.3 |
| Max shared memory / block | 227 KB (1 KB reserved) | [verified] CUDA PG §20.9.3 |
| Max portable cluster size | 8; **16 non-portable on B200** via `cudaFuncAttributeNonPortableClusterSizeAllowed` | [verified] Blackwell Tuning Guide §1.4.1.2, verbatim |
| SMs per GPU | **148** | [verified] `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)` on this node |
| L2 cache | **132,644,864 B = 126.5 MiB** ("126 MB") | [verified] `l2CacheSize` on this node + Blackwell Tuning Guide §1.4.2.2 |
| Memory bus width | 7680 bit (960 B) | [verified] `cuDeviceGetAttribute` |
| SM clock, max / **observed range** | 1965 MHz max; **observed 1050–1965 MHz** | [verified] `nvidia-smi` sampled this session — see note below |
| Memory clock | 3996 MHz | [verified] `nvidia-smi` |
| HBM peak BW, computed / marketed | 960 B × 3996 MHz × 2 = **7.67 TB/s** / 8 TB/s | [verified] arithmetic from device attrs; [verified] DGX B200 page quotes 64 TB/s across 8 GPUs |
| Power limit | 1000 W | [verified] `nvidia-smi` |
| Dense FP4 tensor peak | 9 PFLOPS/GPU (72 PFLOPS per DGX B200, dense) | [verified] nvidia.com DGX B200 spec table |

**On the SM clock — correction to an earlier claim in this document.** A previous revision
asserted "our GPUs run at 1597 MHz under production load". Re-sampling `nvidia-smi` this
session shows 1597 MHz is what the part reports at **0% utilisation with every
`clocks_event_reasons.*` flag Not Active** (~250 W draw). A GPU actually at 100% utilisation
was observed at **1965 MHz**, and power-capped siblings at 1050–1215 MHz with
`clocks_event_reasons` = `0x4` (SW power cap). So 1597 MHz is an idle/persistence clock, not a
load clock, and the "19% optimistic" correction factor derived from it was wrong. **Practical
rule: never convert cycles↔time from a constant. Take `gpc__cycles_elapsed.max` and
`gpu__time_duration.sum` from the same ncu report, or profile with `--clock-control base` when
you want run-to-run comparability.** [verified]

Note the INT32:FP32 ratio of 1:2. Address arithmetic (`IMAD`, `IADD3`, `LEA`, `SHF`) runs on
a datapath half as wide as FP32. In a GEMV inner loop where you emit one `IADD3` per pointer
bump and one `FFMA` per element, the integer pipe is a real co-limiter. That is exactly why
ptxas hoists address math into **uniform registers** (`UIADD3`, `ULEA`, `UMOV`, `USEL`,
`R2UR`) — the uniform datapath computes one value for the whole warp instead of 32.
`UIADD3` was 2.90% and `UMOV` 2.12% of all instructions in the production cubin I analysed
[verified, measured].

### The latency-hiding equation the guide actually states

> The number of instructions required to hide a latency of L clock cycles … If we assume
> instructions with maximum throughput, it is equal to: **4L** for devices of compute
> capability 5.x, 6.1, 6.2, 7.x and 8.x since for these devices, a multiprocessor issues one
> instruction per warp over one clock cycle for four warps at a time. [verified, CUDA PG §8.2.3]

**The list stops at 8.x.** Compute capabilities 9.0 and 10.0 are not in it — this is doc
staleness, not an architectural statement, because §20.9.1 explicitly describes the same
4-scheduler / 1-issue-per-cycle structure for CC 10.0. **4L applies on SM100** [inferred, but
directly from the arch description in the same document].

So: to fully hide a latency of L cycles you need **4L independent instructions in flight per
SM**, from any mix of warps and ILP. That is the whole game.

---

## 2. Tooling: what exists on this box

| Tool | Path | SM100 support |
|---|---|---|
| `nvdisasm` 13.3.73 | `/home/aman/code/NotSglang/.venv/.../nvidia/cu13/bin/nvdisasm` | SM100, SM100a, SM100f, SM101*, SM103*, SM110*, SM120*, SM121* [verified via `--help`] |
| `cuobjdump` | `.../triton/backends/nvidia/bin/cuobjdump` | yes |
| `ptxas` / `nvcc` 13.3.73 | `/home/aman/code/cuda-13.3/nvidia/cu13/bin/` | `-arch=sm_100a` works [verified, compiled] |
| Nsight Compute 2026.1.1 | `/opt/nvidia/nsight-compute/2026.1.1/ncu` | chip `gb100` in `--list-chips` [verified] |
| Nsight Systems 2025.6.3 | `/opt/nvidia/nsight-systems/2025.6.3` | — |

Useful invocations, all verified on this machine:

```bash
NVD=/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvdisasm
CUOBJ=/mnt/persistent/app-data/NotSglang/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump

# what architectures does a .so ship?
$CUOBJ -lelf libfoo.so | awk '{print $NF}' | sed 's/.*\.\(sm_[0-9]*\)\.cubin/\1/' | sort | uniq -c

# pull out just the sm_100 cubins
$CUOBJ -xelf all libfoo.so          # writes <name>.<n>.sm_XXX.cubin into cwd

# register / smem usage without running anything
$CUOBJ -res-usage libfoo.so

# disassemble with the 128-bit encodings (this is what exposes the control bits)
$NVD -c -hex foo.cubin > foo.sass

# per-instruction live register count — the real register-pressure view
$NVD -c -plr -lrm wide foo.cubin | less

# control-flow graph for a single kernel
$NVD -cfg -fun 3 foo.cubin | dot -Tsvg > k.svg
```

`ptxas -v` is still the fastest register-pressure oracle and needs no GPU:

```bash
nvcc -arch=sm_100a -cubin -O3 -Xptxas -v -o /dev/null kernel.cu
# ptxas info : Used 34 registers, used 0 barriers
```

Two `nvdisasm` flags worth knowing that the previous revision missed:

```bash
$NVD -c -json foo.cubin        # --emit-json: machine-readable disassembly, no regex parsing
$NVD -c -novliw foo.cubin      # --no-vliw: "disassemble paired instructions in normal syntax"
```

`-novliw` exists in 13.3 because *some* target in nvdisasm's arch list uses a paired/VLIW
print form. **It is not SM100.** `diff <($NVD -c k.cubin) <($NVD -c -novliw k.cubin)` is empty
for both `sm_100a` and `sm_120` cubins on this box, i.e. there is no paired-instruction
encoding to worry about on Blackwell datacenter parts, which is consistent with the
one-instruction-per-scheduler-per-cycle model in §1 [verified, ran it].

### Is the SASS ISA documented?

**Partly, and better than folklore says.** NVIDIA publishes an *opcode list with one-line
descriptions* per architecture in the CUDA Binary Utilities doc — for us that is
[§4.4 "Blackwell Instruction Set", Table 8](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#blackwell-instruction-set)
(CC 10.0 and 12.0), which also documents the operand-location syntax you will see in
disassembly: `RX`, `URX`, `SRX`, `PX`, `UPX`, `c[X][Y]`, **`desc[URX][RY]`** (memory
descriptor), **`gdesc[URX]`** (global memory descriptor) and **`tmem[URX]`** (tensor memory)
[verified, read the page]. What NVIDIA does **not** publish is operand *encodings* and the
*control field*. Everything in §3 below is reverse-engineered. What is different here from the
usual blog-post situation is that the decode was re-derived and re-validated against real
`sm_100a` output on this machine rather than trusting a table — see the validation
methodology, which is the part you should actually trust.

---

## 3. The control field on SM100 — decoded and validated here

SM100 keeps the Turing/Ampere/Hopper **128-bit fixed-length instruction** format: 16 bytes
per instruction, addresses step by `0x10` [verified — every disassembly on this box].
`nvdisasm -hex` prints it as two 64-bit words:

```
/*0240*/  LDG.E.128.CONSTANT R4, desc[UR4][R16.64] ;  /* 0x00000004100477a5 */
                                                      /* 0x000ea8000c1e1900 */
                                                         ^^^^^^^ control bits live here
```

The scheduling control field occupies **bits 105–127** of the 128-bit word, i.e. bits 41–63
of the high 64-bit word:

```python
ctrl = (hi >> 41) & 0x7FFFFF     # 23 bits
```

### Table 2 — control field layout, validated on sm_100a

| Field | Bits (within ctrl) | Instruction bits | Width | Meaning |
|---|---|---|---|---|
| `stall` | 0–3 | 105–108 | 4 | cycles to wait before issuing the *next* instruction of this warp |
| `yield` | 4 | 109 | 1 | allow the scheduler to switch to another warp after this instruction |
| `wrtdb` | 5–7 | 110–112 | 3 | write-barrier index to allocate (7 = none) |
| `readb` | 8–10 | 113–115 | 3 | read-barrier index to allocate (7 = none) |
| `watdb` | 11–16 | 116–121 | 6 | wait mask — stall until each set barrier clears |
| `reuse` | 17–20 | 122–125 | 4 | operand-reuse-cache hints, one bit per source slot |
| — | 21–22 | 126–127 | 2 | **always zero** across 89,584 production instructions [verified] |

### How I validated it (do this yourself before trusting any decode)

The decode is confirmed by three independent structural signals, all present in the
disassembly of a trivial FP16 GEMV compiled with `nvcc -arch=sm_100a`:

1. **Producer/consumer barrier match.** `LDG.E.128.CONSTANT R4` at `0x0240` allocates write
   barrier **2**; `HADD2.F32 R21, -RZ, R4.H1_H1` at `0x02e0` — the first consumer of `R4` —
   carries wait mask `000100`, i.e. bit 2. The second load at `0x0260` allocates barrier
   **3**, and its consumer at `0x0300` waits on `001000`. A wrong bit assignment cannot
   produce that correlation.
2. **`.reuse` suffix match.** `nvdisasm` prints `.reuse` on source operands. At `0x0120`,
   `LOP3.LUT R4, R6.reuse, 0x300, RZ, ...` has reuse bits `0001` — slot A only, which is the
   operand `nvdisasm` annotated. At `0x0300`, `HADD2.F32 R22, -RZ, R8.reuse.H1_H1` has
   `0100`. The bit index tracks the annotated operand slot.
3. **Bank of allocated indices.** Across 89,584 instructions from real production cubins,
   `wrtdb` and `readb` take values in **{0,1,2,3,4,5,7}** and never 6. 7 is the "none"
   sentinel, giving exactly **6 usable barriers** — the same count as Volta through Hopper.
   A misaligned field would produce a uniform distribution.

**Independent corroboration [verified].** The same bit assignment — stall 105–108, yield 109,
write barrier 110–112, read barrier 113–115, six-bit wait mask 116–121, four reuse flags
122–125, bits 126–127 zero — is stated verbatim by *The Software Frontier*, who decoded it on
five architectures including `sm_100`, and it traces to Jia, Maggioni, Staiger & Scarpazza,
*Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking* (arXiv:1804.06826) and its
Turing successor (arXiv:1903.07486), which are the original public sources for the field
layout. Three independent derivations agreeing is about as good as an undocumented encoding
gets. Note the field is **23 bits extracted, 21 bits used**.

Decoder (drop this in your toolbox):

```python
#!/usr/bin/env python3
"""Decode SM100 SASS scheduling control bits from `nvdisasm -c -hex` output."""
import re, sys
lines = open(sys.argv[1]).read().splitlines()
print(f"{'ADDR':>6} {'STL':>3} {'Y':>1} {'WR':>2} {'RD':>2} {'WAIT':>6} {'REUSE':>5}  INSTRUCTION")
i = 0
while i < len(lines):
    m = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* 0x[0-9a-f]{16} \*/', lines[i])
    if m and i + 1 < len(lines):
        m2 = re.search(r'/\* (0x[0-9a-f]{16}) \*/', lines[i+1])
        if m2:
            c = (int(m2.group(1), 16) >> 41) & 0x7FFFFF
            stall, yld = c & 0xF, (c >> 4) & 1
            wr, rd     = (c >> 5) & 7, (c >> 8) & 7
            wait, reuse = (c >> 11) & 0x3F, (c >> 17) & 0xF
            print(f"{m.group(1):>6} {stall:>3} {yld:>1} "
                  f"{'-' if wr==7 else wr:>2} {'-' if rd==7 else rd:>2} "
                  f"{wait:06b} {reuse:>5b}  {m.group(2).strip()}")
            i += 2; continue
    i += 1
```

### What the fields mean operationally

- **`stall` (0–15)** is a *static* delay the compiler bakes in for fixed-latency dependencies.
  If `FFMA` has a 4-cycle result latency and the next instruction consumes its output, ptxas
  writes `stall=4` on the FFMA rather than burning a scoreboard slot. In ncu this shows up as
  `stall_wait` — "waiting on a fixed latency execution dependency". **`stall_wait` is compiler
  scheduling, not hardware; the fix is ILP, never occupancy.**
- **`yield`** lets the scheduler consider other warps. It is set on **88% of production
  instructions** [verified, measured]. Cleared `yield` (12%) is the compiler saying "keep this
  warp resident, the next instruction is imminent" — typically inside a tight dependent chain.
- **`wrtdb`** = "this instruction produces a result at an unpredictable time; allocate barrier
  *n* and signal it when the destination register is written."
- **`readb`** = "this instruction reads its source registers at an unpredictable time;
  allocate barrier *n* and signal it when the sources have been consumed." This is what
  protects a `LDGSTS` or a `UTCQMMA` source operand from being overwritten too early. It is
  the reason WAR hazards on async instructions cost you a barrier slot too.
- **`watdb`** is a *mask*, not an index: one instruction can wait on up to 6 barriers at once.
  Measured popcount distribution over production SASS: 0 → 89.0%, 1 → 10.6%, 2 → 0.26%,
  3 → 0.13%, 4 → 0.07% [verified, measured].

### Table 3 — measured control-bit statistics, 89,584 SM100 production instructions

Source: 100 kernels in `common_ops.abi3.123.sm_100.cubin`, extracted from
`sgl_kernel/sm100/common_ops.abi3.so` — the CUTLASS FP8/INT8 GEMM epilogue kernels our engine
actually links. All numbers [verified, measured on this box].

| Metric | Value |
|---|---|
| Mean `stall` | **2.17 cycles/instruction** |
| `stall == 1` | 53.8% |
| `stall == 2` | 23.4% |
| `stall == 4` | 6.7% |
| `stall == 5` | 5.6% |
| `stall >= 11` | 1.8% |
| `yield` set | 88.0% |
| Instructions allocating a write barrier | 10.4% (9,284) |
| Instructions allocating a read barrier | 5.6% (5,000) |
| Instructions with a non-empty wait mask | 11.0% |
| Instructions with ≥1 `.reuse` bit | 6.5% |
| Barrier indices ever used | **0,1,2,3,4,5** (never 6) |
| Control bits 21–22 | always 0 |

A mean stall of 2.17 means the compiler expects the SM to spend, on average, ~2 cycles per
instruction not issuing from *that* warp. With 4 schedulers you need roughly
`4 × 2.17 ≈ 9` warps of ILP-free code, or fewer warps with ILP, just to cover the *static*
scheduling delays — before you have covered a single memory access.

### Table 4 — which SM100 opcodes allocate barriers (i.e. are variable-latency)

Measured, top of the distribution [verified]:

| Opcode | Sets write barrier | Sets read barrier | What it is |
|---|---:|---:|---|
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | 1168 | 640 | mbarrier `try_wait` with parity — predicate is variable-latency |
| `LDCU`, `LDCU.64`, `LDCU.128` | 1874 | 0 | uniform-register constant load |
| `SYNCS.EXCH.64` | 808 | 808 | mbarrier arrive/exchange |
| `LDS`, `LDS.128` | 910 | 0 | shared-memory load → `short_scoreboard` |
| `LDSM.16.M88.4` | 720 | 530 | `ldmatrix` |
| `LDC`, `LDC.64` | 948 | 0 | constant-bank load |
| `S2R`, `S2UR` | 688 | 0 | special-register read (tid/ctaid) — yes, these are scoreboarded |
| `LDG.E`, `LDG.E.LTC128B.128` | 448 | 424 | global load → `long_scoreboard` |
| `LDGSTS.E.BYPASS.LTC128B.128` | 0 | **1237** | `cp.async`: no result register, only a WAR barrier on the address |
| `LDGDEPBAR` | 308 | 0 | `cp.async` group commit |
| `FENCE.VIEW.ASYNC.S` | 200 | 0 | async-proxy fence |
| `SHFL.IDX` | 160 | 0 | warp shuffle — **variable latency, costs a barrier** |
| `LDL`, `LDL.LU` | 252 | 0 | local memory (i.e. spills) |
| `UTMACCTL.PF` | 0 | 64 | TMA descriptor prefetch |

Two practical takeaways:

- **`S2R SR_TID.X` and `SHFL` are scoreboarded.** A warp-level reduction with 5 `SHFL.IDX`
  steps burns 5 sequential barrier round-trips. In our DSA indexer and in any GEMV epilogue
  that does a cross-lane reduction, that is a chain of variable-latency dependencies that no
  amount of unrolling in the *k* loop helps. Use `__reduce_add_sync` where the dtype allows
  (it maps to a `REDUX` instruction) or restructure to reduce in shared memory / via the
  tensor-core accumulator.
- **`LDGSTS` allocates only a read barrier**, because the destination is shared memory, not a
  register. Its completion is tracked by `LDGDEPBAR` + the `cp.async` group counters, not by
  the warp scoreboard. That is why `cp.async` waits surface as `short_scoreboard` /
  `mio_throttle`, never as `long_scoreboard`.

---

## 4. Scoreboarding, and why `tcgen05` is different

### 4.1 Barrier semantics

Each of the 6 barriers per warp is a **counter**, not a flag. Multiple instructions may
allocate the same index; the wait clears only when all of them have signalled. This is
directly observable in the compiler's output. From an 8-way-unrolled float4 GEMV I compiled
for `sm_100a` [verified, measured]:

```
  ADDR STL Y WR RD   WAIT REUSE  INSTRUCTION
  0180   4 1  2  - 000000  0000  LDG.E.128.CONSTANT R4,  desc[UR4][R30.64+-0x4000]
  0190   4 1  2  - 000000  0000  LDG.E.128.CONSTANT R8,  desc[UR4][R28.64+-0x4000]
  01a0   4 1  3  - 000000  0000  LDG.E.128.CONSTANT R12, desc[UR4][R30.64+-0x3000]
  01b0   4 1  3  - 000000  0000  LDG.E.128.CONSTANT R16, desc[UR4][R28.64+-0x3000]
  01c0   4 1  4  - 000000  0000  LDG.E.128.CONSTANT R24, desc[UR4][R30.64+-0x2000]
  01d0   1 1  4  - 000000  0000  LDG.E.128.CONSTANT R20, desc[UR4][R28.64+-0x2000]
  01e0   4 0  -  - 000100  0000  FMUL R5, R5, R9              <- waits barrier 2 (both loads)
  01f0   1 1  -  - 000000  0000  FFMA R5, R4, R8, R5
  0200   3 0  -  - 001000  0000  FMUL R13, R13, R17           <- waits barrier 3
  ...
  0240   4 1  2  - 000000  0000  LDG.E.128.CONSTANT R8,  desc[UR4][R30.64+-0x1000]  <- reuses idx 2
  0250   1 1  2  - 000000  0000  LDG.E.128.CONSTANT R4,  desc[UR4][R28.64+-0x1000]
```

Six 128-bit loads are outstanding at `0x01d0` across three barrier indices — **3 KiB of
in-flight data per warp** (6 × 16 B × 32 lanes). The moment barrier 2 clears, the compiler
recycles index 2 for the next pair. This is a hand-rolled software pipeline expressed entirely
in the control bits.

**The practical rule:** the depth of your memory pipeline is limited by how many *distinct
consumption points* you have, not by 6 loads. Group loads that will be consumed together
onto one barrier; use a fresh index for each pipeline stage. Six stages is the ceiling — and
in practice fewer, because `LDS`, `LDC`, `SHFL` and mbarrier ops compete for the same 6 slots.
**A megakernel that fuses many phases into one warp will run out of barrier indices and start
serialising**, which is one of the under-appreciated costs of the megakernel style.

### 4.2 `tcgen05` MMA does not participate in the scoreboard

This is the most important SM100-specific scheduling fact I found, and it is
[verified, measured] from a production cubin containing real tcgen05 code
(`common_ops.abi3.11.sm_100.cubin`, 99,544 instructions, 592 `UTC*` instructions):

```
UTC* write-barrier usage : {none: 972, 1: 18, 2: 6}     <- 97.6% allocate NO write barrier
UTC* read-barrier usage  : {1: 546, 2: 307, 3: 62, 4: 46, none: 35}
UTC* stall counts        : {1: 716, 2: 153, 3: 68, 4: 23, ...}
```

The issue sequence looks like this [verified, real disassembly]:

```
  ADDR STL Y WR RD   WAIT   RU  INSTRUCTION
  63f0   1 1  -  - 000000 0000  UMOV UR5, 0x80004020
  6400   1 1  -  2 000000 0000  UTCQMMA.2CTA gdesc[UR28], gdesc[UR6], tmem[UR11], tmem[UR22], idesc[UR23], !UPT
  6410   1 1  -  2 000000 0000  UTCQMMA.2CTA gdesc[UR52], gdesc[UR4], tmem[UR10], tmem[UR20], idesc[UR21],  UPT
  6420   1 1  -  2 000000 0000  UTCBAR.2CTA.MULTICAST [UR9], URZ, UR8
  6430   1 1  3  - 000000 0000  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR14+0x8], R2
  6440   1 1  -  - 000000 0000  UIADD3 UR50, UPT, UPT, UR28, 0x100, URZ
  6450  11 1  -  - 001100 0000  @!P0 BRA `(.L_x_58)          <- waits barriers 2 and 3
```

Reading it:

- Two `UTCQMMA.2CTA` (FP8 UMMA, `cta_group::2`) issue back-to-back with `stall=1`. They cost
  **one issue slot each** and then run entirely in the background.
- They allocate **read barrier 2** — that protects the uniform registers holding the SMEM/TMEM
  descriptors from being overwritten while the tensor core is still consuming them.
- `UTCBAR.2CTA.MULTICAST` is `tcgen05.commit`: it arms an mbarrier to be signalled on MMA
  completion, multicast across the CTA pair.
- Completion is polled by `SYNCS.PHASECHK.TRANS64.TRYWAIT` (an mbarrier `try_wait` with
  parity), which *does* allocate a write barrier (3, for the predicate), and the spin-branch
  waits on `{2,3}` with `stall=11`.

**Consequences you must internalise before reading any ncu report on SM100:**

1. There is no "MMA busy" warp state. Nsight Compute removed `stall_gmma` after Hopper
   [verified — see §7] precisely because there is nothing left to count: the SASS scoreboard
   no longer knows about MMA completion.
2. Time spent waiting for the tensor core appears as **`stall_short_scoreboard`** (the mbarrier
   `SYNCS` lives in shared memory / MIO) or **`stall_barrier`**. If you see a big
   `short_scoreboard` bar on a Blackwell GEMM, do not go looking for shared-memory bank
   conflicts first — check whether you are simply MMA-latency-bound with too few MMAs in
   flight.
3. Because MMA issue is one slot and completion is decoupled, **the number of in-flight MMAs
   is limited by TMEM accumulator capacity and by how many `tcgen05.commit` groups you can
   track, not by warps.** Colfax/NVIDIA analysis puts speed-of-light at "on the order of 256
   to 1024 in-flight MMA instructions", with real kernels carrying only 1–4 and therefore
   capping around 78–80% of peak [reported — jianyuh.github.io, single author blog, not
   independently verified].
4. `2CTA` in SASS = `cta_group::2` in PTX: one instruction driving the tensor cores of two SMs
   in a cluster pair [verified — the mnemonic is literally `UTCQMMA.2CTA`; semantics
   corroborated by Colfax/CUTLASS docs, reported].

### 4.3 SM100 opcodes worth recognising

Harvested from real cubins on this box [verified, all appear in disassembly]:

| Opcode | Meaning |
|---|---|
| `UTCHMMA` / `UTCQMMA` / `UTCOMMA` / `UTCIMMA` | tcgen05 MMA: FP16/BF16/TF32, FP8(+FP6/FP4), FP4-dense, INT8 |
| `HMMA.16816.F32`, `IMMA.16832.S8.S8.SAT` | classic warp-level `mma.sync` — still emitted for SM80-style CUTLASS kernels compiled to sm_100 |
| `UTCBAR.2CTA.MULTICAST` | `tcgen05.commit` |
| `UTMALDG.3D.2CTA` | TMA bulk tensor load, 3-D, cluster-pair multicast |
| `UTMACCTL.PF` | TMA descriptor prefetch |
| `LDGSTS.E.BYPASS.LTC128B.128` | `cp.async` 16 B, L1-bypass, 128 B L2 sector policy |
| `LDGDEPBAR` | `cp.async.commit_group` |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` / `SYNCS.EXCH.64` / `SYNCS.ARRIVE.TRANS64.RED.A1T0` | mbarrier try_wait / arrive-and-exchange / arrive-with-reduction |
| `FENCE.VIEW.ASYNC.S` | `fence.proxy.async.shared` |
| `LDG.E.128.CONSTANT` | `ld.global.nc` (i.e. `__ldg`) 16 B — the non-coherent/read-only path |
| `LDG.E.LTC128B.128` | 16 B global load with 128 B L2 sector hint |
| `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | FP32→FP8-E4M3 saturating convert-and-pack — our quantisation hot instruction |
| `F2FP.F16.E4M3.UNPACK_B` | FP8-E4M3 → FP16 dequantise |
| `FMUL2`, `FADD2`, `FMNMX3`, `VIADD` | paired/3-input FP32 and vector-int forms |
| `R2UR` | move regular → uniform register |
| `desc[URn][Rm.64]` addressing | descriptor-based global addressing (base+bounds in a uniform register pair) |

The `F2FP.*.E4M3.*` pair being 5.29% of all instructions in one CUTLASS FP8 cubin is worth
noting given our profile shows quant at 2.4% — the conversion work is not free and it lives
inside the GEMM epilogue, not only in standalone quant kernels.

---

## 5. Register banks and the `.reuse` cache

The register file is 65,536 32-bit registers per SM (256 KiB) [verified]. On Volta through
Ampere it is documented in the reverse-engineering literature as being organised in **2 banks
of 2×32-bit** (Volta) with an **operand-reuse cache** in front of the collectors to absorb
conflicts. **I could not find a primary source describing the SM100 register bank count or
mapping. Not sourced.**

What *is* verified is that the mechanism still exists and is still used:

- The `reuse` field is 4 bits, one per source operand slot [verified, decoded].
- 6.5% of production instructions carry at least one reuse bit; 30 instructions in 89,584
  carried two [verified, measured].
- `nvdisasm` annotates it in the mnemonic: `LOP3.LUT R4, R6.reuse, 0x300, RZ, 0xc0, !PT`,
  `SHF.L.U64.HI R5, R12.reuse, 0x1, R13` [verified].

Practical guidance, honestly labelled:

- **Do not hand-tune register bank assignment.** You cannot control it from CUDA C, ptxas
  reassigns registers freely, and the bank layout for SM100 is unpublished. The classic
  Maxwell-era trick of pinning `R4/R5/R6` to different banks is not actionable here.
- **Do control register *pressure*.** Every spill becomes `LDL`/`STL`, which is local memory,
  which goes through L1 and allocates a scoreboard barrier — `LDL`/`LDL.LU` appear 252 times
  as barrier-allocating instructions in the production cubin [verified]. Watch
  `ptxas -v` for "spill stores"/"spill loads" and `local_op_ld/st` sector counts in ncu.
- **`.reuse` only fires on back-to-back instructions using the same operand in the same slot.**
  Writing an inner loop so that the same accumulator/operand register appears in the same
  position of consecutive FFMAs is the one legitimate source-level nudge, and even that is
  advisory [inferred from the mechanism; the compiler decides].
- A real curiosity from my unroll sweep: **more unrolling used *fewer* registers.**
  `-O3`, `sm_100a`, identical source, only `UNROLL` changed:

  | UNROLL | registers | SASS instructions | text bytes | peak distinct in-flight barrier groups |
  |---:|---:|---:|---:|---:|
  | 1 | 40 | 232 | 3712 (3.6 KiB) | 4 |
  | 2 | 44 | 168 | 2688 (2.6 KiB) | 4 |
  | 4 | 42 | 168 | 2688 (2.6 KiB) | 4 |
  | 8 | **34** | 120 | 1920 (1.9 KiB) | 3 |

  [verified, measured with `ptxas -v` and the decoder]. The lesson is that ptxas re-derives
  the schedule from scratch at every unroll factor and its register allocation is not monotone
  in unroll depth. **Measure; do not assume unrolling costs registers.** At UNROLL=8 the
  compiler emitted a fully software-pipelined loop with 16 `LDG.E.128` and interleaved FFMAs,
  which is exactly what you want, in *fewer* registers than the naive version.

---

## 6. Instruction cache pressure

This is real, it bites megakernels and heavily-unrolled code, and on Blackwell it is **almost
unobservable**.

### What I measured

`nvdisasm` gives exact text size: instructions × 16 bytes. From the production CUTLASS cubin
[verified, measured]:

| SASS instructions | text bytes | kernel |
|---:|---:|---|
| 2272 | 35.5 KiB | `cutlass::device_kernel<GemmUniversal<...>>` |
| 2216 | 34.6 KiB | `cutlass::device_kernel<GemmUniversal<...>>` |
| 1976 | 30.9 KiB | `cutlass::device_kernel<GemmUniversal<...>>` |
| 1944 | 30.4 KiB | `cutlass::Kernel2<DefaultGemmWithVisitor<float_e4m3_t,...>>` |
| 1824 | 28.5 KiB | `cutlass::device_kernel<GemmUniversal<...>>` |

The tcgen05 cubin's largest kernels run to **99,544 instructions across the file** — that is
1.5 MB of SASS in one cubin, and individual kernels there are far larger than the 35 KiB
above. Our own compiled GEMV variants are 1.9–3.6 KiB, i.e. i-cache-trivial.

### What the cache actually is

**Not sourced for SM100.** NVIDIA does not publish L0/L1 instruction cache capacities for any
compute-capability generation. The commonly quoted figures for Volta/Turing/Ampere (an ~16 KB
per-SM-sub-partition L0 instruction cache backed by a ~128 KB per-SM L1 instruction cache
backed by L2) come from microbenchmarking papers, and **I found no Blackwell measurement**.
Treat any specific number you see for SM100 i-cache as [unverified].

### How to detect it

There is exactly one counter, and it is a blunt one:

```
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
```

NVIDIA's own definition, read out of the ncu 2026.1.1 metric database on this box [verified]:

> average # of warp cycles spent **waiting to be selected for instruction fetch, or waiting on
> an instruction cache miss**

Note the "or". It conflates fetch arbitration with i-cache misses, and Blackwell has **no
`imc_miss` counter to disambiguate** (§7). So the diagnostic is empirical:

1. If `stall_no_instruction` is a top-3 warp state **and** your kernel text is large
   (> ~16 KiB) **and** control flow is jumping across a wide address range, suspect i-cache.
2. Reduce `#pragma unroll` factors, or hoist rarely-taken branches into `__noinline__`
   functions, or split a megakernel into two kernels and see if the counter drops.
3. `nvdisasm -cfg` will show you whether the hot loop's basic blocks are contiguous. A loop
   whose body straddles a cold error-handling block pays fetch cost for nothing; `__builtin_expect` /
   `[[likely]]` and moving cold code out helps the block layout.

**For our system specifically:** a fully-fused decode megakernel (attention + MoE routing +
expert GEMM + AR in one persistent kernel) is exactly the shape that hits both the 6-barrier
limit (§4.1) and the i-cache limit. That is a concrete argument for a *persistent kernel with
a phase dispatch loop* over a *fully inlined megakernel*: keep each phase's code small enough
to stay resident, and pay a branch instead of a fetch miss.

---

## 7. The Nsight Compute stall taxonomy on Blackwell

### 7.1 What exists, and what Blackwell removed

I diffed the available warp-stall counters between chips using the local ncu install
[verified, reproducible]:

```bash
NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
for c in ga100 gh100 gb100; do
  $NCU --query-metrics --chip $c | awk '{print $1}' \
    | grep '^smsp__warp_issue_stalled' | sed 's/_per_warp_active//' | sort -u > /tmp/$c.txt
done
comm -23 /tmp/gh100.txt /tmp/gb100.txt    # in Hopper, gone in Blackwell
```

Result:

| | |
|---|---|
| Only on `gh100`/`ga100` | `smsp__warp_issue_stalled_imc_miss`, `smsp__warp_issue_stalled_gmma` |
| Only on `gb100` | *(nothing)* |
| Common | 17 stall reasons |

This is corroborated by the shipped section definition
`/opt/nvidia/nsight-compute/2026.1.1/sections/WarpStateStatistics.section`, which guards
"Stall IMC Miss" with `Filter { MaxArch: CC_90 }` and "Stall GMMA" with
`MinArch: CC_90, MaxArch: CC_90` [verified, read the file].

**So: if a blog post tells you to look at `stall_imc_miss` on a B200, it is wrong — the
counter does not exist.** And there is no Blackwell replacement for `stall_gmma`, for the
architectural reason in §4.2.

### 7.2 The 17 stall reasons, verbatim + the concrete fix

Definitions below are **quoted exactly** from the Nsight Compute 2026.1.1 metric database on
this machine (`ncu --query-metrics --chip gb100 --csv`) [verified]. The "what it really means"
and "fix" columns are my analysis [inferred].

Metric name pattern (substitute `<reason>`):
- as a fraction of issue slots: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`
- as warp cycles per instruction: `smsp__average_warp_latency_issue_stalled_<reason>.ratio`
- as a fraction of resident warps: `smsp__warp_issue_stalled_<reason>_per_warp_active.ratio`

| Reason | NVIDIA definition (verbatim) | What it really means on SM100 | Fix |
|---|---|---|---|
| `long_scoreboard` | "waiting for a scoreboard dependency on L1TEX (local, global, surface, tex) operation" | Global/local load latency. The default bottleneck at batch 1. | More outstanding loads: unroll, use distinct barrier indices, `LDG.128`, `__ldg`. Improve locality so it's an L2 hit not HBM. Prefetch into SMEM with `cp.async`/TMA one stage ahead. |
| `short_scoreboard` | "waiting for a scoreboard dependency on MIO operation other than (local, global, surface, tex)" | Shared memory (`LDS`/`LDSM`), constant (`LDC`/`LDCU`), `SHFL`, `MUFU`, **and mbarrier `SYNCS` — i.e. tensor-core waits**. | Identify which. If mbarrier: more MMAs in flight / deeper TMA pipeline. If `LDS`: fix bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared*`). If `SHFL`: restructure the reduction. If `LDC`: hoist constant loads out of the loop. |
| `wait` | "waiting on a fixed latency execution dependency" | The `stall` field of the control word — compiler-scheduled ALU latency. | Pure ILP problem. Independent accumulators, unroll, break dependency chains. More warps also works but is the expensive fix. |
| `not_selected` | "waiting for the microscheduler to select the warp to issue" | The warp was *ready* but another warp won. **This is the good stall.** | If this dominates you have too many warps for the work, i.e. you are issue-bound. Reduce occupancy, increase per-thread work, or shorten the instruction stream. |
| `no_instruction` | "waiting to be selected for instruction fetch, or waiting on an instruction cache miss" | Fetch arbitration or i-cache miss — indistinguishable on Blackwell. | Shrink hot code (§6). Reduce unroll. Split megakernels. Improve basic-block locality. |
| `mio_throttle` | "waiting for a free entry in the MIO instruction queue" | Back-pressure from the shared/constant/special-function pipe. Too many `LDS`/`LDC`/`MUFU`/`SHFL` per unit time. | Convert shared loads to wider ones (`LDS.128`). Move constants to immediates or uniform registers. Replace `MUFU`-heavy math. Spread the pressure across more instructions. |
| `lg_throttle` | "waiting for a free entry in the LSU instruction queue" | Back-pressure on the *local/global* LSU queue. You are issuing global loads faster than the LSU can accept them. **Distinct from `long_scoreboard`**: this is stalling at the load, not at the consumer. | Fewer, wider loads (`LDG.E.128` instead of 4×32-bit). Coalesce. This is often a *good* sign — it means you have saturated the memory path. |
| `tex_throttle` | "waiting for a free entry in the TEX instruction queue" | Texture/surface pipe back-pressure. Rare in LLM inference. | N/A for us unless something is using texture fetches. |
| `math_pipe_throttle` | "waiting for an execution pipe to be available" | The FMA/ALU/FP64/tensor pipe is genuinely saturated. | This is the *good* stall for a compute-bound kernel. If unexpected: you are on the wrong pipe (e.g. FP64 or INT32 at half rate) — check `sm__inst_executed_pipe_*`. |
| `barrier` | "waiting for sibling warps at a CTA barrier" | `__syncthreads()` / `BAR.SYNC` — includes load imbalance *within* the CTA. | Fewer, later barriers. `__syncwarp` where warp scope suffices. Split into more CTAs so imbalance averages out. Named barriers (`bar.sync N`) to sync only the warps that need it. |
| `membar` | "waiting on a memory barrier" | `MEMBAR`/`FENCE`. `__threadfence()`, release/acquire, async proxy fences. | Weaken the scope: `.cta` instead of `.gpu` instead of `.sys`. This matters enormously for our multi-GPU AR path. |
| `drain` | "waiting after EXIT for all memory instructions to complete so that warp resources can be freed" | Tail of the kernel; stores still in flight. | Usually benign. If large: too many outstanding stores at exit, or the kernel is too short relative to its store traffic. |
| `dispatch_stall` | "waiting on a dispatch stall" | Structural hazard between the scheduler and a dispatch port. | Rarely actionable directly; usually a symptom of pipe saturation. |
| `sleeping` | "waiting for a nanosleep to expire" | `__nanosleep()` — typically inside a spin-wait backoff. | Tune the backoff in your spin loop. Relevant to our AR arrival-skew spin. |
| `branch_resolving` | "waiting for a branch target address to be computed, and the warp PC to be updated" | Indirect/computed branches, and long dependent branch chains. | Avoid function pointers / virtual dispatch in hot loops. Predication (`@P0`) instead of branching for short bodies. |
| `misc` | "waiting on a miscellaneous hardware reason" | Grab-bag. | Not actionable. |
| `selected` | "selected by the microscheduler to issue an instruction" | Not a stall — this is issue. | Maximise it. |

### 7.3 The order to read them in

NVIDIA's own guidance in the shipped section file is blunt and correct [verified, quoted from
`WarpStateStatistics.section`]:

> Stalls are not always impacting the overall performance nor are they completely avoidable.
> **Only focus on stall reasons if the schedulers fail to issue every cycle.**

So the first metric is not a stall reason at all:

```
smsp__issue_active.avg.per_cycle_active     # instructions issued per scheduler per active cycle
```

Ceiling is 1.0. If you are at 0.8, stall reasons are noise. If you are at 0.05 — which is
what a batch-1 GEMV looks like — then read the stall chart. And read
`smsp__warps_eligible.avg.per_cycle_active` next: if eligible warps ≈ 0, you are latency-bound
and need more in-flight work; if eligible warps > 1 while issue < 1, you are throughput-bound
somewhere.

---

## 8. ILP vs TLP: why low occupancy is often correct

### The folk wisdom is wrong and has been for 16 years

Volkov's *Better Performance at Lower Occupancy* (GTC 2010) is still the correct mental model
[reported — the paper is real and widely cited; I read the summary, not the original PDF, in
this session]. The argument is Little's Law and it is architecture-independent:

> concurrency needed = latency × throughput

Occupancy supplies concurrency through *warps*. ILP supplies the same concurrency through
*independent instructions inside one warp*. The hardware does not care which.

### The SM100 numbers

To saturate an SM you need `4L` independent instructions in flight (4 schedulers × L cycles).
Working the two regimes:

**Regime A — an FP32/FMA-bound kernel.** Assume ALU result latency L ≈ 4–5 cycles
(the CUDA guide states ~4 for CC 7.x; I have **no measured SM100 ALU latency**, so
[unverified] for Blackwell, and the production `stall` histogram is consistent with it —
mean 2.17, mode 1, with a 6.7% cluster at exactly 4).

- With **zero ILP** you need `4 × 4 = 16` warps per SM = **25% occupancy**.
- With **4 independent accumulators per thread** you need `16 / 4 = 4` warps = **6.25%
  occupancy**.

**Regime B — a tensor-core kernel.** A `tcgen05.mma` is a single issue slot with an
11–14 cycle single-instruction latency [reported, arXiv:2512.02189 Tables V/VI] and a
throughput that scales with tile size, not latency. The instruction stream around it is
mostly TMA issue + mbarrier waits. Here occupancy is nearly irrelevant, because:

- The MMA occupies **one issue slot** and then runs in the background (§4.2).
- Accumulators live in **TMEM**, not registers — Blackwell moved them out of the register file
  precisely so that a big MMA does not force high register pressure [reported].
- The kernel needs a *deep pipeline of TMA stages* (shared memory capacity) far more than it
  needs warps.

This is why CUTLASS SM100 GEMMs run 1–2 CTAs per SM with 128–256 threads and a 200+ KB shared
memory carveout: they trade every warp slot for pipeline depth. Achieved occupancy of
**6–12% is normal and correct** for these kernels [inferred from the resource math; the
carveout options up to 228 KB in Table 1 make >1 CTA/SM impossible for a deep-pipeline GEMM].

### When high occupancy IS the right answer

Be honest about the other side. High occupancy wins when:

- **Instruction streams are short and dependency chains are unavoidable.** Our **DSA indexer**
  (5.8% of profile) and elementwise kernels (3.7%) are this shape: little reuse, one or two
  loads, a bit of math, a store. There is nothing to unroll into. More warps is the only lever.
- **Divergence is high.** With divergent warps, effective per-warp ILP collapses; you need more
  warps to keep schedulers fed.
- **The kernel is bandwidth-bound and you need many concurrent misses.** More warps = more
  outstanding L2/HBM requests. Though note you can get the same from more loads per thread,
  at lower register cost per byte in flight.

### The decision procedure

```
1. ncu: smsp__issue_active.avg.per_cycle_active
     >= 0.6  -> issue-bound. Reduce instruction count. Occupancy is not the problem.
2. ncu: smsp__warps_eligible.avg.per_cycle_active
     ~ 0     -> latency-bound. Go to 3.
     > 1     -> you have enough warps; something downstream is throttling. Read throttles.
3. Which stall dominates?
     wait                -> add ILP (independent accumulators). Occupancy is the wrong fix.
     long_scoreboard     -> add MLP (more outstanding loads). Occupancy also works but costs regs.
     short_scoreboard    -> find out if it's mbarrier (tensor core) or LDS. Different fixes.
     not_selected        -> you already have too many warps. Reduce them.
4. Only if 3 says "no ILP/MLP available in this kernel" do you raise occupancy.
```

---

## 9. The decode regime: hiding latency when every GEMM is a GEMV

### 9.1 What batch 1 actually does to the arithmetic

At concurrency 1 with TP8, each rank's dense projections are `[1, K] × [K, N]` — a GEMV.
Arithmetic intensity is:

```
FLOPs   = 2·K·N
Bytes   = K·N·sizeof(w)  +  K·sizeof(x)  +  N·sizeof(y)
        ≈ K·N·sizeof(w)                       (weights dominate)
AI      = 2·K·N / (K·N·sizeof(w)) = 2 / sizeof(w)
```

| Weight dtype | Arithmetic intensity (FLOP/byte) |
|---|---|
| BF16/FP16 | 1.0 |
| FP8 | 2.0 |
| NVFP4 | 4.0 |

[verified — this is arithmetic, not a claim about hardware]

Compare to the machine balance. B200 HBM3e delivers ~8 TB/s [reported, NVIDIA spec] against
FP4 dense tensor throughput on the order of 7,700 TFLOPS [reported, arXiv:2512.02189
Table VI, measured at 7702.5 TFLOPS for FP4]. Machine balance ≈ **~960 FLOP/byte**. We are at
**1–4**. We are off the roofline knee by roughly **two and a half orders of magnitude**.

**Corollary that should govern every decision:** at batch 1 the tensor cores are irrelevant.
The kernel's job is to *move weights* and to *not stall between moves*. This is also why
NVFP4 helps us so much more than its FLOPs would suggest — it is a 2× bandwidth win over FP8
and a 4× win over BF16, and bandwidth is the only currency in this regime.

Also: it explains our profile. Dense GEMM at 37.1% with AI ≈ 2–4 is not spending time in the
tensor core; it is spending time in `long_scoreboard` waiting for weight bytes, plus per-kernel
launch/dependency overhead.

### 9.2 The complete latency-hiding toolkit for this regime

Ordered by expected value on our system.

---

**(1) Kill the launch/dependency chain — CUDA graphs.**

At 365 tok/s single-stream = 2.74 ms/token; at the measured 1597 MHz that is **4.38 M SM
cycles per token**. If a decode step is ~200 kernels and each carries 3–5 µs of launch +
dependency latency, that is 0.6–1.0 ms — **22–36% of the token budget in launch overhead
alone** [inferred; the per-launch cost is the standard 3–5 µs figure, which I could not source
for CUDA 13.2 on this driver — measure it]. CUDA graphs collapse the whole step into one
graph launch with device-side dependency resolution.

This is not optional; it is the single largest lever in the list. If we are already using
graphs (SGLang does by default for decode), then the remaining item is **graph capture
coverage** — any host round-trip (a `.item()`, a `cudaMemcpy` D2H, a Python branch on a device
tensor) breaks the graph and reintroduces the full latency. Audit for those.

---

**(2) Programmatic Dependent Launch (PDL) to overlap kernel tails and heads.**

PTX `griddepcontrol` [verified, PTX ISA §9.7.13.13, requires sm_90+]:

> `.launch_dependents` … signals that specific dependents the runtime system designated to
> react to this instruction can be scheduled as soon as all other CTAs in the grid issue the
> same instruction or have completed. The dependent may launch before the completion of the
> current grid.
> `.wait` … causes the executing thread to wait until all prerequisite grids in flight have
> completed and all the memory operations from the prerequisite grids are performed and made
> visible to the current grid.

CUDA-level API [verified, CUDA PG]:

```cuda
// producer kernel, after its last write to the data the consumer prologue needs:
cudaTriggerProgrammaticLaunchCompletion();

// consumer kernel: do all input-independent setup FIRST, then:
cudaGridDependencySynchronize();

// launch side:
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
```

The win: the consumer's CTAs get *scheduled* and run their prologue (address computation,
descriptor setup, TMA prefetch, shared-memory init) while the producer's tail CTAs are still
draining. In a decode step that is a long chain of small dependent kernels, this recovers the
launch latency of every link.

**Highest-value application for us: the all-reduce.** 19.6% of our time is in collectives and
**47% of that is rank arrival skew** — ranks sitting idle waiting for the slowest peer. That
is ~9.2% of total kernel time doing nothing. PDL lets the post-AR kernel begin its prologue
during the AR wait, and lets the pre-AR compute of the *next* layer overlap the AR tail.

---

**(3) Maximise outstanding loads (MLP) — the direct attack on `long_scoreboard`.**

Little's Law for memory: to sustain `B` bytes/s at latency `L` seconds you need `B × L` bytes
in flight. Per SM: 8 TB/s / 148 SMs ≈ 54 GB/s/SM; at a ~600–1000 ns HBM latency [reported,
order-of-magnitude only — **I have no measured B200 HBM latency**] you need roughly
**32–54 KB in flight per SM**. With 4 warps/SM that is 8–13 KB per warp, i.e. **16–27
concurrent 16-byte-per-lane `LDG.128`s per warp**.

You cannot get 27 loads onto 6 barriers as separate stages, but you *can* because barriers are
counters: group 4–6 loads per barrier index, use 4–5 indices, get 16–30 in flight. That is
precisely what the compiler did in my UNROLL=8 experiment (§4.1), reaching 6 loads across 3
indices with room to spare.

Concretely:
- **`LDG.E.128`** — always issue 16 bytes per lane. Use `float4`/`int4`/`uint4` or
  `__align__(16)` structs. Verified in our SASS as `LDG.E.128.CONSTANT`.
- **`__ldg` / `const T* __restrict__`** → `ld.global.nc` → `LDG.E.128.CONSTANT` [verified,
  the `.CONSTANT` suffix appeared as soon as I marked the pointers `const __restrict__`].
  This routes through the non-coherent read-only path. Weights are read-only for the whole
  decode; mark every weight pointer this way.
- **Unroll so that all loads of a stage issue before the first consumer.** Check it with the
  decoder: you want a run of `LDG` with `WAIT=000000`, then the consumers.
- **Do not let ptxas sink loads.** If it interleaves loads and consumers, increase unroll or
  use explicit temporaries held across the whole load block.

---

**(4) Get the weights into L2 and keep them there.**

L2 on B200 is per-die and large, but nowhere near 22 GB of per-rank weights. What *does* fit:
- The **DSA index / KV metadata** structures.
- **Small hot tensors**: norms, router weights, embedding slices, EAGLE draft-head weights.
- With **256 experts / 8 active**, the routed-expert weights are 1/32 of the MoE parameter mass
  per token — but the *identity* of the 8 changes per token, so they are not L2-resident across
  tokens. **However**, with EAGLE 3-1-4 speculative decoding the draft and verify passes hit
  the *same* expert set for the accepted prefix, so the verify pass can hit in L2 what the
  draft pass just pulled. Whether it does is an empirical question — measure
  `lts__t_sectors_lookup_hit` on the expert GEMMs across a draft/verify pair.

Mechanism [verified, CUDA PG]:

```cuda
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = weights;
attr.accessPolicyWindow.num_bytes = bytes;        // <= cudaDeviceProp::accessPolicyMaxWindowSize
attr.accessPolicyWindow.hitRatio  = 0.6f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
```

And per-instruction from PTX [verified, PTX ISA]:
`.L2::evict_normal | .L2::evict_first | .L2::evict_last`, plus prefetch hints
`.L2::64B | .L2::128B | .L2::256B` on `ld`. The `LTC128B` suffix we see in production SASS
(`LDGSTS.E.BYPASS.LTC128B.128`, `LDG.E.LTC128B.128`) is this mechanism in the wild [verified].

**The high-value inverse:** mark *streaming* data `evict_first` so it does not evict the
persistent weights. KV-cache reads during decode are pure streaming — they should never
displace anything.

---

**(5) Persistent kernels.**

One grid, `gridDim = numSMs × ctasPerSM`, launched once, looping over work items pulled from a
device-side queue. Eliminates per-kernel launch, keeps L1/SMEM warm across iterations, and lets
you software-pipeline *across* what used to be kernel boundaries.

Costs, honestly:
- You lose the implicit global barrier at kernel boundaries; you need grid sync
  (`cooperative_groups::grid_group::sync()` or a hand-rolled arrive/wait), which is itself a
  `membar` + spin.
- Register allocation becomes the max over all phases, so occupancy drops to the worst phase.
- **You will hit the 6-barrier limit** (§4.1) if you fuse too much into one warp.
- **You will hit i-cache** (§6) if you inline every phase.

Recommendation for our system: persistent **per-phase** (one persistent GEMM kernel, one
persistent attention kernel) rather than one persistent megakernel, and let CUDA graphs +
PDL handle the phase transitions. That gets most of the launch-elimination win without the
barrier and i-cache cliffs.

---

**(6) Eliminate host round-trips.**

Every `.item()`, `.cpu()`, `torch.nonzero()`, Python-level `if tensor.sum() > 0`, and every
dynamic shape recomputation is a full round-trip: kernel drain → D2H copy → host decision →
launch. At 2.74 ms/token, a single 10 µs round-trip is 0.4% of the budget; ten of them is 4%.
And they break graph capture, which costs far more than the round-trip itself. Audit the
decode path for synchronisation points with `nsys` — look for gaps on the CUDA API row
correlated with `cudaMemcpyAsync` + `cudaStreamSynchronize` pairs.

---

**(7) ILP inside the GEMV — independent accumulators.**

Attacks `stall_wait`. A naive GEMV reduction is one dependent FFMA chain: at ~4-cycle latency
that is 4 cycles per FMA, i.e. 25% of the FMA pipe with one warp. Four independent
accumulators reduced at the end gets you to ~1 cycle per FMA. Free — costs 3 extra registers.

**But note:** at AI ≈ 1–4 this is *not* the bottleneck. Do it because it is nearly free, not
because it will show up. If `stall_wait` is not in your top three, skip it.

---

**(8) Split-K / stream-K for skinny GEMMs.**

The real structural problem at batch 1 is **not enough CTAs to fill 148 SMs**. A `[1,K]×[K,N]`
GEMV with N=4096 and a 128-wide output tile gives 32 CTAs — 22% of the machine. Splitting the
K dimension across SMs and reducing (atomically or in a second pass) is how you use the other
78%. This directly trades a reduction pass for SM utilisation and is usually a clear win when
the machine is otherwise idle.

Check: `launch__grid_size` vs 148, and `sm__cycles_active.avg / sm__cycles_elapsed.max` — if
the second is well under 1.0 you have idle SMs, not a kernel problem.

---

**(9) Weaken memory fences.**

`membar` stalls are pure overhead. `__threadfence_block()` (`.cta`) is much cheaper than
`__threadfence()` (`.gpu`), which is much cheaper than `__threadfence_system()` (`.sys`).
The multi-GPU AR path is where `.sys`-scoped fences creep in. Every one you can narrow to
`.gpu` is latency removed from the critical path.

---

## 10. Profiling recipes — exact command lines

All flags verified against `ncu --help` from `/opt/nvidia/nsight-compute/2026.1.1/ncu` on this
box.

### 10.0 Profiling a live serving process — read this first

```bash
NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
```

- **`--clock-control none`.** The default is `boost`, which *locks the clocks* — on a
  production node that changes the behaviour of everything else running. Our GPUs already run
  at 1597 MHz under load vs 1965 MHz max; locking to boost gives you numbers that do not
  reproduce in production. Use `none` when profiling the live engine, `base` when you want
  run-to-run comparability of a microbenchmark.
- **`--pipeline-boost-state stable`** (the default) fixes the Tensor Core boost state. This is
  new in the Blackwell-era tools and it is why ncu tensor numbers can differ from nsys.
- **`--cache-control none`** to *not* flush caches between replay passes — mandatory when you
  are measuring L2 residency of weights, since the default `all` flushes and destroys exactly
  what you are trying to measure.
- **`-k` + `-c` always.** Never profile an unfiltered serving process; kernel replay on 11.3 M
  kernels will not finish.

### 10.1 "Is this kernel at roofline?" — Blackwell tensor-core kernel

```bash
$NCU --set roofline \
     --clock-control none --cache-control none \
     -k regex:'nvjet_sm100.*' -c 5 \
     --target-processes all \
     -o /tmp/roof_nvjet \
     python -m sglang.launch_server ...
```

`--set roofline` pulls in `SpeedOfLight_HierarchicalTensorRooflineChart`, which is the one you
want on SM100 [verified, `ncu --list-sets`]. Open the report and read the *tensor* roofline,
not the FP32 one — a UMMA kernel does approximately zero FP32 FLOPs.

The metrics behind it, if you want them raw:

```bash
$NCU --metrics \
sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp32_sparsity_off.sum,\
sm__ops_path_tensor_op_utcomma_src_fp4_dst_fp32_sparsity_off.sum,\
sm__ops_path_tensor_op_utchmma_src_bf16_dst_fp32_sparsity_off.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,\
gpu__time_duration.sum \
     --clock-control none -k regex:'...' -c 3 ./app
```

Those `utcqmma` / `utcomma` / `utchmma` metric names are **verified present on `gb100`**
(`ncu --query-metrics --chip gb100`). Note the naming: **`utcqmma` covers FP8 *and* FP6 *and*
FP4** (`src_fp4_fp6_fp8`), while **`utcomma` is the dedicated dense-FP4 path**
(`src_fp4_dst_fp32`). For our NVFP4 build you need **both** — check which one your kernel is
actually hitting, because it tells you whether you got the dedicated FP4 datapath or the
shared quad datapath.

Achieved AI = `(tensor ops) / (dram__bytes_read + dram__bytes_write)`. Plot against the
machine balance. At batch 1 you will be far left of the knee — that is the expected answer,
and it means **stop optimising the MMA**.

### 10.2 "Where are my cycles going?" — the stall triage

```bash
$NCU --section SpeedOfLight --section WarpStateStats --section SchedulerStats \
     --section Occupancy --section LaunchStats \
     --clock-control none \
     -k regex:'<kernel>' -c 3 --print-details all ./app
```

Or headless, for scripting:

```bash
$NCU --csv --clock-control none -k regex:'<kernel>' -c 3 --metrics \
smsp__issue_active.avg.per_cycle_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warp_latency_per_inst_issued.ratio \
  ./app
```

Every one of those metric names is **verified present on `gb100`**. Do **not** include
`imc_miss` or `gmma` — they will fail on Blackwell (§7.1).

Read order: `issue_active` first, `warps_eligible` second, stalls third.

### 10.3 Per-instruction attribution — which SASS line is stalling

```bash
# compile with -lineinfo so CUDA source correlates
$NCU --set full --import-source yes \
     --clock-control none -k regex:'<kernel>' -c 1 \
     -o /tmp/src_report ./app

$NCU --import /tmp/src_report.ncu-rep --page source --print-source sass | less
```

`--print-source` accepts `sass`, `ptx`, `cuda`, `cuda,sass` [verified]. Metric-to-source
correlation is only available in `sass` and `cuda,sass` views [verified, from `--help`].

The columns you care about in the SASS view are the per-instruction stall-reason samples
(`smsp__pcsamp_warps_issue_stalled_*` — these exist on `gb100`). Cross-reference the hot line
against the decoded control bits from §3 and you can see exactly which barrier it is waiting
on and which instruction allocated it.

### 10.4 Memory-level parallelism, statically, with no GPU

This is the cheapest useful measurement in the whole document and it needs no profiler and no
idle GPU:

```bash
$NVD -c -hex kernel.cubin > k.sass
python3 ctrl.py k.sass | awk '$4!="-"'    # every instruction that allocates a write barrier
```

Then eyeball the hot loop for the pattern in §4.1: a run of `LDG` with empty wait masks, then
consumers. If you see `LDG; consume; LDG; consume` you have MLP of 1 and you are guaranteed to
be `long_scoreboard`-bound no matter what the profiler says.

### 10.5 Is it launch overhead or kernel time?

`nsys` on the whole decode step, then compare:
- sum of kernel durations (the 11.3 M-kernel figure we already have)
- wall time of the step

The gap is launch + dependency + host. If the gap is >20% you are in graph/PDL territory
(§9.2 items 1, 2, 6) and no amount of SASS tuning will help.

### 10.6 Bank conflicts (when `short_scoreboard` says shared memory)

```bash
$NCU --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum,\
l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld.sum \
     --clock-control none -k regex:'<kernel>' -c 3 ./app
```

All verified present on `gb100`. Note `l1tex__data_bank_conflicts_type_arbitration` also
exists on Blackwell and is worth checking — it separates true bank conflicts from arbitration
losses.

---

## 11. What I could not verify

Stated plainly, because a wrong number is worse than no number.

1. **SM100 instruction latencies.** No published or measured table for FFMA/IMAD/LDS/LDG on
   SM100. The arXiv Blackwell paper covers tensor-core instruction latency only. The "~4
   cycles for most arithmetic on CC 7.x" figure in the CUDA guide is the closest thing and I
   have marked every use of it as [unverified] for Blackwell.
2. **B200 memory hierarchy latencies.** L1 hit, L2 hit, HBM miss in cycles — **not sourced**.
   The arXiv paper's headline "58% reduction in memory access latency… TMEM achieves 420 clock
   cycles… compared to Hopper's 1000-cycle global memory latency" is comparing a **TMEM
   access** against a **Hopper global-memory access**. Those are different things in different
   memory spaces. I would not repeat that number as "B200 memory latency"; it is at best a
   claim about TMEM. Flagging it as a **suspect widely-repeatable claim**.
3. **Instruction cache capacities (L0/L1I) on SM100.** Not published, not measured, and there
   is **no ncu counter on `gb100`** to infer it from.
4. **Register file bank structure on SM100.** Number of banks and the register→bank mapping:
   not sourced. The `.reuse` mechanism is verified to exist and be used; the bank layout is not.
5. **LSU / MIO queue depths** (the thing `lg_throttle` and `mio_throttle` measure back-pressure
   on). Not published.
6. **Arithmetic instruction throughput table for CC 10.0.** The CUDA 12.8 programming guide's
   Table 4 stops at CC 9.0, and CUDA 13.x moved the table to the Best Practices Guide. I could
   not retrieve a CC-10.0 column. The per-SM core counts in Table 1 imply FP32 128/clk,
   INT32 64/clk, FP64 64/clk, SFU 16/clk [inferred from core counts, not from a published
   throughput table].
7. **Per-kernel launch overhead on this driver/runtime.** The 3–5 µs figure is folklore; I did
   not measure it. It is measurable in five minutes on an idle GPU and it determines the value
   of items (1), (2) and (5) in §9.2.
8. **Whether the 6 barriers are per-warp or per-warp-slot.** Every source treats them as
   per-warp state; I did not find a primary confirmation for SM100.

### Measurements to run on this box when a GPU is free

The GPUs were at 89–100% utilisation with 165 GB resident per device throughout this session,
so I deliberately ran **zero GPU microbenchmarks** — any latency number I produced would have
been contaminated and I would have perturbed production. When a GPU is drainable:

| # | Measurement | Method | Answers |
|---|---|---|---|
| 1 | ALU/FFMA/IMAD result latency | dependent-chain microbenchmark, `%clock64` | the `L` in `4L`; validates the `stall` histogram |
| 2 | L1/L2/HBM load latency | pointer chase over increasing footprint | the MLP target in §9.2(3) |
| 3 | Max outstanding LDG per warp | issue N independent `LDG.128`, sweep N, find the knee | real MLP ceiling vs the 6-barrier theory |
| 4 | L0/L1 i-cache size | unrolled `NOP`/`FFMA` chain of growing size, watch `stall_no_instruction` | the megakernel budget in §6 |
| 5 | Kernel launch overhead | empty kernel, 10k launches, with and without CUDA graph | the value of §9.2(1) |
| 6 | L2 hit rate on expert GEMMs across a draft/verify pair | `lts__t_sectors_lookup_hit` with `--cache-control none` | whether EAGLE gives us free L2 reuse |
| 7 | `smsp__issue_active` on our top 4 kernels | §10.2 recipe | whether we are issue-bound or latency-bound, per kernel |

---

## Sources

**Read on this machine (primary, verified):**

- `/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc`, `ptxas` — CUDA 13.3.73, compiled test
  kernels for `sm_100a`
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvdisasm` —
  13.3.73, `--help` (arch list), `-c -hex` disassembly of all cubins below
- `/mnt/persistent/app-data/NotSglang/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump`
- `/opt/nvidia/nsight-compute/2026.1.1/ncu` — `--version`, `--list-chips`, `--list-sets`,
  `--list-sections`, `--help`, `--query-metrics --chip {ga100,gh100,gb100}` (with `--csv` for
  full metric descriptions). **This is the authoritative source for every stall-reason
  definition and metric name in §7 and §10.**
- `/opt/nvidia/nsight-compute/2026.1.1/sections/WarpStateStatistics.section` — the
  `MaxArch: CC_90` filters on IMC Miss and GMMA
- `/opt/nvidia/nsight-compute/2026.1.1/sections/SchedulerStatistics.section`,
  `ComputeWorkloadAnalysis.section`, `SpeedOfLight_HierarchicalTensorRooflineChart.section`
- `/opt/nvidia/nsight-compute/2026.1.1/docs/ProfilingGuide/index.html`,
  `docs/NsightCompute/index.html` (checked — the stall definitions are *not* in the shipped
  HTML, only in the metric database)
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/sgl_kernel/sm100/common_ops.abi3.so`
  — 51 `sm_100` cubins extracted; `common_ops.abi3.123.sm_100.cubin` (100 kernels, 89,584
  instructions) for the control-bit statistics in Table 3/4; `common_ops.abi3.11.sm_100.cubin`
  (99,544 instructions, 592 `UTC*`) for the tcgen05 analysis in §4.2
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13`
  — `cuobjdump -lelf` (5,449 embedded cubins across sm_75…sm_120). Note: the `nvjet_sm100_*`
  kernel names from our nsys profile do **not** appear as plain strings in this library, so I
  could not locate and disassemble `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` specifically.
- `nvidia-smi` on this node — B200 ×8, CC 10.0, 183359 MiB, driver 595.71.05, SM clock
  1965 MHz max / 1597 MHz observed under load, memory clock 3996 MHz, 1000 W limit

**Fetched and read (primary, web):**

- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html — §20.9
  "Compute Capability 10.0" (SM composition, 4 warp schedulers, 1 issue/cycle, shared memory
  carveouts); §8.2.3 (the `4L` latency-hiding equation, and its stale arch list); §
  `accessPolicyWindow`; `cudaTriggerProgrammaticLaunchCompletion` /
  `cudaGridDependencySynchronize` / `cudaLaunchAttributeProgrammaticStreamSerialization`
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html — Table 4
  "Throughput of Native Arithmetic Instructions" (confirmed: **no CC 10.0 column**)
- https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html — PTX ISA:
  `griddepcontrol` §9.7.13.13 (verbatim, sm_90+); `ld.global.nc` §9.7.9.9;
  `createpolicy`/`applypriority`/`discard`; `.L2::evict_*` and `.L2::{64B,128B,256B}` qualifiers
- https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html — 64 warps/SM, 65536 registers/SM,
  255 registers/thread, 228 KB shared memory, 227 KB per block, 32 blocks/SM, 256 KB combined
  L1+SMEM, cluster size 8 portable / 16 opt-in
- https://arxiv.org/html/2512.02189v1 — Jarmusch & Chandrasekaran, "Microbenchmarking NVIDIA's
  Blackwell Architecture: An in-depth Architectural Analysis". Read Tables IV (SASS mapping:
  DMMA/HMMA/OMMA/QMMA/IMMA), V (`tcgen05.mma` single-instruction latency 11.0–11.4 cycles vs
  `wgmma` 32–128), VI (per-precision latency 11.2–14.2 cycles, FP4 7702.5 TFLOPS,
  FP8 1925.3, FP16/FP32-accum 482.4). **Its TMEM-vs-Hopper-global-memory latency comparison is
  flagged as suspect in §11.**
- https://jianyuh.github.io/cuda/2026/04/12/blackwell-sm100.html — SM100 tensor roofline,
  SMEM-vs-math balance, in-flight MMA counts. [reported — single-author blog, not independently
  verified]

**Consulted via search summary only (not read in full — treat as [reported]):**

- Volkov, "Better Performance at Lower Occupancy", GTC 2010 —
  https://dmacssite.github.io/materials/volkov10-GTC.pdf
- https://www.thesoftwarefrontier.com/p/how-cuda-binaries-actually-work — corroborates that the
  Volta-era control-field layout decodes unchanged on sm_100/sm_110/sm_120. My §3 decode was
  derived independently on this box and agrees.
- https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
  — TMEM and `tcgen05` programming model, `cta_group::2`
