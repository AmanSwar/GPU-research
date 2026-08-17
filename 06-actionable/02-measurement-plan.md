# The measurement plan

**What this is.** The corpus at this point holds ~47,000 lines of researched
architecture and ~40 measured serving points. Between them sits a layer of
claims that are neither: they are *inferred* from a spec sheet, or *ranked by
cost* when the decision needs *opportunity*, or *measured once at an operating
point nobody serves*. This file turns that layer into an ordered experiment
programme for the eight B200s in this box, which are idle right now
(`nvidia-smi`: 0 MiB used, 120 MHz idle clock, 187–202 W, 29–33 °C, 2026-08-17)
`[verified]`.

Two classes, both required:

- **Class A — microbenchmarks on idle GPUs.** Establish hardware constants the
  rest of the corpus takes on faith. No server, no model, no serving stack.
- **Class B — engine experiments.** A/Bs against the serving stack, each
  designed to settle exactly one ranked opportunity.

**Every experiment below states five things**: what question it answers, the
exact tool and command, the result expected if current theory holds, what a
*surprising* result would mean, and the confound that has to be held still.
An experiment without a stated falsifier is not on this list.

**Labels.** `[verified]` = read from a primary source or measured on this box.
`[reported]` = a vendor asserts it. `[inferred]` = reasoning from the above.
`[unverified]` = plausible, unsourced. Every substantive claim carries one.

**Prerequisite reading**, in this order:
[`00-hardware/00-this-machine-ground-truth.md` §8](../00-hardware/00-this-machine-ground-truth.md),
[`benchmark/SCORECARD.md` §1d](/home/aman/code/benchmark/SCORECARD.md),
[`NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` §4–§7](/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md).
Nothing here may be run without first checking the ledger's "Tried, with
verdicts" table.

---

## 0. Three corrections found while writing this plan

These came out of reading the *resolved* `ServerArgs` in the run logs rather
than the flags we passed. All three change what is worth measuring, so they
come before the programme rather than inside it.

### 0.1 FlashInfer allreduce fusion has been ON for every latency-mode number

The standing claim across `SCORECARD.md`, the ledger §3 and this project's
framing is: *"Every measurement was taken with two-batch overlap, single-batch
overlap AND FlashInfer allreduce fusion all disabled."* **That is false for
allreduce fusion in latency mode.** `[verified]`

```
/home/aman/code/benchmark/runs/sweep-latency-3-1-4/server.log
  [2026-08-16 06:00:09] Auto-enabling FlashInfer AllReduce Fusion on
                        SM90/SM10X for GlmMoeDsaForCausalLM
  server_args=ServerArgs(... flashinfer_allreduce_fusion_backend='auto' ...)
```

The mechanism is `_flashinfer_allreduce_fusion_auto_enable` in
`python/sglang/srt/arg_groups/overrides.py:1786`, which fires when the arch is
in `_FLASHINFER_ALLREDUCE_FUSION_ARCHS` (it contains `GlmMoeDsaForCausalLM`,
line 1768), the part is SM90/SM100, `tp_size > 1`, `moe_a2a_backend == "none"`
and **`enable_dp_attention` is False**. `[verified — source read]`

Resolved state across the three modes we have run `[verified — grep of each
`server.log`]`:

| mode | `enable_dp_attention` | `flashinfer_allreduce_fusion_backend` | fusion |
|---|---|---|---|
| `latency-3-1-4` (TP8) | False | `'auto'` | **ON** |
| `capacity` (DP8) | True | `None` | off |
| `capacity-overlap` (DP8) | True | `'auto'` | ON (passed explicitly) |

Consequences:

1. The C1 profile's `oneshotAllreduceFusionKernel` — 32 CTAs, **12.0%** of
   decode kernel time, mean 10.12 µs
   ([`05-megakernels…` §2.4](../01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md))
   — *is* the fusion kernel. The C64 capacity capture instead shows
   `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` at 10.5%. The two captures were
   never running the same collective implementation. `[verified — the kernel
   names in the corpus's own two hotspot tables corroborate the flag read]`
2. The ledger's C1-vs-C64 hotspot comparison (§2a) therefore differs in **four**
   variables at once: concurrency (1 → 64), parallelism (TP8 → DP8+DP-attention),
   draft depth (3-1-4 → 2-1-3), *and* collective implementation (fused one-shot
   → NCCL ring). Conclusions of the form "MoE rises 1.6× with concurrency" are
   not supported by that pair. `[inferred — direct]`
3. "Measure allreduce fusion, never done" is the wrong experiment. The right one
   is the **reverse**: `--enforce-disable-flashinfer-allreduce-fusion` on
   latency mode, to price what fusion is already buying. That flag exists
   (`server_args.py:1973`) and is the only clean way to isolate it. `[verified]`

### 0.2 There is no EP8 — `ep_size=1` in every mode we have ever run

`ep_size=1` and `moe_a2a_backend='none'` in the resolved args of
`sweep-latency-3-1-4`, `sweep-capacity` and `sweep-capacity-overlap` alike.
`[verified]` The routed experts are **TP-sharded over the intermediate
dimension**, not expert-parallel.

The ledger's candidate C reads: *"Balance expert routing across ranks — uneven
expert load at EP8 is the most likely upstream cause"* of the rank skew. With
`ep_size=1` there is no EP8, and under TP-sharded experts **every rank computes
a slice of every selected expert on every token**, so token-routing imbalance
cannot by itself produce per-rank work imbalance in the MoE GEMM. `[inferred —
from the verified `ep_size=1` plus the standard TP-sharded MoE dataflow]`

That does not make skew unreal; it relocates the suspect list:

- At **latency/TP8/C1** (`dp_size=1`), all eight ranks execute an identical
  program on identical shapes. Residual skew must be host-side, clock/power, or
  an artefact of collective entry order. `[inferred]`
- At **capacity/DP8** (`dp_size=8`, `enable_dp_attention=True` `[verified]`),
  requests are partitioned across DP ranks, so *attention* work genuinely
  differs per rank while the MoE stays TP-uniform. That is a real and
  addressable imbalance — but it is a request-scheduling problem, not an expert-
  routing one. `[inferred]`

So the expert-distribution recorder (ledger phase 0b) is still worth running,
but as a *falsification* of routing imbalance rather than a search for it, and
the primary skew instrument becomes per-rank *phase* timing, not per-rank expert
counts. Experiment **B5** is written accordingly.

### 0.3 `TRTLLM_ENABLE_PDL` is already on by default

`python/sglang/srt/entrypoints/engine.py:1526` reads
`if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0": os.environ["TRTLLM_ENABLE_PDL"] = "1"`.
`[verified]` The megakernel document's cheapest-intervention table ranks
`TRTLLM_ENABLE_PDL=1` first, "one env var, one restart"
([`05-megakernels…` §10.3](../01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md)).
It is already in effect for every number we have. The measurable question is the
inverse — `TRTLLM_ENABLE_PDL=0` to price what PDL currently buys — which is
worth one run only as a sanity check, not as an optimization.

### What §0 changes, in one table

| corpus claim | status after §0 | replacement experiment |
|---|---|---|
| "all overlap machinery disabled" | **wrong for fusion in latency mode** `[verified]` | **B4a** — disable fusion and price it |
| "uneven expert load at EP8 causes skew" | **unsupported**, `ep_size=1` `[verified]` | **B5** — per-rank phase timing; routing recorder as falsifier |
| "`TRTLLM_ENABLE_PDL=1` is an untried cheap win" | **already on** `[verified]` | **B4f** — `=0` as a control only |
| "C64 says MoE ↑1.6×, collectives ↑" | **4-variable confound** `[inferred]` | **B2** — profile the competitive config under load |

---

## 1. Already measured — do not spend GPU-hours re-deriving

The assignment's Class-A list overlaps substantially with work already in the
corpus. Re-running it would be the most expensive way to learn nothing. This
table is the boundary: left column done, right column is what actually remains.

| question | already measured | source | what remains |
|---|---|---|---|
| HBM peak vs achieved | **6.98–7.28 TB/s sustained read** = 91–95% of the 7.67 TB/s pin rate; 6,753 GB/s on a 4 GiB grid-stride read = 88% | [`02-memory…` §2](../00-hardware/02-memory-hierarchy-and-caches.md), [`06-microbench…` §5](../00-hardware/06-microbenchmarks-and-reverse-engineering.md) `[verified]` | the **gather** pattern an MoE expert GEMM issues → **A1** |
| L1 / L2 / HBM latency | **39.4 / 303 / 762 cycles** (20.1 / 154.2 / 387.8 ns @1.965 GHz), randomised chase | [`02-memory…` §1](../00-hardware/02-memory-hierarchy-and-caches.md) `[verified]` | **peer/NVLink** latency, never measured → **A2** |
| L2 effective capacity | **~63 MiB for all-SM shared data, ~126 MiB for private slices**; two independent knees agree | [`02-memory…`](../00-hardware/02-memory-hierarchy-and-caches.md) `[verified]` | whether that is *two pools* or *one pool with a shared-line limit* → **A3** |
| L2 near/far by SM | per-SM spread 279–324 cy, die halves differ by **~6 cy** on an 8 MiB buffer; SM ids pair even/odd | [`06-microbench…` §4](../00-hardware/06-microbenchmarks-and-reverse-engineering.md) `[verified]` | the 8 MiB buffer averages near+far and cannot isolate it → **A3** |
| persisting-L2 window | `cudaAccessPolicyWindow` gave **+3% best, −5% when sized to 79 MiB**, against a synthetic 1 GiB stream | [`02-memory…` §4.4](../00-hardware/02-memory-hierarchy-and-caches.md) `[verified]` | never tried against a **real GLM-5.2 weight set under real KV traffic** → **A5** |
| NVLink bandwidth | peer read **771 GB/s** (86% of unidirectional), write 686, `cudaMemcpyPeer` 771 | [`06-microbench…` §5](../00-hardware/06-microbenchmarks-and-reverse-engineering.md) `[verified]` | small-message **latency** and the one-shot threshold → **A2**, **A4** |
| collectives: transfer vs wait | measured at **two** operating points; NVLink at **0.6–1.6%** inside collective kernels | ledger §2c `[verified]` | *why* ranks arrive apart → **B5** |
| AA workload shape | **349.5 tok/s C1 / 159.3 C10**, real data, 10k in, temp 0.6 | `SCORECARD.md` §1a `[verified]` | one sample vs AA's 72-h P50 → **B6** |
| C64 profile | taken: capacity mode, sharegpt, 3.6M kernels | ledger §2 `[verified]` (`SCORECARD.md` §4d is stale on this) | the **competitive** config under load is still unprofiled → **B2** |
| decode is graph-replayed | 99–100% of top kernels in CUDA graphs; NVTX module ranges fire 13×/module/20 s | ledger §2e `[verified]` | nothing — layer-level decode attribution is structurally unavailable |
| SM coverage / launch gaps | mean SM coverage **65.4%**; true launch gap only **1.9%** of wall; 1.53× packing ceiling | [`05-megakernels…` §2.5](../01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md) `[verified]` | convert coverage → fraction-of-roofline, needs ncu → **B3** |
| TMEM cost/geometry | alloc+dealloc **442 cy** regardless of columns; 1 CTA/SM with `tcgen05.alloc`; ~585 B/cy/SM read | [`02-memory…` §7](../00-hardware/02-memory-hierarchy-and-caches.md) `[verified]` | per-**kind** MMA issue rate → **A6** |

**Rule.** If a number is in the left column, cite it. Do not re-measure it
unless you can name the specific condition that invalidates it.

---

## 2. Measurement hygiene on this box

### 2.1 Clocks — what 1597 MHz actually is

The corpus repeatedly describes 1597 MHz as a "19% haircut" and speculates about
a hardware base clock or a B200 analogue of H200's 1830 MHz clamp
([`07-power…` §12.1](../00-hardware/07-power-clocks-thermals-and-determinism.md)).
It is neither. **1597 MHz is the harness's own 80th-percentile heuristic.**
`[verified — computed on this box just now]`

```
nvidia-smi -i 0 --query-supported-clocks=graphics  →  247 bins, 1965 … 217 MHz
gpubench/env.py:pick_sustainable_clock(clocks, percentile=0.80)
    ordered = sorted(clocks); idx = round(0.80 * (247-1)) = 197 → 1597 MHz
```

So the 19% is self-inflicted by a percentile choice, not imposed by silicon.
That does not make the choice wrong — the docstring's reasoning (the top boost
bin drops mid-run and produces an unholdable number) is sound, and this box has
**90.9 s of accumulated SW power capping** `[verified]`. But it does mean the
question "what does unlocking cost and buy" is answerable in an afternoon, and
it should be answered (**A7**, **B1**).

**Rules.**

- **Every comparison run locks to 1597 MHz.** `gpu-bench run` does this via
  `sudo -n nvidia-smi -i <n> -lgc 1597,1597`; passwordless sudo works here
  `[verified]`. `RunSpec.identity()` hashes the SM clock, so `gpu-bench diff`
  refuses to compare across locks `[verified — `README.md`, `config.py`]`.
- **Unlocked runs are a separate labelled axis**, never mixed into a table with
  locked ones.
- **Any unlocked run samples clock and throttle reason continuously**, at ≥10 Hz,
  via `nvmlDeviceGetCurrentClockFreqs` + `nvmlDeviceGetCurrentClocksEventReasons`
  (both available through the bundled pynvml at
  `NotSglang/python/sglang/multimodal_gen/third_party/pynvml.py` `[verified]`).
  Reporting a mean over an unlogged mixture of clock states is how a boost
  experiment lies to you.
- **Validate the lock from the trace, not the command.** The C1 capture read GPC
  clock 1592–1605 MHz against the 1597 lock `[verified — ledger §2d]`. Do that
  every time; it is free and it retroactively certifies the whole table.

### 2.2 How many repeats a 4%-variance signal needs

`SCORECARD.md` §1d states run-to-run variance as ~4%, from *two* samples of the
same config (26,454 and 27,593 tok/s) `[verified]`. **Two samples is a range,
not a standard deviation**, and the entire decision framework rests on it. Fix
that first (**B0**).

Treating σ = 4% of the mean, for a two-sided test at α = 0.05 with 80% power,
n per arm ≈ 15.7·(σ/d)²:

| effect size d | in σ | runs per arm | verdict |
|---:|---:|---:|---|
| 4% | 1.0σ | **16** | not worth chasing — 32 runs to see it |
| 6% | 1.5σ | **7** | expensive |
| 8% | 2.0σ | **4** | the practical detection floor |
| 12% | 3.0σ | **2** | comfortable |
| ≥20% | ≥5σ | 1 + 1 confirm | obvious, still confirm once |

**Operating rules.**

1. **Anything below 8% is reported as "not resolved at n=3"**, never as a win or
   a loss. The ledger's SBO result (−12.5%, ledger §4 #9) clears this bar; a
   hypothetical +5% would not.
2. **Interleave arms A/B/A/B**, never all-A-then-all-B. Thermal state, page
   cache and radix-cache contents all drift monotonically over a session, and
   blocked designs alias that drift onto the treatment.
3. **Pair on the prompt set.** Same seed, same sampled requests both arms, and
   compare per-request TPOT pairwise. This removes between-sample variance and
   typically buys back a factor of ~2 in n. `[inferred — standard paired design;
   the effect size on this harness is itself worth measuring in B0]`
4. **Report the interval, not the point.** Every row in the ledger gets
   `mean ± sd (n)`.

### 2.3 Warm vs flushed cache

`Workload.flush_cache` emits `--flush-cache` `[verified — `config.py`]`. The
ledger's row #7 is explicit: prefix caching is worth **1.54×** on realistic
traffic (27,593 → 40,794 tok/s at 54.2% hit) and *"do not benchmark task
workloads with `--flush-cache`"* `[verified]`.

But AA's headline is a **P50 over 72 hours sampled 8×/day** `[verified —
ledger §1]`, which necessarily mixes cold and warm arrivals. So:

- **Decision runs are warm** (no flush), because that is what production does
  and what the 1.54× lives in.
- **Every run records the achieved prefix-cache hit rate** — `gpu-bench report`
  already surfaces it `[verified — `README.md`]` — and a run whose hit rate
  differs by more than a few points from its comparison partner is not a
  comparison.
- **Never compare a flushed run to a warm one.** They differ by more than any
  optimization on the candidate list.
- The cold-start number is measured **once**, separately, as the pessimistic end
  of the P50 band (**B6**).

### 2.4 Real vs synthetic input

Measured twice on this box, and it is not a small effect `[verified]`:

| | synthetic (`random`) | real (`sharegpt`/AA-real) | inflation |
|---|---:|---:|---|
| `repo-baseline` C1, tok/s | 447.8 | 365.5 | **+22%** |
| accept length | 4.00/4 | 3.16/4 | — |
| AA 10k shape C1, tok/s | 441.9 | 349.5 | **+26%** |
| AA 10k shape **C10**, tok/s | **77.2** | **159.3** | **−52%** — the sign flips |

The C10 row is the important one. Synthetic input does not merely inflate; it
**inverts the shape of the concurrency curve**, because ten 10k-token prompts of
incompressible random tokens is a worst-case prefill with zero cache reuse
`[verified — `SCORECARD.md` §1a]`.

**Rule: real data for every decision.** `repo-baseline` is retained
calibration-only, so published third-party figures on random tokens remain
reproducible, and its numbers are not reported `[verified — `README.md`]`.
Any microbenchmark that feeds a tensor core **also uses random data, not zeros**
— zeros understate power ~40% and overstate throughput ~9% on Hopper
`[reported — S3, via [`06-microbench…` §13.5](../00-hardware/06-microbenchmarks-and-reverse-engineering.md)]`.

### 2.5 Profiler overhead and the traps already paid for

Under nsys the observed pace was 9.61 ms per target-forward group against
~8.66 ms implied by the unprofiled 365 tok/s — **~10% profiler overhead**
`[verified — [`05-megakernels…` §2.5](../01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md)]`.
`gpu-bench run` already runs a clean pass then a profiled pass for this reason
`[verified — `run.py`]`. **Never quote a timing from a profiled pass.**

Traps this project has already hit and paid for — do not re-walk them
`[verified — ledger §4 "Measurement-infrastructure fixes" and `profilers/nsys.py`]`:

| trap | cost paid | the rule now |
|---|---|---|
| unbounded nsys capture | 3.8 GB trace → **15 GB SQLite** | bound the capture to a **20 s** window with `nsys start/stop` on a live session |
| O(kernels × ranges) NVTX projection | had not finished after **57 min** | projection is a sweep; aggregation is SQL (11.3M kernels in **5.5 s**) |
| `--nccl-trace=true` | rejected; takes the whole launch down | `--nccl-trace=coll,gpu,kernel-launch` — it is a comma list, not a boolean |
| `--sample` on `nsys launch` | app silently never starts | `--sample` belongs on `nsys start`, with `--output`/`--force-overwrite` |
| missing `--cuda-graph-trace=node` | decode timeline is one opaque blob | always set it; decode is ~100% graph replay |
| `TMPDIR` defaulting to `/tmp/nvidia` | root-owned on this box | `nsys_env(tmpdir=…)` points elsewhere |
| separate nsys sessions per rank | no shared time base → skew is meaningless | **one** session wrapping all 8 ranks |
| classifier keyed on the word "moe" | reported MoE at 4.9% when it was 19.4% — a **4× under-report** | classify NVFP4 expert GEMMs by `bmm_E2m1_*` |

`ncu` serialises kernels (10–100× slowdown) and must never be pointed at a run
whose timings are reported; it needs its own server instance
`[verified — `profilers/ncu.py`]`. It works unprivileged here because the driver
was reloaded with `NVreg_RestrictProfilingToAdminUsers=0` `[verified — same
file]`. Binary at `/opt/nvidia/nsight-compute/2026.1.1/ncu` `[verified]`.
Always pass `--launch-skip` — the first launches include JIT and autotuning
`[verified]`.

**Structurally unavailable, stop asking for it:** per-layer attribution in
decode. Python module forwards do not execute during graph replay, so
`--enable-layerwise-nvtx-marker` yields 13 fires per module across a 20 s
capture — the graph *capture* passes `[verified — ledger §2e]`. Getting it would
require NVTX ranges captured *into* the graph, which is an engine change.

### 2.6 Microbenchmark traps already paid for

From [`06-microbench…` §13](../00-hardware/06-microbenchmarks-and-reverse-engineering.md)
`[verified — learned on this box]`:

- **Randomise the pointer chain.** A sequential chain understated HBM latency by
  ~80 cycles and hid the TLB effect entirely.
- **Do not bracket individual accesses with `clock64`** on SM100 — overhead
  dominates. Time a full lap.
- **Guard against dead-code elimination unconditionally.** Three kernels
  silently reported 0 cycles because the result fed a dead branch. Write the
  final value to a global sink from thread 0 of each block with no predicate,
  and assert the result is non-zero.
- **Sweep the unique-cache-line count, not the region size.** A 4 GiB region at
  1 MiB stride is an L2 benchmark wearing a DRAM benchmark's clothes.
- **Any buffer under ~110 MB is L2-resident.** With 126.5 MiB of L2, a 64 MiB
  "HBM bandwidth" test on this box reads **9.59 TB/s** — above the 7.67 TB/s pin
  rate. That is almost certainly the origin of at least one published >8 TB/s
  B200 figure.

Toolchain: `/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc` (13.3.73), and every
kernel that touches `tcgen05` or block-scaled MMA must build for **`sm_100a`**
`[verified — [`00-this-machine…` §1](../00-hardware/00-this-machine-ground-truth.md)]`.
`ninja` is **not installed**, which blocks FlashInfer's JIT path
`[verified — `shutil.which('ninja')` → None]`; hand-written `.cu` compiled with
`nvcc` directly is unaffected.

### 2.7 One variable per run

The ledger's row #9 changed **two** flags (`--enable-single-batch-overlap` and
`--enable-flashinfer-allreduce-fusion`) and lost 12.5%; the fusion half is
permanently confounded `[verified — ledger §4]`. Every entry in §6 changes
exactly one resolved-`ServerArgs` field, and **verification is by diffing the
resolved `server_args=ServerArgs(...)` line in the two logs**, not by diffing
the command lines. §0.1 exists because those two are not the same thing.

---

## 3. The programme, ordered by information per hour

GPU-hours are the scarce resource; wall-clock includes ~2.5 min of server start
per engine run (`sweep-latency-3-1-4/server.log`: 03:04:28 launch → 03:06:50
ready `[verified]`) and a clean + profiled pass where profiling is on.
All durations `[inferred]`.

| # | id | experiment | class | GPU-h | wall-clock | unblocks |
|---:|---|---|---|---:|---|---|
| 1 | **Z1** | Did fusion / PDL / NUMA-bind actually apply? Resolved-args audit of every historical run | 0 | 0 | 30 min | corrects §0; re-labels the whole ledger |
| 2 | **Z2** | The 24.8% idle budget, from traces already on disk | 0 | 0 | 2 h | ranking slots A, B |
| 3 | **Z3** | Does anything set an L2 eviction policy on the KV path? (source read) | 0 | 0 | 1 h | feeds **A5** |
| 4 | **Z4** | Indexer-KV over-allocation: 78 layers allocated, 22 used | 0 | 0 | 1 h | a new ranking slot |
| 5 | **Z5** | Why is `ncclDevKernel_AllReduce` the only top-10 kernel at 0% graph capture | 0 | 0 | 2 h | ranking slot B |
| 6 | **A7** | Clock/power ceiling on one idle GPU: does `-lgc 1965` take, what is sustained | A | 0.5 | 45 min | **B1** |
| 7 | **B0** | The variance floor: 6 identical runs → a real σ | B | 2 | 2.5 h | **everything below** |
| 8 | **B1** | Unlocked-clock A/B on the engine | B | 3 | 3 h | a *new* top-tier ranking slot |
| 9 | **B2** | Profile the competitive config under load (`latency-3-1-4`, aa-10k-real, C10) | B | 1.5 | 1.5 h | the ranking's premise |
| 10 | **B3** | ncu the top three kernels → fraction-of-roofline | B | 3 | 4 h | ranking slots D, E; gates all Loop K |
| 11 | **A1** | HBM under the MoE gather pattern | A | 1 | 2 h | the MoE roofline denominator |
| 12 | **B4** | Seven never-measured flag flips, one at a time | B | 12 | 2 days | ranking slots H + new |
| 13 | **B5** | Rank-skew attribution: phase timing, NUMA, per-rank clocks | B | 4 | 6 h | ranking slots A, C |
| 14 | **A4** | Small-message all-reduce: size sweep, NVLS vs ring vs trtllm | A | 2 | 4 h | ranking slot B; the NVLS flag |
| 15 | **A5** | Does the persisting-L2 window hold a real weight working set | A | 2 | 4 h | a new ranking slot |
| 16 | **A3** | Is the 126.5 MiB L2 one pool or two per-die pools | A | 1 | 3 h | CTA-placement work; low direct value |
| 17 | **A2** | Peer-NVLink pointer-chase latency and the ping-pong floor | A | 0.5 | 2 h | calibrates **A4**'s model |
| 18 | **B6** | AA repeat protocol: n=8 over 24 h + chat-template variant | B | 8 | 2 days | the headline's credibility |
| 19 | **A6** | `tcgen05` MMA issue rate per block-scaled kind | A | 2 | 2–3 days | the ceiling denominators in **B3** |

**Why this order.** Items 1–5 cost **zero GPU-hours** and two of them
(**Z1**, **Z2**) change how existing measurements are interpreted, which is the
cheapest information in the programme. **B0** comes before every A/B because
without a real σ each subsequent result is a coin flip dressed as a
measurement. **B2** and **B3** come before the flag sweep because the flag sweep
should be aimed, and right now the ranking is aimed by a confounded profile
(§0.1) at kernels with no measured headroom. **A6** is last despite being
intellectually the most interesting: it is days of PTX work and its output only
tightens a denominator that **B3** can bound more cheaply.

---

## 4. Tier 0 — costs no GPU time

### Z1 — Resolved-args audit of every historical run

**Question.** For each of the ~40 measured points, what did SGLang *actually*
resolve, as opposed to what we passed?

**Method.** Every `runs/*/server.log` contains one
`server_args=ServerArgs(...)` line with the fully resolved configuration
`[verified]`. Parse it for all runs, diff against the mode's declared flags in
`gpubench/config.py`, and emit a table of every field that differs.

```bash
cd /home/aman/code/benchmark/runs
for f in */server.log; do
  echo "== $f"
  grep -o "server_args=ServerArgs(.*)" "$f" | head -1
done > /tmp/claude-1000/-home-aman-code/930438ff-5f3c-49e6-a3d9-2663231246c6/scratchpad/resolved-args.txt
```

Then diff the fields that matter for the ledger's conclusions:
`flashinfer_allreduce_fusion_backend`, `enable_dp_attention`, `dp_size`,
`ep_size`, `moe_a2a_backend`, `enable_nccl_nvls`, `enable_symm_mem`,
`num_continuous_decode_steps`, `max_running_requests`, `page_size`,
`speculative_attention_mode`, `disable_shared_experts_fusion`,
`cuda_graph_config`.

**Expected if theory holds.** Nothing new — the flags we passed are the flags
that ran.

**What a surprise means.** §0.1 already found one: it is not. Known deltas so
far `[verified]`, each of which needs a ledger footnote:

- `flashinfer_allreduce_fusion_backend='auto'` in latency mode (§0.1).
- `max_running_requests=48` in `sweep-latency-3-1-4`, against 256 in capacity.
- `cuda_graph_config`: decode `backend='full'`, **prefill `backend='disabled'`**
  — prefill runs without CUDA graphs, which is a TTFT lever nobody has priced.
- `disable_shared_experts_fusion=True` — the shared expert is not fused.
- `enable_nccl_nvls=False` on a fabric with `MULTICAST_SUPPORTED=1`.
- `enable_deepseek_v4_fp4_indexer=False` — an FP4 indexer path exists, unused,
  while the DSA indexer is 5.8% of C1 GPU time.
- `enable_fused_qk_norm_rope=False`, `enable_fused_moe_sum_all_reduce=False`.

**Output.** A footnote on every affected ledger row, and 4–5 new **B4** arms.

**Confound.** None — this is a log read.

### Z2 — Where the 24.8% idle actually goes

**Question.** Device 0 is busy 83.7% of wall at C1 and 75.2% at C64
`[verified — ledger §2c, §0c]`. A quarter of wall clock at serving concurrency
has no kernel resident. Shortening kernels cannot touch it.

**Method.** Over the traces already on disk
(`runs/sweep-latency-3-1-4/trace.sqlite`, `runs/sweep-capacity/`), run nsys's
`cuda_kern_exec_sum` (launch time vs execution time) and `gpu_gaps`, plus the
existing SQL gap analysis. `gpu-bench deepdive <trace>` already does the
collective decomposition `[verified — `cli.py`]`. No server, no new capture.

```bash
nsys stats --report cuda_kern_exec_sum \
  /home/aman/code/benchmark/runs/sweep-latency-3-1-4/trace.nsys-rep
nsys stats --report gpu_gaps \
  /home/aman/code/benchmark/runs/sweep-latency-3-1-4/trace.nsys-rep
/home/aman/code/NotSglang/.venv/bin/gpu-bench deepdive \
  /home/aman/code/benchmark/runs/sweep-latency-3-1-4/trace.sqlite
```

**Expected if theory holds.** The idle splits across three named buckets:
the un-graphed AllReduce paying launch latency outside the graph; rank skew
(a rank waiting appears as idle on every *other* rank); and step-boundary work
the overlap scheduler could not hide (sampling, detokenization, admission).
Note the true *launch* gap is already known to be only **1.9%** of wall
`[verified — `05-megakernels…` §2.5]`, so most of the 24.8% must be the first
two, not classic launch overhead.

**What a surprise means.** If the idle is concentrated in one contiguous region
per step rather than spread across step boundaries, it is a serialised
host round-trip, and the fix is the conditional-node decode loop rather than
anything collective-shaped.

**Duration.** 2 h. **Unblocks** ranking slots A and B — it promotes or demotes
them rather than adding work.

### Z3 — Does anything set an L2 eviction policy on the KV path?

**Question.** KV traffic is streaming and single-use; weights are re-read every
token. Without `evict_first`/`no_allocate` on the KV loads, L2 optimises for
exactly the wrong tenant `[inferred — [`00-this-machine…` §3](../00-hardware/00-this-machine-ground-truth.md)]`.

**Method.** Source read, no GPU. Grep FlashInfer's DSA/MLA kernels and SGLang's
memory pool for `evict_first`, `evict_last`, `no_allocate`, `createpolicy`,
`L2::cache_hint`, `access_property`, `__ldcs`, `.nc`. Start at
`NotSglang/.venv/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/`
and `NotSglang/python/sglang/srt/mem_cache/memory_pool.py`.

**Expected if theory holds.** Nothing sets a policy; every load is default.

**What a surprise means.** If a policy *is* set, **A5**'s design changes — the
weight-protection experiment then has to fight an existing hint rather than
fill a vacuum.

**Known constraint to carry into A5** `[verified — ptxas 13.3.73,
[`02-memory…` §4.1](../00-hardware/02-memory-hierarchy-and-caches.md)]`:
`.L2::evict_first`/`.L2::evict_last` on `ld`/`st` require a **256-bit vector
type** (`.v8.b32` or `.v4.b64`) on sm_100. Narrower forms are rejected. Per-
instruction policy on a `.v4.f32` load must go through `createpolicy` +
`.L2::cache_hint`. This is in no blog post and silently blocks the obvious
implementation.

### Z4 — The indexer-KV over-allocation

**Question.** `DSATokenToKVPool._create_index_buffers` allocates one
`index_k_with_scale_buffer` **per layer, for all 78 layers**, while only 22
carry indexer weights (21 non-MTP + MTP) `[verified — [`05-models/00-local-weights…` §9.4](../05-models/00-local-weights-ground-truth.md)]`.

```
allocated : 132 B × 78 layers = 10,296 B/token
needed    : 132 B × 22 layers =  2,904 B/token
wasted    :                      7,392 B/token = 7.39 GB per 1M tokens per GPU
```

That is 12% of the FP8-KV per-token cost and ~59 GB aggregate across 8 ranks at
a 1M-token cache `[inferred — arithmetic on verified sizes]`. Nothing in
`mem_cache/` is aware of `skip_topk`.

**Method.** Confirm from a live startup log: read the reported
`max_total_num_tokens` and the pool cell size, and check against the computed
61,464 B/token. `sweep-latency-3-1-4` reports `max_total_num_tokens=1744128`
`[verified]` — enough to do the arithmetic without a new run. Then read
`enable_dsa_cache_layer_split` (`server_args.py:1116`, currently `False`
`[verified]`) to see whether the fix already exists behind a flag.

**Expected if theory holds.** The arithmetic reproduces, and reclaiming it
raises `max_total_num_tokens` by ~1.7× on the indexer term alone.

**What a surprise means.** If `enable_dsa_cache_layer_split` already handles it
and is merely gated on `is_deepseek_dsa(hf_config)` (`server_args.py:4963`
`[verified]`), the fix is a predicate change, which is the cheapest item in the
entire programme.

### Z5 — Why is AllReduce the only top-10 kernel outside a CUDA graph?

**Question.** `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` is 10.5% of C64 GPU
time at **0%** graph capture, while AllGather and ReduceScatter in the same
trace are 100% captured `[verified — ledger §2b]`.

**Method.** Source read plus the existing trace's `graphId` column. Note the
prior from §0.1: the two captures used *different* collective implementations,
so first establish whether the un-graphed AllReduce is the DP-attention path's
NCCL call specifically. Read SGLang's graph-capture exclusion logic and NCCL's
graph-capture requirements (see
[`05-megakernels…` §8.7](../01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md)).

**Expected if theory holds.** A specific call site outside the captured region,
or an NCCL configuration that makes the op non-capturable.

**What a surprise means.** If it is capturable and simply is not captured, this
is a small patch against 10.5% of GPU time — the best cost/benefit in the
ledger. If it is structurally non-capturable, ranking slot B is closed with a
recorded negative.

---

## 5. Class A — microbenchmarks on idle GPUs

All of these run with `CUDA_VISIBLE_DEVICES` set to a single idle GPU unless
stated, build with
`/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc -O3 -arch=sm_100a`, and follow
§2.6. Sources go in the session scratchpad alongside the existing `mem*.cu`,
`nf2.cu`, `p2p.cu`, `tmem_bench.cu` set `[verified — those files exist]`.

### A1 — HBM under the gather pattern an MoE expert GEMM actually issues

**Question.** Contiguous streaming reads reach 6.98–7.28 TB/s `[verified]`. The
MoE decode step does not stream contiguously: it selects **8 of 256** experts
and reads each one's weight block. What does *that* achieve? Every MoE roofline
in the corpus currently divides by the contiguous number.

**Why it matters.** The MoE expert GEMMs are 30.8% of C64 GPU time
`[verified — ledger §2a]` and the corpus's own arithmetic puts GEMM2 at
**14.4% of "the roofline"** `[verified — [`03-moe…` §8.3](../01-kernel-optimization/03-moe-kernels-and-expert-parallelism.md)]`.
If the achievable gather bandwidth is materially below 6.98 TB/s, that 14.4%
is understated and the headroom claim behind ranking slot D shrinks.

**Shapes, from the checkpoint** `[verified — [`05-models/00-local-weights…`](../05-models/00-local-weights-ground-truth.md)]`:
hidden 6144, `moe_intermediate` 2048, 256 routed experts, top-8. Under TP8 each
rank holds 2048/8 = 256 intermediate rows per expert, so per expert per rank
`3 × 256 × 6144 = 4.72M` values. At NVFP4 (0.5 B/value + one E4M3 scale per 16
values = 0.5625 B/value) that is **~2.65 MB per expert per rank**, and 8 hot
experts is **~21.2 MB** — consistent with the corpus's "20.25 MiB" figure
`[verified — [`02-memory…`](../00-hardware/02-memory-hierarchy-and-caches.md)]`.
The full 256-expert table is **~679 MB**, comfortably beyond L2's 126.5 MiB, so
this is a genuine HBM test (§2.6).

**Kernel sketch** (`moe_gather.cu`):

```cuda
// 256 expert blocks of BYTES_PER_EXPERT; each iteration reads 8 chosen blocks.
// Sweep: (a) 8 random distinct ids, (b) 8 contiguous ids, (c) one 8x-sized
// contiguous block = the streaming control.
__global__ void expert_gather(const uint4* __restrict__ base,
                              const int*   __restrict__ sel,      // [8]
                              size_t vec_per_expert,
                              uint4* __restrict__ sink) {
  uint4 acc = make_uint4(0,0,0,0);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int e = 0; e < 8; ++e) {
    const uint4* p = base + (size_t)sel[e] * vec_per_expert;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < vec_per_expert; i += stride) {
      uint4 v = __ldg(p + i);                 // widen to ld.global.nc.v8.b32
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = acc;   // DCE guard, unpredicated
}
```

Launch 1184 blocks × 512 threads (8 waves over 148 SMs), as `mem3.cu` did.
Re-randomise `sel` per iteration on device. Time ≥50 laps, report median.

**Command.**
```bash
/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc -O3 -arch=sm_100a \
  -o /tmp/claude-1000/-home-aman-code/930438ff-5f3c-49e6-a3d9-2663231246c6/scratchpad/moe_gather \
  moe_gather.cu
CUDA_VISIBLE_DEVICES=0 ./moe_gather --experts 256 --bytes 2650000 --topk 8
```

**Expected if theory holds.** Within ~10% of the contiguous 6.98 TB/s. Each
expert block is 2.65 MB — thousands of contiguous 128 B lines — so the gather is
coarse-grained and the DRAM page locality should be nearly as good as streaming.
`[inferred]`

**What a surprise means.** If it lands materially below (say <5.5 TB/s), the MoE
roofline denominator in `03-moe…` is wrong and every "fraction of roofline"
there is understated; ranking slot D gets *less* attractive, not more, because
the kernel is closer to its real ceiling than we thought. If it lands *above*
6.98 TB/s, suspect L2 residency — 8 × 2.65 MB = 21.2 MB fits in L2, so the
random-`sel` arm must re-randomise often enough to defeat reuse. Check by
sweeping the number of distinct experts touched per lap from 8 to 256.

**Duration.** 2 h including build. **GPU-h** 1.

### A2 — Peer-NVLink latency and the small-message floor

**Question.** NVLink *bandwidth* is measured (771 GB/s peer read, 86% of
unidirectional `[verified]`). NVLink **latency** is not measured anywhere in
this corpus or, per
[`05-nvlink5…` §8.1](../00-hardware/05-nvlink5-nvswitch-and-collectives.md),
in the public literature. Everything in that document's one-shot/two-shot
analysis hinges on an unmeasured constant `L`.

**Method, two parts.**

1. **Pointer chase over a peer mapping.** Enable peer access 0→1, build a
   randomised chain in device-1 memory, chase it from device 0, one warp, time a
   full lap (§2.6). Extend `p2p.cu`.
2. **Ping-pong through NVSwitch.** Two GPUs, a 16 B flag in peer memory,
   `st.global.release.sys` / spin on `ld.global.acquire.sys`, timed with
   `%globaltimer` (nanosecond, host-comparable). N=10,000 round trips; report
   half the median as one-way.

**Expected if theory holds.** Peer load latency in the low microsecond range
and well above the 762-cycle (388 ns) local HBM figure; ping-pong one-way of a
few µs. `[unverified — this is precisely the number nobody has]`

**What a surprise means.** If one-way latency is small enough that
`8 × L` is negligible against the measured 8.2 µs "transfer" time inside our
collectives `[verified — [`05-nvlink5…` §8.3](../00-hardware/05-nvlink5-nvswitch-and-collectives.md)]`,
then that 8.2 µs is *not* wire time and the collective's cost is entirely
launch, Lamport-clear, publish and spin-exit — which redirects ranking slot B
away from the fabric and onto the kernel's own prologue.

**Duration.** 2 h. **GPU-h** 0.5 (needs 2 GPUs briefly).

### A3 — Is the 126.5 MiB L2 one pool or two per-die pools?

**Question.** The driver reports one aggregate figure. B200 is two dies. The
corpus's per-SM probe found only a ~6-cycle mean difference between SM 0–73 and
SM 74–147 on an 8 MiB buffer `[verified]` — but an 8 MiB buffer is interleaved
across every slice on both dies, so **every SM averages near and far accesses**
and the test cannot isolate the penalty. That is stated in the source document
as an open question.

**The decisive design the corpus has not used: probe a single line, not a
buffer.** A given cache line is homed to exactly one slice on exactly one die.
So:

1. Pick one 128 B line address.
2. Force 148 resident blocks, one per SM (the 100 KB shared-memory hog trick
   from `nf2.cu` `[verified — that technique is already written]`).
3. Have only the block whose `%smid` matches the target read that line, warm,
   repeatedly; time it.
4. Repeat for all 148 SMs, then repeat the whole thing for ~256 randomly chosen
   line addresses.

**Expected if theory holds (two pools).** For each line address, the 148 per-SM
latencies are **bimodal**, splitting ~74/74, and the *membership* of the fast
group flips with the line address. Group means should differ by the die-to-die
cost. `[inferred]`

**What a surprise means.** If every distribution is unimodal across all 256 line
addresses, L2 is functionally one pool for reads and the "~300 cycle die-to-die
penalty" `[reported — S5, restated third-hand]` does not apply to L2 hits at
all. That would close the question and make CTA placement across the die
boundary irrelevant for our collectives — a useful negative that stops a line of
work.

If bimodal, cross-check against the known capacity result: shared data gets
~63 MiB effective while private slices get ~126 MiB `[verified]`, which is
exactly what per-die pools with replication of shared lines would predict.

**Duration.** 3 h. **GPU-h** 1. **Value:** lower than its intellectual interest
— it changes no current candidate. Scheduled 16th for that reason.

### A4 — Small-message all-reduce: the size sweep, and NVLS vs what we run

**Question.** Three things at once. (a) Where is the one-shot/two-shot
crossover on *this* fabric? FlashInfer's MNNVL path and SGLang v2 disagree —
120 KiB vs 512 KiB for the same hardware and world size
`[verified — [`05-nvlink5…` §8.2](../00-hardware/05-nvlink5-nvswitch-and-collectives.md)]`.
(b) `MULTICAST_SUPPORTED = 1` on this fabric `[verified]` and
`enable_nccl_nvls=False` in every run we have ever done `[verified — Z1]`. What
does switch-side reduction buy? (c) How does either compare to the
`oneshotAllreduceFusionKernel` we actually execute?

**Tooling decision.** `nccl-tests` is **not installed** on this box (`find /`
for `all_reduce_perf` returns nothing `[verified]`), and neither is
`nvbandwidth`. But `libnccl.so.2`, `nccl.h` **and `nccl_device.h`** ship in the
venv at
`NotSglang/.venv/lib/python3.12/site-packages/nvidia/nccl/` `[verified]`, and
`flashinfer.comm` exposes `trtllm_ar.py`, `trtllm_mnnvl_ar.py`, `mnnvl.py`
`[verified]`. So: **do not build nccl-tests.** Drive all three from Python in
the existing venv, timing with CUDA events on a dedicated stream, which also
guarantees we are measuring the exact library build the server uses.

**Method.** 8 ranks via `torch.distributed` (NCCL backend), payload sweep
8 B → 256 MiB in powers of 2, 1000 iterations each after 100 warmup, report
median and p99. Three arms, one variable each:

```bash
# arm 1: NCCL ring/tree (what capacity mode runs)
NCCL_ALGO=Ring   torchrun --nproc_per_node=8 ar_sweep.py
NCCL_ALGO=Tree   torchrun --nproc_per_node=8 ar_sweep.py
# arm 2: NVLS / switch-side reduction
NCCL_ALGO=NVLS   NCCL_DEBUG=INFO torchrun --nproc_per_node=8 ar_sweep.py
# arm 3: the trtllm one-shot kernel we actually execute in latency mode
torchrun --nproc_per_node=8 ar_sweep.py --impl flashinfer-trtllm
```

`NCCL_DEBUG=INFO` is mandatory on the NVLS arm — it prints the algorithm
actually selected, and NCCL silently falls back. Confirm from the log, not from
the env var.

**Expected if theory holds.** NVLS wins at large sizes (switch-side reduction
halves wire traffic); one-shot wins below the crossover; the trtllm kernel is
competitive at our C1 payload of 384 KiB
`[verified — [`05-nvlink5…` §8.4](../00-hardware/05-nvlink5-nvswitch-and-collectives.md)]`.

**What a surprise means.** The corpus has already measured, twice, that NVLink
sits at **0.6–1.6%** utilisation while collectives hold 20–25% of GPU time
`[verified — ledger §2c]`. So the expected finding is that **all three arms
have essentially the same latency at our payload sizes**, because none of them
is wire-limited. If that holds, it *confirms* "tuning NCCL cannot help" with a
third independent measurement and closes the question. If instead NVLS is
materially faster at 384 KiB, then `--enable-nccl-nvls` is a free flag flip and
becomes a **B4** arm immediately.

**Confound.** Run on otherwise-idle GPUs; a collective benchmark sharing the box
with a server measures the server. Also pin the payload dtype to bf16 to match
the production kernel.

**Duration.** 4 h. **GPU-h** 2 (all 8).

### A5 — Does the persisting-L2 window hold a real weight working set?

**Question.** 79.06 MiB can be pinned `[verified]`. Against a *synthetic* 1 GiB
stream, `cudaAccessPolicyWindow` bought **+3% at best and lost 5%** when sized
to the full 79 MiB `[verified]`, and the conclusion recorded was that B200's
default replacement policy already keeps a ~48 MiB hot set resident. What has
never been tried is the case that matters: a **realistic decode working set**
held against **realistic KV streaming**.

**Why the synthetic result may not transfer.** The corpus's own L2 budget
analysis `[verified — [`02-memory…` §8](../00-hardware/02-memory-hierarchy-and-caches.md)]`
says the 8 hot experts per layer per rank are 20.25 MiB (fits in the ~63 MiB
shared-effective L2) but **64-way concurrent DSA KV reads are 72 MiB per layer**
(does not). So the pressure regime that would make pinning pay only exists at
concurrency, and the synthetic test did not reproduce it.

**Method.** A two-tenant kernel pair on one idle GPU:

- *Tenant W (weights)*: a resident buffer sized over a sweep
  {8, 16, 24, 32, 48, 63, 79} MiB, re-read every "step".
- *Tenant K (KV)*: a streaming read of a 2–8 GiB buffer, single-use, sized to
  emulate 64-way DSA reads.
- Measure tenant W's achieved read bandwidth and hit rate under three policies:
  (i) default; (ii) `cudaAccessPolicyWindow` pinning W; (iii) per-instruction
  `.L2::evict_first` on **K's** loads — the "protect by demoting the streamer"
  approach, which is the one Z3 says nobody has tried and which §2.6's
  256-bit-vector constraint governs.

**Expected if theory holds.** Policy (iii) beats (ii). Pinning W fights the
replacement policy from the wrong side; demoting K tells the cache the truth
about single-use data. Effect should grow with K's size and vanish when W alone
fits comfortably. `[inferred — from the verified −5% oversizing result plus the
verified ptxas constraint]`

**What a surprise means.** If (iii) also does nothing, then B200's default L2
replacement is already doing the right thing under pressure, and every
persisting-L2 proposal in this corpus should be closed with a recorded negative
— which is worth knowing, because that idea recurs in three documents.

**Follow-on if positive.** The engine-side version is a real kernel change
(FlashInfer's DSA/MLA KV loads would need the hint), so a positive result opens
a Loop-K item, not a flag flip.

**Duration.** 4 h. **GPU-h** 2. **Depends on Z3.**

### A6 — `tcgen05` MMA issue rate per block-scaled kind

**Question.** Build the table that
[arXiv:2512.02189](https://arxiv.org/abs/2512.02189) should have contained and
that this corpus explicitly distrusts `[verified — [`06-microbench…` §2](../00-hardware/06-microbenchmarks-and-reverse-engineering.md)]`:
issue-to-issue throughput and latency for `kind::f16`, `kind::f8f6f4` and
`kind::mxf4nvf4` (ours), across M ∈ {64, 128}, N ∈ {8 … 256},
`cta_group::{1,2}`.

**Why it matters.** Two ceilings in this corpus are currently guesses. The dense
GEMM's theoretical peak is *either* 2.25 or 4.5 PFLOP/s depending on whether
`nvjet_sm100_tst_*` dispatches `.kind::f16` or `.kind::f8f6f4`, and that is
recorded as **not established** `[verified — [`03-tensor-cores…` §14.2](../00-hardware/03-tensor-cores-tcgen05-and-mma.md)]`.
Any fraction-of-roofline from **B3** inherits that 2× ambiguity.

**Method.** Hand-written `.cu` with inline PTX, `-arch=sm_100a`. Two variants
per shape: (a) accumulator-carried dependency → latency; (b) 1/2/4/8/256
independent MMAs in flight → throughput. `tcgen05.alloc` **once** outside the
timing loop — it costs 442 cycles per round trip regardless of column count
`[verified]` and would swamp the measurement. Random inputs, never zeros
(§2.4). Instruction inventory is in the local CCCL headers at
`/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_mma.h`
`[verified — the corpus read them]`.

**Cheaper partial answer, do this first.** **B3** can settle the 2.25-vs-4.5
question for the kernel we care about in one ncu run: read the per-dtype tensor
pipe counters on `nvjet_sm100_tst_*` and see which pipe lights up. That is
hours, not days. Run **A6** only if **B3** shows a kernel with real headroom
whose ceiling is still ambiguous.

**Expected if theory holds.** `mxf4nvf4` at roughly 2× `f8f6f4` throughput per
issue slot; `cta_group::2` at ~2× `::1` for the same tile.

**What a surprise means.** If NVFP4 does **not** deliver its FLOP ratio, the
argument for keeping some layers in FP8 strengthens, and so does the
MXFP4-vs-NVFP4 scale-factor question (NVFP4's block-16 scales are reported to
take 2× the TMEM columns of MXFP4's block-32
`[reported — [`06-microbench…` §13.7](../00-hardware/06-microbenchmarks-and-reverse-engineering.md)]`).

**Duration.** 2–3 days. **GPU-h** 2. **Scheduled last.**

### A7 — Clock and power ceiling, on one idle GPU

**Question.** Three sub-questions, all cheap. Does `-lgc 1965,1965` take on B200,
or is there a clamp (H200 silently clamps ≥1830 to 1830
`[reported — [`07-power…` §12.1](../00-hardware/07-power-clocks-thermals-and-determinism.md)]`)?
What clock does the part *sustain* under a saturating NVFP4 load at the 1000 W
cap? What is the power delta?

**Method.** One GPU, explicit approval to mutate clocks (this is a shared box;
`gpu-bench` already locks and unlocks with `sudo -n` `[verified]`).

```bash
nvidia-smi -i 7 -q -d CLOCK,POWER            # before
sudo -n nvidia-smi -i 7 -lgc 1965,1965
# saturating load, random data, >=120 s, sampling at 10 Hz:
#   nvmlDeviceGetCurrentClockFreqs, nvmlDeviceGetCurrentClocksEventReasons,
#   nvmlDeviceGetTotalEnergyConsumption, power.draw, temperature.gpu
sudo -n nvidia-smi -i 7 -rgc                 # restore
```

Run the load in three flavours: BF16 dense (already characterised at other
clocks), FP8, and **NVFP4 at MoE-realistic shapes**. The NVFP4 flavour is the
one the corpus flags as unmeasured `[verified — [`07-power…` §12.3](../00-hardware/07-power-clocks-thermals-and-determinism.md)]`.

**Expected if theory holds.** The lock takes; the part boosts toward 1965 MHz;
under sustained NVFP4 load it settles somewhere between 1597 and 1965 with
`SW Power Cap` appearing in the clocks-event-reasons bitmask, since this box has
already accumulated 90.9 s of software power capping `[verified]`.

**What a surprise means.** If it holds ~1965 MHz at under 1000 W, the 19% is
free and **B1** should show close to 1.23× on the compute-bound slices. If it
collapses to ~1597 under load, then 1597 is the *emergent DVFS plateau* and the
harness's percentile heuristic accidentally picked the right number — which
closes an open question in two hardware documents at once.

**Duration.** 45 min. **GPU-h** 0.5. **Gates B1.**

---

## 6. Class B — engine experiments

Common protocol: `gpu-bench run --mode <m> --workload <w> --concurrency <c>`,
clocks locked at 1597 unless the experiment is about clocks, **real data only**,
warm cache, arms interleaved, n from §2.2, and the resolved
`server_args=ServerArgs(...)` line diffed between arms (§2.7). Every result gets
a ledger §4 row including negatives.

### B0 — The variance floor (run this first)

**Hypothesis.** Run-to-run σ on `latency-3-1-4 / aa-10k-real / c1` is ≈4% of
the mean, as inferred from a two-sample range `[verified — `SCORECARD.md` §1d]`.

**Method.** Six identical runs, same seed, same prompt set, alternating with a
30-minute idle gap between pairs so thermal state is not monotone. Then six more
of the C10 point, since concurrency changes queueing and probably changes σ.
Report σ for output tok/s, TPOT p50, TTFT p50, accept length, and prefix-cache
hit rate **separately** — they will not share a σ, and the ledger currently
assumes they do.

**Metric that decides it.** The sample standard deviation of each, plus a
paired-vs-unpaired comparison to quantify what §2.2 rule 3 actually buys here.

**Expected.** σ ≈ 4% on throughput; **much smaller on TPOT p50** (a median over
~1500 tokens is a far more stable statistic than an aggregate throughput over 64
requests). If so, **TPOT p50 becomes the primary decision metric** for every
subsequent A/B and the detection floor drops well below 8%.

**Surprise.** If σ > 6%, the entire ledger needs re-reading: several recorded
1.2× results would still stand, but nothing under 1.15× would.

**Confound.** Server restart between runs (include it — it is part of the
variance we actually face). Do **not** vary `num-prompts`; `RunSpec.identity()`
deliberately excludes sample count `[verified]`, so changing it silently
produces "comparable" runs that are not.

**Duration.** 2.5 h. **Unblocks: everything.**

### B1 — Unlocked-clock A/B on the engine

**Hypothesis.** At 1597/1965 = 0.8127 `[verified]`, compute-bound work runs at
81.3% of what this silicon can do. Dense GEMM is 37.1% of C1 GPU time and MoE
expert GEMMs 19.4% `[verified]`, so unlocking should be worth somewhere between
0 and ~1.23× on decode, depending on how much of the step is actually
compute-bound.

**Prior that predicts a small answer.** The C1 counters say **Tensor Active p95
= 2%** against **SMs Active p95 = 77%** `[verified — `SCORECARD.md` §4b]`, and
the corpus's own reading is that decode at C1 is latency-bound, not
compute-bound `[verified — [`02-memory…`](../00-hardware/02-memory-hierarchy-and-caches.md)]`.
Latency in *nanoseconds* through the memory hierarchy is partly clock-invariant
(HBM clock is independent and already at its 3996 MHz maximum `[verified]`), so
the honest prediction is well under 1.23×.

**Method.** Three arms × n from B0, interleaved: locked 1597 (control), locked
1965, unlocked (`-rgc`). Both C1 and C10 on `aa-10k-real`. Continuous clock and
throttle-reason sampling on all 8 GPUs throughout (§2.1).

**Metric that decides it.** TPOT p50 delta, with the *measured mean clock during
the run* reported alongside. A result of "+11% at a measured mean clock of
1780 MHz" is a real datum; "+11% unlocked" is not.

**Expected.** C1: under +10%. C10: larger, because C64 counters show DRAM read
at 56% and SMs Active at 92% p95 `[verified — ledger §2d]`, i.e. a more
compute-engaged regime.

**Surprise.** If C1 gains near the full 23%, then the "latency-bound at batch 1"
reading in two documents is wrong and the whole optimization thesis shifts
toward compute. If it gains ~0% *and* the clock genuinely rose, that is a strong
confirmation of the latency-bound reading and closes the question permanently.

**Confound.** Power cap interaction — with all 8 GPUs unlocked the node may hit
1000 W/GPU and clock back non-uniformly, which would *create* rank skew. Sample
per-GPU clocks and check whether skew rises; that interaction is itself a
finding.

**Duration.** 3 h. **GPU-h** 3. **Depends on A7 and B0.**

### B2 — Profile the competitive configuration under load

**Hypothesis.** The ledger's C1-vs-C64 comparison changes four variables at once
(§0.1). The configuration we are actually trying to win with — `latency-3-1-4`,
TP8, allreduce fusion on — has **never been profiled above C1**.

**Method.** One 8-rank nsys capture, 20 s bounded window, GPU metrics at 10 kHz
(`gb10x` set), `--cuda-graph-trace=node`, `--nccl-trace=coll,gpu,kernel-launch`,
on `latency-3-1-4 / aa-10k-real / c10` — AA's own 10-parallel scenario.

```bash
/home/aman/code/NotSglang/.venv/bin/gpu-bench run \
  --mode latency-3-1-4 --workload aa-10k-real --concurrency 10 \
  --sm-clock 1597
/home/aman/code/NotSglang/.venv/bin/gpu-bench deepdive runs/<dir>/trace.sqlite
```

**Metric that decides it.** The family share table (gemm / moe / collective /
attention / dsa_indexer), the waiting-vs-transfer split, and NVLink utilisation
inside collectives — the same three tables the corpus already has at C1 and at
capacity/C64, now at the point that matters.

**Expected.** Between the two existing captures. Attention and the indexer
should fall from their C1 shares (10.9%, 5.8%) as they amortise. Collectives
should look *different* from the capacity capture because this config runs the
fused one-shot kernel, not NCCL ring.

**Surprise.** If the fused path's waiting fraction is much lower than capacity
mode's 44%, then a large part of the "47% of collective time is skew" headline
is an artefact of the DP-attention configuration and does not describe the
competitive one. That would substantially demote ranking slot A.

**Confound.** ~10% profiler overhead (§2.5) — do not compare this run's tok/s to
any unprofiled number. Bound the capture; an unbounded one produced a 15 GB
SQLite last time (§2.5).

**Duration.** 1.5 h. **GPU-h** 1.5.

### B3 — ncu the top three kernels: fraction-of-roofline

**Hypothesis.** The hotspot table ranks by **cost**, not **opportunity**. A
kernel at 31.7% already at 85% of its ceiling has less to give than one at 10%
sitting at 30%. No headroom claim in this corpus is currently defensible
`[verified — ledger §2e, §7 Phase 0a]`.

**Targets, in order of share** `[verified — ledger §2b]`:
`bmm_E2m1_E2m1E2m1_Fp32_*` (12.4%), `bmm_Bfloat16_E2m1E2m1_Fp32_*tokFp32`
(11.1%), `nvjet_sm100_tst_128x24_64x11_2x1_v_bz_splitK` (7.3%).

**Method.** A dedicated server instance — ncu serialises kernels, so this run's
timings are worthless and must not be reported (§2.5).

```bash
/home/aman/code/NotSglang/.venv/bin/gpu-bench ncu \
  --kernel-regex 'bmm_E2m1_E2m1E2m1' --launch-count 8 --launch-skip 64 \
  -- <server-invocation>
```

Metrics already wired into the harness `[verified — `analysis/hotspots.py`]`:
`dram__throughput.avg.pct_of_peak_sustained_elapsed`,
`sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed`,
`sm__warps_active.avg.pct_of_peak_sustained_active`, and three stall ratios
(long-scoreboard, barrier, mio-throttle). **Add** `dram__bytes_read.sum` — the
MoE document names it as the metric that settles its §5.3 outright
`[verified — [`03-moe…` §9.2](../01-kernel-optimization/03-moe-kernels-and-expert-parallelism.md)]`
— and the per-dtype tensor-pipe counters, which resolve the 2.25-vs-4.5 PFLOP/s
ambiguity for `nvjet` (§A6).

**Metric that decides it.** Fraction of a *defensible* ceiling, plus the
dominant stall reason. **The gate: anything above ~75% of a defensible ceiling
is closed, not optimized**, and gets a ledger row saying so
`[verified — ledger §7 Phase 0a]`.

**Expected.** These are vendor kernels — TRT-LLM-gen and cuBLAS — so the prior
is that they are good. The Kimi K3 precedent in this program: cuBLAS turned out
to be at **82% of bf16 peak** for the attention projections and the correct
decision was to stop `[verified — ledger §7]`.

**Surprise.** If the `bmm_E2m1_*` pair are at 15–20% of a memory roofline with
`long_scoreboard` dominant, then the MoE gather (**A1**) is the mechanism and
there is a real Loop-K target worth 23.5% of GPU time. If they are at >75%,
ranking slots D and E close and the whole programme redirects to Loop S.

**Confound.** ncu's replay changes cache state between passes; use
`--launch-skip 64` to clear JIT/autotune and read several launches. Also:
`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` validity for `tcgen05` NVFP4 MMA on SM100 is
an open question `[verified — [`07-power…` §12.8](../00-hardware/07-power-clocks-thermals-and-determinism.md)]`
— if the tensor counter under-counts block-scaled MMA, every "tensor pipe is
idle" claim in this corpus is wrong. **Cross-check the ncu tensor metric against
achieved FLOP/s computed from the kernel's own shapes and duration.** That check
costs nothing and validates a number the whole corpus leans on.

**Duration.** 4 h. **GPU-h** 3. **This is the gate on all Loop-K work.**

### B4 — The flag flips that have never been measured

Seven arms. **One flag per arm**, each verified by diffing the resolved
`ServerArgs` (§2.7), each n=3 minimum, interleaved with a control, on
`latency-3-1-4 / aa-10k-real` at both C1 and C10. Deciding metric: TPOT p50
(per B0, likely the tightest statistic), with output tok/s reported alongside.

| arm | change | resolved field | hypothesis | expected | confound |
|---|---|---|---|---|---|
| **B4a** | `--enforce-disable-flashinfer-allreduce-fusion` | `flashinfer_allreduce_fusion_backend: 'auto' → None` | prices what fusion is *already* buying us (§0.1) | a loss — fusion should be worth something; magnitude unknown | this is the only clean isolation of fusion; the ledger's #9 is permanently confounded |
| **B4b** | `--mem-fraction-static 0.92` | `0.85 → 0.92` | ~13 GiB/GPU more KV cache; ledger slot H estimates ~1.14× at concurrency | no effect at C1, real effect at C10+ | OOM risk — 166,357 of 183,359 MiB were already in use at probe time `[verified]`; watch for retraction/preemption in the logs, which would show as a TTFT tail rather than a throughput loss |
| **B4c** | `--chunked-prefill-size 16384` (and 4096) | `8192 → …` | AA's 10k input is prefill-dominated for TTFT; chunk size trades TTFT against decode interference | TTFT moves, TPOT roughly flat | prefill CUDA graphs are **disabled** (`cuda_graph_config.prefill.backend='disabled'` `[verified — Z1]`), so this arm is measuring an un-graphed path |
| **B4d** | `--enable-nccl-nvls` | `enable_nccl_nvls: False → True` | `MULTICAST_SUPPORTED=1` and we have never used it | **nothing**, if A4 confirms the link is idle | only meaningful if A4 shows an NVLS advantage at 384 KiB; otherwise skip |
| **B4e** | `--num-continuous-decode-steps 2` | `1 → 2` | amortises the host scheduler across steps | small TPOT gain, TTFT cost | the docstring warns about TTFT; measure both, and AA counts TTFT |
| **B4f** | `TRTLLM_ENABLE_PDL=0` | env | control only — prices the PDL that is already on (§0.3) | a loss | not an optimization; run once |
| **B4g** | `--enable-fused-qk-norm-rope` | `False → True` | removes narrow norm/RoPE kernels; the corpus counts **grid==1 kernels at 10.7% of decode kernel time** `[verified]` | small | **changes numerics order** — this arm requires a GSM8K re-run within noise of 96.00% before its perf number is admissible |

**Deliberately excluded.** `--enable-two-batch-overlap` (refuses to start:
`index_topk_freq=4` `[verified — ledger §4 #10]`) and
`--enable-single-batch-overlap` (measured, −12.5%, and the mechanism is
understood: SBO's overlap is gated on `is_flashinfer_cutedsl()` or non-Blackwell
deep_gemm, and we run `flashinfer_trtllm` `[verified — ledger §4 #9]`). Neither
may be retried without a code change that invalidates the previous result.

**Duration.** ~2 days for all seven at n=3 × 2 concurrencies. **GPU-h** ~12.
Run **B4a** and **B4b** first; they are the two with a real prior.

### B5 — Rank-skew attribution

**Hypothesis.** 44–54% of collective time is ranks arriving apart, mean skew
14.6 µs at C1 and 39.4 µs at C64, max 18.1 ms, rank 0 last in 24% of 114,171
instances `[verified — ledger §2c]`. §0.2 rules out expert-routing imbalance as
the mechanism at `ep_size=1`. So what is it?

**Three sub-experiments, cheapest first.**

**B5a — Is NUMA binding already applied, and correctly?** `SGLANG_NUMA_BIND_V2`
defaults to **True** `[verified — `environ.py:1025`]`, and
`utils/numa_utils.py:configure_subprocess` wraps each scheduler subprocess in
`numactl` with a CPU+memory binding derived from the GPU's node, with a probe
and a warn-or-raise failure path `[verified — source read]`. `numa_node=None`
in our runs means auto-detect, not disabled `[verified — resolved args]`.
GPUs 0–3 are on node 0 and 4–7 on node 1 `[verified]`.

*Method:* read the per-rank startup logs for the numactl binding string and any
`_handle_numa_bind_failure` warning. If binding succeeded on all 8 ranks, the
host-placement hypothesis is **already controlled** and this line of inquiry
closes without a run. If it failed on some ranks (seccomp/cpuset can block
`set_mempolicy`, and the code explicitly handles that case), that asymmetry is a
prime suspect and the fix is explicit `--numa-node 0 0 0 0 1 1 1 1`.

*Cost:* 30 min, zero GPU. **Do this before anything else in B5.**

**B5b — Per-rank clock and power during the capture.** Sample all 8 GPUs at
10 Hz through a C10 and a C64 run. The power document reports GPU 5 measured
−4.7% and an even/odd 3.5 °C delta `[verified — [`07-power…`](../00-hardware/07-power-clocks-thermals-and-determinism.md)]`,
and predicts the spread *widens* with batch size.

*Deciding metric:* correlation between per-rank mean clock and per-rank
last-to-arrive frequency. If rank 0 is systematically the slowest-clocked, the
skew is thermal/power placement and no scheduling change will fix it.

**B5c — Where in the step do ranks diverge?** Under `ep_size=1` and `dp_size=1`
(latency mode) every rank runs an identical program, so divergence must
accumulate somewhere specific. Plot **per-instance skew against collective
index within the forward pass** — there are ~156 chained collectives per forward
`[verified — [`05-nvlink5…` §8.5](../00-hardware/05-nvlink5-nvswitch-and-collectives.md)]`.

*Deciding metric:* the shape of skew-vs-index. **Monotonically growing** ⇒ skew
accumulates and the fix is a resync barrier at a chosen depth. **Flat** ⇒ each
collective independently re-randomises and the cause is per-instance jitter
(host wakeup, interrupt, clock). **Spiky at particular indices** ⇒ a specific
layer or op is the divergence point.

These three have *different* fixes, and the corpus currently cannot tell them
apart. This plot is the single most informative artefact in **B5**.

**B5d — Expert-distribution recorder, as a falsifier.** Per §0.2 the prediction
is *no* per-rank imbalance in expert work under `ep_size=1`.

```
--expert-distribution-recorder-mode per_pass --enable-expert-distribution-metrics
POST /start_expert_distribution_record … /dump_expert_distribution_record
```
`[verified — flags exist at `server_args.py:2352`, `enable_eplb` at 2337]`

*Expected:* token counts per expert are uneven (that is what routing does) but
per-**rank** work is identical, because every rank holds a slice of every expert.
*Surprise:* if per-rank work does differ at `ep_size=1`, something in the MoE
runner is not sharding the way we think, which is a far more interesting bug
than the skew itself.

**Duration.** 6 h total (B5a is free). **GPU-h** 4. **Unblocks ranking slots
A and C.**

### B6 — The AA repeat protocol

**Hypothesis.** Our headline is one sample. AA's is a **P50 over the trailing
72 hours, sampled 8×/day** `[verified — ledger §1]`. A configuration that is
fast warm and slow cold scores its average, and we have never measured our own
spread across a day.

**Method.** Run the AA shape (`aa-10k-real`, C1 and C10, temperature 0.6) on a
3-hourly cadence for 24 h — 8 samples, matching AA's frequency — and report the
median with its interquartile range. Interleave one **cold** arm (flushed) per
day to bound the pessimistic end.

**Second arm: `--apply-chat-template`.** `SCORECARD.md` §2b item 3 flags that
`bench_serving` sends raw completions, so GLM-5.2's chat template and its
default `Reasoning Effort: Max` are **not applied** — and states plainly that
"tokens/sec is tokens/sec" is an assumption, not a verified fact `[verified]`.
AA measures a reasoning model at temperature 0.6 with TTFT counted to the first
token *including* reasoning tokens. Run one arm with the template applied and
compare output tok/s, TTFT and TTFAT.

**Deciding metric.** P50 output tok/s over 8 samples, with IQR; plus the
TTFT/TTFAT split on the template arm.

**Expected.** P50 close to the single measured 349.5, with an IQR of a few
percent. Template-on should not move output tok/s much but *will* change the
TTFT/TTFAT relationship — Baseten publishes 800 ms TTFT against 7.9 s TTFAT for
this model, a 10× difference depending on which is quoted
`[reported — ledger §1]`.

**Surprise.** If the IQR is large, "consistency beats peak" becomes an
optimization target in its own right, and the ranking gains a slot that does not
currently exist. If the template arm moves output tok/s materially, our headline
is not comparable to the board and must be re-stated.

**Confound.** The board is volatile — Databricks' published output speed moved
86 → 336 t/s between two fetches hours apart `[verified — `SCORECARD.md` §2a]`.
**Always snapshot the board with a date when quoting a comparison.** And we
cannot reproduce the public-internet path; our TTFT stays a lower bound.

**Duration.** 2 days elapsed, ~8 GPU-h.

### B7 — (contingent) Whatever B3 promotes

Held open deliberately. If **B3** finds a kernel below ~75% of a defensible
ceiling, Loop K opens against it with the ledger's seven-step gate — and step 6
(re-profile, the targeted counter must have moved) is the one that is always
skipped and must not be `[verified — ledger §7 Loop K]`. If **B3** finds
everything above the bar, Loop K closes with three recorded negatives and the
entire remaining budget goes to Loop S: **Z5**, **B5**, and unblocking TBO.

---

## 7. What unblocks the ranking

`01-ranked-opportunities.md` is being written alongside this file; the shared
vocabulary is the ledger §5 candidate letters (A–J), so the mapping is stated in
those terms.

| ranking slot (ledger §5) | current status | what this plan does to it |
|---|---|---|
| **A** — unblock TBO for index-topk sharing | ranked #1 on "44% of collective time is skew" | **Z2**, **B2**, **B5c** decide whether that 44% describes the *competitive* config at all (§0.1). **B5c**'s skew-vs-index plot decides whether TBO is even the right mechanism. Do not start the engine change until B5c has a shape. |
| **B** — why AllReduce is not CUDA-graphed | ranked #2, "cheap to investigate" | **Z5** settles it for zero GPU-hours. Note §0.1: the un-graphed kernel is the DP-attention path's NCCL call, not the fused kernel latency mode runs. |
| **C** — balance expert routing across ranks | ranked #3 on an EP8 premise | **§0.2 invalidates the premise** (`ep_size=1`). **B5d** falsifies it formally. Re-rank as "balance *request* assignment across DP ranks", which is a different change. |
| **D** — ncu the two `bmm_E2m1_*` expert GEMMs | ranked #4 | **B3** *is* this slot. **A1** supplies the correct roofline denominator. Between them the slot resolves to a number or closes. |
| **E** — explain 9% tensor-pipe p95 vs 92% SM occupancy | Tier 2 | **B3**'s confound check (is `PIPE_TENSOR_ACTIVE` valid for block-scaled MMA on SM100?) may dissolve this slot entirely. Run that cross-check before treating 9% as real. |
| **F** — reduce launch count | Tier 2 | **Z2** already has the answer in hand: true launch gap is **1.9%** of wall `[verified]`. This slot is largely closed; what remains is *grid width*, not launch count. |
| **G** — unblock the draft-depth IMA | Tier 3, ~1.21× | untouched by this plan — it is a debugging task, not a measurement. Keep its rank. |
| **H** — match the published serving config | Tier 3, ~1.14× | **B4b** measures the `mem-fraction` half directly. |
| **I** — TileRT on GLM-5.1 | Tier 3 | untouched; external ceiling check. Note `weights/GLM-5.2-FP8-TileRT/` is **empty** — there is no TileRT build on this box `[verified]`. |
| **J** — set a node price | Tier 3 | untouched; a business input, not a measurement. |
| **new** — unlock the clock | **not currently ranked** | **A7** + **B1**. Potentially the largest single lever nobody has priced, and §2.1 shows the 19% is a harness heuristic, not silicon. |
| **new** — reclaim indexer KV (78 → 22 layers) | **not currently ranked** | **Z4**. ~7.39 GB per 1M tokens per GPU `[verified arithmetic]`, possibly a predicate change. |
| **new** — enable prefill CUDA graphs | **not currently ranked** | surfaced by **Z1**: `cuda_graph_config.prefill.backend='disabled'` `[verified]`. TTFT is an AA-scored metric. |
| **new** — protect weights in L2 by demoting KV | **not currently ranked** | **Z3** + **A5**. |

**The three that gate the most downstream work:** **B0** (without σ nothing else
is a measurement), **B3** (without fraction-of-roofline no headroom claim is
defensible), and **B2** (without a profile of the competitive config the ranking
is aimed by a four-variable confound).

---

## 8. Deliberately not on the list

- **Re-measuring anything in §1's left column.** Cite it instead.
- **Attention and the DSA indexer.** 10.9% → 4.6% and 5.8% → 2.4% from C1 to
  C64 `[verified — ledger §2a]`. They amortise across the batch; optimizing them
  chases a shrinking target. (**B2** will re-check this at C10 — if the fall
  does not reproduce in the competitive config, they come back.)
- **NCCL parameter tuning.** Measured twice as not the bottleneck: NVLink at
  0.6–1.6% while collectives hold 20–25% of GPU time `[verified]`. **A4** is a
  third check, not a tuning exercise.
- **Retrying SBO or TBO** without a code change that invalidates the recorded
  result (ledger §4 #9, #10).
- **A megakernel for GLM-5.2.** The launch overhead a megakernel removes is
  already gone — true launch gap is 1.9% of wall `[verified]` — and the packing
  win is available in cheaper increments `[verified — [`05-megakernels…` §10.3](../01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md)]`.
- **Building `nccl-tests` or `nvbandwidth`.** Neither is installed; both are
  replaceable by Python against the venv's own NCCL and FlashInfer builds
  (**A4**), which has the additional merit of measuring the exact binaries the
  server uses.
- **Layer-level decode attribution.** Structurally unavailable under graph
  replay (§2.5).
- **`--flush-cache` on task workloads.** Recorded as a mistake (ledger §4 #7).

---

## 9. Recording protocol

1. **Every experiment gets a ledger §4 row, including the negatives.** A
   measured negative costs the same to obtain and is worth as much. Three of the
   ledger's ten existing rows are negatives, which is roughly the right ratio.
2. **Every number carries units and conditions**: clock, mode, workload, data
   (real/synthetic), concurrency, warm/cold, n, and mean ± sd.
3. **Every A/B records the resolved-`ServerArgs` diff**, not the command-line
   diff (§2.7, §0.1).
4. **Loop S results are judged on `waiting_ms`, not tok/s** `[verified — ledger
   §7]`. Throughput moves for many reasons; waiting time moving is what proves
   the mechanism. The SBO episode is the cautionary case.
5. **Loop K results are judged on the targeted counter moving**, re-measured
   after the change. A kernel that got faster while the counter did not move
   means the mechanism is not understood.
6. **Correctness gate on anything that touches numerics**: GSM8K within noise of
   the measured 96.00% (200 examples, temp 0.6, and raise the 4096-token cap —
   `truncated_rate` was 3.5% and a truncated answer scores zero
   `[verified — `SCORECARD.md` §3]`).
7. **Microbenchmark sources are committed**, not left in the scratchpad. The
   existing `mem*.cu` / `nf2.cu` / `p2p.cu` / `tmem_bench.cu` set lives only in
   a session-scoped `/tmp` directory `[verified]`; every number in
   `02-memory-hierarchy-and-caches.md` and
   `06-microbenchmarks-and-reverse-engineering.md` depends on files that will
   not survive. **Move them into the repo as the first action of Class A.**

---

## Sources

**Measured or read on this box for this document (2026-08-17):**

- `nvidia-smi --query-supported-clocks=graphics -i 0` → 247 bins, 1965…217 MHz;
  cross-computed against `gpubench/env.py:pick_sustainable_clock` → 1597 MHz
- `nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used`
  → all 8 idle, 0 MiB, 120 MHz, 187–202 W, 29–33 °C
- `/home/aman/code/benchmark/runs/sweep-latency-3-1-4/server.log` — the
  `Auto-enabling FlashInfer AllReduce Fusion` line and the full resolved
  `ServerArgs`; startup 03:04:28 → ready 03:06:50
- `/home/aman/code/benchmark/runs/sweep-capacity/server.log`,
  `/home/aman/code/benchmark/runs/sweep-capacity-overlap/server.log` — resolved
  `flashinfer_allreduce_fusion_backend`, `enable_dp_attention`, `dp_size`,
  `ep_size`, `max_running_requests`
- `NotSglang/python/sglang/srt/arg_groups/overrides.py:1763-1808` —
  `_FLASHINFER_ALLREDUCE_FUSION_ARCHS`, `_flashinfer_allreduce_fusion_auto_enable`,
  `_enforce_disable_allreduce_fusion`
- `NotSglang/python/sglang/srt/server_args.py` — `numa_node` (1213),
  `enable_nccl_nvls` (1885), `enable_dsa_cache_layer_split` (1116),
  `enable_flashinfer_allreduce_fusion` / `enforce_disable_…` /
  `flashinfer_allreduce_fusion_backend` (1970-1976), `enable_eplb` (2337),
  `expert_distribution_recorder_mode` (2352), deprecation handler (3925-3935)
- `NotSglang/python/sglang/srt/entrypoints/engine.py:1526` — `TRTLLM_ENABLE_PDL`
  default
- `NotSglang/python/sglang/srt/managers/scheduler.py:4697-4700` and
  `utils/numa_utils.py`, `environ.py:1025` (`SGLANG_NUMA_BIND_V2 = EnvBool(True)`)
- `benchmark/gpubench/{cli,config,env,run}.py`,
  `gpubench/profilers/{nsys,ncu}.py`, `gpubench/analysis/hotspots.py`,
  `gpubench/deepdive.py` — commands, metrics, capture flags, clock control
- Tool inventory: `nsys` 2025.6.3 at `/usr/local/bin/nsys`; `ncu` 2026.1.1 at
  `/opt/nvidia/nsight-compute/2026.1.1/ncu`; `nvcc` 13.3.73 at
  `/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc`; `ninja` absent;
  `nccl-tests` and `nvbandwidth` absent; `libnccl.so.2` + `nccl.h` +
  `nccl_device.h` present in the venv; `flashinfer.comm` exposes `trtllm_ar`,
  `trtllm_mnnvl_ar`, `mnnvl`, `nvshmem`
- `sudo -n true` succeeds — clock locking is available without a password

**Corpus documents this plan is derived from:**

- [`00-hardware/00-this-machine-ground-truth.md`](../00-hardware/00-this-machine-ground-truth.md) §3, §6, §8
- [`00-hardware/02-memory-hierarchy-and-caches.md`](../00-hardware/02-memory-hierarchy-and-caches.md) §1, §2, §4.1, §4.4, §7, §8, §10
- [`00-hardware/03-tensor-cores-tcgen05-and-mma.md`](../00-hardware/03-tensor-cores-tcgen05-and-mma.md) §14
- [`00-hardware/05-nvlink5-nvswitch-and-collectives.md`](../00-hardware/05-nvlink5-nvswitch-and-collectives.md) §8
- [`00-hardware/06-microbenchmarks-and-reverse-engineering.md`](../00-hardware/06-microbenchmarks-and-reverse-engineering.md) §4, §5, §13
- [`00-hardware/07-power-clocks-thermals-and-determinism.md`](../00-hardware/07-power-clocks-thermals-and-determinism.md) §12
- [`01-kernel-optimization/01-gemm-on-sm100.md`](../01-kernel-optimization/01-gemm-on-sm100.md) §11
- [`01-kernel-optimization/03-moe-kernels-and-expert-parallelism.md`](../01-kernel-optimization/03-moe-kernels-and-expert-parallelism.md) §8.3, §9
- [`01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md`](../01-kernel-optimization/05-megakernels-persistent-kernels-and-launch-overhead.md) §2.4, §2.5, §10.3
- [`05-models/00-local-weights-ground-truth.md`](../05-models/00-local-weights-ground-truth.md) §9.4
- `/home/aman/code/benchmark/SCORECARD.md` §1a–§1d, §2a–§2b, §3, §4
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` §1–§7
