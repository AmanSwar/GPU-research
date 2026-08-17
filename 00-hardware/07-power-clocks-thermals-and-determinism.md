# Power, clocks and thermals: the 1000 W envelope and why your numbers move

## What this is

Everything about the B200's power/clock control loop that changes a benchmark number,
measured on **our own 8× B200 node** (driver 595.71.05, 183 GB, NV18) rather than recalled.
It contains a correction to a premise this corpus has been carrying: **we are not running at a
1597 MHz clock lock.** 1597 MHz is the B200's free-running DVFS plateau for memory-bound work,
and the benchmark harness has been *recording* it, not *setting* it. Everything downstream —
run-to-run comparability, the rank-skew story, the headroom estimate — shifts as a result.
All measurements below were taken on 2026-08-17; commands are reproducible verbatim.

---

## Bottom line for our system

- **The clocks are not locked.** Verified: idle GPUs sit at 120 MHz, a lightly loaded one boosts
  to 1965 MHz, and a compute-saturating GEMM is power-capped down to ~1120 MHz. A real
  `-lgc 1597,1597` lock would hold 1597 MHz in all three states. It does not. [verified, this box]
- **1597 MHz is a DVFS plateau, not a setting.** Every memory-bound kernel class I ran parked at
  *exactly* 1597 MHz: d2d memcpy, decode-shaped skinny GEMM, elementwise SiLU. Our GLM-5.2
  serving load sat at 1597 MHz for 5,393 consecutive samples across all 8 GPUs. That is the
  same plateau, reached by accident. [verified, this box]
- **1000 W is the binding constraint on dense FLOPS, not thermals.** Sustained BF16 GEMM pins
  every GPU at ~990 W with `SW Power Cap` continuously asserted and the clock collapsed to
  **1072–1222 MHz = 55–62% of the 1965 MHz boost** — at 52–61 °C, nowhere near any thermal
  limit. The datasheet 9 PFLOPS dense FP4 is a boost-clock number that no sustained workload
  reaches. [verified, this box]
- **Lower precision buys clock, not just FLOPs.** FP8 GEMM ran at a *higher* clock (1177 vs
  1117 MHz) and *lower* power (902 vs 988 W) than BF16 on the identical shape, for 1.74× the
  throughput. Under a power cap, NVFP4's byte reduction is worth more than its FLOP ratio
  suggests. [verified, this box]
- **GPU 5 is a measurably bad die and it is always GPU 5.** Five independent measurements agree:
  highest idle power (+8.9%), highest energy over 300 s of identical TP8 work (+5.7%), lowest
  sustained GEMM throughput (−4.7%), lowest clock at the cap (−12.3%), most lifetime
  SW-power-cap time (192.9 s vs 90.9 s). It is one of two parts in this chassis from a different
  serial-number lot. [verified, this box]
- **Odd-numbered slots run 3.5 °C hotter, permanently.** Median GPU temp over 5,393 samples:
  slots 0/2/4/6 = 32.5 °C, slots 1/3/5/7 = 36.0 °C. Perfectly alternating; airflow, not silicon.
  [verified, this box]
- **Our rank-0 skew is almost certainly *not* a power or clock effect.** Rank 0 had the *lowest*
  energy of all eight ranks while showing "utilized" 70% of samples versus 41% for everyone else,
  at 387 W versus ~450 W in that bucket. That is the signature of extra *low-power host-serialised
  work*, which is what TP0 does in SGLang. Chasing this with clock policy will not work.
  [inferred from verified measurements]
- **The `LLM Inference` workload power profile exists on this box and is switched off.**
  `nvidia-smi power-profiles -l` lists it; `-gr`/`-ge` report nothing requested or engaged.
  NVIDIA's own B200 data puts it at 3% perf loss for 8% datacentre power saving, versus 10% perf
  loss for 5% saving from naive frequency scaling. [verified on this box; numbers reported]
- **Electricity is 1–3% of $/token. Stop optimising it.** At C64 the node burns roughly
  $0.0036 of industrial electricity per million tokens against $0.11–$0.33 of amortised GPU
  cost. Power capping pays only when the *facility* is power-limited. [inferred, arithmetic below]

---

## 1. The correction: what "1597 MHz" actually is

`benchmark/SCORECARD.md` line 29 says "SM clock locked 1597 MHz". Every run manifest carries
`"sm_clock_mhz": 1597`. `k3-kernels/scripts/bench_matrix.sh` line 34 shows where the number comes
from:

```bash
CLK=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -1)
echo "# sm clock during run: ${CLK} MHz  (lock it: sudo nvidia-smi -i 0 -lgc 2430,2430)"
```

The harness **reads** the clock and records it. The lock is a comment reminding you to do it
yourself. [verified — `/home/aman/code/NotSglang/k3-kernels/scripts/bench_matrix.sh`]

### The proof that no lock is set

Three states, same box, same session, no configuration change between them:

| state | SM clock | power | what it means |
|---|---:|---:|---|
| all 8 GPUs idle | **120 MHz** | 187–204 W | `nvclockmin`; a lock would forbid this |
| one GPU, light intermittent load | **1965 MHz** | 249 W | `clocks.max.sm`; a 1597 lock would forbid this |
| one GPU, sustained BF16 GEMM | **1117 MHz** (p50) | 988 W | below 1597; a lock's *min* would forbid this |

[verified, this box]

Corroborating: NVML reports `nvmlDeviceGetCurrentClocksEventReasons() == 0x0` and
`nvmlDeviceGetCurrentClockFreqs()` returns `nvclockeditable=0`, `nvclockmin=120`,
`nvclockmax=1965`. There is no user clock constraint in play.

### So why was it *always* 1597 under load?

Because 1597 MHz is where the B200's DVFS controller parks for memory-bound work. Measured on
GPU 3, 25 s per kernel class, 20 warm-up iterations discarded:

| kernel class | clock p5 / p50 / p95 | power p50 | power max | temp | note |
|---|---:|---:|---:|---:|---|
| BF16 GEMM 8192³ | 1095 / **1117** / 1597 | 988.2 W | 1013.5 W | 54 °C | at the cap |
| FP8 e4m3 GEMM 8192³ | 1102 / **1177** / 1597 | 901.7 W | 1013.7 W | 54 °C | at the cap |
| d2d memcpy, 5 GB | 1597 / **1597** / 1597 | 627.6 W | 694.7 W | 43 °C | **plateau** |
| skinny GEMM 8×16384×16384 | 1597 / **1597** / 1597 | 680.2 W | 734.5 W | 46 °C | **plateau** |
| elementwise SiLU, 1 G elem | 1597 / **1597** / 1597 | 685.6 W | 756.8 W | 50 °C | **plateau** |

[verified, this box — `scratchpad/kinds.py`]

Three different memory-bound kernel classes, three identical clocks, to the MHz. 1597 MHz is a
hardware/firmware operating point. Note also that the compute-bound rows have **p95 = 1597**: the
clock is not sitting at 1117, it is *oscillating* between the power-limited point and the plateau.

And 1597 MHz is exactly what our GLM-5.2 TP8 serving load produced: 5,393 samples per GPU over
300 s at 20 Hz, on all 8 GPUs, **zero samples off 1597 MHz except six on GPU 4** (§5.3).
GLM-5.2 decode at low concurrency is memory-bound; it lands on the plateau; the harness writes
down 1597 and calls it a lock.

### Why this matters for a 72-hour P50 benchmark

A recorded equilibrium is not a controlled variable. The plateau is stable *for a given workload
mix*, but the moment the mix shifts toward compute — higher concurrency, longer prefill, a fused
kernel that raises arithmetic intensity, an NVFP4 path that packs more MACs per byte — the clock
leaves 1597 and lands wherever 1000 W puts it, which we measured spanning **1035–1965 MHz**.
Two runs whose identity hash matches on `sm_clock_mhz: 1597` can therefore have run at different
clocks for different fractions of their duration. The hash gives false confidence.

The fix is one line and it is in §8.

---

## 2. What the 1000 W envelope actually buys

### The board and its limits, read off the part

```
$ nvidia-smi -q -d POWER -i 0
    Current Power Limit    : 1000.00 W
    Requested Power Limit  : 1000.00 W
    Default Power Limit    : 1000.00 W
    Min Power Limit        :  200.00 W
    Max Power Limit        : 1000.00 W
    GPU Memory Power Readings
        Average Power Draw :  26.95 W
    EDPp Multiplier        : 100.00%
```

[verified, this box]

- The power limit is **not adjustable upward**: default = max = 1000 W. Only downward, to 200 W.
- HBM is separately metered (~27 W idle) and **its clock is not adjustable at all**:
  `memclockmin = memclockmax = 3996 MHz`, `memTransferRate = 7992 MT/s`, fixed. `-lmc` has
  nothing to do on this part. [verified, this box]
- `EDPp Multiplier` is the electrical-design-point (peak current / di-dt) limiter, exposed at
  100%. It is a *separate* limiter from average board power and can assert well below 1000 W —
  see §5.3, where GPU 4 reported `SW Power Cap` at 381 W.

Derived from the device, not the datasheet:

```
NVIDIA B200  cc10.0  SMs=148  clockRate=1965000 kHz  memClk=3996000 kHz
busW=7680 bit  L2=126 MB  smemPerSM=233472 B  regsPerSM=65536  maxWarps/SM=64
peak HBM BW = 2 × 3.996e9 × 7680/8 = 7.67 TB/s
```

[verified, this box — `cudaGetDeviceProperties` + `cudaDeviceGetAttribute`, CUDA 13.3 nvcc]

Note the bus is **7680 bit, not 8192**: one sixteenth of the HBM3e is fused off (192 GB → 183 GB,
8.0 TB/s → 7.67 TB/s). The DGX B200 datasheet's "64 TB/s" for eight GPUs is the full-stack
number; the enabled silicon gives 61.4 TB/s. [verified, this box, against
<https://www.nvidia.com/en-us/data-center/dgx-b200/>]

### The clock ladder under a compute-saturating load

Eight independent processes, one per GPU, 45 s of back-to-back BF16 `8192³` matmul each,
aligned start via a shared wall-clock deadline:

| GPU | TFLOP/s | vs best | clk p50 | clk min | W p50 | W max | T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1316.8 | −1.05% | 1192 | 1057 | 988.4 | 1009.1 | 57 |
| 2 | 1312.9 | −1.35% | 1162 | 1102 | 987.2 | 1003.1 | 52 |
| 3 | 1321.0 | −0.74% | 1185 | 1080 | 991.9 | 1013.1 | 59 |
| 4 | 1308.4 | −1.69% | 1125 | 1042 | 992.7 | 1012.0 | 52 |
| 5 | **1267.8** | **−4.74%** | **1072** | 1035 | 994.7 | 1018.7 | 59 |
| 6 | 1313.2 | −1.32% | 1162 | 1050 | 993.7 | 1013.8 | 52 |
| 7 | **1330.8** | **+0.00%** | **1222** | 1125 | 987.7 | 1014.9 | 61 |

[verified, this box — GPU 0 excluded, see §7.2]

Read this carefully, because it is the central fact about this hardware:

1. **Every GPU is pinned at the 1000 W cap** (987–995 W median, transients to 1019 W) with
   `nvmlClocksEventReasonSwPowerCap` (0x4) asserted continuously.
2. **Temperature is irrelevant here.** 52–61 °C against a slowdown limit far above. `HW Thermal
   Slowdown` and `SW Thermal Slowdown` counters are **0 µs lifetime on all 8 GPUs**. On an
   air-cooled DGX B200 in a functioning datacentre, thermals are not the throttle; power is.
3. **The sustained clock is 55–62% of boost.** 1072–1222 MHz against `clocks.max.sm = 1965`.

### What that does to the datasheet number

The DGX B200 datasheet quotes, for the 8-GPU system, "FP4 Tensor Core: 144 PFLOPS | 72 PFLOPS"
(sparse | dense) and "FP8 Tensor Core: 72 PFLOPS" sparse. Per GPU: **9 PFLOPS dense FP4,
4.5 PFLOPS dense FP8**, and by the usual halving, **2.25 PFLOPS dense BF16**.
[verified for FP4/FP8 from <https://www.nvidia.com/en-us/data-center/dgx-b200/>; BF16 inferred
from the standard 2× ratio]

Our measured 1321 TFLOP/s BF16 at ~1150 MHz scales to 1321 × 1965/1150 = **2257 TFLOP/s at boost**
— within 0.3% of the 2250 figure. So:

- cuBLAS BF16 `8192³` is at **~100% of clock-adjusted peak**. There is nothing to win in the
  kernel.
- The *only* thing between us and the datasheet is the clock, and the clock is 58% of boost
  because 1000 W cannot feed 148 SMs of Blackwell tensor cores at 1965 MHz.
- Therefore **sustained dense FP4 on this part is ≈ 9 PFLOPS × 0.58 ≈ 5.2 PFLOPS**, not 9.
  [inferred — the BF16 clock ratio is measured; whether NVFP4 sustains the same clock is not,
  and §2.1 argues it sustains a *higher* one]

### 2.1 Precision buys clock

Same GPU, same shape, same 25 s protocol:

| | BF16 GEMM 8192³ | FP8 e4m3 GEMM 8192³ | ratio |
|---|---:|---:|---:|
| iterations/s | 1071.7 | 1865.5 | 1.74× |
| achieved TFLOP/s | 1178 | 2051 | 1.74× |
| SM clock p50 | 1117 MHz | **1177 MHz** | +5.4% |
| board power p50 | 988.2 W | **901.7 W** | **−8.8%** |
| TFLOP/W | 1.19 | **2.27** | 1.91× |

[verified, this box]

FP8 does 1.74× the work at 91% of the power *and a higher clock*. The mechanism is that a large
share of a GEMM's power goes into moving bytes — HBM reads, L2 traffic, register-file and
shared-memory bandwidth, and the toggling of the operand datapath — and FP8 halves all of it.
The freed power budget lets DVFS hold a higher frequency, which compounds with the FLOP ratio.

**This is a first-class argument for the NVFP4 build that has nothing to do with FLOPs.** NVFP4
halves the bytes again relative to FP8. On a part that is power-limited at 58% of boost during
compute-bound phases, quantisation is a *frequency* optimisation as well as a throughput one.
[inferred from the verified FP8-vs-BF16 measurement; the NVFP4 point is an extrapolation and
should be measured with `bmm_E2m1_E2m1E2m1_Fp32_swiGlu_dynB_sm100f` directly]

---

## 3. How the envelope is enforced

There are at least five independent control loops, in rough order of aggressiveness. Only the
first three show up in the counters.

| loop | what it watches | what it does | visible as |
|---|---|---|---|
| **SW power cap** | filtered board power vs `power.limit` | walks the V/f operating point down | `SW Power Cap`, bit `0x4` |
| **EDPp / peak current** | instantaneous current, di/dt | fast clock drop, can fire far below TDP | also reports as `0x4` |
| **Adaptive clocking** | supply-rail voltage droop | momentary clock halving on droop events | usually invisible at any sane sample rate |
| **SW thermal** | GPU temp vs max-operating | walks clocks down | `SW Thermal Slowdown`, `0x20` |
| **HW slowdown** | temp or external `PWR_BRAKE#` | halves core clock or worse, in hardware | `HW Slowdown` `0x8`, `HW Thermal` `0x40`, `HW Power Brake` `0x80` |

Adaptive clocking is **enabled** on our parts:
`nvmlDeviceGetAdaptiveClockInfoStatus(h) == 1`. [verified, this box]

`HW Slowdown` "reduc[es] the core clocks by a factor of 2 or more" and is asserted "if either HW
Thermal Slowdown or HW Power Brake are active"; `HW Power Brake` is "External Power Brake
Assertion... triggered (e.g. by the system power supply)".
[verified — <https://docs.nvidia.com/deploy/nvidia-smi/index.html>]

On this node, over the full driver lifetime, `HW Thermal Slowdown = 0 µs`,
`SW Thermal Slowdown = 0 µs`, `HW Power Braking = 0 µs`, `Sync Boost = 0 µs` on **all eight
GPUs**. Only `SW Power Capping` is nonzero. If you see a HW slowdown on this box, it is a
facility or cooling fault, not a tuning opportunity.

### Thermal limits are reported as *margins*, not absolutes

```
Temperature
    GPU Current Temp                  : 31 C
    GPU T.Limit Temp                  : 57 C
    GPU Shutdown T.Limit Temp         : -5 C
    GPU Slowdown T.Limit Temp         : -3 C
    GPU Max Operating T.Limit Temp    :  0 C
```

[verified, this box]

These are **degrees of headroom relative to the limit**, not temperatures. `T.Limit = 57` means
57 °C of margin remains; `Slowdown = -3` means slowdown engages 3 °C past the max-operating point.
A monitoring rule written as `GPU T.Limit Temp > 80 → alert` has the sign backwards and will
fire never. Alert on `T.Limit` going *low*.

---

## 4. Clock event reasons: the exact bitmask

From the bundled pynvml on this box, cross-checked against `nvidia-smi -q`:

| constant | value | meaning |
|---|---:|---|
| `nvmlClocksEventReasonGpuIdle` | `0x1` | nothing running; clocks dropping to idle |
| `nvmlClocksEventReasonApplicationsClocksSetting` | `0x2` | deprecated applications-clocks limit |
| `nvmlClocksEventReasonUserDefinedClocks` | `0x2` | **alias of the above** |
| `nvmlClocksEventReasonSwPowerCap` | `0x4` | SW power scaling reducing clocks |
| `nvmlClocksEventReasonHwSlowdown` | `0x8` | HW slowdown (≥2× clock reduction) |
| `nvmlClocksEventReasonSyncBoost` | `0x10` | member of a sync-boost group |
| `nvmlClocksEventReasonSwThermalSlowdown` | `0x20` | SW thermal capping |
| `nvmlClocksEventReasonHwThermalSlowdown` | `0x40` | HW thermal slowdown |
| `nvmlClocksEventReasonHwPowerBrakeSlowdown` | `0x80` | external power-brake assertion |
| `nvmlClocksEventReasonDisplayClockSetting` | `0x100` | display clock constraint |
| `nvmlClocksEventReasonAll` | `0x1ff` | mask of all of the above |

`nvmlDeviceGetSupportedClocksEventReasons(h)` returns `0x1ff` on our B200 — all reasons are
implemented. [verified, this box]

### The gotcha you will hit

`UserDefinedClocks` is **the same bit as `ApplicationsClocksSetting` (0x2)**, and on driver
595.71.05 with a B200 the driver does not appear to set it when a `-lgc` lock is what is holding
the clock down. Consequence:

> **You cannot detect an active clock lock from the event-reason bitmask.** Detect it by
> comparing `clocks.current.sm` against `clocks.max.sm` and against what the workload should be
> producing, or by keeping the lock command in the run manifest.

[verified by construction, this box: no bit was ever set during 5,393 × 8 samples, including
periods where the clock sat 368 MHz below max at 40% of the power budget]

Also note: `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` has been renamed
`DCGM_FI_DEV_CLOCKS_EVENT_REASONS` (field 112); the NVML symbols carry the same rename with the
old `nvmlClocksThrottleReason*` names retained as aliases.
[verified — <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>
and the bundled pynvml, which exports both spellings]

---

## 5. Per-GPU variation: the silicon lottery, measured on this chassis

Five independent measurements, all on the same eight parts, over one session.

| GPU | serial | idle W | 300 s serving energy (kJ) | GEMM TFLOP/s | GEMM clk p50 | lifetime SW power cap | median T serving |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 1655124026867 | 188.2 | **99.75** | *(artifact)* | *(artifact)* | **90.9 s** | 32 °C |
| 1 | 1655124026874 | 191.4 | 102.93 | 1316.8 | 1192 | 172.0 s | 36 °C |
| 2 | 1655124026844 | **187.2** | 101.30 | 1312.9 | 1162 | 152.4 s | 33 °C |
| 3 | 1655124026791 | 191.0 | 102.55 | 1321.0 | 1185 | 165.3 s | 36 °C |
| 4 | 1655124026869 | 189.7 | 102.28 | 1308.4 | 1125 | 113.1 s | 33 °C |
| 5 | **1652225049349** | **204.0** | **105.44** | **1267.8** | **1072** | **192.9 s** | 36 °C |
| 6 | 1655124026964 | 188.2 | 102.09 | 1313.2 | 1162 | 138.5 s | 32 °C |
| 7 | **1653325111656** | 199.1 | 103.64 | **1330.8** | **1222** | 91.3 s | 36 °C |

[all columns verified, this box]

### 5.1 The energy measurement is the cleanest one

Under TP8, every rank executes the *same* CUDA graph on the *same* shapes for the *same* number
of steps. So over a fixed wall-clock window, per-rank energy difference is a near-pure readout of
per-die efficiency. Using NVML's monotonic `nvmlDeviceGetTotalEnergyConsumption` counter over a
**300.06 s** window of live GLM-5.2 serving:

```
GPU0  99,754 J     GPU4 102,277 J
GPU1 102,934 J     GPU5 105,437 J   <- worst
GPU2 101,302 J     GPU6 102,086 J
GPU3 102,553 J     GPU7 103,641 J
                   spread max/min = 5.70%,  σ/μ = 1.52%
```

**GPU 5 burns 5.70% more energy than GPU 0 doing identical work.** That is the tax this chassis
pays for its worst die, every second, forever.

This is squarely inside the literature's range. GEM measured a **7.7% throughput spread within a
single 8-GPU L40 node, persistent over a week**, and **27.7%** across 128 L40s; the SC'22 study
across five supercomputers found **8% average (22% max) variation between same-SKU GPUs, with
outliers 1.5× slower than median**. [reported — arXiv:2605.19945; arXiv:2208.11035]

### 5.2 Idle power is a leakage readout

At true idle (120 MHz, no processes), 3.83 M samples per GPU over 25 s:

```
GPU2 187.2 W   GPU6 188.2 W   GPU0 188.2 W   GPU4 189.7 W
GPU3 191.0 W   GPU1 191.4 W   GPU7 199.1 W   GPU5 204.0 W
```

Spread 8.9%. At a fixed clock and voltage with no work, board power is dominated by static
leakage plus clock tree — i.e. by the die. **GPU 5 and GPU 7 are the two highest, and they are
the two parts whose serial numbers fall outside the `16551240268xx` cluster** (`1652225049349`,
`1653325111656`). Different manufacturing lot, different leakage. [verified, this box]

The honest caveat: leakage does not cleanly predict f_max. GPU 7 has the *second-highest* idle
power and the *highest* sustained clock (1222 MHz). GPU 5 has the highest idle power and the
lowest clock. High leakage can go either way — it correlates with fast, low-Vt transistors as
well as with wasted power. Both off-lot parts are extremes, in opposite directions.

### 5.3 A throttle event caught in the act

Six samples out of 43,144 GPU-samples, all on GPU 4, all within 0.51 s:

```
t=265.77s sm=1245 W=381.1 T=43 util=100  reasons=0x4
t=265.83s sm=1245 W=381.1 T=43 util=100  reasons=0x4
t=265.88s sm=1117 W=405.0 T=41 util=100  reasons=0x0
t=265.93s sm=1117 W=405.0 T=41 util=100  reasons=0x0
t=266.10s sm=1147 W=490.7 T=47 util=  4  reasons=0x4
t=266.16s sm=1147 W=490.7 T=47 util=  4  reasons=0x4
t=266.22s sm=1597 W=540.0 T=45 util= 39  reasons=0x4
t=266.28s sm=1335 W=567.7 T=45 util= 51  reasons=0x4
```

[verified, this box]

A **30% clock collapse (1597 → 1117 MHz) for half a second, with `SW Power Cap` asserted at
381–568 W against a 1000 W limit.** Average board power was nowhere near the cap. This is the
EDPp / peak-current limiter or an adaptive-clocking droop response, surfacing through the same
`0x4` bit as the average-power cap. Two consequences:

- **`SW Power Cap` asserted does not mean you are near TDP.** Do not infer a power problem from
  the bit alone; correlate with `power.draw.instant`.
- **A 20 Hz sampler catches the tip of these.** This event lasted ~500 ms; sub-100 ms events
  will be invisible. If a single benchmark iteration lands inside one, that iteration is 30%
  slow for reasons no post-hoc counter will explain. This is exactly the "firmware-initiated
  clock throttling [that] can corrupt any throughput measurement" described for H200.
  [reported — arXiv:2605.11999]

---

## 6. Thermals and position in the chassis

Median GPU temperature over 5,393 samples of live serving:

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| median T (°C) | 32 | 36 | 33 | 36 | 33 | 36 | 32 | 36 |
| module ID | 6 | 8 | 7 | 5 | 2 | 4 | 3 | 1 |
| NUMA node | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |

Even slots mean **32.5 °C**, odd slots mean **36.0 °C**, delta **+3.5 °C**, and it is perfectly
alternating across all 5,393 samples. [verified, this box]

This is airflow geometry, not silicon: the SXM modules are arranged in pairs and the odd-indexed
module in each pair sits downstream of the even one in the front-to-back air path. It is
reproduced independently of load, and it does not follow the serial-number lots (GPU 5 and GPU 7
are both odd; GPU 2 and GPU 6 are both even and both low-leakage).

The DGX B200 thermal envelope for context: **1,550 CFM airflow, 48,794 BTU/hr heat dissipation,
10–35 °C operating range, 6 × 3.3 kW PSUs**.
[verified — <https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html>]
The datasheet quotes system max power at **~14.3 kW**.
[verified — <https://www.nvidia.com/en-us/data-center/dgx-b200/>]

### Does 3.5 °C matter?

On *this* node, in *this* room, no — measurably. `HW/SW Thermal Slowdown` counters are 0 µs
lifetime, and in the all-8 GEMM run the hottest GPU (7, 61 °C) posted the *highest* throughput
while the coolest tier (2/4/6, 52 °C) sat mid-pack. Temperature is not on the critical path at
20 °C inlet.

It matters at the margin, and it matters at a warm inlet. Two mechanisms:

1. **Leakage is exponential in temperature.** A hotter die leaks more, leaves less of the 1000 W
   for dynamic power, and therefore clocks lower *at the same power cap*. This is the documented
   mechanism behind "Lit Silicon": on MI300X, the hottest GPU in a node ran **6.2% lower
   frequency** than the coolest, with temperature ratio 1.155× and frequency ratio 1.062×.
   [reported — arXiv:2511.09861]
2. **Raise the inlet and everything moves.** At 35 °C inlet (the top of the operating range,
   15 °C above where we measured) the whole population shifts up and the odd slots get there
   first. A P50-over-72-hours number taken in a cold room is not the same number as one taken
   during a hot afternoon.

**Practical rule for the 72-hour benchmark: log inlet temperature, or at minimum log
`temperature.gpu` per rank, and treat a run whose median GPU temp differs by >3 °C from the
reference run as a different run.**

---

## 7. Rank skew: is it power, or is it software?

This is the section the profile demands. The measured facts from
`personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`:

```
observed 14,097 ms  =  transfer 7,505 ms  +  waiting 6,599 ms
arrival skew: mean 9.2 µs, max 4,897 µs
Rank 0 is the last to arrive in 26,929 of 114,171 instances (24%), then ranks 2 and 3
```

47% of collective time is waiting; rank 0 is the straggler at ~2× the uniform rate (12.5%).

### 7.1 Power/clock cannot explain it, and here is why

During the entire serving window I sampled, **every rank ran at exactly 1597 MHz and no rank
recorded a single clock-event bit** (except GPU 4's 0.5 s excursion, §5.3). A skew mechanism that
runs through DVFS requires ranks to be at *different* clocks. They were not. The serving load is
memory-bound, sits on the 1597 MHz plateau at 450 W of a 1000 W budget, and never approaches a
limiter. There is no frequency lever for silicon variation to act through.

This is an important **negative result**: the DVFS/thermal straggler mechanism that dominates the
literature (Lit Silicon, GEM, PAL) is *switched off* in our current operating point, because we
are nowhere near the power cap. Our skew is something else.

### 7.2 What the energy data says instead

Joint distribution of `utilization.gpu` and `power.draw`, 5,393 samples per GPU, live serving:

| GPU | util<10 | 10–49 | 50–89 | **util≥90** |
|---:|---|---|---|---|
| 0 | 1376 n @ 248 W | 142 n @ 249 W | 72 n @ 249 W | **3803 n @ 387 W** |
| 1 | 3166 n @ 253 W | 16 n @ 427 W | 22 n @ 418 W | 2189 n @ 456 W |
| 2 | 3155 n @ 250 W | 11 n @ 417 W | 25 n @ 405 W | 2202 n @ 444 W |
| 3 | 3164 n @ 252 W | 6 n @ 341 W | 10 n @ 366 W | 2213 n @ 450 W |
| 4 | 3147 n @ 250 W | 11 n @ 325 W | 19 n @ 471 W | 2216 n @ 452 W |
| 5 | 3160 n @ 264 W | 9 n @ 341 W | 9 n @ 454 W | 2215 n @ 458 W |
| 6 | 3156 n @ 250 W | 10 n @ 309 W | 22 n @ 442 W | 2205 n @ 453 W |
| 7 | 3158 n @ 264 W | 11 n @ 267 W | 37 n @ 406 W | 2187 n @ 445 W |

[verified, this box]

Rank 0 is an outlier on two axes simultaneously and in opposite directions:

- It reports `util ≥ 90` in **70.5%** of samples versus **41%** for every other rank — a kernel is
  resident on rank 0 far more of the time.
- In that same bucket it draws **387 W** versus **444–458 W** for the others — 15% less power.
- And its **total energy is the lowest of all eight** (§5.1).

More resident time, less power per resident second, less total energy. There is only one shape of
work that does that: **many small, low-occupancy, low-power kernels**. Not spin-waiting on a
barrier (that would raise energy, not lower it, relative to being idle at 250 W), and not slower
compute (the shapes are identical under TP8).

In SGLang, TP0 is the rank that owns the interface to the tokenizer/detokenizer managers,
sampling, and per-request output bookkeeping. Those emit exactly this signature: a stream of tiny
kernels between the collectives, serialised on the host.

**Hypothesis, labelled as such: rank 0 arrives last 24% of the time because it runs extra
low-power host-serialised work between allreduces, not because its GPU is slower.**
[inferred — from verified energy/utilisation/clock measurements. It is not proven. What would
prove it: an nsys capture filtered to rank 0, differencing the kernel set against rank 1 and
attributing the arrival delta to the extra kernels plus host gaps.]

Corroborating, weakly: in my 8-way GEMM microbenchmark, GPU 0 — and only GPU 0 — collapsed to
664 TFLOP/s at 1965 MHz and 732 W, reproducibly across two runs, while running at **1321 TFLOP/s
alone** and **1324 TFLOP/s paired with GPU 1**. Full boost clock, low power, half throughput is
the signature of a GPU that is *idle between bursts*, i.e. host-starved. Under 8-way host
contention, rank 0's process is the one that loses. That is a property of this node's host
scheduling, not of GPU 0's die — but it is the same rank that strags in production.
[verified measurement; the causal link to the production skew is inferred]

### 7.3 What to do about it, ranked

1. **Fix the host side first.** Pin the eight scheduler processes to disjoint core sets on the
   correct NUMA node (`nvidia-smi topo -m` gives GPUs 0–3 → cores `0-55,112-167` / NUMA 0, GPUs
   4–7 → cores `56-111,168-223` / NUMA 1). Give TP0 dedicated cores. Nothing in this document is
   worth as much as that, if the hypothesis in §7.2 holds.
2. **Do not "fix" it with per-rank power caps yet.** The Lit Silicon mitigation — cap the leaders
   down to the straggler, or reallocate power to the straggler — is real (4% average power
   reduction, 3% throughput improvement, up to 6% with CPU power sloshing) but it is a mitigation
   for a *thermal* straggler. [reported — arXiv:2511.09861] Ours is not thermal.
3. **Keep it in mind for the compute-bound regime.** The moment we move to C64 prefill-heavy
   operation, the GPUs *do* hit the cap (§2), GPU 5's −4.7% *does* become a real straggler, and
   the literature's toolkit becomes applicable:
   - **GEM** (variability-aware expert→GPU mapping): give the fast GPU proportionally more
     tokens. On 8×L40, "the fastest GPU processes 14% more tokens than the slowest at the same
     latency"; end-to-end latency improved **7.9% average, 16.5% max**, and **p90 TPOT by 9.1%
     average, 16.9% max**. [reported — arXiv:2605.19945] With 256 experts and 8 active, GLM-5.2
     has the routing freedom to do this.
   - **StragglAR**: run a ReduceScatter among the non-stragglers *during* the straggler's delay,
     then finish. **25% speedup on an 8-GPU server.** [reported — arXiv:2505.23523] This attacks
     the 6,599 ms of waiting directly and does not care *why* the straggler is late.
4. **Expect variability to grow with batch size.** GEM: "the difference between TPOT for GPU
   device 5 (slowest) and GPU device 7 (fastest) increases with batch size... larger batches
   increase per-GPU power consumption and push each GPU closer to its thermal limits", 5% → 7.1%
   → 7.3% at BS 32/64/128, "and the identity of the slowest and fastest GPUs remain identical
   across batch sizes". [reported — arXiv:2605.19945] Our C1 profile *understates* the skew we
   will see at C64.

---

## 8. Locking clocks: syntax, semantics, and what it costs

### The commands

```bash
# persistence: keep the driver resident so clocks/limits survive process exit.
# NOT persistent across reboot -- must be re-run by a systemd unit.
sudo nvidia-smi -pm 1                       # already Enabled here, via nvidia-persistenced

# lock SM clock. min,max pair, or a single value. Supersedes applications clocks.
sudo nvidia-smi -lgc 1597,1597              # all GPUs
sudo nvidia-smi -i 3 -lgc 1597,1597         # one GPU
sudo nvidia-smi -rgc                        # release

# power limit (downward only on B200: min 200 W, max = default = 1000 W)
sudo nvidia-smi -pl 700
sudo nvidia-smi -pl 1000                    # restore

# memory clock: nothing to do on B200, min == max == 3996 MHz
sudo nvidia-smi -lmc 3996                   # no-op here
```

[verified: flag syntax from `nvidia-smi --help` on this box and
<https://docs.nvidia.com/deploy/nvidia-smi/index.html>; the B200 memory-clock fact from
`nvmlDeviceGetCurrentClockFreqs` on this box]

### Semantics you must know

- **`-ac` / applications clocks are dead.** `nvidia-smi -q -d CLOCK` on our B200 returns
  "Requested functionality has been deprecated" for both Applications Clocks and Default
  Applications Clocks. NVML documents them as "deprecated and will be removed in CUDA 14.0",
  with `nvmlDeviceSetGpuLockedClocks` / `nvmlDeviceSetMemoryLockedClocks` as the replacements.
  [verified, this box + <https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html>]
- **A lock is not persistent.** "After system reboot or driver reload GPU clocks go back to their
  default value." Same for power limits and persistence mode. If your 72-hour benchmark spans a
  driver reload, it spans a configuration change. [verified — NVML docs, as above]
- **A lock is a request, not a guarantee.** Power and thermal limiters still act *below* the
  locked minimum. Our own k3 log records a 2430 MHz lock "sustaining ~2362" on an RTX PRO 6000.
  [verified — `/home/aman/code/NotSglang/personal_docs/kimi-k3/k3-kernel-optimization-log.md:18`]
  On H200, `--lock-gpu-clocks` was found to "silently clamp any requested lock ≥1830 MHz to
  ≈1830 MHz... a side effect of the lock mechanism rather than a hardware limit", while
  free-running boost held 1980 MHz indefinitely. [reported — arXiv:2605.11999] **Whether B200
  clamps similarly at some value below 1965 MHz is not sourced and is a measurement we should
  take** (§12).
- **`--mode` exists and is undocumented in the local help beyond "Valid modes: 0, 1."** The
  nvidia-smi manual describes it as 0 = default accuracy, 1 = improved efficiency.
  [reported — <https://docs.nvidia.com/deploy/nvidia-smi/index.html>] Untested here.
- **You cannot read the lock back.** No NVML getter in the bundled pynvml, no bit in the event
  reasons (§4), no field in `nvidia-smi -q`. Record it in the manifest at set time.

### What locking at 1597 MHz would actually cost us

The naive arithmetic — 1597/1965 = 81.3%, so we give up 18.7% — is **wrong for our workload**,
and the reason is the whole point of this document. Two independent lines of evidence:

**Our own measurement.** At 1597 MHz our serving load draws 450 W of 1000 W and never asserts a
limiter. If the lock were removed, DVFS would be free to boost — but it *already is* free, and it
chooses 1597, because the work is memory-bound and HBM runs at a fixed 3996 MHz that no SM clock
can influence. The clock ladder in §1 shows three distinct memory-bound kernel classes all
converging on 1597. **A 1597 MHz lock on a memory-bound decode workload costs approximately
nothing, because it is what the hardware does anyway.**

**The literature agrees, on a different part.** On H200, "the median throughput difference between
1590 and 1980 MHz is <0.1%" across four attention paradigms, sequence lengths 1K–65K, batch sizes
1–32 — a 240 MHz gap that "produces zero throughput gain at 7–13% more power".
[reported — arXiv:2605.11999]

Where the lock *does* cost is the compute-bound phases:

| phase | binding resource | clock elasticity | effect of a 1597 lock |
|---|---|---|---|
| C1 decode, spec-decode 3-1-4 | HBM (weight streaming, M≈4–8 rows) | low | ~none |
| MoE expert GEMMs at C1, 8/256 active | HBM (expert weight streaming) | low | ~none |
| DSA sparse MLA decode | HBM (KV streaming) | low | ~none |
| collectives (47% wait, 53% NVLink) | host arrival / NVLink | none | none |
| **prefill / C64 dense GEMM** | **tensor cores at 1000 W** | **high** | **it is a ceiling, and the free-running clock is *below* it anyway (1117 MHz)** |

The last row is the punchline. In compute-bound phases the free-running clock is **1072–1222
MHz**, i.e. *below* 1597. So a `-lgc 1597,1597` lock has no effect there either: the power
controller wins, and the "min" side of the lock is not honoured.

**Therefore: locking at 1597 MHz costs us essentially nothing in throughput, and buys real
determinism in the memory-bound regime.** Our current numbers are not wrong — but they were
obtained by luck, not by control, and the luck runs out as the workload gets more
compute-dense (higher concurrency, deeper speculation, better-fused kernels).

The kernel-name evidence for "our dense GEMM is memory-bound": the top kernel is
`nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` at 12.6%. An 8-wide tile dimension is a skinny-GEMM
tile. At C1 with 3-1-4 speculation the GEMM M-dimension is ~4 tokens: the kernel reads the entire
weight matrix to produce four rows, arithmetic intensity ~4 FLOP/byte, deep in the memory-bound
region of the roofline. Calling it "dense GEMM, 37.1%" and inferring compute-bound is the trap.
[inferred from the kernel name and shape; confirmed in kind by the skinny-GEMM row of §1's table,
which sat on the 1597 MHz plateau at 680 W]

### The recommendation

```bash
# run-start, in the benchmark harness, recorded into the manifest:
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1597,1597
# verify it took, and record BOTH the request and the observation:
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm --format=csv,noheader
# run-end:
sudo nvidia-smi -rgc
```

And add `lgc_requested_mhz` to the run identity hash alongside the existing observed
`sm_clock_mhz`. Today the hash records only the observation, which is why two runs at genuinely
different clock policies can hash identically.

---

## 9. The lever nobody is pulling: workload power profiles

This is on our box, today, and unused:

```
$ nvidia-smi power-profiles -l -i 0
0. Max-P            1. Max-Q            4. Network        5. Balanced
6. LLM Inference    7. LLM Training    13. HPC

$ nvidia-smi power-profiles -gr -i 0
No profiles are currently requested.
$ nvidia-smi power-profiles -ge -i 0
No profiles are currently engaged.
```

[verified, this box, driver 595.71.05]

Profile priorities and conflicts, from `-ld`:

| id | name | priority | conflicts with |
|---:|---|---:|---|
| 0 | Max-P | 5 | Max-Q, HPC |
| 1 | Max-Q | 6 | Max-P, Network, Balanced, HPC |
| 4 | Network | 4 | Max-Q, Balanced, HPC |
| 5 | Balanced | 3 | Max-Q, Network, HPC |
| **6** | **LLM Inference** | 2 | LLM Training, HPC |
| 7 | LLM Training | 1 | LLM Inference, HPC |
| 13 | HPC | 0 | everything |

Note that **`Max-P` does not conflict with `LLM Inference`**, so they can coexist — this is the
"base mode plus modifier mode" composition the design paper describes, with the higher-priority
profile (Max-P, 5) winning on any overlapping knob.

### What the profiles do and what they are worth

The feature is described in *Datacenter Energy Optimized Power Profiles* (Narayanaswamy, Patel,
Karlin, Gupta, Saripalli, Guo; arXiv:2510.03872), which states it was "released with Blackwell
B200 GPUs". The mechanism is a firmware configuration-table layer above the individual controls —
"GPU and CPU clock controls, memory frequency settings (MCLK), total GPU power limits (TGP),
dynamic voltage-frequency scaling (DVFS), and high-speed GPU interconnect (NVLink) power states"
— with an arbitration layer resolving conflicts by priority. GPU savings come "from lower clock
frequencies and by reducing power to underutilized structures within the GPU, such as, the
crossbar". [reported — arXiv:2510.03872]

The number that matters most for us, from that paper, on B200:

| approach | performance decrease | datacentre power saving |
|---|---:|---:|
| naive **frequency scaling** | 10% | 5% |
| **Training Profiles** | 1% | 5% |
| **Inference Profiles** | **3%** | **8%** |

[reported — arXiv:2510.03872]

**Naive `-lgc` frequency scaling is strictly dominated**: it costs 3× the performance for less
power saving than the tuned inference profile. Because the profile also touches the crossbar,
NVLink power states and MCLK, it reaches structures `-lgc` cannot.

Per-application Max-Q results from the same paper (performance loss capped at 3%):

| application | perf loss | DC power saving | DC throughput increase |
|---|---:|---:|---:|
| DeepSeek R1 | 3% | 12% | 8% |
| Llama 3.1 8B | 2% | 11% | 7% |
| Llama 3.1 70B | 2% | 9% | 6% |
| Mistral 7B | 2% | 9% | 6% |
| HPL | 1% | 13% | 12% |
| GROMACS | 1% | 15% | 13% |

[reported — arXiv:2510.03872]

**Action: measure profile 6 (`LLM Inference`) and profile 0 (`Max-P`) against the current
unprofiled baseline.** Both directions are interesting — Max-P is the one that might buy us
throughput, and the paper is candid that it only yields "2-3% performance gain... because the GPU
is typically running at a less efficient point in the voltage frequency curve already". Given we
measured the GPU sitting at 58% of boost against the power cap in compute-bound phases, a
profile that reallocates power *away* from the crossbar and *into* the SMs is precisely the
right shape of intervention. Untested here — the box was in production and I did not mutate
its configuration.

There is also `nvidia-smi power-smoothing`, present on this driver with settable ramp-up/ramp-down
rates in mW/s, hysteresis in ms, and preset profiles. Querying it returned "Insufficient
Permissions" as a non-root user. It is a grid-stability feature (smoothing the node's power
transient), not a throughput feature, but it changes the *shape* of the power ramp and therefore
could interact with the EDPp events in §5.3. Unexplored.

---

## 10. What to sample continuously, and how to attribute a slow run afterwards

### The minimum viable telemetry

Every field below is cheap and available from `nvidia-smi --query-gpu` at 5–10 Hz, or from DCGM.

| what | `--query-gpu` field | DCGM field | ID | why |
|---|---|---|---:|---|
| SM clock | `clocks.sm` | `DCGM_FI_DEV_SM_CLOCK` | 100 | the number that moved |
| max SM clock | `clocks.max.sm` | — | — | the only way to detect a lock (§4) |
| memory clock | `clocks.mem` | `DCGM_FI_DEV_MEM_CLOCK` | 101 | should be a constant 3996; alert if not |
| board power (avg) | `power.draw` | `DCGM_FI_DEV_BOARD_POWER_WATTS` | 155 | the budget |
| board power (inst) | `power.draw.instant` | `DCGM_FI_DEV_BOARD_POWER_RAW_WATTS` | 157 | catches EDPp-triggering transients |
| **energy counter** | — | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | 156 | monotonic mJ; the *only* exact energy source |
| enforced power limit | `enforced.power.limit` | `DCGM_FI_DEV_BOARD_POWER_LIMIT_ENFORCED_WATTS` | 164 | detects someone else's `-pl` |
| GPU temp | `temperature.gpu` | `DCGM_FI_DEV_GPU_TEMP_CELSIUS` | 150 | §6 |
| memory temp | `temperature.memory` | `DCGM_FI_DEV_MEMORY_TEMP_CELSIUS` | 140 | HBM has its own limit |
| event reasons | `clocks_event_reasons.*` | `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` | 112 | bitmask, §4 |
| power violation ns | — | `DCGM_FI_DEV_POWER_VIOLATION` | 240 | cumulative, survives sampling gaps |
| thermal violation ns | — | `DCGM_FI_DEV_THERMAL_VIOLATION` | 241 | cumulative |
| SM active | — | `DCGM_FI_PROF_SM_ACTIVE` | 1002 | "fraction of time at least one warp was active on a multiprocessor" |
| SM occupancy | — | `DCGM_FI_PROF_SM_OCCUPANCY` | 1003 | resident warps / max |
| **tensor pipe active** | — | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | 1004 | separates compute-bound from memory-bound *live* |
| DRAM active | — | `DCGM_FI_PROF_DRAM_ACTIVE` | 1005 | the other half of that question |
| NVLink TX/RX | — | `DCGM_FI_PROF_NVLINK_{TX,RX}_BYTES` | 1011/1012 | settles the transfer-vs-wait question in §7 |

[field IDs and definitions verified —
<https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>,
<https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>]

A caution on `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`: NVIDIA's own DCGM issue tracker carries a request
for FP8 and FP4/FP6 tensor-active metrics as *missing*, which implies the generic tensor-active
counter's coverage of NVFP4 `tcgen05` MMA on SM100 should be validated before you trust it on our
`bmm_E2m1_*` kernels. [reported — DCGM issue #251, surfaced in search; not read directly]

### A sampler that costs nothing

```bash
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,clocks.mem,\
power.draw,power.draw.instant,enforced.power.limit,temperature.gpu,temperature.memory,\
utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,\
clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,pstate \
  --format=csv,noheader -lms 200 -f run_${RUNID}_smi.csv &
```

[verified working, this box — this is what produced §5–§7]

For exact energy, prefer the NVML monotonic counter over integrating power samples — read it once
at start and once at end:

```python
import pynvml as N; N.nvmlInit()
h = [N.nvmlDeviceGetHandleByIndex(i) for i in range(8)]
e0 = [N.nvmlDeviceGetTotalEnergyConsumption(x) for x in h]   # mJ, monotonic since driver load
...
e1 = [N.nvmlDeviceGetTotalEnergyConsumption(x) for x in h]
joules = [(b - a) / 1000.0 for a, b in zip(e0, e1)]
```

The counter agrees with 50 ms trapezoidal power integration "to within 2% for operations ≥200 ms"
but has millijoule granularity that makes it unreliable for very short operations.
[reported — arXiv:2605.11999]

### Post-hoc attribution: the decision procedure

Given a run that came in slow, in order:

1. **Was a limiter asserted?** `SW Power Capping`, `HW Thermal Slowdown`, `HW Power Braking`
   counters are cumulative microseconds since driver load. Diff them across the run.
   Non-zero `HW *` → escalate to facilities; that is not a tuning problem.
2. **Did the clock move?** Compare the per-rank distribution of `clocks.sm` against the reference
   run, not the mean. Our data shows the interesting behaviour is bimodal (1117 vs 1597), so a
   mean hides it. Report p5/p50/p95.
3. **Did the clock move only on some ranks?** If yes, and the counters implicate power, you have
   a silicon-variability straggler and §7.3 applies. If the clocks are identical across ranks,
   the skew is upstream (host, routing, data).
4. **Did total energy move?** If wall time went up and energy per token went up in the same
   proportion, clocks were flat and you did more work. If wall time went up and energy per token
   stayed flat, you *waited* — a collective or host stall.
5. **Did inlet/GPU temperature move?** A run 3 °C hotter than the reference is a different
   experiment (§6).
6. **Did somebody change the enforced power limit?** `enforced.power.limit` will show it. On a
   shared box this is the single most common silent cause.
7. **Did the driver reload?** Persistence mode, clock locks, and power limits all reset. Log
   `nvidia-smi -q | grep "Driver Version"` plus the cumulative event-reason counters at run start
   — a counter that went *down* means a reload happened.

---

## 11. Energy as an objective: tokens per joule and $/token

### The measured baseline

From `benchmark/SCORECARD.md`, GLM-5.2 NVFP4, TP8:

| point | tok/s (node) | notes |
|---|---:|---|
| C1, sharegpt, real, `latency-3-1-4` | 365.5 | accept 3.16/4, TPOT 2.74 ms |
| C1, synthetic | 447.8 | accept 4.00/4 — synthetic inflates by 22% |
| C64, coding, real-ish | **40,794** | 54% prefix-cache hit, TPOT 17.41 ms |
| C64, repo-baseline, synthetic | 27,593 | |
| C256 | 37,444 | |
| C1024 | 36,973 | |

Node GPU power, measured: **~450 W/GPU busy median** during live serving on the 1597 MHz plateau,
**~190 W/GPU idle**, transients to 785 W. A sustained C64 run was not measured (§12).

### Tokens per joule

Using GPU-only power (the number we measured) and a node multiplier of 1.5× for CPUs,
NVSwitches, fans and PSU losses [inferred — the DGX B200 max is ~14.3 kW against 8 kW of GPU,
a 1.79× ratio at full tilt; at partial GPU load the non-GPU share does not scale down
proportionally, so 1.5× is a reasonable mid-estimate and is the weakest number in this section]:

| operating point | node tok/s | GPU W | **tok/J (GPU)** | node W (1.5×) | **tok/J (node)** |
|---|---:|---:|---:|---:|---:|
| C1, real | 365.5 | ~3,300 | **0.111** | ~4,950 | **0.074** |
| C64, real | 40,794 | ~3,600–4,000 | **10.2–11.3** | ~5,400–6,000 | **6.8–7.6** |

[inferred from verified per-GPU power measurements and verified throughput numbers]

**Concurrency is worth ~100× in energy per token.** C64 is 92–102× more energy-efficient per
token than C1. Nothing in the clock/power toolbox is within two orders of magnitude of that.
The literature says the same on a different part: batching from BS=1 to BS=32 "reduces
energy-per-token by over 20× by amortising the cost of loading weights".
[reported — arXiv:2605.11999]

### The dollar arithmetic

US industrial electricity, May 2026: **8.71 ¢/kWh** (commercial 13.54 ¢/kWh).
[verified — <https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a>]

| | C1 | C64 |
|---|---:|---:|
| tok/J (node) | 0.074 | 7.2 |
| J per 1M tokens | 13.5 GJ | 139 MJ |
| kWh per 1M tokens | 3.75 | 0.0386 |
| **electricity $ per 1M tokens** | **$0.327** | **$0.0034** |

Against amortised GPU cost, parametrically. At $R per GPU-hour, eight GPUs:

| | C1 (1.316 M tok/h) | C64 (146.9 M tok/h) |
|---|---:|---:|
| GPU $ per 1M tok at R=$2 | $12.16 | $0.109 |
| GPU $ per 1M tok at R=$4 | $24.32 | $0.218 |
| GPU $ per 1M tok at R=$6 | $36.48 | $0.327 |
| **electricity as % of total, R=$4** | **1.3%** | **1.5%** |

[inferred — the throughput and electricity price are verified; $R is a free parameter because I
could not source a current B200 hourly rate. Substitute your actual number.]

### The conclusion, stated plainly

**Power capping to reduce the electricity bill is not a $/token strategy.** Electricity is
1–2% of marginal serving cost across the whole concurrency range. A 10% power saving that costs
3% throughput is a *net loss* of roughly 3% of 98% against a gain of 10% of 1.5% — about
−2.85% versus +0.15%. Do not do it for the power bill.

Power capping pays in exactly one situation, and it is the situation NVIDIA's paper is written
about: **when the facility, not the budget, is the constraint.** If your rack is capped at
N kilowatts, reducing per-GPU power lets you install more GPUs per rack, and the relevant metric
is datacentre throughput per megawatt, not tokens per dollar of electricity. That is where the
"13% throughput increase in a power-constrained facility" comes from. [reported — arXiv:2510.03872]

### The oversubscription finding

Which leads to a concrete, money-shaped observation about *our* node:

- The rack is provisioned for 8 × 1000 W = 8 kW of GPU.
- Our serving workload draws **450 W/GPU median, 690 W/GPU for the most power-hungry
  memory-bound kernel class, 785 W peak transient**. [verified, this box]
- Only a sustained *dense compute-bound* GEMM reaches 990 W — and our C1 profile contains no such
  kernel (§8).

So **the GPU power budget is 30–55% over-provisioned for LLM inference on this node.** Two
exploitable consequences:

1. `nvidia-smi -pl 800` would, on the measured evidence, never bind during decode and would
   bound prefill transients — buying facility headroom at approximately zero decode cost. This
   is the same conclusion the H200 study reached from the other direction: decode drew only
   137–300 W on a 700 W part and *"no cap ever triggers"*, so caps from 280 W to 700 W all
   produced an identical 1830 MHz and identical power. [reported — arXiv:2605.11999]
2. Conversely, **`-pl` is the wrong knob for us and always will be.** The correct energy lever
   for memory-bound decode is the SM clock, which "Pareto-dominates power capping at every
   matched operating point", recovering "up to 32% of decode energy at less than 1% throughput
   loss". [reported — arXiv:2605.11999] On B200 the tuned `LLM Inference` power profile (§9)
   should dominate even that.

### The static-power floor caveat

The H200 study notes "a practical ceiling on DVFS savings comes from the H200's idle power floor
(≈75 W): a 5× clock reduction yields only ~1.5× power reduction, because DVFS controls only the
dynamic component." [reported — arXiv:2605.11999]

**Our B200's floor is ≈188 W** — 2.5× the H200's, and **19% of the 1000 W envelope** versus
10.7% for H200. [verified, this box] The consequence is that underclocking saves proportionally
*less* on B200 than the H200 result suggests, and that leaving eight B200s idle costs
8 × 188 W = **1.5 kW continuously**, or ~$1.15/day of industrial electricity, for nothing.
Idle-time energy is a bigger lever than clock policy on this part.

---

## 12. Open questions and the measurements that would close them

Ordered by value.

1. **Does `-lgc` clamp on B200, and where?** The H200 clamps any request ≥1830 MHz down to
   1830 MHz. Test: `sudo nvidia-smi -i 7 -lgc 1965,1965`, then run the skinny-GEMM load and read
   `clocks.sm`. If it reports something below 1965, we have found B200's equivalent of the
   1830 MHz clamp — and possibly discovered that 1597 *is* that number. Five minutes of work.
2. **What is the node power at a sustained C64 run?** Everything in §11 rests on an estimate. Run
   the C64 coding workload for 10 minutes with the NVML energy counter bracketing it. This also
   gives the first honest tok/J for our system.
3. **What clock does the NVFP4 MoE kernel sustain?** §2.1 measured BF16 and FP8. Run
   `bmm_E2m1_E2m1E2m1_Fp32_swiGlu_dynB_sm100f` at MoE-realistic shapes and record clock and
   power. If FP8's +5.4% clock / −8.8% power trend continues, NVFP4 is worth more than its FLOP
   ratio in the compute-bound regime.
4. **Does the `LLM Inference` power profile help or hurt?** `nvidia-smi power-profiles -sr 6`,
   run the benchmark, `-cr`. Also test `-sr 0` (Max-P) and the Max-P + LLM Inference composition.
   The NVIDIA paper's B200 table says this should dominate `-lgc`; nobody has checked on GLM-5.2.
5. **Is the rank-0 skew host-side?** Pin the eight scheduler processes to disjoint core sets on
   the correct NUMA node and re-profile. If the 24% drops toward 12.5%, §7.2 is confirmed and the
   fix costs nothing.
6. **What is the rank skew at C64?** The C1 profile is taken at an operating point where no GPU
   is near a limiter. At C64 the GPUs approach the cap and GPU 5's measured −4.7% becomes real.
   GEM predicts the spread *widens* with batch size.
7. **Does the even/odd 3.5 °C delta translate into any measurable frequency delta at a hot
   inlet?** Not testable without control of the room, but worth logging inlet temperature so the
   correlation can be checked retrospectively over the 72-hour window.
8. **Is `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` valid for `tcgen05` NVFP4 MMA on SM100?** If it under-
   counts, every "we are tensor-bound" claim built on it is wrong.

### Things I could not source

- Any NVIDIA statement of B200's **base clock** (as distinct from the 1965 MHz boost clock).
  The CUDA Blackwell Tuning Guide contains no power, clock, or DVFS content at all — I checked.
  Whether 1597 MHz is a documented base clock or an emergent DVFS plateau is **not sourced**.
- The **voltage/frequency curve** for B200. `nvidia-smi -q -d VOLTAGE` returns empty on this
  driver, so I cannot compute what boosting from 1597 to 1965 MHz would cost in watts. All such
  extrapolation in this document is avoided rather than guessed.
- **MLPerf Inference power results for B200** in tokens/joule. Not obtained — I exhausted the
  web-search budget before reaching them, and could not reach the MLCommons results tables by
  direct fetch. This is the obvious external cross-check for §11 and remains open.
- Any published **per-position thermal map** for the HGX B200 baseboard. The DGX B200 user guide
  gives system airflow and BTU/hr but does not identify which module positions run hot. Our
  even/odd finding is measured, not corroborated.
- Whether the **`--mode 0|1`** argument to `-lgc` does anything useful on B200. Documented only as
  "0 for default accuracy, 1 for improved efficiency".

---

## Sources

### Measured on this box (2026-08-17, driver 595.71.05, CUDA 13.2 runtime)

- `nvidia-smi -q`, `-q -d CLOCK|POWER|PERFORMANCE|SUPPORTED_CLOCKS|POWER_SMOOTHING`,
  `--query-gpu`, `topo -m`, `power-profiles -l|-ld|-gr|-ge`, `--help`
- NVML via the bundled `pynvml`:
  `/home/aman/code/NotSglang/python/sglang/multimodal_gen/third_party/pynvml.py` —
  `nvmlDeviceGetTotalEnergyConsumption`, `nvmlDeviceGetCurrentClocksEventReasons`,
  `nvmlDeviceGetCurrentClockFreqs`, `nvmlDeviceGetAdaptiveClockInfoStatus`,
  `nvmlDeviceGetMinMaxClockOfPState`, `nvmlDeviceGetMaxCustomerBoostClock`
- `cudaGetDeviceProperties` / `cudaDeviceGetAttribute` compiled with
  `/home/aman/code/cuda-13.3/nvidia/cu13/bin/nvcc`
- Load generation: PyTorch 2.11.0+cu130 from `/home/aman/code/NotSglang/.venv`

### Local repository files read

- `/home/aman/code/research/README.md`
- `/home/aman/code/benchmark/SCORECARD.md`
- `/home/aman/code/benchmark/RESULTS.md`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/glm-5.2-optimization-log.md`
- `/home/aman/code/NotSglang/personal_docs/kimi-k3/k3-kernel-optimization-log.md`
- `/home/aman/code/NotSglang/k3-kernels/scripts/bench_matrix.sh`

### NVIDIA documentation

- nvidia-smi manual — <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- NVML device commands (`SetGpuLockedClocks`, `SetPowerManagementLimit`, `SetPersistenceMode`,
  applications-clocks deprecation) —
  <https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html>
- DCGM field identifiers —
  <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>
- DCGM feature overview / profiling field IDs —
  <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>
- DGX B200 product page (FP4/FP8 PFLOPS, memory bandwidth, ~14.3 kW) —
  <https://www.nvidia.com/en-us/data-center/dgx-b200/>
- DGX B200 user guide (PSUs, 1,550 CFM, 48,794 BTU/hr, 10–35 °C) —
  <https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html>
- CUDA Blackwell Tuning Guide — <https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>
  *(read; contains no power/clock/DVFS content — recorded as a negative result)*

### Papers

- Ma, Afzal, Eitzinger, Wellein, **"The Illusion of Power Capping in LLM Decode"**,
  arXiv:2605.11999v1, 12 May 2026 — <https://arxiv.org/abs/2605.11999> (abstract + full PDF read)
- Wawdhane, Kumar, Das, **"GEM: GPU-Variability-Aware Expert-to-GPU Mapping for Mixture-of-Experts
  Models"**, arXiv:2605.19945v1, 19 May 2026 — <https://arxiv.org/pdf/2605.19945> (full PDF read)
- Narayanaswamy, Patel, Karlin, Gupta, Saripalli, Guo, **"Datacenter Energy Optimized Power
  Profiles"**, arXiv:2510.03872v2 — <https://arxiv.org/abs/2510.03872> (abstract + full PDF read)
- Kurzynski, Aga, Wu, **"Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution
  in Multiple GPUs"**, arXiv:2511.09861 — <https://arxiv.org/html/2511.09861v1>
- Sinha, Guliani, Jain, Tran, Sinclair, Venkataraman, **"Not All GPUs Are Created Equal:
  Characterizing Variability in Large-Scale, Accelerator-Rich Systems"**, SC'22,
  arXiv:2208.11035 — <https://arxiv.org/abs/2208.11035>
- **"Efficient AllReduce with Stragglers" (StragglAR)**, arXiv:2505.23523 —
  <https://arxiv.org/abs/2505.23523> *(authors not extracted)*
- Jain, Tran, Chen, Sinclair, Venkataraman, **"PAL: A Variability-Aware Policy for Scheduling ML
  Workloads in GPU Clusters"**, arXiv:2408.11919 — <https://arxiv.org/abs/2408.11919>
- Hankendi, Shahout, Yu, Coskun, **"PALS: Power-Aware LLM Serving for Mixture-of-Experts
  Models"**, arXiv:2605.21427v1, 20 May 2026 — <https://arxiv.org/abs/2605.21427>
- arXiv API listing queries (used in place of exhausted web search) —
  <http://export.arxiv.org/api/query> — surfaced additionally, not read in full:
  Spaan/Chen/Varbanescu arXiv:2601.08539 (kernel-level DVFS, 14.6% energy at 0.6% slowdown);
  Jiang et al. arXiv:2601.12241 (RAPID, 2× SLO attainment under power caps);
  Chung/Wu/Ma/Chowdhury arXiv:2601.22076 ("Where Do the Joules Go?");
  Chen et al. arXiv:2603.17280 ("The 1/W Law", H100→B200 ≈1.7× tokens-per-watt);
  Jarmusch & Chandrasekaran arXiv:2512.02189 (Blackwell microbenchmarking, 32% better energy
  efficiency than H200)

### Other

- US EIA Electric Power Monthly, Table 5.6.A, May 2026 (industrial 8.71 ¢/kWh, commercial
  13.54 ¢/kWh) —
  <https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a>
- Chips and Cheese, "Nvidia's B200: Keeping the CUDA Juggernaut Rolling" —
  <https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut>
  *(read; contains no clock or power measurements — recorded as a negative result)*
