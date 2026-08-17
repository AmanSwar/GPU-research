# Ranked opportunities: what to work on tomorrow morning

**Date:** 2026-08-17. **Target:** top the Artificial Analysis board for GLM-5.2 on
8×B200 without wrecking the cost plane.

> **⚠ Correction (2026-08-17, later the same day).** The trace archived at
> `benchmark/runs/sweep-latency-3-1-4/trace.sqlite` and cited throughout this
> document as "the C1 profile" is **not a C1 capture**: the file was overwritten
> at 03:11 today, minutes after the AA 10-parallel run, and now holds the **C10**
> capture (verified by mtime and by four independent signatures in the trace
> itself). Every profile-*derived* claim below — the 16.3% idle, the 65.4% SM
> packing, the 10.7% `gridDim==1` share, and the 1.53×/558 tok/s packing ceiling —
> is therefore **dead or unverifiable until a true C1 re-capture is taken**. The
> bench-derived numbers (6.91 ms zero-draft forward, the ~1.4× published gap, all
> SCORECARD rows) are unaffected. The successor document,
> [`03-glm-5.2-performance-gaps.md`](03-glm-5.2-performance-gaps.md), rebuilds the
> analysis on the correctly-attributed C10 trace and supersedes this one where
> they disagree.

This is the payoff document for the corpus. Everything else in
`/home/aman/code/research` is input. It is written to be read by one person
deciding what to start on, so every entry answers six questions: what is the
mechanism, which slice of the measured profile does it attack, what does Amdahl
allow it to return, what is the effort class, what is the risk, and what single
measurement confirms or kills it.

**Confidence labels are on every substantive claim.** `[verified]` = primary
source (engine source read on this box, driver probe, our own trace or run log).
`[reported]` = a vendor or paper asserts it. `[inferred]` = arithmetic from
verified inputs, shown. `[unverified]` = flagged, not checked.

**Read §2 (corrections) before §4 (the ranking).** Six things the existing
ledger and corpus say are wrong or misleading, and four of them change the order
of the list. Two of the "free flag flips" in the brief are already on. The item
that three separate corpus documents call the biggest number on the page is worth
about a fifth of what they claim, and the reason why is the most useful thing in
this file.

---

## 0. How to read this, and the two numbers everything is scored against

### 0.1 Two planes, and they fight

| plane | metric | measured today | reference |
|---|---|---:|---|
| **LATENCY** (the headline) | output tok/s, single prompt, AA shape (10k in / ≥1.5k out, temp 0.6), **real data** | **349.5 tok/s**, TTFT 210 ms `[verified]` | Databricks 336, Makora NVFP4 330 (board 2026-08-16) `[reported]` |
| **LATENCY** (AA's second arm) | output tok/s/stream at **10 concurrent**, same shape, real data | **159.3 tok/s**, TTFT 222 ms `[verified]` | not published per-provider |
| **COST** | aggregate tok/s per node at serving concurrency | **40,794 tok/s** at C64 on a coding workload, 54.2% prefix hit `[verified]` | unmeasurable — no node price set |

Source: `benchmark/SCORECARD.md` §1a, §1c. Run-to-run variance is **~4%**
`[verified]`, so any predicted effect below 4% is unfalsifiable in one run and
must be measured with repeats or with a mechanism counter instead of tok/s.

**The planes conflict and the conflict is structural, not incidental.**
Speculative decoding buys latency by burning FLOPs and expert-weight traffic;
prefix caching buys both; deeper drafts, DP attention and TBO each help one plane
and hurt the other. §6 states each conflict explicitly. A table that pretends
they compose is worse than no table.

### 0.2 The two profiles are not the same configuration

Every share quoted in this corpus comes from one of two nsys captures. They
differ in **four** ways, not one, and the corpus consistently attributes the
difference to concurrency alone. Resolved from the actual run logs `[verified]`
(`benchmark/runs/sweep-latency-3-1-4/server.log`,
`benchmark/runs/sweep-capacity/server.log`):

| | **capture A ("C1")** | **capture B ("C64")** |
|---|---|---|
| mode | `latency-3-1-4` | `capacity` |
| concurrency | 1 | 64 |
| parallelism | **TP8, `dp_size=1`, `enable_dp_attention=False`** | **TP8 + `dp_size=8`, `enable_dp_attention=True`** |
| draft | EAGLE **3-1-4** | EAGLE **2-1-3** |
| FlashInfer allreduce fusion | **`flashinfer_allreduce_fusion_backend='auto'` — ON** | **`= None` — OFF** |
| data | synthetic aa-10k | sharegpt (real) |
| device 0 busy | 83.7% | 75.2% |

So "collectives grow 19.6% → 25.4% with concurrency" is not a fact about
concurrency. It is a fact about a configuration that simultaneously turned on DP
attention (which adds an AllGather and a ReduceScatter per MoE layer) and turned
off allreduce fusion (which had been absorbing the AR into the RMSNorm). Any
C1↔C64 delta in this corpus is a four-variable delta.

### 0.3 The Amdahl budget, with the busy-fraction discount applied

Kernel shares are shares of **GPU-busy** time. Device 0 is busy 83.7% of wall at
C1, so a slice of share `s` is `0.837·s` of wall clock, and deleting it entirely
gives a speedup of `1/(1 − 0.837·s)`. Nobody in the corpus applies this
discount; it costs 3–5 points off every ceiling.

**C1 profile (latency mode, TP8, 3-1-4, fusion ON), 11.3M kernels, all 8 ranks** `[verified]`:

| slice | share of GPU time | share of wall | ceiling if **deleted** | 349.5 tok/s → |
|---|---:|---:|---:|---:|
| dense GEMM (cuBLAS `nvjet_sm100_*`, **BF16**) | 37.1% | 31.0% | **1.450×** | 507 |
| collectives | 19.6% | 16.4% | 1.196× | 418 |
| — of which *waiting* (arrival skew) | 10.6% | 8.9% | **1.097×** | 383 |
| MoE NVFP4 expert GEMMs | 19.4% | 16.2% | 1.194× | 417 |
| attention (DSA sparse MLA) | 10.9% | 9.1% | 1.100× | 385 |
| DSA indexer | 5.8% | 4.9% | 1.051× | 367 |
| elementwise | 3.7% | 3.1% | 1.032× | 361 |
| quant | 2.4% | 2.0% | 1.021× | 357 |
| device-0 **idle** | — | 16.3% | 1.195× if fully recovered | 418 |

**C64 profile (capacity mode, DP8, 2-1-3, fusion OFF), 3.6M kernels** `[verified]`,
busy 75.2%:

| slice | share of GPU time | share of wall | ceiling if deleted |
|---|---:|---:|---:|
| dense GEMM | 31.7% | 23.8% | 1.313× |
| MoE expert GEMMs | 30.8% | 23.2% | 1.302× |
| collectives | 25.4% | 19.1% | 1.236× |
| — of which *waiting* | 11.2% | 8.4% | **1.092×** |
| attention | 4.6% | 3.5% | 1.036× |
| DSA indexer | 2.4% | 1.8% | 1.018× |

Three consequences to keep in front of you:

1. **Nothing on the attention or indexer axis can return more than 10% / 5% at
   C1, and 4% / 2% at C64.** The corpus spends 2,800 lines on those two files.
   They are the wrong place to work.
2. **No single slice reaches 500 tok/s.** Deleting *all* collectives gets 418.
   Deleting *all* dense GEMM gets 507 — and dense GEMM cannot be deleted.
   Reaching 500 requires two or three of these composed, or a structural change
   that shortens every kernel at once (§5.B1).
3. **The idle 16.3% at C1 (24.8% at C64) is the same size as the entire
   collectives bucket and nobody has attributed it.** See §7 item 1.

### 0.4 The one arithmetic that reframes the whole list

From our own measurements, decompose the C1 step `[inferred, from verified rows
in SCORECARD §1b]`:

```
no speculation, TPOT             = 6.91 ms   → a full forward pass costs 6.91 ms
2-1-3, accept 3.00, TPOT 2.76    → verify cycle = 3.00 x 2.76 = 8.28 ms
3-1-4, accept 4.00, TPOT 2.23    → verify cycle = 4.00 x 2.23 = 8.92 ms
  => one extra draft forward + one extra verify token costs 0.64 ms
  => cross-check: 6.91 + 3 x 0.64 = 8.83 ms vs measured 8.92 ms  (1% agreement)
```

So: **a full forward is 6.9 ms and a draft forward is 0.64 ms, i.e. 9.3% of a
full one.** Now put floors under the 6.9 ms:

```
HBM floor    : 8,756 MB/token/rank (NVFP4 build, [inferred] 05-models/01 §3.2)
               / 7.67 TB/s ([verified] driver-derived)          = 1.14 ms
launch floor : ~1,082 kernels/pass/rank ([inferred] 01-kernel-opt/02)
               x 0.70 us minimum in-graph kernel period ([verified, measured])
                                                                = 0.76 ms
measured                                                        = 6.90 ms
```

**The forward pass is 6× its bandwidth floor and 9× its launch floor.** Neither
bytes nor launch count explains it. What does explain it, measured: the mean
kernel duration is **6.78 µs** `[verified]`, and an FP8 GEMM at our exact decode
shape (N=6144, K=2048) takes **3.7–6.0 µs for every M from 1 to 256**
`[verified, measured on this box]` — a fixed floor independent of work. The step
is a serial chain of ~1,000 kernels that each cost 4–7 µs regardless of how much
work they do.

**Therefore every byte-reduction lever in this corpus is scored far too
optimistically.** Halving the weight bytes moves a 1.14 ms term inside a 6.9 ms
step: **at most 8%, and only if that term is on the critical path at all.** The
claims of "+45% to +65%" (05-models/05) and "1.75× cut in the decode bandwidth
floor" (03-papers/08) are statements about the *roofline*, and we are at 24% of
the roofline. They are not predictions about measured throughput. §2.2 and §5.A1.

---

## 1. Executive table — top 16 by expected value

EV ≈ (expected gain × probability it works) ÷ effort. Effort classes:
**flag** (flag or env flip) · **config** (a run with different args, no code) ·
**port** (bounded code change in a known place) · **kernel** (new CUDA/CuTe) ·
**research** (open-ended).

Gains are on the **latency** plane (single-stream, AA shape, real data) unless
the row says otherwise.

| # | opportunity | plane | slice attacked | Amdahl ceiling | expected | P | effort | conf |
|---|---|---|---|---|---|---|---|---|
| **1** | **FP8-checkpoint A/B at C1** — settle whether bytes matter before spending weeks on requantization | decides latency | dense GEMM 37.1% | n/a (a measurement) | −5% to +20%, and it **re-ranks items 4, 8, 12** | 1.0 | **config** | `[verified]` inputs |
| **2** | **Unblock EAGLE 5-1-6**: `--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none` | latency | accept length, not a slice | accept 3.16→~3.8 | **+2% to +7%** (not the +15–25% the corpus claims — see §2.2) | 0.55 | **flag** | `[verified]` cause, `[inferred]` size |
| **3** | **Repair the overlap record and re-run SBO / fusion as one-variable A/Bs** | both | collectives 19.6 / 25.4% | up to 1.09–1.24× | recovers a 12.5% loss or closes it | 0.9 | **config** (2 runs) | `[verified]` |
| **4** | **Cut kernels per forward pass** — the gridDim==1 tail first | latency | *all* slices simultaneously | **1.53×** measured packing ceiling → 558 tok/s | +5% first pass, 1.2–1.5× if pursued | 0.6 | port → kernel | `[verified, measured]` |
| **5** | **Explain and fix the un-graphed AllReduce at C64** (10.5% of GPU time, **0%** graph-captured) | cost | collectives 25.4% @C64 | 1.13× at C64 | +3% to +10% aggregate | 0.5 | trace, then port | `[verified]` |
| **6** | **Step-time bake-off vs SGLang's published cell at fixed accept** (`SGLANG_SIMULATE_ACC_LEN=3.5`) | decides latency | the whole forward pass | n/a | localises a **1.4× forward-pass gap** | 1.0 | **config** | `[verified]` |
| **7** | **Turn allreduce fusion back on under DP attention at C64** — the auto-enable pass declines to | cost | AllReduce 10.5% @C64 | 1.11× at C64 | 0% to +8% aggregate | 0.45 | **flag** | `[verified]` gating |
| **8** | **Hybrid precision: FP8 for the BF16 non-expert GEMMs of the NVFP4 build** | both | dense GEMM 37.1% | 1.18× if bandwidth-bound; **≤1.08× if not** | +3% to +15% | 0.5 | **port** | `[verified]` state, `[inferred]` size |
| **9** | **Launch-structure fusion flags** — `--enable-fused-qk-norm-rope`, `--enable-fused-moe-sum-all-reduce`, both off today | latency | elementwise 3.7% + launch count | 1.03× on the slice; more on launch count | 0% to +4% | 0.5 | **flag** | `[verified]` off |
| **10** | **`--speculative-adaptive`** — batch-adaptive draft depth; the principled fix for the C1↔C10 conflict | both | resolves a conflict | ~+6% avg `[reported]` | +0–6% at C1, +5–15% at C10–C64 | 0.6 | **flag** | `[verified]` present |
| **11** | **MoE down-projection pipeline under-fill** — `K=256` per rank is exactly one `tileK` at TP8 | cost | MoE 30.8% @C64 | 1.05× at C64 | +2% to +5% aggregate | 0.5 | **kernel** | `[verified]` shape |
| **12** | **`SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1`** (covers `q_b_proj` only, 3.7% of decode bytes) | latency | dense GEMM 37.1% | ≤1.02× | 0% to +1.5% | 0.4 | **flag** | `[verified]` |
| **13** | **Unblock two-batch overlap** — **two** blockers, not one, and it is worth zero at C1 | **cost only** | collective waiting 11.2% @C64 | 1.09× at C64 | +3% to +9% at C64+ | 0.4 | **port** | `[verified]` blockers |
| **14** | **Prefix-cache hit rate, and the EAGLE↔radix interaction** | cost (+ TTFT) | prefill, not decode | TTFT only | TTFT −20% to −50% on repeat traffic | 0.5 | config → port | `[reported]` risk |
| **15** | **Dead DSA indexer KV: 78 layers allocated, 22 read** | **cost only** | KV capacity, not time | 0% at our operating points | 0% now; +12% KV capacity | 0.9 | **port** | `[verified]` |
| **16** | **Persistent / megakernel decode step** | latency | all slices | **1.53×** | 1.2–1.5× | 0.3 | **research** | `[inferred]` |

Items 1, 3, 6 are measurements rather than optimizations. They rank at the top
because each costs under a day and each one re-ranks several items below it. The
alternative — starting item 8 (weeks of loader work) before item 1 (one run) —
is exactly the trap that Phase 0 of the ledger's own methodology exists to
prevent.

---

## 2. Corrections to the ledger and the corpus

These are not nitpicks. Four of the six change the ranking.

### 2.1 FlashInfer allreduce fusion was **already on** at C1, and is **structurally off** at C64

The brief and the ledger (§3) both state that all three overlap mechanisms were
disabled in every measurement, and that "SBO and fusion are free flag-flips".
The second half is false.

`[verified]` from `NotSglang/python/sglang/srt/arg_groups/overrides.py:1785-1807`:
a post-process pass auto-enables the fusion when the architecture is in
`_FLASHINFER_ALLREDUCE_FUSION_ARCHS` (which contains `GlmMoeDsaForCausalLM`,
line 1768), on SM100, with `tp_size > 1`, `nnodes == 1`,
`moe_a2a_backend == "none"`, **and `not enable_dp_attention`**.

`[verified]` from our own run log,
`benchmark/runs/sweep-latency-3-1-4/server.log`:

```
[2026-08-17 03:04:28] Auto-enabling FlashInfer AllReduce Fusion on SM90/SM10X for GlmMoeDsaForCausalLM
...
flashinfer_allreduce_fusion_backend='auto'
```

and from `benchmark/runs/sweep-capacity/server.log`:
`flashinfer_allreduce_fusion_backend=None`.

**So the C1 profile — the one that produced "47% of collective time is arrival
skew" — was taken with the fusion enabled**, which is why
`tllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` appears in it at all. And
the C64 profile was taken with the fusion *disabled*, because turning on DP
attention silently suppresses the auto-enable. That is the resolution of the
"unresolved contradiction that blocks every collective measurement" flagged in
`05-models/05` §9: there was no contradiction, only an unread gating condition.

Three consequences:

- The 47%/44% skew figures are a fair reading of SGLang-with-fusion at C1 and
  SGLang-without-fusion at C64. The pessimism about SGLang in ledger §3 should be
  withdrawn for C1.
- **Fusion is not a free flag-flip at C1. It is already spent.** Item 3 of the
  brief loses one of its three legs.
- **At C64 it is a free flag-flip, and nobody has tried it in isolation** — see
  item 7. The C64 top-10 contains `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` at
  10.5% precisely because the fused path is off there.

### 2.2 Draft depth 5-1-6 is worth ~2–7%, not 15–25%

Three corpus documents rank this first (`05-models/01` §1: "the single biggest
number on the page"; `04-industry/08` §1: "+15–21%"; `04-industry/02` §1: "even
half the gap is ~+80 tok/s"). Their arithmetic uses published *accept lengths*
and assumes the extra draft forwards are nearly free. Our own measurements say
otherwise.

`[inferred]` from the decomposition in §0.4 (base forward 6.9 ms, marginal draft
forward 0.64 ms), on **real** data where our measured accept at 3-1-4 is 3.16:

| config | cycle time | accept | tok/s | vs today |
|---|---:|---:|---:|---:|
| 3-1-4 (today, real sharegpt) | 8.66 ms `[verified]` | **3.16** `[verified]` | **365.5** | — |
| 4-1-5 | 6.8 + 4(0.64) = 9.36 ms | ~3.52 | 376 | **+3%** |
| 5-1-6 | 6.8 + 5(0.64) = 10.00 ms | ~3.82 | 382 | **+4.5%** |
| 5-1-6, optimistic accept | 10.00 ms | 4.10 | 410 | +12% |

Accept projections use SemiAnalysis's golden acceptance curve for GLM-5.2 MTP
(k=3 → 2.99, k=4 → 3.33, k=5 → 3.61) `[verified from a published YAML]`, scaled
by the +5.7% our measured 3.16 sits above their k=3 point.

**The mechanism of the demotion: accept length grows sublinearly in draft depth
while draft cost grows linearly.** Going 3→5 steps adds 14.8% to the cycle and
buys ~21% more accepted tokens. The margin is thin and it is inside 2× of our 4%
noise floor.

**Do it anyway** — it is two flags (item 2) and the sign is positive — but stop
treating it as the headline, and stop planning a quarter around it.

And the corollary that matters more: **the published 540 tok/s is not a
draft-depth result.** SGLang's in-repo verified cell for `nvidia/GLM-5.2-NVFP4`
on 8×B200 TP8 is `tpot_ms: 1.85` at 5-1-6 with `SGLANG_SIMULATE_ACC_LEN=3.5`
`[verified, 04-industry/08 §2]`. That is a **6.48 ms verify cycle**, which is
*less than our 6.91 ms zero-draft forward pass* `[verified]`. Five draft forwards
and a 6-token verify fit inside a window shorter than one of our forward passes.
**The gap is the forward pass, by roughly 1.4×, and it has nothing to do with
speculation.** That is what item 6 exists to localise, and it is the single most
important open question in this document.

### 2.3 The 1597 MHz "19% haircut" is not recoverable and is not a lock

`00-hardware/00` says the benchmark runs at 81% of peak clock and calls it "a
standing 19% haircut on every compute-bound number we publish". `00-hardware/07`
measured the same box and contradicts it `[verified, this box]`:

- **The clocks are not locked.** Idle GPUs sit at 120 MHz; a lightly loaded one
  boosts to 1965 MHz; a compute-saturating GEMM is power-capped down to
  **1072–1222 MHz**. A real `-lgc 1597,1597` would hold 1597 in all three states.
- **1597 MHz is a DVFS plateau**, reached by every memory-bound kernel class
  independently (d2d memcpy, decode-shaped skinny GEMM, elementwise SiLU).
- **1000 W is the binding constraint on dense FLOPs, not thermals** — sustained
  BF16 GEMM pins every GPU at ~990 W with `SW Power Cap` continuously asserted,
  at 52–61 °C.

Our decode step is latency-bound (§0.4), so it sits at the plateau because there
is nothing for a higher clock to do. "Unlocking" the clock would push a
compute-bound phase *down* toward 1100 MHz, not up toward 1965. The honest
residual levers are small and named in §7 (`--vboost`, the `LLM Inference` power
profile). Expected: 0–3%.

### 2.4 The L2 persisting window was measured and it lost

`00-hardware/00` calls the 79.06 MiB persisting window "the mechanism for keeping
a hot working set off HBM entirely" and the brief lists it as an untaken lever.
`00-hardware/02` measured it on this box `[verified, measured]`:

- `cudaAccessPolicyWindow` bought **+3% at best and hurt by 5%** when the
  carve-out was sized to the full 79 MiB. The B200 default L2 replacement policy
  already keeps a 48 MiB hot set resident against a 1 GiB stream.
- **Effective L2 for data shared by all 148 SMs is ~63 MiB, not 126.5 MiB** — two
  independent measurements agree (single-SM pointer-chase knee 48–63 MiB; all-SM
  shared-read knee 56–80 MiB).
- The obvious per-instruction alternative is blocked: `.L2::evict_first` /
  `.L2::evict_last` **require a 256-bit vector type** (`.v8.b32` / `.v4.b64`) on
  sm_100; every narrower form is rejected by ptxas 13.3.

This belongs in "not worth it" (§7), not in the opportunity list. It is a closed
negative result and should get a row in the ledger's §4 saying so.

### 2.5 Two-batch overlap has two blockers, and the second one is expensive

The brief says unblocking TBO is "a bounded code change" against the
`index_topk_freq=4` raise. That raise is real `[verified,
server_args.py:5053-5063`]. But there is a second gate `[verified,
server_args.py:8601-8615]`:

```python
def _check_two_batch_overlap(self):
    if (self.enable_two_batch_overlap
            and self.moe_a2a_backend == "none"
            and not self.enable_dp_attention):
        raise ValueError(
            "When enabling two batch overlap without an EP a2a backend "
            "(moe_a2a_backend='none'), --enable-dp-attention is required ...")
```

Our latency mode is TP8 with `moe_a2a_backend='none'` and
`enable_dp_attention=False` `[verified from the run log]`. So TBO in latency mode
needs **both** the topk-index propagation *and* either DP attention or an EP
all-to-all backend. And turning on DP attention **disables the allreduce fusion**
(§2.1). The three overlap mechanisms are interlocked, not independent:

```
TBO  ⇒  needs DP-attention (at moe_a2a_backend=none)  ⇒  kills allreduce fusion
```

In capacity mode DP attention is already on, so there the topk-index propagation
is the only blocker — which makes TBO a **cost-plane project only**. That is
consistent with the ledger's own note that TBO cannot help C1 (no batch to
split) and with Perplexity's measurement that dual-batch overlap is +18% at batch
128 but **negative at batch 32** `[reported]`.

### 2.6 Things the brief lists as untried that are already on

`[verified]` from the resolved `ServerArgs` in
`benchmark/runs/sweep-latency-3-1-4/server.log`:

| item | brief's assumption | actual state |
|---|---|---|
| FlashInfer allreduce fusion | off | **`backend='auto'` at C1** (off at C64) |
| PDL | "compiled in, gated OFF by default" (01-kernel-opt/05) | **on** — `engine.py:1526` sets `TRTLLM_ENABLE_PDL=1` unless explicitly `"0"` |
| `--bf16-gemm-backend cutedsl` | a flag to flip (04-industry/02 §2) | **`'auto'`, which selects `cutedsl` on SM100/SM103** per its own help text |
| per-rank NUMA binding | untried lever | **`SGLANG_NUMA_BIND_V2` defaults `True`**; the node is auto-queried per GPU and applied by wrapping the subprocess in `numactl`. Success is logged at `debug`, which is why our `info`-level logs show nothing |
| shared-expert fusion | a knob | **forced off** — `moe_runner_backend='flashinfer_trtllm'` sets `disable_shared_experts_fusion=True` (`overrides.py:2290-2294`), logged verbatim in our run |

And one that is genuinely off and matters: **`speculative_moe_runner_backend='flashinfer_trtllm'`** `[verified]` — see item 2.

One accuracy warning we have never chased, printed by every rank in every run
`[verified]`: `DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0.
This might cause accuracy degradation on Blackwell.`

---

## 3. Sequencing

Do the measurements first. Each of items 1, 3, 6 is one or two runs and each
re-ranks what follows.

| day | do | why here |
|---|---|---|
| 1 | **item 1** FP8 checkpoint at C1, same 3-1-4, same AA-real workload | one run; decides whether items 8 and 12 exist at all |
| 1 | **item 6** simulated-accept step-time bake-off vs the published cell | one run; localises the 1.4× forward-pass gap |
| 2 | **item 2** the two spec-MoE flags at 5-1-6 | one flag pair; either unblocks depth or closes it |
| 2 | **item 3** SBO alone, then fusion-under-DP alone, at C16 real data | two runs; repairs the evidence for every collective claim |
| 3 | **item 5** un-graphed AllReduce, from the traces we already hold | no new run |
| 3 | **item 9**, **item 12** cheap flag flips with kernel-name verification | 2 runs |
| 4–5 | **item 7** fusion at C64; **item 10** adaptive spec | flags, conditional on item 2 |
| week 2 | **item 4** kernel-count reduction, starting with the gridDim==1 kernels | the largest measured ceiling in the corpus |
| week 2+ | **item 8** hybrid precision — *only if item 1 said bytes matter* | weeks of loader work otherwise wasted |
| later | items 11, 13, 14, 15, 16 | see entries |

---

## 4. Detailed entries — latency levers

### Item 1. FP8-checkpoint A/B at C1 — the decisive cheap experiment

**Mechanism.** `/home/aman/code/weights/GLM-5.2-FP8` is complete on this box: 141
shards, 755.63 GB, an HF snapshot of `zai-org/GLM-5.2-FP8` `[verified]`. It is
the *same model* as the NVFP4 build — identical architecture, identical 753.33 B
parameters to the byte — quantized on a completely different set of modules
`[verified, 05-models/00 §4.1, §5]`. The FP8 build quantizes essentially every
GEMM; the NVFP4 build quantizes **only** `model.layers.{3..77}.mlp.experts.*` and
leaves all attention, all shared experts, the three dense layers, the indexers,
the router and the whole MTP layer in **BF16 — at exactly twice their FP8 size**.

Per-token per-rank decode bytes `[inferred, 05-models/01 §3.2]`:

| build | bytes/token/rank | HBM-floor ceiling |
|---|---:|---:|
| FP8 checkpoint | **6,674 MB** | 1199 tok/s |
| NVFP4 as shipped | **8,756 MB** | 914 tok/s |
| NVFP4 with non-expert GEMMs at FP8 | 6,048 MB | 1323 tok/s |

So the "FP4" build reads **31% more bytes per token** than the "FP8" build.
Running the FP8 build unchanged at 3-1-4 on the AA-real workload is a single
config run that brackets the entire byte-reduction program from below.

**Slice and ceiling.** Attacks dense GEMM (37.1% of C1 GPU time, 31.0% of wall).
If decode at C1 were bandwidth-bound, −24% bytes would be ~−24% on the HBM term.
Per §0.4 the HBM term is 1.14 ms of a 6.90 ms step, so the *predicted* effect is
**1.14 × 0.24 = 0.27 ms = +4%**. If the measured effect is much larger than 4%,
our launch-bound diagnosis is wrong and item 8 becomes the top priority. If it is
~0, item 8 is dead and weeks are saved.

**Expected.** **−5% to +20%**, deliberately wide. Countervailing forces: FP8
experts read 2× the bytes of FP4 experts (experts are 18.2% of decode bytes
`[inferred]`), FP8 needs a dequant path the FP4 kernels avoid, and the FP8 MoE
runner is a different kernel family. Two of the three point the wrong way, which
is why this is a measurement and not a proposal.

**Effort: config.** Weights are on disk. Per-GPU footprint 89.37 GiB vs NVFP4's
56.72 GiB at TP8 `[inferred]` — fits in 170.5 GiB, and the KV pool at
`mem-fraction-static 0.85` will shrink accordingly (today it is 1,744,128
tokens/rank at 1% utilisation, so there is no risk of running out).

**Risk.** None to production; it is a separate server launch. Accuracy is
*better*, not worse (FP8 is the vendor's own build; GSM8K reference 98.2 at
FP8/H200 `[reported]`).

**Confirm or kill.** `gpu-bench run --mode latency-3-1-4 --workload aa-10k-real
--concurrency 1` against each checkpoint, ≥2 repeats each to clear the 4% noise
floor. Read TPOT and accept length. **Kill criterion for item 8:** if the FP8
build's TPOT is within 4% of NVFP4's, the byte-reduction program returns nothing
at C1 and should be re-scoped to the cost plane only.

**Confidence.** `[verified]` for every input; `[inferred]` for the +4% prediction.

---

### Item 2. Unblock EAGLE 5-1-6 with the spec-MoE backend flags

**Mechanism.** Our draft-depth ceiling is a crash, not a kernel: 4-1-5 and 5-1-6
both die with an illegal memory access at `eagle_worker_v2.py:366` during draft
CUDA-graph capture; `--speculative-attention-mode decode` only moves the crash
line `[verified, ledger §4 #3–#6]`. Three verified facts identify the cause:

1. `speculative_moe_runner_backend='flashinfer_trtllm'` in our resolved args
   `[verified, our run log]`.
2. The protective default that would have set it to `triton` **does not apply on
   CUDA**: `_deepseek_spec_moe_resolution` in
   `arg_groups/overrides.py:1663-1676` returns `{}` unless `is_hip()`
   `[verified, read the source]`. So the draft MoE inherits the target's
   Blackwell backend.
3. The NVFP4 build leaves layer 78 — the whole MTP layer — in **BF16**
   `[verified, 05-models/00 §9.5]`. So the draft's expert GEMM is a
   `Bmm_Bfloat16..._dynB_sm100f` flashinfer TRT-LLM cubin — exactly the kernel
   named in the IMA of SGLang issue #30209, which is our configuration
   (GLM-5.2 NVFP4, `GlmMoeDsaForCausalLM`, 8×B200, TP8, EAGLE 5-1-6)
   `[reported, 03-papers/02 §1, read from the issue]`.

The fix reported there, A/B-validated on mirrored production traffic (control
crashed 5× in 5 h; treatment 0 restarts in 5 h, 24 h+ crash-free), is
`--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none`
`[reported]`. Both flags exist in our fork with those exact names
`[verified, server_args.py:2141, 2150]`.

**Slice and ceiling.** Not a profile slice — a cap on accept length. Ceiling is
set by the acceptance curve, and per §2.2 it is **+2% to +7%**, not +15–25%.

**Expected.** 365.5 → **373–392 tok/s** at C1 on real sharegpt; 349.5 → 357–374
on the AA shape. One upside risk and one downside risk, both real:
- *Upside*: on long-reasoning traffic (which AA's temp-0.6 reasoning-max mode
  produces), published 5-1-6 accept reaches 4.61–5.18 `[reported, SGLang PR
  #29787, TP4, temp 0]`. If our real accept lands near 4.3 instead of 3.8, this
  is +18% not +5%.
- *Downside*: the triton draft-MoE runner may be slower per draft forward than
  the trtllm one. Our marginal draft cost is 0.64 ms of an 8.66 ms cycle (7.4%);
  a 50%-slower draft step costs 3.7% of the cycle and eats most of the gain.

**Effort: flag.** `benchmark/gpubench/config.py:111-135` already holds
`LATENCY_MODE` at byte-identical 5-1-6 flags for exactly this purpose; add the
two flags and run it.

**Risk.** Accuracy: none in principle — `speculative_accept_threshold_single=1.0`
and `speculative_accept_threshold_acc=1.0` `[verified]`, so verification is
exactness-preserving and a deeper draft cannot change outputs. Operational: the
triton path is less exercised on this arch. Cost plane: a deeper draft is
*negative* at concurrency (§6.1) — this must be latency-mode only.

**Confirm or kill.** Server starts and captures graphs → measure accept length on
sharegpt at C1 and the per-draft-step cost (from the TPOT/accept decomposition in
§0.4). **Three kill criteria:** still an IMA; accept < 3.4 at 5-1-6; or cycle
time up more than accept.

**Confidence.** `[verified]` that the backend is `flashinfer_trtllm` and that the
protective default is HIP-gated. `[reported]` that the flag pair fixes it.
`[inferred]` the effect size.

---

### Item 3. Repair the overlap record; re-run SBO and fusion as one-variable A/Bs

**Mechanism.** Ledger §4 row #9 records "SBO + allreduce fusion on" as
−12.5% throughput at C16 and concludes SBO restructures the forward pass without
delivering overlap. The run is `capacity-overlap`, which is `CAPACITY_MODE.flags
+ ("--enable-single-batch-overlap", "--enable-flashinfer-allreduce-fusion")`
`[verified, benchmark/gpubench/config.py:169-177]`.

Two problems, both `[verified]`:

1. **Capacity mode has DP attention on, so the fusion auto-enable is
   suppressed** (§2.1). Passing `--enable-flashinfer-allreduce-fusion` therefore
   *turned fusion on where the engine deliberately declines to*. Our own log
   confirms the flag was accepted as a deprecated alias:
   `[2026-08-16 16:11:26] --enable-flashinfer-allreduce-fusion is deprecated.
   Please use --flashinfer-allreduce-fusion-backend=auto instead.`
2. So the −12.5% is a **two-variable** result, and the second variable is a
   configuration the engine's own heuristics reject. The ledger's own note ("two
   flags changed together, so the fusion half is confounded") understates it: the
   fusion half is not merely confounded, it is a config the auto-enable pass was
   written to avoid.

**Slice and ceiling.** Collectives: 19.6% of GPU time at C1, 25.4% at C64.
Ceiling on the *waiting* component alone is 1.097× at C1 and 1.092× at C64.

**Expected.** The A/Bs themselves are worth 0%; what they buy is a correct base
of evidence and, plausibly, the recovery of a 12.5% loss that was attributed to
the wrong cause. Non-trivial chance that **fusion-on-under-DP-attention is the
whole −12.5%**, in which case SBO alone is neutral or positive and item 7 is dead.

**Effort: config** — two runs at C16 on real sharegpt, one variable each.

**Risk.** None.

**Confirm or kill.** Use `waiting_ms` and `max_skew_us` from `gpu-bench
deepdive`, not tok/s, as the primary metric (Loop S rule). SBO alone: does
`waiting_ms` fall? Fusion alone under DP: does the
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` kernel disappear from the top-10 and
is it replaced by `oneshotAllreduceFusionKernel`/`twoshotAllreduceKernel`? Then,
separately, does throughput move?

**Also fix the ledger.** Row #9's verdict text and §3's table are both wrong as
written. A ledger whose "Tried" table blocks retries must be right about what was
tried.

**Confidence.** `[verified]` throughout.

---

### Item 4. Cut the number of kernels per forward pass

**Mechanism.** This is the item §0.4 argues for and it is the largest measured
ceiling in the corpus. Measured on this box, in the steady-state C1 decode window
of `runs/sweep-latency-3-1-4/trace.sqlite` `[verified, 01-kernel-opt/05 §2]`:

- Launch *gaps* are already dead: device 0 spends **1.90%** of wall with no
  kernel resident, gap p50 = 0.22 µs, because **100% of decode kernels are
  CUDA-graph launched**. So this is not a launch-overhead problem.
- **Only 65.4% of the 148 SMs hold a resident CTA on average**, and a quarter of
  decode wall clock runs on **≤36 of 148 SMs**.
- **10.7% of decode GPU time runs on a single SM**: 152,052 launches with
  `gridDim == 1` — `rmsnorm`, `act_and_mul`, `routingIndicesBlockKernel`,
  `fused_k_indexer_norm_rope_store` — burning 530.9 ms with 147 SMs idle.
- **Perfect 148-wide packing ceiling = 1.53×**: wall 3621.5 → 2368.7 ms, which
  maps 365 → **558 tok/s** and brackets TileRT's published 494.2 on GLM-5-FP8.

The concrete first move is not a megakernel. It is the `gridDim == 1` tail: four
named kernels, 10.7% of GPU time, each occupying 1/148 of the machine.
Horizontally fusing or widening them is bounded work in known files.

**Slice and ceiling.** Attacks every slice at once, because it attacks the fixed
per-kernel cost that §0.4 shows dominates. Ceiling **1.53×** → 558 tok/s
`[inferred from a verified measurement]`. Realistic first-pass on the
`gridDim==1` tail alone: if those 530.9 ms shrink 3×, that is 7.1% of GPU time →
5.9% of wall → **+6%**.

**Expected.** +5% for the `gridDim==1` work; 1.2–1.5× if pursued to a persistent
decode structure (item 16).

**Effort: port** for the tail (widening grids, fusing adjacent elementwise ops),
**kernel/research** beyond that. Note the register wall for a full megakernel:
our B200 reports 65,536 registers per SM *and* per block, and `nvjet_sm100_*`
runs 255 registers/thread × 256 threads = 65,280 `[verified]`, so a megakernel
inherits 1 block/SM and cannot co-schedule two instruction types by occupancy.

**Risk.** Correctness of any fusion must be bit-exact against the unfused path.
Accuracy risk zero if bit-exact. Real risk is opportunity cost.

**Confirm or kill.** The metric is **SM-occupancy-weighted wall clock**, not
tok/s: re-run the time-sliced resident-CTA analysis over the decode window and
check that the "≤36 of 148 SMs" quartile shrinks. If wall clock falls but the SM
histogram does not move, the mechanism is not understood.

**Confidence.** `[verified, measured]` for the 1.90% gap, the 65.4% occupancy,
the 10.7% single-SM time, and the 1.53× arithmetic. `[inferred]` for the +6%.

---

### Item 6. Step-time bake-off against the published cell at fixed accept

**Mechanism.** Our TPOT (2.74 ms at 3-1-4, real data) and SGLang's published
1.85 ms at 5-1-6 are not comparable: theirs pins acceptance with
`SGLANG_SIMULATE_ACC_LEN=3.5` `[verified, 04-industry/08 §2]` and ours is
measured. `SGLANG_SIMULATE_ACC_LEN` exists in our fork
(`environ.py:328`, consumed at `speculative/eagle_utils.py:845`) `[verified]`.

Setting the same env var on our box, at the same 5-1-6, converts our number into
**exactly their quantity: engine step time at a fixed accept length.** Any
residual difference is engine work — kernels, scheduling, commit — with
acceptance held constant.

Their published flags are known verbatim `[verified]`:
`--tp 8 --quantization modelopt_fp4 --speculative-algorithm EAGLE
--speculative-num-steps 5 --speculative-eagle-topk 1
--speculative-num-draft-tokens 6 --chunked-prefill-size 8192
--mem-fraction-static 0.85`, at ISL 8192 / OSL 1024. Our `LATENCY_MODE` is
already byte-identical `[verified, config.py:114-133]`.

**Slice and ceiling.** The whole forward pass. §2.2 shows their 6.48 ms cycle at
5 draft steps is *below* our 6.91 ms zero-draft forward — a **~1.4× forward-pass
gap** `[inferred from two verified numbers]`. Closing it is 365 → ~510 tok/s. This
is the single largest quantified gap in the corpus and it is currently
unattributed.

**Expected.** The run itself returns 0%. What it returns is the *location* of
1.4×, which is worth more than any item below it.

**Effort: config** — one run (requires item 2 to have unblocked 5-1-6; otherwise
run the comparison at 3-1-4 with `SGLANG_SIMULATE_ACC_LEN=3.16` and compare
against their number scaled by the acceptance ratio).

**Risk.** Interpretation: `SIMULATE_ACC_LEN` may skip real verify/accept work,
so the comparison measures forward passes plus simulated acceptance, not the
whole engine. Their environment differs in commit, possibly clocks, ISL (8192 vs
our 10k), and node. Enumerate and close those one at a time — that is the point
of the exercise.

**Confirm or kill.** Report cycle time (accept × TPOT) side by side. If our
simulated cycle lands near 6.5 ms, the gap is acceptance and item 2 is the whole
story. If it stays near 9–10 ms, the gap is the forward pass and item 4 is the
whole story. **There is no third answer, which is what makes this experiment
worth doing first.**

**Confidence.** `[verified]` inputs, `[inferred]` the 1.4×.

---

### Item 8. Hybrid precision — FP8 for the BF16 non-expert GEMMs

**Mechanism.** `[verified]` from `weights/GLM-5.2-NVFP4/hf_quant_config.json` and
the shard dtypes: 156 exclude entries keep all `self_attn.*`, all
`mlp.shared_experts.*`, layers 0–2, the whole layer 78, `lm_head` and
`embed_tokens` in BF16. Every excluded module is **exactly 2.00× its size in the
FP8 checkpoint** `[verified, 05-models/00 §4.3]` — 33.4 GB of weights at BF16
where 16.7 GB of FP8 would do.

The largest single item is `fused_qkv_a_proj_with_mqa` (= `q_a_proj` ⊕
`kv_a_proj_with_mqa`, BF16 `[2624, 6144]`), which is a **`ReplicatedLinear`**
`[verified, deepseek_v2.py:1777]` — TP8 does not shard it, so all 8 ranks read
all of it, 78 times per token. That is **2,515 MB/token/rank, 28.7% of the
NVFP4 build's decode bytes** `[inferred, 05-models/01 §2]`.

Three implementation routes, ascending:
- **(a)** `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1` — exists, defaults `False`
  `[verified, environ.py:642`, consumed at `deepseek_v2.py:2235` and
  `deepseek_weight_loader.py:740,757]`. Covers `q_b_proj` only ≈ 3.7% of decode
  bytes. This is item 12.
- **(b)** Extend the same loader hook to `fused_qkv_a_proj_with_mqa`,
  `kv_b_proj`, `o_proj`, `mlp.shared_experts.*` and layers 0–2, sourcing the FP8
  tensors from the FP8 checkpoint already on disk. Architectures are byte-for-byte
  equivalent `[verified, 05-models/00 §2]`, so a splice is mechanically sound.
- **(c)** Re-quantize with ModelOpt. Not worth it given (b).

Three independent teams converged on exactly this precision boundary — NVFP4/MXFP4
routed experts, higher precision everywhere else: Kimi K3 §4.1.4, Xiaomi
MiMo+TileRT, Zhipu's Ascend port `[verified, 04-industry/05]`. And the cross-model
document is explicit that GLM-5.2 is the *one* model in its set where this is
principled, because its BF16 tail is plain softmax-attention projections and
shared experts rather than recurrent-attention state `[inferred, 05-models/05 §2]`.

**Slice and ceiling.** Dense GEMM, 37.1% of C1 GPU time / 31.0% of wall. Two
competing ceilings:
- *If bandwidth-bound*: halving the slice's bytes halves its time → 1.184× → 414
  tok/s.
- *If launch/latency-bound* (which §0.4 argues, and `01-gemm-on-sm100` measured —
  our mean dense-GEMM duration of 5.95 µs sits exactly on the measured 3.7–6.0 µs
  fixed floor for *every* M from 1 to 256 `[verified]`): the HBM term is 1.14 ms
  of 6.90 ms, so −24% bytes buys **≤1.04×**.

**This contradiction is the most consequential unresolved question in the corpus**
and item 1 settles it for one run's cost. Do not start item 8 first.

**Expected.** +3% to +15%, conditional on item 1.

**Effort: port** — loader work in `deepseek_weight_loader.py`, days. Also needs
per-GPU memory re-budgeting (FP8 non-expert weights are smaller than BF16, so
this *frees* memory).

**Risk.** Accuracy: low — FP8 block-128×128 on attention is precisely what
`zai-org/GLM-5.2-FP8` ships, and "Give Me BF16 or Give Me Death?" measured
W8A8-FP recovering ~99.3% with no calibration `[verified]`. But the acceptance
gate must be agentic, not MMLU: Mix-Quant measured NVFP4 dropping agentic scores
6–10 points on models that looked lossless on MMLU `[verified]`, so use τ²-bench
+ BFCL + AIME25 pass@1 over ≥16 samples `[reported recommendation, 03-papers/04]`.
Second risk: quantize kernels are already 2.4% / 3.0 µs each and adding dequant
to the attention path could cost more than the bytes save — which is why
weight-only (W8A16) is the right first shape, not W8A8.

**Confirm or kill.** Item 1 first. Then, if pursued: end-to-end TPOT at C1 with
≥2 repeats, plus `dram__bytes_read.sum` on the `nvjet_sm100_tst_*` kernels from
`gpu-bench ncu` before and after. **The bytes counter must fall by the predicted
ratio.** If bytes fall and time does not, the diagnosis is confirmed as
launch-bound and the item closes with a ledger row.

**Confidence.** `[verified]` the checkpoint state and the byte arithmetic;
`[inferred]` and contested for the effect.

---

### Item 9. Launch-structure fusion flags

**Mechanism.** Two flags are off in every run `[verified from the resolved
ServerArgs]`:
- `enable_fused_qk_norm_rope=False` — fuses the Q/K norm and RoPE into one
  kernel. `fused_k_indexer_norm_rope_store` is one of the four named
  `gridDim == 1` kernels burning 10.7% of decode GPU time on one SM
  `[verified, 01-kernel-opt/05]`.
- `enable_fused_moe_sum_all_reduce=False` — help text: "Enable fused moe triton
  and sum all reduce" `[verified, server_args.py:1929]`. The word *triton* is a
  warning: we run `moe_runner_backend='flashinfer_trtllm'`, so this may be a
  no-op or may force a runner change.

Independent corroboration for the shape of the win: SGLang PR #27705's indexer
prologue fusion took 12 kernels → 4 for **+8% decode at bs=1** `[reported,
04-industry/02 §9]`; vLLM collapsed DeepSeek-V3.2's per-layer pre-attention chain
from ~33 launches/layer toward ~10 for **1.28× at batch 1** on 4×GB200
`[verified, vendor-measured]`.

**Slice and ceiling.** Elementwise is 3.7% + norm 0.2% at C1 → 1.032× if deleted.
The real target is launch *count*, which is item 4's axis; these flags are the
free sample of it.

**Expected.** 0% to +4%.

**Effort: flag.** Two runs.

**Risk.** Numerics: a fused norm+RoPE changes reduction order; must pass GSM8K
within noise of 96.00%. Operational: `enable_fused_moe_sum_all_reduce` may
conflict with the trtllm MoE runner and refuse to start.

**Confirm or kill.** **Do not judge on tok/s** — the effect is inside the noise
floor. Diff the kernel-name histogram from a short nsys capture before and after:
the fused kernel must appear and the unfused pair must disappear. If the histogram
does not change, the flag did not engage and the run means nothing. This is the
same failure mode as the SBO experiment (§2.1).

**Confidence.** `[verified]` both flags are off; `[reported]` the magnitudes from
upstream PRs.

---

### Item 10. `--speculative-adaptive` — batch-adaptive draft depth

**Mechanism.** `speculative_adaptive=False` today `[verified, our run log;
server_args.py:2180]`. The in-tree ladder is
`DEFAULT_ADAPTIVE_CONFIG` at `speculative/adaptive_spec_params.py:22`
`[verified, read the source]`:

```
BS 1  -> candidate_steps [1, 3, 7]
BS 8  -> [0, 1, 3]
BS 32 -> [0, 1]
BS 64 -> [0]          # speculation off
```

That is the engine's own answer to the latency/cost conflict, and it is a flag.
The originating PR reports +6.2% average over a fixed `num_steps=3` and finds
GLM-4.7-FP8 optimal at step 7 at BS 1 and step 3 at BS 64 `[reported]`. The
literature agrees on the shape: TurboSpec measures peak goodput at k=4–5 for
BS=1 versus k=1–2 for BS=64, with speculative decoding turning net-negative at
BS≥16 on Llama2-7B/H100 `[verified]`; vLLM reports MTP *degrading* throughput
above ~256 batch on GB300 DeepSeek-V3.2 even at >80% acceptance `[verified]`.

**Slice and ceiling.** Not a slice. It moves us along the accept-vs-cost curve of
§2.2 automatically. At C1 the ladder's top rung is 7 steps — which **will hit the
same IMA as 4-1-5 and 5-1-6 unless item 2 lands first**. This is a hard
dependency, not a nicety.

**Expected.** +0% to +6% at C1 (conditional on item 2); **+5% to +15% at C10–C64**,
where our fixed 2-1-3 is probably still over-speculating: our own C16 numbers show
accept falling to 2.54/3 in capacity mode `[verified, SCORECARD §1c]`.

**Effort: flag.**

**Risk.** Exactness is preserved (thresholds are 1.0). The real risk is TTFT and
P50 stability: AA's headline is a **median over 72 hours**, and a controller that
oscillates between depths produces a wider distribution than a fixed depth. Since
"consistency beats peak" for a P50 metric, measure the *spread*, not just the
mean.

**Confirm or kill.** Sweep C1, C10, C64 with and without, on real data, and report
p50 **and** p95 of TPOT. Kill if p95 widens more than p50 improves.

**Confidence.** `[verified]` the flag and ladder exist in our tree; `[reported]`
the +6.2%.

---

### Item 12. `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1`

**Mechanism.** SGLang's only shipped escape hatch for the NVFP4 build's BF16
attention. `_get_q_b_proj_quant_config` substitutes an
`Fp8Config(weight_block_size=[128,128])` for `q_b_proj` when the checkpoint is
NVFP4 `[verified, deepseek_v2.py:2234-2243`; loader at
`deepseek_weight_loader.py:740,757]`. Default `False` `[verified, environ.py:642]`.

**Slice and ceiling.** `q_b_proj` is **3.7%** of decode bytes `[inferred,
05-models/01 §2]`. Even if bytes were the whole story, halving 3.7% of the byte
budget is ≤1.9% of the HBM term, i.e. ≤0.3% of wall. **Ceiling ≈ 1.005×.**

**Expected.** 0% to +1.5%. Below the noise floor.

**Effort: flag.** Fold it into item 1's or item 8's run rather than spending a
run on it.

**Risk.** Low. It is a supported code path.

**Confirm or kill.** Not measurable end-to-end at our noise floor. Confirm
engagement by dtype assertion at load (`q_b_proj.weight.dtype is float8_e4m3fn`)
and measure only the kernel, via `gpu-bench ncu --kernel-regex`.

**Confidence.** `[verified]`.

---

## 5. Detailed entries — cost levers

### Item 5. The un-graphed AllReduce at C64

**Mechanism.** In the C64 capture `[verified, deepdive]`:
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL` is **693.5 ms, 10.5% of GPU time, 0%
graph-captured**, while `AllGather_RING_LL` (7.1%) and
`ReduceScatter_Sum_bf16_RING_LL` (6.1%) are both **100% captured**. A collective
outside the graph pays per-launch cost every step and cannot be reordered by the
graph scheduler.

The leading hypothesis, now that §2.1 is settled: capacity mode runs DP attention
(`dp_size=8`, `enable_dp_attention=True`), which (a) suppresses the fusion that
would have absorbed the AR into the RMSNorm and (b) introduces cross-DP metadata
synchronisation that happens at step boundaries, outside the captured region.
The AllGather/ReduceScatter pair *inside* the graph is the DP-attention →
TP-MoE transition; the un-graphed AllReduce is something else.

**Slice and ceiling.** Collectives 25.4% of C64 GPU time (19.1% of wall).
This one kernel is 10.5% → 7.9% of wall → **1.086× if eliminated**. Graphing it
without eliminating it is worth the launch cost only, which at 100% graph
coverage elsewhere is small — so the value is in *what it is*, not in graphing it.

**Expected.** +3% to +10% aggregate at C64, mostly via item 7 if the answer is
"the fused path is off".

**Effort: trace analysis first (no new run)**, then port.

**Risk.** None to investigate.

**Confirm or kill.** Two queries over traces we already hold: (a) `graphId`
coverage and launch-vs-execution time by kernel (`cuda_kern_exec_sum`) to see
whether the AR is launched from the scheduler thread at step boundaries; (b) the
decisive one — **re-profile at C16 in latency-mode parallelism (TP8, no DP)**. If
the un-graphed AllReduce disappears, it is DP-attention-specific and item 7 is
the fix. If it persists, it is in the spec-decode or sampling path and needs
different work. One run.

**Confidence.** `[verified]` the measurement; `[inferred]` the hypothesis.

---

### Item 7. Turn allreduce fusion on under DP attention at C64

**Mechanism.** The auto-enable pass explicitly excludes `enable_dp_attention`
(`overrides.py:1799`) `[verified]`, but a user-supplied
`--flashinfer-allreduce-fusion-backend auto` is not overridden by any subsequent
pass except the deterministic-inference and enforce-disable paths, neither of
which is active for us `[verified, overrides.py:1810-1876;
server_args.py:4952-4953]`. So it can be forced.

**Slice and ceiling.** The 10.5% RING_LL AllReduce at C64 → 1.086× ceiling.
Realistically the fused path replaces it with a cheaper mnnvl one-shot/two-shot
kernel plus an absorbed RMSNorm, so recover part of it plus one kernel launch per
layer.

**Expected.** 0% to +8% aggregate at C64.

**Effort: flag.**

**Risk.** **This is the un-isolated half of the −12.5% in ledger §4 #9** (§2.1).
It may reproduce that loss. That is a reason to run it as one variable, not a
reason not to run it. There is presumably a correctness or performance reason the
auto-enable declines under DP attention; finding out what it is *is* the
experiment. Numerics: the fusion changes the AR+residual+norm reduction order —
gate on GSM8K within noise of 96.00%.

**Confirm or kill.** Kernel histogram must show RING_LL AllReduce replaced by
`tllm_mnnvl_allreduce::*`. Then `waiting_ms` from `gpu-bench deepdive`. Then
aggregate tok/s. In that order.

**Confidence.** `[verified]` the gating; `[inferred]` the effect.

---

### Item 11. MoE down-projection pipeline under-fill

**Mechanism.** `[verified, 01-kernel-opt/03]` At TP8 the MoE `K` per rank is
**exactly 256**, which is **exactly one `tileK`** — so the 9-stage software
pipeline in `bmm_Bfloat16_E2m1E2m1_..._t128x8x256_s9_...` never fills. Measured
consequence: the down-projection moves half the bytes of GEMM1 in 85% of the
time — **≤2.81 TB/s vs GEMM1's ≤4.80 TB/s** (36.7% vs 62.6% of the 7.672 TB/s
peak). Separately, GEMM1 runs **128 CTAs on a 148-SM GPU** — under one wave, so
its ceiling is 86.5% of peak bandwidth before any inefficiency `[verified grid]`.

Levers, in order of cost: a smaller `tileM` or a CGA split-K (`c1x1x2`) to raise
CTA count; fusing GEMM1+GEMM2 persistently; or Kimi K3's token-centric
(WarpDecode-style) MoE decode kernel — one warp per output neuron, lane teams
over disjoint expert subsets, offline weight permutation `[verified,
arXiv:2607.24653 §5.4.2]`.

**Slice and ceiling.** The two `bmm_*` expert GEMMs are 23.5% of C64 GPU time;
the down-projection half is 11.1%. Taking it from 2.81 to 4.0 TB/s is a ~30%
reduction on 11.1% → 3.3% of GPU time → **1.026× at C64**.

**Expected.** +2% to +5% aggregate. Modest. Note the corpus is unanimous that
these kernels are already on the fastest instruction NVIDIA ships:
`tcgen05.mma...kind::mxf4nvf4.block_scale.block16` with no instruction-level
upgrade available, and a tuned public NVFP4 grouped GEMM reaching ~50% of
tensor-core peak end-to-end `[verified PTX ISA + CCCL headers; reported for the
50%]`. The headroom is tiling and pipelining, not math.

**Effort: kernel** — and gated on Phase 0a. **No kernel on this box has a
measured fraction-of-roofline** `[verified, ledger §2b]`. The Kimi-K3 precedent
is the reason to gate: cuBLAS turned out to be at 82% of BF16 peak for the
attention projections there and the correct decision was to stop.

**Risk.** Weeks spent on a vendor kernel that is already near its ceiling.
Accuracy risk zero if bit-exact.

**Confirm or kill.** `gpu-bench ncu --kernel-regex bmm_Bfloat16_E2m1E2m1` for
achieved DRAM throughput, `smsp__inst_executed` and the dominant stall reason.
**Anything above ~75% of a defensible ceiling is closed, not optimized.** Note
that on SM100 a "waiting for MMA" stall never appears — `tcgen05` MMA allocates
no write barrier and completion is tracked through `UTCBAR`/mbarrier, so it shows
up as `short_scoreboard` or `barrier` `[verified, measured across 89,584 real
SM100 production instructions]`. Interpret accordingly.

**Confidence.** `[verified]` the shape and the throughput measurement;
`[inferred]` the effect.

---

### Item 13. Unblock two-batch overlap

**Mechanism.** Propagate `topk_indices` across the TBO micro-batch split. The
carrier already exists: `DeepseekV2Model.forward` threads `prev_topk_indices`
from layer to layer, and `should_run_indexer` decides per layer whether to compute
or forward it `[verified, 05-models/00 §7.1, §7.3]`. The TBO op path does not
carry it, so shared layers would run sparse attention without indices — hence the
hard `raise` `[verified, server_args.py:5053-5063]`.

**But see §2.5**: there is a *second* blocker, and in latency mode satisfying it
requires DP attention, which disables the allreduce fusion. So TBO is a cost-plane
project only.

**Slice and ceiling.** Collective *waiting* at C64: 44% of 25.4% = 11.2% of GPU
time → 8.4% of wall → **1.092× ceiling**. TBO overlaps compute with comms rather
than deleting the wait, so recover a fraction.

**Expected.** +3% to +9% aggregate at C64+. **Zero at C1 by construction** — no
batch to split — and plausibly *negative* below ~C32 (Perplexity: +18% at batch
128, degrades at batch 32) `[reported]`.

**Effort: port** — bounded, in a known file, but the correctness surface is
serious.

**Risk.** **Silent wrong output.** Shared layers running sparse attention with
stale or absent indices produce plausible text with a wrong attention pattern.
This is exactly why upstream hard-raises rather than warning. Any implementation
needs a bit-exactness test of logits against TBO-off on a fixed prompt, per layer,
before any performance measurement.

**Confirm or kill.** Correctness first (bit-exact logits), then `waiting_ms` and
`max_skew_us` from `gpu-bench deepdive` at C64 — **`waiting_ms` must fall and
NVLink utilisation must stay low**; if the link rises instead, the diagnosis was
wrong. Then aggregate tok/s against the 4% floor.

**Confidence.** `[verified]` both blockers and the mechanism; `[inferred]` the
effect.

---

### Item 14. Prefix-cache hit rate, and the EAGLE↔radix interaction

**Mechanism.** Prefix caching is our second-largest measured win: **1.54×** on
realistic traffic (coding workload 27,593 → 40,794 tok/s aggregate at C64, 54.2%
hit, TTFT −29%) `[verified]`. The literature says 54% is far from the ceiling:
TraceLab measures a **95.7% token-weighted prefix hit rate** on 4,300 real Claude
Code and Codex sessions `[verified]`, and LMCache's production traces show hit
rate collapsing from ~85% to ~45% when a sliding window truncates context
`[verified]`.

There is also a specific reported hazard for our exact stack: SGLang issue #32459
reports deep-prompt cache hit falling from **97% to 40–53% with EAGLE on GLM-DSA
NVFP4 8×B200** `[reported, 04-industry/02 §8]`. AA takes a **P50 over 72 hours**
of a repeated ~10k-token prompt family, so if speculation kills prefix reuse our
TTFT is structurally worse than it needs to be — on the one metric where we
currently lead by 3×.

**Slice and ceiling.** Prefill, not decode. **Zero effect on output tok/s**, which
is the AA headline. Real effect on TTFT and on C64 aggregate.

**Expected.** TTFT −20% to −50% on repeat traffic if the EAGLE interaction is real
and fixable; +0% to +15% aggregate at C64 from raising the hit rate.

**Effort: config** to measure, **port** to fix.

**Risk.** None to measure.

**Confirm or kill.** Send the same 10k-token prompt twice with
`--enable-cache-report`, read `cached_tokens` and the TTFT delta, with and
without `--speculative-algorithm EAGLE`. Two runs, minutes each. If cached_tokens
on the second request is near the full prompt in both arms, issue #32459 does not
reproduce here and this closes.

**Confidence.** `[verified]` our own 1.54×; `[reported]` the hazard.

---

### Item 15. The dead DSA indexer KV allocation

**Mechanism.** `[verified, memory_pool.py:4358-4378]`
`DSATokenToKVPool._create_index_buffers` allocates one `index_k_with_scale_buffer`
per layer, `for _ in range(self.layer_num)`, and `_compute_cell_size`
(`pool_configurator.py:224-246`) charges `indexer_size_per_token *
effective_num_layers` against the budget. But only 21 non-MTP layers carry indexer
weights (layers 0,1,2,6,10,…,74) `[verified two independent ways from the
checkpoint and from `dsa_layer_skips_topk()`]`, and SGLang's own comment says so:
*"shared layers' cache is never read, so filling it is dead work."*
(`forward_mla.py:189-190`). Grepping `mem_cache/` for `skip_topk` or
`indexer_layer` returns nothing.

`[inferred]` cost: allocated 132 B × 78 = 10,296 B/token; needed 132 B × 22 =
2,904; **wasted 7,392 B/token/GPU** = 12.0% of the 61,464 B/token FP8-KV cost.
Independent confirmation: TensorRT-LLM PR #16558 (merged 2026-08-05) reduced
Indexer-K accounting by **7,524 B/token** (= 57 shared layers × 132 B) for
"+15.8% effective KV capacity" `[reported]`. SGLang's equivalent, PR #30531, is
still open `[reported]`.

**Slice and ceiling.** **Not a time slice at all.** It buys KV capacity. And at
our measured operating points it buys nothing: `max_total_num_tokens = 1,744,128`
per rank and the decode log reads `token usage: 0.01` `[verified, our run log]`.
**The KV pool is 1% utilised.** This item is worth exactly zero on the AA
benchmark and zero at C64 on our current workloads.

It becomes real at 1M-context serving or at C256+ with long prompts, where it is
~13 GB/GPU at a full pool.

**Expected.** 0% today. +12% KV capacity, i.e. +12% concurrent long streams, in a
regime we do not yet run.

**Effort: port.** Small and mechanical: gate the buffer list on `skip_topk` and
fix `_compute_cell_size` to match.

**Risk.** An off-by-one in the layer set silently breaks sparse attention on a
layer that does read its cache. The layer set is verified two ways, so the risk is
manageable, but it needs a per-layer assertion.

**Confirm or kill.** Log the total bytes of `index_k_with_scale_buffer` at startup
and compare against `132 × layer_num × max_total_num_tokens`. Note that our
startup log's `KV Cache is allocated ... KV size: 89.71 GB` line accounts for
**only** the MLA latent pool (89.71 GB / 1,744,128 tokens = 51,435 B/token ≈
656 B × 78 layers) `[verified, arithmetic on our own log]` — **the indexer
allocation is currently invisible in our logs**, which is its own small bug worth
fixing first.

**Confidence.** `[verified]` the mechanism and the layer sets; `[inferred]` the
byte cost; `[verified]` that it is worth nothing at our operating points.

### Item 15b. `--mem-fraction-static` 0.85 → 0.92

Same story, same verdict, stated separately because the brief lists it
separately. Published configs for this model class use 0.92 and the NVFP4 vendor's
own launch line uses **0.80** `[verified, 05-models/00 §8]`. Raising it buys
~13 GiB/GPU of additional KV. At `token usage: 0.01` `[verified]`, that is 13 GiB
of headroom we are not using. It is a **capacity lever with zero latency effect**
and it carries a real risk: less headroom for CUDA graphs, and our C1 launch
already leaves only **26.05 GB free per GPU after the pools**
`[verified, our run log]`. `[reported]` "+1.14× at concurrency" in the ledger's
candidate H is not supported by any measurement on this box and should be
downgraded to `[unverified]`.

---

### Item 16. Persistent / megakernel decode

**Mechanism.** Fuse the decode step into a persistent kernel driven by an on-GPU
instruction queue, with counter-semaphore synchronisation in global memory
instead of collective barriers. Six independent implementations exist; two were
measured on hardware close to ours: **Event Tensor / ETC** on 8×B200 with
Qwen3-30B-A3B (128 experts, top-8) gets **1.48× over vLLM 0.11.0rc2 and 1.20×
over SGLang 0.5.3rc0 at batch size 1** `[verified]`; DeepGEMM Mega MoE gets
1.50–1.96× over a DeepEP+TileLang baseline at EP8, largest at batch 1
`[verified]`.

**Slice and ceiling.** All of them. **1.53×** → 558 tok/s, from our own measured
SM-packing analysis `[inferred from verified measurement]`.

**Expected.** 1.2–1.5×, over a quarter, with meaningful probability of 1.0×.

**Effort: research.** Constraints already known: the register wall (item 4);
TMEM's 512-column budget means a 128×256 FP32 accumulator plus block-16 scale
factors lands at 304 columns → rounds to a 512-column allocation → **one CTA per
SM** `[inferred from verified TMEM geometry]`; and any kernel touching TMEM
reports 1 CTA/SM occupancy at every block size `[verified, measured]`. Also:
`glm-kernels/` is an explicit zero-kernel scaffold by design and **no
hand-written kernel currently runs for GLM-5.2** `[verified]`, so this starts
from nothing.

**Risk.** A quarter of engineering with a real chance of 1.02×. ETC's own TP4
result was only **0.99–1.06× vs vLLM, with SGLang still winning**, because
SGLang's CPU scheduling was better `[verified]` — megakernels pay most when the
host side is already fixed, and ours is (100% graph replay, overlap scheduler on,
1.90% gap time).

**Confirm or kill.** Do items 1, 4 and 6 first. Item 6 in particular: if the
1.4× forward-pass gap turns out to be one identifiable thing, a megakernel is the
wrong instrument.

**Confidence.** `[verified]` the packing measurement; `[reported]` the external
numbers.

---

## 6. Where latency and cost conflict, explicitly

### 6.1 Speculative decoding: the latency win is a cost tax

Speculative decoding is our largest single lever — **3.09×** at C1 (144.8 →
447.8 tok/s synthetic) `[verified]`. It buys latency by spending FLOPs and expert
bytes on tokens that may be thrown away. Quantified:

- **Verification widens the expert footprint almost linearly.** The expected
  number of distinct experts activated by `T` verified tokens is
  `φ(T)·M` with `φ(T) = 1 − (1 − m/M)^T` — published and validated
  (R² 0.830 → 0.976 on Qwen3-30B-A3B) `[verified]`. For GLM-5.2 (M=256, m=8):
  1 token → 8 experts, 4 → **30.5**, 6 → **44.4**, 8 → **57.4** `[inferred, our
  arithmetic]`. Against 32 slots read serially at T=4, the union of 30.5 is a
  **4.6% saving** — i.e. speculation gives essentially **zero** MoE
  weight-traffic amortisation `[inferred; corroborated in kind by MoE-Spec, where
  a 127-token tree on OLMoE activates 54 of 64 experts per layer, verified]`.
- **The grid says the same thing.** GEMM1's launch grid is literally `(4, 32)` in
  every decode launch at 3-1-4 `[verified, grid distribution query on our trace]`.
- **Our own C64 config already retreated** from 3-1-4 to 2-1-3, and accept fell
  to 2.54/3 `[verified]`.
- **Upstream's ladder turns speculation off entirely at BS 64** `[verified,
  DEFAULT_ADAPTIVE_CONFIG]`.
- **The MTP layer is 19.91 GB in the NVFP4 build vs 10.03 GB in the FP8 build**
  `[verified]` — a +2.36 GiB/GPU tax under TP8 for running speculation at all,
  for zero quality reason.

**Rule:** never quote a spec-decode gain without the concurrency. The latency arm
should run the deepest draft that does not crash; the cost arm should run the
shallowest that still pays; item 10 automates the choice.

### 6.2 DP attention: cost win, latency loss, and it disables fusion

DP attention is right at throughput and wrong at C1. It is also *entangled*:
turning it on suppresses the allreduce-fusion auto-enable (§2.1) and is a
*prerequisite* for TBO (§2.5). Published recipe cells across four models and four
teams agree on the boundary — every low-latency cell is pure TP with no `--dp`;
every balanced/high-throughput cell adds DP attention and EP; the crossover sits
between C16 and C64 `[verified, 05-models/05 §4]`. Our two modes already
straddle it correctly. **Do not unify them.**

### 6.3 Prefix caching: the rare lever that helps both

1.54× aggregate at C64 *and* −29% TTFT `[verified]`. No conflict. It is also
the only lever whose value depends on traffic shape rather than on
configuration — which is why item 14 is a measurement of *our* traffic first.

### 6.4 Memory: capacity and latency are nearly orthogonal here

At `token usage: 0.01` `[verified]`, every KV-capacity lever (items 15, 15b) is
worth zero on the latency plane. Conversely, freeing memory does not make the
step faster. The one place they meet is CUDA-graph headroom: 26.05 GB free per
GPU today `[verified]`, and finer graph batch-size buckets (which cost ~260 MB and
1.17× startup for up to 1.3× aggregate at high concurrency `[reported]`) spend it.

### 6.5 Precision: prefill and decode want different answers

Mix-Quant recovered most of an agentic-accuracy gap by keeping **decode** in
higher precision while quantizing **prefill** to FP4, at up to 3× prefill speedup
on B200 `[verified/reported]`. Their mechanism argument is sound: prefill
quantization error does not feed back within the pass, whereas decode is
sequential and a flipped token snowballs. For us this maps onto the two arms
directly — and it interacts with acceptance, since draft/target distribution
mismatch is what accept length measures `[inferred]`.

---

## 7. Explicitly not worth it

Each row is a *closed* question with the evidence that closed it. These belong in
the ledger's §4 as negative results so nobody re-opens them.

| item | why not | conf |
|---|---|---|
| **L2 persisting window (`cudaAccessPolicyWindow`)** | Measured on this box: **+3% at best, −5% when sized to 79 MiB**. The default B200 L2 policy already holds a 48 MiB hot set against a 1 GiB stream. Effective all-SM-shared L2 is ~63 MiB, not 126.5. Per-instruction `.L2::evict_*` needs 256-bit vector types on sm_100 and ptxas 13.3 rejects narrower forms | `[verified, measured]` |
| **NCCL tuning / bandwidth knobs** | Measured twice: NVLink user-data throughput inside collectives is **0.7% (C1) / 1.6% (C64)** of capacity while collectives hold 19.6–25.4% of GPU time. The link is idle. Also measured: peer bandwidth 771 GB/s read of a 900 GB/s spec — the fabric is fine | `[verified, measured twice]` |
| **Unlocking the SM clock for a "19% haircut"** | 1597 MHz is a DVFS plateau, not a lock; a compute-saturating GEMM already power-caps to **1072–1222 MHz** at 990 W with `SW Power Cap` asserted. There is no 19% to recover. See §2.3 | `[verified, this box]` |
| **EPLB / expert-location rebalancing** | `ep_size=1` in every run `[verified from the resolved args]`, so the MoE is **tensor**-sharded: every rank computes every activated expert on a 1/8 slice of `moe_intermediate_size`. Per-rank expert work is identical **by construction**. The standard MoE-straggler explanation is excluded for this configuration, which retires ledger candidate C. Published EPLB gains (1.49× prefill / 2.54× decode) were measured at EP32–EP72 across 9 nodes | `[verified from config]` |
| **Expert / weight offloading** | PCIe Gen5 ×16 ≈ 63 GB/s; one token's routed experts are 1.593 GB/GPU → **25 ms of PCIe per token** against a 2.24 ms TPOT. Not viable at any concurrency | `[verified link gen/width]` |
| **PD disaggregation for the latency arm** | Costs 50–100 ms of TTFT (Perplexity, Meta) `[reported]`; TaiChi measures **97% SLO attainment for aggregation vs 42% for disaggregation** under tight-TTFT/relaxed-TPOT on Llama-2-70B TP4 `[verified]`. On one 8-GPU node it also halves TP width per phase | `[verified/reported]` |
| **TP16 or inter-node TP** | Independently measured: TPOT **0.86 ms at TP4 intra-node → 11.56 ms at TP8 inter-node** on 4×H100/node + IB NDR400 `[verified]`. TP8 in one NVLink domain is the optimum for latency | `[verified]` |
| **Tree drafts (`--speculative-eagle-topk > 1`)** | Costs three things we depend on: the overlap scheduler requires topk=1 and errors otherwise; adaptive speculation is topk-1 only; and GLM-5.2's `index_share_for_mtp_iteration` — which removes indexer cost from every draft step — is effective only at topk=1. Perplexity separately measured custom attention masks slowing attention **up to 50%** | `[verified]` |
| **MXFP4 for this model** | NVFP4 beats MXFP4-OCP by 5.8 points on Llama-3.1-8B 6-benchmark average and 10.2 on DeepSeek-R1 MMLU-Pro; the gap is mostly the E4M3 scale, not the block size. Both run at the same tensor-core rate on B200, so there is nothing to gain | `[verified]` |
| **INT8 W8A8 / planning a B300 migration around it** | B300 has a **30:1 FP8:INT8 dense ratio**; PTX never exposes the integer path on `sm_103a`, CUTLASS skips INT8 UMMA for 103a, vLLM hard-errors on the first forward | `[verified]` |
| **Hadamard / rotation pre-processing (QuaRot, SpinQuant)** | The one independent controlled study finds rotation gives "little improvement and may even lead to degradation" for MXFP4/NVFP4 **PTQ** — a 16-element block already does what rotation does. It matters for FP4 *training* | `[verified]` |
| **Attention and DSA-indexer kernel work as a throughput play** | 10.9% + 5.8% at C1, falling to 4.6% + 2.4% at C64. Deleting both entirely buys 1.16× at C1 and 1.06× at C64. Also: FA4 is installed here but its SM100 MLA DSA path requires `qhead_per_kvhead == 128` and we have 8 heads/rank at TP8, so it is unreachable | `[verified]` |
| **`SGLANG_SIMULATE_ACC_LEN` for any published number** | It pins acceptance rather than measuring it. Use it only for the like-for-like step-time comparison in item 6, and label every result from it | `[verified]` |
| **Planning against TileRT as a portable target** | `tile-ai/TileRT`'s converter takes `--model_type glm-5`, the shipped backends are `libtilert_dsv32.so` and `libtilert_glm5.so`, and **no GLM-5.2 path is announced**. `weights/GLM-5.2-FP8-TileRT/` is **empty** — 0 files. Their ~500 tok/s is on 1K-in/1K-out synthetic, and SemiAnalysis's config file shows the 8×B200 figure is a **disaggregated two-node deployment with vLLM doing prefill** | `[verified]` |

A note on TileRT: running it on **GLM-5.1** as a ceiling probe is still worth one
day (it is directly installable — pinned Python 3.12, torch 2.11.0+cu130, CUDA
13.2, which is what this box runs `[verified]`), because it would localise the
365 → 500 gap to kernels vs scheduling on identical silicon. But stop treating
500 tok/s as a like-for-like target.

---

## 8. What we would need to know to rank these better

Ordered by how much each would change the list. All are cheap.

1. **Where does the idle GPU time go?** Device 0 is idle **16.3% of wall at C1
   and 24.8% at C64** `[verified]`. That is larger than the entire collectives
   bucket, and shortening kernels cannot touch it. What we already know: decode is
   100% CUDA-graph replayed, the overlap scheduler is on
   (`disable_overlap_schedule=False` `[verified]`), gap p50 between kernels is
   0.22 µs and total gap time is 1.90% `[verified]` — so this is **not** launch
   gaps and **not** Python in the forward path. The remaining candidates are the
   un-graphed AllReduce (item 5), rank skew appearing as idle on peer ranks, and
   step-boundary work (sampling, detokenization, admission). **Instrument:**
   nsys `cuda_kern_exec_sum` and inter-kernel gap attribution over traces we
   already hold. **No new run.** This is the single cheapest high-value item in
   the corpus and it is still not done.

2. **Does any kernel have headroom?** No kernel on this box has a measured
   fraction-of-roofline `[verified, ledger §2b]`. Our top-10 is ranked by *cost*,
   not *opportunity*, and every kernel in it is vendor-optimized (cuBLAS,
   TRT-LLM-gen, NCCL), so the prior should be that they are good. Targets in
   order of share: `bmm_E2m1_E2m1E2m1_*` (12.4% @C64),
   `bmm_Bfloat16_E2m1E2m1_*` (11.1%), `nvjet_sm100_*splitK` (7.3%).
   **Anything above ~75% of a defensible ceiling is closed, not optimized.**
   `gpu-bench ncu` exists and has never been pointed at GLM-5.2; it serialises
   kernels, so it needs its own server load.

3. **Is the 1.4× forward-pass gap to the published cell real, and where is it?**
   Item 6. This is the largest unattributed number in the document.

4. **Per-layer decode attribution.** Structurally unavailable today: NVTX module
   ranges fire **13 times per module** across a 20-second capture (those are the
   graph *capture* passes), because Python module forwards do not execute during
   replay `[verified]`. Family-level classification is our only axis in the regime
   that dominates. Getting layer-level decode attribution needs NVTX ranges
   captured *into* the graph — an engine change, not a flag. Until then we cannot
   say which of 78 layers is slow.

5. **A node price.** The cost plane is entirely unmeasurable without it. Every
   `$/1M` comparison on the board (DeepInfra $0.25 … Scaleway $1.57) is
   inaccessible to us. Note that electricity is **1–3% of $/token** — at C64 the
   node burns ~$0.0036 of industrial electricity per million tokens against
   $0.11–$0.33 of amortised GPU cost `[inferred]` — so this is a capital-cost
   question, not a power question.

6. **Is the rank imbalance real, and where?** Rank 0 arrives last in 24% of
   114,171 collectives at C1; worst-case skew is 18.1 ms at C64 `[verified]`. The
   devices are **property-identical** (the probe diffs every attribute against
   GPU 0 and finds none) `[verified]`, and §7 shows expert-routing imbalance is
   excluded by `ep_size=1`. Power tells against a clock explanation too: rank 0
   had the **lowest** energy of all eight ranks while showing "utilized" in 70% of
   samples versus 41% for the others, at 387 W versus ~450 W — the signature of
   extra *low-power host-serialised work*, which is what TP0 does in SGLang
   `[inferred from verified measurements]`. And per-rank NUMA binding is already
   applied automatically (§2.6). **So the remaining hypothesis is narrow and
   testable: TP0's extra scheduler/IO duties.** Measurement: `py-spy` or a
   per-step timestamp on rank 0's scheduler thread vs ranks 1–7, plus
   `numactl --show` / `/proc/<pid>/numa_maps` on each scheduler PID to confirm the
   automatic binding actually applied. One afternoon.

7. **Is GPU 5 hurting us?** Five independent measurements agree it is a
   measurably worse die: highest idle power (+8.9%), highest energy over 300 s of
   identical TP8 work (+5.7%), lowest sustained GEMM throughput (−4.7%), lowest
   clock at the cap (−12.3%), most lifetime SW-power-cap time (192.9 s vs 90.9 s);
   it is one of two parts from a different serial-number lot `[verified, this
   box]`. In a TP8 collective the slowest rank sets the pace. **Measurement:**
   run the same benchmark on a 7-GPU subset excluding GPU 5 (TP7 will not divide
   the heads cleanly, so instead compare per-rank kernel durations for the same
   kernel across ranks in the existing traces). If GPU 5's expert GEMM is
   consistently 4–5% slower, that is a permanent ~4% tax on every collective and
   it is not fixable in software.

8. **Does the AA harness's chat-template path change anything?** Our
   `bench_serving` sends raw completions, so the chat template and GLM-5.2's
   default `Reasoning Effort: Max` are **not applied** `[verified]`. Note the
   template honours `reasoning_effort` for exactly one value: `'high'`; every
   other string — `"low"`, `"medium"`, `"none"`, and every typo — silently becomes
   `max` `[verified, chat_template.jinja line 2]`. For output *speed* this
   probably does not matter, but it is an assumption, and `--apply-chat-template`
   would test it in one run.

9. **Accuracy coverage is one benchmark deep.** GSM8K 96.00% against a vendor
   98.2 (FP8/H200), with `truncated_rate 3.5%` — so most of the 2.2-point gap is
   plausibly our 4096-token cap `[verified]`. GPQA Diamond is blocked on a gated
   HF dataset. Two standing caveats that no measurement has yet closed: the KV
   cache is `fp8_e4m3` with **no calibration scales** (defaulting to 1.0)
   `[verified]`, and every rank logs `DeepGemm is enabled but the scale_fmt of
   checkpoint is not ue8m0. This might cause accuracy degradation on Blackwell.`
   `[verified]`. Any quantization change (items 1, 8) needs an agentic gate, not
   MMLU — τ²-bench + BFCL + AIME25 pass@1 over ≥16 samples + MRCR at our max
   context.

10. **Where is the honest ceiling?** The corpus offers four incompatible answers
    for the same box: 500 tok/s (TileRT, different benchmark), 558 (our own SM
    packing measurement), 886–1,600 (floor assembly in 03-papers/08), 1,520
    (bandwidth roofline in 04-industry/05). They disagree by 3×. The one derived
    from a measurement on this machine is **558**, and it is the only one worth
    planning against.

---

## 9. Honest assessment of where the research is thin

- **Nothing has been measured with `ncu` on the real model.** Every kernel-level
  recommendation in this corpus, including items 4, 8 and 11, rests on cost share
  rather than on headroom. That is the single largest methodological gap.
- **The C1↔C64 comparison is a four-variable comparison** (§0.2). Several
  conclusions drawn from it — that attention amortises, that MoE grows, that
  collectives grow — are probably right for the right reasons, but they have not
  been isolated.
- **Every effect size below ~8% in this document is a projection, not a
  measurement.** Our noise floor is 4%, so a predicted 5% needs repeats or a
  mechanism counter.
- **The cost plane has no price and therefore no ranking.** Items are ordered by
  latency EV; the cost column is qualitative.
- **We have one accuracy benchmark.** A quantization or fusion change that costs
  2 points of agentic accuracy would currently pass our gate silently.
- **Two claims in the corpus are internally contradictory and I have not resolved
  them, only flagged them:** whether decode at C1 is bandwidth-bound (04-quant,
  08-blackwell, 05-models/01) or fixed-cost-bound (01-gemm, 05-megakernels, and
  §0.4 of this document); and the requantization win, quoted as both 6,048 and
  5,293 MB/rank/token in the same document. Item 1 settles the first for one run's
  cost. The second should be recomputed from the checkpoint rather than argued.

---

## Appendix: file provenance for the load-bearing claims

Read on this box, 2026-08-17:

- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/benchmark/SCORECARD.md`, `RESULTS.md`, `gpubench/config.py`
- `/home/aman/code/benchmark/runs/sweep-latency-3-1-4/server.log` — resolved
  `ServerArgs` for capture A, `Auto-enabling FlashInfer AllReduce Fusion` line,
  KV pool sizes, `token usage`
- `/home/aman/code/benchmark/runs/sweep-capacity/server.log` — resolved
  `ServerArgs` for capture B (`flashinfer_allreduce_fusion_backend=None`)
- `/home/aman/code/benchmark/runs/sweep-capacity-overlap/server.log` — the
  deprecated-alias line proving the ledger §4 #9 run turned fusion on
- `/home/aman/code/benchmark/runs/deepdive-c64.txt` — C64 families, top-10,
  counters, collective decomposition
- `/home/aman/code/NotSglang/python/sglang/srt/arg_groups/overrides.py`
  (:1663-1676 spec-MoE HIP gate, :1763-1807 fusion auto-enable, :2279-2300 shared-expert
  fusion disable)
- `/home/aman/code/NotSglang/python/sglang/srt/server_args.py`
  (:1116 dsa-cache-layer-split, :1213 numa_node, :1728 bf16_gemm_backend,
  :1929 fused_moe_sum_all_reduce, :1970-1990 fusion args, :2141-2159 spec-MoE
  backends, :2180 speculative_adaptive, :2855-2862 TBO/SBO, :5053-5063 index-topk
  TBO raise, :8601-8615 `_check_two_batch_overlap`)
- `/home/aman/code/NotSglang/python/sglang/srt/entrypoints/engine.py:1526` — PDL default
- `/home/aman/code/NotSglang/python/sglang/srt/environ.py` (:328 SIMULATE_ACC_LEN,
  :642-643 NVFP4→FP8 escape hatches, :1025 SGLANG_NUMA_BIND_V2)
- `/home/aman/code/NotSglang/python/sglang/srt/utils/numa_utils.py`
- `/home/aman/code/NotSglang/python/sglang/srt/mem_cache/memory_pool.py:4308-4378`
- `/home/aman/code/NotSglang/python/sglang/srt/model_executor/pool_configurator.py:215-246`
- `/home/aman/code/NotSglang/python/sglang/srt/speculative/adaptive_spec_params.py:22-60`
- `/home/aman/code/NotSglang/python/sglang/srt/models/deepseek_common/utils.py:105-113`
- Every `## Bottom line for our system` section in `research/00-hardware/`,
  `01-kernel-optimization/`, `03-papers/`, `04-industry/`, plus
  `05-models/00`, `01`, `05`, `06`
