# Serving systems and schedulers: the papers that define modern LLM inference

## What this is

The serving-systems literature, read rather than remembered. Every paper below was
fetched as a PDF and read — abstract, mechanism, and the evaluation section with the
numbers. Where a number appears it is tagged:

- `[verified]` — I read this number in the paper's own text or tables.
- `[reported]` — the authors claim it in an abstract or intro I read, but the
  supporting measurement is simulated, indirect, or I did not locate the table.
- `[inferred]` — my own reasoning from the paper plus our hardware. Not the
  authors' claim.

Every citation carries an exact title, first author/lab, venue+year, and an arXiv ID
or URL that was actually fetched. Hardware is always stated, because a 31× speedup on
16×A100-40GB in 2023 with OPT-175B and a PCIe-bound KV swap path has close to zero
predictive power for 8×B200 with 183 GB HBM3e and NV18 NVSwitch.

**The system this serves.** 8× B200 SXM, single node, TP8. GLM-5.2 (256 experts / 8
active, DSA sparse MLA with `index_topk_freq=4`, NVFP4 and FP8 builds), EAGLE 3-1-4
speculative decoding. 365 tok/s single-stream, 40.8k tok/s aggregate at C64. Spec
decoding is worth 3.09×; prefix caching 1.54×. The scoreboard is **P50 latency over
72 hours, measured at concurrency 1 and at 10-parallel**. TileRT does ~500 tok/s on
GLM-5 FP8 on the same box.

That target shape matters enormously for how this literature reads. Most of it
optimises **throughput at high concurrency on multi-node clusters**. We are optimising
**latency at C1–C10 on one node**. A large fraction of the canonical results are
therefore either inapplicable or actively inverted for us, and saying which is the
main analytic contribution of this document.

---

## Bottom line for our system

Ranked by expected effect on our P50-at-C1-and-C10 number.

1. **Do not turn on chunked prefill at C1, and cap it hard at C10.** Chunked prefill
   exists to protect *other users' decodes* from *your* prefill. At C1 there are no
   other decodes; you pay the cost and get none of the benefit. The costs are
   measured and large: O(N²) cumulative KV re-reads (DistServe §2.3 [verified]),
   ~25% prefill overhead at chunk 512 falling to ~0 at 2048 (Sarathi-Serve §5.4
   [verified]), a 32% prefill-time cliff from tile quantization when the chunk is
   257 instead of 256 (Sarathi-Serve §4.3 [verified]), and — the one that hurts us
   specifically — **+39% memory traffic from redundant expert-weight loads on MoE
   models** (Layered Prefill, arXiv:2510.08055 [reported]). With 256 experts and MoE
   GEMMs at 19.4% of our C1 profile, that last one is a direct tax. Expected effect:
   several percent of TTFT at C1, more on long agent prompts. If SGLang forces
   chunking on, set `--chunked-prefill-size -1` for the C1 leaderboard run and sweep
   ≥2048 for C10.

2. **Prefix caching is our largest remaining lever, and the ceiling is far above
   1.54×.** TraceLab (UW, arXiv:2606.30560) measures a **95.7% token-weighted prefix
   hit rate** on 4,300 real Claude Code and Codex sessions, with a median step of
   ~119K prefix tokens against 875 fresh append tokens [verified]. Our 1.54× implies
   we are nowhere near that. Two concrete, cheap fixes from the literature: (a) never
   truncate context to fit a window — LMCache's production traces show prefix hit
   rate collapsing **from ~85% to ~45%** when a sliding window is applied [verified];
   (b) hold KV across tool-call gaps rather than evicting on request completion
   (Continuum, arXiv:2511.02230, >8× JCT on agent workloads [reported]). Expected
   effect: if hit rate moves from wherever 1.54× implies toward 90%+, TTFT on
   multi-turn traffic drops by most of the prefill cost.

3. **Make speculation length adaptive instead of fixed at 3-1-4.** TurboSpec
   (Berkeley/UCSD, arXiv:2406.14066) measures the optimal proposal length as a
   sharply decreasing function of batch size: peak goodput at k=4–5 for BS=1 versus
   k=1–2 for BS=64, and at BS≥16 with Llama2-7B on H100 speculative decoding becomes
   net-negative, so TurboSpec disables it (speedup ≈0.97) [verified]. It also scales
   with acceptance rate: k=3 at acc=0.5, k=7 at acc=0.9 [verified]. We run one fixed
   3-1-4 tree across C1 and C64. At C1 we are almost certainly under-speculating; at
   C64 quite possibly over-speculating. A closed-loop controller keyed on measured
   acceptance rate and current batch size is the single highest-value scheduler
   change available. Caveat: TurboSpec's numbers are Llama2-7B on H100 with a 160M
   draft, not EAGLE on a 400B-class MoE on B200 — the *shape* transfers, the
   crossover point must be re-measured.

4. **Attack rank-arrival skew as a launch-jitter problem, not a communication
   problem.** 47% of our 19.6% collective time (≈9.2% of total) is ranks arriving at
   different times. The closest published measurement is Chen et al.,
   arXiv:2601.17855: **mean 40% / median 41% of per-decode-step time lost to barrier
   idle** in a real industrial trace [verified]. But that paper's imbalance is
   *cross-DP-replica* and its fix is routing. Ours is *intra-TP-group* on one node,
   where routing cannot help — the fix is eliminating per-step CPU-side variance:
   persistent kernels (ExpertPlex's adaptive persistent kernels schedule MoE at tile
   granularity inside one long-lived kernel, arXiv:2607.18002), full-graph capture,
   and vLLM's `async_scheduling` (default-on in current vLLM `SchedulerConfig`
   [verified from source]). [inferred] that most of our 9.2% is recoverable this way.

5. **Do not disaggregate prefill and decode.** On one 8-GPU node, xP+yD means halving
   TP width for each phase, which directly inflates the single-stream latency we are
   scored on. The evidence is also explicit that disaggregation is the wrong regime
   for us: TaiChi (Huawei Cloud, arXiv:2508.01989) measures, on Llama-2-70B TP4 at
   QPS=12, **97% SLO attainment for aggregation vs 42% for disaggregation under tight
   TTFT (5 s) / relaxed TPOT (250 ms)** — and the reverse under the opposite SLO
   [verified]. Latency-first single-node serving is the aggregation corner of that
   plot. Keep colocated.

6. **If you want phase isolation on one node, the mechanism is intra-GPU, not
   inter-GPU.** Nexus (arXiv:2507.06608) partitions a *single* GPU's SMs between
   prefill and decode and reports 1.4× higher throughput than vLLM-disaggregation
   with half the GPUs [reported]. ExpertPlex generalises this to MoE with tile-level
   preemption and reports **1.66× goodput over Green-Context PD colocation on
   GLM-5.1-FP8 + LooGLE** [verified] — the closest published system to ours by both
   model family and mechanism. This is a real engineering project, not a flag.

7. **Consider vAttention's memory model, because we hand-write kernels.** vAttention
   (MSR India, ASPLOS 2025) quantifies what PagedAttention costs in *kernel* terms:
   paged prefill is up to **37% slower in FlashAttention-2 and 42% slower in
   FlashInfer** than the non-paged kernel, vLLM's paged decode kernel is up to **2.8×
   slower than FA2**, and merely changing the block size moves decode-kernel latency
   by up to **1.9×** [verified]. Its alternative — CUDA VMM (`cuMemCreate` /
   `cuMemMap`) to keep the KV cache virtually contiguous while allocating physical
   pages on demand — lets you run non-paged kernels unmodified. For a team writing
   `glm-kernels` by hand on SM100, that is worth more than the 1.23× end-to-end number
   suggests. Cost: it needed a patched open-source UVM driver for 64 KB pages; 2 MB
   is the stock CUDA granularity.

8. **Layered prefill is the MoE-native answer to the chunked-prefill problem and is
   worth prototyping.** Instead of splitting a prompt along tokens, split the model
   along *layers*: partition the decoder into N_lg contiguous layer groups and run
   prefill for exactly one group per iteration. Prefill finishes in exactly N_lg
   iterations, decode never stalls, and every expert weight is loaded exactly once.
   On 8×H100 with Qwen3-235B-A22B (arXiv summarisation, 1.2 req/s) it cuts mean TTFT
   from 4.25 s to 2.26 s and p99 from 11.6 s to 7.42 s versus chunk-512, while also
   improving mean TBT 41.7→33.1 ms [verified]. That is the same architectural class
   as GLM-5.2. It is one preprint with no independent reproduction and no engine
   support — treat as a research bet, not a port.

9. **Adopt Mooncake's cache-aware admission arithmetic even though we have one
   replica.** The durable idea in Mooncake's Algorithm 1 is not routing, it is that
   the scheduler estimates `TTFT = T_queue + T_prefill(len, prefix_len)` *before*
   admitting, and returns HTTP 429 if no placement meets the SLO. On a 72-hour P50
   leaderboard, admitting work you cannot finish on time is strictly worse than
   refusing it. Mooncake also measures the eviction question directly: **LRU beats
   LFU and length-aware caching**, and hit rate saturates — 0.30 at 1,000 blocks,
   0.50 at 50,000, 0.51 at infinite capacity [verified]. Do not build a clever
   eviction policy; build a bigger cache, then stop.

10. **The queueing theory that actually applies says one thing: be
    work-conserving.** Dai, Deng, Li & Peng (arXiv:2504.07347) prove via a fluid
    limit that any work-conserving scheduler is throughput-optimal, and classify real
    systems: Orca and Sarathi-Serve are work-conserving; FasterTransformer and
    *vanilla* vLLM (no mixed batching) are not, and can go unstable under load that
    theory says is servable [verified]. Modern vLLM with chunked prefill is
    work-conserving again. The non-obvious corollary they also prove: a **max batch
    size cap turns the stability region from a scalar threshold into a 2-D convex
    hull**, and even Sarathi-Serve destabilises at token budget 1024 with k_max=100
    in their construction [verified]. Our `max-running-requests` and token budget are
    not independent knobs.

---

## 1. Batching and iteration-level scheduling — the foundation

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Orca: A Distributed Serving System for Transformer-Based Generative Models | Yu et al., SNU + FriendliAI | OSDI 2022 | ≤4× (8× A100-40GB), 1.6 Tb/s IB | 36.9× throughput vs FasterTransformer at median normalized latency 190 ms, GPT-3 175B [verified] | Yes — continuous batching is universal (vLLM, SGLang, TRT-LLM) |
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al., UC Berkeley | SOSP 2023, arXiv:2309.06180 | A100 (GCP A2) | 1.7–2.7× req rate vs Orca(Oracle), up to 22× vs FasterTransformer [verified] | Yes — PagedAttention is the default everywhere |
| vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention | Prabhu et al., MSR India | ASPLOS 2025, arXiv:2405.04437 | 1–2× A100-80GB, 1–2× H100 | 1.23× throughput over paged FA2/FlashInfer; 1.26–1.5× with FA3 [verified] | No — research prototype; needs patched UVM driver |
| Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents | Dai, Deng, Li, Peng; Cornell/Columbia | arXiv:2504.07347 | analysis + real profiling | Work-conserving ⇒ throughput-optimal; vanilla vLLM and FasterTransformer are not [verified] | N/A — theory, but validates current practice |

### 1.1 Orca: what iteration-level scheduling and selective batching actually do

Two separable ideas, often conflated.

**Iteration-level scheduling.** The scheduler and the execution engine exchange
control every *iteration* rather than every *request*. Orca's Algorithm 1 keeps a
`request_pool`, selects at most `max_bs` requests by arrival time, runs exactly one
model iteration, then re-selects. Finished requests return immediately; new arrivals
join at the next iteration boundary. `n_workers` iterations may be in flight at once
to keep a pipeline full.

The scheduler must also reserve KV slots. Orca reserves `req.max_tokens` worth of
`n_slots` at first admission and releases on completion. **This reservation is the
thing PagedAttention later kills** — it is why vLLM's "Orca (Max)" baseline reserves
2048 tokens per request regardless of actual output length.

**Selective batching.** The subtler contribution. In a batch containing requests at
different sequence positions, `[B, L, H]` tensors cannot be formed because L differs.
Orca's observation is that only *Attention* needs the request dimension:

> "Operations such as non-Attention matrix multiplication and layer normalization can
> be made to work with irregularly shaped tensors by flattening the tensors ... This
> tensor can be fed into all non-Attention operations including Linear, LayerNorm,
> Add, and GeLU operations because they do not need to distinguish tensor elements of
> different requests."

So Orca flattens everything to `[ΣL, H]`, inserts an explicit **Split** before
attention, runs attention per-request against a per-request K/V manager, and **Merge**s
back to `[ΣL, H]`. Their justification for not batching attention is that attention
carries no model parameters, so batching it buys no weight-read amortisation.

That justification is now wrong in the direction that matters for us. FlashAttention-
class kernels batch variable-length sequences natively via cumulative-sequence-length
descriptors (`cu_seqlens`), so "Split/Merge" as a literal mechanism is obsolete — but
the *flattened token-major layout* it introduced is exactly what every modern engine
uses, and is the precondition for mixed prefill+decode batches.

**Reading the 36.9× honestly.** It is GPT-3 175B, comparing against
FasterTransformer, which used request-level batching *and* max-length KV reservation.
Roughly: `0.185 req/s → 6.81 req/s` at matched 190 ms median normalized latency
[verified]. Nearly all of that gap is "FasterTransformer had no continuous batching,"
not a property of Orca's specific design. It is a 2022 number against a 2021 baseline
and should never be quoted as a live speedup.

### 1.2 PagedAttention: the numbers that still matter

The mechanism is well known — fixed-size blocks (default **16 tokens**), a per-
sequence block table, copy-on-write for forked sequences. Three measurements from the
paper are still load-bearing:

**Memory waste.** Figure 2 measures that in Orca-class systems only **20.4%–38.2%** of
KV cache memory holds actual token state; the rest is reservation, internal
fragmentation, and external fragmentation [verified]. That gap is the whole 2–4×.

**Preemption policy.** vLLM is FCFS with all-or-nothing eviction of a whole sequence's
blocks, and can either swap to host or recompute. Their §7 finding:

> "recomputation is more efficient when the block size is small, while swapping is
> more efficient when the block size is large, though recomputation overhead is never
> higher than 20% of swapping's latency. For medium block sizes from 16 to 64, the two
> methods exhibit comparable end-to-end performance." [verified]

**Recompute is never worse than 1.2× swap, and is often better.** On B200 with 183 GB
of HBM3e, whose PCIe path is proportionally far slower relative to compute than A100's
was, [inferred] recompute should be the only preemption mode we ever configure.

**Batch-size amplification.** For OPT-13B on ShareGPT at 2 req/s, average concurrent
batch: vLLM 30.42, Orca(Oracle) 13.62, Orca(Pow2) 9.81, Orca(Max) 7.00 [verified].
This is the clean statement of what paging buys: 2.2× more concurrency than a system
with *oracle knowledge of output lengths*, purely from not fragmenting.

### 1.3 vAttention: what PagedAttention costs at the kernel level

This is the paper to read if you write your own attention kernels. The argument is
that PagedAttention implements demand paging *in user space*, forcing every attention
kernel to become block-table-aware — and the resulting kernels are slower and lag the
state of the art.

Measured penalties, Llama-3-8B on one A100 [verified]:

| Kernel | Penalty vs non-paged |
|---|---|
| FlashAttention-2 prefill, paged | up to 37% slower |
| FlashInfer prefill, paged | up to 42% slower |
| FlashAttention-2 decode, paged | up to 12% slower |
| vLLM's own paged decode kernel | up to 2.8× slower than FA2 |
| Decode kernel, block 16 → 128 | up to 1.9× latency swing |

Plus the ecosystem cost they document: FlashAttention-3 and cuDNN-9 SDPA shipped
*without* PagedAttention support; TensorRT-LLM lost >10% throughput in its Python
front-end from paging bookkeeping [verified from their Table 1 citations].

**The mechanism.** Reserve a large *virtual* buffer per layer with
`cuMemAddressReserve`, and map physical handles into it on demand with `cuMemCreate` +
`cuMemMap` + `cuMemSetAccess`. The KV cache stays virtually contiguous, so an
unmodified FA3 kernel works. Three optimisations make it viable: overlap allocation
with compute (invoke the mapping API for iteration *i+1* during iteration *i*),
allocate ahead opportunistically, and defer reclamation (hand a finished request's
already-backed tensor straight to the next arrival). Their enabling observation is
that KV allocation bandwidth demand is tiny — per-token footprint is 64–240 KB across
Yi-6B/Llama-3-8B/Yi-34B, and peak allocation rate never exceeded **750 MB/s**
[verified].

**Why this is more interesting for us than for a vLLM user.** Our attention is DSA
sparse MLA with a top-k indexer, hand-written for SM100. Every time we add paging
awareness to that kernel we pay indirection in the hottest loop of a 10.9%-of-profile
component. A virtually-contiguous KV cache removes that constraint permanently. The
open question is whether 2 MB CUDA page granularity is tolerable for us: with 256
experts and MLA's compressed KV, per-token-per-layer footprint is small, so 2 MB pages
may fragment badly at low concurrency — which is exactly why the authors patched the
driver for 64 KB. [inferred] this needs a direct measurement on our KV geometry before
committing.

---

## 2. The central debate: chunked prefill vs prefill/decode disaggregation

Both techniques solve one problem — **a long prefill iteration stalls everyone's
decode** — and they solve it in opposite directions. Chunked prefill makes the prefill
small enough to hide inside a decode batch. Disaggregation moves the prefill to
different hardware entirely.

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | Agrawal et al., MSR India + Georgia Tech | OSDI 2024, arXiv:2403.02310 | 1–8× A100-80GB, 8× A40 | 2.6× (Mistral-7B), 3.7× (Yi-34B), 5.6× (Falcon-180B + PP) capacity under P99 TBT SLO [verified] | **Yes** — default in vLLM (`enable_chunked_prefill=True`) and available in SGLang (`--chunked-prefill-size`) |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving | Zhong et al., PKU + UCSD + StepFun | OSDI 2024, arXiv:2401.09670 | 32× A100-80GB, 4 nodes, 25 Gbps cross-node | 7.4× more requests or 12.6× tighter SLO [verified] | Yes — the idea; SGLang `--disaggregation-mode`, vLLM KV connectors, NVIDIA Dynamo |
| Splitwise: Efficient generative LLM inference using phase splitting | Patel et al., Microsoft + UW | ISCA 2024, arXiv:2311.18677 | A100/H100 clusters, Azure prod traces | 1.4× throughput at 20% lower cost, or 2.35× at same cost+power [reported, simulation] | Yes — the layer-wise KV streaming pattern is standard |
| Mooncake: Trading More Storage for Less Computation | Qin et al., Moonshot AI + Tsinghua MadSys | FAST 2025, arXiv:2407.00079 | Kimi production; replayed traces on LLaMA2-70B-shaped dummy | 525% throughput in simulation; 75% more requests in production [reported] | **Yes** — runs Kimi; Mooncake Store is an SGLang HiCache backend |
| Prefill-Decode Aggregation or Disaggregation? Unifying Both (TaiChi) | Wang, Zuo et al., CUHK + Huawei Cloud | arXiv:2508.01989 | 8× A100-80GB NVLink (Qwen2.5-14B/32B) + Vidur sim (Llama-2-70B TP4) | +9–47% goodput over PD-agg, +29–77% over PD-disagg [verified] | No |
| How Far Can Disaggregation Go? (Attention–FFN disaggregation DSE) | Wu, Bambhaniya et al., Georgia Tech + Intel + Google | arXiv:2605.28302 | **128× B200** + TensorRT-LLM, model-based DSE | AFD is the only feasible config for DeepSeek-V3.2 under TTFT<50–150 ms / TPOT 15 ms, ~4k tok/s [reported] | Partially — StepFun-style AFD is deployed; not in open engines |
| ExpertPlex: High-Goodput Disaggregated Serving for MoE LLMs with Adaptive Persistent Kernels | Wu, Jin, Zhang et al., PKU | arXiv:2607.18002 | SM90 (DeepGEMM/DeepEP), **GLM-5.1-FP8** and MiniMax-M2.7 | 2.01× goodput vs instance-level PD disagg, 1.66× vs PD colocation [verified] | No |
| From Tokens to Layers: Layered Prefill for MoE Serving | Lee et al., SNU | arXiv:2510.08055 | 2× A100, 2×/8× H100; Qwen3-30B-A3B, Qwen3-235B-A22B, gpt-oss-120b | TTFT −70%, e2e latency −41%, energy −22% vs chunked prefill [reported]; mean TTFT 4.25→2.26 s on 8×H100 Qwen3-235B [verified] | No |
| Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode | Shi, Cai, Du et al. | arXiv:2507.06608 | multi-model | 2.2× throughput / 20× lower TTFT vs vLLM; 1.4× over vLLM-disagg with half the GPUs [reported] | No |

### 2.1 Sarathi-Serve's mechanism, precisely

Two techniques that the paper is explicit only work together.

**Chunked prefill.** Split a prompt into chunks of at most `token_budget` tokens.

**Stall-free batching.** The scheduling order is what makes it work (their Algorithm 3):
1. First admit **all running decodes** into the batch.
2. Then admit any **partially completed prefills**, filling the remaining budget.
3. Only then admit **new requests**, computing the maximum chunk that fits in the
   leftover budget.

This is *decode-prioritising*, in contrast to Orca and early vLLM which are
prefill-prioritising. That inversion is the actual contribution; chunking is the
enabling mechanism.

**How bad is the alternative.** Orca-style hybrid batching (a full prefill coalesced
with running decodes) inflates TBT by **up to 28.3×** versus a decode-only batch
[verified]. That is the generation stall Figure 1 shows lasting several seconds on
Yi-34B.

**What chunking costs.** Three named costs, all measured:

1. *Quadratic KV re-reads.* "if a prefill sequence is split into N chunks, then the
   first chunk's KV-cache is loaded N−1 times, the second chunk's N−2 times, and so
   on." Sarathi's position is that prefill attention is compute-bound even at small
   chunk sizes so this is tolerable; DistServe's position (§2.3) is that it is O(N²)
   versus O(N) and grows with context length. **Both are correct in their own
   regime** — Sarathi tested ~1–4K prompts; agentic traffic at 119K prefix tokens
   (TraceLab) is squarely in DistServe's regime.
2. *Tile quantization.* "using chunk size of 257 can increase prefill time by 32%
   compared to that with chunk size 256" [verified]. Chunk sizes must be multiples of
   the GEMM tile.
3. *Fixed per-iteration overhead.* At chunk 512 the measured overhead is "at most
   ~25%"; at 2048 it is "almost negligible" [verified].

**Choosing the budget.** Sarathi's prescription is one-time profiling: sweep batches of
varying token counts, take the largest that fits the TBT SLO. Then adjust for tile
alignment, and for pipeline bubbles if using PP.

**Independent confirmation on our exact model.** The GLM-5 OpenClaw tuning report
(Hua et al., arXiv:2607.02518) sweeps this on a **two-node 16×H100** GLM-5 FP8
deployment under SGLang with ~28–30K-token agent prompts. Best measured:
`--chunked-prefill-size 3072`, tp=4, pp=4, max-running-requests=24, giving 9,993 tok/s
and 6.69 s mean TTFT vs 9,030 tok/s and 8.98 s at chunk 2048 [verified]. Note it is
*not* 4096 — the 4096 profile was worse. The report is honest that it has no repeats,
no seeds, and no confidence intervals, so treat as a single-sample sweep. It is
nonetheless the only public GLM-5-family chunk sweep I could find, and it says the
optimum is a specific interior point, not "as large as possible."

### 2.2 DistServe: goodput and the placement problem

DistServe's contribution is conceptual first. It defines **goodput** as the maximum
per-GPU request rate sustainable while meeting a stated SLO-attainment target (they
use 90%) on *both* TTFT and TPOT. Throughput is not the objective; SLO-constrained
throughput is.

The motivating measurement (13B, 512-in/64-out, one A100-80GB) is the clearest single
number in the disaggregation literature [verified]:

| Configuration | P90-constrained goodput |
|---|---|
| Colocated (existing systems) | 1.6 rps |
| Prefill-only instance | 5.6 rps |
| Decode-only instance | 10 rps |
| 2 prefill + 1 decode GPU | 10 rps total = **3.3 rps/GPU (2.1×)** |

Beyond the split, DistServe co-optimises the *parallelism plan per phase* — prefill
and decode need not share TP degree — and then places instances according to
**cross-node bandwidth**. Their testbed had only 25 Gbps between nodes, so they used
"low node-affinity" placement in most experiments. This is a critical caveat when
reading 7.4×: with NVLink-class bandwidth the KV transfer cost is far lower, but so is
the interference cost that disaggregation avoids.

DistServe explicitly benchmarks *against* chunked prefill (DeepSpeed-MII) and reports
its failure mode precisely: "chunked prefill is slower than full prefill, so it
struggles to meet the TTFT SLO as a sacrifice for better TPOT" [verified].

### 2.3 Splitwise: what the KV transfer actually costs

Splitwise is the engineering complement. Its durable contributions:

**Layer-wise KV streaming.** As soon as layer ℓ's KV is computed on the prompt
machine, issue an asynchronous transfer for it while computing layer ℓ+1. Implemented
with MSCCL++ **one-sided zero-copy `put`** over InfiniBand — the token machine issues
no receives; the prompt machine signals a semaphore over the same connection when all
layers are sent. Per-request semaphores, since a batch may fan out to different token
machines. They also note the counter-intuitive detail that for *small* prompts they
deliberately use the *serialized* (non-layer-wise) transfer, because fine-grained
per-layer synchronisation would add more TTFT than it hides [verified].

**Production characterisation.** Azure inference traces (coding + conversation,
11 Nov 2023), released at `github.com/Azure/AzurePublicDataset`. These are the traces
much of the later literature replays.

**Caveat on the headline.** 1.4× / 2.35× come from a simulator driven by a piecewise-
linear performance model fitted on A100/H100 profiles, validated end-to-end over 50K
iterations. It is a *cluster design study*, not a measured serving system. Read it for
the mechanism and the trace, not the multiplier.

### 2.4 Mooncake: the strongest production argument for keeping disaggregation

Mooncake is the reference KVCache-centric disaggregated architecture, and it runs
Kimi. What makes it worth reading is that it directly addresses "isn't chunked prefill
enough?" and answers no, for reasons specific to long context:

> "it is worth discussing whether this separation is still necessary with the
> introduction of chunked prefill ... after careful consideration, we decided to
> maintain Mooncake's disaggregated architecture."

Their reasoning is that inlining chunked prefill into a decode batch caps the decode
batch's MFU gain while a long-context prefill still needs cross-node scaling, which
they provide with **chunked pipeline parallelism (CPP)**: group X prefill nodes into a
pipeline, split the request's tokens into chunks no longer than `prefill_chunk`, and
run different chunks of the *same request* on different nodes simultaneously. Versus
sequence parallelism, CPP communicates only at pipeline-stage boundaries (overlappable)
and degrades gracefully for short contexts.

**Conductor's admission algorithm** (their Algorithm 1) is the piece worth stealing
even at one replica. For each candidate prefill instance it computes the prefix match
length, estimates `T_queue` and `T_prefill(len, prefix_len)`, and — if this instance's
prefix match is much worse than the global best — additionally estimates
`T_transfer(best_instance → this_instance, transfer_len)`. It picks the minimum
predicted TTFT. Then:

```
if TTFT > TTFT_SLO or TBT > TBT_SLO:  reject R; return   # HTTP 429
```

Admission control is *inside* the routing loop, using the same cost model.

**Early rejection and its pathology.** Rejecting only when the decode instance is
full wastes the prefill already spent, so Mooncake moves the decode-load check *before*
prefill. That introduces a new problem they document honestly: naive early rejection
causes **load oscillation**, because you are gating on decode load measured one
prefill-duration in the past. Their fix is prediction-based early rejection using
predicted future decode load.

**Cache hit rate vs capacity — the most useful table in the paper** [verified]:

| Block capacity | ∞ | 100,000 | 50,000 | 30,000 | 10,000 | 1,000 |
|---|---|---|---|---|---|---|
| LRUCache | 0.51 | 0.51 | 0.50 | 0.48 | 0.40 | 0.30 |
| LFUCache | 0.51 | 0.51 | 0.49 | 0.43 | 0.35 | 0.30 |
| LengthAwareCache | 0.51 | 0.50 | 0.48 | 0.42 | 0.35 | 0.30 |

Three conclusions: **LRU wins**, hit rate saturates around 50K blocks for this trace,
and there is an absolute ceiling (0.51) set by the workload, not the policy. They also
measure that **over 50% of cache blocks are never reused** while some are accessed tens
of thousands of times — hence hot-block replication.

### 2.5 TaiChi: the paper that settles the debate empirically

This is the single most useful paper for deciding what to do, because it measures both
sides under identical conditions. Setup: Llama-2-70B TP4 on a 4-node 8-GPU A100-DGX
cluster via the Vidur simulator (<3% kernel-latency error), plus real measurements on
8× A100-80GB NVLink with Qwen2.5-14B/32B [verified].

At QPS=12 on arXiv-summarisation:

| SLO regime | PD aggregation | PD disaggregation |
|---|---|---|
| Relaxed TTFT 16 s, **tight TPOT 60 ms** | 7% | **98%** |
| **Tight TTFT 5 s**, relaxed TPOT 250 ms | **97%** | 42% |
| Balanced TTFT 6 s, TPOT 100 ms | 16% | 50% |

[verified]. The clean statement: **each architecture is excellent in exactly one
corner and neither handles the balanced case.** Aggregation lives where TTFT is tight
(no queue behind a decode-only instance, full compute width for prefill);
disaggregation lives where TPOT is tight (no prefill can ever interfere).

TaiChi's own answer is a hybrid: run *both* prefill-heavy instances (large chunk,
fast prefill, interfering decode) and decode-heavy instances (small chunk, slow
prefill, clean decode) in one pool, exposing three sliders — the P-heavy:D-heavy ratio
and the chunk size for each. Then it does **latency shifting**: route short prefills
to D-heavy instances *on purpose* when their projected TTFT still fits the SLO,
freeing P-heavy instances for long, urgent prefills. Result: +9–47% goodput over
aggregation, +29–77% over disaggregation [verified].

**For us:** our leaderboard is single-stream latency, which is the extreme tight-TTFT
corner of that table. That corner belongs unambiguously to aggregation.

### 2.6 Attention–FFN disaggregation: the frontier, and it is B200-native

The Georgia Tech / Intel / Google DSE (arXiv:2605.28302) is the only paper in this
survey whose hardware is our hardware: **128× B200 SXM with TensorRT-LLM**. It
evaluates the third level of disaggregation — put attention and MoE-FFN on separate
GPU groups — across DeepSeek-V3.2 (MLA + DSA sparse attention, i.e. structurally the
same class as GLM-5.2), Qwen3-235B, GPT-OSS-120B, and Nemotron3-120B.

The measurement that matters for anyone serving a DSA/MLA model — attention vs FFN
share of runtime, by context length [verified]:

| Model | 4K | 64K | 128K | 0.5M |
|---|---|---|---|---|
| GPT-OSS-120B (full+window GQA) | 40/60 | 41/59 | 42/58 | 62/38 |
| Qwen3-235B (GQA) | 51/49 | 59/41 | 66/34 | 87/13 |
| **DeepSeek-V3.2 (MLA + sparse)** | **22/78** | **51/49** | **73/27** | **~96/4** |
| Nemotron3-120B (Mamba-2 + GQA) | 15/85 | 15/85 | 19/81 | 15/85 |

DeepSeek-V3.2 is *FFN-dominated at short context and attention-dominated at long*,
crossing over around 64K. This is a direct read on our own profile: our 10.9%
attention / 19.4% MoE split at C1 is consistent with short-to-medium contexts, and
will invert on long agentic prompts.

Their two takeaways, quoted:

> "Key Takeaway 1 — System Throughput: Aggregated wins by data-parallel concurrency
> across fully-replicated workers; disagg/AFD wins when independent sizing of
> prefill/decode or attention/FFN better rate-matches their compute than a uniform
> replica."
>
> "Key Takeaway 2 — User Interactivity: AFD wins user interactivity universally by
> sizing the attention-to-FFN ratio per workload and model."

The optimal DeepSeek-V3.2 layouts are startling: **2 attention GPUs + 126 FFN GPUs**
for agentic coding, 16A+112F for chat — because MLA plus DSA shrinks both attention
compute and KV footprint so far that almost the entire cluster should be FFN.
Caveat: these are model-based DSE estimates combining TRT-LLM cost measurements with
AstraSim communication modelling, verified functionally (not for performance) against
a vLLM AFD prototype. And 128 GPUs is not 8. **[inferred]** AFD's ratio logic does not
port to a single node — with 8 GPUs, a 2A+6F split leaves attention on 2 GPUs, and our
DSA indexer at 5.8% plus attention at 10.9% would not saturate even that.

### 2.7 ExpertPlex: closest published system to ours

PKU's ExpertPlex (arXiv:2607.18002) serves **GLM-5.1-FP8** and MiniMax-M2.7. Its
premise: MoE weights are 96% of GLM-5.1-FP8's footprint, so instance-level PD
disaggregation duplicates them across two full replicas. ExpertPlex instead **shares
the experts across phases and disaggregates only the lightweight attention modules**,
eliminating >95% of duplicate weights.

Three mechanisms:

1. **Adaptive persistent kernels (APK).** One long-lived persistent kernel per MoE GPU
   fuses two grouped GEMMs, activations, MoE pre/post-processing, and a CTA-cluster
   scheduler. Scheduling happens at **tile** granularity (their SM90 DeepGEMM config
   uses 128×192 output tiles with a 128-wide reduction), so prefill and decode work
   can preempt each other *inside* a kernel. This is the answer to why CUDA stream
   priorities, MPS, and Green Contexts all fail: stream priorities order launches but
   give no isolation after launch (decode waits behind a running prefill GEMM); MIG's
   H100 profiles only permit a 3g–4g split with no time multiplexing; Green Contexts
   fix the partition for the whole kernel and cannot track per-layer expert-load
   variation.
2. **Attention-initiated MoE communication** — one-sided, to avoid prefill/decode
   traffic interfering on a shared network and to overlap comms with compute.
3. A tile-to-cluster cost model that optimises (1) and (2) jointly.

Results on MiniMax-M2.7 + ShareGPT: 11.3 req/s/node under a joint SLO — **5.65× over
SGLang chunked prefill, 2.72× over SGLang colocated, 2.01× over SGLang PD-disagg,
1.41× over SGLang PD-multiplexing** [verified]. On GLM-5.1-FP8 + LooGLE the margin
over PD-multiplexing widens to 1.66×.

Note the ordering: **chunked prefill is the *worst* baseline on MoE**, for exactly the
reason layered prefill identifies — "Each chunk rereads model weights and the KV cache
and repeats MoE communication."

### 2.8 Layered prefill: the MoE-native replacement for chunking

The mechanism is one idea and it is elegant. Partition the decoder stack into `N_lg`
contiguous **layer groups**. Each iteration, run prefill for *exactly one* group while
every other group runs decode-only. Prefill completes in exactly `N_lg` iterations,
decode never stalls, and — the point — during prefill **each layer is traversed exactly
once**, so no expert weight is ever reloaded for the same prompt.

`N_lg` plays the role of chunk size: larger `N_lg` → higher TTFT, better TBT.

Measured against chunked prefill [verified, their Table 9]:

| Hardware / model / workload | Method | TTFT mean / p99 (s) | TBT mean / p99 (ms) |
|---|---|---|---|
| 8×H100, Qwen3-235B-A22B, arXiv @1.2 req/s | Chunked (512) | 4.25 / 11.60 | 41.7 / 50.5 |
| | **Layered (N_lg=16)** | **2.26 / 7.42** | **33.1 / 45.0** |
| 8×H100, gpt-oss-120b fp4, arXiv @3.0 req/s | Chunked (512) | 1.050 / 3.450 | 14.4 / 18.6 |
| | **Layered (N_lg=12)** | **0.530 / 1.750** | **11.8 / 19.3** |
| 2×A100, Qwen3-30B-A3B, arXiv @0.5 req/s | Chunked (512) | 1.320 / 5.400 | 22.4 / 70.4 |
| | **Layered (N_lg=16)** | **0.962 / 3.810** | **19.3 / 53.6** |

And against disaggregation on 2×H100 Qwen [verified]:

| Method | TTFT mean / p99 (s) | TBT mean / p99 (ms) |
|---|---|---|
| Chunked prefill | 3.00 / 9.15 | 32.1 / 49.3 |
| Disaggregated (1 GPU each) | 3.94 / 11.20 | **16.4 / 25.8** |
| **Layered prefill** | **1.24 / 4.59** | 19.8 / 35.9 |

Layered prefill dominates chunked on both axes and beats disaggregation badly on TTFT
while giving up some TBT.

**The honest caveats.** The advantage *vanishes* at large chunk sizes: "On Qwen
(H100×2, arXiv, 2.5 req/s), chunked prefill with chunk size 2048 and layered prefill
with N_lg=4 achieve nearly identical TBT and TTFT" [verified]. So this is a technique
for the small-chunk / tight-TBT regime, which is not our leaderboard regime. And it is
a single preprint, no independent reproduction, no engine support.

---

## 3. Prefix and KV cache reuse

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention) | Zheng et al., Berkeley/Stanford | NeurIPS 2024, arXiv:2312.07104 | A10G 24GB, A100 80GB | up to 6.4× throughput; cache-aware scheduling hits 96% of optimal hit rate [verified] | **Yes** — SGLang core; vLLM `--enable-prefix-caching` |
| CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion | Yao et al., UChicago/CUHK-SZ | EuroSys 2025, arXiv:2405.16444 | — | non-prefix KV reuse via selective recomputation | Yes — shipped inside LMCache |
| LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference | Liu, Yao, Cheng et al., Tensormesh + UChicago | tech report 2025, lmcache.ai | 8× H100 (GMI Cloud) | 1.9–8.1× lower TTFT, 2.3–14× higher throughput vs basic vLLM [verified] | **Yes** — vLLM `--kv-offloading-backend lmcache`; SGLang `--enable-lmcache` |
| TraceLab: Characterizing Coding Agent Workloads for LLM Serving | Zhu et al., UW SyFI | arXiv:2606.30560 | trace study (4,300 sessions) | 95.7% token-weighted prefix hit rate; median 119K prefix / 875 append tokens [verified] | Dataset released |
| Not All Tokens Are Worth Caching (SAECache) | Fang et al., PKU | arXiv:2605.18825 | — | 756× reuse-rate variation by token type; +4.8–5.9 pp hit ratio over best baseline [verified] | No |
| Continuum: Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live | Li, He, Mang et al. | arXiv:2511.02230 | Llama-3.1 8B/70B, Gemma-3 12B, GLM-4.5 355B | >8× average JCT on SWE-Bench / BFCL / OpenHands [reported] | No |
| Prefill-as-a-Service: KVCache Could Go Cross-Datacenter | Qin, He, Wang et al., Moonshot + Tsinghua | arXiv:2604.15039 | internal 1T hybrid-attention model | +54% throughput, −64% P90 TTFT vs homogeneous PD [reported] | Moonshot-internal |

### 3.1 RadixAttention: mechanism and the one theorem

**Data structure.** A radix tree mapping token sequences → KV tensors. Edges carry
*sequences* of tokens, not single tokens, so a long shared system prompt is one edge.
Pages are **one token** each (unlike vLLM's 16), so any prefix boundary is
representable. Nodes are reference-counted; eviction is **LRU over leaves**, so a node
is only evictable once none of its descendants are in use.

**Cache-aware scheduling.** The hit rate depends on *the order requests are run in*.
SGLang defines hit rate = (cached prompt tokens) / (prompt tokens) and sorts the
waiting queue by **longest shared prefix first (LPM)**. The theoretical justification:

> **Theorem 3.1.** For a batch of requests, we can achieve an optimal cache hit rate
> by visiting the radix tree of the requests in the depth-first search order, with a
> cache size ≥ the maximum request length. The longest-shared-prefix-first order is
> equivalent to a depth-first search order.

Empirically, on their benchmark suite, cache-aware scheduling reaches **96% of the
optimal hit rate on average**, with hit rates from 50% to 99% depending on workload
[verified].

**The caveat they state and do not solve:** "While greedy cache-aware scheduling can
achieve high throughput, it can lead to starvation." LPM is not fair. This is exactly
the tension that Llumnix, VTC and the fairness literature attack.

**Engine reality today** (verified from `sglang/srt/server_args.py`, current main):
`--schedule-policy` defaults to **`fcfs`**, not `lpm`. Options are
`lpm | random | fcfs | dfs-weight | lof | priority | routing-key`. `--disable-radix-cache`
defaults to `False` (so the radix cache is on). If you want the paper's behaviour you
must ask for `--schedule-policy lpm` explicitly.

### 3.2 Hit-rate models: what the workload actually gives you

Three independent measurements, and they disagree in an informative way.

- **Mooncake (Kimi production, general MaaS traffic):** ceiling of **0.51**, reached
  at ~50K blocks, LRU-optimal.
- **TraceLab (Claude Code + Codex coding agents):** **95.7%** token-weighted hit rate;
  84.4% on user-initiated steps, 97.5% on tool-result steps [verified]. Median step is
  ~119K prefix tokens against 875 fresh tokens. Cache misses cause **3.8× more tokens
  to be prefilled than the truly-unique input would require**.
- **LMCache (enterprise customers):** ~85% baseline, dropping to **~45%** when
  customers apply context truncation to fit a window [verified].

The reconciliation: **hit rate is a property of the traffic, and agentic traffic is a
different distribution from chat.** Prefix caching on chat gives you ~0.5; on coding
agents it gives you ~0.95. Our 1.54× is closer to the chat regime, which either means
our benchmark traffic is chat-shaped, or we are evicting too aggressively.

TraceLab's diagnosis of *where* misses come from is directly actionable: misses cluster
at **human-paced gaps**. "When the gap is larger than 5 minutes, low-hit-rate steps
begin to appear, and after 1 hour, almost all steps miss the cache." Tool-result steps
resume fast and nearly always hit. So the eviction policy should be gap-aware, which
is precisely Continuum's KV-cache TTL: pin a request's KV in GPU memory with a TTL
derived from the reload cost plus the queueing delay eviction would cause, then release
automatically.

### 3.3 Eviction: LRU is hard to beat, but token type is a real signal

Mooncake says LRU beats LFU and length-aware. SAECache (PKU, arXiv:2605.18825) finds
the axis LRU misses — **token type**:

> "the 756× variation between system prompts at 92.3% [reuse] and chain-of-thought
> tokens at 2.2%" [verified]

System prompts are reused constantly; chain-of-thought tokens essentially never are,
because stochastic decoding makes them unique. SAECache routes blocks to per-type
queues with learned weights updated from eviction feedback, and gets **+4.8–5.9
percentage points of hit ratio** over the strongest baseline, with TTFT 1.4–2.7×
better on heterogeneous workloads [verified]. It is honest that on LMSys the TTFT gain
over LRU is only 4–8%, because the extra hits do not always outweigh the multi-queue
management overhead.

**[inferred] for us:** the 756× number is the actionable part even without adopting
SAECache. GLM-5.2 with reasoning tokens emits large volumes of CoT KV that is
near-worthless to cache. A single rule — *do not admit reasoning-span KV blocks to the
prefix cache at all* — captures most of the value with none of the machinery.

### 3.4 Multi-tier KV storage: LMCache, Mooncake Store, NIXL, HiCache

The tiering stack that has actually shipped:

**LMCache** is the KV cache *layer* — it extracts KV out of the engine and stores it
across CPU RAM, local SSD, Redis/Valkey, Mooncake, InfiniStore, S3-compatible object
storage, NIXL, and GDS. It serves both cache offloading (prefix reuse across queries)
and PD disaggregation (cross-engine transfer). Contributions per the tech report:
batched data-movement ops with compute/IO pipelining; a modular connector decoupled
from engine internals; and a control API (`pin`, `lookup`, `cleanup`, `move`,
`compress`, plus `batched_admit` / `batched_evict` for routers to consume).

Measured on 8× H100 across Llama3.1-8B/70B, Qwen2.5-72B, Qwen2.5-Coder-32B, and
Qwen3-Coder-480B-A35B-FP8: **1.9–8.1× smaller TTFT at QPS=1 and 2.3–14× higher
throughput at equal TTFT** vs basic vLLM, and 7–92% smaller ITL [verified].

The report's two production lessons are worth more than the multipliers:

> "**Loading from remote storage is faster than prefill.** ... with Amazon S3 Express,
> the throughput has increased from 100 MB/s ... achieving 22–32% lower TTFT compared
> to full prefill." [verified]
>
> "**Context truncation lowers prefix cache hit rates.** ... prefix cache hit ratios
> drop from roughly 85% to 45% when truncating input contexts to keep only the latest
> tokens." [verified]

**NIXL** (NVIDIA Inference Xfer Library, `github.com/ai-dynamo/nixl`) is the transport
abstraction underneath: a plugin architecture over UCX, GDS, POSIX, object storage,
Azure Blob, HF3FS, Mooncake, GUSLI, UCCL, GPUNETIO, and libfabric, with C++ and Python
APIs. It is Dynamo's transfer layer, and ELDR's evaluation uses vLLM's NIXL connector
for PD disaggregation.

**SGLang HiCache** is the in-engine hierarchical cache. Verified flags and defaults
from current `server_args.py`:

| Flag | Default | Notes |
|---|---|---|
| `--enable-hierarchical-cache` | `False` | master switch |
| `--hicache-ratio` | `2.0` | host pool size ÷ device pool size |
| `--hicache-size` | `0` | GB; overrides ratio if set |
| `--hicache-write-policy` | `write_through` | `write_back` \| `write_through` \| `write_through_selective` |
| `--hicache-io-backend` | `kernel` | `direct` \| `kernel` \| `kernel_ascend` |
| `--hicache-mem-layout` | `page_first` | `layer_first` \| `page_first` \| `page_first_direct` \| `page_first_kv_split` \| `page_head` |
| `--hicache-storage-backend` | `None` | `file` \| `mooncake` \| `hf3fs` \| `nixl` \| `aibrix` \| `dynamic` \| `eic` \| `simm` \| `mori` \| `shm` |
| `--hicache-storage-prefetch-policy` | `timeout` | `best_effort` \| `wait_complete` \| `timeout` |
| `--enable-lmcache` | `False` | alternative hierarchical backend |

vLLM's equivalents: `--kv-offloading-size` (GiB) and `--kv-offloading-backend`
(`lmcache` \| `native`, default `native`) [verified from docs].

**[inferred] for our box:** with 183 GB HBM3e per GPU and 8 GPUs we have a lot of
device-side room, and a two-socket host with a fat DRAM pool. The L1(GPU)→L2(host DRAM)
tier is nearly free to enable (`--enable-hierarchical-cache --hicache-ratio 2.0`) and
directly attacks the human-gap eviction problem TraceLab identifies. NVMe/object tiers
buy nothing for a latency leaderboard.

### 3.5 CacheBlend: reuse beyond the prefix

Standard prefix caching requires the reused text to be an exact *prefix*. In RAG and
agent contexts, the reusable chunks sit in the middle. CacheBlend precomputes KV for
each chunk independently and then **selectively recomputes a small subset of tokens**
to repair the missing cross-attention, recovering full-prefill generation quality. The
recompute is pipelined against the KV fetch, so slower/cheaper storage tiers become
usable without adding latency. Shipped inside LMCache as its non-prefix reuse path.

**[inferred] for us:** low priority. Our leaderboard traffic is prompt-then-generate,
not RAG chunk assembly, and the quality risk of approximate KV fusion is not worth it
for a P50 latency metric.

---

## 4. Routing, load balancing, and the DP-skew problem

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Preble: Efficient Distributed Prompt Scheduling for LLM Serving | Srivatsa, He et al., UCSD | ICLR 2025, arXiv:2407.00023 | 2–4× A6000; 8× H100 (Llama-3-70B TP4+DP) | 1.5–14.5× avg latency, 2–10× p99 vs SGLang [verified] | Ideas yes — SGLang Router, Dynamo, KubeAI implement variants |
| Llumnix: Dynamic Scheduling for Large Language Model Serving | Sun et al., Alibaba | OSDI 2024, arXiv:2406.03243 | 16× A10 24GB (4 VMs, PCIe 4.0) | P99 TTFT up to 15×, P99 decode 2×, 1.5× for high-priority, 36% cost saving [verified] | Open-source (AlibabaPAI/llumnix); ideas in Dynamo |
| A Universal Load Balancing Principle and its Application to LLM Serving | Chen, Bu, Song, Lu, Ye, Zhou; HKUST + PKU | arXiv:2601.17855 | industrial traces + simulation | **>40% of per-decode-step compute lost to barrier idle**; BF-IO cuts energy 28.2% [verified] | No |
| Tackling the Data-Parallel Load Balancing Bottleneck (BalanceRoute) | Bu, Lyu, Chen, Ye, Zhou; HKUST + Huawei | arXiv:2605.06113 | **144-NPU Ascend 910C** cluster | substantial DP imbalance reduction vs vLLM baselines on Azure-2024 + proprietary traces [reported] | No |
| ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving | Choi, Cho, Xiong et al., KAIST + MSR | arXiv:2607.00466 | 5 nodes × 8 **MI300X**, vLLM 0.21 / ROCm 7.2, NIXL | median TPOT −5.9–13.9% vs best load balancer; 0.8% overhead [verified] | No |
| SkyWalker (SkyLB): A Locality-Aware Cross-Region Load Balancer | Xia, Mao et al., Berkeley Sky | arXiv:2505.24095 | multi-region | 1.12–2.06× throughput, 1.74–6.30× lower latency, 25% cost reduction [reported] | No |
| Online Linear Programming for Multi-Objective Routing in LLM Serving | Chen, Ye, Zhou; Stanford/HKUST | arXiv:2607.03948 | Vidur simulator | bid-price control beats heuristics across SLO regimes [reported] | No |

### 4.1 The routing objective: cache locality vs load, and why pure affinity loses

Every practical router balances two signals: **cache overlap** (what fraction of this
request's prefix blocks are already resident on a replica) and **load** (how saturated
that replica is). Mooncake's Algorithm 1 is the canonical formulation — it will send a
request *away* from its best prefix match when `best_prefix_len − prefix_len <
kvcache_balancing_threshold` fails, i.e. when the match advantage is large enough to
justify a KV *transfer* to a less loaded instance instead.

Preble's version is an explicit exploitation/exploration split (their E2 algorithm):
exploit prefix locality when a strong match exists, explore to balance when it does
not, with hierarchical local+global scheduling.

**ELDR provides the cleanest evidence that pure affinity is wrong.** It sweeps a
"locality band" width τ, where τ=0 means route strictly to the best-matching worker and
larger τ means "pick the least-loaded worker among those within τ of the best match."
At τ=0, tail TPOT **regresses versus round-robin on four of six workloads — by as much
as 7.9%** [verified] — because a burst of similar requests all lands on one worker.
They settle on τ=0.1.

**[inferred] for us:** we run one replica, so cross-replica routing is out of scope
today. But the same τ logic will apply the moment we run DP across nodes, and the
lesson — *never route on affinity alone* — is cheap to internalise now.

### 4.2 ELDR: routing on expert locality, an MoE-specific signal

Worth its own paragraph because it is the only paper that routes on something other
than load or prefix. In a PD-disaggregated MoE deployment, each decode step must load
the weights of **every distinct expert its batch activates**. Two equally loaded
workers can therefore have very different latency. ELDR builds an **expert signature**
from a request's *prefill* activations (available before decode begins), partitions
signature space across decode workers with balanced K-means offline, and routes online
to the least-loaded worker within the locality band.

The signature cache is co-indexed with the KV cache at block granularity so signatures
stay exact under prefix caching, and costs **<1% of KV cache** with total routing
overhead of **0.8% of TTFT** [verified]. Gains grow with pool size — 8.0% / 9.8% /
10.2% median-TPOT reduction at 8P8D / 8P16D / 8P24D — because more decoders means
narrower expert coverage per worker.

Hardware caveat: **AMD MI300X**, not NVIDIA. The mechanism is architecture-neutral but
the magnitudes are not portable.

### 4.3 The barrier-idle result, and why it is *not* our 47%

Chen et al. (arXiv:2601.17855) measure, on a real industrial trace, that **mean 40% /
median 41% of per-decode-step time is barrier idle** — faster DP workers waiting at the
synchronisation barrier for the straggler [verified]. Fixing it with their BF-IO policy
reduces total energy 28.2%.

BalanceRoute (same group, arXiv:2605.06113) attacks it with routing. It names the four
things that make LLM DP balancing hard and that generic heuristics ignore: assignments
are **sticky** (migrating KV is expensive), per-request load **grows over time**,
arrivals are **non-stationary**, and the router has a **sub-100 ms decode budget** over
hundreds of waiting requests. Its BR-0 uses a piecewise-linear "F-score" capturing the
asymmetry between admissions that fill safe margin and those that overflow; BR-H adds a
short constant lookahead H with a binary "terminates within H steps" classifier. Their
own justification for the binary classifier over a length regressor is worth quoting:

> "beyond the binary threshold 'terminates within H steps,' the precise heavy-tailed
> value of r_i(k) has only second-order effect on routing — so a binary classifier
> captures the routing-relevant signal more reliably than a full-length regressor."

Evaluated on a **144-NPU Ascend 910C** cluster against vLLM baselines
(round-robin / JSQ / random / power-of-two-choices).

**The distinction that matters for us.** Their 40% is *inter-replica*: DP workers with
different request mixes. Our 47%-of-collectives is *intra-TP-group* on one node: eight
ranks running the same layers on the same batch, arriving at the NVLink collective at
different times. Routing cannot fix that. Ours is CPU-side launch jitter, per-rank
kernel duration variance (very plausible with MoE: expert load differs per rank), and
scheduler-loop nondeterminism. The relevant mitigations are:

- **Persistent kernels** so per-step launch overhead disappears (ExpertPlex APK).
- **Full CUDA-graph capture** of the decode step.
- **`async_scheduling`** — current vLLM ships this and it is on by default:
  "Async scheduling helps to avoid gaps in GPU utilization" [verified from
  `vllm/config/scheduler.py`].
- **`prefill_schedule_interval`** — vLLM's newer knob: "For data-parallel deployments,
  only admit new prefill requests once every N engine steps, **aligned across DP
  ranks**, to better balance per-step forward-pass times" [verified from source]. This
  is the productionised form of the barrier-idle fix, and confirms the problem is real
  enough that vLLM added a scheduler knob for it.
- **Expert-load balancing** — DeepSeek's EPLB and LPLB (linear-programming expert
  parallel load balancer) are the deployed answers for the MoE-specific component of
  per-rank variance.

---

## 5. SLO-aware scheduling, goodput, admission control, and preemption

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| AlpaServe: Statistical Multiplexing with Model Parallelism | Li, Zheng, Zhong et al., Berkeley/PKU/Stanford | OSDI 2023, arXiv:2302.11665 | 64-GPU cluster; 8× V100-16GB microbenchmarks | 10× higher rates, 2.5× lower deadlines, or 6× burstiness at >99% SLO attainment [verified] | Ideas yes; system is pre-LLM-era |
| Fast Distributed Inference Serving for LLMs (FastServe) | Wu et al., PKU | arXiv:2305.05920 (no top-tier venue found) | 16× A100, OPT-175B | 31.4× throughput at same avg latency, 17.9× at same tail, vs vLLM [reported] | No |
| SLOs-Serve: Optimized Serving of Multi-SLO LLMs | Chen, Jia et al., CMU + Google | arXiv:2504.08784 | 6 application scenarios | 2.2× average per-GPU serving capacity [reported] | No |
| Niyama: Breaking the Silos of LLM Inference Serving | Goel, Mohan, Kwatra, Anupindi, Ramjee; MSR India | arXiv:2503.22562 | — | +32% serving capacity; order-of-magnitude fewer SLO violations under extreme load [reported] | No |
| PolyServe: Efficient Multi-SLO Serving at Scale | Zhu, Shi, Xu, Shan, Krishnamurthy, Kasikci; UW + ByteDance | arXiv:2507.17769 | C++ implementation + simulator | 1.23× (PD) / 1.18× (colocated) goodput; 92.5% of optimal [verified] | No |
| Cascade: SLO-aware latency budget for fair, high-goodput serving | Adnan, Mahapatra, Nair; UBC | arXiv:2608.06557 | — | per-request latency budget = SLO − predicted remaining service time | No |
| TurboSpec: Closed-loop Speculation Control for Optimizing Serving Goodput | Liu, Park, Hu, Kwon et al., Berkeley/UCSD | arXiv:2406.14066 | H100 80GB, L40S 48GB; Llama2-7B + 160M draft | optimal k drops with batch size; auto-disables spec at BS≥16 [verified] | Ideas partially in vLLM's spec-decode autotuning |

### 5.1 Goodput is the right objective, and it is not throughput

DistServe's definition — max per-GPU request rate at ≥90% SLO attainment on *both*
TTFT and TPOT — is now standard. TurboSpec extends it in the direction we need:
speculative goodput counts only **accepted** tokens, so a spec-decode configuration
that raises raw token throughput while lowering acceptance can reduce goodput.

For our leaderboard the metric is stricter still: **P50 latency over 72 hours** at C1
and C10. That is a distributional target with a long observation window, which means:

1. **Rare catastrophic events matter more than average efficiency.** Llumnix's
   measurement is the warning: serving LLaMA-7B on vLLM at only **62% average memory
   load, 8% of requests were preempted**, P99 per-token latency was 3.8× P50, and
   preemption accounted for **70% of the P99 request's latency** — one request lost
   50 seconds to two preemptions [verified]. Over 72 hours you will hit this.
2. **Admission control beats optimism.** Mooncake's early rejection exists because
   accepting work you cannot finish wastes the prefill you already paid for.
3. **Preemption should be rare and cheap.** Recompute over swap (vLLM §7), and
   `watermark` headroom to avoid thrash — current vLLM exposes
   `watermark: float = 0.0`, "Fraction of total KV cache blocks to keep free ... helps
   avoid frequent KV cache eviction and the resulting repeated preemption" [verified
   from source], plus `scheduler_reserve_full_isl: bool = True`, "the scheduler checks
   whether the full input sequence length fits in the KV cache before admitting a new
   request, rather than only checking the first chunk. Prevents over-admission and KV
   cache thrashing with chunked prefill" [verified from source].

### 5.2 Multi-SLO scheduling: three approaches

**SLOs-Serve** (CMU + Google) treats it as resource planning: a **dynamic-programming
algorithm over token allocations**, exploring the joint design space of chunked prefill
chunk sizes *and* optional speculative decoding, per stage, per SLO. 2.2× average
per-GPU capacity across summarisation, coding, chatbot, tool calling, and reasoning
[reported]. The idea that chunk size and speculation length are one joint decision
variable is the transferable part.

**Niyama** (MSR India) adds QoS classes with real deadlines, **dynamic chunking**
(chunk size varies by how much slack the batch's requests have), a hybrid
eager-deadline + slack prioritisation, and **selective relegation** — under overload,
demote some requests to a degraded class rather than violating everyone's SLO. +32%
capacity, and an order of magnitude fewer violations under extreme load [reported].

**PolyServe** (UW + ByteDance) partitions the *fleet* into TPOT tiers and routes each
request to the **highest-load server that can still meet its SLO**. The counterintuitive
choice is deliberate: packing servers to just-below-SLO creates a **load gradient** that
makes autoscaling decisions unambiguous, and leaves genuinely idle servers to scale
down. Looser-SLO requests may be lazily promoted onto tighter-SLO instances when their
own tier saturates. 1.23× (PD) / 1.18× (colocated) goodput, 92.5% of theoretical
optimum [verified]. Implementation is C++ plus a simulator, not a deployed engine.

### 5.3 Preemption and migration: Llumnix

Llumnix's mechanism is the best-engineered idea in the fleet-scheduling literature and
it rests on one observation: **the KV cache is append-only**. Already-computed KV never
changes, so you can copy it *while decoding continues*.

Multi-stage migration:
- Stage 0: begin copying all currently-complete KV blocks; keep decoding on the source.
- Stage k: copy the blocks generated during stage k−1; keep decoding.
- Stage N: only one iteration's worth remains — drain the request from the batch, copy
  the last block, resume on the destination.

Downtime is one iteration, **constant in sequence length**. The naive alternatives
(recompute, or stop-and-copy) cost **over 50× the decoding cost** [verified].

The scheduling-policy contribution is **virtual usage**: rather than encode four
different goals (load balancing, de-fragmentation, prioritisation, draining for
autoscale) as four policies, express each as a rule that sets a *fictitious* memory
usage for certain requests, then run one load-balancing policy over virtual usage. To
free space on an instance, inflate the virtual usage of requests there and let load
balancing migrate them out. To reflect a queued request's latent demand, give it
positive virtual usage despite zero physical usage.

Testbed: 16× **A10 (24 GB)** over PCIe 4.0 with 64 Gb/s networking — old, small, and
bandwidth-poor. The 15× P99 TTFT improvement is against INFaaS++ on that hardware.
[inferred] on NVLink5 the migration mechanism is far cheaper still, but the *need* for
migration is lower because 183 GB of HBM makes fragmentation and preemption rarer.

### 5.4 FastServe: read for the diagnosis, not the number

FastServe's diagnosis is genuinely important: it measures that on ShareGPT and Alpaca
at load ≈ 1, **up to 90% of total request latency is queuing delay**, driven purely by
the long-tail output-length distribution causing head-of-line blocking [verified].

Its solution is a **skip-join MLFQ**. Classic MLFQ puts every job in the top queue and
demotes it. LLM inference is *semi-information-agnostic*: output length is unknown but
**input length is known**, and input length determines the first-iteration cost, which
for long-prompt/short-output requests dominates. So a new request *skips* directly to
the queue whose quantum matches its first-iteration time, avoiding a cascade of
demotions.

The 31.4× is not transferable. It is OPT-175B on 16× A100-40GB against 2023-vintage
vLLM in a severely memory-constrained regime — their own numbers show a job's KV at
2.3 GB, decode step at 60 ms, and a PCIe 4.0 ×16 swap at 36 ms, i.e. swap cost is 60%
of a decode step. On B200 with 183 GB, the memory pressure that generates that speedup
does not exist. The paper also appears never to have landed at a top-tier venue.

### 5.5 Speculative decoding as a scheduling decision (TurboSpec)

The most directly actionable paper in this section for us.

**The insight.** Batching gives *inter*-request parallelism; speculation gives *intra*-
request parallelism. They are substitutes competing for the same compute. When batch
size is low (our C1 case), inter-request parallelism is unavailable and speculation is
nearly free. When batch size is high, speculation is stealing compute from real work.

**Measured shape** (Llama2-7B target, Llama2-160M draft, H100) [verified]:

| Condition | Optimal proposal length k |
|---|---|
| BS = 1 | 4–5 |
| BS = 64 | 1–2 |
| acceptance 0.5 | ~3 |
| acceptance 0.9 | ~7 |

and at BS ≥ 16 speculation becomes net-negative for that model, so TurboSpec disables
it and holds speedup at ≈0.97 rather than the significant degradation static
speculation suffers.

**The controller.** Offline: profile execution latency across (batch size, proposal
length). Online: track the empirical acceptance rate and forecast expected accepted
tokens per step; choose the k maximising predicted goodput each step.

**For us.** EAGLE 3-1-4 is a fixed (steps=3, topk=1, draft tokens=4) configuration
serving C1 and C64 identically. Our 3.09× at C1 tells us acceptance is high — and high
acceptance is exactly the regime where TurboSpec says to propose *more*. Meanwhile at
C64, 3-1-4 may be past the crossover. The relevant SGLang knobs are
`--speculative-num-steps`, `--speculative-eagle-topk`, and
`--speculative-num-draft-tokens` [verified present in `server_args.py`], and they are
static. A closed-loop controller over them is a contained engineering project with a
plausible double-digit win at both ends of our concurrency range.

Two honest caveats: TurboSpec measures a 7B dense model with a separate small draft
model, not EAGLE-style self-drafting on a 400B MoE; and on MoE the marginal cost of a
larger draft tree is different, because verifying more tokens changes which experts
activate.

---

## 6. Output-length prediction and SJF-style scheduling

| Paper | Lab | Venue / year | Approach | Result | In production? |
|---|---|---|---|---|---|
| Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction (SSJF) | Qiu et al. | AIOps'24, arXiv:2404.08509 | fine-tuned BERT-base proxy predicts output length; speculative SJF | 30.5–39.6% lower JCT, 2.2–3.6× throughput vs FCFS [reported] | No |
| Efficient LLM Scheduling by Learning to Rank | Fu, Zhu, Su, Qiao, Stoica, Zhang; UCSD/Berkeley/Snowflake | NeurIPS 2024, arXiv:2408.15792 | predict *relative ranks*, not absolute lengths | 2.8× lower latency (chatbot), 6.5× higher throughput (synthetic data gen) [reported] | No — code at hao-ai-lab/vllm-ltr |
| Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions | — | arXiv:2604.00499 | output length is log-t heavy-tailed; Tail Inflated Expectation replaces point estimate in SJF | — | No |
| Robust Length Prediction: Heavy-Tailed Prompt-Conditioned Distributions (ProD) | Wang, Qian, Xue, Qian, Zhao | arXiv:2604.07931 | train on *multiple* generations per prompt; median (ProD-M) or distributional (ProD-D) target | consistent prediction-quality gains [reported] | No |

### The arc of this literature is a retreat, and that is the useful signal

2024's position was "predict the output length and run SJF." 2026's position is
"you cannot, and you should stop trying to."

- **Learning to Rank** made the first retreat: *don't predict lengths, predict the
  ordering*. Ranking is a much easier statistical problem and is all SJF needs.
- **ProD** made the second: even the *label* is wrong. "even under a fixed model and
  decoding setup, the same prompt induces a prompt-conditioned output length
  distribution, not a deterministic scalar, and this distribution is consistent with
  heavy-tailed behavior." Training on a single sampled length is fitting noise.
- **TIE** made the third: if the distribution is heavy-tailed (they fit a **log-t**),
  the *mean* is the wrong statistic for SJF — you need a tail-inflated functional.
- **BalanceRoute** made the fourth and most pragmatic: replace the regressor with a
  **binary classifier** for "will this terminate within H steps," because "the precise
  heavy-tailed value ... has only second-order effect on routing."

**[inferred] for us:** at C1 and C10 there is nothing to schedule — SJF requires a
queue and we barely have one. Length prediction is off the critical path entirely. Its
only relevance is as a *KV admission* signal (do not admit a request whose predicted
length cannot fit), and even there the literature now says a binary classifier is the
right tool, not a regressor.

---

## 7. Queueing theory that actually applies

| Paper | Lab | Venue / year | Result |
|---|---|---|---|
| Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents | Dai, Deng, Li, Peng | arXiv:2504.07347 | work-conserving ⇒ throughput-optimal; classifies real systems [verified] |
| A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints | Nie, Si, Zhou | **ICML 2026**, arXiv:2605.04595 | closed-form stability conditions under joint compute + KV constraints; predictions within ~10% on real GPUs [reported] |
| Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics | Zhou | arXiv:2605.01280 | argues JSQ/RR + FIFO + LRU ignore LLM structure |
| A Universal Load Balancing Principle | Chen, Bu, Song, Lu, Ye, Zhou | arXiv:2601.17855 | >40% barrier idle measured; worst-case guarantees for imbalance reduction |

### The one theorem to internalise

Dai et al. build a fluid-limit model of a multi-class batched processing network under
K-FCFS and prove that **any work-conserving scheduler achieves maximum throughput**,
for single servers, DAG-routed agent workloads, and fork-join topologies. Then they
classify [verified, their Table 1]:

| Algorithm | Priority | Batching | Work-conserving? |
|---|---|---|---|
| FasterTransformer | decode-first | no mixed batching | **No** |
| Vanilla vLLM (2023) | prefill-first | no mixed batching | **No** |
| Orca | prefill-first | mixed batching | Yes |
| Sarathi-Serve | decode-first | chunked prefill + mixed | Yes |

They footnote explicitly that "the latest vLLM with chunked prefill enabled is
work-conserving." So the practical instruction is: *keep mixed batching on*. A system
that ever runs a decode-only batch while prefill work waits (or vice versa) is leaving
capacity on the floor in a way that compounds into instability, not just slowness.

**Two non-obvious corollaries they also prove.**

1. **Batch-size caps break the theorem.** "This constraint fundamentally alters the
   stability region from a scalar threshold to a two-dimensional convex hull, and we
   demonstrate that even work-conserving algorithms can fail under certain
   configurations." Concretely, in their construction with token budget b_max=128
   Sarathi-Serve is the only stable scheduler — it builds batches of exactly 29 prefill
   + 99 decode tokens, hitting the required operating point. At b_max=1024 **Sarathi
   itself destabilises**, because it fills batches to 1024 tokens, accelerates prefill,
   accumulates k_max=100 outstanding decodes, and then runs decode-only batches of 100
   tokens at an inefficient rate [verified]. *Token budget and max-concurrent-requests
   are coupled and must be tuned jointly.*
2. **Cyclic routing can break work-conservation.** Borrowing the Rybko–Stolyar
   counterexample, they construct an agent workload with cyclic routing where a
   work-conserving policy is unstable. Relevant as we serve agentic traffic where a
   request's output feeds a tool that feeds another request.

Nie, Si & Zhou (ICML 2026) complete the picture with the constraint Dai et al. abstract
away: **GPU memory**. They derive stability/instability conditions accounting jointly
for compute and KV capacity, and validate on real GPUs to within ~10%. The practical
use is capacity planning — given an arrival rate, compute the stable service rate and
size the cluster, rather than guessing.

---

## What is NOT worth it

Techniques that read well and either fail in production or are wrong for our target.

**1. Chunked prefill at low concurrency.** Chunking is a *protection* mechanism for
co-running decodes. At C1 there are none. You pay O(N²) KV re-reads, tile-quantization
cliffs, and — on MoE — a measured +39% expert-weight traffic, in exchange for nothing.
Sarathi's own overhead number (~25% at chunk 512) is the price of a benefit we are not
buying. If the engine will not let you disable it, set the budget above the longest
prompt you serve so chunking never triggers.

**2. PD disaggregation on a single 8-GPU node when you are scored on latency.** Every
GPU you dedicate to prefill halves the tensor-parallel width available to decode, which
is the phase our leaderboard measures. TaiChi's table quantifies the corner we are in:
tight TTFT / relaxed TPOT gives aggregation 97% attainment and disaggregation 42%.
Disaggregation is a *cluster cost-efficiency* technique. It has never been a
single-node single-stream latency technique and the papers that promote it do not
claim otherwise.

**3. Swapping KV to host memory as a preemption mechanism.** vLLM's own §7 shows
recompute is never worse than 1.2× swap and is better at small block sizes. FastServe's
architecture is built entirely on proactive host swapping and its numbers come from
16× A100-40GB where the KV simply did not fit. With 183 GB HBM3e per GPU, our
preemption rate should be near zero; if it is not, the fix is a `watermark`, not a
faster swap path.

**4. Beam search KV sharing.** PagedAttention showcases up to 55% memory savings from
beam-search branch sharing, and parallel sampling saves 12% on the shared prompt.
Nobody beam-searches a modern chat or agent model. This capability drove real
complexity in vLLM's block manager (copy-on-write, ref counting, fork/merge) for a
decoding mode that has essentially disappeared. Do not build it.

**5. Output-length regression as a scheduling input.** Four separate 2026 papers
independently conclude the target is heavy-tailed and a point estimate is misspecified;
the field's own retreat path (exact length → rank → distribution → binary
"terminates-within-H" classifier) is the verdict. At C1–C10 there is no queue to sort
anyway.

**6. Clever prefix-cache eviction policies.** Mooncake measured LRU against LFU and a
length-aware policy on production Kimi traffic and LRU won at every capacity. Hit rate
saturated at ~50K blocks with an absolute workload ceiling of 0.51. The returns are in
*capacity* and *retention across idle gaps*, not in policy sophistication. The one
exception worth acting on is coarse and free: don't cache chain-of-thought spans
(2.2% reuse vs 92.3% for system prompts).

**7. Pure prefix-affinity routing.** ELDR measured τ=0 (strict best-match routing)
regressing p99 TPOT versus plain round-robin on four of six workloads, by up to 7.9%.
Every production router — Mooncake's Conductor, Preble's E2, SGLang Router, Dynamo —
mixes locality with load for this reason. "Route to the replica with the longest prefix
match" is a trap.

**8. AlpaServe's core thesis, for us.** Its result is that *deliberately over-
parallelising a model that fits on one GPU* improves statistical multiplexing under
bursty multi-model traffic. It is a genuinely important OSDI paper, but the setting is
many small models sharing a cluster, measured on 8× V100-16GB, in a pre-continuous-
batching world. We serve one large model on a dedicated node. The durable takeaway is
the queueing analysis of model-parallel overhead versus burst absorption, not the
system.

**9. Position-independent / non-prefix KV fusion (CacheBlend-style) for a latency
leaderboard.** It buys real TTFT on RAG assembly workloads, but it approximates the
attention state and requires selective recomputation to recover quality. For a P50
latency score on prompt-then-generate traffic there is no reuse pattern for it to
exploit and a nonzero quality risk.

**10. Chasing 2023-era headline multipliers.** Orca's 36.9×, FastServe's 31.4×, and
vLLM's 22× are all against baselines (FasterTransformer, request-level batching,
max-length KV reservation) that no longer exist anywhere. Every one of those techniques
is already inside SGLang and therefore inside our engine. The remaining headroom is in
the 1.1–1.7× papers, not the 20×+ ones.

---

## Sources

Every URL below was fetched and read during this survey.

**Foundational batching and memory**
1. Yu, Jeong, Kim, Kim, Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* Seoul National University + FriendliAI. **OSDI 2022**. https://www.usenix.org/system/files/osdi22-yu.pdf
2. Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* UC Berkeley et al. **SOSP 2023**. arXiv:2309.06180 — https://arxiv.org/pdf/2309.06180
3. Prabhu, Nayak, Mohan, Ramjee, Panwar. *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention.* Microsoft Research India. **ASPLOS 2025**. arXiv:2405.04437 — https://arxiv.org/pdf/2405.04437

**Chunked prefill vs disaggregation**
4. Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* MSR India + Georgia Tech. **OSDI 2024**. arXiv:2403.02310 — https://arxiv.org/pdf/2403.02310
5. Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* PKU + UCSD + StepFun. **OSDI 2024**. arXiv:2401.09670 — https://arxiv.org/pdf/2401.09670
6. Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini. *Splitwise: Efficient generative LLM inference using phase splitting.* Microsoft + UW. **ISCA 2024**. arXiv:2311.18677 — https://arxiv.org/pdf/2311.18677
7. Qin, Li, He, Zhang, Wu, Zheng, Xu. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (DBLP lists the venue version as *Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot*). Moonshot AI + Tsinghua. **FAST 2025**. arXiv:2407.00079 — https://arxiv.org/pdf/2407.00079
8. Wang, Zuo, Chen, Liang, Yu, Yang. *Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving* (TaiChi). CUHK + Huawei Cloud. arXiv:2508.01989 — https://arxiv.org/pdf/2508.01989
9. Wu, Bambhaniya, Banerjee, Khare, Srinivasan, Subramanian, Kundu, Kumar, Elavazhagan, Won, Yazdanbakhsh, Krishna. *How Far Can Disaggregation Go? A Design-Space Exploration of Attention–FFN Disaggregation for Efficient MoE LLM Serving.* Georgia Tech + Intel + Google/DeepMind. arXiv:2605.28302 — https://arxiv.org/pdf/2605.28302
10. Wu, Jin, Zhang, Wei, Zhong, Zhu et al. *ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels.* Peking University. arXiv:2607.18002 — https://arxiv.org/pdf/2607.18002
11. Lee, Kim, Park, Lee, Ahn. *From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill.* Seoul National University. arXiv:2510.08055 — https://arxiv.org/pdf/2510.08055
12. Shi, Cai, Du et al. *Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving.* arXiv:2507.06608 — https://arxiv.org/pdf/2507.06608
13. Chen, Cui, Zhao et al. *Towards High-Goodput LLM Serving with Prefill-decode Multiplexing* (MuxWise). arXiv:2504.14489 — https://arxiv.org/pdf/2504.14489
14. Ruan, Chen, Tian et al. *DynaServe: Unified and Elastic Execution for Dynamic Disaggregated LLM Serving.* arXiv:2504.09285 — https://arxiv.org/pdf/2504.09285
15. Forys, Wu, Xiao, Nie, Liu, Antonova, Jones, Mullins, Luk, Zhao, Constantinides. *When Does Disaggregation Pay? Simulating Prefill–Decode–Attention–FFN Specialization for Agentic LLM Inference.* Imperial College London + Cambridge. arXiv:2608.03741 — https://arxiv.org/pdf/2608.03741

**Prefix and KV cache reuse**
16. Zheng, Yin, Xie, Sun, Huang, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng. *SGLang: Efficient Execution of Structured Language Model Programs.* Stanford + UC Berkeley + SJTU + Texas A&M. **NeurIPS 2024**. arXiv:2312.07104 — https://arxiv.org/pdf/2312.07104
17. Yao, Li, Liu, Ray, Cheng, Zhang, Du, Lu, Jiang. *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion.* UChicago + CUHK-Shenzhen + Stanford + Microsoft Research. **EuroSys 2025**. arXiv:2405.16444 — https://arxiv.org/pdf/2405.16444
18. Liu, Yao, Cheng, An, Chen, Feng, Huang, Shen, Zhang, Du, Jiang. *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference.* Tensormesh + University of Chicago. Tech report, 2025 — https://lmcache.ai/tech_report.pdf
19. Zhu, Jacob, Ma, Pan, Wang, Krishnamurthy, Kasikci. *TraceLab: Characterizing Coding Agent Workloads for LLM Serving.* University of Washington SyFI. arXiv:2606.30560 — https://arxiv.org/pdf/2606.30560
20. Fang, Li, Wu et al. *Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches* (SAECache). Peking University. arXiv:2605.18825 — https://arxiv.org/pdf/2605.18825
21. Li, He, Mang, Zhang, Mao, Chen et al. *Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live.* arXiv:2511.02230 — https://arxiv.org/pdf/2511.02230
22. Qin, He, Wang, Li, Xu, Wu, Zheng, Zhang. *Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter.* Moonshot AI + Tsinghua. arXiv:2604.15039 — https://arxiv.org/pdf/2604.15039
23. Hu, Huang, Hu, Xu, Chen, Xie, Wang, Wang, Bao, Sun, Shan. *MemServe: Flexible Mem Pool for Building Disaggregated LLM Serving with Caching.* Huawei Cloud + UCAS/ICT CAS + Peking University. arXiv:2406.17565 — https://arxiv.org/pdf/2406.17565 *(fetched; front matter and abstract read only — no claim in this document rests on it)*
24. NVIDIA. *NIXL — NVIDIA Inference Xfer Library.* https://github.com/ai-dynamo/nixl

**Routing and load balancing**
25. Srivatsa, He, Abhyankar, Li, Zhang. *Preble: Efficient Distributed Prompt Scheduling for LLM Serving.* UC San Diego. **ICLR 2025**. arXiv:2407.00023 — https://arxiv.org/pdf/2407.00023
26. Sun, Huang, Zhao, Xiao, Zhang, Li, Lin. *Llumnix: Dynamic Scheduling for Large Language Model Serving.* Alibaba Group. **OSDI 2024**. arXiv:2406.03243 — https://arxiv.org/pdf/2406.03243
27. Chen, Bu, Song, Lu, Ye, Zhou. *A Universal Load Balancing Principle and its Application to Large Language Model Serving.* HKUST + PKU. arXiv:2601.17855 — https://arxiv.org/pdf/2601.17855
28. Bu, Lyu, Chen, Song, Liang, Gurung, Fan, Ye, Zhou. *Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale* (BalanceRoute). HKUST + Huawei. arXiv:2605.06113 — https://arxiv.org/pdf/2605.06113
29. Choi, Cho, Xiong et al. *ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving.* KAIST + Microsoft Research. arXiv:2607.00466 — https://arxiv.org/pdf/2607.00466
30. Xia, Mao, Kerney, Jackson, Li, Xing, Shenker, Stoica. *SkyWalker (SkyLB): A Locality-Aware Cross-Region Load Balancer for LLM Inference.* UC Berkeley Sky Computing. arXiv:2505.24095 — https://arxiv.org/pdf/2505.24095
31. Chen, Ye, Zhou. *Online Linear Programming for Multi-Objective Routing in LLM Serving.* arXiv:2607.03948 — https://arxiv.org/pdf/2607.03948

**SLO-aware scheduling, goodput, admission control**
32. Li, Zheng, Zhong, Liu, Sheng, Jin, Huang, Chen, Zhang, Gonzalez, Stoica. *AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving.* **OSDI 2023**. arXiv:2302.11665 — https://arxiv.org/pdf/2302.11665
33. Wu, Zhong, Zhang, Liu, Liu, Sun, Huang, Liu, Jin. *Fast Distributed Inference Serving for Large Language Models* (FastServe). Peking University. arXiv:2305.05920 — https://arxiv.org/pdf/2305.05920
34. Chen, Jia et al. *SLOs-Serve: Optimized Serving of Multi-SLO LLMs.* CMU + Google. arXiv:2504.08784 — https://arxiv.org/pdf/2504.08784
35. Goel, Mohan, Kwatra, Anupindi, Ramjee. *Niyama: Breaking the Silos of LLM Inference Serving.* Microsoft Research India. arXiv:2503.22562 — https://arxiv.org/pdf/2503.22562
36. Zhu, Shi, Xu, Shan, Krishnamurthy, Kasikci et al. *PolyServe: Efficient Multi-SLO Serving at Scale.* University of Washington + ByteDance. arXiv:2507.17769 — https://arxiv.org/pdf/2507.17769
37. Adnan, Mahapatra, Nair. *Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving.* arXiv:2608.06557 — https://arxiv.org/pdf/2608.06557
38. Liu, Park, Hu, Kwon, Li, Zhang, Du, Mo, You, Cheung, Deng, Stoica, Zhang. *TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput.* UC Berkeley + UCSD. arXiv:2406.14066 — https://arxiv.org/pdf/2406.14066

**Output-length prediction**
39. Qiu et al. *Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction.* **AIOps'24**. arXiv:2404.08509 — https://arxiv.org/pdf/2404.08509
40. Fu, Zhu, Su, Qiao, Stoica, Zhang. *Efficient LLM Scheduling by Learning to Rank.* UCSD + Berkeley + Snowflake. **NeurIPS 2024**. arXiv:2408.15792 — https://arxiv.org/pdf/2408.15792
41. *Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions.* arXiv:2604.00499 — https://arxiv.org/pdf/2604.00499
42. Wang, Qian, Xue, Qian, Zhao. *Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions* (ProD). arXiv:2604.07931 — https://arxiv.org/pdf/2604.07931

**Queueing theory**
43. Dai, Deng, Li, Peng. *Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents.* Cornell ORIE + Columbia. arXiv:2504.07347 — https://arxiv.org/pdf/2504.07347
44. Nie, Si, Zhou. *A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints.* **ICML 2026**. arXiv:2605.04595 — https://arxiv.org/pdf/2605.04595
45. Zhou. *Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics.* arXiv:2605.01280 — https://arxiv.org/pdf/2605.01280

**Model- and engine-specific**
46. Hua, Wang, Yang, Wang. *GLM-5 Serving Parameter Tuning for OpenClaw: Single-Deployment MaaS Inference Optimization for Long-Context Agent Workloads.* Technical report. arXiv:2607.02518 — https://arxiv.org/pdf/2607.02518
47. Tan, Guo, Lv et al. *RTP-LLM: High-Performance Alibaba LLM Inference Engine.* arXiv:2605.29639 — https://arxiv.org/pdf/2605.29639
48. vLLM engine arguments documentation — https://docs.vllm.ai/en/latest/configuration/engine_args.html
49. vLLM `SchedulerConfig` source (`enable_chunked_prefill`, `policy`, `watermark`, `scheduler_reserve_full_isl`, `async_scheduling`, `prefill_schedule_interval`) — https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/scheduler.py
50. SGLang server arguments documentation — https://docs.sglang.io/advanced_features/server_arguments.html
51. SGLang `server_args.py` source (HiCache flags, `--schedule-policy` options, `--disaggregation-mode`, speculative-decoding flags) — https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py

**Referenced but not independently fetched** (cited *by* papers above, listed for
traceability, not treated as verified): DeepSeek EPLB — https://github.com/deepseek-ai/EPLB ;
DeepSeek LPLB — https://github.com/deepseek-ai/LPLB ; Azure LLM inference traces —
https://github.com/Azure/AzurePublicDataset .
