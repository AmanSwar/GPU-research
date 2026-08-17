# Baseten, Modal, fal and the infrastructure-first inference companies

## What this is

A technique-mining pass over the published engineering of the "infrastructure-first"
inference companies — the ones that sell GPU-hours or per-token APIs rather than models,
and that therefore have to be publicly good at inference.

**This document has been through a verification pass.** Every URL below was fetched and
read. Claims that could not be confirmed at the URL given have been deleted or explicitly
downgraded. A log of what the first pass got wrong is in
[Verification log](#verification-log--what-the-first-pass-got-wrong) near the end; read it
before trusting anything you remember from an earlier version of this file.

Scope actually covered, in descending order of yield:

| Company | Verdict | Why |
|---|---|---|
| **Baseten** | Highest-yield source in this file | Per-model Blackwell inference engineering with named techniques and AA-verified numbers. Ships open-source artifacts. Serves GLM-5.2 on B200 — our model, our GPU generation. Does **not** publish parallelism degrees for it. |
| **Cursor (Anysphere)** | Highest-yield *kernel* source, full stop | Not an inference vendor. Publishes the most reimplementable Blackwell MoE kernel work in the industry: a low-batch MoE decode kernel, an MXFP8 training kernel writeup, and an open-source MoE megakernel. |
| **fal** | High yield, narrow | Diffusion-first, but the comm/compute-overlap and NVFP4/MXFP8 kernel posts are 8xB200-native and directly portable. The DSpark post is the best public account of training a speculator for a fine-tuned MoE. |
| **Doubleword** | High yield, Hopper-centric, and the most *usable artifacts* | Per-kernel roofline decomposition of a DSA/MLA MoE decode step structurally identical to our own hotspot table; the best public spec-dec-vs-batching analysis; an MIT-licensed inference simulator with B200 presets; an open-source hardware-aware tokenizer. |
| **Modal** | Medium yield, excellent reference material | The FA4 reverse-engineering post and GPU Glossary are the best free Blackwell attention reference. Their platform work is cold-start, not latency. |
| **BentoML / Modular** | Medium, mostly a handbook | The LLM Inference Handbook is a solid reference with a few honest negative results. |
| **Replicate** | Low — say so plainly | Diffusion and platform posts only. One useful cold-start data point. No LLM inference engineering. |
| **RunPod** | Nothing transferable | Buyer education and fine-tuning VRAM math. Nothing about decode latency. |

Four honesty flags that apply throughout:

- **No number in this file has been independently reproduced.** Every figure is vendor
  self-reported. Cursor's and Doubleword's are the most checkable because the code and
  harnesses are released; Baseten's runtime is closed and none of its serving numbers can
  be reproduced from published artifacts.
- **Every "fastest on Artificial Analysis" claim is a vendor's reading of a third-party
  leaderboard at a point in time.** AA's methodology (~10k in / ≥1500 out, P50 over a
  trailing window, measured over the public internet from GCP) bundles network, TTFT and a
  specific traffic shape. Not comparable to our internal single-stream 365 tok/s unless we
  replicate the harness. Note especially that Baseten's headline 601 tok/s GLM-5.2 figure
  is stated at **~1k output tokens**, not AA's ≥1500. [verified]
- **Baseten publishes almost no parallelism configs.** They name techniques (TP+EP mixture,
  ADP, NVFP4, MTP) but publish TP/EP degrees for exactly one model: GPT-OSS 120B at TP8 EP1
  on 8xB200. Everything else is directionally useful, not a config.
- **Cursor's warp-decode kernel was measured on "a Qwen-3 style model" only.** The MoE shape
  table that includes GLM-5.2 (E=256, H=6144, I=2048, top-k 8) is in a *different* post
  about a *training* megakernel. Do not read the warp-decode result as validated on our
  shape. (The first pass of this file got this wrong.)

---

## Bottom line for our system

Ranked by (expected effect on our two objectives) × (probability it works) ÷ (difficulty).
Our measured C1 hotspots: dense GEMM 37.1%, collectives 19.6% (47% of that is rank arrival
skew), MoE expert GEMMs 19.4%, attention 10.9%, DSA indexer 5.8%.

Every "SGLang change" line below is **[inferred]** — my reasoning about what the technique
implies for our fork. Flag names are from the upstream project and must be checked against
our tree before use.

### Tier 1 — do these

**1. Suffix-automaton speculation layered on top of our EAGLE 3-1-4.**
The single highest expected-value-per-unit-effort item in this file, because the code
exists, is Apache-2.0, and is merged upstream into TensorRT-LLM.

- *Mechanism* [verified]: build a suffix automaton over the prompt on the host while KV
  prefill runs on device; transfer it to device before the first decode step; update it
  *on device* every step from a CUDA-graph-capturable `extend()` kernel launched
  `<<<batch_size, 1>>>`. If the suffix match length exceeds `SA_SPEC_THRESH` (default 4),
  emit the n-gram draft; otherwise fall back to the model draft. The automaton is a
  plain-old-data struct with a header-only core that compiles for both C++ and CUDA, so
  host→device transfer is zero-conversion.
- *Numbers* [reported, vendor]: 30–33% higher acceptance length **and** throughput across
  batch sizes vs MTP alone; a separate 34% acceptance-length figure; "up to 40% higher
  throughput at equal latency and up to 40% lower latency at equal throughput, compared to
  MTP alone" — all on `nvidia/DeepSeek-V3.1-NVFP4` with the `glaiveai/code-edit-samples`
  dataset. Hardware and batch size not published for the headline figures. Separately:
  "We commonly see up to 40% improvements on coding applications for production workloads."
- *Why it is safe*: they proved zero overhead by setting the threshold to infinity —
  disabling SA predictions while still running the computation — and confirming end-to-end
  latency matched baseline MTP.
- *Where it does nothing*: SA alone reaches accept length 10+ on long-context code but
  "accept rates near 0" on reasoning and writing. MTP alone gives 2–4 across domains. So
  this is an **agentic/code-traffic** lever, not a reasoning lever. AA's benchmark mix
  matters here.
- **Effect: large on repetitive/agentic traffic, ~0 on reasoning. Difficulty: low–medium.**
- **SGLang change** [inferred]: this is a *drafter selection* hook, not a new drafter. Port
  `sa_spec`'s POD core verbatim (it has no TRT-LLM dependency in the core algorithm) and
  add a per-request branch in the speculative draft path that chooses between the SA draft
  and the EAGLE draft. The `extend()` kernel is already CUDA-graph-safe by construction, so
  it can live inside our existing captured graph. Keep the threshold env-tunable so we can
  run the infinity-threshold zero-overhead check ourselves.
- Evidence: <https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/>,
  <https://github.com/basetenlabs/sa_spec>, NVIDIA/TensorRT-LLM PR #11434 (reference
  integration #10951).

**2. Pull-based dispatch + push-based combine for MoE all-to-all; stop using NCCL for it.**
This is the only published technique that attacks our specific worst bucket — the 47% of
collectives that is rank arrival skew (9.2% of C1 wall clock).

- *Mechanism* [verified]: with **push**-dispatch a rank must wait on signals from up to 71
  peers plus a rack-wide memory fence before its grouped GEMM can start. With
  **pull**-dispatch, cross-GPU signalling disappears: a rank issues a load, waits for data,
  proceeds; only one entity signals, so overhead does not grow with EP degree.
- *Numbers* [reported, Cursor, on GB300 NVL72]: push-dispatch signalling **103 µs vs 18 µs
  pull-dispatch (5.8x)**. Pull delivers **up to 29% higher NVLink bandwidth utilisation
  under expert imbalance**, because NVLink has separate lanes per direction and pull splits
  protocol metadata across both. Microbenchmark on one 256×256 BF16 tile (131,072 B
  payload): push = 159.6 KB total, 2.9 KB RX (1.84%) / 155.6 KB TX (99.16%); pull = 172.0 KB
  total, 147.5 KB RX (85.71%) / 24.6 KB TX (14.29%). Push moves ~12.4 KB fewer bytes and is
  still worse.
- *Their final scheme* [verified]: "pull-based forward dispatch, push-based forward combine,
  pull-based backward reverse-combine, and push-based backward reverse-dispatch" — which
  also lets one schedule table be built once and reused for all four operations.
- **Honest caveat**: this was measured at **EP degree 64 across an NVL72 rack**, not at EP8
  inside one node. Our 8xB200 NV18 all-to-all has 7 peers, not 71, so the signalling term
  is ~10x smaller and the measured 5.8x will not transfer at face value. What *should*
  transfer is the direction asymmetry (bidirectional lane utilisation) and the removal of
  the fence. **Measure before committing engineering.**
- **Effect: unknown at EP8; upper bound is the 9.2% of C1 we burn on arrival skew.
  Difficulty: high.**
- **SGLang change** [inferred]: this is a replacement for the DeepEP/NCCL dispatch path, not
  a flag. `github.com/cursor/mixture-of-kittens` is Apache-2.0 and targets SM100/SM103, so
  the transport layer is readable, but it is a *training* megakernel — the dispatch/combine
  primitives transfer, the fused backward does not.
- Evidence: <https://cursor.com/blog/mixture-of-kittens>, <https://github.com/cursor/mixture-of-kittens>

**3. Warp decode for the MoE expert GEMMs at C1–C4.**

- *Mechanism* [verified]: flip the parallelism axis. Instead of grouping tokens per expert
  and running grouped GEMMs, **assign each warp exactly one output value (neuron)** for its
  lifetime. Two kernels, `moe_gate_up_3d_batched` and `moe_down_3d_batched`. In gate/up a
  CTA is 8 warps; each warp owns one intermediate neuron for a (token, routed-expert) pair,
  loads the expert ID, reads the gate and up weight rows, streams the activation vector
  once and reuses it for both projections with **no shared-memory staging**, converts MXFP8
  weights to FP32 on the fly, accumulates both dot products in private registers. In down,
  each warp owns one output dimension for one token, loops over all top-k experts, and folds
  each expert's routing weight into a single running FP32 accumulator. Cross-lane reduction
  is a butterfly via `__shfl_xor_sync` → `shfl.sync.bfly`: one hardware primitive, no L1
  round-trip, no bank conflicts, no barriers.
- *What it eliminates* [verified]: padding each expert's token list to power-of-2 or 128-byte
  boundaries (non-amortisable at batch 1); the scatter of 8 intermediate results to memory;
  the separate combine pass; the activation gather buffer (a full copy of data that already
  exists at batch 1); and the per-expert output buffer — **32 KB per token in BF16** at 8
  experts × 2048 intermediate, allocated, written, read once, discarded. Removing that 32 KB
  frees L2 for the weight rows that actually determine performance.
- *Numbers* [reported, Cursor]: **1.84x end-to-end decode throughput on B200**, flat across
  all context-length buckets (confirming it is a pure generation-time win). **3.95 TB/s
  sustained at B=32 = 58% of a measured 6.8 TB/s copy-kernel peak**; the gap is attributed
  to the random-access pattern expert routing creates. Accuracy *improves*: outputs **1.4x
  closer to a full FP32 reference** because activations never round-trip through MXFP8; min
  cosine similarity > 0.999996, max absolute difference 0.001953.
- **Config gap you must know**: the model is described only as "a Qwen-3 style model on
  NVIDIA B200 GPUs" running on Cursor's internal inference system. **No expert count, hidden
  dim, intermediate dim, top-k, batch-size sweep table or dtype for the baseline is
  published.** The 1.84x is therefore not a number you can port to GLM-5.2 by arithmetic.
- *Their own caveat* [verified]: "Prefill and large-batch inference still benefit from
  expert-centric packing." Warp decode wins "when there isn't enough shared work per expert
  to justify that overhead."
- **Effect: plausibly 8–9% of C1 wall clock if the 1.84x holds at our shape — which is
  unproven. Difficulty: high (new CUDA kernel from a prose description; no code released).**
- **SGLang change** [inferred]: a new low-batch MoE runner backend selected by a batch-size
  threshold, sitting alongside (not replacing) the grouped-GEMM path. The routing-weight
  fold into the down-projection accumulator is the cheapest single piece to try first.
- Evidence: <https://cursor.com/blog/warp-decode>

**4. Fuse RMSNorm and SwiGLU into the GEMM epilogue.**
Dense GEMM is our single largest bucket at 37.1%, and fal published the exact SM100 layout
facts plus a measured number.

- *Layout facts* [verified]: the NVFP4 GEMM uses a **256×256 CTA tile**; the collective
  epilogue walks it as **4 subtiles of 128×128** in fixed order 0→1→2→3 with no going back;
  each subtile is partitioned across 128 threads, giving **FragmentSize = 128**. So at
  `head_dim=128` a whole head lives in one thread's registers and RMSNorm fuses in a single
  `visit()` with no cross-thread shuffle.
- *For head_dim=256* [verified], they wrote a **three-touch "revisit" epilogue**, quoted:
  "1. **Tile 0, pass 0**: this is the first half of the head. We compute its sum-of-squares
  and move on; we can't normalize yet without the second half. 2. **Tile 1, pass 1**: the
  second half of the head arrives… We normalize and store this half. 3. **Tile 0, pass 2**
  (**revisit**): we go back to Tile 0 and scale it with the now-known `rstd`."
- *SwiGLU trick* [verified]: the gate/up pair at columns `n` and `n + N/2` can land in
  different fragments, subtiles or CTAs. So **permute B's columns once during weight
  packing** to produce interleaved `[gate0, up0, gate1, up1, …]`; a custom EVT visitor then
  operates on adjacent elements in the same fragment, and cross-tile synchronisation
  disappears entirely. This is a *weight-packing* change, essentially free at runtime.
- *Measured* [reported, fal]: in the companion EVT post, folding post-GEMM ops into the
  epilogue for a real model deployment saved **166 µs (1.28x faster)**. The end-to-end
  Ideogram V4 result is **2.75 s → 0.44 s (6x) at 1K resolution** — but that bundles FP4
  quantisation, CFG-branch collapse and timestep distillation, so it is not an
  epilogue-fusion number.
- **Effect: removes an HBM round-trip per GEMM. Difficulty: medium–high (CUTLASS EVT), but
  the weight-permutation trick alone is cheap and independent.**
- **SGLang change** [inferred]: two separable pieces. (a) Permute the gate/up columns of the
  MLP weight at load time and adjust the SwiGLU kernel to consume interleaved pairs — no
  epilogue work required, and it makes any future fusion possible. (b) An EVT epilogue on
  the NVFP4 GEMM that emits normalised, already-quantised activations for the next layer,
  which also kills a separate quantise kernel.
- Evidence: <https://blog.fal.ai/serving-sub-second-ideogram-v4-without-quality-loss/>,
  <https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/>

**5. Skip the DSA indexer when sequence length < K, and fuse the high-precision indexer
projections.**
The only public statement anyone has made about serving DSA, and it is about our exact
architecture.

- *Quoted* [verified]: "The indexer must scan every token in the sequence to compute which K
  tokens to select." / "At shorter context lengths, this scanning overhead dominates, and
  the sparse attention has a longer prefill time than full attention." / **"If the sequence
  length is less than K, we can skip the indexer and run standard full attention."** /
  "Although the main multi-head attention can be performed in FP8, other large and important
  projections need to be performed in high precision to preserve quality… The numerical
  tolerance is much lower compared to other parts of the model." / "We overlap and fuse
  operations through the indexer code path to minimize the cost of these high-precision
  projections."
- **No numbers are attached to any of this.** It is a mechanism claim, not a measurement.
- **Effect: small but nearly free; our indexer is 5.8% of C1 at `index_topk_freq=4`. At AA's
  ~10k input it will not fire on prefill, but it fires on short requests and early decode.
  Difficulty: low.**
- **SGLang change** [inferred]: a length guard in front of the indexer that routes to the
  dense MLA path, plus a check that our indexer projections are in the same kernel as
  whatever precedes them rather than standalone launches.
- Evidence: <https://www.baseten.co/blog/how-we-built-the-fastest-glm-5-api/>

### Tier 2 — high value, more speculative

**6. Data-parallel attention instead of TP for the MLA path.** [verified mechanism,
reported numbers]
Doubleword got **5,856 → 12,802 tok/s (2.2x)** on DeepSeek-V4-Flash purely by moving from
TP4 to TP1×DP4 — on **4xGH200**, at concurrency 2048→2752, ISL/OSL 1024/1024, vLLM. Reason:
"With only one shared K/V latent, naive tensor parallelism for MLA has to replicate the KV
cache on each accelerator." DP gives each rank its own cache; the weights replicate instead,
which is cheap when attention weights are small. Side effect they call out: every per-token
kernel (elementwise, FP8 quant, sampling) then runs on 1/N of the batch per rank instead of
the full global batch on every rank. Baseten independently confirms the shape — their
*standard* GLM-5.2 API "uses Attention Data Parallelism (ADP) to improve throughput", and
they drop to "Tensor and Expert Parallelism with configs selected for latency" only on the
Fast tier.
**This is a throughput/cost lever, not a C1 latency lever** — Baseten's own tier split says
so explicitly. Likely the single biggest available move for our C64 objective.
**Difficulty: medium — SGLang supports DP attention.**
**SGLang change** [inferred]: enable DP attention on the cost-optimised deployment only;
keep TP+EP for the latency SKU, exactly mirroring Baseten's two-tier split. Watch
`max-num-seqs` — Doubleword had to halve it (2048 → 1024 *per rank*) and raise
`gpu-memory-utilization` 0.92 → 0.95, and cap CUDA graph capture at 1024.

**7. Run the draft and target as one fused model, and never shard draft-head weights.**
[verified quotes]
Baseten: "At Baseten, we run both the draft model and the target model as one big model. By
doing so, the overhead is essentially zero, other than when you predict wrong draft tokens."
fal hit the mirror-image failure independently: their DSpark Markovian heads "were doing
collective operations because they were sharded across the ranks. We fixed it by replicating
the weights instead" — after which TP=4 exceeded 1000 tok/s. Two independent confirmations of
one rule: **draft-head weights replicate, never shard; the draft forward lives in the same
CUDA graph as the target.**
**Effect: removes per-step host overhead from our 3-1-4 EAGLE. Difficulty: medium.**
**SGLang change** [inferred]: audit whether our EAGLE head is TP-sharded. If it is, replicate
it — the head is small and the collective it introduces is on the critical path of every
draft step. Then confirm draft+verify are inside one graph capture, not two.

**8. Confidence-gated ragged draft depth instead of a fixed 3-1-4 tree.**
Doubleword's model [verified]: expected committed tokens at depth γ for sequence *i* is
`m_i(γ) = 1 + Σ_{d=1..γ} Π_{k=1..d} a_k^i`, the cumulative product of the drafter's own
per-depth confidences. They argue for **per-sequence confidence gating with ragged depths**
rather than a batch-wide constant, and acknowledge engine support is limited. Baseten lists
the same idea as future work, verbatim: "**Dynamic-length speculation**, where the draft
length is adjusted based on speculation confidence on a per-request, per-micro-batch basis."
The Modular handbook notes SGLang already switches γ between pre-captured tiers (default
`[1, 3, 7]`) using pre-captured CUDA graphs, so the switch itself is free.
**The cheap version is available today**: use the existing tier mechanism, drive tier
selection from the drafter's own confidence rather than a static config.
**Effect: medium on both objectives. Difficulty: medium.**
**Artifacts that make this cheap** [verified]: `Doubleword/specdec-calibration` on HuggingFace
(4.56 GB; per-round acceptance banks `accept, acc0..acc{D-1}` and speculator confidence banks
`conf0..conf{D-1}` for Qwen3.6-35B-A3B and DeepSeek-V4-Flash, MTP and DFlash drafters, over
SPEED-Bench and HumanEval, plus MoE expert-routing captures) and `doublewordai/inference-lab`
(MIT, Rust discrete-event simulator with WASM/CLI bindings, **shipped B200/B300/H100/GH200
hardware presets**, ~40 model presets, and explicit support for "analytic or trace-replayed
acceptance, fixed and goodput-adaptive draft policies" plus MoE expert loading in its roofline
model). **We can simulate our own depth policy on a B200 preset before writing engine code.**

**9. Async QKV / all-to-all overlap using functional collectives, with a scale-dependent
policy.**
fal's numbers are on **8xB200, our exact topology**. Setup [verified]: bf16, B=1, H=40,
D=128, fixed clocks, cuDNN attention backend, 15 warmup + 40 timed iterations × 3 repeats,
max-rank median; they report both "pre-attention chunk latency" (QKV projection + pre-SDPA
comms) and end-to-end step latency.

| Variant | Mechanism | Chunk latency | End-to-end |
|---|---|---|---|
| Baseline Ulysses | 3 local GEMMs → 3 all-to-alls → SDPA, serialised | — | — |
| **Async Ulysses** (ByteDance VeOmni) | Compute Q → launch Q a2a → compute K while Q in flight → launch K a2a → compute V → launch V a2a → wait → SDPA | **−23% to −25% at 2/4/8 GPUs** | ~−3% |
| Async + PyTorch Symmetric Memory | Same schedule, transfers routed through the **copy engine** so NCCL kernels stop stealing SM cycles | Better at 2 and 4 GPUs, **regresses at 8** | — |
| **Fused QKV** | One packed local weight shard into `torch.ops.symm_mem.fused_all_gather_matmul` | **−37.3% @2, −33.4% @4, only −4.6% @8** | −5.0% / −4.8% / −0.3% |

The single most important sentence in the post for us [verified quote]:
> "NCCL collectives are implemented as GPU kernels that issue loads/stores and reduction
> work, so they compete with GEMMs for SM cycles. In practice, this means timeline overlap
> does not always translate into throughput overlap."

Their conclusion: **overlap is the robust high-scale default; fusion wins at low/mid scale**,
and a lightweight runtime policy should choose based on sequence length, world size and
interconnect. **Caveat they state themselves**: this was dense attention, not MoE dispatch.
They say communication-heavy workloads like MoE routing "should benefit even more" but **did
not measure it**.
**Effect: relevant to the ~10.4% of C1 that is collectives-minus-skew. Difficulty: medium.**
**SGLang change** [inferred]: the fusion path is a dead end for us — at 8 GPUs it delivered
−4.6% chunk and −0.3% end-to-end. The *async schedule* is the transferable part: reorder the
QKV projections so each one's all-to-all is in flight while the next projection computes.

### Tier 3 — worth knowing, lower priority for us

**10. Tokenisation as a TTFT line item — and there is now an open-source option.**
Baseten's argument [verified]: at 1M-token agentic inputs with high prefix-cache hit rates,
**tokenisation happens whether or not the prefill is a cache hit**, so it becomes material to
TTFT. Their `Basetenkenizer` is **not released**. But Doubleword's `gigatoken` is
(`github.com/marcelroed/gigatoken`) and they published its internals: SIMD 64-byte ASCII
classification, memoisation of the token-ID sequence for the exact bytes of each ordinary
pretoken (32-byte entries in 64-byte probe buckets), a dense 16 MiB grid for hot pair
lookups, and parallelisation that "cuts at proven pretoken boundaries, which BPE cannot
cross." Measured: **1,039 MB/s single-threaded, 8,792 MB/s at 16 threads (8.5x scaling)**;
pretokenizer alone 2,460–2,600 MB/s vs 983 MB/s scalar; and **37.6x on a million-token,
1,024-segment workload (159 ms → 4.24 ms)** vs their own multithreaded Rust tokeniser core.
Baseten's own honest cross-comparison: gigatoken is fastest at 10k tokens, the crossover is
before 200k, Baseten leads by 1.41x at 1M, and on gigatoken's home turf (offline files read
in Rust) gigatoken does 2.68 GiB/s vs Baseten's 118 MiB/s Python-string path.
**At our 189 ms TTFT and 10k input this is single-digit milliseconds** [inferred] — but it is
free, open-source and carries no quality risk.

**11. Prefill-priority scheduling.** [verified quote] "Our runtime prioritizes prefill steps
over decode, improving perceived speed with lower TTFT." Cheap scheduler change; trades ITL
for TTFT. Relevant only if we decide TTFT is the constraint.

**12. Turn speculation off above a batch-size threshold.** [verified quote] "Speculative
decoding shines at low batch sizes when there's spare GPU compute. At high batch sizes, it's
dynamically turned off because verification becomes costly under compute saturation." Given
our per-stream speed falls 4.7x from C1 to C16, there is a crossover we should **find rather
than assume**. Corroborated by the Modular handbook's H100 Llama-3.3-70B study: at TP=1 spec
dec gave ~2x TPOT improvement but total throughput **plateaued earlier, around 20–30
concurrent requests**, than baseline.

**13. Measure our own sustained B200 rate; do not trust the datasheet.** [verified]
Doubleword found sustained BF16 on Isambard tops out at **583 TF/s, not the 990 TF/s
datasheet number**, because clocks drop once the GPU draws 530–570 W — a *software power cap*,
not thermal. That moved the ridge point from ~250 to ~146 FLOP/byte and the critical batch
size to ~1,300, which is what made their MoE decode compute-bound at all. Any roofline we
build on datasheet numbers is wrong by a similar factor.

**14. NUMA binding.** [verified quote] "If we don't bind the engines to their NUMA node, then
comms between the host and device go over the interconnect and we lose throughput." We have
2 NUMA nodes. No number attached, but it is a one-line check.

**15. KV cache compaction, for the C64 objective only.** Baseten's research arm published
**Still** [verified]: "a small per-layer Perceiver trained once against a frozen base model
that produces compact keys and values in a single forward pass," achieving **8x to 200x
compression** on Qwen and Gemma at 8k–128k context, exceeding the strongest baseline by
**8–22 points on the RULER grid**. Amortised — no per-context optimisation — so it can be
applied iteratively. Paper: arXiv 2606.07878. **Neither code nor weights are stated as
released**, and no serving latency numbers are given; this is research, not a shippable
artifact. Related and more concrete: *Repeated KV cache for long-running agents* (Attention
Matching, "50× compression with minimal quality loss, in seconds", Qwen3-4B on LongHealth,
with the honest finding that **re-compressing an already-compressed cache costs an extra
4–16% accuracy**).

### Explicitly *not* recommended

- **Do not expect PD disaggregation to help single-stream latency.** The evidence is
  contradictory and none of it is configured. Baseten reports "2x higher tokens per second on
  disaggregated inference" for GLM-5.2 in "head-to-head benchmarks between aggregated and
  disaggregated deployments" — with no worker ratio, transport, batch size or baseline config
  published. At Dynamo Day they claim "up to 6x higher TPS per GPU is realistic", also
  unconfigured, and promised benchmarks that I could not find. Meanwhile the Modular handbook
  reports "If your workload is too small, or your GPU setup isn't tuned for this approach,
  performance can drop (by 20-30% in our tests)", and notes high prefix-cache hit rates favour
  local prefill. Doubleword's analysis is that disaggregation is right once three conditions
  hold, but condition one is *scale* — DeepSeek V3's published shape was **EP32 prefill units
  and EP320 decode units**, thousands of GPUs before rounding stops hurting.
  **On a single 8xB200 node with a C1 objective, this is a cost lever at best.**
  *One salvageable detail* [verified]: Baseten mentions "a kernel to transpose KV blocks
  between layouts" as part of their NIXL-based prefill→decode transfer. If we ever do
  disaggregate, the layout mismatch between prefill and decode KV is a real cost we should
  budget for up front.
- **Do not chase the fusion path for collectives at 8 GPUs.** fal measured
  `fused_all_gather_matmul` at −37.3% chunk latency at 2 GPUs and **−4.6% at 8**, because
  messages get too small to amortise fixed buffering/signalling overhead. Symmetric-memory
  copy-engine transport had the same shape: wins at 2–4, regresses at 8.
- **Do not trust eval loss on a quantisation conversion.** fal's negative result: "Loss went
  down, images did not get better… diffusion training loss does not predict image quality."
  The fix was quantisation-aware distillation where "gradients flow back through the FP4
  rounding via a straight-through estimator, so the weights actively adapt to the quantization
  noise they produce." Different modality, and LLM error does not compound the way diffusion
  latent error does — but the failure mode (frozen quantised student, no gradient through the
  quantiser, loss looks fine) is generic.
- **Do not expect INT4→NVFP4 to recover quality.** Baseten, verbatim: "While NVFP4 is overall
  a more precise data format than INT4, switching does not enhance quality as the INT4
  quantization was performed during training. There is no way to recover the information that
  was lost to compression during training."

---

## Baseten

**What they run.** A proprietary runtime (the "Baseten Inference Stack" / "Baseten Inference
Runtime") that sits *alongside* rather than replaces the open-source engines. Their consistent
public line: benchmark TensorRT-LLM, SGLang and vLLM per model, pick the winner, then layer
proprietary kernels, a "Baseten Speculation Engine", and NVIDIA Dynamo for orchestration on
top. [verified]

> "We like Dynamo because it's framework-agnostic, so it doesn't stop us from choosing the
> best inference engine (TensorRT-LLM/SGLang/vLLM) for a specific model and use case."
> — *The Baseten Inference Stack at NVIDIA Dynamo Day*, Mar 16 2026 [verified]

For Kimi K3 they configured vLLM, SGLang *and* their in-house engine in parallel. **Tri Dao is
a listed author** on the GLM-5 post (with Philip Kiely and Madison Kanna), the GLM-5.2 launch
post (with Alex Korte, Magdy Saleh, Anant Desai, Bryce Dubayah, Abu Qader, Philip Kiely) and
the Kimi K2.5 post (first author, with Michael Feil, Abu Qader, Philip Kiely). [verified]

### Techniques with mechanism + evidence

**1. DSA indexer kernels.** See Tier 1 item 5 above — full quotes there. No numbers attached.
<https://www.baseten.co/blog/how-we-built-the-fastest-glm-5-api/> [verified]

**2. Fused draft+target speculation ("one big model").** See Tier 2 item 7. Same post.
[verified]

**3. Suffix Automaton MTP accelerator — open source, merged upstream.** See Tier 1 item 1.
Implementation detail worth stealing beyond the algorithm: the automaton is a **plain-old-data
struct with a header-only core that compiles for both C++ and CUDA**, so host↔device transfer
is zero-conversion; CUDA-graph compatibility comes from specialising a smart pointer's `memcpy`
and the `extend()` invocation onto the active torch stream. Three Python entry points:
`add_request(request_id, prompt)`, `prepare(request_ids)`,
`extend(draft_tokens_out, accepted_tokens_in)`. Apache-2.0.
<https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/>
(May 5 2026, Mahmoud Hassan + Model Performance Team), <https://github.com/basetenlabs/sa_spec>
[verified]

**4. KV-cache-aware routing (built on NVIDIA Dynamo).**
Mechanism [verified]: "Hashing incoming requests and organizing them in a Radix Tree enables
scalable tracking of cache locations in distributed environments." On arrival, "the LLM Aware
Router calculates an overlap score between the request and the KV cache blocks already active
across all GPUs in the cluster," then routes on a blend of overlap and current load. Baseten
adds custom logic mixing KV routing with round-robin per model.
Numbers [reported]: Qwen3 Coder 480B, **~50k input / ~1k output**, 4 replicas, **89% cache hit
rate**, controlled high-load test: **50% lower TTFT, 34% lower TPOT, 61% more RPS, 62% higher
output TPS**. Shadowing real OpenRouter production traffic: **48% lower P95, 49% lower P99**.
*Baseline is unstated in the post* — "vs random routing" was in the first pass of this file and
I could not confirm it; treat the comparator as unspecified.
<https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/>
(Mar 16 2026) [verified]

**5. Live draft-model training from the serving path.**
Mechanism [verified]: hidden states are extracted from live inference and streamed to training
nodes with no offline storage. Reason given: a single sample on Kimi K2 can exceed 2 GB and
full draft training needs millions of them. Constraints they solved that we would also hit:
added GPU and pinned memory scales with **`max_num_tokens_per_iter`, not sequence length**,
preserving long-context headroom; **CUDA-IPC double buffering into pageable memory via `mmap`**;
**UCXX** for async RDMA between nodes; **Trio** for structured concurrency and failure recovery.
Numbers [reported]: **median 20% increase in accept rate; 100%+ on some traffic patterns.**
No code released.
<https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/> (Jun 25 2026)
[verified]

**6. EAGLE-3 training recipe — the most concrete public one.** [all verified]

| Parameter | Baseten's recommendation |
|---|---|
| TTT-length (test-time-training length) | **7–9** |
| `num_draft_tokens` at inference | **3–4** |
| Optimizer | AdamW at all scales |
| LR, ~3–7B target | 1e-4 |
| LR, ~7–20B target | 5e-5 |
| LR, 20B+ target | 2e-5 |
| Sampling | temp=0 best; "roughly a 15–25% reduction in speedup at temperature=1 vs. temperature=0" |
| Dataset size, generic task | 200k–300k samples (small), ~500k (large) |
| Dataset size, specialised task | ~100k samples |
| Tokens per sample | 1k–2k total |
| Training accuracy target | plateau at **70–80%** |
| Observed production speedup | **1.5–2.5x**, vs 4–6x claimed in papers |

The golden rule, verbatim: "The EAGLE head must predict tokens as the target model would
generate them. If you train on outputs from a different model (or from human-written text),
the draft head's distribution will be misaligned." Frameworks they say all work:
NVIDIA Model-Optimizer, `sgl-project/SpecForge`, `torchspec-project/TorchSpec`.
**Our 3-1-4 config sits inside their recommended `num_draft_tokens` band** — worth checking
our TTT-length against 7–9.
<https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/>
[verified]

**7. DFlash — block-diffusion speculation (ICML 2026, code released).**
Mechanism [verified]: instead of drafting autoregressively, a lightweight block-diffusion model
with **bidirectional attention** predicts γ tokens *in a single forward pass* (8–16 tokens).
Inputs: fused context features from **5–6 evenly-spaced layers of the target model**. Training
weights the *i*th token by `exp(−(i−1)/γ)`; embedding and LM head frozen.
Numbers [reported]: Qwen3-8B, **single B200, concurrency 16**, GSM8k: Baseten's backend
**654 TPS / 1.2 s latency (3x baseline)**; vLLM's DFlash ~594 TPS / 1.6 s; EAGLE ~595 TPS /
1.5 s. So the "10% faster than vLLM" and "25% lower latency than vLLM" framings both check out
arithmetically. Across MATH-500 and Nemotron they claim 10–30% over vLLM.
Paper abstract [verified]: "DFlash achieves over 6x lossless acceleration across a range of
models and tasks, delivering up to 2.5x higher speedup than the state-of-the-art speculative
decoding method EAGLE-3." Authors Jian Chen, Yesheng Liang, Zhijian Liu.
<https://www.baseten.co/blog/dflash-faster-llm-inference/> (May 8 2026),
<https://arxiv.org/abs/2602.06036>, <https://github.com/z-lab/dflash> [verified]

**8. Basetenkenizer.** [verified] Optimisations in their stated order of contribution:
specialised pre-tokenization scanners; a **stack-resident BPE merge tier**; multi-core
semantics (routing identical pre-tokens to the same core; scheduling each 400,000-character
chunk on a separate core); native typed segments and safe chunking; zero-copy NumPy ownership
transfer; smart-pointer PyO3 bindings. Numbers: **>6x faster than tiktoken at short sequences,
18x at million-token sequences, exact token-ID parity**, median of end-to-end encodes on
**52 pinned vCPUs (Intel Xeon Platinum 8480+)**. Built on `fastokens`, `gigatoken`, MiniJinja.
**Not open source** — they released only a Kimi K3 `tokenizer.json` on HuggingFace. Use
`gigatoken` instead (Tier 3 item 10).
<https://www.baseten.co/blog/making-kimi-k3-tokenization-18x-faster-for-million-token-agentic-workloads/>
(Aug 5 2026) [verified]

**9. Negative results and things that broke (TensorRT-LLM spec dec, May 16 2025).**
**Config that the first pass omitted and that changes how you read this**: this is **Hopper**,
not Blackwell — Qwen 2.5 Coder 14B Instruct target + 0.5B draft on a single **H100**, and
Llama 3.1 70B Instruct target + 8B draft on **4xH100**. Still the best public list of two-model
speculation failure modes:
- Draft and target on the same GPU "had a tendency to fight for resources… the model would run
  at half speed when they were in contention." Fix: a synchronised async loop where only one
  runs at a time, which also let them schedule target inference ahead of draft for the first
  token to protect TTFT.
- TensorRT-LLM **was not batching target-model requests at all**.
- **The request scheduler did not account for KV-cache reuse when scheduling with chunked
  prefill**, so "for a load where we might expect a batch size of 10, we were only seeing 2 or 3
  requests processed at once." Patched and upstreamed.
- **One Llama 3.1 benchmark got *worse* p50 latency with spec dec on.** They published it.
- TTFT is "slightly worse despite our optimization efforts."
<https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm/>
[verified]

**10. Baseten Delivery Network (cold starts).** [verified] Three tiers: node-local NVMe; an
in-cluster peer cache on a **consistent hash ring**; mirrored origin blob storage. The hash ring
makes chunk ownership deterministic, so nearly all nodes pull their assigned chunks from origin
concurrently and the node that needs the file fans the rest in from peers. Node-level
**single-flight** coalescing on scale events. Weight delivery happens "before the model
container starts," so transfer sits outside the billable GPU window. Numbers: cold starts
"2-3x faster and more reliable"; **>2 GB/s** onto H100 nodes; mirroring pipeline 1–5 GB/s.
Worked example: 50 replicas of a 140 GB model across 20 cold nodes → origin bandwidth is 1x
model size, not 50x.
<https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/>
(Apr 9 2026) [verified]

**11. Research arm: KV cache compaction.** See Tier 3 item 15. Full research index verified at
<https://www.baseten.co/research/>: *Can a Language Model Learn Facts Continually in Its
Weights?* (Jul 14 2026), *Post-Training Science for Supervised Fine-Tuning* (Jun 18 2026),
*Still: Amortized KV Cache Compaction in a Single Forward Pass* (Jun 10 2026, arXiv 2606.07878),
*Towards infinite context windows: neural KV cache compaction* (Mar 13 2026), *Dense, on-policy,
or both?* (Mar 5 2026), *Repeated KV cache for long-running agents* (Mar 5 2026), *Distillation
without the dark* (Feb 3 2026). [verified]

### Published numbers

All AA figures are Baseten's reading of Artificial Analysis on the stated date. **None of these
is directly comparable to our 365 tok/s single-stream number.** Read the asymmetry column.

| Model | Hardware (as published) | Quant | Speculation | Parallelism (published) | Reported | Date | Config asymmetry vs us |
|---|---|---|---|---|---|---|---|
| GLM-5.2 (launch) | "NVIDIA Blackwell GPUs" — **no count** | NVFP4 from FP8 via ModelOpt | MTP heads | **not published** | **280+ tok/s**, TTFT 800 ms, 7.9 s to first *answer* token | Jun 22–23 2026 | GPU count unknown; 40B active params stated |
| GLM-5.2 (standard, retuned) | **B200**, count not published | "improved NVFP4 weights" | "updated speculative decoding profile" | **ADP** | **601 tok/s** (prior peak 280, prior avg ~100) | Jul 25 2026 | **~10k in / ~1k out**, not AA's ≥1500 out |
| GLM-5.2 Fast | same weights | same | — | **TP + EP, configs selected for latency; substantial reduction in max batch size** | no separate tok/s figure published | Jul 24 2026 | tier launch only |
| GLM-5 | "Blackwell" | NVFP4 | MTP, fused draft+target | not published | **186+ tok/s**, "lowest time to first token" | Mar 9 2026 | no TTFT number given |
| Kimi K2.5 (1T) | "NVIDIA Blackwell GPUs (B200s)" | INT4 → BF16 on the fly → NVFP4 (ModelOpt recalibration) | custom **EAGLE-3, ~1B params**, trained on hidden states from synthetic code/scientific/multilingual queries via DeepSpeed | **not published** | **340+ tok/s** | **Mar 9 2026** | weights already 4-bit from QAT |
| Kimi K2 Thinking (1T) | **single 8xB200 node** | INT4 → BF16 (`compressed-tensors`, "a few hours") → NVFP4 (ModelOpt) | **none at launch** | "a mixture of Tensor Parallelism and Expert Parallelism" | **140+ tok/s, 300 ms TTFT** | Dec 1 2025 | no speculation at all |
| GPT-OSS 120B | **8xB200** | none — native MXFP4 kept | **EAGLE-3** | **TP8 EP1** | ~400 → **650+ tok/s (+60%)** | Oct 24 2025 | 120B dense-ish MoE, no quant step |
| GPT-OSS 120B (earlier post) | H100 and B200 | native MXFP4 | — | TP across 4–8 GPUs | **500+ tok/s** at launch | Nov 10 2025 | **inconsistent with the ~400 baseline above** — see note |
| Qwen3 Coder 480B A35B | multi-replica, GPU type not published | — | — | 4 replicas, 89% KV hit rate | 50%↓TTFT, 34%↓TPOT, 61%↑RPS, 62%↑TPS from KV routing | Mar 16 2026 | ~50k in / ~1k out; comparator unstated |
| Qwen3-8B (DFlash) | **1x B200, concurrency 16** | — | DFlash | — | 654 TPS GSM8k (3x), 1.2 s mean latency | May 8 2026 | 8B dense model |
| Kimi K3 (2.8T) | **GB300 NVL72**; "GB300s come in nodes of four"; **8 GPUs** to fit weights; **9 replicas per rack** | native **MXFP4 weights, MXFP8 activations** | candidates only | TP and EP **across nodes** | **no tok/s published** | Jul 27 2026 | weights >1.4 TB |
| DeepSeek V4 Pro 0813 (1.7T) | **not published** | native MXFP4 | ships with a **DSpark speculator** | "parallelism, KV cache allocation, prefill-decode worker ratio" — values not published | **no tok/s published** | Aug 13 2026 | nothing usable |

**Note on the GPT-OSS inconsistency**: the Oct 24 2025 post reports ~400 → 650+ tok/s (+60%),
while the Nov 10 2025 post describes "500+ tokens per second" as the launch-day peak. Both are
Baseten. I could not reconcile them; treat the 650 figure as the only one with a stated
before/after pair and hardware (8xB200, TP8 EP1). [verified both, unreconciled]

### Open-source artifacts and what is actually usable

| Artifact | Usable? |
|---|---|
| `github.com/basetenlabs/sa_spec` + TensorRT-LLM PR #11434 | **Yes, directly.** Apache-2.0, header-only POD C++/CUDA core, CUDA-graph safe, `SA_SPEC_THRESH` env-tunable. The single most portable thing in this file. |
| `github.com/z-lab/dflash` (Baseten co-authored; ICML 2026) | Yes — reference implementation. Baseten's own backend is closed and claims to beat both vLLM's and SGLang's DFlash implementations. |
| Upstream PRs to TensorRT-LLM and Dynamo | Merged, readable. |
| `Basetenkenizer` | **No.** Described in detail, not released. |
| Baseten Inference Runtime / Speculation Engine | **No.** Closed. |
| *Still* KV compaction | **No code or weights stated as released.** Paper only (arXiv 2606.07878). |
| Engine-builder config surface (docs) | Useful as a **checklist of knobs**, verified against the live docs: `quantization_type` ∈ {`no_quant`, `fp8`, `fp8_kv`, `fp4`, `fp4_kv`, `fp4_mlp_only`}; `speculative_decoding_mode` — **only `LOOKAHEAD_DECODING` is documented**; `enable_b10_lookahead` (their own algorithm); `lookahead_windows_size` / `lookahead_ngram_size` / `lookahead_verification_set_size` (min 1 each); `kv_cache_free_gpu_mem_fraction` (default 0.9); `kv_cache_host_memory_bytes` (set high to enable GPU→host KV offload); `batch_scheduler_policy` ∈ {`max_utilization`, `guaranteed_no_evict`}, default `guaranteed_no_evict`; `paged_kv_cache`; `use_paged_context_fmha`; `use_fp8_context_fmha` (requires `fp8_kv` or `fp4_kv`). Documented caveat: **"Lookahead works best with batch sizes under 32."** |

### What they say about Blackwell/B200 specifically

- **NVFP4 over everything else**, because of its dual scale factors and "deep support in
  Blackwell Tensor Cores." [verified]
- **INT4 → NVFP4 has no direct path.** "Kimi K2 Thinking is optimized for previous-generation
  Hopper GPUs. Blackwell GPUs are not available in China due to export restrictions, so labs
  like Moonshot AI (the makers of Kimi) target Hopper instead." Their route is INT4 → BF16
  (`compressed-tensors`, hours of compute) → NVFP4 (ModelOpt). And the honest caveat quoted in
  the "not recommended" list above: the information lost to QAT is not recoverable. [verified]
- **Native MXFP4 is left alone.** GPT-OSS 120B, Kimi K3 and DeepSeek V4 Pro were all served
  without requantisation. [verified]
- **TP8 EP1 for lowest latency on 8xB200** for GPT-OSS 120B — the only parallelism config they
  publish for a Blackwell deployment. Reasoning, verbatim: "Tensor Parallelism offered better
  latency, while Expert Parallelism offered better system throughput." [verified]
- **On GB300 NVL72**: nodes are 4 GPUs, 8 are needed to fit Kimi K3's >1.4 TB of MXFP4 weights,
  and "the GB300 NVL72 system has a sufficiently fast interconnect between nodes that we can run
  inference with Tensor Parallelism and Expert Parallelism across nodes." Each rack hosts 9
  replicas. [verified]
- **Their configuration sweep list for a new model** is a useful checklist: TP, EP, **ADP**,
  batch sizing, speculative decoder draft lengths, **linear layer caching intervals**, routing
  parameters, engine settings. [verified]
- **Their benchmarking practice** [verified]: they reproduced SemiAnalysis's InferenceMAX on
  **4xB200 at TP4 EP1** for GPT-OSS-120B at 1k/1k using the exact image
  `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2`, ran the benchmark **server-side against
  localhost** — "This isolates the model's inference performance from network variability" —
  and used `/completions` rather than `/chat-completions` plus `ignore_EOS` to reach target
  token counts. **This is the practice to copy if we want an internal number that is not
  polluted by our egress path.**
- **B200 vs H200/H100 platform claims** (customer-observed, vendor-reported, Dec 11 2025):
  "5x higher throughput for high-traffic endpoints"; "More than 50% lower cost per token with
  throughput-optimized deployments"; "Up to 38% lower latency for serving the largest LLMs like
  DeepSeek-R1". **Only the last names a model. None names a config.** Marketing.

---

## Cursor (Anysphere)

Not an inference vendor. Included because they publish the most reimplementable Blackwell MoE
kernel work anywhere.

### Warp decode — MoE inference at low batch on B200

Full mechanism, numbers and caveats are in **Tier 1 item 3** above.
Read the config gap there: model is "a Qwen-3 style model", no shapes, no baseline dtype, no
batch sweep. <https://cursor.com/blog/warp-decode> [verified]

### MXFP8 MoE training kernels — Blackwell numerics and `tcgen05` mechanics

Context [verified]: profiling on B200 showed the MoE layer (MegaBlocks) at "nearly 53% of
forward-pass time and 27% of backward-pass time." They rewrote it "with zero dependencies on any
CUDA libraries" using "pure, good old CUDA and PTX, with a few bits of ThunderKittens sprinkled
in": **3.5x MoE layer speedup on both passes, 1.5x end-to-end on Blackwell, 2x vs their original
Hopper setup.**

The quantisation recipe, stated precisely enough to reimplement [verified]:
```
S   = cast_to_fp8e8m0( absmax(V) / 448 )   # rounds UP to nearest power of 2, min-clamp 2^-127
Q_i = cast_to_fp8e4m3( V_i / S )
```
E4M3 elements, E8M0 scale, **block size 32**.

`tcgen05` constraints they enumerate — all of which bite on inference too [verified]:
- `tcgen05.mma` needs **only a single thread** to launch asynchronously, unlike Hopper's `wgmma`
  which needs a whole warpgroup. The classic 256-thread producer/consumer pattern is wrong on
  Blackwell.
- **2-CTA mode gives "about a 15~20% speedup" for MXFP8 matmuls** vs non-clustered.
- Accumulation happens in **TMEM, not registers**. Custom arithmetic on accumulators requires
  TMEM → registers → CUDA cores → TMEM. This is the central reason block-scaled hardware formats
  beat software dequantisation on Blackwell.
- Scale factors must reach TMEM via **HBM →(`cp.async.bulk`)→ SMEM →(`tcgen05.cp`)→ TMEM**, in
  the layout `tcgen05.mma` expects. There is no direct HBM→TMEM path.
- Budget: **128×512 TMEM and 227 KB SMEM per threadblock.**

**Expert-wise supergrouping for L2** [verified]. Standard supergrouping is "a heuristic from
ThunderKittens kernels that maximizes L2 reuse by ensuring that the region of the output matrix
computed by all 148 SMs at any given time is as square as possible." Their enhancement: apply it
**per expert**, "considering only the submatrix belonging to the current expert rather than the
entire output matrix." Result: **~2,650 TFLOP/s grouped MXFP8, only 4% below non-grouped**. They
note "inefficient HBM access patterns could reduce performance by nearly 50% in grouped matrix
multiplication kernels."

vs **DeepGEMM** (the only Blackwell alternative), average latency in production training
[verified]: grouped Fprop/Dgrad **0.43 ms vs 0.67 ms**; grouped Wgrad **0.65 ms vs 0.71 ms**.

Their MXFP8 quantiser: **6.2+ TB/s while emitting scales in `tcgen05` layout**, vs NVIDIA
TransformerEngine 4.4 TB/s and PyTorch TorchAO 4.5 TB/s (both including reshape). Biggest wins:
"removing TMA swizzling in favor of a manual swizzling pattern to reduce intra-warp overhead,
relying on the warp scheduler and inter-threadblock asynchrony rather than manual
intra-threadblock overlapping, and minimizing SMEM and register usage to increase SM occupancy."
[verified]

| Config | MoE fwd | MoE bwd | End-to-end |
|---|---|---|---|
| Hopper BF16 | 32.36 ms | 63.24 ms | 12k TPS/GPU |
| Blackwell BF16 (direct port) | 25.96 ms | 59.17 ms | 16k TPS/GPU |
| Blackwell MXFP8 | **9.45 ms** | **17.04 ms** | **24k TPS/GPU** |

**These are training numbers.** The `tcgen05` constraints and the quantiser transfer to
inference; the fwd/bwd table does not. <https://cursor.com/blog/kernels> [verified]

### Mixture-of-Kittens — the MoE all-to-all findings

A **training** megakernel for NVL72, open-sourced at `github.com/cursor/mixture-of-kittens`
(Apache-2.0; targets SM100/SM103, i.e. GB200 NVL72 or GB300 NVL72; Python 3.12+, PyTorch 2.10+,
CUDA 13.0+). The *communication* findings are the transferable part. [verified]

Push vs pull, the signalling result, and the four-way direction scheme are in **Tier 1 item 2**.
Additional transferable details, all [verified]:

**Granularity has a closed-form heuristic.** Neither extreme is right: too fine (Comet-style,
256 tokens per MMA) starves the tensor cores with barrier waits; too coarse (DeepEP-style) makes
them wait for the first and last rounds. Target at least two full waves per grouped GEMM:
```
T ≥ 2·C·128·256 / min(2I, H)
```
where T is minibatch size, C is SM count, H hidden dim, I expert intermediate dim. For Kimi 2.5
(H=7168, I=2048, C=148 on Blackwell) this gives **T ≥ 2368**; their measured sweep was
**T=512: 5.981 ms, T=2560: 3.425 ms**. *(The formula as printed here reproduces 2368 from their
stated inputs; a naive scrape of the post inverts it.)*

**Inter-SM overlap, not streams.** They partition SMs in software into comms SMs and comp SMs
signalling through a local counter, because with TMA they saturate NVLink with "less than a
third of the SMs," and because "multiple streams with green contexts [were] unreliable at
partitioning SMs exactly as intended, while software partitioning gives exact allocation
guarantees." **This is directly relevant to our arrival-skew problem** — it is the general fix
for "my collective kernel and my GEMM are fighting for SMs."

**Other details:** a **ring token buffer ("macrobatch")** — "a fixed-size ring buffer of a few
hundred megabytes" cycled at minibatch granularity — avoids both token dropping and CPU-GPU sync
under dynamic expert loads; dispatch-combine interleaving so a buffer slot refills as soon as it
drains; **Cluster Launch Control (CLC)**, "a hardware-native work-stealing feature introduced in
Blackwell," so the megakernel can yield to higher-priority streams instead of serialising them
behind it; in MXFP8 mode **the shared expert stays in BF16** for stability; router-weight
gradients computed **SonicMoE-style** from the inner product of the SwiGLU activation and the
down-projection dgrad.

**The MoE shape table** (this is where GLM-5.2 appears, *not* in the warp-decode post):

| Model | E | H | I | top-k |
|---|---|---|---|---|
| Kimi K2.7 Code | 384 | 7168 | 2048 | 8 |
| **GLM-5.2** | **256** | **6144** | **2048** | **8** |
| Qwen3.5-397B-A17B | 512 | 4096 | 1024 | 10 |
| DeepSeek-V4-Pro | 384 | 7168 | 3072 | 6 |

Numbers [reported]: EP degree 64, 2,048 tokens/GPU before routing, vs the best of NCCL+PyTorch /
DeepEP variants / HybridEP+Megatron: **2.37x MXFP8 forward, 1.78x MXFP8 backward, 1.92x BF16
forward, 1.58x BF16 backward.** End-to-end on 512 GPUs across several GB300 NVL72 racks:
**760.9 → 1,070.2 TPS/GPU (1.41x)** vs their previous DeepEP-based production stack.
<https://cursor.com/blog/mixture-of-kittens>, <https://github.com/cursor/mixture-of-kittens>

---

## fal

**What they run.** SGLang for both diffusion and their LLM prompt-expander path, plus their own
CUTLASS/CuTeDSL/Triton kernels. Everything below is measured on **B200 or 8xB200**.

### Communication–computation overlap on 8xB200

Full table, setup and the "timeline overlap ≠ throughput overlap" quote are in **Tier 2 item 9**.
Two things worth repeating: (a) this was measured on **dense attention**, and they explicitly say
MoE routing "should benefit even more" but **did not measure it**; (b) they note the 8xB200 node's
all-to-all is already very strong, so slower interconnects have more slack to hide — which cuts
against us, since we *are* the strong-interconnect case.
<https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/> [verified]

### MXFP8 quantiser at 6+ TB/s on B200 (Jan 27 2026)

Written in CuTeDSL. The point is that it writes scale factors **directly into the packed
`tcgen05` layout** so a block-scaled GEMM consumes them with no repack; TransformerEngine returns
the same logical bytes in a dense `(M, K/32)` layout that must then be reordered.

Effective bandwidth is defined as `(2·M·K + 1·M·K + 1·M·(K/32)) / t`. Progression [verified]:

1. **Split the grid over K.** Their first version mapped a CTA to a row block and looped over all
   of K; NCU showed Stall Wait dominating. A 2D grid with `rows_per_cta=8, k_tile=256` on a
   16384×16384 input gives **2048 × 64 = 131,072 CTAs instead of 2,048 — 64x more**. This alone
   took them **~1.3 TB/s → ~3.3 TB/s**.
2. **SIMT `cp.async` plateaued at ~3.4–3.6 TB/s** — bound by instructions per byte and
   copy/sync bookkeeping, not DRAM.
3. **Single-bulk-load TMA per CTA tile.** One bulk transaction for the whole `(8, 256)` region,
   wait on the mbarrier, consume. The trap they name: over-pipelining TMA with repeated
   per-subtile barriers, whose cost dominates near saturation.
4. **The unexpected bottleneck: scale-factor stores.** NCU's SASS correlation showed `STG.E.64`
   for the quantised output (good) and **`STG.E.U8`** for the scales (bad) — byte-at-a-time
   stores spray partially-used 32 B sectors. Fix: pack four scale bytes into one 32-bit store
   when 4-byte aligned, fall back otherwise.
5. **Instruction hygiene: 97.9M → 78.9M instructions.** Reciprocal-multiply instead of `fdiv`;
   fuse scale math into an FMA; rely on the pack instruction's built-in saturation instead of
   explicit clamps; packed FP32x2 ops; **compute absmax in the integer domain**.
6. **32-lane CTAs beat 64 and 128** — more CTAs in flight, better latency hiding.

**Negative result, published** [verified]: "Aggressive shared-memory swizzling to reduce bank
conflicts" gave no net gain — the extra index math ate the reduction.
<https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/>

### Inline PTX in Triton, including an NVFP4 quantiser (Feb 10 2026)

`triton.language.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)` injects
element-wise PTX without leaving Triton. Using **`cvt.rn.satfinite.e2m1x2.f32`** (two FP32s →
two packed e2m1 values in one byte) with `pack=4`, they wrote an **NVFP4 quantiser in under 100
lines of Triton that reaches ~7 TB/s on B200**, approaching FlashInfer's and TensorRT-LLM's
2,000+ line CUDA kernels. Caveats they list honestly: you own the correctness of register
constraints, packing factors and dtypes; mis-specified constraints silently produce wrong
results; the kernel becomes architecture-specific; element-wise semantics only — no shared
memory, no sync, no warp-level control. Their conclusion: "write the bulk of the kernel in clean
Triton, and inject PTX only where it truly matters."
<https://blog.fal.ai/instruction-level-control-with-inline-elementwise-asm-in-triton/> [verified]

### Epilogue fusion on SM100 (Feb 3 2026 companion + Ideogram post)

Layout facts and the SwiGLU column-permutation trick are in **Tier 1 item 4**. The companion post
adds the concrete CUTLASS EVT vocabulary [verified]: leaf `Sm90AccFetch` supplies raw accumulator
fragments; `Sm90Compute<Op, …>` applies transformations; `Sm90ScalarBroadcast` / `Sm90ColBroadcast`
supply per-thread/per-row/per-column values; these compose as `Sm90EVT<NodeOp, InputSource>`, e.g.
`Sm90EVT<NodeMultiply, Sm90ScalarBroadcast<ElementScale>, Sm90AccFetch>`. The custom gated-SiLU
visitor uses two hooks: **`visit()`** accumulates fragments into a register tensor, **`end_loop()`**
processes adjacent pairs, computes SiLU and stores the halved output. Measured: **166 µs saved
(1.28x faster)** on a real deployment. Their framing is worth internalising: "you are not
approximating anything. You are just doing the same math earlier."
Ideogram V4 headline: **2.75 s → 0.44 s (6x) at 1K resolution** — bundled result, not an
epilogue-fusion number.
<https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/>,
<https://blog.fal.ai/serving-sub-second-ideogram-v4-without-quality-loss/> [verified]

### DSpark speculator on a fine-tuned MoE, on a single B200 (Jul 8 2026)

The best public end-to-end account of building a speculator for a fine-tuned MoE. They evaluated
**Qwen 3.5 397B MoE** as a baseline and shipped **the 35B MoE model in FP8 on one B200**,
optimising for low concurrency. *(Active-parameter count is not published in this post. Doubleword
separately names a `Qwen3.6-35B-A3B` in their calibration dataset — a different, later model.)*

| Configuration | Throughput | Accept length |
|---|---|---|
| SGLang, FlashInfer backend + FlashInfer/TRTLLM MoE kernels, no spec | 328 tok/s @ C1; <200 @ C8; ~100 @ C16–32 | — |
| + native MTP head (degraded by PEFT, which doesn't fine-tune MTP heads) | 500 tok/s | ~2.5 |
| + DFlash, z-lab public checkpoints (trained for the *base* model) | 468 tok/s | ~2.4 |
| + DFlash retrained from scratch on 250K samples via TorchSpec | fell short — "250K samples were not diverse enough to give very high acceptance, and the training took a long time" | — |
| + **DFlash warm-started** from public weights, then trained on their 250K | **700 tok/s** | **~3.9** |
| + **DSpark**, reimplemented in TorchSpec, warm-started from their own DFlash checkpoint | **830 tok/s at TP=1 (shipped)**; **>1000 tok/s at TP=4** | **4.6** |

Two details worth stealing outright [verified]:
- They reimplemented DSpark in TorchSpec rather than use the released trainer, because
  "Storing ~38TB of hidden states wasn't easy, especially if the disk is networked" — exactly the
  problem Baseten solved with live streaming.
- **TP made it slower until they stopped sharding the drafter.** "The Markovian heads were doing
  collective operations because they were sharded across the ranks. We fixed it by replicating the
  weights instead." They still shipped TP=1: "TP=4 was heavily underutilizing the GPUs and wasn't
  cost-effective."

Framed two ways, honestly: **2.6x throughput at single-user max interactivity**, and **16x total
throughput on the same single B200 under a 300 tok/s-per-user constraint**.
<https://blog.fal.ai/how-we-achieved-1000-tok-s-and-16x-throughput-with-dspark-for-ideogram-v4-prompt-expander/>
[verified]

---

## Doubleword

A UK inference provider (formerly TitanML) with access to Isambard-AI. Written mostly by founder
Fergus Finn. **The most quantitatively honest inference blog in this set, and the one with the
most reusable open artifacts.** Modal cites Finn's B200 draft-length estimator as the basis for
their own speculative-decoding roofline tool.

### The decode-step roofline decomposition (Jul 23 2026)

The finding most worth copying *as a practice*. On **4xGH200** (~96 GiB each, NVLink 4 P2P)
running DeepSeek-V4-Flash at concurrency 2048, ISL/OSL 1024/1024, they parsed the torch profiler
at steady-state decode and mapped every kernel to its inferred roofline ceiling [verified]:

| Bucket | Measured (per rank per step) | Their note |
|---|---|---|
| MoE expert GEMM | 30.3 ms | should be memory bound; at only ~30% of its roof |
| Act + quant | 12.6 ms | **~520 tiny kernel launches per step, of which ~2 ms is actual data movement** |
| Comms (AG + RS) | 10.8 ms | should be memory bound; vs ~3 ms at wire rate |
| Dense GEMMs | 8.9 ms | should be compute bound |
| Attention (MLA) | 5.3 ms | should be memory bound |
| DSA indexer | 1.6 ms | |
| Other | 8.8 ms | not characterised |

The structure is strikingly close to our own C1 profile. **The act+quant observation — a bucket
that looks like compute but is ~84% launch overhead — is the single cheapest thing to check for
in our own trace before optimising anything else.**

### The optimisation ladder, with exact flags [verified]

| Step | Change | Result |
|---|---|---|
| Baseline | `vllm serve --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8 --block-size 256 --gpu-memory-utilization 0.92 --max-num-seqs 2048 --numa-bind` | **5,856 tok/s** @ c2048 (1,464/GPU) |
| DP attention | `--tensor-parallel-size 1 --data-parallel-size 4 --gpu-memory-utilization 0.95 --max-num-seqs 1024` (per rank), `--compilation-config '{"max_cudagraph_capture_size":1024}'` | **12,802 tok/s** @ c2752 — **2.2x** |
| W4A8 MoE kernels | `--moe-backend humming`, `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`, `VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","input_scale_group_size":128}'` | **17,634 tok/s** — a further **+40%** |

Reasoning behind each [verified]:
- **DP attention:** "With only one shared K/V latent, naive tensor parallelism for MLA has to
  replicate the KV cache on each accelerator." Bonus: every per-token kernel then runs on 1/N of
  the batch per rank.
- **`--numa-bind`:** "If we don't bind the engines to their NUMA node, then comms between the host
  and device go over the interconnect and we lose throughput." **Directly relevant — we have 2
  NUMA nodes.**
- **W4A8 over Marlin:** vLLM defaults to Marlin, a weight-only kernel that dequantises to BF16 and
  runs BF16 tensor cores. Humming (Ant Group) "upcast[s] to fp8 in the SMs, and then do[es]
  fp8xfp8 matmuls in the fp8 tensor cores" — same weight transfer, double the FLOPs, double the
  critical batch size. Getting it working needed two upstream fixes: the WGMMA consumer never
  applied per-128 input scales on the accumulator, and the extra accumulator halved the per-tile
  register budget so the tile heuristic's default BlockM spilled. Fork: `doublewordai/humming`.
  **On B200 with native NVFP4 tensor cores this specific fix is likely moot** [inferred] — but the
  *diagnosis* (are we running a weight-only dequant-to-BF16 MoE kernel?) is worth 10 minutes.

**Roofline caveat everyone should copy:** sustained BF16 tops out at **583 TF/s, not the 990 TF/s
datasheet number**, because clocks drop once the GPU draws 530–570 W — a *software power cap*, not
thermal. Ridge point moves from ~250 to ~146 FLOP/byte, critical batch size to ~1,300.
<https://blog.doubleword.ai/throughputmaxxing-v4-flash-single-node> [verified]

### Depth vs width: why speculation beats batching for MoE (Jul 2 2026)

Setup: batch size 1, single GPU, you may work on 2 token positions. Batch 2 sequences, or
speculate 1 token with acceptance α? Their result [verified]: "spending your 2 positions on one
speculating sequence produces, globally, more output tokens per second than spending them on a
batch of 2, even with α=0.9."

Mechanism: MoE expert utilisation is highly non-uniform, and consecutive tokens within a
speculative sequence **co-activate overlapping experts**, so verifying two positions of one
sequence touches a smaller distinct-expert working set than two random sequences. Because MoE
decode is bound by expert weight movement, that is the whole game.

Depth-selection model: `m_i(γ) = 1 + Σ_{d=1..γ} Π_{k=1..d} a_k^i`. Recommendation: **per-sequence
confidence gating with ragged depths**, trimmed greedily, rather than a fixed batch-wide γ. They
acknowledge engine support for ragged speculation is limited.
<https://blog.doubleword.ai/speculating-on-the-margin> [verified]

### Released artifacts — the most usable in this file [verified]

- **`huggingface.co/datasets/Doubleword/specdec-calibration`** — 4.56 GB. Per-round acceptance
  banks (`accept, acc0..acc{D-1}`) and speculator confidence banks (`conf0..conf{D-1}`, softmax
  probabilities) for **Qwen3.6-35B-A3B and DeepSeek-V4-Flash**, drafters **MTP (native heads) and
  DFlash block-diffusion (Qwen only)**, over SPEED-Bench categories and HumanEval, **plus MoE
  expert-routing captures (expert IDs and metadata)**. The dataset viewer is currently erroring;
  download works.
- **`github.com/doublewordai/inference-lab`** — MIT-licensed **discrete-event simulator for LLM
  inference serving**, in Rust with WASM and CLI front-ends. Models speculative decoding with
  "analytic or trace-replayed acceptance, fixed and goodput-adaptive draft policies", MoE expert
  loading in its roofline model, disaggregated prefill/decode, prefix caching and several
  scheduling policies. **Ships B200/B300/H100/GH200 hardware presets and ~40 model presets.**
  This is the cheapest way to sanity-check a draft-depth policy before touching our engine.
- **`github.com/marcelroed/gigatoken`** — the hardware-aware tokenizer, dissected at
  <https://blog.doubleword.ai/inside-gigatoken> (Jul 30 2026). Numbers in Tier 3 item 10.
- **`doublewordai/humming`** — their fork of Ant Group's W4A8 MoE kernels with the two upstream
  fixes.

### When to disaggregate (Aug 11 2026, "The case for disaggregated LLM serving")

Analytical rather than measured — **they publish no A/B throughput numbers.** Useful pieces
[verified]:
- Balanced prefiller fraction: `ϕ = B·ISL / (B·ISL + r_P·t_step·OSL)`
- Fabric feasibility: `n_P·r_P·κ_store ≤ min(m_P·BW_P, m_D·BW_D, BW_fabric)`
- **GLM-5.2 on B200, concretely:** a **56k tokens/second prefill rate** produces **5.3 GB/s of KV
  cache**, against a **400 Gb/s NIC**. So the fabric is roughly an order of magnitude oversupplied
  for one prefiller [inferred from their figures] — the constraint is elsewhere.
- The real barrier is granularity: you lose fractional GPU allocation and the loss scales as 1/N.
  DeepSeek V3's published shape was **EP32 prefill units and EP320 decode units**.
<https://blog.doubleword.ai/when-to-disaggregate>

### Two more, lower priority

- **Reverse-engineering NVIDIA's cuda-checkpoint** (Jul 9 2026): cuda-checkpoint drives a service
  thread inside the target process via a pipe (opcodes 0–3 for lock/checkpoint/restore/unlock);
  GPU state serialises into a host buffer via `mmap(..., MAP_POPULATE)`. Measured on an 8,578 MiB
  device footprint: **4.5 s baseline → 3.9 s direct pipe → 1.6 s with Transparent Huge Pages →
  1.2 s with pre-allocation + async unmapping.** Checkpoint transfers at ~3 GiB/s (pageable PCIe),
  restore at ~8 GiB/s. Bottleneck named: "`mmap(..., MAP_POPULATE)` of the staging region alone
  accounts for 2.08 s of the 2.8 s checkpoint." Cold-start only; nothing for steady-state latency.
- **What happens when a GPU reads memory** (Aug 13 2026): measured **L1 hit ~15.4 ns (40 cycles),
  L2 hit ~127 ns (330 cycles), DRAM ~255 ns (660 cycles), TLB miss refill ~4.4 ns**, plus ~2% of
  accesses taking an extra ~210 ns from DRAM refresh. **Measured on an RTX 4090 (sm_89), not
  Blackwell** — useful as method, not as B200 constants.

---

## Modal

**What they run.** Modal does not build an inference engine. They build the container/scheduling
substrate and tell users to run vLLM, SGLang or TensorRT-LLM on it. Their engineering value to us
is (a) the best free Blackwell attention writeup, (b) the GPU Glossary, (c) honest engine
benchmarks.

### FlashAttention 4, reverse-engineered (Sep 26 2025)

The most useful public description of a production Blackwell attention kernel. Warp
specialisation [verified]:

| Warp role | Count | Job |
|---|---|---|
| Load | 1 | Q/K/V global→shared; **can concurrently load up to three blocks each of K and V**; supports an optional page-table tensor for Paged Attention |
| MMA | 1 | unnormalised scores from Q/K tiles, accumulate score-weighted V tiles |
| Softmax | 8 | normalised scores + running stats (max, sum) |
| Correction | 4 | watch for normalisation-scale updates and re-normalise output tiles |
| Epilogue | 1–2 | completed output tiles shared→global |

Two changes flagged as "new in FA4", both portable [verified]:
1. **Software exponentials to dodge SFU contention.** FA3 used `exp2` mapping to `MUFU.EX2` on the
   Special Function Units, of which there are far fewer than CUDA cores. FA4 uses "a different
   exponentiation algorithm on some iterations with a tunable frequency," only for **smaller
   attention head sizes**, and "stops applying it on a configurable number of the last S tiles"
   (wave quantisation makes SFU contention irrelevant at the tail). It splits `2**x` into
   `2**floor(x)` and a **cubic polynomial approximation of `2**x` on the unit interval**,
   evaluated by Horner as `(((c3 * r) + c2) * r + c1) * r + c0` in **three `fma.rn.ftz.f32x2`**
   instructions — which operate on a vector of two 32-bit values.
2. **Lazy rescaling.** Instead of updating the scale "every time a new maximum appears", FA4
   updates "only when the maximum has changed enough to threaten numerical stability", reportedly
   **reducing corrections by a factor of 10**. Modal's own annotation: "This seems like a good, and
   very portable, idea." **This is the cheapest attention win in the file for our 10.9% bucket.**

Also: FA4 deliberately uses `tcgen05.mma.cta_group::1` (single-CTA MMA), avoiding Blackwell's
2SM/2CTA matmuls, accepting a memory-throughput penalty to simplify tile scheduling — and the
ThunderKittens Blackwell attention kernel makes the opposite choice. Best performance needs
`StaticPersistentTileScheduler`, which launches at most one CTA per SM. Reported **~20% speedup
over cuDNN**.
<https://modal.com/blog/reverse-engineer-flash-attention-4> [verified]

### GPU Glossary and LLM Almanac

The glossary's **perf** section [verified] covers: Performance Bottleneck, Roofline Model,
Compute-bound, Memory-bound, Arithmetic Intensity, Overhead, Little's Law, Memory Bandwidth,
Arithmetic Bandwidth, Latency Hiding, Warp Execution State, Active Cycle, Occupancy, Pipe
Utilization, Peak Rate, Issue Efficiency, SM Utilization, Warp Divergence, Scoreboard Stall,
Branch Efficiency, Memory Coalescing, Bank Conflict, Register Pressure. Little's Law page:
`concurrency (ops) = latency (s) × throughput (ops/s)`; a GPU at 1 instruction/cycle with
400-cycle memory latency needs 400 concurrent ops, at 10 instructions/cycle it needs 4,000. And
Volkov's result (thesis §4.3) that warps needed to hide *memory* latency (~30) and *arithmetic*
latency (~24) are nearly the same — because memory's higher latency is offset by its much lower
bandwidth.

The **LLM Almanac / Engine Advisor** is a live benchmark browser. Findings from their mid-2025
pass — vLLM `0.8.x`/`0.9.x`, SGLang `0.4.6-post5-cu124`, TRT-LLM `0.20.0.rc3`, on **8xH100 SXM
single replicas on Oracle Cloud, CUDA driver 12.8**, headline model Llama 3.1 70B FP8 [verified]:
- **vLLM and SGLang "strikingly similar"** on throughput out of the box; choose on time-to-market
  for features, not speed.
- **TensorRT-LLM worse out of the box**; "can be faster if tuned for very specific workloads, but
  the engineering lift and churn should not be under-estimated."
- **Startup: vLLM ~5 min for 8B models vs SGLang ~1 min**, attributed to vLLM defaulting Torch
  graph compilation on.
- Failures they published: TRT-LLM lacked Gemma 3 and Qwen 3 support; SGLang OOM'd on Qwen 3 235B;
  SGLang had unresolved DeepSeek-V3 INT4 quantization issues.
- Cost datum: Llama 3.1 70B FP8 at **~20k tok/s for ~50¢ per million tokens**, ~17 QPS per replica.

Their **speculative-decoding roofline tool** covers B200/H200/H100 against DeepSeek-V4 Flash/Pro
and Qwen 3.5 4B–397B-A17B; draft length γ*=0–16, seqlen 4,096–131k, batch 8–32, acceptance
75–89%, block size 1–16, producing **1.0x to 1.6x** speedups — with the honest caveat that it
"tends to underestimate the benefit when overhead is a major contributor to latency, e.g. small
batch sizes on small models," i.e. **exactly our C1 regime**. Derived from Fergus Finn's
DeepSeek-V4 Flash B200 optimal-draft-length estimator.

Their **block-quant reference**: NVFP4 "blocks are 16 elements (vs. 32 for MXFP4), and the block
scale uses full FP8 E4M3 rather than a power-of-two E8M0", plus "a second per-tensor FP32 scale
[to] bridge the dynamic range gap."
<https://modal.com/gpu-glossary/perf>, <https://modal.com/llm-almanac/advisor>,
<https://modal.com/llm-almanac/summary>, <https://modal.com/llm-almanac/spec-dec-roofline>,
<https://modal.com/llm-almanac/block-quants> [verified]

### Platform engineering: GPU memory snapshots (Jul 30 2025)

Cold-start work, not latency work, but it is the only public account of using NVIDIA's CUDA
checkpoint/restore API in production. On drivers in the **570 and 575 branches**:
`cuCheckpointProcessLock()` → `cuCheckpointProcessCheckpoint()` → host snapshot →
`cuCheckpointProcessRestore()` + `cuCheckpointProcessUnlock()`. It restores **already-compiled
`torch.compile` artifacts, loaded CUDA kernels, and captured CUDA graphs**, which CPU-only
snapshots could not. Measured: vLLM with Qwen2.5-0.5B-Instruct **45 s → 5 s**; ViT with
`torch.compile` **8.5 s → 2.25 s**; NVIDIA Parakeet **20 s → 2 s**. Enabled by
`experimental_options={"enable_gpu_snapshot": True}`.
<https://modal.com/blog/gpu-mem-snapshots> [verified]

### Blackwell/B200 specifically (May 30 2025)

B200 at $6.25/hr, H200 at $4.54/hr, self-serve. Their published spec table [verified]: B200
180 GB / 8 TB/s / 5 PFLOP/s FP8; H200 141 GB / 4.8 TB/s / 2 PFLOP/s FP8; H100 80 GB / 3.5 TB/s /
2 PFLOP/s FP8. Their measured comparison (vLLM, DeepSeek V3 in native FP8, 1000 in / 128 out,
8xH200 vs 8xB200): **median TTFT 2.5x faster on B200; QPS 1.7x higher.** They also caution the
full B200 benefit "will take some time to realize" as software matures.

A useful taxonomy from a separate post (Feb 24 2025) [verified]: **Allocation Utilization**
(GPU-seconds running application code ÷ GPU-seconds paid for), **Kernel Utilization**
(GPU-seconds running kernels ÷ paid), and **MFU**. They cite the State of AI Infrastructure at
Scale 2024 report that "the majority of organizations achieve less than 70% GPU Allocation
Utilization *when running at peak demand*", note the former Banana platform at ~20% aggregate, and
claim >90% aggregate on Modal (vendor claim, unverifiable).
<https://modal.com/blog/introducing-b200-h200>, <https://modal.com/blog/gpu-utilization-guide>

---

## BentoML / Modular

BentoML joined Modular; the LLM Inference Handbook now lives at `handbook.modular.com`. BentoML
itself wraps vLLM et al.; they do not build an engine. The handbook is a genuine reference with a
few honest measured claims.

**Prefill-decode disaggregation page** — the most useful negative result in this file [verified]:
> "If your workload is too small, or your GPU setup isn't tuned for this approach, performance can
> drop (by 20-30% in our tests)."

Plus: "For shorter prompts or when the decode engine has a high prefix cache hit, running prefill
locally on the decode worker is often faster and simpler." Transports listed: NIXL, CXL, NVMe-oF.
Cross-cluster "Prefill-as-a-Service" reportedly gave **54% higher throughput and 64% lower P90
TTFT** vs a standard disaggregation baseline — **no hardware, model or config given; unusable for
comparison.**

**Speculative decoding page** [verified]: acceptance length `τ = (1 − α^(γ+1)) / (1 − α)`, and a
measured Llama-3.3-70B-Instruct-on-H100 study:
- **TP=1:** ~2x TPOT improvement, but total throughput "plateaued earlier (around 20–30 concurrent
  requests)" than baseline.
- **TP=2:** clear throughput gains over baseline; but **γ=5 "saw larger latency spikes under heavy
  concurrency (40+ requests)."**
- "In practice, however, the speedup was lower than expected" vs the theoretical 2–3x at α ≥ 0.6,
  γ ≥ 5.
- **SGLang uses "a small set of predefined speculative-length tiers (for example, `[1, 3, 7]` by
  default)" with pre-captured CUDA graphs** so switching is cost-free — the cheapest path to our
  dynamic-draft-depth item.

<https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation/>,
<https://handbook.modular.com/inference-optimization/speculative-decoding/>

Their "6 Production-Tested Optimization Strategies" post is a well-organised map, but the
supporting evidence is customer case studies, not inference measurements. **Cut from this file as
padding.**

---

## Replicate

**Honest verdict: nothing substantive on LLM inference.** Model launches, prompting guides,
fine-tuning tutorials, diffusion work. They wrap Cog/containers rather than build an engine. One
thing worth recording:

**`torch.compile` artifact caching** (Sep 8 2025) [verified]. Cache compiled artifacts across
container lifecycles, "keyed on model version and stored close to GPU nodes"; containers check for
the cache on start and update it on graceful shutdown. Measured boot times: `flux-kontext-dev`
~120 s → ~60 s; `prunaai/flux-schnell` ~150 s → ~70 s; `prunaai/flux.1-dev-lora` ~400 s → ~150 s.
They also state the compiled `flux-kontext-dev` "runs over 30% faster than the uncompiled one."
Modal's GPU memory snapshots solve the same problem more generally.
<https://replicate.com/blog/torch-compile-caching>

Their TaylorSeer diffusion-caching post is not transferable to LLM decode.

---

## RunPod

**Honest verdict: nothing transferable.** Buyer education, storage guides, model-launch posts. The
most technical piece is *GPU memory math for full-parameter fine-tuning* (updated Aug 11 2026),
which derives `M_state = P · (b_w + b_g + b_m + b_v1 + b_v2)` = 16 bytes/param for BF16 + AdamW +
FP32 master copy, so "a 7B model in BF16 is about 14 GB of weights" but "roughly 112 GB of resident
state before a single activation tensor is allocated." Correct, and irrelevant to decode latency —
it is a *training* sizing post. They ship inference on vLLM/SGLang without publishing engine work.
<https://www.runpod.io/blog/gpu-memory-math-tuning-sizing> [verified]

---

## Techniques ranked by transferability to our stack

| # | Technique | Source | Attacks | Published effect | Config completeness | Difficulty | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | **Suffix-automaton speculation layered on the existing drafter** | Baseten `sa_spec`, merged in TRT-LLM | Decode steps on repetitive/agentic traffic | +30–33% accept length & throughput; up to 40% at equal latency; provably zero overhead | model + dataset given; **no hardware or batch size** | Low–Medium | **High** — Apache-2.0, upstreamed |
| 2 | **Pull dispatch / push combine; kill cross-rank signalling** | Cursor MoK | Collectives 19.6%, esp. the 47% arrival skew | signalling 103 µs → 18 µs; +29% NVLink util under imbalance | **measured at EP64 on NVL72, not EP8 in-node** | High | Medium — will not transfer at face value at our scale |
| 3 | **Warp decode** — one warp per output neuron; fold routing weight into a register accumulator; `shfl.sync.bfly` reduction | Cursor | MoE expert GEMMs (19.4% C1) | 1.84x MoE decode on B200; 3.95 TB/s @ B=32; +accuracy | **"a Qwen-3 style model"; no shapes, no dtype, no sweep** | High (no code) | Medium — mechanism fully described, result not tied to our shape |
| 4 | **RMSNorm/SwiGLU epilogue fusion; permute B's columns so gate/up are fragment-adjacent** | fal | Dense GEMM (37.1% C1) | 166 µs / 1.28x on one deployment; removes an HBM round-trip | SM100 layout facts exact; the 1.28x has no shape attached | Medium–High (column permute is Low) | High on mechanism |
| 5 | **Lazy softmax rescaling — update the scale only when the max threatens stability** | Modal (FA4) | Attention (10.9% C1) | ~10x fewer correction passes | attribution is second-hand (Dao at Hot Chips) | Low–Medium | High — Modal calls it "very portable" |
| 6 | **Skip the DSA indexer when `seq_len < K`; fuse the high-precision indexer projections** | Baseten | DSA indexer (5.8% C1) | **not quantified at all** | none | Low | High that it's true; unknown magnitude |
| 7 | **Data-parallel attention instead of TP for MLA** | Doubleword (measured); Baseten (ADP on standard tier) | Aggregate throughput / cost per user | 5,856 → 12,802 tok/s (2.2x) | full: 4xGH200, vLLM, c2048→2752, 1024/1024, all flags given | Medium | High — **but a throughput lever, not a C1 lever** |
| 8 | **Fuse draft into the target graph; never shard draft-head weights across TP** | Baseten; fal (Markov-head fix) | Spec-dec host overhead + a hidden collective | "overhead essentially zero"; fal went from TP-slower to >1000 tok/s | fal's side is fully configured; Baseten's is a bare claim | Medium | High — two independent confirmations |
| 9 | **Confidence-gated per-sequence draft depth via pre-captured CUDA graph tiers** | Doubleword (model + data + simulator); handbook (SGLang `[1,3,7]`) | Accept length at C1 and C16 | model + 4.56 GB open calibration set + MIT simulator with B200 presets | simulated, not measured end-to-end | Medium | Medium–High — **best artifact support of anything here** |
| 10 | **Async collective overlap (not fusion) at 8 GPUs** | fal (8xB200) | Collectives minus skew | −23–25% chunk latency at 8 GPUs, ~−3% e2e; fusion −4.6% at 8 | full setup published | Medium | High for dense attention; **unmeasured for MoE dispatch** |
| 11 | **Software-partitioned comms SMs vs streams/green contexts** | Cursor MoK | Collectives competing with GEMMs for SMs | NVLink saturated with <1/3 of SMs | qualitative | Medium–High | High on mechanism |
| 12 | **Per-expert L2 supergrouping in grouped GEMMs** | Cursor | MoE GEMMs | ~2650 TFLOP/s grouped MXFP8, 4% below non-grouped; bad access patterns cost ~50% | training workload | Medium | High |
| 13 | **CUDA-core cubic-polynomial `exp2` mixed with `MUFU.EX2` at tunable frequency** | Modal (FA4) | Attention | avoids SFU queueing; only for smaller head sizes | qualitative | Medium | High |
| 14 | **Check the "act+quant" bucket for launch overhead before optimising anything** | Doubleword | Whole-profile hygiene | 12.6 ms bucket, ~2 ms real data movement, ~520 launches/step | 4xGH200 | **Very low** | High — a profiling practice, not a change |
| 15 | **Measure sustained B200 TFLOP/s under the power cap, not the datasheet** | Doubleword | Every roofline we build | 583 TF/s vs 990 datasheet on GH200 | GH200; must be re-measured on B200 | Very low | High as a practice |
| 16 | **`--numa-bind`-equivalent pinning on multi-socket nodes** | Doubleword | Host↔device transfer | not quantified | flagged as important, no number | Very low | Medium — we have 2 NUMA nodes |
| 17 | **NVFP4 quantiser as <100 lines of Triton with `cvt.rn.satfinite.e2m1x2.f32`** | fal | Quantisation overhead | ~7 TB/s on B200, matches 2000-line CUDA at large shapes | shapes not tabulated | Low | High — code shown inline |
| 18 | **Quantiser that emits scales directly in `tcgen05` packed layout** | fal (6+ TB/s), Cursor (6.2 TB/s) | Quantisation overhead | vs TE 4.4 TB/s / TorchAO 4.5 TB/s | two independent implementations agree | Medium–High | High |
| 19 | **Minibatch heuristic `T ≥ 2C·128·256/min(2I,H)`** | Cursor | Comm/compute granularity | predicted 2368 for Kimi 2.5; measured optimum 2560 | training | Low (it's a formula) | High |
| 20 | **Open-source hardware-aware tokenizer (`gigatoken`)** | Doubleword | TTFT | 1,039 MB/s 1T, 8,792 MB/s 16T; 37.6x on 1M-token/1024-segment | full | Low | High — **released, unlike Basetenkenizer** |
| 21 | **KV-cache-aware routing on a radix tree with overlap+load scoring** | Baseten / Dynamo | TTFT at multi-replica scale | 50%↓TTFT, 34%↓TPOT, 62%↑TPS at 89% hit rate | model + lengths + replicas; **comparator unstated** | Medium | Medium — only pays at ≥2 replicas |
| 22 | **Live draft training from streamed hidden states** (memory ∝ `max_num_tokens_per_iter`) | Baseten | Accept-rate drift | +20% median accept rate | no hardware, no model | High | Medium — no code released |
| 23 | **W4A8 instead of weight-only dequant-to-BF16 MoE kernels** | Doubleword (Humming vs Marlin) | MoE GEMMs | +40% on top of a 2.2x baseline | full flags given | Medium | High on Hopper; **likely moot on B200 NVFP4 tensor cores** |
| 24 | **Amortised KV cache compaction (Still / Attention Matching)** | Baseten research | KV footprint at C64 | 8x–200x compression, RULER +8–22 pts | Qwen/Gemma, 8k–128k; **no serving latency numbers, no code** | Very high | Low for production; interesting for cost |
| 25 | **PD disaggregation** | Baseten (2x, and separately 6x, both unconfigured); handbook (−20–30%); Doubleword (needs 1000s of GPUs) | Aggregate throughput | contradictory across sources | **essentially none from any source** | High | **Low for our C1 objective** |
| 26 | GPU memory snapshots / `torch.compile` artifact caching | Modal, Replicate, Doubleword | Cold start | 9–10x / 2–3x / 4.5 s→1.2 s | full | Medium | High, but orthogonal to our objectives |

---

## Verification log — what the first pass got wrong

Recorded so the same errors are not reintroduced.

1. **Cursor's GLM-5.2 shape was attributed to the wrong post.** The first pass claimed the
   warp-decode post's "MoE shapes list literally includes GLM-5.2 (E=256, H=6144, I=2048, top-k 8)"
   and used that to rate the technique "tested on GLM-class shapes." The shape table is in
   **mixture-of-kittens** (a *training* megakernel post). The warp-decode post names only "a Qwen-3
   style model" and gives no shapes. **Corrected; confidence downgraded from High to Medium.**
2. **`glaiveai/code_edits_sample`** → the real dataset is **`glaiveai/code-edit-samples`**.
3. **The sa_spec "40%" claim was conflated.** The 40% figure is "up to 40% higher throughput at
   equal latency and up to 40% lower latency at equal throughput, compared to MTP alone" on the
   DeepSeek-V3.1-NVFP4 / glaive benchmark, *plus* a separate softer statement that "we commonly see
   up to 40% improvements on coding applications for production workloads." Both now quoted
   separately.
4. **fal's DSpark model was wrong.** First pass said "Qwen3.6 35B-A3B MoE". The post says they
   evaluated **Qwen 3.5 397B MoE** as a baseline and shipped **the 35B MoE model (FP8)**; no active
   parameter count is published there. (`Qwen3.6-35B-A3B` appears in Doubleword's dataset — a
   different model.)
5. **Kimi K2.5 post date was wrong**: Feb 10 2026 → **March 9 2026**.
6. **`speculative_decoding_mode ∈ {LOOKAHEAD_DECODING, DRAFT_TOKENS_EXTERNAL}`** → the live docs
   list **only `LOOKAHEAD_DECODING`**. `DRAFT_TOKENS_EXTERNAL` removed.
7. **The GPT-OSS TP8 quote was invented.** "this configuration yields the lowest latency" is not in
   the post. Replaced with the real sentence: "Tensor Parallelism offered better latency, while
   Expert Parallelism offered better system throughput." Also, the `sota-performance-for-gpt-oss`
   post is **Nov 10 2025**, not Oct 24, and reports 500+ tok/s at launch — which does not reconcile
   with the ~400 baseline in the Oct 24 post. Flagged as unreconciled.
8. **"Dynamic-length speculation" was attributed to the wrong post.** The quote is real and exact,
   but it is in the **suffix-automaton post's "Areas for further work"** (May 5 2026), not the
   GLM-5 or GLM-5.2 posts.
9. **Modal's B200 spec "9 PFLOP/s FP4"** is not on the page. Removed; the published table gives
   FP8 only.
10. **The TRT-LLM spec-dec post's hardware was omitted**, which materially changes how it reads: it
    is **H100 / 4xH100** with Qwen 2.5 Coder 14B+0.5B and Llama 3.1 70B+8B (May 16 2025), i.e.
    Hopper and two-model speculation, not Blackwell and not EAGLE.
11. **"Baseten runs GLM-5.2 on 8xB200 — the exact model/hardware we run"** — they say "NVIDIA B200
    GPUs" with no count. The only published 8xB200 deployments are Kimi K2 Thinking and GPT-OSS
    120B. Downgraded.
12. **KV-routing comparator "vs random routing"** could not be confirmed in the post. Comparator
    now marked unstated.
13. **`modal.com/blog/cuda-graphs` and `modal.com/blog/the-hidden-economics-of-llm-inference` both
    return HTTP 404.** They were listed as "empty body, may contain relevant material." They do not
    exist at those URLs.
14. **The "Still" post was listed as unfindable.** It exists:
    <https://www.baseten.co/research/still-amortized-kv-cache-compaction-in-a-single-forward-pass/>
    (Jun 10 2026, arXiv 2606.07878). Now summarised.
15. **The MoK granularity formula** appears inverted in a naive scrape of the post. The form printed
    here (`T ≥ 2C·128·256 / min(2I,H)`) is the one that reproduces their stated T ≥ 2368 from
    H=7168, I=2048, C=148.

### Still unsourced

- **Baseten's PD-disaggregation configuration.** They report 2x (GLM-5.2 head-to-head) and,
  separately, "up to 6x higher TPS per GPU is realistic" (Dynamo Day), and promised "new benchmarks
  coming out in the next few weeks." I found no post giving prefill:decode worker ratios, transport
  detail, or the aggregated baseline config. **Treat both multiples as marketing.**
- **Baseten's parallelism degrees for GLM-5.x and Kimi.** Only TP8 EP1 for GPT-OSS 120B and TP4 EP1
  for their InferenceMAX reproduction are published.
- **Any fal post on LLM serving beyond the DSpark prompt-expander.** fal is diffusion-first; there
  is no fal equivalent of a "how we serve GLM" post.
- **Hardware/batch config for the sa_spec headline numbers.** Model and dataset are given; GPU and
  batch size are not.
- **Any independent reproduction of any number in this file.** Cursor's and Doubleword's are the
  most checkable because code and harnesses are released.

### Adjacent, for the Kimi K3 work item

Baseten's architecture explainer (Jul 30 2026) states Kimi K3 has **898 experts total — two shared
and processing every token, with the router selecting 16 of the remaining 896 per token** — and
covers Kimi Delta Attention (per-channel decay control) and gated-delta-net lineage. It also notes
"up to 6x higher decode throughput" for Kimi Linear and that "AttnRes adds roughly 2% inference
latency" [reported]. This is an architecture post, not a serving post, but the 2-shared + 16-of-896
routing shape is the number to plan capacity against.
<https://www.baseten.co/blog/22580-gpt-2-to-kimi-k3-explained/> [verified]

---

## Sources

All URLs below were fetched and read during this verification pass.

**Baseten**
- <https://www.baseten.co/blog/how-we-built-the-fastest-glm-5-api/> — DSA indexer, fused MTP, NVFP4, 186+ tok/s; authors Tri Dao, Philip Kiely, Madison Kanna
- <https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/> — 280+ tok/s, TTFT 800 ms, 7.9 s TTFAT, 2x disaggregation, 40B active params, shared DSA, NIXL KV transfer, KV block transpose kernel
- <https://www.baseten.co/blog/how-we-built-the-new-fastest-api-for-glm-52/> — 601 tok/s at ~10k/~1k, ADP on standard tier, TP+EP + reduced max batch on Fast tier
- <https://www.baseten.co/blog/introducing-glm-52-fast/> — the Fast tier, `zai-org/GLM-5.2-Fast`
- <https://www.baseten.co/blog/how-we-built-the-fastest-kimi-k2-5-on-artificial-analysis/> — custom ~1B EAGLE-3, INT4→NVFP4, 340+ tok/s (Mar 9 2026)
- <https://www.baseten.co/blog/kimi-k2-thinking-at-140-tps-on-nvidia-blackwell/> — 8xB200 single node, TP+EP mixture, INT4→BF16→NVFP4, 140+ tok/s / 300 ms TTFT
- <https://www.baseten.co/blog/how-we-made-the-fastest-gpt-oss-on-nvidia-gpus-60-percent-faster/> — TP8 EP1 on 8xB200, EAGLE-3, ~400→650+ tok/s
- <https://www.baseten.co/blog/sota-performance-for-gpt-oss-120b-on-nvidia-gpus/> — TP-vs-EP tradeoff quote, Blackwell-only MoE backend, 500+ tok/s at launch
- <https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/> — suffix automaton, full API, zero-overhead proof, "dynamic-length speculation" future work
- <https://github.com/basetenlabs/sa_spec> — Apache-2.0, `SA_SPEC_THRESH` default 4, TRT-LLM PRs #11434 / #10951
- <https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/> — KV routing, Qwen3 Coder 480B, 89% hit rate, OpenRouter shadow test
- <https://www.baseten.co/blog/nvidia-dynamo-day-baseten-inference-stack/> — engine-agnostic stance, AIConfigurator, NIXL KV Block Manager, "up to 6x" disaggregation claim
- <https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/> — TTT-length, LR table, draft-token counts, the regenerate-with-target rule
- <https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/> — streamed hidden states, UCXX, Trio, +20% accept
- <https://www.baseten.co/blog/dflash-faster-llm-inference/> and <https://arxiv.org/abs/2602.06036> — block-diffusion drafting, Qwen3-8B B200 numbers, ICML 2026
- <https://www.baseten.co/blog/making-kimi-k3-tokenization-18x-faster-for-million-token-agentic-workloads/> — Basetenkenizer internals and the honest gigatoken comparison
- <https://www.baseten.co/blog/how-to-build-a-day-zero-api-for-kimi-k3/> — GB300 NVL72 topology, MXFP4/MXFP8, config sweep list
- <https://www.baseten.co/blog/inference-engineering-for-deepseek-v4-pro-0813/> — 1.7T, native MXFP4, DSpark speculator, no numbers
- <https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm/> — H100/4xH100, draft/target contention, chunked-prefill scheduler bug, the benchmark that got worse
- <https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/> — hash-ring peer cache, single-flight, >2 GB/s
- <https://www.baseten.co/resources/guide/the-baseten-inference-stack/> — prefill prioritisation, spec-dec auto-disable, InfiniBand disk KV offload, logit-biasing structured output
- <https://www.baseten.co/blog/how-to-run-llm-performance-benchmarks-and-why-you-should/> — InferenceMAX reproduction on 4xB200 TP4 EP1, image tag, server-side benchmarking
- <https://www.baseten.co/blog/accelerating-inference-nvidia-b200-gpus/> — B200 platform claims (unconfigured)
- <https://docs.baseten.co/engines/engine-builder-llm/engine-builder-config> — full config surface
- <https://www.baseten.co/research/> — research index
- <https://www.baseten.co/research/still-amortized-kv-cache-compaction-in-a-single-forward-pass/> — Still, arXiv 2606.07878
- <https://www.baseten.co/research/repeated-kv-cache-for-long-running-agents/> — Attention Matching, 50x, chunked compaction
- <https://www.baseten.co/blog/22580-gpt-2-to-kimi-k3-explained/> — Kimi K3 routing shape, KDA

**Cursor (Anysphere)**
- <https://cursor.com/blog/warp-decode> — output-centric MoE decode, 1.84x on B200, 3.95 TB/s @ B=32
- <https://cursor.com/blog/kernels> — MXFP8 recipe, `tcgen05` constraints, expert-wise supergrouping, DeepGEMM comparison, quantiser at 6.2 TB/s
- <https://cursor.com/blog/mixture-of-kittens> — push/pull NVLink analysis, signalling latency, minibatch heuristic, ring token buffers, CLC, MoE shape table with GLM-5.2
- <https://github.com/cursor/mixture-of-kittens> — Apache-2.0, SM100/SM103, training megakernel

**fal**
- <https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/> — 8xB200 overlap/fusion sweep, symmetric memory, "timeline overlap ≠ throughput overlap"
- <https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/> — CuTeDSL quantiser, K-split, TMA, `STG.E.U8` bottleneck, swizzling negative result
- <https://blog.fal.ai/instruction-level-control-with-inline-elementwise-asm-in-triton/> — `inline_asm_elementwise`, `cvt.rn.satfinite.e2m1x2.f32`, NVFP4 in <100 lines
- <https://blog.fal.ai/serving-sub-second-ideogram-v4-without-quality-loss/> — SM100 tile/fragment layout, revisit epilogue, SwiGLU column permutation, QAD with straight-through estimator
- <https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/> — CUTLASS EVT node vocabulary, `visit()`/`end_loop()`, 166 µs / 1.28x
- <https://blog.fal.ai/how-we-achieved-1000-tok-s-and-16x-throughput-with-dspark-for-ideogram-v4-prompt-expander/> — MTP → DFlash → DSpark ladder on one B200, TP drafter-sharding trap

**Doubleword**
- <https://blog.doubleword.ai/throughputmaxxing-v4-flash-single-node> — DP attention 2.2x, Humming W4A8 +40%, per-kernel roofline decomposition, power-cap correction, `--numa-bind`
- <https://blog.doubleword.ai/speculating-on-the-margin> — depth-vs-width, expert-overlap argument, `m(γ)` model
- <https://blog.doubleword.ai/when-to-disaggregate> — allocation and fabric equations, GLM-5.2-on-B200 KV rate
- <https://blog.doubleword.ai/inside-gigatoken> — SIMD pretokenisation, pretoken memoisation, 8,792 MB/s at 16 threads
- <https://blog.doubleword.ai/what-happens-when-you-checkpoint-a-cuda-process> — cuda-checkpoint internals, 4.5 s → 1.2 s
- <https://blog.doubleword.ai/what-happens-when-a-gpu-reads-memory> — measured L1/L2/DRAM latencies (RTX 4090)
- <https://huggingface.co/datasets/Doubleword/specdec-calibration> — 4.56 GB acceptance/confidence banks + MoE routing captures
- <https://github.com/doublewordai/inference-lab> — MIT Rust inference simulator with B200/B300/H100/GH200 presets
- <https://github.com/marcelroed/gigatoken> — the tokenizer

**Modal**
- <https://modal.com/blog/reverse-engineer-flash-attention-4> — warp specialisation, software `exp2`, lazy rescaling, `cta_group::1`
- <https://modal.com/blog/gpu-mem-snapshots> — CUDA checkpoint/restore API, driver branches, measured cold boots
- <https://modal.com/blog/introducing-b200-h200> — specs, pricing, DeepSeek V3 H200-vs-B200 comparison
- <https://modal.com/blog/gpu-utilization-guide> — the three utilizations
- <https://modal.com/gpu-glossary/perf> and <https://modal.com/gpu-glossary/perf/littles-law>
- <https://modal.com/llm-almanac/advisor>, <https://modal.com/llm-almanac/summary>, <https://modal.com/llm-almanac/spec-dec-roofline>, <https://modal.com/llm-almanac/block-quants>

**BentoML / Modular**
- <https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation/> — the −20–30% negative result
- <https://handbook.modular.com/inference-optimization/speculative-decoding/> — τ formula, Llama-3.3-70B H100 TP=1 vs TP=2 study, SGLang `[1,3,7]` tiers

**Replicate**
- <https://replicate.com/blog/torch-compile-caching> — artifact caching boot times

**RunPod**
- <https://www.runpod.io/blog/gpu-memory-math-tuning-sizing> — training VRAM math only

**Confirmed nonexistent (returned HTTP 404 during this pass)**
- `https://modal.com/blog/cuda-graphs`
- `https://modal.com/blog/the-hidden-economics-of-llm-inference`
