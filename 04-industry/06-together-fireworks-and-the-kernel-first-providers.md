# Together AI and Fireworks: the providers who compete on kernels

*Research date: 2026-08-17. All URLs fetched and read on that date unless marked otherwise.*

## What this is

A mine of published engineering from the inference providers who compete on **kernel and
quantization engineering** rather than orchestration — Together AI and Fireworks AI primarily,
plus the providers currently sitting above us on the Artificial Analysis board for GLM-5.2
(Makora, Databricks, Baseten), the speculator-focused clouds (Nebius, FriendliAI), and the
non-GPU accelerators (Groq, Cerebras, SambaNova) read only for the *architectural lesson* about
batch-1 latency.

Everything here is labelled:

- **[verified]** — I fetched the URL given and read the content.
- **[reported]** — the company claims it; I read the claim but it is not independently reproduced.
- **[inferred]** — my own reasoning applied to their published facts, not something they said.
- **[unverified]** — I could not source it and am flagging the gap.

A hard note on vendor numbers up front: **every throughput number in this document that comes
from a vendor blog is marketing until we reproduce it.** Config asymmetries are pervasive and I
call them out inline. The one place we have neutral measurement is the Artificial Analysis
provider table, which is the board we are actually chasing.

### The single most important measured fact in this document

I pulled the live Artificial Analysis provider table for GLM-5.2 **[verified]**
(<https://artificialanalysis.ai/models/glm-5-2/providers>), P50 over a trailing 72 hours:

| Provider | Output tok/s | TTFT (s) | $/1M | Quantization label |
|---|---|---|---|---|
| **Databricks** | **336** | 0.80 | — | (not labelled) |
| **Makora** | **330** | 0.86 | $0.66 | **NVFP4** |
| Baseten (FAST) | 247 | 1.72 | $0.54 | — |
| Nebius | 220 | 1.10 | $1.06 | FP4 |
| Together AI | 209 | 0.66 | $0.67 | — |
| FriendliAI | 189 | 1.29 | $0.42 | — |
| CoreWeave | 188 | 1.39 | $0.49 | — |
| Wafer | 173 | 5.73 | $0.79 | — |
| Parasail | 166 | 1.00 | — | NVFP4 |
| Crusoe | 162 | 0.95 | $0.36 | NVFP4 |
| Baseten | 110 | 1.81 | $0.46 | — |
| SiliconFlow | 101 | 2.03 | $0.62 | FP8 |
| **Fireworks** | **97** | 1.35 | $0.48 | — |
| Scaleway | 82 | 1.68 | $1.57 | — |
| Novita | 79 | 2.04 | $0.42 | FP8 |
| DeepInfra | 64 | 1.19 | $0.49 | FP4 |

**Fireworks publishes "446 tok/sec on Artificial Analysis" for their GLM 5.2 Fast endpoint**
**[reported]** (<https://fireworks.ai/blog/glm-5p2-fast>) while the AA provider table I fetched
lists Fireworks at **97 tok/s**. Either AA measures their Standard endpoint and lists Fast
separately/not at all, or the 446 figure is a peak rather than a trailing-72h P50. This is the
cleanest available illustration of the gap between a vendor's headline and the board's
methodology, and it is a warning for how we report our own 365 tok/s. **[inferred]** Our 365
tok/s on real data would, if AA measured it the same way they measure these providers, sit at
the top of this table — but the board is P50 over 72h **over the public internet from GCP
us-central1-a**, which loads network RTT and sustained-load variance onto us that our internal
measurement does not capture.

**Makora at 330 tok/s on NVFP4 is our closest analogue and the one to study**, because unlike
Databricks they publish their mechanisms. See the Makora section — they have three techniques we
can implement.

---

## Bottom line for our system — ranked list of what we should steal

Ranked by (expected effect on our two objectives) × (confidence it transfers) ÷ (difficulty).
Our measured C1 hotspots for reference: dense GEMM 37.1%, collectives 19.6% (47% of which is rank
arrival skew), MoE expert GEMMs 19.4%, attention 10.9%, DSA indexer 5.8%.

### 1. KV-outer (KV-stationary) sparse-attention kernel — Fireworks, open source, Apache 2.0

**Mechanism [verified]:** invert the attention loop. Instead of FlashAttention's query-stationary
form (one threadblock owns a Q tile, streams selected KV through it), iterate over
`(kv_head, kv_block)` in the outer loop and gather the queries that selected that block. Load the
KV block **once**, run a full `128×128` MMA tile instead of a degenerate decode-shaped
`16×128` tile, and emit one partial `(O_partial, m, l)` per `(query, kv_block)` which a separate
combine kernel LSE-merges.

**Why it is the top pick:** our DSA sparse attention plus indexer is 16.7% of C1 time, and the
query-stationary form is exactly what we are almost certainly running. Fireworks measured
**1.9–2.4× on the attention kernel alone vs FlashInfer's Q-outer**, and **1.18–1.43× on the full
module** including indexing, scheduling and combine, on a single B200 in fp8 **[reported]**
(<https://fireworks.ai/blog/kernel-optimization-for-minimax-m3-on-nvidia-blackwell>).

**Expected effect [inferred]:** if we hit the low end of the module figure (1.18×) on the
attention+indexer 16.7%, that is ~2.5% of C1 wall clock; at the high end (1.43×) ~5%. Meaningful
but not transformative on its own — this is a solid, low-risk win, not the headline.

**Difficulty: medium.** Code is public and Apache 2.0 at `github.com/fw-ai/minimax-kernels`
**[verified]** — CuTe-DSL, targets SM100+, has its own tests and benchmarks, needs CUDA 13 and
`nvidia-cutlass-dsl >= 4.5.1`. **Critical caveat [inferred]:** their kernel is built for GQA
(Hq=64, Hkv=4, D=128, 128-token blocks, top-16). Our DSA is sparse **MLA**, which has a different
head/latent structure. The *crossover analysis* and the *loop inversion idea* transfer cleanly;
the kernel body does not drop in.

**They publish the crossover condition, which is the genuinely reusable part [verified]:**
KV-outer beats Q-outer when `nsb / N < 2.85`, where `nsb` = distinct selected KV blocks and
`N` = query tokens. At batch 1 with heavy block overlap across queries this is strongly
satisfied; at very low concurrency with disjoint selections it may not be. **We should compute
this ratio on our own traces before committing.**

### 2. Attack rank arrival skew with static scheduling and a megakernel-style interpreter

**This is our single largest addressable inefficiency** — 47% of 19.6% ≈ **9.2% of C1 wall clock
is rank arrival skew**, i.e. GPUs waiting for peers at collectives. Nobody in this document
publishes a fix aimed at exactly that, but two published bodies of work bear directly on it.

**Mechanism A — the megakernel [verified]** (Hazy Research / Together lineage,
<https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>): fuse the entire forward pass
into one persistent kernel driven by an **on-GPU interpreter**. Each SM receives a sequence of
instructions **scheduled ahead of time on the Python side**. Seven instruction types cover a
Llama forward pass (fused RMSNorm+QKV+RoPE; attention; attention reduction; O-proj+residual;
fused RMSNorm+up-gate+SiLU; down-proj+residual; RMSNorm+LM head). Shared memory is explicitly
paged — they split "the first 213kB of shared memory on an H100 into 13 16KiB pages" that
instructions request and release, so the interpreter hands a released page straight to the next
instruction. Synchronisation is a **counter system in global memory**: an instruction increments
a counter on completion, dependents wait for a target value. This lets weight loads for
instruction *n+1* begin while instruction *n* is still storing results.

**Their measured numbers [reported]:** Llama-1B (1.24B params) bf16, **batch size 1**, 32-token
prompt / 128 generated tokens. On H100: **78% of memory bandwidth**, vs "vLLM and SGLang – are
only able to use at most 50% of available GPU bandwidth", giving "almost 2.5x faster than vLLM
and over 1.5x faster than SGLang". On B200: **under 680 µs per forward pass**, "over 3.5x" vs
vLLM. They also publish a B200 runtime breakdown of ~600 µs that is the most useful part for us:

| B200 component | Time |
|---|---|
| Storing activations, awaiting consistency, loading them | 250 µs |
| RMSNorm + matrix-vector compute (95% on matvec) | 200 µs |
| Awaiting weights from global memory | 30 µs |
| Low-level cross-warp synchronisation overhead | 40 µs |
| Setup and other overheads | 80 µs |

**The load-bearing lesson [inferred]:** at batch 1, **42% of their B200 time is activation
store/load/consistency at instruction boundaries, not compute.** That is the same class of cost
as our rank arrival skew — inter-step and inter-rank handoff, not math. They also note the
counterfactual: "with 7 kernel launches per layer, and 16 layers, even with an optimistic 5 us of
stalling per kernel... generation would run at just ~770 forward passes per second."

**Mechanism B — kill launch and dispatch overhead in the sparse-attention path [verified]:**
Fireworks explicitly lists two optimisations in the MiniMax M3 post that are pure overhead
removal and directly applicable: (a) **C++ AOT dispatch** — "Kernels AOT-exported per config,
driven from C++ op. Removes Python/CuTe-DSL launch overhead between kernel launches"; and
(b) **D2H elimination** — "All tensor shapes fixed per-request, aligned to `q_len`,
`max_kv_blocks`, `num_splits`... No D2H sync by skipping padded compute with persistent kernel."
Plus (c) a **deterministic, atomic-free load-balanced scheduler** with grid ≈ #SMs that flattens
all `(kv_head, kv_block, query)` work into one sequence per CTA slice.

**Expected effect [inferred]:** the D2H-sync elimination and AOT dispatch are the highest
value-per-hour items in this entire document for our TPOT of 2.74 ms. A single D2H sync per decode
step costs O(10 µs); at 2.74 ms TPOT that is a low single-digit percentage each, and they compound
with skew because a sync point is where skew *materialises*. Fixed-shape, atomic-free,
deterministic scheduling is also the GPU-legal version of the Groq determinism lesson (see the
accelerator section) and should directly reduce arrival-time variance across ranks.

**Difficulty:** AOT dispatch + D2H elimination = **low-medium**, do these first. Full megakernel =
**high**, and note their result is on a 1.24B dense model at TP1 — **[inferred]** a 256-expert MoE
at TP8 with all-to-all is a categorically harder megakernel target, and I found **no published
megakernel for an MoE model at TP8**. Treat the full megakernel as a research bet, the overhead
removals as immediate work.

### 3. MixFP4 — free accuracy on top of our NVFP4 build

**Mechanism [verified]** (<https://makora.com/blog/mixfp4>): NVFP4 uses 16-element blocks, each
with an FP8 E4M3 scale. MixFP4 allows **each block to choose between NVFP4 (E2M1) and INT4**, and
encodes that choice **in the sign bit of the scale factor** — "an 8-bit value composed of an
unsigned FP7 E4M3 scaling factor with its sign bit instead used to indicate the data type of the
block (`0` for INT4 and `1` for NVFP4)". Since the scale was always positive, the sign bit was
dead space: **zero additional memory cost.** Rationale: "Blocks with large outliers benefit from
exponent-heavy NVFP4 representations, while flatter blocks are better represented by an INT4
codebook."

**No calibration set needed [verified]:** "The block format is selected from quantization error on
the weights themselves, so the conversion does not need a representative prompt dataset, a
calibration pass, or application-specific tuning."

**Their measured quality on Qwen3.6-35B-A3B [reported]:**

| Metric | MixFP4 (Makora) | NVFP4 (NVIDIA) | NVFP4 (Unsloth) |
|---|---|---|---|
| Checkpoint size | 21.29 GiB | 21.85 GiB | 23.01 GiB |
| MMLU-Pro (instruct) | 62.62% | 61.80% | 60.80% |
| KL divergence | 0.026935 | 0.038476 | 0.061846 |
| WikiText-2 perplexity | 6.5022 | 6.6129 | 6.6984 |

**Why this matters to us specifically [inferred]:** Makora serves GLM-5.2 at **330 tok/s labelled
NVFP4** on the AA board, and separately publishes MixFP4. It is a reasonable inference — not a
stated fact — that their GLM-5.2 NVFP4 endpoint is MixFP4 under an "NVFP4" label. If we are
quality-constrained on our NVFP4 build and holding back FP4 coverage on sensitive layers, MixFP4
buys accuracy back at **zero memory and zero bandwidth cost**, which converts directly into more
FP4 coverage and therefore into GEMM time (37.1% dense + 19.4% MoE = 56.5% of C1).

**Difficulty: medium.** The format change is trivial; **the cost is entirely in the kernel** — the
GEMM must branch per block on the scale's sign bit and dequantize two ways. Makora is explicit
that their reference implementation is HF transformers and "designed primarily for others to
verify our checkpoints quality, not to deliver fast inference", with production speed requiring
"custom MixFP4 kernels". **They have not open-sourced the fast kernels.** The KL-divergence column
is the right metric and we should adopt it regardless.

### 4. Rebuild the EAGLE draft head as a DSPARK-style semi-autoregressive speculator

We run EAGLE 3-1-4. Makora publishes an ablation that takes accepted length from **3.517 → 4.915**
on GLM-5.2-NVFP4 — our exact model **[verified]**
(<https://makora.com/blog/dspark-for-glm52>). Four stacked changes:

1. **Layer-78 final residual tap** added as a sixth input to the shared projection, initialised
   from GLM's native MTP projection (`eh_proj`).
2. **Per-layer fusion** — keep the five GLM tap contributions separate per draft layer so each
   layer learns its own weighting. Costs "just 25 scalar logits in the five-layer, five-tap run
   (30 in the six-tap Full Recipe)".
3. **Position-specific learned queries** — "Each future slot its own learned starting query. Slot
   1 receives query 1, slot 2 receives query 2." Block-8 uses a 43,008-param table (7×6144);
   block-16 uses 92,160 (15×6144).
4. **Recurrent head** — "Small recurrent state, allowing later positions to use a summary of the
   entire draft prefix." Adds 5.1M params.

**The bigger lever is the data, not the architecture [verified]:** the 50K-example ablation with
the Full Recipe reached only **3.797** (+0.280). The jump to **4.915** came from **500,857
on-policy examples**, "produced from the source prompts using GLM-5.2-NVFP4 at high reasoning
effort", 16,384-token generation limit, drawn from 1.93M source prompts across code, math, chat,
RAG, tool use and long-context. RedHat's public DSpark head baseline is cited at 3.96.

**Expected effect [inferred]:** accepted length 3.5 → 4.9 is roughly a **1.4× decode speedup** if
verify cost per step is flat, which at batch 1 it approximately is. Applied to our 365 tok/s that
is the single biggest headline number available in this document. **Caveat:** Makora publishes
**no hardware, batch size, or tok/s** for this post — only accepted length. Accepted length is not
tok/s; a deeper draft costs more draft-model time and more verify width. Our 3-1-4 tree geometry
must be re-tuned jointly.

**Difficulty: medium.** Architecture changes are small. **The 500K on-policy generation run is the
real cost** and is pure GPU-hours on our own cluster — but it is embarrassingly parallel and needs
no new research.

### 5. SlimSpec — low-rank draft LM head, keeps the full vocabulary

**Mechanism [verified]** (<https://nebius.com/blog/posts/slimspec-faster-speculative-decoding-without-cutting-the-vocabulary>):
factorise the draft model's LM head. Instead of `logits = W_full · h` with `W_full ∈ ℝ^(V×d)`,
use `logits = W_up · W_down · h` with `W_up ∈ ℝ^(V×r)`, `W_down ∈ ℝ^(r×d)`, `r < d`. Cost becomes
`O(Vr + rd)` — roughly `r/d` of baseline when `V >> d`.

**Why it beats vocabulary truncation [verified]:** methods like VocabTrim and FR-Spec cut the
vocabulary, which "impos[es] a hard acceptance ceiling" and creates train/test mismatch — the
drafter is "optimized for a distribution it will never actually be evaluated against". SlimSpec
"keeps full-vocabulary support and reduces the cost of the projection itself".

**Measured [reported]:** ~**4–5× reduction in LM-head latency** vs full-vocabulary baseline,
against only ~60% reduction for VocabTrim/SpecVocab, while "keeping ρ_τ close to 1" (acceptance
essentially unchanged). Config: Llama-3.1-8B-Instruct, GPT-OSS-20B, Qwen3-30B on **H200**, vLLM
0.17.1, **batch sizes 1 and 64**, MT-Bench/HumanEval/GSM8K.

**Expected effect [inferred]:** GLM-class vocabularies are large (~150K). At batch 1, the draft
LM head is a fully bandwidth-bound `V×d` matvec executed **once per draft token** — with our 3-1-4
tree that is several per step. This is a genuinely cheap win and it *compounds* with #4: a deeper
DSPARK draft makes the LM head cost more, so slimming it is what makes deeper drafting affordable.

**Difficulty: low-medium.** Pure draft-model surgery, no serving-engine change, no target-model
risk. **Do this alongside #4.**

### 6. Aurora — online adaptive speculator, MIT licence, already integrates with SGLang

**Mechanism [verified]** (<https://www.together.ai/blog/aurora>, repo
<https://github.com/togethercomputer/aurora>): decouple an inference server (SGLang or vLLM,
running target + draft) from a **training server** that asynchronously pulls accepted/rejected
traces from a distributed buffer, does gradient updates on a draft copy, and **hot-swaps weights
back without service interruption**. Formulated as async RL: draft = policy π, target verifier =
environment, accepted tokens = positive reward, rejected proposals = negative/counterfactual
feedback. Uses "acceptance loss (imitation)" and "rejection loss (Discard Sampling)" plus "a
specialized Tree Attention mechanism". Training orchestrated via Ray.

**Measured [reported]**, MiniMax M2.5 FP8, lookahead 5:

| Batch | Speedup | Acceptance length |
|---|---|---|
| 1 | 1.63× | 2.41 |
| 8 | 1.47× | 2.40 |
| 16 | 1.45× | 2.40 |
| 32 | 1.50× | 2.45 |

Qwen3-Coder-Next-FP8, lookahead 5: batch 1 **1.92× / 3.05**, batch 8 1.74× / 3.10, batch 16
1.60× / 2.96, batch 32 1.57× / 3.00. Headline: "1.25x additional speedup over a well-trained
static speculator" and "1.5x day-0 speedup on recently released frontier" models.

**The strategically interesting part [inferred]:** the AA methodology is **~10k input tokens,
≥1500 output, P50 over a trailing 72 hours**. That is a *stationary, narrow prompt distribution
sustained for days.* An online-adaptive speculator is precisely the tool that converts a narrow
stationary distribution into acceptance rate. Together's own framing for RL workloads
**[verified]** — acceptance rising "from below 10% to above 80% over 1.4k training steps" when
specialising to files being edited (<https://www.together.ai/blog/adaptive-learning-speculator-system-atlas>)
— shows the magnitude available when the distribution is narrow. **I flag this as a real
technique with an obvious dual-use edge; whether specialising to a benchmark's traffic is
legitimate is a call for the team, not for me.** Note it is genuinely lossless in output
distribution: verification "ensures that the quality of the output matches the distribution of
non-speculative decoding".

**Difficulty: medium.** MIT licence, SGLang integration already exists, which matters because we
run an SGLang fork. Setup is "single-node oriented" with example scripts; production needs manual
Ray cluster management.

### 7. Overhead-removal shortlist (do these in week one)

All **[verified]** from the Fireworks MiniMax B200 post unless noted:

- **AOT-export kernels per config, dispatch from C++**, not Python/CuTe-DSL.
- **Fix all tensor shapes per request** (`q_len`, `max_kv_blocks`, `num_splits`) and use a
  persistent kernel to skip padded compute → **eliminate D2H syncs entirely**.
- **Move scattered writes out of the attention epilogue.** "Scattered writes inside the attention
  kernel are instruction-latency-sensitive and easily stall the whole pipeline." Write partial-O
  contiguous and tile-ordered via one bulk-TMA pass; let the *combine* kernel do gathered loads.
- **Let the softmax warpgroup write `(m, l)` stats without synchronising with the store
  warpgroup**, and avoid rescaling O at all in the KV-outer form — "there is one online-softmax
  step and no cross-block correction/rescale."
- **Deterministic atomic-free scheduler**, persistent, grid ≈ #SMs.
- **Respect the 128×128 systolic floor.** From Together's Blackwell ThunderKittens post
  **[verified]**: "a 64 × 64 × 64 GEMM will run at one-quarter the FLOP rate of a
  128 × 128 × 64 GEMM." **[inferred]** This is a direct warning for our MoE expert GEMMs at 19.4%
  — with 8 active experts of 256 at low concurrency, per-expert token counts are small and the
  grouped GEMM tiles may be collapsing below the systolic floor. **Measure the achieved MMA tile
  shape in our grouped GEMM before anything else in the MoE path.**

### 8. Cache-aware prefill/decode disaggregation — for the C64 cost objective only

**Mechanism [verified]** (<https://www.together.ai/blog/cache-aware-disaggregated-inference>):
three tiers instead of two. **Pre-prefill** nodes handle low-reuse (cold) prompts and write KV to
distributed storage; **prefill** nodes serve high-reuse (warm) requests by reading cached KV
blocks over RDMA instead of recomputing; **decode** nodes stay isolated and latency-focused. A
router estimates per-request cache-reusability and steers accordingly, so "large cold prefills
[do not] saturat[e] shared compute". Three-level cache: GPU memory → host DRAM → cluster-wide
RDMA-connected distributed cache. Converts "seconds of compute into hundreds of milliseconds of
transfer".

**Measured [reported]:** **35–40% higher sustainable QPS** vs existing disaggregated designs. On
**B200**, prefill TP4 per node, decode DP with attention sharding across 4 B200s, max inflight
requests capped at 24, 30s ramp + 600s sustain, synthetic coding-agent workload with mixed
warm/cold. Baseline 2P1D flattened at **0.75–0.8 QPS/GPU**; CPD reached **1.1–1.15 QPS/GPU**.

**Relevance [inferred]:** this is a **cost-per-user** technique, not a latency one, and it only
pays when prefix reuse is high. AA's ~10k-input single-stream scenario has essentially no reuse,
so **this will not move our leaderboard number at all.** It is relevant only to the 40.8k tok/s
C64 objective and only if our real traffic has agentic prefix structure.

### 9. What I recommend *against*

- **SMC-SD (Makora)** — genuinely clever, but it is **lossy**, and I am flagging that loudly. See
  the Makora section. Their own framing: it trades "a small, bounded amount of approximation error
  for substantial throughput gains", landing "within 3% of the target model's accuracy". A 3%
  accuracy delta is not acceptable for a leaderboard whose intelligence scores are also published,
  and it is a different product than the one we claim to ship. The **hierarchical composition idea**
  (cheap EAGLE draft feeding a bigger draft feeding the target) is worth understanding; the
  resampling core is not worth the quality risk.
- **TEAL activation sparsity (Together)** — 1.53× at 40% sparsity, 1.8× at 50%, but **A100,
  single-batch only**, and they state plainly that performance "scales poorly at higher batch
  sizes" and Llama-3 degrades more than Llama-2/Mistral **[verified]**
  (<https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models>).
  **[inferred]** Dense-activation sparsity on top of an already-sparse 8-of-256 MoE is unlikely to
  compose well, and the quality budget is better spent on MixFP4.
- **ThunderAgent** — 2× on agentic workloads via program-aware scheduling, but it solves KV-cache
  thrashing across tool-call pauses. **Irrelevant to single-stream AA measurement.**

---

## Together AI

### What they run

Together operate a proprietary **Together Inference Engine** (not a vLLM/SGLang fork) across
Reference / Turbo / Lite tiers, on their own GPU cloud including on-demand B200s. Their
distinguishing asset is that **Tri Dao is their Chief Scientist and the FlashAttention lineage is
theirs**, and they run a dedicated kernels team with an academic pipeline through UCSD, Princeton,
Caltech and Stanford (HazyResearch) **[verified]**
(<https://www.together.ai/blog/inside-the-together-ai-kernels-team>).

They sit at **209 tok/s on the GLM-5.2 AA board** with the best TTFT of any provider there
(0.66s) — **[inferred]** a profile that reads as excellent prefill/scheduling and mid-pack decode,
which is what you would expect from a team whose public strength is attention kernels rather than
speculators.

### The tier structure, technically

**[verified]** (<https://www.together.ai/blog/together-inference-engine-2>) — this is the clearest
statement of what Turbo/Lite actually mean:

- **Together Reference** — full-precision FP16, quality baseline.
- **Together Turbo** — **FP8**, with "the most accurate quantization techniques and proprietary
  innovations". They name **"incoherence processing"** as part of a "multi-dimensional accuracy
  preservation strategy". **[inferred]** Incoherence processing is the QuIP/QuIP# family
  technique — random orthogonal (Hadamard) rotations applied to weights and Hessians to spread
  outlier mass before quantising — which is consistent with their Cornell RelaxML research line
  (QTIP, YAQA). This is the same family as the rotation tricks in QuaRot/SpinQuant.
- **Together Lite** — **INT4**, cost-optimised.

**Their published tier numbers [reported]**, all vs vLLM 0.5.1:

| Claim | Config |
|---|---|
| 2.8–4.5× decoding speedup | Turbo, Llama-3-8B, **1× H100** |
| 2.6–4.3× decoding speedup | Turbo, Llama-3-70B, **8× H100** |
| up to 7× the capacity of vLLM | Turbo |
| beats vLLM FP16 **and** FP8 on 8×H100 by up to 30% | **Lite on 2× A100** |
| 12× cost reduction | Lite vs vLLM FP16 |
| 4× speedup | Reference, 8×H100 |

**Config asymmetry warning:** vLLM 0.5.1 dates this benchmark to mid-2024 and vLLM has moved
substantially since; these ratios should be assumed stale. Batch size and input/output lengths are
not stated for the tier comparisons. Quality claims are AlpacaEval 2.0 win-rate deltas ("up to 1.9
points length-corrected", "+0.29%" vs Groq for Lite) — **[inferred]** AlpacaEval is a weak
instrument for detecting quantisation damage compared to the MMLU-Pro + KL-divergence approach
Makora and Fireworks use.

### FlashAttention-4 — the most technically useful thing they have published for us

**[verified]** (<https://www.together.ai/blog/flashattention-4>; the page cites paper
arXiv:2603.05451 and code at `github.com/Dao-AILab/flash-attention`).

**The framing problem, stated crisply:** from H100 to B200, "BF16 tensor core throughput increases
from 1 to 2.25 PFLOPs, while both the SFU count and shared memory bandwidth remains unchanged."
So attention on Blackwell is bottlenecked by **the exponential unit and shared memory**, not the
tensor cores. Their per-SM roofline for M=N=D=128 on B200: tensor cores 8192 ops/cycle,
**exponential unit 16 ops/cycle**, shared memory 128 bytes/cycle. Forward is "bottlenecked by
compute and exponential"; backward is "bottlenecked by shared memory bandwidth".

**Blackwell features they exploit:**
- **TMEM** — "Each of the 148 SMs has 256 KB of TMEM, an on chip scratchpad wired into the tensor
  cores for warp synchronous intermediate storage."
- **`tcgen05.mma`** — fully asynchronous, accumulates in TMEM, "largest single CTA UMMA tile is
  128×256×16, which is about 2× larger than the largest Hopper WGMMA atom", "launched by a single
  thread, easing register pressure", and can source operand A from TMEM.
- **2-CTA MMA** — one UMMA across a CTA pair in the same cluster spanning both TMEMs, scaling the
  tile to **256×256×16**.

**The two techniques worth stealing directly:**

1. **Software-emulated `exp2`.** Because the SFU is the forward-pass bottleneck, they "distribute
   exp computation across hardware's MUFU and software emulated on FMA" — Cody-Waite range
   reduction decomposing `2^x = 2^n · 2^f`, with polynomial coefficients
   `p0 = 1.0, p1 ≈ 0.6951, p2 ≈ 0.2276, p3 ≈ 0.0771`. **[inferred]** Our DSA indexer at 5.8%
   almost certainly runs a softmax/scoring pass over candidate blocks; if it is SFU-bound this
   applies to the indexer as much as to attention proper.
2. **Conditional rescaling with a dedicated correction warpgroup.** A separate "correction"
   warpgroup "Only rescale[s] when the max jump is large" past a threshold τ, taking non-matmul
   work off the critical path. Forward uses a "ping-pong schedule 2x Q and 2x O tiles per CTA" with
   2 softmax warpgroups.

**Backward-pass details** (less relevant to us, recorded for completeness): overlap "while we
compute softmax for tile j, we already issue the dK and dQ MMAs for tile j−1"; DSMEM exchange
between CTAs partitions dS along the non-reduction axis; halves global atomic reductions for dQ; a
deterministic mode "serializes the global reductions with a semaphore-style lock" and still
reaches **85–90% of nondeterministic throughput**.

**Measured [reported]:** up to **1605 TFLOP/s BF16 on B200 (71% utilisation)**; forward
**1.1–1.3× faster than cuDNN 9.13** and **2.1–2.7× faster than Triton**, across sequence lengths
1K–32K. Note the honest framing: beating cuDNN by 10–30% is a *modest* margin, and they say so.

**Implementation note with real leverage [verified]:** "FA4 is implemented entirely in CuTe-DSL,
CUTLASS' Python kernel DSL... cutting compile times by ~20–30× vs C++ templates." **[inferred]**
For a team iterating on B200 kernels, a 20–30× compile-time reduction is a larger practical
productivity win than most kernel optimisations — but note Fireworks then found they had to
**AOT-export out of CuTe-DSL** to remove launch overhead in production. **Develop in CuTe-DSL,
ship AOT.** That is the combined lesson from both companies.

### ThunderKittens on Blackwell

**[verified]** (<https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus>).
Tile-level DSL; the Blackwell port exposes TMEM tiles and CTA pairs through an `ncta=2` template
parameter on the `mma` instruction:

```cpp
using namespace kittens;
tt d;                      // 128 x 128 FP32 tensor memory tile
__shared__ st_bf a, b;     // 128 x 64 BF16 shared tiles
__shared__ semaphore sem;
mma(d, a, b, sem);         // ncta=2 for CTA-pair execution
```

**Measured [reported]:** GEMMs "running at or near cuBLAS speeds, and up to 2x faster than cuBLAS
GEMMs on H100"; attention "running at near-cuDNN speeds on B200, and up to 2x faster than FA3 on
H100". **Read that carefully — the 2× figures are B200-vs-H100 comparisons, i.e. mostly hardware,
not kernel.** On B200 itself they claim parity with cuBLAS/cuDNN, not a win. Elsewhere they claim
"some of the fastest FP4 and FP8 GEMM kernels available for Blackwell" within one week of GPU
access **[reported]** — unquantified, treat as marketing.

**Design lessons [verified]:** "writing performant kernels on NVIDIA Blackwell GPUs feels a lot
more like programming a dataflow-machine than writing traditional (circa ~2022) CUDA kernels";
tensor cores "behave like 128 × 128 systolics" so M and N must be ≥128; overlap by "launching the
AV MMA's from the previous iteration of the attention loop while starting the QK MMA of the
iteration, and loading the K and V tiles of the next iteration".

**Repos [verified]:** `github.com/HazyResearch/ThunderKittens`, `blackwell` branch —
`kernels/matmul` (BF16 GEMM), `kernels/matmul/FP8_B200/matmul.cu`, `kernels/attn/b200/b200.cu`.

### Speculative decoding lineage

Together have published more speculative-decoding work than anyone else in this document.

**Medusa [verified]** (<https://www.together.ai/blog/medusa>, repo `FasterDecoding/Medusa`):
multiple FFN heads with residual connections on the last hidden state; candidates from the
Cartesian product of per-head top-k; **tree attention** encoding "the dependency graph into the
attention"; **typical acceptance** taking "the minimum of a hard threshold and an
entropy-dependent threshold" rather than importance sampling, worth "a 10% speedup compared to
greedy decoding methods". ~60% top-1 / >80% top-5 accuracy for next-next token on Vicuna; ~**2×**
wall-clock across Vicuna 7B/13B/33B. Trainable in "a few hours to a day... on a single A100-80G".
**[inferred]** Now superseded by EAGLE-3/DSPARK for our purposes, but the *typical acceptance
threshold* is a cheap knob we should confirm we are using.

**MagicDec / long-context speculation [verified]**
(<https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference>).
This one contains a genuinely counterintuitive result with a clean derivation. The conventional
view is that "in the high-throughput regime (i.e., large batch sizes), speculative
decoding... does not make sense, because decoding will be compute-bound". They show this breaks
with long context: **KV-cache operations have constant arithmetic intensity regardless of batch
size, while parameter operations scale with batch size**, so as sequence length grows the forward
pass becomes "dominated by the loading of the KV cache" and **memory-bound again**. Therefore
speculative decoding pays at large batch *and* long context, and **speedups increase with batch
size** because verification time grows faster than draft time.

Their trick: **use the target model as its own draft model with a fixed context window**
(StreamingLLM — sliding window + attention sink, budget 256). "We can afford to use a very large
and powerful draft model — *we can even use the full target model as the draft model*, as long as
it uses a fixed context window."

**Measured [reported]**, 8× **A100** (dated hardware):

| Target | Config | Speedup |
|---|---|---|
| LLaMA-2-7B-32K self-spec | 32K prefill, batch 32, 3 tokens | 1.18× |
| LLaMA-2-7B-32K self-spec | 8K prefill, batch 32, 3 tokens | 2.00× |
| LLaMA-3.1-8B self-spec | 100K prefill, batch 32, 5 tokens | 1.84× |

They also use **adaptive Sequoia trees**, selecting tree size L per sequence length to maximise
`G(L,D) · T_model(b,n,0) / [T_draft(b,n,L) + T_verify(b,n,L)]`. **[inferred]** This is the right
formalism for re-tuning our fixed 3-1-4 geometry: our tree should be a function of context length
and concurrency, not a constant. At C1/10k-input (AA's scenario) versus C64, the optimal L almost
certainly differs, and we are presumably running one geometry for both.

Missing from the post **[verified]** as gaps: no acceptance/rejection rates, no output lengths, no
tree-vs-linear comparison, no non-A100 results, FP8 only "as future work".

**ATLAS [verified]** (<https://www.together.ai/blog/adaptive-learning-speculator-system-atlas>):
dual speculator — a **static** one trained on a broad corpus as a "speed floor", plus an
**adaptive** one taking "rapid, low-overhead updates from real-time traffic" — with a
**confidence-aware controller** that selects between them and "adjusts lookahead for optimal
accuracy and speed". They give the governing variables explicitly: acceptance rate α and draft
speed c, noting "strong speculators (high α, low c) continue gaining from longer lookaheads
(γ = 5+), while weaker configurations plateau early (γ = 3–4)". **[inferred]** That is a direct
statement that our 3-1-4 depth is only correct for a *mediocre* speculator — improving the draft
(steals #4/#5) should be accompanied by *increasing* depth, not holding it fixed.

**Measured [reported]:** DeepSeek-V3.1 fully adapted **500 TPS on 4× B200 at batch size 1**,
2.65× faster than standard decoding, against a **105 TPS FP8 baseline** (4× claimed). Kimi-K2-0905
460 TPS adapted, ~150 TPS out-of-box → 270+ with a custom speculator. Input traffic: Arena Hard.
**Config asymmetries:** 4 GPUs not 8; the 105 TPS FP8 baseline is not identified as any named
engine, so the "4x" is against an unspecified reference. RL-MATH: 60% reduction in overall
training time; acceptance climbing "from below 10% to above 80% over 1.4k training steps".

**Aurora** — see steal #6 above. Paper arXiv:2602.06932, site `aurora-spec-ai.github.io`, test set
of 198 examples on HF **[verified]**.

**Distribution-aware speculative decoding (DAS) [verified]**
(<https://www.together.ai/blog/distribution-aware-speculative-decoding>, paper arXiv:2511.13841),
aimed at RL rollouts rather than serving. Two parts: (a) an **adaptive suffix-tree drafter** built
from recent trajectories **with no gradient updates**, scoring candidates by prefix-match frequency,
where "newly verified tokens are immediately inserted back into the tree", with construction and
cleanup parallelised per problem for "<5% fluctuation in actor update latency"; (b)
**length-aware scheduling** — interleave long requests across ranks to prevent stragglers, start
speculating early on long requests, and classify requests Long/Medium/Short with per-class
speculation budgets where **short requests skip speculation entirely**. Measured: >50% on Math RL
(DeepSeek-R1-Distill-Qwen-7B), ~25% on Code RL (Qwen3-8B), ~30% at 8k, "while matching the baseline
reward curve exactly". **[inferred]** The gradient-free suffix-tree drafter is an attractive
*complement* to a neural draft for our system: near-zero training cost, and it is exactly the right
tool for the repeated-prefix structure of a benchmark harness. The length-aware inter-rank
interleaving is also a rank-skew mitigation, which is our biggest single loss bucket.

### Flash-Decoding

**[verified]** (<https://www.together.ai/blog/flash-decoding-for-long-context-inference>). Split
KV into chunks, compute attention per chunk in parallel writing one extra log-sum-exp scalar per
row per split, then reduce across splits using the LSE to rescale. The motivation is precisely our
regime: "During inference, the query length is typically 1: this means that if the batch size is
smaller than the number of streaming multiprocessors (SMs) on the GPU (108 for an A100), the
operation will only use a small part of the GPU!" Measured on A100 f16: CodeLlama-34B end-to-end
"up to 8x speedups in decoding speed for very large sequences"; micro-benchmarks at B=1
seqlen=65,536 → 64.4 µs, seqlen=131,072 → 106.6 µs, attention itself "up to 50x faster than
FlashAttention". In `flash-attention` ≥2.2 and xFormers ≥0.0.22.

**[inferred]** This is table stakes and we certainly already have split-KV, but the *framing*
matters for the KV-outer decision in steal #1: both Flash-Decoding and KV-outer are answers to
"batch 1 leaves the GPU empty", and KV-outer is the sparse-attention generalisation of the same
idea. Their MiniMax M3 serving post confirms they use the same inversion **[verified]**
(<https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets>):
"each key-value group in the outer loop and calculating attention between query tokens in the
inner loop", with an LSE reduction step — **independently arrived at by both Together and
Fireworks, which is strong evidence the technique is real.**

### Serving MiniMax M3 (Together's version)

**[verified]**, same URL as above. On **B200**. Notable mechanisms:

- **Paged attention integration without rewriting kernels**: build "a page table based on the
  selected blocks, flatten the KV-group dimension into the batch dimension", where "page addresses
  advance by D to choose a virtual page start, while tokens advance by Hkv * D", letting them reuse
  existing GQA kernels. Worth "5% improvement on the decode throughput". **[inferred] This is a
  genuinely clever trick and the kind of thing worth copying for our DSA block selection — express
  sparsity as a page table rather than as a new kernel.**
- **Decode index scoring kernel**: compares query-side index vector against candidate key-side
  index vectors, "reduces each 128-token KV block to a single score", using an "AB-swapped HMMA
  layout" with async copies and bf16. **[inferred] This is the direct analogue of our DSA indexer
  (5.8%) and the AB-swapped HMMA layout is a concrete implementation hint.**
- MSA gains cited as ">9x in the prefilling stage and more than 15x in the decoding stage" vs dense
  **[reported]**.
- Overall: **"81% - 125% increase on various concurrency levels"**, tested with a 60k prefix cache
  at **concurrency 8** — note this is a *cache-heavy agentic* profile, not AA's profile.

### Together's benchmark methodology — unusually honest, and directly comparable to us

**[verified]** (<https://www.together.ai/blog/coding-agent-benchmarks>). This is the one Together
post with a full head-to-head config, and it is against **exactly our engine**:

| Engine | GPUs | Hardware |
|---|---|---|
| Together IE | 4 | B200 |
| TensorRT-LLM | 4 | B200 |
| **SGLang** | **8** | **B200** |

Their stated reason for the asymmetry: "SGLang required 8 GPUs due to higher EAGLE memory
requirements versus TensorRT-LLM." Workload: prompts ~45k–200k tokens, generation averaging 450
tokens (p50 293, p99 2,230), **EAGLE speculative decoding with 3 draft tokens at ~70% acceptance
rate**.

**Results at 2.5M TPM [reported]:** Together IE "delivers 31% more TPS than TensorRT-LLM";
p50 TTFT — Together IE **0.71s**, TensorRT-LLM **1.1s**, SGLang **5.1s**.

**Why this matters to us [inferred]:** (a) it is a *self-serving* comparison where the competitor
was given 2× the GPUs and still lost on TTFT, which either means SGLang's scheduler collapses
under 45k–200k-token prefill saturation or that it was not tuned — we should assume the latter
partly but investigate the former, because **SGLang at 5.1s p50 TTFT vs 1.1s for TRT-LLM is a
7×-scale gap that no amount of untuning fully explains**; (b) their EAGLE acceptance of ~70% with
3 draft tokens is a useful reference point against our 3-1-4; (c) the observation that EAGLE's
memory footprint forced 8 GPUs is a real constraint we live with at TP8.

### Together's negative result — worth reading before we invest in agentic kernel generation

**ParallelKernelBench [verified]** (<https://www.together.ai/blog/parallelkernelbench>, code
`github.com/togethercomputer/ParallelKernelBench`, dataset
`huggingface.co/datasets/togethercomputer/ParallelKernelBench_Problems`). 87 problems from
Megatron-LM, DeepSpeed, TensorRT-LLM, NeMo-RL; each requires "replacing PyTorch + NCCL with a CUDA
kernel that moves data directly over NVLink"; covers tensor, data, context, expert, sequence and
FSDP parallelism.

**Result:** "The best frontier model solves under a third of 87 real-world problems", with only 22
of those beating PyTorch+NCCL zero-shot. Pass@1: GPT-5.5 at **28/87**. Pass@3: **36/87**.
Fast@1→3: 27 faster-than-baseline (31%). Agentic refinement: Gemini 3 Pro reached 35 correct with
26 beating baseline. Models tested include GLM-5.1 and GLM-5.2.

**Failure modes named:** rank coordination and data-partitioning logic; communication mechanism
selection where "most rely on copy engines or SM load/store instructions, while... TMA and NVLS are
almost absent"; stronger models produce "kernels that compile but return incorrect results" or
deadlock, weaker models fail at compile time.

**[inferred] This is the direct counterweight to Makora's agentic-kernel thesis, and it is
specifically about the multi-GPU collective code that is our #2 bottleneck.** The honest reading:
agentic kernel generation works for single-GPU compute kernels (Makora's 600K-kernel dataset is all
single-device Triton) and does **not** yet work for the multi-GPU communication kernels where our
9.2% rank-skew loss lives. **The NVLS/TMA observation is also a hint that NVLS is under-exploited
generally — worth checking whether our collectives use NVLS multicast/reduction on NVSwitch.**

### Together's quantization research

- **YAQA [verified]** (<https://www.together.ai/blog/yaqa>, repo `Cornell-RelaxML/yaqa`):
  weight-only PTQ that **directly minimises KL divergence to the original model** rather than
  layerwise activation error, using "a near-optimal Kronecker-factored approximation of each linear
  layer's Hessian". Key insight: the KL Hessian equals the Fisher Information Matrix, computable by
  backprop without manifesting the O(10¹²)-element matrix; factorise `H ≈ H_O ⊗ H_I` by power
  iteration. Two sketches — A (biased, low-variance), B (unbiased, computes H_O and H_I "in a
  single pass over a dataset"). They note "LDLQ is actually a special case of YAQA's rounding
  algorithm that is theoretically worse than YAQA". Measured: ">30%" KL reduction to the original
  model over existing rounding algorithms, and **lower KL than Google's official Gemma 3 12B
  Instruct QAT model, without finetuning**.
- **QTIP** and **BitDelta** exist in the same Cornell RelaxML line; I did not fetch them in depth.
- **[inferred]** The transferable idea is the **objective**, not the algorithm: optimise
  KL-to-original rather than per-layer MSE, and *report* KL. Both Makora and YAQA use KL as the
  headline quality metric, and it is far more sensitive than MMLU deltas. **We should add
  KL-divergence-to-BF16 to our NVFP4 acceptance criteria.**

### Mixture-of-Agents — noted and dismissed for our purposes

`/blog/together-moa` and `/blog/moaa` exist **[verified via blog index]**. **[inferred]** MoA is an
*inference-time ensembling* technique that multiplies token cost to raise quality. It is orthogonal
to — in fact directly opposed to — both of our objectives. Not relevant; recorded so the gap is not
mistaken for an oversight.

### Together's open-source artifacts and what is actually usable

| Artifact | Licence | Usable for us? |
|---|---|---|
| `Dao-AILab/flash-attention` (FA4, CuTe-DSL, SM100) | BSD-3 | **Yes** — reference for exp emulation, conditional rescale, 2-CTA MMA |
| `HazyResearch/ThunderKittens` (`blackwell` branch) | Apache-ish | **Yes as reference**; adopting the DSL wholesale is a big commitment |
| `togethercomputer/aurora` | **MIT** | **Yes** — integrates with SGLang, which we fork |
| `togethercomputer/ParallelKernelBench` | open | Yes, as an evaluation harness for our own kernel-gen attempts |
| `FasterDecoding/Medusa`, `FasterDecoding/TEAL` | open | Reference only |
| `Cornell-RelaxML/yaqa` | open | Yes — precomputed Hessians and prequantized models on HF |
| Together Inference Engine, Turbo/Lite, CPD, ThunderAgent | **closed** | No |

### What Together has said about Blackwell specifically

- FA4's entire design rationale is Blackwell's asymmetric scaling (tensor cores 2.25×, SFU and
  SMEM bandwidth flat) **[verified]**.
- ThunderKittens Blackwell port: TMEM, `ncta=2` CTA pairs, 128×128 systolic floor **[verified]**.
- CPD benchmarked on B200 with TP4 prefill / DP4 decode **[verified]**.
- Coding-agent benchmark run on B200 against TRT-LLM and SGLang **[verified]**.
- "custom parallelism strategies across 72-GPU meshes, implementing NVFP4 quantization" on GB200
  NVL72 and HGX B200 **[reported]**, unquantified
  (<https://www.together.ai/blog/foundational-research-powering-efficient-inference-at-scale>).
- Voice-agent latency: 281ms → 77ms on Llama-3.2-1B ("3.6x performance improvement and 7.2x better
  unit economics"); Qwen 2.5 1.5B "127ms, down from 292ms on the B200 baseline" **[reported]**
  (<https://www.together.ai/blog/inside-the-together-ai-kernels-team>). **[inferred]** These are
  megakernel-shaped numbers on tiny dense models; do not extrapolate to a 256-expert MoE.
- **Their GTC 2026 post contains no Blackwell engineering** — I fetched it and it is a session and
  model-announcement roundup **[verified]** (<https://www.together.ai/blog/together-ai-at-nvidia-gtc-2026>).
  Recording this so it is not re-researched.

---

## Fireworks AI

### What they run

A fully proprietary stack — "FireAttention" kernels inside a proprietary serving engine, not a
vLLM/SGLang fork. Their public engineering identity is **quantization plus attention kernels**, and
their posts historically carry real numbers with stated configs. They are notable for shipping
GLM-5.2 day zero and for a two-tier Standard/Fast product on the same model.

### The FireAttention series

**FireAttention V1 [verified]**
(<https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs>)
— FP8 on H100, Mixtral-8x7B. "custom CUDA kernel, optimized for Multi-Query Attention models" that
"runs close to the hardware memory bandwidth limit during generation for various batch sizes and
sequence lengths". Weights and KV cache in FP8; **granularity (per-tensor vs per-channel) is not
stated**. One genuinely odd disclosed detail: "Our FP8 implementation runs 3 experts per each token
(as opposed to the default 2)" — i.e. **they spend extra expert compute to buy back quantization
quality**. **[inferred] That is a real design pattern worth considering for our 8-of-256: if NVFP4
costs quality, activating 9 experts may recover it more cheaply than raising precision, since the
MoE GEMMs are 19.4% and adding 1/8 of that is ~2.4% while an FP8 fallback on the same tensors would
cost far more.**

Quality: MMLU 5-shot on 14K examples ("Mixtral achieving 70.6% accuracy"), plus arc-c/winogrande/
GSM8K with the honest caveat that "Because of a small number of examples, arc-c/winogrande/gsm8k
benchmark results differences are only meaningful at ~1%". Benchmarks at prompt length 1K,
50 generated tokens, 8× H100: FP8 vs their own FP16 = "2x improvement of the effective
requests/second"; vs vLLM FP16 = **4×**. Their own caveat, which is the most credible sentence in
the post: "There is no 'one size fits all' regarding LLM performance."

Notable negative results on baselines **[verified]**: vLLM GPTQ Int8 was "single GPU only" due to
multi-GPU incompatibility, and vLLM AWQ Int4 had "total request latencies even for a single
concurrent request... above the 3 sec cut-off".

**FireAttention V2 [verified]**
(<https://fireworks.ai/blog/fireattention-v2-long-context-inference>) — long context. Two changes:
Hopper FP16 **and** FP8 prefill kernels, and a **multi-host deployment mode** "beneficial for use
cases with high traffic and balanced prompt/generation ratios".

| Scenario | Hardware | Models | vs vLLM 0.5.0 |
|---|---|---|---|
| 300-token output, 8K–32K ctx | 8× H100 | Qwen 72B fp16 | ~1.7× throughput, ~3.5× latency |
| 300-token output | 8× H100 | Mixtral 8x7B fp8 | ~5.6× throughput, ~12.2× latency |
| 1200-token output, multi-host | 16× H100 | Qwen 72B fp16 | ~3.7× throughput, ~4× latency |
| 1200-token output, multi-host | 16× H100 | Mixtral 8x7B fp8 | ~8× throughput, ~11.3× latency |

**Config asymmetry, stated by them:** "Multi-host mode enabled for FireAttention only." The 12.2×
and 11.3× latency figures are therefore comparing a multi-host deployment against a single-host
baseline and should be **heavily discounted**.

**Quality methodology is the transferable part [verified]:** they explicitly rejected
needle-in-a-haystack because it "doesn't require any reasoning capabilities", and used **RULER**
instead — multi-needle, variable tracking, and SQuAD-based QA, 100 examples per benchmark.
**[inferred] We should adopt RULER for validating our DSA sparse attention at long context; NIAH
will not detect the failure mode where a sparse indexer retrieves the right block but loses
reasoning-relevant context.**

**FireAttention V3 [verified]** (<https://fireworks.ai/blog/fireattention-v3>) — AMD MI300X port,
attention kernel rewritten from scratch. Recorded for the **portability lessons**, since we are
NVIDIA-only: MI300 has 192GB HBM, 304 CUs, warp size 64, and only "64KB + 32KB L1 vs 256KB combined
on H100" shared memory, plus **no TMA**. Software friction: `__hip_bfloat16` and
`__hip_fp8_e4m3_fnuz` "not auto-converted by hipify"; cooperative-groups API limits forced
alternative reductions. **Their most useful admission [verified]: hipBLASLt limitations make "MoE
models currently non-competitive on AMD."** Results, FP8, 8 GPUs, vs NIM (TensorRT-LLM) and AMD
vLLM: LLaMA 70B (8K in / 256 out) 1.8× avg RPS vs AMD vLLM, up to ~3× vs NIM and ~5.5× vs AMD vLLM
in low-latency, 1.1× minimum latency vs NIM; LLaMA 8B (30K in / 256 out) 1.4× avg RPS, 1.6× minimum
latency.

**FireAttention V4 [verified]** (<https://fireworks.ai/blog/fireattention-v4-fp4-b200>) — the
Blackwell/NVFP4 post, and the most directly relevant of the series.

**Their framing:** FP4 is to Blackwell what "FP16 was for Ampere and FP8 for Hopper".

**Why NVFP4 over MXFP4/MXFP8 [verified]:** "2x FLOPs throughput and needs ~1.5x-2x less memory
reads" vs MXFP6/MXFP8; **NVFP4 block size 16 vs MXFP4 block size 32**; and "NVFP4 allows the use of
mantissa from FP8-based scales". **[inferred] The block-16 + FP8-scale-with-mantissa combination is
exactly the substrate MixFP4 exploits — Fireworks identified the format advantage, Makora found the
free bit inside it.**

**The kernel statement that matters most to us [verified]:** they "fully revamped [their] backend
implementation", and "optimized versions of gemm, grouped gemm, and attention kernels need to be
written from scratch" because "Previous Hopper operations rely on 9.0a architecture, which can't be
executed on Blackwell" — sm90a is forward-*incompatible*, requiring "New TensorCore Gen 5
instructions". **[inferred] This is a hard confirmation that any Hopper-era grouped-GEMM path in
our MoE (19.4%) is not merely suboptimal on B200 but architecturally invalid, and if our fork
inherited sm90a-derived kernels via a compatibility path we are leaving a large fraction of that
19.4% on the floor.**

**Measured [reported]:** ">250 tokens/second on B200" and "3.5X throughput improvement compared to
SGLang H200". Config: **DeepSeek V3 0324, 8 GPUs with NVLink**, comparing "H200 for FP8 precision
on SGLang and B200 for FP4 precision on TRT-LLM".

**Config asymmetries — severe, and they disclose them [verified]:**
1. The 3.5× compares **B200 FP4 against H200 FP8** — different hardware *and* different dtype. A
   large fraction of that is silicon, not engineering.
2. It "disables speculation support ('MTP' in DeepSeek model)" — **speculative decoding off on both
   sides.** Our 365 tok/s is *with* EAGLE 3-1-4, so these numbers are not comparable to ours at all.
3. They admit they were "not able to fully validate TRT-LLM FP4 performance on B200 due to setup
   complexity."

**Quality methodology [verified]:** MMLU and **MMLU Pro with over 10K examples, 5-shot non-CoT**,
with the caveat that this is "not suited for thinking models like R1" and the observation that
"MMLU Pro has much more discriminative power for frontier models compared to MMLU". They claim
FireAttention V4 "outperforms TRT-LLM by a significant margin" on quality at FP4.

**The QAT finding, which is a real and useful negative-then-positive result [verified]:** on a
text-to-SQL dataset (10,000 train / 1,000 eval rows) with DeepSeek V2 Lite Chat as an MoE proxy,
LoRA trained in bf16 then merged — "FP8 QAT outperforms FP4 QAT" initially, but "Both QAT
precisions converge to the same final eval loss" with more training epochs. **[inferred] Direct
implication for us: if NVFP4 costs us quality on GLM-5.2, QAT closes the gap but needs more epochs
than an FP8 QAT run would — budget accordingly rather than concluding FP4 QAT is inferior.**

### The MiniMax M3 Blackwell sparse-attention post — Fireworks' best engineering writing

**[verified]** (<https://fireworks.ai/blog/kernel-optimization-for-minimax-m3-on-nvidia-blackwell>).
Already mined in steal #1 and #7; here is the remaining detail.

**Architecture served:** Hq=64, Hkv=4 (GQA factor 16), D=128, KV partitioned into fixed 128-token
blocks, a lightweight index branch scores blocks and selects **top-16 per GQA group**, main branch
computes exact attention over selected blocks only.

**Four-stage pipeline:** (1) index build — invert per-query selection into a
`(kv_head, kv_block) → queries` CSR map; (2) scheduler — partition into balanced splits across SMs;
(3) sparse attention main kernel — KV-stationary, persistent, emits partials; (4) combine —
LSE-merge each query's partials.

**Warp specialisation in the main kernel:** Load — K/V via **TMA**, Q via **`cp.async` gather**;
MMA — QK and PV on **`tcgen05`**; Softmax — **two dedicated warpgroups**; Output — evacuate PV
accumulator, store partials. Load path uses **3 warps** for gathered Q reads, head-packed, with the
index read overlapped against the previous tile's load, and the Q index gather organised **one per
query-group, not per row**. Store path uses **4 warps** for PV evacuation.

**The I/O cost model, which is the reusable analytical tool [verified]:**

| Path | Bytes |
|---|---|
| Q-outer HBM (dedup) | 4,096·N + 32,768·nsb |
| KV-outer HBM (dedup) | 69,632·N + 32,768·nsb |
| Q-outer L2 (reread) | 528,384·N |

with measured B200 peaks of **24 TB/s L2 and 7.4 TB/s HBM**, and the key subtlety: "While KV fits
L2 (nsb < ~3,200), L2 latency > HBM latency ⇒ L2-bw bound" for Q-outer — so Q-outer is *L2-latency*
bound even when the data fits in L2, and beyond L2 capacity it becomes HBM-bound and worse. They
state the model's limits honestly: "Pure bandwidth roofline (tensor-core and combine compute not
modeled); KV-outer is only moderately HBM-bound (L2 latency ~0.7× HBM latency)."

**Measured [reported]**, single B200, fp8 input / bf16 partial-O:

| Metric | Value |
|---|---|
| Peak attention kernel | **~980 TFLOP/s** at ~4.1 TB/s HBM |
| vs FlashInfer (Q-outer), attention kernel | **1.9–2.4×** |
| vs MiniMax open-source MSA, attention kernel | ~1.6× |
| vs FlashInfer, **full module** | **1.18–1.43×** |
| vs open-source MSA, full module | 1.32–1.41× |

**The honesty is notable and instructive:** they explain the gap between the 1.9–2.4× kernel figure
and the 1.18–1.43× module figure as index mapping + scheduling + combine overhead. **[inferred]
This is exactly the trap we would fall into — optimising the attention kernel and reporting kernel
speedup while the module barely moves. Any DSA work we do must be measured at module level.**

They also document the **difference from MiniMax's own MSA**, which is a design fork worth knowing:
Fireworks stores partial-O tile-ordered and raw and defers scatter/normalisation to the combine
kernel, whereas MSA scatters within the attention epilogue with immediate normalisation; Fireworks
splits load (3-warp `cp.async` gather) and store (4-warp PV evacuation), MSA uses a single 4-warp
group for both. **Fireworks' variant is faster (~1.6×) — so the deferred-scatter design wins.**

Built on the **FlashAttention-4 CuTe-DSL SM100 kernel as foundation** — i.e. Fireworks' best
Blackwell kernel is built on Together's open-source work. Both repos released 2026-07-29:
`github.com/fw-ai/minimax-kernels` and `github.com/MiniMax-AI/MSA/tree/fireworks-msa` **[verified]**.

### GLM 5.2 Fast — the closest published description of our exact problem

**[verified]** (<https://fireworks.ai/blog/glm-5p2-fast>). Mechanisms they name:

- **MoE sharding**: expert parameters, "98% of model", distributed across GPUs to free HBM for KV
  cache and batching, with acknowledged "all-to-all communication" tradeoffs. **[inferred] This is
  EP, and the tradeoff they name — all-to-all — is our 19.6% collectives bucket. They are trading
  the same thing we are.**
- **Sparse attention with "IndexShare"**: "each token runs a cheap indexer over the context,
  selects the most relevant 2,048 prior tokens, and runs the expensive attention on those tokens
  only." **This is the only public number I found for a DSA token budget on GLM-5.2: 2,048
  selected tokens.** The name "IndexShare" implies **sharing indexer results across
  heads/layers/tokens**; the post does not define it further. **[unverified] I could not find any
  Fireworks document defining IndexShare's mechanism.** **[inferred] Given our `index_topk_freq=4`
  — recomputing the index every 4 steps — IndexShare is plausibly the spatial analogue (share
  across heads or layers) to our temporal amortisation. If so, the two compose, and combining them
  could cut our 5.8% indexer cost substantially. This is the single highest-value unknown in this
  report and worth direct experimentation.**
- **Speculative decoding**: "drafting candidate tokens that count only once the full model
  verifies them" — no architecture, depth or acceptance rate given.
- **Prefill/decode separation** so "long agent prefixes do not compete with generation every turn".

**Numbers [reported]:** "446 tok/sec on Artificial Analysis", "2-3x faster than our Standard path",
"about 2x Standard throughput", "consistently ranked as the fastest provider on OpenRouter".
**See the discrepancy flagged at the top of this document — the AA provider table I fetched lists
Fireworks at 97 tok/s.**

**Pricing [verified]** per 1M tokens — the clearest public cost-of-speed datapoint anywhere in this
document:

| Tier | Input | Cached input | Output |
|---|---|---|---|
| Standard | $1.40 | $0.14 | $4.40 |
| Fast | $2.10 | $0.21 | $6.60 |

**Exactly 1.5× for 2–3× the speed**, with cached tokens at a 90% discount. **[inferred] That is the
market's revealed price of latency for this model class, and a useful anchor for our own tiering:
a 2× speed improvement supports roughly a 1.5× price premium, i.e. speed improves margin
super-linearly only if the speedup exceeds ~1.5×.**

**Quality [verified]:** 1M context, "81.8% on SWE-bench Verified", JSON-schema and full BNF grammar
modes, and a good practice worth copying — "before launch, we ran the same eval suite on Standard
and Fast and required them to match."

**Hardware is not disclosed** in either GLM-5.2 post **[verified]**; I checked both.

### Fireworks' other published work

**FireOptimizer [verified]** (<https://fireworks.ai/blog/fireoptimizer>) — adaptation engine across
hardware (accelerator selection, workload distribution), model (quantization, fine-tuning) and
software (request processing, caching) layers. The substantive part is **adaptive speculative
execution**: train custom draft models on the customer's own traced production data or sample
datasets, automatically, "without manual hyperparameter tuning". Numbers **[reported]**: "up to 3x
latency improvements"; a customer case where a generic draft model gave **29% hit rate and was
"1.5x slower"**, while FireOptimizer's custom draft gave **76% hit rate and 2x faster**; Cursor at
"~2x speed improvement". **[inferred] The 29%-vs-76% acceptance gap between generic and
workload-matched drafts is the same finding as Makora's 3.517→4.915 and Together's ATLAS
<10%→>80%. Three independent companies converging on "the draft model's training distribution is
the dominant variable" is the strongest signal in this entire document.**

**Multi-LoRA [verified]** (<https://fireworks.ai/blog/multi-lora>) — thin. "Cross-model
continuous batching" processing requests from multiple LoRAs on the same base model with dynamic
batch sizing, plus dynamic loading with caching so adapter count is "not constrained by GPU
memory". Claims "approach 90% of base model performance" and "100x" cost efficiency vs one GPU per
model. **No adapter counts, kernel details, overhead measurements, or hardware given.** Not useful
to us.

**Disaggregated serving:** Fireworks reference prefill/decode separation in the GLM 5.2 Fast post
but **[unverified] I found no dedicated Fireworks post on disaggregated serving.** Their blog index
is JS-driven and paginates poorly — I fetched `?page=3` and got the same first-page content
**[verified]** — so older technical posts are reachable only by direct slug. The FireAttention V1–V4
and FireOptimizer slugs above are confirmed live.

### Fireworks' open-source artifacts

| Artifact | Licence | Usable? |
|---|---|---|
| `github.com/fw-ai/minimax-kernels` | **Apache 2.0** (BSD-3 portions from FA4) | **Yes** — SM100+, CuTe-DSL, own tests + benchmarks, needs CUDA 13 / `nvidia-cutlass-dsl>=4.5.1` / PyTorch 2.9.0+cu130 / FlashInfer |
| FireAttention V1–V4, FireOptimizer, the serving engine | **closed** | No |

**[verified]** — I read the repo page. It is "architecturally self-contained" but shaped by
Fireworks' design, and its block-sparse GQA formulation will need adaptation for sparse MLA.

### What Fireworks has said about Blackwell specifically

- FP4/NVFP4 is the Blackwell-native precision; sm90a kernels **cannot execute** on Blackwell;
  gemm, grouped gemm and attention all rewritten from scratch for TensorCore Gen 5 **[verified]**.
- Full KV-outer sparse-attention kernel design on B200 with TMA, `cp.async` gather, `tcgen05`,
  warp specialisation, measured L2 24 TB/s and HBM 7.4 TB/s **[verified]**.
- B200 deployments with FireAttention V4 available to enterprise customers **[reported]**.

---

## Makora — the provider at 330 tok/s on NVFP4 for GLM-5.2

**Who they are [verified]:** an inference platform whose thesis is **agentic performance
engineering** — "AI agents across the entire inference stack—from GPU kernels and serving engines
to orchestration and algorithms". Domain `makora.com`; `mako.dev` 302-redirects there; AA lists
them as "Makora"; HF org is `makora-ai`. Products: **MakoraGenerate** (kernel generation, free tool
at `generate.mako.dev`) and **MakoraInference**. Claim: "up to 5x faster" than competitors
**[reported]**, unqualified.

**On the AA board [verified]:** GLM-5.2 (max) NVFP4 at **330 tok/s**, TTFT 0.86s. They also serve
Kimi K3, DeepSeek V4 Flash (306 tok/s), DeepSeek V4 Flash 0731 (261 tok/s), Gemma 4 26B A4B
(165–178 tok/s). All 7 models support JSON mode and function calling, OpenAI-compatible API.

**They are the most important company in this report for us** because they are the only one
publishing mechanisms while sitting at our exact operating point (GLM-5.2, NVFP4, ~330 tok/s).

### DSPARK for GLM-5.2 — see steal #4

**[verified]** (<https://makora.com/blog/dspark-for-glm52>, 2026-07-29). Full detail above. The
ablation table is the load-bearing content:

| Configuration | Accepted length |
|---|---|
| Baseline DSpark | 3.517 |
| RedHat DSpark head (public baseline) | 3.96 |
| 50K training ablations, Full Recipe, epoch 5 | 3.797 |
| **500,857 on-policy examples, block-16, epoch 9** | **4.915** |

**Gap [verified]:** they publish **no hardware, batch size, GPU count, tok/s or TTFT** in this post,
and **no comparison to EAGLE3 or MTP**. Accepted length is the only metric.

**Note on "DSpark" as a shared artifact [verified]:** Baseten independently describe DeepSeek V4 Pro
as shipping "with a DSpark speculator"
(<https://www.baseten.co/blog/inference-engineering-for-deepseek-v4-pro-0813/>), and Makora
reference "RedHat's baseline DSpark". **[inferred] DSpark therefore appears to be an open
speculator-head format/family that multiple parties train against, with Makora's contribution being
the four architectural modifications plus the on-policy data, rather than the format itself.** If
GLM-5.2 ships a DSpark head we may already have a stronger starting point than our EAGLE head.

### SMC-SD — the lossy one

**[verified]** (<https://makora.com/blog/smc-sd>, 2026-04-20). "Sequential Monte Carlo Speculative
Decoding" replaces token-level rejection sampling with "importance-weighted resampling over a
population of N draft particles". Maintain N candidate sequences; each round the draft extends all
N by exactly K+1 tokens; the target scores extensions and samples a bonus token; low-scoring
particles are pruned and high-scoring ones duplicated. "Every round, each particle is extended by
exactly K+1 tokens — no rollback, no truncation." Reweight by importance (p/q), resample "when
weights get lopsided".

**Measured [reported]:** Llama 70B on **4× H100** — **5.2× over autoregressive**, **2.36× over
state-of-the-art speculative decoding (SGLang)**, "within 3% of the target model's accuracy across
reasoning, instruction-following, and coding benchmarks". Batch size not stated.

**The catch, in their own words [verified]:** it trades "a small, bounded amount of approximation
error for substantial throughput gains" via a "tunable Pareto frontier" rather than exact rejection
sampling. **This is not lossless. A 2.36× over SGLang's spec decoding is a headline we cannot match
losslessly, and we should expect competitors quoting numbers like this to be quoting a lossy mode.**

### Hierarchical SMC-SD — the composition idea is worth having

**[verified]** (<https://makora.com/blog/hierarchical-smc-sd>, 2026-06-05). Three tiers: (1) Eagle3
generates short 3–4 token continuations from a co-trained sub-model; (2) SMC-SD's larger draft
(Qwen3-8B) extends those to 16–48 tokens; (3) the target (Qwen3-32B) scores all particles for
resampling. Eagle3 is used "to speed up the generation of draft particles, which are then scored by
a target model".

**Measured [reported]**, NVIDIA H100:

| Configuration | Throughput | Speedup |
|---|---|---|
| Qwen3-32B baseline | 96.3 tok/s | 1.00× |
| Eagle3-SD only | 121.3 tok/s | 1.26× |
| SMC-SD (N=6, K=16) | 173.1 tok/s | 1.80× |
| **Hierarchical SMC-SD** | **200.2 tok/s** | **2.08×** |

They claim "All the schemes above perfectly preserve target model accuracy on both GSM8k and
HumanEval" — **[inferred] which sits awkwardly against the SMC-SD post's "within 3%" framing;
GSM8k and HumanEval are small enough that a 3% shift may be inside noise. Treat the losslessness
claim as unproven.**

**[inferred] The transferable, non-lossy piece is the hierarchy itself: a cheap EAGLE head
accelerating the *drafting* of a larger, more accurate draft model. That structure is compatible
with exact rejection sampling and would let us run a much stronger draft than 3-1-4 currently
affords — the Eagle3-only row (1.26×) versus the tiered rows shows most of the gain comes from
being able to draft *longer*, which is the same conclusion as ATLAS's γ analysis.**

### MixFP4 — see steal #3

**[verified]** (<https://makora.com/blog/mixfp4>, 2026-06-29). Full detail above.

### The 600K Triton kernel dataset

**[verified]** (<https://makora.com/blog/triton-gpu-latency>, 2026-06-22). HF dataset
`makora-ai/triton-gpu-latency`: **544,028 train / 57,024 test rows**, ~1.97 GB Parquet. Each row is
a reference PyTorch problem (matmul, normalization, attention variants) paired with an AI-generated
Triton kernel; label `y` is measured wall-clock runtime, or `None` for compile failures. **~35%
failure rate** across both splits, deliberately retained as training signal. Reference problems come
from **METR's extended KernelBench**. 30 test problems partitioned across `full_holdout`,
`ninety_holdout`, `half_holdout` regimes. Honest framing: "the complete auto-generated output of our
production agent during its first few months of operation... don't reflect current MakoraGenerate
performance."

Stated uses: SFT on ~363k successful candidates; training **latency reward models** to rank
candidates without execution; correctness verifiers from the failure labels.

**[inferred] The latency-reward-model idea is the genuinely valuable one: a model that predicts
kernel runtime without running it turns kernel search from an execution-bound loop into an
inference-bound one. But pair this with Together's ParallelKernelBench result — this dataset is
entirely single-device Triton, and the evidence says the approach does not currently reach the
multi-GPU collective kernels where our largest loss sits.**

**Also published [verified via blog index], not fetched in depth:** "TrainSpotting: Improving
Inference on AWS Trainium2 through End-to-End Agentic Performance Engineering" (2026-08-12),
touching "vLLM, custom kernels, and more".

---

## The other providers on the board

### Baseten — 247 tok/s (FAST) / 110 tok/s on GLM-5.2

**[verified]** (<https://www.baseten.co/blog/inference-engineering-for-deepseek-v4-pro-0813/>).
They run a "proprietary inference engine within the Baseten Inference Stack" — explicitly **not**
TensorRT-LLM, SGLang or vLLM. For DeepSeek V4 Pro: **native MXFP4 weights**; the model "ships with
a DSpark speculator"; they tune "from parallelism to KV cache allocation to prefill-decode worker
ratio". Pricing $1.32 / $0.132 cached / $3.96 output per 1M.

**The single most useful sentence they publish [verified]:** the "native speculator is good for
most common use cases" but "we would want to train a new speculator for most dedicated
deployments." **[inferred] Fourth independent confirmation of the draft-distribution finding.**

**No hardware, no TP/EP values, no tok/s, no TTFT, no kernel detail.** Baseten's blog index
**[verified]** shows nothing on Blackwell, quantization, kernels or disaggregated serving. Their
engineering is real (247 tok/s on FAST is third on the board) but **they do not publish mechanisms**.

### Nebius — 220 tok/s FP4, and the best speculator-training engineering published anywhere

Nebius are the surprise of this research: their blog carries substantive, reproducible engineering
on **training** speculators, which is precisely the bottleneck for steal #4.

**SlimSpec** — see steal #5 **[verified]**.

**Training speculative decoders [verified]**
(<https://nebius.com/blog/posts/training-speculative-decoders>, 2026-08-06). Two bottlenecks, both
solved:

1. **Streaming Cross Entropy.** The logits tensor for Llama 3.1 8B at batch 16, seqlen 32k is
   "16 × 32768 × 128256 × 4 bytes ≈ 250.5 GiB", ~501 GiB with gradients. Fix: chunk the tokens —
   "we can process tokens in smaller chunks independently, sum their losses, and divide by the total
   valid tokens at the end". **The order of operations is load-bearing and they say so:** "Align
   hidden states, targets, and masks for the objective; Apply loss masks; Split the aligned token
   set into chunks" — doing it in this order "prevents boundary tokens from being dropped or
   receiving incorrect targets in draft-head scenarios". Peak logits buffer at chunk_size 1024,
   vocab 128k: "1024 x 150_000 x 4 x 2 ~= 0.97GiB". Full-finetune memory: dense CE
   "59GB + 8GB + 31GB ~= 98GB" vs streaming "59GB + 8GB + 1GB ~= 68GB" (GPT-OSS, 8K context).
2. **Block-sparse FlashAttention for EAGLE-3's banded-diagonal mask.** Naive dense implementations
   grow quadratically; instead "represent valid attention regions at block granularity... Blocks
   outside those regions are skipped."

   | seqlen | batch | JAX TFLOPS | Triton FA TFLOPS | JAX memory | Triton memory |
   |---|---|---|---|---|---|
   | 8k | 2 | 15 | 113–135 | 80.75 GB | 4.45 GB |
   | 32k | 4 | **OOM** | 150–162 | 1287.07 GB | 17.78 GB |

**Production result [reported]**, 8× H100 pod: "GPU memory utilization from 44% down to 25%" and
"tokens per second from 6900 to 11200". Training config given: GPT-OSS-20B-BF16, 8K context,
dp=4, cp=1, batch_size=32, EAGLE-3 with 3 decoding heads, 65,536 tokens per GPU.

**[verified]** No code released; conceptual pseudocode and benchmarks only. No discussion of
on-policy vs off-policy data.

**[inferred] This is the enabling infrastructure for steal #4.** Makora's 4.915 accepted length came
from 500K on-policy examples at up to 16,384 tokens. Training an EAGLE-3 head on 16k-token
sequences against a ~150k vocabulary is exactly the regime where dense CE OOMs — Nebius's own table
shows JAX OOM at 32k/batch-4. **We will hit this wall, and Streaming CE + block-sparse EAGLE
attention is the published fix.**

**Also published [verified via index]:** "Train the draft model for your workload" (2026-06-25) on
workload-specific drafts in their Token Factory; "Inside the Nebius + PyTorch DeepSeek V3 recipe:
NVSHMEM and DeepEP" (2026-07-23) on **256 Nebius B200 GPUs** — training, not inference, but
**[inferred] NVSHMEM/DeepEP is directly relevant to our MoE all-to-all**; MLPerf Training 6.0
results on HGX B300 / GB300 NVL72.

### FriendliAI — 189 tok/s, and the right way to think about the leaderboard

**[verified]** (<https://friendli.ai/blog/choosing-your-inference-provider>, 2026-06-04). No kernel
disclosures, but the most useful *methodological* writing I found, and it bears directly on our
AA-chasing objective:

- "A single number cannot describe Large Language Model (LLM) inference performance." Each
  benchmark is "performance under a specific operating condition rather than the full range of
  latency-throughput trade-offs."
- The concrete failure mode: "At the highlighted benchmark point, Provider A appears superior
  because it achieves higher throughput at the same latency. However, comparing the full Pareto
  front reveals a different picture. Across a broader range of operating conditions, Provider B may
  offer better latency-throughput trade-offs."
- "There is no single 'correct operating' point on a Pareto front."

They list their own techniques as "continuous batching technique, along with quantization,
speculative decoding, KV cache offloading, multi-LoRA serving, and autoscaling" — a list, not a
mechanism. They also have a draft-model speculative decoding post (2026-05-19) I did not fetch.

**[inferred] The direct implication for us: our stated problem — 365 tok/s at C1 but a 4.7×
per-stream falloff from C1 to C16 — is a Pareto-front shape problem, and AA measures two points on
it (1-parallel and 10-parallel). A change that helps C1 and hurts C10 could lower our board
position even while improving our headline. Both AA scenarios must be optimised jointly.**

### Databricks — top of the board at 336 tok/s, publishes nothing

**[verified]** I fetched the Databricks blog index and found **no posts on LLM inference
performance, serving optimisation, GLM, speculative decoding, quantization, Blackwell, kernels, or
Mosaic AI serving performance.** Recent posts are Unity AI Gateway, AI coding costs, smart routing,
and AI_Functions in the warehouse.

**This is an honest negative finding and worth stating plainly: the current leader of the GLM-5.2
AA board has published nothing about how they do it.** There is no technique to extract.
**[inferred] Given they acquired MosaicML and its inference team, the capability is in-house and
deliberately unpublished.**

### Deep Infra, Novita, Hyperbolic, Lambda — nothing substantive

I checked each blog index directly. Stating this plainly rather than padding:

- **DeepInfra [verified]** (<https://deepinfra.com/blog>) — model announcements, pricing
  comparisons, data-sovereignty posts. **No engineering content.** One Nemotron post claims "4x
  higher throughput" with no supporting detail. They sit last on the GLM-5.2 board at 64 tok/s
  despite an FP4 label, which **[inferred]** is consistent with running a stock engine.
- **Novita [verified]** (<https://blogs.novita.ai/>) — coding-agent comparisons, model deployments,
  tool integrations. Explicitly "predominantly marketing-oriented". **No engineering content.**
- **Hyperbolic [verified]** (<https://hyperbolic.ai/blog>) — GPU provisioning ("Introducing Forge"),
  H200 pricing, compute planning, a generic "Hopper or Blackwell?" comparison from Aug 2025.
  **Infrastructure and business content, not inference engineering.**
- **Lambda [verified]** (<https://lambda.ai/blog>) — AI infrastructure financing, orchestration
  layers, prompt-injection security, Kubernetes scheduling, MLPerf **training**. **No inference
  engineering.** They do not appear on the GLM-5.2 provider table at all.

Also on the board but not investigated in depth for lack of published engineering: **Wafer**
(173 tok/s but a 5.73s TTFT, a strange profile), **Parasail** (166, NVFP4), **CoreWeave** (188),
**Crusoe** (162, NVFP4, cheapest at $0.36), **SiliconFlow** (101, FP8), **Scaleway** (82),
**GMI** (FP8, no speed listed).

---

## Groq, Cerebras, SambaNova — the batch-1 architectural lesson

Read only for what a GPU can copy. The assignment is right that this is where the batch-1 ceiling
argument lives.

### The one number that explains everything

**[verified]** (<https://groq.com/blog/the-groq-lpu-explained>): "Groq on-chip SRAM has memory
bandwidth upwards of **80 terabytes/second**, while GPU off-chip HBM clocks in at about **eight
terabytes/second**."

Our B200s have ~8 TB/s HBM3e each. **[inferred] Single-stream decode is a weight-streaming problem:
tok/s ≈ memory_bandwidth / bytes_read_per_token. Groq's 10× bandwidth advantage is the entire
architectural story, and it is not something a GPU can close — the LPU has no HBM at all, so
weights must be sharded across enough chips that they fit in aggregate SRAM.** That is also the
constraint: it takes a very large number of LPUs to hold a 256-expert MoE.

### The second lesson: determinism

**[verified]**, same source: "every execution step is completely predictable to the smallest
execution period (also known as clock cycle)", with dataflow "statically scheduled by the software
during compilation, and execut[ing] the same way every time". Inter-chip: "data conveyor belts to
flow between chips as easily as within a chip. There is no need for routers or controllers for
inter-chip connectivity, even at maximum capacity."

**[inferred] This is the transferable half, and it maps exactly onto our largest single loss.**
Rank arrival skew (9.2% of C1) is *non-determinism* — ranks arriving at collectives at different
times because their work is dynamically scheduled, sized by data-dependent quantities, and
interrupted by host syncs. Everything that makes a GPU schedule more static reduces skew:

- fixed tensor shapes per request (Fireworks' D2H elimination)
- AOT-compiled kernels with no JIT/dispatch variance (Fireworks' C++ AOT)
- persistent kernels with grid ≈ #SMs and **atomic-free deterministic** work assignment
  (Fireworks' scheduler)
- ahead-of-time instruction scheduling on the host (the megakernel interpreter)
- length-aware balancing so long requests are interleaved across ranks rather than clustered
  (Together's DAS)

**A GPU cannot copy 80 TB/s of SRAM. It can copy static scheduling, and that is where our 9.2% is.**

### Cerebras

**[verified]** (<https://cerebras.ai/blog/cerebras-inference-3x-faster>). Claims Llama3.1-70B "at an
astounding 2,100 tokens per second", "16x faster than the fastest GPU solution", full response in
"0.4 of a second" vs "1.1 to 4.2 seconds on GPU based solutions" **[reported]**. Mechanisms
disclosed are thin: they "re-written or optimized the most critical kernels such as MatMul,
reduce/broadcast, element wise ops, and activations", "Wafer IO has been streamlined to run
asynchronously from compute", and they implement speculative decoding.

**Notably absent from the page [verified]:** no SRAM/HBM comparison, no bandwidth figures, no weight
streaming mechanism, no per-model config. **[inferred] The one copyable idea is "wafer IO
asynchronous from compute", which on a GPU is just full async H2D/D2H and comms/compute overlap —
we should verify our all-to-all is genuinely overlapped with expert compute rather than serialised,
since that would show up as both collective time and arrival skew.**

### SambaNova

**[unverified] I did not fetch a SambaNova technical source in this session and will not
characterise their RDU architecture from memory.** The gap is recorded honestly rather than filled.

---

## Techniques ranked by transferability to our stack

Effect estimates marked **[inferred]** are mine, derived from the published numbers plus our
measured hotspot breakdown. They are estimates, not predictions.

| # | Technique | Source | Mechanism in one line | Our target hotspot | Expected effect | Difficulty | Evidence quality |
|---|---|---|---|---|---|---|---|
| 1 | **D2H sync elimination + fixed shapes** | Fireworks MiniMax B200 | Fix `q_len`/`max_kv_blocks`/`num_splits` per request; persistent kernel skips padded compute | TPOT overhead, skew | Low single-digit % of C1 **[inferred]** | **Low** | [verified] mechanism, no isolated number |
| 2 | **C++ AOT kernel dispatch** | Fireworks MiniMax B200 | AOT-export per config, drive from C++; remove Python/CuTe-DSL launch overhead | TPOT overhead | Low single-digit % **[inferred]** | **Low** | [verified] mechanism, no isolated number |
| 3 | **SlimSpec low-rank draft LM head** | Nebius | Factorise `W_full` into `W_up·W_down`, keep full vocab | Draft cost in spec decode | 4–5× on draft LM head **[reported]** | **Low-med** | [reported], H200/vLLM, batch 1 & 64 |
| 4 | **Check MoE grouped-GEMM tile ≥128×128** | Together TK Blackwell | 64×64×64 runs at ¼ the FLOP rate of 128×128×64 | MoE GEMMs 19.4% | Diagnostic; potentially large **[inferred]** | **Low** (to measure) | [verified] hardware property |
| 5 | **Audit for sm90a-derived kernels** | Fireworks V4 | sm90a is forward-incompatible; gemm/grouped-gemm/attention must be rewritten for TensorCore Gen 5 | Dense 37.1% + MoE 19.4% | Diagnostic; potentially large **[inferred]** | **Low** (to measure) | [verified] |
| 6 | **On-policy speculator data at scale** | Makora + Fireworks + Together + Baseten | 500K on-policy generations from the target model itself | Decode throughput | Accepted length 3.517→4.915 **[reported]** | **Medium** (GPU-hours) | [reported]; 4 independent confirmations of the principle |
| 7 | **DSPARK head modifications** | Makora | Layer-78 tap, per-layer fusion, position-specific queries, recurrent head | Decode throughput | Part of the 3.517→4.915 **[reported]** | **Medium** | [reported]; no hardware/tok-s given |
| 8 | **MixFP4** | Makora | Scale sign bit selects INT4 vs NVFP4 per 16-block; zero memory cost | Quality budget → FP4 coverage → 56.5% GEMM | +0.8pt MMLU-Pro, −30% KL vs NVFP4 **[reported]** | **Medium** (kernel work) | [reported]; fast kernels not open-sourced |
| 9 | **KV-outer sparse attention** | Fireworks (+ Together, independently) | Invert loop to KV-stationary; full 128×128 MMA; LSE combine | Attention 10.9% + indexer 5.8% | 1.18–1.43× module **[reported]** | **Medium** | [verified] code, Apache 2.0; GQA→MLA port needed |
| 10 | **Deferred-scatter epilogue** | Fireworks MiniMax B200 | Write partial-O contiguous via bulk TMA; combine kernel does gathered loads | Attention | ~1.6× over MSA's in-epilogue scatter **[reported]** | **Medium** | [verified] |
| 11 | **Aurora online adaptive speculator** | Together | Async RL training server hot-swaps draft weights into a live SGLang server | Decode throughput | 1.25× over a strong static speculator **[reported]** | **Medium** | [verified] MIT repo, SGLang integration |
| 12 | **Streaming CE + block-sparse EAGLE attention** | Nebius | Chunk the CE; block-granular mask for banded EAGLE attention | Enables #6/#7 at 16k context | 250GB→1GB logits buffer **[reported]** | **Medium** | [reported]; no code released |
| 13 | **Re-tune tree depth as f(α, c, ctx, concurrency)** | Together ATLAS + MagicDec | γ=5+ pays only for strong speculators; adaptive Sequoia sizing | Decode throughput | Config-only **[inferred]** | **Low-med** | [verified] principle |
| 14 | **FA4 software exp2 emulation** | Together FA4 | Cody-Waite + FMA polynomial, splitting exp between MUFU and FMA | Attention 10.9%, indexer 5.8% | SFU-bound relief **[inferred]** | **Med-high** | [verified] coefficients published |
| 15 | **Conditional rescale + correction warpgroup** | Together FA4 | Rescale only past threshold τ, in a dedicated warpgroup | Attention | Off critical path **[inferred]** | **Med-high** | [verified] |
| 16 | **Sparsity as a page table** | Together MiniMax M3 | Flatten KV-group into batch dim; page addr advances by D, tokens by Hkv·D — reuse GQA kernels | DSA indexer/attention | +5% decode throughput **[reported]** | **Medium** | [reported] |
| 17 | **Gradient-free suffix-tree drafter** | Together DAS | Build drafter from recent trajectories by prefix-match frequency; insert verified tokens back | Decode throughput | 25–50% on RL rollouts **[reported]** | **Medium** | [reported]; different workload |
| 18 | **Length-aware inter-rank interleaving** | Together DAS | Interleave long requests across ranks to prevent stragglers | **Rank arrival skew 9.2%** | Unquantified **[inferred]** | **Medium** | [verified] mechanism |
| 19 | **Activate 9 of 256 experts to buy back FP4 quality** | Fireworks V1 (3-of-2 precedent) | Extra expert compute is cheaper than extra precision | Quality budget | ~+2.4% MoE time for quality **[inferred]** | **Low** (to test) | [verified] precedent at FP8, not FP4 |
| 20 | **RULER instead of NIAH for long-context validation** | Fireworks V2 | Multi-needle, variable tracking, SQuAD QA — requires reasoning | DSA correctness | Methodology **[inferred]** | **Low** | [verified] |
| 21 | **KL-to-BF16 as the quantization acceptance metric** | Makora + Together YAQA | Report KL divergence, not just MMLU deltas | NVFP4 validation | Methodology **[inferred]** | **Low** | [verified] both |
| 22 | **Hierarchical draft (cheap head → big draft → target)** | Makora | EAGLE3 accelerates drafting for a larger draft model | Decode throughput | 1.26×→2.08× composed **[reported]** | **High** | [reported]; losslessness unproven |
| 23 | **Cache-aware PD disaggregation (CPD)** | Together | Pre-prefill tier writes KV to RDMA store; router steers by reuse | **C64 cost only** | +35–40% QPS/GPU **[reported]** | **High** | [reported]; B200 config given |
| 24 | **Megakernel / on-GPU instruction interpreter** | HazyResearch/Together | Whole forward pass in one persistent kernel, paged SMEM, counter sync | Skew + all launch overhead | 50%→78% of HBM BW **[reported]** | **Very high** | [verified]; 1.24B dense TP1 only — **no MoE/TP8 precedent** |
| 25 | **SMC-SD** | Makora | Importance-weighted resampling over N particles | Decode throughput | 2.36× over SGLang spec **[reported]** | **High** | **LOSSY — "within 3% accuracy". Not recommended.** |
| 26 | **TEAL activation sparsity** | Together | Magnitude-prune hidden states | — | 1.53–1.8× **[reported]** | Medium | **A100, batch-1 only; "scales poorly at higher batch sizes". Not recommended.** |
| 27 | **Agentic multi-GPU kernel generation** | Makora (pro) / Together (con) | LLM-generated collective kernels | Collectives 19.6% | **Best model: 28/87 correct, 22 beat baseline [verified]** | High | **Evidence says this does not work yet for multi-GPU.** |

### The three things I would do first

1. **Diagnostics, this week, near-zero risk:** measure achieved MMA tile shapes in our MoE grouped
   GEMM (#4); grep the build for sm90a-derived paths (#5); count D2H syncs per decode step (#1);
   compute `nsb/N` on our DSA traces to test the KV-outer crossover (#9). Each is a few hours and
   each could reveal a large, cheap win.
2. **The speculator programme (#3, #6, #7, #12, #13):** four independent companies — Makora,
   Fireworks, Together and Baseten — converge on the finding that **draft-model training
   distribution dominates every other speculative-decoding variable.** Nebius supplies the training
   infrastructure to do it at 16k context. This is the clearest path to a large headline number and
   it is engineering, not research.
3. **MixFP4 (#8)** as the quality-budget unlock, because it is the only technique here that makes
   *more aggressive quantization* safe, and quantization is what governs 56.5% of our C1 time.

---

## Honest gaps and things I could not source

- **"IndexShare"** — Fireworks name it as their DSA indexing optimisation for GLM-5.2 but never
  define it. **[unverified]** No other Fireworks document I fetched explains the mechanism. This is
  the highest-value unknown in this report given our indexer costs 5.8%.
- **Fireworks GLM-5.2 hardware** — not disclosed in either the Standard or Fast post
  **[verified]** that it is absent.
- **Makora DSPARK hardware, batch size and tok/s** — the post publishes accepted length only
  **[verified]** that these are absent.
- **The Fireworks 446 vs 97 tok/s discrepancy** — I could not determine whether AA lists GLM 5.2
  Fast as a separate provider row. The AA model overview page I fetched did not render the provider
  table, and the provider table I did fetch shows only one Fireworks row.
- **Databricks** — top of the board, publishes nothing. **[verified]** by reading their blog index.
- **SambaNova** — **[unverified]**, not sourced in this session.
- **Fireworks disaggregated serving** — referenced in passing, no dedicated post found. Their blog
  pagination does not work via WebFetch, so older slugs are reachable only by direct URL.
- **Together Turbo/Lite for MoE models** — the tier documentation I found is from the Llama-3 era
  (vs vLLM 0.5.1). Whether GLM-5.2 on Together is Turbo, Lite or neither is not stated, and their
  AA row carries no quantization label.
- **Search constraint:** the WebSearch budget for this session (200 calls) was exhausted before my
  work began, so all discovery after the first two queries was done by fetching blog indexes and
  following links directly. This biased discovery toward posts linked from index pages and may have
  missed older, unlinked technical posts — particularly on the Fireworks blog, whose index is
  JS-driven and does not paginate under WebFetch. I did not search Chinese-language sources
  (Zhihu, WeChat) for this reason; for these specific companies (both US-based) that is a smaller
  loss than it would be for the Chinese labs.

---

## Sources

All fetched and read 2026-08-17.

**Fireworks AI**
- FireAttention V4 / NVFP4 / B200 — <https://fireworks.ai/blog/fireattention-v4-fp4-b200>
- FireAttention V3 / AMD MI300 — <https://fireworks.ai/blog/fireattention-v3>
- FireAttention V2 / long context — <https://fireworks.ai/blog/fireattention-v2-long-context-inference>
- FireAttention V1 / FP8 — <https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs>
- MiniMax M3 sparse attention on Blackwell — <https://fireworks.ai/blog/kernel-optimization-for-minimax-m3-on-nvidia-blackwell>
- GLM 5.2 Fast — <https://fireworks.ai/blog/glm-5p2-fast>
- GLM 5.2 launch — <https://fireworks.ai/blog/glm-5p2>
- FireOptimizer — <https://fireworks.ai/blog/fireoptimizer>
- Multi-LoRA — <https://fireworks.ai/blog/multi-lora>
- Blog index — <https://fireworks.ai/blog>
- Open-source kernels — <https://github.com/fw-ai/minimax-kernels>

**Together AI**
- FlashAttention-4 — <https://www.together.ai/blog/flashattention-4>
- ThunderKittens on Blackwell — <https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus>
- Inside the kernels team — <https://www.together.ai/blog/inside-the-together-ai-kernels-team>
- Megakernel ("No Bubbles") — <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>
- ATLAS — <https://www.together.ai/blog/adaptive-learning-speculator-system-atlas>
- Aurora — <https://www.together.ai/blog/aurora> · <https://github.com/togethercomputer/aurora>
- Speculative decoding for long context (MagicDec) — <https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference>
- Distribution-aware speculative decoding — <https://www.together.ai/blog/distribution-aware-speculative-decoding>
- Medusa — <https://www.together.ai/blog/medusa>
- Flash-Decoding — <https://www.together.ai/blog/flash-decoding-for-long-context-inference>
- Cache-aware PD disaggregation — <https://www.together.ai/blog/cache-aware-disaggregated-inference>
- Serving MiniMax M3 — <https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets>
- ThunderAgent — <https://www.together.ai/blog/thunderagent>
- Coding agent benchmarks (vs TRT-LLM, SGLang, B200) — <https://www.together.ai/blog/coding-agent-benchmarks>
- ParallelKernelBench — <https://www.together.ai/blog/parallelkernelbench>
- TEAL — <https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models>
- YAQA — <https://www.together.ai/blog/yaqa>
- Together Inference Engine 2.0 (Turbo/Lite) — <https://www.together.ai/blog/together-inference-engine-2>
- Foundational research powering efficient inference — <https://www.together.ai/blog/foundational-research-powering-efficient-inference-at-scale>
- GTC 2026 (no engineering content) — <https://www.together.ai/blog/together-ai-at-nvidia-gtc-2026>
- Blog index — <https://www.together.ai/blog>

**Makora**
- Company — <https://makora.com>
- Blog index — <https://makora.com/blog>
- DSPARK for GLM-5.2 — <https://makora.com/blog/dspark-for-glm52>
- SMC-SD — <https://makora.com/blog/smc-sd>
- Hierarchical SMC-SD — <https://makora.com/blog/hierarchical-smc-sd>
- MixFP4 — <https://makora.com/blog/mixfp4>
- 600K Triton kernel dataset — <https://makora.com/blog/triton-gpu-latency>
- AA provider page — <https://artificialanalysis.ai/providers/makora>

**Other providers**
- Artificial Analysis GLM-5.2 provider table — <https://artificialanalysis.ai/models/glm-5-2/providers>
- Baseten, DeepSeek V4 Pro inference engineering — <https://www.baseten.co/blog/inference-engineering-for-deepseek-v4-pro-0813/>
- Baseten blog index — <https://www.baseten.co/blog/>
- Nebius SlimSpec — <https://nebius.com/blog/posts/slimspec-faster-speculative-decoding-without-cutting-the-vocabulary>
- Nebius training speculative decoders — <https://nebius.com/blog/posts/training-speculative-decoders>
- Nebius blog index — <https://nebius.com/blog>
- FriendliAI on benchmarking — <https://friendli.ai/blog/choosing-your-inference-provider>
- FriendliAI blog index — <https://friendli.ai/blog>
- Databricks blog index (no inference engineering) — <https://www.databricks.com/blog>
- DeepInfra blog index (no engineering) — <https://deepinfra.com/blog>
- Novita blog index (no engineering) — <https://blogs.novita.ai/>
- Hyperbolic blog index (no engineering) — <https://hyperbolic.ai/blog>
- Lambda blog index (no inference engineering) — <https://lambda.ai/blog>

**Accelerators (architectural lesson only)**
- Groq LPU explained — <https://groq.com/blog/the-groq-lpu-explained>
- Groq on MoE — <https://groq.com/blog/from-speed-to-scale-how-groq-is-optimized-for-moe-other-large-models>
- Groq blog index — <https://groq.com/blog>
- Cerebras inference 3x faster — <https://cerebras.ai/blog/cerebras-inference-3x-faster>
