# The scoreboard and the money: how inference is benchmarked and what it costs

## What this is

Two halves of one decision problem.

**Half one** reconstructs, from primary sources, how the public leaderboards we
are chasing actually measure — Artificial Analysis (the board we are on),
SemiAnalysis InferenceX (the successor to InferenceMAX, which benchmarks
*our exact model on our exact hardware* and whose config files are public),
OpenRouter, MLPerf Inference, and the vendor self-published boards. For each I
extract the workload shape, the statistic, the measurement plumbing, and then
analyse **what the methodology rewards** and how a server should be configured
to score well on it inside the rules.

**Half two** builds the cost model: real 2025–2026 B200 rental prices from
fourteen sources, power, the arithmetic that converts tok/s into $/1M tokens,
worked examples at our measured operating points, a comparison against the
32-provider GLM-5.2 API market, DeepSeek's published cost/margin disclosure as
the reference point, and the four levers (batching, prefix cache, speculative
decoding, PD disaggregation) that move $/1M tokens — including exactly where
latency optimisation and cost optimisation fight each other.

Labels used throughout: **[verified]** = I read it at the URL given.
**[reported]** = the company claims it, not independently reproduced.
**[inferred]** = my own arithmetic or reasoning. **[unverified]** = could not
source it.

A note on method: my web-search budget was exhausted early, so most of this
comes from direct fetches of primary artefacts — GitHub raw files, the
SemiAnalysis benchmark repo, the SGLang docs repo, provider APIs and pricing
pages. That turned out to be a feature. The InferenceX config YAML and the
SGLang cookbook data files contain far more actionable detail than any blog
post about them.

---

## Bottom line for our system

Ranked by expected value. Effect estimates are labelled by confidence.

### 1. Move the low-latency arm from EAGLE 3-1-4 to 5-1-6. Expected +15–21% single-stream. Difficulty: trivial (four flags)

We run `--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4`. Both SGLang's own verified B200 low-latency
recipe and the SemiAnalysis golden acceptance curve say that is the wrong point
on the curve for C1.

SemiAnalysis measured GLM-5.2's MTP acceptance length against draft length on
SPEED-Bench `coding`, thinking on **[verified]**
([`glm5.2_mtp.yaml`](https://raw.githubusercontent.com/SemiAnalysisAI/InferenceX/main/golden_al_distribution/glm5.2_mtp.yaml)):

| `num_speculative_tokens` (k) | 1 | 2 | **3** | 4 | **5** | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Golden acceptance length | 1.84 | 2.50 | **2.99** | 3.33 | **3.61** | 3.78 | 3.91 | 4.06 |

Going k=3 → k=5 raises accepted tokens per verification step from 2.99 to 3.61,
**+20.7% decode throughput** at C1 **[inferred, from a verified curve]**. The
cost is a verify batch of 6 instead of 4 tokens — 1.24x the verification FLOPs
per accepted token. At C1 we are memory-bandwidth-bound, not FLOP-bound, so
those FLOPs are nearly free. SGLang's verified B200 low-latency cells for both
FP8 and NVFP4 use exactly `5-1-6` **[verified]**; the balanced cells drop to
`2-1-3` and `1-1-2` because at concurrency the verify overhead stops paying.

The curve is still rising at k=8. Sweep k ∈ {4,5,6,7} at C1 and take the knee.

### 2. Close the gap to SGLang's own published B200 NVFP4 number. We are 1.48x behind a public recipe

SGLang publishes, in-repo, a verified benchmark cell for `nvidia/GLM-5.2-NVFP4`
on 8×B200 TP8 at ISL 8192 / OSL 1024 **[verified]**
([`glm-5.2-benchmarks.jsx`](https://raw.githubusercontent.com/sgl-project/sglang/main/docs/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx)):

> low-latency, C1: `ttft_ms: 295, tpot_ms: 1.85`

TPOT 1.85 ms is **540 tok/s single-stream**. We are at TPOT 2.74 ms = 365 tok/s.
TTFT 295 ms vs our 189 ms — we are *ahead* on TTFT and well behind on TPOT.

Caveat that matters: that cell pins acceptance synthetically with
`SGLANG_SIMULATE_ACC_LEN=3.5` at draft length 5-1-6 **[verified]**. Real
acceptance at k=5 on coding is 3.61, so 3.5 is if anything *conservative* — the
number is not inflated by the simulation. The exact flag set is:

```
--tp 8 --quantization modelopt_fp4
--speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6
--chunked-prefill-size 8192 --mem-fraction-static 0.85
```

This is a reproducible target, not a vendor claim. Reproduce it first, then
diff against our fork. **Difficulty: medium** (it is a bake-off, not a rewrite),
**expected effect: this is the whole 365 → 500+ story.**

### 3. Stop chasing TileRT's 500 tok/s as if it were our workload. It is a different benchmark

TileRT's README claims "up to 500 tokens/second" for GLM-5.1-FP8 on 8×B200
**[reported]** ([tile-ai/tilert](https://github.com/tile-ai/tilert)) — but under
"synthetic workloads with 1K input and 1K output". Artificial Analysis's default
workload is **10K input**. Longer prompts slow output speed, which AA states
explicitly.

Worse for the comparison: SemiAnalysis's TileRT config shows the 8×B200 figure
is a **disaggregated two-node deployment where vLLM does the prefill and TileRT
only does decode** **[verified]**, from `nvidia-master.yaml`:

```yaml
glm5.1-fp8-b200-tilert:
  image: ghcr.io/tile-ai/tilert:0.1.5
  runner: b200-multinode
  multinode: true
  disagg: true
  kv-p2p-transfer: nixl
  router: { name: tilert-pd-router, version: "0.1.5.post2" }
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - spec-decoding: "mtp"
        conc-list: [1]
        prefill:
          num-worker: 1
          tp: 8
          additional-settings:
          - "PREFILL_IMAGE=vllm/vllm-openai:v0.26.0"
          - "PREFILL_NODES=1"
        decode:
          num-worker: 1
          tp: 8
          additional-settings:
          - "DECODE_NODES=1"
```

That is 16 B200s (1 prefill node + 1 decode node), at concurrency 1, ISL 1024.
We run 8 B200s aggregated at ISL ~10K. **The correct like-for-like target is
SGLang's 540 tok/s cell (item 2), not TileRT's 500.** Difficulty: free — this
is a re-framing, and it stops us optimising toward a mirage.

### 4. Exploit the fact that AA's output-speed metric excludes TTFT entirely. Difficulty: free, effect: strategic

AA defines **[verified]**:

```
Output Speed = (Total Tokens − First Chunk Tokens)
             / (Time of Final Token Chunk − Time of First Token Chunk)
```

The clock starts at the *first chunk received* and ends at the last. Prefill
time, queueing, TLS handshake, and the entire GCP→us-central1 network round trip
are outside the numerator and outside the denominator. They land on TTFT, which
is a **separate ranked column**.

Consequences for how we tune:
- Prefill cost does not touch the headline "Output Speed" ranking at all. Do not
  trade decode cadence for TTFT to chase the speed board.
- Inter-token jitter *does* count, because the denominator is wall-clock between
  first and last chunk. A single 200 ms stall from a CUDA-graph miss or an
  expert-routing imbalance spike costs the same as 60 slow tokens.
- Chunk *aggregation* matters. If the server batches several tokens into one SSE
  chunk, "First Chunk Tokens" is subtracted from the numerator but the whole
  first chunk's generation time is subtracted from the denominator too, so
  aggregation is roughly neutral — but it is not neutral for TTFT. Emit the
  first token as its own chunk immediately.

### 5. Recognise the P50-over-72h statistic for what it is: 24 samples, and half of them are free

AA tests the 10K workload **8 times per day, ~every 3 hours**, and reports the
**median over the trailing 72 hours** **[verified]**. That is ~24 samples. The
median of 24 samples is insensitive to up to 11 bad ones.

Practical consequences **[inferred]**:
- A deploy, a cold cache, a node restart, or a bad hour costs nothing on the
  board provided the *majority* of the window is healthy. Do not build heroic
  zero-downtime machinery for leaderboard reasons.
- Conversely, a *sustained* regression shows up fully within 36 hours and takes
  36 hours to wash out after a fix. Plan roll-forwards on that clock.
- Because prompts are unique per run and drawn from a deliberately diverse pool
  (AA say this explicitly, *because* "techniques like speculative decoding mean
  that we see variation in output speeds depending on the type of output"
  **[verified]**), acceptance-length variance across prompt types is a real
  source of P50 movement. Tune the draft length against a *mixed* corpus, not a
  coding corpus.

### 6. The "10 parallel prompts" scenario is a 1K-input test, not a 10K one — and it is a different machine

This is the single most misread part of AA's methodology. **[verified]**:

> For our multiple or parallel workload test, we send 10 concurrent requests of
> our **standard 1k input token workload** once per day at a random time.

So the parallel scenario is C10 at **ISL 1024**, sampled **once per day**, not
eight times. Three implications:

- Its P50-over-72h is a median of **three** samples. It is noisy and one bad day
  moves it a lot. This is the one place where a single incident is expensive.
- Because it is 1K input, prefill is cheap and it is almost purely a decode-
  cadence test at batch 10. Our EAGLE config at C10 sits in the awkward zone
  where a long draft (5-1-6) still mostly pays but verification is starting to
  cost. SGLang's own recipe switches to `2-1-3` at the balanced point
  **[verified]**. Sweep k at C10 separately from C1; do not assume one setting
  wins both.
- Batch composition matters. At C10 with EAGLE, the verify batch is 10×(k+1)
  tokens. At k=5 that is 60 tokens per step — still tiny for a B200, so the long
  draft probably still wins at C10. **[inferred]**

### 7. Adopt the SemiAnalysis synthetic-acceptance discipline internally, as a measurement tool

`SGLANG_SIMULATE_ACC_LEN` / `SGLANG_SIMULATE_ACC_METHOD=match-expected` /
`SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` (SGLang ≥ v0.5.16)
**[verified]** pin acceptance to a fixed target so that a measurement isolates
*system* performance from *draft-head* quality.

We should use this as an internal A/B control, not to publish numbers. When we
change a kernel, a communication schedule, or a memory layout, run with
acceptance pinned so run-to-run acceptance noise does not mask a 3% kernel win.
Then run unpinned for the real number. **Difficulty: trivial. Effect: makes
every subsequent optimisation measurable.** Equivalents exist in vLLM
(`synthetic_acceptance_length`) and TensorRT-LLM
(`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, which counts draft tokens only —
**set it to golden AL minus 1**) **[verified]**.

### 8. HiCache is not optional at concurrency on agentic traffic. Difficulty: medium

SemiAnalysis's GLM-5.2 B200 script documents the failure mode precisely
**[verified]**:

> On the 1M-context agentic corpus the live working set outgrows HBM past
> conc 8 (TP8) / 64 (DP8) and the radix hit rate collapses to <0.1 against a
> ~0.97 theoretical ceiling, so every turn re-prefills its whole history.

Their working config, on a node with the same 180 GB/GPU we have:

```
--enable-hierarchical-cache
--hicache-ratio 0.75           # host pool = 0.75 x device pool, ~128 GB/rank, ~1.0 TB total
--hicache-write-policy write_back
--hicache-io-backend direct
--hicache-mem-layout page_first_direct
```

with two documented negative results: a GB-based sizing of
`TOTAL_CPU_DRAM_GB/TP` (8 × 299 GB) **OOM-killed the node**, and DeepSeek-V4's
own `ratio=2` default (2 × 170 GB × 8 = 2.7 TB) also OOMs because GLM-5.2's
device KV pool is far larger — **169.98 GB per rank, replicated on all 8 ranks**
because DSA/MLA keeps complete per-token KV on every rank **[verified]**.

### 9. Our memory headroom setting is probably wrong for B200 + EAGLE

Same script, with the reasoning spelled out **[verified]**:

> B200: 180 GB HBM3e per GPU against B300's 288 GB. The B300 recipe's 0.85
> leaves 43 GB of non-static headroom there but only 27 GB here, and the EAGLE
> draft head's verification-batch activations and the extra CUDA-graph capture
> at 4 draft tokens both come out of that pool on top of GLM-5.2's DSA indexer
> temporaries. 0.83 restores ~31 GB.

So: `--mem-fraction-static 0.83` for B200 + EAGLE + DSA. Note this gets *worse*
if we take recommendation #1 and go to 6 draft tokens — more graph capture, more
verify activations. Budget for it.

### 10. The economics headline: at C1 we are selling tokens at ~110x our marginal cost floor, and at C64 at ~1.1x

At $6.69/GPU-hr (Lambda 8×B200 on-demand, verified) our node costs $53.52/hr.

| Operating point | Output tok/s (node) | $/1M output tokens |
|---|---|---|
| C1, 365 tok/s | 365 | **$40.73** |
| C16, ~1,243 tok/s | 1,243 | **$11.96** |
| C64, 40.8k total tok/s → ~4,533 output | 4,533 | **$3.28** |

The GLM-5.2 API market clears at **$2.40–4.40 per 1M output tokens** (32
OpenRouter endpoints, verified). **C1 serving is 10–17x underwater against
market price. C64 is roughly break-even to modestly profitable.** Single-stream
latency is a *marketing and leaderboard* product, not a profitable one, and
should be budgeted as such. See "Where latency and cost actively conflict".

---

# PART ONE — BENCHMARK METHODOLOGY

## Artificial Analysis

**What they run.** An independent measurement service that hits providers'
public serverless endpoints from the outside and publishes speed, latency,
price and quality leaderboards per model. This is the board we are on, and the
board TileRT's marketing is implicitly aimed at.

### The methodology, in full

All from
<https://artificialanalysis.ai/methodology/performance-benchmarking>
**[verified]**.

**Workload shapes:**

| Workload type | Input | Minimum answer tokens |
|---|---|---|
| 1k input token | ~1,000 | at least 1,000 |
| **10k input token** | **~10,000** | **at least 1,500** — *default on the site* |
| 100k input token | ~100,000 | at least 2,000 |
| Vision | 1 MP image + ~1,000 tokens | 1,000 |

**Load scenarios:**

| Scenario | Definition |
|---|---|
| Single prompt | one prompt at a time |
| Parallel prompts | 10 prompts sent simultaneously |

**Testing frequency:**

- 1k, 10k and vision: **8 times per day, ~every 3 hours**
- Parallel: **10 concurrent requests of the 1k workload, once per day at a
  random time**
- 100k: **once per week**

**Statistic:** median (P50) over the **trailing 72 hours**; the 100k workload
uses a **14-day** window instead.

**Origin:** "Our primary testing server is a virtual machine hosted in Google
Cloud's `us-central1-a` zone." AA name the resulting bias in their own Known
Limitations: "TTFT is sensitive to server location as it includes network
latency… which may advantage or disadvantage certain providers based on their
server locations."

**Sampling parameters:** `temperature: 0.6` for reasoning models, `0` for
non-reasoning, `top_p: 1`, unless the model creator specifies otherwise.

**Tokenisation:** for *performance* benchmarking they count tokens with OpenAI's
`tiktoken` `o200k_base`, deliberately **not** the model's own tokenizer, "so
that the same text is represented as the same number of tokens". For the
*Intelligence Index* they use provider-reported counts instead, because those
drive cost.

**Prompts:** every run uses a unique freshly generated prompt, run against all
endpoints. Content is long-form (articles) crossed with tasks: summarisation,
Q&A generation, comparative analysis, translation. AA state the reason for the
diversity outright: "techniques like speculative decoding mean that we see
variation in output speeds depending on the type of output."

### The four metric definitions, verbatim

- **Time to First Token** = `Time of First Token Arrival − Time Request Sent`.
  "For reasoning models which return reasoning tokens, **this will be the first
  reasoning token**."
- **Time to First Answer Token (TTFAT)** = `Input Processing Time + (Avg.
  Reasoning Tokens / Reasoning Output Speed)`. Measured after any thinking time.
- **Output Speed** = `(Total Tokens − First Chunk Tokens) / (Time of Final Token
  Chunk Received − Time of First Token Chunk Received)`.
- **End-to-End Response Time** = `Input Processing Time + (Avg. Reasoning Tokens
  / Reasoning Output Speed) + (500 / Answer Output Speed)`.
- **Total Response Time for 100 Output Tokens** = `TTFT + 100/Output Speed`,
  computed synthetically.
- **Average Reasoning Tokens** is measured across 60 prompts drawn from MMLU
  Pro, AIME 2025 and LiveCodeBench; where unavailable **they assume 2k reasoning
  tokens**.
- For reasoning models that do not expose reasoning tokens, output speed is
  computed from **the last 80% of answer chunks**.

### The Integrity Terms — the constraint on how far we can tune for the board

AA publish explicit provider requirements **[verified]**. Providers must not:

- detect or fingerprint AA traffic and serve it differently;
- route AA traffic to dedicated, reserved or non-public resources, separate
  hardware, capacity pools, priority queues or regions;
- serve AA a different model, quantization, context length or endpoint config
  than is publicly advertised under the same name;
- "Serve Artificial Analysis traffic at a batch size, concurrency, or load
  configuration that is not representative of what ordinary traffic on the same
  endpoint receives (**for example, running benchmark requests at reduced batch
  sizes to increase per-request speed**)."

Explicitly *allowed*: "autoscaling, generally available caching, or published
tiering that any customer can access", provided it is not applied selectively.

Enforcement includes withholding measurements, re-measuring from an anonymous
account, delisting, suspending across all leaderboards, and **publicly
disclosing** that a provider served non-representative performance. AA say they
routinely measure from independent test accounts alongside the primary one.

**What this means for us.** The legitimate play is to configure the *whole
public endpoint* for low latency and eat the cost, not to special-case AA
traffic. A published low-concurrency tier that any customer can buy is inside
the rules; a hidden one is not. This is exactly the trade Databricks and Makora
appear to be making — see the numbers below.

### Published numbers: the GLM-5.2 (max) provider board

Fetched from
<https://artificialanalysis.ai/models/glm-5-2/providers> **[verified]**.
Config: **10,000-token input workload, single prompt, P50 over trailing 72h,
measured from GCP us-central1-a over the public internet.**

| Provider | Output speed (tok/s) | TTFT (s) | Thinking time (s) | E2E for 500 tok (s) | Uptime | Context |
|---|---|---|---|---|---|---|
| Databricks | **336** | 0.80 | 5.94 | 8.23 | n/a | 1M |
| Makora (NVFP4) | **330** | 0.86 | 6.06 | 8.44 | 95% | 1M |
| Baseten (FAST) | 247 | 1.72 | 8.10 | 11.85 | n/a | 524k |
| Nebius (FP4) | 220 | 1.10 | 9.11 | 12.48 | 100% | 432k |
| Together AI | 209 | **0.66** | 9.59 | 12.64 | 96% | 262k |
| FriendliAI | 189 | 1.29 | 10.58 | 14.51 | 100% | 1M |
| CoreWeave | 188 | 1.39 | 10.63 | 14.68 | 97% | 262k |
| Wafer | 173 | 5.73 | 11.59 | 20.22 | 98% | 205k |
| Parasail (NVFP4) | 166 | 1.00 | 12.07 | 16.09 | 98% | 1M |
| Crusoe (NVFP4) | 162 | 0.95 | 12.38 | 16.42 | n/a | 1M |
| Baseten | 110 | 1.81 | 18.20 | 24.56 | n/a | 261k |
| SiliconFlow (FP8) | 101 | 2.03 | 19.85 | 26.85 | 99% | 1.05M |
| Fireworks | 97 | 1.35 | 20.58 | 27.07 | 100% | 1M |
| Scaleway | 82 | 1.68 | 24.29 | 32.04 | 75% | 262k |
| Novita (FP8) | 79 | 2.04 | 25.37 | 33.75 | 98% | 1.05M |
| DeepInfra (FP4) | 64 | 1.19 | 31.26 | 40.26 | 73% | 1.05M |

AA's own summary: fastest is 5.3x the slowest; blended price varies 5.2x.
Blended price uses a **7:2:1 cache-hit / input / output ratio** "reflecting
general agentic workload patterns" **[verified]** — cheapest blended are
DeepInfra (FP4) and CoreWeave at $0.49/1M, most expensive Scaleway at $2.52/1M.

**Three readings of this table that matter to us:**

1. **The board leaders are at 330–336 tok/s and we are at 365 on our own data.**
   If our 365 holds up on AA's 10K-input, diverse-prompt workload measured over
   the public internet, we take the #1 slot today. The risk is that our 365 was
   measured on a different prompt mix and without the network hop.

2. **TTFT and thinking time are separate columns, and thinking dominates.**
   Makora: TTFT 0.86 s, thinking 6.06 s. At 330 tok/s, 6.06 s of thinking is
   ~2,000 reasoning tokens — exactly AA's default assumption. The E2E column is
   therefore ~72% reasoning-token volume for the fast providers. **Reasoning
   volume does not affect TTFT at all** (TTFT stops at the first *reasoning*
   token) and does not affect Output Speed (measured on answer chunks, or the
   last 80% of them). It only moves TTFAT and E2E.

   GLM-5.2's chat template defaults to `reasoning_effort: Max` when the client
   passes nothing, and SGLang's cookbook documents that **only `"high"` lowers
   it — every other value including `"low"` and `"medium"` falls through to
   `Max`** **[verified]**. AA pass no `chat_template_kwargs`. So every provider
   on that board is serving Max-effort reasoning, which is why the thinking
   column is so large and so tightly clustered. We cannot game this without
   changing the served model's behaviour for all customers.

3. **Wafer's TTFT of 5.73 s against Together's 0.66 s is the network-plus-prefill
   spread.** For a 10K-token prefill on 8×B200, SGLang's published TTFT is
   295 ms (NVFP4 low-latency, ISL 8192, cold cache) **[verified]**. A
   trans-continental TLS + HTTP round trip from us-central1 is order 30–80 ms
   **[inferred]**. Everything above ~400 ms is queueing, cold cache, or
   admission control. Our 189 ms TTFT is excellent; the exposure is that AA
   measures over the public internet from Iowa, so **our endpoint's physical
   location is worth 100–300 ms of TTFT ranking** **[inferred]**.

### Intelligence-side methodology (relevant because quantization is our lever)

AA Intelligence Index v4.1.1 weights nine evals: Agents 34% (GDPval-AA v2 20%,
τ³-Banking 14%), Coding 24% (Terminal-Bench v2.1 16%, SciCode 8%), General 18%
(AA-LCR 6%, AA-Omniscience 12%), Scientific Reasoning 24% (HLE 12%, GPQA Diamond
6%, CritPt 6%) **[verified]**. Temperature 0/0.6 as above; non-reasoning models
capped at 16,384 output tokens, reasoning models at the creator-disclosed max.
Repeat counts vary from 1 (GDPval) to 5 (τ³-Banking, GPQA Diamond, CritPt).
Claimed 95% CI on the index is <±1%.

This matters because our NVFP4 build is a speed lever with an accuracy cost. AA
note under Known Limitations that they are "moving towards full disclosure of
quantization methods" and already tag providers `(NVFP4)`, `(FP4)`, `(FP8)` on
the board. NVIDIA's NVFP4 build quantizes only the MoE experts' linear weights
and activations, leaving the shared expert unquantized, and is claimed within
~1 point of FP8 on GPQA Diamond, SciCode and IFBench **[reported]**, via the
SGLang cookbook. SGLang's own B300 NVFP4 cell measures AIME25 at 89.58 vs the
variant default 87.7 **[verified]** — i.e. NVFP4 scored *higher* there, which is
within run-to-run noise but at least not evidence of collapse.

### Open-source artifacts

AA publish their performance prompt set at
`https://artificialanalysis.ai/downloads/methodology/performance-prompts.xlsx`
**[verified — link present on the methodology page; I did not download it]**.
That is worth pulling: it is the closest available proxy for the actual scoring
corpus, and acceptance length is workload-dependent.

### On Blackwell/B200 specifically

AA say nothing about hardware. They measure endpoints, not machines. The board's
`(NVFP4)` / `(FP4)` / `(FP8)` tags are the only precision signal, and there is no
disclosure of GPU type, node count, or TP degree by any provider.

---

## SemiAnalysis InferenceX (formerly InferenceMAX)

**This is the most valuable source in this document.** It benchmarks GLM-5.2
NVFP4 on 8×B200 with SGLang — our exact configuration — and every launch script,
flag, env var and design-decision comment is public.

**Repo moved.** `github.com/InferenceMAX/InferenceMAX` is archived and points to
**<https://github.com/SemiAnalysisAI/InferenceX>** **[verified]**. Live dashboard
at <https://inferencex.semianalysis.com/> (also `inferencex.com`).

**What they run.** A continuously-executing benchmark fleet: GB300 NVL72, GB200
NVL72, B300, B200, H200, H100, MI355X/MI325X/MI300X, RTX6000 Pro, with TPUv7x,
MI455 UALoE72, Vera Rubin NVL72 and Rubin NVL8 listed as coming **[verified]**.
Frameworks: SGLang, vLLM, TensorRT-LLM, plus Dynamo and TileRT as routers/
runtimes. Models currently include DeepSeek V4 Pro 1.6T, Qwen3.5-397B-A17B,
MiniMax-M3, Kimi-K3 2.8T, GLM-5.1, GLM-5.2, and Qwen3.8 2.4T listed as TBD.
Results are produced by public GitHub Actions runs so every datapoint is
traceable to a workflow run.

### Scenario definitions

From `MODELS.md` **[verified]**:

| Scenario | ISL/OSL | Status |
|---|---|---|
| **Agentic coding** | long-context, multi-turn realistic traffic trace replay **with sub-agents** | Active; new models onboarded here only, **speculative decoding enabled only** |
| **Single-turn 8k1k** | 8192 / 1024 | Active — the primary fixed-sequence-length scenario |
| Single-turn 1k1k | 1024 / 1024 | Deprecated for all models since 2026-07-17 (#2263) |
| Single-turn 1k8k | 1024 / 8192 | Deprecated for all models since 2026-03-27 (#911) |

The fixed-seq-len harness is `benchmark_serving.py` with **[verified]**:

```
--dataset-name random --random-input-len $ISL --random-output-len $OSL
--random-range-ratio $R --num-prompts $N --max-concurrency $C
--request-rate inf --ignore-eos
--num-warmups $((2 * max_concurrency))
--percentile-metrics 'ttft,tpot,itl,e2el'
```

`--request-rate inf` plus `--max-concurrency` means a closed-loop saturating
client, and `--ignore-eos` forces exactly OSL tokens. Concurrency sweeps default
to **powers of two between `conc-start` and `conc-end`** **[verified]**.

### The AgentX trace-replay scenario

Driven by **AIPerf** (`SemiAnalysisAI/aiperf`) with a custom scenario plugin
`inferencex-agentx-mvp` **[verified]**. Key parameters from `benchmark_lib.sh`:

```
aiperf profile --scenario inferencex-agentx-mvp
  --endpoint /v1/chat/completions --endpoint-type chat --streaming
  --concurrency $CONC --benchmark-duration 3600 --random-seed 42
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75
  --warmup-requests-per-lane 10
  --trace-idle-gap-cap-seconds 300
  --warmup-grace-period 1800
  --use-server-token-count
  --num-dataset-entries 393
  --slice-duration 1.0
  --failed-request-threshold 0.10
```

Design points worth stealing:
- **Pre-canned assistant replay by default** — recorded assistant responses drive
  future prompt construction and live server responses are *discarded*. This
  makes the workload deterministic regardless of what the server generates,
  which is the only way trace replay is comparable across engines.
- Each trajectory's warmup start position is **sampled uniformly from [25%, 75%]
  of the trace's turn count**, so the measurement starts mid-conversation with a
  realistic warm prefix, not from turn 1.
- **Idle-gap capping**: source end-to-start delays are preserved but capped
  (10 s whole-system, 300 s per-trajectory) so a real user's coffee break does
  not become benchmark idle time.
- `--use-server-token-count` uses `prompt_tokens`/`completion_tokens` from the
  API rather than client-side re-tokenisation — they note client-side
  tokenisation was pinning CPU at high concurrency.
- The dataset is the `semianalysisai/cc-traces-weka-*` family — real Claude-Code-
  shaped traces with sub-agents, 393 unique trajectories.

### E2E normalized interactivity — the North Star metric

This is a genuinely well-designed metric and we should adopt it internally.
**[verified]**, from `MODELS.md`:

For every valid request *i* with output length `OSL_i` and end-to-end latency
`E2EL_i` (including TTFT and generation):

```
r_i = E2EL_i / OSL_i                       (seconds per output token)
E2E normalized interactivity_q = 1 / percentile_q({r_i})    (tok/s/user)
```

Dashboard default is **P90**. Their own gloss:

```
OSL / E2EL  ≈  1 / (TPOT + TTFT / OSL)
```

i.e. decode interactivity with a penalty for queueing and prefill. Their stated
reasons: it stops a request being scored slower merely for correctly generating
more output; it captures TTFT and decode cadence in one number so you cannot
improve decode by wrecking TTFT; and it stays higher-is-better in tok/s/user.
They also list the limitations honestly — it is not `1/TPOT`, TTFT amortises
over OSL so short outputs are penalised harder, and it cannot tell you whether a
regression came from TTFT or decode.

**Pareto gating.** The frontier computed against E2E normalized interactivity is
canonical. Every other view (E2E latency, conventional interactivity, TTFT)
shows only the **intersection** of the canonical winner set with the true
frontier in those coordinates. "A point cannot appear on any AgentX Pareto
frontier unless it is both a North Star winner and non-dominated on the selected
chart." This deliberately prevents winning by optimising one secondary metric.

### Synthetic acceptance: the fairness rule that changes how spec decode is compared

The most consequential methodology decision in the whole document
**[verified]**:

> Going forward we no longer benchmark non-spec-decode versus spec-decode as an
> A/B. The non-spec-decode arm existed as a neutral baseline back when
> acceptance length wasn't standardized. That is now solved.

`golden_al_distribution/` commits **one golden acceptance-length curve per model,
thinking mode, and draft length**, measured on the **coding** category of
SPEED-Bench's Qualitative split (880 prompts, 80 in each of 11 categories). Every
submission is then pinned to that curve with synthetic acceptance:

| Engine | Mechanism |
|---|---|
| vLLM | `"rejection_sample_method": "synthetic", "synthetic_acceptance_length": X` (unified in vllm#40662) |
| SGLang | `SGLANG_SIMULATE_ACC_LEN`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` |
| TensorRT-LLM | `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` — **counts draft tokens only, set to golden AL − 1**; fractional supported (integer part always accepted, fraction is the probability of one more) |

How a golden curve is collected **[verified]**: dispatch `speedbench-al.yml` with
draft lengths 1–8, thinking modes on/off, `category=coding`, `output-len=4096`;
launch on a B300 runner; for each (thinking mode, draft length) start a clean
vLLM server with *real* MTP/EAGLE decoding; snapshot cumulative accepted-token
and verification-draft counters; run the whole coding category through
`vllm bench serve`; snapshot again; then

```
AL = 1 + (delta accepted draft tokens / delta verification drafts)
```

The `1` is the target model's guaranteed verification token. Rounded to 2 dp.

Their stated rationale: "This policy follows the same broad principle as MLPerf
Inference: prescribe the workload rules needed for comparable system
measurements. InferenceX is evaluating inference-system performance, not the
ability to fine-tune a benchmark-specific speculative head."

**GLM-5.2 golden curve** (`glm5.2_mtp.yaml`, run 28058352479, measured on
`glm-5.2-fp8` on B300 with vLLM MTP; dataset coding, temperature 1.0, top_p 0.95,
output_len 4096, `enable_thinking: true`) **[verified]**:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AL | 1.84 | 2.50 | 2.99 | 3.33 | 3.61 | 3.78 | 3.91 | 4.06 |

Note: "One curve per model: it was collected on the FP8 checkpoint, and the
NVFP4 checkpoint ships the same nextn head."

### KV-cache offloading policy (a real constraint, and a hint about the future)

**[verified]**: only CPU DRAM offloading is permitted in the initial AgentX
policy, and it is optional. Permitted backends: vLLM Connector, LMCache, SGLang
HiCache, Mooncake CPU DRAM Connector, Dynamo KVBM, CPU DRAM P2P pooling. The
budget rule:

```
allowed CPU DRAM = per-server baseline CPU DRAM × (GPUs used / total GPUs in server)
```

with **at most 3 TB per server** for non-standardised SKUs including HGX B200.
Their stated reason: "prevents an unrealistic memory-capacity race in which
hardware vendors ask CSPs and OEMs to install the maximum number of
high-capacity DIMMs, potentially reaching 6 TB per server. Total cost of
ownership (TCO) per accelerator chip will be normalized by the cost of the
server's total CPU DDR5 capacity."

NVMe offloading is explicitly out of scope for now, "tentatively targeted as a
fast follow-up in InferenceX v3.5 or as part of InferenceX v4."

The exact budget formula emitted to benchmark scripts **[verified]**:

```
floor(min(available MiB, 2_861_022) * 1_048_576 * utilization * tp / gpus-per-node / 1e9)
```

Their B200 fleet (`cluster:b200-dgxc`) declares `available-cpu-dram-mib:
3_095_781` and `gpus-per-node: 8`, so at `dram-utilization: 0.80` and TP8 the
budget is capped by the 3 TB decimal limit at ~2,399 GB **[verified]**.

### Engine submission policy

**[verified]** — GLM-5.2's assigned first-class engine is the
**native/upstream SGLang engine**, with **native MTP** as the agreed
plan-of-record draft model. Providers supporting both vLLM and SGLang must
submit the mapped engine *first* before submitting ATOM, TensorRT-LLM or
TokenSpeed. Two exceptions: brand-new hardware SKUs may use a hardware-specific
engine first, and a new model architecture may use another engine if the mapped
one genuinely cannot support the hardware/model pair.

They state the reason plainly: "Labs have reported that proprietary or
hardware-specific engines such as TensorRT-LLM and ATOM do not always provide
every feature their AgentX workloads require."

**Direct consequence for us:** if we want to appear on InferenceX for GLM-5.2,
our SGLang fork is the *right* vehicle. A custom runtime would be a second-class
submission.

### Published configurations for GLM-5.2 on B200 — the exact recipe

From `configs/nvidia-master.yaml` **[verified]**:

```yaml
glm5.2-fp4-b200-sglang-agentic-mtp:
  image: lmsysorg/sglang:v0.5.16-cu130
  model: nvidia/GLM-5.2-NVFP4
  runner: cluster:b200-dgxc
  precision: fp4
  framework: sglang
  multinode: false
  scenarios:
    agentic-coding:
    - dram-utilization: 0.80
      search-space:
      - { tp: 8, spec-decoding: mtp, kv-offloading: dram,
          kv-offload-backend: { name: hicache }, conc-list: [1, 4, 8, 12, 16] }
```

with this design rationale in the config comment **[verified]**:

> TP8-only for memory as well as comparability — the ~433 GB NVFP4 checkpoint
> needs ~54 GB/GPU across 8 B200s and does not fit below 8.

and a documented **negative result** on the B300 sibling:

> A DEP throughput arm is deliberately not included — its measured frontier peak
> is conc 48, well above that cap, and at conc <= 16 attention-DP leaves 2
> sessions per rank and is strictly dominated by TP8, so it would spend a GPU job
> per point to re-measure a worse curve.

**Read that twice.** At concurrency ≤ 16 on 8 GPUs, **attention data-parallelism
is strictly dominated by plain TP8**. If our C1–C16 configuration uses DP
attention, that is a bug.

The B200 launch script `glm5.2_fp4_b200_sglang_mtp.sh` contains several more
hard-won details **[verified]**:

- Speculative flags: `--speculative-algorithm EAGLE --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` — with the comment
  that this is "3 speculative tokens per verification step … and the draft length
  whose golden AL is pinned below" (2.99).
- Low-latency arm (`DP_ATTENTION=false`): `--kv-cache-dtype fp8_e4m3
  --bf16-gemm-backend cutedsl --max-prefill-tokens 8192`,
  `--chunked-prefill-size 8192`.
- High-throughput arm (`DP_ATTENTION=true`): `--chunked-prefill-size 32768`
  because "chunked-prefill-size is a whole-engine budget split across DP ranks:
  the cookbook HT cell's 8192 becomes 1,024 tokens/rank/step under dp8, which
  starves prefill on the 1M-context agentic corpus."
- `--cuda-graph-max-bs` "counts requests, not verification tokens: SGLang's
  spec-decode graph runner scales each captured batch by
  `--speculative-num-draft-tokens` itself." Capped at 64.
- `MAX_RUNNING_REQUESTS = 2 * CONC`, because "AgentX concurrency counts live
  session trees, not individual requests. Allow subagent fan-out to exceed CONC."
- A real NVFP4 packaging bug and its fix: "GLM-5.2-NVFP4 leaves the MTP/nextn
  layer unquantized (`hf_quant_config` excludes `model.layers.78*`), so the EAGLE
  draft MoE is bf16 and `UnquantizedFusedMoEMethod` pins it to the triton runner
  core. Inheriting the target model's FlashInfer all-to-all then has no
  (flashinfer, triton) pre-permute and the engine dies at init." Fix, needed only
  when EP puts an all-to-all in the MoE path:
  `--speculative-moe-a2a-backend none --speculative-moe-runner-backend triton`.
- Under DP-attention they front the ranks with `sglang_router` using
  `--policy consistent_hashing --request-id-headers x-correlation-id --dp-aware`
  so multi-turn sessions stay on the rank holding their radix prefix, driven by
  `AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true`.
- Timeout hygiene: `SGLANG_TIMEOUT_KEEP_ALIVE=900` because uvicorn's 5 s default
  races AIPerf's 300 s client keep-alive and produces ECONNRESET at warmup.

### What they publish about Blackwell/B200

The GLM-5.2 fleet spans B200, B300, GB200 NVL72, GB300 NVL72 and H200. The
GB300 disagg entry is the most informative about where the frontier sits
**[verified]** — it enumerates four topologies at once:

| Topology | Prefill | Decode | Concurrency |
|---|---|---|---|
| low-latency 1P2D | 1 worker TP4 EP4 dp-attn | 2 workers TP4 | 48 |
| low-latency 1P4D | 1 worker TP4 EP4 dp-attn | 4 workers TP4 | 48 |
| low-latency 1P6D | 1 worker TP4 EP4 dp-attn | 6 workers TP4 | 45 |
| high-throughput 2P1D | 2 workers TP8 EP8 dp-attn | 1 worker **TP16 EP16** dp-attn | 128, 192 |

The pattern: **low latency wants many small decode workers (TP4) fed by one
prefill worker; throughput wants the opposite — wide decode (TP16/EP16) fed by
several prefill workers.** The prefill side uses expert-parallel + attention-DP
in every case; the low-latency decode side uses neither.

### Open-source artifacts and what is actually usable

| Artifact | Path | Usable? |
|---|---|---|
| Full benchmark configs | `configs/nvidia-master.yaml` (284 KB), `amd-master.yaml` | **Yes — copy the GLM-5.2 B200 entry directly** |
| Launch scripts | `benchmarks/single_node/agentic/glm5.2_fp4_b200_sglang_mtp.sh` | **Yes — production-grade, heavily commented** |
| Shared harness | `benchmarks/benchmark_lib.sh` (2,227 lines) | Yes — AIPerf invocation, warmup, health monitoring |
| Golden AL curves | `golden_al_distribution/*.yaml` | **Yes — GLM-5.2, DSv4, Qwen3.5, Kimi K2.5, MiniMax M3** |
| Runner hardware facts | `configs/runners.yaml` | Yes — CPU DRAM per fleet, GPUs/node |
| Config schema | `configs/CONFIGS.md` | Yes |
| Result-processing code | `process_result.py`, InferenceX-app TypeScript ETL | Referenced but not read by me — **[unverified]** |
| The SemiAnalysis blog post announcing InferenceMAX | — | **[unverified]** — I could not locate a working URL; the repo is the better source anyway |

---

## OpenRouter

**What they run.** A routing aggregator in front of many providers, not a
benchmark. But its telemetry is a free, continuously-updated read on the
competitive market — and its routing rules define what "winning" means for a
provider selling through it.

### Routing mechanics **[verified]** (<https://openrouter.ai/docs/features/provider-routing>)

1. **Default is price-weighted load balancing, not speed.** Providers with no
   significant outage in the last 30 seconds are eligible; among those, routing
   weight is **inverse-square in price** — a $1/M provider gets 9x the traffic
   of a $3/M provider.
2. Explicit `sort` values disable load balancing entirely: `"price"`,
   `"throughput"`, `"latency"`.
3. Shortcuts: `:nitro` ≡ `sort: "throughput"`, `:floor` ≡ `sort: "price"`.
4. Performance percentiles (p50/p75/p90/p99 for both latency in seconds and
   throughput in tok/s) are computed over a **5-minute rolling window**.

**What this rewards.** OpenRouter's default traffic goes to the cheapest healthy
provider, quadratically. Being fastest wins you only the `:nitro` and explicit
`sort: "throughput"` traffic. The 5-minute window means a brief stall genuinely
costs you traffic — the opposite of AA's forgiving 72-hour median. **A provider
optimising for OpenRouter volume should optimise price and short-window
consistency; a provider optimising for the AA board should optimise sustained
single-stream speed.** These are different machines.

### The GLM-5.2 market, from the OpenRouter API

Pulled live from `https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints`
**[verified]** — 33 endpoints. USD per 1M tokens.

| Provider | Quant | Input | Output | Cache read | Context | Uptime 1d |
|---|---|---|---|---|---|---|
| Novita | fp8 | 0.339 | 1.065 | 0.063 | 1.05M | 99.5% |
| Sail Research | fp8 | 0.500 | 3.150 | 0.115 | 1.05M | 99.9% |
| DigitalOcean | — | 0.700 | 2.200 | 0.105 | 262k | 99.2% |
| GMICloud | fp8 | 0.742 | 2.332 | 0.138 | 1.05M | 99.5% |
| DeepInfra | **fp4** | 0.750 | 2.400 | 0.140 | 1.05M | 99.3% |
| Inceptron | fp4 | 0.750 | 2.900 | 0.170 | 1.05M | 96.8% |
| StreamLake | fp8 | 0.753 | 2.367 | 0.140 | 1.02M | 99.2% |
| CoreWeave | fp4 | 0.760 | 2.420 | 0.140 | 262k | 98.8% |
| Decart | fp4 | 0.768 | 2.560 | 0.128 | 1.05M | 98.5% |
| AkashML | fp8 | 0.770 | 2.420 | 0.143 | 97k | 99.5% |
| Alibaba | fp8 | 0.966 | 3.036 | 0.193 | 1.05M | 99.9% |
| Morph | fp4 | 1.100 | 4.100 | 0.220 | 1.05M | 99.9% |
| SiliconFlow | fp8 | 1.190 | 3.740 | 0.221 | 1.05M | 99.4% |
| **Z.AI (first-party)** | fp8 | **1.400** | **4.400** | **0.260** | 1.05M | 99.9% |
| Baseten / Fireworks / Together / Crusoe / Parasail / Cloudflare / Venice / Friendli / Baidu | mixed | 1.400 | 4.400 | 0.140–0.260 | varies | 92–100% |
| Wafer / Fireworks / Cloudflare / Baseten (long-ctx tier) | mixed | 2.100 | 6.600 | 0.210 | varies | 98–100% |
| Alibaba (premium tier) | fp8 | 2.310 | 7.260 | 0.462 | 1.05M | 100% |

Model-level aggregate: `z-ai/glm-5.2` at **$0.76 in / $2.42 out / $0.14 cache
read**, 1,048,576 context; a `:batch` variant at $0.70/$2.20/$0.13 and a `:free`
tier **[verified]**.

**Reading:** the first-party list price is $1.40/$4.40. The competitive floor is
Novita at $0.339/$1.065 — **76% below list** — and Novita is serving FP8 at 79
tok/s on the AA board. The market is bifurcated: a fast tier priced at or above
list, and a cheap slow tier at a quarter of list. There is no provider that is
both fastest and cheapest. **That gap is the business opportunity, and it is
exactly what our C1 vs C64 cost analysis in part two quantifies.**

Comparable large open MoEs, same API **[verified]**:

| Model | Input | Output | Cache read | Context |
|---|---|---|---|---|
| DeepSeek V4 Pro | 1.32 | 3.96 | **0.044** | 1.05M |
| DeepSeek V4 Flash | 0.14 | 0.28 | 0.028 | 1.05M |
| Kimi K3 | 3.00 | 15.00 | 0.30 | 1.05M |
| Kimi K2.7-Code | 0.71 | 3.50 | 0.15 | 262k |
| Qwen3.8-2.4T-A95B | 2.00 | 6.00 | 0.25 | 1.05M |
| MiniMax M3 | 0.30 | 1.20 | 0.06 | 1.05M |
| GLM-5.1 | 0.966 | 3.036 | 0.179 | 205k |

DeepSeek V4 Pro's cache-read price of **$0.044/M is 30x below its cache-miss
input price of $1.32/M** **[verified]** — the single strongest pricing signal in
the market about how much prefix caching is worth.

### Open-source artifacts

The public JSON APIs (`/api/v1/models`, `/api/v1/models/{id}/endpoints`) are
unauthenticated and return full pricing, quantization tags, context limits, and
uptime percentiles. **Worth scraping on a cron** as a competitive price tracker.
The `latency_last_30m` / `throughput_last_30m` fields exist in the schema but
returned `null` for every GLM-5.2 endpoint at fetch time **[verified]** — do not
depend on them.

### On Blackwell/B200

Nothing. OpenRouter exposes quantization tags (`fp4`/`fp8`) but no hardware.

---

## MLPerf Inference

**What they run.** The industry's most procedurally rigorous inference
benchmark, and the one whose rules everyone else copies. Not a live leaderboard
— submission rounds with a results table.

### Scenarios **[verified]** (`mlcommons/inference_policies`, `inference_rules.adoc`)

| Scenario | Query generation | Duration | Samples/query | Tail latency | Metric |
|---|---|---|---|---|---|
| Single stream | next query on completion | 600 s | 1 | 90% | 90th-pct early-stopping latency estimate |
| **Server / Interactive** | **Poisson arrivals** | 600 s | 1 | **99%** | max Poisson throughput parameter supported |
| Offline | all samples at start, one query | 600 s | ≥24,576 | n/a | measured throughput |
| Multistream | next query on completion | 600 s | 8 | 99% | 99th-pct early-stopping latency estimate |

The **Poisson arrival process** in the Server scenario is the key methodological
difference from every other board here. AA sends one prompt at a time;
InferenceX sends a closed-loop saturating client; MLPerf sends an open-loop
Poisson stream and asks the largest arrival rate at which you still meet a tail
latency bound. Open-loop is far more punishing — a slow request does not throttle
the arrival of the next one, so queueing compounds.

### LLM latency constraints **[verified]**

| Benchmark | Server TTFT/TPOT | Interactive TTFT/TPOT |
|---|---|---|
| Llama3.1-8B | 2000 ms / 100 ms | 500 ms / 30 ms |
| Llama2-70B | 2000 ms / 200 ms | 450 ms / 40 ms |
| Llama3.1-405B | 6000 ms / 175 ms | 4500 ms / 80 ms |
| Mixtral-8x7B | 2000 ms / 200 ms | — |
| **DeepSeek-R1** | **2000 ms / 80 ms** | **1500 ms / 15 ms** |
| GPT-OSS-120B | 3000 ms / 80 ms | 2000 ms / 20 ms |
| Qwen3-VL-235B-A22B | 12 s | — |

Accuracy gates are hard: DeepSeek-R1 must hit 99% of FP16 exact match
(81.9132%); Llama3.1-405B must hit rougeL 21.6666 / exact_match 90.1335 *and*
tokens per sample within 90–110% of reference.

**The DeepSeek-R1 Interactive constraint of TPOT ≤ 15 ms is the closest public
analogue to our objective.** We are at 2.74 ms. The right framing is that MLPerf
Interactive defines "interactive" as ~67 tok/s/user and we are 5.5x past it —
the leaderboard we are chasing is far beyond what the industry's formal
benchmark even asks for.

### Speculative decoding rules **[verified]**

Allowed only for DeepSeek-R1-Interactive, GPT-OSS-120B-Interactive and
Qwen3.6-27B Edge-Agentic. All implementations must use **the reference MTP head
at the reference precision with the reference algorithm and configuration**:
`speculative-num-steps=3, speculative-eagle-topk=1.0`.

Explicitly disallowed: "Implementations that artificially manipulate acceptance
rates … using a different MTP head or continued pre-training of reference MTP
head, quantization of the reference MTP head weights, post-training techniques
on reference MTP head (like fine-tuning, RLHF etc.)"

Note the contrast with InferenceX: MLPerf freezes the *head and config* and lets
acceptance fall where it may; InferenceX freezes the *acceptance target* and lets
you pick the draft length. MLPerf's approach preserves losslessness; InferenceX's
isolates system performance. **[inferred]** InferenceX's is the better tool for
engineering; MLPerf's is the better tool for procurement.

Also disallowed and relevant: caching queries or responses, coalescing identical
queries, hard-coding query counts, "techniques that boost performance for fixed
length experiments but are inapplicable to long-running services", and using
knowledge of the LoadGen implementation to predict lulls.

### Statistical rigor — worth stealing

MLPerf publishes the query counts required for a given confidence
**[verified]**:

| Tail percentile | Confidence | Margin of error | Inferences required |
|---|---|---|---|
| 90% | 99% | 0.50% | 23,886 → 24,576 |
| 95% | 99% | 0.25% | 50,425 → 57,344 |
| 97% | 99% | 0.15% | 85,811 → 90,112 |
| 99% | 99% | 0.05% | 262,742 → 270,336 |

**Compare to AA's ~24 samples per 72-hour window.** AA's P50 is a *median of 24*;
MLPerf needs 24,576 queries for a 90th percentile at ±0.5%. AA's number is a
usable signal about sustained behaviour and a terrible estimator of the tail.
Do not treat an AA delta of a few percent as real.

### Open-source artifacts

`mlcommons/inference` (reference implementations, LoadGen) and
`mlcommons/inference_policies` (the rules). LoadGen's early-stopping estimator
is a genuinely reusable piece of engineering for our own internal harness
**[inferred]**.

### On Blackwell/B200

v6.1 benchmarks include DeepSeek-R1 and GPT-OSS-120B in the datacenter category
with Interactive variants **[verified]**. Round-by-round B200 result tables were
not fetched — **[unverified]**.

---

## Vendor self-published boards: the SGLang cookbook

Not a leaderboard, but the most directly comparable published numbers for our
exact stack, and they are versioned in git rather than in a blog post.

**What they publish.** `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx` plus two
data files: `glm-5.2.jsx` (serve flags per hardware × quant × strategy) and
`glm-5.2-benchmarks.jsx` (measured TTFT/TPOT/throughput per cell)
**[verified]**.

**Measurement conditions**, stated in the doc **[verified]**:

> Speed numbers are measured with `--random-range-ratio 1.0`, `--flush-cache`,
> on `main @ 09ca4fc` (H200 FP8 cells: `v0.5.14 @ 49e384ce`). Spec cells pin the
> EAGLE acceptance length via the serve env `SGLANG_SIMULATE_ACC_LEN`
> (low-latency 5-1-6 = 3.5; FP8 balanced 1-1-2 = 2; NVFP4 balanced 2-1-3 = 2);
> high-throughput has no spec.

Reproduce command **[verified]**:

```
python3 -m sglang.bench_serving --backend sglang --model $MODEL \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL \
  --random-range-ratio 1.0 --num-prompts $N --max-concurrency $C \
  --warmup-requests 64 --flush-cache
```

with `numPromptsByConc: {1: 8, 16: 64, 64: 128, 256: 512, 1024: 2048}`.

**`--flush-cache` means every cell is a cold-cache measurement.** Combined with
`--random-range-ratio 1.0` (fixed lengths, no jitter) this is the *worst case*
for prefix reuse and therefore a conservative TTFT. Our own numbers should be
compared on the same footing.

**Metric definition**, from the file's own comment **[verified]**:

> `tokens_per_sec_per_gpu` = total (in+out) tok/s/GPU (output/GPU × (isl+osl)/osl)

I verified this arithmetic against the NVFP4 C1 cell: E2E = 0.295 + 1023×0.00185
= 2.188 s; 1024/2.188 = 468 output tok/s; /8 GPUs = 58.5; ×9 = **526.6 ≈ the
published 527**. The metric includes prefill tokens. Anyone comparing a
"tok/s/GPU" number to ours must first establish whether input tokens are counted.

### Published numbers — GLM-5.2 on 8×B200, ISL 8192 / OSL 1024

**[verified]**, from `glm-5.2-benchmarks.jsx`. Derived columns are **[inferred]**
from the published TTFT/TPOT/tok-s-per-GPU.

| Quant | Strategy | Conc | TTFT (ms) | TPOT (ms) | Published tok/s/GPU (in+out) | Per-stream (1/TPOT) | Node output tok/s |
|---|---|---|---|---|---|---|---|
| FP8 | low-latency (5-1-6, mfs 0.8) | 1 | 757 | 3.22 | 288 | 311 | 253 |
| FP8 | low-latency | 16 | 3,188 | 9.12 | 1,476 | 110 | 1,312 |
| FP8 | balanced (dp8+deepep, 1-1-2, cp 32768) | 64 | 5,742 | 17.65 | 3,078 | 57 | 2,736 |
| FP8 | balanced | 256 | 18,744 | 32.61 | 5,022 | 31 | 4,464 |
| FP8 | high-throughput (dp8+deepep, no spec) | 1024 | 177,620 | 47.99 | 4,059 | 21 | 3,608 |
| **NVFP4** | **low-latency (5-1-6, cp 8192, mfs 0.85)** | **1** | **295** | **1.85** | **527** | **540** | **468** |
| NVFP4 | low-latency | 16 | 2,491 | 5.43 | 2,289 | 184 | 2,035 |
| NVFP4 | balanced (dp8, 2-1-3, cp 32768, mfs 0.92) | 64 | 5,837 | 12.70 | 3,770 | 79 | 3,351 |
| NVFP4 | balanced | 256 | 16,736 | 30.00 | 5,343 | 33 | 4,749 |
| NVFP4 | high-throughput (dp8, mfs 0.92, mrr 512) | 1024 | 130,174 | 67.12 | 5,305 | 15 | 4,716 |

For context, non-B200 cells: GB300 FP8 low-latency C1 is TTFT **374 ms**, TPOT
4.55 ms; H200 FP8 low-latency C1 is TTFT 668 ms, TPOT 5.05 ms; MI355X FP8 (no
MTP — spec decode is disabled on AMD in the Deploy panel because the draft
kernel is not yet validated on gfx950) is TTFT 634 ms, TPOT 13.56 ms
**[verified]**.

One cell looks anomalous: B300 NVFP4 low-latency at C16 reports TTFT **274 ms**
against B200's 2,491 ms at the same point, which is not physically plausible as
a like-for-like. Treat the B300 NVFP4 rows with suspicion **[inferred]**.

### Configuration tips worth acting on

All **[verified]** from the cookbook prose:

- **Chunked prefill is the single biggest balanced-point lever.** "At long input
  (8K+) the default `--chunked-prefill-size 2048` is too small and leaves the
  balanced point prefill-bound (queueing dominates TTFT). Raising it to
  `--chunked-prefill-size 32768` on the balanced recipe gave roughly **+34–78%
  output throughput and −39–59% TTFT** on 8×H200 and 8×B200 (8K-in / 1K-out) in
  our testing. It is **neutral for high-throughput** (decode-bound there)."
- **KV capacity sets the real concurrency ceiling.** "~60–90 concurrent 8K+1K FP8
  requests fit on a single 8-GPU node, so pin balanced near
  `--max-running-requests 80`." The B200 config comment says ~89.
- **DSA backends auto-select on Blackwell.** "SGLang also auto-selects the
  KV-cache dtype for DSA models — `fp8_e4m3` on Blackwell (B200/GB300/B300,
  which then routes DSA through the TensorRT-LLM backend) and `bf16` on Hopper."
  No `--kv-cache-dtype` flag needed. All recipes run
  `--dsa-topk-backend sgl-kernel`; "other top-k backend choices have not been
  fully validated on GLM-5.2."
- **`index_share_for_mtp_iteration`** in the model config "reuses the DSA
  indexer's topk across draft steps (effective only at
  `--speculative-eagle-topk 1`)". Since our indexer is 5.8% of C1 time and we run
  topk 1, confirm this is active — it should make longer drafts nearly free on
  the indexer axis. **[inferred: this is why 5-1-6 is affordable on a DSA model.]**
- **Tune draft length to accept length, explicitly**: "GLM-5.2's MTP head is
  strong — accept length runs high (4+ in many workloads, near-saturating at 5–6
  in low-latency runs) … while accept length stays close to the draft-token count
  there is headroom to push them higher; if it falls well below, lower them."
- **Prefill context parallelism** for TTFT under long context:
  `--attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave`, with the
  stated trade-off that it "will introduce extra all-gather operation before
  indexer-topk and attention kernels, so it will increase latency for decode (in
  unified deployment) or short prefill."
- **A prefix-cache trap specific to Claude-Code-style clients**: Claude Code
  prepends a per-request attribution block to the system prompt, and "GLM-5.2's
  chat template renders `tools` **before** `system`, so that per-request hash is
  the first token to diverge between turns and the radix prefix cache
  re-prefills the whole system + history every turn."
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` restores reuse. **If we serve agentic
  clients, audit for exactly this class of bug — a single varying byte early in
  the prompt destroys the entire cache hit.**

### Open-source artifacts

`sgl-project/sglang` `docs/cookbook/` and `docs/src/snippets/configs/` are the
whole thing, in git, with per-cell provenance (`sglang_version: main @ 09ca4fc`).
`sgl-eval` (`pip install git+https://github.com/sgl-project/sgl-eval`) is their
accuracy harness for gsm8k/aime25.

---

## LMArena and Arena-adjacent measurement

**Honest finding: LMArena does not measure serving performance at all.**

I fetched <https://lmarena.ai/how-it-works> **[verified]**. The mechanism is:
user types a prompt, gets two anonymous model responses in battle mode, votes for
the preferred one, identities are revealed. The output is a human-preference Elo
leaderboard. There is no latency, throughput, TTFT or cost measurement anywhere
in the described methodology, and no per-provider dimension — models are ranked,
not endpoints.

**Relevance to us is indirect but real**: Arena is where quantization damage
would show up as an Elo regression. If we ship NVFP4 to a public endpoint that
feeds Arena traffic, that is where a silent accuracy loss surfaces. Our internal
gate should be AA's Intelligence Index components plus `sgl-eval` gsm8k/aime25,
not Arena, because Arena is too slow and too noisy to act on.

I did not find any Arena-operated latency or throughput board. **[unverified]**
whether one exists.

---

## Cross-board synthesis: what each methodology rewards

| Board | Workload | Load | Statistic | Window | What it rewards | What it punishes |
|---|---|---|---|---|---|---|
| **Artificial Analysis** | 10k in / ≥1.5k out (default) | 1 prompt; 10-parallel at 1k once/day | **P50** | **72 h** (14 d for 100k) | Sustained single-stream decode cadence; low inter-token jitter | Sustained regressions only; brief incidents are free |
| **InferenceX AgentX** | real multi-turn coding traces w/ sub-agents | closed loop, conc 1–16 (B200 GLM-5.2) | **P90 of s/token, inverted** | per 3600 s run | Balanced TTFT + decode; prefix-cache hit rate; honest spec decode | Any config that wins one axis by wrecking E2E |
| **InferenceX 8k1k** | 8192 / 1024 random, `--ignore-eos` | closed loop, powers of 2 | mean + percentiles | per run | Raw tok/s/GPU on the Pareto frontier | — |
| **OpenRouter** | organic user traffic | organic | p50–p99 | **5 min** | **Price** (inverse-square), then short-window consistency | Any stall inside a 5-minute window |
| **MLPerf Server/Interactive** | fixed dataset, accuracy-gated | **Poisson open loop** | p99 under a latency bound | 600 s, ≥24,576 queries | Tail latency under bursty arrivals; accuracy | Queueing; any accuracy loss |
| **SGLang cookbook** | 8192/1024 random, **cold cache** | closed loop | published point values | single run | Reproducibility | — |
| **LMArena** | organic human prompts | n/a | Elo | continuous | Answer quality | Quantization damage |

### Specific questions the brief asked

**What does the network hop cost?** On AA, it lands entirely on TTFT and
entirely outside Output Speed, by construction of their formula. A round trip
from GCP us-central1-a is order 30–80 ms to a US endpoint and 150–250 ms
trans-Pacific **[inferred]**. On the current GLM-5.2 board, TTFT spans 0.66 s
(Together) to 5.73 s (Wafer) — a 5-second spread that dwarfs any plausible
network term, so **network is not what separates the board; prefill scheduling
and admission control are.** But the E2E ranking column is TTFT + thinking +
answer, so 200 ms of avoidable geography is worth ~2.4% of an 8.4 s E2E figure.
Colocate a US endpoint.

**What does a cold cache cost on a P50-over-72h metric?** Almost nothing, and
this is counterintuitive. AA generates a **unique prompt for every run**, so
every AA measurement is inherently a cold-prefix measurement in the sense that
matters — there is no cross-run prefix to hit. What *can* be warm is the system
prompt and any shared scaffolding. Since AA's 10K prompts are long-form articles
plus a task, the reusable fraction is small. **Prefix caching does not move the
AA board.** It moves the money enormously (see part two). This is one of the
sharpest divergences between the leaderboard objective and the cost objective.

**What does "10 parallel prompts" do to a speculative-decoding configuration?**
At C10 with ISL 1024, the verify batch is 10×(k+1) tokens per step — 40 at k=3,
60 at k=5. On a B200 that is still far below the arithmetic-intensity knee, so
the long draft should still pay **[inferred]**. But SGLang's own balanced recipe
drops to `1-1-2` at C64 and `2-1-3` for NVFP4, so the crossover is somewhere in
10 < C < 64. **Measure it.** And note the second-order effect: AA's parallel test
is P50 of only ~3 samples, so its variance is large and it is the wrong place to
chase small wins.

**How does reasoning-token volume affect reported TTFT?** It does not. AA's TTFT
stops at the first token of the response, "For reasoning models which return
reasoning tokens, this will be the first reasoning token" **[verified]**.
Reasoning volume moves TTFAT and E2E only. On the current GLM-5.2 board that is
~6 s of the ~8.4 s E2E figure for the fastest providers — i.e. **~72% of the E2E
ranking is determined by how much the model thinks, not by how fast we serve it.**
Since AA passes no `chat_template_kwargs` and GLM-5.2 defaults to
`reasoning_effort: Max`, every provider is equally penalised, and the E2E column
is effectively a re-scaled output-speed column. Optimise Output Speed; E2E
follows.

---

# PART TWO — UNIT ECONOMICS

## The cost of a B200-hour, from fourteen real 2025–2026 price points

All **[verified]** at the URLs in Sources. Per-GPU where the source quotes a node
price, I divide by 8 and show the arithmetic.

| Source | SKU | $/GPU-hr | Terms |
|---|---|---|---|
| Nebius | HGX B200 | **$3.95** | preemptible |
| CoreWeave | HGX B200 ($34.11/node) | **$4.26** | spot |
| GCP (via GetDeploying) | B200, 8x | $4.08 | spot |
| AWS `p6-b200.48xlarge` ($42.251/inst) | 8× B200 | $5.28 | spot |
| Vast.ai (via GetDeploying) | B200 | $5.00 / $5.31 | spot / on-demand |
| RunPod | B200 | $5.98 | community cloud |
| **Lambda** | **8× B200 SXM6** | **$6.69** | **on-demand** |
| RunPod | B200 | $6.79 | secure cloud |
| Together | HGX B200 | $6.79 | reserved, 181+ days |
| GCP (via GetDeploying) | B200 | $7.09 | 36-month commit |
| Nebius | HGX B200 | $7.15 | on-demand |
| Together | HGX B200 | $8.19 | on-demand |
| CoreWeave | HGX B200 ($68.80/node) | $8.60 | on-demand |
| Lambda 1-Click Cluster | 16 / 64 / 256+ B200 | $9.86 / $9.36 / $8.87 | 2 weeks – 1 year |
| AWS `p6-b200.48xlarge` ($113.9328/inst) | 8× B200 | $14.24 | on-demand |
| GCP (via GetDeploying) | B200 | $16.11 | on-demand |

**Spread: 4.1x from $3.95 to $16.11.** Two structural observations
**[inferred]**:

1. **Hyperscalers are 2–4x the neocloud rate on-demand and converge on spot.**
   AWS on-demand $14.24 vs spot $5.28 is a 2.7x self-discount. GCP on-demand
   $16.11 vs spot $4.08 is 3.9x. If your workload can absorb preemption, the
   hyperscalers are competitive; if it cannot, they are not.
2. **Lambda's 1-Click Clusters ($8.87–9.86) cost *more* than their on-demand
   8-GPU instances ($6.69).** That is not a mistake — the clusters ship an
   InfiniBand scale-out fabric that a single-node inference deployment does not
   need. **For single-node 8×B200 inference, buy the instance, not the cluster.**

Reference points for older silicon, same sources: H100 SXM $2.69–6.16, H200
$3.59–5.99, B300 $6.94–7.89 (RunPod), GB200 NVL72 $42.00/instance of 4 GPUs =
$10.50/GPU-hr (CoreWeave) **[verified]**.

CoreWeave state "up to 60% discounts over our On-Demand prices for committed
usage" and offer a separate "Inference Single GPU Price" (H100 at $6.16/hr)
**[verified]**. Nebius state "up to 35% less than on-demand rates by reserving
large-scale clusters for multiple months" **[verified]**.

### Power

I could not source an authoritative per-GPU B200 SXM TDP figure from NVIDIA
**[unverified]** — the datasheet redirects and the line-card PDF did not parse.
What I did verify: the Supermicro SYS-A21GE-NBRT HGX B200 chassis ships **6×
5250 W Titanium PSUs in 3+3 redundancy**, i.e. ~15.75 kW deliverable, with 8×
B200 SXM and 1.4 TB HBM3e **[verified]**. That is an upper bound on node draw,
not a typical figure.

EIA Electric Power Monthly, May 2026 **[verified]**: US average industrial
retail electricity **8.71 ¢/kWh**, commercial 13.54 ¢/kWh; Texas industrial
6.33 ¢, California industrial 20.20 ¢.

Parametric result **[inferred]**, at 8.71 ¢/kWh:

| Node draw | PUE | Facility kW | $/node-hr | $/GPU-hr | $/1M out tok @ 4,533 tok/s |
|---|---|---|---|---|---|
| 10 kW | 1.15 | 11.5 | $1.00 | $0.125 | $0.0068 |
| 12 kW | 1.25 | 15.0 | $1.31 | $0.163 | $0.0089 |
| 14 kW | 1.40 | 19.6 | $1.71 | $0.213 | $0.0116 |

**Conclusion: power is 1.5–3% of the rental rate and about 0.3% of the market
price of the tokens.** Electricity is not a lever in this business at our scale.
Silicon amortisation and utilisation are.

### Amortisation and the $/GPU-hr conversion

I do not have a verified 2026 HGX B200 node capex figure, so I will not invent
one. The honest framing:

```
$/GPU-hr(owned) = (node_capex/(8 × life_hours × utilisation))
                + $/GPU-hr(power)
                + $/GPU-hr(datacenter, networking, staff)
```

What we *can* say from verified data **[inferred]**: the **committed/reserved
cloud rate is a hard upper bound on owner economics**, because a rational
neocloud prices reserved capacity above its own fully-loaded cost. Together's
181-day reserved rate of **$6.79/GPU-hr** and GCP's 3-year CUD of **$7.09** are
therefore ceilings on what a well-run owner pays. The preemptible/spot floor of
**$3.95–4.26** is a rough lower bound on marginal cost plus thin margin, because
spot is priced to clear otherwise-idle inventory.

**Use $4.00/GPU-hr as the optimistic owner case and $6.79 as the realistic
one.** Everything below uses $6.69 (Lambda on-demand) as the central reference
because it is a real, buyable, non-preemptible price for exactly our
configuration.

---

## The arithmetic

For a node of `G` GPUs at rate `R` $/GPU-hr producing `T` output tokens/second:

```
$ per 1M output tokens = (R × G) / (T × 3600) × 1e6
```

At G=8 and R=$6.69, the node costs **$53.52/hour**, and one million output
tokens costs **$14,866.67 / T**.

A second form, useful for capacity planning:

```
$ per 1M tokens (all tokens, in+out) = (R × G) / (T_total × 3600) × 1e6
```

**The single most important thing about this formula is that it is inverse in
throughput.** Halving latency at fixed concurrency doubles throughput and halves
cost; but *reducing concurrency to halve latency* leaves throughput unchanged or
worse and leaves cost unchanged or worse. Latency and cost are only aligned when
the latency win comes from making the machine faster, not from giving each
request more of the machine.

### A units warning about our own numbers

Our stated operating points are "C1: 365 tok/s total" and "C64: 40.8k tok/s
aggregate". Taken literally, 40,800/64 = **637.5 tok/s per stream at C64**, which
would be *faster* than our 365 tok/s at C1. That is impossible, and it also
contradicts our own observation that per-stream speed falls 4.7x from C1 to C16.

The consistent reading **[inferred]** is that 40.8k is **total tokens including
prefill**, matching SGLang's `tokens_per_sec_per_gpu` convention (verified above
as `(isl+osl)/osl × output`). At ISL≈8192 / OSL≈1024 the factor is 9, giving
**~4,533 output tok/s at C64 → 70.8 tok/s per stream**, which sits neatly between
our C16 figure of 77.7 and SGLang's published C64 balanced figure. I use that
reading throughout and show both where it matters.

**Action item independent of everything else in this document: pin down whether
our internal dashboards report output-only or total tokens. A 9x unit error in a
cost model is not survivable.**

Under that reading, and this is genuinely good news: our C64 of 40,800 total
tok/s beats SGLang's published B200 NVFP4 balanced C64 cell of 30,160 total
tok/s by **1.35x** **[inferred, from verified inputs]**. **We are ahead on
throughput and behind on single-stream latency.**

---

## Worked examples at our measured points

At $6.69/GPU-hr, 8 GPUs, $53.52/node-hr:

| Operating point | Output tok/s (node) | Per-stream tok/s | Output tok/s/GPU | **$/1M output tokens** |
|---|---|---|---|---|
| C1 | 365 | 365.0 | 45.6 | **$40.73** |
| C16 | 1,243 | 77.7 | 155.4 | **$11.96** |
| C64 (4,533 output) | 4,533 | 70.8 | 566.7 | **$3.28** |
| *C64 under the literal 40.8k reading* | *40,800* | *637.5* | *5,100* | *$0.36* |

Across the whole price spectrum, at our three points:

| $/GPU-hr | Source | C1 ($/1M out) | C16 ($/1M out) | C64 ($/1M out) |
|---|---|---|---|---|
| $3.95 | Nebius preemptible | $24.05 | $7.06 | $1.94 |
| $4.26 | CoreWeave spot | $25.96 | $7.62 | $2.09 |
| $5.28 | AWS spot | $32.15 | $9.44 | $2.59 |
| $5.98 | RunPod community | $36.41 | $10.69 | $2.93 |
| **$6.69** | **Lambda on-demand** | **$40.73** | **$11.96** | **$3.28** |
| $6.79 | Together reserved | $41.34 | $12.14 | $3.33 |
| $7.09 | GCP 3-yr CUD | $43.17 | $12.68 | $3.48 |
| $8.19 | Together on-demand | $49.86 | $14.64 | $4.02 |
| $8.60 | CoreWeave on-demand | $52.36 | $15.38 | $4.22 |
| $14.24 | AWS on-demand | $86.71 | $25.47 | $6.98 |
| $16.11 | GCP on-demand | $98.08 | $28.81 | $7.90 |

And the same arithmetic applied to SGLang's *published* B200 NVFP4 cells at
$6.69/GPU-hr **[inferred, from verified inputs]**:

| Cell | Conc | Per-stream tok/s | Node output tok/s | $/1M output | $/1M all tokens |
|---|---|---|---|---|---|
| low-latency | 1 | 540 | 468 | $31.74 | $3.53 |
| low-latency | 16 | 184 | 2,035 | $7.31 | $0.81 |
| balanced | 64 | 79 | 3,351 | $4.44 | $0.49 |
| balanced | 256 | 33 | 4,749 | $3.13 | $0.35 |
| high-throughput | 1024 | 15 | 4,716 | $3.15 | $0.35 |

**The throughput curve saturates around C256.** Going from C256 to C1024
*reduces* node output throughput slightly (4,749 → 4,716) while cutting
per-stream speed by 2.2x. On this model, on this hardware, **there is no cost
reason to run above ~C256**, and every latency reason not to. That is a
free finding: our C64 point may already be near the efficient frontier, and
pushing concurrency further buys nothing.

---

## Comparison to published API prices, and the implied margin

Market prices for GLM-5.2, from the 33-endpoint OpenRouter table **[verified]**:

| Tier | Input $/1M | Output $/1M | Cache read $/1M |
|---|---|---|---|
| Z.ai list (first-party) | 1.40 | 4.40 | 0.26 |
| Modal median of fast providers | 0.75–1.40 | 2.40–4.40 | 0.14–0.26 |
| Aggressive discounters (Novita) | 0.339 | 1.065 | 0.063 |
| AA blended floor (7:2:1) | — | — | **$0.49/1M blended** |

### Implied margin at each operating point

Take **$2.42/1M output** (CoreWeave's price, mid-market, FP4) as revenue and
$6.69/GPU-hr as cost, assuming for the moment that all cost is attributed to
output tokens and utilisation is 100% **[inferred]**:

| Operating point | Cost $/1M out | Revenue $/1M out | Gross margin |
|---|---|---|---|
| C1 (365) | $40.73 | $2.42 | **−1,583%** |
| C16 (1,243) | $11.96 | $2.42 | **−394%** |
| C64 (4,533) | $3.28 | $2.42 | **−36%** |
| C256 (SGLang published, 4,749) | $3.13 | $2.42 | **−29%** |

**At mid-market output prices, output tokens alone do not cover the GPU cost at
any concurrency.** This is the correct and important conclusion, and it explains
the entire structure of the market.

The business only works because **input tokens carry most of the revenue and
almost none of the cost.** At AA's stated agentic blend of 7 cache-hit : 2 input
: 1 output, revenue per 10 tokens at CoreWeave pricing is
7×$0.14 + 2×$0.76 + 1×$2.42 = $0.98 + $1.52 + $2.42 = **$4.92 per 10M tokens =
$0.492/1M blended** — matching AA's published $0.49 exactly **[verified by
reconstruction]**.

Now the cost side under the same blend. Prefill of a cached token is nearly free
(a radix-tree pointer). Prefill of an uncached input token costs roughly one
forward pass over one token, which on an 8-active-of-256-expert MoE is far
cheaper per token than decode because prefill is compute-dense and batched, while
decode is memory-bound and reads the whole active weight set per step. Using
SGLang's published totals: at C256 the node moves 42,744 total tok/s of which
4,749 are output — so **8.0 input tokens are processed for every output token, at
the same $53.52/hr** **[inferred]**. That gives $0.35/1M for all tokens
indiscriminately.

So: revenue $0.492/1M blended vs cost ~$0.35/1M all-tokens at C256 → **~29% gross
margin at 100% utilisation** **[inferred]**. At the realistic 40% utilisation a
serverless endpoint sees, that inverts to a loss.

**Implied utilisation required to break even** at CoreWeave-like pricing and
$6.69/GPU-hr **[inferred]**: 0.35/0.492 = 71%. At Nebius preemptible
($3.95/GPU-hr, cost $0.207/1M) the break-even utilisation is 42%. **This is why
the cheap tier of the market runs on spot/preemptible capacity and serves at 64
tok/s, and why the fast tier prices at or above the $1.40/$4.40 list.** There is
no configuration that is simultaneously fastest and cheapest, and the market
table proves nobody has found one.

---

## DeepSeek's disclosure: the reference point

DeepSeek published a full 24-hour cost and revenue accounting for V3/R1 serving
**[verified]** —
<https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md>

**Architecture.** Prefill: Routed Expert EP32, MLA/Shared Expert DP32 across 4
nodes, 9 routed + 1 shared expert per GPU. Decode: Routed Expert EP144, MLA/
Shared Expert DP144 across 18 nodes, 2 routed + 1 shared expert per GPU. Full
prefill/decode disaggregation.

**The 24-hour accounting (Feb 27–28, 2025):**

| Quantity | Value |
|---|---|
| Peak node occupancy | 278 H800 nodes |
| Average node occupancy | 226.75 nodes × 8 H800 |
| Assumed cost | **$2 per H800 GPU-hour** |
| Total daily cost | **$87,072** |
| Input tokens | 608 B |
| — of which cache hits | 342 B (**56.3%**) |
| Output tokens | 168 B |
| Average output speed | 20–22 tok/s |
| Per-node prefill throughput | ~73.7k tokens/s input |
| Per-node decode throughput | ~14.8k tokens/s output |
| Theoretical daily revenue at R1 pricing | **$562,027** |
| Stated cost profit margin | **545%** |

**Derived** **[inferred]**:

| Derived quantity | Value |
|---|---|
| Average GPUs | 1,814 |
| Cost per 1M output tokens (all cost to output) | **$0.518** |
| Cost per 1M total tokens (in+out) | **$0.112** |
| Total tok/s/GPU (in+out) | **4,951** |
| Output tok/s/GPU | **1,072** |
| Revenue / cost ratio | 6.45x (= 84.5% margin on revenue) |

**How to read this honestly:**

1. **The $2/GPU-hour is a modelling assumption, not a market price.** H800 market
   rates in the same period were multiples of that. Their margin figure scales
   inversely: at a realistic $2.69–4.00/GPU-hr the 545% cost margin falls to
   roughly 300–400% **[inferred]**.
2. **They serve at 20–22 tok/s per user.** That is 17x slower than our C1 and 3.5x
   slower than our C64 per-stream. They bought cost with latency, deliberately and
   completely. Their 1,072 output tok/s/GPU on H800 is *twice* our 566 output
   tok/s/GPU on B200 at C64 **[inferred]** — because they run at vastly higher
   concurrency across a 144-way expert-parallel decode cluster.
3. **The 56.3% cache hit rate is the load-bearing number.** More than half of all
   input tokens cost essentially nothing to serve but are billed (at DeepSeek's
   own current V4 pricing, cache hits bill at $0.044/M vs $1.32/M for misses —
   a 30x discount that still leaves the cached token nearly pure margin
   **[inferred]**).
4. **They disclose the revenue as "theoretical"** — actual revenue is lower
   because of free web/app usage, off-peak discounts, and V3's lower price. They
   say so themselves.

**The transferable lesson**: DeepSeek's economics come from (a) very high
concurrency, (b) prefill/decode disaggregation letting each phase run its
optimal parallelism, (c) a 56% cache hit rate, and (d) accepting 20 tok/s. We
are optimising for the exact opposite corner. **We should expect our unit
economics to be roughly an order of magnitude worse per token, and we should
price accordingly rather than pretending otherwise.**

---

## The four levers, quantified

### 1. Batching — the dominant lever, with sharply diminishing returns

From SGLang's published B200 NVFP4 curve **[inferred from verified data]**:

| Concurrency | Node output tok/s | $/1M out | Marginal gain over previous |
|---|---|---|---|
| 1 | 468 | $31.74 | — |
| 16 | 2,035 | $7.31 | **4.3x** |
| 64 | 3,351 | $4.44 | 1.65x |
| 256 | 4,749 | $3.13 | 1.42x |
| 1024 | 4,716 | $3.15 | **0.99x — negative** |

**C1 → C16 is worth 4.3x on cost. C256 → C1024 is worth nothing.** The entire
cost benefit of batching on this model/hardware is captured by C256. Anything
beyond that is pure latency destruction. This is the single most useful curve in
this document for capacity planning.

### 2. Prefix cache hit rate — the biggest lever nobody puts on a leaderboard

Z.ai report that coding-agent workloads average **over 70K input tokens** with
high prefix reuse **[reported]** (<https://z.ai/blog/scaling-pain>). SemiAnalysis
measured, on their agentic corpus, a **~0.97 theoretical hit-rate ceiling** and a
collapse to **<0.1** once the working set outgrows HBM **[verified]**.

The economics **[inferred]**: at an 8:1 input:output ratio, moving the cache hit
rate from 0.1 to 0.9 removes ~80% of prefill work. Using SGLang's C256 cell,
prefill is 38,000 of 42,744 total tok/s of the machine's work; eliminating 80% of
it frees ~71% of the node's token-processing capacity for decode. That is
**worth more than every kernel optimisation in this document combined.**

The mechanisms, all verified:
- **SGLang HiCache** with the exact settings in the Bottom Line section — host
  DRAM tier at 0.75× the device pool.
- **Session-affinity routing.** Under DP-attention you must route a session back
  to the rank holding its prefix or the cache is useless:
  `sglang_router --policy consistent_hashing --request-id-headers x-correlation-id
  --dp-aware`.
- **Z.ai's LayerSplit**: shard the prefill KV cache by layer across context-
  parallel ranks instead of replicating it, and broadcast the owning rank's layer
  before attention, overlapping the broadcast with indexer computation. They
  report **10–132% throughput improvement at 90% cache hit rate for 40K–120K
  contexts, with gains increasing with context length** **[reported]**, and note
  the only residual communication is the indexer cache, "approximately one-eighth
  of the KV Cache". Exposed in SGLang as
  `--enable-dsa-cache-layer-split --attn-cp-size 8 --cp-strategy interleave`
  **[verified]**, and they claim it "can reduce kv cache memory by up to 75%".
- **Audit for prompt-prefix poisoning.** The Claude Code attribution-header bug
  documented in the SGLang cookbook is the archetype: one varying token early in
  the prompt destroys the whole cache. Instrument the actual hit rate, per
  client.

### 3. Speculative decoding — burns FLOPs to buy latency, and the exchange rate is knowable

For draft length k with acceptance length AL(k), each verification step processes
k+1 tokens and yields AL(k) accepted tokens. FLOPs per accepted token scale as
(k+1)/AL(k) **[inferred]**. Using the verified GLM-5.2 golden curve:

| k | AL | Verify tokens | FLOPs per accepted token (rel.) | Decode speedup vs no spec | vs our current k=3 |
|---|---|---|---|---|---|
| 1 | 1.84 | 2 | 1.087 | 1.84x | 0.62x |
| 2 | 2.50 | 3 | 1.200 | 2.50x | 0.84x |
| **3 (ours)** | **2.99** | **4** | **1.338** | **2.99x** | **1.00x** |
| 4 | 3.33 | 5 | 1.502 | 3.33x | 1.11x |
| **5 (SGLang LL)** | **3.61** | **6** | **1.662** | **3.61x** | **1.21x** |
| 6 | 3.78 | 7 | 1.852 | 3.78x | 1.26x |
| 7 | 3.91 | 8 | 2.046 | 3.91x | 1.31x |
| 8 | 4.06 | 9 | 2.217 | 4.06x | 1.36x |

**The exchange rate at the margin.** Going k=3 → k=5 buys **+20.7% decode speed**
for **+24.2% verification FLOPs per accepted token**. At C1 the machine is
memory-bound and those FLOPs are free — take the deal. At C256 the machine is
compute-saturated and those FLOPs come straight out of throughput — refuse it.
This is precisely why SGLang's verified recipes use 5-1-6 at low latency, 2-1-3
at balanced NVFP4, 1-1-2 at balanced FP8, and **no speculation at all** at high
throughput **[verified]**.

**Cost impact, stated plainly** **[inferred]**: speculative decoding at C1
improves both latency *and* cost, because throughput at C1 is latency. At high
concurrency it is a pure cost regression — SGLang turns it off entirely at C1024,
which is the honest admission that it never pays there.

**A free bonus from Z.ai**: speculative-decoding metrics double as an online
correctness monitor. They report that `spec_accept_length` below 1.4 after 128
generated tokens indicates KV-cache corruption (draft/target state mismatch),
and `spec_accept_rate` above 0.96 indicates a degenerate repetition loop; they
proactively terminate and re-dispatch such requests **[reported]**. Cheap,
high-value telemetry we get for free from a feature we already run.

### 4. PD disaggregation — a throughput and TTFT lever, and an availability risk

The verified topology evidence from SemiAnalysis's GB300 GLM-5.2 configs
**[verified]**: low-latency configurations use **1 prefill worker feeding 2–6
small TP4 decode workers**; high-throughput uses **2 wide prefill workers (TP8
EP8 dp-attn) feeding 1 very wide decode worker (TP16 EP16 dp-attn)** at
concurrency 128–192.

DeepSeek's production numbers give the throughput asymmetry that motivates it:
**73.7k input tok/s per node in prefill vs 14.8k output tok/s per node in
decode** — a 5x difference in token rate, and completely different optimal
parallelism (EP32/DP32 vs EP144/DP144) **[verified]**.

**Cost effect** **[inferred]**: disaggregation lets each phase run at its own
optimal batch size and parallelism instead of a compromise. On a single 8-GPU
node it is mostly a loss — you split 8 GPUs into two smaller pools and pay a KV
transfer over NVLink for the privilege. **Its value begins at 2+ nodes.** For our
single-node 8×B200 target it is not the right lever; for a multi-node production
fleet it is.

**The risk it introduces is real and documented.** Z.ai's post-mortem
**[reported]** describes a KV-cache reuse race under PD disaggregation: decode
times out a request and reclaims its KV slots without notifying prefill; in-flight
RDMA writes from the aborted request then land on memory already reassigned to a
new request, corrupting it. They observed **3–5 abnormal outputs per 10,000
requests** offline, and the fix — decode notifies prefill on abort, prefill
returns a safe-to-reclaim signal only when no RDMA writes were started or all have
completed — dropped the production anomaly rate from **~0.1% to below 0.03%**.
They also found a missing load-use ordering bug in HiCache where the DSA indexer
kernel launched before the indexer cache finished loading from host (fix
submitted as SGLang PR #22811). **If we adopt PD disaggregation or HiCache, these
two bugs are the first things to check for in our fork.**

---

## Where latency optimisation and cost optimisation actively conflict

Five places, in decreasing order of how much they matter to us.

**1. Concurrency itself.** This is the whole conflict in one line. Cost is
inverse in throughput; per-stream latency is inverse in concurrency. From the
verified SGLang curve: C1 → C256 improves cost **10.1x** and degrades per-stream
speed **16.4x**. There is no configuration that is good at both, and every
"balanced" recipe in every source is an explicit choice of a point on this curve.
Our two objectives are two different products and should be two different
deployments.

**2. Speculative decoding.** Positive-sum at C1, negative-sum at C256, as
quantified above. The crossover for GLM-5.2 on B200 is somewhere between C16 and
C64 (SGLang keeps 5-1-6 through C16, drops to 2-1-3 at C64, and to nothing at
C1024) **[verified]**.

**3. Parallelism strategy.** At C≤16 on 8 GPUs, plain TP8 **strictly dominates**
attention-DP because DP leaves only ~2 sessions per rank — SemiAnalysis state
this as their reason for not running a DEP arm at all below C16 **[verified]**.
At C64+, DP-attention + DeepEP wins decisively. Same hardware, opposite answer.

**4. Chunked prefill size.** 32768 gives +34–78% throughput and −39–59% TTFT at
the balanced point but is "neutral for high-throughput" **[verified]**, and at
low latency SGLang uses 8192. Under DP8 the budget is per-engine and divides by
rank count, so a value that starves prefill under DP is fine under TP. There is
no single right value.

**5. Quantization.** NVFP4 buys 1.74x on TPOT at C1 (FP8 3.22 ms → NVFP4 1.85 ms
**[verified]**) and, at C256, 4,749 vs 4,464 output tok/s — only **6%** on
throughput **[inferred]**. **NVFP4 is overwhelmingly a latency optimisation, not
a cost optimisation**, and it carries a real (if small) accuracy risk that shows
up on the AA Intelligence Index and Arena. If we were purely cost-optimising,
FP8 would be the better default.

**The synthesis.** Run two fleets. A low-latency fleet — NVFP4, TP8, no
attention-DP, EAGLE 5-1-6, chunked prefill 8192, `mem-fraction-static` 0.83,
small `max-running-requests` — that exists to win the Artificial Analysis board
and to serve latency-sensitive customers at a price that reflects its $31–41/1M
output token cost. And a cost fleet — FP8 or NVFP4, DP8 + DeepEP, EAGLE 2-1-3 or
none, chunked prefill 32768, HiCache on, capped at ~C256 — that serves everyone
else at $3.13/1M. Do not try to make one machine do both; every source in this
document independently reached that conclusion and encoded it as three named
strategies.

---

## Techniques ranked by transferability to our stack

| # | Technique | Source | Mechanism | Expected effect | Difficulty | Risk |
|---|---|---|---|---|---|---|
| 1 | EAGLE 5-1-6 instead of 3-1-4 at C1 | SGLang cookbook + SemiAnalysis golden AL | AL 2.99 → 3.61 at +24% verify FLOPs, free at C1 | **+15–21% single-stream** | Trivial | Needs `mem-fraction-static` headroom |
| 2 | Reproduce SGLang's verified NVFP4 B200 low-latency cell | `glm-5.2-benchmarks.jsx` | TPOT 1.85 ms published vs our 2.74 ms | **up to +48%** | Medium | May not reproduce; the gap may be in our fork |
| 3 | Confirm TP8 (not attention-DP) at C≤16 | SemiAnalysis config comment | DP leaves 2 sessions/rank, strictly dominated | Large if we have this wrong | Trivial | None |
| 4 | `SGLANG_SIMULATE_ACC_*` as an internal A/B control | SemiAnalysis golden AL README | Pins acceptance so kernel deltas are measurable | Makes all future work measurable | Trivial | Must not publish pinned numbers |
| 5 | HiCache host tier, ratio 0.75, `write_back`/`direct`/`page_first_direct` | SemiAnalysis B200 script | Radix hit rate 0.1 → ~0.9 on agentic traffic | **Largest $/token lever** | Medium | Two documented OOM modes; PR #22811 ordering bug |
| 6 | `--mem-fraction-static 0.83` for B200+EAGLE+DSA | SemiAnalysis B200 script | 180 GB card, EAGLE verify + graph capture need ~31 GB | Stability, enables #1 | Trivial | None |
| 7 | Adopt E2E normalized interactivity (P90 of s/tok, inverted) as internal north star | InferenceX `MODELS.md` | Single metric penalising TTFT and decode together | Prevents local optima | Low | None |
| 8 | Chunked prefill 32768 on the throughput fleet | SGLang cookbook | Prefill-bound at C64+ with 2048 default | **+34–78% throughput, −39–59% TTFT** | Trivial | Neutral-to-harmful at C1 and at C1024 |
| 9 | Session-affinity routing under DP (`consistent_hashing`, `--dp-aware`) | SemiAnalysis B200 script | Keeps multi-turn sessions on the rank owning their prefix | Makes #5 actually work under DP | Low | None |
| 10 | Spec-decode metrics as an online corruption monitor | Z.ai scaling-pain | AL<1.4 after 128 tok ⇒ KV corruption; rate>0.96 ⇒ repetition loop | Catches silent quality failures | Low | Needs a retry path |
| 11 | LayerSplit for prefill KV under context parallelism | Z.ai / SGLang `--enable-dsa-cache-layer-split` | Shard KV by layer across CP ranks, broadcast overlapped with indexer | **+10–132% at 90% hit rate** (reported) | High | Multi-node only; unreproduced by us |
| 12 | Cap concurrency at ~C256 | SGLang published curve | C256→C1024 gains 0% throughput, loses 2.2x latency | Free latency | Trivial | None |
| 13 | AA prompt set as an acceptance-length calibration corpus | AA methodology downloads | AL is workload-dependent; tune on the scoring distribution | Better P50 stability | Low | Not the real scoring set |
| 14 | Scrape OpenRouter `/endpoints` on a cron | OpenRouter public API | 33 competitor prices + quant tags + uptime, free | Pricing intelligence | Trivial | None |
| 15 | PD disaggregation | SemiAnalysis GB300 configs, DeepSeek | Independent prefill/decode parallelism | Large at 2+ nodes | High | Two documented race conditions |

---

## Final table: configuration → tok/s/stream → tok/s/GPU → $/1M tokens

All at 8×B200, ISL≈8192, OSL≈1024, `$6.69/GPU-hr` (Lambda 8×B200 on-demand,
**[verified]**) = $53.52/node-hour. Rows marked **(ours)** are our measured
points under the total-token reading; rows marked *(SGLang)* are published,
verified, reproducible cells; the DeepSeek row is a different model and
generation, included as the cost reference point.

| Configuration | Conc | tok/s per stream | Output tok/s per GPU | Total tok/s per GPU (in+out) | **$/1M output tok** | **$/1M all tok** |
|---|---|---|---|---|---|---|
| **(ours) C1, NVFP4, EAGLE 3-1-4, TP8** | 1 | **365.0** | 45.6 | 411 | **$40.73** | $4.53 |
| *(SGLang) NVFP4 low-latency, EAGLE 5-1-6, TP8* | 1 | **540.5** | 58.5 | 527 | $31.74 | $3.53 |
| *(SGLang) FP8 low-latency, EAGLE 5-1-6, TP8* | 1 | 310.6 | 32.0 | 288 | $58.03 | $6.45 |
| *(SGLang) GB300 FP8 low-latency (4 GPU, TP4)* | 1 | 219.8 | — | 459 | — | $4.05 |
| **(ours) C16** | 16 | **77.7** | 155.4 | 1,399 | **$11.96** | $1.33 |
| *(SGLang) NVFP4 low-latency* | 16 | 184.2 | 254.4 | 2,289 | $7.31 | $0.81 |
| *(SGLang) FP8 low-latency* | 16 | 109.6 | 164.0 | 1,476 | $11.33 | $1.26 |
| **(ours) C64, 40.8k total tok/s** | 64 | **70.8** | 566.7 | 5,100 | **$3.28** | $0.36 |
| *(SGLang) NVFP4 balanced, DP8, EAGLE 2-1-3* | 64 | 78.7 | 418.9 | 3,770 | $4.44 | $0.49 |
| *(SGLang) FP8 balanced, DP8, EAGLE 1-1-2* | 64 | 56.7 | 342.0 | 3,078 | $5.43 | $0.60 |
| *(SGLang) NVFP4 balanced* | 256 | 33.3 | 593.6 | 5,343 | **$3.13** | **$0.35** |
| *(SGLang) FP8 balanced* | 256 | 30.7 | 558.0 | 5,022 | $3.33 | $0.37 |
| *(SGLang) NVFP4 high-throughput, DP8, no spec* | 1024 | 14.9 | 589.5 | 5,305 | $3.15 | $0.35 |
| *(SGLang) FP8 high-throughput, DP8, no spec* | 1024 | 20.8 | 451.0 | 4,059 | $4.12 | $0.46 |
| *(SGLang) MI355X FP8, no spec, TP8* | 1 | 73.7 | 9.0 | 81 | — | — |
| DeepSeek V3/R1 on H800, EP144 decode, $2/GPU-hr assumed | very high | 20–22 | 1,072 | 4,951 | **$0.518** | **$0.112** |

**Market reference:** GLM-5.2 output tokens sell for **$1.065–$7.26 per 1M**
across 33 endpoints; the first-party Z.ai list price is **$4.40**; AA's 7:2:1
blended floor is **$0.49/1M** **[verified]**.

Three things this table says at once:

- **Our C1 is 1.48x off a published, reproducible SGLang recipe.** That is the
  gap to close, and item 1 in the Bottom Line is probably a third of it.
- **Our C64 is 1.35x better than SGLang's published C64.** We are genuinely
  strong on throughput. Do not regress it while chasing C1.
- **No row in this table earns money on output tokens alone at mid-market
  prices.** The business is input tokens and cache hits, which is why every
  serious operator's published work — DeepSeek's 56% hit rate, Z.ai's LayerSplit,
  SemiAnalysis's HiCache policy — is about prefill and caching rather than decode
  kernels.

---

## Sources

Every URL below was fetched and read during this work unless explicitly marked
otherwise.

**Artificial Analysis**
- <https://artificialanalysis.ai/methodology/performance-benchmarking> — full performance methodology, metric formulas, integrity terms **[verified]**
- <https://artificialanalysis.ai/methodology> — overview **[verified]**
- <https://artificialanalysis.ai/methodology/intelligence-benchmarking> — Intelligence Index v4.1.1 composition and parameters **[verified]**
- <https://artificialanalysis.ai/models/glm-5-2/providers> — 17-provider GLM-5.2 board with speed/TTFT/blended price **[verified]**
- <https://artificialanalysis.ai/providers/makora> — provider page **[verified]**
- `https://artificialanalysis.ai/downloads/methodology/performance-prompts.xlsx` — prompt set, link verified on the methodology page, **not downloaded**

**SemiAnalysis InferenceX**
- <https://github.com/SemiAnalysisAI/InferenceX> — repo README, hardware and model list **[verified]**
- <https://github.com/InferenceMAX/InferenceMAX> — archived, redirects to the above **[verified]**
- <https://inferencex.semianalysis.com/> — live dashboard **[verified]**
- `raw.githubusercontent.com/SemiAnalysisAI/InferenceX/main/MODELS.md` — scenarios, AgentX guidelines, E2E normalized interactivity, engine submission policy, KV-offload policy **[verified]**
- `.../configs/nvidia-master.yaml` — all NVIDIA configs incl. `glm5.2-fp4-b200-sglang-agentic-mtp` and `glm5.1-fp8-b200-tilert` **[verified]**
- `.../configs/CONFIGS.md` — config schema **[verified]**
- `.../configs/runners.yaml` — fleet hardware facts **[verified]**
- `.../benchmarks/single_node/agentic/glm5.2_fp4_b200_sglang_mtp.sh` — the B200 launch script **[verified]**
- `.../benchmarks/single_node/agentic/README.md` — DRAM offload policy, budget formula **[verified]**
- `.../benchmarks/benchmark_lib.sh` — bench_serving invocation, AIPerf replay construction **[verified]**
- `.../golden_al_distribution/README.md` — synthetic acceptance policy, collection procedure **[verified]**
- `.../golden_al_distribution/glm5.2_mtp.yaml` — GLM-5.2 golden acceptance curve **[verified]**
- `.../docs/index.md`, `.../docs/architecture.md` — orchestration overview **[verified]**
- SPEED-Bench paper, arXiv 2604.09557 — referenced by the golden-AL README, **not fetched**
- A SemiAnalysis blog post announcing InferenceMAX — **[unverified]**, could not locate a working URL

**SGLang**
- `raw.githubusercontent.com/sgl-project/sglang/main/docs/cookbook/autoregressive/GLM/GLM-5.2.mdx` — deployment guide, config tips, chunked-prefill and LayerSplit findings **[verified]**
- `.../docs/src/snippets/configs/zai-org/glm-5.2.jsx` — verified serve flags per hardware × quant × strategy **[verified]**
- `.../docs/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx` — measured TTFT/TPOT/throughput cells **[verified]**
- <https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2> — rendered version **[verified]**

**Z.ai**
- <https://z.ai/blog/scaling-pain> — PD KV-cache race, HiCache load-use ordering bug, LayerSplit, spec-decode anomaly detection **[verified via text proxy]**

**TileRT**
- <https://github.com/tile-ai/tilert> — README, 500 tok/s GLM-5.1 claim, 1K/1K synthetic conditions, PD-disagg architecture **[verified]**

**OpenRouter**
- <https://openrouter.ai/docs/features/provider-routing> — routing, sorting, 5-minute percentile window **[verified]**
- <https://openrouter.ai/api/v1/models> — 414 models with pricing **[verified]**
- <https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints> — 33 GLM-5.2 endpoints **[verified]**
- <https://openrouter.ai/rankings> — usage rankings **[verified, low information]**

**MLPerf**
- `raw.githubusercontent.com/mlcommons/inference_policies/master/inference_rules.adoc` — scenarios, latency constraints, speculative-decoding appendix, statistical requirements **[verified]**
- `raw.githubusercontent.com/mlcommons/inference/master/README.md` — v6.1 benchmark list **[verified]**
- <https://mlcommons.org/benchmarks/inference-datacenter/> — overview **[verified, low information]**

**LMArena**
- <https://lmarena.ai/how-it-works> — human-preference methodology; **no serving-performance measurement** **[verified]**

**DeepSeek**
- <https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md> — the cost/margin disclosure **[verified]**
- <https://api-docs.deepseek.com/quick_start/pricing> — V4 Flash/Pro pricing, off-peak schedule **[verified]**

**Pricing**
- <https://lambda.ai/pricing> — B200 on-demand and 1-Click Cluster **[verified]**
- <https://www.coreweave.com/pricing> — HGX B200/B300, GB200 NVL72, spot **[verified]**
- <https://www.runpod.io/pricing> — B200/B300 community and secure **[verified]**
- <https://nebius.com/prices> — B200 on-demand and preemptible **[verified]**
- <https://www.together.ai/pricing> — GPU rental and serverless token prices **[verified]**
- <https://aws.amazon.com/ec2/instance-types/p6/> — P6-B200 specs **[verified]**
- <https://instances.vantage.sh/aws/ec2/p6-b200.48xlarge> — AWS on-demand and spot **[verified]**
- <https://www.getdeploying.com/reference/cloud-gpu/nvidia-b200> — cross-provider B200 comparison incl. GCP **[verified]**
- <https://cloud.google.com/compute/gpus-pricing>, <https://vast.ai/pricing>, <https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/> — attempted, **no usable figures extracted**

**Hardware and power**
- <https://www.nvidia.com/en-us/data-center/hgx/> — HGX B200 8-GPU, 1.4 TB HBM3e, 14.4 TB/s NVLink, 72 PFLOPS FP4 dense **[verified]**
- <https://www.supermicro.com/en/products/system/gpu/8u/sys-a21ge-nbrt> — 6× 5250 W Titanium PSUs, 3+3 redundant **[verified]**
- <https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a> — May 2026 US industrial 8.71 ¢/kWh **[verified]**
- NVIDIA Blackwell datasheet / data-center line card — **[unverified]**, redirected or unparseable; **per-GPU B200 TDP is not sourced in this document**
