# Baseten, Modal, Fal and the infrastructure-first inference companies

## What this is

A technique-mining pass over the published engineering of the "infrastructure-first"
inference companies — the ones that sell GPU-hours or per-token APIs rather than
models, and that therefore have to be publicly good at inference. Everything here
was read at a URL listed in **Sources**. Nothing is recalled from memory.

Scope actually covered, in descending order of yield:

| Company | Verdict | Why |
|---|---|---|
| **Baseten** | Highest-yield source in this file | Publishes per-model Blackwell inference engineering with named techniques, config, and AA-verified numbers. Ships open-source artifacts. Runs GLM-5.2 on 8xB200 — the exact model/hardware we run. |
| **Cursor (Anysphere)** | Highest-yield *kernel* source, full stop | Not an inference vendor, but publishes the most reimplementable Blackwell MoE kernel work in the industry, including a low-batch MoE decode kernel and an open-source MoE megakernel. Found by following a citation from fal. |
| **fal** | High yield, narrow | Diffusion-first, but their comm/compute-overlap and NVFP4/MXFP8 kernel posts are 8xB200-native and directly portable. Their DSpark post is the best public account of training a speculator for a fine-tuned MoE. |
| **Doubleword** | High yield, Hopper-centric | Publishes a per-kernel roofline decomposition of a DSA/MLA MoE decode step that is structurally identical to our own hotspot table, plus the best public spec-dec-vs-batching analysis. |
| **Modal** | Medium yield, excellent reference material | Their FA4 reverse-engineering post and GPU Glossary are the best free Blackwell attention reference. Their platform work (GPU memory snapshots) is cold-start, not latency. |
| **BentoML / Modular** | Medium, mostly a handbook | The LLM Inference Handbook is a solid reference with a few honest negative results. BentoML was acquired by Modular in Feb 2026. |
| **Replicate** | Low — say so plainly | Diffusion and platform posts only. One useful cold-start data point (`torch.compile` artifact caching). No LLM inference engineering. |
| **RunPod** | Nothing transferable | Blog is buyer-education and fine-tuning VRAM math. Nothing about decode latency. |

Two honesty flags that apply throughout:

- **Every "fastest on Artificial Analysis" claim below is a vendor's reading of a
  third-party leaderboard at a point in time.** AA's methodology (~10k in / ≥1500 out,
  P50 over trailing 72h, measured over the public internet from GCP) means these numbers
  bundle network, TTFT, and a specific traffic shape. They are not comparable to our
  internal single-stream 365 tok/s unless we replicate the harness.
- **Baseten does not publish its parallelism configs for its flagship models.** They name
  the *technique* (TP+EP mixture, ADP, NVFP4, MTP) but publish TP/EP degrees for exactly
  one model (GPT-OSS 120B, TP8 EP1). Treat the rest as directionally useful, not as a config.

---

## Bottom line for our system

Ranked by (expected effect on our two objectives) × (probability it works) ÷ (difficulty).
Our measured C1 hotspots for reference: dense GEMM 37.1%, collectives 19.6% (47% of that
is rank arrival skew), MoE expert GEMMs 19.4%, attention 10.9%, DSA indexer 5.8%.

### Tier 1 — do these

1. **Warp decode for the MoE expert GEMMs at C1–C4.** Cursor's kernel assigns one warp per
   *output neuron* instead of one warp-group per *expert*, folding the top-k routing weight
   into a register accumulator and eliminating padding, scatter, combine, the activation
   gather buffer, and the per-expert output buffer. Measured **1.84x MoE decode throughput on
   B200**, 3.95 TB/s at B=32 (58% of a measured 6.8 TB/s copy-kernel peak), *and* 1.4x closer
   to an FP32 reference because activations stay BF16 and accumulators FP32. This attacks our
   19.4% MoE bucket at exactly the batch sizes we care about, and Cursor is explicit that it
   only wins at low batch — which is our C1 objective precisely.
   **Effect: plausibly 8–9% of C1 wall clock. Difficulty: high (new CUDA kernel), but the
   design is fully described and the shape (256 experts / 8 active) is in their tested set.**

2. **Pull-based dispatch + push-based combine for MoE all-to-all, and stop using NCCL for it.**
   Cursor measured push-based dispatch *signalling* at **103 µs vs 18 µs for pull-based (5.8x)**
   on NVL72, because push requires each rank to wait on up to 71 peer signals plus a rack-wide
   memory fence, while pull lets a rank issue a load and consume the moment data lands. Pull
   also delivered **up to 29% higher NVLink bandwidth utilisation under expert imbalance**,
   because NVLink has separate lanes per direction and pull splits protocol metadata across
   both. Our collectives are 19.6% of C1 and **47% of that is rank arrival skew** — which is
   exactly the failure mode cross-rank signalling causes.
   **Effect: potentially most of the 9.2% of wall clock we currently burn on arrival skew.
   Difficulty: high. Mitigant: `github.com/cursor/mixture-of-kittens` is open source (it is a
   *training* megakernel, but the dispatch/combine/signalling design transfers).**

3. **Suffix-automaton speculation layered on top of our EAGLE 3-1-4.** Baseten open-sourced
   `basetenlabs/sa_spec` and it is **merged into TensorRT-LLM (PR #11434)**. Mechanism: build a
   suffix automaton over the prompt on the host while KV prefill runs on device; update it
   *on device* every decode step from a CUDA-graph-compatible `extend()` kernel launched as
   `<<<batch_size, 1>>>`; if the suffix match length exceeds `SA_SPEC_THRESH` (default 4) use the
   n-gram draft, else fall back to the model draft. Measured **30–33% higher accept length and
   throughput vs MTP alone** on DeepSeek-V3.1-NVFP4 / `glaiveai/code_edits_sample`, up to 40% on
   production agentic coding. Crucially they proved zero overhead by setting the threshold to
   infinity and confirming latency matched vanilla MTP. This stacks multiplicatively with EAGLE
   and needs no change to draft length.
   **Effect: large on agentic/code traffic, ~0 on pure reasoning (they say accept rate is near 0
   there). Difficulty: low-medium — the code is header-only POD C++/CUDA that compiles for both
   host and device.**

4. **Skip the DSA indexer when sequence length < K.** Baseten's GLM-5 post states it directly:
   *"If the sequence length is less than K, we can skip the indexer and run standard full
   attention."* They also note the indexer's high-precision projections must be fused and
   overlapped because their numerical tolerance is far tighter than the FP8 MHA path. Our
   indexer is 5.8% of C1 with `index_topk_freq=4`. At AA's ~10k input this won't fire on prefill,
   but it is free money on short-context requests and on the early decode steps.
   **Effect: small but nearly free. Difficulty: low.**

5. **Fuse RMSNorm and SwiGLU into the GEMM epilogue.** fal published the full Blackwell mechanics:
   their NVFP4 GEMM uses a **256×256 CTA tile**, the epilogue walks it as **4 subtiles of 128×128**,
   each partitioned across 128 threads so each thread holds a **FragmentSize=128 fragment**. At
   `head_dim=128` a whole head lives in one thread's registers, so RMSNorm fuses trivially. At
   `head_dim=256` they wrote a **three-touch "revisit" epilogue** (Tile 0 pass 0: sum of squares;
   Tile 1 pass 1: finish `rstd`, normalise, store; Tile 0 pass 2: revisit and scale) that keeps
   everything on-chip. For SwiGLU they **permute the columns of B at weight-packing time** so
   gate/up pairs land adjacent in the same fragment, avoiding cross-tile synchronisation entirely.
   Dense GEMM is our single largest bucket at 37.1%.
   **Effect: removes an HBM round-trip per GEMM. Difficulty: medium-high (CUTLASS EVT), but
   the weight-permutation trick alone is cheap.**

### Tier 2 — high value, more speculative

6. **Data-parallel attention instead of TP for the MLA path.** Doubleword got a **2.2x
   throughput jump (5,856 → 12,802 tok/s)** on DeepSeek-V4-Flash purely by moving from TP4 to
   TP1×DP4. Reason: MLA compresses KV into a *single shared latent*, so TP has no head dimension
   to shard and must replicate the KV cache N times. A side effect they call out is that every
   per-token kernel (elementwise, FP8 quant, sampling) then runs on 1/N of the batch per rank
   instead of the full global batch on every rank. Baseten independently confirms the shape:
   their *standard* GLM-5.2 API uses **Attention Data Parallelism (ADP)**, and they drop to
   TP+EP only on the latency-tuned "Fast" tier. This is a throughput lever, not a C1 latency
   lever — but it is likely the single biggest thing available for our C64 cost objective.
   **Effect: large on aggregate tok/s. Difficulty: medium — SGLang supports DP attention.**

7. **Run the draft and target as one fused model, not two.** Baseten's stated GLM-5 mechanism:
   *"we run both the draft model and the target model as one big model. By doing so, the overhead
   is essentially zero, other than when you predict wrong draft tokens."* Independently, fal hit
   the mirror-image failure: their DSpark Markovian heads were **sharded across TP ranks, which
   introduced collectives inside the drafter and made TP slower**; replicating the head weights
   instead pushed them past 1000 tok/s. Rule: draft-head weights should be replicated, never
   sharded, and the draft forward should live inside the same CUDA graph as the target.
   **Effect: removes per-step host overhead from our 3-1-4 EAGLE. Difficulty: medium.**

8. **Confidence-gated ragged draft depth instead of a fixed 3-1-4 tree.** Doubleword's model:
   expected committed tokens at depth γ is `m(γ) = 1 + Σ_d Π_k a_k`, where `a_k` is the drafter's
   own per-depth confidence. They argue for **per-sequence** depth gating rather than a
   batch-wide constant, and published the calibration data (`Doubleword/specdec-calibration`,
   half a million draft rounds) plus the simulator (`doublewordai/inference-lab`). Baseten lists
   "dynamic-length speculation, where the draft length is adjusted based on speculation confidence
   on a per-request, per-micro-batch basis" as their explicit next step. The handbook notes SGLang
   already switches γ between pre-captured tiers `[1, 3, 7]` using pre-captured CUDA graphs so the
   switch is free.
   **Effect: medium on both objectives. Difficulty: medium — the CUDA-graph tiering trick makes
   it cheap.**

9. **Async QKV / all-to-all overlap using functional collectives, with a scale-dependent policy.**
   fal's numbers are on **8xB200, exactly our topology**: async Ulysses (overlap each projection's
   compute with the previous branch's all-to-all) cut the pre-attention chunk **23–25% at 2/4/8
   GPUs** for ~3% end-to-end. Fused all-gather+matmul via `torch.ops.symm_mem.fused_all_gather_matmul`
   was far better at 2–4 GPUs (**−37.3% / −33.4% chunk**) but **collapsed to −4.6% at 8 GPUs**
   because messages get too small to amortise fixed overhead. Their sharpest observation is one we
   should internalise: *NCCL collectives are GPU kernels that compete with GEMMs for SM cycles, so
   timeline overlap does not imply throughput overlap.* Copy-engine transport via PyTorch Symmetric
   Memory fixes that at 2–4 GPUs but regresses at 8.
   **Effect: relevant to the 10.4% of C1 that is collectives-minus-skew. Difficulty: medium.
   Caveat: fal measured this on dense attention, not MoE dispatch; they explicitly say MoE
   routing "should benefit even more" but did not measure it.**

### Tier 3 — worth knowing, lower priority for us

10. **A Rust tokenizer.** Baseten built `Basetenkenizer` because at 1M-token agentic inputs with
    high prefix-cache hit rates, **tokenisation happens whether or not the prefill is a cache hit**
    and becomes material to TTFT. Measured **up to 18x faster than tiktoken at 1M tokens, ~6x at
    short lengths, with exact token-ID parity**, on 52 pinned vCPUs. At our 189 ms TTFT and 10k
    input this is probably single-digit milliseconds — but it is a pure TTFT win with no quality risk.
11. **Prefill-priority scheduling.** Baseten's runtime "prioritizes prefill steps over decode",
    accepting a small ITL hit to protect TTFT. Cheap scheduler change; relevant if we chase TTFT.
12. **Turn speculation off above a batch-size threshold.** Baseten states plainly that spec dec
    "is dynamically turned off" at high batch sizes because verification becomes costly under
    compute saturation. Given our per-stream speed falls 4.7x from C1 to C16, there is a crossover
    we should find rather than assume.

### Explicitly *not* recommended

- **Don't expect PD disaggregation to help single-stream latency.** Baseten reports 2x tok/s from
  it on GLM-5.2, but the BentoML/Modular handbook reports "performance can drop by 20–30% in our
  tests" for small workloads or untuned setups, and notes high prefix-cache hit rates favour local
  prefill. Doubleword's analysis is that it is "strictly better once the conditions are met" but
  the conditions are *thousands* of GPUs — DeepSeek V3 used EP32 prefill and EP320 decode units.
  On a single 8xB200 node with a C1 objective, disaggregation is a cost lever, not a latency lever.
- **Don't chase a fixed global-max quantization scale.** fal found that on a diffusion transformer,
  naive FP4 error *compounds* across steps and cannot be repaired post-hoc; the fix required
  gradients flowing through the rounding. For an LLM the error doesn't compound the same way, but
  their negative result on "quantize then distill without gradients through the quantizer" is worth
  remembering before we trust an NVFP4 conversion on eval loss alone.

---

## Baseten

**What they run.** A proprietary runtime ("the Baseten Inference Stack" / "Baseten Inference
Runtime", internally versioned as `briton` and `v2_llm` in their engine-builder config) that sits
*alongside* rather than replaces the open-source engines. Their consistent public line: they
benchmark TensorRT-LLM, SGLang and vLLM per model and pick the winner, then layer proprietary
kernels, a "Baseten Speculation Engine", and NVIDIA Dynamo for orchestration on top. [verified]

> "We like Dynamo because it's framework-agnostic, so it doesn't stop us from choosing the best
> inference engine (TensorRT-LLM/SGLang/vLLM) for a specific model and use case."
> — *The Baseten Inference Stack at NVIDIA Dynamo Day*

For Kimi K3 they were explicit that engineers configured vLLM, SGLang *and* their in-house engine
in parallel, sharing findings across all three. Tri Dao is a listed author on their GLM-5,
Kimi K2.5, Kimi K2 Thinking and GPT-OSS model-performance posts. [verified]

### Techniques with mechanism + evidence

**1. DSA (DeepSeek Sparse Attention) indexer kernels — directly applicable to us**
Mechanism, quoted: the lightning indexer must scan every token to pick top-K, so *"at shorter
context lengths, this scanning overhead dominates, and the sparse attention has a longer prefill
time than full attention."* Their three fixes: (a) if `seq_len < K`, skip the indexer entirely and
run full attention; (b) optimise the indexer code path itself for longer sequences; (c) keep the
multi-head attention in FP8 but run the indexer's projections in high precision — *"the numerical
tolerance is much lower compared to other parts of the model"* — and fuse/overlap those projections
to hide their cost.
Evidence: <https://www.baseten.co/blog/how-we-built-the-fastest-glm-5-api/> [verified]

**2. Fused draft+target speculation ("one big model")**
Mechanism: rather than orchestrating a separate draft process, the MTP draft is executed as part of
the target model's graph, so *"the overhead is essentially zero, other than when you predict wrong
draft tokens."* They note all other optimisations then stack multiplicatively.
Evidence: same GLM-5 post. [verified]

**3. Suffix Automaton MTP accelerator — open source, merged upstream**
Mechanism: hybrid n-gram + model speculation. A suffix automaton (amortised O(1) per update, finds
*arbitrarily long* longest matches unlike fixed-width n-gram lookup) is built on the host during
prefill, transferred to device before the first decode step, and updated on device thereafter. Three
Python entry points: `add_request(request_id, prompt)`, `prepare(request_ids)`,
`extend(draft_tokens_out, accepted_tokens_in)`. `extend` is a CUDA-graph-capturable kernel launched
`<<<batch_size, 1>>>` (one block per batch slot, one thread each) that returns both draft tokens and
match lengths; match length is compared against `SA_SPEC_THRESH` (default 4) to decide SA vs MTP.
Implementation detail worth stealing: the automaton is a **plain-old-data struct with a header-only
core algorithm that compiles for both C++ and CUDA**, so host↔device transfer is zero-conversion;
CUDA-graph compatibility comes from specialising a smart pointer's `memcpy` and the `extend()`
invocation onto the active torch stream.
Numbers: 30–33% higher accept length and throughput vs MTP alone (DeepSeek-V3.1-NVFP4,
`glaiveai/code_edits_sample`); up to 40% higher throughput at equal latency on production agentic
coding. SA alone reaches accept length 10+ on code but ~0 on reasoning; MTP alone gives 2–4.
Zero-overhead proof: threshold set to infinity, latency matched baseline MTP exactly.
Artifacts: <https://github.com/basetenlabs/sa_spec>, merged as NVIDIA/TensorRT-LLM PR #11434
(reference integration PR #10951). [verified]

**4. KV-cache-aware routing (built on NVIDIA Dynamo)**
Mechanism: requests are hashed and organised in a radix tree; the Dynamo LLM-Aware Router computes
an overlap score between an incoming request and the KV blocks live across all GPUs in the cluster,
then routes on a blend of overlap and current load. Baseten adds custom logic to mix KV routing with
round-robin per model.
Numbers (Qwen3 Coder 480B A35B, ~50k input / ~1k output, high-load stress test, 4 replicas,
**89% hit rate**): **50% lower TTFT, 34% lower TPOT, 61% more RPS, 62% higher output TPS** vs random
routing. Shadowing real OpenRouter production traffic: **48% lower P95, 49% lower P99**.
Evidence: <https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/> [verified]

**5. Live draft-model training from the serving path**
Mechanism: hidden states are extracted from live inference and streamed straight to training nodes —
no offline storage at all. They cite the reason: *"A single sample on Kimi K2 can exceed 2GB, and
full draft training requires millions of them."* Implementation constraints they solved and that we
would hit too: all network I/O and buffering offloaded to a dedicated background process with careful
CUDA-event synchronisation on the overlap-scheduler loop, so the main execution thread never stalls;
the extra memory is proportional to `max_num_tokens_per_iter`, **not** `max_sequence_length`, which
preserves long-context headroom; CUDA-IPC double buffering into pageable memory via `mmap`; UCXX for
async RDMA; Trio's guest-loop mode integrated with `torch.cuda.synchronize()` to run an async loop
without new threads, minimising GIL contention. Stated compatible with single-CUDA-graph execution
and the overlap scheduler.
Numbers: **median accept-rate increase of 20%; 100%+ on constrained traffic patterns.**
Evidence: <https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/> [verified]

**6. EAGLE-3 training recipe (the most concrete public one)**

| Parameter | Baseten's recommendation |
|---|---|
| TTT-length (test-time-training length) | **7–9**; too low makes the head brittle when it sees its own drafts |
| `num_draft_tokens` at inference | **3–4**; "going higher (e.g. 8) rarely helps" |
| Optimizer | AdamW at all scales |
| LR, ~3–7B target | 1e-4 |
| LR, ~7–20B target | 5e-5 |
| LR, 20B+ target | 2e-5 |
| Sampling | temperature=0 best; EAGLE paper reports **15–25% speedup reduction at temp=1** (quality stays lossless, only speedup degrades) |
| Dataset size, generic task | 200–300k samples (≤20B target), ~500k (large) |
| Dataset size, specialised task | ~100k samples |
| Tokens per sample | 1k–2k (prompt + completion) |
| Training accuracy target | plateau at **70–80%** |
| Observed production speedup | **1.5–2.5x** — vs 4–6x claimed in papers, which they attribute to serving-framework differences |

The "golden rule": **regenerate the training outputs with the target model.** Training on another
model's or a human's outputs misaligns the draft distribution and tanks acceptance. Debug ladder for
non-convergence: raise TTT-length to 7–9, then fix LR; for low acceptance: drop `num_draft_tokens` to
3–4 and verify the chat template at inference matches training byte-for-byte.
Frameworks they say all work: NVIDIA/Model-Optimizer, sgl-project/SpecForge, torchspec-project/TorchSpec.
Evidence: <https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/> [verified]

**7. DFlash — block-diffusion speculation (ICML 2026, code released)**
Mechanism: instead of drafting autoregressively, a lightweight block-diffusion model with
**bidirectional attention** predicts 8–16 tokens *in a single forward pass*. Inputs: hidden states
from **5–6 evenly-spaced layers of the target model**, the last valid decode token, and a block of
mask tokens (`block_size`). Trained with cross-entropy on the masked tokens with exponential
weighting `exp(−(i−1)/γ)` to prioritise earlier positions; embedding and LM head frozen; random
anchor positions define the context/generation boundary. A single DFlash forward is 2–4x slower than
a single EAGLE forward but faster than EAGLE's *entire* draft phase, which is where the win comes
from — and it permits much deeper drafters.
Numbers (Qwen3-8B, single B200, concurrency 16): GSM8k **654 tok/s, ~3x baseline and 10% faster than
vLLM's implementation**; 1.2 s mean latency, 2.9x improvement, 25% faster than vLLM. Paper claims
>6x lossless acceleration and up to 2.5x over EAGLE-3.
Artifacts: arXiv 2602.06036, <https://github.com/z-lab/dflash>. Baseten runs a custom backend they
claim beats both vLLM's and SGLang's DFlash implementations. [verified: blog + arXiv abstract]

**8. Basetenkenizer**
Mechanism, in order of contribution: specialised hand-written pre-tokenisation scanners that
recognise the model's split regex ahead of time instead of running a general regex engine (they note
the Kimi regex can alternatively be reformulated for the PCRE2 JIT); a **stack-resident BPE merge
tier** for pre-tokens up to 32 bytes using a stack-allocated linked list with a linear minimum-rank
scan and a dense first-round byte-pair table (no heap, no priority queue); routing identical
pre-tokens to the same CPU core so repeated words merge once; scheduling each 400,000-character chunk
(a legacy tiktoken limit) on a separate core; native typed segments so per-span `allow_special`
policy runs in Rust rather than a Python work-list; zero-copy NumPy ownership transfer; `abi3-py310`
PyO3 bindings with strings borrowed as `Cow`. One fused `encode_segments([(text, allow_special)...])`
call replaces a Python loop across the Python/native boundary.
Numbers: **up to 18x faster than tiktoken end-to-end at 1M tokens, >6x at short sequences, exact
token-ID parity**, median of end-to-end encodes on 52 pinned vCPUs (Xeon Platinum 8480+).
Honest comparison they published against themselves: gigatoken is fastest at 10k tokens, crossover
comes before 200k, Baseten leads by 1.41x at 1M; and on gigatoken's own target (offline files read
in Rust) gigatoken does 2.68 GiB/s vs their 118 MiB/s Python-string path.
Built on Crusoe `fastokens`, `gigatoken`, and MiniJinja; upstream PRs listed.
Evidence: <https://www.baseten.co/blog/making-kimi-k3-tokenization-18x-faster-for-million-token-agentic-workloads/> [verified]

**9. Baseten Delivery Network (cold starts)**
Mechanism: (a) *own the source* — weights are mirrored into Baseten-managed blob storage at **push
time**, keyed by content hash or etag, so identical files across fine-tunes are stored once and a
re-push resolves as a metadata check; the mirroring pipeline runs 1–5 GB/s by fanning file chunks
across a worker pool. (b) *three cache tiers* — node-local NVMe, an in-cluster peer cache backed by a
**consistent hash ring spanning essentially all nodes**, and mirrored origin via parallel byte-range
fetches. The peer tier is active, not passive: on an origin fetch, the file is split into fixed-size
chunks, each chunk is assigned to a ring node, nearly all nodes pull their chunks from origin
concurrently, and the node that needs the file fans the rest in from peers over the in-cluster
fabric. (c) *single-flight* at two levels — the CSI driver coalesces same-file requests per
`(node, content-hash)`, and the hash ring makes chunk ownership deterministic so no coordination
protocol is needed. Delivery happens **before the user container starts**, so weight transfer sits
outside the billable GPU window. LRU eviction at tier 1, rate-limited background fills so cold starts
don't starve inference I/O.
Numbers: **2–3x faster cold starts**; **>2 GB/s** onto H100 nodes. Worked example: 50 replicas of a
140 GB model across 20 cold nodes → each node pulls ~7 GB and origin bandwidth is 1x model size,
not 50x.
Evidence: <https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/> [verified]

**10. Negative results and things that broke (May 2025, TensorRT-LLM spec dec)**
Worth reading because these are the failure modes of any two-model speculation setup:
- Draft and target on the same GPU **fought for resources and ran at half speed**. Their fix was not
  separate GPUs (wasteful — a draft model barely uses one) but a **synchronised async loop where
  draft and target workers read from dedicated queues and only one runs at a time**, which also let
  them schedule target inference ahead of draft for the first token to protect TTFT.
- TensorRT-LLM **wasn't batching target-model requests at all** — the target executes one inference
  step, so it fires on the first arrival. Fixed in their custom C++ gRPC server (which replaces
  Triton) by marking requests as batch members and waiting for the batch.
- Batches were 2–3 where 10 was expected. Root cause: **TRT-LLM's request scheduler did not account
  for KV-cache reuse when scheduling against `max_num_tokens` under chunked prefill**, so it thought
  each request needed a full prefill. Patched and upstreamed.
- **One benchmark got *worse* p50 latency with spec dec on.** They published it rather than hiding it.
- TTFT is *slightly worse* with spec dec "despite our optimization efforts."
Evidence: <https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm/> [verified]

### Published numbers

All AA figures are Baseten's reading of Artificial Analysis on the stated date.
AA's config is ~10k input / ≥1500 output, P50 over a trailing window, over the public internet.

| Model | Hardware | Quant | Speculation | Parallelism (published) | Reported | Date |
|---|---|---|---|---|---|---|
| GLM-5.2 (launch) | NVIDIA Blackwell | NVFP4 (from FP8 via ModelOpt) | MTP | not published | **280+ tok/s**, TTFT 0.8 s, **TTFAT 7.9 s** (7.1 s reasoning + 0.8 s input) | Jun 22 2026 |
| GLM-5.2 (standard, retuned) | B200 | "improved NVFP4 weights" | updated spec-dec profile | **Attention Data Parallelism (ADP)** | **601 tok/s** (prior peak 280, prior avg ~100); AA workload ~10k in / ~1k out | Jul 25 2026 |
| GLM-5.2 Fast | B200 | same weights as standard | — | **TP + EP, configs selected for latency; max batch size substantially reduced** to cut resource contention | tier launch, no separate figure | Jul 24 2026 |
| GLM-5 | Blackwell | NVFP4 | MTP, fused draft+target | not published | **186+ tok/s**, best-in-class TTFT | Mar 9 2026 |
| Kimi K2.5 (1T) | Blackwell | **INT4 → BF16 → NVFP4** (ModelOpt, custom calibration set) | custom **EAGLE-3, ~1B params**, trained on hidden states from synthetic code/scientific/multilingual queries via DeepSpeed | not published | **340+ tok/s** | Feb 10 2026 |
| Kimi K2 Thinking (1T) | **single 8xB200 node** | INT4 → BF16 (`compressed-tensors`, "a few hours") → NVFP4 (ModelOpt) | none at launch | **mixture of TP and EP** | **140+ tok/s, 300 ms TTFT** | Dec 1 2025 |
| GPT-OSS 120B | **8xB200** | none — native MXFP4 kept | **EAGLE-3** | **TP8 EP1** ("this configuration yields the lowest latency") | ~400 → **650+ tok/s (+60%)** | Oct 24 2025 |
| Qwen3 Coder 480B A35B | multi-replica | — | — | 4 replicas, 89% KV hit rate | 50% ↓TTFT, 34% ↓TPOT, 61% ↑RPS, 62% ↑TPS from KV routing | Mar 16 2026 |
| Qwen3-8B (DFlash) | 1x B200, concurrency 16 | — | DFlash | — | 654 tok/s GSM8k (3x), 1.2 s mean latency (2.9x) | 2026 |
| Kimi K3 (2.8T) | **8x GB300** (2 NVL72 nodes; 9 replicas/rack) | native **MXFP4 weights, MXFP8 activations** — no NVFP4 port needed | DSpark / DFlash / EAGLE-3 candidates | TP and EP **across nodes**, enabled by NVL72 fabric | no tok/s published | Jul 27 2026 |
| DeepSeek V4 Pro 0813 (1.7T) | not published | native MXFP4 | DSpark speculator | "parallelism, KV cache allocation, prefill-decode worker ratio" — values not published | no tok/s published | Aug 2026 |

**Config asymmetries to keep in mind:** the GLM-5.2 601 tok/s figure is at ~1k output, while AA's
headline methodology uses ≥1500; the GPT-OSS number is on a 120B model with no quantisation step;
and the Kimi K2.5 340 tok/s is on a model whose weights were already 4-bit from QAT. None of these
is directly comparable to our 365 tok/s on GLM-5.2.

### Open-source artifacts and what is actually usable

| Artifact | Usable? |
|---|---|
| `github.com/basetenlabs/sa_spec` + NVIDIA/TensorRT-LLM PR #11434 | **Yes, directly.** Header-only POD C++/CUDA core, CUDA-graph safe, `SA_SPEC_THRESH` env-tunable. The single most portable thing in this file. |
| `github.com/z-lab/dflash` (Baseten co-authored; ICML 2026) | Yes — reference implementation. Baseten's own backend is closed. |
| Their upstream PRs to TensorRT-LLM, Dynamo, `crusoecloud/fastokens` (#8, #9), `mitsuhiko/minijinja` (#912, #913) | Yes, already merged. |
| `Basetenkenizer` | **No.** Described in detail but not released. They did release a Kimi K3 `tokenizer.json` on HuggingFace. |
| The Baseten Inference Runtime / Speculation Engine / `briton` | **No.** Closed. |
| Engine-builder config surface (docs) | Useful as a **checklist of knobs**: `quantization_type` ∈ {`no_quant`, `fp8`, `fp8_kv`, `fp4`, `fp4_kv`, `fp4_mlp_only`}; `speculative_decoding_mode` ∈ {`LOOKAHEAD_DECODING`, `DRAFT_TOKENS_EXTERNAL`}; `enable_b10_lookahead` (their own algorithm); `lookahead_windows_size` / `lookahead_ngram_size` / `lookahead_verification_set_size`; `kv_cache_free_gpu_mem_fraction` (default 0.9); `kv_cache_host_memory_bytes`; `batch_scheduler_policy` ∈ {`max_utilization`, `guaranteed_no_evict`}; `use_fp8_context_fmha`; `use_paged_context_fmha`. Documented caveat: **"Lookahead works best with batch sizes under 32."** |

### What they say about Blackwell/B200 specifically

- **NVFP4 over everything else.** Stated reasons: dual scale factor gives higher precision than
  alternatives at 4 bits, and it has "deep support in Blackwell Tensor Cores."
- **INT4 → NVFP4 has no direct path.** Chinese labs target Hopper (no Blackwell in China), so Kimi
  ships INT4 from QAT. Baseten's route is INT4 → BF16 (`compressed-tensors`, hours of compute; can
  be skipped using third-party BF16 weights) → NVFP4 (ModelOpt). Crucially honest caveat:
  *"While NVFP4 is overall a more precise data format than INT4, switching does not enhance quality
  as the INT4 quantization was performed during training. There is no way to recover the information
  that was lost to compression during training."*
- **Native MXFP4 is left alone.** For GPT-OSS 120B and DeepSeek V4 Pro they explicitly did *not*
  requantise.
- **TP8 EP1 for lowest latency on 8xB200** for a 120B MoE — the only parallelism config they publish
  for a Blackwell deployment. They chose it over EP knowing EP gives better system throughput.
- **B200 vs H200 platform claims** (customer-observed, vendor-reported): 5x higher throughput on
  high-traffic endpoints, >50% lower cost per token on throughput-optimised deployments, up to 38%
  lower latency on DeepSeek-R1-class models.
- **On GB300 NVL72:** nodes are 4 GPUs, not 8, but "the GB300 NVL72 system has a sufficiently fast
  interconnect between nodes that we can run inference with Tensor Parallelism and Expert
  Parallelism across nodes."
- Their configuration sweep list for a new model is a useful checklist: TP/EP settings, **ADP
  toggling**, batch sizing, speculative decoder draft lengths, **linear layer caching intervals**,
  routing parameters, and engine settings.
- Their benchmarking practice: they reproduced SemiAnalysis's InferenceMAX on 4xB200 using the exact
  image from InferenceMAX's GitHub Action (`nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc2`) at
  **TP4 EP1** for GPT-OSS-120B 1k/1k, running the benchmark **server-side against `localhost:8000`
  to eliminate network variance**, and had to use `/completions` rather than `/chat/completions`
  plus `ignore_EOS` to reach the target token counts.

---

## Cursor (Anysphere)

Not an inference vendor. Included because they publish the most reimplementable Blackwell MoE
kernel work anywhere, and because their MoE shapes list literally includes **GLM-5.2 (E=256,
H=6144, I=2048, top-k 8)**. Found by following fal's acknowledgement of "Cursor's kernel
engineering blog."

### Warp decode — MoE inference at low batch on B200

Mechanism: flip the parallelism axis. Instead of grouping tokens per expert and running grouped
GEMMs, **assign each warp exactly one output value (neuron)** for its entire lifetime. Two kernels,
`moe_gate_up_3d_batched` and `moe_down_3d_batched`. In gate/up, a CTA is 8 warps; each warp owns one
intermediate neuron for a (token, routed-expert) pair, loads the expert ID, reads the gate and up
weight rows, streams the activation vector, converts MXFP8 weights to FP32 on the fly, and
accumulates both dot products in private registers — the activation vector is read once and reused
for both projections with **no shared-memory staging**. In down, each warp owns one output dimension
for one token, loops over all top-k experts, and folds each expert's routing weight into a single
running FP32 accumulator. Final reduction across the 32 lanes is a butterfly via `__shfl_xor_sync`,
compiling to `shfl.sync.bfly` — one hardware primitive, no L1 round-trip, no bank conflicts, no
barriers.

What that eliminates, in their words: padding each expert's token list to power-of-2 or 128-byte
boundaries (non-amortisable at batch 1), the scatter of 8 intermediate results to memory, the
separate combine pass, the activation gather buffer (a full copy of data that already exists at
batch 1), and the per-expert output buffer (8 × 2048 × 2 B = **32 KB per token in BF16**, allocated,
written, read once, discarded). Removing that 32 KB frees L2 for the weight rows that actually
determine performance.

Numbers (B200, Qwen-3 style model, internal inference system):
**1.84x end-to-end decode throughput, flat across all context-length buckets** (confirming it's a
pure generation-time win). **3.95 TB/s sustained at B=32 = 58% of a measured 6.8 TB/s copy-kernel
peak**; the gap is attributed to the random-access pattern expert routing creates. Accuracy
*improves*: outputs **1.4x closer to a full FP32 reference** because activations never round-trip
through MXFP8; min cosine similarity > 0.999996, max absolute difference 0.001953.

Their own caveat, which we should honour: *"Warp decode is not a general replacement for
expert-centric execution. Higher-volume workloads like prefill and large-batch inference still
benefit from expert-centric packing."*
Evidence: <https://cursor.com/blog/warp-decode> [verified]

### MXFP8 MoE training kernels — Blackwell numerics and `tcgen05` mechanics

Context: profiling on B200 showed the MoE layer (MegaBlocks) at **53% of forward and 27% of backward**
time. They rewrote it in pure CUDA/PTX plus ThunderKittens with zero CUDA-library dependencies:
**3.5x MoE layer speedup on both passes, 1.5x end-to-end on Blackwell, 2x vs their Hopper setup.**

The recipe, stated precisely enough to reimplement:
```
S   = cast_to_fp8e8m0( absmax(V) / 448 )      # round UP to nearest power of 2, min-clamp 2^-127
Q_i = cast_to_fp8e4m3( V_i / S )              # saturate out-of-range, round-to-nearest-even
```
E4M3 elements, E8M0 scale, **block size 32**. Training loss matched BF16 over 10k steps.

`tcgen05` constraints they enumerate (all of which bite on inference too):
- `tcgen05.mma` needs **only a single thread** to launch, unlike Hopper's `wgmma` which needs a whole
  warpgroup — so the classic 256-thread producer/consumer pattern is wrong on Blackwell.
- **2-CTA mode gives 15–20% speedup** for MXFP8 matmuls (two SMs share the B matrix), "essential for
  peak performance."
- Accumulation happens in **TMEM, not registers**. Any custom arithmetic on accumulators requires
  TMEM → registers → CUDA cores → TMEM, and *"this still kills tensor core occupancy."* This is the
  central reason block-scaled hardware formats beat software dequantisation on Blackwell.
- Scale factors must be in TMEM but there's no HBM→TMEM path: the fastest route is
  HBM →(`cp.async.bulk.tensor`/TMA)→ SMEM →(`tcgen05.cp`)→ TMEM, and the scales must already be in
  the layout `tcgen05.mma` expects.
- Budget: 128×512 TMEM (32-bit cells) and 227 KB contiguous SMEM per threadblock; they used 5 slots
  each for A/B scales and input tiles.

**Expert-wise supergrouping for L2.** Standard supergrouping (a ThunderKittens heuristic) keeps the
output region computed by all 148 SMs as square as possible. Their enhancement: apply it **per
expert**, over the submatrix belonging to the current expert rather than the whole output matrix.
Especially effective for grouped Wgrad, where the reduction axis is narrow. Result: **~2,650 TFLOP/s
grouped MXFP8, only 4% below the non-grouped version**; they note bad HBM access patterns cost
"nearly 50%."

vs DeepGEMM (the only Blackwell alternative), average latency in their production training:
grouped Fprop/Dgrad **0.43 ms vs 0.67 ms**, grouped Wgrad **0.65 ms vs 0.71 ms** — and their
benchmark *excludes* quantisation time, which DeepGEMM ships no optimised kernel for; *"in the worst
case, it can even be slower than BF16 training."*

Their MXFP8 quantiser: **6,212 GB/s while emitting scales in `tcgen05` layout**, vs NVIDIA
TransformerEngine 4,430 GB/s and PyTorch TorchAO 4,524 GB/s *with reshape* (5,236 / 5,245 naive
without). Biggest wins: **removing TMA swizzling in favour of a manual swizzle pattern**, relying on
the warp scheduler and inter-threadblock asynchrony rather than manual intra-threadblock overlap, and
minimising SMEM/register usage for occupancy.

| Config | MoE fwd | MoE bwd | End-to-end |
|---|---|---|---|
| Hopper BF16 | 32.36 ms | 63.24 ms | 12k TPS/GPU |
| Blackwell BF16 (direct port) | 25.96 ms | 59.17 ms | 16k TPS/GPU |
| Blackwell MXFP8 | **9.45 ms** | **17.04 ms** | **24k TPS/GPU** |

Evidence: <https://cursor.com/blog/kernels> [verified]

### Mixture-of-Kittens — the MoE all-to-all findings

A training megakernel for GB300 NVL72, open-sourced at `github.com/cursor/mixture-of-kittens`. The
*communication* findings are the transferable part for us.

**Direction matters, and the reasons are measurable.** Microbenchmark sending one 256×256 BF16 tile
(131,072 B payload) over NVLink:

| | Total | RX | TX | Protocol RX | Protocol TX |
|---|---|---|---|---|---|
| Push | 159.6 KB | 2.9 KB (1.84%) | 155.6 KB (99.16%) | 2.7 KB | 24.6 KB |
| Pull | 172.0 KB | 147.5 KB (85.71%) | 24.6 KB (14.29%) | 16.4 KB | 24.6 KB |

Push moves ~12.4 KB fewer bytes, but NVLink has **separate lanes per direction**, so to reach the
1.8 TB/s spec you need traffic flowing both ways. Empirically **pull delivers up to 29% higher NVLink
bandwidth utilisation when dispatching tokens under expert imbalance**.

**Signalling is the bigger effect.** With push-dispatch, a rank must wait for signals from up to 71
peers and issue a rack-wide memory fence before its grouped GEMM can start. With pull-dispatch,
cross-GPU signalling disappears entirely — a rank issues a load, waits for data, and proceeds; only
one entity signals, so overhead doesn't grow with EP degree. Measured: **push-dispatch signalling
103 µs vs pull-dispatch 18 µs — 5.8x.** Their final scheme: pull forward-dispatch, push
forward-combine, pull backward reverse-combine, push backward reverse-dispatch — which also lets one
schedule table be built once and reused for all four operations.

**Granularity has a closed-form heuristic.** Neither extreme is right: too fine (Comet-style, 256
tokens per MMA) starves the tensor cores with barrier waits; too coarse (DeepEP-style) makes them
wait for the first and last rounds. Target at least two full waves per grouped GEMM:
```
T ≥ 2C · 128 · 256 / min(2I, H)
```
where T is minibatch size, C is SM count, H hidden dim, I expert intermediate dim. For Kimi 2.5
(H=7168, I=2048, C=148 on Blackwell) this gives **T ≥ 2368**; their measured optimum was T=2560 at
3.425 ms vs 5.981 ms at T=512.

**Inter-SM overlap, not streams.** They partition SMs in software into comms SMs and comp SMs
signalling through a local counter, because *"with TMA loads and stores, we can fully saturate NVLink
bandwidth with less than a third of the SMs,"* and because *"multiple streams with green contexts
[were] unreliable at partitioning SMs exactly as intended, while software partitioning gives exact
allocation guarantees."*

Other transferable details: a **ring token buffer ("macrobatch")** of a few hundred MB avoids both
token dropping and CPU-GPU sync for dynamic expert loads; **dispatch-combine interleaving** so a
buffer slot is refilled as soon as it's drained; **Cluster Launch Control (CLC)**, a Blackwell
hardware work-stealing feature, so the megakernel can yield to higher-priority streams (e.g. FSDP
all-gather) instead of serialising them behind it; MXFP8 with the **shared expert kept in BF16** for
stability; router-weight gradients computed SonicMoE-style from the inner product of SwiGLU activation
and down-projection dgrad, fused into the SwiGLU backward.

Numbers: EP degree 64, 2,048 tokens/GPU before routing, single NVL72 rack, vs the best of
NCCL+PyTorch / DeepEP+PyTorch / DeepEP+TransformerEngine / HybridEP+Megatron:
**2.37x MXFP8 forward, 1.78x MXFP8 backward, 1.92x BF16 forward, 1.58x BF16 backward.**
End-to-end on 512 GPUs across several racks: **760.9 → 1,070.2 TPS/GPU (1.41x)** vs their previous
DeepEP-based production stack.
Evidence: <https://cursor.com/blog/mixture-of-kittens> [verified]

---

## fal

**What they run.** SGLang for both diffusion and their LLM prompt-expander path, plus their own
CUTLASS/CuTeDSL/Triton kernels. They patch vLLM and SGLang when a technique they want isn't
supported yet. Everything below is measured on **B200 or 8xB200**.

### Communication–computation overlap on 8xB200 [verified]

Benchmark setup, published: 8 B200s, bf16, B=1, H=40, D=128, fixed GPU clocks, cuDNN attention
backend, 15 warmup + 40 timed iterations × 3 repeats, max-rank median. Both "pre-attention chunk
latency" (QKV projection + pre-SDPA comms) and end-to-end step latency reported.

| Variant | Mechanism | Chunk latency | End-to-end |
|---|---|---|---|
| Baseline Ulysses | 3 local GEMMs → 3 all-to-alls → SDPA, serialised | — | — |
| **Async Ulysses** (ByteDance VeOmni) | Compute Q → launch Q a2a → compute K while Q in flight → launch K a2a → compute V → launch V a2a → wait → SDPA | **−23% to −25% at 2/4/8 GPUs** | ~−3% |
| Async + Symmetric Memory | Same schedule, but route transfers through the **copy engine** to stop NCCL kernels stealing SM cycles from GEMMs | Better at 2 and 4 GPUs, **regresses at 8** (payloads too small to amortise buffering/signalling overhead) | — |
| **Fused QKV** | One packed local weight shard (Q\|K\|V rows for local heads) into `torch.ops.symm_mem.fused_all_gather_matmul`, returning packed `(B, S_global, 3·H_local·D)` | **−37.3% at 2 GPUs, −33.4% at 4, only −4.6% at 8** | −5.0% / −4.8% / −0.3% |

The single most important sentence in the post for us:

> "Even though we launch communication asynchronously on separate streams, they can still run on the
> same underlying SM resources. NCCL collectives are implemented as GPU kernels that issue
> loads/stores and reduction work, so they compete with GEMMs for SM cycles. In practice, this means
> timeline overlap does not always translate into throughput overlap."

Their conclusions: **overlap is the robust high-scale default, fusion wins at low/mid scale**, and a
lightweight runtime policy should pick between packing, overlap and fusion based on sequence length,
world size and interconnect. They also note this was measured on an 8xB200 node where all-to-all is
already very strong, so slower interconnects have more slack to hide — and that
"communication-heavy workloads (for example, MoE routing) should benefit even more than this dense-
attention setup," though they did not measure it.
Evidence: <https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/>

### MXFP8 quantiser at 6+ TB/s on B200 [verified]

Written in CuTeDSL; the point is it writes scale factors **directly into the packed `tcgen05` layout**
(`BlockScaledBasicChunk` / `tile_atom_to_shape_SF`) so a block-scaled GEMM consumes them with no
repack step. TransformerEngine returns the same logical bytes but in a dense `(M, K/32)` layout that
must then be reordered.

Effective bandwidth is defined as `(2·M·K + 1·M·K + 1·M·(K/32)) / t`.

Progression of wins, with numbers:
1. **Split the grid over K.** Their first version mapped a CTA to a row block and looped over all of
   K; NCU showed Stall Wait dominating. Launching a 2D grid `(ceil(M/rows_per_cta), ceil(K/k_tile))`
   with `rows_per_cta=8, k_tile=256` on a 16384×16384 input gives 131,072 CTAs instead of 2,048 —
   **64x more**. This alone took them from **~1.3 TB/s → ~3.3 TB/s**.
2. **SIMT `cp.async` plateaued at ~3.4–3.6 TB/s.** Beyond that the kernel is bound by instructions
   per byte and copy/sync bookkeeping, not DRAM.
3. **Single-bulk-load TMA per CTA tile.** One bulk transaction for the whole `(8, 256)` region, wait
   on the mbarrier, consume. The trap they name: "over-pipelining TMA with repeated per-subtile
   barriers... that repeated barrier cost dominates once you're close to saturation."
4. **The unexpected bottleneck: scale-factor stores.** NCU's SASS correlation showed `STG.E.64` for Q
   (good) and `STG.E.U8` for S (bad) — byte-at-a-time stores spray partially-used 32 B sectors. Fix:
   pack four scale bytes into one 32-bit store when 4-byte aligned, fall back otherwise.
5. **Instruction hygiene: 97.9M → 78.9M instructions.** Reciprocal-multiply instead of `fdiv`; fuse
   scale math into `fma.rn.f32(absmax, 1/448, eps)`; rely on the pack instruction's built-in
   saturation instead of explicit clamps; packed FP32x2 ops; **compute absmax in the integer domain**
   (clear the sign bit, integer max, convert once at the end).
6. **32-lane CTAs beat 64 and 128** — more CTAs in flight, better latency hiding.

**Negative result, published:** *"We tried aggressive shared-memory swizzling to reduce bank
conflicts. Conflicts went down, but the extra index math ate the gains. At some point you're just
moving the bottleneck around."*
Evidence: <https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/>

### Inline PTX in Triton, including an NVFP4 quantiser [verified]

`tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)` lets you inject
element-wise PTX without leaving Triton. Their three examples build to a real result: using
`cvt.rn.satfinite.e2m1x2.f32` (converts two FP32s to two packed e2m1 values in one byte) with
`pack=4` and a `mov.b32 $0, {tmp0,tmp1,tmp2,tmp3}` to assemble a 32-bit register, they wrote an
**NVFP4 quantiser in <100 lines of Triton that matches FlashInfer's and TensorRT-LLM's 2,000+ line
CUDA kernels on B200, and beats them at large shapes, approaching ~7 TB/s**. Smaller shapes still
favour the CUDA kernel.

Caveats they list honestly: you own correctness of register constraints, packing factors and dtypes;
mis-specified constraints silently produce wrong results; the kernel becomes architecture-aware; and
you get element-wise semantics only — no shared memory, no sync, no warp-level control.
Evidence: <https://blog.fal.ai/instruction-level-control-with-inline-elementwise-asm-in-triton/>

### Epilogue fusion on SM100 [verified]

Already summarised in the Bottom Line. The concrete layout facts: NVFP4 GEMM uses a **256×256 CTA
tile**; the collective epilogue walks it as **4 subtiles of 128×128** in fixed order 0→1→2→3 with no
going back; each subtile is partitioned across 128 threads giving **FragmentSize = 128**. So
`head_dim=128` RMSNorm fuses in a single `visit()` with no cross-thread shuffle. `head_dim=256`
required writing a custom epilogue with a **three-touch revisit schedule** and a visitor carrying
reduction state across touches. For gated-SiLU, the gate/up pair at columns `n` and `n + N/2` can land
in different fragments, subtiles or CTAs — so they **permute B's columns once during weight packing**
to produce interleaved `[gate0, up0, gate1, up1, ...]`, after which a custom EVT visitor operates on
adjacent elements in the same fragment.
Headline: Ideogram V4 at 1K resolution, **2.75 s → 0.44 s (6x)** with no visible quality loss.
Two negative results worth carrying: post-hoc colour/luma correction could not repair FP4 error
because it is "a latent trajectory problem, not a display problem"; and **naive distillation with a
frozen quantised student failed — "diffusion training loss does not predict image quality."** The fix
was quantisation-aware distillation with gradients flowing through FP4 rounding via a straight-through
estimator.
Evidence: <https://blog.fal.ai/serving-sub-second-ideogram-v4-without-quality-loss/>

### DSpark speculator on a fine-tuned MoE, on a single B200 [verified]

This is the best public end-to-end account of building a speculator for a fine-tuned MoE. Model:
Qwen3.6 35B-A3B MoE, FP8, one B200, optimising for **low concurrency** (their words: "we aim to serve
[at low concurrency], which is where we aim to give the most interactivity per user").

| Configuration | Throughput | Accept length |
|---|---|---|
| SGLang, FlashInfer backend + FlashInfer/TRTLLM MoE kernels, no spec | 328 tok/s @ C1; <200 @ C8; ~100 @ C16–32 | — |
| + native MTP head (degraded by PEFT, which doesn't fine-tune MTP heads) | 500 tok/s | ~2.5 |
| + DFlash, z-lab public checkpoints (trained for the *base* model) | 468 tok/s | ~2.4 |
| + DFlash retrained from scratch on 250K samples via TorchSpec | fell short — 250K samples not diverse enough, training slow | — |
| + **DFlash warm-started** from public weights, then trained on their 250K | **700 tok/s** | **~3.9 @ block size 8** |
| + **DSpark** (DeepSeek: DFlash block predictor + Markovian heads), reimplemented in TorchSpec, warm-started from their own DFlash checkpoint skipping the Markov heads | **830 tok/s at TP=1 (shipped)**; **>1000 tok/s at TP=4** | **4.6** |

Two details worth stealing outright:
- They reimplemented DSpark in TorchSpec rather than use DeepSeek's released trainer because that
  trainer *"required all hidden states to be stored on disk... ~38TB of hidden states"* — which is
  exactly the problem Baseten solved with live streaming.
- **TP made it slower until they stopped sharding the drafter.** *"The Markovian heads were doing
  collective operations because they were sharded across the ranks. We fixed it by replicating the
  weights instead."* They still shipped TP=1 because TP=4 underutilised the GPUs and wasn't
  cost-effective.

End result framed two ways, honestly: **2.6x throughput at single-user max interactivity**, and
**16x total throughput on the same single B200 under a 300 tok/s-per-user constraint.**
Evidence: <https://blog.fal.ai/how-we-achieved-1000-tok-s-and-16x-throughput-with-dspark-for-ideogram-v4-prompt-expander/>

---

## Doubleword

A UK inference provider (formerly TitanML) with access to Isambard-AI, the UK national AI
supercomputer. Their blog is written by founder Fergus Finn and is the most quantitatively honest
inference blog in this set — Modal cites his DeepSeek-V4-Flash B200 estimator as the basis for their
own speculative-decoding roofline tool.

### The decode-step roofline decomposition [verified]

This is the finding most worth copying *as a practice*. On 4xGH200 running DeepSeek-V4-Flash at
concurrency 2048 (ISL/OSL 1024/1024), they parsed the torch profiler at steady-state decode and
mapped every kernel to its inferred roofline ceiling:

| Bucket | Measured (per rank per step) | Their note |
|---|---|---|
| MoE expert GEMM | 30.3 ms | still at only ~30% of its roof |
| Act + quant | 12.6 ms | **~520 tiny kernel launches per step, of which ~2 ms is actual data movement** |
| Comms (AG + RS) | 10.8 ms | vs ~3 ms at wire rate |
| Dense GEMMs | 8.9 ms | |
| Attention (MLA) | 5.3 ms | |
| DSA indexer | 1.6 ms | |
| Other | 8.8 ms | not characterised |

The structure is strikingly close to our own C1 profile. The act+quant observation — a bucket that
looks like compute but is 84% launch overhead — is the kind of thing we should check for in our own
trace before optimising anything else.

### The optimisation ladder, with flags [verified]

| Step | Change | Result |
|---|---|---|
| Baseline | `vllm serve --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8 --block-size 256 --gpu-memory-utilization 0.92 --max-num-seqs 2048 --numa-bind` | **5,856 tok/s** at c2048 |
| DP attention | `-TP 4` → `+TP 1 +DP 4`, `--gpu-memory-utilization 0.95`, `--max-num-seqs 1024` (per rank), `--compilation-config '{"max_cudagraph_capture_size":1024}'` | **12,802 tok/s** at c2752 — **2.2x** |
| W4A8 MoE kernels | `--moe-backend humming`, `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`, `VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","input_scale_group_size":128}'` | **17,634 tok/s** — a further **+40%** |

Reasoning behind each:
- **DP attention:** MLA compresses KV heads into one shared latent, so TP has no head dimension to
  shard and must **replicate the KV cache N times**. DP attention gives each rank its own cache; the
  weights replicate instead, which is cheap for models with small attention weights. Bonus: "every
  per-token kernel (elementwise ops, FP8 quant, sampling) now runs on 1/N of the batch per rank,
  instead of the full global batch on every rank as it would under TP."
- **`--numa-bind`:** on multi-socket nodes, without it host↔device comms go over the CPU interconnect
  and throughput is lost. (Directly relevant — we have 2 NUMA nodes.)
- **W4A8 over Marlin:** vLLM defaults to Marlin, a weight-only kernel that dequantises FP4 to BF16 and
  runs BF16 tensor cores. Humming (Ant Group) upcasts to FP8 in the SMs and runs FP8×FP8 tensor cores —
  same weight transfer, double the FLOPs, double the critical batch size. They also note Humming's FP8
  path is **batch-invariant** and Marlin's isn't. Getting it working required two upstream fixes in
  Humming's group-scaled path (the WGMMA consumer never applied per-128 input scales on the
  accumulator; and the extra accumulator halves the per-tile register budget so the tile heuristic's
  default BlockM spills catastrophically). Fork: `doublewordai/humming`.

**Roofline caveat everyone should copy:** on Isambard, sustained BF16 tops out at **583 TF/s, not the
990 TF/s datasheet number**, because clocks drop once the GPU draws 530–570 W — a *software power cap*,
not thermal. That moves the ridge point from ~250 to ~146 FLOP/byte and the critical batch size to
~1,300, which is what made this a rare compute-bound MoE decode. **We should measure our own B200
sustained rate rather than trust the datasheet.**
Evidence: <https://blog.doubleword.ai/throughputmaxxing-v4-flash-single-node>

### Depth vs width: why speculation beats batching for MoE [verified]

Setup: batch size 1, single GPU, you may work on 2 token positions. Batch 2 sequences, or speculate
1 token with acceptance α? The naive answer is "batching always, since batched tokens are never
rejected." Their result: **spending both positions on one speculating sequence produces more global
output tokens/s than a batch of 2, even at α = 0.9.**

Mechanism: MoE routing is *"surprisingly non-uniform"* — expert utilisation decays roughly
exponentially by rank. Consecutive tokens within a speculative sequence co-activate **overlapping**
experts, so verifying two positions of one sequence touches a smaller distinct-expert working set
than two random sequences do. Because MoE decode is bound by expert weight movement, that is the
whole game.

Formal model for depth selection: expected committed tokens at depth γ for sequence i is
`m_i(γ) = 1 + Σ_{d=1..γ} Π_{k=1..d} a_k^i`, the cumulative product of the drafter's own per-depth
confidences. Recommendation: **per-sequence confidence gating with ragged depths**, trimmed greedily,
rather than a fixed batch-wide γ. They acknowledge engine support for ragged speculation is limited.
Artifacts: dataset `huggingface.co/datasets/Doubleword/specdec-calibration` (half a million draft
rounds per drafter, with per-depth confidence, committed counts, and full expert-routing captures
including for DeepSeek-V4-Flash); simulator `github.com/doublewordai/inference-lab`.
Evidence: <https://blog.doubleword.ai/speculating-on-the-margin>

### When to disaggregate [verified]

Analytical rather than measured — **they publish no A/B throughput numbers**, and say so implicitly.
Useful pieces:
- Balanced prefiller fraction: `φ = B·ISL / (B·ISL + r_P·t_step·OSL)`
- Fabric feasibility: `n_P·r_P·κ_store ≤ min(m_P·BW_P, m_D·BW_D, BW_fabric)`
- **GLM-5.2 on B200, concretely: 95 KB of KV per token, ~40B active params; a B200 at the FP8
  roofline (4.5 PFLOP/s) prefills 56k tokens/s, producing 5.3 GB/s of KV cache; a 400 Gb/s NIC gives
  50 GB/s.** So the fabric is ~10x oversupplied for one prefiller — the constraint is elsewhere.
- The real barrier is granularity: you lose fractional GPU allocation, and the loss scales as 1/N.
  DeepSeek V3's published shape was **EP32 prefill units and EP320 decode units** — thousands of GPUs
  before rounding stops hurting.
Evidence: <https://blog.doubleword.ai/when-to-disaggregate>

---

## Modal

**What they run.** Modal does not build an inference engine. They build the container/scheduling
substrate (custom Rust container stack, custom distributed filesystem, gVisor-based checkpoint/restore)
and tell users to run vLLM, SGLang or TensorRT-LLM on it. Their engineering value to us is
(a) the best free Blackwell attention writeup, (b) the GPU Glossary, (c) honest engine benchmarks.

### FlashAttention 4, reverse-engineered [verified]

The most useful public description of a production Blackwell attention kernel. Warp specialisation
in FA4:

| Warp role | Count | Job |
|---|---|---|
| Load | 1 | TMA-driven Q/K/V global→shared; supports a paged-KV page table; can hold up to 3 K and 3 V blocks in flight |
| MMA | 1 | `tcgen05.mma.cta_group::1` for QKᵀ→S and PV→O, emitted as inline PTX by a single `leader_thread` |
| Softmax | 8 (2 warpgroups) | online softmax, one per output tile workstream |
| Correction | 4 (1 warpgroup) | rescale O when the normaliser changes; also TMEM→SMEM and the final row-sum scale ("correction epilogue") |
| Epilogue | 1–2 | SMEM→global; 1 warp if TMA is available, 2 and much more complex if not |

Two changes flagged as "new in FA4", both portable:
1. **Software exponentials to dodge SFU contention.** FA3 used the `exp2` PTX intrinsic mapping to
   `MUFU.EX2` on the Special Function Units, of which there are far fewer than CUDA cores. FA4 mixes
   in a CUDA-core implementation on *some* iterations at a **tunable frequency**, only for **smaller
   head sizes**, and **stops applying it for a configurable number of the last S tiles** (wave
   quantisation makes SFU contention irrelevant at the tail). The implementation splits `2**x` into
   `2**floor(x)` and a **cubic polynomial approximation of `2**x` on the unit interval**, evaluated by
   Horner's method as three `fma.rn.ftz.f32x2` — and the approximation matches SFU output for bf16.
2. **Lazy rescaling.** Previously the running max updated the scale factor every time a new maximum
   appeared, forcing a correction pass. FA4 updates *"only when the maximum has changed enough to
   impact numerical stability."* Dao reported at Hot Chips that this **cut corrections by ~10x**.
   Modal's own annotation: "This seems like a good, and very portable, idea."

Also worth noting: FA4 deliberately uses `cta_group::1` (single-CTA MMA), avoiding Blackwell's
2SM/2CTA TPC-based matmuls, accepting a small memory-throughput penalty to simplify tile scheduling —
and ThunderKittens' Blackwell attention kernel made the opposite choice. Best performance requires
`StaticPersistentTileScheduler`, which launches at most one CTA per SM so Epilogue warps for one tile
overlap Load/MMA for the next. Reported ~20% faster than cuDNN.
Evidence: <https://modal.com/blog/reverse-engineer-flash-attention-4> [verified]

### GPU Glossary and LLM Almanac [verified]

The glossary is 84 pages across device-hardware / device-software / host-software / **perf**. The perf
section is the part worth mining: Little's Law, occupancy, latency hiding, pipe utilization,
scoreboard stall, bank conflict, warp divergence, roofline, arithmetic intensity, issue efficiency.
Sample content, Little's Law: `concurrency = latency × throughput`; a GPU at 1 instruction/cycle with
400-cycle memory latency needs 400 concurrent memory ops; at 10 instructions/cycle it needs 4,000.
Volkov's result that warps needed to hide *memory* latency (30) and *arithmetic* latency (24) are
nearly the same is a genuinely useful calibration.

The **LLM Almanac / Engine Advisor** is a live benchmark browser. Executive-summary findings from
their mid-2025 pass on 8xH100 single replicas (vLLM 0.8.x–0.9.x, SGLang 0.4.6-post5, TRT-LLM
0.20.0.rc3):
- **vLLM and SGLang were "strikingly similar"** on throughput out of the box; choose on time-to-market
  for features, not speed.
- **TensorRT-LLM was worse out of the box** and required extensive tuning with *"un- or
  under-documented flags"*; they warn the engineering lift and churn "should not be under-estimated."
- **Startup: vLLM ~5 min for an 8B model vs SGLang ~1 min**, attributed to torch-graph-compilation
  defaults in vLLM.
- Failures they published: TRT-LLM lacked Gemma 3 and Qwen 3 support; SGLang OOM'd on Qwen 3 235B.
- Cost datum: Llama 3.1 70B fp8 batch processing at **~20k tok/s for ~50¢ per million tokens**.

They also publish a **speculative-decoding roofline tool** (models B200/H200/H100 against
DeepSeek-V4-Flash/Pro and Qwen3.5 4B–397B; draft length 0–16, seqlen 4k–131k, batch 8–32, acceptance
75–89%, block size 1–16) producing speedups from 1.0x to **1.6x at γ\*=16** — with the honest caveat
that it *"tends to underestimate the benefit when overhead is a major contributor to latency, e.g.
small batch sizes on small models."* Derived from Fergus Finn's Doubleword estimator.
And a **block-quant reference**: MXFP4 = 32-element blocks with power-of-two E8M0 scales; NVFP4 =
16-element blocks with full **FP8 E4M3** scales plus a second per-tensor FP32 scale bridging the
dynamic range.
Evidence: <https://modal.com/gpu-glossary>, <https://modal.com/llm-almanac/advisor> and subpages [verified]

### Platform engineering: GPU memory snapshots [verified]

Cold-start work, not latency work, but the mechanism is worth knowing because it is the only public
account of using NVIDIA's CUDA checkpoint/restore API in production. On drivers in the **570 and 575
branches**: `cuCheckpointProcessLock()` blocks new CUDA calls and drains running ones,
`cuCheckpointProcessCheckpoint()` copies device memory and CUDA state to host and tears down sessions,
then the CPU snapshot is taken including GPU state; restore reverses via
`cuCheckpointProcessRestore()` + `cuCheckpointProcessUnlock()`. They enumerate active CUDA sessions
and PIDs, poll for `CU_PROCESS_STATE_CHECKPOINTED`, and only snapshot once no active sessions remain.
Integrated with their gVisor checkpoint/restore.

The reason it matters: it restores **already-compiled `torch.compile` artifacts, loaded CUDA kernels,
and captured CUDA graphs**, which the earlier CPU-only snapshots could not. Measured: vLLM with
Qwen2.5-0.5B-Instruct **45 s → 5 s**; ViT with `torch.compile` **8.5 s → 2.25 s**; NVIDIA Parakeet
**20 s → 2 s**. Enabled by `experimental_options={"enable_gpu_snapshot": True}`.
Evidence: <https://modal.com/blog/gpu-mem-snapshots>

### Blackwell/B200 specifically [verified]

B200 at $6.25/hr and H200 at $4.54/hr, self-serve. Their published spec table: B200 180 GB / 8 TB/s /
9 PFLOP/s FP4 / 5 PFLOP/s FP8, vs H200 141 GB / 4.8 TB/s / 2 PFLOP/s FP8, vs H100 80 GB / 3.5 TB/s.
Their measured comparison (vLLM, DeepSeek V3 in native FP8, 1000 in / 128 out, 8xH200 vs 8xB200):
**at 1 RPS, median TTFT is 2.5x faster on B200; at a median TTFT of 1 s, QPS is 1.7x higher.** They
note memory-bound workloads should see >2x from bandwidth alone with no code changes. Also a useful
GPU-utilization taxonomy: **Allocation Utilization** (GPU-seconds running app code / GPU-seconds paid
for), **Kernel Utilization** (what `nvidia-smi` reports), and **MFU**. They cite the State of AI
Infrastructure at Scale 2024 report that most organisations achieve <70% allocation utilisation
*at peak*, and the former Banana platform at ~20% aggregate; they claim >90% aggregate on Modal
(vendor claim, unverified).

---

## BentoML / Modular

BentoML **joined Modular in February 2026** as a strategic product acquisition. Their LLM Inference
Handbook now lives at `handbook.modular.com`. BentoML itself is a serving framework that wraps vLLM
et al.; they do not build an engine.

The handbook is a genuine reference with a few honest measured claims:

**Prefill-decode disaggregation page** — the most useful negative result in this file:
> "Performance can drop (by 20-30% in our tests)" for small workloads or untuned GPU setups.

Plus: high prefix-cache hit rates favour local prefill; short prompts don't justify the transfer
overhead; transports listed are NIXL, CXL and NVMe-oF; and cross-cluster "Prefill-as-a-Service"
reportedly gave 54% higher throughput and 64% lower P90 TTFT vs a standard disaggregation baseline
(no hardware given).

**Speculative decoding page** — acceptance length `τ = (1 − α^(γ+1)) / (1 − α)`, and a measured
Llama-3.3-70B-Instruct-on-H100 study:
- **TP=1:** ~2x TPOT improvement, but total throughput **plateaued earlier, around 20–30 concurrent
  requests**, than baseline.
- **TP=2:** throughput gains hold; γ=3 and γ=5 both beat baseline at 50 concurrent.
- **γ=5 on 2 GPUs showed larger latency spikes under heavy concurrency (40+ requests).**
- Actual speedups were consistently below theoretical predictions.
- Note on adaptive γ: SGLang switches between tiers like `[1, 3, 7]` using **pre-captured CUDA graphs
  so switching is cost-free** — this is the cheapest path to our dynamic-draft-depth item.
Evidence: <https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation/>,
<https://handbook.modular.com/inference-optimization/speculative-decoding/> [verified]

Their strategy post ("6 Production-Tested Optimization Strategies") is a well-organised map of
batching / prefill-decode / KV cache / attention-memory / parallelism / observability, but the
supporting evidence is customer case studies ("shipped 50% more models", "70% less dev time"), not
inference measurements. Useful for framing, not for technique.

---

## Replicate

**Honest verdict: nothing substantive on LLM inference.** Their blog is model launches, prompting
guides, fine-tuning tutorials, and diffusion work. They wrap Cog/containers rather than build an
engine. Two things are worth recording:

1. **`torch.compile` artifact caching** (Sep 8 2025). Cache compiled artifacts across container
   lifecycles, keyed on model version and stored close to GPU nodes; containers check for the cache
   on start and update it on graceful shutdown. Measured boot times: `flux-kontext-dev` **~120 s →
   ~60 s**; `prunaai/flux-schnell` **~150 s → ~70 s**; `prunaai/flux.1-dev-lora` **~400 s → ~150 s**.
   They also state the compiled version of `flux-kontext-dev` runs **>30% faster** than uncompiled.
   Modal's GPU memory snapshots solve the same problem more generally.
   <https://replicate.com/blog/torch-compile-caching> [verified]
2. **TaylorSeer for diffusion** — caching a truncated Taylor expansion of per-layer feature
   derivatives across timesteps rather than the naive "reuse last feature" cache. Not transferable to
   LLM decode. <https://replicate.com/blog/flux-kontext-optimization> [verified]

Their "fine-tuned models now boot in less than one second" post (2023) has no mechanism in it.

---

## RunPod

**Honest verdict: nothing transferable.** Their blog is buyer education (AI Infrastructure 101 series),
storage guides, and model-launch posts. The most technical piece is *GPU memory math for
full-parameter fine-tuning* (Aug 11 2026), which derives VRAM sizing equations for AdamW training
(≈8x weight bytes resident before activations; a 7B BF16 model needs ~112 GB, not the 14 GB the
inference rule of thumb suggests). Correct and useful for *training* capacity planning, irrelevant to
decode latency. They ship inference on vLLM/SGLang without publishing engine work.
<https://www.runpod.io/blog/gpu-memory-math-tuning-sizing> [verified]

---

## Techniques ranked by transferability to our stack

| # | Technique | Source | Attacks | Published effect | Difficulty | Confidence |
|---|---|---|---|---|---|---|
| 1 | **Warp decode** — one warp per output neuron for MoE decode; fold routing weight into a register accumulator; `shfl.sync.bfly` reduction | Cursor | MoE expert GEMMs (19.4% C1) | 1.84x MoE decode on B200; 3.95 TB/s @ B=32; +accuracy | High (new CUDA kernel) | High — mechanism fully described, tested on GLM-class shapes |
| 2 | **Pull dispatch / push combine + no cross-rank signalling** for MoE all-to-all | Cursor (MoK) | Collectives 19.6%, esp. the 47% arrival skew | signalling 103 µs → 18 µs; +29% NVLink util under imbalance | High | High — open source, but it's a training kernel |
| 3 | **Suffix-automaton speculation layered on the existing drafter** | Baseten (`sa_spec`, merged in TRT-LLM) | Decode steps | +30–33% accept length & throughput; up to 40% on agentic code; provably zero overhead | Low–Medium | High — code released and upstreamed |
| 4 | **Skip the DSA indexer when `seq_len < K`; fuse the high-precision indexer projections** | Baseten | DSA indexer (5.8% C1) | not quantified | Low | High — same architecture as ours |
| 5 | **RMSNorm / SwiGLU epilogue fusion; permute B's columns so gate/up are fragment-adjacent** | fal | Dense GEMM (37.1% C1) | removes an HBM round-trip per GEMM | Medium–High | High — layout facts are SM100-specific and exact |
| 6 | **Data-parallel attention instead of TP for MLA** | Doubleword; Baseten (ADP on standard tier) | Aggregate throughput / cost per user | 5,856 → 12,802 tok/s (2.2x) | Medium (SGLang supports it) | High — but it's a throughput lever, not a C1 lever |
| 7 | **Fuse draft into the target graph; never shard draft-head weights across TP ranks** | Baseten ("one big model"); fal (Markov-head fix) | Spec-dec host overhead | "overhead essentially zero"; fal: TP went from slower to >1000 tok/s | Medium | High — two independent confirmations |
| 8 | **Confidence-gated per-sequence draft depth using pre-captured CUDA graph tiers** | Doubleword (model + data); handbook (SGLang `[1,3,7]`) | Accept length at both C1 and C16 | model + open calibration dataset | Medium | Medium — simulated, not measured end-to-end |
| 9 | **Async collective overlap with functional collectives; scale-dependent fusion policy** | fal (8xB200) | Collectives | −23–25% chunk latency at 8 GPUs; fusion −37% at 2–4 but −4.6% at 8 | Medium | High for dense attention; unmeasured for MoE dispatch |
| 10 | **W4A8 instead of weight-only dequant-to-BF16 MoE kernels** | Doubleword (Humming vs Marlin) | MoE GEMMs | +40% on top of a 2.2x baseline | Medium | High on Hopper; on B200 native NVFP4 tensor cores may make this moot |
| 11 | **Per-expert L2 supergrouping in grouped GEMMs** | Cursor | MoE GEMMs | ~2650 TFLOP/s grouped MXFP8, only 4% below non-grouped; bad access patterns cost ~50% | Medium | High |
| 12 | **Lazy softmax rescaling — update the scale only when the max threatens stability** | Modal (FA4) | Attention (10.9% C1) | ~10x fewer correction passes (Dao, Hot Chips) | Low–Medium | High — Modal calls it "very portable" |
| 13 | **CUDA-core polynomial `exp2` mixed in with `MUFU.EX2` at tunable frequency** | Modal (FA4) | Attention | avoids SFU queueing; matches SFU output for bf16 | Medium | High |
| 14 | **KV-cache-aware routing on a radix tree with overlap+load scoring** | Baseten / Dynamo | TTFT (189 ms) at multi-replica scale | 50% ↓TTFT, 34% ↓TPOT, 62% ↑TPS at 89% hit rate | Medium | High — but only pays at ≥2 replicas |
| 15 | **Live draft training from streamed hidden states** (memory ∝ `max_num_tokens_per_iter`) | Baseten | Accept rate drift | +20% median accept rate, +100% on constrained traffic | High | Medium — no code released |
| 16 | **NVFP4 quantiser as ~100 lines of Triton with `cvt.rn.satfinite.e2m1x2.f32`** | fal | Quantisation overhead | matches/beats 2000-line CUDA at large shapes, ~7 TB/s on B200 | Low | High — code shown inline |
| 17 | **MXFP8/NVFP4 quantiser that emits scales directly in `tcgen05` packed layout** | fal (6+ TB/s), Cursor (6.2 TB/s) | Quantisation overhead | vs TE 4.4 TB/s / TorchAO 4.5 TB/s with reshape | Medium–High | High — two independent implementations agree |
| 18 | **Minibatch-size heuristic `T ≥ 2C·128·256/min(2I,H)` for comm/compute granularity** | Cursor | Collectives + MoE | predicted 2368 for Kimi 2.5; measured optimum 2560 | Low (it's a formula) | High |
| 19 | **Rust tokenizer with fused `encode_segments` and per-core chunk scheduling** | Baseten | TTFT | up to 18x vs tiktoken at 1M tokens, exact ID parity | Medium | High — but small absolute win at 10k input |
| 20 | **Prefill-priority scheduling; auto-disable speculation above a batch threshold** | Baseten | TTFT / C16+ throughput | not quantified | Low | Medium |
| 21 | **`--numa-bind`-equivalent pinning on multi-socket nodes** | Doubleword | Host↔device transfer | not quantified, but flagged as "important" | Low | Medium — we have 2 NUMA nodes, worth checking |
| 22 | **PD disaggregation** | Baseten (2x tok/s), handbook (−20–30%), Doubleword (needs 1000s of GPUs) | Aggregate throughput | contradictory across sources | High | **Low for our C1 objective** |
| 23 | GPU memory snapshots / `torch.compile` artifact caching | Modal, Replicate | Cold start | 10x / 2–3x | Medium | High, but orthogonal to our objectives |

---

## What I could not source

- **Baseten's PD-disaggregation configuration.** They report "2x higher tokens per second on
  disaggregated inference" for GLM-5.2 and promised a benchmarks post at Dynamo Day ("new benchmarks
  coming out in the next few weeks"), but I found no post giving prefill:decode worker ratios,
  transport, or the aggregated baseline config. The 2x claim is unconfigured and should be treated as
  marketing until reproduced.
- **Baseten's parallelism configs for GLM-5.x and Kimi.** They name ADP, TP+EP and "reduced max batch
  size" but publish no degrees. The only concrete config is TP8 EP1 for GPT-OSS 120B.
- **Baseten's "Still: Amortized KV Cache Compaction in a Single Forward Pass"** research post (listed
  on their research index, claims 200x KV cache shrink in one forward pass, dated June 10 2026). The
  URL I constructed 404'd and I did not find the real one. Flagging because a 200x KV compaction
  would matter enormously for our C64 objective — worth chasing.
- **Any fal post on LLM serving beyond the DSpark prompt-expander.** fal is diffusion-first; there is
  no fal equivalent of a "how we serve GLM" post.
- Two Modal posts (`the-hidden-economics-of-llm-inference`, `cuda-graphs`) returned empty bodies to
  my fetcher and I did not read them; they may contain relevant material.
- **No independent reproduction exists for any vendor number in this file.** Baseten's, fal's,
  Cursor's and Doubleword's numbers are all self-reported. Cursor's and Doubleword's are the most
  checkable because the code and benchmark harnesses are released.

---

## Sources

**Baseten**
- <https://www.baseten.co/blog/how-we-built-the-fastest-glm-5-api/> — DSA indexer kernels, fused MTP, NVFP4, 186+ tok/s
- <https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/> — NVFP4 from FP8 via ModelOpt, 280+ tok/s, TTFT 0.8 s / TTFAT 7.9 s, 2x from disaggregation
- <https://www.baseten.co/blog/how-we-built-the-new-fastest-api-for-glm-52/> — 601 tok/s, ADP on standard tier, TP+EP + reduced batch on Fast tier
- <https://www.baseten.co/blog/introducing-glm-52-fast/> — the Fast tier, `zai-org/GLM-5.2-Fast`
- <https://www.baseten.co/blog/how-we-built-the-fastest-kimi-k2-5-on-artificial-analysis/> — custom ~1B EAGLE-3, INT4→NVFP4, 340+ tok/s
- <https://www.baseten.co/blog/kimi-k2-thinking-at-140-tps-on-nvidia-blackwell/> — 8xB200, TP+EP mixture, INT4→BF16→NVFP4 route, 140+ tok/s / 300 ms TTFT
- <https://www.baseten.co/blog/how-we-made-the-fastest-gpt-oss-on-nvidia-gpus-60-percent-faster/> — TP8 EP1 on 8xB200, EAGLE-3, 400→650 tok/s, what they were *not* doing
- <https://www.baseten.co/blog/sota-performance-for-gpt-oss-120b-on-nvidia-gpus/> — TRT-LLM dev builds, TP vs EP tradeoff, Blackwell-only MoE backend
- <https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/> — suffix automaton MTP accelerator, full API and zero-overhead proof
- <https://github.com/basetenlabs/sa_spec> — `SA_SPEC_THRESH` default 4, TRT-LLM PRs #11434 / #10951
- <https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/> — KV routing, Qwen3 Coder 480B, 89% hit rate, OpenRouter shadow test
- <https://www.baseten.co/blog/nvidia-dynamo-day-baseten-inference-stack/> — engine-agnostic stance, AIConfigurator, KV offload via NIXL
- <https://www.baseten.co/blog/how-to-train-custom-eagle-3-heads-for-speculative-decoding/> — TTT-length, LR table, draft-token counts, the regenerate-with-target rule
- <https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/> — streamed hidden states, UCXX, Trio, +20% accept
- <https://www.baseten.co/blog/dflash-faster-llm-inference/> and <https://arxiv.org/abs/2602.06036> — block-diffusion drafting, Qwen3-8B B200 numbers
- <https://www.baseten.co/blog/making-kimi-k3-tokenization-18x-faster-for-million-token-agentic-workloads/> — Basetenkenizer internals and honest comparisons
- <https://www.baseten.co/blog/how-to-build-a-day-zero-api-for-kimi-k3/> — GB300 NVL72 topology, config sweep list, frontend-is-the-quality-risk
- <https://www.baseten.co/blog/inference-engineering-for-deepseek-v4-pro-0813/> — MXFP4 native, DSpark
- <https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm/> — draft/target contention, chunked-prefill scheduler bug, full YAML configs, the benchmark that got worse
- <https://www.baseten.co/blog/how-the-baseten-delivery-network-bdn-makes-cold-starts-fast/> and <https://www.baseten.co/blog/baseten-delivery-network-fast-cold-starts-big-models/> — hash-ring peer cache, single-flight, >2 GB/s
- <https://www.baseten.co/resources/guide/the-baseten-inference-stack/> — request prioritization, spec-dec auto-disable, KV disk offload, structured output via logit biasing
- <https://www.baseten.co/blog/how-to-run-llm-performance-benchmarks-and-why-you-should/> — InferenceMAX reproduction, TRT-LLM image tag, server-side benchmarking
- <https://www.baseten.co/blog/accelerating-inference-nvidia-b200-gpus/> — B200 platform claims
- <https://www.baseten.co/blog/ai-model-performance-metrics-explained/> — TTFT vs TPS vs E2E framing
- <https://docs.baseten.co/engines/engine-builder-llm/engine-builder-config> — full config surface, lookahead flags, `guaranteed_no_evict`
- <https://www.baseten.co/research/> — research index (Still, KV compaction, continual learning)

**Cursor (Anysphere)**
- <https://cursor.com/blog/warp-decode> — output-centric MoE decode, 1.84x on B200, 3.95 TB/s @ B=32
- <https://cursor.com/blog/kernels> — MXFP8 recipe, `tcgen05` constraints, expert-wise supergrouping, DeepGEMM comparison, quantiser at 6.2 TB/s
- <https://cursor.com/blog/mixture-of-kittens> — push/pull NVLink analysis, signalling latency, minibatch heuristic, ring token buffers, CLC
- <https://github.com/cursor/mixture-of-kittens> — open source

**fal**
- <https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/> — 8xB200 overlap/fusion sweep, symmetric memory, "timeline overlap ≠ throughput overlap"
- <https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/> — CuTeDSL quantiser, K-split, TMA, `STG.E.U8` bottleneck, swizzling negative result
- <https://blog.fal.ai/instruction-level-control-with-inline-elementwise-asm-in-triton/> — `tl.inline_asm_elementwise`, `cvt.rn.satfinite.e2m1x2.f32`, NVFP4 in <100 lines
- <https://blog.fal.ai/serving-sub-second-ideogram-v4-without-quality-loss/> — SM100 tile/fragment layout, revisit epilogue, SwiGLU column permutation, QAD
- <https://blog.fal.ai/how-we-achieved-1000-tok-s-and-16x-throughput-with-dspark-for-ideogram-v4-prompt-expander/> — MTP → DFlash → DSpark ladder on one B200, TP drafter-sharding trap
- <https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/> — companion EVT writeup

**Doubleword**
- <https://blog.doubleword.ai/throughputmaxxing-v4-flash-single-node> — DP attention 2.2x, Humming W4A8 +40%, per-kernel roofline decomposition, power-cap roofline correction
- <https://blog.doubleword.ai/speculating-on-the-margin> — depth-vs-width, expert-overlap argument, `m(γ)` model, `specdec-calibration` dataset
- <https://blog.doubleword.ai/when-to-disaggregate> — allocation and fabric equations, GLM-5.2-on-B200 KV rate numbers

**Modal**
- <https://modal.com/blog/reverse-engineer-flash-attention-4> — warp specialisation table, software `exp2`, lazy rescaling, `cta_group::1`
- <https://modal.com/blog/gpu-mem-snapshots> — CUDA checkpoint/restore API, driver branches, measured cold boots
- <https://modal.com/blog/introducing-b200-h200> — specs and the DeepSeek V3 H200-vs-B200 comparison
- <https://modal.com/blog/gpu-utilization-guide> — the three utilizations
- <https://modal.com/gpu-glossary> (84 pages; `perf/` section notably) and <https://modal.com/gpu-glossary/perf/littles-law>
- <https://modal.com/llm-almanac/advisor>, <https://modal.com/llm-almanac/summary>, <https://modal.com/llm-almanac/spec-dec-roofline>, <https://modal.com/llm-almanac/block-quants>

**BentoML / Modular**
- <https://handbook.modular.com/> (LLM Inference Handbook, formerly bentoml.com/llm)
- <https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation/> — the −20–30% negative result
- <https://handbook.modular.com/inference-optimization/speculative-decoding/> — τ formula, Llama-3.3-70B H100 TP=1 vs TP=2 study
- <https://www.bentoml.com/blog/6-production-tested-optimization-strategies-for-high-performance-llm-inference>
- <https://www.bentoml.com/blog/bentoml-is-joining-modular>

**Replicate**
- <https://replicate.com/blog/torch-compile-caching> — artifact caching boot times
- <https://replicate.com/blog/flux-kontext-optimization> — TaylorSeer
- <https://replicate.com/blog/fine-tune-cold-boots>

**RunPod**
- <https://www.runpod.io/blog/gpu-memory-math-tuning-sizing>
