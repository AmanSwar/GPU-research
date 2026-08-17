# SGLang: what the engine we fork actually implements, and what it has published

## What this is

A mine of SGLang's own engineering output, read at source, for techniques we can
lift into `NotSglang` on 8xB200 running GLM-5.2. It is not a survey of what
SGLang *is*; it is a list of mechanisms with evidence, plus three specific
answers the team asked for.

Two evidence channels were used and they are labelled differently throughout:

- **Published prose** — LMSYS blog posts, release notes, docs, cookbook pages,
  GitHub issues and PRs. Fetched at the URLs given. Marked `[verified]` when I
  read the page itself.
- **Source** — upstream `sgl-project/sglang` cloned at
  `f019f0b064ff13ef0700088f37dd44cd0d791b8d` (HEAD, Mon 17 Aug 2026, PR #35068
  merged), compared against our fork at `/home/aman/code/NotSglang`. Every file
  path and line number below is from that clone unless it says "fork". Source
  reads are `[verified]` — I read the code — but a source read tells you what
  the code does, not what it is worth in wall-clock.

Labels used: `[verified]` I read it at the URL/path given · `[reported]` SGLang
or a vendor claims it and I have not reproduced it · `[inferred]` my reasoning
· `[unverified]` I could not source it.

A standing caveat on every number in this file: **all of them are SGLang's own,
none are independently reproduced, and almost all carry a config asymmetry** —
different hardware (B300/GB300, not B200), different acceptance-length regime
(several headline cells use a *simulated* acceptance length via
`SGLANG_SIMULATE_ACC_LEN`, not real acceptance), different input/output shape,
and different cache-hit assumptions. I flag the asymmetry at each table.

---

## Bottom line for our system

Ranked by expected effect on our two objectives (C1 single-stream latency;
cost/user at C64), with difficulty. Everything here has a mechanism section
below.

| # | Steal | Expected effect | Difficulty | Why |
|---|---|---|---|---|
| 1 | **Unblock draft depth 5-1-6** (see Q2) | Published B200 NVFP4 low-latency cell is **TPOT 1.85 ms / 295 ms TTFT** at 5-1-6 vs our 2.74 ms at 3-1-4. Even half the gap is ~+80 tok/s | Medium — three published candidate root causes, one is an 11-line patch | Our single biggest known cap is a crash, not a kernel |
| 2 | **`--bf16-gemm-backend cutedsl`** (FlashInfer TGV BF16 GEMM) | LMSYS measured **~4% end-to-end decode at bs=1** on GLM-5.2; 1.08x avg on fused QKV, 1.05x on `o_proj` | Low — one flag; `auto` already picks it on SM10x since v0.5.17 | Dense GEMM is **37.1%** of our C1 kernel time and `nvjet_sm100_*` = cuBLAS. This is a direct hit on our top hotspot |
| 3 | **Confirm TopK-V2 is dispatching** (`SGLANG_OPT_USE_TOPK_V2=1`, default on) | 40.7 → 17.5 µs per call at 80K ISL (2.33x); scales to 10.17x at 1M | Low | Our DSA indexer is 5.8%. At AA's 10k ISL the win is smaller than the blog's 80K figure — verify with a trace before believing it |
| 4 | **Post-warmup `synchronize()` + TP `barrier()` before `torch.cuda.CUDAGraph()`** (PR #33795) | Removes a JIT-inside-capture race that manifests as a *random-kernel* IMA at capture. Fork does **not** have it | Low — 2 lines in `full_cuda_graph_backend.py:105-114` | Leading published explanation for capture-time IMAs whose reported kernel varies run to run |
| 5 | **Route the small draft-side TP all-reduces through FlashInfer `kAllReduce`** (PR #32461, `SGLANG_FLASHINFER_SMALL_AR_MAX_BYTES`) | Measured **374.83 → 381.85 tok/s (+1.87%)** at bs=1 on GLM-5.2-FP8 TP8; NCCL all-reduces/spec-step **12 → 0** (0.450 ms) | Medium — PR is closed unmerged with red CI; port by hand | Collectives are **19.6%** of our C1 time. This targets exactly the NextN draft's unfused ARs |
| 6 | **Do NOT chase TBO** | Zero at C1 by construction; SGLang itself disabled TBO on NVLink-domain hardware | — | See Q1 and the TBO section. Saves us weeks |
| 7 | **`--chunked-prefill-size 32768` for the C10/C64 arms** | Cookbook: **+34–78% output tok/s, −39–59% TTFT** at 8K-in/1K-out on 8xB200; neutral at high throughput | Low | AA's shape is ~10k input. TTFT is half the AA headline |
| 8 | **Investigate the EAGLE-vs-radix-reuse collapse** (issue #32459) | Reported 97% → 40–53% deep-prompt cache hit with EAGLE on GLM-DSA NVFP4 8xB200 | Medium — reproduce first | AA is P50 over 72 h of a repeated ~10k prompt. If spec kills prefix reuse, our TTFT is structurally worse than it needs to be |
| 9 | **Indexer prologue fusion** (PR #27705) — check we have the *full* fused pair, not just `wk_weights_proj` | 12 kernels → 4; **+8% decode at bs=1**, +5% at bs=128 | Low if on a recent base (fork has `wk_weights_proj`) | Small-kernel launch cost dominates our C1 profile |
| 10 | **TCP-layer TTFT fixes** (PR #33026: `SO_RCVBUF` 16 MiB on the listener, `TCP_NODELAY`) | Rust server: fresh-connection TTFT 192 → 157 ms at 16K input; a raw-socket probe went **208 ms → 0.4 ms** | Low | AA measures TTFT **over the public internet from GCP us-central1-a**. A 200 ms RTO retransmit on first-request body would be pure, invisible loss |
| 11 | **Adopt the published launch cells verbatim as a control arm** (see Q3) | Removes "did we configure it wrong" from every regression argument | Low | We are chasing a published number; run their exact command first |
| 12 | **Ignore SBO at TP8/`--moe-a2a-backend none`** | SBO's overlap code lives inside `forward_deepep`; with no a2a backend it never runs | — | Prevents a wasted flag-flip experiment |

Two things NOT to steal, with reasons: **EPLB** (published gains are 1.49x
prefill / 2.54x decode but measured at EP32–EP72 with large batches; at C1 with
8 of 256 experts active per token there is no statistical imbalance to
rebalance) and **HiCache/HiSparse** (they buy capacity and TTFT on
cache-friendly multi-turn traffic; AA's single-prompt scenario and our C1
latency objective get nothing).

---

## SGLang core runtime: the scheduler, overlap, and CUDA graphs

### What they run

A Python scheduler process driving model-worker processes, with the front half
(ingress → tokenized request) being migrated to Rust as of v0.5.17
([#29799](https://github.com/sgl-project/sglang/pull/29799)) `[verified]` in
release notes. Continuous batching with chunked prefill; a "zero-overhead"
overlap scheduler; full and breakable CUDA graphs.

### Zero-overhead batch scheduler

**Mechanism** `[verified]`
(<https://lmsys.org/blog/2024-12-04-sglang-v0-4/>): the scheduler runs *one
batch ahead*, preparing all metadata for batch N+1 while the GPU executes batch
N. Dependencies are resolved by creating **future tokens** — placeholder token
IDs that get materialised when the previous forward's sampling lands — plus
explicit CUDA event scheduling. Idea credited to NanoFlow. Claimed **1.1x** over
SGLang v0.3 and **1.3x** over then-SOTA baselines, largest on small models and
large TP. On by default; `--disable-overlap-schedule` turns it off.

**Source detail** `[verified]`, `python/sglang/srt/managers/scheduler.py`:

- Three streams: `schedule_stream` (priority 0), `forward_stream`,
  `copy_stream`. The scheduler explicitly **re-draws `schedule_stream` up to 64
  times if PyTorch's round-robin stream pool hands it the same handle as
  `forward_stream`**, because aliasing would silently eliminate the overlap
  (`scheduler.py:1682-1696`). That is a real trap worth copying.
- `_apply_war_barrier()` (`scheduler.py:1702-1718`): after each launch, the
  scheduler orders its *next* shared-buffer write behind the forward's
  shared-buffer **read**-done event, not behind the whole forward. Fast path
  waits on `runner.shared_read_done_event`; coarse fallback is
  `wait_stream(forward_stream)`, forceable with
  `SGLANG_FORCE_COARSE_WAR_BARRIER`. Under speculative decoding the
  read-done event is published by **draft_extend**, "the step's last
  shared-buffer-reading phase" (`speculative/eagle_worker_v2.py:470-475`).
- Result D2H runs on `copy_stream` gated by `copy_done`, so it overlaps the next
  forward instead of serialising on `forward_stream`
  (`scheduler.py:3734-3738`).

**Why it matters to us** `[inferred]`: this is the machinery whose failure shows
up as inter-iteration bubbles. If our C1 device-busy is 83.7%, the 16.3% idle is
exactly the quantity this subsystem exists to remove, and the GLM-5.2 blog's
"11% end-to-end TPS" came from closing four specific holes in it.

### Spec V2 (the overlap scheduler for speculative decoding)

**Mechanism** `[verified]`
(<https://lmsys.org/blog/2026-07-13-glm52-optimization/>): "While the GPU runs
the current model forward on the forward stream, it does the next step's KV
allocation and metadata preparation on the plan stream." Turned on by default.
The blog names the four fixes that were needed before the overlap actually paid:

1. made the DSA **draft-extend path CUDA-graphable**,
2. made `seq_lens_cpu` **optional for DSA** to drop a D2H sync,
3. removed the remaining **H2D** syncs,
4. **fused the small eager metadata ops** in `_apply_cuda_graph_metadata`.

Result: **11% end-to-end TPS**, and the published trace shows "no bubble between
`run_batch` iterations".

**Source** `[verified]`: the plan stream is gated by
`SGLANG_ENABLE_OVERLAP_PLAN_STREAM` (`environ.py:1114`, **default `False`**),
constructed in `speculative/spec_utils.py:1047-1055`. The DFlash launch command
in the LMSYS blog exports `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` explicitly
`[verified]`
(<https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/>).
`[inferred]` — worth an A/B on our stack; it is off by default and the one
published launch command that uses it is a latency-oriented one.

`decide_needs_cpu_seq_lens()` (`managers/overlap_utils.py:23-48`) is the
D2H-elimination switch: it ORs per-backend `needs_cpu_seq_lens`, and **forces
`True` under TBO** — "FIXME: support TBO without seq lens cpu value". So
enabling TBO reintroduces a host sync the DSA path worked to remove.

### CUDA graph padding and bucketing

**Source** `[verified]`,
`model_executor/runner/base_cuda_graph_runner.py:64-102`:

- Capture list comes from `cuda_graph_config.decode.bs`, then is filtered:
  `bs * alignment_width % mul_base == 0` where `mul_base =
  get_cuda_graph_batch_size_alignment(server_args)` (attn-TP/CP alignment) and
  `alignment_width = captured_req_width`, **forced to 1 when TBO is on**
  ("TBO splits each request's rows across two micro-batches, so the alignment
  constraint applies per request rather than per token row").
- Then clamped to `req_to_token_pool.size`, and if `max(capture_bs) >
  num_max_requests` the max is appended so the top bucket always exists.
- Replay pads with `_pad_to_bucket(raw_size, buckets)` = smallest bucket ≥ raw.
- `--disable-cuda-graph-padding` is **rejected together with
  `--enable-torch-compile`** because it would force one compile+autotune cycle
  per distinct batch size (`server_args.py`, `check_server_args`).

`[inferred]` For our C1 target the capture list should be *dense at the bottom*
(1,2,3,4…) and short — every extra bucket is capture time and memory, and issue
#32459 reports 5-1-6 booting at `--max-running-requests 128` and **crashing at
capture with 256** on 8xB200 NVFP4, i.e. the capture list length itself is part
of the failure surface.

**Breakable / piecewise CUDA graph**: v0.5.17 made **breakable prefill CUDA
graph on by default for DP attention** ([#31682](https://github.com/sgl-project/sglang/pull/31682))
`[verified]` in release notes; there are `tc_piecewise` and `full` backends
selectable per phase (`--cuda-graph-backend-prefill tc_piecewise` appears in the
DFlash launch command).

---

## RadixAttention, HiCache, and the Unified Radix Cache

### Tree and eviction

**Source** `[verified]`, `mem_cache/`:

- `radix_cache.py:376-406` — `match_prefix(params)` walks the tree for the
  longest cached prefix. The **namespace is `(token ids, extra_key)`**: entries
  with the same leading tokens but different `extra_key` are deliberately kept
  disjoint (LoRA IDs, sampling salt, cache version). With `page_size > 1` the
  key is truncated to a multiple of page size before matching. Matching **may
  split an existing node** if the match ends mid-segment.
- `radix_cache.py:592-604` — `evict()` builds a heap over
  `self.evictable_leaves` keyed by `eviction_strategy.get_priority(node)`.
- `evict_policy.py` — pluggable strategies, all present: `LRU`
  (`last_access_time`), `LFU` (`(hit_count, last_access_time)`), `FIFO`
  (`creation_time`), `MRU`, `FILO`, `Priority` (`(node.priority,
  last_access_time)`), and `SLRU` — a two-segment probationary/protected scheme
  with `protected_threshold=2` hits, where segment always dominates recency.
- `TreeNode` carries `host_value`, `host_ref_counter`, `protect_host()` /
  `release_host()` so a node's host (L2) copy has an independent lifetime from
  its device copy.

### Cache-aware routing

**Mechanism** `[verified]`
(<https://lmsys.org/blog/2024-12-04-sglang-v0-4/>): the router keeps an
**approximate** radix tree mirroring each worker's real tree, updated lazily
from routed requests — "communication-free", no worker synchronisation. Written
in Rust; claimed 2x over a Python router. Reported **82,665 → 158,596 tok/s and
20% → 75% cache hit rate** on a synthetic multi-long-prefix-group workload,
8x A100 80GB. `[reported]` — synthetic workload, 2024 hardware, useless as an
absolute number but the mechanism is sound.

Today this lives in **SGLang Model Gateway** (`docs/docs/advanced_features/sgl_model_gateway.mdx`)
`[verified]`: a Rust gateway with cache-aware and power-of-two policies fed by a
Load Monitor, plus PD routing, gRPC, circuit breakers, and K8s service
discovery.

### HiCache

**Mechanism** `[verified]` (<https://lmsys.org/blog/2025-09-10-sglang-hicache/>):
a `HiRadixTree` acting as a page table over GPU + CPU + external tiers.

- **Data plane**: GPU-assisted I/O kernels giving up to **3x** over
  `cudaMemcpyAsync` for CPU↔GPU; host pool uses a **page-first** layout while
  the GPU pool stays layer-first, so host↔storage transactions are large —
  up to **2x** with zero-copy.
- **Control plane**: layer-wise overlap (load layer N+1 while layer N computes)
  for the CPU→GPU tier; opportunistic prefetch for the storage tier with
  configurable policies (`best-effort`, terminate-on-due, aggressive staging).
- **Write policies**: `write_through`, `write_through_selective` (hit-count
  gated, back up only hot spots), `write_back`.
- **Backend contract is three functions**: `get(key)`, `exist(key)`,
  `set(key, value)`. Mooncake, 3FS, NIXL, local file implemented.
- Flags: `--enable-hierarchical-cache --hicache-ratio 2 --hicache-io-backend
  kernel --hicache-mem-layout page_first --hicache-storage-backend {hf3fs,
  mooncake, nixl, file} --hicache-storage-prefetch-policy {wait_complete,
  timeout, best_effort} --hicache-write-policy write_through`.

Numbers `[reported]`: up to **6x throughput and 80% TTFT reduction** in LMSYS's
own measurement; Novita reports 56% average TTFT drop and 40% → 80% hit rate on
a Qwen3-Coder-480B agent workload; Ant Group reports 84% TTFT reduction on cache
hits for DeepSeek-R1-671B under PD. All community-supplied, none reproduced.

### Unified Radix Cache (Aug 2026)

`[verified]` <https://lmsys.org/blog/2026-08-11-unified-radix-cache/>. The
newest cache architecture, relevant to us mainly as a signal of where the code
is going.

**Mechanism**: one token-keyed radix topology provides canonical prefix
identity; reuse *validity* is delegated to `TreeComponent`s — `FULL` (path
reuse), `SWA` (trailing-window reuse), `MAMBA` (exact-checkpoint reuse). During
matching each active component creates a validator and **the reusable boundary
advances only when every validator accepts the candidate node**; traversal
continues past a rejection because a later node may be accepted. Component hooks
cover match / split / insert / lock / evict. Pools that do not define a reuse
boundary register as **sidecars** that follow a declared source pool's indices —
DeepSeek-V4's C4 and C128 compressed-KV pools, indexer buffers, and compressor
states are all sidecars.

Numbers `[reported]`: L3 (500 GiB Mooncake) held later-round hit rates near
**98%** on DeepSeek-V4-Flash and **96.8%** on Inkling-Small; session-aware
eviction gave **2.9–16.6% lower TTFT** than HiRadixCache+LRU on SWE-bench
replays; an experimental **Rust tree core** gave up to **42% lower TTFT over
turns 176–200** of a sliding-window benchmark. Enabled with
`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`.

Also in v0.5.17: `--enable-session-radix-cache` — requests carry a stable
`session_id` so eviction knows which prefixes an active session still
references; released with `/close_session`
([#29173](https://github.com/sgl-project/sglang/pull/29173)) `[verified]`.

---

## Speculative decoding: EAGLE / EAGLE-3 / MTP, and what has replaced them

### Worker structure

`[verified]`, `python/sglang/srt/speculative/`. 17,020 lines across 29 files.
The V2 workers are the default; V1 is deprecated.

A counterintuitive fact stated in the DFlash blog `[verified]` and confirmed in
source: **the draft worker is the one the scheduler talks to.** `EAGLEWorkerV2`
wraps the target worker and calls it for verification — `forward_batch_generation`
lives on the draft worker (`eagle_worker_v2.py:1160`). Read traces with that in
mind.

`_capture_cuda_graphs()` (`eagle_worker_v2.py:343-490`) captures **two** graph
families:

1. **draft decode**, only if `speculative_num_steps > 1`, via
   `EAGLEDraftCudaGraphRunner` — logged as `num_tokens_per_req={self.topk}`.
2. **draft extend**, via `EAGLEDraftExtendCudaGraphRunner`, logged as
   `num_tokens_per_req={self.speculative_num_draft_tokens}`, gated on the
   draft-extend attention backend being in a hard-coded allowlist
   (`TritonAttnBackend`, `TRTLLMMLABackend`, `TRTLLMHAAttnBackend`,
   `TokenspeedMLABackend`, `FlashInferAttnBackend`, plus
   `DeepseekSparseAttnBackend`, `DeepseekV4AttnBackend`, `FlashMLABackend` on
   CUDA), and disableable with `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH`.

Draft-decode static buffers scale with depth
(`eagle_draft_cuda_graph_runner.py:167-175`):
`out_cache_loc = zeros(max_num_token * speculative_num_steps)` where
`max_num_token = max_bs * captured_req_width`.

### topk / steps / num-draft-tokens

- `--speculative-num-steps` = autoregressive drafting depth;
  `--speculative-eagle-topk` = branching factor per step;
  `--speculative-num-draft-tokens` = max tokens submitted for verification
  `[verified]`, `docs/docs/advanced_features/speculative_decoding.mdx`.
- **The overlap scheduler only supports `--speculative-eagle-topk 1`.** The doc
  is emphatic: set it explicitly, because auto-tuning may pick `topk > 1` for
  some models and that "may not always trigger an immediate config error"
  `[verified]`.
- `--speculative-attention-mode {prefill, decode}` (default `prefill`) selects
  the attention mode for speculative ops.

### Known limitations, stated

`[verified]`, docs + issues:

- topk > 1 is incompatible with the default overlap path (above).
- OOM guidance is a documented ladder: lower `--mem-fraction-static`, then
  `--cuda-graph-max-bs-decode`, then the draft tree size, then
  `--max-running-requests`.
- GLM-5.2 cookbook: `index_share_for_mtp_iteration` is **"effective only at
  `--speculative-eagle-topk 1`"**.
- AMD gfx950: MTP is omitted from the published recipes; "at
  `--speculative-num-steps > 3` hits a separate build issue" `[verified]`,
  `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx`.

### The tuning rule SGLang actually publishes

`[verified]`, GLM-5.2 cookbook: *"Tune the draft length to the accept length…
while accept length stays close to the draft-token count there is headroom to
push them higher; if it falls well below, lower them — every rejected draft
token is wasted verification compute."* And the observed regime for GLM-5.2:
accept length "4+ in many workloads, near-saturating at 5–6 in low-latency
runs."

This is why every published GLM-5.2 low-latency cell is **5-1-6** and every
balanced cell is **1-1-2** (FP8) or **2-1-3** (NVFP4), with the NVFP4 balanced
cell carrying an explicit comment: *"Shorter draft than low-latency's 5-1-6: at
this concurrency the verify overhead of a long draft outweighs the accept-length
gain."*

### What is replacing EAGLE

Two published successors, both with real numbers, both worth watching:

**DSpark** `[verified]` <https://lmsys.org/blog/2026-07-06-dspark-sglang>.
Semi-autoregressive block drafter (whole gamma-token block per draft forward,
with a lightweight Markov/RNN sequential head) + a **confidence head** whose
per-token survival probabilities, calibrated by *Sequential Temperature
Scaling*, drive a **per-request verify budget**. Three verify modes via
`SGLANG_RAGGED_VERIFY_MODE={static,compact,cap-accept}`. The engineering that
makes it pay:

- **Ragged verify under full CUDA graphs**: per-request windows are front-packed
  into one `cu_seqlens`-style varlen buffer, and the graph is keyed on the
  **total token count** rounded to the nearest captured tier — so a trimmed
  batch replays a genuinely smaller graph, not a masked full-width one. Under DP
  attention all ranks share the largest tier any rank needs and step down
  together.
- **Additive cost model** `T(bs, K) = bias + alpha(bs) + theta(M)`, `M = bs + K`,
  profiled offline (`sglang.benchmark.dspark_sps_profiler`) and read online.
  Only `theta` is recoverable by trimming, which is why trimming is a
  high-batch-only win.
- `[reported]` **383.7 tok/s at accept length ~5, batch size 1, DeepSeek-V4-Pro,
  TP8, B300.** Decode loop "about 1.5x tighter" with ZOS on.
- Mixed-traffic result: window contracts 5.24 / 3.78 / 2.91 tokens across
  gsm8k / arena-hard / poetry while utilisation against the untrimmed ceiling
  stays 0.88–0.97.
- **Negative result stated plainly**: "At batch size 1 the target verify does not
  slow down much with more tokens, so trimming saves little and the two arms
  tie." `[inferred]` So for our C1 objective DSpark's *scheduler* is worth
  nothing; its *block drafter* is the interesting half.
- GLM-5.2 DSpark support is roadmapped but not shipped: the GLM-5.2 blog's
  "What's Next" lists "Supporting DSpark for GLM 5.2", and there is a separate
  roadmap issue [#30734](https://github.com/sgl-project/sglang/issues/30734).

**DFlash / DFlash-v2** `[verified]`
<https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/>.
Block-diffusion drafter that emits a whole block in one forward, plus **KV
injection**: the target model's hidden states are pushed through the draft
model's KV projection and written into *every* draft layer's KV cache, instead
of only conditioning the draft input as EAGLE-3 does. Ablations on Qwen3-4B,
reported as `acc_len / speedup`:

| Task | EAGLE-3 (5 layers) | DFlash | diffusion only | injection only |
|---|---|---|---|---|
| GSM8K | 4.2 / 2.1x | 4.2 / **3.3x** | 3.5 / 2.9x | 4.8 / 2.4x |
| HumanEval | 4.3 / 2.2x | 4.0 / **3.2x** | 3.5 / 2.9x | 4.6 / 2.3x |
| MT-Bench | 3.1 / 1.4x | 3.0 / **2.2x** | 2.6 / 2.0x | 3.4 / 1.5x |

`[reported]`, and the headline: Qwen3.5-397B-A17B BF16 on **8xB200 (Modal)**,
HumanEval, greedy, thinking on, concurrency 1 — DFlash at block size 16 achieves
**>4.3x baseline and 1.5x MTP** (MTP at 7 steps). The ablation table is the
important part: **KV injection buys acceptance, block diffusion buys draft
latency, and the two are independent.** `[inferred]` For GLM-5.2 the MTP head is
already strong (accept ~5), so the acceptance half is nearly saturated and the
*draft latency* half is where an EAGLE→DFlash-style change would pay — which is
exactly the axis a deeper draft chain (5-1-6) attacks by amortisation.

---

## MoE, expert parallelism, and the two overlap schemes

### DeepEP and DeepGEMM

`[verified]` <https://lmsys.org/blog/2025-05-05-large-scale-ep/> and
`docs/docs/advanced_features/expert_parallelism.mdx`.

- **DeepEP** has two dispatch modes: `normal` (throughput, long sequences,
  **incompatible with CUDA graph**) and `low_latency` (decode, fixed shapes,
  CUDA-graph compatible). `--deepep-mode auto` switches at runtime; PD
  disaggregation makes the choice per role (`normal` on prefill, `low_latency`
  on decode).
- **DeepGEMM** supplies **contiguous-layout** grouped GEMMs for prefill (dynamic
  shapes) and **masked-layout** grouped GEMMs for decode (fixed shapes, graph
  compatible). Normal-dispatch output needs a **custom Triton permute kernel**
  to reach contiguous layout.
- `SGL_ENABLE_JIT_DEEPGEMM=1` (blog); today `SGLANG_ENABLE_JIT_DEEPGEMM` is
  `EnvBool(True)` by default `[verified]`, `environ.py:951`.
- DeepEP-family backends require `ep_size == tp_size`; for hybrid EP<TP only the
  `none` backend (all-reduce / all-gather dispatch) works `[verified]`, EP doc.

### Two-Batch Overlap (TBO)

**Mechanism** `[verified]`, `python/sglang/srt/batch_overlap/`. The layer is
decomposed into named ops (`op_comm_prepare_attn`, `self_attn.op_prepare`,
`self_attn.op_core`, `op_comm_prepare_mlp`, `mlp.op_gate`,
`mlp.op_select_experts`, `mlp.op_dispatch_a/b`, `mlp.op_experts`,
`mlp.op_combine_a/b`, `mlp.op_shared_experts`, `mlp.op_output`,
`op_comm_postprocess_layer`) separated by `YieldOperation()` markers.
`operations.py::execute_overlapped_operations` runs two `_StageExecutor`s, `a`
and `b`, with `b` lagging by `delta_stages`.

The two published DeepSeek strategies (`operations_strategy.py:100-160`):

- **prefill**, `tbo_delta_stages=0`, `deep_gemm_num_sms = total_SMs -
  DeepEPConfig.num_sms`; two yields — after `dispatch_a` and after `combine_a`.
- **decode**, `tbo_delta_stages=2`; five yields, splitting attention
  prepare/core and interleaving shared-expert compute with dispatch.

Each TBO child sub-batch dispatches against its **own** attention backend
(`_resolve_tbo_child_contexts()` swaps `ctx.attn_backend` to
`TboAttnBackend.children[i]` per stage), so the parent's primary metadata is
never used by a child.

**Numbers** `[reported]`, large-scale EP blog, 96x H100, DeepSeek V3/R1:
prefill throughput **+27–35%** (and 8,192 → 16,384 tokens/device without OOM);
decode **+25.5%** at batch 256 (22,310 tok/s/node); decode **+35%** with
simulated MTP at 128 seqs (17,552 tok/s/node).

**Constraints and negative results** `[verified]`:

- Split index at decode/verify is `(num_tokens // token_num_per_seq) // 2`
  (`two_batch_overlap.py:82-98`). **At batch size 1 that is 0.** TBO is
  structurally inert at C1.
- `tbo_token_distribution_threshold` default **0.48** (`server_args.py:2963-2967`);
  below it, a single sequence is cut in half ("two-chunk overlap") instead of
  splitting at a sequence boundary. There is a documented degenerate case where
  the two-chunk cut leaves child A with more sequences than query tokens,
  violating "the DSV4 compress planner invariant `batch_size <= num_q_tokens`
  and crash[ing] the kernel" — so the code falls back to a seq-boundary split
  (`two_batch_overlap.py:101-130`).
- TBO forces `needs_cpu_seq_lens = True`, reinstating a D2H sync
  (`overlap_utils.py:34-38`).
- **DeepSeek-V4 decode TBO is not implemented**, with the reason in a comment:
  *"ATOM data: decode TBO regresses; needs cuda-graph capture work. Prefill-only
  for now."* (`operations_strategy.py:170-180`).
- On GB200 NVL72, SGLang's own conclusion was to **turn TBO off**: *"the
  pure-NVLink solution … significantly reduces communication latency. This
  allows two-batch overlap to be disabled, resulting in both kernel performance
  speedup and avoiding waste when overlapped communication is longer than
  computation"* `[verified]`,
  <https://lmsys.org/blog/2025-06-16-gb200-part-1/>.
- Community bug reports of TBO regressing on a single node exist
  ([#12061](https://github.com/sgl-project/sglang/issues/12061)) `[verified]`
  title only.

### Single-Batch Overlap (SBO)

**Mechanism** `[verified]`. Prose: <https://lmsys.org/blog/2025-09-25-gb200-part-2/>
— *"The two-batch overlap used in the previous hardware may not be the most
suitable given the significantly increased communication bandwidth, thus we
adopt a fine-grained overlapping approach. For simplicity, we overlap the
combine communication with both the down GEMM and the shared experts. When
implementing signaling in GEMM, we use atomic instructions with release
semantics after the TMA store wait … and `cp.async.bulk.wait_group` PTX …
removing the `.read` suffix."* The GB300 blog restates it as *"combining
communication to run concurrently with down-GEMM computation in a
producer–consumer pattern, while overlapping shared-expert computation on an
additional CUDA stream."*

Source (`batch_overlap/single_batch_overlap.py`, `models/deepseek_v2.py`)
`[verified]`:

- `SboFlags.enable_combine_down_gemm_two_stream_overlap()` requires the MoE
  runner backend to be `flashinfer_cutedsl`, or `deep_gemm` on non-Blackwell.
- `SboFlags.enable_dispatch_shared_one_stream_overlap()` is **disabled on
  Blackwell** (`is_sbo_enabled() and not is_blackwell()`), so on B200 the shared
  experts overlap the **combine** on a second stream, not the dispatch.
- SM count split: `communicate_num_sms = 32 if is_blackwell() else 3`
  (`SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS`, default 32), `compute_num_sms =
  total - communicate_num_sms`. A **10x larger comm SM budget on Blackwell than
  Hopper** — a concrete tuned constant.
- Producer/consumer signalling: `combine_signal = zeros(num_local_experts,
  dtype=uint32)` on Blackwell (per-expert), vs a per-block-M int32 signal array
  on Hopper.
- **SBO raises on SM90** with the current sgl-deep-gemm wheel
  (`layers/moe/utils.py:328-333`) — a stated negative.
- **Critically for us**: all of this lives inside
  `DeepseekV2MoE.forward_deepep()` (`deepseek_v2.py:1239-1420`). With
  `--moe-a2a-backend none` — the TP8 low-latency configuration — SBO code never
  executes. `[inferred]` `--enable-single-batch-overlap` on our current launch is
  a no-op.

`SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` (default `False`) moves
shared-expert overlap out of SBO entirely `[verified]`, `environ.py:998`.

### EPLB

`[verified]` EP doc + large-scale EP blog. DeepSeek's EPLB, driven by expert
activation statistics; redundant experts are added (e.g. 32 on top of 256 → a
288-expert pool) and placed to minimise per-GPU variance. `--enable-eplb`.
`[reported]` **1.49x prefill / 2.54x decode** on 96x H100. Stated limitation:
tuned on in-distribution data; real-world drift untested.

### Newer MoE parallelism worth knowing

**DWDP (Distributed Weight Data Parallelism)**, v0.5.17
([#29778](https://github.com/sgl-project/sglang/pull/29778)) `[verified]` in
release notes: prefetch peer expert weights over NVLink P2P and compute **all**
experts locally, removing the EP all-to-all dispatch entirely. `[reported]`
On **4x B200, gpt-oss-120b, prefill only**, DWDP4 reaches **1.92x over DEP4** at
MNT 32K / ISL 32K and **506K vs 329K tok/s (1.54x)** at saturation (conc 128,
ISL 8K). Enabled with `--dwdp-size`; authors mark it early-development.
`[inferred]` The idea — trade NVLink weight movement for eliminated token
all-to-all — is very interesting on an NV18 all-to-all box like ours, but it is
a *prefill* technique and our bottleneck is decode.

---

## Collectives on Blackwell: FlashInfer all-reduce fusion

This section matters most to us: collectives are **19.6%** of our C1 kernel time
and 47% of that is rank-arrival skew.

**Mechanism** `[verified]`, `layers/flashinfer_comm_fusion.py`,
`layers/communicator.py`, `arg_groups/overrides.py`.

- Backends: `mnnvl` and `trtllm`. `_resolve_backend("auto")` returns **`mnnvl`
  on any SM100** (single- *or* multi-node) and `trtllm` on SM90 single-node.
- **Auto-enable rule** (`overrides.py:1983-2004`) — fusion is switched on only
  when *all* of: architecture is in `_FLASHINFER_ALLREDUCE_FUSION_ARCHS`
  (**`GlmMoeDsaForCausalLM` is in the list**, line 1965), SM90 or SM100,
  `tp_size > 1`, **`not enable_dp_attention`**, `nnodes == 1 or SM100`, and
  **`moe_a2a_backend == "none"`**.
- **Per-call gate**: `apply_flashinfer_allreduce_fusion(batch_size)` requires
  `not is_dp_attention_enabled()` and `0 < batch_size <=
  FUSE_ALLREDUCE_MAX_BATCH_SIZE`, where the constant is **2048**
  (`communicator.py:162-179`).
- **What gets fused**: `should_fuse_mlp_allreduce_with_next_layer()`
  (`communicator.py:809-866`) makes the *next* layer's residual-add + LayerNorm
  absorb the post-MoE all-reduce, marking the tensor
  `hidden_states._sglang_needs_allreduce_fusion = True` and skipping
  `postprocess_layer` (`deepseek_v2.py:2525-2537`).
- **Documented refusals to fuse**, each with a stated reason:
  - MoE-CP all-gather active (the fusion path skips the moe_cp scatter → shape
    mismatch);
  - **hybrid EP+TP** (`moe_ep_size > 1 and moe_tp_size > 1`) — the post-experts
    reduction spans two disjoint groups and the fused kernel reduces over one,
    so it would "silently return under-reduced activations";
  - **DP attention + an EAGLE-family speculative algorithm**;
  - the last layer (nothing to fuse into);
  - scattered MLP mode.
- Historical negative: a comment cites
  [#17237](https://github.com/sgl-project/sglang/issues/17237) — "flashinfer
  0.6.1 caused performance regression on sm100 for allreduce fusion".

`[inferred]` Our profile shows `tllm_mnnvl_allreduce::oneshotAllreduceFusionKernel`
at **8.2%** and `trtllm_mnnvl_allreduce::twoshotAllreduceKernel` at **4.3%** of
device-0 kernel time, so fusion *is* engaged and both the one-shot and two-shot
kernels are in play. Two follow-ups: (a) find and tune the one-shot/two-shot
size boundary inside FlashInfer for our token counts; (b) note that turning on
`--enable-dp-attention` for a throughput arm **silently disables fusion
entirely** — that is a large hidden term in any TP-vs-DP comparison we run.

**PR #32461 — small-AR routing** `[verified]`
<https://github.com/sgl-project/sglang/pull/32461> (closed, unmerged, CI red).
Diagnosis: in the NextN draft model the post-MoE and post-embedding TP
all-reduces fall back to NCCL because the draft layer *is* the last layer (no
next-layer norm to fuse into) and `VocabParallelEmbedding` reduces internally.
NCCL `RING_LL` serves those tiny bf16 messages (1–6 tokens x hidden) at **23–33
µs/call**; the FlashInfer one-shot kernel does equivalent shapes at **~8 µs**.
Fix: `SGLANG_FLASHINFER_SMALL_AR_MAX_BYTES` (default 0 = off) diverts 2-D bf16
TP all-reduces at or below the threshold through
`AllReduceFusionPattern.kAllReduce` on the existing workspace, returning `None`
(and falling back) on any mismatch.

`[reported]` GLM-5.2-FP8, **bs=1, EAGLE 5-1-6, TP8 over 2x4 GB300 (MNNVL)**,
`SGLANG_FLASHINFER_SMALL_AR_MAX_BYTES=196608`:

| Metric | Before | After |
|---|---:|---:|
| Decode throughput (bs=1) | 374.83 tok/s | **381.85 tok/s (+1.87%)** |
| Spec accept length | 3.8636 | 3.8623 |
| TTFT P50 | 0.2187 s | 0.2107 s |
| NCCL all-reduces / spec step | 12 (0.450 ms) | **0** |
| MNNVL one-shot kernels / spec step | 160 (1.284 ms) | 172 (1.348 ms) |
| GSM8K 200q | 0.980 | 0.985 |

Honest caveat stated by the author: one-shot changes bf16 summation order, so
greedy output is not byte-identical (111/120 identical, divergences
deterministic). `[inferred]` This is the single most directly applicable
published collectives result to our C1 profile — same model family, same spec
config, same TP size, batch 1.

Other comm items in v0.5.17 `[verified]` release notes: multi-node custom-AR v2
on a single NVLink clique ([#32339](https://github.com/sgl-project/sglang/pull/32339)),
graph capture + MSCCL++ for attention-TP groups
([#31629](https://github.com/sgl-project/sglang/pull/31629)), and disabling an
extra NCCL CUDA-event sync under symmetric memory
([#27089](https://github.com/sgl-project/sglang/pull/27089)). There is a whole
roadmap issue for **NCCL 2.30 features**
([#32774](https://github.com/sgl-project/sglang/issues/32774)).

---

## PD disaggregation and the router

`[verified]` large-scale EP blog + `docs/docs/advanced_features/pd_disaggregation.mdx`
+ v0.5.17 notes.

**Why PD**: eliminate prefill interruption of decode, resolve DP-attention
imbalance, and let each role pick its DeepEP dispatch mode. Transfer is
non-blocking and RDMA-based; backends are Mooncake, NIXL, and (for DSv4)
page-indexed transport that "stays oblivious to DSv4's non-uniform on-device
layout".

`[reported]` The flagship result: 12 nodes / 96 H100 giving **52.3k input and
22.3k output tokens/s per node** at 2000-token sequences, costed at
**$0.20/1M output tokens** — with the stated limitation of **TTFT 2–5 s and ITL
~100 ms**. That is the opposite corner of the design space from ours.

Router: `sglang_router.launch_router --pd-disaggregation`, now the SGLang Model
Gateway. GLM-5.2 cookbook notes Mooncake **auto-detects the InfiniBand HCA**,
and that without exposing IB into a container Mooncake **silently falls back to
TCP** `[verified]`.

`[inferred]` PD is not a C1 technique. Its relevance to us is the C64 arm and the
AA 10-parallel scenario, and only if we ever run more than one node.

---

## Blackwell: NVFP4, FP8, and the B200-specific work

### Quantization state on SM100

`[verified]`, v0.5.17 release notes + `server_args.py` + GLM-5.2 cookbook:

- **`--bf16-gemm-backend {auto, cutedsl, torch}`**, default `auto`, which
  *"selects `cutedsl` on SM10x GPUs, except deterministic inference selects
  `torch`; otherwise uses cuBLAS via `torch.nn.functional.linear`"*
  (`server_args.py:1764-1772`). **CuteDSL BF16 GEMM on SM100 became on-by-default
  in v0.5.17** ([#30567](https://github.com/sgl-project/sglang/pull/30567)) and
  is listed as a breaking change.
- GLM-5.2's NVFP4 recipe **keeps attention projections and the shared-expert MLP
  in BF16** and quantises only the routed experts — which is exactly why a BF16
  GEMM backend matters on an "NVFP4" model.
- `--quantization modelopt_fp4`; NVFP4_AWQ checkpoints supported
  ([#31825](https://github.com/sgl-project/sglang/pull/31825)); FP8 KV
  auto-selected for DSA on Blackwell (cookbook: *"SGLang also auto-selects the
  KV-cache dtype for DSA models — `fp8_e4m3` on Blackwell (B200/GB300/B300,
  which then routes DSA through the TensorRT-LLM backend) and `bf16` on
  Hopper"*).
- **Known issue, v0.5.17**: *"The FlashInfer rmsnorm + quant fusion for
  SM90/SM100/SM120 was landed and reverted this cycle
  ([#32994](https://github.com/sgl-project/sglang/pull/32994),
  [#33455](https://github.com/sgl-project/sglang/pull/33455)). It is not in this
  release."* `[verified]` — a fusion we might otherwise have expected to have.
- Also known-broken on Blackwell-family: a DeepGEMM regression on B300 (sm_103),
  `CUDA_ERROR_ILLEGAL_ADDRESS` in `fp8_fp4_gemm_nt` TMA descriptor init
  ([#25551](https://github.com/sgl-project/sglang/issues/25551)), and *"Small
  probability random crashes of B200 and B300 on sglang"*
  ([#27520](https://github.com/sgl-project/sglang/issues/27520)) `[verified]`
  titles only.

### The GLM-5.2 kernel work (this is our model)

`[verified]` <https://lmsys.org/blog/2026-07-13-glm52-optimization/>, "Serving
GLM5.2 NVFP4 Agentic Workload with SGLang: Reaching 500 TPS in 2 Weeks", 14 July
2026. Four techniques, with PR numbers given in the appendix:

1. **TopK-V2** (PRs #26788, #30274). Treats top-k as *selection*, not sorting.
   Short/medium rows go register-resident or single-CTA streaming; long rows use
   **a cluster of eight CTAs**, each building a local **10-bit radix histogram**
   over FP16-rounded, order-preserving unsigned keys (1,024 bins, atomic
   increments); a cluster-wide reduction finds the bin containing the 2048th
   largest score; values above it are emitted directly and boundary candidates
   go through **exact FP32 radix tie-break**. A **planning kernel** picks the
   cluster cutoff from the batch's sequence-length distribution and builds a work
   list for a persistent cluster pool — the plan is built once per forward and
   **reused across DSA layers**. Selection and the page-table transform are fused
   into one kernel. Runtime k supported to 2048.
   `[reported]` **40.7 → 17.5 µs at 80K ISL (2.33x)**; **372.1 → 36.6 µs at 1M
   ISL (10.17x)**. Measured on target-model verification, batch 1, 6 draft
   tokens. Enabled with `SGLANG_OPT_USE_TOPK_V2=1` (source default: `True`,
   `environ.py:1042`).
2. **Indexer prologue fusion** (PR #27705). Fuses `wk` and `weights_proj` into a
   single BF16 `wk_weights_proj`; then fuses the elementwise tails — key path =
   LayerNorm + RoPE + FP8 quant + paged indexer-KV store; query path = RoPE +
   FP8 quant + head-gate scaling. **Drops the Hadamard transform** on the
   argument that an orthonormal transform applied to both Q and K preserves
   inner products *before* quantisation, so its only real effect was on the
   quantised representation. `[reported]` **12 kernels → 4**; **~8% decode
   throughput at bs=1**, ~5% at bs=128. Our fork has `wk_weights_proj`
   (`layers/attention/dsa/dsa_indexer.py:425-430`) `[verified]`.
3. **CuTe DSL BF16 GEMM** (PR #30177), from FlashInfer's TGV GEMM. Warp-dedicated
   task model (load-only / compute-only / store-only warps) plus very aggressive
   multi-tile pipelining using nearly all shared memory — the stated edge over
   cuBLAS is that at decode-sized M these GEMMs are memory-latency-bound, so
   deeper prefetch wins. Per-shape tile tuning plus a runtime heuristic that
   falls back to cuBLAS. `[reported]` at TP4: fused QKV `(M, 2624, 6144)`
   averages **1.08x** (peak 1.13x), `o_proj` `(M, 6144, 4096)` averages **1.05x**
   (peak 1.08x), swept M=1..32; **~4% end-to-end decode at bs=1**.
4. **IndexShare MTP** (PRs #27114, #29654, #29787, #30839, #30992). Reuses the
   DSA indexer's top-k across draft steps — computed at draft step 0, held and
   passed to later steps. The seed comes from the *previous* `run_batch`
   iteration's draft-extend and had to be **threaded through the overlap
   scheduler's relay buffer**. `[reported]` **draft-step cost down up to ~1.9x
   at long context, no output-quality change.**

Headline `[reported]`: **>500 TPS on 8xB300 at bs=1**; 18–34% single-user
interactivity improvement since day-0; 6–11% peak throughput at batch 8;
GLM-5.2 ~1.4x (4xGB300) and ~1.3x (8xB300) over GLM-5.1 in single-user
interactivity per-GPU throughput. **Config asymmetries to hold in mind**: B300 /
GB300, *not* B200; the workload is an **OpenHands multi-turn agentic replay with
~80K input, 220 output tokens/turn, 13 turns, and a ~92% aggregate prefix-cache
hit rate**; real EAGLE acceptance (not simulated) for the Pareto curves but
`SGLANG_SIMULATE_ACC_LEN` is used for the ISL ablation. That workload shares
almost nothing with AA's 10k-in / ≥1500-out single-shot shape.

### The B300/GB300 marketing tier

`[reported]` <https://lmsys.org/blog/2026-02-20-gb300-inferencex/>: "up to 25x"
DeepSeek-R1 on GB300 NVL72 vs H200 on SemiAnalysis InferenceXv2, and "up to 8x"
more tokens/GPU on GB200 in under four months. The blog itself discloses that
the H200 baseline is taken **at a 50 TPS/user interactivity constraint** and
that unconstrained H200 reaches similar throughput. Treat the 25x as a
constrained-operating-point ratio, not a speedup. The technically useful content
is one sentence, already quoted above: **SBO replaced TBO on NVL72.**

---

## Answers to the three specific questions

### Q1. Why does SGLang refuse `--enable-two-batch-overlap` with DSA index-topk sharing?

**The gate** `[verified]`, upstream `python/sglang/srt/server_args.py:5316-5327`
(identical in our fork at `5051-5062`):

```python
index_topk_freq = getattr(hf_config, "index_topk_freq", 1) or 1
index_topk_pattern = getattr(hf_config, "index_topk_pattern", None)
if self.enable_two_batch_overlap and (
    index_topk_freq > 1
    or (index_topk_pattern is not None and "S" in index_topk_pattern)
):
    raise ValueError(
        "--enable-two-batch-overlap is not supported with DSA "
        "index-topk sharing (index_topk_freq > 1 or an "
        "index_topk_pattern containing shared layers): the TBO op "
        "path does not propagate topk indices across layers, so "
        "shared layers would run sparse attention without indices."
    )
```

**The actual defect the gate protects against** `[verified]`,
`python/sglang/srt/models/deepseek_v2.py:2033-2042`:

```python
def op_core(self, state):
    result = self.forward_core(state.pop("attn_intermediate_state"))
    # forward_core may return (hidden_states, topk_indices) for DSA models
    # with index cache enabled. In the TBO path, topk_indices is not
    # propagated between layers, so we discard it here.
    if isinstance(result, tuple):
        state.hidden_states_after_attn = result[0]
    else:
        state.hidden_states_after_attn = result
```

So the TBO op path **literally throws the indices away**. The non-TBO path does
the opposite: `DeepseekV2DecoderLayer.forward()` takes `prev_topk_indices`,
passes it into `self_attn(...)`, unpacks `(hidden_states, topk_indices)` and
**returns `topk_indices` as a third value** (`deepseek_v2.py:2442-2537`); the
model loop then feeds it through an `IndexTopKShareState`
(`layers/attention/index_topk_share.py`, `deepseek_v2.py:2765-2907`) which also
handles the MTP seed and the PP proxy-tensor hand-off across pipeline stages.

Layer-level skip is decided at construction (`deepseek_v2.py:1835-1851`):
`self.skip_topk = dsa_layer_skips_topk(config, layer_id)` and `self.next_skip_topk
= dsa_layer_skips_topk(config, layer_id + 1)`. `dsa_layer_skips_topk`
(`configs/model_config.py:191-212`) returns `pattern[layer_id] == "S"` if an
`index_topk_pattern` exists, else `max(layer_id - 1, 0) % freq != 0` (with an
`index_skip_topk_offset` variant). **With `index_topk_freq = 4`, three of every
four layers are `skip_topk` layers that need the previous layer's indices.**
Running those under TBO with `topk_indices` discarded means sparse attention
with garbage or missing indices — hence the hard `ValueError` rather than a
silent wrong answer.

**Provenance** `[verified]`: the gate is upstream, not fork-local (present at
upstream HEAD `f019f0b0`). I could **not** find an issue or PR that introduced or
discusses it — GitHub's code-search API returned 0 for this repo under my token,
and issue search on "index-topk sharing" / "index_topk_freq" turned up nothing.
So: **`[unverified]` — no upstream issue or design doc for this specific gate.**
The nearest related open work is
[#32529](https://github.com/sgl-project/sglang/pull/32529) "[WIP][Spec] Enable
Two-Batch Overlap in Draft Models" and
[#7892](https://github.com/sgl-project/sglang/issues/7892), neither of which is
about index sharing.

**What a fix would take** `[inferred]`, from reading the code:

1. Add a `topk_indices` slot to the per-ubatch `_StateDict` in
   `batch_overlap/operations.py`. Each `_StageExecutor` (`a`, `b`) already owns a
   private state dict, so per-microbatch correctness is free — the two ubatches
   are disjoint token sets and each has its own layer chain.
2. In `op_core`, stash `result[1]` into `state.topk_indices` instead of dropping
   it; in the next layer's `op_prepare` / `forward_prepare`, read it as
   `prev_topk_indices`. The threading pattern already exists verbatim in
   `forward()`.
3. Seed and publish at the chain ends: create an `IndexTopKShareState` per ubatch
   at model-loop entry and call `.publish()` at exit, mirroring
   `deepseek_v2.py:2765-2907`. The PP proxy-tensor path shows the same
   cross-boundary hand-off already works.
4. Verify `TboAttnBackend.children[i]` metadata carries the per-child indexer
   buffers; `layers/attention/dsa_backend.py` sizes graph buffers as
   `max_bs * dsa_index_topk` (line ~438), which under TBO must be per-child.

Effort: `[inferred]` a day or two of plumbing, plus a real correctness harness.

**But do not do it.** Three independent reasons: (a) TBO's decode split index is
`(num_tokens // token_num_per_seq) // 2`, which is **0 at batch size 1**, so TBO
contributes exactly nothing to our C1 objective; (b) SGLang's own conclusion on
NVLink-domain hardware was to disable TBO and use SBO instead; (c) TBO forces
`needs_cpu_seq_lens = True`, undoing a D2H-sync removal the GLM-5.2 blog
specifically counted as part of its 11%. If we want overlap at C64, the
published answer for Blackwell is SBO — which needs `--moe-a2a-backend deepep`,
which is what the published *balanced* and *high-throughput* GLM-5.2 cells use.

### Q2. What causes illegal memory accesses in `eagle_worker_v2` during draft CUDA-graph capture at larger draft depths?

There is **no single published root cause**. There are five documented, distinct
mechanisms in this family, three merged, two open. Ranked by how well each fits
our symptom (deterministic IMA at capture, appearing at `num_steps >= 4`,
disappearing at 3):

**(a) JIT-inside-capture race — PR [#33795](https://github.com/sgl-project/sglang/pull/33795), OPEN, leading candidate** `[verified]`

> `FullCudaGraphBackend.capture_one()` synchronizes the device **before** each of
> the two warmup iterations, but there is **no synchronization between the last
> warmup and `torch.cuda.CUDAGraph()` creation**. Warmup forward passes trigger
> asynchronous JIT compilation … if that compilation is still in flight when
> capture begins, the JIT issues CUDA driver calls (`cuModuleLoadData`, etc.)
> **inside the capture context**, which is illegal — the first kernel to run
> after that is the one that reports the IMA.

Signature match is strong: the reported kernel **varies across runs and ranks**
(the PR lists DeepGEMM, c128_v2, Triton routing, `fused_norm_rope_v2` on
different TP ranks), which is exactly what issue #31093 describes for GLM-5.2
NVFP4 + EAGLE ("on v0.5.15 it fired on TP rank 7 at ~91% of the small-batch-size
decode captures; on dev same signature on TP rank 2"). It also explains
depth-dependence: **a new draft depth produces new shapes, which produce fresh
JIT compilations during warmup**, opening the race window; the PR observes
exactly this ("skipping the failing shape just moves the failure to the next
non-uniform tier").

Fix, verbatim from the PR:

```python
for _ in range(2):
    self._device_module.synchronize()
    self._tp_group.barrier()
    forward_fn()
    if post_warmup_hook is not None:
        post_warmup_hook()

# Ensure async JIT compilation and lazy initialization triggered during
# warmup completes before entering the capture context.
self._device_module.synchronize()
self._tp_group.barrier()

graph = torch.cuda.CUDAGraph()
```

`[verified]` Neither upstream HEAD
(`model_executor/runner_backend/full_cuda_graph_backend.py:103-114`) nor our fork
(same file, ~lines 96-114) has this sync. The companion PR for the breakable
backend is [#34286](https://github.com/sgl-project/sglang/pull/34286), also open.
`[reported]` Result: H200x4 DSpark, bs=60 crashed 3/3 runs before, passes after;
full 70-shape capture completes; cost is one sync+barrier per captured shape.

Corroborating evidence for the JIT-load hypothesis:
[#33642](https://github.com/sgl-project/sglang/issues/33642) `[verified]` — all 8
schedulers hang inside `cuModuleLoadData` while Triton lazily loads
`fused_dsa_target_verify_metadata` on the first EAGLE verify, i.e. DSA
speculative metadata kernels do JIT-load late, at exactly the wrong moments.

**(b) `custom_mask` double-count → `torch.cat` during capture → dangling pointer — issue [#32666](https://github.com/sgl-project/sglang/issues/32666) / PR [#32691](https://github.com/sgl-project/sglang/pull/32691), OPEN** `[verified]`

Present unchanged at upstream HEAD and in our fork,
`speculative/eagle_info.py:119-136`:

```python
mask_numel = (
    paged_kernel_lens_sum * self.draft_token_num
    + (self.draft_token_num**2) * batch_size
)
if self.custom_mask.numel() < mask_numel:
    # FIXME(attn): temporary fix for custom mask padding with cuda graph
    self.custom_mask = torch.cat([...])
```

The PR's claim: `paged_kernel_lens_sum` already includes the
`batch_size * draft_token_num` verify query tokens, so the second term
double-counts; the inflated `mask_numel` eventually exceeds the static buffer
(allocated as `(max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_req`
in `runner_utils/buffers.py:119-122`), the `torch.cat` allocates a **temporary
during capture**, the graph memorises its pointer, and the tensor is freed when
capture ends. The reporter's ladder is striking: depth 3 and depth 5 capture all
23 shapes cleanly, **depth 4 dies at `bs=5` with `num_tokens_per_req=5`** — the
batch size equalling the draft-token count. Fix is 4 files, ~12 lines: drop the
extra term and double the static buffer.

`[inferred]` This only bites backends that actually consume `custom_mask`.
v0.5.17 added three PRs to skip/compact the verify mask when nothing reads it
([#32886](https://github.com/sgl-project/sglang/pull/32886),
[#32920](https://github.com/sgl-project/sglang/pull/32920),
[#33127](https://github.com/sgl-project/sglang/pull/33127)), which suggests the
DSA/TRTLLM path may not read it at all — worth checking before spending time
here.

**(c) Negative `seq_lens` on DP-padded draft-extend rows → uint32 wrap → top-k v2 IMA — PR [#30378](https://github.com/sgl-project/sglang/pull/30378), MERGED 7 Jul 2026** `[verified]`

Exact GLM-5.2 MTP failure. Under DP attention, idle-companion / DP-padded rows of
a **draft-extend-v2** graph replay carry the graph's seq_len fill value **1**,
smaller than `qo_len = speculative_num_draft_tokens = 6`. Visible KV lengths are
expanded as `kv - qo + 1 + i`, giving `[-4, -3, -2, -1, 0, 1]`. The top-k v2
kernel reads lengths as `uint32_t`, so `-4` becomes ~4.29e9, the row takes the
cluster path over the max-context score buffer, output slots under-fill, and the
page-table transform gathers `page_table[garbage >> page_bits]` →
`CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS` in `topk_small_batch_kernel`, grid
`(6,8,1)`. Fix: clamp expanded seq_lens to `>= 0` in
`fused_dsa_draft_extend_metadata` and `seqlens_expand_kernel`.

Key diagnostic in the PR: *"The verify-side expansion (`kv + off + 1`) is additive
and stays positive → only draft-extend companions crash. TP-only MTP has no DP
idle companions (only occasional bucket-padding rows) → passes / flaky."*
`[inferred]` **We run TP8 without DP attention, so we are in the "flaky, only
bucket-padding rows" regime of exactly this bug** — and bucket-padding rows are
what a capture pass is made of. Even with the merged clamp, the open PR
[#34455](https://github.com/sgl-project/sglang/pull/34455) "Fix DSA metadata row
count for DP-padded idle speculative batches" says the family is not closed.

**(d) Top-k v2 tie overflow leaving unwritten output slots — PR [#30512](https://github.com/sgl-project/sglang/pull/30512), MERGED 8 Jul 2026** `[verified]`

Also exactly our model and config: *"Serving GLM-5.2 with MTP (EAGLE) and the
fused DSA top-k v2 path (`SGLANG_OPT_USE_TOPK_V2=1`, default) hits a CUDA illegal
memory access within ~1 min of decode once sequences exceed `index_topk` (2048),
under CUDA graphs with long `--context-length` (TP-only; `--disable-cuda-graph`
does not reproduce)."* Root cause: `collect` kept at most `kMaxNumTie = 1024`
ties, so when `above_count < topk - 1024` the tie handler filled fewer than
`topk` slots while the transform pass page-translated **all** `topk` slots,
gathering uninitialised staging memory. Measured: 41 rows with `equal_count >
1024` (up to 1536) in a ~9-minute bench, 4 of which had 70–135 unwritten slots.
Accuracy after the fix: GSM8K 0.9545 (vs legacy top-k 0.9500), AIME25 0.9208 ±
0.005, on GLM-5.2-NVFP4 4xGB300 TP4 **EAGLE 5-1-6** with fp8 KV.

`[verified]` **Our fork already carries a stronger version of this fix**:
`python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh:201-212`
sets `kMaxNumTie = 2048` with `static_assert(kMaxNumTie >= kMaxTopK ...)` and a
comment saying a smaller cap "leaves slots that handle_tie can only pad". So (d)
is closed for us. But the PR hands us a **workaround for everything in the top-k
family: `SGLANG_OPT_USE_TOPK_V2=0`** falls back to the legacy kernel at ~0.5
points of GSM8K and a large kernel-latency cost — a diagnostic, not a
configuration.

**(e) The open issue that is closest to us verbatim — [#31093](https://github.com/sgl-project/sglang/issues/31093), OPEN** `[verified]`

GLM-5.2 NVFP4, `GlmMoeDsaForCausalLM`, `--quantization modelopt_fp4`, EAGLE
**6-1-7**, 8xB200 TP8, driver 580.159.04, `--cuda-graph-max-bs-decode 48`,
`--mem-fraction-static 0.85`. Crashes with an async IMA during decode CUDA-graph
capture on v0.5.15 and on main `b94ac87e`, but boots cleanly on the older
`dev-glm52-nvfp4` branch image (commit `430418e2`, 3 Jul 2026). Both cu12 and
cu13 builds reproduce, ruling out the toolchain. Two collaborators replied; one
**could not reproduce on 8xB200 for v0.5.15 or v0.5.15.post1**. Still open.

One extra detail from that issue worth acting on: the reporter runs with
**`logits_processor.py`'s multimem all-gather patched out**
(`enabled=self.do_tensor_parallel_all_gather and not self.use_attn_tp_group` →
`enabled=False`), because *"with it enabled even the 07-03 image hits a
draft-CUDA-graph IMA on this model."* `[verified]` The corresponding upstream
work is [#32551](https://github.com/sgl-project/sglang/pull/32551) "[Perf] Allow
multimem all-gather on cross-node NVLink cliques (MNNVL)". That one-line patch is
a cheap experiment for us.

**The datapoint that says 5-1-6 is achievable on our exact hardware** `[verified]`,
issue [#32459](https://github.com/sgl-project/sglang/issues/32459): GLM-5.2-NVFP4,
`lmsysorg/sglang:v0.5.16-cu129`, **8xB200 (p6-b200.48xlarge), TP8,
`--speculative-num-steps 5 --speculative-num-draft-tokens 6`**, boots and serves.
That reporter's differences from the crashing configs are:
`--context-length 1048576`, `--max-running-requests 128`,
`--mem-fraction-static 0.84`, `--kv-cache-dtype fp8_e4m3`,
`--chunked-prefill-size 8192`. And they add: *"EAGLE + `--max-running-requests
256` crashes at CUDA graph capture with `cudaErrorIllegalAddress` on this model
(fine at 128; fine at 256 without EAGLE)."*

**Recommended experiment ladder for our fork** `[inferred]`, cheapest first:

1. Apply the #33795 two-line sync to `full_cuda_graph_backend.capture_one()` (and
   #34286's to the breakable backend). Retry 4-1-5 and 5-1-6.
2. Shrink and densify the capture list — `--cuda-graph-max-bs-decode 16` with an
   explicit small `--cuda-graph-bs-decode` list. #32459 says capture-list size is
   causal on this exact hardware.
3. `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` to bisect which of the two draft
   graph families is at fault; our crash is at the **draft decode** runner
   construction (`eagle_worker_v2.py:366` in the fork = the
   `Device2DraftCudaGraphRunner[...](self)` line, which only runs when
   `speculative_num_steps > 1`), so if disabling draft-extend changes nothing
   that is informative.
4. `SGLANG_OPT_USE_TOPK_V2=0` — if the crash disappears, it is in the top-k
   family and PRs #30378/#30512/#34455 are the reading list.
5. Patch out the multimem all-gather in `logits_processor.py` per #31093.
6. Apply #32691's `mask_numel` fix; check first whether our backend reads
   `custom_mask` at all.
7. Run under `compute-sanitizer` with `CUDA_LAUNCH_BLOCKING=1` — issue #28569
   documents that this is what moves the reported site from
   `overlap_utils.resolve_seq_lens_cpu` (a sync point) to the real replay site.

And file the result upstream on #31093 — it is open, has a maintainer asking for
a reproducer, and we have the box.

### Q3. Recommended server flags for a large MoE on 8xB200

SGLang publishes these three cells verbatim, marked `verified: true`
`[verified]`, `docs/src/snippets/configs/zai-org/glm-5.2.jsx`.

**GLM-5.2 NVFP4, 8xB200, low-latency** (lines 700-717):

```
--model-path nvidia/GLM-5.2-NVFP4 --tp 8 --quantization modelopt_fp4
--speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6
--chunked-prefill-size 8192 --mem-fraction-static 0.85
```

**GLM-5.2 NVFP4, 8xB200, balanced** (lines 718-741) — note DP attention and the
*shorter* draft:

```
--tp 8 --quantization modelopt_fp4 --dp 8 --enable-dp-attention
--speculative-algorithm EAGLE --speculative-num-steps 2
--speculative-eagle-topk 1 --speculative-num-draft-tokens 3
--chunked-prefill-size 32768 --mem-fraction-static 0.92
--max-running-requests 256
```

**GLM-5.2 NVFP4, 8xB200, high-throughput** (lines 742-757) — no speculation:

```
--tp 8 --quantization modelopt_fp4 --dp 8 --enable-dp-attention
--chunked-prefill-size 32768 --mem-fraction-static 0.92
--max-running-requests 512
```

The **FP8** cells differ mainly by adding `--moe-a2a-backend deepep` alongside
`--dp 8 --enable-dp-attention` on balanced/high-throughput, and by
`--mem-fraction-static 0.8` / `0.85`.

The GLM-5.2 optimization blog's own reproduction command adds `[verified]`:

```
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1
... --tensor-parallel-size 8 --quantization modelopt_fp4
    --context-length 90000 --max-running-requests 16
    --max-prefill-tokens 8192 --chunked-prefill-size 8192
    --cuda-graph-max-bs-decode 16 --mem-fraction-static 0.87
    --kv-cache-dtype fp8_e4m3 --bf16-gemm-backend cutedsl
    --reasoning-parser glm45 --tool-call-parser glm47
    --speculative-algorithm EAGLE --speculative-num-steps 5
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
    --enable-cache-report
```

(Both env vars are already `True` by default at HEAD — `environ.py:1005, 1042`
— so the exports are belt-and-braces.)

**Published numbers for these cells** `[reported]`, from
`glm-5.2-benchmarks.jsx`. Workload: `random`, ISL 8192, OSL 1024,
`--random-range-ratio 1.0`, `--flush-cache` every run, `main @ 09ca4fc` (H200 FP8
cells on `v0.5.14 @ 49e384ce`). **`tokens_per_sec_per_gpu` is total (in+out)
per GPU, not output speed.**

| HW | Quant | Strategy | Spec | Conc | TTFT (ms) | TPOT (ms) | tok/s/GPU |
|---|---|---|---|---:|---:|---:|---:|
| 8xB200 | NVFP4 | low-latency | 5-1-6 | 1 | **295** | **1.85** | 527 |
| 8xB200 | NVFP4 | low-latency | 5-1-6 | 16 | 2491 | 5.43 | 2289 |
| 8xB200 | NVFP4 | balanced | 2-1-3 | 64 | 5837 | 12.70 | 3770 |
| 8xB200 | NVFP4 | balanced | 2-1-3 | 256 | 16736 | 30.00 | 5343 |
| 8xB200 | NVFP4 | high-thr | none | 1024 | 130174 | 67.12 | 5305 |
| 8xB200 | FP8 | low-latency | 5-1-6 | 1 | 757 | 3.22 | 288 |
| 8xB200 | FP8 | low-latency | 5-1-6 | 16 | 3188 | 9.12 | 1476 |
| 8xB200 | FP8 | balanced | 1-1-2 | 64 | 5742 | 17.65 | 3078 |
| 8xB200 | FP8 | high-thr | none | 1024 | 177620 | 47.99 | 4059 |
| 8xB300 | NVFP4 | low-latency | 5-1-6 | 1 | 196 | 1.86 | 459 |
| 8xH200 | FP8 | low-latency | 5-1-6 | 1 | 668 | 5.05 | 197 |

**The asymmetry that matters most**: the page's own note says *"Spec cells pin
the EAGLE acceptance length via the serve env `SGLANG_SIMULATE_ACC_LEN`
(low-latency 5-1-6 = 3.5; FP8 balanced 1-1-2 = 2; NVFP4 balanced 2-1-3 = 2)"*
`[verified]`. So **the 1.85 ms TPOT is at a *forced* acceptance length of 3.5 on
synthetic random tokens** — not real acceptance on real prompts, and not the
same measurement as our 2.74 ms on real data. Ours is the harder number. The
comparison that would settle it is our stack at 5-1-6 with
`SGLANG_SIMULATE_ACC_LEN=3.5` on ISL 8192 / OSL 1024 random, which is a run we
can make once Q2 is unblocked.

**Additional published tuning guidance** `[verified]`:

- `docs/docs/advanced_features/hyperparameter_tuning.mdx`: aim for
  `token usage > 0.9` and `#queue-req` in 100–2000; tune
  `--schedule-conservativeness` down to ~0.3 when the server is too conservative,
  up to ~1.3 when retracting; target **5–8 GB `available_gpu_mem`** in the
  pre-ready log line as the `--mem-fraction-static` calibration; increase
  `--cuda-graph-max-bs-decode` for large-TP models (512–768 is sometimes
  useful); `--schedule-policy lpm` when the workload has many shared prefixes.
- GLM-5.2 cookbook: chunked-prefill is regime-dependent — raising it to 32768 on
  the balanced recipe gave **+34–78% output throughput and −39–59% TTFT** on
  8xH200 and 8xB200 at 8K-in/1K-out, and is **neutral for high-throughput**
  (decode-bound). `--max-running-requests` "tracks KV capacity, not a tuning
  free-for-all": ~60–90 concurrent 8K+1K FP8 requests fit one 8-GPU node.
- Warning on the DSA top-k backend: *"All recipes here run the DSA indexer top-k
  on the default `--dsa-topk-backend sgl-kernel`. Other top-k backend choices
  have not been fully validated on GLM-5.2."*
- Prefill context parallelism for TTFT: `--attn-cp-size 8 --enable-prefill-cp
  --cp-strategy interleave`, with the stated trade-off that it "will introduce
  extra all-gather operation before indexer-topk and attention kernels, so it
  will increase latency for decode (in unified deployment) or short prefill".
  `[inferred]` For AA's 10k input and 189 ms TTFT, this is unlikely to pay.

---

## Roadmap: what is landing next

`[verified]` from open roadmap issues and v0.5.17 release notes.

- **Scheduler refactor** ([#11762](https://github.com/sgl-project/sglang/issues/11762),
  PoC @hnyls2002): make forward-mode more general, make the scheduler more
  stateless, fully support mixed chunked prefill, **hide CPU overhead in the
  scheduler for all cases**, and **move more preparation into the CUDA graph**.
  The last item is directly the mechanism behind the GLM-5.2 blog's 11%.
- **Rust migration** ([#22558](https://github.com/sgl-project/sglang/issues/22558)):
  scheduler, API server, prefix tree. v0.5.17 shipped the Rust ingress/tokenizer/
  OpenAI server + PD support; the Unified Radix Cache blog reports a **Rust tree
  core prototype at up to 42% lower TTFT on long prefixes**.
- **CUDA graph backends** ([#23004](https://github.com/sgl-project/sglang/issues/23004)):
  `(decode, prefill) x (full, breakable, torch-compile-based pcg)`; breakable
  prefill CUDA graph on by default. Already default for DP attention in v0.5.17.
- **Speculative decoding** ([#23005](https://github.com/sgl-project/sglang/issues/23005)):
  general abstraction for more algorithms and for spec graph preparation/init;
  **adaptive spec configuration per request and batch size**
  ([#23705](https://github.com/sgl-project/sglang/issues/23705),
  `speculative/adaptive_spec_params.py` exists in tree). Also a **parallel
  speculative decoding** roadmap ([#27462](https://github.com/sgl-project/sglang/issues/27462)).
- **DSpark**: parent roadmap [#30344](https://github.com/sgl-project/sglang/issues/30344),
  correctness/robustness roadmap [#34297](https://github.com/sgl-project/sglang/issues/34297)
  (updated 11 Aug 2026), and **GLM-5.2 + AMD/ROCm DSpark support**
  [#30734](https://github.com/sgl-project/sglang/issues/30734). #34297 is worth
  reading in full: it explicitly refuses to force every CUDA fault into one root
  cause and layers them as *distributed control-plane consistency → verifier
  state/write-window → kernel plan/index correctness → capture-init ordering →
  PD draft-state handoff → replay-time ownership/lifetime/stream contracts.*
  That taxonomy is a good template for our own crash triage.
- **DCP / Helix** ([#29736](https://github.com/sgl-project/sglang/issues/29736)):
  `--dcp-comm-backend {ag_rs, a2a, fi_a2a}` and `--dcp-replicate-q-proj` landed
  in v0.5.17; `fi_a2a` delegates the cross-rank exchange to the FlashInfer MNNVL
  kernel on GB200.
- **Quantization H2 2026** ([#31783](https://github.com/sgl-project/sglang/issues/31783))
  and **FP4 KV** ([#29913](https://github.com/sgl-project/sglang/issues/29913)).
  `[inferred]` FP4 KV is the one to watch for us — we already run fp8 KV.
- **NCCL 2.30 integration** ([#32774](https://github.com/sgl-project/sglang/issues/32774)).
- **GB200/GB300 NVL72 optimizations** ([#19650](https://github.com/sgl-project/sglang/issues/19650),
  PoC @Fridge003).
- **HiSparse for long-context sparse serving** ([#28874](https://github.com/sgl-project/sglang/issues/28874))
  and **LayerSplit** (GLM-5.2 cookbook: `--enable-dsa-cache-layer-split` shards
  KV over the CP attention group and prefetches, *"can reduce kv cache memory by
  up to 75%"* on prefill nodes under PD).
- Not yet published as a document: **there is no "Development Roadmap (2026 Q3)"
  issue** — the most recent general one is Q2
  ([#22949](https://github.com/sgl-project/sglang/issues/22949)); only an AMD Q3
  roadmap ([#35003](https://github.com/sgl-project/sglang/issues/35003), 16 Aug
  2026) exists. `[verified]` by issue search.

---

## Techniques ranked by transferability to our stack

"Fit" = how well the published measurement conditions match ours (8xB200, TP8,
GLM-5.2 NVFP4, C1 latency / C64 cost, AA's 10k-in ≥1500-out shape).

| Technique | Mechanism in one line | Published effect (config) | Fit | Effort | Verdict |
|---|---|---|---|---|---|
| CuteDSL BF16 GEMM | Warp-specialised, deeply pipelined TGV GEMM for the BF16 layers an NVFP4 checkpoint keeps in BF16 | ~4% e2e decode at bs=1; 1.08x QKV, 1.05x o_proj (GLM-5.2, TP4, GB300) | High — same model, same layers, bs=1 | Flag | **Take now** |
| TopK-V2 | Cluster-of-8 radix-select with 10-bit FP16-keyed histogram + FP32 boundary refine, fused with the page-table transform | 2.33x at 80K ISL; 10.17x at 1M (GLM-5.2, bs=1, 6 draft tokens) | Medium — our ISL is 10k, where the win shrinks | Default on; verify | **Verify, then measure at our ISL** |
| Indexer prologue fusion | Fuse `wk`+`weights_proj`; fuse LN+RoPE+FP8-quant+cache-store and RoPE+quant+head-gate; drop Hadamard | 12→4 kernels; +8% at bs=1 (GLM-5.2) | High | Already in fork (partial) | **Confirm full fusion is active** |
| IndexShare MTP | Reuse draft-step-0 indexer top-k across draft steps, seed relayed through the overlap buffer | draft-step cost −1.9x at long context | High — our model has `index_topk_freq=4` | Already in fork | **Confirm it is on; it blocks TBO** |
| Spec V2 / zero-overlap fixes | Graph-able DSA draft-extend, optional `seq_lens_cpu`, no H2D, fused eager metadata | +11% e2e TPS (GLM-5.2) | High | Base-version dependent | **Confirm all four fixes present** |
| Post-warmup sync before capture | `synchronize()` + TP `barrier()` between last warmup and `CUDAGraph()` | Turns 3/3 crashes into passes (DSpark, H200x4) | High — same symptom class | 2 lines, unmerged | **Take now (Q2 step 1)** |
| Small-AR via FlashInfer `kAllReduce` | Divert tiny bf16 TP all-reduces off NCCL RING_LL (23–33 µs) onto the one-shot kernel (~8 µs) | +1.87% at bs=1; 12→0 NCCL ARs/step (GLM-5.2 FP8, TP8, GB300) | High — same model, same spec config, bs=1 | Hand-port, red CI | **Take, guarded by an env flag** |
| FlashInfer AR fusion (existing) | Next layer's residual+LN absorbs the post-MoE all-reduce; mnnvl on SM100 | — (already active for us) | High | Already on | **Tune the one-shot/two-shot boundary** |
| Chunked prefill 32768 | Larger prefill chunks unblock the queueing-bound regime | +34–78% output tok/s, −39–59% TTFT at 8K/1K on 8xB200 | High for the C10/C64 arm | Flag | **Take for throughput arms** |
| TCP `SO_RCVBUF` + `TCP_NODELAY` | Avoid a ~200 ms RTO retransmit on the first large request body; kill Nagle on keep-alive | TTFT 192→157 ms at 16K in; probe 208 ms→0.4 ms | Medium — AA measures over the internet; our body is ~40 KB vs a 75–91 KB bisected threshold | Config | **Measure before assuming** |
| DFlash (block diffusion + KV injection) | One forward emits a whole block; target hidden states injected into every draft layer's KV | >4.3x baseline, 1.5x MTP at C1 (Qwen3.5-397B BF16, 8xB200, HumanEval) | Medium — needs a trained GLM-5.2 drafter that does not exist | Very high | **Watch** |
| DSpark | Confidence-scheduled variable verify length + ragged full-CUDA-graph verify | 383.7 tok/s at accept ~5, bs=1 (DSv4-Pro, TP8, B300) | Low at C1 (authors say trimming ties at bs=1); medium at C64 | Very high; not supported on GLM-5.2 | **Watch** |
| SBO | Combine comm producer/consumer-overlapped with down-GEMM; shared experts on a second stream; 32 comm SMs on Blackwell | Qualitative in the GB200/GB300 blogs | Zero at `--moe-a2a-backend none`; medium on the DeepEP balanced arm | Flag + DeepEP | **Only for the EP arms** |
| TBO | Split the batch, interleave attention with dispatch/combine | +27–35% prefill, +25.5% decode@bs256 (96xH100) | **Zero at C1** (split index is 0); SGLang disabled it on NVLink | — | **Do not pursue** |
| EPLB | Redundant-expert placement from activation statistics | 1.49x prefill / 2.54x decode (96xH100, EP32/EP72) | Low at our scale | Flag + stats | **Skip** |
| HiCache / Unified Radix / session cache | Tiered KV with page-first host layout, component-validated reuse boundaries | up to 6x throughput, 80% TTFT (community); L3 hit ~98% | Low for AA single-prompt and C1 | High | **Skip for now** |
| DWDP | Prefetch peer expert weights over NVLink; compute all experts locally, no a2a | 1.92x over DEP4 (4xB200, gpt-oss-120b, **prefill only**) | Low — prefill technique | High, early-dev | **Watch** |
| PD disaggregation | Split prefill and decode instances; per-role DeepEP mode | 52.3k in / 22.3k out per node, $0.20/1M out (96xH100), **TTFT 2–5 s** | Low — wrong latency corner, single node | High | **Skip** |

---

## Sources

All URLs below were fetched and read during this work unless marked otherwise.

**LMSYS / SGLang blog**

- <https://lmsys.org/blog/2026-07-13-glm52-optimization/> — "Serving GLM5.2 NVFP4
  Agentic Workload with SGLang: Reaching 500 TPS in 2 Weeks", 14 Jul 2026. The
  single highest-value source here.
- <https://lmsys.org/blog/2026-07-06-dspark-sglang> — DSpark integration.
- <https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/>
  — DFlash + Spec V2.
- <https://lmsys.org/blog/2026-04-25-deepseek-v4/> — ShadowRadix, HiSparse,
  Flash Compressor, Lightning TopK, hierarchical multi-stream overlap.
- <https://lmsys.org/blog/2026-08-11-unified-radix-cache/> — Unified Radix Cache.
- <https://lmsys.org/blog/2026-02-20-gb300-inferencex/> — GB300 / InferenceXv2;
  the SBO-replaces-TBO statement.
- <https://lmsys.org/blog/2025-09-25-gb200-part-2/> — FP8 attention + NVFP4 MoE
  on GB200; the original SBO description with PTX-level detail.
- <https://lmsys.org/blog/2025-06-16-gb200-part-1/> — Blackwell DeepGEMM/DeepEP/
  FMHA/CUTLASS MLA; "allows two-batch overlap to be disabled".
- <https://lmsys.org/blog/2025-09-10-sglang-hicache/> — HiCache design.
- <https://lmsys.org/blog/2025-05-05-large-scale-ep/> — PD + large-scale EP, TBO
  and EPLB ablations.
- <https://lmsys.org/blog/2024-12-04-sglang-v0-4/> — zero-overhead batch
  scheduler, cache-aware load balancer, DP attention.

Referenced but not read in this pass (listed for completeness; do not cite
their contents): `2026-08-07-hpc-ops-sglang`, `2026-07-27-kimi-k3-day0-support`,
`2026-06-26-waterfill-lplb`, `2026-01-15-chunked-pipeline`,
`2025-09-29-deepseek-V32`, `2025-07-20-k2-large-scale-ep`,
`2025-09-22-sglang-deterministic`.

**GitHub — sgl-project/sglang**

- Release v0.5.17 (8 Aug 2026), v0.5.16, v0.5.15, v0.5.14 release notes, read via
  `gh release view`.
- Issue [#31093](https://github.com/sgl-project/sglang/issues/31093) — GLM-5.2
  NVFP4 + EAGLE 6-1-7 capture IMA on 8xB200. OPEN.
- Issue [#32666](https://github.com/sgl-project/sglang/issues/32666) /
  PR [#32691](https://github.com/sgl-project/sglang/pull/32691) — draft-depth
  ladder + `custom_mask` double-count. OPEN.
- PR [#33795](https://github.com/sgl-project/sglang/pull/33795) — JIT-inside-
  capture race, FullCudaGraphBackend. OPEN.
- PR [#34286](https://github.com/sgl-project/sglang/pull/34286) — same for
  BreakableCudaGraphBackend. OPEN.
- PR [#30378](https://github.com/sgl-project/sglang/pull/30378) — negative
  seq_lens → uint32 wrap → DSA top-k v2 IMA. MERGED.
- PR [#30512](https://github.com/sgl-project/sglang/pull/30512) — top-k v2 tie
  overflow. MERGED.
- PR [#33872](https://github.com/sgl-project/sglang/pull/33872) — DSV4 silent KV
  corruption at draft tokens > 4 (`kMaxMTPDraftTokens = 4` hard-code). OPEN.
- PR [#32461](https://github.com/sgl-project/sglang/pull/32461) — small-AR via
  FlashInfer kAllReduce. CLOSED unmerged.
- PR [#33026](https://github.com/sgl-project/sglang/pull/33026) — Rust server TCP
  TTFT stalls. MERGED.
- Issue [#32459](https://github.com/sgl-project/sglang/issues/32459) — EAGLE vs
  radix prefix reuse on GLM-DSA NVFP4, 8xB200. OPEN.
- Issue [#28569](https://github.com/sgl-project/sglang/issues/28569) — EAGLE3
  draft graph replay IMA as batch shrinks. OPEN.
- Issue [#33642](https://github.com/sgl-project/sglang/issues/33642) —
  `cuModuleLoadData` hang on first EAGLE verify. OPEN.
- Issue [#19796](https://github.com/sgl-project/sglang/issues/19796) — EAGLE V2
  NaN on radix prefix hit, GLM-5 NVFP4. CLOSED.
- Roadmaps: [#22949](https://github.com/sgl-project/sglang/issues/22949) (Q2
  2026 general), [#34297](https://github.com/sgl-project/sglang/issues/34297)
  (DSpark CUDA graph), [#30344](https://github.com/sgl-project/sglang/issues/30344),
  [#29736](https://github.com/sgl-project/sglang/issues/29736) (DCP/Helix),
  [#32774](https://github.com/sgl-project/sglang/issues/32774) (NCCL 2.30),
  [#31783](https://github.com/sgl-project/sglang/issues/31783) (Quantization H2),
  [#29913](https://github.com/sgl-project/sglang/issues/29913) (FP4 KV),
  [#28874](https://github.com/sgl-project/sglang/issues/28874) (HiSparse),
  [#19650](https://github.com/sgl-project/sglang/issues/19650) (GB200/GB300).

**Docs (read from the repo tree, published at docs.sglang.io)**

- `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx`
- `docs/src/snippets/configs/zai-org/glm-5.2.jsx` and `glm-5.2-benchmarks.jsx`
- `docs/docs/advanced_features/expert_parallelism.mdx`
- `docs/docs/advanced_features/speculative_decoding.mdx`
- `docs/docs/advanced_features/hyperparameter_tuning.mdx`
- `docs/docs/advanced_features/sgl_model_gateway.mdx`

**Source read at upstream `f019f0b064ff13ef0700088f37dd44cd0d791b8d` (17 Aug 2026)**

`server_args.py` · `arg_groups/overrides.py` · `configs/model_config.py` ·
`models/deepseek_v2.py` · `batch_overlap/{operations,operations_strategy,two_batch_overlap,single_batch_overlap}.py` ·
`layers/{communicator,flashinfer_comm_fusion,logits_processor}.py` ·
`layers/attention/{dsa_backend,index_topk_share,tbo_backend}.py` ·
`layers/moe/utils.py` · `managers/{scheduler,overlap_utils}.py` ·
`mem_cache/{radix_cache,evict_policy}.py` ·
`model_executor/runner/base_cuda_graph_runner.py` ·
`model_executor/runner_backend/full_cuda_graph_backend.py` ·
`model_executor/runner_utils/buffers.py` ·
`speculative/{eagle_worker_v2,eagle_draft_cuda_graph_runner,eagle_info,spec_utils}.py` ·
`environ.py`

**Fork read at `/home/aman/code/NotSglang` (`a43988c2b`, 16 Aug 2026)**

`python/sglang/srt/server_args.py` · `python/sglang/srt/speculative/{eagle_worker_v2,eagle_info}.py` ·
`python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py` ·
`python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` ·
`python/sglang/srt/layers/attention/dsa/dsa_indexer.py`

**Could not source**

- Any issue, PR, RFC or design doc explaining or introducing the
  TBO-vs-`index_topk_freq` gate. GitHub code search returned zero results for
  this repository under my token; issue search on the error text and on
  `index_topk_freq` found nothing. The gate's rationale is documented only in its
  own error message and the `op_core` comment.
- A "Development Roadmap (2026 Q3)" issue. Does not appear to exist.
- Any independent (non-SGLang, non-vendor) reproduction of *any* number in this
  file.
