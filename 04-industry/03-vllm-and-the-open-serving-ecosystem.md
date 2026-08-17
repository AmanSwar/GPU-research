# vLLM and the wider open serving ecosystem: architecture, performance work, and what to borrow

*Competitive-intelligence dossier for the 8×B200 / GLM-5.2 / SGLang-fork stack. Compiled 2026-08-17.*

## What this is

A mine of **published** engineering from vLLM and the surrounding open serving ecosystem, filtered for
mechanisms we can reimplement on 8×B200 SXM with GLM-5.2 (MoE 256/8, DSA sparse MLA, NVFP4+FP8, TP8,
EAGLE 3-1-4). Everything below was read at a URL given in the Sources section. Labels:

- **[verified]** — I fetched the page and read the claim there. The URL is given.
- **[reported]** — the publisher's own claim; verified that they said it, *not* independently reproduced.
- **[inferred]** — my reasoning on top of verified facts.
- **[unverified]** — I could not source it. Stated as such, never filled in.

Two framing facts that reset the target, both **[verified]**:

1. **The "TileRT ≈ 500 tok/s" figure is a 1k-input number, not an Artificial-Analysis-shaped number.**
   SemiAnalysis measured TileRT FP8 (MXFP8) on GLM-5.1-744B, 8×B200, batch size 1, and got
   **494.2 tok/s/user at 1k/1k but 340 tok/s/user at 8k/1k**. AA uses ~10k input. So the honest
   TileRT-equivalent target at AA input length is ~340, not ~500 — and we are at 365 tok/s on real
   data. ([InferenceX, 2026-08-10](https://inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia))
   *This is the single most useful number in this document.*
2. **vLLM's own AA-topping result is well below both.** vLLM reports 230 output tok/s/user for
   DeepSeek-V3.2 on the AA leaderboard, achieved with 8×B300, TP4+TP4 P/D disaggregation and MTP=3.
   ([vLLM, 2026-05-11](https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis))

vLLM is therefore not the latency engine to beat. It *is* the engine with the most published,
reproducible, individually-attributed micro-optimizations — and several of them are directly portable
into an SGLang fork.

---

## Bottom line for our system

Ranked by (expected effect on our measured hotspots) × (confidence) ÷ (difficulty). Hotspot reminder at
C1: dense GEMM 37.1%, collectives 19.6% (47% of that is rank arrival skew), MoE expert GEMMs 19.4%,
attention 10.9%, DSA indexer 5.8%.

| # | Steal | Mechanism | Expected effect | Difficulty | Evidence |
|---|---|---|---|---|---|
| 1 | **Share the DSA indexer top-k across MTP/EAGLE draft iterations** | vLLM's GLM-5.2 path reuses the top-K sparse indices produced by the DSA indexer inside the MTP iteration, via a flag they call `index_share_for_mtp_iteration`. With `index_topk_freq=4` and a 3-1-4 EAGLE tree we currently pay indexer cost per draft step. | Removes most of the *draft-step* multiplier on the 5.8% indexer cost. At a 4-token tree that is plausibly 3–4% of C1 step time. | **Low–Med** | [verified] [vLLM GLM-5.2 on B300](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd) |
| 2 | **Collapse the per-layer pre-attention op chain into ≤2 fused kernels** | Fuse Q-norm, KV-norm, RoPE(Q), RoPE(KV), indexer layernorm, indexer RoPE, FP8 quant, and KV-cache write. vLLM took DeepSeek-V3.2 from ~33 kernel launches/layer toward ~10. | **1.28× at batch size 1** measured (85.8 → 109.3 tok/s, 4×GB200). Also shrinks rank-arrival skew because there are fewer launch/sync points per rank. | **Med–High** | [verified, vendor-measured] [vLLM AA post](https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis) |
| 3 | **Fuse AllReduce + RMSNorm (+quant) with a one-shot algorithm below a byte threshold** | FlashInfer exposes `trtllm_allreduce_fusion` / `allreduce_fusion` with residual-add + RMSNorm + FP8/NVFP4 quant epilogues and an explicit `use_oneshot` switch; vLLM wires it in as a torch.compile pass. | vLLM measures **up to 15%** from AllReduce+RMSNorm alone and ~5% TPOT from `allreduce_rms` pattern recognition on Qwen3.5. Directly attacks our 19.6% collectives. | **Med** | [verified] [FlashInfer comm API](https://docs.flashinfer.ai/api/comm.html), [vLLM torch.compile](https://vllm.ai/blog/2025-08-20-torch-compile) |
| 4 | **Speculative padding at the prefill→decode handoff** | Prevent mixed (prefill+decode) batches at handoff by padding the speculative dimension, so the decode instance never falls off its full-CUDA-graph path. | **18 ms of TPOT eliminated** on GLM-5.2 B300 (TPOT ~40 ms → ~22 ms). Only relevant if/when we disaggregate — but it is the largest single win in that post. | **Med** | [verified] [vLLM GLM-5.2](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd) |
| 5 | **GPU-native input preparation + zero CPU↔GPU sync (Model Runner V2 pattern)** | Build `input_ids`, `positions`, `query_start_loc`, `seq_lens` with Triton kernels on-device; let the prep kernel consume rejection-sampling results directly; move outputs off on a separate CUDA stream. Triton sampler uses Gumbel-Max without materializing softmax. | **−6.3% TPOT** (GLM-4.7-FP8, 4×GB200, MTP=1) and **−11% TPOT** (GLM-5.2, B300). +56% throughput on a tiny model where CPU dominates. | **High** (architectural) | [verified] [MRV2](https://vllm.ai/blog/2026-03-24-mrv2), [GLM-5.2](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd) |
| 6 | **Multi-stream overlap of indexer / KV-compression / cache-insert pipelines** | vLLM's DeepSeek-V4 path runs three independent pipelines (indexer compute, main-attention KV compression, sliding-window insertion) on separate CUDA streams. | **5–6% end-to-end latency reduction at low batch size** — explicitly a low-batch win, which is exactly our C1 regime. | **Med** | [verified] [DeepSeek V4 in vLLM](https://vllm.ai/blog/2026-04-24-deepseek-v4) |
| 7 | **Adaptive / per-request verification budget for EAGLE** | Learned confidence head → per-position survival probability (running product) → global top-B over survival scores, admitting a contiguous prefix per request. Budget B maximizes expected tokens per step-time. | On DeepSeek-V4-Pro the 7th drafted token survives <10% of the time vs >70% for the first — so a fixed 3-1-4 tree is provably wasting verification FLOPs at some positions. Pareto-optimal across C1–C256 in vLLM's sweep. | **Med–High** (needs a trained confidence head) | [verified] [DSpark adaptive verification](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) |
| 8 | **Lazy Pre-Norm: fold affine-free RMSNorm into the *next* GEMM as a prologue** | Because `(A * rstd[:,None]) @ B == (A @ B) * rstd[:,None]`, you can accumulate the row square-sum *in parallel with* the k-loop and apply `rstd` as an epilogue. No cyclic dependency, no extra HBM round-trip. | Meta reports hiding **up to 90% of a norm kernel's latency**. Norms are ~10% of a typical LLM's latency; our dense GEMM is 37.1%, so this is a direct nibble at the largest bucket. | **Med** (Triton/TLX/Helion kernel work) | [verified] [PyTorch: Towards Free Normalization](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/) |
| 9 | **Two-sided NVLink all-to-all backend for MoE dispatch/combine** | vLLM's `--all2all-backend=flashinfer_nvlink_two_sided` on the decode instance. | **−4% TPOT** on GLM-5.2 decode. Small but nearly free if the backend exists. | **Low** (if FlashInfer already linked) | [verified] [vLLM GLM-5.2](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd) |
| 10 | **NVFP4 dispatch for MoE all-to-all** | Quantize token activations to FP4 *before* the all-to-all, not after. | **4× reduction in all-to-all volume** vs FP16 dispatch. Attacks the 19.6% collectives bucket from the volume side rather than the latency side. | **Med** | [verified] [vLLM WideEP on GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) |
| 11 | **Decode Context Parallelism with query replication** | Shard the KV/latent cache along the *sequence* dimension across TP ranks; `AllGather Q → compute → AllGather+ReduceScatter` with online-softmax reweighting. `VLLM_DCP_Q_REPLICATE=1` skips the Q all-gather entirely by replicating the q-projection at load time. | For cost-per-user at C64+: 1,863 → 6,091 tok/s/GPU on 8×B200 Kimi-K2.6 NVFP4 with ~67k median input. Our C1→C16 4.7× fall-off is partly a KV-capacity problem. | **High** | [verified] [Decode Context Parallelism](https://vllm.ai/blog/2026-08-07-decode-context-parallelism) |
| 12 | **`FULL_DECODE_ONLY` CUDA graphs + dynamic speculative lengths under full graphs** | Capture full graphs only for uniform-decode batches (`max_query_len == 1 + num_spec_tokens`), leaving prefill eager/piecewise. MRV2 adds dynamic spec lengths *inside* full graphs. | Removes graph-capture memory pressure and gets us full-graph coverage on the latency path even when the draft length varies. | **Low–Med** | [verified] [vLLM CUDA graph modes](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) |
| 13 | **Nightly Pareto-sweep perf CI with accuracy gates** | vLLM runs 17 model×hardware recipes nightly: `vllm-bench` for TTFT/TPOT, `lm-eval` for GSM8K/GPQA/AIME, BFCL for tool-calling. Results into ClickHouse, dashboards at ci.vllm.ai, a bot that nominates culprit commits (~70% correct). | Not a speedup — but it is why their numbers are trustworthy and ours need to be. Cheap to copy. | **Low** | [verified] [Keeping vLLM Production Quality](https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality) |

**Things explicitly *not* worth stealing** (negative results, all [verified]):
- **TP2 + EP is worse than plain TP2** at 2-GPU prefill scale — all-to-all overhead exceeds the benefit
  (vLLM GLM-5.2 post). Don't reach for EP below ~4 ranks.
- **MTP degrades throughput above ~256 batch** on GB300 DeepSeek-V3.2 even at >80% acceptance; TPOT
  *increases*. Our C64 aggregate objective should have a spec-decode-off crossover point.
- **Hadamard transforms in the DSA indexer were removed with no observed accuracy effect** (vLLM
  DeepSeek-V3.2 post). If our indexer still does them, they're free to delete.
- **Bitwise-deterministic (batch-invariant) kernels cost 2.4×.** Only pay this for RL, never for serving.

---

## vLLM

### What they run

vLLM is the reference open engine: V1 core (Jan 2025), Model Runner V2 experimental core (Mar 2026),
torch.compile-by-default, FlashInfer as the primary Blackwell kernel provider, `KVConnectorBase_V1`
for everything KV-shaped, and a plugin surface for hardware backends. Current line at time of writing
is **v0.27.1** (11 Aug), preceded by v0.27.0 (10 Aug, 561 commits / 242 contributors, Kimi-K3 support,
PyTorch 2.13.0, FlashAttention 4) and v0.26.0 (27 Jul).
[verified — [releases](https://github.com/vllm-project/vllm/releases); note the GitHub listing's year
field rendered ambiguously in my extraction, but the content (Kimi K3, PyTorch 2.13, SM107/Rubin)
places these in 2026.]

### The V1 rewrite: what it actually fixed

The V1 motivation is a CPU-overhead story, and the numbers from the earlier v0.6.0 post are the clearest
statement of the problem anywhere in open serving: on **Llama-3-8B, 1×H100**, the HTTP API server took
**33%** of total execution time, scheduling took **29%**, and only **38%** was actual GPU execution.
[verified, [vLLM v0.6.0 perf update](https://vllm.ai/blog/2024-09-05-perf-update)]

The fixes, each with its own measurement:
- **API server in a separate process over ZMQ** (PR #6883) — removes GIL contention when streaming
  (~76 stream objects/sec for an 8B model).
- **Multi-step scheduling** (PR #7000) — schedule + prepare inputs once, run the model for `n`
  consecutive steps. **+28% throughput** on Llama-70B / 4×H100.
- **Asynchronous output processing** (PRs #7049, #7921, #8050) — process step *n*'s tokens while step
  *n+1* runs on the GPU. **−8.7% TPOT** on Llama-70B.
- **Object caching** (#7162) — **+24% e2e**; non-blocking H2D copies (#7172); fast sampling path (#7117).
- Net: **2.7× throughput and 5× lower TPOT** on Llama-3-8B / 1×H100 / ShareGPT.

V1 then generalized this into architecture [verified, [V1 alpha](https://vllm.ai/blog/2025-01-27-v1-alpha-release)]:
- **EngineCore / AsyncLLM split.** `EngineCore` is synchronous and step-based (schedule → forward →
  sample). `AsyncLLM` is asyncio and owns tokenization, detokenization, multimodal preprocessing and
  streaming. They run in different processes over ZMQ sockets, so all CPU work overlaps the GPU loop.
- **Symmetric tensor parallelism.** Worker processes cache request state and receive only *diffs*; the
  scheduler no longer has to be colocated with worker 0.
- **Zero-overhead prefix caching.** Constant-time eviction and fewer Python objects, measured at
  **<1% throughput loss at a 0% hit rate** — which is why it became default-on.
- **Scheduler as a dict.** A scheduling decision is `{request_id: num_tokens}`, which makes chunked
  prefill, prefix caching and speculative decoding one abstraction rather than three special cases.
- **Piecewise CUDA graphs** — split the captured graph into graph-safe and graph-unsafe regions.
- Reported gain: **up to 1.7×** throughput vs V0 on Llama text models.

The `Inside vLLM` post adds implementation detail worth copying [verified,
[Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)]:
- DP replica selection scores engines as `len(waiting) * 4 + len(running)`.
- Prefix cache blocks are 16 tokens; block hash chains the previous block's hash with the current tokens
  plus optional LoRA id and cache salt (`hash_request_tokens`, `cached_block_hash_to_block`,
  `find_longest_cache_hit`).
- Scheduler prioritizes the `running` (decode) queue before pulling from `waiting`; a prefill longer than
  `long_prefill_token_threshold` is capped to exactly that value.
- KV connector lifecycle: `get_num_new_matched_tokens` → `update_state_after_alloc` →
  `build_connector_meta` → `start_load_kv` (pre-forward) → `wait_for_save` (post-forward).

### Model Runner V2 — the piece most worth porting

`export VLLM_USE_V2_MODEL_RUNNER=1`. Lives at `vllm/v1/worker/gpu/` (`input_batch.py`, `states.py`,
`model_states/`, `sample/`, `async_utils.py`, `attn_utils.py`, `cudagraph_utils.py`, `block_table.py`,
`warmup.py`, `kv_connector.py`, `pp_utils.py` / `dp_utils.py` / `cp_utils.py`). Maintained by Woosuk Kwon,
marked experimental. [verified, [dir listing](https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker/gpu)]

Four design moves [verified, [MRV2 post](https://vllm.ai/blog/2026-03-24-mrv2)]:
1. **Decoupled persistent batch.** Each active request owns a stable row in a fixed-size state table;
   per-step input tensors are produced by *gather* ops. Request insertion/removal/reordering no longer
   perturbs the block table layout, and the fragile `CachedRequestState` backup disappears.
2. **GPU-native input preparation.** `input_ids`, `positions`, `query_start_loc`, `seq_lens` are built
   on-device with Triton kernels.
3. **Async-first with zero CPU↔GPU synchronization.** Prep kernels consume rejection-sampling results
   directly on the GPU; outputs are transferred on a separate CUDA stream.
4. **Triton-native sampler.** Gumbel-Max sampling without materializing softmax; top-k logprobs computed
   by finding top-k logits first; prompt logprobs chunked; spec decode handled by indirection rather than
   by expanding request state.
5. **`ModelState` ABC** isolates model-specific logic (multimodal embeddings, extra inputs, attention
   metadata, CUDA-graph capture). Largest file dropped from 6,700+ lines to under 1,300.

Measured: **+56% throughput** (Qwen3-0.6B, 1×GB200: 25K vs 16K output tok/s — a CPU-bound stress case)
and **−6.3% TPOT** (GLM-4.7-FP8, 4×GB200, MTP=1). On the GLM-5.2 B300 deployment it contributed **−11%
TPOT** together with Triton JIT warmup, local argmax reduction for multi-GPU MTP, and dynamic speculative
lengths under full CUDA graphs. [verified — vendor-measured, not independently reproduced]

Unsupported at v0.18.0: linear-attention models, spec methods beyond Eagle/Eagle3/MTP, EPLB, DBO, logits
processors, LoRA. By v0.27 it had grown pooling/embedding/encoder tasks.

### Scheduler, chunked prefill, and the flags that matter for latency

[verified, [optimization docs](https://docs.vllm.ai/en/latest/configuration/optimization.html)]

- Chunked prefill is **on by default** in V1. `max_num_batched_tokens` is the ITL/TTFT dial:
  **~2048 for better inter-token latency**, higher (>8192) for throughput. GLM-5.2 decode used **1024**.
- `-O0` (no compile, fastest startup) … `-O1` (PIECEWISE) … `-O2` (default, FULL_AND_PIECEWISE) … `-O3`
  (currently equivalent to `-O2`).
- **`--numa-bind`**, with `--numa-bind-nodes 0 0 1 1` or `--numa-bind-cpus 0-3 4-7 48-51 52-55`. We have
  **2 NUMA nodes** — this is a direct, cheap check for us.
- CPU core budget: `2 + N` physical cores minimum (1 API server, 1 engine core, N GPU workers); with DP,
  `A + DP + N + (1 if DP>1)`. The engine-core scheduler is a busy loop and is *very* sensitive to CPU
  starvation. [inferred: if our C1 profile shows collectives skew, verify no core is being stolen from
  a rank's launch thread before blaming the network.]
- `VLLM_USE_FASTOKENTS=1` for BPE tokenizers (Qwen/Llama/DeepSeek family); `--api-server-count N`
  when tokenization is the bottleneck.
- Startup: torch.compile artifacts cached in `~/.cache/vllm`; `VLLM_FORCE_AOT_LOAD=1` to hard-fail on a
  cache miss; `--kv-cache-memory <bytes>` to skip memory profiling on subsequent boots.

### CUDA graph modes

[verified, [design/cuda_graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html)]

`CompilationConfig.cudagraph_mode` ∈ {`NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`,
`FULL_AND_PIECEWISE`}. Backends declare `AttentionCGSupport` ∈ {`ALWAYS` (FA3), `UNIFORM_BATCH` (FA2),
`UNIFORM_SINGLE_TOKEN_DECODE` (FlashInfer, Mamba), `NEVER`}, and vLLM auto-downgrades incompatible modes.

The important semantic for us: **"uniform decode" includes speculative decode** — a batch where every
request has `query_len == 1 + num_spec_tokens` qualifies for full-graph capture. That means an EAGLE
3-1-4 verification step *can* run entirely under a full CUDA graph as long as the tree width is uniform
across the batch. `FULL_AND_PIECEWISE` is the default and the recommendation for low latency;
`FULL_DECODE_ONLY` is the recommendation for a dedicated decode instance in a P/D split (and is exactly
what vLLM used for GLM-5.2 decode).

### torch.compile integration and the fusion pass catalogue

[verified, [torch.compile post](https://vllm.ai/blog/2025-08-20-torch-compile)]

vLLM compiles by default (disable with `-O0` / `--enforce-eager`). TorchDynamo traces to fx; TorchInductor
lowers to Triton/C++; vLLM adds its own custom passes. The catalogue, with vLLM's reported gains:

| Fusion pass | Reported gain |
|---|---|
| RMSNorm + Quant (FP8) | — |
| SiLU-Mul + Quant (FP8) | up to 8% throughput |
| Attention + Quant (FP8) | up to 7% |
| **AllReduce + RMSNorm** | **up to 15%** |
| AllReduce + RMSNorm + Quant | — |
| Sequence Parallelism & Async TP | up to 10% |
| No-op elimination, fix-functionalization | — |
| Pad + Quant (MoE), Finalize + Slice | ~6% (PR #30647) |
| (upcoming) Attention + Quant FP4, SiLU-Mul + Quant FP4 | — |

Config: `compile_sizes: [1, 2, 4]` to specialize small batch sizes — *directly relevant to a C1 latency
target*. Cache at `~/.cache/vllm/torch_compile_cache`, disabled with `VLLM_DISABLE_COMPILE_CACHE=1`,
reusable across machines with identical environments. Acknowledged pain: startup time, dependence on
private torch.compile APIs, "weird caching issues", and the fact that **custom ops (attention,
collectives, sub-byte quant) still require hand-written fusion passes** — the compiler does not find
these itself.

### GLM-5.2 on B300: the closest published analogue to our system

[verified, [From Day 0 to Production SLAs: Serving GLM-5.2](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd)]
**vendor-measured; config asymmetries noted below.**

Topology: 3 servers × 8 B300 = 24 GPUs, 4 prefill instances (TP1 DP4 EP, 4 GPUs each) + 1 decode
instance (TP1 DP8 EP), KV over NIXL. Model `GLM-5.2-NVFP4`, 744B total / 40B active.

Decode command (abridged, verbatim flags):
```
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /mnt/model/glm/GLM-5.2-NVFP4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --max-num-batched-tokens 1024 \
  -ep -tp 1 -dp 8 \
  --gpu-memory-utilization 0.90 \
  --all2all-backend=flashinfer_nvlink_two_sided \
  --speculative-config='{"method":"mtp","num_speculative_tokens":3}' \
  --fingerprint-mode=none
```
Prefill is identical except `-dp 4`, `kv_role: kv_producer`, `gpu-memory-utilization 0.92`, and
**`num_speculative_tokens: 1`** — the asymmetry is deliberate: prefill optimizes for fast handoff,
decode amortizes over the latency path.

Attributed wins:
- **Speculative padding: −18 ms** (TPOT ~40 → ~22 ms) by preventing mixed batches at the P/D handoff.
- **Model Runner V2: −11% TPOT.**
- **FlashInfer NVLink two-sided all-to-all: −4% TPOT.**
- **DSA index sharing with MTP** via `index_share_for_mtp_iteration`. Supporting merged PRs:
  **#45895** (fixed GLM-5.2 MTP normalization; acceptance improved from ~3 to ~4 accepted tokens),
  **#47238** (index buffer layout for batched requests), **#47448** (reuse post-final-norm hidden state).
- Final: **TPOT ~17 ms**, mean TTFT ≤ 2.5 s, 700 concurrent at 8K / 300 at 16K / 25 at 256K under SLA.

Negative results and operational scars, all published:
- **TP2 + EP performed worse than plain TP2** at two-GPU prefill scale.
- **TP1 DP2 EP was the most per-GPU-efficient prefill config but had insufficient KV capacity**; they
  chose DP4 and paid ~8% per-GPU efficiency to reach 1M-token context.
- **A host-memory leak**: `SingleTypeKVCacheManager.new_block_ids` entries were never cleared for
  non-Mamba models, causing linear host memory growth over hours. Fixed by **PR #44490**.
- They call **MTP acceptance rate "among the earliest warning signals available"** and monitor it as a
  first-class production metric alongside per-pool TTFT/TPOT percentiles and KV-transfer queue depth.

Benchmark methodology: `vllm bench serve --dataset-name random --random-input-len 16384
--random-output-len 1000 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90`, with
**no prefix-cache hits** (worst case). Note this is 16K/1K, not AA's ~10K/1.5K.

### Blackwell / B200 specifically

**WideEP on GB200** [verified, [Part I](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1)] — vendor-measured
26.2K prefill TPGS and 10.1K decode TPGS at 2K/2K, from 4 prefill instances (2 GB200 each) + 1 decode
instance (8 GB200); claimed 3–5× over an H200 deployment that used 16 prefill + 32 decode GPUs. Mechanisms:

- **NVFP4 GEMM** for MoE experts and o-proj via **FlashInfer TRT-LLM-Gen** kernels; **FP8 GEMM** kept for
  the MLA query up-projection to preserve attention quality.
- **NVFP4 MoE dispatch** — quantize activations to FP4 *before* the all-to-all: **4× less comm volume**.
- **Kernel fusions**: `RoPE + Quant + Q-write` (decode, eliminates two HBM round-trips);
  `RoPE + Quant` (prefill); FlashInfer **`concat_mla_k`** for concatenating `k_nope` and `k_rope`, with
  one warp per token/head chunk (16 heads at a time), 8-byte nope / 4-byte rope vectorized loads,
  software pipelining with L2 prefetch, and register reuse of rope values across 128 heads.
- **Scale prefill *down***: microbenchmarks showed MLA and MoE throughput **plateau at ~64K batch**, so
  going from 4 GPUs to 2 halves the NCCL all_gather/reduce_scatter cost with no compute loss.
- **Weight Offloading v2**: asynchronous prefetch on a separate CUDA stream (v1 used UVA/PCIe and was
  slow). Knobs: `group_size`, `num_in_group`, `prefetch_step`. DeepSeek-R1 prefill offloads one of every
  two MoE GEMM weight sets.
- **Chunking knobs, with their GB200 settings** — these are the ones we should audit in our fork:

| Env var | Default | GB200 setting |
|---|---|---|
| `VLLM_ENABLE_MOE_DP_CHUNK` | enabled | **disabled for prefill**; set to batch size for decode |
| `VLLM_MOE_DP_CHUNK_SIZE` | 256 tokens | (see above) |
| `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` | enabled | **disabled** (enough memory for full batches) |
| `VLLM_FUSED_MOE_CHUNK_SIZE` | 16K tokens | — |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | 128 | **2048** for throughput-optimized decode |

**GPT-OSS on Blackwell** [verified, [post](https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations)] — **+38%
max-throughput, +13% min-latency**. `--cuda-graph-capture-size 2048`, `--api-server-count 20` (or
`--stream-interval 20`), `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`. Async scheduling (PR #23569) ≈ **+10%**.
**Stream interval** (PR #27869) buffers tokens before network dispatch while preserving first-token
latency: **+57% end-to-end at high concurrency** by cutting HTTP/gRPC overhead. FlashInfer MoE backends
landed as `trtllm-gen` (PR #23819) and `cutlass` (PR #23696); FP8 KV attention (PR #25674).

**GB300 DeepSeek-V3.2** [verified, [post](https://vllm.ai/blog/2026-02-13-gb300-deepseek)] — the most
useful part is the *negative* result: the **sparse-attention indexer path is 2.7× longer in kernel time
than standard MLA**, and `DeepseekV32IndexerBackend` overhead is significant below ~10k context. R1
(dense-attention MoE) prefill throughput is ~3× V3.2's. Software: vLLM v0.14.1 + PR #32698, CUDA 13.0,
`VLLM_USE_FLASHINFER_MOE_FP4=1`.

| Config | Model | Workload | Result |
|---|---|---|---|
| NVFP4 + TP2, 1×GB300 | DeepSeek-V3.2 | ISL 2k, OSL 1, batch 64 | 7,360 tok/GPU/s prefill (1.8× over FP8) |
| NVFP4 + TP2, 1×GB300 | DeepSeek-V3.2 | ISL 2k, OSL 1k | 2,816 tok/GPU/s |
| NVFP4 + EP2, 2×GB300 | DeepSeek-R1 | ISL 2k, OSL 1, batch 256 | 22,476 tok/GPU/s prefill |
| NVFP4 + EP2, 2×GB300 | DeepSeek-R1 | ISL 2k, OSL 1k | 3,072 tok/GPU/s |

**Qwen3.5-397B-A17B-NVFP4 at 25K TPS/GPU on GB200 NVL72**
[verified, [post](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)] — ISL 8192 / OSL 1024, concurrency
64–5120, 1× DEP8 decode + 4–8 DEP2 prefill, `vllm/vllm-openai:nightly-d223c90` + Dynamo 1.2.0.dev.
`--async-scheduling` called "one of the key features behind crossing 25K". Also:
`--gdn-prefill-backend flashinfer` (FlashInfer PR #3001 gave 1.02–5.78× kernel speedup; vLLM PR #40717
gave up to 5.92× microbenchmark, 1.13× e2e prefill throughput, −12% mean TTFT); `--language-model-only`
unlocks a fused QK-norm + RoPE + gate path; `--stream-interval 100`; `--max-num-batched-tokens 16384`
(2× ISL, **+8% total TPS** at high concurrency); `--max-cudagraph-capture-size 640–768` for cc≥4096;
`VLLM_SSM_CONV_STATE_LAYOUT=DS`. Two async-scheduling race fixes were prerequisites: **PR #48481**
(races in KV block transfer between prefill and decode) and **PR #45357** (defer block freeing until
in-flight async steps complete). Explicitly **did not measure concurrency 1–32**.

### Sparse attention (DSA) in vLLM — directly comparable to our indexer

[verified, [DeepSeek-V3.2-Exp in vLLM](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)]

- Lightning indexer computes top-**2048** tokens per query. Logits shape `(n, h)` per query, head-weighted,
  then row-wise top-k. Prefill and decode need separate handling; `ks`/`ke` tensors mark per-query context
  boundaries, and batching requires index-offset bookkeeping (`ks = [0]*q1 + [q1]*q2 ...`).
- **Separate cache buffers** for the indexer K-cache and the MLA K-cache.
- **FP8 KV layout, 656 bytes/token**: 512 B quantized NoPE (`float8_e4m3`) + 16 B (four `float32` scales)
  + 128 B unquantized RoPE (`bfloat16`).
- Indexer cache stored per-block with **block size forced to 64** because FlashMLA is tailored to it;
  layout is `x_fp8[:, :block_size*head_dim]` values then scales.
- External kernels: DeepGEMM `fp8_mqa_logits()` for relevance scores, FlashMLA for sparse attention,
  a TileLang kernel (DeepSeek's reference) for fused top-k.
- Negative/unresolved: expert parallelism had a bug at publication; **Hadamard transforms were removed
  with no observed accuracy effect**; materializing the logits tensor at high batch × long context is
  still a problem.

**DeepSeek-V4 in vLLM** [verified, [post](https://vllm.ai/blog/2026-04-24-deepseek-v4)] is worth reading
in full before we take on V4, but three items transfer now:
- Fusion speedups measured: **compressor + RMSNorm + RoPE + cache-insert ≈ 1.4–3×**,
  **inverse-RoPE + fp8 quant ≈ 2–3×**, **fused Q-norm + KV-RoPE + K-insert ≈ 10–20×**.
- **Multi-stream parallelization** of three independent pipelines (indexer, main-attention KV compression,
  sliding-window insertion) → **5–6% end-to-end latency reduction at low batch sizes**.
- **Unified block sizing**: fix the logical block at 256 native token positions for *every* compressed
  layer, so slot mapping and scheduler accounting stay uniform regardless of compression ratio. Five cache
  types collapse into three page-size buckets.
- Launch flags: `--kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --data-parallel-size 8
  --attention_config.use_fp4_indexer_cache=True --tokenizer-mode deepseek_v4
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` on 8×B200/B300.
- KV cache at 1M tokens: **9.62 GiB (bf16) vs 83.9 GiB** V3.2-style — **8.7× smaller**.

### Speculative decoding — the deepest published body of work in the ecosystem

Supported methods [verified, [docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)]:
EAGLE, EAGLE3, MTP, draft-model, PARD, MLP speculator, n-gram, suffix decoding, DFlash, DSpark, and a
custom proposer backend. Key `--speculative-config` fields: `method`, `model`, `num_speculative_tokens`,
`draft_tensor_parallel_size`, `parallel_drafting`, `rejection_sample_method` (standard | synthetic |
block), `use_heterogeneous_vocab`; suffix decoding adds `suffix_decoding_max_tree_depth` (default 24)
and `suffix_decoding_max_cached_requests` (10000). Pipeline parallelism is incompatible with spec decode.

**P-EAGLE** [verified, [post](https://vllm.ai/blog/2026-03-13-p-eagle), arXiv 2602.01469, vLLM PR #32887,
v0.16.0+]. Mechanism: standard EAGLE needs K sequential forward passes for K draft tokens. P-EAGLE emits
all K in **one** pass — position 1 uses the generated token embedding + `h_context`; positions 2..K use a
**learned shared mask-token embedding and shared hidden state** as placeholders. Implementation details
worth copying: a single fused Triton kernel does batch-metadata reconstruction (copy/expand the target
batch, insert MASK tokens, build rejection masks, compute hidden-state mappings) in one pass; rejected
tokens map to `PADDING_SLOT_ID = -1`; CUDA-graph capture range extends by `K × max_num_seqs`. Training
needs a **sequence-partition algorithm** because N×K positions at N=8192, K=8 means 65,536 positions and
a 65K×65K attention (8 GB bf16).

| Benchmark | GPT-OSS-20B, 1×B200, c=1, K=7 | Acceptance length vs EAGLE-3 |
|---|---|---|
| MT-Bench | 1.55× | 3.70 vs 3.27 (+13%) |
| HumanEval | 1.55× | 3.94 vs 3.03 (+30%) |
| SPEED-Bench | 1.69× | 3.38 vs 2.59 (+31%) |

Negative: **at c=64 the speedup collapses to 1.05–1.25×**, and vanilla EAGLE checkpoints cannot be used —
the drafter must be trained for parallel drafting. Checkpoints published: `amazon/gpt-oss-120b-p-eagle`,
`amazon/GPT-OSS-20B-P-EAGLE`, `amazon/Qwen3-Coder-30B-A3B-Instruct-P-EAGLE`.

**EAGLE 3.1** [verified, [post](https://vllm.ai/blog/2026-05-26-eagle-3-1), vLLM v0.22.0+]. Two changes
against "attention drift": **FC normalization after each target hidden state and before the FC layer**,
and **feeding post-norm hidden states into the next decoding step** — the stated design goal is to make
the drafter behave like a recursive invocation rather than extra layers appended to the target. Reported:
**up to 2× longer acceptance length** in long-context workloads; on Kimi K2.6, **2.03× per-user output
throughput at concurrency 1**, 1.71× at C=4, 1.66× at C=16. Backward compatible with EAGLE-3 checkpoints;
draft published at `lightseekorg/kimi-k2.6-eagle3.1-mla`. **[inferred] Given we run EAGLE 3-1-4 and are
targeting Kimi K3 next, EAGLE 3.1's two-line architectural change is the highest-value-per-hour item in
the spec-decode family — it needs a retrained drafter but no serving-side redesign.**

**DSpark adaptive verification** [verified, [post](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification),
PR #47808]. A learned confidence head scores each drafted token's survival probability; the scheduler turns
these into running-product survival probabilities and does a **global top-B over survival scores**, which
admits a contiguous prefix of each request's draft with no additional constraint. B is chosen to maximize
expected tokens per step-time (numerator: bonus tokens + draft survival; denominator: profiled step cost
in μs). CPU-side sizing runs on one-step-old confidences while the GPU works; GPU-side sizing uses current
confidences via a compiled PyTorch/Triton path.

```
--speculative-config '{"method":"dspark","attention_backend":"FLASH_ATTN",
  "num_speculative_tokens":7,"draft_sample_method":"probabilistic",
  "enable_adaptive_verification":true}'
--kv-cache-dtype fp8 --max-num-seqs 256 --gpu-memory-utilization 0.8
```
Measured on DeepSeek-V4-Pro-0813, 8×B300 (SM100), TP=8 + EP, FP8 KV, 880 prompts, temp 1.0, ≤2048 output.
Constraints: requires `AttentionCGSupport.ALWAYS`; incompatible with `--enforce-eager`, LoRA, and pipeline
parallelism; output logprobs are rejected when enabled. **The load-bearing observation: on this model the
7th drafted token survives <10% of the time versus >70% for the first.**

**DFlash** [verified, [parallel drafting post](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting)]
routes verifier hidden states *into the speculator's KV cache* rather than expanding the input sequence,
and generates candidate blocks by block diffusion; training uses sequence-length sparsification (block
predictions only at random anchor points). Note the post carries a 2026-07-29 errata: Figure 1 was
replotted because of an erroneous environment setup — a reminder that even first-party plots move.

**`vllm-project/speculators`** [verified, [repo](https://github.com/vllm-project/speculators)] is the
usable artifact: a HF-compatible checkpoint format with a standardized `speculator_config`, training for
single- and multi-layer drafters across MoE and non-MoE (and VLM) targets, conversion tooling for external
research checkpoints, and `vllm serve <speculator_model>` deployment. **This is a real thing to adopt
rather than rebuild.**

### PD disaggregation and the KV connector API

[verified, [disagg_prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)]

Three abstractions: **Connector** (retrieve KV between producer and consumer), **LookupBuffer**
(`insert` non-blocking, `drop_select` blocking), **Pipe** (`send_tensor` / `recv_tensor`). Config is a
single JSON blob: `{"kv_connector": "...", "kv_role": "kv_producer|kv_consumer|kv_both",
"kv_connector_extra_config": {...}}`. Shipped connectors: `NixlConnector` (async, UCX/GDS backends),
`LMCacheConnectorV1`, `MooncakeConnector`, `MoRIIOConnector` (ROCm), `MultiConnector` (chains several),
`OffloadingConnector` (CPU), `FlexKVConnectorV1`, P2P connectors, and an `ExampleConnector` reference.
The decode stage can reuse prefill's token IDs through `kv_transfer_params`, skipping re-tokenization.

**The TileRT integration is the sharpest demonstration of why this API matters**
[verified, [vLLM × TileRT](https://vllm.ai/blog/2026-07-14-vllm-tilert-pd)]. A `KVConnectorBase_V1`
implementation acting as a pure `kv_producer`; a lightweight router marks requests with `max_tokens=1`
and embeds the target decode node in `kv_transfer_params`; the TileRT connector **claims only marked
requests and is a strict no-op for everything else**, so it composes under `MultiConnector` with the
normal decode pool. State moves by **RDMA one-sided writes into pre-registered GPU buffers** (Mooncake or
NIXL transfer engine), carrying compressed KV caches, **sparse-attention index caches**, and metadata,
fully overlapped with the prefill forward window. The vLLM prefill side runs `--enforce-eager
--kv-cache-dtype fp8_ds_mla --tensor-parallel-size 8` with MTP=1, loading the connector via
`kv_connector_module_path: tilert.pd_vllm.prefill_connector`. Limits: **one in-flight request per TileRT
decode node**, and only GLM-5/5.1 and DeepSeek-V3.2 are covered.

**[inferred] The design lesson for our fork: make the decode side pluggable behind a claim-filtering
connector. It lets us run a persistent-kernel / megakernel decode path for C1 latency traffic without
forking the whole scheduler, and it is exactly how Z.ai ships GLM-5.1-highspeed alongside a normal pool.**

Other disaggregation work, briefly:
- **KV offloading connector** [verified, [post](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)]:
  `--kv_offloading_backend native --kv_offloading_size <GB>` (0.14.0+). The decisive change was a **0.12.0
  memory-layout rework making blocks physically contiguous across all layers**, taking block size from
  32 KB to 2 MB and improving transfer throughput "an order of magnitude". `cudaMemcpyAsync` (DMA) hits
  **83.4 GB/s bidirectional** vs a custom copy kernel's 68.5 GB/s, and the custom kernel costs ~6% of
  model throughput at 0% hit rate because it burns SMs. Negative: DMA loses to the custom kernel on
  sub-2 MB single-direction blocks.
- **Mooncake Store** [verified, [post](https://vllm.ai/blog/2026-05-06-mooncake-store)]: GPUDirect RDMA,
  fully async on dedicated I/O threads (no kernel-launch interference), MultiConnector chaining. On real
  agentic traces (Kimi-2.5 NVFP4, 12 GB200, 1P1D): **3.8× throughput, 46× lower P50 TTFT, 8.6× lower e2e
  latency, cache hit rate 1.7% → 92.2%**. Near-linear scaling 12 → 60 GPUs at >95% hit rate.
- **AFD plugin** (attention/FFN disaggregation) [verified, [post](https://vllm.ai/blog/2026-07-23-vllm-afd-plugin)]:
  all published numbers are on **Ascend 910C**, not NVIDIA, and the honest result is mixed —
  48A16F was **worse** than the EP64 baseline (−5.3% at 16K, −10.0% at 32K) while 64A16F was better
  (+11.3%, +9.0%). Connectors: `P2pNcclAFDConnector` (GPU), `CAMP2pAFDConnector` / `CAMAsyncAFDConnector`
  (NPU). Model runner V1 only, decode-only graph modes, max two ubatches for DBO.
- **Elastic EP** [verified, [post](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism)]:
  `POST /scale_elastic_ep {"new_data_parallel_size": 8}`. The interesting engineering is **standby
  communication groups** (`StatelessGroupCoordinator`) that span the target rank set while the old groups
  keep serving, plus NIXL's `connect_ranks()` / `disconnect_ranks()` for incremental transitions, plus a
  **two-stage barrier** (first with timeout, ranks that arrive early return to the engine loop; second
  without timeout) to avoid deadlocking asynchronous DP engine cores. **No performance numbers are
  published**, and it is TP=1 / Ray-DP-backend only. Honest assessment: architecturally instructive,
  operationally not yet real.

### Expert parallelism and MoE kernels

`--all2all-backend` options and their intended use
[verified, [EP deployment docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)]:

| Backend | Use case |
|---|---|
| `allgather_reducescatter` | default, general-purpose |
| `deepep_high_throughput` | multi-node **prefill**; grouped GEMM, contiguous layout |
| `deepep_low_latency` | multi-node **decode**; CUDA-graph support, masked layout |
| `flashinfer_nvlink_one_sided` | MNNVL systems |
| `flashinfer_nvlink_two_sided` | MNNVL systems — the one that gave −4% TPOT on GLM-5.2 decode |

`EP_SIZE = TP_SIZE × DP_SIZE`. EPLB via `--enable-eplb` and `--eplb-config` JSON
(`window_size` 1000, `step_interval` 3000 engine steps, `num_redundant_experts` 0, `log_balancedness`,
`policy`, `communicator`). `--enable-dbo` with `--dbo-decode-token-threshold` overlaps all-to-all with
compute via microbatch worker threads that yield during GPU ops; vLLM reports it "dramatically reduces
MoE Dispatch/Combine latency in profiling traces"
[verified, [large-scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving)] — that post is the
2.2k tok/s/H200 wide-EP result on CoreWeave with ConnectX-7 / InfiniBand.

The **modular fused-MoE kernel matrix**
[verified, [moe_kernel_features](https://docs.vllm.ai/en/latest/design/moe_kernel_features.html)] is
architecturally the right idea and worth copying wholesale: kernel families (Triton standard/batched,
DeepGemm standard/batched, Cutlass FP4/FP8, FlashInfer, Marlin, TRTLlm MxFP4/NvFP4, HPC, GPT-OSS Triton,
ROCm Aiter) × quantization schemes × compatible all2all backends, with an explicit activation-format
contract (standard vs batched) so that a prepare/finalize stage and an experts stage can be mixed and
matched. Triton supports every quant type; DeepGemm is FP8-only; Cutlass FP4 is NvFP4-only; TRTLlm covers
MxFP4/NvFP4; FlashInfer covers NvFP4/FP8/mxfp8.

### Quantization on Blackwell

- `VLLM_USE_FLASHINFER_MOE_FP4=1` for NVFP4 MoE; `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` for the GPT-OSS
  MXFP4 weight / MXFP8 activation path. [verified]
- FP8 KV cache: dedicated post (`The State of FP8 KV-Cache and Attention Quantization in vLLM`,
  2026-04-22) covering Hopper and Blackwell validation and Flash-Attention-3 fixes; DeepSeek-style
  `fp8_ds_mla` KV dtype exists as a distinct option. [verified — index entry and usage in TileRT config;
  I did not fetch the FP8-KV post itself.]
- `llm-compressor` / `compressed-tensors` is the checkpoint production path; AutoRound × LLM Compressor
  (2025-12-09) covers W4A16 tuning-based PTQ. Machete/Marlin remain the mixed-precision GEMM kernels for
  W4A16 on pre-Blackwell parts; on Blackwell the FP4 path goes through FlashInfer/CUTLASS instead.
  [verified — blog index and MoE feature matrix. I did not fetch the AutoRound post.]
- **[unverified]** I could not source a vLLM-published head-to-head accuracy table of NVFP4 vs FP8 for
  GLM-5.2 specifically. The GLM-5.2 post gives NVFP4 accuracy scores (AIME 2025 86.67, GPQA 92.89,
  LongBench V2 64.01, MMLU-Pro 86.3, SWE-bench Verified 85.2) but no FP8 comparator.

### Batch invariance / determinism

[verified, [bitwise-consistent post](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference)]
The mechanism is that high-batch kernels parallelize across the batch dimension while low-batch kernels
parallelize *within* an instance, so reduction order changes with batch size. vLLM audited every kernel
invocation in the forward pass for bitwise equivalence with TorchTitan and made fused SiLU-MLP and
RMSNorm-with-residual batch-invariant. **Cost: the bitwise RL run is 2.4× slower.** Serving takeaway:
do not enable this; but the *diagnosis* is useful if we ever see acceptance-rate drift correlated with
batch size in our EAGLE path.

### Production stack and orchestration

`vllm-project/production-stack` [verified, [repo](https://github.com/vllm-project/production-stack)] is a
Helm-deployed reference stack: router with session-based routing (maximize KV reuse), round-robin,
Kubernetes service discovery, Prometheus/Grafana dashboards (QPS, TTFT, pending/running requests, KV cache
usage and hit rates). **Prefix-aware routing is marked WIP**; autoscaling and disaggregated prefill are on
the roadmap. No performance numbers published. Honest read: this is a starting template, not a competitive
advantage.

**Ray Serve LLM: I could not source it.** Both `docs.ray.io/en/latest/serve/llm/overview.html` and
`.../serving-llms.html` returned 404, and the `python/ray/llm` tree page carried no README content.
**[unverified]** — I will not describe its architecture from memory.

**KServe** [verified, [repo](https://github.com/kserve/kserve)] advertises vLLM and llm-d backends, an
OpenAI-compatible inference protocol, KV-cache offloading to CPU/disk, model caching, and request-based
autoscaling for generative workloads. The README does **not** document the `LLMInferenceService` CRD,
disaggregated P/D, or Gateway-API-based KV-aware routing with version numbers, so I am not claiming those.

---

## FlashInfer

**What it is:** the kernel library under vLLM, SGLang, TensorRT-LLM, TGI, MLC-LLM, LightLLM, lorax and
ScaleLLM. Unified Python APIs over multiple backends — FlashAttention-2/3, cuDNN, CUTLASS, and TensorRT-LLM
(`trtllm-gen`) — with JIT compilation, autotuning and a kernel cache, all CUDA-graph and torch.compile
compatible. Targets SM75 through Blackwell (and Jetson Thor SM110). Three install flavors:
`flashinfer-python` (JIT on demand), `flashinfer-cubin` (precompiled binaries), `flashinfer-jit-cache`
(prebuilt kernel cache). [verified, [repo](https://github.com/flashinfer-ai/flashinfer)]

**This is the highest-leverage external dependency in the whole ecosystem for us**, because it is where
NVIDIA's TRT-LLM kernels become callable from a Python serving engine without adopting TensorRT-LLM.

### Blackwell-specific and model-specific state, v0.6.17 (11 Aug 2026)

[verified, [release notes](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17)]

- **Kimi K3 MLA decode on Blackwell** — supports **96 global query heads against one KV head**, TP-local
  head counts down to 6, via packed query-token/head rows (**#4178**). Sparse MLA extended to
  non-power-of-two head counts and native `qk_rope_head_dim=0` (**#4108**). *We are targeting Kimi K3; this
  kernel exists today.*
- **MiniMax-M3 sparse attention** accepts vLLM's packed paged-KV layout with reduced per-call overhead
  (**#4039**, **#4324**).
- **MegaMoE expert-parallel path production-ready in vLLM**: full CUDA-graph capture and replay, a **fused
  single-launch quantize-and-stage hot path**, prequantized weight packs, and **persistent knob caching so
  there is zero in-engine autotuning** (**#4079**, **#4183**, **#4101**, **#4348**). A fault-tolerance rank
  mask masks peer timeouts during dispatch/combine.
- **SM12x W4A4** fixed for output quality — two NVFP4 quantization bugs, new `input_global_scale`
  parameter; W4A16 gained cooperative persistent launches, a **tensor-core decode path for small batches**,
  and recompile-free batch-size changes.
- **Unified MoE API** grew MXFP4 W4A8/W4A16, per-tensor routed FP8, **shared-expert fusion for FP4**, SiTU
  activation, and an unpacked pre-routed FP4 mode taking `topk_ids` / `topk_weights` as separate tensors.
- **Ulysses sequence parallelism** exposed publicly with a **fused NVLink-P2P kernel that folds the layout
  permutation into the cross-GPU writes** (**#3820**, **#4240**).
- CuTe-DSL floor lowered to 4.5.2 for broader vLLM compatibility.

### The communication APIs — read these before touching our 19.6% collectives

[verified, [comm API docs](https://docs.flashinfer.ai/api/comm.html)]

- `trtllm_allreduce_fusion` — allreduce fused with RMSNorm and optional quantization.
- `trtllm_moe_allreduce_fusion` — MoE-specific, with residual and normalization.
- `trtllm_moe_finalize_allreduce_fusion` — the final MoE stage with permutation handling.
- `allreduce_fusion` — unified API over multiple backends.
- Fusion patterns: **residual add + RMSNorm + FP8/NVFP4 quantization with per-token-group block scaling**,
  plus an optional weight-bias term (1.0 for Gemma/Qwen3.5-style norms, 0.0 for standard).
- **One-shot vs two-shot** is selected by internal heuristics on token count and hidden dimension, with an
  explicit `use_oneshot` override on some entry points. *This is precisely the "fuse the all-reduce into
  the RMSNorm using a one-shot algorithm below N bytes" pattern, already implemented and callable.*
- Workspaces: `create_allreduce_fusion_workspace()`, `TRTLLMAllReduceFusionWorkspace`,
  `MNNVLAllReduceFusionWorkspace`.
- `UlyssesCommunicator` for head-scatter/sequence-gather with NVLink-P2P (world sizes 2, 4, 6, 8);
  fused-transpose NVLink all-to-all for attention reshaping.

### Attention and MoE surface

[verified, [attention](https://docs.flashinfer.ai/api/attention.html),
[fused_moe](https://docs.flashinfer.ai/api/fused_moe.html)]

Attention: `trtllm_batch_decode_with_kv_cache`, `trtllm_batch_context_with_kv_cache`,
`cudnn_batch_decode_with_kv_cache`, `xqa_batch_decode_with_kv_cache`,
`BatchDecodeWithPagedKVCacheWrapper` (FA2/FA3/TRT-LLM/CuTe-DSL backends),
`BatchDecodeMlaWithPagedKVCacheWrapper`, and a Blackwell-oriented task-scheduled family (`prims_ts`) for
fixed/packed-ragged context attention and native-CSR paged decode. Sparse MLA:
`batch_decode_sparse_mla_dsv4`, `convert_compressed_page_aligned_sparse_indices_to_hca_metadata`,
`DSV4HCAMetadata`. **The XQA family supports speculative decoding with draft-token masking, sliding
windows and FP8 quantization for both standard and MLA patterns** — that is directly the shape of our
EAGLE verification step.

MoE: `trtllm_fp4_block_scale_moe` / `_routed_moe`, `trtllm_fp8_block_scale_moe`,
`trtllm_fp8_per_tensor_scale_moe`, `trtllm_mxint4_block_scale_moe`, `trtllm_bf16_moe`, `cutlass_fused_moe`,
and CuTe-DSL variants (`cute_dsl_fused_moe_nvfp4`, `cute_dsl_fused_moe_mxfp8_mxfp4`, `b12x_fused_moe`) for
SM100+/SM120. Routing: `fused_topk_deepseek` (V3-style) and **`hash_topk` (DeepSeek-V4)**. Weight-layout
helpers `convert_to_block_layout`, `reorder_rows_for_gated_act_gemm`. Autotuning contexts are exposed.

**FlashInfer's blog is stale** — the most recent post is *FlashInfer-Bench* (2025-10-21); there is no
Blackwell blog post. All the current information lives in release notes and API docs.
[verified, [flashinfer.ai](https://flashinfer.ai/)]

---

## The KV-cache layer: LMCache, Mooncake, and vLLM-native offloading

**LMCache** [verified, [repo](https://github.com/LMCache/LMCache), [docs](https://docs.lmcache.ai/)]
runs as a **standalone daemon process, independent of the inference engine** — so KV survives an engine
crash ("no fate sharing"). Tiered across GPU / CPU RAM / local SSD / remote, with backends for Redis or
Valkey, Mooncake, InfiniStore, S3-compatible object storage, NIXL, and GPU Direct Storage. Two techniques
are genuinely novel:
- **CacheGen** — KV cache compression and streaming, so cache movement is bandwidth-cheap.
- **CacheBlend** — **reuse of cached KV blocks at *any* position in the prompt, not just as a prefix**,
  with selective token recomputation to recover quality. For agentic/RAG traffic where a retrieved chunk
  appears mid-prompt, this is the only published mechanism that gets a hit.

Integration is one flag [verified, [quickstart](https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html)]:
```
PYTHONHASHSEED=0 LMCACHE_CONFIG_FILE=lmcache_config.yaml \
vllm serve <model> --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```
`PYTHONHASHSEED=0` is required so that every process computes the same KV-cache block hash — a subtle
correctness trap worth noting if we build anything similar. Neither the README nor the docs pages I read
contain measured TTFT numbers with config, so I am not quoting any.

**[inferred] For us:** AA sends ~10k input tokens per request over a 72-hour P50 window. If AA reuses a
prompt corpus with common prefixes, a warm prefix cache moves TTFT (currently 189 ms) far more cheaply
than any kernel work. But AA's methodology as described does not guarantee prefix overlap, and TTFT is a
small share of a 1500-token generation. Priority: low for the leaderboard, high for cost-per-user on real
agentic traffic — where Mooncake's measured **1.7% → 92.2% hit rate** and **46× P50 TTFT reduction** on
real agentic traces is the strongest published evidence in the space.

---

## KTransformers

**What it uniquely does:** CPU-GPU heterogeneous MoE execution — **hot experts on GPU, cold experts on
CPU** — with Intel AMX and AVX512/AVX2 INT4/INT8 kernels and NUMA-aware memory management, plus GPU-side
GPTQ, FP8 per-channel, and hybrid IQ1_S/FP8 weight schemes. Integrates with SGLang for serving and
LLaMA-Factory for SFT. [verified, [repo](https://github.com/kvcache-ai/ktransformers)]

Published numbers: DeepSeek-R1-0528 FP8 on **8×L20 + Xeon Gold 6454S** — 227.85 tok/s total, 87.58 tok/s
output at 8-way concurrency. DeepSeek-V3 SFT at 3.7 it/s on 4× RTX 4090. Recent model coverage includes
DeepSeek-V4-Flash, GLM-5.2, MiniMax-M3, Kimi-K2.5, Qwen3-Next.

**Transferable to us: almost nothing directly** — we have 183 GB × 8 of HBM3e and no need to offload.
The one idea worth filing is **hot/cold expert classification itself**: if a small set of experts absorbs
a disproportionate share of routing on GLM-5.2, that same statistic can drive expert *placement* under
EP (which is what EPLB does) or an L2-resident hot-expert weight cache. **[inferred]**

---

## MLC-LLM

TVM/Relax-based compiler and deployment stack with a unified `MLCEngine` behind OpenAI-compatible APIs
across REST, Python, JS, iOS and Android; backends span CUDA, ROCm, Vulkan, Metal, WebGPU/WASM, OpenCL
(Adreno/Mali). Repository is active (1,805 commits on main, 23.1k stars).
**The README publishes no performance benchmarks or configurations.**
[verified, [repo](https://github.com/mlc-ai/mlc-llm)]

Honest assessment: MLC's value is portability across a dozen backends, not datacenter Blackwell latency.
The one architecturally interesting property — that the whole model is compiled through TensorIR rather
than assembled from hand-written kernels — is the same bet TileRT is making at tile granularity, and
TileRT is currently winning it. Nothing here to steal for 8×B200.

---

## LightLLM

[verified, [repo](https://github.com/ModelTC/LightLLM)]

What it uniquely does: **pure-Python, token-level KV cache management** (not block/page-level), a
multi-process design, and no-padding batching. Two research contributions have merit independent of the
engine:
- **Pre³** — deterministic-automaton-based constrained decoding, ACL 2025 outstanding paper. Relevant if
  we ever need structured output on the latency path.
- **Past-Future Scheduler** — request scheduling under SLA guarantees, ASPLOS '25.

They claim "fastest DeepSeek-R1 serving on a single H200 machine" in v1.0.0 but the README carries no
numbers. Their kernels are referenced by vLLM, SGLang, Aphrodite and Microsoft ParrotServe; academic
systems LoongServe (SOSP '24), S-LoRA (MLSys '24) and OmniKV (ICLR '25) build on it.
**Assessment: a kernel/algorithm source, not a serving competitor.**

---

## Text Generation Inference (TGI)

**TGI is archived.** As of **March 2026** the repository is read-only and in maintenance mode; the
maintainers accept only minor bug fixes and documentation, and explicitly recommend vLLM, SGLang,
llama.cpp or MLX for new deployments. [verified, [repo](https://github.com/huggingface/text-generation-inference)]

Its one enduring architectural idea is the **Rust router in front of a Python model server** — the same
"get the request path off Python" instinct that drove vLLM's API-server process split (33% of execution
time) and that TokenSpeed has now taken further with a C++ control plane. That idea is worth taking; the
codebase is not.

---

## llama.cpp and Ollama

**llama.cpp** [verified, [repo](https://github.com/ggml-org/llama.cpp)]: plain C/C++ with no dependencies,
1.5–8 bit integer quantization, 15+ backends (CUDA, HIP, Metal, Vulkan, SYCL…), CPU+GPU hybrid inference
for models exceeding VRAM. **The README does not document the graph-reuse, CUDA-graph, or speculative
decoding mechanics in enough detail to extract a transferable technique, and I did not find a separate
design document I could fetch.** Rather than reconstruct from memory, I am reporting this as
**nothing sourced beyond capability lists**. The general principle its community has demonstrated — that
at batch 1 the win is eliminating per-token CPU work and kernel launches, not adding FLOPs — is already
better documented by vLLM's v0.6.0 profile and by TileRT's persistent kernel, both cited above.

**Ollama** [verified, [repo](https://github.com/ollama/ollama)]: a local model runner built on llama.cpp
with a REST API. **No multi-GPU, distributed, or datacenter content in the README.** Nothing to steal.

---

## TokenSpeed / LightSeek Foundation — a new entrant worth watching

[verified, [repo](https://github.com/lightseekorg/tokenspeed),
[PyTorch blog](https://pytorch.org/blog/lightseek-tokenspeed-kernel/)]

Self-described as "a speed-of-light LLM inference engine designed for agentic workloads, with
TensorRT-LLM-level performance and vLLM-level usability." Four-layer architecture:
- **Modeling layer:** a static compiler with a **local-SPMD** design.
- **Scheduler:** a **C++ control plane** with an explicit finite-state-machine request lifecycle.
- **Kernels:** a pluggable, layered kernel system behind a portable public API.
- **Entrypoint:** AsyncLLM for low-overhead CPU-side handling.

Claims: 580 tok/s on Qwen3.5-397B-A17B for agentic workloads, and Pareto-optimal curves against
TensorRT-LLM. **[reported], no config disclosed in what I read — treat as marketing until reproduced.**
It advertises "one of the fastest MLA implementations" on Blackwell.

The **TokenSpeed-kernel** design is the genuinely transferable part, and it is the right answer to a
problem we will have as we add Kimi K3, Qwen3.8 and DeepSeek V4:

> The runtime calls platform-agnostic operator APIs (`mha_prefill`, `mha_decode_with_kvcache`,
> `moe_apply`). Backends self-register through `@register_kernel`, declaring operator family and mode,
> solution name, **platform capability requirements**, supported tensor-format signatures, feature
> traits, and priority. A selector filters by capability/format/trait, ranks candidates, and **caches the
> resolved callable**. Developers can force a solution for debugging. The same registry metadata drives
> the numerics-verification and standalone-benchmark CLIs, so a kernel is verified against a reference,
> benchmarked on real shapes, and then auto-selected — without touching model code.

Their AMD results validate the layering (Gluon kernels 1.4–2.3× over Triton on 14 of 15 GPT-OSS prefill
shapes; 1.6–3.6× end-to-end over the portable Triton path on MI355X), and the NVIDIA path uses
`trtllm` MHA and `flashinfer_trtllm` MXFP4 MoE behind the *same* API. The AMD kernel package was
published standalone as `tokenspeed-kernel-amd` and **adopted by vLLM in PR #46742**.

**[inferred] Steal the registry, not the engine.** A capability/trait-keyed kernel registry with a cached
resolved callable, plus CLI-driven numerics + benchmark harnesses reusing the same metadata, is maybe two
weeks of work and pays for itself the first time we bring up a new model or a new quantization format.

---

## PyTorch-side work that matters

### torch.compile for inference

Covered above under vLLM. The two structural facts: **Inductor will not find fusions across custom ops**
(attention, collectives, sub-byte quant) — those need hand-written passes — and **startup time is the
dominant complaint**, addressed by an on-disk artifact cache that is reusable across machines with
identical environments (`VLLM_FORCE_AOT_LOAD=1` to fail loudly on a miss).

### FlexAttention

[verified, [post](https://pytorch.org/blog/flexattention/)]

`score_mod(score, b, h, q_idx, kv_idx)` modifies attention scores before softmax; `mask_mod(b, h, q_idx,
kv_idx) -> bool` plus `create_block_mask` produces a `BlockMask` that lets the kernel **skip fully-masked
blocks and skip mask application on fully-computed blocks**. `torch.compile` lowers this to a single fused
FlashAttention-style Triton kernel; the backward is generated automatically.

Why `mask_mod` is separate from `score_mod`: applying masking to every computed element costs **15–20%
performance**, so the BlockMask records which blocks are "fully computed" (skip masking) vs "partially
computed" (apply masking). Performance: **90% of FlashAttention-2 forward, 85% backward on A100**.
`create_block_mask` is expensive — use `_compile=True` (an order-of-magnitude improvement in their
testing) or hand-write the BlockMask, whose metadata is just two tensors (`num_blocks`, `indices`).
BlockMask memory at BLOCK_SIZE=128 for a 1M sequence is ~60 MB; at 1024 it is under 1 MB.

**[inferred] Relevance to DSA:** our indexer produces exactly a per-query top-k block pattern. That is a
`BlockMask` with hand-written `num_blocks`/`indices` — no `create_block_mask` cost, no mask application on
fully-selected blocks. FlexAttention will not beat FlashMLA on the MLA-specific path, but it is a very
cheap way to prototype indexer variants (different `index_topk_freq`, different top-k schedules) without
writing a kernel per variant. PyTorch 2.13 added a deterministic backward (`compute_dq_write_order`,
<1% overhead) which matters for drafter training, not serving.

### Async TP and SymmetricMemory

[verified, [dev-discuss](https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798)]

```python
import torch.distributed._symmetric_memory as symm_mem
t   = symm_mem.empty(4096, device="cuda")
hdl = symm_mem.rendezvous(t, dist.group.WORLD)
peer_buf = hdl.get_buffer(next_rank, t.shape, t.dtype)
hdl.barrier(channel=0)          # CUDA-graph compatible
hdl.buffer_ptrs, hdl.multicast_ptr, hdl.signal_pad_ptrs   # for custom kernels
```

What it unlocks, all of it relevant to our 19.6% collectives and 47%-of-that arrival skew:
- **Peer memory access over NVLink** from SM instructions (`ld`/`st`/`atom`/`red`/`multimem`) and from the
  copy engine, with virtual addressing.
- **NVLS multicast and in-switch reduction** on NVSwitch V3+ — which is what NV18 all-to-all NVLink5 gives
  us — reducing link traffic for broadcast/reduce/all-gather/all-reduce.
- **Low-latency collectives you can own**: `one_shot_all_reduce`, `two_shot_all_reduce`,
  `multimem_all_reduce`, `multimem_all_gather`, in CUDA and Triton, reported competitive with or better
  than NCCL ring at small sizes.
- **Async TP two ways**: a host-API decomposition (LoopedCollectiveEinsum) that overlaps all-gather with
  matmul for medium/large problems, and **single-kernel async-TP** (CUTLASS and Triton variants) where a
  communication-aware matmul rasterizes output blocks as all-gather chunks arrive, signalled with
  `stream_write_value32` — this is the variant that wins at *small* problem sizes, i.e. batch 1.
- **Low-contention copy-engine AG/RS** that trades a little collective throughput to stop stealing SMs
  from the ambient matmul.

Known limitation: multi-node NVLink (GB200-class) support has reported issues with allocations from
overlapping devices across ranks. On a single 8×B200 node this does not apply to us.

**[inferred] This is the most direct published tool for the rank-arrival-skew problem.** A one-shot
allreduce written against SymmetricMemory with an explicit signal-pad barrier removes NCCL's own
scheduling and gives us a place to *measure* per-rank arrival directly (the signal pad is readable), which
NCCL does not. Combine with FlashInfer's `trtllm_allreduce_fusion` for the fused-norm epilogue.

### Fusing normalization into GEMM and attention

[verified, [Towards Free Normalization](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/);
benchmarks bf16 on **B200** at a 750 W cap; code at `facebookresearch/ads_model_kernel_library`]

Three techniques, in increasing generality:

1. **Naive tile-stretch fusion** — force the GEMM's N tile to span the whole inner dimension. Works for
   small N: **17–32% of the LayerNorm kernel's latency saved at N=64/128**, then regresses badly past
   N=128 because the tiling deviates too far from the GEMM's optimum. Napkin math on Blackwell (228 KB
   SMEM, bf16, 2 pipeline stages, tile_m=tile_k=32) caps `tile_n` at 512.
2. **Lazy Pre-Norm** — the prologue fusion for **affine-free RMSNorm**. Key identity:
   `(A * rstd[:,None]) @ B == (A @ B) * rstd[:,None]`. Accumulate the row square-sum in the k-loop
   *in parallel with* the matmuls (it blocks nothing), then apply `rstd` as an epilogue. Pseudocode:
   ```
   for each k_tile:
       acc        += tile_A @ tile_B
       square_sum += (tile_A * tile_A).sum(-1)   # overlaps the MMA
   rstd = rsqrt(square_sum / K + eps)
   acc *= rstd[:, None]
   ```
   Limitations, stated: no elementwise affine (that is a *column*-wise multiply), no LayerNorm (the
   mean-subtraction is not a row-wise multiply), and a tricky backward because `rmsnorm(A)` is never
   materialized.
3. **Multi-CTA Norm** — the general epilogue fusion. Borrowed from Quack: partition large N across CTAs
   in a **CTA cluster** and reduce collaboratively through **distributed shared memory**, then drop that
   as the GEMM's epilogue. Costs: adjacent CTAs in a cluster must share `m_tile` and differ in `n_tile`,
   which kills **paired-CTA MMA** and **tile super-grouping**; N is still capped at
   512 × max cluster size = **4096 on Blackwell**.
4. **Fusion regrouping for backward** — forward epilogue fusion becomes backward *prologue* fusion, which
   is bad (norm-backward sits on the critical path each k-iteration and is redundantly recomputed). The
   fix: fuse the norm into a *different* linear in backward than in forward, so both directions are
   epilogue fusions and the backward kernel is structurally identical to the forward one.
5. **FlashNormAttention** — folds a LayerNorm, an RMSNorm and two residuals into a FlashAttention-variant
   kernel. **Up to 35% kernel speedup.** The optimization list is the interesting part for anyone writing
   a Blackwell megakernel: SMEM/TMEM buffer reuse (reuse the TMA store staging buffer for `q`);
   **keeping the residual addend in the TMEM accumulator so TensorCore does the add** via tcgen05 MMA
   semantics; register subtiling to avoid spills; a **fifth warp partition dedicated to the prologue
   LayerNorm** (8 warps for activation, 4 for LayerNorm, with asymmetric register allocation); register
   pre-loading of `q` to unblock the next iteration's TMA prefetch; and `TMA_REDUCE_ADD` in the backward
   to push a residual straight from SMEM to HBM — trading IO back for pipeline efficiency once IO is no
   longer the bottleneck.

Headline: **up to 90% of a normalization kernel's latency hidden by fusing with GEMMs.** They note norms
are ~10% of total latency in a typical (compute-bound) LLM.

### PyTorch Blackwell enablement

[verified, [2.13 release blog](https://pytorch.org/blog/pytorch-2-13-release-blog/), 8 Jul 2026]

- **CuTeDSL "Native DSL" backend for Inductor** — a second high-performance code path alongside Triton for
  **GEMM and RMSNorm**, using Quack-derived kernel overrides, with faster compilation. Kernel compilation
  moved from a thread pool to a **subprocess pool**, removing the GIL bottleneck (PRs #181267, #182108,
  #186310).
- FlexAttention deterministic backward on CUDA (`compute_dq_write_order`, +0.2% at S=32768).
- `torchcomms` — new communications backend for PyTorch Distributed with better fault tolerance and
  collective tracing; collectives renamed to `all_gather_single` / `reduce_scatter_single`.
- **`CUDAGraph.get_graph_data()`** — exposes node types, kernel names, dependency edges, and **IDs remapped
  to match CUPTI profiler output**. *This is a directly useful debugging tool for us: it lets us correlate
  a captured decode graph against a profile and find serialization inside the graph.* (PR #183165)
- **Experimental CUPTI monitor profiler** — collects GPU metrics asynchronously, completely off the GIL,
  removing the synchronization points that distort timing in the existing profiler. (PRs #186037, #186295)
- CUDA 13.0 is the default build; Triton pin 3.7.1; CUDA 12.8/12.9 builds removed.

---

## Third-party head-to-head benchmarks: InferenceMAX / InferenceX

### The methodology, and where it is and is not fair

[verified, [inferencex.semianalysis.com](https://inferencex.semianalysis.com/),
[repo](https://github.com/SemiAnalysisAI/InferenceX)]

InferenceMAX was renamed **InferenceX** and moved to `SemiAnalysisAI/InferenceX`. What makes it credible:
**every datapoint is produced by a public GitHub Actions workflow run**, recipes are committed as shell
scripts, full logs and artifacts stay viewable, weekly database snapshots ship as GitHub Releases, and
roughly 1,000+ new datapoints land per week on nightly runs. Frameworks: **vLLM, SGLang, TensorRT-LLM**.
Hardware: GB300/GB200 NVL72, B300, B200, H200, H100, MI355X/MI325X/MI300X, TPU v7 Ironwood. Models include
Kimi K3 (2.8T) / K2.5 / K2.7-Code, MiniMax M3/M2.5/M2.7, DeepSeek V4 Pro (1.6T), Qwen 3.5 (397B), GLM-5
and GLM-5.1. Methodology is Pareto-frontier over concurrency, not single points.

**Where the comparisons are unfair — be explicit about this internally.** The MiniMax-M2.5 study is the
clearest case [verified,
[post](https://inferencex.semianalysis.com/blog/b200-minimax-m2-5-vllm-nvfp4-vs-h100-fp8-perf-per-dollar)]:

| Lane | Parallelism search space explored |
|---|---|
| H100 vLLM FP8 | **TP=8 only** |
| B200 vLLM FP8 | TP=2 and TP=4 |
| B200 vLLM NVFP4 | **TP=1, 2, 4 and 8** |

The B200 NVFP4 frontier is drawn over four topologies and the H100 frontier over one. The headline "8.2×
better performance per dollar" therefore conflates hardware, precision, *and search-space breadth*. The
cost model is SemiAnalysis's own AI Cloud TCO ($1.30/GPU/hr H100, $1.95/GPU/hr B200), which is an
assumption, not a market price. They also note that their supplementary 1K/1K kernel comparison
"exaggerates the kernel gap relative to the 8K/1K" — a rare and welcome self-caveat.

Second structural issue: **the blog index contains no direct vLLM-vs-SGLang-vs-TRT-LLM head-to-head at
matched config.** Every post fixes model, GPU family, precision, workload, framework and serving mode and
compares a framework *against itself over time*, or compares hardware. SGLang dominates the AMD MI355X
lanes; vLLM and Dynamo+TRT-LLM dominate the NVIDIA lanes. **Anyone claiming "framework X beats framework
Y" from InferenceX is over-reading it.** The honest use of InferenceX is as a *floor*: it tells you what a
tuned public recipe achieves, which is the number our fork has to beat to be worth maintaining.

### The TileRT datapoint — the one that matters most to us

[verified, [Ultra-High Interactivity on NVIDIA GPUs? TileRT on InferenceX](https://inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia), 2026-08-10]

Config: **8× B200, single HGX node, GLM-5.1 744B, MXFP8, batch size 1.** Active-parameter traffic quoted
at ~21 GB/token against 64 TB/s aggregate HBM.

| Workload | TileRT (FP8, bs1) | Best conventional | Ratio |
|---|---|---|---|
| 1k/1k | **494.2 tok/s/user** | 256.3 (GB200 NVL72, Dynamo+TRT-LLM, FP4) | 1.9× |
| 1k/1k | 494.2 | 136.3 (B300, FP8) | 3.6× |
| **8k/1k** | **340 tok/s/user** | 181.4 (GB300 NVL72, Dynamo+TRT-LLM, NVFP4+MTP) | 1.9× |
| 1k/1k e2e decode tail | 3.01 s | 6.54 s (best NVFP4+MTP); 18.18 s (MI355X) | 4.5× |

Aggregate cost: at 8k/1k TileRT delivers **160.4 tok/s/GPU at batch 1** versus ~240 tok/s/GPU for the
conventional stack at concurrency 12. Going from 25 to 260 tok/s/user costs roughly **30× per-GPU
throughput**.

Mechanism, as described:
- **One persistent Engine Kernel.** The entire decode graph compiles into a single kernel that stays
  resident on the GPU for the whole decode lifecycle — no per-op launches, no inter-kernel barriers.
- **Warp specialization inside it**: separate warp groups for asynchronous data movement, tensor
  computation, and communication. "Where stages used to run serially as load → barrier → compute →
  barrier, they now overlap at tile granularity."
- **Model-specific GPU specialization**: for GLM-5.1, **GPU 0 acts as the sparse indexer doing Top-K
  selection while GPUs 1–7 execute MLA** — an asymmetric rank assignment that removes redundant
  synchronization between homogeneous ranks.
- **Communication runs inside the tile flow**: "an entire attention layer corresponds to a single kernel
  launch," replacing compute → sync → compute with continuously overlapping compute ↔ comm ↔ compute.

Costs and caveats they publish: batch size 1 only ("a single-passenger rocket ship"); model catalogue is
GLM-5/5.1 and DeepSeek-V3.2 only; static AOT compilation means hard-pinned dependencies and real
engineering effort per architecture; **no FP4 support yet**. Production users cited: Xiaomi MiMo V2.5 Pro
UltraSpeed and Z.ai GLM-5.1 HighSpeed.

**[inferred] Three conclusions for us.** (a) The 8k/1k number, 340, is the honest AA-shaped target, and we
are at 365 on real data — we may already be at or ahead of TileRT at AA input length, and should verify
that before spending months chasing "500". (b) Their GPU-0-as-indexer trick is directly applicable to our
DSA indexer at 5.8% and is *cheap* — it is a scheduling decision, not a kernel rewrite. (c) They are still
on FP8 while we have NVFP4; their per-token weight traffic is ~2× ours on the MoE path. If they ship FP4
the gap moves, so the FP4 advantage is a temporary moat, not a permanent one.

### SGLang, measured by the same harness

[verified, [SGLang 0.5.6 on B200](https://inferencex.semianalysis.com/blog/sglang-0-5-6-b200-deepseek-r1-fp4-up-to-1-8x)]
B200, DeepSeek-R1 NVFP4, TP4/EP4, ISL 8192 / OSL 1024, 16-GPU non-disaggregated pool,
`lmsysorg/sglang:v0.5.5-cu129-amd64` → `v0.5.6-cu129-amd64`:

| Concurrency | 0.5.5 tok/s/GPU | 0.5.6 tok/s/GPU | Gain |
|---|---|---|---|
| 4 | 508 | 907 | **1.79×** |
| 8 | 903 | 1,437 | 1.59× |
| 16 | 1,471 | 1,500 | 1.02× |
| 32 | 2,302 | 3,063 | 1.33× |
| 64 | 3,323 | 3,419 | 1.03× |
| 128 | 4,430 | 5,145 | 1.16× |

TPOT was unchanged; the gain came entirely from serving more concurrent users at the same per-user rate.
The changes credited: **piecewise CUDA graph extended to DeepSeek-V3 and the MLA attention path**, a
**unified event loop across PD-disaggregated / overlap / DP-attention serving modes**, **JIT kernels for
lower startup cost and runtime shape specialization**, FP4 support in the MHA/MLA KV caches, re-enabling
the FlashInfer TRT-LLM-GEN MHA path, and bumping FlashInfer to 0.5.2.
**Note the shape of that gain — biggest at concurrency 4–8, i.e. the low-concurrency regime we care about,
and driven by CUDA-graph coverage plus JIT specialization, not by kernels.**

SGLang's own DeepSeek-V4 GB300 writeup [verified,
[PyTorch blog](https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/)]
is the best-documented "day-0 to production" post in the ecosystem and is worth reading against our own
GLM-5.2 process. Two items in it are relevant even though it is a competitor's post:
- **A one-line correctness fix moved acceptance from 0.57 to 0.70** — PR #25733 converted an `fp8_einsum`
  input scale to `ue8m0` to fix a Blackwell NaN, and the MTP path recovered as a side effect. Acceptance
  rate is a numerics canary, not just a model-quality metric.
- **Token-bucket prewarm** (PR #25810): pre-warm the MHC token-count buckets the model actually visits so
  the first real requests don't pay lazy compile cost. "A serving system that already knows its hot shapes
  should not spend the critical path rediscovering them at runtime."

---

## Techniques ranked by transferability to our stack

| Technique | Source | Mechanism in one line | Attacks | Effect (as published) | Difficulty | Confidence |
|---|---|---|---|---|---|---|
| DSA index sharing across MTP/draft steps | vLLM GLM-5.2 | Reuse indexer top-K inside the speculative iteration | DSA indexer 5.8% × draft steps | not separately quantified; PR #45895 raised acceptance ~3→~4 | Low–Med | High |
| Pre-attention megafusion (norm+RoPE+quant+KV write) | vLLM DSv3.2 | ~33 → ~10 launches/layer, 2 fused kernels | attention 10.9%, launch overhead, skew | **1.28× at bs=1** | Med–High | High |
| AllReduce+RMSNorm(+quant) fusion, one-shot below threshold | FlashInfer + vLLM | `trtllm_allreduce_fusion`, `use_oneshot` | collectives 19.6% | up to **15%** (AR+RMSNorm) | Med | High |
| One-shot/multimem allreduce on SymmetricMemory | PyTorch | own the collective; signal pads give per-rank arrival visibility | rank arrival skew (47% of 19.6%) | competitive with/better than NCCL ring at small sizes | Med–High | Med |
| Speculative padding at P/D handoff | vLLM GLM-5.2 | pad spec dim so decode never sees a mixed batch | TPOT | **−18 ms** | Med | High |
| GPU-native input prep + zero CPU↔GPU sync | vLLM MRV2 | Triton kernels build all per-step tensors on device | CPU overhead at C1 | **−6.3% / −11% TPOT** | High | High |
| Multi-stream indexer/compression/insert overlap | vLLM DSv4 | 3 pipelines on 3 CUDA streams | low-batch latency | **−5–6% e2e at low batch** | Med | High |
| Adaptive verification budget | vLLM DSpark | confidence head → survival product → global top-B | wasted verify FLOPs in 3-1-4 tree | Pareto-optimal C1–C256 | Med–High | Med |
| EAGLE 3.1 FC-norm + post-norm feedback | vLLM/TorchSpec | drafter behaves as recursive invocation | acceptance length | **2.03× at C1** on Kimi K2.6 | Med (retrain) | Med |
| Parallel drafting (P-EAGLE) | vLLM/Amazon | K draft tokens in one forward via learned mask embeddings | draft latency at C1 | **1.55–1.69× at C1**, collapses at C64 | High (retrain) | Med |
| Lazy Pre-Norm into GEMM prologue | PyTorch/Meta | `(A*rstd)@B == (A@B)*rstd` | dense GEMM 37.1% | up to **90%** of norm latency hidden | Med | Med |
| Multi-CTA norm epilogue (clusters + DSMEM) | PyTorch/Meta | cluster-collaborative reduction in the GEMM epilogue | dense GEMM 37.1% | (charted, not tabulated) | High | Med |
| GPU-role specialization (one rank = indexer) | TileRT via InferenceX | asymmetric rank assignment removes redundant sync | skew + indexer | part of a 1.9× stack result | Low–Med | Med |
| Persistent decode megakernel | TileRT via InferenceX | whole decode graph as one resident kernel, tile-level comm overlap | everything at C1 | **494 @1k/1k, 340 @8k/1k**, bs1 only | Very High | High |
| NVFP4 MoE dispatch | vLLM WideEP | quantize activations before all-to-all | collectives volume | **4× less A2A volume** | Med | High |
| `flashinfer_nvlink_two_sided` A2A | vLLM GLM-5.2 | swap the all-to-all backend | collectives | **−4% TPOT** | Low | High |
| FULL_DECODE_ONLY graphs + dynamic spec lengths | vLLM | full graphs only for uniform decode batches | launch overhead | enables full coverage on latency path | Low–Med | High |
| Async scheduling | vLLM | decouple CPU scheduling from GPU execution | CPU overhead | **~10%**; "key feature behind 25K TPS" | Med | High |
| Stream interval / token buffering before dispatch | vLLM | buffer N tokens before network write, keep TTFT | frontend overhead | **+57% e2e** at high concurrency | Low | High |
| NUMA binding of workers | vLLM | `--numa-bind-nodes 0 0 1 1` | CPU/GPU affinity; we have 2 NUMA nodes | not quantified | **Very Low** | High |
| Kernel registry with capability/trait selection | TokenSpeed | `@register_kernel` + selector + cached callable + shared bench/numerics CLIs | maintainability across K3/Q3.8/V4 | AMD path 1.6–3.6× e2e via swap-in | Med | High |
| DCP + `VLLM_DCP_Q_REPLICATE` | vLLM | shard KV on sequence dim; replicate q-proj at load | cost/user at C16–C64 | 1,863 → **6,091 tok/s/GPU** | High | High |
| CacheBlend non-prefix KV reuse | LMCache | reuse KV blocks at any prompt position + selective recompute | TTFT on agentic traffic | no numbers published | High | Low |
| Token-bucket prewarm of hot shapes | SGLang DSv4 | pre-warm the compile buckets actually visited | first-request latency | qualitative | Low | High |
| Nightly Pareto perf CI + accuracy gates | vLLM perf-eval | 17 recipes, `vllm-bench` + `lm-eval` + BFCL, ClickHouse dashboards | trust in our own numbers | — | Low | High |

---

## What vLLM does better than SGLang today, and what to port into our fork

Stated carefully: SGLang leads on several axes this document does not cover (it wins most AMD lanes on
InferenceX, and its DeepSeek-V4 GB300 work is excellent). The list below is only about capabilities where
**vLLM has published a mechanism SGLang has not**, and where the mechanism is portable.

1. **Model Runner V2's execution core.** `vllm/v1/worker/gpu/`. Port in this order:
   `input_batch.py` + `states.py` (decoupled persistent batch: stable row per request, gather to build
   per-step tensors) → the Triton input-prep kernels (`input_ids`, `positions`, `query_start_loc`,
   `seq_lens` built on device) → `sample/` (Gumbel-Max sampler, top-k logprobs by finding top-k logits
   first) → `async_utils.py` (outputs on a separate stream). The stated target is **zero CPU↔GPU
   synchronization**, with prep kernels consuming rejection-sampling results directly. Evidence: −6.3%
   TPOT on GLM-4.7-FP8/4×GB200/MTP=1, −11% on GLM-5.2/B300.
   ([MRV2](https://vllm.ai/blog/2026-03-24-mrv2))

2. **The torch.compile custom-pass catalogue as a first-class subsystem.** SGLang has piecewise CUDA
   graphs and JIT kernels; vLLM has a *named, individually-measured* pass registry: AllReduce+RMSNorm
   (**PR #20691**, up to 15%), RMSNorm+Quant, SiLU-Mul+Quant (up to 8%), Attention+Quant FP8 (up to 7%),
   Sequence Parallelism & Async TP (up to 10%), Pad+Quant, Finalize+Slice (**PR #30647**, ~6%), plus
   no-op elimination and fix-functionalization. Port the *registry pattern* (a pass is a named,
   independently benchmarkable object) before porting individual passes.
   ([torch.compile post](https://vllm.ai/blog/2025-08-20-torch-compile))

3. **`KVConnectorBase_V1` with claim-filtering and `MultiConnector` composition.** The TileRT integration
   proves the API is strong enough that a *completely different runtime* can take over decode without
   patching the host engine: connector claims only requests the router marked, is a strict no-op
   otherwise, and composes with the normal decode pool under `MultiConnector`. If we want a
   persistent-kernel decode path for C1 traffic, this is the seam to build it behind.
   ([vLLM × TileRT](https://vllm.ai/blog/2026-07-14-vllm-tilert-pd),
   [connector docs](https://docs.vllm.ai/en/latest/features/disagg_prefill.html))

4. **The `speculators` library and its checkpoint format.** A standardized HF-compatible
   `speculator_config`, training for single- and multi-layer drafters, conversion tooling for external
   research checkpoints, and one-command serving. Four algorithms behind one interface: EAGLE-3,
   P-EAGLE (`"parallel_drafting": true`, **PR #32887**, v0.16.0+), DFlash, DSpark.
   ([repo](https://github.com/vllm-project/speculators))

5. **Adaptive verification** — `enable_adaptive_verification` (**PR #47808**). Nothing equivalent is
   published on the SGLang side. Given our 3-1-4 tree and the published acceptance decay curve (<10%
   survival at position 7 vs >70% at position 1), a per-request budget is very likely a real win at C1
   and a clear one at C16+.
   ([DSpark post](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification))

6. **The CUDA-graph mode taxonomy with backend capability declarations.** `AttentionCGSupport` ∈ {ALWAYS,
   UNIFORM_BATCH, UNIFORM_SINGLE_TOKEN_DECODE, NEVER} with automatic mode downgrade, and the semantic
   that a speculative-decode batch counts as "uniform decode" when `query_len == 1 + num_spec_tokens`.
   This is the cleanest published framing of the graph/attention compatibility problem and it directly
   determines whether our EAGLE verification step runs under a full graph.
   ([design/cuda_graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html))

7. **The MoE modular-kernel contract.** A prepare/finalize stage and an experts stage with an explicit
   activation-format contract (standard vs batched), so `{DeepGemm, Cutlass FP4/FP8, TRTLlm MxFP4/NvFP4,
   FlashInfer, Marlin, Triton}` × `{deepep_high_throughput, deepep_low_latency, allgather_reducescatter,
   flashinfer_nvlink_*}` compose without N×M code. With 256 experts / 8 active and both NVFP4 and FP8
   builds, we will hit this combinatorial problem.
   ([moe_kernel_features](https://docs.vllm.ai/en/latest/design/moe_kernel_features.html))

8. **Elastic EP's standby communication groups and two-stage barrier.** Even without adopting elastic
   scaling, `StatelessGroupCoordinator` (build the new comm group while the old one still serves) and the
   timeout-then-untimed two-stage barrier are the right primitives for any online topology or EPLB change.
   No perf numbers published — port the pattern, not the promise.
   ([Elastic EP](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism))

9. **`perf-eval` as a process.** 17 model×hardware recipes nightly; `vllm-bench` for TTFT/TPOT;
   `lm-eval` for GSM8K/GPQA/AIME; BFCL for tool calling; ClickHouse-backed dashboards refreshed every
   15 minutes at ci.vllm.ai; three release gates, all of which must pass; a bot that nominates the culprit
   commit and is right ~70% of the time. Cheapest high-value item on this list.
   ([Keeping vLLM Production Quality](https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality))

10. **Operational hygiene we can copy in an afternoon**: `--numa-bind` (we have 2 NUMA nodes and no
    published evidence we've checked this), `--stream-interval` / `--api-server-count` for frontend
    overhead, `VLLM_FORCE_AOT_LOAD=1` so a compile-cache miss fails loudly rather than silently costing a
    minute, `--kv-cache-memory` to skip memory profiling on restart, and the `2 + N` physical-core
    minimum with a busy-loop scheduler that starves badly if a core is stolen.
    ([optimization docs](https://docs.vllm.ai/en/latest/configuration/optimization.html))

**Where vLLM is behind and we should not copy it:** its single-stream latency is not competitive
(230 tok/s/user on AA for DeepSeek-V3.2 vs TileRT's 340 at 8k/1k on GLM-5.1), most of its recent
headline work optimizes aggregate throughput at high concurrency (25K TPS/GPU at concurrency 64–5120,
with concurrency 1–32 explicitly unmeasured), and its own roadmap for GLM-5.2 lists **PDL, persistent
kernels, hierarchical all2all** as *future* work — i.e. it is where we and TileRT already are.

---

## Sources

All URLs below were fetched and read. Two web searches were used to seed discovery; everything cited is
from a fetched page.

**vLLM blog**
- https://vllm.ai/blog (index)
- https://vllm.ai/blog/2024-09-05-perf-update
- https://vllm.ai/blog/2025-01-27-v1-alpha-release
- https://vllm.ai/blog/2025-08-20-torch-compile
- https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- https://vllm.ai/blog/2025-09-29-deepseek-v3-2
- https://vllm.ai/blog/2025-10-09-blackwell-inferencemax
- https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference
- https://vllm.ai/blog/2025-12-17-large-scale-serving
- https://vllm.ai/blog/2026-01-08-kv-offloading-connector
- https://vllm.ai/blog/2026-02-01-gpt-oss-optimizations
- https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1
- https://vllm.ai/blog/2026-02-13-gb300-deepseek
- https://vllm.ai/blog/2026-03-13-p-eagle
- https://vllm.ai/blog/2026-03-24-mrv2
- https://vllm.ai/blog/2026-04-24-deepseek-v4
- https://vllm.ai/blog/2026-05-06-mooncake-store
- https://vllm.ai/blog/2026-05-11-vllm-tops-artificial-analysis
- https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism
- https://vllm.ai/blog/2026-05-26-eagle-3-1
- https://vllm.ai/blog/2026-07-14-vllm-tilert-pd
- https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality
- https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd
- https://vllm.ai/blog/2026-07-23-vllm-afd-plugin
- https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting
- https://vllm.ai/blog/2026-08-06-qwen35-25k-tps
- https://vllm.ai/blog/2026-08-07-decode-context-parallelism
- https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification

**vLLM docs and repos**
- https://docs.vllm.ai/en/latest/configuration/optimization.html
- https://docs.vllm.ai/en/latest/design/cuda_graphs.html
- https://docs.vllm.ai/en/latest/design/moe_kernel_features.html
- https://docs.vllm.ai/en/latest/features/disagg_prefill.html
- https://docs.vllm.ai/en/latest/features/speculative_decoding/
- https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html
- https://github.com/vllm-project/vllm/releases
- https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker
- https://github.com/vllm-project/vllm/tree/main/vllm/v1/worker/gpu
- https://github.com/vllm-project/production-stack
- https://github.com/vllm-project/speculators

**FlashInfer**
- https://github.com/flashinfer-ai/flashinfer
- https://github.com/flashinfer-ai/flashinfer/releases
- https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17
- https://flashinfer.ai/
- https://docs.flashinfer.ai/api/comm.html
- https://docs.flashinfer.ai/api/attention.html
- https://docs.flashinfer.ai/api/fused_moe.html

**KV cache layer**
- https://github.com/LMCache/LMCache
- https://docs.lmcache.ai/
- https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html

**Other engines**
- https://github.com/kvcache-ai/ktransformers
- https://github.com/mlc-ai/mlc-llm
- https://github.com/ModelTC/LightLLM
- https://github.com/huggingface/text-generation-inference
- https://github.com/ggml-org/llama.cpp
- https://github.com/ollama/ollama
- https://github.com/lightseekorg/tokenspeed
- https://github.com/kserve/kserve

**Benchmarks**
- https://inferencex.semianalysis.com/
- https://inferencex.semianalysis.com/blog
- https://inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia
- https://inferencex.semianalysis.com/blog/b200-minimax-m2-5-vllm-nvfp4-vs-h100-fp8-perf-per-dollar
- https://inferencex.semianalysis.com/blog/sglang-0-5-6-b200-deepseek-r1-fp4-up-to-1-8x
- https://github.com/SemiAnalysisAI/InferenceX

**PyTorch**
- https://pytorch.org/blog/ (index)
- https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/
- https://pytorch.org/blog/lightseek-tokenspeed-kernel/
- https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/
- https://pytorch.org/blog/flexattention/
- https://pytorch.org/blog/pytorch-2-13-release-blog/
- https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798

### Could not source

- **Ray Serve LLM architecture.** `docs.ray.io/en/latest/serve/llm/overview.html` and
  `.../serving-llms.html` both returned HTTP 404; `github.com/ray-project/ray/tree/master/python/ray/llm`
  returned a directory listing with no README. No claims made.
- **KServe `LLMInferenceService` CRD, disaggregated P/D, KV-aware routing details.** The repo README lists
  vLLM/llm-d support, OpenAI-compatible protocol, KV offload to CPU/disk and request-based autoscaling,
  but no CRD or version specifics. `kserve.github.io/.../huggingface/` returned empty content.
- **llama.cpp graph-reuse / CUDA-graph / speculative-decoding mechanics.** Not documented in the README
  and I did not locate a fetchable design doc. Reported as nothing sourced rather than reconstructed.
- **A vLLM-published NVFP4-vs-FP8 accuracy comparison for GLM-5.2.** Only NVFP4 scores were published.
- **A first-party, matched-config vLLM-vs-SGLang-vs-TensorRT-LLM head-to-head.** InferenceX compares each
  framework against itself over time or compares hardware; it does not publish a same-config three-way.
- **FlashInfer allreduce-fusion performance numbers.** The API and fusion patterns are documented; I found
  no published latency table for the fused vs unfused path.
