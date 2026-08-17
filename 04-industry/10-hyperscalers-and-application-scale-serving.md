# Hyperscalers and application-scale serving: Perplexity, Databricks, Character, Anthropic-scale lessons

## What this is

A mine of published engineering work from companies that serve LLMs at *application* scale —
where the constraint is not "make one GPU fast" but "keep P50 latency at a leaderboard number
while running the fleet hot enough to be profitable." That is exactly our tension: 365 tok/s
single-stream and 40.8k tok/s aggregate at C64, from the same 8×B200.

Coverage and honesty notes:

- Every claim is labelled `[verified]` (I fetched the URL given and the content below came back
  from it), `[reported]` (the company asserts it, not independently reproduced),
  `[inferred]` (my reasoning), or `[unverified]` (could not source).
- Two sources I read *end to end as raw text* rather than through a summariser: the Perplexity
  `fabric-lib` MLSys'26 paper (arXiv 2510.27656v2, 17 pages) and the Sarathi-Serve OSDI'24 paper
  (arXiv 2403.02310). Numbers from those are quoted from the actual tables. Everything else came
  through a fetch-and-extract pipeline; where exact wording is load-bearing I quoted it.
- Perplexity's `perplexity.ai/hub/blog` returns HTTP 403 to direct fetches. I read those posts
  through the `r.jina.ai` text proxy. URLs given are the canonical ones.
- **Vendor benchmark numbers are marketing until independently reproduced.** Where a number has a
  config asymmetry (batch size, ISL/OSL, dtype, speculative decoding on/off, disaggregated or not)
  I say so inline. Several headline numbers in this file are *not* comparable to each other.

Companies where I found **nothing substantive beyond marketing**, stated plainly rather than padded:

- **Databricks on *how* they lead the GLM-5.2 board.** They lead it (verified on Artificial
  Analysis), Matei Zaharia and the team have posted about it on LinkedIn, and a third-party Medium
  write-up lists techniques — but Databricks has published **no engineering post on the GLM-5.2
  optimisation itself**. Their real technical posts are about *reliability and capacity*, which
  turn out to be more useful to us anyway. See §2.
- **Microsoft/Azure on B200/GB200 serving guidance.** I could not locate an Azure engineering post
  with Blackwell inference configuration or numbers. Microsoft's transferable work is Splitwise,
  DeepSpeed-FastGen and Sarathi-Serve — all pre-Blackwell but structurally sound.
- **OpenAI on serving architecture.** Nothing disclosed beyond API-surface behaviour (caching,
  batch, flex, priority). `openai.com/index/scaling-chatgpt/` is a 404. Everything else circulating
  is speculation and I have excluded it.
- **Anthropic on serving architecture.** The engineering blog has no inference-architecture post.
  What *is* disclosed comes from two postmortems and the API docs, and it is genuinely informative
  (multi-platform serving, sticky routing, a precision bug). See §10.

---

## Bottom line for our system

Ranked by expected effect on our two objectives, with difficulty. "AA" = the Artificial Analysis
leaderboard run we are chasing (~10k in, ≥1500 out, single-stream and 10-parallel, P50 over 72h,
measured over the public internet from GCP us-central1-a).

| # | Steal | Mechanism | Expected effect | Difficulty | Source |
|---|-------|-----------|-----------------|------------|--------|
| 1 | **Kill the network and LB hops before touching kernels** | Meta measured 75 ms CA→NY RTT *plus another ~75 ms* from "naive host selection, health check, load balancing" on top of model time | AA measures from GCP us-central1-a over the public internet. Our TTFT is 189 ms. If even 50 ms of that is placement/LB, it is the single cheapest win on the board and it costs zero GPU work | Low | Meta (§7) |
| 2 | **Do not use tree-structured EAGLE drafts** | Perplexity: "custom attention masks for a whole sequence significantly slows down attention by up to 50%, nullifying some speedup." They abandoned tree traversal for single-token prediction | We run EAGLE 3-1-4 (a tree). Attention is 10.9% at C1; a 50% attention regression on the verify pass could be eating most of the tree's acceptance gain. Test 3-1-1 vs 3-1-4 with the mask cost isolated | Low to test, medium to re-tune | Perplexity (§1) |
| 3 | **Share the DSA top-k index across draft steps** | vLLM's `IndexerCache` + `index_share_for_mtp_iteration` reuses the Top-K sparse indices produced by the GLM-5.2 DSA indexer across MTP draft iterations; LongBench V2 64.01 says accuracy holds | DSA indexer is 5.8% of C1. With a 3-1-4 spec tree the indexer runs repeatedly per accepted token. Direct removal of redundant indexer work; combines with our `index_topk_freq=4` | Medium | vLLM/GLM-5.2 (§11) |
| 4 | **Speculative padding on decode** | vLLM pad new requests to match the existing batch's speculative shape, so a joining request does not force a mixed-batch fallback. Reported the single largest win in their GLM-5.2 run: TPOT ~40 ms → ~22 ms | Our per-stream speed falls 4.7× from C1 to C16 — mixed-batch fallback on request arrival is a prime suspect for part of that cliff | Medium | vLLM/GLM-5.2 (§11) |
| 5 | **Signal the host/NIC *before* issuing NVLink writes, behind a grid barrier** | Perplexity: NVLink stores are fire-and-forget until a memory barrier, and the barrier's scope includes the host — so a host-ordering barrier gets slowed by previously issued NVLink writes. They reorder to signal first, then grid-barrier, then NVLink-write. "increases the total execution time of send kernels, but reduces latency on the critical path" | 47% of our collective time is rank arrival skew. This is a concrete, published pattern for shaving the *critical path* of a communication kernel at the cost of its total time — exactly the C1 trade we want | Medium-high (kernel work) | Perplexity (§1) |
| 6 | **Token-budget selection derived from the TBT SLO, and tile-aligned** | Sarathi-Serve: pick the forward-pass token budget by one-time profiling as "the maximum number of tokens that can be packed in a batch without violating TBT SLO"; and beware tile quantization — "chunk size of 257 can increase prefill time by 32% compared to chunk size 256" | Dense GEMM is 37.1% of C1. Any padding boundary we cross (spec-decode draft lengths, chunk sizes, MoE group sizes) can cost double-digit percentages. Audit every shape against SM100 tile sizes | Low (audit) to medium (retune) | Microsoft (§5) |
| 7 | **Overlap is worthless at low batch — stop trying, reduce latency instead** | Perplexity Table 7: dual-batch overlap gives +18% at batch 128 but *degrades* performance at batch 32 (32.0 → 30.2 tok/s). "For latency-sensitive workloads with small batches, computation and communication are sequential due to data dependencies, making low communication latency essential" | Settles a design argument for C1: our 19.6% collectives cannot be hidden, only shortened. Invest in one-shot/low-latency collectives, not in micro-batch pipelining, for the single-stream objective | Low (a decision) | Perplexity (§1) |
| 8 | **One-shot direct-access allreduce (DDA-flat) for the latency path** | Meta's Direct Data Access allreduce reduces latency from O(N) to O(1) by letting each rank load peer memory directly; DDA-tree does reduce-scatter + all-gather with direct access. 10–50% faster than RCCL for decode, ~10% TTIT reduction | Allreduce is "up to 30% of latency" in Meta's measurement; ours is 19.6% of C1. On NV18 all-to-all NVLink5 a one-shot algorithm below a byte threshold is the standard answer and Meta has published that it works | Medium | Meta (§7) |
| 9 | **Prefix-cache-aware + load-aware routing, not either alone** | llm-d: prefix-only scoring collapsed at high QPS on a high-sharing workload (≈55% success, TTFT >140 s); prefix **+** load scoring held 100% success, ITL ~30 ms vs ~160 ms, throughput scaling to ~60k tok/s instead of plateauing at 2–3k | For the C64 cost objective and for AA's 72-hour P50 window (10k-token prompts repeated across runs share prefixes). Cache-affinity routing without a load term is a documented way to make things *worse* | Medium | Google/llm-d (§8) |
| 10 | **Stepped prefix truncation to protect cache hits** | Character.AI's Prompt Poet keeps the truncation boundary *fixed* and moves it only every k turns (`truncation_step`, e.g. 4000 tokens), deliberately discarding more tokens than necessary so the prefix hash stays stable | Turns a variable-prefix workload into a cacheable one. Applies to any agentic/multi-turn traffic we serve, and to how we construct the AA-style 10k prompt | Low | Character.AI (§3) |
| 11 | **Capacity accounting in "model units", routing and autoscaling on them** | Databricks models request cost as a multi-dimensional function of input tokens, output tokens and prefill/decode character, with coefficients from automated per-model per-hardware benchmarking; Dicer routes on *server load in model units*, and the autoscaler triggers on model-unit utilisation. "over 80% GPU savings compared to static provisioning at peak" | Direct answer to "cost per user at C64 without destroying C1." Request-count routing is provably wrong when one 10k-token request costs 100× another | Medium-high | Databricks (§2) |
| 12 | **Warp-specialised dequant fused into attention + query-head packing** | Character.AI use "producer/consumer warp-specialization to effectively overlap dequantization with the MMA instructions", and pack multiple query heads into one threadblock so the 64-row WGMMA warpgroup tile is not mostly empty during decode. Measured 1,337.08 µs → 51.90 µs decode at ctx=131072, batch=2 with splitKV + packQ | We run NVFP4 and FP8 builds on SM100 with DSA-MLA. Decode has few query rows per head; the tile-underfill problem is identical and the fix is published | High (kernel work) | Character.AI (§3) |
| 13 | **Hierarchical KV cache: HBM → DRAM → flash** | Meta tiers by access frequency: HBM for shared system prompts, DRAM for active chat history (touched every minute), flash for cold users. "over 50% of reduction for both latency and capacity" | For the cost objective. Note AWS's threshold (below ~1000 tokens the remote fetch costs as much as recomputing prefill) — tier by prompt length, not just by recency | Medium | Meta (§7), AWS (§6) |
| 14 | **Health checks at the highest scheduling priority** | Databricks: giving black-box health checks top scheduling priority took "false liveness probe failures ... from several per week to zero" | Cheap operational hygiene. Under load, a busy inference process fails its own liveness probe and gets killed — a self-inflicted capacity loss at exactly the moment you need capacity | Low | Databricks (§2) |
| 15 | **Measure the way the board measures, and instrument acceptance rate as a first-class metric** | Anyscale: include TTFT inside ITL so late-streaming systems are penalised; use real text, not random tokens (random tokens hide speculative decoding); vary output lengths. vLLM: monitor "MTP acceptance rate and mean accepted length" alongside per-pool TTFT/TPOT percentiles | AA takes P50 over 72 hours. A regression in acceptance rate on real traffic will not show in a synthetic bench but will move the board number | Low | Anyscale (§4), vLLM (§11) |

Two warnings that cut the other way:

- **Disaggregation costs ~50–100 ms of TTFT.** Perplexity measured "~100 ms TTFT penalty"; Meta says
  "another 50 to 100 milliseconds to your TTFT if you are doing disagg". At our TTFT of 189 ms and
  a single-node 8×B200, PD disaggregation is very likely a **net loss for the single-stream objective**
  and only a win at C10/C64. Do not let a throughput-motivated architecture change wreck the
  leaderboard run.
- **The AA TTFT column is not our TTFT.** AA's TTFT "accounts for reasoning model 'thinking' time" —
  Databricks shows 6.74 s for GLM-5.2 (max). Our 189 ms is prefill-to-first-token. Do not compare them.

---

## 1. Perplexity AI

By a wide margin the most technically substantive publisher in this cohort. They serve their own
Sonar models plus large open MoEs (DeepSeek-V3/R1, Qwen3-Coder 480B, Kimi-K2 class) at web-search
scale, and they publish mechanisms with numbers.

### What they run

- H100 and H200 clusters; **AWS p5/p5en with EFA** as well as ConnectX-7/InfiniBand estates
  `[verified]`. Their portability work exists because they run on both.
- Disaggregated prefill/decode in production, with a global scheduler that picks a prefiller and a
  decoder per request `[verified]`.
- Large-EP MoE serving (EP8 → EP128), NVSHMEM- and now RDMA-based dispatch/combine `[verified]`.
- Triton Inference Server + TensorRT-LLM for the classic dense-model fleet, Kubernetes pods, an
  in-house front-end scheduler `[verified, via NVIDIA]`.

### Techniques

**1.1 `fabric-lib` / TransferEngine — portable one-sided RDMA with `IMMCOUNTER`**
`[verified]` https://arxiv.org/abs/2510.27656 (MLSys'26), code at
https://github.com/perplexityai/pplx-garden

Mechanism. Collectives (NCCL, `torch.distributed`) are unusable for disaggregation and MoE routing
because of four properties they name explicitly: fixed membership, synchronised initialisation,
operation ordering, and shape uniformity. Their insight is that **ConnectX RC and AWS EFA SRD share
a common substrate: reliable-but-unordered delivery** (RC can simply ignore ordering; SRD is
inherently unordered). So they build everything on one-sided `WRITEIMM` plus a novel `IMMCOUNTER`:
per-immediate-value counters incremented from completion-queue polling, delivered via callback,
atomic flag, or GDRCopy to the GPU. No operation in the library has ordering guarantees.

Correctness argument worth stealing verbatim: `IMMCOUNTER` is safe under unordered transport because
the RDMA spec requires the `WRITEIMM` payload to be issued before the immediate value, and the PCIe
switch orders writes to the same device — so once the CPU observes the count, any subsequent
CPU→GPU transaction is ordered after the NIC→GPU data writes.

Implementation details that matter:

- One worker thread per DOMAINGROUP, **pinned to a core on the NUMA node the devices hang off**;
  domain-specific data structures allocated *after* pinning so memory lands in the right NUMA node.
  One worker handles up to 4 DOMAINs (one NIC each); a separate dedicated thread polls the GPU for
  UVM watchers. Lock-free queues between them. (We have 2 NUMA nodes on our box — same problem.)
- ConnectX path: **two RC queue pairs per peer** — one for `SEND`/`RECV`, one for `WRITE`/`WRITEIMM`
  — because both `RECV` and `WRITEIMM` completions consume work requests in posting order and would
  otherwise interfere. Plus **work-request chaining** (up to 4 WRs linked through
  `ibv_send_wr.next`, cutting doorbell rings) and `IBV_ACCESS_RELAXED_ORDERING` to permit
  out-of-order PCIe transactions between NIC and GPU memory.
- EFA path: work-request templating (pre-populate and retain common `libfabric` descriptor fields),
  and they must supply valid descriptors even for immediate-only zero-sized writes because EFA
  diverges from the spec there.
- `alloc_uvm_watcher`: a UVM word that a kernel *inside a CUDA graph* can increment, polled by a CPU
  thread via GDRCopy, with the callback receiving both old and new values so it can tolerate missed
  intermediate updates. This is how they trigger layer-by-layer KV transfer without breaking CUDA
  graphs.

Measured `[verified, Table 2]` — TransferEngine throughput, 8×H200 with 2×200 Gbps EFA vs 8×H100
with 400 Gbps ConnectX-7:

| Op | Size | EFA | ConnectX-7 |
|---|---|---|---|
| Single `WRITE` | 64 KiB | 16 Gbps | 44 Gbps |
| Single `WRITE` | 256 KiB | 54 Gbps | 116 Gbps |
| Single `WRITE` | 1 MiB | 145 Gbps | 245 Gbps |
| Single `WRITE` | 32 MiB | 336 Gbps | 378 Gbps |
| Paged `WRITE` | 1 KiB | 17 Gbps / 2.11M op/s | 91 Gbps / 11.10M op/s |
| Paged `WRITE` | 8 KiB | 138 Gbps / 2.10M op/s | 320 Gbps / 4.89M op/s |
| Paged `WRITE` | 16 KiB | 274 Gbps / 2.08M op/s | 367 Gbps / 2.80M op/s |
| Paged `WRITE` | 64 KiB | 364 Gbps / 0.69M op/s | 370 Gbps / 0.71M op/s |

Key structural finding: **single writes need ≥16 MiB to saturate; paged writes saturate at 32–64 KiB.**
If you can express a transfer as paged, do — the saturation threshold drops by ~500×.

**1.2 Layer-by-layer KV transfer for disaggregation, with the HND layout trick**
`[verified]` https://www.perplexity.ai/hub/blog/disaggregated-prefill-and-decode and §4 of the paper.

Mechanism. During chunked prefill they increment the UVM watcher **after the attention output
projection of each layer** (CUDA-graph compatible). TransferEngine sees the change and fires
`submit_paged_writes` for that layer's pages from the current chunk. The final chunk's context
(last-token hidden states and logits for speculative decoding) goes via `submit_single_write`. The
prefiller **never sends an explicit completion message** — the decoder knows the expected transfer
count in advance and uses `expect_imm_count`.

Two layout details:

- KV caches are laid out **heads-before-pages (`HND`, not the default `NHD`)** so consecutive heads
  are contiguous, which lets a sharded transfer be one RDMA write instead of one per head.
- Under MLA with TP, the compressed KV entries are *replicated* across ranks, so they **randomly
  match prefiller ranks to decoder ranks** to spread the RDMA writes across NICs rather than
  hammering one.

Production robustness detail: cancellation from a decoder must be **explicitly confirmed by the
prefiller**, because KV pages cannot be reused while a remote write might still clobber them.
Heartbeats detect transport failure; a per-request cancellation token stops future transfers and
waits for pending ones before confirming.

Measured `[verified, Table 3]` — Qwen3-235B, H200, TP4, 2×200 Gbps EFA, page = 32 kB (128 tokens),
chunked prefill up to 16384 tokens, CUDA graphs on:

| Seqlen | TTFT non-disagg (ms) | TTFT disagg (ms) | Per-layer compute (ms) | Per-layer transfer (ms) | Chunks | Pages |
|---|---|---|---|---|---|---|
| 4K | 214 | 260 | 2.267 | 0.661 | 1 | 256 |
| 8K | 433 | 501 | 4.578 | 0.952 | 1 | 512 |
| 16K | 929 | 1042 | 9.860 | 1.610 | 1 | 1024 |
| 32K | 2179 | 2317 | 13.295 | 1.606 | 2 | 1024 |
| 64K | 5681 | 5852 | 20.344 | 1.611 | 4 | 1024 |
| 128K | 16735 | 17056 | 34.895 | 1.609 | 8 | 1024 |

Notable honest disclosure: they attribute the TTFT gap **mostly to their engine performing one extra
decode pass for the final input token, not to the KV transfer** — because the prefiller deliberately
does not sample (that would require synchronising schema-processor implementations across the two
services). Transfer is fully hidden behind compute at every length.

`UvmWatcher` callback latency under CUDA graph `[verified, Table 4]`, in µs:

| Callback | avg | min | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| Rust | 6.3 ± 1.3 | 2.5 | 6.2 | 7.0 | 12.6 | 19.4 | 64.8 |
| Python | 9.8 ± 9.0 | 6.1 | 9.3 | 11.1 | 20.4 | 41.7 | **3325.0** |

That max is the lesson: a Python callback on the critical path has a 3.3 ms tail.

**1.3 Host-proxy MoE dispatch/combine that beats IBGDA DeepEP**
`[verified]` §6 of the paper; earlier NVSHMEM version at
https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication

Mechanism. Split send/receive kernels for both dispatch and combine, with a **host proxy thread**
polling the GPU via GDRCopy and calling TransferEngine when send buffers are ready. Peers first
exchange per-expert token counts to claim a unique range in a *contiguous* receive buffer, so each
rank issues at most **2 writes for dispatch and 1 for combine per inter-node peer** (versus DeepEP's
per-token transfers over an RC QP). The routing-exchange latency is hidden by **speculatively
dispatching a small number of tokens into private per-source buffers** before routing info arrives.

The critical-path ordering trick (steal #5 above): loads over NVLink block the pipeline; stores are
fire-and-forget until a memory barrier, and the barrier's scope includes the host. So they **signal
the host first, then grid-barrier, then issue NVLink writes.** Total kernel time goes up; latency to
the first RDMA transfer goes down.

Measured `[verified]`:

- Launch of dispatch kernel → first transfer: **~15 µs at EP=64.**
- End-to-end decode speed, DeepSeek-V3 with MTP (draft length 1, acceptance 80%), EP=DP=64,
  tokens/s per user `[Table 6]`:

| Cluster | Kernel | batch=2 | batch=8 | batch=32 |
|---|---|---|---|---|
| H200 + EFA | fabric-lib | **66.75** | **56.46** | **32.00** |
| H200 + EFA | pplx-kernels (NVSHMEM) | 20.97 | 11.61 | 4.90 |
| H100 + CX-7 | fabric-lib | **78.42** | **67.67** | 36.07 |
| H100 + CX-7 | DeepEP | 73.76 | 65.79 | **36.25** |

- Dual-batch overlap ablation `[Table 7]`, EFA, tokens/s:

| Batch | fabric-lib no-overlap | fabric-lib dual-batch | pplx no-overlap | pplx dual-batch |
|---|---|---|---|---|
| 128 | 11.81 | **13.92** | 1.55 | 1.45 |
| 96 | 14.35 | **16.49** | 2.01 | 1.81 |
| 64 | 21.26 | 21.44 | 2.82 | 2.43 |
| 48 | 24.22 | 24.20 | 3.58 | 3.27 |
| 32 | **32.00** | 30.24 | 4.90 | 4.83 |

  Overlap helps only at large per-GPU batch, and **consistently degrades** the high-latency kernels.
- Kernel-level decode latency at EP=64 (DeepSeek-V3/R1 shapes: 7168×fp8 tokens with 56×fp32 scales,
  8 experts/token, batch 128), read from Figure 9 bar labels `[verified, figure]`: dispatch
  ≈286 µs (ours-EFA) / ≈190 µs (ours-CX7) / ≈180 µs (DeepEP-CX7-IBGDA); combine ≈406 / ≈311 / ≈327.
  At EP=16 and EP=32 fabric-lib beats DeepEP on both; at EP=64 DeepEP wins dispatch (proxy CPU cost
  of enqueuing to 56 inter-node peers) while fabric-lib still wins combine. At EP=8 (intra-node) they
  are ~2 µs *slower* than DeepEP, which they attribute to using NICs to exchange routing info.
- **EFA trails ConnectX-7 by only ~30% on MoE decode despite 256 KiB writes achieving less than half
  the throughput** — because decode does not saturate bandwidth.
- Host-proxy CPU overhead `[Table 8, EP64, µs]`: app enqueue 0.120 p50 / 3.484 p99.9; worker enqueue
  0.855 p50; before posting first `WRITE` 0.441 p50; **after posting last `WRITE`: 8.502 p50 on CX-7,
  27.886 p50 on EFA.** Scatter posting time scales roughly linearly with EP `[Table 9]`:
  CX-7 EP8/16/32/64 = 0.842 / 1.926 / 4.140 / 8.502 µs p50; EFA = 3.081 / 6.536 / 13.374 / 27.886.
- Private-buffer ablation: you need **≥24 tokens on ConnectX-7 and ≥32 on EFA** in the speculative
  first burst to hide route exchange; below that, up to 60% slowdown on EFA.

Negative results they publish, which are the most valuable part:

- For **prefill**, DeepEP is better: DeepEP pre-accumulates tokens over NVLink on the sender node,
  cutting RDMA bytes, and uses less buffer memory by batching subsets of tokens. Perplexity's
  decode-optimised kernels have no chunking, and "the memory overhead of our decode-optimized
  kernels limits the set of models for which a deployment is viable."
- DeepEP's sender-side partial sum on combine reduces RDMA bytes a lot but **accumulates in bf16 in
  a non-fixed order — implications for accuracy and determinism.** Perplexity accumulate on the
  receiver in fp32 and take the latency hit.
- GPU-initiated RDMA (IBGDA) "is not yet available on most cloud instances (AWS p5/p5e, eRDMA) and
  remains preliminary on p5en." And, directly relevant to us: **"For MoE, next-generation GPUs with
  wide NVLink domain (e.g. GB200 NVL72) shift communication off RDMA entirely."**

**1.4 Multi-node MoE parallelism study (EP8 → EP128)**
`[verified]` https://www.perplexity.ai/hub/blog/lower-latency-and-higher-throughput-with-multi-node-deepseek-deployment

The headline claim inverts the usual intuition: *"MoE models can simultaneously achieve higher
throughput and lower latency when utilizing more GPUs."* Reason given: decode is memory-bandwidth
bound; raising EP cuts experts-per-GPU, cutting the weight bytes each GPU must stream per token,
which allows larger batches without latency degradation.

Numbers `[reported]`: EP128 gave "about 5× higher throughput than single-node deployment at the same
output speed" in the 40–80 tok/s range. Per-MoE-layer latency at batch 128: NoOverlap 2667 µs,
DispatchOverlap 2651 µs (0.6% — i.e. **dispatch-only overlap is worthless**), MicroBatch 1896 µs
(29%). GroupGEMM time halved from 555 µs (EP8) to 270 µs (EP128) while communication grew 213 µs.
Micro-batching **degrades performance 5–40% at batch sizes < 32**. At batch ≥64 per GPU, single-node
slightly beat multi-node because NVLink beats InfiniBand. They also admit their all-to-all kernels
"currently achieve only half of the InfiniBand bandwidth."

Config asymmetry: H200 (141 GB) single-node vs H100 (80 GB) multi-node; KV length fixed at 5,000.

**1.5 Speculative decoding at Perplexity — including what failed**
`[verified]` https://www.perplexity.ai/hub/blog/accelerating-sonar-through-speculation

- **EAGLE trees rejected.** "Custom attention masks for a whole sequence significantly slows down
  attention by up to 50%, nullifying some speedup." They deployed single-token prediction instead.
  This is the single most directly actionable negative result in this file for our 3-1-4 config.
- **Draft/target scheduling.** The draft runs only on the leader rank (TP mismatch). They **carry
  logits over between runs so one draft execution overlaps the CPU-side batch scheduling work** for
  the next iteration.
- **MTP scheduling.** Processes `2 * D` tokens for decode batch D — which they note is ideal for MoE
  micro-batching over slow interconnects. To avoid a CPU–GPU sync after acceptance they **do
  redundant GPU work on non-accepted tokens**, which is invisible because the draft is small.
- **KV repopulation.** With a separate draft model, the draft has not seen the accepted token, so it
  is re-run to fill its KV entries. Under MTP they run the draft on the whole sequence using the
  *target's* hidden states as inputs, for accuracy.
- **Training.** MTP heads trained on a single 8×H100 node "in about one day" for Llama-1B → 70B.
  Reintroducing **RMSNorm layers in the MTP heads** "not only allowed training to converge, but it
  also boosted accuracy by a few percentage points" at longer sequences.
- **Padding.** "MLP layers create measurable overhead for longer sequences due to padding boundaries
  (64 tokens), though attention overhead is negligible."
- No acceptance rates or end-to-end speedups published. `[unverified]` for those.

**1.6 Point-to-point RL weight transfer (structurally relevant to fast model swaps)**
`[verified]` §5 + Table 5 of the paper. 1.3 s to push **Kimi-K2-1T** weights from 256 training GPUs
(BF16, FSDP/PP/EP = 16/2/8) to 128 inference GPUs (FP8, EP=32); DeepSeek-V3-671B and Qwen3-235B are
1.2–2 s. Instead of gathering to training `Rank0` and broadcasting (bottlenecked on one NIC), each
training GPU RDMA-`WRITE`s directly to inference GPU memory using a static schedule computed once
from parameter metadata. Four-stage pipeline: H2D memcpy → `full_tensor()` + projection fusion +
quantise → RDMA → GLOO barrier over Ethernet. A configurable **GPU memory watermark** gates new
tasks so `full_tensor()` does not OOM.

Per-rank breakdown: total 1233 ms; `full_tensor()` 518 ms (974 calls); waiting for other ranks
357 ms; H2D memcpy 184 ms; quantise 88 ms; RDMA submit **26 ms for 1144 work requests (23 µs each)**;
fuse projections 18 ms; only **42 ms of transfer time not hidden by the pipeline**.

**1.7 EFA saturation engineering**
`[verified]` https://www.perplexity.ai/hub/blog/high-performance-gpu-memory-transfer-on-aws
AWS p5: 32 network cards, 3200 Gbps aggregate, four 100 Gbps EFA cards per PCIe switch, two sockets,
four PCIe switches. They reached **3,108 Gbps (97.1% of theoretical)**. Levers: CPU core pinning to
avoid NUMA effects, NUMA-aware `libfabric` resource allocation, pre-established connections
("network warmup"), dedicated thread per GPU, state sharding to reduce thread contention, operation
batching and lazy posting. Explicit rejection of NCCL: "requires establishing a static 'world',
which requires restarting the entire cluster when adjusting the participating nodes."

**1.8 Older but numerically clean: Llama-2-70B on H100 vs A100**
`[verified]` https://www.perplexity.ai/hub/blog/turbocharging-llama-2-70b-with-nvidia-h100
TensorRT-LLM v0.5.0, 1024 in / 512 out. H100 fp8 vs A100 fp16 at TP-8, BS-128: 49% lower latency and
202% higher throughput; with a latency constraint, 251%; **H100 fp8 at TP-2 gives 373% the
throughput of A100 fp16 TP-8 for <10% latency increase** — i.e. dropping TP is a throughput lever if
you can afford the latency. Peak 767 output tok/s/GPU (H100/fp8/TP-2/BS-128) but at 42 s latency.
fp8 and A100 SmoothQuant w8a8: ~10% latency improvement, <1% perplexity change. **w4a16 rejected for
production: 7% perplexity degradation** despite better low-batch performance.

### Open-source artifacts, and what is actually usable

| Artifact | URL | Usable for us? |
|---|---|---|
| `pplx-garden` (`fabric-lib`, `p2p-all-to-all`, `pplx-unigram`) | https://github.com/perplexityai/pplx-garden | The **kernel design patterns** are the value. The RDMA library itself is irrelevant on a single 8×B200 node — but the NVLink write-ordering trick, the contiguous-receive-buffer packing, the private-buffer speculation, and the HND KV layout all transfer to intra-node code |
| `pplx-kernels` | https://github.com/ppl-ai/pplx-kernels | **Deprecated**, superseded by pplx-garden. sm90a only (`TORCH_CUDA_ARCH_LIST=9.0a+PTX`). NVLink dispatch 41.6 ± 1.3 µs at 1 token/GPU, 83.6 ± 1.0 µs at 128 tokens/GPU |
| `bumblebee` | https://github.com/perplexityai/bumblebee | Not inference — a supply-chain scanner. Noted so nobody wastes time |

### Blackwell/B200 specifically

One statement, and it is a strategic one `[verified]`: *"For MoE, next-generation GPUs with wide
NVLink domain (e.g. GB200 NVL72) shift communication off RDMA entirely."* Nothing published on B200
tuning. `[inferred]` Their entire MoE-comms body of work is aimed at the inter-node regime we do not
have; what survives for us is the *ordering and packing discipline*, not the transport.

---

## 2. Databricks / Mosaic

Databricks is **the current Artificial Analysis leader for GLM-5.2**, which makes them the direct
competitor named in our brief.

### The leaderboard position, verified

`[verified]` https://artificialanalysis.ai/models/glm-5-2/providers and
https://artificialanalysis.ai/providers/databricks — measured 2026-08:

| Provider | Output tok/s | TTFT (s) | Price $/1M | Context |
|---|---|---|---|---|
| **Databricks** | **336.5** | 6.74 | 0.90 | 1M |
| Makora (NVFP4) | 329.8 | 6.92 | 0.66 | 1M |
| Baseten (FAST) | 246.8 | 9.82 | 0.54 | 524k |
| Nebius (FP4) | 219.6 | 10.21 | 1.06 | 432k |
| Together AI | 208.6 | 10.24 | 0.67 | 262k |
| Parasail (NVFP4) | 166.0 | 1.00 | — | 1M |
| Fireworks | 97.0 | 1.35 | 0.48 | 1M |
| Novita (FP8) | 79.0 | 2.04 | 0.42 | 1.05M |
| DeepInfra (FP4) | 64.0 | 1.19 | 0.25 | 1.05M |

AA's own summary: "Databricks offers the best performance with both the highest speed and lowest
latency." Databricks also serves Kimi K3 (max) at 105 tok/s / 20.27 s TTFT.

**Read the TTFT column carefully.** AA states TTFT "accounts for reasoning model 'thinking' time".
Providers at ~1 s TTFT and ~100–166 tok/s are almost certainly serving with reasoning suppressed or
different defaults. This column is not comparable across rows and is not comparable to our 189 ms.

### What they claim, and what they have actually published

- `[reported]` Matei Zaharia, LinkedIn: "Databricks delivers the fastest inference for GLM 5.2. Our
  team has made some awesome optimizations here and we're not done!" A reshare states **392 tok/s**
  vs a previous 328. https://www.linkedin.com/posts/mateizaharia_databricks-delivers-the-fastest-inference-activity-7475939456869457920-SHs0
- `[reported, third-party]` A Medium write-up attributes the result to a move from **H200 to B300**,
  DeepGEMM FP8 grouped-GEMM kernels, DeepEP for expert parallelism, custom kernels, speculative
  decoding tuned for GLM-5.2, and FP8 KV cache (~160 GB → ~80 GB). It cites **no primary Databricks
  source.** Treat every element as unconfirmed.
  https://dbxdev.medium.com/392-tokens-per-second-how-databricks-and-the-b300-just-reset-the-ai-speed-limit-7d84285b7e6e
- **I could not find any Databricks engineering post describing the GLM-5.2 work.** `[unverified]`

So the useful Databricks material is elsewhere — and it is about the part of the problem we are
weakest on: running hot without wrecking latency.

### 2.1 "Reliable LLM Inference at Scale" — model units, Dicer, autoscaling

`[verified]` https://www.databricks.com/blog/reliable-llm-inference-scale

Scale claim `[reported]`: **more than 125T tokens per month.**

The framing is the best statement of our problem I found anywhere: *"the cost to serve a request is
highly variable and hard to estimate a priori"* — output length dominates latency and is
unpredictable, so classical capacity planning fails.

Mechanisms:

- **Model units.** A VM-like capacity abstraction. Request cost is modelled as a *multi-dimensional*
  function of input tokens, output tokens, and prefill/decode character. **Coefficients come from
  automated benchmarking of each model on each hardware type.** This is the thing to build: a
  calibrated cost model per (model, GPU, dtype, spec-config), not a token count.
- **Cost-aware routing.** Their Dicer auto-sharder makes "routing decisions based on server load in
  model units" rather than request counts or queue depth. Explicit rationale: long-context requests
  create hotspots while short requests leave capacity unused.
- **Stateful sticky sessions.** Workloads route to a consistent subset of servers, which improves
  cache hit rate *and* limits failure blast radius.
- **Autoscaling on model-unit utilisation**, triggering when the engine approaches a percentage of
  its maximum model units. Result `[reported]`: **"over 80% GPU savings compared to static
  provisioning at peak"** for bursty traffic.
- **Prioritised health checks.** Black-box health checks get the highest scheduling priority so they
  are not starved under load. Result `[reported]`: false liveness-probe failures went "from several
  per week to zero"; sub-5-minute detection and recovery from silent server hangs (which they say
  come from edge cases like structured output and multimodal inputs).
- **A profiling war story.** Image processing consumed ~10× more CPU than anything else; switching
  to Torchvision-based processors and setting `OMP_NUM_THREADS` correctly made "requests completed
  per second jump >3× with the same replicas and load." `[reported]`

`[inferred]` The `OMP_NUM_THREADS` finding generalises: on a 2-NUMA-node 8×B200 box, CPU-side
oversubscription in the tokeniser/detokeniser/sampler path is a plausible and unglamorous source of
our C1→C16 cliff. Cheap to check.

### 2.2 Serving-platform mechanics

`[verified]` https://www.databricks.com/blog/ai-serving-platform-adapts-your-model
Two-axis autoscaling: horizontal (request-based, reacts in **5–20 s**) and vertical (concurrency
adjustment, **30 s** intervals). Scale-up aggressive — "10 to 10K QPS in <60 seconds"; scale-down
conservative over a 5-minute window. Fully isolated Kubernetes deployment per endpoint, warm node
pools with pre-provisioned base images, parallel model-image download from cached storage. Claimed
**300K+ QPS at <10 ms p99 latency overhead** for the routing layer. Engines: vLLM, Triton, or the
customer's own runtime.

### 2.3 Foundational: MBU, and the batching hierarchy

`[verified]` https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
Defines **Model Bandwidth Utilization** = achieved memory bandwidth / peak, where achieved =
(model params + KV cache size) / TPOT. Complements MFU. Static batching on MPT-7B / 1×A100
(512 in, 64 out): batch 1 → 0.9 req/s, batch 16 → 8.0 (8.9×), batch 64 → 12.5 (14×) — sublinear, and
they note continuous batching gives 10–20× over dynamic batching. Llama2-70B TTFT for 512-token
input at batch 1: 154 ms on 4×A100, 114 ms on 8×A100 — **diminishing returns past 4–8 GPUs due to
falling MBU and comms overhead.** `[inferred]` Worth recomputing MBU for our TP8 GLM-5.2 at C1: if
MBU is high and dense GEMM is still 37.1%, we are compute-bound at C1 and the levers are shapes and
dtype, not bandwidth.

### 2.4 TensorRT-LLM integration, and DBRX

`[verified]` https://www.databricks.com/blog/Integrating-NVIDIA-TensorRT-LLM — six-month integration
from Jan 2024; the useful nugget is that NVIDIA added "Python support for the batch manager written
in C++" so TRT-LLM could be embedded in their Python backend server. No benchmark numbers.

`[verified]` https://www.databricks.com/blog/accelerated-dbrx-inference-mosaic-ai-model-serving —
DBRX 132B total / 36B active, 16 experts choose 4. Fused kernels, custom **GroupGEMM** for MoE,
**dropless MoE routing via block-sparse ops** (variable-capacity expert assignment). Claims `[reported]`:
<0.4× the request latency of a comparable dense model at low batch; 2.5× relative throughput at high
batch; **"over 8× as many total tokens per second" at a <30 ms latency target**; ~150 tok/s observed
in their playground; 8-bit halves serving cost. Benchmarked on H100, 8-way TP, 16-bit.

### Blackwell/B200 specifically

`[reported, third-party only]` H200 → B300 for GLM-5.2. Nothing first-party. `[unverified]` on
dtype, parallelism, spec-decode config, or disaggregation.

---

## 3. Character.AI

Predates Blackwell entirely and the techniques are structural, exactly as the brief anticipated.
This is the most aggressive published KV-cache-reduction programme in the industry.

### What they run

Their own **Kaiju** dense models — Small 13B, Medium 34B, Large 110B — trained on H100s in GCP with
tensor+sequence parallelism inside nodes and FSDP across nodes `[verified]`. Explicitly *not*
optimising for benchmarks: they accept a "measurable, negative impact on some AGI benchmarks like
MMLU" from MQA because "we're not optimizing for AGI."

### 3.1 The >20× KV cache reduction stack

`[verified]` https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/ and
https://blog.character.ai/inside-kaiju-building-conversational-models-at-scale/

| Technique | Detail | Reduction |
|---|---|---|
| Multi-Query Attention | MQA in **all** attention layers; head dim 128, **16 query heads, 1 KV head** | 8× vs GQA |
| Hybrid attention horizons | Local window **1024 tokens**; **1 in 6 layers global** (Kaiju describes it as roughly 5:1 interleave). O(len²) → O(len). "little to no drop in needle-in-the-haystack retrieval." **No attention sinks** | linear scaling |
| Cross-layer KV sharing | Adjacent layers with identical attention share a KV cache; typically **2–3 layers per cache**. Global layers share across *non-adjacent* layers | 2–3× |
| int8 KV | Cache stored and computed in int8, via QAT not PTQ | 2× |
| **Combined** | | **"more than 20X without regressing quality"** |

`[inferred]` The cross-layer-sharing idea is the one that transfers cleanly to a DSA-MLA model. MLA
already compresses KV; layer-tying is orthogonal to it and is a training-time decision, so it is a
lever for *our next model*, not for GLM-5.2. The 1-in-6-global-layers pattern is likewise a model
architecture note for Kimi K3 / Qwen3.8 / DeepSeek V4 evaluation.

### 3.2 int8 everywhere, trained in

`[verified]` int8 for weights, activations **and** KV, with **quantization-aware training from
scratch** — not post-training. Precision split during training: int8 for forward-pass weights and
KV; bf16 for activations and local gradients; fp32 for gradient accumulation and FSDP master
weights. QAT made training **20–30% faster** while maintaining "bf16-level model accuracy". Support
machinery: pre-layer RMSNorm, **dynamic activation clamping** for stability, "virtual scalars
(Bungee)" for int8 training stability, 6-bit **Squinch** blockwise gradient compression modelling
gradient magnitudes as log-uniform, and ternary weight updates compressing broadcasts to
**1.6 bits/parameter**.

`[inferred]` The strategic point for us: Character.AI's cost position comes from **co-designing the
model for the serving stack**, and the quantisation is *trained in*, so there is no accuracy cliff
to manage at deploy time. We inherit GLM-5.2 as-is, so our NVFP4 build carries a risk Character.AI
engineered away. Their split (low precision for weights+KV, bf16 activations, fp32 accumulation) is
a sane default to verify against.

### 3.3 The attention kernel work — most transferable item in this section

`[verified]` https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/
Built on a **custom FlashAttention-3 fork**, H100 SXM5, CUDA 12.4, head dim 128, 16 Q heads, 1 KV head.

- **Fused dequantisation via producer/consumer warp specialisation.** Dequant is overlapped with the
  MMA instructions rather than run as a separate pass.
- **Query-head packing (`packQ`).** In decode with short sequences, a single query head leaves the
  64-row WGMMA warpgroup tile almost empty. They pack multiple query heads into one threadblock.
  Reported **up to 9.3× time reduction at large batch, 6.4× at smaller scales.**
- **Half INT8 vs Full INT8.** Half INT8 = first matmul on int8 tensor cores, second in bf16 (V must
  be upcast). Full INT8 = both matmuls int8, no V upcast but the *probabilities* get quantised.
  **They chose Half INT8 "to avoid potential regression in model quality."** A published negative
  result on quantising the P·V matmul.

Measured `[verified]`, TFLOP/s, seqlen 32768, batch 1, H100 SXM5:

| Masking | BF16 | Half INT8 | FP8 |
|---|---|---|---|
| No causal mask | 632 | 742 | 958 |
| Causal mask | 634 | 763 | 947 |

Decode, context 131072, batch 2: **1,337.08 µs → 51.90 µs with splitKV + packQ (~25.8×).**
Overall vs a Triton baseline: **+10% prefill, +30% decode.**

`[inferred]` Directly applicable to our DSA-MLA decode on SM100. Our decode has very few query rows
per KV head; if our attention kernel is not packing heads into the tensor-core tile, 10.9% attention
time is mostly tile underfill. Same class of problem as the Sarathi tile-quantisation finding (§5).

### 3.4 Stateful caching — 95% hit rate

`[verified]` LRU cache with a **tree structure indexed by rolling hash**. Reported **95% cache rate**
on a workload where "the average message has a dialogue history of 180 messages."

### 3.5 Prompt Poet — cache-preserving truncation

`[verified]` https://github.com/character-ai/prompt-poet
YAML + Jinja2 prompt templating with priority-based truncation. The mechanism worth stealing:
**the truncation boundary is fixed and only moves approximately every k turns**, controlled by
`truncation_step` (example: 4000 tokens against a 128,000 default limit). This deliberately drops
more tokens than necessary in exchange for a stable prefix hash. They define the objective
explicitly: **prefix cache hit rate = prompt tokens retrieved from cache / total prompt tokens.**

`[inferred]` This is the cheapest lever on the list for any repeated-prefix workload. It converts a
sliding window (0% cache hits) into a stepped window (near-100% between steps).

### 3.6 Cost claims

`[reported]`, and unverifiable from outside: ~20,000 queries/s ("about 20% of the request volume
served by Google Search"); "less than one cent per hour of conversation"; serving costs reduced
"by at least 33×" since 2022; "13.5 times less to serve our traffic than it would cost a competitor
building on top of the most efficient leading commercial APIs." Their illustrative model: at 100M
DAU × 1 h/day, $365M/yr for them vs $4.75B on commercial APIs. Treat as directional.

### Open-source artifacts

`prompt-poet` (usable today, ~1.2k stars). No kernels released — the FA-3 fork is described but not
published `[unverified]`.

### Blackwell/B200

Nothing. All published work is H100-era.

---

## 4. Anyscale / Ray

### 4.1 The benchmarking-methodology critique — still the best statement of it

`[verified]` https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference

The critique, which applies directly to how we should measure ourselves against AA:

- **Random tokens are wrong.** They "fail to represent real data distributions, potentially masking
  performance optimizations like speculative decoding that depend on authentic patterns." Our
  365 tok/s is on real data — good; keep it that way and distrust any synthetic number.
- **Fixed-size inputs and outputs are wrong.** They "ignore algorithmic innovations such as paged
  attention and continuous batching, whose benefits emerge precisely from handling size variations."
- **ITL should include TTFT**, deliberately, so systems that "start streaming very late" are
  penalised.
- **Tokenizer bias is real.** They standardise on the Llama 2 fast tokenizer to compare providers
  "in a system independent way", noting ~1.5 tokens/word for Llama 2 vs ~1.33 for the ChatGPT
  tokenizer. `[inferred]` AA's tok/s number for GLM-5.2 is in *GLM's* tokens. A provider with a
  different effective tokenisation is not directly comparable — but for us, competing on the same
  model, this cancels out.
- **The 100:1 rule.** Measured on Llama 2 70B: each input token contributes 0.3–0.7 ms to end-to-end
  time; each output token 30–60 ms. "100 input tokens have approximately the same impact on latency
  as a single output token." `[inferred]` For AA's ~10k in / ≥1500 out shape, this says output-side
  work is ~15× the input-side work in latency terms. Optimise decode.
- Their own comparison workload: 550 ± 150 input tokens, 150 ± 20 output, 5 concurrent requests as
  the reference point — and they show the winner *changes* with concurrency (a 15% end-to-end lead
  at 5 concurrent compressed to 5% at 30). `[inferred]` AA measures both single-stream and
  10-parallel; expect our ranking to differ between the two, and tune for both.

### 4.2 LLMPerf

`[verified]` https://github.com/ray-project/llmperf — load test (inter-token latency and generation
throughput, per-request and across concurrent requests) + correctness test (number-word → digits).
Prompts built from randomly sampled Shakespeare sonnets. Tokens counted with a single
LlamaTokenizer across all providers. Their own stated caveat: "The endpoints provider backend might
vary widely, so this is not a reflection on how the software runs on a particular hardware," plus
time-of-day and load variability.

### 4.3 PD disaggregation economics with goodput SLOs (AMD MI325X)

`[verified]` https://www.anyscale.com/blog/ray-vllm-prefill-decode-disaggregation-amd-mi325x-67-percent-savings
Qwen3-235B and DeepSeek-V3 on MI325X (288 GB HBM3e, 8/node), 8× Mellanox ConnectX over RoCE.
Methodology is the part to copy: **same total GPU budget, split P/D vs aggregated, compare QPS under
identical SLAs.** Results `[reported]`: 1.3–2.3× more QPS depending on workload, up to **67% compute
cost reduction**; TPOT advantage 5–10 ms/token under load; **PD sustains flat TPOT while aggregated
degrades linearly with QPS**; E2E advantage grows from 12% at OSL=140 to 17% at OSL=4K. Aggregated
wins on pure TTFT because there is no KV transfer step.

Optimal P:D ratios by workload `[reported]` — a useful prior:

| Workload | Cache hit rate | Optimal ratio |
|---|---|---|
| 16K in / 1K out | 0% | 2P:1D |
| 16K in / 4K out | 0% | 1P:3D |
| Multi-turn, high reuse | 80% | 1P:2D |
| Multi-turn, moderate reuse | 30–60% | 1P:1D to 1P:2D |

Operational landmine `[verified]`: "TCP fallback degraded throughput catastrophically – up to 19×"
versus proper RDMA. Config: vLLM 0.18.0 on ROCm 7.0, RIXL (AMD's NIXL equivalent) at commit
f33a5599, `UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc`, `VLLM_ROCM_USE_AITER=1`,
`VLLM_ROCM_USE_AITER_MOE=1`.

### 4.4 Ray Serve LLM routing

`[verified]` https://www.anyscale.com/blog/high-performance-distributed-inference-ray-serve-llm-vllm-google-kubernetes-gke
The architectural point: **decouple the routing decision from the data path.** HAProxy queries a
request router with the request content for a routing decision, then opens a **direct** HTTP
connection to the chosen replica for token streaming — eliminating an intermediate ingress layer
that previously forwarded every token. HAProxy is C-based and **disables Nagle's algorithm by
default** for streaming. Reported on 8×H100 / 192 CPU cores: prefill-heavy (ISL 8000, OSL 50) 4.4×
throughput vs baseline; decode-heavy (ISL 50, OSL 500) 24×; prefill TTFT 355 ms vs 389 ms for
vllm-router; decode TPOT 13.6 ms vs 14.8 ms at concurrency 256. **Models tested were tiny
(Qwen3-0.6B, Phi-tiny-MoE) — these numbers measure the serving layer, not the model.** Flags:
`RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1`, `RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY=1`. They cite
"96% effective shared prefix rate per session" in agentic workloads as the motivation for consistent
hashing.

`[inferred]` For AA measured over the public internet, Nagle and an extra proxy hop are real
per-token latency. At TPOT 2.74 ms, a 40 ms Nagle delay is 15 tokens' worth.

---

## 5. Microsoft

Three substantial pieces of work, all pre-Blackwell, all structurally live.

### 5.1 Splitwise — phase splitting, with real Azure production traces

`[verified]` https://arxiv.org/html/2311.18677v2 (ISCA'24),
https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/

The trace characterisation is the durable contribution, and Microsoft **released the traces**:
https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md (CC-BY,
sampled 2023-11-11, fields `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`, two workloads).

| Service | Median prompt tokens | Median output tokens |
|---|---|---|
| Coding | 1,500 | 13 |
| Conversation | 1,020 | 129 |

`[verified]` Conversation workloads "spend 60–70% of time running only 20 tokens or fewer" —
i.e. severe decode underutilisation.

Power finding, and it is the one nobody copies `[verified]`: on H100, prompt-phase power rises
linearly with batch size while **token-phase power stays flat at ~70% of TDP regardless of batch**.
Capping power during the token phase by 50% causes "almost no latency impact"; the same cap during
prompt causes substantial degradation. `[inferred]` If our 8×B200 box is ever power- or
thermal-limited (and at 40.8k tok/s aggregate it plausibly is), asymmetric power capping — full
power for prefill, capped for decode — is free headroom.

KV transfer: "serialized transfer linearly increases with prompt size" but **layer-wise transfer
hides most of it**, leaving "8 ms constant non-overlapped transfer time" on A100 and **5 ms on
H100**. Cluster results `[reported]`: Splitwise-AA gives 2.15× throughput at same power and cost vs
Baseline-A100; Splitwise-HA 1.18× at 10% lower cost; Splitwise-AA 1.4× vs Baseline-H100 at iso-cost
(25% higher power); Splitwise-HHcap matches Baseline-H100 throughput at **25% lower power**.
Negative result: heterogeneous designs showed a "7% throughput setback" when a coding-optimised
cluster ran conversation traffic — **the pool split is workload-specific and does not transfer.**

### 5.2 DeepSpeed-FastGen — Dynamic SplitFuse

`[verified]` https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen

Mechanism, with the reasoning given, which is the part worth internalising:

1. Long prompts are decomposed into small chunks scheduled across multiple forward passes, **only
   the final pass generating**.
2. Short prompts are composed to **exactly fill a target token budget**; even short prompts may be
   split to hit the budget precisely.

Justification: batch *composition* barely matters compared to raw token count per forward pass;
models have a memory-bound and a compute-bound regime; and **because the throughput curve is concave
in token count, the optimal partition divides tokens equally across passes.** That last line is a
one-sentence proof that a constant forward-pass token budget is optimal.

Measured `[reported]`, Llama-2 70B on 4×A100-80GB: up to 2× throughput at identical 9 s latency
(1.36 vs 0.67 rps); up to 50% latency reduction at equal throughput (7 s vs 14 s). Under an SLA
model (prompt SLA = `tokens_in_prompt/512` seconds; generation SLA of 2/4/6 tok/s per user), 2.3×
higher throughput than vLLM; at the 4 tok/s SLA, **1.42 q/s with <1% SLA violations vs vLLM's
0.63 q/s with 28% violations.** **P95 generation latency 3.7× lower**, because vLLM preempts
generation to process new prompts while FastGen processes prompt and generation concurrently.

`[inferred]` The P95 mechanism is the same failure mode as our C1→C16 cliff: an arriving request
stalls the in-flight decodes. See also vLLM's speculative padding (§11), which is the modern
spec-decode-aware version of this fix.

### 5.3 Sarathi-Serve — the token budget derived from the SLO

`[verified]` https://arxiv.org/abs/2403.02310 (OSDI'24), code at
https://github.com/microsoft/sarathi-serve. Read in full from the PDF.

Two ideas: **chunked-prefills** (split a prefill into near-equal chunks) and **stall-free batching**
(add new requests to a batch without pausing ongoing decodes, filling the leftover token budget with
prefill chunks). Together they produce "uniform compute hybrid batches in most cases, which helps
reduce bubbles when using pipeline parallelism."

**How to pick the token budget** — the procedure, verbatim in substance: one-time profiling of
batches with different token counts, setting the budget to "the maximum number of tokens that can be
packed in a batch without violating TBT SLO." Three forces pull against each other:

1. **Repeated KV reads.** If a prefill is split into N chunks, chunk 1's KV is loaded N−1 times,
   chunk 2's N−2 times, etc. But they find "even at small chunk sizes attention prefill operation is
   compute bound," so this is a second-order cost.
2. **Tile quantisation.** GPUs tile matmuls; non-divisible dimensions make some thread blocks do
   extraneous work. **"Using chunk size of 257 can increase prefill time by 32% compared to that
   with chunk size 256."** Measured overhead of chunking on Yi-34B (TP-2) prefill: **at most ~25% at
   chunk size 512, near-zero at 2048.**
3. **Pipeline bubbles.** Larger chunks → higher inter-batch runtime variance → more bubbles.

They use Vidur (their profiler/simulator) to pick the budget. Deployed budgets: **2048 relaxed,
512 strict**; 1536 for LLaMA2-70B relaxed, to limit pipeline bubbles.

Capacity results `[verified]`, QPS under P99-TBT SLOs (strict/relaxed: Mistral-7B 0.1/0.5 s,
Yi-34B 0.2/1 s, LLaMA2-70B and Falcon-180B 1/5 s), on `openchat_sharegpt4` (median 1730 prompt /
415 output tokens) and `arxiv_summarization` (7059 / 208):

| Model / HW | vs Orca | vs vLLM |
|---|---|---|
| Mistral-7B, 1×A100 | up to 2.78× | — |
| Yi-34B, 2×A100 TP2 | up to 4.00× | up to 3.7× |
| LLaMA2-70B, 8×A40 TP4-PP2 | up to 6.31× | up to 4.3× |
| Falcon-180B, 4×A100 ×2 nodes TP4-PP2 | up to 5.62× | — |

Ablation `[verified, Table 4]`, Yi-34B on 2×A100, token budget 1024, seconds:

| Scheduler | P50 TTFT (sharegpt) | P99 TBT (sharegpt) | P50 TTFT (arxiv) | P99 TBT (arxiv) |
|---|---|---|---|---|
| hybrid-batching only | 0.53 | 0.68 | 3.78 | 1.38 |
| chunked-prefills only | 1.04 | 0.17 | 5.38 | 0.20 |
| **Sarathi-Serve (both)** | 0.76 | **0.14** | 3.90 | **0.17** |

Neither technique alone is good: chunking alone inflates TTFT, hybrid batching alone leaves TBT
spikes. Their own honest note on disaggregation: it eliminates prefill/decode interference entirely
and yields better TTFT (full prefills are more efficient than chunked ones), but requires migrating
KV and **under-utilises the prefill replicas' GPU memory capacity**; they leave the quantitative
comparison to future work.

### Blackwell/B200 from Microsoft

`[unverified]` — I could not find an Azure engineering post with Blackwell inference configuration
or numbers. Azure's AI/ML and Compute blog indexes surfaced nothing on GB200/GB300/ND-v6 inference
performance.

---

## 6. AWS

### 6.1 Tiered KV cache on SageMaker HyperPod with Curvine

`[verified]` https://aws.amazon.com/blogs/machine-learning/tiered-kv-cache-for-large-llms-on-amazon-sagemaker-hyperpod-with-curvine/ (2026-08-12)

Three tiers: **L0** = vLLM paged attention in HBM; **L1** = LMCache offload to host DRAM; **L2** =
Curvine, a distributed cache filesystem pooling node-local NVMe into one namespace, mounted as a
ReadWriteMany PVC and addressed by LMCache through an `fs://` connector.

Measured `[reported]`, on a deliberately modest setup (2× ml.g6e.4xlarge, 48 GB/GPU,
Qwen2-7B-Instruct fp16, TP1, 2 vLLM replicas, 20% instance memory to L1):

- Cross-pod L2 reuse hit rate **100% (1,925/1,925 tokens)** for identical prompts across nodes.
- Same-node L2 write ~**9.6 GB/s**; cross-node L2 read ~**1.8 GB/s**; cross-node load ~**56 ms for a
  ~1,900-token prompt**.
- Single-shot TTFT speedups: 500 tokens **0.99×** (no benefit), 1000 tokens 1.7×, 2500 tokens 2.7×
  (774 ms → 287 ms), 3000 tokens 2.2× (490 ms saved).
- Four-turn dialogue, 530 → 2,114 accumulated tokens: overall 1.30× (4.21 s → 3.25 s).

**The threshold is the finding** `[verified]`: "Below roughly 1,000 tokens the L2 round-trip costs
about as much as simply recomputing the prefill (0.99× at 500 tokens) ... workloads dominated by
prompts under approximately 1,000 tokens should rely on L0/L1 rather than L2."

Operational gotchas worth copying: `PYTHONHASHSEED=0` so cache keys are identical across pods (a
silent 0%-hit-rate bug otherwise); `LMCACHE_REMOTE_SERDE=naive` to avoid a zip-serialisation bug in
cachegen; `LMCACHE_REMOTE_URL=fs://localhost:0/mnt/curvine/l2cache/`.

`[inferred]` For AA's ~10k-token prompts this tier is comfortably above the break-even threshold. For
short agentic turns it is a net loss. Gate the L2 lookup on prompt length.

### Blackwell/B200 from AWS

`[unverified]` — I found no AWS engineering post giving Blackwell (P6/P6e) LLM serving configuration
or measured numbers. Perplexity's EFA work (§1.7) is the best AWS-hardware serving material
available and it is written by a customer, not by AWS.

---

## 7. Google

### 7.1 GKE Inference Gateway — routing signals and priority shedding

`[verified]` https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway

Signals consumed from model servers: **KV cache hits**, **accelerator utilisation** ("percentage of
time the GPU or TPU is actively processing"), and **request queue depth**. The endpoint picker
scores on prefix-cache match length ("tracks available prefix cache indexes on each model server,
and gives a higher score to a server with a longer prefix cache match"), server load (KV-cache
utilisation + pending queue depth), and LoRA adapter affinity.

**Priority-based shedding**: `InferenceObjective` carries an integer `Priority`; "requests with a
Priority less than 0 are considered lower priority and are dropped first" under resource pressure.
`[inferred]` This is the mechanism that lets you run at high utilisation without destroying P50 for
the traffic you care about — you *shed* rather than *queue*. If we chase a P50 leaderboard number
while also serving bulk traffic, an explicit priority class with negative-priority shedding for the
bulk tier is the correct design, not a shared queue.

No numeric benchmarks published in the doc `[verified]` — "specific numerical benchmarks are not
provided."

### 7.2 llm-d / gateway-api-inference-extension — the numbers Google's docs omit

`[verified]` https://llm-d.ai/blog/intelligent-inference-scheduling-with-llm-d

EPP cycle: endpoint discovery → filtering → scoring → selection, with tie-breaking and fallback.
Two scorer configurations compared on a **high prefix-sharing** workload:

| Scorer | Success rate | TTFT at high QPS | ITL | Throughput |
|---|---|---|---|---|
| Prefix-only | ~55% | >140 s | ~160 ms | plateaus at 2–3k tok/s |
| **Prefix + Load** | **100%** | near-zero | **~30 ms** | scales linearly to ~60k tok/s |

On a **low prefix-sharing** workload the two are indistinguishable (both ~60k tok/s at 20 QPS, 100%
success). Hardware and model not specified `[verified — the post does not state them]`.

`[verified]` https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm —
Llama 3.1 70B on 4× MI300X (`tensor-parallel-size=4`, `gpu-memory-utilization=0.90`,
`--max-model-len=65536`): **3× output tok/s and 2× lower TTFT** from prefix-cache-aware routing via
Envoy + Gateway API Inference Extension, versus round-robin. They name three production problems
solved: NFS storage bottlenecks on large model files, node-affinity rigidity, and round-robin
failing to exploit GPU caches.

Note `[verified]`: the Endpoint Picker has moved out of `kubernetes-sigs/gateway-api-inference-extension`
into `llm-d/llm-d-router`; the sigs repo keeps a "lightweight EPP (LWEPP)" for conformance only.

### Blackwell/B200 from Google

`[unverified]` — nothing found. Their published inference work is TPU- and Kubernetes-flavoured.

---

## 8. Meta

Two excellent sources: an engineering blog post on parallelism, and a conference talk that is by far
the most candid operational account in this file.

### 8.1 Parallelism: DDA allreduce, context parallelism, expert parallelism

`[verified]` https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/

Stated targets: **TTFT < 350 ms for prefill, TTIT < 25 ms for decoding.**

- **Allreduce is up to 30% of latency.** Their fix is **Direct Data Access (DDA)**:
  - *DDA flat*: reduces latency from **O(N) to O(1)** by having each rank directly load peer memory
    (a one-shot algorithm).
  - *DDA tree*: reduce-scatter + all-gather phases with direct access, matching the ring algorithm's
    data volume at a constant-factor latency advantage.
  - Measured on AMD MI300X `[reported]`: **10–50% faster than RCCL for decode, 10–30% for prefill,
    ~10% TTIT reduction**, with performance parity to NVIDIA H100.
- **Context parallelism** via ring-attention variants, with two modes: **Pass-KV** (query stays
  local, K/V exchanged) and **Pass-Q** (queries exchanged). Reported for Llama 3 405B: 128K prefill
  in **3.8 s over 16 nodes**; 1M-token prefill in **77 s**; 10M tokens in under 60 s across 32 H100
  hosts; near-linear scaling. `[reported]` — I am relying on the extracted figures here.
- **Expert parallelism** uses a **two-shot all-to-all**; EP communication is cited as 10–30% of
  latency. Optimisations: dynamic all-to-all with sub-chunk routing, and **persistent all-to-all**
  to amortise memory-handle exchange overhead.
- Future direction stated: N-D parallelism and **disaggregated inference tiers** — compute-optimised
  hardware for prefill, memory-bandwidth-optimised for decode; kernel-level integration and
  device-initiated operations to cut CPU overhead.

No open-source artifacts named in the post `[verified]`.

`[inferred]` DDA-flat is the published confirmation that a one-shot direct-load allreduce beats a
ring on the decode path. On NV18 all-to-all NVLink5 with 8 ranks and small messages, this is
precisely our 19.6%-of-C1 collective bill. Combined with the arrival-skew number (47% of that
19.6%), the ordering is: (a) reduce skew by balancing the pre-collective work, (b) shorten the
collective with a one-shot algorithm below a byte threshold.

### 8.2 The operational talk — Ye (Charlotte) Qi, InfoQ

`[verified]` https://www.infoq.com/presentations/llm-meta/ (summary also at
https://www.zenml.io/llmops-database/scaling-llm-inference-infrastructure-at-meta-from-model-runner-to-production-platform)

This is the closest thing published to a manual for our exact tension.

- **The 10× reality gap.** "It's common to lose 50% of effective FLOPS at earliest kernel
  benchmarking; combining latency bounds and buffers, 10x loss is common." Peak FLOPS is not a
  planning input.
- **Latency budget decomposition** for a 70B-class product path: 75 ms network round trip (CA↔NY),
  75 ms from "naive host selection, health check, load balancing", 150 ms multimodal image
  downloads, 400 ms+ business logic (safety, search, function calling). `[inferred]` **This is the
  #1 item on our steal list.** AA measures from GCP us-central1-a over the public internet. Every
  millisecond of placement, TLS, LB and health-check overhead is on our TTFT.
- **Disaggregation is not free**: "another 50 to 100 milliseconds to your TTFT if you are doing
  disagg", because hundreds of MB of KV must move; TCP/IP commonly adds 50–100 ms.
- **Hierarchical KV cache**: HBM for shared system prompts, DRAM for active user chat history
  (touched every minute), flash for less-engaged users. Result `[reported]`: **"over 50% reduction
  for both latency and capacity"**, described as a lossless optimisation.
- **Autoscaling is a shard-placement problem.** QPS-based autoscaling fails because the bottleneck
  moves between compute-bound and bandwidth-bound; they use "a deployment solver that actually
  treats autoscaling as a shard placement problem." (Compare Databricks' model units, §2.1 — two
  independent groups reaching the same conclusion that request counts are the wrong unit.)
- **Blast radius.** With context parallelism over 40 GPUs, "40 GPUs in one partition will take down
  your entire process group"; with ~3% random GPU failure plus maintenance, risk compounds
  exponentially. `[inferred]` Argues for the smallest parallel group that meets the SLO, not the
  largest that fits — relevant to any temptation to go beyond TP8 on our box.
- **Quantisation caution.** "When you release the quantization to production, some of your customers
  are just showing up and saying, something is not working." MMLU-equivalent scores do not imply
  product-equivalent quality; they mandate product-specific evals and slow rollout. `[inferred]`
  Direct warning for our NVFP4 build.
- **Silent failure.** "LLMs are probabilistic models. It's possible you have something horribly
  wrong, but the result comes out still decently correct." They apply continuous benchmarking and
  CI/CD discipline to stop "models from getting dumber over time."
- **Tail deployments.** "90% people focus on one, two, three head models" but "there are so many
  tail deployments than you can imagine, and collectively, they might consume even more GPUs than
  your main model."

### 8.3 Adjacent: Meta Adaptive Ranking Model

`[verified]` https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/
Not an LLM, but the kernel discipline transfers: O(1T) parameters served within an **O(100 ms)**
budget at **35% MFU** across multiple hardware types. Techniques: **selective FP8 — "deploys FP8
only in layers with high precision-loss tolerance", chosen by micro-benchmark**; operator fusion of
thousands of small ops; **grouped GEMM for horizontal fusion**; Top-K reduced from O(N log N) to
O(N) with a GPU-native kernel; "In-Kernel Broadcast optimization, which shares request-level
embeddings across ad candidates directly within the GPU kernel."

`[inferred]` "Selective FP8 chosen per layer by micro-benchmark" is a better default than a uniform
NVFP4/FP8 build. Our dense GEMM is 37.1% of C1 — a per-layer precision search over the dense stack,
scored on both latency and a product eval, is a well-supported approach.

---

## 9. OpenAI

Nothing about serving architecture is disclosed. What *is* disclosed is API-surface behaviour, and
it leaks a surprising amount about the caching implementation.

### 9.1 Prompt caching

`[verified]` https://developers.openai.com/api/docs/guides/prompt-caching

| Property | Disclosed value |
|---|---|
| Minimum prefix | 1,024 tokens (strict for GPT-5.6+; 1,024–2,048 with inconsistent behaviour just above 1,024 on earlier models) |
| Granularity | **128-token increments** on earlier models; explicit breakpoints on GPT-5.6+ |
| Routing | "based on `prompt_cache_key`, with a hash of the initial prefix of the prompt as a secondary key" |
| Lifetime | GPT-5.6+: 30-minute exact TTL via `prompt_cache_options.ttl`. Earlier: 5–10 min of inactivity, max ~1 hour in memory; extended retention up to 24 h |
| Invalidation | Any change before the breakpoint — tool definitions, schemas, message ordering, images, content |
| Discount | Cached input at **0.1×**; cache writes on GPT-5.6+ at **1.25×** |
| Traffic requirement | ~**15 requests per minute** per `prompt_cache_key` to avoid misses |

`[inferred]` Three design facts fall out. (1) There is a **sticky-routing layer keyed by a
client-supplied string with the prefix hash as a tiebreak** — i.e. cache affinity is explicit and
client-steerable, not inferred. (2) The **15 rpm floor per key** tells you the cache is per-replica
and evicted on inactivity, so the router is balancing affinity against replica load exactly as
llm-d found necessary (§7.2). (3) 128-token granularity implies **block-aligned prefix hashing**, so
block size and tokenisation boundaries are part of the cache design.

### 9.2 Batch API and latency tiers

`[verified]` https://developers.openai.com/api/docs/guides/batch — 24-hour completion window; **50%
discount**; a **separate pool of significantly higher rate limits** that does not consume standard
per-model limits; up to 50,000 requests and 200 MB per batch; up to 2,000 batches queued per hour.
**The docs do not disclose why batch is cheaper** — no mechanism is given.

`[verified]` https://developers.openai.com/api/docs/guides/flex-processing — flex is priced at Batch
API rates, "slower response times and occasional resource unavailability", may return
**429 Resource Unavailable with no charge**, default 10-minute timeout with a 15-minute
recommendation. `developers.openai.com/api/docs/guides/priority-processing` returns 404
`[verified]`.

`[verified]` https://developers.openai.com/api/docs/guides/rate-limits — usage tiers are
spend-based; limits on RPM/RPD/TPM/TPD/IPM plus audio minutes and queued batch tokens; **no capacity
or latency guarantees stated anywhere.**

`[inferred]` The economically interesting disclosure is the *unpriced* one: flex returns 429 free of
charge rather than queueing. That is a shed-not-queue policy identical in spirit to GKE Inference
Gateway's negative-priority dropping (§7.1). Three independent organisations converge on shedding as
the way to protect P50 at high utilisation.

---

## 10. Anthropic

No inference-architecture post exists on `anthropic.com/engineering` `[verified — I enumerated the
index]`. What is disclosed comes from API docs and two postmortems, and the postmortems are
unusually specific about serving.

### 10.1 Disclosed serving facts, from the September 2025 postmortem

`[verified]` https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues

- **"We deploy Claude across multiple hardware platforms, namely AWS Trainium, NVIDIA GPUs, and
  Google TPUs."** Serving the same model on three accelerator families is itself the headline
  architectural fact.
- **Context-window routing with sticky sessions.** A misconfiguration sent short-context requests to
  servers configured for the 1M-token context window. Because routing is sticky, "once a request was
  served by the incorrect server, subsequent follow-ups were likely to be served by the same
  incorrect server." A routine load-balancing change on 29 August took affected traffic from **0.8%
  to 16% of Sonnet 4 requests** at peak. `[inferred]` Confirms a *fleet segmented by context-window
  configuration* plus sticky session routing — the same shape as Databricks' stateful sessions and
  OpenAI's `prompt_cache_key`. And it demonstrates the failure mode: **stickiness turns a transient
  misroute into a persistent one**, and a load-balancer change amplified it 20×.
- **A mixed-precision compiler bug.** "Models compute probabilities in bf16 ... the vector processor
  is fp32-native, so the TPU compiler (XLA) can optimize runtime by converting some operations to
  fp32." The **approximate top-k** operation, run at inconsistent precision, would occasionally
  "drop the highest probability token entirely." `[inferred]` This is the single most relevant
  disclosed item for us: we run **approximate top-k in the DSA indexer** and **NVFP4/FP8 mixed
  precision**. A precision-dependent top-k that silently drops the argmax is a class of bug our
  stack can have, and it will not show up in throughput metrics. Add an exact-vs-approximate top-k
  agreement check to CI.
- **Detection failure, stated plainly.** Evaluations "simply didn't capture the degradation users
  were reporting"; privacy controls limiting engineer access to user interactions "prevents
  engineers from examining the problematic interactions needed to identify or reproduce bugs."
  Committed fixes: more sensitive evals that distinguish working from broken implementations,
  **continuous production monitoring rather than periodic testing**, and better debugging tooling.

### 10.2 The April 2026 postmortem — a caching bug as a cost bug

`[verified]` https://www.anthropic.com/engineering/april-23-postmortem
A `clear_thinking_20251015` header with `keep:1` cleared reasoning on **every** turn instead of once,
invalidating the cached prefix each turn and causing continuous cache misses — surfacing to users as
"usage limits draining faster than expected." It survived "human and automated code reviews, unit
tests, end-to-end tests, automated verification, and dogfooding."

`[inferred]` A general lesson: **cache hit rate is a first-class production metric, not a
performance nicety.** A silent drop to 0% is invisible in latency dashboards (it looks like longer
prefills) but shows up immediately in cost and in P50 TTFT. Alert on hit rate directly.

### 10.3 Prompt caching design, as disclosed

`[verified]` https://platform.claude.com/docs/en/build-with-claude/prompt-caching

| Property | Disclosed value |
|---|---|
| Breakpoints | Up to **4** explicit `cache_control` breakpoints per request |
| Prefix order | `tools` → `system` → `messages` |
| Minimum cacheable | 512 tokens (Opus 5, Fable 5, Mythos 5); 1,024 (Opus 4.8, Sonnet 5/4.6/4.5); 2,048 (Mythos Preview, Opus 4.7, Haiku 3.5); 4,096 (Opus 4.5/4.6, Haiku 4.5) |
| TTL | 5 min default; 1 h optional. Longer TTLs must appear **before** shorter ones in the same request |
| Pricing | write 1.25× (5 min), 2× (1 h); read/refresh **0.1×** |
| Lookback | **at most 20 blocks** backward from a breakpoint when searching for a prior cache entry |
| Concurrency | "A cache entry only becomes available after the first response **begins**" — parallel requests must wait for the first response to get hits |
| Isolation | Workspace-level on the Claude API, Claude on AWS and Microsoft Foundry; organisation-level on Bedrock and Google Cloud |
| TTL clock | Measured from the **start** of the writing/reading request; generation time counts against the TTL |

Invalidation is enumerated per-scope: tool definitions invalidate everything; web-search/citations/
speed toggles invalidate only the `tools` prefix; tool choice invalidates `tools`+`system`; images
added/removed invalidate `tools`+`system`; thinking/effort settings are model-specific.

`[inferred]` The **20-block lookback** and the **write-only-at-your-breakpoint** rule together
describe a block-indexed radix/prefix structure with a bounded search, which is what a production
prefix cache actually looks like. The "cache available only after the response begins" note reveals
that the cache entry is committed at generation start, not completion — a detail worth matching in
our own implementation so that a burst of parallel identical requests does not stampede.

### 10.4 Service tiers — the disclosed price of an SLO

`[verified]` https://platform.claude.com/docs/en/api/service-tiers
Three tiers: Priority, Standard, Batch. Priority Tier commitments are **no longer sold** (existing
contracts continue). A commitment = input tokens/minute + output tokens/minute + duration
(1/3/6/12 months) + a **specific model version**. **"Priority Tier targets 99.5% uptime."** Requests
beyond committed capacity fall back to standard. Burndown accounting mirrors pricing: cache reads
0.1 tokens/token, 5-min cache writes 1.25, 1-hour writes 2.00, US-only inference 1.1×. Requests
assigned Priority "pull from both the Priority Tier capacity and the regular rate limits."

`[inferred]` Two structural lessons. (1) Reserved capacity is denominated in **tokens per minute
split by input and output**, not requests — the same conclusion as Databricks' model units and
Meta's shard-placement framing. (2) Committing to a **specific model version** is the honest
admission that per-token capacity is not portable across model versions, because the serving
configuration changes with the model. Any internal SLO we offer should be versioned the same way.

### 10.5 Infrastructure variance, quantified

`[verified]` https://www.anthropic.com/engineering/infrastructure-noise
On Terminal-Bench 2.0, the gap between the most- and least-resourced container setups was **6
percentage points**; infrastructure error rates fell from **5.8% under strict (1×) resource
enforcement to 0.5% uncapped**. They note "time of day" affects pass rates "due to API latency
variations", alongside cluster health, concurrency and egress bandwidth.

`[inferred]` Directly relevant to an AA run scored as a **P50 over a trailing 72 hours**: time-of-day
variance in our own serving is inside the measurement window. If our box is shared or the network
path is congested at peak, the P50 we get is not the P50 we can produce. Measure our own diurnal
variance before attributing a leaderboard gap to the engine.

---

## 11. Adjacent data points on the same model (useful calibration)

Not in my assigned set, but they are measuring the same model on the same class of hardware and the
numbers frame our target. Flagged so the rest of the team can cross-reference.

**vLLM on 24× B300, GLM-5.2 NVFP4, PD-disaggregated** `[verified]`
https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd — 4 prefill nodes (16 GPUs, TP1 DP4 EP) +
1 decode node (8 GPUs, TP1 DP8 EP). SLAs: mean TTFT ≤ 2.5 s, mean TPOT ≤ 20 ms. TPOT went ~40 ms →
~17 ms over the optimisation campaign. Concurrency ~700 at 8K input down to ~25 at 256K.

Ranked optimisations, and these map onto our hotspots:

1. **Speculative padding on decode** — largest single win, ~40 ms → ~22 ms TPOT, by padding new
   requests to match existing batch shapes so MTP does not fall back to a mixed batch.
2. **Model Runner V2** — ~11% TPOT, including prefill-metadata kernel warmup, **local argmax
   reduction for multi-GPU MTP**, and dynamic speculative-length CUDA-graph support.
3. **`--all2all-backend=flashinfer_nvlink_two_sided`** — ~4% TPOT.
4. `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` to avoid prefill graph compilation.

Other flags named: `VLLM_USE_V2_MODEL_RUNNER=1`, `--max-num-batched-tokens 1024`,
`--speculative-config='{"method":"mtp","num_speculative_tokens":1}'` for prefill and `3` for decode,
`--gpu-memory-utilization` 0.92 prefill / 0.90 decode, NixlConnector for KV transfer. They chose
TP1 DP4 EP for prefill over the higher-throughput TP1 DP2 EP **specifically to preserve KV capacity
for the 1M context**. The **`IndexerCache` / `index_share_for_mtp_iteration`** work
(vllm-project/vllm PR #44420) reuses DSA Top-K indices across MTP draft steps; LongBench V2 scored
64.01 with sharing on. They call out monitoring **MTP acceptance rate and mean accepted length as
first-class metrics** and report linear host-memory growth in long-running stability tests.

**Baseten, GLM-5.2 on Blackwell NVFP4** `[reported]`
https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/ — NVFP4 converted from
FP8 weights with NVIDIA ModelOpt, validated on BFCL as "roughly equivalent" to FP8. 280+ tok/s as
measured by AA on 2026-06-22; 800 ms average TTFT; 7.9 s average time-to-first-*answer* token
(7.1 s reasoning + 0.8 s prefill) — a clean illustration of why AA's TTFT column reads in seconds.
Named techniques: shared-DSA support in their engine, **KV-aware routing via NVIDIA Dynamo**, PD
disaggregation ("2× higher tokens per second" vs aggregated), and GLM-5.2's MTP layer.

**Independent benchmark, B200 vs B300 on GLM-5/5.1 (744B), FP4, 8k/1k** `[reported, with caveats]`
https://inferencex.semianalysis.com/compare/glm-5-1-b200-vs-b300

| Interactivity | B200 tok/s/chip | B300 tok/s/chip | B200 $/M | B300 $/M |
|---|---|---|---|---|
| 37 tok/s/user | 1631.4 | 1269.2 | 0.295 | 0.495 |
| 63 tok/s/user | 1121.2 | 941.3 | 0.429 | 0.667 |
| 88 tok/s/user | 780.1 | 683.0 | 0.616 | 0.919 |

Caveats I must state: the page says data is "interpolated from real benchmark data"; it does not
disclose the serving framework, TP/EP config, or whether speculative decoding was on; and the cost
column depends entirely on assumed rental prices. **Do not treat "B200 beats B300" as established.**
It is, however, a useful counterweight to the claim that the Databricks lead is explained by B300
alone.

---

## Techniques ranked by transferability to our stack

Scoring: **Fit** = how directly it applies to 8×B200 / GLM-5.2 / SGLang fork / TP8 / EAGLE 3-1-4.
**Evidence** = strength of the published number. **Effort** = engineering cost.

| Technique | Source | Fit | Evidence | Effort | Which objective |
|---|---|---|---|---|---|
| Remove network/LB hops; place near GCP us-central1-a | Meta (75 ms + 75 ms decomposition) | High | Strong (measured, production) | Low | Single-stream (TTFT) |
| Drop tree-structured EAGLE drafts; test single-token | Perplexity ("up to 50%" attention regression from full-sequence masks) | High | Medium (no absolute numbers given) | Low | Single-stream |
| Share DSA top-k indices across draft steps | vLLM `IndexerCache` (PR #44420), LongBench V2 64.01 | Very high (same model) | Strong | Medium | Both |
| Speculative padding on decode | vLLM GLM-5.2 on B300 (40 → 22 ms TPOT) | Very high (same model) | Strong | Medium | Both, esp. C1→C16 cliff |
| Tile-aligned shapes everywhere (chunk, draft length, MoE group) | Sarathi-Serve (257 vs 256 → +32% prefill) | High (dense GEMM 37.1%) | Strong | Low to audit | Both |
| Token budget derived from a TBT SLO by one-time profiling | Sarathi-Serve | High | Strong (OSDI'24, 2.6–6.3× capacity) | Medium | Cost per user |
| One-shot direct-load allreduce below a byte threshold | Meta DDA-flat (O(N)→O(1), 10–50% vs RCCL decode) | High (collectives 19.6%) | Medium-strong | Medium | Single-stream |
| Signal host before NVLink writes, behind a grid barrier | Perplexity MoE send kernels | High (arrival skew 47% of collectives) | Medium (mechanism + relative results) | Medium-high | Single-stream |
| Stop pursuing dual-batch overlap at low batch | Perplexity Table 7 (degrades below batch ~48) | High | Strong | Low (a decision) | Single-stream |
| Query-head packing to fill the tensor-core tile in decode | Character.AI (`packQ`, up to 9.3×) | High (MLA decode, attention 10.9%) | Strong | High | Single-stream |
| Warp-specialised dequant fused into attention | Character.AI (producer/consumer) | High (NVFP4/FP8 builds) | Medium-strong | High | Both |
| Do not quantise the P·V matmul | Character.AI (chose Half INT8 over Full INT8) | Medium-high | Medium (quality reasoning, no numbers) | Low (a constraint) | Quality guardrail |
| Per-layer selective FP8/NVFP4 chosen by micro-benchmark | Meta MARM | High (dense GEMM 37.1%) | Medium (different model class) | Medium | Both |
| Prefix-cache-aware **+ load-aware** routing | llm-d (55% → 100% success; ITL 160 → 30 ms) | Medium-high (multi-replica only) | Strong | Medium | Cost per user |
| Stepped prefix truncation (`truncation_step`) | Character.AI Prompt Poet | High (any repeated-prefix traffic) | Medium (mechanism, no A/B numbers) | Low | Both |
| Model units: calibrated per-request cost for routing + autoscaling | Databricks (>80% GPU savings vs static peak) | Medium-high | Medium-strong | Medium-high | Cost per user |
| Priority classes with negative-priority shedding | GKE Inference Gateway; OpenAI flex 429-no-charge | Medium-high | Medium | Medium | Protects P50 at high util |
| Hierarchical KV cache HBM/DRAM/flash | Meta (>50% latency and capacity) | Medium | Medium (no config detail) | High | Cost per user |
| Gate remote KV fetch on prompt length (~1000-token break-even) | AWS Curvine (0.99× at 500 tokens) | Medium | Strong (explicit threshold) | Low | Cost per user |
| Layer-by-layer KV transfer triggered by a CUDA-graph-safe UVM watcher | Perplexity (transfer fully hidden at all lengths) | Medium (only if we disaggregate) | Strong | Medium-high | Cost per user |
| HND (heads-before-pages) KV layout for single-write transfers | Perplexity | Medium (only if we disaggregate) | Medium | Medium | Cost per user |
| Rust, not Python, on any GPU-progress callback path | Perplexity Table 4 (Python max 3325 µs) | High | Strong | Low-medium | Single-stream tail |
| Health checks at top scheduling priority | Databricks (several/week → zero false failures) | High | Medium-strong | Low | Availability |
| Cache hit rate as an alerted production metric | Anthropic April 2026 postmortem | High | Medium (incident evidence) | Low | Cost per user |
| Exact-vs-approximate top-k agreement check in CI | Anthropic XLA/bf16 postmortem | Very high (DSA indexer does approx top-k) | Medium (incident evidence) | Low | Quality guardrail |
| Asymmetric power capping: full for prefill, capped for decode | Splitwise (token phase flat at ~70% TDP) | Medium (only if power-limited) | Strong | Medium | Cost per user |
| Constant forward-pass token budget (concavity argument) | DeepSpeed-FastGen SplitFuse (P95 3.7× lower) | Medium-high | Strong | Medium | Cost per user |
| Measure with real text, variable lengths, ITL including TTFT | Anyscale | High | Strong (methodology) | Low | Both (measurement) |
| Track MTP/EAGLE acceptance rate and mean accepted length as SLIs | vLLM GLM-5.2 | Very high | Medium | Low | Both |
| **Anti-pattern: PD disaggregation for single-stream latency** | Perplexity (~100 ms), Meta (50–100 ms) | High | Strong | — | Avoid for C1 |
| **Anti-pattern: dispatch-only communication overlap** | Perplexity (0.6% at batch 128) | High | Strong | — | Avoid |
| **Anti-pattern: w4a16 weight-only quantisation** | Perplexity (7% perplexity degradation, rejected) | Medium | Strong | — | Avoid |
| **Anti-pattern: routing on request counts or queue depth alone** | Databricks, Meta, llm-d (three independent) | High | Strong | — | Avoid |

---

## What I could not source

Stated plainly, per the honesty rules:

1. **Databricks' actual GLM-5.2 optimisation.** No first-party technical post exists. The claimed
   392 tok/s and the H200→B300 story come from LinkedIn and a third-party Medium article
   respectively. Their AA-measured 336.5 tok/s is verified; the *how* is not.
2. **Azure/Microsoft Blackwell serving guidance.** Nothing found on the Azure AI/ML or Compute blog
   indexes about GB200/GB300/ND-v6 inference configuration or numbers.
3. **AWS Blackwell (P6/P6e) LLM serving guidance.** Nothing found. AWS's most recent relevant
   inference post is the Curvine tiered-KV-cache work, run on G6e instances.
4. **Google Blackwell serving guidance.** Nothing found; their published inference work is TPU- and
   GKE-flavoured.
5. **OpenAI's serving architecture.** Nothing disclosed. `openai.com/index/scaling-chatgpt/` 404s;
   Evan Morikawa's personal site has no such post. I excluded all secondary speculation.
6. **Anthropic's serving architecture.** No engineering post. The three hardware platforms, the
   sticky context-window routing, and the bf16/fp32 approximate-top-k bug are the only disclosed
   internals, all from postmortems.
7. **Perplexity's speculative decoding results.** They describe the design in detail but publish no
   acceptance rates, speedups, or latency numbers.
8. **Character.AI's attention kernels.** Described in detail; the FlashAttention-3 fork is not
   released.
9. **Meta's DDA and CP implementations.** No open-source artifact named in the post.
10. **Exact per-bar values in Perplexity's Figure 9/10** (MoE decode/prefill latency). I read the
    bar labels out of the PDF text layer and the series-to-value mapping is my reading, not a table.
    The textual claims around them are unambiguous and I have relied on those.
11. **llm-d's intelligent-scheduling hardware and model.** The post gives dramatic numbers but does
    not state what was running.
12. Independent reproduction of *any* of the vendor throughput claims in this file. None exists that
    I could find, with the partial exception of Artificial Analysis (third-party, but a black-box
    API measurement, not a controlled hardware comparison) and SemiAnalysis InferenceX (interpolated,
    undisclosed framework).

---

## Sources

Read in full as raw text:

- Perplexity AI, "fabric-lib: RDMA Point-to-Point Communication for LLM Systems", MLSys'26 —
  https://arxiv.org/abs/2510.27656 (v2, 13 Apr 2026)
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI'24 —
  https://arxiv.org/abs/2403.02310

Perplexity AI:

- https://www.perplexity.ai/hub/blog/disaggregated-prefill-and-decode
- https://www.perplexity.ai/hub/blog/lower-latency-and-higher-throughput-with-multi-node-deepseek-deployment
- https://www.perplexity.ai/hub/blog/accelerating-sonar-through-speculation
- https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication
- https://www.perplexity.ai/hub/blog/high-performance-gpu-memory-transfer-on-aws
- https://www.perplexity.ai/hub/blog/turbocharging-llama-2-70b-with-nvidia-h100
- https://github.com/perplexityai/pplx-garden
- https://github.com/ppl-ai/pplx-kernels (deprecated)
- https://developer.nvidia.com/blog/spotlight-perplexity-ai-serves-400-million-search-queries-a-month-using-nvidia-inference-stack/

Databricks / Mosaic:

- https://www.databricks.com/blog/reliable-llm-inference-scale
- https://www.databricks.com/blog/ai-serving-platform-adapts-your-model
- https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
- https://www.databricks.com/blog/Integrating-NVIDIA-TensorRT-LLM
- https://www.databricks.com/blog/accelerated-dbrx-inference-mosaic-ai-model-serving
- https://artificialanalysis.ai/models/glm-5-2/providers
- https://artificialanalysis.ai/providers/databricks
- https://www.linkedin.com/posts/mateizaharia_databricks-delivers-the-fastest-inference-activity-7475939456869457920-SHs0
- https://dbxdev.medium.com/392-tokens-per-second-how-databricks-and-the-b300-just-reset-the-ai-speed-limit-7d84285b7e6e (third-party, no primary source)

Character.AI:

- https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/
- https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/
- https://blog.character.ai/inside-kaiju-building-conversational-models-at-scale/
- https://github.com/character-ai/prompt-poet

Anyscale / Ray:

- https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference
- https://github.com/ray-project/llmperf
- https://www.anyscale.com/blog/ray-vllm-prefill-decode-disaggregation-amd-mi325x-67-percent-savings
- https://www.anyscale.com/blog/high-performance-distributed-inference-ray-serve-llm-vllm-google-kubernetes-gke

Microsoft:

- https://arxiv.org/html/2311.18677v2 (Splitwise, ISCA'24)
- https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/
- https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md
- https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-fastgen
- https://github.com/microsoft/sarathi-serve

AWS:

- https://aws.amazon.com/blogs/machine-learning/tiered-kv-cache-for-large-llms-on-amazon-sagemaker-hyperpod-with-curvine/

Google:

- https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-gke-inference-gateway
- https://github.com/kubernetes-sigs/gateway-api-inference-extension
- https://llm-d.ai/blog/intelligent-inference-scheduling-with-llm-d
- https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm

Meta:

- https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/
- https://www.infoq.com/presentations/llm-meta/ (Ye "Charlotte" Qi)
- https://www.zenml.io/llmops-database/scaling-llm-inference-infrastructure-at-meta-from-model-runner-to-production-platform
- https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/

OpenAI (API surface only):

- https://developers.openai.com/api/docs/guides/prompt-caching
- https://developers.openai.com/api/docs/guides/batch
- https://developers.openai.com/api/docs/guides/flex-processing
- https://developers.openai.com/api/docs/guides/rate-limits

Anthropic:

- https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
- https://www.anthropic.com/engineering/april-23-postmortem
- https://www.anthropic.com/engineering/infrastructure-noise
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://platform.claude.com/docs/en/api/service-tiers

Adjacent calibration:

- https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd
- https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/
- https://inferencex.semianalysis.com/compare/glm-5-1-b200-vs-b300
