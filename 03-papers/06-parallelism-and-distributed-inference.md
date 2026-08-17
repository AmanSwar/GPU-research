# Parallelism strategy for inference: TP, EP, PP, SP/CP, and disaggregation

## What this is

A survey of the literature on **how to split a large model across GPUs for
inference**, written against one specific target: 8× B200 SXM (SM100, 183 GB
HBM3e, NV18 all-to-all NVLink5/NVSwitch) serving a GLM-5.2-class MoE
(256 experts / 8 active, DSA sparse MLA, NVFP4 + FP8 builds, TP8, EAGLE 3-1-4),
at two operating points that fight each other: minimum single-stream latency and
maximum tok/s/GPU at concurrency.

Every paper cited here was fetched and read (abstract + method + evaluation
numbers). Results are labelled `[verified]` (I read the number in the paper /
measured it here), `[reported]` (author or vendor claim in text I read), or
`[inferred]` (my own arithmetic). Hardware and model size are stated for every
result, because a 1.9× on 8×A100 with Llama-2-70B is not a 1.9× on 8×B200 with a
671B MoE, and pretending otherwise is how you waste a quarter.

**A note on model shape.** GLM-5.2's config is not public. Where the arithmetic
needs concrete numbers I use the two closest *published* configs as brackets:

| | hidden | layers | routed experts | top-k | moe ffn | attention | KV/token/layer |
|---|---|---|---|---|---|---|---|
| **DeepSeek-V3.2-Exp** (`config.json`, fetched) | 7168 | 61 (3 dense) | 256 | 8 | 2048 | MLA, `kv_lora_rank`=512, `qk_rope`=64, `index_topk`=2048, `index_n_heads`=64 | 576 elem |
| **GLM-4.6** (`config.json`, fetched) | 5120 | 92 (3 dense) | 160 | 8 | 1536 | GQA 96q/8kv, head_dim 128 | 2048 elem |

Our stated model — 256 experts / 8 active with DSA sparse MLA — is architecturally
a DeepSeek-V3.2 descendant, so I weight that shape more heavily. GLM-4.6's 92
layers is the pessimistic bracket for collective *count*.

---

## Bottom line for our system

Ranked by expected effect on our measured profile (dense GEMM 37.1%, collectives
19.6% of which 47% is rank-arrival skew, MoE GEMM 19.4%, attention 10.9%, DSA
indexer 5.8%; 365 tok/s single-stream = 2.740 ms/token).

1. **Collectives are a latency problem, not a bandwidth problem, and the fix is
   to reduce the *number* of synchronisation points — not to make each one
   faster.** At batch 1 a TP8 all-reduce on a 7168-hidden model moves **14.0 KiB**
   of payload; the ring wire time on NVLink5 is **0.028 µs** `[inferred]`. The
   measured floor is 3–9 µs. You are paying 100–300× the transfer cost in
   synchronisation. Ceiling: removing *all* collective time takes us from 365 →
   **454 tok/s** (1.24×) `[inferred]`. That alone does not reach TileRT's 500.

2. **Attack rank-arrival skew before you attack transfer.** 47% of our collective
   time (252 µs/token, 9.2% of total budget) is ranks idling at the barrier.
   Removing only the skew is worth **402 tok/s (1.10×)** `[inferred]` and is the
   cheapest win in the whole document. Sources of skew on our config: MoE expert
   load imbalance across the 8 ranks (fix: EPLB — 1.49× prefill / 2.54× decode
   measured by SGLang on 96×H100 for DeepSeek-V3 `[verified]`), DSA indexer
   variance, and CPU-side launch jitter. This is a load-balancing and
   CUDA-graph-capture problem, not a NCCL problem.

3. **Single-batch overlap (SBO) at tile/expert granularity is the right overlap
   family for us, and we already have the machinery.** Two-batch overlap is
   *structurally unavailable*: our own `server_args.py` rejects
   `--enable-two-batch-overlap` when `index_topk_freq > 1`, because the TBO op
   path does not propagate DSA topk indices across shared layers `[verified,
   local source]`. And TBO is a throughput technique that costs latency at C1 —
   SGLang measured **−27%** at 32 tokens/device `[verified]`. SBO in our tree
   already runs the DeepEP combine on a 32-SM side stream on Blackwell
   (`communicate_num_sms = 32 if is_blackwell()`) with the down-projection GEMM
   signalling per-expert completion `[verified, local source]`. Extend the same
   signal-driven pattern to the attention output projection and the dense-layer
   all-reduce. Literature support: Comet 1.96× per MoE layer / 1.71× end-to-end
   on 8×H800 `[verified]`; Flux 63–72% average communication-overlap efficiency
   on NVLink `[verified]`.

4. **Move to DP-attention + EP8 if you have not already, and kill the vocab
   all-gather.** DP attention removes the per-layer attention all-reduce entirely
   (each rank owns its own tokens' KV) and replaces the MoE all-reduce with
   dispatch/combine all-to-all, which for a top-8-of-256 routing touches only
   **5.25 of 8 ranks in expectation** `[inferred]` rather than all 8. SGLang's
   memory analysis says DP wins over TP whenever
   `TP > sqrt(N_param / ((1+k)·N_hidden_state))`, which for decode at 128
   tokens/device gives optimal TP ≤ 6 `[verified]` — i.e. TP8 is already past the
   crossover. Flags: `--enable-dp-attention`, `--ep-size 8`,
   `--moe-a2a-backend deepep`, `--enable-dp-lm-head` `[verified from SGLang
   `server_args.py`]`.

5. **Speculative decoding is, among other things, a communication-amortisation
   technique — count it that way.** EAGLE 3-1-4 at 3.09× acceptance divides the
   fixed per-forward collective cost by 3.09 across emitted tokens `[inferred]`.
   This is why spec decoding is worth *more* on TP8 than the raw compute
   arithmetic suggests, and why increasing draft depth has a second-order payoff
   we are probably not modelling. Corollary: any overlap work that assumes
   batch-1 shapes must be evaluated at the *verify* batch (4 tokens), not 1.

6. **Do not add pipeline parallelism inside the node, and do not add TP beyond
   8.** PP bubble is `(p-1)/m` `[verified, Megatron SC'21]`; at single-stream
   decode `m = 1`, so PP8 idles 7/8 of the machine. And the OSU characterisation
   paper measured TPOT going from 0.86 ms (TP4, intra-node) to **11.56 ms**
   (TP8 crossing a node boundary) on H100 `[verified]` — a 13× regression from
   one hop off NVLink. TP is an NVLink-domain-only technique. Full stop.

7. **When we grow past one node, grow with EP over the fabric — not TP.**
   DeepSeek runs decode at **EP320 across 40 nodes with TP only 4**
   `[verified, DeepSeek-V3 report]`; SGLang reproduced EP72/9 nodes at 22.3k
   output tok/s/node, **5.2× vs TP16** `[verified]`. NVIDIA measured EP32 giving
   **1.8× higher per-GPU output tok/s than EP8** at a 100 tok/s/user SLO for
   DeepSeek-R1 on GB200 NVL72 `[reported]`. On plain 8-GPU nodes with IB, apply
   node-limited routing (≤4 nodes per token) as DeepSeek does `[verified]`.

8. **PD disaggregation is a *throughput* play for us, not a latency play — and
   our KV transfer is nearly free because of MLA.** MLA at 61 layers costs
   **34.3 KiB/token in FP8**; an 8K prompt's KV is 288 MB, which crosses NVLink5
   in **0.32 ms** and 8×400 Gb IB in **0.72 ms** `[inferred]`. The same prompt
   under GLM-4.6-style GQA costs 1.54 GB and 3.86 ms over IB — 5.4× worse. So
   disaggregation is much more attractive on the MLA models (GLM-5.2, DeepSeek
   V4, Kimi K3) than on GQA models. NVIDIA's large simulation study found
   disaggregation wins for **prefill-heavy traffic and models >10B**, and loses
   to chunked prefill for generation-heavy traffic `[verified]`.

9. **Our prefix-cache hit rate changes which ring-attention variant is correct
   for long prefill.** Meta's context-parallelism paper switches between pass-KV
   and pass-Q based on cache-miss rate, with threshold `T/(T+P) ≥ 2·(N_KV/N_H)`
   `[verified]`. At our 1.54× prefix-caching benefit the hit rate is high enough
   that pass-Q is the right variant if we ever do multi-node long-context
   prefill. Not urgent at 8 GPUs.

10. **Sequence/context parallelism buys us nothing at 8 GPUs today.** Ulysses is
    capped at `P ≤ n_heads` `[verified]` (fine, we have 64–128), but at TP8 in one
    NVLink domain the all-gather/reduce-scatter of Megatron-SP costs the same
    bandwidth as the all-reduce it replaces `[verified, Korthikanti et al.]` and
    only buys activation memory, which we are not short of. Revisit only for
    >256K contexts or multi-node.

---

## The arithmetic for our case

This section is the point of the assignment. All numbers `[inferred]` from the
published configs above and NVLink5 specs (18 links × 50 GB/s per direction =
**900 GB/s unidirectional, 1.8 TB/s bidirectional per GPU**; NVSwitch
non-blocking all-to-all).

### Bytes per all-reduce, TP8, pure tensor parallelism

Megatron TP puts **exactly two all-reduces in the forward pass of every
transformer layer** `[verified, Shoeybi et al. 2019]`: one after the attention
output projection (row-parallel), one after the FFN down-projection
(row-parallel). Payload is the residual-stream activation, `B × hidden × 2` bytes
in BF16.

**DeepSeek-V3.2 shape (hidden 7168, 61 layers) → 122 all-reduces per forward:**

| batch | AR payload | ring wire/rank | ring transfer | one-shot wire/rank | one-shot transfer |
|---|---|---|---|---|---|
| 1 | 14.0 KiB | 24.5 KiB | **0.028 µs** | 98.0 KiB | 0.112 µs |
| 4 (EAGLE verify) | 56.0 KiB | 98.0 KiB | 0.112 µs | 392.0 KiB | 0.446 µs |
| 8 | 112.0 KiB | 196.0 KiB | 0.223 µs | 784.0 KiB | 0.892 µs |
| **64** | **896.0 KiB** | 1568.0 KiB | **1.784 µs** | 6272.0 KiB | 7.136 µs |
| 256 | 3584.0 KiB | 6272.0 KiB | 7.136 µs | 25088.0 KiB | 28.545 µs |

**GLM-4.6 shape (hidden 5120, 92 layers) → 184 all-reduces per forward:**
batch 1 = 10.0 KiB payload / 0.020 µs ring; batch 64 = 640.0 KiB / 1.274 µs.

### What NVLink5 latency floor that implies

Published and measured floors for small-message all-reduce:

| source | hardware | measurement | label |
|---|---|---|---|
| Demystifying NCCL (arXiv 2507.04786) | analysis | LL ≈ 1 µs/hop, LL128 ≈ 2 µs/hop, Simple ≈ 6 µs/hop | `[verified]` |
| nccl-tests issue #333 (user `GindaChen`) | 8×B200, NCCL 2.27 + symmetric memory | 2-GPU ≈ **9 µs**, 4-GPU ≈ **22 µs** small-msg AR | `[verified — user-measured, not NVIDIA]` |
| NVIDIA blog, quoted in that issue | 32×B200 | "expected ~6.3 µs" small-msg AR | `[reported, second-hand]` |

Take a realistic 3–5 µs exposed floor for a well-implemented one-shot
NVLS/multimem all-reduce inside a CUDA graph on 8 B200s:

| layers | AR/forward | @2 µs | @3 µs | @4 µs | @5 µs | @8 µs |
|---|---|---|---|---|---|---|
| 61 | 122 | 0.244 ms | 0.366 ms | **0.488 ms** | 0.610 ms | 0.976 ms |
| 92 | 184 | 0.368 ms | 0.552 ms | **0.736 ms** | 0.920 ms | 1.472 ms |

**This is the structural bound on TPOT.** At a 2.0 ms/token budget (= 500 tok/s,
TileRT's number), 61 layers of pure-TP8 all-reduce at a 4 µs floor consumes
**24% of the entire budget before a single byte of useful work**. At 92 layers it
is 37%. You cannot engineer your way out of that with a faster NCCL; you have to
remove synchronisation points or hide them behind compute.

### The latency/bandwidth crossover

Setting ring-all-reduce transfer time equal to the latency floor:

| shape | @3 µs floor | @4 µs floor | @5 µs floor |
|---|---|---|---|
| hidden 7168 | B ≈ 108 tok | **B ≈ 143 tok** | B ≈ 179 tok |
| hidden 5120 | B ≈ 151 tok | **B ≈ 201 tok** | B ≈ 251 tok |

**Below ~150 tokens per forward, TP all-reduce is 100% latency.** Our C64
operating point (64 concurrent streams × 4 verify tokens = 256 tokens/forward) is
*just past* the crossover; our C1 point (4 tokens) is three orders of magnitude
below it. Two completely different regimes, one config. This is the single most
important fact for the decision matrix at the end.

### Consistency check against our own profile

365 tok/s = **2.740 ms/emitted token**. Collectives 19.6% = **537 µs/token**;
skew 47% of that = **252 µs/token**.

Under the assumption of 2 collectives/layer and EAGLE acceptance 3.09, the
collectives per *emitted* token are `(2L + draft)/3.09` ≈ 41 (61 layers) or 60
(92 layers), implying **13.1 µs / 9.0 µs exposed cost per collective**
`[inferred]`. That is 2–4× above the achievable floor, and the gap is almost
exactly the reported 47% skew plus protocol overhead. If instead the engine runs
DP-attention + EP (4 collective-ish events per layer: dispatch, combine, and the
two DP boundary ops), the count roughly doubles and the implied per-event cost
lands at 5–7 µs — right at the NCCL/DeepEP floor. **Either way the conclusion is
the same: we are at the synchronisation floor, and only count reduction and
overlap move the number.**

### Amdahl on our profile

| intervention | ms saved | new ms | tok/s | gain |
|---|---|---|---|---|
| remove ALL collective time (upper bound) | 0.537 | 2.203 | **454** | 1.24× |
| remove rank-arrival skew only (47%) | 0.252 | 2.487 | 402 | 1.10× |
| remove 80% of collectives (Flux-class) | 0.430 | 2.310 | 433 | 1.19× |
| remove 50% of collectives | 0.268 | 2.471 | 405 | 1.11× |
| remove 30% (TokenWeave-class) | 0.161 | 2.579 | 388 | 1.06× |
| **collectives gone + 20% off dense GEMM** | 0.740 | **1.999** | **500** | 1.37× |
| collectives gone + 20% dense + 20% MoE GEMM | 0.847 | 1.893 | 528 | 1.45× |

**To match TileRT at 500 tok/s we must remove 0.740 ms = 27.0% of the current
budget.** Communication work alone cannot do it. Communication + a 20% dense-GEMM
improvement exactly hits it. That is the plan.

### MoE all-to-all sizing (EP8, top-8 of 256, hidden 7168)

With uniform routing, the expected number of *distinct* destination ranks for one
token's 8 experts on EP8 is `8·(1 − (7/8)^8) = ` **5.25** `[inferred]`.

| batch/rank | dispatch out (FP8) | transfer | combine in (BF16) | transfer |
|---|---|---|---|---|
| 1 | 36.8 KiB | 0.04 µs | 73.5 KiB | 0.08 µs |
| 4 | 147.0 KiB | 0.17 µs | 294.1 KiB | 0.33 µs |
| 64 | 2.30 MiB | 2.68 µs | 4.59 MiB | 5.35 µs |
| 256 | 9.19 MiB | 10.71 µs | 18.38 MiB | 21.41 µs |

DeepEP's own measured numbers on **SM100 (Blackwell) intranode NVLink EP8**, at
8K tokens/batch, 7168 hidden, top-8, FP8 dispatch / BF16 combine `[verified,
DeepEP README fetched]`:

| SMs | dispatch | combine |
|---|---|---|
| 64 (max) | 726 GB/s | 740 GB/s |
| 24 (min) | 643 GB/s | 675 GB/s |

Note that dropping from 64 SMs to 24 costs only 11% of dispatch bandwidth. **On
Blackwell you can buy back 40 SMs for compute at an 11% communication-bandwidth
cost** — that is the trade SBO is making when it pins the combine to 32 SMs.

### KV-cache transfer cost for PD disaggregation

| model shape | KV/token | 2K prompt | 8K prompt | 32K prompt |
|---|---|---|---|---|
| **MLA 61L, FP8** (`kv_lora`512 + rope 64) | **34.3 KiB** | 72 MB | 288 MB | 1.15 GB |
| — over NVLink5 (900 GB/s) | | 0.08 ms | **0.32 ms** | 1.28 ms |
| — over 8×400 Gb IB (400 GB/s) | | 0.18 ms | **0.72 ms** | 2.88 ms |
| — over 1×400 Gb IB (50 GB/s) | | 1.44 ms | 5.76 ms | 23.03 ms |
| **GQA 8kv×128, 92L, FP8** | **184.0 KiB** | 386 MB | 1.54 GB | 6.17 GB |
| — over 8×400 Gb IB (400 GB/s) | | 0.96 ms | **3.86 ms** | 15.44 ms |

Compare against DistServe's own estimate: a 512-token OPT-66B request is ~1.13 GB
of KV, and at 10 rps that is 90 Gbps, which they call negligible on an 800 Gbps
IB cluster; they measured KV transmission at **<0.1% of total latency** even for
OPT-175B `[verified]`.

---

## Technique family 1 — Tensor parallelism (Megatron)

| paper | lab | venue+year | hardware | headline | production? |
|---|---|---|---|---|---|
| Megatron-LM: Training Multi-Billion Parameter LMs Using Model Parallelism (arXiv 1909.08053) | NVIDIA (Shoeybi et al.) | arXiv 2019 | 512× V100 | 15.1 PFLOPS, 76% scaling efficiency vs single-GPU `[verified]` | **Universal** — every engine |
| Efficient Large-Scale LM Training on GPU Clusters Using Megatron-LM (arXiv 2104.04473) | NVIDIA/Stanford/MSR (Narayanan et al.) | SC 2021 | up to 3072× A100 | bubble `(p−1)/m`, interleaved `(p−1)/(v·m)` `[verified]` | Yes |
| Reducing Activation Recomputation in Large Transformer Models (arXiv 2205.05198) | NVIDIA (Korthikanti et al.) | arXiv 2022 | 2240× A100, 530B GPT | 5× activation memory reduction; 54.2% MFU vs 42.1% `[verified]` | Yes (Megatron-LM, NeMo) |
| Characterizing Communication Patterns in Distributed LLM Inference (arXiv 2507.14392) | Ohio State (Xu, Panda et al.) | arXiv 2025 | 4× H100 + IB NDR400, vLLM 0.8.5 | TP8 across nodes: TPOT **11.56 ms** vs TP4 intra-node **0.86 ms** `[verified]` | n/a (measurement study) |
| Flash Communication (arXiv 2412.04964) | Meituan (Li et al.) | arXiv 2024 | not stated in abstract | >3× intra-node comm speedup, 2× TTFT `[reported]` | No — one paper |

### The mechanism, exactly

**MLP.** `Y = GeLU(X A) B`. `A` is split **column-wise** (`A = [A₁, A₂]`), `B`
**row-wise**. Column-first is forced by the nonlinearity: `GeLU(X₁A₁ + X₂A₂) ≠
GeLU(X₁A₁) + GeLU(X₂A₂)` `[verified, Shoeybi et al.]`, so a row-parallel first
GEMM would need a synchronisation *before* the activation. With column-then-row,
the only sync is one all-reduce after `B`.

**Attention.** Whole heads are assigned to ranks (so QKV projections are
column-parallel by construction), and the output projection is row-parallel →
one all-reduce.

**Per layer, forward:** 2 all-reduces. Backward adds 2 more (the `f` operator is
a no-op forward / all-reduce backward; `f̄` is the conjugate) `[verified]`. For
inference only the 2 forward ones matter.

**Communication volume**, from the SC'21 paper: TP moves
`8·b·s·h·(t−1)/t` bytes per layer versus PP's `b·s·h` point-to-point per
microbatch between adjacent stages `[verified]`. TP is roughly `8(t−1)/t ≈ 7×`
more traffic than PP per layer at t=8 — which is exactly why the paper's
recommendation is *"tensor model parallelism should generally be used up to
degree g when using g-GPU servers, and then pipeline model parallelism can be
used to scale up to larger models across servers"* `[verified]`.

### Sequence parallelism (Megatron-SP) — the free memory trick

Korthikanti et al. observe that the LayerNorm and Dropout regions *between* the
two TP-parallel blocks are replicated across ranks and waste activation memory.
SP shards those regions along the sequence dimension, and replaces:

- `g` (entering the TP region): **all-gather** forward / reduce-scatter backward
- `ḡ` (leaving the TP region): **reduce-scatter** forward / all-gather backward

The key claim, quoted: *"The communication bandwidth used for tensor parallelism
and tensor together with sequence parallelism are the same. Therefore, sequence
parallelism does not introduce any communication overhead."* `[verified]` — this
holds because a ring all-reduce **is** a reduce-scatter followed by an all-gather.

Activation memory per layer becomes `(sbh/t)(34 + 5as/(ht))` `[verified]`.

**Relevance to us:** SP is free bandwidth-wise but *doubles the number of
collective calls* (2 all-reduces → 2 all-gathers + 2 reduce-scatters). In a
bandwidth-bound regime that is a wash. **In our latency-bound regime at batch 1
it is a 2× regression on the thing that is actually killing us.** Do not enable
SP for low-latency decode. Enable it for prefill, where the batch is large and
the memory saving is real. This is a genuinely different answer for prefill and
decode, and is an argument for disaggregation on its own.

---

## Technique family 2 — Sequence / context parallelism

| paper | lab | venue+year | hardware | headline | production? |
|---|---|---|---|---|---|
| Ring Attention with Blockwise Transformers (arXiv 2310.01889) | UC Berkeley (Liu, Zaharia, Abbeel) | arXiv 2023 (listed at NeurIPS 2023) | 32× A100, TPUv4-1024 | 4096K ctx on 32×A100 (7B); 256–512× prior SOTA `[verified]` | Yes — in most CP impls |
| DeepSpeed-Ulysses (arXiv 2309.14509) | Microsoft (DeepSpeed) | arXiv 2023 | up to 256× A100, GPT 1.2/7/30B | 2.5× faster at 4× longer seq vs Megatron-SP; per-link volume `4Nh/P` vs `4Nh` `[verified]` | Yes — DeepSpeed, SGLang, vLLM |
| Striped Attention (arXiv 2311.09431) | MIT CSAIL (Brandon, Ragan-Kelley et al.) | arXiv 2023 | 8× A100, 16× TPUv4 | **1.45×** @256K on A100; **1.65×** @786K on TPUv4 `[verified]` | Partially — "zigzag" ring in most libs |
| USP: A Unified Sequence Parallelism Approach (arXiv 2405.07719) | Tencent (Fang, Zhao) | arXiv 2024 | 2×8 A800, L20 PCIe | 47% MFU LLAMA3-8B @208K on 2 nodes `[verified]` | Yes — yunchang/xDiT |
| Context Parallelism for Scalable Million-Token Inference (arXiv 2411.01783) | Meta (Yang et al.) | MLSys 2025 | 128× H100 / 16 nodes, Llama3 405B | **1M prefill in 77 s**, 93% parallelisation efficiency, 63% FLOPS util `[verified]` | Yes — Meta production |

### Ring Attention

Each device holds one query block permanently and passes KV blocks around a ring;
after `N−1` steps every device has seen every KV block. The send of block `i+1`
overlaps the attention compute on block `i`. The overlap condition, quoted from
the paper: block size `c ≥ F/B` where `F` = FLOPS per host and `B` = inter-host
bandwidth `[verified]` — derived from `4dc²/F ≥ 4cd/B`. Memory is `6bch` bytes
per layer, **independent of sequence length** `[verified]`.

On B200 with NVLink5: `F/B ≈ 2.25e15 / 900e9 ≈ 2500`, so blocks of ≥2500 tokens
hide the transfer `[inferred]`. That is easy at prefill and impossible at decode.

### Striped Attention — the causal-mask fix

Ring Attention with a causal mask is catastrophically imbalanced: on all rounds
after the first, some devices compute a fully-masked (zero-value) block while
others compute a fully-unmasked one, and *iteration latency is set by the
slowest device*, so the savings from masking are never realised `[verified]`. The
fix is to stripe: device `i` owns tokens `i, i+N, i+2N, …` instead of a
contiguous chunk, so **every** device sees a roughly triangular mask on **every**
round. Theoretical max speedup 1.85× (TPUv4) / 1.72× (A100); achieved **1.65×**
at 786K on 16 TPUv4 and **1.45×** at 256K on 8×A100 `[verified]`.

**This is the same class of bug as our 47% rank-arrival skew.** The lesson
generalises: in any collective-synchronised loop, the cost is `max` over ranks,
so *balancing* is worth more than *speeding up*.

### Ulysses vs Ring — the communication comparison that matters

Ulysses shards the sequence, then does an **all-to-all on Q, K, V** to convert
sequence-sharding into head-sharding, computes full-sequence attention on a
subset of heads locally, then a **second all-to-all** on the output to convert
back. Two all-to-alls per attention block `[verified]`.

| | collective | per-link volume | scaling |
|---|---|---|---|
| DeepSpeed-Ulysses | 2× all-to-all | `4Nh/P` | **constant** if N and P scale together `[verified]` |
| Megatron-SP | all-gather + reduce-scatter | `4Nh` | linear in N `[verified]` |
| Ring Attention | P2P ring | `4·O(b·s·d)` but overlapped | hidden if `c ≥ F/B` `[verified]` |

Ulysses' hard limit: **`P ≤ number of attention heads`** (or KV heads under
GQA) `[verified, USP]`. For DeepSeek-style MLA with 128 heads that is not
binding at P=8; for a GQA model with 8 KV heads it binds immediately at P=8.

USP composes the two on a 2D mesh — Ring across mesh columns, Ulysses across mesh
rows — precisely so the Ulysses degree can stay ≤ head count while total SP
degree grows `[verified]`. Their recommendation, quoted in effect: prefer DP over
SP whenever batch permits; use `ulysses-degree=8` on NVLink and mixed degrees on
PCIe `[verified]`.

### Meta's context parallelism — the only one designed for *inference*

Meta's paper is the important one for us because it separates the three
inference regimes:

- **Full prefill** (no cache): use **pass-KV** — ring the KV, because for GQA
  models KV is smaller than Q.
- **Persistent-KV prefill** (partial prefix-cache hit): choose dynamically. The
  threshold is `T/(T+P) ≥ 2·(N_KV/N_H)` where T = new tokens, P = persistent
  tokens `[verified]`. High hit rate ⇒ pass-Q, because ringing the full
  persistent KV would cost far more than ringing Q.
- **Decode**: use **pass-Q**, with a batched round-robin offset so KV shards stay
  balanced `[verified]`.

Numbers, Llama3 405B on H100 Grand Teton `[verified]`:

| config | 128K prefill | 1M prefill |
|---|---|---|
| 1 node, TP8 | ~42 s | — |
| 8 nodes, CP8 | 5.85 s | — |
| 16 nodes, CP16 | 3.8 s | **77 s** (93% par. eff, 63% FLOPS util) |

And the honest negative result: **CP does not help decode.** TTIT went from
46.26 ms (CP1+TP8) to 71.31 ms (CP4+TP8) `[verified]` — ring communication is
pure added latency when there is one token of work. They also report *similar
scalability on 100 Gb/s TCP as on 400 Gb/s RDMA* for long-context prefill
`[verified]`, which is a strong statement that prefill CP is compute-bound, not
network-bound.

**Verdict for us:** not now. Revisit if (a) we go multi-node and (b) contexts
exceed ~256K. Our `--attn-cp-size` / `--dcp-size` flags exist in the tree.

---

## Technique family 3 — Pipeline parallelism

| paper | lab | venue+year | hardware | headline | production? |
|---|---|---|---|---|---|
| Megatron SC'21 (arXiv 2104.04473) | NVIDIA | SC 2021 | 3072× A100 | bubble `(p−1)/m` → `(p−1)/(v·m)` interleaved `[verified]` | Yes (training) |
| DistServe (arXiv 2401.09670) | PKU + UCSD | OSDI 2024 | 32× A100-80GB | inter-op (PP) "almost linearly scales throughput" for decode; intra-op (TP) has diminishing returns `[verified]` | Concepts adopted widely |
| Mooncake (arXiv 2407.00079) | Moonshot AI + Tsinghua | arXiv 2024 (rev. 2025) | Kimi production cluster | chunked pipeline parallelism for long ctx; **75% more requests** in real workload `[verified]` | **Yes** — Kimi production |
| MegaScale-Infer (arXiv 2504.02263) | ByteDance + PKU | SIGCOMM 2025 | H20 / L40S / A100 | ping-pong PP between attention and FFN nodes; **1.90×** per-GPU throughput `[verified]` | ByteDance production |

### Why PP fails at low batch, quantitatively

Bubble fraction is `(p−1)/m` where `p` = stages, `m` = microbatches in flight
`[verified]`. Decoding a single stream gives `m = 1` and a bubble of `p−1 = 7` —
i.e. each GPU is busy 1/8 of the time and single-stream latency is *unchanged*
from TP1 while burning 8 GPUs. Interleaved 1F1B reduces the bubble to
`(p−1)/(v·m)` with `v` virtual stages `[verified]`, but `v` cannot help when
`m = 1` because the dependency is genuinely serial.

PP's *only* wins for inference are:

1. **Memory**: it fits models that TP8 cannot, with `b·s·h` P2P traffic per
   microbatch instead of `8·b·s·h·(t−1)/t` all-reduce per layer `[verified]`.
2. **Throughput at high concurrency**: DistServe's queueing analysis shows
   inter-op parallelism scales decode throughput near-linearly while intra-op
   hits diminishing returns from communication and reduced per-GPU utilisation
   `[verified]`.
3. **Cross-node scaling** where TP would have to cross the NVLink boundary.

MegaScale-Infer is the interesting variant: instead of pipelining *layers*, it
pipelines *module types* — attention nodes and FFN/expert nodes are separate
deployments, and micro-batches "ping-pong" between them. Their three conditions
for hiding communication are worth memorising `[verified]`:

- `T_attn ≈ T_ffn` (balance the two module pools; they derive
  `n_a = (b_e·E)/(b_a·K)`)
- `T_comm < T_max(compute)`
- `m ≥ 2·(1 + T_comm/T_compute)` micro-batches — typically **3–4**

**For our 8-GPU box: do not use PP.** `--pp-size 1`. Revisit only for
multi-node cost-optimised serving where TPOT SLO is loose.

---

## Technique family 4 — Expert parallelism

| paper / system | lab | venue+year | hardware | headline | production? |
|---|---|---|---|---|---|
| DeepSeek-V3 Technical Report (arXiv 2412.19437) | DeepSeek | arXiv 2024 | H800 clusters | prefill EP32/TP4/DP8 (32 GPUs); decode **EP320**/TP4/DP80 (320 GPUs) `[verified]` | **Yes** |
| DeepSeek-V3/R1 Inference System Overview (open-infra-index, Feb 2025) | DeepSeek | blog 2025 | H800, 226.75 nodes avg | **73.7k input / 14.8k output tok/s per node**; 56.3% cache hit; $87,072/day cost `[verified]` | **Yes** |
| DeepEP (github.com/deepseek-ai/DeepEP) | DeepSeek | library 2025 | SM90/SM100 + CX7 | SM100 NVLink EP8: dispatch **726 GB/s** / combine **740 GB/s** @64 SMs `[verified]` | **Yes** — SGLang, vLLM |
| Deploying DeepSeek with PD Disagg + Large-Scale EP (LMSYS blog, May 2025) | LMSYS/SGLang | blog 2025 | 96× H100 (12 nodes) | 52.3k input / 22.3k output tok/s/node; **5.2× vs TP16** decode `[verified]` | **Yes** |
| MegaScale-Infer (arXiv 2504.02263) | ByteDance + PKU | SIGCOMM 2025 | H20/L40S/A100 | 1.90× per-GPU tput, 1.7× per-dollar; M2N cuts P99 vs NCCL by 92.9% `[verified]` | Yes |
| Scaling Large MoE Models with Wide-EP on NVL72 (NVIDIA blog) | NVIDIA / TensorRT-LLM | blog 2025 | GB200 NVL72 | **EP32 = 1.8× per-GPU output tok/s vs EP8** @100 tok/s/user, DeepSeek-R1 `[reported]` | Yes — TRT-LLM |
| UCCL-EP (arXiv 2512.19849) | (fetched abstract) | arXiv 2025 | NVIDIA+EFA, AMD+Broadcom | 2.1× dispatch/combine vs best existing on EFA; +40% SGLang token tput `[reported]` | Emerging |

### The mechanism

EP shards **experts** across ranks; every rank keeps the full attention stack.
Per MoE layer the collective pattern is:

1. **dispatch** — all-to-all: send each token to the ranks owning its top-k
   experts. Payload FP8. Variable-size (routing-dependent), which is why
   CUDA-graph capture needs a fixed-size "low-latency" mode.
2. expert GEMMs (grouped/masked GEMM over per-expert token buckets)
3. **combine** — all-to-all: return partial outputs and weighted-sum them.
   Payload BF16 (2× dispatch).

Two collectives per MoE layer, same count as TP's two all-reduces — **but the
payload is `topk/n_experts_per_rank` of the tokens rather than all of them, and
only ~5.25 of 8 ranks are involved per token** `[inferred]`. EP is strictly
better than TP for the MoE part on any NVLink-connected box.

### The three things that make EP work in practice

**(a) Node-limited routing.** DeepSeek-V3 caps each token at **≤4 nodes**
`[verified]`, so the IB fan-out is bounded and the NVLink→IB forwarding kernels
can be written with a fixed structure. Their measurement: **20 SMs are sufficient
to fully saturate both IB and NVLink bandwidth** `[verified]`. On a single
8×B200 node this constraint is vacuous — everything is NVLink — which is exactly
why single-node EP8 is easy and multi-node EP is hard.

**(b) Load balancing (EPLB).** Routed-expert popularity is heavily skewed; a
single hot expert stalls its rank and every other rank waits at the combine
barrier. EPLB allocates *redundant* experts (e.g. 32 extra on top of 256, giving
a 288-expert pool) and places duplicates of hot experts on separate GPUs, also
enabling non-power-of-two EP degrees like 12 or 72 `[verified, LMSYS]`. Measured
effect on 96×H100 DeepSeek-V3: **1.49× prefill, 2.54× decode** `[verified]`.
DeepSeek runs three separate balancers — prefill (balance core-attention compute
and input tokens per GPU), decode (balance KV-cache usage and request count), and
expert-parallel (minimise max dispatch-receive load) `[verified]`.

**(c) Two dispatch modes.** DeepEP ships `normal` (throughput-optimised,
symbolic shapes, **CUDA-graph incompatible**) and `low_latency` (pure RDMA,
preallocated fixed buffers, CUDA-graph compatible) `[verified, LMSYS]`. This is a
hard argument for PD disaggregation: with a fused engine you must pick one; with
separate prefill and decode pools you use `normal` for prefill and `low_latency`
for decode. Our flag is `--deepep-mode auto` `[verified]`.

### Wide EP: the scaling law

The reason to grow EP is not communication, it is **weight-read amortisation**.
At EP8 each B200 holds 32 of 256 experts; at EP64 it holds 4. Fewer experts per
GPU ⇒ less HBM spent on expert weights ⇒ more HBM for KV cache ⇒ larger batch ⇒
better weight-read amortisation on the *remaining* experts.

- NVIDIA, DeepSeek-R1 on GB200 NVL72: EP32 gives **1.8×** the per-GPU output
  tok/s of EP8 at a fixed 100 tok/s/user `[reported]`.
- SGLang on 96×H100: EP72 decode gives **5.2×** the output throughput of TP16
  `[verified]`, and EP32 prefill gives 3.3× `[verified]`.

Our 8-GPU box caps us at EP8. **Growing the NVLink domain is the single
highest-leverage hardware change available to us**, and it is worth pricing an
NVL72 against 9 separate 8-GPU boxes.

---

## Technique family 5 — Data parallelism with attention-DP

This is the SGLang/DeepSeek pattern and it is probably the most important
*architectural* idea in this document for a MoE at moderate concurrency.

**The problem with TP for attention.** Under TP8, MLA's compressed KV (a single
512-dim latent + 64-dim rope per token per layer) has no head dimension to split
along, so it is **replicated on all 8 ranks**. You pay 8× the KV memory for zero
parallelism. Helix Parallelism states the general form: once TP width exceeds the
number of KV heads, *"each additional shard must store a full copy of the KV
cache to serve its assigned query heads, despite splitting computation"*
`[verified]`.

**The fix.** Run **attention data-parallel** (each rank owns a disjoint set of
requests and their KV) and **FFN/MoE expert-parallel**. Per layer:

- attention: **no collective at all** (rank-local)
- entering MoE: dispatch all-to-all (which also performs the DP→EP redistribution)
- leaving MoE: combine all-to-all
- optionally a reduce-scatter/all-gather pair at the DP boundary

SGLang's memory criterion, which I could not find stated anywhere else: DP beats
TP whenever

```
TP  >  sqrt( N_param / ((1+k) · N_hidden_state) )
```

and for decode at 128 tokens/device with k=3 this gives optimal **TP ≤ 6**
`[verified, LMSYS blog]`. **TP8 is already on the wrong side of that line for
decode.** Combined with `--enable-dp-lm-head`, which makes the vocabulary
projection parallel across the attention-TP group to *avoid the all-gather across
DP groups* `[verified, SGLang `server_args.py`]`, the pattern eliminates the
largest single collective in the model (vocab is 129K–151K wide).

### Helix Parallelism — the same idea taken further

Helix (NVIDIA, arXiv 2507.07120) splits the *temporal* structure of a decode step
instead of the batch: attention runs as `KVP × TP_A` (KV-cache sharded along the
**sequence** dimension across KVP GPUs, TP_A ≤ number of KV heads), then a
**single all-to-all over the query-head axis** rescales and sums the partial
softmax results exactly, then the *same* GPUs are re-provisioned as `TP_F × EP`
for the FFN `[verified]`.

`HOP-B` (Helix Overlap Pipeline, Batch-wise) overlaps that all-to-all with the
attention compute of the *next* item in the batch `[verified]`.

Results — **simulated**, on an in-house GB200 NVL72 model at FP4, at 1M context
`[verified that they are simulated]`: up to **1.5× TTL reduction** and **32×
larger batch at the same latency** for DeepSeek-R1; **4×** throughput/batch for
Llama-405B. HOP-B ablation: removing it costs ~1% for DeepSeek-R1 (communication
is a minor bottleneck under MLA) but ~12% for Llama-405B `[verified]`.

**Read that ablation carefully before adopting HOP-B for GLM-5.2.** MLA models
have small enough attention communication that batch-wise overlap of the
attention all-to-all is nearly worthless. Our collectives are dominated by the
*FFN/MoE* side, which is where SBO already operates.

---

## Technique family 6 — Compute/communication overlap (our highest-leverage area)

| system | lab | venue+year | granularity | hardware | measured | production? |
|---|---|---|---|---|---|---|
| **CoCoNet** (arXiv 2105.05720) | Microsoft Research | ASPLOS 2022 | operator + chunk (DSL) | 256× V100 (16 DGX-2) | model-parallel inference **1.51×**; pipeline-parallel inference **1.77×**; MatMul+AllReduce **1.36×** hiding 80% of compute `[verified]` | Ideas absorbed |
| **Megatron async-TP / TE** | NVIDIA | 2023– | chunked GEMM ↔ AG/RS | A100/H100 | Flux measured TE overlap efficiency at **−125% to +74%** (often worse than no overlap) `[verified, Flux]` | Yes (TransformerEngine `--tp-comm-overlap`) |
| **PyTorch async-TP** | Meta / PyTorch | 2024 | micro-pipelined AG↔matmul over SymmetricMemory | H100 | flag `torch._inductor.config._micro_pipeline_tp = True` | Yes (torchtitan) |
| **Flux** (arXiv 2406.06858) | ByteDance | arXiv 2024 | **tile**, fused into CUTLASS epilogue/prologue | 8×A100 PCIe / 8×A100 NVLink / 8×H800 NVLink; 128 GPU | prefill **1.66×**, decode **1.30×** vs vLLM; overlap efficiency avg **63%** (A100 NVLink) / **72%** (H800) `[verified]` | Yes — open source |
| **T3** (arXiv 2401.16677) | AMD Research | ASPLOS 2024 | tile, **zero SMs** (hardware track-and-trigger + near-memory reduce) | simulated, validated vs MI210 | sublayer **30% geomean** (max 47%); data movement −22%; e2e training +10%; prompt-phase inference +12%; **+29% for 500B models** `[verified]` | **No** — needs new hardware |
| **Comet** (arXiv 2502.19811) | ByteDance Seed + SJTU | MLSys 2025 | tile + thread-block specialisation | 8×H800 NVLink, 8×L20 PCIe | MoE layer **1.96×**, e2e **1.71×**; hides **86.5%** of comm (vs Tutel 68.6%, FasterMoE 29.2%) `[verified]` | **Yes** — 10k-GPU production |
| **TileLink** (arXiv 2503.20313) | ByteDance Seed | MLSys 2025 | tile-centric primitives, Triton codegen | 8×H800, 16×H800 | 1.17–20.76× vs non-overlap; **94.5%** of Flux on AG+GEMM, **128%** of Flux on GEMM+RS; e2e 1.32× vs PyTorch `[verified]` | Open source |
| **NanoFlow** (arXiv 2408.12757) | Univ. of Washington | OSDI 2025 | **nano-batch**, intra-device unit scheduling | 8×A100-80GB SXM | LLaMA-2-70B **1270 tok/s = 68.5% of the 1857 tok/s optimum**; **1.91×** vs TensorRT-LLM `[verified]` | Research |
| **TokenWeave** (arXiv 2505.11329) | Microsoft Research India | arXiv 2025/26 | coarse 2-way token split + fused multimem AR+RMSNorm | 8×H100 DGX, 8×B200 DGX | **1.28×** latency, **1.19×** throughput vs vLLM-multimem; comm is 9–23% of e2e `[verified]` | Emerging |
| **SGLang TBO** (LMSYS blog) | LMSYS | 2025 | micro-batch (2) | 96×H100 | prefill **+27–35%**; decode **+25.5%** @128 seq/dev; **−27% @32 tok/dev** `[verified]` | **Yes** — `--enable-two-batch-overlap` |
| **SGLang SBO** | SGLang | 2025 | intra-batch, expert-signal | Blackwell | combine on 32-SM side stream, down-GEMM signals per expert `[verified, local source]` | **Yes** — `--enable-single-batch-overlap` |

### The granularity ladder, and what each level costs you

**Layer granularity (naive).** Launch the collective, wait, launch the next GEMM.
Zero overlap. This is the baseline everything is measured against.

**Micro-batch granularity (SGLang TBO, DeepSeek dual-batch, MegaScale-Infer
ping-pong).** Split the request batch in two; run micro-batch A's compute while
micro-batch B's communication is in flight. DeepSeek does exactly this: for
prefill, *"overlapping the attention and MoE of one micro-batch with the dispatch
and combine of another"*; for decode, *"overlap the attention of one micro-batch
with the dispatch+MoE+combine of another"* and, in the production system, a
**5-stage pipeline** because the decode stage durations are unbalanced
`[verified]`.

*Cost:* halves the GEMM's M dimension. At C1 with 4 verify tokens, splitting to 2
tokens per micro-batch makes every GEMM catastrophically small. **This is why
SGLang measures −27% at 32 tokens/device.** TBO is a throughput technique.

*And for us it is unavailable anyway:* our engine raises on
`--enable-two-batch-overlap` when `index_topk_freq > 1` because *"the TBO op path
does not propagate topk indices across layers, so shared layers would run sparse
attention without indices"* `[verified, local source]`. We run
`index_topk_freq=4`. Either fix that plumbing or write TBO off.

**Tile granularity (Flux, Comet, TileLink, TokenWeave, our SBO).** Do not split
the batch; split the *kernel*. The producer GEMM signals completion per output
tile, and the communication consumes tiles as they become ready.

- **Flux** fuses the communication into the CUTLASS kernel itself: reduce-scatter
  as an **epilogue** (each thread block computes `TileCoord(threadblock_id,
  rank_id, N_TP)` to pick which peer's output pointer to write, using P2P writes
  intra-node and NVSHMEM `put` inter-node), all-gather as a **prologue** (host
  issues tiled async transfers and sets signals with `cuStreamWriteValue`; the
  kernel spins on `WaitSignal()` before computing each tile) `[verified]`. Tile
  swizzle is **shifted by rank index to avoid write-conflict hot-spotting on the
  memory controller** `[verified]` — a detail worth stealing directly.
- **Comet** deliberately does *not* fuse vertically. It uses **thread-block-level
  isolation**: separate producer blocks and consumer blocks inside one kernel,
  with the split `(n_p, n_c)` chosen by profiling. Their reported optimum moves
  from `n_c = 18` to `n_c = 26` when sequence length goes 4096 → 16384
  `[verified]` — i.e. the SM split is *shape-dependent and must be tuned*, not a
  constant. Their decomposition rule is the useful part: for
  **all-to-all→GEMM** decompose the shared tensor along **M** (tokens are
  independent); for **GEMM→all-to-all** you must decompose along **N**, because
  the top-k reduction creates dependencies along M `[verified]`.
- **TokenWeave** argues *against* fine granularity: coarse 2-way splitting is
  enough for a pipeline, and more splits *"increase decomposition overhead
  without providing additional overlap opportunities"* `[verified]`. Their real
  contribution is a fused **AllReduce + RMSNorm** kernel using NVLink multimem
  that needs only **2–8 SMs on an 8×H100 DGX** versus 16–20+ for communication
  alone in prior work `[verified]`, plus **wave-aware splitting** — the split
  offset is chosen so the two halves' wave counts sum to the unsplit wave count,
  avoiding wave quantisation `[verified]`. They also tested on **8×B200 DGX**,
  making this the most directly transferable result in the table.
- **T3** is the "right" answer that needs silicon we do not have: a 256-entry
  tracker in the memory controller watches stores to the producer GEMM's output
  region and **triggers a DMA with zero SM involvement**, with the reduction done
  by near-bank ALUs in compute-enhanced HBM `[verified]`. Their stated reason for
  avoiding kernel fusion is worth quoting because it is the strongest argument
  against the Flux approach: BLAS libraries have *"hundreds of GEMM kernels
  optimized for different input sizes and GPU architecture, generated via an
  expensive tuning process"*, and building fused variants of every
  (GEMM × collective) pair is *"extremely complex and expensive"* `[verified]`.

**Intra-device unit granularity (NanoFlow).** The most aggressive framing:
classify every operation as compute-bound (dense GEMM), memory-bound (decode
attention), or network-bound (collectives), split into **nano-batches**, and
co-schedule so that all three hardware resources are saturated simultaneously.
They model pairwise kernel interference with a profiled lookup table and require
`Σ R ≤ 1.0` of GPU resource `[verified]`. On 8×A100 with LLaMA-2-70B they hit
**1270 tok/s = 68.5% of a derived 1857 tok/s optimum** (optimum =
`Compute / (2 × P_model)`) `[verified]`. That derived-optimum framing is a good
one to import: we should compute our own `Compute/(2·P)` ceiling and report
against it.

### What this means concretely for our 47% skew

Skew is *not* fixed by any of these. Overlap hides communication behind compute
on the *same* rank; skew is other ranks not having arrived yet. The literature's
answers to skew are:

1. **Balance the work** — Striped Attention's insight for causal masks, EPLB's
   for expert routing (1.49×/2.54× measured `[verified]`), DeepSeek's three
   separate balancers `[verified]`.
2. **Reduce the number of barriers** so each skew event is paid fewer times —
   DP attention, `--enable-dp-lm-head`, fusing AR+RMSNorm (TokenWeave).
3. **Make the barrier cheap** — one-shot/multimem all-reduce instead of ring.
   Our tree already has a FlashInfer fusion path with a documented
   *"max token num to 128 for allreduce fusion with min-latency case
   (use_oneshot=True)"* `[verified, local source]` — verify it is actually
   engaging at our decode batch sizes, because 128 tokens is above our C1 verify
   batch of 4 and below our C64 batch of 256.

---

## Technique family 7 — Prefill/decode disaggregation as a parallelism decision

| paper | lab | venue+year | hardware | headline | production? |
|---|---|---|---|---|---|
| **DistServe** (arXiv 2401.09670) | PKU + UCSD | OSDI 2024 | 32× A100-80GB, 4 nodes, 25 Gbps inter-node | **7.4× more requests** or **12.6× tighter SLO** vs vLLM at 90% attainment; KV transfer **<0.1% of latency** `[verified]` | Concepts everywhere |
| **Splitwise** (arXiv 2311.18677) | Microsoft + UW | arXiv 2023 (rev. 2024) | heterogeneous A100/H100 | **1.4× throughput at 20% lower cost**, or **2.35× throughput** at same cost+power `[reported]` | Azure |
| **Mooncake** (arXiv 2407.00079) | Moonshot AI + Tsinghua | arXiv 2024 (rev. 2025) | Kimi production | **+525%** throughput in simulation; **+75% requests** in real workload `[verified]` | **Yes** — Kimi |
| **MegaScale-Infer** (arXiv 2504.02263) | ByteDance + PKU | SIGCOMM 2025 | H20, L40S, A100 | **1.90×** per-GPU tput; M2N cuts median latency **68.2%** and P99 **92.9%** vs NCCL at 256 KB `[verified]` | ByteDance |
| **Beyond the Buzz** (arXiv 2506.05508) | NVIDIA | arXiv 2025 | simulated Blackwell FP4, 100k+ design points | disaggregation wins for **prefill-heavy** traffic and models **>10B**; loses to chunked prefill for generation-heavy; fixed 3.5:1 context:generation ratio only works at relaxed latency `[verified]` | Guidance |
| **NVIDIA Dynamo + GB200 NVL72** (dev blog) | NVIDIA | blog 2025 | GB200 NVL72 (simulated) | **6×** throughput for DeepSeek-R1, **3×** for Llama-70B in the medium-latency regime, disagg vs colocated `[reported]` | Yes — Dynamo |

### The parallelism argument (the part that matters here)

Disaggregation is usually sold as interference elimination. The deeper reason is
that **prefill and decode want opposite parallelism**:

| | prefill | decode |
|---|---|---|
| batch (tokens/forward) | thousands | ones to hundreds |
| bottleneck | FLOPs | HBM bandwidth + collective latency |
| TP | helps: reduces execution time, and comm is bandwidth-bound so it amortises | hurts past the KV-head/√ crossover; adds latency-bound barriers |
| SP | free (bandwidth-neutral, saves activation memory) | **doubles collective count** — harmful |
| EP dispatch mode | `normal` (throughput, no CUDA graph) | `low_latency` (fixed buffers, CUDA graph) |
| PP | poor (single long request, m=1) | good for throughput at high concurrency |

DistServe formalises the first row with M/D/1 queueing: for prefill, intra-op
(TP) wins at low arrival rate because execution time dominates, and inter-op (PP)
wins at high rate because queueing dominates; for decode, once batch approaches
the compute-bound regime *"intra-op parallelism reduces latency with diminishing
returns, caused by communication and reduced utilization after partitioning.
Inter-op parallelism can almost linearly scale the throughput"* `[verified]`.

The DeepEP mode incompatibility is the single most concrete argument for
disaggregation on an MoE: `normal` mode produces symbolic shapes and **cannot be
CUDA-graph captured**, `low_latency` requires preallocated fixed buffers and
cannot handle long inputs efficiently `[verified, LMSYS]`. A fused engine must
pick one or pay `auto`-mode switching cost. A disaggregated one gets both.

### KV transfer cost — the honest version

Our MLA numbers (above) say KV transfer is 0.32 ms for an 8K prompt over NVLink5
and 0.72 ms over 8×400 Gb IB `[inferred]`. For reference an 8K prefill on this
class of model takes tens to hundreds of ms, so the transfer is 1–3% — consistent
with DistServe's measured **<0.1%** on OPT-175B over 25 Gbps `[verified]` and
NVIDIA's finding that *"existing provisioned datacenter bandwidth is sufficient
to support KV cache transfer without becoming a bottleneck"* for DeepSeek-R1
`[verified]`.

**But this is an MLA-specific result.** The same table shows a GQA-shaped 92-layer
model costs 5.4× more per token, putting an 8K prompt at 3.86 ms over 400 GB/s.
Do not carry the "KV transfer is free" conclusion from a DeepSeek-shaped model to
a GQA-shaped one without redoing the arithmetic.

SGLang's implementation supports **heterogeneous TP** between the two pools —
*"prefill and decode use different tensor parallelism (TP) sizes (e.g., prefill
TP=4, decode DP attention with TP=1)"* — via a GPU staging buffer that
gathers/scatters between the two KV layouts `[verified, SGLang docs]`. Backends:
Mooncake (supports `INTRA_NODE_NVLINK` transport), NIXL (UCX), Ascend. Flags:
`--disaggregation-mode prefill|decode`, `--disaggregation-transfer-backend`,
`--disaggregation-ib-device` `[verified]`.

### Our verdict

On **one 8-GPU node**, disaggregation costs you GPUs on both sides of a split
that is too small to rate-match, and NVIDIA's study says the fixed 3.5:1
context:generation ratio degrades as the latency target tightens `[verified]`.
**Do not disaggregate within one node for the latency objective.** Do
disaggregate when we have ≥2 nodes and are optimising cost/user, because that is
where the SP-on/SP-off, `normal`/`low_latency`, and TP-degree splits all pay at
once.

---

## Technique family 8 — Multi-node and rack-scale

**The wall.** TP is an NVLink-domain technique. The OSU measurement is the
cleanest evidence: on 4×H100 hosts with NDR400 IB, Llama-3.2-3B TPOT was 1.17 ms
at TP2, **0.86 ms at TP4 (intra-node)**, and **11.56 ms at TP8 (crossing a node
boundary)** `[verified]`. TTFT kept improving (30 ms at TP8) because prefill is
bandwidth-amortised; decode fell off a cliff because it is latency-bound. One IB
hop, 13× TPOT.

**What you do instead.** Everything published at scale grows EP and DP over the
fabric while keeping TP inside the node:

| deployment | TP | EP | DP | nodes | result |
|---|---|---|---|---|---|
| DeepSeek-V3 prefill | 4 (+SP) | 32 | 8 | 4 (32 GPU) | 32 redundant experts, 1 extra/GPU `[verified]` |
| DeepSeek-V3 decode | 4 (+SP) | **320** | 80 | 40 (320 GPU) | 1 expert/GPU, 64 GPUs host redundant+shared `[verified]` |
| DeepSeek production (Feb 2025) | — | prefill EP32/DP32 (4 nodes), decode **EP144**/DP144 (18 nodes) | | | 73.7k in / 14.8k out tok/s/node `[verified]` |
| SGLang on 96×H100 | — | prefill EP32 (4 nodes), decode EP72 (9 nodes) | | 12 | 52.3k in / 22.3k out tok/s/node `[verified]` |
| SGLang on GB200 NVL72 | — | wide EP, pure NVLink | | 14 (2 prefill / 12 decode) | **7,583 tok/s/GPU decode, 2.7× vs H100** `[verified]` |

**Node-limited routing** (≤4 nodes per token) bounds the IB fan-out and lets the
NVLink→RDMA forwarding be written as a fixed pipeline; DeepSeek reports **20 SMs
saturate both IB and NVLink** `[verified]`. NVLink is 160 GB/s vs IB's 50 GB/s in
their setup, so the asymmetric-domain forwarding kernel is the whole game
`[verified]`.

**What changes on a rack-scale NVLink domain (GB200/GB300 NVL72).** 72 GPUs in
one coherent NVLink domain at 1.8 TB/s per GPU and 130 TB/s aggregate
`[reported, NVIDIA]`. Three qualitative changes, all documented:

1. **EP can be wide without touching IB.** NVIDIA measures **EP32 at 1.8× the
   per-GPU output tok/s of EP8** for DeepSeek-R1 at a 100 tok/s/user SLO
   `[reported]`.
2. **Overlap becomes less necessary.** SGLang's GB200 report states the larger
   NVLink domain *"allows two-batch overlap to be disabled, resulting in both
   kernel performance speedup and avoiding waste when overlapped communication is
   longer than computation"* `[verified]`. This is a striking result: **better
   interconnect made a software overlap technique net-negative.** Expect the same
   for us whenever our compute time per stage drops below the collective time.
3. **The disaggregation split moves inside the rack** — 2 prefill nodes / 12
   decode nodes in SGLang's config `[verified]`.

For us: if we buy more B200 capacity as separate 8-GPU nodes, we get EP8 per node
and IB between them. If we buy NVL72 we get EP72 in one domain. Based on the
measured EP scaling curves, **the NVLink domain size, not the FLOPs, is the
variable that sets our cost/user at concurrency.**

---

## Decision matrix

Reasoning shown. `h` = hidden, `L` = layers, `n_kv` = KV heads (∞ for MLA-latent).

| model shape | batch / concurrency | objective | configuration | why |
|---|---|---|---|---|
| MoE 256E/8A, MLA, 61–92L, fits in 8×183 GB | **C1, 4 verify tokens** | **min latency** | **TP8 + DP-attention + EP8**, SBO on, TBO off, SP off, PP1, CP1, one-shot/multimem AR, CUDA graph, EAGLE 3-1-4 | AR payload 14–56 KiB ⇒ 100% latency-bound (§arithmetic). Every barrier costs 3–9 µs regardless of size, so minimise *count*: DP-attention removes the attention AR; `--enable-dp-lm-head` removes the vocab all-gather. SP would double barrier count. TBO halves an already-tiny GEMM (and is blocked by `index_topk_freq=4` in our tree). |
| same | **C16–C64 (64–256 tok/forward)** | **max tok/s/GPU** | TP8 + DP-attention + EP8, **SBO on, TBO evaluated at the boundary**, `--deepep-mode auto`, EPLB on | 256 tok/forward is just past the 143-token latency/bandwidth crossover, so overlap starts paying. SGLang measured TBO **+25.5% at 128 seq/device** but **−27% at 32 tok/device** — the sign flips inside our operating range, so this must be measured, not assumed. |
| same | **C256+** | max tok/s/GPU | as above + chunked prefill, larger CUDA-graph max bs | fully bandwidth-bound; ring AR at 3584 KiB takes 7.1 µs of real transfer, comparable to the barrier. Now transfer optimisation (multimem, FP8 AR) matters. |
| **dense** or low-expert-count model | C1 | min latency | TP8, no EP, tile-level overlap of AG+GEMM / GEMM+RS | no all-to-all to hide behind; Flux/TileLink territory. Flux measured prefill **1.66×** / decode **1.30×** vs vLLM on 8×H800 `[verified]`. |
| GQA model with `n_kv = 8` | C1 | min latency | **TP8 max** — TP>8 replicates KV | Helix: past TP = `n_kv`, each shard stores a full KV copy `[verified]`. If you need more parallelism, use KV-parallelism (Helix) or DP-attention, not more TP. |
| any | **prefill pool** (thousands of tokens) | TTFT | TP8 **+ SP on**, `deepep normal`, chunked prefill, high EP if multi-node | prefill is bandwidth-amortised: SP is free (Korthikanti) and saves activation memory; `normal` dispatch maximises throughput; CUDA graph is irrelevant at these shapes. |
| any | **decode pool** | TPOT / cost | DP-attention + EP, `deepep low_latency`, CUDA graph, SP **off**, PP only if TPOT SLO is loose | opposite of the row above on every axis — which is the whole argument for disaggregation. |
| model too big for 8×183 GB | any | fit | **PP across nodes, not TP across nodes** | TP inter-node costs 13× TPOT (OSU `[verified]`); PP moves `b·s·h` P2P vs TP's `8bsh(t−1)/t` all-reduce `[verified]`. |
| >256K context, multi-node | prefill | TTFT | **CP (ring, striped/zigzag ordering) + pass-KV/pass-Q switching**, Ulysses degree ≤ `n_kv` | Meta: 1M prefill in 77 s at 93% efficiency on 16 nodes `[verified]`; scales as well on 100 Gb TCP as 400 Gb RDMA `[verified]`. |
| >256K context | decode | TPOT | **do not use CP** | Meta measured TTIT 46.26 ms → 71.31 ms going CP1→CP4 at TP8 `[verified]`. Use Helix-style KV parallelism instead if you must shard KV. |
| multi-node MoE, cost-optimised | C high | $/1M tok | **wide EP + PD disagg + EPLB + node-limited routing** | DeepSeek EP320 decode `[verified]`; SGLang EP72 = 5.2× TP16 `[verified]`; NVIDIA EP32 = 1.8× EP8 `[reported]`; EPLB 2.54× decode `[verified]`. |

---

## What is NOT worth it

**1. Pipeline parallelism inside the 8-GPU node.** Bubble `(p−1)/m` `[verified]`;
at single-stream decode `m=1` so PP8 wastes 7/8 of the box and does not improve
latency at all. Interleaving cannot help because the dependency is genuinely
serial. `--pp-size 1`.

**2. Tensor parallelism across a node boundary.** 11.56 ms vs 0.86 ms TPOT,
measured `[verified, OSU/H100]`. There is no tuning that recovers this.

**3. Megatron sequence parallelism for low-latency decode.** It is
bandwidth-neutral by construction `[verified]` but converts 2 all-reduces into 2
all-gathers + 2 reduce-scatters, doubling the barrier count in a regime where
barriers *are* the cost. It is a memory optimisation sold in a paper about
training, where batches are large. Use it in the prefill pool only.

**4. Two-batch overlap at concurrency 1.** SGLang measured **−27%** at 32
tokens/device `[verified]`; we would be at 4. It halves the GEMM M-dimension in
exchange for hiding a collective that is latency-bound, not bandwidth-bound —
strictly the wrong trade. Additionally blocked in our tree by DSA index-topk
sharing `[verified, local source]`.

**5. Ring/context parallelism for decode.** Meta measured a **54% TTIT
regression** going CP1→CP4 `[verified]`. Ring communication with one token of
work is pure serialised latency.

**6. TransformerEngine's `--tp-comm-overlap` as a drop-in.** Flux measured TE's
overlap efficiency at **−125% to +36% on PCIe and −99% to +74% on NVLink** —
i.e. frequently *worse than not overlapping*, because splitting the GEMM into
several smaller kernels destroys GEMM efficiency `[verified]`. Any
decomposition-based (as opposed to fusion-based) overlap has this failure mode,
and the smaller your batch the worse it gets. At C1 you should assume
decomposition-based overlap is net-negative until measured otherwise.

**7. T3-style hardware-triggered collectives.** Beautiful (30% geomean sublayer
speedup, zero SM cost `[verified]`) and completely unavailable: it requires a
tracker in the memory controller and near-memory ALUs in HBM, and the results are
from a simulator validated against MI210. File under "argue for it in the next
hardware cycle."

**8. Chasing a faster NCCL.** Our all-reduce moves 14 KiB at batch 1; the wire
time is 0.028 µs against a 3–9 µs floor `[inferred]`. There is no bandwidth to
recover. Algorithm selection (`NCCL_ALGO=NVLS`, `NCCL_PROTO=LL`) and one-shot
custom all-reduce matter because they change the *synchronisation* cost, not the
transfer cost — a distinction worth being explicit about when someone proposes
"tuning NCCL."

**9. Compressing the all-reduce (Flash Communication and friends).** >3× intra-node
communication speedup and 2× TTFT reported `[reported]` — but the speedup is on
*bandwidth*, which we do not spend at decode batch sizes, and it costs accuracy
risk on a production model. Reconsider only for the prefill pool at very large
batch, or if we ever run TP across IB (which we should not).

**10. Splitting into more than 2 chunks for overlap.** TokenWeave's explicit
finding: more than two segments *"increases decomposition overhead without
providing additional compute-communication overlap opportunities"* `[verified]`.
Depth of pipelining is not the axis to push; fusion granularity is.

---

## Sources

All URLs below were fetched and read for this document.

**Tensor & sequence parallelism**
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, NVIDIA, arXiv:1909.08053 — https://ar5iv.labs.arxiv.org/html/1909.08053
- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC'21, arXiv:2104.04473 — https://ar5iv.labs.arxiv.org/html/2104.04473
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, NVIDIA, arXiv:2205.05198 (2022) — https://ar5iv.labs.arxiv.org/html/2205.05198
- Xu, Kandadi Suresh, Anthony, Alnaasan, Panda, *Characterizing Communication Patterns in Distributed Large Language Model Inference*, Ohio State, arXiv:2507.14392 — https://arxiv.org/html/2507.14392
- Li et al., *Flash Communication: Reducing Tensor Parallelization Bottleneck for Fast Large Language Model Inference*, arXiv:2412.04964 — https://arxiv.org/abs/2412.04964

**Sequence / context parallelism**
- Liu, Zaharia, Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context*, UC Berkeley, arXiv:2310.01889 (2023) — https://ar5iv.labs.arxiv.org/html/2310.01889
- Microsoft DeepSpeed team, *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*, Microsoft, arXiv:2309.14509 — https://ar5iv.labs.arxiv.org/html/2309.14509
- Brandon, Nrusimha, Qian, Ankner, Jin, Song, Ragan-Kelley, *Striped Attention: Faster Ring Attention for Causal Transformers*, MIT CSAIL, arXiv:2311.09431 — https://ar5iv.labs.arxiv.org/html/2311.09431
- Fang, Zhao, *USP: A Unified Sequence Parallelism Approach for Long Context Generative AI*, Tencent, arXiv:2405.07719 — https://ar5iv.labs.arxiv.org/html/2405.07719
- Yang, Yang, Ibrahim, Xie, Tang, Sizov, Reizenstein, Park, Huang, *Context Parallelism for Scalable Million-Token Inference*, Meta, MLSys 2025, arXiv:2411.01783 — https://arxiv.org/html/2411.01783v3

**Expert parallelism & MoE serving**
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437 — https://arxiv.org/html/2412.19437v2
- DeepSeek-AI, *Day 6: One More Thing — DeepSeek-V3/R1 Inference System Overview*, open-infra-index, Feb 2025 — https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- DeepSeek-AI, *DeepEP: an efficient expert-parallel communication library* — https://github.com/deepseek-ai/DeepEP
- Zhu et al., *MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism*, ByteDance + PKU, SIGCOMM 2025, arXiv:2504.02263 — https://arxiv.org/html/2504.02263v2
- LMSYS/SGLang, *Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs*, May 2025 — https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- NVIDIA, *Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack-Scale Systems* — https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/
- *UCCL-EP: Portable Expert-Parallel Communication*, arXiv:2512.19849 — https://arxiv.org/abs/2512.19849 (abstract only)

**Compute/communication overlap**
- Jangda, Huang, Liu, Nodehi Sabet, Maleki, Miao, Musuvathi, Mytkowicz, Saarikivi, *Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads* (CoCoNet), Microsoft Research, ASPLOS'22, arXiv:2105.05720 — https://ar5iv.labs.arxiv.org/html/2105.05720
- Chang, Bao, Hou, Jiang, Zheng, Zhong, Zhang, Song, Yao, Jiang, Lin, Jin, Liu, *FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion*, ByteDance, arXiv:2406.06858 — https://ar5iv.labs.arxiv.org/html/2406.06858 · https://github.com/bytedance/flux
- Pati, Aga et al., *T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives*, AMD Research, ASPLOS'24, arXiv:2401.16677 — https://arxiv.org/html/2401.16677v1
- Zhang, Zheng, Lin, Jiang, Bao, Jiang, Hou, Cui, Zheng, Chang, Chen, Liu, *Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts*, ByteDance Seed + SJTU, MLSys 2025, arXiv:2502.19811 — https://arxiv.org/html/2502.19811v3
- Zheng, Fang, Zheng, Hou, Bao, Zheng, Jiang, Wang, Ye, Lin, Chang, Liu, *TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives*, ByteDance Seed, MLSys 2025, arXiv:2503.20313 — https://arxiv.org/html/2503.20313v1
- Zhu, Gao, Zhao et al., *NanoFlow: Towards Optimal Large Language Model Serving Throughput*, University of Washington, OSDI 2025, arXiv:2408.12757 — https://arxiv.org/html/2408.12757v2
- Gond, Kwatra, Ramjee, *TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference*, Microsoft Research India, arXiv:2505.11329 — https://arxiv.org/html/2505.11329v5
- PyTorch/torchtitan async tensor parallelism (`torch._inductor.config._micro_pipeline_tp`, SymmetricMemory) — https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487 *(mechanism corroborated via two search results; the forum post itself was not fetched, so no speedup number is quoted here)*

**Disaggregation**
- Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang, *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, PKU + UCSD, OSDI 2024, arXiv:2401.09670 — https://arxiv.org/html/2401.09670v3
- Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini, *Splitwise: Efficient generative LLM inference using phase splitting*, Microsoft + UW, arXiv:2311.18677 (Nov 2023, rev. May 2024) — https://arxiv.org/abs/2311.18677
- Qin, Li, He, Zhang, Wu, Zheng, Xu, *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*, Moonshot AI + Tsinghua, arXiv:2407.00079 (Jun 2024, rev. Sep 2025) — https://arxiv.org/abs/2407.00079
- Mitra, Borkar, Bhatia, Matas, Raj, Mudigere, Zhao, Golub, Dutta, Madduri, Jani, Pharris, Darvish Rouhani, *Beyond the Buzz: A Pragmatic Take on Inference Disaggregation*, NVIDIA, arXiv:2506.05508 — https://arxiv.org/html/2506.05508v1
- NVIDIA, *How NVIDIA GB200 NVL72 and NVIDIA Dynamo Boost Inference Performance for MoE Models* — https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/
- SGLang docs, *PD Disaggregation* — https://docs.sglang.io/advanced_features/pd_disaggregation.html

**Hybrid / KV parallelism**
- Bhatia, More, Borkar, Mitra, Matas, Zhao, Golub, Mudigere, Pharris, Darvish Rouhani, *Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding*, arXiv:2507.07120 — https://arxiv.org/html/2507.07120v1 · https://arxiv.org/abs/2507.07120

**Interconnect & collectives**
- *Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms*, arXiv:2507.04786 — https://arxiv.org/html/2507.04786v1
- NVIDIA/nccl-tests issue #333 (B200 symmetric-memory small-message all-reduce measurements) — https://github.com/NVIDIA/nccl-tests/issues/333
- NVIDIA, *GB200 NVL72* product page — https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- NVIDIA NCCL environment variables (NCCL_ALGO / NCCL_PROTO / NCCL_NVLS_ENABLE) — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- FlashInfer communication API (`trtllm_allreduce_fusion`, one-shot vs two-shot, MNNVL) — https://docs.flashinfer.ai/api/comm.html
- LMSYS/SGLang, *Running DeepSeek on GB200 NVL72* — https://lmsys.org/blog/2025-06-16-gb200-part-1/

**Engine flags & model configs**
- SGLang `server_args.py` (upstream) — https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py
- SGLang docs, *Expert Parallelism* — https://docs.sglang.io/advanced_features/expert_parallelism.html
- SGLang docs, *Server Arguments* — https://docs.sglang.io/advanced_features/server_arguments.html
- `zai-org/GLM-4.6` config.json — https://huggingface.co/zai-org/GLM-4.6/blob/main/config.json
- `deepseek-ai/DeepSeek-V3.2-Exp` config.json — https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json
- GLM-4.5 Team, *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*, arXiv:2508.06471 — https://arxiv.org/abs/2508.06471

**Local sources (this machine, `[verified]` by reading the file)**
- `/home/aman/code/NotSglang/python/sglang/srt/server_args.py` — `enable_two_batch_overlap`, `enable_single_batch_overlap`, `tbo_token_distribution_threshold`; the DSA `index_topk_freq > 1` guard that blocks TBO
- `/home/aman/code/NotSglang/python/sglang/srt/batch_overlap/single_batch_overlap.py` — `SboFlags`, `communicate_num_sms = 32 if is_blackwell() else 3`, per-expert combine signalling
- `/home/aman/code/NotSglang/python/sglang/srt/layers/communicator.py` — FlashInfer all-reduce fusion, one-shot max token num 128
