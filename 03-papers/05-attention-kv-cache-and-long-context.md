# Attention, KV cache and long context: algorithms that change the inference roofline

## What this is

A survey of the *algorithmic* literature that changes how many bytes and how many FLOPs
attention costs at inference time, written against a specific target: GLM-5.2-class MoE
models (MLA + DeepSeek Sparse Attention, 256/8 experts, EAGLE 3-1-4 speculation) served
on 8x B200 SXM under two objectives — minimum single-stream latency and minimum cost per
user at concurrency.

Results are labelled:

- **[verified]** — I read the number in the paper/repo/source text.
- **[reported]** — the authors claim it in an abstract or README I read, but I did not see
  the supporting table.
- **[inferred]** — my own arithmetic or reasoning, shown so you can check it.

Hardware is stated for every result. A 9x speedup on A100 with a 27B model is not a 9x
speedup on B200 with a 744B MoE, and where that matters I say so.

> **Audit note (this revision).** Every citation in this document was re-fetched and
> re-verified against arXiv, the ACM/USENIX record, GitHub source, and vendor blogs.
> **No fabricated citations were found** — every paper, blog, repo and PR named here
> exists with the stated title, authors and venue. Eleven *numeric or attribution* errors
> were found and corrected; each correction is marked inline as **[CORRECTED]** so you can
> see what changed and why. Six papers and one large block of production flag names were
> added. The two most consequential corrections are (a) our index sharing is
> *training-aware*, which inverts the advice that was previously in Bottom Line #2, and
> (b) GLM-5's indexer has **32** heads, not 64, which changes the Part 8 FLOP arithmetic.

Three config facts anchor the whole document, all read from source:

- **DeepSeek-V3.2 reference config** (`inference/config_671B_v3.2.json`, fetched verbatim
  from the DeepSeek-V3.2-Exp repo): `n_layers: 61`, `n_dense_layers: 3`, `n_heads: 128`,
  `q_lora_rank: 1536`, `kv_lora_rank: 512`, `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`,
  `v_head_dim: 128`, `dtype: fp8`, `scale_fmt: ue8m0`, `index_n_heads: 64`,
  `index_head_dim: 128`, `index_topk: 2048`. **[verified]**
- **GLM-5** (arXiv:2602.15763): 744B total / 40B active, 256 experts, 80 layers, MLA with
  latent KV dim 576 and head dim 256 (up from 192) with head count reduced by 1/3, DSA,
  3 MTP layers with 4 speculative steps and mean accepted length 2.76 vs DeepSeek-V3.2's
  2.55, `torch.topk` as the default top-k operator in the DSA indexer for determinism.
  **[verified from the paper HTML]**
- **GLM-5.2**, per Raschka's write-up: 78 transformer layers, hidden size 6,144, first
  three FFN blocks dense, 256 routed experts / 8 active, indexer top-2048, and an
  **IndexShare** pattern of `full, shared, shared, shared` — 1 indexer per 4 layers,
  *introduced during continued mid-training with 128K-token sequences* so the retained
  indexers could adapt to the layers depending on them. He reports a **2.9x reduction in
  per-token FLOPs at 1M context** for IndexShare. **[reported]** Our build reports 79
  layers / 22 indexer layers / `index_topk_freq=4`; I use *our* numbers for arithmetic and
  flag the discrepancy rather than pretending the public numbers are ours.

**That "introduced during continued mid-training" clause is load-bearing** and is the
single most important thing this audit corrected. It means our sharing is *trained in*,
not bolted on — which flips the recommendation about calibration sweeps. See Bottom
Line #2 and Part 4.

---

## Bottom line for our system

Ranked by expected value on our hardware and our two objectives. Each item states the
expected effect and the reason, and is honest about what is measured vs modelled.

1. **Move attention from TP8 to DP8 (data-parallel attention) if we haven't. Expected
   effect: ~8x long-context concurrency, zero quality cost.** MLA keeps a single latent KV
   per token — there is no KV-head axis to shard. Under TP8 every rank holds the *entire*
   KV cache, so the KV pool is ~112 GB, not 896 GB. My arithmetic in Part 8.2 puts that at
   **~15.6 concurrent 128k streams under TP8 vs ~125 under DP8** [inferred]. SGLang moved
   to DP attention for exactly this reason and reports 52.3k input tok/s and 22.3k output
   tok/s per node on 96xH100 for DeepSeek-V3, up to 5x over vanilla tensor parallelism on
   the same 96 GPUs [verified from the LMSYS blog]. The flag is
   **`--enable-dp-attention`** with **`--dp-size`** (SGLang `server_args.py`, verified in
   source); pair it with **`--enable-dp-lm-head`**. NVIDIA's Helix Parallelism
   (arXiv:2507.07120) makes the same diagnosis more sharply — with MLA, "TP>1 causes
   duplication of KV cache" because the latent is a single KV head shared across all 128
   query heads — and proposes sharding KV along the *sequence* axis instead (KV
   parallelism), reporting up to 1.5x TTL improvement and 32x larger batch at 1M context
   for DeepSeek-R1 [reported]. Caveat that matters: **Helix's results are from an in-house
   simulator on GB200 NVL72, not measured hardware** [verified]. Treat it as a design
   argument, not a benchmark.
2. **Do NOT run IndexCache's greedy layer-calibration sweep on our model. [CORRECTED — this
   reverses the previous revision's advice.]** The previous version of this document told
   you to greedy-search our layer pattern because uniform 1-in-4 was "leaving quality on
   the table". That is right for a model where sharing is bolted on post hoc, and wrong for
   ours. IndexCache (arXiv:2603.12201, THUDM) measures both regimes and states the
   distinction explicitly: in the **training-free** setting, uniform 1/4 on GLM-5 collapses
   Long Avg from 78.4 to **72.7** (GraphWalks 92.7 → 74.9) while the greedy-searched
   pattern recovers 78.0 [verified]. But in the **training-aware** setting, "the pattern
   sensitivity observed in training-free IndexCache vanishes with training" — uniform 1/2
   scores 51.6 Long Avg against a 51.0 baseline, and uniform 1/4 scores 50.6 vs 51.0
   [verified]. GLM-5.2's IndexShare was trained in during continued mid-training
   [reported, Raschka]. **So our uniform 1-in-4 is the validated configuration, and a
   calibration sweep would be measuring a problem we do not have.** The residual action is
   cheap and different: confirm from our config loader whether we are actually uniform, and
   note that SGLang reads a per-layer **`index_topk_pattern`** string from the HF config
   (an F/S pattern, same notation as IndexCache) in addition to `index_topk_freq`, so a
   non-uniform pattern is expressible upstream if we ever want one [verified in SGLang
   `server_args.py`].
3. **A/B the DSA decode kernel and the indexer logits kernel on B200. Expected effect:
   unknown but potentially large; cost is one afternoon.** This is the highest
   effort-adjusted item in the list because the flags already exist. FlashMLA's own README
   reports sparse MLA *decoding* at 410 TFLOPS on H800 SXM5 but only **350 TFLOPS on
   B200 — and says why, in the README: "(which is not really optimized yet)"**
   [verified; **[CORRECTED]** — the previous revision claimed the README gave no
   explanation]. Meanwhile SGLang exposes **`--dsa-decode-backend`** and
   **`--dsa-prefill-backend`** with choices `flashmla_sparse`, `flashmla_sparse_q8`,
   `flashmla_kv`, `flashmla_auto`, **`flashinfer_sparse_mla`**, `fa3`, `tilelang`, `aiter`,
   `trtllm`, and **`--dsa-paged-mqa-logits-backend`** whose `cutedsl` option is documented
   in-source as "CuTe DSL kernel, SM 100 (Blackwell) only; **wins at low batch size and
   long context**" [all verified in SGLang `server_args.py`]. Low batch size and long
   context is precisely our C1 objective. There is also **`--dsa-topk-backend`**
   (`sgl-kernel` default, `torch`, `flashinfer`) and the env var `SGLANG_DSA_FUSE_TOPK`.
   Running a Hopper-tuned kernel on Blackwell when a Blackwell-specific one is one flag
   away is free money.
4. **Share the index across the EAGLE draft positions within a verify step. Expected
   effect: up to ~2.76x reduction in the dominant long-context term. Nobody has published
   this.** Speculative decoding amortises weight loading over the accepted length but does
   **not** amortise the indexer, because the indexer must score the full prefix for the
   query position(s). GVR (NVIDIA, arXiv:2604.22312) measured **35-50% top-K overlap
   between consecutive decode steps in layers 20-60** on real DeepSeek-V3.2 traffic, and
   only 1-2% in layers 0-1 [verified] — the same Toeplitz/RoPE argument applies with even
   more force *within* a verify step, where the draft positions are adjacent by
   construction. Index once per verify step, reuse across draft positions, re-index on the
   accepted token. Correctness is preserved because the accepted token always gets a fresh
   index. GLM-5's mean accepted length is 2.76 [verified], so that is the amortisation
   factor. This remains the single highest-value unimplemented idea in this literature for
   our specific stack. Quality risk is unmeasured — that is the honest caveat.
5. **Adopt MISA-style indexer head routing. Expected effect: ~4x fewer indexer head-token
   products on GLM-5 shapes; training-free; already validated on GLM-5.** MISA
   (arXiv:2605.07363, PKU MuLab + HITSZ) routes each query to h=8 of the indexer's heads
   using cheap block-pooled statistics. On DeepSeek-V3.2 (H^I=64) that is 8x; **on GLM-5,
   H^I=32, so it is 4x** [verified; **[CORRECTED]** — the previous revision assumed 64
   indexer heads for GLM-5 and therefore claimed 8x]. LongBench 46.43 vs a DSA baseline of
   46.01 on GLM-5 — i.e. it slightly *beat* the baseline — and 50.85 vs 51.05 on
   DeepSeek-V3.2; recovers >92% of DSA's per-layer selected tokens; NIAH fully green to
   128k; 3.82x indexer-kernel speedup from their TileLang kernel on a single H200
   [verified]. The paper explicitly states it is complementary to IndexCache because it
   operates on the head axis [verified]. Training-free and GLM-5-validated is a rare
   combination; this is the best unexplored option we have.
6. **Adopt NVIDIA's Guess-Verify-Refine top-K kernel — but expect ~2.4-3.5%, not 7.5%,
   because we speculate. [CORRECTED]** arXiv:2604.22312 is *literally our workload*:
   DeepSeek-V3.2 NVFP4 on B200x8, k=2048, TEP8 min-latency, TensorRT-LLM. 1.88x average
   single-operator speedup over the production radix-select, up to 2.42x per layer per step
   (layer 22 averages 2.04x), bit-exact index sets vs `torch.topk` for N=8k-131k
   [verified]. But the end-to-end TPOT reduction *degrades as speculation depth grows*:
   **5.47% at 64k and 7.52% at 100k with MTP=0; 4.36% and 6.30% with MTP=1; and only 2.40%
   and 3.45% with MTP=3** [verified]. The previous revision quoted "4.36-7.52% including
   with MTP=1" and omitted the MTP=3 row, which is the row nearest our EAGLE 3-1-4 config.
   At 365 tok/s single-stream, 2.4-3.5% is ~9-13 tok/s. Still worth taking — it is
   bit-exact, so it cannot cost quality, and GLM-5 explicitly chose deterministic
   `torch.topk` for RL stability [verified], which a non-exact heuristic would have broken.
   Merged in TensorRT-LLM PR #12385 (merged 2026-04-03) behind
   `DeepSeekSparseAttentionConfig(enable_heuristic_topk=True)`, **default False**, and
   auto-disabled on pre-Blackwell [verified from the PR]. If we are on SGLang rather than
   TensorRT-LLM, the equivalent lever is `--dsa-topk-backend`.
7. **Make prefix caching MLA-native and position-independent. Expected effect: our 1.54x
   becomes materially larger on agentic and multi-turn traffic.** Irminsul
   (arXiv:2605.05696) exploits the fact that MLA's cached row factors into a position-free
   `c_KV` and a 64-dim `k_R` that is correctable by a closed-form delta rotation, giving
   content-addressed rather than prefix-addressed caching. They extend SGLang's radix cache
   with content-hash keying over CDC-chunked segments and recover **up to ~83% of prompt
   tokens above exact-prefix hit rate on agentic traffic, with 63% prefill energy savings
   per hit** [verified from the abstract]. Evaluated on DeepSeek-V2-Lite (16B/2.4B), Kimi
   Moonlight-16B-A3B and JoyAI-Flash (48B/3B) — all much smaller than us, and hardware is
   not stated. **This is a genuinely MLA-specific capability that GQA models cannot have**,
   and we are running an MLA model. Our 1.54x is a strict-prefix number; agentic traffic is
   where the headroom is.
8. **Know that two-batch overlap and index sharing are mutually exclusive in SGLang. No
   action, but do not be surprised.** SGLang hard-errors: `--enable-two-batch-overlap` "is
   not supported with DSA index-topk sharing (`index_topk_freq > 1` or an
   `index_topk_pattern` containing 'S')" [verified in `server_args.py`]. TBO is worth
   27-35% prefill throughput and 35% decode speedup under simulated MTP on DeepSeek-V3
   [verified from the LMSYS blog]. So our `index_topk_freq=4` is *buying* long-context
   decode headroom at the price of TBO's prefill throughput. That is a real trade with a
   real number on both sides, and it belongs in any capacity planning we do. Whether the
   incompatibility is fundamental or just unimplemented is worth asking upstream.
9. **If KV memory ever binds before bandwidth, go FP8 latent before you go sparser.**
   vLLM's DeepSeek-V3.2 layout is 656 B/token/layer — 512 bytes of FP8 e4m3 NoPE + 16 bytes
   of FP32 scales + 128 bytes of **BF16, unquantised** RoPE — vs 1152 B BF16 [verified from
   the vLLM blog]. A 1.76x cut with no algorithmic risk. DeepSeek-V4 (arXiv:2606.19348)
   ships exactly this split: "BF16 precision is used for the rotary positional embedding
   (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions"
   [verified]. Note TensorRT-LLM supports FP8 KV cache + MLA on Hopper and Blackwell for
   V3/R1 but **only on Blackwell for V3.2-Exp**, with GPQA accuracy drop under 1%
   [verified from the TRT-LLM README].
10. **Do not spend engineering on training-free KV eviction for a reasoning model.** "Hold
    Onto That Thought" (arXiv:2512.12008, UMD et al., measured on an RTX A6000 48GB) shows
    **PyramidKV at 0.00 accuracy on MATH-500 with DeepSeek-R1-Distill-Llama-8B at every
    budget from 128 to 512** (Appendix Table 11) and **KNorm at 0.00 on GSM8K at budgets
    128 and 256 vs 0.70 full cache** [verified]. Eviction commits at prompt time; a
    reasoning model's critical tokens are produced thousands of steps later. DSA is a
    *selection* method — nothing is destroyed — which is the right side of the line. Do not
    bolt eviction onto it.
11. **Watch DeepSeek-V4's Compressed Sparse Attention as the next architectural step.** V4
    compresses every m tokens into one KV entry and *then* runs DSA top-k over the
    compressed entries, plus a Heavily Compressed Attention dense branch and a sliding
    window branch, with attention sinks, partial RoPE on the last 64 dims, and query/KV
    normalisation. Reported: **10% of V3.2's KV cache at 1M context for V4-Pro (1.6T/49B),
    7% for V4-Flash (284B/13B), and 27%/10% of single-token inference FLOPs** [verified].
    SGLang already carries a `DeepseekV4ForCausalLM` code path [verified]. If we are going
    to 1M context, this is the shape the field is converging on — not more aggressive top-k
    on uncompressed KV.

---

## Part 0 — the roofline everything below is trying to move

Fix the machine: B200 SXM, 183 GB HBM3e, ~8 TB/s. The SNU/UIUC systems paper
(arXiv:2507.15465) models B200 SXM6 at **2250 TFLOPS BF16, 8000 GB/s and 192 GB**, giving a
**ridge point of 281.25 Op/B** [verified from their Table I]. FP8 doubles peak, so the FP8
ridge is ~562 Op/B; NVFP4 doubles again to ~1125 Op/B.

> **[CORRECTED] — paper title.** This paper was submitted as *"The New LLM Bottleneck: A
> Systems Perspective on Latent Attention and Mixture-of-Experts"* (v1) and **retitled in
> v3 to "Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and
> Mixture-of-Experts"**. Both names refer to arXiv:2507.15465. Cite the current one.

Decode-time arithmetic intensity of the core attention, per token, absorbed form, ignoring
the small RoPE terms [inferred, arithmetic shown so you can check it]:

| Attention form | KV bytes/token/layer (BF16) | FLOPs/token/layer/kv-token | Intensity |
|---|---|---|---|
| MHA, 128 heads x 128 | 65,536 | 65,536 | **1 Op/B** |
| GQA-8, 128 q heads x 128 | 4,096 | 65,536 | **16 Op/B** |
| MLA absorbed (d_c=512, d_r=64, 128 heads) | 1,152 | 278,528 | **~242 Op/B** |
| MLA absorbed, FP8 latent (656 B layout) | 656 | 278,528 | **~425 Op/B** |
| MLA absorbed, q_len=4 (EAGLE verify) | 1,152 | 1,114,112 | **~968 Op/B** |

My 242 comes from charging the latent read **once** and using it for both the score and the
context GEMM: FLOPs = 2·128·576 + 2·128·512 = 278,528; bytes = 576·2 = 1,152.

> **[CORRECTED] — the independent cross-check.** The previous revision claimed the SNU/UIUC
> paper "independently puts FlashMLA at ~256 Op/B on B200" and that "my 242 and their 256
> agree". That is wrong. Their Table III gives **≈512 Op/B for MLA *without* reordering**;
> with reordering the decode core-attention intensity is
> `n_hd·d_KVco/(n_hd+d_KVco) ≈ 100 Op/B` for DeepSeek-R1, and **FlashMLA doubles that to
> ≈200 Op/B** by reusing the loaded `C_KV` across the score and context layers [verified].
> So the correct statement is: their accounting gives ~200, mine gives ~242, and the gap is
> bookkeeping about how the RoPE dims and the score/context reuse are charged. The
> *conclusion* is unchanged and is what matters — they write that the doubled intensity
> "closely approaches the ridge point... exhibiting a balance between computation and
> memory bandwidth", while MHA-based core attention "presents an ArI of approximately 1
> even with batching" [verified].

**This is the single most important fact in this document.** MLA does not merely shrink the
KV cache; it moves decode attention from 1 Op/B to 200-250 Op/B, i.e. from ~280x below the
B200 ridge to *at* the ridge. Every technique below either (a) reduces the bytes further,
which now buys less than it used to, or (b) reduces the FLOPs, which now buys more than it
used to.

Two consequences the same paper quantifies and we should internalise [verified]:

- Layer reordering (absorption) reduces decode attention-block latency **by up to 103.12x**
  but **increases prefill attention-block latency by up to 2.21x**. They conclude: "the
  prefill stage uses MLA without reordering and the decode stage uses MLA with reordering."
  Two code paths are mandatory, not optional.
- Sparse attention (DSA) reduces both bytes and FLOPs, but introduces an indexer whose
  arithmetic intensity is *fixed at 2·H^I* — **64 Op/B on GLM-5 shapes (H^I=32)**, 128 Op/B
  on DeepSeek-V3.2 shapes (H^I=64) — independent of context length and batch size
  [inferred; derivation in Part 8.5]. Both are **below** the BF16 ridge and cannot climb.
  So the indexer is irreducibly memory-bound while the attention it feeds is not. That
  asymmetry is why every 2026 paper in Part 4 attacks the indexer and not the attention.

---

## Part 1 — Exact attention kernels: the online-softmax lineage

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Online normalizer calculation for softmax (arXiv:1805.02867) | Milakov, Gimelshein — NVIDIA | arXiv 2018 | GPU (unspecified) | Softmax 1.3x; fused safe-softmax + top-K 5x [verified] | Yes — the primitive under every FlashAttention |
| FlashAttention (arXiv:2205.14135) | Dao, Fu, Ermon, Rudra, Ré — Stanford | NeurIPS 2022 | A100-class | 3x on GPT-2 seq 1k; 15% BERT-large 512 [verified] | Universal |
| FlashAttention-2 (arXiv:2307.08691) | Dao | arXiv 2023 | A100 | 225 TFLOPs/s, 72% MFU [verified] | Universal |
| FlashAttention-3 (arXiv:2407.08608) | Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao — Colfax/NVIDIA/Meta/Together/Princeton | arXiv 2024 | H100 | FP16 740 TFLOPs/s (75%), FP8 ~1.2 PFLOPs/s [verified] | Yes (Hopper default) |
| FlashAttention-4 (arXiv:2603.05451) | Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao | arXiv Mar 2026 | **B200 / GB200** | 1.1-1.3x vs cuDNN 9.13.0, 2.1-2.7x vs Triton, peak 1613 TFLOPs/s BF16 (71%) [verified] | Open source, CuTe-DSL |
| Flash-Decoding (PyTorch/Together blog) | Meta + Together (Dao, Haziza, Massa, Sizov) | blog, Oct 2023 | **A100** | attention up to 50x vs FA2; up to 8x end-to-end CodeLlama-34B [verified] | Universal (FA >= 2.2, xFormers >= 0.0.22) |
| FlashInfer (arXiv:2501.01005) | Ye, Chen, Lai, Lin et al. — UW / NVIDIA / CMU | **MLSys 2025** | not stated in abs | 29-69% inter-token-latency reduction vs compiler backends; 28-30% for long-context [verified] | **Yes — integrated in SGLang, vLLM, MLC-Engine** |
| FlashMLA (github.com/deepseek-ai/FlashMLA) | DeepSeek | repo | H800 SXM5, B200 | dense MLA decode 660 TFLOPS / 3000 GB/s (H800); **sparse MLA decode 410 TFLOPS H800 vs 350 TFLOPS B200 "not really optimized yet"**; sparse prefill 640 TFLOPS H800 / 1450 TFLOPS B200 [verified] | Yes — default sparse-MLA backend in TensorRT-LLM on Hopper [verified] |

### The mechanism, briefly, and why FA4 matters on our box

Online softmax (Milakov & Gimelshein) computes `m_j = max(m_{j-1}, x_j)` and
`d_j = d_{j-1}·exp(m_{j-1} - m_j) + exp(x_j - m_j)` in a single pass, so the normalizer
never needs a second read of the logits. FlashAttention lifts that from a vector to a tile:
tile Q, K, V into SRAM, accumulate `O` and the running `(m, l)` statistics, and never
materialise the `N x N` score matrix in HBM. FlashAttention-2 fixes the *work partitioning*
— parallelise over query blocks, not just batch x head — and cuts non-matmul FLOPs, which
matter because non-matmul throughput on a tensor-core GPU is ~1/16 of matmul throughput.
FlashAttention-3 adds Hopper's asynchrony: warp-specialised producer/consumer with TMA,
ping-pong scheduling that overlaps the softmax of tile j with the GEMM of tile j+1, and FP8
with block quantisation.

**FlashAttention-4 is the one that matters for B200.** Its framing is asymmetric hardware
scaling: tensor cores doubled, but shared memory bandwidth and the exponential unit did
not. Concretely from the paper [all verified]:

- **Software-emulated exp.** BF16 MMA runs at **8192 ops/clock/SM** on B200 (derived in the
  paper as 2.25 PFLOPS / 1850 MHz / 148 SMs), doubled from Hopper's 4096. The multifunction
  unit (MUFU) that computes `ex2` runs at **16 ops/clock/SM — the same as Hopper**. FA4
  therefore computes `2^x = 2^floor(x) · 2^(x - floor(x))`, getting the integer part by
  IEEE-754 bit manipulation and the fractional part from a **degree-3 polynomial with
  maximum relative error 8.8e-5** (~600x worse than hardware at FP32, but after BF16
  rounding the BF16 quantisation error of ~3.9e-3 dominates for all degrees >= 3). Note:
  B300/GB300 double MUFU throughput to 32 ops/clock/SM, which weakens this argument on
  future silicon.
- **Conditional softmax rescaling.** Standard online softmax rescales the accumulator every
  time the running max moves. FA4 skips the update when `m_j - m_{j-1} <= tau`, with
  `tau = log2(256) = 8.0`, tracking the total scaling and correcting at final
  normalisation.
- **2-CTA MMA + tensor memory.** 256 KB of on-chip tensor memory per SM, allocated in
  32-column (16 KB) granules with explicit programmer management; MMA tiles of 128x128 or
  128x256 in 2-CTA mode; the backward dQ step uses distributed shared memory to halve
  global atomic reductions.
- Written entirely in **CuTe-DSL** (Python-embedded), 20-30x faster compile than C++
  templates.

Two facts from the paper that bear directly on our kernel choices:

- **"FlashAttention-3 does not run on B200"** [verified, footnote]. If any part of our stack
  still dispatches to FA3, it is silently falling back on Blackwell.
- **"Since the initial release of our implementation, newer versions of cuDNN have
  incorporated many of the techniques described in this paper, yielding similar performance
  to FA4"** [verified]. The 1.1-1.3x margin over cuDNN 9.13.0 is a snapshot, not a durable
  advantage. Do not build a port on the assumption that FA4 stays ahead of cuDNN.

Head dims tested are 64, 128, and `(192, 128)` "for DeepSeek V3 compatibility" (16 heads,
192 query dims, 128 KV dims). **MLA's 576/512 is explicitly not tested, and sparse attention
is future work** [verified]. So FA4 is the right reference for our *dense* prefill attention
and for any GQA-shaped model we add (Qwen3.8), but it does not replace FlashMLA for the MLA
decode path.

### Flash-decoding / split-KV

The decode-time problem FlashAttention-2 does not solve: at q_len=1 and batch 1, there is
one query block, so parallelism = batch x heads, which on a 108-SM A100 uses <1% of the
GPU. Flash-decoding adds a third parallel axis — split the KV sequence into chunks, run
FlashAttention per chunk writing out a per-row log-sum-exp scalar, then a second kernel
reduces across splits using those LSEs. Measured on **A100**, CodeLlama-34B shapes (16 q
heads, 2 kv heads, d=128), B=1 seqlen 131072: **PyTorch eager 2664 us, FlashAttention 2.0.9
4592.2 us, Flash-Decoding 106.6 us** [verified from the blog's table]. Attention runtime is
near-constant from 512 up to 32k.

This is table stakes and is in every engine, but the split count is a tuning knob that
matters a lot at our concurrency: too many splits and the reduction kernel and the LSE
traffic dominate; too few and you idle SMs. At C64 with 8 GPUs you already have
64 x n_heads work items, so the optimal split count at C64 is much smaller than at C1 —
worth a sweep, since we care about both regimes.

### FlashInfer — the engine, not the kernel

FlashInfer (MLSys 2025) is worth knowing because it is *what our engine actually calls*.
It stores KV in a block-sparse format with composable layouts, JIT-compiles a customisable
attention template, and load-balances scheduling while staying CUDAGraph-compatible
[verified from the abstract]. It is integrated into SGLang, vLLM and MLC-Engine [verified].
Concretely for us, `flashinfer_sparse_mla` is one of SGLang's `--dsa-decode-backend`
choices and `flashinfer` is one of the `--dsa-topk-backend` choices [verified in source] —
so "try FlashInfer instead of FlashMLA on B200" is a one-flag experiment, not a port.

---

## Part 2 — KV cache shape: MQA, GQA, and Multi-head Latent Attention

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Fast Transformer Decoding: One Write-Head is All You Need (arXiv:1911.02150) | Shazeer — Google | arXiv 2019 | TPU | MQA: single KV head, "much faster to decode", "only minor quality degradation" [verified] | Yes (PaLM, Falcon) |
| GQA (arXiv:2305.13245) | Ainslie et al. — Google | EMNLP 2023 | TPU | Uptrain MHA->MQA/GQA with **5% of original pretraining compute**; GQA ~ MHA quality at ~MQA speed [verified] | Universal (Llama 2/3, Mistral, Qwen) |
| DeepSeek-V2 (arXiv:2405.04434) | DeepSeek-AI | arXiv 2024 | H800 | 236B/21B active; MLA: KV cache **-93.3%**, max generation throughput **5.76x** vs DeepSeek 67B, training cost -42.5% [verified] | Yes |
| DeepSeek-V3 (arXiv:2412.19437) | DeepSeek-AI | arXiv 2024 | H800 | 671B/37B active, MLA + DeepSeekMoE + MTP, 14.8T tokens, 2.788M H800-hours [verified] | Yes |
| Hardware-Centric Analysis of MLA (arXiv:2506.02523) | Geens, Verhelst — KU Leuven | Electronics Letters 2025 | modelled accelerator, 400 GB/s | Decode OI: MHA ~0.5-1.5 Op/B, MLA-recompute ~3-5 Op/B and cache-size-insensitive [verified] | analysis |
| Rethinking LLM Inference Bottlenecks (arXiv:2507.15465) *(v1: "The New LLM Bottleneck")* | Yun et al. — SNU + UIUC | arXiv 2025 | **B200 SXM6 modelled + DGX H100 measured** | MHA/GQA ~1 Op/B; MLA un-reordered ~512; MLA reordered ~100; **FlashMLA ~200**; B200 ridge **281.25**; reorder = 103.12x decode / 2.21x prefill penalty [verified] | analysis |
| Hardware-Efficient Attention for Fast Decoding (arXiv:2505.21487) | Zadouri, Strauss, Dao | arXiv 2025 | not stated in abs | GTA = GQA quality at ~half KV; GLA = MLA quality, shardable; **GLA kernel up to 2x faster than FlashMLA when q_len > 1**; up to 2x online-serving throughput [reported] | Research |
| Helix Parallelism (arXiv:2507.07120) | Bhatia, More, Borkar, Mitra — **NVIDIA** | arXiv Jul 2025 | **GB200 NVL72, FP4 — SIMULATED** | KV parallelism during attention, TP/TPxEP during FFN; **1.5x TTL and 32x batch for DeepSeek-R1 at 1M**; 1.13x / 4x for Llama-405B [reported] | Research (NVIDIA) |
| GQLA (arXiv:2605.15250) | Meng — PKU MuLab | arXiv 2026 | H100, H20 | Two algebraically-equivalent decode paths from one weight set; TransGQLA converts a GQA checkpoint; **28.125% of GQA per-token KV on the absorb path (LLaMA-3-8B)** [verified] | Research |
| QK-Normed MLA (arXiv:2606.16310) | Han, Zhao, Zhou, Li, Sun | arXiv 2026 | **H800**, 400M models / 100B tokens | QK-RMSNorm made compatible with latent caching; **<2% decode latency overhead up to 256k** [verified] | Research |
| Why Megatron-Core Forbids Absorption / LAGA (arXiv:2607.17644) | Ma | arXiv 2026 | **A100** (memory), **8x Ascend 910B** (LAGA) | Absorbed MLA in *training* inflates activation memory 20-34%, up to 9.2 GB (n_h=128, seq 16384, SP=8), 19.2 GB fused; LAGA cuts comm 1.98x [verified] | analysis |

### MLA, in enough detail to implement

**The compression.** For token `t` with hidden state `h_t`, MLA does not cache K and V. It
caches one low-rank latent:

```
c^KV_t = W^DKV h_t                (d_c = 512)
k^C_t  = W^UK c^KV_t              (reconstruct 128 heads x 128 dims)
v^C_t  = W^UV c^KV_t
```

Queries get their own (non-cached) low-rank path, `c^Q_t = W^DQ h_t` with `q_lora_rank =
1536`, then `q^C_t = W^UQ c^Q_t` [verified from the V3.2 config JSON].

**Decoupled RoPE — the part people get wrong.** RoPE is position-dependent and does not
commute with the up-projection `W^UK`: if you rotate the reconstructed key, you cannot fold
`W^UK` into the query anymore, because the rotation sits between them and depends on the
*relative* position of query and key. MLA's fix splits the key into two pieces:

- a **NoPE** piece of `qk_nope_head_dim = 128` per head, reconstructed from the latent and
  therefore absorbable;
- a **decoupled RoPE** piece `k^R_t` of `qk_rope_head_dim = 64`, produced by a separate
  projection *from `h_t` directly*, carrying RoPE, **shared across all heads** (MQA-style),
  and cached alongside the latent.

So the cache per token per layer is `d_c + d_h^R = 512 + 64 = 576` values, versus
`2 · n_h · d_h = 32768` for MHA. DeepSeek-V2's table states MLA's cache is equivalent to
"GQA with only 2.25 groups" while beating MHA on quality [verified].

**Weight absorption.** At decode you never want to materialise 128 heads of K and V from the
latent — that is 32768 values of write traffic per token per layer to save 576 values of
read traffic. Instead, because

```
q^T k = (W^UQ c^Q)^T (W^UK c^KV) = c^Q^T (W^UQ^T W^UK) c^KV
```

you precompute or recompute `W^UQ^T W^UK` and apply it to the query, then dot the resulting
"absorbed query" (dimension 576 per head) directly against the cached latent. Symmetrically
`W^UV` folds into `W^O` on the output side. The attention then looks like **MQA with
head_dim_k = 576 and head_dim_v = 512** — exactly the shape FlashMLA's support matrix
advertises as "MQA mode" [verified from the FlashMLA README].

**Numerical and systems caveats of absorption — four of them, all real:**

1. **Absorb vs recompute is a real trade, not a strict win.** Geens & Verhelst name the two
   schemes `MLA_rc` (recompute the composite on the fly) and `MLA_ru` (precompute and
   reuse). `MLA_rc` gives ~3-5 Op/B at decode and is insensitive to KV size; `MLA_ru` scales
   its OI with KV size and is poor for small caches. Their conclusion: **`MLA_rc` wins on
   essentially all commercial accelerators and at batch=1**; `MLA_ru` only wins when compute
   is scarce relative to bandwidth, "an uncommon case" [verified]. Their model is a 400 GB/s
   accelerator, not B200 — treat the crossover as directional only.
2. **Absorption is a decode-only optimisation and actively hurts prefill.** The SNU/UIUC
   paper measures it: **103.12x decode benefit, 2.21x prefill penalty** [verified]. Two code
   paths are required. TensorRT-LLM and SGLang both do this.
3. **Absorption is a memory trap in training.** arXiv:2607.17644 shows Megatron-Core
   hard-asserts absorption off during training
   (`assert not (self.training and self.cache_mla_latents)`) and quantifies why: the
   absorbed intermediates live in `n_h x d_kv` per token, *larger* than the per-head K/V
   they replace, inflating activation memory **20-34%, up to 9.2 GB at DeepSeek-V3 scale
   (n_h=128, seq=16384, SP=8, eager kernel), widening to 19.2 GB with a fused kernel**,
   cross-verified on **NVIDIA A100** [verified]. Their LAGA fix all-gathers the latent (1.98x
   comm reduction, measured on **8x Ascend 910B**) but reconstructs per-head K/V locally
   rather than absorbing. Relevant to us only if we fine-tune or do RL on this model.
4. **Precision.** The composite `W^UQ^T W^UK` is a product of two learned matrices — its
   dynamic range is the product of theirs. Two published consequences: (a) QK-RMSNorm, the
   standard stabiliser, *appears* incompatible with latent caching because post-projection
   normalisation needs the full key; arXiv:2606.16310 shows the static weight absorbs into
   the query side and the dynamic statistic reduces to one inverse-RMS scalar per token per
   KV group, restoring exact equivalence at <2% decode overhead on H800 up to 256k
   [verified]; (b) FlashMLA's sparse decode kernel keeps the KV in FP8 *with per-block scale
   factors* and vLLM's layout keeps the 64 RoPE dims in **BF16, unquantised** — 128 of the
   656 bytes [verified]. If you are tempted to quantise the RoPE half, note that DeepSeek's
   own V4 design and vLLM's layout both refuse to. That is a strong signal.

**Tensor-parallel consequence — the one that costs us money.** With MHA or GQA, TP splits the
KV heads across ranks and the KV cache shrinks per rank. With MLA there is **one** latent per
token; there is no head axis to split. Under TP8 the latent cache is replicated 8x. Helix
Parallelism states this in one sentence: with MLA, "key and value projections are absorbed
into a latent space" yielding "a single KV head shared across all 128 query heads", so
"TP>1 causes duplication of KV cache" [verified]. SGLang's answer is DP attention —
"eliminates KV cache duplication across devices, significantly reducing memory overhead"
[verified] — behind `--enable-dp-attention`. Helix's answer is KV parallelism: shard the KV
cache along the *sequence* dimension across GPUs, run attention on the shards, then reuse
the same GPUs for TP (dense) or TPxEP (MoE) during the FFN, with a lightweight all-to-all in
between whose cost is hidden by batch-wise overlap ("Helix HOP-B"). Helix's numbers are
**simulated on GB200 NVL72 at FP4**, not measured [verified] — a caveat that should follow
the citation everywhere.

Our measured 19.6% collectives with 47% rank-arrival skew is partly a symptom of the TP8
choice: TP8 attention forces an all-reduce per layer on the output projection and every rank
must arrive. DP attention replaces that with an all-gather/reduce-scatter over the token
axis, which has different skew behaviour and can be overlapped with the MoE dispatch. SGLang
also exposes `--attention-context-parallel-size` and, for DSA specifically,
`--enable-dsa-cache-layer-split`, documented as splitting "DSA GPU KV/indexer cache layers
across context-parallel ranks to reduce per-rank KV memory" — but it is a **prefill-CP-only**
optimisation, rejected on decode workers, and currently requires the mooncake transfer
backend [verified in source].

### The alternatives to MLA worth knowing

**GTA / GLA** (Zadouri, Strauss, Dao, arXiv:2505.21487). GTA ties and reuses K and V states
to halve GQA's cache at matched quality. GLA is "grouped latent attention" — MLA-quality but
with a group axis, so it **shards under TP without replication**. The claim that "our
optimized GLA kernel is up to 2x faster than FlashMLA... in a speculative decoding setting
when the query length exceeds one" [reported] is directly relevant: it says MLA's kernel
leaves performance on the table exactly in the regime EAGLE puts us in. We cannot change
GLM-5.2's architecture, but we can take the kernel lesson — **benchmark our attention kernel
at q_len=4-5, not q_len=1.**

**GQLA** (arXiv:2605.15250, PKU) is the pragmatic version: expose *both* an MQA-absorb path
and a per-group-expanded GQA path over the same weights, and pick at runtime by hardware and
by `s_q` (query length). They target `s_q=1` on H100 and `s_q=2` on H20, and note that MLA
"yields no Multi-Token Prediction gain on commodity inference GPUs such as the
export-restricted H20" [verified]. This is the right mental model for our serving stack:
**one weight set, two attention code paths, dispatched on q_len and on whether we are
latency- or throughput-bound.**

---

## Part 3 — Sparse attention that requires training

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| NSA (arXiv:2502.11089) | Yuan et al. — DeepSeek + PKU + UW | ACL 2025 (long) | **8x A100** | 64k: fwd 9.0x, bwd 6.0x, **decode 11.6x (memory-access-derived)**; 27B/3B MoE, **260B tokens**; LongBench 0.469 vs 0.437 full attention [verified] | Research; ideas shipped in DSA |
| MoBA (arXiv:2502.13189) | Lu et al. — Moonshot AI + Tsinghua + Zhejiang | arXiv 2025 | not stated (Llama-8B shapes) | **6.5x prefill at 1M**, 16x attention-compute reduction at 10M; val-loss gap within 1e-3 [verified] | **Yes** — deployed for Kimi long-context |
| DeepSeek-V3.2 / DSA (arXiv:2512.02556) | DeepSeek-AI | arXiv Dec 2025 | H800 | Lightning indexer + top-k=2048, FP8 indexer, MQA-mode MLA; O(L^2) -> O(Lk) core attention [verified] | **Yes** — vLLM, SGLang, TensorRT-LLM |
| GLM-5 (arXiv:2602.15763) | GLM-5 Team (Zhipu/Z.ai + Tsinghua) | arXiv Feb 2026 | not stated | DSA on 744B/40B, 80 layers; "90% of attention entries in long contexts are indeed redundant"; **"DSA reduces the attention computation by roughly 1.5-2x for long sequences"**; 20B tokens sparse training [verified] | **Yes — this is our model's lineage** |
| MiniMax Sparse Attention (arXiv:2606.13392) | Lai et al. — MiniMax + PKU + NVIDIA | arXiv Jun 2026 | **H800** | GQA-native block sparse, k=16 blocks of 128 per group; 109B/6B on 3T tokens; **28.4x per-token attention FLOPs at 1M; 14.2x prefill / 7.6x decode wall-clock at 1M** [verified] | Production model family |
| DeepSeek-V4 (arXiv:2606.19348) | DeepSeek-AI | arXiv 2026 | NVIDIA GPUs + Huawei Ascend NPUs | CSA (compress m tokens -> 1 entry, then DSA top-k) + HCA + sliding window; **KV = 10% (Pro, 1.6T/49B) / 7% (Flash, 284B/13B) of V3.2 at 1M**; FLOPs 27%/10%; FP4 experts, FP8 KV with BF16 RoPE dims, FP4 indexer [verified] | Announced; SGLang has a `DeepseekV4ForCausalLM` path [verified] |

### NSA — the design constraints everyone else inherited

NSA runs three branches in parallel and gates them with a learned sigmoid MLP:
`o_t = sum_c g_t^c · Attn(q_t, K~_t^c, V~_t^c)` over `c in {compression, selection, sliding}`
[verified, Eq. 5].

- **Compression**: pool blocks of length `l = 32` with stride `d = 16` into single compressed
  KV entries. Cheap global view.
- **Selection**: the important trick — the *selection* importance scores are derived from the
  *compression* branch's attention scores, not from a separate scorer.
  `p^cmp_t = softmax(q_t^T K~^cmp_t)`, then map to selection blocks of size `l' = 64` by
  summing the compression scores that overlap each selection block, then take top-`n = 16`
  blocks (including 1 initial + 2 local, forced) [verified].
- **Sliding window** `w = 512`, to stop the other branches from being forced to learn local
  patterns.

Two design constraints matter more than the branches:

1. **Blocks, not tokens.** "Token-granular selection... leads to the need to load a large
   number of individual tokens from the KV cache during attention computation. This
   non-contiguous memory access prevents efficient adaptation of fast attention techniques
   like FlashAttention" [verified quote]. Note that **DSA went token-granular anyway** and
   solved it with a bespoke gather kernel (FlashMLA's `indices` tensor) — a real engineering
   divergence.
2. **GQA-consistent selection.** All heads in a KV group must select the same blocks, so the
   scores are summed across heads in the group before top-k [verified]. Without this, a
   group's shared KV read is the union of per-head selections and the sparsity evaporates.

Their decode memory-access table is the cleanest statement of why sparse decode helps
[verified]: at 8k/16k/32k/64k, full attention loads 8192/16384/32768/65536 token-equivalents
vs NSA's 2048/2560/3584/5632 — **expected** speedups 4x/6.4x/9.1x/11.6x. Read that word
"expected": 11.6x is a memory-access ratio the paper then reports as achieved, not an
independent wall-clock measurement of a full model.

Config: **8-GPU A100**, 27B total / 3B active MoE, 30 layers, hidden 2560, GQA 4 groups / 64
heads, d_q=d_k=192, d_v=128, **260B tokens** at 8k then continued at 32k
[verified; **[CORRECTED]** — the previous revision said 270B]. Quality: general average 0.456
vs 0.443 full attention; LongBench 0.469 vs 0.437; AIME at 16k generation limit after
R-distillation 0.146 vs 0.092 [verified].

**Transfer caveat:** A100, 27B, 64k. The speedup *ratios* should transfer (they are
memory-access ratios), the absolute kernel efficiency will not.

### DSA — what our model actually runs

From the DeepSeek-V3.2 paper [verified] plus the reference implementation [verified]:

```
I_{t,s} = sum_{j=1..H^I} w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )
```

- `H^I = index_n_heads = 64` on DeepSeek-V3.2; `index_head_dim = 128`; `index_topk = 2048`.
  **On GLM-5, `H^I = 32`** [verified from MISA, which uses both models].
- **ReLU, not softmax** — the paper says "we choose ReLU as the activation function for
  throughput consideration" [verified]. No normalisation pass, no exponential, so the indexer
  is a plain FP8 GEMM plus a ReLU plus a weighted reduce.
- `w^I` comes from `weights_proj = Linear(dim, n_heads, dtype=float32)`, i.e. a per-head
  scalar gate computed from the token, in FP32, then scaled by `n_heads^-0.5` and by the FP8
  quantisation scales [verified from `model.py`].
- The indexer key `k^I_s` is **one vector per token per layer** (MQA-style), 128 dims, FP8.
  That is what makes its cost `O(L · d_I)` bytes rather than `O(L · H^I · d_I)`.
- The main attention runs under the **MQA mode of MLA** over the selected top-k entries.

**Training recipe** [verified, exact]: two stages. (1) Dense warm-up: freeze everything except
the indexer, train it with a KL loss against the main model's aggregated attention
distribution, lr 1e-3, 1000 steps x 16 sequences x 128K tokens = **2.1B tokens**. (2) Sparse
training: top-k=2048 active, all parameters trained, lr 7.3e-6, 15000 steps x 480 sequences x
128K = **943.7B tokens**; crucially the indexer is optimised *only* by the indexer loss and
the main model *only* by the LM loss. GLM-5's version is a shorter 1000-step warm-up plus
**20B tokens** of sparse continued pretraining [verified].

**The honest limitation the paper itself states**: "the lightning indexer retains O(L^2)
complexity" [verified]. DSA converts core attention from O(L^2) to O(Lk) and leaves a
cheaper-but-still-quadratic term behind. Part 8 quantifies exactly how much that costs us.

**Production status, with flag names** [all verified from source]:

- **vLLM** shipped DSA the day V3.2-Exp released: DeepGEMM's `fp8_mqa_logits()` for indexer
  logits, a fused top-K derived from DeepSeek's TileLang reference, FlashMLA's sparse
  attention kernel. KV layout **656 bytes/token/layer** (512 B FP8 e4m3 NoPE + 16 B FP32
  scales + 128 B BF16 RoPE) plus a separate per-block indexer K cache; **block size 64 is
  the only supported block size** because FlashMLA is tailored to it and the indexer cache
  requires it; supported on 16xH100, 8xH200, 8xB200.
- **SGLang** treats DeepSeek-V3.2 and GLM-5 as one family (`is_deepseek_dsa(hf_config)
  # DeepSeek 3.2/GLM 5`) and exposes `--dsa-prefill-backend` / `--dsa-decode-backend`
  (`flashmla_sparse`, `flashmla_sparse_q8`, `flashmla_kv`, `flashmla_auto`,
  `flashinfer_sparse_mla`, `fa3`, `tilelang`, `aiter`, `trtllm`), `--dsa-topk-backend`
  (`sgl-kernel` default / `torch` / `flashinfer`), `--dsa-paged-mqa-logits-backend`
  (`auto` / `deepgemm` / `cutedsl` / `aiter`), plus `--enable-hisparse` with
  `--hierarchical-sparse-attention-extra-config`
  (e.g. `{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}`). The old
  `--nsa-*` spellings are deprecated aliases. `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`
  defaults to the model's `index_topk`, below which prefill uses dense attention (correct,
  since selecting 2048 of fewer than 2048 tokens is a no-op).
- **TensorRT-LLM**: FlashMLA is the default sparse-MLA backend on Hopper for V3.2-Exp;
  `trtllm-gen` codegen for FP8 MLA on Blackwell; NVFP4 on B200/GB200; MTP via
  `--spec_decode_algo MTP --spec_decode_max_draft_len N`; relaxed acceptance for the
  reasoning phase via `--use_relaxed_acceptance_for_thinking --relaxed_topk 15
  --relaxed_delta 0.5`. FP8 KV cache with MLA is supported on Hopper and Blackwell for
  V3/R1 but **only on Blackwell for V3.2-Exp**, with GPQA accuracy drop under 1%.

### MoBA — the MoE framing, and the hybrid recipe

MoBA (Moonshot) partitions context into blocks and routes each query to top-k blocks by
`q · mean_pool(K_block)` — a **parameterless** gate. Causality is handled two ways: future
blocks are hard-masked, and the *current* block is handled by a separate causal-masked
attention because its mean-pooled key would otherwise leak future tokens [verified]. The
kernel splits into "historical blocks" (varlen FlashAttention over gathered query groups) and
"current block" (causal FlashAttention), then merges with online softmax [verified]. Block
sizes 512-4096, top-k 3-12; 6.5x prefill at 1M and 16x attention-compute reduction at 10M vs
full FlashAttention on Llama-8B shapes [verified]. The validation-loss claim is from a
three-model comparison at 1.5B parameters / 30B tokens / 32K context at up to 75% sparsity,
where "the validation loss differences between these two attention mechanisms remain
consistent within a range of 1e-3" [verified].

The transferable detail: **in the 1M continued-pretraining recipe they keep the last 3 of 32
layers on full attention**, with top-k 12 and block size 4096 (95.31% sparsity), and follow a
similar strategy for SFT [verified]. Their stated motivation is specifically SFT: "MoBA
sometimes results in suboptimal performance during SFT... we speculate that this may be
attributed to the loss masking employed in SFT — prompt tokens are typically excluded from
the loss calculation, which can pose a sparse gradient challenge for sparse attention
methods" [verified]. Any hybrid dense/sparse layer schedule we consider should follow that
shape — sparse in the bulk, dense at the boundaries.

### MiniMax Sparse Attention — the GQA-native counterpoint

MSA is worth reading because it is the *other* production answer. Instead of DSA's
token-level, all-heads-shared selection, MSA does **block-level (size 128), per-GQA-group**
selection with k=16 blocks, from an "ultra-lightweight" Index Branch with one index query
head per GQA group and a single index key head shared across groups (two projection
matrices, `W_q^idx` and `W_k^idx`), scoring by max-pooling at block granularity. It is
trained with a KL alignment loss against the main branch and a **40B-token indexer warm-up**
with a stop-gradient on the Index Branch input so the KL never touches the backbone
[verified]. Results on a 109B/6B MoE trained on 3T tokens: 28.4x per-token attention FLOPs at
1M, and **14.2x prefill / 7.6x decode wall-clock on H800 at 1M**, matching the GQA baseline
[verified]. They ship both a from-scratch variant (MSA-PT) and a checkpoint-conversion
variant (MSA-CPT).

The design lesson for us: **block-level selection keeps KV reads contiguous**; DSA's
token-level selection needs a gather. On B200 with 8 TB/s and a large L2 the gather penalty
is smaller than it was on A100, but it is not zero — and it is precisely the thing the
FlashMLA sparse decode kernel has to do well, which is exactly the kernel the README admits
is unoptimised for B200.

---

## Part 4 — Index sharing across layers (this is our `index_topk_freq=4`)

This is the sub-literature that most directly analyses our configuration. It is new (all
2026) and small, but it is real and two of the papers evaluate on GLM-5 itself.

| Paper | Lab | Venue / year | Hardware | Headline result | Training needed? |
|---|---|---|---|---|---|
| IndexCache (arXiv:2603.12201) | Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li — THUDM/Tsinghua (work done at Z.ai) | arXiv Mar 2026 | **H100 node, SGLang with dp_size=8** | Remove **75% of indexers**; 1.82x prefill / 1.48x decode at 200k on a 30B DSA model; GLM-5 (1/4) >=1.3x both beyond 100k [verified] | Both variants offered |
| You Only Index Once / CLSA (arXiv:2606.06467) | Sun, Zhang, Dong, Wang, Wei | arXiv Jun 2026 | **B200** | One indexer feeds 16 cross-decoder layers; **7.6x decode (3276.8 vs 431.2 tok/s) / 17.1x throughput (1068.1 vs 62.5 tok/s) at 128k**, quality >= dense [verified] | 2-stage distill, no from-scratch pretrain |
| MISA (arXiv:2605.07363) | Zhou, Meng, Xu, Liu, Lu, Zhang, Pei — PKU MuLab + HITSZ | arXiv May 2026 | **8x H200** | Route to h=8 of H^I heads (8x on V3.2 with H^I=64, **4x on GLM-5 with H^I=32**); **3.82x indexer-kernel speedup on one H200**; LongBench 50.85 vs 51.05 (V3.2), 46.43 vs 46.01 (GLM-5) [verified] | **Training-free** |
| HISA (arXiv:2603.28458) | Xu, Meng et al. — PKU MuLab + Tencent | arXiv Mar 2026 | **A100** | Two-stage block-then-token indexing; O(L^2/B + LmB); **2.16x at 4:1, 3.75x at an 8k budget**, both at 64k; LongBench 50.78 vs DSA 51.05 (V3.2), 46.32 vs 46.01 (GLM-5) [verified] | **Training-free** |
| GVR (arXiv:2604.22312) | Cheng, Zhao, Liu, Li, Qiao, Duan, Chen, Chen, Rouhani, Yang — **NVIDIA** | arXiv Apr 2026 | **B200x8, DeepSeek-V3.2 NVFP4, TEP8** | Exact top-K via temporal reuse: 1.88x avg kernel, up to 2.42x per-layer-step, **7.52% TPOT at MTP=0 falling to 3.45% at MTP=3**; bit-exact [verified] | None |
| ReTopK (arXiv:2607.27692) | Yao, Zhou, Shao, Chen, Ning, Feng, Huang, Tang | arXiv Jul 2026 | **single L20, BF16** | Similarity-guided top-K reuse across decode steps; **3.07x attention computation at 128k, K=512, +0.50% ppl** [verified] | None |

### What IndexCache actually measured — read this table twice

IndexCache is the closest published analysis of our setting. Its empirical claim: **adjacent
layers share 70-100% of their selected tokens**, with early and late layers showing much
lower overlap (<=0.4) and "distinct functional blocks exhibiting mutually high overlap
internally but rapid decay at cluster boundaries" [verified].

Two methods:

- **Training-free**: start with every layer running its own indexer (all "F"), then greedily
  convert one layer at a time to "S" (reusing the nearest F layer's indices), each step
  choosing the conversion that minimises LM loss on a small calibration set.
- **Training-aware**: distil each retained indexer with a KL loss against the *average* of
  the attention distributions of all layers it serves — exactly the right objective if one
  index must serve four layers, and mathematically the weighted sum of the per-layer KL
  terms [verified].

**Training-free, on GLM-4.7-Flash (30B-A3B MoE, MLA, 47 layers)** [verified, Table 2]:

| Config | Long Avg | G&R Avg |
|---|---|---|
| Original DSA | 50.2 | 74.6 |
| 1/2 uniform interleave | 47.4 | 74.3 |
| 1/2 greedy-searched | 50.3 | 74.4 |
| **1/4 uniform interleave** | **43.0** | 73.8 |
| **1/4 greedy-searched** | **49.9** | 74.9 |
| 1/8 uniform interleave | 35.3 | 70.0 |
| 1/8 greedy-searched | 46.1 | 73.7 |

**Training-free, on GLM-5 (744B/40B)** [verified, Table 4 — this table was missing from the
previous revision]:

| Config | Long Avg | MRCR v2 | GraphWalks | LongBench v2 | RULER | AA-LCR |
|---|---|---|---|---|---|---|
| Original DSA | 78.4 | 71.1 | 92.7 | 64.5 | 97.7 | 66.2 |
| 1/2 uniform | 78.1 | 72.8 | 90.2 | 65.1 | 97.6 | 64.6 |
| 1/2 greedy-searched | **78.7** | 72.3 | 90.8 | 66.0 | 97.3 | 67.2 |
| **1/4 uniform** | **72.7** | 65.8 | **74.9** | 62.2 | 96.2 | 64.6 |
| 1/4 greedy-searched | 78.0 | 70.8 | 90.3 | 63.7 | 97.6 | 67.6 |

**Training-aware, on the 30B model, uniform interleaving** [verified, Table 3]:

| Config | Long Avg | G&R Avg |
|---|---|---|
| Original DSA (retrained baseline) | 51.0 | 74.2 |
| 1/2 uniform, training-aware | **51.6** | 74.5 |
| 1/2 greedy-searched, training-aware | 50.6 | 73.6 |
| 1/2 uniform, **without** cross-layer loss | 49.8 | 74.5 |
| 1/4 uniform, training-aware | 50.6 | 74.1 |

**Now read the three tables together, because this is where the previous revision went
wrong.** Training-free, uniform 1/4 is a disaster (43.0 on the 30B, 72.7 on GLM-5) and greedy
search rescues it. Training-aware, uniform 1/4 is *fine* (50.6 vs a 51.0 baseline) and the
greedy pattern is actually slightly *worse* than uniform at 1/2 (50.6 vs 51.6). The paper
states the mechanism explicitly: "when the model is retrained with a sharing-aware objective,
the S layers learn to adapt their attention to inherited indices, and the retained indexers
simultaneously learn to produce selections that generalize across their served layers. This
joint adaptation eliminates the layer-specific sensitivity entirely" [verified].

**GLM-5.2's IndexShare was trained in during continued mid-training** [reported, Raschka]. We
are therefore in the training-aware column, where uniform 1-in-4 is validated and calibration
search is not just unnecessary but mildly counterproductive.

One more result worth keeping: removing the cross-layer distillation loss (distilling each
retained indexer only against its own layer) drops Long Avg from 51.6 to 49.8, with AA-LCR
falling from 49.8 to 44.0 [verified]. If we ever retrain, the *multi-layer* objective is the
part that matters, and it is ~30 lines.

Speed at 200k on an H100 node with SGLang dp_size=8, 1/4 retention: prefill 19.5s -> 10.7s
(**1.82x**), per-request decode 58 -> 86 tok/s (**1.48x**), full-KV-cache decode
197 -> 297 tok/s (**1.51x**, part of a 22-51% band across context lengths). At 10k context
prefill is only 1.27x — the benefit is strictly a long-context benefit. On GLM-5 they report
"at least 1.3x improvement in both prefill latency and decode throughput at context lengths
beyond 100K" at 1/4 retention, and ~1.2x end-to-end at 1/2 retention with performance
"nearly identical" to original GLM-5 across the Artificial Analysis Index [verified].

A negative result from their appendix worth internalising: patterns chosen to maximise
*cross-layer similarity* (via dynamic programming) perform no better than naive uniform
interleaving. Only the LM-loss-based greedy search works. "Per-layer output similarity is a
local metric" and does not predict end-to-end damage [verified]. If anyone proposes picking
our pattern by measuring index overlap, this is the paper that says it will not work.

### The other three axes of index amortisation

The literature has now found **four orthogonal axes** for making the indexer cheaper. We are
using exactly one.

1. **Layer axis** — IndexCache, CLSA. Share the index across layers. *We do this
   (`index_topk_freq=4`), and ours is trained in.*
2. **Head axis** — MISA. The indexer has H^I heads; route each query to only h=8 of them
   using cheap block-pooled statistics, then only those 8 do token-level scoring. Cost drops
   from `O(H^I·L)` to `O(H^I·M + h·L)` where `M = ceil(L/B) << L`. **Training-free drop-in**
   on pretrained DSA models. On GLM-5 that is a **4x** reduction in the dominant `h·L` term
   (h=8 of H^I=32), and LongBench actually improves slightly (46.43 vs 46.01) [verified].
   They also offer MISA-dagger, which keeps an enlarged candidate set from the routed pass
   and re-ranks it with the original DSA indexer to recover the exact top-k. The paper
   explicitly notes it is **complementary to IndexCache** because it operates on the head
   axis [verified]. **This is the highest-value unexplored option for us.** Caveat worth
   knowing: MISA's DSA / Block-Sparse / HISA baseline rows are *reproduced from the HISA
   paper*, not independently re-measured [verified].
3. **Granularity axis** — HISA. Two-stage: mean-pool the prefix into blocks of size B, score
   the `ceil(L/B)` block representatives, keep top-m blocks, then run the *original DSA
   indexer* only inside those blocks. Per-query cost `O(L/B + mB)` instead of `O(L)`.
   Training-free, no KV layout change, no architecture change, "a plug-and-play replacement
   for the indexer". 2.16x at 4:1 compression (16k candidate budget) and 3.75x with a fixed
   8k candidate budget, both at 64k context on A100. LongBench 50.78 vs DSA 51.05 vs 49.54
   for a plain block-sparse baseline on DeepSeek-V3.2, and 46.32 vs 46.01 vs 42.67 on GLM-5,
   **both without any finetuning** [verified]. SGLang carries an `--enable-hisparse`
   "hierarchical sparse attention" path [verified in source], though I did not confirm it is
   this paper's algorithm.
4. **Time axis** — GVR, ReTopK. Reuse the previous decode step's selection. GVR's data on
   real DeepSeek-V3.2 traffic: **35-50% average step-to-step top-K overlap in layers 20-60
   (max ~60%), near-zero (1-2%) in layers 0-1** [verified]. Their explanation is the Toeplitz
   structure of RoPE scores — scores depend on `delta = n - m`, so advancing the query by one
   shifts the landscape smoothly.

### GVR in implementable detail (because it is exact, and it is our exact hardware)

Four phases [all verified from the paper; the merged TensorRT-LLM PR describes the shipped
kernel as a 3-phase adaptive threshold search with a 2048-bin parallel histogram]:

1. **Pre-indexed statistics.** Gather the values at the *previous* step's top-K indices and
   compute mean/min/max to form an initial threshold estimate.
2. **Secant threshold search.** Iterate the secant method on the counting function
   `f(T)` = number of logits `>= T`, targeting `K <= f(T) <= C` with `K = 2048` and
   `C = MAX_CANDIDATES = 6144`. Converges in **1-2 iterations on real data**, versus 3-4 full
   passes for radix-select.
3. **Ballot-free candidate collection.** Collect everything above threshold into shared memory
   without `__ballot_sync` barriers (which serialise the memory pipeline), reusing the
   per-thread partial counts cached in phase 2 to avoid a second full scan.
4. **Histogram refinement.** If the candidate count != K, warp-parallel histogram and "snap"
   iterations resolve the exact K-th largest value.

Correctness: **bit-exact top-K index sets vs `torch.topk` for N = 8k-131k** [verified] — which
matters because GLM-5's own report says they use deterministic `torch.topk` as the default
top-k operator in the DSA indexer [verified], and a heuristic that changed the selection would
break whatever that determinism is protecting.

Results on B200x8 with DeepSeek-V3.2 NVFP4 under TEP8: 1.88x average single-operator, up to
2.42x per layer per step (layer 22 averages 2.04x), and end-to-end TPOT reduction of **5.47%
(64k) / 7.52% (100k) at MTP=0, 4.36% / 6.30% at MTP=1, and 2.40% / 3.45% at MTP=3**
[verified]. Stated limitations: minimal benefit below 16k; single-CTA design limits occupancy
at large batch; degraded prediction in layers 0-1 [verified].

**For us**: the min-latency (C1) objective is exactly the TEP8 min-latency config they
measured. But we run EAGLE 3-1-4, which is nearest their MTP=3 row, so budget ~2.4-3.5%, i.e.
~9-13 tok/s on our 365. Cheap, exact, worth doing — but not the 7.5% the headline suggests.

### CLSA / "You Only Index Once" — the aggressive version, and its caveats

CLSA builds on YOCO-style KV sharing: a 32-layer / 4B model (hidden 2560, FFN 7680, 20
attention heads, 4 KV heads) with 16 self-decoder layers and 16 cross-decoder layers;
**one indexer, computed once, shared by all 16 cross-decoder layers** (a 16:1 ratio, vs our
4:1). Adapted from a dense YOCO checkpoint by (1) distilling the indexer against dense
cross-attention with the backbone frozen (2500 steps, distillation loss only), then (2) joint
LM + distillation fine-tuning with `lambda = 0.1` (2500 steps) — **no from-scratch
pretraining** [verified]. 2048 selected tokens. Measured on **NVIDIA B200** at 128k: decode
throughput 3276.80 vs 431.16 tok/s (**7.6x**) and end-to-end throughput 1068.06 vs 62.53
tok/s (**17.1x**), with quality *above* dense YOCO on ARC-C (0.465 vs 0.461), GSM8K
(0.470 vs 0.430), DROP (0.391 vs 0.387) and RULER-32k (53.1 vs 52.3) [verified].

The sentence most relevant to us, from their per-layer latency analysis at 128k:
**"the amortized top-k becomes efficient. Without amortization, the unamortized top-k stage
can be comparable to or even larger than dense attention"** [verified;
**[CORRECTED]** — the previous revision quoted a differently-worded sentence about
"token-level selection per layer matching dense attention cost", which does not appear in the
paper]. That is the same phenomenon my Part 8 arithmetic produces from first principles, and
it is specifically about the top-k *selection kernel*, not the indexer scoring GEMM.

Caveats: 4B model, and it requires the YOCO KV-sharing backbone. Not a drop-in for us. But it
is the strongest evidence that 4:1 is conservative and that the quality risk of deeper sharing
is smaller than intuition suggests **when the indexer is distilled for the job** — which is
the same conclusion IndexCache's training-aware table reaches.

---

## Part 5 — Training-free KV selection, eviction and compression

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| H2O (arXiv:2306.14048) | Zhang, Sheng et al. — UT Austin / Stanford / CMU / Meta | NeurIPS 2023 | not stated; OPT-6.7B / OPT-30B | 20% heavy hitters; up to 29x / 29x / 3x throughput vs DeepSpeed Zero-Inference / HF Accelerate / FlexGen; up to 1.9x latency [verified] | No — vLLM issue #3532 open, never merged |
| StreamingLLM (arXiv:2309.17453) | Xiao, Tian, Chen, Han, Lewis — MIT / Meta / CMU | ICLR 2024 | not stated | 4M tokens stable; **22.2x vs sliding-window recomputation** [verified] | **Sinks yes, eviction no** |
| SnapKV (arXiv:2404.14469) | Li, Huang et al. — UIUC / Cohere | arXiv 2024 | **A100-80GB** | 3.6x generation speed, 8.2x memory at 16K inputs; 380K context on one A100-80GB [verified] | Partially (some engines) |
| PyramidKV (arXiv:2406.02069) | Cai, Zhang et al. | arXiv 2024 | not stated | Matches full cache at **12% KV** on LongBench; at 0.7% KV, +20.5 pts on TREC vs baselines [verified] | No |
| Quest (arXiv:2406.10774) | Tang, Zhao, Zhu, Xiao, Kasikci, Han — MIT / UW | **ICML 2024** | not stated in abs | 2.23x self-attention, **7.03x inference latency reduction** [verified] | Reference implementation; ideas widely copied |
| DuoAttention (arXiv:2410.10819) | Xiao, Tang, Zuo, Guo et al. — MIT Han Lab | arXiv 2024 | not stated in abs | Splits heads into retrieval heads (full KV) and streaming heads (sink + window) [verified abstract] | Ideas adopted (LServe) |
| LServe (arXiv:2502.14866) | Yang, Guo, Tang, Hu et al. — MIT Han Lab | **MLSys 2025** | not stated in abs | Unified static + dynamic block sparsity for prefill and decode; half the heads become streaming heads; **2.9x prefill, 1.3-2.1x decode over vLLM** [verified] | Open source (omniserve) |
| SeerAttention-R (arXiv:2506.08889) | Gao, Guo, Cao, Xia — Microsoft Research Asia | arXiv Jun 2025 | **H100** | Self-distilled gating for *reasoning decode*, trained on 0.4B tokens; near-lossless AIME at 4K token budget; **up to 9x over FlashAttention-3 at 90% sparsity** [verified] | Open source (microsoft/SeerAttention) |
| UNIQUE (arXiv:2605.27740) | Deng, Ling, Fan, Li — **Microsoft** | arXiv May 2026 | **H100**, 32 q / 8 kv heads, dim 128 | Page score `q·mean_p + 0.5·||q||·std_p`; **11.4x attention kernel at 32k vs dense FlashInfer, 5.3x e2e vs vLLM**; LongBench-Pro 36.58 @512-page budget vs Quest 21.72 / H2O 29.04 / InfLLM 34.99; 97.0% of full attention [verified] | Research |
| Hold Onto That Thought (arXiv:2512.12008) | Liu, Palnitkar, Rabbani et al. — UMD / UChicago / Utah | arXiv Dec 2025 | **RTX A6000 48GB** | **Failure catalogue** — see below [verified] | evaluation |

### The mechanisms, and the one thing they all get wrong for us

- **H2O**: accumulate attention scores per key across the generated sequence; the top ~20% by
  accumulated score are "heavy hitters"; keep those plus a recent window, evict the rest.
  Formulated as dynamic submodular maximisation with a guarantee.
- **StreamingLLM**: the first few tokens absorb enormous attention mass regardless of
  semantics ("attention sink"), because softmax must put its mass *somewhere* and early tokens
  are visible to every query. Keep the first ~4 tokens plus a rolling window, and **re-index
  positions by position-within-cache rather than absolute position**. That last detail is what
  makes it work at all.
- **SnapKV**: use a short observation window at the *end* of the prompt, compute its attention
  to all earlier prompt tokens, pool (1-D pooling over the score vector, which preserves
  contiguous spans instead of scattered singletons), keep the top positions per head.
  Prompt-time only; the cache is fixed after prefill.
- **PyramidKV**: allocate a *layer-varying* budget — more cache in early layers, less in late
  layers — motivated by "pyramidal information funneling".
- **Quest**: per KV *page*, store elementwise min and max of the keys. At decode, upper-bound
  the page's max possible score by `sum_i max(q_i·min_i, q_i·max_i)`, rank pages by that
  bound, load only the top pages. This is a *selection* method, not an eviction method — the
  full cache stays resident, only the reads are sparse.
- **DuoAttention / LServe**: classify heads offline into *retrieval* heads that need the full
  KV and *streaming* heads that only need sinks plus a local window. LServe converts about
  half the heads to streaming in both prefill and decode, then layers a query-centric
  hierarchical page-selection policy on top, and finds "only a constant number of KV pages is
  required to preserve long-context and reasoning capabilities, irrespective of context
  length" [verified abstract]. That constant-budget finding is the same shape as DSA's fixed
  k=2048.
- **SeerAttention-R**: the one built for *our* failure mode. A lightweight plug-in gating
  module, self-distilled from the model's own attention, with query pooling removed so it
  works autoregressively; trained on 0.4B tokens without touching the original parameters.
  Near-lossless AIME accuracy at a 4K token budget with block sizes 64/128, and a TileLang
  decode kernel reaching **up to 9x over FlashAttention-3 on H100 at 90% sparsity**
  [verified]. It is the learned-gating counterpart to DSA's learned indexer, and it is what
  you would reach for on a GQA model that was not pretrained sparse.
- **UNIQUE (2026)**: the modern refinement of Quest's idea. Score a page by
  `q·mean_p + lambda·||q||_2·std_p` with `lambda = 0.5` — the mean term is the cheap
  representative, the std term compensates for averaging diluting a single important key.
  Adds a fused criticality kernel (1.73x over naive) and a radix-based top-k (2.0x over
  `flashinfer.top_k`, 4.8x over `torch.topk`, on H100). Training-free deployment *plus* a
  sparsity-aware training mode using a soft sigmoid mask around the top-k boundary so
  gradients flow without auxiliary losses [verified]. Their diagnostic table is the useful
  part: Quest holds up on RULER (84.14 at a 512-page budget) but collapses on LongBench-Pro
  (21.72), i.e. min/max bounds are fine for needle retrieval and bad for realistic reasoning
  context; H2O shows the opposite. **Transfer caveat: measured on a GQA shape (32 query
  heads, 8 KV heads, dim 128), not MLA.**

**The structural distinction that decides everything: selection vs eviction.** Quest, UNIQUE,
LServe, SeerAttention-R and DSA *select* which resident pages to read — the information is
never destroyed, so a later query can retrieve a token an earlier query ignored. H2O, SnapKV,
PyramidKV and StreamingLLM *evict* — the information is gone. For a reasoning model that
thinks for thousands of tokens, eviction decisions made at prompt time or early-generation
time are made without knowledge of what the model will need. DSA is a selection method. That
is the right side of the line, and it is why we should not bolt an eviction method onto it.

---

## Part 6 — Linear and hybrid attention, only where production uses it

| Model / paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Kimi Linear (arXiv:2510.26692) | Kimi Team — Moonshot | arXiv Oct 2025 | not stated for speed tests | KDA (channel-wise gated delta rule) **3:1 with MLA-NoPE** full layers; 48B total / 3B active; **KV cache -75%, 6x TPOT at 1M (1.84 ms vs 11.48 ms)**, prefill **2.3x @512k / 2.9x @1M**; MMLU-Pro 51.0 vs 47.2 MLA, RULER-128k 84.3 vs 81.3 [verified] | **Yes** — weights + vLLM impl released |
| Qwen3-Next (vLLM blog, Sept 2025) | Qwen / Alibaba | blog | not stated | Gated DeltaNet interleaved 3:1 with full attention [reported] | **Yes** |
| gpt-oss-120b (arXiv:2508.10925) | OpenAI | arXiv 2025 | — | MoE transformer, 117B/5.1B active, MXFP4 MoE weights [verified from abstract; **layer/attention details not in the page I fetched — I will not state its sink or window config from memory**] | Yes |

The relevant fact for us is narrow: **hybrid linear attention is now a mainstream production
choice, and the full-attention layers in those hybrids are MLA.** Kimi Linear's 1-in-4 full
layers use MLA with **NoPE** — no positional encoding at all in the full layers, since the
KDA layers carry position [verified]. That is a structurally different bet from ours
(sparse-everywhere) and it wins on a different axis: KDA's state is fixed-size, so KV memory
stops growing with context, whereas DSA's KV still grows linearly with L even though the
*reads* are O(k).

**For our roadmap, this is no longer speculation.** SGLang's argument parser dispatches
`KimiLinearForCausalLM` and **`KimiK3ForConditionalGeneration`** through the same hook
(`kimi_k3_hook.apply_kimi_k3_linear_attn_defaults` and `apply_kimi_k3_spec_backend_defaults`),
and its `LINEAR_ATTN_KERNEL_BACKEND_CHOICES` include `flashkda`, `nvidia_kda` and `ptx_kda`
[all verified in source]. **Kimi K3 is a hybrid linear-attention model in the Kimi Linear
lineage.** So our KV manager will need to handle **two cache types in the same model** — a
fixed-size recurrent state for the linear layers and a paged latent cache for the MLA layers
— with different eviction, different prefix-cache semantics (recurrent state is *not*
prefix-shareable in the usual way; you must snapshot the state at the shared-prefix boundary),
and different offload behaviour. That is a serving-stack change, not a kernel change, and it
should be scoped now rather than when the weights land.

---

## Part 7 — KV cache management: paging, reuse, offload, routing

| Paper / system | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| PagedAttention / vLLM (arXiv:2309.06180) | Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica — UC Berkeley | **SOSP 2023** | not stated in abs | **2-4x throughput** vs FasterTransformer / Orca; near-zero KV fragmentation [verified] | Universal |
| vAttention (arXiv:2405.04437) | Prabhu, Nayak, Mohan, Ramjee, Panwar — Microsoft Research India | **ASPLOS 2025** | not stated in abs | Contiguous virtual KV via CUDA VMM APIs; **up to 1.23x** over PagedAttention-based FA/FlashInfer kernels [verified] | Research / partial adoption |
| SGLang + RadixAttention (arXiv:2312.07104) | Zheng, Yin, Xie, Sun, Huang, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng | arXiv 2023 | not stated in abs | **up to 6.4x throughput** via radix-tree KV reuse + compressed FSM [verified] | Universal |
| Mooncake (arXiv:2407.00079) | Qin, Li, He, Zhang, Wu, Zheng, Xu — **Moonshot AI** | arXiv 2024 (v4 Sep 2025) | Kimi production | KVCache-centric PD disaggregation over CPU DRAM/SSD; **+525% throughput in simulation, +75% real request capacity** [verified] | **Yes — serves Kimi** |
| CacheBlend (arXiv:2405.16444) | Yao, Li, Liu, Ray, Cheng, Zhang, Du, Lu, Jiang — UChicago | arXiv 2024 | not stated in abs | **Non-prefix** KV reuse with selective recompute; TTFT 2.2-3.3x lower, throughput 2.8-5x, quality preserved [verified] | Yes — in LMCache |
| LMCache (docs.lmcache.ai) | LMCache project | docs | — | Tiered GPU / CPU DRAM / local disk / Redis-Valkey / S3; CacheBlend for non-prefix reuse [verified] | Yes — vLLM production integration |
| Irminsul (arXiv:2605.05696) | Ma, Eitzinger, Köstler | arXiv May 2026 | not stated | **MLA-native position-independent caching** over SGLang's radix cache; ~83% of prompt tokens recovered above exact-prefix on agentic traffic; 63% prefill energy per hit [verified abstract] | Research |
| Understanding Bottlenecks for KV Offloading (arXiv:2601.19910) | Meng (UPenn), Lee (UPenn), Wang (Intel) | arXiv Dec 2025, submitted to MLSys 2026 | GPU + PCIe | Defines `kappa_crit`; **99% of latency is data transfer; GPU at 28% of rated TDP** when serving offloaded requests [verified] | analysis |
| Unified KV Pooling (arXiv:2606.14779) | Kang et al. — Korea Univ. | arXiv Jun 2026 | host DRAM + SSD | SPDK/KV-passthrough unified pool; **~4.1x TTFT** vs SOTA; **84% of SSD KV retrieval time was in the kernel filesystem**; blocked I/O down up to 23.2x [verified] | Research |

### What actually applies to us

**Paging.** PagedAttention is settled. The one thing to know for DSA: vLLM only supports
**block size 64** for DeepSeek-V3.2, because FlashMLA's sparse kernel is tailored to it and
the indexer cache stores scale factors per 64-token block [verified]. If our KV manager
assumes 16 or 128 anywhere, that is a bug waiting to happen. vAttention's 1.23x is real but
small and it does not compose trivially with a sparse gather kernel; not a priority.

**Prefix caching.** Our measured 1.54x is a strict-prefix number. Three ways to extend it:

1. **CacheBlend / non-prefix reuse.** Reuse cached KV for chunks that are *not* a prefix, then
   selectively recompute a small subset of tokens (the ones whose KV deviates most) to repair
   cross-chunk attention. 2.2-3.3x TTFT, 2.8-5x throughput, quality preserved [verified].
   This is what makes RAG-style multi-chunk prompts cacheable. Shipped in LMCache.
2. **Position-independent MLA caching (Irminsul).** MLA's cached row is `c_KV` (position-free)
   plus a 64-dim `k_r` correctable by a closed-form delta rotation. So you can content-hash a
   token's KV over CDC-chunked segments and *re-position it* at cache-hit time by applying the
   delta rotation to the 64-dim part only. That converts prefix caching into
   **content-addressed** caching, which is exactly what agentic traffic (tools inserted
   mid-context, turns reordered) needs. Their framing of the problem is worth quoting:
   "Agentic LLM workloads put bit-identical tokens at shifted positions every turn, voiding
   prefix caches at the first byte of divergence"; they cite operator reports of TTFT spikes
   of 10-16s [verified abstract]. Tested on DeepSeek-V2-Lite, Kimi Moonlight-16B-A3B and
   JoyAI-Flash — all far smaller than us. **Highest-leverage unexplored item in this section.**
3. **Cache-aware routing.** SGLang's RadixAttention plus cache-aware scheduling is the
   baseline. The 2026 literature has moved on to the second-order problem: BanaServe
   (arXiv:2510.13223, *Software: Practice and Experience* 2026) documents that prefix-cache-
   aware routing skews load distribution because high-hit-rate prefill nodes attract
   disproportionately more requests [reported]; "Preserving Admission Responsibility in
   Multi-Tenant LLM Prefix Caches" (arXiv:2608.01657, Wang & Buyya — the system is called
   PrefixShield) measures a cache-pollution failure where victim hit ratio collapses from
   84.87% to 4.92% under LRU and recovers +9.39 points with admission control [reported];
   PrefixPlace (arXiv:2608.01655, same authors) reports 40.3% better materialisation-cost
   savings than vLLM's automatic prefix caching [reported]. If we run multi-tenant, the
   *pollution* result is the one to act on.

**Offloading — read the bad news first.** arXiv:2601.19910 is the sober analysis: for KV
offload to host DRAM over PCIe, **99% of latency is data transfer and the GPU sits at 28% of
rated TDP** [verified]. They define `kappa_crit`, the cached-to-prefill token ratio above
which you are memory-bound, and show typical workloads exceed it "by orders of magnitude".
On our box: PCIe Gen5 x16 is ~64 GB/s per direction against 8 TB/s of HBM — a **125x** gap
[inferred]. Moving one 128k MLA cache (6.79 GB FP8, computed in Part 8) over PCIe takes
~106 ms [inferred]; recomputing that prefill on 8 B200s takes on the order of a few hundred
ms. **So host-DRAM offload of MLA KV is roughly break-even against recompute, and only wins
when the prefill is expensive relative to the cache size** — which for MLA is unusual
precisely *because* MLA's cache is so small relative to its prefill cost. Offload is a much
better deal for GQA models than for MLA models. Say this out loud before anyone builds it.

---

## Part 8 — Where KV memory and attention bandwidth actually bind, on 8xB200, at 128k

All arithmetic below is **[inferred]** from verified config numbers. Assumptions stated.
Shapes: 79 layers, 22 indexer layers (`index_topk_freq=4`), MLA latent `d_c + d_h^R = 576`,
`index_head_dim = 128`, `index_topk = 2048`, B200 at 8 TB/s and 183 GB.

> **Assumption flagged:** MISA verifies GLM-5 has `H^I = 32` indexer heads but does not state
> its `index_head_dim`. The byte-traffic arithmetic in 8.1-8.4 depends only on
> `index_head_dim = 128` (DeepSeek-V3.2's verified value), so it is unaffected by the head
> count. Only 8.5's FLOP arithmetic depends on `H^I`, and is corrected there.

### 8.1 Per-token footprint

| Component | Bytes / token / layer | Layers | Bytes / token | @128k (131,072 tok) |
|---|---|---|---|---|
| MLA latent, BF16 (576 x 2) | 1,152 | 79 | 91,008 (88.9 KiB) | **11.93 GB** |
| MLA latent, FP8 layout (512 + 16 + 128, per vLLM) | 656 | 79 | 51,824 (50.6 KiB) | **6.79 GB** |
| Indexer K, FP8 (128 + scales) | ~132 | 22 | 2,904 | **0.38 GB** |
| Indexer K if *no* sharing | ~132 | 79 | 10,428 | **1.37 GB** |
| *(reference)* GQA-8, 128 dim, BF16, 80 layers | 4,096 | 80 | 327,680 (320 KiB) | **42.95 GB** |

MLA-BF16 is **3.6x** smaller per token than GQA-8; MLA-FP8 is **6.3x** smaller.

### 8.2 Capacity: TP8 vs DP8 attention

Budget per GPU: 183 GB. NVFP4 weights for 744B params at ~0.55 B/param including scales
≈ 410 GB total ≈ **51 GB/GPU**. Reserve **20 GB/GPU** for activations, EAGLE draft model,
comm buffers, CUDA graphs, workspace. **Free ≈ 112 GB/GPU.**

Per-token cost with FP8 latent + shared indexer = 51,824 + 2,904 = **54,728 B/token**.

- **TP8 attention (KV replicated across all 8 ranks):** usable KV pool = 112 GB **once**.
  `112e9 / 54,728 = 2.05M tokens` -> **~15.6 concurrent 128k sequences**.
  With BF16 latent (93,912 B/token): 1.19M tokens -> **~9.1 sequences**.
- **DP8 attention (each rank owns distinct sequences, EP for the MoE):** pool = 8 x 112 =
  896 GB -> `16.4M tokens` -> **~125 concurrent 128k sequences**.

**This is an 8x difference in long-context concurrency for zero quality cost.** It is the
single largest number in this document. At short context it matters much less, which is why it
can hide: at 4k context TP8 still gives ~500 concurrent streams.

### 8.3 Bandwidth per decode step, one sequence at 128k

| Read | Bytes | Time @8 TB/s |
|---|---|---|
| Dense MLA attention (all 131,072 tokens, FP8) | 6.79 GB | 849 us |
| **DSA sparse attention** (k=2048, FP8, 79 layers) | 2048 x 656 x 79 = **106.1 MB** | **13.3 us** |
| **Lightning indexer** (L=131,072, FP8, 22 layers) | 131,072 x 132 x 22 = **380.6 MB** | **47.6 us** |
| Lightning indexer if *no* sharing (79 layers) | **1.37 GB** | 171 us |

**DSA + index sharing turns 849 us of attention KV traffic into 61 us — a 13.9x reduction.
But 78% of what remains is the indexer, not the attention.**

### 8.4 The crossover: when does the indexer cost more than the attention it feeds?

Sparse attention bytes are **constant in L**: `k · B_kv · n_layers`.
Indexer bytes are **linear in L**: `L · B_idx · n_idx_layers`. Setting them equal:

```
L* = (k · B_kv · n_layers) / (B_idx · n_idx_layers)
```

| Config | L* (crossover context) |
|---|---|
| FP8 latent (656 B), 22 index layers | **~36,500 tokens** |
| BF16 latent (1152 B), 22 index layers | **~64,200 tokens** |
| FP8 latent, 79 index layers (no sharing) | **~10,200 tokens** |
| BF16 latent, 79 index layers (no sharing) | **~17,900 tokens** |

Read the first and third rows together: **`index_topk_freq=4` moves the point at which the
indexer starts dominating from 10k context to 36k.** That is what our config is buying, and it
is why removing it would be catastrophic for long-context serving, not merely 5% worse. It is
also the number to weigh against SGLang's refusal to run two-batch overlap alongside index
sharing.

### 8.5 FLOPs and arithmetic intensity of the indexer — **[CORRECTED for GLM-5 shapes]**

Per layer, per query token: `FLOPs = 2 · H^I · d_I · L`. Bytes read (FP8 indexer K, one key
vector per token): `d_I · L`. So

```
Indexer arithmetic intensity = 2 · H^I  FLOP/byte
```

**Independent of L, of d_I, and of batch size.** With `H^I = 64` (DeepSeek-V3.2) that is 128
Op/B; **with `H^I = 32` (GLM-5, verified from MISA) it is 64 Op/B.** B200's BF16 ridge is
281.25 Op/B and its FP8 ridge is ~562 Op/B. **On GLM-5 shapes the lightning indexer sits ~8.8x
below the FP8 ridge and can never climb — batching does not help, because each sequence has
its own indexer K cache.** It is a permanently memory-bound kernel bolted onto an attention
that MLA has made compute-bound. That single sentence explains why every 2026 paper in Part 4
attacks the indexer and not the attention. The previous revision used H^I=64 here and so
understated the imbalance by 2x.

Absolute FLOPs at 128k, **DeepSeek-V3.2 shape** (verified config: 61 layers, H^I=64, 128
attention heads, all layers indexed):

- Indexer: `61 · 2 · 64 · 128 · 131,072` = **131 GFLOP / token**
- Sparse absorbed MLA: `61 · (2·128·576·2048 + 2·128·512·2048)` = **34.8 GFLOP / token**
- **The indexer costs 3.8x the FLOPs of the attention it selects for.**

At **our shape** (79 layers, 22 indexed, H^I=32):

- Indexer: `22 · 2 · 32 · 128 · 131,072` = **23.6 GFLOP / token**
- Sparse absorbed MLA, *if* per-layer attention FLOPs match V3.2's: `79 · 5.704e8` =
  **45.1 GFLOP / token** [inferred; GLM-5's head count is "reduced by 1/3" from 128 and its
  head dim is 256, which I could not pin down precisely enough to do better than this
  order-of-magnitude comparison]
- **Ratio ≈ 0.52 — the indexer costs roughly half the FLOPs of the attention, not 3.8x.**

So `index_topk_freq=4` plus GLM-5's halved indexer head count together turn a 3.8x FLOP
overhead into a 0.5x one. That is a 7x swing, and it is the quantitative justification for
both design choices. It also explains GLM-5's own modest claim that DSA delivers only
"roughly 1.5-2x" attention compute reduction for long sequences [verified] — far less than the
naive `L/k = 64x` you would expect from top-k alone, because on the *un*-shared, 64-head
configuration the indexer eats most of it.

Note carefully that the FLOP ratio and the *byte* ratio point in opposite directions: by bytes
(8.3) the indexer is 78% of the remaining traffic; by FLOPs it is a third. That is exactly what
"the indexer is memory-bound and the attention is not" means, and it is why the right fixes are
the ones that cut indexer *reads* (head routing, hierarchical indexing, temporal reuse) rather
than indexer arithmetic.

### 8.6 At concurrency 64

Sparse attention scales linearly with batch (each sequence selects different tokens):
`64 x 106.1 MB = 6.79 GB` -> **849 us/step**. Indexer: `64 x 380.6 MB = 24.4 GB` ->
**3.05 ms/step**. Our measured 40.8k tok/s at C64 implies ~1.57 ms/step, so **at 128k context,
C64, the indexer alone exceeds the entire current step budget by ~2x** and the combined
attention path is ~2.5x the budget. Concretely: **our C64 aggregate throughput cannot survive a
move to 128k contexts without MISA-style head routing, HISA-style hierarchical indexing,
deeper sharing, or all three.** Without any index sharing at all it would be 11 ms/step — 7x
over budget.

### 8.7 What binds, summarised

| Regime | Binding resource |
|---|---|
| C1, short context (<16k) | Dense GEMM + collectives. Attention is a rounding error. Matches our 37.1% / 19.6% / 10.9% profile. GVR also reports minimal benefit below 16k. |
| C1, 128k | Indexer HBM bandwidth (47.6 us) > sparse attention bandwidth (13.3 us). Top-k selection kernel is the next term (GVR: 2.4-3.5% TPOT at our speculation depth). |
| C64, short context | MoE expert GEMMs + collective skew. |
| C64, 128k | **Indexer bandwidth, by a factor of 2 over everything else.** |
| Any regime, capacity | KV pool size, and therefore TP8-vs-DP8 attention, long before HBM bandwidth. |
| With EAGLE 3-1-4 | Attention intensity multiplies by draft length -> attention moves compute-bound; the indexer does **not**, so speculation *increases* the indexer's relative share. Also shrinks GVR's payoff (7.52% -> 3.45%). |

That last row is the non-obvious one. Speculative decoding is worth 3.09x to us because it
amortises weight loading over multiple tokens. It does **not** amortise the indexer, because
the indexer must score the full prefix for the query position(s). If we index once per verify
step and share that index across all draft positions — justified by GVR's 35-50% step-to-step
overlap in deep layers, which should be *higher* within a single verify window — the indexer's
cost per accepted token drops by the acceptance length, currently ~2.76 for GLM-5's MTP
[verified]. That is a **2.76x reduction in the dominant long-context term**, and it is the
single highest-value unimplemented idea I found in this literature for our specific stack.

---

## What is NOT worth it

1. **Attention-score KV eviction (H2O, SnapKV, PyramidKV, StreamingLLM-as-eviction) on a
   reasoning model.** "Hold Onto That Thought" (arXiv:2512.12008) tested these on
   DeepSeek-R1-Distill-Qwen-7B/14B, R1-Distill-Llama-8B, Nemotron-Nano-8B and
   Llama-3.1-8B-Instruct across GSM8K, MATH-500, FOLIO, DROP, ReClor, StrategyQA,
   CommonsenseQA and OpenBookQA, on an **RTX A6000 48GB**. Verified results at budgets
   128-512: **PyramidKV scores 0.00 on MATH-500 with R1-Distill-Llama-8B at every budget from
   128 through 512** (Appendix Table 11) and 0.00/0.01 on GSM8K with the same model;
   **KNorm scores 0.00 on GSM8K at budgets 128 and 256 vs 0.70 full cache**, and 0.00 on
   MATH-500 at 128 and 256 on a model whose full-cache score is 0.16. The authors document
   that KNorm "results in long circular babble... that never produces an answer" [verified].
   **[CORRECTED]:** the previous revision claimed StreamingLLM scored **0.00** on ReClor at
   budget 128; the actual figures are **0.02 and 0.03** on the two reasoning models, against
   full-cache 0.45 and 0.48. Still catastrophic, but the number was wrong and is now right.
   Root cause is unchanged: eviction commits at prompt time or early generation; a reasoning
   model's critical tokens are produced thousands of steps later. **Selection (Quest, UNIQUE,
   LServe, SeerAttention-R, DSA) does not have this failure mode because nothing is
   destroyed.** We already have DSA. Do not add eviction.
2. **Host-DRAM / NVMe offload of the MLA KV cache.** PCIe Gen5 x16 at ~64 GB/s vs 8 TB/s HBM
   is a 125x gap [inferred]; arXiv:2601.19910 measures 99% of latency as transfer and the GPU
   at 28% TDP under offload [verified]. MLA's cache is already 3.6-6.3x smaller than GQA's, so
   the thing offload is designed to solve barely exists for us. Sound for GQA models at 320
   KiB/token; roughly break-even against recomputing prefill at our 50.6 KiB/token. The
   exception worth revisiting: cross-*request* KV reuse (Mooncake, LMCache) where the
   alternative is a cache miss and a full prefill, not a resident cache — different economics,
   and it does pay.
3. **Running IndexCache's greedy layer search on our model. [CORRECTED — this item is new.]**
   The greedy search exists to repair the distributional shift you get when you retrofit
   sharing onto a model trained with per-layer indexers. IndexCache's own training-aware
   results show uniform interleaving matching or beating the searched pattern once the model
   is trained for sharing (51.6 vs 50.6 Long Avg at 1/2 retention) and states outright that
   "the pattern sensitivity observed in training-free IndexCache vanishes with training"
   [verified]. GLM-5.2 trained IndexShare in during continued mid-training [reported]. A
   calibration sweep on our model measures a problem we do not have, and the paper's own
   appendix shows a *similarity*-based pattern search — the obvious cheap proxy — performs no
   better than uniform [verified].
4. **Going to `index_topk_freq=8` without a training pass.** IndexCache measured training-free
   1/8 retention at **46.1 Long Avg with the greedy pattern and 35.3 with uniform, vs 50.2
   baseline** [verified]. They published no distilled 1/8 number, so treat 1/8 as unvalidated
   in both regimes. The knee is exactly where we are sitting. CLSA's 16:1 result is the
   counter-evidence, but it is a 4B model on a YOCO backbone with a purpose-built distillation,
   not a drop-in.
5. **Porting NSA/MoBA/MSA-style architecture changes to an already-trained model.** NSA
   requires pretraining with the three-branch structure; MoBA requires continued training and
   is explicitly not a drop-in; MSA needs a 40B-token indexer warm-up [verified]. DSA already
   gave us the sparsity. The marginal architectures are for the *next* pretrain, not this
   deployment. The training-free improvements (MISA, HISA, GVR, ReTopK, UNIQUE) are where our
   engineering time goes.
6. **Chasing FlashAttention-4 for the MLA decode path.** FA4 is excellent and it is the right
   kernel for dense prefill and for GQA models, but the paper explicitly does not cover head
   dim 576/512, does not cover MLA, and lists sparse attention as future work [verified]. Two
   further reasons to temper enthusiasm, both from the paper itself: **"newer versions of cuDNN
   have incorporated many of the techniques described in this paper, yielding similar
   performance to FA4"**, and B300/GB300 double MUFU throughput, which erodes the
   software-`exp` argument [verified]. Its transferable content for us is the *techniques* —
   polynomial `exp` on FMA units and conditional softmax rescaling at `tau = log2(256)` —
   which should be ported into whatever sparse-MLA decode kernel we end up using, since that
   kernel also runs a softmax over 2048 entries per head per layer. The one *actionable*
   FA4 fact: **FA3 does not run on B200**, so check nothing in our stack is silently falling
   back.
7. **vAttention as a replacement for PagedAttention in our stack.** 1.23x over
   PagedAttention-based kernels [verified] is real but small, and it requires kernels that
   assume contiguous virtual KV — our hot path is FlashMLA's sparse gather, built around the
   64-token paged layout with per-block FP8 scales [verified from vLLM]. Integration cost
   exceeds benefit.
8. **Treating Helix Parallelism's numbers as measurements.** The paper's 1.5x TTL and 32x
   batch figures for DeepSeek-R1 at 1M come from "an in-house high-fidelity simulator" on
   GB200 NVL72, not hardware [verified]. The *architectural* argument (MLA + TP>1 duplicates
   KV; shard along sequence instead) is sound and worth acting on; the speedup numbers are
   not evidence.

---

## Sources

Every URL below was fetched during this pass.

**Kernels and the online-softmax lineage**
- Milakov & Gimelshein (NVIDIA), "Online normalizer calculation for softmax", arXiv:1805.02867 — https://arxiv.org/abs/1805.02867
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022, arXiv:2205.14135 — https://arxiv.org/abs/2205.14135
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", arXiv:2307.08691 — https://arxiv.org/abs/2307.08691
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", arXiv:2407.08608 — https://arxiv.org/abs/2407.08608
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling", arXiv:2603.05451 — https://arxiv.org/html/2603.05451v1
- Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference", PyTorch blog, Oct 2023 — https://pytorch.org/blog/flash-decoding/
- Ye, Chen, Lai, Lin et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving", MLSys 2025, arXiv:2501.01005 — https://arxiv.org/abs/2501.01005
- DeepSeek, FlashMLA — https://github.com/deepseek-ai/FlashMLA

**KV cache shape: MQA / GQA / MLA, and how to shard it**
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150 — https://arxiv.org/abs/1911.02150
- Ainslie et al. (Google), "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023, arXiv:2305.13245 — https://arxiv.org/abs/2305.13245
- DeepSeek-AI, "DeepSeek-V2", arXiv:2405.04434 — https://arxiv.org/abs/2405.04434
- DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437 — https://arxiv.org/abs/2412.19437
- Geens & Verhelst (KU Leuven), "Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention", Electronics Letters 2025, arXiv:2506.02523 — https://arxiv.org/html/2506.02523v1
- Yun et al. (SNU + UIUC), "Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts" (v1 title: "The New LLM Bottleneck"), arXiv:2507.15465 — https://arxiv.org/html/2507.15465v3
- Zadouri, Strauss, Dao, "Hardware-Efficient Attention for Fast Decoding", arXiv:2505.21487 — https://arxiv.org/abs/2505.21487
- Bhatia, More, Borkar, Mitra et al. (NVIDIA), "Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding", arXiv:2507.07120 — https://arxiv.org/html/2507.07120v1
- Meng (PKU MuLab), "GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding", arXiv:2605.15250 — https://arxiv.org/abs/2605.15250
- Han, Zhao, Zhou, Li, Sun, "QK-Normed MLA: QK normalization without full key caching", arXiv:2606.16310 — https://arxiv.org/abs/2606.16310
- Ma, "A Training-Memory Regression in MLA Sequence Parallelism: Why Megatron-Core Forbids Absorption, and LAGA", arXiv:2607.17644 — https://arxiv.org/abs/2607.17644

**Sparse attention (trained)**
- Yuan et al. (DeepSeek + PKU + UW), "Native Sparse Attention", ACL 2025, arXiv:2502.11089 — https://arxiv.org/html/2502.11089v2
- Lu et al. (Moonshot AI + Tsinghua + Zhejiang), "MoBA: Mixture of Block Attention for Long-Context LLMs", arXiv:2502.13189 — https://arxiv.org/html/2502.13189v1
- DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", arXiv:2512.02556 — https://arxiv.org/html/2512.02556v1
- DeepSeek-V3.2-Exp reference implementation and config — https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py and .../config_671B_v3.2.json
- GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763 — https://arxiv.org/html/2602.15763v2
- Lai et al. (MiniMax + PKU + NVIDIA), "MiniMax Sparse Attention", arXiv:2606.13392 — https://arxiv.org/html/2606.13392v2
- DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv:2606.19348 — https://arxiv.org/html/2606.19348v1
- vLLM blog, "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action", Sep 2025 — https://vllm.ai/blog/2025-09-29-deepseek-v3-2

**Index sharing and indexer cost**
- Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li (THUDM / Z.ai), "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse", arXiv:2603.12201 — https://arxiv.org/html/2603.12201v1
- Sun, Zhang, Dong, Wang, Wei, "You Only Index Once: Cross-Layer Sparse Attention with Shared Routing", arXiv:2606.06467 — https://arxiv.org/html/2606.06467v1
- Zhou, Meng, Xu, Liu, Lu, Zhang, Pei (PKU MuLab + HITSZ), "MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference", arXiv:2605.07363 — https://arxiv.org/html/2605.07363v1
- Xu, Meng et al. (PKU MuLab + Tencent), "HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention", arXiv:2603.28458 — https://arxiv.org/html/2603.28458v3
- Cheng, Zhao, Liu, Li, Qiao, Duan, Chen, Chen, Rouhani, Yang (NVIDIA), "Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation", arXiv:2604.22312 — https://arxiv.org/html/2604.22312v1
- NVIDIA, TensorRT-LLM PR #12385, "Temporally-Correlated Heuristic-guided Indexer TopK for Sparse Attention" (merged 2026-04-03) — https://github.com/NVIDIA/TensorRT-LLM/pull/12385
- Yao et al., "Recall Before You Rank: Similarity-Guided Top-K Reuse for Efficient Long-Context Attention", arXiv:2607.27692 — https://arxiv.org/html/2607.27692v1
- Raschka, "GLM-5.2 and IndexShare for Long-Context Sparse Attention", 2026 — https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html

**Training-free KV selection / eviction**
- Zhang, Sheng et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023, arXiv:2306.14048 — https://arxiv.org/abs/2306.14048
- Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks", ICLR 2024, arXiv:2309.17453 — https://arxiv.org/abs/2309.17453
- Li, Huang et al., "SnapKV: LLM Knows What You are Looking for Before Generation", arXiv:2404.14469 — https://arxiv.org/abs/2404.14469
- Cai, Zhang et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling", arXiv:2406.02069 — https://arxiv.org/abs/2406.02069
- Tang, Zhao, Zhu, Xiao, Kasikci, Han, "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024, arXiv:2406.10774 — https://arxiv.org/abs/2406.10774
- Xiao, Tang, Zuo, Guo et al. (MIT Han Lab), "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads", arXiv:2410.10819 — https://arxiv.org/abs/2410.10819
- Yang, Guo, Tang, Hu et al. (MIT Han Lab), "LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention", MLSys 2025, arXiv:2502.14866 — https://arxiv.org/abs/2502.14866
- Gao, Guo, Cao, Xia et al. (Microsoft Research Asia), "SeerAttention-R: Sparse Attention Adaptation for Long Reasoning", arXiv:2506.08889 — https://arxiv.org/abs/2506.08889
- Jiang, Li, Zhang, Wu et al. (Microsoft), "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention", arXiv:2407.02490 — https://arxiv.org/abs/2407.02490
- Xu, Xiao, Huang, Guo et al. (MIT Han Lab), "XAttention: Block Sparse Attention with Antidiagonal Scoring", arXiv:2503.16428 — https://arxiv.org/abs/2503.16428
- Deng, Ling, Fan, Li (Microsoft), "UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training", arXiv:2605.27740 — https://arxiv.org/html/2605.27740v1
- Liu, Palnitkar, Rabbani et al., "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning", arXiv:2512.12008 — https://arxiv.org/html/2512.12008v1

**Linear / hybrid attention in production**
- Kimi Team (Moonshot), "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv:2510.26692 — https://arxiv.org/html/2510.26692v2
- vLLM blog, "vLLM Now Supports Qwen3-Next: Hybrid Architecture with Extreme Efficiency", Sep 2025 — https://vllm.ai/blog/2025-09-11-qwen3-next
- OpenAI, "gpt-oss-120b & gpt-oss-20b Model Card", arXiv:2508.10925 — https://arxiv.org/abs/2508.10925

**KV cache management systems**
- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023, arXiv:2309.06180 — https://arxiv.org/abs/2309.06180
- Prabhu, Nayak, Mohan, Ramjee, Panwar (MSR India), "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention", ASPLOS 2025, arXiv:2405.04437 — https://arxiv.org/abs/2405.04437
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv:2312.07104 — https://arxiv.org/abs/2312.07104
- LMSYS/SGLang, "Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism", May 2025 — https://lmsys.org/blog/2025-05-05-large-scale-ep/
- SGLang server arguments (source of truth for all flag names quoted above) — https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py
- Qin, Li, He, Zhang, Wu, Zheng, Xu (Moonshot AI), "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", arXiv:2407.00079 — https://arxiv.org/abs/2407.00079
- Yao, Li, Liu, Ray, Cheng, Zhang, Du, Lu, Jiang, "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion", arXiv:2405.16444 — https://arxiv.org/abs/2405.16444
- LMCache documentation — https://docs.lmcache.ai/
- Ma, Eitzinger, Köstler, "Irminsul: MLA-Native Position-Independent Caching for Agentic LLM Serving", arXiv:2605.05696 — https://arxiv.org/abs/2605.05696
- Meng, Lee, Wang (UPenn + Intel), "Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading", arXiv:2601.19910 — https://arxiv.org/abs/2601.19910
- NVIDIA, TensorRT-LLM DeepSeek-V3 example README — https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/models/core/deepseek_v3/README.md
- He, Xu, Wu, Hu et al., "BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving", *Software: Practice and Experience* 2026, arXiv:2510.13223 — https://arxiv.org/abs/2510.13223
- Wang & Buyya, "Preserving Admission Responsibility in Multi-Tenant Large Language Model Prefix Caches" (PrefixShield), arXiv:2608.01657 — https://arxiv.org/abs/2608.01657
- Wang & Buyya, "PrefixPlace: Provable Prefix Key-Value Placement for LLM Serving under Heterogeneous Compute and Transfer Costs", arXiv:2608.01655 — https://arxiv.org/abs/2608.01655
- Kang, Shin, Jeong, Park et al. (Korea Univ.), "Unified KV Pooling to Accelerate Long-Context LLM Serving", arXiv:2606.14779 — https://arxiv.org/abs/2606.14779

*Removed in this revision:* Hassani et al., "Generalized Neighborhood Attention"
(arXiv:2504.16922). The paper is real and its numbers check out (1.3 PFLOP/s FP16 on Blackwell,
28-46% end-to-end on B200 for Cosmos-7B / HunyuanVideo / FLUX), but it was listed in Sources
without being cited anywhere in the body, and its domain is vision/video diffusion, not LLM
decode. Not a fabrication — just dead weight.

---

## Open questions this survey did not settle

1. Our build reports **79 layers / 22 indexer layers**; the GLM-5 technical report says 80
   layers and Raschka's GLM-5.2 write-up says 78 with a clean `full, shared, shared, shared`
   4-layer group. 22 is not 79/4 = 19.75, so some layers run their own indexer outside the
   pattern. SGLang reads both `index_topk_freq` and a per-layer `index_topk_pattern` string
   from the HF config [verified], so the answer is in our config, not in the literature. Worth
   five minutes with the config loader — and if the extra indexers sit at the boundary layers,
   our pattern is already doing what IndexCache's overlap data recommends.
2. Nobody has published the interaction between **cross-layer index sharing and speculative
   decoding**. GVR handles the time axis, IndexCache the layer axis; the product (share the
   index across both the 4-layer group *and* the EAGLE draft positions) is a ~11x amortisation
   that nobody has measured for quality. This is the biggest gap in the literature relative to
   our stack.
3. **MISA + IndexCache composition** is claimed to be orthogonal by MISA's authors but was not
   measured. On GLM-5 shapes the theoretical product is `4x (layer) · 4x (head) = 16x`; the
   real number is unknown.
4. **Why is TBO incompatible with index sharing in SGLang?** The assertion is unambiguous
   [verified], but no design doc explains whether the conflict is fundamental (the shared index
   creates a cross-microbatch dependency) or merely unimplemented. Given TBO is worth 27-35% on
   prefill, this is worth asking upstream.
5. FlashMLA's **sparse decode reaching only 350 TFLOPS on B200 vs 410 on H800** is stated in
   the repo README with the parenthetical "which is not really optimized yet". So the *cause*
   is known; what is unknown is whether `flashinfer_sparse_mla` or `trtllm` closes it on our
   shapes. That needs a local A/B, not a literature search.
6. No published measurement exists of **DSA quality at 1M context with index sharing**. GLM
   trains to 200K and extends to 1M; IndexCache's longest measurement is 200k; Raschka reports
   a 2.9x FLOP reduction for IndexShare at 1M but no quality number at that length.
7. **Does MISA's head routing survive our trained-in sharing?** MISA was validated on GLM-5,
   whose indexers are per-layer. Routing 8 of 32 heads in an indexer that must already serve
   four layers is a compounding approximation nobody has tested.
