# Attention, KV cache and long context: algorithms that change the inference roofline

## What this is

A survey of the *algorithmic* literature that changes how many bytes and how many FLOPs
attention costs at inference time, written against a specific target: GLM-5.2-class MoE
models (MLA + DeepSeek Sparse Attention, 256/8 experts, EAGLE 3-1-4 speculation) served
on 8x B200 SXM under two objectives — minimum single-stream latency and minimum cost per
user at concurrency.

Every paper below was fetched and read (abstract + method + evaluation numbers) during
this pass. Results are labelled:

- **[verified]** — I read the number in the paper/repo text.
- **[reported]** — the authors claim it in an abstract or README I read, but I did not see
  the supporting table.
- **[inferred]** — my own arithmetic or reasoning, shown so you can check it.

Hardware is stated for every result. A 9x speedup on A100 with a 27B model is not a 9x
speedup on B200 with a 744B MoE, and where that matters I say so.

Two config facts that anchor the whole document, both read from source:

- **DeepSeek-V3.2 reference config** (`inference/config_671B_v3.2.json`, fetched from the
  DeepSeek-V3.2-Exp repo, verbatim): `n_layers: 61`, `n_heads: 128`, `kv_lora_rank: 512`,
  `qk_nope_head_dim: 128`, `qk_rope_head_dim: 64`, `v_head_dim: 128`, `dtype: fp8`,
  `index_n_heads: 64`, `index_head_dim: 128`, `index_topk: 2048`. **[verified]**
- **GLM-5** (arXiv:2602.15763): 744B total / 40B active, 256 experts, 80 layers, MLA with
  latent KV dim 576 and head dim 256, DSA, 3 MTP layers with 4 speculative steps and mean
  accepted length 2.76 vs DeepSeek-V3.2's 2.55. **[verified from the paper HTML]**
  Raschka's GLM-5.2 write-up reports 78 layers and an indexer pattern of
  `full, shared, shared, shared` — 1 indexer per 4 layers. **[reported]** Our build reports
  79 layers / 22 indexer layers / `index_topk_freq=4`; I use *our* numbers for arithmetic
  and flag the discrepancy rather than pretending the public numbers are ours.

---

## Bottom line for our system

Ranked by expected value on our hardware and our two objectives.

1. **Move attention from TP8 to DP8 (data-parallel attention) if we haven't.** MLA keeps a
   single latent KV per token — there is no KV-head axis to shard. Under TP8 every rank
   holds the *entire* KV cache, so the KV pool is ~112 GB, not 896 GB. My arithmetic below
   puts that at **~15 concurrent 128k streams under TP8 vs ~125 under DP8** [inferred].
   SGLang moved to DP attention for exactly this reason and reports ~5x over vanilla TP16
   on 96xH100 for DeepSeek-V3 [reported]. This is the single biggest cost-per-user lever
   at long context and it costs nothing in quality.
2. **Verify that index sharing is actually saving what it should, and consider pushing it
   further.** At 128k with `index_topk_freq=4`, the lightning indexer reads **3.6x more
   bytes per decode step than the sparse attention it feeds** and costs roughly as many
   FLOPs as the entire sparse attention [inferred, arithmetic shown in Part 8]. IndexCache
   (arXiv:2603.12201, THUDM) measured 70-100% top-k overlap between adjacent layers on a
   30B MLA+DSA model and got 1.82x prefill / 1.48x decode at 200k on H100 by removing 75%
   of indexers — with a *greedy, loss-calibrated* layer choice that beat uniform
   interleaving at the same ratio [verified]. Our uniform 1-in-4 is leaving quality (or
   sharing ratio) on the table; a calibration sweep is cheap.
3. **Adopt NVIDIA's Guess-Verify-Refine top-K kernel.** arXiv:2604.22312 (NVIDIA) is
   *literally our workload*: DeepSeek-V3.2 NVFP4 on B200, k=2048, TEP8 min-latency. 1.88x
   average speedup over the production radix-select, up to 2.42x per layer per step, and
   **4.36-7.52% end-to-end TPOT reduction at 64k-100k, including with MTP=1**, bit-exact
   vs `torch.topk` [verified]. It is merged in TensorRT-LLM (PR #12385) behind
   `enable_heuristic_topk` (default false). If we are not on TensorRT-LLM, the algorithm is
   ~200 lines and portable.
4. **Exploit speculative decoding to make MLA attention compute-bound, and pick kernels
   accordingly.** Absorbed MLA decode at q_len=1 has arithmetic intensity ~242 FLOP/byte
   (BF16 latent) [inferred; the SNU/UIUC roofline paper independently puts FlashMLA at
   ~256 Op/B on B200, verified]. B200's BF16 ridge point is 281 Op/B [verified]. With
   EAGLE 3-1-4 the verify batch multiplies intensity by the draft length, pushing attention
   *past* the ridge into compute-bound territory. Zadouri/Strauss/Dao (arXiv:2505.21487)
   report their GLA kernel is **up to 2x faster than FlashMLA specifically when query
   length exceeds one** [reported] — i.e. exactly the speculative-decode regime. Our
   attention kernel choice should be benchmarked at q_len=4-5, not q_len=1.
5. **Check FlashMLA's Blackwell decode path — it may be the wrong kernel on B200.** The
   FlashMLA README reports sparse MLA *decoding* at 410 TFLOPS on H800 SXM5 but only
   **350 TFLOPS on B200**, while sparse *prefill* hits 1450 TFLOPS on B200 [reported]. A
   Hopper-tuned decode kernel running slower on Blackwell than on Hopper is a red flag
   worth an afternoon of profiling. FlashAttention-4 (arXiv:2603.05451, Dao et al.) shows
   what Blackwell actually needs — software-emulated `exp` on FMA units because MUFU is
   16 ops/clk vs 8192 ops/clk for tensor cores, and conditional softmax rescaling — and
   explicitly does *not* cover head_dim 576/512 or MLA [verified].
6. **Prefix caching is worth more than 1.54x if we make it MLA-native and
   position-independent.** Irminsul (arXiv:2605.05696) exploits the fact that MLA's row
   factors into a position-free `c_KV` and a 64-dim `k_R` that is correctable by a
   closed-form rotation, giving content-addressed (not prefix-addressed) caching; they
   recover **~83% of prompt tokens above exact-prefix hit rate on agentic traffic** and cut
   prefill energy 63% per hit [reported]. Our 1.54x is a strict-prefix number; agentic and
   multi-turn traffic is where the remaining headroom is.
7. **Do not spend engineering on training-free KV eviction (H2O / SnapKV / PyramidKV /
   StreamingLLM) for a reasoning model.** The 2025 evaluation "Hold Onto That Thought"
   (arXiv:2512.12008) shows PyramidKV at **0.00 accuracy on MATH-500 with
   DeepSeek-R1-Distill-Llama-8B at budget 256**, StreamingLLM at **0.00 on ReClor at budget
   128** vs 0.45-0.51 full cache, and eviction producing "repetitive, dead-end
   chain-of-thought" [verified]. These methods evict based on prompt-time attention; a
   reasoning model's important tokens are generated later. DSA already gives us the win
   without the failure mode.
8. **Cross-layer index sharing has a second axis we are not using: time.** GVR measured
   **35-50% top-k overlap between consecutive decode steps in layers 20-60** (and only 1-2%
   in layers 0-1) on real DeepSeek-V3.2 traffic [verified]. ReTopK (arXiv:2607.27692) turns
   that into 3.07x selector speedup at 128k/k=512 with +0.50% perplexity on L20 [reported].
   The natural fusion for us: reuse the index across the *speculative draft positions*
   within one verify step, which is free correctness-wise if we re-index on the accepted
   token.
9. **If KV memory ever becomes the binding constraint before bandwidth, go FP8 latent
   before you go sparse-er.** vLLM's DeepSeek-V3.2 layout is 656 B/token/layer (512 FP8
   NoPE + 16 B of FP32 scales + 128 B BF16 RoPE) vs 1152 B BF16 [verified] — a 1.76x cut
   with no algorithmic risk. DeepSeek-V4 (arXiv:2606.19348) ships exactly this: BF16 for
   RoPE dims, FP8 for the rest [verified].
10. **Watch DeepSeek-V4's Compressed Sparse Attention as the next architectural step.** V4
    compresses every m tokens into one KV entry and *then* runs DSA top-k over the
    compressed entries, plus a heavily-compressed dense branch and a sliding window branch.
    Reported result: **10% of V3.2's KV cache at 1M context for V4-Pro, 7% for V4-Flash,
    and 27%/10% of single-token FLOPs** [verified]. If we are going to 1M context, this is
    the shape the field is converging on, not more aggressive top-k on uncompressed KV.

---

## Part 0 — the roofline everything below is trying to move

Fix the machine: B200 SXM, 183 GB HBM3e, ~8 TB/s. The SNU/UIUC systems paper
(arXiv:2507.15465) models B200 SXM6 at 2250 TFLOPS BF16 and 8000 GB/s, giving a **ridge
point of ~281 Op/B** [verified]. FP8 doubles peak, so the FP8 ridge is ~562 Op/B; NVFP4
doubles again to ~1125 Op/B.

Decode-time arithmetic intensity of the core attention, per token, absorbed form, ignoring
the small RoPE terms [inferred, but cross-checked against the paper below]:

| Attention form | KV bytes/token/layer (BF16) | FLOPs/token/layer/kv-token | Intensity |
|---|---|---|---|
| MHA, 128 heads x 128 | 65,536 | 65,536 | **1 Op/B** |
| GQA-8, 128 q heads x 128 | 4,096 | 65,536 | **16 Op/B** |
| MLA absorbed (d_c=512, d_r=64, 128 heads) | 1,152 | 278,528 | **~242 Op/B** |
| MLA absorbed, FP8 latent (656 B layout) | 656 | 278,528 | **~425 Op/B** |
| MLA absorbed, q_len=4 (EAGLE verify) | 1,152 | 1,114,112 | **~968 Op/B** |

The SNU/UIUC paper's Table III independently reports MHA/GQA core attention at ~1 Op/B and
FlashMLA at ~256 Op/B on B200, concluding MLA is "balanced between compute-bound and
memory-bound" while MHA/GQA is "strictly memory-bound" [verified]. My 242 and their 256
agree to within kernel bookkeeping.

**This is the single most important fact in this document.** MLA does not merely shrink the
KV cache; it moves decode attention from 1 Op/B to ~250 Op/B, i.e. from 280x below the
ridge to *at* the ridge. Every technique below either (a) reduces the bytes further, which
now buys less than it used to, or (b) reduces the FLOPs, which now buys more than it used
to. Sparse attention (DSA) reduces both, but introduces an indexer whose intensity is
*fixed at 2 x index_n_heads = 128 Op/B* (FP8 keys) — **below** the BF16 ridge, and
independent of context length and batch size [inferred; derivation in Part 8]. So the
indexer is irreducibly memory-bound while the attention it feeds is not. That asymmetry is
why cross-layer index sharing exists.

---

## Part 1 — Exact attention kernels: the online-softmax lineage

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Online normalizer calculation for softmax (arXiv:1805.02867) | Milakov, Gimelshein — NVIDIA | arXiv 2018 | GPU (unspecified) | Softmax 1.3x; fused softmax+topk 5x [verified] | Yes — the primitive under every FlashAttention |
| FlashAttention (arXiv:2205.14135) | Dao, Fu, Ermon, Rudra, Ré — Stanford | NeurIPS 2022 (venue not shown on abs page) | A100-class | 3x on GPT-2 seq 1k; 15% BERT-large 512 [verified] | Universal |
| FlashAttention-2 (arXiv:2307.08691) | Dao | arXiv 2023 | A100 | 225 TFLOPs/s, 72% MFU, 50-73% of peak [verified] | Universal |
| FlashAttention-3 (arXiv:2407.08608) | Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao — Colfax/NVIDIA/Meta/Together/Princeton | arXiv 2024 | H100 | FP16 740 TFLOPs/s (75%), FP8 ~1.2 PFLOPs/s, FP8 error 2.6x lower than baseline FP8 [verified] | Yes (Hopper default) |
| FlashAttention-4 (arXiv:2603.05451) | Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao | arXiv Mar 2026 | **B200 / GB200** | 1.1-1.3x vs cuDNN 9.13, 2.1-2.7x vs Triton, peak 1613 TFLOPs/s BF16 (71%) [verified] | Open source, CuTe-DSL |
| Flash-Decoding (PyTorch/Together blog, Dao/Haziza/Massa/Sizov) | Meta + Together | blog, Oct 2023 | A100 | attention up to 50x vs FA2 at 128k B=1; up to 8x end-to-end CodeLlama-34B [verified] | Universal (FA >= 2.2, xFormers >= 0.0.22) |
| FlashMLA (github.com/deepseek-ai/FlashMLA) | DeepSeek | repo | H800 SXM5, B200 | dense MLA decode 660 TFLOPS / 3000 GB/s (H800); **sparse MLA decode 410 TFLOPS H800 vs 350 TFLOPS B200**; sparse prefill 1450 TFLOPS B200 [reported] | Yes — default sparse-MLA backend in TensorRT-LLM on Hopper |

### The mechanism, briefly, and why FA4 matters on our box

Online softmax (Milakov & Gimelshein) computes `m_j = max(m_{j-1}, x_j)` and
`d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)` in a single pass, so the normalizer
never needs a second read of the logits. FlashAttention lifts that from a vector to a tile:
tile Q, K, V into SRAM, accumulate `O` and the running `(m, l)` statistics, and never
materialise the `N x N` score matrix in HBM. FlashAttention-2 fixes the *work partitioning*
— parallelise over query blocks (not just batch x head), and cut non-matmul FLOPs, which
matter because non-matmul throughput on a tensor-core GPU is ~1/16 of matmul throughput.
FlashAttention-3 adds Hopper's asynchrony: warp-specialised producer/consumer with TMA,
ping-pong scheduling that overlaps the softmax of tile j with the GEMM of tile j+1, and FP8
with block quantisation + incoherent processing.

**FlashAttention-4 is the one that matters for B200.** Its framing is "asymmetric hardware
scaling": tensor cores doubled, but shared memory bandwidth and the exponential unit did
not. Concretely from the paper [all verified]:

- **Software-emulated exp.** MUFU (the SFU path for `ex2`) runs at 16 ops/clock against
  8192 ops/clock for tensor cores. FA4 computes `2^x = 2^floor(x) * 2^(x - floor(x))`,
  getting the integer part by IEEE-754 bit manipulation and the fractional part from a
  **degree-3 polynomial with max relative error 8.8e-5**, which after BF16 rounding matches
  hardware to within 1 ULP on 99% of inputs. Only 10-25% of softmax entries need the
  emulated path ("partial emulation") — the rest still go to MUFU, which is now no longer
  the critical path.
- **Conditional softmax rescaling.** Standard online softmax rescales the accumulator every
  time the running max moves. FA4 skips the update when `m_j - m_{j-1} <= tau`, with
  `tau = log2(256) = 8.0`, and corrects at final normalisation. The paper reports this cuts
  rescaling operations by roughly 10x.
- **2-CTA MMA + tensor memory.** 256 KB of on-chip tensor memory per SM; MMA tiles of
  128x128 or 128x256 in 2-CTA mode; the backward dQ step uses distributed shared memory to
  halve global atomic reductions. Deterministic backward reaches 75% of nondeterministic
  speed.
- Written entirely in **CuTe-DSL** (Python-embedded), 20-30x faster compile than C++
  templates.

Head dims tested include `(192, 128)` "for DeepSeek V3 compatibility". **MLA's 576/512 is
explicitly not tested, and sparse attention is future work** [verified]. So FA4 is the
right reference for our *dense* prefill attention and for any GQA-shaped model we add
(Qwen3.8), but it does not replace FlashMLA for the MLA decode path — and FlashMLA's own
B200 sparse-decode number is the weak link.

### Flash-decoding / split-KV

The decode-time problem FlashAttention-2 does not solve: at q_len=1 and batch 1, there is
one query block, so parallelism = batch x heads, which on a 108-SM A100 uses <1% of the
GPU. Flash-decoding adds a third parallel axis — split the KV sequence into chunks, run
FlashAttention per chunk writing out a per-row log-sum-exp scalar, then a second kernel
reduces across splits using those LSEs. Measured on A100, CodeLlama-34B shapes (16 q heads,
2 kv heads, d=128), B=1 seqlen 131072: **PyTorch eager 2664 us, FlashAttention 2.0.9
4592 us, Flash-Decoding 106.6 us** [verified]. Attention runtime is near-constant from 512
up to 32k. This is table stakes and is in every engine, but the split count is a tuning
knob that matters a lot at our concurrency: too many splits and the reduction kernel and
the LSE traffic dominate; too few and you idle SMs. At C64 with 8 GPUs you have 64 x n_heads
work items already, so the optimal split count at C64 is much smaller than at C1 — worth a
sweep, since we care about both regimes.

---

## Part 2 — KV cache shape: MQA, GQA, and Multi-head Latent Attention

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Fast Transformer Decoding: One Write-Head is All You Need (arXiv:1911.02150) | Shazeer — Google | arXiv 2019 | TPU | MQA: single KV head, "much faster to decode", "minor quality degradation" [verified abstract] | Yes (PaLM, Falcon) |
| GQA (arXiv:2305.13245) | Ainslie et al. — Google | EMNLP 2023 | TPU | Uptrain MHA->MQA/GQA with **5% of pretraining compute**; GQA ~ MHA quality at ~MQA speed [verified] | Universal (Llama 2/3, Mistral, Qwen) |
| DeepSeek-V2 (arXiv:2405.04434) | DeepSeek-AI | arXiv 2024 | H800 | MLA: KV cache **-93.3%**, max generation throughput **5.76x** vs DeepSeek 67B, training cost -42.5% [verified] | Yes |
| DeepSeek-V3 (arXiv:2412.19437) | DeepSeek-AI | arXiv 2024 | H800 | 671B/37B active, MLA + DeepSeekMoE + MTP, 14.8T tokens, 2.788M H800-hours [verified] | Yes |
| Hardware-Centric Analysis of MLA (arXiv:2506.02523) | Geens, Verhelst — KU Leuven | Electronics Letters 2025 | modelled accelerator, 400 GB/s | Decode OI: MHA ~0.5-1.5 Op/B, MLA-recompute ~3-5 Op/B and cache-size-insensitive [verified] | analysis |
| The New LLM Bottleneck (arXiv:2507.15465) | Yun et al. — SNU + UIUC | arXiv 2025 | **B200 SXM6 modelled** | MHA/GQA ~1 Op/B; MLA reordered ~512 Op/B; FlashMLA ~256 Op/B; B200 ridge ~281 Op/B [verified] | analysis |
| Hardware-Efficient Attention for Fast Decoding (arXiv:2505.21487) | Zadouri, Strauss, Dao | arXiv 2025 | not stated in abs | GTA = GQA quality at ~half KV; GLA = MLA quality, easier to shard; **GLA kernel up to 2x faster than FlashMLA when q_len > 1**; up to 2x online-serving throughput [reported] | Research |
| GQLA (arXiv:2605.15250) | Meng — PKU MuLab | arXiv 2026 | H100, H20 | Two algebraically-equivalent decode paths (MQA-absorb and per-group GQA) from one weight set; TransGQLA converts a GQA checkpoint, 28.125% of GQA KV on the absorb path [verified] | Research |
| QK-Normed MLA (arXiv:2606.16310) | Han, Zhao, Zhou, Li, Sun | arXiv 2026 | H800 | QK-RMSNorm made compatible with latent caching; **<2% decode latency overhead up to 256k** [verified abstract] | Research |
| Why Megatron-Core Forbids Absorption / LAGA (arXiv:2607.17644) | Ma | arXiv 2026 | DeepSeek-V3 scale | Absorbed MLA in *training* inflates activation memory 20-34%, up to 9.2 GB (n_h=128, seq 16384, SP=8) [verified] | analysis |

### MLA, in enough detail to implement

**The compression.** For token `t` with hidden state `h_t`, MLA does not cache K and V. It
caches one low-rank latent:

```
c^KV_t = W^DKV h_t                (d_c = 512)
k^C_t  = W^UK c^KV_t              (reconstruct 128 heads x 128 dims)
v^C_t  = W^UV c^KV_t
```

Queries get their own (non-cached) low-rank path, `c^Q_t = W^DQ h_t` with
`d_c' = 1536` in V2/V3, then `q^C_t = W^UQ c^Q_t` [verified from DeepSeek-V2's config and
the V3.2 JSON: `q_lora_rank: 1536`, `kv_lora_rank: 512`].

**Decoupled RoPE — the part people get wrong.** RoPE is position-dependent and does not
commute with the up-projection `W^UK`: if you rotate the reconstructed key, you cannot fold
`W^UK` into the query anymore, because the rotation sits between them and depends on the
*relative* position of query and key. MLA's fix is to split the key into two pieces:

- a **NoPE** piece of `qk_nope_head_dim = 128` per head, reconstructed from the latent and
  therefore absorbable;
- a **decoupled RoPE** piece `k^R_t` of `qk_rope_head_dim = 64`, produced by a separate
  projection *from `h_t` directly*, carrying RoPE, **shared across all heads** (MQA-style),
  and cached alongside the latent.

So the cache per token per layer is `d_c + d_h^R = 512 + 64 = 576` values, versus
`2 * n_h * d_h = 32768` for MHA. DeepSeek-V2's own table states MLA's cache is equivalent
to "GQA with only 2.25 groups" while beating MHA on quality [verified].

**Weight absorption.** At decode you never want to materialise 128 heads of K and V from
the latent — that is 32768 values of write traffic per token per layer to save 576 values
of read traffic. Instead, because

```
q^T k = (W^UQ c^Q)^T (W^UK c^KV) = c^Q^T (W^UQ^T W^UK) c^KV
```

you precompute or recompute `W^UQ^T W^UK` and apply it to the query, then dot the resulting
"absorbed query" (dimension `d_c + d_h^R = 576` per head) directly against the cached
latent. Symmetrically `W^UV` folds into `W^O` on the output side. The attention then looks
like **MQA with head_dim_k = 576 and head_dim_v = 512** — which is exactly the shape
FlashMLA advertises [verified from the FlashMLA README].

**Numerical and systems caveats of absorption — four of them, all real:**

1. **Absorb vs recompute is a real trade, not a strict win.** Geens & Verhelst name the two
   schemes `MLA_rc` (recompute the composite `W^UQ W^UK,T` on the fly) and `MLA_ru`
   (precompute and reuse it). `MLA_rc` gives ~3-5 Op/B at decode and is insensitive to KV
   size; `MLA_ru` scales its OI with KV size and is poor for small caches. Their conclusion:
   **`MLA_rc` (recompute) wins on essentially all commercial accelerators and at batch=1**;
   `MLA_ru` only wins when compute is scarce relative to bandwidth, "an uncommon case"
   [verified]. Note their model is a 400 GB/s accelerator, not B200, so treat the crossover
   as directional.
2. **Absorption is a decode-only optimisation and hurts prefill.** The same body of work
   notes reordering "increases the latency during the prefill stage" because it changes
   FLOPs, memory accesses and arithmetic intensity in the wrong direction when q_len is
   large [reported]. Two code paths are required. TensorRT-LLM and SGLang both do this.
3. **Absorption is a memory trap in training.** arXiv:2607.17644 shows Megatron-Core
   hard-asserts absorption off during training
   (`assert not (self.training and self.cache_mla_latents)`) and quantifies why: the
   absorbed intermediates live in `n_h x d_kv` per token, *larger* than the per-head K/V
   they replace, inflating activation memory **20-34%, up to 9.2 GB at DeepSeek-V3 scale
   (n_h=128, seq=16384, SP=8), widening to 19.2 GB with a fused kernel** [verified]. Their
   LAGA fix all-gathers the latent (1.98x comm reduction) but reconstructs per-head K/V
   locally rather than absorbing. Relevant to us if we ever fine-tune or do RL on this
   model, not to pure serving.
4. **Precision.** The absorbed query has dimension 576 per head and the composite matrix
   `W^UQ^T W^UK` is a product of two learned matrices — its dynamic range is the product of
   theirs. Two concrete published consequences: (a) QK-RMSNorm, the standard stabiliser,
   *appears* incompatible with latent caching because post-projection normalisation needs
   the full key; arXiv:2606.16310 shows the static weight can be absorbed into the query
   side and the dynamic statistic reduces to one inverse-RMS scalar per token per KV group,
   restoring exact equivalence at <2% decode overhead on H800 [verified abstract]; (b)
   FlashMLA's sparse decode kernel keeps the KV in FP8 *with per-block scale factors* and
   vLLM's layout keeps the 64 RoPE dims in **BF16, unquantised** — 128 bytes of the 656
   [verified]. If you are tempted to quantise the RoPE half, note that both DeepSeek's own
   V4 design and vLLM's layout keep it in higher precision. That is a strong signal.

**Tensor-parallel consequence — the one that costs us money.** With MHA or GQA, TP splits
the KV heads across ranks and the KV cache shrinks per rank. With MLA there is **one**
latent per token; there is no head axis to split. Under TP8 the latent cache is replicated
8x. SGLang's large-scale EP writeup states DP attention "eliminates KV cache duplication
across devices, significantly reducing memory overhead" and reports 52.3k input tok/s and
22.3k output tok/s per node on 96xH100 for DeepSeek-V3, ~5x over vanilla TP16 [reported].
Their stack pairs **DP attention + EP MoE + PD disaggregation + two-batch overlap** (TBO,
27-35% prefill throughput, 35% under simulated MTP) [reported]. Our measured 19.6%
collectives with 47% rank-arrival skew is partly a symptom of this: TP8 attention forces an
all-reduce per layer on the output projection, and every rank must arrive; DP attention
replaces that with an all-gather/reduce-scatter over the token axis, which has different
skew behaviour and can be overlapped with the MoE dispatch.

### The alternatives to MLA worth knowing

**GTA / GLA** (Zadouri, Strauss, Dao, arXiv:2505.21487). GTA ties and reuses K and V states
to halve GQA's cache at matched quality. GLA is "grouped latent attention" — MLA-quality
but with a group axis, so it **shards under TP without replication**. The abstract's claim
that "GLA reduces end-to-end latency and increases throughput in online serving benchmarks
by up to 2x" and that "our optimized GLA kernel is up to 2x faster than FlashMLA... in a
speculative decoding setting when the query length exceeds one" [reported] is directly
relevant to us: it is a statement that MLA's kernel leaves performance on the table exactly
in the regime EAGLE puts us in. We cannot change GLM-5.2's architecture, but we can take
the kernel lesson.

**GQLA** (arXiv:2605.15250, PKU) is the pragmatic version: expose *both* an MQA-absorb path
and a per-group-expanded GQA path over the same weights, and pick at runtime by hardware
and by `s_q` (query length). They explicitly target `s_q=1` on H100 and `s_q=2` on H20
[verified]. This is the right mental model for our serving stack: **one weight set, two
attention code paths, dispatched on q_len and on whether we are latency- or
throughput-bound.**

---

## Part 3 — Sparse attention that requires training

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| NSA (arXiv:2502.11089) | Yuan et al. — DeepSeek + PKU + UW | ACL 2025 (long) | **A100** | 64k: fwd 9.0x, bwd 6.0x, **decode 11.6x**; 27B/3B MoE beats full attention on LongBench 0.469 vs 0.437 [verified] | Research; ideas shipped in DSA |
| MoBA (arXiv:2502.13189) | Lu et al. — Moonshot AI + Tsinghua + Zhejiang | arXiv 2025 | not stated (Llama-8B shapes) | **6.5x at 1M**, 16x attention-compute reduction at 10M; val-loss gap <1e-3 at 8k [verified] | **Yes** — deployed for Kimi long-context |
| DeepSeek-V3.2 / DSA (arXiv:2512.02556) | DeepSeek-AI | arXiv Dec 2025 | H800 | Lightning indexer + top-k=2048, FP8 indexer, MQA-mode MLA; O(L^2) -> O(Lk) core attention [verified] | **Yes** — vLLM, SGLang, TensorRT-LLM |
| GLM-5 (arXiv:2602.15763) | GLM-5 Team (Zhipu/Z.ai + Tsinghua) | arXiv Feb 2026 | not stated | DSA on 744B/40B, 80 layers; "90% of attention entries in long contexts are indeed redundant"; ~1.5-2x compute reduction long-seq [verified] | **Yes — this is our model's lineage** |
| MiniMax Sparse Attention (arXiv:2606.13392) | Lai et al. — MiniMax + PKU + NVIDIA | arXiv Jun 2026 | **H800** | GQA-native block sparse, k=16 blocks of 128 per group; **28.4x per-token attention compute at 1M; 14.2x prefill / 7.6x decode wall-clock** [verified] | Production model family |
| DeepSeek-V4 (arXiv:2606.19348) | DeepSeek-AI | arXiv Jun 2026 | NVIDIA + Ascend | CSA (compress m tokens -> 1 entry, then DSA top-k) + HCA + sliding window; **KV = 10% (Pro) / 7% (Flash) of V3.2 at 1M**; FLOPs 27%/10%; FP4 experts, FP8 KV, FP4 indexer [verified] | Announced |

### NSA — the design constraints everyone else inherited

NSA runs three branches in parallel and gates them with a learned sigmoid MLP:
`o_t = sum_c g_t^c * Attn(q_t, K~_t^c, V~_t^c)` over `c in {compression, selection, sliding}`
[verified, Eq. 5].

- **Compression**: pool blocks of length `l = 32` with stride `d = 16` into single compressed
  KV entries. Cheap global view.
- **Selection**: the important trick — the *selection* importance scores are derived from
  the *compression* branch's attention scores, not from a separate scorer.
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
   solved it with a bespoke gather kernel (FlashMLA's `indices` tensor) — a real
   engineering divergence.
2. **GQA-consistent selection.** All heads in a KV group must select the same blocks, so
   the scores are summed across heads in the group before top-k:
   `p^slc'_t = sum_h p^slc,(h)_t` [verified]. Without this, a group's shared KV read is
   the union of per-head selections and the sparsity evaporates.

Their decode memory-access table is the cleanest statement of why sparse decode helps
[verified]: at 8k/16k/32k/64k, full attention loads 8192/16384/32768/65536 token-equivalents
vs NSA's 2048/2560/3584/5632 — expected speedups 4x/6.4x/9.1x/**11.6x**. Measured on
**A100** with a 27B/3B MoE (30 layers, 2560 hidden, GQA 4 groups / 64 heads), 270B tokens
at 8k then continued at 32k. Quality: general average 0.456 vs 0.443 full attention;
LongBench 0.469 vs 0.437; AIME-16k after R-distillation 0.146 vs 0.092 [verified].

**Transfer caveat:** A100, 27B, 64k. The speedup ratios should transfer (they are
memory-access ratios), the absolute kernel efficiency will not.

### DSA — what our model actually runs

From the DeepSeek-V3.2 paper [verified] plus the reference implementation [verified]:

```
I_{t,s} = sum_{j=1..H^I} w^I_{t,j} * ReLU( q^I_{t,j} . k^I_s )
```

- `H^I = index_n_heads = 64`, `index_head_dim = 128`, `index_topk = 2048`.
- **ReLU, not softmax** — "chosen for throughput efficiency". No normalisation pass, no
  exponential, so the indexer is a plain FP8 GEMM plus a ReLU plus a weighted reduce.
- `w^I` comes from `weights_proj = Linear(dim, n_heads, dtype=float32)`, i.e. a per-head
  scalar gate computed from the token, in FP32, then scaled by `n_heads^-0.5` and by the
  FP8 quantisation scales [verified from `model.py`].
- The indexer key `k^I_s` is **one vector per token per layer** (MQA-style), 128 dims, FP8.
  That is what makes its cost `O(L * d_I)` bytes rather than `O(L * H^I * d_I)`.
- The main attention runs under the **MQA mode of MLA** over the selected top-k entries.

**Training recipe** [verified]: two stages. (1) Dense warm-up: freeze everything except the
indexer, train it with a KL loss against the main model's aggregated attention distribution,
lr 1e-3, 1000 steps x 16 seq x 128k = **2.1B tokens**. (2) Sparse training: top-k=2048
active, all parameters trained, lr 7.3e-6, 15000 steps x 480 seq x 128k = **943.7B tokens**;
crucially the indexer is optimised *only* by the indexer loss and the main model *only* by
the LM loss (gradients are not mixed). GLM-5's version is a shorter 1000-step warm-up on
202,752-token sequences plus **20B tokens** of sparse continued pretraining [verified].

**The honest limitation the paper itself states**: "the lightning indexer retains O(L^2)
complexity" [verified]. DSA converts core attention from O(L^2) to O(Lk) and leaves a
cheaper-but-still-quadratic term behind. Part 8 quantifies exactly how much that costs us.

**Production status**: vLLM shipped DSA the day V3.2-Exp released, using DeepGEMM indexer
logit kernels, a fused top-K from DeepSeek's TileLang reference, and FlashMLA's sparse
attention kernel; KV layout **656 bytes/token/layer** (512 B FP8 NoPE + 16 B FP32 scales +
128 B BF16 RoPE) plus a separate indexer K cache stored per block of 64; **block size 64 is
the only supported block size** because both FlashMLA and the per-block indexer scales
require it; supported on 16xH100, 8xH200, 8xB200 [verified]. TensorRT-LLM: FlashMLA is the
default sparse-MLA backend on Hopper, `trtllm-gen` codegen for FP8 MLA on Blackwell, NVFP4
on B200/GB200, MTP via `--spec_decode_algo MTP --spec_decode_max_draft_len N`, with
`--relaxed_topk` / `--relaxed_delta` for reasoning-phase relaxed acceptance [verified].

### MoBA — the MoE framing, and the hybrid recipe

MoBA (Moonshot) partitions context into blocks and routes each query to top-k blocks by
`q . mean_pool(K_block)` — a **parameterless** gate. Causality is handled two ways: future
blocks are hard-masked, and the *current* block is handled by a separate causal-masked
attention because its mean-pooled key would otherwise leak future tokens [verified]. The
kernel splits into "historical blocks" (varlen FlashAttention over gathered query groups)
and "current block" (causal FlashAttention), then merges with online softmax [verified].
Block sizes 512-4096, top-k 3-12, 75-95% sparsity; validation-loss gap within 1e-3 at 8k;
6.5x at 1M and 16x attention-compute reduction at 10M vs full FlashAttention on Llama-8B
shapes [verified].

The transferable detail: **during SFT they keep the last 3 of 32 layers on full attention**
because MoBA's sparse gradient flow interacts badly with prompt-masked SFT [verified]. Any
hybrid dense/sparse layer schedule we consider should follow that shape — sparse in the
bulk, dense at the boundaries.

### MiniMax Sparse Attention — the GQA-native counterpoint

MSA is worth reading because it is the *other* production answer. Instead of DSA's
token-level, all-heads-shared selection, MSA does **block-level (size 128), per-GQA-group**
selection with k=16 blocks, from an "ultra-lightweight" Index Branch with just two
projection matrices, trained with a KL alignment loss against the main branch and a **40B
token indexer warm-up** with gradient detach so the KL never touches the backbone
[verified]. Per-token index FLOPs `H_kv * d_idx * N^2` vs GQA's `2 * H_q * d_h * N^2`.
Results on a 109B MoE trained on 3T tokens: 28.4x per-token attention compute at 1M, and
**14.2x prefill / 7.6x decode wall-clock on H800**, matching the GQA baseline on language,
math, code and multimodal [verified]. They ship both a from-scratch variant (MSA-PT) and a
checkpoint-conversion variant (MSA-CPT).

The design lesson for us: **block-level selection keeps KV reads contiguous**; DSA's
token-level selection needs a gather. On B200 with 8 TB/s and large L2, the gather penalty
is smaller than it was on A100, but it is not zero — and it is precisely the thing the
FlashMLA sparse decode kernel has to do well.

---

## Part 4 — Index sharing across layers (this is our `index_topk_freq=4`)

This is the sub-literature that most directly analyses our configuration. It is new (all
2026) and small, but it is real and one of the papers evaluates on GLM-5 itself.

| Paper | Lab | Venue / year | Hardware | Headline result | Training needed? |
|---|---|---|---|---|---|
| IndexCache (arXiv:2603.12201) | Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li — THUDM/Tsinghua | arXiv Mar 2026 | **H100** | Remove **75% of indexers**; 1.82x prefill / 1.48x decode at 200k on a 30B DSA model; GLM-5 preliminary >=1.3x both beyond 100k [verified] | Both variants offered |
| You Only Index Once / CLSA (arXiv:2606.06467) | Sun, Zhang, Dong, Wang, Wei | arXiv Jun 2026 | **B200** | One indexer feeds 16 cross-decoder layers; **7.6x decode / 17.1x throughput at 128k**, quality >= dense [verified] | 2-stage distill, no from-scratch pretrain |
| MISA (arXiv:2605.07363) | Zhou, Meng, Xu, Liu, Lu, Zhang, Pei — PKU MuLab + HITSZ | arXiv May 2026 | **H200** | Route to h=8 of H^I=64 indexer heads; **3.82x indexer-kernel speedup**; LongBench 50.85 vs 51.05 on DeepSeek-V3.2 [verified] | **Training-free** |
| HISA (arXiv:2603.28458) | Xu, Meng et al. — PKU MuLab + Tencent | arXiv Mar 2026 | A100 | Two-stage block-then-token indexing; O(L^2/B + LmB); **2.16x at 4:1, 3.75x at 8k budget** at 64k; LongBench 50.78 vs DSA 51.05 [verified] | **Training-free** |
| GVR (arXiv:2604.22312) | Cheng, Zhao, Liu, Li, Qiao, Duan, Chen, Chen, Rouhani, Yang — **NVIDIA** | arXiv Apr 2026 | **B200, DeepSeek-V3.2 NVFP4** | Exact top-K via temporal reuse: 1.88x avg kernel, 2.42x per-layer-step, **4.36-7.52% end-to-end TPOT**; bit-exact [verified] | None |
| ReTopK (arXiv:2607.27692) | Yao, Zhou, Shao, Chen, Ning, Feng, Huang, Tang | arXiv Jul 2026 | L20 | Similarity-guided top-K reuse across decode steps; **3.07x at 128k, K=512, +0.50% ppl** [verified] | None |

### What IndexCache actually measured, and what we should copy

IndexCache is the closest published analysis of our setting. Its empirical claim:
**"adjacent layers share 70-100% of their selected tokens"**, with early and late layers
showing much lower overlap (<=0.4) — i.e. the *middle* of the stack is where sharing is
safe and the *ends* are where it is not [verified].

Two methods:

- **Training-free**: start with every layer running its own indexer (all "Full"), then
  greedily convert one layer at a time to "Shared" (reusing the nearest Full layer's
  indices), each step choosing the conversion that minimises LM loss on a small calibration
  set from pretraining data. **This greedy choice consistently beat uniform interleaving at
  the same retention ratio** [verified].
- **Training-aware**: distil each retained indexer with a KL loss against the *average* of
  the attention distributions of all layers it serves — which is exactly the right objective
  if one index must serve four layers, and is mathematically the weighted sum of the
  per-layer KL terms [verified].

Numbers on GLM-4.7-Flash (30B-A3B MoE, MLA, **47 layers**) [verified]:

| Config | Long-context avg | Reasoning avg |
|---|---|---|
| Full DSA baseline | 50.2 (51.0 in the training-aware table) | 74.6 (74.2) |
| Training-free, 1/4 retention | 49.9 | 74.9 |
| Training-free, 1/8 retention | **46.1** | — |
| Training-aware, 1/4 retention | 50.6 | 74.1 |

**Read that table carefully. 1/4 retention is nearly free. 1/8 retention costs 4.1
long-context points.** Our `index_topk_freq=4` sits exactly at the knee. Going to 8 without
training-aware distillation is the wrong move; going to 8 *with* it is an open question
their paper does not answer.

Speedups at 200k on H100 with 1/4 retention: prefill 19.5s -> 10.7s (**1.82x**), per-request
decode 58 -> 86 tok/s (**1.48x**), full-KV-cache decode 197 -> 297 tok/s (**1.51x**). At 10k
context prefill is only 1.27x — the benefit is strictly a long-context benefit. On GLM-5
(744B/40B) they report "at least 1.3x improvement in both prefill latency and decode
throughput at context lengths beyond 100K" [verified].

**Actionable**: (a) run their greedy calibration on our 79-layer stack instead of assuming
uniform 1-in-4 is optimal — their data says the ends of the stack want their own indexers
and the middle can share more than 4 ways; (b) if we ever do a continued-pretraining pass,
the multi-layer distillation loss is ~30 lines and lets uniform patterns match full-indexer
accuracy.

### The other three axes of index amortisation

The literature has now found **four orthogonal axes** for making the indexer cheaper. We are
using exactly one.

1. **Layer axis** — IndexCache, CLSA. Share the index across layers. *We do this
   (`index_topk_freq=4`).*
2. **Head axis** — MISA. The indexer has `H^I = 64` heads; route each query to only `h = 8`
   of them using cheap block-pooled statistics, then only those 8 do token-level scoring.
   Cost drops from `O(H^I L)` to `O(hL + H^I M)` where `M = ceil(L/B) << L`. **Training-free
   drop-in** on pretrained DSA models. 3.82x indexer-kernel speedup on a single H200;
   LongBench 50.85 vs DSA 51.05 on DeepSeek-V3.2 and *46.43 vs 46.01 on GLM-5* (i.e. it
   slightly beat the baseline there); recovers >92% of DSA's per-layer selected tokens;
   NIAH fully green to 128k [verified]. The paper explicitly notes it is **complementary to
   IndexCache** because it operates on the head axis, not the layer or token axis
   [verified]. **This is the highest-value unexplored option for us: training-free, GLM-5
   validated, orthogonal to what we already do.**
3. **Granularity axis** — HISA. Two-stage: mean-pool the prefix into blocks of size B, score
   the `ceil(L/B)` block representatives, keep top-m blocks, then run the *original DSA
   indexer* only inside those blocks. Per-query cost `O(L/B + mB)` instead of `O(L)`.
   Training-free, no KV layout change, no architecture change. 2.16x at 4:1 compression and
   3.75x with a fixed 8k candidate budget at 64k context on A100; LongBench 50.78 vs DSA
   51.05, vs 49.54 for a plain block-sparse baseline; validated on **both DeepSeek-V3.2 and
   GLM-5 without fine-tuning** [verified].
4. **Time axis** — GVR, ReTopK. Reuse the previous decode step's selection. GVR's data on
   real DeepSeek-V3.2 traffic: **35-50% step-to-step top-K overlap in layers 20-60, 1-2% in
   layers 0-1** [verified]. Their explanation is the Toeplitz structure of RoPE scores —
   scores depend on `delta = n - m`, so advancing the query by one shifts the landscape
   smoothly.

### GVR in implementable detail (because it is free and it is our exact hardware)

Four phases [all verified from the paper]:

1. **Pre-indexed statistics.** Gather the values at the *previous* step's top-K indices and
   compute mean/min/max to form an initial threshold estimate `T_0`.
2. **Secant threshold search.** Iterate the secant method on the counting function `f(T)` =
   number of logits `>= T`, targeting `K <= f(T) <= 6144` with `K = 2048`. Converges in
   **1-2 iterations on real data**, versus 3-4 full passes for radix-select.
3. **Ballot-free candidate collection.** Collect everything above threshold into shared
   memory without `__ballot_sync` barriers (which serialise the memory pipeline), reusing
   the per-thread partial counts cached in phase 2 to avoid a second full scan.
4. **Histogram refinement.** If the candidate count != K, warp-parallel histogram + "snap"
   iterations resolve the exact K-th largest value.

Correctness: **bit-exact top-K index sets vs `torch.topk` for N = 8k-131k** [verified] —
which matters because GLM-5's own report says they switched to deterministic `torch.topk`
for RL stability [verified], and a heuristic that changed the selection would break that.

Results on B200 with DeepSeek-V3.2 NVFP4: 1.88x average single-operator, 1.59x (L0) to
2.42x (L22, L41) per layer per step, **7.52% end-to-end TPOT reduction at 100k in the TEP8
min-latency config, 5.47% at 64k, 4.36-6.30% with MTP=1** [verified]. Limitations they
state: minimal benefit below 16k; single-CTA design limits occupancy at large batch;
degraded prediction in layers 0-1 [verified]. Integrated in TensorRT-LLM PR #12385 behind
`enable_heuristic_topk` (default false), auto-disabled on pre-Blackwell.

**For us**: the min-latency (C1) objective is exactly the TEP8 min-latency config they
measured, and 5-7.5% TPOT at our 365 tok/s is ~20-27 tok/s. Cheap.

### CLSA / "You Only Index Once" — the aggressive version, and its B200 caveat

CLSA builds on YOCO-style KV sharing: a 32-layer / 4B model with 16 self-decoder layers and
16 cross-decoder layers; **one indexer, computed once, shared by all 16 cross-decoder
layers** (a 16:1 ratio, vs our 4:1). Adapted from a dense YOCO checkpoint by (1) distilling
the indexer against dense cross-attention with the backbone frozen, then (2) joint LM +
distillation fine-tuning (`lambda = 0.1`) at 32,768 tokens — **no from-scratch pretraining**
[verified]. 2048 selected tokens (~1:16). Measured on **NVIDIA B200**: 7.6x decode speedup
and 17.1x throughput at 128k, with quality *above* dense YOCO on ARC-C (0.465 vs 0.461),
GSM8K (0.470 vs 0.430), DROP (0.391 vs 0.387), RULER-32k (53.1 vs 52.3) [verified].

The one sentence in that paper most relevant to us: **"at 128K context, token-level
selection per layer can match dense attention cost despite selecting far fewer tokens"** —
they measured the unoptimised top-k routing becoming the bottleneck, and CLSA amortises it
down to ~0.08 ms when divided across 16 layers [verified]. That is the same phenomenon my
Part 8 arithmetic produces from first principles.

Caveat: 4B model, and it requires the YOCO KV-sharing backbone. Not a drop-in for us. But
it is the strongest evidence that 4:1 is conservative and that the quality risk of deeper
sharing is smaller than intuition suggests when the indexer is distilled for the job.

---

## Part 5 — Training-free KV selection, eviction and compression

| Paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| H2O (arXiv:2306.14048) | Zhang, Sheng et al. — UT Austin / Stanford / CMU / Meta | NeurIPS 2023 | not stated in abs | 20% heavy hitters; up to 29x/29x/3x throughput vs DeepSpeed-ZI / HF Accelerate / FlexGen; 1.9x latency [verified] | No — vLLM issue #3532 open, never merged |
| StreamingLLM (arXiv:2309.17453) | Xiao, Tian, Chen, Han, Lewis — MIT / Meta / CMU | ICLR 2024 | not stated in abs | 4M tokens stable; **22.2x vs sliding-window recomputation** [verified] | **Sinks yes, eviction no** |
| SnapKV (arXiv:2404.14469) | Li, Huang et al. — UIUC / Cohere | arXiv 2024 | **A100-80GB** | 3.6x decode, 8.2x memory; 380k tokens on one A100-80GB; negligible NIAH drop [verified] | Partially (some engines) |
| PyramidKV (arXiv:2406.02069) | Cai, Zhang et al. | arXiv 2024 | Llama-3-70B | Matches full cache at **12% KV** on LongBench [verified] | No |
| Quest (arXiv:2406.10774) | Tang, Zhao, Zhu, Xiao, Kasikci, Han — MIT / UW | **ICML 2024** | not stated in abs | 2.23x self-attention, **7.03x inference latency**, negligible accuracy loss on long-dependency tasks [verified] | Reference implementation; ideas widely copied |
| UNIQUE (arXiv:2605.27740) | Deng, Ling, Fan, Li — **Microsoft** | arXiv May 2026 | **H100** | Page score `q.mean_p + 0.5*||q||*std_p`; **11.4x attention kernel at 32k vs dense FlashInfer, 5.3x e2e decode vs vLLM**; LongBench-Pro 36.58 @512-token budget vs Quest 21.72, H2O 29.04; 97.0% of full attention [verified] | Research |
| Hold Onto That Thought (arXiv:2512.12008) | Liu, Palnitkar et al. — UMD / UChicago / Utah | arXiv Dec 2025 | reasoning models 7-14B | **Failure catalogue** — see below [verified] | evaluation |

### The mechanisms, and the one thing they all get wrong for us

- **H2O**: accumulate attention scores per key across the generated sequence; the top ~20%
  by accumulated score are "heavy hitters"; keep those plus a recent window, evict the rest.
  Formulated as dynamic submodular maximisation with a guarantee.
- **StreamingLLM**: the observation is that the first few tokens absorb enormous attention
  mass regardless of semantics ("attention sink"), because softmax must put its mass
  *somewhere* and early tokens are visible to every query. Keep the first ~4 tokens plus a
  rolling window, and **re-index positions by position-within-cache rather than absolute
  position**. That last detail is what makes it work at all.
- **SnapKV**: use a short observation window at the *end* of the prompt, compute its
  attention to all earlier prompt tokens, pool (a 1-D pooling over the score vector, which
  preserves contiguous spans instead of scattered singletons), and keep the top positions
  per head. Prompt-time only; the cache is fixed after prefill.
- **PyramidKV**: allocate a *layer-varying* budget — more cache in early layers, less in
  late layers — motivated by "pyramidal information funneling" (attention is diffuse early,
  concentrated late).
- **Quest**: per KV *page*, store elementwise min and max of the keys. At decode, upper-bound
  the page's max possible score by `sum_i max(q_i * min_i, q_i * max_i)`, rank pages by that
  bound, load only the top pages. This is a *selection* method, not an eviction method — the
  full cache stays resident, only the reads are sparse. That is why it survives where H2O
  does not.
- **UNIQUE (2026)**: the modern refinement of Quest's idea. Score a page by
  `q . mean_p + lambda * ||q||_2 * std_p` with `lambda = 0.5` — the mean term is the cheap
  representative, the std term compensates for the fact that averaging dilutes a single
  important key in a page. Adds a fused criticality kernel (1.73x over naive) and a
  radix-based top-k (up to 2.0x over FlashInfer). Training-free deployment *plus* a
  sparsity-aware training mode using a soft sigmoid mask around the top-k boundary so
  gradients flow without auxiliary losses [verified].

**The structural distinction that decides everything: selection vs eviction.**
Quest and UNIQUE *select* which resident pages to read — the information is never destroyed,
so a later query can retrieve a token an earlier query ignored. H2O, SnapKV, PyramidKV and
StreamingLLM *evict* — the information is gone. For a reasoning model that thinks for
thousands of tokens, eviction decisions made at prompt time or early-generation time are
made without knowledge of what the model will need. DSA is a selection method. That is the
right side of the line, and it is why we should not bolt an eviction method onto it.

---

## Part 6 — Linear and hybrid attention, only where production uses it

| Model / paper | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| Kimi Linear (arXiv:2510.26692) | Kimi Team — Moonshot | arXiv Oct 2025 | not stated in abs | KDA (channel-wise gated delta rule) 3:1 with **MLA-NoPE** full-attention layers; 48B-A3B; **KV cache -75%, 6x TPOT at 1M (1.84 ms vs 11.48 ms)**, prefill 2.3x @512k / 2.9x @1M; MMLU-Pro 51.0 vs 47.2 MLA, RULER 84.3 vs 81.3 [verified] | **Yes** — weights + vLLM impl released |
| Qwen3-Next (vLLM blog, Sept 2025) | Qwen / Alibaba | blog | not stated | Gated DeltaNet interleaved 3:1 with full attention [reported] | **Yes** |
| gpt-oss-120b (arXiv:2508.10925) | OpenAI | arXiv 2025 | — | MoE transformer, 117B/5.1B active, MXFP4 MoE weights [verified from abstract; **layer/attention details not in the page I fetched — I am not going to state its sink or window config from memory**] | Yes |

The relevant fact for us is narrow: **hybrid linear attention is now a mainstream production
choice, and the full-attention layers in those hybrids are MLA.** Kimi Linear's 1-in-4 full
layers use MLA with **NoPE** (no positional encoding at all in the full layers, since the
KDA layers carry position) [verified]. That is a structurally different bet from ours
(sparse-everywhere) and it wins on a different axis: KDA's state is fixed-size, so KV memory
stops growing with context, whereas DSA's KV still grows linearly with L even though the
*reads* are O(k).

For our roadmap: if Kimi K3 ships as a Kimi-Linear-shaped hybrid, our KV manager needs to
handle **two cache types in the same model** — a fixed-size recurrent state for the linear
layers and a paged latent cache for the MLA layers — with different eviction, different
prefix-cache semantics (recurrent state is *not* prefix-shareable in the usual way; you must
snapshot the state at the shared-prefix boundary), and different offload behaviour. That is
a serving-stack change, not a kernel change, and it is worth scoping before the model lands.

---

## Part 7 — KV cache management: paging, reuse, offload, routing

| Paper / system | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| PagedAttention / vLLM (arXiv:2309.06180) | Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica — UC Berkeley | **SOSP 2023** | not stated in abs | **2-4x throughput** vs FasterTransformer / Orca; near-zero KV fragmentation [verified] | Universal |
| vAttention (arXiv:2405.04437) | Prabhu, Nayak, Mohan, Ramjee, Panwar — Microsoft Research India | **ASPLOS 2025** | not stated in abs | Contiguous virtual KV via CUDA VMM APIs; **up to 1.23x** over PagedAttention kernels in FA/FlashInfer [verified] | Research / partial adoption |
| SGLang + RadixAttention (arXiv:2312.07104) | Zheng, Yin, Xie, Sun, Huang, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng | arXiv 2023 | not stated in abs | **up to 6.4x throughput** via radix-tree KV reuse + compressed FSM [verified] | Universal |
| Mooncake (arXiv:2407.00079) | Qin, Li, He, Zhang, Wu, Zheng, Xu — **Moonshot AI** | arXiv 2024 (v4 Sep 2025) | Kimi production | KVCache-centric PD disaggregation over CPU DRAM/SSD via RDMA; **+525% throughput simulated, +75% real request capacity** [verified] | **Yes — serves Kimi** |
| CacheBlend (arXiv:2405.16444) | Yao, Li, Liu, Ray, Cheng, Zhang, Du, Lu, Jiang — UChicago | arXiv 2024 | not stated in abs | **Non-prefix** KV reuse with selective recompute; TTFT 2.2-3.3x lower, throughput 2.8-5x, quality preserved [verified] | Yes — in LMCache |
| LMCache (docs.lmcache.ai) | LMCache project | docs | — | Tiered GPU / CPU DRAM / local disk / Redis-Valkey / S3 / Bigtable; CacheBlend for non-prefix reuse [verified] | Yes — vLLM production integration |
| Irminsul (arXiv:2605.05696) | Ma, Eitzinger, Köstler | arXiv May 2026 | not stated | **MLA-native position-independent caching**; ~83% of prompt tokens recovered above exact-prefix on agentic traffic; 63% prefill energy per hit [reported] | Research |
| Understanding Bottlenecks for KV Offloading (arXiv:2601.19910) | Meng (UPenn/Intel), Lee (UPenn), Wang (Intel) | arXiv Dec 2025 | GPU + PCIe | Defines `kappa_crit`; **99% of latency is data transfer; GPU at 28% of rated TDP** when serving offloaded requests [verified] | analysis |
| Unified KV Pooling (arXiv:2606.14779) | Kang et al. — Korea Univ. | arXiv Jun 2026 | host DRAM + SSD | SPDK bypass unified pool; **~4.1x TTFT** vs SOTA, eliminates 84% of filesystem overhead [reported] | Research |

### What actually applies to us

**Paging.** PagedAttention is settled. The one thing to know for DSA: vLLM only supports
**block size 64** for DeepSeek-V3.2, because FlashMLA's sparse kernel is tailored to it and
the indexer cache stores scale factors per 64-token block [verified]. If our KV manager
assumes 16 or 128 anywhere, that is a bug waiting to happen. vAttention's 1.23x is real but
small and it does not compose trivially with a sparse gather kernel; not a priority.

**Prefix caching.** Our measured 1.54x is a strict-prefix number. Three ways to extend it:

1. **CacheBlend / non-prefix reuse.** Reuse cached KV for chunks that are *not* a prefix,
   then selectively recompute a small subset of tokens (the ones whose KV deviates most) to
   repair cross-chunk attention. 2.2-3.3x TTFT, quality preserved [verified]. This is what
   makes RAG-style multi-chunk prompts cacheable. Shipped in LMCache.
2. **Position-independent MLA caching (Irminsul).** MLA's latent row is
   `c_KV` (position-free) + a 64-dim `k_R` correctable by a closed-form rotation. So you can
   content-hash a token's KV and *re-position it* at cache-hit time by applying the delta
   rotation to the 64-dim part only. That converts prefix caching into **content-addressed**
   caching, which is exactly what agentic traffic (tools inserted mid-context, turns
   reordered) needs. Tested on DeepSeek-V2-Lite, Kimi Moonlight-16B-A3B, JoyAI-Flash
   [reported]. **This is a genuinely MLA-specific capability that GQA models cannot have**,
   and we are running an MLA model. Highest-leverage unexplored item in this section.
3. **Cache-aware routing.** SGLang's RadixAttention plus cache-aware scheduling is the
   baseline. The 2026 literature has moved on to the second-order problem: BanaServe
   (arXiv:2510.13223) documents that "prefix cache aware routing skews load distribution, as
   high cache hit rate prefill nodes attract disproportionately more requests" [reported];
   PrefixShield (arXiv:2608.01657) measures a multi-tenant cache-pollution failure where
   victim hit ratio collapses from 84.87% to 4.92% under LRU and recovers +9.39 points with
   admission control [reported]; PrefixPlace (arXiv:2608.01655) reports 40.3% better
   materialisation-cost savings than vLLM's automatic prefix caching [reported]. If we run
   multi-tenant, the *pollution* result is the one to act on.

**Offloading — read the bad news first.** arXiv:2601.19910 is the sober analysis: for
KV offload to host DRAM over PCIe, **99% of latency is data transfer and the GPU sits at
28% of rated TDP** [verified]. They define `kappa_crit`, the cached-to-prefill token ratio
above which you are memory-bound, and show typical workloads blow past it. On our box the
numbers are worse than the paper's baseline in one way and better in another: PCIe Gen5 x16
is ~64 GB/s per direction against 8 TB/s of HBM — a **125x** gap. Moving one 128k MLA cache
(6.79 GB FP8, computed in Part 8) over PCIe takes ~106 ms; recomputing that prefill on 8
B200s takes on the order of a few hundred ms. **So host-DRAM offload of MLA KV is roughly
break-even against recompute, and only wins when the prefill is expensive relative to the
cache size** — which for MLA is unusual precisely *because* MLA's cache is so small relative
to its prefill cost. Offload is a much better deal for GQA models than for MLA models. Say
this out loud before anyone builds it.

---

## Part 8 — Where KV memory and attention bandwidth actually bind, on 8xB200, at 128k

All arithmetic below is **[inferred]** from verified config numbers. Assumptions stated.
Shapes: 79 layers, 22 indexer layers (`index_topk_freq=4`), MLA latent `d_c + d_h^R = 576`,
`index_head_dim = 128`, `index_n_heads = 64`, `index_topk = 2048`, B200 at 8 TB/s and 183 GB.

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
single largest number in this document. (At short context it matters much less, which is
why it can hide: at 4k context TP8 still gives ~500 concurrent streams.)

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

Sparse attention bytes are **constant in L**: `k * B_kv * n_layers`.
Indexer bytes are **linear in L**: `L * B_idx * n_idx_layers`.
Setting them equal:

```
L* = (k * B_kv * n_layers) / (B_idx * n_idx_layers)
```

| Config | L* (crossover context) |
|---|---|
| FP8 latent (656 B), 22 index layers | **~36,500 tokens** |
| BF16 latent (1152 B), 22 index layers | **~64,200 tokens** |
| FP8 latent, 79 index layers (no sharing) | **~10,200 tokens** |
| BF16 latent, 79 index layers (no sharing) | **~17,900 tokens** |

Read the first and third rows together: **`index_topk_freq=4` moves the point at which the
indexer starts dominating from 10k context to 36k.** That is what our config is buying, and
it is why removing it would be catastrophic for long-context serving, not merely 5% worse.

### 8.5 FLOPs and arithmetic intensity of the indexer

Per layer, per query token: `FLOPs = 2 * H^I * d_I * L = 2 * 64 * 128 * L = 16,384 L`.
Bytes read (FP8 indexer K, one key vector per token): `d_I * L = 128 L`.

```
Indexer arithmetic intensity = 16,384 L / 128 L = 2 * H^I = 128 FLOP/byte
```

**Independent of L, of d_I, and of batch size.** B200's BF16 ridge is ~281 Op/B and its FP8
ridge is ~562 Op/B. **The lightning indexer sits 4.4x below the FP8 ridge and can never
climb — batching does not help, because each sequence has its own indexer K cache.** It is a
permanently memory-bound kernel bolted onto an attention that MLA has made compute-bound.
That single sentence explains why every 2026 paper in Part 4 attacks the indexer and not the
attention.

Absolute FLOPs at 128k (DeepSeek-V3.2 shape, 128 heads, verified config, all 61 layers
indexed):

- Indexer: `61 x 2 x 64 x 128 x 131,072` = **131 GFLOP / token**
- Sparse absorbed MLA: `61 x (2 x 128 x 576 x 2048 + 2 x 128 x 512 x 2048)` = **34.8 GFLOP /
  token**

**The indexer costs 3.8x the FLOPs of the attention it selects for.** With our 22/79 sharing
the ratio falls to roughly parity. This is the quantitative justification for `index_topk_freq`
existing at all, and it matches GLM-5's own claim that DSA delivers only "approximately
1.5-2x" compute reduction for long sequences [verified] — far less than the naive
`L/k = 64x` you would expect from top-k alone, because the indexer eats most of it.

### 8.6 At concurrency 64

Sparse attention scales linearly with batch (each sequence selects different tokens):
`64 x 106.1 MB = 6.79 GB` -> **849 us/step**. Indexer: `64 x 380.6 MB = 24.4 GB` ->
**3.05 ms/step**. Our measured 40.8k tok/s at C64 implies ~1.57 ms/step, so **at 128k
context, C64, the indexer alone exceeds the entire current step budget by ~2x** and the
combined attention path is ~2.5x the budget. Concretely: **our C64 aggregate throughput
cannot survive a move to 128k contexts without either deeper index sharing, MISA-style head
routing, HISA-style hierarchical indexing, or all three.** Without any index sharing at all
it would be 11 ms/step — 7x over budget.

### 8.7 What binds, summarised

| Regime | Binding resource |
|---|---|
| C1, short context (<16k) | Dense GEMM + collectives. Attention is a rounding error. Matches our 37.1% / 19.6% / 10.9% profile. |
| C1, 128k | Indexer HBM bandwidth (47.6 us) > sparse attention bandwidth (13.3 us). Top-k selection kernel is the next term (GVR measures 4-7.5% TPOT). |
| C64, short context | MoE expert GEMMs + collective skew. |
| C64, 128k | **Indexer bandwidth, by a factor of 2 over everything else.** |
| Any regime, capacity | KV pool size, and therefore TP8-vs-DP8 attention, long before HBM bandwidth. |
| With EAGLE 3-1-4 | Attention intensity multiplies by draft length -> attention moves compute-bound; the indexer does **not** (it runs once per verify step regardless), so speculation *increases* the indexer's relative share. |

That last row is the non-obvious one. Speculative decoding is worth 3.09x to us because it
amortises weight loading over multiple tokens. It does **not** amortise the indexer, because
the indexer must score the full prefix for the query position(s). If we index once per
verify step and share that index across all draft positions (justified by GVR's 35-50%
step-to-step overlap in deep layers), the indexer's cost per accepted token drops by the
acceptance length — currently ~2.76 for GLM-5's MTP [verified]. That is a **2.76x reduction
in the dominant long-context term**, and it is the single highest-value unimplemented idea I
found in this literature for our specific stack.

---

## What is NOT worth it

1. **Attention-score KV eviction (H2O, SnapKV, PyramidKV, StreamingLLM-as-eviction) on a
   reasoning model.** "Hold Onto That Thought" (arXiv:2512.12008) tested these on
   DeepSeek-R1-Distill-Qwen-7B/14B, R1-Distill-Llama-8B, Nemotron-Nano-8B across GSM8K,
   MATH-500, FOLIO, DROP, ReClor, StrategyQA. Results at budget 128-256 [verified]:
   PyramidKV **0.00** on MATH-500 with R1-Distill-Llama-8B at budget 256; StreamingLLM
   **0.00** on ReClor vs 0.45-0.51 full cache; KNorm **0.00** on GSM8K vs 0.70 full cache.
   The paper's qualitative finding is that eviction produces "repetitive, dead-end
   chain-of-thought" — the model loses the thread and loops. Root cause: eviction commits at
   prompt time or early generation; a reasoning model's critical tokens are produced
   thousands of steps later. **Selection (Quest, UNIQUE, DSA) does not have this failure
   mode because nothing is destroyed.** We already have DSA. Do not add eviction.
2. **Host-DRAM / NVMe offload of the MLA KV cache.** PCIe Gen5 x16 at ~64 GB/s vs 8 TB/s
   HBM is a 125x gap; arXiv:2601.19910 measures 99% of latency as transfer and the GPU at
   28% TDP under offload [verified]. MLA's cache is already 3.6-6.3x smaller than GQA's, so
   the thing offload is designed to solve barely exists for us. It is a sound technique for
   GQA models with 320 KiB/token; for our 50.6 KiB/token it is roughly break-even against
   just recomputing the prefill. The exception worth revisiting: cross-*request* KV reuse
   (Mooncake, LMCache) where the alternative is a cache miss and a full prefill, not a
   resident cache — that is a different economic calculation and it does pay.
3. **Porting NSA/MoBA-style architecture changes to an already-trained model.** NSA requires
   pretraining with the three-branch structure; MoBA "requires continue training of existing
   models and is not a drop-in sparse attention solution" [reported from the MoBA repo];
   MSA needs a 40B-token indexer warm-up [verified]. DSA already gave us the sparsity. The
   marginal architectures are for the *next* pretrain, not this deployment. The training-free
   improvements (MISA, HISA, GVR, ReTopK, UNIQUE) are where our engineering time goes.
4. **Chasing FlashAttention-4 for the MLA decode path.** FA4 is excellent and it is the right
   kernel for dense prefill and for GQA models, but the paper explicitly does not cover head
   dim 576/512, does not cover MLA, and lists sparse attention as future work [verified].
   Its actual transferable content for us is the *techniques* — polynomial `exp` on FMA
   units (MUFU is 16 ops/clk vs 8192 for tensor cores) and conditional softmax rescaling at
   `tau = log2(256)` — which should be ported into whatever sparse-MLA decode kernel we end
   up using, since that kernel also runs a softmax over 2048 entries per head per layer.
5. **Going to `index_topk_freq=8` without training-aware distillation.** IndexCache measured
   training-free 1/8 retention at **46.1 long-context average vs 50.2 baseline** — a 4.1
   point drop, versus 49.9 at 1/4 [verified]. The knee is exactly where we are sitting. With
   their multi-layer distillation loss, 1/4 recovered to 50.6 (above the 51.0 baseline's
   neighbourhood); they did not publish a distilled 1/8 number, so treat 1/8 as unvalidated.
6. **Uniform layer selection for index sharing.** Not "not worth it" so much as "leaving
   money on the table": IndexCache's greedy loss-calibrated selection "consistently
   outperforms uniform interleaving at equivalent retention ratios", and their overlap data
   shows early and late layers have <=0.4 cross-layer agreement while the middle has 0.7-1.0
   [verified]. Uniform 1-in-4 is simultaneously too aggressive at the ends of the stack and
   too conservative in the middle.
7. **vAttention as a replacement for PagedAttention in our stack.** 1.23x over
   PagedAttention kernels [verified] is real, but it requires kernels that assume contiguous
   virtual KV — and our hot path is FlashMLA's sparse gather, which is built around the
   64-token paged layout with per-block FP8 scales [verified from vLLM]. The integration
   cost exceeds the benefit.

---

## Sources

Every URL below was fetched during this pass.

**Kernels and the online-softmax lineage**
- Milakov & Gimelshein (NVIDIA), "Online normalizer calculation for softmax", arXiv:1805.02867 — https://arxiv.org/abs/1805.02867
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135 — https://arxiv.org/abs/2205.14135
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", arXiv:2307.08691 — https://arxiv.org/abs/2307.08691
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", arXiv:2407.08608 — https://arxiv.org/abs/2407.08608
- Zadouri, Hoehnerbach, Shah, Liu, Thakkar, Dao, "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling", arXiv:2603.05451 — https://arxiv.org/html/2603.05451v1
- Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference", PyTorch blog, Oct 2023 — https://pytorch.org/blog/flash-decoding/
- DeepSeek, FlashMLA — https://github.com/deepseek-ai/FlashMLA
- Hassani et al., "Generalized Neighborhood Attention", arXiv:2504.16922 (B200 sparse-attention kernels, 1.3 PFLOP/s FP16) — abstract via arXiv API

**KV cache shape: MQA / GQA / MLA**
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150 — https://arxiv.org/abs/1911.02150
- Ainslie et al. (Google), "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023, arXiv:2305.13245 — https://arxiv.org/abs/2305.13245
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434 — https://arxiv.org/html/2405.04434v5
- DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437 — https://arxiv.org/abs/2412.19437
- Geens & Verhelst (KU Leuven), "Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention", Electronics Letters 2025, arXiv:2506.02523 — https://arxiv.org/html/2506.02523v1
- Yun et al. (SNU + UIUC), "The New LLM Bottleneck: A Systems Perspective on Latent Attention and Mixture-of-Experts", arXiv:2507.15465 — https://arxiv.org/html/2507.15465v1
- Zadouri, Strauss, Dao, "Hardware-Efficient Attention for Fast Decoding", arXiv:2505.21487 — https://arxiv.org/abs/2505.21487
- Meng (PKU MuLab), "GQLA: Group-Query Latent Attention for Hardware-Adaptive LLM Decoding", arXiv:2605.15250 — https://arxiv.org/abs/2605.15250
- Han, Zhao, Zhou, Li, Sun, "QK-Normed MLA: QK normalization without full key caching", arXiv:2606.16310 — https://arxiv.org/abs/2606.16310
- Ma, "A Training-Memory Regression in MLA Sequence Parallelism: Why Megatron-Core Forbids Absorption, and LAGA", arXiv:2607.17644 — https://arxiv.org/abs/2607.17644

**Sparse attention (trained)**
- Yuan et al. (DeepSeek + PKU + UW), "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention", ACL 2025, arXiv:2502.11089 — https://arxiv.org/html/2502.11089v1 and .../v2
- Lu et al. (Moonshot AI + Tsinghua + Zhejiang), "MoBA: Mixture of Block Attention for Long-Context LLMs", arXiv:2502.13189 — https://arxiv.org/html/2502.13189v1
- DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", arXiv:2512.02556 — https://arxiv.org/html/2512.02556v1
- DeepSeek-V3.2-Exp reference implementation and config — https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py and .../config_671B_v3.2.json
- GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763 — https://arxiv.org/html/2602.15763v2
- Lai et al. (MiniMax + PKU + NVIDIA), "MiniMax Sparse Attention", arXiv:2606.13392 — https://arxiv.org/html/2606.13392v1
- DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv:2606.19348 — https://arxiv.org/html/2606.19348v1
- vLLM blog, "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action", Sep 2025 — https://vllm.ai/blog/2025-09-29-deepseek-v3-2

**Index sharing and indexer cost**
- Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li (THUDM), "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse", arXiv:2603.12201 — https://arxiv.org/html/2603.12201v1
- Sun, Zhang, Dong, Wang, Wei, "You Only Index Once: Cross-Layer Sparse Attention with Shared Routing", arXiv:2606.06467 — https://arxiv.org/html/2606.06467v1
- Zhou, Meng, Xu, Liu, Lu, Zhang, Pei (PKU MuLab), "MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference", arXiv:2605.07363 — https://arxiv.org/html/2605.07363v1
- Xu, Meng et al. (PKU MuLab), "HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention", arXiv:2603.28458 — https://arxiv.org/html/2603.28458v3
- Cheng, Zhao, Liu, Li, Qiao, Duan, Chen, Chen, Rouhani, Yang (NVIDIA), "Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation", arXiv:2604.22312 — https://arxiv.org/html/2604.22312v1
- Yao et al., "Recall Before You Rank: Similarity-Guided Top-K Reuse for Efficient Long-Context Attention", arXiv:2607.27692 — https://arxiv.org/html/2607.27692v1
- Raschka, "GLM-5.2 and IndexShare for Long-Context Sparse Attention", Jun 2026 — https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html

**Training-free KV selection / eviction**
- Zhang, Sheng et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023, arXiv:2306.14048 — https://arxiv.org/abs/2306.14048
- Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks", ICLR 2024, arXiv:2309.17453 — https://arxiv.org/abs/2309.17453
- Li, Huang et al., "SnapKV: LLM Knows What You are Looking for Before Generation", arXiv:2404.14469 — https://arxiv.org/abs/2404.14469
- Cai, Zhang et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling", arXiv:2406.02069 — https://arxiv.org/abs/2406.02069
- Tang, Zhao, Zhu, Xiao, Kasikci, Han, "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024, arXiv:2406.10774 — https://arxiv.org/abs/2406.10774
- Deng, Ling, Fan, Li (Microsoft), "UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training", arXiv:2605.27740 — https://arxiv.org/html/2605.27740v1
- Liu, Palnitkar et al., "Hold Onto That Thought: Assessing KV Cache Compression On Reasoning", arXiv:2512.12008 — https://arxiv.org/html/2512.12008v1

**Linear / hybrid attention in production**
- Kimi Team (Moonshot), "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv:2510.26692 — https://arxiv.org/html/2510.26692v2
- vLLM blog, "vLLM Now Supports Qwen3-Next: Hybrid Architecture with Extreme Efficiency", Sep 2025 — https://vllm.ai/blog/2025-09-11-qwen3-next
- OpenAI, "gpt-oss-120b & gpt-oss-20b Model Card", arXiv:2508.10925 — https://arxiv.org/abs/2508.10925

**KV cache management systems**
- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023, arXiv:2309.06180 — https://arxiv.org/abs/2309.06180
- Prabhu, Nayak, Mohan, Ramjee, Panwar (MSR India), "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention", ASPLOS 2025, arXiv:2405.04437 — https://arxiv.org/abs/2405.04437
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", arXiv:2312.07104 — https://arxiv.org/abs/2312.07104
- LMSYS/SGLang, "Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism", May 2025 — https://lmsys.org/blog/2025-05-05-large-scale-ep/
- Qin, Li, He, Zhang, Wu, Zheng, Xu (Moonshot AI), "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", arXiv:2407.00079 — https://arxiv.org/abs/2407.00079
- Yao, Li, Liu, Ray, Cheng, Zhang, Du, Lu, Jiang, "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion", arXiv:2405.16444 — https://arxiv.org/abs/2405.16444
- LMCache documentation — https://docs.lmcache.ai/
- Ma, Eitzinger, Köstler, "Irminsul: MLA-Native Position-Independent Caching for Agentic LLM Serving", arXiv:2605.05696 — https://arxiv.org/abs/2605.05696
- Meng, Lee, Wang (UPenn + Intel), "Understanding Bottlenecks for Efficiently Serving LLM Inference With KV Offloading", arXiv:2601.19910 — https://arxiv.org/abs/2601.19910
- NVIDIA, TensorRT-LLM DeepSeek-V3 example README — https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/models/core/deepseek_v3/README.md
- He et al., "BanaServe: Unified KV Cache and Dynamic Module Migration...", arXiv:2510.13223; Wang & Buyya, "PrefixShield" arXiv:2608.01657 and "PrefixPlace" arXiv:2608.01655; Kang et al., "Unified KV Pooling to Accelerate Long-Context LLM Serving", arXiv:2606.14779 — all via arXiv API listing

---

## Open questions this survey did not settle

1. Our build reports **79 layers / 22 indexer layers**; the GLM-5 technical report says 80
   layers and Raschka's GLM-5.2 write-up says 78 with a clean 4-layer group. 22 is not
   79/4 = 19.75, so some layers (probably the first few dense layers, or the boundary
   layers) run their own indexer outside the pattern. Worth reading our own config loader to
   find out which — because IndexCache's data says the boundary layers are exactly the ones
   that *should* be exempt, and if that is already true, our pattern is better than uniform.
2. Nobody has published the interaction between **cross-layer index sharing and speculative
   decoding**. GVR handles the time axis, IndexCache the layer axis; the product (share the
   index across both the 4-layer group *and* the 4 EAGLE draft positions) is a 16x
   amortisation that nobody has measured for quality.
3. **MISA + IndexCache composition** is claimed to be orthogonal by MISA's authors but was
   not measured. `4x (layer) * 8x (head) = 32x` indexer reduction is the theoretical
   product; the real number is unknown.
4. FlashMLA's **sparse decode being slower on B200 (350 TFLOPS) than H800 (410 TFLOPS)** is
   reported in the repo README with no explanation. Either the kernel is untuned for SM100
   or the benchmark configs differ. This needs a local profile, not a literature search.
5. No published measurement exists of **DSA quality at 1M context with index sharing** — GLM
   trains to 200K and extends to 1M, and IndexCache's longest measurement is 200k.
