# Speculative decoding: the literature, and how far it can be pushed

## What this is

A survey of the speculative-decoding literature written against one specific
question: **we measure 3.09× from EAGLE/MTP speculation at concurrency 1
(144.8 → 447.8 tok/s, accept length 3.16 of 4 on ShareGPT-like traffic), we are
stuck at a `3-1-4` draft configuration, and published recipes for the same model
on the same hardware run `5-1-6`. What does the literature say about how much
further this goes, and what stops us?**

Every paper below was fetched and read (abstract + method + evaluation numbers).
Every number carries the hardware and the model it was measured on, because a
2.9× on Vicuna-7B/A100 and a 3.09× on a 256-expert MoE with sparse attention on
8×B200 are not the same claim. Labels used throughout:

- `[verified]` — I read this number in the paper / source / repo.
- `[reported]` — the authors or vendor assert it in text I read, without me
  seeing the underlying measurement.
- `[inferred]` — my arithmetic or reasoning, not stated by any source.
- `[unverified]` — plausible, stated somewhere, not independently checked.

A note on scope: the SGLang tree used here is `sgl-project/sglang@main` as of
2026-08-17, which has moved well past the published papers — it ships `EAGLE`,
`EAGLE3`, `NEXTN`/`FROZEN_KV_MTP`, `STANDALONE`, `NGRAM`, `DFLASH`, `DSPARK`,
adaptive per-batch-size depth, and decoupled draft/verify engines. Source
excerpts below are from that tree.

---

## Bottom line for our system

Ranked by expected value, with the reasoning compressed. Details in the numbered
sections.

1. **The `3-1-4` ceiling is almost certainly not an `eagle_worker_v2` bug — it is
   the draft-MoE runner backend.** SGLang issue
   [#30209](https://github.com/sgl-project/sglang/issues/30209) is our exact
   configuration (GLM-5.2 NVFP4, `GlmMoeDsaForCausalLM`, 8×B200, TP8, EAGLE
   `5-1-6`) crashing with an illegal memory access inside a flashinfer TRTLLM
   `Bmm_Bfloat16..._dynB_sm100f` batched-GEMM cubin used by the **bf16 nextn/MTP
   draft layer's MoE**. The fix, A/B-validated on mirrored production traffic
   (control crashed 5× in 5h; treatment 0 restarts in 5h, 24h+ crash-free), is
   `--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none`
   `[verified — read the issue and the closed PR]`. The protective default
   **still does not apply on CUDA**: `_deepseek_spec_moe_resolution` in
   `arg_groups/overrides.py` returns `{}` unless `is_hip()`, and PR
   [#30210](https://github.com/sgl-project/sglang/pull/30210) that removes the
   gate was **closed unmerged** `[verified — `merged: false` from the GitHub
   API]`. So the draft MoE inherits `moe_runner_backend`, which on Blackwell is
   `flashinfer_trtllm`, the crashing path. **Set those two flags first.** Cost:
   one flag pair. Expected effect: unblocks `5-1-6`.
2. **Our verification is nearly free, which makes depth the cheapest lever we
   have — cheaper than the literature assumes.** From our own numbers: plain
   decode step = 1000/144.8 = **6.906 ms**; spec cycle = 3.16×1000/447.8 =
   **7.057 ms**; the entire cost of 3 draft-model forwards plus verifying 4
   tokens instead of 1 is **+2.18%** `[inferred from our measured numbers]`.
   That is far below what a bandwidth model predicts, and it is consistent with
   our profile: 19.6% collectives of which 47% is rank-arrival skew, i.e. the
   C1 decode step is launch/sync-latency-bound, not bytes-bound. Under a
   latency-bound step the marginal cost of extra verify tokens is ~0. Push depth.
3. **Expected value of `3-1-4` → `5-1-6` on our traffic: accept length
   3.16 → ~3.9–4.1, cycle time +3–12%, net +15–25% ⇒ ~515–560 tok/s**
   `[inferred]`. Two independent anchors: (a) a depth-decayed acceptance fit to
   our own 3.16@k=3 gives E(k=5) = 3.85–4.10 across plausible decay rates;
   (b) SGLang PR [#29787](https://github.com/sgl-project/sglang/pull/29787)
   publishes measured accept lengths for **GLM-5.2 NVFP4 at exactly `5-1-6`,
   TP4, temp 0**: mtbench 3.66, humaneval 4.33, gsm8k 4.49, math500 4.61, aime
   4.63, openhands-longctx 5.18 `[verified]`. Our 3.16@`3-1-4` on ShareGPT sits
   right where that curve predicts.
4. **Do not chase `topk > 1`.** In current SGLang, a tree draft costs you three
   things we depend on: the overlap scheduler (spec-v2) *requires*
   `--speculative-eagle-topk 1` and errors otherwise `[verified — docs]`;
   adaptive speculative decoding is topk-1 only `[verified — `adaptive_spec_params.py`]`;
   and GLM-5.2's `index_share_for_mtp_iteration` — which reuses the DSA
   indexer's top-k across draft steps instead of recomputing it, i.e. removes
   our 5.8% indexer cost from every draft step — is "effective only at
   `--speculative-eagle-topk 1`" `[verified — SGLang GLM-5.2 cookbook]`. The
   EAGLE-2/Sequoia tree literature buys ~20% accept length; those three
   losses cost more.
5. **Retrain the draft head on our own traffic.** This is the largest
   accept-length lever after depth and it is not a systems change. The pattern
   is now standard: LMSYS's SpecBundle ships a *regenerated* dataset per target
   model (the target's own outputs, not off-the-shelf ShareGPT text) for every
   released EAGLE3 draft `[verified — SpecForge docs]`; EAGLE-3's "training-time
   test" feeds the draft its own predictions during training `[verified]`; HASS
   (ICLR 2025) reports 8–20% over EAGLE-2 from fixing exactly this
   train/decode mismatch `[reported]`. Expect +0.3–0.8 accept length from an
   on-policy-distilled head; that is worth more than any kernel we could write.
6. **Turn on `--speculative-adaptive` and stop hand-tuning depth per batch
   size.** SGLang's built-in ladder is `BS 1 → [1,3,7]`, `BS 8 → [0,1,3]`,
   `BS 32 → [0,1]`, `BS 64 → [0]` (speculation off) `[verified — `DEFAULT_ADAPTIVE_CONFIG`]`.
   The originating PR reports GLM-4.7-FP8 optimal at step=7 at BS 1 and step=3
   at BS 64, and +6.2% average over a fixed `num_steps=3` `[reported]`.
7. **Never quote an accept length measured on `--dataset-name random`, and never
   quote a throughput measured with `SGLANG_SIMULATE_ACC_LEN`.** The published
   "540 tok/s at `5-1-6`" for GLM-5.2 NVFP4 on B200 is TPOT 1.85 ms from
   SGLang's own benchmark file — measured on the *random* dataset with the
   acceptance length **pinned to 3.5 by environment variable**, not measured
   `[verified — `glm-5.2-benchmarks.jsx` + the cookbook note]`. It is a clean
   measurement of *engine step time at a fixed accept length*. It is not a claim
   about what the MTP head achieves on prose. Our 447.8 tok/s at a *measured*
   3.16 is a different quantity and should be compared to their step time, not
   their token rate.
8. **`--speculative-token-map` (FR-Spec) is free money if a map exists for our
   vocab.** FR-Spec profiles the EAGLE draft head at 128k vocab and finds the LM
   head alone is **49% of drafting time** (62% with softmax); restricting the
   *draft* to a 32k frequency-ranked subset (25% of vocab, 95% of occurrences)
   gives +1.12× over EAGLE-2 with the output distribution unchanged, because
   verification still runs the full vocabulary `[verified — A800, Llama-3-8B]`.
   GLM-class vocabularies are ≥150k. SGLang supports it for EAGLE-2 only
   (ignored for EAGLE3) `[verified — docs]`.
9. **Budget for the MoE verification tax before we raise depth at concurrency.**
   The expected number of distinct experts activated by T verified tokens is
   φ(T)·M with φ(T) = 1 − (1 − m/M)^T — a formula published and empirically
   validated (R² 0.830 → 0.976 on Qwen3-30B-A3B) in *An Interpretable Latency
   Model for Speculative Decoding in LLM Serving* `[verified]`. For GLM-5.2
   (M=256, m=8): 1 token → 8 experts, 4 → 30.5, 6 → 44.4, 8 → 57.4
   `[inferred, our arithmetic]`. At C1 our measured 2.18% overhead says this is
   *not* binding today (we are launch-bound, not weight-bound). At C32–C64 it
   will be. EVICT and D-cut are the published fixes.
10. **Complement, don't replace: NGRAM/suffix drafting for agentic traffic.**
    Our prefix caching is already worth 1.54×, which means our traffic is
    repetitive. SuffixDecoding (NeurIPS 2025 spotlight) reports up to 5.3× and
    2.8× over EAGLE-2/3 *on agentic workloads specifically* `[reported]`. SGLang
    ships `--speculative-algorithm NGRAM`, but it is CUDA-only, disables the
    overlap scheduler, and is incompatible with DP attention `[verified]` — so
    it is a separate deployment, not a knob.

---

## 1. The lossless core: speculative sampling and why it preserves the distribution

### 1.1 The two founding papers

| Title | Lab | Venue+Year | Hardware | Headline | In production? |
|---|---|---|---|---|---|
| Fast Inference from Transformers via Speculative Decoding | Google Research (Leviathan, Kalman, Matias) | ICML 2023 (arXiv 2211.17192) | single TPU-v4 | 3.4× on T5-XXL 11B EnDe at T=0; 2.6× at T=1 `[verified]` | yes — every engine |
| Accelerating Large Language Model Decoding with Speculative Sampling | DeepMind (Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper) | arXiv 2302.01318, Feb 2023 | 16 TPU-v4 | Chinchilla 70B: 14.1 → 7.52 ms/token (1.92×) XSum nucleus; 5.73 ms/token (2.46×) HumanEval `[verified]` | yes |

Both describe the same algorithm, independently. The draft model `q` proposes
γ tokens autoregressively; the target `p` scores all γ+1 prefixes in **one**
forward pass; tokens are accepted left-to-right.

### 1.2 The acceptance rule, exactly

For draft token `x` at position i, draw `r ~ U(0,1)` and

```
accept  iff  r < min(1, p_i(x) / q_i(x))
```

On the first rejection at position n, do **not** discard the step — resample the
replacement token from the residual distribution

```
p'(x) = norm( max(0, p_n(x) − q_n(x)) )
```

If all γ are accepted, sample one **bonus** token from `p_{γ+1}` directly. So a
step emits between 1 and γ+1 tokens and never zero.

**Why it is lossless.** Leviathan et al.'s Appendix A.1 decomposes P(output = x′)
into the accept path (probability `q(x′)·min(1, p(x′)/q(x′)) = min(p(x′), q(x′))`)
plus the reject-then-resample path, and shows the two sum to exactly `p(x′)`
`[verified]`. The guarantee is per-token and therefore per-sequence by
induction. Note the sharp consequence: **losslessness does not depend on the
draft being good.** A terrible draft costs you speed, never quality.

The acceptance rate is
`β = 1 − D_LK(p,q) = Σ_x min(p(x), q(x))`, i.e. one minus the total-variation
distance `[verified]`, and with `α = E[β]` constant across positions:

```
E[tokens per step] = (1 − α^(γ+1)) / (1 − α)
walltime improvement = (1 − α^(γ+1)) / ((1 − α)(γ·c + 1))
```

where `c` = (cost of one draft forward) / (cost of one target forward)
`[verified]`. That denominator is the whole optimization problem: **every extra
draft step costs `c` unconditionally and buys `α^γ` in expectation.**

Reported α values for reference (T5-XXL, EnDe, T=0): T5-small 0.75, T5-base
0.80, T5-large 0.82; a bigram model gets α ≈ 0.20 `[verified]`.

### 1.3 What SGLang actually runs

`python/sglang/kernels/ops/speculative/reject_sampling.py`,
`speculative_sampling_classic_kernel`, is the textbook algorithm verbatim
`[verified — read the Triton source]`:

```python
coin = UniformSamples[step-1]
if coin * q < p:            # i.e. coin < p/q
    accept; cur_prob_row = step
else:
    break
...
# final: sample from  max(0, p − q)  on rejection, or pure p if all accepted
val = p_val if all_drafts_accepted else max(p_val − q_val, 0)
```

Two things worth knowing:

- This chain kernel runs only under `--speculative-use-rejection-sampling`, which
  **requires `topk == 1`** `[verified — server_args.py]`. The default path uses
  the threshold kernel with `--speculative-accept-threshold-single` and
  `--speculative-accept-threshold-acc` (both default `1.0`). At the defaults it
  is the greedy/argmax match; lowering `threshold_acc` below 1.0 raises the
  accept probability from `p` to `min(1, p/threshold_acc)` and is **lossy**.
  Know which one you are running before you claim losslessness.
- The kernel has a NaN guard on `q` ("degenerate draft rows → residual falls back
  to p") and a degenerate-residual fallback that returns `VOCAB_SIZE − 1`. These
  are correctness bandaids for numerically sick draft distributions; if you see
  the last vocab id appearing in output, that's the tell.

---

## 2. Token trees, tree attention, and what `3-1-4` means

### 2.1 SpecInfer — the origin of tree verification

*SpecInfer: Accelerating Large Language Model Serving with Tree-based
Speculative Inference and Verification* — Miao, Oliaro, Zhang et al., CMU,
**ASPLOS 2024**, arXiv 2305.09781.

Contributions that everyone downstream inherited `[verified]`:

- **Token tree**: a candidate set organised as a tree, each root-to-leaf path a
  candidate continuation. Expansion is described by a vector ⟨k₁,…,k_m⟩ where m
  is the number of draft steps and kᵢ is the branching factor at step i. (`⟨2,2,1⟩`
  yields 4 sequences.) This is the direct ancestor of SGLang's
  `num_steps` / `eagle_topk` pair.
- **Topology-aware causal mask**: rather than launching one attention kernel per
  candidate, store all tree nodes' K/V in the cache in tree topology and rewrite
  the causal mask so each node attends only to its ancestors. One fused kernel.
- **Depth-first KV reuse**: the same KV cache is reused across all sequences in
  the tree via DFS traversal, instead of caching per-sequence.
- **Multi-step speculative sampling (MSS)**: extends rejection sampling to a tree
  losslessly. Theorem 4.2 proves distribution equality; Theorem 4.3 proves MSS
  has uniformly lower rejection probability than naive per-branch sampling, worth
  1.27× more verified tokens per step `[verified]`.

Results: 1.5–2.5× single-node multi-GPU, 2.4–2.8× multi-node, 1.2–1.5× over
sequence-based speculation, on **4×A10 24GB** with LLaMA-7B/OPT-13B/OPT-30B/
LLaMA-65B `[verified]`. Note the hardware: A10s, 2023. The *mechanism* generalises;
the speedups do not.

### 2.2 The mask, concretely, in SGLang

`build_tree_kernel_efficient` in `speculative/eagle_utils.py` `[verified]`.
Three mask modes:

| mode | shape | who uses it |
|---|---|---|
| `FULL_MASK` | `seq_lens_sum × num_verify_tokens + num_verify_tokens² × bs` booleans | GPU default |
| `QLEN_ONLY` | `num_verify_tokens² × bs` | CPU (intel_amx) verify kernel |
| `QLEN_ONLY_BITPACKING` | `num_verify_tokens × bs` packed into uint8/16/32 | packed backends |

The FULL_MASK shape is the one to watch: it is **linear in total context length ×
draft width × batch**. The source itself notes the fill can reach "up to 100s of
MB" per step and skips the memset when nothing reads the prefix columns. At
bs=256, 128k context, 6 draft tokens that is ~200 MB of memset per decode step —
a real reason deep+wide speculation collapses at high concurrency with long
contexts, entirely separate from FLOPs.

Alongside the mask the kernel emits `retrieve_index`, `retrieve_next_token`,
`retrieve_next_sibling` — the tree walked as a first-child/next-sibling linked
structure so the accept path can be traversed without recursion.

### 2.3 `3-1-4`: what the notation means

The shorthand is SGLang/SpecForge's `--config-list` format, documented as
`<batch-size>,<num-steps>,<topk>,<num-draft-tokens>` `[verified — SpecForge
`bench_eagle3.py` docs and the SpecBundle deployment widget]`. Dropping the batch
size, a serving config `S-K-D` is:

| flag | our value | meaning |
|---|---|---|
| `--speculative-num-steps` | **3** | depth of autoregressive drafting |
| `--speculative-eagle-topk` | **1** | branching factor per step |
| `--speculative-num-draft-tokens` | **4** | tokens submitted to the target per verify |

And the constraint SGLang applies: *"If `topk=1`, [`num_draft_tokens`] is adjusted
to `num_steps + 1`"* `[verified — docs]`. So at topk=1 the draft is a **chain**,
not a tree: `D = S + 1` always, and the third number carries no information.
`3-1-4` = 3 drafted tokens + 1 bonus slot = 4 verified positions.
`5-1-6` = 5 drafted + 1 bonus. `6-1-7`, `7-1-8` likewise.

Auto-defaults when unset: `num_steps` 5 for Llama/Grok, **3 for everything else**;
`eagle_topk` 4 for Llama/Grok, **1 for everything else**; `num_draft_tokens` 8 /
**4** `[verified — docs]`. Our `3-1-4` is literally the default. It was never tuned.

One subtlety in the implementation: `draft_forward` loops
`for i in range(num_steps)` and **breaks before the forward on the last
iteration** `[verified — eagle_worker_v2.py]`. The first draft token comes from
the *draft-extend* pass (which also refreshes the draft's KV with the accepted
tokens). So `S` steps = **S−1 draft-decode forwards + 1 draft-extend forward**,
producing S draft tokens. Going 3 → 5 adds exactly **two** draft-model forwards.

### 2.4 Static per-request width is why CUDA graphs work at all

`resolve_num_tokens_per_req` in `speculative/spec_utils.py` `[verified]`:

```
draft_decode   → width = speculative_eagle_topk        (= 1 for us)
draft_extend   → width = speculative_num_draft_tokens  (= 4 for us)
target_verify  → width = f(num_draft_tokens)
```

Every phase has a **compile-time-constant per-request token width**. That is the
whole trick: the number of tokens per request never varies at runtime, only the
number of requests does, and that is handled by bucketing + padding. After
verification the number of *accepted* tokens is ragged, but draft-extend still
runs at full `num_draft_tokens` width and masks — so the graph shape is stable.

The spec-decode capture bucket list is deliberately finer than the non-spec one
(`1..8` step 1, `10..32` step 2, `40..64` step 4, `72..256` step 8, vs
`1,2,4,8,12` then step 8 for non-spec) `[verified — server_args.py]`, because
each padded request costs `num_draft_tokens` wasted token-slots rather than 1.

---

## 3. Draft-head families

| Method | Lab | Venue+Year | Hardware | Headline (speedup / accept length τ) | Adopted? |
|---|---|---|---|---|---|
| Medusa | Cai, Li, Geng, Peng, Lee, Chen, Dao (Princeton/Together/UIUC/CMU) | ICML 2024, arXiv 2401.10774 | A100 | Vicuna-7B 2.18× (Medusa-1) / 2.83× (Medusa-2) `[verified]` | vLLM, TRT-LLM; largely superseded |
| Hydra | Ankner, Parthasarathy, Nrusimha, Rinard, Ragan-Kelley, Brandon | arXiv 2402.05109 (Feb 2024) | A100-40/80GB | Hydra++ 2.70×/2.50×/2.53× on Vicuna 7/13/33B; 1.27–1.31× over Medusa `[verified]` | no |
| EAGLE | Li, Wei, Zhang, Zhang (PKU/MSRA/Waterloo) | **ICML 2024**, arXiv 2401.15077 | A100-40G ×4, RTX3090 | Vicuna-13B 3.13×, τ=3.98; LLaMA2-70B 2.7–3.5×, τ=3.81 `[verified]` | vLLM, SGLang, TRT-LLM |
| EAGLE-2 | same | **EMNLP 2024**, arXiv 2406.16858 | as EAGLE | Vicuna-13B 4.26×, τ=4.83 `[verified]` | SGLang (`EAGLE`), vLLM |
| EAGLE-3 | same | **NeurIPS 2025**, arXiv 2503.01840 | H100 (SGLang), A100/3090 (vLLM) | Vicuna-13B 5.51×, τ=6.62 mean; LLaMA-3.3-70B 4.12×, τ=5.88 `[verified]` | SGLang (`EAGLE3`), vLLM |
| HASS | Zhang, Wang, Huang, Xu (Xiaohongshu) | **ICLR 2025**, arXiv 2408.15766 | not stated in abstract | 2.81–4.05× on LLaMA; +8–20% over EAGLE-2 `[reported]` | SpecForge training recipes |
| CORAL | Weng et al. | ACL 2025, arXiv 2502.16880 | — | cross-step representation alignment for multi-step draft training `[reported]` | no |
| FR-Spec | Zhao, Pan, Han et al. (Tsinghua) | arXiv 2502.14856 | A800 | Llama-3-8B: EAGLE-2 2.03× → 2.27× `[verified]` | SGLang `--speculative-token-map` |
| PARD | An, Bai, Liu, Li, Barsoum (AMD) | arXiv 2504.18583 | — | target-independent parallel draft `[reported]` | vLLM (`method: pard`) |

### 3.1 Medusa — and why it is not lossless

Architecture: K extra heads on the final hidden state, each one FFN layer with a
residual, `p_t^(k) = softmax(W₂^(k)(SiLU(W₁^(k) h_t) + h_t))`, initialised so
W₂ copies the original LM head and W₁ = 0 `[verified]`. Medusa-1 freezes the
backbone (5 hours on one A100 for Vicuna-7B); Medusa-2 fine-tunes jointly with
differential learning rates (heads 4× backbone) and a heads-warmup stage.

Candidates are the Cartesian product of the top-sₖ from each head, so the tree
has `Σₖ Πᵢ≤ₖ sᵢ` nodes. The tree is then greedily pruned using calibration
accuracies `a_k^(i)`, adding the node maximising `Π_j a_j^(i_j)` until the budget
is hit — the direct ancestor of every later tree-optimisation paper.

**The catch**: Medusa's default acceptance is *typical acceptance*, accepting any
candidate whose target probability exceeds `min(ε, δ·exp(−H(p)))` `[verified]`.
This is **not lossless** — the paper says so, describing it as selecting
"plausible candidates rather than using rejection sampling", trading exact
distribution matching for speed. Any comparison of Medusa's speedup against a
lossless method is apples-to-oranges. The heads are also *sequentially
independent* — head k does not see what head k−1 proposed — which caps accept
length around 2–2.5 in practice.

### 3.2 Hydra — sequential dependence, minimal change

Hydra's one idea, stated exactly `[verified]`:

```
Medusa:  p_draft(x̂_{t+i} | x_{≤t})                      = f_i(h_{t−1})
Hydra:   p_draft(x̂_{t+i} | x_{≤t}, x̂_{t+1},…,x̂_{t+i−1}) = f_i(h_{t−1}, x_t, x̂_{t+1},…,x̂_{t+i−1})
```

implemented by concatenating the base hidden state with the embeddings of the
already-speculated tokens along the feature dimension. Hydra++ adds: a 4-layer
MLP (5+ layers gave nothing), self-distillation against the base model's
*distribution* rather than the true token, and one extra self-attention "prefix"
decoder layer queried **once per decoding step** rather than per candidate
position. The paper's own comparison to EAGLE is honest: comparable throughput,
but EAGLE queries a full self-attention block per candidate position while
Hydra++ queries one per step `[verified]`.

Take-away: sequential dependence is worth ~0.46 accept length over independent
heads `[verified]`, and it is the reason every serious draft head since 2024 is
autoregressive.

### 3.3 EAGLE-1 — autoregress on features, disambiguate with the shifted token

The draft consumes **two aligned sequences** `[verified]`:

- `F_{1:i}` — hidden states from the target's **second-to-top** layer (pre-LM-head)
- `T_{2:i+1}` — token embeddings **shifted one step ahead**

concatenated, projected by an FC layer back to `hidden_dim`, run through **one**
transformer decoder layer, then through the *target's own frozen LM head*.

The shift is the whole insight. Predicting `f_{i+1}` from `f_i` alone is
ill-posed: "I am"/"I always" produce different feature trajectories from the same
`f_i`. Feeding the *realised* sampled token tells the draft which branch was
taken, lifting draft top-1 accuracy from ~0.6 (Medusa) to ~0.8 `[verified]`.

Loss: `L = L_reg + 0.1·L_cls`, smooth-L1 on features plus cross-entropy on the
induced token distribution, trained on 68k ShareGPT dialogues with uniform ±0.1
noise added to input features to simulate the error accumulation the draft will
face at inference `[verified]`.

Trainable parameters: 0.24B for a 7B target, 0.99B for 70B. Training: 1–2 days
on 4×A100-40G for the 70B head `[verified]`.

Position-wise acceptance for LLaMA2-Chat-70B: `0-α = 0.75`, `1-α = 0.69`,
`4-α = 0.64` `[verified]`. **That decay — roughly −0.025/step — is the single
most useful number in this whole survey**, because it is what governs the return
on depth. It is also mild: a 15% relative drop over four steps.

### 3.4 EAGLE-2 — the draft's own confidence is a calibrated acceptance predictor

Two phases per draft step `[verified]`:

- **Expand**: at each layer, take the top-k nodes by *global* value
  `V_i ≈ Π_{j ∈ ancestors(i)} c_j` (product of draft confidences root-to-node),
  and expand only those.
- **Rerank**: after expansion, rank *all* nodes by value and keep the top-m,
  which is guaranteed to be a connected tree because a node's value never exceeds
  its parent's.

The justification is an empirical calibration curve: draft tokens with confidence
< 0.05 are accepted ~4% of the time; tokens with confidence > 0.95 are accepted
~98% of the time `[verified]`. So the draft's softmax is a usable, free estimate
of acceptance probability — **no extra target call needed**. Every subsequent
budget-allocation paper (Sequoia, EVICT, D-cut, ECHO, SMART, vLLM's "adaptive
verification") is built on this observation.

Result: τ 3.98 → 4.83 on Vicuna-13B, 20–40% faster than EAGLE-1 `[verified]`.

### 3.5 EAGLE-3 — drop feature prediction, fuse three layers, train on your own mistakes

Three changes `[verified — read the HTML full text]`:

1. **Abandon feature prediction.** EAGLE's `l_fea` term "limits the expressiveness
   of the draft model and makes it difficult to benefit from increased data".
   EAGLE-3 trains on token cross-entropy only. Consequence stated in the paper:
   scaling draft training data now scales the speedup proportionally, which it
   did not for EAGLE.
2. **Multi-layer fusion.** Take low/mid/high hidden states `l, m, h` from the
   target, concatenate to 3k dims, FC down to k dims → `g`. This replaces
   top-layer-only conditioning. The stated reason: top-layer features are
   *inherently one-token-lookahead*, so predicting the next-next token from them
   is structurally hard.
3. **Training-time test.** Simulate multi-step drafting during training: step 1
   on ground truth with causal masking; step 2 feeds the draft's *own* step-1
   outputs back in with a modified (diagonal-except-against-original-data)
   attention mask; and so on. This is the same disease HASS names and cures
   independently.

Numbers (T=0, mean over MT-bench/HumanEval/GSM8K/Alpaca/CNN-DM) `[verified]`:

| target | EAGLE-2 | EAGLE-3 |
|---|---|---|
| Vicuna-13B | 4.22× (τ 4.83) | **5.51× (τ 6.62)** |
| LLaMA-3.1-8B-Instruct | 3.23× (τ 4.11) | **4.44× (τ 6.23)** |
| LLaMA-3.3-70B-Instruct | 2.85× (τ 3.78) | **4.12× (τ 5.88)** |
| DeepSeek-R1-Distill-LLaMA-8B | 3.26× (τ 3.92) | **4.16× (τ 5.84)** |

And the number our whole document should be measured against — **SGLang, 1×H100,
LLaMA-3.1-8B, MT-bench, batch 1** `[verified]`:

| | tokens/s |
|---|---|
| SGLang baseline | 158.34 |
| + EAGLE-2 | 244.10 |
| + EAGLE-3 | **373.25** |

Batch-size sweep on the same setup (throughput ratio vs no-spec) `[verified]`:

| BS | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 56 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE | 1.40 | 1.38 | 1.23 | 1.02 | **0.93** | 0.94 | 0.88 | 0.99 | 0.99 |
| EAGLE-3 | 1.81 | 1.82 | 1.62 | 1.48 | 1.39 | 1.32 | 1.38 | 1.34 | 1.38 |

EAGLE-1 goes **below 1.0 at BS 24** — speculation makes the server *slower*. This
is the canonical published evidence for the "disable under load" rule.

### 3.6 MTP: the draft head trained into the model

**Gloeckle et al., *Better & Faster Large Language Models via Multi-token
Prediction*, Meta, arXiv 2404.19737** — n *independent* output heads on a shared
trunk, as an auxiliary pre-training objective. Improves downstream capability,
increasingly so with model size, at no training-time cost `[reported]`.

**DeepSeek-V3 (arXiv 2412.19437)** deliberately differs `[verified]`:

- D sequential MTP modules, each = shared embedding + shared output head +
  per-depth transformer block `TRM_k` + projection `M_k` combining depth-(k−1)
  representation with the embedding of the (i+k)-th token.
- *"sequentially predicts additional tokens and keeps the complete causal chain
  at each prediction depth"* — explicitly contrasted with Gloeckle's independent
  parallel heads. This is architecturally an EAGLE head trained during
  pre-training.
- Loss `L_MTP = (λ/D)·Σ_k L_MTP^k`, λ = 0.3 for the first 10T tokens then 0.1 for
  the remaining 4.8T. **D = 1.**
- Ablation: HumanEval 44.5 → 53.7, GSM8K 72.3 → 74.0 on the 20.9B-activated MoE.
- And the number everyone quotes: *"the acceptance rate of the second token
  prediction ranges between 85% and 90% across various generation topics …
  delivering 1.8 times TPS"* `[verified — quoted verbatim from §5.4.3]`.

**GLM-4.5 (arXiv 2508.06471)**: *"we add an MoE layer as the MTP (Multi-Token
Prediction) layer to support speculative decoding during inference"* — 1 MTP
layer for both GLM-4.5 (355B/32B-active, 89 MoE layers) and GLM-4.5-Air; the
comparison table lists DeepSeek-V3 with 1 MTP layer and **Kimi K2 with 0**
`[verified]`.

Three consequences for us:

- Our draft head **is an MoE layer**, with the same 256-expert / top-8 routing
  and the same all-to-all as a target layer. That is why the draft path has its
  own `--speculative-moe-runner-backend` / `--speculative-moe-a2a-backend`
  flags — and why the draft MoE has its own crash surface (§8).
- A single MTP layer out of ~78 means `c ≈ 1/78 ≈ 1.3%` on weight-bandwidth
  grounds `[inferred]`; our measured total overhead of 2.18% for three draft
  forwards plus a 4-wide verify says the *real* `c` at C1 is even lower, because
  the step is launch-bound rather than bandwidth-bound.
- Kimi K2 shipping with zero MTP layers is why the community had to train an
  external EAGLE3 head for it (`AQ-MedAI/Kimi-K2-Instruct-eagle3`). Worth knowing
  before Kimi K3 lands.

---

## 4. Model-free drafting

| Method | Lab | Venue+Year | Hardware | Headline | Adopted? |
|---|---|---|---|---|---|
| Lookahead Decoding | Fu, Bailis, Stoica, Zhang (UCSD/Google/Berkeley) | **ICML 2024**, arXiv 2402.02057 | A100-80G; 8×A100-40G | 1.8× MT-bench LLaMA-2-7B; 2.3× CodeLLaMA code completion; 4× with Lookahead Parallelism on 8 GPUs `[verified]` | HF, some engines; niche |
| REST | He, Zhong, Cai, Lee, He (PKU/Princeton) | **NAACL 2024**, arXiv 2311.08252 | 1×A6000 | CodeLlama-7B HumanEval 2.36× greedy; Vicuna-7B MT-bench 1.69× `[verified]` | no |
| Prompt Lookup Decoding | Apoorv Saxena (independent) | GitHub, Nov 2023 | A100-40G, Mistral-7B | ~2.4× on summarisation and context-QA `[reported]` | HF `prompt_lookup_num_tokens`, vLLM `ngram` |
| SuffixDecoding | Oliaro, Jia, Campos, Qiao (CMU/Snowflake) | **NeurIPS 2025 (Spotlight)**, arXiv 2411.04975 | not stated in abstract | up to 5.3×; 2.8× over EAGLE-2/3 on agentic workloads `[reported]` | Snowflake ArcticInference; vLLM `suffix` |

**Lookahead** reframes decoding as Jacobi fixed-point iteration and keeps a 2D
window (W lookahead positions × N lookback steps) whose trajectory yields n-grams
for free; a verification branch checks up to G candidates from an n-gram pool.
The honest framing in the paper: it trades log(FLOPs) for linearly fewer decoding
steps, and *"requires surplus compute capacity; it underperforms in compute-bound
scenarios (large batch sizes)"* `[verified]`. For us at C1 that surplus exists;
at C64 it does not.

**REST** builds a suffix array over a corpus, retrieves the longest matching
suffix (greedy from n_max = 16 downwards), assembles retrieved continuations into
a Trie, keeps the top c = 64 highest-weight prefixes, and verifies with tree
attention. Retrieval overhead < 6% `[verified]`. The failure mode is obvious:
performance is a function of how well your corpus covers the traffic.

**Prompt lookup** is REST with the corpus replaced by the prompt itself: match
the last `max_ngram_size` (≈3) generated tokens against the prompt, and if found,
propose the next `num_pred_tokens` (≈10) tokens verbatim. Twenty lines of code,
2–4× on input-grounded tasks, zero on open generation.

**SuffixDecoding** is the version that matters for agentic serving: suffix trees
built over *previous outputs of the same deployment* plus the current prompt,
candidate trees scored by empirical frequency, and — importantly — an **adaptive
speculation length** that speculates far when the empirical continuation is
confident and barely at all when it is not.

**SGLang's `NGRAM`** implements this family in-engine: a trie with
`--speculative-ngram-max-trie-depth 18`, capacity 10M entries, BFS (recency) or
PROB (frequency) expansion, breadth 1–10, default 12 draft tokens `[verified]`.
Constraints that matter: CUDA-only, no `--enable-dp-attention`, and it **disables
the overlap scheduler and mixed chunked prefill**.

---

## 5. Self-speculation and layer skipping

| Method | Lab | Venue+Year | Idea | Verdict |
|---|---|---|---|---|
| Draft & Verify | Zhang, Wang, Li et al. (Zhejiang) | ACL 2024, arXiv 2309.08168 | skip intermediate layers to draft; verify with the full model; no extra params, no training | works, modest |
| LayerSkip | Elhoushi et al. (Meta) | ACL 2024, arXiv 2404.16710 | train with layer dropout + early-exit loss so early exits are accurate; self-speculate from an early exit | needs re-training |
| SWIFT | Xia, Li, Zhang, Du, Li (PolyU) | **ICLR 2025**, arXiv 2410.06916 | on-the-fly search for which layers to skip, per model/task, plug-and-play | training-free |

These matter when you cannot ship a draft head. We can — GLM-5.2 ships one. The
relevant negative is that layer-skipping drafts have `c` around 0.3–0.5 (you are
still running half the model), versus `c ≈ 0.013` for a 1-layer MTP head. From
Leviathan's `(1−α^(γ+1)) / ((1−α)(γc+1))`, a `c` that large caps γ at 1–2 no
matter how good α is. Not for us.

---

## 6. The 2026 frontier: block-diffusion drafters

This is the most active area right now, and it exists because of the term `γ·c`
in the denominator. If you can produce all γ draft tokens in **one** forward
pass, `γ·c` collapses to `c`, and depth becomes nearly free.

**DFlash: Block Diffusion for Flash Speculative Decoding** — Jian Chen, Yesheng
Liang, Zhijian Liu (z-lab), **ICML 2026**, arXiv 2602.06036. `[verified — read
the HTML]`

- Drafter: 5 layers (8 for Qwen3-Coder), block size 16 (10 for Llama-3.1).
  Target hidden features from **5 layers uniformly selected between layer 2 and
  the third-to-last layer**, projected once and shared by all draft layers; at
  each draft layer the draft tokens produce queries while both target features
  and draft tokens produce keys/values.
- All masked positions in a block are decoded **in parallel in one forward pass**.
- Results: Qwen3-8B **4.86× with accept length 6.49** at T=0, vs EAGLE-3 2.02×;
  reasoning (MATH-500, thinking on) 4.64×. Hardware: **H200** for the main
  tables, **a single B200 with the FlashAttention-4 backend** for the SGLang
  experiments.
- Absolute throughput in SGLang, Qwen3-8B: baseline 230 tok/s → **1175 tok/s** at
  concurrency 1; at concurrency 32, 5,694 → 16,076 tok/s. Speedups hold from 5.1×
  (C1) to 2.8× (C32) on math.
- SGLang ships it as `--speculative-algorithm DFLASH` with released checkpoints
  (e.g. `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`) `[verified]`.

The known weakness, and the entire 2026 follow-on literature: a block-diffusion
draft samples **per-position marginals, not a joint distribution** — draft token
i+1 is not conditioned on the sampled token i. The result is sequences that are
individually likely but jointly improbable, which a left-to-right verifier
rejects early. The fixes, all fetched and confirmed to exist:

| paper | arXiv | fix |
|---|---|---|
| DDTree — Ringel & Romano | 2604.12989 | build a draft *tree* from the per-position marginals; best-first expansion |
| CaDDTree — Zhang et al. | 2606.01813 | DDTree always favours larger trees; optimise **throughput** not accept length, giving a principled budget |
| Bastion — Oh et al. | 2605.29727 | query-dependent tree topology under a hardware budget; 6.61× |
| DominoTree — Lin & Jang | 2607.08642 | score paths with Domino's GRU causal correction (non-factorised); 6.6× |
| xPress — Wang et al. (IBM) | 2608.02438 | parallel refinement pass restoring causal dependencies; ~30% acceptance gain |
| DFlare — Zhang et al. | 2606.02091 | per-draft-layer fusion instead of one shared fused representation |
| HyperDFlash — Lin et al. | 2606.26744 | adapts block drafting to DeepSeek-V4's Hyper-Connections multi-path residual |
| DARTree — Li et al. | 2608.13524 | training-free AR correction head extended chain→tree; 12.97 tokens/round |
| JetSpec — Hu, Feng, … Zhang | 2606.18394 | causal parallel draft head over fused frozen-target hidden states: one-pass drafting **with** branch-wise causal conditioning. Up to 9.64× on MATH-500, 4.58× on chat, **H100**, Qwen3 dense+MoE `[verified]` |
| AngelSpec — Liu et al. | 2607.25852 | co-specialise: MTP for high-entropy chat, block-diffusion for code/math; runtime routing. 1.98–2.40× vs AR; **+10.5–11.8% throughput over DFlash**; +~30% accept length on Hy3-A21B `[reported]` |
| AdaFlash — Qian et al. | 2607.19223 | on-policy distilled diffusion drafters; identifies bidirectional attention as a variance source |

SGLang has already absorbed the direction: `DFLASH` and `DSPARK` are first-class
algorithms with their own workers, ragged-verify scheduler, cost tables
(`dspark_sps_profiler`) and confidence calibration (`dspark_sts_fit`)
`[verified — server_args.py]`.

**For us**: this is a 6–12 month bet, not a this-quarter action. It requires
training a new drafter for GLM-5.2 and there is no released checkpoint. But
JetSpec/AngelSpec are the shape of the thing that beats a 1-layer MTP head, and
AngelSpec's finding — that MTP wins on chat and block-diffusion wins on
code/math — matches our own workload split.

---

## 7. The optimization angle

### 7.1 What governs acceptance rate

Four factors, in decreasing order of effect size:

1. **Draft/target distributional distance.** `β = Σ_x min(p(x), q(x)) = 1 − TV(p,q)`
   `[verified — Leviathan]`. Everything else is a way of shrinking TV.
2. **Training/inference distribution mismatch.** The draft is trained on
   ground-truth prefixes but at inference consumes *its own* outputs. EAGLE-1
   patched this with input noise; HASS and EAGLE-3 solve it properly with
   simulated multi-step drafting. HASS reports 8–20% over EAGLE-2 from this alone
   `[reported]`.
3. **Task / register.** *Acceptance Dynamics Across Cognitive Domains in
   Speculative Decoding* (Saif Mahmoud, arXiv 2604.14682, TinyLlama-1.1B drafting
   Llama-2-7B-Chat-GPTQ, 99,768 speculative nodes from 200 prompts) finds **task
   type is a stronger predictor of acceptance than tree depth**, and — the
   counterintuitive result — chat has both the *highest entropy* and the
   *highest* acceptance, attributed to "the lexical predictability of
   RLHF-aligned register" `[verified]`. Entropy–acceptance correlation is
   "consistently negative but weak", ρ ∈ [−0.20, −0.15].
   Corroborated by the GLM-5.2 measurements in §7.2: gsm8k/math/aime accept
   ~4.5–4.6 while mtbench accepts 3.66.
4. **Depth.** EAGLE-1's LLaMA2-70B position-wise curve: 0-α 0.75 → 4-α 0.64
   `[verified]`. About −0.025 per step. Real but second-order compared to (1)–(3).

### 7.2 Real prose vs synthetic input: why we see 4.00/4 on random tokens and 3.16/4 on ShareGPT

Three mechanisms, and one of them has a primary-source confirmation.

**(a) Degenerate output inflates acceptance, and it is a known measurement
artifact.** SGLang PR [#29787](https://github.com/sgl-project/sglang/pull/29787),
in the middle of a GLM-5.2 accept-length benchmark, states plainly:

> *"a prior cross-framework '≈0.6 higher accept' observation on OpenHands turned
> out to be a **degenerate-output artifact** (target+draft drift into repeated
> tokens, trivially inflating accept). It is not used as a target here."*
> `[verified]`

Feed a model uniformly random token ids and it has nothing to condition on; it
falls into a low-entropy attractor — repetition, a single punctuation token, a
filler loop. `p` collapses toward a point mass, `TV(p,q) → 0`, and the MTP head
predicts it perfectly. **4.00/4 is not a good acceptance rate; it is a
degenerate-generation detector.**

**(b) `--dataset-name random` in SGLang's bench harness produces exactly that
input.** The GLM-5.2 benchmark cells are all measured with
`--random-range-ratio 1.0` on the `random` dataset `[verified]`. Random *input*
tokens, model-generated output — i.e. case (a).

**(c) Even where acceptance is genuinely measured, SGLang's published low-latency
spec cells do not measure it at all.** The cookbook note says: *"Spec cells pin
the EAGLE acceptance length via the serve env `SGLANG_SIMULATE_ACC_LEN`
(low-latency 5-1-6 = 3.5; FP8 balanced 1-1-2 = 2; NVFP4 balanced 2-1-3 = 2)"*
`[verified]`. Reading `generate_simulated_accept_index` in `spec_utils.py`
confirms what that does: it *forces* the accept count to a Bernoulli mix of
⌊3.5⌋ and ⌈3.5⌉ and, unless `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`,
fills the emitted tokens with the literal constant `100` `[verified]`.

So: **the widely-cited "GLM-5.2 NVFP4 on B200, `5-1-6`, ~540 tok/s" is TPOT
1.85 ms from `glm-5.2-benchmarks.jsx`, on synthetic input, with acceptance pinned
at 3.5** `[verified]`. It is a legitimate and useful number — it isolates engine
step time — but it is *not* 540 tok/s of real generation, and our 447.8 at a
measured 3.16 is not 17% behind it.

**The honest comparison.** Their pinned 3.5/6 over a 1.85 ms TPOT implies a cycle
time of 3.5 × 1.85 = **6.48 ms** at `5-1-6` on NVFP4. Our measured cycle at
`3-1-4` on FP8-class weights is **7.06 ms**. If we reach their cycle time at
`5-1-6` *and* our real accept length on ShareGPT lands at 3.9–4.1, we would be at
601–632 tok/s. If we keep our current cycle time and simply add two draft steps
(+3–4%), we land at ~515–560 tok/s. Both beat the TileRT 500 tok/s reference.

**Measurement rule for us, going forward**: report accept length from
`generation_tokens_total / spec_verify_calls_total` on `/metrics` over *real*
traffic (this is exactly what PR #29787 used), never from a `random`-dataset run,
and refuse any number produced with `SGLANG_SIMULATE_ACC_LEN` set.

### 7.3 Depth vs verification cost, and where the optimum is

The governing expression is Leviathan's, generalised by Sequoia to a
hardware-measured verify cost:

```
Speedup(n, d) = G(n, d) / ( t(n) + d·c )
```

where `G(n,d)` = expected tokens from a tree of `n` nodes and depth `d`, `t(n)` =
**measured** target-forward time for `n` verified tokens on *this* GPU, and `c` =
relative draft cost `[verified — Sequoia, arXiv 2402.12374]`. Sequoia's key
systems point: because expected tokens grow only logarithmically in budget, the
optimum sits **before** verification time becomes negligible, so you must measure
`t(n)` rather than assume it constant — worth up to 40% over a fixed tree size
`[verified]`.

Sequoia also contributes the exact optimal-topology DP under a positional-
acceptance assumption:
`c(n) = max over child allocations { 1 + Σᵢ pᵢ·c(aᵢ) }` with `Σaᵢ = n−1`, solved
in `O(n²k)` `[verified]`. And a sampling-without-replacement scheme that
satisfies **both** the optimal-transport property (good at high temperature) and
the cover property (good at low temperature), where SpecInfer/SpecTr satisfy only
the former and top-k only the latter — worth 1.65× over SpecInfer and 1.27× over
top-k across temperatures `[verified]`. Results: Llama2-7B 4.04× (5.08 tok/step)
on **A100**; Llama2-70B **9.96×** (12.18 tok/step) in an *offloading* regime on
**L40**, where `t(n)` is essentially flat because everything is PCIe-bound.

**SMART** (Wang & Zhou, arXiv 2604.09731, ECCV 2026) turns this into an online
rule: expand a node **iff its marginal benefit–cost ratio exceeds the current
tree-level speedup** — a clean marginal-analysis stopping criterion needing only
the verification-cost slope and the marginal acceptance probability. Reported as
a plug-and-play controller giving +15.4% on Llama-3.1/DeepSeek-R1 in
compute-bound batching regimes `[reported]`.

**SpecDec++** (Huang, Guo, Wang, Princeton, **COLM 2025**, arXiv 2405.19715)
proves the MDP formulation of "when to stop drafting" has a **threshold-policy
optimum**, and trains an acceptance-prediction head on the draft to implement it:
stop when predicted rejection probability exceeds the threshold. 2.04–2.26× on
Llama-2-Chat 7B/70B, +7–11% over fixed candidate lengths `[reported]`.

**Applied to us**, with our measured constants:

- `t(4) + 3c = 7.057 ms`, `t(1) = 6.906 ms` ⇒ everything speculation adds today
  is **0.151 ms**, 2.18%.
- Each additional draft step is one MTP-layer forward + one LM-head projection.
  Bandwidth floor: 1/78 of the target's weight stream plus a ~150k×5120 fp8 LM
  head sharded 8 ways ⇒ ~0.09 + ~0.03 = **~0.12 ms** `[inferred]`. Our measured
  total is smaller than 3× that, which means we are not bandwidth-bound — the
  launch/collective-skew term dominates and the draft forwards partly hide in it.
- So `3-1-4 → 5-1-6` costs **+0.15 to +0.30 ms** on a 7.06 ms cycle = **+2 to
  +4%** `[inferred]`, against an accept-length gain of **+23 to +30%**.
- `5-1-6 → 7-1-8` costs another +2 to +4% against a gain of ~+8 to +10%
  (E(k=7) ≈ 4.2–4.8 from the same fit). Still positive, but this is where it
  starts to close, and where the MoE expert-union term (§7.5) starts to bite.

**Recommendation: test `5-1-6`, `6-1-7`, `7-1-8` and stop where measured
tok/s stops rising.** Do not extrapolate past `7-1-8` — issue #32666 shows a
depth-gated silent corruption at 5–6 draft tokens on a different model, and PR
#33872 shows a `kMaxMTPDraftTokens = 4` constant that silently corrupted DSv4
compress state above 4 draft tokens. Depth is a correctness surface, not just a
performance knob.

### 7.4 Why speculative decoding degrades at concurrency, and what to do

The mechanism is unanimous across the literature: as batch grows, the target
forward moves from memory-bound to compute-bound, so verifying `k+1` tokens per
request stops being free and starts costing ~`k+1`× the FLOPs — while the accept
length does not change. `t(n)` goes from flat to linear.

| paper | model of the effect | headline |
|---|---|---|
| Su, Giannoula, Pekhimenko, *The Synergy of Speculative Decoding and Batching* (arXiv 2310.18813) | `t ≈ N(α_b·s + β + t_S(b,1)·s)/(c·s^γ + 1)`; optimal `s` solves `K·α_b·s^γ − L·s^(γ−1) + α_b = 0`. Since `α_b` grows with batch, `s_opt` shrinks. | Optimal speculation length: **6–8 at BS 1, 3–4 at BS 8, ≤2 at BS 32** (OPT-1.3B/6.7B/Llama-7B on RTX3090/4090/A100). Adaptive strategy: 2.73× at BS 1, 1.31× at BS 32, +9% vs fixed under time-varying traffic `[verified]` |
| Kong, Flynn, Peng, Shavit, Kurtz, Marques, *An Interpretable Latency Model for Speculative Decoding in LLM Serving* (arXiv 2605.15051) | `L = [c₁ₚ + (g/E)(c₁ᵥ + k·c₁d)] / [1 − RPS·(c₂ₚ + (g/E)(c₂ᵥ + k·c₂d))]`, `E = (1−α^(k+1))/(1−α)`, batch from Little's Law | Speedup rises with load iff `C₂ᵣ < 1`, and `C₂ᵣ > 1` is the common case. **The `k` minimising the load-independent term is much larger than the `k` minimising the load-dependent term** — so "draft lengths optimized for batch size 1 are generally suboptimal for high throughput". R² ≈ 0.99, **A100 SXM** primary, H100 check; Llama-3.1 8B/70B, Qwen3 0.6B–235B, gpt-oss-20b `[verified]` |
| EAGLE-3 §Table 3 | measurement, not model | EAGLE-1 throughput ratio drops **below 1.0 at BS 24** on H100 `[verified]` |
| Nightjar (Li et al., arXiv 2512.22420) | multi-armed-bandit planner | picks speculative length per batch size, **disables speculation** when the MAB says it is unprofitable and offloads the draft model to CPU to reclaim KV memory. +14.76% throughput, −20.18% latency `[reported]` |
| ECHO (Hu et al., arXiv 2604.09603) | budgeted scheduling | batched requests managed as one super-tree with sparse confidence gating; "elastic pivoting budget between depth and width". 5.35× peak, >20% over SOTA, Qwen3-235B, **integrated into SGLang** `[reported]` |
| D-cut (Liu et al., arXiv 2607.14647) | cross-request budget | selects draft tokens **jointly across the batch**, concentrating verification budget on the tokens most likely to be accepted; runtime cost model per GPU/parallelism. 1.26× → **1.65×** under high concurrency on dense; up to 3.0× on MoE `[reported]` |
| FlexDraft (Zhang et al., arXiv 2605.20022) | mode switching | parallel draft+verify at small batch, sequential at large batch; verification length from draft confidence `[reported]` |

**What SGLang already implements.** `--speculative-adaptive` +
`--speculative-adaptive-config`. The policy `[verified — `adaptive_spec_params.py`
and `adaptive_speculative_decoding.mdx`]`:

- Independent EMA per batch-size slot (α = 0.2, `warmup_batches` 10,
  `update_interval` 5), so BS-1 observations never pollute the BS-64 signal.
- `target_steps ≈ clamp(round(ema_accept_len) + 1, min, max)` — "probe one step
  beyond observed acceptance".
- Hysteresis both ways (`down_hysteresis −0.25`, `up_hysteresis 0.0`) and an
  optional `ceiling_coeff` that caps steps downward only, never blocking
  exploration upward.
- Each candidate step count owns **pre-captured** CUDA graphs and attention
  backends, so a tier switch is a pointer swap, never a recapture. Switches
  happen only between rounds.
- Built-in ladder: `{"1": [1,3,7], "8": [0,1,3], "32": [0,1], "64": [0]}`. Note
  **`0` means speculation off** at BS ≥ 64.
- Restrictions: EAGLE/EAGLE3 only, `topk == 1` only, no DP attention (tier
  decisions are not synchronised across DP ranks), no multi-layer EAGLE, no TBO,
  no PDMux.

And the originating PR ([#22246](https://github.com/sgl-project/sglang/pull/22246))
supplies the model-specific evidence `[reported]`: *"Qwen3-235B-A22B performs
best with step=4 at BS=1 but step=2 at BS=64; GLM-4.7-FP8 performs best with
step=7 at BS=1 but step=3 at BS=64"*, and +3.4% (Qwen) / **+6.2% (GLM)** average
over a fixed `num_steps=3`, up to +12.8% at individual batch sizes.

**GLM-4.7 wanting step=7 at BS=1** is a strong prior that GLM-5.2 also wants far
more than 3.

### 7.5 The MoE-specific story — and it cuts both ways

**MoE benefits *more* than dense at medium batch.** *MoESD: Unveil Speculative
Decoding's Potential for Accelerating Sparse MoE* — Huang, Zhu, Zhan et al.,
**NeurIPS 2025 (spotlight)**, arXiv 2505.19645 `[verified]`:

- Roofline argument: above a threshold batch size all experts activate but each
  is *underutilised*, so the MoE FFN is memory-bound with low arithmetic
  intensity. Verifying extra draft tokens adds compute without proportionally
  adding weight loads — "leveraging spare resources without notably increasing
  processing time". A dense model in the same regime is already compute-bound.
- Average tokens per expert `T_exp(t, ρ) = ρt / (1 − (1−ρ)^t)`; sparser ⇒ more
  memory-bound ⇒ **wider** batch-size range where SD helps. Full-activation
  threshold `T_thres = ⌈log_{1−ρ}(1−τ)⌉`.
- They introduce **target efficiency** `= T_T(B,1) / T_T(B,γ)` — the ratio of
  single-token to multi-token target forward time — as the system-side factor
  that is invisible to acceptance-rate research. Two systems with identical
  acceptance can have wildly different speedups.
- Measured up to **2.29× for Qwen2-57B-A14B at medium batch sizes**; peak
  speedup typically at batches 12–32 for K=8.

For GLM-5.2, ρ = 8/256 = 0.03125, so `T_thres` at τ = 0.99 is
`log(0.01)/log(0.96875) ≈ 145` tokens — we are nowhere near full expert
activation at C1–C16 `[inferred]`. That is MoESD's regime, and it says our
speculation should keep paying well past the batch sizes where a dense model
would give up.

**But wider trees cost more on MoE than on dense.** *Making Every Verified Token
Count: Adaptive Verification for MoE Speculative Decoding* (EVICT) — Pan, Tao,
Pang, Wang, Zhao, Zhang, arXiv 2605.00342 `[verified]`: as a draft tree expands,
different branches route to different experts, so the **union** of activated
experts grows and target-side verification cost grows with it. EVICT truncates
the draft tree before verification, keeping only cost-effective prefixes, using
fine-grained drafter signals for benefit and offline-profiled verification costs
for cost. Training-free, hyperparameter-free, lossless; up to 2.35× vs
autoregressive and **1.21× average over EAGLE-3**; "highly compatible with …
SGLang".

The quantitative form is published in the interpretable-latency-model paper as
`φ(T) = 1 − (1 − m/M)^T`, which lifted MoE fit quality from R² 0.830 to 0.976 on
Qwen3-30B-A3B `[verified]`. For GLM-5.2 (M = 256, m = 8) `[inferred, our
arithmetic]`:

| verified tokens T | expected distinct experts | vs T=1 |
|---|---|---|
| 1 | 8.0 | 1.00× |
| 2 | 15.8 | 1.97× |
| **4** (our `3-1-4`) | **30.5** | 3.82× |
| **6** (`5-1-6`) | **44.4** | 5.55× |
| 8 (`7-1-8`) | 57.4 | 7.18× |
| 16 | 102.0 | 12.75× |

If MoE weight traffic scaled with that union, going 1 → 4 tokens would add
2.82 × 19.4% = **+55%** to our step time. Our measured overhead is **+2.18%**.
The model is off by more than an order of magnitude for us. The reconciliation
`[inferred]`: at C1 with TP8, our decode step is not bandwidth-bound at all —
19.6% of it is collectives and 47% of *that* is rank-arrival skew, i.e. idle
waiting. Extra verify tokens ride inside slack that already exists. **The union
tax is real but it is currently hidden by our sync overhead; it will surface as a
first-order cost the moment the batch is large enough to make the MoE GEMMs
genuinely bandwidth-bound.** That is the point at which EVICT / D-cut style
budget truncation stops being academic for us.

Related MoE-specific work found and confirmed to exist (not read in depth):
AcceptMoE (arXiv 2608.02989, commitment-weighted verifier expert sets, 2.06×
under expert offloading at batch 1, served with SGLang), SpecMoE (arXiv
2604.10152, self-speculation, up to 4.30×), ELMoE-3D (arXiv 2604.14626,
hybrid-bonding hardware co-design, 6.6× across BS 1–16).

### 7.6 Systems interactions

**CUDA graphs.** Not a problem in the way people assume: as shown in §2.4, the
per-request token width is *static* in every phase, so graphs capture cleanly.
What varies is batch size, handled by bucketing. Three real hazards:

- Anything not capture-safe inside the draft loop breaks capture. Issue
  [#32105](https://github.com/sgl-project/sglang/issues/32105) is the exemplar:
  `--enable-expert-distribution-metrics` performs a rank-0 `.item()` GPU→CPU sync
  in its per-forward-pass hook; the draft graph capture drives a **full
  `ModelRunner.forward` per draft step**, so `num_steps >= 2` puts that sync
  inside `torch.cuda.graph` and the server crash-loops with
  `cudaErrorStreamCaptureUnsupported`. `num_steps = 1` starts fine `[verified]`.
  **Any depth increase enlarges the surface of "things that must be
  capture-safe".** That is an independent reason our `3-1-4` limit is not
  necessarily a memory bug.
- Draft-extend graphs are memory-hungry. SGLang added
  `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` because the draft-extend graph pool
  cost ~4.5 GB and OOM'd a DSv4 test `[verified — PR #30944]`. The PR also notes
  the capture decision is duplicated across `eagle_worker_v2.py` and
  `multi_layer_eagle_worker_v2.py` with hardcoded platform checks, and that
  draft-extend graphs live outside the per-phase `cuda_graph_config` framework
  and borrow the decode phase's bs list. That is fragile code.
- Replay-time OOB inside the captured graph. Issue
  [#28569](https://github.com/sgl-project/sglang/issues/28569): EAGLE3 draft
  graph replay IMA on gpt-oss-120b, deterministically once the running batch
  shrinks from 32 to ~12. All graph *inputs* asserted in bounds; `--disable-cuda-graph`
  does not reproduce. Diagnosis: OOB in the KV indexing built for the padded
  bucket shape.

**MoE routing.** The draft's MoE has its own backend selection
(`--speculative-moe-runner-backend`, `--speculative-moe-a2a-backend`), its own
default resolution path, and — for us — its own crash (§8).

**Chunked prefill.** SGLang **forbids mixed chunked prefill under speculation**:
`assert not self.enable_mixed_chunk` when `speculative_algorithm is not None`
`[verified — server_args.py]`. So prefill chunks and spec-decode batches strictly
alternate; a prefill chunk cannot be co-scheduled into a verify batch. Practical
effect: TTFT work fully preempts decode, and the spec cycle's fixed overhead is
paid separately from prefill. Also note the GLM-5.2 cookbook's measured result
that raising `--chunked-prefill-size` from 2048 to 32768 gave **+34–78% output
throughput and −39–59% TTFT** at 8K input on 8×H200/8×B200 for the balanced
recipe `[reported]` — a bigger lever than most spec tuning at that operating
point.

**Paged KV.** Two things. (1) The draft has its **own** KV pool, allocated with
one slot per target token because draft and target share a slot index space; the
docs warn a 5-layer DFLASH draft costs 10240 bytes/token in bf16 and that
`--speculative-draft-kv-cache-dtype fp8_e4m3` halves it `[verified]`. (2) The
draft-decode KV index buffer must be sized `num_seqs × topk × max_context_len`,
with the source noting *"the topk factor is mandatory — dropping it
under-allocates and overflows the row (#27338, #27460)"* and an assert that
`num_seqs × topk × max_context_len < 2³¹` `[verified — spec_utils.py]`. At
topk=1 that is one index per branch per step, which is another reason chain
drafting is systemically safer than tree drafting at long context.

**DSA indexer (specific to GLM-5.2 / DeepSeek-V3.2 / V4).** GLM-5.2's config
exposes `index_share_for_mtp_iteration`, which **reuses the DSA indexer's top-k
across draft steps instead of recomputing it**, and is "effective only at
`--speculative-eagle-topk 1`" `[verified — cookbook]`. PR #29787 moved the
anchor for that shared top-k from the first draft-decode step to the
**draft-extend** step, so the seed comes from the last *verified* (target-derived)
hidden state. Measured effect on accept length, GLM-5.2-NVFP4 TP4 `5-1-6` temp 0
`[verified]`: +0.061 on `openhands_longctx`, within noise on short-context sets —
"larger as context grows". Our indexer is 5.8% of the C1 profile; sharing removes
it from S−1 of the S draft steps. Confirm it is on.

---

## 8. Our specific blocker: the `3-1-4` ceiling

Three GitHub threads describe our exact configuration. Reading them together
gives a concrete first move.

**Issue [#30209](https://github.com/sgl-project/sglang/issues/30209)** (open,
2026-07-06) — `nvidia/GLM-5.2-NVFP4`, `GlmMoeDsaForCausalLM`, TP4/TP8, **B200 &
B300**, `modelopt_fp4`, kv `fp8_e4m3`, EAGLE `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`.
Crashes with `CUDA error: an illegal memory access` under real production traffic
(~10 min per replica at ~1 req/s). Disabling speculation → zero crashes.
Exception-state coredump names the faulting kernel `[verified]`:

```
Bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128u2_..._dynB_sm100f   grid (48,8,1)
```

— a flashinfer TRTLLM batched-GEMM cubin used by the **flashinfer_trtllm MoE
runner for the bf16 nextn/EAGLE draft layer** of the FP4 checkpoint, sibling of
[flashinfer#3722](https://github.com/flashinfer-ai/flashinfer/issues/3722)
(scale-factor buffer under-padded for the swizzled layout → TMA descriptor
computes an OOB address → Xid 13 MMU fault, B200/SM100 only, input-data
dependent, does not reproduce with random weights) `[verified — read both]`.

Fix, A/B-validated on synchronized-mirrored production traffic, 8×B200 per arm:
`--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none`.
Control crashed 5× in 5h; treatment 0 restarts in 5h, then 24h+ crash-free with
**accept length 3.19, accept rate 0.44, 2.5–2.8× TPOT improvement vs no-spec on
real data** `[verified — quoted from the issue]`.

**The config gap is still open.** `_deepseek_spec_moe_resolution` in
`python/sglang/srt/arg_groups/overrides.py` on `main` today:

```python
if model_arch not in _DEEPSEEK_FAMILY_ARCHS:   # includes GlmMoeDsaForCausalLM
    return {}
if not is_hip():                                # <-- CUDA users get nothing
    return {}
...
return {"speculative_moe_runner_backend": "triton",
        "speculative_moe_a2a_backend": "none"}
```

and further down, `_speculative_moe_runner_default` falls the draft MoE through
to `moe_runner_backend` — `flashinfer_trtllm` on Blackwell. PR
[#30210](https://github.com/sgl-project/sglang/pull/30210), which removes the two
`is_hip()` lines, is **closed and unmerged** (`merged: false`, closed
2026-07-06) `[verified — GitHub API]`.

**Issue [#31093](https://github.com/sgl-project/sglang/issues/31093)** (open,
2026-07-14) — same model/hardware at `6-1-7`, IMA **during decode CUDA-graph
capture**, on `v0.5.15` and `main@b94ac87e`, but *not* on the older
`dev-glm52-nvfp4` image from 2026-07-03. The async IMA surfaces at an RNG-state
sync under dynamo compile and is reported as `Capture cuda graph failed`; rank
and progress vary run to run, consistent with an asynchronous fault. Notably the
reporter also has to disable the multimem all-gather in `logits_processor.py`,
"because with it enabled even the 07-03 image hits a draft-CUDA-graph IMA on this
model" `[verified]`.

**Diagnosis and ordered plan** `[inferred, but each step cites a verified
source]`:

1. **Set `--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none`
   and retry `5-1-6`.** Highest prior: identical model, identical GPU, identical
   spec config, crashing kernel family identified by coredump, fix validated in
   production, and the protective default provably absent on CUDA. Zero risk —
   worst case the crash persists and you've eliminated the leading hypothesis.
2. If capture still fails, **bisect the capture surface**, not the depth. Set
   `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` (PR #30944) to run draft-extend
   eager; if that fixes it, the fault is in the draft-extend graph, whose bs list
   is borrowed from the decode phase and whose pool can cost GBs.
3. **Audit for non-capture-safe hooks.** Per #32105, anything doing a device→host
   sync inside `ModelRunner.forward` (expert-distribution metrics, custom
   profiling, invariant checks) becomes fatal at `num_steps >= 2` and *more*
   likely to fire as depth grows. Check `--enable-expert-distribution-metrics`
   and `SGLANG_INVARIANT_CHECK`.
4. **Run capture with `CUDA_LAUNCH_BLOCKING=1`** to attribute the async IMA to a
   real kernel rather than to the RNG sync point — this is what turned #28569
   from "crash in `resolve_seq_lens_cpu`" into "crash in draft graph replay".
5. **Check the draft KV index buffer arithmetic.** `draft_kv_indices_buffer_width
   = num_seqs × topk × max_context_len` with a hard `< 2³¹` assert. At topk=1
   and 1M context, `num_seqs × 1 × 1,048,576 < 2³¹` caps `num_seqs` at 2047 —
   fine — but `out_cache_loc` is allocated as
   `max_num_token × speculative_num_steps`, which grows linearly in depth
   `[verified — eagle_draft_cuda_graph_runner.py]`. If capture OOMs rather than
   IMAs at depth, that's the term.
6. **Treat depth ≥ 5 as a correctness risk until proven otherwise.** Two
   independent precedents: issue #32666 (Frozen-KV MTP on Gemma-4-31B-NVFP4,
   single B200 — 23/24 outputs corrupt at `num_steps 5`, clean at 3, and
   `num_steps 4` **fails verify-graph capture with an IMA**; explicitly *not* a
   batch-size effect, since depth 3 at concurrency 6 is clean while depth 5 at
   concurrency 4 corrupts at the same 24 tokens/verify-batch) `[verified]`; and
   PR #33872 (merged) which found `kMaxMTPDraftTokens = 4` hardcoded in DSv4's
   `c_plan.cuh`, silently under-writing the compress-state ring for
   `num_draft_tokens > 4` — *"no illegal memory access, no NaN, no assertion …
   simply computed from stale slots, which shows up only as a quiet accuracy
   regression"* `[verified]`. **Run an accuracy gate (gsm8k / humaneval / a
   long-context set) at every new depth, not just a smoke test.**

---

## 9. Training a better draft head

The single highest-leverage non-systems change. Ordered by evidence quality.

1. **Regenerate the training data with the target model.** SpecBundle — the
   LMSYS/SpecForge + Ant Group + Meituan + Nex-AGI + EigenAI initiative — ships a
   `PerfectBlend-Regenerated-<target>` dataset alongside every released EAGLE3
   draft `[verified — SpecForge docs]`. Their stated diagnosis of why public
   drafts underperform: *"Most available draft models are trained on small or
   curated datasets and fail to generalize to the large, diverse corpora used in
   modern LLM training, resulting in low token acceptance rates."* Distilling on
   the target's own outputs directly minimises the TV distance that `β` measures.
2. **Train with the draft's own predictions in the loop.** EAGLE-3's
   training-time test (§3.5) and HASS's harmonized context alignment are the same
   idea arrived at independently; HASS also reweights the loss toward the
   target's top-ranked tokens rather than the true next token, aligning the
   training objective with the acceptance objective `[reported]`. CORAL (ACL
   2025) adds cross-step representation alignment for the same reason. 2026 work
   pushes further: Draft-OPD (arXiv 2605.29343) does on-policy distillation
   focused on verification-exposed errors; PARD-2 (arXiv 2605.08632) reformulates
   the objective from token accuracy to **acceptance length** directly.
3. **Specialise by domain.** TAPS (arXiv 2603.27027) trains HASS and EAGLE-2
   drafters on MathInstruct vs ShareGPT vs mixed and shows clear specialisation
   by acceptance length; AngelSpec (arXiv 2607.25852) productionises this by
   co-specialising an MTP drafter for chat and a block-diffusion drafter for
   code/math with runtime routing `[reported]`. Given our traffic is a mix, a
   single head is leaving accept length on the table.
4. **Compress the draft's vocabulary, not the target's.** FR-Spec (§Bottom line
   #8). Draft on a 32k frequency-ranked subset; verify on the full vocabulary;
   distribution unchanged.
5. **Published accept lengths for large MoE targets — the realistic bar.**

| target | drafter | config | dataset | accept len | source |
|---|---|---|---|---|---|
| GLM-5.2-NVFP4 (256E/top-8, DSA) | bundled MTP | 5-1-6, T=0, TP4 | mtbench | **3.66** | SGLang PR #29787 `[verified]` |
| " | " | " | humaneval | **4.33** | " |
| " | " | " | gsm8k | **4.49** | " |
| " | " | " | math500 | **4.61** | " |
| " | " | " | aime | **4.63** | " |
| " | " | " | openhands-longctx | **5.18** | " |
| GLM-5.2-NVFP4 | bundled MTP | 5-1-6, TP8/B200 | real prod traffic | **3.19** (accept rate 0.44) | SGLang #30209 `[verified]` |
| DeepSeek-V3 (671B/37B-act) | MTP D=1 | 1 extra token | mixed | **85–90%** on token 2 → 1.8× TPS | DeepSeek-V3 report `[verified]` |
| Mixtral-8x7B-Instruct | EAGLE-1 | tree | MT-bench | τ **3.25**, 1.50× | EAGLE-1 `[verified]` |
| Qwen2-57B-A14B | Qwen2-0.5B | γ=4 | HumanEval | 2.18× at B=1, 1.96× at B=16 | MoESD `[verified]` |
| LLaMA-3.3-70B (dense) | EAGLE-3 | tree | mean of 5 | τ **5.88**, 4.12× | EAGLE-3 `[verified]` |

Note the spread within one model: **3.66 on chat vs 5.18 on long-context
agentic**, a 41% difference on identical weights and identical config. Any single
accept-length number for a model is meaningless without the workload.

Also note our **3.16 at `3-1-4`** and their **3.19 at `5-1-6` on production
traffic**. Two very different configs landing at the same accept length is a
strong hint that *their* production traffic is harder than *their* mtbench run —
which is the normal state of affairs and the reason to measure on our own
traffic.

---

## 10. What is NOT worth it

**Medusa-style independent heads.** Superseded on both axes: lower accept length
(no sequential dependence — Hydra's whole paper) and lossy by default (typical
acceptance). If you see a Medusa speedup quoted against a lossless baseline,
discount it. Still shipped in vLLM/TRT-LLM for compatibility; nobody should start
here in 2026.

**Lookahead decoding at concurrency.** The paper is explicit that it needs
surplus compute and underperforms in compute-bound regimes. It also produces no
distribution guarantee beyond exact-match verification, and its speedups (1.5–2.3×
on A100 LLaMA-2-7B, ~30% on a 3090) are well below what a trained draft head
gets. Its one genuinely interesting result — 4× with "Lookahead Parallelism" on
8×A100 — is a multi-GPU trick that does not compose with TP8 serving.

**REST-style external-corpus retrieval.** The datastore is a coverage bet on
traffic you have not seen. SuffixDecoding dominates it for the case where the
bet pays (repetitive agentic traffic) by building the corpus from your *own*
recent outputs, adaptively, at zero curation cost.

**Layer-skipping self-speculation for us.** `c ≈ 0.3–0.5` versus `c ≈ 0.013` for
a 1-layer MTP head. From the walltime formula, that alone caps γ at 1–2. It is
the right answer only when you cannot ship a draft head at all.

**`topk > 1` in our stack, today.** Not because trees are bad — EAGLE-2 shows
+20% accept length and Sequoia shows exact optimal topologies — but because in
current SGLang a tree costs you the overlap scheduler, adaptive depth, and DSA
index sharing simultaneously, and adds the `num_seqs × topk × max_context_len` KV
index buffer and the `seq_lens_sum × num_draft_tokens` mask memset. Revisit if
and when spec-v2 overlap supports topk > 1.

**Chasing the published 540 tok/s as a target number.** It is a pinned-acceptance
measurement on synthetic input (§7.2). The correct target is *their cycle time*
(6.48 ms at `5-1-6`) against *our* real accept length.

**`SGLANG_SIMULATE_ACC_LEN` in any number that leaves the team.** It is a
perfectly good engine-microbenchmark tool. It is not an acceptance measurement,
and in default `fixed` token mode it emits token id `100` for every accepted
position — the output is garbage by construction.

**Lowering `--speculative-accept-threshold-acc` below 1.0 to "improve" accept
length.** It raises acceptance from `p` to `min(1, p/threshold)` and is exactly
Medusa's typical-acceptance trade under a different name. If you do it, you no
longer have a lossless system and every quality number needs re-measuring.

**Deeper-is-always-better past the accuracy gate.** Two verified precedents of
depth causing *silent* wrongness (issue #32666 output corruption gated on depth;
PR #33872 `kMaxMTPDraftTokens = 4` silent KV corruption above 4 draft tokens). A
throughput win with an unmeasured accuracy regression is not a win.

---

## Sources

Papers (all fetched; arXiv IDs verified against the abstract page):

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv 2023. https://arxiv.org/abs/2302.01318
- Cai, Li, Geng, Peng, Lee, Chen, Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. https://arxiv.org/abs/2401.10774
- Ankner, Parthasarathy, Nrusimha, Rinard, Ragan-Kelley, Brandon. *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding.* arXiv 2024. https://arxiv.org/abs/2402.05109
- Li, Wei, Zhang, Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML 2024. https://arxiv.org/abs/2401.15077
- Li, Wei, Zhang, Zhang. *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees.* EMNLP 2024. https://arxiv.org/abs/2406.16858
- Li, Wei, Zhang, Zhang. *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.* NeurIPS 2025. https://arxiv.org/abs/2503.01840
- Miao, Oliaro, Zhang et al. *SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference and Verification.* ASPLOS 2024. https://arxiv.org/abs/2305.09781
- Chen, May, Svirschevski, Huang, Ryabinin, Jia, Chen. *Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding.* NeurIPS 2024 (as *Sequoia: Scalable and Robust Speculative Decoding*). https://arxiv.org/abs/2402.12374
- Fu, Bailis, Stoica, Zhang. *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* ICML 2024. https://arxiv.org/abs/2402.02057
- He, Zhong, Cai, Lee, He. *REST: Retrieval-Based Speculative Decoding.* NAACL 2024. https://arxiv.org/abs/2311.08252
- Oliaro, Jia, Campos, Qiao. *SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications.* NeurIPS 2025 (Spotlight). https://arxiv.org/abs/2411.04975
- Zhang, Wang, Huang, Xu. *Learning Harmonized Representations for Speculative Sampling (HASS).* ICLR 2025. https://arxiv.org/abs/2408.15766
- Weng, Mei, Qiu, Chen, Liu, Tian. *CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter.* ACL 2025. https://arxiv.org/abs/2502.16880
- Zhao, Pan, Han et al. *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling.* arXiv 2025. https://arxiv.org/abs/2502.14856
- An, Bai, Liu, Li, Barsoum. *PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation.* arXiv 2025. https://arxiv.org/abs/2504.18583
- An, Liu, Liu, Li, Liu, Barsoum. *PARD-2: Target-Aligned Parallel Draft Model for Dual-Mode Speculative Decoding.* arXiv 2026. https://arxiv.org/abs/2605.08632
- Huang, Guo, Wang. *SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths.* COLM 2025. https://arxiv.org/abs/2405.19715
- Su, Giannoula, Pekhimenko. *The Synergy of Speculative Decoding and Batching in Serving Large Language Models.* arXiv 2023. https://arxiv.org/abs/2310.18813
- Kong, Flynn, Peng, Shavit, Kurtz, Marques. *An Interpretable Latency Model for Speculative Decoding in LLM Serving.* arXiv 2026. https://arxiv.org/abs/2605.15051
- Huang, Zhu, Zhan, Hu, Mao, Yu, Liu, Zhang. *MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE.* NeurIPS 2025 (spotlight). https://arxiv.org/abs/2505.19645
- Pan, Tao, Pang, Wang, Zhao, Zhang. *Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding (EVICT).* arXiv 2026. https://arxiv.org/abs/2605.00342
- Liu, Shen, Cen, Shi, Zhang, Qin, Liu, Liu, Yu, Zhu. *D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding.* arXiv 2026. https://arxiv.org/abs/2607.14647
- Hu, Shen, Zhang, Zhang, Dai, Ge, Chen, Li, Wan. *ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios.* arXiv 2026. https://arxiv.org/abs/2604.09603
- Wang, Zhou. *SMART: When is it Actually Worth Expanding a Speculative Tree?* ECCV 2026. https://arxiv.org/abs/2604.09731
- Li, Zhang, Zhang, Wang, Fu, Lai. *Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving.* arXiv 2025–26. https://arxiv.org/abs/2512.22420
- Zhang, Huang, Ke, Han, Long, Zhao, Qi, Zhang. *FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration.* arXiv 2026. https://arxiv.org/abs/2605.20022
- Chen, Liang, Liu. *DFlash: Block Diffusion for Flash Speculative Decoding.* ICML 2026. https://arxiv.org/abs/2602.06036
- Hu, Feng, Wu, Yuan, Zhao, Qian, Wang, Zhao, Jiang, Zhu, Rosing, Zhang. *JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting.* arXiv 2026. https://arxiv.org/abs/2606.18394
- Liu, Cen, Shi, Qin, Zhang, Liu et al. *AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding.* arXiv 2026. https://arxiv.org/abs/2607.25852
- Ringel, Romano. *Accelerating Speculative Decoding with Block Diffusion Draft Trees (DDTree).* arXiv 2026. https://arxiv.org/abs/2604.12989
- Zhang, Qiu, He, Dai. *Cost-Aware Diffusion Draft Trees for Speculative Decoding (CaDDTree).* arXiv 2026. https://arxiv.org/abs/2606.01813
- Oh, Cao, Kim, Jung, Ahmad, Bae. *Bastion: Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting.* arXiv 2026. https://arxiv.org/abs/2605.29727
- Lin, Jang. *DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding.* arXiv 2026. https://arxiv.org/abs/2607.08642
- Wang, Wertheimer, Lim, Srivatsa, Ganti, Zhang. *xPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding.* arXiv 2026. https://arxiv.org/abs/2608.02438
- Li, Luo, Shang, Shen. *DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees.* arXiv 2026. https://arxiv.org/abs/2608.13524
- Mahmoud. *Acceptance Dynamics Across Cognitive Domains in Speculative Decoding.* arXiv 2026. https://arxiv.org/abs/2604.14682
- Zbib, Bazzi, Mohanna, Hammoud, Ghanem. *TAPS: Task Aware Proposal Distributions for Speculative Sampling.* arXiv 2026. https://arxiv.org/abs/2603.27027
- Gloeckle, Youbi Idrissi, Rozière, Lopez-Paz, Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* arXiv 2024. https://arxiv.org/abs/2404.19737
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv 2412.19437 (MTP §3.3, acceptance §5.4.3).
- GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models.* arXiv 2508.06471.
- Zhang, Wang, Li, Shou, Chen, Chen. *Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding.* ACL 2024. https://arxiv.org/abs/2309.08168
- Elhoushi et al. *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.* ACL 2024. https://arxiv.org/abs/2404.16710
- Xia, Li, Zhang, Du, Li. *SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration.* ICLR 2025. https://arxiv.org/abs/2410.06916

Engine source and documentation (read directly from `sgl-project/sglang@main`,
2026-08-17):

- `python/sglang/srt/server_args.py` — all `--speculative-*` flags; `enable_mixed_chunk` assertion; spec CUDA-graph bucket list.
- `python/sglang/srt/arg_groups/overrides.py` — `_deepseek_spec_moe_resolution` (the `is_hip()` gate), `_speculative_moe_runner_default`.
- `python/sglang/srt/speculative/spec_utils.py` — `resolve_num_tokens_per_req`, `select_top_k_tokens`, `draft_kv_indices_buffer_width`, `sample_simulated_acc_len`, `generate_simulated_accept_index`.
- `python/sglang/srt/speculative/eagle_utils.py` — `build_tree_kernel_efficient`, tree-mask modes and shapes.
- `python/sglang/srt/speculative/eagle_worker_v2.py` — draft loop, graph capture, DSA `IndexTopKShareState`.
- `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` — capture buffers, `out_cache_loc` sizing.
- `python/sglang/srt/speculative/adaptive_spec_params.py` — `DEFAULT_ADAPTIVE_CONFIG`, `AdaptiveStepSlot`.
- `python/sglang/kernels/ops/speculative/reject_sampling.py` — the chain rejection-sampling Triton kernel.
- `python/sglang/srt/environ.py` — `SGLANG_SIMULATE_ACC_LEN`, `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH`, `SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND`.
- `docs/docs/advanced_features/speculative_decoding.mdx`, `.../adaptive_speculative_decoding.mdx`
- `docs/cookbook/autoregressive/GLM/GLM-5.2.mdx` and `docs/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx`
- `docs/cookbook/specbundle/*`; `sgl-project/SpecForge` `docs/benchmarks/benchmark.md`, `docs/community_resources/specbundle.md`

Issues and PRs (fetched via the GitHub API):

- sgl-project/sglang [#30209](https://github.com/sgl-project/sglang/issues/30209) — GLM-5.2 FP4 + EAGLE IMA in flashinfer_trtllm draft-MoE bmm; validated fix.
- sgl-project/sglang [#30210](https://github.com/sgl-project/sglang/pull/30210) — the fix PR; **closed unmerged**.
- sgl-project/sglang [#31093](https://github.com/sgl-project/sglang/issues/31093) — GLM-5.2 NVFP4 + EAGLE `6-1-7` IMA during decode CUDA-graph capture on 8×B200.
- sgl-project/sglang [#29787](https://github.com/sgl-project/sglang/pull/29787) — GLM-5.2 MTP IndexShare anchor; the `5-1-6` accept-length table.
- sgl-project/sglang [#32666](https://github.com/sgl-project/sglang/issues/32666) — depth-gated output corruption + depth-4 verify-graph capture IMA.
- sgl-project/sglang [#33872](https://github.com/sgl-project/sglang/pull/33872) — DSv4 silent KV corruption for `num_draft_tokens > 4` (`kMaxMTPDraftTokens`).
- sgl-project/sglang [#32105](https://github.com/sgl-project/sglang/issues/32105) — expert-distribution metrics + `num_steps >= 2` breaks graph capture.
- sgl-project/sglang [#28569](https://github.com/sgl-project/sglang/issues/28569) — EAGLE3 draft-graph replay IMA as the batch shrinks.
- sgl-project/sglang [#30944](https://github.com/sgl-project/sglang/pull/30944) — `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH`.
- sgl-project/sglang [#22246](https://github.com/sgl-project/sglang/pull/22246) — AutoSpec; per-model optimal steps vs batch size.
- flashinfer-ai/flashinfer [#3722](https://github.com/flashinfer-ai/flashinfer/issues/3722) — scale-factor padding → TMA OOB on B200, sibling kernel family.

Other:

- Apoorv Saxena. *Prompt Lookup Decoding.* https://github.com/apoorvumang/prompt-lookup-decoding
- vLLM speculative-decoding documentation. https://docs.vllm.ai/en/latest/features/speculative_decoding/
