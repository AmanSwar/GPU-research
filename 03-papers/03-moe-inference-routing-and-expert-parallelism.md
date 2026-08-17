# MoE inference: routing, expert parallelism, load balance and offloading

## What this is

A survey of the Mixture-of-Experts literature *as it bears on serving*, not on training.
Every paper listed here was fetched and read during this pass — abstract at minimum,
method and evaluation section where the numbers mattered. Nothing is cited from memory.
Where a paper's PDF resisted extraction and I only got the abstract, the claim is
labelled `[reported]` rather than `[verified]`.

Confidence labels used throughout:

- `[verified]` — I read the number in the paper body or a primary artefact (README, docs).
- `[reported]` — the authors claim it in an abstract or blog post I read, but I did not
  see the supporting table.
- `[inferred]` — my own arithmetic or reasoning, not stated in any source.

**Grounding assumption.** GLM-5.2's exact hidden/expert dimensions are not in any public
document I could fetch. Where arithmetic below needs them I use GLM-4.5's published
values — hidden `d = 5120`, MoE expert intermediate `d_ff = 1536`, gated FFN (3 matrices
per expert), 1 shared expert, ~89 MoE layers `[verified, arXiv:2508.06471 Table 1]` — and
substitute our stated 256 routed experts / top-8. Every derived number is marked
`[inferred]` and every assumption is stated inline. If the real GLM-5.2 dims differ, the
*shapes* of the conclusions hold; the magnitudes shift.

---

## Bottom line for our system

Ranked by expected effect on our two objectives, given 8×B200, TP8, 1.46 TB of HBM,
256 experts / top-8, and EAGLE 3-1-4.

1. **First, determine whether our MoE is running TP-sharded or EP-sharded. The entire
   load-balancing literature applies to one and not the other.** Under pure TP8 every
   rank computes exactly 1/8 of every activated expert, so expert-load imbalance is
   *identically zero* and EPLB, redundant experts, and expert placement are all no-ops.
   Under EP8 (SGLang's `--enable-ep-moe` / `--moe-a2a-backend deepep`) imbalance is
   severe and dominates. Our measured "47% of collective time is rank-arrival skew"
   means very different things in the two cases. Diagnostic: log per-rank MoE grouped-GEMM
   wall time for a fixed batch. If the max/mean across ranks is ~1.0, the skew is *not*
   MoE and the entire EPLB branch of this survey is irrelevant to us. `[inferred]`

2. **At concurrency 1, TP8 is the right MoE parallelism and EP8 is a latency trap — by
   roughly 2–3× on the expert-GEMM portion.** With 8 expert draws spread over 8 EP ranks,
   the *max-loaded* rank gets ~3 experts while ~2.7 ranks get zero, and the layer's
   latency is set by the max. Under TP8 every rank reads exactly 8/8 of an expert's worth
   of weights. Balls-in-bins skew at batch 1 is irreducible: it is not persistent expert
   popularity, so no EPLB-class rebalancer can remove it. The literature almost never
   evaluates this regime — it studies EP32–EP320 at thousands of tokens per step, where
   the law of large numbers restores balance. `[inferred]`, arithmetic in §5.

3. **Speculative decoding gives us essentially zero MoE weight-traffic amortisation, and
   this bounds how much draft depth can ever buy.** On a dense model, verifying N draft
   tokens reads the weights once instead of N times. On a 256-expert top-8 model, N=4
   draft tokens activate a *union* of ~30.5 distinct experts versus 32 read serially — a
   4.6% saving `[inferred]`. Confirmed in kind by MoE-Spec: a 127-token tree on OLMoE
   (64 experts, top-8) activates 54 of 64 experts per layer `[verified, arXiv:2602.16052]`.
   Our 3.09× spec gain is therefore coming from attention, dense GEMM and collective
   amortisation, and the MoE term is close to linear in draft length. Expect sharply
   diminishing returns from lengthening the draft.

4. **The single most transferable speculative+MoE result is AcceptMoE, and it was measured
   on Blackwell with all experts resident — our exact regime.** 1.29× throughput with full
   residency, accuracy within 0.27 pp of baseline, by restricting the verifier to a
   self-sized top-B expert set weighted by the probability each draft position survives
   `[verified, arXiv:2608.02989]`. Caveat that must not be glossed: this makes the verifier
   no longer the true model, so speculative decoding stops being exactness-preserving.
   That is a product decision, not a systems one.

5. **Attack the collective *latency*, not the collective *bandwidth*.** At C=1 the TP
   all-reduce payload is ~51 KB `[inferred]` — several microseconds of NVLink5 at
   900 GB/s each way, i.e. the cost is barrier + round trip, not transfer. That is
   consistent with 47% of our collective time being arrival skew. The levers are one-shot /
   NVLS-multimem small-message all-reduce, fewer sync points per layer, and removing
   whatever causes ranks to arrive at different times — *not* DeepEP, not COMET, not
   2DH all-to-all, all of which optimise bandwidth at scale.

6. **Run the shared expert on a second stream, concurrent with router + dispatch.** GLM-4.5
   carries 1 shared expert per MoE layer `[verified, arXiv:2508.06471]`. The shared expert
   has no routing dependency, so its GEMM is the one piece of MoE work that can start
   before the router finishes. DeepSeek's production deployment co-locates shared experts
   with redundant experts precisely to exploit this `[verified, DeepSeek inference system
   overview]`. Low risk, no numerics change.

7. **Do not offload anything. Ever.** 1.46 TB of HBM against a model that is ~400–500 GB
   at FP8 and ~200–250 GB at NVFP4 `[inferred]`. Every offloading paper in §8 —
   MoE-Infinity, Fiddler, Klotski, HOBBIT, MoE-Lightning, ExpertFlow — optimises a
   constraint we do not have. Their *routing-prediction* machinery is reusable (see 8);
   their storage hierarchies are not.

8. **The one idea worth stealing from the offloading literature is the router lookahead
   predictor, repurposed to start collectives early.** PROBE's gate-initialised lookahead
   predictor distils the target layer's router from the *previous* layer's hidden states
   and hits ~90% top-k accuracy one layer ahead, with 2×top-K recall approaching 100%
   `[verified, arXiv:2602.00509]`. We do not need it to prefetch weights. We could need it
   to issue the dispatch/all-to-all (or the tile schedule) before the router result is
   available, shaving a serialisation point off every MoE layer.

9. **If we ever run EP8 for the cost-per-user objective, budget the SMs the collective
   steals.** DeepEP on SM100 (our silicon) reaches 726 GB/s dispatch / 740 GB/s combine on
   EP8 NVLink using 64 SMs, and 643/675 GB/s using 24 SMs `[verified, DeepEP README]`. On a
   148-SM B200, 24–64 SMs is 16–43% of the machine taken away from compute. At C=1 that
   trade is plainly bad; at C=64 it may not be.

10. **Dropless is settled; do not reintroduce capacity factors.** MegaBlocks measured that
    a capacity-factor-1 MoE gave a 0.15 validation-loss reduction while the dropless
    variant gave 0.26 — 1.73× larger `[verified, arXiv:2211.15841]`. Every modern engine
    uses grouped/block-sparse GEMM with no token dropping. Any "capacity factor" knob that
    appears in our stack is a bug surface, not a tuning opportunity.

---

## 1. The architecture papers that define the serving problem

| Title | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) | Google | arXiv 2006.16668, ICLR 2021 | 2048 TPUv3 | 600B-param MoE translation model trained in 4 days `[reported]` | Concepts yes (top-2 + capacity + aux loss), code no |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | Google | JMLR 2022, arXiv 2101.03961 | TPUv3 | 7× pre-training speedup; top-1 routing viable `[reported]` | Concepts yes |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | Mistral AI | arXiv 2401.04088 (Jan 2024) | — | 8 experts, top-2, 47B total / 13B active; matches Llama-2 70B `[reported]` | Yes — the reference small MoE |
| [DeepSeekMoE: Towards Ultimate Expert Specialization in MoE Language Models](https://arxiv.org/abs/2401.06066) | DeepSeek-AI | arXiv 2401.06066 | — | Fine-grained segmentation + shared-expert isolation; 2B matches GShard 2.9B `[reported]` | Yes — the template for everything since |
| [Auxiliary-Loss-Free Load Balancing Strategy for MoE](https://arxiv.org/abs/2408.15664) | Wang, Gao, Zhao, Sun, Dai (DeepSeek) | arXiv 2408.15664 (Aug 2024) | — | Bias-only balancing beats aux-loss on both quality and balance, ≤3B / 200B tokens `[reported]` | Yes — DeepSeek-V3, GLM-4.5 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | DeepSeek-AI | arXiv 2412.19437 | H800 | 256 routed + 1 shared, top-8, node-limited M=4; full inference deployment recipe `[verified]` | Yes |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | Qwen Team, Alibaba | arXiv 2505.09388 | — | 128 experts, top-8, **no shared expert**, global-batch balance loss `[verified]` | Yes |
| [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) | Moonshot AI | arXiv 2507.20534 | — | 1T total / 32B active MoE, MuonClip, 15.5T tokens `[reported]` | Yes |
| [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471) | Z.ai / GLM Team | arXiv 2508.06471 | — | 355B/32B, 160 experts top-8 + 1 shared, 89 MoE layers, MoE-layer MTP head `[verified]` | Yes — our lineage |
| [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) | DeepSeek-AI | arXiv 2512.02556 (Dec 2025) | — | DeepSeek Sparse Attention (DSA) `[reported]` | Yes |
| [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) | DeepSeek-AI | arXiv 2606.19348 (2026) | — | V4-Pro 1.6T/49B, V4-Flash 284B/13B; 27% of V3.2's per-token FLOPs, 10% of KV `[reported]` | Announced; we target it |

### What actually matters from this row of papers

**Capacity factor and token dropping (GShard, Switch) are historical.** Switch defines
expert capacity as `(tokens per batch / num experts) × capacity factor`, and on overflow
"computation is skipped and the token representation is passed directly to the next layer
through the residual connection" `[verified, ar5iv 2101.03961]`. Their own table shows the
quality/throughput trade: CF 2.0 → −1.554 neg-log-perp at 860 ex/sec; CF 1.25 → −1.553 at
910; CF 1.0 → −1.561 at 1000 `[verified]`. The load-balancing auxiliary loss is
`α·N·Σᵢ fᵢ·Pᵢ` with `α = 10⁻²` `[verified]`. Both mechanisms exist because the 2020-era
kernels needed static shapes. MegaBlocks killed the need (§2).

**Fine-grained experts are the reason batch-1 MoE is hard.** DeepSeekMoE's contribution is
"finely segmenting the experts into mN ones and activating mK from them" plus shared-expert
isolation `[verified, arXiv:2401.06066]`. The serving consequence: an expert is now a
`3 × 5120 × 1536` FFN ≈ 23.6M params ≈ 23.6 MB at FP8 `[inferred]`, and a single token
touches 8 of them scattered across a 256-way population. Every batch-1 problem in §6 and
every load-imbalance problem in §4 is downstream of this choice. PROBE states the general
form of it: finer expert granularity "transforms load skew into a severe straggler effect,
bounding layer latency by the most heavily loaded device" `[verified, arXiv:2602.00509]`.

**Auxiliary-loss-free balancing changed what the serving system inherits.** DeepSeek-V3's
rule adds a per-expert bias `bᵢ` to the affinity score *for top-k selection only*, not for
the gate weight: `g'ᵢ,ₜ = sᵢ,ₜ if (sᵢ,ₜ + bᵢ) ∈ Topk({sⱼ,ₜ + bⱼ}, Kᵣ) else 0`, with the
bias decremented by γ when an expert is overloaded and incremented by γ when underloaded,
γ = 0.001 for the first 14.3T tokens and 0.0 for the final 500B `[verified, arXiv:2412.19437]`.
A complementary sequence-wise balance loss survives with α = 0.0001 — "an extremely small
value" `[verified]`. GLM-4.5 uses the same loss-free sigmoid-gate routing `[verified,
arXiv:2508.06471]`. Qwen3 went the other way and adopted a global-batch load-balancing loss,
and explicitly dropped shared experts `[verified, arXiv:2505.09388]`.

The serving-side reading: loss-free balancing balances *training-corpus* load, per step,
globally. It says nothing about the load induced by one deployment's traffic mix at one
instant, which is why DeepSeek still ships an online expert-parallel load balancer in
production (§4).

**Node-limited routing is a deployment constraint baked into the model.** DeepSeek-V3
restricts each token to at most M=4 nodes, chosen by summing the highest `Kᵣ/M` affinity
scores per node `[verified]`. On a single 8-GPU node this is inert for us. It matters only
if we ever go multi-node EP.

**Architecture table for our lineage** `[verified, arXiv:2508.06471 Table 1]`:

| | GLM-4.5 | GLM-4.5-Air |
|---|---|---|
| total / active params | 355B / 32B | 106B / 12B |
| dense layers / MoE layers / MTP | 3 / 89 / 1 | 1 / 45 / 1 |
| hidden dim | 5120 | 4096 |
| MoE intermediate dim | 1536 | 1408 |
| routed experts / active / shared | 160 / 8 / 1 | 128 / 8 / 1 |
| attn heads / KV heads | 96 / 8 | 96 / 8 |

Note the last row of the report's own commentary: GLM-4.5 is "deeper" — more layers, fewer
experts — than DeepSeek-V3 and Kimi K2, and the MoE layer is used *as* the MTP layer to
support speculative decoding `[verified]`. That last fact is directly relevant to §7: our
draft head is itself a MoE layer, so drafting also pays MoE weight traffic.

---

## 2. Systems foundations: making the expert GEMM not be terrible

| Title | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| [DeepSpeed-MoE: Advancing MoE Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596) | Microsoft | ICML 2022, arXiv 2201.05596 | not stated in abstract | 7.3× better latency+cost vs prior MoE inference `[reported]` | DeepSpeed-MII; largely superseded |
| [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models](https://doi.org/10.1145/3503221.3508418) | Tsinghua (He, Zhai et al.) | PPoPP 2022 | ≤64 GPUs | 1.37×–17.87× vs ZeRO / GShard / BASE Layers `[reported]` | Ideas yes (shadowing), code niche |
| [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382) | Microsoft Research | MLSys 2023, arXiv 2206.03382 | up to 2048 A100 SXM 80GB | 4.96× (16 GPU) / 5.75× (2048 GPU) per MoE layer vs Fairseq; 2.11× end-to-end inference on 128 GPUs `[verified]` | Yes — `microsoft/tutel` |
| [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841) | Stanford / Microsoft / Google (Gale, Narayanan, Young, Zaharia) | MLSys 2023, arXiv 2211.15841 | 8× A100 SXM4 80GB | 1.38×–4.35× vs Tutel; 1.8–2.4× vs Megatron-LM `[verified]` | Yes — vLLM, Megatron, MegaBlocks pkg |
| [Scattered Mixture-of-Experts Implementation (ScatterMoE)](https://arxiv.org/abs/2403.08245) | Tan, Shen, Panda, Courville (Mila / IBM) | arXiv 2403.08245 | not stated in abstract | Higher throughput + lower memory than MegaBlocks by avoiding padding/copies `[reported]` | Yes — used in several trainers |
| [Static Batching of Irregular Workloads on GPUs: Framework and Application to Efficient MoE Model Inference](https://arxiv.org/abs/2501.16103) | Alibaba (Li, Li, Zhang et al.) | arXiv 2501.16103 (Jan 2025) | H800, H20 | 91% (H800) / 95% (H20) of peak Tensor Core throughput on MoE `[reported]` | Ideas in production kernels |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | DeepSeek-AI | GitHub (2025–) | SM90, **SM100** | Contiguous + **masked** grouped GEMM layouts; up to 1550 TFLOPS dense on H800 `[verified]` | Yes — SGLang, vLLM |

### MegaBlocks: the mechanism, precisely

MoE's awkwardness is that expert `i` receives a variable number of tokens. Dense frameworks
handle this by padding every expert to a fixed capacity (wasting FLOPs) or by dropping
overflow (losing quality). MegaBlocks removes the choice by expressing the whole MoE FFN as
**block-sparse matrix multiplication**.

- Format: **blocked-CSR (BCSR)** as the primary encoding, augmented with materialised row
  indices per nonzero block — a hybrid **blocked-CSR-COO** — so that the sparse-output
  (SDD) kernel can be parallelised without a serial row scan `[verified, ar5iv 2211.15841]`.
- Block size: **128 × 128**, chosen empirically; "128×128 tiles consistently perform on-par
  or better than other configurations" on A100 `[verified]`.
- A 2-layer expert MLP becomes **an SDD followed by a DSD** — sparse-output dense×dense,
  then dense×sparse-dense `[verified]`. The token→expert assignment defines which blocks
  are nonzero; every expert's token count is rounded up to a 128-row block, which is the
  only padding that remains.
- Backward passes need `Aᵀ`. Rather than transposing nonzero values, they build **transpose
  indices** — metadata "allowing efficient iteration through the matrix in transposed order
  with a layer of indirection" `[verified]`.

Evaluation: 8× A100 SXM4 80GB with 8-way expert model parallelism. Speedups over Tutel's
dMoE — MoE-XS 1.38×, MoE-Small 2.0×, MoE-Medium 4.35× — and 1.8–2.4× end-to-end over
Megatron-LM at equal validation loss `[verified]`. The quality argument is the one that
mattered historically: CF=1 MoE achieved a 0.15 validation-loss reduction; the dropless MoE
achieved 0.26, "1.73× larger" `[verified]`.

**For us:** this is settled infrastructure. Our concern is not whether to be dropless but
whether the grouped-GEMM kernel is efficient at *tiny M*. MegaBlocks' 128-row block
granularity is designed for training-scale token counts. At C=1 with 5 speculative tokens
and 8 experts, we have ≤5 rows per expert against a 128-row block — 96% padding waste in
the block-sparse formulation `[inferred]`. This is exactly why DeepGEMM's **masked** layout
exists.

### DeepGEMM's masked layout is the batch-1-relevant piece

DeepGEMM offers two grouped layouts `[verified, DeepGEMM README]`:

- **Contiguous layout** — group along M, N and K fixed; each expert's segment aligned to the
  GEMM M block size. For training forward and inference *prefill*.
- **Masked layout** — for *decoding with CUDA graphs*, where "the CPU cannot determine token
  distribution per expert" at capture time. "By providing a mask tensor, the kernel computes
  only the valid portions."

The masked layout is what makes CUDA-graph-captured MoE decode possible at all: without it,
the per-expert token counts would force a device→host sync every layer, which at ~90 layers
would be fatal. SM100 (our silicon) is supported with CUDA 12.9+ `[verified]`. If our
`glm-kernels` MoE path is not already using a masked/mask-equivalent formulation under CUDA
graphs, that is the first thing to check.

Alibaba's *Static Batching of Irregular Workloads* attacks the same problem from the other
side: statically batch the irregular per-expert work into a **single kernel** with a runtime
task-mapping mechanism, reaching 91% of peak Tensor Core throughput on H800 and 95% on H20
for MoE inference `[reported, arXiv:2501.16103]`. The persistent-kernel + runtime-tile-map
pattern is the right one for B200 too, and generalises to our DSA indexer.

### Tutel: adaptive parallelism, and the 2DH all-to-all

Tutel is the most useful of the older systems papers for us because its central claim is
that *the right parallelism changes with the workload*, and it makes switching free.

1. **Switchable parallelism.** Tutel shows the seven possible DP/MP/EP combinations reduce
   to two — pure DP, and EP+DP+MP — and that both can use an **identical tensor layout**
   following ZeRO-DP stage-3 partitioning. A single control parameter `r` ranges from 0
   (pure DP) to `⌈W/E⌉` (pure EP+MP); switching adjusts `r` and does local repeat/sum
   operations only. "All partitioning and reshaping operations are done inline", so there
   is no parameter migration and no copy `[verified, ar5iv 2206.03382]`.
2. **2DH all-to-all.** Linear all-to-all splits data into `n` chunks for `n` GPUs, so at
   2048 GPUs the messages become tiny and NVLink/IB go idle. 2DH runs four phases —
   strided local copy to align same-node destinations, intra-node all-to-all, second strided
   alignment, inter-node all-to-all — keeping "constant and low latency regardless of n in
   the first three phases" `[verified]`.
3. **Adaptive pipelining.** Tokens are partitioned along the *capacity* dimension only
   across the two all-to-alls and the intervening expert compute (not the whole layer), and
   submitted asynchronously to comm and compute streams. Pipeline degree is chosen from
   {1,2,4,8} × {Linear, 2DH} by ternary search over a precomputed dictionary, giving
   "9%–101% improvement in average" over a fixed pipeline `[verified]`.
4. **Flexible all-to-all.** Reshapes the output layout to `(Eg, C, D)` so the expert GEMM has
   "the same-shaped matrix multiplication at any scale", decoupling GEMM efficiency from
   world size `[verified]`.

**For us:** items 2 and 3 are scale techniques that do nothing on 8 NVLink-connected GPUs.
Item 1 is the interesting one — it is the published precedent for *the thing we should
probably want*: TP-sharded MoE at low concurrency, EP-sharded at high concurrency, with a
layout that permits switching. Neither SGLang nor vLLM exposes a runtime switch today; both
fix the MoE parallelism at engine start `[verified, vLLM/SGLang docs]`.

### FasterMoE's dynamic shadowing — the idea that keeps getting reinvented

FasterMoE (Tsinghua, PPoPP 2022) contributes a roofline-like performance model, a
**dynamic shadowing** approach for load imbalance, a **smart fine-grained schedule** that
splits operations and executes them concurrently, and a **congestion-avoiding expert
selection** strategy `[verified, PPoPP 2022 program page]`. Speedups 1.37×–17.87× against
ZeRO, GShard and BASE Layers on ≤64 GPUs `[reported]`.

Dynamic shadowing is the seed of everything in §4: when an expert is hot, *broadcast its
weights* to all ranks and compute it locally, instead of sending all those tokens to the
one rank that owns it. The trade is `weight bytes` versus `token bytes`. For a fine-grained
expert at 23.6 MB FP8, broadcasting beats shipping tokens once more than
`23.6 MB / (5120 × 1 B) ≈ 4600` tokens would have to be routed to it `[inferred]` — which is
a prefill-scale number, not a decode-scale one. MoE-Prefill (§3) rediscovers exactly this
and formalises it as AsyncEP.

---

## 3. Expert parallelism, the all-to-all, and the 2025–26 communication stack

| Title | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | DeepSeek-AI | GitHub, Feb 2025 | SM90, **SM100** | EP8 NVLink: **726 GB/s dispatch, 740 GB/s combine** (64 SMs); 643/675 (24 SMs) `[verified]` | Yes — SGLang, vLLM |
| [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811) | ByteDance Seed + SJTU | arXiv 2502.19811 | 8× H800 80GB NVLink; 8× L20 PCIe | 1.96× per MoE layer, 1.71× end-to-end; hides 86.5% of comm vs 68.6% Tutel, 29.2% FasterMoE `[verified]` | Yes — ByteDance, open-sourced |
| [MegaScale-Infer: Serving MoE at Scale with Disaggregated Expert Parallelism](https://arxiv.org/abs/2504.02263) | ByteDance Seed + PKU | SIGCOMM 2025 | 8×8 A100 80GB, 400GB/s NVLink, 200Gbps IB; H20+L40S hetero | 1.90× per-GPU throughput, 1.7× throughput/$ vs SOTA `[reported]`; Scaled-MoE 7.11× vs vLLM, 1.90× vs TRT-LLM `[verified]` | ByteDance internal |
| [NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL](https://arxiv.org/abs/2603.13606) | **NVIDIA** (18 authors) | arXiv 2603.13606 (Mar 2026) | H100 cluster | `ncclEpDispatch` / `ncclEpCombine` primitives; native NCCL path replacing DeepEP/Hybrid-EP `[reported]` | Becoming the supported path |
| [UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods](https://arxiv.org/abs/2607.06202) | NJU + Huawei | arXiv 2607.06202 (Jul 2026) | NVL72/576, CloudMatrix384 | All-to-all latency −52.4%; TPOT −11.1% `[reported]` | Huawei production |
| [Eliminating Hidden Serialization in Multi-Node Megakernel Communication (Perseus)](https://arxiv.org/abs/2605.00686) | Oh, Singh | arXiv 2605.00686 (May 2026) | multi-node, ≤8 nodes | MoE megakernels regress **up to 10×** on 8 nodes; Perseus recovers up to 10.3× `[reported]` | Research |
| [HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs](https://arxiv.org/abs/2605.23764) | Huawei + USTC | arXiv 2605.23764 (May 2026) | Ascend A3 | Dispatch-to-Combine MoE-FFN latency −1.58× `[reported]` | Huawei MindSpore |
| [Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](https://arxiv.org/abs/2503.04398) | Li, Zhang, Wang, Chen, Zheng | ICLR 2026, arXiv 2503.04398 v5 | not extracted | Expert clustering + request/token scheduling to cut all-to-all `[reported]` | Research |
| [MoE-Prefill: Zero Redundancy Overheads in MoE Prefill Serving](https://arxiv.org/abs/2605.02960) | Su, Ruwase, Ganesan, Qiao, Rajbhandari, Yang, Cheng, He | arXiv 2605.02960 (May 2026) | 4 hw/precision configs | AsyncEP (weight AllGather, not activation AllToAll): 1.35–1.37× real, 1.59× long-context; 29.8–36.2% MFU `[reported]` | Research (Snowflake/DeepSpeed lineage) |

### DeepEP's SM100 numbers are the most directly applicable datum in this survey

DeepEP's benchmark table is measured on **SM100** — the same architecture as our B200 —
with 8K tokens/batch, hidden 7168, top-8, FP8 dispatch, BF16 combine `[verified, README]`:

| Arch | Topology | Dispatch BW | Combine BW | SMs |
|---|---|---|---|---|
| SM100 | EP 8 (NVLink) | 726 GB/s | 740 GB/s | 64 (max) |
| SM100 | EP 8 (NVLink) | 643 GB/s | 675 GB/s | 24 (min) |
| SM100 | EP 8×2 (CX7 RDMA) | 90 GB/s | 91 GB/s | 12 |
| SM90 | EP 8×2 (CX7 RDMA) | 90 GB/s | 81 GB/s | 12 |

Two readings. First, **intra-node NVLink EP is ~8× the bandwidth of the fastest inter-node
path** — so if we ever consider multi-node EP for a bigger model, the all-to-all falls off a
cliff. Second, and more important for us: **the SM count is the price.** Going 24 → 64 SMs
buys only 13% more dispatch bandwidth. On a 148-SM B200, dedicating 64 SMs to communication
to gain 13% is a bad trade at any concurrency we care about; 24 SMs (16% of the GPU) is the
sane setting if we run EP at all `[inferred]`.

DeepEP also provides a **hook-based overlap** that "does not occupy any SM resource" for the
low-latency decode kernels, plus an `EventOverlap` interface where the user overlaps compute
while communication is in flight and calls `event.current_stream_wait()` before consuming
`[verified]`. Note the V2 caveat in the README: "0 SM RDMA low-latency EP is no longer
supported" `[verified]` — the zero-SM trick was inter-node-specific and has been withdrawn.

### COMET: the state of the art in comm/compute overlap, and why it does not help at C=1

COMET's framing number: across popular MoE models on Megatron-LM on 8× H800, "the
communication among devices accounts for 47% of the total execution time on average"
`[verified, arXiv:2502.19811]`. (Coincidentally close to our 47%-of-collective-time skew
figure — the two are unrelated quantities and must not be conflated.)

The two mechanisms:

1. **Shared-tensor-based dependency resolving.** Communication is token-granular; computation
   is tile-granular. COMET decomposes the shared buffer along the dimension where data stays
   independent — the M (token) dimension in MoE layer 0, the N (embedding) dimension in layer
   1 — then reschedules sub-tensors into computation tiles, sorting tokens by source rank so
   local data is processed first, and in layer 1 executing GroupGEMM column-wise so the
   reduction proceeds incrementally instead of waiting for all experts `[verified]`.
2. **Adaptive workload assignment via thread-block specialisation.** Rather than mixing comm
   and compute inside one thread block ("vertical fusion"), COMET dedicates whole thread
   blocks to one or the other. Since on Hopper "each SM only accommodate one thread block",
   the producer/consumer block ratio `(n_p, n_c)` *is* the SM split. A library of
   pre-compiled kernels at different division points is selected at runtime from profiled
   metadata `[verified]`.

Result: 1.96× per MoE layer, 1.71× end-to-end, hiding 86.5% of communication vs Tutel's
68.6% and FasterMoE's 29.2% `[verified]`. Models: Mixtral-8x7B (8 experts, top-2),
Qwen2-MoE-2.7B (64, top-4), Phi-3.5-MoE (16, top-2).

**For us:** COMET overlaps communication with the *expert GEMM of the same layer*. At C=1
with 5 tokens there is barely any expert GEMM to hide behind — the layer is a sequence of
short latency-bound steps. COMET is a throughput technique. It is worth revisiting only if
we build a C≥64 cost-optimised deployment on EP.

### MegaScale-Infer and Janus: attention/expert disaggregation

MegaScale-Infer's premise is that MoE sparsity "shifts feed-forward networks from being
compute-intensive to memory-intensive during inference", so attention and FFN should be
disaggregated onto separate pools with independent parallelism and even different GPUs
`[verified, arXiv:2504.02263]`. Attention nodes replicate attention with DP and hold the KV
cache; expert nodes hold FFN with EP. **Ping-pong pipeline parallelism** splits the batch
into `m` micro-batches shuttled between the two pools, with three stated conditions:
`Ta ≈ Te`, `Tc < Tf`, and `m × Tf ≥ 2 × (Tf + Tc)` — needing at least 3 micro-batches on
fast networks and 4 on slow ones `[verified]`. Their **M2N** library replaces NCCL for the
many-to-many attention↔expert traffic, eliminating GPU–CPU copies, group init overhead and
GPU synchronisation, giving 68.2% lower median latency and 4.2× throughput at 256 KB with
8 senders/receivers `[verified]`.

Results: Mixtral-8x22B 2.56× / 1.28× per-GPU decoding throughput vs vLLM / TensorRT-LLM;
Scaled-MoE (317B) 7.11× / 1.90×; heterogeneous H20+L40S up to 3.24× / 1.86× on unit-cost
throughput `[verified]`. Hardware: 8 nodes × 8 A100 80GB.

[Janus (arXiv:2512.13525)](https://arxiv.org/abs/2512.13525) — a different paper from the
SIGCOMM'23 Janus — takes the same disaggregation further with "a lightweight,
microsecond-scale activation scheduler that balances per-layer activated experts across MoE
instances", claiming up to 4.7× per-GPU throughput under token-level SLOs `[reported]`.
[FinDEP (arXiv:2512.21487)](https://arxiv.org/abs/2512.21487) adds shared-expert support and
a task-granularity scheduling solver to disaggregated EP, reporting up to 1.61× over prior
methods and 1.24× on 32 GPUs with DeepSeek-V2 and Qwen3-MoE `[reported]`.

**For us: not applicable, and the reason is structural.** Disaggregation buys the ability to
provision attention and FFN independently and to keep both busy via micro-batch ping-pong.
On a single 8-GPU NVLink node serving one model, splitting the 8 GPUs into an attention pool
and an expert pool halves the parallelism available to each phase and adds a hop that NVLink
does not need. The ping-pong condition `m ≥ 3` also cannot be met at C=1. File under
"revisit if we ever go multi-node".

### MoE-Prefill inverts the whole thing: move weights, not tokens

MoE-Prefill's claim is that for long-sequence, large-batch prefill you should **gather
experts by weight rather than route them by activation** — replacing the synchronous
per-layer activation AllToAll with an asynchronous weight AllGather overlapped with compute
(AsyncEP) `[verified, arXiv:2605.02960]`. Qwen3-235B-A22B across four hardware/precision
configurations: 1.35–1.37× on real workloads, up to 1.59× on long-context synthetic, 29.8–36.2%
per-GPU MFU `[reported]`.

This is FasterMoE's dynamic shadowing taken to its limit, and it has a crisp break-even
that we can compute for ourselves. Broadcasting an expert's weights costs `3·d·d_ff·b` bytes
once; routing tokens to it costs `d·b` bytes per token per direction. For our assumed
GLM-5.2 expert at FP8: break-even at roughly `3 × 1536 / 2 ≈ 2300` tokens routed to that
expert per layer `[inferred]`. At 8k-token prefill batches with 256 experts and top-8, mean
tokens per expert is `8192 × 8 / 256 = 256` — an order of magnitude below break-even, so
routing tokens still wins for us at prefill. It flips only at very large prefill batches or
for models with far fewer, fatter experts.

---

## 4. Expert load imbalance: what is actually measured, and what fixes it

| Title | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| [EPLB — Expert Parallelism Load Balancer](https://github.com/deepseek-ai/EPLB) | DeepSeek-AI | GitHub, Feb 2025 | — | Redundant-expert replication + hierarchical/global packing `[verified]` | **Yes** — vLLM, SGLang, TRT-LLM |
| [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | DeepSeek-AI | Open Source Week, Feb 2025 | H800 nodes | Prefill EP32 + 32 redundant experts; decode EP144 + 32 redundant; 73.7k in / 14.8k out tokens/s/node `[verified]` | Yes — production |
| [Large-Scale Expert Parallelism on 96 H100 GPUs](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) | LMSYS / SGLang | Blog, May 2025 | 12 nodes × 8 H100 | 52.3k in / 22.3k out tokens/s/node; **EPLB 1.49× prefill, 2.54× decode**; TBO +27–35% `[reported]` | Yes — SGLang |
| [Scaling Large MoE Models with Wide EP on NVL72](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/) | NVIDIA | Blog | GB200 NVL72 | EP32 vs EP8 → up to **1.8× per-GPU throughput** at 100 tok/s/user `[reported]` | Yes — TensorRT-LLM |
| [UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing](https://arxiv.org/abs/2606.04101) | Wei, Jin, Dai, Zhong et al. | arXiv 2606.04101 (Jun 2026) | 64-GPU RSN ×4 (≤256 GPUs) | Inter-rank imbalance **1.30–4.01× → 1.01–1.04×**; +42% train vs Megatron; **1.56× vs SGLang, 1.29× vs EPLB** prefill `[verified]` | Research |
| [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509) | Zhu, Ye, Liu, Ouyang, Song | arXiv 2602.00509 (Feb 2026) | 8× Hopper-141GB, 900 GB/s NVSwitch | Prefill IR > 2.6; decode IR 1.43–2.28; ~50% of global compute idles at barriers; 1.32× prefill / 1.26× decode `[verified]` | Research |
| [ViBE: Co-Optimizing Workload Skew and Hardware Variability for MoE Serving](https://arxiv.org/abs/2606.00735) | Go, Scrbak, Wu, Manne, Mahajan | arXiv 2606.00735 (May 2026) | not extracted | Hardware variability is a *second* straggler source; +14% SLO attainment, −45% P90 TTFT `[reported]` | Research |
| [EasyBalance: Cross-Layer Load Balancing in Distributed MoE Inference](https://arxiv.org/abs/2608.07964) | Wu, Gao, Li, Wu | arXiv 2608.07964 (Aug 2026) | not extracted | Balances across *layers* with no change to expert–device mapping; GPU idling −40%+ `[reported]` | Research |
| [FreeBalance: Pre-Routing Online MoE Load Balancing via Residual Workload Prediction](https://arxiv.org/abs/2608.14205) | Chen, Wu, Kuang, Gao, Li | arXiv 2608.14205 (Aug 2026) | not extracted | max/mean rank load −32.8%; prefill latency −13.1%; hides 5.1 experts/layer of migration `[reported]` | Research |
| [Director: Accelerating Distributed MoE Serving via Online Proactive Expert Placement](https://arxiv.org/abs/2607.08782) | Liu, Guo, Hong, Li, Chen, Wang, Lin, Guo | arXiv 2607.08782 (Jun 2026) | not extracted | (1+ε) polynomial-time placement optimiser; end-to-end latency −11–55% `[reported]` | Research |
| [MoETuner: Optimized MoE Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643) | Go, Mahajan | arXiv 2502.06643 (Feb 2025) | single- + multi-node | ILP placement exploiting cross-layer routing dependency; 9.3% / 17.5% end-to-end `[reported]` | Research |
| [Accelerating MoE Model Inference with Expert Sharding (MoEShard)](https://arxiv.org/abs/2503.08467) | Balmau, Kermarrec, Pires, Espírito Santo, de Vos, Vujasinovic | EuroMLSys 2025, arXiv 2503.08467 | not extracted | Row/column sharding of every expert → *perfect* balance, full token retention; up to 6.4× TTFT `[reported]` | Research (= TP for MoE) |
| [ReaLB: Real-Time Load Balancing for Multimodal MoE Inference](https://arxiv.org/abs/2604.19503) | HKUST (Guangzhou) | arXiv 2604.19503 (Apr 2026) | 8× RTX 5090 | Balance by dropping straggler ranks BF16→**FP4**; 1.29× MoE layer, 1.53× e2e, ≤1.2 pt accuracy `[verified]` | Research |
| [Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference](https://arxiv.org/abs/2510.05497) | Yu, Guan, Yu, Zhou, Hu, Pei, Kang, Ding, Tsai | arXiv 2510.05497 (v5 May 2026) | wafer-scale + existing GPU | 24k+ requests profiled over 4 models 200B–1000B; 6.6× wafer, 1.25× GPU `[reported]`; traces public | Research; **traces useful** |
| [Scaling Multi-Node MoE Inference Using Expert Activation Patterns](https://arxiv.org/abs/2604.23150) | Bambhaniya, Jeong, Park, Yu, Lee, Wang, Kim, Tang, Krishna (Meta + GaTech) | arXiv 2604.23150 (Apr 2026) | not extracted | 100k+ traces from Llama-4 Maverick, DeepSeek-V3, Qwen3-235B; task-specific expert preference `[reported]` | Research |
| [MoEless: Efficient MoE LLM Serving via Serverless Computing](https://arxiv.org/abs/2603.06350) | Stevens Institute + UMD | arXiv 2603.06350 (Mar 2026) | 8× A6000 48GB | Layer-aware load predictor; 43% latency, 84% cost reduction `[reported]` | Research |

### What the measurements actually say

The honest summary of the measured skew literature:

- **Persistent (popularity) skew is real but moderate.** UltraEP measures per-expert
  imbalance of ~1.5–2.0× on Qwen3-235B at EP64 across science/coding/mixed-domain traffic,
  and *rank-level* imbalance of 1.30–4.01× before balancing, reduced to 1.01–1.04× after
  `[verified, arXiv:2606.04101]`.
- **Skew is worse in prefill than decode.** PROBE reports imbalance ratio above 2.6 even at
  ~32K-token batches during prefill, and 1.43–2.28 fluctuating during decode
  `[verified, arXiv:2602.00509]`.
- **Lower top-k is worse.** PROBE finds GPT-OSS-120B (top-4) has higher inherent skew than
  Qwen3-235B (top-8) `[verified]`. Each token drawing more experts averages the load.
- **The straggler cost is roughly half the machine.** PROBE attributes "approximately 50% of
  global compute capacity to idle at synchronization barriers" `[verified]`.
- **It moves with the traffic mix.** UltraEP: "expert popularity varies sharply across
  semantic transitions" `[verified]`. Meta/GaTech's 100k-trace study finds "task-specific
  expert preferences (across code, math, chat domains)" `[reported, arXiv:2604.23150]`.
- **There is a second, non-routing straggler source.** ViBE argues manufacturing variation,
  power limits and thermals create "measurable execution-time differences across nominally
  identical GPUs", and that placement should assign high-load experts to faster devices
  `[reported, arXiv:2606.00735]`. **This one applies to us even under TP8**, and is a
  candidate explanation for our 47% rank-arrival skew.

### EPLB: the mechanism

DeepSeek's EPLB is the production baseline everyone compares against. It works by
**redundant experts** — duplicating heavy-loaded experts and packing the replicas onto GPUs
to equalise load `[verified, EPLB README]`. Two policies:

- **Hierarchical**, used when the node count divides the expert-group count: distribute
  expert groups across nodes to balance load, replicate within each node, then pack replicas
  onto GPUs. Intended for **prefill** with smaller EP size. It exploits *group-limited expert
  routing* to keep same-group experts co-located and cut inter-node traffic `[verified]`.
- **Global**, otherwise: replicate globally ignoring groups, then pack. Intended for
  **decode** with larger EP size `[verified]`.

API: `eplb.rebalance_experts(...)` returns physical→logical map, logical→physical map, and
logical expert counts `[verified]`.

DeepSeek's own production numbers: prefill runs Routed-Expert EP32 / MLA+Shared DP32 over 4
nodes with **32 redundant routed experts**, "each GPU handles 9 routed experts and 1 shared
expert"; decode runs EP144 / DP144 over 18 nodes with **32 redundant routed experts**, "each
GPU manages 2 routed experts and 1 shared expert", and 64 GPUs are dedicated to hosting
redundant + shared experts. The redundant set is "periodically determined based on
statistical expert load" `[verified]`. Throughput: ~73.7k tokens/s input per H800 node
(prefill, cache hits included) and ~14.8k tokens/s output (decode) `[verified]`.

Note the prefill arithmetic: 256 routed + 32 redundant = 288 replicas over 32 GPUs = 9 each.
That is a **12.5% weight-memory overhead** for balance `[inferred]`.

Reported gains: SGLang measured EPLB at **1.49× prefill and 2.54× decode** on 96 H100
`[reported, LMSYS blog]`. UltraEP reports 1.29× *over* EPLB on prefill and +42% vs
Megatron-LM in training where EPLB gives +20%, LPLB +12%, EPLB++ +29% `[verified]`.

**Production flags.** vLLM: `--enable-expert-parallel`, `--enable-eplb`, and
`--eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":0,
"log_balancedness":false,"use_async":true}'`, with `--all2all-backend` ∈
{`allgather_reducescatter`, `deepep_high_throughput`, `deepep_low_latency`,
`flashinfer_nvlink_one_sided`, `flashinfer_nvlink_two_sided`}; `EP_SIZE = TP_SIZE × DP_SIZE`
`[verified, vLLM docs]`. SGLang: `--enable-eplb`, `--moe-a2a-backend` ∈ {none, deepep,
mooncake, nixl, mori, flashinfer, ascend_fuseep, pplx}, `--moe-runner-backend` ∈ {auto,
triton, deep_gemm, cutlass, flashinfer_trtllm, …}, `--deepep-mode` ∈ {auto, normal,
low_latency}, `--enable-two-batch-overlap`, `--enable-single-batch-overlap`,
`--enable-dp-attention`, `--moe-dense-tp-size=1` `[verified, SGLang docs + LMSYS blog]`.
TensorRT-LLM has both static (precomputed mapping) and online EPLB, with "a containerized
design that allows experts to flow in and out of container allocations without breaking the
CUDA graph", weight updates scheduled non-blockingly between forward passes
`[reported, NVIDIA blog]`.

### The distinction the literature blurs, and that decides our case

There are **two** kinds of expert load skew and they need completely different fixes:

- **Persistent skew** — expert 47 is genuinely popular for this traffic mix, across many
  batches. Fixed by replication and placement: EPLB, Director, MoETuner, UltraEP.
- **Transient skew** — with only `T × k` expert draws in this particular forward pass, the
  multinomial simply does not concentrate. Cannot be fixed by placement, because there is
  no stable "hot expert" to replicate.

At C=1 with 5 speculative tokens, we have 40 draws over 256 experts. *All* of our skew is
transient `[inferred]`. Every rebalancer in the table above targets persistent skew and would
buy us nothing at concurrency 1. At C=64 with ~256 tokens per step (2048 draws) persistent
skew begins to dominate and EPLB becomes meaningful — but only if we are running EP at all.

Two papers in the table do attack transient skew and are therefore the more interesting ones
for a latency-oriented deployment:

- **EasyBalance** balances *across layers* rather than across experts: "experts from other
  layers serve as natural redundancy for the current layer", greedily scheduling a subset of
  cross-layer workloads each MoE step and deferring the rest — with **no change to the
  expert–device mapping**, hence no migration and "essentially no additional overhead",
  cutting GPU idling by "mostly over 40%" `[reported, arXiv:2608.07964]`. Zero-migration is
  the property that makes this compatible with CUDA graphs.
- **ReaLB** balances by changing *precision* rather than placement: detect straggler ranks in
  real time and switch their experts BF16→FP4 on the fly (no second weight copy), buying
  ~4× tensor-core throughput on the overloaded rank, for ≤1.2 points average accuracy loss —
  1.29× MoE layer, 1.53× end-to-end on 8× RTX 5090 `[verified, arXiv:2604.19503]`. We already
  have an NVFP4 build; a mixed FP8/NVFP4 rank-adaptive path is mechanically available to us.
  It is also numerically alarming, since different ranks would compute different answers.

### MoEShard is worth naming for what it actually is

MoEShard achieves "perfect load balancing" by "row- and column-wise decomposition of expert
matrices" across GPUs with full token retention, reporting up to 6.4× TTFT over DeepSpeed
`[reported, arXiv:2503.08467]`. **This is tensor parallelism applied to experts** — i.e. it
is what we already do under TP8. It is useful to have a paper that says out loud that the
balance property is free, because it makes the trade in §5 explicit: TP buys perfect balance
and pays in GEMM shape.

---

## 5. EP vs TP vs hybrid for 256 experts on 8 GPUs: the arithmetic

This section is `[inferred]` throughout, from the assumed dims (`d=5120`, `d_ff=1536`, gated
FFN, 256 routed experts, top-8, ~89 MoE layers). Notation: `P=8` GPUs, `T` tokens in the
step, `b` bytes per element.

### Weight memory: a wash

Per MoE layer, per GPU:

| Scheme | Params held per GPU per layer | FP8 bytes |
|---|---|---|
| TP8 (each expert sharded 8 ways) | `256 × 3·d·d_ff / 8` = 755 M | 755 MB |
| EP8 (32 whole experts per GPU) | `32 × 3·d·d_ff` = 755 M | 755 MB |
| EP8 + 4 redundant experts/GPU | `36 × 3·d·d_ff` = 850 M | 850 MB (+12.5%) |

**Weight memory does not distinguish EP from TP at equal degree.** It only distinguishes
them when you add redundancy (EP-only cost) or when EP degree exceeds the expert count.
This kills the most common informal argument for EP on a single node.

### Activation traffic: TP is ~4.5× cheaper at top-8 / EP8

Under EP8 with uniform routing, a token's 8 expert draws land on
`8 × (1 − (7/8)⁸) = 5.25` distinct ranks in expectation. A well-implemented dispatch (DeepEP
does this) sends each token once per *destination rank*, not once per expert.

| Scheme | Collective | Bytes moved per token per MoE layer |
|---|---|---|
| TP8 | all-reduce, payload `d` | `2(P−1)/P · d · b` = `1.75 · 5120 · 2` = **17.9 KB** |
| EP8 | dispatch (FP8) + combine (BF16) | `5.25·d·1 + 5.25·d·2` = **80.6 KB** |

**EP moves ~4.5× more activation bytes than TP** at top-8 on 8 ranks. EP's traffic advantage
only appears when the EP degree is large enough that `E[ranks touched] ≪ k` — i.e. at EP32+
where a token's 8 experts land on ≤8 of 32 ranks and the all-to-all is sparse. On 8 ranks
with top-8, the all-to-all is nearly dense and TP's single all-reduce wins.

This is the quantitative reason NVIDIA's wide-EP result (EP32 beating EP8 by 1.8× per-GPU
throughput on GB200 NVL72 `[reported]`) does *not* transfer to an 8-GPU box: their win comes
from going *wider* than 8, which we cannot do on one node.

### Expert GEMM shape: EP's only real advantage

| Scheme | Grouped-GEMM per rank | N per GEMM |
|---|---|---|
| TP8 | 256 groups (all experts), M = tokens routed | `d_ff/8` = **192** |
| EP8 | ≤32 groups, M = tokens routed to that rank | `d_ff` = **1536** |

N=192 is 1.5 × a 128-wide MMA tile → wave quantisation, and on B200 with NVFP4 the natural
tile is wider still. This is the fine-grained-expert tax: DeepSeekMoE already shrank
`d_ff` to 1536, and TP8 shrinks it to 192. **At high concurrency this is where EP earns its
keep.** At batch 1 it is irrelevant, because the GEMM is bounded by reading the weights, not
by tensor-core occupancy.

### Balance at low token count: the decisive term at C=1

With `T·k` expert draws over `P` EP ranks, MoE layer latency is set by the max-loaded rank.

| Regime | draws | E[max/mean] over 8 ranks |
|---|---|---|
| C=1, 1 token | 8 | **2.59** |
| C=1, 5 spec tokens | 40 | **1.67** |
| C=8, 4 spec tokens (32 tok) | 256 | 1.26 |
| C=64, 4 spec tokens (256 tok) | 2048 | 1.09 (+ persistent skew 1.4–2.3× per PROBE) |
| TP8, any | — | **1.00 exactly** |

(Monte-Carlo, 20k trials per row; uniform routing. `[inferred]`)

At batch 1 under EP8, ~2.7 of 8 GPUs receive *zero* experts and do nothing while the busiest
reads ~3 experts. Under TP8 all 8 read exactly one-eighth of each of the 8 activated experts.
**The MoE portion of a C=1 decode step is therefore ~2–3× faster under TP8 than EP8**, and no
rebalancer can close it because the skew is multinomial noise, not popularity.

### The composite verdict

| | C = 1 (latency) | C = 64 (cost/user) |
|---|---|---|
| weight memory | tie | tie |
| activation bytes | **TP 4.5× better** | **TP 4.5× better** |
| collective count/latency | TP: 1 all-reduce/layer, latency-bound | tie |
| GEMM shape | tie (memory-bound) | **EP 8× wider N** |
| balance | **TP perfect; EP 1.67–2.59×** | EP 1.09× transient + 1.4–2.3× persistent |
| SM tax | TP 0; EP 24–64 SMs for DeepEP | same |
| **verdict** | **TP8, decisively** | **measure — EP8 plausible if GEMM efficiency dominates** |

**Hybrid worth testing: TP4 × EP2.** N = `1536/4` = **384** (exactly 3 × 128 tiles, clean),
8 draws over 2 EP ranks gives E[max/mean] ≈ 1.25, and activation traffic sits between the
two. This is the operating point the literature never evaluates because nobody runs
8-GPU-total EP.

---

## 6. Batch-1 MoE: what the literature says, and the roofline it implies

At concurrency 1 the MoE layer is a pure HBM-bandwidth problem: read the activated experts'
weights, do ~nothing with them, discard. The arithmetic for our machine `[inferred]`, using
GLM-4.5-shaped dims:

- Activated MoE params per token ≈ `89 layers × 9 experts × 3·5120·1536` ≈ **18.9 B**.
  (Matches GLM-4.5's stated 32B activated once attention and dense layers are added
  `[verified, arXiv:2508.06471]`.)
- Total activated weight bytes per token: **~32 GB at FP8**, **~17 GB at NVFP4**
  (0.5 B/param + block scales).
- Under TP8 each GPU reads 1/8: **4.0 GB (FP8)** or **2.1 GB (NVFP4)**.
- B200 HBM3e at 8 TB/s → **500 µs/token (FP8)**, **265 µs/token (NVFP4)**.
- Roofline single-stream ceiling: **~2000 tok/s (FP8)**, **~3800 tok/s (NVFP4)**.

We measure 365 tok/s = 2740 µs/token. **Weight streaming is ~18% of our step time at FP8.**
The other 82% is kernel launch, dependency chains, collective latency and skew. This is the
most important framing fact in this document: **we are not bandwidth-bound at C=1, and no
amount of MoE weight-traffic optimisation will get us to TileRT's ~500 tok/s.** The 47%
rank-arrival skew inside 19.6% collectives is ~250 µs/token, worth ~10% if eliminated
entirely `[inferred]`.

### What the literature offers for this regime

Honestly: not much, because almost nobody publishes on single-stream MoE latency on
datacentre GPUs with the whole model resident. What exists:

- **Routing has temporal locality, which is exploitable.** Mixtral's own routing analysis
  finds consecutive-token repetition of the first-choice expert at ~14% (layer 0), ~27–28%
  (layer 15), ~20–27% (layer 31) against a 12.5% random baseline; for first-or-second choice,
  ~47–50% / ~62–67% / ~44–53% against a ~46% baseline `[verified, arXiv:2401.04088 §5]`.
  Higher layers are markedly stickier. Notably they find *no* domain specialisation — routing
  is syntactic, not topical `[verified]`.
- **Within a single request, expert reuse is heavily skewed; across requests it is not.**
  MoE-Infinity: "fewer than 5% of experts are repeatedly activated when decoding tokens for
  a single request" in 100+-expert models, with some experts activated 7× more than others —
  but "after processing multiple requests, the skew disappears"
  `[verified, arXiv:2401.14361]`. For a *cache* this is the whole game. For us, with
  everything resident, it means an L2-residency or weight-layout optimisation targeting the
  ~5% hot set could plausibly matter, and that it must be per-request, not global.
- **Router lookahead prediction works one layer ahead at ~90% top-k accuracy.** PROBE's
  gate-initialised lookahead predictor "distills the routing logic of the target layer" using
  frozen router parameters plus a lightweight residual MLP applied to the *previous* layer's
  hidden states, trained by online distillation against the true router. It reports ≈90%
  top-K accuracy with "Top-Half-K Hit-Rate and 2×Top-K Recall both approach 100%"
  `[verified, arXiv:2602.00509]`. FreeBalance gets similar leverage from "cross-layer
  similarities in hidden representations within residual networks"
  `[reported, arXiv:2608.14205]`.
- **The kernel shape problem is real and has a published answer.** DeepGEMM's masked layout
  and Alibaba's static-batching-into-one-kernel framework (91%/95% of peak tensor core on
  H800/H20 `[reported, arXiv:2501.16103]`) are the two production-grade approaches to
  variable per-expert token counts under CUDA graphs.

### What this implies we should do

At C=1, per MoE layer, the serialisation chain is: `attention out → router GEMM → topk →
build index/mask → grouped GEMM (gate/up) → activation → grouped GEMM (down) → all-reduce`.
Three of those steps are pure latency with no arithmetic. The wins available:

1. **Shared expert on a concurrent stream** — it depends only on the layer input, so it can
   run under the router + index build `[inferred, mechanism per arXiv:2412.19437 deployment]`.
2. **Predicted routing to start work early** — PROBE's predictor gives layer `L+1`'s top-k
   from layer `L`'s hidden state at ~90%. Under TP8 there is no dispatch to start, but the
   *index/mask construction* and the grouped-GEMM tile schedule could be built speculatively
   and validated against the true router, converting a serialisation into a comparison
   `[inferred]`.
3. **One all-reduce per layer, not two, and make it one-shot.** At 17.9 KB payload the
   collective is latency, so NVLink multimem/NVLS one-shot reduction beats ring.
4. **Fuse.** Alibaba-style single-kernel MoE removes 3–5 launches per layer × 89 layers.

---

## 7. MoE × speculative decoding: the interaction nobody warned us about

| Title | Lab | Venue / year | Hardware | Headline result | In production? |
|---|---|---|---|---|---|
| [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052) | Franklin & Marshall + Meta Reality Labs | arXiv 2602.16052 (Feb 2026) | A100 80GB | 127-token tree activates **54 of 64** experts/layer on OLMoE; Mixtral 2.3× vs EAGLE 1.7× `[verified]` | Research |
| [AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding](https://arxiv.org/abs/2608.02989) | Liang, Chen, Mo, Wang, Li, Ma, Luk (Imperial + MSR) | arXiv 2608.02989 (Aug 2026) | **RTX PRO 6000 Blackwell**; RTX 5090 | **1.29× fully resident**, 2.06× offloaded; H2D traffic −73.6–77.1%; accuracy within 0.27 pp `[verified]` | Research |
| [XShare: Collaborative in-Batch Expert Sharing for Faster MoE Inference](https://arxiv.org/abs/2602.07265) | Vankov, Ivkin, Ulrich, Song, Khetan, Karypis (AWS) | arXiv 2602.07265 (Feb 2026) | not extracted | Expert activation −30%; peak GPU load −3× under EP; **+14% throughput in speculative decoding** `[reported]` | Research |
| [DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference](https://arxiv.org/abs/2607.24434) | Han | arXiv 2607.24434 (Jul 2026) | RTX 4090; Hexagon HTP v81 NPU | 1.45× decode; 84–87% acceptance; 86–88% prefetch hit `[verified]` | Research |
| [SpecMD: A Comprehensive Study On Speculative Expert Prefetching](https://arxiv.org/abs/2602.03921) | Hoang, Jaiswal, Samragh, Cho (Apple) | arXiv 2602.03921 (Feb 2026) | GPU (not extracted) | Study of prefetch strategies across Mixtral/DeepSeek-V2/OLMo-MoE/Phi-3/Qwen-MoE `[reported]` | Research |

### The core problem, and our numbers

Verifying `N` draft tokens activates the **union** of the experts each token routes to.
On a dense model, N tokens read the weights once — that is the entire economic basis of
speculative decoding. On a fine-grained MoE it is almost false.

Expected distinct experts for `N` tokens, top-8 of 256, uniform routing
`= 256 · (1 − (1 − 8/256)^N)` `[inferred]`:

| N draft tokens | distinct experts | vs N×8 serial | weight-traffic saving |
|---|---|---|---|
| 1 | 8.0 | 8 | — |
| 2 | 15.8 | 16 | 1.6% |
| **4 (our 3-1-4)** | **30.5** | 32 | **4.6%** |
| 5 | 37.6 | 40 | 6.1% |
| 8 | 57.4 | 64 | 10.3% |
| 16 | 102.0 | 128 | 20.3% |
| 32 | 163.3 | 256 | 36.2% |
| 64 | 222.4 | 512 | 56.6% |

**At our draft length the MoE reads essentially the same weight bytes as unspeculated
decoding.** Real routing is more concentrated than uniform, which *helps*: MoE-Spec measures
a 127-token tree activating 54/64 experts on OLMoE where uniform would predict ~64, and
notes "the top 32 of 64 experts capture 93% of routing weight" `[verified]`. AcceptMoE gives
the sharpest version of the asymmetry: on Qwen3-30B-A3B, a token-pruning baseline achieved
"74.7% fewer verified tokens than EAGLE-3 but only 32.5% fewer activated experts"
`[verified]` — cutting draft tokens buys you less than half as much expert traffic as you'd
expect.

Consequences for us `[inferred]`:

- Our measured 3.09× speculative gain is coming from attention, dense GEMM, DSA indexer and
  collective amortisation. The MoE term (19.4% of C=1 time) is amortising ~4%.
- Lengthening the draft beyond 4 will show returns capped by the MoE weight traffic growing
  near-linearly. The 3-1-4 configuration is probably near the right place already, and the
  ceiling on any deeper tree is lower than a dense-model intuition predicts.
- GLM-4.5 uses "the MoE layer as the MTP layer" `[verified, arXiv:2508.06471]`, so our
  *draft* model also pays MoE weight traffic per drafted token. Draft cost is not negligible.
- Under EP, speculative decoding *increases* peak rank load — XShare reports up to 3× peak
  GPU load reduction from its expert-sharing under EP, and explicitly frames batching and
  speculative decoding as things that "significantly amplify expert activation, eroding
  these efficiency benefits" `[reported, arXiv:2602.07265]`. Another reason EP is wrong for
  our latency deployment.

### AcceptMoE is the one to try, with eyes open

AcceptMoE is the most relevant because it is measured on Blackwell with **all experts
resident** — our configuration — and still finds 1.29× `[verified, arXiv:2608.02989]`.
Mechanism, three parts:

1. **Commitment-weighted demand.** Score each expert by the target router's probability,
   weighted by an offline-estimated probability that the draft position actually survives
   into the accepted output. Prior work aggregates uniformly across the tree; positions deep
   in a draft branch rarely survive and should not drive expert selection.
2. **Self-sizing expert sets.** Derive the eligible-set cardinality from the *effective rank*
   (entropy) of the demand distribution rather than a hand-set budget `B`. When routing is
   concentrated the set shrinks; when it is diffuse it grows.
3. **Residency-aware pruning.** Under offloading, prune low-demand non-resident experts based
   on GPU cache state, no learned predictor needed. (Inert for us.)

Evaluated on Qwen3-30B-A3B-Instruct, Qwen3-Coder-30B-A3B and GPT-OSS-120B across GSM8K,
MATH500, HumanEval, MBPP; mean accuracy within 0.27 pp `[verified]`.

MoE-Spec is the simpler predecessor: aggregate routing probabilities across the tree, take
top-B by accumulated importance, and truncate or substitute tokens routing outside the
budget. On A100 it gives Mixtral 2.3× (vs EAGLE 1.7×), Qwen3-30B-A3B 2.4× (vs 1.9×), OLMoE
2.5× (vs 2.1×), with "only a 1.4% reduction" in mean accepted length `[verified,
arXiv:2602.16052]`.

**The caveat that decides whether we ship this.** Both techniques change which experts the
*verifier* uses. The verifier is then not the target model, so the accept/reject step no
longer guarantees the output distribution equals the target model's. Standard speculative
decoding is exactness-preserving; this is not. Both papers measure the quality cost as small
(0.27 pp, 1.4%), and for many products that is fine — but it must be a deliberate decision,
and it should be gated behind a flag with a lossless default.

---

## 8. Offloading, honestly

| Title | Lab | Venue / year | Hardware | Headline result | Relevant to us? |
|---|---|---|---|---|---|
| [MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache](https://arxiv.org/abs/2401.14361) | Edinburgh (Xue, Fu, Lu, Mai, Marina) | arXiv 2401.14361 v3 | **RTX A5000 24GB**, PCIe 4.0 32 GB/s | 3.1–16.7× per-token latency vs vLLM/Ollama/DeepSpeed/BrainStorm `[reported]`; DeepSeek-V2-Lite 155 ms vs vLLM 485 ms TPOT `[verified]` | **No** — but its skew measurement is |
| [Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models](https://arxiv.org/abs/2402.07033) | UW + Tsinghua (Kamahori, Tang, Gu, Zhu, Kasikci) | ICLR 2025, arXiv 2402.07033 | Quadro RTX 6000 24GB / RTX 6000 Ada 49GB | Runs 90 GB Mixtral on 24 GB GPU >3 tok/s; 19.4×/8.2× vs DeepSpeed-MII / Mixtral-offloading `[reported]` | No |
| [Klotski: Efficient MoE Inference via Expert-Aware Multi-Batch Pipeline](https://arxiv.org/abs/2502.06888) | SYSU + PCL + Huawei | arXiv 2502.06888 | not extracted | Up to 85.12× throughput `[reported]` | No |
| [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/abs/2411.11217) | UC Berkeley (Cao, Liu, Griggs, … Zaharia, Stoica) | arXiv 2411.11217 | single **T4** | CGOPipe CPU-GPU-I/O pipelining + HRM roofline model `[reported]` | No |
| [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433) | SJTU + CUHK | arXiv 2411.01433 | edge devices | Token-level dynamic expert loading, layer prefetch, sequence cache `[reported]` | No |
| [ExpertFlow: Efficient MoE Inference via Predictive Expert Caching and Token Scheduling](https://arxiv.org/abs/2410.17954) | HKBU + HKUST + A*STAR | arXiv 2410.17954 | single-GPU | Transformer routing-path predictor estimating expert usage for *all* layers in one forward pass `[reported]` | Predictor idea only |
| [ExpertFlow: Adaptive Expert Scheduling and Memory Coordination for Efficient MoE Inference](https://arxiv.org/abs/2510.26730) | Shen, Chu, Zhang, Xiang, Wu, Zhang | arXiv 2510.26730 | not extracted | Adaptive prediction horizon from runtime bandwidth/feedback `[reported]` | No |
| [Fast MoE Inference via Predictive Prefetching and Expert Replication](https://arxiv.org/abs/2605.11537) | Iowa State | arXiv 2605.11537 (May 2026) | single A100 80GB | SRU predictor + replication; 82–96% GPU util, ~3× speed, 90–95% of baseline accuracy `[reported]` | No |

**Verdict: none of this applies to us, and the reason is one number.** 8 × 183 GB = 1.46 TB
of HBM. GLM-5.2 at NVFP4 is ~200–250 GB, at FP8 ~400–500 GB `[inferred]`. Kimi K2 at 1T
params is ~1 TB at FP8 and fits `[inferred]`. There is no memory pressure to relieve. Every
paper above optimises the ratio between a 24–80 GB GPU and a PCIe link; we have neither
constraint.

Two things are worth extracting anyway:

1. **MoE-Infinity's skew measurement** (§6): <5% of experts repeatedly activated *within a
   request*, skew vanishing across requests `[verified]`. That is a statement about routing,
   not about caching, and it holds regardless of where the weights live.
2. **The routing predictors.** ExpertFlow (2410.17954) predicts expert usage across *all*
   layers in a single forward pass; PROBE predicts one layer ahead at ~90%; MoEless fine-tunes
   gate networks selectively, noting "early layers have lower input similarity across gate
   networks, while later layers maintain higher input similarity" and targeting layers below
   an 80% accuracy threshold `[verified, arXiv:2603.06350]`. Repurposed from "prefetch weights
   over PCIe" to "start the index build / tile schedule / collective before the router
   resolves", this is a latency technique for a fully-resident system.

One genuine warning from the offloading literature: **Fiddler's core insight is that below a
break-even batch size it is cheaper to compute an expert on the CPU than to move its weights
to the GPU** — 300 MB of Mixtral expert weights at FP16 versus ~4 KB of activation per token
`[verified, arXiv:2402.07033]`. That is the same weight-vs-token trade as FasterMoE shadowing
and MoE-Prefill AsyncEP, appearing for the third time. It is the fundamental economic
relation of MoE systems and worth internalising: *move whichever is smaller, and the crossover
is `3·d_ff/2` tokens per expert.*

---

## 9. Overlapping the collective with compute

Ranked by applicability to us.

- **Two-Batch Overlap (TBO) / dual micro-batch.** DeepSeek uses "a dual-batch overlap
  strategy to hide communication costs" in prefill and "a 5-stage pipeline" in decode
  `[verified]`. SGLang's TBO "splits a single batch into two micro-batches, allowing
  computation and communication to overlap", giving 27–35% throughput in prefill and up to
  35% in decode with MTP `[reported, LMSYS]`; flag `--enable-two-batch-overlap`
  `[verified]`. **Requires ≥2 micro-batches, so it is structurally impossible at C=1.**
  SGLang's `--enable-single-batch-overlap` is the C=1-relevant sibling and is the flag we
  should be testing.
- **Ping-pong pipelining** (MegaScale-Infer) needs `m ≥ 3` micro-batches `[verified]`. Same
  disqualification.
- **Kernel-level fused overlap** (COMET, HyperParallel-MoE, Perseus). COMET's thread-block
  specialisation is the cleanest formulation and gives 86.5% comm hiding `[verified]`. It
  needs an expert GEMM long enough to hide the transfer behind, which C=1 does not provide.
  Perseus is a warning label rather than a technique for us: MoE megakernels regress up to
  10× when crossing nodes because proxy-based RDMA transports impose ordering fences that
  drain the NIC pipeline `[reported, arXiv:2605.00686]` — irrelevant intra-node, decisive if
  we ever scale out.
- **DeepEP hooks / `EventOverlap`.** Only meaningful under EP.
- **Small-message collective latency.** This is our actual problem and the MoE literature
  does not address it; it lives in the collectives literature. The vLLM knobs
  `flashinfer_nvlink_one_sided` / `flashinfer_nvlink_two_sided` `[verified, vLLM docs]`
  and TensorRT-LLM's custom all-reduce are the production instances.

---

## What is NOT worth it

Techniques that look good in the literature and should not be built for our system, with
reasons.

1. **EPLB / redundant experts / dynamic expert re-placement — while we run TP8.** Under
   TP8 the per-rank MoE load is identical by construction; there is nothing to balance. Even
   under EP8, at C=1 the skew is multinomial noise over 40 draws, not expert popularity, and
   replication cannot fix noise. EPLB's measured 1.49×/2.54× came from EP32/EP144 at high
   token counts `[reported, LMSYS]`. **Do not port EPLB before confirming we are EP-sharded
   and measuring persistent (not transient) skew.**

2. **Expert offloading to host or NVMe, in any form.** 1.46 TB of HBM. See §8. This is the
   clearest "no" in the survey and it disposes of seven papers at once.

3. **Attention/expert disaggregation (MegaScale-Infer, Janus, FinDEP) on a single node.**
   Splitting 8 GPUs into two pools halves each phase's parallelism and adds a hop NVLink does
   not need. The ping-pong condition `m × Tf ≥ 2(Tf + Tc)` `[verified]` cannot be satisfied at
   C=1. Revisit only when we serve across nodes.

4. **Two-Batch Overlap for the latency objective.** Structurally impossible at C=1; it is a
   throughput technique. Keep it for the C=64 cost deployment, where LMSYS measured 27–35%
   `[reported]`.

5. **2DH / hierarchical all-to-all (Tutel, DeepSpeed-MoE).** These solve message
   fragmentation at 2048 GPUs `[verified]`. On 8 GPUs on one NVSwitch, all-to-all is already
   a single hop at 726 GB/s on SM100 `[verified, DeepEP]`. Zero upside.

6. **Wide EP (EP32+) as NVIDIA recommends.** NVIDIA's 1.8× per-GPU throughput result requires
   going wider than 8 GPUs — "around four experts per GPU, which requires 64 GPUs to
   accommodate the full 256 routed experts during decoding" `[verified, NVIDIA blog]`. We
   have 8. The recommendation does not apply, and adopting EP8 because "wide EP is the
   direction" would be a misreading.

7. **Capacity factors and token dropping.** Settled against by MegaBlocks (0.26 vs 0.15
   validation-loss reduction, 1.73× `[verified]`). Any capacity knob in our stack is dead
   weight.

8. **CPU-offloaded expert compute (Fiddler) and serverless expert placement (MoEless).**
   Both target deployments where GPU memory or GPU count is the scarce resource. Ours is
   neither. MoEless's 84% cost reduction is against a serverless baseline on A6000s
   `[reported]`.

9. **Rank-adaptive precision balancing (ReaLB), for now.** Elegant and directly implementable
   given our FP8+NVFP4 builds `[verified]`, but under TP8 there is no rank skew to correct,
   and having different ranks compute in different precisions makes numerics
   non-deterministic across ranks in a way that will make every future debugging session
   worse. Park it.

10. **Full COMET-style fused comm/compute kernels.** ~1.7× end-to-end on 8×H800 at training
    scale `[verified]`, but the whole win is hiding all-to-all behind expert GEMM, and at C=1
    we have neither a large all-to-all (TP) nor a long GEMM. Very high engineering cost for a
    C=64-only payoff.

11. **Blindly increasing EAGLE draft depth.** §7's table: the MoE weight traffic grows nearly
    linearly in draft length up to N≈16, so the usual "longer draft is nearly free" intuition
    from dense models is wrong here `[inferred]`.

---

## Sources

Every URL below was fetched during this survey.

**Architecture**
- GShard — https://arxiv.org/abs/2006.16668
- Switch Transformers — https://arxiv.org/abs/2101.03961 (full text via https://ar5iv.labs.arxiv.org/html/2101.03961)
- Mixtral of Experts — https://arxiv.org/abs/2401.04088 (routing analysis via https://arxiv.org/html/2401.04088v1)
- DeepSeekMoE — https://arxiv.org/abs/2401.06066
- Auxiliary-Loss-Free Load Balancing — https://arxiv.org/abs/2408.15664
- DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437 (https://arxiv.org/html/2412.19437v2)
- Qwen3 Technical Report — https://arxiv.org/abs/2505.09388 (https://arxiv.org/html/2505.09388v1)
- Kimi K2 — https://arxiv.org/abs/2507.20534
- GLM-4.5 — https://arxiv.org/abs/2508.06471 (https://arxiv.org/html/2508.06471v1)
- DeepSeek-V3.2 — https://arxiv.org/abs/2512.02556
- DeepSeek-V4 — https://arxiv.org/abs/2606.19348

**Kernels and dropless MoE**
- DeepSpeed-MoE — https://arxiv.org/abs/2201.05596
- FasterMoE — https://doi.org/10.1145/3503221.3508418 ; https://ppopp22.sigplan.org/details/PPoPP-2022-main-conference/20/FasterMoE-Modeling-and-Optimizing-Training-of-Large-Scale-Dynamic-Pre-Trained-Models
- Tutel — https://arxiv.org/abs/2206.03382 (https://ar5iv.labs.arxiv.org/html/2206.03382)
- MegaBlocks — https://arxiv.org/abs/2211.15841 (https://ar5iv.labs.arxiv.org/html/2211.15841)
- ScatterMoE — https://arxiv.org/abs/2403.08245
- Static Batching of Irregular Workloads — https://arxiv.org/abs/2501.16103
- DeepGEMM — https://github.com/deepseek-ai/DeepGEMM

**Expert parallelism and communication**
- DeepEP — https://github.com/deepseek-ai/DeepEP
- COMET — https://arxiv.org/abs/2502.19811 (https://arxiv.org/html/2502.19811v3)
- MegaScale-Infer — https://arxiv.org/abs/2504.02263 (https://arxiv.org/html/2504.02263v2)
- NCCL EP (NVIDIA) — https://arxiv.org/abs/2603.13606
- UBEP — https://arxiv.org/abs/2607.06202
- Perseus / hidden serialization — https://arxiv.org/abs/2605.00686
- HyperParallel-MoE — https://arxiv.org/abs/2605.23764
- Semantic Parallelism (Sem-MoE) — https://arxiv.org/abs/2503.04398
- MoE-Prefill — https://arxiv.org/abs/2605.02960
- Lina — https://arxiv.org/abs/2210.17223
- Janus (disaggregation, 2025) — https://arxiv.org/abs/2512.13525
- FinDEP — https://arxiv.org/abs/2512.21487
- ELDR — https://arxiv.org/pdf/2607.00466

**Load balance and placement**
- EPLB — https://github.com/deepseek-ai/EPLB/blob/main/README.md
- DeepSeek V3/R1 Inference System Overview — https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- LMSYS large-scale EP on 96 H100 — https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- NVIDIA wide EP on NVL72 — https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/
- NVIDIA GB200 NVL72 + Dynamo for MoE — https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/
- UltraEP — https://arxiv.org/abs/2606.04101 (https://arxiv.org/html/2606.04101)
- PROBE — https://arxiv.org/abs/2602.00509 (https://arxiv.org/html/2602.00509v2)
- ViBE — https://arxiv.org/abs/2606.00735
- EasyBalance — https://arxiv.org/abs/2608.07964
- FreeBalance — https://arxiv.org/abs/2608.14205
- Director — https://arxiv.org/abs/2607.08782
- MoETuner — https://arxiv.org/abs/2502.06643
- MoEShard / Expert Sharding — https://arxiv.org/abs/2503.08467
- ReaLB — https://arxiv.org/abs/2604.19503 (https://arxiv.org/html/2604.19503v1)
- Patterns behind Chaos — https://arxiv.org/abs/2510.05497
- Scaling Multi-Node MoE Inference Using Expert Activation Patterns — https://arxiv.org/abs/2604.23150
- MoEless — https://arxiv.org/abs/2603.06350 (https://arxiv.org/html/2603.06350v1)

**Speculative decoding × MoE**
- MoE-Spec — https://arxiv.org/abs/2602.16052 (https://arxiv.org/html/2602.16052)
- AcceptMoE — https://arxiv.org/abs/2608.02989 (https://arxiv.org/html/2608.02989)
- XShare — https://arxiv.org/abs/2602.07265
- DraftExpert — https://arxiv.org/abs/2607.24434 (https://arxiv.org/html/2607.24434v1)
- SpecMD — https://arxiv.org/abs/2602.03921

**Offloading**
- MoE-Infinity — https://arxiv.org/abs/2401.14361 (https://arxiv.org/html/2401.14361v3)
- Fiddler — https://arxiv.org/abs/2402.07033 (https://arxiv.org/html/2402.07033v2)
- Klotski — https://arxiv.org/abs/2502.06888
- MoE-Lightning — https://arxiv.org/abs/2411.11217
- HOBBIT — https://arxiv.org/abs/2411.01433
- ExpertFlow (predictive caching) — https://arxiv.org/abs/2410.17954
- ExpertFlow (adaptive scheduling) — https://arxiv.org/abs/2510.26730
- Predictive Prefetching and Expert Replication — https://arxiv.org/abs/2605.11537

**Engine documentation**
- vLLM expert parallel deployment — https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- SGLang expert parallelism — https://docs.sglang.io/advanced_features/expert_parallelism.html
