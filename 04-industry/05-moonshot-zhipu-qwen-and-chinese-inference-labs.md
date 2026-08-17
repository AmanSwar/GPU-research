# Moonshot, Z.ai/Zhipu, Qwen, MiniMax and ByteDance: serving-layer innovation from Chinese labs

## What this is

A competitive-intelligence sweep of published serving-layer engineering from the Chinese
frontier labs, mined for mechanisms we can reimplement on 8×B200 running GLM-5.2 and,
next, Kimi K3 / Qwen3.8 / DeepSeek V4.

Scope and method: every claim below was read at a URL given in Sources. Where a PDF was
involved I extracted the text and quote the actual table or paragraph rather than a
summariser's paraphrase — several summariser passes produced plausible-looking but
fabricated numbers (notably for the Mooncake evaluation), and those were discarded.
WebSearch quota for this session was exhausted early, so navigation was done via the
arXiv API, GitHub's REST API, and direct fetches; a few things I would normally have
found by search are flagged as unsourced rather than guessed at.

Labels used throughout:
- **[verified]** — I read it at the URL given.
- **[reported]** — the company claims it; not independently reproduced.
- **[inferred]** — my arithmetic or reasoning on top of verified facts.
- **[unverified]** — I could not source it. Treat as absent.

The single most important thing in this document is the TileRT section. TileRT is not a
marketing artifact; it is a real, installable, binary-only wheel that runs on exactly our
hardware, and it has been independently benchmarked. It also has a structural weakness we
can exploit.

---

## Bottom line for our system

Ranked by (expected effect on our two objectives) ÷ (difficulty). Our measured C1 profile
is the target: dense GEMM 37.1%, collectives 19.6% (47% of which is rank arrival skew →
9.2% of total wall clock), MoE expert GEMMs 19.4%, attention 10.9%, DSA indexer 5.8%.

### The roofline we are actually chasing

TileRT's own engineering blog states that GLM-5.1 reads **~42 GB of activated weights per
decode step** and that an 8×H200 node's 38 TB/s aggregate bandwidth therefore implies a
"theoretical upper limit approaching 1000 token/s" [verified —
tilert.ai/blog/speed-as-the-next-scaling-law-zh.html].

Redo that arithmetic for our box [inferred]: 8×B200 HBM3e ≈ 8 TB/s each = **64 TB/s
aggregate**. 64,000 GB/s ÷ 42 GB = **~1,520 tok/s** non-speculative roofline; a 0.66 ms
TPOT floor. We are at 365 tok/s / 2.74 ms TPOT = **24% of roofline**. TileRT's published
500 tok/s on GLM-5-FP8 is **33%**. The headroom is real and it is not in exotic places —
it is in the 37.1% dense GEMM and the 19.6% collectives.

Second piece of arithmetic that reframes the target [inferred]: with EAGLE 3-1-4 at
acceptance length ~3, each forward step emits ~3 tokens for one 42 GB weight read, so a
perfectly bandwidth-bound speculative decoder would run at ~3× the non-speculative
roofline. Our 365 tok/s is therefore ~8% of the *speculative* ideal. The gap is
overwhelmingly launch-boundary and synchronisation overhead, which is precisely the thing
TileRT's persistent Engine Kernel is designed to delete.

### Ranked steal list

**1. Fuse the residual/AttnRes merge + RMSNorm into the TP all-reduce epilogue.**
Mechanism: Kimi K3 states plainly that "the merging of the AttnRes output with its
partial-sum update, together with the subsequent RMSNorm, is fused into the preceding TP
all-reduce, eliminating a dedicated kernel for the intra-block phase" [verified —
arXiv:2607.24653 §5.4.2].
Expected effect: attacks the 19.6% collectives directly, plus removes a full kernel launch
and a round-trip to HBM per layer × 78 layers. Difficulty: **medium** — needs a custom
all-reduce with a programmable epilogue; on B200 a one-shot NVLS/multimem all-reduce with
a fused epilogue is the natural shape.

**2. Token-centric (WarpDecode-style) MoE decode kernel to replace tile-centric grouped GEMM.**
Mechanism: at C1 with 8 routed experts under TP8, the expert group GEMMs "reduce to
memory-bound streaming of weight matrices — a regime for which conventional tile-centric
kernels are poorly suited due to their compute-oriented design and preprocessing
overheads." Kimi K3's fix: **one warp per output neuron**, streaming its weights straight
from HBM; **subdivide each warp into finer-grained lane teams**, each handling a disjoint
subset of experts, followed by a warp-wide reduction; and **permute the weight layout
offline** at one-time preprocessing cost to cut runtime dequantisation overhead [verified
— arXiv:2607.24653 §5.4.2, crediting Cursor's WarpDecode].
Expected effect: this is aimed squarely at our 19.4% MoE expert GEMMs, and the offline
weight permutation also helps our NVFP4 build's dequant cost. Difficulty: **medium-high**
(new kernel) but it is the single most concretely specified win in the corpus.

**3. Mixed-precision by module, not uniform: NVFP4 experts, FP8 everything else.**
Mechanism: Xiaomi MiMo + TileRT "selectively quantize only the MoE Experts to FP4 while
preserving original precision for all other modules", applied as MXFP4 with QAT, described
as "a deliberate, joint engineering trade-off calculated based on hardware physical
boundaries" [verified — mimo.xiaomi.com/blog/mimo-tilert-1000tps]. Kimi K3 independently
does the same: MoE expert weights MXFP4, activations MXFP8, while "all non-expert
components (attention projections, latent MoE projections, shared experts, and MoE
routers) remain in higher precision", with QAT through both SFT and RL [verified —
arXiv:2607.24653 §4.1.4]. Zhipu's Ascend port does the INT analogue: attention and MLP
W8A8, MoE experts W4A8 [verified — arXiv:2602.15763 §5].
Expected effect: three independent teams converged on the same boundary. If our NVFP4
build quantises uniformly we are paying accuracy for nothing on the ~40% of activated
bytes that are not expert weights, and paying dequant latency in the attention path.
Difficulty: **low-medium** — mostly a quantisation-recipe change, no new kernels.

**4. Decompose the TP all-reduce into reduce-scatter → compute → all-gather, with the
kernel *inside*.**
Mechanism: for Block AttnRes prefill, Kimi K3 adopts sequence parallelism for activations
where "the TP all-reduce is decomposed into a reduce-scatter and an all-gather, with the
intra-block kernel inserted between the two collectives, operating on the sequence-sharded
hidden states so that the block representations of each token are materialized on exactly
one rank" [verified — arXiv:2607.24653 §5.4.2].
Expected effect: attacks TTFT (we are at 189 ms) and eliminates redundant per-rank
materialisation. Difficulty: **medium**.

**5. Fuse the all-gather into the GEMM epilogue using `multimem` store instructions.**
Mechanism: Kimi K3's latent-MoE path does three things — (a) fuse the latent
down-projection with the MoE router into a single GEMM; (b) shard latent weights across
ranks and "fuse the output all-gather into the GEMM epilogue using multimem store
instructions"; (c) overlap the resulting communication with the shared-expert computation
[verified — arXiv:2607.24653 §5.4.2].
Expected effect: `multimem.st` is NVLS/NVSwitch multicast — native on our NV18 NVLink5
fabric and under-used by SGLang. Directly attacks the 19.6% collectives and specifically
the arrival-skew component, because a multicast store has no gather-side rendezvous.
Difficulty: **medium**. This is the highest-leverage B200-specific item here.

**6. GPU specialisation: give the DSA indexer its own rank.**
Mechanism: TileRT extends warp specialisation to whole devices — "warp specialization →
block specialization → GPU specialization". For GLM-5.1 attention specifically they split
into heterogeneous workers: **GPU 0 = sparse-index worker (Top-K selection, sparse index
construction); GPUs 1–7 = MLA workers (RMSNorm, GEMM, Flash Sparse Attention, AllReduce)**
[verified — tilert.ai/blog/speed-as-the-next-scaling-law.html and the -zh version].
Expected effect: our indexer is 5.8%. Note that GLM-5.2's `index_topk_freq = 4` [verified
— config.json] means the index is only rebuilt every 4th decode step, so a permanently
dedicated rank would idle 75% of the time; the transferable form is a **pipelined**
split where rank 0 computes step *t+4*'s index while ranks 1–7 run step *t*'s MLA.
Difficulty: **high** (breaks TP8 symmetry), but it is the one architectural idea here that
nobody else has published.

**7. Train the EAGLE draft against acceptance rate directly, not KL.**
Mechanism: Kimi K3 fine-tunes its pre-trained MTP layer into an EAGLE-3-style draft (the
MTP layer "mirrors the structure of a backbone block" and EAGLE-3's draft "comprises a
single decoder layer whose structure matches the MTP layer"), target frozen, draft
unrolled **seven steps** during training, and — critically — "since minimizing the
conventional KL-divergence surrogate does not guarantee maximizing this rate for a
capacity-limited draft model, we directly optimize the likelihood-based LK loss, the
negative logarithm of the acceptance rate itself", L_LK = −log Σ_x min(p(x), q(x)), at
temperature 1 with no auxiliary cross-entropy term [verified — arXiv:2607.24653 §4.1.4,
citing arXiv:2602.23881]. The draft is fine-tuned under the same QAT config as the target.
Expected effect: acceptance length is a pure multiplier on our 365 tok/s. Difficulty:
**medium** (a training change, not a serving change), but it is the cheapest multiplier
available and applies to every model we serve.

**8. Block-masked parallel drafting (DFlash) instead of autoregressive drafting.**
Mechanism: Xiaomi MiMo's DFlash replaces autoregressive drafting with block-level masked
parallel prediction — the draft "fills an entire block of masked positions in a single
forward pass", block size capped at 8 to bound verification cost; the draft uses **sliding
window attention exclusively**, which takes "per-prediction compute from context-length-linear
to constant"; trained with Muon plus self-distillation. Measured average accepted lengths:
**coding 6.30 (max 7.14), math/reasoning 5.56, agent 4.29** [verified —
mimo.xiaomi.com/blog/mimo-tilert-1000tps]. They also note the motivating problem: "each
additional LM Head in traditional MTP architectures independently incurs dozens of
microseconds of execution overhead."
Expected effect: acceptance of 6.3 vs our 3-1-4 tree would be transformative. Caveat:
these are Xiaomi's own numbers on their own model, not reproduced [reported]. Difficulty:
**high** — new draft architecture and training run.

**9. Heterogeneous PD disaggregation: a different engine for prefill than for decode.**
Mechanism: TileRT v0.1.5 runs **stock, unpatched vLLM for prefill and TileRT for decode**
behind one OpenAI endpoint, via vLLM's V1 `KVConnector` (`kv_connector_module_path`
= `tilert.pd_vllm.prefill_connector`), RDMA handoff over NIXL or Mooncake, fp8 KV on both
ends (`--kv-cache-dtype fp8_ds_mla` on prefill, `fp8` on decode; mismatch is rejected at
the connector handshake), and MTP-aware handoff where "the prefill populates the
draft-layer KV that decode-side speculation resumes from" [verified — TileRT README and
tilert.ai/blog/tilert-vllm-disaggregation.html]. Under `MultiConnector` a shared prefill
pool can feed a TileRT decode pool and a native vLLM decode pool simultaneously, with a
static `_claim` filter routing marked requests.
Expected effect: lets us optimise decode for single-stream latency without regressing
TTFT, and lets latency-critical and throughput traffic share one prefill pool — which is
close to a direct answer to our two-objective problem. Difficulty: **medium**; the
integration pattern is fully documented and copyable.

**10. Sub-block prefix caching: decouple hash granularity from physical block size.**
Mechanism: Kimi K3 runs "prefix hashing on fine hash blocks (e.g., 512 tokens) inside MLA
pages, while the physical block remains the coarse allocation unit", with chained hashes
so "matching an endpoint certifies the whole prefix up to it", and partially-filled pages
registered under the chained hash of their last complete hash block [verified —
arXiv:2607.24653 §5.4.1].
Expected effect: AA sends ~10k input tokens repeatedly over a 72-hour window. Any prefix
hit is pure TTFT. Difficulty: **medium**.

**11. Deterministic top-k in the DSA indexer — and a warning about SGLang's.**
Mechanism / negative result: the GLM-5 team reports that "compared with the
non-deterministic CUDA-based top-k implementation used in **SGLang's DSA Indexer**,
directly using the naive `torch.topk` is slightly slower but deterministic… In contrast,
other non-deterministic top-k operators (e.g., CUDA or TileLang implementations) caused
drastic performance degradation during RL after only a few steps, accompanied by a sharp
drop in entropy" [verified — arXiv:2602.15763 §3]. They use `torch.topk` throughout RL and
freeze indexer parameters by default.
Expected effect: this is an RL finding, not an inference finding, so it does **not** mean
our decode is wrong. But we fork SGLang, our indexer is 5.8% of C1, and we are being scored
on a **P50 over a trailing 72 hours** — non-determinism in the top-k is a source of
run-to-run output variance that could move benchmark scores. Worth measuring before
optimising the kernel further. Difficulty: **low** (a measurement).

**12. Overlap the D2H sampling copy with next-step preparation.**
Mechanism: Zhipu's Ascend work "implemented a mechanism to overlap the 'Device-to-Host'
(D2H) sampling copies with the preparation of the next decode step, effectively eliminating
scheduling bubbles" [verified — arXiv:2602.15763 §5]. Same family: Kimi K3 launches the
inter-block AttnRes kernel "on a side stream so that it overlaps with independent
computation on the main stream" [verified — arXiv:2607.24653 §5.4.2].
Expected effect: at TPOT 2.74 ms a D2H round trip is a meaningful fraction. Difficulty:
**low**. Check whether our SGLang fork already does this (upstream has async scheduling;
verify it survives our fork).

### Things I would *not* copy

- **ByteDance Flux** — the compute/comm overlap library builds for SM80/SM89/SM90 only. No
  SM100 support in the README [verified — github.com/bytedance/flux]. The *idea* transfers;
  the code does not.
- **RTP-LLM's headline wins** — all throughput- and fleet-oriented (model loading, cache
  reuse, prefill machine count). Almost nothing there moves single-stream TPOT.
- **Mooncake's chunked pipeline parallelism** — designed to cut TTFT for long context by
  spreading one request across *multiple nodes*. We have one node. The Transfer Engine is
  still worth having as a PD transport.

---

## Z.ai / Zhipu and TileRT

### What they run

Z.ai serves the GLM series. Model lineage on HuggingFace `zai-org`: GLM-4.5 (355B),
GLM-4.7, GLM-5 (744B/754B), GLM-5.1 (754B), **GLM-5.2 (753B)**, each with an FP8 variant.
No NVFP4 or "Air" variant of GLM-5.2 is published by zai-org [verified —
huggingface.co/zai-org]. Our NVFP4 build is therefore ours, not theirs.

**GLM-5 architecture, from Table 10 of the tech report** [verified — arXiv:2602.15763
Appendix A]:

| Field | GLM-4.5 | GLM-5 |
|---|---|---|
| Total parameters | 355B | **744B** |
| Activated parameters | 32B | **40B** |
| Dense layers | 3 | 3 |
| MoE layers | 89 | 75 |
| MTP layers | 1 | **1** |
| Hidden dim | 5120 | **6144** |
| Dense intermediate dim | 12288 | 12288 |
| MoE intermediate dim | 1536 | **2048** |
| QK head dim | 128 | **192** |
| V head dim | 128 | **256** |
| Q LoRA dim | – | **2048** |
| KV LoRA dim | – | **512** |
| Attention heads | 96 | **64** |
| KV heads | 8 | – |
| Indexer attention heads | – | **32** |
| Indexer head dim | – | **128** |
| Experts (total) | 160 | **256** |
| Routed experts | 8 | **8** |
| Shared experts | 1 | 1 |
| Vocabulary | 151552 | **154880** |

Parameter counts include MTP layers but exclude embeddings and the output layer.

**GLM-5.2 `config.json`** confirms the family carries forward, with 78 total layers
[verified — huggingface.co/zai-org/GLM-5.2/raw/main/config.json]:

```
num_hidden_layers: 78          hidden_size: 6144         vocab_size: 154880
n_routed_experts: 256          num_experts_per_tok: 8    n_shared_experts: 1
moe_intermediate_size: 2048    num_attention_heads: 64
qk_nope_head_dim: 192          qk_rope_head_dim: 64      v_head_dim: 256
q_lora_rank: 2048              kv_lora_rank: 512
index_topk: 2048               index_n_heads: 32         index_head_dim: 128
index_topk_freq: 4             num_nextn_predict_layers: 1
max_position_embeddings: 1048576                          rope_theta: 8000000
```

Two things to internalise. **`index_topk = 2048`**: the lightning indexer selects 2048 KV
entries per query. The GLM-5 report makes the RL consequence explicit — "the k = 2048 used
by the indexer is much larger than the k typically used in MoE, and storing all these
indices would incur enormous storage costs as well as significant communication overhead"
[verified]. **`index_topk_freq = 4`**: the index is refreshed every 4 steps, which is why
our indexer is only 5.8% of C1 and why a permanently-dedicated indexer rank would idle.

### Why DSA, in their words

The GLM-5 report contains a genuinely useful architecture-selection negative result. They
compared interleaved SWA, search-based layer selection, GDN, and SimpleGDN against DSA:

> "Naively interleaved sliding window attention (SWA) causes catastrophic degradation on
> long-context tasks (e.g., −30.35 on RULER@128K)… Linear attention variants such as GDN
> further improve quality but at the cost of additional parameters; SimpleGDN strikes the
> best balance by maximally reusing pre-trained weights. Nevertheless, all of these methods
> incur an inherent accuracy gap on fine-grained retrieval tasks — up to 5.69 points on
> RULER@128K and 7.33 on RepoQA@128K… In contrast, DSA is lossless by construction: its
> lightning indexer achieves token-level sparsity without discarding any long-range
> dependencies, enabling application to all layers with no quality degradation."
> [verified — arXiv:2602.15763 §2.1]

Their DSA recipe, if we ever need to re-derive an indexer: (i) warmup training the indexer
alone for **1,000 steps at batch size 16** with the base model frozen, LR from 5e-3 down to
2e-4; (ii) joint co-training of model and indexer on **150B tokens** at a constant 1e-5.
RULER at 128K: baseline GLM-4.7-Flash 79.21 → warmup-only 71.35 → full DSA 78.86
[verified — Table 6].

Note the strategic tension: Zhipu argues DSA is the right sparsity because it is lossless,
while Moonshot (K3) and Alibaba (Qwen3.8) both shipped 3:1 hybrid *linear* attention. See
the cross-cutting section.

### TileRT — the engine to beat

TileRT is developed by the tile-ai org (the TileLang/TileScale people), in production at
Z.ai for GLM-5.1-highspeed and at Xiaomi for MiMo V2.5 Pro UltraSpeed.

#### Design, as published

The core claim is a **persistent Engine Kernel**: instead of per-operator launches,

> "the host launches only once, execution remains resident on the GPU, much of runtime
> orchestration moves into compile time."
> [verified — tilert.ai/blog/speed-as-the-next-scaling-law.html]

Their diagnosis of why conventional engines under-deliver is worth quoting because it
matches our own profile:

> "GPU utilization was not particularly low… yet token latency remained stubbornly high.
> The GPU was not short on compute. Compute was trapped between execution boundaries."

Mechanisms named:

1. **Static expansion into a persistent Engine Kernel** ahead of time; execution
   reorganised into a tile-level pipeline where compute, communication, and async IO
   "continuously progress inside the GPU".
2. **Warp/block specialisation** — "Different warp groups assume different
   responsibilities: asynchronous data movement, tensor computation, communication
   overlap", replacing `load → barrier → compute → barrier`.
3. **Register / shared-memory / L2 dataflow** — "Intermediate results no longer repeatedly
   spill back into global memory, but continue flowing forward through registers, shared
   memory, and L2 cache."
4. **Compute↔communication pipelining** rather than `compute → sync → compute`.
5. **GPU specialisation** — the heterogeneous worker split described in steal-item 6. Their
   summary line: "An entire attention layer corresponds to only a single Engine Kernel
   launch. Broadcasts, reductions, and synchronization increasingly execute directly inside
   the tile-level execution flow."

The Chinese version adds the roofline framing: 8×H200 ≈ 38 TB/s aggregate, GLM-5.1 decode
activates ~42 GB, so ~1000 tok/s is the theoretical ceiling, while "real systems often
deliver only a few dozen token/s" [verified —
tilert.ai/blog/speed-as-the-next-scaling-law-zh.html].

The compiler techniques are explicitly **not** open-sourced: "the underlying compiler
techniques will be gradually shared with the community as they are integrated into TileLang
and TileScale" [verified — README].

#### Published numbers

| Date | Version | Model | Hardware | Config | Claim | Label |
|---|---|---|---|---|---|---|
| 2025-11-22 | v0.1.0-alpha.1 | DeepSeek-V3.2-Exp, no quant | 8×B200, CUDA 12.9 | 1K in / 1K out, **BS=1**, vs SGLang 0.5.5 & vLLM 0.11.0 | "significantly outperforms existing inference systems" (chart only, no number in text) | [reported] |
| 2025-12-23 | v0.1.1 | — | 8×B200 | — | **~35% latency reduction** vs v0.1.0; README milestone says "3~4× speedup over baseline" | [reported] |
| 2026-01-26 | v0.1.2-alpha.1 | DeepSeek-V3.2, **mtp=3** | 8×B200 | synthetic | **590 tok/s** synthetic; **~440 tok/s on real generation tasks** | [reported] |
| 2026-02-14 | v0.1.3 | GLM-5-FP8 / DeepSeek-V3.2 | 8×B200 | — | **up to 500 tok/s GLM-5-FP8**, **up to 600 tok/s DeepSeek-V3.2**; context 200K / 160K | [reported] |
| 2026-05-22 | production | **GLM-5.1-highspeed** | — | Zhipu MaaS | **400 tok/s** in production | [reported] |
| 2026-06-08 | — | MiMo-V2.5-Pro-UltraSpeed, **1T params** | single 8-GPU node | MXFP4 experts + QAT, DFlash | **1000+ tok/s** (demos ~1200) | [reported] |
| 2026-06-01 | v0.1.4 | DSV3.2 + GLM-5 | 8×B200 | — | "major decoding speed-up", no number given | [reported] |
| current | v0.1.5 | **GLM-5.1-FP8** | 8×B200 | out 1K, **in 1K–192K** | chart compares no-MTP vs **MTP at avg acceptance 3.2** vs peak at acceptance 4.0 | [reported] |

Two disclosures matter more than the headlines. First, the **590 synthetic / ~440 real**
split at v0.1.2 is an unusually honest admission that synthetic acceptance does not survive
real data — the same gap we should expect between a synthetic bench and our 365 tok/s on
real data. Second, the v0.1.5 README chart is explicitly labelled **"average acceptance
length 3.2"**, so the ~500 tok/s class number is a *with-MTP* number, not a raw decode
number.

#### Independent measurement — and the crack in the armour

SemiAnalysis's InferenceX benchmarked TileRT on GLM-5 (744B) FP8 on a single 8×B200 HGX
node [verified — inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia]:

| Scenario | TileRT | Comparison |
|---|---|---|
| 1k in / 1k out, FP8, BS=1 | **494.2 tok/s/user** | 3.0× the best FP8-on-B300-with-MTP result (136.3); 1.9× the best conventional FP4 result (256.3) |
| **8k in / 1k out** | **340 tok/s/user** | — |
| 1k/1k end-to-end decode tail | 3.01 s | vs 6.54 s for best NVFP4+MTP (4.5× better) |
| Throughput cost at 340 tok/s/user | 160.4 tok/s/GPU | vs 240 tok/s/GPU for conventional engines |

**This is the most consequential number in the whole report.** TileRT falls from 494 to
**340 tok/s/user when input goes from 1k to 8k**. Artificial Analysis uses ~10k input
tokens. Current board leaders for this model sit at 330–336 tok/s. Those two facts line up
almost exactly [inferred]: at AA's input length, a TileRT-class engine lands right where
the leaderboard already is, and the "500 tok/s" headline simply does not survive 10k of
context. Our 365 tok/s on real data may already be at or ahead of TileRT's 8k/1k figure —
we should confirm our number's input length before assuming we are behind.

The reviewers also record hard structural limits [verified]:
- **"TileRT as of publication also serves only one in-flight request per decode node."**
  Batch size 1, period. TileRT cannot serve AA's 10-parallel scenario on one node, and it
  has no answer at all to our C64 / 40.8k tok/s aggregate objective.
- Aggregate throughput is *worse* than conventional engines (160.4 vs 240 tok/s/GPU), so
  cost-per-token is worse.
- "a tiny model catalog, hard-pinned dependencies"; "TileRT development is slow" because of
  per-model ahead-of-time compilation.

Strategic read [inferred]: TileRT is a latency-only point solution with a per-model compile
cost and no concurrency story. We do not need to beat it on its own 1k/1k benchmark; we
need to beat it at 10k input while also holding C16–C64, which is a fight it is structurally
unequipped for. Its ceiling is also our ceiling — the persistent-megakernel idea is the
right idea, and it is the one we should port, not the product.

#### What is actually usable

- **Wheel**: `pip install tilert==0.1.5.post1`. Hard-pinned: 8×B200, CUDA 13.2 driver,
  Python 3.12, `torch==2.11.0+cu130`, `transformers 4.46.3`, `tokenizers 0.20.3`,
  manylinux_2_28. Docker: `ghcr.io/tile-ai/tilert:cu132-latest` [verified — README].
- **Closed**: the two backend libraries `libtilert_dsv32.so` and `libtilert_glm5.so` are
  pre-compiled binaries; no source. Only one may be loaded per Python process via
  `tilert.load_backend(model_type)` — DeepSeek-V3.2 and GLM-5 cannot coexist in one
  interpreter [verified].
- **Open (MIT)**: the CLI, the Python generator API, the weight converter, and all of
  `tilert.pd_vllm` (prefill connector, decode server, OpenAI router).
- **Weight converter** rewrites an HF checkpoint into 8 per-device shards with keys suffixed
  `*_dev_{0..7}` plus a fresh index; the original checkpoint is then unneeded. Note
  v0.1.4 "requires a weight conversion step to adapt the original weights to the updated
  runtime format" — i.e. the on-disk layout is a versioned part of the optimisation.
- **MTP** via `--with-mtp` / `with_mtp=True`; it prints `Accepted length: mean=2.77, min=1,
  max=4` [verified — README sample output]. That mean of 2.77 on a jokes prompt is a useful
  reality check against the 3.2 used in the marketing chart.
- **PD disaggregation launch commands** are given verbatim in the README for both
  topologies, including `--kv-cache-dtype fp8_ds_mla` on the vLLM prefill,
  `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`,
  `--enforce-eager`, `--gpu-memory-utilization 0.75`, `--max-seq-len 202752`, and the
  warning to pin NIXL to RDMA NICs via `UCX_NET_DEVICES=mlx5_1:1,mlx5_2:1,...` "otherwise
  UCX may pick the wrong interface" [verified].

The disaggregation blog gives the design rationale — "prefill is dominated by compute,
while decode is bound by memory bandwidth" — and the transport detail: state extraction
happens inside vLLM's forward window into a staging buffer, then is "immediately handed to
an independent background async sender", so "the forward pass itself never stalls waiting
on the network"; the wire operation is an "RDMA one-sided WRITE — no intermediate
serialization" into pre-registered GPU memory. Transferred payload is the compressed KV
cache, the **sparse-attention index caches**, and metadata [verified —
tilert.ai/blog/tilert-vllm-disaggregation.html].

### GLM-5.1-highspeed in production

Zhipu's docs state GLM-5.1-highspeed "模型输出速度达到 400 tokens/s", 200K context, 128K max
output, text-only, and — importantly — that it is "仅面向智谱开放平台**部分企业客户**定向开放"
(open only to *selected enterprise customers*). Capability is claimed unchanged from
GLM-5.1. Optimisation is described as spanning three levels: 推理引擎、调度系统与底层基础设施
(inference engine, scheduling system, underlying infrastructure), built jointly by Zhipu's
GLM team and the TileRT team [verified — docs.bigmodel.cn].

Consequence for our leaderboard chase [inferred]: because highspeed is enterprise-gated and
not on the public API, it is probably **not** what AA is measuring. The 330–336 tok/s board
leaders are the standard endpoint. We are chasing a public-API number, not the 400/500
figure.

### Zhipu's own inference engineering (the Ascend case study)

The GLM-5 report's §5 is the only place Zhipu documents its own serving stack in detail,
and it does so for Chinese silicon — but the techniques are hardware-agnostic [verified —
arXiv:2602.15763 §5]:

- **W4A8 mixed precision** to fit 750B on one Atlas 800T A3: attention + MLP W8A8 (INT8),
  MoE experts W4A8 (INT4), via `msModelSlim`, with **QuaRot** for outlier suppression and
  `Flex_AWQ_SSZ` for scaling calibration.
- **Lightning Indexer fusion**: "integrates score calculation, ReLU, and TopK operations
  into a single kernel, allowing the NPU to overlap computation with memory access." We
  should check whether our indexer path is three kernels or one.
- **Sparse Flash Attention**: "handles the selection of TopK tokens from the KV cache and
  sparse attention computation in parallel" — i.e. gather and attend are fused, not
  sequenced.
- **MLAPO**: "fuses 13 small pre-processing operators into one 'super operator'", exploiting
  parallelism between Vector and Cube units. The MLA pre-processing chain is exactly the
  kind of small-op pile-up that hurts us at TPOT 2.74 ms.
- **FlashComm**: "splits AllReduce operations to hide communication latency behind
  computation" — the reduce-scatter/all-gather decomposition again, from a third team.
- Attention **DP** + MoE **EP** hybrid parallelism; RadixCache + a system-RAM prefix cache;
  MTP; async scheduling overlapping D2H sampling copies with next-step prep.
- Claimed outcome: "GLM-5 on a single Chinese node achieves performance comparable to
  dual-GPU international clusters, while reducing deployment costs in long-sequence
  scenarios by 50%" [reported — vendor claim, no benchmark table, config asymmetries
  unstated].

### One third-party GLM-5 serving-config paper

arXiv:2607.02518, "GLM-5 Serving Parameter Tuning for OpenClaw", by Minjie Hua, Ning Wang,
Peijun Yang, Kai Wang, Shiguo Lian — **not** a Zhipu paper despite an earlier search tool
labelling it as one; correct that if it propagates. Workload: OpenClaw traffic, ~28k–30k
input, ~500 output. Two-node 16-GPU cluster (GPU model not stated in what I read).

| | Baseline | Tuned |
|---|---|---|
| chunked prefill size | 2048 | **3072** |
| TP | 4 | 4 |
| PP | 4 | 4 |
| max running requests | 16 | **24** |
| request throughput | 0.43 req/s | 0.48 req/s |
| token throughput | 9,029.64 tok/s | 9,993.23 tok/s |
| avg TTFT | 8.98 s | 6.69 s |
| P90 latency | 40.23 s | 32.64 s |

[verified — arXiv:2607.02518]. Relevance to us is low (TP4/PP4 across two nodes, throughput
regime), but it is a datapoint that chunked-prefill size is worth sweeping and that PP is
being used for GLM-5 in the wild.

---

## Moonshot AI / Kimi

### Kimi K3 — the next model on our list, and it is not DSA

**arXiv:2607.24653v2, "Kimi K3: Open Frontier Intelligence", 27 Jul 2026** (v2 7 Aug 2026),
Kimi Team, weights at `moonshotai/Kimi-K3` [verified].

Architecture comparison table, verbatim from Table 1:

| | Kimi K2 | **Kimi K3** | Δ |
|---|---|---|---|
| Architecture | MoE | MoE | – |
| #Layers | 61 | **93** | ↑52% |
| Total Parameters | 1.04T | **2.78T** | ↑167% |
| Activated Parameters | 32.6B | **104.2B** | ↑220% |
| Hidden Dimension | 7,168 | 7,168 | = |
| Latent MoE Dimension | – | **3584 (0.5×)** | – |
| MoE Hidden Dim per Expert | 2,048 | **3,072** | ↑50% |
| Routed Experts | 384 | **896** | ↑133% |
| Experts Active per Token | 8 | **16** | ↑100% |
| Shared Experts | 1 | **2** | ↑100% |
| Attention Heads | 64 | **96** | ↑50% |
| Dense Layers | 1 | 1 | = |
| Vocabulary | 160K | 160K | = |
| Training Context Length | 128K | **1M** | 8× |
| Attention Mechanism | MLA | **Hybrid KDA–MLA** | – |
| Activation Function | SwiGLU | **SiTU-GLU** | – |
| Attention-Layer Composition | 61 MLA | **69 KDA + 24 MLA** | – |
| MTP Layers | 1 | **1** | = |
| ViT params / layers / patch / heads | – | 401M / 27 / 14 / 12 | – |

Structure: each block is **3 KDA layers + 1 Gated MLA layer** (3:1), with an extra Gated
MLA at the end of the backbone "ensuring that the final layer always performs global
attention". Each attention layer is paired with a Stable LatentMoE FFN. **Attention
Residuals (AttnRes)** use learned pseudo-queries to derive attention weights over the
embedding and preceding block outputs [verified — §2].

**What this means for our engine** [inferred]: K3 is a completely different decode problem
from GLM-5.2. 104.2B activated (2.6× GLM-5's 40B) means ~104 GB read per step at FP8 → ~13
GB/GPU → a **~1.6 ms TPOT floor on 8×B200 even at perfect bandwidth**, before any
inefficiency. And 69 of 93 attention layers hold a *fixed-size recurrent state* rather than
a growing KV cache, so long-context decode stops being KV-bandwidth-bound and becomes
expert-streaming-bound. Our DSA/MLA-specialised kernels and our KV paging will not carry
over. Plan for this now.

MXFP4 makes the memory story survivable: expert weights MXFP4 with MXFP8 activations gets
the dominant parameter mass to ~4 bits.

### Kimi K3's serving stack (§5.4) — the richest published material in this sweep

Framing: "the hybrid KDA–MLA architecture maintains two fundamentally different caches that
must be managed jointly at million-token contexts, its new modules and highly sparse experts
demand kernels tailored to each, and production traffic mixes requests whose per-request
cost spans three orders of magnitude."

**5.4.1 KDA-aware prefix cache.**
- *Unified paged layout*: KDA recurrent states are packed into the **same paged block pool**
  as the MLA KV cache, "unifying pages to the same byte size so that both page types share
  one implementation of allocation, reference counting, and eviction."
- Within a page, "the states of all heads are stored contiguously head by head, so that each
  head's byte stream is self-contained and serves as **the minimal unit of cross-node
  transfer**." Consequence: "Under prefill/decode disaggregation, when prefill and decode
  nodes adopt different TP degrees, **re-layout is performed on the transfer path with zero
  GPU-side reshuffling**." This is a clean answer to heterogeneous-TP PD, which we will want
  if we ever run prefill and decode at different parallelism.
- *Granularity decoupling* (the key idea): a KDA layer keeps one large state per sequence, so
  snapshots are affordable only at sparse boundaries, which would force a shared block size
  of **1024–6144 tokens** — "At such a coarse granularity caching is nearly useless: requests
  shorter than one block can never be reused, and chunked prefill exports no cacheable prefix
  until it crosses a full block boundary." Fix: hash on **512-token hash blocks** inside MLA
  pages while keeping the physical block coarse; save KDA checkpoints only at a sparse subset
  of those hash endpoints. Worked example from Fig. 12: a 6144-token physical block holds 12
  hash blocks; a request matching 2800 tokens hits at **B = 2560 = 5×512**, deep inside the
  physical block, and resumes prefill from B.
- *Concurrency invariants*, each stated as a fix for a concrete failure mode: (a) all cache
  groups draw from one shared free list, so **every hit block is pinned across all groups
  before any allocation**; (b) blocks allocated or registered within the current scheduling
  step are **excluded from matching until their copies land**, because the private copy runs
  on the GPU immediately before the forward pass; (c) evicting one KDA group's checkpoint
  **atomically invalidates its siblings** — "a checkpoint is either hittable in every group or
  in none."

**5.4.2 Kernels.**
- *KDA decode + speculative rollback*. The problem: the recurrent state is updated in place,
  so if MTP verification rejects drafts "the state has already advanced beyond the last
  accepted token and cannot be trivially rolled back", and snapshotting per draft position
  "would multiply state traffic — a cost that dominates at the large batch sizes typical of
  online serving." The fix: "The state after any accepted draft prefix… is fully determined
  by the projected inputs of the draft tokens, which are far smaller than the state itself.
  We therefore **cache only these projected inputs, rebuild the states of accepted tokens
  on-chip, and write back the states of the verified and bonus tokens**" — independently
  proposed as ReplaySSM (Dao AI Lab). Replayed tokens, bonus token and next draft window
  "share one recurrent loop inside a **single fused kernel** covering short convolution,
  input normalization, gating, the KDA recurrence, and output normalization." Verification
  latency grows **sub-linearly** in tokens verified. And because the projection caches never
  leave decode, "prefix caching and prefill–decode disaggregation operate on the same payload
  as in non-speculative serving."
- *Block AttnRes prefill*: reduce-scatter → intra-block kernel on sequence-sharded states →
  all-gather (steal-item 4).
- *Block AttnRes decode*: inter-block kernel on a **side stream**; intra-block merge +
  RMSNorm **fused into the preceding TP all-reduce** (steal-item 1).
- *Stable LatentMoE*: (a) latent down-projection fused with the MoE router into one GEMM;
  (b) latent weights sharded, output all-gather fused into the **GEMM epilogue via `multimem`
  stores**; (c) that communication overlapped with shared-expert compute.
- *Routed experts*: WarpDecode-style token-centric decode kernel (steal-item 2).

**5.4.3 Fleet scheduling.**
- *Cache-aware affinity*: "At 1M context, a typical coding input carries a prefix of 400K
  tokens but requires a prefill increment of only 4K tokens", so a miss costs orders of
  magnitude more than a hit. Sessions are routed to the cluster holding their prefix cache.
  Failure containment via **consistent hashing pinning each session to two clusters** — a
  primary and a pre-assigned secondary that must re-prefill on failover; because consistent
  hashing spreads secondaries uniformly, "this re-prefill work is divided among many clusters
  rather than concentrated on one."
- *Budget-based admission control*: traffic spans 2K to 1M tokens, "so the per-request cost
  spans roughly three orders of magnitude… Capacity planning, queueing models, and
  rate-limiting quotas based on the 'average request' all break down under this variance."
  Failure mode: a burst of long-context requests saturates compute and short requests'
  TTFT collapses. Fix: separate resource budgets per request class.

That last one is directly relevant to our C1-vs-C64 tension [inferred]: it is the published
answer to "how do you protect single-stream latency while also running a throughput fleet",
and it is a scheduler change, not a kernel change.

Also open-sourced by Moonshot and named in K3's references: **FlashKDA**
(`github.com/MoonshotAI/FlashKDA`) and **AgentENV** (`github.com/kvcache-ai/AgentENV`)
[verified — reference list]. I did not fetch FlashKDA; flagging it as the obvious first stop
when we start on K3.

### Kimi Linear / KDA

**arXiv:2510.26692v2, 30 Oct 2025** [verified]. Abstract, verbatim in the relevant parts:

> "At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that
> extends Gated DeltaNet with a finer-grained gating mechanism… Our bespoke chunkwise
> algorithm achieves high hardware efficiency through a specialized variant of the
> Diagonal-Plus-Low-Rank (DPLR) transition matrices, which substantially reduces computation
> compared to the general DPLR formulation while remaining more consistent with the classical
> delta rule."

Pretrained at 3B activated / 48B total, layerwise hybrid of KDA and MLA. Reported results:
outperforms full MLA across all evaluated tasks under an identical recipe, **"reducing KV
cache usage by up to 75% and achieving up to 6 times decoding throughput for a 1M context"**
[reported — Moonshot's own comparison]. KDA kernel and vLLM implementations open-sourced.

**The K3 kernel improvement over Kimi Linear is a beautiful, directly-instructive example of
numerics-driven kernel design** [verified — arXiv:2607.24653 §2.1.1]:

Kimi Linear parameterises log-decay as g = −e^A · Softplus(z) ∈ (−∞, 0), which is unbounded.
The chunkwise form rescales keys by the reciprocal cumulative decay 1/Γ, and "because Γ is a
product of retention factors in (0,1), this reciprocal can grow without bound and overflow in
finite precision." Kimi Linear's workaround was to compute relative decay in log space and
split chunks into secondary 16-token tiles, but "the diagonal tiles… still require explicit
position-pair computations, which remain the main intra-chunk bottleneck."

K3 instead bounds the log-decay with a scaled sigmoid: **g = g_min · Sigmoid(e^A · z)**, with
**g_min = −5** fixed and A a learnable per-head log-scale initialised at 0. Then every
retention factor satisfies α > e^−5 ≈ 6.7×10⁻³, the cumulative log-decay over a 16-token tile
lies in (−80, 0), and the reciprocal rescaling factor stays below e⁸⁰ — **inside BF16 dynamic
range**. "This finite range allows both diagonal and off-diagonal tiles to use dense Tensor
Core matrix multiplications, eliminating the position-pair diagonal path."

That is: a one-line change to the *model's* activation parameterisation removed an entire
non-Tensor-Core code path from the kernel. It is the cleanest model/system co-design result
in this whole sweep, and a template for arguing to a model team that a numerics constraint is
worth training around.

### Mooncake

**arXiv:2407.00079v4 (3 Sep 2025), FAST 2025 best paper**, Moonshot AI + MadSys @ Tsinghua
[verified].

Headline claims, verbatim: "Mooncake can achieve up to a **525% increase in throughput in
certain simulated scenarios** while adhering to SLOs. Under real workloads, Mooncake's
innovative architecture enables Kimi to handle **75% more requests**." Note the hedge in the
first — it is a simulated scenario; the 75% is the number that means something.

Evaluation config, so the numbers can be discounted properly [verified]: **8× NVIDIA
A800-SXM4-80GB per node, NVLink, 800 Gbps RDMA**, one instance (prefill or decode) per node,
and — importantly — "all the experimental results reported in this paper are based on a
**dummy model that follows the same architecture as LLaMA2-70B**". Datasets:

| Dataset | Avg input | Avg output | Cache ratio | Arrival |
|---|---|---|---|---|
| ArXiv Summarization | 8088 | 229 | ~0% | Poisson |
| L-Eval | 19019 | 72 | >80% | Poisson |
| Simulated | 16k/32k/64k/128k | 512 | 50% | Poisson |
| Real traces (23,000) | 7955 | 194 | ~50% | timestamp replay |

Mechanisms:

- **Chunked Pipeline Parallelism (CPP)**, for long-context prefill across nodes. "We group
  every X nodes in the prefill cluster into a pipelined prefill node group. For each request,
  its input tokens are partitioned into chunks, each no longer than the prefill_chunk.
  Different chunks of the same request can be processed simultaneously by different nodes."
  Two stated benefits: cross-node communication only at pipeline-stage boundaries, "which can
  be easily overlapped with computation", giving better MFU and less contention with KVCache
  transfer; and it "naturally fits both short and long contexts", avoiding the frequent
  dynamic repartitioning that elastic sequence parallelism needs. They explicitly rejected
  elastic SP because "SP still requires frequent cross-node communication, which lowers the
  MFU and competes with network resources for transferring KVCache across nodes" — a useful
  recorded negative result. Supplemented by **layer-wise prefill** that streams KVCache out
  as it is produced, overlapping transfer with compute.
- **KVCache-centric scheduler (Conductor)** with distinct objectives per stage: prefill
  maximises cache reuse subject to TTFT SLO / minimum MFU / KVCache < DRAM; decode maximises
  throughput subject to TBT SLO / KVCache < VRAM.
- **Heuristic hot-spot migration** replicating hot KVCache blocks "without requiring precise
  predictions of future KVCache usage."
- **Early Rejection**: load is measured as *SLO satisfaction* (predicted max TTFT vs l_ttft,
  predicted max TBT vs l_tbt), not queue occupancy. The insight: if a request is rejected by
  the decode instance *after* prefill has run, the prefill compute is wasted, so "advance the
  load assessment of the decoding instance to precede the beginning of the prefill stage."
  Conductor accepts on the **greater** of prefill-pool and decode-pool load.
- **Prediction-based Early Rejection**, motivated by an observed failure of plain early
  rejection: over a 20-minute window on a 20-machine cluster they saw "significant anti-phase
  fluctuations between prefill and decoding machines", worse with fewer prefill machines. The
  fix is to reject against *predicted future* decode load rather than instantaneous load.

The anti-phase oscillation result is the most transferable part [inferred] — any admission
controller that gates on instantaneous downstream load will oscillate, and we should expect
the same if we build one for the C64 path.

### Mooncake as shipped software

`github.com/kvcache-ai/Mooncake` [verified]:

- **Transfer Engine**: multi-NIC RDMA bandwidth aggregation, topology-aware device selection
  by NUMA affinity, automatic failover with alternative path routing; transports include TCP,
  RDMA, AWS EFA, NVMe-oF, **NVLink**, HIP. Reported bandwidth: **up to 87 GB/s on 4×200 Gbps
  RoCE (~2.4× TCP)** and **190 GB/s on 8×400 Gbps RoCE (~4.6× TCP)**, measured on 40 GB
  transfers (stated as equivalent to a 128k-token KVCache for LLaMA3-70B) [reported].
- **Mooncake Store**: large-object striping with parallel multi-NIC I/O, DRAM+SSD tiering,
  per-object placement policies (replicas, soft/hard pins, preferred segments).
- **Mooncake EP / Process Group**: expert-parallel dispatch with rank-activeness awareness,
  a PyTorch-compatible distributed PG backend, elastic rank recovery without full restart.
- Integrations: vLLM (`MooncakeConnector`, `MooncakeStoreConnector`), SGLang (PD disagg,
  HiCache multi-tier, EPD disaggregation for multimodal), TensorRT-LLM, LMCache, LMDeploy,
  vLLM-Ascend. Reported: SGLang PD-disaggregated on **128 H200** hitting "224k tokens/sec
  prefill and 288k tokens/sec decode"; and RDMA P2P weight sync for RL giving "7× faster
  weight updates" on a 1T model (53 s → 7.2 s) [reported].

Relevance to a single node is limited, but note that TileRT accepts Mooncake as a PD
transport (`--transport mooncake`), and the NUMA-affinity-aware device selection is directly
relevant to our 2-NUMA-node box.

### Kimi K2 and the K-series lineage

**arXiv:2507.20534v2, "Kimi K2: Open Agentic Intelligence"**, 28 Jul 2025: 1.04T total /
32.6B activated MoE, **MuonClip** optimiser with QK-clip, pretrained on 15.5T tokens "with
zero loss spike" [verified — abstract]. K2.5 and K2.6 exist and are widely benchmarked in the
literature [verified — they appear as evaluated models across many arXiv papers], but I did
not locate standalone K2.5/K2.6 technical reports in the arXiv corpus I searched; treat their
serving details as **[unverified]**.

Moonshot has published **no cost figures** I could source in this sweep. The assignment asked
for them; I did not find them, and I am not going to estimate them.

---

## Qwen / Alibaba

### Qwen3.8 exists — and it is a hybrid linear-attention MoE

`huggingface.co/Qwen` currently lists **Qwen3.8-2.4T-A95B** and **Qwen3.8-2.4T-A95B-FP8** as
the flagship, plus Qwen3.8-27B / -27B-FP8. There is no Qwen3.5 or Qwen4 in the org listing
[verified]. No Qwen3.8 technical report appears on arXiv — the most recent official Qwen
reports I could find are Qwen3.5-Omni (2604.15804), Qwen3-Coder-Next (2603.00729),
Qwen3-ASR, Qwen3-TTS, Qwen3-VL (2511.21631). So **Qwen3.8's architecture is documented only
by its model card** [verified].

Model card architecture:

- **2.4T total / 95B activated**, 92 layers, hidden dim 8192, padded token embedding 248,320.
- Layout: **23 × ( 3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE) )**.
- **Gated DeltaNet**: 128 linear-attention V heads, 16 QK heads, head dim 128.
- **Gated Attention**: 64 Q heads, 4 KV heads, head dim 256, partial RoPE over 64 dims.
- **MoE**: 512 experts total, **10 routed + 1 shared** activated per token, expert
  intermediate dim 2048.
- Context: **262,144 native, extensible to 1,010,000**.
- **MTP trained with multiple steps**.
- Recommended sampling: temperature 1.0, top_p 0.95, top_k 20.
- Deployment instructions are bare (`vllm serve …`, `sglang.launch_server --model-path …`) —
  no tuned flags published.

**The strategic signal is unmistakable** [inferred]. Kimi K3 is 3 KDA : 1 MLA. Qwen3.8 is 3
Gated DeltaNet : 1 Gated Attention. Two independent frontier labs shipped the *same 3:1
linear-to-full hybrid ratio* within months of each other, while Zhipu went the DSA route and
argued in print that linear attention is lossy on fine-grained retrieval. Whatever we build
next must handle a **fixed-size recurrent state per layer alongside a paged KV cache** — the
GLM-5.2 DSA/MLA specialisation is a local optimum, not a platform.

### RTP-LLM — Alibaba's production engine

**arXiv:2605.29639v1, 28 May 2026** — Alibaba's Foundation Model Inference Team with Zhejiang
University, "battle-tested across production deployments serving Taobao, Tmall, and Cainiao",
"serving over 100 million users", open-source [verified].

Headline results, verbatim from the abstract, evaluated across 8B–235B models against vLLM
and SGLang:

| Area | Result |
|---|---|
| Model loading | **4.7×–6.3×** speedup (intro says 1.4×–6.3×); "minute-level deployment of 600B+ parameter models" |
| Production traffic scheduling | **35–37% TTFT P95** reduction, **215% cache reuse** improvement, enabling **75% reduction in prefill machine count** |
| Speculative decoding | **1.12×–2.48×** throughput |
| Multimodal | **1.86×–2.52×** throughput |
| Quantized inference | **35–40% batch latency** reduction, **1.9×–3.0× TTFT** improvement |

Mechanisms worth noting:

- **Optimized model loading**: file-order-driven I/O, shared-memory reuse, parallel
  I/O–communication overlap. Boring but real: our own iteration speed on a 753B checkpoint is
  a research-velocity multiplier, and a 4.7× loading win is free.
- **Hierarchical multi-tiered KV cache** spanning GPU memory → local CPU memory → remote CPU
  memory via RDMA → distributed storage, "with unified hash-based prefix matching".
- **Modular speculative decoding framework**, implemented in C++, decomposed into four
  components: `ProposeExecutor` (naive spec sampling, Prompt Lookup, Eagle, MTP),
  `ScoreExecutor` (parallel target-model scoring of all k candidates), `SpeculativeSampler`
  (verification/acceptance), `SpeculativeUpdater` (splice accepted tokens into the stream).
  This decomposition is worth copying as an *interface* even if we keep our own kernels — it
  lets us A/B EAGLE against MTP against Prompt Lookup without touching the engine.
- **Prompt Lookup** as a first-class proposer: n-gram matching against the input prompt,
  "particularly effective for extractive scenarios, where generated content can be directly
  copied from input prompts", called out for code editing. For AA's ~10k-input / ≥1500-output
  workload with heavy quotation, a Prompt-Lookup proposer layered onto EAGLE could raise
  effective acceptance cheaply [inferred].
- Adaptive KV cache quantization; decoupled multimodal processing; TP/PP/DP/EP multi-level
  parallelism.

Caveat: nearly all of this is throughput and fleet engineering. RTP-LLM does not publish a
single-stream TPOT figure I could find, and its wins would not move our 365 tok/s.

### Aegaeon and PAI

**[unverified]** — I was told to look for Aegaeon (GPU-pooling work) and PAI serving
publications. An arXiv search for "Aegaeon" returned only papers about Saturn's moon; no
Alibaba systems paper surfaced in the corpus I could reach, and with WebSearch exhausted I
could not check conference proceedings directly. I am not going to describe a paper I did not
read. If Aegaeon is real it is likely a SOSP/OSDI-venue paper and should be re-checked.

---

## MiniMax

**arXiv:2501.08313, MiniMax-01 (14 Jan 2025)** [verified — abstract]: MiniMax-Text-01 and
MiniMax-VL-01, "The core lies in lightning attention and its efficient scaling", integrated
with MoE to give **456B total / 45.9B activated, 32 experts**. Explicitly: "We develop an
optimized parallel strategy and highly efficient computation-communication overlap techniques
for MoE and lightning attention."

**arXiv:2506.13585, MiniMax-M1 (16 Jun 2025)** [verified — abstract]: "the world's first
open-weight, large-scale hybrid-attention reasoning model", built on MiniMax-Text-01's
456B/45.9B, natively 1M context.

**arXiv:2605.26494v2, MiniMax-M2 series (26 May 2026)** [verified — abstract]: flagship M2 is
**229.9B total / 9.8B activated** — a sharp *reduction* in both total and activated
parameters versus M1, branded "mini activations unleashing max real-world intelligence".
Named systems components: **Forge**, an agent-native RL system with "windowed-FIFO
scheduling, prefix-tree merging, inference optimization, and a clean
training-inference-agent decoupling". The M2.7 checkpoint "takes an early step toward
self-evolution — autonomously debugging training runs and modifying its own scaffold."

Lineage of lightning attention itself, for completeness [verified — abstracts]: Lightning
Attention (arXiv:2405.17381) and Lightning Attention-2 (arXiv:2401.04658) by Zhen Qin,
Weigao Sun et al., which split attention into intra-block (conventional computation) and
inter-block (linear kernel trick) to defeat the cumsum bottleneck and hold constant training
speed across sequence lengths; TransNormerLLM (arXiv:2307.14995) is the precursor.

**What I could not source, and it matters.** The M2 abstract does not state M2's attention
mechanism, and the HuggingFace model card for `MiniMaxAI/MiniMax-M2` does not disclose it
either — no attention type, no layer count, no expert count [verified — I read both]. The
widely-discussed question of whether MiniMax stepped *back* from lightning attention to full
attention for M2, and their stated reasoning, is **[unverified]**: the blog URL I tried
returned 404 and I had no search budget to find the correct one. Do not repeat the "MiniMax
abandoned linear attention" claim from memory without a source. This is a real gap in this
report.

One M2 serving detail that *is* documented and does affect us: M2 uses **interleaved
thinking** in `<think>...</think>` tags, and "it is important to retain the thinking content
from the assistant's turns within the historical messages" — removing them degrades
performance [verified — model card]. That is a chat-template and prefix-cache correctness
constraint, not a kernel one, but it silently breaks multi-turn prefix reuse if the template
strips think blocks.

---

## ByteDance

### Flux — tile-level compute/communication overlap

`github.com/bytedance/flux`, from ByteDance's Seed Team, Apache-2.0 [verified].

Mechanism: fuse collectives into GEMM kernels so communication overlaps computation **at
tile granularity** rather than at whole-collective granularity. Provided ops:

- **AllGather-GEMM** (dense MLP layer 0)
- **GEMM-ReduceScatter** (dense MLP layer 1)
- **Grouped GEMM + AllGather** (MoE layer 0)
- **Grouped GEMM + ReduceScatter** (MoE layer 1)

Install `pip install byte-flux`; build with `./build.sh --arch 90 --nvshmem`. Depends on NCCL
and NVSHMEM. Paper: Chang et al., "FLUX: Fast Software-based Communication Overlap On GPUs
Through Kernel Fusion", arXiv:2406.06858.

**Hardware support: Ampere (SM80), Ada (SM89), Hopper (SM90). No SM100.** [verified] The
library is not usable on B200 as shipped. Its architecture — a fused collective epilogue on a
grouped GEMM — is exactly what steal-items 1 and 5 describe, and Kimi K3's `multimem`-epilogue
variant is the Blackwell-native form of the same idea.

### COMET — fine-grained MoE overlap, in production

**arXiv:2502.19811** [verified — abstract]. The framing number is one we should hold onto:

> "The inter-device communication of a MoE layer can occupy **47% time of the entire model
> execution** with popular models and frameworks. Therefore, existing methods suggest the
> communication in a MoE layer to be pipelined with the computation for overlapping. However,
> these coarse grained overlapping schemes introduce a notable impairment of computational
> efficiency and the latency concealing is sub-optimal."

COMET's answer is "data dependency analysis and task rescheduling" for "precise fine-grained
overlapping", plus "adaptive workload assignment" to eliminate fine-grained communication
bottlenecks. Results: **1.96× on a single MoE layer, 1.71× end-to-end on average**, and
"COMET has been adopted in the production environment of clusters with ten-thousand-scale of
GPUs, achieving savings of millions of GPU hours" [reported]. It is referenced from the Flux
README as the MoE extension of that line of work.

The critique of coarse-grained overlap is the transferable insight [inferred]: our collectives
are 19.6% with 47% of that being *rank arrival skew*. Coarse-grained overlap (stream-level,
whole-collective) does not fix skew; it just hides some of it behind compute while the skew
itself persists. Tile-level fusion and multicast stores attack skew at the source by removing
the rendezvous.

### ByteScale and other ByteDance infra

**arXiv:2502.21231, ByteScale** [verified — abstract]: "Efficient Scaling of LLM Training with
a 2048K Context Length on More Than 12,000 GPUs". The contribution is dissolving the static
DP×CP device mesh — "Current training frameworks predominantly treat the two techniques as
orthogonal, and establish static communication groups to organize the devices as a static
mesh (e.g., a 2D mesh). However, the sequences for LLM training typically vary in lengths."
This is a **training** paper. No inference transfer.

**veLLM and ByteDance Seed inference publications: [unverified].** I could not source either
in this sweep. ServerlessLLM (arXiv:2401.14351) is Edinburgh, not ByteDance — do not
mis-attribute it.

---

## Ant Group (inclusionAI) — Ling / Ring 2.6

**arXiv:2606.15079, "Ling and Ring 2.6 Technical Report: Efficient and Instant Agentic
Intelligence at Trillion-Parameter Scale", 13 Jun 2026** [verified]. Note: an intermediate
search tool mislabelled this as a MiniMax paper; it is the Ling/Ring line (inclusionAI / Ant
Group). The arXiv landing page I read did not print affiliations, so treat the Ant Group
attribution as **[inferred]** from the model line, not verified in the PDF.

Abstract, verbatim opening: "Efficient and scalable agentic intelligence requires models that
can deliver both **low-latency responses** and strong reasoning capabilities while remaining
practical to train, serve, and deploy… **Ling-2.6 is optimized for instant response
generation and high capability per output token**, whereas Ring-2.6 is tailored for deeper
reasoning and more advanced agentic workflows. Instead of training from scratch, we upgrade
the Ling-2.0 base model through **architectural migration pre-training** and large-scale
post-training."

Reported architecture direction: a **hybrid of Lightning Attention with MLA** for long-context
efficiency; a trillion-parameter Ring-2.6-1T variant; a **KPop** RL framework with
asynchronous scheduling [reported — I read the abstract and landing page, not the full PDF].

Two things here are notable for us [inferred]. First, **"architectural migration
pre-training"** — converting an existing full-attention base into a hybrid rather than
training from scratch — is a third data point for the linear-hybrid trend, and it is the
cheapest possible path to it. Second, Ling-2.6 being explicitly split out as the
*latency-optimised sibling* of a reasoning model is the same product decomposition as
GLM-5.1-highspeed and MiMo-UltraSpeed: **the Chinese labs are shipping a separate
low-latency SKU rather than making one model fast for everyone.** If we are chasing an AA
leaderboard number, that is a hint about what the leaders are actually serving.

I did not extract Ling/Ring's inference section; that is the highest-value unfinished thread
in this report.

---

## Xiaomi MiMo (in scope as a TileRT co-designer)

Not on the original list, but MiMo-V2.5-Pro-UltraSpeed is the fastest published
single-node result in the corpus and it shares TileRT with Z.ai, so it constrains our estimate
of what TileRT can do.

**1000+ tok/s on a 1T-parameter model on one 8-GPU node** (demos ~1200), "a first without
custom silicon" [reported — mimo.xiaomi.com/blog/mimo-tilert-1000tps and
tilert.ai/blog/breaking-1000-tps.html]. Note that neither post discloses which 8 GPUs
[verified — I checked; the hardware is not named in either], so the comparison to our B200
box is not apples-to-apples.

Mechanisms disclosed:

- **Selective MXFP4 with QAT**: "selectively quantize only the MoE Experts to FP4 while
  preserving original precision for all other modules", MXFP4 format, QAT, "capability
  essentially on par with the original model" despite ~50% parameter-memory reduction. The
  TileRT-side framing: "FP4 quantization is applied exclusively to the MoE Experts, while the
  rest of the network maintains FP8… a deliberate, joint engineering trade-off calculated
  based on hardware physical boundaries."
- **DFlash** block-masked parallel drafting (steal-item 8), with the acceptance numbers given
  there. Tuning surface named: "DFlash module structures, sliding window sizes, and Attention
  Sinks, to acceptance lengths versus verification costs."
- **Persistent engine + tile pipelining + warp specialisation + heterogeneous workers**, i.e.
  the TileRT stack.
- Pricing signal: the UltraSpeed SKU was offered at "3× the cost of MiMo-V2.5-Pro, but
  delivering approximately 10× the generation speed" during a June 9–23 2026 trial
  [reported]. A 3×-price / 10×-speed point is a useful market datapoint for how a
  latency-optimised SKU gets sold.

---

## Tencent, Baidu, InfiniAI, SiliconFlow — honest accounting

**Tencent (Hunyuan): one relevant artifact found.** **arXiv:2602.21233, "AngelSlim: A more
accessible, comprehensive, and efficient toolkit for large model compression"**, authored by
the **Hunyuan AI Infra Team** [verified — author list and title via arXiv API]. Covers FP8 and
INT8 post-training quantization, speculative decoding, and sparse attention, with reported
**1.8×–2.0× throughput gains**. I did not fetch the full paper; the abstract-level detail is
all I can stand behind. Tencent's other recent inference-adjacent output is concentrated in
video diffusion (SPADE, DisCa, PISA on Hunyuan-Video), which is off-topic for us.

**Baidu (ERNIE): nothing found.** No official ERNIE 4.5 technical report surfaced in the arXiv
corpus I searched; ERNIE 4.5-300B appears only as an evaluated model in third-party
benchmarks [verified — search result]. If Baidu has published serving-layer engineering it is
somewhere I could not reach without search.

**InfiniAI (无问芯穹) and SiliconFlow (硅基流动): nothing found.** My one attempt to search for
their engineering blogs was blocked by a CAPTCHA and I had no WebSearch budget left. I am
recording this as a gap rather than padding it. Both are plausible sources of real technique
(SiliconFlow in particular publishes serving detail on WeChat), and this should be re-run with
search available.

**A note on what "nothing found" means here.** WebSearch was exhausted at the start of this
session, so Chinese-language sources reachable only by search — Zhihu posts, WeChat public
accounts, company engineering blogs without predictable URLs — are systematically
under-represented in this report. The Chinese-language material I *did* read
(tilert.ai's -zh blog, docs.bigmodel.cn) was materially richer than its English counterpart in
both cases: the -zh version of TileRT's blog is where the 42 GB / 1000 tok/s roofline and the
explicit GPU-0-is-the-indexer split appear. That pattern strongly suggests a re-run with
search budget would be worth it.

---

## Techniques ranked by transferability to our stack

| # | Technique | Source | Attacks | Expected effect | Difficulty | Evidence |
|---|---|---|---|---|---|---|
| 1 | Fuse residual-merge + RMSNorm into the TP all-reduce epilogue | Kimi K3 §5.4.2 | collectives 19.6% | removes 1 kernel + 1 HBM round-trip × 78 layers | Medium | [verified] |
| 2 | Token-centric MoE decode kernel (warp-per-output-neuron, lane teams, offline-permuted weights) | Kimi K3 §5.4.2 / Cursor WarpDecode | MoE GEMM 19.4% | right kernel shape for BS=1 memory-bound expert streaming | Med-High | [verified] |
| 3 | NVFP4 experts / FP8 non-experts (never uniform) | Xiaomi+TileRT; Kimi K3 §4.1.4; Zhipu Ascend W4A8 | MoE + dense GEMM | 3 labs converged; recovers accuracy and dequant latency | Low-Med | [verified]×3 |
| 4 | `multimem` store to fuse all-gather into GEMM epilogue | Kimi K3 §5.4.2 | collectives + arrival skew | NVLS multicast removes gather-side rendezvous; B200-native | Medium | [verified] |
| 5 | Reduce-scatter → compute → all-gather with kernel between halves | Kimi K3 §5.4.2; Zhipu FlashComm | TTFT 189 ms | kills redundant per-rank materialisation | Medium | [verified]×2 |
| 6 | LK loss (−log Σ min(p,q)) for draft training, 7-step unroll, QAT-matched | Kimi K3 §4.1.4 | all (acceptance multiplier) | direct acceptance-rate optimisation vs KL surrogate | Medium (training) | [verified] |
| 7 | Persistent Engine Kernel / single-launch decode | TileRT blog | launch-boundary overhead everywhere | the whole 24%→33% roofline gap | Very High | [verified] design, [reported] numbers |
| 8 | Heterogeneous PD: fast decode engine behind stock vLLM prefill via `KVConnector` | TileRT v0.1.5 | TTFT/decode decoupling | lets us tune decode for C1 without TTFT regression | Medium | [verified] |
| 9 | Cache projected draft inputs, replay state on-chip (never snapshot recurrent state) | Kimi K3 §5.4.2 / ReplaySSM | K3-class models only | makes spec decoding viable on linear-attention layers | High | [verified] |
| 10 | Sub-block prefix caching (512-token hash blocks inside coarse physical pages) | Kimi K3 §5.4.1 | TTFT on repeated 10k prefixes | AA replays similar prefixes for 72 h | Medium | [verified] |
| 11 | Fuse indexer score+ReLU+TopK into one kernel; fuse TopK gather with sparse attention | Zhipu Ascend §5 | DSA indexer 5.8% + attention 10.9% | check how many kernels our indexer path is today | Medium | [verified] |
| 12 | MLAPO-style super-operator (13 small pre-ops → 1) | Zhipu Ascend §5 | attention 10.9% | small-op pile-up is brutal at 2.74 ms TPOT | Medium | [verified] |
| 13 | GPU specialisation: pipelined indexer rank | TileRT blog | DSA indexer 5.8% | only viable pipelined, given `index_topk_freq=4` | High | [verified] design |
| 14 | Budget-based admission control per request class | Kimi K3 §5.4.3 | C1-vs-C64 SLO conflict | protects single-stream latency under bursty load | Low-Med | [verified] |
| 15 | Overlap D2H sampling copy with next-step prep; side streams for independent work | Zhipu Ascend §5; Kimi K3 §5.4.2 | scheduling bubbles | cheap; verify our fork has it | Low | [verified]×2 |
| 16 | Modular spec-decode interface (Propose/Score/Sample/Update) + Prompt-Lookup proposer | RTP-LLM §6 | acceptance on quote-heavy output | lets us A/B proposers; n-gram lookup is nearly free | Low-Med | [verified] |
| 17 | Bounded log-decay (g_min sigmoid) so all tiles hit Tensor Cores | Kimi K3 §2.1.1 | future K3-class serving | template for numerics-driven co-design asks | N/A (model-side) | [verified] |
| 18 | DFlash block-masked parallel drafting, SWA-only draft | Xiaomi MiMo | acceptance (6.30 coding) | large if it reproduces | High | [reported] |
| 19 | Head-contiguous cache layout → zero-reshuffle re-layout on transfer path | Kimi K3 §5.4.1 | heterogeneous-TP PD | enables different TP on prefill vs decode | Medium | [verified] |
| 20 | Prediction-based early rejection (and its anti-phase oscillation failure mode) | Mooncake §7 | C64 admission | any instantaneous-load gate will oscillate | Medium | [verified] |
| 21 | Mooncake Transfer Engine as PD transport (NUMA-affinity device selection) | Mooncake repo | 2-NUMA-node topology | already accepted by TileRT and SGLang | Low | [verified] |
| 22 | ByteDance Flux kernels as-is | Flux repo | — | **SM80/89/90 only; will not build for SM100** | N/A | [verified] |

---

## Open questions and gaps

1. **What input length is our 365 tok/s measured at?** If it is ~10k, we may already be at or
   above TileRT's independently-measured 340 tok/s at 8k/1k, and the framing of this whole
   effort changes. This is the cheapest, highest-value thing to check.
2. **MiniMax's attention rationale for M2** — unsourced, and I refused to reconstruct it from
   memory. Needs a search-enabled follow-up.
3. **Ling/Ring 2.6's inference section** — I read only the abstract. A trillion-parameter
   model explicitly branded for "instant" response is the closest published analogue to our
   objective.
4. **Aegaeon, veLLM, ByteDance Seed inference, InfiniAI, SiliconFlow** — not sourced. Do not
   let these appear in downstream summaries as if they were covered.
5. **FlashKDA** (`github.com/MoonshotAI/FlashKDA`) — named in K3's references, not fetched.
   First stop for K3 enablement.
6. **Whether our SGLang fork's DSA indexer top-k is the non-deterministic CUDA one** that the
   GLM-5 team flagged. Cheap to check, and relevant to P50-over-72h scoring stability.

---

## Sources

All URLs below were fetched and read during this session.

**TileRT / Z.ai**
- https://github.com/tile-ai/TileRT — README (raw: `raw.githubusercontent.com/tile-ai/TileRT/main/README.md`)
- https://api.github.com/repos/tile-ai/TileRT/releases — all 7 release bodies, verbatim
- https://github.com/tile-ai/TileRT/releases/tag/v0.1.3
- https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html
- https://www.tilert.ai/blog/speed-as-the-next-scaling-law-zh.html (richer than the English version)
- https://www.tilert.ai/blog/tilert-vllm-disaggregation.html
- https://www.tilert.ai/blog/breaking-1000-tps.html
- https://inferencex.semianalysis.com/blog/ultra-high-interactivity-on-nvidia — independent benchmark
- https://docs.bigmodel.cn/cn/guide/models/text/glm-5.1-highspeed

**Zhipu / GLM**
- https://arxiv.org/abs/2602.15763 and https://arxiv.org/pdf/2602.15763v1 — GLM-5 tech report (40 pp, full text extracted)
- https://arxiv.org/abs/2607.02518 — GLM-5 Serving Parameter Tuning for OpenClaw (third-party)
- https://huggingface.co/zai-org
- https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json

**Moonshot / Kimi**
- https://arxiv.org/abs/2607.24653 and https://arxiv.org/pdf/2607.24653v2 — Kimi K3 (47 pp, full text extracted; §5.4 is the key section)
- https://arxiv.org/abs/2510.26692 — Kimi Linear
- https://arxiv.org/abs/2507.20534 — Kimi K2 (abstract)
- https://arxiv.org/abs/2407.00079 and https://arxiv.org/pdf/2407.00079v4 — Mooncake (23 pp, full text extracted)
- https://github.com/kvcache-ai/Mooncake

**Qwen / Alibaba**
- https://huggingface.co/Qwen
- https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B
- https://arxiv.org/abs/2605.29639 and https://arxiv.org/pdf/2605.29639v1 — RTP-LLM (14 pp, full text extracted)

**MiniMax**
- https://arxiv.org/abs/2605.26494 — MiniMax-M2 series
- https://huggingface.co/MiniMaxAI/MiniMax-M2
- Abstracts via arXiv API: 2501.08313 (MiniMax-01), 2506.13585 (MiniMax-M1), 2401.04658 (Lightning Attention-2), 2405.17381 (Lightning Attention), 2307.14995 (TransNormerLLM)

**ByteDance**
- https://github.com/bytedance/flux
- https://arxiv.org/abs/2502.19811 — COMET
- Abstract via arXiv API: 2502.21231 (ByteScale), 2406.06858 (FLUX paper, referenced)

**Ant Group / Xiaomi / Tencent**
- https://arxiv.org/abs/2606.15079 — Ling and Ring 2.6
- https://mimo.xiaomi.com/blog/mimo-tilert-1000tps
- Abstract/metadata via arXiv API: 2602.21233 (AngelSlim, Hunyuan AI Infra Team)

**Referenced but not fetched** (named here so they are not mistaken for read sources):
`github.com/MoonshotAI/FlashKDA`; `cursor.com/blog/warp-decode` (WarpDecode);
`tridao.me/blog/2026/replayssm/` (ReplaySSM); arXiv:2602.23881 (LK Losses).
