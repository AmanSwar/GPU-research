# The open-source shelf: kernels and implementations worth taking

**Research date:** 2026-08-17
**Target:** GLM-5.2 (753 B total / 40.3 B active, MLA + DeepSeek Sparse Attention,
256 routed experts top-8) on 8x NVIDIA B200 SXM (SM100), driver 595.71.05,
183,359 MiB/GPU `[verified]` `nvidia-smi`.
**Engine:** `/home/aman/code/NotSglang`, a fork of SGLang at
`0.0.0.dev15965+g96ca6f8cb` `[verified]` from the installed dist-info.

**Label convention** (same as the rest of this series)

| label | meaning |
|---|---|
| `[verified]` | read directly out of a file on this box, or out of a source file in the upstream repo that I fetched; path/URL given |
| `[reported]` | stated by the project in a README / release note / model card that I read, but not cross-checked against its build system |
| `[inferred]` | derived from `[verified]` facts; the reasoning is shown |
| `[unverified]` | flagged for someone to check; I did not establish it |

> **Method note, and a limitation to declare up front.** WebSearch was exhausted
> for this session before this document started, so every external fact below
> comes from a *direct fetch of a specific URL* (repo page, raw source file,
> release page, PyPI page) rather than from search. That biases the inventory
> toward repos I could name in advance. Two consequences: (a) the "less-known
> repos" hunt in §10 is thinner than it should be and is explicitly marked
> incomplete; (b) dates rendered by the fetch tool were wrong at least once
> (the CUTLASS releases page came back with 2024 dates on 4.x releases, which
> is impossible), so **I quote version numbers with more confidence than
> dates**, and dates that could not be corroborated are marked `[unverified]`.

---

## Status

Everything named in the assignment brief exists and was located, with four
findings that change the shopping list:

1. **We are already running most of this shelf.** The C1 profile's top three
   kernels are not ours and not SGLang's: `nvjet_sm100_*` is cuBLASLt,
   `tllm_mnnvl_allreduce::*` and `Bmm_E2m1_*sm100f` are TensorRT-LLM's
   `trtllm-gen` cubins delivered through FlashInfer, and
   `parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen` is a trtllm-gen FMHA
   cubin. `[verified]` — 394 cubins are sitting in
   `/home/aman/.cache/flashinfer/cubins/`, including the exact
   `Bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_..._swiGlu_dynB_sm100f.cubin`
   that the ledger measures at 6.0% of GPU time. The question is therefore not
   "what can we adopt" but "what can we adopt *instead of* trtllm-gen and
   cuBLASLt".
2. **`glm-kernels/` is empty and that is fine.** `[verified]` (recorded in
   `00-local-weights-ground-truth.md` §9.9). There is no in-house kernel to
   defend, so every entry below is a greenfield decision.
3. **TileRT is real, MIT-licensed, built for exactly 8xB200, and does not
   support GLM-5.2.** `[reported]` It supports GLM-5 / GLM-5.1 / DeepSeek-V3.2
   and publishes ~500 tok/s on GLM-5-FP8 against our 365. The local
   `GLM-5.2-FP8-TileRT` weights directory is empty `[verified]`, which is
   consistent: there is nothing to put in it.
4. **Two of the most interesting things on the shelf are already pinned as
   dependencies and nobody has looked at them**: `tokenspeed_mla` (MIT,
   "speed-of-light MLA kernels for Blackwell SM100/SM103") and `humming-kernels`
   — and humming's FP4 path is documented as **SM120+ only, i.e. not B200**
   `[reported]`, so it is dead weight on this box for the format we care about.

Nothing in this document invents a specification. Where a repo's README claims
architecture support I went to its build system; where I could not, the claim is
labelled `[reported]` and the shortlist in §11 includes the check.

---

## What this is

A buy/build inventory for the three kernel families that dominate our C1 profile
`[verified]` from `NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` §2:

| family | GPU ms | share | who owns the kernel today |
|---|---:|---:|---|
| dense GEMM | 3547 | **37.1%** | cuBLASLt (`nvjet_sm100_*`) `[inferred]` from the kernel-name prefix |
| collectives | 1872 | **19.6%** | TensorRT-LLM MNNVL allreduce via FlashInfer `[verified]` kernel names |
| MoE expert GEMM | 1859 | **19.4%** | trtllm-gen blockscaled BMM cubins `[verified]` cubin filenames |
| attention (DSA MLA) | 1040 | 10.9% | trtllm-gen FMHA + SGLang DSA backend `[verified]` |
| DSA indexer | 556 | 5.8% | SGLang `dsa_backend` + CuTe-DSL paged MQA logits `[verified]` |

Plus the supporting cast: quantization toolchains that produced our checkpoints,
the transfer plane we are not yet using, and the measurement tools that would let
us say "this kernel is at X% of roofline" — which, per the ledger's own §2b, we
currently cannot say about **any** kernel.

---

## Bottom line for serving GLM-5.2 on 8xB200

Ranked by expected value per unit of effort. Detail and sourcing in the sections.

1. **Attack the 37.1% dense-GEMM share by getting off cuBLASLt's default
   heuristic, not by writing a kernel.** At C1 the dense GEMMs are `M=1`
   skinny-N shapes (`2624x6144`, `4096x2048`, `6144x2048` — `[verified]`
   ground-truth §6.4) run 78-100x per token, and `nvjet_sm100_tst_64x8_...`
   alone is 12.6% of GPU time. Three cheap levers exist before any kernel work:
   FlashInfer's `trtllm_low_latency_gemm`, DeepGEMM's SM100 path (already
   installed as `sgl-deep-gemm`), and cuBLASLt heuristic search
   (`LtSgemmCustomFind` / `LtFp8CustomFind` pattern). §3, §10.
2. **The NVFP4 checkpoint runs its attention and shared experts in BF16**
   `[verified]` ground-truth §5 — that is 33.4 GB of weights at 2x the bytes
   they need to be, and it lands squarely in the 37.1%. `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN`
   covers only `q_b_proj`. Widening that is a config/loader change, not a kernel. §3.
3. **Try TokenSpeed's MLA kernels.** MIT, explicitly SM100/SM103, already a
   pinned dependency at 0.1.8 while upstream is at 0.2.5 `[reported]`. This is
   the lowest-effort swap against the 10.9% attention share that exists. §2.
4. **Take FlashMLA's SM100 sparse kernels seriously for prefill, and with
   scepticism for decode.** Its own README publishes **1450 TFLOPS sparse MLA
   prefill on B200** but says SM100 sparse *decode* is "not really optimized
   yet" `[reported]`. Our TTFT is 189 ms; that is where it pays. §2.
5. **Collectives: the fix is scheduling, not a faster allreduce.** 47% of
   collective time is rank-arrival skew `[verified]` ledger §2a. Nothing on this
   shelf fixes skew except overlap (SBO/TBO, already in-tree) and NCCL CTA
   policy. Swapping the allreduce implementation addresses at most the other
   53%. §5.
6. **Upgrade NCCL and NVSHMEM before believing any collective measurement.**
   Installed: NCCL 2.28.9, NVSHMEM 3.4.5 `[verified]`. Upstream: NCCL 2.31.x
   with Blackwell CFT device APIs, NVSHMEM 3.7.2 `[reported]`. §5.
7. **Run TileRT on GLM-5.1 as a ceiling probe.** It cannot load GLM-5.2, but it
   runs the same kernel shapes on the same box and would tell us how much of the
   365→500 tok/s gap is kernels versus scheduling. §6.
8. **Build the measurement floor first.** No kernel on this box has a measured
   fraction-of-roofline `[verified]` ledger §2b. nvbandwidth + an ncu pass on the
   top three kernels is a day of work and re-ranks everything above it. §10, §11.
9. **DeepEP is a C64 concern, not a C1 one**, and its upstream build system has
   no SM100 gencode at all `[verified]` — our Docker build forces it. §5.
10. **Do not port ThunderKittens, Megakernels, or Mirage for this model.** All
    three are real and Blackwell-capable; none has an MoE + MLA + sparse-attention
    implementation, and writing one is a quarter, not a sprint. §6.

---

## 0. What is already installed on this box

`[verified]` — `ls /home/aman/code/NotSglang/.venv/lib/python*/site-packages`.
This is the true starting inventory and several entries below are "upgrade",
not "adopt".

| package | installed | upstream latest | gap |
|---|---|---|---|
| `flashinfer_python` | 0.6.15.post1 | 0.6.17 (2026-08-11) `[reported]` | 2 minor |
| `sglang_kernel` | 0.4.5 | 0.4.6.post1 (2026-08-06) `[reported]` | 1 patch |
| `sgl_deep_gemm` | 0.1.5.post1 | — | SGLang's DeepGEMM fork |
| `nvidia_cutlass_dsl` | 4.6.0 | CUTLASS 4.7.0 `[reported]` | 1 minor |
| `flash_attn_4` | 4.0.0b19 | — | FA4 beta |
| `tilelang` | 0.1.11 | 0.1.13 (2026-08-03) `[reported]` | 2 patch |
| `tokenspeed_mla` | 0.1.8 | 0.2.5 (2026-08-09) `[reported]` | **1 minor + 5 patch** |
| `quack_kernels` | 0.6.3 | — | CuTe-DSL memory-bound kernels |
| `humming_kernels` | 0.1.10 | 0.1.12 (2026-07-31) `[reported]` | FP4 is SM120+ only |
| `helion` | 0.2.6 | — | PyTorch DSL |
| `triton` | 3.6.0 | — | |
| `torch` | 2.11.0 | — | |
| `nvidia_nccl_cu13` | 2.28.9 | 2.31.2-1 `[reported]` | **3 minor** |
| `nvidia_nvshmem_cu13` | 3.4.5 | 3.7.2 `[reported]` | **3 minor** |
| `compressed_tensors` | 0.18.0 | — | |
| **absent** | `deep_ep`, `mooncake`, `nixl`, `lmcache`, `modelopt` | | not installed at all `[verified]` |

The FlashInfer cubin cache holds **394 cubins** under
`~/.cache/flashinfer/cubins/`, split across `fmha/trtllm-gen/` and
`batched_gemm-da58956-b4ac80e/` `[verified]`. Reading the filenames is free
performance archaeology: the MoE kernel we run is
`Bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_**t128x8x512u2**_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schedS_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f`
— tile **128x8x512**, 5-stage pipeline, MMA 128x8x64, fused SwiGLU, dynamic
batch, `sm100f`. `[inferred]` from the naming convention. An N-tile of 8 against
a per-expert N of 2048/TP8 = 256 is the shape story of our decode MoE.

---

## 1. What the model demands of a kernel

Every entry below has to satisfy these. All `[verified]` from
`/home/aman/code/weights/GLM-5.2-*/config.json` and recorded in
`00-local-weights-ground-truth.md`; I am restating, not re-deriving.

| constraint | value | why a kernel library lives or dies on it |
|---|---|---|
| attention variant | MLA, latent width 576 (`kv_lora_rank` 512 + `qk_rope_head_dim` 64) | identical to DeepSeek-V3/V3.2 and Kimi-K2/K3, so DeepSeek-targeted MLA kernels transfer |
| head counts | 64 Q heads, `qk_nope` 192, `v_head_dim` **256** | **not** DeepSeek's 128/128/128. A kernel with a hardcoded `v_head_dim==qk_nope_head_dim` will not compile or will be silently wrong |
| sparse attention | DSA, `index_topk` 2048, indexer on 21 of 78 layers (`index_topk_freq` 4) | needs a *sparse* MLA kernel with a top-k gather, not dense MLA |
| MoE | 256 routed, top-8, 1 shared, `moe_intermediate_size` 2048, `n_group=1` | no group-limited routing => a token's 8 experts can land on all 8 ranks; all-to-all fan-out is unbounded |
| shared-expert fusion | SGLang fuses shared into the routed kernel => **257 experts** under plain TP8, but not under EP `[verified]` ground-truth §7.5 | a grouped-GEMM kernel sees 257, not 256 |
| quant, NVFP4 build | only `layers.{3..77}.mlp.experts.*` are FP4; group size 16, E4M3 scale plane + FP32 global | needs *NVFP4 blockscaled* GEMM (group 16), not MXFP4 (group 32) |
| quant, FP8 build | block 128x128 weight scale, dynamic per-128 activation | needs *blockwise* FP8 GEMM, not per-tensor |
| KV cache | FP8 latent, 656 B/token/layer + 132 B indexer; page size asserted **64** | a paged kernel with page 16/32/128 does not drop in |
| RoPE | interleaved (`rope_interleave: true`), theta 8e6, no scaling, 1 M context | `is_neox_style=False` variants only |
| MTP | 1 full decoder layer at index 78 | draft path runs the same kernels at `M = num_draft_tokens` (4 at 3-1-4) |

Two of these are quiet killers for third-party code: **`v_head_dim=256 != qk_nope_head_dim=192`**,
and **page size 64**.

---

## 2. Attention

### FlashInfer — `https://github.com/flashinfer-ai/flashinfer` — Apache-2.0

**Implements:** paged/ragged attention, MLA, sparse attention, top-k, FP4/FP8
GEMM, fused MoE, sampling, and the `comm` module (§5). Also the *delivery
mechanism* for TensorRT-LLM's `trtllm-gen` cubins.
**SM100 status: verified.** The README's own build line is
`FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 10.3a 10.7a 11.0a 12.0f"`
`[reported]`, and independently we have 394 `sm100f` cubins in the local cache
`[verified]`.
**Published performance:** none in a form comparable to our workload. Release
notes for 0.6.17 cite Kimi-K3 MLA decode (96 Q heads : 1 KV head, down to 6
TP-local heads) and MiniMax-M3 sparse attention `[reported]`.
**Maintenance:** very strong — 0.6.16.post3, .post4, 0.6.17 all inside four days
of 2026-08-08..11 `[reported]`. Also a warning sign: 0.6.16.post3 was a *revert*
of an SM90 CUTLASS MoE backend "due to stability concerns", and .post4 fixed a
Python 3.10/3.11 type-annotation break. Pin exactly; do not track main.
**Modules relevant to us:** `mla/`, `sparse.py`, `topk.py`, `topk_varlen/`,
`dsv3_ops/`, `gemm/`, `trtllm_low_latency_gemm.py`, `deep_gemm.py`, `comm/`,
`cute_dsl/`, `autotuner/` `[verified]` from the package listing.
**How we would use it:** (a) bump 0.6.15.post1 → 0.6.17 behind an A/B, since
0.6.16/0.6.17 touched MoE EP and MLA decode; (b) `trtllm_low_latency_gemm` is
the single most on-the-nose thing in the whole shelf for our 37.1% dense-GEMM
problem at `M=1`; (c) `autotuner/` is how you make the trtllm-gen cubin choice
shape-aware instead of accepting the default.

### FlashAttention / FA4 — `https://github.com/Dao-AILab/flash-attention` — BSD-3-Clause

**Implements:** FA2/FA3/FA4. FA4 is "written in CuTeDSL and optimized for Hopper
and Blackwell GPUs (e.g. H100, B200)" `[reported]`, shipped as PyPI `flash-attn-4`.
**SM100 status: reported, not verified** — I did not read FA4's arch flags. But
`flash_attn_4-4.0.0b19` is installed here `[verified]` and the fork lists
`flash-attn-4>=4.0.0b18` as a hard dependency `[verified]` `python/pyproject.toml:33`.
**Relevance to GLM-5.2: low.** FA is MHA/GQA. Our decode path is absorbed MLA
with a 576-wide latent and one KV head; FA4 has nothing to say about it. It
matters for the vision/multimodal siblings and for any MHA prefill fallback.
**Note:** sgl-kernel does not vendor Dao-AILab's repo — it FetchContents
`sgl-project/sgl-attn@f89bc23` `[verified]` `python/sglang/kernels/aot/CMakeLists.txt:82-88`.
That is SGLang's own FA fork and is the thing actually compiled.

### FlashMLA — `https://github.com/deepseek-ai/FlashMLA` — MIT

**Implements:** MLA decode/prefill, dense and DSA-sparse.
**SM100 status: verified from `setup.py`.** It emits
`-gencode arch=compute_100f,code=sm_100f` (gated on NVCC >= 12.9, disableable via
`FLASH_MLA_DISABLE_SM100`) and compiles `csrc/sm100/prefill/dense/`,
`csrc/sm100/prefill/sparse/`, `csrc/sm100/decode/` `[verified]`.
**Kernel-by-arch matrix** `[reported]` from the README:

| kernel | SM90 | SM100 |
|---|:--:|:--:|
| dense decode | yes | — |
| sparse decode (FP8) | yes | yes |
| dense prefill (MHA) | — | yes |
| sparse prefill (MQA) | yes | yes |

**Published performance** `[reported]`: sparse MLA **prefill on B200 up to 1450
TFLOPS**; MHA prefill on B200 up to 1460 TFLOPS fwd / 1000 bwd; sparse MLA decode
on B200 "up to 350 TFlops" with the caveat that SM100 sparse decode is **"not
really optimized yet"**. H800 reference points: dense decode 3000 GB/s / 660
TFLOPS, sparse decode 410 TFLOPS, sparse prefill 640 TFLOPS.
**Maintenance:** DeepSeek's own repo, tracks their model releases; not a
continuously-maintained library.
**How we would use it:** this is the closest public match to our attention. The
`v_head_dim=256` mismatch against DeepSeek's 128 is the risk — check the sparse
kernel's head-dim template instantiations before budgeting time. SGLang already
has a `flashmla_backend.py` and 27 files referencing `flashmla` `[verified]`, so
the wiring exists. Target **prefill first** (TTFT 189 ms), because that is where
FlashMLA's B200 numbers are strong and its own authors say decode is not.

### TokenSpeed MLA — `https://github.com/lightseekorg/tokenspeed` — MIT

**Implements:** PyPI `tokenspeed-mla` is described as "Speed-of-light TokenSpeed
MLA kernels for Blackwell (SM100/SM103)", prefill and decode, aimed at
"coding agent style use cases with high request concurrency, short decode steps,
and strict TTFT/TPOT requirements" `[reported]` — which is precisely the
Artificial Analysis workload the ledger §1 is optimizing for.
**SM100 status: reported (SM100/SM103 named explicitly on PyPI); not verified
against build flags** — the repo page I fetched did not expose CMakeLists.
**Published performance** `[reported]`: 580 TPS on Qwen3.5-397B-A17B for agentic
workloads; a Pareto curve vs TensorRT-LLM on Kimi K2.5 on **B200**. No
FlashMLA/FlashInfer head-to-head published.
**Maintenance:** 834 commits, 1.9k stars, MIT, releases through August 2026
`[reported]`.
**How we would use it:** it is *already a pinned dependency* — `tokenspeed_mla==0.1.8`
in `pyproject.toml` and installed in the venv `[verified]` — and SGLang has a
`tokenspeed_mla_backend.py` plus an arg-group branch that only fires for
Kimi-K3 DCP `[verified]` `arg_groups/overrides.py:331,381-398`. So the
integration is written and gated away from us. **Bump to 0.2.5 and force the
backend on GLM-5.2** is a one-line experiment. This is the highest
value-per-minute item in the entire document.

### TileLang DSA kernels — `https://github.com/tile-ai/tilelang` — MIT

**Implements:** a Pythonic tile DSL over TVM, plus a large example set including
**DeepSeek V3.2 sparse attention with top-k**, MLA, FlashAttention on SM100,
block-sparse attention, FP8/FP4 blockscaled GEMM `[reported]`.
**SM100 status: reported** — "code paths from SM70 through SM120", with named
Blackwell features (MXFP8 block-scaled GEMM, FlashAttention on SM100).
`[unverified]` whether the DSA example is instantiated for SM100 or only SM90.
**Local reality check `[verified]`:** SGLang's tilelang DSA path is guarded —
`overrides.py:1464` says *"tilelang's fp8 KV path is ROCm-only; the CUDA kernel
hardcodes bfloat16"*, and `hisparse_hook.py` only selects tilelang for ROCm.
Our KV is `fp8_e4m3`. **So on this box, today, the TileLang DSA backend is
unusable without kernel work.** That is a concrete, nameable gap, not a vague one.
**How we would use it:** as the authoring environment if we decide to write a
GLM-5.2-specific DSA decode kernel (the `v_head_dim=256` problem makes a
template-instantiation fight with FlashMLA plausible). v0.1.13 is out; we pin
0.1.11.

---

## 3. Dense GEMM — the 37.1%

### cuBLASLt (`nvjet_sm100_*`) — proprietary, ships with CUDA

**What it is:** the incumbent. `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` is
**1206 ms / 12.6% of GPU time**, the single largest kernel in the profile
`[verified]` ledger §2. The `64x8` in the name is the tile — 8 wide in N, which
is what you get for `M=1`-ish decode shapes.
**Why it is 37.1%:** three compounding facts, all `[verified]` from ground-truth
§4.5/§6.4 — (a) `fused_qkv_a_proj_with_mqa` [2624,6144] and `indexer.wq_b`
[4096,2048] are `ReplicatedLinear`, so **all 8 ranks do the same work**, 78 and
22 times per token; (b) in the NVFP4 build all of attention is **BF16**, double
the bytes of the FP8 build; (c) at C1 every one of these is bandwidth-bound with
no arithmetic to hide behind.
**How we would use the shelf against it,** cheapest first:
1. **`SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN`** — exists, covers only `q_b_proj`
   `[verified]` `deepseek_v2.py:2234-2243`. Extend it to the fused a-projection
   and `o_proj` and you halve the bytes on the two largest dense GEMMs.
2. **FlashInfer `trtllm_low_latency_gemm`** — purpose-built for this regime.
3. **DeepGEMM** (below) for the FP8 blockwise path.
4. **cuBLASLt heuristic search** (§10) — the default heuristic is picked without
   knowing our shape distribution.

### DeepGEMM — `https://github.com/deepseek-ai/DeepGEMM` — MIT

**Implements:** FP8 dense GEMM, grouped GEMM (contiguous and masked layouts) for
MoE, K-grouped, **FP8xFP4 GEMM**, "Mega MoE" fusing dispatch + linear + SwiGLU +
combine, and **MQA scoring kernels (weighted ReLU MQA logits) for the lightning
indexer** `[reported]` — i.e. it has a kernel aimed at exactly our DSA indexer.
**SM100 status: verified by the README's Requirements section** — "NVIDIA SM90 or
SM100 architecture GPU", "CUDA 12.9 or higher for SM100", CUTLASS 4.0+ `[reported]`.
Build is runtime JIT (optionally NVRTC), so there is no gencode list to read;
the arch gate is a runtime check.
**Published performance:** "up to 1550 TFLOPS on H800" `[reported]`. **No SM100
performance table is published** — I looked and it is not there. Treat B200
numbers as unmeasured.
**Maintenance:** MIT, latest documented update 2026-04-16 `[reported]`. We run
SGLang's fork, `sgl-deep-gemm==0.1.5.post1`, installed `[verified]`; 49 files in
`srt/` reference `deep_gemm` and there is a dedicated
`srt/layers/deep_gemm_wrapper/` with a `configurer.py` `[verified]`.
**How we would use it:** it is already wired. The open question is whether it is
*on* for GLM-5.2's shapes and whether its masked grouped-GEMM beats the trtllm-gen
`Bmm_E2m1_*` cubin at N=8. The MQA-logits kernel is a direct alternative to the
5.8% DSA indexer share.

### CUTLASS — `https://github.com/NVIDIA/cutlass` — BSD-3-Clause

**SM100 status: verified from `examples/CMakeLists.txt`** — the Blackwell example
set is real and named `[verified]`:

| example | what it gives us |
|---|---|
| `70_blackwell_gemm`, `71_..._collective_builder` | baseline tcgen05 GEMM |
| **`72_blackwell_narrow_precision_gemm`** | **NVFP4/MXFP8 blockscaled — the reference for our expert GEMM** |
| `73_blackwell_gemm_preferred_cluster`, `74_..._streamk` | cluster/StreamK scheduling |
| **`75_blackwell_grouped_gemm`** | **grouped GEMM — the MoE shape** |
| **`77_blackwell_fmha`** | **FMHA reference** |
| `81_blackwell_gemm_blockwise` | **blockwise scaling — the FP8 128x128 checkpoint format** |
| `82_blackwell_distributed_gemm` | GEMM+comm overlap |
| `83`, `84` | sparse and narrow-precision sparse GEMM |
| `86_blackwell_mixed_dtype_gemm` | BF16 activation x FP4 weight |
| **`92_blackwell_moe_gemm`** | **MoE GEMM, added recently** |
| `93_blackwell_low_latency_gqa` | low-latency decode attention |
| `95_blackwell_gemm_green_context` | green contexts / SM partitioning |

**Versions:** 4.7.0 is the newest release listed; 4.6.0 added a "ptr-array TMA
collective for tensor/token-scaled FP8 grouped GEMM" and compile-time register-spill
diagnostics; 4.7.0's FMHA example has a 2-kernel backward ~25% faster than
1-kernel at FP8 on **SM103** `[reported]`. **Dates on that releases page came back
as 2024 and are wrong — treat all CUTLASS dates here as `[unverified]`.**
We pin `nvidia-cutlass-dsl==4.6.0` `[verified]`, and sgl-kernel FetchContents
CUTLASS at commit `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` `[verified]`.
**How we would use it:** 72, 75, 81 and 92 are the four to read before writing
anything. 95 (green context) is the sleeper — carving SMs between the DSA indexer
and the MoE could attack the arrival skew in §5 in a way no collective change can.

### humming — `https://github.com/inclusionAI/humming` — Apache-2.0

**Implements:** JIT GEMM library for quantized inference, dense + MoE, "any weight
type under 8-bit" against FP16/BF16/FP8/FP4/INT8/INT4 activations `[reported]`.
**SM100 status: reported as ABSENT for the format we need.** The support matrix
reads FP16/INT8 SM75+, BF16/INT4 SM80+, FP8 SM89+, **FP4 SM120+** `[reported]`.
SM100 is not named anywhere. **`[inferred]`: humming's FP4 path targets consumer
/ RTX PRO Blackwell (SM120), not B200.**
**Yet it is a hard dependency:** `humming-kernels[cu13]==0.1.10`, installed, with
12 files in `srt/` referencing it including `moe_runner/humming.py` and
`quantization/humming.py` `[verified]`.
**How we would use it:** we would not, for NVFP4 on B200. Worth a 10-minute check
that no default path silently selects it and falls back. This is the one entry
on the shelf I would actively remove from the decision space.

---

## 4. MoE and blockscaled FP4 — the 19.4%

### TensorRT-LLM `trtllm-gen` — `https://github.com/NVIDIA/TensorRT-LLM` — Apache-2.0

**The incumbent for our MoE.** Kernel directories under
`cpp/tensorrt_llm/kernels/` `[verified]` from the tree listing:

| directory | relevance |
|---|---|
| `trtllmGenKernels/{batchedGemm, blockScaleMoe, fmha, gemm, gemmGatedAct}` | **all four of our hot kernel families** |
| `flashMLA` | TRT-LLM's own MLA import |
| `customAllReduceKernels`, `communicationKernels`, `userbuffers` | the MNNVL allreduce we run |
| `arcquantFP4`, `fusedCatFp4`, `fusedCatFp8`, `marlin` | FP4/FP8 support ops |
| `moeLoadBalance`, `customMoeRoutingKernels` | EPLB and routing |
| `contextFusedMultiHeadAttention`, `decoderMaskedMultiheadAttention` | XQA-lineage decode attention |

**SM100 status: verified indirectly and decisively** — we are executing its
`sm100f` cubins right now `[verified]`, 394 of them in the FlashInfer cache.
**Delivery:** the cubins arrive via `flashinfer-cubin==<version>` (CUDA-agnostic)
and optionally `flashinfer-jit-cache` `[verified]` `docker/Dockerfile:379-387`.
You do not need to build TensorRT-LLM to use trtllm-gen.
**How we would use it:** we already do. The actionable part is **cubin
selection**: the profile shows one `t128x8x512u2` variant dominating, and the
cache holds 394. FlashInfer's `autotuner/` is the supported way to choose per
shape. That is a measurement task, not a kernel task.

### vLLM's kernel set — `https://github.com/vllm-project/vllm` — Apache-2.0

**SM100 status: verified by reading `CMakeLists.txt`**, and the answer is
mostly *no* for the interesting parts `[verified]`:

| kernel family | arch gate in vLLM's CMakeLists |
|---|---|
| **machete** | `cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" ...)` — **Hopper only** |
| **marlin** (fp16 out) | `"8.0+PTX;12.0f"` (CUDA 13) / `"8.0+PTX;12.0a;12.1a"` — **no 10.0** |
| marlin (bf16 out) | `"8.0+PTX;9.0+PTX;12.0f"` — **no 10.0** |
| marlin (fp8 in) | `"8.9;12.0f"` — **no 10.0** |
| cutlass w8a8 scaled_mm | `"9.0a"`, plus `"10.0f;10.7f;11.0f"` (CUDA 13) / `"10.0a;10.1a;10.3a"` — **yes, SM100** |
| **NVFP4** | `FP4_SM100_ARCHS "10.0f;10.7f;11.0f"`, CUDA 12.8+ — **yes, SM100** |
| DSV3 router GEMM | SM90+ |

**`[inferred]`: machete and marlin are not options on B200.** Machete is a
Hopper mixed-input GEMM and was never ported; marlin's Blackwell support jumped
straight to SM120. The transferable pieces are vLLM's **CUTLASS w8a8 scaled_mm**
and its **NVFP4** paths — and SGLang already carries `marlin_utils_fp4.py`,
`marlin_utils_fp8.py` and 58 marlin-referencing files `[verified]`, which is
worth an audit precisely *because* marlin has no SM100 gencode: any path that
selects it here either dead-codes or falls back.
**Also worth taking:** `torch_symm_mem.py` in our tree is literally
"Adapted from https://github.com/vllm-project/vllm/.../symm_mem.py" `[verified]`
— vLLM's distributed layer is a productive source even where its kernels are not.

### sgl-kernel (`sglang-kernel`) — `https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot` — Apache-2.0

**SM100 status: verified from the local `CMakeLists.txt`** `[verified]`
`python/sglang/kernels/aot/CMakeLists.txt`:
```
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)
    "-gencode=arch=compute_100a,code=sm_100a"
    "-gencode=arch=compute_120a,code=sm_120a"
    if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
        "-gencode=arch=compute_103a,code=sm_103a"
```
and `-DENABLE_NVFP4=1` on CUDA >= 12.8.
**A defect worth reporting upstream `[verified]`:** the InfLLM-v2 flash block at
lines 507-514 adds `compute_120a` under the *SM100A* gate but never adds
`compute_100a`:
```
list(APPEND INFLLM_FLASH_CUDA_FLAGS "-gencode=arch=compute_90,code=sm_90")
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)
    list(APPEND INFLLM_FLASH_CUDA_FLAGS "-gencode=arch=compute_120a,code=sm_120a")
endif()
```
`[inferred]` those kernels reach B200 only through SM90 PTX JIT, if at all.
Not on our critical path (InfLLM is a different sparse-attention scheme) but it
is the kind of thing that makes a "supported" claim false.
**Third-party it vendors** `[verified]` `CMakeLists.txt:48-89`: CUTLASS
`@57e3cfb`, fmt `@553ec11`, Triton `v3.6.0`, FlashInfer `@bc29697`,
`sgl-project/sgl-attn@f89bc23`.
**Maintenance:** upstream 0.4.6.post1 (2026-08-06); we pin 0.4.5 `[reported]`/`[verified]`.
**Note the packaging change:** the standalone `sgl-kernel` directory no longer
exists at the sglang repo root (my fetch 404'd `[verified]`); it moved under
`python/sglang/kernels/` per RFC #29630, with a registry/selector layer
(`spec.py`, `registry.py`, `selector.py`) and 20 operator groups `[verified]`
from the local README. **`select_kernel` does no auto-tuning** — "an op with
several backends must be resolved by naming one; the extra backends are
inventory only" `[verified]`. That registry is the natural insertion point for
anything we adopt from this document.

---

## 5. Collectives — the 19.6%, of which 47% is skew

### The finding that reframes this whole section

`[verified]` ledger §2a: `observed 14,097 ms = transfer 7,505 ms + waiting
6,599 ms`; arrival skew mean 9.2 us, max 4,897 us; rank 0 arrives last in 24% of
114,171 instances. **A faster allreduce addresses the 53%, not the 47%.**

### FlashInfer `comm` — Apache-2.0

**Implements** `[reported]` from the API docs, exact names:
`trtllm_allreduce_fusion`, `trtllm_custom_all_reduce`, `trtllm_moe_allreduce_fusion`,
`trtllm_moe_finalize_allreduce_fusion`, `trtllm_mnnvl_allreduce`,
`trtllm_mnnvl_fused_allreduce_add_rmsnorm`, `..._quant`, `MnnvlMemory`,
`McastGPUBuffer`, plus vLLM's custom AR (`vllm_init_custom_ar`, ...) and a
unified `AllReduceFusionWorkspace` with one-shot/two-shot strategies.
`AllReduceFusionOp` covers RESIDUAL_RMS_NORM with optional FP8/NVFP4 quant and
**per-token-group blockwise (DeepSeek-style) quant**.
**SM100 status: verified in our own tree.** `flashinfer_comm_fusion.py` gates on
`is_sm100_supported()` and allows the mnnvl backend unconditionally on SM100
`[verified]` `srt/layers/flashinfer_comm_fusion.py:39-48`.
**How we would use it — and an open contradiction.** The ledger states FlashInfer
allreduce fusion was **off** in every run `[verified]` ledger §2c, yet the profile's
#2 and #6 kernels are `tllm_mnnvl_allreduce::oneshotAllreduceFusionKernel`
(783 ms, 8.2%) and `trtllm_mnnvl_allreduce::twoshotAllreduceKernel` (407 ms, 4.3%)
`[verified]` ledger §2. `[inferred]`: the MNNVL allreduce *transport* is active
independently of the *fusion* flag. **Resolving that is a prerequisite to
interpreting any collectives number** and is item 8 in §11.

### NCCL — `https://github.com/NVIDIA/nccl` — BSD-3-Clause

**Installed 2.28.9; latest 2.31.2-1** `[verified]` / `[reported]`.
**Blackwell-relevant in recent releases** `[reported]`: Compute Fabric Transport
(CFT) host+device APIs for window registration and device-side Put/Get/Red/NVLS,
"supported on Blackwell with CUDA 13.3+"; TMA support in built-in symmetric
kernels, enabled by default and in the cost model; NVLink multicast for
AllGather small-message latency; hierarchical NVLS-in-node + PAT-across-node;
per-collective algorithm selection APIs; MNNVL cross-clique.
**Tuner plugin API** `[reported]` from the NCCL env docs: `NCCL_TUNER_PLUGIN`
takes a suffix or library name, falls back to `libnccl-tuner-<suffix>.so`, then
the net plugin, then the internal tuner. Adjacent knobs that matter to us:
`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_NVLS_ENABLE` (0/1/2, default 2 = enable with
fallback), `NCCL_MAX_CTAS`/`NCCL_MIN_CTAS` (replacing the deprecated
`NCCL_*_NCHANNELS`), and **`NCCL_CTA_POLICY`** with values `DEFAULT`/`EFFICIENCY`/`ZERO`,
combinable with `|`.
`[unverified]` — I did not read the `ext-tuner` example header (the tree URL
404'd), so the exact tuner API version and callback signature are unconfirmed.
**How we would use it:** at C1 with 114k collective launches in 20 s, the
interesting knob is **CTA count**, not algorithm: every SM NCCL takes is an SM
the MoE kernel does not have, and every SM it does not take is latency. A
`NCCL_MIN_CTAS`/`NCCL_MAX_CTAS`/`NCCL_CTA_POLICY` sweep is an afternoon and needs
no code. Writing a tuner plugin is the follow-on if the sweep shows a shape-
dependent optimum.

### torch symmetric memory — PyTorch, BSD-3-Clause

**Status: verified in-tree** — `srt/distributed/device_communicators/torch_symm_mem.py`
wraps `torch.distributed._symmetric_memory`, "chooses between 'multimem' and
'two-shot' all-reduce kernels", with size caps in `all_reduce_utils.py`
`[verified]`. There is also `triton_symm_mem_ag.py` (Triton symmetric-memory
all-gather) and a JIT `custom_all_reduce_v2.py` with **three algorithms
(1shot_push / 1shot_pull / 2shot_pull) and three pull sources (eager workspace,
CUDA-graph pointer table, multicast address)** `[verified]`.
**How we would use it:** this is a fourth allreduce implementation already in the
tree and, on the evidence of the module docstring, the most thoughtfully built
one for CUDA-graph capture. A 4-way bake-off (trtllm mnnvl one-shot, trtllm
two-shot, torch symm-mem multimem, custom_all_reduce_v2 1shot_pull) at our exact
message sizes is a contained experiment.

### DeepEP — `https://github.com/deepseek-ai/DeepEP` — MIT

**Implements:** MoE dispatch/combine all-to-all, NVLink intranode + RDMA
internode, low-latency kernels, "0 SM" experimental modes `[reported]`.
**SM100 status: this is the honest one.** The README Requirements say
**"Hopper (SM90) GPUs, or other architectures with SM90 PTX ISA support"** and
"CUDA 12.3 and above for SM90 GPUs" `[reported]`. Its `setup.py` **has no SM100
gencode at all** — it defaults `TORCH_CUDA_ARCH_LIST` to `'9.0'` and, critically,
**sets `DISABLE_AGGRESSIVE_PTX_INSTRS=1` whenever the arch list is not exactly
`'9.0'`** `[verified]`. But the README *does* publish an SM100 table `[reported]`:

| Arch | Topo | dispatch BW | combine BW | #SMs |
|---|---|---|---|---|
| SM100 | EP 8 (NVLink) | 726 GB/s | 740 GB/s | 64 (max perf) |
| SM100 | EP 8 (NVLink) | 643 GB/s | 675 GB/s | 24 (min SM) |
| SM100 | EP 8x2, CX7 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |

`[inferred]`: **DeepEP runs on B200 via SM90 PTX with its aggressive PTX path
disabled.** That is a real performance caveat that the README's benchmark table
does not flag.
Our Docker builds it from `deepseek-ai/DeepEP@9af0e0d0` with branches
`hybrid-ep` and `antgroup-opt` and a fork `fzyzcjy/DeepEP` as alternatives, and
the K3 patch script forces `TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a` `[verified]`
`docker/Dockerfile:10-11,284-308`, `docker/kimi_k3/apply_deepep_k3_patch.sh:22`.
**DeepEP is not installed in the local venv** `[verified]`.
**How we would use it:** not at C1 (EP disables SGLang's shared-expert fusion —
`[verified]` ground-truth §7.5 — changing the MoE kernel shape from 257 back to
256 experts and adding an all-to-all where there was none). At C64 it is the
standard answer, and it is where `n_group=1` bites: a token's 8 experts can land
on all 8 ranks with no group limit `[verified]` ground-truth §1.4.

### NVSHMEM / MSCCL++ / gdrcopy

**NVSHMEM** (NVIDIA, proprietary-but-redistributable): installed **3.4.5**,
latest **3.7.2** `[verified]`/`[reported]`. It is DeepEP's substrate; no SGLang
`srt/` file references it directly (0 hits `[verified]`).
**MSCCL++** — `https://github.com/microsoft/mscclpp`, MIT. Pinned as
`MSCCLPP_VERSION=sglang-v0.9.1` (an SGLang-tagged branch) `[verified]`
`docker/Dockerfile:24,576`; 6 `srt/` files reference it via `pymscclpp.py`.
A fifth allreduce backend, already wired.
**gdrcopy 2.5.1** `[verified]` `docker/Dockerfile:17` — infrastructure, not a
lever for single-node NVLink.

---

## 6. Whole-engine and megakernel implementations

### TileRT — `https://github.com/tile-ai/tilert` — MIT

**The most directly relevant repo in this document, and the one we cannot use.**
**Implements:** a compiler-driven tile-level runtime — "LLM operators are
decomposed into fine-grained tile-level tasks, while the runtime dynamically
reschedules computation, I/O, and communication across multiple devices in a
highly overlapped manner" `[reported]`.
**Hardware: 8x NVIDIA B200, exactly our box** — "the v0.1.5 wheel is specifically
compiled for this configuration"; CUDA 12.9, Python 3.11-3.12, Linux x86_64
`[reported]`.
**Models: DeepSeek-V3.2, GLM-5, GLM-5.1. No GLM-5.2.** `[reported]` — checked
across all six releases; none mentions it.
**Published performance** `[reported]`: up to 600 tok/s DeepSeek-V3.2, up to
500 tok/s GLM-5-FP8, MTP decode up to 590 tok/s synthetic / ~440 tok/s real at
MTP=3, and a "1000 TPS on a 1T model" co-design claim for MiMo-V2.5-Pro.
**Releases** `[reported]`: v0.1.0-alpha.1 (Nov 22) → v0.1.1 (Dec 23, "35% latency
reduction") → v0.1.2-alpha.1 (Jan 26, MTP) → v0.1.3 (Feb 14, GLM-5, 200K context)
→ v0.1.4 (Jun 2) → v0.1.5 / v0.1.5.post2 (Jul 14 / Aug 6, PD disaggregation with
vLLM prefill + TileRT decode). Years `[unverified]`.
**How we would use it:** the ledger's §3/§5-F already frames this correctly —
**run it on GLM-5.1 as a ceiling probe.** Same box, same kernel shapes, an engine
that publishes 500 tok/s. If TileRT gets 500 on GLM-5.1 here and SGLang gets ~365
on GLM-5.2, the gap decomposes into "GLM-5.2 is harder" versus "our scheduling is
worse", and no other experiment separates those. The upstream note that "the
underlying compiler techniques will be gradually shared with the community as
they are integrated into TileLang and TileScale" `[reported]` is the porting path,
and it is slow.
**Also note the honest counter-argument, already in our ledger §2c `[verified]`:**
SGLang implements the same thesis at *micro-batch* granularity (TBO/SBO/allreduce
fusion), all three of which were **off** in every measurement. TileRT's genuine
differentiator is *tile*-granularity overlap, which is the part that still helps
at C1 where there is no batch to split.

### HazyResearch Megakernels — `https://github.com/HazyResearch/Megakernels` — MIT

**Implements:** a single-kernel forward pass for **Llama-1B**, built on
ThunderKittens.
**SM100 status: verified from the Makefile** — `export GPU=H100 # options are
{H100, B200}, else defaults to B200` `[reported]` from the repo page, i.e. B200
is the *default* target.
**Published performance** `[reported]`: <1 ms/forward on H100 (~2.5x vLLM,
>1.5x SGLang), **<680 us on B200** (>3.5x vLLM, >1.5x SGLang), 78% of H100 HBM
bandwidth vs ~50% for existing systems.
**Maintenance:** 804 stars, 6 commits on main — a research artifact, not a library.
**How we would use it: as evidence, not as code.** Dense 1B is a different
universe from 753B MoE with sparse MLA. The transferable claim is the *bandwidth
utilization target*: 78% is what a megakernel buys, and it is the number to hold
our own roofline measurements against once §10 gives us one. Porting is a
quarter-scale project with no MoE, no MLA, and no TP in the existing code.

### Mirage / MPK — `https://github.com/mirage-project/mirage` — Apache-2.0

**Implements:** a compiler+runtime that "automatically transforms LLM inference
into a single megakernel" `[reported]`. Claims 1.2x-6.7x latency reduction over
baselines, with no vLLM/SGLang head-to-head and no absolute tok/s `[reported]`.
**SM100 status: `[unverified]`** — the repo page names no GPU architecture at
all, which for a megakernel compiler in 2026 is itself a signal.
**Maintenance:** 504 commits, **196 open issues** `[reported]`.
**How we would use it:** we would not, yet. Revisit if it publishes a Blackwell
MoE result.

### ThunderKittens — `https://github.com/HazyResearch/ThunderKittens` — MIT

**Implements:** a header-only C++ tile abstraction (CUDA 12.8+, C++20), attention
(causal/non-causal), GEMM, linear attention, and the megakernel demos `[reported]`.
**SM100 status: reported strongly** — TK 2.0 (Jan 2026) "brings full support for
Blackwell GPUs along with MXFP8 and NVFP4 precision"; "mainly built and tested for
Hopper and Blackwell"; Ampere deprecated. Arch selection is via Makefile options
(SM90/SM100/SM120) `[reported]`. `[unverified]` — I did not read the Makefile.
**Published performance:** ~855 TFLOPS on H100 matmul (86% of theoretical)
`[reported]` — an SM90 number.
**How we would use it:** as the authoring layer *if* we conclude we must write a
fused GLM-5.2 decode kernel. Against CUTLASS CuTe DSL and TileLang it trades
ergonomics for less coverage of blockscaled MoE. Not a first move.

---

## 7. DSLs and authoring environments

| repo | license | SM100 | note |
|---|---|---|---|
| **CUTLASS CuTe DSL** — `NVIDIA/cutlass` | BSD-3 | yes `[verified]` via examples | installed as `nvidia-cutlass-dsl==4.6.0`; 131 `srt/` files reference `cute` `[verified]`; FlashInfer's Blackwell CuTe-DSL kernels want "the CUDA 13 extra" `[reported]` |
| **TileLang** — `tile-ai/tilelang` | MIT | reported (SM70-SM120 paths, FlashAttention on SM100) | v0.1.13; we pin 0.1.11. DSA path is **ROCm-only for FP8 KV** in our tree `[verified]` |
| **Triton** — `triton-lang/triton` | MIT | via PTX | 3.6.0 installed; vendored into sgl-kernel at `v3.6.0` `[verified]`. `tokenspeed_triton 3.8.10.post20260721` is also installed `[verified]` — an out-of-tree Triton build, worth understanding before it surprises someone |
| **Helion** — PyTorch | BSD-3 | `[unverified]` | `helion==0.2.6` installed, 4 `srt/` references `[verified]` |
| **QuACK** — `Dao-AILab/quack` | Apache-2.0 | **yes, "B200/B300" named** `[reported]` | CuTe-DSL memory-bound kernels: RMSNorm, softmax, cross-entropy, LayerNorm, GEMM+epilogue. `quack_kernels 0.6.3` installed but **0 `srt/` references** `[verified]` — installed and entirely unused. Our norm share is only 0.5%, so the upside is small but the integration cost is near zero |

---

## 8. KV cache and the transfer plane

None of these is installed `[verified]`. All are C64/multi-node concerns; listed
because the brief asks and because PD disaggregation is candidate G in the ledger.

| repo | license | what | SM100 relevance | how we would use it |
|---|---|---|---|---|
| **LMCache** — `LMCache/LMCache` | Apache-2.0 | KV offload to CPU/SSD/remote, prefix sharing, CacheBlend non-prefix reuse, P2P over NVLink/RDMA/TCP, engine-independent daemon `[reported]` | none (host-side) | our measured 54.2% prefix-cache hit on coding traffic gave 1.54x `[verified]` ledger §4.7 — LMCache extends that beyond GPU HBM. 11.2k stars, active `[reported]` |
| **Mooncake** — `kvcache-ai/Mooncake` | Apache-2.0 | KV-cache-centric disaggregated store + Transfer Engine (RDMA/TCP/NVLink/NVMe-oF, multi-NIC aggregation, topology-aware paths) `[reported]`. Published 87 GB/s on 4x200G RoCE, 190 GB/s on 8x400G `[reported]` | none | already pinned in Docker at `MOONCAKE_VERSION=0.3.12.post1` with `-DUSE_MNNVL=ON -DWITH_EP=ON` `[verified]` `docker/Dockerfile:23-26`. Adopted by SGLang, vLLM, TRT-LLM, NIXL `[reported]` |
| **NIXL** — `ai-dynamo/nixl` | Apache-2.0 | transfer abstraction with UCX/GDS/Mooncake/POSIX/libfabric/GPUNETIO plugins `[reported]` | none | pinned in our ROCm Dockerfile only (`NIXL_COMMIT=c28061f9`) `[verified]`; the CUDA image does not build it |

---

## 9. Quantization toolchains

### NVIDIA TensorRT Model Optimizer — `https://github.com/NVIDIA/TensorRT-Model-Optimizer` — Apache-2.0

**This produced our NVFP4 checkpoint.** `[verified]` — `hf_quant_config.json`
records `producer: modelopt 0.46.0.dev65+g977d34dc3`, `quant_algo: NVFP4`,
group_size 16, `kv_cache_quant_algo: FP8` (ground-truth §2.1).
**Implements:** NVFP4, FP8, W4A8, QAT, distillation, speculative decoding,
pruning `[reported]`. Exports an HF checkpoint consumable by TRT-LLM, vLLM and
SGLang `[reported]`.
**How we would use it:** two specific jobs the shipped checkpoint leaves open.
(1) **Re-quantize the MTP layer.** The NVFP4 build leaves `model.layers.78` in
BF16 — 19.91 GB vs the FP8 build's 10.03 GB, i.e. **+2.36 GiB/GPU under TP8 for
running speculative decoding** `[verified]` ground-truth §9.5. (2) **Calibrate KV
scales.** The checkpoint ships `amax=dynamic` and **no static k_scale/v_scale**
`[verified]` ground-truth §2.1, so our FP8 KV runs uncalibrated — a standing
accuracy caveat in the ledger §6. ModelOpt is the tool for both. It is **not
installed** `[verified]`.

### llm-compressor / compressed-tensors — `https://github.com/vllm-project/llm-compressor` — Apache-2.0

**Implements:** FP8 block, NVFP4, W4A16, GPTQ, AWQ, SmoothQuant, AutoRound, plus
a **REAP expert-pruning modifier for MoE** `[reported]`. Exports
`compressed-tensors` format. 3.7k stars, active `[reported]`.
**Status here:** `compressed_tensors==0.18.0` is installed and SGLang has a
`quantization/compressed_tensors/` package `[verified]` — we can *read* the
format; we do not currently *produce* it.
**How we would use it:** the credible alternative to ModelOpt if we want an FP8
build with a different exclusion policy than Z.ai's, or want to try W4A8 on the
experts. **AutoAWQ/GPTQ successors:** llm-compressor has absorbed both as
modifiers; `auto-round>=0.13.1` also appears in our test extras `[verified]`, and
SGLang ships `quantization/auto_round.py`, `awq/`, `gptq/` `[verified]`. For a
753B MoE none of the INT4 paths is attractive against NVFP4 with hardware
blockscaling.

---

## 10. Tooling: heuristics, SASS, microbenchmarks

### cuBLASLt heuristics — `NVIDIA/CUDALibrarySamples/cuBLASLt` — BSD-3-Clause

**Samples relevant to us** `[reported]`: `LtSgemmSimpleAutoTuning`
("algorithm auto tuning by querying cublasLt heuristics"), `LtSgemmCustomFind`
("running through multiple algo and config attributes combination"),
`LtFp8CustomFind`, `LtFp8Matmul`, **`LtBlk128x128Fp8Matmul`** (the exact block
format of our FP8 checkpoint), **`LtNvfp4Matmul`**, `LtMxfp8Matmul`.
**How we would use it:** `LtSgemmCustomFind` is the template for a standalone
harness that enumerates `cublasLtMatmulAlgoGetHeuristic` results for our ~12
distinct decode GEMM shapes (ground-truth §6.4) and reports how far the default
pick is from the best. Given that one `nvjet` kernel is 12.6% of GPU time, even
a 10% win there is ~1.3% end-to-end — and the harness is a day. It also produces
the artifact needed to argue for or against replacing cuBLASLt at all.

### SM100 SASS tooling — **the gap is real**

**CuAssembler** — `https://github.com/cloudcores/CuAssembler`, MIT — supports
**SM60, SM61, SM70, SM75, SM80, SM86** `[reported]`. **Blackwell is not
supported.** 68 commits; the README itself says the library "is still in its
infancy" and that legal assembly does not imply a legal program `[reported]`.
`[unverified]` — I found no public SM100 assembler or SASS-editing tool in this
session, and lacking WebSearch I cannot claim none exists. **What we do have,
and should use:** `nvdisasm` and `cuobjdump` ship with CUDA 13.2 and will
disassemble the 394 cached `sm100f` cubins today; Nsight Compute gives SASS-level
source correlation and stall reasons. Reading trtllm-gen's SASS is possible;
*editing* it is not.

### Microbenchmarks we could run on idle GPUs

| repo | license | what | Blackwell |
|---|---|---|---|
| **nvbandwidth** — `NVIDIA/nvbandwidth` | Apache-2.0 | H2D/D2H/D2D bandwidth, CE vs SM copy methods, bidirectional, multinode via IMEX; `./nvbandwidth -l` to list, `-t <case>` to select `[reported]` | not stated in docs; CUDA 11+ `[reported]`, so `[inferred]` it runs. **Run this first** — it establishes the NV18 NVLink5 all-to-all number that every collective claim is measured against |
| **gpu-benches** — `te42kyfo/gpu-benches` | **GPL-3.0** | gpu-stream (occupancy sweep 3%→100%), gpu-cache (L1/L2), gpu-l2-cache, gpu-latency (pointer chase), gpu-strides, gpu-roofline, cuda-incore `[reported]` | published results stop at H100/MI300X `[reported]`; no Blackwell numbers |
| **"Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks"** — arXiv 2507.10789 | paper | latency, throughput, cache behavior, scheduling, 5th-gen tensor cores incl. FP4/FP6, power `[reported]` | **measured on a GeForce RTX 5080 (SM120), not B200** `[reported]`. Its numbers do **not** transfer to SM100 — different tensor-core config, different memory system. Useful for method, not for values. No code release stated `[reported]` |

**`[inferred]` and worth saying plainly:** there is no well-known public *B200*
microbenchmark suite with published SM100 numbers that I could locate in this
session. The gpu-benches roofline/occupancy sweep is the closest reusable
harness, and its GPL-3.0 license means we run it as a tool and do not vendor it.

### Near-peak SM100 blockscaled FP4 GEMM — what actually exists

- **trtllm-gen** cubins: the incumbent, executing on this box, no published
  TFLOPS `[verified]` that it runs; `[unverified]` how close to peak.
- **CUTLASS example 72** (`blackwell_narrow_precision_gemm`): the reference
  implementation `[verified]` it exists; no headline number published there.
- **DeepGEMM** FP8xFP4 `[reported]`; **no SM100 performance table published**.
- **Colfax Research** (`https://research.colfax-intl.com/`) publishes the best
  public writeups — "CUTLASS Tutorial: Hardware-supported Block-scaling with
  NVIDIA Blackwell" (Mar 2026), "NVFP4 Blockscaled GEMM on NVIDIA RTX Pro
  Blackwell GPUs (SM12x)" (Jun 2026), "Optimizing an NVFP4 Blockscaled GEMM on
  RTX PRO 6000 Blackwell" (Aug 2026), "Dynamic persistent tile scheduling with
  Cluster Launch Control (CLC)" (May 2026) `[reported]`. **Note the two NVFP4
  articles are SM12x, not SM100** — the block-scaling tutorial is the
  SM100-relevant one.

**`[inferred]` conclusion:** nobody has published a "we hit X% of B200 NVFP4
peak" open-source GEMM. Our best available comparison is internal: measure the
trtllm-gen `Bmm_E2m1_*` cubin's achieved TFLOPS with ncu and compare against
B200 NVFP4 dense peak. That number does not exist anywhere and would be worth
having.

---

## 11. RANKED shortlist: the top 10 to evaluate first

Effort is engineer-days for one person who knows this tree.

| # | thing | the experiment that settles it | effort | kills/confirms |
|---|---|---|---|---|
| 1 | **TokenSpeed MLA** (`tokenspeed_mla` 0.1.8 → 0.2.5) | Bump the pin; force `--attention-backend tokenspeed_mla` past the Kimi-K3 gate in `overrides.py:331,381-398`; A/B decode TPOT and the attention share of an nsys trace at C1 | **0.5-1 d** | Confirms or kills the cheapest possible win against the 10.9% attention share. Failure mode is a `v_head_dim=256` assertion, which is a 10-minute answer |
| 2 | **ncu the top three kernels + nvbandwidth** | `nvbandwidth -t device_to_device_memcpy_*` for the NVLink5 ceiling; ncu on `nvjet_sm100_tst_64x8_...`, `Bmm_E2m1_*sm100f`, `oneshotAllreduceFusionKernel` for achieved BW / fraction of peak | **1 d** | This is the prerequisite the ledger §2b already names. It re-ranks every other row in this table. Do not skip it to feel fast |
| 3 | **Widen `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN`** | Extend beyond `q_b_proj` (`deepseek_v2.py:2234-2243`) to the fused a-projection and `o_proj`; measure dense-GEMM share and gsm8k against the 0.92 registered baseline | **1-2 d** | 33.4 GB of BF16 weights halved on the hottest family. Confirms/kills the largest single structural inefficiency in the NVFP4 build |
| 4 | **FlashInfer 0.6.15.post1 → 0.6.17 + `trtllm_low_latency_gemm`** | Pin bump, then swap the `M=1` dense GEMMs onto the low-latency path; A/B `nvjet` share | **1-2 d** | Directly targets the 37.1%. Risk is real: 0.6.16.post3 was a stability revert `[reported]`, so gate behind an eval |
| 5 | **cuBLASLt heuristic harness** (`LtSgemmCustomFind` / `LtFp8CustomFind` pattern) | Standalone binary enumerating heuristic candidates for our 12 decode shapes (ground-truth §6.4); report default-vs-best gap | **1-2 d** | Tells us whether cuBLASLt is leaving money on the table *before* we spend weeks replacing it. A null result here is worth as much as a win |
| 6 | **Allreduce 4-way bake-off** | At our exact message sizes: trtllm mnnvl one-shot, trtllm two-shot, torch symm-mem multimem, `custom_all_reduce_v2` 1shot_pull — plus an `NCCL_MIN_CTAS`/`NCCL_MAX_CTAS`/`NCCL_CTA_POLICY` sweep | **2 d** | Bounds the 53% of collective time that is actually transfer. All four backends are already in-tree `[verified]`; this is configuration, not code |
| 7 | **TileRT on GLM-5.1** | Install the v0.1.5.post2 wheel (CUDA 12.9, py3.11-3.12, 8xB200 — matches this box); run `sgl-eval` and a latency sweep against SGLang on the same GLM-5.1 weights | **2-3 d** (+ weights download) | The only experiment that separates "GLM-5.2 is harder" from "our scheduling is worse". Ledger candidate F |
| 8 | **Resolve the MNNVL-allreduce contradiction** | Determine why `tllm_mnnvl_allreduce::*` kernels are 12.5% of GPU time when the ledger says allreduce fusion was off in every run | **0.5 d** | Cheap, and until it is resolved no collective measurement means anything |
| 9 | **FlashMLA SM100 sparse prefill** | Build with `compute_100f`; wire `flashmla_backend.py` for the DSA prefill path; measure TTFT against the current 189 ms | **3-5 d** | Its own README publishes 1450 TFLOPS sparse prefill on B200 and warns SM100 sparse *decode* is unoptimized `[reported]`, so scope to prefill. Head-dim 256 is the risk |
| 10 | **DeepGEMM masked grouped GEMM vs trtllm-gen at N=8** | Microbenchmark `sgl-deep-gemm`'s masked grouped path against the cached `Bmm_E2m1_*t128x8x512u2*` cubin at our expert shape (M=1..4, N=256/rank, K=6144) | **2-3 d** | Decides ownership of the 19.4% MoE share. DeepGEMM publishes no SM100 number `[reported]`, so this measurement does not exist publicly |

**Explicitly deferred, with reasons:** DeepEP (C64 only; disables shared-expert
fusion; no SM100 gencode upstream). ThunderKittens / Megakernels / Mirage (no MoE
+ MLA implementation; quarter-scale). Machete and Marlin (**no SM100 gencode in
vLLM's CMakeLists** `[verified]`). humming FP4 (**SM120+ only** `[reported]`).
LMCache / Mooncake / NIXL (C64 and PD disaggregation, not single-stream latency).
CuAssembler (no Blackwell).

---

## 12. Open questions and what to measure on our box

1. `[unverified]` **Why are trtllm MNNVL allreduce kernels 12.5% of GPU time
   when allreduce fusion is documented as off?** Shortlist #8. Everything in §5
   is contingent on this.
2. `[unverified]` **Does FlashMLA's SM100 sparse kernel instantiate
   `v_head_dim=256`?** GLM-5.2's V dim is 256 against DeepSeek's 128
   `[verified]`. Read the template instantiation list before budgeting #9.
3. `[unverified]` **Is `humming` selected on any default path here?** Its FP4 is
   SM120+ `[reported]` and it is a hard dependency `[verified]`. A silent
   fallback would be invisible in a profile.
4. `[unverified]` **Does the sgl-kernel InfLLM arch-flag bug** (`compute_120a`
   under an SM100 gate, no `compute_100a` — `[verified]` `CMakeLists.txt:512-513`)
   affect any kernel we execute? Probably not; confirm and report upstream.
5. `[unverified]` **What fraction of B200 NVFP4 peak does the trtllm-gen expert
   BMM achieve?** Nobody has published this for any implementation. Shortlist #2.
6. `[unverified]` **Which of the 394 cached cubins are actually used**, and does
   FlashInfer's `autotuner/` change the selection for our shapes?
7. `[unverified]` **NCCL 2.28.9 → 2.31.x and NVSHMEM 3.4.5 → 3.7.2**: does the
   CFT device API or TMA-in-symmetric-kernels change our small-message allreduce
   latency? Requires CUDA 13.3+ for CFT `[reported]`; we run 13.2.
8. `[unverified]` **Is there a public SM100 SASS assembler?** I could not
   establish this without WebSearch. Someone with a search budget should spend
   20 minutes; the answer changes what "kernel work" means for us.
9. `[verified] as a gap` **The DSA indexer KV cache is allocated for all 78
   layers but read by 22** — 7,392 B/token wasted, 7.39 GB per 1M tokens per GPU
   (ground-truth §9.4). No repo on this shelf fixes it; it is ours, it is cheap,
   and it is the best HBM-per-line-of-code ratio available.
10. `[unverified]` **TileRT's GLM-5.2 support timeline.** The ledger calls the
    absence "a window that will not stay open." Nothing in the release history
    signals when it closes.

---

## 13. Sources

**Local files (ground truth, `[verified]`)**

- `/home/aman/code/weights/GLM-5.2-{FP8,NVFP4}/config.json`, `hf_quant_config.json`
- `/home/aman/code/research/05-models/00-local-weights-ground-truth.md` (§1-§9)
- `/home/aman/code/NotSglang/python/pyproject.toml` (dependency pins)
- `/home/aman/code/NotSglang/python/sglang/kernels/aot/CMakeLists.txt` (arch gencodes, FetchContent pins)
- `/home/aman/code/NotSglang/python/sglang/kernels/README.md` (registry/selector design)
- `/home/aman/code/NotSglang/docker/Dockerfile` (DeepEP/FlashInfer/Mooncake/MSCCL++/gdrcopy pins)
- `/home/aman/code/NotSglang/docker/kimi_k3/apply_deepep_k3_patch.sh` (`TORCH_CUDA_ARCH_LIST=9.0;10.0a;10.3a`)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/flashinfer_comm_fusion.py`
- `/home/aman/code/NotSglang/python/sglang/srt/distributed/device_communicators/{torch_symm_mem,custom_all_reduce_v2}.py`
- `/home/aman/code/NotSglang/python/sglang/srt/arg_groups/overrides.py` (tilelang/tokenspeed gating)
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/NotSglang/.venv/lib/python*/site-packages` (installed versions)
- `/home/aman/.cache/flashinfer/cubins/` (394 `sm100f` cubins)

**Fetched URLs**

- https://github.com/flashinfer-ai/flashinfer and /releases
- https://docs.flashinfer.ai/api/comm.html
- https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer
- https://github.com/Dao-AILab/flash-attention
- https://github.com/Dao-AILab/quack
- https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/setup.py and /README.md
- https://github.com/deepseek-ai/FlashMLA/tree/main/csrc/sm100
- https://github.com/deepseek-ai/DeepGEMM and raw README.md
- https://github.com/deepseek-ai/DeepEP and raw setup.py, raw README.md
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/CMakeLists.txt
- https://github.com/NVIDIA/cutlass/releases (dates unreliable)
- https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels and /trtllmGenKernels
- https://raw.githubusercontent.com/vllm-project/vllm/main/CMakeLists.txt
- https://github.com/vllm-project/vllm/tree/main/csrc
- https://pypi.org/project/sglang-kernel/
- https://github.com/tile-ai/tilelang
- https://github.com/tile-ai/tilert and /releases
- https://pypi.org/project/tokenspeed-mla/ and https://github.com/lightseekorg/tokenspeed
- https://pypi.org/project/humming-kernels/ and https://github.com/inclusionAI/humming
- https://github.com/HazyResearch/ThunderKittens
- https://github.com/HazyResearch/Megakernels
- https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- https://github.com/mirage-project/mirage
- https://github.com/NVIDIA/nccl/releases
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/index.html
- https://github.com/kvcache-ai/Mooncake
- https://github.com/ai-dynamo/nixl
- https://github.com/LMCache/LMCache
- https://github.com/NVIDIA/TensorRT-Model-Optimizer
- https://github.com/vllm-project/llm-compressor
- https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt
- https://github.com/cloudcores/CuAssembler
- https://github.com/NVIDIA/nvbandwidth
- https://github.com/te42kyfo/gpu-benches
- https://arxiv.org/abs/2507.10789
- https://research.colfax-intl.com/

**Fetches that failed (recorded so nobody repeats them):**
`github.com/sgl-project/sglang/tree/main/sgl-kernel` (404 — moved to
`python/sglang/kernels/`), `raw.githubusercontent.com/sgl-project/sgl-kernel/main/CMakeLists.txt`
(404 — no such standalone repo), `github.com/NVIDIA/nccl/tree/master/ext-tuner[/example]`
(404), `docs.pytorch.org/docs/stable/distributed.symmetric_memory.html` (404).
