# Low-precision inference literature: from INT8 to NVFP4, and what it costs in quality

## What this is

A survey of the quantization literature, read with one question in mind: **what actually runs
fast on Blackwell tensor cores, and what does it cost us in output quality on a 256-expert MoE
with sparse MLA attention, served at both concurrency 1 and concurrency 64?**

Every paper below was fetched and read (abstract + method + evaluation tables) during this pass.
Results carry one of three labels:

- **[verified]** — I read the number in a table or the results text of the paper/model card.
- **[reported]** — the authors claim it in an abstract or blog I read, but I did not see the
  underlying table.
- **[inferred]** — my own reasoning from the numbers, explicitly not a paper claim.

Hardware is stated for every result. A 3.3× speedup on an RTX 3090 with Llama-2-7B does not
transfer to a B200 with a 381B MoE, and saying so is part of the work.

Our deployment context, for reference: 8× B200 SXM (SM100, 183 GB HBM3e, NV18 NVLink5),
GLM-5.2 (MoE 256/8, DSA sparse MLA, `index_topk_freq=4`), NVFP4 and FP8 builds, TP8,
EAGLE 3-1-4 spec decode, `fp8_e4m3` KV cache **with no calibration (scales = 1.0)**.
Measured C1 profile: dense GEMM 37.1%, collectives 19.6%, MoE expert GEMMs 19.4%,
attention 10.9%, DSA indexer 5.8%. 365 tok/s single-stream; 40.8k tok/s aggregate at C64.

---

## Bottom line for our system

Ranked by expected value, highest first.

1. **Our NVFP4 build is probably only quantizing ~19% of our C1 critical path, and that is the
   single biggest finding in this document.** NVIDIA's own reference checkpoint
   (`nvidia/GLM-5.2-NVFP4`, ModelOpt v0.46.0) states: *"Only the weights and activations of the
   linear operators within transformer blocks in MoE experts are quantized. The shared expert is
   not quantized."* [verified — model card]. Our profile says MoE expert GEMMs are 19.4% of C1
   time while **dense GEMM is 37.1%** (attention projections, shared expert, MLA up/down). If our
   build follows that recipe, we are leaving the larger bucket in FP8. **Action: audit which
   modules in our checkpoint are actually NVFP4, then extend NVFP4 to the dense projections and
   measure.** Expected effect at C1: the C1 regime is weight-bandwidth-bound, so halving weight
   bytes on 37.1% of the timeline is worth up to ~1.2× end-to-end on its own [inferred], and
   substantially more when combined with the MoE bucket.

2. **Our uncalibrated `fp8_e4m3` KV is *probably* fine, but MLA is precisely the case the
   literature flags.** vLLM's April 2026 study ran *all* its evaluations at
   `scale = 1.0`, calling it "the worst-case scenario for accuracy," and still got 97–99% recovery
   on reasoning and 93–98% AUC recovery on 1M-token MRCR [verified — vLLM blog]. **But** it
   singles out one failure: Kimi-K2.5 on the FlashMLA backend showed "a consistent downward shift
   across sequence-length buckets," described as "systematic rather than random," and the
   recommendation was to calibrate [verified]. GLM-5.2's DSA sparse MLA is the same structural
   family. **Action: dump per-layer amax of the stored KV latent over 200 real requests and check
   headroom against E4M3's 448 ceiling and 2⁻⁶ normal floor. This is a 30-minute experiment that
   either closes the risk or finds a real bug.** See the KV deep-dive for why scale=1.0 is
   *usually* harmless for FP8 and exactly when it is not.

3. **Standard benchmarks will not detect the damage we care about; agentic benchmarks will.**
   Mix-Quant (arXiv 2605.20315) measured NVFP4 on agentic suites (BFCL v4, LongMemEval,
   τ²-bench): Gemma-4-26B-A4B dropped **66.07 → 55.95** and Qwen3.5-9B **77.31 → 70.37**
   [verified]. Those are 6–10 point drops on the exact workload class we serve, from a
   configuration that would look near-lossless on MMLU. Meanwhile the Huawei/Noah MXFP benchmark
   found reasoning benchmarks recovering **60–75%** under W4A4 where non-reasoning recovered
   **87–97%** [verified]. **Action: our quantization acceptance gate must be τ²-bench + BFCL +
   AIME25 pass@1 over ≥16 samples + MRCR at our max context — not MMLU.**

4. **Keep the router, shared expert, embeddings, lm_head, and the first/last blocks out of FP4.**
   NVIDIA's own NVFP4 *pretraining* recipe keeps ~15–16% of the network in BF16, concentrated at
   the end (first 2 + last 8 blocks of the 12B model) [verified, arXiv 2509.25149]. Penn State's
   layer-wise diagnosis found `down_proj` and `up_proj` are the most FP4-sensitive projections at
   every scale, `q_proj` the least, with `down_proj` activation max/P99.9 ratios of 80–334×
   [verified, arXiv 2603.08747]. MoE-specific work (EAQuant, QuantMoE-Bench) adds: routing
   consistency must be preserved and **shared experts need higher precision** [reported].

5. **If we calibrate anything, calibrate with expert-balanced data.** A 256-expert model with
   top-8 routing means a naive 128-sample calibration set touches most experts a handful of times
   and some near-zero times. MoEQuant (2505.03804) and EAQuant (2506.13329) both identify
   inter-expert and intra-expert calibration imbalance as a first-order error source; EAQuant
   reports **+1.15–13.81%** average accuracy over baselines across three MoE architectures, with
   the largest gains on reasoning [reported]. NVIDIA's DeepSeek-R1 FP4 card used CNN/DailyMail
   [verified] — generic web text, not expert-balanced. **Action: build a calibration set that
   drives every expert at least N times, verified by logging router histograms.**

6. **Block-scale initialization is free accuracy.** Both ScaleSweep (2606.07618) and the
   Huawei MXFP benchmark (2601.09555) find that the default AbsMax block-scale init
   (`scale = amax/6`) is measurably suboptimal, and a cheap search fixes it. The MXFP paper's
   fix is embarrassingly simple: pre-scale inputs by **3/4** before quantizing, which moved
   openPangu-7B W4A4 recovery from **52.39% → 56.76%** [verified]. **Action: if we own the
   quantization step, sweep the block scale; it is a compile-time cost with zero inference cost.**

7. **NVFP4 > MXFP4 is settled, and the gap is large enough to matter.** On the same models with
   plain OCP direct-cast: Llama-3.1-8B-Instruct 6-benchmark average BF16 **70.53**, MXFP4-OCP
   **61.25**, NVFP4 **67.02**; Qwen3-8B BF16 **73.47**, MXFP4-OCP **65.50**, NVFP4 **71.48**;
   DeepSeek-R1 MMLU-Pro BF16 **83.19**, MXFP4-OCP **72.52**, NVFP4 **82.69** [verified,
   arXiv 2603.08713]. Two causes: 16-element blocks instead of 32, and an E4M3 scale with
   mantissa bits instead of a power-of-two E8M0 scale. Do not accept an MXFP4 build for GLM.

8. **FP8 is the safe default and W8A8-FP needs no calibration at all.** "Give Me BF16 or Give Me
   Death?" (ACL 2025) ran >500k evaluations on Llama-3.1 8B/70B/405B: W8A8-FP recovered ~99.3% on
   OpenLLM v1 and 99–101% on the harder V2 + real-world suite, using dynamic per-token activation
   scales and symmetric per-output-channel weight RTN — **no calibration** [verified]. That is our
   floor. Everything below FP8 is a deliberate quality trade, not a free win.

9. **Consider phase-split precision: FP4 prefill, higher-precision decode.** Mix-Quant recovered
   most of the agentic gap by keeping decode in BF16 (Gemma-4-26B 55.95 → 61.67; Qwen3.5-9B
   70.37 → 74.68) [verified] at up to 3× prefill speedup on B200/RTX 5090 with vLLM [reported].
   Their mechanism argument is sound: prefill quantization errors do not feed back within the
   pass, whereas decode is a sequential decision process where a flipped token snowballs. **This
   maps directly onto our two objectives** — FP4 prefill helps TTFT and C64 throughput; keeping
   decode weights at FP8 protects single-stream quality and our 3.09× spec-decode acceptance
   rate, which is itself sensitive to draft/target distribution mismatch [inferred].

10. **Watch for CoT token inflation, not just accuracy.** Two papers disagree: the COLM 2025
    study found "quantized models do not exhibit increased output lengths" [reported,
    arXiv 2504.04823], while arXiv 2606.25519 (June 2026) reports that INT4/INT3 PTQ *preserves
    accuracy while inflating reasoning-token counts* on math, code, and tool-use [reported].
    Token inflation is a direct cost-per-user regression that per-token speedups can mask.
    **Action: log mean output tokens per task in every quantization A/B. It costs nothing.**

---

## Survey by technique family

### Family 1 — The INT8 era: discovering the outlier problem

| Paper | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (arXiv 2208.07339) | Dettmers, Lewis, Belkada, Zettlemoyer (UW / Meta / HF) | NeurIPS 2022 | not stated in fetched text; consumer + A100-class | OPT-13B C4 ppl: FP32 12.45, INT8 absmax 19.08, INT8 vector-wise 16.48, **LLM.int8() 12.45** [verified] | Yes — `bitsandbytes`, HF `load_in_8bit` |
| ZeroQuant (arXiv 2206.01861) | Yao et al., Microsoft DeepSpeed | NeurIPS 2022 | not stated in abstract | W8A8 up to **5.19×/4.16×** speedup; W4A8 3× memory cut [reported] | Yes — DeepSpeed-Inference |
| SmoothQuant (arXiv 2211.10438) | Xiao, Lin, Seznec, Wu, Demouth, Han (MIT / NVIDIA) | ICML 2023 | A100 80GB | OPT-175B zero-shot avg **71.6 → 71.1**; BLOOM-176B 68.2 → 67.4; GLM-130B 73.8 → 72.8 [verified]; up to 1.56×, 2× memory | Yes — TensorRT-LLM `int8_sq`, ONNX Runtime, SageMaker |

### Family 2 — Weight-only PTQ

| Paper | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| GPTQ (arXiv 2210.17323) | Frantar, Ashkboos, Hoefler, Alistarh (ISTA / ETH) | ICLR 2023 | A100, A6000 | 175B quantized in ~4 GPU-hours; **3.25× (A100), 4.5× (A6000)** vs FP16 [reported] | Yes — everywhere; `--quantization gptq` |
| AWQ (arXiv 2306.00978) | Lin, Tang, Tang et al. (MIT Han Lab) | MLSys 2024 (best paper) | RTX 4090, 4070, Jetson Orin | Llama-2 INT4-g128 WikiText2: 7B 5.47→**5.60**, 13B 4.88→**4.97**, 70B 3.32→**3.41** [verified]; TinyChat 2.7–3.9× on 4090 [verified] | Yes — `--quantization awq`, TinyChat |
| SVDQuant (arXiv 2411.05007) | Li, Lin et al. (MIT Han Lab) | ICLR 2025 spotlight | RTX 4090, RTX 5090 | 3.5× memory, 3.0× on 4090, 3.1× on 5090 for FLUX.1-12B [reported] | Nunchaku; **diffusion only, not LLMs** |

### Family 3 — Rotation / Hadamard outlier suppression

| Paper | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| QuaRot (arXiv 2404.00456) | Ashkboos et al. (ETH / ISTA / Microsoft) | NeurIPS 2024 | RTX 3090 | Llama-2 W4A4KV4 WikiText2: 7B 5.47→**6.10** (GPTQ) / 8.37 (RTN); 70B 3.32→**3.79**; 70B zero-shot 77.07→**75.98** [verified]; prefill 2.16–3.33× [verified] | Partially — in ModelOpt/llm-compressor lineage |
| SpinQuant (arXiv 2405.16406) | Liu, Zhao et al. (Meta) | ICLR 2025 | not stated | W4A4KV4 zero-shot avg: Llama-2-7B FP16 66.9 → **64.0** (RTN 37.1); Llama-3-8B FP16 69.6 → **65.5** (QuaRot 63.3, RTN 43.1) [verified]; ~8% online overhead [verified] | Reference impl; used in llm-compressor recipes |
| FlatQuant (arXiv 2410.09426) | Sun, Liu, Bai et al. (Huawei Noah / HKU) | ICML 2025 | not stated in abstract | **<1% drop at W4A4 on Llama-3-70B**, beating SpinQuant by 7.5 pts [reported]; 2.3× prefill / 1.7× decode [reported] | Reference impl only |
| TORQ (arXiv 2605.19561) | Xu, Hu, Yang | arXiv 2026-05 | not stated | Qwen3-32B avg acc: BF16 74.82%, MXFP4 baseline **38.40%**, TORQ **73.63%** [verified] | No — single paper |
| DuQuant++ (arXiv 2604.17789) | Lin et al. | arXiv 2026-04 | not stated | Rotation block size aligned to MX group size; SOTA W4A4 on Llama-3 [reported] | No |
| HadaCore (arXiv 2412.08832) | Agarwal, Astra, Hoque et al. (IBM / Meta / PyTorch) | arXiv 2024-12 | A100, H100 | 1.1–1.4× (A100) / 1.0–1.3× (H100) over prior FWHT kernels, peak 3.5–3.6× [verified] | Yes — PyTorch/torchao ecosystem |

### Family 4 — Low-bit integer serving systems

| Paper | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| Atom: W4A4 (arXiv 2310.19102) | Zhao, Lin, Zhu et al. (UW / SJTU / CMU / OctoML) | MLSys 2024 | RTX 4090 | Llama-2 W4A4 WikiText2: 7B **6.03**, 13B **5.27**, 70B **3.68** [verified]; **7.73× vs FP16**, 2.53× vs INT8 throughput [verified] | Research; ideas absorbed elsewhere |
| QServe / QoQ W4A8KV4 (arXiv 2405.04532) | Lin, Tang, Yang et al. (MIT Han Lab / NVIDIA) | MLSys 2025 | A100, L40S | WikiText2: Llama-2-7B 5.47→**5.67**, 70B 3.32→**3.46**, Llama-3-8B 6.14→**6.70** [verified]; L40S 13B throughput 440→**1327 tok/s** (3.02×) [verified] | `omniserve` reference; W4A8 in TRT-LLM |

### Family 5 — FP8

| Paper | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| FP8 Formats for Deep Learning (arXiv 2209.05433) | Micikevicius et al. (NVIDIA / Arm / Intel) | arXiv 2022 (→ OCP FP8 spec) | H100-era | E4M3 (no inf, 1 NaN pattern) + E5M2; matches 16-bit training quality up to 175B [reported] | **Yes — the standard.** |
| DeepSeek-V3 Technical Report (arXiv 2412.19437) | DeepSeek-AI | arXiv 2024-12 | H800 | Fine-grained FP8: activations **1×128 tiles**, weights **128×128 blocks**; FP32 promotion every N_C=128; **relative loss error <0.25%** vs BF16 over ~1T tokens [verified] | **Yes** — DeepGEMM, copied by SGLang/vLLM for DS-V3/R1 |
| "Give Me BF16 or Give Me Death?" (arXiv 2411.02355) | Kurtic, Marques, Pandit, Kurtz, Alistarh (Neural Magic/RedHat + ISTA) | ACL 2025 | vLLM, A100/H100-class | Llama-3.1 8B/70B/405B, >500k evals: W8A8-FP ~**99.3%** recovery (v1), 99–101% (v2 + real-world); W4A16 ~99.4% (v1), 96.1%+ (v2) [verified] | The methodology is now standard practice |

### Family 6 — Microscaling (MX) and MXFP4

| Paper | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| Microscaling Data Formats for Deep Learning (arXiv 2310.10537) | Rouhani et al. (Microsoft + OCP consortium: NVIDIA, AMD, Intel, Meta, Arm, Qualcomm) | arXiv 2023 → OCP MX v1.0 | mixed | **Block 32, E8M0 scale.** GPT3-175B Lambada direct-cast: FP32 0.755, MXFP6 0.745, **MXFP4 0.623** [verified]. MXINT8 matched FP32 within stdev [verified] | **Yes — the OCP standard**, native on Blackwell + MI355X |
| gpt-oss model card (arXiv 2508.10925) | OpenAI | arXiv 2025-08 | 80GB GPU / 16GB | MXFP4 on **MoE weights = 90+% of parameters, 4.25 bits/param, applied during post-training**; gpt-oss-120b 116.8B total / 5.1B active, 60.8 GB checkpoint, fits one 80GB GPU [verified] | **Yes — shipped production MXFP4 MoE** |
| Unveiling the Potential of Quantization with MXFP4 (arXiv 2603.08713) | Chhugani, Jeong, Su et al. (Meta-affiliated author list) | arXiv 2026 | not stated | Closes MXFP4→NVFP4 gap from ~10% to **<1%** via Overflow-Aware Scaling + Macro Block Scaling; 6.2% prefill GEMM overhead [verified] | No — but the OAS trick is nearly free |
| Benchmarking PTQ of LLMs under MXFP Formats (arXiv 2601.09555) | Zhang, Li, Sun, Bai, Zhen, Dong, Yu (Huawei Noah's Ark) | arXiv 2026-01 | not stated | 7 algorithms × 15 benchmarks × 4 models. W8A8 ~100% recovery; W4A8 ~97.7%; **W4A4 ~94.98% non-reasoning but 60–75% on reasoning** [verified] | Benchmark, not a method |
| Pretraining LLMs with MXFP4 on Native FP4 Hardware (arXiv 2605.09825) | Cim et al. | arXiv 2026-05 | **AMD MI355X** | Weight-gradient quantization is the primary convergence driver; deterministic Hadamard rotations stabilize [reported] | No |

### Family 7 — NVFP4

| Paper / source | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| Pretraining Large Language Models with NVFP4 (arXiv 2509.25149) | NVIDIA | arXiv 2025-09 | GB200 / GB300 | 12B hybrid Mamba-Transformer, 10T tokens: NVFP4 tracks FP8, **<1.5% relative loss error**; MMLU 77.36 (FP8) vs **76.57**; GSM8K-CoT 89.08 vs **92.27**; MBPP++ 59.11 vs **55.91** [verified]. 8B/1T: NVFP4 1.5% vs MXFP4 2.5% loss error; **MXFP4 needs 36% more tokens to match** [verified] | Transformer Engine |
| Nemotron 3 Ultra (arXiv 2606.15007) | NVIDIA (571 authors) | arXiv 2026-06 | Blackwell | 550B total / 55B active MoE hybrid Mamba-Attention, 1M ctx, **pre-trained in NVFP4**; ~6× inference throughput vs SOTA open LLMs [reported] | Yes — released weights |
| Diagnosing FP4 Inference (arXiv 2603.08747) | Cim, Topcu, Kandemir (Penn State) | arXiv 2026-03 | **RTX 5090, RTX 6000 Pro** | Qwen2.5 0.5/7/14B, WikiText-2. Per-projection ΔPPL (7B): `down_proj` NVFP4 **+0.11** vs MXFP4 **+0.50**; `q_proj` +0.02/+0.07. `down_proj` activation max/P99.9 = **80–334×** [verified] | Analysis paper |
| Mix-Quant (arXiv 2605.20315) | Lu, Chen, Fang, Ma, Wang (NUS) | arXiv 2026-05 | **B200 + RTX 5090, vLLM** | Agentic: Gemma-4-26B-A4B BF16 66.07 → NVFP4 **55.95** → Mix-Quant **61.67**; Qwen3.5-9B 77.31 → **70.37** → **74.68** [verified]; up to 3× prefill [reported] | No — but trivially implementable |
| SharQ (arXiv 2606.26587) | Meng, Luo, Zhao et al. | arXiv 2026-06 | RTX 5090 | Training-free, calibration-free; recovers **43–63% of the NVFP4→FP16 gap**; 1.2–1.4× throughput over FP8 [reported] | No |
| ScaleSweep (arXiv 2606.07618) | Lin & Wan (PKU) | arXiv 2026-05 | not stated | Block-scale search beats AbsMax init; full W+A+KV+Q NVFP4 preserves **>93%** of FP performance [reported] | No |
| QUADS (arXiv 2607.15810) | Zhuge, Yu, Wang et al. | arXiv 2026-07 | not stated | NVFP4 RL for MoE **collapses after ~150 steps**; activation error (not weight error) dominates; +21.49 pass@1 over naive NVFP4 RL [reported] | No |
| Full-Stack FP4 (arXiv 2607.04422) | Ding, Ma, Tong et al. (CAS) | arXiv 2026-07 | RTX 5090 | 3B pretraining loss gap **0.838%** (BF16 2.267 vs 2.286); mixed-precision attention keeps softmax-sensitive ops in BF16 [verified] | No |
| `nvidia/GLM-5.2-NVFP4` (HF model card) | NVIDIA | 2026 | **B200 / B300, TP8** | Only MoE-expert linear weights+activations NVFP4; **shared expert not quantized**. vs FP8 baseline: GPQA-D 89.52→**89.39**; SciCode 49.85→**49.04**; IFBench 74.95→**75.81**; AA-LCR 69.38→**70.13**; τ²-Bench Telecom 97.9→**98.25** [verified] | **Yes — SGLang ≥0.23.0, vLLM ≥0.23.0** |
| `nvidia/DeepSeek-R1-0528-FP4` (HF model card) | NVIDIA | 2025 | **8× B200, TRT-LLM** | Linear ops only; CNN/DailyMail calibration; ModelOpt v0.31.0. FP8→FP4: MMLU-Pro 85→**84.2**; GPQA-D 81→**80.0**; LiveCodeBench 77→**76.3**; MATH-500 98→**98.1**; AIME24 89→**91.3** [verified] | **Yes** |

### Family 8 — KV cache quantization

| Paper / source | Lab | Venue / year | Hardware | Headline result | Production? |
|---|---|---|---|---|---|
| KIVI (arXiv 2402.02750) | Liu, Yuan, Jin et al. (Rice / CMU) | ICML 2024 | not stated | **Key per-channel, Value per-token** + full-precision residual window; 2.6× peak memory cut, 2.35–3.47× throughput [reported] | Ideas absorbed; not a default |
| KVQuant (arXiv 2401.18079) | Hooper, Kim, Mohammadzadeh et al. (UC Berkeley) | NeurIPS 2024 | A100-80GB, A6000 | LLaMA-7B WikiText2: 4-bit **5.69 (−0.01)**, 3-bit **5.75 (+0.07)**, 2-bit **6.02 (+0.34)**. Pre-RoPE keys worth +0.65 ppl at 3-bit; NUQ +0.33; 1% dense-and-sparse +0.25 [verified]. Passkey 100% at 3/4-bit to 32K [verified] | Reference impl |
| QServe SmoothAttention (arXiv 2405.04532) | MIT Han Lab | MLSys 2025 | A100, L40S | K has fixed per-head outlier channels **~10× larger** than typical activations; V has none. Scale K down by λ=max(\|K\|)^α, fold into Q [verified] | TRT-LLM W4A8 path |
| RotateKV (arXiv 2501.16383) | — | arXiv 2025-01 | not stated | Outlier-aware adaptive rotations for 2-bit KV [reported — abstract only, not deeply read] | No |
| The State of FP8 KV-Cache and Attention Quantization in vLLM | vLLM project | blog, 2026-04-22 | **H100, H200, B200** | **All evals at uncalibrated scale=1.0.** Qwen3-30B-A3B-Thinking ≤1–2 pt loss (min 97% on GPQA-D); Qwen3.5-27B ≤0.7 pt; MRCR 1M **fully recovers**; B200+FlashInfer 93–96% AUC [verified] | **Yes — `--kv-cache-dtype fp8`** |
| Does Accuracy Equal Evidence? (arXiv 2608.01631) | Ai, He, Guo (UIUC) | arXiv 2026-08 | not stated | 10 eviction methods + 1 quantization method: eviction preserves answers while destroying rationale support; **quantization is "substantially less affected"** [verified] | Evaluation methodology |

### Family 9 — MoE-specific quantization

| Paper | Lab | Venue / year | Headline result | Production? |
|---|---|---|---|---|
| QuantMoE-Bench (arXiv 2406.08155) | Li et al. | arXiv 2024-06 | **Shared experts need higher precision** than sparse routed experts due to consistent activation [reported] | Benchmark |
| MoEQuant (arXiv 2505.03804) | Hu et al. | arXiv 2025-05 | Inter-expert and intra-expert calibration imbalance; expert-balanced sampling + affinity guidance [reported] | No |
| EAQuant (arXiv 2506.13329) | Fu, Zhao, Ding, Yu et al. (Huawei) | arXiv 2025-06 (v3 2026-02) | Expert-aware smoothing + **router logits distribution alignment** + expert-level calibration balance; **+1.15–13.81%** across 3 MoE architectures, largest on reasoning; W4A4/W3A4/W3A3/W2A4 [reported] | Code released |
| MoPEQ (arXiv 2509.02512) | Chitty-Venkata et al. (Argonne) | arXiv 2025-09 | Per-expert bit allocation via Hessian trace [reported] | No |
| Routing-Consistent Quantization of MoE (arXiv 2606.05688) | Park et al. | arXiv 2026-06 | Quantization can flip expert selection; align values *and* decision boundaries [reported] | No |
| Dynamic Expert Quantization (arXiv 2511.15015) | Chu et al. | arXiv 2025-11 | Keep high-traffic experts at higher precision, offload the rest [reported] | No |

### Family 10 — Evaluation methodology

| Paper | Lab | Venue / year | Headline result |
|---|---|---|---|
| Quantization Hurts Reasoning? (arXiv 2504.04823) | Liu, Sun, Zhang, Bai, Yu, Hou (Tsinghua / Huawei) | COLM 2025 | R1-distills 1.5B–70B, QwQ-32B, Qwen3-8B on AIME/MATH-500/GPQA/LiveCodeBench: **W8A8 or W4A16 lossless; below that is risky**. Explicitly: "quantized models do not exhibit increased output lengths" [reported] |
| Quantization Inflates Reasoning (arXiv 2606.25519) | Lian, Krichene, Huang, Tanaka, Ruwase, Zhang, Zhang | arXiv 2026-06 | INT4/INT3 PTQ preserves accuracy **but inflates CoT token counts** on math/code/sci-QA/tool-use; QAT mitigates better than PTQ [reported] — **directly contradicts the above on output length** |
| Why Do Some Inputs Break Low-Bit LLM Quantization? (EMNLP 2025) | Chang, Zhang, Thomason, Jia (USC) | EMNLP 2025 | 3–4 bit weight-only, 7B–70B: errors across **50 method pairs correlate at avg ρ=0.82**; full-precision residual-stream magnitude predicts future quantization error; damaged examples depend on precise late-layer residuals and MLP gate outputs [verified] |
| "Give Me BF16 or Give Me Death?" (arXiv 2411.02355) | Kurtic et al. | ACL 2025 | Warns that tasks with FP baseline <40% give unreliable recovery signal; TruthfulQA had the lowest recovery on v1; GPQA/MuSR too noisy on small models [verified] |

---

## Deep dive 1 — Why block-scaled FP4 works at all

### The problem being solved

A 4-bit float in E2M1 has exactly 16 codes: `±{0, 0.5, 1.0, 1.5, 2, 3, 4, 6}`. Eight magnitudes.
That is a dynamic range of 12× between the smallest normal (0.5) and the largest (6), and a
worst-case relative step of 33% (between 4 and 6). Nothing survives that at tensor granularity.

The reason it works anyway is that **the offending structure in transformer activations is
low-rank and channel-localized, not diffuse**. LLM.int8() established this empirically at INT8:
at ~6.7B parameters a phase transition occurs where outliers appear in all layers and ~75% of
sequence dimensions, but they concentrate in only **6 feature dimensions** out of thousands
(~0.1% of features), with magnitudes 3–20× the normal [−3.5, 3.5] band and observed peaks above
60 [verified, arXiv 2208.07339]. Penn State's FP4 diagnosis measured the same structure in FP4
terms a generation later: `down_proj` inputs have **max/P99.9 ratios of 80–334×**, while
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` sit at **2.6–11×** [verified,
arXiv 2603.08747].

If the outliers were spread uniformly, no block size would help. Because they are localized, a
small enough block *isolates* the outlier: the block containing an outlier burns its scale on
that outlier, and the other N−1 blocks in the row keep a scale matched to their own local range.

This also explains a result that otherwise looks strange in the Penn State paper: `up_proj` is
almost as FP4-sensitive as `down_proj` despite max/P99.9 ratios of only 5.5–10×. Their conclusion
was that "outliers alone don't fully explain FP4 sensitivity" [verified]. Sensitivity is a
function of both the outlier ratio *and* how much the downstream computation amplifies the error
— which is exactly the mechanism the USC EMNLP 2025 paper isolates from the other direction:
damaged examples are those relying on precise late-layer residual activations, and **MLP gate
outputs are the critical component** [verified].

### Block size vs accuracy

Effective bits per weight:

- **MXFP4**: 4 bits + 8-bit E8M0 scale / 32 elements = **4.25 bits/weight**. (Confirmed by the
  gpt-oss card, which states "4.25 bits per parameter" [verified].)
- **NVFP4**: 4 bits + 8-bit E4M3 scale / 16 elements + one FP32 per tensor ≈ **4.5 bits/weight**.
  (The edge-characterization paper computes 4.5078 bits/input at N=4096, block 16 [verified,
  arXiv 2606.06527].)

So NVFP4 buys its accuracy for 0.25 bits/weight — about 6% more memory. On a 381B-parameter MoE
that is roughly 12 GB across 8 GPUs, or 1.5 GB/GPU out of 183 GB [inferred]. Trivially worth it.

Smaller is not monotonically better in practice. The NVFP4 pretraining paper's justification for
16 is that it "more effectively captures the local dynamic range" than 32 while the scale
metadata stays affordable [verified]. The edge study concludes 16 is "a practical
accuracy/storage trade-off" [verified] but does not publish an 8/16/32/64 accuracy sweep, which
is a genuine gap in the literature — I could not find a clean published block-size ablation on a
large model.

The most important quantitative statement about block size is indirect and comes from the
Meta-affiliated MXFP4 paper: with **Macro Block Scaling** they add a second, coarser 1×128 scale
on top of the 1×16 compute blocks and recover almost the entire MXFP4→NVFP4 gap [verified].
That is evidence that the win from NVFP4's 16-element block is at least partly about *scale
representation quality*, not block size per se.

### E8M0 vs E4M3: why the scale dtype matters more than you would think

**E8M0** is an unsigned 8-bit pure exponent — a power of two, no mantissa, range roughly
2⁻¹²⁷ to 2¹²⁷. Its virtues are real: the scale multiply is exact (no rounding error introduced by
the scale itself), the dynamic range is enormous, and the hardware is a shift.

Its defect is that **the scale itself is quantized to power-of-two granularity**. Concretely, from
the MXFP4 OAS analysis [verified, arXiv 2603.08713]: to encode a block you want
`SF = 6.0/amax` so the block's largest element maps exactly onto FP4's top code, 6. But E8M0 must
round `SF` to a power of two, which maps `amax` somewhere into the half-open interval `(3, 6]`.
If your block's `amax` lands at 3.1 after scaling, you are using codes 0.5 through 3 and never
touching 4 or 6 — **you have thrown away the top of a 16-code alphabet**, roughly half a bit,
on that block.

**E4M3** has 3 mantissa bits, so scale granularity is ~1/8 in the mantissa — `amax` lands much
closer to 6.0 and the full FP4 alphabet gets used. The cost is that E4M3's own range is only
±448, so it cannot by itself span the dynamic range of a whole tensor's worth of block maxima.

That constraint is what forces the second level.

### The two-level scaling in NVFP4

NVFP4's decode is:

```
value ≈ e2m1_element × s_block(E4M3) × s_tensor(FP32)
```

Per the paper: "a per-tensor FP32 scale remaps all values within a tensor into representable range
of a block (FP4 × FP8), then a per-block E4M3 scale moves values within a block into FP4
representable range" [verified, arXiv 2509.25149].

The standard encoding, consistent with the llm-compressor NVFP4 documentation ("per-tensor global
scales and per-group (size 16) local quantization scales for weights" [verified]), is
[**inferred** from those two descriptions — verify against ModelOpt source before implementing]:

```
s_tensor  = amax(tensor) / (6.0 * 448.0)          # 448 = E4M3 max, 6.0 = FP4 max
s_block_i = quantize_e4m3( amax(block_i) / (6.0 * s_tensor) )
q_ij      = round_e2m1( x_ij / (s_block_i * s_tensor) )
```

The per-tensor FP32 scale exists purely so the *block scales* fit in E4M3. MXFP4 does not need it
because E8M0 already spans 2±127.

A second, subtler consequence the paper makes explicit: because each block's amax is what defines
the scale, NVFP4 "encodes at least 6.25% of values in a block (the amax values in each block of
16 elements) at near-FP8 precision, while storing the remaining values in FP4" [verified]. That is
a real information-theoretic advantage, and it is why the format is more robust to the localized
outliers described above than the bit count suggests.

### Overflow-Aware Scaling — a nearly free MXFP4 fix worth understanding

Even if we never ship MXFP4, the OAS mechanism is the clearest explanation in the literature of
what E8M0 costs. Standard MXFP4 maps `amax` into `(3, 6]`. OAS observes that when
`amax ∈ [3, 3.5]`, doubling the scale factor pushes the absmax to `[6, 7]` — which *saturates*
the top element but "preserves the relative quantization error," because the relative error is
identical either way and only the single largest element clips. Mapping into `(3.5, 7]` instead
of `(3, 6]` benefits **~15% of blocks** at zero overhead, worth +0.5 dB QSNR [verified].

Macro Block Scaling then adds a 1×128 macro block with an **E0M8** (mantissa-only,
`1 ≤ factor < 2`) scale on top, computed from the max of the eight constituent 1×16 sub-blocks —
i.e., MBS reintroduces exactly the fractional precision that E8M0 threw away, at coarser
granularity. Static variant ~1.1 dB, dynamic (search over 16 candidates minimizing SSE, weights
only) ~1.6 dB [verified].

---

## Deep dive 2 — Head-to-head: NVFP4 vs MXFP4 vs FP8 vs BF16 on the same model

This is the section the assignment specifically asked for. There are exactly three published
comparisons I could find that hold the model fixed and vary only the format.

### (a) Same-model direct-cast comparison (arXiv 2603.08713, Table 1–3) [verified]

6-benchmark average (MMLU-Pro, GSM8K, HellaSwag, Winogrande, ARC-C, ARC-E):

| Model | BF16 | MXFP4-OCP | MXFP4+OAS | MXFP4+MBS-Hybrid | **NVFP4** |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | **70.53** | 61.25 | 63.39 | 66.50 | **67.02** |
| Qwen3-8B | **73.47** | 65.50 | 69.71 | 70.84 | **71.48** |

DeepSeek-R1 (671B MoE — the closest published analogue to our workload):

| Precision | MMLU-Pro | GSM8K |
|---|---|---|
| BF16 | **83.19** | 95.98 |
| MXFP4-OCP | 72.52 | 95.91 |
| MXFP4 + OAS | 75.36 | 96.29 |
| MXFP4 + MBS-Static | 82.37 | 96.82 |
| **NVFP4** | **82.69** | **96.36** |

Read this carefully. Three things jump out:

1. **NVFP4 beats MXFP4-OCP by 5.8 points on Llama-3.1-8B, 6.0 on Qwen3-8B, and 10.2 points on
   DeepSeek-R1 MMLU-Pro.** The gap *widens* on the large MoE.
2. **GSM8K is completely insensitive** — MXFP4-OCP scores 95.91 vs BF16's 95.98 on DeepSeek-R1,
   while MMLU-Pro on the same model and same checkpoint collapses by 10.7 points. If you had
   evaluated only GSM8K you would have shipped a badly broken model. This is the single best
   piece of evidence in the corpus for "your benchmark choice determines whether you see the
   damage."
3. NVFP4 still costs **~3.5 points vs BF16** on Llama-3.1-8B and **~0.5 points** on DeepSeek-R1
   MMLU-Pro. The large model is far more robust — consistent with Penn State's finding that
   per-projection ΔPPL shrinks by ~10× from 0.5B to 7B [verified].

Caveat: the paper presents these as format comparisons under its own quantization pipeline; it is
not stated whether an outlier-suppression algorithm (rotation/AWQ) was applied. Treat these as
the **direct-cast / lightly-processed** frontier, not the best achievable.

### (b) NVFP4 vs FP8 at scale, from production checkpoints [verified — model cards]

`nvidia/DeepSeek-R1-0528-FP4` (671B MoE, 8× B200, TRT-LLM, CNN/DailyMail calibration):

| Benchmark | FP8 | NVFP4 | Δ |
|---|---|---|---|
| MMLU-Pro | 85 | 84.2 | −0.8 |
| GPQA Diamond | 81 | 80.0 | −1.0 |
| LiveCodeBench | 77 | 76.3 | −0.7 |
| SciCode | 40 | 40.1 | +0.1 |
| MATH-500 | 98 | 98.1 | +0.1 |
| AIME 2024 | 89 | 91.3 | +2.3 |

`nvidia/GLM-5.2-NVFP4` (381B MoE, B200/B300, TP8, MoE-expert-linears only):

| Benchmark | FP8 | NVFP4 | Δ |
|---|---|---|---|
| GPQA Diamond | 89.52 | 89.39 | −0.13 |
| SciCode | 49.85 | 49.04 | −0.81 |
| IFBench | 74.95 | 75.81 | +0.86 |
| AA-LCR | 69.38 | 70.13 | +0.75 |
| τ²-Bench Telecom | 97.9 | 98.25 | +0.35 |

**The AIME +2.3 and IFBench +0.86 are not evidence that quantization helps.** They are evidence of
evaluation noise: AIME 2024 is 30 problems, so 89 → 91.3 is under one problem of difference. The
"Give Me BF16" paper makes this point directly — recovery percentages on small/hard benchmarks are
unreliable, especially where the FP baseline is near chance [verified]. Report pass@1 averaged
over ≥16 samples or do not report AIME.

### (c) NVFP4 vs FP8 vs MXFP4 in *training*, same model [verified, arXiv 2509.25149]

- 12B hybrid Mamba-Transformer, 10T tokens: NVFP4 validation loss stays **<1.5% relative error**
  vs the FP8 reference through the stable phase.
- 8B, 1T tokens: NVFP4 **1.5%** relative loss error, MXFP4 **2.5%** — and **MXFP4 required 36%
  more tokens (1.36T vs 1T) to match NVFP4's loss.** That is the cleanest single number in the
  literature quantifying the format gap.
- Downstream (12B, FP8 vs NVFP4): MMLU 77.36/76.57, MMLU-Pro 62.62/62.58, GSM8K-CoT 89.08/92.27,
  MATH 83.32/81.48, HumanEval++ 59.93/57.43, MBPP++ 59.11/55.91, ARC-C 91.81/91.81,
  HellaSwag 83.83/83.09, PIQA 82.64/82.70.
- Note the pattern: **the code benchmarks lose the most** (HumanEval++ −2.5, MBPP++ −3.2), the
  commonsense benchmarks lose nothing. Same lesson as (a).

### (d) Agentic — where the format gap is largest [verified, arXiv 2605.20315]

BFCL v4 / LongMemEval / τ²-bench average, B200 + RTX 5090 with vLLM:

| Model | BF16 | NVFP4 | Mix-Quant (FP4 prefill / BF16 decode) |
|---|---|---|---|
| Qwen3-8B | 42.85 | 38.64 | 41.45 |
| Qwen3.5-9B | 77.31 | 70.37 | 74.68 |
| Gemma-4-26B-A4B-it | 66.07 | **55.95** | 61.67 |
| Gemma-4-31B-it | 77.63 | 76.21 | 77.14 |

Reasoning / long-context average: Qwen3.5-9B 72.04 / 63.26 / 70.59; Gemma-4-26B 71.94 / 66.31 /
71.93.

A 10-point agentic drop on Gemma-4-26B-A4B — an MoE — from a configuration that would score
"1% degradation" on the NVIDIA-style benchmark panel. **This is the number that should make us
nervous, and note that the MoE in the set is the worst affected.**

### Reconciling (b) and (d)

The NVIDIA model cards show ~1 point; Mix-Quant shows up to 10. The differences are:

1. **Scope.** NVIDIA quantizes only MoE-expert linears (GLM) or "linear operators within
   transformer blocks" (R1); Mix-Quant applies NVFP4 to the whole model including decode.
2. **Calibration.** NVIDIA calibrates with ModelOpt; Mix-Quant's NVFP4 baseline appears to be the
   stock vLLM path.
3. **Benchmark class.** NVIDIA reports GPQA/SciCode/MMLU-Pro; Mix-Quant reports BFCL/τ²/LongMemEval.
4. **Model scale.** 671B/381B vs 26B — and large models are consistently more robust.

All four push the same direction, and none of them contradict each other. The honest synthesis:
**a carefully scoped, calibrated NVFP4 build of a very large MoE costs ~1 point on standard
benchmarks; nobody has published what it costs on long-horizon agentic work for those specific
checkpoints, and the one paper that measured that class found 6–10 point drops on smaller models.**

---

## Deep dive 3 — KV cache quantization, and the risk in our uncalibrated `fp8_e4m3`

### The structural facts every KV paper agrees on

1. **Keys have per-channel outliers; Values do not.** KIVI's central finding is that "the key
   cache should be quantized per-channel while the value cache should be quantized per-token"
   [verified, ICML 2024]. QServe measured the magnitude: K has fixed outlier channels in each head
   roughly **10× larger** than typical activations, while "Value matrices show no significant
   outlier pattern" [verified]. This is why QServe's SmoothAttention scales only K, by
   `λ_i = max(|K_i|)^α`, folding `Λ⁻¹` into K and `Λ` into Q so the QKᵀ product is unchanged.
2. **RoPE destroys the channel structure.** KVQuant's largest single win is quantizing keys
   **pre-RoPE**, worth **+0.65 perplexity at 3-bit**, because RoPE "mix[es] pairs of channels by
   different amounts," smearing a clean per-channel outlier pattern across channel pairs
   [verified].
3. **Keys are more sensitive than values.** QuaRot's ablation: 3-bit K and V cost ≤0.07 ppl on
   Llama-2-70B, with keys the more sensitive of the two [verified].
4. **A tiny sparse set carries disproportionate weight.** KVQuant keeps 1% of entries in FP16
   (+0.25 ppl at 3-bit) and keeps the **attention sink / first token** in FP16, which "provides
   consistent gains across bit widths, particularly pronounced at 2-bit" [verified].

### Why per-tensor scale = 1.0 is *usually* harmless for FP8 — and exactly when it is not

This is the crux for our deployment, and the reasoning is different from INT8 intuition.

For **INT8**, the scale *is* the quantization step. Get it wrong by 4× and you either clip or you
waste 2 bits. Calibration is mandatory.

For **FP8 E4M3**, relative precision is **scale-invariant within the normal range**. E4M3 has 3
mantissa bits, so the relative rounding error is ~2⁻⁴ = 6.25% worst case, ~3% typical, *regardless
of whether your values are around 0.1 or around 100*. Multiplying everything by a constant just
moves you along the exponent axis; it does not change the mantissa resolution.

So an FP8 KV scale only does two things:

- **Prevent saturation at the top.** E4M3 max is **±448.0** [verified, vLLM docs]. Values above
  that clip.
- **Prevent underflow at the bottom.** E4M3's smallest normal is 2⁻⁶ ≈ 0.0156; below that you are
  in subnormals and progressively lose mantissa bits, hitting zero at 2⁻⁹ ≈ 0.00195.

With `scale = 1.0` you get the raw tensor values cast to E4M3. **If your K/V values live inside
roughly [0.02, 400], scale=1.0 is not merely acceptable — it is optimal, because any rescaling
would only risk pushing you toward one of the two cliffs.** For typical post-RoPE K and V in
O(0.1)–O(10), that is exactly the situation. This is why vLLM was able to publish a whole study
at scale=1.0 and call the results good.

### What the literature says the risk actually is

vLLM's April 2026 study is the most directly applicable source we have, because it used our exact
configuration [all verified from the blog]:

- **Setup:** "All evaluations use per-tensor uncalibrated quantization scales (i.e., scale = 1.0)"
  — described as "the worst-case scenario for accuracy." `--kv-cache-dtype fp8` quantizes the
  cache **and runs the QK and ScoreV matmuls in FP8 e4m3**.
- **Reasoning:** Qwen3-30B-A3B-Thinking-2507 lost at most 1–2 points, minimum recovery **97%**
  (GPQA-Diamond). Qwen3.5-27B lost at most 0.7 points, minimum recovery **99%** (AIME25).
- **Long context (MRCR):** Llama-3.3-70B-Instruct at 128k recovered **97–98%** of baseline AUC.
  Qwen3-30B-A3B-Instruct at 256k recovered **94–98%**. Qwen3.5-27B at 1M **fully recovered** the
  aggregated AUC@1M. On **B200 with FlashInfer**, 93–96% AUC recovery.
- **The failure case — and it is ours.** "Models with non-standard kernels (e.g. Kimi-K2.5 with
  FlashMLA) showed a consistent downward shift across sequence-length buckets… the degradation is
  systematic rather than random." The recommendation: "start with uncalibrated FP8 because it is
  simple and often good enough, but calibrate if you observe this kind of persistent downward
  shift on your real workload."
- **A separate Hopper-only hazard, which we should still understand.** FA3 on Hopper loses
  precision when the contraction dimension is large; on a 128k needle-in-a-haystack task accuracy
  fell from **91% (BF16) to 13% (FP8)**. The fix — explicit FP32 register writes between
  accumulation steps — restored **89%** but raised prefill latency, with **~1.6× TTFT for
  head_dim = 256**. The blog states FlashInfer on Blackwell has no such accumulation issue.
- **Explicit "don't use FP8 KV if" list:** contexts under ~7k tokens; `head_dim = 256` with
  TTFT-sensitive prefill; uncalibrated accuracy under 95% on your workload; many small
  sliding-window attention layers (use `--kv-cache-dtype-skip-layers sliding_window`).
- **Break-even:** H100 ~7k tokens, **B200 ~4k tokens** (Llama-3.1-8B). ITL slope 54% of BF16.
  Under throughput load: +14.9% output throughput, −14.8% median ITL.

### Why MLA specifically is the risk

An MLA/DSA KV cache does not store per-head K and V. It stores a **compressed latent**, produced
by a learned down-projection, whose per-element statistics are not the O(1) post-RoPE values that
scale=1.0 is implicitly tuned for. A latent with a larger norm concentrated in fewer dimensions
is exactly the shape that saturates 448, and a latent with a very small norm is exactly the shape
that falls into E4M3 subnormals. **Neither failure mode is visible in aggregate benchmark
accuracy until it is catastrophic** — which is why the Kimi-K2.5 signature was a *shift*, not a
cliff.

Our DSA indexer adds a second concern: the index/top-k selection consumes the same cached state.
A quantization error that is harmless for the attention weighted-sum can flip a top-k selection,
which is a discrete decision with no error-averaging. Nobody has published on FP8 KV interacting
with learned sparse-attention index selection — this is genuinely uncharted, and worth treating
as such.

### The concrete experiments that would expose it

Ranked by cost-to-information ratio:

1. **Instrument the cache (30 min, zero risk).** Over ~200 real requests, log per-layer
   `amax(|kv_latent|)`, `p99.9`, and the fraction of elements below 2⁻⁶. If amax is comfortably
   under ~100 and the sub-2⁻⁶ fraction is small, scale=1.0 is proven safe on both cliffs and we
   can stop worrying. If amax approaches or exceeds 448 in any layer, we have found a real bug.
2. **MRCR at our maximum served context, FP8-KV vs BF16-KV, AUC recovery.** This is the exact
   metric vLLM used and the one that separated the good models from Kimi-K2.5. Accept ≥95%.
3. **RULER at 128k**, specifically the multi-key NIAH, variable-tracking, and aggregation
   subtasks. Single-needle passkey is too easy — KVQuant showed 100% passkey retention even at
   3-bit to 32K [verified], so passkey proves nothing.
4. **Per-example KL divergence against a BF16-KV reference** on a few hundred held-out prompts.
   This is far more sensitive than aggregate accuracy: the USC EMNLP 2025 result is that damage
   concentrates on a predictable minority of inputs (errors across methods correlate at ρ=0.82),
   so a mean accuracy that moves 0.3 points can hide a tail that moves a lot [verified].
5. **DSA top-k agreement rate**: fraction of positions where the FP8-KV indexer selects the same
   top-k set as BF16-KV. Cheap, and directly measures the failure mode nobody has published on.
6. **τ²-bench / BFCL with FP8 KV vs BF16 KV**, given Mix-Quant's finding that agentic is where
   4-bit damage shows up first [verified] — plausibly the same ordering holds for KV precision.

### If we do need to calibrate

- vLLM supports calibrated scales through `llm-compressor`, and **per-head scale arrays** on the
  FA3 kernel [verified, vLLM blog + docs]. Per-head is the granularity that matters, since the
  K outlier channels QServe measured are per-head phenomena.
- The historical vLLM doc is explicit that "If this JSON is not specified, scaling factors default
  to 1.0" and that scales are "typically obtained when running an unquantized model through a
  quantizer tool" [verified].
- `--kv-cache-dtype-skip-layers` lets us keep specific layers in BF16 — the right escape hatch if
  the instrumentation finds one or two bad layers rather than a global problem.

---

## Deep dive 4 — Evaluation methodology: what actually detects quantization damage

### Benchmarks ranked by sensitivity, from the papers that measured it

**Insensitive — will not detect real damage:**

- **GSM8K.** DeepSeek-R1 under MXFP4-OCP: 95.91 vs 95.98 BF16, while MMLU-Pro on the *same*
  checkpoint fell 83.19 → 72.52 [verified, arXiv 2603.08713]. A 10.7-point failure invisible to
  GSM8K.
- **Commonsense QA (PIQA, HellaSwag, Winogrande, ARC-Easy).** The NVFP4 pretraining paper's 12B
  results move by <1 point on all of these while code benchmarks move by 2.5–3.2 [verified].
- **Single-needle passkey retrieval.** 100% at 3-bit and 4-bit KV to 32K [verified, KVQuant].
- **Perplexity alone.** The Penn State work measures everything in WikiText-2 ΔPPL and explicitly
  flags this as a limitation, recommending extension to "reasoning and coding benchmarks"
  [verified].

**Moderately sensitive:**

- MMLU-Pro, GPQA-Diamond, ARC-Challenge (the MXFP benchmark found ARC-C consistently more
  sensitive than ARC-E [verified]).
- HumanEval++/MBPP++ — the largest movers in the NVFP4 12B downstream table [verified].

**Most sensitive — these are the gates:**

- **Agentic / tool-use suites** (BFCL v4, τ²-bench, LongMemEval): 6–10 point NVFP4 drops
  [verified, Mix-Quant].
- **Reasoning under W4A4**: 60–75% recovery vs 87–97% for non-reasoning on the same models
  [verified, arXiv 2601.09555].
- **Long-context AUC (MRCR, RULER multi-key)**: the metric vLLM used to separate healthy FP8-KV
  models from Kimi-K2.5 [verified].
- **Per-example divergence from the FP baseline**: the ρ=0.82 correlation result means damage is
  systematic and predictable per-input, so aggregate means understate it [verified, EMNLP 2025].

### Perplexity vs downstream accuracy

The literature does not support a clean "perplexity is useless" verdict, but it does support
"perplexity is necessary and nowhere near sufficient." QuaRot, Atom, QServe, KVQuant, and the
Penn State diagnosis all rank methods by WikiText-2 perplexity and the rankings are stable and
useful — QoQ-g128 at 5.67 vs Atom at 6.12 vs QuaRot at 6.00 on Llama-2-7B is a real ordering
[verified]. What perplexity cannot see is (a) the reasoning/agentic collapse at W4A4, (b) the
per-example tail, (c) CoT token inflation, and (d) long-context retrieval degradation.

### The "quantization hurts reasoning more" claim — is it supported?

**Partially, with important qualifications.**

*Supporting:*
- The Huawei MXFP benchmark is the cleanest evidence: on the same models, same formats, reasoning
  benchmarks (MATH-500, AIME24, AIME25) recovered **60–75%** under W4A4 while non-reasoning
  benchmarks recovered **87–97%** [verified].
- Mix-Quant's entire premise, with a mechanistic argument: decode is "a sequential decision
  process" where "token prediction changes can trigger a snowball effect," whereas prefill errors
  "do not recursively affect future inputs within the same prefill pass" [verified]. Long
  reasoning chains are long decode.
- EAQuant reports its gains are "particularly pronounced in reasoning tasks" [reported].

*Qualifying:*
- The COLM 2025 study concluded that at **W8A8 or W4A16 the loss is essentially zero even on
  reasoning** across R1-distills 1.5B–70B, QwQ-32B, and Qwen3-8B on AIME/MATH-500/GPQA/
  LiveCodeBench [reported]. The claim is scoped to *aggressive* quantization, not quantization
  in general.
- NVIDIA's DeepSeek-R1 FP4 card shows MATH-500 98 → 98.1 and AIME24 89 → 91.3 [verified] — a
  671B MoE at 4 bits with no reasoning degradation on those two benchmarks. Scale and calibration
  quality matter enormously.

*Genuinely unresolved:*
- **Output length.** COLM 2025: "quantized models do not exhibit increased output lengths"
  [reported]. arXiv 2606.25519 (14 months later): INT4/INT3 PTQ *does* inflate CoT tokens on math,
  code, sci-QA, and tool use, with a "CoT Token Inflation Ratio" metric [reported]. Different
  models, different bit-widths, and I did not obtain the underlying tables for either. **Measure
  it yourself; it is one number and it is free.**

*The best-supported synthesis:* it is not that reasoning is uniquely fragile; it is that
**reasoning tasks have long serial decode chains, and serial decode is where quantization error
compounds.** Anything with that structure — reasoning, agentic loops, long tool-calling
trajectories — inherits the sensitivity. That predicts our agentic workload is the exposed one,
and Mix-Quant's numbers confirm it.

### Additional methodology notes worth adopting

- **Don't trust benchmarks where the FP baseline is near chance.** "Give Me BF16" found GPQA and
  MuSR unreliable on smaller models for exactly this reason, and TruthfulQA had the lowest
  recovery on OpenLLM v1 [verified].
- **Report generation similarity, not just accuracy.** That paper used ROUGE-1/ROUGE-L/BERTScore/
  STS against the FP16 model's own outputs: 70B/405B held ROUGE-1 0.7 and ROUGE-L 0.56; 8B fell
  to 0.62/0.46 while BERTScore stayed at 0.92 [verified]. Semantic metrics saturate; lexical
  metrics still discriminate.
- **Measure the rationale, not just the answer.** "Does Accuracy Equal Evidence?" (arXiv
  2608.01631) introduces the *answer-evidence gap* — final accuracy preserved while the visible
  supporting chain degrades. Their good news for us: this is mainly a **token-eviction** pathology;
  "a coverage-preserving quantization control is substantially less affected" [verified].
- **Calibration data quality is a first-order variable.** "Give Me BF16" found that W4A16 GPTQ
  with OpenPlatypus calibration beat random-token calibration substantially, and that GPTQ's
  advantage over AWQ on coding tasks came from MSE-optimal clipping plus better calibration data
  "rather than fundamental algorithmic advantages" [verified].

---

## What is NOT worth it

**1. W4A4 integer (Atom, QuaRot-INT, QServe-style W4A4) on Blackwell.**
The whole point of INT4 W4A4 was to use INT4 tensor cores that Blackwell has largely superseded
with native block-scaled FP4. QServe's own motivating measurement is that W4A4 systems suffer
"20% to 90%" runtime overhead from dequantization on CUDA cores, where each such op costs the
equivalent of "50 tensor core MACs" [verified]. NVFP4 does the scaling **in the tensor core**, so
it dominates. And the accuracy is worse: QuaRot W4A4KV4 costs Llama-2-7B 5.47 → 6.10 ppl and 70B
zero-shot 77.07 → 75.98 [verified], against NVFP4's ~1 point on much larger models.

**2. MXFP4 for a model we care about.** 5.8–10.2 points behind NVFP4 on the same models
[verified, arXiv 2603.08713]; needs 36% more training tokens to reach the same loss [verified,
arXiv 2509.25149]. Blackwell supports both; there is no throughput reason to pick MXFP4. The one
counter-case is gpt-oss, where OpenAI applied MXFP4 to MoE weights **during post-training**
[verified] — i.e., the model was adapted to the format. Direct-cast MXFP4 on a model trained in
BF16/FP8 is a different and much worse proposition.

**3. Learned-rotation methods (SpinQuant, FlatQuant) as an inference-time addition to NVFP4.**
The accuracy case is real for W4A4 *integer* — SpinQuant takes Llama-3-8B from RTN's 43.1 to 65.5
zero-shot average [verified]. But the online rotations cost ~8% latency [verified, SpinQuant] and
QuaRot reports ≤7% [verified], and that is on top of an already-tight decode loop. For us, at
C1, 8% is ~29 tok/s off 365 — more than we would gain from most other optimizations. If we ever
need it, the correct form is **block-size-aligned rotation** (TORQ, DuQuant++) so the rotation
mixes within the 16-element scaling group rather than across it, and HadaCore for the kernel.
Not now.

**4. INT8 KV cache.** vLLM's `kv_cache_dtype` accepts only `fp8`, `fp8_e4m3`, and `fp8_e5m2`
[verified, vLLM docs] — there is no INT8 KV path. FP8 E4M3 is strictly easier anyway
(scale-invariant relative precision, so a bad scale only risks the two cliffs rather than the step
size) and is what the Blackwell attention kernels are built around. No reason to reach for INT8.

**5. 2-bit KV cache.** KIVI and KVQuant both make 2-bit work on *perplexity* (KVQuant LLaMA-7B
2-bit: 6.02 vs 5.69 FP16, +0.34) [verified], and both need machinery we would have to build:
per-channel pre-RoPE key layout, non-uniform k-means codebooks, a 1% FP16 sparse side-channel,
attention-sink special-casing, and a full-precision residual window. For a serving system whose KV
is already FP8 and whose prefix cache is worth 1.54×, the memory is not the binding constraint.
The engineering cost is enormous and the retrieval risk is real.

**6. Quantizing the router, the shared expert, embeddings, or lm_head.** NVIDIA's GLM-5.2 card
excludes the shared expert [verified]; DeepSeek-V3's FP8 training excludes "the embedding module,
the output head, MoE gating modules, normalization operators, and attention operators"
[verified]; llm-compressor's NVFP4 recipe ignores `lm_head` [verified]; QuantMoE-Bench finds
shared experts need higher precision [reported]. These are a rounding error in parameter count and
a large fraction of the risk.

**7. Full-model NVFP4 including decode, if single-stream quality matters.** Mix-Quant's Gemma-4-26B
agentic result (66.07 → 55.95) is the cautionary number [verified]. And QUADS found NVFP4 MoE
training "collapses after roughly 150 steps" with **activation error, not weight error**, as the
dominant cause [reported] — the same asymmetry the edge-characterization paper found for
inference: "FP8 and FP16 weights provide only modest gains over FP4 weights under the same NVFP4
activation path, suggesting that activation quantization and scaling dominate much of the accuracy
behavior" [verified, arXiv 2606.06527]. **If you must keep something at higher precision, keep the
activations, not the weights.** That is also the memory-cheap choice — weight-only NVFP4
(`NVFP4A16` in llm-compressor [verified]) is the conservative first step.

**8. Chasing AIME/GSM8K deltas.** Both are too small and too saturated. The AIME 89 → 91.3 on
NVIDIA's R1-FP4 card is under one problem [inferred]; GSM8K is insensitive to a 10-point MMLU-Pro
collapse [verified]. Neither belongs in an acceptance gate without ≥16-sample averaging.

---

## The accuracy-vs-speed frontier for a large MoE on B200

### What the best-known published configuration is

Synthesizing the production checkpoints and the recipe papers, the current frontier for a
300B–700B MoE on Blackwell looks like this:

| Component | Precision | Source / justification |
|---|---|---|
| MoE routed expert linears (W and A) | **NVFP4**, block 16, E4M3 block scale + FP32 tensor scale | `nvidia/GLM-5.2-NVFP4`, `nvidia/DeepSeek-R1-0528-FP4` [verified] |
| Shared expert | **FP8** (not quantized to FP4) | GLM-5.2-NVFP4 card explicitly [verified]; QuantMoE-Bench [reported] |
| Attention QKV/O projections | **FP8** (NVFP4 possible but unvalidated for MLA) | DeepSeek-V3 keeps attention operators out of FP8 entirely [verified]; Full-Stack FP4 keeps softmax-sensitive ops in BF16 [verified] |
| Softmax / attention accumulation | **FP32** | Full-Stack FP4 [verified]; vLLM FA3 accumulation fix [verified] |
| KV cache | **FP8 e4m3**, per-tensor or per-head scales | GLM-5.2-NVFP4 card ships `kv-cache-dtype fp8_e4m3` [verified] |
| Router / gate | **BF16** | DeepSeek-V3 [verified]; routing-consistency literature [reported] |
| Embeddings, lm_head, norms | **BF16** | DeepSeek-V3 [verified]; llm-compressor ignores `lm_head` [verified] |
| First ~2 and last ~8 blocks | Consider **FP8/BF16** | NVFP4 pretraining keeps 15–16% high-precision, weighted to the end [verified] |
| Calibration | Expert-balanced, not generic web text | MoEQuant / EAQuant [reported]; NVIDIA used CNN/DailyMail [verified] |
| Block scale init | Swept, not AbsMax | ScaleSweep [reported]; MXFP benchmark 3/4 pre-scale [verified] |

### What it costs in quality

- **~1 point or less on standard reasoning/knowledge/code benchmarks vs the FP8 build.** DeepSeek-R1:
  MMLU-Pro −0.8, GPQA-D −1.0, LiveCodeBench −0.7 [verified]. GLM-5.2: GPQA-D −0.13,
  SciCode −0.81 [verified].
- **Unknown, plausibly 2–6 points, on long-horizon agentic work.** Nobody has published this for
  the large-MoE production checkpoints. The one data point in the right direction is GLM-5.2's
  τ²-Bench Telecom at 97.9 → 98.25 [verified] — but that is a single, near-saturated agentic
  benchmark. The one data point in the wrong direction is Mix-Quant's 10-point Gemma-4-26B-A4B
  drop [verified]. **This is the gap we should close ourselves.**
- **Roughly 3–4 points if you go to NVFP4 *without* the exclusions and calibration** — the
  direct-cast Llama-3.1-8B number (70.53 → 67.02) [verified].

### What it buys in speed, mapped onto our profile

Blackwell FP4 tensor cores run at **2× the FP8 math rate on GB200 and 3× on GB300** [verified,
arXiv 2509.25149], and NVFP4 halves weight bytes vs FP8. Those two facts pay off in different
regimes:

- **At concurrency 1, we are weight-bandwidth-bound, not FLOP-bound.** With 8 of 256 experts
  active per token, every expert GEMM is a skinny matrix-vector — the tensor cores are idle
  waiting on HBM. NVFP4's win here is entirely **bytes moved**. Our C1 profile has 37.1% dense
  GEMM + 19.4% MoE expert GEMM = **56.5% of the timeline in weight-load-bound GEMMs**. If the
  current build quantizes only the 19.4%, extending NVFP4 to the dense projections attacks
  roughly twice as much of the critical path [inferred].
- **At concurrency 64, we shift toward compute-bound**, and the 2× FP4 math rate on B200 starts
  to matter directly for the aggregate 40.8k tok/s number [inferred].
- **NVFP4 does nothing for our 19.6% collectives bucket**, and since 47% of that is rank-arrival
  skew rather than transfer, making the GEMMs faster may make the skew *relatively* worse unless
  the skew source is addressed independently [inferred].
- **Reference points for scale:** NVIDIA reported >250 tok/s/user and >30,000 tok/s aggregate for
  DeepSeek-R1-671B in FP4 on a single 8-GPU DGX B200, >3× a DGX H200 running FP8 [reported].
  The GLM coding-agent case study measured "approximately 185 tok/s under interactive agent load"
  for GLM-5.1/5.2 (753B total / 40B active) at NVFP4 + FP8 KV on **4× B200** with vLLM [verified,
  arXiv 2607.13080]. Our 365 tok/s on 8× B200 is comfortably ahead of that data point; TileRT's
  ~500 tok/s remains the bar.

### One field report worth reading carefully, and not over-reading

arXiv 2607.13080 (Peng, Lin, Lee) is the only published deployment study I found of a GLM-class
model at NVFP4 in a real agentic workload. Their measurement: **Fix Commit Ratio 74.93%** for
on-prem GLM-5.2-NVFP4 vs **45.93%** for the Claude Opus API, Mantel-Haenszel odds ratio **3.61
(95% CI 2.46–5.30)** — i.e. the local deployment produced far more repair commits per unit of
work. TCO under shared allocation was 40.1% lower [verified].

**But the authors themselves disclaim the causal reading**, and we should honor that: "We did not
deploy FP16 or FP8 baselines of GLM-5.2 for a controlled quantization ablation… our defect analysis
cannot empirically separate quantization-induced degradation from the base model's intrinsic
capability gap" [verified]. The gap bundles base-model capability (AA Intelligence 56 vs 51),
NVFP4, vLLM serving, and harness co-design (Claude Code is co-tuned with Opus; Opencode is
provider-agnostic). It is a warning that an NVFP4 agentic deployment can feel materially worse in
production, and it is **not** evidence that NVFP4 caused it.

This is precisely the ablation we are positioned to run and they were not: same model, same
harness, same traffic, FP8 vs NVFP4. If we do it, it is a publishable result.

---

## Sources

**Papers fetched and read (38).** All arXiv IDs and venues verified by fetching the listed URL.

INT8 era:
1. Dettmers, Lewis, Belkada, Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339 — https://arxiv.org/abs/2208.07339 (full text via https://ar5iv.labs.arxiv.org/html/2208.07339)
2. Yao, Yazdani Aminabadi, Zhang, Wu, Li, He (Microsoft DeepSpeed). *ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers.* NeurIPS 2022. arXiv:2206.01861 — https://arxiv.org/abs/2206.01861
3. Xiao, Lin, Seznec, Wu, Demouth, Han (MIT / NVIDIA). *SmoothQuant.* ICML 2023. arXiv:2211.10438 — https://arxiv.org/html/2211.10438v6

Weight-only PTQ:
4. Frantar, Ashkboos, Hoefler, Alistarh (ISTA / ETH). *GPTQ.* ICLR 2023. arXiv:2210.17323 — https://arxiv.org/abs/2210.17323
5. Lin, Tang, Tang et al. (MIT Han Lab). *AWQ.* MLSys 2024 (best paper). arXiv:2306.00978 — https://ar5iv.labs.arxiv.org/html/2306.00978
6. Li, Lin et al. (MIT Han Lab). *SVDQuant.* ICLR 2025 spotlight. arXiv:2411.05007 — https://arxiv.org/abs/2411.05007

Rotation / Hadamard:
7. Ashkboos, Mohtashami et al. (ETH / ISTA / Microsoft). *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS 2024. arXiv:2404.00456 — https://arxiv.org/html/2404.00456v2
8. Liu, Zhao, Fedorov et al. (Meta). *SpinQuant: LLM Quantization with Learned Rotations.* ICLR 2025. arXiv:2405.16406 — https://arxiv.org/html/2405.16406v4
9. Sun, Liu, Bai et al. (Huawei Noah's Ark). *FlatQuant: Flatness Matters for LLM Quantization.* ICML 2025. arXiv:2410.09426 — https://arxiv.org/abs/2410.09426
10. Xu, Hu, Yang. *TORQ: Two-Level Orthogonal Rotation for MXFP4 Quantization.* arXiv:2605.19561 — https://arxiv.org/abs/2605.19561
11. Agarwal, Astra, Hoque, Srivatsa, Ganti, Wright, Chen. *HadaCore: Tensor Core Accelerated Hadamard Transform Kernel.* arXiv:2412.08832 — https://arxiv.org/abs/2412.08832

Low-bit serving systems:
12. Zhao, Lin, Zhu et al. (UW / SJTU / CMU). *Atom: Low-bit Quantization for Efficient and Accurate LLM Serving.* MLSys 2024. arXiv:2310.19102 — https://arxiv.org/html/2310.19102v3
13. Lin, Tang, Yang et al. (MIT Han Lab / NVIDIA). *QServe: W4A8KV4 Quantization and System Co-design.* MLSys 2025. arXiv:2405.04532 — https://arxiv.org/html/2405.04532v3

FP8:
14. Micikevicius et al. (NVIDIA / Arm / Intel). *FP8 Formats for Deep Learning.* arXiv:2209.05433 — https://arxiv.org/abs/2209.05433
15. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437 — https://arxiv.org/html/2412.19437v2
16. Kurtic, Marques, Pandit, Kurtz, Alistarh. *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization.* ACL 2025. arXiv:2411.02355 — https://arxiv.org/html/2411.02355v4

Microscaling / MXFP4:
17. Rouhani et al. (Microsoft + OCP consortium). *Microscaling Data Formats for Deep Learning.* arXiv:2310.10537 — https://ar5iv.labs.arxiv.org/html/2310.10537
18. OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card.* arXiv:2508.10925 — https://ar5iv.labs.arxiv.org/html/2508.10925
19. Chhugani, Jeong, Su, Pan, Yang, Ankit, Yu, Deng, Chen, Satish, Kim. *Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction.* arXiv:2603.08713 — https://arxiv.org/html/2603.08713v1
20. Zhang, Li, Sun, Bai, Zhen, Dong, Yu (Huawei Noah's Ark). *Benchmarking Post-Training Quantization of LLMs under Microscaling Floating Point Formats.* arXiv:2601.09555 — https://arxiv.org/html/2601.09555v1
21. Hu, Zhang, Zhang et al. *M2XFP: A Metadata-Augmented Microscaling Data Format.* arXiv:2601.19213 — abstract only

NVFP4:
22. NVIDIA. *Pretraining Large Language Models with NVFP4.* arXiv:2509.25149 — https://arxiv.org/html/2509.25149v1
23. NVIDIA. *Nemotron 3 Ultra.* arXiv:2606.15007 — https://arxiv.org/abs/2606.15007
24. Cim, Topcu, Kandemir (Penn State). *Diagnosing FP4 Inference: A Layer-Wise and Block-Wise Sensitivity Analysis of NVFP4 and MXFP4.* arXiv:2603.08747 — https://arxiv.org/html/2603.08747
25. Lu, Chen, Fang, Ma, Wang (NUS). *Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs.* arXiv:2605.20315 — https://arxiv.org/html/2605.20315v1
26. Meng, Luo, Zhao, Liu, Zheng, Ma, Zhang. *SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference.* arXiv:2606.26587 — https://arxiv.org/abs/2606.26587
27. Lin, Wan (PKU). *ScaleSweep: Accurate NVFP4 Post-Training Quantization of LLMs via Block Scale Initialization.* arXiv:2606.07618 — https://arxiv.org/abs/2606.07618
28. Zhuge, Yu, Wang, Li, Cao, Liu, Zhang. *QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE.* arXiv:2607.15810 — https://arxiv.org/abs/2607.15810
29. Ding, Ma, Tong, Xing, Wang, Li (CAS). *Full-Stack FP4: Stable LLM Pretraining with Quantized Projections, Optimizers, and Attention.* arXiv:2607.04422 — https://arxiv.org/abs/2607.04422
30. Sen, Kamineni, Lobo, Bhunia, Ewetz, Chatterjee. *Characterizing the Impact of NVFP4 Quantization for Low-Power Edge AI Deployment.* arXiv:2606.06527 — https://arxiv.org/abs/2606.06527
31. Peng, Lin, Lee. *Inference Economics of Enterprise Coding Agents: A Case Study of Cloud vs. On-Premise LLMs.* arXiv:2607.13080 — https://arxiv.org/html/2607.13080v1

KV cache:
32. Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu (Rice / CMU). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. arXiv:2402.02750 — https://arxiv.org/abs/2402.02750
33. Hooper, Kim, Mohammadzadeh, Mahoney, Shao, Keutzer, Gholami (UC Berkeley). *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS 2024. arXiv:2401.18079 — https://arxiv.org/html/2401.18079v3
34. Ai, He, Guo (UIUC). *Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression.* arXiv:2608.01631 — https://arxiv.org/abs/2608.01631

Evaluation methodology:
35. Liu, Sun, Zhang, Bai, Yu, Yu, Yuan, Hou (Tsinghua / Huawei). *Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models.* COLM 2025. arXiv:2504.04823 — https://arxiv.org/abs/2504.04823
36. Lian, Krichene, Huang, Tanaka, Ruwase, Zhang, Zhang. *Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models.* arXiv:2606.25519 — https://arxiv.org/abs/2606.25519
37. Chang, Zhang, Thomason, Jia (USC). *Why Do Some Inputs Break Low-Bit LLM Quantization?* EMNLP 2025 — https://aclanthology.org/2025.emnlp-main.168/

MoE:
38. Fu, Zhao, Ding, Yu, Li, Tang, Wang (Huawei). *EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization.* arXiv:2506.13329 — https://arxiv.org/abs/2506.13329

Hardware:
39. Jarmusch, Chandrasekaran (Univ. of Delaware). *Microbenchmarking NVIDIA's Blackwell Architecture: An In-depth Architectural Analysis.* arXiv:2512.02189 — https://arxiv.org/abs/2512.02189

**Non-paper primary sources (fetched and read):**

- vLLM project. *The State of FP8 KV-Cache and Attention Quantization in vLLM*, 2026-04-22 — https://vllm-project.github.io/2026/04/22/fp8-kvcache.html
- vLLM docs. *Quantized KV Cache* — https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vLLM docs (v0.4.1). *FP8 E4M3 KV Cache* — https://docs.vllm.ai/en/v0.4.1/quantization/fp8_e4m3_kvcache.html
- llm-compressor docs. *W4A4 FP4 (NVFP4) example* — https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/
- SGLang docs. *Quantization* — https://docs.sglang.io/advanced_features/quantization.html
- NVIDIA. *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference* — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- NVIDIA. *NVFP4 Trains with Precision of 16-bit and Speed and Efficiency of 4-bit* — https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/
- NVIDIA. *NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance* — https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/
- Hugging Face model card. `nvidia/GLM-5.2-NVFP4` — https://huggingface.co/nvidia/GLM-5.2-NVFP4
- Hugging Face model card. `nvidia/DeepSeek-R1-0528-FP4` — https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4

**Papers identified but read at abstract level only** (listed for completeness, not cited for
numbers above): RotateKV (2501.16383), DuQuant++ (2604.17789), MoEQuant (2505.03804),
QuantMoE-Bench (2406.08155), MoPEQ (2509.02512), Routing-Consistent MoE Quantization (2606.05688),
Dynamic Expert Quantization (2511.15015), Pretraining LLMs with MXFP4 on Native FP4 Hardware
(2605.09825), Practical FP4 Training for Large-Scale MoE on Hopper (2603.02731),
TriAxialKV (2605.17170), SPECTRA (2608.07915), HyperQuant (2606.23406),
AdaMX / Heterogeneity-Aware Microscaling (2608.03867), FOCUS (2608.01847), MixFP4 (2605.31035),
Grid Games (2605.12327), ScaleSearch (2605.12464), ReSET (2606.13233).
