# GPU Research — B200 inference at the hardware limit

Research corpus for pushing LLM inference on **8× NVIDIA B200 (SM100, NVLink5 /
NVSwitch, 183 GB HBM3e per GPU)** to the theoretical limit of the hardware —
along both planes that matter and that trade against each other:

- **latency / speed** — tokens/s for one stream, TTFT, TPOT
- **cost per user** — tokens/s/GPU at concurrency, $/1M tokens

## Why this repo exists

The engine work lives elsewhere (`NotSglang`, `benchmark`). This repo is the
*knowledge base* behind it: what the hardware can actually do, what the
literature has established, what the fastest inference companies have published,
and what is worth trying next — each claim carrying its source.

## Grounding: the system being optimized

| | |
|---|---|
| hardware | 8× B200 SXM, 183 GB HBM3e each, NV18 all-to-all (NVSwitch), 2 NUMA nodes |
| driver / CUDA | 595.71.05 / CUDA 13.2 runtime, 13.3 toolkit |
| primary model | GLM-5.2 (FP8 and NVFP4 builds), TP8 |
| also targeted | Kimi K3, Qwen3.8, DeepSeek V4 |
| engine | NotSglang (SGLang derivative) + hand-written `glm-kernels` / `k3-kernels` |
| measured today | 365 tok/s single-stream (real sharegpt data), 40.8k tok/s aggregate @ C64 |

Measured hotspot distribution at concurrency 1 (nsys, 11.3M kernels, all 8 ranks):
dense GEMM **37.1%**, collectives **19.6%**, MoE expert GEMMs **19.4%**,
attention **10.9%**, DSA indexer **5.8%**. Of collective time, **47% is
rank-arrival skew rather than data movement**.

Every document here is written to serve that profile.

## Map

| directory | what is in it |
|---|---|
| [`00-hardware/`](00-hardware/) | B200 / SM100 architecture from first principles — SMs, tensor cores, TMEM, memory hierarchy, TMA, clusters, NVLink5, power |
| [`01-kernel-optimization/`](01-kernel-optimization/) | how to actually write kernels that hit peak: scheduling, pipelining, cache control, quantization formats, GEMM/attention/MoE recipes |
| [`02-inference-systems/`](02-inference-systems/) | serving-layer techniques: batching, paged/prefix KV, speculative decoding, PD disaggregation, parallelism strategy, overlap |
| [`03-papers/`](03-papers/) | the academic literature, distilled with citations |
| [`04-industry/`](04-industry/) | what inference companies have published — Together, Modal, Baseten, Fireworks, DeepSeek, Moonshot, Qwen, and others |
| [`05-models/`](05-models/) | per-model serving analysis: GLM-5.2, Kimi K3, Qwen3.8, DeepSeek V4 |
| [`06-actionable/`](06-actionable/) | the synthesis — ranked, costed, tied to our measured hotspots |

## Rules for this corpus

1. **Every non-obvious claim carries a source.** A URL, a paper, a CUDA doc
   section, or a measurement in our own repos.
2. **Confidence is labelled.** `[verified]` (primary source, or measured here),
   `[reported]` (a vendor or company claims it), `[inferred]` (reasoning from
   architecture, not stated anywhere), `[unverified]` (plausible, unsourced).
3. **Numbers get units and conditions.** "1.5× faster" without batch size,
   sequence length and dtype is noise.
4. **Negative results are kept.** A measured failure costs the same to obtain as
   a win and is worth as much.
