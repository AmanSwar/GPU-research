# Low precision on Blackwell: NVFP4 vs MXFP4 vs FP8, and quantizing the KV cache

**What this is.** Everything the SM100 tensor cores will accept below 16 bits, the
engineering around each format, and what our GLM-5.2 deployment is actually running.
Every claim is labelled `[verified]` (read in a primary source, path/URL given),
`[reported]` (a vendor asserts it), `[inferred]` (my reasoning from architecture), or
`[unverified]`. Several widely-repeated claims turned out to be unsourceable or wrong
and are called out as such. The checkpoint numbers in §6 were read off the safetensors
headers in `/home/aman/code/weights/GLM-5.2-NVFP4`, not from a blog post.

---

## Bottom line for our system

- **The 37.1% dense-GEMM slice is BF16, not FP8 and not FP4.** `[verified]` — every
  `self_attn.*` linear, every `mlp.shared_experts.*`, the router `mlp.gate`, layers 0–2
  and the indexer are in the `exclude_modules` list of
  `weights/GLM-5.2-NVFP4/hf_quant_config.json`, and the shards confirm BF16 dtype.
  BF16 is 57.11 GB of a 464.80 GB checkpoint (12.3% of bytes); ~34 GB of that is read
  **in full on every token**, while only 8/256 = 3.125% of the NVFP4 expert bytes are
  touched. Per MoE layer that is **408.7 MB BF16 vs 169.9 MB FP4 read per token — a
  2.41× byte ratio**, which lines up with the measured 37.1% : 19.4% time split. This is
  the largest untouched lever in the profile.
- **The MTP draft layer's 256 experts are BF16, not NVFP4** — 19.33 GB, 4.2% of the
  checkpoint `[verified]`. It reads **604 MB of expert weights per draft token vs
  169.9 MB for a main-model layer (3.55×)**, and at MTP 3-1-4 it runs three times per
  verify step. Quantizing it is a one-layer change with a measurable target
  (draft-step latency) and one specific risk (acceptance length).
- **Our KV cache is worse than "uncalibrated per-tensor FP8".** `[verified]` — the
  checkpoint declares `kv_cache_quant_algo: "FP8"` but ships **zero `k_scale`/`v_scale`
  tensors**, so `BaseKVCacheMethod.process_weights_after_loading` takes the
  `k_scale < 0 and v_scale < 0` branch and sets both to 1.0. And on the `trtllm` DSA
  path our box selects, `MLATokenToKVPool._write_mla_kv_buffer` falls to a bare
  `.to(torch.float8_e4m3fn)` — **the 64 RoPE dims are cast to FP8 too, and they are the
  only part of the latent that never passes through an RMSNorm**. That is the specific
  thing to measure.
- The `flashmla` DSA path would give us dynamic per-128 block FP32 scales *and* BF16
  RoPE for +13.9% KV bytes (656 vs 576 B/token/layer) `[verified]` — a cheap A/B if the
  measurement in §7.7 shows damage.
- **NVFP4 and MXFP4 run at the same tensor-core rate on B200** (both `4x Hopper FP8`
  per NVIDIA's own CUTLASS table) `[verified]`. The throughput cliff that actually
  exists is `kind::mxf8f6f4` (2×) vs `kind::mxf4`/`mxf4nvf4` (4×) — run FP4 through the
  mixed-precision kind and you lose half the FLOPs *and* pay 8-bit containers for 4-bit
  data.
- The accuracy gap between NVFP4 and MXFP4 is **mostly the E4M3 scale, not the block
  size**: on an independent WikiText2 sweep, E8M0→E4M3 at fixed g32 buys 15.17→12.04
  PPL; g32→g16 at fixed E8M0 buys only 15.17→14.52 `[verified]`.
- **Hadamard/rotation (QuaRot/SpinQuant) is probably not worth it for us.** The one
  independent controlled study finds rotation gives "little improvement and may even
  lead to degradation" for MXFP4/NVFP4 PTQ, because a 16-element block already does
  what rotation does `[verified]`. It matters for FP4 *training* (Wgrad), not FP4 PTQ.
- Quantize kernels are 2.4% / 76,137 launches ≈ **3.0 µs each** — small, but at C1 with
  M≈4–8 tokens they are pure overhead on a memory-bound GEMM. Any dense-path
  quantization we add should be **weight-only (W4A16 / W8A16)** first, not W4A4.

---

## 1. The format zoo on SM100

### 1.1 Element formats the tensor cores accept

All from the PTX ISA "Alternate Floating-Point Data Formats" section
(`docs.nvidia.com/cuda/parallel-thread-execution`, §5.2.3) and the CUDA 13.3 header
`cuda_fp8.hpp` / `cuda_fp4.hpp`. `[verified]`

| format | bits | S/E/M | max finite | min normal | min subnormal | Inf | NaN |
|---|---:|---|---:|---:|---:|---|---|
| `e4m3` | 8 | 1/4/3 | **448** (`0x7E` = 0x1.Cp+8) | 2⁻⁶ | 2⁻⁹ | no | `0x7F` only |
| `e5m2` | 8 | 1/5/2 | **57344** (`0x7B` = 0x1.Ep+15) | 2⁻¹⁴ | 2⁻¹⁶ | yes | yes |
| `e3m2` | 6 | 1/3/2 | 28 | 2⁻² | — | no | no |
| `e2m3` | 6 | 1/2/3 | 7.5 | 2⁻⁰ | — | no | no |
| `e2m1` | 4 | 1/2/1 | **6.0** (`FP4_MAXNORM = 0x7`) | 1.0 | 0.5 | no | no |
| `ue8m0` | 8 | 0/8/0 | 2¹²⁷ | 2⁻¹²⁷ | — | no | `0xFF` only |
| `ue4m3` | **7** | 0/4/3 | 448 | 2⁻⁶ | 2⁻⁹ | no | `0x7F` only |
| `s2f6` | 8 | fixed-point xx.xxxxxx | +1.984375 / −2.0 | — | — | no | no |

E4M3 constants are read directly from
`/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda_fp8.hpp:138-157`
(`FP8_EXP_BIAS = 7`, `FP8_MAXNORM = 0x7E`, `FP8_MINNORM = 2^-6`, `mindenorm/2 = 2^-10`).
`FP4_MAXNORM` and `maxnorm = 6.0` are at `cuda_fp4.hpp:117-133`. `[verified]`

Two things worth internalising:

1. **`ue4m3` is a 7-bit format**, not 8: the PTX ISA says "A register variable
   containing single ue4m3 value must be declared with `.b8` type having MSB bit padded
   with zero", NaN limited to `0x7F`. `[verified]` It still costs a byte of storage and
   bandwidth, so NVFP4's scale overhead is a full byte per 16 elements.
2. **`e2m1` has exactly 8 non-negative values**: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. This is
   confirmed independently by FlashInfer's decode LUT
   (`_E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, …]` in
   `flashinfer/quantization/fp4_quantization.py`). `[verified]` Worst-case
   round-to-nearest relative error inside the normal range is 20% (midpoints 1.25 and
   5.0); in the subnormal region (below 1.0 × scale) it is worse. FP4 is a *range*
   format, and everything depends on the scale putting your data in [1, 6].
- `s2f6` is new in this PTX ISA revision and I could not source what it is for in an
  LLM context. Not sourced.

### 1.2 The block-scaled wrappers

CUTLASS's own table (`media/docs/cpp/blackwell_functionality.md`, "Blackwell Block
Scaled Narrow Precision Data Types"): `[verified]`

| CUTLASS type | element | scale type | SF vector size (dense) | SF vec (sparse) | OCP compliant |
|---|---|---|---:|---:|---|
| `mx_float8_t<any f8>` | e4m3/e5m2 | `float_ue8m0_t` | 32 | 64 | Yes |
| `mx_float6_t<any f6>` | e3m2/e2m3 | `float_ue8m0_t` | 32 | 64 | Yes |
| `mx_float4_t` | e2m1 | `float_ue8m0_t` | 32 | 64 | Yes |
| `nv_float4_t` | e2m1 | `float_ue4m3_t` | **16** | 32 | **No** |

NVFP4 adds a **second-level per-tensor FP32 scale** that is *not* part of the MMA
instruction — the tensor core only sees the per-block `ue4m3`. The FP32 global scale
lives in the checkpoint (`weight_scale_2`, `input_scale`) and is folded into the
epilogue or into the block scales at quantization time. `[verified]` — confirmed both
by the NVFP4 pretraining paper's Appendix B (below) and by the tensor names in our
own checkpoint.

**Bits per element, honestly:**

| format | payload | scale amortized | effective bits |
|---|---:|---:|---:|
| MXFP8 | 8 | 8/32 = 0.25 | 8.25 |
| MXFP6 | 6 | 0.25 | 6.25 |
| MXFP4 | 4 | 0.25 | **4.25** |
| NVFP4 | 4 | 8/16 = 0.50 | **4.5** |

ModelOpt encodes exactly this: `modelopt_recipes/configs/numerics/nvfp4.yaml` carries
`effective_bits: 4.5`. `[verified]` Our checkpoint confirms it empirically: 362.39 GB of
packed FP4 against 45.30 GB of E4M3 scales = **12.500%** overhead, i.e. 0.5 bits per
4-bit element (§6). NVFP4 costs ~5.9% more storage than MXFP4 for the same weights.

### 1.3 The three block-scaled MMA kinds, and their real throughput

From CUTLASS `blackwell_functionality.md` (SM100 section) `[verified]`:

| PTX instruction | throughput | layouts |
|---|---|---|
| `tcgen05.mma(.sp).cta_group::[1\|2].kind::mxf8f6f4.block_scale` | **2× Hopper FP8** | TN, NT, TT, NN |
| `tcgen05.mma(.sp).cta_group::[1\|2].kind::mxf4.block_scale` | **4× Hopper FP8** | **TN only** |
| `tcgen05.mma(.sp).cta_group::[1\|2].kind::mxf4nvf4.block_scale.scale_vec_size::[2X\|4X]` | **4× Hopper FP8** | **TN only** |

**Finding: there is no documented hardware throughput difference between `mxf4` and
`mxf4nvf4` on B200.** NVIDIA lists both at 4× Hopper FP8 on SM100, and both at
"2× Ada FP8 (4× for FP32 accumulator)" on SM120. The NVFP4 pretraining paper's Table 1
independently lists MXFP4 and NVFP4 at the *same* 4× BF16 speedup on GB200 and 6× on
GB300. `[verified]` If you have seen the claim "NVFP4 runs at half rate because of the
16-element blocks", I could not source it from NVIDIA and the two NVIDIA sources I read
contradict it. Treat it as false until someone microbenchmarks it.

What *is* different between the two kinds:

| | `kind::mxf4` | `kind::mxf4nvf4` |
|---|---|---|
| scale type | `ue8m0` only | `ue8m0` **or** `ue4m3` |
| `scale_vectorsize` | `.scale_vec::2X` / `.block32` | `.2X`/`.block32` **and** `.4X`/`.block16` |
| scale_A shape | M × 2 | M × 2 or **M × 4** |
| TMEM columns for SFA (128×256 tile) | 8 | **16** |
| TMEM columns for SFB | up to 16 | up to **32** (48 total max) |
| scale bytes moved per K-tile | 1× | **2×** |

TMEM column counts are from the Colfax tutorial's tables; the shape table is PTX ISA
Table 59. `[verified]` So NVFP4 costs you **twice the scale-factor bandwidth and twice
the TMEM columns** — that is a real occupancy/SMEM-budget cost inside a kernel, not a
FLOP cost.

PTX ISA Table 60, verbatim structure `[verified]`:

```
.kind::mxf8f6f4   E4M3,E5M2,E2M3,E3M2,E2M1  x UE8M0  -> .scale_vec::1X / .block32
.kind::mxf4       E2M1                      x UE8M0  -> .scale_vec::2X / .block32
.kind::mxf4nvf4   E2M1                      x UE8M0  -> .scale_vec::2X/.block32, .4X/.block16
.kind::mxf4nvf4   E2M1                      x UE4M3  -> .scale_vec::4X / .block16
```

`.block16` is an alias for `.scale_vec::4X`; `.block32` aliases `1X` or `2X` depending
on kind and K. There is also a K=96 case that is "semantically equivalent to
`scale_vec::3X`" (block32) or `6X` (block16). `[verified]`

### 1.4 The padding trap in `kind::mxf8f6f4`

PTX ISA §9.7.17.10.4 `[verified]`:

- `kind::mxf8f6f4`, shared memory: "The 4-bit and 6-bit floating point elements in
  shared memory must be contiguously packed **along with padding**." Tensor memory: the
  4-bit and 6-bit elements "must be packed in an **8-bit container**".
- `kind::mxf4` / `kind::mxf4nvf4`, shared memory: "pack two 4-bit elements in an 8-bit
  container, **with no padding**."

CUTLASS's alignment table makes the consequence concrete: `float4_t` through
`kind::f8f6f4`/`mxf8f6f4` requires **128-element alignment**, while pure `mxf4`/`nvf4`
requires 32 and 16 respectively. `[verified]`

So routing FP4 through `mxf8f6f4` — which is what you get if you need NT/NN/TT layouts,
or mix FP4×FP8 operands — costs you **2× the FLOPs and 2× the SMEM/TMEM footprint for
the operands**. That is the only "FP4 is slow" story that is actually real on this chip.
`[inferred, from the two verified facts above]`

### 1.5 No in-MMA transpose for FP4

PTX ISA Table 54 `[verified]`:

| MMA-kind | transpose A/B | negate A/B |
|---|---|---|
| `.kind::tf32` / `.f16` / `.f8f6f4` / `.mxf8f6f4` | Yes | Yes |
| `.kind::i8` | Yes | No |
| **`.kind::mxf4`** | **No** | Yes |
| **`.kind::mxf4nvf4`** | **No** | Yes |

Combined with "TN layouts" in the CUTLASS table, this means an FP4 GEMM at full rate
requires A row-major (K-major) and B column-major (K-major) *in memory*. Any kernel that
wants a different majorness must physically transpose before the MMA. For MoE weights
this is a load-time concern (do it once); for activations it constrains the fusion you
can do upstream. `[inferred from verified constraints]`

### 1.6 Conversion instructions

From the PTX ISA `cvt` section `[verified]`:

```
cvt.rn{.relu}{.satfinite}.e2m1x2.f32          d, a, b;         // 2×f32 -> packed e2m1
cvt.rn{.relu}{.satfinite}.e2m1x4.f32          d, {a,b,e,f};    // 4×f32 -> packed e2m1
cvt.rs{.relu}.satfinite.f4x4type.f32          d, {a,b,e,f}, rbits;  // STOCHASTIC ROUNDING
cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.f4x2type d, a{, scale-factor};
cvt.frnd3{.satfinite}.ue8m0x2.f32             d, a, b;         // compute MX scales
cvt.rn.bf16x2.ue8m0x2                         d, a;
```

Two consequences that matter for kernel writing:

- **Hardware stochastic rounding exists for FP4/FP6/FP8 down-conversion** (`cvt.rs`,
  taking a `.b32 rbits` operand of random bits). For `.e2m1x4`, "lower 8-bits from both
  16-bit halves of `rbits` are used for operands e, f and upper 8-bits … for a, b."
  `[verified]` This is the primitive the NVFP4 training recipe leans on. It is
  irrelevant to inference (SR on forward-pass tensors is *harmful* — see §4.6).
- **The fused scaled convert only exists for `ue8m0`.** Every `.scaled::n2::ue8m0`
  variant in the ISA takes a UE8M0 scale; there is no `.scaled::…::ue4m3` form.
  `[verified — grepped the whole PTX ISA for `scaled::`, 12 hits, all ue8m0]`
  So: **MXFP4/MXFP6/MXFP8 get a hardware dequant-with-scale in one instruction; NVFP4
  does not.** Inside `tcgen05.mma` this is irrelevant (the tensor core applies both
  kinds of scale). Outside it — a Marlin-style W4A16 kernel, a small-M dequant GEMM, a
  KV-cache dequant — MXFP4 is genuinely cheaper to unpack. `[inferred]`
  Our own optimization log already noticed the `cvt.rn.bf16x2.e2m1x2` converter needs
  CUDA ≥ 13.3 (`personal_docs/glm-5.2/glm-5.2-optimization-log.md:152`).

### 1.7 A published number that looks wrong

`arXiv:2512.02189` (Jarmusch & Chandrasekaran, *Microbenchmarking NVIDIA's Blackwell
Architecture*) reports measured B200 tensor-core throughput `[verified, read the PDF]`:

| precision | B200 TFLOPS | % of their peak | H200 | speedup |
|---|---:|---:|---:|---:|
| FP32 | 482.0 | 96.4% | 378.4 | 1.27× |
| BF16 | 1926.4 | 96.3% | 1513.5 | 1.27× |
| FP8 | 3850.6 | 96.3% | 3026.9 | 1.27× |
| **FP6** | **5134.4** | 96.0% | — | new |
| FP4 | 7700.2 | 96.2% | — | new |
| INT8 | 3928.5 TOPS | 98.2% | 3088.4 | 1.27× |

Single-instruction latency 11.0–12.6 cycles for `tcgen05.mma` across all precisions and
tile sizes 64×64 → 256×256, vs 32/64/128 cycles for Hopper `wgmma` at m64n64/128/256k16.
That latency result is credible and useful.

**But the FP6 number (1.33× FP8) contradicts NVIDIA.** FP6 on SM100 runs through
`kind::mxf8f6f4`, which CUTLASS lists at the *same* 2× Hopper-FP8 rate as FP8, and the
NVFP4 paper's Table 1 lists MXFP6 and MXFP8 both at 2× BF16 on GB200/GB300.
`[verified, two NVIDIA sources]` The paper also cites a `kind::mxf6` PTX opcode which
**does not exist** in the PTX ISA (the kinds are `f16, tf32, f8f6f4, i8, mxf8f6f4,
mxf4, mxf4nvf4`). `[verified — grepped the ISA]` Conclusion: **treat 5134 TFLOPS FP6 as
unreliable; plan on FP6 == FP8 rate with 0.75× the bytes.** Their FP4 number
(7700 ≈ 2× FP8) is consistent with everything else.

---

## 2. Why NVFP4 beats MXFP4 at the same nominal width

### 2.1 The mechanism, precisely

Quantizing a block means choosing a scale `s` so that `amax_block / s` lands at or just
below the format's max (6.0 for e2m1).

- **MXFP4 (UE8M0)** can only represent `s = 2^k`. You must round `k` up to avoid
  saturation, so the block's largest value lands anywhere in `[3, 6)` after scaling —
  in the worst case you *lose the top binade*, and with it the codes ±4 and ±6. The
  NVFP4 paper states this directly: MXFP4 "can potentially lose up to one binade of
  dynamic range (and four samples: ±4 and ±6) because of power-of-two scale factor
  rounding". `[verified — arXiv:2509.25149 §2]` A 4-bit format that loses 4 of its 16
  codes is effectively a 3.5-bit format on that block.
- **NVFP4 (UE4M3)** has 3 mantissa bits in the scale, so `s` can be chosen to within
  ~6% of ideal and the block amax lands near 6.0 essentially always. The cost is that
  UE4M3 only spans 2⁻⁹…448 instead of 2⁻¹²⁷…2¹²⁷, which is why the second-level FP32
  per-tensor scale exists: it re-centres the whole tensor so the per-block scales fit
  in E4M3 range.

The exact two-level recipe, from the paper's Appendix B `[verified]`:

```
s_enc = 448 * 6 / amax_x            # global FP32 encode scale, amax over whole tensor
s_dec,b            = amax_block / 6
s_dec,b,e4m3       = e4m3( s_dec,b * s_enc )        # RNE
s_enc,b            = 1 / ( fp32(s_dec,b,e4m3) * s_dec )
x̂_i                = q( x_i * s_enc,b )              # RNE to e2m1
```
with `6` and `448` the max magnitudes of E2M1 and E4M3. The paper is explicit that
`s_enc,b · s_dec · s_dec,b,e4m3 ≈ 1` must hold, "since failing to do so can impact model
accuracy" — the inverse-of-a-quantized-scale round-trip is the subtle part most
reimplementations get wrong. `[verified]`

The paper's own framing of the benefit: NVFP4 "encodes at least 6.25% of values in a
block (the amax values in each block of 16 elements) at near-FP8 precision". `[reported]`

### 2.2 Which of the two changes actually pays

The independent evaluation `arXiv:2507.17417` runs the 2×2 ablation on Llama-3.2-1B,
WikiText2 PPL, W4A4, BF16 baseline **9.76**: `[verified — read the HTML, Table XIII]`

| method | INT4 ch | FP4 ch | INT4 g16 | FP4 g16 | **MXFP4 g32/E8M0** | FP4 g32/E4M3 | FP4 g16/E8M0 | **NVFP4 g16/E4M3** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTN | 256.97 | 135.68 | 11.75 | 11.34 | **15.17** | 12.04 | 14.52 | **11.54** |
| GPTQ | 161.04 | 101.36 | 11.47 | 10.87 | 13.24 | 11.36 | 12.77 | **10.98** |
| Low-rank | 179.87 | 132.14 | 11.69 | 11.32 | 15.56 | 11.98 | 14.67 | 11.48 |
| GPTQ+Low-rank | 139.31 | 96.33 | 10.92 | 10.76 | 12.69 | 11.18 | 12.25 | **10.83** |

Decomposition of the MXFP4 → NVFP4 gap (RTN row, 15.17 → 11.54, Δ 3.63 PPL):

| change | PPL delta | share of gap |
|---|---:|---:|
| block 32 → 16, scale fixed at E8M0 | 15.17 → 14.52 = **0.65** | 18% |
| scale E8M0 → E4M3, block fixed at 32 | 15.17 → 12.04 = **3.13** | 86% |

**The scale format carries roughly 5× more of the benefit than the block size.**
`[verified, from the table]` That is worth knowing because it says a hypothetical
"MXFP4 with a 16-element block" would buy almost nothing, whereas "FP4 with an E4M3
scale at block 32" (which `kind::mxf4nvf4` cannot express — UE4M3 requires `4X`) would
capture most of NVFP4's advantage at MXFP4's storage cost.

Same paper, Table XV, on how far you can push the scale down (g16, two-level): `[verified]`

| scale format (W/A) | E4M3/E4M3 (=NVFP4) | E2M1/E4M3 | E2M1/E2M1 | INT4/E4M3 | INT4/INT4 |
|---|---:|---:|---:|---:|---:|
| RTN | 11.54 | 14.10 | **4130** | 14.66 | **4777** |
| GPTQ | 10.98 | 13.38 | 3618 | 17.96 | 3803 |

4-bit scales on weights only are survivable; 4-bit scales on activations destroy the
model. `[verified]`

### 2.3 Training-side evidence (different regime, same direction)

`arXiv:2509.25149`, 8B hybrid Mamba-Transformer, 1T tokens, BF16 reference: MXFP4
reaches ~2.5% relative loss error vs NVFP4's ~1.5%, and **MXFP4 needs 36% more tokens
(1.36T vs 1.0T) to match NVFP4's final loss.** `[verified]` This is a pretraining
result and does not transfer to PTQ, but it is the largest controlled comparison
published.

---

## 3. The scale-factor layout — the expensive bug

This is the thing that silently produces garbage or a 3× slowdown. The tensor core does
not read scale factors in the obvious row-major order.

### 3.1 The 512-byte atom

CUTLASS, verbatim: "The scale factor layout consists of a **512B basic-block**
structure … Each block contains **128 M/N dimension and 4 scale factors (SF) along the K
dimension**. The byte order of the basic storage chunk is row-major, meaning that
**M0SF0 to M0SF3, M32SF0 to M32SF3, M64SF0 to M64SF3, and M96SF0 to M96SF3 are stored
consecutively** in GMEM." Multiple atoms are arranged **K-major**. `[verified]`

CUTLASS's constructor:

```cpp
using SfConfig = Sm1xxBlockScaledConfig<SFVecSize>;
// SFA shape: ((32,4), ceil(M/128)), ((SFVecSize,4), ceil(K/4), L)
auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(problem_shape);
auto layout_sfb = SfConfig::tile_atom_to_shape_SFB(problem_shape);
```

Our own engine implements exactly the same permutation in
`NotSglang/python/sglang/srt/layers/quantization/utils.py:597`: `[verified]`

```python
def swizzle_blockscale(scale: torch.Tensor):          # dtype must be float8_e4m3fn
    M_padded = round_up(M, 128); K_padded = round_up(K, 4)
    padded_scale = zeros(B, M_padded, K_padded)       # zero-pad, do not replicate
    padded_scale = padded_scale.reshape(B, rows//128, 4, 32, cols//4, 4)
    swizzled     = padded_scale.permute((0, 1, 4, 3, 2, 5)).contiguous()
```

Decoding the permutation: input index `[b, m128, a, b32, k4, s]` with
`row = m128*128 + a*32 + b32`, `col = k4*4 + s`; after the permute the contiguous order
is `s` fastest (4 K-scales), then `a` (which of the +0/+32/+64/+96 row offsets), then
`b32` (row within 32). That reproduces CUTLASS's "M0SF0..M0SF3, M32SF0..M32SF3,
M64.., M96.." byte order **exactly**, from two independent implementations.
`[verified — cross-checked]`

Colfax gives the equivalent as an interleave function and as CuTe layouts, e.g. for
`nvf4` `block16`, 128×256 tile:
`SFA = ((((32,4),1),(16,4)),1,4,3):((((16,4),0),(0,1)),0,512,2048)`. `[verified]`

### 3.2 Padding rules

FlashInfer's helper is the cleanest statement of the allocation rule `[verified,
`flashinfer/quantization/fp4_quantization.py:72`]:

```python
def _compute_swizzled_layout_sf_size(total_row, total_column, row_size=128):
    padded_row = round_up(total_row, row_size)     # 128, or 8 for the trtllm-gen layout
    padded_column = round_up(total_column, 4)      # 4 SFs along K
    return padded_row * padded_column
```

So the SF buffer for an `M × K` NVFP4 tensor is
`round_up(M,128) × round_up(K/16, 4)` bytes. Note the two row granularities: **128×4**
(CUTLASS/cuBLAS path) and **8×4** (TRT-LLM-gen path). Our engine exposes this as a
boolean: `fp4_quantize(..., is_sf_swizzled_layout=True, is_sf_8x4_layout=False)` in
`NotSglang/python/sglang/srt/layers/quantization/fp4_utils.py`. `[verified]` Getting
this flag wrong against a trtllm-gen MoE kernel produces wrong numbers, not an error.

FlashInfer also ships `shuffle_matrix_sf_a(input_tensor, epilogue_tile_m,
num_elts_per_sf=16)`, which does a **row permutation keyed on the epilogue tile M**
followed by `block_scale_interleave` into 128×4. Its docstring notes it "expects
`input_tensor` in the *linear* layout that is used by quantized NVFP4 checkpoints" —
i.e. checkpoints on disk are linear, and the swizzle is a load-time transform.
`[verified]`

### 3.3 TMEM side and `tcgen05.cp`

PTX ISA §9.7.17.10.7 `[verified]`:

- "Scale factors for A and B matrices need to be **duplicated to all 32 lane partitions
  of tensor memory**."
- `scale_vec::1X`: SF in a 1-byte-aligned sub-column, `SFA_ID` = byte offset (0–3).
- `scale_vec::2X`: 2-byte-aligned sub-column, `SFA_ID` ∈ {0, 2}.
- `scale_vec::4X` (NVFP4): 4-byte-aligned, **`SFA_ID` must be 0** — all columns used.

The SMEM→TMEM move is `tcgen05.cp` with `.multicast::warpx4` (Colfax names the op
`Cp4x32x128bOp`) to broadcast one scale block to all four 32-lane quadrants.
`[verified — Colfax; PTX ISA confirms the duplication requirement]`

The instruction form:

```
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
    [d-tmem], a-desc, b-desc, idesc,
    [scale-A-tmem], [scale-B-tmem], enable-input-d;
```

### 3.4 The same problem shows up outside GEMM

Our repo already hit it for an MXFP8 **KV cache**:
`NotSglang/python/sglang/kernels/ops/quantization/mxfp8_interleave_sf.py` `[verified]`:

> "When `page_size=128` and `sf_vec_size=32`, FA4 expects scale factors in the
> `BlockScaledBasicChunk` atom layout: `[num_pages, nheads, 32, 4, 4]`. The interleave
> mapping for a token at page offset `t` (0–127), head `h`, scale index `s` (0–3) is
> `output[page, h, t % 32, t // 32, s]`."

Same 32×4×4 atom, applied along the token axis instead of the M axis. If we ever put
the MLA latent in a block-scaled format, this is the shape the write path must produce.

### 3.5 Symptoms of getting it wrong

`[inferred, but each symptom maps to a specific mistake]`

| symptom | likely cause |
|---|---|
| output is plausible but ~2–10% off, worse at large K | SF laid out row-major instead of 32×4×4 atom |
| output correct for M ≤ 128, wrong beyond | missing the `ceil(M/128)` outer atom / K-major atom ordering |
| correct on cuBLAS/CUTLASS, wrong on trtllm-gen MoE | `is_sf_8x4_layout` mismatch (8×4 vs 128×4) |
| silent NaN in a few output rows | SF buffer not zero-padded to `round_up(M,128)` |
| kernel runs at ~half expected FLOPs | fell back to `kind::mxf8f6f4` (padded 8-bit containers) |
| illegal-instruction / ptxas reject | `SFA_ID != 0` with `scale_vec::4X` |

---

## 4. Quantization method: PTQ recipes for NVFP4

### 4.1 What "NVFP4" means as a recipe

`compressed-tensors` preset (`quantization/quant_scheme.py:180`) `[verified]`:

```python
NVFP4 = dict(
    weights=QuantizationArgs(num_bits=4, type=FLOAT, strategy=TENSOR_GROUP,
        symmetric=True, dynamic=False, group_size=16,
        scale_dtype=float8_e4m3fn, zp_dtype=float8_e4m3fn),
    input_activations=QuantizationArgs(num_bits=4, type=FLOAT, strategy=TENSOR_GROUP,
        symmetric=True, dynamic=DynamicType.LOCAL, group_size=16,
        observer="static_minmax", scale_dtype=float8_e4m3fn, zp_dtype=float8_e4m3fn),
)
MXFP4 = dict(  # group_size=32, scale_dtype=torch.uint8 (= ue8m0), dynamic=True for acts
```

`TENSOR_GROUP` is the two-level scheme; `dynamic=LOCAL` for activations means **the
per-16 E4M3 block scales are computed at runtime, while the per-tensor FP32 global scale
is calibrated offline**. MXFP4 activations are fully dynamic (no calibration at all,
because E8M0 needs no global re-centring). `[verified]`

ModelOpt says the same in YAML — `modelopt_recipes/configs/numerics/`: `[verified]`

```yaml
# nvfp4.yaml                      # mxfp4.yaml
num_bits: e2m1                    num_bits: e2m1
block_sizes: {-1: 16,             block_sizes: {-1: 32,
  type: dynamic, scale_bits: e4m3}  type: dynamic, scale_bits: e8m0}
effective_bits: 4.5
```

There is also `nvfp4_static.yaml` (`type: static`) for fully static activation
quantization, and `nvfp4_bs32.yaml` (E4M3 scales at block 32 — the "best of both"
config from §2.2, used by the W4A8 preset). `[verified]`

### 4.2 Which layers stay in higher precision

ModelOpt's `configs/ptq/units/default_disabled_quantizers.yaml`, verbatim list
`[verified]`:

```
*block_sparse_moe.gate*   *linear_attn.conv1d*   *linear_attn.in_proj_a/b*
*lm_head*                 *mixer.conv1d*          *mlp.gate.*
*mlp.shared_expert_gate.* *output_layer*          *proj_out.*
*router*                  mtp.*                   output.*
*embed_vision* *vision_tower* *visual* *vision_model* *multi_modal_projector*
parent_class: nn.Embedding / nn.BatchNorm{1,2,3}d / nn.LeakyReLU
```

Note what is **not** on the list: attention projections and shared experts. Those are
excluded per-model, not by default — and our GLM-5.2 checkpoint does exclude them (§6).
The vision-branch exclusions carry an NVBug rationale worth remembering:
`vision_model.patch_embedding.linear` has `in_features=588`, "not divisible by the NVFP4
block size". **Any linear whose K is not a multiple of 16 cannot be NVFP4.** `[verified]`

llm-compressor's minimal recipe is one line `[verified,
`examples/quantization_w4a4_fp4/llama3_example.py`]:

```python
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
```

### 4.3 Calibration

| source | samples | dataset | what is calibrated |
|---|---:|---|---|
| llm-compressor NVFP4 example | **20** (README: "typically a small number") | ultrachat_200k | per-tensor **global** activation scale only |
| llm-compressor FP8 KV example (in vLLM docs) | **512** ("good starting value") | ultrachat_200k, seq 2048 | k/v (and q) per-tensor or per-head scales |
| NVIDIA DeepSeek-R1-0528-FP4 card | not stated | **cnn_dailymail** | ModelOpt v0.31.0 |

`[verified]` The asymmetry is the point: **NVFP4 weights need no calibration data at all
(the block scales are a deterministic function of the weights); only the activation
global scale does, and it is one scalar per tensor, so 20 samples suffice.** KV-cache
scales are a different animal — they gate a *range*, and the vLLM guidance is 512
samples. `[inferred from the two verified recipes]`

llm-compressor README, verbatim: "`nvfp4` quantization generates per-tensor global
scales and per-group (size 16) local quantization scales for the weights, as well as
per-tensor global scales for the activations. **Per-group local activation quantization
scales are generated dynamically during inference time.**" `[verified]`

### 4.4 Does SmoothQuant / AWQ / GPTQ apply to FP4?

- **GPTQ: yes, and it helps.** In the independent table above GPTQ moves NVFP4 from
  11.54 → 10.98 PPL and MXFP4 from 15.17 → 13.24. `[verified]` GPTQ is a weight-update
  method and is format-agnostic.
- **AWQ: ModelOpt ships `NVFP4_AWQ_LITE_CFG`, `NVFP4_AWQ_CLIP_CFG`,
  `NVFP4_AWQ_FULL_CFG`.** `[verified, `modelopt/torch/quantization/config.py:1786-1792`]
  So NVIDIA thinks per-channel activation-aware scaling is still worth it on top of
  16-element blocks. I found no published ablation quantifying the gain for NVFP4
  specifically. Not sourced.
- **SmoothQuant: the migration transform is a per-input-channel rescale.** It was
  designed to fix *per-tensor* activation quantization. With NVFP4's per-16 activation
  blocks the outlier problem it solves is already mostly handled. ModelOpt's SmoothQuant
  presets are INT8-only in the docs I read. `[verified — the ModelOpt "choosing quant
  methods" guide lists SmoothQuant only under INT8]`
- ModelOpt also ships FP4-specific weight solvers: `nvfp4_w4a4_weight_mse_fp8_sweep`,
  `nvfp4_w4a4_weight_local_hessian`, `nvfp4_svdquant`. `[verified — file listing]`
  Semantics not sourced.

### 4.5 Hadamard / rotation (QuaRot, SpinQuant)

The papers `[verified — abstracts read]`:

- **QuaRot** (arXiv:2404.00456): rotate the residual stream so outliers vanish under a
  computational invariance; quantizes weights, activations **and KV cache** to 4 bits;
  ≤0.47 WikiText-2 PPL loss on LLaMa2-70B, 99% of zero-shot retained. Note the baseline
  there is INT4 per-channel, not block-scaled FP4.
- **SpinQuant** (arXiv:2405.16406): learned rather than random rotations; reports up to
  **13 points** of downstream difference between good and bad random rotations.

**But for block-scaled FP4 PTQ, the independent evaluation says rotation does not help**
`[verified — arXiv:2507.17417, §III-C1]`:

> "Rotation-based methods yield smaller improvements for MXFP4 and NVFP4 than for INT4
> (per-channel). … For MXFP4 and NVFP4 formats, the group size is extremely small, so
> data rotation contributes little **and may even lead to degradation**."

Their Table XIV bears this out: NVFP4 + GPTQ goes from **10.97** (scaling only) to
**11.04** (optimized rotation + scaling) — rotation makes it *worse*. MXFP4 improves
only 13.24 → 12.37. `[verified]`

Where rotation *does* earn its keep on FP4:

- **FP4 pretraining, Wgrad only.** NVIDIA applies **16×16 Random Hadamard transforms to
  the inputs of the weight-gradient GEMM**, with a single shared random sign vector,
  and explicitly reports "no measurable benefit for Fprop and Dgrad at smaller scales".
  d=16 chosen; d=4 worse, d=128 no better. `[verified — arXiv:2509.25149 §4.2, E.4.2]`
- **KV cache.** ModelOpt has a dedicated `kv/nvfp4_rotate.yaml` preset that rotates
  **K only** (`q_bmm_quantizer` disabled, `v_bmm_quantizer` unrotated) `[verified]`:

```yaml
- quantizer_name: '*q_bmm_quantizer'   {cfg: {rotate: true}, enable: false}
- quantizer_name: '*k_bmm_quantizer'   {cfg: {$import: nvfp4, rotate: true}}
- quantizer_name: '*v_bmm_quantizer'   {cfg: {$import: nvfp4}}
```

  That asymmetry (rotate K, not V) is consistent with the KV-distribution literature:
  K has per-channel outliers, V does not. `[inferred; KIVI arXiv:2402.02750 abstract
  motivates per-channel K / per-token V, which is the same observation]`

There is a Hadamard kernel in our tree already
(`NotSglang/python/sglang/kernels/ops/quantization/hadamard.py`) if we want to test it.
`[verified — file exists]`

### 4.6 QAT / distillation for FP4

The only large-scale public result is NVIDIA's 12B hybrid Mamba-Transformer trained on
10T tokens in NVFP4 `[verified — arXiv:2509.25149]`. Its recipe, compressed:

| component | choice | ablation verdict |
|---|---|---|
| high-precision layers | first 2 + last 8 blocks BF16 (**16%** of linears) | "training diverges when every linear layer is quantized to FP4"; last-4-only also converges |
| always high precision | embeddings, output head, norms, non-linearities, **all attention incl. softmax and the QK/PV BMMs** | not ablated |
| weights | **2D 16×16** block scaling (fwd/bwd consistency) | removing it worsens loss |
| activations & gradients | standard 1×16 NVFP4 | — |
| rounding | RNE for weights/activations, **stochastic for gradients only** | "applying stochastic rounding to the forward pass tensors is **detrimental**" |
| RHT | 16×16 on Wgrad inputs only | removing it worsens loss |
| optimizer state, master weights, weight grads | FP32 | — |
| TP reductions | BF16 | — |

Final accuracy, 12B, NVFP4 vs FP8 pretraining (evaluated in BF16) `[verified, Table 2]`:

| task group | FP8 | NVFP4 | task group | FP8 | NVFP4 |
|---|---:|---:|---|---:|---:|
| General | 68.99 | **69.82** | Code | **59.52** | 56.67 |
| MMLU | **77.36** | 76.57 | HumanEval++ | **59.93** | 57.43 |
| MMLU-Pro 5-shot | 62.62 | 62.58 | MBPP++ | **59.11** | 55.91 |
| Math | 86.20 | **86.88** | Commonsense | **77.29** | 76.75 |
| GSM8k CoT | 89.08 | **92.27** | HellaSwag | **83.83** | 83.09 |
| MATH | **83.32** | 81.48 | Winogrande | **80.58** | 78.77 |
| Multilingual | 77.93 | **80.24** | ARC-C | 91.81 | 91.81 |

Relative validation-loss error stays <1% through the stable phase and rises "slightly
above 1.5%" during LR decay; the authors suggest switching to BF16/MXFP8 for the decay
phase. **Coding is the one consistently worse group (−2.85 avg).** `[verified]`

For *our* purposes this is a training result, not a PTQ result — but the layer-selection
lesson transfers: **attention math and the last blocks are where FP4 hurts.**

---

## 5. Accuracy evidence, sorted by who is claiming it

### 5.1 Vendor-reported

NVIDIA, DeepSeek-R1-0528, FP8 vs NVFP4 (ModelOpt v0.31.0, cnn_dailymail calibration,
tested on B200) `[reported — HF card `nvidia/DeepSeek-R1-0528-FP4`]`:

| precision | MMLU Pro | GPQA Diamond | LiveCodeBench | SCICODE | MATH-500 | AIME 2024 |
|---|---:|---:|---:|---:|---:|---:|
| FP8 (AA ref) | 85 | 81 | 77 | 40 | 98 | 89 |
| **NVFP4** | 84.2 | 80.0 | 76.3 | **40.1** | **98.1** | **91.3** |

Note the FP8 row is quoted to 2 significant figures from a third party (Artificial
Analysis) while the FP4 row is to 3 — they are not a matched pair, and AIME-2024 at
n=30 questions has a ±5-point standard error. "NVFP4 beats FP8 on AIME" is noise.
`[inferred]`

NVIDIA also claims: "~88% lower quantization error than power-of-two scaling", MSE
0.72 (E8M0) → 0.08 (E4M3), 3.5× memory reduction vs FP16 and 1.8× vs FP8.
`[reported — developer.nvidia.com NVFP4 blog]` The 1.8× (not 2×) is the 4.5-effective-bit
number, consistent with §1.2.

### 5.2 Independent

- `arXiv:2507.17417` reproduces the ordering NVFP4 > FP4-g32-E4M3 > FP4-g16-E8M0 >
  MXFP4 on WikiText2, and states it "match[es] NVIDIA's official conclusions".
  `[verified]` This is the only independent controlled MXFP4-vs-NVFP4 PTQ comparison I
  found. It is a 1B model at W4A4, i.e. the hardest possible setting.
- I found **no independent replication of the DeepSeek-R1 FP4 benchmark table**.
  Not sourced.

### 5.3 Our own reference points

From `personal_docs/glm-5.2/hotspots-and-optimization-ledger.md` `[verified, local]`:
vendor GSM8K 98.2 / AIME25 87.7 for FP8 on H200; "NVIDIA reports the NVFP4 build within
~1 point of FP8 on GPQA Diamond, SciCode and IFBench". Our own GSM8K/GPQA runs on
`latency-3-1-4` are still pending — until they land we have **no measured accuracy
number for the checkpoint we are serving**.

---

## 6. What our GLM-5.2 NVFP4 checkpoint actually is

All `[verified]`, read from `/home/aman/code/weights/GLM-5.2-NVFP4/` (safetensors
headers + `hf_quant_config.json` + `model.safetensors.index.json`).

### 6.1 Header

```json
"producer": {"name": "modelopt", "version": "0.46.0.dev65+g977d34dc3"},
"quantization": {
  "quant_algo": "NVFP4",
  "kv_cache_quant_algo": "FP8",
  "group_size": 16,
  "exclude_modules": ["lm_head", "model.embed_tokens",
                      "model.layers.0*", "model.layers.1.*", "model.layers.2.*",
                      "model.layers.<N>.self_attn*",            // every layer
                      "model.layers.<N>.mlp.shared_experts*",   // every layer
                      ...]
}
```

So **the only thing quantized is `mlp.experts.{gate,up,down}_proj` in layers 3–78.**

### 6.2 Byte census (all 47 shards)

| dtype | bytes | share | tensors | what |
|---|---:|---:|---:|---|
| `U8` (packed e2m1, 2/byte) | **362.39 GB** | 78.0% | 57,600 | routed expert weights, layers 3–77 |
| `F8_E4M3` | **45.30 GB** | 9.7% | 57,600 | per-16 block scales |
| `BF16` | **57.11 GB** | 12.3% | 1,909 | see split below |
| `F32` | 0.0002 GB | — | 115,276 | `weight_scale_2` + `input_scale` (2 per quantized tensor) |
| **total** | **464.80 GB** | | | |

`57,600 = 75 layers × 256 experts × 3 projections` — exactly layers 3 through 77. The
BF16 57.11 GB splits as:

| BF16 component | bytes | read per token? |
|---|---:|---|
| layers 3–77 non-expert (attn 330.0 + shared 75.5 + router/norms 3.2 MB each, + indexer on 18) | **30.99 GB** | **yes, 100%** |
| layers 0–2 (dense MLP, fully excluded, 801.8 MB each) | 2.41 GB | yes, 100% |
| **layer 78 (MTP) routed experts — BF16, not NVFP4** | **19.33 GB** | 8/256 per draft step |
| layer 78 attn/shared/indexer/misc | 0.58 GB | yes, per draft step |
| `embed_tokens` + `lm_head` (154880×6144 each) | 3.81 GB | lm_head once/token |

Scale overhead = 45.30 / 362.39 = **12.500%** → 4.500 effective bits, exactly as spec'd.
Had this been MXFP4 the scales would be 22.65 GB and the checkpoint 442.15 GB (−4.9%).

**The MTP layer is the surprise.** `model.layers.78.mlp.experts.*.{gate,up,down}_proj`
are `BF16 [2048,6144]` / `[6144,2048]` — 256 experts × 37.75 M params × 2 B =
**19.33 GB**, consistent with ModelOpt's `mtp.*` default-disabled rule. Per draft token
it reads `19.33 GB × 8/256 = 604 MB` of expert weights against `169.9 MB` for a
main-model layer — **3.55×, purely because of the dtype** — and at MTP 3-1-4 the draft
runs three times per verify step. `[verified, arithmetic on the shard headers]`

Tensor shapes for one expert (layer 6, expert 0):

```
U8       [2048, 3072]   mlp.experts.0.gate_proj.weight          # logical K=6144, 2 nibbles/byte
F8_E4M3  [2048,  384]   mlp.experts.0.gate_proj.weight_scale    # 6144/16 = 384 blocks
F32      []             mlp.experts.0.gate_proj.weight_scale_2  # per-tensor level-2 scale
F32      []             mlp.experts.0.gate_proj.input_scale     # static global activation scale
```

Everything else in a MoE layer, in BF16:

```
BF16 [16384,  2048]  self_attn.q_b_proj        BF16 [ 6144, 16384]  self_attn.o_proj
BF16 [ 2048,  6144]  self_attn.q_a_proj        BF16 [28672,   512]  self_attn.kv_b_proj
BF16 [  576,  6144]  self_attn.kv_a_proj_with_mqa
BF16 [ 4096,  2048]  self_attn.indexer.wq_b    BF16 [  128,  6144]  self_attn.indexer.wk
BF16 [   32,  6144]  self_attn.indexer.weights_proj
BF16 [ 2048,  6144]  mlp.shared_experts.{gate,up}_proj   BF16 [6144,2048] .down_proj
BF16 [  256,  6144]  mlp.gate                  # the router
```

### 6.3 Per-layer bytes, and why the profile looks the way it does

| component | bytes/layer | read per token @ top-8/256 |
|---|---:|---:|
| routed experts (NVFP4 + scales) | 5435.8 MB | **169.9 MB** (8/256) |
| `self_attn.*` (BF16) | 330.0 MB | **330.0 MB** (100%) |
| `mlp.shared_experts.*` (BF16) | 75.5 MB | **75.5 MB** |
| router + norms | 3.2 MB | 3.2 MB |
| indexer (22 of 79 layers) | 18.7 MB | 18.7 MB on those layers |
| **BF16 subtotal** | **408.7 MB** | **408.7 MB** |

**Ratio of BF16 bytes to FP4 bytes touched per token: 2.41 : 1.** Measured GPU-time
ratio of dense GEMM to MoE GEMM at C1: 37.1 / 19.4 = **1.91 : 1**. `[verified from the
profile in the ledger + the checkpoint]` The two agree to within the difference you
would expect from MoE kernels being less bandwidth-efficient than dense ones. That is
strong evidence the C1 dense path is bandwidth-bound on BF16 weights. `[inferred]`

Indexer layers: exactly **22** (`0,1,2,6,10,…,74,78`) carry `indexer.wq_b`, confirming
`index_topk_freq = 4`. `[verified]`

### 6.4 The implied lever

If the `self_attn.*` + `shared_experts.*` path (405.5 MB/layer) moved to FP8, per-token
BF16 traffic halves; to NVFP4 (0.5625 B/param) it drops 3.55×. Applied to a 37.1% slice
that is **up to ~18.5 pp of total kernel time (FP8) or ~27 pp (NVFP4)** at C1.
`[inferred — assumes the slice is purely bandwidth-bound, which is exactly what §6.3
supports but has not been confirmed with ncu]`

Caveats that must be respected before acting:
- At C1 with MTP 3-1-4 the GEMM M is ~4–8. `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`
  is a 64×8-shaped tile — a skinny GEMM. **Tensor-core rate is irrelevant here; weight
  bytes are everything.** So the right target is **W4A16 / W8A16 weight-only**, not
  W4A4: quantizing activations at M=8 adds a quantize kernel and buys ~nothing.
  Our tree already has the machinery: `marlin_utils_fp4.prepare_nvfp4_layer_for_marlin`
  and `apply_fp4_marlin_linear`. `[verified — imported in `modelopt_quant.py`]`
- `kv_b_proj` is `[28672, 512]` — K=512 is fine for NVFP4 (multiple of 16). `q_b_proj`
  K=2048, `o_proj` K=16384, shared experts K=6144/2048: all fine. **No shape blocks us.**
  `[verified]`
- The NVFP4 training paper keeps *all* attention in high precision. That is about
  gradient stability, not inference. For inference, NVIDIA's own DeepSeek-R1-0528-FP4
  quantizes "the weights and activations of the linear operators within transformer
  blocks" — including attention projections. `[verified — HF card]` So attention
  *projections* in FP4 is a shipped configuration; attention *math* (QK/PV) in FP4 is
  not.
- The o_proj at `[6144, 16384]` is 100.7 MB/layer of the 330 MB — it alone is 24.6% of
  the BF16 traffic. If we only do one layer type, do `o_proj`. `[inferred, arithmetic]`

---

## 7. KV cache quantization

### 7.1 What FP8 E4M3 at scale = 1.0 actually costs

This is the part where INT8 intuition misleads. For an **integer** format the scale sets
both range and resolution, so a wrong scale is catastrophic. For a **floating** format
it only sets range:

- Inside E4M3's normal range (2⁻⁶ ≤ |x| ≤ 448), relative resolution is **fixed at
  2⁻⁴ = 6.25% worst-case RNE error**, regardless of scale. `[inferred from the verified
  3-mantissa-bit format]`
- Above 448 you **saturate** (with `.satfinite`) or get NaN (without). `[verified,
  `cuda_fp8.hpp:180-196`]`
- Between 2⁻⁹ and 2⁻⁶ you are subnormal and lose mantissa bits progressively.
- Below 2⁻⁹ ≈ 0.00195 you **flush to zero**.

So the failure modes of `k_scale = 1.0` are exactly two: **clipping at 448** and
**flushing tiny values**. There is no "wasted dynamic range" penalty in between. This
is why uncalibrated FP8 KV usually works and occasionally falls off a cliff.

E5M2 is the opposite trade: range to 57344, but only 2 mantissa bits → 12.5% worst-case
relative error. For KV cache, e4m3 is the right default. `[inferred]`

### 7.2 What our engine does, verified path by path

**Scale loading** — `NotSglang/python/sglang/srt/layers/quantization/kv_cache.py`
`[verified]`:

```python
# Initialize the KV cache scales to -1.0, which is an invalid value.
layer.k_scale = Parameter(tensor(-1.0)); layer.v_scale = Parameter(tensor(-1.0))
...
elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
    # If no scales were loaded ... use the default value of 1.0
    k_scale = 1.0; v_scale = 1.0
...
raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")
```

**Our checkpoint has zero `k_scale`/`v_scale` tensors.** I enumerated all 232,385 keys
in `model.safetensors.index.json`: `len([k for k in keys if 'k_scale' in k or 'v_scale'
in k]) == 0`. `[verified]` So the `-1.0` sentinel survives and both scales become 1.0,
despite `hf_quant_config.json` advertising `kv_cache_quant_algo: "FP8"`. **The
checkpoint's declared KV quantization is unbacked by any calibration artefact.** That is
a finding, not a configuration choice we made.

**Where the bytes go** — `mem_cache/kv_cache_configurator.py:1940-1974` `[verified]`:

```python
if server_args.dsa_prefill_backend == "trtllm" or dsa_decode_backend == "trtllm":
    return kv_cache_dim                     # <-- our box takes this branch
...
if kv_cache_dtype == torch.float8_e4m3fn:   # flashmla / other backends
    return (kv_lora_rank                    # 512 fp8 bytes
            + kv_lora_rank // 128 * 4       # 4 × fp32 per-128 block scales = 16 B
            + qk_rope_head_dim * 2)         # rope kept in BF16 = 128 B
```

Our resolved config is `dsa_prefill_backend = trtllm`, `dsa_decode_backend = trtllm`
(`personal_docs/glm-5.2/glm-5.2-optimization-log.md`, resolved by actually running
`prepare_server_args`) `[verified, local]`. Therefore:

| path | bytes/token/layer | latent scales | rope |
|---|---:|---|---|
| **ours (trtllm)** | **576** | none (global 1.0) | **FP8** |
| flashmla / deepgemm | 656 (+13.9%) | 4 × FP32 per 128 elements, **dynamic** | **BF16** |
| BF16 baseline | 1152 | — | BF16 |

And on the trtllm path, `MLATokenToKVPool._write_mla_kv_buffer` reaches the final `else`
because `dsa_kv_cache_store_fp8` is False (it requires `override_kv_cache_dim is not
None`) `[verified, `memory_pool.py:3908-3919, 4053-4065`]`:

```python
if cache_k_nope.dtype != self.dtype:
    cache_k_nope = cache_k_nope.to(self.dtype)   # bare cast, no scale
    cache_k_rope = cache_k_rope.to(self.dtype)   # bare cast, no scale
```

**Neither `k_scale` nor any block scale is applied. It is a raw `bf16 → e4m3` cast.**

**The DSA indexer cache is fine.** `DSATokenToKVPool._index_buffer_shape` allocates
`page_size * (128 + 128//128 * 4)` = 64 × 132 bytes, i.e. **FP8 keys + one FP32 scale
per token per layer, computed dynamically**. `[verified]` No calibration needed there.

### 7.3 The MLA-specific risk: the RoPE dims are never normalized

In DeepSeek-style MLA, `kv_a_proj_with_mqa` produces a 576-wide vector that is split
into a 512-wide latent and 64 RoPE dims, and **only the 512-wide latent goes through
`kv_a_layernorm`**. Our model definition confirms it:
`self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=...)` — width 512, not 576.
`[verified, `models/deepseek_v2.py:1870`]`

Consequences `[inferred, but from verified structure]`:

- The 512 latent dims are RMSNorm output scaled by a learned gain — magnitudes are
  O(1) by construction. E4M3 at scale 1.0 is close to ideal for them: centred in the
  normal range, nowhere near 448 or 2⁻⁹.
- The 64 RoPE dims are raw `kv_a_proj` outputs, then rotated. Rotation is norm-preserving
  per 2-dim pair, so it does not create outliers, but it does not bound them either.
  **Their scale is whatever the projection learned.** This is the only part of the KV
  cache with no normalization upstream of the FP8 cast, and on our path it *is* cast to
  FP8 with scale 1.0.
- The flashmla path deliberately keeps RoPE in BF16 — the code comment says so: "rope
  dimension is stored in original dtype (bf16), not quantized to fp8". `[verified]`
  Somebody thought this was worth 128 bytes per token per layer.

**This is the single highest-value measurement in this document** and it is a 20-line
script (§7.7).

Counter-evidence from our own tree, worth weighing: the SM90 q8kv8 sparse-prefill path
carries the comment "Identity per-tensor scales (scalar 1.0) are used: a raw bf16→fp8
cast of q/kv is accurate on real DeepSeek-V3 magnitudes, so no dynamic rescaling is
applied." `[reported — in-repo assertion at `dsa_backend.py:2626`, no measurement cited]`
That is an assertion about DeepSeek-V3 magnitudes, not GLM-5.2's, and it is about the
SM90 path, not the trtllm one.

### 7.4 The best public evidence on uncalibrated FP8 KV

The vLLM team ran a large validation in April 2026 and — critically — **used per-tensor
uncalibrated scale = 1.0 throughout**, calling it "the worst-case scenario for accuracy
… a lower bound". `[verified — vllm.ai/blog/2026-04-22-fp8-kvcache]`

Accuracy `[verified]`:

| model | benchmark | FP8 KV + FP8 attention recovery |
|---|---|---|
| Qwen3-30B-A3B-Thinking-2507 | AIME25/GPQA-D/MATH500/LCB-v6 | 1–2 points lost, worst 97% (GPQA-D) |
| Qwen3.5-27B | same | ≤0.7 points, worst 99% |
| Llama-3.3-70B-Instruct | MRCR AUC@128k | 97–98% |
| Qwen3-30B-A3B-Instruct-2507 | MRCR AUC@256k | 94% (BF16 model) / 98% (FP8 model) |
| Qwen3.5-27B | MRCR AUC@1M | full recovery |
| Qwen3-30B-A3B-Instruct-2507 on **B200/FlashInfer** | MRCR AUC@256k | 93% / 96% |
| **Kimi-K2.5 via FlashMLA** | MRCR | "consistent downward shift across sequence-length buckets … systematic rather than random" |

**The one model that showed systematic degradation is the MLA model on the FlashMLA
backend.** Their own conclusion: "This is especially relevant for models using
non-standard attention backends (e.g., FlashMLA) where the FP8 kernel behavior may
differ from the well-validated FA3 and FlashInfer paths." `[verified]` GLM-5.2 is an
MLA model on a trtllm-gen MLA kernel. We are in exactly the population where their
result says *calibrate*.

Their explicit "avoid FP8 KV" list `[verified]`: contexts < ~7k tokens; `head_dim = 256`
with prefill-dominated workloads; "uncalibrated accuracy drops below 95% on your
workload"; many small sliding-window layers.

Performance, for calibration of expectations `[verified]`:

| platform | model | ITL slope FP8 / BF16 | break-even |
|---|---|---:|---:|
| H100 / FA3 | Llama-3.1-8B | 54% | ~7.0k tokens |
| H100 / FA3 | gpt-oss-20b | 80% (71% skip-SW) | 22.1k (7.7k) |
| **B200 / FlashInfer** | Llama-3.1-8B | 1.80e-05 → 9.72e-06 = **54%** | ~4k tokens |
| B200 / FlashInfer | gpt-oss-20b | 3.56e-06 → 2.06e-06 = 58% | ~13k |

Also: the Hopper FA3 FP8 path needed a **two-level accumulation fix** (write partials to
a real FP32 register, per SageAttention2) because FP32 accumulation degrades when the
contraction dim reaches ~100k — 128k NIAH went 91% (BF16) → 13% (FP8) → 89% (fixed).
**"On B200, the accumulation issue is fixed, hence no explicit two-level accumulation is
needed."** `[verified]` Good news for us; it also means Hopper-derived FP8-KV folklore
does not apply here.

### 7.5 Scale granularity: the menu

| granularity | who supports it | cost | when it matters |
|---|---|---|---|
| per-tensor, uncalibrated (=1.0) | everything; **our current state** | free | baseline |
| per-tensor, calibrated | llm-compressor `kv_cache_scheme`, ModelOpt `kv/fp8.yaml` (`axis: null`) | one offline pass, 512 samples | fixes clipping / flushing |
| **per-attention-head** | vLLM + FA3 only, requires llm-compressor (`strategy="attn_head"`, `k/v_scale = [num_kv_heads]`) | array of scales in the kernel | heterogeneous heads |
| per-token (dynamic) | not for KV: the cache is written once, read many times | — | — |
| **per-block (dynamic)** | flashmla MLA layout: 4×FP32 per 512 latent dims | +16 B/token/layer | **no calibration needed at all** |
| affine / with zero-point | ModelOpt `kv/fp8_affine.yaml` (`bias: {-2:, -4:, type: static}`) | extra bias tensor | asymmetric K distributions |

`[verified — vLLM docs `features/quantization/quantized_kvcache.md`, ModelOpt
`configs/ptq/units/kv_*.yaml`]`

**MLA has a structural advantage here.** Per-head scales are meaningless for MLA: there
is exactly one latent "head" of width 576 shared by all 64 query heads. The natural
granularity is per-block along the 512-wide latent, which is precisely what the flashmla
layout does — **and a dynamic per-block scale needs no calibration**. For an MLA model,
"switch to the 656-byte layout" is strictly better engineering than "calibrate a
per-tensor scale", at the cost of 13.9% more KV bytes. `[inferred]`

### 7.6 INT4 / FP4 KV cache

- **NVFP4 KV is a shipped ModelOpt/TensorRT-LLM feature**: `kv/nvfp4.yaml`,
  `kv/nvfp4_affine.yaml`, `kv/nvfp4_cast.yaml`, `kv/nvfp4_rotate.yaml`; TRT-LLM exposes
  `KvCacheConfig(dtype='nvfp4')` and it "requires offline quantization with ModelOpt".
  `[verified]`
- **Our engine has an FP4 MLA KV pool already**: `MLATokenToKVPoolFP4` in
  `memory_pool.py:4149`, storing `(m, 1, k//2)` uint8 payload plus a separate
  `(m, k//16)` scale buffer, via `FP4MXBlock16KVQuantizeUtil`. `[verified]` Note the
  name says **MX**-block-16 — a 16-element block with (presumably) E8M0 scales, i.e. not
  standard NVFP4 and not standard MXFP4. Semantics not sourced; read
  `layers/quantization/kvfp4_tensor.py` before trusting it.
- **QServe (arXiv:2405.04532)** is the reference W4A8**KV4** system; its headline
  finding is that INT4 dequantization overhead is 20–90% of runtime if you are careless
  about where the dequant happens. `[verified, abstract]` The same warning applies to
  FP4 KV: you pay a dequant in the innermost attention loop.
- **KIVI (arXiv:2402.02750)** established the per-channel-K / per-token-V asymmetry that
  the ModelOpt `nvfp4_rotate` preset (rotate K only) reflects. `[verified, abstract]`
- For us, at 576 bytes/token/layer × 78 layers ≈ 43.9 KB/token FP8 (plus 10.1 KB/token
  indexer), the KV cache is not currently the binding constraint at our concurrencies.
  FP4 KV is a *capacity* play for the C1024 regime, not a latency play at C1.
  `[inferred, arithmetic]`

### 7.7 How to measure the scale = 1.0 risk on our box

Concrete, cheap, and it settles §7.3. `[inferred — this is a proposed procedure, not a
sourced one]`

1. **Instrument the write path.** In `MLATokenToKVPool._write_mla_kv_buffer`, before the
   `.to(self.dtype)` casts, accumulate per-layer running statistics on `cache_k_nope`
   and `cache_k_rope` separately:
   - `amax`, and the p99.9 of `|x|`;
   - `frac_over_448` (would saturate);
   - `frac_under_2^-9` (would flush to zero, excluding exact zeros);
   - `frac_subnormal` = fraction in `[2^-9, 2^-6)`.
   Run 200 real ShareGPT-shaped requests at 10k input. This costs one reduction per
   write and can be a debug-only env flag.
2. **Decision rule.** If `frac_over_448 > 0` on *any* layer, or `frac_under_2^-9 > 1%`
   on the RoPE half, scale = 1.0 is measurably lossy and the numbers tell you the
   correct per-tensor scale directly (`s = amax / 448`).
3. **End-to-end confirmation.** Three-way A/B on GSM8K + GPQA-Diamond + a 128k NIAH or
   MRCR bucket:
   - `--kv-cache-dtype bfloat16` (reference),
   - current `fp8_e4m3` @ 1.0,
   - `fp8_e4m3` with calibrated per-tensor scales (llm-compressor
     `kv_cache_scheme=QuantizationArgs(num_bits=8, type="float", strategy="tensor")`,
     512 ultrachat samples — the vLLM-documented recipe),
   - and, since it is a flag rather than a recalibration, the **flashmla** backend
     (656-byte layout, dynamic per-128 block scales, BF16 RoPE).
   The vLLM study's threshold — "uncalibrated accuracy drops below 95% on your workload"
   — is a reasonable action line.
4. **Long context specifically.** All the damage the vLLM team found on the MLA/FlashMLA
   path showed up as a *systematic downward shift across context buckets*, not as a
   short-prompt regression. GSM8K alone will not catch it. Use MRCR/NIAH buckets at
   32k / 64k / 128k.
5. **Cost of being wrong in the other direction:** the flashmla layout is +13.9% KV
   bytes and a different kernel family; at C1 our attention is 10.9% of kernel time, so
   a 13.9% KV-byte increase is at most ~1.5 pp if attention is fully bandwidth-bound.
   Cheap insurance. `[inferred]`

---

## 8. Activation quantization and the cost of the quantize kernels

### 8.1 Static vs dynamic, and what NVFP4 actually chose

NVFP4 activations are a **hybrid**: the per-tensor FP32 global scale is *static*
(calibrated, stored as `input_scale` in the checkpoint — 115,276 F32 scalars in ours),
and the per-16 E4M3 block scales are *dynamic* (computed in the quantize kernel every
forward). `compressed-tensors` encodes this as `dynamic=DynamicType.LOCAL` +
`observer="static_minmax"`. `[verified]`

This is the right split, and the reason is the two-level scheme: the global scale exists
only to bring block scales into E4M3's range, and that range requirement is a
distribution property that barely moves between batches. The block scales carry all the
per-token adaptivity and cost nothing extra to compute (you are already touching the
data to pack it).

MXFP4 activations are **fully dynamic** (`dynamic=True`, no observer) — E8M0's 2⁻¹²⁷…2¹²⁷
range means no global re-centring is ever needed. `[verified]` One fewer calibration
artefact to lose, which is not nothing given §7.2.

| scheme | activation scale | calibration | risk |
|---|---|---|---|
| FP8 per-tensor static | 1 FP32/tensor, offline | yes | distribution shift at serve time |
| FP8 per-token dynamic | 1 FP32/token | no | one extra reduction per token |
| FP8 block (DeepSeek) | 1×128 blocks | no | — |
| **NVFP4** | static FP32/tensor **+** dynamic E4M3/16 | global only | global scale stale → block scales clip in E4M3 |
| MXFP4 | dynamic E8M0/32 | none | coarser |

The FP8 reference recipe for contrast — NVIDIA's own FP8 baseline for the 12B model
follows DeepSeek-V3: "Scale factors apply on **128×128 blocks for weights** and
**1×128 blocks for activations and gradients**. They are computed online for each block,
stored in FP32." `[verified — arXiv:2509.25149 Appendix A.1]` Note that is *finer* than
the per-tensor FP8 that ModelOpt's `numerics/fp8.yaml` (`axis: null`) produces.

### 8.2 What the quantize kernels cost us

From the profile: **quant = 227 ms, 2.4% of kernel time, 76,137 launches** →
**≈ 2.98 µs per launch.** `[verified, arithmetic on the ledger table]`

For scale: attention is 1040 ms and the DSA indexer 556 ms. The quantize kernels are
40% of the size of the indexer. They are not the problem, but three observations:

- **2.98 µs is close to launch-bound.** At C1 the tensor being quantized is
  `[M≈4..8, 6144]` = ~100 KB. At 8 TB/s that is ~12 ns of memory time. We are paying
  ~250× that in overhead. `[inferred]`
- 76,137 launches over the window against 252,993 MoE-GEMM launches ≈ **one quantize per
  3.3 expert GEMMs** — consistent with quantizing the MoE input once per layer and
  reusing it across gate/up/down. `[inferred]`
- The obvious fix is fusion, and our tree already has an example:
  `kernels/ops/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py` — GEMM + SwiGLU +
  requantize-to-NVFP4 in one kernel. `[verified — file exists]` The profile's
  `bmm_E2m1_E2m1E2m1_Fp32_swiGlu_dynB_sm100f` (6.0%) is already the fused
  trtllm-gen variant, so some of this is done.
- **Do not add activation quantization to the dense path at C1.** A W4A4 conversion of
  `o_proj` would add a ~3 µs quantize to a GEMM that is bandwidth-bound on weights. Use
  W4A16/W8A16 there. `[inferred]`

---

## 9. Decision table for GLM-5.2 on 8×B200

Ordered by expected value. "Measurement" is what must exist before the change is
defensible.

| # | component | now | proposed | why | measurement that justifies it |
|---|---|---|---|---|---|
| 1 | `self_attn.o_proj` (100.7 MB/layer BF16) | BF16 | **NVFP4 or FP8 weight-only (W4A16/W8A16)** | 24.6% of the BF16 byte stream; K=16384 is FP4-legal; M≈8 at C1 so activation quant buys nothing | ncu on `nvjet_sm100_tst_64x8_*`: DRAM throughput vs peak. If >70% of HBM peak, the byte reduction converts ~1:1 |
| 2 | `q_b_proj`, `kv_b_proj`, `q_a_proj`, `kv_a_proj` | BF16 | same as #1 | remaining 229 MB/layer | same |
| 3 | `mlp.shared_experts.*` (75.5 MB/layer) | BF16 | NVFP4 (it is an FFN like the routed experts, K=6144/2048) | read on every token, unlike routed experts | GSM8K + GPQA delta vs the BF16 shared-expert build |
| 4 | KV cache RoPE dims | **FP8 @ scale 1.0** | measure first (§7.7); then either calibrate or move to flashmla 656-B layout | the only unnormalized tensor in the cache; the one public MLA data point (Kimi-K2.5) shows systematic loss | `frac_over_448` / `frac_under_2^-9` histogram on the write path + MRCR at 32k/64k/128k |
| 5 | KV cache latent (512 dims) | FP8 @ 1.0 | probably leave it | RMSNorm output → magnitudes O(1), well inside E4M3's normal range | same histogram; expect `frac_over_448 == 0` |
| 6 | routed experts | **NVFP4 W4A4** | keep | vendor-quantized, matches the hardware's best kind (`mxf4nvf4`, 4× Hopper FP8), 2.41:1 byte advantage already realised | none — this is the working part |
| 7 | router `mlp.gate` `[256,6144]` | BF16 | keep BF16 | on ModelOpt's default-disabled list; top-8-of-256 argmax is precision-sensitive and it is 1.5 MB/layer | none needed |
| 8 | `lm_head` `[154880,6144]`, embeddings | BF16 | keep BF16 | default-disabled everywhere; 1.9 GB each, read once per token but not per layer | if profiling shows lm_head is >2% of C1 time, revisit with W8A16 |
| 9 | first 3 layers (dense MLP) | BF16 (excluded) | keep | matches both ModelOpt exclusions and the FP4-training finding that early/late blocks are sensitive | — |
| 10 | **MTP / nextn layer 78 routed experts (19.33 GB BF16)** | BF16 (`mtp.*` default-disabled) | **NVFP4** — it is structurally identical to the main-model experts | 604 MB read per draft token vs 169.9 MB for a main layer (3.55×), and the draft runs 3× per verify step at 3-1-4 | draft-step latency before/after **and** accept-length (currently 3.16/4 on real data). Accept-length is the gate, not perplexity |
| 11 | attention math (QK, PV) | BF16/FP8 per backend | **do not** move to FP4 | no vendor ships FP4 attention math; the FP4 training recipe keeps softmax and both BMMs in high precision | — |
| 12 | MXFP4 anywhere | — | **no** | NVFP4 is same speed, strictly more accurate, and GLM-5.2 ships NVFP4 only | — |
| 13 | Hadamard/rotation PTQ | none | **no** (for weights/activations) | the only controlled study finds rotation neutral-to-harmful for 16-element-block FP4 | if we ever do FP4 *KV*, revisit — `kv/nvfp4_rotate.yaml` rotates K |
| 14 | quantize kernels (2.4%) | 76k launches @ 3.0 µs | fuse into the preceding norm/GEMM epilogue | launch-bound at C1 | nsys: gap between the quant kernel and its consumer |

**The one-line version:** the main model's routed experts are already in the best format
the hardware offers; the profile's biggest slice is the 12.3% of the checkpoint that
nobody quantized (attention, shared experts, and the entire MTP draft layer); and the KV
cache is running a configuration whose declared quantization algorithm has no
calibration artefact behind it.

**What would falsify the §6/§9 argument:** an ncu run showing
`nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT` is *not* DRAM-bound at C1 — e.g. sitting at
<40% of HBM peak because it is launch- or L2-bound at M≈8. In that case reducing weight
bytes buys nothing and the right move is kernel fusion / persistent kernels instead.
The whole chain in §6.4 rests on one unmeasured assumption, and §9 row 1's measurement
column exists to break it.

---

## Sources

### Read locally (files on this box)

- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda_fp8.hpp` — E4M3/E5M2 constants
  (lines 136–157), saturation behaviour (180–196)
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cuda_fp8.h`, `cuda_fp4.h`,
  `cuda_fp4.hpp` (`FP4_MAXNORM`, `maxnorm = 6.0` at 117–133), `cuda_fp6.h`
- `/home/aman/code/weights/GLM-5.2-NVFP4/hf_quant_config.json`
- `/home/aman/code/weights/GLM-5.2-NVFP4/model.safetensors.index.json` (232,385 keys)
- `/home/aman/code/weights/GLM-5.2-NVFP4/model-*.safetensors` (47 shard headers, byte
  census and dtype/shape enumeration)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/quantization/utils.py:597`
  (`swizzle_blockscale`)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/quantization/kv_cache.py`
  (`BaseKVCacheMethod`, the −1.0 → 1.0 fallback)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/quantization/fp4_utils.py`
  (`fp4_quantize`, `is_sf_8x4_layout`, `Fp4GemmRunnerBackend`)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/quantization/modelopt_quant.py`
- `/home/aman/code/NotSglang/python/sglang/srt/mem_cache/memory_pool.py`
  (`MLATokenToKVPool`, `MLATokenToKVPoolFP4`, `DSATokenToKVPool`)
- `/home/aman/code/NotSglang/python/sglang/srt/mem_cache/kv_cache_configurator.py:1940`
  (656-byte FP8 MLA layout vs trtllm passthrough)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/attention/dsa_backend.py:2623`
  (q8kv8 identity-scale rationale)
- `/home/aman/code/NotSglang/python/sglang/srt/models/deepseek_v2.py:1870`
  (`kv_a_layernorm` width = `kv_lora_rank`)
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/quantization/mxfp8_interleave_sf.py`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/glm-5.2-optimization-log.md`

### Read on the web

- PTX ISA 9.3 — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  (fetched in full; §5.2.3 alternate FP formats, §9.7.15.3 `mma` block scaling,
  §9.7.17.10 `tcgen05.mma` incl. Tables 54, 56, 57, 59, 60 and the packing-format
  subsections, `cvt` §9.7.9)
- CUTLASS Blackwell functionality —
  https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md
  (throughput table, narrow-precision type tables, alignment tables, scale-factor layout)
- https://docs.nvidia.com/cutlass/4.3.1/media/docs/cpp/blackwell_functionality.html
- Colfax Research, *CUTLASS Tutorial: Hardware-supported Block-scaling with NVIDIA
  Blackwell GPUs* —
  https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/
- NVIDIA, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference* —
  https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- *Pretraining Large Language Models with NVFP4*, arXiv:2509.25149 —
  https://arxiv.org/html/2509.25149v1 (Tables 1–2, §2, §4.1–4.4, §5, Appendices A.1, B, B.4)
- *A Comprehensive Evaluation on Quantization Techniques for Large Language Models*,
  arXiv:2507.17417 — https://arxiv.org/html/2507.17417v3 (§II-C, §III-C, Tables XIII–XV)
- Jarmusch & Chandrasekaran, *Microbenchmarking NVIDIA's Blackwell Architecture*,
  arXiv:2512.02189 — https://arxiv.org/pdf/2512.02189 (Tables IV–VII; FP6 figure disputed
  in §1.7)
- vLLM blog, *The State of FP8 KV-Cache and Attention Quantization in vLLM* (2026-04-22) —
  https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/main/_posts/2026-04-22-fp8-kvcache.md
- vLLM docs, *Quantized KV Cache* —
  https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/quantized_kvcache.md
- llm-compressor NVFP4 example + README —
  https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w4a4_fp4
- compressed-tensors preset schemes —
  https://raw.githubusercontent.com/neuralmagic/compressed-tensors/main/src/compressed_tensors/quantization/quant_scheme.py
- TensorRT Model Optimizer: `modelopt/torch/quantization/config.py`;
  `modelopt_recipes/configs/numerics/{nvfp4,nvfp4_bs32,nvfp4_static,nvfp4_four_over_six,mxfp4,fp8}.yaml`;
  `modelopt_recipes/configs/ptq/units/{default_disabled_quantizers,w4a4_nvfp4_nvfp4,kv_fp8,kv_fp8_affine,kv_nvfp4,kv_nvfp4_rotate}.yaml`;
  `modelopt_recipes/configs/ptq/presets/model/{nvfp4,mxfp4,nvfp4_experts_only,w4a8_nvfp4_fp8}.yaml`
  — https://github.com/NVIDIA/TensorRT-Model-Optimizer
- ModelOpt *Best practices to choose the right quantization methods* —
  `docs/source/guides/_choosing_quant_methods.rst`
- NVIDIA `nvidia/DeepSeek-R1-0528-FP4` model card —
  https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4/raw/main/README.md
- TensorRT-LLM *Quantization* feature docs —
  https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- FlashInfer `flashinfer/quantization/fp4_quantization.py` (SF size/padding helpers,
  `shuffle_matrix_sf_a`, E2M1 LUT) — https://github.com/flashinfer-ai/flashinfer
- Abstracts only: QuaRot arXiv:2404.00456; SpinQuant arXiv:2405.16406;
  QServe arXiv:2405.04532; KIVI arXiv:2402.02750; SmoothQuant arXiv:2211.10438;
  AWQ arXiv:2306.00978

### Explicitly not sourced

- The OCP Microscaling Formats v1.0 PDF itself (I relied on the PTX ISA and CUTLASS for
  MX semantics rather than claiming to have read the spec).
- Semantics of ModelOpt's `four_over_six: true` / FlashInfer's `FLASHINFER_NVFP4_4OVER6*`
  env vars. The flags exist `[verified]`; what they compute, not sourced.
- Semantics of `s2f6` in an LLM context.
- Semantics of `FP4MXBlock16KVQuantizeUtil` (our own FP4 KV pool) — read
  `layers/quantization/kvfp4_tensor.py` before using it.
- Any independent replication of NVIDIA's DeepSeek-R1-0528 NVFP4 benchmark table.
- Any measured accuracy number for our own GLM-5.2 NVFP4 + FP8-KV deployment; the
  GSM8K/GPQA runs are still pending.
