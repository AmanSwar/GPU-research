# Attention on Blackwell: FlashAttention on SM100, MLA, sparse attention, and decode kernels

## What this is

A working reference for the 16.7% of our GPU time that is attention (10.9% sparse MLA
+ 5.8% DSA indexer) on the 8×B200 GLM-5.2 node. It covers the SM100 attention
landscape as of August 2026 — FlashAttention 4, CUTLASS 77, cuDNN, FlashInfer,
trtllm-gen, FlashMLA — the algorithmic core (online softmax, two-pass, rescaling),
the Blackwell-specific implementation constraints (TMEM, tcgen05, exp2 roofline),
decode-regime kernel structure, MLA, and sparse/top-k attention. It ends with a
byte-level identification of `parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen`
from the cubin on this box, and the launch-grid arithmetic that follows from it.

Every claim is labelled `[verified]` (read in a primary source, path/URL given),
`[reported]` (a vendor/author asserts it), `[inferred]` (my reasoning, stated as
such), or `[unverified]`. Numbers I could not source say "not sourced".

---

## Bottom line for our system

- **The 6.0% kernel is fully identified.** `parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen`
  is the last 50 characters of
  `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512[HVPerCta{128,256}]PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen`,
  a trtllm-gen SM100 **sparse-MLA generation** kernel: FP8-E4M3 Q/K/V, BF16 out,
  headDimQk 576 / headDimV 512, **page size 1**, static token sparsity, KV split
  across CTAs with gmem reduction, Q tile 8, KV tile 128, A/B operands swapped.
  The cubin is on this box. `[verified]` §8.
- **It is not bandwidth-bound at C1 — but the "40× off roofline" framing is wrong, and
  the real headroom is ~2×.** The naive HBM floor for one layer of top-2048 sparse MLA
  decode is 1.18 MB → 144 ns at 8.18 TB/s, but that floor is *unreachable at this size*.
  Measured on this box (§2.6): an **empty** 128-CTA kernel costs **0.70 µs** back-to-back
  inside a CUDA graph, and a vectorised read of exactly 1.18 MB at 128 CTAs costs
  **2.17 µs**. Our ~5.7 µs is therefore **~2.6× a same-shape read**, not 40× off anything.
  The fix is still not a faster inner loop, but the prize is ~3 µs/launch, not 5.6.
  `[verified]` measurement, `[inferred]` attribution. §2.6, §5.5.
- **Attention is not where 365 → 500 tok/s comes from.** Attention + indexer is 16.7% of
  GPU time; deleting it *entirely* buys 1.20× → ~438 tok/s, still short of TileRT's
  published bar. Dense GEMM (37.1%) and the 47%-of-19.6% rank-arrival skew in the
  collectives are each larger prizes. Read this document to stop attention from being a
  *latency* tax, not expecting it to close the gap alone. `[inferred]` §5.5.
- **Kernel *count* is a first-order cost, and it is measurable.** 11.3M kernels / 8 ranks
  / ~1305 forward passes ≈ **1082 kernels per pass per rank**; at the measured 0.70 µs
  minimum kernel period that is **~758 µs/pass of irreducible launch/teardown**, ~11.7%
  of GPU-busy time. That is the quantitative case for TileRT's fused tile-runtime thesis.
  `[inferred]` on `[verified]` inputs. §2.6.
- **The four EAGLE draft tokens each get their own CTA row and their own KV gather.**
  trtllm-gen's sparse-MLA cubins all have `mGroupsTokensHeadsQ = false`, and
  `computeCtaAndClusterConfig` then sets `numCtasPerSeqQ = mMaxSeqLenQ` literally.
  Q-tile-32 sparse cubins exist but are unreachable at TP8 because
  `numHeadsPerCta = min(numHeadsQPerKv, stepQ) = 8`. `[verified]` §8.4.
- **The FP8 softmax scaling constant is visible in the SASS**: `FFMA2 ... , 8.8073549270629882812`
  = log2(448), i.e. the exponent is biased so `exp2` output saturates exactly at
  E4M3's max representable value before `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C`.
  That is the whole FP8-attention accuracy story in one constant. `[verified]` §3.6, §8.3.
- **The KV gather uses `UTMALDG.2D.GATHER4`** — the SASS form of
  `cp.async.bulk.tensor.2d...tile::gather4`, which fetches **4 arbitrary rows per
  instruction** with 5 coordinates. This is the hardware feature that makes page-size-1
  token-sparse attention cheap on SM100 and did not exist on Hopper. `[verified]` §7.5.
- **`index_topk_freq=4` buys more than it looks.** 21 of 78 layers run the indexer
  (layers 0,1,2,6,10,…,74, from SGLang's own formula and our config). Without sharing,
  the indexer would be ~21% of GPU time instead of 5.8%. `[verified]` §7.3.
- **FA4 exists, is installed here (`flash_attn_4-4.0.0b19`), and ships an SM100 MLA
  kernel with a DSA top-k gather path** (`flash_fwd_mla_sm100.py`,
  `is_topk_gather=True`, `topk_length=2048`). It requires `qhead_per_kvhead == 128`
  — "require MQA 128 for DSA path" — which we do **not** satisfy at TP8 (8 heads/rank).
  `[verified]` §3.5, §6.5.
- **Softmax, not MMA, is the structural bottleneck for dense attention on Blackwell**:
  8192 tensor-core ops/clk/SM vs 16 MUFU ops/clk/SM `[reported]`. FA4 answers with a
  degree-3 polynomial exp2 emulation on the FMA pipe. Our sparse decode kernel does
  *not* use emulation — 21 static `MUFU.EX2` and no Cody-Waite pattern. `[verified]` §2.3, §8.3.

---

## 1. The algorithmic core, precisely

### 1.1 The problem

For a query block `Q ∈ R^{M×d}`, keys `K ∈ R^{N×d}`, values `V ∈ R^{N×dv}`:

```
S = Q K^T / sqrt(d)            (M×N)
P = softmax_rowwise(S)         (M×N)
O = P V                        (M×dv)
```

Materializing `S` costs `O(M·N)` memory. FlashAttention tiles `N` into blocks and keeps
only `O(M)` running state.

### 1.2 Online softmax (the actual recurrence)

Process KV blocks `j = 1..T`. Maintain per row: running max `m`, running sum `ℓ`, and
unnormalized output accumulator `O`.

```
m_j   = max(m_{j-1}, rowmax(S_j))
α_j   = exp(m_{j-1} - m_j)                  # correction factor, ≤ 1
P̃_j   = exp(S_j - m_j)
ℓ_j   = α_j · ℓ_{j-1} + rowsum(P̃_j)
O_j   = α_j · O_{j-1} + P̃_j V_j
```
Final: `O = O_T / ℓ_T`, and `LSE = m_T + log(ℓ_T)`.

Two things matter for kernel design:

1. **`α_j` multiplies the whole `O` accumulator** (`M × dv` floats) every block. That is
   the "rescale" and it is *not* free — on Blackwell `O` lives in TMEM, so rescaling
   means `tcgen05.ld` → FMUL → `tcgen05.st` round-trips.
2. **Everything is done base-2 in practice.** `exp(x) = exp2(x · log2 e)`, and the
   `log2 e` factor is folded into the softmax scale once (`scale_log2`), so the inner
   loop is a single `exp2`. `[verified]` — FA4's `Softmax.online_softmax` takes
   `scale_log2` and calls `cute.math.exp2(acc_S_row * scale_log2 - row_max_cur_scaled)`:
   `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flash_attn/cute/softmax.py:170-181`.

### 1.3 Rescale skipping

The rescale is only *necessary* when the running max actually moved enough to threaten
overflow. FA4 skips it when `m_j − m_{j-1} ≤ τ`:

> "only rescale when mⱼ−mⱼ₋₁>τ, where τ is a threshold (typically set to log₂(256)=8.0,
> corresponding to a rescaling factor of 256.0)" `[reported]` — FA4 paper, arXiv 2603.05451.

The mechanism is in the shipped source as `SoftmaxSm100.rescale_threshold`:

```python
acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
acc_scale  = cute.math.exp2(acc_scale_, fastmath=True)
if cutlass.const_expr(self.rescale_threshold > 0.0):
    if acc_scale_ >= -self.rescale_threshold:
        row_max_new  = row_max_old
        row_max_safe = row_max_old
        acc_scale    = 1.0
```
`[verified]` `flash_attn/cute/softmax.py:376-386`. Note `acc_scale = 1.0` — the FMUL
still happens but the correction warp can skip the TMEM round-trip.

trtllm-gen has the same idea at the *kernel-variant* level, not the iteration level:
the metadata struct carries a `mSkipsSoftmaxWhenPossible` flag and the cubin set
contains `...VarSeqSkipsSoftmaxQ8Kv128...` variants. `[verified]`
`flashInferMetaInfo.h` struct field `bool mSkipsSoftmaxWhenPossible;`.

### 1.4 Split-KV (flash-decoding) and the LSE reduction

When `M` is tiny (decode) there is no parallelism in the query dimension. Split the KV
axis across `S` CTAs; each produces a partial `(O^{(s)}, LSE^{(s)})`. Combine:

```
m*  = max_s LSE^{(s)}
O   = Σ_s exp(LSE^{(s)} − m*) · O^{(s)}   /   Σ_s exp(LSE^{(s)} − m*)
```

This is exact — it is the same online-softmax merge, applied across splits instead of
across blocks. Three places to do the reduction, all present in trtllm-gen as an enum:

```cpp
enum class MultiCtasKvMode {
  Disabled = 0,
  GmemReduction,                    // global memory + atomic counters
  GmemReductionWithSeparateKernel,  // 2-CTA / keepsMmaAb MLA with large reduction tiles
  CgaSmemReduction                  // CGA (cluster) remote shared memory
};
```
`[verified]` `flashinfer/data/include/flashinfer/trtllm/fmha/fmhaRunnerParams.h:136-147`.

`CgaSmemReduction` is the Blackwell-specific one: it uses distributed shared memory
across a thread-block cluster (max cluster dim 16) instead of a gmem round-trip. **It is
disabled for us** — see §8.5.

---

## 2. What Blackwell (SM100) changed, and what it costs attention

### 2.1 Measured machine facts on this box

| quantity | value | source |
|---|---|---|
| SM count | **148** | `torch.cuda.get_device_properties(0).multi_processor_count` `[verified]` |
| HBM3e capacity | 183,359 MiB | `nvidia-smi` `[verified]` |
| Memory clock (max) | **3996 MHz** | `nvidia-smi -q` `[verified]` |
| L2 cache | **132,644,864 B (126.5 MiB)** | `torch` device props `[verified]` |
| Max shared mem / block (opt-in) | **232,448 B** | `torch` device props `[verified]` |
| Registers / SM | 65,536 | `torch` device props `[verified]` |
| Compute capability | 10.0 | `torch` device props `[verified]` |
| HBM bandwidth | 8192 bit × 3996 MHz × 2 / 8 = **8.18 TB/s** | `[inferred]` from clock + 8-stack HBM3e; vendor figure 8 TB/s `[reported]` |

The 232,448 B smem limit is load-bearing: the biggest sparse-MLA cubins request
226,120 B (`...MultiCtasKvCgaVarSeqQ32Kv128...`), leaving essentially no headroom.
`[verified]` from `flashInferMetaInfo.h`.

### 2.2 Tensor Memory (TMEM)

- **256 KB per SM, 512 columns × 128 lanes of 32-bit cells.** `[reported]` Colfax;
  independently corroborated by FA4's `SM100_TMEM_CAPACITY_COLUMNS = 512`
  (`flash_attn/cute/flash_fwd_mla_sm100.py:236`) `[verified]`.
- TMEM address: bits 31–16 = lane, bits 15–0 = column. `[reported]` Colfax.
- **Allocation must be a power of 2 and ≥ 32 columns**, from a single warp, and the same
  warp must deallocate. `[reported]` Colfax.
- **A warp can only access 32 of the 128 lanes** (warp 0 → lanes 0–31, warp 1 → 32–63, …).
  `[reported]` Colfax. This is why FA4's softmax warpgroups are shaped the way they are.

The PTX surface, read from the local CUDA 13.3 CCCL headers `[verified]`
(`/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/`):

```
tcgen05.alloc.cta_group::{1,2}.sync.aligned.shared::cta.b32 [dst], nCols;
tcgen05.dealloc.cta_group::{1,2}.sync.aligned.b32 taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group::{1,2}.sync.aligned;
```
Available shapes for the register↔TMEM data path (`generated/tcgen05_ld.h`,
`generated/tcgen05_st.h`):

| shape | `.num` multiplicities available |
|---|---|
| `.16x32bx2` | x1 … x128 |
| `.16x64b`   | x1 … x64 |
| `.16x128b`  | x1 … x64 |
| `.16x256b`  | x1 … x32 |

plus `.pack::16b` / `.unpack::16b` variants for 16-bit packing. `[verified]`

The `tcgen05.mma` `.kind` space (`generated/tcgen05_mma.h`) `[verified]`:
`kind::f16`, `kind::tf32`, `kind::f8f6f4`, `kind::i8`, `kind::mxf8f6f4`,
`kind::mxf4`, `kind::mxf4nvf4`, with `.block_scale.block16` / `.block32` and
`scale_vec::{1,2,4}`, and the `collector::a::{fill,use,lastuse,discard}` qualifiers
that let a kernel keep the A operand resident across MMAs.

### 2.3 The exp2 roofline — the single most important Blackwell attention fact

> "Blackwell doubles the FP16/BF16 tensor core throughput compared to Hopper (2.25
> PFLOPS vs 1 PFLOPS per GPU), but shared memory bandwidth and exponential unit
> throughput remain unchanged." Tensor cores: **8192 ops/clock/SM**. Exponential unit
> (MUFU): **16 ops/clock/SM**, unchanged from Hopper. `[reported]` FA4 paper.

> "for typical attention workloads on Blackwell, surprisingly, shared memory traffic and
> exponential operations now dominate execution time, exceeding MMA compute by 25–60%."
> `[reported]` FA4 paper, Table 1.

Do the arithmetic for a standard head-dim-128 tile: `S` is `M×N` and needs `M·N` exp2s;
the two MMAs are `2·M·N·d + 2·M·N·dv = 4·M·N·128` FLOPs at d=dv=128, i.e. 512 FLOP per
exp2. With a 512:1 hardware ratio (8192:16), **exp and MMA are exactly balanced at
d=128** — which means any inefficiency in softmax immediately becomes the critical path,
and any head dim below 128 makes softmax dominant outright.

FA4's answer is **partial emulation**: compute some exp2s on the FMA pipe with a
polynomial so MUFU and FMA both contribute. The shipped implementation `[verified]`
(`flash_attn/cute/utils.py:32-48, 735-795`):

```python
POLY_EX2 = {
  3: (1.0, 0.695146143436431884765625,
           0.227564394474029541015625,
           0.077119089663028717041015625),
  ...
}

def ex2_emulation(x, poly_degree=3):
    fp32_round_int = float(2**23 + 2**22)          # 12582912.0
    x_clamped   = cute.arch.fmax(x, -127.0)
    x_rounded   = add_round_down(x_clamped, fp32_round_int)   # "add.rm.ftz.f32"
    x_rounded_back = x_rounded - fp32_round_int
    x_frac      = x_clamped - x_rounded_back        # in [0,1)
    x_frac_ex2  = evaluate_polynomial(x_frac, POLY_EX2[poly_degree])
    return combine_int_frac_ex2(x_rounded, x_frac_ex2)
```
and the recombination is pure integer bit-twiddling on the exponent field:

```ptx
shl.b32 x_rounded_e, x_rounded_i, 23;
add.s32 out_i,       x_rounded_e, frac_ex_i;   // add.s32 → LEA (ALU); add.u32 → IMAD (FMA pipe)
```
`[verified]` `flash_attn/cute/utils.py:735-757`. The comment in the source explicitly
notes the ALU-vs-FMA-pipe choice — this is pipe-balancing, not numerics.

`ex2_emulation_2` does two at a time on the **packed FP32x2** datapath
(`cute.arch.fma_packed_f32x2`, `add_packed_f32x2`), which is the Blackwell FP32 vector
path that shows up in SASS as `FFMA2 / FMUL2 / FADD2 ... .F32x2.HI_LO`. `[verified]`

> Accuracy: "Degree-3 polynomial has a maximum relative error of 8.8×10⁻⁵" at FP32,
> "after rounding to BF16, the errors become nearly indistinguishable". `[reported]` FA4 paper.

**For us this is mostly not the binding constraint.** Our decode kernel computes
`8 rows × 2048 keys = 16,384` exp2 per layer per token — 1024 per CTA at 16 CTAs.
At 16/clk/SM that is 64 clocks. Irrelevant. The exp2 roofline matters for *prefill*,
where GLM-5.2 at 10k input does `10k × 2048` scores per layer.

### 2.4 The two-GEMM dependency and where softmax runs

The chain is `MMA(QK) → softmax → MMA(PV)`. On Hopper (FA3) the accumulator lives in
registers, so softmax reads it directly and the warpgroup that issued `wgmma` also does
the softmax; overlap comes from ping-ponging two query tiles.

On Blackwell the accumulator lives in **TMEM**, which is not registers. So:

1. `tcgen05.mma` writes `S` to TMEM asynchronously and signals an mbarrier.
2. A **softmax warp** must `tcgen05.ld` `S` into registers, do max/exp2/sum, and
   `tcgen05.st` the `P` back (or write it to smem for the second MMA's A operand).
3. A **correction warp** applies `α_j` to the `O` accumulator, also in TMEM.
4. The MMA warp issues the second `tcgen05.mma` for `P·V`.

FA4 turns each of these into a dedicated warp. From the shipped source `[verified]`
(`flash_attn/cute/flash_fwd_sm100.py:254-260`):

```python
self.softmax0_warp_ids   = (0, 1, 2, 3)
self.softmax1_warp_ids   = (4, 5, 6, 7)
self.correction_warp_ids = (8, 9, 10, 11)
self.mma_warp_id         = 12
self.epilogue_warp_ids   = (13,)
self.load_warp_ids       = (14,)
self.empty_warp_ids      = (15,)
```

16 warps = 512 threads. Two softmax warpgroups ping-pong on two `S` stages; the
correction warpgroup takes the `O` rescale **off the critical path**; one warp drives
both `tcgen05.mma` and TMA. `[verified]` — and this matches the paper's prose
`[reported]`.

The crucial Blackwell-specific detail: `SoftmaxSm100` sets `num_rows = 1`
(`flash_attn/cute/softmax.py:250-260`) `[verified]`. **One thread owns one entire row of
`S`.** That removes the 4-thread `warp_reduction_max(…, threads_in_group=4)` shuffle that
the Ampere/Hopper `Softmax` class needs (`softmax.py:158`). It is possible only because
TMEM's `16x256b` load shape lets one thread pull a full row segment.

### 2.5 2-CTA MMA

`tcgen05.mma.cta_group::2` lets a CTA *pair* in a cluster cooperatively execute one MMA,
with M = 128 or 256 partitioned across the pair. `[verified]` PTX qualifier exists in
`generated/tcgen05_mma.h`; `[reported]` FA4 paper for the M-partitioning semantics.

FA4's MLA forward uses it unconditionally `[verified]`
(`flash_fwd_mla_sm100.py:169-173`):

```python
self.use_2cta_instrs = True
self.cta_group       = tcgen05.CtaGroup.TWO
self.cta_group_size  = 2
self.cluster_shape_mn = (2, 1)
```

trtllm-gen ships 2-CTA sparse-MLA cubins only for `tileQ=64, hdPerCtaV=256`
(`...VarSeqQ64Kv128Static2CtaKeepsAbForGen`). `[verified]` metadata scan.

---

## 3. FlashAttention on SM100

### 3.1 Status

FA3 was Hopper-only: `wgmma` + TMA + warp specialization + register accumulators. None
of that transfers — `wgmma` does not exist on SM100 and register accumulators are
replaced by TMEM.

**FlashAttention 4 is real, released, and installed on this box.** Version
`flash_attn_4-4.0.0b19`, at
`/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flash_attn/`. `[verified]`
Paper: arXiv 2603.05451, "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design
for Asymmetric Hardware Scaling". `[verified]` (fetched).

It is written **entirely in CuTe-DSL embedded in Python** — 47,907 lines across
`flash_attn/cute/*.py`, no C++ templates. `[verified]` (`wc -l`). The paper claims
"20-30× faster compile times compared to traditional C++ template-based approaches"
`[reported]`.

### 3.2 The SM100 file map (local, verified)

| file | lines | what |
|---|---:|---|
| `flash_fwd_sm100.py` | 3150 | dense forward, warp-specialized, 16 warps |
| `flash_fwd_mla_sm100.py` | 3160 | **MLA forward incl. DSA top-k gather** |
| `flash_bwd_sm100.py` | 4041 | dense backward |
| `flash_bwd_mla_sm100.py` | 2133 | MLA backward |
| `sm100_hd256_2cta_fmha_forward.py` | 1889 | head-dim-256 2-CTA forward |
| `topk_gather_kv.py` | — | `CpasyncGatherKVManager` for sparse KV |
| `block_sparse_utils.py` | 1543 | block-sparse masks |
| `flash_fwd_combine.py` | 698 | split-KV LSE reduction |
| `softmax.py` | 698 | `Softmax`, `SoftmaxSm100` |
| `mma_sm100_desc.py` | — | UMMA descriptors |

`[verified]` directory listing + `wc -l`.

### 3.3 Performance claims

> "reaching up to 1613 TFLOPs/s (71% utilization)" on B200; "Up to 1.3× speedup over
> cuDNN 9.13 and 2.7× over Triton" (head dim 128, seqlens 1k–32k). `[reported]` FA4 paper.

Note what that is *not*: it is BF16 dense prefill. **The paper contains no FP8 attention
results and no decode-specific algorithm** — it mentions varlen scheduling for mixed
prefill/decode but no decode kernel study. `[reported]` (fetched, explicitly checked).

### 3.4 Backward pass (for completeness)

Five MMAs per iteration; `S`/`P` share one TMEM block at offset 0, `dP`/`dS`/`dQ` share
the other; 2-CTA variant uses M=256 except `dQ` which uses M=128 with a double reduction
2N=256. `[reported]` FA4 paper. Not relevant to inference.

### 3.5 FA4's MLA + DSA path (directly relevant)

`FlashAttentionMLAForwardSm100.__init__` `[verified]`
(`flash_fwd_mla_sm100.py:48-200`):

```python
def __init__(self, is_causal=False, use_cpasync_load_KV=False,
             topk_length=2048, is_topk_gather=True, pack_gqa=False,
             qhead_per_kvhead=1, nheads_kv=1, hdim=64, hdimv=512, ...):
    if is_topk_gather:
        assert pack_gqa
        assert qhead_per_kvhead == 128, "require MQA 128 for DSA path"
        assert use_cpasync_load_KV
```

Shape and resource constants `[verified]`:

| knob | value |
|---|---|
| `cta_tile_m` / `cluster_tile_m` | 64 / 128 (2-CTA) |
| `tile_n` | 128 |
| `hdim` (rope) / `hdimv` (latent) | 64 / 512 |
| `num_hdimv_splits` | 2 — "split hdimv in half for our Qv @ V^T and P @ V mmas" |
| threads | 512 (16 warps) or 384 (12 warps) |
| warp roles | softmax (0–3), epilogue (4–7), load 8, mma 9, clc 10, relay 11, cpasync load (12–15) |
| regs (16-warp) | load 112, mma 112, **softmax 192**, epilogue 128, cpasync 80, other 48 |
| pipeline stages | Q 1, K 1, Qv 2, **V 4**, S 2, P 1, Oi 1, sm_stats 2, bitmask 2 |

TMEM budget, computed from the source constants `[verified]`:

```
tmem_cols_S  = tile_n / cta_group_size            = 128/2  = 64   per stage, ×2 stages = 128
tmem_cols_Oi = (hdimv / num_hdimv_splits) / 2     = 256/2  = 128  per O buffer, ×2      = 256
total                                                              = 384 of 512 columns
```
i.e. 75% of TMEM is live. That is the real reason `num_hdimv_splits = 2` exists.

The MMA tilers `[verified]`:
```python
mma_tiler_QK  = (cluster_tile_m=128, tile_n=128, hdim=64)          # rope part
mma_tiler_QvV = (128, 128, hdimv//2 = 256)                          # latent part of scores
mma_tiler_PVt = (128, hdimv//2 = 256, tile_n=128)                   # output
```
This is the absorbed-MLA decomposition made explicit: the score is the sum of a 64-deep
rope GEMM and a 512-deep latent GEMM, and the same latent buffer is then re-used as `V`.

**Gather is `cp.async`, not TMA.** `CpasyncGatherKVManager` (`topk_gather_kv.py`) takes
`mIndexTopk`, requires `num_threads == 128`, `hdim % 64 == 0`, uses 128-bit universal
copies, and maintains an optional `sBitmask` for out-of-range indices. `[verified]`
This is the interesting divergence from trtllm-gen, which uses `tile::gather4` TMA
(§7.5).

### 3.6 FP8 attention: scaling and accuracy

The FA4 paper has nothing on FP8 `[reported]`. But the practice is visible in the
trtllm-gen SASS on this box, and it is worth writing down because it is the whole
answer to "how do you keep FP8 attention accurate":

1. **Q, K in E4M3, per-tensor descale.** `S` accumulates in FP32 in TMEM. The descale is
   folded into `scale_log2` so it costs nothing.
2. **`P` must be quantized to E4M3 for the second MMA.** E4M3 max is 448. Since
   `P ∈ (0, 1]` after the max subtraction, naively casting wastes ~7 exponent bits.
   The fix: bias the exponent by `log2(448)` so the largest probability maps to exactly
   448, then divide the final `O` by 448.
3. In SASS this is literally one constant:
   ```
   FFMA2 R26, R26.F32x2.HI_LO, R23.F32, 8.8073549270629882812 ;
   MUFU.EX2 R28, R8 ;
   ...
   F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R4, R35, R34, RZ ;
   ```
   `log2(448) = 8.807354922...`. `[verified]` — disassembly of
   `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen.cubin`,
   offsets `0xa280`–`0xa440`.
4. **`V` in E4M3 with the output in BF16** — our kernel's metadata is
   `dtQ=dtK=dtV=E4M3, dtO=BF16`. `[verified]`

The accuracy consequence: the exponent bias is exact (a power-of-2 scale is
representable exactly), so the only FP8 error is the 3-bit mantissa on `P` and `V`.
`[inferred]`

---

## 4. The other SM100 attention stacks

### 4.1 CUTLASS example 77 (`77_blackwell_fmha`)

Provides forward (context + generation, fp8/fp16/bf16), backward
(`FmhaKernelBwdSumOdO`, `Sm100FmhaBwdKernelTmaWarpSpecialized`, `FmhaKernelBwdConvert`,
`Sm100FmhaBwdMlaKernelTmaWarpSpecialized`), and a dedicated **MLA inference kernel in
the "weight-absorbed regime" supporting latent head dim 512 and rope head dim 64**.
`[reported]` (fetched README).

Tile shapes `[reported]`: context M=256 (seqlen-Q) × N=128 (seqlen-K); generation
M=128 × N ∈ {64, 128, 256}. Head dims: forward 32/64/128, backward 64/128. Context uses
TMA; **generation uses `cp.async` for flexible load patterns**; the MLA sample supports
"TMA (either without paging or with page size 128) or `cp.async`".

Design: "reuses the selection logic of the collective gemm builder", warp-specialized,
"two tiles to each threadblock, and pingpongs between them". Fusion hook is
`collective/fmha_fusion.hpp :: apply_mask`. `[reported]`

CUTLASS 4.x changelogs mention "Softmax skip correction" added to the FMHA kernels
`[reported]` — the same τ-threshold idea as §1.3.

### 4.2 cuDNN

FA4 benchmarks against **cuDNN 9.13** and beats it by 1.1–1.3× at head dim 128
non-causal `[reported]`. That makes cuDNN the strongest general-purpose SM100 attention
baseline outside FA4. FlashInfer exposes it as `cudnn_batch_decode_with_kv_cache`
`[reported]` (FlashInfer docs). I did not find a public cuDNN SM100 attention
architecture document; **not sourced**.

### 4.3 FlashInfer

Backends exposed `[reported]` (docs.flashinfer.ai/api/attention.html, fetched):

| backend | notes |
|---|---|
| FA2 / FA3 | split-KV decode, varlen, `window_left` |
| `trtllm-gen` | production; RoPE, ALiBi, soft-cap, sinks; **default on SM100** |
| `cudnn` | `cudnn_batch_decode_with_kv_cache` |
| `cute-dsl` | Blackwell GQA decode; **requires equal `head_dim_qk`/`head_dim_vo`, no RoPE/ALiBi/soft-cap** |
| PrimTS (`flashinfer.attention.prims_ts`) | experimental Blackwell task-scheduled FMHA context / decode / MLA decode; automatic tile selection + split-KV reduction; "accuracy and performance signoff is on SM100a/B200" |
| MLA | `BatchMLAPagedAttentionWrapper`, `trtllm_batch_decode_with_kv_cache_mla`, `prims_ts_batch_decode_with_kv_cache_mla` |
| XQA | `xqa()` — multi-beam decode with speculative decoding support |
| MSA sparse | `msa_sparse_attention`, `msa_proxy_score`, `msa_topk_select` |

Page sizes supported: 16, 32, 64, 128 `[reported]`. Note the **sparse-MLA path is
special-cased to page size 1** — see §7.5.

Version on this box: `flashinfer` 0.6.15.post1 `[verified]` (optimization log).

### 4.4 trtllm-gen

This is what actually runs for us. It is a **cubin database**, not a template library:
FlashInfer downloads pre-compiled `.cubin`s and a generated metadata header listing every
variant. On this box:

```
/home/aman/.cache/flashinfer/cubins/158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/
  ├── 13 *.cubin (the ones our workload actually touched)
  └── include/flashInferMetaInfo.h   (5.4 MB, 13,452 kernel rows)
```
`[verified]`

The metadata struct is the naming key. Full field list `[verified]`
(`flashInferMetaInfo.h:30-67`):

```cpp
struct TllmGenFmhaKernelMetaInfo {
  Data_type mDataTypeQ, mDataTypeKv, mDataTypeK, mDataTypeV, mDataTypeO;
  int mTileSizeQ, mTileSizeKv, mStepQ, mStepKv;
  int mHeadDimPerCtaV, mHeadDimQk, mHeadDimV;
  int mSM; const unsigned char* mCubin; unsigned int mCubinSize;
  const char* mFuncName; int mSharedMemBytes; int mThreadsPerCTA;
  int mQkvLayout; int mNumTokensPerPage; int mMaskType; int mKernelType;
  int mTileScheduler; int mMultiCtasKvMode;
  int mNumEltsPerSageAttnBlkQ/K/P/V;
  bool mGroupsHeadsQ; bool mGroupsTokensHeadsQ;
  bool mReuseSmemKForV; bool m2CtaMma;
  int mSparseAttn; bool mSkipsSoftmaxWhenPossible;
  bool mReserved1, mReserved2; const char* sha256;
};
```

Enums `[verified]` (`fmhaRunnerParams.h`):

```
QkvLayout            : SeparateQkv=0, PackedQkv=1, PagedKv=2, ContiguousKv=3
FmhaKernelType       : Context=0, Generation=1, SwapsMmaAbForGeneration=2,
                       KeepsMmaAbForGeneration=3, SpecDecodingGeneration=4
TileScheduler        : Static=0, Persistent=1
MultiCtasKvMode      : Disabled=0, GmemReduction=1,
                       GmemReductionWithSeparateKernel=2, CgaSmemReduction=3
TrtllmGenSparseMlaType: None=0, StaticTokenSparse=1, DynamicTokenSparse=2
```

And the SwapsAb rule, verbatim from the header comment `[verified]`:

> "Choose the best generation kernel based on the heuristic: use
> SwapsMmaAbForGeneration kernels when numHeadsQPerKv <= 16, otherwise
> KeepsMmaAbForGeneration." / "Swap tensor A and tensor B of Mma, which only supports
> numHeadsQPerKv <= 16."

**Swapping A and B is the decode trick**: with `numHeadsQPerKv = 8`, the Q side of the
QK GEMM is only 8 rows. tcgen05 MMA wants M = 128. So put the **KV tile in M** (128) and
the grouped-heads Q in N (8). MMA-M is then fully occupied and only the (cheap) N
dimension is small.

### 4.5 XQA

TensorRT-LLM's decode-specialized attention kernel, exposed via FlashInfer's `xqa()`
with multi-beam and speculative-decoding support `[reported]` (FlashInfer docs). I could
not find a primary architecture description for the SM100 XQA implementation;
**not sourced**. It is not on our path (we use trtllm-gen sparse MLA).

### 4.6 FlashMLA

DeepSeek's MLA kernel library. `[reported]` (fetched README):

| item | value |
|---|---|
| Architectures | **SM90 / SM100** |
| Kernels | dense prefill/decode; **token-level sparse prefill**; **token-level sparse decode with FP8 KV cache** |
| MQA head dims | `head_dim_k = 576`, `head_dim_v = 512` |
| MHA head dims | `head_dim_k = 192/128`, `head_dim_v = 128` |
| Dense decode | "up to 3000 GB/s memory-bound and 660 TFLOPS compute-bound on H800 SXM5" |
| Sparse decode | "410 TFLOPS compute-bound on H800 SXM5" |
| Sparse prefill | "up to 640 TFlops forward on H800 SXM5" |
| MHA prefill on **B200** | "up to 1460 TFlops forward and 1000 TFlops backward" |
| API | `get_mla_metadata`, `flash_mla_with_kvcache`, `flash_mla_sparse_fwd` |

Note all the MLA numbers are quoted on **H800**, not B200. The only B200 number is for
dense MHA prefill. `[reported]` — so FlashMLA's B200 sparse-MLA decode performance is
**not sourced**.

SGLang wires it in as `--dsa-prefill-backend flashmla_sparse` (bf16) and
`flashmla_sparse_q8` (native FP8, **SM90-only, prefill-only** — it asserts on
compute capability). `[verified]`
`/home/aman/code/NotSglang/python/sglang/srt/layers/attention/dsa_backend.py:486-511`.

### 4.7 TileRT (the thing to beat)

Local checkout: `/home/aman/code/third_party/TileRT` v0.1.5. `[verified]`

Its sparse MLA decode op `[verified]`
(`tilert/models/deepseek_v3_2/ops/flash_sparse_mla.py`):

```python
def flash_sparse_mla(query, query_pe, key_value, key_pe, indices, cur_pos,
                     output, profile_logs, split_size=64,
                     compute_kernel_type="bf16mma", *, model_arch):
    if split_size != 64: raise ValueError(...)   # only 64
    if batch != 1:       raise ValueError(...)   # only batch 1
    if seqlen > 4:       raise ValueError(...)   # only ≤4 draft tokens
    max_num_splits = 32
    if heads not in (8, 10, 16, 20): raise ValueError(...)
```

Read that carefully:

- `split_size = 64`, `max_num_splits = 32` → `64 × 32 = 2048` exactly. **The split
  structure is hard-wired to the DSA top-k.** `[verified]`
- `heads ∈ {8, 10, 16, 20}` — 8 is precisely GLM's 64 heads at TP8. `[verified]`
- `seqlen ≤ 4` — matches an MTP/EAGLE draft width of ≤4, and **they process all draft
  tokens in one call**, unlike trtllm-gen. `[verified]`
- `compute_kernel_type = "bf16mma"` — they run the sparse MLA MMA in **BF16**, not FP8.
  `[verified]` (only `BF16MMA` is in `_SUPPORTED_ALGORITHMS`.)
- Separate `FlashSparseMLACombine` module for the 32-way LSE reduction. `[verified]`

Grid implied: 32 splits × (heads/tile) × 1 batch. With 8 heads in one tile that is 32
CTAs — *fewer* than trtllm-gen's 128 (§8.5), but each CTA does 4 draft tokens at once and
produces all 512 output dims. `[inferred]`

TileRT publishes **~500 tok/s on GLM-5-FP8 and ~600 on DeepSeek-V3.2 on 8×B200**
`[reported]` (README). No GLM-5.2 support announced `[verified]` (README model list).

---

## 5. Decode attention: our regime

### 5.1 Why decode is a different kernel

At concurrency 1 with EAGLE 3-1-4 we have `M ∈ {1, 4}` query positions against
`N ∈ {2048}` selected keys. Consequences:

- **No query-dimension parallelism.** Parallelism must come from batch, heads, KV splits,
  or output-dim splits.
- **Arithmetic intensity collapses.** See §5.4.
- **The MMA is mostly idle.** Even with A/B swapped so KV occupies M=128, the N
  dimension is `numHeadsQPerKv = 8`. tcgen05 allows N in multiples of 8, so N=8 is legal
  but uses 8/256 = 3% of the maximum MMA output tile. `[inferred]` from the PTX N range
  and our `mStepQ = 8`.
- **Prologue and epilogue stop being amortizable.** TMA descriptor prefetch, TMEM alloc,
  index-buffer load, and the cross-CTA reduction are a fixed cost against a 2-iteration
  mainloop.

### 5.2 Split-KV / flash-decoding structure

Three orthogonal split axes, all present in trtllm-gen:

| axis | field | effect |
|---|---|---|
| KV | `mMultiCtasKvMode` + computed `numCtasPerSeqKv` | more CTAs, needs LSE reduction |
| output dim | `mHeadDimPerCtaV` (128 / 256 / 512) | more CTAs, **re-reads K in every CTA** |
| heads | `mGroupsHeadsQ`, `mStepQ` | fewer CTAs, no re-read |

The KV-split cap is deliberately conservative `[verified]`
(`fmhaKernels.cuh:490-497`):

```cpp
// The factor of 2 is applied here to ensure the reduction overhead does not
// outweigh the benefits of a shorter mainloop.
int const maxNumCtasPerSeqKv =
    (maxAttentionWindow + 2 * kernelMeta.mStepKv - 1) / (2 * kernelMeta.mStepKv);
```
With `maxAttentionWindow = min(seqLenKv, topK) = 2048` and `stepKv = 128`:
`maxNumCtasPerSeqKv = 2048 / 256 = 8`. **Each KV CTA is guaranteed ≥ 2 KV tiles.**

### 5.3 Paged / blocked KV and page size

`QkvLayout::PagedKv` means the K/V buffer is `[batchSize, 2, maxNumPagesPerSeq]` of
*logical page indices*, and each page is `[numHeadsKv, pageSize, headDim]`. `[verified]`
`fmhaRunnerParams.h:95-112`.

Page size controls two things:

1. **Coalescing.** A page of `pageSize × headDim` FP8 elements is contiguous. For MLA,
   `headDim = 576` FP8 bytes = 576 B, which is already 4.5 × 128 B sectors — so even
   page size 1 gives a 576-byte contiguous run per token. This is why page-size-1 sparse
   MLA is viable at all, and why page-size-1 sparse **MHA** (head dim 128 → 128 B) would
   be much worse. `[inferred]`
2. **TMA descriptor reshaping.** trtllm-gen tries to widen the TMA box to 128 B
   (`canReshapeTmaKv`, requires `headDimQk == headDimV` — false for MLA at 576 vs 512).
   `[verified]` `kernelParams.h:704-713`.

Page sizes present in the cubin metadata: `{0, 1, 16, 32, 64, 128}` (0 = contiguous,
1 = sparse-MLA). `[verified]` metadata scan.

**In SGLang our two KV pools use different page sizes** `[verified]`
(`memory_pool.py`, `dsa_indexer.py`):

| pool | page size | bytes / token / layer |
|---|---:|---:|
| Main MLA latent+rope (`kv_buffer`) | 64 (allocator) / **1 (kernel view, sparse)** | **576** (E4M3, `kv_cache_dim = 512+64`) |
| DSA indexer K (`index_k_with_scale_buffer`) | **64** (`assert self.page_size == 64`) | **132** = 128 E4M3 + 4 B FP32 scale |

`kv_cache_dim = 576` for us specifically because the trtllm backend does **not** use the
scaled layout `[verified]` (`kv_cache_configurator.py:1939-1947`):

```
// TRTLLM backend does not override kv_cache_dim for MLA kv cache
if (dsa_prefill_backend == "trtllm" || dsa_decode_backend == "trtllm") return kv_cache_dim;
```

Total KV footprint: `(576 + 132) × 78 = 55.2 KB per token` per rank (MLA KV is not
sharded by TP — one KV head). At 1M context that is 55.2 GB of the 183 GB. `[inferred]`

### 5.4 The decode roofline

Per rank, per layer, per query position, with top-k = 2048:

```
bytes  = 2048 × 576 B                                = 1.179 MB
FLOPs  = 2·(8 heads × 2048 × 576)  [QK]              = 18.87 MFLOP
       + 2·(8 heads × 2048 × 512)  [PV]              = 16.78 MFLOP
                                                     = 35.65 MFLOP
AI     = 35.65e6 / 1.179e6                           = 30.2 FLOP/byte
```

B200 ridge point for FP8 dense: `~4.5 PFLOP/s / 8.18 TB/s ≈ 550 FLOP/byte` `[inferred]`.
At AI = 30 we are at **5.5% of the compute roof** — deeply memory-bound, which is the
expected and correct regime.

Whole-step floors, per rank, per decoded position:

| context L | main attn bytes | indexer bytes (21 layers) | total | time @ 8.18 TB/s |
|---:|---:|---:|---:|---:|
| 2,048 | 78 × 2048 × 576 = 92.0 MB | 0 (skipped, L ≤ topk) | 92.0 MB | **11.3 µs** |
| 10,000 | 92.0 MB | 21 × 10⁴ × 132 = 27.7 MB | 119.7 MB | **14.6 µs** |
| 131,072 | 92.0 MB | 21 × 131072 × 132 = 363.3 MB | 455.3 MB | **55.7 µs** |
| 131,072 (dense MLA, no DSA) | 78 × 131072 × 576 = 5.89 GB | — | 5.89 GB | 720 µs |

`[inferred]` arithmetic on `[verified]` byte counts. DSA is **12.9× cheaper than dense
MLA at 128k**, and the indexer skip when `L ≤ index_topk` is real `[verified]`
(`dsa_indexer.py:550-556`, `_should_skip_logits_computation` returns
`max_kv_len <= self.index_topk`).

**Crossover:** sparse+indexer beats dense when
`2048·576 + (21/78)·L·132 < L·576` → `L > 2184`. Below ~2.2k context DSA costs more bytes
than it saves. `[inferred]`

### 5.5 The gap between the roofline and reality

From the profile (`personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`)
`[verified]`: attention family 1040 ms over 107,189 launches = **9.70 µs mean**;
`dsa_indexer` 556 ms over 90,181 launches = **6.17 µs mean**.

Sanity check on the launch counts `[inferred]`: 107,189 launches over an 8473 ms span,
at ~79 attention kernels per target forward pass + draft passes, gives ≈154 forward
passes/s; at an EAGLE-3-1-4 acceptance of ~2.4 tokens/pass that is ≈370 tok/s — which
matches the measured 365 tok/s. The launch accounting is consistent.

If `parseP1…` (578 ms) runs once per layer per pass, that is ≈101,800 launches →
**≈5.7 µs per launch against a 144 ns byte floor: ~40× off**. `[inferred]`

**That 40× is the finding.** It is not bandwidth. Candidate causes, in the order I would
test them:

1. **Fixed per-launch cost** (TMA descriptor prefetch, `tcgen05.alloc`, gather-index
   load, gmem-reduction epilogue + atomic counters) against a 2-iteration mainloop.
2. **4× redundant work for the 4 draft tokens** (§8.4).
3. **4× redundant K reads** from `headDimPerCtaV = 128` splitting (§8.5) — L2-absorbed,
   but still SM cycles and smem traffic.
4. **`MultiCtasKvMode::GmemReduction`** rather than `CgaSmemReduction`, which is
   disabled for `headDimV >= 512` (§8.5).

None of these is measurable without `ncu`. See §9.

---

## 6. MLA (Multi-head Latent Attention)

### 6.1 The representation

Standard MHA caches `K, V ∈ R^{L × H × d}`. MLA caches a single **latent** vector per
token plus a small shared rope vector:

```
c_t   ∈ R^{512}    (kv_lora_rank)      — compressed KV, shared across all heads
k^R_t ∈ R^{64}     (qk_rope_head_dim)  — RoPE'd key, shared across all heads
```

GLM-5.2's geometry, read from the checkpoint `[verified]`
(`/home/aman/code/weights/GLM-5.2-NVFP4/config.json`):

| field | value |
|---|---|
| `num_hidden_layers` | 78 |
| `num_attention_heads` | 64 |
| `kv_lora_rank` | 512 |
| `qk_rope_head_dim` | 64 |
| `qk_nope_head_dim` | 192 |
| `qk_head_dim` | 256 |
| `v_head_dim` | 256 |
| `q_lora_rank` | 2048 |
| `head_dim` | 192 |
| `index_head_dim` | 128 |
| `index_n_heads` | 32 |
| `index_topk` | **2048** |
| `index_topk_freq` | **4** |
| `index_skip_topk_offset` | **3** |
| `index_share_for_mtp_iteration` | true |
| `indexer_rope_interleave` | true |
| `num_nextn_predict_layers` | 1 |
| `first_k_dense_replace` | 3 |
| `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | 256 / 8 / 1 |

`512 + 64 = 576` is the `headDimQk` in the kernel name. `[verified]`

### 6.2 Absorb vs materialize

Two ways to run MLA:

**Materialize.** Reconstruct per-head `K_h = c W^{UK}_h` and `V_h = c W^{UV}_h`, then run
ordinary attention at head dim 192 (+64 rope) / 256. FLOPs scale with `L × H × d`, and
you have to write `L × H × 256` values. Correct for prefill where you are compute-bound
anyway and want the large-K GEMM shapes.

**Absorb.** Push `W^{UK}` into the query and `W^{UV}` into the output projection:

```
S_h = (q^C_h W^{UK}_h) · c^T   +   q^R_h · (k^R)^T
    = q̃_h · c^T + q^R_h · (k^R)^T          with q̃_h ∈ R^{512}
O_h = (P_h · c) W^{UV}_h
```

Now the *only* thing read from cache is `c` (512) and `k^R` (64) — **576 values shared
across all 64 heads** instead of 64 × 448. The attention becomes **MQA with head dim 576
for scores and 512 for values.** That is exactly `HQk576HV512`.

Why this changes decode fundamentally `[inferred]`:

| | dense MHA-ish (materialized) | absorbed MLA |
|---|---:|---:|
| bytes/token/layer (FP8) | 64 heads × (256+256) = 32,768 B | **576 B** |
| FLOPs/token/layer (8 heads/rank) | 2·8·(256+256) = 8,192 | 2·8·(576+512) = 17,408 |
| AI (FLOP/byte, per rank at TP8) | 0.25 | **30.2** |

Absorption trades ~2× more FLOPs for ~57× fewer bytes. At a ridge point of 550 FLOP/byte
both are memory-bound, so it is a near-pure 57× win on the *only* axis that matters for
decode.

The cost is on the query side: `q̃` is `64 heads × 512` per token instead of
`64 × 192`, so the Q-projection GEMM gets bigger — and GLM's `qk_nope_head_dim = 192`
makes that a non-power-of-2 K=192 split-K path that DeepSeek's 128 never hits
`[verified]` (noted in our own optimization log, `glm-5.2-optimization-log.md`).

For prefill the tradeoff inverts: with `M` large you want the `L × L` score matrix to be
computed at the smaller head dim, so materialize. The standard split is
**materialize for prefill, absorb for decode.**

### 6.3 TP behaviour

MLA has **one KV head**. It cannot be split across tensor-parallel ranks, so the latent
cache is **replicated on all 8 ranks** and only the 64 query heads are sharded (8/rank).
Consequences for us `[inferred]`:

- KV memory is 8× what a sharded scheme would use, but at 55.2 KB/token/rank that is
  affordable (1M ctx = 55.2 GB of 183 GB).
- Every rank reads the same 1.18 MB per layer → aggregate 9.4 MB/layer of HBM across the
  node for what is 1.18 MB of information.
- `numHeadsQPerKv = 8` at TP8, which is what forces `SwapsMmaAbForGeneration`
  (≤16) and `mStepQ = 8` and therefore the small MMA-N. **Head count per rank is the
  single parameter that determines which kernel family you land in.**

### 6.4 Known MLA kernel implementations

| implementation | arch | notes |
|---|---|---|
| **trtllm-gen** `fmhaSm100f…HQk576HV512…ForGen` | SM100 | what we run; §8 |
| **FlashMLA** `flash_mla_with_kvcache`, `flash_mla_sparse_fwd` | SM90/SM100 | dense + token-sparse, FP8 sparse decode `[reported]` |
| **FA4** `FlashAttentionMLAForwardSm100` | SM100 | 2-CTA, cpasync top-k gather, needs 128 q-heads/kv-head `[verified]` |
| **CUTLASS 77** MLA inference kernel | SM100 | "weight-absorbed regime", latent 512 + rope 64 `[reported]` |
| **FlashInfer** `BatchMLAPagedAttentionWrapper`, `trtllm_batch_decode_with_kv_cache_mla`, `prims_ts_batch_decode_with_kv_cache_mla` | SM100 | wrappers `[reported]` |
| **SGLang** `trtllm_mla_backend.py`, `dsa_backend.py` | — | dispatch layer, 1561 + 3692 lines `[verified]` |
| **TileRT** `flash_sparse_mla_op` | SM100 | BF16 MMA, split 64 × 32 `[verified]` |

### 6.5 Why FA4's MLA kernel is not usable for us as-is

`assert qhead_per_kvhead == 128, "require MQA 128 for DSA path"` `[verified]`
(`flash_fwd_mla_sm100.py:80`). At TP8 we have 8. This is not a soft heuristic — it is an
assert, and `cluster_tile_m = 128` is built around it (`cta_tile_m=64` × 2-CTA). To use
it we would need DeepSeek-V3.2's 128 heads at TP1-attention, or attention-DP. Attention
DP at C1 puts all the work on one rank, which is the wrong trade for latency. `[inferred]`

---

## 7. Sparse attention

### 7.1 The general shape

```
1. score:   cheap surrogate s(q, k_j) for all j ≤ t          — O(L) per query
2. select:  I = top-k_j s(q, k_j)                            — O(L) selection
3. gather:  K_I, V_I from the paged cache                    — O(k · d) traffic
4. attend:  ordinary flash attention over (q, K_I, V_I)      — O(k · d) FLOPs
```

Everything interesting is in how cheap you can make (1)–(3) relative to (4).

### 7.2 DeepSeek Sparse Attention (DSA) / the lightning indexer

`[reported]` DeepSeek-V3.2 (arXiv 2512.02556) and the vLLM blog; `[verified]` for the
SGLang implementation on this box.

- A **lightning indexer**: a narrow attention-like scorer with few heads, FP8, ReLU
  nonlinearity, producing an index logit per (query, past token). `[reported]`
- **Top-k with k = 2048** selects the positions that go to the real MLA. `[reported]`,
  and `[verified]` in our config (`index_topk: 2048`).
- Trained by **KL-divergence imitation of the dense model's head-summed attention
  distribution**, first with a frozen main model, then jointly with gradients kept
  separate from the LM loss. `[reported]`
- Claimed ~3–6× cost reduction at 128k with benchmark parity. `[reported]` — consistent
  with my 12.9× byte-count estimate in §5.4 being an upper bound that ignores the
  indexer's own compute and the top-k.

GLM-5.2's indexer, from the config and code `[verified]`:

| | value |
|---|---|
| `index_n_heads` | 32 (vs the model's 64 attention heads) |
| `index_head_dim` | 128 |
| indexer K cache | E4M3, **132 B/token/layer** (128 data + 4 B FP32 scale per 128-element block) |
| indexer K page size | **64** (`assert self.page_size == 64`) |
| scoring kernel | DeepGEMM `fp8_paged_mqa_logits` (`blocksize = page_size = 64`, "hardcoded in deep_gemm"), or a CuTe-DSL variant "tuned for the GLM-5.2 32-head path" |
| top-k kernel | `sgl_kernel.fast_topk_v2` (default `dsa_topk_backend = sgl-kernel`) |
| fused variant | `topk_transform_512_v2` — fuses top-k **and the page-table transform** in one JIT kernel, ABI requires `score_stride % 4 == 0` |
| forced positions | `num_init_tokens` (attention-sink prefix) and `num_local_tokens` (recent window) are scattered with `+inf` into the logits before top-k |

`[verified]` `dsa_indexer.py:852-1000`, `dsa_topk_backend.py:16-302`,
`paged_mqa_logits.py`.

Two implementation details worth stealing or watching:

- **The top-k output is not required to be the true top-k indices.** SGLang's
  `topk_transform` docstring says so explicitly: "the result of this topk_transform may
  not be the topk indices of the input logits" — the kernel is allowed to fuse a
  page-table translation into the output. `[verified]` `dsa_indexer.py:323-337`.
- **Index broadcast from rank 0.** `SGLANG_DSA_TOPK_BROADCAST` broadcasts the finalized
  top-k from rank 0 to all TP ranks instead of recomputing. `[verified]`
  `dsa_indexer.py:233-267`. Relevant to our rank-skew finding: this is a synchronization
  point that *adds* a collective but removes 7/8 of the indexer's redundant compute.

### 7.3 Index sharing across layers (`index_topk_freq = 4`)

SGLang's rule `[verified]` (`configs/model_config.py:190-211`):

```python
def dsa_layer_skips_topk(config, layer_id) -> bool:
    pattern = getattr(config, "index_topk_pattern", None)          # None for us
    freq   = getattr(config, "index_topk_freq", 1)                 # 4
    offset = getattr(config, "index_skip_topk_offset", None)       # 3
    if offset is not None:
        return max(layer_id - offset + 1, 0) % freq != 0
    return max(layer_id - 1, 0) % freq != 0
```

Evaluated against our config `[verified]` (computed):

```
layers that COMPUTE top-k (21 of 78):
  0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74
layers that SHARE the previous layer's top-k: 57
```

(The checkpoint also carries `indexer_types` = 78 entries of `{'full','shared'}`, an
independent encoding of the same split. `[verified]`) Adding the MTP layer —
`num_nextn_predict_layers = 1`, `index_share_for_mtp_iteration = True` — gives the
22/57 split quoted in our profile notes.

**What it costs us elsewhere:** SGLang refuses to start two-batch overlap with
`index_topk_freq > 1`, because the TBO op path does not propagate top-k indices across
the micro-batch split. `[verified]` (quoted in our ledger). That is a real, bounded
engineering gap, not an architectural one.

### 7.4 NSA (Native Sparse Attention) — the other design point

`[reported]` arXiv 2502.11089 (fetched HTML).

Three branches combined by a learned sigmoid gate:
```
o*_t = Σ_{c ∈ {cmp, slc, win}} g^c_t · Attn(q_t, K̃^c_t, Ṽ^c_t)
```
- **cmp**: attention over compressed blocks (block size `l = 32`, stride `d = 16`).
- **slc**: top-`n` selected blocks (`l' = 64`, `n = 16`, including 1 fixed initial block
  and 2 local blocks).
- **win**: sliding window `w = 512`.

The key trick: **selection scores come free from the compression branch.**
`p^cmp_t = Softmax(q_t^T K̃^cmp_t)`; when `l' = l = d` the selection scores *are* the
compression scores; otherwise they are aggregated spatially; for GQA they are summed
across heads in a group. `[reported]` Eq. 8–12.

Kernel design is **group-centric**: load all heads of a GQA group for position `t`, then
loop over the selected KV blocks. Reported: 9.0× forward / 6.0× backward at 64k training,
**up to 11.6× decode speedup at 64k**, "expected speedup approximately linear to memory
access volume reduction". `[reported]`

**DSA vs NSA, the design difference that matters for kernels:**

| | NSA | DSA (DeepSeek / GLM) |
|---|---|---|
| granularity | **blocks** (64 tokens) | **individual tokens** |
| scorer | reuse of the compression branch | separate lightning indexer |
| selection | top-16 blocks = 1024 tokens | top-2048 tokens |
| gather | contiguous 64-token runs → page-friendly | arbitrary tokens → **page size 1** |
| hardware need | ordinary paged TMA | `tile::gather4` or `cp.async` scatter-gather |

Block selection is kinder to the memory system; token selection is more expressive per
byte. Blackwell's `tile::gather4` is what makes the token-granular choice affordable.
`[inferred]`

### 7.5 The gather primitive: `cp.async.bulk.tensor.2d…tile::gather4`

From the CUDA 13.3 CCCL header on this box `[verified]`
(`__ptx/instructions/generated/cp_async_bulk_tensor_gather_scatter.h:7-50`):

```
// cp.async.bulk.tensor.2d.dst.src.tile::gather4.mbarrier::complete_tx::bytes
//     [dstMem], [tensorMap, tensorCoords], [smem_bar];
//   PTX ISA 86, SM_100
//   .dst = { .shared::cta }
//   .src = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor_tile_gather4(
  cuda::ptx::space_shared_t, cuda::ptx::space_global_t,
  void* dstMem, const void* tensorMap,
  const int32_t (&tensorCoords)[5],       // <-- FIVE coordinates
  uint64_t* smem_bar);
```

emitted as
```ptx
cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes
    [%0], [%1, {%2, %3, %4, %5, %6}], [%7];
```

**Five coordinates for a 2-D tensor**: coord 0 is the column offset, coords 1–4 are
**four independent row indices**. One instruction gathers 4 arbitrary rows into a
contiguous smem tile, with mbarrier transaction accounting. Variants exist for
`.cta_group::{1,2}` (SM_100a/f, 103a/f, 110a/f) and `.multicast::cluster`. `[verified]`

For a 128-token KV tile that is **32 gather4 instructions**. Our disassembled kernel
contains **144 static `UTMALDG.2D.GATHER4`** (§8.3) — consistent with an unrolled
multi-stage pipeline over several tiles. `[verified]` + `[inferred]`

There is no Hopper equivalent. FA4 works around its absence on paths where TMA gather is
unavailable by using `cp.async` with a per-thread index (`CpasyncGatherKVManager`), which
is why its DSA path asserts `use_cpasync_load_KV`. `[verified]`

### 7.6 Cost model for the sparse pipeline (our numbers)

Per layer, per query position, at context `L` `[inferred]` on `[verified]` byte counts:

| stage | cost | at L=10k |
|---|---|---|
| indexer logits (21/78 layers) | `L × 132 B` read, `32 × L` MACs | 1.32 MB, 320 kMAC |
| top-k of L → 2048 | `L` FP32 scores read + selection | 40 kB |
| gather + attend | `2048 × 576 B` | 1.18 MB |
| **dense MLA for comparison** | `L × 576 B` | 5.76 MB |

The indexer is **11%** of the main attention's byte traffic per computing layer at 10k,
and 0 on the other 57 layers. Measured share: 5.8% vs attention's 10.9% — a ratio of
0.53 against a byte-ratio of ~0.30, so the indexer is *relatively* more expensive than
its bytes, which is what you would expect from the top-k and the extra kernel launches.
`[inferred]`

---

## 8. Identifying `parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen`

### 8.1 The name is a suffix, and the prefix is on disk

`[verified]`. The string in our profile is 50 characters. The kernel entry symbol in the
cubin on this box is 118 characters, and the profile string is exactly its last 50:

```python
n = 'fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen'
p = 'parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen'
len(n), len(p), n.endswith(p), n.find(p)   # -> (118, 50, True, 68)
```

`n[:68] = "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenS"`.
So `"TokenS" + "parseP1"` = **`TokenSparseP1`**. The apparent word "parse" is the tail of
"**S**parse". `[verified]`

Confirmed against the ELF symbol table, not just the filename `[verified]`:

```
$ cuobjdump -symbols fmhaSm100fKernel_...SwapsAbForGen.cubin
STT_FUNC STB_GLOBAL STO_ENTRY  fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen
STT_FUNC STB_GLOBAL STO_ENTRY  ...SwapsAbForGenGetSmemSize
$ cuobjdump -elf ...
64-bit ELF: type=ET_EXEC, ABI=8, sm=100, toolkit=12.9
```
File: `/home/aman/.cache/flashinfer/cubins/158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/`.

**Ambiguity, stated honestly:** the `HVPerCta128` and `HVPerCta256` variants are 129
characters and their last 50 characters are *the same string*. The profile label
therefore aggregates all three `headDimPerCtaV` variants, and cannot by itself tell us
which one ran. §8.5 argues from the selection code that it is **128**.

### 8.2 Every field, decoded

The metadata row `[verified]` (`flashInferMetaInfo.h:9739`):

```c
{ DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16,
  8, 128, 8, 128, 512, 576, 512, kSM_100f, nullptr, 0,
  "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
  166688, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false, false, 1, false, false, false,
  "54ef64241e7f37e69b56cea37d4de5a79468cfbf62ac4bf87fd2b5c06fb6266a" }
```

| field | value | meaning |
|---|---|---|
| `mDataTypeQ/Kv/K/V` | `E4M3` | FP8 Q, K, V |
| `mDataTypeO` | `BF16` | BF16 output |
| `mTileSizeQ`, `mStepQ` | **8, 8** | 8 rows of Q per CTA = 8 grouped heads at TP8 |
| `mTileSizeKv`, `mStepKv` | **128, 128** | 128 keys per mainloop iteration |
| `mHeadDimPerCtaV` | 512 (this row) | full V head dim in one CTA |
| `mHeadDimQk` / `mHeadDimV` | **576 / 512** | absorbed MLA: 512 latent + 64 rope / 512 latent |
| `mSM` | `kSM_100f` | SM100 **family** target (`sm_100f`) |
| `mSharedMemBytes` | **166,688** | 162.8 KiB of the 227 KiB opt-in limit |
| `mThreadsPerCTA` | **512** | 16 warps |
| `mQkvLayout` | 2 | `PagedKv` |
| `mNumTokensPerPage` | **1** | the `P1` — page size 1, forced for all sparse-MLA kernels |
| `mMaskType` | 0 | dense (no causal mask; the top-k already encodes causality) |
| `mKernelType` | **2** | `SwapsMmaAbForGeneration` |
| `mTileScheduler` | 0 | `Static` (non-persistent) |
| `mMultiCtasKvMode` | **1** | `GmemReduction` |
| `mNumEltsPerSageAttnBlk{Q,K,P,V}` | 0,0,0,0 | SageAttention quantization off |
| `mGroupsHeadsQ` | **true** | heads folded into the Q tile |
| `mGroupsTokensHeadsQ` | **false** | tokens **not** folded — see §8.4 |
| `mReuseSmemKForV` | false | separate K and V smem (they are the same buffer logically but 576≠512) |
| `m2CtaMma` | false | single-CTA MMA |
| `mSparseAttn` | **1** | `TrtllmGenSparseMlaType::StaticTokenSparse` |
| `mSkipsSoftmaxWhenPossible` | false | full softmax |

Page size 1 is not a heuristic — it is forced `[verified]` (`fmhaKernels.cuh:397-400`):

```cpp
if (isSparseMla(params.mSparseMlaType)) {
  // SparseMla kernels use a fixed numTokensPerPage = 1.
  selectKernelParams.mNumTokensPerPage = 1;
}
```

### 8.3 What the SASS says

`cuobjdump -sass` of the cubin, 10,222 lines. Static opcode histogram of the
interesting families `[verified]`:

| opcode | count | reading |
|---|---:|---|
| `UTMALDG.2D.GATHER4` | **144** | `cp.async.bulk.tensor.2d…tile::gather4` — the sparse KV gather `[inferred, high confidence]` |
| `UTMALDG.4D` | 5 | ordinary TMA loads (Q, and the descriptor-prefetch path) |
| `UTCQMMA` | **68** | `tcgen05.mma` `[inferred, high confidence]` — "U"niform "T"ensor"C"ore MMA |
| `UTCBAR` | 25 | tcgen05 commit/barrier |
| `LDTM.16dp256bit`, `LDTM.x4` | 20 + 2 | `tcgen05.ld` (16 lanes × 256 bit) |
| `STTM.16dp256bit`, `STTM.x4` | 8 + 2 | `tcgen05.st` |
| `MUFU.EX2` | **21** | hardware exp2 — **no polynomial emulation in this kernel** |
| `MUFU.RCP` | 14 | final `1/ℓ` |
| `FMUL2` / `FFMA2` / `FADD2` | 54 / 21 / 11 | Blackwell packed FP32×2 (`.F32x2.HI_LO`) |
| `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | (present) | FP32 → E4M3 packing of `P` |
| `F2FP.SATFINITE.BF16.F32.PACK_AB` | 20 | FP32 → BF16 output |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | 139 | mbarrier try-wait |
| `NANOSLEEP.SYNCS` | 59 | backoff in the wait loops |
| `UTCATOMSWS.FIND_AND_SET.ALIGN` | 2 | TMEM allocator |
| `LDGSTS.E.BYPASS.LTC128B.128` | 4 | `cp.async` 128 B bypassing L1 |
| `ELECT` | 21 | `elect.sync` single-thread election for TMA/MMA issue |

The softmax inner block, verbatim `[verified]` (offsets `0xa280`–`0xa440`):

```
/*a280*/  FFMA2 R26, R26.F32x2.HI_LO, R23.F32, 8.8073549270629882812 ;   // += log2(448)
/*a2a0*/  FFMA.FTZ R8, R8, R18, R26 ;
/*a2c0*/  FFMA2 R10, R18.F32, R10.F32x2.HI_LO, R26.F32x2.HI_LO ;
/*a2f0*/  MUFU.EX2 R28, R8 ;
/*a310*/  MUFU.EX2 R29, R9 ;
/*a330*/  MUFU.EX2 R34, R10 ;
...       (8 consecutive MUFU.EX2)
/*a400*/  F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R4, R35, R34, RZ ;
/*a420*/  F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R4, R29, R28, R4 ;
```

Three things fall out:
1. **The FP8 `P` scale is `448` and is applied in the exponent**, exactly as described in
   §3.6. `log2(448) = 8.80735492...` matches the immediate to all printed digits.
2. **Hardware `MUFU.EX2`, not FA4-style emulation.** No `add.rm` + `shl 23` Cody-Waite
   pattern in the softmax region. For a 2048-key decode that is the right call.
3. **The FP32×2 packed datapath is used** for the pre-exp affine (`FFMA2 … .F32x2.HI_LO`),
   halving instruction count on the scale-and-shift.

Static counts are not dynamic counts. The dynamic split needs `ncu`.

### 8.4 The speculative-decoding penalty (a real, actionable finding)

`[verified]` `fmhaKernels.cuh:424-438`:

```cpp
int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;
if (params.mMaxSeqLenQ > 1 && !isContextKernel(params.mKernelType)) {
  // Each CTA handles one tokenQ by default for spec-decoding generation kernel.
  if (!kernelMeta.mGroupsTokensHeadsQ) {
    numCtasPerSeqQ = params.mMaxSeqLenQ;                       // <-- 4 for EAGLE 3-1-4
  } else {
    int numTokensPerCtaQ = std::max(1, kernelMeta.mStepQ / params.mNumHeadsQPerKv);
    numCtasPerSeqQ = ceil_div(params.mMaxSeqLenQ, numTokensPerCtaQ);
  }
}
```
and
```cpp
int numHeadsPerCta = kernelMeta.mGroupsHeadsQ
                   ? std::min(params.mNumHeadsQPerKv, kernelMeta.mStepQ) : 1;
```

A full scan of the 13,452 metadata rows `[verified]` shows:

- `mGroupsTokensHeadsQ == true` on **6,984 rows** — but *none* of them is a
  `HQk576HV512` sparse-MLA kernel. All 94 sparse-MLA rows have it **false**.
- Sparse-MLA `mStepQ` values present: **8, 16, 32, 64**. `mStepQ = 64` rows are the
  `KeepsAb` family (numHeadsQPerKv > 16).

So at TP8 with `numHeadsQPerKv = 8`:
- `numHeadsPerCta = min(8, mStepQ)` = 8 no matter which cubin we pick. The `Q16` and
  `Q32` sparse cubins are **unreachable** — they would only be selected for 16 or 32
  query heads per KV head.
- Each of the 4 EAGLE draft tokens gets its **own CTA row**, its own top-k index load,
  its own gather, its own softmax. With `index_share_for_mtp_iteration = True` the four
  index vectors are *identical* `[verified]` (config), so this is 4× redundant gather
  and 4× redundant MMA against 1× of information.

HBM traffic is probably fine (1.18 MB fits trivially in 126 MB of L2), but the
instruction count, TMA issue, and smem traffic are 4×. `[inferred]`

**The ask this generates:** a trtllm-gen sparse-MLA cubin with
`mGroupsTokensHeadsQ = true` and `mStepQ = 32` would let one CTA cover
`4 draft tokens × 8 heads`. The selection code already handles it
(`numTokensPerCtaQ = mStepQ / numHeadsQPerKv = 32/8 = 4`). The cubin does not exist.
TileRT's kernel *does* take all 4 tokens in one call (§4.7) — that is a concrete
structural advantage they have over the stock trtllm-gen path. `[inferred]`

### 8.5 The launch grid, derived

`[verified]` selection code; `[inferred]` arithmetic. Inputs at C1 decode with EAGLE
3-1-4: `mBatchSize = 1`, `mMaxSeqLenQ = 4`, `mNumHeadsQ = mNumHeadsQPerKv = 8`,
`mHeadDimQk = 576`, `mHeadDimV = 512`, `mSparseMlaTopK = 2048`,
`mMultiProcessorCount = 148`.

First, a mode flag fires `[verified]` (`fmhaKernels.cuh:418-419`):

```cpp
bool isDsv3MinLatencyMode = params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 &&
                            params.mMaxSeqLenQ <= 16 && params.mHeadDimQk == 576 &&
                            params.mHeadDimV == 512;
```
→ **true for us.** It sets `corrFactor = 1` in the head-dim split heuristic.

Pass 1 (kernel starts at `mHeadDimPerCtaV = 512`):
```
numCtasPerSeqQ    = mMaxSeqLenQ                    = 4          (mGroupsTokensHeadsQ=false)
numHeadsPerCta    = min(8, 8)                      = 8
numCtasForAllHeadsQ = 8/8                          = 1
numCtasPerHeadDim = 512/512                        = 1
numCtasY = 1, numCtasZ = 1, numCtasX = 4
maxAttentionWindow  = min(seqLenKv, 2048)          = 2048
maxNumCtasPerSeqKv  = ceil(2048 / (2*128))         = 8
numCtasPerSeqKv     = min(8, max(1, 148/(4*1*1)))  = 8
numCtasX = 4*8 = 32,  totalNumCtas = 32
```
CGA reduction is then **skipped**, by an explicit guard `[verified]`
(`fmhaKernels.cuh:519-529`), whose comment is worth quoting in full:

> "headDimV >= 512 is excluded: the current trtllm-gen cubin ships no SwapsMmaAb
> CgaSmemReduction kernels at headDimV >= 512 (covers both MLA headDimQk=576/V=512 and
> non-MLA H=512), and for tileSizeQ >= 32 the CGA variant also exceeds the device smem
> limit."

Then the head-dim split fires `[verified]` (`fmhaKernels.cuh:555-570`):
```
totalNumCtas * corrFactor = 32 * 1 = 32 <= 148            → split
mHeadDimPerCtaV = (32 * 2 * 1 = 64 <= 148) ? 128 : 256    → 128
mSelectNewKernel = true
```

Pass 2 (`mHeadDimPerCtaV = 128`):
```
numCtasPerHeadDim = 512/128 = 4  →  numCtasY = 1*4 = 4
numCtasPerSeqKv   = min(8, max(1, 148/(4*4*1))) = min(8, 9) = 8
numCtasX = 4*8 = 32
totalNumCtas = 32 * 4 * 1 = 128 CTAs on 148 SMs
```

**Conclusion `[inferred]`:** the kernel that actually ran is almost certainly the
`HVPerCta128` variant, launching **128 CTAs**, each responsible for
`1 draft token × 8 heads × 256 of the 2048 selected keys (2 tiles) × 128 of the 512
output dims`, with a global-memory + atomic-counter reduction across the 8 KV CTAs and a
concatenation across the 4 head-dim CTAs.

Redundancy in that decomposition `[inferred]`:
- The 4 head-dim CTAs sharing a KV slice each read the **full 576-wide K** for `QK^T`
  (only `V` is split) → 4× K re-read.
- The 4 draft-token CTAs read the same gathered KV → another 4×.
- Net: ~16× L2 traffic amplification over the 1.18 MB of unique HBM bytes, i.e. ~18.9 MB
  of L2 reads per layer. L2 can absorb it; the SM-side smem and TMA issue cannot be
  wished away.

### 8.6 Confidence summary

| claim | confidence |
|---|---|
| It is the trtllm-gen SM100 sparse-MLA generation kernel, FP8 in / BF16 out, 576/512, page 1 | **certain** — symbol table + metadata row on this box |
| The workload reaches it via `dsa_decode_backend = trtllm` → FlashInfer → cubin | **certain** — resolved server args in our own log |
| `headDimPerCtaV = 128`, 128 CTAs | **high** — derived from the shipped selection code, not measured |
| ~5.7 µs/launch, ~40× off the byte roofline | **medium** — launch count is inferred, not read from the trace |
| The 4 draft tokens cause 4× redundant gather/MMA | **high** — `mGroupsTokensHeadsQ=false` verified, code path verified |

---

## 9. What to do and measure next on this box

Ordered by expected value, with the measurement that would settle each.

1. **`ncu` the kernel.** Nothing above about *why* it is 40× off roofline is measured.
   Wanted counters: `dram__bytes.sum`, `lts__t_bytes.sum` (to confirm the 16× L2
   amplification), `sm__cycles_active.avg` vs `gpu__time_duration`, `smsp__inst_executed`
   split by warp, and the launch overhead (`gpc__cycles_elapsed` minus mainloop).
   Until then no headroom claim here is defensible.
2. **Count `parseP1…` launches directly from the trace** (`trace.sqlite`) instead of
   inferring them. That converts the 5.7 µs from `[inferred]` to `[verified]` and fixes
   the roofline ratio.
3. **Test whether the 4 draft tokens are actually 4 separate CTA rows** by launching with
   `mMaxSeqLenQ = 1` (MTP off) and comparing per-launch time. If per-launch time is
   ~unchanged, the kernel is prologue-bound and §8.4 is not the lever. If it drops ~4×,
   it is.
4. **Force `headDimPerCtaV = 512`** (single CTA per output) and compare. The 4-way split
   costs 4× K re-read to buy 4× CTAs; at 32 vs 128 CTAs on 148 SMs both are plausible and
   the heuristic (`corrFactor = 1` under `isDsv3MinLatencyMode`) is admittedly a guess in
   NVIDIA's own comment ("TODO: find better heuristic of splitting headDimV across
   multiple CTAs").
5. **Try `TRTLLM_GEN_ENABLE_TILE_SIZE_KV64`.** `[verified]` env var
   (`flashinfer/trtllm/common.h:200-201`). **Caveat: it is gated on
   `!isSparseMla(...)`, so it will do nothing for us.** Listed so nobody spends a day on
   it. `[verified]` `fmhaKernels.cuh:508-515`.
6. **Ask for / build a `GroupsTokensHeadsQ` sparse-MLA cubin at `stepQ = 32`.** The host
   selection logic already supports it; only the cubin is missing. This is the single
   change that would make the EAGLE path 1× instead of 4× (§8.4).
7. **Benchmark TileRT's `flash_sparse_mla_op` in isolation** against the trtllm-gen
   kernel at our shape (batch 1, 4 tokens, 8 heads, topk 2048). Both are installed on
   this box. Their kernel takes all 4 draft tokens in one call and uses BF16 MMA; ours
   uses FP8 MMA and 4 calls. A head-to-head is a 30-minute experiment and would tell us
   how much of their 500 vs our 365 tok/s lives in attention.
8. **Measure achieved HBM bandwidth on an idle GPU** to replace the computed 8.18 TB/s
   with a real number (STREAM-style or `nvbandwidth`). The device was at 165 GB used
   during this investigation, so I did not run it.
9. **Re-derive all of this at C64.** Everything above is a C1 story. At C64 the batch
   dimension supplies parallelism, `numCtasPerSeqKv` collapses toward 1, the selection
   code switches `mTileScheduler` to `Persistent` and disables `MultiCtasKvMode`
   `[verified]` (`fmhaKernels.cuh:503-508`), and a completely different kernel runs.

**One thing that is settled and needs no measurement:** the profile label
`parseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen` should be rewritten in our tooling
as `trtllm-gen sparse-MLA decode (FP8, 576/512, page-1, KV-split)`. "parse" is a
truncation artifact and will mislead the next person who reads the table.

---

## Open questions I could not close

- **B200 FP8 dense peak TFLOP/s.** I used ~4.5 PFLOP/s as 2× the 2.25 PFLOP/s BF16 figure
  the FA4 paper quotes `[reported]`. I did not find a primary NVIDIA datasheet in this
  session. The ridge-point conclusion (deeply memory-bound) is robust to a 2× error here.
- **cuDNN's SM100 attention internals.** Only its role as an FA4 baseline is sourced.
- **XQA's SM100 kernel structure.** Named in FlashInfer's API; no architecture source found.
- **FlashMLA sparse decode performance on B200.** Its published numbers are H800.
- **Whether `MUFU.EX2` throughput is genuinely 16/clk/SM on B200.** This is the FA4
  paper's number `[reported]`; I did not find it in an NVIDIA document and did not
  microbenchmark it.
- **The exact truncation mechanism** that produced a 50-character kernel label in our
  profile (nsys vs our classifier). Cosmetic, but it caused the "parse" confusion.

---

## Sources

Read directly on this box:

- `/home/aman/code/weights/GLM-5.2-NVFP4/config.json` — model geometry, DSA config
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/hotspots-and-optimization-ledger.md`
- `/home/aman/code/NotSglang/personal_docs/glm-5.2/glm-5.2-optimization-log.md`
- `/home/aman/code/NotSglang/python/sglang/srt/configs/model_config.py` (`dsa_layer_skips_topk`)
- `/home/aman/code/NotSglang/python/sglang/srt/layers/attention/dsa_backend.py`
- `/home/aman/code/NotSglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- `/home/aman/code/NotSglang/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`
- `/home/aman/code/NotSglang/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py`
- `/home/aman/code/NotSglang/python/sglang/srt/mem_cache/memory_pool.py` (`DSATokenToKVPool`)
- `/home/aman/code/NotSglang/python/sglang/srt/mem_cache/kv_cache_configurator.py` (`calculate_mla_kv_cache_dim`)
- `/home/aman/code/NotSglang/k3-kernels/src/attn/mla_decode.cu` (our own CUDA-core baseline)
- `/home/aman/code/third_party/TileRT/README.md`
- `/home/aman/code/third_party/TileRT/tilert/models/deepseek_v3_2/ops/flash_sparse_mla.py`
- `/home/aman/code/NotSglang/.venv/lib/python3.12/site-packages/flash_attn/cute/softmax.py`
- `…/flash_attn/cute/utils.py` (`POLY_EX2`, `ex2_emulation`, `combine_int_frac_ex2`)
- `…/flash_attn/cute/flash_fwd_sm100.py` (warp-role assignment)
- `…/flash_attn/cute/flash_fwd_mla_sm100.py` (MLA SM100, TMEM budget, MMA tilers)
- `…/flash_attn/cute/topk_gather_kv.py` (`CpasyncGatherKVManager`)
- `/home/aman/code/cuda-13.3/nvidia/cu13/include/cccl/cuda/__ptx/instructions/generated/tcgen05_alloc.h`
- `…/generated/tcgen05_ld.h`, `…/generated/tcgen05_st.h`, `…/generated/tcgen05_mma.h`
- `…/generated/cp_async_bulk_tensor_gather_scatter.h` (`tile::gather4`)
- `/home/aman/.cache/flashinfer/cubins/158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/include/flashInferMetaInfo.h` (13,452 kernel rows)
- `…/fmha/trtllm-gen/fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen.cubin` (disassembled)
- `/home/aman/.cache/uv/archive-v0/Ot08FUXlQNXgIElb/flashinfer/data/include/flashinfer/trtllm/fmha/fmhaKernels.cuh` (kernel selection + grid computation)
- `…/flashinfer/trtllm/fmha/fmhaRunnerParams.h` (all enums)
- `…/flashinfer/trtllm/fmha/kernelParams.h` (TMA descriptor construction)
- `…/flashinfer/trtllm/common.h` (`TRTLLM_GEN_ENABLE_TILE_SIZE_KV64`)
- `nvidia-smi`, `nvidia-smi -q`, `torch.cuda.get_device_properties(0)` on this node
- `cuobjdump -symbols / -elf / -sass` (from `triton/backends/nvidia/bin`)

Fetched from the web:

- https://arxiv.org/html/2603.05451v1 — FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling
- https://arxiv.org/abs/2512.02556 — DeepSeek-V3.2 (abstract; DSA)
- https://arxiv.org/html/2502.11089v2 — Native Sparse Attention (NSA)
- https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/README.md
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/77_blackwell_fmha/README.md
- https://docs.flashinfer.ai/api/attention.html
- https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (TOC only; the tcgen05 and `cp.async.bulk.tensor` sections were not reachable through the fetch — the local CCCL headers were used instead)

Search-result pages consulted but not used as primary evidence:
https://deepwiki.com/Dao-AILab/flash-attention/4-blackwell-(sm100sm120)-architecture,
https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet,
https://vllm.ai/blog/2025-09-29-deepseek-v3-2,
https://docs.nvidia.com/cutlass/4.3.0/CHANGELOG.html,
https://github.com/NVIDIA/TensorRT-LLM/issues/11799.
