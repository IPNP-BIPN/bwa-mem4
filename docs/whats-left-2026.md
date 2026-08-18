# What is left to make this faster, at 2026, under byte-identity

Surveyed 2026-08-18, after the x86 profile. It updates
[`state-of-the-art.md`](state-of-the-art.md) (2026-08-09) rather than repeating it: that file
surveys the TOOLS, this one asks what an ALGORITHM or an implementation could still buy, and answers
mostly from our own measurements, because that is where the answer turned out to be.

The constraint has not moved: SAM output byte-identical to bwa-mem2 2.3. It rules out most of what
the literature offers, and the interesting question is what survives it.

## What 2026 published, and what of it survives the constraint

| work | what it is | verdict here |
|---|---|---|
| **QuadRank** (Groot Koerkamp, 2026) | rank at 2.29 bits/base, all four symbols per query | Evaluated in depth on 2026-08-09 and stopped on a measurement: growing `cp_occ` 32x costs 2.4 ns of 15.4 per query, so the whole footprint prize caps near 6% end-to-end, for 1652 lines of AVX2 with no NEON path. Unchanged. |
| **minibwa** (Li & Homer, 2026) | batched SMEM with prefetch, batched locate, 10-mer ds-interval cache, q-mer rescue prefilter | The two prefetch ideas are **already here** (`prefetch_occ` fetches both interval ends; `get_sa_batch` is a 128-wide lockstep). The 10-mer cache was tried here and measured zero. The q-mer prefilter and the seed-order heuristics change output. |
| **ropebwt3 / BML** (Li) | Boyer-Moore-like skipping for MEM finding, bidirectional index | Aimed at LONG MEMs in pangenomes. At `min_seed_len` 19 on 150 bp reads there is almost nothing to skip, and our SMEM set must stay bwa's exactly. |
| **KeBaB** (2025) | k-mer breaking to skip regions that cannot hold a long MEM | Same reason: a filter whose value grows with the MEM length threshold. |
| **Singletrack** (2025) | one DP matrix for gap-affine BACKTRACE, 1.2-2.1x over KSW2/WFA2 | Our extension kernel stores no traceback at all: it returns scores and coordinates. It would apply only to the CIGAR-producing global alignment, which is a small part of `sam_emit`. Worth remembering, not worth doing first. |
| **WFA / BiWFA** | O(ns) instead of O(nm) | Already recorded in `optimization-roadmap.md` as a gated spike: the blocker is not speed, it is that WFA's tie-breaks are not bwa's, and `score2`/`te2`/`gscore`/`max_off` are bwa bookkeeping a wavefront does not produce. |
| **Sassy / Sassy2** (2026) | SIMD fuzzy DNA search, batches of patterns | A different query shape (search patterns in text), not SMEM seeding against an FM index. |
| **LexicMap** (2025) | alignment against millions of prokaryotic genomes | Different problem entirely. |

**Conclusion of the survey: nothing published in 2026 beats what we already measured.** The two
ideas that would matter most (QuadRank's smaller rank structure, WFA's cell count) are both already
on file with the number or the blocker that stops them.

## Where our own numbers say the remaining upside is

The x86 profile changed the picture more than the literature did.

### 1. The build, not the code (largest, and packaging-only)

fg-labs/bwa-mem3 compiles its whole non-kernel codebase at **AVX2** (`BASELINE_ARCH ?= avx2`) and
four kernel TUs at five ISA tiers with runtime dispatch. We compile everything at the x86-64
baseline (SSE2) and dispatch only hand-written kernels. Full teardown in
[`fork-teardown.md`](fork-teardown.md).

Our x86 profile has the matching fingerprint: per read pair against the same profile on an M4, the
vector kernels are 2.57x and the rescue kernel 4.56 against 10.03 Gcell/s, which is two cores; but
`sam_emit` is 5.15x, `chain_flt` 4.3x and the SA-walk remainder 3.5x, which is two compilers. On
ARM the same choice costs nothing, because the aarch64 baseline already includes NEON (measured:
`target-cpu=native` 36.35 s against baseline 36.48 s, a tie).

### 2. Only 40% of the rescue kernel's own throughput reaches the pipeline on x86

| | kernel micro-probe | in production | delivered |
|---|---|---|---|
| Zen 3, AVX2 u8 | 11.27 Gcell/s | 4.56 | **40%** |
| M4 Max, NEON u8 | ~16 Gcell/s | 10.03 | **63%** |

Same code, same probe, two platforms: the glue around the kernel (job collection, the `finish_row`
and `extract_group` scalar passes, the memory traffic) eats a third more of the budget on x86 than
on ARM. #46 tuned those scalar passes on NEON only. This is the same root cause as (1) seen from the
other end, and it is worth about 8% of the x86 wall on its own.

### 3. The extension kernel costs 2.0 operations per cell, the rescue kernel 0.95

Measured on the same Zen 3 runner, same day: rescue **11.27 Gcell/s**, batched extension **2.29**.
A factor of five per cell, on the kernel that is **half of `align`'s CPU** (275 s of 545 s), where
`align` is 64% of the wall. #48 tried three levers on it and measured +5.9% on SSE4.1 and nil on
AVX2, so the constant factor is not going to fall to tuning; what has never been tried is the
structural change, and the dead-ends file says why row pairing is fatal HERE (band retightening)
while being fine in the rescue kernel.

This is the single largest compute lever left in the aligner, and it is also the hardest: it is the
one kernel where byte-identity constrains the recurrence itself.

## Ranked, what to do

1. **Ship per-ISA x86 artefacts** once the baseline/v2/v3 arm reports. Packaging, no algorithm risk,
   and it addresses (1) and part of (2).
2. **Port #46's scalar-pass work to the x86 kernels** (vector-gated `finish_row`, vector `minsc`
   pre-filter in `extract_group`). It was measured on NEON and never applied to AVX2/SSE4.1, and (2)
   is exactly the shape of a gap that leaves.
3. **The SAM emitters**, continued: the integer and SEQ/QUAL work landed today; the CIGAR and tag
   builders have not been audited.
4. **A sort-free adjacent-SMEM dedup**, from the fork's `smem_dedup.cpp`: 34 lines, no allocation,
   and it attacks the 86.5M SA lookups per million pairs directly. Byte-identity is the open
   question, and it is a measurement.
5. **The extension kernel's 2.0 ops/cell**, as a spike with a real design, not as tuning.

Everything else surveyed this year is either already measured here or changes the output.
