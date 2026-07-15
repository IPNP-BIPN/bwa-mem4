# CPU optimization roadmap (bwa-mem3-rs)

Synthesis of a deep-research pass (2026-07) into CPU-optimization state of the art for a
byte-identical bwa-mem2 reimplementation. Grounded in our profile: **Smith-Waterman extension is
~85% of cycles**, seeding is already lockstep+prefetch batched, NEON 128-bit/16-lane is the width
ceiling on Apple Silicon (SME/SVE rejected: max-plus DP is not multiply-accumulate), and the
learned index (LISA/BWA-MEME) was measured ~5x slower here (see `lisa-learned-index-dead-end`).

## The reframing

Per-cell *cost* is already at the ceiling: inter-sequence 16-lane u8 / 8-lane i16 (SWIPE-style) is
the optimal SIMD layout for batched short reads. Striped/Farrar/lazy-F/prefix-scan solve an
intra-sequence dependency we don't have; difference-recurrence (KSW2) is intra-sequence; Block
Aligner / adaptive banding are heuristic (break byte-identity). So the wins are **not "faster
cells"** — they are: fill the idle pipes, compute fewer cells, skip full DP.

## Prioritized levers

| # | Lever | Mechanism | Gain | Byte-identity | Effort |
|---|---|---|---|---|---|
| 1 | **Fill the 4 NEON pipes** | The `for j` inner loop carries `f_v`/`h1_v` — a serial chain (`f_v→h_v→f_new`, ~3-4 dependent ops, each 2-3cy latency). M-series retires 4 vec ops/cy across 4 pipes → we run at ~25% of peak. Interleave 3-4 **independent alignment groups'** row-sweeps so the scheduler has ready uops. | ~3-4x on the kernel | Identical (same math, reordered) | Medium-high (restructure `batched.rs` macro) |
| 2 | **Difference recurrence** (Suzuki-Kasahara / KSW2) | Store adjacent score *differences*, not absolutes: critical path ~8→4 ops/cell, differences bounded so int8 is unconditional. Orthogonal to layout; stacks with #1. | ~2.1x reported | Score exact; verify CIGAR tie-break | Medium-high |
| 3 | **Ungapped-first / skip-DP** (minibwa q-mer prefilter) | Most 150bp extensions are gapless. Cheap ungapped score first; full gap-affine DP only when a gap could improve. minibwa: q=7, run SW only if max q-mer count ≥ 10. | Removes a slice of the 85% | Exact only if the skip provably can't change the winner (else a validated divergence) | Low-medium |
| 4 | **WFA for hard extensions** (hybrid) | O(n·s): ~10x fewer cells for low-error reads (2-6x vs KSW2). Keep SIMD banded DP for tiny bands, dispatch WFA only for wide bands. | High for the hard subset | Risky: WFA's M>X>D>I CIGAR tie-break ≠ bwa's; needs canonicalization | High (spike) |
| 5 | **Micro-opts** | `vmax` over `cmp+bsl`; plain `vadd` where diffs permit (vs `vqadd` lat 3); horizontal reductions out of the inner loop; register-tile the band, cut `eh_h`/`eh_e` traffic; LTO + `target-cpu=apple-m1` + PGO (escape-branch layout). | 5-15% each | Identical | Low |

**Redundant-work audit (free, identity-required):** confirm contained-seed drop, chain filtering,
Z-drop cell, and band width match bwa-mem2 exactly — any deviation is both a correctness bug and
wasted SW. Highest ROI, zero risk.

## Do NOT
- Switch to striped/lazy-F/prefix-scan (wrong toolbox for inter-sequence).
- Adopt Block Aligner / adaptive banding / strobemer seeding (heuristic → breaks byte-identity).
- Chase AMX/SME/wider lanes (MAC-only, no max-plus; 16-lane NEON is the hard ceiling).
- Resurrect LISA on Apple Silicon.
- SIMD-vectorize short-read chaining (negligible `n`, breaks identity) or ERT (60GB RAM, redundant
  with our lockstep hiding).

## Recommended sequence
1. **Spike lever #1** — biggest, precision-preserving win; the kernel is latency-bound with 3 of 4
   NEON pipes idle. Measure pipe occupancy (M-series `INST_SIMD` / top-down) before/after.
2. Stack **#2** (difference recurrence) to shorten the chain and lock int8.
3. Land **#3** (ungapped skip) — independent, high value on the 85%.
4. **#4 (WFA)** only as a gated spike once #1-3 land and CIGAR byte-identity is proven.

## Closest analog
**minibwa** (Heng Li, 2026): same lineage, same NEON target, ~2.5x over bwa-mem2 — its wins are the
skip-DP / prefilter / prefetch class (byte-safe-friendly), not a magic SW kernel. Confirms the
above. Strobealign/Accel-Align win by *fuzzy seeding* (fewer SW calls) but change output.

Sources: Suzuki-Kasahara 2018; Rognes SWIPE 2011; Marco-Sola WFA 2021 / BiWFA 2023; Dougall Johnson
M1 Firestorm SIMD tables; Hajime Suzuki M1 notes; minibwa 2026; ERT ISCA 2021; strobealign 2022;
Accel-Align 2021; mm2-fast 2022.
