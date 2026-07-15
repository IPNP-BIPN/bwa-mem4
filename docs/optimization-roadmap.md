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
| 1 | ~~**Fill the 4 NEON pipes**~~ **(TESTED — no gain on NEON, register-bound)** | The `for j` inner loop carries `f_v`/`h1_v` — a serial chain (`f_v→h_v→f_new`, ~3-4 dependent ops, each 2-3cy latency). M-series retires 4 vec ops/cy across 4 pipes → we run at ~25% of peak. Interleave 3-4 **independent alignment groups'** row-sweeps so the scheduler has ready uops. | **Predicted ~3-4x; measured ~0%** | Identical (proven, gated) | Medium-high (restructure `batched.rs` macro) |

### Lever #1 result (2026-07): measured, negative — abandoned on NEON

Built and validated a 2-group ILP-interleaved u8 kernel (`define_sw_kernel_ilp!`, byte-identical to
scalar across 400 randomized rounds straddling the group boundary). **It gave no speedup.**

- **Latency headroom is real.** Breaking the `f_v` loop-carried dependency (same op count, disjoint
  probe) sped the single-group kernel up **~1.55x** (batch 16: 77.9 → 50.7 ms). So the recurrence *is*
  latency-bound with ~3 of 4 NEON pipes idle, exactly as predicted.
- **But it can't be captured on NEON.** Interleaving a second independent group needs a second copy of
  the live vector state. Register accounting: **2 groups × 9 live vectors (f/h1/rowmax/mj + beg/end/
  active/t/t_is_n) + 9 loop-invariant constants + `j_v` = 28 live before ~15 temporaries**, vs only
  **32 NEON registers**. Disassembly of the x2 kernel: **102 vector spills** in the function. The spill
  traffic exactly cancels the ILP.
- **Measured (uniform lengths, both groups equal work, batch 32/64):** single-group 92.4 / 93.8 ms vs
  x2 interleave 91.5 / 91.6 ms → **parity (<1%)**. With random lengths, group-length mismatch makes x2
  slightly *worse*.

Conclusion: the SW kernel is at the practical byte-identical ceiling on Apple Silicon — 16-lane NEON
saturates the register file, and the only ILP source (a 2nd independent alignment group) doesn't fit.
This confirms the systems-agent caveat and the `apple-silicon-simd-width-ceiling` finding. The x2 code
was reverted (not shipped); a native-NEON byte-identity gate (`neon_verify`) was kept.

**x86 caveat:** AVX2 has only **16** vector registers (worse — would spill harder), but **AVX-512 has
32 zmm**. A 2-group interleave *might* pay off on AVX-512 (more registers + the recurrence is the same
latency-bound shape). Untestable on Apple; revisit only when the Intel/AVX-512 path gets CI. The
`define_sw_kernel_ilp!` approach (in git history on `feat/sw-kernel-ilp`) is the starting point.
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

## Recommended sequence (revised after lever #1 tested negative)
1. ~~Spike lever #1~~ — **done, no gain on NEON (register-bound, see above).** The latency headroom
   exists but the register file can't hold a 2nd interleaved group. Do not retry on NEON/AVX2; only
   reconsider on AVX-512 (32 zmm) when the Intel path gets CI.
2. **#2 (difference recurrence, Suzuki-Kasahara)** is now the only remaining kernel-side lever that
   attacks the *same* latency chain without adding registers: it *shortens* the critical path (~8→4
   ops/cell) rather than parallelizing it, so it sidesteps the register wall. Highest-value next spike,
   but verify the CIGAR tie-break stays byte-identical.
3. **#3 (ungapped skip)** — independent of the kernel, removes a slice of the 85% outright. Low-risk;
   pairs with the redundant-work audit. Do this regardless.
4. **Safe, identity-preserving wins that don't need a kernel breakthrough** (land these now): PGO/BOLT
   (`cargo-pgo`, ~10-15% on the driver's branchy dispatch), kill the per-batch `regs_all[i].clone()`
   allocation, per-thread scratch reuse. These are the realistic near-term gains.
5. **#4 (WFA)** only as a gated spike once #2-3 land and CIGAR byte-identity is proven.

## Closest analog
**minibwa** (Heng Li, 2026): same lineage, same NEON target, ~2.5x over bwa-mem2 — its wins are the
skip-DP / prefilter / prefetch class (byte-safe-friendly), not a magic SW kernel. Confirms the
above. Strobealign/Accel-Align win by *fuzzy seeding* (fewer SW calls) but change output.

Sources: Suzuki-Kasahara 2018; Rognes SWIPE 2011; Marco-Sola WFA 2021 / BiWFA 2023; Dougall Johnson
M1 Firestorm SIMD tables; Hajime Suzuki M1 notes; minibwa 2026; ERT ISCA 2021; strobealign 2022;
Accel-Align 2021; mm2-fast 2022.
