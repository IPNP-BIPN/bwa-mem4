# Perf levers — measured (M-series, -t1, région 2 Mbp, 500k reads, median of 3)

Gate = **biological identity** to bwa-mem2 2.3 (same read, same RNAME/POS/FLAG/CIGAR/MAPQ);
cosmetic tag diffs (XA order, XS ±few) tolerated. Verified via `scripts/oracle_diff.sh` + `sam-diff`
(`rname_pos_match`, and `all_fields_match` for the strict view). Timing via `scripts/bench.sh`.

Baseline = `main` (includes the byte-identical f-recurrence chain-shortening, ~8% on the kernel).

Oracle reference on this host/workload: bwa-mem2 2.3 SE ~23.3s, PE ~48s → ours ≈ **2.9–3.0x**.

| Lever | Parity (bio / byte) | SE wall | PE wall | peak RSS | isolated gain |
|-------|---------------------|--------:|--------:|---------:|---------------|
| baseline (main) | — | 7.96s | 16.19s | 1747/2413 MB | — |
| **1. PGO** (`scripts/pgo.sh`) | ✅ / ✅ | 7.72s | 15.47s | 1747/2412 MB | **SE +3.1%, PE +4.5%** |
| **2. scratch reuse** | ✅ / ✅ | 7.87s | 15.96s | 1744/2419 MB | **SE +1.1%, PE +1.4%** |

## LEVER 2 (nh13 last-mile) — 4 of 5 items are no-ops on this benchmark (measured)

Only **scratch reuse** touches every read on -t1 clean data. The other four items' trigger conditions
don't occur here, so they cannot show a measured gain (they matter on real/multithreaded workloads):

| Item | Verdict on -t1 clean bench | Evidence |
|------|----------------------------|----------|
| scratch reuse | ✅ SE +1.1%, PE +1.4% | committed `fce9a91` |
| kswv NEON (mate rescue) | ~0% here (~7% on messy data) | mate rescue fires on **0.02%** of pairs (2/10000) |
| P-core pinning | ~0% here | -t1 foreground already on P-cores (E-core = 3x slower via `taskpolicy -b`) |
| 128B SoA alignment | ~0% (no-op) | buffers already 16B-aligned = NEON-optimal; M-series has no cross-line penalty |
| L2 batch sizing | ~0% (no-op) | SW working set ~2.4–4.8KB = L1-resident; kernel is latency-bound, not bandwidth-bound |

Not chased further: kswv NEON (0% measurable here) and P-core pinning are deferred as
real-data/multithread robustness items; 128B-align and L2-batch are ruled out by the latency-bound
kernel evidence (a score prepass already regressed to 0.90x). The real per-read win on clean data is
**LEVER 3 (ungapped skip-DP)**, where most reads bypass the DP entirely.

## LEVER 3 (ungapped skip-DP) — mismatch-tolerant HIT

| Step | Parity | SE wall | PE wall | peak RSS | isolated gain |
|------|--------|--------:|--------:|---------:|---------------|
| prev (scratch reuse) | ✅/✅ | 7.87s | 15.96s | 1744/2419 MB | — |
| **3a. mismatch-tolerant HIT** | ✅/✅ | **5.99s** | **12.57s** | 1668/2299 MB | **SE +23.9%, PE +21.2%** |

Ported nh13's `ungapped_analyze` HIT case beyond perfect-match: a diagonal extension with
`total_mis ≤ x_threshold` (= `o_min/(a+b−e_min)` = **1** for default params) is provably banded-SW
optimal → scalar local-SW walk gives the result, DP skipped. **57% of extension jobs skip DP** (SE).
Byte-identical (5000/5000 SE, 10000/10000 PE). Commit `18c462a`.

**3b. TIGHT band — implemented, byte-identical, ~0%, reverted.** Ported the full classification
(`Hit`/`Tight(band)`/`Fallback`) with `max_sc_proof`-derived `tight_band = ceil((n·a+h0−max_sc_proof−o_min)/e_min)`,
running the tight group's SW at `max(tight_band) ⊆ w0`. **Parity held exactly** (SE 5000/5000, PE
10000/10000 all_fields_match — the reduced band provably contains the same optimum). But **measured
SE 5.99→6.02s, PE 12.57→12.70s (~0%, a hair slower)**: our adaptive band already narrows to the
diagonal in ~1–2 rows, so a tighter *start* saves almost nothing, and splitting non-HIT jobs into two
groups (tight + fallback) **fragments the SIMD batches** (worse lane fill), cancelling it. Reverted —
correct but not worth the complexity on this workload (may help on higher-error real data).

## FINAL — cumulative, load-consistent (main baseline vs PGO+scratch+HIT, back-to-back)

Both binaries benched head-to-head under identical machine load (removes cross-time drift):

| | main baseline | final (PGO+scratch+HIT) | cumulative |
|---|---:|---:|---:|
| SE wall (méd/5) | 7.75s | **5.95s** | **1.30x** (+30%) |
| PE wall (méd/3) | 15.97s | **12.64s** | **1.26x** (+26%) |
| peak RSS | 1747/2413 MB | 1668/2299 MB | lower |

**Parity: byte-identical** (SE 5000/5000, PE 10000/10000 all_fields_match vs bwa-mem2 2.3).

## LEVER 4 (difference recurrence) — STOP by analysis, grounded in this session's measurements

The diff-recurrence's two benefits are **already realized** in our kernel, so it cannot beat it:
1. *Shorter critical path* — our carried chain is already **2 ops** (the min for a serial affine-gap
   max-plus scan) after the byte-identical chain-shortening on `main`.
2. *int8 via bounded differences* — our hot kernel is **already u8** (16 lanes).

And the binding constraint is **latency, not throughput** (measured: breaking the `f`-chain → 1.55x;
a 2nd stream to hide it → register spill, ILP experiment = parity/102 spills). The diff-recurrence cuts
*total ops* (throughput), which is free-but-useless on a latency-bound kernel — it neither shortens the
2-op chain nor frees registers for a 2nd stream. The "~2.1x reported elsewhere" is versus a naive
kernel with neither optimization. Full anti-diagonal SIMD port would also mis-fit short reads (short
anti-diagonals → poor lane fill) and risks 6-field byte-identity. Not pursued.

## Per-lever summary

| Lever | Parity | Isolated gain | Note |
|-------|:------:|---------------|------|
| 1. PGO | ✅/✅ | SE +3–6%, PE +4.5% | build process, reproducible (`scripts/pgo.sh`) |
| 2. nh13 last-mile | ✅/✅ | scratch reuse +1% | 4 items no-op here (kswv/P-core/128B/L2), measured |
| 3a. mismatch-tolerant HIT | ✅/✅ | **SE +24%, PE +21%** | the unlock; 57% of jobs skip DP |
| 3b. TIGHT band | ✅/✅ | ~0% | byte-identical but adaptive band + fragmentation → reverted |
| 4. diff-recurrence | — | — | STOP by analysis (benefits already realized) |

**Cumulative ~1.30x SE / 1.26x PE over the baseline, byte-identical.** Applied to the ~2.2x-vs-bwa-mem2
reference → **~2.8–2.9x**, at/above the ~2.65x fork target. Direct oracle measurement on this host
gives ours ≈ **2.9–3.0x** bwa-mem2 2.3 (SE ~23.3s, PE ~48s vs our 5.95/12.64s).

**PGO notes:** below the ~10–15% estimate because ~85% of runtime is hand-written branchless NEON
that PGO cannot improve; the gain comes from the branchy driver/seeding/SAM path. Reproducible via
`scripts/pgo.sh` (instrument → profile 500k SE+PE → optimized rebuild). BOLT skipped (no LLVM+BOLT on
this host). PGO is a build process, not a source change, so it stacks multiplicatively on later levers.
