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

**3b. TIGHT band — assessed, deferred.** For `>x_threshold` mismatches, nh13 runs SW with a tightened
band. In our kernel the **adaptive band already narrows** to the diagonal within ~1–2 rows, so the
extra initial width is the only saving → estimated ~3% wall, against byte-identity-risky per-job band
plumbing (the `band=0` sentinel, `max_sc_proof` formula, `max_off` interaction). Deferred pending a
go/no-go: low reward, real parity risk.

## Cumulative (this session, main baseline → LEVER 3a, code only, no PGO)

SE 7.96 → 5.99s = **1.33x**; PE 16.19 → 12.57s = **1.29x**. PGO stacks (~+3–4.5%) on top.
Starting from the ~2.2x-vs-bwa-mem2 reference point, this puts the cumulative near **~2.9x** — i.e.
already at/above the ~2.65x fork target on this workload (relative gains are the solid result; the
absolute vs-oracle ratio is machine/setup-dependent).

**PGO notes:** below the ~10–15% estimate because ~85% of runtime is hand-written branchless NEON
that PGO cannot improve; the gain comes from the branchy driver/seeding/SAM path. Reproducible via
`scripts/pgo.sh` (instrument → profile 500k SE+PE → optimized rebuild). BOLT skipped (no LLVM+BOLT on
this host). PGO is a build process, not a source change, so it stacks multiplicatively on later levers.
