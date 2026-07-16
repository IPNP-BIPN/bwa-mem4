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

**⚠️ The +3.1% / +4.5% above is a `région 2 Mbp` number and understates PGO by ~2x for real use.**
Re-measured 2026-07-16 on the **genome** index (`work/genome.fa`, 500k reads, `-t8`, quiet host,
every binary pre-warmed, interleaved x4):

| | SE `-t8` | PE `-t8` |
|---|---|---|
| PGO, region-trained | **1.061x** (1.070/1.060/1.054/1.059) | — |
| PGO, genome-trained | **1.061x** (1.063/1.060/1.057/1.062) | **1.085x** (1.080/1.091/1.083) |

**PGO is worth ~+6.1% SE / +8.5% PE at genome scale**, and byte-identical: 500k SE + 1M PE records
`cmp`-clean against the non-PGO binary (only `@PG CL:` differs, which records the invocation path).

**Why the région number was low, and why the old explanation was backwards.** This file said PGO is
"below the ~10-15% estimate because ~85% of runtime is hand-written branchless NEON PGO cannot
improve; the gain comes from the branchy driver/seeding/SAM path". The second half is right; the
first half is a **region-index artifact**. `région 2 Mbp` has a cache-resident BWT, so seeding looks
nearly free and extension looks like ~85%. On the genome, seeding + chaining are **~78%** and
extension is ~22% (see the box atop `docs/optimization-roadmap.md`). PGO targets exactly the branchy
share, so it is worth **more** where seeding dominates, i.e. in every real run.

**The training set does not matter** (measured, not assumed): region-trained and genome-trained land
on 1.061x and 1.061x. So `scripts/pgo.sh` keeps its fast `région` default; `IDX=... scripts/pgo.sh`
overrides it if you ever want to check that again. Genome training costs minutes and buys nothing.

**Measurement trap found here:** the *first* execution of a freshly-built/copied binary on macOS is
validated by the OS and ran **7-25x slower** (one rep showed 88.0s vs a true 3.44s). Warm **every
binary**, not just the index, or a first-run arm will look catastrophic. `scripts/bench.sh`'s
median-of-3 hides this; an interleaved A/B does not.

PGO is a build process, not a source change, so nothing lands in `main` that makes `cargo build
--release` faster: **ship the `scripts/pgo.sh` output**, and it stacks multiplicatively on later
levers. BOLT skipped (no LLVM+BOLT on this host).
