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


## Real-world speed vs bwa-mem2 2.3 (2026-07-17, post-rebase `main`)

Genome index, 500k reads, `-t8`, quiet host, every binary pre-warmed, interleaved x3, PGO build.

| config | SE | PE |
|---|---|---|
| plain FASTQ -> `/dev/null`, `-K` 100M | 2.75-2.79x | 2.37-2.44x |
| `.gz` -> file, `-K` 100M (**one batch**, pipeline inert) | 2.60-2.63x | 2.27-2.37x |
| **`.gz` -> file, `-K` 10M (7.5 batches, pipeline live)** | **2.83-2.85x** | **2.43-2.47x** |
| **`.gz` -> `.sam.gz`** (vs `bwa-mem2 \| bgzip -@8`) | **2.81x** | — |

**~2.85x SE / ~2.45x PE at `-t8` under conditions a user actually runs**, and that is *better* than
the artificial plain-in/`/dev/null`-out benchmark.

### ⚠️ Measurement trap: count your batches before concluding anything about I/O

`-K 100000000` on 500k x 150bp reads = 75M bases = **0.8 batches**. The reader/writer pipeline
(`run_pipeline`, `69394ba`) overlaps batch N+1's read and batch N-1's write with batch N's compute --
**with a single batch it has nothing to overlap and is structurally inert**. Measured at `-K 10M`
(7.5 batches) the pipeline is worth **+8-9%**: our SE `-t8` goes 3.68s -> 3.35s while bwa-mem2 does
not move (9.6 -> 9.5s), so it is our overlap delivering, not the baseline degrading. It more than
pays for the gzip decode and the file write combined.

The production default (`chunk_size * threads` = 80M) yields thousands of batches on a real WGS run,
so it is fine; the 500k benchmark is simply too small to produce more than one. **Any benchmark that
touches I/O must report its batch count**, or it measures a configuration with half the work
disabled. This mistake was made three times in a row here: first claiming the pipeline "was missing"
(it was in PR #1, unfetched), then measuring gzip decode and SAM write at "~0 cost" on a binary that
had no pipeline, then measuring the pipeline itself at `-K` 100M where it cannot act.


## Thread sweep: there is no single "x vs bwa-mem2", there is a decaying curve

Same config (`.gz` -> file, `-K` 10M, 500k reads, genome index, PGO build, both tools at the same
`-t`, min of 2), M4 Max = **12 P-cores + 4 E-cores**:

| `-t` | bwa-mem2 | ours | ratio | our scaling |
|---|---|---|---|---|
| 1 | 53.97s | 16.44s | **3.28x** | 1.00x |
| 4 | 15.63s | 5.20s | 3.00x | 3.16x |
| 8 | 9.28s | 3.30s | 2.81x | 4.98x |
| **12** | 7.16s | **2.87s** | 2.49x | 5.72x |
| 16 | 6.98s | 2.84s | 2.45x | 5.78x |

**Quote the thread count with the ratio, always.** 3.28x and 2.45x are the same binary on the same
data.

**Why it decays: bwa-mem2 scales better than we do.** 53.97/6.98 = **7.73x** on 16 threads against
our **5.79x**. That is the shared memory system, and it is the direct cost of being faster per
thread: we do the same memory work in less time, so we reach the shared ceiling sooner. Every
per-thread win we land makes the `-t` curve decay a little more steeply. It is not a regression, it
is what winning per-thread looks like against a memory wall.

**`-t12` is the knee.** The 4 E-cores buy ~1% (2.87 -> 2.84s). Note the pipeline spends **2 extra
threads** (reader + writer) on top of `-t`, so `-t16` asks for 18 threads on a 16-core part; `-t12`
(= the P-core count) leaves room for them.
