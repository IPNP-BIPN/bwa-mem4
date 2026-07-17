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


## ⚠️ The numbers above are wgsim. Real data is lower, and PE is much lower.

Everything above uses `work/r1_500k.fq` = **wgsim-simulated** reads. The I/O config was realistic;
the *data* was not. Re-measured on **real GIAB HG002** (500k pairs, genome index, `-t8`, warm,
min-of-2, same everything else):

| | wgsim (quoted above) | **real GIAB** |
|---|---|---|
| SE `-t8` | 2.83-2.85x | **2.61x** |
| PE `-t8` | 2.43-2.47x | **1.90x** |

**Quote the GIAB numbers.** The old `work/giab/bench.log` (SE 2.54-2.66x / PE 2.02-2.04x on 4M pairs)
was right all along.

**Why wgsim flatters PE so much: mate rescue never fires.** Measured directly with bwa-mem2's `-S -P`
(skip pairing + rescue), its PE-specific work is **0.49 us/read on wgsim** and **15.37 us/read on
GIAB — 31x more**. wgsim pairs are unique-locus and align cleanly, so the rescue path is dead code
on that data. **Benchmarking PE on simulated reads measures a pipeline with half of it asleep.**

This is the third benchmark in this repo that structurally hid what it claimed to measure:
`work/region.fa` hid seeding (the roadmap said SW was 85%; it is 4%), `-K` 100M on 500k reads hid the
reader/writer pipeline (worth +8-9%), and wgsim hides mate rescue. **Check what your benchmark
disables before trusting it.**

### The one lever this exposes: mate rescue is 64% of our PE compute

On real GIAB, pairing + mate rescue is **12.09 of our 18.85 us/read of PE compute = 64%**, against
**15.37 of 35.41 = 43%** for bwa-mem2. We are **1.27x faster** at it than they are, and it *still*
dominates us more -- because we optimised everything around it away. Amdahl: the part you do not
touch becomes the whole.

| if mate rescue got | PE compute | PE ratio |
|---|---|---|
| 1.5x faster | 18.9 -> 14.8 us/read | 1.90 -> **2.39x** |
| 2.0x faster | 18.9 -> 12.8 us/read | 1.90 -> **2.77x** |

Caveat on the arithmetic: bwa-mem2's share is measured **directly** (`-S -P`); ours is **decomposed**
(`PE - 2 x SE`), which assumes seeding+extending a read costs the same in SE and PE. Exposing our own
`-S`/`-P` would make it a like-for-like measurement.

## PE profile on real GIAB: mate rescue is 47%, and it looks nothing like SE

`BWA3_NO_RESCUE=1` skips mate rescue entirely -- the analogue of `bwa-mem2 -S`. It is a
**measurement gate, not a lever**: it changes the output by design (63,102 records differ on real
GIAB; on wgsim it barely moves, which is the whole point). It exists so our rescue cost can be
measured **directly** rather than decomposed as `PE - 2 x SE`.

Measured directly on both sides (real GIAB, 500k pairs, genome index, `-t8`, warm, min-of-2):

| | full PE | rescue off | rescue cost |
|---|---|---|---|
| bwa-mem2 | 38.71s | 22.52s (`-S`) | **16.19s** |
| ours | 22.93s | 10.06s | **12.87s** |

**We are 1.26x faster at mate rescue than bwa-mem2** (the decomposition said 1.27x -- the two methods
agree), and it is still **56% of our PE wall** against **42% of theirs**. Amdahl: we optimised
everything around it away, so the part we did not touch became the whole.

Sampler agrees (PE, real GIAB, `-t8`, leaf frames in our binary):

| frame | share |
|---|---|
| **`matesw`** | **47.3%** |
| `batched_extend` (SW extension) | 14.7% |
| `primary::mem_sort_dedup_patch` | **11.0%** |
| `get_sa` | 4.8% |
| `backward_ext` + `LsSlot` + `mem_collect_smem` (seeding) | ~8% |

**The PE profile is not the SE profile.** In SE, seeding is ~78% and the SW kernel ~4%. In PE, seeding
collapses to ~8% and mate rescue is half the run. Three sessions of seeding work (`get_sa` batching,
the lockstep, the LUT experiments) touch **~13% of PE**. And `mem_sort_dedup_patch` at 11% is larger
than `get_sa` here and has never been looked at.

**Sizing the lever** (on the direct wall, which includes the ~1.08s index load): rescue at 1.5x takes
PE 22.93 -> 18.7s = **1.90 -> 2.07x**; at 2x, -> 16.5s = **2.35x**.

**Before optimising it, re-verify the claim in `mate-rescue-vectorized-and-scaling` that the rescue
kernel is "memory-bandwidth-bound".** That note predates the finding that the aligner uses ~20% of a
single core's bandwidth, and it is the same class of claim that has been wrong every time this
session.

### Why the rescue kernel cannot be optimised much: the cell count IS the algorithm

`BWA3_MATESW_TIME=1` counts the rescue's DP cells. Real GIAB, 500k pairs, `-t1`:

```
1,838,008 jobs, 381,032,465,824 DP cells in 67.8s CPU -> 5.6 Gcell/s/thread
mean query = 148 bp, mean target window = 1401 bp -> 207,307 cells/job
```

**381 BILLION cells.** The target window is bwa's insert-size interval `[pes.low, pes.high]` ~= 1401
bp: we do a full 148 x 1401 local Smith-Waterman per anchor, 1.84M times. For scale, SE extension is
~22 G cells -- **the rescue does ~17x the DP work of the entire extension stage**. That, not kernel
quality, is why it is 47% of PE.

**So the kernel is not the lever.** We already run the *same* cell count 1.26x faster than bwa-mem2
(12.87s vs 16.19s, both measured directly). Going meaningfully faster means doing **fewer cells**, and
the cell count is bwa's algorithm. Changing it changes the output.

The obvious escape does not work either: an ungapped prefilter like the extension's `ungapped_hit`
would have to scan 148 bases across 1401 window positions = **207k comparisons, exactly the cell count
it claims to avoid**. The mate's position in the window is unknown, so there is nothing to band and
nothing to prune.

Same wall as seeding, in a different place: **byte-identity fixes the work, and we already execute it
better than the reference.** The only way past it is a different rescue (seed the mate instead of
full-window SW, which is what hash-seeded aligners do) -- i.e. a different aligner.

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
