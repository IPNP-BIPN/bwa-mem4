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

> ### ⚠️ On Apple Silicon. Off it, PGO LOSES.
>
> Measured 2026-08-18 on hosted runners, chr21, 1M wgsim pairs, three interleaved repetitions each,
> training on a different seed than the measurement, `llvm-profdata` from the toolchain:
>
> | host | plain | PGO | verdict |
> |---|---|---|---|
> | M4 Max | (above) | (above) | **+12.4%** |
> | Ampere Altra (Neoverse, hosted ARM) | 69.43 / 69.90 / 69.69 s | 69.99 / 70.32 / 69.95 s | **-0.6%** |
> | AMD EPYC 7763 (Zen 3) | 103.95 / 103.59 / 104.01 s | 108.66 / 107.98 / 107.80 s | **-4.1%** |
>
> Three architectures, no repetition crossing on either non-Apple host, and @nh13's independent
> -0.4% on Graviton4 agrees with the Altra column. So `scripts/pgo.sh` is an **Apple Silicon lever**,
> not a property of this code, and a release process must not apply it to x86_64 artefacts: it would
> ship a 4% slower binary to the platform where most WGS actually runs. Issue #33.
>
> Why that is plausible rather than mysterious: what PGO buys here is branch layout and inlining on
> the branchy driver, seeding and SAM path, and how much that is worth depends entirely on the core's
> own branch predictor. A core that already predicts these branches well gains nothing from being
> told about them.

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

`BWA4_NO_RESCUE=1` skips mate rescue entirely -- the analogue of `bwa-mem2 -S`. It is a
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

`BWA4_MATESW_TIME=1` counts the rescue's DP cells. Real GIAB, 500k pairs, `-t1`:

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

## Where the `-t16` wall actually goes: `BWA4_STAGE_TIME=1`

The three probes that predate this one (`BWA4_CHAIN_TIME`, `BWA4_MATESW_TIME`, `BWA4_TRAFFIC`) all
sum **CPU** ns across every worker via relaxed atomics. That answers "how much work did stage X do",
which is the right question for a kernel and the wrong one for a scaling deficit: it cannot say
whether the pool was idle. `BWA4_STAGE_TIME=1` is the opposite shape -- **wall** clock, main thread
only, `Instant` pairs, no atomics anywhere near a worker -- and it prints a per-stage table at the
end of a run. A rayon `par_iter` shows up as the wall time of its fork/join, stragglers included.

First run, chr21 (46 Mbp index), PE, `-K` 10M, 30 batches, plain FASTQ, M4 Max:

| stage | `-t1` | `-t8` | scaling |
|---|---|---|---|
| align (seed+chain+extend) | 103.87s (63.0%) | 15.14s (62.1%) | 6.86x |
| rescue | 48.13s (29.2%) | 7.03s (28.9%) | 6.85x |
| sam_emit | 11.25s (6.8%) | 1.44s (5.9%) | 7.82x |
| dedup_prep | 0.90s | 0.12s | 7.28x |
| **encode (serial)** | 0.102s | 0.102s | **1.00x** |
| **pestat (serial)** | 0.035s | 0.034s | **1.00x** |
| **concat (serial)** | 0.039s | 0.034s | **1.00x** |
| wait_read | 0.011s | 0.009s | -- |
| wait_write | 0.000s | 0.000s | -- |

Total 164.8s -> 24.15s = **6.76x on 8 threads, 84% efficiency.**

### Two theories this killed

**"The serial per-batch stages are the scaling deficit" -- NO.** They are **0.7%** of the `-t8`
wall, and they are **flat in `-t`, not quadratic**. The tempting argument is that `-K` defaults to
`10M * threads`, so every serial stage's per-batch cost grows 16x from `-t1` to `-t16`. True, but
the batch COUNT shrinks 16x with it: total bytes encoded, total SAM bytes concatenated and total
pairs scanned by `mem_pestat` are all fixed by the input file. Fitting Amdahl to the thread sweep
above agrees: S ~= 1.43s of the 16.44s `-t1` wall, of which ~1.08s is the index load, leaving ~2%.
They were still worth removing (they cost nothing to remove and the concat halved peak SAM RSS),
but they were never the 1.27x.

**"18 threads on 16 cores hurts us" -- NO.** `reference/bwa-mem2/src/fastmap.cpp:971` passes
`pipe_threads = 2`, so bwa-mem2 at `-t16` is also 18. `reference/bwa-mem3-cpp/src/fastmap.cpp:1961`
uses `pipe_workers = 3`, so the fork is **19**, and it beats us. We are the most frugal of the three.
Do not change what `-t` means.

**The reader is not the bottleneck either, at least not here.** Gzipped input moved `wait_read` from
0.009s to 0.048s of a 24.5s run (0.2%). Switching flate2 to zlib-ng (`--features fast-gzip`) moved
it to 0.033s and the wall by 1.4%, **under the 3% floor**. That is why `fast-gzip` is opt-in rather
than default: it buys a C build dependency (cmake, libz-ng-sys) for a gain nobody has measured yet.
The fork's own docs claim 2.2x on their read stage and -7.8% end-to-end **at 96 cores**, so re-test
this at high `-t` before dismissing it.

### What the chr21 measurement structurally cannot see

chr21's index is 152 MB; GRCh38's `cp_occ` alone is 6.2 GB. The memory-latency terms that dominate
seeding at genome scale are simply absent. And macOS on arm64 uses a **16 KiB** base page against
Linux's 4 KiB, so the TLB-reach effect that `MADV_HUGEPAGE` addresses cannot appear on this host at
all. Both caveats point the same way: **the deficit Nils measured is not reproducible on this
machine, and the next measurement has to happen on Linux at genome scale.**

## Genome scale, PE, 500k pairs, `-K` 10M: the stage table across `-t`

Same instrument, the real index (GRCh38 primary assembly, 3,099,750,718 bases, 194 contigs, rebuilt
2026-07-28 and re-verified byte-identical to `bwa-mem2 index` on all five files). M4 Max, non-PGO
release, 15 batches.

| stage | `-t1` | `-t4` | `-t8` | `-t12` | `-t1`/`-t12` |
|---|---|---|---|---|---|
| align | 28.687s | 7.661s | 4.180s | 3.110s | **9.22x (77%)** |
| rescue | 2.645s | 0.841s | 0.556s | 0.495s | **5.35x (45%)** |
| sam_emit | 1.337s | 0.360s | 0.250s | 0.323s | 4.14x |
| dedup_prep | 0.175s | 0.049s | 0.027s | 0.021s | 8.33x |
| encode | 0.055s | 0.016s | 0.013s | 0.010s | (now parallel) |
| pestat | 0.016s | 0.015s | 0.015s | 0.015s | 1.00x (serial) |
| wait_read / wait_write | 0.010 / 0.000 | 0.010 / 0.000 | 0.010 / 0.000 | 0.010 / 0.000 | -- |
| **unaccounted (index load)** | 0.856s | 0.861s | 0.872s | 0.869s | **1.00x** |
| **total wall** | 33.781s | 9.815s | 5.926s | 4.855s | **6.96x** |

Two things fall out.

**The index load is the largest single serial term, and it is FLAT.** 0.87s regardless of `-t`, so
2.5% of the `-t1` wall and **17.9% of the `-t12` wall**. It is most of the Amdahl constant. Note it
is a fixed cost per RUN, not per read: at Nils's 5M pairs it is 0.8% and irrelevant. It only
distorts small benchmarks, which is most of ours. Subtract it before quoting a scaling figure.

**Mate rescue scales at 45% efficiency against align's 77%.** That is the one stage that scales
badly, and on real GIAB it is 47-64% of paired-end compute rather than the 10% wgsim shows here.

### Rescue chunking on REAL data: a null result, and two measurement traps

An earlier pass today concluded the `CHUNKS_PER_WORKER = 2` parameterization was wrong and changed
it to target a chunk SIZE. Re-measured on real data, that conclusion does not hold and the change
was reverted. Both the original tuning and its replacement had been derived from wgsim.

**Why wgsim could not settle this.** `BWA4_STAGE_TIME` on 1M real GIAB HG002 pairs, GRCh38, PE,
`-t12`, `-K` 10M:

| stage | wall | share |
|---|---|---|
| **rescue** | **19.848s** | **59.5%** |
| align | 11.355s | 34.1% |
| sam_emit | 0.757s | 2.3% |
| dedup_prep | 0.257s | 0.8% |
| index load (unaccounted) | 1.024s | 3.1% |
| encode / pestat / deinterleave / waits | < 0.06s each | ~0.2% |

On wgsim the same stage is **10%**. Every earlier tuning of this constant optimised the dominant
paired-end stage on a workload where that stage is asleep.

**The A/B that settles it.** The two candidate formulas, in the configuration that matters (`-t16`,
DEFAULT `-K`, 540k pairs per batch, so `CHUNKS_PER_WORKER = 2` yields 16892 pairs/chunk against the
size-targeting 768), interleaved within each rep, rescue stage / total wall:

| rep | `CHUNKS_PER_WORKER = 2` | size-targeting 768 |
|---|---|---|
| 1 | 15.622 / 27.566 | 14.480 / 26.167 |
| 2 | 16.184 / 27.962 | 15.843 / 28.759 |
| 3 | 19.241 / 32.908 | 16.611 / 29.033 |

Best-of-3 favours the alternative by 5.1% of wall, the MEDIAN favours the incumbent by 2.8%, and one
arm drifts 19% against ITSELF (27.566 -> 32.908). Null result; the incumbent stays.

**Trap 1: forcing a chunk SIZE on a fixed batch confounds size with count.** Sweeping sizes through
`BWA4_RESCUE_PAIRS_PER_CHUNK` at `-K` 10M (33,784 pairs per batch) produced an alarming right
branch, `-t16` rescue: 512 -> 15.68s, 1024 -> 16.83s, 2048 -> 24.06s, 4096 -> 24.44s, 8192 -> 37.32s,
16384 -> 68.36s. That is not a chunk-size cost. 16384 on a 33,784-pair batch is **3 chunks for 16
workers**, i.e. 13 idle. The shipped formula cannot produce that, because it fixes the count at
`2 * workers`. The same 16892 at the default `-K` costs 15.6s, not 68s. **Sweep formulas, or hold the
chunk count fixed while varying size; never force a size against a fixed batch and read the result
as a size effect.**

**Trap 2: this host is too noisy for sub-10% effects, and only interleaving survives it.** The same
configuration measured 16.890s in one session and 21.598s in another (28% apart). Cause: an IDE
running rust-analyzer, `rustc` at ~98% of a core throughout. Consequences, stated precisely: the
sequential sweeps in this section support only their SHAPE, not fine ordering; the interleaved
three-arm `fork_bench.sh` result below is unaffected, because it alternates arms within each rep.
That rule is in the method notes for exactly this reason and it is what caught the error.

**What the sweeps do establish.** The left branch is a real size effect and justifies
`RESCUE_MIN_PAIRS_PER_CHUNK`: `-t12` rescue is 36.3s at 64 pairs/chunk, 25.9s at 128, 21.6s at 256,
then flat through 768. Below ~256 the batched kernel runs out of jobs to fill its lanes. The `-t12`
and `-t16` curves lie on top of each other throughout, so the effect is per-worker, not scheduling.

## Three-arm head-to-head, genome scale, PE, `-t12`: wgsim says we win, real data says we lose

Two runs of `scripts/fork_bench.sh pe 3`, same host, same index, same `-K` 10M, same non-PGO
`cargo build --release`, arms interleaved within each rep, all three binaries warmed. The ONLY
difference is the reads.

**wgsim, 500k pairs** (2026-07-28):

| arm | wall s (median) | peak RSS MB | vs bwa-mem2 |
|---|---|---|---|
| bwa-mem2 2.3 | 12.54 | 16642 | 1.00x |
| fg-labs/bwa-mem3 | 7.43 | 10846 | 1.68x |
| bwa-mem4 | **5.28** | 10880 | **2.37x** |

us vs fork: **1.407x wall** (we win), 0.996x RSS.

**Real GIAB HG002, 1M pairs, 148 bp** (2026-07-29):

| arm | wall s (median) | peak RSS MB | vs bwa-mem2 |
|---|---|---|---|
| bwa-mem2 2.3 | 58.90 | 16780 | 1.00x |
| fg-labs/bwa-mem3 | **29.23** | 11067 | **2.01x** |
| bwa-mem4 | 32.28 | 11250 | 1.82x |

us vs fork: **0.905x wall** (we LOSE by 1.10x), 0.983x RSS.

Both are byte-identical runs on our side (2,013,247 records `cmp`-clean against bwa-mem2); the fork
differs from bwa-mem2 beyond its `HN:i` tag in both.

### Why this matters more than either number

**A 1.55x swing from the reads alone.** The wgsim result is not a small overstatement, it inverts
the verdict. `docs/perf-levers.md` already said "quote the GIAB numbers"; this is the cost of not
doing so, measured. Any claim about this project's speed made on `work/r1_500k.fq` is void.

**Mechanism: it is the mate rescue, and wgsim does not exercise it.** `BWA4_STAGE_TIME` puts rescue
at **59.5%** of the real-data PE wall against **10%** on wgsim. The Phase B decomposition in
`ROADMAP.md` says the same thing from the other side: everything that is not the rescue, we win
1.32x; the rescue itself, we lose 1.45x. At 59.5% of the wall, losing 1.45x on it is the whole
deficit and more.

**The Graviton deficit is not about Graviton.** Nils measured 1.10x against us at PE `-t1` and 1.35x
at PE `-t8` on Graviton4. We now reproduce **1.10x on an M4 Max** simply by using real reads. There
is nothing to explain about that hardware: the same deficit is here, on the dev machine, and has
been all along, hidden behind a simulated read set. **This removes the Graviton box from the
critical path entirely.**

### PGO closes the whole real-data gap

Same reads, same index, same `-K`, same interleaved harness, our arm rebuilt with `scripts/pgo.sh`:

| arm | wall s (median) | peak RSS MB | vs bwa-mem2 |
|---|---|---|---|
| bwa-mem2 2.3 | 54.91 | 16777 | 1.00x |
| fg-labs/bwa-mem3 | 28.18 | 11092 | 1.94x |
| **bwa-mem4 (PGO)** | **28.71** | 11251 | **1.91x** |

us vs fork: **0.981x wall**, 0.985x RSS. Per rep the arms alternate (fork by 0.3%, us by 0.8%, fork
by 1.9%), so this is a **tie** inside the 3% floor, and we are the byte-identical one.

**PGO was worth 12.4% on real paired-end data** (32.28s -> 28.71s), against the 8.5% previously
recorded on wgsim. **Superseded 2026-08-09: it is now 3.0%** (28.76 -> 27.91 CPU-s, -t4, 5/5, and
2.5% when the profile is trained on the very data being measured). The lever shrank because the
optimisations PGO used to find have since been made by hand: const-generic monomorphisation of the
kernels, `#[inline(always)]` on `backward_ext`, the statistics branches compiled out of the hot
paths, and the `OnceLock` reads hoisted out of the loops. See ROADMAP.md, section of that date. Like everything else in this section, the wgsim figure understated the lever that
matters, because it understates the stage that dominates.

### Tested and dead: flattening the rescue rounds (the fork's shape)

`batch_mate_rescue` issues one `batched_ksw_align2` call per ROUND, and a round only holds the pairs
whose anchor list is still that deep, so later rounds look like they must be running near-empty SIMD
batches. `fg-labs/bwa-mem3` flattens every (pair, anchor) into one array and calls its kernel once
(`mem_sam_pe_batch_pre`, `bwamem_pair.cpp:733-745`). Copying that is not free for us: `matesw_collect`
computes its `skip` mask from the mate's CURRENT region list (`pe.rs:601-604`), which earlier rounds
mutate, so round `k+1` genuinely depends on round `k`. The byte-identical version would have to
compute every round speculatively and re-evaluate `skip` at apply time, trading wasted DP for lane
fill.

`BWA4_RESCUE_ROUNDS=1` sizes that trade before building it. Real GIAB, 1M pairs, PE, `-t12`:

```
36000 kernel calls, 3680602 jobs, 720 chunks, mean max_rounds 50.0, deepest 50
round 0 carries 140834 of 3680602 jobs (3.8%)
   jobs/call       calls          jobs   %_jobs
       16-31          37          1070     0.0%
       32-63        4262        237024     6.4%
      64-127       23163       2039742    55.4%
     128-511        8538       1402766    38.1%
```

**The lanes are not starving.** 102 jobs per call on average, and the lightest bucket carrying any
weight starts at 32 jobs, against 16 NEON u8 lanes and 8 i16 lanes. 93.5% of the work sits in calls
of 64+ jobs, so even a 64-lane AVX-512 u8 kernel is fed. Flattening would move 102 jobs/call to
~5100, and buy nothing, because a 16-lane vector is already full at 16.

**Do not build the speculative-batching version.** Sixth lever killed by measuring the mechanism
instead of the intuition.

What the probe does show is that `max_rounds` is **50.0 on average and 50 at most**, i.e. every chunk
saturates `-m` (`opt.max_matesw`), and that 1M pairs produce 3.68M rescue jobs (3.68 per pair). At the
~207k DP cells per job recorded above, that is ~762 G cells. The rescue cost is its cell count, fixed
by bwa's algorithm, exactly as this document already concluded.

**Stale figure to stop quoting:** `ROADMAP.md`'s Phase B decomposition ("the rescue, we lose 1.45x")
predates the current build. With PGO on real data we are at 0.981x against the fork overall, so there
is no rescue deficit left to close.

### The question this hands to the Graviton benchmark

Nils built our arm with a plain `cargo build --release`, so his numbers carry this 12.4% handicap.
More interesting: he **also measured PGO on Graviton4 and got -0.4%** (113.81s vs 113.33s), where it
is worth +12.4% here. Those cannot both be a property of the code. Either that PGO build did not
apply its profile, or the benefit is specific to the Apple microarchitecture. That is now the single
most actionable open question in this comparison, and it is one command to settle on his side.

Applying our measured PGO factor to his wgs-5M number as an upper bound on what it can explain:
113.33 / 1.124 = 100.8s against the fork's 90.59s, i.e. 1.27x -> 1.11x. It closes most but not all
of the gap, and only if PGO works there at all.

### What this makes the next lever

`crates/bwa-neon/src/matesw.rs`, compared line by line against the fork's `kswv`
(`reference/bwa-mem3-cpp/`), which `ROADMAP.md` already nominated and which has never been done.
Sizing from `docs/perf-levers.md`: rescue 1.5x faster takes PE 1.90 -> 2.07x, 2x faster -> 2.35x.

Caveat that cuts the other way, and it should be measured before any kernel work starts: our arm
above is a plain `cargo build --release`. PGO is worth ~8.5% on PE here, which alone would put us at
~29.8s against the fork's 29.2s, i.e. a tie. Re-run this table with the PGO binary first.

## Linux arm64 at genome scale, without leaving the Mac

`docs/optimization-roadmap.md` has repeatedly been unable to test anything page-table related,
because macOS on arm64 uses a **16 KiB** base page and offers no `MADV_HUGEPAGE`. Docker Desktop's
VM is Linux aarch64 with a **4 KiB** page and working THP, 16 vCPU and the host's RAM, so the whole
axis is testable locally after all. It is virtualised and it is not Graviton, so treat it as
directional, but the page size and the THP machinery are real.

```sh
# THP must be `madvise`, not `always`: in `always` both A/B arms get huge pages and the hint
# measures nothing.
docker run --rm --privileged --platform linux/arm64 alpine \
  sh -c 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled'
docker run --rm --platform linux/arm64 -v "$PWD/work:/work:ro" -v bwa4-target:/target \
  rust:1.96-bookworm bash -c 'cd /work && BWA4_STAGE_TIME=1 /target/release/bwa-mem4 mem ...'
```

Build the Linux binary once into a named volume (`apt-get install build-essential clang libclang-dev
zlib1g-dev`; rust-htslib's bindgen needs libclang). Bind-mount `work/` rather than copying it:
Docker Desktop's VM disk defaults to ~8 GB, which a 10 GB index does not fit. Virtiofs makes the
index load slow, which is why the A/B below compares the `align` STAGE and not total wall.

**Byte-identity holds across architectures**: the Linux aarch64 build produces the same SAM as the
macOS aarch64 build, and both match bwa-mem2, on the 500k-pair genome PE set
(`305454c9523d64444ab276d7c98996fa`) and the SE set (`6a6cf659e3fc208bc563b6efc353c223`). This is a
gate the repo did not previously have.

### Transparent huge pages: worth ~1.7x, and we already had them

GRCh38, 500k pairs, PE, `-K` 10M, `align` stage, best of 3:

| THP mode | peak `AnonHugePages` | align `-t12` |
|---|---|---|
| `never` | 0 MB | **5.714s** |
| `madvise`, `madvise(MADV_HUGEPAGE)` hint ON | 9866 MB | 3.456s |
| `madvise`, hint OFF (`BWA4_NO_HUGEPAGE=1`) | **9866 MB** | **3.309s** |

The TLB argument is correct: a 6.2 GB `cp_occ` walked at computed addresses under 4 KiB pages is
**1.7x slower** than the same walk under 2 MiB pages, and that is one of the largest single effects
in the aligner on Linux.

**But our explicit hint is not what obtains them.** `AnonHugePages` is identical to the kilobyte in
both arms, because mimalloc already hints THP on the arenas these allocations come from. The
on/off wall difference is noise in both directions across reps.

So `fg-labs/bwa-mem3`'s `bwa_madvise.h` is **not** the explanation for its Graviton advantage: the
fork ships mimalloc too, so it was already getting huge pages before that header existed. The
ranked-first hypothesis from the 2026-07-28 planning pass is refuted, mechanistically and twice
over. The `madvise` call is kept only because `bwa-mem4-index` is published as a library and a
consumer on the system allocator would get none of mimalloc's hinting.

**Fifth lever killed by existing machinery having already eaten it**, after LISA, the flat SA,
minibwa's 10-mer cache and the `get_sa_batch` prefetch. The pattern is now the single most reliable
prior in this project: before building a memory-system lever, check whether something already does it.

## Where the real-data paired-end time actually goes (2026-07-29 profile)

`sample` on a live `-t8` run, real GIAB HG002 1M pairs, GRCh38, PE, taking the "sort by top of
stack" section (SELF time, not the call tree). Percentages are of BUSY samples; the two idle frames
(`semaphore_wait_trap`, `__psynch_cvwait`, 67,354 samples between them) are excluded because at `-t8`
on a 12P+4E part several pool threads are legitimately parked.

| frame | % of busy |
|---|---|
| `matesw::fwd_local_sw_batch` (rescue kernel) | **37.4%** |
| `batched_extend_neon_u8` (seed extension) | 17.4% |
| seeding, inlined into the `batched_regs` rayon helper | 14.0% |
| **`primary::mem_sort_dedup_patch`** | **13.3%** |
| `get_sa_batch` | 5.4% |
| `build_chains_from_resolved` | 2.7% |
| `matesw::extract_group` (rescue lane gather/scatter) | 2.1% |
| `bns_fetch_seq` | 1.5% |
| `batch_mate_rescue` (the round loop itself) | 1.2% |
| `gen_cigar2` | 1.1% |

Two things worth knowing from this.

**`mem_sort_dedup_patch` is 13.3%, and it is a rescue cost, not a dedup-stage cost.** `BWA4_STAGE_TIME`
puts the `dedup_prep` stage at 0.8% of wall, which looks like a contradiction until you notice that
`matesw_apply` calls `mem_sort_dedup_patch` again after **every orientation that inserts a region**
(`pe.rs:788`), so nearly all of the 13.3% is inside the `rescue` stage. This is `ROADMAP.md`'s open
item #2 ("11% du profil PE, jamais regarde"), now measured and located.

**The rescue kernel runs at 8.15 Gcell/s/thread.** `BWA4_MATESW_TIME` on real data: 3,680,602 jobs,
759.7 G DP cells, 93.24s CPU, mean query 148 bp against a mean 1395 bp target window = 206,400
cells/job. The "56 Gcell/s ISA ceiling" the probe prints is naive (it assumes one cell per lane per
cycle, where the H/E/F recurrence needs several ops per cell), so the real headroom is unknown but
smaller than that ratio suggests.

### Attempted and null: hoisting the dead merge check out of the dedup scan

`mem_patch_reg` returns `None` immediately when `codes` is empty (`primary.rs:147`), and mate rescue
always calls `mem_sort_dedup_patch` that way. The scan was therefore cloning two 96-byte `MemAlnReg`s
per merge candidate purely to feed a call that could not use them. Hoisted to a `merging_enabled`
flag; byte-identical (verified against the oracle on both wgsim and real GIAB).

Interleaved A/B, 1M real pairs, `-t12`, 3 reps, rescue stage / wall:

| rep | before | after |
|---|---|---|
| 1 | 17.402 / 31.296 | 17.903 / 32.078 |
| 2 | 17.946 / 32.022 | 18.109 / 32.288 |
| 3 | 18.019 / 32.285 | 19.562 / 34.667 |

**No gain.** The "after" arm reads slower in all three reps, which is an ORDERING artifact: it always
ran second, and the six runs drift upward monotonically (17.40 -> 19.56) as the host warms. Kept
regardless, because it is strictly less work for provably identical output, but it is not a measured
win, and it tells us the 13.3% is the two `ks_introsort_by` calls plus the O(n^2) backward scan
rather than the clones.

**Harness bug to fix before the next A/B:** alternate which arm runs first between reps. A fixed
order plus a drifting host produces a systematic bias of the same size as the effects being chased.

### `mem_sort_dedup_patch`: the shape of the 13.3%, measured

`BWA4_DEDUP_SHAPE=1` counts calls and region-vector sizes. Counts, not timings, so this one is
immune to the host drift. Real GIAB, 1M pairs, PE, `-t12`:

```
5680602 calls, mean n = 94.87, mean n^2 = 32947.6
         n   from_rescue     from_prep
       0-1         61225        253351
       2-2        332981       1382600
       3-4         77575         85852
       5-8        110602         61702
      9-16        187498         49005
     17-64       1072167         85642
       65+       1838554         81848
```

**Mean n is 94.87, not a handful**, and the two callers are completely different workloads. The dedup
stage mostly calls it on `n = 2` (1.38M times). Mate rescue calls it **1.84M times with n >= 65** and
1.07M more with n in 17-64.

The mechanism: `matesw_apply` inserts one region per accepted hit and then re-runs the whole
sort/dedup (`pe.rs:788`), once per orientation, across up to 50 rounds. The vector grows as rounds
proceed, so we re-sort an ever-longer array after every single insertion. Two `ks_introsort_by`
passes over ~95 elements, ~2.9M times, is what the profile is seeing.

This answers `ROADMAP.md` open item #2 ("11% du profil PE, jamais regarde"): it is not the dedup
stage at all, it is mate rescue's incremental re-sorting.

**Why it was not fixed here.** The obvious shapes all carry byte-identity risk that this host cannot
cheaply validate:
* Skipping the re-sort when the insertion cannot change the outcome requires proving the dedup pass
  is a fixed point, and the merge branch mutates entries in place.
* Replacing the sorts is out: `ks_introsort_by`'s exact unstable permutation is output-observable
  (see its doc comment), so only its *implementation* may change, not the algorithm.
* An incremental structure (keep the vector sorted, insert in place, dedup locally) changes the order
  in which the O(n^2) scan visits pairs, and that order decides which of two equal-scoring regions
  survives.

Sizing, honestly: 13.3% of busy with maybe half of it removable is ~5% end to end. Worth doing, worth
doing carefully, and worth doing on a quiet host with the oracle gate in the loop rather than at the
end. That is the highest-value known lever left on the CPU side.

### Attacking the 13.3%: two byte-identical changes, ~2% together

**1. Skip the no-op re-sorts (`pe.rs`, `matesw_apply`).** `n` counts orientations that RAN, not ones
that INSERTED, so the C re-runs the whole sort/dedup at the end of every later orientation even when
the vector is bit-for-bit what the previous dedup returned. A `dirty` flag skips those.

Identity argument: `mem_sort_dedup_patch` is a fixed point on its own output. Its last pass sorts by
`(score desc, rb, qb)` and then removes entries equal on all three, so the comparator is a strict
TOTAL order on the survivors: the result depends on the surviving SET, not on the input permutation,
even though the sort is unstable. Feeding that output back in cannot reorder it, and the redundancy
scan cannot kill anything either, since every surviving pair already failed that test.

Measured effect on call count (`BWA4_DEDUP_SHAPE`): **5,680,602 -> 4,873,604 calls, -14.2%**.

**2. Fixed-array stack in `ks_introsort_by` (`bwa-chain`).** klib declares `ks_isort_stack_t
stack[64]`; the port used a `Vec`, i.e. one heap allocation per sort as soon as any partition is
pushed, at ~10M sorts per real paired-end run. 64 slots cannot overflow: a push only ever happens for
the larger half while the loop continues on the smaller, so depth is bounded by repeated halving.
The comparison and swap sequence is untouched, so the permutation is identical.

**Measured, 1M real GIAB pairs, PE `-t12`, arm order ALTERNATED between reps:**

| rep | order | before (rescue / wall) | after (rescue / wall) |
|---|---|---|---|
| 1 | before, after | 15.308 / 27.717 | 14.979 / 27.210 |
| 2 | after, before | 15.291 / 27.648 | 14.904 / 27.061 |
| 3 | before, after | 15.585 / 28.141 | 15.306 / 28.099 |
| 4 | after, before | 16.309 / 29.496 | 15.574 / 28.402 |

"after" wins **4 of 4** on both metrics, in both orders. Paired within-rep wall improvement: 1.8%,
2.1%, 0.1%, 3.7%, mean **1.9%**. Best-of-4: rescue 2.5%, wall 2.1%.

**Verdict: real but below the 3% floor.** Kept, because both are strictly less work for provably
identical output and the direction is consistent across a paired design, but not counted as a banked
gain. Gates: `check.sh`, wgsim genome SE + PE, real GIAB PE against the bwa-mem2 oracle
(`a29240f4398fde87dbdfc94bd741de31`, 2,013,247 records), and `-t1` == `-t12` on real data.

**The harness fix that made this readable.** The previous A/B ran a fixed arm order and produced a
systematic bias the size of the effect. Alternating the order per rep removed it: here each arm lands
within 0.1% of itself whether it runs first or second (before: 15.308 first vs 15.291 second).
Always alternate.

**What is left of the 13.3%.** The remaining cost is the two `ks_introsort_by` passes over a ~105
element vector, ~4.9M times, which is inherent: the sorts cannot be skipped when a region really was
inserted, and their permutation is output-observable so the algorithm cannot be replaced. Beating it
needs an incremental structure that reproduces klib's exact tie order, which is a much larger and
riskier piece of work than anything attempted here.

### Flat arena for the mate-rescue reverse pass: NULL

The reverse pass of the batched mate-rescue kernel (`bwa-neon/src/matesw.rs`) materialised the
reversed query and target prefixes as two fresh `Vec<u8>` per qualifying job. On a real paired-end
run that is ~3.68M jobs, so ~7M allocations per run, and the profile attributes ~1.7% of busy time to
mimalloc. Replaced by a single arena sized exactly by a counting pass, with jobs pointing into it via
`(offset, len)` spans and a `debug_assert` that the computed size equals the fill.

The bytes and their order are unchanged, so the change is byte-identical by construction.

**Measured, 1M real GIAB pairs, PE `-t12`, arm order ALTERNATED between reps:**

| rep | order | noarena (rescue / wall) | arena (rescue / wall) |
|---|---|---|---|
| 1 | noarena, arena | 14.870 / 27.015 | 14.473 / 26.679 |
| 2 | arena, noarena | 14.440 / 26.608 | 14.292 / 26.234 |
| 3 | noarena, arena | 14.910 / 27.489 | 14.496 / 26.700 |
| 4 | arena, noarena | 14.515 / 26.545 | 14.812 / 27.288 |

Means: wall 26.914 vs 26.725 (0.70%), rescue stage 14.684 vs 14.518 (1.13%). The arena wins 3 reps of
4 and **loses rep 4 by 2.8% on wall**, which is larger than the mean effect.

**Verdict: null.** Far below the 3% floor and, unlike the two changes above, the sign is not stable
across a paired alternating design. Kept only because it is strictly fewer allocator calls for
provably identical output; it is not a banked gain and must not be quoted as one. The corollary is
that the ~1.7% the profile attributes to the allocator is not recoverable at the wall-clock level
here: the two largest allocation sites in the rescue path have now both been removed (introsort
stack, reverse-pass buffers) and neither moved the needle beyond noise.

Gates passed: `clippy -D warnings`, `cargo test -p bwa-mem4-neon --lib` 5/5 including
`neon_verify::neon_u8_and_i16_match_scalar`, wgsim genome PE `305454c9523d64444ab276d7c98996fa`, real
GIAB PE against the bwa-mem2 oracle `a29240f4398fde87dbdfc94bd741de31` (2,013,247 records), and
`-t1` == `-t12` on real data.

### Inflate backend for gzipped FASTQ: real on the reader, ~1.2% ceiling on wall

Nils benchmarks `.fq.gz`; our own benches had always used plain `.fq`, so the input path had never
been measured in his regime. `needletail` builds the decompressor itself, and flate2 resolved to
`miniz_oxide`, the slowest mainstream inflate. In paired-end mode ONE reader thread inflates BOTH
mate files (`PairedFastqReader`), so that thread is the entire input throughput.

The `fast-gzip` feature already existed, disabled, with a "turn it on, measure" note never honoured.
Done now, for both candidate backends.

**Measured, 1M real GIAB pairs, `.fq.gz`, PE `-t12`, `BWA4_STAGE_TIME=1`:**

| backend | `wait_read` | vs miniz |
|---|---|---|
| `miniz_oxide` (was default) | 0.786s (3/3 identical) | — |
| `zlib-ng` (C, needs cmake) | 0.565s | -28% |
| `zlib-rs` (pure Rust) | 0.469-0.506s | **-38%** |

**Why this cannot clear the 3% floor here, and why no wall A/B will show otherwise.** `wait_read` is
the time the main thread sits blocked waiting for a batch, i.e. the part of the input path that is
NOT already hidden behind compute by `BATCH_READAHEAD = 2`. It is 3.2% of a 24.7s run under miniz.
Removing 38% of it is 0.3s, a **1.2% ceiling on wall**. A wall A/B was run anyway and was
uninterpretable: the host drifted monotonically from 25.6s to 30.0s across the six runs, an order of
magnitude more than the effect. The stage measure is the right instrument here and it bounds the
lever below the floor. Do not re-run the wall A/B.

**Defaulted anyway.** `zlib-rs` beats `zlib-ng` AND is pure Rust, so unlike zlib-ng it costs no cmake
and no C toolchain: the reason the feature was off by default no longer exists. The default now
lives on `bwa-cli` (not `bwa-io`) so that `cargo build -p bwa-mem4 --no-default-features` can still
reach back and select stock miniz_oxide; a default on `bwa-io` would be un-disableable from the
binary's build line.

**Identity.** Inflate is a bijection, so the backend cannot change the decompressed bytes, hence not
the records, hence not the `-K` batch boundaries. Verified rather than asserted: all three backends
produce the same SAM md5 on gzipped real GIAB input, `67aa384bcc8af01f3618f1816f6b7606`.

**What the stage table says about the rest of the input path.** Under `-t12` on gzipped real data:
`encode` 0.1%, `deinterleave` 0.0%, `pestat` 0.2%, `dedup_prep` 0.6%, `sam_emit` 2.8%, `wait_write`
0.0%. `align` 37.0% and `rescue` 52.4%. Every serial per-batch stage that Phase 1 of the plan set
out to parallelise is now individually under 3%, and they total ~3.7% including `sam_emit`. Phase 1
is finished as a lever: there is nothing left in the serial prologue worth taking.

### Index-tag (indirect) introsort in the dedup sorts: REGRESSION, reverted

The two `ks_introsort_by` calls in `mem_sort_dedup_patch` sort `MemAlnReg`, which is **104 bytes**,
at a mean `n` of ~95 and ~4.9M calls per real paired-end run. The obvious idea: sort a `u32` tag
array with the comparator `lt(a[idx[x]], a[idx[y]])`, then apply the permutation to `a` by cycle
walking. Swaps drop from 104 bytes to 4, and the final permutation moves each element at most once
(`n - 1` swaps) instead of the `n log n` element moves the quicksort plus final insertion pass do.

**The permutation is provably identical**, which is the only reason this was worth trying at all:
`ks_introsort_by` is deterministic and touches elements only through `lt` and through positional
swaps, so running it over tags feeds it the same comparison results, hence the same swap sequence,
hence the same element identity at every final position. Ties included, which matters because this
sort's tie order is output-observable. A dedicated test was written asserting arrangement equality
(not merely sortedness) against the direct sort, over tie-dense keys at n = 0,1,2,3,5,16,17,63,64,
95,400. It caught a real bug on the first run: the natural-looking `while idx[i] != i { swap(i,
idx[i]) }` loop applies the INVERSE permutation. With the cycle-walk fixed, output was byte-identical
in practice too (same SAM md5 `67aa384bcc8af01f3618f1816f6b7606`).

**Measured, 1M real GIAB pairs, `.fq.gz`, PE `-t12`, arm order ALTERNATED:**

| rep | order | direct (rescue / wall) | indirect (rescue / wall) |
|---|---|---|---|
| 1 | direct, indirect | 13.681 / 25.58 | 14.247 / 26.53 |
| 2 | indirect, direct | 15.029 / 27.67 | 16.657 / 29.85 |
| 3 | direct, indirect | 13.547 / 25.44 | 13.688 / 25.69 |
| 4 | indirect, direct | 14.032 / 25.98 | 14.133 / 26.50 |

The indirect sort **loses 4 of 4 on both metrics, in both arm orders**: wall 26.17 vs 27.14 mean, a
**3.7% regression**. Reverted, helper and test removed.

**Why the reasoning was wrong.** The move-size argument ignored that at n ~ 95 the whole array is
~10 KB and sits in L1 for the duration, so the 104-byte swaps are cheap streaming stores, while the
tag comparator turns every comparison into a dependent load (`idx[x]` then `a[idx[x]].re`) that the
branch predictor and the prefetcher cannot see through. Indirection wins when the array does not fit
in cache; here it does. Do not retry this without first changing the size regime.

**Status of the 13.3%.** Three distinct attacks have now failed or come in below the floor: removing
the dead clones (null), skipping no-op re-sorts plus the fixed-array stack (~1.9%, kept, below
floor), and the indirect sort (-3.7%, reverted). What remains is the O(n^2) backward scan and two
sorts whose algorithm is frozen by output-observable tie order. Anyone picking this up next should
target the scan, not the sorts.

## The mate-rescue kernel: padding-free column range (BANKED, -9.7% wall)

### How the kernel was found to be the lever

`BWA4_MATESW_TIME` on 200k real GIAB pairs, `-t8`:

```
739868 jobs, 154252207608 DP cells in 20.56s CPU -> 7.50 Gcell/s/thread
mean query = 148 bp, mean target window = 1409 bp -> 208486 cells/job
```

7.5 Gcell/s against 16 u8 lanes at ~4 GHz. The kernel was running at roughly an eighth of what the
lane count suggests, which makes it, not the DP volume, the thing to attack. Two cheaper hypotheses
were tested first and both died:

- **Lane divergence.** The kernels take `jobs.chunks(LANES)` in caller order and run each group to
  `max(qpad) x max(tlen)` over its lanes, so one long window is paid for in all 16 lanes. Probe
  (`EXEC` vs `CELLS`): executed 168.2G vs 154.3G nominal, a **1.09x tax** -- and essentially all of
  it is the query padding (160 padded / 148 real = 1.081), not divergence. The counterfactual where
  the batch is length-sorted before grouping executes **exactly the same count**: rescue windows are
  all `2 * max_dist` wide except at contig edges, so there is nothing to sort. Length sorting is a
  dead end here, and the stale claim that the kernel "bins jobs by length internally" (it does not)
  should not be revived.
- **Duplicate jobs.** 19 of 739,868 within a call, 0.0%. Not a lever.

That leaves instruction count per cell, and the per-cell body was carrying work that is constant
over most of the matrix.

### The change

Per cell the u8 NEON kernel spent ~28 vector ops, of which six were padding bookkeeping: `t == 4`
(row-invariant), `q == ZPAD` plus its select, and `(t | q) & 0x80` (two ops) plus its select. The
column loop is now split at `n_fast`, the shortest live query in the group:

- **`j < n_fast`**: no live lane can be showing `ZPAD` or `PAD`, so neither padding blend is emitted.
- **`j >= n_fast`**: the original body, unchanged.

Plus two hoists that apply to both halves: the `t == 4` compare moves out of the column loop, and the
argmax column becomes a carried vector counter (`vaddq`) instead of `vdupq_n_u8(j as u8)`, taking a
GPR-to-vector transfer off the argmax dependency chain.

On real data `n_fast` covers 148 of 160 padded columns, i.e. **92.5% of the matrix runs the short
body**.

**Why dropping the target-side PAD kill is exact.** Past a short lane's window `seq_t` really does
hold `PAD`, and those cells now carry a value instead of being forced to 0. Every op in the kernel is
lane-local, and every reader of a lane's results is already guarded by `i >= tlen[l]` (`rowmax` is
not written, `gmax`/`te`/`qe` are not updated, `extract_group` scans only to `limit[l]`), so the
values are unobservable. A `PAD` target against a real query also indexes `score_tbl` out of range
and `vqtbl1q` returns 0 there, so the cell decays rather than scoring a false match. The one case
that WOULD read a false match, `q == PAD` (XOR 0 hits the match slot), cannot occur below `n_fast`.

### Measured

Both arms are the SAME binary, selected by `BWA4_RESCUE_FASTCOL` (`0` forces `n_fast = 0`, which
routes every column through the original body), so the comparison carries no build-to-build
confound. Two independent instruments, arm order alternated:

**Kernel CPU time inside `batched_ksw_align2`** (200k real pairs, `-t8`):

| rep | order | off | on |
|---|---|---|---|
| 1 | off, on | 21.15s (7.29 Gcell/s) | 18.35s (8.41) |
| 2 | on, off | 22.26s (6.93) | 18.72s (8.24) |
| 3 | off, on | 22.55s (6.84) | 19.55s (7.89) |

Mean 21.99s vs 18.87s: **-14.2% kernel CPU, 3 of 3, in both orders.**

**Wall** (1M real GIAB pairs, `.fq.gz`, PE `-t12`):

| rep | order | off (rescue / wall) | on (rescue / wall) |
|---|---|---|---|
| 1 | off, on | 15.090 / 28.05 | 12.684 / 24.79 |
| 2 | on, off | 16.885 / 31.63 | 14.541 / 27.78 |
| 3 | off, on | 16.440 / 29.22 | 15.237 / 28.08 |
| 4 | on, off | 17.726 / 31.37 | 14.963 / 27.92 |

Mean wall 30.07s vs 27.14s: **-9.7%, 4 of 4, in both orders**; rescue stage 16.54 vs 14.36 (-13.2%),
which matches the kernel-CPU measure. **First banked gain above the 3% floor in this comparison.**

Note the host was drifting 28-31s during the wall runs. That is exactly why the paired alternating
design plus a second, host-insensitive instrument (kernel CPU ns) is the standard here: either alone
would have been unconvincing.

### Ported to every kernel

The same split is now in all six: NEON u8/i16, AVX2 u8/i16, AVX-512BW u8/i16. x86 is compile-checked
via the `x86_64-apple-darwin` cross-check; its kernels are exercised by CI's runtime tests
(`avx2_matesw_*`), which do not run on this host. **The x86 speed-up is NOT measured yet** and must
be confirmed on the x86 runner before being quoted.

Gates on aarch64: `check.sh`, `cargo test -p bwa-mem4-neon --lib` 5/5 (including
`matesw_equals_scalar` and `neon_u8_and_i16_match_scalar`), and real GIAB PE against the bwa-mem2
oracle md5 `67aa384bcc8af01f3618f1816f6b7606`.

### On-demand argmax column in the rescue kernel: REGRESSION (+18% kernel CPU), reverted

Next idea after the column split: the row-argmax column costs a compare and a select in every cell
(`vcgtq` + `vbslq`, 2 of ~22 ops in the fast body), yet `qe` is only ever read for rows that raise a
lane's global max. Replaced by an on-demand recovery: drop the per-cell tracking, and when
`row_max > gmax[l]`, scan `h_cur` for the first column equal to `row_max`. That scan returns exactly
what the strict `>` kept (the smallest attaining column), so the change is byte-identical -- and it
was, md5 `67aa384bcc8af01f3618f1816f6b7606` on real GIAB.

**Measured (kernel CPU, 200k real pairs, `-t8`, alternating):** per-cell 19.31 / 19.32 / 18.56s
against on-demand 22.90 / 22.85 / 21.92s. **+18%, 3 of 3.** Reverted.

The premise was wrong, not the arithmetic: a lane's global max climbs in a large fraction of the
~1400 rows, not "a few dozen", so the recovery fires constantly, and each one is a 160-element
strided scalar scan over 2.5 KB. Anyone tempted to retry this should first instrument how many rows
actually improve `gmax`; the per-cell form is cheap precisely because it is branch-free and stays in
registers.

### The same change under PGO, and the resulting standing against the fork

The -9.7% above was measured on a plain `cargo build --release`. PGO recovers part of the same win on
its own, so on the shipped (PGO) binary the column split is worth less. Measured with the knob on the
PGO binary, 1M real GIAB pairs, plain `.fq`, PE `-t12 -K 10M`, alternating:

| rep | order | off (rescue / wall) | on (rescue / wall) |
|---|---|---|---|
| 1 | off, on | 13.598 / 25.93 | 12.779 / 25.29 |
| 2 | on, off | 14.559 / 27.49 | 13.344 / 26.33 |
| 3 | off, on | 14.433 / 27.16 | 13.454 / 26.27 |
| 4 | on, off | 16.970 / 31.59 | 14.825 / 28.69 |

Mean wall 28.04 vs 26.65: **-5.0%, 4 of 4**; rescue 14.89 vs 13.60, -8.7%. Above the floor, and the
honest number to quote for a release build is **-5%**, not -9.7%.

**Head-to-head, PGO binary against fg-labs/bwa-mem3**, same input, `-t12 -K 10M`, arm order
alternated, 5 reps:

| rep | fork | bwa-mem4 | fork / mem4 |
|---|---|---|---|
| 1 | 26.19 | 24.89 | 1.052 |
| 2 | 26.72 | 25.63 | 1.043 |
| 3 | 25.89 | 25.79 | 1.004 |
| 4 | 27.91 | 26.41 | 1.057 |
| 5 | 30.67 | 26.32 | 1.165 |

Mean 27.48 vs 25.81, median 26.72 vs 25.79. **We win 5 of 5**, by ~5% on the median paired ratio.
Before this change the same comparison was 0.981x, i.e. a 2% loss. The standing is now:

- **bwa-mem4 is faster than the fork on real GIAB data**, on the machine that reproduces the deficit
  Nils measured on Graviton4.
- bwa-mem4 stays **byte-identical to bwa-mem2 2.3** (2,013,247 records); the fork does not
  (`fork_bench` flags it as differing beyond its `HN:i` tag).
- Peak RSS 11253 MB vs 11069 MB, 0.98x, i.e. level.

Caveat kept in the open: this is aarch64/M4 Max. The x86 kernels carry the same change but their
speed-up is unmeasured, and Graviton4 has not been re-run.

## Two target rows per pass in the rescue kernel (kernel -8.9%, wall -2.2%)

The column split removed padding work per cell. This removes **memory traffic** per cell, which the
profile said was the next thing in the way: the one-row loop pays five memory operations per cell
(load the query column, load `e[j]`, load `h_prev[j]`, store `e[j]`, store `h_cur[j]`).

Rows `i` and `i+1` share every one of them:

- both rows read the **same query column** `seq_q[j]`;
- row `i+1`'s diagonal is `H(i, j-1)`, which row `i` produced in a register one column earlier;
- row `i+1`'s E carry is `E(i+1, j)`, which row `i` computes and would otherwise store only for row
  `i+1` to load straight back;
- only row `i+1`'s H reaches `h_cur`, because `H(i, .)` is never read outside the pair.

So a pair costs 5 memory ops for 2 cells instead of 10, and the two H chains are independent, which
gives the core two streams to interleave instead of one serial chain.

**Byte-identical**: the per-cell arithmetic is untouched, and the two row epilogues still run in row
order, so freezing at row `i` suppresses row `i+1` exactly as before. The early-exit check moves to
the end of the pair, which can compute one extra row after every lane has frozen; that row's results
are discarded by the same `i >= tlen[l] || frozen[l]` guard that already existed, and `extract_group`
never reads past `limit[l]`.

**Measured, kernel CPU inside `batched_ksw_align2`** (200k real GIAB pairs, `-t8`, one binary, arm
selected by `BWA4_RESCUE_ROWPAIR`, order alternated):

| rep | order | one row | two rows |
|---|---|---|---|
| 1 | single, pair | 17.75s (8.69 Gcell/s) | 16.23s (9.50) |
| 2 | pair, single | 17.84s (8.65) | 16.25s (9.49) |
| 3 | single, pair | 17.89s (8.62) | 16.25s (9.49) |

Mean 17.83 vs 16.24: **-8.9%, 3 of 3**, and note the spread inside each arm is 0.14s and 0.02s
respectively -- this instrument is far quieter than wall clock.

**Wall, PGO binary** (1M real GIAB pairs, PE `-t12 -K 10M`, alternating): rescue stage 13.50 vs
12.65 (-6.3%, 3 of 4), wall 26.57 vs 25.98 (**-2.2%**, 2 of 4 with two large host-noise outliers).
Below the 3% floor at the wall, consistent with the kernel being ~37% of busy. Kept on the strength
of the kernel-level measure, which is unambiguous, plus the fact that it is strictly less work.

**Standing against the fork after this change** (PGO, 1M real GIAB pairs, PE `-t12 -K 10M`,
alternating, 5 reps):

| rep | fork | bwa-mem4 | fork / mem4 |
|---|---|---|---|
| 1 | 31.35 | 29.87 | 1.050 |
| 2 | 32.64 | 30.02 | 1.087 |
| 3 | 32.30 | 28.14 | 1.148 |
| 4 | 28.83 | 26.87 | 1.073 |
| 5 | 28.50 | 26.61 | 1.071 |

Mean 30.72 vs 28.30. **5 of 5, median paired ratio 1.073**, up from 1.052 before this change. (The
whole run is ~10% slower than the previous session's absolute numbers, both arms alike; the ratio is
the signal, the absolutes are not.)

**Ported to the u8 kernels only**: NEON, AVX2 and AVX-512BW. The i16 kernels keep the one-row loop on
purpose -- they are the cold path (mate rescue with 150 bp reads and `a = 1` has a score ceiling of
150, under the 250 that selects u8), so pairing them would add a second large unsafe rewrite for no
measurable gain.

### x86 verification is no longer compile-only

Found while validating this: **Rosetta 2 on this machine executes AVX2**, so building the workspace
tests for `x86_64-apple-darwin` and running them actually exercises the AVX2 kernels rather than
skipping them.

```sh
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo test --workspace --target x86_64-apple-darwin --release
```

`-C target-cpu=x86-64-v3` is required and replaces the workspace's `native` (`apple-m4` is not an x86
CPU name; rustc otherwise aborts with "64-bit code requested on a subtarget that doesn't support
it"). With it, `avx2_matesw_u8_matches_scalar`, `avx2_matesw_i16_matches_scalar` and
`avx2_u8_and_i16_match_scalar` run for real and pass, so the AVX2 paired kernel is **verified, not
merely compiled**. AVX-512 is not covered: Rosetta has no `avx512bw`, so those tests take their
feature-detect early return. Added to `scripts/check.sh`.

## Round 3: three structural ideas, all priced before building, all under the floor

Following the literature review, three ideas that had never been measured here. Each was priced with
a probe first; none survived. Recording them so nobody re-derives them.

### The rescue window is bwa's, exactly

Checked against `reference/bwa-mem2/src/bwamem_pair.cpp:186-201`: the window is
`re - rb = (pes[r].high - pes[r].low) + l_ms`, clamped to the contig. Our measured mean of 1409 bases
is that formula, not slack of ours. There is no over-computation to remove.

### Union of one read's rescue windows: 1.03x redundancy

The structure the reference exposes is striking: `mem_matesw` runs once per anchor, and
`BWA4_RESCUE_ROUNDS` shows **mean `max_rounds` = 50.0 (the `-m` cap), with only 3.8% of jobs in round
0**. So a single (pair, direction) can run 50 full 150 x 1400 Smith-Watermans, all with the SAME mate
query, differing only in which window each anchor selects.

That invites an exact optimisation with a clean proof. In local SW no positive-scoring alignment can
span more than `l_ms + (a * l_ms - o) / e` target rows -- 294 for a 150 bp mate under `a=1, o=6, e=1`,
since a deletion run of length `L` costs `o + L*e` and cannot be paid for beyond that. So for two
windows `[s,e)` and `[S,E)` with `[s,e)` inside `[S,E)`, **H, E and F agree exactly for every row
`i >= s + 294`**: every positive-scoring alignment ending there starts inside both windows. One pass
over the union would answer all sub-windows, per-row maxima included, at a burn-in of 294 rows each.

**Measured (`BWA4_ANCHOR_SPREAD`, 40k real GIAB pairs):** 10,589 multi-anchor groups, 192,731 anchors
(18.2 per group), window span 269,823,400 bases against a union of 263,050,548 -- **1.03x**, and
**0 groups (0.0%)** fit inside a single window. The anchors of a repetitive read are scattered across
the genome, not clustered, so there is essentially no overlap to share. Dead on the data, not on the
theory.

### Band-free column loop in the seed-extension kernel: fires on 15% of rows

The extension kernel spends ~32 vector ops per cell, half again what the rescue kernel spends, and
five of them exist only to apply the per-lane band mask (`band = active && beg <= j < end`, then five
blends that select new-vs-old). When every active lane shares the same `beg` and `end` AND every lane
is active, `band` is provably all-ones for the whole row: the mask is not built and all five blends
collapse. That is ten of ~32 ops.

Implemented behind `BWA4_EXTEND_UNIFORM`, byte-identical on both arms (md5
`67aa384bcc8af01f3618f1816f6b7606`). **Measured: the specialised loop fires on 1,161,024 of 7,728,070
DP rows, 15.0%.** Ten ops of 32 on 15% of rows is 4.7% of the extension kernel, under 1% of busy, and
the wall A/B confirmed it: align stage 13.32s off against 13.91s on, i.e. nothing, if anything worse
from the duplicated loop body. Reverted.

The 15% has a clear cause: `end[l] = min(prev_end, i + w[l] + 1, qlen[l])` and `w[l]` comes from
`clamp_band(w0, qlen[l], ...)`, so lanes need identical extension lengths to share a band, and a
single lane finishing early kills the all-active half. Relaxing to "band hoisted but still masked"
would cover more rows but save only 3 ops instead of 10 -- the same sub-floor product.

## AVX-512BW seed extension, and a NEON epilogue that measured null

### AVX-512BW extension kernel (new, x86, unmeasured here)

The seed-extension dispatch stopped at AVX2, so on an AVX-512 host it ran 32 u8 lanes where 64 were
available, on 18% of busy. Now instantiated: `define_sw_kernel!` gains `batched_extend_avx512_u8`
(64 lanes) and `batched_extend_avx512_i16` (32 lanes), and `sw_kernel_u8` / `sw_kernel_i16` became
runtime dispatchers preferring AVX-512BW, with `BWA4_EXTEND_TIER` (`avx2` / `avx512`) to force an arm
the way `BWA4_RESCUE_TIER` already does for mate rescue. A forced `avx512` on a host without the
feature falls through to scalar rather than silently taking AVX2, so a forced comparison stays
meaningful.

Byte-identical by construction: the macro body is shared, and only the lane count changes. The one
place AVX-512 is not a drop-in is comparison, which returns opmask registers where the shared body
wants all-ones vectors; each compare therefore materialises its mask with `movm` (one extra
instruction, no semantic change), and `bsl` is a single `ternarylogic` with immediate `0xCA`, the
truth table of `mask ? a : b`.

A verification test `avx512_verify::avx512_u8_and_i16_match_scalar` mirrors the AVX2 one: 200
randomized rounds, both lane widths, every field compared against the scalar `ksw_extend2`. It is
present and compiles, and it **returns early on this host**, because Rosetta executes AVX2 but has no
`avx512bw`. So:

- AVX2 extension and rescue kernels: **verified** locally (Rosetta) and in CI.
- AVX-512 extension and rescue kernels: **compile-checked only** until a native AVX-512BW runner runs
  the tests. The expected win (up to 2x on the extension kernel, ~9% end to end) is a projection from
  the lane count, **not a measurement**, and must not be quoted as one.

### Row maxima published with one vector store: null on this host, kept

The rescue kernel's per-row epilogue wrote `rowmax[row * 16 + l]` with 16 scalar `i32` stores under a
per-lane guard. `rowmax` is now `u8` (the lane's own width) and the whole row is published with a
single `vst1q_u8`, unconditionally.

Safe despite dropping the guard: a lane's row maxima are read only for rows `0..=limit[l]`, and
`limit[l]` is `tlen[l] - 1` or the freeze row, so every slot the unconditional store writes beyond the
old guard is one that nothing reads. `extract_group` became generic over the row-max element type so
the i16 kernels keep their `i32` buffer.

**Measured (kernel CPU, 200k real pairs, `-t8`, alternating): 26.48 / 26.60s scalar against 26.52 /
26.59s vector, i.e. nothing.** The per-lane epilogue was not the bottleneck it looked like. Kept
anyway: it is strictly fewer instructions for provably identical output, and it shrinks the buffer
4x (22 KB instead of 89 KB per 1400-row group), which is a cache argument that may read differently
on a machine with less L2 than an M4 Max. Not counted as a gain.

Note on this measurement: the host had drifted ~60% slower than the morning's runs on the identical
probe (26.5s against 16.2s for the same input and binary), so only the paired within-run comparison
is meaningful here, and further micro-measurements were stopped for the day rather than trusted.

## Vectorising what was left scalar: one kept, one reverted

Two genuinely scalar hot spots from the profile were rewritten. Both are byte-identical and both were
tested for it; only one survived measurement.

### `.pac` unpack: vectorised, kept (unmeasured on a drifting host)

`bns_fetch_seq` built its ~1400-byte rescue window one base at a time through `FmIndex::base`: a
branch (which half of the doubled space), a shift, a mask and a conditional complement per base, over
~5 G bases per real paired-end run. 565 of ~33.5k busy samples.

New `FmIndex::bases(rb, re)` / `unpack_pac_range`: four packed bytes hold sixteen bases, so the whole
expansion is one `vqtbl1q_u8` (spread each byte over four lanes), one `vshlq_u8` with the per-lane
shifts `[-6,-4,-2,0]` and one mask. **Six instructions per sixteen bases instead of roughly ninety.**
The reverse-complement half is the same packed bytes read forward then flipped end for end with
`b ^ 3`, since `3 - b == b ^ 3` for 2-bit codes; a range straddling `L` cannot come from a
contig-clamped window and falls back to the per-base loop.

Verified by `unpack_range_matches_per_base`, which compares the vectorised range against
`unpack_pac_base` for **every** start in a 2L space and ten lengths, covering both halves and every
alignment against the 4-base packing. End-to-end md5 unchanged.

Not measured: the wall A/B ran while the host drifted 32% within the run (25.4s to 33.6s across eight
alternating runs), which is larger than anything this could produce. Kept on the strength of the
instruction count and the test, not on a number. Re-measure on a quiet host before quoting it.

### Row-major `score2` scan: REGRESSION (+7.3% kernel CPU), reverted

`extract_group` walks one lane at a time over up to 1400 rows reading `rowmax[i * 16 + l]`, a
stride-16 access, and almost every read is discarded because `push_row` returns immediately below
`minsc`. Turning it row-major looked obvious: one `vld1q_u8` loads all sixteen lanes of a row, one
compare finds the lanes at or above their own `minsc`, one horizontal max decides whether any lane
cares, and sixteen loads/compares/branches collapse into three instructions.

Byte-identical (md5 unchanged): the skipped rows are exactly the rows `push_row` no-ops on, and its
merge rule only looks at the previous ACCEPTED row, so dropping no-op calls cannot change the
tracker.

**Measured (kernel CPU, 200k real pairs, `-t8`, alternating): 18.41 / 18.54 / 19.24 / 17.43s scalar
against 19.78 / 19.57 / 19.95 / 19.74s vectorised. +7.3%, 4 of 4, in both orders.** Reverted.

The cause is the cross-lane reduction: `vmaxvq_u8` once per row, 1400 times per group, is a
high-latency horizontal op that serialises, and it replaced an inner loop the branch predictor got
right essentially every time (load a byte, compare against 19, fall through). Vectorising a
perfectly-predicted scalar filter costs more than it saves. This is the third idea this session whose
premise was "SIMD must be faster here" and the third to measure worse; the pattern is that
inter-sequence SIMD punishes anything needing a per-lane answer or a cross-lane reduction.

## Retraining PGO on the production workload: rejected by measurement

`scripts/pgo.sh` trains on a 2 Mbp region with 500k simulated reads, while production is a 3.15 GB
genome with real reads. That looked like a serious mismatch, because the regime difference is
categorical rather than gradual: on simulated reads both mates place cleanly and mate rescue is ~10%
of wall, on real GIAB reads it is ~59%. The profile appeared to be telling the compiler that the
hottest production path is cold.

`pgo.sh` now takes `IDX` / `READS` / `READS2` / `T` so either workload can be trained on, and a
binary was built on the production shape (whole genome, 100k real GIAB pairs, `-t8`, `-K 10M`).
Output is byte-identical either way, as PGO cannot change semantics: md5
`67aa384bcc8af01f3618f1816f6b7606` from both binaries.

**In regime (real GIAB, 1M pairs, PE `-t12`, alternating):** region-trained 25.31 / 29.75 / 30.78 /
29.39 against real-trained 25.68 / 27.79 / 30.20 / 29.43. **Two wins each**, means 28.81 vs 28.28,
an 1.8% difference inside a run where the host drifted from 25s to 31s. No demonstrable gain.

**Out of regime (500k simulated reads, same genome index, the generalisation check):** region-trained
5.39 / 5.66 / 5.79 against real-trained 5.75 / 5.78 / 6.18. The retrained binary **loses 3 of 3, by
~5%**.

So retraining buys nothing where it was aimed and costs 5% elsewhere. **Rejected**; the default
profile stays, and the parametrisation stays with this note attached so the experiment is not redone.

The useful conclusion is about robustness, not speed. PGO's sensitivity to the training workload
turns out to be second-order here (±5% on a workload it never saw, ~0 on the one it was retrained
for), which is the opposite of the worry that motivated the experiment. It also means the shipped
binary is not quietly tuned to one dataset: the small, reproducible, simulated training set is the
robust choice, and the profile has no way to encode anything about a particular reference index,
since it records branch frequencies in the code, not data.

For the record, what is genuinely data-independent: every kernel change banked this session (the
padding-free column range, two-row blocking, the vectorised `.pac` unpack) is an unconditional
reduction in instructions per cell or per base. Those hold for any genome, any read set, any index.

## Measuring in Nils's actual regime: we are TIED at `-t16`, not ahead

Every head-to-head in this document until now used `-t12 -K 10000000`. **That is not the gist's
regime.** Nils benchmarks `-t16` with the DEFAULT `-K`, which is `10M * threads` = 160M bases, so his
batches are 16x ours, and his thread sweep says the fork's advantage grows with thread count
(0.99x at `-t1` SE, 1.35x at `-t8` PE, 1.27-1.50x at `-t16`).

Re-measured in that regime (1M real GIAB pairs, gzipped, `-t16`, default `-K`, output to
`/dev/null`, arm order alternated, 6 reps, on the quietest the host got):

| rep | fork | bwa-mem4 | fork / mem4 |
|---|---|---|---|
| 1 | 23.12 | 23.23 | 0.995 |
| 2 | 25.26 | 25.08 | 1.007 |
| 3 | 25.52 | 26.29 | 0.971 |
| 4 | 31.30 | 28.56 | 1.096 |
| 5 | 28.51 | 27.54 | 1.035 |
| 6 | 26.63 | 27.19 | 0.979 |

**Median 1.001, mean 1.015, three wins each.** So the honest statement is:

- At `-t12`, `-K` 10M or default: **we win, ~1.07**, 5 of 5 and 3 of 3 respectively.
- At `-t16`, default `-K`: **parity**.

The `-K` axis is not the cause. Separating the two variables: `-t12` with default `-K` (120M) keeps
the ~1.07 advantage, while `-t16` with `-K` 10M already loses it. **It is the thread count**, which is
the same axis the gist's own sweep identifies and the same one `docs/optimization-roadmap.md` records
(we scale 5.79x on 16 threads against bwa-mem2's 7.73x).

### Two hypotheses for the `-t16` loss, both tested, both dead tonight

**Rescue chunk stragglers: refuted.** At `-t16` with default `-K` a batch is ~533k pairs and
`CHUNKS_PER_WORKER = 2` gives 32 chunks of ~16,600 pairs, which looked like a straggler trap given
how skewed rescue cost is per pair (18.2 anchors per multi-anchor group, 50 rounds at the `-m` cap).
Swept `BWA4_RESCUE_PAIRS_PER_CHUNK` over 16.6k / 4096 / 2048 / 1024 / 512, two passes:

| chunk | pass 1 rescue | pass 2 rescue |
|---|---|---|
| default 16.6k | **10.475s** | **11.617s** |
| 4096 | 10.558 | 12.363 |
| 2048 | 10.693 | 12.305 |
| 1024 | 10.845 | 12.733 |
| 512 | 11.429 | 13.854 |

Monotonically worse as chunks get finer, in both passes. The coarse default is optimal and the
barrier is not straggler-bound. Note also that the formula is already `-K`-invariant in the way that
matters: `n_pairs / (workers * 2)` fixes the chunk COUNT at two per worker whatever `-K` is, so the
parallel decomposition is identical at 10M and at 160M. Pinning an absolute chunk size would break
that, and every absolute size tested is worse.

**Thread oversubscription (16 rayon workers + reader + writer = 18 threads on 16 cores):
untestable here.** Swept `-t16 / -t15 / -t14 / -t12` at a FIXED `-K 160M` so only the thread count
varied. Pass 1: 26.01 / 23.96 / 23.62 / 24.04. Pass 2: 24.66 / 25.20 / 25.71 / 26.52. The two passes
contradict each other and inside each pass the times track the ORDER the configs ran, not the
configuration. That is host drift, not signal.

### Why this machine cannot settle it

Two structural reasons, and they are the reason to stop measuring `-t16` here:

1. **M4 Max is heterogeneous**: 12 performance cores plus 4 efficiency cores. `-t16` therefore puts a
   quarter of the workers on cores that are a fraction of the speed, which is not what a Graviton4
   `m8g.4xlarge` does with 16 identical vCPUs. A `-t16` result here measures core heterogeneity as
   much as it measures our scaling.
2. **The host drifted 30%+ within single A/B runs all evening** (25.4s to 33.6s across eight
   alternating runs of the same binary and input). That is larger than the entire effect under study.

The `-t12` numbers in this document are on 12 homogeneous P-cores and are the trustworthy ones. The
`-t16` question needs a quiet, homogeneous 16-core box, i.e. exactly the machine the gist used.

## The `-t16` gap was our own P-core cap. Removing it: 1.10x against the fork, 6 of 6

The `-t16` parity recorded above had a cause nobody had looked for, and the barrier probe found it in
one run.

### What the barrier probe actually showed

`BWA4_BARRIER_TIME=1` times every worker inside each rayon fork/join region and reports occupancy
(`busy_sum / (wall * workers)`) and tail (`1 - slowest / wall`). At `-t16`, default `-K`, real GIAB:

```
region        wall_s    busy_s  slowest_s   occ%   tail%
encode         0.106     0.164      0.019  13.0%   81.9%
align          9.550   113.739      9.543  99.2%    0.1%
dedup_prep     0.168     1.861      0.156  92.4%    7.3%
rescue        11.588   135.580     11.588  97.5%    0.0%
sam_emit       0.718     8.496      0.708  98.6%    1.3%
```

**There is no barrier imbalance.** Align 99.2%, rescue 97.5%, sam_emit 98.6%, and tails at 0-1%. The
hypothesis this probe was built to test is refuted outright.

But the arithmetic gives it away: `113.739 / 9.550 = 11.9` threads busy, and an occupancy of 99.2%
means the divisor was **12, not 16**. The pool had twelve workers on a `-t16` run.

### The cap, and why it was wrong here

`cmd_mem.rs` capped the rayon pool to the performance-core count on Apple Silicon (12 on an M4 Max),
on the grounds that "the efficiency cores add no measurable throughput and cost ~8% more CPU". So at
`-t16` we ran on 12 cores while fg-labs/bwa-mem3 ran on 16, and still tied.

Re-measured on the real workload (1M real GIAB pairs, gzipped, PE, `-t16`, default `-K`, alternating,
3 reps):

| arm | wall | CPU |
|---|---|---|
| capped to 12 P cores | 24.25 / 26.31 / 26.39 (mean 25.65) | 260 / 285 / 290 |
| **all 16 cores** | **23.00 / 23.09 / 23.82 (mean 23.30)** | 282 / 307 / 324 |
| fg-labs/bwa-mem3 | 25.38 / 24.74 / 27.37 (mean 25.83) | 320 / 322 / 361 |

Uncapped wins **3 of 3 against the capped build, -9.2%**. The cap's claim is right about CPU and
wrong about wall: uncapped burns more CPU and finishes sooner, because a slow core doing some of the
work beats an idle one. The measurement that justified the cap was made on the small simulated
benchmark, where mate rescue is nearly cold and one E-core straggler dominates a short parallel
region; on real paired-end data rescue is ~49% of wall and there is enough work to hide them behind.

**The cap is now opt-in** (`BWA4_PCORE_CAP=1` restores it) rather than default. It cannot affect
output, and that is verified rather than argued: at a fixed `-K` the md5 is unchanged
(`-t16 -K 120000000` gives `67aa384bcc8af01f3618f1816f6b7606`, the same as `-t12` at its default
`-K` of 120M). The `-t16` default-`-K` md5 differs only because `-K = 10M * threads` moves the batch
boundaries, which is bwa's own behaviour.

### Final standing, in the gist's exact regime

PGO binary, `-t16`, default `-K`, gzipped FASTQ, output to `/dev/null`, 1M real GIAB pairs, arm order
alternated, 6 reps:

| rep | fork | bwa-mem4 | fork / mem4 |
|---|---|---|---|
| 1 | 23.41 | 20.98 | 1.116 |
| 2 | 24.41 | 21.29 | 1.147 |
| 3 | 25.32 | 23.82 | 1.063 |
| 4 | 27.26 | 25.54 | 1.067 |
| 5 | 27.14 | 24.61 | 1.103 |
| 6 | 26.54 | 24.02 | 1.105 |

**6 of 6, median 1.104**, means 25.68 vs 23.38. CPU time agrees: 342.3s vs 313.0s, 1.094. Where the
session began, the same comparison was 0.981x at `-t12` and the gist reported 1.27-1.50x in the
fork's favour at `-t16`.

Three things got us here, in order of size: the padding-free column range in the rescue kernel
(-14.2% kernel CPU), removing the P-core cap (-9.2% wall at `-t16`), and two-row blocking (-8.9%
kernel CPU). All three are byte-identical, and none of them depends on the data.

## Final standing on the gist's own benchmark (giab-4m, `-t16`, default `-K`)

Reproduced with the shipped script, which now takes `K=default` to omit `-K` and let every binary
pick bwa's own `10M * threads`:

```sh
M4=target/aarch64-apple-darwin/release/bwa-mem4 IDX=work/genome.fa \
  READS=work/giab_small/r1_4m.fq READS2=work/giab_small/r2_4m.fq \
  T=16 K=default scripts/fork_bench.sh pe 3
```

4M real GIAB pairs, GRCh38, `-t16`, default `-K` (160M, giving **8 batches**, so the reader/writer
pipeline is fully active -- at 1M pairs it is only 2 batches and the script warns that this
understates us by 8-9%):

| arm | wall s (median) | peak RSS MB | vs bwa-mem2 |
|---|---|---|---|
| bwa-mem2 2.3 | 188.24 | 22242 | 1.00x |
| fg-labs/bwa-mem3 | 98.13 | 16772 | 1.91x |
| **bwa-mem4** | **91.01** | 17260 | **2.06x** |

Per rep: fork 94.51 / 99.09 / 98.13 against mem4 87.53 / 91.47 / 91.01, i.e. ratios 1.080 / 1.083 /
1.078. **1.078x, 3 of 3, spread 0.5%.** RSS 0.971x (we use 2.9% more). Byte-identity holds at this
scale: 8,052,432 records identical to bwa-mem2, while the fork differs on 18 of them beyond its
`HN:i` tag.

**The gist reports the fork at 1.29x in its favour on giab-4m.** The total swing is therefore
1.29 x 1.078 = **~1.39x**, on the same dataset, the same thread count and the same `-K` policy.

Everything that produced it, in order of measured size, all byte-identical and none of it dependent
on the input data:

| change | measured |
|---|---|
| padding-free column range, rescue kernel (all 6 kernels) | -14.2% kernel CPU, 3/3 |
| P-core cap made opt-in | -9.2% wall at `-t16`, 3/3 |
| two-row blocking, rescue kernel (3 u8 kernels) | -8.9% kernel CPU, 3/3 |
| vectorised `.pac` unpack | 6 instructions per 16 bases instead of ~90; unmeasured |
| zlib-rs inflate by default | `wait_read` -38%; ~1.2% ceiling |

Gates at the end of the session: `check.sh` (fmt, clippy `-D warnings`, workspace tests, x86_64
cross-check, x86_64 tests under Rosetta), `scripts/opt_parity.sh` **58 passed / 0 failed** across SE,
PE, interleaved, `-a -M -Y -5 -q -H -w -t`, file and BGZF output, and the oracle md5 at fixed `-K`.
