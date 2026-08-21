# What fg-labs/bwa-mem3 does that we do not

A read of the fork's source (MIT, v0.9.0, cloned 2026-08-18) with one question: **on x86_64 we are
0.74x its speed while being at parity on Apple Silicon, so what is in there that is not in here?**

The answer turned out to be mostly not in the kernels. It is in the build, and in a handful of
places where they stopped doing something per byte or per field. This file records what was found,
what was taken, and what was measured and left, so the next person reads a list instead of a
repository.

Provenance and licence: bwa-mem3 is MIT and already credited in [DEPENDENCIES.md](../DEPENDENCIES.md)
for earlier work. Nothing here is copied; the ideas are reimplemented in Rust and the measurements
are our own.

## First, what their numbers mean

Their default path is **not** byte-identical to bwa-mem2: it reproduces the alignments but adds two
tags (`MQ:i`, `HN:i`) and a header block, and `--compat=bwa-mem2` is what makes the record stream
byte-identical. Separately, `--fast`, `--rescue-skip` and `--seed-order local-longest` are
**opt-in speed levers that change output** and are mutually exclusive with `--compat`.

That matters twice over. Every benchmark in this repo compares against their DEFAULT, which does
slightly more work than a compat run, so the comparison is fair and if anything conservative. And
@nh13's table has a `mem3 --fast` column at roughly half the default's time: that column is not a
number we can chase without giving up the property this project exists for.

## The finding that explains the x86 gap

`Makefile`, line 645:

```make
BASELINE_ARCH ?= avx2
```

with the comment that says why:

> every host that runs bwa-mem3 in practice has AVX2 (Haswell, 2013+) [...] and dropping the
> baseline below avx2 measurably slows hot non-kernel paths (chain extension, FMI BWT walks, mate
> scoring) because the compiler can no longer auto-vectorize them at 256-bit width.

**Their whole non-kernel codebase is compiled at AVX2.** Ours is compiled at the x86-64 baseline,
i.e. SSE2, because a release artefact must run everywhere and our SIMD is dispatched at runtime per
function. Runtime dispatch covers the hand-written kernels and nothing else: every scalar loop in
seeding, chaining, SA resolution and SAM formatting is compiled for a 2003 CPU.

On ARM this costs nothing, which is exactly why it was never noticed: the aarch64 baseline includes
NEON, and a local A/B of `target-cpu=native` against baseline lands on 36.35 s versus 36.48 s, a
tie. The x86 profile shows the shape of the damage: per read pair against the same profile on an M4,
the vector kernels are 2.57x and the rescue kernel 4.56 against 10.03 Gcell/s/thread, which is two
different cores; but `sam_emit` is 5.15x, `chain_flt` 4.3x and the SA-walk remainder 3.5x, which is
two different compilers.

On top of the AVX2 baseline they compile **four translation units at five ISA tiers each**
(`sse41`, `sse42`, `avx`, `avx2`, `avx512bw`) and dispatch at runtime through `simd_dispatch.cpp`:
`bandedSWA.cpp`, `kswv.cpp`, `ksw.cpp` and `sam_encode.cpp`. Note the fourth one.

**What to do about it** is a packaging decision, not a code one, and it is the single largest lever
in this file: ship per-ISA artefacts (`x86-64-v2`, `x86-64-v3`) the way bwa-mem2 ships
`bwa-mem2.avx2` and `bwa-mem2.avx512bw`, and let the launcher pick. The `perf-x86` workflow measures
what each level is worth before anything is shipped.

## `src/sam_encode.cpp` — taken, in portable form

SEQ and QUAL are written 16 bytes at a time through a byte-LUT shuffle (`_mm_shuffle_epi8` on x86,
`vqtbl1q_u8` on NEON), replacing `dst[i] = "ACGTN"[src[i]]`. Reverse-complement is the same shuffle
with a reversed lane mask.

We were doing worse than the loop they replaced: a `Vec::push` per base (a capacity check per byte)
and, for every integer field of every record, a `.to_string()`, i.e. **a heap allocation per
number** — the allocation probe counts 40 allocations per read in that stage. `crates/bwa-mem/src/emit.rs`
now writes integers into a stack buffer and SEQ/QUAL in 16-byte blocks into reserved capacity, in
safe portable Rust with no per-tier build. Byte-identical, verified on 1M chr21 pairs and 500k real
GIAB pairs.

## `src/smem_dedup.cpp` — worth testing, not yet tried

Thirty-four lines: after the SMEM sort, walk the array once and drop entries identical to the last
kept one on all of `(rid, m, n, k, l, s)`. O(n), no allocation, no extra sort.

This project measured SA-lookup duplicates at 20-24% and rejected deduplication because sorting
first cost as much as the walks it saved (`across.rs`, the `BWA4_SA_DUP` probe). Their version does
not sort: it exploits an order the pipeline already established. At 86.5M SA lookups per million
pairs, and 163 ns per lookup on Zen 3, even a few percent removed is real. The open question is
byte-identity, since dropping duplicate SMEMs changes what the chain builder sees; that is a
measurement, not a guess.

## `src/read_arena.h` — measured, not worth it here

A per-chunk bump arena for the per-read name/seq/qual strings, replacing ~3 mallocs and ~3
cross-thread frees per read.

We have the equivalent finding and the opposite conclusion, because our allocator is not under the
same pressure: the allocation probe puts the whole allocator at **1.8% of busy time**, and a
per-worker arena would return at most that for a refactor across four crates with byte-identity in
play. Their motivation is explicitly the cross-thread free path, which klib's `kt_pipeline` makes
worse than rayon's ownership does here.

## The rest, briefly

| File | What it is | Our position |
|---|---|---|
| `fast_reader.c`, `fr_fastq.c` | hand-written FASTQ reader replacing kseq | We use `needletail` plus a parallel inflate; the reader is 0.2% of our allocation volume and never appears in a profile. Nothing to take yet. |
| `bwa_shm.cpp` | index kept in POSIX shared memory across runs | Real lever for a pipeline that runs the aligner many times; irrelevant to a single WGS run. Not our bottleneck (index load is amortised over hours). |
| `pdqsort_wrap.h` | pdqsort instead of klib introsort | We already replaced the allocating introsort stack; the remaining sort cost is bounded by klib's exact tie order, which we must reproduce. See #38. |
| `seed_order.cpp` | five seed-ordering heuristics, one of them a `--fast` lever | Changes output. Out of scope while byte-identity is the criterion. |
| `stage_prof.cpp`, `profiling.cpp` | their per-stage profiler | We have `BWA4_STAGE_TIME` and friends, which is how the numbers in this file were produced. |
| `compat_target.cpp` | `--compat=bwa-mem\|bwa-mem2` as a data table | Interesting design: output shaping as a row in a table rather than flags scattered through the emitters. Worth remembering if we ever need a second target. |
| `Makefile` `pgo-generate` | PGO build support | We have `scripts/pgo.sh`, and measured PGO to be an Apple Silicon lever: +12.4% on M4, -0.6% on Ampere, -4.1% on Zen 3 (#33). |

## Ranked, what this teardown says to do

1. **Ship per-ISA x86 artefacts.** Their baseline is AVX2 and ours is SSE2, and our profile's worst
   stages are exactly the scalar ones. Measurement first (`perf-x86`'s baseline/v2/v3 arm), then a
   release-workflow change.
2. **Keep going on the emitters.** The SAM path was allocating per number and pushing per byte; that
   is fixed, and the same audit is worth running over the CIGAR and tag builders.
3. **Try the adjacent-SMEM dedup**, with the byte-identity gate as the referee.
4. Leave the arena, the reader and the shared-memory index alone until a profile asks for them.
