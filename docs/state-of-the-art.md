# Who else is making BWA-MEM fast, and what of it applies here

Surveyed 2026-08-09. Every headline number below is quoted from its source, and every number in the
"measured here" column was taken on this machine against this repository's `dev`. The point of the
table is not admiration: it is to separate the claims that survive our acceptance criterion (SAM
output byte-identical to bwa-mem2 2.3) from the ones that only look like they do.

## The landscape

| tool | claim | output | cost | what it actually means for us |
|---|---|---|---|---|
| **[BWA-MEME](https://academic.oup.com/bioinformatics/article/38/9/2404/6543607)** (KAIST, 2022) | "3.32x higher **seeding throughput**", **"up to 1.4x"** end-to-end | identical to bwa-mem2 | index **118 GB** (mode 3), min 64 GB RAM, 140-192 GB recommended | the quoted 3.32x is one stage, not the run. 1.4x for +100 GB of RAM is a bad trade at our sizes, and this project already measured a learned index (LISA) at ~5x slower here |
| **[BWA-MEM-SCALE](https://dl.acm.org/doi/fullHtml/10.1145/3545008.3545033)** (ETRI, ICPP'22) | 3.19-3.32x over bwa-mem2; **1.97-2.03x from pipeline and I/O rework alone** | identical | +104 GB for the full version, **nothing** for the pipeline part | the memory-free half is the interesting half. Their gain comes from a read-ahead thread, batched writes, an in-memory index store and 1 GiB HugeTLB, on a bwa-mem2 baseline that has none of it. We already have a reader thread and a rayon pipeline, which is part of why we are at 3.5x bwa-mem2 |
| **[minibwa](https://arxiv.org/pdf/2606.15357)** (Li & Homer, 2026) | "about four times as fast as BWA-MEM and more than twice as fast as BWA-MEM2" | **not identical**: new SMEM algorithm (ropebwt3), more ungapped alignment, q-mer prefilter on mate rescue | none | measured here: 38,6 CPU-s against our 46,8 at `-t1`, and **62 % scaling efficiency at `-t16` against our 64 %**. It wins by doing less work, not by scaling better. Its heuristics are outside our contract |
| **[BWA-FastAlign](https://github.com/zzhofict/BWA-FastAlign)** | "2.27x ~ 3.28x throughput", avg 2.85x, "100% identical output" | claims identical | hybrid multi-stage index (Kmer + FMT + Direct) | no paper and no benchmark table published; the claim of intra-query SIMD parallelism in extension is interesting and unverified. Worth a measurement, not a port |
| **[ERT](https://www.biorxiv.org/content/10.1101/2020.03.23.003897.full.pdf)** (2020) | faster seeding | identical | ~60 GB index | superseded by BWA-MEME, which reports 1.72x over it. Already ruled out here on memory |
| **[QuadRank / QuadFm](https://arxiv.org/pdf/2602.04103)** (Groot Koerkamp, 2026) | rank at **2.29 bits/bp**, "only a single cache line per query", 4x over genedex | it is a rank structure: same values, so identical by construction | a load-time rebuild, or a sidecar | **the one to look at.** See below |
| Sentieon BWA | proprietary drop-in | claims identical | licence | not inspectable |
| [fg-labs/bwa-mem3](https://github.com/fg-labs/bwa-mem3) v0.9.0 | correctness release | byte-identical with `--compat=bwa-mem2`, verified to our own md5 | none | measured here: we lead 1,089x / 1,063x / 1,053x at `-t1` / `-t4` / `-t16` |

## Why QuadRank is the one that matters

Its acknowledgements say what it is for: *"I thank Heng Li for motivating me to start this project in
an attempt to speed up BWA-MEM."* It is a Rust crate, and it answers exactly the query our hot loop
answers.

The number that makes it interesting is space, not speed. Our `cp_occ` stores one 64-byte checkpoint
per 64 characters, i.e. **8 bits per base**, where the text itself needs 2. QuadRank achieves the same
rank queries at **2.29 bits per base**, so the same structure would be **3.5x smaller**: 6.2 GB
becomes about 1.8 GB. It also returns *all four symbol ranks at once*, which is precisely what
`backward_ext` needs for its `l` recurrence, and it supports batched prefetching, which is what our
lockstep driver already does.

This is the first structural lever surveyed all year that **removes** bytes rather than adding them.
Every previous one (LISA's learned index, a flat 49.6 GB suffix array, minibwa's 10-mer cache, an
extra `get_sa_batch` prefetch, a THP hint, co-locating the sampled-SA arrays) either added a structure
or removed accesses that our batching already overlapped, and all six measured zero. Per
[`scaling-model.md`](scaling-model.md), footprint is one of only two terms the model allows to matter.

**What it does not obviously buy.** Our measured traffic is already **1.65 cache lines per
`backward_ext`**, so QuadRank's headline win (one line instead of two) is mostly ours already. A 3.5x
smaller structure improves TLB and cache behaviour by shrinking the *spread*, not by cutting the
number of lines fetched, and 1.8 GB is still 32x this machine's 56 MiB TLB reach. So the expected gain
is real but not calculable from the paper, and it has to be measured.

**What it would cost.** The on-disk index must stay byte-identical (`scripts/index_diff.sh`), so the
structure has to be rebuilt at load from `cp_occ`, or written as a sidecar. For reference, this
project measured a comparable load-time repack of 3.9 GB at **+0.50 s wall / +0.60 s CPU**, against a
current total startup of 0.23 s. That startup cost is trivial on a WGS run and dominant on a 500k-pair
benchmark, so the two regimes must be reported separately.

## What the survey rules out

- **Buying speed with RAM.** BWA-MEME's honest end-to-end number is 1.4x for +100 GB; ERT is 60 GB;
  BWA-MEM-SCALE's full version is +104 GB. We are already 3.5x bwa-mem2 on this machine at bwa-mem2's
  own memory footprint.
- **Heuristics.** minibwa's lead and BWA-FastAlign's pruning both change what gets emitted. That is a
  different product, not a faster version of this one.
- **Better thread scaling.** No implementation in this table claims it, and the two we could measure
  (the fork, minibwa) land within 5 points of us, which
  [`scaling-model.md`](scaling-model.md) predicts from the machine alone.
