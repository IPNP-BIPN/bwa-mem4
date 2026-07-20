# bwa-mem3-rs

A native Rust reimplementation of the short-read aligner
[bwa-mem2](https://github.com/bwa-mem2/bwa-mem2), indexer included, whose acceptance criterion is
**byte-identical output**: the same index files and the same SAM bytes as the reference C binary.

Not "equivalent alignments". The same bytes.

## Status

| | |
|---|---|
| Index | byte-identical, all five files |
| Single-end | byte-identical on **353,517,767** records |
| Paired-end | byte-identical on **707,312,349** records |
| Speed | SE **2.62x**, PE **1.85x** vs bwa-mem2 at `-t8` |

Measured on a real 32.9x human WGS (GIAB HG002, 2x150), not simulated reads and not a sampled
subset. Both aligners read the same on-disk index. Reproduce with `scripts/giab30x_pe.sh`.

Speed is on an Apple M4 Max. The ratio is not constant across thread counts: it decays as `-t`
rises, so quote it with the thread count attached.

## Why byte-identity is the hard part

Reproducing an aligner's *results* is not especially difficult. Reproducing its *bytes* means
reproducing its accidents too. Examples that cost real debugging time here, all now in the code
with an explanation next to them:

- bwa truncates a read-pair id to `int`, so at pair 2^23 a shift overflows into the sign bit and
  sign-extends. Reproducing that overflow is required; "fixing" it diverges on 3% of records.
- Two scoring thresholds are `float` literals compared against a `double`. The real thresholds are
  therefore 0.05000000074505806 and 0.89999997615814209, not 0.05 and 0.9.
- The chain filter's `break` skips its loop header's increment, so the chain that trips the cap is
  demoted. Off by one there changes which alignment survives.
- Several sorts are unstable, and the tie order is observable in the output.

## A caveat on the word "identical"

bwa-mem2 does not agree with itself across platforms under a non-default match score. Measured on
8000 simulated reads at `-A 2`, upstream's x86_64 binary against the arm64 build (Homebrew, SSE
translated through sse2neon), with our output matching the arm64 one exactly:

| | records |
|---|---|
| Identical | 7795 / 8000 |
| Differ in `XS` only | 194 |
| Differ in `POS`/`CIGAR`/`AS` | 11 |

On those 11, **our score is never lower** than the x86 build's: strictly higher on 5, equal on 6.
The x86 build soft-clips where we align through. And sweeping `-A` over one read shows the x86
build breaking a scaling law the algorithm mandates:

| `-A` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| linear expectation | 49 | **98** | 147 | 196 | 245 | 294 |
| bwa-mem2 x86_64 | 49 | **86** | 147 | 196 | 245 | 294 |
| bwa-mem2 arm64, and bwa-mem3 | 49 | **98** | 147 | 196 | 245 | 294 |

`update_a` multiplies every scoring parameter by `A`, so the DP surface at `-A k` is an exact
scaled copy of the one at `-A 1` and every score must scale by exactly `k`. A different suboptimal
candidate would deviate at every `A`; a lone deviation at `A = 2` is a kernel artifact. The
mechanism is at `bwamem.cpp:2302`, where the 8-bit versus 16-bit SIMD kernel is chosen on
`h0 + min(len1, len2) * opt->a`, a threshold that moves with `-A`.

**At default scoring the two builds agree exactly**, so the 30x results above are unaffected.
Parity here is stated against the arm64 build, which is the side that stays consistent.

## Verification

Three gates, in increasing order of what they can prove:

```sh
bash scripts/check.sh          # fmt, clippy, unit tests
```

```sh
# Byte-for-byte SAM comparison against bwa-mem2 across 52 option combinations.
python3 scripts/make_test_reads.py testdata/tiny/tiny.fa /tmp/ci --n 8000
IDX=testdata/tiny/tiny.fa R1=/tmp/ci_1.fq R2=/tmp/ci_2.fq \
  bash scripts/opt_parity.sh ./target/release/bwa-mem3
```

```sh
bash scripts/alt_parity.sh     # ALT contigs: real GRCh38 analysis set + the real bwakit .alt
bash scripts/giab30x_pe.sh     # the real thing: a full 30x WGS, several hours
```

```sh
bash scripts/upstream_repros.sh   # open bwa-mem2 crash issues, run against both aligners
```

The first two run in CI on every push. The rest are manual: the ALT fixture is 3.2 GB and its
index build peaks near 80 GB of RAM, and the 30x WGS takes hours.

**A passing gate proves nothing until you have watched it fail on a bug you know is there.** Every
byte-identity bug found in this project so far has lived exactly where the gate was not looking:
past 2^23 read pairs, under an untested option, or with non-default scoring. The option harness is
verified to go red on a pre-fix binary before being trusted.

Note that `cargo test` does **not** relink `target/release/bwa-mem3`, so always
`cargo build --release` before any differential run, or you will silently measure the old binary.

## Layout

A Cargo workspace, one crate per pipeline stage. The binary is `bwa-mem3` (`index`, `mem`).

| Crate | Responsibility |
|---|---|
| `bwa-core` | Types, constants, alignment options, scoring matrix |
| `bwa-io` | FASTA/FASTQ input, hand-formatted SAM output |
| `bwa-index` | FM-index construction and loading |
| `bwa-seed` | SMEM seeding |
| `bwa-chain` | Seed chaining, plus a hand port of klib's kbtree and unstable sort |
| `bwa-extend` | Scalar reference Smith-Waterman, and the `SwBackend` acceptance harness |
| `bwa-neon` | Vectorised SW kernels: NEON on aarch64, AVX2 on x86_64 |
| `bwa-mem` | Extension, dedup, primary marking, MAPQ, CIGAR, tags, paired-end |
| `bwa-cli` | The `bwa-mem3` binary |
| `bwa-diff` | Field-level SAM concordance (`sam-diff`) |
| `bwa-sam` | Empty. Reserved and never filled in; nothing depends on it |

[ARCHITECTURE.md](ARCHITECTURE.md) is the guide for someone who knows neither this code nor
bioinformatics: one read's journey from FASTQ to SAM line, a glossary of every abbreviation, and
the rules for changing this code without breaking parity.

## GPU

There is none, deliberately. A Metal backend existed and was removed: on a whole genome the
Smith-Waterman kernel is about **4%** of runtime, while seeding is about 78%. Amdahl caps any
SW-offload backend at a few percent, and each one adds a byte-identity surface that must be proven
against the scalar reference. The Metal shader had shipped a real bug (it opened gaps from `H`
instead of `M`) precisely because that proof was too weak. It is in the git history if the profile
ever changes.

## Building

```sh
cargo build --release
```

Rust 1.96. macOS and Linux, on both x86_64 and aarch64.

## Upstream bugs

bwa-mem2's behaviour is reproduced byte for byte, and that includes its bugs. Several open upstream
issues are therefore **deliberately not fixed here**, because fixing them would break the
acceptance criterion: [#293](https://github.com/bwa-mem2/bwa-mem2/issues/293) (`-R` can produce a
technically invalid BAM), [#278](https://github.com/bwa-mem2/bwa-mem2/issues/278) (`MQ` tags absent
where bwa emits them), [#260](https://github.com/bwa-mem2/bwa-mem2/issues/260) (MAPQ of
supplementary alignments).

Crashes are the exception: a run that aborts produces no output, so there is nothing to be
identical to. `scripts/upstream_repros.sh` runs three of them against both aligners.
[#269](https://github.com/bwa-mem2/bwa-mem2/issues/269) reproduces exactly, on the 345 read pairs
attached to that issue: bwa-mem2 2.3 dies with `assert failed for seqPair size` and emits nothing,
while we align all 345 pairs.

[#297](https://github.com/bwa-mem2/bwa-mem2/issues/297), the x86-vs-arm64 disagreement described
above, was filed from this project.

## Licence

MIT, in [LICENSE](LICENSE). This is a derivative work of
[bwa-mem2](https://github.com/bwa-mem2/bwa-mem2) (Copyright 2019 Intel Corporation, Heng Li), which
is also MIT licensed; its copyright notice is retained in full in that file.
