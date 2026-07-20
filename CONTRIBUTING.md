# Contributing

Welcome, and thanks for collaborating. This note is written mainly for **@nh13 (Nils Homer)** to get
started on the NEON backend, but it applies to any contributor. The project docs (`ROADMAP.md`,
`DIVERGENCES.md`, `DEPENDENCIES.md`) are in French; this file is in English.

## Branches and releases

- **`dev` is the default branch and the target for every pull request.** `main` is the released
  line: it only ever advances by merging `dev`, so anything on `main` has been through CI and, for
  a release, through the manual gates below.
- Work happens on a topic branch off `dev` (`feat/...`, `fix/...`, `perf/...`, `ci/...`), and lands
  in `dev` by PR. Both CI workflows run on every PR, so a red gate is visible before review rather
  than after merge.
- **What runs where, and why.** A pull request builds and tests on Linux x86_64 and Linux arm64,
  plus the full 58-case parity gate. A push to `dev` or `main` additionally builds on macOS arm64
  and macOS x86_64. Nothing runs twice: `push` is limited to the long-lived branches, so a commit
  on a PR branch no longer triggers an identical second run.

  This is a cost decision, and the reason is a billing multiplier rather than a slow job. On a
  private repository GitHub bills macOS runners at 10x: measured here, macos-15-intel took 2.9
  minutes and was charged 29, macos-14 took 1.5 and was charged 15, while the two Linux jobs
  together took 2.2 and were charged 2. Two jobs that were not the slowest accounted for 96% of the
  bill. Neither SIMD path loses coverage, because `ubuntu-24.04-arm` compiles and runs the NEON
  kernels and `ubuntu-22.04` the AVX2 ones. What macOS adds is the operating system itself, checked
  before code reaches a long-lived branch and again on every release.
- A **release** is one act: push an annotated `vX.Y.Z` tag on `dev`. Everything else is automatic.
  `.github/workflows/release.yml` then, in this order: refuses to start unless the tag matches the
  workspace version, builds the four supported targets, proves each binary rebuilds the committed
  `testdata/tiny` index byte-identically and aligns 500 pairs into exactly 1000 records,
  **fast-forwards `main` to the tagged commit**, and only then publishes the GitHub Release.

  The promotion is ordered before publishing on purpose: if `main` cannot be fast-forwarded, no
  Release appears pointing at a commit `main` does not contain. And it is a fast-forward only. If
  `main` has diverged, the job stops and asks for a human, because resolving that automatically
  inside a release job is how an unreviewed tree gets published.

  The tag's own annotation becomes the release notes, so write it as if users will read it, because
  they will.
- The tag must match the workspace version in `Cargo.toml`. The release workflow refuses to build
  otherwise, and that check is not bureaucratic: the version is stamped into `@PG VN:` on every
  SAM/BAM/CRAM the binary writes, so a mismatch mislabels other people's data.

**CI and the release build on deliberately different macOS runners, in opposite directions.** CI
uses the newest (`macos-26`, `macos-26-intel`) because the point there is to learn early that the
current OS broke something. The release uses older ones (`macos-14`, `macos-15-intel`) because the
build host sets the FLOOR: `hts-sys` compiles vendored htslib with `cc`, which without an explicit
target adopts the host's deployment version, so building on macOS 26 would stamp a minimum of 26
onto the artifact and lock out every user below it. `MACOSX_DEPLOYMENT_TARGET` is pinned per target
and then verified on the built binary with `otool`, so a future runner swap cannot silently raise
the floor. Measured on v3.0.0: arm64 requires macOS 11.0, x86_64 requires 10.12.

Release artifacts are built with an empty `RUSTFLAGS`, overriding `.cargo/config.toml`'s
`-C target-cpu=native`. A binary built with `native` runs only on CPUs at least as capable as the
build machine's, so publishing one would ship `SIGILL` to every user with an older processor.
Nothing is lost: the SIMD kernels are chosen at runtime, not by `-C target-cpu`.

**Release notes must not @-mention strangers.** Every SAM header tag starts with `@`, and GitHub
renders `@PG` in a release body as a mention of the user `pg`. That pulled an unrelated person onto
the v3.0.0 release as a participant. The publish step now wraps bare `@XX` tokens in backticks
before creating the release, and warns if any survive. Write tags with the tags quoted anyway.

**The release workflow also runs on pull requests**, building all four targets and publishing
nothing, the way COMBINE-lab/salmon and oarfish do it. It matters here more than usual: that
workflow is the only place the artifact byte-identity gate runs, so without a PR trigger a broken
release path would be discovered while tagging, which is the worst possible moment. Manual runs
take a `dry_run` switch (as FelixKrueger's repos do) that builds and validates while touching
neither `main` nor the Releases page.

Before tagging, run the two manual gates that CI cannot: `scripts/alt_parity.sh` (ALT contigs,
needs the 3.2 GB GRCh38 analysis set) and, for anything touching the aligner, `scripts/giab30x_pe.sh`.

## What this is

A **from-scratch native Rust reimplementation of bwa-mem2** (indexer included), whose acceptance
target is **byte-identical output** to the patched bwa-mem2 2.3 oracle: both the index files and the
SAM. No FFI, no linking the C++; every stage is reimplemented (indexer, FM-index, seeding, chaining,
banded Smith-Waterman, SE + PE, tags). SIMD acceleration came after correctness was locked, and a
GPU backend was tried and removed (see the README: the SW kernel is ~4% of whole-genome runtime).

The oracle is the **patched** Homebrew build: bwa-mem2 `v2.3` at rev `7aa5ff6c` + `fastmap.patch` +
`bandedSWA.cpp.patch`, built with sse2neon on arm64. All parity claims are against that exact binary.

## Current status (parity)

Measured on a real 32.9x human WGS (GIAB HG002, 2x150), not simulated reads:

- **Single-end: byte-identical on 353,517,767 records.**
- **Paired-end: byte-identical on 707,312,349 records.**
- **Index: `cmp`-identical** to `bwa-mem2 index`, all five files.
- **Multithreading is byte-identical to `-t1`** at fixed `-K` (only the `@PG` line differs).
- Speed: SE 2.62x, PE 1.85x at `-t8` on an Apple M4 Max. The ratio decays as `-t` rises, so always
  quote the thread count.

Reproduce with `scripts/giab30x_pe.sh`. Simulated reads are NOT sufficient evidence: a wgsim gate
once scored 5000/5000 while real reads diverged, because the missing pass only mattered inside
tandem repeats. See `DIVERGENCES.md` for the diagnostic method: we build a **bit-identical
instrumented oracle** and diff internal state.

## Architecture

Cargo workspace, one crate per stage (mirrors bwa-mem2):

| Crate | Role |
|---|---|
| `bwa-core` | types, `MemOpt` (exact defaults), DNA tables |
| `bwa-io` | FASTA/FASTQ IO, `-K` batching, SAM header/record writing |
| `bwa-index` | indexer (`.pac/.ann/.amb/.bwt.2bit.64/.0123`, SA-IS) + loader (`get_occ`/`get_sa`/`backward_ext`) |
| `bwa-seed` | SMEM collection + 3-round reseeding |
| `bwa-chain` | chaining, `mem_chain_flt` (incl. the ported unstable `ks_introsort`) |
| **`bwa-extend`** | **banded Smith-Waterman (`ksw_extend2`), the `SwBackend` trait, z-drop** |
| `bwa-neon` | NEON SIMD SW kernels: batched cross-read extension, and the mate-rescue kernel |
| `bwa-mem` | primary marking, MAPQ, CIGAR, tags, PE (`mem_pestat`/`mem_matesw`/`mem_pair`), pipeline glue |
| `bwa-cli` | the `bwa-mem3` binary (`index` + `mem`) |
| `bwa-diff` | field-level SAM concordance (`sam-diff`) |

For the end-to-end picture (one read's journey, a glossary of every abbreviation, and the rules for
not breaking parity), read [ARCHITECTURE.md](ARCHITECTURE.md) first.

**Before ANY differential run, `cargo build --release`.** `cargo test --release` does not relink
`target/release/bwa-mem3`, because nothing in the test tree depends on the bin target. A comparison
run against a stale binary silently measures the old code, which has already cost real debugging
time here: a verified fix appeared to change nothing.

## Where the NEON work plugs in (phase 9a)

The seam is already in place in **`crates/bwa-extend/src/lib.rs`**:

- `trait SwBackend` with an `extend(...)` method mirroring `ksw_extend2` exactly.
- `struct ScalarBackend` (delegates to `ksw_extend2`), the authoritative reference.
- `pub fn assert_backend_matches_scalar<B: SwBackend>(backend: &B)` runs a deterministic 2000-case
  random sweep and asserts your backend returns **integer-identical** `ExtendResult`s to the scalar
  kernel. **This is the acceptance gate.**

Suggested approach: add a `NeonBackend` (a new `bwa-neon` crate, or feature-gated in `bwa-extend`)
implementing `SwBackend`, port your PR #288 optimizations into it, and make it pass
`assert_backend_matches_scalar` plus `scripts/oracle_diff.sh` at 100%. Because the scores are
integer, byte-identity must hold; the scalar path stays the source of truth and the default.

Optimizations to port from your `bwa-mem2#288` and `fg-labs/bwa-mem3` fork (credited in
`DEPENDENCIES.md`): native NEON `kswv` (~7%), `vbslq` blendv in `bandedSWA` (~4%), P/E-core + L2 +
128-byte cache-line tuning, `arch=arm64` + PGO. The tuning bits (thread/QoS/cache) live outside the
byte-identity path, so they are free to land independently.

## Build, test, gates

```sh
cargo build --release              # builds the bwa-mem3 binary
bash scripts/check.sh              # fmt --check + clippy -D warnings + cargo test --workspace
bash scripts/oracle_diff.sh        # end-to-end SAM diff vs the installed bwa-mem2 (needs test data)
bash scripts/index_diff.sh         # cmp our 5 index files vs bwa-mem2 index
```

Every PR must keep `scripts/check.sh` green (fmt, clippy with `-D warnings`, all tests) and must not
regress `oracle_diff.sh`. New behavior is test-first (TDD); accelerated backends additionally pass
`assert_backend_matches_scalar`.

## Workflow

You have **read + fork** access. Fork `IPNP-BIPN/bwa-mem3-rs`, push a branch to your fork, and open a
PR against `main`. One topic per branch, commit often. Please end commit messages with a
`Co-Authored-By:` line if you pair, and avoid em-dashes in prose (a house style rule). Licensing
follows bwa-mem2 (MIT); ports of your NEON code keep their attribution.

Questions or design discussion: open a GitHub issue/discussion here, or continue from the
bwa-mem2#288 thread. Glad to have you on it.
