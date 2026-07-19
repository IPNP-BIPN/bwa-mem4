# Contributing

Welcome, and thanks for collaborating. This note is written mainly for **@nh13 (Nils Homer)** to get
started on the NEON backend, but it applies to any contributor. The project docs (`ROADMAP.md`,
`DIVERGENCES.md`, `DEPENDENCIES.md`) are in French; this file is in English.

## What this is

A **from-scratch native Rust reimplementation of bwa-mem2** (indexer included), whose acceptance
target is **byte-identical output** to the patched bwa-mem2 2.3 oracle: both the index files and the
SAM. No FFI, no linking the C++; every stage is reimplemented (indexer, FM-index, seeding, chaining,
banded Smith-Waterman, SE + PE, tags). GPU/SIMD acceleration is deliberately last, so correctness is
locked first.

The oracle is the **patched** Homebrew build: bwa-mem2 `v2.3` at rev `7aa5ff6c` + `fastmap.patch` +
`bandedSWA.cpp.patch`, built with sse2neon on arm64. All parity claims are against that exact binary.

## Current status (parity)

Measured on 5000 wgsim read pairs over chr20:2,000,000-4,000,000, `-t1 -K 10000000`:

- **Single-end: 5000/5000 SAM lines byte-identical (100%).**
- **Paired-end: 9999/10000 records byte-identical.** The single residual is a cosmetic `XS` tag
  (MAPQ unaffected); root cause documented in `DIVERGENCES.md`.
- **Index: `cmp`-identical** to `bwa-mem2 index` on tiny + chr20 + full chr1 (our own SA-IS, now
  in-place at ~16 bytes/base so the full 3.1 Gbp genome projects to ~100 GB).
- **Multithreading (rayon) is byte-identical to `-t1`** at fixed `-K` (only the `@PG` line differs).

The hard part (bit-exact seeding, chaining tie-breaks, MAPQ, PE stats) is done. See `DIVERGENCES.md`
for the diagnostic method: we build a **bit-identical instrumented oracle** and diff internal state.

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
| `bwa-sam` | **empty.** Reserved for the above and never filled in; it holds no code and nothing depends on it |
| `bwa-cli` | the `bwa-mem3` binary (`index` + `mem`) |
| `bwa-diff` | field-level SAM concordance (`sam-diff`) |
| `bwa-gpu` | Metal backend for the SW kernel |

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
