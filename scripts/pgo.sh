#!/usr/bin/env bash
# Reproducible PGO build: instrument -> profile on the canonical workload (500k SE + PE,
# région 2 Mbp) -> optimized rebuild. Produces target/aarch64-apple-darwin/release/bwa-mem4.
#
# Requires cargo-pgo (`cargo install cargo-pgo`) and an llvm-profdata on PATH compatible with
# rustc's LLVM. On this homebrew-rust host, homebrew LLVM provides it:
#   export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
# BOLT is intentionally skipped (needs an LLVM built with BOLT; not available here).
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
# Training workload. The defaults are the small, fast, REPRODUCIBLE one (a 2 Mbp region and 500k
# simulated reads); override them to train on the production shape instead.
#
# Why that matters more than it looks: PGO optimises what the profile says is hot, and the default
# workload is hot in the wrong places. On simulated reads against a 2 Mbp reference, mate rescue is
# ~10% of wall; on real GIAB reads against the whole genome it is ~59%. Training on the former tells
# the compiler the hottest production path is nearly cold. The index size matters too: a 2 Mbp
# FM-index sits in cache, so every seeding branch profiles differently from a 3 Gbp one where each
# probe misses to DRAM.
#
#   IDX=work/genome.fa READS=work/giab_real_r1.fq READS2=work/giab_real_r2.fq T=8 scripts/pgo.sh
IDX="${IDX:-work/region.fa}"
K="${K:-100000000}"
READS="${READS:-work/r1_500k.fq}"
READS2="${READS2:-work/r2_500k.fq}"
T="${T:-1}"

command -v cargo-pgo >/dev/null || { echo "cargo-pgo not installed (cargo install cargo-pgo)" >&2; exit 1; }
command -v llvm-profdata >/dev/null || { echo "llvm-profdata not on PATH" >&2; exit 1; }

echo "[pgo] clean previous profiles"
rm -rf target/pgo-profiles

echo "[pgo] 1/3 instrumented build"
cargo pgo build >/dev/null

INSTR=target/aarch64-apple-darwin/release/bwa-mem4
echo "[pgo] 2/3 profiling runs (SE + PE) on $IDX with $READS"
"$INSTR" mem -t"$T" -K "$K" "$IDX" "$READS"            >/dev/null 2>&1
"$INSTR" mem -t"$T" -K "$K" "$IDX" "$READS" "$READS2" >/dev/null 2>&1

echo "[pgo] 3/3 optimized rebuild"
cargo pgo optimize build >/dev/null

echo "[pgo] done -> $INSTR"
