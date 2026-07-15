#!/usr/bin/env bash
# Reproducible PGO build: instrument -> profile on the canonical workload (500k SE + PE,
# région 2 Mbp) -> optimized rebuild. Produces target/aarch64-apple-darwin/release/bwa-mem3.
#
# Requires cargo-pgo (`cargo install cargo-pgo`) and an llvm-profdata on PATH compatible with
# rustc's LLVM. On this homebrew-rust host, homebrew LLVM provides it:
#   export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
# BOLT is intentionally skipped (needs an LLVM built with BOLT; not available here).
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
IDX="work/region.fa"
K="${K:-100000000}"

command -v cargo-pgo >/dev/null || { echo "cargo-pgo not installed (cargo install cargo-pgo)" >&2; exit 1; }
command -v llvm-profdata >/dev/null || { echo "llvm-profdata not on PATH" >&2; exit 1; }

echo "[pgo] clean previous profiles"
rm -rf target/pgo-profiles

echo "[pgo] 1/3 instrumented build"
cargo pgo build >/dev/null

INSTR=target/aarch64-apple-darwin/release/bwa-mem3
echo "[pgo] 2/3 profiling runs (SE + PE, 500k)"
"$INSTR" mem -t1 -K "$K" "$IDX" work/r1_500k.fq            >/dev/null 2>&1
"$INSTR" mem -t1 -K "$K" "$IDX" work/r1_500k.fq work/r2_500k.fq >/dev/null 2>&1

echo "[pgo] 3/3 optimized rebuild"
cargo pgo optimize build >/dev/null

echo "[pgo] done -> $INSTR"
