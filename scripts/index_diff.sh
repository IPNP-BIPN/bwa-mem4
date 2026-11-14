#!/usr/bin/env bash
# Phase-1 gate: our `bwa-mem4 index` must be byte-identical to `bwa-mem2 index`.
# Builds both indexes for the same FASTA in separate dirs and `cmp`s the five files.
# Usage: REF=genome.fa[.gz] (bgzipped or plain, faidx-able) [REGION=chr:from-to] scripts/index_diff.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${REF:?set REF to a faidx-able FASTA (plain or bgzipped)}"
REGION="${REGION:-20:2000000-2200000}"
W=work/index_diff
mkdir -p "$W/oracle" "$W/ours"

samtools faidx "$REF" "$REGION" > "$W/oracle/ref.fa"
cp "$W/oracle/ref.fa" "$W/ours/ref.fa"

cargo build --release --quiet
OURS=target/release/bwa-mem4

# A failing indexer fails the gate (set -e); its stderr is kept so the cause is visible.
"$OURS" index "$W/ours/ref.fa" >/dev/null

bwa-mem2 index "$W/oracle/ref.fa" 2>/dev/null

status=0
for ext in pac ann amb bwt.2bit.64 0123; do
  if cmp -s "$W/oracle/ref.fa.$ext" "$W/ours/ref.fa.$ext"; then
    echo "  .$ext: IDENTICAL"
  else
    echo "  .$ext: DIFFER"
    status=1
  fi
done
[ "$status" = 0 ] && echo "PHASE-1 GATE: PASS" || { echo "PHASE-1 GATE: FAIL"; exit 1; }
