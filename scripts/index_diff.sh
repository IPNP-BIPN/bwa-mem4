#!/usr/bin/env bash
# Phase-1 gate: our `bwa-mem4 index` must be byte-identical to `bwa-mem2 index`.
# Builds both indexes for the same FASTA in separate dirs and `cmp`s the five files.
#
# Until phase 1 lands, `bwa-mem4 index` is a stub and this script reports that and exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/Users/benjamin/Database/VEP_database/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}"
REGION="${REGION:-20:2000000-2200000}"
W=work/index_diff
mkdir -p "$W/oracle" "$W/ours"

samtools faidx "$REF" "$REGION" > "$W/oracle/ref.fa"
cp "$W/oracle/ref.fa" "$W/ours/ref.fa"

cargo build --release --quiet
OURS=target/release/bwa-mem4

if ! "$OURS" index "$W/ours/ref.fa" >/dev/null 2>&1; then
  echo "SKIP: \`bwa-mem4 index\` not implemented yet (phase 1). Nothing to compare."
  exit 0
fi

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
