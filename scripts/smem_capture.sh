#!/usr/bin/env bash
# Record bwa-mem2's SMEMs (all three seeding rounds) for a read set, so bwa-seed's
# `smem_replay` test can check our seeding against the oracle without running it.
#
# Needs an instrumented bwa-mem2 v2.3: reference/bwa-mem2 patched with
# scripts/bwa-mem2-smem-dump.patch (Homebrew's arm64 patch + an env-guarded dump in
# mem_collect_smem's caller). See that patch's header for the build line.
#
#   scripts/smem_capture.sh                       # rewrites testdata/smem/ (tiny index)
#   scripts/smem_capture.sh IDX READS.fq OUT.tsv  # any index/reads, e.g. work/region.fa
set -euo pipefail
cd "$(dirname "$0")/.."
M2="${M2:-$(git rev-parse --path-format=absolute --git-common-dir)/../reference/bwa-mem2/bwa-mem2}"
[ -x "$M2" ] || { echo "no instrumented bwa-mem2 at $M2 (set M2=)" >&2; exit 1; }

if [ $# -eq 3 ]; then
  IDX=$1 FQ=$2 OUT=$3
else
  IDX=testdata/tiny/tiny.fa FQ=testdata/smem/reads.fq OUT=testdata/smem/oracle.tsv
  mkdir -p testdata/smem
  TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
  # High substitution rate breaks reads into several SMEMs, so rounds 2 and 3 have work.
  python3 scripts/make_test_reads.py "$IDX" "$TMP/r" --n 300 --sub-rate 30 --seed 7 >/dev/null
  # Short names keep the fixture small; every 7th read gets an N, which ends SMEMs.
  cat "$TMP/r_1.fq" "$TMP/r_2.fq" | awk 'NR%4==1{i++; print "@r" i; next}
    NR%4==2 && i%7==0 {$0 = substr($0,1,60) "N" substr($0,62)} {print}' > "$FQ"
fi

# -t1: the dump is not thread-safe. -K: one batch, so the dump order is the input order.
BWA2_SMEM_DUMP="$OUT" "$M2" mem -t1 -K 100000000 "$IDX" "$FQ" > /dev/null 2>&1
echo "$(wc -l < "$OUT") SMEMs from $(($(wc -l < "$FQ") / 4)) reads -> $OUT"
