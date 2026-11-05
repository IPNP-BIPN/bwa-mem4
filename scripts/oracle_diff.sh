#!/usr/bin/env bash
# End-to-end oracle-diff: run bwa-mem2 (oracle) and bwa-mem4 (ours) on identical reads at a fixed
# batch (-K) single-threaded. Gate: @SQ header AND every alignment record byte-identical, SE and
# (when work/r2.fq exists) PE. sam-diff's per-field report is printed to explain a failure.
set -euo pipefail
cd "$(dirname "$0")/.."

W=work
IDX="$W/region.fa"
K="${K:-10000000}"
[ -f "$IDX.ann" ] || { echo "no test index; run scripts/make_testdata.sh first" >&2; exit 1; }

# Preflight: confirm the oracle is the expected build.
V="$(bwa-mem2 version 2>&1 | tail -1)"
[ "$V" = "2.3" ] || echo "WARN: oracle bwa-mem2 version '$V' != expected 2.3" >&2

cargo build --release --quiet
OURS=target/release/bwa-mem4

echo "[run] oracle + ours (SE, -t1 -K $K)"
bwa-mem2 mem -t1 -K "$K" "$IDX" "$W/r1.fq" 2>/dev/null > "$W/oracle_se.sam"
"$OURS" mem -t1 -K "$K" "$IDX" "$W/r1.fq" > "$W/ours_se.sam"
MODES=se
if [ -f "$W/r2.fq" ]; then
  echo "[run] oracle + ours (PE, -t1 -K $K)"
  bwa-mem2 mem -t1 -K "$K" "$IDX" "$W/r1.fq" "$W/r2.fq" 2>/dev/null > "$W/oracle_pe.sam"
  "$OURS" mem -t1 -K "$K" "$IDX" "$W/r1.fq" "$W/r2.fq" > "$W/ours_pe.sam"
  MODES="se pe"
fi

echo "[gate] @SQ header byte-identity"
if diff <(grep '^@SQ' "$W/oracle_se.sam") <(grep '^@SQ' "$W/ours_se.sam") >/dev/null; then
  echo "  @SQ: IDENTICAL"
  SQ_OK=1
else
  echo "  @SQ: DIFFER"
  diff <(grep '^@SQ' "$W/oracle_se.sam") <(grep '^@SQ' "$W/ours_se.sam") || true
  SQ_OK=0
fi

OK=$SQ_OK
for m in $MODES; do
  # Records only: the @PG line carries each tool's own name and command line by design.
  if cmp -s <(grep -v '^@' "$W/oracle_$m.sam") <(grep -v '^@' "$W/ours_$m.sam"); then
    echo "  $m records: IDENTICAL ($(grep -vc '^@' "$W/ours_$m.sam"))"
  else
    echo "  $m records: DIFFER, per-field report -> $W/report_$m.json"
    target/release/sam-diff "$W/oracle_$m.sam" "$W/ours_$m.sam" --json "$W/report_$m.json" \
      | grep -E '"(oracle_records|our_records|compared|rname_pos_match|all_fields_match|only_in_)' || true
    OK=0
  fi
done

echo
if [ "$OK" = 1 ]; then
  echo "ORACLE GATE: PASS (@SQ and records byte-identical: $MODES)"
else
  echo "ORACLE GATE: FAIL"
  exit 1
fi
