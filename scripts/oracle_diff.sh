#!/usr/bin/env bash
# End-to-end oracle-diff: run bwa-mem2 (oracle) and bwa-mem4 (ours) on identical reads at a fixed
# batch (-K) single-threaded, compare the SAM header and per-field alignment concordance.
#
# Phase-0 gate: @SQ header byte-identical + a concordance JSON produced (alignment concordance is
# ~0 until real alignment lands).
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

echo "[gate] @SQ header byte-identity"
if diff <(grep '^@SQ' "$W/oracle_se.sam") <(grep '^@SQ' "$W/ours_se.sam") >/dev/null; then
  echo "  @SQ: IDENTICAL"
  SQ_OK=1
else
  echo "  @SQ: DIFFER"
  diff <(grep '^@SQ' "$W/oracle_se.sam") <(grep '^@SQ' "$W/ours_se.sam") || true
  SQ_OK=0
fi

echo "[diff] per-field concordance -> $W/report_se.json"
target/release/sam-diff "$W/oracle_se.sam" "$W/ours_se.sam" --json "$W/report_se.json" \
  | grep -E '"(oracle_records|our_records|compared|rname_pos_match|all_fields_match|only_in_)' || true

echo
if [ "$SQ_OK" = 1 ]; then
  echo "PHASE-0 GATE: PASS (@SQ identical, report produced)"
else
  echo "PHASE-0 GATE: FAIL (@SQ differs)"
  exit 1
fi
