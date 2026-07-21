#!/usr/bin/env bash
# PE-only 30x re-run, to prove PE byte-identity at WGS scale after the mem_pair `int id`
# overflow fix (the divergence started at pair 2^23, so only a full-depth run exercises it).
# Same harness as giab30x_bench.sh: records piped straight through md5+wc, nothing large written.
set -uo pipefail
cd "$(dirname "$0")/.."

M2=bwa-mem2
M3=./target/release/bwa-mem4
IDX=work/genome.fa
R1=work/giab30x/HG002_30x_R1.fastq.gz
R2=work/giab30x/HG002_30x_R2.fastq.gz
K=100000000
T="${T:-8}"

for f in "$R1" "$R2" "$IDX.bwt.2bit.64" "$M3"; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
done

TS=$(date +%Y%m%d_%H%M%S)
OUT="work/giab30x/pe_$TS"; mkdir -p "$OUT"
LOG="$OUT/results.log"
say() { echo "$@" | tee -a "$LOG"; }

# $1=out prefix, rest=alignment command. BSD /usr/bin/time -p (macOS date has no %N).
run() {
  local of="$1"; shift
  /usr/bin/time -p bash -c "$* 2>'$of.err' | grep -v '^@' | tee >(wc -l >'$of.n') | md5 >'$of.md5'" 2>"$of.time"
  local real; real=$(awk '/^real/{print $2}' "$of.time")
  # A real 30x PE run is hours; anything fast means the aligner errored out.
  if awk "BEGIN{exit !($real < 30)}"; then
    echo "ABORT: $(basename "$of") returned in ${real}s (aligner failed). stderr tail:" >&2
    tail -3 "$of.err" >&2
    echo "$real"; return 1
  fi
  echo "$real"
}

say "############################################################"
say "# GIAB HG002 ~30x  PE identity re-run  $TS   (t$T, -K $K)"
say "# bwa-mem4 @ $(git rev-parse --short HEAD) + uncommitted id_shift_c fix"
say "############################################################"

a=$(run "$OUT/m2_pe" $M2 mem -t"$T" -K $K "$IDX" "$R1" "$R2") \
  || { say "  [ABORT] mem2 failed (see $OUT/m2_pe.err)"; exit 1; }
say "  mem2 done: ${a}s"
b=$(run "$OUT/m3_pe" $M3 mem -t"$T" -K $K "$IDX" "$R1" "$R2") \
  || { say "  [ABORT] mem3 failed (see $OUT/m3_pe.err)"; exit 1; }
say "  mem3 done: ${b}s"
say "  speedup=$(echo "scale=3; $a/$b" | bc)x"

if diff -q "$OUT/m2_pe.md5" "$OUT/m3_pe.md5" >/dev/null && diff -q "$OUT/m2_pe.n" "$OUT/m3_pe.n" >/dev/null; then
  say "  [PASS] PE BYTE-IDENTICAL: $(cat "$OUT/m2_pe.n") records, md5=$(cat "$OUT/m2_pe.md5")"
else
  say "  [FAIL] PE differs: mem2 $(cat "$OUT/m2_pe.n")rec/$(cat "$OUT/m2_pe.md5")  vs  mem3 $(cat "$OUT/m3_pe.n")rec/$(cat "$OUT/m3_pe.md5")"
fi

say ""
say "GIAB30X_PE_DONE  results in $LOG"
