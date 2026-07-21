#!/usr/bin/env bash
# 30x real-WGS head-to-head: bwa-mem4 vs bwa-mem2 on GIAB HG002 (NIST HiSeq 300x subset, ~30x).
# t8-only (t1 is days at this depth; t1 ratio comes from the depth-invariant small benches).
# Both tools use the single byte-identical index work/genome.fa. Every timed run also pipes its
# records through md5, so identity is checked on every pass at zero extra passes.
set -uo pipefail
cd "$(dirname "$0")/.."

M2=bwa-mem2
M3=./target/release/bwa-mem4
IDX=work/genome.fa
R1=work/giab30x/HG002_30x_R1.fastq.gz
R2=work/giab30x/HG002_30x_R2.fastq.gz
K=100000000
T="${T:-8}"
SEREPS="${SEREPS:-2}"
PEREPS="${PEREPS:-1}"

for f in "$R1" "$R2" "$IDX.bwt.2bit.64"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done

TS=$(date +%Y%m%d_%H%M%S)
OUT="work/giab30x/bench_$TS"; mkdir -p "$OUT"
LOG="$OUT/results.log"
say() { echo "$@" | tee -a "$LOG"; }

# timed run through md5+count. $1=out prefix, rest=alignment command.
# Uses BSD /usr/bin/time -p (macOS date has no %N); bash -c for process substitution.
# Alignment stderr is kept in $of.err (small: just bwa progress) so failures are diagnosable.
run() {
  local of="$1"; shift
  local cmd="$*"
  /usr/bin/time -p bash -c "$cmd 2>'$of.err' | grep -v '^@' | tee >(wc -l >'$of.n') | md5 >'$of.md5'" 2>"$of.time"
  local real; real=$(awk '/^real/{print $2}' "$of.time")
  # A real 30x SE/PE run is >>60s; anything faster means the aligner errored out.
  if awk "BEGIN{exit !($real < 30)}"; then
    echo "ABORT: $(basename "$of") returned in ${real}s (aligner failed). stderr tail:" >&2
    tail -3 "$of.err" >&2
    echo "$real"
    return 1
  fi
  echo "$real"
}

say "############################################################"
say "# GIAB HG002 ~30x head-to-head  $TS   (t$T, -K $K)"
say "# binary: $(git rev-parse --short HEAD)"
PAIRS=$(( $(gunzip -c "$R1" | wc -l) / 4 ))
COV=$(echo "scale=1; $PAIRS*300/3200000000" | bc)
say "# reads: $PAIRS pairs  (~${COV}x coverage, 2x150)"
say "############################################################"

########## SE ##########
say ""
say "==== SE  -t$T  x$SEREPS ===="
for i in $(seq 1 "$SEREPS"); do
  a=$(run "$OUT/m2_se_$i" $M2 mem -t"$T" -K $K "$IDX" "$R1") || { say "  [ABORT] SE mem2 rep$i failed fast (see $OUT/m2_se_$i.err)"; exit 1; }
  b=$(run "$OUT/m3_se_$i" $M3 mem -t"$T" -K $K "$IDX" "$R1") || { say "  [ABORT] SE mem3 rep$i failed fast (see $OUT/m3_se_$i.err)"; exit 1; }
  sp=$(echo "scale=3; $a/$b" | bc)
  say "  rep$i  mem2=${a}s  mem3=${b}s  speedup=${sp}x"
done
# identity from rep1
if diff -q "$OUT/m2_se_1.md5" "$OUT/m3_se_1.md5" >/dev/null && diff -q "$OUT/m2_se_1.n" "$OUT/m3_se_1.n" >/dev/null; then
  say "  [PASS] SE BYTE-IDENTICAL: $(cat "$OUT/m2_se_1.n") records, md5=$(cat "$OUT/m2_se_1.md5")"
else
  say "  [FAIL] SE differs: mem2 $(cat "$OUT/m2_se_1.n")rec/$(cat "$OUT/m2_se_1.md5")  vs  mem3 $(cat "$OUT/m3_se_1.n")rec/$(cat "$OUT/m3_se_1.md5")"
fi

########## PE ##########
say ""
say "==== PE  -t$T  x$PEREPS ===="
for i in $(seq 1 "$PEREPS"); do
  a=$(run "$OUT/m2_pe_$i" $M2 mem -t"$T" -K $K "$IDX" "$R1" "$R2") || { say "  [ABORT] PE mem2 rep$i failed fast (see $OUT/m2_pe_$i.err)"; exit 1; }
  b=$(run "$OUT/m3_pe_$i" $M3 mem -t"$T" -K $K "$IDX" "$R1" "$R2") || { say "  [ABORT] PE mem3 rep$i failed fast (see $OUT/m3_pe_$i.err)"; exit 1; }
  sp=$(echo "scale=3; $a/$b" | bc)
  say "  rep$i  mem2=${a}s  mem3=${b}s  speedup=${sp}x"
done
if diff -q "$OUT/m2_pe_1.md5" "$OUT/m3_pe_1.md5" >/dev/null && diff -q "$OUT/m2_pe_1.n" "$OUT/m3_pe_1.n" >/dev/null; then
  say "  [PASS] PE BYTE-IDENTICAL: $(cat "$OUT/m2_pe_1.n") records, md5=$(cat "$OUT/m2_pe_1.md5")"
else
  say "  [FAIL] PE differs: mem2 $(cat "$OUT/m2_pe_1.n")rec/$(cat "$OUT/m2_pe_1.md5")  vs  mem3 $(cat "$OUT/m3_pe_1.n")rec/$(cat "$OUT/m3_pe_1.md5")"
fi

say ""
say "GIAB30X_DONE  results in $LOG"
