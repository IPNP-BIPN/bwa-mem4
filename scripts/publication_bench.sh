#!/usr/bin/env bash
# Full publication benchmark battery: precision (byte-identity) + speed vs bwa-mem2.
# Runs the four existing gates end-to-end and tees everything to a timestamped results dir.
# Each phase prints a PHASE_DONE marker so a monitor can track progress.
set -uo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
OUT="work/pubbench_$TS"
mkdir -p "$OUT"
LOG="$OUT/results.log"

M3=./target/release/bwa-mem4
say() { echo "$@" | tee -a "$LOG"; }

say "############################################################"
say "# bwa-mem4 publication benchmark  $TS"
say "# host: $(uname -mnrs)"
say "# binary: $(git rev-parse --short HEAD) ($(git log -1 --format=%s | cut -c1-60))"
say "############################################################"

########################################################################
say ""
say "==== PHASE 1: INDEX BYTE-IDENTITY (precision foundation) ===="
say "[gate] full-genome index, ours vs bwa-mem2 oracle"
IDX_OK=1
for ext in pac ann amb bwt.2bit.64 0123; do
  if cmp -s "work/genome.fa.$ext" "work/genome_oracle.fa.$ext"; then
    say "  .$ext: IDENTICAL"
  else
    say "  .$ext: DIFFER"; IDX_OK=0
  fi
done
[ "$IDX_OK" = 1 ] && say "  [PASS] index byte-identical" || say "  [FAIL] index diverged"
say "PHASE_DONE 1 index-identity"

########################################################################
say ""
say "==== PHASE 2: GIAB HG002 REAL DATA (precision + speed) ===="
REPS=3 T=8 bash scripts/giab_bench.sh 2>&1 | tee -a "$LOG"
say "PHASE_DONE 2 giab"

########################################################################
say ""
say "==== PHASE 3: vs bwa-mem2 GENOME-WIDE 500k (speed SE/PE t1/t8) ===="
bash scripts/vs_mem2.sh 2>&1 | tee -a "$LOG"
say "PHASE_DONE 3 vs_mem2"

########################################################################
say ""
say "==== PHASE 4: REGION MICRO-BENCH -t1 (kernel wall + RSS) ===="
say "-- SE --"
bash scripts/bench.sh "$M3" se 3 2>&1 | tee -a "$LOG"
say "-- PE --"
bash scripts/bench.sh "$M3" pe 3 2>&1 | tee -a "$LOG"
say "PHASE_DONE 4 region"

say ""
say "ALL_PHASES_DONE  results in $OUT/results.log"
