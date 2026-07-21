#!/usr/bin/env bash
# Interleaved head-to-head bwa-mem2 vs bwa-mem4 (drift-cancelled). SE/PE x t1/t8.
set -euo pipefail
cd "$(dirname "$0")/.."
M2IDX=work/genome_oracle.fa
M3IDX=work/genome.fa
R1=work/r1_500k.fq
R2=work/r2_500k.fq
OURS=./target/release/bwa-mem4

t() { /usr/bin/time -p "$@" >/dev/null 2>/tmp/vt.log; grep '^real' /tmp/vt.log | awk '{print $2}'; }

echo "warming..."
bwa-mem2 mem -t1 "$M2IDX" work/r1.fq >/dev/null 2>&1
"$OURS" mem -t1 "$M3IDX" work/r1.fq >/dev/null 2>&1

for cfg in "SE t1:-t1:$R1" "PE t1:-t1:$R1 $R2" "SE t8:-t8:$R1" "PE t8:-t8:$R2 $R2"; do
  label="${cfg%%:*}"; rest="${cfg#*:}"; threads="${rest%%:*}"; reads="${rest#*:}"
  # fix PE t8 reads (typo guard): rebuild reads list explicitly
  case "$label" in
    "SE t1"|"SE t8") reads="$R1" ;;
    "PE t1"|"PE t8") reads="$R1 $R2" ;;
  esac
  echo "=== $label ==="
  for i in 1 2 3; do
    a=$(t bwa-mem2 mem $threads -K 100000000 "$M2IDX" $reads)
    b=$(t "$OURS" mem $threads -K 100000000 "$M3IDX" $reads)
    echo "  mem2=$a  mem3=$b  speedup=$(echo "scale=2; $a/$b" | bc)x"
  done
done
echo "DONE"
