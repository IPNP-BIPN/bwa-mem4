#!/usr/bin/env bash
# Real-data head-to-head: bwa-mem3 vs bwa-mem2 on GIAB HG002 (NIST HiSeq 300x, Sample_2A1).
# Our routine benches use wgsim-simulated reads; this validates that the speedup AND the
# byte-identity hold on real sequencing data (real error profiles, adapters, N's, duplicates).
#
# Interleaved A/B (alternating tools) so host memory-bandwidth drift cancels — the same
# methodology the rest of the perf work uses.
set -euo pipefail
cd "$(dirname "$0")/.."

M3=./target/release/bwa-mem3
M2=bwa-mem2
IDX3=work/genome.fa          # our index
IDX2=work/genome_oracle.fa   # bwa-mem2's (byte-identical index files)
R1=work/giab/HG002_R1.fastq
R2=work/giab/HG002_R2.fastq
K=100000000
REPS="${REPS:-3}"
T="${T:-8}"

t() { /usr/bin/time -p "$@" >/dev/null 2>/tmp/gt.log; grep '^real' /tmp/gt.log | awk '{print $2}'; }

echo "=== GIAB HG002 real reads: $(($(wc -l < "$R1")/4)) pairs ==="

echo "--- byte-identity (SE, core fields + full line) ---"
$M2 mem -t"$T" -K $K "$IDX2" "$R1" 2>/dev/null > /tmp/giab_m2.sam
$M3 mem -t"$T" -K $K "$IDX3" "$R1" 2>/dev/null > /tmp/giab_m3.sam
paste <(grep -v '^@' /tmp/giab_m2.sam | cut -f1-6) <(grep -v '^@' /tmp/giab_m3.sam | cut -f1-6) \
  | awk -F'\t' '{n++; if($1!=$7||$2!=$8||$3!=$9||$4!=$10||$5!=$11||$6!=$12) d++}
      END{printf "  core-field diffs: %d / %d records (%.6f%%)\n", d+0, n, 100*(d+0)/n}'
diff <(grep -v '^@' /tmp/giab_m2.sam) <(grep -v '^@' /tmp/giab_m3.sam) | grep -c '^<' \
  | awk '{printf "  full-line diffs (incl tags): %d\n", $1}' || true

for mode in SE PE; do
  echo "--- $mode -t$T (interleaved x$REPS) ---"
  for i in $(seq 1 "$REPS"); do
    if [ "$mode" = SE ]; then
      a=$(t $M2 mem -t"$T" -K $K "$IDX2" "$R1"); b=$(t $M3 mem -t"$T" -K $K "$IDX3" "$R1")
    else
      a=$(t $M2 mem -t"$T" -K $K "$IDX2" "$R1" "$R2"); b=$(t $M3 mem -t"$T" -K $K "$IDX3" "$R1" "$R2")
    fi
    echo "  mem2=${a}s  mem3=${b}s  speedup=$(echo "scale=2; $a/$b" | bc)x"
  done
done
echo "GIAB_BENCH_DONE"
