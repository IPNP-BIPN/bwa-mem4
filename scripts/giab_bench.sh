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

echo "--- identity vs bwa-mem2 (SE, alignment records) ---"
$M2 mem -t"$T" -K $K "$IDX2" "$R1" 2>/dev/null > /tmp/giab_m2.sam
$M3 mem -t"$T" -K $K "$IDX3" "$R1" 2>/dev/null > /tmp/giab_m3.sam
# Headers legitimately differ (@PG carries each tool's name and command line), so identity is over
# the alignment records.
grep -v '^@' /tmp/giab_m2.sam > /tmp/giab_m2.rec
grep -v '^@' /tmp/giab_m3.sam > /tmp/giab_m3.rec
n2=$(wc -l < /tmp/giab_m2.rec | tr -d ' ')
n3=$(wc -l < /tmp/giab_m3.rec | tr -d ' ')

if [ "$n2" -ne "$n3" ]; then
  # This must come FIRST. Any positional comparison (paste, diff, cut+awk) lines record i of one
  # file up against record i of the other, so a single missing record shifts every later line and
  # the reported rate becomes meaningless -- it converges on 100%, which is exactly how the previous
  # version of this check produced "core-field diffs: 4025121 / 4025145 records (99.999404%)" and
  # was read as a 99.999% *match*.
  echo "  [FAIL] RECORD COUNT DIFFERS: bwa-mem2 $n2 vs bwa-mem3 $n3 (delta $((n3 - n2)))"
  echo "         This is itself a real divergence. Positional comparison suppressed: it could only"
  echo "         report a meaningless ~100% rate. Diagnose with:"
  echo "           comm -3 <(cut -f1,2 /tmp/giab_m2.rec|sort) <(cut -f1,2 /tmp/giab_m3.rec|sort) | head"
elif cmp -s /tmp/giab_m2.rec /tmp/giab_m3.rec; then
  echo "  [PASS] BYTE-IDENTICAL: $n3 records, cmp-clean"
else
  echo "  [FAIL] not byte-identical ($n3 records; counts match, so the comparisons below are valid)"
  # \x01 cannot occur in a SAM record, so it is a safe field separator for whole-line comparison.
  paste -d $'\x01' /tmp/giab_m2.rec /tmp/giab_m3.rec \
    | awk -F$'\x01' '{n++; if($1!=$2) d++}
        END{printf "         full-line differing (incl tags): %d / %d (%.6f%% differ)\n", d+0, n, 100*(d+0)/n}'
  paste <(cut -f1-6 /tmp/giab_m2.rec) <(cut -f1-6 /tmp/giab_m3.rec) \
    | awk -F'\t' '{n++; if($1!=$7||$2!=$8||$3!=$9||$4!=$10||$5!=$11||$6!=$12) d++}
        END{printf "         core fields (QNAME/FLAG/RNAME/POS/MAPQ/CIGAR) differing: %d / %d (%.6f%% differ)\n", d+0, n, 100*(d+0)/n}'
  echo "         first 3 core-field divergences:"
  paste <(cut -f1-6 /tmp/giab_m2.rec) <(cut -f1-6 /tmp/giab_m3.rec) \
    | awk -F'\t' '$1!=$7||$2!=$8||$3!=$9||$4!=$10||$5!=$11||$6!=$12 {
        printf "           mem2: %s %s %s %s %s %s\n           mem3: %s %s %s %s %s %s\n\n",
               $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12; if (++c==3) exit }'
fi

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
