#!/usr/bin/env bash
# Reproducible perf harness: median-of-3 wall-clock + peak RSS for a given binary on the
# canonical perf workload (région 2 Mbp index, 500k reads), single-threaded (-t1).
#
# Usage: scripts/bench.sh <binary> [se|pe] [reps]
#   scripts/bench.sh target/release/bwa-mem3 se 3
# Prints: "<mode> median_wall_s=<x> peak_rss_mb=<y>" plus the raw per-rep numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:?usage: bench.sh <binary> [se|pe] [reps]}"
MODE="${2:-se}"
REPS="${3:-3}"
K="${K:-100000000}"
IDX="work/region.fa"

case "$MODE" in
  se) READS=(work/r1_500k.fq) ;;
  pe) READS=(work/r1_500k.fq work/r2_500k.fq) ;;
  *) echo "mode must be se|pe" >&2; exit 1 ;;
esac
for f in "${READS[@]}"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done

walls=(); rss=()
for r in $(seq 1 "$REPS"); do
  out=$(/usr/bin/time -l "$BIN" mem -t1 -K "$K" "$IDX" "${READS[@]}" 2>&1 >/dev/null)
  # BSD /usr/bin/time -l: "<real> real ..." line + "<bytes> maximum resident set size"
  w=$(printf '%s\n' "$out" | awk '/ real /{print $1; exit}')
  m=$(printf '%s\n' "$out" | awk '/maximum resident set size/{print $1; exit}')
  walls+=("$w"); rss+=("$m")
  printf '  rep%s wall=%ss rss=%.0fMB\n' "$r" "$w" "$(echo "$m/1048576" | bc -l)"
done
# median (reps small; sort + middle)
med=$(printf '%s\n' "${walls[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
maxrss=$(printf '%s\n' "${rss[@]}" | sort -n | tail -1)
printf '%s median_wall_s=%s peak_rss_mb=%.0f\n' "$MODE" "$med" "$(echo "$maxrss/1048576" | bc -l)"
