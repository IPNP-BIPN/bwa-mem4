#!/usr/bin/env bash
# Three-arm head-to-head at iteration scale: bwa-mem2 2.3 (oracle), fg-labs/bwa-mem3 (@nh13's C++
# fork), and us. Wall time, peak RSS and batch count per arm, plus the md5 of the alignment records
# so identity is checked on the same pass that is timed.
#
# Usage: scripts/fork_bench.sh [se|pe] [reps]
#   T=8 K=10000000 READS=work/r1_4m.fq scripts/fork_bench.sh se 3
#
# Method rules, each one paid for by a past error in this repo:
#   - arms are interleaved WITHIN a rep, never run as three separate blocks. Repeated identical
#     runs spread ~2.4%, and numbers taken minutes apart are worthless.
#   - every binary is warmed once before the first timed rep, on the same index and reads. Cold
#     starts have produced a "13.89x prefetch speedup" here whose real value was 1.02x, and an
#     "11.76x binning win" that was a loop warming the cache.
#   - the genome index only. region.fa's 2 Mbp BWT is cache-resident and hides seeding, which is
#     ~78% of the real profile.
#   - our arm should be the PGO binary (scripts/pgo.sh). A cargo build --release is ~15% slower and
#     is not what we ship, so timing it would flatter the fork by 15%.
#   - nothing under 3% is a gain. Host noise is ~2.4%.
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-se}"
REPS="${2:-3}"
T="${T:-8}"
K="${K:-10000000}"
IDX="${IDX:-work/genome.fa}"
M2="${M2:-bwa-mem2}"
FORK="${FORK:-reference/bwa-mem3-cpp/bwa-mem3.arm64}"
M4="${M4:-./target/release/bwa-mem4}"

case "$MODE" in
  se) READ_FILES=("${READS:-work/r1_4m.fq}") ;;
  pe) READ_FILES=("${READS:-work/r1_4m.fq}" "${READS2:-work/r2_4m.fq}") ;;
  *) echo "mode must be se|pe" >&2; exit 1 ;;
esac

for f in "${READ_FILES[@]}" "$IDX.bwt.2bit.64" "$FORK" "$M4"; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
done
command -v "$M2" >/dev/null || { echo "missing $M2 on PATH (see scripts/setup_reference.sh)" >&2; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT="work/forkbench/$TS"; mkdir -p "$OUT"
LOG="$OUT/results.log"
say() { echo "$@" | tee -a "$LOG"; }

# One timed run. $1=arm label, $2=rep, rest=command. Echoes "wall_s rss_mb batches".
# /usr/bin/time -l yields both wall seconds and peak RSS in bytes on macOS. The record stream goes
# through md5 on the timed pass, so identity costs no extra run.
run() {
  local arm="$1" rep="$2"; shift 2
  local of="$OUT/${arm}_${MODE}_${rep}"
  # The `sed` strips the fork's extra `HN:i:N` tag, which it appends to EVERY record and which
  # neither bwa-mem2 nor we emit. Without it the fork trivially "differs" on every line and the
  # identity check says nothing. It runs on all three arms, not just the fork, so the pipelines
  # stay symmetric and no arm carries a cost its rivals do not: it is a no-op on the other two.
  /usr/bin/time -l bash -c "$* 2>'$of.err' | grep -v '^@' | sed 's/\tHN:i:[0-9]*//' | tee >(wc -l >'$of.n') | md5 >'$of.md5'" 2>"$of.time"
  local real rssb nb
  real=$(awk '/ real /{print $1}' "$of.time" | head -1)
  rssb=$(awk '/maximum resident set size/{print $1}' "$of.time")
  # Only our arm reports its batch count; the other two show "-".
  nb=$(sed -n 's/.*processed \([0-9]*\) batches.*/\1/p' "$of.err" | tail -1)
  [ -n "$nb" ] || nb="-"
  echo "$real $(( rssb / 1048576 )) $nb"
}

say "############################################################"
say "# fork_bench  $TS   mode=$MODE  -t$T  -K $K  reps=$REPS"
say "# idx=$IDX  reads=${READ_FILES[*]}"
say "# bwa-mem4=$(git rev-parse --short HEAD)"
say "############################################################"

CMD_M2="$M2 mem -t$T -K $K $IDX ${READ_FILES[*]}"
CMD_FORK="$FORK mem -t$T -K $K $IDX ${READ_FILES[*]}"
CMD_M4="$M4 mem -t$T -K $K $IDX ${READ_FILES[*]}"

say "warming all three binaries (untimed)..."
for c in "$CMD_M2" "$CMD_FORK" "$CMD_M4"; do bash -c "$c" >/dev/null 2>&1; done

declare -a w_m2 w_fk w_m4 r_m2 r_fk r_m4
for i in $(seq 1 "$REPS"); do
  read -r a_w a_r _a_b <<<"$(run m2   "$i" "$CMD_M2")"
  read -r f_w f_r _f_b <<<"$(run fork "$i" "$CMD_FORK")"
  read -r o_w o_r  o_b <<<"$(run m4   "$i" "$CMD_M4")"
  w_m2+=("$a_w"); r_m2+=("$a_r")
  w_fk+=("$f_w"); r_fk+=("$f_r")
  w_m4+=("$o_w"); r_m4+=("$o_r")
  say "  rep$i  mem2=${a_w}s/${a_r}MB  fork=${f_w}s/${f_r}MB  mem4=${o_w}s/${o_r}MB (batches=$o_b)"
done

med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }
MW2=$(med "${w_m2[@]}"); MWF=$(med "${w_fk[@]}"); MW4=$(med "${w_m4[@]}")
MR2=$(med "${r_m2[@]}"); MRF=$(med "${r_fk[@]}"); MR4=$(med "${r_m4[@]}")

say ""
say "| arm | wall s (median) | peak RSS MB | vs bwa-mem2 |"
say "|---|---|---|---|"
say "| bwa-mem2 2.3 | $MW2 | $MR2 | 1.00x |"
say "| fg-labs/bwa-mem3 | $MWF | $MRF | $(echo "scale=2; $MW2/$MWF" | bc)x |"
say "| bwa-mem4 | $MW4 | $MR4 | $(echo "scale=2; $MW2/$MW4" | bc)x |"
say ""
say "us vs fork: $(echo "scale=3; $MWF/$MW4" | bc)x wall, $(echo "scale=3; $MRF/$MR4" | bc)x RSS  (>1 means we win)"

# ---- Identity, from rep 1 ----
if diff -q "$OUT/m2_${MODE}_1.md5" "$OUT/m4_${MODE}_1.md5" >/dev/null 2>&1 \
   && diff -q "$OUT/m2_${MODE}_1.n" "$OUT/m4_${MODE}_1.n" >/dev/null 2>&1; then
  say "[PASS] bwa-mem4 BYTE-IDENTICAL to bwa-mem2: $(cat "$OUT/m2_${MODE}_1.n") records"
else
  say "[FAIL] bwa-mem4 differs: mem2 $(cat "$OUT/m2_${MODE}_1.n")rec vs mem4 $(cat "$OUT/m4_${MODE}_1.n")rec"
fi
if diff -q "$OUT/m2_${MODE}_1.md5" "$OUT/fork_${MODE}_1.md5" >/dev/null 2>&1; then
  say "[note] the fork is byte-identical to bwa-mem2 too, once its HN:i tag is stripped"
else
  say "[note] the fork DIFFERS from bwa-mem2 beyond its HN:i tag ($(cat "$OUT/fork_${MODE}_1.n") records)"
fi

# ---- The -K trap ----
NB=$(sed -n 's/.*processed \([0-9]*\) batches.*/\1/p' "$OUT/m4_${MODE}_1.err" | tail -1)
if [ -n "$NB" ] && [ "$NB" -lt 4 ] 2>/dev/null; then
  say "WARNING: only $NB batches. The reader/writer pipeline is inert below ~4 batches and this"
  say "         run understates bwa-mem4 by 8-9%. Lower -K or use more reads."
fi
say "FORK_BENCH_DONE  $LOG"
