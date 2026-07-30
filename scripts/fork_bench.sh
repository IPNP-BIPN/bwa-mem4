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
# `-K` batch size. Set `K=default` to omit the flag entirely and let every binary use bwa's own
# default of `10M * threads`, which is what the gist's benchmark does and what the standing number
# in docs/perf-levers.md is measured at:
#
#   T=16 K=default READS=work/giab_small/r1_1m.fq.gz READS2=work/giab_small/r2_1m.fq.gz \
#     IDX=work/genome.fa scripts/fork_bench.sh pe 6
#
# Keeping an explicit `-K` is still the right choice when comparing two builds of ONE binary, since
# it pins the batch boundaries and therefore the output.
K="${K:-10000000}"
IDX="${IDX:-work/genome.fa}"
M2="${M2:-bwa-mem2}"
# The in-tree arm64 build if it is there (the macOS dev host), else whatever `bwa-mem3` is on PATH
# (a Linux benchmark box, where it comes from bioconda). Override with FORK=.
if [ -z "${FORK:-}" ]; then
  if [ -x reference/bwa-mem3-cpp/bwa-mem3.arm64 ]; then
    FORK=reference/bwa-mem3-cpp/bwa-mem3.arm64
  else
    FORK=$(command -v bwa-mem3 || echo reference/bwa-mem3-cpp/bwa-mem3.arm64)
  fi
fi
M4="${M4:-./target/release/bwa-mem4}"

# ---- Portability: this script was written on macOS and has to run on the Linux benchmark box ----
# Two BSD-only tools are involved. Probing beats uname: a Mac with GNU coreutils first on PATH, or a
# Linux box without GNU time, both get the right answer this way.
#   * `/usr/bin/time`: BSD spells the verbose form `-l` and reports peak RSS in BYTES; GNU spells it
#     `-f FORMAT` and reports it in KILOBYTES. Both are parsed into bytes below.
#   * `md5` vs `md5sum`: only ever compared against each other, so the differing output format
#     (bare hash vs "hash  -") is immaterial as long as one tool is used for every arm.
if /usr/bin/time -l true >/dev/null 2>&1; then
  TIME_KIND=bsd
elif /usr/bin/time -f '%e %M' true >/dev/null 2>&1; then
  TIME_KIND=gnu
else
  echo "need /usr/bin/time supporting BSD -l or GNU -f" >&2; exit 1
fi
MD5=$(command -v md5 || command -v md5sum) || { echo "need md5 or md5sum" >&2; exit 1; }

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
  local inner="$* 2>'$of.err' | grep -v '^@' | sed 's/\tHN:i:[0-9]*//' | tee >(wc -l >'$of.n') | $MD5 >'$of.md5'"
  local real rssb nb
  if [ "$TIME_KIND" = bsd ]; then
    /usr/bin/time -l bash -c "$inner" 2>"$of.time"
    real=$(awk '/ real /{print $1}' "$of.time" | head -1)
    rssb=$(awk '/maximum resident set size/{print $1}' "$of.time")
  else
    # GNU: one line, "<wall_seconds> <peak_rss_kilobytes>". Scaled to bytes so the caller's
    # `/ 1048576` gives MB on both platforms.
    /usr/bin/time -f '%e %M' bash -c "$inner" 2>"$of.time"
    real=$(awk 'NF==2{w=$1; r=$2} END{print w}' "$of.time")
    rssb=$(awk 'NF==2{w=$1; r=$2} END{print r*1024}' "$of.time")
  fi
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

# `K=default` means "pass no -K at all", i.e. each binary picks bwa's `10M * threads`.
if [ "$K" = "default" ]; then KFLAG=""; else KFLAG="-K $K"; fi
CMD_M2="$M2 mem -t$T $KFLAG $IDX ${READ_FILES[*]}"
CMD_FORK="$FORK mem -t$T $KFLAG $IDX ${READ_FILES[*]}"
CMD_M4="$M4 mem -t$T $KFLAG $IDX ${READ_FILES[*]}"

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
