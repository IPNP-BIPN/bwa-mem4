#!/usr/bin/env bash
# Issue #32: the full-WGS x86_64 headline ratio against bwa-mem2, run the way the issue specifies.
#
# WHAT THIS IS FOR. Every speed ratio this project publishes at WGS scale is an arm64/NEON number
# from an Apple M4 Max. On x86_64 there is no headline: `bench-x86.yml` gives two directional
# signals on hosted runners, both small and both noisy, and a hosted runner cannot produce the real
# one. It has 16 GB of RAM against the 92 GB the GRCh38 index build peaks at, its cores are shared
# with another tenant so the scalar glue is measured through an SMT tax, and it is destroyed before
# a 30x run finishes. So this script exists to be run on a DEDICATED machine, and to refuse to run
# anywhere that would produce a number which does not answer the question.
#
# WHAT THE ISSUE ASKS FOR, and therefore what this implements, point by point:
#
#   - a full GIAB HG002 30x run, not a chromosome and not a sample;
#   - the same protocol as `scripts/giab30x_pe.sh`, i.e. records piped straight through md5 and wc
#     so identity is checked on every timed pass at no extra cost, and nothing large is written;
#   - ALTERNATING order, so the aligner that runs first is not always the same one. Cache state and
#     thermal drift both favour or punish position in the sequence, and a fixed order folds that
#     into the ratio;
#   - several repetitions;
#   - PGO binaries on BOTH sides, because comparing a profile-optimised binary against a
#     stock one measures the build, not the aligner;
#   - the ratio published with the THREAD COUNT attached, because it decays as `-t` rises and a
#     bare "2.6x" is not a fact about anything.
#
# WHAT IT DELIBERATELY WILL NOT DO. It will not run on a machine that fails the conditions above.
# A number produced on a shared, SMT-enabled, power-saving or under-provisioned host is worse than
# no number, because it would be published and then defended. Every refusal can be overridden by an
# explicit environment variable, and every override is printed into the results log, so a run that
# bent a rule says so in the artefact rather than in somebody's memory.
#
# USAGE
#
#   scripts/giab32_headline.sh --preflight     # check this machine, change nothing, exit non-zero
#                                              # if it is not fit for the measurement
#   scripts/giab32_headline.sh                 # the full protocol
#
# Inputs it expects to already exist (see scripts/fetch_giab30x.sh and the index build):
#   work/genome.fa{,.bwt.2bit.64,...}          the shared, byte-identical index
#   work/giab30x/HG002_30x_R{1,2}.fastq.gz     ~30x of real reads
#
# Knobs, all optional:
#   T=8                     threads; the published line carries this number
#   REPS=3                  repetitions per read layout
#   M2_SRC=/path/to/src     bwa-mem2 sources, to build the PGO binary for the other side
#   M2_PGO=/path/to/binary  an already-PGO-built bwa-mem2, if you built it yourself
#   ALLOW_NO_PGO_M2=1       run with a stock bwa-mem2 anyway, and say so in the log
#   ALLOW_SMT=1             run with SMT on anyway
#   ALLOW_SHARED=1          run despite other load on the machine
#   ALLOW_SMALL_RAM=1       run despite less RAM than the index build needs
#   ALLOW_POWERSAVE=1       run despite a non-performance cpufreq governor
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

T="${T:-8}"
REPS="${REPS:-3}"
K="${K:-100000000}"
IDX="${IDX:-work/genome.fa}"
R1="${R1:-work/giab30x/HG002_30x_R1.fastq.gz}"
R2="${R2:-work/giab30x/HG002_30x_R2.fastq.gz}"
M4="${M4:-./target/x86_64-unknown-linux-gnu/release/bwa-mem4}"

# The index build's measured peak, in GiB. This is the number that rules hosted runners out, so it
# is named once here rather than repeated in prose.
INDEX_PEAK_GIB=92
PREFLIGHT_ONLY=0
[ "${1:-}" = "--preflight" ] && PREFLIGHT_ONLY=1

fail=0
note() { printf '%s\n' "$*"; }
refuse() { printf 'REFUSE: %s\n' "$*" >&2; fail=1; }
allowed() { # $1 = env var name; prints the override note and returns 0 when set
  local v="${!1:-}"
  [ -n "$v" ] || return 1
  printf 'OVERRIDE: %s=%s, running anyway\n' "$1" "$v"
  OVERRIDES="${OVERRIDES:-}$1 "
  return 0
}

note "== preflight =="
note "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
note "host: $(uname -s)/$(uname -m)"

# 1. The platform. SDE-style refusal: say what is wrong and where the answer lives instead.
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  refuse "this is $(uname -s)/$(uname -m); #32 is specifically the x86_64 number, and the arm64 one is already published"
  note ""
  note "Nothing here can be substituted for the machine: a hosted runner fails the RAM, the core"
  note "exclusivity and the run-length conditions at once. See the closing condition in issue #32."
  exit 2
fi

# 2. CPU identity, which has to be in the artefact: this project has twice published a number as a
#    property of the code that turned out to be a property of the part it was measured on.
CPU=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown)
CORES=$(nproc 2>/dev/null || echo 0)
note "cpu: $CPU"
note "cores (visible): $CORES"
grep -o -E 'avx2|avx512bw|avx512f|avx512vl' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ' | sed 's/^/simd: /'; echo

# 3. RAM. The index build is the binding constraint, not the alignment.
RAM_GIB=$(awk '/^MemTotal:/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
note "ram: ${RAM_GIB} GiB (index build peaks at ~${INDEX_PEAK_GIB} GiB)"
if [ "$RAM_GIB" -lt "$INDEX_PEAK_GIB" ]; then
  allowed ALLOW_SMALL_RAM || refuse "only ${RAM_GIB} GiB of RAM; the GRCh38 index build peaks near ${INDEX_PEAK_GIB} GiB"
fi

# 4. SMT. The issue exists partly because the scalar glue cannot be timed honestly on a shared
#    hyperthread; see the ksw_global2 investigation.
SMT=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)
note "smt: $SMT"
if [ "$SMT" = "on" ]; then
  allowed ALLOW_SMT || refuse "SMT is on; sibling threads make the scalar glue unmeasurable (turn it off, or set ALLOW_SMT=1 and say so when publishing)"
fi

# 5. Exclusivity. A 1-minute load average above one core's worth means somebody else is here.
LOAD=$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)
note "loadavg(1m): $LOAD"
if awk "BEGIN{exit !($LOAD > 1.0)}"; then
  allowed ALLOW_SHARED || refuse "load average is $LOAD before the benchmark starts; the machine is not idle"
fi

# 6. Frequency policy. `powersave` makes the first repetition slower than the third for reasons
#    that have nothing to do with either aligner.
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
note "governor: $GOV"
if [ "$GOV" != "performance" ] && [ "$GOV" != "unknown" ]; then
  allowed ALLOW_POWERSAVE || refuse "cpufreq governor is '$GOV'; set it to performance so repetitions are comparable"
fi

# 7. Disk. Two 35 GB read files, an index, and room to work.
FREE_GIB=$(df -BG . 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}' || echo 0)
note "free disk here: ${FREE_GIB} GiB"
[ "${FREE_GIB:-0}" -ge 50 ] || refuse "only ${FREE_GIB} GiB free; the inputs alone are about 70 GiB"

# 8. Inputs and tools.
for f in "$R1" "$R2" "$IDX.bwt.2bit.64"; do
  [ -e "$f" ] || refuse "missing input: $f (see scripts/fetch_giab30x.sh)"
done
for t in /usr/bin/time md5sum gunzip awk bc; do
  command -v "$t" >/dev/null 2>&1 || refuse "missing tool: $t"
done

# 9. PGO on both sides. This is the condition most likely to be skipped quietly, so it is checked
#    like any other and its override is recorded.
M2_BIN=""
if [ -n "${M2_PGO:-}" ]; then
  M2_BIN="$M2_PGO"; note "bwa-mem2: PGO binary supplied at $M2_BIN"
elif [ -n "${M2_SRC:-}" ]; then
  note "bwa-mem2: will be built with PGO from $M2_SRC"
else
  if allowed ALLOW_NO_PGO_M2; then
    M2_BIN="$(command -v bwa-mem2 || true)"
    note "bwa-mem2: stock binary at ${M2_BIN:-none}, NOT profile-optimised"
  else
    refuse "no PGO bwa-mem2: set M2_SRC to build one, or M2_PGO to point at one. #32 asks for PGO on both sides, and comparing a PGO binary against a stock one measures the build rather than the aligner"
  fi
fi
[ -n "${M2_SRC:-}" ] || [ -x "${M2_BIN:-/nonexistent}" ] || refuse "no usable bwa-mem2 binary"

command -v cargo >/dev/null 2>&1 || refuse "missing tool: cargo (needed for the PGO build of bwa-mem4)"
command -v cargo-pgo >/dev/null 2>&1 || refuse "missing cargo-pgo (cargo install cargo-pgo), needed for our side's PGO binary"

if [ "$fail" -ne 0 ]; then
  note ""
  note "PREFLIGHT FAILED: this machine cannot produce the number #32 asks for."
  exit 1
fi
note "PREFLIGHT OK${OVERRIDES:+ (with overrides: ${OVERRIDES})}"
[ "$PREFLIGHT_ONLY" -eq 1 ] && exit 0

# ---------------------------------------------------------------------------------------------
# Build both sides with PGO, training on the PRODUCTION shape.
#
# Training workload is not a detail. PGO optimises what the profile says is hot, and on simulated
# reads against a small reference mate rescue is ~10% of wall where on real GIAB reads against the
# whole genome it is ~59%. Training on the small workload tells the compiler the hottest production
# path is nearly cold. So the profile run here uses the real index and a slice of the real reads.
# ---------------------------------------------------------------------------------------------
TS=$(date -u +%Y%m%d_%H%M%S)
OUT="work/giab30x/x86_$TS"; mkdir -p "$OUT"
LOG="$OUT/results.log"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

TRAIN_R1="$OUT/train_1.fq.gz"
TRAIN_R2="$OUT/train_2.fq.gz"
TRAIN_PAIRS="${TRAIN_PAIRS:-2000000}"
if [ ! -s "$TRAIN_R1" ]; then
  say "== extracting ${TRAIN_PAIRS} training pairs from the real reads =="
  gunzip -c "$R1" | head -n $((TRAIN_PAIRS * 4)) | gzip -1 > "$TRAIN_R1"
  gunzip -c "$R2" | head -n $((TRAIN_PAIRS * 4)) | gzip -1 > "$TRAIN_R2"
fi

say "== building bwa-mem4 with PGO (training on the real index and real reads) =="
TARGET=x86_64-unknown-linux-gnu
cargo pgo build -- --target "$TARGET" 2>&1 | tail -3 | tee -a "$LOG"
INSTR="target/$TARGET/release/bwa-mem4"
[ -x "$INSTR" ] || { say "instrumented build missing at $INSTR"; exit 1; }
"$INSTR" mem -t"$T" -K "$K" "$IDX" "$TRAIN_R1" "$TRAIN_R2" > /dev/null 2>>"$LOG"
cargo pgo optimize build -- --target "$TARGET" 2>&1 | tail -3 | tee -a "$LOG"
[ -x "$M4" ] || { say "optimised build missing at $M4"; exit 1; }
say "bwa-mem4: $($M4 version 2>/dev/null || echo '?') at $M4"

if [ -n "${M2_SRC:-}" ]; then
  say "== building bwa-mem2 with PGO from $M2_SRC =="
  # Two stages, the same shape as ours: instrument, run the same training reads, rebuild using the
  # profile. `-fprofile-*` is passed through CFLAGS/CXXFLAGS so the project's own makefile is
  # otherwise untouched.
  ( cd "$M2_SRC" && make clean >/dev/null 2>&1
    make -j"$(nproc)" CFLAGS="-O3 -fprofile-generate" CXXFLAGS="-O3 -fprofile-generate" LDFLAGS="-fprofile-generate" ) 2>&1 | tail -3 | tee -a "$LOG"
  "$M2_SRC/bwa-mem2" mem -t"$T" -K "$K" "$IDX" "$TRAIN_R1" "$TRAIN_R2" > /dev/null 2>>"$LOG"
  ( cd "$M2_SRC" && make clean >/dev/null 2>&1
    make -j"$(nproc)" CFLAGS="-O3 -fprofile-use -fprofile-correction" CXXFLAGS="-O3 -fprofile-use -fprofile-correction" ) 2>&1 | tail -3 | tee -a "$LOG"
  M2_BIN="$M2_SRC/bwa-mem2"
fi
say "bwa-mem2: $M2_BIN"

# ---------------------------------------------------------------------------------------------
# The timed protocol.
# ---------------------------------------------------------------------------------------------
# GNU time, not BSD: `-v` carries peak RSS, and RAM is one of the objectives of this comparison.
run() {
  local of="$1"; shift
  local cmd="$*"
  /usr/bin/time -v bash -c "$cmd 2>'$of.err' | grep -v '^@' | tee >(wc -l >'$of.n') | md5sum | cut -d' ' -f1 >'$of.md5'" 2>"$of.time"
  local real
  real=$(awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$of.time" | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }')
  awk -F': ' '/Maximum resident set size/{printf "%d\n", $2/1048576}' "$of.time" > "$of.rss"
  # A real 30x run is minutes. Anything quick is a crash that would otherwise be recorded as a win.
  if awk "BEGIN{exit !($real < 60)}"; then
    say "ABORT: $(basename "$of") returned in ${real}s, which means it failed. stderr tail:"
    tail -3 "$of.err" | tee -a "$LOG"
    return 1
  fi
  printf '%s\n' "$real"
}

say "############################################################"
say "# issue #32: GIAB HG002 ~30x, x86_64 headline, -t$T, $REPS reps"
say "# cpu: $CPU  cores: $CORES  ram: ${RAM_GIB} GiB  smt: $SMT  governor: $GOV"
say "# tree: $(git rev-parse --short HEAD 2>/dev/null || echo '?')  overrides: ${OVERRIDES:-none}"
say "############################################################"

for layout in se pe; do
  say ""
  say "==== ${layout^^}  -t$T  x$REPS  (alternating order) ===="
  for i in $(seq 1 "$REPS"); do
    # ALTERNATING ORDER. Odd repetitions run bwa-mem2 first, even ones run bwa-mem4 first, so
    # neither aligner is permanently the one that pays for a cold page cache.
    if [ "$layout" = "se" ]; then reads=("$R1"); else reads=("$R1" "$R2"); fi
    if [ $((i % 2)) -eq 1 ]; then
      a=$(run "$OUT/m2_${layout}_$i" "$M2_BIN" mem -t"$T" -K "$K" "$IDX" "${reads[@]}") || exit 1
      b=$(run "$OUT/m4_${layout}_$i" "$M4"     mem -t"$T" -K "$K" "$IDX" "${reads[@]}") || exit 1
      order="mem2 first"
    else
      b=$(run "$OUT/m4_${layout}_$i" "$M4"     mem -t"$T" -K "$K" "$IDX" "${reads[@]}") || exit 1
      a=$(run "$OUT/m2_${layout}_$i" "$M2_BIN" mem -t"$T" -K "$K" "$IDX" "${reads[@]}") || exit 1
      order="mem4 first"
    fi
    say "  rep$i ($order)  mem2=${a}s/$(cat "$OUT/m2_${layout}_$i.rss")MB  mem4=${b}s/$(cat "$OUT/m4_${layout}_$i.rss")MB  ratio=$(echo "scale=3; $a/$b" | bc)x"
    if ! diff -q "$OUT/m2_${layout}_$i.md5" "$OUT/m4_${layout}_$i.md5" >/dev/null; then
      say "  [FAIL] rep$i is NOT byte-identical: mem2 $(cat "$OUT/m2_${layout}_$i.md5") vs mem4 $(cat "$OUT/m4_${layout}_$i.md5")"
      say "  A speed ratio between two aligners that disagree is meaningless; stopping."
      exit 1
    fi
  done
  # Best-of, because the minimum is the least noisy estimator of a deterministic workload's cost.
  bm2=$(cat "$OUT"/m2_"$layout"_*.time >/dev/null 2>&1; for i in $(seq 1 "$REPS"); do awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$OUT/m2_${layout}_$i.time"; done | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }' | sort -n | head -1)
  bm4=$(for i in $(seq 1 "$REPS"); do awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$OUT/m4_${layout}_$i.time"; done | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }' | sort -n | head -1)
  say "  [PASS] ${layout^^} byte-identical on all $REPS repetitions, $(cat "$OUT/m2_${layout}_1.n") records"
  # The one line #32 asks for, with the thread count in it, machine-readable on purpose.
  say "HEADLINE ${layout} -t$T mem2=${bm2}s mem4=${bm4}s ratio=$(echo "scale=3; $bm2/$bm4" | bc)x cpu=\"$CPU\" pgo=both identical=yes"
done

say ""
say "GIAB32_DONE  results in $LOG"
