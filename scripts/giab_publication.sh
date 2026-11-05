#!/usr/bin/env bash
# Publication benchmark: bwa-mem4 vs bwa-mem2 on real GIAB HG002 reads, GRCh38.
#
# Protocol (everything a reviewer needs is written to $OUT):
#   0. provenance: host, OS, compilers, binary sha256, git commit, input checksums
#   1. index: each tool builds its own index from the same FASTA (timed, peak RSS), files cmp'd
#   2. warm-up: one discarded SE subset run per tool, so both start with a warm page cache
#   3. subset: SUB_REPS interleaved reps (ABBA order) of SE and PE, for variance
#   4. full sample: FULL_REPS reps of SE and PE on the whole ~33x run
# Every run: wall/user/sys/peak RSS via /usr/bin/time -l, records (no header) md5 + count, so
# byte-identity is checked on every single run. Each run waits for a quiet host first and logs
# the load average it started at; a run that started above QUIET (wait timed out) is flagged.
#
# Usage: DATA=/path/to/work/giab scripts/giab_publication.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

DATA="${DATA:-work/giab}"
M4="${M4:-$ROOT/target/release/bwa-mem4}"
M2="${M2:-$(command -v bwa-mem2)}"
FASTA="${FASTA:-$DATA/genome.fa}"
SUB1="${SUB1:-$DATA/g2m_1.fq}"; SUB2="${SUB2:-$DATA/g2m_2.fq}"
FULL1="${FULL1:-$DATA/giab30x/HG002_30x_R1.fastq.gz}"; FULL2="${FULL2:-$DATA/giab30x/HG002_30x_R2.fastq.gz}"
T="${T:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc)}"   # all performance cores
K=100000000            # fixed batch size: output is then independent of -t for both tools
SUB_REPS="${SUB_REPS:-5}"; FULL_REPS="${FULL_REPS:-1}"
QUIET="${QUIET:-4}"    # 1-min load average a run waits for before starting (an idle desktop sits at 2-4)
SKIP_FULL="${SKIP_FULL:-0}"

for f in "$M4" "$M2" "$FASTA" "$SUB1" "$SUB2"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
[ "$SKIP_FULL" = 1 ] || for f in "$FULL1" "$FULL2"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done

OUT="${OUT:-$DATA/publication_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
LOG="$OUT/results.log"; TSV="$OUT/runs.tsv"; cp "$0" "$OUT/"
say() { echo "$@" | tee -a "$LOG"; }
printf "phase\tmode\ttool\trep\twall_s\tuser_s\tsys_s\trss_gb\trecords\tmd5\tload_max\tload_start\n" >"$TSV"

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
wait_quiet() {
  local i=0
  while awk -v l="$(load1)" -v q="$QUIET" 'BEGIN{exit !(l>q)}' && [ $i -lt 120 ]; do sleep 15; i=$((i+1)); done
}

# run PHASE MODE TOOL REP cmd... : one timed alignment, appended to runs.tsv. Aborts on failure.
run() {
  local phase=$1 mode=$2 tool=$3 rep=$4; shift 4
  local of="$OUT/$phase.$mode.$tool.$rep"
  wait_quiet; local l0; l0=$(load1)
  ( while true; do load1; sleep 30; done ) >"$of.load" & local lp=$!
  /usr/bin/time -l bash -c "set -o pipefail; $* 2>'$of.err' | grep -v '^@' | tee >(wc -l | tr -d ' ' >'$of.n') | md5 >'$of.md5'" 2>"$of.time"
  local rc=$?
  kill $lp 2>/dev/null; wait $lp 2>/dev/null; sleep 1
  if [ $rc -ne 0 ]; then say "ABORT: $phase $mode $tool rep$rep failed (rc=$rc), see $of.err"; tail -3 "$of.err" | tee -a "$LOG"; exit 1; fi
  local row
  row=$(awk -v p="$phase" -v m="$mode" -v t="$tool" -v r="$rep" -v n="$(cat "$of.n")" -v h="$(cat "$of.md5")" \
            -v l="$(sort -n "$of.load" | tail -1)" -v l0="$l0" '
    / real /{w=$1; u=$3; s=$5} /maximum resident/{rss=$1/1e9}
    END{printf "%s\t%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%s\t%s\t%s\t%s\n", p,m,t,r,w,u,s,rss,n,h,l,l0}' "$of.time")
  echo "$row" >>"$TSV"
  say "  $(echo "$row" | awk -F'\t' '{printf "%-6s %s %s rep%s  wall=%ss user=%ss rss=%sGB records=%s md5=%s load_start=%s", $1,$2,$3,$4,$5,$6,$8,$9,$10,$12}')"
}

mem() { # mem TOOL MODE READS... -> command line
  local tool=$1 mode=$2; shift 2
  local bin=$M4 idx=$OUT/idx_m4; [ "$tool" = m2 ] && { bin=$M2; idx=$OUT/idx_m2; }
  echo "$bin mem -t$T -K $K $idx $*"
}

########## 0. provenance ##########
say "# bwa-mem4 vs bwa-mem2 publication benchmark, GIAB HG002"
say "date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "host:        $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null)P+$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null)E cores, $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB RAM"
say "os:          $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null) ($(uname -r))"
say "bwa-mem4:    $(git rev-parse HEAD)$( [ -n "$(git status --porcelain --untracked-files=no)" ] && echo ' (DIRTY TREE)'), $(rustc -V)"
say "             build: cargo build --release (lto=fat, codegen-units=1, panic=abort, .cargo/config: target-cpu=native)"
say "             sha256 $(shasum -a 256 "$M4" | cut -d' ' -f1)"
say "bwa-mem2:    $("$M2" version 2>/dev/null) ($(readlink -f "$M2"))"
say "             sha256 $(shasum -a 256 "$(readlink -f "$M2")" | cut -d' ' -f1)"
say "threads:     -t$T   batch: -K $K"
say "reference:   $FASTA  md5 $(md5 -q "$FASTA")  $(head -1 "$FASTA" | cut -c1-70)"
say "subset:      $SUB1 + _2: $(( $(wc -l <"$SUB1") / 4 )) pairs, md5 $(md5 -q "$SUB1") / $(md5 -q "$SUB2")"
if [ "$SKIP_FULL" != 1 ]; then
  say "full sample: $FULL1 + R2, md5 $(md5 -q "$FULL1") / $(md5 -q "$FULL2")"
  [ -f "$DATA/giab30x/r1_urls.txt" ] && cp "$DATA/giab30x/r1_urls.txt" "$OUT/" && say "             source FASTQ URLs: r1_urls.txt ($(wc -l <"$DATA/giab30x/r1_urls.txt" | tr -d ' ') R1 files)"
fi

########## 1. index ##########
say ""; say "== 1. index (each tool, same FASTA) =="
for tool in m4 m2; do
  bin=$M4; [ $tool = m2 ] && bin=$M2
  wait_quiet; l0=$(load1)
  /usr/bin/time -l "$bin" index -p "$OUT/idx_$tool" "$FASTA" >"$OUT/index.$tool.err" 2>&1 || { say "ABORT: $tool index failed"; exit 1; }
  awk -v t=$tool -v l0="$l0" '/ real /{w=$1;u=$3;s=$5} /maximum resident/{r=$1/1e9}
    END{printf "index\tbuild\t%s\t1\t%.2f\t%.2f\t%.2f\t%.2f\t-\t-\t-\t%s\n",t,w,u,s,r,l0}' "$OUT/index.$tool.err" >>"$TSV"
  say "  $tool  $(tail -1 "$TSV" | awk -F'\t' '{printf "wall=%ss user=%ss rss=%sGB", $5,$6,$8}')"
done
IDX_OK=1
for ext in pac ann amb bwt.2bit.64 0123; do
  if cmp -s "$OUT/idx_m2.$ext" "$OUT/idx_m4.$ext"; then say "  .$ext identical"; else say "  .$ext DIFFERS"; IDX_OK=0; fi
done

########## 2. warm-up ##########
say ""; say "== 2. warm-up (discarded) =="
run warmup se m4 0 "$(mem m4 se "$SUB1")"
run warmup se m2 0 "$(mem m2 se "$SUB1")"

########## 3. subset, interleaved ABBA ##########
say ""; say "== 3. subset x$SUB_REPS, interleaved =="
for i in $(seq 1 "$SUB_REPS"); do
  order="m2 m4"; [ $((i % 2)) = 0 ] && order="m4 m2"
  for mode in se pe; do
    for tool in $order; do
      if [ $mode = se ]; then run subset se $tool $i "$(mem $tool se "$SUB1")"
      else run subset pe $tool $i "$(mem $tool pe "$SUB1" "$SUB2")"; fi
    done
  done
done

########## 4. full sample ##########
if [ "$SKIP_FULL" != 1 ]; then
  say ""; say "== 4. full sample x$FULL_REPS =="
  for i in $(seq 1 "$FULL_REPS"); do
    for mode in se pe; do
      for tool in m4 m2; do
        if [ $mode = se ]; then run full se $tool $i "$(mem $tool se "$FULL1")"
        else run full pe $tool $i "$(mem $tool pe "$FULL1" "$FULL2")"; fi
      done
    done
  done
fi

########## summary ##########
say ""; say "== summary (median over reps; speedup = bwa-mem2 / bwa-mem4) =="
say "$(printf "%-14s %5s %12s %12s %8s %10s %10s  %s" "phase/mode" "reps" "mem2 wall" "mem4 wall" "speedup" "mem2 RSS" "mem4 RSS" "identity")"
awk -F'\t' -v Q="$QUIET" -v idx="$IDX_OK" '
  NR==1 || $1=="warmup" {next}
  { k=$1"\t"$2; n[k,$3]++; w[k,$3,n[k,$3]]=$5; r[k,$3,n[k,$3]]=$8; keys[k]=1
    if ($10!="-") { if (!(k in h)) h[k]=$10; else if (h[k]!=$10) bad[k]=1 }
    if ($12+0 > Q) busy[k]=1 }   # the run itself drives the load past T, so only the start counts
  function med(a, k, t, c,   i, j, x, v) { for (i=1;i<=c;i++) v[i]=a[k,t,i]
    for (i=2;i<=c;i++) { x=v[i]; for (j=i-1;j>0 && v[j]>x;j--) v[j+1]=v[j]; v[j+1]=x }
    return c%2 ? v[(c+1)/2] : (v[c/2]+v[c/2+1])/2 }
  END {
    for (k in keys) {
      c2=n[k,"m2"]; c4=n[k,"m4"]; a=med(w,k,"m2",c2); b=med(w,k,"m4",c4)
      id = bad[k] ? "RECORDS DIFFER" : "records identical"
      split(k, p, "\t"); if (p[1]=="index") id = idx ? "files identical" : "FILES DIFFER"
      printf "%-14s %5d %11.1fs %11.1fs %7.2fx %9.1fG %9.1fG  %s%s\n", p[1]"/"p[2], c4, a, b, a/b, med(r,k,"m2",c2), med(r,k,"m4",c4), id, busy[k] ? "  (a run started on a busy host)" : "" }
  }' "$TSV" | sort | tee -a "$LOG"
say ""; say "PUBLICATION_BENCH_DONE  $OUT"
