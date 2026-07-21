#!/usr/bin/env bash
# Fast PE divergence hunt on the FIRST N pairs (default 80M), both tools in PARALLEL (128GB RAM),
# same -K as the full run so batch boundaries coincide with the full run's first batches.
# If the divergence lives in one of those batches it reproduces identically; if not -> escalate.
set -uo pipefail
cd "$(dirname "$0")/.."

NPAIRS="${NPAIRS:-80000000}"
T="${T:-6}"           # two -t6 jobs share the 12 P-cores
K=100000000
M2=bwa-mem2
M3=./target/release/bwa-mem4
IDX=work/genome.fa
GZ1=work/giab30x/HG002_30x_R1.fastq.gz
GZ2=work/giab30x/HG002_30x_R2.fastq.gz
OUT=work/giab30x/diverge_sub
mkdir -p "$OUT"
LOG="$OUT/hunt.log"
say() { echo "$@" | tee -a "$LOG"; }
SUB1="$OUT/sub_R1.fq"; SUB2="$OUT/sub_R2.fq"
NL=$((NPAIRS*4))

say "############################################################"
say "# PE divergence hunt (subset $NPAIRS pairs)  $(date '+%Y-%m-%d %H:%M:%S')  t$T x2 parallel"
say "############################################################"

say "[extract] first $NPAIRS pairs -> plain fastq"
gunzip -c "$GZ1" 2>/dev/null | head -n "$NL" > "$SUB1" &
gunzip -c "$GZ2" 2>/dev/null | head -n "$NL" > "$SUB2" &
wait
say "  R1=$(($(wc -l <"$SUB1")/4)) reads  R2=$(($(wc -l <"$SUB2")/4)) reads"

# both tools in parallel; each: records -> fullmd5 + core(f1-9,12-)
cap() { local bin="$1" of="$2"
  /usr/bin/time -p bash -c "$bin mem -t$T -K $K $IDX $SUB1 $SUB2 2>'$of.err' | grep -v '^@' | tee >(md5 >'$of.fullmd5') | cut -f1-9,12- >'$of.core'" 2>"$of.time"
}
say "[align] mem2 || mem3  $(date '+%H:%M:%S')"
cap "$M2" "$OUT/m2" & P2=$!
cap "$M3" "$OUT/m3" & P3=$!
wait $P2 $P3
say "  mem2 real=$(awk '/^real/{print $2}' "$OUT/m2.time")s md5=$(cat "$OUT/m2.fullmd5") lines=$(wc -l <"$OUT/m2.core"|tr -d ' ')"
say "  mem3 real=$(awk '/^real/{print $2}' "$OUT/m3.time")s md5=$(cat "$OUT/m3.fullmd5") lines=$(wc -l <"$OUT/m3.core"|tr -d ' ')"

say ""
if [ "$(cat "$OUT/m2.fullmd5")" = "$(cat "$OUT/m3.fullmd5")" ]; then
  say "==== RESULT: IDENTICAL on $NPAIRS pairs -> divergence NOT in first $NPAIRS pairs. ESCALATE. ===="
  say "PE_SUBSET_DONE identical"
  exit 0
fi
say "==== RESULT: REPRODUCED. Divergence is in the first $NPAIRS pairs. Analysing... ===="
python3 - "$OUT/m2.core" "$OUT/m3.core" 2>&1 <<'PY' | tee -a "$LOG"
import sys
p2,p3=sys.argv[1],sys.argv[2]; CORE=9
n=diff=qm=cd=to=0; ex=[]
with open(p2) as f2, open(p3) as f3:
    for a,b in zip(f2,f3):
        n+=1
        if a==b: continue
        diff+=1
        fa=a.rstrip('\n').split('\t'); fb=b.rstrip('\n').split('\t')
        if fa[0]!=fb[0]: qm+=1
        if fa[:CORE]!=fb[:CORE]:
            cd+=1
            if len(ex)<10:
                w=[i+1 for i in range(min(len(fa),len(fb))) if fa[i]!=fb[i]]
                ex.append((n,f"CORE fld{w}","\t".join(fa[:CORE]),"\t".join(fb[:CORE])))
        else:
            to+=1
            if len(ex)<10: ex.append((n,"TAG-only",a.rstrip()[:150],b.rstrip()[:150]))
print(f"records compared : {n}")
print(f"differing        : {diff}  ({100*diff/n:.6f}%)")
print(f"  core-field diff : {cd}")
print(f"  tag-only diff   : {to}")
print(f"  QNAME misaligned: {qm}  (>0 => reordering)")
print("--- first divergences ---")
for ln,k,a,b in ex:
    print(f"[rec {ln}] {k}\n  mem2: {a}\n  mem3: {b}")
PY
say "PE_SUBSET_DONE reproduced"
