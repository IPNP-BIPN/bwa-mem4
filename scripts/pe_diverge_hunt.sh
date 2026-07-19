#!/usr/bin/env bash
# PE divergence evidence-gathering: re-run both tools PE at 30x, save core+tag fields (not SEQ/QUAL,
# which are the identical input reads), confirm the full-record md5 reproduces the bench, then diff
# exhaustively to locate and categorise the divergence (core vs tag vs reordering).
set -uo pipefail
cd "$(dirname "$0")/.."

M2=bwa-mem2
M3=./target/release/bwa-mem3
IDX=work/genome.fa
R1=work/giab30x/HG002_30x_R1.fastq.gz
R2=work/giab30x/HG002_30x_R2.fastq.gz
K=100000000
T=8
OUT=work/giab30x/diverge
mkdir -p "$OUT"
LOG="$OUT/hunt.log"
say() { echo "$@" | tee -a "$LOG"; }

# expected full-record md5 from the bench (reproducibility sanity)
EXP_M2=d818f019363f6879e52d5a9825f88fcf
EXP_M3=66d8cde1982e68f5c5f9d9da0a5a95fe

# $1=tool $2=out prefix ; saves cut-f1-9,12- to $2.core and full-record md5 to $2.fullmd5
capture() {
  local bin="$1" of="$2"
  say "[capture $(basename "$of")] $(date '+%H:%M:%S')"
  /usr/bin/time -p bash -c \
    "$bin mem -t$T -K $K $IDX $R1 $R2 2>'$of.err' | grep -v '^@' | tee >(md5 >'$of.fullmd5') | cut -f1-9,12- >'$of.core'" \
    2>"$of.time"
  say "  real=$(awk '/^real/{print $2}' "$of.time")s  fullmd5=$(cat "$of.fullmd5")  lines=$(wc -l <"$of.core" | tr -d ' ')"
}

say "############################################################"
say "# PE divergence hunt  $(date '+%Y-%m-%d %H:%M:%S')   (30x, t$T)"
say "############################################################"

capture "$M2" "$OUT/m2_pe"
[ "$(cat "$OUT/m2_pe.fullmd5")" = "$EXP_M2" ] && say "  [ok] mem2 md5 reproduces bench" || say "  [WARN] mem2 md5 != bench ($EXP_M2)"

capture "$M3" "$OUT/m3_pe"
[ "$(cat "$OUT/m3_pe.fullmd5")" = "$EXP_M3" ] && say "  [ok] mem3 md5 reproduces bench" || say "  [WARN] mem3 md5 != bench ($EXP_M3)"

say ""
say "==== DIFF ANALYSIS ===="
say "[cmp] first raw byte difference:"
cmp "$OUT/m2_pe.core" "$OUT/m3_pe.core" 2>&1 | tee -a "$LOG" || true

say "[stream-diff] locating & categorising (python)..."
python3 - "$OUT/m2_pe.core" "$OUT/m3_pe.core" 2>&1 <<'PY' | tee -a "$LOG"
import sys
p2, p3 = sys.argv[1], sys.argv[2]
CORE = 9  # fields 1..9 are core (QNAME FLAG RNAME POS MAPQ CIGAR RNEXT PNEXT TLEN); rest are tags
n=diff=qname_mismatch=core_diff=tag_only=0
examples=[]
with open(p2) as f2, open(p3) as f3:
    for a,b in zip(f2,f3):
        n+=1
        if a==b: continue
        diff+=1
        fa=a.rstrip('\n').split('\t'); fb=b.rstrip('\n').split('\t')
        if fa[0]!=fb[0]:
            qname_mismatch+=1
        core_a=fa[:CORE]; core_b=fb[:CORE]
        if core_a!=core_b:
            core_diff+=1
            if len(examples)<8:
                which=[i for i in range(min(len(core_a),len(core_b))) if core_a[i]!=core_b[i]]
                examples.append((n, "CORE fld"+str([i+1 for i in which]),
                                 "\t".join(core_a), "\t".join(core_b)))
        else:
            tag_only+=1
            if len(examples)<8:
                examples.append((n, "TAG-only", a.rstrip('\n')[:160], b.rstrip('\n')[:160]))
print(f"records compared : {n}")
print(f"differing        : {diff}  ({100*diff/n:.6f}%)")
print(f"  core-field diff : {core_diff}")
print(f"  tag-only diff   : {tag_only}")
print(f"  QNAME misaligned: {qname_mismatch}   (>0 => reordering, positional diff invalid)")
print("--- first divergences ---")
for ln,kind,a,b in examples:
    print(f"[rec {ln}] {kind}")
    print(f"  mem2: {a}")
    print(f"  mem3: {b}")
PY

say ""
say "PE_DIVERGE_HUNT_DONE"
