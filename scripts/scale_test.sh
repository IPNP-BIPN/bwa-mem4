#!/usr/bin/env bash
# Scaling gate: build our index and bwa-mem2's for a whole chromosome (or any region), assert the
# 5 index files are byte-identical, then run PE alignment and report concordance vs the oracle.
#
# Env overrides: REF (bgzipped FASTA), REGION (samtools region, e.g. "20" or "20:1-1000000"),
# NREADS, THREADS.
set -euo pipefail
cd "$(dirname "$0")/.."

REF="${REF:-/Users/benjamin/Database/VEP_database/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}"
REGION="${REGION:-20}"
NREADS="${NREADS:-5000}"
THREADS="${THREADS:-8}"
W=work8
mkdir -p "$W"

cargo build --release --quiet
OURS=target/release/bwa-mem3

echo "[extract] $REGION"
samtools faidx "$REF" "$REGION" > "$W/ref.fa"
cp "$W/ref.fa" "$W/ref_oracle.fa"

echo "[index] ours + oracle"
"$OURS" index "$W/ref.fa" 2>/dev/null
bwa-mem2 index "$W/ref_oracle.fa" 2>/dev/null

echo "[gate] index byte-identity"
IDX_OK=1
for ext in pac ann amb bwt.2bit.64 0123; do
  if cmp -s "$W/ref.fa.$ext" "$W/ref_oracle.fa.$ext"; then
    echo "  .$ext: IDENTICAL"
  else
    echo "  .$ext: DIFFER"; IDX_OK=0
  fi
done
[ "$IDX_OK" = 1 ] || { echo "INDEX GATE: FAIL" >&2; exit 1; }

echo "[wgsim] $NREADS pairs (seed 11)"
wgsim -S 11 -N "$NREADS" -1 150 -2 150 -e 0.005 -r 0.001 \
  "$W/ref.fa" "$W/r1.fq" "$W/r2.fq" >/dev/null 2>&1

echo "[align] PE oracle vs ours (-t$THREADS -K 10000000)"
bwa-mem2 mem -t"$THREADS" -K 10000000 "$W/ref.fa" "$W/r1.fq" "$W/r2.fq" 2>/dev/null > "$W/oracle_pe.sam"
"$OURS" mem -t"$THREADS" -K 10000000 "$W/ref.fa" "$W/r1.fq" "$W/r2.fq" 2>/dev/null > "$W/ours_pe.sam"

python3 - "$W/oracle_pe.sam" "$W/ours_pe.sam" <<'PY'
import sys
def load(p):
    r={}
    for l in open(p):
        if l.startswith('@'): continue
        f=l.rstrip().split('\t'); fl=int(f[1])
        r.setdefault((f[0], fl&0x1c0), []).append(l.rstrip())
    return r
o=load(sys.argv[1]); u=load(sys.argv[2])
t=s=0
for k in o:
    for a,b in zip(o[k], u.get(k,[])): t+=1; s+= (a==b)
print(f"PE concordance: {s}/{t} ({100*s/t:.2f}%) byte-identical records")
PY
echo "SCALE GATE: index PASS"
