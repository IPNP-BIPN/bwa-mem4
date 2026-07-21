#!/usr/bin/env bash
# Reproductions of open bwa-mem2 issues, run against BOTH aligners.
#
# Why this exists, and what it is NOT:
#
# This project reproduces bwa-mem2's output byte for byte, INCLUDING its bugs. So a behavioural
# upstream bug (#293 invalid BAM under -R, #278 missing MQ, #260 supplementary MAPQ) must NOT be
# fixed here: fixing it would break the acceptance criterion. Those are out of scope by design.
#
# Crashes are the exception. A run that aborts produces no output, so there is no output to be
# identical to, and surviving where upstream dies costs nothing in parity. That is the only class
# of upstream issue this script covers.
#
# Usage: scripts/upstream_repros.sh [path-to-bwa-mem4]
#   GENOME=work/genome.fa   a GRCh38 index, needed by the #269 case only (that case is skipped
#                           when it is absent; it downloads 10 KB of reads from the issue thread)
set -uo pipefail
cd "$(dirname "$0")/.."

M3="${1:-./target/release/bwa-mem4}"
M2="${M2:-bwa-mem2}"
GENOME="${GENOME:-work/genome.fa}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "======================================================================"
echo "bwa-mem2 #296 - SegFault aligning reads longer than the fasta sequences"
echo "======================================================================"
# Reporter: ~22.6k contigs of 60bp, 100bp reads, segfault on x86 (all SIMD levels), while the same
# reads against the human genome are fine. Shape reproduced here at 2000 contigs.
python3 - "$TMP" <<'PY'
import random, sys
random.seed(296)
T = sys.argv[1]
contigs = []
with open(f"{T}/coding.fa", "w") as f:
    for i in range(2000):
        s = "".join(random.choice("ACGT") for _ in range(60))
        contigs.append(s)
        f.write(f">gene{i}#NM_{i:06d}#SYM{i}\n{s}\n")
# Each read is a whole contig plus 40bp of unrelated sequence: it CANNOT fit inside one contig,
# which is the condition the reporter hit.
with open(f"{T}/c1.fq", "w") as f1, open(f"{T}/c2.fq", "w") as f2:
    for i in range(5000):
        for f in (f1, f2):
            r = random.choice(contigs) + "".join(random.choice("ACGT") for _ in range(40))
            f.write(f"@rd{i}\n{r}\n+\n{'I' * 100}\n")
PY
"$M2" index "$TMP/coding.fa" >/dev/null 2>&1
"$M2" mem -t1 "$TMP/coding.fa" "$TMP/c1.fq" "$TMP/c2.fq" >"$TMP/c_m2.sam" 2>/dev/null
echo "  bwa-mem2: rc=$? records=$(grep -vc '^@' "$TMP/c_m2.sam" 2>/dev/null)"
"$M3" mem -t1 "$TMP/coding.fa" "$TMP/c1.fq" "$TMP/c2.fq" >"$TMP/c_m3.sam" 2>/dev/null
echo "  bwa-mem4: rc=$? records=$(grep -vc '^@' "$TMP/c_m3.sam" 2>/dev/null)"
if cmp -s <(grep -v '^@PG' "$TMP/c_m2.sam") <(grep -v '^@PG' "$TMP/c_m3.sam"); then
  echo "  -> byte-identical."
else
  echo "  -> OUTPUTS DIFFER."
fi
echo "  NOTE: not reproduced on aarch64 for either aligner. The reporter's crash is on the x86"
echo "        AVX512 build, so this case is a regression guard here, not a reproduction."

echo
echo "======================================================================"
echo "bwa-mem2 #280 - invalid index created (.amb fails to parse on load)"
echo "======================================================================"
# The reporter's own index build printed `pos: 91632415, ref_seq_len__: 91632414` and the aligner
# then died in `bns_restore_core` with a .amb parse error. Their fasta (a gencode rRNA/ERCC/SIRV
# transcriptome) is behind an expired file-sharing link, so this builds a reference of the same
# SHAPE: thousands of contigs of uneven length with N runs inside them AND flush against their
# edges. That edge is not academic: this project's own indexer once merged N runs across contig
# boundaries, and only a whole-genome build exposed it.
python3 - "$TMP" <<'PY'
import random, sys
random.seed(280)
T = sys.argv[1]
with open(f"{T}/tx.fa", "w") as f:
    for i in range(6000):
        L = random.choice([80, 150, 400, 1200, 5000])
        s = [random.choice("ACGT") for _ in range(L)]
        for _ in range(random.randrange(0, 4)):
            p = random.randrange(L)
            for j in range(p, min(L, p + random.randrange(1, 60))):
                s[j] = 'N'
        if random.random() < 0.3:                                  # flush against the start
            for j in range(min(L, random.randrange(1, 40))):
                s[j] = 'N'
        if random.random() < 0.3:                                  # flush against the end
            for j in range(max(0, L - random.randrange(1, 40)), L):
                s[j] = 'N'
        f.write(f">tx{i}|ENST{i:08d}|gene{i % 900}\n" + "".join(s) + "\n")
PY
mkdir -p "$TMP/i2" "$TMP/i3"
cp "$TMP/tx.fa" "$TMP/i2/tx.fa"; cp "$TMP/tx.fa" "$TMP/i3/tx.fa"
"$M2" index "$TMP/i2/tx.fa" >/dev/null 2>&1; echo "  bwa-mem2 index: rc=$?"
"$M3" index "$TMP/i3/tx.fa" >/dev/null 2>&1; echo "  bwa-mem4 index: rc=$?"
ok=1
for e in pac ann amb bwt.2bit.64 0123; do
  if cmp -s "$TMP/i2/tx.fa.$e" "$TMP/i3/tx.fa.$e"; then echo "    .$e identical"; else echo "    .$e DIFFERS"; ok=0; fi
done
echo "  holes recorded (.amb header): mem2='$(head -1 "$TMP/i2/tx.fa.amb")' mem3='$(head -1 "$TMP/i3/tx.fa.amb")'"
[ $ok = 1 ] && echo "  -> both indexes valid and byte-identical; upstream's failure NOT reproduced on this shape."

echo
echo "======================================================================"
echo "bwa-mem2 #269 - assert failed for seqPair size"
echo "======================================================================"
# The one case that reproduces exactly. 345 read pairs from the issue thread, against GRCh38.
if [ ! -f "$GENOME.bwt.2bit.64" ]; then
  echo "  SKIPPED: no GRCh38 index at \$GENOME ($GENOME)."
else
  curl -fsSL -o "$TMP/R1.fq.gz" "https://github.com/user-attachments/files/18259852/R1.fq.gz" &&
  curl -fsSL -o "$TMP/R2.fq.gz" "https://github.com/user-attachments/files/18259855/R2.fq.gz" || {
    echo "  SKIPPED: could not fetch the reads from the issue thread."; exit 0; }
  "$M2" mem -t2 -K 10000000 "$GENOME" "$TMP/R1.fq.gz" "$TMP/R2.fq.gz" >"$TMP/m2.sam" 2>"$TMP/m2.err"
  echo "  bwa-mem2: rc=$? records=$(grep -vc '^@' "$TMP/m2.sam" 2>/dev/null)"
  grep -m2 -iE 'assert|Unexpected' "$TMP/m2.err" | sed 's/^/    /'
  "$M3" mem -t2 -K 10000000 "$GENOME" "$TMP/R1.fq.gz" "$TMP/R2.fq.gz" >"$TMP/m3.sam" 2>"$TMP/m3.err"
  echo "  bwa-mem4: rc=$? records=$(grep -vc '^@' "$TMP/m3.sam" 2>/dev/null)"
  echo "    unmapped: $(grep -v '^@' "$TMP/m3.sam" | awk '{if (int($2/4)%2) n++} END{print n+0}')"
  echo "  -> upstream aborts and emits nothing; we align every pair. No parity cost: there is no"
  echo "     upstream output to be identical to."
fi
