#!/usr/bin/env bash
# ALT-contig parity against bwa-mem2, on the real GRCh38 analysis set and the real bwakit `.alt`.
#
# Why this cannot live in CI, and why it exists anyway:
#
# ALT contigs change the output in ways NO other test in this repo reaches. They change the header
# (`AH:*`), they add a tag (`pa:f`), they reorder every read's regions (`alnreg_hlt2`), they change
# MAPQ (the primary-assembly-only second marking round), and they add a supplementary record per
# end in the paired branch. Until 2026-07-20 none of that was implemented and no test would have
# noticed, because every fixture in the repo is a reference with no ALT contigs.
#
# The fixture is 3.2 GB and the index build peaks around 80 GB of RAM, so this is a manual gate in
# the same spirit as giab30x_pe.sh: run it before tagging a release, not on every push.
#
# Usage:
#   scripts/alt_parity.sh [path-to-bwa-mem4]
#
# Inputs, all overridable:
#   REF=work/hs38a.fa       GRCh38 analysis set INCLUDING the _alt contigs
#   ALT=work/hs38DH.fa.alt  the bwakit .alt naming them
#   N=200000                read pairs to simulate
#
# Fetching the inputs (once):
#   curl -fL -o work/hs38a.fna.gz \
#     https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_full_analysis_set.fna.gz
#   gzip -dc work/hs38a.fna.gz > work/hs38a.fa
#   # the .alt ships only in the bwa.kit binary release, not in the bwa git tree:
#   curl -fL -o /tmp/bwakit.tar.bz2 \
#     https://downloads.sourceforge.net/project/bio-bwa/bwakit/bwakit-0.7.15_x64-linux.tar.bz2
#   tar xjf /tmp/bwakit.tar.bz2 --strip-components=2 -C work bwa.kit/resource-GRCh38/hs38DH.fa.alt
set -uo pipefail
cd "$(dirname "$0")/.."

M3="${1:-./target/release/bwa-mem4}"
M2="${M2:-bwa-mem2}"
REF="${REF:-work/hs38a.fa}"
ALT="${ALT:-work/hs38DH.fa.alt}"
N="${N:-200000}"
OUT="${OUT:-work/altparity}"
mkdir -p "$OUT"

for f in "$REF" "$ALT"; do
  [ -f "$f" ] || { echo "missing input: $f (see the header of this script)"; exit 1; }
done

echo "== inputs =="
echo "  reference : $REF ($(grep -c '^>' "$REF") contigs, $(grep -c '^>.*_alt' "$REF") of them _alt)"
echo "  alt file  : $ALT ($(grep -vc '^@' "$ALT") non-header lines)"

# ONE index, read by BOTH aligners. That is the project's standing rule: it isolates the aligner
# from the indexer, so a difference here can only come from alignment. The `.alt` is a sibling of
# the index prefix and is read at load time by each aligner independently.
if [ ! -f "$REF.bwt.2bit.64" ]; then
  echo "== building the index (peaks near 80 GB RSS; several minutes) =="
  "$M2" index "$REF" || exit 1
fi
[ -f "$REF.alt" ] || cp "$ALT" "$REF.alt"

# Reads simulated FROM the reference, so a good fraction genuinely originate on ALT contigs and
# exercise the primary-vs-ALT ranking rather than merely passing through it.
if [ ! -f "$OUT/r1.fq" ]; then
  echo "== simulating $N read pairs =="
  python3 scripts/make_test_reads.py "$REF" "$OUT/r" --n "$N" || exit 1
  mv "$OUT/r_1.fq" "$OUT/r1.fq"; mv "$OUT/r_2.fq" "$OUT/r2.fq"
fi

run() { # $1 = label, rest = extra options
  local label="$1"; shift
  echo "== $label =="
  "$M2" mem -t8 -K 10000000 "$@" "$REF" "$OUT/r1.fq" "$OUT/r2.fq" 2>/dev/null | grep -v '^@PG' > "$OUT/m2.sam"
  "$M3" mem -t8 -K 10000000 "$@" "$REF" "$OUT/r1.fq" "$OUT/r2.fq" 2>/dev/null | grep -v '^@PG' > "$OUT/m3.sam"
  if cmp -s "$OUT/m2.sam" "$OUT/m3.sam"; then
    echo "  BYTE-IDENTICAL ($(grep -vc '^@' "$OUT/m2.sam") records)"
    return 0
  fi
  echo "  DIFFER"
  echo "    records: mem2=$(grep -vc '^@' "$OUT/m2.sam") mem3=$(grep -vc '^@' "$OUT/m3.sam")"
  echo "    differing lines: $(paste -d'\001' "$OUT/m2.sam" "$OUT/m3.sam" | awk -F'\001' '$1!=$2' | wc -l)"
  # Which FIELD differs is the first thing worth knowing, since the ALT work touches exactly four:
  # the AH:* header suffix, the pa:f tag, MAPQ, and the supplementary ALT record.
  echo "    header lines differing: $(diff <(grep '^@' "$OUT/m2.sam") <(grep '^@' "$OUT/m3.sam") | grep -c '^[<>]')"
  echo "    pa:f present: mem2=$(grep -c 'pa:f:' "$OUT/m2.sam") mem3=$(grep -c 'pa:f:' "$OUT/m3.sam")"
  echo "    first 4 differing records:"
  diff "$OUT/m2.sam" "$OUT/m3.sam" | head -8 | sed 's/^/      /'
  return 1
}

fail=0
# Defaults: the case every user hits.
run "PE, defaults (ALT active)" || fail=1
# -j must make the whole thing behave exactly as an index with no .alt would.
run "PE, -j (ALT flags cleared)" -j || fail=1
# -a changes which shadowed regions are emitted, and the ALT branch rewrites `secondary_all`, which
# is precisely what -a reads. The two interact; neither alone would catch a mistake in the pairing.
run "PE, -a (all alignments)" -a || fail=1

echo
if [ $fail = 0 ]; then echo "RESULT: all ALT cases byte-identical"; else echo "RESULT: FAILURES above"; fi
exit $fail
