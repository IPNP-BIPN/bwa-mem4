#!/usr/bin/env bash
# Phase 11 gate: does byte-identity survive all the way to called variants?
#
# The project's acceptance criterion is that our SAM bytes equal bwa-mem2's. That is the strongest
# statement an aligner can make, and it is the wrong LANGUAGE for the person who has to sign off on
# a clinical pipeline: they ask about precision and recall against a truth set, not about md5sums.
# This script answers in their language. It aligns the same reads with both aligners, calls variants
# from both BAMs with the same caller, and scores both VCFs against the GIAB truth set.
#
# The expected result is not a discovery. If the BAMs are identical the VCFs are identical and the
# two rows of the table are the same row. That IS the deliverable: a reproducible artefact that says
# so, with the numbers, rather than an argument that it must be so.
#
# Two failure modes it can actually catch, which is why it is worth running rather than reasoning
# about:
#   1. Byte-identity holding on the records while something outside them (a header field, the sort
#      order a downstream tool depends on) differs enough to move a caller.
#   2. A future change that keeps `oracle_diff.sh` green on a small fixture while diverging on real
#      data at real depth, where the truth set has coverage and the fixture does not.
#
# Usage:
#   REF=hg38.fa R1=hg002_1.fq.gz R2=hg002_2.fq.gz \
#   TRUTH_VCF=HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
#   TRUTH_BED=HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
#   scripts/giab_happy.sh
#
# Optional:
#   REGION=chr21     restrict calling and scoring to one contig (the CI-sized mode)
#   THREADS=8        default: all cores
#   OUT=work/happy   output directory
#   MEM4=...         path to the bwa-mem4 binary (default target/release/bwa-mem4)
#   ORACLE=bwa-mem2  the oracle to compare against, on PATH or as a path
#
# Needs on PATH: samtools, bcftools, and one of `rtg` (rtg-tools) or `hap.py`. The oracle
# (`bwa-mem2`) is needed for the comparison arm; without it the script still scores our own calls
# and says the oracle arm was skipped.
#
#   micromamba install -c conda-forge -c bioconda samtools bcftools rtg-tools bwa-mem2
set -euo pipefail
cd "$(dirname "$0")/.."

: "${REF:?set REF to the reference FASTA (the same prefix the index was built from)}"
: "${R1:?set R1 to the first read file}"
: "${R2:?set R2 to the second read file}"
: "${TRUTH_VCF:?set TRUTH_VCF to the GIAB benchmark VCF}"
: "${TRUTH_BED:?set TRUTH_BED to the GIAB confident-regions BED}"
THREADS="${THREADS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu) )}"
OUT="${OUT:-work/happy}"
MEM4="${MEM4:-target/release/bwa-mem4}"
ORACLE="${ORACLE:-bwa-mem2}"
REGION="${REGION:-}"

mkdir -p "$OUT"
echo "== configuration =="
echo "ref=$REF reads=$R1,$R2 threads=$THREADS region=${REGION:-whole genome} out=$OUT"

# Align, sort, index. One function, called once per aligner, so the two arms cannot drift apart in
# anything but the aligner itself: same threads, same sort, same everything downstream.
#
# `-M` is deliberately NOT passed. bwa's supplementary-alignment flagging changes which records a
# caller sees, and the point here is to compare two aligners under the SAME options, whatever they
# are, not to reproduce a particular pipeline's preferences.
align() {
  local label="$1" tool="$2"; shift 2
  local bam="$OUT/$label.bam"
  if [ -s "$bam" ]; then
    echo "== $label: reusing $bam =="
    return
  fi
  echo "== $label: aligning =="
  # Piped straight into `samtools sort`: writing the SAM out and reading it back would double the
  # I/O for no benefit, and on a 30x genome that is hundreds of gigabytes.
  "$tool" mem -t "$THREADS" "$REF" "$R1" "$R2" 2> "$OUT/$label.aln.log" \
    | samtools sort -@ "$THREADS" -o "$bam" -
  samtools index -@ "$THREADS" "$bam"
}

# Call variants. bcftools rather than GATK or DeepVariant, for one reason: this gate is about the
# DIFFERENCE between two BAMs, and the cheapest caller that is sensitive to that difference is the
# right one. A heavier caller would add hours and its own nondeterminism without making the
# comparison sharper. The absolute precision/recall figures below are therefore bcftools' figures,
# not the best a modern pipeline can do, and the table says so.
call() {
  local label="$1"
  local bam="$OUT/$label.bam" vcf="$OUT/$label.vcf.gz"
  if [ -s "$vcf" ]; then
    echo "== $label: reusing $vcf =="
    return
  fi
  echo "== $label: calling =="
  # `-a AD` because the scorers want allele depths; `-Ou` between the two so the intermediate BCF is
  # never serialised to text.
  # shellcheck disable=SC2086
  bcftools mpileup -Ou -f "$REF" -a AD ${REGION:+-r "$REGION"} --threads "$THREADS" "$bam" \
    | bcftools call -mv -Oz --threads "$THREADS" -o "$vcf" -
  bcftools index -t "$vcf"
}

# Score one VCF against the truth set, inside the confident regions.
#
# `rtg vcfeval` is preferred over `hap.py`: it is a conda package rather than a container, and its
# variant matching is the same haplotype-aware comparison hap.py performs through its engine. When
# only `hap.py` is present it is used instead, and the header of the printed table says which
# produced the numbers, because the two do not always agree to the last decimal on complex indels.
score() {
  local label="$1"
  local vcf="$OUT/$label.vcf.gz"
  echo "== $label: scoring =="
  if command -v rtg > /dev/null; then
    # The SDF is the reference in rtg's own format, built once and reused by both arms.
    [ -d "$OUT/sdf" ] || rtg format -o "$OUT/sdf" "$REF" > /dev/null
    rm -rf "$OUT/eval-$label"
    rtg vcfeval -b "$TRUTH_VCF" -c "$vcf" -e "$TRUTH_BED" -t "$OUT/sdf" \
      ${REGION:+--region "$REGION"} -o "$OUT/eval-$label" > /dev/null
    echo "-- $label (rtg vcfeval) --"
    cat "$OUT/eval-$label/summary.txt"
  elif command -v hap.py > /dev/null; then
    rm -rf "$OUT/eval-$label"; mkdir -p "$OUT/eval-$label"
    hap.py "$TRUTH_VCF" "$vcf" -f "$TRUTH_BED" -r "$REF" \
      ${REGION:+-l "$REGION"} -o "$OUT/eval-$label/happy" > /dev/null
    echo "-- $label (hap.py) --"
    # The summary CSV's first columns are type, filter, then the counts; the whole file is small
    # enough to print rather than parse.
    cat "$OUT/eval-$label/happy.summary.csv"
  else
    echo "neither rtg nor hap.py on PATH: cannot score $label" >&2
    return 1
  fi
}

align mem4 "$MEM4"
if command -v "$ORACLE" > /dev/null || [ -x "$ORACLE" ]; then
  align oracle "$ORACLE"
  HAVE_ORACLE=1
else
  echo "== oracle ($ORACLE) not found: scoring our arm only =="
  HAVE_ORACLE=0
fi

# The identity check, before any scoring: if this passes, the rest of the script is a formality and
# the table below is guaranteed to have two identical rows. If it FAILS, the table is the
# interesting part, so the script continues rather than exiting.
if [ "$HAVE_ORACLE" = 1 ]; then
  echo "== record-level identity of the two BAMs =="
  a=$(samtools view "$OUT/mem4.bam" | md5sum | cut -d' ' -f1)
  b=$(samtools view "$OUT/oracle.bam" | md5sum | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then
    echo "BAM records identical ($a)"
  else
    echo "BAM records DIFFER: mem4=$a oracle=$b"
    echo "(the table below is then a real comparison, not a formality)"
  fi
fi

call mem4
[ "$HAVE_ORACLE" = 1 ] && call oracle

if [ "$HAVE_ORACLE" = 1 ]; then
  echo "== VCF identity =="
  # Compares the CALLS, not the file: a gzip member's bytes depend on the compressor's state and
  # every VCF carries the command line that produced it in its header.
  va=$(bcftools view -H "$OUT/mem4.vcf.gz" | md5sum | cut -d' ' -f1)
  vb=$(bcftools view -H "$OUT/oracle.vcf.gz" | md5sum | cut -d' ' -f1)
  [ "$va" = "$vb" ] && echo "VCF records identical ($va)" || echo "VCF records DIFFER: mem4=$va oracle=$vb"
fi

score mem4
[ "$HAVE_ORACLE" = 1 ] && score oracle

echo
echo "== what this table is =="
echo "Precision and recall are bcftools' numbers on this data, not a statement about the best"
echo "achievable pipeline. The load-bearing claim is that the two arms agree, which is what makes"
echo "byte-identity mean something to a variant-calling user."
