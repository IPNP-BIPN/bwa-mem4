#!/usr/bin/env bash
# Differential gate for the long-read path: `bwa-mem4 mem -x <long-read preset>` must produce the
# same SAM records as the rammap binary it delegates to.
#
# This is the only path in this repository whose oracle is NOT bwa-mem2. `-x pacbio|pbref|ont2d`
# leaves bwa's algorithm entirely (see `crates/bwa-cli/src/cmd_longread.rs` for why), so the claim
# under test is "we produce rammap's output", and the only thing that can prove it is rammap.
#
# THE READ-NAME CAVEAT, which is why the fixture is built rather than reused. Our FASTQ reader
# strips a trailing `/1` or `/2` from read names, because that is what bwa does and every other path
# in this binary depends on it. rammap keeps the suffix AND uses the read name in its tie-breaking,
# so feeding both the same file with Illumina-style names yields a handful of equal-scoring
# secondary hits chosen differently (8 records in 1015, measured). That is a naming convention
# difference, not a mapping difference: with the suffix removed from the input, the two outputs are
# byte-identical.
#
# Real long-read data has no such suffix (`/1` and `/2` are a paired-end Illumina convention), so
# this fixture uses realistic long-read naming and the comparison is exact. If you point this script
# at reads whose names end in `/1`, expect a small, explainable diff.
#
# Usage: scripts/longread_parity.sh [path-to-bwa-mem4]
# Env:   RAMMAP  path to the rammap binary (default: `rammap` on PATH)
#        IDX     reference FASTA (default: work/region.fa)
#        READS   long reads to map (default: a subset of work/r1_50k.fq, names de-suffixed)
set -euo pipefail
cd "$(dirname "$0")/.."

M4="${1:-./target/release/bwa-mem4}"
RAMMAP="${RAMMAP:-rammap}"
IDX="${IDX:-work/region.fa}"

command -v "$RAMMAP" >/dev/null 2>&1 || {
  echo "SKIP: no rammap binary on PATH (set RAMMAP=/path/to/rammap)."
  echo "      Install the version this build links: cargo install rammap --version 1.1.2"
  exit 0
}
[ -f "$IDX" ] || { echo "SKIP: no reference at $IDX"; exit 0; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Fixture: real long-read names carry no mate suffix, so strip it. Built here rather than committed
# so it cannot drift from whatever reads the working tree has.
SRC="${READS:-work/r1_50k.fq}"
[ -f "$SRC" ] || SRC=work/r1_500k.fq
[ -f "$SRC" ] || { echo "SKIP: no reads at $SRC"; exit 0; }
# `head` FIRST: with awk upstream it keeps writing after head exits, and the resulting SIGPIPE is
# fatal under `pipefail`. 20000 lines is 5000 reads, enough for the repeats that make the two
# implementations disagree if they are going to.
head -20000 "$SRC" | awk 'NR%4==1{sub(/\/[12]$/,"",$1); print $0; next} {print}' > "$TMP/reads.fq"

fail=0
# `-x pacbio` and `-x pbref` are the same preset in bwa (one shared branch), so both must land on
# rammap's map-pb; running both is what would catch a typo in the mapping table.
for pair in "ont2d:map-ont" "pacbio:map-pb" "pbref:map-pb"; do
  bwa_x="${pair%%:*}"; ram_x="${pair##*:}"
  "$RAMMAP" -a -x "$ram_x" -t 4 "$IDX" "$TMP/reads.fq" 2>/dev/null | grep -v '^@' > "$TMP/ref.sam"
  "$M4" mem -t 4 -x "$bwa_x" "$IDX" "$TMP/reads.fq" 2>/dev/null | grep -v '^@' > "$TMP/got.sam"
  if cmp -s "$TMP/ref.sam" "$TMP/got.sam"; then
    printf '  -x %-9s -> %-8s [PASS] %s records\n' "$bwa_x" "$ram_x" "$(wc -l < "$TMP/got.sam" | tr -d ' ')"
  else
    printf '  -x %-9s -> %-8s [FAIL]\n' "$bwa_x" "$ram_x"
    diff "$TMP/ref.sam" "$TMP/got.sam" | head -4
    fail=1
  fi
done

# Thread count is a speed knob and must not reach the records. This is the same invariant the
# short-read path holds at a fixed -K, and it is worth pinning separately here because the long-read
# path batches by BASES and chunks by worker count, so a bug would show up as a thread-dependent
# ordering rather than as wrong alignments.
"$M4" mem -t 1 -x ont2d "$IDX" "$TMP/reads.fq" 2>/dev/null | grep -v '^@' > "$TMP/t1.sam"
"$M4" mem -t 8 -x ont2d "$IDX" "$TMP/reads.fq" 2>/dev/null | grep -v '^@' > "$TMP/t8.sam"
if cmp -s "$TMP/t1.sam" "$TMP/t8.sam"; then
  echo "  -t1 == -t8                [PASS]"
else
  echo "  -t1 == -t8                [FAIL]"
  fail=1
fi

# `-x intractg` is NOT routed: it is contig alignment, not long reads, and it must still come from
# bwa's own code path. The check is that its @PG names bwa-mem4, i.e. that the routing table did not
# quietly swallow it.
# Written to a file first, not piped into `grep -q`: grep exits at the first match, the aligner
# takes SIGPIPE, and `pipefail` turns that into a failed check. Same trap as the `head` above.
"$M4" mem -x intractg "$IDX" "$TMP/reads.fq" 2>/dev/null > "$TMP/intractg.sam"
if grep -q '^@PG.*ID:bwa-mem4' "$TMP/intractg.sam"; then
  echo "  -x intractg stays on bwa  [PASS]"
else
  echo "  -x intractg stays on bwa  [FAIL]"
  fail=1
fi

[ "$fail" -eq 0 ] && echo "RESULT: long-read path matches rammap" || echo "RESULT: FAILED"
exit "$fail"
