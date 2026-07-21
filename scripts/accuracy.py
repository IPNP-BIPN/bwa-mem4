#!/usr/bin/env python3
"""Measure alignment accuracy against known truth, for reads from `make_test_reads.py`.

WHY THIS EXISTS. Every gate in this project compares against bwa-mem2 with `cmp`: byte-identity is
the acceptance criterion, so "correct" has so far meant "identical to the oracle". The moment output
diverges on purpose, that stops being a test and becomes a diff to read. This measures something the
oracle cannot: whether a read is placed where it actually came from.

It answers a different question from `opt_parity.sh`, and neither replaces the other. Parity asks
"did anything change?". This asks "is it right?". A change can be identical and wrong, or different
and better, and only one of these two scripts can tell you which.

TRUTH SEMANTICS, from `make_test_reads.py`:
  - the reference is read as ONE concatenated sequence, so the fixture must be single-contig;
  - `sim{i}_{start}` carries `start`, a 0-BASED offset, so the expected SAM POS is `start + 1`;
  - mate 1 is the forward strand at `start`. Mate 2 is the reverse complement of the fragment's far
    end, and the fragment length is jittered and NOT recorded in the name, so mate 2 has no exact
    truth. Only mate 1 is scored. Half the reads is plenty and a wrong answer is worse than a
    smaller sample.

Usage:  scripts/accuracy.py <sam> [--tol 5]
        bwa-mem2 mem ... | scripts/accuracy.py /dev/stdin
"""

import argparse
import sys

# MAPQ buckets. The high end is what matters: a caller trusts MAPQ >= 30 and an aligner that is
# confidently wrong there does more damage than one that declines to answer.
BUCKETS = [(60, 61), (30, 60), (10, 30), (1, 10), (0, 1)]


def parse(path, tol):
    """Score every primary mate-1 record in `path`.

    Returns `(n, mapped, correct, per_bucket)` where `per_bucket` maps a bucket label to its
    `[seen, correct]` pair. A record counts as correct when its POS is within `tol` of the truth
    encoded in its name; `tol` absorbs the indels the generator plants, which shift the leftmost
    position without making the placement wrong.
    """
    n = mapped = correct = 0
    per_bucket = {f"{lo}-{hi - 1}" if hi - lo > 1 else str(lo): [0, 0] for lo, hi in BUCKETS}
    contigs = set()

    with open(path) as fh:
        for line in fh:
            if line.startswith("@"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            name, flag, rname, pos, mapq = f[0], int(f[1]), f[2], int(f[3]), int(f[4])
            # Primary mate 1 only: skip secondary (0x100), supplementary (0x800), and mate 2
            # (0x80). Without the first two a multi-mapping read would be counted several times.
            if flag & 0x100 or flag & 0x800 or flag & 0x80:
                continue
            if "_" not in name or not name.startswith("sim"):
                continue
            n += 1
            if flag & 0x4:  # unmapped
                continue
            mapped += 1
            contigs.add(rname)
            truth = int(name.rsplit("_", 1)[1]) + 1  # 0-based offset -> 1-based SAM POS
            ok = abs(pos - truth) <= tol
            correct += ok
            for lo, hi in BUCKETS:
                if lo <= mapq < hi:
                    label = f"{lo}-{hi - 1}" if hi - lo > 1 else str(lo)
                    per_bucket[label][0] += 1
                    per_bucket[label][1] += ok
                    break

    if len(contigs) > 1:
        sys.exit(
            f"error: {len(contigs)} contigs in the output, but the truth encoding assumes a "
            f"single-contig reference. Re-run against a single-contig fixture."
        )
    return n, mapped, correct, per_bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sam")
    ap.add_argument("--tol", type=int, default=5, help="bp of slack on the reported position")
    args = ap.parse_args()

    n, mapped, correct, per_bucket = parse(args.sam, args.tol)
    if n == 0:
        sys.exit("error: no `sim<i>_<start>` reads found; this needs make_test_reads.py output")

    pct = lambda a, b: f"{100.0 * a / b:6.2f}%" if b else "     -"
    print(f"reads (mate 1)      {n}")
    print(f"mapped              {mapped:>8}  {pct(mapped, n)}")
    print(f"correct (+/-{args.tol}bp)   {correct:>8}  {pct(correct, n)} of all, {pct(correct, mapped)} of mapped")
    print()
    print("  MAPQ      seen   correct    wrong   error rate")
    for lo, hi in BUCKETS:
        label = f"{lo}-{hi - 1}" if hi - lo > 1 else str(lo)
        seen, ok = per_bucket[label]
        # The error rate inside a MAPQ bucket is the number that matters: MAPQ is a promise about
        # exactly this, so a bucket whose error rate exceeds its own claim is a miscalibration.
        print(f"  {label:>7} {seen:>9} {ok:>9} {seen - ok:>8}   {pct(seen - ok, seen)}")


if __name__ == "__main__":
    main()
