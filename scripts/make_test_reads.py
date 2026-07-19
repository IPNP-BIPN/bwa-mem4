#!/usr/bin/env python3
"""Generate a deterministic paired-end FASTQ pair from a reference FASTA.

Exists so CI can run the parity harness without committing a binary read fixture. The output is a
pure function of (reference, seed, count), so a CI failure is reproducible locally with the same
command line.

WHAT THIS CAN AND CANNOT CATCH. These reads are simulated from a reference, so most of them map to
a unique locus. That is exactly the blind spot documented in this project's history: the missing
discard pass diverged only inside tandem repeats, and a simulated-read gate scored 5000/5000 while
real GIAB reads diverged. So treat the CI parity job as a REGRESSION net (did this change break
something that used to work), not as proof of byte-identity. Proof still requires real reads at
depth, which is what scripts/giab30x_pe.sh does.

Usage:
    make_test_reads.py REF.fa OUT_PREFIX [--n 20000] [--len 150] [--seed 42]

Writes OUT_PREFIX_1.fq and OUT_PREFIX_2.fq.
"""

import argparse
import sys

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def read_fasta(path):
    """Concatenate every record's sequence, uppercased. Contig boundaries do not matter here:
    reads are drawn from the concatenation and a read that happens to straddle a boundary is
    still a legitimate input, it just will not align well."""
    seq = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                seq.append(line.strip().upper())
    return "".join(seq)


class Lcg:
    """The same 48-bit LCG as POSIX drand48, so a given seed reproduces bit for bit on every
    platform and every Python version. `random` is not used: its algorithm is not guaranteed
    stable across releases, which would make CI failures irreproducible."""

    def __init__(self, seed):
        self.state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def next(self):
        self.state = (self.state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.state >> 17

    def below(self, n):
        return self.next() % n


def mutate(bases, rng, sub_rate_pct, indel_rate_pct):
    """Apply substitutions and short indels so the reads exercise the DP, not just exact seeding.
    A read with no mismatches never leaves the ungapped fast path, and would test almost nothing."""
    out = []
    for b in bases:
        roll = rng.below(1000)
        if roll < sub_rate_pct:
            # Substitute: pick one of the three other bases so the change is always visible.
            out.append("ACGT"[(("ACGT".find(b) if b in "ACGT" else 0) + 1 + rng.below(3)) % 4])
        elif roll < sub_rate_pct + indel_rate_pct:
            if rng.below(2):
                continue  # deletion: drop this base
            out.append(b)
            out.append("ACGT"[rng.below(4)])  # insertion
        else:
            out.append(b)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("out_prefix")
    ap.add_argument("--n", type=int, default=20000, help="read pairs to emit")
    ap.add_argument("--len", dest="rlen", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--insert", type=int, default=350, help="mean fragment length")
    ap.add_argument("--sub-rate", type=int, default=8, help="substitutions per 1000 bases")
    ap.add_argument("--indel-rate", type=int, default=1, help="indels per 1000 bases")
    args = ap.parse_args()

    ref = read_fasta(args.reference)
    if len(ref) < args.insert * 2:
        sys.exit(f"reference too short ({len(ref)} bp) for insert {args.insert}")

    rng = Lcg(args.seed)
    span = len(ref) - args.insert - 1

    with open(f"{args.out_prefix}_1.fq", "w") as f1, open(f"{args.out_prefix}_2.fq", "w") as f2:
        for i in range(args.n):
            # Fragment start, then a fragment length jittered around the mean so mem_pestat sees a
            # real insert-size distribution rather than a single spike.
            start = rng.below(span)
            frag = args.insert + rng.below(101) - 50
            frag = max(frag, args.rlen + 10)
            end = min(start + frag, len(ref))

            r1 = mutate(ref[start : start + args.rlen], rng, args.sub_rate, args.indel_rate)
            # Mate 2 is the reverse complement of the fragment's far end: the FR orientation every
            # Illumina library produces, and the one mem_pestat expects to find.
            mate_src = ref[max(end - args.rlen, 0) : end]
            r2 = mutate(mate_src.translate(COMPLEMENT)[::-1], rng, args.sub_rate, args.indel_rate)

            name = f"sim{i}_{start}"
            # Fixed quality string: quality does not affect placement in bwa mem (it is not used in
            # scoring), so a constant keeps the fixture small and the diff readable.
            f1.write(f"@{name}/1\n{r1}\n+\n{'I' * len(r1)}\n")
            f2.write(f"@{name}/2\n{r2}\n+\n{'I' * len(r2)}\n")


if __name__ == "__main__":
    main()
