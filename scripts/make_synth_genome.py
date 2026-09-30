#!/usr/bin/env python3
"""Generate a deterministic synthetic genome with a human-like repeat landscape.

Exists for hosts that cannot reach a real reference (a sandbox whose egress blocks the genome
mirrors, a CI runner) but still need an index LARGER THAN THE LAST-LEVEL CACHE. The tiny 200 kb
fixture keeps the whole BWT in L2, so seeding nearly vanishes from the profile and extension's share
inflates several-fold (see the warning in scripts/bench.sh). A 120 Mb genome gives a ~1 GB
index, past any L3 shipped today, so FM-index traffic is DRAM-bound the way it is on GRCh38.

What is modelled, and why each one matters to an aligner profile:

- background with a slowly varying GC content (isochores), so k-mer frequencies are not uniform;
- interspersed repeat families with per-copy divergence: an Alu-like 300 bp family at high copy
  number, and an L1-like 6 kb family, often 5'-truncated. These are what make SMEMs hit many loci,
  drive the occurrence filter, chaining of multiple candidates, and secondary alignments;
- micro- and minisatellites (tandem repeats), where band retightening and z-drop are exercised;
- segmental duplications (1-40 kb copies at 1-5 % divergence), which create the MAPQ-0 and
  mate-rescue workload of real data.

It is NOT a substitute for real reads on GRCh38: the proportions are plausible, not measured, and a
result obtained here is a relative A/B on one host, never a headline number.

Usage:
    make_synth_genome.py OUT.fa [--mb 120] [--seed 7] [--contigs 4]
"""

import argparse
import random

BASES = "ACGT"


def mutate(seq, rate, rng):
    """Substitute, insert or delete at `rate` per base (80/10/10)."""
    out = []
    for c in seq:
        r = rng.random()
        if r >= rate:
            out.append(c)
            continue
        k = rng.random()
        if k < 0.8:
            out.append(rng.choice(BASES.replace(c, "")))
        elif k < 0.9:
            out.append(c)
            out.append(rng.choice(BASES))
        # else: deletion
    return "".join(out)


def background(n, rng):
    """Random sequence with GC drifting between isochore levels every ~100 kb."""
    parts = []
    left = n
    while left > 0:
        block = min(left, rng.randint(50_000, 300_000))
        gc = rng.choice([0.37, 0.41, 0.46, 0.53])
        w = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        parts.append("".join(rng.choices(BASES, weights=w, k=block)))
        left -= block
    return "".join(parts)


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def build_contig(n, rng, alu, l1):
    seq = list(background(n, rng))
    pieces = []  # (position, inserted string), applied as overwrites to keep length fixed

    # Alu-like: ~1 copy per 3 kb (GRCh38 has ~1.2 M Alus in 3.1 Gb, one per ~2.6 kb).
    for _ in range(n // 3000):
        s = mutate(alu, rng.uniform(0.02, 0.18), rng)
        pieces.append(s if rng.random() < 0.5 else revcomp(s))
    # L1-like: ~1 per 20 kb, most 5'-truncated.
    for _ in range(n // 20000):
        keep = len(l1) if rng.random() < 0.1 else rng.randint(500, len(l1))
        s = mutate(l1[-keep:], rng.uniform(0.03, 0.2), rng)
        pieces.append(s if rng.random() < 0.5 else revcomp(s))
    # Tandem repeats: unit 1-60 bp, 20-3000 bp long, slightly degenerate.
    for _ in range(n // 15000):
        unit = "".join(rng.choices(BASES, k=rng.choice([1, 2, 2, 3, 4, 5, 6, 12, 24, 48, 60])))
        s = (unit * (rng.randint(20, 3000) // len(unit) + 1))[: rng.randint(20, 3000)]
        pieces.append(mutate(s, 0.02, rng))

    for p in pieces:
        pos = rng.randrange(0, max(1, n - len(p)))
        seq[pos : pos + len(p)] = p
    seq = seq[:n]

    # Segmental duplications: copy an existing stretch elsewhere, 1-5 % diverged.
    for _ in range(n // 400_000):
        ln = rng.randint(1000, 40000)
        a = rng.randrange(0, n - ln)
        b = rng.randrange(0, n - ln)
        s = mutate("".join(seq[a : a + ln]), rng.uniform(0.01, 0.05), rng)[:ln]
        if rng.random() < 0.3:
            s = revcomp(s)
        seq[b : b + len(s)] = s
    # A few N gaps, as assemblies have.
    for _ in range(max(1, n // 20_000_000)):
        ln = rng.randint(1000, 50000)
        a = rng.randrange(0, n - ln)
        seq[a : a + ln] = "N" * ln
    return "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mb", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--contigs", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    alu = "".join(rng.choices(BASES, weights=[0.25, 0.3, 0.3, 0.15], k=300))
    l1 = "".join(rng.choices(BASES, weights=[0.33, 0.17, 0.17, 0.33], k=6000))
    total = a.mb * 1_000_000
    with open(a.out, "w") as f:
        for c in range(a.contigs):
            n = total // a.contigs
            s = build_contig(n, rng, alu, l1)
            f.write(f">synth{c + 1}\n")
            for i in range(0, len(s), 60):
                f.write(s[i : i + 60])
                f.write("\n")


if __name__ == "__main__":
    main()
