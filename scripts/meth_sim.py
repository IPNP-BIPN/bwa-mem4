#!/usr/bin/env python3
"""Simulate directional bisulfite / EM-seq (or TAPS) paired-end reads, for `meth_oracle_diff.sh`.

    meth_sim.py REF.fa OUT_PREFIX [--pairs N] [--len L] [--seed S] [--taps]

Writes OUT_PREFIX_1.fq, OUT_PREFIX_2.fq and OUT_PREFIX_truth.tsv (qname, origin). Directional
library: read 1 comes from the original top (OT) or original bottom (OB) strand and carries C->T,
read 2 is its mate and carries G->A.
Methylation: CpG 75%, elsewhere 2%. EM-seq converts UNmethylated C, TAPS (--taps) methylated C.
On top of that: 0.5% sequencing errors, a SNP every ~1 kb (shared by all reads, so some look
like conversions), 5% fragments with a 1-3 bp indel, 3% chimeric read 1, 1% reads with an N,
2% unmappable random pairs, and a FASTQ comment carrying
SAM tags on 5% of reads so the comment pass-through is exercised. Deterministic for a seed.
"""
import argparse
import random

COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(COMP)[::-1]


def read_fasta(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf).upper()
    return seqs


def convert(strand, rng, taps):
    """Bisulfite-convert one strand 5'->3' (only its own Cs change)."""
    out = list(strand)
    for i, c in enumerate(strand):
        if c != "C":
            continue
        cpg = i + 1 < len(strand) and strand[i + 1] == "G"
        methylated = rng.random() < (0.75 if cpg else 0.02)
        if methylated == taps:  # EM-seq converts unmethylated C, TAPS methylated C
            out[i] = "T"
    return "".join(out)


def noisy(read, rng):
    out = list(read)
    for i in range(len(out)):
        if rng.random() < 0.005:
            out[i] = rng.choice([b for b in "ACGT" if b != out[i]])
    if rng.random() < 0.01:
        out[rng.randrange(len(out))] = "N"
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--len", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--taps", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    genome = read_fasta(a.ref)
    # Shared SNPs, so a variant can sit where a conversion could.
    for name, s in list(genome.items()):
        s = list(s)
        for i in range(0, len(s), 1000):
            j = min(len(s) - 1, i + rng.randrange(1000))
            if s[j] in "ACGT":
                s[j] = rng.choice([b for b in "ACGT" if b != s[j]])
        genome[name] = "".join(s)
    names = list(genome)
    weights = [len(genome[n]) for n in names]
    L = a.len
    with open(a.out + "_1.fq", "w") as f1, open(a.out + "_2.fq", "w") as f2, open(
        a.out + "_truth.tsv", "w"
    ) as truth:
        for k in range(a.pairs):
            qname = f"sim{k}"
            if rng.random() < 0.02:
                r1 = "".join(rng.choice("ACGT") for _ in range(L))
                r2 = "".join(rng.choice("ACGT") for _ in range(L))
                where = "random"
            else:
                while True:
                    name = rng.choices(names, weights)[0]
                    g = genome[name]
                    frag = max(L, int(rng.gauss(300, 50)))
                    if len(g) <= frag:
                        continue
                    s = rng.randrange(len(g) - frag)
                    top = g[s : s + frag]
                    if top.count("N") < frag // 2:
                        break
                strand = "OT" if rng.random() < 0.5 else "OB"
                src = convert(top if strand == "OT" else rc(top), rng, a.taps)
                if rng.random() < 0.05:  # a short indel, so XM:Z meets I and D
                    i, n = rng.randrange(20, len(src) - 20), rng.randrange(1, 4)
                    src = src[:i] + src[i + n :] if rng.random() < 0.5 else src[:i] + "A" * n + src[i:]
                r1, r2 = src[:L], rc(src[-L:])
                if rng.random() < 0.03:  # chimeric read 1: split alignment, SA:Z, hard clips
                    other = genome[rng.choices(names, weights)[0]]
                    o = rng.randrange(len(other) - L)
                    r1 = r1[: L // 2] + convert(other[o : o + L - L // 2], rng, a.taps)
                where = f"{name}:{s + 1}:{strand}"
            r1, r2 = noisy(r1, rng), noisy(r2, rng)
            comment = f" BC:Z:ACGTACGT\tXi:i:{k}" if rng.random() < 0.05 else ""
            q1 = "".join(chr(33 + rng.randrange(20, 41)) for _ in r1)
            q2 = "".join(chr(33 + rng.randrange(20, 41)) for _ in r2)
            truth.write(f"{qname}\t{where}\n")
            f1.write(f"@{qname}{comment}\n{r1}\n+\n{q1}\n")
            f2.write(f"@{qname}{comment}\n{r2}\n+\n{q2}\n")


if __name__ == "__main__":
    main()
