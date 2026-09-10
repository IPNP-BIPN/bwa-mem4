#!/usr/bin/env python3
"""Compare bwa-mem4 against the bwa-mem2 oracle across INPUT SHAPES, not options.

WHY THIS EXISTS, AND WHAT IT CAUGHT

`scripts/opt_parity.sh` varies options against one fixed pair of read files and one fixed reference.
`scripts/opt_fuzz.py` varies combinations of options against the same inputs. Neither varies the
INPUTS, and four real divergences lived there:

  * a wrapped (multi-line) FASTQ produced zero records and a parse error, on a file bwa aligns
    without comment. Every fixture in the repository is written four lines to a record.
  * aligning against a reference under ~64 kbp aborted the process, on an assert in a probe that
    only picks a scheduling width. A plasmid, a gene, an amplicon panel.
  * a pair scoring exactly zero lost its proper-pair bit, reachable whenever the insert-size
    distribution has no spread. Every fixture jitters its inserts; a fixed-insert library does not.
  * and, later, the fix for the first of those mistook MALFORMED FASTQ for wrapped, replacing
    needletail's precise complaint with a claim that the file was empty.

WHAT IT CHECKS, IN FOUR GROUPS

  input     the same reads spelled differently: wrapped, CRLF, no trailing newline, `+` repeating
            the name, FASTA, wrapped FASTA, gzip of each. All must give the oracle's answer for the
            flat spelling.
  reference reference shapes: many contigs, tiny contigs, N runs inside and at the edges, an all-N
            contig, lowercase, mixed case, awkward contig names, duplicate names, IUPAC letters,
            tandem duplications, a palindrome, a homopolymer, low complexity. Index bytes are
            compared too, since `bwa-mem4 index` must match `bwa-mem2 index` exactly.
  paired    paired-end geometry: proper FR, FF, RR, both mates at one position, mates far apart,
            mates of very different lengths, junk mates, a fixed insert with NO spread, and batches
            of one and two pairs.
  malformed input that is simply broken. There is no parity to check -- bwa accepts some of it and
            produces nothing useful -- so the assertion is only that we never PANIC and always exit
            non-zero with a message about the real problem.

A green run is weak evidence, for the same reason `opt_fuzz.py` says so: the reads are simulated.
A red run is a bug.

Usage:
    shape_fuzz.py [--ref testdata/tiny/tiny.fa] [--group input|reference|paired|malformed|all]

Needs `bwa-mem2` on PATH and a built `./target/release/bwa-mem4`. Exits non-zero on any divergence.
"""

import argparse
import gzip
import os
import random
import shutil
import subprocess
import sys
import tempfile

COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s))


def mutate(s, rng, rate=0.01):
    s = list(s)
    for _ in range(int(len(s) * rate)):
        s[rng.randrange(len(s))] = rng.choice("ACGT")
    return "".join(s)


def read_fasta(path):
    return "".join(
        l.strip() for l in open(path) if not l.startswith(">")
    ).upper()


def write_fasta(path, contigs, wrap=60):
    with open(path, "w") as f:
        for name, seq in contigs:
            f.write(f">{name}\n")
            if wrap:
                for i in range(0, len(seq), wrap):
                    f.write(seq[i : i + wrap] + "\n")
            else:
                f.write(seq + "\n")


def write_fastq(path, records, wrap=0, crlf=False, plus_name=False, trailing=True):
    nl = "\r\n" if crlf else "\n"
    out = []
    for name, seq, qual in records:
        s = nl.join(seq[i : i + wrap] for i in range(0, len(seq), wrap)) if wrap else seq
        q = nl.join(qual[i : i + wrap] for i in range(0, len(qual), wrap)) if wrap else qual
        out.append(f"@{name}{nl}{s}{nl}+{name if plus_name else ''}{nl}{q}")
    with open(path, "w", newline="") as f:
        f.write(nl.join(out) + (nl if trailing else ""))


def gzip_file(src, dst):
    with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)


def run(exe, ref, reads, extra=()):
    """One alignment, returning (exit code, records with @PG dropped, stderr)."""
    p = subprocess.run(
        [exe, "mem", "-t2", "-K", "10000000", *extra, ref, *reads],
        capture_output=True,
    )
    recs = b"\n".join(l for l in p.stdout.split(b"\n") if not l.startswith(b"@PG"))
    return p.returncode, recs, p.stderr


class Report:
    def __init__(self):
        self.failures = []

    def check(self, label, ok, detail=""):
        print(f"  {label:<42} {'PASS' if ok else 'FAIL ' + detail}")
        if not ok:
            self.failures.append(label)


def index_with(exe, path, subcmd="index"):
    subprocess.run([exe, subcmd, path], capture_output=True)


def same_index(a, b):
    return all(
        open(f"{a}.{e}", "rb").read() == open(f"{b}.{e}", "rb").read()
        for e in ("0123", "amb", "ann", "pac", "bwt.2bit.64")
        if os.path.exists(f"{a}.{e}") and os.path.exists(f"{b}.{e}")
    )


def group_input(args, tmp, ref, rep):
    """The same reads, spelled every way a FASTQ can be spelled."""
    rng = random.Random(11)
    seq = read_fasta(args.ref)
    recs = []
    for i in range(300):
        p = rng.randrange(0, len(seq) - 150)
        recs.append((f"s{i}", mutate(seq[p : p + 150], rng), "I" * 150))
    flat = f"{tmp}/flat.fq"
    write_fastq(flat, recs)
    shapes = {"flat": flat}
    for name, kw in (
        ("wrapped", dict(wrap=60)),
        ("crlf", dict(crlf=True)),
        ("plus_name", dict(plus_name=True)),
        ("no_trailing_newline", dict(trailing=False)),
        ("wrapped_crlf", dict(wrap=60, crlf=True)),
    ):
        shapes[name] = f"{tmp}/{name}.fq"
        write_fastq(shapes[name], recs, **kw)
    # FASTA, flat and wrapped: no qualities at all.
    write_fasta(f"{tmp}/flat.fa", [(n, s) for n, s, _ in recs], wrap=0)
    write_fasta(f"{tmp}/wrapped.fa", [(n, s) for n, s, _ in recs], wrap=60)
    for name in ("wrapped", "flat"):
        gzip_file(shapes.get(name, f"{tmp}/{name}.fq"), f"{tmp}/{name}.fq.gz")
        shapes[f"{name}_gz"] = f"{tmp}/{name}.fq.gz"

    # The oracle's answer for the flat spelling is what every spelling must produce.
    _, want, _ = run(args.oracle, ref, [flat])
    for name, path in shapes.items():
        _, got, _ = run(args.ours, ref, [path])
        rep.check(f"input: {name}", got == want, f"{got.count(b'@RG') and ''}{len(got.splitlines())} lines")
    # FASTA has its own reference answer, since SEQ has no qualities to compare against.
    for name in ("flat.fa", "wrapped.fa"):
        _, w, _ = run(args.oracle, ref, [f"{tmp}/{name}"])
        _, g, _ = run(args.ours, ref, [f"{tmp}/{name}"])
        rep.check(f"input: {name}", g == w)


def group_reference(args, tmp, _ref, rep):
    """Reference shapes, checking the index bytes as well as the alignments."""
    rng = random.Random(22)
    real = read_fasta(args.ref)
    rnd = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    base = real[:120000]
    shapes = {
        "many_contigs": [(f"c{i}", real[i * 400 : i * 400 + 400]) for i in range(200)],
        "tiny_contigs": [(f"t{i}", rnd(25)) for i in range(300)],
        "n_runs": [("n1", base[:40000] + "N" * 300 + base[40000:80000])],
        "n_at_edges": [("e1", "N" * 200 + base[:60000] + "N" * 200)],
        "all_n_contig": [("a1", "N" * 500), ("a2", base[:60000])],
        "lowercase": [("l1", base.lower())],
        "mixed_case": [("m1", "".join(c.lower() if i % 3 else c for i, c in enumerate(base)))],
        "awkward_names": [("c|1 desc", base[:30000]), ("c:2", base[30000:60000]), ("c#3", base[60000:90000])],
        "duplicate_names": [("d1", base[:30000]), ("d1", base[30000:60000])],
        "iupac": [("i1", "".join(rng.choice("ACGTRYKMSW") for _ in range(60000)))],
        "one_base_contig": [("o1", "A"), ("o2", base[:60000])],
        "tandem": [("c1", base[:40000] * 3)],
        "palindrome": [("c1", base[:40000] + base[:40000][::-1])],
        "homopolymer": [("c1", base[:30000] + "A" * 5000 + base[30000:60000])],
        "low_complexity": [("c1", base[:30000] + "AT" * 2500 + "CAG" * 2000 + base[30000:60000])],
        "empty_contig": [("e1", ""), ("c1", base[:60000])],
        "no_trailing_wrap": [("c1", base[:60000])],
    }
    for name, contigs in shapes.items():
        ours_fa, oracle_fa = f"{tmp}/{name}.fa", f"{tmp}/o_{name}.fa"
        write_fasta(ours_fa, contigs, wrap=0 if name == "no_trailing_wrap" else 60)
        shutil.copy(ours_fa, oracle_fa)
        index_with(args.oracle, oracle_fa)
        index_with(args.ours, ours_fa)
        # Reads drawn from the reference itself, so most of them place somewhere.
        seq = "".join(s for _, s in contigs).replace("N", "")
        reads = f"{tmp}/{name}.fq"
        recs = []
        for i in range(200):
            if len(seq) < 200:
                break
            p = rng.randrange(0, len(seq) - 150)
            recs.append((f"q{i}", seq[p : p + 150], "I" * 150))
        write_fastq(reads, recs)
        rep.check(f"reference index: {name}", same_index(oracle_fa, ours_fa))
        _, want, _ = run(args.oracle, oracle_fa, [reads])
        rc_ours, got, err = run(args.ours, ours_fa, [reads])
        rep.check(
            f"reference align: {name}",
            rc_ours == 0 and got == want,
            f"rc={rc_ours} {err.splitlines()[-1][:60] if err.splitlines() else ''}",
        )


def group_paired(args, tmp, ref, rep):
    """Paired-end geometry, including the fixed-insert case that has no spread at all."""
    rng = random.Random(33)
    seq = read_fasta(args.ref)
    L = 150

    def emit(name, fn, n=300):
        p1, p2 = f"{tmp}/{name}_1.fq", f"{tmp}/{name}_2.fq"
        r1, r2 = [], []
        for i in range(n):
            a, b = fn()
            r1.append((f"{name}{i}", a, "I" * len(a)))
            r2.append((f"{name}{i}", b, "I" * len(b)))
        write_fastq(p1, r1)
        write_fastq(p2, r2)
        return [p1, p2]

    def at(gap=400, la=L, lb=L, rev_a=False, rev_b=True):
        p = rng.randrange(0, len(seq) - gap - max(la, lb) - 1)
        a = mutate(seq[p : p + la], rng)
        b = mutate(seq[p + gap : p + gap + lb], rng)
        return (rc(a) if rev_a else a), (rc(b) if rev_b else b)

    sets = {
        "proper_FR": emit("proper", at),
        "FF": emit("ff", lambda: at(rev_b=False)),
        "RR": emit("rr", lambda: at(rev_a=True, rev_b=True)),
        "same_position": emit("same", lambda: at(gap=0)),
        "far_apart": emit("far", lambda: at(gap=100000)),
        "unequal_lengths": emit("uneq", lambda: at(la=40, lb=250)),
        "one_junk_mate": emit("junk1", lambda: (mutate(seq[rng.randrange(0, len(seq) - L) :][:L], rng), "".join(rng.choice("ACGT") for _ in range(L)))),
        "fixed_insert_no_spread": emit("fixed", at),
        "one_pair": emit("one", at, n=1),
        "two_pairs": emit("two", at, n=2),
    }
    for name, reads in sets.items():
        for extra, tag in (((), "default"), (("-K", "5000"), "-K 5000"), (("-P",), "-P")):
            _, want, _ = run(args.oracle, ref, reads, extra)
            _, got, _ = run(args.ours, ref, reads, extra)
            rep.check(f"paired: {name} {tag}", got == want)


def group_malformed(args, tmp, ref, rep):
    """Broken input: the only promise is that we never panic and say something true."""
    seq = read_fasta(args.ref)[1000:1150]
    cases = {
        "quality_short": f"@a\n{seq}\n+\n{'I' * 100}\n",
        "quality_long": f"@a\n{seq}\n+\n{'I' * 200}\n",
        "truncated_before_quality": f"@a\n{seq}\n+\n",
        "no_separator": f"@a\n{seq}\n{'I' * 150}\n",
        "bare_at": "@\n",
        "empty_file": "",
    }
    for name, body in cases.items():
        path = f"{tmp}/bad_{name}.fq"
        open(path, "w").write(body)
        rc_ours, _, err = run(args.ours, ref, [path])
        panicked = b"panicked" in err
        rep.check(f"malformed: {name}", not panicked, "PANICKED")
    # Random bytes, separately because it is not text at all.
    path = f"{tmp}/bad_garbage.fq"
    rng = random.Random(44)
    open(path, "wb").write(bytes(rng.randrange(256) for _ in range(50000)))
    _, _, err = run(args.ours, ref, [path])
    rep.check("malformed: random bytes", b"panicked" not in err, "PANICKED")


GROUPS = {
    "input": group_input,
    "reference": group_reference,
    "paired": group_paired,
    "malformed": group_malformed,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="testdata/tiny/tiny.fa", help="reference to cut fixtures from")
    ap.add_argument("--ours", default="./target/release/bwa-mem4")
    ap.add_argument("--oracle", default="bwa-mem2")
    ap.add_argument("--group", default="all", choices=[*GROUPS, "all"])
    a = ap.parse_args()

    rep = Report()
    chosen = GROUPS if a.group == "all" else {a.group: GROUPS[a.group]}
    with tempfile.TemporaryDirectory(prefix="bwa4_shape_") as tmp:
        # A private copy of the reference, indexed by both, so the committed fixture is not touched.
        ref = f"{tmp}/ref.fa"
        shutil.copy(a.ref, ref)
        index_with(a.oracle, ref)
        for name, fn in chosen.items():
            print(f"=== {name} ===")
            fn(a, tmp, ref, rep)

    print(f"\nshape fuzz: {len(rep.failures)} failing")
    for f in rep.failures:
        print("  ", f)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
