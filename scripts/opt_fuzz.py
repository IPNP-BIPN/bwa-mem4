#!/usr/bin/env python3
"""Compare bwa-mem4 against the bwa-mem2 oracle on RANDOM COMBINATIONS of options.

WHY THIS EXISTS, AND WHAT IT CAUGHT

`scripts/opt_parity.sh` checks options one at a time, which is the right shape for "is this option
implemented" but blind to a whole class of bug: an option whose effect only becomes VISIBLE once a
second option puts the aligner in the right state. Three real divergences were found this way after
that sweep had been green for months:

  * `-5` set `MEM_F_PRIMARY5` and nothing read it. At the default `-w` these fixtures produce almost
    no split alignments, so `-5` alone had nothing to reorder and passed. It took `-w 3 -5`.
  * `mem_reg2sam` ignored `-M`. Reachable only once a read emits a supplementary on that path, which
    needs `-X` above 1.0 to stop overlapping alignments being marked as shadowed.
  * the same branch capped supplementary MAPQ under `-q`, which exists to suppress the cap.

So: a green run here is weak evidence and a red run is a bug. Treat it as a hunting tool, not a
gate; `opt_parity.sh` is the gate, and every divergence this finds should end up there as a fixed
case with a fixture that has teeth.

WHAT IT DOES NOT COVER. The reads are simulated, so most map to a unique locus; this is the same
blind spot `make_test_reads.py` documents. It also cannot see anything that needs a real ALT index,
and it deliberately never passes `-x`, three of whose four presets are routed away from bwa's code
on purpose.

Usage:
    opt_fuzz.py [--seed N] [--n N] [--idx REF.fa] [--reads R1.fq[,R2.fq]]...

`--reads` may be given several times; each is one read set to draw from, single-end if it names one
file and paired if it names two. Defaults to the work/ scratch files the other scripts build.
Exits non-zero if any combination diverges.
"""

import argparse
import itertools
import random
import subprocess
import sys

# Options that take a value, with the range to draw from. Ranges stay inside what bwa documents,
# except `-X` (mask_level), which is deliberately allowed past 1.0 because that is what exposed the
# `-M` bug: above 1.0 the overlap test can never be satisfied and reads emit supplementaries they
# otherwise would not.
VALUED = [
    ("-k", lambda r: str(r.randint(10, 30))),      # min seed length
    ("-w", lambda r: str(r.randint(0, 200))),      # band width
    ("-d", lambda r: str(r.randint(20, 200))),     # z-dropoff
    ("-r", lambda r: f"{r.uniform(1.0, 3.0):.1f}"),  # re-seed trigger
    ("-y", lambda r: str(r.randint(5, 40))),       # seed occurrence for the 3rd round
    ("-c", lambda r: str(r.randint(50, 1000))),    # skip seeds with more than this many occurrences
    ("-D", lambda r: f"{r.uniform(0.1, 0.9):.2f}"),  # drop chains shorter than this fraction
    ("-W", lambda r: str(r.randint(0, 60))),       # min chain weight
    ("-m", lambda r: str(r.randint(5, 100))),      # max mate-rescue rounds
    ("-A", lambda r: str(r.randint(1, 5))),        # match score
    ("-B", lambda r: str(r.randint(1, 9))),        # mismatch penalty
    ("-O", lambda r: str(r.randint(2, 20))),       # gap open
    ("-E", lambda r: str(r.randint(1, 5))),        # gap extend
    ("-L", lambda r: str(r.randint(0, 10))),       # clipping penalty
    ("-U", lambda r: str(r.randint(1, 30))),       # unpaired penalty
    ("-T", lambda r: str(r.randint(10, 60))),      # min score to output
    ("-N", lambda r: str(r.randint(1, 10))),       # max chain extension rounds
    ("-G", lambda r: str(r.randint(100, 50000))),  # max gap length
    ("-X", lambda r: f"{r.uniform(0.3, 2.0):.2f}"),  # mask_level; see the note above
    ("-Q", lambda r: str(r.randint(1, 30))),       # min base quality
    ("-s", lambda r: str(r.randint(1, 20))),       # split-seed length
    ("-h", lambda r: str(r.randint(1, 20))),       # max XA hits
]

# Flags with no value. `-x` is absent on purpose: `pacbio`, `pbref` and `ont2d` are routed to rammap
# and are not meant to reproduce bwa-mem2, so asserting parity for them would assert the opposite of
# what the code promises. `scripts/opt_parity.sh` owns `-x intractg`, which does stay on bwa's path.
BOOLEAN = ["-a", "-M", "-Y", "-C", "-V", "-5", "-j", "-S", "-P"]


def run(exe, args, idx, reads):
    """One aligner invocation, returning its exit code and its records with `@PG` dropped.

    `@PG` legitimately differs between the two tools (it records which binary wrote the file), so it
    is excluded here exactly as every other script in this repository excludes it.
    """
    p = subprocess.run(
        [exe, "mem", "-t2", "-K", "10000000", *args, idx, *reads],
        capture_output=True,
    )
    records = b"\n".join(l for l in p.stdout.split(b"\n") if not l.startswith(b"@PG"))
    return p.returncode, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=200, help="combinations to try")
    ap.add_argument("--idx", default="work/region.fa")
    ap.add_argument("--ours", default="./target/release/bwa-mem4")
    ap.add_argument("--oracle", default="bwa-mem2")
    ap.add_argument(
        "--reads",
        action="append",
        default=None,
        help="one read set, as R1.fq or R1.fq,R2.fq; repeatable",
    )
    a = ap.parse_args()
    sets = [s.split(",") for s in (a.reads or ["work/f1.fq,work/f2.fq"])]

    rng = random.Random(a.seed)
    failures = []
    for _ in range(a.n):
        args = []
        for flag, gen in rng.sample(VALUED, rng.randint(1, 6)):
            args += [flag, gen(rng)]
        args += rng.sample(BOOLEAN, rng.randint(0, 3))
        chosen = rng.choice(sets)
        # Single-end runs of a paired set are worth drawing: several divergences found here were on
        # one path and not the other.
        reads = chosen if len(chosen) == 1 or rng.random() < 0.6 else chosen[:1]
        label = f"{' '.join(args)} [{' '.join(reads)}]"

        rc_oracle, oracle = run(a.oracle, args, a.idx, reads)
        if rc_oracle != 0:
            # The oracle refused these options or crashed on them; there is no output to compare
            # against, so this says nothing about us either way.
            print(f"[skip: oracle exited {rc_oracle}] {label}")
            continue
        rc_ours, ours = run(a.ours, args, a.idx, reads)
        if rc_ours != 0:
            print(f"[FAIL: exited {rc_ours}] {label}")
            failures.append(label)
            continue
        if ours != oracle:
            differing = sum(
                1
                for x, y in itertools.zip_longest(oracle.split(b"\n"), ours.split(b"\n"))
                if x != y
            )
            print(f"[DIFF: {differing} records] {label}")
            failures.append(label)

    print(f"\nfuzz: {a.n} combinations, {len(failures)} failing")
    for f in failures:
        print("  ", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
