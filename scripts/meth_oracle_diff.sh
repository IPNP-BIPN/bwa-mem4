#!/usr/bin/env bash
# `--meth` oracle diff: bwa-mem4 against fg-labs/bwa-mem3 0.16 (the fork, its prebuilt binary) on
# simulated directional bisulfite and TAPS reads over a small reference. Every configuration must
# give the same SAM byte for byte, @PG lines excepted (they name each tool).
#
#   scripts/meth_oracle_diff.sh
#
# FORK    the fork's binary [reference/bwa-mem3-cpp/bwa-mem3.arm64]
# GENOME  a GRCh38 FASTA with .fai (contigs 20, 22, MT), only needed to build work/meth/ref.fa
#         the first time [work/giab/genome.fa]
# PAIRS   simulated pairs per library [5000]
# OURS    our binary [target/release/bwa-mem4, built first]
#
# The reference is chr20:44.0-44.4M, chr22:20.0-20.2M with a 30-N run and a 200 bp lowercase run
# injected, and chrM: two nuclear contigs, ambiguous bases, soft-masked bases and a contig end.
set -euo pipefail
cd "$(dirname "$0")/.."

FORK="${FORK:-reference/bwa-mem3-cpp/bwa-mem3.arm64}"
GENOME="${GENOME:-work/giab/genome.fa}"
PAIRS="${PAIRS:-5000}"
W=work/meth
mkdir -p "$W/fork" "$W/ours" "$W/sim" "$W/out"
[ -x "$FORK" ] || { echo "no fork binary at $FORK (set FORK=)" >&2; exit 1; }

if [ ! -s "$W/ref.fa" ]; then
  [ -s "$GENOME.fai" ] || { echo "need GENOME=<GRCh38.fa with .fai> to build $W/ref.fa" >&2; exit 1; }
  samtools faidx "$GENOME" 20:44000001-44400001 22:20000001-20200001 MT > "$W/raw.fa"
  python3 - "$W/raw.fa" "$W/ref.fa" <<'EOF'
import sys
names = {"20": "chr20", "22": "chr22", "MT": "chrM"}
seqs, order = {}, []
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith(">"):
        name = names[line[1:].split(":")[0]]
        order.append(name)
        seqs[name] = []
    else:
        seqs[name].append(line)
with open(sys.argv[2], "w") as out:
    for name in order:
        s = "".join(seqs[name])
        if name == "chr22":
            s = s[:100000] + "N" * 30 + s[100030:150000] + s[150000:150200].lower() + s[150200:]
        out.write(f">{name}\n")
        for i in range(0, len(s), 60):
            out.write(s[i : i + 60] + "\n")
EOF
fi

# OURS: another build of ours (e.g. an x86_64 one under Rosetta) instead of this tree's release.
if [ -z "${OURS:-}" ]; then
  cargo build --release --quiet
  OURS=target/release/bwa-mem4
fi

echo "[index] fork and ours, --meth"
for t in fork ours; do cp "$W/ref.fa" "$W/$t/ref.fa"; done
"$FORK" index --meth "$W/fork/ref.fa" > /dev/null 2>&1
"$OURS" index --meth "$W/ours/ref.fa" 2> /dev/null
for f in ref.fa.meth.fa ref.fa.pac ref.fa.ann ref.fa.amb ref.fa.meth.pac ref.fa.meth.ann ref.fa.meth.amb; do
  cmp -s "$W/fork/$f" "$W/ours/$f" || { echo "  index file $f DIFFERS" >&2; exit 1; }
done
# The fork's .bwt.2bit.64 carries an 8-byte footer the format does not need (see index_diff.sh).
for f in ref.fa.bwt.2bit.64 ref.fa.meth.bwt.2bit.64; do
  n=$(stat -f %z "$W/ours/$f" 2>/dev/null || stat -c %s "$W/ours/$f")
  head -c "$n" "$W/fork/$f" | cmp -s - "$W/ours/$f" || { echo "  index file $f DIFFERS" >&2; exit 1; }
done
echo "  identical (bwt files up to the fork's footer)"

echo "[sim] $PAIRS EM-seq pairs, $PAIRS TAPS pairs"
python3 scripts/meth_sim.py "$W/ref.fa" "$W/sim/em" --pairs "$PAIRS" --seed 11
python3 scripts/meth_sim.py "$W/ref.fa" "$W/sim/taps" --pairs "$PAIRS" --seed 7 --taps

E=("$W/sim/em_1.fq" "$W/sim/em_2.fq")
T=("$W/sim/taps_1.fq" "$W/sim/taps_2.fq")
fail=0
# One configuration: NAME, then the read files (1 or 2), `--`, then the mem options.
run() {
  local name=$1; shift
  local reads=()
  while [ "$1" != "--" ]; do reads+=("$1"); shift; done
  shift
  "$FORK" mem "$@" "$W/fork/ref.fa" "${reads[@]}" 2> /dev/null | grep -v '^@PG' > "$W/out/$name.fork.sam"
  "$OURS" mem "$@" "$W/ours/ref.fa" "${reads[@]}" 2> /dev/null | grep -v '^@PG' > "$W/out/$name.ours.sam"
  local n d
  n=$(grep -vc '^@' "$W/out/$name.fork.sam" || true)
  d=$(diff "$W/out/$name.fork.sam" "$W/out/$name.ours.sam" | grep -c '^<' || true)
  printf '  %-12s %6s records  %s\n' "$name" "$n" "$([ "$d" = 0 ] && echo identical || echo "$d DIFFER")"
  if [ "$n" = 0 ] || [ "$d" != 0 ]; then
    fail=1
    diff "$W/out/$name.fork.sam" "$W/out/$name.ours.sam" | head -6 | cut -c1-200 >&2 || true
  fi
}

echo "[mem] fork vs ours, SAM without @PG"
# What Bismark runs (FelixKrueger/Bismark#1209), on TAPS-like reads.
run bismark "${T[@]}" -- --meth=5base --meth-scoring neutral --meth-context all \
  --meth-seed-prune=spec30 --meth-tags none -K 100000000 -t1
run emseq_pe "${E[@]}" -- --meth -K 10000000 -t1
run emseq_se "${E[0]}" -- --meth -K 10000000 -t1
run taps_pe "${T[@]}" -- --meth=taps -K 10000000 -t1
run taps_se "${T[0]}" -- --meth=taps -K 10000000 -t1
run genomic "${E[@]}" -- --meth --meth-scoring genomic -K 10000000 -t1
run neutral "${E[@]}" -- --meth=bisulfite --meth-scoring=neutral --meth-tags=^XM -K 10000000 -t1
run tags_xr_xg "${E[@]}" -- --meth --meth-tags XR,XG -K 10000000 -t1
run prune_off "${E[@]}" -- --meth --meth-seed-prune=off -K 10000000 -t1
run baseline "${E[@]}" -- --meth --meth-seed-prune=baseline -K 10000000 -t1
run qc "${E[@]}" -- --meth --set-as-failed r --chimera-qc -K 10000000 -t1
run threads "${E[@]}" -- --meth -K 10000000 -t4
run scaled "${E[@]}" -- --meth -A2 -K 10000000 -t1
run all_alns "${E[0]}" -- --meth -a -K 10000000 -t1
run softclip "${E[@]}" -- --meth -Y -K 10000000 -t1

if [ "$fail" = 0 ]; then echo "PASS: every configuration identical"; else echo "FAIL" >&2; exit 1; fi
