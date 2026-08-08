#!/usr/bin/env bash
# GPU byte-identity gate (issue #54). Written BEFORE the first GPU kernel, on purpose: a GPU backend
# that is 99.99% right is a wrong backend, and the precedent (GPU-BWA-MEM, ICS 2023) claims identity
# with no published validation method. This script is the method.
#
# It checks three things, in this order:
#
#   1. DETERMINISM. Three runs of the CPU path must produce three EQUAL md5s. Nothing else here
#      means anything if the reference is not reproducible, and on a GPU this is the check that
#      catches an uncorrected memory error or a driver that reorders.
#   2. PARITY. BWA4_GPU=off against BWA4_GPU=<backend>, same binary, same reads: equal md5.
#   3. SPLIT INVARIANCE (issue #54 trap 7). BWA4_GPU_SPLIT sweeps the fraction of jobs sent to the
#      GPU over 0.0, 0.25, 0.5, 0.75, 1.0 and all five md5s must be equal. If that passes, dynamic
#      CPU/GPU co-scheduling is safe by construction and never has to be re-proved.
#
# Steps 2 and 3 SKIP, loudly, until a GPU backend exists. Step 1 runs today and is a real gate.
#
# Usage: scripts/gpu_parity.sh [backend]
#   backend: metal | cuda, default taken from BWA4_GPU_BACKEND, else "metal"
# Environment:
#   IDX      reference index                (default work/genome.fa)
#   R1, R2   paired FASTQ                   (default work/r1_500k.fq / work/r2_500k.fq)
#   T        threads                        (default 8)
#   K        batch size, fixed on purpose   (default 100000000)
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${1:-${BWA4_GPU_BACKEND:-metal}}"
IDX="${IDX:-work/genome.fa}"
R1="${R1:-work/r1_500k.fq}"
R2="${R2:-work/r2_500k.fq}"
T="${T:-8}"
K="${K:-100000000}"
BIN=target/release/bwa-mem4

[ -f "$IDX.ann" ] || { echo "missing index $IDX (set IDX=)" >&2; exit 1; }
for f in "$R1" "$R2"; do [ -f "$f" ] || { echo "missing reads $f" >&2; exit 1; }; done

cargo build --release --quiet

# md5 of the ALIGNMENT RECORDS only. The @PG header carries the command line, which differs between
# any two invocations that pass different options, so including it would make every comparison here
# fail for a reason that has nothing to do with alignment.
body_md5() {
  env "$@" "$BIN" mem -t"$T" -K "$K" "$IDX" "$R1" "$R2" 2>/dev/null | grep -v '^@' | md5sum 2>/dev/null | awk '{print $1}' ||
  env "$@" "$BIN" mem -t"$T" -K "$K" "$IDX" "$R1" "$R2" 2>/dev/null | grep -v '^@' | md5
}

echo "[1/3] determinism: three CPU runs must agree"
D1=$(body_md5 BWA4_GPU=off)
D2=$(body_md5 BWA4_GPU=off)
D3=$(body_md5 BWA4_GPU=off)
echo "  $D1"
echo "  $D2"
echo "  $D3"
if [ "$D1" != "$D2" ] || [ "$D2" != "$D3" ]; then
  echo "DETERMINISM: FAIL (the reference path is not reproducible; nothing below is meaningful)" >&2
  exit 1
fi
echo "  determinism: PASS"
REF="$D1"

# `version` lists the backends the binary was built with. Until a GPU one appears, the two gates
# below cannot run, and they say so rather than passing vacuously.
if ! "$BIN" version 2>&1 | grep -qi "$BACKEND"; then
  echo
  echo "[2/3] parity vs $BACKEND: SKIPPED, no $BACKEND backend in this binary"
  echo "[3/3] BWA4_GPU_SPLIT sweep:  SKIPPED, same reason"
  echo
  echo "GPU PARITY GATE: reference md5 $REF (determinism only; the GPU arms are not built yet)"
  exit 0
fi

echo
echo "[2/3] parity: BWA4_GPU=off vs BWA4_GPU=$BACKEND"
G=$(body_md5 "BWA4_GPU=$BACKEND")
echo "  cpu $REF"
echo "  gpu $G"
[ "$REF" = "$G" ] || { echo "PARITY: FAIL" >&2; exit 1; }
echo "  parity: PASS"

echo
echo "[3/3] split invariance: BWA4_GPU_SPLIT in 0.0 0.25 0.5 0.75 1.0"
for s in 0.0 0.25 0.5 0.75 1.0; do
  M=$(body_md5 "BWA4_GPU=$BACKEND" "BWA4_GPU_SPLIT=$s")
  printf '  split %-5s %s\n' "$s" "$M"
  [ "$REF" = "$M" ] || { echo "SPLIT INVARIANCE: FAIL at $s" >&2; exit 1; }
done

echo
echo "GPU PARITY GATE: PASS (md5 $REF, determinism + parity + split invariance)"
