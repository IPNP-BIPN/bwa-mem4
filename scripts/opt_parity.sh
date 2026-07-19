#!/usr/bin/env bash
# Per-option parity harness: for each bwa-mem2 option, run both aligners with it and compare the
# alignment records byte-for-byte (@PG legitimately differs: it carries each tool's name/command).
# An option that is parsed but not acted upon shows up here as a FAIL, which is the point.
#
# Usage: scripts/opt_parity.sh [path-to-bwa-mem3]
set -uo pipefail
cd "$(dirname "$0")/.."

M3="${1:-./target/release/bwa-mem3}"
M2=bwa-mem2
IDX=work/region.fa
R1=work/r1_50k.fq
R2=work/r2_50k.fq
# small, fast inputs; fall back to the 500k set if the 50k one was cleaned away
[ -f "$R1" ] || { R1=work/r1_500k.fq; R2=work/r2_500k.fq; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0; failed_opts=()

# $1 = label, $2 = "se"|"pe", rest = option words
# Modes `se!`/`pe!` mean "the caller supplies -t and -K itself". Needed because clap rejects a
# repeated flag where bwa's getopt silently lets the last one win, so testing -t or -K by appending
# a second copy would fail on the CLI layer and tell us nothing about the aligner.
check() {
  local label="$1" mode="$2"; shift 2
  # `${base[@]+"${base[@]}"}` at the call sites, not a bare `"${base[@]}"`: macOS ships bash 3.2,
  # which treats an empty array's expansion as an unbound variable under `set -u`, and `set -u`
  # exits the shell, firing the EXIT trap that deletes $TMP. Every later case then "fails" against
  # a missing directory.
  local base=(-t2 -K 10000000)
  case "$mode" in *'!') base=(); mode="${mode%!}";; esac
  local reads=("$R1"); [ "$mode" = pe ] && reads=("$R1" "$R2")
  $M2 mem ${base[@]+"${base[@]}"} "$@" "$IDX" "${reads[@]}" 2>/dev/null | grep -v '^@PG' > "$TMP/a.sam"
  local rc2=$?
  $M3 mem ${base[@]+"${base[@]}"} "$@" "$IDX" "${reads[@]}" 2>/dev/null | grep -v '^@PG' > "$TMP/b.sam"
  local rc3=$?
  if [ $rc3 -ne 0 ] && [ $rc2 -eq 0 ]; then
    printf '  %-28s %-3s [FAIL] mem3 exited non-zero\n' "$label" "$mode"; fail=$((fail+1)); failed_opts+=("$label"); return
  fi
  if cmp -s "$TMP/a.sam" "$TMP/b.sam"; then
    printf '  %-28s %-3s [PASS]\n' "$label" "$mode"; pass=$((pass+1))
  else
    # Count differing RECORDS, not bytes. `cmp -l` compares by byte position, so a single
    # length difference shifts the rest of the file and reports a number in the millions for
    # what is actually a handful of records. That mistake has been made here before.
    local l2 l3 d
    l2=$(wc -l <"$TMP/a.sam"|tr -d ' '); l3=$(wc -l <"$TMP/b.sam"|tr -d ' ')
    d=$(paste -d'\x01' "$TMP/a.sam" "$TMP/b.sam" | awk -F'\x01' '$1 != $2' | wc -l | tr -d ' ')
    printf '  %-28s %-3s [FAIL] %s recs mem2 / %s mem3, %s differing records\n' "$label" "$mode" "$l2" "$l3" "$d"
    fail=$((fail+1)); failed_opts+=("$label")
  fi
}

echo "=== baseline (no options) ==="
check "(defaults)" se
check "(defaults)" pe

echo "=== algorithm scalars ==="
check "-k 15"  se -k 15
check "-w 50"  se -w 50
check "-d 50"  se -d 50
check "-r 2.0" se -r 2.0
check "-y 10"  se -y 10
check "-c 100" se -c 100
check "-D 0.3" se -D 0.3
check "-W 10"  se -W 10
check "-m 20"  pe -m 20

echo "=== scoring ==="
check "-A 2"    se -A 2
check "-B 3"    se -B 3
check "-O 5"    se -O 5
check "-O 5,7"  se -O 5,7
check "-E 2"    se -E 2
check "-E 2,3"  se -E 2,3
check "-L 3"    se -L 3
check "-L 3,4"  se -L 3,4
check "-U 10"   pe -U 10
check "-T 20"   se -T 20
check "-A 2 -B 3" se -A 2 -B 3

echo "=== flags affecting output ==="
check "-a"     se -a
check "-M"     se -M
check "-Y"     se -Y
check "-C"     se -C
check "-V"     se -V
check "-5"     se -5
check "-q"     se -q
check "-j"     se -j
check "-h 3"   se -h 3
check "-h 3,100" se -h 3,100
check "-S"     pe -S
check "-P"     pe -P

echo "=== I/O features ==="
check "-R rg"  se -R '@RG\tID:foo\tSM:bar'
check "-I 300" pe -I 300
check "-v 1"   se -v 1

# Options that were implemented but NEVER exercised until 2026-07-18. That gap is not academic:
# `-N` was off by one chain the whole time (the C's `break` skips the loop header's `++i`, so the
# chain that trips the cap is demoted; we skipped past it). Our `-N 1` hashed exactly equal to
# bwa-mem2's `-N 2`. Same shape as the two other parity bugs found this year: the failure always
# sits in whatever the gate does not run. Every option we accept must appear below.
echo "=== previously untested options ==="
check "-N 1"    se -N 1
check "-N 5"    se -N 5
check "-G 1000" se -G 1000
check "-X 0.8"  se -X 0.8
check "-Q 10"   se -Q 10
check "-s 5"    se -s 5
check "-K 5000" 'se!' -t2 -K 5000
check "-t 1"    'se!' -t1 -K 10000000
check "-t 4"    'se!' -t4 -K 10000000
check "-H hdr"  se -H '@CO\textra header line'
# `-w 1` pins the band-retry `prev` semantics: with w <= 1 the `max_off < (w>>1)+(w>>2)` acceptance
# test degenerates to `0 < 0`, so only the C's `prev = a->score` (not -1) accepts at round 0.
check "-w 1"    se -w 1
check "-w 0"    se -w 0

echo ""
echo "RESULT: $pass passed, $fail failed"
[ $fail -gt 0 ] && printf 'FAILING: %s\n' "${failed_opts[*]}"
exit 0
