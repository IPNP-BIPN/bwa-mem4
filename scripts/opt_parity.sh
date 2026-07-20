#!/usr/bin/env bash
# Per-option parity harness: for each bwa-mem2 option, run both aligners with it and compare the
# alignment records byte-for-byte (@PG legitimately differs: it carries each tool's name/command).
# An option that is parsed but not acted upon shows up here as a FAIL, which is the point.
#
# Usage: scripts/opt_parity.sh [path-to-bwa-mem3]
set -uo pipefail
cd "$(dirname "$0")/.."

M3="${1:-./target/release/bwa-mem3}"
# Overridable so CI can point at the committed testdata/tiny fixture and generated reads. Locally
# they default to the scratch inputs under work/, which is gitignored.
M2="${M2:-bwa-mem2}"
IDX="${IDX:-work/region.fa}"
R1="${R1:-work/r1_50k.fq}"
R2="${R2:-work/r2_50k.fq}"
# small, fast inputs; fall back to the 500k set if the 50k one was cleaned away
[ -f "$R1" ] || { R1=work/r1_500k.fq; R2=work/r2_500k.fq; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# `-p` reads both mates from ONE file, mates adjacent. Built here rather than committed so it can
# never drift from $R1/$R2: any difference between this file and the two-file cases would be an
# artifact of the fixture, not of the aligner.
IL="$TMP/interleaved.fq"
paste -d'\n' <(paste - - - - <"$R1") <(paste - - - - <"$R2") | tr '\t' '\n' > "$IL"

pass=0; fail=0; failed_opts=()

# $1 = label, $2 = "se"|"pe"|"pi", rest = option words
# Modes `se!`/`pe!` mean "the caller supplies -t and -K itself". Needed because clap rejects a
# repeated flag where bwa's getopt silently lets the last one win, so testing -t or -K by appending
# a second copy would fail on the CLI layer and tell us nothing about the aligner.
# Mode `pi` is paired-end from the single interleaved file, i.e. what `-p` consumes.
check() {
  local label="$1" mode="$2"; shift 2
  # `${base[@]+"${base[@]}"}` at the call sites, not a bare `"${base[@]}"`: macOS ships bash 3.2,
  # which treats an empty array's expansion as an unbound variable under `set -u`, and `set -u`
  # exits the shell, firing the EXIT trap that deletes $TMP. Every later case then "fails" against
  # a missing directory.
  local base=(-t2 -K 10000000)
  case "$mode" in *'!') base=(); mode="${mode%!}";; esac
  local reads=("$R1")
  [ "$mode" = pe ] && reads=("$R1" "$R2")
  [ "$mode" = pi ] && reads=("$IL")
  $M2 mem ${base[@]+"${base[@]}"} "$@" "$IDX" "${reads[@]}" 2>/dev/null | grep -v '^@PG' > "$TMP/a.sam"
  local rc2=$?
  $M3 mem ${base[@]+"${base[@]}"} "$@" "$IDX" "${reads[@]}" 2>/dev/null | grep -v '^@PG' > "$TMP/b.sam"
  local rc3=$?
  if [ $rc3 -ne 0 ] && [ $rc2 -eq 0 ]; then
    printf '  %-28s %-3s [FAIL] mem3 exited non-zero\n' "$label" "$mode"; fail=$((fail+1)); failed_opts+=("$label"); return
  fi
  # KNOWN UPSTREAM DIVERGENCE, on `-A` cases only, and only against the x86_64 oracle.
  #
  # bwa-mem2 does not agree with itself across platforms under a non-default match score. Measured
  # on this fixture at `-A 2`: 205 of 8000 records differ. 194 differ in XS alone; the other 11
  # differ in POS/CIGAR/AS, and on those our score is NEVER lower than the x86 build's (5 strictly
  # higher, 6 equal). Our XS also scales exactly linearly in `-A` over 1..6 while the x86 build
  # breaks that linearity at `-A 2` alone, so we and the arm64 build are the consistent side. See
  # the README for the full table and the mechanism (bwamem.cpp:2302, where the 8-bit vs 16-bit
  # kernel choice moves with opt->a).
  #
  # Comparing against a disagreeing oracle proves nothing, so instead we pin OUR OWN output: any
  # change to it fails, which is the regression protection we actually want here. The pins come
  # from a binary verified against the arm64 oracle at 49/49. Regenerate deliberately, never to
  # make CI go green.
  if [ "${KNOWN_X86_XS_DIVERGENCE:-0}" = "1" ] && [ "$label" != "${label#-A }" ]; then
    local want got
    case "$label" in
      "-A 2")      want="ed462c8932d21ed640f7ad9448e724a8" ;;
      "-A 2 -B 3") want="5281a55603c17c2a4e71541a5de9f8fc" ;;
      *)           want="" ;;
    esac
    if [ -n "$want" ]; then
      got=$( (md5sum "$TMP/b.sam" 2>/dev/null || md5 -q "$TMP/b.sam") | awk '{print $1}' )
      if [ "$got" = "$want" ]; then
        printf '  %-28s %-3s [KNOWN] upstream x86/arm64 disagree; our output unchanged\n' "$label" "$mode"
        pass=$((pass+1)); return
      fi
      printf '  %-28s %-3s [FAIL] our output CHANGED (want %s, got %s)\n' "$label" "$mode" "${want:0:8}" "${got:0:8}"
      fail=$((fail+1)); failed_opts+=("$label"); return
    fi
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

# `-o` writes the SAM to a file instead of stdout. Compared against the oracle like any other
# option, except that neither side's output arrives on a pipe.
check_o() {
  local label="-o file" mode="pe"
  local base=(-t2 -K 10000000)
  $M2 mem "${base[@]}" -o "$TMP/o_a.sam" "$IDX" "$R1" "$R2" 2>/dev/null
  $M3 mem "${base[@]}" -o "$TMP/o_b.sam" "$IDX" "$R1" "$R2" 2>/dev/null
  local rc3=$?
  if [ $rc3 -ne 0 ]; then
    printf '  %-28s %-3s [FAIL] mem3 exited non-zero\n' "$label" "$mode"; fail=$((fail+1)); failed_opts+=("$label"); return
  fi
  grep -v '^@PG' "$TMP/o_a.sam" > "$TMP/o_a2.sam"
  grep -v '^@PG' "$TMP/o_b.sam" > "$TMP/o_b2.sam"
  if cmp -s "$TMP/o_a2.sam" "$TMP/o_b2.sam"; then
    printf '  %-28s %-3s [PASS]\n' "$label" "$mode"; pass=$((pass+1))
  else
    printf '  %-28s %-3s [FAIL] -o output differs from the oracle\n' "$label" "$mode"
    fail=$((fail+1)); failed_opts+=("$label")
  fi
}

# BGZF output (`-o out.gz`) is OURS, not bwa-mem2's, so there is no oracle to compare against and
# this is a round-trip check instead: decompressing it must reproduce the plain `-o` bytes exactly,
# and samtools must accept it. `@PG` is stripped on both sides because its `CL:` records the output
# filename, which necessarily differs between the two runs. Getting that wrong makes this test
# look broken when it is fine; it has already happened once.
check_bgzf() {
  local label="-o file.gz (bgzf)" mode="pe"
  local base=(-t2 -K 10000000)
  # Fail, do not skip. `bgzip` lives in the `tabix` package, not in `samtools`, and a missing tool
  # silently turning this case into a no-op is exactly how an output path stops being covered.
  for tool in bgzip samtools; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      printf '  %-28s %-3s [FAIL] %s not installed (apt: samtools tabix / brew: samtools htslib)\n' "$label" "$mode" "$tool"
      fail=$((fail+1)); failed_opts+=("$label"); return
    fi
  done
  $M3 mem "${base[@]}" -o "$TMP/z.sam"    "$IDX" "$R1" "$R2" 2>/dev/null
  $M3 mem "${base[@]}" -o "$TMP/z.sam.gz" "$IDX" "$R1" "$R2" 2>/dev/null
  if ! bgzip -d -c "$TMP/z.sam.gz" > "$TMP/z_rt.sam" 2>/dev/null; then
    printf '  %-28s %-3s [FAIL] bgzip could not decompress it\n' "$label" "$mode"
    fail=$((fail+1)); failed_opts+=("$label"); return
  fi
  grep -v '^@PG' "$TMP/z.sam"    > "$TMP/z2.sam"
  grep -v '^@PG' "$TMP/z_rt.sam" > "$TMP/z_rt2.sam"
  if ! cmp -s "$TMP/z2.sam" "$TMP/z_rt2.sam"; then
    printf '  %-28s %-3s [FAIL] round-trip differs from plain -o\n' "$label" "$mode"
    fail=$((fail+1)); failed_opts+=("$label"); return
  fi
  # A valid BGZF file is more than a gzip stream: samtools needs the BC extra field and the EOF
  # block, and `gzip -d` would happily accept a file that samtools rejects.
  if ! samtools view -H "$TMP/z.sam.gz" >/dev/null 2>&1; then
    printf '  %-28s %-3s [FAIL] samtools rejects the BGZF file\n' "$label" "$mode"
    fail=$((fail+1)); failed_opts+=("$label"); return
  fi
  printf '  %-28s %-3s [PASS]\n' "$label" "$mode"; pass=$((pass+1))
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

# `-p` and the output sinks were the last accepted options with no differential coverage at all,
# found while auditing for the 3.0.0 release. `-p` is a whole input path of its own (one file, mates
# adjacent, de-interleaved internally), not a scalar knob, so "it parses" proved nothing about it.
# The output-shaping flags above are all tested SINGLE-END, and single-end never reaches the paired
# emission branch or `mem_reg2sam`'s pairing fallback. Running `-a` paired-end for the first time,
# on 2026-07-20, immediately found a real byte-parity bug: we emitted XA:Z where bwa emits none,
# because `-a` (MEM_F_ALL) suppresses XA entirely and both PE emitters generated it unconditionally.
# Same shape as `-N` and `-p` before it: the failure sits wherever the gate does not run.
echo "=== output-shaping flags, paired-end ==="
check "-a (pe)"  pe -a
check "-M (pe)"  pe -M
check "-Y (pe)"  pe -Y
check "-5 (pe)"  pe -5
check "-q (pe)"  pe -q
check "-a -Y (pe)" pe -a -Y

echo "=== input and output paths ==="
check "-p (interleaved)" pi -p
check_o
check_bgzf

echo ""
echo "RESULT: $pass passed, $fail failed"
[ $fail -gt 0 ] && printf 'FAILING: %s\n' "${failed_opts[*]}"
exit 0
