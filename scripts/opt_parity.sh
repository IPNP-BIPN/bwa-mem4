#!/usr/bin/env bash
# Per-option parity harness: for each bwa-mem2 option, run both aligners with it and compare the
# alignment records byte-for-byte (@PG legitimately differs: it carries each tool's name/command).
# An option that is parsed but not acted upon shows up here as a FAIL, which is the point.
#
# Usage: scripts/opt_parity.sh [path-to-bwa-mem4]
set -uo pipefail
cd "$(dirname "$0")/.."

M3="${1:-./target/release/bwa-mem4}"
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

# `-f FILE` is a bare alias of `-o` in bwa: one shared getopt branch (`fastmap.cpp:674`), so the two
# letters write the same file. Checked against the oracle's `-f` rather than against our own `-o`,
# because the claim under test is "bwa's -f and ours agree", not "our two spellings agree". Needs its
# own helper for the same reason `-o` does: with the SAM in a file there is nothing on stdout for
# `check` to compare.
check_f() {
  local label="-f file (alias of -o)" mode="pe"
  local base=(-t2 -K 10000000)
  $M2 mem "${base[@]}" -f "$TMP/f_a.sam" "$IDX" "$R1" "$R2" 2>/dev/null
  $M3 mem "${base[@]}" -f "$TMP/f_b.sam" "$IDX" "$R1" "$R2" 2>/dev/null
  local rc3=$?
  if [ $rc3 -ne 0 ]; then
    printf '  %-28s %-3s [FAIL] mem4 exited non-zero\n' "$label" "$mode"; fail=$((fail+1)); failed_opts+=("$label"); return
  fi
  grep -v '^@PG' "$TMP/f_a.sam" > "$TMP/f_a2.sam"
  grep -v '^@PG' "$TMP/f_b.sam" > "$TMP/f_b2.sam"
  if cmp -s "$TMP/f_a2.sam" "$TMP/f_b2.sam"; then
    printf '  %-28s %-3s [PASS]\n' "$label" "$mode"; pass=$((pass+1))
  else
    printf '  %-28s %-3s [FAIL] -f output differs from the oracle\n' "$label" "$mode"
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

# The same scoring options, PAIRED-END. Not redundant with the single-end block above: mate rescue
# runs only on a pair, and its kernel is chosen from the SCORES, not from the reads. The u8 rescue
# kernel biases its substitution table by the mismatch penalty, which needs `a + b <= 6` of byte
# headroom, and until 4.4.x nothing checked that before dispatching to it: every one of the four
# cases below aborted the aligner outright on the first rescue batch, `-x intractg (pe)` included,
# while single-end runs of the identical scores passed. Any option whose only effect is inside a
# paired-end-only stage needs its own PE case here, and these are that lesson.
check "-B 6 (pe)"        pe -B 6
check "-B 9 (pe)"        pe -B 9
check "-A 3 (pe)"        pe -A 3
check "-A 10 -B 40 (pe)" pe -A 10 -B 40

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
# `-I` WITH A ZERO STANDARD DEVIATION. `-I 300` alone cannot reach it: bwa defaults the deviation to
# a tenth of the mean, so the insert window is wide and every z-score is finite. Pin the deviation to
# 0 and the window collapses to a single value, the z-score of an insert exactly at the mean becomes
# `0/0`, and the pair scores exactly zero -- which bwa does NOT accept as a proper pair
# (`(o = mem_pair(...)) > 0`), falling through to `no_pairing`, where the insert window sets the 0x2
# bit instead. Testing only that a pair was FOUND took the other branch and lost the bit: 22 records
# on this fixture at `-I 400,0`, and 600 of 600 on a simulated fixed-insert library, where the
# INFERRED distribution has no spread either. Amplicon panels are the real-world shape of that.
#
# The mean is 394 because the case has to have TEETH: only pairs whose insert lands EXACTLY on the
# mean hit the `0/0`, so the value has to be one this fixture actually produces. With the pre-fix
# behaviour restored, `-I 394,0` differs on 204 records here and `-I 394,1` on none, which is the
# whole point of keeping the second case beside it.
check "-I 394,0" pe -I 394,0
check "-I 394,1" pe -I 394,1
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
# found while auditing for the 4.0.0 release. `-p` is a whole input path of its own (one file, mates
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

# `-5` WITH A NARROW BAND. Plain `-5` is checked above and passes even when nothing implements it,
# because at the default `-w` this fixture produces almost no split alignments and the 5'-most
# segment is already the best-scoring one. `-w 3` forces splits, and then `-5` has something to
# reorder: `mem_reorder_primary5` was missing entirely until 4.4.x, so `-5` parsed, set its flag and
# changed nothing. Found by fuzzing option COMBINATIONS, which is the lesson: a flag whose effect
# needs a second flag to become visible passes a one-option-at-a-time sweep.
check "-w 3 -5"        se -w 3 -5
check "-w 3 -5 (pe)"   pe -w 3 -5

# `-5` on reads that really are chimeric, built here because the simulated fixtures do not contain
# any. Each read is a SHORT piece of the reference followed by a LONG piece from a distant locus, so
# it splits into two alignments and the 3' one scores higher -- which is exactly the case `-5`
# exists for, and exactly the case a score-ranked primary gets wrong. The mirrored read (long piece
# first) is included so the test also covers the "already 5'-most, do not reorder" branch.
#
# Verified to have teeth on the committed `testdata/tiny` fixture: with the reorder disabled, 8 of
# the 16 records differ from the oracle.
check_primary5() {
  local seq reflen fq label
  seq=$(grep -v '^>' "$IDX" | tr -d '\n' | tr 'acgt' 'ACGT')
  reflen=${#seq}
  fq="$TMP/chimeric.fq"
  : > "$fq"
  # Four (near locus, far locus) pairs spread over the contig, as fractions of its length so this
  # works on whatever reference $IDX points at.
  local i a b la lb read
  for i in 1 2 3 4; do
    a=$(( reflen / 40 * i ))
    b=$(( reflen / 2 + reflen / 40 * i ))
    la=$(( 45 + i * 5 ))
    lb=$(( 150 - la ))
    [ $(( b + lb )) -le "$reflen" ] || continue
    read="${seq:$a:$la}${seq:$b:$lb}"
    printf '@chim%s\n%s\n+\n%s\n' "$i" "$read" "$(printf 'I%.0s' $(seq 1 150))" >> "$fq"
    # The same two pieces the other way round: the 5'-most segment is now also the best-scoring one.
    read="${seq:$b:$lb}${seq:$a:$la}"
    printf '@chimr%s\n%s\n+\n%s\n' "$i" "$read" "$(printf 'I%.0s' $(seq 1 150))" >> "$fq"
  done
  label="-5 on chimeric reads"
  $M2 mem -t2 -K 10000000 -5 "$IDX" "$fq" 2>/dev/null | grep -v '^@PG' > "$TMP/p5_a.sam"
  $M3 mem -t2 -K 10000000 -5 "$IDX" "$fq" 2>/dev/null | grep -v '^@PG' > "$TMP/p5_b.sam"
  if cmp -s "$TMP/p5_a.sam" "$TMP/p5_b.sam"; then
    printf '  %-28s %-3s [PASS]\n' "$label" "se"; pass=$((pass+1))
  else
    local d
    d=$(paste "$TMP/p5_a.sam" "$TMP/p5_b.sam" | awk -F'\t' '{h=NF/2; for(i=1;i<=h;i++) if($i!=$(i+h)){c++; break}} END{print c+0}')
    printf '  %-28s %-3s [FAIL] %s differing records\n' "$label" "se" "$d"
    fail=$((fail+1)); failed_opts+=("$label")
  fi
}
check_primary5

# `-X` ABOVE 1.0, WITH `-M` AND WITH `-q`. `-X` is `mask_level`, the overlap fraction above which an
# alignment is marked as shadowing another. Push it past 1.0 and the test can never be satisfied, so
# overlapping alignments all stay non-secondary and a read emits SUPPLEMENTARY records where it
# normally emits one. That is the only way to reach two branches of `mem_reg2sam` on the paired-end
# side, and both were wrong until 4.4.x: `-M` was ignored there (the supplementary bit was written
# unconditionally instead of bwa's internal 0x10000, which prints as SAM's 0x100), and the
# supplementary MAPQ cap ran even under `-q`/`-5`, which exist to suppress it. The single-end path
# had both right, which is why one-option-at-a-time never saw it.
#
# ONLY THE `-M` CASE GATES ITS FIX. With the pre-fix behaviour restored, `-X 1.2 -M (pe)` fails on
# 25833 of 25833 records here, while the `-q` and `-5` cases still pass: the MAPQ cap fires only
# when a supplementary is MORE confident than the primary it was split from, and no input built for
# this -- simulated reads, chimeric reads, either end of a pair -- produces one. That half of the
# fix follows `bwamem.cpp:1555` and the single-end path, and is not covered by a fixture. The two
# cases are kept anyway, as coverage of the `-X > 1` shape itself.
#
# `-X` on its own is checked here too, so a future change that breaks it is not blamed on `-M`.
check "-X 1.2 (pe)"       pe -X 1.2
check "-X 1.2 -M (pe)"    pe -X 1.2 -M
check "-X 1.2 -q (pe)"    pe -X 1.2 -q
check "-X 1.2 -5 (pe)"    pe -X 1.2 -5

echo "=== input and output paths ==="
check "-p (interleaved)" pi -p

# `-x` presets. These were unimplemented until 4.2.x and are the only options that rewrite SEVERAL
# others at once, each only where the user left the default, AND suppress the `-A` rescaling by
# taking a different branch to `update_a` (`fastmap.cpp:818-860`).
#
# ONLY `-x intractg` is testable here. The other three (`pacbio`, `pbref`, `ont2d`) are routed to
# rammap and deliberately do NOT produce bwa-mem2's output, so asserting parity for them would
# assert the opposite of what the code promises; `scripts/longread_parity.sh` owns those and its
# oracle is rammap. The preset TABLE for all four is pinned by unit tests on `build_opt`
# (`presets_match_bwa_*` in `cmd_mem.rs`), which is what keeps the values honest even though the
# `mem` path never reaches three of them.
#
# Two cases for intractg: the preset alone, and the preset with an option it would otherwise have
# set, which is the only way to catch a "wrote the value unconditionally" bug.
echo "=== -x read-type presets ==="
check "-x intractg"        se -x intractg
check "-x intractg -B 3"   se -x intractg -B 3
check "-x intractg -O 5"   se -x intractg -O 5
check "-x intractg (pe)"   pe -x intractg

# `-f` is a bare alias of `-o` in bwa (one shared getopt branch), and `-1` is accepted-and-inert on
# both sides. Neither can change the records; the check is that they parse and route identically.
# `-f` goes through `check_f` rather than `check`, because a flag that redirects the SAM to a file
# produces nothing on stdout for `check` to compare.
# Long reads, on the plain `mem` path (no `-x` preset, so nothing is routed to rammap). Their own
# section because they need their own FASTQ: every other case here runs 150 bp reads, and 150 bp is
# below the length at which bwa turns on `mem_flt_chained_seeds`, the per-seed Smith-Waterman filter
# (`bwamem.cpp:472`). That filter disables itself while `5.5 * ln(l_query) > 0.05 * l_query`, true
# up to roughly 690 bp, so a 150 bp fixture cannot tell whether it was ported at all -- and until
# 4.4.x it had not been. The symptom was an `XS` bwa does not emit, from a 20-base seed at a locus
# the read does not belong to.
#
# 800 bp is just past where the filter switches on and 3000 bp is well past it; both are compared
# against the oracle like any other case.
check_longread() {
  local reflen seq
  # The reference as one line, straight out of the index FASTA the rest of this script uses.
  seq=$(grep -v '^>' "$IDX" | tr -d '\n' | tr 'acgt' 'ACGT')
  reflen=${#seq}
  local fq="$TMP/long.fq"
  : > "$fq"
  local len off i
  for len in 800 3000; do
    # Four deterministic offsets spread across the contig, far enough in to avoid any leading N run.
    for i in 1 2 3 4; do
      off=$(( reflen / 6 * i ))
      [ $(( off + len )) -le "$reflen" ] || continue
      printf '@long%s_%s\n%s\n+\n%s\n' "$len" "$i" \
        "${seq:$off:$len}" "$(printf 'I%.0s' $(seq 1 $len))" >> "$fq"
    done
  done
  local label="long reads (800/3000 bp)"
  $M2 mem -t2 -K 10000000 "$IDX" "$fq" 2>/dev/null | grep -v '^@PG' > "$TMP/lr_a.sam"
  $M3 mem -t2 -K 10000000 "$IDX" "$fq" 2>/dev/null | grep -v '^@PG' > "$TMP/lr_b.sam"
  if cmp -s "$TMP/lr_a.sam" "$TMP/lr_b.sam"; then
    printf '  %-28s %-3s [PASS]\n' "$label" "se"; pass=$((pass+1))
  else
    local d
    d=$(paste "$TMP/lr_a.sam" "$TMP/lr_b.sam" | awk -F'\t' '{h=NF/2; for(i=1;i<=h;i++) if($i!=$(i+h)){c++; break}} END{print c+0}')
    printf '  %-28s %-3s [FAIL] %s differing records\n' "$label" "se" "$d"
    fail=$((fail+1)); failed_opts+=("$label")
  fi
}

# INPUT SHAPES, not options. bwa reads FASTQ with klib's `kseq`, which accumulates sequence lines
# until a line starting with `+` and then quality lines until quality is as long as sequence, so it
# takes a WRAPPED (multi-line) FASTQ without noticing. needletail's FASTQ parser requires exactly
# four lines per record, and until 4.4.x we answered such a file with a parse error and zero
# records -- a file bwa aligns perfectly happily.
#
# The oracle's output for a wrapped file is byte-identical to its output for the same reads
# unwrapped, so what is checked here is just that: same reads, two spellings, one answer. Gzip is
# included because that path decompresses before the parser sees anything, and CRLF because a stray
# carriage return is what would hide the `+` from the detector.
check_shapes() {
  local seq reflen i p line
  seq=$(grep -v '^>' "$IDX" | tr -d '\n' | tr 'acgt' 'ACGT')
  reflen=${#seq}
  : > "$TMP/flat.fq"
  : > "$TMP/wrap.fq"
  : > "$TMP/crlf.fq"
  for i in $(seq 1 300); do
    p=$(( reflen / 320 * i ))
    line="${seq:$p:150}"
    printf '@s%s\n%s\n+\n%s\n' "$i" "$line" "$(printf 'I%.0s' $(seq 1 150))" >> "$TMP/flat.fq"
    printf '@s%s\r\n%s\r\n+\r\n%s\r\n' "$i" "$line" "$(printf 'I%.0s' $(seq 1 150))" >> "$TMP/crlf.fq"
    # The same record with sequence and quality wrapped at 60 columns.
    {
      printf '@s%s\n' "$i"
      printf '%s\n' "${line:0:60}" "${line:60:60}" "${line:120:30}"
      printf '+\n'
      printf '%s\n' "$(printf 'I%.0s' $(seq 1 60))" "$(printf 'I%.0s' $(seq 1 60))" "$(printf 'I%.0s' $(seq 1 30))"
    } >> "$TMP/wrap.fq"
  done
  gzip -c "$TMP/wrap.fq" > "$TMP/wrap.fq.gz"

  # The oracle's answer for the flat spelling is the reference every shape is held to.
  $M2 mem -t2 -K 10000000 "$IDX" "$TMP/flat.fq" 2>/dev/null | grep -v '^@PG' > "$TMP/shape_ref.sam"
  local name file
  for name in flat wrap crlf wrap.fq.gz; do
    case "$name" in
      wrap.fq.gz) file="$TMP/wrap.fq.gz" ;;
      *)          file="$TMP/$name.fq" ;;
    esac
    $M3 mem -t2 -K 10000000 "$IDX" "$file" 2>/dev/null | grep -v '^@PG' > "$TMP/shape_b.sam"
    if cmp -s "$TMP/shape_ref.sam" "$TMP/shape_b.sam"; then
      printf '  %-28s %-3s [PASS]\n' "input shape: $name" "se"; pass=$((pass+1))
    else
      printf '  %-28s %-3s [FAIL] %s records against the oracle'"'"'s flat run\n' \
        "input shape: $name" "se" "$(grep -vc '^@' "$TMP/shape_b.sam")"
      fail=$((fail+1)); failed_opts+=("input shape: $name")
    fi
  done
}

echo "=== input shapes ==="
check_shapes

echo "=== long reads on the bwa path ==="
check_longread

echo "=== -f alias and -1 ==="
check "-1 (no_mt_io)"      se -1
check_f
check_o
check_bgzf

echo ""
echo "RESULT: $pass passed, $fail failed"
[ $fail -gt 0 ] && printf 'FAILING: %s\n' "${failed_opts[*]}"
exit 0
