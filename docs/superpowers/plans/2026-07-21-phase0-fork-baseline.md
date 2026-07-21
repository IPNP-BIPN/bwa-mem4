# Phase 0: Fork Baseline Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a trustworthy three-arm measurement (bwa-mem2 2.3 / `fg-labs/bwa-mem3` / bwa-mem4) of wall time and peak RSS at `-t8` end-to-end on real WGS, so the rest of the campaign is aimed at a real deficit rather than a phase-9a-era one.

**Architecture:** One new Rust-visible behaviour (bwa-mem4 prints how many batches it processed), then two shell harnesses: `scripts/fork_bench.sh` at iteration scale (1-4M reads, minutes) and a third arm added to the existing `scripts/giab30x_bench.sh` at validation scale (30x, hours). The harnesses interleave arms within a pass, warm every binary first, and record RSS alongside wall time.

**Tech Stack:** Rust 1.96 (workspace `bwa-mem4`), bash, BSD `/usr/bin/time -l`, `md5`.

## Global Constraints

- Byte-identity to bwa-mem2 2.3 is non-negotiable. No task here may change alignment output; the only Rust change is a stderr line, and stderr is outside every parity gate.
- Genome index only (`work/genome.fa`). `work/region.fa` is inadmissible: its 2 Mbp BWT is cache-resident and hides seeding, which is ~78% of the real profile.
- Our arm must be the output of `scripts/pgo.sh`, never a plain `cargo build --release` (~15% slower, and not what ships).
- Arms interleaved within one pass. Host noise is ~2.4%; nothing under 3% is a gain.
- Every binary and every mapping warmed before the first timed run.
- Fork binary: `reference/bwa-mem3-cpp/bwa-mem3.arm64`. It reads the same index files (it is a bwa-mem2 derivative) and prints `Executing in neon mode!!` to stderr.

---

### Task 1: bwa-mem4 reports its batch count

The `-K` trap: `-K 100000000` on 500k x 150bp reads is 0.8 batches, which makes the reader/writer pipeline structurally inert and silently costs our arm 8-9%. This project has fallen into it three times. A benchmark that cannot see its batch count is measuring a configuration with half the work disabled.

**Files:**
- Modify: `crates/bwa-cli/src/cmd_mem.rs:1328` (add `verbose` param to `run_pipeline`), `:1362-1368` (count), `:1594-1603` (`run_pe` call site), `:1685` (`run_pipeline` SE call site), `:2165-2177` (`run_pe` signature), `:2397` (`run_pipeline` PE call site)
- Create: `crates/bwa-cli/tests/batch_count.rs`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a stderr line of the exact form `[M::main_mem] processed N batches (-K K)`, which Tasks 3 and 4 grep for. `N` and `K` are decimal integers with no separators.

- [ ] **Step 1: Write the failing test**

Create `crates/bwa-cli/tests/batch_count.rs`:

```rust
//! The batch count is a measurement instrument, not a nicety: `-K` too large collapses the
//! reader/writer pipeline to a single batch and removes 8-9% of our throughput with no other
//! symptom. Every benchmark in this project must be able to read the batch count back, so it is
//! asserted here rather than left to a manual eyeball.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Repo root, derived from this crate's manifest dir (`crates/bwa-cli`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Write `n` reads of length `len` taken from consecutive offsets of `tiny.fa`'s sequence, as a
/// minimal FASTQ with constant quality. Deterministic, so the batch count is reproducible.
fn write_reads(path: &Path, n: usize, len: usize) {
    let fa = std::fs::read_to_string(repo_root().join("testdata/tiny/tiny.fa")).unwrap();
    let seq: String = fa.lines().skip(1).collect();
    let mut out = String::new();
    for i in 0..n {
        let start = (i * 37) % (seq.len() - len);
        out.push_str(&format!(
            "@r{}\n{}\n+\n{}\n",
            i,
            &seq[start..start + len],
            "I".repeat(len)
        ));
    }
    std::fs::write(path, out).unwrap();
}

/// `-K` is a base count, so 100 reads x 100 bp (10,000 bases) with `-K 2000` must produce several
/// batches rather than one. The exact number is deliberately not pinned: where a batch boundary
/// falls is bwa's accumulate-until-`-K` rule, which we match rather than define, and pinning it
/// here would turn a faithful port into a test failure. What must hold is that the count is
/// reported, that it echoes `-K`, and that it is greater than one.
#[test]
fn reports_batch_count_on_stderr() {
    let dir = std::env::temp_dir().join("bwa4_batch_count_test");
    std::fs::create_dir_all(&dir).unwrap();
    let fq = dir.join("r.fq");
    write_reads(&fq, 100, 100);

    let out = Command::new(env!("CARGO_BIN_EXE_bwa-mem4"))
        .arg("mem")
        .arg("-t1")
        .arg("-K")
        .arg("2000")
        .arg(repo_root().join("testdata/tiny/tiny.fa"))
        .arg(&fq)
        .output()
        .unwrap();

    let err = String::from_utf8_lossy(&out.stderr);
    let line = err
        .lines()
        .find(|l| l.starts_with("[M::main_mem] processed "))
        .unwrap_or_else(|| panic!("stderr did not report the batch count:\n{err}"));
    assert!(line.ends_with(" batches (-K 2000)"), "malformed line: {line}");
    let n: usize = line
        .trim_start_matches("[M::main_mem] processed ")
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(n > 1, "10,000 bases at -K 2000 should be several batches, got {n}");
}

/// `-v 2` quietens bwa's own progress lines, and must quieten this one too, or a script that
/// asked for silence gets a surprise line in the middle of its output.
#[test]
fn batch_count_respects_verbosity() {
    let dir = std::env::temp_dir().join("bwa4_batch_count_quiet_test");
    std::fs::create_dir_all(&dir).unwrap();
    let fq = dir.join("r.fq");
    write_reads(&fq, 100, 100);

    let out = Command::new(env!("CARGO_BIN_EXE_bwa-mem4"))
        .arg("mem")
        .arg("-t1")
        .arg("-v")
        .arg("2")
        .arg("-K")
        .arg("2000")
        .arg(repo_root().join("testdata/tiny/tiny.fa"))
        .arg(&fq)
        .output()
        .unwrap();

    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!err.contains("processed"), "batch count leaked at -v 2:\n{err}");
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p bwa-mem4 --test batch_count`
Expected: `reports_batch_count_on_stderr` FAILS with `stderr did not report the batch count` (the line does not exist yet). `batch_count_respects_verbosity` PASSES vacuously, since a line that is never printed is also never printed at `-v 2`; it earns its keep only after Step 3.

- [ ] **Step 3: Add the counter to `run_pipeline`**

In `crates/bwa-cli/src/cmd_mem.rs`, change the signature at line 1328 to take the verbosity:

```rust
fn run_pipeline<B: Send>(
    out: Output,
    read_batches: impl FnOnce(std::sync::mpsc::SyncSender<(B, u64)>) -> anyhow::Result<()> + Send,
    process: impl Fn(B, u64) -> Vec<u8>,
    // `-v`, so the count obeys the same quiet switch as bwa's own progress lines. It is a
    // measurement instrument (see `crates/bwa-cli/tests/batch_count.rs`), so it is on by default.
    verbose: i32,
    // `-K` in bases, echoed in the line so a log records the setting that produced the count.
    k_batch: usize,
) -> anyhow::Result<()> {
```

Replace the consume loop at lines 1362-1368 with:

```rust
        let mut n_batches = 0usize;
        for (batch, base_id) in batch_rx {
            n_batches += 1;
            // This batch's complete SAM text, all records concatenated in read order.
            let buf = process(batch, base_id);
            if sam_tx.send(buf).is_err() {
                break; // writer exited; its error surfaces on join below
            }
        }
        drop(sam_tx);
        // The pipeline overlaps batch N+1's read and N-1's write with N's compute, so at one batch
        // it is structurally inert and this run is not measuring the shipped configuration.
        if verbose >= 3 {
            eprintln!("[M::main_mem] processed {n_batches} batches (-K {k_batch})");
        }
```

- [ ] **Step 4: Update the three call sites**

Line 1685 (single-end, inside `pub fn run`, where `args` and `k_batch` are both in scope):

```rust
    run_pipeline(
        out,
        read_batches,
        process,
        args.verbose.unwrap_or(3),
        k_batch,
    )?;
```

`run_pe` has no verbosity of its own, so thread it through. Add to the signature at line 2165, after `pes0`:

```rust
    pes0: Option<[PeStat; 4]>,
    // `-v`, forwarded to `run_pipeline` for the batch-count line. `run_pe` has no `MemArgs`.
    verbose: i32,
) -> anyhow::Result<()> {
```

Line 2397 (paired-end):

```rust
    run_pipeline(out, read_batches, process, verbose, k_batch)
}
```

And the `run_pe` call at line 1594:

```rust
        run_pe(
            &fm,
            &bns,
            &opt,
            &args.reads,
            reads2.as_deref(),
            k_batch,
            out,
            pes0,
            args.verbose.unwrap_or(3),
        )?;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p bwa-mem4 --test batch_count`
Expected: PASS, 2 passed.

- [ ] **Step 6: Verify no output changed**

Run: `cargo test --workspace --quiet 2>&1 | grep -E "^test result" | grep -v " 0 failed" ; echo "exit=$?"`
Expected: no lines printed before `exit=1` (grep finds nothing, meaning no suite had failures).

Run: `scripts/oracle_diff.sh` (SE 5000/5000 and PE 10000/10000).
Expected: 100% on both. This is belt-and-braces: the change only writes to stderr, and every parity gate compares `@SQ` plus alignment records on stdout.

- [ ] **Step 7: Commit**

```bash
git add crates/bwa-cli/src/cmd_mem.rs crates/bwa-cli/tests/batch_count.rs
git commit -m "feat(mem): report the batch count, so a benchmark can see the -K trap

-K 100000000 on 500k x 150bp reads is 0.8 batches. The reader/writer pipeline
overlaps batch N+1's read and N-1's write with N's compute, so at a single batch
it is structurally inert and the run silently loses the 8-9% that pipeline is
worth. Three benchmarks in this project have been published against that
configuration without anyone noticing, because nothing in the output says how
many batches ran.

stderr only, gated on -v like bwa's own progress lines, so no parity gate sees
it."
```

---

### Task 2: `scripts/bench.sh` stops defaulting to the region index

`bench.sh` hardcodes `IDX="work/region.fa"`. Every number it has ever produced inflates extension and hides seeding, which is the trap that put "SW extension is ~85% of cycles" into this project's docs for months. It already measures wall time and peak RSS correctly, so it is worth keeping, but not with that default.

**Files:**
- Modify: `scripts/bench.sh:15` (the `IDX` line) and the usage comment at `:4-6`

**Interfaces:**
- Consumes: nothing.
- Produces: `IDX` as an environment variable, defaulting to `work/genome.fa`; the script refuses `work/region.fa` unless `ALLOW_REGION=1`.

- [ ] **Step 1: Write the failing test**

There is no test framework for these scripts, so the test is the script's own guard, exercised directly. Run this first to see it fail:

```bash
IDX=work/region.fa scripts/bench.sh target/release/bwa-mem4 se 1 2>&1 | head -3
```

Expected before the change: the script runs the benchmark on the region index (wrong).
Expected after: it refuses with the message added in Step 2.

- [ ] **Step 2: Change the default and add the guard**

In `scripts/bench.sh`, replace line 15 (`IDX="work/region.fa"`) with:

```bash
# The genome index, NOT work/region.fa: region's 2 Mbp BWT is cache-resident, so seeding (~78% of
# the real profile) nearly disappears and extension's share inflates 4-20x. Docs in this repo
# carried a "SW extension is ~85% of cycles" figure for months because of a region-measured
# profile. Override deliberately with ALLOW_REGION=1 if you really want the small index.
IDX="${IDX:-work/genome.fa}"
if [ "$IDX" = "work/region.fa" ] && [ "${ALLOW_REGION:-0}" != "1" ]; then
  echo "refusing to benchmark on work/region.fa (cache-resident BWT hides seeding)." >&2
  echo "set ALLOW_REGION=1 if that is genuinely what you want." >&2
  exit 1
fi
```

Update the usage comment at lines 4-6 to say the index is `work/genome.fa` by default and settable with `IDX=`.

- [ ] **Step 3: Verify the guard fires and the default is the genome**

Run: `IDX=work/region.fa scripts/bench.sh target/release/bwa-mem4 se 1 2>&1 | head -2`
Expected: the two refusal lines, exit status 1.

Run: `ALLOW_REGION=1 IDX=work/region.fa scripts/bench.sh target/release/bwa-mem4 se 1 2>&1 | tail -1`
Expected: a normal `se median_wall_s=... peak_rss_mb=...` line.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench.sh
git commit -m "fix(bench): stop defaulting the perf harness to the region index

work/region.fa has a 2 Mbp BWT that stays in cache, so seeding looks nearly free
and extension's share inflates 4-20x. That is the measurement that put 'SW
extension is ~85% of cycles' into this repo's docs and aimed years of kernel
work at 4% of the real runtime.

Default is now work/genome.fa; the region index needs ALLOW_REGION=1, so
choosing it is an act rather than an accident."
```

---

### Task 3: `scripts/fork_bench.sh`, the three-arm iteration harness

Iteration scale: 1-4M reads, minutes per pass, ratios depth-invariant. This is the script the rest of the campaign runs after every change.

**Files:**
- Create: `scripts/fork_bench.sh`

**Interfaces:**
- Consumes: the `[M::main_mem] processed N batches (-K K)` line from Task 1.
- Produces: a results table on stdout and in `work/forkbench/<timestamp>/results.log`, with one row per arm: wall seconds (median), peak RSS MB, batch count, and the md5 of the alignment records.

- [ ] **Step 1: Write the script**

Create `scripts/fork_bench.sh`:

```bash
#!/usr/bin/env bash
# Three-arm head-to-head at iteration scale: bwa-mem2 2.3 (oracle), fg-labs/bwa-mem3 (@nh13's C++
# fork), and us. Wall time, peak RSS and batch count per arm, plus the md5 of the alignment records
# so identity is checked on every pass at no extra cost.
#
# Usage: scripts/fork_bench.sh [se|pe] [reps]
#   T=8 K=10000000 READS=work/r1_4m.fq scripts/fork_bench.sh se 3
#
# Method rules, each one paid for by a past error in this repo:
#   - arms are interleaved WITHIN a rep, never run as three separate blocks. Repeated identical
#     runs spread ~2.4%, and numbers taken minutes apart are worthless.
#   - every binary is warmed once before the first timed rep, on the same index and reads. Cold
#     starts have produced a "13.89x prefetch speedup" here whose real value was 1.02x.
#   - the genome index only. region.fa's BWT is cache-resident and hides seeding.
#   - our arm should be the PGO binary (scripts/pgo.sh); cargo build --release is ~15% slower and
#     is not what we ship, so timing it would flatter the fork by 15%.
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-se}"
REPS="${2:-3}"
T="${T:-8}"
K="${K:-10000000}"
IDX="${IDX:-work/genome.fa}"
M2="${M2:-bwa-mem2}"
FORK="${FORK:-reference/bwa-mem3-cpp/bwa-mem3.arm64}"
M4="${M4:-./target/release/bwa-mem4}"

case "$MODE" in
  se) READS=("${READS:-work/r1_4m.fq}") ;;
  pe) READS=("${READS:-work/r1_4m.fq}" "${READS2:-work/r2_4m.fq}") ;;
  *) echo "mode must be se|pe" >&2; exit 1 ;;
esac

for f in "${READS[@]}" "$IDX.bwt.2bit.64" "$FORK" "$M4"; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
done
command -v "$M2" >/dev/null || { echo "missing $M2 on PATH" >&2; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUT="work/forkbench/$TS"; mkdir -p "$OUT"
LOG="$OUT/results.log"
say() { echo "$@" | tee -a "$LOG"; }

# One timed run. $1=arm label, $2=rep, rest=command.
# /usr/bin/time -l gives both wall seconds and peak RSS in bytes on macOS.
# The record stream goes through md5 so identity is verified on the same pass that is timed.
run() {
  local arm="$1" rep="$2"; shift 2
  local of="$OUT/${arm}_${MODE}_${rep}"
  /usr/bin/time -l bash -c "$* 2>'$of.err' | grep -v '^@' | tee >(wc -l >'$of.n') | md5 >'$of.md5'" 2>"$of.time"
  local real rssb
  real=$(awk '/real/{print $1}' "$of.time" | head -1)
  rssb=$(awk '/maximum resident set size/{print $1}' "$of.time")
  # Our arm reports its batch count; the other two cannot, so they show "-".
  local nb; nb=$(sed -n 's/.*processed \([0-9]*\) batches.*/\1/p' "$of.err" | tail -1)
  [ -n "$nb" ] || nb="-"
  echo "$real $(( rssb / 1048576 )) $nb"
}

say "############################################################"
say "# fork_bench  $TS   mode=$MODE  -t$T  -K $K  reps=$REPS"
say "# idx=$IDX  reads=${READS[*]}"
say "# bwa-mem4=$(git rev-parse --short HEAD)"
say "############################################################"

CMD_M2="$M2 mem -t$T -K $K $IDX ${READS[*]}"
CMD_FORK="$FORK mem -t$T -K $K $IDX ${READS[*]}"
CMD_M4="$M4 mem -t$T -K $K $IDX ${READS[*]}"

# ---- Warm every binary and the index mapping once, untimed ----
say "warming (3 untimed passes)..."
for c in "$CMD_M2" "$CMD_FORK" "$CMD_M4"; do
  bash -c "$c" >/dev/null 2>&1
done

# ---- Interleaved timed reps ----
declare -a w_m2 w_fk w_m4 r_m2 r_fk r_m4
for i in $(seq 1 "$REPS"); do
  read -r a_w a_r a_b <<<"$(run m2   "$i" "$CMD_M2")"
  read -r f_w f_r f_b <<<"$(run fork "$i" "$CMD_FORK")"
  read -r o_w o_r o_b <<<"$(run m4   "$i" "$CMD_M4")"
  w_m2+=("$a_w"); r_m2+=("$a_r")
  w_fk+=("$f_w"); r_fk+=("$f_r")
  w_m4+=("$o_w"); r_m4+=("$o_r")
  say "  rep$i  mem2=${a_w}s/${a_r}MB  fork=${f_w}s/${f_r}MB  mem4=${o_w}s/${o_r}MB (batches=$o_b)"
done

med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}'; }
MW2=$(med "${w_m2[@]}"); MWF=$(med "${w_fk[@]}"); MW4=$(med "${w_m4[@]}")
MR2=$(med "${r_m2[@]}"); MRF=$(med "${r_fk[@]}"); MR4=$(med "${r_m4[@]}")

say ""
say "| arm | wall s (median) | peak RSS MB | vs bwa-mem2 |"
say "|---|---|---|---|"
say "| bwa-mem2 2.3 | $MW2 | $MR2 | 1.00x |"
say "| fg-labs/bwa-mem3 | $MWF | $MRF | $(echo "scale=2; $MW2/$MWF" | bc)x |"
say "| bwa-mem4 | $MW4 | $MR4 | $(echo "scale=2; $MW2/$MW4" | bc)x |"
say ""
say "us vs fork: $(echo "scale=3; $MWF/$MW4" | bc)x wall, $(echo "scale=3; $MRF/$MR4" | bc)x RSS  (>1 means we win)"

# ---- Identity, from rep 1 ----
if diff -q "$OUT/m2_${MODE}_1.md5" "$OUT/m4_${MODE}_1.md5" >/dev/null \
   && diff -q "$OUT/m2_${MODE}_1.n" "$OUT/m4_${MODE}_1.n" >/dev/null; then
  say "[PASS] bwa-mem4 BYTE-IDENTICAL to bwa-mem2: $(cat "$OUT/m2_${MODE}_1.n") records"
else
  say "[FAIL] bwa-mem4 differs: mem2 $(cat "$OUT/m2_${MODE}_1.n")rec vs mem4 $(cat "$OUT/m4_${MODE}_1.n")rec"
fi
if diff -q "$OUT/m2_${MODE}_1.md5" "$OUT/fork_${MODE}_1.md5" >/dev/null; then
  say "[note] the fork is also byte-identical to bwa-mem2 on this input"
else
  say "[note] the fork DIFFERS from bwa-mem2 on this input ($(cat "$OUT/fork_${MODE}_1.n") records)"
fi

# ---- The -K trap ----
NB=$(sed -n 's/.*processed \([0-9]*\) batches.*/\1/p' "$OUT/m4_${MODE}_1.err" | tail -1)
if [ -n "$NB" ] && [ "$NB" -lt 4 ] 2>/dev/null; then
  say "WARNING: only $NB batches. The reader/writer pipeline is inert below ~4 batches and this"
  say "         run understates bwa-mem4 by 8-9%. Lower -K or use more reads."
fi
say "FORK_BENCH_DONE  $LOG"
```

- [ ] **Step 2: Make it executable and smoke-test the plumbing on the small index**

```bash
chmod +x scripts/fork_bench.sh
ALLOW_REGION=1 IDX=work/region.fa READS=work/r1_500k.fq K=5000000 T=4 scripts/fork_bench.sh se 1
```

Expected: the table prints with three populated rows, a `batches=` value greater than 1, and both identity lines. This is a plumbing check only — the region index makes the numbers meaningless, which is why the real runs in Task 5 use the genome.

- [ ] **Step 3: Verify the batch-count warning fires**

```bash
ALLOW_REGION=1 IDX=work/region.fa READS=work/r1_500k.fq K=100000000 T=4 scripts/fork_bench.sh se 1 | tail -4
```

Expected: the `WARNING: only 1 batches` block appears.

- [ ] **Step 4: Commit**

```bash
git add scripts/fork_bench.sh
git commit -m "bench: three-arm harness against the nh13 fork, with RSS and batch count

The only number we have against fg-labs/bwa-mem3 is phase-9a-era, -t1,
align-only, on a 2 Mbp region whose BWT sits in cache. That regime excludes the
two things a bwa-mem2 derivative cannot do (mmap instead of a 16 GB memcpy at
load, and a reader/writer pipeline), so it cannot answer whether we are behind
where a user actually runs.

Interleaves the three arms within each rep rather than running three blocks,
warms every binary first, reports peak RSS next to wall time because RAM is an
objective here too, and warns when the batch count is low enough that the
pipeline is inert."
```

---

### Task 4: the fork joins the 30x validation harness

`scripts/giab30x_bench.sh` already runs bwa-mem2 against us at 30x with per-pass md5 identity and a fast-return crash guard. It gains a third arm and RSS, and it does not otherwise change: it is the script that produced the numbers in the README.

**Files:**
- Modify: `scripts/giab30x_bench.sh:9-11` (binary list), `:29-42` (the `run` helper), `:56-62` and `:73-79` (the SE and PE rep loops)

**Interfaces:**
- Consumes: the batch-count line from Task 1.
- Produces: the same log format as before plus a `fork=` column and a `rss=` figure per arm.

- [ ] **Step 1: Add the fork binary**

After line 11 (`M3=./target/release/bwa-mem4`), add:

```bash
FORK="${FORK:-reference/bwa-mem3-cpp/bwa-mem3.arm64}"
```

and add `"$FORK"` to the existence check on line 19.

- [ ] **Step 2: Switch the `run` helper from `time -p` to `time -l` so it also yields RSS**

Replace the `/usr/bin/time -p` line in `run` with:

```bash
  /usr/bin/time -l bash -c "$cmd 2>'$of.err' | grep -v '^@' | tee >(wc -l >'$of.n') | md5 >'$of.md5'" 2>"$of.time"
```

and replace the `real` extraction with both figures:

```bash
  local real; real=$(awk '/real/{print $1}' "$of.time" | head -1)
  awk '/maximum resident set size/{printf "%d\n", $1/1048576}' "$of.time" > "$of.rss"
```

Keep the existing `< 30` crash guard exactly as it is: a real 30x pass is far longer, so a fast return still means the aligner died.

- [ ] **Step 3: Add the fork to both rep loops**

In the SE loop, between the `m2` and `m3` runs, add:

```bash
  f=$(run "$OUT/fk_se_$i" $FORK mem -t"$T" -K $K "$IDX" "$R1") || { say "  [ABORT] SE fork rep$i failed fast (see $OUT/fk_se_$i.err)"; exit 1; }
```

and extend the `say` line to:

```bash
  say "  rep$i  mem2=${a}s/$(cat "$OUT/m2_se_$i.rss")MB  fork=${f}s/$(cat "$OUT/fk_se_$i.rss")MB  mem4=${b}s/$(cat "$OUT/m3_se_$i.rss")MB  vs-mem2=${sp}x  vs-fork=$(echo "scale=3; $f/$b" | bc)x"
```

Make the identical change in the PE loop with `fk_pe_$i`, `$R1 "$R2"`.

- [ ] **Step 4: Verify the script parses and the ordering is interleaved**

Run: `bash -n scripts/giab30x_bench.sh; echo "syntax=$?"`
Expected: `syntax=0`.

Run: `grep -n "run \"\$OUT" scripts/giab30x_bench.sh`
Expected: within each loop the three `run` calls appear consecutively (m2, fk, m3), so the arms are interleaved within a rep rather than run as three blocks.

- [ ] **Step 5: Commit**

```bash
git add scripts/giab30x_bench.sh
git commit -m "bench(30x): add the fork as a third arm, and record peak RSS

The 30x harness is what produced the README's SE 2.62x / PE 1.85x, so it is
where the fork comparison has to end up to be quotable. Arms stay interleaved
within a rep. time -p becomes time -l for the RSS, which is an objective of this
campaign and not a footnote; the fast-return crash guard is untouched."
```

---

### Task 5: run Phase 0 and write down what it says

The deliverable of Phase 0 is a table, not a speedup. All three outcomes are admissible and were accepted in advance: we already win (Phase B is cancelled), we lose (Phase B is justified and aimed by the profile), or we win on time and lose on RSS (the effort moves to memory).

**Files:**
- Modify: `ROADMAP.md` (a new `## Phase 0 de la campagne` section, in French like the rest of that document)
- Create: `work/forkbench/<timestamp>/results.log` (gitignored output, referenced by path in the ROADMAP entry)

**Interfaces:**
- Consumes: `scripts/fork_bench.sh` from Task 3.
- Produces: the measured table that gates Phase B.

- [ ] **Step 1: Build the arm we actually ship**

Run: `scripts/pgo.sh`
Expected: it finishes and leaves an instrumented-then-optimised `target/release/bwa-mem4`. A plain `cargo build --release` here would understate us by ~15%.

- [ ] **Step 2: Prepare a 4M-read iteration input if it does not exist**

```bash
[ -f work/r1_4m.fq ] || gunzip -c work/giab30x/HG002_30x_R1.fastq.gz | head -16000000 > work/r1_4m.fq
[ -f work/r2_4m.fq ] || gunzip -c work/giab30x/HG002_30x_R2.fastq.gz | head -16000000 > work/r2_4m.fq
wc -l work/r1_4m.fq
```

Expected: `16000000 work/r1_4m.fq` (4M reads at 4 lines each).

- [ ] **Step 3: Run the iteration-scale benchmark, SE and PE**

```bash
T=8 K=10000000 scripts/fork_bench.sh se 3
T=8 K=10000000 scripts/fork_bench.sh pe 3
```

Expected: both tables print, `[PASS] bwa-mem4 BYTE-IDENTICAL to bwa-mem2` on both, and no low-batch warning (4M reads at `-K 10000000` is ~60 batches).

- [ ] **Step 4: Record the result in ROADMAP.md**

Add a section after `## Statut (4.0.0)`, filling in the measured numbers (do not invent them; copy from the log):

```markdown
## Phase 0 de la campagne : mesure de reference contre le fork

Trois bras entrelaces dans la meme passe (`scripts/fork_bench.sh`), index genome, binaire PGO,
4M reads GIAB HG002, `-t8`, `-K` 10M (~60 batches), mediane de 3.

| bras | temps (s) | pic RSS (Mo) | vs bwa-mem2 |
|---|---|---|---|
| `bwa-mem2` 2.3 | ... | ... | 1,00x |
| `fg-labs/bwa-mem3` | ... | ... | ...x |
| **bwa-mem4** | ... | ... | **...x** |

Nous contre le fork : ...x en temps, ...x en RSS.

Le chiffre de la phase 9a (1,29x derriere, `-t1` align-only, region 2 Mbp) n'etait pas
comparable : ce regime exclut le mmap de l'index et le pipeline lecteur/ecrivain, et son BWT
tient en cache. Log complet : `work/forkbench/<TS>/results.log`.
```

- [ ] **Step 5: Commit**

```bash
git add ROADMAP.md
git commit -m "measure: the fork baseline, -t8 end-to-end, RSS included

Replaces the phase-9a number (1.29x behind, -t1, align-only, 2 Mbp region) that
this project has been quoting for months. That regime hid the two things that
decide the comparison at -t8: the index load, where we mmap and a bwa-mem2
derivative memcpys 16 GB, and the reader/writer pipeline.

Whatever this table says is what the rest of the campaign is aimed at."
```

- [ ] **Step 6: Decide Phase B, in writing**

Read the `us vs fork` line. If wall-clock ratio >= 1.03 (we win by more than host noise), append to the ROADMAP section: `Phase B (diff du seeding du fork) annulee : nous sommes devant dans le regime cible.` If it is <= 0.97, Phase B proceeds and its target is set by re-profiling at genome scale, not by the old `-t1` figure. Between 0.97 and 1.03 the arms are tied and the RSS column decides where the effort goes.

---

## Notes for whoever runs this

- The 30x validation pass (Task 4's harness) is deliberately not part of Phase 0's iteration loop. Three arms, SE and PE, at 32.9x is roughly a day of machine time; run it once, on the final answer, before anything is published.
- If `bwa-mem2` is not on PATH, `scripts/setup_reference.sh` fetches and patches the oracle.
- The fork prints `Executing in neon mode!!` and its SA-compression banner to stderr on every run. That is why every arm's stderr goes to `$of.err` rather than the terminal.
