//! The batch count is a measurement instrument, not a nicety: `-K` too large collapses the
//! reader/writer pipeline to a single batch and removes 8-9% of our throughput with no other
//! symptom. Three benchmarks in this project have been published against that configuration
//! without anyone noticing, because nothing in the output says how many batches ran. So every
//! benchmark here must be able to read the count back, and that is asserted rather than eyeballed.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Repo root, derived from this crate's manifest dir (`crates/bwa-cli`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Write `n` reads of length `len` taken from rotating offsets of `tiny.fa`'s sequence, as a
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

/// Run `bwa-mem4 mem` on `n` synthetic reads and return its stderr.
fn run_mem(dir_name: &str, extra: &[&str]) -> String {
    let dir = std::env::temp_dir().join(dir_name);
    std::fs::create_dir_all(&dir).unwrap();
    let fq = dir.join("r.fq");
    write_reads(&fq, 100, 100);

    let out = Command::new(env!("CARGO_BIN_EXE_bwa-mem4"))
        .arg("mem")
        .arg("-t1")
        .args(extra)
        .arg("-K")
        .arg("2000")
        .arg(repo_root().join("testdata/tiny/tiny.fa"))
        .arg(&fq)
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// `-K` is a base count, so 100 reads x 100 bp (10,000 bases) at `-K 2000` must produce several
/// batches rather than one. The exact number is deliberately not pinned: where a batch boundary
/// falls is bwa's accumulate-until-`-K` rule, which we match rather than define, and pinning it
/// here would turn a faithful port into a test failure. What must hold is that the count is
/// reported, that it echoes `-K`, and that it is greater than one.
#[test]
fn reports_batch_count_on_stderr() {
    let err = run_mem("bwa4_batch_count_test", &[]);
    let line = err
        .lines()
        .find(|l| l.starts_with("[M::main_mem] processed "))
        .unwrap_or_else(|| panic!("stderr did not report the batch count:\n{err}"));
    assert!(
        line.ends_with(" batches (-K 2000)"),
        "malformed line: {line}"
    );
    let n: usize = line
        .trim_start_matches("[M::main_mem] processed ")
        .split(' ')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        n > 1,
        "10,000 bases at -K 2000 should be several batches, got {n}"
    );
}

/// `-v 2` quietens bwa's own progress lines, and must quieten this one too, or a script that asked
/// for silence gets a surprise line in the middle of its output.
#[test]
fn batch_count_respects_verbosity() {
    let err = run_mem("bwa4_batch_count_quiet_test", &["-v", "2"]);
    assert!(
        !err.contains("processed"),
        "batch count leaked at -v 2:\n{err}"
    );
}
