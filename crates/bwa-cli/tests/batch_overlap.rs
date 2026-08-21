//! `BWA4_NO_BATCH_OVERLAP` halves the pipeline's batch memory, and the whole case for it rests on
//! one claim: it cannot move a byte of output. That claim is not obvious from the code. The switch
//! changes how many batches are being aligned at once, on different threads, and this project has
//! learned repeatedly that output identity survives a scheduling change only when something holds
//! it there. Here it is the join: a batch's bytes are sent to the writer only after its worker has
//! been joined, and the next batch's handle takes the slot only after that send, so output order is
//! input order at any number of batches in flight. And no batch's result depends on another's,
//! since `-K` fixes the boundaries and the tie-break hash keys on the global read id.
//!
//! So this test runs the same input twice, once per setting, and compares the SAM bytes. It is the
//! cheap version of what `scripts/oracle_diff.sh` does against bwa-mem2: not a parity gate, but a
//! gate on the one property the memory lever promises.
//!
//! Single-end and paired-end are both exercised, because they are different pipelines: the paired
//! path adds mate rescue and the per-batch insert-size model (`mem_pestat`), and the insert model
//! is the part most likely to notice a change in batch scheduling if one ever crept in.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Repo root, derived from this crate's manifest dir (`crates/bwa-cli`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Write `n` reads of `len` bases taken from rotating offsets of `tiny.fa`, as a minimal FASTQ with
/// constant quality. Deterministic, and the same shape `batch_count.rs` uses.
///
/// # Parameters
///
/// - `path`: file to write.
/// - `n`: how many reads.
/// - `len`: read length in bases.
/// - `stride`: offset step between consecutive reads. Two different strides give two files whose
///   reads pair up without being identical, which is what the paired-end arm needs.
fn write_reads(path: &Path, n: usize, len: usize, stride: usize) {
    let fa = std::fs::read_to_string(repo_root().join("testdata/tiny/tiny.fa")).unwrap();
    let seq: String = fa.lines().skip(1).collect();
    let mut out = String::new();
    for i in 0..n {
        let start = (i * stride) % (seq.len() - len);
        out.push_str(&format!(
            "@r{}\n{}\n+\n{}\n",
            i,
            &seq[start..start + len],
            "I".repeat(len)
        ));
    }
    std::fs::write(path, out).unwrap();
}

/// Align `reads` with the overlap either on or off, and return the SAM text.
///
/// `-K 2000` on 10,000 bases is several batches, which is the whole point: at one batch the switch
/// has nothing to change and the test would pass without testing anything. `-t2` so there is a pool
/// to overlap into.
///
/// # Parameters
///
/// - `reads`: one or two FASTQ paths (paired when two).
/// - `no_overlap`: whether to set `BWA4_NO_BATCH_OVERLAP`.
///
/// # Returns
///
/// The complete SAM output, header included.
fn align(reads: &[PathBuf], no_overlap: bool) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_bwa-mem4"));
    cmd.arg("mem")
        .arg("-t2")
        .arg("-K")
        .arg("2000")
        .arg(repo_root().join("testdata/tiny/tiny.fa"))
        .args(reads);
    if no_overlap {
        cmd.env("BWA4_NO_BATCH_OVERLAP", "1");
    } else {
        // Explicitly cleared: the test must not silently pass because the developer running it had
        // the variable exported in their shell, which would make both arms the same arm.
        cmd.env_remove("BWA4_NO_BATCH_OVERLAP");
    }
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "bwa-mem4 failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The two arms must agree byte for byte, `@PG` included: both are the same binary invoked the same
/// way, so even the command line recorded in the header matches.
#[test]
fn single_end_output_is_identical_with_and_without_the_overlap() {
    let dir = std::env::temp_dir().join("bwa4_overlap_se");
    std::fs::create_dir_all(&dir).unwrap();
    let fq = dir.join("r.fq");
    write_reads(&fq, 100, 100, 37);
    let reads = vec![fq];

    let with = align(&reads, false);
    let without = align(&reads, true);
    assert!(
        with.lines().filter(|l| !l.starts_with('@')).count() >= 100,
        "the fixture produced no records, so the comparison would be vacuous"
    );
    assert_eq!(
        with, without,
        "BWA4_NO_BATCH_OVERLAP changed the single-end output"
    );
}

/// Same, on the paired path, which additionally runs mate rescue and rebuilds the insert-size model
/// once per batch.
#[test]
fn paired_end_output_is_identical_with_and_without_the_overlap() {
    let dir = std::env::temp_dir().join("bwa4_overlap_pe");
    std::fs::create_dir_all(&dir).unwrap();
    let (r1, r2) = (dir.join("r1.fq"), dir.join("r2.fq"));
    write_reads(&r1, 100, 100, 37);
    write_reads(&r2, 100, 100, 41);
    let reads = vec![r1, r2];

    let with = align(&reads, false);
    let without = align(&reads, true);
    assert!(
        with.lines().filter(|l| !l.starts_with('@')).count() >= 200,
        "the fixture produced no records, so the comparison would be vacuous"
    );
    assert_eq!(
        with, without,
        "BWA4_NO_BATCH_OVERLAP changed the paired-end output"
    );
}
