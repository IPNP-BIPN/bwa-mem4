//! What the binary does when it is handed something wrong, or asked for something it cannot honour.
//! Each of these used to panic, abort, or quietly write the wrong thing, so what matters is the exit
//! code and that the message names the cause.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// `testdata/tiny/tiny.fa`, which is also the prefix of the committed index.
fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny/tiny.fa")
}

/// A fresh directory holding one read taken from the reference.
fn scratch(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("bwa4_bad_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fa = std::fs::read_to_string(tiny()).unwrap();
    let seq: String = fa.lines().skip(1).collect();
    let fq = dir.join("r.fq");
    let read = &seq[1000..1150];
    std::fs::write(&fq, format!("@r\n{read}\n+\n{}\n", "I".repeat(150))).unwrap();
    (dir, fq)
}

fn mem(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bwa-mem4"))
        .arg("mem")
        .args(args)
        .output()
        .unwrap()
}

/// Exited with an error that mentions `needle`, rather than a panic or a signal.
fn assert_refused(out: &Output, needle: &str) {
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains(needle), "wanted {needle:?} in:\n{err}");
    assert!(!err.contains("panicked"), "{err}");
}

/// A copy of the tiny index under `dir/t`, for a test to damage.
fn copy_index(dir: &Path) -> PathBuf {
    let prefix = dir.join("t");
    for ext in ["amb", "ann", "pac", "0123", "bwt.2bit.64"] {
        let from = format!("{}.{ext}", tiny().display());
        std::fs::copy(from, format!("{}.{ext}", prefix.display())).unwrap();
    }
    prefix
}

/// Keep only the first `len` bytes of one index file.
fn truncate(prefix: &Path, ext: &str, len: usize) {
    let path = format!("{}.{ext}", prefix.display());
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..len]).unwrap();
}

#[test]
fn damaged_index_files_are_named() {
    let (dir, fq) = scratch("index");
    let prefix = copy_index(&dir);
    let run = || mem(&[prefix.as_os_str(), fq.as_os_str()]);

    truncate(&prefix, "pac", 10_000);
    assert_refused(&run(), "t.pac: 10000 bytes");

    let prefix2 = copy_index(&dir);
    assert_eq!(prefix, prefix2);
    truncate(&prefix, "bwt.2bit.64", 300_000);
    assert_refused(&run(), "t.bwt.2bit.64: 300000 bytes");

    truncate(&prefix, "bwt.2bit.64", 10);
    assert_refused(&run(), ".bwt.2bit.64 is only 10 bytes");

    copy_index(&dir);
    let ann = format!("{}.ann", prefix.display());
    let text = std::fs::read_to_string(&ann).unwrap();
    let (_, rest) = text.split_once(' ').unwrap();
    std::fs::write(&ann, format!("1000 {rest}")).unwrap();
    assert_refused(&run(), "mixed index from two builds");

    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(feature = "hts-output")]
#[test]
fn unparsable_header_line_is_refused_before_a_bam_exists() {
    let (dir, fq) = scratch("hdr");
    let bam = dir.join("x.bam");
    let out = mem(&[
        "-H".as_ref(),
        "@XX\tbad".as_ref(),
        "-o".as_ref(),
        bam.as_os_str(),
        tiny().as_os_str(),
        fq.as_os_str(),
    ]);
    assert_refused(&out, "cannot parse the SAM header");
    assert!(!bam.exists(), "a refused run must not leave a file behind");

    // A line htslib can read still goes through.
    let out = mem(&[
        "-H".as_ref(),
        "@CO\tfine".as_ref(),
        "-o".as_ref(),
        bam.as_os_str(),
        tiny().as_os_str(),
        fq.as_os_str(),
    ]);
    assert!(out.status.success());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn long_read_mode_refuses_output_it_cannot_write() {
    let (dir, fq) = scratch("lr_out");
    for name in ["x.bam", "x.cram", "x.sam.gz", "x.BGZ"] {
        let path = dir.join(name);
        let out = mem(&[
            "-x".as_ref(),
            "ont2d".as_ref(),
            "-o".as_ref(),
            path.as_os_str(),
            tiny().as_os_str(),
            fq.as_os_str(),
        ]);
        assert_refused(&out, "SAM text only");
        assert!(!path.exists(), "{name}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn long_read_mode_carries_the_read_group_and_extra_header() {
    let (dir, fq) = scratch("lr_rg");
    // A private copy of the FASTA, so the minimizer cache is not written beside the fixture.
    let fa = dir.join("ref.fa");
    std::fs::copy(tiny(), &fa).unwrap();
    let out = mem(&[
        "-x".as_ref(),
        "ont2d".as_ref(),
        "-R".as_ref(),
        "@RG\tID:lane1\tSM:s".as_ref(),
        "-H".as_ref(),
        "@CO\thello".as_ref(),
        fa.as_os_str(),
        fq.as_os_str(),
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    let sam = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = sam.lines().collect();
    assert!(lines.contains(&"@RG\tID:lane1\tSM:s"), "{sam}");
    assert!(lines.contains(&"@CO\thello"), "{sam}");
    let pg = lines.iter().find(|l| l.starts_with("@PG")).unwrap();
    assert_eq!(pg.matches('\t').count(), 4, "a real tab split @PG: {pg}");
    let record = lines.iter().find(|l| !l.starts_with('@')).unwrap();
    assert!(record.contains("\tRG:Z:lane1"), "{record}");
    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(unix)]
#[test]
fn a_non_utf8_argument_is_an_error_not_a_panic() {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::OsStr::from_bytes(b"/nonexistent/caf\xE9.fq");
    let out = mem(&[tiny().as_os_str(), path]);
    assert_refused(&out, "No such file");
}
