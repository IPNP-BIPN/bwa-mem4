//! Genome-scale seeding benchmark: FM `mem_collect_smem_batched` vs the RAM-optimized hybrid
//! `mem_collect_smem_hybrid` (LISA round-1 + FM rounds 2/3). Reports per-batch seeding time (median
//! of a few runs), verifies the SMEM sets are identical, and prints peak RSS.
//!
//! Usage: hybrid_bench <fm_prefix> <reads.fq> <sa_file>   (env NREADS, LEAVES)

use bwa_core::MemOpt;
use bwa_index::lisa::LearnedSa;
use bwa_index::FmIndex;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// Encode one ASCII base as BWA's 2-bit "nt4" alphabet.
///
/// # Parameters
/// - `b`: a single ASCII byte from a FASTQ sequence line, either case.
///
/// # Returns
/// 0/1/2/3 for A/C/G/T respectively, and 4 for ANY other byte (N, IUPAC codes, junk). 4 is the
/// standard BWA "ambiguous" code: seeding cannot extend through it, so such positions simply break
/// SMEMs rather than being an error.
fn nt4(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4,
    }
}

/// Load up to `limit` reads from a FASTQ, converting each to nt4 codes.
///
/// # Parameters
/// - `path`: FASTQ path. `needletail` sniffs the format, so plain or gzipped both work.
/// - `limit`: maximum number of reads to keep. Comes from the `NREADS` env var (default 50_000). It
///   sets the size of the benchmark batch, so it directly scales both arms' wall time; too small and
///   fixed per-batch costs dominate the comparison.
///
/// # Returns
/// One `Vec<u8>` per read, in file order, holding nt4 codes (0..=4). Quality strings and read names
/// are discarded: seeding uses neither. Reads are NOT length-filtered, so a read shorter than the
/// minimum seed length simply yields no SMEMs.
fn read_fastq(path: &str, limit: usize) -> Vec<Vec<u8>> {
    let mut r = needletail::parse_fastx_file(path).expect("open fastq");
    // Accumulates the decoded reads. Growth is unbounded up to `limit`.
    let mut out = Vec::new();
    while let Some(rec) = r.next() {
        if out.len() >= limit {
            break;
        }
        out.push(rec.expect("rec").seq().iter().map(|&b| nt4(b)).collect());
    }
    out
}

/// Read a flat suffix array off disk into memory.
///
/// # Parameters
/// - `path`: file containing exactly `len` little-endian `i64`s and nothing else, dumped from an
///   index built on the SAME reference. There is no header and no checksum, so a mismatched file is
///   caught only by the length (`read_exact` fails) or, silently, not at all.
/// - `len`: number of SA entries expected, supplied by the caller as `ref_len + 1` (the `+1` is the
///   sentinel row). Determines the allocation: 8 bytes each, so ~50 GB for a human genome.
///
/// # Returns
/// `sa[i]` = the reference POSITION of the suffix at SA ROW `i`. Read fully into RAM (not mmapped)
/// because `LearnedSa::from_sa` consumes it and the load time is reported once, outside all timing.
fn load_sa(path: &str, len: usize) -> Vec<i64> {
    // Load timer: reported for operator awareness only, never part of a benchmark figure.
    let t = Instant::now();
    let mut sa = vec![0i64; len];
    // Reinterpret the i64 buffer as raw bytes so the file can be slurped straight into it with no
    // intermediate copy or per-element decode. Assumes little-endian, which matches the dump format
    // and every host this runs on.
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(sa.as_mut_ptr() as *mut u8, len * 8) };
    // 16 MB read buffer (1 << 24): large enough that a ~50 GB sequential read is not dominated by
    // syscall overhead. The value is a throughput tuning knob only, nothing depends on it.
    BufReader::with_capacity(1 << 24, File::open(path).expect("open sa"))
        .read_exact(bytes)
        .expect("read sa");
    eprintln!("SA loaded in {:.0}s", t.elapsed().as_secs_f64());
    sa
}

fn main() {
    // argv[1] = FM index prefix, argv[2] = reads FASTQ, argv[3] = flat SA dump. All three are
    // required and indexed unchecked, so a missing argument panics with an index-out-of-bounds.
    let a: Vec<String> = std::env::args().collect();
    let (prefix, reads_path, sa_path) = (&a[1], a[2].as_str(), a[3].as_str());
    // Batch size in READS (env NREADS, default 50_000). Sets how much work each timed call does;
    // both arms always get the identical batch, so it does not bias the ratio.
    let nreads: usize = std::env::var("NREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    // Number of leaves in the learned suffix-array model (env LEAVES, default 2^22 = 4.19M). This is
    // the hybrid arm's accuracy/memory knob: more leaves means a tighter predicted position range,
    // so a shorter last-mile scan, but more RAM and a longer build. It affects the hybrid arm's
    // SPEED only; the SMEMs it returns must be identical either way, which the diff check verifies.
    let leaves: usize = std::env::var("LEAVES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 22);
    // Alignment parameters at their BWA-MEM defaults. Only the seeding-related fields matter here
    // (minimum seed length, re-seed ratio); both arms are handed the same struct.
    let opt = MemOpt::default();

    // The benchmark batch: nt4-encoded reads, owned.
    let reads = read_fastq(reads_path, nreads);
    // Borrowed view of the same reads, which is the shape both seeding entry points take. Built once
    // so neither timed closure pays for it.
    let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
    eprintln!("{} reads", reads.len());

    let fm = FmIndex::load(Path::new(prefix)).expect("load fm");
    // Reference length in BASES (forward strand plus reverse complement), used to size the SA read.
    let ref_len = fm.reference().len();
    // Owned copy of the nt4 reference. Copied rather than borrowed because `LearnedSa::from_sa`
    // takes ownership; this is the single largest transient allocation in the program.
    let refseq = fm.reference().to_vec();
    // Flat suffix array, `ref_len + 1` entries: the extra row is the sentinel suffix.
    let sa = load_sa(sa_path, ref_len + 1);
    // Build timer for the learned index. Reported separately and deliberately NOT charged to the
    // hybrid arm: it is a one-off cost an aligner would pay at index-build time, not per batch.
    let tb = Instant::now();
    // The learned suffix-array model that replaces FM round 1 in the hybrid arm. Consumes `refseq`
    // and `sa`, which is why both are unavailable afterwards.
    let lsa = LearnedSa::from_sa(refseq, sa, leaves);
    eprintln!(
        "LearnedSa (packed) built in {:.0}s",
        tb.elapsed().as_secs_f64()
    );

    // Timing harness. `label` is the printed row name; `f` is the arm under test, taken as a `&dyn
    // Fn` so both arms share one code path and neither gets inlined differently from the other.
    // Returns the median wall time in SECONDS for ONE full pass over the whole read batch (not per
    // read). The timed region is exactly one call to `f`, so it includes allocating that call's
    // output `Vec<Vec<Smem>>`, which both arms pay alike.
    let time = |label: &str, f: &dyn Fn() -> Vec<Vec<bwa_index::Smem>>| {
        // warm + 3 runs, report the median.
        // The discarded warm-up run faults in the index pages and settles the clock, so the ramp is
        // not charged to the first measured run.
        let _ = f();
        // Three wall times in seconds. Median of 3 (index [1] after sorting) rather than the mean:
        // it discards a single OS-scheduling outlier without needing more reps.
        let mut ts: Vec<f64> = (0..3)
            .map(|_| {
                let t = Instant::now();
                let _ = f();
                t.elapsed().as_secs_f64()
            })
            .collect();
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        eprintln!("{label}: {:.3}s (median of 3)", ts[1]);
        ts[1]
    };

    // Correctness arms, computed once and outside all timing. Each is one `Vec<Smem>` per input
    // read, in read order: `fm_smems` from pure FM seeding (the reference behaviour), `hy_smems`
    // from the LISA-round-1 hybrid (the candidate).
    let fm_smems = bwa_seed::mem_collect_smem_batched(&fm, &refs, &opt);
    let hy_smems = bwa_seed::mem_collect_smem_hybrid(&fm, &lsa, &refs, &opt);
    // Byte-identity: per-read SMEM sets (chaining sorts by (m,n), so the set is what matters).
    // Canonical form of one read's SMEM list: `(m, n, k, s)` = query start, query end, first SA ROW,
    // and SA row COUNT. Sorted, because the two arms may emit the same SMEMs in a different order
    // and downstream chaining re-sorts anyway, so order is not observable. `rid` and `l` are omitted
    // deliberately: `rid` is constant per read, and `l` (the reverse-strand start) is a function of
    // the rest. Comparing these keys is what "byte-identical seeding" means here.
    let key = |v: &[bwa_index::Smem]| {
        let mut k: Vec<(u32, u32, i64, i64)> = v.iter().map(|s| (s.m, s.n, s.k, s.s)).collect();
        k.sort_unstable();
        k
    };
    // Count of READS (not SMEMs) whose canonical SMEM set differs between the arms. Must be 0: a
    // nonzero count means the hybrid changes which seeds the aligner sees, which propagates into
    // chain order and out to the SAM, so any speedup it shows would be bought with different output.
    let diffs = fm_smems
        .iter()
        .zip(&hy_smems)
        .filter(|(f, h)| key(f) != key(h))
        .count();
    eprintln!(
        "SMEM-set diffs (FM vs hybrid): {diffs} / {} reads",
        reads.len()
    );

    // Median seconds per full-batch seeding pass for each arm. FM is timed first; there is no
    // interleaving, so slow host drift over the run would bias the hybrid arm.
    let t_fm = time("FM   seeding", &|| {
        bwa_seed::mem_collect_smem_batched(&fm, &refs, &opt)
    });
    let t_hy = time("hybrid seeding", &|| {
        bwa_seed::mem_collect_smem_hybrid(&fm, &lsa, &refs, &opt)
    });
    eprintln!("seeding speedup: {:.2}x", t_fm / t_hy);
}
