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

fn nt4(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4,
    }
}

fn read_fastq(path: &str, limit: usize) -> Vec<Vec<u8>> {
    let mut r = needletail::parse_fastx_file(path).expect("open fastq");
    let mut out = Vec::new();
    while let Some(rec) = r.next() {
        if out.len() >= limit {
            break;
        }
        out.push(rec.expect("rec").seq().iter().map(|&b| nt4(b)).collect());
    }
    out
}

fn load_sa(path: &str, len: usize) -> Vec<i64> {
    let t = Instant::now();
    let mut sa = vec![0i64; len];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(sa.as_mut_ptr() as *mut u8, len * 8) };
    BufReader::with_capacity(1 << 24, File::open(path).expect("open sa"))
        .read_exact(bytes)
        .expect("read sa");
    eprintln!("SA loaded in {:.0}s", t.elapsed().as_secs_f64());
    sa
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (prefix, reads_path, sa_path) = (&a[1], a[2].as_str(), a[3].as_str());
    let nreads: usize = std::env::var("NREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let leaves: usize = std::env::var("LEAVES").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 22);
    let opt = MemOpt::default();

    let reads = read_fastq(reads_path, nreads);
    let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
    eprintln!("{} reads", reads.len());

    let fm = FmIndex::load(Path::new(prefix)).expect("load fm");
    let ref_len = fm.reference().len();
    let refseq = fm.reference().to_vec();
    let sa = load_sa(sa_path, ref_len + 1);
    let tb = Instant::now();
    let lsa = LearnedSa::from_sa(refseq, sa, leaves);
    eprintln!("LearnedSa (packed) built in {:.0}s", tb.elapsed().as_secs_f64());

    let time = |label: &str, f: &dyn Fn() -> Vec<Vec<bwa_index::Smem>>| {
        // warm + 3 runs, report the median.
        let _ = f();
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

    let fm_smems = bwa_seed::mem_collect_smem_batched(&fm, &refs, &opt);
    let hy_smems = bwa_seed::mem_collect_smem_hybrid(&fm, &lsa, &refs, &opt);
    // Byte-identity: per-read SMEM sets (chaining sorts by (m,n), so the set is what matters).
    let key = |v: &[bwa_index::Smem]| {
        let mut k: Vec<(u32, u32, i64, i64)> = v.iter().map(|s| (s.m, s.n, s.k, s.s)).collect();
        k.sort_unstable();
        k
    };
    let diffs = fm_smems
        .iter()
        .zip(&hy_smems)
        .filter(|(f, h)| key(f) != key(h))
        .count();
    eprintln!("SMEM-set diffs (FM vs hybrid): {diffs} / {} reads", reads.len());

    let t_fm = time("FM   seeding", &|| bwa_seed::mem_collect_smem_batched(&fm, &refs, &opt));
    let t_hy = time("hybrid seeding", &|| {
        bwa_seed::mem_collect_smem_hybrid(&fm, &lsa, &refs, &opt)
    });
    eprintln!("seeding speedup: {:.2}x", t_fm / t_hy);
}
