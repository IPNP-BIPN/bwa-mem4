//! Genome-scale seeding benchmark: FM-index `mem_collect_smem` vs learned-index `mem_collect_smem_lsa`.
//!
//! Two modes (run as separate processes so the FM index and the ~50GB learned SA never coexist):
//!
//!   cargo run --release -p bwa-seed --example lisa_bench -- fm   <prefix> <reads.fq> <sa_file>
//!   cargo run --release -p bwa-seed --example lisa_bench -- lisa <prefix> <reads.fq> <sa_file>
//!
//! `fm` loads `<prefix>.{bwt.2bit.64,0123}`, and if `<sa_file>` does not exist extracts the full
//! suffix array from the FM index (`get_sa_batch`, lockstep) and writes it as raw little-endian i64.
//! Then it times `mem_collect_smem` over the reads. `lisa` loads `<prefix>.0123` + `<sa_file>`, builds
//! `LearnedSa::from_sa`, and times `mem_collect_smem_lsa`. Both print a checksum over every SMEM's
//! `(m, n, k, s)` — equal checksums across the two runs prove byte-identical seeding on the genome.
//!
//! Env: `NREADS` (default 200000) caps reads; `LEAVES_SHIFT` (default 7) sets RMI leaves = n >> shift.

use bwa_core::MemOpt;
use bwa_index::lisa::LearnedSa;
use bwa_index::{FmIndex, Smem};
use bwa_seed::lisa_seed::mem_collect_smem_lsa;
use bwa_seed::mem_collect_smem;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

fn encode(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .map(|&b| match b {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => 4,
        })
        .collect()
}

fn read_fastq(path: &str, limit: usize) -> Vec<Vec<u8>> {
    let f = BufReader::new(File::open(path).expect("open reads"));
    let mut reads = Vec::new();
    let mut lines = f.lines();
    while let Some(h) = lines.next() {
        let _ = h.expect("header");
        let seq = lines.next().expect("seq line").expect("seq");
        let _ = lines.next(); // '+'
        let _ = lines.next(); // qual
        reads.push(encode(seq.as_bytes()));
        if reads.len() >= limit {
            break;
        }
    }
    reads
}

/// FNV-1a over the SMEM `(m, n, k, s)` fields — an order-sensitive fingerprint of the seed set.
fn hash_smems(h: &mut u64, smems: &[Smem]) {
    let mix = |v: u64, h: &mut u64| {
        *h ^= v;
        *h = h.wrapping_mul(0x100000001b3);
    };
    for s in smems {
        mix(u64::from(s.m), h);
        mix(u64::from(s.n), h);
        mix(s.k as u64, h);
        mix(s.s as u64, h);
    }
}

fn extract_sa(fm: &FmIndex, sa_path: &str) {
    let n = fm.ref_seq_len as usize;
    eprintln!("extracting SA for {n} rows -> {sa_path}");
    let t = Instant::now();
    let mut file = BufWriter::with_capacity(1 << 24, File::create(sa_path).expect("create sa"));
    const CHUNK: usize = 16_000_000;
    let mut pos = vec![0i64; CHUNK];
    let mut out = vec![0i64; CHUNK];
    let mut base = 0usize;
    while base < n {
        let w = CHUNK.min(n - base);
        for (i, p) in pos[..w].iter_mut().enumerate() {
            *p = (base + i) as i64;
        }
        fm.get_sa_batch(&pos[..w], &mut out[..w]);
        // Reinterpret the i64 slice as bytes and write in one go (little-endian on this platform).
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, w * 8) };
        file.write_all(bytes).expect("write sa");
        base += w;
        if base % (CHUNK * 8) < CHUNK {
            eprintln!("  {:.1}% ({:.0}s)", 100.0 * base as f64 / n as f64, t.elapsed().as_secs_f64());
        }
    }
    file.flush().expect("flush sa");
    eprintln!("SA extracted in {:.0}s", t.elapsed().as_secs_f64());
}

fn load_sa(sa_path: &str, expect_len: usize) -> Vec<i64> {
    eprintln!("loading SA from {sa_path} ({expect_len} rows)");
    let t = Instant::now();
    let mut sa = vec![0i64; expect_len];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(sa.as_mut_ptr() as *mut u8, expect_len * 8) };
    let mut f = BufReader::with_capacity(1 << 24, File::open(sa_path).expect("open sa"));
    f.read_exact(bytes).expect("read sa");
    eprintln!("SA loaded in {:.0}s", t.elapsed().as_secs_f64());
    sa
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: lisa_bench <fm|lisa> <prefix> <reads.fq> <sa_file>");
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let prefix = args[2].clone();
    let reads_path = args[3].as_str();
    let sa_path = args[4].as_str();
    let limit: usize = std::env::var("NREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let leaves_shift: usize =
        std::env::var("LEAVES_SHIFT").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let opt = MemOpt::default();

    let reads = read_fastq(reads_path, limit);
    let total_bases: usize = reads.iter().map(|r| r.len()).sum();
    eprintln!("{} reads, {} bases", reads.len(), total_bases);

    match mode {
        "fm" => {
            let fm = FmIndex::load(Path::new(&prefix)).expect("load fm");
            if !Path::new(sa_path).exists() {
                extract_sa(&fm, sa_path);
            }
            let t = Instant::now();
            let mut hash = 0xcbf29ce484222325u64;
            let mut nsmem = 0usize;
            for r in &reads {
                let smems = mem_collect_smem(&fm, r, &opt);
                nsmem += smems.len();
                hash_smems(&mut hash, &smems);
            }
            let el = t.elapsed().as_secs_f64();
            println!(
                "FM   : {} reads, {} smems, {:.3}s, {:.0} reads/s, {:.2} Mbase/s, checksum {:016x}",
                reads.len(), nsmem, el, reads.len() as f64 / el, total_bases as f64 / el / 1e6, hash
            );
        }
        "lisa" => {
            let reference = std::fs::read(format!("{prefix}.0123")).expect("read .0123");
            let sa = load_sa(sa_path, reference.len() + 1);
            let n_leaves = (sa.len() >> leaves_shift).max(1);
            eprintln!("building LearnedSa (ref {}, sa {}, {} leaves)...", reference.len(), sa.len(), n_leaves);
            let tb = Instant::now();
            let lsa = LearnedSa::from_sa(reference, sa, n_leaves);
            eprintln!("LearnedSa built in {:.0}s", tb.elapsed().as_secs_f64());
            let t = Instant::now();
            let mut hash = 0xcbf29ce484222325u64;
            let mut nsmem = 0usize;
            for r in &reads {
                let smems = mem_collect_smem_lsa(&lsa, r, &opt);
                nsmem += smems.len();
                hash_smems(&mut hash, &smems);
            }
            let el = t.elapsed().as_secs_f64();
            println!(
                "LISA : {} reads, {} smems, {:.3}s, {:.0} reads/s, {:.2} Mbase/s, checksum {:016x}",
                reads.len(), nsmem, el, reads.len() as f64 / el, total_bases as f64 / el / 1e6, hash
            );
        }
        _ => {
            eprintln!("unknown mode {mode}");
            std::process::exit(2);
        }
    }
}
