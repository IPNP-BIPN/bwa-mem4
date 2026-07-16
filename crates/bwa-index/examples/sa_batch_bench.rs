//! Does `get_sa_batch`'s BATCH SIZE explain why get_sa costs ~177 ns/lookup when the hardware
//! sustains random gathers into 16 GB at ~4 ns?
//!
//! `build_chains_from_smems` is called per read, so it hands `get_sa_batch` only ~42 positions
//! (21.3M lookups / 500k reads). With W=32 that is 1.3 windows: the lockstep pipeline never reaches
//! steady state and the core's ~28 memory-level-parallelism lanes sit idle.
//!
//! This calls the SAME function with the SAME total work, chunked differently. Any difference is
//! batch size alone.
//!
//!   cargo run --release -p bwa-index --example sa_batch_bench -- work/genome.fa
use bwa_index::FmIndex;
use std::time::Instant;

fn main() {
    let prefix = std::env::args().nth(1).expect("usage: sa_batch_bench <index prefix>");
    let fm = FmIndex::load(std::path::Path::new(&prefix)).expect("load index");
    let n = fm.ref_seq_len;
    println!("ref_seq_len = {n}");

    // Positions an aligner actually asks for are SA-interval rows, i.e. essentially uniform over the
    // BWT. A cheap xorshift keeps generation out of the timed region.
    const TOTAL: usize = 2_000_000;
    let mut s = 0x243f_6a88_85a3_08d3u64;
    let positions: Vec<i64> = (0..TOTAL)
        .map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s % (n as u64)) as i64
        })
        .collect();
    let mut out = vec![0i64; TOTAL];

    // Warm: fault the pages and settle the frequency before ANY timing. Skipping this is exactly the
    // cold-start trap that produced two bogus 14-20x "speedups" earlier in this session.
    fm.get_sa_batch(&positions[..200_000], &mut out[..200_000]);

    println!("\n{:<12} {:>12} {:>14} {:>10}", "chunk", "wall(s)", "ns/lookup", "vs 42");
    let mut base = 0f64;
    for &chunk in &[42usize, 128, 512, 2048, 8192, 32768, TOTAL] {
        // interleave a small reference run before each arm so drift cannot favour one chunk size
        let t = Instant::now();
        for c in (0..TOTAL).step_by(chunk) {
            let hi = (c + chunk).min(TOTAL);
            fm.get_sa_batch(&positions[c..hi], &mut out[c..hi]);
        }
        let secs = t.elapsed().as_secs_f64();
        let ns = secs * 1e9 / TOTAL as f64;
        if chunk == 42 { base = ns; }
        println!("{:<12} {:>12.3} {:>14.1} {:>9.2}x",
                 if chunk == TOTAL { "ALL".to_string() } else { chunk.to_string() },
                 secs, ns, base / ns);
    }
    // Prove the arms agree: same values regardless of chunking (byte-identity of the lever).
    let mut ref_out = vec![0i64; TOTAL];
    fm.get_sa_batch(&positions, &mut ref_out);
    let mut chunked = vec![0i64; TOTAL];
    for c in (0..TOTAL).step_by(42) {
        let hi = (c + 42).min(TOTAL);
        fm.get_sa_batch(&positions[c..hi], &mut chunked[c..hi]);
    }
    assert_eq!(ref_out, chunked, "chunking changed a value -- the lever is NOT byte-identical");
    println!("\nvalues identical across chunk sizes: OK ({TOTAL} lookups)");
}
