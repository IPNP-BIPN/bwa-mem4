//! Zhang et al. (CCGrid'13 §IV) bin occurrence computations by BWT region so a round's accesses land
//! in a window whose pages fit the TLB. They report 1.43-1.54x single-thread / 1.21-1.36x
//! multithreaded -- the only lever in this codebase's history that survives thread contention,
//! because it cuts shared page-walk traffic rather than hiding per-thread latency.
//!
//! The mechanism needs page REUSE. A bin of 2^S rows is 2^S bytes of cp_occ = 2^S/16384 pages, and
//! Zhang gets many accesses per page only because their batch is 2^24 computations. Our lockstep runs
//! N=16, which would give ~0 accesses per page. **The question this answers: at what batch N does
//! binning start to pay?** Our slots carry a 3.6 KB `prev[]` each, so if the knee sits above what RAM
//! allows, the idea is dead before it is built.
//!
//! Both arms are lockstep W=16 with prefetch (our production design); arm B pays its counting sort
//! inside the timed region. Warm before timing and interleave the arms -- measuring this cold makes
//! the random arm look FASTER as N grows, which is how the first version of this file lied.
//!
//!   cargo run --release -p bwa-index --example bwt_binning -- work/genome.fa
use bwa_index::{FmIndex, Smem};
use std::time::Instant;

const W: usize = 16;

fn pass(fm: &FmIndex, pos: &[i64], c: &[i64; 5]) -> i64 {
    let mut acc = 0i64;
    for chunk in pos.chunks(W) {
        for &p in chunk {
            fm.prefetch_occ(p, p + 1);
        }
        for &p in chunk {
            let s = Smem { rid: 0, m: 0, n: 0, k: p, l: c[0], s: 1 };
            acc = acc.wrapping_add(fm.backward_ext(s, (p & 3) as usize).k);
        }
    }
    acc
}

/// Counting-sort `pos` by BWT bin. Returns the reordered vector; cost belongs to arm B.
fn bin_sort(pos: &[i64], shift: u32, nbins: usize) -> Vec<i64> {
    let mut cnt = vec![0u32; nbins + 1];
    for &p in pos { cnt[(p >> shift) as usize + 1] += 1; }
    for i in 0..nbins { cnt[i + 1] += cnt[i]; }
    let mut out = vec![0i64; pos.len()];
    for &p in pos { let b = (p >> shift) as usize; out[cnt[b] as usize] = p; cnt[b] += 1; }
    out
}

fn main() {
    let prefix = std::env::args().nth(1).expect("usage: bwt_binning <index prefix>");
    let fm = FmIndex::load(std::path::Path::new(&prefix)).expect("load index");
    let c = fm.counts();
    let n = fm.ref_seq_len;
    let cp_bytes = ((n >> 6) + 1) as f64 * 64.0;
    println!("cp_occ spans {:.1} GB = {:.0}k pages of 16 KB (L2 dTLB holds ~3072)\n",
             cp_bytes / 1073741824.0, cp_bytes / 16384.0 / 1000.0);

    for logn in [12u32, 16, 20, 24] {
        let nq = 1usize << logn;
        let mut st = 0x243F6A8885A308D3u64;
        let pos: Vec<i64> = (0..nq).map(|_| {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            (st % (n as u64)) as i64
        }).collect();

        println!("--- batch N = {nq} ---");
        println!("  {:>8} {:>7} {:>9} {:>10} {:>11} {:>11} {:>8}",
                 "bin rows", "bins", "pages/bin", "acc/bin", "random(ns)", "binned(ns)", "speedup");
        for shift in [20u32, 24, 28] {
            let nbins = ((n >> shift) + 1) as usize;
            // A bin of 2^shift BWT rows covers 2^shift/64 cp_occ blocks x 64 B = 2^shift BYTES.
            let bin_bytes = (1u64 << shift) as f64;
            let pages_per_bin = bin_bytes / 16384.0;
            let acc_per_bin = nq as f64 / nbins as f64;

            let binned = bin_sort(&pos, shift, nbins);
            // Warm BOTH arms fully before timing either: the first touch of these positions faults
            // pages and ramps the clock, and it would be charged entirely to whichever arm ran first.
            let w1 = pass(&fm, &pos, &c);
            let w2 = pass(&fm, &binned, &c);
            assert_eq!(w1, w2, "binning changed the result -- the arms are not equivalent");

            let (mut best_r, mut best_b) = (f64::MAX, f64::MAX);
            for _ in 0..3 {
                let t = Instant::now(); let a = pass(&fm, &pos, &c);
                let r = t.elapsed().as_secs_f64() * 1e9 / nq as f64;
                let t = Instant::now();
                let bs = bin_sort(&pos, shift, nbins);   // arm B pays its own sort, every rep
                let b2 = pass(&fm, &bs, &c);
                let b = t.elapsed().as_secs_f64() * 1e9 / nq as f64;
                assert_eq!(a, b2);
                best_r = best_r.min(r); best_b = best_b.min(b);
            }
            println!("  {:>8} {:>7} {:>9.0} {:>10.2} {:>11.1} {:>11.1} {:>7.2}x",
                     format!("2^{shift}"), nbins, pages_per_bin, acc_per_bin, best_r, best_b, best_r / best_b);
        }
        println!();
    }
}
