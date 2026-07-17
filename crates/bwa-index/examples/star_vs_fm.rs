//! **The untested cell**: is a STAR-style search (exact prefix table -> narrowed binary search over a
//! flat SA) faster than FM backward extension when BOTH are lockstep-batched?
//!
//! Prior evidence is contradictory. LISA round-1 with co-located keys measured 63k reads/s vs FM
//! round-1's 30k (2.1x) -- but BOTH unbatched. The hybrid then measured 0.47x, because our FM is
//! lockstep-batched and LISA was not. So the comparison that decides it has never been run.
//!
//! Model: resolving a 32-mer costs FM ~32 dependent rounds (one per base). STAR costs 1 table lookup
//! + ~2*log2(23) binary-search probes; without co-located keys each probe is 2 dependent accesses
//! (read sa[mid], then the reference at that position), so ~19 rounds. That is the PESSIMISTIC
//! variant -- BWA-MEME's 13-byte pos+key record makes a probe 1 access (~10 rounds) but costs 81 GB.
//!
//!   cargo run --release -p bwa-index --example star_vs_fm -- work/genome.fa work/genome.lisa_sa
use bwa_index::{FmIndex, Smem};
use std::time::Instant;

const K: usize = 14; // STAR's genomeSAindexNbases default
const QLEN: usize = 32;
const W: usize = 16; // lockstep width, same as our production seeding

fn now() -> Instant { Instant::now() }

fn fwd_ext(fm: &FmIndex, s: Smem, a: usize) -> Smem {
    let mut f = s;
    std::mem::swap(&mut f.k, &mut f.l);
    let mut e = fm.backward_ext(f, 3 - a);
    std::mem::swap(&mut e.k, &mut e.l);
    e
}

fn main() {
    let mut args = std::env::args().skip(1);
    let prefix = args.next().expect("usage: star_vs_fm <index prefix> <flat sa>");
    let sa_path = args.next().expect("usage: star_vs_fm <index prefix> <flat sa>");
    let fm = FmIndex::load(std::path::Path::new(&prefix)).expect("load index");
    let refseq = fm.reference();
    let n = fm.ref_seq_len;

    let f = std::fs::File::open(&sa_path).expect("open flat sa");
    let mm = unsafe { memmap2::Mmap::map(&f).expect("mmap flat sa") };
    let sa: &[i64] = unsafe { std::slice::from_raw_parts(mm.as_ptr().cast(), mm.len() / 8) };
    println!("flat SA: {} entries (ref_seq_len {})", sa.len(), n);
    assert_eq!(sa.len() as i64, n, "flat SA does not match this index");

    // The prefix table IS the FM bi-interval of each K-mer: [k, k+s) is exactly STAR's SA range.
    // Build it breadth-first over the trie (a node's 4 children share one cp_occ block pair).
    let t = now();
    let c = fm.counts();
    let mut level: Vec<Smem> = (0..4)
        .map(|a| Smem { rid: 0, m: 0, n: 0, k: c[a], l: c[3 - a], s: c[a + 1] - c[a] })
        .collect();
    for d in 1..K {
        let mut next = Vec::with_capacity(level.len() * 4);
        for node in &level {
            for a in 0..4 { next.push(fwd_ext(&fm, *node, a)); }
        }
        level = next;
        eprint!("\r  building {}-mer table: depth {}/{} ({} nodes)   ", K, d + 1, K, level.len());
    }
    eprintln!();
    let table: Vec<(i64, i64)> = level.iter().map(|m| (m.k, m.s)).collect();
    let mean_s: f64 = table.iter().map(|e| e.1 as f64).sum::<f64>() / table.len() as f64;
    println!("built 4^{K} = {} entries in {:.1}s; mean SA range = {:.1} rows -> ~{:.1} probes/search",
             table.len(), t.elapsed().as_secs_f64(), mean_s, 2.0 * mean_s.log2());

    // Queries: real 32-mers taken from random genome positions (so every one exists).
    const NQ: usize = 200_000;
    let mut st = 0x9E3779B97F4A7C15u64;
    let mut queries: Vec<[u8; QLEN]> = Vec::with_capacity(NQ);
    while queries.len() < NQ {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        let p = (st % (n as u64 - QLEN as u64 - 1)) as usize;
        let mut q = [0u8; QLEN];
        let mut ok = true;
        for i in 0..QLEN {
            let b = refseq[p + i];
            if b >= 4 { ok = false; break; }
            q[i] = b;
        }
        if ok { queries.push(q); }
    }
    println!("{NQ} real {QLEN}-mers from the genome\n");

    // The table-average SA range is not what a read hits. Queries are drawn FROM the genome, so a
    // K-mer occurring t times is t times likelier to be sampled: the distribution a real search sees
    // is occurrence-weighted, and repeats dominate it. Measure that before trusting any probe-count
    // model built on the table average.
    let mut ranges: Vec<i64> = queries.iter().map(|q| {
        let mut idx = 0usize;
        for b in 0..K { idx = (idx << 2) | q[b] as usize; }
        table[idx].1
    }).collect();
    ranges.sort_unstable();
    let pick = |f: f64| ranges[((ranges.len() - 1) as f64 * f) as usize];
    let wmean: f64 = ranges.iter().map(|&r| r as f64).sum::<f64>() / ranges.len() as f64;
    println!("SA range actually hit by a read (occurrence-weighted over {NQ} real {K}-mers):");
    println!("  table average : {:>12.1} rows  -> {:>5.1} probes", mean_s, 2.0 * mean_s.log2());
    println!("  weighted mean : {:>12.1} rows  -> {:>5.1} probes", wmean, 2.0 * wmean.max(1.0).log2());
    println!("  median        : {:>12} rows", pick(0.5));
    println!("  p90 / p99     : {:>12} / {} rows", pick(0.90), pick(0.99));
    println!("  max           : {:>12} rows  -> {:.1} probes", ranges[ranges.len()-1],
             2.0 * (ranges[ranges.len()-1].max(1) as f64).log2());
    println!();
    if std::env::var_os("STAR_VS_FM_TIME").is_none() {
        println!("(timing skipped: the flat SA + table + index need ~70 GB and this host thrashes.\n \
                  set STAR_VS_FM_TIME=1 to run it anyway)");
        return;
    }

    // ---- Arm A: FM forward extension, lockstep W, prefetched. 32 dependent rounds.
    let t = now();
    let mut out_a: Vec<Smem> = Vec::with_capacity(NQ);
    for chunk in queries.chunks(W) {
        let mut s: Vec<Smem> = chunk.iter()
            .map(|q| { let a = q[0] as usize; Smem { rid: 0, m: 0, n: 0, k: c[a], l: c[3-a], s: c[a+1]-c[a] } })
            .collect();
        for j in 1..QLEN {
            for x in &s { fm.prefetch_occ(x.l, x.l + x.s); }
            for (i, x) in s.iter_mut().enumerate() { *x = fwd_ext(&fm, *x, chunk[i][j] as usize); }
        }
        out_a.extend_from_slice(&s);
    }
    let ta = t.elapsed().as_secs_f64();

    // ---- Arm B: table lookup + lockstep binary search on the flat SA.
    let t = now();
    let mut out_b: Vec<(i64, i64)> = Vec::with_capacity(NQ);
    for chunk in queries.chunks(W) {
        let m = chunk.len();
        // Table gives the SA range of the first K bases in ONE access.
        let mut lo = [0i64; W]; let mut hi = [0i64; W];
        for i in 0..m {
            let mut idx = 0usize;
            for b in 0..K { idx = (idx << 2) | chunk[i][b] as usize; }
            let (k, s) = table[idx];
            lo[i] = k; hi[i] = k + s;
        }
        // Two lockstep binary searches (lower then upper bound) over the remaining QLEN-K bases.
        for bound in 0..2 {
            let (mut l, mut h) = (lo, hi);
            loop {
                let mut any = false;
                let mut mid = [0i64; W];
                for i in 0..m {
                    if l[i] < h[i] { mid[i] = (l[i] + h[i]) / 2; any = true;
                        unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) sa.as_ptr().add(mid[i] as usize), options(nostack, readonly, preserves_flags)); } }
                }
                if !any { break; }
                let mut pos = [0i64; W];
                for i in 0..m {
                    if l[i] < h[i] { pos[i] = sa[mid[i] as usize];
                        let r = (pos[i] + K as i64) as usize;
                        if r < refseq.len() { unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) refseq.as_ptr().add(r), options(nostack, readonly, preserves_flags)); } } }
                }
                for i in 0..m {
                    if l[i] >= h[i] { continue; }
                    // Compare the reference suffix at sa[mid] against the query beyond the table's K bases.
                    let mut less = false;
                    for b in K..QLEN {
                        let rp = pos[i] + b as i64;
                        let rb = if (rp as usize) < refseq.len() { refseq[rp as usize] } else { 4 };
                        if rb != chunk[i][b] { less = rb < chunk[i][b]; break; }
                        if b == QLEN - 1 { less = bound == 1; } // ties go left for lower, right for upper
                    }
                    if less { l[i] = mid[i] + 1; } else { h[i] = mid[i]; }
                }
            }
            if bound == 0 { for i in 0..m { lo[i] = l[i]; } } else { for i in 0..m { hi[i] = l[i]; } }
        }
        for i in 0..m { out_b.push((lo[i], hi[i] - lo[i])); }
    }
    let tb = t.elapsed().as_secs_f64();

    // ---- Both must produce the same SA interval.
    let mut bad = 0;
    for i in 0..NQ {
        if out_a[i].k != out_b[i].0 || out_a[i].s != out_b[i].1 {
            if bad < 3 { eprintln!("  MISMATCH q{i}: FM (k={}, s={}) vs STAR (k={}, s={})",
                                   out_a[i].k, out_a[i].s, out_b[i].0, out_b[i].1); }
            bad += 1;
        }
    }
    println!("{:<34} {:>9} {:>14}", "", "wall(s)", "ns/query");
    println!("{:<34} {:>9.3} {:>14.0}", "A: FM fwd extension (lockstep W=16)", ta, ta * 1e9 / NQ as f64);
    println!("{:<34} {:>9.3} {:>14.0}", "B: STAR table + binary search", tb, tb * 1e9 / NQ as f64);
    println!("\nB vs A: {:.2}x", ta / tb);
    println!("agreement: {} / {} queries {}", NQ - bad, NQ,
             if bad == 0 { "IDENTICAL" } else { "*** MISMATCHED -- arm B is wrong, the timing is meaningless ***" });
}
