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
//! Reading the output. The example runs in two stages.
//!
//! Stage 1 (always runs, cheap) is the SA-range distribution. The probe-count model above is only as
//! good as the SA range a real query hits, and the table AVERAGE is the wrong statistic: queries are
//! drawn from the genome, so a K-mer occurring t times is sampled t times as often and repeats
//! dominate. The `weighted mean` / `median` / `p90` / `p99` / `max` rows are the honest inputs to
//! `2*log2(range)`. If the weighted mean is orders of magnitude above the table average, the STAR
//! model's "~19 rounds" is fiction and arm B loses before it is timed.
//!
//! Stage 2 (gated behind `STAR_VS_FM_TIME=1`) is the head-to-head:
//!   - `A: FM fwd extension`  : ns/query for 32 lockstep-batched forward extensions.
//!   - `B: STAR table + binary search` : ns/query for one table lookup plus two lockstep binary
//!                              searches.
//!   - `B vs A`               : ta / tb, so ABOVE 1.00x means STAR wins. A number near or below 1.00x
//!                              retires the STAR/learned-index line for this codebase, consistent
//!                              with the LISA result already on record.
//!   - `agreement`            : must print IDENTICAL. If it prints MISMATCHED, arm B computed the
//!                              wrong interval and its timing means nothing, fast or slow.
//!
//!   STAR_VS_FM_TIME=1 cargo run --release -p bwa-index --example star_vs_fm -- \
//!       work/genome.fa work/genome.lisa_sa
//!
//! Arg 1 is the index prefix. Arg 2 is a FLAT suffix array: `ref_seq_len` little-endian `i64`s with
//! `sa[i] == FmIndex::get_sa(i)` (49.6 GB on GRCh38). UNVERIFIED: no producer for this file exists in
//! this tree; it was dumped during the (since-removed) LISA work and lives in `work/` by hand. The
//! `assert_eq!` on `sa.len()` is the only guard that it belongs to this index.
use bwa_index::{FmIndex, Smem};
use std::time::Instant;

/// Prefix-table depth in BASES: the table stores the SA range of every K-mer, so it has 4^K entries.
/// 14 is STAR's `genomeSAindexNbases` default. Each +1 quadruples the table (4^14 = 268M entries at
/// 16 B = 4.3 GB here) and shortens arm B's binary search by ~2 probes; each -1 halves the memory and
/// lengthens the search. Changing it changes arm B's cost profile, not its correctness.
const K: usize = 14; // STAR's genomeSAindexNbases default
/// Query length in BASES. 32 is the seed length the model at the top of the file is written for: FM
/// pays one dependent round per base (32), STAR pays 1 table lookup plus the binary search over the
/// remaining `QLEN - K` = 18 bases. Must be > K or arm B has nothing left to search.
const QLEN: usize = 32;
/// Lockstep window width: queries processed together so their misses overlap. Applies to BOTH arms
/// (arm A's extension window and arm B's binary-search window), which is the whole point: the prior
/// LISA-vs-FM comparisons were invalid precisely because only one side was batched.
const W: usize = 16; // lockstep width, same as our production seeding

/// Timer start, wrapped only so the timed regions below read uniformly.
fn now() -> Instant {
    Instant::now()
}

/// Extend `s` by base `a` on the RIGHT. Our `backward_ext` only goes left, but the index is
/// bidirectional (FMD): swapping `k` and `l` puts the interval on the reverse-complement strand,
/// where a left-extension by the complementary base `3 - a` is a right-extension by `a`; swapping
/// back returns to the forward strand. Standard FMD trick, and it is why arm A needs no extra tables.
///
/// # Parameters
/// - `fm`: the loaded FMD index; read-only.
/// - `s`: the current bi-interval. `k` is the first SA ROW on the forward strand, `l` the first row
///   of the reverse-complement partner interval, `s` the shared row COUNT. `m`/`n`/`rid` are query
///   bookkeeping and are carried through untouched.
/// - `a`: the base to append on the right, encoded 0..=3 for A/C/G/T. Values >= 4 (an N) are invalid:
///   `3 - a` would underflow. Callers here guarantee 0..=3 by rejecting queries containing N.
///
/// # Returns
/// The bi-interval of the extended pattern, in the same convention as `s`. An empty result has
/// `s == 0`, meaning the extended pattern does not occur in the reference.
fn fwd_ext(fm: &FmIndex, s: Smem, a: usize) -> Smem {
    let mut f = s;
    std::mem::swap(&mut f.k, &mut f.l);
    let mut e = fm.backward_ext(f, 3 - a);
    std::mem::swap(&mut e.k, &mut e.l);
    e
}

fn main() {
    let mut args = std::env::args().skip(1);
    // argv[1]: index prefix. argv[2]: path to the flat suffix array dump described in the header.
    let prefix = args
        .next()
        .expect("usage: star_vs_fm <index prefix> <flat sa>");
    let sa_path = args
        .next()
        .expect("usage: star_vs_fm <index prefix> <flat sa>");
    let fm = FmIndex::load(std::path::Path::new(&prefix)).expect("load index");
    // The FMD reference as one base per byte, values 0..=3 for A/C/G/T and >= 4 for N. Indexed by
    // reference POSITION (forward strand then reverse complement). Arm B reads it to compare
    // suffixes; the query generator reads it to cut real K-mers out of the genome.
    let refseq = fm.reference();
    // Reference length in bases = number of SA rows. Bounds both coordinate spaces here.
    let n = fm.ref_seq_len;

    // The flat SA is mmapped rather than read: it is ~49.6 GB on GRCh38, so the OS pages in only the
    // entries the binary searches actually probe. `f` must outlive `mm`, and `mm` outlive `sa`.
    let f = std::fs::File::open(&sa_path).expect("open flat sa");
    let mm = unsafe { memmap2::Mmap::map(&f).expect("mmap flat sa") };
    // `sa[i]` is the reference POSITION of the suffix at SA ROW `i`: the mapping from row space to
    // position space, and the same function `FmIndex::get_sa(i)` computes by LF-walking. Arm B uses
    // it directly, which is exactly the tradeoff under test (one big table vs a walk).
    let sa: &[i64] = unsafe { std::slice::from_raw_parts(mm.as_ptr().cast(), mm.len() / 8) };
    println!("flat SA: {} entries (ref_seq_len {})", sa.len(), n);
    assert_eq!(sa.len() as i64, n, "flat SA does not match this index");

    // The prefix table IS the FM bi-interval of each K-mer: [k, k+s) is exactly STAR's SA range.
    // Build it breadth-first over the trie (a node's 4 children share one cp_occ block pair).
    //
    // Building it FROM the FM index rather than from the flat SA is what makes the comparison fair:
    // both arms then start from provably the same K-mer -> SA-range mapping, so any divergence in the
    // final `agreement` check is arm B's binary search, never a table-construction bug. Breadth-first
    // is also the cheap order: all 4 children of a node reuse the parent's two checkpoint blocks.
    // Cost is 4^K = 268M entries, ~4.3 GB; this is the reason the whole example needs ~70 GB.
    // Timer for table construction only; reported once, never part of either arm's cost.
    let t = now();
    // C array: c[a] = number of reference bases lexicographically below base `a`, c[4] = total.
    let c = fm.counts();
    // Depth-1 roots: the interval of the single base `a` is `[c[a], c[a+1])` on the forward strand,
    // and its reverse-complement partner starts at `c[3 - a]` (complement of `a`). Same seeding the
    // aligner's own SMEM walk uses.
    // The trie frontier: at the top of iteration `d` it holds the bi-intervals of all 4^d d-mers, in
    // lexicographic order of the d-mer (so index = the d-mer read as a base-4 number). Grows x4 per
    // level and ends at 4^K entries.
    let mut level: Vec<Smem> = (0..4)
        .map(|a| Smem {
            rid: 0,
            m: 0,
            n: 0,
            k: c[a],
            l: c[3 - a],
            s: c[a + 1] - c[a],
        })
        .collect();
    for d in 1..K {
        // Next frontier, built by appending each of the 4 bases to every node, in base order, which
        // is what preserves the base-4 index property stated above.
        let mut next = Vec::with_capacity(level.len() * 4);
        for node in &level {
            for a in 0..4 {
                next.push(fwd_ext(&fm, *node, a));
            }
        }
        level = next;
        eprint!(
            "\r  building {}-mer table: depth {}/{} ({} nodes)   ",
            K,
            d + 1,
            K,
            level.len()
        );
    }
    eprintln!();
    // The finished prefix table, indexed by the K-mer as a base-4 number. Each entry is
    // `(first SA row, row count)`, i.e. STAR's SA range `[k, k + s)` for that K-mer. The `l` field is
    // dropped because arm B never extends, it only binary-searches inside the range.
    let table: Vec<(i64, i64)> = level.iter().map(|m| (m.k, m.s)).collect();
    // Unweighted mean SA range over ALL 4^K table entries, in rows. Reported for reference only, and
    // it is the statistic the header warns against: it is dominated by the many rare K-mers, whereas
    // a real query hits an occurrence-weighted range (measured a few blocks below).
    let mean_s: f64 = table.iter().map(|e| e.1 as f64).sum::<f64>() / table.len() as f64;
    println!(
        "built 4^{K} = {} entries in {:.1}s; mean SA range = {:.1} rows -> ~{:.1} probes/search",
        table.len(),
        t.elapsed().as_secs_f64(),
        mean_s,
        2.0 * mean_s.log2()
    );

    // Queries: real 32-mers taken from random genome positions (so every one exists).
    // Synthetic random 32-mers would almost all be absent, and an absent key ends both searches early
    // with an empty interval: that would time the miss path, which no aligner spends its time in.
    // Positions containing a code >= 4 (an N in the reference) are rejected and redrawn, because a
    // query with an N has no SA range and would be a third, meaningless case.
    // Number of queries both arms process. 200k is large enough that the percentile rows below are
    // stable and per-call overhead is amortized, small enough that the query array (200k x 32 B =
    // 6.4 MB) is negligible next to the 4.3 GB table. Both arms use the identical set.
    const NQ: usize = 200_000;
    // xorshift64 state seeded with the golden-ratio constant 2^64/phi. Fixed, so the query set is
    // reproducible run to run; the specific value carries no meaning beyond being nonzero.
    let mut st = 0x9E3779B97F4A7C15u64;
    // The query set: NQ genuine QLEN-mers cut from the reference, each guaranteed N-free.
    let mut queries: Vec<[u8; QLEN]> = Vec::with_capacity(NQ);
    while queries.len() < NQ {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        // Candidate start, a reference POSITION. The modulus reserves QLEN+1 bases of headroom so
        // the whole k-mer lies inside the reference.
        let p = (st % (n as u64 - QLEN as u64 - 1)) as usize;
        // Scratch k-mer, bases encoded 0..=3. Pushed only if `ok` survives the N check.
        let mut q = [0u8; QLEN];
        let mut ok = true;
        for i in 0..QLEN {
            let b = refseq[p + i];
            if b >= 4 {
                ok = false;
                break;
            }
            q[i] = b;
        }
        if ok {
            queries.push(q);
        }
    }
    println!("{NQ} real {QLEN}-mers from the genome\n");

    // The table-average SA range is not what a read hits. Queries are drawn FROM the genome, so a
    // K-mer occurring t times is t times likelier to be sampled: the distribution a real search sees
    // is occurrence-weighted, and repeats dominate it. Measure that before trusting any probe-count
    // model built on the table average.
    // One SA range SIZE (in rows) per query: the number of BWT rows sharing that query's first K
    // bases, i.e. how wide arm B's binary search starts. `idx` is the K-mer packed as a base-4
    // number, 2 bits per base, matching the table's construction order. Sorted ascending in place so
    // the percentile picks below are valid.
    let mut ranges: Vec<i64> = queries
        .iter()
        .map(|q| {
            let mut idx = 0usize;
            for b in 0..K {
                idx = (idx << 2) | q[b] as usize;
            }
            table[idx].1
        })
        .collect();
    ranges.sort_unstable();
    // Percentile picker over the sorted `ranges`: `f` in 0.0..=1.0 (0.5 = median), returns the range
    // size in rows at that quantile. Nearest-rank, no interpolation, which is fine at NQ = 200k.
    let pick = |f: f64| ranges[((ranges.len() - 1) as f64 * f) as usize];
    // Occurrence-weighted mean range size in rows: the average a real read actually faces, as
    // opposed to `mean_s` which averages over distinct K-mers. This is the honest input to the
    // `2 * log2(range)` probe-count model.
    let wmean: f64 = ranges.iter().map(|&r| r as f64).sum::<f64>() / ranges.len() as f64;
    println!("SA range actually hit by a read (occurrence-weighted over {NQ} real {K}-mers):");
    println!(
        "  table average : {:>12.1} rows  -> {:>5.1} probes",
        mean_s,
        2.0 * mean_s.log2()
    );
    println!(
        "  weighted mean : {:>12.1} rows  -> {:>5.1} probes",
        wmean,
        2.0 * wmean.max(1.0).log2()
    );
    println!("  median        : {:>12} rows", pick(0.5));
    println!("  p90 / p99     : {:>12} / {} rows", pick(0.90), pick(0.99));
    println!(
        "  max           : {:>12} rows  -> {:.1} probes",
        ranges[ranges.len() - 1],
        2.0 * (ranges[ranges.len() - 1].max(1) as f64).log2()
    );
    println!();
    // Everything above is cheap and is the part worth having; the timing below needs the flat SA
    // (49.6 GB) resident alongside the table (4.3 GB) and the index, and on a host that cannot hold
    // that it measures swap, not algorithms. Opt in explicitly rather than silently reporting garbage.
    if std::env::var_os("STAR_VS_FM_TIME").is_none() {
        println!(
            "(timing skipped: the flat SA + table + index need ~70 GB and this host thrashes.\n \
                  set STAR_VS_FM_TIME=1 to run it anyway)"
        );
        return;
    }

    // ---- Arm A: FM forward extension, lockstep W, prefetched. 32 dependent rounds.
    //
    // FAIRNESS CAVEAT: each arm is run exactly once, A before B, with no warm-up. A therefore pays the
    // page faults on `cp_occ` and the frequency ramp, and B starts on a warm clock. That biases the
    // result IN FAVOUR of B, so a `B vs A` at or below 1.00x is a safe verdict against STAR, while a
    // narrow win for B is not trustworthy without re-running with the arms swapped.
    let t = now();
    // Arm A's answers, one bi-interval per query, in query order. `k` and `s` are what the final
    // agreement check compares.
    let mut out_a: Vec<Smem> = Vec::with_capacity(NQ);
    for chunk in queries.chunks(W) {
        // The lockstep window's live state: one bi-interval per query in this chunk, all seeded from
        // the query's FIRST base. Invariant at the top of the `j` loop: every `s[i]` is the
        // bi-interval of `chunk[i][0..j]`, so all slots are at the same depth (that is the lockstep).
        let mut s: Vec<Smem> = chunk
            .iter()
            .map(|q| {
                let a = q[0] as usize;
                Smem {
                    rid: 0,
                    m: 0,
                    n: 0,
                    k: c[a],
                    l: c[3 - a],
                    s: c[a + 1] - c[a],
                }
            })
            .collect();
        for j in 1..QLEN {
            // Prefetch on `l`, not `k`: `fwd_ext` swaps the two before calling `backward_ext`, so the
            // blocks the next round actually touches are those under the reverse-strand start.
            for x in &s {
                fm.prefetch_occ(x.l, x.l + x.s);
            }
            for (i, x) in s.iter_mut().enumerate() {
                *x = fwd_ext(&fm, *x, chunk[i][j] as usize);
            }
        }
        out_a.extend_from_slice(&s);
    }
    // Arm A total wall time in SECONDS for all NQ queries (each 32 extensions). Divided by NQ later
    // to get ns/query. The timed region covers the window setup, prefetch and extensions; it
    // excludes query generation and table construction.
    let ta = t.elapsed().as_secs_f64();

    // ---- Arm B: table lookup + lockstep binary search on the flat SA.
    let t = now();
    // Arm B's answers as `(first SA row, row count)`, in query order, directly comparable with arm
    // A's `(k, s)`.
    let mut out_b: Vec<(i64, i64)> = Vec::with_capacity(NQ);
    for chunk in queries.chunks(W) {
        // Live slots in this window. Equals W except possibly for the last chunk. Every `0..m` loop
        // below is over live slots only; entries m..W of the fixed-size arrays are untouched padding.
        let m = chunk.len();
        // Table gives the SA range of the first K bases in ONE access.
        // Half-open SA ROW range per slot. Initially the K-mer's range straight from the table; after
        // the two bound passes, `lo[i]` is the first row matching the full QLEN-mer and `hi[i]` is
        // one past the last. Fixed W-sized arrays so they live in registers/stack, never the heap.
        let mut lo = [0i64; W];
        let mut hi = [0i64; W];
        for i in 0..m {
            let mut idx = 0usize;
            for b in 0..K {
                idx = (idx << 2) | chunk[i][b] as usize;
            }
            let (k, s) = table[idx];
            lo[i] = k;
            hi[i] = k + s;
        }
        // Two lockstep binary searches (lower then upper bound) over the remaining QLEN-K bases.
        // Two passes rather than one because the answer is a RANGE: the SA rows matching the full
        // 32-mer form a contiguous sub-range of the K-mer's range, and locating both of its ends is
        // what makes arm B's cost ~2*log2(range) probes in the model at the top of this file.
        //
        // Lockstep, like arm A: the loop runs until EVERY slot in the window has converged (`any`),
        // so slots that finish early idle. That wastes work but keeps the memory pipeline the thing
        // being measured, which is the entire point of the comparison.
        // `bound == 0` locates the LOWER bound of the matching sub-range, `bound == 1` the upper.
        // The only difference is the tie rule at the last compared base, see below.
        for bound in 0..2 {
            // Per-pass working copies of the search window (copied, since `lo`/`hi` are arrays), so
            // pass 1 restarts from the K-mer range rather than from pass 0's narrowed one.
            // Invariant at the top of each `loop` iteration: for every live slot, the sought boundary
            // row lies in `[l[i], h[i])`, and the window halves each iteration.
            let (mut l, mut h) = (lo, hi);
            loop {
                // True if at least one slot still has a non-empty window. When it stays false the
                // whole lockstep window has converged and the search ends.
                let mut any = false;
                // Probe row per slot: the SA ROW this iteration will test. Stale for converged slots,
                // which is why every consumer re-tests `l[i] < h[i]`.
                let mut mid = [0i64; W];
                for i in 0..m {
                    if l[i] < h[i] {
                        mid[i] = (l[i] + h[i]) / 2;
                        any = true;
                        unsafe {
                            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) sa.as_ptr().add(mid[i] as usize), options(nostack, readonly, preserves_flags));
                        }
                    }
                }
                if !any {
                    break;
                }
                // Second, DEPENDENT access: `sa[mid]` had to arrive before the reference address is
                // known. This is the pessimistic (non-co-located) variant from the header: 2 rounds
                // per probe. BWA-MEME's 13-byte pos+key record collapses it to 1, at 81 GB.
                // Reference POSITION of each slot's probe row, i.e. `sa[mid[i]]`. This is the
                // row-space to position-space conversion; everything after it indexes `refseq`.
                let mut pos = [0i64; W];
                for i in 0..m {
                    if l[i] < h[i] {
                        pos[i] = sa[mid[i] as usize];
                        let r = (pos[i] + K as i64) as usize;
                        if r < refseq.len() {
                            unsafe {
                                std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) refseq.as_ptr().add(r), options(nostack, readonly, preserves_flags));
                            }
                        }
                    }
                }
                for i in 0..m {
                    if l[i] >= h[i] {
                        continue;
                    }
                    // Compare the reference suffix at sa[mid] against the query beyond the table's K bases.
                    // Only bases K..QLEN are compared: rows in this range already agree on the first K
                    // by construction of the table.
                    // An off-the-end reference position yields 4, which is greater than any real base,
                    // so a suffix truncated by the end of the reference sorts after every full match.
                    // "The reference suffix at the probe row sorts BEFORE the query", which is the
                    // standard binary-search decision: true moves the lower edge up, false moves the
                    // upper edge down.
                    let mut less = false;
                    for b in K..QLEN {
                        // `rp`: reference POSITION of the b-th base of the probed suffix.
                        // `rb`: that base, 0..=3, or the sentinel 4 when the suffix runs off the end.
                        let rp = pos[i] + b as i64;
                        let rb = if (rp as usize) < refseq.len() {
                            refseq[rp as usize]
                        } else {
                            4
                        };
                        if rb != chunk[i][b] {
                            less = rb < chunk[i][b];
                            break;
                        }
                        if b == QLEN - 1 {
                            less = bound == 1;
                        } // ties go left for lower, right for upper
                    }
                    if less {
                        l[i] = mid[i] + 1;
                    } else {
                        h[i] = mid[i];
                    }
                }
            }
            // On convergence `l[i] == h[i]` is the boundary row. Pass 0 writes it back as the new
            // `lo` (first matching row), pass 1 as the new `hi` (one past the last).
            if bound == 0 {
                for i in 0..m {
                    lo[i] = l[i];
                }
            } else {
                for i in 0..m {
                    hi[i] = l[i];
                }
            }
        }
        for i in 0..m {
            out_b.push((lo[i], hi[i] - lo[i]));
        }
    }
    // Arm B total wall time in SECONDS for all NQ queries. The timed region covers the table lookups,
    // both binary-search passes and the suffix comparisons. It excludes building the table, which is
    // arm B's one-off setup and is reported separately.
    let tb = t.elapsed().as_secs_f64();

    // ---- Both must produce the same SA interval.
    // Number of queries where the two arms disagree on the SA interval. MUST be 0: arm B reaching a
    // different interval means its search is wrong, and a wrong search is usually a faster one, so a
    // nonzero count invalidates the timing above rather than merely flagging a bug.
    let mut bad = 0;
    for i in 0..NQ {
        if out_a[i].k != out_b[i].0 || out_a[i].s != out_b[i].1 {
            if bad < 3 {
                eprintln!(
                    "  MISMATCH q{i}: FM (k={}, s={}) vs STAR (k={}, s={})",
                    out_a[i].k, out_a[i].s, out_b[i].0, out_b[i].1
                );
            }
            bad += 1;
        }
    }
    println!("{:<34} {:>9} {:>14}", "", "wall(s)", "ns/query");
    println!(
        "{:<34} {:>9.3} {:>14.0}",
        "A: FM fwd extension (lockstep W=16)",
        ta,
        ta * 1e9 / NQ as f64
    );
    println!(
        "{:<34} {:>9.3} {:>14.0}",
        "B: STAR table + binary search",
        tb,
        tb * 1e9 / NQ as f64
    );
    println!("\nB vs A: {:.2}x", ta / tb);
    println!(
        "agreement: {} / {} queries {}",
        NQ - bad,
        NQ,
        if bad == 0 {
            "IDENTICAL"
        } else {
            "*** MISMATCHED -- arm B is wrong, the timing is meaningless ***"
        }
    );
}
