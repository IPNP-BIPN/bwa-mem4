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
//! Reading the output. One block per batch size N, one row per bin width:
//!   - `bin rows`   : 2^shift BWT rows per bin (the sort key is `pos >> shift`).
//!   - `bins`       : how many bins the whole BWT splits into at that width.
//!   - `pages/bin`  : 2^shift bytes / 16 KB. The L2 dTLB holds ~3072 entries, so a bin whose page
//!                    count exceeds that cannot be TLB-resident and the whole premise fails.
//!   - `acc/bin`    : N / bins, the expected accesses per bin. THIS is the decisive column. Zhang's
//!                    win needs many accesses per page; `acc/bin` below `pages/bin` means fewer than
//!                    one access per page, i.e. binning has bought literally no reuse and only added
//!                    the sort.
//!   - `random(ns)` / `binned(ns)` : best of 3 reps, ns per extension. `binned` includes its sort.
//!   - `speedup`    : random / binned. Above 1.00x binning pays.
//!
//! What the answer looks like. Zhang reports 1.43-1.54x. A speedup below ~1.05x here (this host's
//! noise floor is ~2.4%, so read only the trend across N, never one cell) means the knee has not been
//! reached at that N. The verdict this example exists to deliver is *where the knee is*: if it sits
//! above N = 2^24 then binning cannot be built here, because our lockstep window carries a ~3.6 KB
//! `prev[]` per slot and 2^24 slots is far past what RAM allows.
//!
//!   cargo run --release -p bwa-index --example bwt_binning -- work/genome.fa
//!
//! Argument is the index prefix; only `<prefix>.bwt.2bit.64` and `<prefix>.0123` are read. Needs a
//! real genome: on a small index the whole `cp_occ` fits in cache and every row reads ~1.00x.
use bwa_index::{FmIndex, Smem};
use std::time::Instant;

/// Lockstep window width: how many independent backward extensions are prefetched together and then
/// consumed together. 16 is our production seeding width, chosen because it is roughly the number of
/// outstanding misses an M-series core sustains. Changing it changes what is being measured (both
/// arms would move), so it is fixed here to keep the comparison about binning and nothing else.
const W: usize = 16;

/// One measured pass: `pos.len()` backward extensions, run in lockstep windows of W with the whole
/// window prefetched before any of it is consumed. This is our production seeding shape, so whatever
/// binning does here is what it would do in the aligner.
///
/// The `Smem` is synthetic, not a real seeding interval: `k = p`, `s = 1` makes `backward_ext` read
/// the checkpoint blocks at `p >> 6` and `(p+1) >> 6`, i.e. one random `cp_occ` line per call. That
/// is deliberate. The experiment is about WHERE in the BWT the accesses land, so each call is reduced
/// to exactly one address drawn from the distribution under test, with no interval-size confound.
///
/// The return value exists to (a) stop the optimizer deleting the loop and (b) serve as the
/// equivalence check between arms: `wrapping_add` over a multiset is order-independent, so the random
/// and binned arms must produce the identical accumulator despite visiting `pos` in different orders.
///
/// # Parameters
/// - `fm`: the loaded index. Only `backward_ext` and `prefetch_occ` are used, both read-only, so the
///   function is pure apart from cache state.
/// - `pos`: the workload, one BWT ROW index per extension (not a reference position). Every element
///   must be in `0..fm.ref_seq_len`; the caller guarantees this by generating `pos % n`. Length is
///   the batch size N and may be any value, including one not divisible by `W` (the final chunk is
///   simply short). The two arms pass the same multiset here, permuted differently.
/// - `c`: the FM index's C array, `c[a]` = number of reference bases lexicographically smaller than
///   base `a`, with `c[4]` the total. Obtained from `FmIndex::counts()`. Only `c[0]` is read, as a
///   syntactically valid `l` for the synthetic interval; its value never affects the address touched.
///
/// # Returns
/// An order-independent checksum of the extensions performed (`wrapping_add` of every resulting `k`).
/// Meaningless as a number; used only to compare arms and to keep the loop alive.
fn pass(fm: &FmIndex, pos: &[i64], c: &[i64; 5]) -> i64 {
    // Order-independent checksum over the whole batch, in "BWT row index" units summed modulo 2^64.
    // Invariant at the top of each chunk iteration: `acc` holds the wrapping sum of the result `k`
    // of every extension for the rows already consumed.
    let mut acc = 0i64;
    for chunk in pos.chunks(W) {
        for &p in chunk {
            fm.prefetch_occ(p, p + 1);
        }
        for &p in chunk {
            // Synthetic single-row interval anchored at BWT row `p`: `s = 1` means it spans one row,
            // so the extension reads exactly the checkpoint blocks around `p`. `rid`/`m`/`n` are the
            // read id and query span, unused here and left 0.
            let s = Smem {
                rid: 0,
                m: 0,
                n: 0,
                k: p,
                l: c[0],
                s: 1,
            };
            // The extending base is derived from `p` itself (`p & 3`, so 0..3 = A/C/G/T) rather than
            // drawn separately: it keeps the pass deterministic and identical between arms, since
            // the base travels with the row through the sort.
            acc = acc.wrapping_add(fm.backward_ext(s, (p & 3) as usize).k);
        }
    }
    acc
}

/// Counting-sort `pos` by BWT bin. Returns the reordered vector; cost belongs to arm B.
///
/// Counting sort, not a comparison sort, because the bin index is a plain `pos >> shift` and the
/// sort must be cheap enough that binning can win: a sort costing more than the misses it saves
/// would answer the wrong question. Note this is a permutation of `pos`, so arm B does exactly the
/// same extensions as arm A, only reordered.
///
/// # Parameters
/// - `pos`: BWT row indices to reorder, each in `0..ref_seq_len`. Length is the batch size N and
///   must fit in `u32` because the counting-sort offsets are `u32` (N up to 2^24 here, fine).
/// - `shift`: log2 of the bin width in BWT ROWS, so the bin of a row is `pos >> shift`. Supplied by
///   the sweep as 20 / 24 / 28. Larger shift = fewer, wider bins = less locality but a cheaper sort.
/// - `nbins`: number of bins, which the caller computes as `(n >> shift) + 1`. It MUST be consistent
///   with `shift` and with the maximum element of `pos`, otherwise the histogram indexing panics.
///
/// # Returns
/// A permutation of `pos`, stable within a bin and ascending by bin index. Same length, same
/// multiset, so arm B provably performs the same extensions as arm A.
fn bin_sort(pos: &[i64], shift: u32, nbins: usize) -> Vec<i64> {
    // Histogram, offset by one slot: after the two loops below `cnt[b]` is the index in `out` where
    // bin `b` begins (the classic counting-sort prefix-sum layout, which is why it has nbins+1 slots).
    let mut cnt = vec![0u32; nbins + 1];
    for &p in pos {
        cnt[(p >> shift) as usize + 1] += 1;
    }
    // Prefix sum: turns per-bin counts into per-bin start offsets.
    for i in 0..nbins {
        cnt[i + 1] += cnt[i];
    }
    // Scatter destination. During the loop `cnt[b]` is the write cursor for bin `b` and advances as
    // rows are placed, so on exit it has been consumed up to the start of bin b+1.
    let mut out = vec![0i64; pos.len()];
    for &p in pos {
        let b = (p >> shift) as usize;
        out[cnt[b] as usize] = p;
        cnt[b] += 1;
    }
    out
}

fn main() {
    // argv[1]: index prefix (path the index side-files sit next to). No default; operator-supplied.
    let prefix = std::env::args()
        .nth(1)
        .expect("usage: bwt_binning <index prefix>");
    let fm = FmIndex::load(std::path::Path::new(&prefix)).expect("load index");
    // C array: c[a] = count of reference bases lexicographically below base `a`, c[4] = total.
    // Passed through to `pass` purely to supply a well-formed `l` field.
    let c = fm.counts();
    // FMD reference length in bases (forward strand + reverse complement), equal to the number of
    // BWT rows. The modulus for random row generation and the divisor for the bin count.
    let n = fm.ref_seq_len;
    // Size of the checkpointed occurrence table in BYTES. The layout is one 64-byte cache line per
    // 64 BWT rows, hence `(n >> 6) + 1` blocks of 64 B. Reported so the reader can see how the bin
    // widths below compare with the TLB's reach.
    let cp_bytes = ((n >> 6) + 1) as f64 * 64.0;
    println!(
        "cp_occ spans {:.1} GB = {:.0}k pages of 16 KB (L2 dTLB holds ~3072)\n",
        cp_bytes / 1073741824.0,
        cp_bytes / 16384.0 / 1000.0
    );

    // 2^12 .. 2^24: from roughly our lockstep window scale up to Zhang's own batch size (2^24
    // computations), so the sweep brackets the published result rather than sampling one point.
    for logn in [12u32, 16, 20, 24] {
        // Batch size N for this block: how many extensions one "round" performs. This is the axis the
        // whole example sweeps, because binning's payoff depends on accesses per page and that is
        // N / bins. 2^24 is the top because it is Zhang's own batch size.
        let nq = 1usize << logn;
        // Uniform random BWT rows. Real SMEM intervals are not uniform, but they are close enough to
        // it that this is the honest pessimistic case for binning: any clustering a real workload has
        // would only help arm B, so a null result here is a null result there too.
        // xorshift64 state, seeded with the fractional bits of pi. Fixed and re-seeded identically
        // for every N, so the smaller batches are prefixes-in-distribution of the larger ones and
        // the trend across N is not confounded by a different random stream.
        let mut st = 0x243F6A8885A308D3u64;
        // The workload: `nq` BWT ROW indices drawn uniformly from 0..n. Not reference positions.
        // Generated outside every timed region.
        let pos: Vec<i64> = (0..nq)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                (st % (n as u64)) as i64
            })
            .collect();

        println!("--- batch N = {nq} ---");
        println!(
            "  {:>8} {:>7} {:>9} {:>10} {:>11} {:>11} {:>8}",
            "bin rows", "bins", "pages/bin", "acc/bin", "random(ns)", "binned(ns)", "speedup"
        );
        // Bin widths 2^20 / 2^24 / 2^28 rows = 1 MB / 16 MB / 256 MB of cp_occ: respectively well
        // inside the TLB's reach, around it, and hopelessly past it. Three points is enough to see
        // whether the trend has a knee at all.
        for shift in [20u32, 24, 28] {
            // Number of bins the BWT's n rows split into at this width. `+1` because bin indices run
            // 0..=(n >> shift) inclusive, so the final partial bin needs a slot.
            let nbins = ((n >> shift) + 1) as usize;
            // A bin of 2^shift BWT rows covers 2^shift/64 cp_occ blocks x 64 B = 2^shift BYTES.
            // Bytes of cp_occ one bin spans (2^shift, by the identity in the line above).
            let bin_bytes = (1u64 << shift) as f64;
            // Same figure in 16 KB pages, the unit the TLB counts in. Compare with ~3072 L2 dTLB
            // entries: above that a bin cannot be TLB-resident and Zhang's mechanism cannot fire.
            let pages_per_bin = bin_bytes / 16384.0;
            // Expected number of accesses landing in one bin, N / bins. The decisive column: reuse
            // requires this to exceed `pages_per_bin`, otherwise binning buys under one access per
            // page and has only added the sort's cost.
            let acc_per_bin = nq as f64 / nbins as f64;

            // Arm B's workload: the same rows, bin-ordered. Built here only for the warm-up and the
            // equivalence assert; the timed reps below re-sort so that arm B is charged for it.
            let binned = bin_sort(&pos, shift, nbins);
            // Warm BOTH arms fully before timing either: the first touch of these positions faults
            // pages and ramps the clock, and it would be charged entirely to whichever arm ran first.
            // Warm-up checksums, discarded as timings but compared: `w1` (random order) and `w2`
            // (bin order) must be equal, which is the proof that the sort is a pure permutation.
            let w1 = pass(&fm, &pos, &c);
            let w2 = pass(&fm, &binned, &c);
            assert_eq!(
                w1, w2,
                "binning changed the result -- the arms are not equivalent"
            );

            // Best of 3, not mean: we want the least-disturbed run. A mean is dragged around by
            // whatever else the OS scheduled, and the question here is about the hardware, not the
            // host's mood. The two arms alternate inside each rep so slow drift hits both equally.
            // Running minima over the 3 reps, in ns per extension: `best_r` for the random arm,
            // `best_b` for the binned arm. Seeded at +inf so the first rep always wins. Invariant at
            // the top of each rep: each holds the fastest per-extension cost seen so far for its arm.
            let (mut best_r, mut best_b) = (f64::MAX, f64::MAX);
            for _ in 0..3 {
                // Arm A timed region: `nq` extensions in random row order, nothing else.
                // `a` is the checksum; `r` is the arm's cost amortized to ns per single extension.
                let t = Instant::now();
                let a = pass(&fm, &pos, &c);
                let r = t.elapsed().as_secs_f64() * 1e9 / nq as f64;
                // Arm B timed region: the counting sort PLUS the same `nq` extensions. The sort is
                // deliberately inside the timer, since a real implementation could not avoid it.
                let t = Instant::now();
                let bs = bin_sort(&pos, shift, nbins); // arm B pays its own sort, every rep
                let b2 = pass(&fm, &bs, &c);
                let b = t.elapsed().as_secs_f64() * 1e9 / nq as f64;
                // Same multiset extended both ways must give the same order-independent checksum.
                assert_eq!(a, b2);
                best_r = best_r.min(r);
                best_b = best_b.min(b);
            }
            println!(
                "  {:>8} {:>7} {:>9.0} {:>10.2} {:>11.1} {:>11.1} {:>7.2}x",
                format!("2^{shift}"),
                nbins,
                pages_per_bin,
                acc_per_bin,
                best_r,
                best_b,
                best_r / best_b
            );
        }
        println!();
    }
}
