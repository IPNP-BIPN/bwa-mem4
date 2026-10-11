//! Exact mate-rescue window pruning: prove from exact K-mer matches that a rescue job cannot reach
//! the rescue threshold (skip it), or which rows of its window can hold any value that matters
//! (run the DP on that sub-window only). Port of fg-labs/bwa-mem3's `rescue_prune.h` (PRs #533,
//! #541, #548), scalar filter only.
//!
//! # What a rescue job contributes
//!
//! `matesw_apply` (bwa-mem's `mem_matesw`) reads a job's `score`, `qb`, `qe`, `tb`, `te` and
//! `score2`, and only when `score >= minsc` (`minsc = min_seed_len * a`; below it the reverse
//! start-recovery pass does not run, `qb` stays -1 and the hit is dropped). So a job whose score is
//! provably `< minsc` can be replaced by the empty result, and a job can run on any sub-window that
//! reproduces every cell whose `H` is `>= minsc`: those cells alone decide `score`, `te` (first row
//! reaching the max), `qe` (leftmost column of the max in that row), the 2nd-best list (rows whose
//! max is `>= minsc`) and the reverse pass (which walks back from `te` and stops inside the
//! alignment it is recovering).
//!
//! # The lemma
//!
//! With match `a`, mismatch `-b` and gaps costing `o + e*L`, pick `K` with `(K-1)*a` at most the
//! cheapest separator (a mismatch, a deletion, an insertion). A run of `r` matches holds
//! `max(0, r-K+1)` K-mer hits, so its score is at most `a*(K-1) + a*hits`, and every separator
//! between runs costs at least `a*(K-1)`, so it cancels the next run's free `a*(K-1)`. What is left
//! of each gap is at least `c` per diagonal step, `c = min(e, o + e - a*(K-1))` over both gap types.
//! An alignment spanning diagonals `[d1, d2]` therefore scores at most
//!
//! ```text
//!   a*(K-1) + c + sum_{d in [d1, d2]} (a*cnt_d - c)
//! ```
//!
//! with `cnt_d` the number of K-mer hits on diagonal `d = i - j` (row minus column). The best such
//! interval is a maximum-subarray (Kadane) over the diagonals. If no interval reaches `minsc`, the
//! job fails (`Fail`). Otherwise only the diagonal components whose best interval reaches `minsc`
//! can hold a cell `>= minsc`, and in each of them:
//!
//! - the best alignment ending at such a cell can be cut to start at its first K-mer hit (the
//!   prefix before it nets `<= 0`), so starting the DP at the component's first hit row keeps it;
//! - its tail past the last hit diagonal loses at least `c` per step, so its end row is bounded by
//!   `last hit diagonal + qpad - 1 + tail`, where `qpad` is the query padded to the kernel's
//!   profile width: ksw scores the pad columns 0, so a value keeps sliding down the diagonal through
//!   them and can still be a later row's max (hence `score2`).
//!
//! The union of those row ranges is the `Rows` decision. At bwa's defaults (`-A1 -B4 -O6 -E1`,
//! `-k 19`) that is `K = 5`, `c = 1`, bound `5 + sum (cnt_d - 1)`.
//!
//! One deliberate difference from the C: the fork distinguishes the deletion and insertion costs
//! in the validity test and the tail. Here both are taken at the cheaper of the two gap types, so
//! the bound holds whichever axis a gap type is charged to. That can only keep more rows (or prune
//! fewer jobs) at asymmetric `-O`/`-E`, never change a result; at symmetric gap costs (the
//! defaults) it is the fork's bound exactly.
//!
//! An `N` (code 4) in the mate or the window breaks the 2-bit K-mer encoding and scores -1, which
//! the lemma does not model, so any `N` returns `Full`.

/// Largest K the filter takes. 4^8 = 65536 codes, so the per-thread code tables stay small.
pub const KMAX: i32 = 8;

/// The bound's constants for one scoring and threshold, or `valid = false` when the lemma does not
/// hold (then every window runs in full).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PruneParams {
    /// K-mer length.
    pub k: i32,
    /// Match score.
    pub a: i32,
    /// Per-diagonal charge: what each diagonal step of a gap costs beyond the separator.
    pub c: i32,
    /// The cheaper gap open, and the cheaper gap extend, over both gap types (the tail bound).
    pub o_min: i32,
    pub e_min: i32,
    /// The rescue threshold, `min_seed_len * a`.
    pub minsc: i32,
    pub valid: bool,
}

impl PruneParams {
    /// Derive the bound for scoring matrix `mat` (5x5, row-major), gap costs and threshold.
    ///
    /// `mat` must be bwa's uniform DNA matrix (`a` on the 4x4 diagonal, `-b` off it); anything else
    /// is refused, because the lemma reasons about two cell scores only. K is the largest value in
    /// `k_min..=k_max` (capped at [`KMAX`]) that the separator costs admit and for which the charge
    /// is positive and a hit-free window cannot reach `minsc`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        minsc: i32,
        k_min: i32,
        k_max: i32,
    ) -> Self {
        let mut p = PruneParams {
            k: 5,
            a: 1,
            c: 1,
            o_min: 0,
            e_min: 1,
            minsc,
            valid: false,
        };
        if mat.len() < 25 {
            return p;
        }
        let a = i32::from(mat[0]);
        let b = -i32::from(mat[1]);
        for t in 0..4 {
            for q in 0..4 {
                let want = if t == q { a } else { -b };
                if i32::from(mat[t * 5 + q]) != want {
                    return p;
                }
            }
        }
        if a < 1 || b < 1 || e_del < 1 || e_ins < 1 || o_del < 0 || o_ins < 0 {
            return p;
        }
        let (o_min, e_min) = (o_del.min(o_ins), e_del.min(e_ins));
        // The cheapest separator of each kind: a mismatch `b`, and a one-base gap `o + e`. The
        // fork also requires the deletion open alone to reach the separator cost; taking it on
        // the cheaper open keeps that condition whichever axis is the deletion.
        let sep = b.min(o_min).min(o_del + e_del).min(o_ins + e_ins);
        let km1 = (k_max.min(KMAX) - 1).min(sep / a);
        let mut k = km1 + 1;
        while k >= k_min.max(3) {
            let s = a * (k - 1);
            if sep >= s {
                let c = e_min.min(o_del + e_del - s).min(o_ins + e_ins - s);
                if c > 0 && minsc > s {
                    p.k = k;
                    p.a = a;
                    p.c = c;
                    p.o_min = o_min;
                    p.e_min = e_min;
                    p.valid = true;
                    return p;
                }
            }
            k -= 1;
        }
        p
    }

    /// `a*(K-1) + c`: the bound's constant term.
    fn base(&self) -> i32 {
        self.a * (self.k - 1) + self.c
    }

    /// One diagonal's contribution to an interval's bound.
    fn weight(&self, cnt: u32) -> i32 {
        self.a * cnt as i32 - self.c
    }

    /// Rows past the last hit diagonal's end that an alignment whose interval bound is `ub` can
    /// still reach at score `>= tau`: each further row is a deletion step costing at least
    /// `e_min`, after an open the separator accounting has not already paid for.
    fn tail(&self, ub: i32, tau: i32) -> i32 {
        let n = ub - tau - self.o_min + (self.k - 1) * self.a;
        if n <= 0 {
            0
        } else {
            n / self.e_min
        }
    }
}

/// What the filter proved about one rescue job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prune {
    /// Nothing: run the whole window.
    Full,
    /// The job's best local score is `< minsc`: its outputs are never read.
    Fail,
    /// Only target rows `hb..=he` (inclusive) can hold a cell `>= minsc`. A DP on that sub-window,
    /// with `tb`/`te`/`te2` shifted back by `hb`, reproduces every consumed field.
    Rows(usize, usize),
}

/// Per-thread scratch. The query tables depend only on the oriented mate, which repeats across the
/// anchors rescued with it, so they are cached by content.
#[derive(Default)]
pub struct PruneScratch {
    /// `head[code]`: last query position whose K-mer has this code, or -1.
    head: Vec<i32>,
    /// `qcnt[code]`: how many query positions have this code.
    qcnt: Vec<u32>,
    /// `nxt[j]`: previous query position with the same code as position `j`, or -1.
    nxt: Vec<i32>,
    /// Codes set by the cached query, so a new query clears only those.
    touched: Vec<usize>,
    qcache: Vec<u8>,
    qk: i32,
    /// Per diagonal: hit count, first hit's start row, Kadane sums ending / starting there.
    cnt: Vec<u32>,
    minrow: Vec<i32>,
    fwd: Vec<i32>,
    bwd: Vec<i32>,
}

/// Decide how much of rescue window `target` the job against mate `query` must compute.
///
/// `max_hits`: return [`Prune::Full`] when the two share more K-mer hits than this; the filter
/// would then cost more than the rows it can save.
pub fn prune_window(
    target: &[u8],
    query: &[u8],
    p: &PruneParams,
    max_hits: u32,
    s: &mut PruneScratch,
) -> Prune {
    let k = p.k as usize;
    let (len1, len2) = (target.len(), query.len());
    if !p.valid || len1 < k || len2 < k || len2 >= i32::MAX as usize / 2 {
        return Prune::Full;
    }
    if target.iter().chain(query).any(|&x| x > 3) {
        return Prune::Full;
    }
    let mask = (1usize << (2 * k)) - 1;
    if s.qk != p.k || s.qcache.as_slice() != query {
        let ncode = 1usize << (2 * k);
        if s.head.len() != ncode {
            s.head = vec![-1; ncode];
            s.qcnt = vec![0; ncode];
        } else {
            for &c in &s.touched {
                s.head[c] = -1;
                s.qcnt[c] = 0;
            }
        }
        s.touched.clear();
        s.nxt.clear();
        s.nxt.resize(len2, -1);
        let mut c = 0usize;
        for &x in &query[..k - 1] {
            c = (c << 2) | x as usize;
        }
        for j in k - 1..len2 {
            c = ((c << 2) | query[j] as usize) & mask;
            if s.qcnt[c] == 0 {
                s.touched.push(c);
            }
            s.nxt[j] = s.head[c];
            s.head[c] = j as i32;
            s.qcnt[c] += 1;
        }
        s.qcache.clear();
        s.qcache.extend_from_slice(query);
        s.qk = p.k;
    }
    let mut c0 = 0usize;
    for &x in &target[..k - 1] {
        c0 = (c0 << 2) | x as usize;
    }
    // Total hits from the per-code query counts, before enumerating any of them.
    let mut c = c0;
    let mut nhits = 0u32;
    for &x in &target[k - 1..] {
        c = ((c << 2) | x as usize) & mask;
        nhits += s.qcnt[c];
    }
    if nhits > max_hits {
        return Prune::Full;
    }
    // The rescue kernels pad the query to a whole number of profile vectors (16 columns at u8, 8 at
    // i16, `matesw::ksw_padded_qlen`) and score the pad columns 0, so a value can slide diagonally
    // through them into later rows, where it still counts toward a row max and so toward `score2`.
    // The end-row bound must therefore run to the padded width; the u8 quantum covers both.
    let quanta = len2.div_ceil(16) * 16;
    // Diagonal `d = i - j` in [-(quanta-1), len1-1], stored at index `d + off`.
    let off = quanta as i32;
    let nd = len1 + quanta + 1;
    s.cnt.clear();
    s.cnt.resize(nd, 0);
    s.minrow.resize(nd, 0);
    s.fwd.resize(nd, 0);
    s.bwd.resize(nd, 0);
    c = c0;
    for i in k - 1..len1 {
        c = ((c << 2) | target[i] as usize) & mask;
        let mut j = s.head[c];
        while j >= 0 {
            let d = (i as i32 - j + off) as usize;
            if s.cnt[d] == 0 {
                s.minrow[d] = (i + 1 - k) as i32;
            }
            s.cnt[d] += 1;
            j = s.nxt[j as usize];
        }
    }
    let base = p.base();
    let (mut best, mut f) = (-p.c, 0i32);
    for d in 0..nd {
        f = p.weight(s.cnt[d]) + f.max(0);
        s.fwd[d] = f;
        best = best.max(f);
    }
    if base + best < p.minsc {
        return Prune::Fail;
    }
    let mut b = 0i32;
    for d in (0..nd).rev() {
        b = p.weight(s.cnt[d]) + b.max(0);
        s.bwd[d] = b;
    }
    // `bnd(d)`: the best interval bound through diagonal `d`.
    let bnd = |s: &PruneScratch, d: usize| base + s.fwd[d] + s.bwd[d] - p.weight(s.cnt[d]);
    let (mut lo, mut hi) = (len1 as i32, -1i32);
    let mut d = 0usize;
    while d < nd {
        if bnd(s, d) < p.minsc {
            d += 1;
            continue;
        }
        let (mut ub, mut i0, mut dmax) = (0i32, len1 as i32, -1i32);
        while d < nd {
            let v = bnd(s, d);
            if v < p.minsc {
                break;
            }
            ub = ub.max(v);
            if s.cnt[d] > 0 {
                i0 = i0.min(s.minrow[d]);
                dmax = d as i32;
            }
            d += 1;
        }
        if dmax < 0 {
            continue;
        }
        lo = lo.min(i0);
        hi = hi.max((dmax - off) + quanta as i32 - 1 + p.tail(ub, p.minsc));
    }
    if hi < 0 {
        return Prune::Fail;
    }
    let hb = lo.max(0) as usize;
    let he = (hi as usize).min(len1 - 1);
    if hb == 0 && he == len1 - 1 {
        return Prune::Full;
    }
    Prune::Rows(hb, he)
}

/// Whether rescue pruning runs: opt-in, `BWA4_RESCUE_PRUNE=1`. Output is identical either way, but
/// with the scalar filter it is a net loss on the data measured so far (chr21, 1M wgsim pairs, M4
/// Max: +8-10 s user CPU against ~6 s of kernel saved, because 94% of window rows survive there;
/// whole hg38, 1M wgsim pairs: 6.9% of jobs proven failing, 82.7% of rows kept, still +8 s user CPU
/// and +0.4-0.6 s wall at -t16). A NEON filter, or banding the kept rows, would be needed to win.
/// Read once, cached.
#[must_use]
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_PRUNE").is_some_and(|v| v != "0"))
}

/// The hit gate: above this many K-mer hits the filter is not worth its cost and the window runs in
/// full. `BWA4_RESCUE_PRUNE_MAX_HITS` overrides the default (1000, the fork's aarch64 value).
#[must_use]
pub fn max_hits() -> u32 {
    use std::sync::OnceLock;
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BWA4_RESCUE_PRUNE_MAX_HITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000)
    })
}

/// `BWA4_MATESW_TIME=1` counters: how the filter decided, and how many window rows the kernel was
/// left with.
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static FULL: AtomicU64 = AtomicU64::new(0);
    pub static FAIL: AtomicU64 = AtomicU64::new(0);
    pub static ROWS: AtomicU64 = AtomicU64::new(0);
    /// Window rows of every job seen, and of what was handed to the kernel.
    pub static ROWS_IN: AtomicU64 = AtomicU64::new(0);
    pub static ROWS_KEPT: AtomicU64 = AtomicU64::new(0);

    /// Print the counters, if any job was seen.
    pub fn dump() {
        let (full, fail, rows) = (FULL.load(Relaxed), FAIL.load(Relaxed), ROWS.load(Relaxed));
        let n = full + fail + rows;
        if n == 0 {
            return;
        }
        let pct = |x: u64, d: u64| 100.0 * x as f64 / d.max(1) as f64;
        let (rin, rkept) = (ROWS_IN.load(Relaxed), ROWS_KEPT.load(Relaxed));
        eprintln!(
            "[rescue-prune] {n} jobs: {fail} proven failing ({:.1}%), {rows} narrowed ({:.1}%), {full} full ({:.1}%); window rows kept {:.1}%",
            pct(fail, n),
            pct(rows, n),
            pct(full, n),
            pct(rkept, rin)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::{ksw_align2, KswAlignResult};

    fn mat(a: i8, b: i8) -> Vec<i8> {
        let mut m = vec![-1i8; 25];
        for t in 0..4 {
            for q in 0..4 {
                m[t * 5 + q] = if t == q { a } else { -b };
            }
        }
        m
    }

    /// The fields `matesw_apply` reads, when it reads them.
    fn consumed(r: &KswAlignResult, minsc: i32) -> Option<(i32, i32, i32, i32, i32, i32)> {
        (r.score >= minsc && r.qb >= 0).then_some((r.score, r.qb, r.qe, r.tb, r.te, r.score2))
    }

    /// Every decision reproduces the full-window `ksw_align2` on the consumed fields, across
    /// scorings, thresholds and windows built to sit near the threshold (planted copies of the mate
    /// at varying divergence, a weaker second copy for `score2`, tandem repeats, poly-A).
    #[test]
    fn decisions_match_full_window() {
        let mut st = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as u32
        };
        let scorings: [(i8, i8, i32, i32, i32, i32); 7] = [
            (1, 4, 6, 1, 6, 1),
            (1, 6, 6, 1, 6, 1),
            (1, 4, 8, 2, 8, 2),
            (2, 8, 12, 2, 12, 2),
            (1, 4, 6, 1, 9, 2),
            (1, 5, 9, 1, 6, 3),
            (1, 3, 5, 1, 5, 1),
        ];
        let (mut n_fail, mut n_rows, mut n_full) = (0, 0, 0);
        let mut scratch = PruneScratch::default();
        // `PRUNE_ROUNDS` raises the count for a longer soak.
        let rounds = std::env::var("PRUNE_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20000u32);
        for round in 0..rounds {
            let (a, b, od, ed, oi, ei) = scorings[(next() % 7) as usize];
            let m = mat(a, b);
            let minsc = [5, 10, 19, 19, 19, 25, 32][(next() % 7) as usize] * i32::from(a);
            let p = PruneParams::new(&m, od, ed, oi, ei, minsc, 5, 8);
            let qlen = 20 + (next() % 140) as usize;
            let q: Vec<u8> = match next() % 6 {
                0 => (0..qlen).map(|j| [0u8, 1, 0, 2][j % 4]).collect(),
                1 => (0..qlen).map(|j| if j % 17 == 0 { 2 } else { 0 }).collect(),
                _ => (0..qlen).map(|_| (next() % 4) as u8).collect(),
            };
            let tlen = qlen + (next() % 900) as usize;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            // Plant up to two mutated copies (or fragments) of the mate.
            for _ in 0..(next() % 3) {
                let div = next() % 30;
                let (qb, qe) = {
                    let x = (next() as usize) % qlen;
                    let y = (next() as usize) % qlen;
                    (x.min(y), x.max(y) + 1)
                };
                let mut pos = (next() as usize) % tlen;
                for &base in &q[qb..qe] {
                    if pos >= tlen {
                        break;
                    }
                    let r = next() % 100;
                    if r < div / 2 {
                        t[pos] = (next() % 4) as u8;
                        pos += 1;
                    } else if r < div * 3 / 4 {
                        pos += 1 + (next() % 3) as usize; // deletion
                        if pos < tlen {
                            t[pos] = base;
                            pos += 1;
                        }
                    } else if r < div {
                        // insertion: skip the base
                    } else {
                        t[pos] = base;
                        pos += 1;
                    }
                }
            }
            if next() % 8 == 0 {
                let x = (next() as usize) % tlen;
                for v in t.iter_mut().skip(x).take(60) {
                    *v = 0;
                }
            }
            let lanes = if qlen as i32 * i32::from(a) < 250 {
                16
            } else {
                8
            };
            let full = ksw_align2(&q, &t, 5, &m, od, ed, oi, ei, minsc, i32::from(a), lanes);
            let want = consumed(&full, minsc);
            match prune_window(&t, &q, &p, u32::MAX, &mut scratch) {
                Prune::Full => n_full += 1,
                Prune::Fail => {
                    n_fail += 1;
                    assert_eq!(want, None, "round {round}: Fail on a passing job {full:?}");
                }
                Prune::Rows(hb, he) => {
                    n_rows += 1;
                    let mut cut = ksw_align2(
                        &q,
                        &t[hb..=he],
                        5,
                        &m,
                        od,
                        ed,
                        oi,
                        ei,
                        minsc,
                        i32::from(a),
                        lanes,
                    );
                    if cut.tb >= 0 {
                        cut.tb += hb as i32;
                    }
                    if cut.te >= 0 {
                        cut.te += hb as i32;
                    }
                    assert_eq!(
                        consumed(&cut, minsc),
                        want,
                        "round {round}: rows {hb}..={he} of {tlen}, a{a} b{b} o{od},{oi} e{ed},{ei} minsc {minsc}\n full {full:?}\n cut {cut:?}\nQ {q:?}\nT {t:?}"
                    );
                }
            }
        }
        // The generator must exercise every decision, or the test proves nothing.
        eprintln!("fail {n_fail} rows {n_rows} full {n_full}");
        assert!(
            n_fail > 1000 && n_rows > 1000 && n_full > 100,
            "{n_fail} {n_rows} {n_full}"
        );
    }

    #[test]
    fn params_at_defaults() {
        let p = PruneParams::new(&mat(1, 4), 6, 1, 6, 1, 19, 5, 8);
        assert!(p.valid);
        assert_eq!((p.k, p.c, p.base()), (5, 1, 5));
        // A threshold the constant term already reaches proves nothing.
        assert!(!PruneParams::new(&mat(1, 4), 6, 1, 6, 1, 4, 5, 8).valid);
        // -B 6 admits K = 7.
        assert_eq!(PruneParams::new(&mat(1, 6), 6, 1, 6, 1, 19, 5, 8).k, 7);
    }
}
