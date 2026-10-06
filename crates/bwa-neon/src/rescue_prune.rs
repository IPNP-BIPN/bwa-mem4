//! Exact pruning of mate-rescue windows from K-mer hits.
//!
//! A rescue job (mate `q` against reference window `t`) contributes to the output only when its
//! score reaches `minsc = min_seed_len * a`: below that, `mem_matesw` drops it (`qb` stays -1 and
//! the `aln.score >= min_seed_len && aln.qb >= 0` gate fails). This filter proves, from the exact
//! K-mer matches between `q` and `t`, either that the score cannot reach `minsc` (the job need not
//! run at all), or which band of target rows can hold a row whose maximum reaches it (the DP can
//! run on that sub-window and reproduce every consumed field).
//!
//! The idea and the lemma are fg-labs/bwa-mem3's (`src/rescue_prune.h`, v0.14.0, MIT, PRs #533 and
//! #541); this is a reimplementation of its scalar filter, not a copy, checked against our own
//! scalar `ksw_align2`.
//!
//! # The bound
//!
//! Default scoring (match 1, mismatch 4, gap open 6, extend 1), no N anywhere. A run of `r`
//! consecutive matches holds `max(0, r - 4)` exact 5-mers, and every separator between runs (a
//! mismatch or a gap) costs at least 4. So a local alignment scores at most
//! `4 + hits - sum over gaps of (2 + |diagonal change|)`. Charging each diagonal an alignment
//! touches 1, an alignment whose diagonals (`d = row - col`) span `[d1, d2]` scores at most
//! `5 + sum_{d1 <= d <= d2} (cnt_d - 1)`, with `cnt_d` the number of 5-mer hits on diagonal `d`.
//! The best interval is a Kadane scan over the diagonals.
//!
//! Other scorings: the same argument with K-mers, where `K - 1 <= min(b, o_del, o_ins + e_ins) / a`,
//! a per-diagonal charge `c = min(e_del, e_ins, o_del + e_del - (K-1)a, o_ins + e_ins - (K-1)a)`,
//! and `base = (K-1)a + c` in place of 5. [`Params::from`] picks the largest valid K and refuses a
//! scoring for which none is (then every window runs in full).
//!
//! # The query-pad columns
//!
//! Our kernels pad the query to [`ksw_padded_qlen`](super) columns that score 0. Pad columns lie
//! after every real one, so a path through them only appends cells worth `<= 0`: they cannot raise
//! a score. They DO keep a row's maximum alive for up to `pad` more rows along a diagonal, which
//! matters for `te2`/`score2`; the hull's end therefore adds the padded length, rounded to 16, which
//! is never less than what either kernel width pads to.
//!
//! # Decisions
//!
//! * [`Decision::Fail`]: no diagonal interval reaches `minsc`, so the score is proven `< minsc`.
//! * [`Decision::Hull`]: rows `hb..=he` contain every row whose maximum can reach `minsc`. A
//!   zero-state DP on `t[hb..=he]` gives the same `score`, `qe`, `score2` and, shifted by `hb`, the
//!   same `te`/`te2`, because every alignment scoring `>= minsc` lies inside it (it starts at or
//!   after the first hit row of its component, the prefix before it netting `<= 0`, and its tail
//!   past the last hit diagonal is bounded by the deletion cost), and cells outside it can only
//!   lower H values that are below `minsc` anyway, which no consumed field reads.
//! * [`Decision::Full`]: nothing proven (an N, a length out of range, an invalid scoring, or more
//!   hits than the gate allows, where the filter would cost more than it saves).

use std::cell::RefCell;

/// Largest K the code tables are sized for: `4^8` entries.
const KMAX: usize = 8;
/// Longest query the filter takes; query positions are stored in `i16`.
const QCAP: usize = 1024;
/// Longest window the filter takes.
const WCAP: usize = 30_000;

/// The bound's scoring parameters, all in score units. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Params {
    k: usize,
    a: i32,
    c: i32,
    o_del: i32,
    e_del: i32,
    minsc: i32,
}

impl Params {
    /// The bound for this scoring and threshold, or `None` when the lemma does not hold for it.
    ///
    /// `K` is the largest value in `5..=KMAX` for which every separator (mismatch `b`, deletion
    /// open `o_del` alone and with its extend, insertion open plus extend) costs at least `(K-1)a`,
    /// the per-diagonal charge `c` is positive (so the bound decays along a hit-free stretch), and a
    /// hit-free alignment, worth at most `(K-1)a`, cannot reach `minsc`.
    pub(crate) fn from(
        a: i32,
        b: i32,
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        minsc: i32,
    ) -> Option<Params> {
        if a < 1 || b < 1 || e_del < 1 || e_ins < 1 || o_del < 0 || o_ins < 0 {
            return None;
        }
        let km1 = (KMAX as i32 - 1).min(b.min(o_del).min(o_ins + e_ins) / a);
        let mut k = km1 + 1;
        while k >= 5 {
            let s = a * (k - 1);
            let ok = b >= s && o_del >= s && o_del + e_del >= s && o_ins + e_ins >= s;
            let c = e_del
                .min(e_ins)
                .min(o_del + e_del - s)
                .min(o_ins + e_ins - s);
            if ok && c > 0 && minsc > s {
                return Some(Params {
                    k: k as usize,
                    a,
                    c,
                    o_del,
                    e_del,
                    minsc,
                });
            }
            k -= 1;
        }
        None
    }

    /// The interval bound's constant term, `(K-1)a + c` (5 at the defaults).
    fn base(&self) -> i32 {
        self.a * (self.k as i32 - 1) + self.c
    }

    /// One diagonal's contribution to an interval bound, `a * cnt - c`.
    fn weight(&self, cnt: u16) -> i32 {
        self.a * cnt as i32 - self.c
    }

    /// Rows past the last hit diagonal's end that an alignment with interval bound `ub` can still
    /// reach at a score `>= tau`: each such row is a deleted base, and the deletion has to be paid
    /// for out of the bound's margin (`ub - tau`) plus the `(K-1)a` slack a trailing run carries.
    fn tail(&self, ub: i32) -> i32 {
        let n = ub - self.minsc - self.o_del + (self.k as i32 - 1) * self.a;
        if n <= 0 {
            0
        } else {
            n / self.e_del
        }
    }
}

/// What a rescue job needs, as proven by [`decide`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Run the whole window.
    Full,
    /// The score is below `minsc`; the job's result is never consumed.
    Fail,
    /// Run only target rows `hb..=he` (inclusive) and add `hb` back to the row outputs.
    Hull { hb: usize, he: usize },
}

/// Per-thread scratch. The query tables depend only on the oriented mate, which repeats across
/// every anchor rescued with it, so they are rebuilt only when the mate or K changes.
struct Scratch {
    /// Last query position (`j`) whose K-mer has this code, or -1; chained through `nxt`.
    head: Vec<i16>,
    /// Number of query K-mers with this code, for the hit gate.
    qcnt: Vec<u16>,
    /// Previous query position with the same K-mer code, or -1.
    nxt: Vec<i16>,
    /// Codes set by the cached query, so a new query clears only those.
    touched: Vec<u32>,
    /// The query the tables describe, its K, and whether it holds an N.
    qcache: Vec<u8>,
    qk: usize,
    q_has_n: bool,
    /// Per diagonal: hit count, and first row of its first hit (valid where `cnt != 0`).
    cnt: Vec<u16>,
    minrow: Vec<i32>,
    /// The current window's K-mer codes ([`window_codes`]).
    codes: Vec<u16>,
    /// One bit per diagonal holding a hit, and those diagonals in ascending order.
    occ_bits: Vec<u64>,
    occ: Vec<u32>,
    /// Kadane sums ending at (`fo`) and starting at (`bo`) each diagonal of `occ`, index-parallel.
    fo: Vec<i32>,
    bo: Vec<i32>,
}

impl Scratch {
    fn new() -> Scratch {
        Scratch {
            head: Vec::new(),
            qcnt: Vec::new(),
            nxt: vec![-1; QCAP],
            touched: Vec::with_capacity(QCAP),
            qcache: Vec::with_capacity(QCAP),
            qk: 0,
            q_has_n: false,
            cnt: Vec::new(),
            minrow: Vec::new(),
            codes: Vec::new(),
            occ_bits: Vec::new(),
            occ: Vec::new(),
            fo: Vec::new(),
            bo: Vec::new(),
        }
    }

    /// Index the query's K-mers, unless the tables already describe this query at this K. A query
    /// holding an N is only flagged: the caller returns `Full` for it.
    fn load_query(&mut self, q: &[u8], k: usize) {
        if self.qk == k && self.qcache == q {
            return;
        }
        self.qcache.clear();
        self.qcache.extend_from_slice(q);
        self.qk = k;
        self.q_has_n = q.iter().fold(0u8, |acc, &b| acc | b) & !3 != 0;
        let ncode = 1usize << (2 * k);
        if self.head.len() < ncode {
            self.head = vec![-1; ncode];
            self.qcnt = vec![0; ncode];
        } else {
            for &c in &self.touched {
                self.head[c as usize] = -1;
                self.qcnt[c as usize] = 0;
            }
        }
        self.touched.clear();
        if self.q_has_n {
            return;
        }
        let mask = (ncode - 1) as u32;
        let mut code = 0u32;
        for &b in &q[..k - 1] {
            code = (code << 2) | b as u32;
        }
        for j in k - 1..q.len() {
            code = ((code << 2) | q[j] as u32) & mask;
            let c = code as usize;
            if self.qcnt[c] == 0 {
                self.touched.push(code);
            }
            self.nxt[j] = self.head[c];
            self.head[c] = j as i16;
            self.qcnt[c] += 1;
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

/// Decide how much of window `t` the rescue DP of mate `q` must see. See the module docs.
///
/// # Parameters
/// - `t`, `q`: target window and oriented mate, bases 0..=3, 4 = N (any N gives `Full`).
/// - `p`: the scoring's bound, from [`Params::from`].
/// - `max_hits`: return `Full` when `t` and `q` share more K-mer hits than this; a dense window
///   costs the filter more than the rows it can save. Changes speed only, never a decision's
///   correctness.
pub(crate) fn decide(t: &[u8], q: &[u8], p: &Params, max_hits: usize) -> Decision {
    let (len1, len2, k) = (t.len(), q.len(), p.k);
    if len1 < k || len2 < k || len1 > WCAP || len2 > QCAP {
        return Decision::Full;
    }
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        match count_hits(s, t, q, p, max_hits) {
            Some(nd) => sparse_decision(s, p, len1, nd),
            None => Decision::Full,
        }
    })
}

/// The hit gate, then the per-diagonal hit counts. Returns the number of diagonal slots, or
/// `None` when the job is not filtered (an N in either sequence, or more hits than `max_hits`).
/// A job proven to fail by the hit total alone comes back with no diagonal occupied.
fn count_hits(s: &mut Scratch, t: &[u8], q: &[u8], p: &Params, max_hits: usize) -> Option<usize> {
    let (len1, len2, k) = (t.len(), q.len(), p.k);
    s.load_query(q, k);
    if s.q_has_n {
        return None;
    }
    // ---- The window's K-mer codes, once, for the gate and the enumeration. Each code is built
    //      from its own K bytes rather than rolled from the previous one, so the loop carries no
    //      dependency and vectorises; `& 3` keeps an N's code in range, and an N sends the job to
    //      `Full` anyway ----
    if t.iter().fold(0u8, |acc, &b| acc | b) & !3 != 0 {
        return None;
    }
    let mut codes = std::mem::take(&mut s.codes);
    match k {
        5 => window_codes::<5>(t, &mut codes),
        6 => window_codes::<6>(t, &mut codes),
        7 => window_codes::<7>(t, &mut codes),
        _ => window_codes::<8>(t, &mut codes),
    }
    let verdict = count_hits_from(s, &codes, len1, len2, p, max_hits);
    s.codes = codes;
    verdict
}

/// `out[i]` = the 2-bit code of `t[i..i + K]`, for every K-mer of `t`.
fn window_codes<const K: usize>(t: &[u8], out: &mut Vec<u16>) {
    out.clear();
    out.extend(t.windows(K).map(|w| {
        let mut c = 0u16;
        for &b in w {
            c = (c << 2) | (b & 3) as u16;
        }
        c
    }));
}

/// [`count_hits`] past the N test, on the window's K-mer codes (`codes[i]` ends at row
/// `i + K - 1`).
fn count_hits_from(
    s: &mut Scratch,
    codes: &[u16],
    len1: usize,
    len2: usize,
    p: &Params,
    max_hits: usize,
) -> Option<usize> {
    let k = p.k;
    // ---- Hit gate, from per-code counts, without enumerating the hits ----
    let nhits: usize = codes.iter().map(|&c| s.qcnt[c as usize] as usize).sum();
    if nhits > max_hits {
        return None;
    }

    // ---- Per-diagonal hit counts. Diagonal `d = i - j` is stored at `d + off`, with `off` the
    //      query rounded up to 16 so the pad columns' diagonals have slots too ----
    let off = len2.div_ceil(16) * 16;
    let nd = len1 + off + 1;
    s.cnt.clear();
    s.cnt.resize(nd, 0);
    s.minrow.resize(nd, 0);
    s.occ_bits.clear();
    s.occ_bits.resize(nd.div_ceil(64), 0);
    // Early fail: every interval is non-empty and sums `a * cnt - c` over its diagonals, so no
    // interval can beat `base + a * nhits - c`. Leaving every diagonal empty says exactly that.
    if p.base() + p.a * nhits as i32 - p.c < p.minsc {
        return Some(nd);
    }
    for (r, &code) in codes.iter().enumerate() {
        let i = r + k - 1;
        let mut j = s.head[code as usize];
        while j >= 0 {
            let d = i + off - j as usize;
            if s.cnt[d] == 0 {
                // Rows ascend, so the first hit seen on a diagonal is its earliest; its K-mer
                // starts K - 1 rows above the row where it ends.
                s.minrow[d] = (i + 1 - k) as i32;
                s.occ_bits[d / 64] |= 1 << (d % 64);
            }
            s.cnt[d] += 1;
            j = s.nxt[j as usize];
        }
    }
    Some(nd)
}

/// The decision from the per-diagonal counts, walking only the diagonals that hold a hit.
///
/// The bound is a Kadane scan over all `nd` diagonals with weight `a * cnt - c`, but between two
/// occupied diagonals every weight is `-c`, so the scan across a gap has a closed form, and so
/// does the bound on each empty diagonal of the gap. Three facts make the occupied diagonals
/// enough, each checked against the dense scan ([`tests::dense_decision`]):
///
/// * a component's best bound is reached on an occupied diagonal: the best interval through an
///   empty diagonal holds an occupied one (an all-empty interval is worth `base - c * len`, below
///   any valid `minsc`), and that interval lies inside the component, so the occupied diagonal's
///   bound is at least as high;
/// * a component without a hit contributes nothing (the dense scan skips it too);
/// * two consecutive occupied diagonals share a component exactly when every empty diagonal
///   between them reaches `minsc`, and the bound across a gap is convex, so its minimum is at an
///   end or next to a breakpoint ([`gap_min`]).
fn sparse_decision(s: &mut Scratch, p: &Params, len1: usize, nd: usize) -> Decision {
    let (base, c, minsc) = (p.base(), p.c, p.minsc);
    s.occ.clear();
    for (w, &bits) in s.occ_bits.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            s.occ.push((w * 64) as u32 + bits.trailing_zeros());
            bits &= bits - 1;
        }
    }
    let n = s.occ.len();
    if n == 0 {
        // Only hit-free intervals, worth at most `base - c = (K-1)a < minsc`.
        return Decision::Fail;
    }
    debug_assert!((*s.occ.last().unwrap() as usize) < nd);
    // ---- Forward Kadane over the occupied diagonals. After `g` empty diagonals a positive sum
    //      `x` has decayed to `max(x - g c, 0)` before it is carried ----
    s.fo.clear();
    s.fo.reserve(n);
    let mut best = i32::MIN;
    let (mut prev_d, mut prev_f) = (0u32, 0i32);
    for (idx, &d) in s.occ.iter().enumerate() {
        let carry = if idx == 0 {
            0
        } else {
            let gap = (d - prev_d - 1) as i32;
            (prev_f.max(0) - gap * c).max(0)
        };
        let f = p.weight(s.cnt[d as usize]) + carry;
        s.fo.push(f);
        best = best.max(f);
        (prev_d, prev_f) = (d, f);
    }
    if base + best < minsc {
        return Decision::Fail;
    }
    // ---- Backward Kadane, the mirror image ----
    s.bo.clear();
    s.bo.resize(n, 0);
    let (mut next_d, mut next_b) = (0u32, 0i32);
    for idx in (0..n).rev() {
        let d = s.occ[idx];
        let carry = if idx == n - 1 {
            0
        } else {
            let gap = (next_d - d - 1) as i32;
            (next_b.max(0) - gap * c).max(0)
        };
        let b = p.weight(s.cnt[d as usize]) + carry;
        s.bo[idx] = b;
        (next_d, next_b) = (d, b);
    }

    // ---- Components, as in the dense scan: runs of diagonals whose bound reaches `minsc` ----
    let (mut lo, mut hi) = (len1 as i64, -1i64);
    // The open component: its best bound, earliest hit row and last hit diagonal.
    let mut open: Option<(i32, i64, i64)> = None;
    let close = |comp: (i32, i64, i64), lo: &mut i64, hi: &mut i64| {
        let (ub, i0, dmax) = comp;
        *lo = (*lo).min(i0);
        // The last hit diagonal `dmax - off` reaches row `(dmax - off) + (off - 1)` at the last pad
        // column (`off` is the padded query length), then `tail` more rows through a deletion.
        *hi = (*hi).max(dmax - 1 + p.tail(ub) as i64);
    };
    for idx in 0..n {
        let d = s.occ[idx];
        let bnd = base + s.fo[idx] + s.bo[idx] - p.weight(s.cnt[d as usize]);
        if bnd < minsc {
            if let Some(comp) = open.take() {
                close(comp, &mut lo, &mut hi);
            }
            continue;
        }
        let (row, dd) = (s.minrow[d as usize] as i64, d as i64);
        open = match open {
            Some((ub, i0, _))
                if gap_min(p, s.fo[idx - 1], s.bo[idx], d - s.occ[idx - 1] - 1) >= minsc =>
            {
                Some((ub.max(bnd), i0.min(row), dd))
            }
            prev => {
                if let Some(comp) = prev {
                    close(comp, &mut lo, &mut hi);
                }
                Some((bnd, row, dd))
            }
        };
    }
    if let Some(comp) = open {
        close(comp, &mut lo, &mut hi);
    }
    if hi < 0 {
        return Decision::Fail;
    }
    Decision::Hull {
        hb: lo.max(0) as usize,
        he: hi.min(len1 as i64 - 1) as usize,
    }
}

/// The smallest interval bound on the `gap` empty diagonals between two occupied ones, the left
/// with forward sum `f_left`, the right with backward sum `b_right`; `i32::MAX` for no gap.
///
/// On the k-th empty diagonal (`1 <= k <= gap`) the forward sum is `max(p - (k-1)c, 0) - c` with
/// `p = max(f_left, 0)`, the backward sum `max(q - (gap-k)c, 0) - c` with `q = max(b_right, 0)`,
/// and the weight `-c`, so the bound is `base - c + max(p - (k-1)c, 0) + max(q - (gap-k)c, 0)`:
/// a convex function of k, whose integer minimum sits at an end of `[1, gap]` or on either side of
/// one of its two breakpoints.
fn gap_min(p: &Params, f_left: i32, b_right: i32, gap: u32) -> i32 {
    if gap == 0 {
        return i32::MAX;
    }
    let (c, g) = (p.c, gap as i32);
    let (pl, qr) = (f_left.max(0), b_right.max(0));
    let at = |k: i32| {
        let k = k.clamp(1, g);
        p.base() - c + (pl - (k - 1) * c).max(0) + (qr - (g - k) * c).max(0)
    };
    let (k1, k2) = (1 + pl / c, g - qr / c);
    [1, g, k1, k1 + 1, k2, k2 - 1]
        .into_iter()
        .map(at)
        .min()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_extend::ksw_align2;

    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut m = vec![0i8; 25];
        for i in 0..5 {
            for j in 0..5 {
                m[i * 5 + j] = if i == 4 || j == 4 {
                    -1
                } else if i == j {
                    a
                } else {
                    -b
                };
            }
        }
        m
    }

    /// The dense Kadane scan over every diagonal, the specification [`sparse_decision`] must
    /// reproduce: the fork's scalar filter, transcribed.
    pub(super) fn dense_decision(s: &Scratch, p: &Params, len1: usize, nd: usize) -> Decision {
        let base = p.base();
        let (mut fwd, mut bwd) = (vec![0i32; nd], vec![0i32; nd]);
        let (mut best, mut f) = (-p.c, 0i32);
        for d in 0..nd {
            f = p.weight(s.cnt[d]) + f.max(0);
            fwd[d] = f;
            best = best.max(f);
        }
        if base + best < p.minsc {
            return Decision::Fail;
        }
        let mut b = 0i32;
        for d in (0..nd).rev() {
            b = p.weight(s.cnt[d]) + b.max(0);
            bwd[d] = b;
        }
        let bound = |d: usize| base + fwd[d] + bwd[d] - p.weight(s.cnt[d]);
        let (mut lo, mut hi) = (len1 as i64, -1i64);
        let mut d = 0;
        while d < nd {
            if bound(d) < p.minsc {
                d += 1;
                continue;
            }
            let (mut ub, mut i0, mut dmax) = (0i32, len1 as i64, -1i64);
            while d < nd && bound(d) >= p.minsc {
                ub = ub.max(bound(d));
                if s.cnt[d] != 0 {
                    i0 = i0.min(s.minrow[d] as i64);
                    dmax = d as i64;
                }
                d += 1;
            }
            if dmax >= 0 {
                lo = lo.min(i0);
                hi = hi.max(dmax - 1 + p.tail(ub) as i64);
            }
        }
        if hi < 0 {
            return Decision::Fail;
        }
        Decision::Hull {
            hb: lo.max(0) as usize,
            he: hi.min(len1 as i64 - 1) as usize,
        }
    }

    /// Sparse and dense scans agree on every job, including clustered, repetitive and hit-dense
    /// windows and every scoring of the other tests.
    #[test]
    fn sparse_scan_equals_dense_scan() {
        let mut state = 0x5555_0123_dead_beefu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let params: Vec<Params> = [
            (1, 4, 6, 1, 6, 1, 19),
            (1, 4, 6, 1, 6, 1, 7),
            (1, 6, 6, 1, 6, 1, 19),
            (1, 4, 8, 2, 8, 2, 30),
            (2, 8, 12, 2, 12, 2, 38),
            (1, 5, 5, 3, 7, 1, 12),
        ]
        .iter()
        .filter_map(|&(a, b, o, e, oi, ei, m)| Params::from(a, b, o, e, oi, ei, m))
        .collect();
        let mut s = Scratch::new();
        let (mut hulls, mut fails) = (0, 0);
        for iter in 0..20000 {
            let qlen = 10 + next() % 200;
            let tlen = qlen + next() % 1500;
            let alpha = [4, 4, 4, 2, 3][next() % 5];
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % alpha) as u8).collect();
            let q: Vec<u8> = (0..qlen).map(|_| (next() % alpha) as u8).collect();
            for _ in 0..next() % 4 {
                let l = (5 + next() % 60).min(qlen);
                let (qs, at) = (next() % (qlen - l + 1), next() % (tlen - l + 1));
                t[at..at + l].copy_from_slice(&q[qs..qs + l]);
                for _ in 0..next() % 3 {
                    t[at + next() % l] = (next() % 4) as u8;
                }
            }
            let p = &params[iter % params.len()];
            if let Some(nd) = count_hits(&mut s, &t, &q, p, usize::MAX) {
                let want = dense_decision(&s, p, tlen, nd);
                assert_eq!(sparse_decision(&mut s, p, tlen, nd), want, "iter {iter}");
                match want {
                    Decision::Hull { .. } => hulls += 1,
                    Decision::Fail => fails += 1,
                    Decision::Full => {}
                }
            }
        }
        assert!(hulls > 2000 && fails > 2000, "hulls {hulls}, fails {fails}");
    }

    #[test]
    fn params_match_the_lemma_at_known_scorings() {
        let p = Params::from(1, 4, 6, 1, 6, 1, 19).unwrap();
        assert_eq!((p.k, p.c, p.base()), (5, 1, 5));
        // -B 6 admits 7-mers (mismatch and gap opens >= 6).
        assert_eq!(Params::from(1, 6, 6, 1, 6, 1, 19).unwrap().k, 7);
        // A threshold a hit-free window reaches proves nothing.
        assert!(Params::from(1, 4, 6, 1, 6, 1, 4).is_none());
        // Mismatch 3 caps K at 4, below the floor of 5.
        assert!(Params::from(1, 3, 6, 1, 6, 1, 19).is_none());
    }

    /// Generated rescue-shaped jobs, many of them near the threshold: every `Fail` must be a job
    /// whose real score is below `minsc`, and every `Hull` must reproduce `score`, `qe`, `score2`
    /// and the shifted `te`/`te2` of the full window, at several scorings and thresholds.
    /// The whole pruned rescue path on one mixed batch: wherever `ksw_align2` scores `>= minsc`,
    /// every field matches; everywhere else the job is rejected the way `mem_matesw` rejects it.
    #[test]
    fn batched_mate_rescue_keeps_every_consumed_field() {
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut bufs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for _ in 0..3000 {
            let qlen = 30 + next() % 140;
            let tlen = qlen + next() % 1400;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            for _ in 0..next() % 3 {
                let l = (8 + next() % 50).min(qlen);
                let (qs, at) = (next() % (qlen - l + 1), next() % (tlen - l + 1));
                t[at..at + l].copy_from_slice(&q[qs..qs + l]);
                for _ in 0..next() % 3 {
                    t[at + next() % l] = (next() % 4) as u8;
                }
            }
            if next() % 20 == 0 {
                t[next() % tlen] = 4;
            }
            bufs.push((q, t));
        }
        let jobs: Vec<crate::KswJob> = bufs
            .iter()
            .map(|(q, t)| crate::KswJob {
                query: q,
                target: t,
            })
            .collect();
        for &(a, b, o, e, minsc) in &[(1i8, 4i8, 6, 1, 19), (1, 6, 6, 1, 19), (2, 8, 12, 2, 38)] {
            let mat = scmat(a, b);
            // The pruned path itself, whatever `BWA4_RESCUE_PRUNE` says.
            let p = Params::from(a as i32, b as i32, o, e, o, e, minsc);
            assert!(p.is_some());
            let got = crate::matesw::batched_align(&jobs, 5, &mat, o, e, o, e, minsc, a as i32, p);
            for (i, j) in jobs.iter().enumerate() {
                let lanes = if (j.query.len() as i32) * (a as i32) < 250 {
                    16
                } else {
                    8
                };
                let want = ksw_align2(
                    j.query, j.target, 5, &mat, o, e, o, e, minsc, a as i32, lanes,
                );
                if want.score >= minsc {
                    assert_eq!(got[i], want, "job {i}, scoring ({a}, {b}, {o}, {e})");
                } else {
                    assert!(
                        got[i].score < minsc && got[i].qb < 0 && got[i].tb < 0,
                        "job {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn decisions_reproduce_ksw_align2() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let scorings: &[(i8, i8, i32, i32, i32, i32)] = &[
            (1, 4, 6, 1, 6, 1),
            (1, 6, 6, 1, 6, 1),
            (1, 4, 8, 2, 8, 2),
            (2, 8, 12, 2, 12, 2),
            (1, 5, 5, 3, 7, 1),
        ];
        let (mut fails, mut hulls, mut rows_kept, mut rows_all) = (0, 0, 0usize, 0usize);
        for iter in 0..6000 {
            let qlen = 20 + next() % 160;
            let tlen = qlen + next() % 1200;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            // Low-complexity windows (tandem repeats, poly-A) every so often.
            if next() % 6 == 0 {
                let period = 1 + next() % 4;
                let at = next() % tlen;
                let n = (next() % 300).min(tlen - at);
                for x in 0..n {
                    t[at + x] = t[at + x % period];
                }
            }
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            // Plant a mutated copy of a query substring, sized around the threshold, sometimes
            // twice so score2 has something to find.
            for _ in 0..next() % 3 {
                let l = 10 + next() % qlen.min(60);
                let l = l.min(qlen);
                let qs = next() % (qlen - l + 1);
                let at = next() % (tlen - l + 1);
                for x in 0..l {
                    t[at + x] = q[qs + x];
                }
                for _ in 0..next() % 4 {
                    t[at + next() % l] = (next() % 4) as u8;
                }
                // An indel inside the planted copy, now and then.
                if next() % 3 == 0 && l > 8 {
                    let p = at + 2 + next() % (l - 4);
                    if next() % 2 == 0 {
                        t.remove(p);
                        t.push((next() % 4) as u8);
                    } else {
                        t.insert(p, (next() % 4) as u8);
                        t.pop();
                    }
                }
            }
            if next() % 10 == 0 {
                q.reverse();
            }
            let (a, b, o_del, e_del, o_ins, e_ins) = scorings[iter % scorings.len()];
            let mat = scmat(a, b);
            let minsc = [12, 19, 25, 32][next() % 4] * a as i32;
            let Some(p) = Params::from(a as i32, b as i32, o_del, e_del, o_ins, e_ins, minsc)
            else {
                continue;
            };
            let lanes = if (qlen as i32) * (a as i32) < 250 {
                16
            } else {
                8
            };
            let full = ksw_align2(
                &q, &t, 5, &mat, o_del, e_del, o_ins, e_ins, minsc, a as i32, lanes,
            );
            match decide(&t, &q, &p, usize::MAX) {
                Decision::Full => {}
                Decision::Fail => {
                    fails += 1;
                    assert!(
                        full.score < minsc,
                        "iter {iter}: Fail but score {}",
                        full.score
                    );
                }
                Decision::Hull { hb, he } => {
                    hulls += 1;
                    rows_kept += he + 1 - hb;
                    rows_all += tlen;
                    let sub = ksw_align2(
                        &q,
                        &t[hb..=he],
                        5,
                        &mat,
                        o_del,
                        e_del,
                        o_ins,
                        e_ins,
                        minsc,
                        a as i32,
                        lanes,
                    );
                    if full.score >= minsc {
                        let sh = |x: i32| if x >= 0 { x + hb as i32 } else { x };
                        assert_eq!(
                            (sub.score, sub.qe, sh(sub.te), sub.score2, sh(sub.te2)),
                            (full.score, full.qe, full.te, full.score2, full.te2),
                            "iter {iter}: hull {hb}..={he} of {tlen}, minsc {minsc}"
                        );
                        assert_eq!((sh(sub.tb), sub.qb), (full.tb, full.qb), "iter {iter}");
                    } else {
                        assert!(sub.score < minsc, "iter {iter}");
                    }
                }
            }
        }
        // The generator must actually exercise both shortcuts.
        assert!(fails > 300 && hulls > 300, "fails {fails}, hulls {hulls}");
        assert!(rows_kept < rows_all, "no hull narrowed anything");
    }
}
