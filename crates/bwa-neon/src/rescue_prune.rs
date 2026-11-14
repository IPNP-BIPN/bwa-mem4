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
    pub(crate) k: usize,
    pub(crate) a: i32,
    pub(crate) c: i32,
    pub(crate) o_del: i32,
    pub(crate) e_del: i32,
    pub(crate) minsc: i32,
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
    #[cfg(test)]
    fn weight(&self, cnt: u16) -> i32 {
        self.a * cnt as i32 - self.c
    }

    /// Rows past the last hit diagonal's end that an alignment with interval bound `ub` can still
    /// reach at a score `>= tau`: each such row is a deleted base, and the deletion has to be paid
    /// for out of the bound's margin (`ub - tau`) plus the `(K-1)a` slack a trailing run carries.
    fn tail(&self, ub: i32) -> i32 {
        self.tail_at(ub, self.minsc)
    }

    /// [`Self::tail`] at threshold `tau` in place of `minsc`.
    pub(crate) fn tail_at(&self, ub: i32, tau: i32) -> i32 {
        let n = ub - tau - self.o_del + (self.k as i32 - 1) * self.a;
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

/// Per-thread scratch. The query table depends only on the oriented mate, which repeats across
/// every anchor rescued with it, so it is rebuilt only when the mate or K changes.
// The NEON counter's buffers are unused on other targets, which run the scalar counter.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
struct Scratch {
    /// Per K-mer code, 0 when the cached query does not hold it, else
    /// `n << 16 | (n > 1) << 15 | j`: `n` occurrences, the latest ending at query position `j`.
    /// `touched` lists the codes set, so the next query clears only those.
    tab: Vec<u32>,
    touched: Vec<u16>,
    /// Previous query position with the same K-mer code, or -1.
    nxt: Vec<i16>,
    /// The query the table describes, its K, and whether it holds an N.
    qcache: Vec<u8>,
    qk: usize,
    q_has_n: bool,
    /// Per diagonal: hit count, and first row of its first hit (valid where `cnt != 0`).
    cnt: Vec<u16>,
    minrow: Vec<u16>,
    /// Four partial hit counts, one per window row modulo 4, `stride` apart, each with a trash slot
    /// past the diagonals for rows without a hit. Consecutive rows of a run hit one diagonal, and a
    /// single counter would chain each increment on the previous store. All zero between jobs.
    cnt4: Vec<u16>,
    /// One bit per window row whose K-mer occurs more than once in the query, 64 rows a word.
    multi: Vec<u64>,
    /// The current window's K-mer codes, `codes[r]` for `t[r..r + K]`.
    codes: Vec<u16>,
    /// At K = 5, one bit per 5-mer code the cached query holds (bit `c & 7` of byte `c >> 3`).
    pres: [u8; 128],
    /// The NEON counter's buffers: the zero-padded window, the hit rows and their codes, the run
    /// boundaries (list index, diagonal, row) and the rows of repeated codes with their position.
    rbuf: Vec<u8>,
    hrow: Vec<u16>,
    hcode: Vec<u16>,
    rb_idx: Vec<u16>,
    rb_d: Vec<u16>,
    rb_r: Vec<u16>,
    mrow: Vec<u16>,
    mj: Vec<u16>,
    /// One bit per diagonal holding a hit.
    occ_bits: Vec<u64>,
    /// [`sparse_decision`]'s lists: per occupied diagonal in ascending order `[d, P, M, X]` (`n_occ`
    /// of them), and per occupied diagonal whose bound reaches `minsc`, in descending order,
    /// `[index, X, bound, 0]` (`n_hot` of them).
    fw: Vec<[i32; 4]>,
    hot: Vec<[i32; 4]>,
    n_occ: usize,
    n_hot: usize,
    /// The components at `minsc`.
    c19: Vec<Comp>,
    /// The hit total of the last count, and [`dense`]'s buffers: per diagonal the segment-local
    /// prefix sums and running minima (segment-major), the bounds (diagonal order), and the
    /// bitsets of diagonals at threshold and of those holding a hit. `dense` says which scan
    /// produced `c19`.
    nhits: usize,
    /// `occ_bits` not yet derived from `cnt` (see [`occ_ready`]): the NEON counter leaves it to
    /// the sparse scan, its only reader.
    occ_stale: bool,
    dp: Vec<i16>,
    dm: Vec<i16>,
    bnd: Vec<i16>,
    mw: Vec<u64>,
    hw: Vec<u64>,
    dense: bool,
}

/// [`Scratch::tab`]'s flag for a code the query holds more than once, and its position field.
const TAB_MULTI: u32 = 1 << 15;
const TAB_POS: u32 = TAB_MULTI - 1;

impl Scratch {
    fn new() -> Scratch {
        Scratch {
            tab: Vec::new(),
            touched: Vec::with_capacity(QCAP),
            nxt: vec![-1; QCAP],
            qcache: Vec::with_capacity(QCAP),
            qk: 0,
            q_has_n: false,
            cnt: Vec::new(),
            minrow: Vec::new(),
            cnt4: Vec::new(),
            multi: Vec::new(),
            codes: Vec::new(),
            pres: [0; 128],
            rbuf: Vec::new(),
            hrow: Vec::new(),
            hcode: Vec::new(),
            rb_idx: Vec::new(),
            rb_d: Vec::new(),
            rb_r: Vec::new(),
            mrow: Vec::new(),
            mj: Vec::new(),
            occ_bits: Vec::new(),
            fw: Vec::new(),
            hot: Vec::new(),
            n_occ: 0,
            n_hot: 0,
            c19: Vec::new(),
            nhits: 0,
            occ_stale: false,
            dp: Vec::new(),
            dm: Vec::new(),
            bnd: Vec::new(),
            mw: Vec::new(),
            hw: Vec::new(),
            dense: false,
        }
    }

    /// Index the query's K-mers, unless the table already describes this query at this K. A query
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
        if self.tab.len() != ncode {
            self.tab = vec![0; ncode];
            self.touched.clear();
        }
        for &c in &self.touched {
            self.tab[c as usize] = 0;
        }
        self.touched.clear();
        self.pres = [0; 128];
        if self.q_has_n {
            return;
        }
        let mask = (ncode - 1) as u32;
        let mut code = 0u32;
        for &b in &q[..k - 1] {
            code = (code << 2) | b as u32;
        }
        // Branch-free and unchecked: a query is rebuilt for nearly every rescue job (a round holds
        // about one job per mate), and whether a code repeats is data, not a predictable branch.
        // `code <= mask < tab.len()`, `j < q.len() <= QCAP = nxt.len()`, and `touched` keeps
        // QCAP capacity, more than the distinct codes of one query.
        debug_assert!(q.len() <= self.nxt.len() && q.len() <= self.touched.capacity());
        let (tab, nxt) = (self.tab.as_mut_ptr(), self.nxt.as_mut_ptr());
        let touched = self.touched.as_mut_ptr();
        let mut nt = 0usize;
        // SAFETY: the bounds above.
        unsafe {
            for j in k - 1..q.len() {
                code = ((code << 2) | u32::from(*q.get_unchecked(j))) & mask;
                let v = *tab.add(code as usize);
                if k == 5 {
                    self.pres[(code >> 3) as usize] |= 1 << (code & 7);
                }
                let seen = v != 0;
                *touched.add(nt) = code as u16;
                nt += usize::from(!seen);
                *nxt.add(j) = if seen { (v & TAB_POS) as i16 } else { -1 };
                *tab.add(code as usize) =
                    ((v >> 16) + 1) << 16 | if seen { TAB_MULTI } else { 0 } | j as u32;
            }
            self.touched.set_len(nt);
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
///   costs the bound more than the rows it can save. Changes speed only, never a decision's
///   correctness.
pub(crate) fn decide(t: &[u8], q: &[u8], p: &Params, max_hits: usize) -> Decision {
    let (len1, len2, k) = (t.len(), q.len(), p.k);
    if len1 < k || len2 < k || len1 > WCAP || len2 > QCAP {
        return Decision::Full;
    }
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        match count_hits(s, t, q, p, max_hits) {
            Some(nd) => decide_counts(s, p, len1, nd),
            None => Decision::Full,
        }
    })
}

/// The per-diagonal hit counts (`cnt`, `minrow` where `cnt != 0`, `occ_bits`). Returns the number of
/// diagonal slots, or `None` when the job is not filtered (an N in either sequence, or more hits than
/// `max_hits`). A job proven to fail by the hit total alone comes back with no diagonal occupied.
fn count_hits(s: &mut Scratch, t: &[u8], q: &[u8], p: &Params, max_hits: usize) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    if p.k == 5 {
        // SAFETY: NEON is part of the aarch64 baseline.
        return unsafe { neon::count_hits(s, t, q, p, max_hits) };
    }
    #[cfg(target_arch = "x86_64")]
    if p.k == 5 && sse::available() {
        // SAFETY: SSSE3 and SSE4.1 were just detected.
        return unsafe { sse::count_hits(s, t, q, p, max_hits) };
    }
    count_hits_scalar(s, t, q, p, max_hits)
}

/// [`count_hits`] in portable code: the reference the NEON counter is checked against, and the
/// counter for every K on other targets and for K > 5.
fn count_hits_scalar(
    s: &mut Scratch,
    t: &[u8],
    q: &[u8],
    p: &Params,
    max_hits: usize,
) -> Option<usize> {
    let (len1, len2, k) = (t.len(), q.len(), p.k);
    s.load_query(q, k);
    if s.q_has_n || t.iter().fold(0u8, |acc, &b| acc | b) & !3 != 0 {
        return None;
    }
    // ---- Diagonal `d = i - j` is stored at `d + off`, with `off` the query rounded up to 16 so
    //      the pad columns' diagonals have slots too. `span`: the diagonals rounded up to whole
    //      bitmap words; `trash`: the slot rows without a hit count into, past every word ----
    let off = len2.div_ceil(16) * 16;
    let nd = len1 + off + 1;
    let nw = nd.div_ceil(64);
    let span = nw * 64;
    let (stride, trash) = (span + 8, span);
    if s.cnt4.len() < 4 * stride {
        s.cnt4.resize(4 * stride, 0);
    }
    s.minrow.clear();
    s.minrow.resize(span + 1, u16::MAX);
    s.multi.clear();
    s.multi.resize((len1 + 1 - k).div_ceil(64), 0);
    // Each code from its own K bytes rather than rolled from the previous one, so the loop carries
    // no dependency and vectorises.
    let mut codes = std::mem::take(&mut s.codes);
    codes.clear();
    match k {
        5 => codes.extend(t.windows(5).map(code_of)),
        6 => codes.extend(t.windows(6).map(code_of)),
        7 => codes.extend(t.windows(7).map(code_of)),
        _ => codes.extend(t.windows(8).map(code_of)),
    }
    let nhits = primary_hits(s, &codes, k, off, stride, trash);
    s.nhits = nhits;
    s.occ_bits.clear();
    s.occ_bits.resize(nw, 0);
    s.cnt.clear();
    s.cnt.resize(span, 0);
    // Early fail: every interval is non-empty and sums `a * cnt - c` over its diagonals, so no
    // interval can beat `base + a * nhits - c`. Leaving every diagonal empty says exactly that.
    let early_fail = p.base() + p.a * nhits as i32 - p.c < p.minsc;
    if nhits > max_hits || early_fail {
        for c in s.cnt4.chunks_exact_mut(stride).take(4) {
            c[..=trash].fill(0);
        }
        s.codes = codes;
        return (nhits <= max_hits).then_some(nd);
    }
    // ---- The other occurrences of repeated K-mers, in any order (`minrow` takes the min) ----
    {
        let (tab, nxt) = (&s.tab[..], &s.nxt[..]);
        let (cnt4, minrow) = (&mut s.cnt4[..], &mut s.minrow[..]);
        for (w, &word) in s.multi.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let r = w * 64 + word.trailing_zeros() as usize;
                word &= word - 1;
                let mut j = nxt[(tab[codes[r] as usize] & TAB_POS) as usize];
                while j >= 0 {
                    let d = r + k - 1 + off - j as usize;
                    cnt4[(r & 3) * stride + d] += 1;
                    minrow[d] = minrow[d].min(r as u16);
                    j = nxt[j as usize];
                }
            }
        }
    }
    // ---- Fold the four partial counts into `cnt`, zero them again, and mark the occupied
    //      diagonals ----
    {
        let (c0, rest) = s.cnt4.split_at_mut(stride);
        let (c1, rest) = rest.split_at_mut(stride);
        let (c2, c3) = rest.split_at_mut(stride);
        for d in 0..span {
            s.cnt[d] = c0[d] + c1[d] + c2[d] + c3[d];
        }
        for c in [c0, c1, c2, &mut c3[..stride]] {
            c[..=trash].fill(0);
        }
    }
    for (w, bits) in s.occ_bits.iter_mut().enumerate() {
        *bits = occupied_word(&s.cnt[w * 64..w * 64 + 64]);
    }
    s.codes = codes;
    Some(nd)
}

/// The 2-bit code of a K-mer, first base highest.
#[inline(always)]
fn code_of(w: &[u8]) -> u16 {
    w.iter().fold(0u16, |c, &b| (c << 2) | b as u16)
}

/// One pass over the window rows, latest first: each row's K-mer hit at its latest query
/// occurrence counts into `cnt4` and `minrow` (rows descending, so the last store to `minrow[d]` is
/// the diagonal's earliest row), and a row whose K-mer occurs again in the query gets a `multi`
/// bit. Returns the exact hit total. No branch per row: a row without a hit counts into `trash`.
/// The `multi` bits stay in a register for 64 rows: a list indexed by a running count would make
/// every row's loads wait for the previous row's store address, which depends on its own loads.
#[inline(always)]
fn primary_hits(
    s: &mut Scratch,
    codes: &[u16],
    k: usize,
    off: usize,
    stride: usize,
    trash: usize,
) -> usize {
    let (tab, cnt4, minrow, multi) = (
        &s.tab[..],
        &mut s.cnt4[..],
        &mut s.minrow[..],
        &mut s.multi[..],
    );
    let base = k - 1 + off;
    let mut nhits = 0usize;
    for w in (0..multi.len()).rev() {
        let mut bits = 0u64;
        let chunk = &codes[w * 64..codes.len().min(w * 64 + 64)];
        for (x, &code) in chunk.iter().enumerate().rev() {
            let r = w * 64 + x;
            let v = tab[code as usize];
            nhits += (v >> 16) as usize;
            let d = if v != 0 {
                r + base - (v & TAB_POS) as usize
            } else {
                trash
            };
            cnt4[(r & 3) * stride + d] += 1;
            minrow[d] = r as u16;
            bits |= u64::from((v & TAB_MULTI) != 0) << x;
        }
        multi[w] = bits;
    }
    nhits
}

/// One bit per nonzero count, bit `x` for `cnt[x]`.
#[inline]
fn occupied_word(cnt: &[u16]) -> u64 {
    let cnt = &cnt[..64];
    // SSE2 (x86-64 baseline): LLVM's own vectorisation widened every count to 64 bits, ~65
    // instructions a word; compare, pack and movemask take 16 counts in four.
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let mut zero_bits = 0u64;
        // SAFETY: SSE2 is part of x86-64; the eight unaligned 16-byte loads stay inside `cnt`.
        unsafe {
            let z = _mm_setzero_si128();
            let p = cnt.as_ptr().cast::<__m128i>();
            for g in 0..4 {
                let a = _mm_cmpeq_epi16(_mm_loadu_si128(p.add(2 * g)), z);
                let b = _mm_cmpeq_epi16(_mm_loadu_si128(p.add(2 * g + 1)), z);
                let m = _mm_movemask_epi8(_mm_packs_epi16(a, b)) as u32;
                zero_bits |= u64::from(m) << (16 * g);
            }
        }
        !zero_bits
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut bits = 0u64;
        for (x, &c) in cnt.iter().enumerate() {
            bits |= u64::from(c != 0) << x;
        }
        bits
    }
}

/// The K = 5 hit counter in NEON, after fg-labs/bwa-mem3's `lean_neon` (`src/rescue_prune_neon.h`,
/// steps 1 to 5): the window's rows are tested 16 at a time against the query's 5-mer presence
/// bitmap and only the rows with a hit (about a third) are kept; their hits at the latest query
/// occurrence are taken 8 rows at a time, and rows that continue a run along one diagonal fold into
/// the run, so `cnt` takes one store per run rather than one per hit. Repeated codes' other
/// occurrences go through the scalar chain walk. Same `cnt`, `minrow` and `occ_bits` as
/// [`count_hits_scalar`].
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Params, Scratch, TAB_MULTI, TAB_POS};
    use std::arch::aarch64::*;

    /// `SHUF[m]`: the `vqtbl1q_u8` indices that left-pack the u16 lanes set in `m`.
    static SHUF: [[u8; 16]; 256] = {
        let mut t = [[0xFFu8; 16]; 256];
        let mut m = 0;
        while m < 256 {
            let (mut n, mut l) = (0, 0);
            while l < 8 {
                if (m >> l) & 1 == 1 {
                    t[m][2 * n] = 2 * l as u8;
                    t[m][2 * n + 1] = 2 * l as u8 + 1;
                    n += 1;
                }
                l += 1;
            }
            m += 1;
        }
        t
    };

    /// Store the lanes of `v` selected by `m` to `out[at..]`, packed, and return the new `at`.
    /// Writes 8 lanes: `out` needs 8 lanes of slack past the last packed one.
    #[inline(always)]
    unsafe fn pack(out: &mut [u16], at: usize, v: uint16x8_t, m: u32) -> usize {
        debug_assert!(at + 8 <= out.len());
        let sh = vld1q_u8(SHUF[m as usize].as_ptr());
        let x = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(v), sh));
        vst1q_u16(out.as_mut_ptr().add(at), x);
        // A table, not `count_ones`: the cursor chains every pack on the previous, and a scalar
        // popcount crosses to the vector unit and back (fg-labs/bwa-mem3's `pc[]`).
        at + PC[m as usize] as usize
    }

    /// `PC[m]`: the number of lanes set in `m`.
    static PC: [u8; 256] = {
        let mut t = [0u8; 256];
        let mut m = 0;
        while m < 256 {
            t[m] = (m as u32).count_ones() as u8;
            m += 1;
        }
        t
    };

    #[inline(always)]
    fn grow(v: &mut Vec<u16>, n: usize) {
        if v.len() < n {
            v.resize(n, 0);
        }
    }

    pub(super) unsafe fn count_hits(
        s: &mut Scratch,
        t: &[u8],
        q: &[u8],
        p: &Params,
        max_hits: usize,
    ) -> Option<usize> {
        const K: usize = 5;
        let (len1, len2) = (t.len(), q.len());
        s.load_query(q, K);
        // ---- 1. N check while copying the window into a zero-padded buffer, so the 16-row code
        //      blocks never read past it ----
        // One pass: copy 16 bytes, OR them in. `rbuf` keeps `len1 + 32` bytes, the last 32 zero.
        if s.rbuf.len() < len1 + 32 {
            s.rbuf.resize(len1 + 32, 0);
        }
        let rb = s.rbuf.as_mut_ptr();
        let mut ov = vdupq_n_u8(0);
        // Stored through an opaque zero: a plain load-store loop is turned into a `memcpy` call
        // plus a second pass for the OR (0.1% of M4 PE in the call alone).
        let z = std::hint::black_box(vdupq_n_u8(0));
        let mut i = 0;
        while i + 16 <= len1 {
            let x = vld1q_u8(t.as_ptr().add(i));
            ov = vorrq_u8(ov, x);
            vst1q_u8(rb.add(i), vorrq_u8(x, z));
            i += 16;
        }
        let mut tail = 0u8;
        while i < len1 {
            tail |= t[i];
            *rb.add(i) = t[i];
            i += 1;
        }
        vst1q_u8(rb.add(len1), vdupq_n_u8(0));
        vst1q_u8(rb.add(len1 + 16), vdupq_n_u8(0));
        if s.q_has_n || (vmaxvq_u8(ov) | tail) & !3 != 0 {
            return None;
        }
        let off = len2.div_ceil(16) * 16;
        let nd = len1 + off + 1;
        let nw = nd.div_ceil(64);
        let span = nw * 64;
        s.occ_bits.clear();
        s.occ_bits.resize(nw, 0);
        s.occ_stale = false;
        // ---- 3. hit rows, 16 at a time: 5-mer codes and a presence-bitmap lookup (two 64-byte
        //      TBLs), the rows with a hit and their codes left-packed ----
        let nrow = len1 + 1 - K;
        grow(&mut s.hrow, nrow + 16);
        grow(&mut s.hcode, nrow + 16);
        let mut np = 0usize;
        {
            let pr = s.pres.as_ptr();
            let plo = uint8x16x4_t(
                vld1q_u8(pr),
                vld1q_u8(pr.add(16)),
                vld1q_u8(pr.add(32)),
                vld1q_u8(pr.add(48)),
            );
            let phi = uint8x16x4_t(
                vld1q_u8(pr.add(64)),
                vld1q_u8(pr.add(80)),
                vld1q_u8(pr.add(96)),
                vld1q_u8(pr.add(112)),
            );
            const P2: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0];
            const BW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            const IO: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let (p2, bw) = (vld1q_u8(P2.as_ptr()), vld1q_u8(BW.as_ptr()));
            let (k64, k7, k8) = (vdupq_n_u8(64), vdupq_n_u8(7), vdupq_n_u16(8));
            let mut pos = vld1q_u16(IO.as_ptr());
            let rb = s.rbuf.as_ptr();
            let mut b = 0usize;
            while b < nrow {
                let buf = rb.add(b);
                let a0 = vld1q_u8(buf);
                let a1 = vld1q_u8(buf.add(1));
                let a2 = vld1q_u8(buf.add(2));
                let a3 = vld1q_u8(buf.add(3));
                let a4 = vld1q_u8(buf.add(4));
                // code & 255 = a1 a2 a3 a4 (2 bits each), code >> 8 = a0
                let lo = vsliq_n_u8::<6>(vsliq_n_u8::<4>(vsliq_n_u8::<2>(a4, a3), a2), a1);
                let idx = vsraq_n_u8::<3>(vshlq_n_u8::<5>(a0), lo); // code >> 3
                                                                    // idx >= 64 wraps to >= 192 under the subtraction for idx < 64, which TBX leaves
                let byte = vqtbx4q_u8(vqtbl4q_u8(plo, idx), phi, vsubq_u8(idx, k64));
                let hit = vtstq_u8(byte, vqtbl1q_u8(p2, vandq_u8(lo, k7)));
                let mut m = vandq_u8(hit, bw);
                m = vpaddq_u8(m, m);
                m = vpaddq_u8(m, m);
                m = vpaddq_u8(m, m);
                let mut m16 = u32::from(vgetq_lane_u16::<0>(vreinterpretq_u16_u8(m)));
                if b + 16 > nrow {
                    m16 &= (1u32 << (nrow - b)) - 1;
                }
                let c0 = vreinterpretq_u16_u8(vzip1q_u8(lo, a0));
                let c1 = vreinterpretq_u16_u8(vzip2q_u8(lo, a0));
                let (ml, mh) = (m16 & 0xFF, m16 >> 8);
                let at = pack(&mut s.hrow, np, pos, ml);
                pack(&mut s.hcode, np, c0, ml);
                let pos8 = vaddq_u16(pos, k8);
                np = pack(&mut s.hrow, at, pos8, mh);
                pack(&mut s.hcode, at, c1, mh);
                pos = vaddq_u16(pos8, k8);
                b += 16;
            }
        }
        if np == 0 {
            return Some(nd);
        }
        // Every hit row holds at least one hit, so the gate can already refuse: the densest jobs
        // then skip layer 1, their most expensive part.
        if np > max_hits {
            return None;
        }
        // ---- 4. layer 1, 8 hit rows at a time: each row's hit at the latest query occurrence of
        //      its code, d = row + K - 1 + off - j. A row continuing the previous row's run
        //      (row + 1, same diagonal) is no boundary; boundaries are left-packed as (list index,
        //      diagonal, row), rows of repeated codes as (row, j). The exact hit total comes out of
        //      the same pass ----
        grow(&mut s.rb_idx, np + 16);
        grow(&mut s.rb_d, np + 16);
        grow(&mut s.rb_r, np + 16);
        grow(&mut s.mrow, np + 16);
        grow(&mut s.mj, np + 16);
        let (mut nb, mut nm) = (0usize, 0usize);
        let mut extra = vdupq_n_u32(0);
        {
            const WL: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
            const WH: [u16; 8] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
            const IO: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let (wl, wh) = (vld1q_u16(WL.as_ptr()), vld1q_u16(WH.as_ptr()));
            let mut idx = vld1q_u16(IO.as_ptr());
            let (k8, one) = (vdupq_n_u16(8), vdupq_n_u16(1));
            let koff = vdupq_n_u16((off + K - 1) as u16);
            let kpos = vdupq_n_u16(TAB_POS as u16);
            let kmulti = vdupq_n_u16(TAB_MULTI as u16);
            let vn = vdupq_n_u16(np as u16);
            let (mut prev_r, mut prev_d) = (vdupq_n_u16(0), vdupq_n_u16(0xFFFF));
            let tab = s.tab.as_ptr();
            // Codes past `np` are stale but in range (< 1024); their lanes are masked.
            let look4 = |c: *const u16| -> uint16x8_t {
                let x = (c as *const u64).read_unaligned();
                let e = |sh: u32| u64::from(*tab.add(((x >> sh) & 1023) as usize));
                let a = e(0) | e(16) << 32;
                let b = e(32) | e(48) << 32;
                vreinterpretq_u16_u64(vcombine_u64(vcreate_u64(a), vcreate_u64(b)))
            };
            let mut k0 = 0usize;
            while k0 < np {
                let r = vld1q_u16(s.hrow.as_ptr().add(k0));
                let e0 = look4(s.hcode.as_ptr().add(k0));
                let e1 = look4(s.hcode.as_ptr().add(k0 + 4));
                let (lo, occ) = (vuzp1q_u16(e0, e1), vuzp2q_u16(e0, e1));
                let j = vandq_u16(lo, kpos);
                let d = vsubq_u16(vaddq_u16(r, koff), j);
                let rp = vextq_u16::<7>(prev_r, r);
                let dp = vextq_u16::<7>(prev_d, d);
                (prev_r, prev_d) = (r, d);
                let valid = vcltq_u16(idx, vn);
                let cont = vandq_u16(vceqq_u16(r, vaddq_u16(rp, one)), vceqq_u16(d, dp));
                let bd = vbicq_u16(valid, cont);
                let mu = vandq_u16(vtstq_u16(lo, kmulti), valid);
                extra = vpadalq_u16(extra, vandq_u16(vsubq_u16(occ, one), valid));
                let m = u32::from(vaddvq_u16(vorrq_u16(vandq_u16(bd, wl), vandq_u16(mu, wh))));
                let (mb, mm) = (m & 0xFF, m >> 8);
                pack(&mut s.rb_idx, nb, idx, mb);
                pack(&mut s.rb_d, nb, d, mb);
                nb = pack(&mut s.rb_r, nb, r, mb);
                pack(&mut s.mrow, nm, r, mm);
                nm = pack(&mut s.mj, nm, j, mm);
                idx = vaddq_u16(idx, k8);
                k0 += 8;
            }
        }
        let nhits = np + vaddvq_u32(extra) as usize;
        s.nhits = nhits;
        if nhits > max_hits {
            return None;
        }
        // Early fail: no interval can beat `base + a * nhits - c` (see the scalar counter).
        if p.base() + p.a * nhits as i32 - p.c < p.minsc {
            return Some(nd);
        }
        // Zeroed only now: the returns above leave no diagonal occupied, and `cnt`/`minrow` are
        // read only under an `occ_bits` bit.
        s.cnt.clear();
        s.cnt.resize(span, 0);
        if s.minrow.len() < span {
            s.minrow.resize(span, 0);
        }
        // ---- 5. layer 1's runs, last to first: rows descend, so the last store to `minrow[d]`
        //      is the diagonal's earliest row. Then the repeated codes' other occurrences ----
        s.rb_idx[nb] = np as u16;
        {
            // Unchecked: every `d` is a diagonal below `nd`, and `cnt` and `minrow` span the
            // rounded-up `nd`; `rb_idx` holds `nb + 1` entries.
            let (cnt, minrow) = (s.cnt.as_mut_ptr(), s.minrow.as_mut_ptr());
            let (bd, bi, br) = (s.rb_d.as_ptr(), s.rb_idx.as_ptr(), s.rb_r.as_ptr());
            for x in (0..nb).rev() {
                let d = *bd.add(x) as usize;
                *cnt.add(d) += *bi.add(x + 1) - *bi.add(x);
                *minrow.add(d) = *br.add(x);
            }
        }
        // ---- Layers >= 2: layer L holds each row whose code occurs at least L times, with its
        //      L-th latest occurrence. Within a layer, rows still ascend, so runs collapse as in
        //      layer 1; `minrow` takes the min. The next layer (rows whose chain continues) is
        //      packed in the same pass ----
        let nxt = s.nxt.as_ptr();
        // `nxt` of four query positions (past the list's end the lanes are stale, masked to stay
        // in `nxt`, whose QCAP entries cover every query position).
        let nxt4 = |j: *const u16| -> u64 {
            let x = (j as *const u64).read_unaligned();
            let e = |sh: u32| u64::from(*nxt.add(((x >> sh) & 1023) as usize) as u16);
            e(0) | e(16) << 16 | e(32) << 32 | e(48) << 48
        };
        for x in 0..nm {
            s.mj[x] = *nxt.add(s.mj[x] as usize) as u16;
        }
        let mut n = nm;
        grow(&mut s.hrow, n + 16);
        grow(&mut s.hcode, n + 16);
        let (mut rows, mut js) = (std::mem::take(&mut s.mrow), std::mem::take(&mut s.mj));
        let (mut rows2, mut js2) = (std::mem::take(&mut s.hrow), std::mem::take(&mut s.hcode));
        {
            const WL: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
            const WH: [u16; 8] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
            const IO: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let (wl, wh) = (vld1q_u16(WL.as_ptr()), vld1q_u16(WH.as_ptr()));
            let (k8, one) = (vdupq_n_u16(8), vdupq_n_u16(1));
            let koff = vdupq_n_u16((off + K - 1) as u16);
            while n > 0 {
                let vn = vdupq_n_u16(n as u16);
                let mut idx = vld1q_u16(IO.as_ptr());
                let (mut prev_r, mut prev_d) = (vdupq_n_u16(0), vdupq_n_u16(0xFFFF));
                let (mut nb, mut n2) = (0usize, 0usize);
                let mut k0 = 0usize;
                while k0 < n {
                    let r = vld1q_u16(rows.as_ptr().add(k0));
                    let j = vld1q_u16(js.as_ptr().add(k0));
                    let jn = vreinterpretq_s16_u64(vcombine_u64(
                        vcreate_u64(nxt4(js.as_ptr().add(k0))),
                        vcreate_u64(nxt4(js.as_ptr().add(k0 + 4))),
                    ));
                    let d = vsubq_u16(vaddq_u16(r, koff), j);
                    let rp = vextq_u16::<7>(prev_r, r);
                    let dp = vextq_u16::<7>(prev_d, d);
                    (prev_r, prev_d) = (r, d);
                    let valid = vcltq_u16(idx, vn);
                    let cont = vandq_u16(vceqq_u16(r, vaddq_u16(rp, one)), vceqq_u16(d, dp));
                    let keep = vandq_u16(vcgezq_s16(jn), valid);
                    let m = u32::from(vaddvq_u16(vorrq_u16(
                        vandq_u16(vbicq_u16(valid, cont), wl),
                        vandq_u16(keep, wh),
                    )));
                    let (mb, mk) = (m & 0xFF, m >> 8);
                    pack(&mut s.rb_idx, nb, idx, mb);
                    pack(&mut s.rb_d, nb, d, mb);
                    nb = pack(&mut s.rb_r, nb, r, mb);
                    pack(&mut rows2, n2, r, mk);
                    n2 = pack(&mut js2, n2, vreinterpretq_u16_s16(jn), mk);
                    idx = vaddq_u16(idx, k8);
                    k0 += 8;
                }
                s.rb_idx[nb] = n as u16;
                // Unchecked as in layer 1.
                let (cnt, minrow) = (s.cnt.as_mut_ptr(), s.minrow.as_mut_ptr());
                let (bd, bi, br) = (s.rb_d.as_ptr(), s.rb_idx.as_ptr(), s.rb_r.as_ptr());
                for x in 0..nb {
                    let (d, row) = (*bd.add(x) as usize, *br.add(x));
                    let c = *cnt.add(d);
                    // A stale `minrow` (c == 0) ORed to u16::MAX: one `min`, no branch. Written
                    // as `if`, LLVM sank the load into a branch on `c`, mispredicted per run.
                    let stale = u16::from(c == 0).wrapping_neg();
                    *minrow.add(d) = (*minrow.add(d) | stale).min(row);
                    *cnt.add(d) = c + *bi.add(x + 1) - *bi.add(x);
                }
                n = n2;
                std::mem::swap(&mut rows, &mut rows2);
                std::mem::swap(&mut js, &mut js2);
            }
        }
        (s.mrow, s.mj, s.hrow, s.hcode) = (rows, js, rows2, js2);
        // `occ_bits` is derived from `cnt` only if the sparse scan runs (`occ_ready`).
        s.occ_stale = true;
        Some(nd)
    }

    /// `occ_bits` from `cnt` in one pass, not per run: consecutive runs mostly land in one
    /// 64-diagonal word, and a read-modify-write per run chains them through memory.
    ///
    /// # Safety
    /// NEON; `cnt` spans `occ_bits.len() * 64` diagonals.
    pub(super) unsafe fn occ_from_cnt(s: &mut Scratch) {
        const BW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bw = vld1q_u8(BW.as_ptr());
        let c = s.cnt.as_ptr();
        for (w, word) in s.occ_bits.iter_mut().enumerate() {
            let mut bits = 0u64;
            for g in 0..4 {
                let p = c.add(w * 64 + g * 16);
                let (a, b) = (vld1q_u16(p), vld1q_u16(p.add(8)));
                let m = vcombine_u8(vmovn_u16(vtstq_u16(a, a)), vmovn_u16(vtstq_u16(b, b)));
                let mut x = vandq_u8(m, bw);
                x = vpaddq_u8(x, x);
                x = vpaddq_u8(x, x);
                x = vpaddq_u8(x, x);
                bits |= u64::from(vgetq_lane_u16::<0>(vreinterpretq_u16_u8(x))) << (16 * g);
            }
            *word = bits;
        }
    }
}

/// [`neon`]'s K = 5 hit counter on x86 in SSSE3/SSE4.1, after fg-labs/bwa-mem3's `lean_x86`
/// (`src/rescue_prune_x86.h`): the same layout and passes lane for lane, so the same `cnt`,
/// `minrow` and `occ_bits`. The 128-byte presence lookup is eight 16-byte PSHUFBs joined by a
/// blend tree on the index bits 4 to 6; masks come from MOVEMASK instead of pairwise adds.
#[cfg(target_arch = "x86_64")]
mod sse {
    use super::{Params, Scratch, TAB_MULTI, TAB_POS};
    use std::arch::x86_64::*;

    /// Whether the host runs this counter. Cached by `std`'s feature detection.
    pub(super) fn available() -> bool {
        is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("sse4.1")
    }

    /// `SHUF[m]`: the PSHUFB indices that left-pack the u16 lanes set in `m` (0x80 zeroes a lane).
    static SHUF: [[u8; 16]; 256] = {
        let mut t = [[0x80u8; 16]; 256];
        let mut m = 0;
        while m < 256 {
            let (mut n, mut l) = (0, 0);
            while l < 8 {
                if (m >> l) & 1 == 1 {
                    t[m][2 * n] = 2 * l as u8;
                    t[m][2 * n + 1] = 2 * l as u8 + 1;
                    n += 1;
                }
                l += 1;
            }
            m += 1;
        }
        t
    };

    /// `PC[m]`: the number of lanes set in `m`.
    static PC: [u8; 256] = {
        let mut t = [0u8; 256];
        let mut m = 0;
        while m < 256 {
            t[m] = (m as u32).count_ones() as u8;
            m += 1;
        }
        t
    };

    /// Writes 8 lanes: `out` needs 8 lanes of slack past the last packed one.
    #[inline(always)]
    unsafe fn pack(out: &mut [u16], at: usize, v: __m128i, m: u32) -> usize {
        debug_assert!(at + 8 <= out.len());
        let sh = _mm_loadu_si128(SHUF[m as usize].as_ptr() as *const __m128i);
        _mm_storeu_si128(
            out.as_mut_ptr().add(at) as *mut __m128i,
            _mm_shuffle_epi8(v, sh),
        );
        at + PC[m as usize] as usize
    }

    #[inline(always)]
    fn grow(v: &mut Vec<u16>, n: usize) {
        if v.len() < n {
            v.resize(n, 0);
        }
    }

    #[inline(always)]
    unsafe fn ld(p: *const u8) -> __m128i {
        _mm_loadu_si128(p as *const __m128i)
    }

    /// Eight u16 lanes from two u64s (low lanes first).
    #[inline(always)]
    unsafe fn from2(a: u64, b: u64) -> __m128i {
        _mm_set_epi64x(b as i64, a as i64)
    }

    /// Lanes `(prev[7], cur[0..7])`, NEON's `vextq_u16::<7>(prev, cur)`.
    #[inline(always)]
    unsafe fn shift_in(prev: __m128i, cur: __m128i) -> __m128i {
        _mm_alignr_epi8::<14>(cur, prev)
    }

    /// See [`super::neon::count_hits`]; same results.
    ///
    /// # Safety
    /// SSSE3 and SSE4.1.
    #[target_feature(enable = "ssse3,sse4.1")]
    pub(super) unsafe fn count_hits(
        s: &mut Scratch,
        t: &[u8],
        q: &[u8],
        p: &Params,
        max_hits: usize,
    ) -> Option<usize> {
        const K: usize = 5;
        let (len1, len2) = (t.len(), q.len());
        s.load_query(q, K);
        // ---- 1. N check while copying the window into a zero-padded buffer ----
        if s.rbuf.len() < len1 + 32 {
            s.rbuf.resize(len1 + 32, 0);
        }
        let rb = s.rbuf.as_mut_ptr();
        let mut ov = _mm_setzero_si128();
        // Through an opaque zero, or the loop becomes a `memcpy` call (see the NEON version).
        let z = std::hint::black_box(_mm_setzero_si128());
        let mut i = 0;
        while i + 16 <= len1 {
            let x = ld(t.as_ptr().add(i));
            ov = _mm_or_si128(ov, x);
            _mm_storeu_si128(rb.add(i) as *mut __m128i, _mm_or_si128(x, z));
            i += 16;
        }
        let mut tail = 0u8;
        while i < len1 {
            tail |= t[i];
            *rb.add(i) = t[i];
            i += 1;
        }
        _mm_storeu_si128(rb.add(len1) as *mut __m128i, _mm_setzero_si128());
        _mm_storeu_si128(rb.add(len1 + 16) as *mut __m128i, _mm_setzero_si128());
        if s.q_has_n || _mm_testz_si128(ov, _mm_set1_epi8(!3)) == 0 || tail & !3 != 0 {
            return None;
        }
        let off = len2.div_ceil(16) * 16;
        let nd = len1 + off + 1;
        let nw = nd.div_ceil(64);
        let span = nw * 64;
        s.occ_bits.clear();
        s.occ_bits.resize(nw, 0);
        s.occ_stale = false;
        // ---- 3. hit rows, 16 at a time ----
        let nrow = len1 + 1 - K;
        grow(&mut s.hrow, nrow + 16);
        grow(&mut s.hcode, nrow + 16);
        let mut np = 0usize;
        {
            let pr = s.pres.as_ptr();
            let pt: [__m128i; 8] = std::array::from_fn(|x| ld(pr.add(16 * x)));
            const P2: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0];
            const IO: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let p2 = ld(P2.as_ptr());
            let (k7, k1f, k8) = (_mm_set1_epi8(7), _mm_set1_epi8(0x1F), _mm_set1_epi16(8));
            let mut pos = ld(IO.as_ptr() as *const u8);
            let rb = s.rbuf.as_ptr();
            let mut b = 0usize;
            while b < nrow {
                let buf = rb.add(b);
                let a0 = ld(buf);
                let (a1, a2, a3, a4) = (
                    ld(buf.add(1)),
                    ld(buf.add(2)),
                    ld(buf.add(3)),
                    ld(buf.add(4)),
                );
                // Bases are 0..=3, so the 16-bit shifts never carry across bytes.
                let lo = _mm_or_si128(
                    _mm_or_si128(_mm_slli_epi16::<6>(a1), _mm_slli_epi16::<4>(a2)),
                    _mm_or_si128(_mm_slli_epi16::<2>(a3), a4),
                );
                // code >> 3 = a0 << 5 | lo >> 3, below 128
                let idx = _mm_or_si128(
                    _mm_slli_epi16::<5>(a0),
                    _mm_and_si128(_mm_srli_epi16::<3>(lo), k1f),
                );
                // PSHUFB reads index bits 0-3 (bit 7 is clear); bits 4, 5, 6 pick the table,
                // each moved to bit 7 for BLENDV (the 16-bit shifts only spill into lower bits).
                let t: [__m128i; 8] = std::array::from_fn(|x| _mm_shuffle_epi8(pt[x], idx));
                let (s4, s5, s6) = (
                    _mm_slli_epi16::<3>(idx),
                    _mm_slli_epi16::<2>(idx),
                    _mm_slli_epi16::<1>(idx),
                );
                let u0 = _mm_blendv_epi8(t[0], t[1], s4);
                let u1 = _mm_blendv_epi8(t[2], t[3], s4);
                let u2 = _mm_blendv_epi8(t[4], t[5], s4);
                let u3 = _mm_blendv_epi8(t[6], t[7], s4);
                let v0 = _mm_blendv_epi8(u0, u1, s5);
                let v1 = _mm_blendv_epi8(u2, u3, s5);
                let byte = _mm_blendv_epi8(v0, v1, s6);
                let bit = _mm_shuffle_epi8(p2, _mm_and_si128(lo, k7));
                let hit = _mm_cmpeq_epi8(_mm_and_si128(byte, bit), bit);
                let mut m16 = _mm_movemask_epi8(hit) as u32;
                if b + 16 > nrow {
                    m16 &= (1u32 << (nrow - b)) - 1;
                }
                let c0 = _mm_unpacklo_epi8(lo, a0);
                let c1 = _mm_unpackhi_epi8(lo, a0);
                let (ml, mh) = (m16 & 0xFF, m16 >> 8);
                let at = pack(&mut s.hrow, np, pos, ml);
                pack(&mut s.hcode, np, c0, ml);
                let pos8 = _mm_add_epi16(pos, k8);
                np = pack(&mut s.hrow, at, pos8, mh);
                pack(&mut s.hcode, at, c1, mh);
                pos = _mm_add_epi16(pos8, k8);
                b += 16;
            }
        }
        if np == 0 {
            return Some(nd);
        }
        if np > max_hits {
            return None;
        }
        // ---- 4. layer 1, 8 hit rows at a time (see the NEON counter) ----
        grow(&mut s.rb_idx, np + 16);
        grow(&mut s.rb_d, np + 16);
        grow(&mut s.rb_r, np + 16);
        grow(&mut s.mrow, np + 16);
        grow(&mut s.mj, np + 16);
        let (mut nb, mut nm) = (0usize, 0usize);
        let mut extra = _mm_setzero_si128();
        {
            const IO: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let mut idx = ld(IO.as_ptr() as *const u8);
            let (k8, one) = (_mm_set1_epi16(8), _mm_set1_epi16(1));
            let koff = _mm_set1_epi16((off + K - 1) as i16);
            let kpos = _mm_set1_epi16(TAB_POS as i16);
            let kmulti = _mm_set1_epi16(TAB_MULTI as u16 as i16);
            let low16 = _mm_set1_epi32(0xFFFF);
            // Below 32768 (np <= WCAP), so the signed compare is the unsigned one.
            let vn = _mm_set1_epi16(np as i16);
            let (mut prev_r, mut prev_d) = (_mm_setzero_si128(), _mm_set1_epi16(-1));
            let tab = s.tab.as_ptr();
            // Codes past `np` are stale but in range (< 1024); their lanes are masked.
            let look4 = |c: *const u16| -> __m128i {
                let x = (c as *const u64).read_unaligned();
                let e = |sh: u32| u64::from(*tab.add(((x >> sh) & 1023) as usize));
                from2(e(0) | e(16) << 32, e(32) | e(48) << 32)
            };
            let mut k0 = 0usize;
            while k0 < np {
                let r = ld(s.hrow.as_ptr().add(k0) as *const u8);
                let e0 = look4(s.hcode.as_ptr().add(k0));
                let e1 = look4(s.hcode.as_ptr().add(k0 + 4));
                let lo = _mm_packus_epi32(_mm_and_si128(e0, low16), _mm_and_si128(e1, low16));
                let occ = _mm_packus_epi32(_mm_srli_epi32::<16>(e0), _mm_srli_epi32::<16>(e1));
                let j = _mm_and_si128(lo, kpos);
                let d = _mm_sub_epi16(_mm_add_epi16(r, koff), j);
                let rp = shift_in(prev_r, r);
                let dp = shift_in(prev_d, d);
                (prev_r, prev_d) = (r, d);
                let valid = _mm_cmplt_epi16(idx, vn);
                let cont = _mm_and_si128(
                    _mm_cmpeq_epi16(r, _mm_add_epi16(rp, one)),
                    _mm_cmpeq_epi16(d, dp),
                );
                let bd = _mm_andnot_si128(cont, valid);
                let mu = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(lo, kmulti), kmulti), valid);
                extra = _mm_add_epi32(
                    extra,
                    _mm_madd_epi16(_mm_and_si128(_mm_sub_epi16(occ, one), valid), one),
                );
                let m = _mm_movemask_epi8(_mm_packs_epi16(bd, mu)) as u32;
                let (mb, mm) = (m & 0xFF, m >> 8);
                pack(&mut s.rb_idx, nb, idx, mb);
                pack(&mut s.rb_d, nb, d, mb);
                nb = pack(&mut s.rb_r, nb, r, mb);
                pack(&mut s.mrow, nm, r, mm);
                nm = pack(&mut s.mj, nm, j, mm);
                idx = _mm_add_epi16(idx, k8);
                k0 += 8;
            }
        }
        let ex: [u32; 4] = std::mem::transmute(extra);
        let nhits = np + ex.iter().map(|&x| x as usize).sum::<usize>();
        s.nhits = nhits;
        if nhits > max_hits {
            return None;
        }
        if p.base() + p.a * nhits as i32 - p.c < p.minsc {
            return Some(nd);
        }
        s.cnt.clear();
        s.cnt.resize(span, 0);
        if s.minrow.len() < span {
            s.minrow.resize(span, 0);
        }
        // ---- 5. layer 1's runs, last to first; then the repeated codes' other occurrences ----
        s.rb_idx[nb] = np as u16;
        {
            let (cnt, minrow) = (s.cnt.as_mut_ptr(), s.minrow.as_mut_ptr());
            let (bd, bi, br) = (s.rb_d.as_ptr(), s.rb_idx.as_ptr(), s.rb_r.as_ptr());
            for x in (0..nb).rev() {
                let d = *bd.add(x) as usize;
                *cnt.add(d) += *bi.add(x + 1) - *bi.add(x);
                *minrow.add(d) = *br.add(x);
            }
        }
        let nxt = s.nxt.as_ptr();
        let nxt4 = |j: *const u16| -> u64 {
            let x = (j as *const u64).read_unaligned();
            let e = |sh: u32| u64::from(*nxt.add(((x >> sh) & 1023) as usize) as u16);
            e(0) | e(16) << 16 | e(32) << 32 | e(48) << 48
        };
        for x in 0..nm {
            s.mj[x] = *nxt.add(s.mj[x] as usize) as u16;
        }
        let mut n = nm;
        grow(&mut s.hrow, n + 16);
        grow(&mut s.hcode, n + 16);
        let (mut rows, mut js) = (std::mem::take(&mut s.mrow), std::mem::take(&mut s.mj));
        let (mut rows2, mut js2) = (std::mem::take(&mut s.hrow), std::mem::take(&mut s.hcode));
        {
            const IO: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let (k8, one, neg1) = (_mm_set1_epi16(8), _mm_set1_epi16(1), _mm_set1_epi16(-1));
            let koff = _mm_set1_epi16((off + K - 1) as i16);
            while n > 0 {
                let vn = _mm_set1_epi16(n as i16);
                let mut idx = ld(IO.as_ptr() as *const u8);
                let (mut prev_r, mut prev_d) = (_mm_setzero_si128(), _mm_set1_epi16(-1));
                let (mut nb, mut n2) = (0usize, 0usize);
                let mut k0 = 0usize;
                while k0 < n {
                    let r = ld(rows.as_ptr().add(k0) as *const u8);
                    let j = ld(js.as_ptr().add(k0) as *const u8);
                    let jn = from2(nxt4(js.as_ptr().add(k0)), nxt4(js.as_ptr().add(k0 + 4)));
                    let d = _mm_sub_epi16(_mm_add_epi16(r, koff), j);
                    let rp = shift_in(prev_r, r);
                    let dp = shift_in(prev_d, d);
                    (prev_r, prev_d) = (r, d);
                    let valid = _mm_cmplt_epi16(idx, vn);
                    let cont = _mm_and_si128(
                        _mm_cmpeq_epi16(r, _mm_add_epi16(rp, one)),
                        _mm_cmpeq_epi16(d, dp),
                    );
                    let keep = _mm_and_si128(_mm_cmpgt_epi16(jn, neg1), valid);
                    let m = _mm_movemask_epi8(_mm_packs_epi16(_mm_andnot_si128(cont, valid), keep))
                        as u32;
                    let (mb, mk) = (m & 0xFF, m >> 8);
                    pack(&mut s.rb_idx, nb, idx, mb);
                    pack(&mut s.rb_d, nb, d, mb);
                    nb = pack(&mut s.rb_r, nb, r, mb);
                    pack(&mut rows2, n2, r, mk);
                    n2 = pack(&mut js2, n2, jn, mk);
                    idx = _mm_add_epi16(idx, k8);
                    k0 += 8;
                }
                s.rb_idx[nb] = n as u16;
                let (cnt, minrow) = (s.cnt.as_mut_ptr(), s.minrow.as_mut_ptr());
                let (bd, bi, br) = (s.rb_d.as_ptr(), s.rb_idx.as_ptr(), s.rb_r.as_ptr());
                for x in 0..nb {
                    let (d, row) = (*bd.add(x) as usize, *br.add(x));
                    let c = *cnt.add(d);
                    let stale = u16::from(c == 0).wrapping_neg();
                    *minrow.add(d) = (*minrow.add(d) | stale).min(row);
                    *cnt.add(d) = c + *bi.add(x + 1) - *bi.add(x);
                }
                n = n2;
                std::mem::swap(&mut rows, &mut rows2);
                std::mem::swap(&mut js, &mut js2);
            }
        }
        (s.mrow, s.mj, s.hrow, s.hcode) = (rows, js, rows2, js2);
        s.occ_stale = true;
        Some(nd)
    }
}

/// The interval bounds and components as dense scans over every diagonal (fg-labs/bwa-mem3's
/// `lean_neon_core` steps 6 and 7), for jobs whose sums fit in `i16`. With `P` the prefix sum of the
/// weights `a cnt - c`, the bound is `base + max P[d..] - min(0, P[..d])` and the best Kadane sum is
/// `max (P[d] - min(0, P[..d]))`, as in [`sparse_decision`]. The diagonals are split into 8 segments
/// of `L` (a multiple of 8), one per lane, and an 8x8 transpose turns 8 consecutive diagonals of
/// every segment into 8 vectors, so each scan step is one vertical operation; the segments are
/// scanned from zero and combined through per-segment carries. A component at `tau` is then a run
/// of diagonals whose bound reaches `tau` holding a hit: [`components`] after [`extend`], which the
/// sparse path pins.
#[cfg(target_arch = "aarch64")]
mod dense {
    use super::{Comp, Params, Scratch};
    use std::arch::aarch64::*;

    /// Transpose 8 vectors of 8 lanes.
    #[inline(always)]
    unsafe fn tr8(v: &mut [int16x8_t; 8]) {
        let a: [int16x8_t; 8] = [
            vtrn1q_s16(v[0], v[1]),
            vtrn2q_s16(v[0], v[1]),
            vtrn1q_s16(v[2], v[3]),
            vtrn2q_s16(v[2], v[3]),
            vtrn1q_s16(v[4], v[5]),
            vtrn2q_s16(v[4], v[5]),
            vtrn1q_s16(v[6], v[7]),
            vtrn2q_s16(v[6], v[7]),
        ];
        let w = |x: int16x8_t| vreinterpretq_s32_s16(x);
        let b = [
            vtrn1q_s32(w(a[0]), w(a[2])),
            vtrn1q_s32(w(a[1]), w(a[3])),
            vtrn2q_s32(w(a[0]), w(a[2])),
            vtrn2q_s32(w(a[1]), w(a[3])),
            vtrn1q_s32(w(a[4]), w(a[6])),
            vtrn1q_s32(w(a[5]), w(a[7])),
            vtrn2q_s32(w(a[4]), w(a[6])),
            vtrn2q_s32(w(a[5]), w(a[7])),
        ];
        let d = |x: int32x4_t| vreinterpretq_s64_s32(x);
        let r = |x: int64x2_t| vreinterpretq_s16_s64(x);
        for k in 0..4 {
            v[k] = r(vtrn1q_s64(d(b[k]), d(b[k + 4])));
            v[k + 4] = r(vtrn2q_s64(d(b[k]), d(b[k + 4])));
        }
    }

    /// The bounds of every diagonal into `s.bnd`. `None` when a sum could leave `i16` (the caller
    /// then scans sparsely), `Some(false)` when no interval reaches `minsc` (the job fails),
    /// `Some(true)` otherwise.
    ///
    /// # Safety
    /// NEON; `s.cnt` holds the count of `nd` diagonals, zero up to the next multiple of 64.
    pub(super) unsafe fn bounds(s: &mut Scratch, p: &Params, nd: usize) -> Option<bool> {
        let l = nd.div_ceil(64) * 8;
        let nt = 8 * l;
        // |P| stays under a hits + c (NT + L): every value below fits `i16` with room to spare.
        if i64::from(p.a) * s.nhits as i64 + i64::from(p.c) * (nt + l) as i64 > 32000 {
            return None;
        }
        debug_assert!(s.cnt.len() >= nt && s.minrow.len() >= nt);
        for v in [&mut s.dp, &mut s.dm, &mut s.bnd] {
            if v.len() < nt {
                v.resize(nt, 0);
            }
        }
        let (zero, big, small) = (vdupq_n_s16(0), vdupq_n_s16(i16::MAX), vdupq_n_s16(i16::MIN));
        let (wa, wc) = (vdupq_n_s16(p.a as i16), vdupq_n_s16(p.c as i16));
        let cnt = s.cnt.as_ptr() as *const i16;
        let (dp, dm) = (s.dp.as_mut_ptr(), s.dm.as_mut_ptr());
        // Per segment: P before it (`cp`), min(0, P before it) (`mb`), max P after it (`xa`).
        let (cp, mb, xa);
        {
            let (mut pv, mut rm, mut mx, mut bestl) = (zero, big, small, small);
            let mut t = 0;
            while t < l {
                let mut c: [int16x8_t; 8] = std::array::from_fn(|k| vld1q_s16(cnt.add(k * l + t)));
                tr8(&mut c);
                for (k, &ck) in c.iter().enumerate() {
                    pv = vaddq_s16(pv, vsubq_s16(vmulq_s16(ck, wa), wc));
                    vst1q_s16(dp.add((t + k) * 8), pv);
                    // The segment's exclusive running minimum.
                    vst1q_s16(dm.add((t + k) * 8), rm);
                    bestl = vmaxq_s16(bestl, vqsubq_s16(pv, rm));
                    rm = vminq_s16(rm, pv);
                    mx = vmaxq_s16(mx, pv);
                }
                t += 8;
            }
            // Across lanes: exclusive prefix sum of the totals, exclusive prefix min (with 0) of
            // the minima, exclusive suffix max of the maxima.
            let mut x = pv;
            x = vaddq_s16(x, vextq_s16::<7>(zero, x));
            x = vaddq_s16(x, vextq_s16::<6>(zero, x));
            x = vaddq_s16(x, vextq_s16::<4>(zero, x));
            cp = vextq_s16::<7>(zero, x);
            let mut g = vaddq_s16(cp, rm);
            g = vminq_s16(g, vextq_s16::<7>(big, g));
            g = vminq_s16(g, vextq_s16::<6>(big, g));
            g = vminq_s16(g, vextq_s16::<4>(big, g));
            mb = vminq_s16(vextq_s16::<7>(zero, g), zero);
            let h = vaddq_s16(cp, mx);
            let mut hs = vmaxq_s16(h, vextq_s16::<1>(h, small));
            hs = vmaxq_s16(hs, vextq_s16::<2>(hs, small));
            hs = vmaxq_s16(hs, vextq_s16::<4>(hs, small));
            xa = vextq_s16::<1>(hs, small);
            let best = vmaxq_s16(bestl, vsubq_s16(h, mb));
            if p.base() + i32::from(vmaxvq_s16(best)) < p.minsc {
                return Some(false);
            }
        }
        // Backward, in each segment's own frame: bnd = base + max(sx, Xa - CP) - min(Mb - CP, TM).
        let basev = vdupq_n_s16(p.base() as i16);
        let ml = vsubq_s16(mb, cp);
        let mut sx = vqsubq_s16(xa, cp);
        let bnd = s.bnd.as_mut_ptr();
        let mut t = l;
        while t > 0 {
            t -= 8;
            let mut bd = [zero; 8];
            for k in (0..8).rev() {
                let o = (t + k) * 8;
                sx = vmaxq_s16(sx, vld1q_s16(dp.add(o)));
                bd[k] = vaddq_s16(basev, vsubq_s16(sx, vminq_s16(ml, vld1q_s16(dm.add(o)))));
            }
            tr8(&mut bd);
            for (k, &b) in bd.iter().enumerate() {
                vst1q_s16(bnd.add(k * l + t), b);
            }
        }
        Some(true)
    }

    /// The components at `tau`, from the bounds [`bounds`] left: each run of diagonals whose bound
    /// reaches `tau` and that holds a hit, with its best bound, earliest hit row and last hit
    /// diagonal. `off` shifts the diagonals back to `d = i - j`.
    ///
    /// # Safety
    /// NEON, after `bounds(s, _, nd)` returned `Some(true)`.
    pub(super) unsafe fn components(
        s: &mut Scratch,
        tau: i32,
        off: i32,
        nd: usize,
        out: &mut Vec<Comp>,
    ) {
        out.clear();
        let nw = nd.div_ceil(64);
        for v in [&mut s.mw, &mut s.hw] {
            v.clear();
            v.resize(nw, 0);
        }
        const BW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bw = vld1q_u8(BW.as_ptr());
        let tv = vdupq_n_s16(tau.clamp(-32768, 32767) as i16);
        let (bnd, cnt) = (s.bnd.as_ptr(), s.cnt.as_ptr());
        // 64 byte masks to 64 bits: weigh each byte by its bit, then three rounds of pairwise
        // adds fold the four vectors into bytes 0..8 in diagonal order.
        let bits64 = |g: [uint8x16_t; 4]| -> u64 {
            let x01 = vpaddq_u8(vandq_u8(g[0], bw), vandq_u8(g[1], bw));
            let x23 = vpaddq_u8(vandq_u8(g[2], bw), vandq_u8(g[3], bw));
            let x = vpaddq_u8(x01, x23);
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(x, x)))
        };
        for w in 0..nw {
            let (mut ge, mut gh) = ([vdupq_n_u8(0); 4], [vdupq_n_u8(0); 4]);
            for g in 0..4 {
                let at = w * 64 + g * 16;
                ge[g] = vcombine_u8(
                    vmovn_u16(vcgeq_s16(vld1q_s16(bnd.add(at)), tv)),
                    vmovn_u16(vcgeq_s16(vld1q_s16(bnd.add(at + 8)), tv)),
                );
                let (c0, c1) = (vld1q_u16(cnt.add(at)), vld1q_u16(cnt.add(at + 8)));
                let hit = vcombine_u8(vmovn_u16(vtstq_u16(c0, c0)), vmovn_u16(vtstq_u16(c1, c1)));
                gh[g] = vandq_u8(ge[g], hit);
            }
            s.mw[w] = bits64(ge);
            s.hw[w] = bits64(gh);
        }
        if !nd.is_multiple_of(64) {
            let keep = (1u64 << (nd % 64)) - 1;
            s.mw[nw - 1] &= keep;
            s.hw[nw - 1] &= keep;
        }
        // The first diagonal at or after `from` whose `bits ^ inv` bit is set, or `nd`.
        let next = |bits: &[u64], from: usize, inv: u64| -> usize {
            let mut w = from / 64;
            if w >= nw {
                return nd;
            }
            let mut x = (bits[w] ^ inv) & (!0u64 << (from % 64));
            loop {
                if x != 0 {
                    return (w * 64 + x.trailing_zeros() as usize).min(nd);
                }
                w += 1;
                if w >= nw {
                    return nd;
                }
                x = bits[w] ^ inv;
            }
        };
        const IO: [i16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let io = vld1q_s16(IO.as_ptr());
        let minrow = s.minrow.as_ptr();
        let mut d = next(&s.mw, 0, 0);
        while d < nd {
            let (a, b) = (d, next(&s.mw, d, !0));
            // The last hit diagonal of [a, b).
            let mut dmax = None;
            let mut w = (b - 1) / 64;
            loop {
                let lo_mask = if w == a / 64 { !0u64 << (a % 64) } else { !0 };
                let hi_mask = if w == (b - 1) / 64 {
                    !0u64 >> (63 - (b - 1) % 64)
                } else {
                    !0
                };
                let x = s.hw[w] & lo_mask & hi_mask;
                if x != 0 {
                    dmax = Some(w * 64 + 63 - x.leading_zeros() as usize);
                    break;
                }
                if w == a / 64 {
                    break;
                }
                w -= 1;
            }
            if let Some(dmax) = dmax {
                let (va, vb) = (vdupq_n_s16(a as i16), vdupq_n_s16(b as i16));
                let (mut mx, mut mn) = (vdupq_n_s16(i16::MIN), vdupq_n_u16(u16::MAX));
                let mut x = a & !7;
                while x < b {
                    let pos = vaddq_s16(io, vdupq_n_s16(x as i16));
                    let inr = vandq_u16(vcgeq_s16(pos, va), vcltq_s16(pos, vb));
                    let c = vld1q_u16(cnt.add(x));
                    mx = vmaxq_s16(
                        mx,
                        vbslq_s16(inr, vld1q_s16(bnd.add(x)), vdupq_n_s16(i16::MIN)),
                    );
                    mn = vminq_u16(
                        mn,
                        vbslq_u16(
                            vandq_u16(inr, vtstq_u16(c, c)),
                            vld1q_u16(minrow.add(x)),
                            vdupq_n_u16(u16::MAX),
                        ),
                    );
                    x += 8;
                }
                out.push(Comp {
                    ub: i32::from(vmaxvq_s16(mx)),
                    i0: i32::from(vminvq_u16(mn)),
                    dlo: a as i32 - off,
                    dhi: b as i32 - 1 - off,
                    dmaxhit: dmax as i32 - off,
                    kf: 0,
                    kl: 0,
                });
            }
            d = next(&s.mw, b, 0);
        }
    }
}

/// [`dense`] on x86 in SSSE3/SSE4.1: the same segments, scans and carries, vector for vector
/// (`int16x8_t` and `__m128i` hold the same eight lanes), so the same bounds and components.
#[cfg(target_arch = "x86_64")]
mod dense {
    use super::{Comp, Params, Scratch};
    use std::arch::x86_64::*;

    /// Transpose 8 vectors of 8 lanes.
    #[inline(always)]
    unsafe fn tr8(v: &mut [__m128i; 8]) {
        let a = [
            _mm_unpacklo_epi16(v[0], v[1]),
            _mm_unpackhi_epi16(v[0], v[1]),
            _mm_unpacklo_epi16(v[2], v[3]),
            _mm_unpackhi_epi16(v[2], v[3]),
            _mm_unpacklo_epi16(v[4], v[5]),
            _mm_unpackhi_epi16(v[4], v[5]),
            _mm_unpacklo_epi16(v[6], v[7]),
            _mm_unpackhi_epi16(v[6], v[7]),
        ];
        let b = [
            _mm_unpacklo_epi32(a[0], a[2]),
            _mm_unpackhi_epi32(a[0], a[2]),
            _mm_unpacklo_epi32(a[1], a[3]),
            _mm_unpackhi_epi32(a[1], a[3]),
            _mm_unpacklo_epi32(a[4], a[6]),
            _mm_unpackhi_epi32(a[4], a[6]),
            _mm_unpacklo_epi32(a[5], a[7]),
            _mm_unpackhi_epi32(a[5], a[7]),
        ];
        for k in 0..4 {
            v[2 * k] = _mm_unpacklo_epi64(b[k], b[k + 4]);
            v[2 * k + 1] = _mm_unpackhi_epi64(b[k], b[k + 4]);
        }
    }

    /// NEON's `vextq_s16::<N>(a, b)`: lanes `a[N..8]` then `b[0..N]`.
    macro_rules! ext {
        ($n:literal, $a:expr, $b:expr) => {
            _mm_alignr_epi8::<{ 2 * $n }>($b, $a)
        };
    }

    #[inline(always)]
    unsafe fn ld(p: *const i16) -> __m128i {
        _mm_loadu_si128(p as *const __m128i)
    }

    #[inline(always)]
    unsafe fn st(p: *mut i16, v: __m128i) {
        _mm_storeu_si128(p as *mut __m128i, v)
    }

    /// Signed `a >= b` per lane.
    #[inline(always)]
    unsafe fn cge(a: __m128i, b: __m128i) -> __m128i {
        _mm_cmpeq_epi16(_mm_max_epi16(a, b), a)
    }

    /// Horizontal signed max, through the unsigned horizontal min (32767 - v reverses the order).
    #[inline(always)]
    unsafe fn hmax(v: __m128i) -> i16 {
        let k = _mm_set1_epi16(i16::MAX);
        (32767 - i32::from(_mm_extract_epi16::<0>(_mm_minpos_epu16(_mm_sub_epi16(k, v))) as u16))
            as i16
    }

    /// See the NEON `bounds`; same results.
    ///
    /// # Safety
    /// SSSE3 and SSE4.1; `s.cnt` holds the count of `nd` diagonals, zero up to the next multiple
    /// of 64.
    #[target_feature(enable = "ssse3,sse4.1")]
    pub(super) unsafe fn bounds(s: &mut Scratch, p: &Params, nd: usize) -> Option<bool> {
        let l = nd.div_ceil(64) * 8;
        let nt = 8 * l;
        if i64::from(p.a) * s.nhits as i64 + i64::from(p.c) * (nt + l) as i64 > 32000 {
            return None;
        }
        debug_assert!(s.cnt.len() >= nt && s.minrow.len() >= nt);
        for v in [&mut s.dp, &mut s.dm, &mut s.bnd] {
            if v.len() < nt {
                v.resize(nt, 0);
            }
        }
        let (zero, big, small) = (
            _mm_setzero_si128(),
            _mm_set1_epi16(i16::MAX),
            _mm_set1_epi16(i16::MIN),
        );
        let (wa, wc) = (_mm_set1_epi16(p.a as i16), _mm_set1_epi16(p.c as i16));
        let cnt = s.cnt.as_ptr() as *const i16;
        let (dp, dm) = (s.dp.as_mut_ptr(), s.dm.as_mut_ptr());
        let (cp, mb, xa);
        {
            let (mut pv, mut rm, mut mx, mut bestl) = (zero, big, small, small);
            let mut t = 0;
            while t < l {
                let mut c: [__m128i; 8] = std::array::from_fn(|k| ld(cnt.add(k * l + t)));
                tr8(&mut c);
                for (k, &ck) in c.iter().enumerate() {
                    pv = _mm_add_epi16(pv, _mm_sub_epi16(_mm_mullo_epi16(ck, wa), wc));
                    st(dp.add((t + k) * 8), pv);
                    st(dm.add((t + k) * 8), rm);
                    bestl = _mm_max_epi16(bestl, _mm_subs_epi16(pv, rm));
                    rm = _mm_min_epi16(rm, pv);
                    mx = _mm_max_epi16(mx, pv);
                }
                t += 8;
            }
            let mut x = pv;
            x = _mm_add_epi16(x, ext!(7, zero, x));
            x = _mm_add_epi16(x, ext!(6, zero, x));
            x = _mm_add_epi16(x, ext!(4, zero, x));
            cp = ext!(7, zero, x);
            let mut g = _mm_add_epi16(cp, rm);
            g = _mm_min_epi16(g, ext!(7, big, g));
            g = _mm_min_epi16(g, ext!(6, big, g));
            g = _mm_min_epi16(g, ext!(4, big, g));
            mb = _mm_min_epi16(ext!(7, zero, g), zero);
            let h = _mm_add_epi16(cp, mx);
            let mut hs = _mm_max_epi16(h, ext!(1, h, small));
            hs = _mm_max_epi16(hs, ext!(2, hs, small));
            hs = _mm_max_epi16(hs, ext!(4, hs, small));
            xa = ext!(1, hs, small);
            let best = _mm_max_epi16(bestl, _mm_sub_epi16(h, mb));
            if p.base() + i32::from(hmax(best)) < p.minsc {
                return Some(false);
            }
        }
        let basev = _mm_set1_epi16(p.base() as i16);
        let ml = _mm_sub_epi16(mb, cp);
        let mut sx = _mm_subs_epi16(xa, cp);
        let bnd = s.bnd.as_mut_ptr();
        let mut t = l;
        while t > 0 {
            t -= 8;
            let mut bd = [zero; 8];
            for k in (0..8).rev() {
                let o = (t + k) * 8;
                sx = _mm_max_epi16(sx, ld(dp.add(o)));
                bd[k] = _mm_add_epi16(basev, _mm_sub_epi16(sx, _mm_min_epi16(ml, ld(dm.add(o)))));
            }
            tr8(&mut bd);
            for (k, &b) in bd.iter().enumerate() {
                st(bnd.add(k * l + t), b);
            }
        }
        Some(true)
    }

    /// See the NEON `components`; same results.
    ///
    /// # Safety
    /// SSSE3 and SSE4.1, after `bounds(s, _, nd)` returned `Some(true)`.
    #[target_feature(enable = "ssse3,sse4.1")]
    pub(super) unsafe fn components(
        s: &mut Scratch,
        tau: i32,
        off: i32,
        nd: usize,
        out: &mut Vec<Comp>,
    ) {
        out.clear();
        let nw = nd.div_ceil(64);
        for v in [&mut s.mw, &mut s.hw] {
            v.clear();
            v.resize(nw, 0);
        }
        let tv = _mm_set1_epi16(tau.clamp(-32768, 32767) as i16);
        let zero = _mm_setzero_si128();
        let (bnd, cnt) = (s.bnd.as_ptr(), s.cnt.as_ptr() as *const i16);
        for w in 0..nw {
            let (mut mwb, mut hwb) = (0u64, 0u64);
            for g in 0..4 {
                let at = w * 64 + g * 16;
                let ge = _mm_packs_epi16(cge(ld(bnd.add(at)), tv), cge(ld(bnd.add(at + 8)), tv));
                let empty = _mm_packs_epi16(
                    _mm_cmpeq_epi16(ld(cnt.add(at)), zero),
                    _mm_cmpeq_epi16(ld(cnt.add(at + 8)), zero),
                );
                mwb |= u64::from(_mm_movemask_epi8(ge) as u32 as u16) << (16 * g);
                hwb |= u64::from(_mm_movemask_epi8(_mm_andnot_si128(empty, ge)) as u32 as u16)
                    << (16 * g);
            }
            s.mw[w] = mwb;
            s.hw[w] = hwb;
        }
        if !nd.is_multiple_of(64) {
            let keep = (1u64 << (nd % 64)) - 1;
            s.mw[nw - 1] &= keep;
            s.hw[nw - 1] &= keep;
        }
        let next = |bits: &[u64], from: usize, inv: u64| -> usize {
            let mut w = from / 64;
            if w >= nw {
                return nd;
            }
            let mut x = (bits[w] ^ inv) & (!0u64 << (from % 64));
            loop {
                if x != 0 {
                    return (w * 64 + x.trailing_zeros() as usize).min(nd);
                }
                w += 1;
                if w >= nw {
                    return nd;
                }
                x = bits[w] ^ inv;
            }
        };
        const IO: [i16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let io = ld(IO.as_ptr());
        let minrow = s.minrow.as_ptr() as *const i16;
        let mut d = next(&s.mw, 0, 0);
        while d < nd {
            let (a, b) = (d, next(&s.mw, d, !0));
            let mut dmax = None;
            let mut w = (b - 1) / 64;
            loop {
                let lo_mask = if w == a / 64 { !0u64 << (a % 64) } else { !0 };
                let hi_mask = if w == (b - 1) / 64 {
                    !0u64 >> (63 - (b - 1) % 64)
                } else {
                    !0
                };
                let x = s.hw[w] & lo_mask & hi_mask;
                if x != 0 {
                    dmax = Some(w * 64 + 63 - x.leading_zeros() as usize);
                    break;
                }
                if w == a / 64 {
                    break;
                }
                w -= 1;
            }
            if let Some(dmax) = dmax {
                let (va, vb) = (_mm_set1_epi16(a as i16), _mm_set1_epi16(b as i16));
                let (mut mx, mut mn) = (_mm_set1_epi16(i16::MIN), _mm_set1_epi16(-1));
                let (smin, umax) = (_mm_set1_epi16(i16::MIN), _mm_set1_epi16(-1));
                let mut x = a & !7;
                while x < b {
                    let pos = _mm_add_epi16(io, _mm_set1_epi16(x as i16));
                    let inr = _mm_andnot_si128(cge(pos, vb), cge(pos, va));
                    let c = ld(cnt.add(x));
                    mx = _mm_max_epi16(mx, _mm_blendv_epi8(smin, ld(bnd.add(x)), inr));
                    let take = _mm_andnot_si128(_mm_cmpeq_epi16(c, zero), inr);
                    mn = _mm_min_epu16(mn, _mm_blendv_epi8(umax, ld(minrow.add(x)), take));
                    x += 8;
                }
                out.push(Comp {
                    ub: i32::from(hmax(mx)),
                    i0: i32::from(_mm_extract_epi16::<0>(_mm_minpos_epu16(mn)) as u16),
                    dlo: a as i32 - off,
                    dhi: b as i32 - 1 - off,
                    dmaxhit: dmax as i32 - off,
                    kf: 0,
                    kl: 0,
                });
            }
            d = next(&s.mw, b, 0);
        }
    }
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
    occ_ready(s);
    // With `P[d]` the sum of the weights `a cnt - c` through `d` (`P[-1] = 0`), the Kadane sum ending
    // at `d` is `f = P[d] - M[d-1]`, `M` the running minimum of `P` from `-1`, the one starting at `d`
    // is `b = X[d] - P[d-1]`, `X` the suffix maximum, and the interval bound is `base + X[d] - M[d-1]`.
    // Between two occupied diagonals `P` only falls, so `M[d-1]` is the minimum of the `Q = P[d-1]`
    // of the occupied diagonals up to `d`, and `X[d]` the maximum of their `P` from `d` on.
    let (base, a, c, minsc) = (p.base(), p.a, p.c, p.minsc);
    let n: usize = s.occ_bits.iter().map(|w| w.count_ones() as usize).sum();
    if n == 0 {
        // Only hit-free intervals, worth at most `base - c = (K-1)a < minsc`.
        return Decision::Fail;
    }
    if s.fw.len() < n {
        s.fw.resize(n, [0; 4]);
        s.hot.resize(n, [0; 4]);
    }
    let (mut pp, mut m, mut best, mut prev, mut k) = (0i32, 0i32, i32::MIN, -1i32, 0usize);
    for (w, &bits) in s.occ_bits.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            let d = (w * 64) as i32 + bits.trailing_zeros() as i32;
            bits &= bits - 1;
            let hit = a * s.cnt[d as usize] as i32;
            pp += hit - c * (d - prev);
            let q = pp - hit + c;
            m = m.min(q);
            best = best.max(pp - m);
            s.fw[k] = [d, pp, m, q];
            (prev, k) = (d, k + 1);
        }
    }
    debug_assert!((prev as usize) < nd);
    if base + best < minsc {
        return Decision::Fail;
    }
    // Backward: keep the diagonals whose bound reaches `minsc` (store always, advance on a pass).
    let (mut x, mut nh) = (i32::MIN, 0usize);
    for k in (0..n).rev() {
        let [_, pk, mk, _] = s.fw[k];
        x = x.max(pk);
        s.fw[k][3] = x;
        let bnd = base + x - mk;
        s.hot[nh] = [k as i32, x, bnd, 0];
        nh += (bnd >= minsc) as usize;
    }
    (s.n_occ, s.n_hot) = (n, nh);
    let off = (nd - len1 - 1) as i32;
    let mut c19 = std::mem::take(&mut s.c19);
    components(s, p, minsc, off, nd, false, &mut c19);
    s.c19 = c19;
    hull_of(&s.c19, p, off, len1)
}

/// Derive `occ_bits` if the counter left it to its reader.
fn occ_ready(s: &mut Scratch) {
    if s.occ_stale {
        s.occ_stale = false;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline; the counter sized `cnt` to the words.
        unsafe {
            neon::occ_from_cnt(s)
        };
        #[cfg(not(target_arch = "aarch64"))]
        for (w, bits) in s.occ_bits.iter_mut().enumerate() {
            *bits = occupied_word(&s.cnt[w * 64..w * 64 + 64]);
        }
    }
}

/// The decision from the per-diagonal counts: [`dense`]'s scan where it applies, else
/// [`sparse_decision`]. Both leave the components at `minsc` in `s.c19`; `s.dense` says which.
fn decide_counts(s: &mut Scratch, p: &Params, len1: usize, nd: usize) -> Decision {
    s.dense = false;
    // A stale `occ_bits` means the counter got past its early returns: some diagonal holds a hit.
    if dense_enabled() && (s.occ_stale || s.occ_bits.iter().any(|&w| w != 0)) {
        // SAFETY: `dense_enabled` holds NEON (aarch64 baseline) or SSSE3+SSE4.1 (checked on x86);
        // the count just filled `s` for `nd`.
        match unsafe { dense::bounds(s, p, nd) } {
            Some(false) => return Decision::Fail,
            Some(true) => {
                let off = (nd - len1 - 1) as i32;
                let mut c19 = std::mem::take(&mut s.c19);
                // SAFETY: as above, after `bounds`.
                unsafe { dense::components(s, p.minsc, off, nd, &mut c19) };
                s.c19 = c19;
                s.dense = true;
                return hull_of(&s.c19, p, off, len1);
            }
            None => {}
        }
    }
    sparse_decision(s, p, len1, nd)
}

/// `BWA4_PRUNE_DENSE=0` keeps the sparse scan everywhere (A/B handle). On x86 the scan also
/// needs SSSE3 and SSE4.1, which the dense calls below rely on.
fn dense_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if !sse::available() {
            return false;
        }
        std::env::var_os("BWA4_PRUNE_DENSE").is_none_or(|v| v != "0")
    })
}

/// The hull of the components at `minsc`, or `Fail` when there is none.
fn hull_of(c19: &[Comp], p: &Params, off: i32, len1: usize) -> Decision {
    let (mut lo, mut hi) = (len1 as i64, -1i64);
    for c in c19 {
        lo = lo.min(c.i0 as i64);
        // The last hit diagonal `dmaxhit` reaches row `dmaxhit + off - 1` at the last pad column
        // (`off` is the padded query length), then `tail` more rows through a deletion.
        hi = hi.max(c.dmaxhit as i64 + off as i64 - 1 + p.tail(c.ub) as i64);
    }
    if hi < 0 {
        return Decision::Fail;
    }
    Decision::Hull {
        hb: lo.max(0) as usize,
        he: hi.min(len1 as i64 - 1) as usize,
    }
}

/// A diagonal component at some threshold `tau`: the diagonals `dlo..=dhi` (`d = i - j`, unshifted)
/// whose interval bound reaches `tau`, its best bound, the earliest start row of its hits and its
/// last diagonal holding a hit. `kf..=kl` are its occupied diagonals' indices in `Scratch::fw`;
/// `dlo`/`dhi` cover the empty diagonals at its ends only after [`extend`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Comp {
    pub ub: i32,
    pub i0: i32,
    pub dlo: i32,
    pub dhi: i32,
    pub dmaxhit: i32,
    kf: usize,
    kl: usize,
}

/// The components at `tau >= minsc`, from the lists [`sparse_decision`] left in `s`: runs of
/// occupied diagonals whose bound reaches `tau`, joined across a gap whose bounds all reach `tau`.
/// On a gap `(d1, d2)` every bound is at least `base + X[d2] - M[d1]` (`X` and `M` only fall),
/// which settles most gaps without [`gap_min`].
///
/// Why this equals the dense scan's components:
/// * a component's best bound is reached on an occupied diagonal: the best interval through an
///   empty diagonal holds an occupied one (an all-empty interval is worth `base - c * len`, below
///   any valid `minsc`), and that interval lies inside the component, so the occupied diagonal's
///   bound is at least as high;
/// * a component without a hit contributes nothing (the dense scan skips it too);
/// * two consecutive occupied diagonals share a component exactly when every empty diagonal
///   between them reaches `tau`, and the bound across a gap is convex, so its minimum is at an
///   end or next to a breakpoint ([`gap_min`]).
fn components(
    s: &Scratch,
    p: &Params,
    tau: i32,
    off: i32,
    nd: usize,
    ext: bool,
    out: &mut Vec<Comp>,
) {
    let (base, a, c) = (p.base(), p.a, p.c);
    out.clear();
    let mut open: Option<Comp> = None;
    let (mut pk, mut pd, mut pm, mut pf) = (-2i32, 0i32, 0i32, 0i32);
    for &[k, x, bnd, _] in s.hot[..s.n_hot].iter().rev() {
        if bnd < tau {
            continue;
        }
        let [d, pk_, mk, _] = s.fw[k as usize];
        let row = i32::from(s.minrow[d as usize]);
        let qk = pk_ - a * i32::from(s.cnt[d as usize]) + c;
        open = match open {
            Some(cur)
                if k == pk + 1
                    && (base + x - pm >= tau
                        || gap_min(p, pf, x - qk, (d - pd - 1) as u32) >= tau) =>
            {
                Some(Comp {
                    ub: cur.ub.max(bnd),
                    i0: cur.i0.min(row),
                    dhi: d - off,
                    dmaxhit: d - off,
                    kl: k as usize,
                    ..cur
                })
            }
            prev => {
                if let Some(cur) = prev {
                    out.push(cur);
                }
                Some(Comp {
                    ub: bnd,
                    i0: row,
                    dlo: d - off,
                    dhi: d - off,
                    dmaxhit: d - off,
                    kf: k as usize,
                    kl: k as usize,
                })
            }
        };
        (pk, pd, pm, pf) = (k, d, mk.min(pk_), pk_ - mk);
    }
    if let Some(cur) = open {
        out.push(cur);
    }
    if ext {
        for cm in out.iter_mut() {
            extend(s, p, tau, nd, cm);
        }
    }
}

/// Widen `cm` to the empty diagonals at its two ends whose bound reaches `tau`: on the gap next to
/// an end the bound is convex ([`gap_min`]), so those diagonals are the longest run from the end
/// on which it stays `>= tau`, found by bisection on the monotone side of the minimum.
fn extend(s: &Scratch, p: &Params, tau: i32, nd: usize, cm: &mut Comp) {
    let (a, c) = (p.a, p.c);
    // The gap between occupied indices `kl` (`-1`: before the first) and `kr` (`n`: past the last):
    // its length and the clamped Kadane sums on either side.
    let gap = |kl: i32, kr: usize| -> (i32, i32, i32) {
        let (dl, pl) = if kl < 0 {
            (-1, 0)
        } else {
            let [d, pp, m, _] = s.fw[kl as usize];
            (d, (pp - m).max(0))
        };
        let (dr, qr) = if kr >= s.n_occ {
            (nd as i32, 0)
        } else {
            let [d, pp, _, x] = s.fw[kr];
            (d, (x - (pp - a * i32::from(s.cnt[d as usize]) + c)).max(0))
        };
        (dr - dl - 1, pl, qr)
    };
    let at = |pl: i32, qr: i32, g: i32, j: i32| {
        p.base() - c + (pl - (j - 1) * c).max(0) + (qr - (g - j) * c).max(0)
    };
    let argmin = |pl: i32, qr: i32, g: i32| {
        let (k1, k2) = (1 + pl / c, g - qr / c);
        [1, g, k1, k1 + 1, k2, k2 - 1]
            .into_iter()
            .map(|j| j.clamp(1, g))
            .min_by_key(|&j| at(pl, qr, g, j))
            .unwrap()
    };
    // Right of `kl`: the longest prefix `1..=j` of the gap with bound `>= tau`.
    let (g, pl, qr) = gap(cm.kl as i32, cm.kl + 1);
    if g > 0 && at(pl, qr, g, 1) >= tau {
        let jm = argmin(pl, qr, g);
        let mut j = if at(pl, qr, g, jm) >= tau { g } else { 1 };
        if j < g {
            // Bisection on `[1, jm)`, nonincreasing: `at(j) >= tau`, `at(jm) < tau`.
            let mut hi = jm;
            while hi - j > 1 {
                let mid = (j + hi) / 2;
                if at(pl, qr, g, mid) >= tau {
                    j = mid;
                } else {
                    hi = mid;
                }
            }
        }
        cm.dhi += j;
    }
    // Left of `kf`: the longest suffix `j..=g` with bound `>= tau`.
    let (g, pl, qr) = gap(cm.kf as i32 - 1, cm.kf);
    if g > 0 && at(pl, qr, g, g) >= tau {
        let jm = argmin(pl, qr, g);
        let mut j = if at(pl, qr, g, jm) >= tau { 1 } else { g };
        if j > 1 {
            // Bisection on `(jm, g]`, nondecreasing: `at(jm) < tau`, `at(j) >= tau`.
            let mut lo = jm;
            while j - lo > 1 {
                let mid = (lo + j) / 2;
                if at(pl, qr, g, mid) >= tau {
                    j = mid;
                } else {
                    lo = mid;
                }
            }
        }
        cm.dlo -= g - j + 1;
    }
}

/// The banded kernel's round-1 threshold `T1 = max(minsc, 5 * ub2 / 8)`, `ub2` the second-best
/// component bound. Speed only: round 1 is final when `S >= T1` and `score2 >= T1` (or the hull
/// lies in the `score2` exclusion zone), and round 2 reruns the rest at `minsc`, so a higher `T1`
/// trades narrower round-1 bands for more round-2 parents. Band cells on 100k GIAB pairs by the
/// fraction of `ub2`: 1/4 40.4G, 1/2 (fg-labs/bwa-mem3's) 32.3G, 9/16 31.6G, 3/5 31.4G, 5/8 31.4G,
/// 2/3 31.5G, 3/4 32.3G, 1 41.1G.
pub(crate) fn round1_threshold(minsc: i32, ub2: i32) -> i32 {
    minsc.max(ub2 * 5 / 8)
}

/// [`decide`], and on a `Hull` the components the banded kernel runs: `c19` at `minsc` and, when
/// the band threshold [`round1_threshold`] exceeds `minsc`, `ct` at it. Returns `T1`, or 0 when
/// the decision is not a hull.
pub(crate) fn decide_bands(
    t: &[u8],
    q: &[u8],
    p: &Params,
    max_hits: usize,
    c19: &mut Vec<Comp>,
    ct: &mut Vec<Comp>,
) -> (Decision, i32) {
    let (len1, len2, k) = (t.len(), q.len(), p.k);
    if len1 < k || len2 < k || len1 > WCAP || len2 > QCAP {
        return (Decision::Full, 0);
    }
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        let Some(nd) = count_hits(s, t, q, p, max_hits) else {
            return (Decision::Full, 0);
        };
        let dec = decide_counts(s, p, len1, nd);
        if !matches!(dec, Decision::Hull { .. }) {
            return (dec, 0);
        }
        let off = (nd - len1 - 1) as i32;
        let (mut ub1, mut ub2) = (0, 0);
        for c in &s.c19 {
            if c.ub > ub1 {
                (ub2, ub1) = (ub1, c.ub);
            } else if c.ub > ub2 {
                ub2 = c.ub;
            }
        }
        let t1 = round1_threshold(p.minsc, ub2);
        std::mem::swap(&mut s.c19, c19);
        if s.dense {
            // The dense components are the threshold's whole runs: already extended.
            if t1 > p.minsc {
                // SAFETY: `s.dense` means `dense_enabled` held (see `decide_counts`); `bounds` filled `s`.
                unsafe { dense::components(s, t1, off, nd, ct) };
            }
            return (dec, t1);
        }
        for cm in c19.iter_mut() {
            extend(s, p, p.minsc, nd, cm);
        }
        if t1 > p.minsc {
            components(s, p, t1, off, nd, true, ct);
        }
        (dec, t1)
    })
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
pub(crate) mod tests {
    use super::*;
    use bwa_extend::ksw_align2;

    /// The SIMD [`occupied_word`] sets bit `x` exactly when `cnt[x] != 0`, counts up to `u16::MAX`.
    #[test]
    fn occupied_word_marks_nonzero_counts() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..2000 {
            let cnt: Vec<u16> = (0..64)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    [0, 0, 1, 255, 256, u16::MAX, x as u16][(x >> 32) as usize % 7]
                })
                .collect();
            let want = (0..64).fold(0u64, |b, i| b | u64::from(cnt[i] != 0) << i);
            assert_eq!(occupied_word(&cnt), want);
        }
    }

    /// [`super::count_hits`] with `cnt` made dense: the counter leaves `cnt` stale where no
    /// `occ_bits` bit is set (production reads it only under one), these tests read every slot.
    fn count_hits(
        s: &mut Scratch,
        t: &[u8],
        q: &[u8],
        p: &Params,
        max_hits: usize,
    ) -> Option<usize> {
        let nd = super::count_hits(s, t, q, p, max_hits)?;
        occ_ready(s);
        s.cnt.resize(s.cnt.len().max(nd), 0);
        for d in 0..nd {
            if s.occ_bits[d / 64] >> (d % 64) & 1 == 0 {
                s.cnt[d] = 0;
            }
        }
        Some(nd)
    }

    pub(crate) fn scmat(a: i8, b: i8) -> Vec<i8> {
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
        // And every real job of a dump, when one is given.
        if let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") {
            let p = &params[0];
            for (n, (t, q)) in load_dump(&path).iter().enumerate() {
                if let Some(nd) = count_hits(&mut s, t, q, p, usize::MAX) {
                    let want = dense_decision(&s, p, t.len(), nd);
                    assert_eq!(sparse_decision(&mut s, p, t.len(), nd), want, "job {n}");
                }
            }
        }
    }

    /// The dense scan gives the sparse scan's decision and components (at `minsc` after
    /// [`extend`], and at the band threshold `T1`) on every job it takes.
    #[test]
    fn dense_scan_equals_sparse_scan() {
        let mut state = 0x0bad_cafe_1234_5678u64;
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
        let key = |c: &Comp| (c.ub, c.i0, c.dlo, c.dhi, c.dmaxhit);
        let mut s = Scratch::new();
        let check = |s: &mut Scratch, t: &[u8], q: &[u8], p: &Params, what: &str| -> u32 {
            let Some(nd) = count_hits(s, t, q, p, usize::MAX) else {
                return 0;
            };
            if s.occ_bits.iter().all(|&w| w == 0) {
                return 0;
            }
            let (len1, off) = (t.len(), (nd - t.len() - 1) as i32);
            let want = sparse_decision(s, p, len1, nd);
            let mut c19 = s.c19.clone();
            // SAFETY: NEON is part of the aarch64 baseline.
            let got = match unsafe { dense::bounds(s, p, nd) } {
                None => return 0,
                Some(false) => Decision::Fail,
                Some(true) => {
                    let mut d19 = Vec::new();
                    unsafe { dense::components(s, p.minsc, off, nd, &mut d19) };
                    let got = hull_of(&d19, p, off, len1);
                    if let Decision::Hull { .. } = want {
                        for cm in c19.iter_mut() {
                            extend(s, p, p.minsc, nd, cm);
                        }
                        assert_eq!(
                            c19.iter().map(key).collect::<Vec<_>>(),
                            d19.iter().map(key).collect::<Vec<_>>(),
                            "{what}: c19"
                        );
                        let (mut ub1, mut ub2) = (0, 0);
                        for c in &c19 {
                            if c.ub > ub1 {
                                (ub2, ub1) = (ub1, c.ub);
                            } else if c.ub > ub2 {
                                ub2 = c.ub;
                            }
                        }
                        let t1 = round1_threshold(p.minsc, ub2);
                        let (mut ct, mut dt) = (Vec::new(), Vec::new());
                        components(s, p, t1, off, nd, true, &mut ct);
                        unsafe { dense::components(s, t1, off, nd, &mut dt) };
                        assert_eq!(
                            ct.iter().map(key).collect::<Vec<_>>(),
                            dt.iter().map(key).collect::<Vec<_>>(),
                            "{what}: ct at {t1}"
                        );
                    }
                    got
                }
            };
            assert_eq!(got, want, "{what}");
            1
        };
        let mut taken = 0;
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
            taken += check(&mut s, &t, &q, p, &format!("iter {iter}"));
        }
        assert!(taken > 5000, "taken {taken}");
        if let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") {
            for (n, (t, q)) in load_dump(&path).iter().enumerate() {
                check(&mut s, t, q, &params[0], &format!("job {n}"));
            }
        }
    }

    /// Real rescue jobs dumped by `BWA4_RESCUE_DUMP` (u32 tlen, u32 qlen, target, query), for
    /// offline timing: `BWA4_PRUNE_BENCH=jobs.bin cargo test --release -p bwa-mem4-neon
    /// prune_bench -- --ignored --nocapture`.
    pub(crate) fn load_dump(path: &str) -> Vec<(Vec<u8>, Vec<u8>)> {
        let raw = std::fs::read(path).unwrap();
        let (mut out, mut at) = (Vec::new(), 0);
        while at + 8 <= raw.len() {
            let tl = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
            let ql = u32::from_le_bytes(raw[at + 4..at + 8].try_into().unwrap()) as usize;
            at += 8;
            out.push((
                raw[at..at + tl].to_vec(),
                raw[at + tl..at + tl + ql].to_vec(),
            ));
            at += tl + ql;
        }
        out
    }

    /// The per-hit enumeration the run-collapsed one must reproduce: `(cnt, minrow where cnt > 0)`.
    fn naive_counts(t: &[u8], q: &[u8], k: usize) -> (Vec<u16>, Vec<i32>) {
        let off = q.len().div_ceil(16) * 16;
        let nd = t.len() + off + 1;
        let (mut cnt, mut minrow) = (vec![0u16; nd], vec![i32::MAX; nd]);
        for i in k - 1..t.len() {
            for j in k - 1..q.len() {
                if t[i + 1 - k..=i] == q[j + 1 - k..=j] {
                    let d = i + off - j;
                    cnt[d] += 1;
                    minrow[d] = minrow[d].min((i + 1 - k) as i32);
                }
            }
        }
        for d in 0..nd {
            if cnt[d] == 0 {
                minrow[d] = 0;
            }
        }
        (cnt, minrow)
    }

    /// Run-collapsed counting equals per-hit counting, on generated jobs with planted runs,
    /// repeated query K-mers (tandem and poly-A mates) and, when `BWA4_PRUNE_BENCH` points at a
    /// dump, on the first real jobs of it.
    #[test]
    fn run_counts_equal_naive_counts() {
        let mut state = 0xabcd_ef01_2345_6789u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut jobs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for _ in 0..3000 {
            let qlen = 10 + next() % 150;
            let tlen = qlen + next() % 600;
            let alpha = [4, 4, 2, 1, 3][next() % 5].max(1);
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            let q: Vec<u8> = (0..qlen).map(|_| (next() % alpha) as u8).collect();
            for _ in 0..next() % 4 {
                let l = (5 + next() % 80).min(qlen);
                let (qs, at) = (next() % (qlen - l + 1), next() % (tlen - l + 1));
                t[at..at + l].copy_from_slice(&q[qs..qs + l]);
            }
            jobs.push((t, q));
        }
        if let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") {
            jobs.extend(load_dump(&path).into_iter().take(3000));
        }
        let mut s = Scratch::new();
        for k in [5usize, 7] {
            let p = Params {
                k,
                a: 1,
                c: 1,
                o_del: 6,
                e_del: 1,
                minsc: 1,
            };
            for (n, (t, q)) in jobs.iter().enumerate() {
                if t.len() < k || q.len() < k {
                    continue;
                }
                // A threshold of 1 so the early fail never skips the enumeration.
                let p = Params { minsc: 1, ..p };
                let Some(nd) = count_hits(&mut s, t, q, &p, usize::MAX) else {
                    continue;
                };
                let (cnt, minrow) = naive_counts(t, q, k);
                assert_eq!(&s.cnt[..nd], &cnt[..], "job {n}, k {k}: cnt");
                let got: Vec<i32> = (0..nd)
                    .map(|d| {
                        if s.cnt[d] > 0 {
                            i32::from(s.minrow[d])
                        } else {
                            0
                        }
                    })
                    .collect();
                assert_eq!(got, minrow, "job {n}, k {k}: minrow");
                for d in 0..nd {
                    let bit = s.occ_bits[d / 64] >> (d % 64) & 1;
                    assert_eq!(bit == 1, s.cnt[d] > 0, "job {n}, k {k}: occ bit {d}");
                }
            }
        }
    }

    /// The NEON counter against the scalar one: same verdict, same `cnt`, same `minrow` wherever
    /// `cnt != 0`, same `occ_bits`, on generated jobs (many repeats, poly-A, tiny lengths) and on
    /// the real dump when `BWA4_PRUNE_BENCH` names one.
    #[test]
    fn neon_counts_equal_scalar_counts() {
        let mut state = 0x5eed_1234_abcd_0001u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut jobs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for _ in 0..4000 {
            let qlen = 5 + next() % 200;
            let tlen = 5 + next() % 1500;
            let alpha = [4, 4, 2, 1, 3][next() % 5];
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % alpha) as u8).collect();
            let q: Vec<u8> = (0..qlen).map(|_| (next() % alpha) as u8).collect();
            for _ in 0..next() % 4 {
                let l = (5 + next() % 80).min(qlen).min(tlen);
                let (qs, at) = (next() % (qlen - l + 1), next() % (tlen - l + 1));
                t[at..at + l].copy_from_slice(&q[qs..qs + l]);
            }
            if next() % 20 == 0 {
                let at = next() % tlen;
                t[at] = 4;
            }
            jobs.push((t, q));
        }
        if let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") {
            jobs.extend(load_dump(&path));
        }
        let (mut a, mut b) = (Scratch::new(), Scratch::new());
        for minsc in [1, 19] {
            let p = Params {
                minsc,
                ..Params::from(1, 4, 6, 1, 6, 1, 19).unwrap()
            };
            for mh in [400usize, usize::MAX] {
                for (n, (t, q)) in jobs.iter().enumerate() {
                    if t.len() < 5 || q.len() < 5 || t.len() > WCAP || q.len() > QCAP {
                        continue;
                    }
                    let x = count_hits(&mut a, t, q, &p, mh);
                    let y = count_hits_scalar(&mut b, t, q, &p, mh);
                    assert_eq!(x, y, "job {n} minsc {minsc} mh {mh}: verdict");
                    let Some(nd) = x else { continue };
                    let nw = nd.div_ceil(64);
                    assert_eq!(a.occ_bits[..nw], b.occ_bits[..nw], "job {n}: occ_bits");
                    if a.occ_bits.iter().all(|&w| w == 0) {
                        continue;
                    }
                    assert_eq!(a.cnt[..nd], b.cnt[..nd], "job {n}: cnt");
                    for d in 0..nd {
                        if a.cnt[d] != 0 {
                            assert_eq!(a.minrow[d], b.minrow[d], "job {n}: minrow {d}");
                        }
                    }
                }
            }
        }
    }

    /// Per-row maxima of the full window's zero-state DP (gaps opened from H, query padded to 16
    /// columns of score 0), the rows `ksw_local_fwd` feeds its tracker. Default scoring.
    fn row_maxima(q: &[u8], t: &[u8]) -> Vec<i32> {
        let qp = q.len().div_ceil(16) * 16;
        let (mut hp, mut hc, mut e) = (vec![0i32; qp], vec![0i32; qp], vec![0i32; qp]);
        let mut out = Vec::with_capacity(t.len());
        for &tb in t {
            let (mut f, mut hd, mut im) = (0i32, 0i32, 0i32);
            for j in 0..qp {
                let sc = if j < q.len() {
                    if tb > 3 || q[j] > 3 {
                        -1
                    } else if tb == q[j] {
                        1
                    } else {
                        -4
                    }
                } else {
                    0
                };
                let h = (hd + sc).max(0).max(e[j]).max(f);
                im = im.max(h);
                hc[j] = h;
                e[j] = (e[j] - 1).max(h - 7).max(0);
                f = (f - 1).max(h - 7).max(0);
                hd = hp[j];
            }
            out.push(im);
            std::mem::swap(&mut hp, &mut hc);
        }
        out
    }

    /// The diagonal components at threshold `tau` from the counts `count_hits` left in `s`:
    /// `(ub, first row, first hit slot, last hit slot, tail rows)`, from the dense bound.
    fn components_at(
        s: &Scratch,
        p: &Params,
        nd: usize,
        tau: i32,
    ) -> Vec<(i32, i64, i64, i64, i64)> {
        let tail =
            |ub: i32| ((ub - tau - p.o_del + (p.k as i32 - 1) * p.a).max(0) / p.e_del) as i64;
        let (bnd, mut out) = (dense_bound(s, p, nd), Vec::new());
        let mut open: Option<(i32, i64, i64, i64)> = None;
        for d in 0..=nd {
            if d == nd || bnd[d] < tau {
                if let Some((ub, i0, lo, hi)) = open.take() {
                    if lo >= 0 {
                        out.push((ub, i0, lo, hi, tail(ub)));
                    }
                }
                continue;
            }
            let (mut ub, mut i0, mut lo, mut hi) = open.unwrap_or((bnd[d], i64::MAX, -1, -1));
            ub = ub.max(bnd[d]);
            if s.cnt[d] != 0 {
                i0 = i0.min(s.minrow[d] as i64);
                hi = d as i64;
                if lo < 0 {
                    lo = d as i64;
                }
            }
            open = Some((ub, i0, lo, hi));
        }
        out
    }

    // ---------------------------------------------------------------------------------------
    // Scalar prototype of banded rescue (the fork's rescue_band, against OUR score2 rule).
    // ---------------------------------------------------------------------------------------

    /// A component of the dense bound at some threshold: diagonal slots `[xlo, xhi]` (slot =
    /// `i - j + off`), best bound, earliest hit start row, last slot holding a hit.
    #[derive(Clone, Copy, Debug)]
    struct BComp {
        xlo: usize,
        xhi: usize,
        ub: i32,
        i0: i64,
        xmaxhit: usize,
    }

    /// Dense per-diagonal bound `base + fwd + bwd - w` from the counts `count_hits` left in `s`.
    fn dense_bound(s: &Scratch, p: &Params, nd: usize) -> Vec<i32> {
        let w = |x: usize| p.weight(s.cnt[x]);
        let (mut fw, mut bw) = (vec![0i32; nd], vec![0i32; nd]);
        for x in 0..nd {
            fw[x] = w(x) + if x > 0 { fw[x - 1].max(0) } else { 0 };
        }
        for x in (0..nd).rev() {
            bw[x] = w(x) + if x + 1 < nd { bw[x + 1].max(0) } else { 0 };
        }
        (0..nd).map(|x| p.base() + fw[x] + bw[x] - w(x)).collect()
    }

    fn bcomps(s: &Scratch, bnd: &[i32], tau: i32) -> Vec<BComp> {
        let mut out = Vec::new();
        let mut x = 0;
        while x < bnd.len() {
            if bnd[x] < tau {
                x += 1;
                continue;
            }
            let a = x;
            let (mut ub, mut i0, mut xm) = (i32::MIN, i64::MAX, usize::MAX);
            while x < bnd.len() && bnd[x] >= tau {
                ub = ub.max(bnd[x]);
                if s.cnt[x] > 0 {
                    i0 = i0.min(s.minrow[x] as i64);
                    xm = x;
                }
                x += 1;
            }
            if xm != usize::MAX {
                out.push(BComp {
                    xlo: a,
                    xhi: x - 1,
                    ub,
                    i0,
                    xmaxhit: xm,
                });
            }
        }
        out
    }

    /// Rows `[r0, r1]` and diagonals `[dlo, dhi]` (`d = i - j`) of a component's band at `tau`.
    fn band_of(
        c: &BComp,
        p: &Params,
        tau: i32,
        len1: usize,
        off: usize,
    ) -> Option<(i64, i64, i64, i64)> {
        let tail = {
            let x = c.ub - tau - p.o_del + (p.k as i32 - 1) * p.a;
            if x <= 0 {
                0
            } else {
                (x / p.e_del) as i64
            }
        };
        let off = off as i64;
        let r0 = c.i0.max(0);
        let r1 = (len1 as i64 - 1).min(c.xmaxhit as i64 - 1 + tail);
        let dlo = (c.xlo as i64 - off).max(r0 - off + 1);
        let dhi = (c.xhi as i64 - off).min(r1);
        let (r0, r1) = (r0.max(dlo), r1.min(dhi + off - 1));
        (r0 <= r1 && dlo <= dhi).then_some((r0, r1, dlo, dhi))
    }

    /// Zero-state DP restricted to one band, default scoring, query padded to `off` columns of
    /// score 0 (our kernels' 16-lane pad). Merges each row's max (and its smallest column) into
    /// `rmax` / `rcol`.
    fn band_dp(
        q: &[u8],
        t: &[u8],
        off: usize,
        b: (i64, i64, i64, i64),
        rmax: &mut [i32],
        rcol: &mut [i32],
    ) {
        let (r0, r1, dlo, dhi) = b;
        let (mut hp, mut hc, mut e) = (vec![0i32; off], vec![0i32; off], vec![0i32; off]);
        let inb = |i: i64, j: i64| j >= 0 && (j as usize) < off && (dlo..=dhi).contains(&(i - j));
        for i in r0..=r1 {
            let tb = t[i as usize];
            let (mut f, mut hd, mut im, mut ic) = (0i32, 0i32, 0i32, -1i32);
            for j in 0..off {
                if !inb(i, j as i64) {
                    // Nonexistent cell: H = E = F = 0.
                    hc[j] = 0;
                    e[j] = 0;
                    f = 0;
                    hd = hp[j];
                    continue;
                }
                let sc = if j < q.len() {
                    if tb > 3 || q[j] > 3 {
                        -1
                    } else if tb == q[j] {
                        1
                    } else {
                        -4
                    }
                } else {
                    0
                };
                let h = (hd + sc).max(0).max(e[j]).max(f);
                if h > im {
                    (im, ic) = (h, j as i32);
                }
                hc[j] = h;
                e[j] = (e[j] - 1).max(h - 7).max(0);
                f = (f - 1).max(h - 7).max(0);
                hd = hp[j];
            }
            let r = i as usize;
            if im > rmax[r] || (im == rmax[r] && im > 0 && ic < rcol[r]) {
                (rmax[r], rcol[r]) = (im, ic);
            }
            std::mem::swap(&mut hp, &mut hc);
        }
    }

    /// `(score, te, qe, score2, te2)` from merged row maxima, by `ksw_local_fwd`'s own rules.
    fn epilogue(rmax: &[i32], rcol: &[i32], minsc: i32) -> (i32, i32, i32, i32, i32) {
        let (mut g, mut te) = (0i32, -1i32);
        let mut b = bwa_extend::SuboptimalTracker::new();
        for (i, &v) in rmax.iter().enumerate() {
            b.push_row(i as i32, v, minsc);
            if v > g {
                (g, te) = (v, i as i32);
            }
        }
        let qe = if te >= 0 { rcol[te as usize] } else { -1 };
        let (s2, t2) = b.finish(g, te, 1);
        (g, te, qe, s2, t2)
    }

    /// The prototype against `ksw_local_fwd` on every `STRIDE`-th dumped job: round 2 (bands at
    /// `minsc`) must always match, and round 1 (bands at `T1`) must match whenever the fork's
    /// finality rule accepts it.
    ///
    /// Why that rule holds under OUR score2 (ksw_u8's, not kswv's): band row maxima never exceed the
    /// true ones and equal them on every row whose true max is `>= T1` (its alignment lies in a `T1`
    /// band). A tracker entry moves to row `r` only if `r` directly follows the stored row and is
    /// strictly higher, so whether a row `>= T1` ends up stored depends only on its neighbours' values
    /// when those are `>= T1` (exact on both sides) and otherwise on their being lower (true on both
    /// sides). So the entries `>= T1` are the same in both, as are `S`, `te` and `qe`; if `S >= T1`
    /// and the banded `score2 >= T1`, the true `score2`/`te2` are the same entry. If instead every row
    /// that can reach `minsc` lies inside the exclusion zone, neither side has an entry outside it.
    /// `BWA4_PRUNE_BENCH=jobs.bin STRIDE=50 cargo test --release -p bwa-mem4-neon band_proto -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn band_proto() {
        let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") else {
            return;
        };
        let stride: usize = std::env::var("STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        let jobs = load_dump(&path);
        let p = Params::from(1, 4, 6, 1, 6, 1, 19).unwrap();
        let mat: Vec<i8> = (0..25)
            .map(|k| {
                let (i, j) = (k / 5, k % 5);
                if i == 4 || j == 4 {
                    -1
                } else if i == j {
                    1
                } else {
                    -4
                }
            })
            .collect();
        let mut s = Scratch::new();
        let (mut n, mut r1n, mut fin, mut bad, mut r2_bad) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let (mut c_hull, mut c_band) = (0u64, 0u64);
        let consumed = |x: (i32, i32, i32, i32, i32)| if x.0 >= p.minsc { Some(x) } else { None };
        for (idx, (t, q)) in jobs.iter().enumerate() {
            if idx % stride != 0 || q.len() >= 250 {
                continue;
            }
            let off = q.len().div_ceil(16) * 16;
            let Some(nd) = count_hits(&mut s, t, q, &p, usize::MAX) else {
                continue;
            };
            let bnd = dense_bound(&s, &p, nd);
            let c0 = bcomps(&s, &bnd, p.minsc);
            if c0.is_empty() {
                continue;
            }
            n += 1;
            let full =
                bwa_extend::ksw_local_fwd(q, t, 5, &mat, 6, 1, 6, 1, p.minsc, i32::MAX, 1, 16);
            let run = |tau: i32, comps: &[BComp]| {
                let (mut rm, mut rc) = (vec![0i32; t.len()], vec![i32::MAX; t.len()]);
                let mut cells = 0u64;
                for c in comps {
                    if let Some(b) = band_of(c, &p, tau, t.len(), off) {
                        cells += ((b.1 - b.0 + 1) * (b.3 - b.2 + 1).min(off as i64)) as u64;
                        band_dp(q, t, off, b, &mut rm, &mut rc);
                    }
                }
                (rm, rc, cells)
            };
            let b0: Vec<_> = c0
                .iter()
                .filter_map(|c| band_of(c, &p, p.minsc, t.len(), off))
                .collect();
            let (hlo, hhi) = (
                b0.iter().map(|b| b.0).min().unwrap_or(0),
                b0.iter().map(|b| b.1).max().unwrap_or(-1),
            );
            c_hull += (hhi - hlo + 1).max(0) as u64 * off as u64;
            let mut ubs: Vec<i32> = c0.iter().map(|c| c.ub).collect();
            ubs.sort_unstable_by(|a, b| b.cmp(a));
            let t1 = p.minsc.max(ubs.get(1).copied().unwrap_or(0) / 2);
            let round2 = || {
                let (rm, rc, cells) = run(p.minsc, &c0);
                (epilogue(&rm, &rc, p.minsc), cells)
            };
            if t1 == p.minsc {
                let (r2, cells) = round2();
                r2_bad += (consumed(r2) != consumed(full)) as u64;
                c_band += cells;
                continue;
            }
            r1n += 1;
            let (rm1, rc1, cells1) = run(t1, &bcomps(&s, &bnd, t1));
            let r1 = epilogue(&rm1, &rc1, p.minsc);
            let (zl, zh) = (r1.1 as i64 - r1.0 as i64, r1.1 as i64 + r1.0 as i64);
            c_band += cells1;
            if r1.0 >= t1 && (r1.3 >= t1 || (hlo >= zl && hhi <= zh)) {
                fin += 1;
                if consumed(r1) != consumed(full) {
                    bad += 1;
                    eprintln!("ROUND 1 WRONG job {idx}: band {r1:?} full {full:?} T1 {t1}");
                }
            } else {
                let (r2, cells) = round2();
                r2_bad += (consumed(r2) != consumed(full)) as u64;
                c_band += cells;
            }
        }
        eprintln!(
            "PROTO jobs {n} (T1 > minsc {r1n}, round 1 final {fin}, wrong {bad})  round-2 mismatches {r2_bad}\n\
             band cells {:.1}% of the minsc hull",
            100.0 * c_band as f64 / c_hull.max(1) as f64
        );
        assert_eq!((bad, r2_bad), (0, 0));
    }

    /// Prices fg-labs/bwa-mem3's round-1 rescue bands offline (no kernel): per job, the hull our
    /// filter keeps against the diagonal bands of the components at `T1` ([`round1_threshold`]), and
    /// how many jobs round 1 cannot finish (checked with the scalar DP on every `STRIDE`-th job).
    /// `BWA4_PRUNE_BENCH=jobs.bin cargo test --release -p bwa-mem4-neon band_t1_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn band_t1_bench() {
        let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") else {
            return;
        };
        let stride: usize = std::env::var("STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let jobs = load_dump(&path);
        let p = Params::from(1, 4, 6, 1, 6, 1, 19).unwrap();
        let mat = {
            let mut m = [0i8; 25];
            for i in 0..5 {
                for j in 0..5 {
                    m[i * 5 + j] = if i == 4 || j == 4 {
                        -1
                    } else if i == j {
                        1
                    } else {
                        -4
                    };
                }
            }
            m
        };
        let mut s = Scratch::new();
        let (mut c_all, mut c_now, mut c_band, mut c_band_r2) = (0u64, 0u64, 0u64, 0u64);
        let mut nzone = 0u64;
        let mut nsafe = 0u64;
        let (mut n, mut nhull, mut nr1, mut nfinal, mut nchk, mut nsingle) =
            (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for (idx, (t, q)) in jobs.iter().enumerate() {
            if idx % stride != 0 {
                continue;
            }
            n += 1;
            let ql = q.len() as u64;
            let cells = t.len() as u64 * ql;
            c_all += cells;
            let mh: usize = std::env::var("MH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(400);
            let dec = decide(t, q, &p, mh);
            let (hb, he) = match dec {
                Decision::Fail => continue,
                Decision::Full => {
                    c_now += cells;
                    c_band += cells;
                    c_band_r2 += cells;
                    continue;
                }
                Decision::Hull { hb, he } => (hb, he),
            };
            nhull += 1;
            let hull = (he + 1 - hb) as u64 * ql;
            c_now += hull;
            // Rebuild the counts and Kadane sums (decide's scratch is thread-local, not `s`).
            let nd = count_hits(&mut s, t, q, &p, usize::MAX).unwrap();
            let c0 = components_at(&s, &p, nd, p.minsc);
            let mut ubs: Vec<i32> = c0.iter().map(|c| c.0).collect();
            ubs.sort_unstable_by(|a, b| b.cmp(a));
            let ub2 = ubs.get(1).copied().unwrap_or(0);
            if c0.len() == 1 {
                nsingle += 1;
            }
            let t1 = round1_threshold(p.minsc, ub2);
            let band_cells = |cs: &[(i32, i64, i64, i64, i64)]| -> u64 {
                cs.iter().map(|c| (c.3 - c.2 + 1 + c.4) as u64 * ql).sum()
            };
            let b1 = band_cells(&components_at(&s, &p, nd, t1));
            let b0 = band_cells(&c0);
            // Round 1 is final iff T1 == minsc, or S >= T1 and (score2 >= T1 or the hull lies in
            // the zone around te); otherwise round 2 adds the components at minsc.
            let mut need_r2 = false;
            if t1 > p.minsc {
                nr1 += 1;
                if idx % (stride * 4) == 0 {
                    nchk += 1;
                    let lanes = if ql < 250 { 16 } else { 8 };
                    let r = ksw_align2(q, t, 5, &mat, 6, 1, 6, 1, p.minsc, 1, lanes);
                    let z = r.score as i64;
                    let in_zone = hb as i64 >= r.te as i64 - z && he as i64 <= r.te as i64 + z;
                    // Sound variant: no stretch of rows covered by a minsc component crosses the
                    // zone's edge, so no true row run can merge across it.
                    let (zl, zh) = (r.te as i64 - z, r.te as i64 + z);
                    let mut spans: Vec<(i64, i64)> =
                        c0.iter().map(|c| (c.1, c.3 - 1 + c.4)).collect();
                    spans.sort_unstable();
                    let mut crossing = false;
                    let mut cur: Option<(i64, i64)> = None;
                    for (a0, a1) in spans
                        .into_iter()
                        .chain(std::iter::once((i64::MAX, i64::MAX)))
                    {
                        match cur {
                            Some((c0s, c1)) if a0 <= c1 + 1 => cur = Some((c0s, c1.max(a1))),
                            Some((c0s, c1)) => {
                                if (c0s < zl && c1 >= zl) || (c0s <= zh && c1 > zh) {
                                    crossing = true;
                                }
                                cur = Some((a0, a1));
                            }
                            None => cur = Some((a0, a1)),
                        }
                    }
                    // Refined: unsafe only if a crossing stretch holds a row >= T1 outside the zone.
                    let rm = row_maxima(q, t);
                    let mut spans2: Vec<(i64, i64)> =
                        c0.iter().map(|c| (c.1, c.3 - 1 + c.4)).collect();
                    spans2.sort_unstable();
                    let mut merged: Vec<(i64, i64)> = Vec::new();
                    for (a0, a1) in spans2 {
                        match merged.last_mut() {
                            Some(m) if a0 <= m.1 + 1 => m.1 = m.1.max(a1),
                            _ => merged.push((a0, a1)),
                        }
                    }
                    let unsafe_cross = merged.iter().any(|&(s0, s1)| {
                        let crosses = (s0 < zl && s1 >= zl) || (s0 <= zh && s1 > zh);
                        crosses
                            && (s0.max(0)..=s1.min(t.len() as i64 - 1))
                                .any(|row| (row < zl || row > zh) && rm[row as usize] >= t1)
                    });
                    if r.score >= t1 && r.score2 >= t1 && !unsafe_cross {
                        nsafe += 1;
                    }
                    let _ = crossing;
                    if r.score >= t1 && r.score2 >= t1 && !crossing {
                        nzone += 1;
                    }
                    if r.score >= t1 && (r.score2 >= t1 || in_zone) {
                        nfinal += 1;
                    } else {
                        need_r2 = true;
                    }
                }
            }
            c_band += b1.min(hull);
            c_band_r2 += if need_r2 {
                (b1 + b0).min(hull + b1)
            } else {
                b1.min(hull)
            };
        }
        let pc = |x: u64| 100.0 * x as f64 / c_all as f64;
        eprintln!(
            "BAND jobs {n} hull {nhull} single-comp {nsingle} T1>minsc {nr1} (checked {nchk}, final {nfinal}, no crossing {nzone}, no high crossing {nsafe})\n\
             cells of all: now {:.1}%  band-r1 {:.1}%  band with sampled r2 {:.1}%",
            pc(c_now),
            pc(c_band),
            pc(c_band_r2)
        );
    }

    #[test]
    #[ignore]
    fn prune_bench() {
        let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") else {
            return;
        };
        let jobs = load_dump(&path);
        let p = Params::from(1, 4, 6, 1, 6, 1, 19).unwrap();
        // `BWA4_PRUNE_BENCH_MH=1000,1000`: only these gates (one profile per gate).
        let mhs: Vec<usize> = std::env::var("BWA4_PRUNE_BENCH_MH")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_else(|| vec![0, 400, 1000, 100_000]);
        for mh in mhs {
            for _rep in 0..2 {
                let t0 = std::time::Instant::now();
                let (mut fail, mut hull, mut full) = (0, 0, 0);
                // DP cells (target rows x query length): all of them, then those each decision keeps.
                let (mut c_all, mut c_hull_in, mut c_hull_kept, mut c_full) =
                    (0u64, 0u64, 0u64, 0u64);
                for (t, q) in &jobs {
                    let cells = (t.len() * q.len()) as u64;
                    c_all += cells;
                    match decide(t, q, &p, mh) {
                        Decision::Fail => fail += 1,
                        Decision::Hull { hb, he } => {
                            hull += 1;
                            c_hull_in += cells;
                            c_hull_kept += ((he + 1 - hb) * q.len()) as u64;
                        }
                        Decision::Full => {
                            full += 1;
                            c_full += cells;
                        }
                    }
                }
                let ns = t0.elapsed().as_nanos() as f64 / jobs.len() as f64;
                let pc = |x: u64| 100.0 * x as f64 / c_all as f64;
                eprintln!(
                    "BENCH mh={mh:>6} {ns:7.0} ns/job  fail {fail} hull {hull} full {full}  cells: \
                     full {:.1}%  hull jobs {:.1}% -> kept {:.1}%  run {:.1}%",
                    pc(c_full),
                    pc(c_hull_in),
                    pc(c_hull_kept),
                    pc(c_full + c_hull_kept)
                );
            }
        }
        if std::env::var_os("BWA4_PRUNE_BENCH_MH").is_some() {
            return;
        }
        // The query-table rebuild alone, in the dump's order (as production sees it).
        {
            let mut s = Scratch::new();
            let t0 = std::time::Instant::now();
            for (_, q) in &jobs {
                s.load_query(q, 5);
            }
            eprintln!(
                "BENCH load_query: {:.0} ns/job",
                t0.elapsed().as_nanos() as f64 / jobs.len() as f64
            );
            let t0 = std::time::Instant::now();
            let mut acc = 0u8;
            for (t, q) in &jobs {
                acc |= t.iter().chain(q.iter()).fold(0u8, |a, &b| a | b);
            }
            std::hint::black_box(acc);
            eprintln!(
                "BENCH N scan: {:.0} ns/job",
                t0.elapsed().as_nanos() as f64 / jobs.len() as f64
            );
        }
        // Phase split at an open gate: counting alone (gate + enumeration) vs the whole decision.
        let mut s = Scratch::new();
        let t0 = std::time::Instant::now();
        let mut nds = Vec::with_capacity(jobs.len());
        for (t, q) in &jobs {
            nds.push(super::count_hits(&mut s, t, q, &p, 100_000));
        }
        let count_ns = t0.elapsed().as_nanos() as f64 / jobs.len() as f64;
        let t0 = std::time::Instant::now();
        for ((t, q), _) in jobs.iter().zip(&nds) {
            if let Some(nd) = super::count_hits(&mut s, t, q, &p, 100_000) {
                std::hint::black_box(sparse_decision(&mut s, &p, t.len(), nd));
            }
        }
        let all_ns = t0.elapsed().as_nanos() as f64 / jobs.len() as f64;
        eprintln!("BENCH split: count_hits {count_ns:.0} ns/job, + sparse = {all_ns:.0} ns/job");
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
