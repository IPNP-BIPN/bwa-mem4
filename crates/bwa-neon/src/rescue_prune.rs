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
    /// The query the tables describe, and its K.
    qcache: Vec<u8>,
    qk: usize,
    /// Per diagonal: hit count, first row of its first hit, Kadane sums ending / starting there.
    cnt: Vec<u16>,
    minrow: Vec<i32>,
    fwd: Vec<i32>,
    bwd: Vec<i32>,
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
            cnt: Vec::new(),
            minrow: Vec::new(),
            fwd: Vec::new(),
            bwd: Vec::new(),
        }
    }

    /// Index the query's K-mers, unless the tables already describe this query at this K.
    fn load_query(&mut self, q: &[u8], k: usize) {
        if self.qk == k && self.qcache == q {
            return;
        }
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
        self.qcache.clear();
        self.qcache.extend_from_slice(q);
        self.qk = k;
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
    if t.iter().chain(q).fold(0u8, |acc, &b| acc | b) & !3 != 0 {
        return Decision::Full;
    }
    SCRATCH.with(|s| decide_in(&mut s.borrow_mut(), t, q, p, max_hits))
}

fn decide_in(s: &mut Scratch, t: &[u8], q: &[u8], p: &Params, max_hits: usize) -> Decision {
    let (len1, len2, k) = (t.len(), q.len(), p.k);
    s.load_query(q, k);
    let mask = ((1u64 << (2 * k)) - 1) as u32;
    let mut code0 = 0u32;
    for &b in &t[..k - 1] {
        code0 = (code0 << 2) | b as u32;
    }

    // ---- Hit gate, from per-code counts, without enumerating the hits ----
    let mut code = code0;
    let mut nhits = 0usize;
    for &b in &t[k - 1..] {
        code = ((code << 2) | b as u32) & mask;
        nhits += s.qcnt[code as usize] as usize;
    }
    if nhits > max_hits {
        return Decision::Full;
    }
    // Early fail: every interval is non-empty and sums `a * cnt - c` over its diagonals, so no
    // interval can beat `base + a * nhits - c`.
    if p.base() + p.a * nhits as i32 - p.c < p.minsc {
        return Decision::Fail;
    }

    // ---- Per-diagonal hit counts. Diagonal `d = i - j` is stored at `d + off`, with `off` the
    //      query rounded up to 16 so the pad columns' diagonals have slots too ----
    let off = len2.div_ceil(16) * 16;
    let nd = len1 + off + 1;
    s.cnt.clear();
    s.cnt.resize(nd, 0);
    s.minrow.resize(nd, 0);
    s.fwd.resize(nd, 0);
    s.bwd.resize(nd, 0);
    code = code0;
    for i in k - 1..len1 {
        code = ((code << 2) | t[i] as u32) & mask;
        let mut j = s.head[code as usize];
        while j >= 0 {
            let d = i + off - j as usize;
            if s.cnt[d] == 0 {
                // Rows ascend, so the first hit seen on a diagonal is its earliest; its K-mer
                // starts K - 1 rows above the row where it ends.
                s.minrow[d] = (i + 1 - k) as i32;
            }
            s.cnt[d] += 1;
            j = s.nxt[j as usize];
        }
    }

    // ---- Kadane: best interval sums ending at (fwd) and starting at (bwd) each diagonal ----
    let base = p.base();
    let mut best = -p.c;
    let mut f = 0i32;
    for d in 0..nd {
        f = p.weight(s.cnt[d]) + f.max(0);
        s.fwd[d] = f;
        best = best.max(f);
    }
    if base + best < p.minsc {
        return Decision::Fail;
    }
    let mut b = 0i32;
    for d in (0..nd).rev() {
        b = p.weight(s.cnt[d]) + b.max(0);
        s.bwd[d] = b;
    }

    // ---- Components: maximal runs of diagonals whose best interval through them reaches
    //      `minsc`. The hull spans from the earliest hit row of any component to the last row its
    //      last hit diagonal can reach, through the pad columns and a deletion tail ----
    // `bound(d)`: the best interval bound over intervals containing diagonal `d`.
    let bound = |s: &Scratch, d: usize| base + s.fwd[d] + s.bwd[d] - p.weight(s.cnt[d]);
    let (mut lo, mut hi) = (len1 as i64, -1i64);
    let mut d = 0;
    while d < nd {
        if bound(s, d) < p.minsc {
            d += 1;
            continue;
        }
        let (mut ub, mut i0, mut dmax) = (0i32, len1 as i64, -1i64);
        while d < nd {
            let bnd = bound(s, d);
            if bnd < p.minsc {
                break;
            }
            ub = ub.max(bnd);
            if s.cnt[d] != 0 {
                i0 = i0.min(s.minrow[d] as i64);
                dmax = d as i64;
            }
            d += 1;
        }
        if dmax < 0 {
            continue;
        }
        lo = lo.min(i0);
        // The last hit diagonal `dmax - off` reaches row `(dmax - off) + (off - 1)` at the last pad
        // column (`off` is the padded query length), then `tail` more rows through a deletion.
        hi = hi.max(dmax - 1 + p.tail(ub) as i64);
    }
    if hi < 0 {
        return Decision::Fail;
    }
    Decision::Hull {
        hb: lo.max(0) as usize,
        he: hi.min(len1 as i64 - 1) as usize,
    }
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
    /// The whole rescue entry point on one mixed batch: wherever `ksw_align2` scores `>= minsc`,
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
            let got = crate::batched_mate_rescue(&jobs, 5, &mat, o, e, o, e, minsc, a as i32);
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
