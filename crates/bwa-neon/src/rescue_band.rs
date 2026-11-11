//! Banded mate rescue: pass 0 of a pruned rescue job runs only inside the diagonal bands of its
//! K-mer components, and the hull job's `(score, te, qe, score2, te2)` is reassembled from the
//! bands' per-row maxima. The method is fg-labs/bwa-mem3's `rescue_band` (argument in
//! `docs/superpowers/plans/2026-11-06-catch-the-fork.md`, prototype `band_proto` in
//! `rescue_prune.rs`):
//!
//! * every alignment scoring `>= tau` lies inside one component at `tau`, starts at or after the
//!   component's first hit row and ends by `dmaxhit + quanta - 1 + tail(ub, tau)`;
//! * a zero-state DP restricted to any cell set never exceeds the full DP and is exact on every
//!   cell of an alignment it contains, so per-row maxima over the bands are `<=` the true ones and
//!   equal wherever an alignment scoring `>= tau` ends;
//! * round 1 runs the components at `T1 = max(minsc, ub2 / 2)`; its merged outputs are exact iff
//!   `T1 == minsc`, or `S >= T1` and either `score2 >= T1` or the whole hull lies in the `score2`
//!   exclusion zone around `te`. Otherwise round 2 runs the components at `minsc`.
//!
//! The kernel works in band coordinates `k = r + o - j` (`o = r0 - dlo`): the diagonal predecessor
//! of cell `(r, k)` is `(r - 1, k)`, the vertical one `(r - 1, k - 1)`, the in-row one `(r, k + 1)`,
//! so H and E are indexed by `k` alone and a row costs its band width, not the query length.
//! Sixteen band jobs share a vector, grouped by width class and row count; a lane widened to its
//! group's width only gains cells, which keeps every property above.
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use crate::matesw::KswJob;
use crate::rescue_prune::{self, Comp, Decision, Params};
use bwa_extend::KswAlignResult;

/// The band kernel of this architecture: NEON (16 lanes) or AVX2 (32 lanes), the same `Work`.
#[cfg(target_arch = "aarch64")]
use neon as kernel;
#[cfg(target_arch = "x86_64")]
use x86 as kernel;

/// Whether the kernel takes these gap costs: the x86 kernels compute in signed bytes, so each
/// open-plus-extend must stay under 128 (NEON's unsigned lanes take any byte).
pub(crate) fn gaps_ok(o_del: i32, e_del: i32, o_ins: i32, e_ins: i32) -> bool {
    cfg!(target_arch = "aarch64") || (o_del + e_del).max(o_ins + e_ins) < 128
}

/// Whether this machine runs the band kernel.
pub(crate) fn kernel_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
}

/// A band in full-window coordinates: target rows `r0..=r1`, diagonals `dlo..=dhi` (`d = i - j`).
#[derive(Clone, Copy, Debug)]
struct Band {
    r0: i32,
    r1: i32,
    dlo: i32,
    dhi: i32,
    ub: i32,
}

/// A banded rescue job: its hull, threshold, band lists (round 1: `b0..b0+nb`; round 2:
/// `c0..c0+nc`) and the state the rounds fill.
struct Parent {
    job: u32,
    hb: i32,
    he: i32,
    t1: i32,
    b0: u32,
    nb: u32,
    c0: u32,
    nc: u32,
    s: i32,
    te: i32,
    qe: i32,
    s2: i32,
    te2: i32,
    /// Offset of the parent's merged row-max buffer (hull rows) in `pool`, or -1 (one band).
    buf: i32,
    round: u8,
}

/// One band to run: its parent and band, the group sort key.
struct Job {
    parent: u32,
    band: Band,
    key: u32,
}

/// Fixed per-row cost of a band lane, in cells, for the cost gate.
const OVH: i64 = 8;
/// More components than this at `minsc`: keep the hull.
const COMP_CAP: usize = 64;
/// Rows of H the band kernels keep: a row's `qe` scan waits until the row is about to be
/// overwritten, and is skipped for the lanes that improved again by then (most of them: a
/// mismatch costs a run about six rows). A multiple of 4, under 256.
const RING: usize = 16;

/// Per-thread banding state, reused across kernel calls.
#[derive(Default)]
pub(crate) struct Batch {
    bands: Vec<Band>,
    parents: Vec<Parent>,
    jobs: Vec<Job>,
    /// Jobs in group order, as `key << 32 | index` words: a primitive sort, with no key lookup
    /// per comparison. The index is the low half.
    order: Vec<u64>,
    r2: Vec<u32>,
    pool: Vec<u8>,
    c19: Vec<Comp>,
    ct: Vec<Comp>,
    work: kernel::Work,
}

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("BWA4_RESCUE_BAND").is_none_or(|v| v != "0"))
}

/// The band of component `c` at threshold `tau`, or `None` when it leaves the hull or holds no
/// cell (the parent then keeps the hull path).
fn to_band(
    c: &Comp,
    p: &Params,
    tau: i32,
    len1: i32,
    quanta: i32,
    hb: i32,
    he: i32,
) -> Option<Band> {
    let i0 = c.i0.max(0);
    let i1 = (len1 - 1).min(c.dmaxhit + quanta - 1 + p.tail_at(c.ub, tau));
    if i0 < hb || i1 > he || i0 > i1 {
        return None;
    }
    // Cells with `j >= quanta` (below `dlo`) and `j < 0` (above `dhi`) do not exist.
    let dlo = c.dlo.max(i0 - quanta + 1);
    let dhi = c.dhi.min(i1);
    if dlo > dhi {
        return None;
    }
    // Rows with no live cell: `r < dlo` (every `j < 0`) and `r > dhi + quanta - 1`.
    let (r0, r1) = (i0.max(dlo), i1.min(dhi + quanta - 1));
    (r0 <= r1).then_some(Band {
        r0,
        r1,
        dlo,
        dhi,
        ub: c.ub,
    })
}

/// Width class of a band of `w` diagonals: classes of 4 up to 16, of 8 up to 64, of 16 up to 128,
/// of 32 beyond. Lanes of one class share their live-cell clipping.
fn width_bucket(w: i32) -> u32 {
    (if w <= 16 {
        (w + 3) >> 2
    } else if w <= 64 {
        4 + ((w - 16 + 7) >> 3)
    } else if w <= 128 {
        10 + ((w - 64 + 15) >> 4)
    } else {
        14 + ((w - 128 + 31) >> 5)
    }) as u32
}

impl Batch {
    pub(crate) fn clear(&mut self) {
        self.bands.clear();
        self.parents.clear();
    }

    /// The prune decision of job `idx` (window `t`, mate `q`, padded to `quanta` columns), or
    /// `None` when the job is banded: this batch then owns it and [`Self::run`] writes its result.
    #[cfg(test)]
    pub(crate) fn plan(
        &mut self,
        idx: usize,
        t: &[u8],
        q: &[u8],
        p: &Params,
        max_hits: usize,
        quanta: usize,
    ) -> Option<Decision> {
        let (dec, t1) = self.decide(t, q, p, max_hits);
        self.commit(idx, t, q, dec, t1, p, quanta)
    }

    /// The prune decision of window `t` against mate `q` and its round-1 threshold, leaving the
    /// components in this batch for [`Self::commit`].
    pub(crate) fn decide(
        &mut self,
        t: &[u8],
        q: &[u8],
        p: &Params,
        max_hits: usize,
    ) -> (Decision, i32) {
        rescue_prune::decide_bands(t, q, p, max_hits, &mut self.c19, &mut self.ct)
    }

    /// [`Self::plan`] after [`Self::decide`] gave `(dec, t1)` for the same `t` and `q`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit(
        &mut self,
        idx: usize,
        t: &[u8],
        q: &[u8],
        dec: Decision,
        t1: i32,
        p: &Params,
        quanta: usize,
    ) -> Option<Decision> {
        let (len1, quanta) = (t.len() as i32, quanta as i32);
        if !enabled() || q.len().div_ceil(16) * 16 != quanta as usize {
            return Some(dec);
        }
        let (hb, he) = match dec {
            Decision::Hull { hb, he } => (hb as i32, he as i32),
            // A whole window without an N (the hit gate, not an N, refused the filter) runs as one
            // band covering every cell: the same DP, on the faster kernel.
            Decision::Full if t.iter().chain(q).fold(0u8, |a, &b| a | b) & !3 == 0 => {
                self.single(idx, 0, len1 - 1, quanta, p.minsc);
                return None;
            }
            _ => return Some(dec),
        };
        if self.c19.is_empty() || self.c19.len() > COMP_CAP {
            self.single(idx, hb, he, quanta, p.minsc);
            return None;
        }
        let b0 = self.bands.len();
        let round1 = if t1 > p.minsc { &self.ct } else { &self.c19 };
        let mut cost = 0i64;
        for c in round1 {
            let Some(b) = to_band(c, p, t1, len1, quanta, hb, he) else {
                self.bands.truncate(b0);
                self.single(idx, hb, he, quanta, p.minsc);
                return None;
            };
            // Per row the kernel computes at most `min(width, quanta)` cells.
            cost += (b.r1 - b.r0 + 1) as i64 * ((b.dhi - b.dlo + 1).min(quanta) as i64 + OVH);
            self.bands.push(b);
        }
        if cost >= (he - hb + 1) as i64 * quanta as i64 {
            self.bands.truncate(b0);
            self.single(idx, hb, he, quanta, p.minsc);
            return None;
        }
        let c0 = self.bands.len();
        if t1 > p.minsc {
            for c in &self.c19 {
                let Some(b) = to_band(c, p, p.minsc, len1, quanta, hb, he) else {
                    self.bands.truncate(b0);
                    self.single(idx, hb, he, quanta, p.minsc);
                    return None;
                };
                self.bands.push(b);
            }
        }
        self.parents.push(Parent {
            job: idx as u32,
            hb,
            he,
            t1,
            b0: b0 as u32,
            nb: (c0 - b0) as u32,
            c0: c0 as u32,
            nc: (self.bands.len() - c0) as u32,
            s: 0,
            te: -1,
            qe: 0,
            s2: -1,
            te2: -1,
            buf: -1,
            round: 0,
        });
        None
    }

    /// One band holding every cell of rows `hb..=he` (the hull, or the whole window): the plain
    /// DP of those rows on the band kernel, final in round 1.
    fn single(&mut self, idx: usize, hb: i32, he: i32, quanta: i32, minsc: i32) {
        let b0 = self.bands.len();
        self.bands.push(Band {
            r0: hb,
            r1: he,
            dlo: hb - (quanta - 1),
            dhi: he,
            ub: 0,
        });
        self.parents.push(Parent {
            job: idx as u32,
            hb,
            he,
            t1: minsc,
            b0: b0 as u32,
            nb: 1,
            c0: b0 as u32 + 1,
            nc: 0,
            s: 0,
            te: -1,
            qe: 0,
            s2: -1,
            te2: -1,
            buf: -1,
            round: 0,
        });
    }

    /// Run the banded jobs: `out[parent.job]` gets exactly what the hull job's forward pass would
    /// give (`qb`/`tb` stay -1 for the reverse pass).
    ///
    /// # Safety
    /// NEON must be available; `mat` must be the standard 5x5 table and every banded job's score
    /// ceiling under the u8 limit (the caller's u8-kernel conditions).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run(
        &mut self,
        jobs: &[KswJob],
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        minsc: i32,
        max_sc: i32,
        out: &mut [KswAlignResult],
    ) {
        if self.parents.is_empty() {
            return;
        }
        let gaps = [
            (o_del + e_del) as u8,
            e_del as u8,
            (o_ins + e_ins) as u8,
            e_ins as u8,
        ];
        let tbl = crate::matesw::u8_score_table(mat);
        let probe = crate::matesw::cells::enabled();
        let t_all = probe.then(std::time::Instant::now);
        let cells0 = self.work.cells;
        // ---- round 1 ----
        self.jobs.clear();
        self.pool.clear();
        for pi in 0..self.parents.len() {
            self.add_jobs(pi, 1, false, 0, 0);
        }
        self.run_jobs(jobs, &tbl, gaps, minsc, max_sc);
        self.r2.clear();
        for pi in 0..self.parents.len() {
            self.finish(pi, minsc, max_sc);
            let p = &self.parents[pi];
            let z = (p.s + max_sc - 1) / max_sc;
            let done = p.t1 == minsc
                || (p.s >= p.t1 && (p.s2 >= p.t1 || (p.te - z <= p.hb && p.te + z >= p.he)));
            if done {
                write(p, &mut out[p.job as usize]);
            } else {
                self.r2.push(pi as u32);
            }
        }
        // ---- round 2: the components at `minsc`, skipping those confined to `(te - Z, te)` with a
        //      bound below the round-1 `S` when that `S` (`>= T1`) is already exact ----
        if let Some(t0) = t_all {
            use crate::matesw::cells::*;
            use std::sync::atomic::Ordering::Relaxed;
            BAND_R2.fetch_add(self.r2.len() as u64, Relaxed);
            BAND_CELLS.fetch_add(
                (self.work.cells - cells0) * self.work.lanes() as u64,
                Relaxed,
            );
            BAND_KERNEL_NS.fetch_add(self.work.kernel_ns, Relaxed);
            BAND_OTHER_NS.fetch_add(
                t0.elapsed().as_nanos() as u64 - self.work.kernel_ns,
                Relaxed,
            );
            self.work.kernel_ns = 0;
        }
        if self.r2.is_empty() {
            return;
        }
        let t_all = probe.then(std::time::Instant::now);
        let cells0 = self.work.cells;
        self.jobs.clear();
        self.pool.clear();
        for i in 0..self.r2.len() {
            let pi = self.r2[i] as usize;
            let p = &self.parents[pi];
            let (sh, teh) = if p.s >= p.t1 { (p.s, p.te) } else { (0, 0) };
            let zh = (sh + max_sc - 1) / max_sc;
            self.add_jobs(pi, 2, sh >= minsc, teh - zh, teh);
        }
        self.run_jobs(jobs, &tbl, gaps, minsc, max_sc);
        for i in 0..self.r2.len() {
            let pi = self.r2[i] as usize;
            self.finish(pi, minsc, max_sc);
            let p = &self.parents[pi];
            write(p, &mut out[p.job as usize]);
        }
        if let Some(t0) = t_all {
            use crate::matesw::cells::*;
            use std::sync::atomic::Ordering::Relaxed;
            BAND_CELLS.fetch_add(
                (self.work.cells - cells0) * self.work.lanes() as u64,
                Relaxed,
            );
            BAND_KERNEL_NS.fetch_add(self.work.kernel_ns, Relaxed);
            BAND_OTHER_NS.fetch_add(
                t0.elapsed().as_nanos() as u64 - self.work.kernel_ns,
                Relaxed,
            );
            self.work.kernel_ns = 0;
        }
    }

    /// Queue parent `pi`'s bands of `round`, resetting its state; in round 2 a band confined to
    /// rows `(zlo, zhi)` with `ub < S` is skipped when `skip` is set.
    fn add_jobs(&mut self, pi: usize, round: u8, skip: bool, zlo: i32, zhi: i32) {
        let p = &mut self.parents[pi];
        let sh = p.s;
        (p.s, p.te, p.qe, p.s2, p.te2, p.buf, p.round) = (0, -1, 0, -1, -1, -1, round);
        let (b0, n) = if round == 1 {
            (p.b0, p.nb)
        } else {
            (p.c0, p.nc)
        };
        let first = self.jobs.len();
        for k in 0..n as usize {
            let b = self.bands[b0 as usize + k];
            if skip && b.ub < sh && b.r0 > zlo && b.r1 < zhi {
                continue;
            }
            let w = b.dhi - b.dlo + 1;
            let rows = (b.r1 - b.r0 + 1).min(65535) as u32;
            self.jobs.push(Job {
                parent: pi as u32,
                band: b,
                key: width_bucket(w) << 16 | rows,
            });
        }
        if self.jobs.len() - first > 1 {
            // Merged row maxima over the hull rows.
            let h = (p.he - p.hb + 1) as usize;
            p.buf = self.pool.len() as i32;
            self.pool.resize(self.pool.len() + h, 0);
        }
    }

    /// Run the queued jobs in groups of 16 and merge each lane into its parent.
    unsafe fn run_jobs(
        &mut self,
        jobs: &[KswJob],
        tbl: &[u8; 16],
        gaps: [u8; 4],
        minsc: i32,
        max_sc: i32,
    ) {
        let n = self.jobs.len();
        self.order.clear();
        self.order.extend(
            self.jobs
                .iter()
                .enumerate()
                .map(|(i, j)| u64::from(j.key) << 32 | i as u64),
        );
        self.order.sort_unstable();
        let gl = self.work.lanes();
        // Once per call, not per group (a `memset` of the whole array each time): a group writes
        // and reads only its first `nl` lanes.
        let mut lanes = [kernel::Lane::default(); kernel::L];
        for g in (0..n).step_by(gl) {
            let nl = gl.min(n - g);
            let (mut wmax, mut omax) = (1, i32::MIN);
            for l in 0..nl {
                let j = &self.jobs[self.order[g + l] as u32 as usize];
                let b = &j.band;
                let kj = &jobs[self.parents[j.parent as usize].job as usize];
                lanes[l] = kernel::Lane {
                    q: kj.query,
                    t: &kj.target[b.r0 as usize..=b.r1 as usize],
                    quanta: kj.query.len().div_ceil(16) as i32 * 16,
                    o: b.r0 - b.dlo,
                    w: b.dhi - b.dlo + 1,
                    target: 0,
                    thr: minsc.clamp(0, 255) as u8,
                };
                wmax = wmax.max(lanes[l].w);
                omax = omax.max(lanes[l].o);
            }
            // Band shift: a lane narrower than the group gets its spare diagonals below its band
            // whenever that moves its query offset up toward the group's largest, so the lanes'
            // live columns line up and the per-row union shrinks. The lane only gains diagonals.
            for lane in lanes[..nl].iter_mut() {
                lane.o += (omax - lane.o).min(wmax - lane.w);
            }
            // Timed only for the probe: two clock reads per group were paid on every run.
            let t0 = crate::matesw::cells::enabled().then(std::time::Instant::now);
            self.work.run(&lanes[..nl], wmax as usize, tbl, gaps);
            if let Some(t0) = t0 {
                self.work.kernel_ns += t0.elapsed().as_nanos() as u64;
            }
            for l in 0..nl {
                let ji = self.order[g + l] as u32 as usize;
                let (pi, b) = (self.jobs[ji].parent as usize, self.jobs[ji].band);
                let p = &mut self.parents[pi];
                let g8 = i32::from(self.work.gmax[l]);
                let (te, qe) = if g8 > 0 {
                    (b.r0 + self.work.te[l], i32::from(self.work.qe[l]))
                } else {
                    (-1, 0)
                };
                let nrows = (b.r1 - b.r0 + 1) as usize;
                let rl = self.work.lane_rows(l, nrows);
                if p.buf < 0 {
                    // The parent's only band: its outputs are the parent's.
                    (p.s, p.te, p.qe) = (g8, te, qe);
                    (p.s2, p.te2) = if g8 >= minsc {
                        score2(rl, b.r0, g8, te, minsc, max_sc)
                    } else {
                        (-1, -1)
                    };
                } else {
                    let buf = &mut self.pool[p.buf as usize + (b.r0 - p.hb) as usize..];
                    for (x, &v) in buf[..nrows].iter_mut().zip(rl) {
                        *x = (*x).max(v);
                    }
                    if g8 > p.s || (g8 == p.s && g8 > 0 && (te < p.te || (te == p.te && qe < p.qe)))
                    {
                        (p.s, p.te, p.qe) = (g8, te, qe);
                    }
                }
            }
        }
    }

    /// The reverse pass (start recovery) of `jobs`, each `(reversed query prefix, reversed target
    /// prefix, score S)`: `out[i] = (score, te, qe)` as the forward kernel on the same prefixes
    /// stopped at `S` would give. The DP runs in the band of diagonals `[-Imax, Dmax]`
    /// (`d = row - column`): an alignment scoring `S` over `qe + 1` query bases with `D` deleted
    /// and `I` inserted bases has `S <= a (qe + 1) - o_del - e_del D` and
    /// `S <= a (qe + 1 - I) - o_ins - e_ins I`, so its every prefix lies on those diagonals
    /// (fg-labs/bwa-mem3's pass-1 band). A zero-state DP restricted to the band never exceeds
    /// the full one and holds `S` where the alignment ends, so the first row reaching `S` and the
    /// first column holding it there are the same, and the group stops once every lane has them.
    ///
    /// # Safety
    /// As [`Self::run`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run_reverse(
        &mut self,
        jobs: &[(&[u8], &[u8], i32)],
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        max_sc: i32,
        out: &mut [(i32, i32, i32)],
    ) {
        let gaps = [
            (o_del + e_del) as u8,
            e_del as u8,
            (o_ins + e_ins) as u8,
            e_ins as u8,
        ];
        let tbl = crate::matesw::u8_score_table(mat);
        self.jobs.clear();
        for (i, &(q, t, s)) in jobs.iter().enumerate() {
            let (len2, nrows) = (q.len() as i32, t.len() as i32);
            let dmax = ((max_sc * len2 - s - o_del) / e_del).max(0);
            let imax = ((max_sc * len2 - s - o_ins) / (max_sc + e_ins)).max(0);
            let b = Band {
                r0: 0,
                r1: nrows - 1,
                dlo: -imax,
                dhi: dmax,
                ub: s,
            };
            self.jobs.push(Job {
                parent: i as u32,
                band: b,
                key: width_bucket(b.dhi - b.dlo + 1) << 16 | nrows.min(65535) as u32,
            });
        }
        let n = self.jobs.len();
        self.order.clear();
        self.order.extend(
            self.jobs
                .iter()
                .enumerate()
                .map(|(i, j)| u64::from(j.key) << 32 | i as u64),
        );
        self.order.sort_unstable();
        let gl = self.work.lanes();
        // Once per call, not per group (a `memset` of the whole array each time): a group writes
        // and reads only its first `nl` lanes.
        let mut lanes = [kernel::Lane::default(); kernel::L];
        for g in (0..n).step_by(gl) {
            let nl = gl.min(n - g);
            let (mut wmax, mut omax) = (1, i32::MIN);
            for l in 0..nl {
                let j = &self.jobs[self.order[g + l] as u32 as usize];
                let (q, t, s) = jobs[j.parent as usize];
                lanes[l] = kernel::Lane {
                    q,
                    t,
                    quanta: q.len().div_ceil(16) as i32 * 16,
                    o: -j.band.dlo,
                    w: j.band.dhi - j.band.dlo + 1,
                    target: s as u8,
                    thr: s as u8,
                };
                wmax = wmax.max(lanes[l].w);
                omax = omax.max(lanes[l].o);
            }
            for lane in lanes[..nl].iter_mut() {
                lane.o += (omax - lane.o).min(wmax - lane.w);
            }
            let t0 = crate::matesw::cells::enabled().then(std::time::Instant::now);
            self.work.run(&lanes[..nl], wmax as usize, &tbl, gaps);
            if let Some(t0) = t0 {
                self.work.kernel_ns += t0.elapsed().as_nanos() as u64;
            }
            for l in 0..nl {
                let i = self.jobs[self.order[g + l] as u32 as usize].parent as usize;
                let g8 = i32::from(self.work.gmax[l]);
                out[i] = if g8 > 0 {
                    (g8, self.work.te[l], i32::from(self.work.qe[l]))
                } else {
                    (0, -1, -1)
                };
            }
        }
        // Flushed here, or the next `run` subtracts this pass's kernel time from its own clock
        // and the probe's "other" wraps around.
        if crate::matesw::cells::enabled() {
            crate::matesw::cells::BAND_KERNEL_NS
                .fetch_add(self.work.kernel_ns, std::sync::atomic::Ordering::Relaxed);
            self.work.kernel_ns = 0;
        }
    }

    /// `score2`/`te2` of a merged parent from its row maxima.
    fn finish(&mut self, pi: usize, minsc: i32, max_sc: i32) {
        let p = &mut self.parents[pi];
        if p.buf < 0 {
            return;
        }
        (p.s2, p.te2) = if p.s >= minsc {
            let h = (p.he - p.hb + 1) as usize;
            let buf = &self.pool[p.buf as usize..p.buf as usize + h];
            score2(buf, p.hb, p.s, p.te, minsc, max_sc)
        } else {
            (-1, -1)
        };
    }
}

/// The kernels' suboptimal tracker (`SuboptimalTracker::push_row`) streamed: `cs`/`cr` the open
/// candidate (`cr == i32::MIN` when none), `s2`/`te2` the best closed one outside `[low, high]`.
/// Its candidates are evaluated as each one closes instead of being stored: `s` and `te` are known
/// up front, so `finish`'s exclusion test runs on the fly, in the same row order, keeping the
/// first strict maximum as it does.
struct Tracker {
    s2: i32,
    te2: i32,
    cs: i32,
    cr: i32,
    low: i32,
    high: i32,
}

impl Tracker {
    #[inline(always)]
    fn close(&mut self) {
        if self.cr != i32::MIN && (self.cr < self.low || self.cr > self.high) && self.cs > self.s2 {
            (self.s2, self.te2) = (self.cs, self.cr);
        }
    }

    /// One row reaching `minsc`: it raises the candidate it directly follows (the candidate's row
    /// is that of its last raise), else closes it and opens its own.
    #[inline(always)]
    fn push(&mut self, row: i32, v: i32) {
        if self.cr != i32::MIN && self.cr + 1 == row {
            if self.cs < v {
                (self.cs, self.cr) = (v, row);
            }
        } else {
            self.close();
            (self.cs, self.cr) = (v, row);
        }
    }

    /// Push the rows `rows[from..to]` (target rows `row0 + x`) that reach `minsc`, skipping those
    /// not above `s2`, the best kept so far. Such a row can never be kept (a candidate's score is
    /// a row's), nor stop a better candidate from being: one row after it ends that candidate's
    /// growth whatever its score, and skipping it only delays the candidate's close, which `s2`
    /// does not change in between (every row that could change it is still pushed). The first row
    /// above `s2` leaves the same state either way, opening or raising to `(v, row)`.
    fn scan(&mut self, rows: &[u8], row0: i32, minsc: i32, from: usize, to: usize) {
        #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
        let mut r = from;
        // Rows are `u8`, so the vector test needs `minsc` in `1..=255`; any other `minsc` takes
        // the row loop below for every row.
        #[cfg(target_arch = "aarch64")]
        if (1..=255).contains(&minsc) {
            use std::arch::aarch64::*;
            // SAFETY: NEON is part of the aarch64 baseline; each load reads 16 bytes inside `rows`
            // (`r + 16 <= to <= rows.len()`).
            unsafe {
                while r + 16 <= to {
                    // The bar a row must reach: `minsc`, or above `s2` once one is kept. Past 255
                    // no `u8` row can.
                    let bar = minsc.max(self.s2 + 1);
                    if bar > 255 {
                        return;
                    }
                    let a = vld1q_u8(rows.as_ptr().add(r));
                    // One vector max per chunk: almost every row maximum is below the bar.
                    if vmaxvq_u8(a) >= bar as u8 {
                        // Four bits per row, set where the row reaches the bar.
                        let mut nib = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(
                            vreinterpretq_u16_u8(vcgeq_u8(a, vdupq_n_u8(bar as u8))),
                        ))) & 0x1111_1111_1111_1111;
                        while nib != 0 {
                            let x = r + nib.trailing_zeros() as usize / 4;
                            self.push(row0 + x as i32, i32::from(rows[x]));
                            nib &= nib - 1;
                        }
                    }
                    r += 16;
                }
            }
        }
        // The same scan with SSE2 (x86_64 baseline): `max(a, bar) == a` is `a >= bar` unsigned.
        #[cfg(target_arch = "x86_64")]
        if (1..=255).contains(&minsc) {
            use std::arch::x86_64::*;
            // SAFETY: SSE2 is part of the x86_64 baseline; each load reads 16 bytes inside `rows`
            // (`r + 16 <= to <= rows.len()`).
            unsafe {
                while r + 16 <= to {
                    let bar = minsc.max(self.s2 + 1);
                    if bar > 255 {
                        return;
                    }
                    let a = _mm_loadu_si128(rows.as_ptr().add(r).cast());
                    let ge = _mm_cmpeq_epi8(_mm_max_epu8(a, _mm_set1_epi8(bar as u8 as i8)), a);
                    let mut bits = _mm_movemask_epi8(ge) as u32;
                    while bits != 0 {
                        let x = r + bits.trailing_zeros() as usize;
                        self.push(row0 + x as i32, i32::from(rows[x]));
                        bits &= bits - 1;
                    }
                    r += 16;
                }
            }
        }
        for (x, &v) in rows.iter().enumerate().take(to).skip(r) {
            if i32::from(v) >= minsc && i32::from(v) > self.s2 {
                self.push(row0 + x as i32, i32::from(v));
            }
        }
    }
}

/// `(score2, te2)` from the row maxima `rows` of target rows `row0..`, by the kernels' tracker.
///
/// The rows inside the exclusion zone `[low, high]` (around `te`, where the rows reaching `minsc`
/// crowd) are mostly skipped: a candidate whose row is in the zone is never kept, so once the open
/// candidate's row is in the zone the only thing the zone's remaining rows decide is whether the
/// candidate open at its last row `h` was raised there (then `(rows[h], h)`, which the next row may
/// continue). That follows `u_i = p_i && !(u_(i-1) && v_(i-1) >= v_i)` (`p_i`: row `i` reaches
/// `minsc`): `u_i = p_i` where the previous row is not pushed or is lower, and `u` alternates along
/// a run of pushed, non-increasing rows. So `u_h` comes from walking that run back from `h`.
fn score2(rows: &[u8], row0: i32, s: i32, te: i32, minsc: i32, max_sc: i32) -> (i32, i32) {
    let half = (s + max_sc - 1) / max_sc;
    let (low, high) = (te - half, te + half);
    let mut t = Tracker {
        s2: -1,
        te2: -1,
        cs: 0,
        cr: i32::MIN,
        low,
        high,
    };
    let n = rows.len();
    // The zone's rows `[b0, b1)`, as indices into `rows`.
    let idx = |row: i64| (row - i64::from(row0)).clamp(0, n as i64) as usize;
    let (b0, b1) = if high - low >= 2 {
        (idx(i64::from(low)), idx(i64::from(high) + 1))
    } else {
        (n, n)
    };
    t.scan(rows, row0, minsc, 0, b0);
    // Row by row until the open candidate's row enters the zone (at most two pushed rows).
    let mut x = b0;
    while x < b1 && !(low..=high).contains(&t.cr) {
        if i32::from(rows[x]) >= minsc {
            t.push(row0 + x as i32, i32::from(rows[x]));
        }
        x += 1;
    }
    if x < b1 {
        // The candidate was raised at row `x - 1`, so `u` is true there; walk back from the last
        // zone row `h` along pushed, non-increasing rows, stopping at `x - 1` at the latest.
        let (h, known) = (b1 - 1, x - 1);
        let p = |i: usize| i32::from(rows[i]) >= minsc;
        let raised = p(h) && {
            let mut j = h;
            while j > known && p(j - 1) && rows[j - 1] >= rows[j] {
                j -= 1;
            }
            // `u_j` is true (`j == known`, or the run starts at `j` with `p_j`); it alternates.
            (h - j) % 2 == 0
        };
        // Not raised at `h`: any zone row other than `h` stands for the candidate, which the next
        // row cannot continue and which, being in the zone, is never kept.
        (t.cs, t.cr) = if raised {
            (i32::from(rows[h]), row0 + h as i32)
        } else {
            (t.cs, low)
        };
    }
    t.scan(rows, row0, minsc, b1, n);
    t.close();
    if s > 0 {
        (t.s2, t.te2)
    } else {
        (-1, -1)
    }
}

fn write(p: &Parent, o: &mut KswAlignResult) {
    *o = KswAlignResult {
        score: p.s,
        qb: -1,
        qe: if p.te >= 0 { p.qe } else { -1 },
        tb: -1,
        te: p.te,
        score2: p.s2,
        te2: p.te2,
    };
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use super::RING;
    use std::arch::aarch64::*;

    pub(super) const L: usize = 16;
    /// Query code of a cell outside the query (`j < 0` or `j >= quanta`): dead, scores 0.
    const DEAD: u8 = 0xC0;
    /// Query code of a ksw pad column (`len2 <= j < quanta`): scores 0, carries the diagonal.
    const QPAD: u8 = 0x40;
    /// Reference code of a row past a lane's band: its row maxima are ignored.
    const RPAD: u8 = 0x80;

    /// One lane's band: query, padded width, the band's target rows, query offset `o = r0 - dlo`
    /// (`j = r + o - k`) and width.
    #[derive(Clone, Copy, Default)]
    pub(super) struct Lane<'a> {
        pub q: &'a [u8],
        pub t: &'a [u8],
        pub quanta: i32,
        pub o: i32,
        pub w: i32,
        /// Early exit (the reverse pass): the group stops once every lane's best score has
        /// reached its target; 0 means none.
        pub target: u8,
        /// `qe` is only read off H while the lane's best reaches this: a lower result is dropped
        /// by the caller (forward pass: `minsc`; reverse pass: `target`).
        pub thr: u8,
    }

    /// Kernel scratch and per-lane outputs: the best score, its first row (band-relative) and
    /// first query column there, and the row maxima, lane-major (`rl[lane * rstride + row]`).
    #[derive(Default)]
    pub(super) struct Work {
        st: Vec<u8>,
        a: Vec<u8>,
        refv: Vec<u8>,
        /// H of the last [`RING`] rows, row `x` in slot `x % RING`.
        h: Vec<u8>,
        e: Vec<u8>,
        r: Vec<u8>,
        rl: Vec<u8>,
        rstride: usize,
        pub gmax: [u8; L],
        pub te: [i32; L],
        pub qe: [u8; L],
        /// Cell vectors computed and kernel time, for the probes.
        pub cells: u64,
        pub kernel_ns: u64,
    }

    fn fit(v: &mut Vec<u8>, n: usize) {
        if v.len() < n {
            v.resize(n, 0);
        }
    }

    /// Transpose 16 lane-major rows of `stride` bytes (`n` of them used, a multiple of 16) into
    /// position-major vectors at `dst`.
    ///
    /// # Safety
    /// NEON; `src` holds `16 * stride` bytes, `dst` `n * 16`.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn transpose_to(src: *const u8, stride: usize, n: usize, dst: *mut u8) {
        let mut b = 0;
        while b < n {
            let mut v = [vdupq_n_u8(0); 16];
            for (l, x) in v.iter_mut().enumerate() {
                *x = vld1q_u8(src.add(l * stride + b));
            }
            let v = transpose16(v);
            for (t, x) in v.iter().enumerate() {
                vst1q_u8(dst.add((b + t) * 16), *x);
            }
            b += 16;
        }
    }

    /// The inverse: `n` position-major vectors (a multiple of 16) at `src` into 16 lane-major rows
    /// of `stride` bytes at `dst`.
    ///
    /// # Safety
    /// NEON; `src` holds `n * 16` bytes, `dst` `16 * stride`.
    #[target_feature(enable = "neon")]
    unsafe fn transpose_from(src: *const u8, n: usize, dst: *mut u8, stride: usize) {
        let mut b = 0;
        while b < n {
            let mut v = [vdupq_n_u8(0); 16];
            for (t, x) in v.iter_mut().enumerate() {
                *x = vld1q_u8(src.add((b + t) * 16));
            }
            let v = transpose16(v);
            for (l, x) in v.iter().enumerate() {
                vst1q_u8(dst.add(l * stride + b), *x);
            }
            b += 16;
        }
    }

    /// 16x16 byte transpose in registers: `out[t][l] = v[l][t]`.
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn transpose16(v: [uint8x16_t; 16]) -> [uint8x16_t; 16] {
        let mut a = v;
        // Bytes.
        for i in 0..8 {
            let (x, y) = (a[2 * i], a[2 * i + 1]);
            a[2 * i] = vtrn1q_u8(x, y);
            a[2 * i + 1] = vtrn2q_u8(x, y);
        }
        // Halfwords: pairs (0,2) (1,3) (4,6) ...
        for i in 0..4 {
            for j in 0..2 {
                let (p, q) = (4 * i + j, 4 * i + j + 2);
                let (x, y) = (vreinterpretq_u16_u8(a[p]), vreinterpretq_u16_u8(a[q]));
                a[p] = vreinterpretq_u8_u16(vtrn1q_u16(x, y));
                a[q] = vreinterpretq_u8_u16(vtrn2q_u16(x, y));
            }
        }
        // Words: pairs (0,4) (1,5) (2,6) (3,7) (8,12) ...
        for i in 0..2 {
            for j in 0..4 {
                let (p, q) = (8 * i + j, 8 * i + j + 4);
                let (x, y) = (vreinterpretq_u32_u8(a[p]), vreinterpretq_u32_u8(a[q]));
                a[p] = vreinterpretq_u8_u32(vtrn1q_u32(x, y));
                a[q] = vreinterpretq_u8_u32(vtrn2q_u32(x, y));
            }
        }
        // Doublewords: pairs (i, i+8).
        let mut out = [vdupq_n_u8(0); 16];
        for j in 0..8 {
            let (x, y) = (vreinterpretq_u64_u8(a[j]), vreinterpretq_u64_u8(a[j + 8]));
            out[j] = vreinterpretq_u8_u64(vtrn1q_u64(x, y));
            out[j + 8] = vreinterpretq_u8_u64(vtrn2q_u64(x, y));
        }
        out
    }

    impl Work {
        /// Lane `l`'s row maxima for its `nrows` rows, after [`Self::run`].
        pub(super) fn lanes(&self) -> usize {
            L
        }

        pub(super) fn lane_rows(&self, l: usize, nrows: usize) -> &[u8] {
            &self.rl[l * self.rstride..l * self.rstride + nrows]
        }

        /// Run one group of `lanes` (at most `L`) at width `w`.
        ///
        /// # Safety
        /// NEON available.
        pub(super) unsafe fn run(
            &mut self,
            lanes: &[Lane],
            w: usize,
            tbl: &[u8; 16],
            gaps: [u8; 4],
        ) {
            if gaps[0] == gaps[2] {
                self.run_impl::<true>(lanes, w, tbl, gaps)
            } else {
                self.run_impl::<false>(lanes, w, tbl, gaps)
            }
        }

        #[target_feature(enable = "neon")]
        unsafe fn run_impl<const SHARE: bool>(
            &mut self,
            lanes: &[Lane],
            w: usize,
            tbl: &[u8; 16],
            gaps: [u8; 4],
        ) {
            let nr = lanes.iter().map(|l| l.t.len()).max().unwrap_or(0);
            let np = nr + w - 1;
            let c = w as i32 - 1;
            let (npp, nrp) = ((np + 3).next_multiple_of(16), nr.next_multiple_of(16));
            let stride = npp.max(nrp);
            // ---- SoA build through a lane-major staging area and a 16x16 transpose: the query in
            //      band layout (`p = r + c - k`), then the reference rows ----
            fit(&mut self.st, stride * L);
            fit(&mut self.a, npp * L);
            fit(&mut self.refv, nrp * L);
            // H and E carry 3 slack vectors before slot 0 and a few past `w - 1`: the skewed body's
            // masked cells read and write there (see the row loop).
            let hs = w + 7;
            fit(&mut self.h, RING * hs * L);
            fit(&mut self.e, (w + 9) * L);
            fit(&mut self.r, nrp * L);
            // All of it: a row reads the row above's cells past that row's own top, never written.
            self.h[..RING * hs * L].fill(0);
            self.e[..(w + 9) * L].fill(0);
            let (mut omax, mut ominq, mut omaskq) = (i32::MIN, i32::MAX, i32::MIN);
            let mut ob = [0u8; L];
            let st = self.st.as_mut_ptr();
            for l in 0..L {
                let row = std::slice::from_raw_parts_mut(st.add(l * stride), npp);
                crate::batched::fill_fast(row, DEAD);
                let Some(lane) = lanes.get(l) else { continue };
                let pq0 = c - lane.o; // `p` where `j = 0`
                let len2 = lane.q.len() as i32;
                let (p0, p1) = (pq0.max(0), (pq0 + len2).min(np as i32));
                if p0 < p1 {
                    crate::batched::copy_fast(
                        &mut row[p0 as usize..p1 as usize],
                        &lane.q[(p0 - pq0) as usize..(p1 - pq0) as usize],
                    );
                }
                let (p0, p1) = ((pq0 + len2).max(0), (pq0 + lane.quanta).min(np as i32));
                if p0 < p1 {
                    crate::batched::fill_fast(&mut row[p0 as usize..p1 as usize], QPAD);
                }
                ob[l] = lane.o as u8;
                omax = omax.max(lane.o);
                ominq = ominq.min(lane.o - lane.quanta + 1);
                omaskq = omaskq.max(lane.o - lane.quanta + 1);
            }
            transpose_to(st, stride, npp, self.a.as_mut_ptr());
            for l in 0..L {
                let row = std::slice::from_raw_parts_mut(st.add(l * stride), nrp);
                let Some(lane) = lanes.get(l) else {
                    crate::batched::fill_fast(row, RPAD);
                    continue;
                };
                {
                    crate::batched::copy_fast(&mut row[..lane.t.len()], lane.t);
                    crate::batched::fill_fast(&mut row[lane.t.len()..], RPAD);
                    // A target N scores -1 against everything through the table's slots 8 and
                    // 12-15 once it is re-encoded as 12 (`N_TARGET`); as 4 it would read the
                    // match slot against a query N. A pruned job holds none; a reverse-pass job
                    // may. An OR fold, not `any`: it vectorises, and `b > 3` is `b & !3 != 0`.
                    if lane.t.iter().fold(0u8, |m, &b| m | b) & !3 != 0 {
                        for b in row[..lane.t.len()].iter_mut() {
                            if *b == 4 {
                                *b = 12;
                            }
                        }
                    }
                }
            }
            transpose_to(st, stride, nrp, self.refv.as_mut_ptr());

            // ---- DP ----
            let tblv = vld1q_u8(tbl.as_ptr());
            let zero = vdupq_n_u8(0);
            let one = vdupq_n_u8(1);
            let deadv = vdupq_n_u8(DEAD);
            let rpadv = vdupq_n_u8(RPAD);
            let (oe_del, e_del, oe_ins, e_ins) = (
                vdupq_n_u8(gaps[0]),
                vdupq_n_u8(gaps[1]),
                vdupq_n_u8(gaps[2]),
                vdupq_n_u8(gaps[3]),
            );
            let ov = vld1q_u8(ob.as_ptr());
            let mut tb = [0u8; L];
            for (l, lane) in lanes.iter().enumerate() {
                tb[l] = lane.target;
            }
            let early = tb.iter().any(|&t| t > 0);
            let tgt = vld1q_u8(tb.as_ptr());
            let a = self.a.as_ptr();
            let e = self.e.as_mut_ptr().add(3 * L);
            let rout = self.r.as_mut_ptr();
            let h0p = self.h.as_mut_ptr().add(3 * L);
            // Row `x`'s H (`x` = -1: slot RING - 1, still zero).
            let slot = |x: usize| h0p.add(x % RING * hs * L);
            let li = L as isize;
            let mut gmax = zero;
            // `te` as its low and high bytes per lane, written with the row on every improving
            // row; a lane that never improves keeps `gmax == 0` and reads back as -1.
            let (mut te_lo, mut te_hi) = (zero, zero);
            let mut qev = zero;
            let wm1 = w as i32 - 1;
            // `qe` is not tracked per cell. A lane's (te, qe) is that of its last improving row, so
            // `qe` is read off H only where a lane's run of improving rows ended and no later row
            // improved before the ring overwrites it (`pm`: the lanes whose `te` row is unread):
            // the largest k in the row's computed range whose H equals gmax. That is the cell the
            // per-cell tracking picked (the first, in k-descending order, reaching the live row
            // max): cells above a lane's range are j < 0 and hold 0 < gmax, and its dead
            // (j >= quanta) cells all lie below its live ones, so the largest match is live.
            let mut pm = zero;
            let mut th = [0u8; L];
            for (l, lane) in lanes.iter().enumerate() {
                th[l] = lane.thr;
            }
            let thrv = vld1q_u8(th.as_ptr());
            let qe_scan =
                |buf: *const u8, row: i32, msk: uint8x16_t, gmax: uint8x16_t, qev: uint8x16_t| {
                    let (kl, kh) = (0.max(row + ominq), wm1.min(row + omax));
                    let mut jv = vaddq_u8(vdupq_n_u8((row - kl) as u8), ov);
                    let mut best = zero;
                    for k in kl..=kh {
                        let eq = vceqq_u8(vld1q_u8(buf.add(k as usize * L)), gmax);
                        best = vbslq_u8(eq, jv, best);
                        jv = vsubq_u8(jv, one);
                    }
                    vbslq_u8(msk, best, qev)
                };
            // Read `qe` off rows `x0..x0 + n` (`n <= RING`) for the lanes whose `te` is one of
            // them. Every earlier row was read the same way, so a `pm` lane's `te` is at least
            // `x0` and below `x0 + 256`: its low byte names it.
            macro_rules! due {
                ($x0:expr, $n:expr) => {{
                    let x0: usize = $x0;
                    let off = vsubq_u8(te_lo, vdupq_n_u8(x0 as u8));
                    if vmaxvq_u8(vandq_u8(vcltq_u8(off, vdupq_n_u8($n as u8)), pm)) != 0 {
                        for x in x0..x0 + $n {
                            let m = vandq_u8(vceqq_u8(te_lo, vdupq_n_u8(x as u8)), pm);
                            if vmaxvq_u8(m) != 0 {
                                qev = qe_scan(slot(x), x as i32, m, gmax, qev);
                                pm = vbicq_u8(pm, m);
                            }
                        }
                    }
                }};
            }

            // One cell: `d` the diagonal H, `ein` the E carry, `f` the row's F carry (updated),
            // `top` the validity mask of a cell past `w - 1` (`None`: valid), `LIVE` whether a lane
            // can be on a dead `j >= quanta` cell here. Returns `(H, E for the row below)`.
            macro_rules! cell {
                ($d:expr, $ein:expr, $f:ident, $q:expr, $rref:expr, $rmax:ident, $top:expr,
                 $live:expr) => {{
                    let sc = vqtbl1q_u8(tblv, veorq_u8($q, $rref));
                    let mut h0 = vmaxq_u8(vsqaddq_u8($d, vreinterpretq_s8_u8(sc)), $ein);
                    if let Some(m) = $top {
                        h0 = vandq_u8(h0, m);
                    }
                    let mut h = vmaxq_u8(h0, $f);
                    if let Some(m) = $top {
                        h = vandq_u8(h, m);
                    }
                    let hl = if $live {
                        vandq_u8(h, vcltq_u8($q, deadv))
                    } else {
                        h
                    };
                    $rmax = vmaxq_u8($rmax, hl);
                    // Both gaps open from G = max(diagonal, E), not H (fg-labs/bwa-mem3's FScan
                    // cell): E then no longer waits on F, which takes F's max off the row-to-row
                    // E chain of the skewed step. It only drops paths with an in-row gap run
                    // directly followed by a vertical one; each such optimal path has a twin with
                    // the two runs swapped, same score, which F admits, and which the band holds
                    // wherever the band must be exact (rescue_prune's containment for round 1,
                    // the [-Imax, Dmax] diagonal range for the reverse pass): gmax, te, the qe
                    // cells and the row maxima are unchanged. In a whole-window band every H is.
                    // `SHARE`: one open for both gaps when `oe_del == oe_ins` (bwa's default).
                    let t = vqsubq_u8(h0, oe_del);
                    let eout = vmaxq_u8(t, vqsubq_u8($ein, e_del));
                    let ti = if SHARE { t } else { vqsubq_u8(h0, oe_ins) };
                    $f = vmaxq_u8(ti, vqsubq_u8($f, e_ins));
                    (h, eout)
                }};
            }
            // Row bookkeeping: publish the row max, update the lanes' best.
            macro_rules! finish_row {
                ($row:expr, $rmax:expr, $rref:expr) => {{
                    vst1q_u8(rout.add($row * L), $rmax);
                    let imp = vandq_u8(vcgtq_u8($rmax, gmax), vcltq_u8($rref, rpadv));
                    gmax = vbslq_u8(imp, $rmax, gmax);
                    te_lo = vbslq_u8(imp, vdupq_n_u8($row as u8), te_lo);
                    if $row >= 256 {
                        te_hi = vbslq_u8(imp, vdupq_n_u8(($row >> 8) as u8), te_hi);
                    }
                    pm = vorrq_u8(pm, vandq_u8(imp, vcgeq_u8($rmax, thrv)));
                }};
            }

            let mut r = 0usize;
            while r < nr {
                let ri = r as i32;
                let khi = wm1.min(ri + omax);
                let klo = 0.max(ri + ominq);
                let kun = klo.max(ri + omaskq);
                let mut k = khi;
                let hp = slot(r + RING - 1);
                if r + 3 < nr {
                    // Rows `r..r + 4` take the slots of `r - RING..`: read those first.
                    if r >= RING {
                        due!(r - RING, 4);
                    }
                    let (hr0, hr1, hr2, hc) = (slot(r), slot(r + 1), slot(r + 2), slot(r + 3));
                    // Four rows per step, skewed: `(r, k)`, `(r+1, k+1)`, `(r+2, k+2)`, `(r+3, k+3)`
                    // share the query position `p = r + c - k`; each row's diagonal and E carry come
                    // from the row above's previous step and this step, in registers. Cells past
                    // `w - 1` (the first steps when the top is clamped) and before 0 (the last
                    // steps when the bottom is clamped, `k < 0`) are masked to 0, which is their
                    // value; their loads and stores land in the buffers' zero slack.
                    self.cells += 4 * (khi - klo + 1).max(0) as u64;
                    let rref0 = vld1q_u8(self.refv.as_ptr().add(r * L));
                    let rref1 = vld1q_u8(self.refv.as_ptr().add((r + 1) * L));
                    let rref2 = vld1q_u8(self.refv.as_ptr().add((r + 2) * L));
                    let rref3 = vld1q_u8(self.refv.as_ptr().add((r + 3) * L));
                    let (mut f0, mut f1, mut f2, mut f3) = (zero, zero, zero, zero);
                    let (mut d1, mut d2, mut d3) = (zero, zero, zero);
                    let (mut rmax0, mut rmax1, mut rmax2, mut rmax3) = (zero, zero, zero, zero);
                    macro_rules! step4 {
                        ($mask:expr, $live:expr) => {{
                            let ko = k as isize;
                            let valid = |i: i32| -> Option<uint8x16_t> {
                                if $mask {
                                    Some(vdupq_n_u8(if (0..=wm1).contains(&(k + i)) {
                                        0xff
                                    } else {
                                        0
                                    }))
                                } else {
                                    None
                                }
                            };
                            let q = vld1q_u8(a.offset((r as isize + w as isize - 1 - ko) * li));
                            let e0 = vld1q_u8(e.offset(ko * li));
                            let d0 = vld1q_u8(hp.offset(ko * li));
                            let (h0, e1) = cell!(d0, e0, f0, q, rref0, rmax0, valid(0), $live);
                            let (h1, e2) = cell!(d1, e1, f1, q, rref1, rmax1, valid(1), $live);
                            let (h2, e3) = cell!(d2, e2, f2, q, rref2, rmax2, valid(2), $live);
                            let (h3, e4) = cell!(d3, e3, f3, q, rref3, rmax3, valid(3), $live);
                            vst1q_u8(hr0.offset(ko * li), h0);
                            vst1q_u8(hr1.offset((ko + 1) * li), h1);
                            vst1q_u8(hr2.offset((ko + 2) * li), h2);
                            vst1q_u8(hc.offset((ko + 3) * li), h3);
                            vst1q_u8(e.offset((ko + 4) * li), e4);
                            (d1, d2, d3) = (h0, h1, h2);
                            k -= 1;
                        }};
                    }
                    while k >= klo && k > wm1 - 3 {
                        step4!(true, true);
                    }
                    while k >= kun {
                        step4!(false, false);
                    }
                    while k >= klo {
                        step4!(false, true);
                    }
                    // Bottom clamped at 0: rows `r+1..r+3` still have cells `k+i` in `0..i`.
                    let kend = (ri + ominq).max(-3);
                    while k >= kend {
                        step4!(true, true);
                    }
                    finish_row!(r, rmax0, rref0);
                    finish_row!(r + 1, rmax1, rref1);
                    finish_row!(r + 2, rmax2, rref2);
                    finish_row!(r + 3, rmax3, rref3);
                    r += 4;
                    if early && vminvq_u8(vcgeq_u8(gmax, tgt)) == 0xff {
                        break;
                    }
                } else {
                    if r >= RING {
                        due!(r - RING, 1);
                    }
                    let hc = slot(r);
                    self.cells += (khi - klo + 1).max(0) as u64;
                    let rref = vld1q_u8(self.refv.as_ptr().add(r * L));
                    let (mut f, mut rmax) = (zero, zero);
                    macro_rules! step1 {
                        ($live:expr) => {{
                            let ko = k as isize;
                            let q = vld1q_u8(a.offset((r as isize + w as isize - 1 - ko) * li));
                            let e0 = vld1q_u8(e.offset(ko * li));
                            let d0 = vld1q_u8(hp.offset(ko * li));
                            let (h, e1) = cell!(d0, e0, f, q, rref, rmax, None, $live);
                            vst1q_u8(hc.offset(ko * li), h);
                            vst1q_u8(e.offset((ko + 1) * li), e1);
                            k -= 1;
                        }};
                    }
                    while k >= kun {
                        step1!(false);
                    }
                    while k >= klo {
                        step1!(true);
                    }
                    finish_row!(r, rmax, rref);
                    r += 1;
                    if early && vminvq_u8(vcgeq_u8(gmax, tgt)) == 0xff {
                        break;
                    }
                }
            }
            let x0 = r.saturating_sub(RING);
            due!(x0, r - x0);
            vst1q_u8(self.gmax.as_mut_ptr(), gmax);
            let (mut lo, mut hi) = ([0u8; L], [0u8; L]);
            vst1q_u8(lo.as_mut_ptr(), te_lo);
            vst1q_u8(hi.as_mut_ptr(), te_hi);
            self.te = std::array::from_fn(|l| {
                if self.gmax[l] > 0 {
                    i32::from(lo[l]) | i32::from(hi[l]) << 8
                } else {
                    -1
                }
            });
            vst1q_u8(self.qe.as_mut_ptr(), qev);
            // Row maxima back to lane-major, zero past `nr`.
            self.rstride = nrp;
            fit(&mut self.rl, nrp * L);
            self.r[nr * L..nrp * L].fill(0);
            transpose_from(self.r.as_ptr(), nrp, self.rl.as_mut_ptr(), nrp);
        }
    }
}

/// The AVX2 band kernel: the NEON kernel's cell at 32 lanes, two skewed rows per step (16 vector
/// registers), and two encodings for what x86 has no NEON TBL / USQADD for. `PSHUFB` zeroes an
/// entry only for an index with bit 7 set, so the pad and nonexistent query codes both carry it
/// (`QPAD` 0x80, `DEAD` 0xC0). And H, E and F live in the signed domain `H - 128` (`HZ` is the
/// DP's 0), where `adds_epi8(H, score)` is USQADD's `clamp(H + score, 0, 255)` exactly and every
/// gap subtract saturates at 0 as `subs_epu8` did: [`super::gaps_ok`] keeps the gap constants
/// under 128. A pad column's table entry is then 0 and carries the diagonal, as NEON's `+0` does.
/// The reference pad code `RPAD` (0x80) scores against a pad column, which does not matter: those
/// rows are past the lane's range, excluded from the row bookkeeping, and feed only later rows of
/// the same lane.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::RING;
    use crate::batched::{x86_row::t16, ABuf};
    use std::arch::x86_64::*;

    pub(super) const L: usize = 32;
    const DEAD: u8 = 0xC0;
    const QPAD: u8 = 0x80;
    const RPAD: u8 = 0x80;
    /// The DP's 0 in the signed domain the kernels compute in.
    pub(super) const HZ: u8 = 0x80;

    /// As the NEON kernel's `Lane`.
    #[derive(Clone, Copy, Default)]
    pub(super) struct Lane<'a> {
        pub q: &'a [u8],
        pub t: &'a [u8],
        pub quanta: i32,
        pub o: i32,
        pub w: i32,
        /// Early exit (the reverse pass): the group stops once every lane's best score has
        /// reached its target; 0 means none.
        pub target: u8,
        /// `qe` is only read off H while the lane's best reaches this: a lower result is dropped
        /// by the caller (forward pass: `minsc`; reverse pass: `target`).
        pub thr: u8,
    }

    /// As the NEON kernel's `Work`.
    #[derive(Default)]
    pub(super) struct Work {
        st: Vec<u8>,
        a: ABuf<u8>,
        refv: ABuf<u8>,
        /// H of the last [`RING`] rows, row `x` in slot `x % RING`.
        h: ABuf<u8>,
        e: ABuf<u8>,
        r: ABuf<u8>,
        rl: Vec<u8>,
        rstride: usize,
        pub gmax: [u8; L],
        pub te: [i32; L],
        pub qe: [u8; L],
        pub cells: u64,
    }

    fn fit(v: &mut Vec<u8>, n: usize) {
        if v.len() < n {
            v.resize(n, 0);
        }
    }

    impl Work {
        pub(super) fn lane_rows(&self, l: usize, nrows: usize) -> &[u8] {
            &self.rl[l * self.rstride..l * self.rstride + nrows]
        }

        /// Run one group of `lanes` (at most `L`) at width `w`; `tbl` is the signed score table,
        /// `gaps` each under 128.
        ///
        /// # Safety
        /// AVX2 available.
        pub(super) unsafe fn run(
            &mut self,
            lanes: &[Lane],
            w: usize,
            tbl: &[u8; 16],
            gaps: [u8; 4],
        ) {
            if gaps[0] == gaps[2] {
                self.run_impl::<true>(lanes, w, tbl, gaps)
            } else {
                self.run_impl::<false>(lanes, w, tbl, gaps)
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn run_impl<const SHARE: bool>(
            &mut self,
            lanes: &[Lane],
            w: usize,
            tbl: &[u8; 16],
            gaps: [u8; 4],
        ) {
            let nr = lanes.iter().map(|l| l.t.len()).max().unwrap_or(0);
            let np = nr + w - 1;
            let c = w as i32 - 1;
            let (npp, nrp) = ((np + 4).next_multiple_of(16), nr.next_multiple_of(16));
            let stride = npp.max(nrp);
            let hs = w + 7;
            fit(&mut self.st, 16 * stride);
            self.a.fit(npp * L, 0);
            self.refv.fit(nrp * L, 0);
            self.h.fit(RING * hs * L, HZ);
            self.e.fit((w + 9) * L, HZ);
            self.r.fit(nrp * L, 0);
            // All of it: a row reads the row above's cells past that row's own top, never written.
            self.h[..RING * hs * L].fill(HZ);
            self.e[..(w + 9) * L].fill(HZ);
            let (mut omax, mut ominq, mut omaskq) = (i32::MIN, i32::MAX, i32::MIN);
            let mut ob = [0u8; L];
            // The SoA buffers through a lane-major staging area, 16 lanes at a time, and 16x16
            // transposes (as the NEON kernel): the query in band layout, then the reference rows.
            let st = self.st.as_mut_ptr();
            for blk in 0..L / 16 {
                for i in 0..16 {
                    let row = std::slice::from_raw_parts_mut(st.add(i * stride), npp);
                    crate::batched::fill_fast(row, DEAD);
                    let l = blk * 16 + i;
                    let Some(lane) = lanes.get(l) else { continue };
                    let pq0 = c - lane.o;
                    let len2 = lane.q.len() as i32;
                    let (p0, p1) = (pq0.max(0), (pq0 + len2).min(np as i32));
                    if p0 < p1 {
                        crate::batched::copy_fast(
                            &mut row[p0 as usize..p1 as usize],
                            &lane.q[(p0 - pq0) as usize..(p1 - pq0) as usize],
                        );
                    }
                    let (p0, p1) = ((pq0 + len2).max(0), (pq0 + lane.quanta).min(np as i32));
                    if p0 < p1 {
                        crate::batched::fill_fast(&mut row[p0 as usize..p1 as usize], QPAD);
                    }
                    ob[l] = lane.o as u8;
                    omax = omax.max(lane.o);
                    ominq = ominq.min(lane.o - lane.quanta + 1);
                    omaskq = omaskq.max(lane.o - lane.quanta + 1);
                }
                for b in (0..npp).step_by(16) {
                    t16(
                        st.add(b),
                        stride,
                        self.a.as_mut_ptr().add(b * L + blk * 16),
                        L,
                    );
                }
                for i in 0..16 {
                    let row = std::slice::from_raw_parts_mut(st.add(i * stride), nrp);
                    let Some(lane) = lanes.get(blk * 16 + i) else {
                        crate::batched::fill_fast(row, RPAD);
                        continue;
                    };
                    crate::batched::copy_fast(&mut row[..lane.t.len()], lane.t);
                    crate::batched::fill_fast(&mut row[lane.t.len()..], RPAD);
                    // Target N re-encoded as 12 (see the NEON kernel).
                    if lane.t.iter().fold(0u8, |m, &b| m | b) & !3 != 0 {
                        for b in row[..lane.t.len()].iter_mut() {
                            if *b == 4 {
                                *b = 12;
                            }
                        }
                    }
                }
                for b in (0..nrp).step_by(16) {
                    t16(
                        st.add(b),
                        stride,
                        self.refv.as_mut_ptr().add(b * L + blk * 16),
                        L,
                    );
                }
            }
            // The signed table in both 128-bit halves.
            let t128 = _mm_loadu_si128(tbl.as_ptr() as *const __m128i);
            let tblv = _mm256_set_m128i(t128, t128);
            let hz = _mm256_set1_epi8(HZ as i8);
            let nil = _mm256_setzero_si256();
            let one = _mm256_set1_epi8(1);
            let deadv = _mm256_set1_epi8(DEAD as i8);
            // `dead ^ 0x7F`: -128 on a dead lane, 127 on a live one, the bound `min` clamps H to.
            let x7f = _mm256_set1_epi8(0x7F);
            let (oe_del, e_del, oe_ins, e_ins) = (
                _mm256_set1_epi8(gaps[0] as i8),
                _mm256_set1_epi8(gaps[1] as i8),
                _mm256_set1_epi8(gaps[2] as i8),
                _mm256_set1_epi8(gaps[3] as i8),
            );
            let ov = _mm256_loadu_si256(ob.as_ptr() as *const __m256i);
            // Targets in the H domain; an absent lane's (HZ) is always reached.
            let mut tb = [HZ; L];
            for (l, lane) in lanes.iter().enumerate() {
                tb[l] = lane.target ^ HZ;
            }
            let early = lanes.iter().any(|l| l.target > 0);
            let tgt = _mm256_loadu_si256(tb.as_ptr() as *const __m256i);
            let a = self.a.as_ptr();
            let e = self.e.as_mut_ptr().add(3 * L);
            let rout = self.r.as_mut_ptr();
            let h0p = self.h.as_mut_ptr().add(3 * L);
            // Row `x`'s H (`x` = -1: slot RING - 1, still zero).
            let slot = |x: usize| h0p.add(x % RING * hs * L);
            let li = L as isize;
            let mut gmax = hz;
            // `te` as its low and high bytes per lane, written with the row on every improving
            // row; a lane that never improves keeps `gmax == 0` and reads back as -1.
            let (mut te_lo, mut te_hi, mut qev) = (nil, nil, nil);
            let wm1 = w as i32 - 1;
            // `qe` from H as the NEON kernel's `qe_scan` and `due`; `thrv` in the H domain.
            let mut pm = nil;
            let mut th = [HZ; L];
            for (l, lane) in lanes.iter().enumerate() {
                th[l] = lane.thr ^ HZ;
            }
            let thrv = _mm256_loadu_si256(th.as_ptr() as *const __m256i);

            macro_rules! ld {
                ($p:expr) => {
                    _mm256_loadu_si256($p as *const __m256i)
                };
            }
            macro_rules! st {
                ($p:expr, $v:expr) => {
                    _mm256_storeu_si256($p as *mut __m256i, $v)
                };
            }
            macro_rules! qe_scan {
                ($buf:expr, $row:expr, $msk:expr) => {{
                    let (kl, kh) = (0.max($row + ominq), wm1.min($row + omax));
                    let mut jv = _mm256_add_epi8(_mm256_set1_epi8(($row - kl) as i8), ov);
                    let mut best = nil;
                    for k in kl..=kh {
                        let eq = _mm256_cmpeq_epi8(ld!($buf.add(k as usize * L)), gmax);
                        best = _mm256_blendv_epi8(best, jv, eq);
                        jv = _mm256_sub_epi8(jv, one);
                    }
                    qev = _mm256_blendv_epi8(qev, best, $msk);
                }};
            }
            // One cell, as the NEON kernel's (gaps open from G = max(diagonal, E)): `d` the
            // diagonal H, `ein` the E carry, `f` the row's F carry (updated), `top` whether a cell
            // past `w - 1` is valid (`None`: it is), `clamp` the live bound (`None`: no dead lane
            // here). Returns `(H, E for the row below)`.
            macro_rules! cell {
                ($d:expr, $ein:expr, $f:ident, $q:expr, $rref:expr, $rmax:ident, $top:expr,
                 $clamp:expr) => {{
                    let sc = _mm256_shuffle_epi8(tblv, _mm256_xor_si256($q, $rref));
                    let mut h0 = _mm256_max_epi8(_mm256_adds_epi8($d, sc), $ein);
                    let ok: Option<bool> = $top;
                    if ok == Some(false) {
                        h0 = hz;
                    }
                    let mut h = _mm256_max_epi8(h0, $f);
                    if ok == Some(false) {
                        h = hz;
                    }
                    let cl: Option<__m256i> = $clamp;
                    let hl = match cl {
                        Some(b) => _mm256_min_epi8(h, b),
                        None => h,
                    };
                    $rmax = _mm256_max_epi8($rmax, hl);
                    let t = _mm256_subs_epi8(h0, oe_del);
                    let eout = _mm256_max_epi8(t, _mm256_subs_epi8($ein, e_del));
                    let ti = if SHARE {
                        t
                    } else {
                        _mm256_subs_epi8(h0, oe_ins)
                    };
                    $f = _mm256_max_epi8(ti, _mm256_subs_epi8($f, e_ins));
                    (h, eout)
                }};
            }
            // Row bookkeeping: publish the row max, update the lanes' best, close the runs of
            // improving rows that ended on the previous row. `$buf` holds this row's H.
            // Rows `x0..x0 + n` (`n <= RING`), as the NEON kernel's `due`.
            macro_rules! due {
                ($x0:expr, $n:expr) => {{
                    let (x0, n): (usize, usize) = ($x0, $n);
                    // `off < n` unsigned, as `min(off, n - 1) == off`.
                    let off = _mm256_sub_epi8(te_lo, _mm256_set1_epi8(x0 as i8));
                    let near = _mm256_cmpeq_epi8(
                        _mm256_min_epu8(off, _mm256_set1_epi8(n.max(1) as i8 - 1)),
                        off,
                    );
                    if n > 0 && _mm256_movemask_epi8(_mm256_and_si256(near, pm)) != 0 {
                        for x in x0..x0 + n {
                            let m = _mm256_and_si256(
                                _mm256_cmpeq_epi8(te_lo, _mm256_set1_epi8(x as i8)),
                                pm,
                            );
                            if _mm256_movemask_epi8(m) != 0 {
                                qe_scan!(slot(x), x as i32, m);
                                pm = _mm256_andnot_si256(m, pm);
                            }
                        }
                    }
                }};
            }
            // Row bookkeeping: publish the row max, update the lanes' best.
            macro_rules! finish_row {
                ($row:expr, $rmax:expr, $rref:expr) => {{
                    st!(rout.add($row * L), _mm256_xor_si256($rmax, hz));
                    let imp = _mm256_andnot_si256(
                        _mm256_cmpgt_epi8(nil, $rref),
                        _mm256_cmpgt_epi8($rmax, gmax),
                    );
                    gmax = _mm256_blendv_epi8(gmax, $rmax, imp);
                    te_lo = _mm256_blendv_epi8(te_lo, _mm256_set1_epi8($row as i8), imp);
                    if $row >= 256 {
                        te_hi = _mm256_blendv_epi8(te_hi, _mm256_set1_epi8(($row >> 8) as i8), imp);
                    }
                    pm = _mm256_or_si256(
                        pm,
                        _mm256_andnot_si256(_mm256_cmpgt_epi8(thrv, $rmax), imp),
                    );
                }};
            }

            let mut r = 0usize;
            while r < nr {
                let ri = r as i32;
                let khi = wm1.min(ri + omax);
                let klo = 0.max(ri + ominq);
                let kun = klo.max(ri + omaskq);
                let mut k = khi;
                let hp = slot(r + RING - 1);
                if r + 1 < nr {
                    // Rows `r`, `r + 1` take the slots of `r - RING..`: read those first.
                    if r >= RING {
                        due!(r - RING, 2);
                    }
                    let (hr0, hc) = (slot(r), slot(r + 1));
                    self.cells += 2 * (khi - klo + 1).max(0) as u64;
                    let rref0 = ld!(self.refv.as_ptr().add(r * L));
                    let rref1 = ld!(self.refv.as_ptr().add((r + 1) * L));
                    let (mut f0, mut f1, mut d1) = (hz, hz, hz);
                    let (mut rmax0, mut rmax1) = (hz, hz);
                    macro_rules! step2 {
                        ($mask:expr, $live:expr) => {{
                            let ko = k as isize;
                            let valid = |i: i32| -> Option<bool> {
                                if $mask {
                                    Some((0..=wm1).contains(&(k + i)))
                                } else {
                                    None
                                }
                            };
                            let q = ld!(a.offset((r as isize + w as isize - 1 - ko) * li));
                            // DEAD is the only query code at or above 0xC0.
                            let clamp = if $live {
                                Some(_mm256_xor_si256(_mm256_cmpeq_epi8(q, deadv), x7f))
                            } else {
                                None
                            };
                            let e0 = ld!(e.offset(ko * li));
                            let d0 = ld!(hp.offset(ko * li));
                            let (h0, e1) = cell!(d0, e0, f0, q, rref0, rmax0, valid(0), clamp);
                            let (h1, e2) = cell!(d1, e1, f1, q, rref1, rmax1, valid(1), clamp);
                            st!(hr0.offset(ko * li), h0);
                            st!(hc.offset((ko + 1) * li), h1);
                            st!(e.offset((ko + 2) * li), e2);
                            d1 = h0;
                            k -= 1;
                        }};
                    }
                    while k >= klo && k > wm1 - 1 {
                        step2!(true, true);
                    }
                    while k >= kun {
                        step2!(false, false);
                    }
                    while k >= klo {
                        step2!(false, true);
                    }
                    let kend = (ri + ominq).max(-1);
                    while k >= kend {
                        step2!(true, true);
                    }
                    finish_row!(r, rmax0, rref0);
                    finish_row!(r + 1, rmax1, rref1);
                    r += 2;
                } else {
                    if r >= RING {
                        due!(r - RING, 1);
                    }
                    let hc = slot(r);
                    self.cells += (khi - klo + 1).max(0) as u64;
                    let rref = ld!(self.refv.as_ptr().add(r * L));
                    let (mut f, mut rmax) = (hz, hz);
                    macro_rules! step1 {
                        ($live:expr) => {{
                            let ko = k as isize;
                            let q = ld!(a.offset((r as isize + w as isize - 1 - ko) * li));
                            let clamp = if $live {
                                Some(_mm256_xor_si256(_mm256_cmpeq_epi8(q, deadv), x7f))
                            } else {
                                None
                            };
                            let e0 = ld!(e.offset(ko * li));
                            let d0 = ld!(hp.offset(ko * li));
                            let (h, e1) = cell!(d0, e0, f, q, rref, rmax, None, clamp);
                            st!(hc.offset(ko * li), h);
                            st!(e.offset((ko + 1) * li), e1);
                            k -= 1;
                        }};
                    }
                    while k >= kun {
                        step1!(false);
                    }
                    while k >= klo {
                        step1!(true);
                    }
                    finish_row!(r, rmax, rref);
                    r += 1;
                }
                if early && _mm256_movemask_epi8(_mm256_cmpgt_epi8(tgt, gmax)) == 0 {
                    break;
                }
            }
            let x0 = r.saturating_sub(RING);
            due!(x0, r - x0);
            st!(self.gmax.as_mut_ptr(), _mm256_xor_si256(gmax, hz));
            let (mut lo, mut hi) = ([0u8; L], [0u8; L]);
            st!(lo.as_mut_ptr(), te_lo);
            st!(hi.as_mut_ptr(), te_hi);
            st!(self.qe.as_mut_ptr(), qev);
            self.te = std::array::from_fn(|l| {
                if self.gmax[l] > 0 {
                    i32::from(lo[l]) | i32::from(hi[l]) << 8
                } else {
                    -1
                }
            });
            // Row maxima to lane-major, zero past `nr`.
            self.rstride = nrp;
            fit(&mut self.rl, nrp * L);
            self.r[nr * L..nrp * L].fill(0);
            for blk in 0..L / 16 {
                for b in (0..nrp).step_by(16) {
                    t16(
                        self.r.as_ptr().add(b * L + blk * 16),
                        L,
                        self.rl.as_mut_ptr().add(blk * 16 * nrp + b),
                        nrp,
                    );
                }
            }
        }
    }
}

/// The AVX-512BW band kernel: the AVX2 kernel at 64 lanes, with mask registers for the live
/// lanes and the row bookkeeping.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::avx2::HZ;
    use super::RING;
    use crate::batched::{x86_row::t16, ABuf};
    use std::arch::x86_64::*;

    pub(super) const L: usize = 64;
    const DEAD: u8 = 0xC0;
    const QPAD: u8 = 0x80;
    const RPAD: u8 = 0x80;

    pub(super) use super::avx2::Lane;

    /// As the NEON kernel's `Work`.
    pub(super) struct Work {
        st: Vec<u8>,
        a: ABuf<u8>,
        refv: ABuf<u8>,
        /// H of the last [`RING`] rows, row `x` in slot `x % RING`.
        h: ABuf<u8>,
        e: ABuf<u8>,
        r: ABuf<u8>,
        rl: Vec<u8>,
        rstride: usize,
        pub gmax: [u8; L],
        pub te: [i32; L],
        pub qe: [u8; L],
        pub cells: u64,
    }

    impl Default for Work {
        fn default() -> Self {
            Work {
                st: Vec::new(),
                a: ABuf::new(),
                refv: ABuf::new(),
                h: ABuf::new(),
                e: ABuf::new(),
                r: ABuf::new(),
                rl: Vec::new(),
                rstride: 0,
                gmax: [0; L],
                te: [0; L],
                qe: [0; L],
                cells: 0,
            }
        }
    }

    fn fit(v: &mut Vec<u8>, n: usize) {
        if v.len() < n {
            v.resize(n, 0);
        }
    }

    impl Work {
        pub(super) fn lane_rows(&self, l: usize, nrows: usize) -> &[u8] {
            &self.rl[l * self.rstride..l * self.rstride + nrows]
        }

        /// Run one group of `lanes` (at most `L`) at width `w`; `tbl` is the signed score table,
        /// `gaps` each under 128.
        ///
        /// # Safety
        /// AVX-512BW available.
        pub(super) unsafe fn run(
            &mut self,
            lanes: &[Lane],
            w: usize,
            tbl: &[u8; 16],
            gaps: [u8; 4],
        ) {
            if gaps[0] == gaps[2] {
                self.run_impl::<true>(lanes, w, tbl, gaps)
            } else {
                self.run_impl::<false>(lanes, w, tbl, gaps)
            }
        }

        #[target_feature(enable = "avx512bw")]
        unsafe fn run_impl<const SHARE: bool>(
            &mut self,
            lanes: &[Lane],
            w: usize,
            tbl: &[u8; 16],
            gaps: [u8; 4],
        ) {
            let nr = lanes.iter().map(|l| l.t.len()).max().unwrap_or(0);
            let np = nr + w - 1;
            let c = w as i32 - 1;
            let (npp, nrp) = ((np + 4).next_multiple_of(16), nr.next_multiple_of(16));
            let stride = npp.max(nrp);
            let hs = w + 7;
            fit(&mut self.st, 16 * stride);
            self.a.fit(npp * L, 0);
            self.refv.fit(nrp * L, 0);
            self.h.fit(RING * hs * L, HZ);
            self.e.fit((w + 9) * L, HZ);
            self.r.fit(nrp * L, 0);
            // All of it: a row reads the row above's cells past that row's own top, never written.
            self.h[..RING * hs * L].fill(HZ);
            self.e[..(w + 9) * L].fill(HZ);
            let (mut omax, mut ominq, mut omaskq) = (i32::MIN, i32::MAX, i32::MIN);
            let mut ob = [0u8; L];
            // The SoA buffers through a lane-major staging area, 16 lanes at a time, and 16x16
            // transposes (as the NEON kernel): the query in band layout, then the reference rows.
            let st = self.st.as_mut_ptr();
            for blk in 0..L / 16 {
                for i in 0..16 {
                    let row = std::slice::from_raw_parts_mut(st.add(i * stride), npp);
                    crate::batched::fill_fast(row, DEAD);
                    let l = blk * 16 + i;
                    let Some(lane) = lanes.get(l) else { continue };
                    let pq0 = c - lane.o;
                    let len2 = lane.q.len() as i32;
                    let (p0, p1) = (pq0.max(0), (pq0 + len2).min(np as i32));
                    if p0 < p1 {
                        crate::batched::copy_fast(
                            &mut row[p0 as usize..p1 as usize],
                            &lane.q[(p0 - pq0) as usize..(p1 - pq0) as usize],
                        );
                    }
                    let (p0, p1) = ((pq0 + len2).max(0), (pq0 + lane.quanta).min(np as i32));
                    if p0 < p1 {
                        crate::batched::fill_fast(&mut row[p0 as usize..p1 as usize], QPAD);
                    }
                    ob[l] = lane.o as u8;
                    omax = omax.max(lane.o);
                    ominq = ominq.min(lane.o - lane.quanta + 1);
                    omaskq = omaskq.max(lane.o - lane.quanta + 1);
                }
                for b in (0..npp).step_by(16) {
                    t16(
                        st.add(b),
                        stride,
                        self.a.as_mut_ptr().add(b * L + blk * 16),
                        L,
                    );
                }
                for i in 0..16 {
                    let row = std::slice::from_raw_parts_mut(st.add(i * stride), nrp);
                    let Some(lane) = lanes.get(blk * 16 + i) else {
                        crate::batched::fill_fast(row, RPAD);
                        continue;
                    };
                    crate::batched::copy_fast(&mut row[..lane.t.len()], lane.t);
                    crate::batched::fill_fast(&mut row[lane.t.len()..], RPAD);
                    // Target N re-encoded as 12 (see the NEON kernel).
                    if lane.t.iter().fold(0u8, |m, &b| m | b) & !3 != 0 {
                        for b in row[..lane.t.len()].iter_mut() {
                            if *b == 4 {
                                *b = 12;
                            }
                        }
                    }
                }
                for b in (0..nrp).step_by(16) {
                    t16(
                        st.add(b),
                        stride,
                        self.refv.as_mut_ptr().add(b * L + blk * 16),
                        L,
                    );
                }
            }
            // The signed table in every 128-bit lane.
            let tblv = _mm512_broadcast_i32x4(_mm_loadu_si128(tbl.as_ptr() as *const __m128i));
            let hz = _mm512_set1_epi8(HZ as i8);
            let nil = _mm512_setzero_si512();
            let one = _mm512_set1_epi8(1);
            let deadv = _mm512_set1_epi8(DEAD as i8);
            let rpadv = _mm512_set1_epi8(RPAD as i8);
            let (oe_del, e_del, oe_ins, e_ins) = (
                _mm512_set1_epi8(gaps[0] as i8),
                _mm512_set1_epi8(gaps[1] as i8),
                _mm512_set1_epi8(gaps[2] as i8),
                _mm512_set1_epi8(gaps[3] as i8),
            );
            let ov = _mm512_loadu_si512(ob.as_ptr() as *const _);
            // Targets in the H domain; an absent lane's (HZ) is always reached.
            let mut tb = [HZ; L];
            for (l, lane) in lanes.iter().enumerate() {
                tb[l] = lane.target ^ HZ;
            }
            let early = lanes.iter().any(|l| l.target > 0);
            let tgt = _mm512_loadu_si512(tb.as_ptr() as *const _);
            let a = self.a.as_ptr();
            let e = self.e.as_mut_ptr().add(3 * L);
            let rout = self.r.as_mut_ptr();
            let h0p = self.h.as_mut_ptr().add(3 * L);
            // Row `x`'s H (`x` = -1: slot RING - 1, still zero).
            let slot = |x: usize| h0p.add(x % RING * hs * L);
            let li = L as isize;
            let mut gmax = hz;
            // `te` as its low and high bytes per lane, written with the row on every improving
            // row; a lane that never improves keeps `gmax == 0` and reads back as -1.
            let (mut te_lo, mut te_hi, mut qev) = (nil, nil, nil);
            let wm1 = w as i32 - 1;
            // `qe` from H as the NEON kernel's `qe_scan` and `due`; `thrv` in the H domain.
            let mut pm: __mmask64 = 0;
            let mut th = [HZ; L];
            for (l, lane) in lanes.iter().enumerate() {
                th[l] = lane.thr ^ HZ;
            }
            let thrv = _mm512_loadu_si512(th.as_ptr() as *const _);

            macro_rules! ld {
                ($p:expr) => {
                    _mm512_loadu_si512($p as *const _)
                };
            }
            macro_rules! st {
                ($p:expr, $v:expr) => {
                    _mm512_storeu_si512($p as *mut _, $v)
                };
            }
            macro_rules! qe_scan {
                ($buf:expr, $row:expr, $msk:expr) => {{
                    let (kl, kh) = (0.max($row + ominq), wm1.min($row + omax));
                    let mut jv = _mm512_add_epi8(_mm512_set1_epi8(($row - kl) as i8), ov);
                    let mut best = nil;
                    for k in kl..=kh {
                        let eq = _mm512_cmpeq_epi8_mask(ld!($buf.add(k as usize * L)), gmax);
                        best = _mm512_mask_mov_epi8(best, eq, jv);
                        jv = _mm512_sub_epi8(jv, one);
                    }
                    qev = _mm512_mask_mov_epi8(qev, $msk, best);
                }};
            }
            // One cell, as the AVX2 kernel's; `live` is the mask of lanes on a live cell (`None`:
            // all).
            macro_rules! cell {
                ($d:expr, $ein:expr, $f:ident, $q:expr, $rref:expr, $rmax:ident, $top:expr,
                 $live:expr) => {{
                    let sc = _mm512_shuffle_epi8(tblv, _mm512_xor_si512($q, $rref));
                    let mut h0 = _mm512_max_epi8(_mm512_adds_epi8($d, sc), $ein);
                    let ok: Option<bool> = $top;
                    if ok == Some(false) {
                        h0 = hz;
                    }
                    let mut h = _mm512_max_epi8(h0, $f);
                    if ok == Some(false) {
                        h = hz;
                    }
                    let lv: Option<__mmask64> = $live;
                    $rmax = match lv {
                        Some(m) => _mm512_mask_max_epi8($rmax, m, $rmax, h),
                        None => _mm512_max_epi8($rmax, h),
                    };
                    let t = _mm512_subs_epi8(h0, oe_del);
                    let eout = _mm512_max_epi8(t, _mm512_subs_epi8($ein, e_del));
                    let ti = if SHARE {
                        t
                    } else {
                        _mm512_subs_epi8(h0, oe_ins)
                    };
                    $f = _mm512_max_epi8(ti, _mm512_subs_epi8($f, e_ins));
                    (h, eout)
                }};
            }
            // Rows `x0..x0 + n` (`n <= RING`), as the NEON kernel's `due`.
            macro_rules! due {
                ($x0:expr, $n:expr) => {{
                    let (x0, n): (usize, usize) = ($x0, $n);
                    let off = _mm512_sub_epi8(te_lo, _mm512_set1_epi8(x0 as i8));
                    if _mm512_mask_cmplt_epu8_mask(pm, off, _mm512_set1_epi8(n as i8)) != 0 {
                        for x in x0..x0 + n {
                            let m =
                                _mm512_mask_cmpeq_epi8_mask(pm, te_lo, _mm512_set1_epi8(x as i8));
                            if m != 0 {
                                qe_scan!(slot(x), x as i32, m);
                                pm &= !m;
                            }
                        }
                    }
                }};
            }
            // Row bookkeeping: publish the row max, update the lanes' best.
            macro_rules! finish_row {
                ($row:expr, $rmax:expr, $rref:expr) => {{
                    st!(rout.add($row * L), _mm512_xor_si512($rmax, hz));
                    let imp =
                        _mm512_cmpgt_epi8_mask($rmax, gmax) & _mm512_cmplt_epu8_mask($rref, rpadv);
                    gmax = _mm512_mask_mov_epi8(gmax, imp, $rmax);
                    te_lo = _mm512_mask_mov_epi8(te_lo, imp, _mm512_set1_epi8($row as i8));
                    if $row >= 256 {
                        te_hi =
                            _mm512_mask_mov_epi8(te_hi, imp, _mm512_set1_epi8(($row >> 8) as i8));
                    }
                    pm |= imp & _mm512_cmpge_epi8_mask($rmax, thrv);
                }};
            }

            let mut r = 0usize;
            while r < nr {
                let ri = r as i32;
                let khi = wm1.min(ri + omax);
                let klo = 0.max(ri + ominq);
                let kun = klo.max(ri + omaskq);
                let mut k = khi;
                let hp = slot(r + RING - 1);
                if r + 1 < nr {
                    // Rows `r`, `r + 1` take the slots of `r - RING..`: read those first.
                    if r >= RING {
                        due!(r - RING, 2);
                    }
                    let (hr0, hc) = (slot(r), slot(r + 1));
                    self.cells += 2 * (khi - klo + 1).max(0) as u64;
                    let rref0 = ld!(self.refv.as_ptr().add(r * L));
                    let rref1 = ld!(self.refv.as_ptr().add((r + 1) * L));
                    let (mut f0, mut f1, mut d1) = (hz, hz, hz);
                    let (mut rmax0, mut rmax1) = (hz, hz);
                    macro_rules! step2 {
                        ($mask:expr, $live:expr) => {{
                            let ko = k as isize;
                            let valid = |i: i32| -> Option<bool> {
                                if $mask {
                                    Some((0..=wm1).contains(&(k + i)))
                                } else {
                                    None
                                }
                            };
                            let q = ld!(a.offset((r as isize + w as isize - 1 - ko) * li));
                            let live = if $live {
                                Some(_mm512_cmplt_epu8_mask(q, deadv))
                            } else {
                                None
                            };
                            let e0 = ld!(e.offset(ko * li));
                            let d0 = ld!(hp.offset(ko * li));
                            let (h0, e1) = cell!(d0, e0, f0, q, rref0, rmax0, valid(0), live);
                            let (h1, e2) = cell!(d1, e1, f1, q, rref1, rmax1, valid(1), live);
                            st!(hr0.offset(ko * li), h0);
                            st!(hc.offset((ko + 1) * li), h1);
                            st!(e.offset((ko + 2) * li), e2);
                            d1 = h0;
                            k -= 1;
                        }};
                    }
                    while k >= klo && k > wm1 - 1 {
                        step2!(true, true);
                    }
                    while k >= kun {
                        step2!(false, false);
                    }
                    while k >= klo {
                        step2!(false, true);
                    }
                    let kend = (ri + ominq).max(-1);
                    while k >= kend {
                        step2!(true, true);
                    }
                    finish_row!(r, rmax0, rref0);
                    finish_row!(r + 1, rmax1, rref1);
                    r += 2;
                } else {
                    if r >= RING {
                        due!(r - RING, 1);
                    }
                    let hc = slot(r);
                    self.cells += (khi - klo + 1).max(0) as u64;
                    let rref = ld!(self.refv.as_ptr().add(r * L));
                    let (mut f, mut rmax) = (hz, hz);
                    macro_rules! step1 {
                        ($live:expr) => {{
                            let ko = k as isize;
                            let q = ld!(a.offset((r as isize + w as isize - 1 - ko) * li));
                            let live = if $live {
                                Some(_mm512_cmplt_epu8_mask(q, deadv))
                            } else {
                                None
                            };
                            let e0 = ld!(e.offset(ko * li));
                            let d0 = ld!(hp.offset(ko * li));
                            let (h, e1) = cell!(d0, e0, f, q, rref, rmax, None, live);
                            st!(hc.offset(ko * li), h);
                            st!(e.offset((ko + 1) * li), e1);
                            k -= 1;
                        }};
                    }
                    while k >= kun {
                        step1!(false);
                    }
                    while k >= klo {
                        step1!(true);
                    }
                    finish_row!(r, rmax, rref);
                    r += 1;
                }
                if early && _mm512_cmpge_epi8_mask(gmax, tgt) == u64::MAX {
                    break;
                }
            }
            let x0 = r.saturating_sub(RING);
            due!(x0, r - x0);
            st!(self.gmax.as_mut_ptr(), _mm512_xor_si512(gmax, hz));
            let (mut lo, mut hi) = ([0u8; L], [0u8; L]);
            st!(lo.as_mut_ptr(), te_lo);
            st!(hi.as_mut_ptr(), te_hi);
            st!(self.qe.as_mut_ptr(), qev);
            self.te = std::array::from_fn(|l| {
                if self.gmax[l] > 0 {
                    i32::from(lo[l]) | i32::from(hi[l]) << 8
                } else {
                    -1
                }
            });
            // Row maxima to lane-major, zero past `nr`.
            self.rstride = nrp;
            fit(&mut self.rl, nrp * L);
            self.r[nr * L..nrp * L].fill(0);
            for blk in 0..L / 16 {
                for b in (0..nrp).step_by(16) {
                    t16(
                        self.r.as_ptr().add(b * L + blk * 16),
                        L,
                        self.rl.as_mut_ptr().add(blk * 16 * nrp + b),
                        nrp,
                    );
                }
            }
        }
    }
}

/// The x86 band kernel: AVX-512BW (64 lanes) where available and not capped by
/// `BWA4_RESCUE_TIER`, else AVX2 (32 lanes). Same outputs either way; the group size is the only
/// thing that differs, read through [`Work::lanes`].
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{avx2, avx512};
    pub(super) use avx2::Lane;

    /// The widest group either kernel takes.
    pub(super) const L: usize = avx512::L;

    pub(super) struct Work {
        w2: avx2::Work,
        w5: avx512::Work,
        wide: bool,
        pub gmax: [u8; L],
        pub te: [i32; L],
        pub qe: [u8; L],
        pub cells: u64,
        pub kernel_ns: u64,
    }

    impl Default for Work {
        fn default() -> Self {
            Work {
                w2: Default::default(),
                w5: Default::default(),
                wide: crate::matesw::rescue_avx512_ok(),
                gmax: [0; L],
                te: [0; L],
                qe: [0; L],
                cells: 0,
                kernel_ns: 0,
            }
        }
    }

    impl Work {
        /// Lanes per group of the kernel this machine runs.
        pub(super) fn lanes(&self) -> usize {
            if self.wide {
                avx512::L
            } else {
                avx2::L
            }
        }

        pub(super) fn lane_rows(&self, l: usize, nrows: usize) -> &[u8] {
            if self.wide {
                self.w5.lane_rows(l, nrows)
            } else {
                self.w2.lane_rows(l, nrows)
            }
        }

        /// # Safety
        /// AVX2 available (AVX-512BW too when `wide`, which `rescue_avx512_ok` checked).
        pub(super) unsafe fn run(
            &mut self,
            lanes: &[Lane],
            w: usize,
            tbl: &[u8; 16],
            gaps: [u8; 4],
        ) {
            let n = lanes.len();
            if self.wide {
                self.w5.run(lanes, w, tbl, gaps);
                self.gmax[..n].copy_from_slice(&self.w5.gmax[..n]);
                self.te[..n].copy_from_slice(&self.w5.te[..n]);
                self.qe[..n].copy_from_slice(&self.w5.qe[..n]);
                self.cells = self.w5.cells;
            } else {
                self.w2.run(lanes, w, tbl, gaps);
                self.gmax[..n].copy_from_slice(&self.w2.gmax[..n]);
                self.te[..n].copy_from_slice(&self.w2.te[..n]);
                self.qe[..n].copy_from_slice(&self.w2.qe[..n]);
                self.cells = self.w2.cells;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matesw::{batched_align, KswJob};
    use crate::rescue_prune::tests::{load_dump, scmat};
    use bwa_extend::ksw_align2;

    /// The streamed `score2` equals `SuboptimalTracker` on random row maxima: plateaus, gaps,
    /// values around `minsc`, rows offset by `row0`.
    #[test]
    fn score2_equals_tracker() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as usize
        };
        for _ in 0..20000 {
            let n = next() % 300;
            // Half the rounds walk in small steps, for plateaus and slopes (the tracker's
            // alternating case), the rest jump.
            let walk = next() % 2 == 0;
            let mut cur = 30i32;
            let rows: Vec<u8> = (0..n)
                .map(|_| {
                    if walk {
                        cur = (cur + (next() % 5) as i32 - 2).clamp(0, 60);
                        return cur as u8;
                    }
                    match next() % 4 {
                        0 => 0,
                        1 => (15 + next() % 10) as u8,
                        _ => (next() % 60) as u8,
                    }
                })
                .collect();
            let minsc = match next() % 8 {
                0 => 300,
                1 => (next() % 3) as i32 - 1,
                _ => 1 + (next() % 60) as i32,
            };
            let (row0, max_sc) = ((next() % 50) as i32, 1 + (next() % 2) as i32);
            let s = i32::from(rows.iter().copied().max().unwrap_or(0));
            let te = row0 + if n > 0 { (next() % n) as i32 } else { -1 };
            let mut tr = bwa_extend::SuboptimalTracker::new();
            for (i, &v) in rows.iter().enumerate() {
                tr.push_row(row0 + i as i32, i32::from(v), minsc);
            }
            assert_eq!(
                score2(&rows, row0, s, te, minsc, max_sc),
                tr.finish(s, te, max_sc),
                "rows {rows:?} row0 {row0} te {te} minsc {minsc}"
            );
        }
    }

    /// The batched path (which bands every eligible pruned job) against `ksw_align2` on the full
    /// window: every consumed result (`score >= minsc`) must be identical, on generated jobs with
    /// planted copies, repeats and indels, and on every `STRIDE`-th dumped real job when
    /// `BWA4_PRUNE_BENCH` names a dump.
    #[test]
    fn bands_reproduce_ksw_align2() {
        let mut state = 0x7a3d_19c4_e5b2_0f61u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut gen: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for _ in 0..8000 {
            let qlen = 20 + next() % 160;
            let tlen = qlen + next() % 1400;
            let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 4) as u8).collect();
            if next() % 6 == 0 {
                let period = 1 + next() % 4;
                let at = next() % tlen;
                let n = (next() % 300).min(tlen - at);
                for x in 0..n {
                    t[at + x] = t[at + x % period];
                }
            }
            let mut q: Vec<u8> = (0..qlen).map(|_| (next() % 4) as u8).collect();
            for _ in 0..next() % 4 {
                let l = (10 + next() % qlen.min(80)).min(qlen);
                let (qs, at) = (next() % (qlen - l + 1), next() % (tlen - l + 1));
                t[at..at + l].copy_from_slice(&q[qs..qs + l]);
                for _ in 0..next() % 4 {
                    t[at + next() % l] = (next() % 4) as u8;
                }
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
            gen.push((t, q));
        }
        let mut sets = vec![(gen, 19)];
        if let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") {
            let stride: usize = std::env::var("STRIDE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20);
            let jobs: Vec<_> = load_dump(&path).into_iter().step_by(stride).collect();
            sets.push((jobs, 19));
        }
        let mat = scmat(1, 4);
        let p = Params::from(1, 4, 6, 1, 6, 1, 19).unwrap();
        let (mut banded, mut checked) = (0usize, 0usize);
        for (set, minsc) in &sets {
            for chunk in set.chunks(1000) {
                let jobs: Vec<KswJob> = chunk
                    .iter()
                    .map(|(t, q)| KswJob {
                        query: q,
                        target: t,
                    })
                    .collect();
                let got = batched_align(&jobs, 5, &mat, 6, 1, 6, 1, *minsc, 1, Some(p));
                for (i, (t, q)) in chunk.iter().enumerate() {
                    let full = ksw_align2(q, t, 5, &mat, 6, 1, 6, 1, *minsc, 1, 16);
                    let consumed = |r: &KswAlignResult| r.score >= *minsc;
                    checked += 1;
                    if consumed(&full) || consumed(&got[i]) {
                        assert_eq!(got[i], full, "job {i} of a set of {}", set.len());
                    }
                }
                let mut b = Batch::default();
                for (i, (t, q)) in chunk.iter().enumerate() {
                    let quanta = q.len().div_ceil(16) * 16;
                    if q.len() < 250 && b.plan(i, t, q, &p, usize::MAX, quanta).is_none() {
                        banded += 1;
                    }
                }
            }
        }
        eprintln!("bands: {banded} banded of {checked}");
        assert!(
            banded > checked / 20,
            "too few banded jobs: {banded} of {checked}"
        );
    }

    /// The 64-lane AVX-512 band kernel against two 32-lane AVX2 groups on the same lanes: every
    /// lane's best score, its end, and its row maxima must match. Runs only where `avx512bw` is
    /// detected (the `avx512-check` workflow runs it under Intel SDE).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_band_kernel_matches_avx2() {
        if !std::arch::is_x86_feature_detected!("avx512bw") {
            eprintln!("skipping avx512: avx512bw not detected");
            return;
        }
        let mut state = 0x51d2_88a3_04c7_e6b9u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let tbl = crate::matesw::u8_score_table(&scmat(1, 4));
        let gaps = [7u8, 1, 7, 1];
        let (mut w2, mut w5) = (avx2::Work::default(), avx512::Work::default());
        for round in 0..300 {
            let w = 1 + next() % 48;
            let seqs: Vec<(Vec<u8>, Vec<u8>, i32)> = (0..64)
                .map(|_| {
                    let qlen = 20 + next() % 180;
                    let tlen = 1 + next() % 300;
                    let q: Vec<u8> = (0..qlen).map(|_| (next() % 5) as u8).collect();
                    let mut t: Vec<u8> = (0..tlen).map(|_| (next() % 5) as u8).collect();
                    // Plant a stretch of the query so lanes score above zero.
                    let l = (next() % qlen).min(tlen);
                    let at = next() % (tlen - l + 1);
                    t[at..at + l].copy_from_slice(&q[..l]);
                    let o = (next() % qlen) as i32;
                    (q, t, o)
                })
                .collect();
            let lanes: Vec<avx2::Lane> = seqs
                .iter()
                .map(|(q, t, o)| avx2::Lane {
                    q,
                    t,
                    quanta: q.len().div_ceil(16) as i32 * 16,
                    o: *o,
                    w: w as i32,
                    target: 0,
                    thr: (next() % 40) as u8,
                })
                .collect();
            let n5 = 33 + next() % 32;
            unsafe { w5.run(&lanes[..n5], w, &tbl, gaps) };
            for half in 0..2 {
                let (lo, hi) = (half * 32, (half * 32 + 32).min(n5));
                if lo >= hi {
                    continue;
                }
                unsafe { w2.run(&lanes[lo..hi], w, &tbl, gaps) };
                for l in lo..hi {
                    let k = l - lo;
                    let msg = format!("round {round} lane {l} w {w}");
                    assert_eq!(w5.gmax[l], w2.gmax[k], "gmax {msg}");
                    assert_eq!(w5.te[l], w2.te[k], "te {msg}");
                    if w2.gmax[k] > 0 && w2.gmax[k] >= lanes[l].thr {
                        assert_eq!(w5.qe[l], w2.qe[k], "qe {msg}");
                    }
                    let nr = lanes[l].t.len();
                    assert_eq!(w5.lane_rows(l, nr), w2.lane_rows(k, nr), "rows {msg}");
                }
            }
        }
    }

    /// [`bands_reproduce_ksw_align2`] again, named for the `avx512-check` filter: under SDE the x86
    /// kernel resolves to AVX-512, so this is the end-to-end check of the 64-lane groups.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_bands_reproduce_ksw_align2() {
        if !std::arch::is_x86_feature_detected!("avx512bw") {
            eprintln!("skipping avx512: avx512bw not detected");
            return;
        }
        assert!(Batch::default().work.lanes() == 64);
        bands_reproduce_ksw_align2();
    }

    /// Planning statistics on a dump at gate 1000: why parents are declined, and the cells the
    /// bands cost against their hulls.
    /// `BWA4_PRUNE_BENCH=jobs.bin cargo test --release -p bwa-mem4-neon band_stats -- --ignored --nocapture`
    static WHY: [std::sync::atomic::AtomicU64; 4] =
        [const { std::sync::atomic::AtomicU64::new(0) }; 4];
    #[test]
    #[ignore]
    fn band_stats() {
        let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") else {
            return;
        };
        let jobs = load_dump(&path);
        let p = Params::from(1, 4, 6, 1, 6, 1, 19).unwrap();
        let mut b = Batch::default();
        let (mut hull, mut banded, mut t1n) = (0u64, 0u64, 0u64);
        let (mut c_hull_all, mut c_hull_banded, mut c_band1, mut c_band_hull_declined) =
            (0u64, 0u64, 0u64, 0u64);
        let (mut nc19, mut nct, mut ncap, mut nnone, mut ncost) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let (mut wsum, mut rsum, mut nb) = (0u64, 0u64, 0u64);
        for (i, (t, q)) in jobs.iter().enumerate() {
            if q.len() >= 250 {
                continue;
            }
            let quanta = q.len().div_ceil(16) * 16;
            b.clear();
            let (dec, t1) = rescue_prune::decide_bands(t, q, &p, 1000, &mut b.c19, &mut b.ct);
            let Decision::Hull { hb, he } = dec else {
                continue;
            };
            hull += 1;
            let hc = (he - hb + 1) as u64 * quanta as u64;
            c_hull_all += hc;
            nc19 += b.c19.len() as u64;
            if t1 > p.minsc {
                t1n += 1;
                nct += b.ct.len() as u64;
            }
            if b.c19.len() > COMP_CAP {
                ncap += 1;
                continue;
            }
            let r1 = if t1 > p.minsc { &b.ct } else { &b.c19 };
            let mut cost = 0i64;
            let mut bad = false;
            for c in r1 {
                let i0 = c.i0.max(0);
                let i1 =
                    (t.len() as i32 - 1).min(c.dmaxhit + quanta as i32 - 1 + p.tail_at(c.ub, t1));
                if i0 < hb as i32 || i1 > he as i32 || i0 > i1 {
                    WHY[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    let dlo = c.dlo.max(i0 - quanta as i32 + 1);
                    let dhi = c.dhi.min(i1);
                    if dlo > dhi {
                        WHY[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else if i0.max(dlo) > i1.min(dhi + quanta as i32 - 1) {
                        WHY[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                match to_band(
                    c,
                    &p,
                    t1,
                    t.len() as i32,
                    quanta as i32,
                    hb as i32,
                    he as i32,
                ) {
                    Some(bd) => {
                        cost += (bd.r1 - bd.r0 + 1) as i64
                            * ((bd.dhi - bd.dlo + 1).min(quanta as i32) as i64 + OVH);
                        wsum += (bd.dhi - bd.dlo + 1) as u64;
                        rsum += (bd.r1 - bd.r0 + 1) as u64;
                        nb += 1;
                    }
                    None => bad = true,
                }
            }
            if bad {
                nnone += 1;
                continue;
            }
            if cost >= hc as i64 {
                ncost += 1;
                c_band_hull_declined += hc;
                continue;
            }
            if b.plan(i, t, q, &p, 1000, quanta).is_some() {
                nnone += 1;
                continue;
            }
            banded += 1;
            c_hull_banded += hc;
            c_band1 += cost as u64;
        }
        eprintln!(
            "WHY none: hull {} dlo>dhi {} w>255 {} rows {}",
            WHY[0].load(std::sync::atomic::Ordering::Relaxed),
            WHY[1].load(std::sync::atomic::Ordering::Relaxed),
            WHY[2].load(std::sync::atomic::Ordering::Relaxed),
            WHY[3].load(std::sync::atomic::Ordering::Relaxed)
        );
        eprintln!("STATS hull {hull}: banded {banded}, declined cap {ncap} none {nnone} cost {ncost}; T1>minsc {t1n}; avg c19 {:.1} ct {:.1}; bands avg w {:.1} rows {:.1}",
            nc19 as f64 / hull as f64, nct as f64 / t1n.max(1) as f64, wsum as f64 / nb.max(1) as f64, rsum as f64 / nb.max(1) as f64);
        eprintln!("STATS cells: hull all {c_hull_all}; banded parents hull {c_hull_banded} -> bands {c_band1} ({:.1}%); cost-declined hull {c_band_hull_declined}",
            100.0 * c_band1 as f64 / c_hull_banded.max(1) as f64);
    }

    /// Kernel timing on a dump at gate 1000, in chunks of 1000 jobs as production sees them:
    /// the bands against the hull jobs through the forward kernel.
    /// `BWA4_PRUNE_BENCH=jobs.bin cargo test --release -p bwa-mem4-neon band_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn band_bench() {
        let Ok(path) = std::env::var("BWA4_PRUNE_BENCH") else {
            return;
        };
        let jobs = load_dump(&path);
        let p = Params::from(1, 4, 6, 1, 6, 1, 19).unwrap();
        let mat = scmat(1, 4);
        let mut b = Batch::default();
        let (mut t_band, mut t_hull, mut t_plan) = (0u128, 0u128, 0u128);
        let (mut hull_cells, mut nband, mut nhull) = (0u64, 0u64, 0u64);
        let mut sink = 0i64;
        for chunk in jobs.chunks(1000) {
            let kj: Vec<KswJob> = chunk
                .iter()
                .map(|(t, q)| KswJob {
                    query: q,
                    target: t,
                })
                .collect();
            let mut out = vec![
                KswAlignResult {
                    score: 0,
                    qb: -1,
                    qe: -1,
                    tb: -1,
                    te: -1,
                    score2: -1,
                    te2: -1
                };
                kj.len()
            ];
            b.clear();
            let mut hulls: Vec<crate::matesw::FwdJob> = Vec::new();
            let t0 = std::time::Instant::now();
            for (i, j) in kj.iter().enumerate() {
                if j.query.len() >= 250 {
                    continue;
                }
                let quanta = j.query.len().div_ceil(16) * 16;
                if let Some(Decision::Hull { hb, he }) =
                    b.plan(i, j.target, j.query, &p, 1000, quanta)
                {
                    hulls.push(crate::matesw::FwdJob {
                        query: j.query,
                        target: &j.target[hb..=he],
                        minsc: 19,
                        endsc: i32::MAX,
                    });
                }
            }
            t_plan += t0.elapsed().as_nanos();
            // The banded parents' hulls, for the comparison.
            let mut hull_of_banded: Vec<crate::matesw::FwdJob> = Vec::new();
            for pa in &b.parents {
                let j = &kj[pa.job as usize];
                hull_cells += (pa.he - pa.hb + 1) as u64 * j.query.len().div_ceil(16) as u64 * 16;
                hull_of_banded.push(crate::matesw::FwdJob {
                    query: j.query,
                    target: &j.target[pa.hb as usize..=pa.he as usize],
                    minsc: 19,
                    endsc: i32::MAX,
                });
            }
            nband += b.parents.len() as u64;
            nhull += hulls.len() as u64;
            let t0 = std::time::Instant::now();
            unsafe { b.run(&kj, &mat, 6, 1, 6, 1, 19, 1, &mut out) };
            t_band += t0.elapsed().as_nanos();
            sink += out.iter().map(|o| o.score as i64).sum::<i64>();
            let t0 = std::time::Instant::now();
            let r = crate::matesw::fwd_local_sw_batch(&hull_of_banded, 5, &mat, 6, 1, 6, 1, 1);
            t_hull += t0.elapsed().as_nanos();
            sink += r.iter().map(|o| o.0 as i64).sum::<i64>();
        }
        std::hint::black_box(sink);
        let cells = b.work.cells * b.work.lanes() as u64;
        eprintln!(
            "BAND bench: {nband} banded ({nhull} hull-only): plan {:.0} ns/job; bands {:.2} s for {cells} cells ({:.2} Gcell/s) vs their hulls {:.2} s for {hull_cells} cells ({:.2} Gcell/s)",
            t_plan as f64 / jobs.len() as f64,
            t_band as f64 / 1e9, cells as f64 / t_band as f64,
            t_hull as f64 / 1e9, hull_cells as f64 / t_hull as f64,
        );
    }
}
