//! A hook through which another crate runs the mate-rescue band kernel on a device (`bwa-metal`,
//! the Apple GPU). This crate does not depend on any device crate: the binary registers an
//! implementation once with [`set_band_offload`], and the forward band pass of
//! `rescue_band::Batch::run` gives it a share of each call's bands. Nothing changes when none is
//! registered.
//!
//! A band lane's outputs are defined by [`band_reference`], the NEON kernel's cell for one lane
//! with no group widening: a device must reproduce it field for field. The parent results are
//! independent of which lanes ran where, because merging a lane into its parent is a max (row
//! maxima) and a lexicographic max (`(score, -te, -qe)`), both order-free.

use std::sync::OnceLock;

/// One band of a pruned rescue job, in band coordinates: target rows `t` (the band's rows of the
/// window), diagonal index `k` in `0..w`, query column `j = r + o - k`. Columns `len(q)..quanta`
/// are ksw's zero-scoring pad; columns outside `0..quanta` do not exist.
#[derive(Clone, Copy, Debug)]
pub struct BandTask<'a> {
    pub q: &'a [u8],
    pub t: &'a [u8],
    pub quanta: i32,
    pub o: i32,
    pub w: i32,
    /// `qe` is reported only when the lane's best reaches this, else 0.
    pub thr: u8,
}

/// A band lane's best score, its first row (band-relative, -1 when the best is 0) and the first
/// query column reaching it on that row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BandOut {
    pub gmax: u8,
    pub te: i32,
    pub qe: u8,
}

/// The uniform scoring the device kernels take: `a` on a match, `b` (signed) on a mismatch, `n`
/// on any cell with an N, and the gap costs as open-plus-extend and extend.
#[derive(Clone, Copy, Debug)]
pub struct BandScoring {
    pub a: i32,
    pub b: i32,
    pub n: i32,
    pub oe_del: i32,
    pub e_del: i32,
    pub oe_ins: i32,
    pub e_ins: i32,
}

/// Bands in flight on a device.
pub trait BandPending {
    fn is_done(&self) -> bool;
    /// Every task's [`BandOut`] and its row maxima, concatenated in task order (`t.len()` bytes
    /// each); `None` when the device failed, and the caller then runs the bands itself.
    fn wait(self: Box<Self>) -> Option<(Vec<BandOut>, Vec<u8>)>;
}

/// A device that runs band lanes.
pub trait BandOffload: Sync {
    /// Whether the device takes this band (width, scores within its kernel's bounds).
    fn takes(&self, t: &BandTask, sc: &BandScoring) -> bool;
    /// Fraction of a call's band cells to hand over.
    fn band_share(&self) -> f64;
    /// Start `tasks`; `None` when the device cannot take them now.
    fn submit<'s>(
        &'s self,
        tasks: &[BandTask],
        sc: &BandScoring,
    ) -> Option<Box<dyn BandPending + 's>>;
}

static HOOK: OnceLock<&'static dyn BandOffload> = OnceLock::new();

/// Register the band device for the process; false when one already is.
pub fn set_band_offload(h: &'static dyn BandOffload) -> bool {
    HOOK.set(h).is_ok()
}

pub(crate) fn band_offload() -> Option<&'static dyn BandOffload> {
    HOOK.get().copied()
}

/// One band lane, scalar: the NEON band kernel's cell (gaps open from `G = max(diag + s, E)`,
/// saturating at 0) on the lane's own band. Cells outside `0..quanta` hold 0 or lie downstream of
/// every live cell, and are not counted. Scores must stay below 256, as the u8 kernel's do.
pub fn band_reference(task: &BandTask, sc: &BandScoring) -> (BandOut, Vec<u8>) {
    let w = task.w.max(0) as usize;
    let nr = task.t.len();
    let (mut hp, mut ep) = (vec![0i32; w], vec![0i32; w]);
    let (mut hc, mut ec) = (vec![0i32; w], vec![0i32; w]);
    let mut rows = vec![0u8; nr];
    let (mut g, mut te, mut qe) = (0i32, -1i32, 0i32);
    let len2 = task.q.len() as i32;
    for r in 0..nr {
        let tb = task.t[r];
        let mut f = 0i32;
        let (mut rmax, mut jbest) = (0i32, 0i32);
        for k in (0..w).rev() {
            let j = r as i32 + task.o - k as i32;
            let s = if (0..len2).contains(&j) {
                let qb = task.q[j as usize];
                if qb >= 4 || tb >= 4 {
                    sc.n
                } else if qb == tb {
                    sc.a
                } else {
                    sc.b
                }
            } else {
                0
            };
            let ein = if k > 0 { ep[k - 1] } else { 0 };
            let h0 = (hp[k] + s).max(0).max(ein);
            let h = h0.max(f);
            ec[k] = (h0 - sc.oe_del).max(ein - sc.e_del).max(0);
            f = (h0 - sc.oe_ins).max(f - sc.e_ins).max(0);
            hc[k] = h;
            // k descending is j ascending: strict `>` keeps the first column.
            if (0..task.quanta).contains(&j) && h > rmax {
                (rmax, jbest) = (h, j);
            }
        }
        rows[r] = rmax as u8;
        if rmax > g {
            (g, te, qe) = (rmax, r as i32, jbest);
        }
        std::mem::swap(&mut hp, &mut hc);
        std::mem::swap(&mut ep, &mut ec);
    }
    let out = BandOut {
        gmax: g as u8,
        te: if g > 0 { te } else { -1 },
        qe: if g >= i32::from(task.thr) {
            qe as u8
        } else {
            0
        },
    };
    (out, rows)
}

/// Counters for `BWA4_MATESW_TIME`: band cells sent to the device, used from it, taken back by
/// the CPU, and the host time spent submitting and waiting.
pub mod probe {
    use std::sync::atomic::AtomicU64;
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub static SENT: AtomicU64 = AtomicU64::new(0);
    pub static DELIVERED: AtomicU64 = AtomicU64::new(0);
    pub static STOLEN: AtomicU64 = AtomicU64::new(0);
    pub static SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
    pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);
    /// Band cells by width: <=64, <=128, <=256, <=512, <=1024, wider.
    pub static BY_W: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
    /// Bands by width, same classes.
    pub static N_BY_W: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
}
