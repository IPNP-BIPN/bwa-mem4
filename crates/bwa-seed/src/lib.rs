//! SMEM seeding via the FMD index, mirroring bwa-mem2's `getSMEMsOnePosOneThread` /
//! `getSMEMsAllPosOneThread` (`reference/bwa-mem2/src/FMI_search.cpp`) and the seed derivation of
//! `get_sa_entries`.
//!
//! Phase 3 implements round 1 (all-position SMEM collection, `min_intv = 1`) and turns SMEM
//! intervals into reference-coordinate seeds. Reseeding rounds 2/3 (`getSMEMsOnePos` re-seeding of
//! long/repetitive SMEMs and `bwtSeedStrategy`) are layered on later; the end-to-end byte-identity
//! gate for seeding is the SE SAM concordance in phase 6.

use bwa_core::MemOpt;
use bwa_index::{FmIndex, Smem};

pub mod lisa_seed;
pub use lisa_seed::collect_smems_lsa_zigzag;

/// A seed: an exact match between the read and the reference (bwa-mem2's `mem_seed_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemSeed {
    /// Reference begin, in the 2L forward++RC coordinate space.
    pub rbeg: i64,
    /// Query begin (0-based).
    pub qbeg: i32,
    /// Seed length.
    pub len: i32,
    /// Seed score (length, for exact matches).
    pub score: i32,
}

/// Collect all round-1 SMEMs of `codes` (2-bit encoded read, N = 4) with the given seed length and
/// minimum interval size. Mirrors `getSMEMsAllPosOneThread` (repeatedly calling the one-position
/// routine, advancing the start to `next_x`).

/// Lockstep width: how many independent FM walks are kept in flight, so each slot's `cp_occ`
/// prefetch has a full cycle to land before the block is used. **This is the aligner's
/// memory-level-parallelism knob**: N slots means at most N outstanding DRAM misses per core.
///
/// The default 16 was tuned on `work/region.fa`, whose BWT is cache-resident -- i.e. on an index
/// with no DRAM latency to hide, where extra slots are pure overhead and the knee necessarily lands
/// low. `BWA3_LOCKSTEP_N` re-sweeps it on a real index. Scheduling only: every slot walks its own
/// read deterministically, so N cannot change a result.
fn lockstep_width() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BWA3_LOCKSTEP_N").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(16)
    })
}

pub fn collect_smems(fm: &FmIndex, codes: &[u8], min_seed_len: i32, min_intv: i64) -> Vec<Smem> {
    let mut out = Vec::new();
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    let mut x = 0usize;
    while x < codes.len() {
        x = smems_from_pos(fm, codes, x, min_seed_len, min_intv, &mut scratch, &mut out);
    }
    out
}

/// Lockstep round-1 SMEM collection across a batch of reads (bwa-mem2's batched FM-index search,
/// Vasimuddin et al. IPDPS 2019; nh13's `getSMEMsOnePosOneThread_lockstep`). Each read is a slot
/// whose SMEM walk is a state machine (forward extension / backward search) stepped one FM operation
/// at a time; `N` slots advance round-robin. The per-step `cp_occ` prefetch (already in the walk)
/// then covers a full `N`-slot cycle before the block is used, hiding the DRAM latency of the
/// data-dependent checkpoint loads — the dominant cost of FM-index seeding on a genome-scale index.
///
/// Result-identical to calling [`collect_smems`] on each read: every slot walks its own read
/// deterministically and appends SMEMs in the same per-position order, so `out[r]` equals
/// `collect_smems(fm, reads[r], ..)`.
pub fn collect_smems_batched(
    fm: &FmIndex,
    reads: &[&[u8]],
    min_seed_len: i32,
    min_intv: i64,
) -> Vec<Vec<Smem>> {
    /// Lockstep width: independent walks kept in flight so each slot's prefetch has a full cycle to
    /// land before the block is used. 16 measured ~2.8% faster than 8 on a genome-scale index (M4 Max,
    /// SE); 24 ties it and 32 regresses, so 16 is the knee. Shared by all three batched seeding rounds.
    let n_lock = lockstep_width();

    let counts = fm.counts();
    let mut output: Vec<Vec<Smem>> = (0..reads.len()).map(|_| Vec::new()).collect();
    if reads.is_empty() {
        return output;
    }
    let max_len = reads.iter().map(|r| r.len()).max().unwrap_or(0);

    let mut slots: Vec<Option<LsSlot>> = Vec::with_capacity(n_lock);
    let mut next_read = 0usize;
    for _ in 0..n_lock {
        if next_read < reads.len() {
            slots.push(Some(LsSlot::new(next_read, 0, min_intv, false, max_len)));
            next_read += 1;
        } else {
            slots.push(None);
        }
    }

    let mut live = slots.iter().filter(|s| s.is_some()).count();
    while live > 0 {
        for slot_opt in slots.iter_mut() {
            let Some(slot) = slot_opt.as_mut() else {
                continue;
            };
            slot.step(fm, reads[slot.ridx], min_seed_len, &counts);
            if slot.phase == LsPhase::Done {
                output[slot.ridx] = std::mem::take(&mut slot.out);
                if next_read < reads.len() {
                    slot.reset_to(next_read, 0, min_intv, false);
                    next_read += 1;
                } else {
                    *slot_opt = None;
                    live -= 1;
                }
            }
        }
    }
    output
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LsPhase {
    Start,
    Fwd,
    BwdInit,
    Bwd,
    PosDone,
    Done,
}

/// One read's round-1 SMEM walk as a resumable state machine (see [`collect_smems_batched`]). Mirrors
/// [`smems_from_pos`] step by step: `Start` seeds the single-base interval at `x`, `Fwd` runs one
/// forward-extension iteration, `BwdInit` does the fwd->bwd housekeeping (append + reverse), `Bwd`
/// runs one backward-search outer iteration, `PosDone` emits the final SMEM and advances `x`.
struct LsSlot {
    ridx: usize,
    x: usize,
    /// Occurrence-count floor for this walk (`min_intv`). Round 1 shares one value across the batch;
    /// round-2 re-seeds each carry their own (`parent.s + 1`), so it lives per-slot.
    min_intv: i64,
    /// Round 2 re-seeds a *single* position and stops (`single_pos`); round 1 sweeps every position.
    single_pos: bool,
    phase: LsPhase,
    smem: Smem,
    num_prev: usize,
    j: usize,
    jj: i64,
    next_x: usize,
    prev: Vec<Smem>,
    out: Vec<Smem>,
}

impl LsSlot {
    fn new(ridx: usize, x: usize, min_intv: i64, single_pos: bool, max_len: usize) -> Self {
        LsSlot {
            ridx,
            x,
            min_intv,
            single_pos,
            phase: LsPhase::Start,
            smem: Smem::default(),
            num_prev: 0,
            j: 0,
            jj: -1,
            next_x: 0,
            prev: vec![Smem::default(); max_len + 2],
            out: Vec::new(),
        }
    }

    /// Re-point this slot at a new walk, reusing the `prev` buffer (sized to the batch max length).
    fn reset_to(&mut self, ridx: usize, x: usize, min_intv: i64, single_pos: bool) {
        self.ridx = ridx;
        self.x = x;
        self.min_intv = min_intv;
        self.single_pos = single_pos;
        self.phase = LsPhase::Start;
        self.num_prev = 0;
        self.out.clear();
    }

    fn step(&mut self, fm: &FmIndex, codes: &[u8], min_seed_len: i32, counts: &[i64; 5]) {
        let min_intv = self.min_intv;
        let readlength = codes.len();
        match self.phase {
            LsPhase::Start => {
                if self.x >= readlength {
                    self.phase = LsPhase::Done;
                    return;
                }
                self.next_x = self.x + 1;
                let a = codes[self.x];
                if a >= 4 {
                    // No SMEM at an ambiguous base; advance one position (smems_from_pos returns x+1).
                    // A round-2 re-seed handles exactly one position, so it stops here regardless.
                    self.x = self.next_x;
                    if self.single_pos || self.x >= readlength {
                        self.phase = LsPhase::Done;
                    }
                    return;
                }
                let a = a as usize;
                self.smem = Smem {
                    rid: 0,
                    m: self.x as u32,
                    n: self.x as u32,
                    k: counts[a],
                    l: counts[3 - a],
                    s: counts[a + 1] - counts[a],
                };
                self.num_prev = 0;
                self.j = self.x + 1;
                self.phase = LsPhase::Fwd;
            }
            LsPhase::Fwd => {
                if self.j >= readlength {
                    self.phase = LsPhase::BwdInit;
                    return;
                }
                let aj = codes[self.j];
                self.next_x = self.j + 1;
                if aj >= 4 {
                    self.phase = LsPhase::BwdInit;
                    return;
                }
                let mut fwd = self.smem;
                std::mem::swap(&mut fwd.k, &mut fwd.l);
                let ext = fm.backward_ext(fwd, 3 - aj as usize);
                let mut new_smem = ext;
                std::mem::swap(&mut new_smem.k, &mut new_smem.l);
                new_smem.n = self.j as u32;

                self.prev[self.num_prev] = self.smem;
                if new_smem.s != self.smem.s {
                    self.num_prev += 1;
                }
                if new_smem.s < min_intv {
                    self.next_x = self.j;
                    self.phase = LsPhase::BwdInit;
                    return;
                }
                self.smem = new_smem;
                // Next forward step swaps k/l, so its backward_ext reads the blocks at `new_smem.l`.
                fm.prefetch_occ(new_smem.l, new_smem.l + new_smem.s);
                self.j += 1;
            }
            LsPhase::BwdInit => {
                if self.smem.s >= min_intv {
                    self.prev[self.num_prev] = self.smem;
                    self.num_prev += 1;
                }
                self.prev[..self.num_prev].reverse();
                self.jj = self.x as i64 - 1;
                self.phase = LsPhase::Bwd;
            }
            LsPhase::Bwd => {
                if self.jj < 0 {
                    self.phase = LsPhase::PosDone;
                    return;
                }
                let a = codes[self.jj as usize];
                if a > 3 {
                    self.phase = LsPhase::PosDone;
                    return;
                }
                let a = a as usize;
                let mut num_curr = 0usize;
                let mut curr_s = -1i64;

                let mut p = 0usize;
                while p < self.num_prev {
                    let sm = self.prev[p];
                    let mut new_smem = fm.backward_ext(sm, a);
                    new_smem.m = self.jj as u32;
                    if new_smem.s < min_intv
                        && (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len)
                    {
                        self.out.push(sm);
                        break;
                    }
                    if new_smem.s >= min_intv && new_smem.s != curr_s {
                        curr_s = new_smem.s;
                        self.prev[num_curr] = new_smem;
                        num_curr += 1;
                        fm.prefetch_occ(new_smem.k, new_smem.k + new_smem.s);
                        break;
                    }
                    p += 1;
                }
                p += 1;
                while p < self.num_prev {
                    let sm = self.prev[p];
                    let mut new_smem = fm.backward_ext(sm, a);
                    new_smem.m = self.jj as u32;
                    if new_smem.s >= min_intv && new_smem.s != curr_s {
                        curr_s = new_smem.s;
                        self.prev[num_curr] = new_smem;
                        num_curr += 1;
                        fm.prefetch_occ(new_smem.k, new_smem.k + new_smem.s);
                    }
                    p += 1;
                }
                self.num_prev = num_curr;
                self.jj -= 1;
                if num_curr == 0 {
                    self.phase = LsPhase::PosDone;
                }
            }
            LsPhase::PosDone => {
                if self.num_prev != 0 {
                    let sm = self.prev[0];
                    if (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len) {
                        self.out.push(sm);
                    }
                }
                self.x = self.next_x;
                // Round-2 re-seeds a single position (matching one `smems_from_pos` call); round 1
                // sweeps the whole read.
                self.phase = if self.single_pos || self.x >= readlength {
                    LsPhase::Done
                } else {
                    LsPhase::Start
                };
            }
            LsPhase::Done => {}
        }
    }
}

/// One-position SMEM search starting at `x`, appending SMEMs to `out` and returning `next_x`.
/// Faithful port of `getSMEMsOnePosOneThread`'s inner body.
fn smems_from_pos(
    fm: &FmIndex,
    codes: &[u8],
    x: usize,
    min_seed_len: i32,
    min_intv: i64,
    prev: &mut [Smem],
    out: &mut Vec<Smem>,
) -> usize {
    let readlength = codes.len();
    let counts = fm.counts();
    let mut next_x = x + 1;
    let a = codes[x];
    if a >= 4 {
        return next_x;
    }

    // Initial single-base interval.
    let a = a as usize;
    let mut smem = Smem {
        rid: 0,
        m: x as u32,
        n: x as u32,
        k: counts[a],
        l: counts[3 - a],
        s: counts[a + 1] - counts[a],
    };
    let mut num_prev = 0usize;

    // Forward extension (backward extension on the RC via swapped k/l and complemented base).
    let mut j = x + 1;
    while j < readlength {
        let aj = codes[j];
        next_x = j + 1;
        if aj >= 4 {
            break;
        }
        let mut fwd = smem;
        std::mem::swap(&mut fwd.k, &mut fwd.l);
        let ext = fm.backward_ext(fwd, 3 - aj as usize);
        let mut new_smem = ext;
        std::mem::swap(&mut new_smem.k, &mut new_smem.l);
        new_smem.n = j as u32;

        prev[num_prev] = smem;
        if new_smem.s != smem.s {
            num_prev += 1;
        }
        if new_smem.s < min_intv {
            next_x = j;
            break;
        }
        smem = new_smem;
        j += 1;
    }
    if smem.s >= min_intv {
        prev[num_prev] = smem;
        num_prev += 1;
    }

    prev[..num_prev].reverse();

    // Backward extension.
    let mut jj = x as i64 - 1;
    while jj >= 0 {
        let a = codes[jj as usize];
        if a > 3 {
            break;
        }
        let a = a as usize;
        let mut num_curr = 0usize;
        let mut curr_s = -1i64;

        let mut p = 0usize;
        while p < num_prev {
            let sm = prev[p];
            let mut new_smem = fm.backward_ext(sm, a);
            new_smem.m = jj as u32;
            if new_smem.s < min_intv
                && (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len)
            {
                out.push(sm);
                break;
            }
            if new_smem.s >= min_intv && new_smem.s != curr_s {
                curr_s = new_smem.s;
                prev[num_curr] = new_smem;
                num_curr += 1;
                // Prefetch the checkpoint blocks the next SMEM step's backward_ext on this kept
                // interval will touch, one step ahead (bwa-mem2 / nh13 `ENABLE_PREFETCH`).
                fm.prefetch_occ(new_smem.k, new_smem.k + new_smem.s);
                break;
            }
            p += 1;
        }
        p += 1;
        while p < num_prev {
            let sm = prev[p];
            let mut new_smem = fm.backward_ext(sm, a);
            new_smem.m = jj as u32;
            if new_smem.s >= min_intv && new_smem.s != curr_s {
                curr_s = new_smem.s;
                prev[num_curr] = new_smem;
                num_curr += 1;
                fm.prefetch_occ(new_smem.k, new_smem.k + new_smem.s);
            }
            p += 1;
        }
        num_prev = num_curr;
        if num_curr == 0 {
            break;
        }
        jj -= 1;
    }
    if num_prev != 0 {
        let sm = prev[0];
        if (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len) {
            out.push(sm);
        }
    }
    next_x
}

/// Round-3 forward-only seeding (`bwtSeedStrategyAllPosOneThread`): emit a seed when the interval
/// first drops below `max_intv` and the seed is at least `min_seed_len` long.
fn bwt_seed_strategy(
    fm: &FmIndex,
    codes: &[u8],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut Vec<Smem>,
) {
    let counts = fm.counts();
    let readlength = codes.len();
    let mut x = 0usize;
    while x < readlength {
        let mut next_x = x + 1;
        if codes[x] < 4 {
            let a = codes[x] as usize;
            let mut smem = Smem {
                rid: 0,
                m: x as u32,
                n: x as u32,
                k: counts[a],
                l: counts[3 - a],
                s: counts[a + 1] - counts[a],
            };
            let mut j = x + 1;
            while j < readlength {
                next_x = j + 1;
                let aj = codes[j];
                if aj >= 4 {
                    break;
                }
                let mut fwd = smem;
                std::mem::swap(&mut fwd.k, &mut fwd.l);
                let ext = fm.backward_ext(fwd, 3 - aj as usize);
                let mut new_smem = ext;
                std::mem::swap(&mut new_smem.k, &mut new_smem.l);
                new_smem.n = j as u32;
                smem = new_smem;
                if smem.s < max_intv
                    && (i64::from(smem.n) - i64::from(smem.m) + 1) >= i64::from(min_seed_len)
                {
                    if smem.s > 0 {
                        out.push(smem);
                    }
                    break;
                }
                j += 1;
            }
        }
        x = next_x;
    }
}

#[derive(PartialEq, Eq)]
enum BwtPhase {
    Start,
    Fwd,
    Done,
}

/// One read's round-3 (`bwt_seed_strategy`) forward-seeding walk as a resumable state machine, so a
/// batch of reads can run their walks in lockstep (see [`bwt_seed_strategy_batched`]) and hide the
/// `cp_occ` DRAM latency of the forward extension. Mirrors [`bwt_seed_strategy`] exactly: `Start`
/// seeds the single-base interval at `x`, `Fwd` runs one forward-extension iteration.
struct BwtSeedSlot {
    ridx: usize,
    x: usize,
    next_x: usize,
    j: usize,
    smem: Smem,
    phase: BwtPhase,
    out: Vec<Smem>,
}

impl BwtSeedSlot {
    fn new(ridx: usize) -> Self {
        BwtSeedSlot {
            ridx,
            x: 0,
            next_x: 0,
            j: 0,
            smem: Smem::default(),
            phase: BwtPhase::Start,
            out: Vec::new(),
        }
    }

    fn reset(&mut self, ridx: usize) {
        self.ridx = ridx;
        self.x = 0;
        self.phase = BwtPhase::Start;
        self.out.clear();
    }

    /// End the current position: advance `x` to `next_x` and return to `Start` (or finish).
    #[inline]
    fn end_pos(&mut self, readlength: usize) {
        self.x = self.next_x;
        self.phase = if self.x >= readlength {
            BwtPhase::Done
        } else {
            BwtPhase::Start
        };
    }

    fn step(
        &mut self,
        fm: &FmIndex,
        codes: &[u8],
        max_intv: i64,
        min_seed_len: i32,
        counts: &[i64; 5],
    ) {
        let readlength = codes.len();
        match self.phase {
            BwtPhase::Start => {
                if self.x >= readlength {
                    self.phase = BwtPhase::Done;
                    return;
                }
                self.next_x = self.x + 1;
                if codes[self.x] >= 4 {
                    self.end_pos(readlength);
                    return;
                }
                let a = codes[self.x] as usize;
                self.smem = Smem {
                    rid: 0,
                    m: self.x as u32,
                    n: self.x as u32,
                    k: counts[a],
                    l: counts[3 - a],
                    s: counts[a + 1] - counts[a],
                };
                self.j = self.x + 1;
                self.phase = BwtPhase::Fwd;
            }
            BwtPhase::Fwd => {
                if self.j >= readlength {
                    self.end_pos(readlength);
                    return;
                }
                self.next_x = self.j + 1;
                let aj = codes[self.j];
                if aj >= 4 {
                    self.end_pos(readlength);
                    return;
                }
                let mut fwd = self.smem;
                std::mem::swap(&mut fwd.k, &mut fwd.l);
                let ext = fm.backward_ext(fwd, 3 - aj as usize);
                let mut new_smem = ext;
                std::mem::swap(&mut new_smem.k, &mut new_smem.l);
                new_smem.n = self.j as u32;
                self.smem = new_smem;
                if self.smem.s < max_intv
                    && (i64::from(self.smem.n) - i64::from(self.smem.m) + 1)
                        >= i64::from(min_seed_len)
                {
                    if self.smem.s > 0 {
                        self.out.push(self.smem);
                    }
                    self.end_pos(readlength);
                    return;
                }
                // Next forward step swaps k/l, so its backward_ext reads the blocks at `smem.l`.
                fm.prefetch_occ(self.smem.l, self.smem.l + self.smem.s);
                self.j += 1;
            }
            BwtPhase::Done => {}
        }
    }
}

/// Batched round-3 seeding: run every read's [`bwt_seed_strategy`] walk in lockstep (N in flight,
/// round-robin) so the forward-extension `cp_occ` loads of independent reads overlap. Appends each
/// read's round-3 SMEMs to `out[ridx]`, byte-identical to calling [`bwt_seed_strategy`] per read.
fn bwt_seed_strategy_batched(
    fm: &FmIndex,
    reads: &[&[u8]],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut [Vec<Smem>],
) {
    let n_lock = lockstep_width();
    if reads.is_empty() {
        return;
    }
    let counts = fm.counts();
    let mut slots: Vec<Option<BwtSeedSlot>> = Vec::with_capacity(n_lock);
    let mut next_read = 0usize;
    for _ in 0..n_lock {
        if next_read < reads.len() {
            slots.push(Some(BwtSeedSlot::new(next_read)));
            next_read += 1;
        } else {
            slots.push(None);
        }
    }
    let mut live = slots.iter().filter(|s| s.is_some()).count();
    while live > 0 {
        for slot_opt in slots.iter_mut() {
            let Some(slot) = slot_opt.as_mut() else {
                continue;
            };
            slot.step(fm, reads[slot.ridx], max_intv, min_seed_len, &counts);
            if slot.phase == BwtPhase::Done {
                out[slot.ridx].append(&mut slot.out);
                if next_read < reads.len() {
                    slot.reset(next_read);
                    next_read += 1;
                } else {
                    *slot_opt = None;
                    live -= 1;
                }
            }
        }
    }
}

/// Collect SMEMs across bwa-mem2's three rounds (`mem_collect_smem`): round-1 all-position SMEMs,
/// round-2 re-seeding of long non-repetitive SMEMs from their midpoint, and round-3 interval-capped
/// forward seeding. This is the full seed set feeding chaining.
pub fn mem_collect_smem(fm: &FmIndex, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    // Round 1, then rounds 2/3 which depend on it.
    let mut smems = collect_smems(fm, codes, opt.min_seed_len, 1);
    smem_rounds_2_3(fm, codes, opt, &mut smems);
    smems
}

/// Batched seeding for a read batch: round-1 SMEMs are collected in **lockstep**
/// ([`collect_smems_batched`], hiding FM-index latency), then rounds 2/3 (re-seeding + interval-capped
/// forward seeding) run per read. Returns, for every read, exactly what [`mem_collect_smem`] would.
pub fn mem_collect_smem_batched(fm: &FmIndex, reads: &[&[u8]], opt: &MemOpt) -> Vec<Vec<Smem>> {
    let mut per_read = collect_smems_batched(fm, reads, opt.min_seed_len, 1);
    // Rounds 2 and 3 both run in lockstep across the batch so their data-dependent `cp_occ` loads
    // overlap — the same latency-hiding trick as round 1. Order per read is preserved (round 1,
    // then round 2, then round 3), so this is identical to the per-read path.
    // `BWA3_SEED_R2_SERIAL` forces the old per-read round 2, for regression/parity verification.
    if std::env::var_os("BWA3_SEED_R2_SERIAL").is_some() {
        for (r, codes) in reads.iter().enumerate() {
            smem_round_2(fm, codes, opt, &mut per_read[r]);
        }
    } else {
        smem_round_2_batched(fm, reads, opt, &mut per_read);
    }
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_batched(fm, reads, opt.max_mem_intv, opt.min_seed_len + 1, &mut per_read);
    }
    per_read
}

/// Hybrid seeding: **round 1 via the learned index** (`collect_smems_lsa_zigzag` — BWA-MEME's fast
/// zigzag, ~2x the FM-index round-1 at genome scale because it jumps straight to each LEM instead of
/// walking `cp_occ` per base), then **rounds 2/3 via the FM index** (the batched reseeding, which
/// needs no ISA). Byte-identical to [`mem_collect_smem_batched`]: the LISA round-1 SMEM *set* is proven
/// identical to the FM round-1 (`LearnedSa::bi_interval` == `FmIndex::backward_ext`), and chaining
/// re-sorts SMEMs by `(m, n)`, so round-1 *order* is irrelevant. Selected only when RAM fits the
/// learned index (see `bwa_core::sysram`); otherwise the caller uses [`mem_collect_smem_batched`].
pub fn mem_collect_smem_hybrid(
    fm: &FmIndex,
    lsa: &bwa_index::lisa::LearnedSa,
    reads: &[&[u8]],
    opt: &MemOpt,
) -> Vec<Vec<Smem>> {
    let mut per_read: Vec<Vec<Smem>> = reads
        .iter()
        .map(|codes| collect_smems_lsa_zigzag(lsa, codes, opt.min_seed_len))
        .collect();
    smem_round_2_batched(fm, reads, opt, &mut per_read);
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_batched(fm, reads, opt.max_mem_intv, opt.min_seed_len + 1, &mut per_read);
    }
    per_read
}

/// Rounds 2 (midpoint re-seeding of long non-repetitive SMEMs) and 3 (interval-capped forward
/// seeding), appended to the round-1 `smems` in place. Shared by the per-read and batched entry
/// points so they stay identical.
fn smem_rounds_2_3(fm: &FmIndex, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    smem_round_2(fm, codes, opt, smems);
    // Round 3.
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy(fm, codes, opt.max_mem_intv, opt.min_seed_len + 1, smems);
    }
}

/// Round 2: re-seed each long, non-repetitive round-1 SMEM from its midpoint (appends in place).
fn smem_round_2(fm: &FmIndex, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
    let num1 = smems.len();
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    for idx in 0..num1 {
        let p = smems[idx];
        let start = p.m as i32;
        let end = p.n as i32 + 1;
        if end - start < split_len || p.s > i64::from(opt.split_width) {
            continue;
        }
        let x = ((end + start) >> 1) as usize;
        smems_from_pos(fm, codes, x, opt.min_seed_len, p.s + 1, &mut scratch, smems);
    }
}

/// A single round-2 re-seed: re-run the SMEM walk of read `ridx` from position `x` with occurrence
/// floor `min_intv` (the parent SMEM's `s + 1`). Enumerated up front from the round-1 output so the
/// walks across the whole batch can run in lockstep.
struct ReseedJob {
    ridx: usize,
    x: usize,
    min_intv: i64,
}

/// Batched round 2: the per-read [`smem_round_2`] runs each midpoint re-seed as an isolated,
/// latency-exposed [`smems_from_pos`] walk — measured as ~half of all seeding time on a genome-scale
/// index, because unlike rounds 1 and 3 it is not lockstepped, so every data-dependent `cp_occ` load
/// stalls on DRAM. This enumerates all re-seed jobs across the batch (in `(read, round-1 SMEM index)`
/// order) and advances `N` of them round-robin, so each slot's prefetch covers a full cycle before
/// its block is used. Each job's SMEMs are appended to its read in job order, identical to running
/// [`smem_round_2`] per read.
fn smem_round_2_batched(fm: &FmIndex, reads: &[&[u8]], opt: &MemOpt, per_read: &mut [Vec<Smem>]) {
    let n_lock = lockstep_width();
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;

    // Enumerate the re-seed jobs, preserving per-read append order (round-1 SMEM index ascending).
    let mut jobs: Vec<ReseedJob> = Vec::new();
    for (ridx, smems) in per_read.iter().enumerate() {
        for p in smems.iter() {
            let start = p.m as i32;
            let end = p.n as i32 + 1;
            if end - start < split_len || p.s > i64::from(opt.split_width) {
                continue;
            }
            jobs.push(ReseedJob {
                ridx,
                x: ((end + start) >> 1) as usize,
                min_intv: p.s + 1,
            });
        }
    }
    if jobs.is_empty() {
        return;
    }

    let counts = fm.counts();
    let max_len = reads.iter().map(|r| r.len()).max().unwrap_or(0);
    // Each job's output, filled as it completes; appended in job order at the end.
    let mut results: Vec<Vec<Smem>> = (0..jobs.len()).map(|_| Vec::new()).collect();

    let mut slots: Vec<Option<(usize, LsSlot)>> = Vec::with_capacity(n_lock);
    let mut next_job = 0usize;
    for _ in 0..n_lock {
        if next_job < jobs.len() {
            let j = &jobs[next_job];
            slots.push(Some((
                next_job,
                LsSlot::new(j.ridx, j.x, j.min_intv, true, max_len),
            )));
            next_job += 1;
        } else {
            slots.push(None);
        }
    }

    let mut live = slots.iter().filter(|s| s.is_some()).count();
    while live > 0 {
        for slot_opt in slots.iter_mut() {
            let Some((job_id, slot)) = slot_opt.as_mut() else {
                continue;
            };
            slot.step(fm, reads[slot.ridx], opt.min_seed_len, &counts);
            if slot.phase == LsPhase::Done {
                results[*job_id] = std::mem::take(&mut slot.out);
                if next_job < jobs.len() {
                    let j = &jobs[next_job];
                    *job_id = next_job;
                    slot.reset_to(j.ridx, j.x, j.min_intv, true);
                    next_job += 1;
                } else {
                    *slot_opt = None;
                    live -= 1;
                }
            }
        }
    }

    // Append each job's SMEMs to its read, in job order (= per-read round-1-index order).
    for (job, res) in jobs.iter().zip(results) {
        per_read[job.ridx].extend(res);
    }
}

/// Turn one SMEM into reference-coordinate seeds, sampling up to `max_occ` occurrences
/// (bwa-mem2's `get_sa_entries` stride sampling).
pub fn seeds_from_smem(fm: &FmIndex, smem: &Smem, max_occ: i32) -> Vec<MemSeed> {
    let len = (i64::from(smem.n) - i64::from(smem.m) + 1) as i32;
    let max_occ = i64::from(max_occ);
    let step = if smem.s > max_occ {
        smem.s / max_occ
    } else {
        1
    };
    let mut seeds = Vec::new();
    let mut c = 0i64;
    let mut j = smem.k;
    while j < smem.k + smem.s && c < max_occ {
        seeds.push(MemSeed {
            rbeg: fm.get_sa(j),
            qbeg: smem.m as i32,
            len,
            score: len,
        });
        j += step;
        c += 1;
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn tiny() -> FmIndex {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        FmIndex::load(Path::new(prefix)).unwrap()
    }

    /// Hybrid seeding (LISA round-1 + FM rounds 2/3) must produce, per read, the same SMEM SET as the
    /// full FM seeding — after chaining's `(m, n)` sort, the set is what determines the alignment, so
    /// this is the seeding-level byte-identity gate for the hybrid path.
    #[test]
    fn hybrid_seeding_equals_fm() {
        let fm = tiny();
        let lsa = bwa_index::lisa::LearnedSa::build(fm.reference().to_vec(), 4096);
        let opt = MemOpt::default();
        let l_pac = fm.l_pac() as usize;
        let mut state = 0xdead_beef_0042u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..200 {
            let len = 40 + (next() as usize % 120);
            let start = next() as usize % (l_pac - len);
            let mut r: Vec<u8> = (0..len).map(|i| fm.base((start + i) as i64)).collect();
            for _ in 0..(next() % 4) {
                let p = next() as usize % len;
                r[p] = (next() % 4) as u8;
            }
            reads.push(r);
        }
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let fm_all = mem_collect_smem_batched(&fm, &refs, &opt);
        let hy_all = mem_collect_smem_hybrid(&fm, &lsa, &refs, &opt);
        let key = |v: &[Smem]| -> Vec<(u32, u32, i64, i64)> {
            let mut k: Vec<_> = v.iter().map(|s| (s.m, s.n, s.k, s.s)).collect();
            k.sort_unstable();
            k
        };
        for (r, (f, h)) in fm_all.iter().zip(hy_all.iter()).enumerate() {
            assert_eq!(key(f), key(h), "read {r} (len {})", reads[r].len());
        }
    }

    /// Occurrences of `pat` in the binary reference (both strands, since .0123 is fwd++RC).
    fn naive_occ(reference: &[u8], pat: &[u8]) -> i64 {
        reference.windows(pat.len()).filter(|w| *w == pat).count() as i64
    }

    #[test]
    fn smems_are_exact_matches() {
        let fm = tiny();
        // Read = a 120bp forward slice of the reference; must seed back to its origin.
        let start = 50_000i64;
        let len = 120usize;
        let read: Vec<u8> = (0..len).map(|i| fm.base(start + i as i64)).collect();

        let smems = collect_smems(&fm, &read, 19, 1);
        assert!(!smems.is_empty(), "no SMEMs found");

        // Each SMEM's read substring occurs exactly `s` times in the reference.
        for sm in &smems {
            let sub = &read[sm.m as usize..=sm.n as usize];
            assert_eq!(
                sm.s,
                naive_occ(fm.reference(), sub),
                "SMEM interval size wrong"
            );
            assert!((sm.n - sm.m + 1) as i32 >= 19);
        }

        // The full-length SMEM must exist and seed to the origin position.
        let full = smems.iter().find(|s| s.m == 0 && s.n as usize == len - 1);
        let full = full.expect("no full-length SMEM covering the read");
        let seeds = seeds_from_smem(&fm, full, 500);
        assert!(
            seeds
                .iter()
                .any(|s| s.rbeg == start && s.qbeg == 0 && s.len == len as i32),
            "no seed mapping back to the origin at {start}"
        );
    }

    #[test]
    fn batched_smems_equal_per_read() {
        let fm = tiny();
        // A batch of varied reads: exact reference slices (deep SMEM walks), random reads (shallow),
        // reads with N bases, and short/empty — enough to exercise every state-machine transition.
        let mut state = 0xBA7C_4EED_0000_0001u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let reflen = fm.reference().len() as i64;
        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..60 {
            let kind = next() % 4;
            let len = 1 + (next() % 160) as usize;
            let mut r: Vec<u8> = match kind {
                0 => {
                    // exact reference slice
                    let start = (next() as i64) % (reflen - len as i64).max(1);
                    (0..len).map(|i| fm.base(start + i as i64)).collect()
                }
                _ => (0..len).map(|_| (next() % 4) as u8).collect(),
            };
            if kind == 3 && !r.is_empty() {
                let p = (next() as usize) % r.len();
                r[p] = 4; // inject an N
            }
            reads.push(r);
        }
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();

        for &(msl, mi) in &[(19i32, 1i64), (17, 1), (19, 2), (11, 1)] {
            let batched = collect_smems_batched(&fm, &refs, msl, mi);
            for (r, read) in reads.iter().enumerate() {
                let per_read = collect_smems(&fm, read, msl, mi);
                assert_eq!(
                    batched[r],
                    per_read,
                    "batched != per-read at read {r} (len {}, msl={msl}, mi={mi})",
                    read.len()
                );
            }
        }
    }

    #[test]
    fn batched_full_seeding_equals_per_read() {
        // Full 3-round seeding (mem_collect_smem_batched, incl. the lockstep round 2) must match the
        // per-read mem_collect_smem for every read. Favor long exact slices so round-2 midpoint
        // re-seeding actually fires (SMEMs >= split_len), plus mutated slices and randoms.
        let fm = tiny();
        let mut state = 0x51A7_ED00_C0FF_EE01u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let reflen = fm.reference().len() as i64;
        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..80 {
            let kind = next() % 5;
            let len = 40 + (next() % 150) as usize;
            let mut r: Vec<u8> = match kind {
                0 => (0..len).map(|_| (next() % 4) as u8).collect(), // random (shallow)
                _ => {
                    // exact reference slice (deep SMEM -> triggers round 2)
                    let start = (next() as i64) % (reflen - len as i64).max(1);
                    (0..len).map(|i| fm.base(start + i as i64)).collect()
                }
            };
            // Mutate a couple of bases in some slices (splits SMEMs, varied reseed geometry).
            if kind >= 3 && r.len() > 4 {
                for _ in 0..2 {
                    let p = (next() as usize) % r.len();
                    r[p] = (next() % 4) as u8;
                }
            }
            reads.push(r);
        }
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();

        let opt = MemOpt::default();
        let batched = mem_collect_smem_batched(&fm, &refs, &opt);
        for (r, read) in reads.iter().enumerate() {
            let per_read = mem_collect_smem(&fm, read, &opt);
            assert_eq!(
                batched[r], per_read,
                "batched full != per-read at read {r} (len {})",
                read.len()
            );
        }
    }

    #[test]
    fn smems_cover_repeated_region() {
        let fm = tiny();
        // A short read is still collected as an SMEM if >= min_seed_len.
        let start = 123_456i64.min(fm.l_pac() - 60);
        let read: Vec<u8> = (0..60).map(|i| fm.base(start + i as i64)).collect();
        let smems = collect_smems(&fm, &read, 19, 1);
        for sm in &smems {
            let sub = &read[sm.m as usize..=sm.n as usize];
            assert_eq!(sm.s, naive_occ(fm.reference(), sub));
        }
    }
}
