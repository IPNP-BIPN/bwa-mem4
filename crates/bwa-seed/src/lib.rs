//! SMEM seeding via the FMD index, mirroring bwa-mem2's `getSMEMsOnePosOneThread` /
//! `getSMEMsAllPosOneThread` (`reference/bwa-mem2/src/FMI_search.cpp`) and the seed derivation of
//! `get_sa_entries`.
//!
//! Phase 3 implements round 1 (all-position SMEM collection, `min_intv = 1`) and turns SMEM
//! intervals into reference-coordinate seeds. Reseeding rounds 2/3 (`getSMEMsOnePos` re-seeding of
//! long/repetitive SMEMs and `bwtSeedStrategy`) are layered on later; the end-to-end byte-identity
//! gate for seeding is the SE SAM concordance in phase 6.
//!
//! # Background, for a reader who has not met an FM index
//!
//! **What seeding is for.** Aligning a 150bp read against a 3Gbp genome by dynamic programming is
//! hopeless directly. So the aligner first finds *exact* substring matches (seeds), which pin the
//! read to a handful of candidate loci, and only then runs Smith-Waterman around those loci. This
//! crate produces the seeds; `bwa-chain` groups them; `bwa-extend` does the DP.
//!
//! **What an SMEM is.** A MEM (maximal exact match) is a read substring that occurs somewhere in
//! the reference and cannot be extended left or right without losing all its occurrences. A
//! *super*-maximal exact match (SMEM) is a MEM that is not contained inside another MEM of the same
//! read. Intuitively: for each read position, the longest exact match covering it. Reporting only
//! SMEMs keeps the seed set small without losing the loci a longer match would have found.
//!
//! **What the index is.** An FM index over the reference stores, for the *sorted list of all
//! suffixes* of the text, enough information to answer "which rows of that sorted list start with
//! pattern P?" in O(|P|) steps. The answer is always a contiguous row range, because the list is
//! sorted. bwa's text is `[forward genome][reverse complement of the genome]` concatenated (the "2L"
//! space, `l_pac` bases forward then `l_pac` bases RC), which is what lets one index serve both
//! strands: a hit at `rbeg >= l_pac` is a reverse-strand hit.
//!
//! **The interval `(k, l, s)`** (bwa's `SMEM` struct, `FMI_search.h`) is that row range, kept
//! bidirectionally:
//! - `k`: first row of the range in the *forward* index. Units: BWT row index, `0 ..= ref_seq_len`.
//! - `l`: first row of the *same pattern reverse-complemented*, in the same index. Only needed so
//!   that forward extension can be expressed as backward extension (see below); nothing downstream
//!   reads it.
//! - `s`: the range's *size*, i.e. how many times the pattern occurs in the 2L text. This is the
//!   "occurrence count". `s == 0` means the pattern does not occur.
//! - `m`, `n`: the pattern's span in the read, inclusive on both ends. Length is `n - m + 1`.
//!
//! **Backward search / `backward_ext`.** Given the interval of pattern `P`, the interval of `cP`
//! (one base `c` *prepended*) is computed from two occurrence counts: how many `c`s occur in the
//! BWT before row `k`, and before row `k + s`. That is `FmIndex::backward_ext` /
//! `FMI_search::backwardExt` (`FMI_search.cpp:1025`). Those two loads are data-dependent, land
//! anywhere in a multi-GB array, and are therefore the dominant cost of seeding: hence the prefetch
//! and lockstep machinery below.
//!
//! **Forward extension is backward extension on the complement.** There is no separate forward
//! index. To append base `c` on the *right*, swap `k` and `l` (which reinterprets the interval as
//! that of the reverse complement) and prepend the complement base `3 - c`, then swap back. That is
//! exactly the `mem::swap(&mut fwd.k, &mut fwd.l)` / `3 - aj` dance repeated throughout this file,
//! and it mirrors `FMI_search.cpp:546-556` comment "Forward extension is backward extension with
//! the BWT of reverse complement". Base codes are 2-bit `A=0 C=1 G=2 T=3`, so `3 - c` is the
//! complement; `4` means `N` and stops every walk.
//!
//! **The three seeding rounds** (`mem_collect_smem`, `bwamem.cpp:626`):
//! 1. All-position SMEMs with `min_intv = 1` (accept any pattern that occurs at all).
//! 2. Re-seed: every round-1 SMEM that is long (`>= split_len`) but not repetitive
//!    (`s <= split_width`) is searched again from its *midpoint*, demanding strictly more
//!    occurrences than the parent (`min_intv = parent.s + 1`). This recovers seeds that a single
//!    long SMEM swallowed.
//! 3. Forward-only seeding capped by `max_mem_intv`: walk right from each position and emit as soon
//!    as the occurrence count drops below the cap. Rescues short reads / repetitive regions.
//!
//! **Why the lockstep design.** All of the above is pointer chasing with no instruction-level
//! parallelism inside one read: step `i + 1` needs step `i`'s result. The fix is *memory*-level
//! parallelism across reads: keep `N = 16` independent walks in flight, advance them round-robin,
//! and prefetch each walk's next checkpoint block one step ahead. By the time a slot is revisited,
//! its line has landed. A consequence recorded in this project: several published "fewer memory
//! accesses" FM-index optimizations measure as no gain here, because the latency they remove is
//! already hidden.
//!
//! # Glossary: the names deliberately kept identical to the C
//!
//! These are NOT renamed, and must not be. Diffing this file line by line against
//! `FMI_search.cpp` is the workflow that has found every parity bug in this project so far, and a
//! rename breaks that diff. What each one means in plain language:
//!
//! | name | C origin | plain-language meaning |
//! |---|---|---|
//! | `k` | `SMEM.k` | First row of the FM interval: a suffix-array row number, i.e. the rank of one suffix in the sorted list of ALL suffixes of the reference. Every occurrence of the pattern is one row in `[k, k + s)`, and they are contiguous only because the list is sorted. |
//! | `l` | `SMEM.l` | The same interval for the pattern reverse-complemented. Bookkeeping only: it exists so that "append a base on the right" can be expressed as "prepend a base on the left of the reverse complement". Nothing downstream reads it. |
//! | `s` | `SMEM.s` | The interval's SIZE, i.e. the occurrence count of the pattern in the doubled reference. `s == 0` means "does not occur". `s` never grows as the pattern gets longer. |
//! | `m`, `n` | `SMEM.m/.n` | The match's span in the READ, inclusive at both ends. Length is `n - m + 1`, which is why every length test here carries a `+ 1`. |
//! | `x` | `query_pos_array[i]` | The read position the current SMEM search must cover. The outer sweep advances it to `next_x`, which is NOT always `x + 1`. |
//! | `j` | `j` (forward loop) | Cursor of the forward phase: the read position being appended on the RIGHT. |
//! | `jj` | `j` (backward loop) | Cursor of the backward phase: the read position being prepended on the LEFT. Signed, because it walks down past 0 to `-1`. We spell it `jj` only because `j` is already taken by the forward cursor in the same scope; the C reuses one variable. |
//! | `p` | `p` | Index into the candidate list `prev` during the backward phase. |
//! | `a`, `aj` | `a` | The 2-bit base code (`A=0 C=1 G=2 T=3`, `4` = `N`) at the position being extended. The C reuses one `a`; we split it so the forward cursor's base is `aj` ("the base at `j`"). |
//! | `prev`, `num_prev` | `prevArray`, `numPrev` | The live candidate intervals of the current position, and how many are live. Compacted in place, never reallocated. |
//! | `curr_s`, `num_curr` | `curr_s`, `numCurr` | The last occurrence count kept this backward round, and how many survivors have been kept. |
//! | `occ` | `GET_OCC` | "How many times does base `c` appear in the BWT before row `p`". This single count is what makes each extension step O(1), and its data-dependent memory load is the cost this whole file is organized around. |
//! | `rbeg`, `qbeg` | `mem_seed_t` | Reference begin and query (read) begin of a seed. |
//! | `l_pac` | `bns->l_pac` | Length of the FORWARD reference in bases. The searched text is `2 * l_pac` long: the forward genome followed by its reverse complement. So `rbeg < l_pac` is a forward-strand hit and `rbeg >= l_pac` is a reverse-strand hit, and strand costs nothing to determine. |
//!
//! # Reading order for this file
//!
//! 1. [`smems_from_pos`]: the algorithm itself, one position, straight-line and readable.
//! 2. [`collect_smems`]: the outer sweep that calls it for every read position (round 1).
//! 3. [`smem_round_2`] and [`bwt_seed_strategy`]: rounds 2 and 3.
//! 4. [`seeds_from_smem`]: turning an interval into reference coordinates.
//! 5. Only then the lockstep machinery ([`LsSlot`], [`collect_smems_batched`],
//!    [`smem_round_2_batched`], [`BwtSeedSlot`], [`bwt_seed_strategy_batched`]), which is those
//!    same three algorithms re-expressed as resumable state machines. It adds no logic; if the
//!    state machine and the straight-line version ever disagree, the straight-line one is right.

use bwa_core::MemOpt;
use bwa_index::{FmIndex, Smem};

/// Round-1 seeding through BWA-MEME's learned suffix array instead of the FM index. Kept a separate
/// module because it needs no BWT walk at all: it predicts the interval's location and corrects.
pub mod lisa_seed;
pub use lisa_seed::collect_smems_lsa_zigzag;

/// A seed: one *occurrence* of one SMEM, resolved to a reference coordinate (bwa-mem2's
/// `mem_seed_t`). An SMEM with `s` occurrences yields up to `max_occ` of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemSeed {
    /// Reference begin, in the 2L forward++RC coordinate space: `0 ..< l_pac` is the forward strand,
    /// `l_pac ..< 2*l_pac` the reverse complement. Obtained from the suffix array (`fm.get_sa`).
    pub rbeg: i64,
    /// Query begin (0-based), i.e. the SMEM's `m`.
    pub qbeg: i32,
    /// Seed length, i.e. the SMEM's `n - m + 1`. The match is exact over the whole length.
    pub len: i32,
    /// Seed score. For an exact match bwa sets `score = len` (`bwamem.cpp:912`,
    /// `s.score = s.len = slen`); the extension stage overwrites it later.
    pub score: i32,
}

/// Base code for `N` / any non-ACGT character, and the first code that terminates a walk. Base
/// codes are `A=0 C=1 G=2 T=3` (`nst_nt4_table`, `bntseq.cpp:40`), so anything `>= 4` is ambiguous
/// and no exact match may cover it.
///
/// The value is not a tunable: it is the encoding the FASTQ reader emits and the alphabet size the
/// index's `counts` array is built for (`counts` has 5 entries, `counts[4]` being the total). The
/// tests here compare against `>= AMBIGUOUS_BASE` and, in the backward loop, against a literal `> 3`
/// (which is the same test, kept literal because the C is literal there). Lowering this would make
/// real bases look ambiguous and silently truncate every seed; raising it would let `N` index past
/// `counts[4]`.
const AMBIGUOUS_BASE: u8 = 4;

/// The highest base code, and therefore the complement operator: `complement(c) == MAX_BASE - c`
/// for `c` in `0..=3` (A<->T, C<->G). This is the `3 - a` seen throughout the file.
///
/// Worked micro-example, base `G`: the 2-bit code is `2`, so `MAX_BASE - 2 == 1 == C`, and `G`'s
/// complement is indeed `C`. Likewise `A` (0) maps to `3` (`T`). The identity `3 - c` only holds
/// because the code order is `A C G T`, i.e. complementary bases sum to 3; it is the reason forward
/// extension can be expressed as a backward extension of the complement base on the k/l-swapped
/// interval. Changing this constant, or the code order, breaks every strand relationship in the file.
const MAX_BASE: usize = 3;

/// Default lockstep width when `BWA4_LOCKSTEP_N` is unset: how many independent FM walks are kept in
/// flight per lockstep driver, i.e. the ceiling on outstanding DRAM misses per core. Units: slots.
/// Any value `>= 1` is correct; 16 is the measured knee (8 is ~2.8% slower, 24 ties, 32 regresses).
/// Changing it changes only speed and peak scratch memory (each slot owns a `prev` buffer sized to
/// the batch's longest read), never the SMEMs produced. See [`lockstep_width`].
const DEFAULT_LOCKSTEP_WIDTH: usize = 16;

/// The C's rounding fudge in `split_len = (int)(min_seed_len * split_factor + .499)`
/// (`bwamem.cpp:630`). Truncation of `x + 0.499` is round-half-down; `0.5` would be round-half-up
/// and would change `split_len` on exact halves, so the digits are copied verbatim and the `f32`
/// arithmetic must stay `f32`.
const SPLIT_LEN_ROUNDING: f32 = 0.499;

/// Lockstep width: how many independent FM walks are kept in flight, so each slot's `cp_occ`
/// prefetch has a full cycle to land before the block is used. **This is the aligner's
/// memory-level-parallelism knob**: N slots means at most N outstanding DRAM misses per core.
///
/// The default 16 was tuned on `work/region.fa`, whose BWT is cache-resident -- i.e. on an index
/// with no DRAM latency to hide, where extra slots are pure overhead and the knee necessarily lands
/// low. `BWA4_LOCKSTEP_N` re-sweeps it on a real index. Scheduling only: every slot walks its own
/// read deterministically, so N cannot change a result.
///
/// # Returns
/// The number of lockstep slots, always `>= 1`. Read once from the environment and cached for the
/// process, so all three batched rounds agree and the value cannot change mid-run.
fn lockstep_width() -> usize {
    // Process-wide cache of the parsed width. `OnceLock` because the value must be identical for
    // every thread and every batch: a mid-run change would not corrupt results (scheduling only) but
    // would make benchmarks meaningless.
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    // Parse `BWA4_LOCKSTEP_N`; anything missing, unparseable, or `0` falls back to the default (a
    // width of 0 would mean no slots at all and the driver would never run).
    *N.get_or_init(|| {
        std::env::var("BWA4_LOCKSTEP_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_LOCKSTEP_WIDTH)
    })
}

/// Collect all round-1 SMEMs of one read. Mirrors `getSMEMsAllPosOneThread`
/// (`FMI_search.cpp:672`), which just calls the one-position routine in a loop, restarting at the
/// `next_x` that call reports.
///
/// # Parameters
/// - `fm`: the loaded FM index (supplies `counts`, `backward_ext`, `get_sa`). Borrowed read-only and
///   shared across threads; built by `bwa-index` from the reference FASTA.
/// - `codes`: the read, 2-bit encoded, `A=0 C=1 G=2 T=3`, `4` = `N`/any other character. Supplied by
///   `bwa-io`'s FASTQ reader. Not reverse-complemented: the index covers both strands. Any length
///   including 0 is accepted; indices into it are READ offsets, never reference positions.
/// - `min_seed_len`: minimum SMEM length to emit, in bases. `opt.min_seed_len`, default 19 (`-k`).
///   Shorter SMEMs are found but discarded. Must be `>= 1`.
/// - `min_intv`: minimum occurrence count `s` for an interval to stay alive, in occurrences. 1 in
///   round 1 (accept anything that occurs at all); `parent.s + 1` in round 2. Must be `>= 1`.
///
/// # Returns
/// The round-1 SMEMs of this read, in ascending order of the position `x` they were found from
/// (which is close to, but not exactly, sorted by `m`). Each carries its FM interval `(k, l, s)` and
/// its READ span `(m, n)`; no reference coordinate is resolved here.
///
/// Note the whole read shares one `scratch` buffer of `codes.len() + 2` entries: this is the `prev`
/// working set of `smems_from_pos`, whose invariant is documented there.
pub fn collect_smems(fm: &FmIndex, codes: &[u8], min_seed_len: i32, min_intv: i64) -> Vec<Smem> {
    let mut out = Vec::new();
    // Candidate list reused by every position of this read. Sized `codes.len() + 2` to satisfy the
    // capacity invariant of `smems_from_pos` (at most one candidate per read position plus the final
    // one, plus slack). Contents are garbage between calls; each call writes before it reads.
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    // Outer sweep cursor: the READ offset whose covering SMEMs are being collected next.
    let mut x = 0usize;
    // `smems_from_pos` returns the next start position, which is NOT simply `x + 1`: it is set to
    // wherever the forward extension died, so a long SMEM skips the whole span it covered. That
    // skipping is what makes all-position SMEM collection linear rather than quadratic.
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
///
/// # Parameters
/// - `fm`: the loaded FM index, shared read-only by all slots.
/// - `reads`: one 2-bit-encoded read per entry (same encoding as [`collect_smems`]'s `codes`).
///   Supplied by the batch driver in `bwa-mem`. May be empty; reads may have differing lengths and
///   may be empty individually. The slice index `r` is the `ridx` used throughout the slot machinery.
/// - `min_seed_len`: emission length floor in bases, shared by the whole batch (`opt.min_seed_len`).
/// - `min_intv`: occurrence-count floor, `1` for round 1. Shared by the whole batch here; the
///   per-slot copy exists because round 2 reuses the same slots with per-job values.
///
/// # Returns
/// One `Vec<Smem>` per input read, indexed by the same `r`, each equal to what [`collect_smems`]
/// would return for that read.
pub fn collect_smems_batched(
    fm: &FmIndex,
    reads: &[&[u8]],
    min_seed_len: i32,
    min_intv: i64,
) -> Vec<Vec<Smem>> {
    // Lockstep width: independent walks kept in flight so each slot's prefetch has a full cycle to
    // land before the block is used. 16 measured ~2.8% faster than 8 on a genome-scale index (M4 Max,
    // SE); 24 ties it and 32 regresses, so 16 is the knee. Shared by all three batched seeding rounds.
    let lockstep_slots = lockstep_width();

    // The index's C array, hoisted once: `counts[b]` is the number of reference bases strictly
    // smaller than `b`, so it is also the first BWT ROW whose suffix begins with `b`. Copied out of
    // `fm` so the hot slot loop does not re-borrow it per step.
    let counts = fm.counts();
    // Per-read result slots, pre-sized so a finished walk can be moved straight into `output[ridx]`
    // regardless of the order in which reads complete.
    let mut output: Vec<Vec<Smem>> = (0..reads.len()).map(|_| Vec::new()).collect();
    if reads.is_empty() {
        return output;
    }
    // Longest read in the batch, in bases. Sizes every slot's `prev` buffer once (`max_len + 2`) so
    // a slot recycled onto any other read in the batch still satisfies the capacity invariant of
    // `smems_from_pos`.
    let max_len = reads.iter().map(|r| r.len()).max().unwrap_or(0);

    // The lockstep slots. `None` means the slot is retired (no work left to hand it); a slot retires
    // only once `next_read` has passed the end of the batch.
    let mut slots: Vec<Option<LsSlot>> = Vec::with_capacity(lockstep_slots);
    // Index of the next read not yet assigned to a slot; reads `0..next_read` are started or done.
    let mut next_read = 0usize;
    for _ in 0..lockstep_slots {
        if next_read < reads.len() {
            slots.push(Some(LsSlot::new(next_read, 0, min_intv, false, max_len)));
            next_read += 1;
        } else {
            slots.push(None);
        }
    }

    // The lockstep driver. One pass over `slots` advances every live walk by exactly one FM
    // operation; a walk that finishes is immediately refilled with the next unstarted read, so the
    // pipeline stays full until the batch runs out. Each slot issues its prefetch at the end of its
    // step, and is not touched again until the other `N - 1` slots have had their turn, which is the
    // whole point: that gap is the latency being hidden.
    //
    // Invariant at the top of each `while` iteration: `live` is exactly the number of `Some` slots,
    // every `Some` slot holds a walk that has not yet reached `Done`-and-been-harvested, and reads
    // `next_read..` have not been started. Progress is guaranteed because every `step` either
    // advances a cursor or moves the phase forward.
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

/// Where a slot's walk currently sits inside [`smems_from_pos`]. One `step` call advances exactly one
/// phase iteration, so the FM operations of different reads interleave.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LsPhase {
    /// Set up the single-base interval at `x` (or skip an `N`).
    Start,
    /// One forward-extension iteration: append `codes[j]` on the right.
    Fwd,
    /// Forward-to-backward handover: push the final candidate and reverse to longest-first. Costs no
    /// FM operation, but is its own phase so the state machine stays a faithful line-by-line mirror.
    BwdInit,
    /// One backward-search outer iteration: prepend `codes[jj]` to every live candidate.
    Bwd,
    /// Emit the boundary SMEM and advance `x` to `next_x`.
    PosDone,
    /// Walk complete; the driver harvests `out` and refills the slot.
    Done,
}

/// One read's round-1 SMEM walk as a resumable state machine (see [`collect_smems_batched`]). Mirrors
/// [`smems_from_pos`] step by step: `Start` seeds the single-base interval at `x`, `Fwd` runs one
/// forward-extension iteration, `BwdInit` does the fwd->bwd housekeeping (append + reverse), `Bwd`
/// runs one backward-search outer iteration, `PosDone` emits the final SMEM and advances `x`.
struct LsSlot {
    /// Index of the read this slot is currently walking, within the caller's `reads` slice. The
    /// slot is recycled onto a new read (a new `ridx`) as soon as this walk finishes.
    ridx: usize,
    /// The read position the current SMEM search must cover (the C's `query_pos_array` entry).
    x: usize,
    /// Occurrence-count floor for this walk (`min_intv`). Round 1 shares one value across the batch;
    /// round-2 re-seeds each carry their own (`parent.s + 1`), so it lives per-slot.
    min_intv: i64,
    /// Round 2 re-seeds a *single* position and stops (`single_pos`); round 1 sweeps every position.
    /// Set at `new`/`reset_to` by the driver and read only at the two places a position ends
    /// (`Start`'s ambiguous-base early exit and `PosDone`).
    single_pos: bool,
    /// Which phase of [`smems_from_pos`] the next `step` call will execute. Starts at `Start`, ends
    /// at `Done`, at which point the driver harvests `out`.
    phase: LsPhase,
    /// The interval currently being extended in the forward phase (`smems_from_pos`'s `smem`).
    /// `k`/`l` are BWT ROW numbers (`0 ..= ref_seq_len`), `s` an occurrence count, `m`/`n` inclusive
    /// READ offsets. Live only between `Start` and `BwdInit`; the backward phase works out of `prev`.
    smem: Smem,
    /// Number of live candidates in `prev`, i.e. the valid prefix is `prev[..num_prev]`. Same
    /// variable as the C's `numPrev`. Grows during `Fwd`/`BwdInit`, only shrinks during `Bwd`.
    num_prev: usize,
    /// Forward-phase cursor: the READ offset being appended on the right (never a reference position).
    /// Runs `x + 1 ..= codes.len()`.
    j: usize,
    /// Backward-phase cursor: the READ offset being prepended on the left. Signed because it walks
    /// down to `-1` at the read start. Runs `x - 1` down to `-1`.
    jj: i64,
    /// Where the outer sweep resumes after this position (`smems_from_pos`'s return value): a READ
    /// offset, not necessarily `x + 1`, since a long SMEM skips the span it covered.
    next_x: usize,
    /// Candidate buffer. Allocated once at `max_len + 2` (the batch's longest read) and reused across
    /// every walk this slot runs, which is the point of `reset_to`: the invariant from
    /// [`smems_from_pos`] holds for any read in the batch.
    prev: Vec<Smem>,
    /// SMEMs emitted by the current walk, moved out when the slot reaches `Done`.
    out: Vec<Smem>,
}

impl LsSlot {
    /// Create a slot ready to walk read `ridx` from position `x`.
    ///
    /// # Parameters
    /// - `ridx`: index into the driver's `reads` slice of the read to walk.
    /// - `x`: READ offset to start the first SMEM search from. `0` for round 1; the parent SMEM's
    ///   midpoint for round 2.
    /// - `min_intv`: occurrence-count floor for this walk, `>= 1`.
    /// - `single_pos`: `true` for a round-2 re-seed (one position then stop), `false` for a round-1
    ///   full sweep.
    /// - `max_len`: length in bases of the longest read in the batch. Sizes `prev` to `max_len + 2`
    ///   so this slot can later be recycled onto any read in the batch without reallocating.
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
    ///
    /// # Parameters
    /// Same meanings as [`LsSlot::new`]: `ridx` the new read index, `x` the READ offset to start
    /// from, `min_intv` the new occurrence floor, `single_pos` whether this is a one-position
    /// re-seed. `prev`'s contents are deliberately left stale (the walk writes before it reads) and
    /// `out` is cleared because the previous walk's SMEMs have already been harvested.
    fn reset_to(&mut self, ridx: usize, x: usize, min_intv: i64, single_pos: bool) {
        self.ridx = ridx;
        self.x = x;
        self.min_intv = min_intv;
        self.single_pos = single_pos;
        self.phase = LsPhase::Start;
        self.num_prev = 0;
        self.out.clear();
    }

    /// Advance this walk by exactly one phase iteration (at most one FM extension of one candidate
    /// list), then return so the driver can give the other slots their turn.
    ///
    /// # Parameters
    /// - `fm`: the shared FM index.
    /// - `codes`: the read this slot is walking, i.e. the driver must pass `reads[self.ridx]`.
    ///   Passing any other read would corrupt the walk silently.
    /// - `min_seed_len`: emission length floor in bases (same value for the whole batch).
    /// - `counts`: the index's C array, hoisted by the driver. `counts[b]` is the first BWT ROW whose
    ///   suffix starts with base `b`; `counts[4]` is the total row count.
    fn step(&mut self, fm: &FmIndex, codes: &[u8], min_seed_len: i32, counts: &[i64; 5]) {
        // Per-walk occurrence floor: round 1 shares one value, round 2 carries the parent's `s + 1`.
        let min_intv = self.min_intv;
        // Length of this read in bases; the exclusive upper bound of every READ offset below.
        let readlength = codes.len();
        match self.phase {
            LsPhase::Start => {
                if self.x >= readlength {
                    self.phase = LsPhase::Done;
                    return;
                }
                self.next_x = self.x + 1;
                // 2-bit code of the read base at READ offset `x`; `>= 4` means `N`.
                let a = codes[self.x];
                if a >= AMBIGUOUS_BASE {
                    // No SMEM at an ambiguous base; advance one position (smems_from_pos returns x+1).
                    // A round-2 re-seed handles exactly one position, so it stops here regardless.
                    self.x = self.next_x;
                    if self.single_pos || self.x >= readlength {
                        self.phase = LsPhase::Done;
                    }
                    return;
                }
                let a = a as usize;
                // The single-base interval for `codes[x]`: `k` is the first BWT ROW starting with
                // that base, `l` the first row of its complement (the RC-side interval, which makes
                // the later k/l swap valid), `s` the number of times the base occurs in the 2L text.
                self.smem = Smem {
                    rid: 0,
                    m: self.x as u32,
                    n: self.x as u32,
                    k: counts[a],
                    l: counts[MAX_BASE - a],
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
                // Base code at the READ offset being appended on the right.
                let aj = codes[self.j];
                self.next_x = self.j + 1;
                if aj >= AMBIGUOUS_BASE {
                    self.phase = LsPhase::BwdInit;
                    return;
                }
                // Forward extension = backward extension on the reverse complement: swap k/l to
                // reinterpret the interval as the RC one, prepend the complement base `3 - aj`,
                // swap back. See the module docs.
                // `rc_view`: the same interval reinterpreted as that of the pattern's reverse
                // complement (k and l are BWT ROWS; swapping them swaps which strand's rows lead).
                let mut rc_view = self.smem;
                std::mem::swap(&mut rc_view.k, &mut rc_view.l);
                // `rc_extended`: interval of `complement(aj)` prepended to the RC pattern, which is
                // the RC of `pattern + aj`.
                let rc_extended = fm.backward_ext(rc_view, MAX_BASE - aj as usize);
                // `new_smem`: swapped back, so it is the forward interval of the read substring
                // `codes[x ..= j]`, one base longer on the RIGHT than `self.smem`.
                let mut new_smem = rc_extended;
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
                // Base code at the READ offset being prepended on the left this round.
                let a = codes[self.jj as usize];
                if a > 3 {
                    self.phase = LsPhase::PosDone;
                    return;
                }
                let a = a as usize;
                // Survivors kept so far this round; also the write cursor for the in-place compaction
                // of `prev`.
                let mut num_curr = 0usize;
                // Occurrence count of the last survivor kept this round. `-1` is impossible for a
                // real count, so the first survivor always passes the `!=` test.
                let mut curr_s = -1i64;

                // Read cursor over the live candidates. Invariant at the top of each iteration:
                // `prev[..num_curr]` holds this round's survivors (already left-extended by `a`) and
                // `prev[p..num_prev]` still holds last round's candidates, longest first. The write
                // cursor never overtakes the read cursor, so no candidate is clobbered before use.
                let mut p = 0usize;
                while p < self.num_prev {
                    let candidate = self.prev[p];
                    // Interval of `a` prepended to the candidate: same read substring one base longer
                    // on the LEFT. Plain backward extension, no k/l swap.
                    let mut new_smem = fm.backward_ext(candidate, a);
                    new_smem.m = self.jj as u32;
                    if new_smem.s < min_intv
                        && (i64::from(candidate.n) - i64::from(candidate.m) + 1)
                            >= i64::from(min_seed_len)
                    {
                        self.out.push(candidate);
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
                // Second loop over the remaining, strictly shorter candidates. They can no longer be
                // emitted this round; they survive only if they add a distinct occurrence count.
                while p < self.num_prev {
                    let candidate = self.prev[p];
                    let mut new_smem = fm.backward_ext(candidate, a);
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
                    // Candidates are kept longest-first, so `prev[0]` is the longest survivor. The
                    // backward phase ended at the read start or at an `N` rather than because a
                    // candidate died, so this one is left-maximal by boundary: emit it if long enough.
                    let longest = self.prev[0];
                    if (i64::from(longest.n) - i64::from(longest.m) + 1) >= i64::from(min_seed_len)
                    {
                        self.out.push(longest);
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
/// Faithful port of `getSMEMsOnePosOneThread`'s inner body (`FMI_search.cpp:496-670`).
///
/// # The algorithm (Li 2012, "Exploring single-sample SNP and INDEL calling", SMEM search)
///
/// Every SMEM that *covers position `x`* is found in two phases:
///
/// 1. **Forward phase.** Start from the single base `codes[x]` and extend right one base at a time.
///    Each extension shrinks the occurrence count `s` (monotonically: a longer pattern cannot occur
///    more often). Every time `s` actually *changes*, the interval just before the change is a
///    candidate right-boundary and is pushed onto `prev`. Intervals where `s` did not change are
///    overwritten, because a longer match with the same occurrence set strictly dominates the
///    shorter one: that is precisely the "super-maximal" filter, applied on the fly.
///
/// 2. **Backward phase.** Take those candidates (longest first, hence the `reverse`) and extend them
///    all *leftward* in lockstep, one base per outer iteration. A candidate that would drop below
///    `min_intv` on this base is maximal: it cannot grow left any further, so it is emitted (if long
///    enough). Candidates that survive with a *distinct* `s` are kept; a candidate whose `s` equals
///    one already kept this round is redundant (same occurrence set, shorter span) and dropped.
///
/// # Parameters
/// - `fm`: the shared FM index; supplies `counts`, `backward_ext` and `prefetch_occ`.
/// - `codes`: the 2-bit-encoded read. All indices below are READ offsets into this slice.
/// - `x`: read position the SMEMs must cover, `0 ..< codes.len()`. A READ offset, never a reference
///   position. The caller must ensure it is in range: `codes[x]` is indexed unconditionally.
/// - `min_seed_len`: length floor for *emission* only (bases). Short SMEMs are still walked, since
///   they gate the maximality tests above.
/// - `min_intv`: occurrence-count floor, `>= 1`. See [`collect_smems`].
/// - `prev`: caller-owned scratch, the candidate list. **Invariant: `prev.len() >= codes.len() + 2`.**
///   The forward phase pushes at most one entry per read position plus one final entry, and the
///   backward phase only ever compacts in place (`num_curr <= num_prev`), so it can never grow past
///   that. The C uses a VLA `SMEM prevArray[max_readlength]` (`FMI_search.cpp:510`); we size it
///   generously rather than exactly.
/// - `out`: SMEMs are appended, never cleared. Order matters for byte-identity downstream.
///
/// Returns `next_x`, the position [`collect_smems`] resumes from.
fn smems_from_pos(
    fm: &FmIndex,
    codes: &[u8],
    x: usize,
    min_seed_len: i32,
    min_intv: i64,
    prev: &mut [Smem],
    out: &mut Vec<Smem>,
) -> usize {
    // ---- Section 0: the single-base seed interval at `x` -------------------------------------
    // Reminder (see the module glossary): `x` is the read position every SMEM found here must
    // cover; `k`/`l`/`s` are the FM interval's forward start row, reverse-complement start row and
    // occurrence count; `m`/`n` are the match's inclusive span in the read.
    // Read length in bases: the exclusive bound on every READ offset in this function.
    let readlength = codes.len();
    // The index's C array. `counts[b]` = number of reference bases strictly smaller than `b`, which
    // is also the first BWT ROW whose suffix begins with `b`.
    let counts = fm.counts();
    // Where the outer sweep will resume. Provisionally `x + 1`; the forward phase may push it
    // further right (past a whole covered span) or pull it back to `j` when extension dies.
    let mut next_x = x + 1;
    // Base code at READ offset `x`, the position every SMEM found here must cover.
    let a = codes[x];
    // `N` (code 4, or anything non-ACGT): no exact match can cover it, so there is no SMEM at `x`.
    // Skip exactly one base, as the C does by falling through the `if (a < 4)` block.
    if a >= AMBIGUOUS_BASE {
        return next_x;
    }

    // Initial single-base interval. `counts` is the FM index's C array: `counts[b]` = number of
    // reference bases strictly smaller than `b`, so the rows beginning with base `b` are exactly
    // `[counts[b], counts[b+1])` and the pattern "b" occurs `counts[b+1] - counts[b]` times.
    // `l` starts at `counts[3 - a]`: the RC-side interval of a single base is the range of its
    // complement, which is what makes the k/l swap trick valid from step one
    // (`FMI_search.cpp:530-533`).
    let a = a as usize;
    // The interval currently being extended. Right now it is the interval of the one-base pattern
    // `codes[x]`; the forward loop grows it rightwards one base per iteration.
    let mut smem = Smem {
        rid: 0,
        m: x as u32,
        n: x as u32,
        k: counts[a],
        l: counts[MAX_BASE - a],
        s: counts[a + 1] - counts[a],
    };
    // Number of right-boundary candidates collected so far; the valid prefix is `prev[..num_prev]`.
    let mut num_prev = 0usize;

    // ---- Section 1: forward phase, collect right-boundary candidates ---------------------------
    // Forward extension (backward extension on the RC via swapped k/l and complemented base).
    // `j` is the read position being appended on the right.
    let mut j = x + 1;
    // Invariant at the top of each iteration: `smem` is the FM interval of the read substring
    // `codes[x ..= j - 1]` (so `smem.m == x`, `smem.n == j - 1`), `smem.s >= min_intv`, and
    // `prev[..num_prev]` holds the strictly shorter prefixes at which the occurrence count changed,
    // shortest first. `j` is the READ offset about to be appended on the right.
    while j < readlength {
        // Base code at the read offset being appended.
        let aj = codes[j];
        // `next_x` is updated *before* the N test, so a walk stopped by an N resumes past it.
        next_x = j + 1;
        if aj >= AMBIGUOUS_BASE {
            break;
        }
        // Swap k<->l to view the interval as its reverse complement, prepend the complement base
        // `3 - aj` (which is appending `aj` on the right), then swap back. See the module docs.
        // `rc_view`: `smem` reinterpreted as the interval of the reverse-complemented pattern (its
        // `k` and `l` are both BWT ROWS, so swapping them is legal and costs nothing).
        let mut rc_view = smem;
        std::mem::swap(&mut rc_view.k, &mut rc_view.l);
        // `rc_extended`: the RC pattern with `complement(aj)` prepended, i.e. the RC of the pattern
        // we actually want.
        let rc_extended = fm.backward_ext(rc_view, MAX_BASE - aj as usize);
        // `new_smem`: swapped back to the forward view. It is now the interval of `codes[x ..= j]`,
        // one base longer on the RIGHT than `smem`, and its `s` is `<= smem.s`.
        let mut new_smem = rc_extended;
        std::mem::swap(&mut new_smem.k, &mut new_smem.l);
        new_smem.n = j as u32;

        // Unconditional store, conditional bump: the C is `prevArray[numPrev] = smem; numPrev +=
        // s_neq_mask;` (`FMI_search.cpp:557-559`). If `s` did NOT change, the next iteration writes
        // over this slot, discarding the shorter match with the identical occurrence set. Only
        // length-vs-occurrence *breakpoints* survive as candidates.
        prev[num_prev] = smem;
        if new_smem.s != smem.s {
            num_prev += 1;
        }
        if new_smem.s < min_intv {
            // Extension died here. Resume at `j`, not `j + 1`: `codes[j]` itself may start a new
            // SMEM, so it must be re-examined. This one assignment is what makes the outer sweep
            // cover every position exactly.
            next_x = j;
            break;
        }
        smem = new_smem;
        j += 1;
    }
    // The final (longest) surviving interval is a candidate too, unless it fell below `min_intv`.
    if smem.s >= min_intv {
        prev[num_prev] = smem;
        num_prev += 1;
    }

    // Candidates were pushed shortest-first; the backward phase needs longest-first, because the
    // first entry to hit its left boundary is the longest one and that is the SMEM to emit.
    prev[..num_prev].reverse();

    // ---- Section 2: backward phase, emit the maximal candidates -------------------------------
    // Backward extension: `jj` walks left from `x - 1`, extending every candidate by the same base.
    let mut jj = x as i64 - 1;
    // Invariant at the top of each iteration: `prev[..num_prev]` are the live candidates, all of
    // them intervals of read substrings that start at READ offset `jj + 1` and end at their own `n`,
    // ordered longest-first, with strictly increasing occurrence counts `s`, all `>= min_intv`.
    // `jj` is the READ offset about to be prepended on the left; it is signed so the loop can fall
    // off the read start to `-1`.
    while jj >= 0 {
        // Base code at the read offset being prepended.
        let a = codes[jj as usize];
        if a > 3 {
            break;
        }
        let a = a as usize;
        // Candidates are processed longest-first, and `s` is non-decreasing along that order (a
        // shorter pattern occurs at least as often). `curr_s` is the last occurrence count kept this
        // round; `-1` is an impossible count, so the first survivor always passes the `!=` test.
        let mut num_curr = 0usize;
        let mut curr_s = -1i64;

        // First loop: scan until *one* of two things happens, then stop. Splitting it in two (rather
        // than one loop with a flag) is how the C is written (`FMI_search.cpp:606-632`), and the
        // split matters: whichever event fires, the very next candidate is skipped by the bare
        // `p += 1` below.
        // `p` reads last round's candidates, `num_curr` writes this round's survivors into the same
        // buffer. Invariant: `prev[..num_curr]` are already-extended survivors and `prev[p..num_prev]`
        // are not-yet-examined candidates; since `num_curr <= p` always, nothing is clobbered early.
        let mut p = 0usize;
        while p < num_prev {
            // One live candidate: the interval of `codes[jj + 1 ..= candidate.n]`.
            let candidate = prev[p];
            // Plain backward_ext here: no k/l swap, because this really is a left extension.
            // `new_smem` is the interval of `codes[jj ..= candidate.n]`, one base longer on the LEFT.
            let mut new_smem = fm.backward_ext(candidate, a);
            new_smem.m = jj as u32;
            // (a) This candidate cannot survive base `a` on the left, so it is left-maximal, and
            // being the longest live candidate it is also right-maximal: an SMEM. Emit it and stop
            // the whole backward phase for this position.
            if new_smem.s < min_intv
                && (i64::from(candidate.n) - i64::from(candidate.m) + 1) >= i64::from(min_seed_len)
            {
                out.push(candidate);
                break;
            }
            // (b) This candidate survives with a new occurrence count: keep it as the round's first
            // survivor and hand off to the second loop.
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
        // Unconditional skip: `p` is advanced past the candidate the loop stopped on, whether it was
        // emitted, kept, or the loop simply ran out. Matches the bare `p++` at `FMI_search.cpp:633`.
        p += 1;
        // Second loop: the remaining (shorter) candidates. They can no longer be emitted this round,
        // only kept if they contribute a *distinct* occurrence count. Note `prev[num_curr] = ...`
        // compacts in place over entries already consumed, so no second buffer is needed.
        while p < num_prev {
            // A strictly shorter candidate than the one the first loop stopped on.
            let candidate = prev[p];
            let mut new_smem = fm.backward_ext(candidate, a);
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
    // ---- Section 3: boundary case ------------------------------------------------------------
    // The loop ended at the read start (`jj < 0`) or at an `N`, not because a candidate died. Then
    // nothing was emitted above, and the longest survivor `prev[0]` is left-maximal by boundary.
    if num_prev != 0 {
        // Survivors are kept longest-first, so `prev[0]` is the longest one.
        let longest = prev[0];
        if (i64::from(longest.n) - i64::from(longest.m) + 1) >= i64::from(min_seed_len) {
            out.push(longest);
        }
    }
    next_x
}

/// Round-3 forward-only seeding (`bwtSeedStrategyAllPosOneThread`, `FMI_search.cpp:726`): from every
/// position, extend right until the occurrence count first drops below `max_intv`, and emit that
/// interval if it is at least `min_seed_len` long.
///
/// Why this exists at all when rounds 1/2 already produce SMEMs: an SMEM in a repetitive region can
/// have thousands of occurrences, of which only `max_occ` are ever sampled. Round 3 instead asks for
/// the *shortest* match that is already rare enough (`s < max_intv`, default `max_mem_intv = 20`),
/// which pins repeats far more precisely. It only ever emits one seed per starting position.
///
/// # Parameters
/// - `fm`: the shared FM index.
/// - `codes`: the 2-bit-encoded read; all offsets below are READ offsets into it.
/// - `max_intv`: occurrence-count *ceiling* (note: opposite sense to `min_intv` elsewhere). Callers
///   pass `opt.max_mem_intv`, default 20. Round 3 is skipped entirely when that is `<= 0`, so this
///   function is only ever called with a positive value.
/// - `min_seed_len`: emission length floor in bases. Callers pass `opt.min_seed_len + 1`, exactly as
///   `bwamem.cpp:779` does; the `+ 1` is bwa's, not a typo here.
/// - `out`: SMEMs are appended, never cleared. Callers pass the same vector that already holds the
///   round-1 and round-2 SMEMs, so round 3's entries land last.
fn bwt_seed_strategy(
    fm: &FmIndex,
    codes: &[u8],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut Vec<Smem>,
) {
    let counts = fm.counts();
    let readlength = codes.len();
    // Outer sweep cursor: the READ offset each forward-only walk starts from.
    let mut x = 0usize;
    // Invariant at the top of each iteration: every start position `< x` has been walked and has
    // contributed at most one seed to `out`.
    while x < readlength {
        // Where the sweep resumes; the inner loop pushes it right as it consumes bases.
        let mut next_x = x + 1;
        if codes[x] < AMBIGUOUS_BASE {
            let a = codes[x] as usize;
            // Single-base interval at `x`, grown rightwards only (round 3 has no backward phase).
            let mut smem = Smem {
                rid: 0,
                m: x as u32,
                n: x as u32,
                k: counts[a],
                l: counts[MAX_BASE - a],
                s: counts[a + 1] - counts[a],
            };
            let mut j = x + 1;
            // Invariant at the top of each iteration: `smem` is the interval of `codes[x ..= j - 1]`
            // and its occurrence count is still `>= max_intv` (or the match is still too short), so
            // the walk must keep growing. `j` is the READ offset about to be appended on the right.
            while j < readlength {
                next_x = j + 1;
                let aj = codes[j];
                if aj >= AMBIGUOUS_BASE {
                    break;
                }
                // Same forward-extension dance as elsewhere: view as reverse complement (swap the
                // two BWT ROW fields), prepend the complement base, swap back.
                let mut rc_view = smem;
                std::mem::swap(&mut rc_view.k, &mut rc_view.l);
                let rc_extended = fm.backward_ext(rc_view, MAX_BASE - aj as usize);
                // Interval of `codes[x ..= j]`, one base longer on the RIGHT.
                let mut new_smem = rc_extended;
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

/// Where a [`BwtSeedSlot`]'s round-3 walk currently sits inside [`bwt_seed_strategy`]. One `step`
/// call advances exactly one phase iteration, so different reads' FM operations interleave.
#[derive(PartialEq, Eq)]
enum BwtPhase {
    /// Set up the single-base interval at `x` (or skip an `N` and move to the next position).
    Start,
    /// One forward-extension iteration: append `codes[j]` on the right, and emit plus end the
    /// position if the occurrence count has dropped below `max_intv` at a sufficient length.
    Fwd,
    /// The whole read has been swept; the driver harvests `out` and refills the slot.
    Done,
}

/// One read's round-3 (`bwt_seed_strategy`) forward-seeding walk as a resumable state machine, so a
/// batch of reads can run their walks in lockstep (see [`bwt_seed_strategy_batched`]) and hide the
/// `cp_occ` DRAM latency of the forward extension. Mirrors [`bwt_seed_strategy`] exactly: `Start`
/// seeds the single-base interval at `x`, `Fwd` runs one forward-extension iteration.
struct BwtSeedSlot {
    /// Index of the read this slot is currently walking, within the caller's `reads` slice.
    ridx: usize,
    /// The starting READ offset of the current forward walk, `0 ..< readlength`.
    x: usize,
    /// Where the outer sweep resumes after this position: a READ offset, `x + 1` at minimum and up
    /// to one past wherever the forward walk stopped.
    next_x: usize,
    /// Forward cursor: the READ offset being appended on the right. Runs `x + 1 ..= readlength`.
    j: usize,
    /// The interval being extended: `k`/`l` are BWT ROWS, `s` the occurrence count, `m`/`n` the
    /// inclusive READ span. Valid only while `phase == Fwd`.
    smem: Smem,
    /// Which phase the next `step` call executes. `Done` tells the driver to harvest and refill.
    phase: BwtPhase,
    /// SMEMs emitted by the current walk, moved out when the slot reaches `Done`.
    out: Vec<Smem>,
}

impl BwtSeedSlot {
    /// Create a slot that will sweep read `ridx` (an index into the driver's `reads` slice) from
    /// READ offset 0. Round 3 always starts at the read start, so there is no `x` parameter.
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

    /// Re-point this slot at read `ridx`, restarting the sweep at READ offset 0 and clearing the
    /// output (the previous walk's SMEMs have already been harvested by the driver).
    fn reset(&mut self, ridx: usize) {
        self.ridx = ridx;
        self.x = 0;
        self.phase = BwtPhase::Start;
        self.out.clear();
    }

    /// End the current position: advance `x` to `next_x` and return to `Start` (or finish).
    ///
    /// # Parameters
    /// - `readlength`: length of the current read in bases, i.e. the exclusive bound past which the
    ///   sweep is complete.
    #[inline]
    fn end_pos(&mut self, readlength: usize) {
        self.x = self.next_x;
        self.phase = if self.x >= readlength {
            BwtPhase::Done
        } else {
            BwtPhase::Start
        };
    }

    /// Advance this round-3 walk by exactly one phase iteration (at most one FM extension).
    ///
    /// # Parameters
    /// - `fm`: the shared FM index.
    /// - `codes`: the read this slot is walking; the driver must pass `reads[self.ridx]`.
    /// - `max_intv`: occurrence-count ceiling, `> 0` (see [`bwt_seed_strategy`]).
    /// - `min_seed_len`: emission length floor in bases (`opt.min_seed_len + 1` from the caller).
    /// - `counts`: the index's C array, hoisted by the driver; `counts[b]` is the first BWT ROW whose
    ///   suffix begins with base `b`.
    fn step(
        &mut self,
        fm: &FmIndex,
        codes: &[u8],
        max_intv: i64,
        min_seed_len: i32,
        counts: &[i64; 5],
    ) {
        // Read length in bases: the exclusive bound on every READ offset here.
        let readlength = codes.len();
        match self.phase {
            BwtPhase::Start => {
                if self.x >= readlength {
                    self.phase = BwtPhase::Done;
                    return;
                }
                self.next_x = self.x + 1;
                if codes[self.x] >= AMBIGUOUS_BASE {
                    self.end_pos(readlength);
                    return;
                }
                // Base code at the walk's start offset; known non-ambiguous by the test above.
                let a = codes[self.x] as usize;
                // Single-base interval at `x`: `k` the first BWT ROW starting with that base, `l` the
                // first row of its complement, `s` the base's occurrence count in the 2L text.
                self.smem = Smem {
                    rid: 0,
                    m: self.x as u32,
                    n: self.x as u32,
                    k: counts[a],
                    l: counts[MAX_BASE - a],
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
                // Base code at the READ offset being appended on the right.
                let aj = codes[self.j];
                if aj >= AMBIGUOUS_BASE {
                    self.end_pos(readlength);
                    return;
                }
                // Forward extension via the reverse-complement view (swap the two BWT ROW fields,
                // prepend the complement base, swap back).
                let mut rc_view = self.smem;
                std::mem::swap(&mut rc_view.k, &mut rc_view.l);
                let rc_extended = fm.backward_ext(rc_view, MAX_BASE - aj as usize);
                // Interval of `codes[x ..= j]`, one base longer on the RIGHT than before.
                let mut new_smem = rc_extended;
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
///
/// # Parameters
/// - `fm`: the shared FM index.
/// - `reads`: the batch, one 2-bit-encoded read per entry; index `r` is the slot's `ridx`.
/// - `max_intv`: occurrence-count ceiling, `> 0` (callers gate on `opt.max_mem_intv > 0`).
/// - `min_seed_len`: emission length floor in bases (`opt.min_seed_len + 1`).
/// - `out`: per-read SMEM vectors, already holding rounds 1 and 2. Must have the same length as
///   `reads`, since it is indexed by `ridx`. Round-3 SMEMs are **appended**, never replacing.
fn bwt_seed_strategy_batched(
    fm: &FmIndex,
    reads: &[&[u8]],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut [Vec<Smem>],
) {
    let lockstep_slots = lockstep_width();
    if reads.is_empty() {
        return;
    }
    // The index's C array, hoisted out of the hot loop (see `collect_smems_batched`).
    let counts = fm.counts();
    // The lockstep slots; `None` marks a retired slot with no read left to take.
    let mut slots: Vec<Option<BwtSeedSlot>> = Vec::with_capacity(lockstep_slots);
    // Index of the next read not yet handed to a slot.
    let mut next_read = 0usize;
    for _ in 0..lockstep_slots {
        if next_read < reads.len() {
            slots.push(Some(BwtSeedSlot::new(next_read)));
            next_read += 1;
        } else {
            slots.push(None);
        }
    }
    // Number of `Some` slots. Invariant at the top of each pass: every live slot holds an unfinished
    // walk, reads `next_read..` are unstarted, and a slot only retires once both are exhausted.
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
///
/// The returned order is round 1, then round 2, then round 3, i.e. **not** sorted by read position.
/// bwa sorts at this point (`sortSMEMs`, `bwamem.cpp:786`); we instead sort inside chaining
/// (`bwa_chain::sa_positions_for_read` orders by `(m, n)`), so the order here is not observable.
///
/// # Parameters
/// - `fm`: the shared FM index.
/// - `codes`: one 2-bit-encoded read.
/// - `opt`: the alignment options. Only `min_seed_len` (`-k`), `split_factor` (`-s`), `split_width`
///   and `max_mem_intv` are read here; they come from the command line via `bwa_core::MemOpt`.
///
/// # Returns
/// All three rounds' SMEMs for this read, still in FM-interval form (BWT rows plus READ span);
/// resolving them to reference positions is [`seeds_from_smem`]'s job.
pub fn mem_collect_smem(fm: &FmIndex, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    // Round 1, then rounds 2/3 which depend on it.
    let mut smems = collect_smems(fm, codes, opt.min_seed_len, 1);
    smem_rounds_2_3(fm, codes, opt, &mut smems);
    smems
}

/// Batched seeding for a read batch: round-1 SMEMs are collected in **lockstep**
/// ([`collect_smems_batched`], hiding FM-index latency), then rounds 2/3 (re-seeding + interval-capped
/// forward seeding) run per read. Returns, for every read, exactly what [`mem_collect_smem`] would.
///
/// # Parameters
/// - `fm`: the shared FM index.
/// - `reads`: the batch of 2-bit-encoded reads; the returned outer index matches this one.
/// - `opt`: alignment options; same fields as [`mem_collect_smem`].
///
/// # Returns
/// One SMEM vector per read, ordered round 1, then round 2, then round 3.
pub fn mem_collect_smem_batched(fm: &FmIndex, reads: &[&[u8]], opt: &MemOpt) -> Vec<Vec<Smem>> {
    // Round 1 in lockstep, `min_intv = 1` (accept any pattern that occurs at all). This vector is
    // then grown in place by rounds 2 and 3.
    let mut per_read = collect_smems_batched(fm, reads, opt.min_seed_len, 1);
    // Rounds 2 and 3 both run in lockstep across the batch so their data-dependent `cp_occ` loads
    // overlap — the same latency-hiding trick as round 1. Order per read is preserved (round 1,
    // then round 2, then round 3), so this is identical to the per-read path.
    // `BWA4_SEED_R2_SERIAL` forces the old per-read round 2, for regression/parity verification.
    if std::env::var_os("BWA4_SEED_R2_SERIAL").is_some() {
        for (r, codes) in reads.iter().enumerate() {
            smem_round_2(fm, codes, opt, &mut per_read[r]);
        }
    } else {
        smem_round_2_batched(fm, reads, opt, &mut per_read);
    }
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_batched(
            fm,
            reads,
            opt.max_mem_intv,
            opt.min_seed_len + 1,
            &mut per_read,
        );
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
///
/// # Parameters
/// - `fm`: the FM index, still needed for rounds 2 and 3.
/// - `lsa`: the learned suffix array (BWA-MEME's index), used for round 1 only. Must be built over
///   the same reference as `fm`, or the two rounds would disagree about coordinates.
/// - `reads`: the batch of 2-bit-encoded reads.
/// - `opt`: alignment options; same fields as [`mem_collect_smem`].
///
/// # Returns
/// One SMEM vector per read, same content as [`mem_collect_smem_batched`] up to round-1 ordering.
pub fn mem_collect_smem_hybrid(
    fm: &FmIndex,
    lsa: &bwa_index::lisa::LearnedSa,
    reads: &[&[u8]],
    opt: &MemOpt,
) -> Vec<Vec<Smem>> {
    // Round 1 from the learned index, per read (the zigzag walk has no lockstep variant and does not
    // need one: it jumps straight to each LEM instead of walking `cp_occ` per base).
    let mut per_read: Vec<Vec<Smem>> = reads
        .iter()
        .map(|codes| collect_smems_lsa_zigzag(lsa, codes, opt.min_seed_len))
        .collect();
    smem_round_2_batched(fm, reads, opt, &mut per_read);
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_batched(
            fm,
            reads,
            opt.max_mem_intv,
            opt.min_seed_len + 1,
            &mut per_read,
        );
    }
    per_read
}

/// Rounds 2 (midpoint re-seeding of long non-repetitive SMEMs) and 3 (interval-capped forward
/// seeding), appended to the round-1 `smems` in place. Shared by the per-read and batched entry
/// points so they stay identical.
///
/// # Parameters
/// - `fm`: the shared FM index.
/// - `codes`: the 2-bit-encoded read that produced `smems`. Must be the same read, since round 2
///   re-searches READ offsets taken from those SMEMs' spans.
/// - `opt`: alignment options; reads `min_seed_len`, `split_factor`, `split_width`, `max_mem_intv`.
/// - `smems`: the read's round-1 SMEMs on entry, extended in place with rounds 2 and 3.
fn smem_rounds_2_3(fm: &FmIndex, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    smem_round_2(fm, codes, opt, smems);
    // Round 3.
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy(fm, codes, opt.max_mem_intv, opt.min_seed_len + 1, smems);
    }
}

/// Round 2: re-seed each long, non-repetitive round-1 SMEM from its midpoint (appends in place).
/// Port of the second stage of `mem_collect_smem` (`bwamem.cpp:696-712`).
///
/// The idea: a very long SMEM may be hiding shorter matches that, individually, map somewhere else
/// (a repeat copy, a chimeric junction). Searching again from the SMEM's midpoint while *demanding*
/// more occurrences than the parent (`min_intv = p.s + 1`) forces the walk to stop earlier and
/// surface exactly those shorter, more-frequent matches. Since the new SMEMs must occur strictly
/// more often, they can never be duplicates of the parent.
///
/// Selection criteria, both from bwa:
/// - `end - start >= split_len` where `split_len = round(min_seed_len * split_factor)`. With the
///   defaults (19, 1.5) that is 28 bases; `+ 0.499` reproduces the C's `(int)(... + .499)` truncation
///   at `bwamem.cpp:630`, so the `f32` arithmetic and the cast must stay exactly as written.
/// - `p.s <= split_width` (`opt.split_width`, default 10): only *non*-repetitive SMEMs are re-seeded.
///   Re-seeding a repeat would just produce more repeat hits.
///
/// `n_round1_smems` (the C's `num_smem1`) is captured before the loop so the round-2 SMEMs being
/// appended are not themselves re-seeded (the C iterates over `num_smem1` for the same reason).
///
/// # Parameters
/// - `fm`: the shared FM index.
/// - `codes`: the 2-bit-encoded read the SMEMs came from.
/// - `opt`: alignment options; reads `min_seed_len`, `split_factor` and `split_width`.
/// - `smems`: the read's round-1 SMEMs on entry; round-2 SMEMs are appended in place.
fn smem_round_2(fm: &FmIndex, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    // Length threshold in bases above which a round-1 SMEM is considered "long" enough to re-seed.
    // Defaults: 19 * 1.5 + 0.499 truncates to 28.
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + SPLIT_LEN_ROUNDING) as i32;
    // Number of round-1 SMEMs, captured before the loop so appended round-2 SMEMs are not re-seeded.
    let n_round1_smems = smems.len();
    // Candidate scratch shared by every re-seed walk (see `collect_smems` for the sizing rule).
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    for idx in 0..n_round1_smems {
        // The parent SMEM being considered for re-seeding.
        let p = smems[idx];
        // Parent's READ span as a half-open interval `[start, end)`; both are READ offsets.
        let start = p.m as i32;
        let end = p.n as i32 + 1;
        if end - start < split_len || p.s > i64::from(opt.split_width) {
            continue;
        }
        // Midpoint of the half-open span `[start, end)`, matching `query_pos_ar[pos] = (end +
        // start) >> 1` at `bwamem.cpp:707`. Arithmetic shift on a non-negative i32, so it is a floor.
        let x = ((end + start) >> 1) as usize;
        smems_from_pos(fm, codes, x, opt.min_seed_len, p.s + 1, &mut scratch, smems);
    }
}

/// A single round-2 re-seed: re-run the SMEM walk of read `ridx` from position `x` with occurrence
/// floor `min_intv` (the parent SMEM's `s + 1`). Enumerated up front from the round-1 output so the
/// walks across the whole batch can run in lockstep.
struct ReseedJob {
    /// Index of the read to re-seed, within the batch's `reads` slice.
    ridx: usize,
    /// READ offset to restart the SMEM walk from: the midpoint of the parent SMEM's span.
    x: usize,
    /// Occurrence-count floor for this walk: the parent SMEM's `s + 1`, so any SMEM found must occur
    /// strictly more often than the parent and can never duplicate it. Always `>= 2`.
    min_intv: i64,
}

/// Batched round 2: the per-read [`smem_round_2`] runs each midpoint re-seed as an isolated,
/// latency-exposed [`smems_from_pos`] walk — measured as ~half of all seeding time on a genome-scale
/// index, because unlike rounds 1 and 3 it is not lockstepped, so every data-dependent `cp_occ` load
/// stalls on DRAM. This enumerates all re-seed jobs across the batch (in `(read, round-1 SMEM index)`
/// order) and advances `N` of them round-robin, so each slot's prefetch covers a full cycle before
/// its block is used. Each job's SMEMs are appended to its read in job order, identical to running
/// [`smem_round_2`] per read.
///
/// # Parameters
/// - `fm`: the shared FM index.
/// - `reads`: the batch of 2-bit-encoded reads.
/// - `opt`: alignment options; reads `min_seed_len`, `split_factor` and `split_width`.
/// - `per_read`: each read's round-1 SMEMs on entry (must be indexed the same as `reads`, since
///   `ridx` indexes both). Round-2 SMEMs are appended in place. Jobs are enumerated from the entry
///   contents, so the SMEMs appended here are never themselves re-seeded.
fn smem_round_2_batched(fm: &FmIndex, reads: &[&[u8]], opt: &MemOpt, per_read: &mut [Vec<Smem>]) {
    let lockstep_slots = lockstep_width();
    // Same threshold as `smem_round_2`, in bases: 28 with the defaults. The `f32` arithmetic must
    // stay identical to the per-read path or the two would select different SMEMs.
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + SPLIT_LEN_ROUNDING) as i32;

    // Enumerate the re-seed jobs, preserving per-read append order (round-1 SMEM index ascending).
    let mut jobs: Vec<ReseedJob> = Vec::new();
    for (ridx, smems) in per_read.iter().enumerate() {
        for p in smems.iter() {
            // Parent SMEM's READ span as a half-open interval `[start, end)`.
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

    // The index's C array, hoisted out of the hot loop.
    let counts = fm.counts();
    // Longest read in the batch, in bases; sizes each slot's `prev` buffer so a slot can be recycled
    // onto a job belonging to any other read.
    let max_len = reads.iter().map(|r| r.len()).max().unwrap_or(0);
    // Each job's output, filled as it completes; appended in job order at the end.
    let mut results: Vec<Vec<Smem>> = (0..jobs.len()).map(|_| Vec::new()).collect();

    // Slots pair a job id (index into `jobs`/`results`) with the walk state, because jobs complete
    // out of order and the result must go back to the right `results` entry.
    let mut slots: Vec<Option<(usize, LsSlot)>> = Vec::with_capacity(lockstep_slots);
    // Index of the next job not yet handed to a slot.
    let mut next_job = 0usize;
    for _ in 0..lockstep_slots {
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

    // Number of `Some` slots. Invariant at the top of each pass: every live slot is mid-walk on the
    // job named by its `job_id`, jobs `next_job..` are unstarted, and `results[..]` holds output only
    // for jobs already completed. A slot retires only when both are exhausted.
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
/// (bwa-mem2's `get_sa_entries`, `FMI_search.cpp:1077-1101`).
///
/// The SMEM's interval `[k, k + s)` names `s` BWT rows; each row's suffix-array value is one
/// reference position. When `s` exceeds `max_occ` (default 500, `-c`), bwa does not take the first
/// `max_occ` rows: it *strides* by `step = s / max_occ` so the sample is spread across the interval.
/// The two loop bounds `j < k + s` and `c < max_occ` are both needed because integer division floors
/// `step`, so striding alone would overshoot the count.
///
/// `fm.get_sa` is the expensive part (each call may LF-walk to the nearest suffix-array sample); the
/// chaining crate batches these across a whole read via `get_sa_batch` instead of calling here.
///
/// # Parameters
/// - `fm`: the shared FM index; only `get_sa` is used, to turn a BWT ROW into a reference position.
/// - `smem`: one SMEM, i.e. an FM interval `[k, k + s)` of BWT ROWS plus the READ span `(m, n)`.
///   `s` may be 0 (no occurrences), in which case no seeds are produced.
/// - `max_occ`: cap on how many occurrences to sample, in seeds. `opt.max_occ`, default 500 (`-c`).
///   Must be `> 0`: it is a divisor below.
///
/// # Returns
/// Up to `max_occ` seeds, each carrying a 2L-space reference position (`rbeg < l_pac` forward strand,
/// `rbeg >= l_pac` reverse strand), the READ offset `qbeg`, and the exact-match length.
pub fn seeds_from_smem(fm: &FmIndex, smem: &Smem, max_occ: i32) -> Vec<MemSeed> {
    // Reminder: `k` is the interval's first BWT row, `s` its size (occurrence count), `m`/`n` the
    // match's inclusive span in the read. `j` below is a BWT row and `c` the number of occurrences
    // sampled so far; both keep the C's names (`get_sa_entries`, `FMI_search.cpp:1077`).
    // Exact-match length in bases; the `+ 1` because `m`/`n` are inclusive READ offsets.
    let len = (i64::from(smem.n) - i64::from(smem.m) + 1) as i32;
    let max_occ = i64::from(max_occ);
    // Row stride between sampled occurrences. `1` when the interval already fits under the cap;
    // otherwise `s / max_occ` (floored), which spreads the sample across the whole interval instead
    // of taking the first `max_occ` rows, whose reference positions would be arbitrarily clustered.
    let step = if smem.s > max_occ {
        smem.s / max_occ
    } else {
        1
    };
    let mut seeds = Vec::new();
    // Occurrences sampled so far; the floored `step` can under-shoot, so this cap is also needed.
    let mut c = 0i64;
    // Current BWT ROW being sampled (a suffix-array row number, not a reference position). Walks
    // `k, k + step, k + 2*step, ...` while inside the interval `[k, k + s)`.
    let mut j = smem.k;
    while j < smem.k + smem.s && c < max_occ {
        seeds.push(MemSeed {
            // `get_sa` converts the BWT ROW `j` into a 2L-space reference POSITION: the offset in
            // `[forward genome][reverse complement]` where this occurrence starts.
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

    /// Load the small checked-in test index (`testdata/tiny/tiny.fa` plus its `.bwt`/`.sa` siblings).
    /// Small enough to be cache-resident, so these tests exercise correctness, never latency.
    fn tiny() -> FmIndex {
        // The index prefix: the FASTA path, with the index files sitting next to it.
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        FmIndex::load(Path::new(prefix)).unwrap()
    }

    /// Hybrid seeding (LISA round-1 + FM rounds 2/3) must produce, per read, the same SMEM SET as the
    /// full FM seeding — after chaining's `(m, n)` sort, the set is what determines the alignment, so
    /// this is the seeding-level byte-identity gate for the hybrid path.
    #[test]
    fn hybrid_seeding_equals_fm() {
        let fm = tiny();
        // Learned suffix array over the same reference, so the two round-1 paths are comparable.
        // 4096 is the LISA leaf size (a build-time space/speed knob, not a correctness one).
        let lsa = bwa_index::lisa::LearnedSa::build(fm.reference().to_vec(), 4096);
        let opt = MemOpt::default();
        // Length of the FORWARD reference in bases; reads are drawn from the forward strand only, so
        // start positions must stay below it.
        let l_pac = fm.l_pac() as usize;
        // Seed of a deterministic LCG, so a failure is reproducible.
        let mut state = 0xdead_beef_0042u64;
        // Deterministic 64-bit LCG (Knuth's multiplier); returns the top 31 bits, which are the
        // well-mixed ones. Used only to build test reads, never in the aligner itself.
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..200 {
            // Read length in bases, 40..=159.
            let len = 40 + (next() as usize % 120);
            // Forward-strand reference POSITION the read is copied from.
            let start = next() as usize % (l_pac - len);
            let mut r: Vec<u8> = (0..len).map(|i| fm.base((start + i) as i64)).collect();
            for _ in 0..(next() % 4) {
                // A READ offset to mutate, so SMEMs get split at varied places.
                let p = next() as usize % len;
                r[p] = (next() % 4) as u8;
            }
            reads.push(r);
        }
        // Borrowed view of the batch, in the `&[&[u8]]` shape the batched entry points take.
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        let fm_all = mem_collect_smem_batched(&fm, &refs, &opt);
        let hy_all = mem_collect_smem_hybrid(&fm, &lsa, &refs, &opt);
        // Order-insensitive comparison key: the sorted set of `(read span m, read span n, first BWT
        // row k, occurrence count s)`. Order is deliberately not compared, since chaining re-sorts.
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
    ///
    /// # Parameters
    /// - `reference`: the whole 2L text as 2-bit codes, `2 * l_pac` bases long.
    /// - `pat`: the read substring to count, same encoding, non-empty.
    ///
    /// # Returns
    /// The brute-force occurrence count, which must equal the FM interval's `s`.
    fn naive_occ(reference: &[u8], pat: &[u8]) -> i64 {
        reference.windows(pat.len()).filter(|w| *w == pat).count() as i64
    }

    #[test]
    fn smems_are_exact_matches() {
        let fm = tiny();
        // Read = a 120bp forward slice of the reference; must seed back to its origin.
        // `start` is a forward-strand reference POSITION, `len` a length in bases.
        let start = 50_000i64;
        let len = 120usize;
        let read: Vec<u8> = (0..len).map(|i| fm.base(start + i as i64)).collect();

        let smems = collect_smems(&fm, &read, 19, 1);
        assert!(!smems.is_empty(), "no SMEMs found");

        // Materialize the 2L reference once; `reference()` now allocates, so it must not be called
        // per SMEM inside the loop.
        let reference = fm.reference();
        // Each SMEM's read substring occurs exactly `s` times in the reference.
        for sm in &smems {
            // The read substring this SMEM's interval stands for (`m`/`n` are inclusive READ offsets).
            let sub = &read[sm.m as usize..=sm.n as usize];
            assert_eq!(
                sm.s,
                naive_occ(&reference, sub),
                "SMEM interval size wrong"
            );
            assert!((sm.n - sm.m + 1) as i32 >= 19);
        }

        // The full-length SMEM must exist and seed to the origin position.
        // "Full length" = READ span `[0, len - 1]`, i.e. the entire read matched exactly.
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
        // Deterministic 64-bit LCG (Knuth's multiplier); returns the top 31 bits, which are the
        // well-mixed ones. Used only to build test reads, never in the aligner itself.
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        // Length of the whole 2L text in bases (`2 * l_pac`): slices may straddle into the RC half,
        // which is fine here since only batched-vs-per-read agreement is being checked.
        let reflen = 2 * fm.l_pac(); // 2L, without materializing the reference
        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..60 {
            // Read flavour: 0 = exact reference slice, 1/2 = random, 3 = random with an injected `N`.
            let kind = next() % 4;
            // Read length in bases, 1..=160, so empty-ish and single-base edge cases are covered.
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
        // Borrowed view of the batch, in the `&[&[u8]]` shape the batched entry points take.
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();

        // Sweep `(min_seed_len, min_intv)` pairs: the default 19/1, a shorter floor, a round-2-like
        // occurrence floor of 2, and a very short floor that makes almost every position emit.
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
        // Deterministic 64-bit LCG (Knuth's multiplier); returns the top 31 bits, which are the
        // well-mixed ones. Used only to build test reads, never in the aligner itself.
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        // Length of the whole 2L text in bases (`2 * l_pac`).
        let reflen = 2 * fm.l_pac(); // 2L, without materializing the reference
        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..80 {
            // Read flavour: 0 = random (shallow SMEMs), 1..=4 = exact reference slice, and 3/4 also
            // get two point mutations.
            let kind = next() % 5;
            // Read length in bases, 40..=189: long enough that SMEMs can exceed `split_len` (28) and
            // so actually trigger round-2 re-seeding.
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
        // Borrowed view of the batch, in the `&[&[u8]]` shape the batched entry points take.
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();

        let opt = MemOpt::default();
        let batched = mem_collect_smem_batched(&fm, &refs, &opt);
        for (r, read) in reads.iter().enumerate() {
            let per_read = mem_collect_smem(&fm, read, &opt);
            assert_eq!(
                batched[r],
                per_read,
                "batched full != per-read at read {r} (len {})",
                read.len()
            );
        }
    }

    #[test]
    fn smems_cover_repeated_region() {
        let fm = tiny();
        // A short read is still collected as an SMEM if >= min_seed_len.
        // Forward-strand reference POSITION, clamped so the 60bp slice stays inside the forward half.
        let start = 123_456i64.min(fm.l_pac() - 60);
        let read: Vec<u8> = (0..60).map(|i| fm.base(start + i as i64)).collect();
        let smems = collect_smems(&fm, &read, 19, 1);
        // Materialize the 2L reference once; `reference()` now allocates.
        let reference = fm.reference();
        for sm in &smems {
            // The read substring this SMEM's interval stands for.
            let sub = &read[sm.m as usize..=sm.n as usize];
            assert_eq!(sm.s, naive_occ(&reference, sub));
        }
    }
}
