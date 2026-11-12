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
//! # Rust mechanics used in this file
//!
//! Two structures dominate this file and neither is obvious from the outside.
//!
//! The first is the LOCKSTEP driver. Seeding is bound by memory latency, not by arithmetic: each
//! index lookup reaches into a multi-gigabyte array and waits. So instead of finishing one read
//! before starting the next, the driver keeps several reads in flight at once, advancing each by
//! one step per pass, so the waits overlap. `Vec<Option<LsSlot>>` is that set of in-flight reads,
//! and a slot becoming `None` is how a finished read leaves the rotation.
//!
//! The second is `std::mem::swap` and `std::mem::take`. Rust will not let you move a value out of
//! a struct field or a vector slot, because that would leave behind something the type system has
//! no name for. These two functions are the sanctioned way around it: `swap` exchanges the contents
//! of two places, and `take` removes a value and leaves the type's default behind. Both are used
//! here to move whole vectors around without copying their contents.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `Vec<Option<LsSlot>>` | the in-flight read slots. `Some` is an active read, `None` an empty seat. |
//! | `std::mem::take(&mut x)` | takes `x`'s value and leaves the default (an empty vector) in its place. Used to hand a finished read's seed list to the output without copying it. |
//! | `std::mem::swap(&mut a, &mut b)` | exchanges two values. In the interval arithmetic it swaps the forward and reverse-complement bounds, which is the C's pointer trick expressed without pointers. |
//! | `Vec<Vec<Smem>>` | one seed list per input read, indexed by read number. |
//! | `&mut Vec<Smem>` as a parameter | an output buffer supplied by the caller and appended to in place, so one allocation is reused across every read instead of one per read. |
//! | `&mut [Smem]` | an exclusive borrow of a scratch buffer whose length is fixed. Reused the same way. |
//! | `(0..n).map(\|_\| Vec::new()).collect()` | builds `n` independent empty vectors. `_` marks the counter as unused; `vec![Vec::new(); n]` would not do, since that would require cloning. |
//! | `.filter(\|s\| s.is_some()).count()` | counts the still-active slots, which is the rotation's stopping condition. |
//! | `.map(\|r\| r.len()).max().unwrap_or(0)` | the longest read, with 0 for an empty input. |
//! | `Vec::with_capacity(n)` | reserves room up front so the slot table never reallocates mid-run. |
//! | `Smem::default()` | the type's zero value, from a derived `Default`. `vec![Smem::default(); n]` pre-fills the scratch buffer. |
//! | `as usize` / `as i64` | explicit conversions between the signed arithmetic the C uses and the unsigned type Rust needs for indexing. Frequent here for that reason alone. |
//! | `&[i64; 5]` | a borrowed FIXED-size array of five counts, one per base code including N. The length is part of the type, so a mis-sized table will not compile. |
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
/// Any value `>= 1` is correct. **32 is the knee on a real index**, measured on `work/genome.fa`
/// at -t4: medians of 6 give 13.00 CPU-s at 16 and 12.835 at 32, -1.27%, and 32 wins all 6 rounds
/// (24 and 28 land between, 48 and 64 regress). The previous default of 16 was the knee on
/// `work/region.fa`, whose BWT is cache-resident: with no DRAM latency to hide, extra slots are pure
/// overhead and the knee necessarily lands low, which is why that measurement did not transfer.
///
/// **aarch64 only.** It is a memory-level-parallelism knob, so the knee is a property of the machine
/// rather than of the code, and 32 was measured on an M4 Max. x86 keeps 16 because no x86 machine
/// has measured this: the only x86 instrument available here is Rosetta, which is 2.4x slower than
/// native and runs on this same M4 memory system, so it cannot answer a memory-parallelism question
/// (it put 16 ahead by 0.5%, one round out of three, which is noise from a translated binary).
/// Raising x86 to 32 on the strength of an arm64 measurement would be exactly the kind of projection
/// this file exists to avoid. `BWA4_LOCKSTEP_N` re-sweeps it, and the CI x86 job can settle it.
///
/// Changing it changes only speed and peak scratch memory (each slot owns a `prev` buffer sized to
/// the batch's longest read), never the SMEMs produced. See [`lockstep_width`].
#[cfg(target_arch = "aarch64")]
const DEFAULT_LOCKSTEP_WIDTH: usize = 32;
/// See the aarch64 arm above: 16 is the value x86 has always run, and no x86 measurement exists to
/// move it.
#[cfg(not(target_arch = "aarch64"))]
const DEFAULT_LOCKSTEP_WIDTH: usize = 16;

/// How many of a backward round's live candidates get their checkpoint blocks prefetched before the
/// round extends any of them.
///
/// The first loop of a round usually stops within the first few candidates (it breaks on the first
/// survivor with a new occurrence count, or on the first that dies), so prefetching the whole array
/// would mostly issue loads for candidates the round never reaches. Eight covers the common case
/// and costs eight instructions when it does not. Scheduling only: prefetch moves cache lines, it
/// cannot change which intervals are produced.
const PREFETCH_AHEAD_DEFAULT: usize = 8;

/// [`PREFETCH_AHEAD_DEFAULT`], overridable by `BWA4_SEED_PREFETCH` so the depth can be swept, and
/// set to `0` to turn the batch prefetch off entirely for an A/B.
fn prefetch_ahead() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("BWA4_SEED_PREFETCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(PREFETCH_AHEAD_DEFAULT)
    })
}

/// Hint the next read's bases into the cache when a slot takes the one before it. They were
/// encoded with the whole batch and are cold by the time a walk reaches them, and its first step
/// loads them (on EPYC 9V45 that load was 6.8% of `mem_collect_smem_batched`). The first 256 bases
/// and the last; a longer read's walk streams on from there. No effect on any result.
#[inline(always)]
fn prefetch_read(r: Option<&[u8]>) {
    let Some(r) = r else { return };
    let last = r.len().saturating_sub(1);
    for off in (0..r.len().min(256))
        .step_by(64)
        .chain(std::iter::once(last))
    {
        let p = r.as_ptr().wrapping_add(off);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a prefetch never faults, whatever the address.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        unsafe {
            std::arch::x86_64::_mm_prefetch(p.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = p;
    }
}

/// The C's rounding fudge in `split_len = (int)(min_seed_len * split_factor + .499)`
/// (`bwamem.cpp:630`). Truncation of `x + 0.499` is round-half-down; `0.5` would be round-half-up
/// and would change `split_len` on exact halves, so the digits are copied verbatim and the `f32`
/// arithmetic must stay `f32`.
const SPLIT_LEN_ROUNDING: f32 = 0.499;

/// Measure this machine's memory-level parallelism and return the lockstep width to use.
///
/// # Why this is measured rather than hard-coded
///
/// The lockstep width should equal the number of independent cache-line fetches one core can keep
/// in flight, and that number is not a property of the architecture. Published pointer-chase
/// measurements on current server CPUs (Lemire, 2026): **Graviton 5 sustains 19 concurrent
/// fetches, Apple M4 28, Intel Granite Rapids 30, AMD Zen 5 Turin 58**, and AMD's own progression
/// runs Naples 15 -> Milan 22 -> Turin 58. A constant cannot track a factor of three between
/// contemporary machines, and this file previously carried two constants that were each measured on
/// exactly one host.
///
/// So the width is measured here, in about 15 ms at startup, by the same experiment those numbers
/// come from: chase `k` independent random cycles at once and see where adding lanes stops paying.
///
/// # Method
///
/// A pointer cycle over a buffer far larger than any cache, walked with `k` independent chains.
/// Time per access falls as `k` rises while the machine still has miss slots, and flattens once it
/// does not. The knee is taken as the smallest `k` within 5% of the best time, so a flat top picks
/// the cheaper width rather than the widest one.
///
/// Scheduling only, and it cannot move a byte of output: the value chooses how many FM walks are
/// interleaved, never which rows any walk visits.
fn probe_lockstep_width(fm: &FmIndex) -> usize {
    /// Candidate widths, spanning the published range from Graviton 5 (19) to Zen 5 Turin (58) with
    /// headroom at both ends.
    const CANDIDATES: [usize; 7] = [8, 12, 16, 24, 32, 48, 64];
    /// Accesses per candidate. Equal across widths, so no candidate is measured warmer than another.
    const ACCESSES: usize = 200_000;

    // Empty means the index is too small to measure (see `probe_concurrency`); the `find` below
    // then yields nothing and the default width is used, which is exactly the intended fallback.
    let times = fm.probe_concurrency(&CANDIDATES, ACCESSES);
    let best = times.iter().cloned().fold(f64::MAX, f64::min);
    // Smallest width within 5% of the best: on a flat top, fewer slots means less scratch memory and
    // less bookkeeping for the same latency hiding. Halved, because a slot step prefetches two
    // blocks (`k` and `k + s`), so a slot holds two of the chase's lines in flight: on an M4 the
    // chase picks 64 (sometimes 32) and the walk is 2% slower in cycles at 64 than at 16 or 32 (2M
    // GIAB reads, -t8). The fork's tuned widths sit at the same half: 16 on M4 (chase knee ~28) and
    // 32 on Zen 5 (~58).
    //
    // Except on Zen 5, which runs a slot's step fast enough that half the knee no longer covers
    // the latency: EPYC 9V45's chase knee is 12 on CI's index (as on Zen 4's 9V74), and same-host
    // A/B against the halved 8 gave 12 -2.7% (SMEM -4%), 16 -1.3%, 32 -0.9%, where Zen 3 and 4
    // lose at wider widths. The fork also widens on Zen 5 only (`bwa3_cpu_is_amd_zen5`).
    let zen5 = amd_zen5();
    let width = CANDIDATES
        .iter()
        .zip(&times)
        .find(|(_, &ns)| ns <= best * 1.05)
        .map(|(k, _)| {
            if zen5 {
                *k
            } else {
                (*k / 2).max(CANDIDATES[0])
            }
        })
        .unwrap_or(DEFAULT_LOCKSTEP_WIDTH);

    if std::env::var_os("BWA4_LOCKSTEP_PROBE").is_some() {
        eprintln!("[lockstep-probe] ns/access by width, chasing this index's cp_occ:");
        for (k, ns) in CANDIDATES.iter().zip(&times) {
            eprintln!("[lockstep-probe]   {k:>3} -> {ns:6.2} ns");
        }
        eprintln!("[lockstep-probe] chosen width {width}");
    }
    width
}

/// Whether this host is an AMD Zen 5 (family 0x1A, the model ranges Linux's `bsp_init_amd`
/// assigns to Zen 5).
fn amd_zen5() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let (v, id) = (
            std::arch::x86_64::__cpuid(0),
            std::arch::x86_64::__cpuid(1).eax,
        );
        let amd = (v.ebx, v.edx, v.ecx) == (0x6874_7541, 0x6974_6E65, 0x444D_4163);
        let family = (id >> 8 & 0xF) + (id >> 20 & 0xFF);
        let model = (id >> 4 & 0xF) | (id >> 12 & 0xF0);
        amd && zen5_model(family, model)
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Family 0x1A models that are Zen 5: 0x00-0x2F, 0x40-0x4F, 0x60-0x7F, 0xD0-0xD7 (the rest of
/// the family is Zen 6).
#[cfg(any(target_arch = "x86_64", test))]
fn zen5_model(family: u32, model: u32) -> bool {
    family == 0x1A && matches!(model, 0x00..=0x2F | 0x40..=0x4F | 0x60..=0x7F | 0xD0..=0xD7)
}

/// Lockstep width: how many independent FM walks are kept in flight, so each slot's `cp_occ`
/// prefetch has a full cycle to land before the block is used. **This is the aligner's
/// memory-level-parallelism knob**: N slots means at most N outstanding DRAM misses per core.
///
/// The default is 32 on aarch64, the knee measured on a real index, and 16 elsewhere (see
/// [`DEFAULT_LOCKSTEP_WIDTH`]).
/// `BWA4_LOCKSTEP_N` re-sweeps it on another machine. Scheduling only: every slot walks its own
/// read deterministically, so N cannot change a result.
///
/// # Returns
/// The number of lockstep slots, always `>= 1`. Read once from the environment and cached for the
/// process, so all three batched rounds agree and the value cannot change mid-run.
fn lockstep_width(fm: &FmIndex) -> usize {
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
            // Measured, not assumed: see `probe_lockstep_width`. `BWA4_LOCKSTEP_NO_PROBE` falls back
            // to the compiled-in constant, for a machine too noisy to measure on.
            .unwrap_or_else(|| {
                if std::env::var_os("BWA4_LOCKSTEP_NO_PROBE").is_some() {
                    DEFAULT_LOCKSTEP_WIDTH
                } else {
                    probe_lockstep_width(fm)
                }
            })
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

/// Round 3's lockstep width: at least [`BSD_LOCKSTEP_MIN`], whatever rounds 1 and 2 use. Its walks
/// are forward-only and shallow, so they want more in flight (fg-labs/bwa-mem3 also keeps its
/// bwtSeed width apart: 24, 32 on AMD Zen 5). Same-host A/B against the shared width (8 on these
/// hosts): Neoverse N2 -0.55% / -0.8% cycles; `mem_collect_smem_batched`, where round 3 inlines,
/// -100 (EPYC 9V74), -136 (Xeon 8573C) samples; M4 already probes 32.
fn bsd_lockstep_width(fm: &FmIndex) -> usize {
    lockstep_width(fm).max(BSD_LOCKSTEP_MIN)
}

/// Round 3's width floor (see [`bsd_lockstep_width`]).
const BSD_LOCKSTEP_MIN: usize = 32;

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
    collect_smems_batched_bsd(fm, reads, min_seed_len, min_intv, None).0
}

/// [`collect_smems_batched`], also recording each read's round-3 first walk when `bsd` is
/// `Some((max_intv, min_len))` (round 3's arguments): see [`FirstWalk`].
#[cfg_attr(bwa4_wide_split, inline(always))]
fn collect_smems_batched_bsd(
    fm: &FmIndex,
    reads: &[&[u8]],
    min_seed_len: i32,
    min_intv: i64,
    bsd: Option<(i64, i32)>,
) -> (Vec<Vec<Smem>>, Vec<FirstWalk>) {
    // Lockstep width: independent walks kept in flight so each slot's prefetch has a full cycle to
    // land before the block is used. 16 measured ~2.8% faster than 8 on a genome-scale index (M4 Max,
    // SE); 24 ties it and 32 regresses, so 16 is the knee. Shared by all three batched seeding rounds.
    let lockstep_slots = lockstep_width(fm);

    // The index's C array, hoisted once: `counts[b]` is the number of reference bases strictly
    // smaller than `b`, so it is also the first BWT ROW whose suffix begins with `b`. Copied out of
    // `fm` so the hot slot loop does not re-borrow it per step.
    let counts = fm.counts();
    // Per-read result slots, pre-sized so a finished walk can be moved straight into `output[ridx]`
    // regardless of the order in which reads complete.
    let mut output: Vec<Vec<Smem>> = (0..reads.len()).map(|_| Vec::new()).collect();
    let mut walks = vec![FirstWalk::NONE; if bsd.is_some() { reads.len() } else { 0 }];
    if reads.is_empty() {
        return (output, walks);
    }
    // Armed only on round 1's x = 0 walk at min_intv 1, which round 3's first walk is a prefix of.
    let bsd = bsd.filter(|&(mi, ml)| min_intv == 1 && mi >= 1 && ml >= 1);
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
            let mut slot = LsSlot::new(next_read, 0, min_intv, false, max_len);
            slot.bsd_cfg = bsd;
            slots.push(Some(slot));
            next_read += 1;
            prefetch_read(reads.get(next_read).copied());
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
        // Rust: `.iter_mut()` walks the slot table yielding an exclusive borrow of each entry, so
        // each in-flight read can be advanced in place. `.as_mut()` on the `Option` turns "maybe a
        // slot" into "maybe an exclusive borrow of a slot" without moving anything out, and the
        // let-else skips the empty seats.
        for slot_opt in slots.iter_mut() {
            let Some(slot) = slot_opt.as_mut() else {
                continue;
            };
            slot.step(fm, reads[slot.ridx], min_seed_len, &counts);
            if slot.phase == LsPhase::Done {
                // Copied out with room for rounds 2 and 3, which append in place: on GIAB a read
                // leaves round 1 with 2.5 SMEMs and ends with 12.8, and handing over the walk's own
                // buffer made both rounds regrow it (a realloc and a copy each). The slot keeps its
                // buffer for the next read.
                let mut v = Vec::with_capacity(slot.out.len() + SMEM_HEADROOM);
                v.extend_from_slice(&slot.out);
                slot.out.clear();
                output[slot.ridx] = v;
                if bsd.is_some() {
                    walks[slot.ridx] = slot.bsd;
                }
                if next_read < reads.len() {
                    slot.reset_to(next_read, 0, min_intv, false);
                    next_read += 1;
                    prefetch_read(reads.get(next_read).copied());
                } else {
                    *slot_opt = None;
                    live -= 1;
                }
            }
        }
    }
    (output, walks)
}

/// Round 3's first walk on a read (from x = 0), recorded by round 1's x = 0 forward walk (the
/// fork's `BsdFirstWalk`). Both step every base from read base 0 with the same forward extension,
/// so round 3's walk is a prefix of round 1's (min_intv 1, which goes on while s >= 1). Round 1
/// notes the step where round 3 would stop (`s < max_intv` at length `>= min_len`), or where it
/// would end when round 1 stops first, and round 3 emits that seed and starts at `x` instead.
#[derive(Clone, Copy)]
struct FirstWalk {
    /// The walk's last interval, emitted as its seed (m = 0, n) when s > 0. After a unique match's
    /// text compare `l` holds that match's `sa_tag` (m = 0, so it is `SA[k]`), not a row.
    k: i64,
    l: i64,
    s: i64,
    n: u32,
    /// Where round 3 goes on; < 0: not recorded.
    x: i32,
}

impl FirstWalk {
    const NONE: FirstWalk = FirstWalk {
        k: 0,
        l: 0,
        s: 0,
        n: 0,
        x: -1,
    };
}

/// `BWA4_SEED_STATS=1`: how the seeding FM traffic is distributed, so a proposed cache can be sized
/// before it is written.
///
/// Seeding is the largest single CPU item in the aligner and it is DRAM-latency bound. The obvious
/// lever is a k-mer table that replaces the first few FM steps of each SMEM start with one lookup
/// into a small array. Whether that is worth building depends on a ratio nothing else reports: how
/// many extension steps happen right after a start, against how many happen in total.
pub mod seed_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// SMEM starts, i.e. entries into the `Start` phase that actually set up an interval.
    pub static STARTS: AtomicU64 = AtomicU64::new(0);
    /// Forward-extension steps, one `backward_ext` each.
    pub static FWD_STEPS: AtomicU64 = AtomicU64::new(0);
    /// Backward-search steps, one `backward_ext` per live candidate.
    pub static BWD_STEPS: AtomicU64 = AtomicU64::new(0);
    /// Forward steps that fall within the first `K` bases after their start, for K = 10. These are
    /// exactly the steps a depth-10 prefix table could answer without touching the index.
    pub static FWD_FIRST10: AtomicU64 = AtomicU64::new(0);

    /// Whether the probe is on; read once.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_SEED_STATS").is_some())
    }

    /// Print the counters once, after the workers have joined. No-op unless enabled.
    pub fn dump() {
        if !enabled() {
            return;
        }
        let (st, fwd, bwd, f10) = (
            STARTS.load(Ordering::Relaxed),
            FWD_STEPS.load(Ordering::Relaxed),
            BWD_STEPS.load(Ordering::Relaxed),
            FWD_FIRST10.load(Ordering::Relaxed),
        );
        let total = fwd + bwd;
        eprintln!(
            "[seed-stats] {st} starts, {fwd} fwd steps, {bwd} bwd steps ({total} extends, \
             {:.1} per start); first-10-after-start = {f10} ({:.1}% of all extends)",
            total as f64 / st.max(1) as f64,
            100.0 * f10 as f64 / total.max(1) as f64,
        );
    }
}

/// Where a slot's walk currently sits inside [`smems_from_pos`]. One `step` call advances exactly one
/// phase iteration, so the FM operations of different reads interleave.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LsPhase {
    /// Set up the single-base interval at `x` (or skip an `N`).
    Start,
    /// One forward-extension iteration: append `codes[j]` on the right.
    Fwd,
    /// The first forward iterations of a walk, read from the seed jump table (see the arm).
    FwdTable,
    /// A unique forward match: LF-walking its row to a sampled SA row, then comparing the
    /// rest of the read with the text (see the `Fwd` arm).
    S1Lf,
    S1Cmp,
    /// One backward-search outer iteration: prepend `codes[jj]` to every live candidate.
    Bwd,
    /// Emit the boundary SMEM and advance `x` to `next_x`.
    PosDone,
    /// Walk complete; the driver harvests `out` and refills the slot.
    Done,
}

/// One read's round-1 SMEM walk as a resumable state machine (see [`collect_smems_batched`]). Mirrors
/// [`smems_from_pos`] step by step: `Start` seeds the single-base interval at `x`, `Fwd` runs one
/// forward-extension iteration, `bwd_init` does the fwd->bwd housekeeping (append + reverse), `Bwd`
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
    /// READ offsets. Live only between `Start` and `bwd_init`; the backward phase works out of `prev`.
    smem: Smem,
    /// Number of live candidates in `prev`, i.e. the valid prefix is `prev[..num_prev]`. Same
    /// variable as the C's `numPrev`. Grows during `Fwd`/`bwd_init`, only shrinks during `Bwd`.
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
    /// The unique match's SA walk: current row and LF steps taken; then its text position.
    lf_row: i64,
    lf_off: i64,
    tpos: i64,
    /// Forward steps the seed jump table serves for this walk (`read[x .. x + jdp)` is ACGT and
    /// `jdp <= fm.seed_jump_depth()`), and those bases' 2-bit code; 0 once taken.
    jdp: usize,
    jcode: u32,
    /// Round 3's `(max_intv, min_len)` when this driver records [`FirstWalk`]s, else `None`.
    bsd_cfg: Option<(i64, i32)>,
    /// The forward step at which round 3's walk may stop (`min_len - 1`) while a record is
    /// pending, `usize::MAX` otherwise; `bsd_intv` is round 3's `max_intv`.
    bsd_j: usize,
    bsd_intv: i64,
    bsd: FirstWalk,
    prev: Vec<Smem>,
    /// SMEMs emitted by the current walk, moved out when the slot reaches `Done`.
    out: Vec<Smem>,
}

/// A forward walk at a unique match hands off to the text compare only with at least this many
/// read bases left: resolving the match's text position costs an LF walk to a sampled SA row
/// (about 4 steps at 1 sample in 8 rows) and a pac read.
const S1_MIN_REST: usize = 16;

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
            lf_row: 0,
            lf_off: 0,
            tpos: 0,
            jdp: 0,
            jcode: 0,
            bsd_cfg: None,
            bsd_j: usize::MAX,
            bsd_intv: 0,
            bsd: FirstWalk::NONE,
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
        self.bsd_j = usize::MAX;
        self.bsd = FirstWalk::NONE;
    }

    /// Forward step `j` reached `e` (`read[0..=j]`), with `j >= bsd_j`: round 3 stops here when
    /// `e.s < max_intv`, emitting `e` when `e.s > 0`, and goes on at `j + 1`.
    #[inline]
    fn bsd_note(&mut self, j: usize, e: &Smem) {
        if e.s < self.bsd_intv {
            self.bsd = FirstWalk {
                k: e.k,
                l: e.l,
                s: e.s,
                n: j as u32,
                x: (j + 1) as i32,
            };
            self.bsd_j = usize::MAX;
        }
    }

    /// Round 1's walk stopped before round 3's at read base `p`: the read's end, an `N`, or a
    /// base the interval does not extend with (s = 0 from step `p` on). Round 3 then steps with
    /// s = 0, ending at the first `N` (x = p + 1), at the first step `>= bsd_j` (no seed,
    /// x = p + 1), or at the read's end.
    #[cold]
    fn bsd_resolve(&mut self, codes: &[u8], mut p: usize) {
        let rl = codes.len();
        while p < rl && codes[p] < AMBIGUOUS_BASE && p < self.bsd_j {
            p += 1;
        }
        self.bsd = FirstWalk {
            x: if p < rl { p + 1 } else { rl } as i32,
            ..FirstWalk::NONE
        };
        self.bsd_j = usize::MAX;
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
    /// End of the forward walk (the fork's `ls_prepare_backward`): push the last interval, put the
    /// candidates longest-first and enter `Bwd`. Returns whether the first backward round reads
    /// the index, in which case the caller yields so the hints issued here land first.
    #[inline(always)]
    fn bwd_init(&mut self, fm: &FmIndex, min_intv: i64) -> bool {
        if self.smem.s >= min_intv {
            self.prev[self.num_prev] = self.smem;
            self.num_prev += 1;
            // The forward steps hinted every other candidate's blocks when they pushed it; this
            // last one is hinted only now (a hint cannot change a result).
            fm.prefetch_occ(self.smem.k, self.smem.k + self.smem.s);
        }
        self.prev[..self.num_prev].reverse();
        self.jj = self.x as i64 - 1;
        self.phase = LsPhase::Bwd;
        self.jj >= 0 && self.num_prev != 0
    }

    #[inline(always)]
    fn step(&mut self, fm: &FmIndex, codes: &[u8], min_seed_len: i32, counts: &[i64; 5]) {
        // Read once per call rather than per extension, and compiled out entirely unless the
        // `seed-stats` feature is on: this sits on a path that runs 1.7 billion times per 500k
        // pairs, where even a predictable branch is measurable (it cost about 3% of CPU when it
        // was unconditional).
        #[cfg(feature = "seed-stats")]
        let stats_on = seed_stats::enabled();
        // Per-walk occurrence floor: round 1 shares one value, round 2 carries the parent's `s + 1`.
        let min_intv = self.min_intv;
        // Length of this read in bases; the exclusive upper bound of every READ offset below.
        let readlength = codes.len();
        // Phases that touch no index memory run on at once; the others return after issuing
        // their prefetches, so the other slots' loads overlap this one's.
        'walk: loop {
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
                        m: self.x as u32,
                        n: self.x as u32,
                        k: counts[a],
                        l: counts[MAX_BASE - a],
                        s: counts[a + 1] - counts[a],
                    };
                    self.num_prev = 0;
                    self.j = self.x + 1;
                    self.phase = LsPhase::Fwd;
                    if let (0, false, Some((mi, ml))) = (self.x, self.single_pos, self.bsd_cfg) {
                        self.bsd_j = ml as usize - 1;
                        self.bsd_intv = mi;
                    }
                    #[cfg(feature = "seed-stats")]
                    if stats_on {
                        seed_stats::STARTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    // The first forward steps from the table (see `FwdTable`): as many as the
                    // table is deep and the read stays ACGT. Their lines are prefetched now and
                    // read on the next visit, so the other slots' loads overlap these.
                    let lim = fm.seed_jump_depth().min(readlength - self.x);
                    let mut jdp = 1;
                    let mut code = a as u32;
                    while jdp < lim && codes[self.x + jdp] < AMBIGUOUS_BASE {
                        code = code << 2 | u32::from(codes[self.x + jdp]);
                        jdp += 1;
                    }
                    if jdp >= 2 {
                        self.jdp = jdp;
                        self.jcode = code;
                        for d in 2..=jdp {
                            fm.prefetch_seed_jump(d, code >> (2 * (jdp - d)));
                        }
                        self.phase = LsPhase::FwdTable;
                        return;
                    }
                }
                LsPhase::FwdTable => {
                    // The forward steps to depths `2..=jdp`, each with the bookkeeping of a `Fwd`
                    // step and the interval read from the table instead of the FM index (the
                    // table holds what that step would compute; fg-labs/bwa-mem3's
                    // `ls_advance_forward_table`). Every base is ACGT and inside the read, so only
                    // the `s < min_intv` exit and the unique-match hand-off can end the run early.
                    let (jdp, code) = (self.jdp, self.jcode);
                    self.jdp = 0;
                    // `self.smem` through the steps, stored back at each exit: kept in the slot,
                    // each step read the interval whole just after the last one stored it field by
                    // field, a failed store forward per step on x86.
                    let mut cur = self.smem;
                    for d in 2..=jdp {
                        let e = fm.seed_jump(d, code >> (2 * (jdp - d)));
                        let j = self.x + d - 1;
                        self.next_x = j + 1;
                        let new_smem = Smem {
                            m: self.x as u32,
                            n: j as u32,
                            k: e.k,
                            l: e.l,
                            s: e.s,
                        };
                        self.prev[self.num_prev] = cur;
                        if new_smem.s != cur.s {
                            self.num_prev += 1;
                            // The first backward round reads this candidate's blocks: hinted now, visits ahead.
                            fm.prefetch_occ(cur.k, cur.k + cur.s);
                        }
                        if new_smem.s < min_intv {
                            self.smem = cur;
                            if self.bsd_j != usize::MAX {
                                self.bsd_resolve(codes, j);
                            }
                            self.next_x = j;
                            if self.bwd_init(fm, min_intv) {
                                return;
                            }
                            // On to `Bwd`, not the table's next depth: the walk has ended.
                            continue 'walk;
                        }
                        if j >= self.bsd_j {
                            self.bsd_note(j, &new_smem);
                        }
                        cur = new_smem;
                        self.j = j + 1;
                        if new_smem.s == 1 && readlength - self.j >= S1_MIN_REST {
                            self.smem = cur;
                            self.lf_row = new_smem.k;
                            self.lf_off = 0;
                            self.phase = LsPhase::S1Lf;
                            fm.prefetch_sa_row(new_smem.k);
                            return;
                        }
                    }
                    self.smem = cur;
                    self.phase = LsPhase::Fwd;
                    fm.prefetch_occ(cur.l, cur.l + cur.s);
                    return;
                }
                LsPhase::Fwd => {
                    if self.j >= readlength {
                        if self.bsd_j != usize::MAX {
                            self.bsd_resolve(codes, self.j);
                        }
                        if self.bwd_init(fm, min_intv) {
                            return;
                        }
                        continue;
                    }
                    // Base code at the READ offset being appended on the right.
                    let aj = codes[self.j];
                    self.next_x = self.j + 1;
                    if aj >= AMBIGUOUS_BASE {
                        if self.bsd_j != usize::MAX {
                            self.bsd_resolve(codes, self.j);
                        }
                        if self.bwd_init(fm, min_intv) {
                            return;
                        }
                        continue;
                    }
                    // Forward extension = backward extension on the reverse complement (swap k/l,
                    // prepend the complement base `3 - aj`, swap back; see the module docs), as
                    // `forward_ext` computes it.
                    #[cfg(feature = "seed-stats")]
                    if stats_on {
                        use std::sync::atomic::Ordering::Relaxed;
                        seed_stats::FWD_STEPS.fetch_add(1, Relaxed);
                        // `j - x` is how many bases have been appended since this start, so the first
                        // ten steps of a start are the ones a depth-10 prefix table could serve.
                        if self.j - self.x < 10 {
                            seed_stats::FWD_FIRST10.fetch_add(1, Relaxed);
                        }
                    }
                    // `new_smem`: the forward interval of the read substring `codes[x ..= j]`, one
                    // base longer on the RIGHT than `self.smem` (a unique match near the read's end
                    // steps here rather than in `S1Cmp`).
                    let mut new_smem = if self.smem.s == 1 {
                        fm.forward_ext_s1(self.smem, aj as usize)
                    } else {
                        fm.forward_ext(self.smem, aj as usize)
                    };
                    new_smem.n = self.j as u32;

                    self.prev[self.num_prev] = self.smem;
                    if new_smem.s != self.smem.s {
                        self.num_prev += 1;
                        // The first backward round reads this candidate's blocks: hinted now, visits ahead.
                        fm.prefetch_occ(self.smem.k, self.smem.k + self.smem.s);
                    }
                    if new_smem.s < min_intv {
                        if self.bsd_j != usize::MAX {
                            self.bsd_resolve(codes, self.j);
                        }
                        self.next_x = self.j;
                        if self.bwd_init(fm, min_intv) {
                            return;
                        }
                        continue;
                    }
                    if self.j >= self.bsd_j {
                        self.bsd_note(self.j, &new_smem);
                    }
                    self.smem = new_smem;
                    self.j += 1;
                    // A unique match (`s == 1`, so `min_intv <= 1`) keeps its row `k` on every further
                    // forward step: the one suffix it names is either extended or dropped. So instead
                    // of stepping the rest of the read a base at a time, resolve that suffix's text
                    // position from `k` and compare the read with the text (fg-labs/bwa-mem3).
                    if new_smem.s == 1 && readlength - self.j >= S1_MIN_REST {
                        self.lf_row = new_smem.k;
                        self.lf_off = 0;
                        self.phase = LsPhase::S1Lf;
                        fm.prefetch_sa_row(new_smem.k);
                        return;
                    }
                    // Next forward step swaps k/l, so its backward_ext reads the blocks at `new_smem.l`.
                    fm.prefetch_occ(new_smem.l, new_smem.l + new_smem.s);
                    return;
                }
                LsPhase::S1Lf => match fm.sa_step(self.lf_row, self.lf_off) {
                    Err(next) => {
                        self.lf_row = next;
                        self.lf_off += 1;
                        fm.prefetch_sa_row(next);
                        return;
                    }
                    Ok(v) => {
                        // `v = SA[k]`, where `read[m..=n]` starts in the text; read base `j` faces
                        // text base `v + (j - m)`.
                        self.tpos = v + (self.j as i64 - i64::from(self.smem.m));
                        // Keep the resolved position for the SA lookup of this match and of its
                        // backward extensions (`sa_tag`); `l` is dead from here on.
                        self.smem.l = bwa_index::fmindex::sa_tag(v - i64::from(self.smem.m));
                        self.phase = LsPhase::S1Cmp;
                        let tend = 2 * fm.l_pac();
                        if self.tpos < tend {
                            fm.prefetch_pac(
                                self.tpos,
                                (self.tpos + (readlength - self.j) as i64).min(tend),
                            );
                        }
                        return;
                    }
                },
                LsPhase::S1Cmp => {
                    // The exit the base-at-a-time walk would take at the first base that ends it: the
                    // read's end, a non-ACGT read base (`next_x = j + 1`), or a base the text does not
                    // continue with, the text's end included (`next_x = j`). Every exit keeps the
                    // match (`s = 1 >= min_intv`) with `n = j - 1`, which `bwd_init` records.
                    let j = fm.text_match(codes, self.j, self.tpos);
                    self.next_x = if j >= readlength {
                        readlength
                    } else if codes[j] >= AMBIGUOUS_BASE {
                        j + 1
                    } else {
                        j
                    };
                    // A pending round-3 walk had s = 1 on read bases `self.j .. j` and stops on the
                    // first of them at `bsd_j` when 1 < max_intv (no earlier step reached `bsd_j`,
                    // or `bsd_note` would have recorded it); otherwise it goes on from base `j`.
                    if self.bsd_j != usize::MAX {
                        if self.bsd_intv > 1 && self.bsd_j < j {
                            let n = self.bsd_j;
                            let e = Smem {
                                n: n as u32,
                                ..self.smem
                            };
                            self.bsd_note(n, &e);
                        } else {
                            self.bsd_resolve(codes, j);
                        }
                    }
                    self.smem.n = (j - 1) as u32;
                    if self.bwd_init(fm, min_intv) {
                        return;
                    }
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
                    #[cfg(feature = "seed-stats")]
                    if stats_on {
                        seed_stats::BWD_STEPS
                            .fetch_add(self.num_prev as u64, std::sync::atomic::Ordering::Relaxed);
                    }
                    let num_curr = bwd_round_call(
                        fm,
                        &mut self.prev[..self.num_prev],
                        a as usize,
                        self.jj as u32,
                        min_intv,
                        i64::from(min_seed_len),
                        &mut self.out,
                    );
                    self.num_prev = num_curr;
                    self.jj -= 1;
                    if num_curr == 0 {
                        self.phase = LsPhase::PosDone;
                    }
                    return;
                }
                LsPhase::PosDone => {
                    if self.num_prev != 0 {
                        // Candidates are kept longest-first, so `prev[0]` is the longest survivor. The
                        // backward phase ended at the read start or at an `N` rather than because a
                        // candidate died, so this one is left-maximal by boundary: emit it if long enough.
                        let longest = self.prev[0];
                        if (i64::from(longest.n) - i64::from(longest.m) + 1)
                            >= i64::from(min_seed_len)
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
                LsPhase::Done => return,
            }
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
    // As in the lockstep driver: read once per call, and compiled out without the feature.
    #[cfg(feature = "seed-stats")]
    let stats_on = seed_stats::enabled();
    // As above: hoisted out of the round loop.
    let pf_ahead = prefetch_ahead();
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
        // Same batch prefetch as the lockstep driver; see there for why it cannot change a result.
        for c in prev[..num_prev.min(pf_ahead)].iter() {
            fm.prefetch_occ(c.k, c.k + c.s);
        }
        let mut p = 0usize;
        while p < num_prev {
            // One live candidate: the interval of `codes[jj + 1 ..= candidate.n]`.
            let candidate = prev[p];
            // Plain backward_ext here: no k/l swap, because this really is a left extension.
            // `new_smem` is the interval of `codes[jj ..= candidate.n]`, one base longer on the LEFT.
            #[cfg(feature = "seed-stats")]
            if stats_on {
                seed_stats::BWD_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let mut new_smem = fm.backward_ext_konly(candidate, a);
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
            #[cfg(feature = "seed-stats")]
            if stats_on {
                seed_stats::BWD_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let mut new_smem = fm.backward_ext_konly(candidate, a);
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
    /// Read the planned seed jump entry (see the `Start` arm) and resume `Fwd` past it.
    Jump,
    /// Read the seed jump entry of the walk's last `jd` bases up to `min_seed_len` (see the `Start`
    /// arm) and go on to `Bwd`.
    JumpB,
    /// One k-only backward step towards `x` over the walk's first `min_seed_len` bases.
    Bwd,
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
/// `cp_occ` DRAM latency of the forward extension. The same SMEMs as [`bwt_seed_strategy`] (up to
/// `l`): `Start` seeds the single-base interval at `x`, `Fwd` runs one forward-extension iteration,
/// and `JumpB` / `Bwd` find a walk's interval at its `min_seed_len`-th base right to left.
struct BwtSeedSlot {
    /// Index of the read this slot is currently walking, within the caller's `reads` slice.
    ridx: usize,
    /// The starting READ offset of the current forward walk, `0 ..< readlength`.
    x: usize,
    /// Where the outer sweep resumes after this position: a READ offset, `x + 1` at minimum and up
    /// to one past wherever the forward walk stopped.
    next_x: usize,
    /// Forward cursor: the READ offset being appended on the right. Runs `x + 1 ..= readlength`.
    /// In `Bwd`, the leftmost READ offset of `smem`, moving down to `x`.
    j: usize,
    /// The interval being extended: `k`/`l` are BWT ROWS, `s` the occurrence count, `m`/`n` the
    /// inclusive READ span. Valid only while `phase` is `Fwd`, or `Bwd` (`l` unset there).
    smem: Smem,
    /// Which phase the next `step` call executes. `Done` tells the driver to harvest and refill.
    phase: BwtPhase,
    /// The planned seed jump (see `Jump`, `JumpB`): its length and the 2-bit code of its bases.
    jd: usize,
    jcode: u32,
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
            jd: 0,
            jcode: 0,
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

    /// Take round 1's record of this read's first walk: emit its seed and go on where it ended.
    fn take_first_walk(&mut self, walks: &[FirstWalk]) {
        if let Some(r) = walks.get(self.ridx).filter(|r| r.x > 0) {
            if r.s > 0 {
                self.out.push(Smem {
                    m: 0,
                    n: r.n,
                    k: r.k,
                    l: r.l,
                    s: r.s,
                });
            }
            self.x = r.x as usize;
        }
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
                // A walk emits only at a step `min_seed_len` long, which must end inside the read,
                // so no walk from here on emits (the fork's `bwtseed_x_end`).
                if self.x + (min_seed_len.max(1) as usize) > readlength {
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
                    m: self.x as u32,
                    n: self.x as u32,
                    k: counts[a],
                    l: counts[MAX_BASE - a],
                    s: counts[a + 1] - counts[a],
                };
                self.j = self.x + 1;
                self.phase = BwtPhase::Fwd;
                // Nothing below can end a walk before it is `min_seed_len` long, so its first
                // `jd` bases come from the seed jump table in one entry when they are all ACGT
                // (the fork's `bsd_plan_jump`); the entry is prefetched here and read by `Jump`.
                let msl = min_seed_len.max(1) as usize;
                let jd = fm.seed_jump_depth().min(msl - 1);
                // Most walks end at their `msl`-th base, where the count first falls under
                // `max_intv`. So an N-free first `msl` bases are searched right to left instead:
                // the last `jd` from the jump table, the rest by k-only backward steps, half a
                // forward step's work (`l`, which only a forward step needs, is not computed; the
                // emitted SMEM's is 0, and nothing downstream reads it). Only a count still at
                // `max_intv` there sends the walk forward, from the start. `x + msl <= readlength`
                // (checked above).
                if jd >= 2
                    && codes[self.x + 1..self.x + msl]
                        .iter()
                        .all(|&b| b < AMBIGUOUS_BASE)
                {
                    self.jd = jd;
                    self.jcode = codes[self.x + msl - jd..self.x + msl]
                        .iter()
                        .fold(0u32, |c, &b| c << 2 | u32::from(b));
                    fm.prefetch_seed_jump(jd, self.jcode);
                    self.phase = BwtPhase::JumpB;
                    return;
                }
                if jd >= 2
                    && self.x + jd <= readlength
                    && codes[self.x + 1..self.x + jd]
                        .iter()
                        .all(|&b| b < AMBIGUOUS_BASE)
                {
                    self.jd = jd;
                    self.jcode = codes[self.x..self.x + jd]
                        .iter()
                        .fold(0u32, |c, &b| c << 2 | u32::from(b));
                    fm.prefetch_seed_jump(jd, self.jcode);
                    self.phase = BwtPhase::Jump;
                }
            }
            BwtPhase::Jump => {
                // The walk's state after its first `jd - 1` forward steps, `next_x` included.
                let e = fm.seed_jump(self.jd, self.jcode);
                self.smem = Smem {
                    m: self.x as u32,
                    n: (self.x + self.jd - 1) as u32,
                    k: e.k,
                    l: e.l,
                    s: e.s,
                };
                self.j = self.x + self.jd;
                self.next_x = self.j;
                self.phase = BwtPhase::Fwd;
                fm.prefetch_occ(self.smem.l, self.smem.l + self.smem.s);
            }
            BwtPhase::JumpB => {
                // The interval of `read[j .. x + msl)`, `j = x + msl - jd`.
                let e = fm.seed_jump(self.jd, self.jcode);
                let msl = min_seed_len.max(1) as usize;
                self.j = self.x + msl - self.jd;
                self.smem = Smem {
                    m: self.j as u32,
                    n: (self.x + msl - 1) as u32,
                    k: e.k,
                    l: 0,
                    s: e.s,
                };
                self.next_x = self.x + msl;
                self.phase = BwtPhase::Bwd;
                fm.prefetch_occ(e.k, e.k + e.s);
            }
            BwtPhase::Bwd => {
                // An empty interval stays empty: the walk ends at its `msl`-th base, unemitted.
                if self.smem.s == 0 {
                    self.end_pos(readlength);
                    return;
                }
                self.j -= 1;
                self.smem = fm.backward_ext_konly(self.smem, codes[self.j] as usize);
                self.smem.m = self.j as u32;
                if self.j > self.x {
                    fm.prefetch_occ(self.smem.k, self.smem.k + self.smem.s);
                    return;
                }
                // The interval of the walk's first `msl` bases, where the forward walk first
                // tests its count.
                if self.smem.s < max_intv {
                    if self.smem.s > 0 {
                        self.out.push(self.smem);
                    }
                    self.end_pos(readlength);
                    return;
                }
                // Still `max_intv` rows or more: the forward walk, from its first `jd` bases.
                self.jcode = codes[self.x..self.x + self.jd]
                    .iter()
                    .fold(0u32, |c, &b| c << 2 | u32::from(b));
                fm.prefetch_seed_jump(self.jd, self.jcode);
                self.phase = BwtPhase::Jump;
            }
            BwtPhase::Fwd => {
                if self.smem.s == 0 {
                    // An empty interval stays empty and is never emitted, so where the walk ends
                    // depends on the read alone: at its end, at its first `N`, or at the step
                    // where it is `min_seed_len` long. Found here without the FM steps (14% of
                    // round 3's on chr21). `next_x == j` on entry to `Fwd`.
                    let last = self.x + min_seed_len.max(1) as usize - 1;
                    let mut j = self.j;
                    self.next_x = loop {
                        if j >= readlength {
                            break j;
                        }
                        if codes[j] >= AMBIGUOUS_BASE || j >= last {
                            break j + 1;
                        }
                        j += 1;
                    };
                    self.end_pos(readlength);
                    return;
                }
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
                // Interval of `codes[x ..= j]`, one base longer on the RIGHT than before; a third
                // of round 3's steps start from one row (chr21).
                let mut new_smem = if self.smem.s == 1 {
                    fm.forward_ext_s1(self.smem, aj as usize)
                } else {
                    fm.forward_ext(self.smem, aj as usize)
                };
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
#[cfg_attr(bwa4_wide_split, inline(always))]
fn bwt_seed_strategy_batched(
    fm: &FmIndex,
    reads: &[&[u8]],
    max_intv: i64,
    min_seed_len: i32,
    walks: &[FirstWalk],
    out: &mut [Vec<Smem>],
) {
    let lockstep_slots = bsd_lockstep_width(fm);
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
            let mut slot = BwtSeedSlot::new(next_read);
            slot.take_first_walk(walks);
            slots.push(Some(slot));
            next_read += 1;
            prefetch_read(reads.get(next_read).copied());
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
                    slot.take_first_walk(walks);
                    next_read += 1;
                    prefetch_read(reads.get(next_read).copied());
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

/// Spare SMEM slots a read's round-1 vector is given for rounds 2 and 3 (see
/// `collect_smems_batched_bsd`).
const SMEM_HEADROOM: usize = 16;

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
    // A/B (`--cfg bwa4_wide_split`): one copy of the three rounds per index kind, so neither
    // carries the other's count code between its hot blocks.
    #[cfg(bwa4_wide_split)]
    return if fm.wide() {
        collect_rounds_w::<true>(fm, reads, opt)
    } else {
        collect_rounds_w::<false>(fm, reads, opt)
    };
    #[cfg(not(bwa4_wide_split))]
    collect_rounds(fm, reads, opt)
}

/// [`collect_rounds`] for an index whose [`FmIndex::wide`] is `W`.
#[cfg(bwa4_wide_split)]
#[inline(never)]
fn collect_rounds_w<const W: bool>(fm: &FmIndex, reads: &[&[u8]], opt: &MemOpt) -> Vec<Vec<Smem>> {
    // SAFETY: the only caller, `mem_collect_smem_batched`, picks `W` from `fm.wide()`.
    unsafe { std::hint::assert_unchecked(fm.wide() == W) };
    collect_rounds(fm, reads, opt)
}

/// One backward round of a slot: [`FmIndex::backward_round`], or under `--cfg bwa4_wide_split`
/// the copy for this index's kind (one test, folded inside [`collect_rounds_w`]).
#[inline(always)]
fn bwd_round_call(
    fm: &FmIndex,
    prev: &mut [Smem],
    a: usize,
    m: u32,
    min_intv: i64,
    min_seed_len: i64,
    out: &mut Vec<Smem>,
) -> usize {
    #[cfg(bwa4_wide_split)]
    return if fm.wide() {
        fm.backward_round_w::<true>(prev, a, m, min_intv, min_seed_len, out)
    } else {
        fm.backward_round_w::<false>(prev, a, m, min_intv, min_seed_len, out)
    };
    #[cfg(not(bwa4_wide_split))]
    fm.backward_round(prev, a, m, min_intv, min_seed_len, out)
}

/// The body of [`mem_collect_smem_batched`]: rounds 1, 2 and 3.
#[inline(always)]
fn collect_rounds(fm: &FmIndex, reads: &[&[u8]], opt: &MemOpt) -> Vec<Vec<Smem>> {
    // Round 1 in lockstep, `min_intv = 1` (accept any pattern that occurs at all). This vector is
    // then grown in place by rounds 2 and 3.
    // Round 1 also records round 3's first walk on each read, which round 3 then skips.
    let r3 = (opt.max_mem_intv > 0).then_some((opt.max_mem_intv, opt.min_seed_len + 1));
    let (mut per_read, walks) = collect_smems_batched_bsd(fm, reads, opt.min_seed_len, 1, r3);
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
            &walks,
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
            &[],
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
#[cfg_attr(bwa4_wide_split, inline(always))]
fn smem_round_2_batched(fm: &FmIndex, reads: &[&[u8]], opt: &MemOpt, per_read: &mut [Vec<Smem>]) {
    let lockstep_slots = lockstep_width(fm);
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
    // Every job's output in completion order, and where each job's lies in it: appended per read in
    // job order at the end. One buffer, where a vector per job was an allocation and a free per job.
    let mut done: Vec<Smem> = Vec::new();
    let mut range: Vec<(u32, u32)> = vec![(0, 0); jobs.len()];

    // Slots pair a job id (index into `jobs`/`range`) with the walk state, because jobs complete
    // out of order and the result must go back to the right `range` entry.
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
            prefetch_read(jobs.get(next_job).map(|j| reads[j.ridx]));
        } else {
            slots.push(None);
        }
    }

    // Number of `Some` slots. Invariant at the top of each pass: every live slot is mid-walk on the
    // job named by its `job_id`, jobs `next_job..` are unstarted, and `range[..]` names output only
    // for jobs already completed. A slot retires only when both are exhausted.
    let mut live = slots.iter().filter(|s| s.is_some()).count();
    while live > 0 {
        for slot_opt in slots.iter_mut() {
            let Some((job_id, slot)) = slot_opt.as_mut() else {
                continue;
            };
            slot.step(fm, reads[slot.ridx], opt.min_seed_len, &counts);
            if slot.phase == LsPhase::Done {
                range[*job_id] = (done.len() as u32, slot.out.len() as u32);
                // Moves the SMEMs and keeps the slot's capacity for its next job.
                done.append(&mut slot.out);
                if next_job < jobs.len() {
                    let j = &jobs[next_job];
                    *job_id = next_job;
                    slot.reset_to(j.ridx, j.x, j.min_intv, true);
                    next_job += 1;
                    prefetch_read(jobs.get(next_job).map(|j| reads[j.ridx]));
                } else {
                    *slot_opt = None;
                    live -= 1;
                }
            }
        }
    }

    // Append each job's SMEMs to its read, in job order (= per-read round-1-index order), each
    // read's vector grown once.
    let mut add = vec![0usize; per_read.len()];
    for (job, &(_, n)) in jobs.iter().zip(&range) {
        add[job.ridx] += n as usize;
    }
    for (v, &n) in per_read.iter_mut().zip(&add) {
        v.reserve(n);
    }
    for (job, &(at, n)) in jobs.iter().zip(&range) {
        per_read[job.ridx].extend_from_slice(&done[at as usize..][..n as usize]);
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
    use super::zen5_model;

    #[test]
    fn zen5_models() {
        // Turin, Turin dense, Strix Point: Zen 5. Genoa (family 0x19) and a Zen 6 model: not.
        assert!(zen5_model(0x1A, 0x02) && zen5_model(0x1A, 0x11) && zen5_model(0x1A, 0x24));
        assert!(!zen5_model(0x19, 0x11) && !zen5_model(0x1A, 0x50) && !zen5_model(0x1A, 0xD8));
    }

    use super::*;
    use std::path::Path;

    /// Seeding on a real index and real reads, for timing variants of the lockstep walk:
    /// `BWA4_SEED_BENCH_INDEX=<prefix> BWA4_SEED_BENCH_FQ=<fastq> [BWA4_SEED_BENCH_N=200000]
    /// cargo test --release -p bwa-mem4-seed smem_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn smem_bench() {
        let (Ok(idx), Ok(fq)) = (
            std::env::var("BWA4_SEED_BENCH_INDEX"),
            std::env::var("BWA4_SEED_BENCH_FQ"),
        ) else {
            return;
        };
        let n: usize = std::env::var("BWA4_SEED_BENCH_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200_000);
        let fm = FmIndex::load(Path::new(&idx)).unwrap();
        let text = std::fs::read_to_string(&fq).unwrap();
        let reads: Vec<Vec<u8>> = text
            .lines()
            .skip(1)
            .step_by(4)
            .take(n)
            .map(|l| {
                l.bytes()
                    .map(|b| match b {
                        b'A' | b'a' => 0,
                        b'C' | b'c' => 1,
                        b'G' | b'g' => 2,
                        b'T' | b't' => 3,
                        _ => 4,
                    })
                    .collect()
            })
            .collect();
        let opt = MemOpt::default();
        // `BWA4_SEED_BENCH_ROUNDS=1` or `2` stops after that round, so a hardware counter over the
        // whole process (`/usr/bin/time -l`) can be split by round.
        let rounds: u32 = std::env::var("BWA4_SEED_BENCH_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let mut total = 0usize;
        let (mut t1, mut t2, mut t3) = (0u128, 0u128, 0u128);
        let t0 = std::time::Instant::now();
        for chunk in reads.chunks(512) {
            let refs: Vec<&[u8]> = chunk.iter().map(|r| r.as_slice()).collect();
            #[cfg(feature = "seed-stats")]
            let snap = || {
                use std::sync::atomic::Ordering::Relaxed;
                (
                    seed_stats::STARTS.load(Relaxed),
                    seed_stats::FWD_STEPS.load(Relaxed),
                    seed_stats::BWD_STEPS.load(Relaxed),
                )
            };
            #[cfg(feature = "seed-stats")]
            let s0 = snap();
            let a = std::time::Instant::now();
            // As `mem_collect_smem_batched`: round 1 also records round 3's first walks.
            let r3 = (opt.max_mem_intv > 0).then_some((opt.max_mem_intv, opt.min_seed_len + 1));
            let (mut per_read, walks) =
                collect_smems_batched_bsd(&fm, &refs, opt.min_seed_len, 1, r3);
            t1 += a.elapsed().as_nanos();
            #[cfg(feature = "seed-stats")]
            let s1 = snap();
            if rounds < 2 {
                total += per_read.iter().map(|v| v.len()).sum::<usize>();
                continue;
            }
            let a = std::time::Instant::now();
            smem_round_2_batched(&fm, &refs, &opt, &mut per_read);
            t2 += a.elapsed().as_nanos();
            #[cfg(feature = "seed-stats")]
            {
                let s2 = snap();
                use std::sync::atomic::Ordering::Relaxed;
                R1.fetch_add(s1.0 - s0.0, Relaxed);
                R1F.fetch_add(s1.1 - s0.1, Relaxed);
                R1B.fetch_add(s1.2 - s0.2, Relaxed);
                R2.fetch_add(s2.0 - s1.0, Relaxed);
                R2F.fetch_add(s2.1 - s1.1, Relaxed);
                R2B.fetch_add(s2.2 - s1.2, Relaxed);
            }
            let a = std::time::Instant::now();
            if opt.max_mem_intv > 0 && rounds >= 3 {
                bwt_seed_strategy_batched(
                    &fm,
                    &refs,
                    opt.max_mem_intv,
                    opt.min_seed_len + 1,
                    &walks,
                    &mut per_read,
                );
            }
            t3 += a.elapsed().as_nanos();
            total += per_read.iter().map(|v| v.len()).sum::<usize>();
        }
        let ns = t0.elapsed().as_nanos() as f64;
        let n = reads.len() as f64;
        eprintln!(
            "SEED bench: {} reads, {total} smems, {:.0} ns/read (round 1 {:.0}, round 2 {:.0}, round 3 {:.0}), {:.2} s",
            reads.len(),
            ns / n,
            t1 as f64 / n,
            t2 as f64 / n,
            t3 as f64 / n,
            ns / 1e9
        );
        #[cfg(feature = "seed-stats")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            seed_stats::dump();
            let n = reads.len() as f64;
            eprintln!(
                "per read: round 1 starts {:.2} fwd {:.1} bwd {:.1}; round 2 starts {:.2} fwd {:.1} bwd {:.1}",
                R1.load(Relaxed) as f64 / n, R1F.load(Relaxed) as f64 / n, R1B.load(Relaxed) as f64 / n,
                R2.load(Relaxed) as f64 / n, R2F.load(Relaxed) as f64 / n, R2B.load(Relaxed) as f64 / n
            );
        }
    }
    #[cfg(feature = "seed-stats")]
    static R1: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "seed-stats")]
    static R1F: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "seed-stats")]
    static R1B: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "seed-stats")]
    static R2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "seed-stats")]
    static R2F: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    #[cfg(feature = "seed-stats")]
    static R2B: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// An SMEM without its `l`, which the production walks do not maintain (see `backward_ext_konly`).
    fn sans_l(s: &Smem) -> (u32, u32, i64, i64) {
        (s.m, s.n, s.k, s.s)
    }

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
            assert_eq!(sm.s, naive_occ(&reference, sub), "SMEM interval size wrong");
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
                // `l` is not compared: the batched walk leaves it stale past a unique match (the
                // text compare) and after every backward step, and nothing reads it after seeding.
                assert_eq!(
                    batched[r].iter().map(sans_l).collect::<Vec<_>>(),
                    per_read.iter().map(sans_l).collect::<Vec<_>>(),
                    "batched != per-read at read {r} (len {}, msl={msl}, mi={mi})",
                    read.len()
                );
            }
        }
    }

    /// A batch of reads no longer than the seed-jump table is deep, starting with strings the
    /// reference lacks: a walk from base 0 that ends inside the table must stop there, or it keeps
    /// pushing candidates into a `prev` sized by the batch's longest read.
    #[test]
    fn short_reads_ending_inside_the_jump_table() {
        let fm = tiny();
        let mut state = 0x0005_4017_0000_0003u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let reads: Vec<Vec<u8>> = (0..400)
            .map(|_| (0..10).map(|_| (next() % 4) as u8).collect())
            .collect();
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();
        for &msl in &[19i32, 8, 5] {
            let batched = collect_smems_batched(&fm, &refs, msl, 1);
            for (r, read) in reads.iter().enumerate() {
                assert_eq!(
                    batched[r].iter().map(sans_l).collect::<Vec<_>>(),
                    collect_smems(&fm, read, msl, 1)
                        .iter()
                        .map(sans_l)
                        .collect::<Vec<_>>(),
                    "read {r}, msl={msl}"
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
        for _ in 0..240 {
            // Read flavour: 0 = random (shallow SMEMs), 1..=5 = exact reference slice, 3/4 also
            // get two point mutations, 5 an `N` in its first 30 bases (round 3's first walk).
            let kind = next() % 6;
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
            if (3..=4).contains(&kind) && r.len() > 4 {
                for _ in 0..2 {
                    let p = (next() as usize) % r.len();
                    r[p] = (next() % 4) as u8;
                }
            }
            if kind == 5 {
                r[(next() % 30) as usize] = 4;
            }
            reads.push(r);
        }
        // Borrowed view of the batch, in the `&[&[u8]]` shape the batched entry points take.
        let refs: Vec<&[u8]> = reads.iter().map(|r| r.as_slice()).collect();

        // Round 3's first walk comes from round 1's record: vary where it stops (`max_mem_intv`,
        // 1 makes a unique match go on) and how long it must be (`-k`).
        for (max_intv, k) in [(20, 19), (1, 19), (2, 12), (500, 25), (0, 19)] {
            let mut opt = MemOpt::default();
            opt.max_mem_intv = max_intv;
            opt.min_seed_len = k;
            let batched = mem_collect_smem_batched(&fm, &refs, &opt);
            for (r, read) in reads.iter().enumerate() {
                let per_read = mem_collect_smem(&fm, read, &opt);
                assert_eq!(
                    batched[r].iter().map(sans_l).collect::<Vec<_>>(),
                    per_read.iter().map(sans_l).collect::<Vec<_>>(),
                    "batched full != per-read at read {r} (len {}, max_intv {max_intv}, k {k})",
                    read.len()
                );
            }
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
