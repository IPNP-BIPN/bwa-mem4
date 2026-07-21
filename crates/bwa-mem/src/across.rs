//! Batched seed extension across a whole read batch, porting bwa-mem2's
//! `mem_chain2aln_across_reads_V2` (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! bwa-mem2 does not extend one seed at a time: it collects every seed's **left** and **right**
//! one-sided extension across all reads into two job arrays, sorts each by length so a SIMD batch
//! packs similar-length alignments into its lanes (bwa-mem2 also bins int8/int16/scalar by length),
//! runs them through the vectorized banded Smith-Waterman (`bandedSWA`), and scatters the results
//! back with the exact `MAX_BAND_TRY` band-doubling acceptance logic. Right extensions use each
//! region's post-left score as `h0`, so left must complete before right.
//!
//! Because each region's extension result depends only on its own `(query, target, h0, w)`, batching
//! and length-sorting are **result-preserving**: [`align_reads_batched`] returns, for every read, the
//! same `Vec<MemAlnReg>` as calling [`crate::align_read`] per read (checked by an equivalence test),
//! while routing the DP through a batched [`SwBackend`] (the NEON kernel). The retry `prev` semantics
//! mirror the per-read [`crate::extend_side`] (`prev` reset to -1 per side) so the two paths are
//! bit-for-bit identical.
//!
//! # Shape of the port, and where it deliberately differs from the C
//!
//! bwa-mem2 keeps the pending extensions in two flat `SeqPair` arrays plus two byte arenas
//! (`seqBufLeftQer` / `seqBufLeftRef` and the right-hand pair), each pair recording only
//! `(idq, idr, len1, len2)` offsets into those arenas (`bwamem.cpp:2076-2087`, filled at
//! `bwamem.cpp:2247-2317` for left and `bwamem.cpp:2324-2415` for right). We keep the query and
//! target bytes inline in [`SideJob`] instead. That is a pure allocation-strategy change: the DP
//! never sees the arena, only the `(query, target)` pair, so it cannot affect output.
//!
//! bwa-mem2 also *bins* the pairs three ways by length and by the worst-case score
//! `h0 + min(len1, len2) * opt->a` (`bwamem.cpp:2303-2312`): under `MAX_SEQ_LEN8` it runs the int8
//! kernel, under `MAX_SEQ_LEN16` the int16 kernel, otherwise a scalar fallback, and each bin gets its
//! own independent `MAX_BAND_TRY` loop (the six near-identical loops at `bwamem.cpp:2472`, `2536`,
//! `2604`, `2688`, `2749`, `2819`). We run a single loop per side and let the [`SwBackend`] pick its
//! own lane width internally, because the three C kernels are specified to return identical scores;
//! only their saturation limits differ, and the binning exists precisely to stay inside them.
//!
//! Two more structural differences worth knowing before reading the code:
//!
//! * bwa-mem2 recomputes `seedcov` *inside* every scatter, guarded by all four of `rb/qb/qe/re`
//!   being off the `H0_` sentinel (`bwamem.cpp:2506-2516`, `2723-2732`, and the seed-terminal case at
//!   `bwamem.cpp:2419-2431`). That means it runs up to three times per region and only the last one
//!   counts. We hoist it into one pass over the final bounds after both sides have landed, which is
//!   the same fixed point.
//! * bwa-mem2 threads the retry queue by *compaction*: rejected pairs are copied into
//!   `pair_ar_aux` and the two arrays are swapped (`bwamem.cpp:2519-2524`). We keep a per-job
//!   `active` flag and re-filter. Same set, and since we re-sort by length every round the surviving
//!   order is regenerated rather than inherited, which is fine because the acceptance test is
//!   per-job and order-independent.
//!
//! # Reading order
//!
//! 1. [`align_reads_batched`] is the entry point and the spine: seed and chain the batch, collect
//!    region slots and one-sided DP jobs, run left then right, then `seedcov`, then the discard pass.
//! 2. [`SideJob`] is the unit of work the collection pass emits.
//! 3. [`run_side`] consumes those jobs: band-doubling rounds, acceptance test, scatter into regions.
//! 4. [`discard_contained`] is the post-extension purge; [`RegMeta`] is the back-pointer state it
//!    needs, and [`seed_ext_redundant`] is the up-front skip that shares its containment argument.
//! 5. [`skip_contained_enabled`] / [`discard_enabled`] are the two environment gates.
//!
//! # Glossary of names kept from the C
//!
//! Several locals here deliberately keep bwa-mem2's short names, because the correspondence to
//! `bwamem.cpp` / `ksw.cpp` is the thing a reader needs to check. In plain language:
//!
//! | Name | Meaning |
//! |------|---------|
//! | `h0` | starting DP score handed to `ksw_extend`, so z-drop and clipping judge the whole alignment rather than this side alone. Left jobs: the seed's own match score. Right jobs: the region's post-left `score`. |
//! | `qle` | **q**uery **le**ngth consumed by the best *local* endpoint of the extension. |
//! | `tle` | **t**arget (reference) length consumed by that same local endpoint. |
//! | `gscore` | best score among alignments that consume the *entire* remaining query ("global"/glocal on the query side). |
//! | `gtle` | target length consumed by the `gscore` alignment. |
//! | `qb` / `qe` | region's **q**uery **b**egin / **e**nd, half-open offsets into the read (i32). |
//! | `rb` / `re` | region's **r**eference **b**egin / **e**nd, positions in the `0..2*l_pac` forward+reverse concatenation (i64). Note: inside the right-extension collection block, a local `re` is instead an offset *into `rseq`*; that one is flagged where it appears. |
//! | `rmax` (`rmax0`, `rmax1`) | reference window `[rmax0, rmax1)` spanning a whole chain, wide enough for any extension the chain could produce. |
//! | `rseq` | the bases of that window, materialised once per chain; index into it is `position - rmax0`. |
//! | `frac_rep` | fraction of the chain's seeds that fell in repetitive regions; feeds MAPQ. |
//! | `seedcov` | number of query bases covered by seeds lying entirely inside the finished region. Feeds MAPQ and the dedup redundancy test. |
//! | `truesc` | the alignment's "true" score, i.e. the score excluding the `h0` head start, accumulated across the two sides. |
//! | `sub` | best score among *other* (suboptimal) regions overlapping this one; the MAPQ denominator. |
//! | `csub` | same idea restricted to the region's own chain ("chained sub"). |
//! | `seedlen0` | length of the seed that founded the region. The discard pass compares against it. |
//! | `w` | band half-width actually used, doubled per `MAX_BAND_TRY` round. |
//! | `max_off` | furthest excursion of the DP optimum from the main diagonal; drives acceptance test 2. |
//! | `lim` | in the discard pass: how many surviving regions this read has produced so far, which bounds the containment scan. |

use crate::{cal_max_gap, MemAlnReg, H0_SENTINEL, MAX_BAND_TRY};
use bwa_chain::{build_chains_from_smems, mem_chain_flt, MemChain};
use bwa_core::MemOpt;
use bwa_extend::{ExtendJob, SwBackend};
use bwa_index::{BntSeq, FmIndex};
use bwa_seed::{mem_collect_smem_batched, MemSeed};

/// A seed counts as "comparably long" to the candidate when it is at least this fraction of the
/// candidate's length. The C spells the test `t->len < s->len * .95` (`bwamem.cpp:2969`), an
/// int-vs-double comparison, so this must stay `f64`: `0.95` is not exactly representable and its
/// rounding is load-bearing at the tie boundary.
const COMPARABLE_LEN_FRAC: f64 = 0.95;

/// Slack in the discard pass's "this seed is materially better evidence" test: a contained seed more
/// than this fraction of the read longer than the founding region's `seedlen0` is kept regardless.
/// C: `if (s->len - p->seedlen0 > .1 * l_query)` (`bwamem.cpp:2943`).
const SEEDLEN_SLACK_FRAC: f64 = 0.1;

/// Alphabet size for the substitution matrix: 2-bit codes 0..=3 plus N, so `opt.mat` is a flattened
/// 5x5. The C passes the same `5` when constructing its `BandedPairWiseSW` (`bwamem.cpp:2452-2458`).
const ALPHABET_SIZE: usize = 5;

/// nh13's `mem_seed_ext_redundant` (`--skip-contained-ext`): true when seed `si` is strictly
/// contained, on the same diagonal, in a longer seed of the same chain, and no comparably long seed
/// interferes on a different diagonal. Skipping its banded-SW saves ~7% SE.
///
/// **On by default, and byte-identical to bwa-mem2** -- verified over 1M real HG002 reads and 1M
/// pairs with it both on and off. A skipped seed still gets a region slot, because the discard pass
/// reproduces bwa-mem2's slot-ordered scan; the slot carries no DP result, so it starts purged
/// (`qb = qe = -1`) and the compaction before `mem_sort_dedup_patch` drops it. That compaction is
/// what makes this safe: the placeholder bounds never reach the dedup's `re` sort, which is unstable
/// and decides who survives a score tie. Without it the fake bounds shifted real regions across ties.
///
/// The skip is sound rather than lucky: it fires only when the seed is contained, on the same
/// diagonal, in a *longer* seed of the same chain. That container is extended earlier (descending
/// score order), its region covers the seed, the shared diagonal makes `qd - rd == 0` trivially
/// within the band, and the `seedlen0` guard cannot fire because the container is longer -- so the
/// discard pass would purge this seed anyway. nh13's test is a conservative subset of the purge.
///
/// `BWA4_NO_SKIP_CONTAINED` opts out. Cached: the two extension paths (batched
/// [`align_reads_batched`] and per-read `mem_chain2aln`) must agree, so they share this one decision.
///
/// # Parameters
///
/// None. The single input is the process environment, read exactly once.
///
/// # Returns
///
/// True when the skip is enabled (the default: the variable is absent), false when
/// `BWA4_NO_SKIP_CONTAINED` is set to anything at all, including the empty string. The answer is
/// decided on the first call and frozen for the life of the process, so setting the variable from
/// inside the program after alignment has begun has no effect.
pub(crate) fn skip_contained_enabled() -> bool {
    // Process-wide memo of the environment decision. `OnceLock` rather than a plain `static mut` so
    // the first caller from any worker thread wins and every later caller observes that same answer;
    // a per-call `var_os` would also take a lock inside libstd on each seed.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("BWA4_NO_SKIP_CONTAINED").is_none())
}

/// True when seed `si` of `seeds` may have its banded-SW skipped: it is strictly contained, on the
/// same diagonal, inside a longer seed of the same chain, and nothing comparably long interferes.
///
/// Note this predicate has **no counterpart in the vendored bwa-mem2** (`grep seed_ext_redundant`
/// over `reference/bwa-mem2/src` finds nothing): it is nh13's later `--skip-contained-ext` idea from
/// upstream lh3/bwa, reconstructed here. Its correctness therefore does not rest on matching a C
/// line but on the containment argument in [`skip_contained_enabled`]'s docs, plus the 1M-read /
/// 1M-pair byte-identity check recorded there.
///
/// # Parameters
///
/// * `seeds`: one chain's seeds, in the chain's own (query-position ascending) order, *not* the
///   descending-score order the extension loop iterates in. Supplied by the collection pass as
///   `&chain.seeds`. The predicate is order-independent: it quantifies over all other seeds of the
///   chain, so passing a different permutation of the same chain gives the same answer. Must be the
///   candidate's own chain: seeds of other chains are not comparable here.
/// * `si`: index into `seeds` of the candidate seed, the one whose banded-SW we are deciding whether
///   to skip. Range `0..seeds.len()`; the function indexes `seeds[si]` directly, so an out-of-range
///   value panics.
///
/// # Returns
///
/// True only if skipping the candidate's DP is safe (a strictly longer same-diagonal container
/// exists and nothing comparably long interferes). False is always the conservative answer: it costs
/// a DP, never correctness.
#[must_use]
pub(crate) fn seed_ext_redundant(seeds: &[MemSeed], si: usize) -> bool {
    // `cand` is the C's `s`: the seed we are deciding whether to skip.
    let cand = seeds[si];
    // Diagonal id. A seed occupies reference positions rbeg..rbeg+len opposite query qbeg..qbeg+len,
    // so `rbeg - qbeg` is constant along an ungapped alignment: two seeds share a diagonal exactly
    // when this differs by zero, i.e. one can be reached from the other with no indel. `qbeg` is i32
    // and `rbeg` i64 (a position in the 2*l_pac forward+reverse concatenation), hence the widening.
    let cand_diag = cand.rbeg - i64::from(cand.qbeg);

    // ---- step 1: is there a strictly longer, same-diagonal seed that contains the candidate? ----
    // Invariant at the top of each iteration: false means no seed among `seeds[..j]` qualified as a
    // container. The loop breaks the moment one does, so the flag is only ever set once.
    let mut has_container = false;
    for (j, container) in seeds.iter().enumerate() {
        if j == si || container.len <= cand.len {
            continue; // must be strictly longer
        }
        if container.rbeg - i64::from(container.qbeg) != cand_diag {
            continue; // must be the same diagonal
        }
        // Query-interval containment. Because the diagonals already match, containment on the query
        // implies containment on the reference too, so one axis suffices.
        if cand.qbeg >= container.qbeg && cand.qbeg + cand.len <= container.qbeg + container.len {
            has_container = true;
            break;
        }
    }
    if !has_container {
        return false;
    }

    // ---- step 2: interference guard ----
    // Mirrors the PE18 purge: a seed at least `COMPARABLE_LEN_FRAC` of the candidate's length, and
    // overlapping the candidate on a *different* diagonal by at least `cand.len/4`, could lead to a
    // distinct alignment, so the candidate must be extended after all.
    //
    // This is deliberately the same three tests as the confirmation loop in `discard_contained`
    // below, which ports `bwamem.cpp:2965-2977`. Keeping them textually parallel is the point: if
    // this predicate were ever *weaker* than the purge's, we would skip a DP that bwa-mem2 keeps.
    //
    // The C form is `t->len < s->len * .95`, an int-vs-double comparison where `s->len * .95`
    // promotes to double. We spell out the `f64::from` on both sides so Rust performs the same f64
    // comparison rather than an f32 or integer one. `0.95` is not exactly representable, so the
    // rounding of `cand.len * COMPARABLE_LEN_FRAC` is load-bearing at the tie boundary.
    for (j, other) in seeds.iter().enumerate() {
        if j == si || (f64::from(other.len)) < f64::from(cand.len) * COMPARABLE_LEN_FRAC {
            continue;
        }
        // Case A: cand starts at or before other. Overlap on the query is
        // `cand.qbeg + cand.len - other.qbeg`, and it must be at least a quarter of cand
        // (`cand.len >> 2`, an arithmetic shift on a non-negative i32, so exactly floor(len/4)). The
        // final clause is the diagonal test written as a difference of differences:
        // `other.qbeg - cand.qbeg != other.rbeg - cand.rbeg` means the two seeds sit on different
        // diagonals, i.e. reconciling them would need an indel, i.e. a genuinely distinct alignment.
        if cand.qbeg <= other.qbeg
            && cand.qbeg + cand.len - other.qbeg >= cand.len >> 2
            && i64::from(other.qbeg) - i64::from(cand.qbeg) != other.rbeg - cand.rbeg
        {
            return false;
        }
        // Case B: the mirror image, other starting at or before cand. Note the quarter-overlap
        // threshold stays `cand.len >> 2` (the *candidate's* length) in both cases, as in the C.
        if other.qbeg <= cand.qbeg
            && other.qbeg + other.len - cand.qbeg >= cand.len >> 2
            && i64::from(cand.qbeg) - i64::from(other.qbeg) != cand.rbeg - other.rbeg
        {
            return false;
        }
    }
    true
}

/// Where a region came from, so the discard pass can recover its seed. `pos` is the seed's index in
/// the chain's descending-score order, which is also the region's offset within its chain's block of
/// slots, so region `idx` of `(chain, q)` is `idx - pos + q`.
///
/// This replaces bwa-mem2's two pieces of back-pointer state, which we have no equivalent of:
/// `mem_seed_t::aln` (the seed's slot index, set at `bwamem.cpp:2216`) and the global `srtgg` /
/// `srt2` arrays of sorted seed indices that the discard pass walks (`bwamem.cpp:2920-2926`).
/// bwa-mem2 can afford a seed -> region pointer because its regions are one `calloc`ed array per
/// read; we rebuild the mapping in the other direction instead.
#[derive(Clone, Copy)]
pub(crate) struct RegMeta {
    /// Index into the read's `Vec<MemChain>` of the chain this region came from.
    pub chain: u32,
    /// Rank of the seed within its chain's descending-`(score, index)` order, i.e. `k` counted from
    /// the top in bwa-mem2's `for (k = c->n-1; k >= 0; k--)` loop (`bwamem.cpp:2205`). Doubles as the
    /// region's offset inside its chain's contiguous block of slots, which is what lets
    /// `discard_contained` find the chain's first slot as `idx - pos`.
    pub pos: u32,
    /// Index into `chain.seeds`, i.e. bwa-mem2's `srt2[k]` payload (the low 32 bits of the packed
    /// `score<<32 | i` sort key built at `bwamem.cpp:2189`).
    pub seed: u32,
}

/// bwa-mem2's discard pass (the tail of `mem_chain2aln_across_reads_V2`, `bwamem.cpp:2895-2990`).
///
/// bwa-mem2 extends every seed up front, then walks each read's chains (seeds in descending score
/// order, the same order the collection pass emits slots in) and **purges** the region of any seed
/// that a previously-kept region already covers within the band -- unless a comparably long seed on
/// a different diagonal interferes, meaning the seed could still yield a distinct alignment. Purged
/// regions are marked `qb = qe = -1`, exactly as bwa-mem2 does; `mem_sort_dedup_patch`'s compaction
/// drops them later.
///
/// This is what keeps repeat-region reads from accumulating near-duplicate regions that survive the
/// dedup's redundancy test and inflate `sub` (hence collapse MAPQ). It is not an optimization: the
/// extensions have already run.
///
/// `lim` (seeds kept so far for this read) bounding the scan is bwa-mem2's own, and it is
/// load-bearing: it caps how many regions are examined, so the outcome depends on the **slot order**
/// of `regs[r]`. That order must therefore match bwa-mem2's `s->aln` slots, i.e. one slot per seed.
///
/// One structural note: bwa-mem2 nests two loops (over chains `j`, then over that chain's seeds
/// `k`), sharing a single per-read `lim[l]` counter across both. Because our `regs` already lays the
/// chains out back to back in exactly that order, the flat `for idx in 0..n_slots` here visits the same
/// seeds in the same sequence with the same shared `lim`. Flattening is only legal because `lim` is
/// per-read, not per-chain: `bwamem.cpp:2989` increments `lim[l]`, indexed by read.
///
/// # Parameters
///
/// * `opt`: alignment options, the same struct the whole aligner shares. Only `w` (band half-width,
///   as the ceiling on `prior.w`) and the scoring fields `a`, `o_del/e_del`, `o_ins/e_ins` are read,
///   the latter four indirectly via [`cal_max_gap`]. Never mutated.
/// * `l_query`: this read's length in bases (a positive i32, typically 100-300 for Illumina). Used
///   solely as the base of the `0.1 * l_query` seed-length slack below; it is not a bound on any
///   index here.
/// * `chains`: this read's chains *after* `mem_chain_flt`, in the same order the collection pass
///   walked them. Indexed by `RegMeta::chain`, so it must be the identical vector that produced
///   `meta`, not a re-filtered copy.
/// * `regs`: this read's regions, exactly one per seed and **in slot order** (chains back to back,
///   seeds within a chain in descending-score order). Mutated in place, and only ever by writing
///   `qb = qe = -1` on a purged entry; every other field, and the vector's length, are left alone.
///   `regs.len()` must equal `meta.len()` and `preskip.len()`.
/// * `meta`: parallel to `regs`, one [`RegMeta`] per slot, giving the chain / order-position / seed
///   that slot came from. Read-only. Supplied by the collection pass in [`align_reads_batched`].
/// * `preskip`: parallel to `regs`, true for a slot whose extension nh13's `seed_ext_redundant`
///   elided. Read-only; it seeds the local `purged` vector so a never-extended slot behaves exactly
///   like one the C had already tombstoned with `srt2[k] = UINT_MAX`.
pub(crate) fn discard_contained(
    opt: &MemOpt,
    l_query: i32,
    chains: &[MemChain],
    regs: &mut Vec<MemAlnReg>,
    meta: &[RegMeta],
    preskip: &[bool],
) {
    // Number of region slots for this read, i.e. its total seed count across all surviving chains.
    // Fixed for the whole pass: purging blanks a slot's bounds, it never removes the slot.
    let n_slots = regs.len();
    // `lim`: surviving regions produced so far for this read (see the glossary in the module header).
    // Invariant at the top of each `idx` iteration: `lim` is the number of slots in `0..idx` that
    // were kept, and it is the horizon (a count, not an index) the containment scan may search.
    let mut lim: i32 = 0;
    // bwa-mem2's `srt2[k] = UINT_MAX`. Seeds skipped up front (nh13's `seed_ext_redundant`) start
    // out purged: their slot exists to preserve scan order, but they were never extended.
    // Invariant: `purged[i]` is true exactly when slot `i` is pre-skipped or has been tombstoned by
    // an earlier iteration of this pass. Read back in step 2, where a purged seed may not vouch.
    let mut purged: Vec<bool> = preskip.to_vec();
    for idx in 0..n_slots {
        if purged[idx] {
            continue; // pre-skipped: never extended, and contributes no `lim`
        }
        // Back-pointers for the slot under judgement: the chain it came from, and `seed`, the C's
        // `s`, the actual seed whose containment decides whether slot `idx` survives.
        let cur_meta = meta[idx];
        let chain = &chains[cur_meta.chain as usize];
        let seed = chain.seeds[cur_meta.seed as usize];

        // ---- step 1: has an earlier surviving region already covered this seed? ----
        // "test whether extension has been made before" (`bwamem.cpp:2929`): scan this read's regions
        // in slot order, stopping once `lim` non-purged ones have been examined without finding a
        // container. Two exits matter and they mean opposite things:
        //   - fall out with `n_rejected == lim`: no earlier region swallows this seed, so keep it.
        //   - `break` with `n_rejected < lim`: region `prior` swallows it, so it is a purge candidate.
        // `n_rejected` counts *rejected* regions (the C's `v`), so `n_rejected < lim` after the loop
        // can only be reached via `break`.
        // Loop state for the containment scan. Invariant at the top of each iteration: `scan_idx` is
        // the next slot to examine, and `n_rejected` counts how many *non-purged* slots in
        // `0..scan_idx` failed to contain the seed. The scan stops when `n_rejected` reaches `lim`
        // (the C's budget of previously kept regions) or slots run out.
        let mut n_rejected: i32 = 0;
        let mut scan_idx = 0usize;
        while scan_idx < n_slots && n_rejected < lim {
            // `prior` is the C's `p`: an already-emitted region of this same read.
            let prior = &regs[scan_idx];
            if prior.qb == -1 && prior.qe == -1 {
                scan_idx += 1;
                continue; // already purged: not counted against `lim`
            }
            // The seed's [qbeg, qbeg+len) x [rbeg, rbeg+len) box must lie wholly inside `prior`
            // on *both* axes (`bwamem.cpp:2941-2942`). Note the asymmetric widths: `qb/qe` are i32
            // query offsets, `rb/re` i64 positions in the doubled pac, hence the `i64::from`.
            if seed.rbeg < prior.rb
                || seed.rbeg + i64::from(seed.len) > prior.re
                || seed.qbeg < prior.qb
                || seed.qbeg + seed.len > prior.qe
            {
                n_rejected += 1;
                scan_idx += 1;
                continue; // not fully contained
            }
            // `bwamem.cpp:2943`: `if (s->len - p->seedlen0 > .1 * l_query)`. `seedlen0` is the length
            // of the seed that *founded* `prior`. If this seed is more than 10% of the read longer
            // than that founder, it is materially better evidence and could pull out a better
            // alignment, so `prior`'s existence is not a reason to drop it. In C the left-hand side
            // is an int promoted to double; `f64::from` on the already-computed i32 difference
            // reproduces that exactly, including the case where the difference is negative.
            if f64::from(seed.len - prior.seedlen0) > SEEDLEN_SLACK_FRAC * f64::from(l_query) {
                n_rejected += 1;
                scan_idx += 1;
                continue; // this seed may give a better alignment
            }
            // Ahead of the seed: is it "around" this hit, within the gap the band still allows?
            // `qd`/`rd` are the query and reference distances from `prior`'s start to the seed's
            // start (`bwamem.cpp:2945`). Both are non-negative here, since containment was just
            // checked.
            // `qd`: query bases between `prior`'s start and the seed's start. `rd`: the same gap
            // measured on the reference. Both in bases, both non-negative here.
            let qd = i64::from(seed.qbeg - prior.qb);
            let rd = seed.rbeg - prior.rb;
            // `cal_max_gap(opt, min(qd, rd))` is the longest indel an alignment of that length could
            // afford before the gap penalty eats the match score: it solves
            // `qlen*a - o = e*l` for l, takes the larger of the deletion and insertion answers, and
            // clamps to `[1, 2*opt.w]` (`bwamem.cpp:66-76`). The `as i32` narrowing is safe because
            // `min(qd, rd)` is bounded by the region's query span, which is a read length.
            // `max_gap`: longest indel, in bases, an alignment of `min(qd, rd)` length could pay for.
            let max_gap = i64::from(cal_max_gap(opt, qd.min(rd) as i32));
            // Bounded by the band actually used for `prior`, since a shift wider than the band could
            // not have been found by `prior`'s own DP anyway. `w` is therefore the diagonal slack, in
            // bases, that the test just below allows between the seed and `prior`.
            let w = max_gap.min(i64::from(prior.w));
            // `|qd - rd| < w` written as two one-sided tests, exactly as the C does. This asks
            // whether the seed sits on a diagonal reachable from `prior`'s start within the band: if
            // so, `prior`'s alignment already explores this diagonal and the seed adds nothing.
            if qd - rd < w && rd - qd < w {
                break;
            }
            // Same test behind the seed: distances from the seed's end to `prior`'s end. These
            // `qd`/`rd` shadow the previous pair, matching the C's reuse of the same locals.
            let qd = i64::from(prior.qe - (seed.qbeg + seed.len));
            let rd = prior.re - (seed.rbeg + i64::from(seed.len));
            // Same two quantities as above, recomputed for the trailing gap: affordable indel length,
            // then the diagonal slack after clamping to `prior`'s band.
            let max_gap = i64::from(cal_max_gap(opt, qd.min(rd) as i32));
            let w = max_gap.min(i64::from(prior.w));
            if qd - rd < w && rd - qd < w {
                break;
            }
            n_rejected += 1;
            scan_idx += 1;
        }

        // ---- step 2: confirm no comparably long off-diagonal seed of this chain interferes ----
        if n_rejected < lim {
            // The seed is (almost) contained in an existing alignment. Confirm it cannot lead to a
            // different one: look for a comparably long, already-processed seed of the same chain
            // that overlaps it on a *different* diagonal. bwa scans `srt2[k+1..]` (higher scores,
            // i.e. our earlier positions); only whether one exists matters, not which.
            //
            // Why `srt2[k+1..]` maps to `0..cur_meta.pos`: `srt2` is sorted *ascending* by the packed
            // `score<<32 | i` key and the C walks `k` downward from `c->n-1`, so indices above `k`
            // are the higher-scoring, already-visited seeds. Our `pos` counts from the top of the
            // same order, so "already visited" is `pos` strictly less than ours, i.e.
            // `0..cur_meta.pos`.
            // bwa expresses "none found" as the loop running to completion (`v == c->n` at
            // `bwamem.cpp:2978`); we use an explicit `interferes` flag for the same thing.
            // Slot index of this chain's *first* seed. Valid because a chain's slots are contiguous
            // and `pos` is the seed's offset inside that block, so `idx - pos` cannot underflow.
            let chain_first_slot = idx - cur_meta.pos as usize;
            // Invariant at the top of each iteration: false means no already-visited seed of this
            // chain has yet been found that is comparably long, overlapping, and off-diagonal. Set
            // once and broken out of; true is the C's "v != c->n" outcome, meaning do NOT purge.
            let mut interferes = false;
            for q in 0..cur_meta.pos as usize {
                // The slot of the q-th higher-scoring seed of this chain, i.e. one already judged.
                let earlier_slot = chain_first_slot + q;
                // `bwamem.cpp:2967`: `if (srt2[v] == UINT_MAX) continue;`. A seed purged earlier in
                // this very pass is not allowed to vouch for keeping this one. Our `purged` vector
                // is the stand-in for that sentinel write, and it is seeded from `preskip` so the
                // nh13 pre-skips are treated identically.
                if purged[earlier_slot] {
                    continue;
                }
                // The competing seed itself (the C's `t`): a higher-scoring, still-alive seed of the
                // same chain, the only kind allowed to vouch for keeping `seed`.
                let earlier_seed = chain.seeds[meta[earlier_slot].seed as usize];
                // Only a comparably long seed can plausibly anchor a different alignment.
                if f64::from(earlier_seed.len) < f64::from(seed.len) * COMPARABLE_LEN_FRAC {
                    continue;
                }
                if seed.qbeg <= earlier_seed.qbeg
                    && seed.qbeg + seed.len - earlier_seed.qbeg >= seed.len >> 2
                    && i64::from(earlier_seed.qbeg - seed.qbeg) != earlier_seed.rbeg - seed.rbeg
                {
                    interferes = true;
                    break;
                }
                if earlier_seed.qbeg <= seed.qbeg
                    && earlier_seed.qbeg + earlier_seed.len - seed.qbeg >= seed.len >> 2
                    && i64::from(seed.qbeg - earlier_seed.qbeg) != seed.rbeg - earlier_seed.rbeg
                {
                    interferes = true;
                    break;
                }
            }
            if !interferes {
                // `bwamem.cpp:2979-2982`: purge the alignment and blank the sort slot. `-1` (not the
                // `H0_SENTINEL`) is bwa's chosen tombstone, and it is exactly what the
                // `prior.qb == -1 && prior.qe == -1` skip at the top of the scan and the final
                // `retain(|reg| reg.qe > reg.qb)` compaction both key off. Do not "tidy" it to an
                // Option.
                regs[idx].qb = -1;
                regs[idx].qe = -1;
                purged[idx] = true;
                continue; // purged seeds do not count towards `lim`
            }
        }
        // ---- step 3: the seed survives ----
        // Reached only for a kept seed. `lim` is therefore "number of surviving regions produced so
        // far for this read", which is the horizon the next seed's containment scan is allowed to
        // search. Growing it monotonically is what makes the pass O(n^2) worst case but also what
        // makes the result depend on slot order, hence the invariant stated in the doc comment.
        lim += 1;
    }
}

/// One pending one-sided extension, with a back-pointer to the region it fills.
///
/// The Rust analogue of bwa-mem2's `SeqPair` (`bandedSWA.h`), except that `query`/`target` hold the
/// bytes inline where the C stores `(idq, idr, len1, len2)` offsets into shared arenas. `read`/`reg`
/// are the C's `sp.seqid` / `sp.regid` (`bwamem.cpp:2251-2252` for left, `2331-2332` for right),
/// which is how the scatter finds `av_v[sp->seqid].a[sp->regid]`.
struct SideJob {
    /// Index into the batch's `reads` / `regs` outer vectors (the C's `seqid`).
    read: usize,
    /// Index into `regs[read]`, i.e. the region slot this job's result lands in (the C's `regid`).
    /// Fixed at collection time and never re-derived, so it stays valid across the length sorts.
    reg: usize,
    /// Query bases, 2-bit codes 0..=3 with 4 for N, oriented **outward from the seed**: for a left
    /// job the read is reversed so index 0 is the base immediately left of the seed. This is why the
    /// same DP kernel serves both sides.
    query: Vec<u8>,
    /// Reference bases in the same outward orientation, taken from the chain's fetched window.
    target: Vec<u8>,
    /// Starting DP score handed to `ksw_extend`. For left jobs it is `seed.len * opt.a` (the seed's
    /// own match score, `bwamem.cpp:2249`). For right jobs it is filled in later from the region's
    /// post-left `score`, which is why it starts at the `H0_SENTINEL` placeholder, matching the C's
    /// `sp.h0 = H0_; //random number` at `bwamem.cpp:2330` and its overwrite at `bwamem.cpp:2676`.
    h0: i32,
    /// Previous round's score, for the band-doubling acceptance test (`score == prev` means widening
    /// the band bought nothing, so the result is accepted).
    ///
    /// Its value before the FIRST round is not always -1, and that detail is load-bearing: it is
    /// whatever the C's `int prev = a->score;` (`bwamem.cpp:2492`) reads off the region. A left job
    /// sees the -1 the region was initialised with, but a right job sees the left extension's score,
    /// which is that job's own `h0`. See the fixup loop that sets `job.prev = job.h0` for why
    /// passing -1 on both sides breaks for `w <= 1`.
    prev: i32,
    /// Still needs another (wider) band pass.
    active: bool,
}

/// Align a batch of reads (2-bit codes) through seeding, chaining, and **batched** extension,
/// returning each read's alignment regions (pre-dedup), byte-identical to [`crate::align_read`].
///
/// The five phases below run strictly in order, and the order is forced by data dependencies rather
/// than taste: seed+chain all reads, collect one region slot and up to two extension jobs per seed,
/// run every left extension, run every right extension (needs the left scores as `h0`), then
/// `seedcov` and the discard pass (both need final bounds for the whole read).
///
/// # Parameters
///
/// * `fm`: the loaded FM index (BWT plus occurrence and suffix-array samples). Used read-only for
///   seeding, SA resolution, and `fm.base` reference-base lookups. Must be the index built from the
///   same FASTA as `bns`.
/// * `bns`: the `.ann`/`.amb` metadata (contig names, lengths, holes). `bns.l_pac` is the
///   forward-strand packed length in bases; reference coordinates in this function run over
///   `0..2*l_pac` (forward strand, then its reverse complement).
/// * `opt`: alignment options, straight from the CLI. Read-only. Fields consumed here: `w` (band
///   half-width), `a` (match score), the four gap penalties, `mat`, `pen_clip5`/`pen_clip3`, and
///   `zdrop`.
/// * `reads`: one `Vec<u8>` of 2-bit codes per read, values 0..=3 with 4 for N. One entry per read
///   in the batch, and batch size is the caller's choice. Already in the orientation the caller
///   wants: this function never reverse-complements a read. Empty reads are tolerated (they simply
///   produce no chains).
/// * `backend`: the batched Smith-Waterman implementation (in production the NEON kernel). Called
///   once per band-doubling round per side. Swapping it must not change the output, which is what
///   the equivalence test at the bottom of this file pins down.
///
/// # Returns
///
/// One `Vec<MemAlnReg>` per read, in `reads` order and one-to-one with it (a read with no alignment
/// yields an empty vector). The regions are **pre-dedup**, so the caller still runs
/// `mem_sort_dedup_patch`, but post-compaction: slots purged by the discard pass are already gone.
///
/// Batch size is the caller's business. The only thing that scales with it here is the seeding and
/// SA-resolution lockstep, which is where the win comes from.
pub fn align_reads_batched<B: SwBackend>(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    reads: &[Vec<u8>],
    backend: &B,
) -> Vec<Vec<MemAlnReg>> {
    // Length of the forward strand in bases. Every reference coordinate below lives in `0..2*l_pac`,
    // so `l_pac` is also the forward/reverse boundary that a DP window must never straddle.
    let l_pac = bns.l_pac;

    // ---- phase 1: seeding (round-1 SMEMs, then SA resolution, then chaining) ----
    // Round-1 SMEMs for the whole batch in lockstep (hides FM-index latency across reads), then
    // chain each read. Result-identical to per-read `build_chains` (batched seeding is verified equal
    // to per-read `collect_smems`).
    // Borrowed view of `reads`, the shape the batched seeder wants. No copy of the bases.
    let refs: Vec<&[u8]> = reads.iter().map(|c| c.as_slice()).collect();
    // One vector of round-1 SMEMs per read, parallel to `reads`. At this point each SMEM carries an
    // SA *interval* (`k`, `s`), not yet resolved reference positions.
    let per_read_smems = mem_collect_smem_batched(fm, &refs, opt);
    if std::env::var_os("BWA4_DUMP_SMEMS").is_some() {
        for sm in &per_read_smems {
            eprintln!("SMEM tot={}", sm.len());
            for p in sm {
                eprintln!(
                    "  smem q[{},{}) len={} s={} k={}",
                    p.m,
                    p.n + 1,
                    p.n + 1 - p.m,
                    p.s,
                    p.k
                );
            }
        }
    }
    // Debug gate for the two chain dumps below, hoisted out of the per-read loop so the environment
    // is read once rather than once per read. Off in any normal run.
    let dump_chains = std::env::var_os("BWA4_DUMP_CHAINS").is_some();

    // Resolve EVERY read's SA occurrences in ONE lockstep pass instead of one ~42-position call per
    // read. `get_sa_batch`'s W=32 window needs a long run of independent walks to keep the core's
    // ~28 memory-level-parallelism lanes busy; ~42 positions is 1.3 windows, so the pipeline never
    // reaches steady state. Measured (bwa-index's sa_batch_bench, genome index): 275.6 ns/lookup at
    // chunk=42 vs 186.9 at chunk>=128.
    //
    // Byte-identical by construction: `sa_positions_for_read` touches no FM index, `get_sa_batch` is
    // result-identical at any chunk size, and each read's slice keeps its own positions in their own
    // order. BWA4_SA_PER_READ=1 restores the per-read calls for A/B.
    // Rebound as `mut` because `sa_positions_for_read` filters and re-orders each read's SMEMs in
    // place while it enumerates their SA positions. Same data, no copy.
    let mut per_read_smems: Vec<Vec<bwa_index::Smem>> = per_read_smems;
    // Whether SA resolution runs as one batch for the whole read batch (the default and the fast
    // path) or falls back to per-read calls. A/B switch only: both produce identical positions.
    let batched_sa = std::env::var_os("BWA4_SA_PER_READ").is_none();
    // `all_positions`: every read's SA-interval offsets concatenated, in read order then SMEM order,
    // ready for one lockstep `get_sa_batch`. `per_read_counts`: per read, how many positions each of
    // its SMEMs contributed, which is what lets the chain builder re-split the flat result. Both are
    // left empty on the per-read fallback path, where nothing reads them.
    let (all_positions, per_read_counts): (Vec<i64>, Vec<Vec<i64>>) = if batched_sa {
        let mut pos = Vec::new();
        let counts = per_read_smems
            .iter_mut()
            .map(|smems| bwa_chain::sa_positions_for_read(opt, smems, &mut pos))
            .collect();
        (pos, counts)
    } else {
        (Vec::new(), Vec::new())
    };
    // Resolved reference start positions, one per entry of `all_positions` and in the same order:
    // `all_rbegs[i]` is the genome coordinate (in `0..2*l_pac`) of suffix-array entry
    // `all_positions[i]`. Sliced per read by `sa_cursor` below. Empty on the fallback path.
    let all_rbegs: Vec<i64> = if batched_sa {
        // Output buffer for `get_sa_batch`, pre-sized and zero-filled because the batch API writes by
        // index rather than pushing.
        let mut rbegs = vec![0i64; all_positions.len()];
        // Start instant for the optional `get_sa` profiling counters, or `None` when profiling is
        // off, in which case no clock is read at all.
        let sa_timer = bwa_chain::chain_time::enabled().then(std::time::Instant::now);
        if std::env::var_os("BWA4_SA_SORT").is_some() {
            // SPIKE (Zhang et al., CCGrid 2013 "Optimizing BWT-Based Sequence Alignment on Multicore
            // Architectures", §IV): resolve the SA lookups in ASCENDING POSITION order so that
            // consecutive walks land in nearby cp_occ pages, amortising page walks over many lookups
            // instead of one. A full sort is the pessimistic bound on the idea -- O(n log n) where a
            // 1-pass radix partition would be O(n) -- so if this loses, radix cannot save it.
            //
            // Byte-identical: get_sa_batch resolves each position independently, so resolution order
            // is unobservable; the original order is restored by the packed index before use.
            // Number of low bits of the packed sort key reserved for the original array index, so up
            // to 2^26 = 67.1M SA lookups per batch fit while the position keeps the high 38 bits.
            // The split is checked by the assert below, not assumed: raising it shrinks the position
            // field and would silently truncate coordinates on a large genome, lowering it caps the
            // batch size. Only the `BWA4_SA_SORT` spike path uses this; it does not affect output.
            const IDX_BITS: u32 = 26;
            assert!(
                all_positions.len() < (1 << IDX_BITS) && fm.ref_seq_len < (1 << (64 - IDX_BITS)),
                "packed (position, index) key would overflow"
            );
            // One packed `(position << IDX_BITS) | original_index` key per lookup. Sorting these
            // orders the lookups by position while keeping the return path back to the caller's
            // slot, so the original order can be restored after resolution.
            let mut keyed: Vec<u64> = all_positions
                .iter()
                .enumerate()
                .map(|(i, &p)| ((p as u64) << IDX_BITS) | (i as u64))
                .collect();
            keyed.sort_unstable();
            // The same lookups as `all_positions` but in ascending position order (index field
            // stripped back off), which is the locality the spike is testing.
            let sorted_positions: Vec<i64> = keyed.iter().map(|k| (k >> IDX_BITS) as i64).collect();
            // Results in that sorted order; the scatter loop below moves them back into `rbegs` at
            // each key's original index, so `rbegs` ends up identical to the unsorted path.
            let mut sorted_rbegs = vec![0i64; sorted_positions.len()];
            fm.get_sa_batch(&sorted_positions, &mut sorted_rbegs);
            for (i, k) in keyed.iter().enumerate() {
                rbegs[(k & ((1 << IDX_BITS) - 1)) as usize] = sorted_rbegs[i];
            }
        } else {
            fm.get_sa_batch(&all_positions, &mut rbegs);
        }
        if let Some(sa_timer) = sa_timer {
            use std::sync::atomic::Ordering::Relaxed;
            bwa_chain::chain_time::GET_SA_NS
                .fetch_add(sa_timer.elapsed().as_nanos() as u64, Relaxed);
            bwa_chain::chain_time::GET_SA_N.fetch_add(all_positions.len() as u64, Relaxed);
        }
        rbegs
    } else {
        Vec::new()
    };
    // Running read cursor into the flat `all_rbegs`. Invariant at the top of each read's closure
    // body: it points at the first resolved position belonging to that read, and it advances by that
    // read's total position count. Only meaningful on the batched path.
    let mut sa_cursor = 0usize;

    // Build and filter each read's chains from the (now resolved) SMEM occurrences.
    // One entry per read, in `reads` order: the chains that survive `mem_chain_flt`, each holding
    // the seeds the extension phase will work through. Everything downstream indexes this by read.
    let per_read_chains: Vec<Vec<MemChain>> = per_read_smems
        .into_iter()
        .enumerate()
        .zip(reads.iter())
        .map(|((ridx, smems), codes)| {
            // This read's chains *before* filtering (the C's pre-`mem_chain_flt` list).
            let pre = if batched_sa {
                // `counts`: positions contributed by each of this read's SMEMs. `n`: their total,
                // i.e. how far `sa_cursor` must advance. `rbegs`: this read's own slice of the flat
                // resolved-position array, in the order the chain builder expects.
                let counts = &per_read_counts[ridx];
                let n: usize = counts.iter().map(|c| *c as usize).sum();
                let rbegs = &all_rbegs[sa_cursor..sa_cursor + n];
                sa_cursor += n;
                bwa_chain::build_chains_from_resolved(bns, opt, codes, 0, &smems, counts, rbegs)
            } else {
                build_chains_from_smems(fm, bns, opt, codes, 0, smems)
            };
            if dump_chains {
                eprintln!("PRECHAIN nchains={}", pre.len());
                for (ci, c) in pre.iter().enumerate() {
                    eprintln!(
                        "  prechain{ci} pos={} nseed={} w={} qbeg={} qend={}",
                        c.pos,
                        c.seeds.len(),
                        bwa_chain::mem_chain_weight(c),
                        c.seeds.first().map_or(0, |s| s.qbeg),
                        c.seeds.last().map_or(0, |s| s.qbeg + s.len),
                    );
                }
            }
            // The kept chains: overlapping and dominated chains dropped, `kept`/`w` filled in. This
            // is the value the closure yields into `per_read_chains`.
            let out = mem_chain_flt(opt, pre);
            if dump_chains {
                eprintln!("CHAIN nchains={}", out.len());
                for (ci, c) in out.iter().enumerate() {
                    eprintln!(
                        "  chain{ci} pos={} nseed={} w={} kept={}",
                        c.pos,
                        c.seeds.len(),
                        c.w,
                        c.kept
                    );
                }
            }
            out
        })
        .collect();

    // The output under construction: per read, one region slot per seed, in slot order (chains back
    // to back, seeds within a chain in descending-score order). Filled skeleton-first by the
    // collection pass, completed by `run_side`'s scatter, then compacted at the very end.
    let mut regs: Vec<Vec<MemAlnReg>> = vec![Vec::new(); reads.len()];
    // region -> owning chain index (for the final seedcov pass).
    let mut reg_chain: Vec<Vec<usize>> = vec![Vec::new(); reads.len()];
    // Per-region (chain, order position, seed) for the discard pass.
    let mut reg_meta: Vec<Vec<RegMeta>> = vec![Vec::new(); reads.len()];
    // Seeds whose extension nh13's skip elided: they keep a slot (so the discard pass reproduces
    // bwa-mem2's scan order) but start purged.
    let mut reg_preskip: Vec<Vec<bool>> = vec![Vec::new(); reads.len()];

    // The two pending-extension queues, pooled across the WHOLE batch rather than per read: that is
    // what gives the SIMD kernel enough same-length work to fill its lanes. A seed contributes a left
    // job only if it has query bases before it, and a right job only if it has query bases after it,
    // so neither queue is one-per-seed. Both are consumed (and drained via `active`) by `run_side`.
    let mut left_jobs: Vec<SideJob> = Vec::new();
    let mut right_jobs: Vec<SideJob> = Vec::new();

    // Skip banded-SW for same-diagonal contained seeds (nh13 --skip-contained-ext). The discard pass
    // below needs one slot per seed to reproduce bwa-mem2's scan order, so a skipped seed would shift
    // every later slot; the two are mutually exclusive for now.
    //
    // UNVERIFIED / likely stale: the last sentence above describes an earlier design. As the code
    // stands the two features run together (both flags are read here and both are on by default),
    // and the conflict is resolved by `reg_preskip`: a skipped seed still gets a slot, it just starts
    // purged. The `skip_contained_enabled` doc comment describes the current arrangement. Left in
    // place rather than deleted because it is not certain which of the two is the intended contract.
    // The two environment gates, resolved once for the whole batch. `purge`: run bwa-mem2's discard
    // pass after extension (default true; false makes output non-byte-identical). `skip_contained`:
    // elide the DP for same-diagonal contained seeds (default true, byte-identical).
    let purge = discard_enabled();
    let skip_contained = skip_contained_enabled();

    // ---- collection pass: one region skeleton + up to one left and one right job per seed ----
    for (r, codes) in reads.iter().enumerate() {
        // This read's length in bases, and its surviving chains. `l_query` is the 3' limit every
        // right-side decision below compares against.
        let l_query = codes.len() as i32;
        let chains = &per_read_chains[r];
        for (ci, chain) in chains.iter().enumerate() {
            if chain.seeds.is_empty() {
                continue;
            }
            // Reference window spanning the chain (mirrors mem_chain2aln, `bwamem.cpp:2144-2166`).
            // Start the running min at the maximum possible coordinate and the running max at 0, so
            // the first seed sets both. `l_pac << 1` is the end of the forward+reverse concatenation.
            // Running window bounds, reference positions in `0..2*l_pac`. Invariant at the top of
            // each seed iteration: `[rmax0, rmax1)` already covers every reach computed for the
            // seeds seen so far. Deliberately started inverted (min at the maximum coordinate, max at
            // 0) so the first seed replaces both; the pair is meaningless if the chain has no seeds,
            // which the guard above rules out.
            let mut rmax0 = l_pac << 1;
            let mut rmax1 = 0i64;
            for seed in &chain.seeds {
                // How far left the alignment could possibly reach: the `qbeg` unaligned query bases
                // ahead of the seed, plus room for the longest indel those bases could pay for.
                let win_beg =
                    seed.rbeg - (i64::from(seed.qbeg) + i64::from(cal_max_gap(opt, seed.qbeg)));
                // Symmetrically to the right, over the `tail` bases after the seed: `tail` is the
                // count of unaligned query bases past the seed's 3' end.
                let tail = l_query - seed.qbeg - seed.len;
                let win_end = seed.rbeg
                    + i64::from(seed.len)
                    + (i64::from(tail) + i64::from(cal_max_gap(opt, tail)));
                rmax0 = rmax0.min(win_beg);
                rmax1 = rmax1.max(win_end);
            }
            rmax0 = rmax0.max(0);
            rmax1 = rmax1.min(l_pac << 1);
            // A window must not straddle the forward/reverse boundary at `l_pac`: positions on the
            // two sides are unrelated sequence, so a DP crossing it would align a read against its
            // own reverse complement. Clamp to whichever half the chain's first seed lives in
            // (`bwamem.cpp:2168-2172`). Using `seeds[0]` rather than a majority vote is bwa's
            // choice; all seeds of a chain are same-strand by construction.
            if rmax0 < l_pac && l_pac < rmax1 {
                if chain.seeds[0].rbeg < l_pac {
                    rmax1 = l_pac;
                } else {
                    rmax0 = l_pac;
                }
            }
            // `bns_fetch_seq`: trim the window to the seed's contig so extension cannot run off its
            // end into the next contig's sequence (visible on the circular MT genome). The rebinding
            // shadows the mutable pair above with the final immutable window bounds.
            let (rmax0, rmax1, _rid) = bns.fetch_bounds(rmax0, rmax1, chain.seeds[0].rbeg);
            // Materialise the window's bases once per chain, so both sides of every seed slice out of
            // it. `fm.base` unpacks the 2-bit pac; the C gets the same bytes from `bns_fetch_seq_v2`
            // (`bwamem.cpp:2175-2181`). Indices into `rseq` are therefore `position - rmax0`.
            let rseq: Vec<u8> = (rmax0..rmax1).map(|p| fm.base(p)).collect();

            // Seeds in descending (score, index) order.
            //
            // bwa-mem2 builds `srt[i] = (uint64_t)score<<32 | i` (`bwamem.cpp:2189`), introsorts it
            // *ascending*, and walks it backwards (`bwamem.cpp:2205`). We sort descending directly,
            // which is the same permutation. Two details are load-bearing:
            //   - packing the index into the low 32 bits makes the order total, so ties in score are
            //     broken by seed index and the sort's own stability never matters. That is exactly
            //     why we can use `sort_by_key` here while the dedup elsewhere must stay unstable to
            //     match the C.
            //   - `score as u32` before widening: `score` is i32 and the C reinterprets it as the
            //     high half of a u64, so a (never observed in practice) negative score would sort as
            //     a huge positive one. Reproducing the reinterpretation rather than sign-extending is
            //     the conservative choice.
            let mut order: Vec<usize> = (0..chain.seeds.len()).collect();
            order.sort_by_key(|&i| {
                std::cmp::Reverse((u64::from(chain.seeds[i].score as u32) << 32) | i as u64)
            });

            for (pos, &si) in order.iter().enumerate() {
                if skip_contained && seed_ext_redundant(&chain.seeds, si) {
                    // Keep the slot, skip the DP: the discard pass would purge this seed anyway
                    // (its container is a longer same-diagonal seed, extended earlier).
                    reg_chain[r].push(ci);
                    reg_meta[r].push(RegMeta {
                        chain: ci as u32,
                        pos: pos as u32,
                        seed: si as u32,
                    });
                    reg_preskip[r].push(true);
                    // A placeholder region, born purged. Bounds are `-1` (the discard pass's
                    // tombstone) rather than `H0_SENTINEL`, so the seedcov pass skips it, the
                    // discard pass's `qb == -1 && qe == -1` check skips it, and the final
                    // `retain(|reg| reg.qe > reg.qb)` drops it. `seedlen0` and `w` are still filled
                    // in faithfully: they are cheap, and leaving them at 0 would be a trap if the
                    // purge is ever disabled.
                    regs[r].push(MemAlnReg {
                        rb: -1,
                        re: -1,
                        qb: -1,
                        qe: -1,
                        rid: chain.rid,
                        score: -1,
                        truesc: -1,
                        sub: 0,
                        csub: 0,
                        sub_n: 0,
                        seedcov: 0,
                        seedlen0: chain.seeds[si].len,
                        secondary: -1,
                        secondary_all: -1,
                        w: opt.w,
                        frac_rep: chain.frac_rep,
                        is_alt: chain.is_alt,
                        // Set only by the ALT branch of mem_mark_primary_se; 0 everywhere it is constructed.
                        alt_sc: 0,
                        hash: 0,
                        n_comp: 1,
                    });
                    continue;
                }
                let seed = chain.seeds[si];
                // Region skeleton (the C's `a`), matching the `memset(a, 0, ...)` plus explicit field
                // writes at `bwamem.cpp:2211-2222`. All four bounds (`qb`/`qe` query offsets,
                // `rb`/`re` reference positions) start at `H0_SENTINEL` (bwa's `H0_`, -99): a value
                // no real coordinate can take, so the seedcov pass and the C's scatter guards can
                // tell "never written" from "written to 0". Whichever side has no extension job
                // overwrites its pair of bounds immediately below; the other pair is filled by the
                // scatter in `run_side`.
                let mut reg = MemAlnReg {
                    rb: H0_SENTINEL,
                    re: H0_SENTINEL,
                    qb: H0_SENTINEL as i32,
                    qe: H0_SENTINEL as i32,
                    rid: chain.rid,
                    score: -1,
                    truesc: -1,
                    sub: 0,
                    csub: 0,
                    sub_n: 0,
                    seedcov: 0,
                    seedlen0: seed.len,
                    secondary: -1,
                    secondary_all: -1,
                    w: opt.w,
                    frac_rep: chain.frac_rep,
                    is_alt: chain.is_alt,
                    // Set only by the ALT branch of mem_mark_primary_se; 0 everywhere it is constructed.
                    alt_sc: 0,
                    hash: 0,
                    n_comp: 1,
                };

                // ---- left extension job, or seed-terminal left edge (`bwamem.cpp:2245-2322`) ----
                if seed.qbeg > 0 {
                    // Both sequences are reversed so the DP runs outward from the seed. Query: the
                    // `qbeg` bases before the seed, last-first (C: `qs[i] = query[s->qbeg-1-i]`,
                    // `bwamem.cpp:2286`). Target: the `rbeg - rmax0` window bases before the seed,
                    // likewise reversed (C: `rs[i] = rseq[tmp-1-i]`, `bwamem.cpp:2299`).
                    let query: Vec<u8> = (0..seed.qbeg).rev().map(|i| codes[i as usize]).collect();
                    // Bases of the window lying left of the seed, i.e. how much reference the left
                    // extension may consume. Non-negative because `rmax0 <= seed.rbeg` by
                    // construction of the window above.
                    let rlen = (seed.rbeg - rmax0) as usize;
                    let target: Vec<u8> = (0..rlen).rev().map(|i| rseq[i]).collect();
                    // Provisional bounds at the seed's left edge; `run_side` walks them further left
                    // by subtracting the DP's `qle` (query length consumed) / `tle` (target length
                    // consumed).
                    reg.qb = seed.qbeg;
                    reg.rb = seed.rbeg;
                    left_jobs.push(SideJob {
                        read: r,
                        // `regs[r].len()` is this region's future index: `reg` has not been pushed
                        // yet, and every path below pushes exactly once, so this is the slot it will
                        // take.
                        reg: regs[r].len(),
                        query,
                        target,
                        // The seed itself scores `len` matches at `opt.a` each; the extension starts
                        // from there so z-drop and clipping are judged against the whole alignment.
                        h0: seed.len * opt.a,
                        prev: -1,
                        active: true,
                    });
                } else {
                    // Seed already flush with the read's 5' end: nothing to extend, so the region is
                    // final on this side and its score is just the seed's (`bwamem.cpp:2319-2321`).
                    reg.score = seed.len * opt.a;
                    reg.truesc = reg.score;
                    reg.qb = 0;
                    reg.rb = seed.rbeg;
                }

                // ---- right extension job, or seed-terminal right edge (`bwamem.cpp:2324-2431`) ----
                if seed.qbeg + seed.len != l_query {
                    // Query offset just past the seed's 3' end: where the right extension starts, and
                    // the region's provisional `qe` before the DP walks it further right.
                    let qe = seed.qbeg + seed.len;
                    // Careful: this local `re` is an offset *into `rseq`*, not a reference position
                    // (unlike the region field of the same name). The window starts at `rmax0`, so
                    // subtracting it rebases the seed's right edge (C: `bwamem.cpp:2327`, which
                    // asserts `re >= 0`). Both slices then run to the end of their buffer, which is
                    // the C's `len2 = l_query - qe` and `len1 = rmax[1] - rmax[0] - re`.
                    let re = seed.rbeg + i64::from(seed.len) - rmax0;
                    let query: Vec<u8> = codes[qe as usize..].to_vec();
                    let target: Vec<u8> = rseq[re as usize..].to_vec();
                    reg.qe = qe;
                    // Back to an absolute reference position for the region's bound.
                    reg.re = rmax0 + re;
                    right_jobs.push(SideJob {
                        read: r,
                        reg: regs[r].len(),
                        query,
                        target,
                        h0: H0_SENTINEL as i32, // filled from reg.score after left completes
                        prev: -1,
                        active: true,
                    });
                } else {
                    // Seed flush with the read's 3' end. Unlike the left case this writes no score:
                    // the score is whatever the left extension produced (or was set above).
                    reg.qe = l_query;
                    reg.re = seed.rbeg + i64::from(seed.len);
                }

                // Push metadata and region together, in lockstep, so the four per-read vectors stay
                // index-aligned. `discard_contained` relies on that alignment.
                reg_chain[r].push(ci);
                reg_meta[r].push(RegMeta {
                    chain: ci as u32,
                    pos: pos as u32,
                    seed: si as u32,
                });
                reg_preskip[r].push(false);
                regs[r].push(reg);
            }
        }
    }

    // ---- left extensions (h0 already set), then fill right h0 and run right extensions ----
    // `pen_clip5` for the left side, `pen_clip3` for the right: bwa constructs two
    // `BandedPairWiseSW` objects differing only in that penalty (`bwamem.cpp:2452-2458`). The
    // penalty decides whether a soft clip beats running the alignment to the end of the query.
    run_side(backend, opt, &mut left_jobs, &mut regs, opt.pen_clip5, true);

    // The hard serialisation point, and the reason there are two job arrays rather than one: a right
    // extension resumes from the score the left extension reached, so every left job in the whole
    // batch must have landed before any right job can even be parameterised. bwa-mem2 does the same
    // fixup in its own loop at `bwamem.cpp:2672-2677` (`sp->h0 = a->score;`).
    for job in &mut right_jobs {
        job.h0 = regs[job.read][job.reg].score;
        // `prev` starts at the region's current score, not -1, because that is literally what the C
        // reads: `int prev = a->score;` (`bwamem.cpp:2492`) runs on every band retry including the
        // first, and by now `a->score` holds the left extension's result. For a *left* job the
        // region still carries the -1 it was initialised with (`bwamem.cpp:2218`), which is why -1
        // remains right there.
        //
        // This used to be -1 on both sides, justified by an equivalence argument: a right extension
        // whose round-0 score equals `h0` gained nothing, and `ksw_extend2` only updates `max_off`
        // inside `if (m > max)` with `max` starting at `h0` (`ksw.cpp:479-480`), so `max_off` is
        // still 0 and acceptance test 2 fires anyway. That argument is sound, but it has a hole:
        // test 2 reads `max_off < (w >> 1) + (w >> 2)`, which for `w <= 1` is `0 < 0`, false. With
        // `-w 0` or `-w 1` the C accepts at round 0 via test 1 while we would retry at double the
        // band and could return a different alignment. Reproducing the C removes the reasoning
        // instead of narrowing it.
        job.prev = job.h0;
    }
    run_side(
        backend,
        opt,
        &mut right_jobs,
        &mut regs,
        opt.pen_clip3,
        false,
    );

    // ---- seedcov, per region, from final bounds (mirrors mem_chain2aln's tail) ----
    // `seedcov` is the number of query bases covered by seeds that fall entirely inside the finished
    // region. It feeds MAPQ and the `mem_sort_dedup_patch` redundancy test, so it must be exact.
    //
    // bwa-mem2 computes this three separate times (`bwamem.cpp:2419-2431` for the seed-terminal case,
    // `2506-2516` in the left scatter, `2723-2732` in the right scatter), each guarded by all four
    // bounds being off `H0_`. Only the last execution survives, and by then the bounds are final, so
    // hoisting it to one pass here reaches the same values with a third of the work. The guard is the
    // same one: skip regions whose extension never landed, and skip `-1` (a pre-skipped slot; the
    // discard pass has not run yet, so `-1` here can only come from `reg_preskip`).
    for r in 0..reads.len() {
        for (idx, reg) in regs[r].iter_mut().enumerate() {
            if reg.rb != H0_SENTINEL && reg.qb != H0_SENTINEL as i32 && reg.qb != -1 {
                let chain = &per_read_chains[r][reg_chain[r][idx]];
                reg.seedcov = 0;
                // Every seed of the *owning chain*, not just the one that founded this region: a
                // wide extension can swallow several of its chain's seeds.
                for seed in &chain.seeds {
                    if seed.qbeg >= reg.qb
                        && seed.qbeg + seed.len <= reg.qe
                        && seed.rbeg >= reg.rb
                        && seed.rbeg + i64::from(seed.len) <= reg.re
                    {
                        reg.seedcov += seed.len;
                    }
                }
            }
        }
    }

    // ---- bwa-mem2's discard pass, after every extension has landed ----
    if purge {
        for r in 0..reads.len() {
            let l_query = reads[r].len() as i32;
            discard_contained(
                opt,
                l_query,
                &per_read_chains[r],
                &mut regs[r],
                &reg_meta[r],
                &reg_preskip[r],
            );
        }
    }

    // Drop the purged regions before returning, as bwa-mem2 does between the discard pass and
    // `mem_sort_dedup_patch`. Not cosmetic: the dedup sorts by `re` with an *unstable* introsort, so
    // leaving purged entries in would change the array it partitions and hence the order of
    // equal-`re` regions -- which is exactly what decides who survives a score tie.
    for read_regs in regs.iter_mut() {
        read_regs.retain(|reg| reg.qe > reg.qb);
    }

    regs
}

/// Whether bwa-mem2's post-extension discard pass runs (`BWA4_NO_DISCARD` opts out).
///
/// Cached in a `OnceLock` for the same reason as [`skip_contained_enabled`]: the batched and
/// per-read extension paths must make the identical decision, and re-reading the environment per
/// read would also cost a lock per call. Turning it off is a debugging aid only: the output is then
/// *not* byte-identical to bwa-mem2, because the surviving near-duplicate regions inflate `sub` and
/// collapse MAPQ, as noted on [`discard_contained`].
///
/// # Parameters
///
/// None. The single input is the process environment, read exactly once.
///
/// # Returns
///
/// True when the discard pass should run (the default: the variable is absent), false when
/// `BWA4_NO_DISCARD` is set to any value, including the empty string. Frozen after the first call.
pub(crate) fn discard_enabled() -> bool {
    // Process-wide memo of the environment decision; see `skip_contained_enabled` for why this is a
    // `OnceLock` rather than a per-call `var_os`.
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("BWA4_NO_DISCARD").is_none())
}

/// Run one side (left or right) of all pending extensions through `MAX_BAND_TRY` band-doubling
/// rounds. Each round batches the still-active jobs (sorted by length so SIMD lanes pack tightly),
/// applies the exact `extend_side` acceptance test, and scatters accepted results into their region.
///
/// The band-doubling idea: banded DP only explores `w` cells either side of the main diagonal, so an
/// alignment whose optimum wandered near the band edge may be an artefact of the band. Re-run at
/// double the width; if the score stops changing, or the optimum stayed comfortably inside, trust it.
/// `MAX_BAND_TRY` is 2 (`bwamem.cpp:51`), so at most one retry ever happens.
///
/// # Parameters
///
/// * `backend`: the batched Smith-Waterman kernel. Invoked once per round with all of that round's
///   still-active jobs; its results must depend only on each job's own `(query, target, h0, w)`.
/// * `opt`: alignment options. Read here for `w` (the round-0 band half-width), `mat`, the four gap
///   penalties, and `zdrop`. Never mutated.
/// * `jobs`: all pending jobs for this side of the whole batch, mutated in place, and only in the
///   two fields `prev` (carried forward on a retry) and `active` (cleared on acceptance). Jobs that
///   arrive already inactive are ignored, so the same slice could in principle be re-run. On return
///   every job has `active == false` unless the slice was empty.
/// * `regs`: the whole batch's regions, indexed `regs[job.read][job.reg]`, i.e. the scatter target.
///   Each accepted job writes `score`, one pair of bounds, `truesc`, and `w` on exactly one region;
///   `job.reg` was fixed at collection time so the indices survive the per-round length sorts.
/// * `pen_clip`: soft-clip penalty in score units, `opt.pen_clip5` for the left side and
///   `opt.pen_clip3` for the right. It is the margin by which a query-consuming ("global") alignment
///   must beat the local one to be preferred.
/// * `is_left`: true for the left pass, false for the right. Selects the coordinate update: left
///   walks `qb`/`rb` backwards and *assigns* `truesc`; right walks `qe`/`re` forwards and
///   *accumulates* into `truesc`. Getting this branch wrong is silent: the scores stay plausible and
///   only some MAPQs move. Must agree with which queue was passed as `jobs`.
fn run_side<B: SwBackend>(
    backend: &B,
    opt: &MemOpt,
    jobs: &mut [SideJob],
    regs: &mut [Vec<MemAlnReg>],
    pen_clip: i32,
    is_left: bool,
) {
    for round in 0..MAX_BAND_TRY {
        // ---- step 1: gather this round's still-active jobs ----
        // Collect active job indices, sorted by length to cluster similar sizes in a batch.
        // Indices into `jobs` of everything still needing a pass. Invariant at the top of each round:
        // it holds exactly the jobs no acceptance test has yet passed, so it shrinks monotonically
        // and the loop stops as soon as it is empty.
        let mut active_idxs: Vec<usize> = (0..jobs.len()).filter(|&k| jobs[k].active).collect();
        if active_idxs.is_empty() {
            break;
        }
        // Purely a throughput sort: a SIMD batch runs for as long as its longest lane, so grouping
        // similar lengths cuts wasted lanes. bwa-mem2 does the same with a histogram sort,
        // `sortPairsLenExt` (`bwamem.cpp:2446`, and again for the right side at `bwamem.cpp:2679`),
        // which also keys on length. Unobservable in the output because each job's result depends
        // only on its own `(query, target, h0, w)`, which is the invariant this whole file rests on.
        active_idxs
            .sort_by_key(|&k| std::cmp::Reverse(jobs[k].query.len().max(jobs[k].target.len())));

        // ---- step 2: run the batch at this round's band width ----
        // Band width for this round: `opt.w`, then `2*opt.w` (`bwamem.cpp:2474`).
        // Band half-width in cells for this round: `opt.w`, then `2*opt.w` (`bwamem.cpp:2474`). Every
        // job in the round shares it, and it is also what the acceptance test and `reg.w` record.
        let w = opt.w << round;
        // The kernel's view of this round's work, parallel to `active_idxs` (NOT to `jobs`): lane
        // `i` of `batch` is job `active_idxs[i]`, which is how the results are scattered back.
        let batch: Vec<ExtendJob> = active_idxs
            .iter()
            .map(|&k| ExtendJob {
                query: &jobs[k].query,
                target: &jobs[k].target,
                // `h0`: the score the seed (plus, for right jobs, the left extension) already earned.
                h0: jobs[k].h0,
            })
            .collect();
        // `opt.mat` is the flattened `ALPHABET_SIZE x ALPHABET_SIZE` substitution matrix.
        // `results` is parallel to `batch`, hence to `active_idxs`, not to `jobs`.
        let results = backend.extend_batch(
            &batch,
            ALPHABET_SIZE,
            &opt.mat,
            opt.o_del,
            opt.e_del,
            opt.o_ins,
            opt.e_ins,
            w,
            pen_clip,
            opt.zdrop,
        );

        // ---- step 3: acceptance test, then scatter or requeue each job ----
        for (lane, &job_idx) in active_idxs.iter().enumerate() {
            // `res`: this lane's DP outcome (`score`, `qle`/`tle`, `gscore`/`gtle`, `max_off`; see
            // the module glossary). `prev`: the score this job reached in the previous round, or its
            // seed value for round 0 (-1 for a left job, the left extension's score for a right one).
            // `score`: this round's best local score, including the `h0` head start.
            let res = &results[lane];
            let prev = jobs[job_idx].prev;
            let score = res.score;
            // The acceptance test, character for character `bwamem.cpp:2495-2496`. Three ways in:
            //   1. `score == prev`: doubling the band did not improve anything, so it is converged.
            //   2. `max_off < (w>>1) + (w>>2)`: the optimum's furthest excursion from the main
            //      diagonal stayed under 3/4 of the band (`w/2 + w/4`, written as shifts because the
            //      C does and because it avoids the rounding of `0.75 * w`). Comfortably interior,
            //      so a wider band would not have found more.
            //   3. Out of retries.
            //
            // `prev` here is per-job and starts at -1, matching the per-read [`crate::extend_side`]
            // (lib.rs). bwa-mem2 instead reads `int prev = a->score;` off the region
            // (`bwamem.cpp:2492`), which for a *right* job at round 0 is the left extension's score,
            // i.e. equal to `h0`, not -1. The two differ only when a right extension's round-0 score
            // is exactly `h0`, meaning it extended nothing, in which case `max_off` is 0 and test 2
            // accepts anyway.
            // UNVERIFIED: that argument is reasoning about ksw_extend's postconditions, not a proof,
            // and it has not been checked against an instrumented bwa-mem2 run. The 1M-read /
            // 1M-pair byte-identity results are the empirical evidence that it does not bite.
            // True when this round's result is final for the job: it is then scattered into its
            // region and the job goes inactive. False means one more round at double the band.
            let accept =
                score == prev || res.max_off < (w >> 1) + (w >> 2) || round + 1 == MAX_BAND_TRY;
            if !accept {
                // Retry at the wider band. Note we do NOT write `score` into the region, unlike the
                // C, which stores `a->score = sp->score` unconditionally before testing; that store
                // is how the C carries `prev` forward, and we carry it in the job instead.
                jobs[job_idx].prev = score;
                continue;
            }
            jobs[job_idx].active = false;
            // The one region this job fills. Exclusive for the duration of the scatter: at most one
            // left job and one right job ever name a given slot, and the two sides run in separate
            // calls, so no other job in this round touches it.
            let reg = &mut regs[jobs[job_idx].read][jobs[job_idx].reg];
            // Local vs global ("glocal") choice, `bwamem.cpp:2498-2504` (left) and `2714-2722`
            // (right). `score`/`qle`/`tle` describe the best *local* endpoint; `gscore`/`gtle`
            // describe the best alignment that consumes the entire remaining query. Prefer the
            // global one only when it exists (`gscore > 0`) and beats the local score by more than
            // the clipping penalty would cost, i.e. `gscore > score - pen_clip`. Written as the
            // negation so the local branch is first, exactly as the C has it.
            if is_left {
                reg.score = score;
                if res.gscore <= 0 || res.gscore <= score - pen_clip {
                    // Local: walk the region's start back by the consumed query/target lengths.
                    reg.qb -= res.qle;
                    reg.rb -= i64::from(res.tle);
                    reg.truesc = score;
                } else {
                    // Global: the alignment reaches the read's 5' end, so `qb` is 0 by definition
                    // and only the reference side needs the `gtle` walk-back.
                    reg.qb = 0;
                    reg.rb -= i64::from(res.gtle);
                    reg.truesc = res.gscore;
                }
            } else {
                // The score this right extension started from, i.e. the left extension's result. It
                // is the part of `score` this side did not earn, so both `truesc` updates below
                // subtract it.
                let h0 = jobs[job_idx].h0;
                reg.score = score;
                if res.gscore <= 0 || res.gscore <= score - pen_clip {
                    reg.qe += res.qle;
                    reg.re += i64::from(res.tle);
                    // `truesc` accumulates rather than assigns on the right side: the DP restarted
                    // from `h0` (the left extension's score), so `score - h0` is this side's own
                    // contribution. Subtracting `h0` is what stops the seed's score being counted
                    // twice (C: `a->truesc += a->score - sp->h0`, `bwamem.cpp:2717`).
                    reg.truesc += score - h0;
                } else {
                    // qe = l_query: the region's read length. `reg.qe` was set to
                    // (seed.qbeg + seed.len) at collection; the query slice length is
                    // (l_query - qe), so adding it restores l_query.
                    //
                    // The C writes `a->qe = l_query` directly (`bwamem.cpp:2719-2720`), reading the
                    // read length back out of `seq_[sp->seqid].l_seq`. We have no read handle inside
                    // `run_side`, so we add the slice length instead. Identical by the arithmetic
                    // above, and it avoids threading `reads` through purely for this one line.
                    reg.qe += jobs[job_idx].query.len() as i32;
                    reg.re += i64::from(res.gtle);
                    reg.truesc += res.gscore - h0;
                }
            }
            // Record the widest band this region was ever aligned with. The discard pass reads it
            // back as `prior.w` to bound how far a later seed may sit off that region's diagonal, so
            // it must be the accepted round's `w`, not `opt.w`
            // (C: `a->w = max_(a->w, w)`, `bwamem.cpp:2505`).
            reg.w = reg.w.max(w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align_read;
    use bwa_neon::NeonBackend;
    use std::path::Path;

    /// Load the checked-in toy index. Small enough that 400 reads exercise every branch quickly,
    /// large enough that chains carry several seeds (which is what the discard pass needs).
    ///
    /// # Returns
    ///
    /// The index and the `.ann`/`.amb` metadata for `testdata/tiny/tiny.fa`, both loaded from the
    /// checked-in files. Panics if they are missing or malformed, which is the wanted behaviour in a
    /// test.
    fn tiny() -> (FmIndex, BntSeq) {
        // Path stem shared by every index file (`.bwt`, `.sa`, `.ann`, ...), resolved at compile
        // time from the crate directory so the test does not depend on the working directory.
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        (
            FmIndex::load(Path::new(prefix)).unwrap(),
            BntSeq::load(Path::new(prefix)).unwrap(),
        )
    }

    /// The batched across-reads path (through the NEON backend) must produce, for every read, the
    /// exact same regions as calling `align_read` per read. A diverse read set (forward / RC slices,
    /// mismatches, insertions, deletions, truncations) exercises left+right extension, band-doubling,
    /// and the gscore/z-drop branches.
    #[test]
    fn batched_across_reads_equals_per_read() {
        let (fm, bns) = tiny();
        // Stock CLI defaults: the parity claim is about the shipped configuration.
        let opt = MemOpt::default();
        // Forward-strand length of the toy reference, the upper bound on where a synthetic read may
        // start.
        let l_ref = bns.l_pac;

        // A fixed-seed LCG (Knuth's MMIX multiplier/increment) rather than `rand`: the read set must
        // be identical on every run and every machine, so a failure is reproducible from the seed
        // alone. `>> 33` discards the low bits, whose period is short in any LCG.
        // The generator's whole state: an arbitrary but fixed odd seed. Changing it changes the read
        // set and therefore which branches the test happens to cover.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        // Draws the next pseudo-random value, in `0..2^31`. Every read length, position, mutation
        // site, and orientation below comes from this one stream, so the whole read set is a
        // function of the seed alone.
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        // The synthetic read set under test: 2-bit codes, one entry per read, built below.
        let mut reads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..400 {
            let len = 60 + (next() % 120) as i64; // 60..180
                                                  // Reference offset the read is copied from. `.max(1)` keeps the modulus positive on a
                                                  // reference barely longer than the read.
            let start = (next() as i64) % (l_ref - len - 1).max(1);
            // The read under construction: an exact copy of the reference slice, perturbed below.
            let mut r: Vec<u8> = (0..len).map(|i| fm.base(start + i)).collect();
            // Perturb: mismatches, an insertion, a deletion.
            // Number of substitutions to apply to this read, 0..=5.
            let nmut = (next() % 6) as usize;
            for _ in 0..nmut {
                // Offset in the read to mutate.
                let p = (next() as usize) % r.len();
                // `+1 + rand%3` then mod 4 guarantees a *different* base: the offset is 1..=3, so it
                // can never land back on the original code. A plain `rand%4` would silently produce
                // no-op mutations a quarter of the time and weaken the test.
                r[p] = ((r[p] as u64 + 1 + next() % 3) % 4) as u8;
            }
            if next() % 3 == 0 && r.len() > 20 {
                let p = (next() as usize) % r.len(); // offset the extra base is spliced in before
                r.insert(p, (next() % 4) as u8); // insertion
            }
            if next() % 3 == 0 && r.len() > 20 {
                let p = (next() as usize) % r.len(); // offset of the base to drop
                r.remove(p); // deletion
            }
            // Reverse-complement half of them.
            if next() % 2 == 0 {
                r = r
                    .iter()
                    .rev()
                    .map(|&c| if c < 4 { 3 - c } else { c })
                    .collect();
            }
            reads.push(r);
        }

        // The path under test: every read's regions computed in one batched pass.
        let batched = align_reads_batched(&fm, &bns, &opt, &reads, &NeonBackend);
        assert_eq!(batched.len(), reads.len());
        for (i, codes) in reads.iter().enumerate() {
            // The reference answer for this read, from the per-read extension path.
            let per_read = align_read(&fm, &bns, &opt, codes);
            // Compared through `Debug` strings, not `PartialEq`: `MemAlnReg` carries f32 fields
            // (`frac_rep`) where we want exact bit equality and a readable diff on failure. The
            // formatted form gives both, and it also catches a field being added without the
            // comparison being updated.
            assert_eq!(
                format!("{:?}", batched[i]),
                format!("{:?}", per_read),
                "read {i} (len {}) diverged between batched and per-read extension",
                codes.len()
            );
        }
    }
}
