//! Seed chaining and chain filtering, mirroring bwa-mem2's `mem_chain` / `test_and_merge` /
//! `mem_chain_weight` / `mem_chain_flt` (`reference/bwa-mem2/src/bwamem.cpp`).
//!
//! Chains collinear seeds into candidate alignments, then filters by weight and overlap. The
//! end-to-end byte-identity gate is the SE SAM concordance in phase 6.
//!
//! # What chaining is for
//!
//! Seeding ([`bwa_seed`]) hands us a pile of *exact* matches: for one read, typically tens to
//! hundreds of `(read offset, reference offset, length)` triples, scattered over every repeat copy
//! and every spurious short hit in the genome. A real alignment is not one seed, it is a run of
//! seeds that are **collinear**: increasing in read coordinate and in reference coordinate together,
//! at roughly the same rate (small differences are indels). A *chain* is such a run. Chaining turns
//! the pile into a handful of candidate alignment loci, each of which the extension stage then fills
//! in with Smith-Waterman. Without it, the aligner would run DP hundreds of times per read.
//!
//! The whole pass is a single greedy sweep (`mem_chain`, `bwamem.cpp:~880-950`): seeds arrive in
//! read order, each is offered to *one* existing chain (the one whose start position is the closest
//! at or below the seed's), and if that chain refuses, the seed starts a new chain. There is no
//! optimal-chaining DP here, unlike minimap2.
//!
//! # Why the collection type is a hand-written B-tree
//!
//! "The chain whose start position is closest at or below" is looked up in a klib `kbtree` keyed on
//! `mem_chain_t::pos`, and `chain_cmp` compares `pos` alone, so the tree **permits duplicate keys**.
//! When several chains start at the same reference position, which one the lookup returns depends on
//! klib's node layout and split history, not on insertion order. Since only that one chain is
//! offered the seed, the answer is observable in the output. See [`kbtree`] for the ported quirks.
//!
//! # Coordinates and units used throughout
//!
//! - Reference positions (`rbeg`, `MemChain::pos`) are in the 2L forward++RC space: `< l_pac` is
//!   forward strand, `>= l_pac` is reverse. `l_pac` comes from the `.ann`/`.amb` via `BntSeq`.
//! - Query positions (`qbeg`) are 0-based offsets into the read.
//! - Lengths and weights are in bases.
//!
//! # Glossary: the names deliberately kept identical to the C
//!
//! These are NOT renamed. Diffing this file line by line against `bwamem.cpp` is the workflow that
//! has found every parity bug in this project, and renaming breaks the diff. In plain language:
//!
//! | name | C origin | plain-language meaning |
//! |---|---|---|
//! | `l_pac` | `bns->l_pac` | Length of the FORWARD reference in bases. The searched text is `2 * l_pac` long: the forward genome followed by its reverse complement. A position `>= l_pac` is therefore a reverse-strand hit, which is why strand tests here are plain coordinate comparisons. |
//! | `rid` | `bns_intv2rid` | Reference-sequence index, i.e. which contig (chromosome) a position falls in. Negative means "no single contig": the interval bridges a contig boundary (-1) or the forward/reverse seam at `l_pac` (-2), and bwa discards such seeds. |
//!
//! # Rust mechanics used in this file
//!
//! Two things here look strange to a reader coming from the C and are worth flagging up front.
//! First, the sort routines are ports of klib's, and they take the comparison as a FUNCTION passed
//! in by the caller, so one sort implementation serves every ordering in the file. Second, the
//! reorder step near the end of `build_chains_from_resolved` moves values out of a vector one at a
//! time in an arbitrary order, which Rust does not permit directly; the `Vec<Option<_>>` there is
//! the standard way around it, explained at the site.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `&MemChain` / `&mut MemChain` | a borrow of one chain: read-only, or exclusive and writable. A function taking `&mut` modifies the caller's chain in place and returns nothing. |
//! | `&mut [T]` | an exclusive borrow of a contiguous run of values. The sorts take this, so they reorder the caller's storage rather than producing a new list. |
//! | `fn f<T>(...)` | a GENERIC function: `T` stands for whatever element type the caller has. One compiled copy is produced per type actually used, so this is not slower than writing it out by hand. |
//! | `impl Fn(&T, &T) -> bool` | a parameter that is itself a function: "give me something callable that takes two elements and says whether the first sorts before the second". It is how the klib sorts stay one implementation while their orderings differ per call site. |
//! | `\|x, y\| x.w > y.w` | an anonymous inline function written at the call site, supplying exactly that comparison. |
//! | `&impl Fn(...)` | the same thing, borrowed, so the inner sorts can be handed the caller's comparison without taking ownership of it. |
//! | `Vec<Option<MemChain>>` | a list whose slots can be individually emptied. See the comment at the reorder step for why this is necessary rather than fussy. |
//! | `.take()` | on an `Option`: removes the value, leaving `None` behind, and hands it over. This is what lets one slot be moved out while the rest of the vector stays intact. |
//! | `.expect("...")` | unwrap a value, crashing with that message if it is absent. Used only where absence would mean an internal invariant is already broken, in which case continuing would be worse. |
//! | `.into_iter()` | walks a collection by CONSUMING it, yielding owned values rather than borrows. `.iter()` by contrast borrows and leaves the collection usable afterwards. |
//! | `.map(Some)` | wraps every element in `Some`, passing the constructor by name instead of writing `\|c\| Some(c)`. |
//! | `.collect()` | gathers a walk into a container, chosen from the declared type of the destination. |
//! | `.retain(\|c\| ...)` | keeps only the elements passing the test, dropping the rest, in place. |
//! | `.sort_by_key(\|s\| ...)` | sorts by a value computed per element. The key here packs two 32-bit fields into one 64-bit integer so a single comparison orders by both, in that priority. |
//! | `u64::from(x) << 32 \| u64::from(y)` | that packing: widen, shift the first field into the high half, bitwise-OR the second into the low half. |
//! | `Vec::with_capacity(n)` | reserves room up front so the vector does not repeatedly grow and copy. |
//! | `&mut Vec<i64>` as a parameter | an output buffer supplied by the caller and filled in place, so a hot loop can reuse one allocation across reads instead of returning a fresh vector each time. |
//! | `as f32` / `as f64` | explicit integer-to-float conversions. Rust performs none implicitly, so every one of them is visible. |
//! | `x << 1` | a left shift by one bit, i.e. multiply by two. |
//! | `debug_assert_eq!` | a check compiled into debug and test builds only, and absent from a release binary. Used for invariants that must hold but are too costly to verify on every release run. |
//! | `if let Some(t) = ...` | run this block only when there is a value, binding it. |
//! | `rbeg` | `mem_seed_t.rbeg` | Reference begin of a seed, in 2L space. |
//! | `qbeg` | `mem_seed_t.qbeg` | Query begin of a seed: a 0-based offset into the read. |
//! | `w` | `mem_chain_t.w` | Chain WEIGHT: the smaller of (bases of the read covered by the chain's seeds) and (bases of the reference covered), counting each covered base once even where seeds overlap. A crude "how much of this alignment is actually supported by exact matches" score, used to rank chains. |
//! | `kept` | `mem_chain_t.kept` | The filtering verdict, 0 to 3. See [`MemChain::kept`]. |
//! | `occ` / `max_occ` | `opt->max_occ` | Occurrence count of a seed pattern, and the cap (default 500) on how many of those occurrences are ever materialized. |
//! | `s` (on a `Smem`) | `SMEM.s` | The seed pattern's occurrence count in the doubled reference. `s > max_occ` means the pattern is too repetitive to sample fully, which is what `frac_rep` measures. |
//! | `m`, `n` (on a `Smem`) | `SMEM.m/.n` | The match's span in the read, inclusive at both ends, so its length is `n - m + 1`. |
//! | `k` (on a `Smem`) | `SMEM.k` | First suffix-array row of the seed's interval. Row `k + i` is the `i`-th occurrence, and `FmIndex::get_sa` turns a row into a reference position. |
//! | `i` / `j` in `mem_chain_flt` | `i` / `j` | `i` indexes the chain being judged, `j` an already-accepted chain it is compared against. The `ib`/`ie`/`iw` and `jb`/`je`/`jw` bundles are those two chains' query begin, query end and weight. |
//!
//! # Reading order for this file
//!
//! 1. [`MemChain`]: the data structure and what each field means.
//! 2. [`test_and_merge`]: the single decision "does this seed join this chain", which is the whole
//!    chaining rule.
//! 3. [`build_chains_from_resolved`]: the greedy sweep that applies it, plus `frac_rep`.
//! 4. [`mem_chain_weight`] then [`mem_chain_flt`]: scoring and filtering.
//! 5. [`ks_introsort_by`] (and its `ks_combsort_by` / `ks_insertsort_by` helpers): a hand port of
//!    klib's unstable sort, present only so equal-weight ties break exactly as bwa's do.
//! 6. [`kbtree`]: the hand-ported B-tree, for the same tie-break reason.

mod kbtree;

use bwa_core::MemOpt;
use bwa_index::{BntSeq, FmIndex};
use bwa_seed::{mem_collect_smem, MemSeed};

/// A chain of collinear seeds (bwa-mem2's `mem_chain_t`).
#[derive(Debug, Clone)]
pub struct MemChain {
    /// Index of the read this chain belongs to within the batch (bwa's `tmp.seqid = l`,
    /// `bwamem.cpp:948`). Single-read here, so it is just passed through.
    pub seqid: i32,
    /// Reference-sequence (contig) index from `bns_intv2rid`. All seeds in a chain share it.
    pub rid: i32,
    /// The chain's key: the reference position of its **first** seed, fixed at creation and never
    /// updated as seeds are appended. This is what the kbtree is keyed on.
    pub pos: i64,
    /// Weight, filled by [`mem_chain_weight`] during filtering. 0 until then.
    pub w: i32,
    /// Filtering verdict from [`mem_chain_flt`]: 0 dropped, 1 "shadowed but recorded" (kept only so
    /// mapq can account for it), 2 kept despite a large overlap, 3 kept cleanly.
    pub kept: u8,
    /// Whether `rid` is an ALT contig. ALT chains never shadow primary ones during filtering.
    pub is_alt: bool,
    /// Index of the first chain this one shadows, or -1. Used by `mem_chain_flt` to promote that
    /// chain to `kept = 1` so mapq can see it. Not a seed index.
    pub first: i32,
    /// Fraction of the read covered by SMEMs that were too repetitive to sample fully
    /// (`s > max_occ`). Read-wide, so every chain of a read gets the same value. Feeds mapq.
    pub frac_rep: f32,
    /// The chain's seeds, in the order they were appended, which is ascending `qbeg`.
    pub seeds: Vec<MemSeed>,
}

/// Chain span on the **query**, from `bwamem.cpp`'s `chn_beg`/`chn_end` macros. Half-open
/// `[chn_beg, chn_end)`. Valid only because seeds are appended in ascending `qbeg` and never sorted.
///
/// # Parameters
/// - `c`: a chain with at least one seed. Both helpers index `seeds` unguarded, so an empty chain
///   panics; no code path here can produce one (a chain is created with exactly one seed and only
///   ever grows).
///
/// # Returns
/// `chn_beg`: the QUERY (read) offset, 0-based in bases, where the chain's first seed starts.
/// `chn_end`: the exclusive QUERY offset one past the last seed's last base. Neither is a reference
/// coordinate: no 2L-space value is involved.
#[inline]
fn chn_beg(c: &MemChain) -> i32 {
    c.seeds[0].qbeg
}
#[inline]
fn chn_end(c: &MemChain) -> i32 {
    // Last seed by append order, which is also the largest `qbeg` in the chain.
    let last = c.seeds.last().unwrap();
    last.qbeg + last.len
}

/// Try to absorb seed `p` (on contig `seed_rid`) into chain `c`; returns whether it was
/// merged/contained. Faithful port of `test_and_merge` (`bwamem.cpp:357-398`).
///
/// Returning `true` means "the caller must not create a new chain for this seed", which covers two
/// cases: the seed was appended, or the seed was already *contained* in the chain and is simply
/// dropped. Returning `false` means the chain refuses it, and since only this one chain was ever
/// consulted, the seed becomes a new chain even if some other existing chain would have taken it.
///
/// The four tests, in the C's order (order matters, they are not disjoint):
/// 1. Different contig: refuse. A chain may not straddle a reference sequence boundary.
/// 2. Contained in the chain's current query *and* reference span: absorb silently, adding nothing.
/// 3. Strand mismatch: if any existing endpoint is on the forward half (`< l_pac`) and the new seed
///    is on the reverse half (`>= l_pac`), refuse. The 2L layout makes strand a coordinate test.
/// 4. Collinearity: with `x` the query gap and `y` the reference gap from the chain's last seed,
///    require `y >= 0` (never go backwards on the reference), `|x - y| <= opt.w` (band width,
///    default 100: how much indel drift is tolerated) and both gaps beyond the last seed's end below
///    `opt.max_chain_gap` (default 10000 bases). `x` is non-negative by construction because seeds
///    arrive in ascending `qbeg`, which is why the C comments it "always non-negtive" and only tests
///    `y >= 0`.
///
/// # Parameters
/// - `opt`: the alignment options. Only `opt.w` (band half-width, bases, default 100) and
///   `opt.max_chain_gap` (bases, default 10000) are read here. Supplied by the CLI/`MemOpt::default`.
/// - `l_pac`: forward-strand reference length in bases, from `BntSeq::l_pac`. It is the boundary of
///   the 2L space: a reference position `< l_pac` is forward strand, `>= l_pac` is reverse.
/// - `c`: the one candidate chain, mutated in place (a seed is pushed onto `c.seeds` on acceptance).
///   Its `seeds` is non-empty and ascending in `qbeg`; `c.pos` is NOT updated by this function.
/// - `p`: the seed being offered. `p.qbeg` is a 0-based READ offset, `p.rbeg` a REFERENCE position in
///   2L space, `p.len` a length in bases. Precondition from the caller's sweep: `p.qbeg` is >= the
///   `qbeg` of every seed already in `c` (SMEMs are visited in ascending `m`).
/// - `seed_rid`: the contig index `p` falls in, from `BntSeq::intv2rid`. Always >= 0 here; the caller
///   has already dropped negative rids (boundary-straddling seeds).
///
/// # Returns
/// `true` if the caller must NOT start a new chain for `p` (either appended, or already contained and
/// therefore discarded); `false` if the chain refuses it.
///
/// `l_pac`: forward-strand length in bases, from `BntSeq::l_pac`.
fn test_and_merge(opt: &MemOpt, l_pac: i64, c: &mut MemChain, p: &MemSeed, seed_rid: i32) -> bool {
    // The chain's current frontier: the seed appended most recently, hence the largest `qbeg`.
    let last = *c.seeds.last().unwrap();
    // End of that frontier seed, exclusive. `qend` is a READ offset, `rend` a 2L-space REFERENCE
    // position. They are the upper bounds of the chain's span used by the containment test below.
    let qend = last.qbeg + last.len;
    let rend = last.rbeg + i64::from(last.len);

    if seed_rid != c.rid {
        return false;
    }
    if p.qbeg >= c.seeds[0].qbeg
        && p.qbeg + p.len <= qend
        && p.rbeg >= c.seeds[0].rbeg
        && p.rbeg + i64::from(p.len) <= rend
    {
        return true; // contained
    }
    if (last.rbeg < l_pac || c.seeds[0].rbeg < l_pac) && p.rbeg >= l_pac {
        return false; // different strand
    }
    // `x` = QUERY gap in bases from the frontier seed's start to the new seed's start. Non-negative
    // by construction (seeds arrive in ascending `qbeg`), which is why only `y` is sign-tested.
    // `y` = REFERENCE gap in bases between the same two starts, both in 2L space, so the subtraction
    // is well defined only because the strand test above already rejected a forward/reverse mix.
    // `x - y` is the DIAGONAL drift between the two seeds (a third coordinate space: reference minus
    // query). Zero means perfectly collinear; non-zero means an indel of that many bases. `opt.w`
    // caps |drift|, and `max_chain_gap` caps each gap measured beyond the frontier seed's end.
    let x = i64::from(p.qbeg - last.qbeg);
    let y = p.rbeg - last.rbeg;
    if y >= 0
        && x - y <= i64::from(opt.w)
        && y - x <= i64::from(opt.w)
        && x - i64::from(last.len) < i64::from(opt.max_chain_gap)
        && y - i64::from(last.len) < i64::from(opt.max_chain_gap)
    {
        c.seeds.push(*p);
        return true;
    }
    false
}

/// Chain weight = min of non-overlapping query and reference coverage, in bases. Port of
/// `mem_chain_weight` (`bwamem.cpp:429`).
///
/// Both loops are the same sweep: walk the seeds in order, tracking `end` = the furthest coordinate
/// covered so far, and add only the part of each seed beyond it. Seeds may overlap each other
/// (round 2 and round 3 deliberately produce nested/overlapping seeds), so a plain sum of `len`
/// would double count. Taking the **min** of the query-side and reference-side totals penalizes
/// chains that cover a lot of one and little of the other, which is what a bad indel-heavy chain
/// looks like.
///
/// The `1 << 30` clamp is bwa's own final line (`return w < 1<<30? w : (1<<30)-1;`). Note the C
/// accumulates into an `int w` while we use `i64`; the clamp makes the two agree for any weight a
/// real read can produce, and only reachable-by-overflow inputs could tell them apart.
///
/// # Parameters
/// - `c`: the chain to score. Must have at least one seed, and its seeds must be in ascending `qbeg`
///   (the append order the merge produces); both sweeps assume that order and would undercount if the
///   seeds were sorted differently. Only `c.seeds` is read; `c.w` is not written here (the caller in
///   [`mem_chain_flt`] stores the result).
///
/// # Returns
/// The weight in bases, in `0 ..= (1 << 30) - 1`.
pub fn mem_chain_weight(c: &MemChain) -> i32 {
    // ---- Pass 1: non-overlapping coverage on the QUERY (read) side ---------------------------
    // `w` accumulates covered read bases; `covered_to` is the largest exclusive READ offset already
    // counted. Invariant at the top of each iteration: `w` = size of the union of the query spans of
    // all seeds seen so far, and `covered_to` = the right edge of that union (valid as a single
    // number only because the spans arrive left to right).
    let mut w = 0i64;
    let mut covered_to = 0i64;
    for seed in &c.seeds {
        // `beg` is a READ offset here, `len` the seed length in bases.
        let (beg, len) = (i64::from(seed.qbeg), i64::from(seed.len));
        if beg >= covered_to {
            w += len;
        } else if beg + len > covered_to {
            w += beg + len - covered_to;
        }
        covered_to = covered_to.max(beg + len);
    }
    // Total read bases covered by the chain, each counted once. Saved because `w`/`covered_to` are
    // reused for the reference sweep.
    let query_coverage = w;

    // ---- Pass 2: the same sweep on the REFERENCE side ----------------------------------------
    // Same invariant, but `covered_to` is now a REFERENCE position in 2L space rather than a read
    // offset. Starting it at 0 is harmless: real `rbeg` values are far above 0, so the first seed
    // takes the `beg >= covered_to` branch and contributes its full length.
    w = 0;
    covered_to = 0;
    for seed in &c.seeds {
        // `beg` is a 2L-space REFERENCE position here (contrast pass 1, where it was a read offset).
        let (beg, len) = (seed.rbeg, i64::from(seed.len));
        if beg >= covered_to {
            w += len;
        } else if beg + len > covered_to {
            w += beg + len - covered_to;
        }
        covered_to = covered_to.max(beg + len);
    }

    // ---- Combine and clamp -------------------------------------------------------------------
    // `w` is now the reference-side coverage; take the min with the query side.
    w = w.min(query_coverage);
    if w < WEIGHT_CLAMP {
        w as i32
    } else {
        (WEIGHT_CLAMP - 1) as i32
    }
}

/// bwa's own ceiling on a chain weight, `1 << 30` (`return w < 1<<30? w : (1<<30)-1;`,
/// `mem_chain_weight`, `bwamem.cpp`). It exists so the `int` accumulator in the C cannot overflow;
/// no real read produces a weight anywhere near it.
const WEIGHT_CLAMP: i64 = 1 << 30;

/// Build seed chains for a single read (2-bit codes) with the given `seqid`. Port of `mem_chain`:
/// seeds the read, resolves suffix-array positions, then runs the greedy merge.
///
/// `codes` is used for seeding and then only for its **length** (the `frac_rep` denominator).
/// All three seeding rounds run (`mem_collect_smem`), and the "closest lower chain" lookup is the
/// hand-ported [`kbtree`], not an ordered map: see the module docs for why that distinction is
/// load-bearing. An earlier revision of this doc comment said round-1 only and `BTreeMap`; both were
/// stale and have been corrected here.
///
/// # Parameters
/// - `fm`: the FM index, used for seeding and to resolve suffix-array rows to reference positions.
/// - `bns`: the `.ann`/`.amb` metadata. Supplies `l_pac` (forward length in bases) and the
///   position-to-contig map used to reject boundary-straddling seeds.
/// - `opt`: alignment options; `max_occ`, `w`, `max_chain_gap` and the seeding knobs are read.
/// - `codes`: the read as 2-bit base codes, one byte per base (0..=3, 4 for N). Its length in bases is
///   the `frac_rep` denominator.
/// - `seqid`: this read's index within the batch, copied verbatim into every chain's `seqid`.
///
/// # Returns
/// The read's chains, ordered by the kbtree's in-order traversal (ascending `pos`, klib array order
/// within a tie). Weights and `kept` are still zero: [`mem_chain_flt`] fills them.
pub fn build_chains(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    seqid: i32,
) -> Vec<MemChain> {
    let smems = mem_collect_smem(fm, codes, opt);
    build_chains_from_smems(fm, bns, opt, codes, seqid, smems)
}

/// A run of suffix-array rows that is an exact repeat of an earlier run in the same batch.
///
/// Produced by [`sa_positions_for_read`] when two adjacent SMEMs are identical after the sort, which
/// happens for **3.7% of SMEMs on simulated reads and 9.8% on real GIAB** and, because a duplicate
/// of a high-occurrence SMEM repeats all of its rows, accounts for **11.9% and 17.7% of the rows the
/// walk visits**. Those rows are the expensive part: 60 ns each on an M4 and 163 ns on Zen 3.
///
/// Deleting the duplicate SMEM would be simpler and is NOT byte-identical: it removes seeds, and the
/// discard pass reproduces bwa-mem2's scan order slot by slot, so the region set shifts and `XS`
/// moves on a handful of records per million (measured, see `BWA4_SMEM_DEDUP`). Keeping the SMEM and
/// skipping only the redundant WALK changes nothing downstream: the position list, the per-SMEM
/// counts and the resolved rows are all identical, byte for byte, because the duplicate's answers
/// are copied from the run they duplicate rather than recomputed.
#[derive(Clone, Copy, Debug)]
pub struct RepeatedRun {
    /// Index into the flat position list where the duplicate run starts.
    pub dst: usize,
    /// Index where the run it duplicates starts. Always an earlier, non-duplicate run.
    pub src: usize,
    /// How many rows the run covers.
    pub len: usize,
}

/// Sort a read's SMEMs into bwa's intra-read order and enumerate every sampled occurrence position
/// the chain merge will consume, in exactly that order, plus each SMEM's sampled count.
///
/// Split out of [`build_chains_from_smems`] so a caller can gather the positions of **many reads**
/// and resolve them in one big [`FmIndex::get_sa_batch`]. `get_sa_batch` is result-identical at any
/// chunk size (asserted by `bwa-index`'s `sa_batch_bench`), and this touches no FM index at all, so
/// batching across reads cannot change a value or an order.
///
/// Returns one entry per SMEM: how many occurrences that SMEM contributed. `positions` is **appended
/// to**, so a caller can accumulate across reads; the per-SMEM counts are what lets it slice the
/// flat result back apart. Invariant for a single read: `counts.iter().sum() == positions.len()`.
///
/// # Parameters
/// - `opt`: only `opt.max_occ` is read (default 500): the cap on how many occurrences of one SMEM are
///   ever materialized.
/// - `smems`: this read's SMEMs, **sorted in place** into bwa's `(m, n)` ascending order. The caller
///   must pass the same slice on to the merge, because the merge consumes seeds in exactly this order
///   and the order is observable in the output.
/// - `positions`: output, **appended to** (never cleared), so a caller can accumulate across many
///   reads before one batched `get_sa` call. Each pushed value is a suffix-array ROW (`p.k + k`), not
///   yet a reference position.
///
/// # Returns
/// One count per SMEM, in the post-sort SMEM order: how many rows that SMEM appended to `positions`.
/// Each count is in `1 ..= max_occ` (a SMEM always has `s >= 1`).
pub fn sa_positions_for_read(
    opt: &MemOpt,
    smems: &mut Vec<bwa_index::Smem>,
    positions: &mut Vec<i64>,
    repeats: &mut Vec<RepeatedRun>,
) -> Vec<i64> {
    // bwa sorts the whole batch's SMEMs with `compare_smem` = (rid, m, n) ascending (`sortSMEMs`,
    // via `FMI_search.cpp:987`). Within one read `rid` is constant, so packing `(m, n)` into a single
    // u64 key (`m` in the high half, `n` in the low half, both u32) yields the same permutation. It
    // must happen before the position walk, because the chain merge consumes seeds strictly in this
    // order and the greedy "offer to one chain" rule makes that order observable.
    //
    // Rust: one integer key encoding a two-level ordering. `s.m` is widened to 64 bits and shifted
    // into the high half, `s.n` occupies the low half, so comparing the packed values compares `m`
    // first and only falls through to `n` when the `m`s are equal. This is the standard way to get a
    // multi-field sort out of a single-key sort, and it is exact rather than approximate because
    // both fields are known to fit in 32 bits.
    smems.sort_by_key(|s| (u64::from(s.m) << 32) | u64::from(s.n));
    // `BWA4_SMEM_DUP=1`: how many of this read's SMEMs are EXACT duplicates of the one before them,
    // now that they are sorted. fg-labs/bwa-mem3 compacts those away in `smem_dedup.cpp` before the
    // position walk, on the grounds that they enumerate the same suffix-array rows twice; the
    // question here is whether that is worth anything at our seed lengths, and it is a count before
    // it is a change. Counts only, so it cannot move a byte.
    if smem_dup::enabled() {
        let (mut dups, mut dup_positions, mut positions_total) = (0u64, 0u64, 0u64);
        let max_occ_probe = i64::from(opt.max_occ);
        for (i, p) in smems.iter().enumerate() {
            // How many suffix-array rows this SMEM will walk, by the same stride rule as the loop
            // below. Counting it here rather than reusing that loop keeps the probe out of it.
            let step = if p.s > max_occ_probe {
                p.s / max_occ_probe
            } else {
                1
            };
            // Ceiling division written out: `i64::div_ceil` is unstable on the pinned toolchain.
            let walked = (((p.s + step - 1) / step).min(max_occ_probe)).max(0) as u64;
            positions_total += walked;
            if i > 0 {
                let a = &smems[i - 1];
                if a.m == p.m && a.n == p.n && a.k == p.k && a.s == p.s {
                    dups += 1;
                    dup_positions += walked;
                }
            }
        }
        smem_dup::record(smems.len() as u64, dups, positions_total, dup_positions);
    }
    // `BWA4_SMEM_DEDUP=1`: drop SMEMs identical to their predecessor, which the sort above has just
    // made adjacent. Each one would otherwise walk the same suffix-array rows a second time, and the
    // probe above measures that at 11.9% of walks on simulated reads and 17.7% on real GIAB.
    //
    // OFF BY DEFAULT, and it stays off: MEASURED 2026-08-18, it changes the output. chr21, 1M
    // simulated pairs, 2,000,000 records: **8 lines differ, and only in `XS:i`** (50 against 45 on
    // the first), i.e. four records per million reads report a different suboptimal score because a
    // redundant candidate region no longer exists to contribute one. Placement, CIGAR and MAPQ are
    // untouched on this data, but `XS` feeds MAPQ, so the divergence is not cosmetic by
    // construction, only by luck on this fixture.
    //
    // That is a fail under this project's criterion and a non-issue under the fork's:
    // fg-labs/bwa-mem3 compacts unconditionally (`src/smem_dedup.cpp`) and lists `score2`/MAPQ
    // convergence among its accepted differences from bwa-mem2. Kept behind the flag with the
    // measurement rather than deleted, so the idea is not rediscovered a third time, and so that a
    // future compat mode that accepts `XS` drift has it ready.
    if smem_dedup::enabled() {
        let mut write = 0usize;
        for read in 1..smems.len() {
            let (a, b) = (smems[write], smems[read]);
            if a.m != b.m || a.n != b.n || a.k != b.k || a.s != b.s {
                write += 1;
                smems[write] = b;
            }
        }
        if !smems.is_empty() {
            smems.truncate(write + 1);
        }
    }
    // Cap on materialized occurrences per SMEM (default 500), widened to i64 to compare against `s`.
    let max_occ = i64::from(opt.max_occ);
    let mut counts: Vec<i64> = Vec::with_capacity(smems.len());
    // The last SMEM whose rows were actually enumerated for walking, and where its run starts. A
    // duplicate does not replace it, so a run of identical SMEMs all point at the first.
    let mut previous: Option<(bwa_index::Smem, usize)> = None;
    for p in smems.iter() {
        // Same stride sampling as `bwa_seed::seeds_from_smem` / `bwamem.cpp:897`: take at most
        // `max_occ` (default 500) of the SMEM's `s` occurrences, spread evenly across the interval
        // rather than taken from the front. `k` is an offset within it, so the BWT row is `p.k + k`.
        // Stride through the SMEM's `s` occurrences, in suffix-array rows. `>= 1` always; integer
        // division means `s / max_occ` rounds down, so slightly more than `max_occ` rows can be
        // reachable, which is why the loop also tests `count < max_occ`.
        let step = if p.s > max_occ { p.s / max_occ } else { 1 };
        // `k` is an OFFSET within the SMEM's suffix-array interval (0..s), not a genomic coordinate:
        // the absolute row is `p.k + k`. `count` is how many rows this SMEM has emitted so far.
        let mut k = 0i64;
        let mut count = 0i64;
        // Where this SMEM's rows begin in the flat list, so a later duplicate can point at them.
        let run_start = positions.len();
        while k < p.s && count < max_occ {
            positions.push(p.k + k);
            k += step;
            count += 1;
        }
        // An SMEM identical to its predecessor enumerates the identical rows, in the identical
        // order, because the enumeration reads nothing but `(k, s)` and `max_occ`. Record the
        // repeat so the caller can copy the answers instead of walking the index again. The
        // positions themselves are still pushed: the list, the counts and everything downstream stay
        // exactly as they were, which is what makes this byte-identical where deleting the SMEM is
        // not. `prev_start` chains to the FIRST copy, so three identical SMEMs give two repeats both
        // pointing at the original rather than a chain that has to be resolved in order.
        if let Some((prev, prev_start)) = previous {
            if prev == *p && count as usize == positions.len() - run_start {
                repeats.push(RepeatedRun {
                    dst: run_start,
                    src: prev_start,
                    len: count as usize,
                });
            } else {
                previous = Some((*p, run_start));
            }
        } else {
            previous = Some((*p, run_start));
        }
        counts.push(count);
    }
    counts
}

/// `BWA4_SMEM_DUP=1`: are there exact duplicate SMEMs to compact away, and how many?
///
/// The fork's `smem_dedup.cpp` drops SMEMs identical to their predecessor on `(rid, m, n, k, l, s)`
/// after the sort, which removes duplicate suffix-array walks for free. Whether that is a lever
/// here or a rounding error is a count, and this is the count. Off by default; one cached bool load
/// per read when off.
pub mod smem_dup {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    /// SMEMs seen, summed over reads.
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    /// Of those, how many were exact duplicates of their immediate predecessor.
    static DUPS: AtomicU64 = AtomicU64::new(0);
    /// Suffix-array rows the walk will visit in total.
    static POS: AtomicU64 = AtomicU64::new(0);
    /// Of those, how many are visited a second time because of a duplicate SMEM. This is the number
    /// that matters: duplicates are only worth removing in proportion to the walks they cause, and a
    /// duplicate of a one-occurrence SMEM costs one lookup while a duplicate of a 500-occurrence one
    /// costs 500.
    static DUP_POS: AtomicU64 = AtomicU64::new(0);

    /// Whether the probe is on. Read once and cached.
    ///
    /// # Returns
    /// True if `BWA4_SMEM_DUP` is set to anything at all.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_SMEM_DUP").is_some())
    }

    /// Add one read's counts.
    ///
    /// # Parameters
    /// - `total`: SMEMs this read had.
    /// - `dups`: how many were exact duplicates of the previous one.
    /// - `positions`: suffix-array rows the read's walk will visit.
    /// - `dup_positions`: of those, how many are due to duplicates.
    pub fn record(total: u64, dups: u64, positions: u64, dup_positions: u64) {
        TOTAL.fetch_add(total, Relaxed);
        DUPS.fetch_add(dups, Relaxed);
        POS.fetch_add(positions, Relaxed);
        DUP_POS.fetch_add(dup_positions, Relaxed);
    }

    /// Print the count once at end of run. No-op unless [`enabled`].
    pub fn dump() {
        if !enabled() {
            return;
        }
        let (t, d) = (TOTAL.load(Relaxed), DUPS.load(Relaxed));
        let (p, dp) = (POS.load(Relaxed), DUP_POS.load(Relaxed));
        eprintln!(
            "[smem-dup] {d} exact duplicate SMEMs of {t} ({:.2}%), costing {dp} of {p} \
             suffix-array walks ({:.2}%) -- the latter is the ceiling on a sort-free dedup",
            100.0 * d as f64 / t.max(1) as f64,
            100.0 * dp as f64 / p.max(1) as f64,
        );
    }
}

/// `BWA4_SMEM_DEDUP=1`: the switch for the sort-free duplicate compaction in
/// [`sa_positions_for_read`]. Separate module so the flag is read once and cached, like every other
/// gate in this tree.
pub mod smem_dedup {
    use std::sync::OnceLock;

    /// Whether duplicate SMEMs are compacted away before the suffix-array walk.
    ///
    /// # Returns
    /// True if `BWA4_SMEM_DEDUP` is set to anything at all.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_SMEM_DEDUP").is_some())
    }
}

/// Build chains from **pre-computed** SMEMs (e.g. from batched lockstep seeding). Identical to
/// [`build_chains`] given the same seed set; only the SMEM source differs.
///
/// # Parameters
/// - `fm`, `bns`, `opt`, `codes`, `seqid`: as in [`build_chains`].
/// - `smems`: the read's SMEMs in ANY order; they are sorted into bwa's order here, so a batched
///   seeder does not have to pre-sort. Taken by value because the sort is destructive.
///
/// # Returns
/// Same as [`build_chains`].
pub fn build_chains_from_smems(
    fm: &FmIndex,
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    seqid: i32,
    mut smems: Vec<bwa_index::Smem>,
) -> Vec<MemChain> {
    // Suffix-array ROWS to resolve, flat and in SMEM order. Not reference coordinates yet.
    let mut positions: Vec<i64> = Vec::new();
    // The per-read path resolves its own rows and is not the hot one (the batched seeder in
    // `bwa-mem4-mem` is), so it walks every row and discards the repeat list rather than carrying
    // the copy machinery for one read's worth of rows.
    let mut repeats: Vec<RepeatedRun> = Vec::new();
    let counts = sa_positions_for_read(opt, &mut smems, &mut positions, &mut repeats);
    // Output buffer for the resolved 2L-space REFERENCE positions, one per row, same index.
    let mut rbegs = vec![0i64; positions.len()];
    // `Some(start instant)` only under `BWA4_CHAIN_TIME=1`; `None` (and zero cost) otherwise.
    let t_sa = chain_time::enabled().then(std::time::Instant::now);
    fm.get_sa_batch(&positions, &mut rbegs);
    if let Some(t) = t_sa {
        use std::sync::atomic::Ordering::Relaxed;
        chain_time::GET_SA_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        chain_time::GET_SA_N.fetch_add(positions.len() as u64, Relaxed);
    }
    build_chains_from_resolved(bns, opt, codes, seqid, &smems, &counts, &rbegs)
}

/// The chain merge, given SMEMs already sorted by [`sa_positions_for_read`] and their occurrence
/// positions already resolved to `rbegs` (same values, same order). Touches no FM index.
///
/// This is the ONLY copy of the merge, so the per-read path ([`build_chains_from_smems`]) and a
/// cross-read batched caller cannot drift apart.
///
/// # Parameters
/// (and their invariants; violating any of the length relations panics or silently misassigns seeds)
/// - `bns`: reference metadata. Supplies `l_pac` (forward length in bases, the 2L-space boundary) and
///   `intv2rid` for the contig lookup.
/// - `opt`: only `max_occ` (the `frac_rep` threshold) plus whatever [`test_and_merge`] reads
///   (`w`, `max_chain_gap`).
/// - `codes`: only its length is used, as the `frac_rep` denominator (the read length in bases).
///   Must be non-zero, else `frac_rep` is NaN.
/// - `seqid`: the read's batch index, copied into every chain.
/// - `smems`: this read's SMEMs, already sorted by [`sa_positions_for_read`]. Order is observable.
/// - `counts`: one per SMEM, from the same call. `counts.len() == smems.len()`.
/// - `rbegs`: the resolved reference positions in 2L space, flat, in SMEM order.
///   `rbegs.len() == counts.sum()`; the `pos_cursor` below relies on that exactly. For a cross-read
///   batched caller this is the slice belonging to THIS read, not the whole batch.
///
/// # Returns
/// The read's chains in kbtree in-order order (see pass 3). Every chain has `frac_rep` set and
/// `w`/`kept`/`is_alt`/`first` still at their defaults.
pub fn build_chains_from_resolved(
    bns: &BntSeq,
    opt: &MemOpt,
    codes: &[u8],
    seqid: i32,
    smems: &[bwa_index::Smem],
    counts: &[i64],
    rbegs: &[i64],
) -> Vec<MemChain> {
    let t_total = chain_time::enabled().then(std::time::Instant::now);
    // Repetitive length -> frac_rep. Union of the query spans of SMEMs too repetitive to sample
    // fully (`s > max_occ`, i.e. the stride sampling threw occurrences away). Because `smems` is
    // sorted by `m`, the union is computed by a single merge-adjacent-intervals sweep: `[b, e)` is
    // the interval being accumulated, and a new span either starts past `e` (close the old one, start
    // fresh) or overlaps it (extend `e`). The final `l_rep += e - b` closes the last interval, and is
    // harmless when no SMEM qualified because `b == e == 0`.
    // ---- Pass 1: repetitive read coverage (`frac_rep`) ---------------------------------------
    // Reminder (see the module glossary): `s` is a seed pattern's occurrence count, `m`/`n` its
    // inclusive span in the read, `max_occ` the cap on occurrences ever materialized.
    // `l_rep`: total read bases (union, no double counting) covered by over-repetitive SMEMs, closed
    // intervals only. `rep_beg`/`rep_end`: the half-open READ span currently being accumulated, not
    // yet added to `l_rep`. Both are query offsets, never reference positions. Invariant at the top of
    // each iteration: `l_rep` covers every qualifying SMEM strictly before `rep_beg`, and
    // `[rep_beg, rep_end)` is the still-open merged interval. `(0, 0)` is a valid empty start.
    let mut l_rep = 0i64;
    let (mut rep_beg, mut rep_end) = (0i64, 0i64);
    for smem in smems.iter() {
        if smem.s <= i64::from(opt.max_occ) {
            continue;
        }
        // Spans are stored inclusive (`m..=n`); convert to half-open for the interval arithmetic.
        // This SMEM's own half-open READ span, in query offsets.
        let (span_beg, span_end) = (i64::from(smem.m), i64::from(smem.n) + 1);
        if span_beg > rep_end {
            l_rep += rep_end - rep_beg;
            rep_beg = span_beg;
            rep_end = span_end;
        } else {
            rep_end = rep_end.max(span_end);
        }
    }
    l_rep += rep_end - rep_beg;

    // Forward-strand reference length in bases: the 2L-space split point handed to `test_and_merge`.
    let l_pac = bns.l_pac;
    // Chains in CREATION order. The kbtree stores indices into this vector, so it must not be
    // reordered or have elements removed while the merge runs. Pass 3 permutes it at the very end.
    let mut chains: Vec<MemChain> = Vec::new();
    // bwa keys chains by `pos` in a klib kbtree, whose exact shape is observable (see `kbtree`).
    let mut tree = crate::kbtree::KbTree::new();

    // ---- Pass 2: the greedy merge ------------------------------------------------------------
    // Replays the original per-occurrence logic with precomputed `rbeg` values
    // (same values, same order -> byte-identical chains).
    // `pi` walks `rbegs` monotonically: SMEM `si` owns the next `counts[si]` entries. This is the
    // only coupling between the three input slices, and it is why `rbegs` must not be reordered.
    //
    // Loop invariant at the top of each occurrence iteration: `chains` holds every chain created so
    // far (none is ever closed or removed: any of them can still absorb a later seed), `tree` maps
    // each chain's `pos` to its index in `chains` with one entry per chain, and `pos_cursor` is the
    // number of occurrences already consumed across all previous SMEMs plus this one's.
    let mut pos_cursor = 0usize;
    for (smem_index, smem) in smems.iter().enumerate() {
        // Seed length = the SMEM's span. `n + 1 - m` in u32 (no underflow: `n >= m` always).
        let slen = (smem.n + 1 - smem.m) as i32;
        for _ in 0..counts[smem_index] {
            // This occurrence's REFERENCE start in 2L space (>= l_pac means reverse strand). The
            // matching query start is `smem.m`, the same for every occurrence of this SMEM.
            let rbeg = rbegs[pos_cursor];
            pos_cursor += 1;
            // One exact match: `rbeg` a 2L-space REFERENCE start, `qbeg` a 0-based READ offset, `len`
            // its length in bases (identical on both sides, since a seed is an exact match with no
            // indels). `score` is initialized to the length, matching bwa's `s->score = s->len`.
            let seed = MemSeed {
                rbeg,
                qbeg: smem.m as i32,
                len: slen,
                score: slen,
            };
            // Which reference sequence does `[rbeg, rbeg + slen)` land in? Negative means the seed
            // straddles a contig boundary or the forward/reverse-complement seam at `l_pac`, in
            // which case bwa simply drops it (`bwamem.cpp:921`, with its own TODO about splitting
            // rather than discarding). We reproduce the discard, TODO and all.
            let rid = bns.intv2rid(rbeg, rbeg + i64::from(slen));
            if rid < 0 {
                continue;
            }
            // Only this one chain is offered the seed: if it declines, bwa starts a new chain even
            // though another chain might have accepted it. The C guards this with `if (kb_size(tree))
            // { kb_intervalp(...); if (!lower || !test_and_merge(...)) to_add = 1; }`
            // (`bwamem.cpp:922-926`); on an empty tree `lower()` returns `None`, so the two agree.
            // Index into `chains` of the chain whose `pos` is the largest one at or below this seed's
            // `rbeg`, or `None` if the tree is empty or every chain starts above it.
            let lower = tree.lower(rbeg);
            // Set false only if that one chain absorbed the seed (appended or already contained).
            let mut to_add = true;
            if let Some(chain_index) = lower {
                if test_and_merge(opt, l_pac, &mut chains[chain_index], &seed, rid) {
                    to_add = false;
                }
            }
            if to_add {
                // Slot the new chain will occupy in `chains`, and the payload stored in the tree.
                let idx = chains.len();
                chains.push(MemChain {
                    seqid,
                    rid,
                    pos: rbeg,
                    w: 0,
                    kept: 0,
                    // `tmp.is_alt = !!bns->anns[rid].is_alt` (`bwamem.cpp:948`). `rid >= 0` here:
                    // the negative case was discarded above.
                    is_alt: bns.contigs[rid as usize].is_alt,
                    first: -1,
                    frac_rep: 0.0,
                    seeds: vec![seed],
                });
                // bwa keys chains by `pos` in a kbtree, which *permits duplicates*: when several
                // chains share a pos, `kb_intervalp`'s lower_bound returns the first one inserted
                // (klib appends later duplicates after it). A plain map would overwrite, silently
                // redirecting later seeds to the most recent chain instead, which changes which
                // chain they merge into and can lose a chain entirely.
                tree.put(rbeg, idx);
            }
        }
    }

    // ---- Pass 3: stamp frac_rep, then emit in kbtree in-order order ---------------------------
    // Fraction of the READ covered by over-repetitive SMEMs, in 0.0..=1.0. Read-wide, so every chain
    // gets the same value; mapq later uses it to discount hits in repeats.
    let frac = l_rep as f32 / codes.len() as f32;
    for c in &mut chains {
        c.frac_rep = frac;
    }
    // bwa-mem2 stores chains in a position-keyed kbtree and emits them via an in-order traversal, so
    // `mem_chain_flt` receives them by `pos` ascending -- and, within one pos, in kbtree *array*
    // order rather than insertion order. Replay the tree instead of sorting: a stable sort by pos
    // would keep insertion order for duplicates. This ordering drives the (unstable) tie-break among
    // equal-weight chains in `mem_chain_flt`.
    // A permutation of `0..chains.len()`: indices into `chains` in the emission order. Every chain
    // appears exactly once, since `put` was called once per created chain and nothing is deleted.
    let order: Vec<usize> = tree.in_order();
    debug_assert_eq!(order.len(), chains.len());
    // `chains` re-wrapped so each element can be moved out once, by index, in `order`'s order.
    //
    // Rust: the reason for the `Option` wrapper. The chains must come out in `order`, which is an
    // arbitrary permutation, and each must be moved out whole rather than copied (a `MemChain` owns
    // a vector of seeds). Rust forbids moving a value out of a vector by index, because that would
    // leave a hole the type system cannot describe. Wrapping each element in an `Option` gives the
    // hole a name: `.take()` below removes the chain and leaves `None` in its place, which IS a
    // valid value, so the vector stays consistent throughout. No chain is cloned.
    let mut slots: Vec<Option<MemChain>> = chains.into_iter().map(Some).collect();
    let out = order
        .into_iter()
        .map(|i| {
            slots[i]
                .take()
                .expect("each chain is inserted exactly once")
        })
        .collect();
    if let Some(t) = t_total {
        chain_time::TOTAL_NS.fetch_add(
            t.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    out
}

/// Insertion sort over `a`, moving an element left while it is strictly `lt` its predecessor.
/// Faithful port of klib's `__ks_insertsort` (stable for equal keys). `lt(x, y)` = `x < y`.
///
/// # Parameters
/// - `a`: the slice, sorted in place. Any length including 0.
/// - `lt`: strict less-than under the desired order. In [`mem_chain_flt`] it is `x.w > y.w`, so
///   "less" means "heavier" and the result is weight-descending. It must be a strict weak ordering:
///   `lt(x, x)` returning true would loop the inner while past the front.
// Rust: read the signature as "sorts a run of ANY element type, using a comparison the caller
// supplies". `<T>` is the element-type blank; `&mut [T]` is the caller's own storage, reordered in
// place; `&impl Fn(&T, &T) -> bool` is a borrowed callable taking two elements and answering
// whether the first comes first. Passing the ordering in is why one implementation serves every
// sort in this file, and why "less" can be made to mean "heavier" at the `mem_chain_flt` call site
// without a second copy of the algorithm.
//
// `a.swap(i, j)` exchanges two elements of a slice. It exists as a method because the obvious
// three-line temporary-variable version would need two exclusive borrows of `a` at once, which the
// compiler refuses.
fn ks_insertsort_by<T>(a: &mut [T], lt: &impl Fn(&T, &T) -> bool) {
    for i in 1..a.len() {
        let mut j = i;
        while j > 0 && lt(&a[j], &a[j - 1]) {
            a.swap(j, j - 1);
            j -= 1;
        }
    }
}

/// Comb sort, klib's `ks_combsort` (introsort's depth-limit fallback), ending with an insertion
/// sort pass. `lt(x, y)` = `x < y`.
///
/// Comb sort is bubble sort with a shrinking stride: compare-and-swap elements `gap` apart, shrink
/// `gap`, repeat until `gap == 1` and a full pass makes no swap. It is O(n log n) in practice and,
/// crucially for us, deterministic, which is why klib uses it instead of heapsort as the
/// depth-limit escape.
///
/// # Parameters
/// - `a`: the sub-range to sort in place (in [`ks_introsort_by`] it is one partition, not the whole
///   array). Any length including 0.
/// - `lt`: same contract as [`ks_insertsort_by`], and it must be the SAME comparator, since the
///   introsort mixes the two algorithms' outputs.
fn ks_combsort_by<T>(a: &mut [T], lt: &impl Fn(&T, &T) -> bool) {
    let n = a.len();
    if n == 0 {
        return;
    }
    // klib's shrink factor, `1 / (1 - 1/e^phi)`. The literal must be copied digit for digit: `gap`
    // is derived by float division and truncation, so a different constant changes the gap sequence
    // and therefore the exact permutation of equal elements.
    const SHRINK: f64 = 1.2473309501039786540366528676643;
    // Current comparison stride, in elements. Starts at `n`, which is `>= n` so the first pass does
    // nothing but shrink; converges to 1. The loop exits only once `gap <= 2` AND a whole pass swapped
    // nothing, which is the sortedness proof.
    let mut gap = n;
    loop {
        if gap > 2 {
            gap = (gap as f64 / SHRINK) as usize;
            // The classic "comb sort 11" rule: gaps of 9 or 10 are pathological (they leave a
            // characteristic pattern that costs an extra pass), so they are bumped to 11.
            // `ksort.h:172`, copied verbatim.
            if gap == 9 || gap == 10 {
                gap = 11;
            }
        }
        // True if this pass exchanged anything; a `gap <= 2` pass with no swap means sorted.
        let mut do_swap = false;
        if gap < n {
            let mut i = 0;
            while i < n - gap {
                // Partner element `gap` slots to the right of `i`; both are plain array indices.
                let j = i + gap;
                if lt(&a[j], &a[i]) {
                    a.swap(i, j);
                    do_swap = true;
                }
                i += 1;
            }
        }
        if !(do_swap || gap > 2) {
            break;
        }
    }
    if gap != 1 {
        ks_insertsort_by(a, lt);
    }
}

/// Introspective sort, a faithful port of klib's `ks_introsort` (median-of-3 quicksort with a
/// depth-limited comb-sort fallback and a final insertion-sort pass). This is deliberately the
/// exact same (unstable) permutation bwa-mem2 applies in `mem_chain_flt`, so equal-weight chains
/// resolve identically to the oracle. `lt(x, y)` = `x < y`. Source: `ksort.h:185-234`.
///
/// **Why a hand port instead of `sort_unstable_by`.** Chains routinely tie on weight, and
/// `mem_chain_flt` processes them in the sorted order, so which of two equal-weight chains comes
/// first decides which one shadows the other, which decides the SAM output. Any unstable sort gives
/// *a* correct ordering but not *bwa's* ordering. Rust's `sort_unstable_by` (pattern-defeating
/// quicksort) would give a different permutation on ties. So the exact algorithm is reproduced:
/// median-of-3 quicksort, an explicit stack, recursion only into the larger half, the 16-element
/// cutoff, the depth limit `2 * ceil(log2 n)`, and the single final insertion-sort pass that cleans
/// up every sub-16 run left behind.
///
/// # Parameters
/// - `a`: the slice, sorted in place. Any length; 0, 1 and 2 are special-cased.
/// - `lt`: strict less-than under the desired order, a strict weak ordering. Callers here pass
///   `|x, y| x.w > y.w` (weight descending) and the tests pass `|x, y| x > y`. The Hoare partition
///   below relies on `!lt(pivot, pivot)` to stop its unbounded inner scan, so a comparator that
///   reports an element less than itself would read out of bounds.
pub fn ks_introsort_by<T>(a: &mut [T], lt: impl Fn(&T, &T) -> bool) {
    let n = a.len();
    if n < 1 {
        return;
    }
    if n == 2 {
        if lt(&a[1], &a[0]) {
            a.swap(0, 1);
        }
        return;
    }
    // `dd = ceil(log2 n)`, clamped below at 2 by the starting value. C: `for (d = 2; 1ul<<d < n; ++d);`
    let mut dd = 2usize;
    while (1usize << dd) < n {
        dd += 1;
    }
    // Explicit stack of pending partitions, so recursion depth is bounded. `s`/`t` are the inclusive
    // bounds of the partition being worked on; `d` is the remaining depth budget for it.
    // Fixed array, not a `Vec`, exactly as klib declares it (`ks_isort_stack_t stack[64]`). A `Vec`
    // here costs one heap allocation per sort the moment any partition is pushed, and this function
    // is called ~10M times per real paired-end run (twice per `mem_sort_dedup_patch`, which mate
    // rescue drives ~4.9M times; see `BWA4_DEDUP_SHAPE`). 64 slots cannot overflow: a push only ever
    // happens for the LARGER half while the loop continues on the smaller, so the pending count is
    // bounded by the recursion depth of repeated halving, i.e. `log2(usize::MAX)`.
    //
    // Pure implementation detail: the comparison and swap sequence is untouched, so the permutation
    // is identical, which is the only property this port has to preserve.
    let mut stack = [(0usize, 0usize, 0i64); 64];
    // Number of occupied `stack` slots; `stack[top - 1]` is the most recently pushed partition.
    let mut top = 0usize;
    // `s`/`t`: inclusive lo/hi array indices of the partition currently being worked on. Invariant at
    // the top of the loop: everything outside every pending partition (`stack` plus `[s, t]`) is
    // already in its final position relative to the other partitions, so partitions never interleave.
    let mut s = 0usize;
    let mut t = n - 1;
    // Depth budget `2 * ceil(log2 n)`: quicksort should need about `log2 n` levels, so twice that
    // means the pivots are pathological and it is time to bail out to comb sort.
    let mut d: i64 = (dd as i64) << 1;
    loop {
        if s < t {
            d -= 1;
            if d == 0 {
                // Budget exhausted: comb-sort this partition outright, then `t = s` makes the next
                // iteration fall into the `else` branch and pop the stack. Note `d` is decremented
                // per partition and inherited by both halves, exactly as klib does.
                ks_combsort_by(&mut a[s..=t], &lt);
                t = s;
                continue;
            }
            // Hoare scan cursors, both array indices into `[s, t]`. `i` sweeps right from `s`, `j`
            // left from `t`; both are pre-incremented/decremented before first use, so they start one
            // slot outside the range they will examine.
            let mut i = s;
            let mut j = t;
            // Median-of-3 pivot selection over `a[s]`, `a[k]`, `a[t]`. klib's `k` is deliberately
            // NOT the exact midpoint: it is `s + (t - s) / 2 + 1`, biased one slot right. Copied as
            // written, because a different pivot index reorders ties.
            let mut k = i + ((j - i) >> 1) + 1;
            if lt(&a[k], &a[i]) {
                if lt(&a[k], &a[j]) {
                    k = j;
                }
            } else {
                k = if lt(&a[j], &a[i]) { i } else { j };
            }
            if k != t {
                a.swap(k, t);
            }
            // Pivot value now lives at index `t` and is untouched until the final swap below. klib
            // instead keeps a *copy* `rp = *k` before the swap and compares against that; since the
            // pivot slot is never written during partitioning, comparing against `a[t]` is the same.
            //
            // Hoare partition. The inner `i` scan has no upper bound test: it is safe only because
            // the pivot sits at `t` and stops it (`!lt(a[t], a[t])`). Same sentinel argument as the C.
            loop {
                loop {
                    i += 1;
                    if !lt(&a[i], &a[t]) {
                        break;
                    }
                }
                loop {
                    j -= 1;
                    if !(i <= j && lt(&a[t], &a[j])) {
                        break;
                    }
                }
                if j <= i {
                    break;
                }
                a.swap(i, j);
            }
            // Drop the pivot into its final position `i`.
            a.swap(i, t);
            // Sizes of the two halves. Push the LARGER half and continue on the smaller, which is
            // what bounds the stack at O(log n). Halves of 16 or fewer elements are neither pushed
            // nor descended into: the final `ks_insertsort_by` pass sorts all of them at once, which
            // is faster than partitioning tiny ranges.
            // Element counts of the two halves around the settled pivot at `i`: `is` = the left half
            // `[s, i-1]`, `ti` = the right half `[i+1, t]`. Both exclude the pivot itself.
            let is = i - s;
            let ti = t - i;
            if is > ti {
                if is > 16 {
                    debug_assert!(top < stack.len(), "introsort stack overflow");
                    stack[top] = (s, i - 1, d);
                    top += 1;
                }
                s = if ti > 16 { i + 1 } else { t };
            } else {
                if ti > 16 {
                    debug_assert!(top < stack.len(), "introsort stack overflow");
                    stack[top] = (i + 1, t, d);
                    top += 1;
                }
                t = if is > 16 { i - 1 } else { s };
            }
        } else if top > 0 {
            top -= 1;
            let (l, r, dep) = stack[top];
            s = l;
            t = r;
            d = dep;
        } else {
            ks_insertsort_by(a, &lt);
            return;
        }
    }
}

/// Below this length the indirection in [`ks_introsort_by_key`] costs more than the moves it saves,
/// so that function sorts the elements directly instead. 16 is also klib's own partition cutoff, so
/// an array this short never partitions at all: [`ks_introsort_by`] goes straight to its final
/// insertion-sort pass, which is already the cheapest thing available for it.
const INDIRECT_SORT_MIN: usize = 16;

/// [`ks_introsort_by`] with the element moves done once, on a permutation, instead of throughout the
/// sort. Produces **exactly** the array `ks_introsort_by` would.
///
/// # Why this exists
///
/// `mem_sort_dedup_patch` sorts `MemAlnReg`, which is 96 bytes, and it is the single most-called
/// sort in the run: 970 814 calls on 200k real GIAB pairs, mean length 105, and mate rescue drives
/// 570k of those with 65 elements or more (`BWA4_DEDUP_SHAPE`). Every quicksort swap therefore moves
/// 192 bytes. Sorting `(key, index)` pairs instead moves 12 to 32 bytes per swap, and the elements
/// move exactly once, at the end.
///
/// # Why the permutation is identical, which is the only thing that matters
///
/// `ks_introsort_by`'s control flow depends on nothing but the sequence of `lt` outcomes. Here the
/// key travels WITH its index in the same slot, so at every step of the algorithm the pair at
/// position `x` carries the key of whatever element the direct sort would have at position `x`.
/// Every comparison therefore has the same operands and the same outcome, every swap moves the same
/// two logical elements, and the final index order is precisely the permutation the direct sort
/// applies. That equality is asserted on random inputs, including heavily tied ones, by
/// `indirect_sort_matches_direct` below.
///
/// This matters because the permutation is OUTPUT-OBSERVABLE: `mem_sort_dedup_patch` sorts by `re`
/// alone and then drops, among regions that tie, the one the sort happened to put first. A sort that
/// merely produces *a* correct order (Rust's `sort_unstable_by`, the C++ fork's pdqsort) picks a
/// different survivor and a different SAM byte.
///
/// # Parameters
///
/// - `a`: the vector, sorted in place. It is REPLACED rather than permuted in place, which is what
///   lets the final gather be a straight forward-order fill.
/// - `perm`: scratch for the `(key, index)` pairs. Contents are discarded on entry; the caller keeps
///   the ALLOCATION between calls, which is the whole reason it is a parameter.
/// - `spare`: scratch for the gather. On return it holds `a`'s previous buffer, so a caller that
///   passes the same `spare` every time never allocates for either vector after the first call.
/// - `key`: extracts the sort key. Must depend only on the element, never on its position.
/// - `lt`: strict less-than on keys, with the same strict-weak-ordering requirement
///   [`ks_introsort_by`] documents (its Hoare scan uses the pivot as a sentinel).
///
/// # Why the scratch is not just allocated here
///
/// It was, first, and that version measured NEUTRAL end to end even though it removed 7.7% of
/// `mem_sort_dedup_patch`'s samples. Two `Vec`s per sort at roughly 10 million sorts per million real
/// pairs is on the order of 20 million allocations, which is the same size as the saving. The
/// buffers are passed in so that number is one per thread instead.
pub fn ks_introsort_by_key<T: Clone, K: Copy>(
    a: &mut Vec<T>,
    perm: &mut Vec<(K, u32)>,
    spare: &mut Vec<T>,
    key: impl Fn(&T) -> K,
    lt: impl Fn(&K, &K) -> bool,
) {
    let n = a.len();
    if n <= INDIRECT_SORT_MIN {
        ks_introsort_by(a.as_mut_slice(), |x, y| lt(&key(x), &key(y)));
        return;
    }
    // The permutation being sorted: each slot carries an element's key and where that element
    // currently lives in `a`. `u32` because a read's region list is orders of magnitude below 4
    // billion, and halving the index width halves what each swap moves.
    perm.clear();
    perm.extend(a.iter().enumerate().map(|(i, e)| (key(e), i as u32)));
    ks_introsort_by(perm, |x, y| lt(&x.0, &y.0));
    // The single gather. Reading `a` out of order is cheap here: the whole array is a few kilobytes
    // and has just been walked twice, so it is resident.
    spare.clear();
    spare.extend(perm.iter().map(|&(_, i)| a[i as usize].clone()));
    // `a` takes the gathered order and `spare` walks away with `a`'s old buffer, ready to be the
    // gather target again next call.
    std::mem::swap(a, spare);
}

/// Filter chains for a single read: drop light chains, then prune overlapping ones. Port of
/// `mem_chain_flt` (`bwamem.cpp:506`), restricted to one `seqid` group.
///
/// The C splits its input into runs of equal `seqid` and runs the body per run (`bwamem.cpp:527-549`
/// builds `range`); we are called per read, so there is exactly one run and the splitting is gone.
/// Everything inside the run is reproduced exactly.
///
/// Three stages:
/// 1. **Weight gate.** Compute `w` for each chain, drop anything below `opt.min_chain_weight`
///    (default 0, so by default nothing is dropped here). `first`/`kept` are reset at the same time.
/// 2. **Overlap prune.** Process chains from heaviest to lightest, keeping a list of accepted ones.
///    A new chain is compared against every accepted chain; if their query spans overlap
///    "significantly" and the new chain is much lighter, it is discarded as a shadow of the heavier
///    one. Otherwise it is accepted, marked 2 if it had any large overlap and 3 if it was clean.
/// 3. **Extension cap.** `opt.max_chain_extend` limits how many `kept == 1|2` chains get extended.
///    The default is `1 << 30`, so this is effectively off.
///
/// Relevant knobs (`MemOpt`, all with bwa's defaults):
/// - `mask_level` 0.50 (`-R`): overlap must cover at least this fraction of the shorter chain's
///   query span to count as significant.
/// - `max_chain_gap` 10000: an overlap is only significant if the shorter span is below this.
/// - `drop_ratio` 0.50 (`-D`): a chain is shadowed if its weight is below this fraction of the
///   heavier one's.
/// - `min_seed_len` 19: the `jw - iw >= min_seed_len << 1` guard (38) means a chain is never dropped
///   over a small absolute weight difference, however bad the ratio looks.
///
/// # Parameters
/// - `opt`: supplies `min_chain_weight`, `mask_level`, `max_chain_gap`, `drop_ratio`,
///   `min_seed_len` and `max_chain_extend` (see the list above for defaults and meaning).
/// - `chains`: ONE read's chains, taken by value because they are reordered and filtered in place.
///   Must all share a `seqid` (the C's per-`seqid` run splitting is absent here for that reason), and
///   must arrive in the kbtree in-order order [`build_chains_from_resolved`] produces: that order is
///   the tie-break input for the unstable sort below.
///
/// # Returns
/// The surviving chains, weight-descending, each with `w` computed and `kept` in 1..=3. Chains with
/// `kept == 0` are gone. May be empty.
pub fn mem_chain_flt(opt: &MemOpt, chains: Vec<MemChain>) -> Vec<MemChain> {
    if chains.is_empty() {
        return chains;
    }
    // ---- Stage 1: weight gate ----------------------------------------------------------------
    // Chains that passed the weight gate, still in input (kbtree) order until the sort below. This
    // vector's indices are what `kept_idx` and the `first` field refer to from here on, so it must not
    // be reordered again after the sort.
    // `with_capacity`, not `new`: this runs once per read, and growing from empty reallocates about
    // log2(n) times per call for a vector whose element carries its own `Vec<MemSeed>`. The filter
    // below can only shrink the input, so the input length is an exact upper bound.
    let mut ranked: Vec<MemChain> = Vec::with_capacity(chains.len());
    for mut c in chains {
        c.first = -1;
        c.kept = 0;
        c.w = mem_chain_weight(&c);
        if c.w >= opt.min_chain_weight {
            ranked.push(c);
        }
    }
    if ranked.is_empty() {
        return ranked;
    }

    // ---- Stage 2: sort heaviest-first, then prune shadowed chains -----------------------------
    // Sort by weight descending with bwa-mem2's exact (unstable) `ks_introsort`, so equal-weight
    // overlapping chains are ordered identically to the oracle (`flt_lt(a, b) = a.w > b.w`).
    ks_introsort_by(&mut ranked, |x, y| x.w > y.w);

    // The heaviest chain is always kept unconditionally, and seeds the accepted list.
    ranked[0].kept = 3;
    // Indices into `ranked` of the chains accepted so far, ascending (so also weight-descending).
    // Invariant at the top of the `i` loop: `kept_idx` holds exactly the chains in `0..i` that were
    // not shadowed, each with `kept` 2 or 3, and every chain in `i..` still has `kept == 0`.
    // Per-thread scratch for the same reason: one allocation per read, and the accepted list is
    // usually a handful of entries, so the allocation dominated what it held.
    let mut kept_idx: Vec<usize> = KEPT_IDX.with(|c| std::mem::take(&mut *c.borrow_mut()));
    kept_idx.clear();
    kept_idx.push(0);
    // `broke` mirrors the C's `if (k == chains.n)` test after the inner loop (`bwamem.cpp:585`):
    // "the inner loop ran to completion" means "not shadowed", so the chain is accepted. Breaking
    // early means shadowed, so it is silently dropped (its `kept` stays 0 and `retain` removes it).
    for i in 1..ranked.len() {
        // `ib`/`ie` = this chain's half-open QUERY span in read offsets (no reference coordinate is
        // consulted anywhere in this loop), `iw` its weight in bases, `ialt` whether it is on an ALT
        // contig. Hoisted because `ranked[j]` is borrowed mutably inside the inner loop.
        let (ib, ie, iw, ialt) = (
            chn_beg(&ranked[i]),
            chn_end(&ranked[i]),
            ranked[i].w,
            ranked[i].is_alt,
        );
        // Set once any accepted chain overlaps this one "significantly"; decides `kept` 2 vs 3.
        let mut large_ovlp = false;
        // Set when the inner loop breaks early, i.e. this chain is shadowed and will be dropped.
        let mut broke = false;
        for &j in &kept_idx {
            // Same four quantities for the already-accepted (and therefore heavier or equal) chain
            // being compared against; `jb`/`je` are again QUERY offsets.
            let (jb, je, jw, jalt) = (
                chn_beg(&ranked[j]),
                chn_end(&ranked[j]),
                ranked[j].w,
                ranked[j].is_alt,
            );
            // Query-span intersection `[b_max, e_min)`; non-empty iff `e_min > b_max`.
            // Both are QUERY offsets; `e_min - b_max` is the overlap length in read bases.
            let b_max = jb.max(ib);
            let e_min = je.min(ie);
            // `(!jalt || ialt)`: an ALT-contig chain is never allowed to shadow a primary one. It
            // may shadow another ALT chain, and a primary may shadow anything.
            if e_min > b_max && (!jalt || ialt) {
                // QUERY span lengths of the two chains, in read bases, and the shorter of them.
                // `mask_level` is measured against `min_l`, so a short chain buried inside a long one
                // counts as a large overlap even though it covers little of the long one.
                let li = ie - ib;
                let lj = je - jb;
                let min_l = li.min(lj);
                if (e_min - b_max) as f32 >= min_l as f32 * opt.mask_level
                    && min_l < opt.max_chain_gap
                {
                    large_ovlp = true;
                    // Record the FIRST chain this accepted chain shadows, so mapq can later account
                    // for the fact that a competing alignment existed ("keep the first shadowed hit
                    // s.t. mapq can be more accurate", `bwamem.cpp:578`). Only the first: later ones
                    // are not tracked.
                    if ranked[j].first < 0 {
                        ranked[j].first = i as i32;
                    }
                    if (iw as f32) < jw as f32 * opt.drop_ratio && jw - iw >= opt.min_seed_len << 1
                    {
                        broke = true;
                        break;
                    }
                }
            }
        }
        if !broke {
            kept_idx.push(i);
            ranked[i].kept = if large_ovlp { 2 } else { 3 };
        }
    }
    // Resurrect each accepted chain's first shadow at `kept = 1`. Note this can raise a chain that
    // the loop above dropped (its `kept` was 0), which is the point: a shadowed chain still informs
    // mapq, so it must survive the `retain` even though it will not be extended.
    for &ci in &kept_idx {
        // Index into `ranked` of the first chain `ci` shadowed, or -1 if it shadowed none.
        let f = ranked[ci].first;
        if f >= 0 {
            ranked[f as usize].kept = 1;
        }
    }
    // ---- Stage 3: extension cap ---------------------------------------------------------------
    // max_chain_extend demotion (default is 1<<30, effectively never). Count `kept == 1|2` chains in
    // array order; once `max_chain_extend` of them have been seen, every remaining chain with
    // `kept < 3` is demoted to 0 and hence dropped. `kept == 3` chains are exempt from counting, but
    // NOT from the second loop's scan (they simply fail its `kept < 3` test). `bwamem.cpp:597-603`.
    // The two loops share `i` deliberately, so the second resumes where the first stopped.
    //
    // THE INCREMENT ORDER IS LOAD-BEARING. The C is
    //
    //     for (i = k = 0; i < n_chn; ++i) {
    //         if (a[i].kept == 0 || a[i].kept == 3) continue;
    //         if (++k >= opt->max_chain_extend) break;
    //     }
    //     for (; i < n_chn; ++i) if (a[i].kept < 3) a[i].kept = 0;
    //
    // `break` leaves the loop *without* running the header's `++i`, so `i` still points AT the
    // chain that hit the cap, and the second loop (which reuses `i`) demotes that chain too: it has
    // `kept` 1 or 2 by construction, so `kept < 3` holds and it is zeroed. An earlier version of
    // this port incremented `i` before testing the cap, which let the triggering chain survive, so
    // we kept one chain bwa drops. Only reachable with a small `-N` (`max_chain_extend` defaults to
    // `1 << 30`), which is why no parity run caught it: `-N` is not in `scripts/opt_parity.sh`.
    // `k` counts chains with `kept` 1 or 2 seen so far (the ones that will actually be extended);
    // `i` is the shared cursor into `ranked` that the second loop resumes from.
    let mut k = 0i32;
    let mut i = 0usize;
    while i < ranked.len() {
        if ranked[i].kept == 0 || ranked[i].kept == 3 {
            i += 1;
            continue;
        }
        k += 1;
        if k >= opt.max_chain_extend {
            break;
        }
        i += 1;
    }
    while i < ranked.len() {
        if ranked[i].kept < 3 {
            ranked[i].kept = 0;
        }
        i += 1;
    }
    ranked.retain(|c| c.kept != 0);
    // Hand the accepted-list buffer back for the next read on this thread.
    KEPT_IDX.with(|c| *c.borrow_mut() = kept_idx);
    ranked
}

thread_local! {
    /// Per-thread scratch for [`mem_chain_flt`]'s accepted-chain list. See its use: this function
    /// runs once per read, so a fresh `Vec` here is one allocation per read.
    static KEPT_IDX: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_index::BntSeq;
    use std::path::Path;

    /// Load the checked-in miniature reference (`testdata/tiny`) and its `.ann`/`.amb` metadata.
    ///
    /// # Returns
    /// The FM index and `BntSeq` for that reference. Panics if the index files are missing, which
    /// means the testdata was not generated.
    fn tiny() -> (FmIndex, BntSeq) {
        // Index basename, not a file that exists: the loaders append the `.bwt`/`.ann`/... suffixes.
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        (
            FmIndex::load(Path::new(prefix)).unwrap(),
            BntSeq::load(Path::new(prefix)).unwrap(),
        )
    }

    // A brute-force reference introsort matching klib's structure is unnecessary; we assert the
    // two invariants that must hold for any faithful port: the result is fully sorted by the
    // comparator, and it is a permutation of the input. Byte-identity to the oracle's *unstable*
    // permutation is covered by the end-to-end SAM gate.
    #[test]
    fn introsort_sorts_and_permutes_various_sizes() {
        // Deterministic LCG so the test is reproducible without extra deps.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i32 % 100
        };
        for &n in &[0usize, 1, 2, 3, 5, 16, 17, 33, 64, 200, 1000] {
            let orig: Vec<i32> = (0..n).map(|_| next()).collect();
            let mut a = orig.clone();
            ks_introsort_by(&mut a, |x, y| x > y); // descending, like flt_lt
            for w in a.windows(2) {
                assert!(w[0] >= w[1], "not sorted desc at n={n}: {a:?}");
            }
            let mut s1 = orig.clone();
            let mut s2 = a.clone();
            s1.sort_unstable();
            s2.sort_unstable();
            assert_eq!(s1, s2, "not a permutation at n={n}");
        }
    }

    /// The whole contract of [`ks_introsort_by_key`]: not "sorted", which any sort gives, but the
    /// SAME ARRAY [`ks_introsort_by`] produces, element for element. The elements carry a payload
    /// that the comparator never looks at, so a run of tied keys can only come out in the same order
    /// if the two sorts really applied the same permutation.
    ///
    /// The key space is deliberately tiny (`% 8` at the widest, `% 3` at the narrowest) so that ties
    /// are the common case rather than a corner one, and the sizes straddle both cutoffs that matter:
    /// `INDIRECT_SORT_MIN` (16, below which this delegates) and klib's own partition cutoff.
    #[test]
    fn indirect_sort_matches_direct() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for &n in &[0usize, 1, 2, 3, 15, 16, 17, 33, 64, 105, 200, 1000] {
            for &modulus in &[3u32, 8, 1000] {
                // `.1` is the payload: distinct, never compared, and therefore the only thing that
                // can reveal a difference in how ties were ordered.
                let orig: Vec<(u32, u32)> = (0..n).map(|i| (next() % modulus, i as u32)).collect();
                let mut direct = orig.clone();
                ks_introsort_by(&mut direct, |x, y| x.0 < y.0);
                let mut indirect = orig.clone();
                let (mut perm, mut spare) = (Vec::new(), Vec::new());
                ks_introsort_by_key(&mut indirect, &mut perm, &mut spare, |e| e.0, |x, y| x < y);
                assert_eq!(direct, indirect, "n={n}, keys mod {modulus}");
            }
        }
    }

    /// The second sort in `mem_sort_dedup_patch` is a three-field comparator (score descending, then
    /// `rb`, then `qb`), so the key is a tuple rather than a scalar. Same equality, same reason.
    #[test]
    fn indirect_sort_matches_direct_on_tuple_keys() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i32
        };
        // (score, rb, qb, payload), with the three key fields drawn from small ranges so that
        // partial ties (same score, different rb) and full ties both occur.
        let orig: Vec<(i32, i64, i32, u32)> = (0..300)
            .map(|i| (next() % 5, (next() % 7) as i64, next() % 3, i as u32))
            .collect();
        let lt = |x: &(i32, i64, i32), y: &(i32, i64, i32)| {
            x.0 > y.0 || (x.0 == y.0 && (x.1 < y.1 || (x.1 == y.1 && x.2 < y.2)))
        };
        let mut direct = orig.clone();
        ks_introsort_by(&mut direct, |x, y| lt(&(x.0, x.1, x.2), &(y.0, y.1, y.2)));
        let mut indirect = orig.clone();
        let (mut perm, mut spare) = (Vec::new(), Vec::new());
        ks_introsort_by_key(
            &mut indirect,
            &mut perm,
            &mut spare,
            |e| (e.0, e.1, e.2),
            lt,
        );
        assert_eq!(direct, indirect);
    }

    #[test]
    fn introsort_two_element_special_case() {
        let mut a = [1i32, 2];
        ks_introsort_by(&mut a, |x, y| x > y);
        assert_eq!(a, [2, 1]);
        let mut b = [5i32, 3];
        ks_introsort_by(&mut b, |x, y| x > y);
        assert_eq!(b, [5, 3]);
    }

    #[test]
    fn single_slice_makes_one_chain() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        // Forward-strand REFERENCE position (`< l_pac`) to lift the synthetic read from; also the
        // `rbeg` the resulting chain's first seed must land on.
        let start = 40_000i64;
        // A perfect 150-base copy of the reference at `start`, as 2-bit codes.
        let read: Vec<u8> = (0..150).map(|i| fm.base(start + i)).collect();
        let chains = build_chains(&fm, &bns, &opt, &read, 0);
        let chains = mem_chain_flt(&opt, chains);
        assert!(!chains.is_empty());
        // A chain whose first seed sits at the origin, covering the read.
        let c = chains
            .iter()
            .find(|c| c.seeds.iter().any(|s| s.rbeg == start))
            .expect("origin chain");
        assert_eq!(mem_chain_weight(c), 150);
        // Seeds within a chain are collinear and non-decreasing in qbeg.
        for w in c.seeds.windows(2) {
            assert!(w[1].qbeg >= w[0].qbeg);
        }
    }

    #[test]
    fn two_slices_make_two_chains() {
        let (fm, bns) = tiny();
        let opt = MemOpt::default();
        // Concatenate two distant reference slices -> two separate chains.
        let mut read: Vec<u8> = (0..80).map(|i| fm.base(10_000 + i)).collect();
        read.extend((0..80).map(|i| fm.base(120_000 + i)));
        let chains = build_chains(&fm, &bns, &opt, &read, 0);
        let chains = mem_chain_flt(&opt, chains);
        assert!(
            chains.len() >= 2,
            "expected >=2 chains, got {}",
            chains.len()
        );
    }
}

/// `BWA4_CHAIN_TIME=1` probe: how much of `build_chains_from_smems` is the inlined `get_sa_batch`
/// SA walk. The genome-scale sampler attributes ~37.6% of work samples to this function as a leaf,
/// but LTO inlines `get_sa_batch` into it, so sampling alone cannot separate the SA walk from the
/// merge. This can.
pub mod chain_time {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    /// Nanoseconds spent inside `get_sa_batch`, summed over every thread and every read. Only ever
    /// incremented when [`enabled`] is true, so it stays 0 in normal runs.
    pub static GET_SA_NS: AtomicU64 = AtomicU64::new(0);
    /// Number of suffix-array rows resolved, the denominator for the "ns per lookup" figure.
    pub static GET_SA_N: AtomicU64 = AtomicU64::new(0);
    /// Of those, how many were NOT walked because they repeated a run an identical SMEM had already
    /// resolved (see [`RepeatedRun`]). Reported so the per-lookup figure can be read against the
    /// rows actually walked rather than the rows asked for; without it, skipping work makes the
    /// per-lookup number look better for the wrong reason.
    pub static GET_SA_SKIPPED: AtomicU64 = AtomicU64::new(0);
    /// Nanoseconds spent in `build_chains_from_resolved` overall (the SA walk included), so
    /// `GET_SA_NS / TOTAL_NS` is the fraction attributable to the index rather than the merge.
    pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0);

    /// Whether `BWA4_CHAIN_TIME` is set in the environment. Read once and cached, so toggling the
    /// variable mid-process has no effect and the disabled path costs one atomic load.
    ///
    /// # Returns
    /// True if the probe should record timings.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_CHAIN_TIME").is_some())
    }

    /// Print the accumulated counters to stderr, once, at the end of a run. A no-op unless
    /// [`enabled`]. Not thread-synchronized beyond the relaxed loads: call it after the worker
    /// threads have joined.
    pub fn dump() {
        if !enabled() {
            return;
        }
        // `sa` and `tot` in seconds (converted from ns), `n` a plain lookup count.
        let (sa, tot, n) = (
            GET_SA_NS.load(Ordering::Relaxed) as f64 / 1e9,
            TOTAL_NS.load(Ordering::Relaxed) as f64 / 1e9,
            GET_SA_N.load(Ordering::Relaxed),
        );
        // Rows asked for, rows actually walked, and the per-row cost of the SECOND, because that is
        // the one the FM index pays. Reporting only the first made a run that skipped work look like
        // a run with faster lookups.
        let skipped = GET_SA_SKIPPED.load(std::sync::atomic::Ordering::Relaxed);
        let walked = n.saturating_sub(skipped);
        eprintln!(
            "[chain-time] build_chains_from_smems={:.3}s of which get_sa_batch={:.3}s ({:.0}%), \
             {} SA lookups of which {} skipped as exact repeats ({:.1}%), {:.0} ns per row walked",
            tot,
            sa,
            100.0 * sa / tot.max(1e-9),
            n,
            skipped,
            100.0 * skipped as f64 / (n.max(1) as f64),
            1e9 * sa / (walked.max(1) as f64),
        );
    }
}
